"""Frozen user-channel NLL via Moshi ``LMModel.forward`` and a delay-NaN mask.

Moshi's training forward undelays per-codebook streams and fills the vacated
tail with ``NaN`` (``_undelay_sequence(..., fill_value=float('NaN'))``). A
naive mean over those logits is NaN. This module:

1. Reimplements the delay / undelay mask (numpy; no ``moshi`` import).
2. Reduces cross-entropy only where the mask is true and the logit is finite.
3. Calls a loaded ``LMModel.forward`` when the caller already has weights.

Spark already scored a 10-clip DuplexChat EN slice. Those numbers live in
``latent_timing_duplex.spark_slice`` and must not be re-measured here.

BayLing-Duplex uses a 168960-piece vocab; its token NLL is not comparable to
Moshi codebook NLL. This extractor still accepts any ``FrozenDuplexModel``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from latent_timing_duplex.models.base import FrozenDuplexModel
from latent_timing_duplex.spark_slice import MOSHI_ENV
from latent_timing_duplex.types import ChunkSignal, DualChannelSession, iter_chunks

# kyutai-labs/moshi default ``_lm_kwargs["delays"]`` (text + 16 audio).
MOSHI_DEFAULT_DELAYS: tuple[int, ...] = (
    0,
    0,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    0,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
)
MOSHI_AUDIO_OFFSET = 1
MOSHI_DEP_Q = 8
MOSHI_ZERO_TOKEN_ID = -1
MOSHI_FRAME_RATE_HZ = 12.5  # 80 ms; matches configs/default.yaml chunk.duration_s


@dataclass(frozen=True)
class NLLReduction:
    """Unweighted (mean of clip means) and duration-weighted NLL."""

    unweighted: float
    duration_weighted: float
    n_tokens: int
    duration_s: float
    n_clips: int


@dataclass(frozen=True)
class MoshiChannelNLL:
    """Per-step user-channel NLL plus clip-level reductions."""

    audio_per_step: list[float]
    text_per_step: list[float]
    audio: NLLReduction
    text: NLLReduction


def prepare_moshi_forward_env() -> None:
    """Match the Spark NLL job: no CUDA graphs, no torch.compile."""
    for item in MOSHI_ENV:
        key, _, value = item.partition("=")
        os.environ.setdefault(key, value)


def delay_nan_mask(
    delays: Sequence[int],
    n_timesteps: int,
    batch: int = 1,
) -> np.ndarray:
    """Boolean mask ``[B, K, T]``: False on the undelay tail Moshi fills with NaN.

    For delay ``d`` on codebook ``k``, the last ``d`` frames after undelay are
    invalid (``_undelay_sequence`` sets them to NaN and ``mask=0``).
    """
    if n_timesteps < 0:
        raise ValueError("n_timesteps must be non-negative")
    k = len(delays)
    mask = np.ones((batch, k, n_timesteps), dtype=bool)
    for i, delay in enumerate(delays):
        if delay < 0:
            raise ValueError(f"delay must be >= 0, got {delay}")
        if delay > 0 and n_timesteps:
            take = min(delay, n_timesteps)
            mask[:, i, -take:] = False
    return mask


def undelay_sequence(
    tensor: np.ndarray,
    delays: Sequence[int],
    fill_value: float = float("nan"),
) -> tuple[np.ndarray, np.ndarray]:
    """Numpy clone of Moshi ``_undelay_sequence``.

    ``tensor`` is ``[B, K, T, ...]``. Delayed streams are rolled left; the
    vacated tail is ``fill_value`` (NaN) and masked out.
    """
    if tensor.ndim < 3:
        raise ValueError(f"expected [B, K, T, ...], got shape {tensor.shape}")
    batch, n_codebooks, n_timesteps = tensor.shape[:3]
    if len(delays) != n_codebooks:
        raise ValueError(f"expected {n_codebooks} delays, got {len(delays)}")
    mask = delay_nan_mask(delays, n_timesteps, batch=batch)
    if all(d == 0 for d in delays):
        return np.array(tensor, copy=True), mask
    outs: list[np.ndarray] = []
    for k, delay in enumerate(delays):
        line = np.roll(tensor[:, k], -delay, axis=1)
        if delay > 0:
            line[:, -delay:] = fill_value
        outs.append(line)
    return np.stack(outs, axis=1), mask


def apply_zero_token_mask(
    mask: np.ndarray,
    codes: np.ndarray,
    zero_token_id: int = MOSHI_ZERO_TOKEN_ID,
) -> np.ndarray:
    """Drop positions Moshi marks as ``zero_token_id`` (-1)."""
    return mask & (codes != zero_token_id)


def log_softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    """Finite-aware log-softmax. All-NaN slices stay NaN."""
    x = np.asarray(logits, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        shifted = x - np.nanmax(x, axis=axis, keepdims=True)
        exp = np.exp(shifted)
        denom = np.nansum(exp, axis=axis, keepdims=True)
        return shifted - np.log(denom)


def masked_nll_from_logits(
    logits: np.ndarray,
    targets: np.ndarray,
    mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-position NLL ``[B, K, T]`` and a nan-safe valid mask.

    Invalid = delay tail, non-finite logits, or ``mask==False``. Those
    positions are NaN in the returned NLL so a naive ``mean`` is still NaN
    (the Spark failure mode) unless the caller uses ``valid``.
    """
    logits = np.asarray(logits, dtype=np.float64)
    targets = np.asarray(targets)
    if logits.ndim != 4:
        raise ValueError(f"logits must be [B, K, T, card], got {logits.shape}")
    if targets.shape != logits.shape[:3]:
        raise ValueError(
            f"targets shape {targets.shape} must match logits {logits.shape[:3]}"
        )
    if mask is None:
        mask = np.ones(targets.shape, dtype=bool)
    else:
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != targets.shape:
            raise ValueError(f"mask shape {mask.shape} must match {targets.shape}")

    finite = np.isfinite(logits).all(axis=-1)
    valid = mask & finite
    log_probs = log_softmax(logits, axis=-1)
    nll = np.full(targets.shape, np.nan, dtype=np.float64)
    if not valid.any():
        return nll, valid

    batch, n_codebooks, n_timesteps, card = logits.shape
    clipped = np.clip(targets, 0, card - 1)
    # Gather log p(target) without a Python loop over T.
    b_ix = np.arange(batch)[:, None, None]
    k_ix = np.arange(n_codebooks)[None, :, None]
    t_ix = np.arange(n_timesteps)[None, None, :]
    gathered = -log_probs[b_ix, k_ix, t_ix, clipped]
    nll = np.where(valid, gathered, np.nan)
    return nll, valid


def mean_nll(nll: np.ndarray, valid: np.ndarray) -> float:
    """Mean over valid positions. Empty → NaN (honest, not a fake 0)."""
    nll = np.asarray(nll, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    if not valid.any():
        return float("nan")
    return float(nll[valid].mean())


def reduce_clip_nlls(
    clip_means: Sequence[float],
    durations_s: Sequence[float],
    n_tokens: Sequence[int] | None = None,
) -> NLLReduction:
    """Unweighted = mean of per-clip means. Duration-weighted = by seconds.

    Token-level mean equals duration-weighted when frame rate is constant;
    Spark reported them separately because clips differ in length.
    """
    means = np.asarray(clip_means, dtype=np.float64)
    durs = np.asarray(durations_s, dtype=np.float64)
    if means.size == 0:
        return NLLReduction(
            unweighted=float("nan"),
            duration_weighted=float("nan"),
            n_tokens=0,
            duration_s=0.0,
            n_clips=0,
        )
    if means.shape != durs.shape:
        raise ValueError("clip_means and durations_s must have the same length")
    finite = np.isfinite(means) & np.isfinite(durs) & (durs > 0)
    if not finite.any():
        tokens = int(np.asarray(n_tokens).sum()) if n_tokens is not None else 0
        return NLLReduction(
            unweighted=float("nan"),
            duration_weighted=float("nan"),
            n_tokens=tokens,
            duration_s=float(np.nansum(durs)),
            n_clips=int(means.size),
        )
    unweighted = float(means[finite].mean())
    duration_weighted = float(np.sum(means[finite] * durs[finite]) / durs[finite].sum())
    tokens = int(np.asarray(n_tokens).sum()) if n_tokens is not None else 0
    return NLLReduction(
        unweighted=unweighted,
        duration_weighted=duration_weighted,
        n_tokens=tokens,
        duration_s=float(durs.sum()),
        n_clips=int(means.size),
    )


def nll_from_lm_output(
    audio_logits: np.ndarray,
    audio_targets: np.ndarray,
    audio_mask: np.ndarray,
    text_logits: np.ndarray | None = None,
    text_targets: np.ndarray | None = None,
    text_mask: np.ndarray | None = None,
    frame_rate_hz: float = MOSHI_FRAME_RATE_HZ,
) -> MoshiChannelNLL:
    """Nan-safe NLL from tensors shaped like Moshi ``LMOutput``.

    Audio logits are ``[B, dep_q, T, card]``. Text logits are ``[B, 1, T, text_card]``.
    Per-step values average valid codebooks at that frame (80 ms grid).
    """
    audio_nll, audio_valid = masked_nll_from_logits(audio_logits, audio_targets, audio_mask)
    audio_step, audio_step_ok = _per_step_mean(audio_nll, audio_valid)
    n_audio = int(audio_valid.sum())
    duration_s = float(audio_nll.shape[-1] / frame_rate_hz) if audio_nll.size else 0.0
    audio_red = reduce_clip_nlls(
        [mean_nll(audio_nll, audio_valid)],
        [duration_s],
        n_tokens=[n_audio],
    )

    if text_logits is None or text_targets is None:
        text_step: list[float] = []
        text_red = NLLReduction(
            unweighted=float("nan"),
            duration_weighted=float("nan"),
            n_tokens=0,
            duration_s=duration_s,
            n_clips=0,
        )
    else:
        text_nll, text_valid = masked_nll_from_logits(text_logits, text_targets, text_mask)
        text_step, _ = _per_step_mean(text_nll, text_valid)
        text_red = reduce_clip_nlls(
            [mean_nll(text_nll, text_valid)],
            [duration_s],
            n_tokens=[int(text_valid.sum())],
        )

    return MoshiChannelNLL(
        audio_per_step=audio_step,
        text_per_step=text_step,
        audio=audio_red,
        text=text_red,
    )


def _per_step_mean(nll: np.ndarray, valid: np.ndarray) -> tuple[list[float], np.ndarray]:
    """Mean over codebooks at each time step. Invalid steps stay NaN."""
    # nll, valid: [B, K, T]
    step_valid = valid.any(axis=(0, 1))
    with np.errstate(invalid="ignore"):
        numbered = np.where(valid, nll, np.nan)
        step = np.nanmean(numbered, axis=(0, 1))  # all-masked steps stay NaN
    step = np.where(step_valid, step, np.nan)
    return [float(x) for x in step], step_valid


def nll_from_moshi_lm(
    lm: Any,
    codes: Any,
    *,
    user_in_predicted_slots: bool = True,
) -> MoshiChannelNLL:
    """Teacher-forced NLL from a loaded Moshi ``LMModel``.

    ``codes`` is ``[B, K, T]`` (text + audio). LEFT=user is placed in the
    predicted ``dep_q`` slots when ``user_in_predicted_slots`` is true (the
    Spark 10-clip convention). The model is not downloaded here.

    Requires ``torch`` and a real ``LMModel``. Call ``prepare_moshi_forward_env``
    first (Spark used ``NO_CUDA_GRAPH=1 NO_TORCH_COMPILE=1``).
    """
    del user_in_predicted_slots  # documented convention; codes are already laid out
    prepare_moshi_forward_env()
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "torch is required to call Moshi LMModel.forward. "
            "See docs/SPARK.md (aarch64 cu130). The delay-NaN helpers in this "
            "module do not need torch."
        ) from exc

    if not hasattr(lm, "forward"):
        raise TypeError("lm must look like moshi.models.lm.LMModel (has forward)")

    tensor = codes if isinstance(codes, torch.Tensor) else torch.as_tensor(codes)
    if tensor.ndim != 3:
        raise ValueError(f"codes must be [B, K, T], got {tuple(tensor.shape)}")

    was_training = bool(getattr(lm, "training", False))
    lm.eval()
    try:
        with torch.inference_mode():
            out = lm.forward(tensor)
    finally:
        if was_training:
            lm.train()

    audio_offset = int(getattr(lm, "audio_offset", MOSHI_AUDIO_OFFSET))
    dep_q = int(getattr(lm, "dep_q", MOSHI_DEP_Q))
    zero_id = int(getattr(lm, "zero_token_id", MOSHI_ZERO_TOKEN_ID))

    audio_logits = _as_numpy(out.logits)
    audio_mask = _as_numpy(out.mask).astype(bool)
    audio_targets = _as_numpy(tensor[:, audio_offset : audio_offset + dep_q])
    audio_mask = apply_zero_token_mask(audio_mask, audio_targets, zero_id)
    # Delay undelay already put NaN in logits; keep only finite + mask.
    audio_mask = audio_mask & np.isfinite(audio_logits).all(axis=-1)

    text_logits = _as_numpy(out.text_logits)
    text_mask = _as_numpy(out.text_mask).astype(bool)
    text_targets = _as_numpy(tensor[:, :1])
    text_mask = apply_zero_token_mask(text_mask, text_targets, zero_id)
    text_mask = text_mask & np.isfinite(text_logits).all(axis=-1)

    return nll_from_lm_output(
        audio_logits,
        audio_targets,
        audio_mask,
        text_logits=text_logits,
        text_targets=text_targets,
        text_mask=text_mask,
    )


def _as_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        return value.detach().float().cpu().numpy()
    return np.asarray(value)


def perplexity(nll: float) -> float:
    """``exp(nll)``. BayLing Spark ppl used this on token NLL."""
    return float(np.exp(nll))


class FrozenNLLExtractor:
    """Per-chunk user-channel NLL from a frozen duplex model.

    Thin adapter: ``model.user_channel_nll(session)`` then ``iter_chunks``.
    The model must already be loaded from a local directory. This class does
    not download weights and does not re-run the Spark 10-clip job.
    """

    def __init__(self, model: FrozenDuplexModel, chunk_duration_s: float = 0.08) -> None:
        self.model = model
        self.chunk_duration_s = chunk_duration_s

    def extract(self, session: DualChannelSession) -> list[ChunkSignal]:
        """Call ``model.user_channel_nll`` and wrap the floats for the harness."""
        values = self.model.user_channel_nll(session)
        return self.wrap_values(session, values)

    def wrap_values(
        self, session: DualChannelSession, values: list[float]
    ) -> list[ChunkSignal]:
        """Map a per-chunk float list onto the 80 ms harness grid."""
        del session
        return iter_chunks(
            duration_s=len(values) * self.chunk_duration_s,
            chunk_duration_s=self.chunk_duration_s,
            values=values,
            name=f"nll:{getattr(self.model, 'model_id', 'model')}",
        )
