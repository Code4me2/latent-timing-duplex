"""Frozen Voice Activity Projection baseline (ErikEkstedt/VAP), CPU-ok.

Ekstedt & Skantze, Interspeech 2022, arXiv:2205.09812.
https://github.com/ErikEkstedt/VAP

VAP is 50 Hz stereo. The Phase 0 harness is an 80 ms (12.5 Hz) grid, so four
VAP frames are mean-pooled into one chunk. LEFT=user is speaker 0; VAP's
``p_now`` is P(speaker 0 in the near bins), so

    p(shift) = 1 - p_now

matches the Spark 10-clip duration-weighted figure 0.627834 = 1 - 0.372166.

Weights are not bundled. ``load(local_checkpoint=...)`` reads a file you
already have (``examples/VAP_*.pt`` or a Lightning ckpt). Inference defaults
to CPU. This module does not re-run the Spark job.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from latent_timing_duplex.exceptions import WeightsNotBundled
from latent_timing_duplex.spark_slice import (
    VAP_DURATION_WEIGHTED_P_FUTURE,
    VAP_DURATION_WEIGHTED_P_NOW,
    VAP_DURATION_WEIGHTED_P_SHIFT,
)
from latent_timing_duplex.types import ChunkSignal, DualChannelSession, iter_chunks

VAP_PAPER = "https://arxiv.org/abs/2205.09812"
VAP_CODE = "https://github.com/ErikEkstedt/VAP"
VAP_FRAME_HZ = 50
VAP_SAMPLE_RATE = 16000
DEFAULT_CHUNK_S = 0.08

# Documented Spark CPU aggregates (same 10-clip tar, LEFT=user).
VAP_SPARK_P_NOW = VAP_DURATION_WEIGHTED_P_NOW
VAP_SPARK_P_FUTURE = VAP_DURATION_WEIGHTED_P_FUTURE
VAP_SPARK_P_SHIFT = VAP_DURATION_WEIGHTED_P_SHIFT


def p_shift_from_p_now(p_now: np.ndarray, speaker: int = 0) -> np.ndarray:
    """Shift probability for the current speaker.

    VAP ``p_now`` is always P(channel 0). If the current speaker is 0
    (LEFT=user), that value is a hold probability and must be inverted.
    """
    p = np.asarray(p_now, dtype=np.float64)
    if speaker == 0:
        return 1.0 - p
    return p


def pool_to_chunk_grid(
    values: np.ndarray,
    src_hz: float = VAP_FRAME_HZ,
    chunk_duration_s: float = DEFAULT_CHUNK_S,
) -> np.ndarray:
    """Mean-pool a 50 Hz series onto the 80 ms harness grid (4 frames / chunk)."""
    x = np.asarray(values, dtype=np.float64).reshape(-1)
    frames = int(round(chunk_duration_s * src_hz))
    if frames <= 0:
        raise ValueError("chunk_duration_s * src_hz must be >= 1")
    n = (x.size // frames) * frames
    if n == 0:
        return np.zeros(0, dtype=np.float64)
    return x[:n].reshape(-1, frames).mean(axis=1)


def duration_weighted_mean(
    clip_means: np.ndarray | list[float],
    durations_s: np.ndarray | list[float],
) -> float:
    """Weight per-clip means by clip duration (Spark VAP reduction)."""
    m = np.asarray(clip_means, dtype=np.float64)
    d = np.asarray(durations_s, dtype=np.float64)
    if m.size == 0:
        return float("nan")
    ok = np.isfinite(m) & np.isfinite(d) & (d > 0)
    if not ok.any():
        return float("nan")
    return float(np.sum(m[ok] * d[ok]) / d[ok].sum())


def vap_probs_to_chunks(
    p_now: np.ndarray,
    p_future: np.ndarray | None = None,
    *,
    chunk_duration_s: float = DEFAULT_CHUNK_S,
    src_hz: float = VAP_FRAME_HZ,
    speaker: int = 0,
    name: str = "vap:p_shift",
) -> list[ChunkSignal]:
    """Pool VAP frame probs and emit ``p(shift)`` on the harness grid."""
    shift = p_shift_from_p_now(np.asarray(p_now, dtype=np.float64).reshape(-1), speaker)
    pooled = pool_to_chunk_grid(shift, src_hz=src_hz, chunk_duration_s=chunk_duration_s)
    del p_future
    if pooled.size == 0:
        raise ValueError("not enough VAP frames for one 80 ms chunk")
    return iter_chunks(
        duration_s=float(pooled.size * chunk_duration_s),
        chunk_duration_s=chunk_duration_s,
        values=pooled.tolist(),
        name=name,
    )


def stereo_from_session(
    session: DualChannelSession,
    sample_rate: int = VAP_SAMPLE_RATE,
) -> np.ndarray:
    """Stack LEFT=user, RIGHT=assistant as ``[2, T]`` float32.

    Resamples with linear interpolation if ``session.sample_rate`` differs.
    Missing assistant audio is zeros of the user length (user-only slice).
    """
    user = _as_mono(session.user_audio)
    if user is None:
        raise ValueError(
            f"session {session.session_id!r} has no user_audio (LEFT). "
            "VAP needs speaker-separated stereo."
        )
    other = _as_mono(session.assistant_audio)
    if other is None:
        other = np.zeros_like(user)
    n = min(user.size, other.size)
    stereo = np.stack([user[:n], other[:n]], axis=0).astype(np.float32)
    src_sr = session.sample_rate or sample_rate
    if src_sr != sample_rate and n > 1:
        stereo = _resample_linear(stereo, src_sr, sample_rate)
    return stereo


def _as_mono(audio: object | None) -> np.ndarray | None:
    if audio is None:
        return None
    arr = np.asarray(audio, dtype=np.float64)
    if arr.ndim == 2:
        arr = arr[0] if arr.shape[0] <= arr.shape[1] else arr[:, 0]
    if arr.ndim != 1:
        raise ValueError(f"expected mono waveform, got shape {arr.shape}")
    return arr


def _resample_linear(stereo: np.ndarray, src_hz: int, dst_hz: int) -> np.ndarray:
    """Linear resample ``[2, T]`` without torchaudio (missing from upstream reqs)."""
    _, n = stereo.shape
    n_out = max(1, int(round(n * dst_hz / src_hz)))
    src_t = np.linspace(0.0, 1.0, n, endpoint=False)
    dst_t = np.linspace(0.0, 1.0, n_out, endpoint=False)
    out = np.empty((2, n_out), dtype=np.float32)
    for ch in range(2):
        out[ch] = np.interp(dst_t, src_t, stereo[ch]).astype(np.float32)
    return out


class VAPBaseline:
    """Frozen ErikEkstedt/VAP wrapper. CPU by default. No download."""

    def __init__(self, chunk_duration_s: float = DEFAULT_CHUNK_S) -> None:
        self.model_id = "ErikEkstedt/VAP"
        self.chunk_duration_s = chunk_duration_s
        self.device = "cpu"
        self._local_checkpoint: str | None = None
        self._model: Any = None

    def load(
        self,
        local_checkpoint: str | None = None,
        device: str = "cpu",
        model: Any | None = None,
    ) -> None:
        """Attach a local state dict / Lightning ckpt, or an injected model (tests)."""
        if model is not None:
            self._model = model
            self._local_checkpoint = local_checkpoint or "<injected>"
            self.device = device
            return
        if local_checkpoint is None:
            raise WeightsNotBundled(
                "VAP checkpoints are not bundled. The upstream repo "
                f"({VAP_CODE}) ships an example state dict under examples/. "
                "Download or train it yourself, then pass local_checkpoint=. "
                "This package will not fetch or guess a path."
            )
        path = Path(local_checkpoint)
        if not path.is_file():
            raise FileNotFoundError(
                f"VAP local_checkpoint {local_checkpoint!r} is not a file. "
                "Pass a state_dict.pt or Lightning .ckpt you already have."
            )
        self._model = self._load_local(path, device=device)
        self._local_checkpoint = str(path)
        self.device = device

    def _load_local(self, path: Path, device: str) -> Any:
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError(
                "torch is required to load ErikEkstedt/VAP. CPU is enough; "
                "see docs/SPARK.md (torchaudio was missing from upstream reqs)."
            ) from exc
        try:
            from vap.modules.lightning_module import VAPModule
        except ImportError:
            VAPModule = None  # type: ignore[assignment]
        mapped = torch.device(device)
        if VAPModule is not None:
            try:
                model = VAPModule.load_model(str(path))
                return model.to(mapped).eval()
            except Exception:
                # Bare state_dict.pt — fall through to torch.load.
                pass
        blob = torch.load(path, map_location=mapped, weights_only=False)
        if hasattr(blob, "eval"):
            return blob.to(mapped).eval()
        if isinstance(blob, dict) and "state_dict" in blob and VAPModule is not None:
            try:
                return VAPModule.load_model(str(path)).to(mapped).eval()
            except Exception as exc:
                raise RuntimeError(
                    f"Could not rebuild VAP from Lightning checkpoint {path}. "
                    "Install ErikEkstedt/VAP and pass a compatible ckpt."
                ) from exc
        raise RuntimeError(
            f"Could not interpret {path} as a VAP model. Install "
            f"ErikEkstedt/VAP and use VAPModule.load_model, or pass model= "
            "into load() for tests."
        )

    def score_session(self, session: DualChannelSession) -> list[ChunkSignal]:
        """Per-chunk ``p(shift)`` on the 80 ms grid (LEFT=user)."""
        if self._model is None:
            raise WeightsNotBundled(
                f"VAP is not loaded. Call load(local_checkpoint=...) with a "
                f"local ErikEkstedt/VAP file before scoring {session.session_id!r}."
            )
        probs = self._infer_probs(session)
        p_now = np.asarray(probs["p_now"], dtype=np.float64).reshape(-1)
        p_future = np.asarray(probs["p_future"], dtype=np.float64).reshape(-1)
        return vap_probs_to_chunks(
            p_now,
            p_future,
            chunk_duration_s=self.chunk_duration_s,
            speaker=0,
        )

    def clip_aggregates(self, session: DualChannelSession) -> dict[str, float]:
        """Clip-level duration-style means (same keys as the Spark table)."""
        if self._model is None:
            raise WeightsNotBundled(
                "VAP is not loaded. Call load(local_checkpoint=...) first."
            )
        probs = self._infer_probs(session)
        p_now = np.asarray(probs["p_now"], dtype=np.float64).reshape(-1)
        p_future = np.asarray(probs["p_future"], dtype=np.float64).reshape(-1)
        p_shift = p_shift_from_p_now(p_now, speaker=0)
        return {
            "p_now": float(p_now.mean()) if p_now.size else float("nan"),
            "p_future": float(p_future.mean()) if p_future.size else float("nan"),
            "p_shift": float(p_shift.mean()) if p_shift.size else float("nan"),
            "duration_s": float(session.duration_s),
        }

    def _infer_probs(self, session: DualChannelSession) -> dict[str, np.ndarray]:
        model = self._model
        if hasattr(model, "probs"):
            waveform = self._waveform_tensor(session)
            out = model.probs(waveform)
            return {
                "p_now": _detach(out["p_now"]),
                "p_future": _detach(out["p_future"]),
            }
        if callable(getattr(model, "score_session", None)):
            # Injected test double.
            return model.score_session(session)
        raise TypeError(
            "loaded object has neither .probs(waveform) nor .score_session; "
            "not a VAP model"
        )

    def _waveform_tensor(self, session: DualChannelSession) -> Any:
        stereo = stereo_from_session(session)
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError(
                "torch is required to run VAP.probs. The pooling helpers in "
                "this module work on numpy without torch."
            ) from exc
        wav = torch.from_numpy(stereo).unsqueeze(0)  # [1, 2, T]
        return wav.to(self.device)
