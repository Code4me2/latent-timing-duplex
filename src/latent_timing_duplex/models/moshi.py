"""Moshi inference wrapper: local load + frozen user-channel NLL.

Moshi (Défossez et al., 2024; arXiv:2410.00037). Official code:
https://github.com/kyutai-labs/moshi

Published PyTorch BF16 checkpoints (do not download from this package):

- kyutai/moshiko-pytorch-bf16  (Moshiko)
- kyutai/moshika-pytorch-bf16  (Moshika)

``load(local_dir=...)`` reads files you already placed on disk. It does not
call ``huggingface_hub``. On Spark, NLL used ``LMModel.forward`` with
``NO_CUDA_GRAPH=1 NO_TORCH_COMPILE=1`` and the delay-NaN mask in
``extract.nll``. LEFT=user goes in the predicted ``dep_q`` audio slots.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from latent_timing_duplex.exceptions import WeightsNotBundled
from latent_timing_duplex.types import DualChannelSession

MOSHI_PAPER = "https://arxiv.org/abs/2410.00037"
MOSHI_CODE = "https://github.com/kyutai-labs/moshi"
MOSHI_WEIGHTS = {
    "moshiko": "kyutai/moshiko-pytorch-bf16",
    "moshika": "kyutai/moshika-pytorch-bf16",
}

# Filenames used by kyutai-labs/moshi loaders (local only).
_MOSHI_WEIGHT_NAMES = (
    "model.safetensors",
    "model.q8.safetensors",
    "moshi.safetensors",
)
_MIMI_WEIGHT_NAMES = (
    "tokenizer-e351c8d8-checkpoint125.safetensors",
    "tokenizer.safetensors",
    "mimi.safetensors",
)
_TEXT_TOKENIZER_NAMES = (
    "tokenizer_spm_32k_3.model",
    "tokenizer.model",
)


class MoshiWrapper:
    """Local-only wrapper around kyutai-labs/moshi.

    Pass ``local_dir`` only when you have already cloned or downloaded a
    public checkpoint yourself. There is no default install path.
    """

    def __init__(self, variant: str = "moshiko") -> None:
        if variant not in MOSHI_WEIGHTS:
            known = ", ".join(sorted(MOSHI_WEIGHTS))
            raise ValueError(f"unknown Moshi variant {variant!r}; expected one of: {known}")
        self.variant = variant
        self.model_id = MOSHI_WEIGHTS[variant]
        self._local_dir: str | None = None
        self._lm: Any = None
        self._mimi: Any = None

    def load(self, local_dir: str | None = None, device: str = "cpu") -> None:
        if local_dir is None:
            raise WeightsNotBundled(
                "Moshi weights are not bundled. Documented public ids: "
                f"{MOSHI_WEIGHTS}. Download them yourself via the official "
                f"stack in {MOSHI_CODE}, then pass local_dir= to load(). "
                "This package will not fetch or guess a path. Spark install "
                "notes (aarch64 cu130, no flash-attn, sphn sdist) are in "
                "docs/SPARK.md."
            )
        root = Path(local_dir)
        if not root.is_dir():
            raise FileNotFoundError(
                f"Moshi local_dir {local_dir!r} is not a directory. "
                "Populate it from kyutai/moshiko-pytorch-bf16 (or moshika) "
                "yourself; this call does not download."
            )
        from latent_timing_duplex.extract.nll import prepare_moshi_forward_env

        prepare_moshi_forward_env()
        self._lm, self._mimi = _load_moshi_local(root, device=device)
        self._local_dir = str(root)

    def user_channel_nll(self, session: DualChannelSession) -> list[float]:
        """Per-step user-channel audio NLL (80 ms), nan-safe delay mask."""
        if self._lm is None:
            raise WeightsNotBundled(
                f"Moshi is not loaded. Call load(local_dir=...) before "
                f"user_channel_nll({session.session_id!r})."
            )
        codes = _codes_from_session(session, self._lm, self._mimi)
        from latent_timing_duplex.extract.nll import nll_from_moshi_lm

        result = nll_from_moshi_lm(self._lm, codes)
        return result.audio_per_step


def _first_existing(root: Path, names: tuple[str, ...]) -> Path | None:
    for name in names:
        path = root / name
        if path.is_file():
            return path
    # One more pass: any matching suffix in the directory.
    if root.is_dir():
        for path in sorted(root.iterdir()):
            if path.name in names:
                return path
    return None


def _load_moshi_local(root: Path, device: str) -> tuple[Any, Any]:
    """Build Moshi + Mimi from a local directory. Never calls the Hub."""
    try:
        from moshi.models.loaders import CheckpointInfo
    except ImportError as exc:
        raise RuntimeError(
            "The moshi package is not installed. On Spark use the aarch64 "
            "cu130 notes in docs/SPARK.md (no flash-attn, sphn sdist, "
            "tiktoken/torchaudio extra). This wrapper will not pip-install "
            "or download weights."
        ) from exc

    moshi_weights = _first_existing(root, _MOSHI_WEIGHT_NAMES)
    mimi_weights = _first_existing(root, _MIMI_WEIGHT_NAMES)
    tokenizer = _first_existing(root, _TEXT_TOKENIZER_NAMES)
    missing = [
        label
        for label, path in (
            ("moshi weights (model.safetensors)", moshi_weights),
            ("mimi weights (tokenizer-*.safetensors)", mimi_weights),
            ("text tokenizer (tokenizer_spm_*.model)", tokenizer),
        )
        if path is None
    ]
    if missing:
        raise FileNotFoundError(
            f"Moshi local_dir {root} is missing: {', '.join(missing)}. "
            "Copy the official files here; do not point this loader at HF."
        )

    lm_config = None
    config_path = root / "config.json"
    if config_path.is_file():
        import json

        raw = json.loads(config_path.read_text())
        # Drop loader-only keys the way CheckpointInfo.from_hf_repo does.
        lm_config = dict(raw)
        for key in (
            "moshi_name",
            "mimi_name",
            "mimi_config_name",
            "tokenizer_name",
            "lora_name",
            "model_type",
            "lm_gen_config",
            "tts_config",
            "stt_config",
            "model_id",
        ):
            lm_config.pop(key, None)

    info = CheckpointInfo(
        moshi_weights=moshi_weights,
        mimi_weights=mimi_weights,
        tokenizer=tokenizer,
        lm_config=lm_config,
    )
    try:
        import torch

        torch_device = torch.device(device)
    except ImportError as exc:
        raise RuntimeError(
            "torch is required to load Moshi. See docs/SPARK.md for the "
            "aarch64 cu130 wheel (no GB10 kernel fork)."
        ) from exc
    lm = info.get_moshi(device=torch_device)
    mimi = info.get_mimi(device=torch_device)
    lm.eval()
    mimi.eval()
    return lm, mimi


def _codes_from_session(session: DualChannelSession, lm: Any, mimi: Any) -> Any:
    """Build ``[1, K, T]`` codes. LEFT=user in predicted ``dep_q`` slots.

    ``session.user_audio`` may be a waveform (encoded with Mimi) or a
    precomputed ``[n_q, T]`` / ``[K, T]`` code matrix (tests / Spark dumps).
    """
    import torch

    audio_offset = int(getattr(lm, "audio_offset", 1))
    n_q = int(getattr(lm, "n_q", 16))
    dep_q = int(getattr(lm, "dep_q", 8))
    k_total = audio_offset + n_q
    pad_id = int(getattr(lm, "existing_text_padding_id", 3))
    device = next(lm.parameters()).device

    user = session.user_audio
    if user is None:
        raise ValueError(
            f"session {session.session_id!r} has no user_audio (LEFT). "
            "Encode the user channel or attach precomputed Mimi codes."
        )

    user_codes = _maybe_codes(user)
    if user_codes is None:
        user_codes = _encode_mimi(mimi, user, device)

    assistant_codes = None
    if session.assistant_audio is not None:
        assistant_codes = _maybe_codes(session.assistant_audio)
        if assistant_codes is None:
            assistant_codes = _encode_mimi(mimi, session.assistant_audio, device)

    if user_codes.ndim == 3:
        user_codes = user_codes[0]
    n_steps = int(user_codes.shape[-1])
    codes = torch.full((1, k_total, n_steps), pad_id, dtype=torch.long, device=device)
    # Text stream: padding (delay 0). Audio: user in predicted slots.
    pred = user_codes[:dep_q, :n_steps]
    codes[:, audio_offset : audio_offset + pred.shape[0], : pred.shape[1]] = pred
    other_q = n_q - dep_q
    if assistant_codes is not None and other_q > 0:
        if assistant_codes.ndim == 3:
            assistant_codes = assistant_codes[0]
        other = assistant_codes[:other_q, :n_steps]
        start = audio_offset + dep_q
        codes[:, start : start + other.shape[0], : other.shape[1]] = other
    return codes


def _maybe_codes(audio: object) -> Any | None:
    """Return a long tensor if ``audio`` is already codes, else None."""
    import torch
    import numpy as np

    if isinstance(audio, dict) and "codes" in audio:
        audio = audio["codes"]
    if isinstance(audio, torch.Tensor) and audio.ndim >= 2 and audio.dtype in (
        torch.int16,
        torch.int32,
        torch.int64,
        torch.long,
    ):
        return audio.long()
    arr = np.asarray(audio)
    if arr.dtype.kind in "iu" and arr.ndim >= 2:
        return torch.as_tensor(arr, dtype=torch.long)
    return None


def _encode_mimi(mimi: Any, waveform: object, device: Any) -> Any:
    import torch
    import numpy as np

    wav = waveform
    if not isinstance(wav, torch.Tensor):
        wav = torch.as_tensor(np.asarray(wav, dtype=np.float32))
    wav = wav.float()
    if wav.ndim == 1:
        wav = wav[None, None, :]
    elif wav.ndim == 2:
        wav = wav[:1].unsqueeze(0) if wav.shape[0] <= wav.shape[1] else wav[:, :1].T.unsqueeze(0)
    wav = wav.to(device)
    with torch.inference_mode():
        codes = mimi.encode(wav)
    if codes.ndim == 3:
        return codes[0]
    return codes
