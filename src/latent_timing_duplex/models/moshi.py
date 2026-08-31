"""Moshi inference wrapper stub.

Moshi (Défossez et al., 2024; arXiv:2410.00037) is a speech-text foundation
model and full-duplex spoken dialogue system. Official code:
https://github.com/kyutai-labs/moshi

Published PyTorch BF16 checkpoints (do not download from this skeleton):

- kyutai/moshiko-pytorch-bf16  (Moshiko)
- kyutai/moshika-pytorch-bf16  (Moshika)

Weights are CC-BY 4.0 on Hugging Face. This module records those ids and
reserves ``load`` / ``user_channel_nll``. It does not call ``huggingface_hub``,
does not invent a cache path, and does not run Moshi.
"""

from __future__ import annotations

from latent_timing_duplex.exceptions import Phase0NotImplemented, WeightsNotBundled
from latent_timing_duplex.types import DualChannelSession

MOSHI_PAPER = "https://arxiv.org/abs/2410.00037"
MOSHI_CODE = "https://github.com/kyutai-labs/moshi"
MOSHI_WEIGHTS = {
    "moshiko": "kyutai/moshiko-pytorch-bf16",
    "moshika": "kyutai/moshika-pytorch-bf16",
}


class MoshiWrapper:
    """Reserved wrapper around kyutai-labs/moshi.

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

    def load(self, local_dir: str | None = None) -> None:
        if local_dir is None:
            raise WeightsNotBundled(
                "Moshi weights are not bundled. Documented public ids: "
                f"{MOSHI_WEIGHTS}. Download them yourself via the official "
                f"stack in {MOSHI_CODE}, then pass local_dir= to load(). "
                "This skeleton will not fetch or guess a path."
            )
        raise Phase0NotImplemented(
            f"Moshi local load from {local_dir!r} is a Phase 0 fill-in. "
            f"Wire kyutai-labs/moshi against {self.model_id} here. "
            "No inference is implemented in the skeleton."
        )

    def user_channel_nll(self, session: DualChannelSession) -> list[float]:
        raise Phase0NotImplemented(
            f"Frozen Moshi user-channel NLL for session {session.session_id!r} "
            "is reserved. Implement after a working local load()."
        )
