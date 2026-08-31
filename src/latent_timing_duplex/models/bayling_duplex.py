"""BayLing-Duplex inference wrapper stub.

BayLing-Duplex (Fang, Guo, Feng, 2026; arXiv:2606.14528) is a native
full-duplex speech dialogue model on a GLM-4-Voice / GLM-4-9B backbone.
Turn-taking is next-token prediction over a multi-channel interleaved
sequence — the implicit signal this project later compares to an explicit
latent predictor.

Public artifacts (do not download from this skeleton):

- Weights:  Hugging Face ``BayLing-Models/BayLing-Duplex``
            4 safetensors shards (``model-00001-of-00004.safetensors`` …
            ``model-00004-of-00004.safetensors``), about 19.09 GB on disk.
            The card's "516k params" figure is a Hugging Face display bug;
            the index reports ~9.54B BF16 parameters.
- Tokenizer: ``zai-org/glm-4-voice-tokenizer``
- Decoder:   ``zai-org/glm-4-voice-decoder``
- Code:      https://github.com/BayLing-Models/BayLing-Duplex

This module does not call ``huggingface_hub`` and does not invent a local
weights directory.
"""

from __future__ import annotations

from latent_timing_duplex.exceptions import Phase0NotImplemented, WeightsNotBundled
from latent_timing_duplex.types import DualChannelSession

BAYLING_PAPER = "https://arxiv.org/abs/2606.14528"
BAYLING_CODE = "https://github.com/BayLing-Models/BayLing-Duplex"
BAYLING_WEIGHTS_ID = "BayLing-Models/BayLing-Duplex"
BAYLING_TOKENIZER_ID = "zai-org/glm-4-voice-tokenizer"
BAYLING_DECODER_ID = "zai-org/glm-4-voice-decoder"
BAYLING_N_SHARDS = 4
BAYLING_APPROX_BYTES = 19_088_029_827
BAYLING_APPROX_BF16_PARAMS = 9_542_557_728


class BayLingDuplexWrapper:
    """Reserved wrapper around BayLing-Models/BayLing-Duplex."""

    model_id = BAYLING_WEIGHTS_ID

    def __init__(self) -> None:
        self._local_dir: str | None = None

    def documented_ids(self) -> dict[str, str | int]:
        return {
            "weights": BAYLING_WEIGHTS_ID,
            "tokenizer": BAYLING_TOKENIZER_ID,
            "decoder": BAYLING_DECODER_ID,
            "code": BAYLING_CODE,
            "n_safetensor_shards": BAYLING_N_SHARDS,
            "approx_bytes": BAYLING_APPROX_BYTES,
            "approx_bf16_params": BAYLING_APPROX_BF16_PARAMS,
            "hf_param_display_bug": "516k",
        }

    def load(
        self,
        local_dir: str | None = None,
        tokenizer_dir: str | None = None,
        decoder_dir: str | None = None,
    ) -> None:
        if local_dir is None:
            raise WeightsNotBundled(
                "BayLing-Duplex weights are not bundled. Public ids: "
                f"weights={BAYLING_WEIGHTS_ID}, tokenizer={BAYLING_TOKENIZER_ID}, "
                f"decoder={BAYLING_DECODER_ID}. Four safetensors shards, ~19.1 GB, "
                "~9.54B BF16 params (the HF card '516k params' is a display bug). "
                f"Use the official download steps in {BAYLING_CODE}, then pass "
                "local_dir= (and tokenizer/decoder dirs). This skeleton will not "
                "fetch or guess a path."
            )
        raise Phase0NotImplemented(
            f"BayLing-Duplex local load from {local_dir!r} "
            f"(tokenizer={tokenizer_dir!r}, decoder={decoder_dir!r}) is a "
            "Phase 0 fill-in. No inference is implemented in the skeleton."
        )

    def user_channel_nll(self, session: DualChannelSession) -> list[float]:
        raise Phase0NotImplemented(
            f"Frozen BayLing-Duplex user-channel NLL for session "
            f"{session.session_id!r} is reserved. Implement after a working "
            "local load()."
        )
