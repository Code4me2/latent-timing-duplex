"""Frozen-model NLL extraction (Moshi LMModel.forward + delay-NaN mask)."""

from latent_timing_duplex.extract.nll import (
    MOSHI_DEFAULT_DELAYS,
    FrozenNLLExtractor,
    delay_nan_mask,
    masked_nll_from_logits,
    nll_from_lm_output,
    nll_from_moshi_lm,
    reduce_clip_nlls,
    undelay_sequence,
)

__all__ = [
    "MOSHI_DEFAULT_DELAYS",
    "FrozenNLLExtractor",
    "delay_nan_mask",
    "masked_nll_from_logits",
    "nll_from_lm_output",
    "nll_from_moshi_lm",
    "reduce_clip_nlls",
    "undelay_sequence",
]
