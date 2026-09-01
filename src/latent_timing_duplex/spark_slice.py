"""Reference numbers from a finished Spark job. Do not re-run Spark from here.

These are the measured 10-clip DuplexChat English slice on one tar, LEFT=user.
They are documentation constants for Phase 0 write-ups and tests. This module
does not load audio, does not download CANDOR, and does not launch inference.

Hypothesis (non-binding): Moshi NaNs came from the delay pattern.
``LMModel.forward`` undelays with ``fill_value=NaN``; a naive mean over
codebooks is NaN unless those positions are masked.
"""

from __future__ import annotations

from typing import Final

# Same tar, 10 English DuplexChat clips, LEFT = user.
SLICE_ID: Final = "duplexchat-en-10clip"
SLICE_SOURCE: Final = "DuplexChat EN 10-clip (same tar, LEFT=user)"
SLICE_DURATION_S: Final = 1383.44

# Moshi frozen user-channel NLL via nan-safe LMModel.forward.
# Flags on Spark: NO_CUDA_GRAPH=1 NO_TORCH_COMPILE=1.
MOSHI_UNWEIGHTED_AUDIO_NLL: Final = 0.407020
MOSHI_UNWEIGHTED_TEXT_NLL: Final = 0.023778
MOSHI_DURATION_WEIGHTED_AUDIO_NLL: Final = 0.488034
MOSHI_DURATION_WEIGHTED_TEXT_NLL: Final = 0.025773
MOSHI_FORWARD: Final = "LMModel.forward"
MOSHI_NAN_SAFE: Final = True
MOSHI_ENV: Final = ("NO_CUDA_GRAPH=1", "NO_TORCH_COMPILE=1")

# BayLing-Duplex token NLL. Vocab 168960 — not comparable to Moshi codebook NLL.
BAYLING_UNWEIGHTED_NLL: Final = 9.759206
BAYLING_UNWEIGHTED_PPL: Final = 17313
BAYLING_DURATION_WEIGHTED_NLL: Final = 7.167710
BAYLING_DURATION_WEIGHTED_PPL: Final = 1297
BAYLING_N_TOKENS: Final = 17280
BAYLING_VOCAB_SIZE: Final = 168960

# Frozen ErikEkstedt/VAP on CPU. p(shift) = 1 - p_now when LEFT=user (speaker 0).
VAP_DEVICE: Final = "cpu"
VAP_DURATION_WEIGHTED_P_NOW: Final = 0.372166
VAP_DURATION_WEIGHTED_P_FUTURE: Final = 0.412835
VAP_DURATION_WEIGHTED_P_SHIFT: Final = 0.627834

DELAY_NAN_HYPOTHESIS: Final = (
    "Non-binding: Moshi NaNs were caused by the acoustic delay pattern. "
    "LMModel.forward undelays delayed codebooks with fill_value=NaN; "
    "reducing NLL without that mask yields NaN."
)


def reference_table() -> dict[str, object]:
    """Machine-readable copy of the Spark 10-clip reference numbers."""
    return {
        "slice_id": SLICE_ID,
        "source": SLICE_SOURCE,
        "duration_s": SLICE_DURATION_S,
        "do_not_rerun_spark": True,
        "hypothesis": DELAY_NAN_HYPOTHESIS,
        "moshi": {
            "forward": MOSHI_FORWARD,
            "nan_safe": MOSHI_NAN_SAFE,
            "env": list(MOSHI_ENV),
            "unweighted": {
                "audio_nll": MOSHI_UNWEIGHTED_AUDIO_NLL,
                "text_nll": MOSHI_UNWEIGHTED_TEXT_NLL,
            },
            "duration_weighted": {
                "audio_nll": MOSHI_DURATION_WEIGHTED_AUDIO_NLL,
                "text_nll": MOSHI_DURATION_WEIGHTED_TEXT_NLL,
            },
        },
        "bayling_duplex": {
            "unweighted": {
                "nll": BAYLING_UNWEIGHTED_NLL,
                "ppl": BAYLING_UNWEIGHTED_PPL,
            },
            "duration_weighted": {
                "nll": BAYLING_DURATION_WEIGHTED_NLL,
                "ppl": BAYLING_DURATION_WEIGHTED_PPL,
            },
            "n_tokens": BAYLING_N_TOKENS,
            "vocab_size": BAYLING_VOCAB_SIZE,
            "comparable_to_moshi_codebook_nll": False,
        },
        "vap": {
            "device": VAP_DEVICE,
            "duration_weighted": {
                "p_now": VAP_DURATION_WEIGHTED_P_NOW,
                "p_future": VAP_DURATION_WEIGHTED_P_FUTURE,
                "p_shift": VAP_DURATION_WEIGHTED_P_SHIFT,
            },
        },
    }
