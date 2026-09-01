"""Turn-taking baselines (frozen VAP, CPU-ok)."""

from latent_timing_duplex.baselines.vap import (
    VAP_CODE,
    VAP_PAPER,
    VAPBaseline,
    p_shift_from_p_now,
    pool_to_chunk_grid,
    vap_probs_to_chunks,
)

__all__ = [
    "VAP_CODE",
    "VAP_PAPER",
    "VAPBaseline",
    "p_shift_from_p_now",
    "pool_to_chunk_grid",
    "vap_probs_to_chunks",
]
