"""Predictor-horizon indexing on the 80 ms (12.5 Hz) chunk grid.

Phase 1 ablates look-ahead of 80 ms, 1 s, and 5 s. Moshi / Mimi frames are
80 ms, so 1 s and 5 s are not integer multiples (12.5 and 62.5 frames).
Offsets use **half-up** rounding so the target is at least the requested
horizon:

    80 ms → 1 frame
    1 s   → 13 frames (1.04 s)
    5 s   → 63 frames (5.04 s)

Turn-event eval horizons stay in ``configs/eval.yaml`` / the Phase 0 harness.
"""

from __future__ import annotations

import math

import numpy as np

CHUNK_DURATION_S = 0.08
PHASE1_HORIZONS_S: tuple[float, ...] = (0.08, 1.00, 5.00)

# Locked table for the *plan* ablation grid (half-up of horizon / 0.08).
CANONICAL_HORIZON_STEPS: dict[float, int] = {
    0.08: 1,
    1.00: 13,
    5.00: 63,
}

# Spark-trained predictor offsets (floor of the same seconds). mid-180 at
# 12.5 Hz is exactly 2250 frames. Do not silently convert 1 s → 13 here:
# checkpoints under h12_* / h62_* use these integers.
SPARK_TRAINED_HORIZON_FRAMES: tuple[int, ...] = (1, 12, 62)
MID180_N_FRAMES: int = 2250  # 180.0 / 0.08
MOSHI_HIDDEN_DIM: int = 4096
PROTOCOL_SEED: int = 20260903
PRIMARY_LAMBDA: float = 0.01
REFERENCE_LAMBDA: float = 0.0


def horizon_steps(horizon_s: float, chunk_duration_s: float = CHUNK_DURATION_S) -> int:
    """Integer frame offset for a look-ahead horizon (half-up rounding)."""
    if horizon_s <= 0:
        raise ValueError(f"horizon_s must be positive, got {horizon_s}")
    if chunk_duration_s <= 0:
        raise ValueError(f"chunk_duration_s must be positive, got {chunk_duration_s}")
    ratio = horizon_s / chunk_duration_s
    steps = int(math.floor(ratio + 0.5))
    if steps < 1:
        raise ValueError(
            f"horizon {horizon_s}s is shorter than one chunk ({chunk_duration_s}s)"
        )
    return steps


def target_index(
    t: int,
    horizon_s: float,
    n_chunks: int | None = None,
    chunk_duration_s: float = CHUNK_DURATION_S,
) -> int:
    """Return the target chunk index for source index ``t``.

    Raises ``IndexError`` when the target would fall outside ``[0, n_chunks)``.
    """
    if t < 0:
        raise IndexError(f"source index must be >= 0, got {t}")
    tgt = t + horizon_steps(horizon_s, chunk_duration_s)
    if n_chunks is not None and tgt >= n_chunks:
        raise IndexError(
            f"target index {tgt} is past n_chunks={n_chunks} "
            f"(t={t}, horizon_s={horizon_s})"
        )
    return tgt


def pair_indices_frames(
    n_chunks: int,
    horizon_frames: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Valid ``(source, target)`` index vectors for an explicit frame offset.

    Use this for Spark checkpoints keyed by ``H∈{1,12,62}``. The half-up
    helper ``pair_indices`` (1 s → 13) is the plan table, not the trained grid.
    """
    if n_chunks < 0:
        raise ValueError("n_chunks must be non-negative")
    if horizon_frames < 1:
        raise ValueError(f"horizon_frames must be >= 1, got {horizon_frames}")
    n_pairs = n_chunks - int(horizon_frames)
    if n_pairs <= 0:
        empty = np.zeros(0, dtype=np.int64)
        return empty, empty
    src = np.arange(n_pairs, dtype=np.int64)
    tgt = src + int(horizon_frames)
    return src, tgt


def pair_indices(
    n_chunks: int,
    horizon_s: float,
    chunk_duration_s: float = CHUNK_DURATION_S,
) -> tuple[np.ndarray, np.ndarray]:
    """Valid ``(source, target)`` index vectors for a session of ``n_chunks``."""
    if n_chunks < 0:
        raise ValueError("n_chunks must be non-negative")
    offset = horizon_steps(horizon_s, chunk_duration_s)
    return pair_indices_frames(n_chunks, offset)
