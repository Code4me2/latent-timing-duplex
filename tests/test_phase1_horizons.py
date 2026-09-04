"""Predictor-horizon indexing: 80 ms / 1 s / 5 s on the 12.5 Hz grid."""

from __future__ import annotations

import pytest

from latent_timing_duplex.phase1.horizons import (
    CANONICAL_HORIZON_STEPS,
    CHUNK_DURATION_S,
    MID180_N_FRAMES,
    MOSHI_HIDDEN_DIM,
    PHASE1_HORIZONS_S,
    PRIMARY_LAMBDA,
    PROTOCOL_SEED,
    REFERENCE_LAMBDA,
    SPARK_TRAINED_HORIZON_FRAMES,
    horizon_steps,
    pair_indices,
    pair_indices_frames,
    target_index,
)


def test_canonical_ablation_grid() -> None:
    assert PHASE1_HORIZONS_S == (0.08, 1.00, 5.00)
    assert CHUNK_DURATION_S == 0.08
    assert CANONICAL_HORIZON_STEPS[0.08] == 1
    assert CANONICAL_HORIZON_STEPS[1.00] == 13
    assert CANONICAL_HORIZON_STEPS[5.00] == 63


def test_horizon_steps_half_up() -> None:
    assert horizon_steps(0.08) == 1
    assert horizon_steps(1.00) == 13  # 12.5 frames → 13 (covers ≥1 s)
    assert horizon_steps(5.00) == 63  # 62.5 frames → 63
    assert horizon_steps(0.16) == 2
    assert horizon_steps(0.80) == 10


def test_horizon_steps_rejects_non_positive() -> None:
    with pytest.raises(ValueError, match="positive"):
        horizon_steps(0.0)
    with pytest.raises(ValueError, match="positive"):
        horizon_steps(1.0, chunk_duration_s=0.0)


def test_target_index_and_bounds() -> None:
    assert target_index(0, 0.08) == 1
    assert target_index(4, 1.00) == 17
    assert target_index(0, 0.08, n_chunks=10) == 1
    with pytest.raises(IndexError, match="past n_chunks"):
        target_index(0, 1.00, n_chunks=10)
    with pytest.raises(IndexError, match="source"):
        target_index(-1, 0.08)


def test_pair_indices_alignment() -> None:
    src, tgt = pair_indices(20, 0.08)
    assert src.tolist() == list(range(19))
    assert tgt.tolist() == list(range(1, 20))
    src1, tgt1 = pair_indices(20, 1.00)
    assert src1.tolist() == list(range(7))
    assert (tgt1 - src1).tolist() == [13] * 7
    empty_s, empty_t = pair_indices(10, 1.00)
    assert empty_s.size == 0 and empty_t.size == 0


def test_spark_trained_grid_is_explicit_frames() -> None:
    assert SPARK_TRAINED_HORIZON_FRAMES == (1, 12, 62)
    assert MID180_N_FRAMES == 2250
    assert MOSHI_HIDDEN_DIM == 4096
    assert PROTOCOL_SEED == 20260903
    assert PRIMARY_LAMBDA == 0.01
    assert REFERENCE_LAMBDA == 0.0
    src, tgt = pair_indices_frames(20, 12)
    assert src.tolist() == list(range(8))
    assert (tgt - src).tolist() == [12] * 8
    # Plan half-up stays 13; do not silently rewrite Spark H=12.
    assert horizon_steps(1.00) == 13
    assert horizon_steps(1.00) != 12


def test_five_second_offset() -> None:
    src, tgt = pair_indices(80, 5.00)
    assert tgt[0] - src[0] == 63
    assert len(src) == 80 - 63
    assert src[-1] == 16
    assert tgt[-1] == 79
