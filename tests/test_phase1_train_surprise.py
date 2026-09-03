"""Head-only train stub and surprise → existing turn-event harness."""

from __future__ import annotations

import numpy as np
import pytest

from latent_timing_duplex.data.synthetic import generate_synthetic_session
from latent_timing_duplex.exceptions import Phase1NotImplemented, Phase2OutOfScope
from latent_timing_duplex.phase1.heads import MLPPredictor
from latent_timing_duplex.phase1.surprise import (
    score_surprise,
    surprise_from_sequences,
    surprise_to_chunks,
    surprise_values,
)
from latent_timing_duplex.phase1.train import TrainConfig, aligned_pairs, train_head_step, train_loop


def test_aligned_pairs_horizon() -> None:
    hidden = np.arange(20, dtype=np.float64).reshape(20, 1)
    target = np.arange(20, dtype=np.float64).reshape(20, 1) + 100
    src, tgt = aligned_pairs(hidden, target, horizon_s=0.08)
    assert src.shape == (19, 1)
    assert tgt[0, 0] == 101  # target index 1
    src1, tgt1 = aligned_pairs(np.zeros((20, 2)), np.zeros((20, 3)), horizon_s=1.00)
    assert src1.shape == (7, 2)
    assert tgt1.shape == (7, 3)


def test_train_step_reduces_mse() -> None:
    rng = np.random.default_rng(0)
    hidden = rng.normal(size=(64, 8))
    weight = rng.normal(size=(8, 4))
    target = hidden @ weight
    head = MLPPredictor(hidden_dim=8, embed_dim=4, width=8, n_layers=2, seed=1)
    cfg = TrainConfig(lambda_reg=0.0, lr=0.02, freeze_backbone=True, max_steps=1)
    first = train_head_step(head, hidden[:32], target[:32], cfg)
    last = first
    for _ in range(12):
        last = train_head_step(head, hidden[:32], target[:32], cfg)
    assert last.mse < first.mse


def test_train_loop_refuses_unfreeze_and_gpu() -> None:
    hidden = np.zeros((20, 8))
    target = np.zeros((20, 4))
    with pytest.raises(Phase2OutOfScope, match="Phase 2"):
        train_loop(hidden, target, TrainConfig(freeze_backbone=False, max_steps=1))
    with pytest.raises(Phase1NotImplemented, match="CPU stub"):
        train_loop(hidden, target, TrainConfig(device="cuda", max_steps=1))


def test_train_loop_cpu_smoke() -> None:
    rng = np.random.default_rng(2)
    hidden = rng.normal(size=(40, 8))
    target = hidden @ rng.normal(size=(8, 4))
    result = train_loop(
        hidden,
        target,
        TrainConfig(horizon_s=0.08, lambda_reg=0.1, max_steps=12, batch_size=16, lr=0.03),
    )
    assert len(result.steps) == 12
    assert result.steps[-1].mse < result.steps[0].mse
    assert result.head is not None
    pred = result.head.forward(hidden[:5])
    assert pred.shape == (5, 4)


def test_surprise_values_and_chunks() -> None:
    pred = np.array([[1.0, 0.0], [0.0, 1.0]])
    target = np.zeros((2, 2))
    mse = surprise_values(pred, target, kind="mse")
    assert mse.shape == (2,)
    assert mse[0] == pytest.approx(0.5)
    chunks = surprise_to_chunks(mse.tolist())
    assert chunks[0].name == "jepa:surprise"
    assert chunks[0].t_end == pytest.approx(0.08)
    assert chunks[1].value == pytest.approx(0.5)


def test_surprise_hooks_existing_harness() -> None:
    bundle = generate_synthetic_session(duration_s=12.0, seed=0)
    n = int(12.0 / 0.08)
    hidden = np.linspace(0.0, 1.0, n)[:, None].repeat(4, axis=1)
    target = hidden + 0.1
    head = MLPPredictor(hidden_dim=4, embed_dim=4, width=8, n_layers=1, seed=0)
    chunks = surprise_from_sequences(hidden, target, head.forward, horizon_s=0.08)
    rows = score_surprise(bundle.session, chunks, horizons_s=(0.50, 1.00))
    kinds = {r.event_kind for r in rows}
    assert kinds == {"turn_shift", "backchannel", "barge_in"}
    assert all(r.n_chunks > 0 for r in rows)
