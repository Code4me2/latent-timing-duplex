"""Numpy checkpoint round-trip and h*_lam* path resolution."""

from __future__ import annotations

import json

import numpy as np
import pytest

from latent_timing_duplex.exceptions import Phase1EvalInputMissing
from latent_timing_duplex.phase1.checkpoint import (
    format_run_dirname,
    load_mlp_checkpoint,
    load_selection_lock,
    parse_run_dirname,
    resolve_checkpoint_path,
    save_mlp_checkpoint,
    surprise_from_checkpoint,
)
from latent_timing_duplex.phase1.heads import MLPPredictor
from latent_timing_duplex.phase1.horizons import pair_indices_frames


def test_run_dirname_roundtrip() -> None:
    assert format_run_dirname(12, 0.01) == "h12_lam0.01"
    assert format_run_dirname(1, 0.0) == "h1_lam0"
    assert parse_run_dirname("h62_lam0.01") == (62, 0.01)
    assert parse_run_dirname("h1_lam0") == (1, 0.0)
    assert parse_run_dirname("notes.txt") is None


def test_save_load_tiny_mlp(tmp_path) -> None:
    rng = np.random.default_rng(0)
    hidden = rng.normal(size=(20, 8))
    head = MLPPredictor(hidden_dim=8, embed_dim=4, width=6, n_layers=2, seed=3)
    path = tmp_path / "h12_lam0.01" / "checkpoint.npz"
    save_mlp_checkpoint(path, head, horizon_frames=12, lambda_reg=0.01)
    loaded = load_mlp_checkpoint(path)
    assert loaded.horizon_frames == 12
    assert loaded.lambda_reg == pytest.approx(0.01)
    got = loaded.head.forward(hidden)
    want = head.forward(hidden)
    assert got.shape == (20, 4)
    assert np.allclose(got, want)


def test_surprise_from_checkpoint_matches_manual() -> None:
    rng = np.random.default_rng(1)
    hidden = rng.normal(size=(30, 8))
    target = rng.normal(size=(30, 4))
    head = MLPPredictor(hidden_dim=8, embed_dim=4, width=6, n_layers=1, seed=4)
    from latent_timing_duplex.phase1.checkpoint import LoadedCheckpoint

    ckpt = LoadedCheckpoint(
        head=head,
        horizon_frames=12,
        lambda_reg=0.01,
        path=__file__,  # unused
        format="memory",
    )
    surprise = surprise_from_checkpoint(ckpt, hidden, target)
    src, tgt = pair_indices_frames(30, 12)
    pred = head.forward(hidden[src])
    mse = np.mean((pred - target[tgt]) ** 2, axis=1)
    assert surprise.shape == (18,)
    assert np.allclose(surprise, mse)


def test_resolve_checkpoint_and_selection(tmp_path) -> None:
    run = tmp_path / "h12_lam0.01"
    run.mkdir()
    head = MLPPredictor(hidden_dim=4, embed_dim=2, width=4, n_layers=1, seed=0)
    save_mlp_checkpoint(run / "checkpoint.npz", head, horizon_frames=12, lambda_reg=0.01)
    lock = {
        "seed": 20260903,
        "horizon_frames": [1, 12, 62],
        "primary_lambda": 0.01,
        "reference_lambda": 0.0,
        "window": "mid180",
    }
    (tmp_path / "SELECTION_LOCKED.json").write_text(json.dumps(lock), encoding="utf-8")
    sel = load_selection_lock(tmp_path)
    assert sel.primary_lambda == 0.01
    assert sel.horizon_frames == (1, 12, 62)
    assert sel.window_mode == "mid"
    found = resolve_checkpoint_path(tmp_path, 12, 0.01, selection=sel)
    assert found.name == "checkpoint.npz"
    with pytest.raises(Phase1EvalInputMissing):
        resolve_checkpoint_path(tmp_path, 62, 0.01, selection=sel)
