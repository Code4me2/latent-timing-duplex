"""Numpy checkpoint round-trip and h*_lam* path resolution."""

from __future__ import annotations

import json

import numpy as np
import pytest

from latent_timing_duplex.exceptions import Phase1EvalInputMissing
from latent_timing_duplex.phase1.checkpoint import (
    coerce_horizon_frames,
    format_run_dirname,
    load_mlp_checkpoint,
    load_mlp_from_state_tree,
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


def test_coerce_horizon_frames_accepts_h_set_dict() -> None:
    assert coerce_horizon_frames([1, 12, 62]) == (1, 12, 62)
    assert coerce_horizon_frames({"H_set": [1, 12, 62]}) == (1, 12, 62)
    assert coerce_horizon_frames({"horizons": {"H_set": [1, 12, 62]}}) == (1, 12, 62)
    assert coerce_horizon_frames({"H": [1, 12, 62]}) == (1, 12, 62)
    assert coerce_horizon_frames(None) == (1, 12, 62)


def test_selection_lock_spark_h_set_shape(tmp_path) -> None:
    """Exact Spark shape that used to raise int('H_set')."""
    lock = {
        "seed": 20260903,
        "horizons": {"H_set": [1, 12, 62], "notes": "floor grid"},
        "primary_lambda": 0.01,
        "reference_lambda": 0.0,
        "window": "mid180",
    }
    path = tmp_path / "SELECTION_LOCKED.json"
    path.write_text(json.dumps(lock), encoding="utf-8")
    sel = load_selection_lock(path)
    assert sel.horizon_frames == (1, 12, 62)
    sel_dir = load_selection_lock(tmp_path)
    assert sel_dir.horizon_frames == (1, 12, 62)


def test_nested_mlp_state_dict_without_torch() -> None:
    rng = np.random.default_rng(6)
    # torch layout [out, in] under mlp_state_dict / net.*.weight
    w0 = rng.normal(size=(6, 8))
    w1 = rng.normal(size=(4, 6))
    tree = {
        "mlp_state_dict": {
            "net.0.weight": w0,
            "net.0.bias": np.zeros(6),
            "net.2.weight": w1,
            "net.2.bias": np.zeros(4),
        },
        "horizon_frames": 12,
        "lambda_reg": 0.01,
    }
    loaded = load_mlp_from_state_tree(tree)
    assert loaded.horizon_frames == 12
    assert loaded.lambda_reg == pytest.approx(0.01)
    hidden = rng.normal(size=(5, 8))
    out = loaded.head.forward(hidden)
    assert out.shape == (5, 4)
    # Reconstruct expected torch Linear: x @ W.T + b
    want = np.maximum(hidden @ w0.T, 0.0) @ w1.T
    assert np.allclose(out, want)
