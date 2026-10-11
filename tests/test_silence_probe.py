"""Frozen semantic-silence probe checks."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from latent_timing_duplex.silence import (
    fit_softmax_probe,
    fixed_width_projection,
    multiclass_metrics,
    paired_lineage_bootstrap_delta,
)

ROOT = Path(__file__).resolve().parents[1]


def test_softmax_probe_learns_separable_data() -> None:
    x = np.asarray([[-2, 0], [-1, 0], [1, 0], [2, 0]], dtype=float)
    y = np.asarray(["wait", "wait", "respond", "respond"])
    probe = fit_softmax_probe(x, y, steps=300)
    probabilities = probe.predict_proba(x)
    metrics = multiclass_metrics(y, probabilities, probe.classes)
    assert metrics["accuracy"] == 1.0
    assert metrics["macro_auprc"] == 1.0


def test_macro_auprc_requires_every_predeclared_class() -> None:
    metrics = multiclass_metrics(
        np.asarray(["wait", "respond"]),
        np.asarray([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1]]),
        np.asarray(["wait", "respond", "repair"]),
    )
    assert metrics["macro_auprc"] is None


def test_fixed_width_projection_is_reproducible() -> None:
    x = np.arange(30, dtype=float).reshape(5, 6)
    a = fixed_width_projection(x, 3, seed=7)
    b = fixed_width_projection(x, 3, seed=7)
    assert a.shape == (5, 3)
    np.testing.assert_allclose(a, b)
    assert fixed_width_projection(x[:, :2], 3, seed=7).shape == (5, 3)


def test_paired_bootstrap_uses_lineages() -> None:
    labels = np.asarray(["wait", "respond"] * 4)
    classes = np.asarray(["wait", "respond"])
    better = np.asarray([[0.9, 0.1], [0.1, 0.9]] * 4)
    worse = np.full((8, 2), 0.5)
    groups = np.asarray(["a", "a", "b", "b", "c", "c", "d", "d"])
    result = paired_lineage_bootstrap_delta(
        labels, better, worse, classes, groups, n_boot=50, seed=3
    )
    assert result["delta"] is not None and result["delta"] > 0
    assert result["n_boot"] == 50


def test_probe_cli_writes_development_report(tmp_path) -> None:
    rng = np.random.default_rng(4)
    classes = np.asarray(["wait", "ack", "respond", "repair"])
    n = 64
    labels = np.tile(classes, n // len(classes))
    split = np.asarray(["train"] * 32 + ["development"] * 32)
    signal = np.eye(4)[np.asarray([np.flatnonzero(classes == y)[0] for y in labels])]
    path = tmp_path / "features.npz"
    np.savez(
        path,
        session_id=np.asarray([f"s{i // 4}" for i in range(n)]),
        lineage_id=np.asarray([f"g{i // 8}" for i in range(n)]),
        split=split,
        action=labels,
        context=np.asarray(["joint_silence"] * n),
        elapsed_s=np.tile([0.24, 0.48, 0.96, 2.0], n // 4),
        model_cutoff_s=np.arange(n, dtype=float) + 2.0,
        available_at_s=np.arange(n, dtype=float) + 2.0,
        acoustic=rng.normal(size=(n, 3)),
        hidden=np.concatenate([signal, rng.normal(scale=0.01, size=(n, 4))], axis=1),
        predicted_latent=signal,
    )
    output = tmp_path / "report.json"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "experiments/silence_graft/train_probe.py"),
            "--features",
            str(path),
            "--output",
            str(output),
            "--width",
            "8",
            "--steps",
            "100",
            "--n-boot",
            "20",
        ],
        cwd=ROOT,
        check=True,
    )
    report = json.loads(output.read_text())
    assert report["evaluation_split"] == "development"
    assert report["test_opened"] is False
    assert "hidden_predicted_latent" in report["results"]
