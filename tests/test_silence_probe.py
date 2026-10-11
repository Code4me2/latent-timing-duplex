"""Frozen semantic-silence probe checks."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from experiments.silence_graft.create_analysis_lock import create_analysis_lock
from experiments.silence_graft.receipts import receipt_bytes, sign_receipt
from experiments.silence_graft.train_probe import _load, _run
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
    with pytest.raises(ValueError, match="optimizer"):
        fit_softmax_probe(x, y, learning_rate=float("nan"))


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
    assert result["valid_fraction"] == 1.0


def test_paired_bootstrap_withholds_ci_when_too_few_draws_contain_all_classes() -> None:
    classes = np.asarray(["wait", "ack", "respond", "repair"])
    labels = classes.copy()
    probabilities = np.eye(4)
    result = paired_lineage_bootstrap_delta(
        labels, probabilities, np.full((4, 4), 0.25), classes,
        np.asarray(["a", "b", "c", "d"]), n_boot=200, seed=9,
    )
    assert result["n_boot_requested"] == 200
    assert result["n_boot_valid"] < 180
    assert result["valid_fraction"] < 0.9
    assert result["ci_low"] is None and result["ci_high"] is None


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
        item_id=np.asarray([f"item-{i}" for i in range(n)]),
        session_id=np.asarray([f"s{i // 4}" for i in range(n)]),
        source=np.asarray(["synthetic"] * n),
        lineage_id=np.asarray([f"g{i // 8}" for i in range(n)]),
        split=split,
        action=labels,
        context=np.asarray(["joint_silence"] * n),
        elapsed_s=np.tile([0.24, 0.48, 0.96, 2.0], n // 4),
        model_cutoff_s=np.arange(n, dtype=float) + 2.0,
        available_at_s=np.arange(n, dtype=float) + 2.0,
        acoustic=rng.normal(size=(n, 3)),
        energy_change=rng.normal(size=(n, 2)),
        history_truncated=np.zeros((n, 2)),
        sampling_weight_status=np.asarray(["not_identifiable"] * n),
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
            "--primary-metric",
            "nll",
        ],
        cwd=ROOT,
        check=True,
    )
    report = json.loads(output.read_text())
    assert report["evaluation_split"] == "development"
    assert report["test_opened"] is False
    assert report["primary_metric"] == "nll" and report["paired_primary_metric"]
    assert "hidden_predicted_latent" in report["results"]

    blocked = subprocess.run(
        [
            sys.executable,
            str(ROOT / "experiments/silence_graft/train_probe.py"),
            "--features", str(path),
            "--output", str(tmp_path / "test-report.json"),
            "--evaluate-test",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert blocked.returncode != 0
    assert "analysis-lock receipt" in blocked.stderr


def test_probe_archive_rejects_nonfinite_times_and_split_leakage(tmp_path) -> None:
    base = {
        "item_id": np.asarray(["i1", "i2"]),
        "session_id": np.asarray(["s1", "s2"]),
        "source": np.asarray(["synthetic", "synthetic"]),
        "lineage_id": np.asarray(["g1", "g2"]),
        "split": np.asarray(["train", "development"]),
        "action": np.asarray(["wait", "respond"]),
        "context": np.asarray(["joint_silence", "joint_silence"]),
        "elapsed_s": np.asarray([0.48, 0.48]),
        "model_cutoff_s": np.asarray([1.0, 1.0]),
        "available_at_s": np.asarray([1.0, 1.0]),
        "acoustic": np.ones((2, 1)),
        "energy_change": np.ones((2, 1)),
        "history_truncated": np.zeros((2, 1)),
        "sampling_weight_status": np.asarray(["not_identifiable", "not_identifiable"]),
        "hidden": np.ones((2, 1)),
    }
    path = tmp_path / "bad.npz"
    np.savez(path, **{**base, "elapsed_s": np.asarray([0.48, np.nan])})
    with pytest.raises(ValueError, match="finite row vector"):
        _load(path)

    np.savez(path, **{**base, "lineage_id": np.asarray(["g1", "g1"])})
    with pytest.raises(ValueError, match="lineage split bindings"):
        _load(path)

    np.savez(path, **{**base, "session_id": np.asarray(["s1", "s1"])})
    with pytest.raises(ValueError, match="session split bindings"):
        _load(path)

    np.savez(path, **{
        **base,
        "lineage_id": np.asarray(["g1", "g1"]),
        "split": np.asarray(["train", "train"]),
        "source": np.asarray(["conversation", "podcast"]),
    })
    with pytest.raises(ValueError, match="lineage split bindings"):
        _load(path)

    np.savez(path, **{**base, "source": np.asarray([["synthetic"], ["synthetic"]])})
    with pytest.raises(ValueError, match="one-dimensional string vector"):
        _load(path)


def _locked_archive(path: Path, split: np.ndarray, prefix: str) -> None:
    classes = np.asarray(["wait", "ack", "respond", "repair"])
    n = len(split)
    labels = np.resize(classes, n)
    signal = np.eye(4)[np.arange(n) % 4]
    np.savez(
        path,
        item_id=np.asarray([f"{prefix}-item-{i}" for i in range(n)]),
        session_id=np.asarray([f"{prefix}-session-{i}" for i in range(n)]),
        source=np.asarray(["synthetic"] * n),
        lineage_id=np.asarray([f"{prefix}-lineage-{i // 4}" for i in range(n)]),
        split=split,
        action=labels,
        context=np.asarray(["joint_silence"] * n),
        elapsed_s=np.resize(np.asarray([0.24, 0.48, 0.96, 2.0]), n),
        model_cutoff_s=np.arange(n, dtype=float) + 1.0,
        available_at_s=np.arange(n, dtype=float) + 1.0,
        acoustic=signal,
        energy_change=signal[:, :2],
        history_truncated=np.zeros((n, 2)),
        sampling_weight_status=np.asarray(["not_identifiable"] * n),
        hidden=signal,
        predicted_latent=np.arange(n * 2, dtype=float).reshape(n, 2),
    )


def test_analysis_lock_fails_before_test_open_and_permits_committed_archive(
    tmp_path: Path, monkeypatch
) -> None:
    package = tmp_path / "feature-package"
    package.mkdir()
    features = package / "features.npz"
    _locked_archive(
        features,
        np.asarray(["train"] * 32 + ["development"] * 32),
        "ordinary",
    )
    key = tmp_path / "hmac.key"
    key.write_bytes(b"locked-test-key-material-is-at-least-32-bytes")
    key.chmod(0o600)
    provenance = {
        "model_id": "synthetic-model", "model_revision": "test", "layer": "layer-1",
        "checkpoint_sha256": "1" * 64, "extractor_code_sha256": "2" * 64,
        "context_start": "authenticated_render_record_start_frame",
        "state_reset": "per_prefix_clip",
        "history_truncation": "renderer_prefix_or_session_start",
    }
    acoustic_recipe = {"input": "authenticated_prefix_clip_only", "version": "test"}
    feature_receipt = package / "feature_receipt.json"
    feature_receipt.write_bytes(receipt_bytes(sign_receipt({
        "schema": "semantic-silence-r2-feature-receipt/v1",
        "protocol_version": "semantic-silence-pilot/v1",
        "frozen_feature_provenance": provenance,
        "acoustic_recipe": acoustic_recipe,
        "sha256": {
            "features.npz": hashlib.sha256(features.read_bytes()).hexdigest(),
            "dataset_receipt": "d" * 64,
            "adjudication_receipt": "3" * 64,
            "assignment_receipt": "4" * 64,
            "private_assignment_source_map": "5" * 64,
            "renderer_receipt": "6" * 64,
            "external_feature_npz": "7" * 64,
            "external_feature_receipt": "8" * 64,
            "config": "9" * 64,
        },
        "outputs": {"features.npz": hashlib.sha256(features.read_bytes()).hexdigest()},
    }, key.read_bytes())))
    (package / "READY").write_text("ready\n", encoding="utf-8")
    test_features = tmp_path / "sealed-test.npz"
    _locked_archive(test_features, np.asarray(["test"] * 32), "test")
    test_dataset_receipt_hash = "e" * 64
    test_feature_receipt = tmp_path / "sealed-test.receipt.json"

    def write_test_receipt() -> bytes:
        payload = receipt_bytes(sign_receipt({
            "schema": "semantic-silence-r2-feature-receipt/v1",
            "protocol_version": "semantic-silence-pilot/v1",
            "frozen_feature_provenance": provenance,
            "acoustic_recipe": acoustic_recipe,
            "sha256": {
                "features.npz": hashlib.sha256(test_features.read_bytes()).hexdigest(),
                "dataset_receipt": test_dataset_receipt_hash,
                "adjudication_receipt": "3" * 64,
                "assignment_receipt": "4" * 64,
                "private_assignment_source_map": "5" * 64,
                "renderer_receipt": "6" * 64,
                "external_feature_npz": "7" * 64,
                "external_feature_receipt": "8" * 64,
                "config": "9" * 64,
            },
            "outputs": {"features.npz": hashlib.sha256(test_features.read_bytes()).hexdigest()},
        }, key.read_bytes()))
        test_feature_receipt.write_bytes(payload)
        return payload

    original_test_receipt = write_test_receipt()
    analysis_config = tmp_path / "analysis.json"
    analysis_config.write_text(json.dumps({
        "schema": "semantic-silence-r2-analysis-config/v1",
        "primary_metric": "macro_auprc",
        "bootstrap": {"unit": "lineage_id", "replicates": 5, "minimum_valid_fraction": 0.9},
        "required_elapsed_strata_s": {"short": [0.0, 1.0], "long": [3.0, None]},
        "replay_metrics_status": "required_but_unavailable_until_replay",
        "observational_hazard": {
            "confirmatory_status": "development_only_exploratory",
            "bin_edges_s": [float(index) for index in range(9)],
            "events": ["user_resume", "recorded_partner_entry", "no_event"],
            "primary_metric": "mean_item_nll",
            "tie_rule": "user_resume",
        },
        "thresholds": {
            "macro_auprc_improvement_min": 0.05,
            "paired_ci_lower_bound_strictly_above": 0.0,
            "acceptable_onset_hit_rate_improvement_min": 0.05,
            "premature_interruption_degradation_max": 0.02,
        },
    }), encoding="utf-8")
    lock_path = tmp_path / "analysis-lock.json"
    create_analysis_lock(
        features, feature_receipt, hashlib.sha256(test_features.read_bytes()).hexdigest(),
        hashlib.sha256(original_test_receipt).hexdigest(), test_dataset_receipt_hash,
        analysis_config, key, lock_path, width=8, steps=20, n_boot=5,
    )
    args = argparse.Namespace(
        features=features,
        output=tmp_path / "unused.json",
        seed=20261009,
        width=8,
        steps=20,
        learning_rate=0.1,
        n_boot=5,
        classes=["wait", "ack", "respond", "repair"],
        primary_metric="macro_auprc",
        evaluate_test=True,
        test_features=test_features,
        test_feature_receipt=test_feature_receipt,
        analysis_lock=lock_path,
        hmac_key=key,
    )

    original_load = np.load
    opened_test = False

    def tracking_load(source, *values, **kwargs):
        nonlocal opened_test
        if isinstance(source, io.BytesIO) or source == test_features:
            opened_test = True
        return original_load(source, *values, **kwargs)

    monkeypatch.setattr("experiments.silence_graft.train_probe.np.load", tracking_load)
    missing = argparse.Namespace(**{**vars(args), "analysis_lock": None})
    with pytest.raises(ValueError, match="analysis-lock receipt"):
        _run(missing)
    assert opened_test is False

    valid_lock = lock_path.read_bytes()
    assert "eval/metrics.py" in json.loads(valid_lock)["sha256"]
    changed = json.loads(valid_lock)
    changed["probe"]["width"] = 9
    lock_path.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ValueError, match="HMAC verification"):
        _run(args)
    assert opened_test is False

    lock_path.write_bytes(valid_lock)
    mismatched = argparse.Namespace(**{**vars(args), "width": 9})
    with pytest.raises(ValueError, match="exact CLI hyperparameters"):
        _run(mismatched)
    assert opened_test is False

    original_test = test_features.read_bytes()
    with np.load(test_features, allow_pickle=False) as archive:
        overlapping = {name: archive[name] for name in archive.files}
    overlapping["lineage_id"] = overlapping["lineage_id"].astype("<U64")
    overlapping["lineage_id"][0] = "ordinary-lineage-0"
    np.savez(test_features, **overlapping)
    changed_test_receipt = write_test_receipt()
    changed = json.loads(valid_lock)
    changed.pop("hmac_sha256")
    changed.pop("key_fingerprint")
    changed["sha256"]["sealed_test_commitment"] = hashlib.sha256(test_features.read_bytes()).hexdigest()
    changed["sha256"]["sealed_test_feature_receipt_commitment"] = hashlib.sha256(changed_test_receipt).hexdigest()
    lock_path.write_bytes(receipt_bytes(sign_receipt(changed, key.read_bytes())))
    with pytest.raises(ValueError, match="overlaps train/development lineage_id"):
        _run(args)
    test_features.write_bytes(original_test)
    test_feature_receipt.write_bytes(original_test_receipt)
    lock_path.write_bytes(valid_lock)
    opened_test = False

    report = _run(args)
    assert opened_test is True
    assert report["evaluation_split"] == "test" and report["test_opened"] is True
    changed_fraction = report["shuffled_latent_changed_row_fraction"]
    assert changed_fraction["train_development"] > 0
    assert changed_fraction["evaluation"] > 0
