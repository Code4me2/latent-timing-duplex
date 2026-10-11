#!/usr/bin/env python3
"""Train equal-protocol linear probes on a locked causal feature NPZ."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from latent_timing_duplex.silence import (
    fit_softmax_probe,
    fixed_width_projection,
    multiclass_metrics,
    paired_lineage_bootstrap_delta,
)

REQUIRED = {
    "session_id",
    "lineage_id",
    "split",
    "action",
    "context",
    "elapsed_s",
    "model_cutoff_s",
    "available_at_s",
    "acoustic",
    "hidden",
}


def _matrix(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64)
    return value[:, None] if value.ndim == 1 else value


def _stable_seed(seed: int, name: str) -> int:
    return int.from_bytes(hashlib.sha256(f"{seed}:{name}".encode()).digest()[:4], "big")


def _shuffle_latent(
    latent: np.ndarray,
    split: np.ndarray,
    lineage: np.ndarray,
    elapsed: np.ndarray,
    *,
    seed: int,
) -> np.ndarray:
    """Shuffle within lineage and coarse duration stratum, separately by split."""
    result = np.asarray(latent).copy()
    bins = np.digitize(elapsed, [1.0, 3.0])
    rng = np.random.default_rng(seed)
    keys = np.asarray([f"{a}|{b}|{c}" for a, b, c in zip(split, lineage, bins)])
    for key in np.unique(keys):
        rows = np.flatnonzero(keys == key)
        if rows.size > 1:
            result[rows] = result[rng.permutation(rows)]
    return result


def _load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        missing = REQUIRED - set(archive.files)
        if missing:
            raise ValueError(f"feature archive missing arrays: {sorted(missing)}")
        data = {name: archive[name] for name in archive.files}
    n = len(data["action"])
    if any(len(value) != n for value in data.values()):
        raise ValueError("every feature archive array must share its first dimension")
    for name in ("available_at_s", "model_cutoff_s", "elapsed_s"):
        values = np.asarray(data[name], dtype=np.float64)
        if values.shape != (n,) or not np.all(np.isfinite(values)):
            raise ValueError(f"feature archive {name} must be a finite row vector")
    if np.any(data["elapsed_s"] < 0):
        raise ValueError("feature archive elapsed_s must be non-negative")
    if np.any(data["available_at_s"] > data["model_cutoff_s"] + 1e-9):
        raise ValueError("feature archive contains future-observing rows")
    session, lineage, split = (
        data[name].astype(str) for name in ("session_id", "lineage_id", "split")
    )
    if any(not value for values in (session, lineage, split) for value in values):
        raise ValueError("session, lineage, and split identities must be non-empty")
    for values, groups, where in (
        (split, lineage, "lineage"),
        (np.asarray([f"{a}\0{b}" for a, b in zip(lineage, split)]), session, "session"),
    ):
        if any(np.unique(values[groups == group]).size != 1 for group in np.unique(groups)):
            raise ValueError(f"feature archive has inconsistent {where} split bindings")
    return data


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261009)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument(
        "--classes", nargs="+", default=["wait", "ack", "respond", "repair"]
    )
    parser.add_argument("--evaluate-test", action="store_true")
    args = parser.parse_args()
    if args.evaluate_test:
        raise ValueError(
            "test evaluation requires an authenticated analysis-lock receipt; not implemented"
        )

    data = _load(args.features)
    classes = np.asarray(args.classes)
    split = data["split"].astype(str)
    labels = data["action"].astype(str)
    lineage = data["lineage_id"].astype(str)
    elapsed = np.asarray(data["elapsed_s"], dtype=np.float64)
    eligible = np.isin(labels, classes) & (data["context"].astype(str) == "joint_silence")
    train = eligible & (split == "train")
    evaluation_name = "test" if args.evaluate_test else "development"
    evaluate = eligible & (split == evaluation_name)
    if train.sum() == 0 or evaluate.sum() == 0:
        raise ValueError(f"need eligible train and {evaluation_name} rows")

    duration = np.log1p(elapsed)[:, None]
    acoustic = np.concatenate([duration, _matrix(data["acoustic"])], axis=1)
    hidden = np.concatenate([duration, _matrix(data["hidden"])], axis=1)
    feature_sets: dict[str, np.ndarray] = {
        "duration_acoustic": acoustic,
        "hidden": hidden,
    }
    if "hidden_velocity" in data:
        feature_sets["hidden_velocity"] = np.concatenate(
            [duration, _matrix(data["hidden_velocity"])], axis=1
        )
    if "predicted_latent" in data:
        latent = _matrix(data["predicted_latent"])
        feature_sets["hidden_predicted_latent"] = np.concatenate(
            [hidden, latent], axis=1
        )
        shuffled = _shuffle_latent(
            latent, split, lineage, elapsed, seed=args.seed
        )
        feature_sets["hidden_duration_matched_shuffled_latent"] = np.concatenate(
            [hidden, shuffled], axis=1
        )
    for key in ("untrained_latent", "shuffled_target_latent"):
        if key in data:
            feature_sets[f"hidden_{key}"] = np.concatenate(
                [hidden, _matrix(data[key])], axis=1
            )

    results: dict[str, dict[str, object]] = {}
    predictions: dict[str, np.ndarray] = {}
    projection_seed = _stable_seed(args.seed, "shared_projection")
    for name, raw in feature_sets.items():
        # A shared seed keeps the projection of common hidden columns identical
        # in aligned and shuffled-latent ablations.
        projected = fixed_width_projection(raw, args.width, seed=projection_seed)
        probe = fit_softmax_probe(
            projected[train],
            labels[train],
            classes=classes,
            steps=args.steps,
            learning_rate=args.learning_rate,
        )
        probabilities = probe.predict_proba(projected[evaluate])
        predictions[name] = probabilities
        results[name] = {
            "raw_width": raw.shape[1],
            "probe_width": projected.shape[1],
            **multiclass_metrics(labels[evaluate], probabilities, classes),
        }

    comparisons: dict[str, object] = {}
    for name, probabilities in predictions.items():
        for baseline in ("hidden", "duration_acoustic"):
            if name == baseline:
                continue
            comparisons[f"{name}_minus_{baseline}"] = paired_lineage_bootstrap_delta(
                labels[evaluate],
                probabilities,
                predictions[baseline],
                classes,
                lineage[evaluate],
                n_boot=args.n_boot,
                seed=_stable_seed(args.seed, f"{name}:{baseline}"),
            )

    report = {
        "schema": "semantic-silence-probe-report/v1",
        "feature_sha256": hashlib.sha256(args.features.read_bytes()).hexdigest(),
        "seed": args.seed,
        "classes": classes.tolist(),
        "evaluation_split": evaluation_name,
        "test_opened": bool(args.evaluate_test),
        "n_train": int(train.sum()),
        "n_evaluate": int(evaluate.sum()),
        "results": results,
        "paired_macro_auprc": comparisons,
        "limitations": [
            "Linear frozen-probe screening only; this report is not a graft efficacy result.",
            "Acceptable-onset and premature-interruption gates require replay evaluation.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
