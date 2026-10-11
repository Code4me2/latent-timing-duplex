#!/usr/bin/env python3
"""Train equal-protocol linear probes on locked causal feature archives."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
from typing import Any

import numpy as np

from latent_timing_duplex.silence import (
    fit_softmax_probe,
    fixed_width_projection,
    multiclass_metrics,
    paired_lineage_bootstrap_delta,
)

try:
    from experiments.silence_graft.receipts import (
        canonical_json,
        read_hmac_key,
        require_digest,
        sha256_bytes,
        sha256_file,
        verify_receipt,
    )
    from experiments.silence_graft.render_clips import _json
except ModuleNotFoundError:  # Direct script execution from the repository root.
    from receipts import canonical_json, read_hmac_key, require_digest, sha256_bytes, sha256_file, verify_receipt
    from render_clips import _json

REQUIRED = {
    "item_id",
    "session_id",
    "source",
    "lineage_id",
    "split",
    "action",
    "context",
    "elapsed_s",
    "model_cutoff_s",
    "available_at_s",
    "acoustic",
    "energy_change",
    "history_truncated",
    "sampling_weight_status",
    "hidden",
}
_LOCK_SCHEMA = "semantic-silence-analysis-lock/v1"
_FEATURE_RECEIPT_SCHEMA = "semantic-silence-r2-feature-receipt/v1"
_BASELINES = ("hidden", "duration_acoustic", "duration_energy_change_onset")


def _matrix(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64)
    return value[:, None] if value.ndim == 1 else value


def _stable_seed(seed: int, name: str) -> int:
    return int.from_bytes(hashlib.sha256(f"{seed}:{name}".encode()).digest()[:4], "big")


def _shuffle_latent(
    latent: np.ndarray,
    split: np.ndarray,
    source: np.ndarray,
    elapsed: np.ndarray,
    *,
    seed: int,
) -> np.ndarray:
    """Shuffle within split × source × configured elapsed stratum."""
    result = np.asarray(latent).copy()
    bins = np.digitize(elapsed, [1.0, 3.0])
    rng = np.random.default_rng(seed)
    keys = np.asarray([f"{a}|{b}|{c}" for a, b, c in zip(split, source, bins)])
    for key in np.unique(keys):
        rows = np.flatnonzero(keys == key)
        if rows.size > 1:
            result[rows] = result[rng.permutation(rows)]
    return result


def _load(source: Path | bytes) -> dict[str, np.ndarray]:
    with np.load(io.BytesIO(source) if isinstance(source, bytes) else source, allow_pickle=False) as archive:
        missing = REQUIRED - set(archive.files)
        if missing:
            raise ValueError(f"feature archive missing arrays: {sorted(missing)}")
        data = {name: archive[name] for name in archive.files}
    n = len(data["action"])
    if n == 0 or any(value.ndim == 0 or len(value) != n for value in data.values()):
        raise ValueError("every feature archive array must share a non-empty first dimension")
    for name in ("available_at_s", "model_cutoff_s", "elapsed_s"):
        values = np.asarray(data[name], dtype=np.float64)
        if values.shape != (n,) or not np.all(np.isfinite(values)):
            raise ValueError(f"feature archive {name} must be a finite row vector")
    if np.any(data["elapsed_s"] < 0):
        raise ValueError("feature archive elapsed_s must be non-negative")
    if np.any(data["available_at_s"] > data["model_cutoff_s"] + 1e-9):
        raise ValueError("feature archive contains future-observing rows")
    metadata_names = (
        "item_id", "session_id", "source", "lineage_id", "split", "action", "context",
        "sampling_weight_status",
    )
    for name in metadata_names:
        if data[name].shape != (n,) or data[name].dtype.kind not in {"U", "S"}:
            raise ValueError(f"feature archive {name} must be a one-dimensional string vector")
    item, session, source, lineage, split = (
        data[name].astype(str) for name in ("item_id", "session_id", "source", "lineage_id", "split")
    )
    if len(set(item.tolist())) != n:
        raise ValueError("feature archive item_id values must be unique")
    if any(not value for values in (item, session, source, lineage, split) for value in values):
        raise ValueError("item, session, source, lineage, and split identities must be non-empty")
    for values, groups, where in (
        (np.asarray([f"{a}\0{b}" for a, b in zip(source, split)]), lineage, "lineage"),
        (np.asarray([f"{a}\0{b}\0{c}" for a, b, c in zip(lineage, source, split)]), session, "session"),
    ):
        if any(np.unique(values[groups == group]).size != 1 for group in np.unique(groups)):
            raise ValueError(f"feature archive has inconsistent {where} split bindings")
    for name in (
        "acoustic", "energy_change", "history_truncated", "hidden", "hidden_previous", "hidden_velocity",
        "predicted_latent", "untrained_latent", "shuffled_target_latent",
    ):
        if name not in data:
            continue
        values = _matrix(data[name])
        if values.ndim != 2 or values.shape[1] == 0 or not np.isfinite(values).all():
            raise ValueError(f"feature archive {name} must be a finite matrix")
    if np.any(data["sampling_weight_status"].astype(str) != "not_identifiable"):
        raise ValueError("pilot sampling weights must remain not_identifiable")
    return data


def _feature_set_names(data: dict[str, np.ndarray]) -> list[str]:
    names = ["duration_acoustic", "duration_energy_change_onset", "hidden"]
    if "hidden_velocity" in data:
        names.append("hidden_velocity")
    if "predicted_latent" in data:
        names.extend(("hidden_predicted_latent", "hidden_duration_matched_shuffled_latent"))
    for key in ("untrained_latent", "shuffled_target_latent"):
        if key in data:
            names.append(f"hidden_{key}")
    return names


def _feature_sets(data: dict[str, np.ndarray], seed: int) -> tuple[dict[str, np.ndarray], float | None]:
    elapsed = np.asarray(data["elapsed_s"], dtype=np.float64)
    duration = np.log1p(elapsed)[:, None]
    truncation = _matrix(data["history_truncated"])
    acoustic = np.concatenate([duration, _matrix(data["acoustic"]), truncation], axis=1)
    energy_change = np.concatenate(
        [duration, _matrix(data["energy_change"]), truncation], axis=1
    )
    hidden = np.concatenate([duration, _matrix(data["hidden"])], axis=1)
    result: dict[str, np.ndarray] = {
        "duration_acoustic": acoustic,
        "duration_energy_change_onset": energy_change,
        "hidden": hidden,
    }
    if "hidden_velocity" in data:
        result["hidden_velocity"] = np.concatenate(
            [duration, _matrix(data["hidden_velocity"])], axis=1
        )
    changed: float | None = None
    if "predicted_latent" in data:
        latent = _matrix(data["predicted_latent"])
        result["hidden_predicted_latent"] = np.concatenate([hidden, latent], axis=1)
        shuffled = _shuffle_latent(
            latent,
            data["split"].astype(str),
            data["source"].astype(str),
            elapsed,
            seed=seed,
        )
        changed = float(np.mean(np.any(shuffled != latent, axis=1)))
        result["hidden_duration_matched_shuffled_latent"] = np.concatenate(
            [hidden, shuffled], axis=1
        )
    for key in ("untrained_latent", "shuffled_target_latent"):
        if key in data:
            result[f"hidden_{key}"] = np.concatenate([hidden, _matrix(data[key])], axis=1)
    return result, changed


def _comparisons(names: list[str]) -> list[str]:
    return [
        f"{name}_minus_{baseline}"
        for name in names
        for baseline in _BASELINES
        if name != baseline
    ]


def _verify_analysis_lock(args: argparse.Namespace) -> tuple[dict[str, Any], str, bytes]:
    """Verify every pre-test commitment without touching the test archive."""
    if (
        args.analysis_lock is None or args.hmac_key is None
        or args.test_features is None or args.test_feature_receipt is None
    ):
        raise ValueError(
            "test evaluation requires an authenticated analysis-lock receipt plus "
            "--test-features, --test-feature-receipt, and --hmac-key"
        )
    key = read_hmac_key(args.hmac_key)
    lock_payload = Path(args.analysis_lock).read_bytes()
    lock = _json(lock_payload, "analysis lock")
    verify_receipt(lock, key, "analysis lock")
    if lock.get("schema") != _LOCK_SCHEMA or lock.get("evaluation_split") != "test":
        raise ValueError("invalid analysis-lock schema or evaluation split")
    if lock.get("development_only_selection") is not True:
        raise ValueError("analysis lock must attest development-only selection")
    if lock.get("test_access_security_model") != (
        "shared-HMAC integrity check only; external sealed custody and role-separated "
        "commitment are required for confirmatory use"
    ):
        raise ValueError("analysis lock must state the shared-key test-access limitation")
    expected = {
        "protocol_version": "semantic-silence-pilot/v1",
        "taxonomy": args.classes,
        "eligible_context": "joint_silence",
        "exclusions": ["unratable", "technical"],
        "metrics": ["macro_auprc", "nll"],
        "bootstrap": {
            "unit": "lineage", "seed": args.seed, "replicates": args.n_boot,
            "minimum_valid_fraction": 0.9,
        },
        "probe": {"width": args.width, "steps": args.steps, "learning_rate": args.learning_rate},
        "primary_metric": args.primary_metric,
    }
    if any(lock.get(name) != value for name, value in expected.items()):
        raise ValueError("analysis lock does not match exact CLI hyperparameters")
    if lock["primary_metric"] not in lock["metrics"]:
        raise ValueError("analysis lock has an invalid primary metric")
    config = lock.get("analysis_config")
    if not isinstance(config, dict) or not isinstance(config.get("thresholds"), dict) or not config["thresholds"]:
        raise ValueError("analysis lock lacks frozen R2 thresholds")
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not np.isfinite(float(value))
        for value in config["thresholds"].values()
    ):
        raise ValueError("analysis lock has invalid frozen R2 thresholds")
    hashes = lock.get("sha256")
    required = {
        "dataset_receipt", "feature_receipt", "train_development_features",
        "sealed_test_commitment", "sealed_test_feature_receipt_commitment",
        "sealed_test_dataset_receipt_commitment", "analysis_config", "analysis_config_source_bytes", "train_probe.py",
        "silence/probe.py", "silence/metrics.py", "eval/metrics.py",
    }
    if not isinstance(hashes, dict) or set(hashes) != required:
        raise ValueError("analysis lock has missing or unknown SHA-256 commitments")
    for name in required:
        require_digest(hashes[name], f"analysis lock sha256.{name}")
    if hashes["analysis_config"] != sha256_bytes(canonical_json(config)):
        raise ValueError("analysis config does not match analysis lock")
    root = Path(__file__).resolve().parents[2]
    code_paths = {
        "train_probe.py": Path(__file__),
        "silence/probe.py": root / "src/latent_timing_duplex/silence/probe.py",
        "silence/metrics.py": root / "src/latent_timing_duplex/silence/metrics.py",
        "eval/metrics.py": root / "src/latent_timing_duplex/eval/metrics.py",
    }
    if any(hashes[name] != sha256_file(path) for name, path in code_paths.items()):
        raise ValueError("analysis code does not match analysis lock")
    feature_payload = Path(args.features).read_bytes()
    if hashes["train_development_features"] != sha256_bytes(feature_payload):
        raise ValueError("train/development feature bytes do not match analysis lock")
    test_receipt_path = Path(args.test_feature_receipt)
    if test_receipt_path.is_symlink() or not test_receipt_path.is_file():
        raise ValueError("test feature receipt must be a regular non-symlink file")
    test_receipt_payload = test_receipt_path.read_bytes()
    if sha256_bytes(test_receipt_payload) != hashes["sealed_test_feature_receipt_commitment"]:
        raise ValueError("sealed test feature receipt commitment mismatch")
    test_receipt = _json(test_receipt_payload, "test feature receipt")
    verify_receipt(test_receipt, key, "test feature receipt")
    test_acoustic_recipe = test_receipt.get("acoustic_recipe")
    test_hashes = test_receipt.get("sha256")
    required_feature_hashes = {
        "features.npz", "dataset_receipt", "adjudication_receipt", "assignment_receipt",
        "private_assignment_source_map", "renderer_receipt", "external_feature_npz",
        "external_feature_receipt", "config",
    }
    if not isinstance(test_hashes, dict) or set(test_hashes) != required_feature_hashes:
        raise ValueError("test feature receipt lacks the production provenance chain")
    for name, value in test_hashes.items():
        require_digest(value, f"test feature receipt sha256.{name}")
    if (
        test_receipt.get("schema") != _FEATURE_RECEIPT_SCHEMA
        or test_receipt.get("protocol_version") != lock["protocol_version"]
        or test_receipt.get("outputs") != {"features.npz": hashes["sealed_test_commitment"]}
        or test_receipt.get("frozen_feature_provenance") != lock.get("frozen_feature_provenance")
        or not isinstance(test_acoustic_recipe, dict)
        or sha256_bytes(canonical_json(test_acoustic_recipe)) != lock.get("acoustic_recipe_sha256")
        or test_hashes["features.npz"] != hashes["sealed_test_commitment"]
        or test_hashes["dataset_receipt"] != hashes["sealed_test_dataset_receipt_commitment"]
        or test_hashes["config"] != lock.get("feature_config_sha256")
    ):
        raise ValueError("test feature receipt does not match the sealed analysis commitments")
    return lock, sha256_bytes(lock_payload), feature_payload


def _run(args: argparse.Namespace) -> dict[str, Any]:
    if args.evaluate_test:
        lock, analysis_lock_hash, feature_payload = _verify_analysis_lock(args)
        data = _load(feature_payload)
        feature_hash = sha256_bytes(feature_payload)
    else:
        lock, analysis_lock_hash = None, None
        data = _load(args.features)
        feature_hash = sha256_file(args.features)
    ordinary_splits = set(data["split"].astype(str).tolist())
    if not ordinary_splits <= {"train", "development"}:
        raise ValueError("ordinary --features archive must not contain test rows")
    train_sets, train_changed = _feature_sets(data, args.seed)
    names = list(train_sets)
    test_feature_hash: str | None = None
    if lock is not None:
        if lock.get("feature_sets") != names or lock.get("comparisons") != _comparisons(names):
            raise ValueError("analysis lock feature sets or comparisons do not match the archive")
        test_path = Path(args.test_features)
        if test_path.is_symlink() or not test_path.is_file() or Path(args.features).samefile(test_path):
            raise ValueError("test features must be a physically separate regular non-symlink archive")
        # Deliberately first content access to --test-features: every lock/code/config/CLI check above has passed.
        test_payload = test_path.read_bytes()
        test_feature_hash = sha256_bytes(test_payload)
        if test_feature_hash != lock["sha256"]["sealed_test_commitment"]:
            raise ValueError("sealed test feature commitment mismatch")
        evaluation_data = _load(test_payload)
        if set(evaluation_data["split"].astype(str).tolist()) != {"test"}:
            raise ValueError("sealed test archive must contain only test rows")
        for identity in ("item_id", "session_id", "lineage_id"):
            if set(data[identity].astype(str)) & set(evaluation_data[identity].astype(str)):
                raise ValueError(f"sealed test archive overlaps train/development {identity} values")
        evaluation_sets, evaluation_changed = _feature_sets(evaluation_data, args.seed)
        if list(evaluation_sets) != names:
            raise ValueError("train/development and test feature sets differ")
        evaluation_name = "test"
    else:
        evaluation_data, evaluation_sets = data, train_sets
        evaluation_changed = train_changed
        evaluation_name = "development"

    classes = np.asarray(args.classes)
    labels = data["action"].astype(str)
    split = data["split"].astype(str)
    eligible = np.isin(labels, classes) & (data["context"].astype(str) == "joint_silence")
    train = eligible & (split == "train")
    evaluation_labels = evaluation_data["action"].astype(str)
    evaluation_eligible = np.isin(evaluation_labels, classes) & (
        evaluation_data["context"].astype(str) == "joint_silence"
    )
    evaluate = evaluation_eligible & (evaluation_data["split"].astype(str) == evaluation_name)
    if train.sum() == 0 or evaluate.sum() == 0:
        raise ValueError(f"need eligible train and {evaluation_name} rows")

    results: dict[str, dict[str, object]] = {}
    predictions: dict[str, np.ndarray] = {}
    projection_seed = _stable_seed(args.seed, "shared_projection")
    for name in names:
        raw_train, raw_evaluate = train_sets[name], evaluation_sets[name]
        if raw_train.shape[1] != raw_evaluate.shape[1]:
            raise ValueError(f"feature width mismatch for {name}")
        # Recreate one deterministic projection over the shared raw width.
        joined = np.concatenate([raw_train, raw_evaluate], axis=0)
        projected = fixed_width_projection(joined, args.width, seed=projection_seed)
        projected_train, projected_evaluate = projected[: len(raw_train)], projected[len(raw_train) :]
        probe = fit_softmax_probe(
            projected_train[train], labels[train], classes=classes,
            steps=args.steps, learning_rate=args.learning_rate,
        )
        probabilities = probe.predict_proba(projected_evaluate[evaluate])
        predictions[name] = probabilities
        results[name] = {
            "raw_width": raw_train.shape[1],
            "probe_width": projected_train.shape[1],
            **multiclass_metrics(evaluation_labels[evaluate], probabilities, classes),
        }

    lineage = evaluation_data["lineage_id"].astype(str)
    statistic = None
    if args.primary_metric == "nll":
        statistic = lambda y, p, c: -float(multiclass_metrics(y, p, c)["nll"])
    comparisons: dict[str, object] = {}
    for name, probabilities in predictions.items():
        for baseline in _BASELINES:
            if name == baseline:
                continue
            comparisons[f"{name}_minus_{baseline}"] = paired_lineage_bootstrap_delta(
                evaluation_labels[evaluate], probabilities, predictions[baseline], classes,
                lineage[evaluate], n_boot=args.n_boot,
                seed=_stable_seed(args.seed, f"{name}:{baseline}"), statistic=statistic,
            )

    elapsed_evaluate = np.asarray(evaluation_data["elapsed_s"][evaluate], dtype=np.float64)
    source_evaluate = evaluation_data["source"].astype(str)[evaluate]
    evaluate_lineage = lineage[evaluate]
    candidate = "hidden_predicted_latent"

    def grouped_comparisons(rows: np.ndarray, group: str) -> dict[str, object]:
        group_lineages = evaluate_lineage[rows]
        if rows.sum() == 0 or np.unique(group_lineages).size < 2:
            return {"status": "insufficient_lineages", "n_rows": int(rows.sum())}
        estimates = {
            f"{candidate}_minus_{baseline}": paired_lineage_bootstrap_delta(
                evaluation_labels[evaluate][rows], predictions[candidate][rows],
                predictions[baseline][rows], classes, group_lineages,
                n_boot=args.n_boot,
                seed=_stable_seed(args.seed, f"{group}:{candidate}:{baseline}"),
                statistic=statistic,
            )
            for baseline in _BASELINES
        }
        return {
            "status": (
                "estimated" if any(value["delta"] is not None for value in estimates.values())
                else "non_estimable_missing_classes"
            ),
            "n_rows": int(rows.sum()),
            **estimates,
        }

    strata: dict[str, object] = {}
    source_diagnostics: dict[str, object] = {}
    if candidate in predictions:
        strata = {
            name: grouped_comparisons(rows, name)
            for name, rows in {
                "short_elapsed_lt_1s": elapsed_evaluate < 1.0,
                "medium_elapsed_1_to_lt_3s": (elapsed_evaluate >= 1.0) & (elapsed_evaluate < 3.0),
                "long_elapsed_ge_3s": elapsed_evaluate >= 3.0,
            }.items()
        }
        source_diagnostics = {
            source: grouped_comparisons(source_evaluate == source, f"source:{source}")
            for source in np.unique(source_evaluate)
        }

    return {
        "schema": "semantic-silence-probe-report/v1",
        "feature_sha256": feature_hash,
        "test_feature_sha256": test_feature_hash,
        "analysis_lock_sha256": analysis_lock_hash,
        "seed": args.seed,
        "primary_metric": args.primary_metric,
        "classes": classes.tolist(),
        "evaluation_split": evaluation_name,
        "test_opened": bool(args.evaluate_test),
        "n_train": int(train.sum()),
        "n_evaluate": int(evaluate.sum()),
        "shuffled_latent_changed_row_fraction": {
            "train_development": train_changed,
            "evaluation": evaluation_changed,
        },
        "results": results,
        "paired_primary_metric": comparisons,
        "required_elapsed_strata": strata,
        "source_diagnostics": source_diagnostics,
        "comparison_direction": "positive delta means the feature set before '_minus_' screens better",
        "limitations": [
            "Linear frozen-probe screening only; this report is not a graft efficacy result.",
            "Pilot prevalence, population calibration, and population hazard incidence are not recoverable.",
            "Acceptable-onset and premature-interruption gates require replay evaluation.",
            "The shared-HMAC lock checks integrity, not role separation or pre-test chronology; confirmatory access requires external sealed custody.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261009)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--n-boot", type=int, default=10000)
    parser.add_argument("--classes", nargs="+", default=["wait", "ack", "respond", "repair"])
    parser.add_argument("--primary-metric", choices=("macro_auprc", "nll"), default="macro_auprc")
    parser.add_argument("--evaluate-test", action="store_true")
    parser.add_argument("--test-features", type=Path)
    parser.add_argument("--test-feature-receipt", type=Path)
    parser.add_argument("--analysis-lock", type=Path)
    parser.add_argument("--hmac-key", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite existing report: {args.output}")
    report = _run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")


if __name__ == "__main__":
    main()
