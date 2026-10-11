#!/usr/bin/env python3
"""Create the authenticated, pre-test R2 probe analysis lock."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

try:
    from experiments.silence_graft.receipts import (
        canonical_json,
        read_hmac_key,
        receipt_bytes,
        require_digest,
        require_disjoint_output,
        sha256_bytes,
        sha256_file,
        sign_receipt,
        verify_receipt,
    )
    from experiments.silence_graft.render_clips import _json, _require_ready
    from experiments.silence_graft.train_probe import _comparisons, _feature_set_names, _load
except ModuleNotFoundError:  # Direct script execution from the repository root.
    from receipts import (
        canonical_json,
        read_hmac_key,
        receipt_bytes,
        require_digest,
        require_disjoint_output,
        sha256_bytes,
        sha256_file,
        sign_receipt,
        verify_receipt,
    )
    from render_clips import _json, _require_ready
    from train_probe import _comparisons, _feature_set_names, _load

_SCHEMA = "semantic-silence-analysis-lock/v1"
_FEATURE_SCHEMA = "semantic-silence-r2-feature-receipt/v1"
_ANALYSIS_CONFIG_SCHEMA = "semantic-silence-r2-analysis-config/v1"
_LOCKED_THRESHOLDS = {
    "macro_auprc_improvement_min": 0.05,
    "paired_ci_lower_bound_strictly_above": 0.0,
    "acceptable_onset_hit_rate_improvement_min": 0.05,
    "premature_interruption_degradation_max": 0.02,
}


def _analysis_config(path: Path) -> tuple[bytes, dict[str, Any]]:
    payload = path.read_bytes()
    value = _json(payload, "analysis config")
    if value.get("schema") != _ANALYSIS_CONFIG_SCHEMA:
        raise ValueError("analysis config has an invalid schema")
    thresholds = value.get("thresholds")
    if thresholds != _LOCKED_THRESHOLDS:
        raise ValueError("analysis config does not match the locked R2 thresholds")
    if any(
        not isinstance(name, str)
        or not name
        or isinstance(threshold, bool)
        or not isinstance(threshold, (int, float))
        or not math.isfinite(float(threshold))
        for name, threshold in thresholds.items()
    ):
        raise ValueError("analysis config thresholds must be named finite numbers")
    if value.get("observational_hazard") != {
        "confirmatory_status": "development_only_exploratory",
        "bin_edges_s": [float(index) for index in range(9)],
        "events": ["user_resume", "recorded_partner_entry", "no_event"],
        "primary_metric": "mean_item_nll",
        "tie_rule": "user_resume",
    }:
        raise ValueError("analysis config has an invalid observational hazard specification")
    return payload, value


def create_analysis_lock(
    features: Path,
    feature_receipt: Path,
    sealed_test_commitment: str,
    sealed_test_feature_receipt_commitment: str,
    sealed_test_dataset_receipt_commitment: str,
    analysis_config: Path,
    hmac_key: Path,
    output: Path,
    *,
    classes: tuple[str, ...] = ("wait", "ack", "respond", "repair"),
    seed: int = 20261009,
    width: int = 128,
    steps: int = 1000,
    learning_rate: float = 0.1,
    n_boot: int = 10000,
    primary_metric: str = "macro_auprc",
) -> Path:
    """Sign a new-only lock after development selection and before test access."""
    features, feature_receipt, analysis_config, output = map(
        Path, (features, feature_receipt, analysis_config, output)
    )
    require_disjoint_output(output, [features.parent])
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing analysis lock: {output}")
    if features.name != "features.npz" or feature_receipt.name != "feature_receipt.json":
        raise ValueError("feature package requires canonical feature and receipt names")
    if features.parent != feature_receipt.parent:
        raise ValueError("features and feature receipt must come from one package")
    _require_ready(features.parent, "feature package")
    key = read_hmac_key(hmac_key)
    receipt_payload = feature_receipt.read_bytes()
    feature_lock = _json(receipt_payload, "feature receipt")
    verify_receipt(feature_lock, key, "feature receipt")
    if (
        feature_lock.get("schema") != _FEATURE_SCHEMA
        or feature_lock.get("protocol_version") != "semantic-silence-pilot/v1"
    ):
        raise ValueError("invalid feature receipt schema or protocol")
    feature_hash = sha256_file(features)
    if feature_lock.get("outputs") != {"features.npz": feature_hash}:
        raise ValueError("feature bytes do not exactly match feature receipt")
    if any(path.is_symlink() for path in features.parent.iterdir()):
        raise ValueError("feature package must not contain symlinks")
    actual = {path.name for path in features.parent.iterdir()}
    if actual != {features.name, feature_receipt.name, "READY"}:
        raise ValueError("feature package contains unrecorded or missing files")
    require_digest(sealed_test_commitment, "sealed test commitment")
    require_digest(sealed_test_feature_receipt_commitment, "sealed test feature receipt commitment")
    require_digest(sealed_test_dataset_receipt_commitment, "sealed test dataset receipt commitment")
    if not classes or len(classes) != len(set(classes)) or any(
        not isinstance(value, str) or not value for value in classes
    ):
        raise ValueError("classes must be unique non-empty names")
    if seed < 0 or width < 1 or steps < 1 or not math.isfinite(learning_rate) or learning_rate <= 0 or n_boot < 1:
        raise ValueError("invalid locked probe hyperparameters")
    if primary_metric not in {"macro_auprc", "nll"}:
        raise ValueError("primary metric must be macro_auprc or nll")

    data = _load(features)
    splits = set(data["split"].astype(str).tolist())
    if not splits <= {"train", "development"} or not {"train", "development"} <= splits:
        raise ValueError("analysis-lock features must contain train/development only")
    feature_sets = _feature_set_names(data)
    comparisons = _comparisons(feature_sets)
    config_payload, config_value = _analysis_config(analysis_config)
    if (
        config_value.get("primary_metric") != primary_metric
        or config_value.get("bootstrap") != {
            "unit": "lineage_id", "replicates": n_boot, "minimum_valid_fraction": 0.9
        }
        or config_value.get("required_elapsed_strata_s") != {"short": [0.0, 1.0], "long": [3.0, None]}
        or config_value.get("replay_metrics_status") != "required_but_unavailable_until_replay"
    ):
        raise ValueError("analysis config does not match locked CLI or R2 analysis settings")
    root = Path(__file__).resolve().parents[2]
    code_paths = {
        "train_probe.py": Path(__file__).with_name("train_probe.py"),
        "silence/probe.py": root / "src/latent_timing_duplex/silence/probe.py",
        "silence/metrics.py": root / "src/latent_timing_duplex/silence/metrics.py",
        "eval/metrics.py": root / "src/latent_timing_duplex/eval/metrics.py",
    }
    feature_hashes = feature_lock.get("sha256")
    provenance = feature_lock.get("frozen_feature_provenance")
    provenance_fields = {
        "model_id", "model_revision", "layer", "checkpoint_sha256", "extractor_code_sha256",
        "context_start", "state_reset", "history_truncation",
    }
    acoustic_recipe = feature_lock.get("acoustic_recipe")
    required_feature_hashes = {
        "features.npz", "dataset_receipt", "adjudication_receipt", "assignment_receipt",
        "private_assignment_source_map", "renderer_receipt", "external_feature_npz",
        "external_feature_receipt", "config",
    }
    if (
        not isinstance(feature_hashes, dict) or set(feature_hashes) != required_feature_hashes
        or any(not isinstance(value, str) or len(value) != 64 for value in feature_hashes.values())
        or not isinstance(provenance, dict) or set(provenance) != provenance_fields
        or not isinstance(acoustic_recipe, dict) or not acoustic_recipe
    ):
        raise ValueError("feature receipt requires complete extractor and acoustic provenance")
    for name, value in feature_hashes.items():
        require_digest(value, f"feature receipt sha256.{name}")
    if feature_hashes["features.npz"] != feature_hash:
        raise ValueError("feature receipt input hash does not match feature bytes")
    dataset_receipt = require_digest(
        feature_hashes.get("dataset_receipt"), "feature receipt sha256.dataset_receipt"
    )
    lock = sign_receipt({
        "schema": _SCHEMA,
        "protocol_version": feature_lock.get("protocol_version"),
        "evaluation_split": "test",
        "taxonomy": list(classes),
        "eligible_context": "joint_silence",
        "exclusions": ["unratable", "technical"],
        "feature_sets": feature_sets,
        "comparisons": comparisons,
        "metrics": ["macro_auprc", "nll"],
        "primary_metric": primary_metric,
        "development_only_selection": True,
        "test_access_security_model": (
            "shared-HMAC integrity check only; external sealed custody and role-separated "
            "commitment are required for confirmatory use"
        ),
        "frozen_feature_provenance": provenance,
        "feature_config_sha256": feature_hashes["config"],
        "acoustic_recipe_sha256": sha256_bytes(canonical_json(acoustic_recipe)),
        "analysis_config": config_value,
        "bootstrap": {
            "unit": "lineage", "seed": seed, "replicates": n_boot,
            "minimum_valid_fraction": 0.9,
        },
        "probe": {"width": width, "steps": steps, "learning_rate": learning_rate},
        "sha256": {
            "dataset_receipt": dataset_receipt,
            "feature_receipt": sha256_bytes(receipt_payload),
            "train_development_features": feature_hash,
            "sealed_test_commitment": sealed_test_commitment,
            "sealed_test_feature_receipt_commitment": sealed_test_feature_receipt_commitment,
            "sealed_test_dataset_receipt_commitment": sealed_test_dataset_receipt_commitment,
            "analysis_config": sha256_bytes(canonical_json(config_value)),
            "analysis_config_source_bytes": sha256_bytes(config_payload),
            **{name: sha256_file(path) for name, path in code_paths.items()},
        },
    }, key)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as handle:
        handle.write(receipt_bytes(lock))
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--feature-receipt", type=Path, required=True)
    parser.add_argument("--sealed-test-commitment", required=True)
    parser.add_argument("--sealed-test-feature-receipt-commitment", required=True)
    parser.add_argument("--sealed-test-dataset-receipt-commitment", required=True)
    parser.add_argument("--analysis-config", type=Path, required=True)
    parser.add_argument("--hmac-key", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--classes", nargs="+", default=["wait", "ack", "respond", "repair"])
    parser.add_argument("--seed", type=int, default=20261009)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--n-boot", type=int, default=10000)
    parser.add_argument("--primary-metric", choices=("macro_auprc", "nll"), default="macro_auprc")
    args = parser.parse_args()
    create_analysis_lock(
        args.features, args.feature_receipt, args.sealed_test_commitment,
        args.sealed_test_feature_receipt_commitment, args.sealed_test_dataset_receipt_commitment,
        args.analysis_config, args.hmac_key, args.output, classes=tuple(args.classes),
        seed=args.seed, width=args.width, steps=args.steps,
        learning_rate=args.learning_rate, n_boot=args.n_boot,
        primary_metric=args.primary_metric,
    )


if __name__ == "__main__":
    main()
