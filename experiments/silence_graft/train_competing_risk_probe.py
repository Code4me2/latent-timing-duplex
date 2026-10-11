#!/usr/bin/env python3
"""Screen observed first-event timing with a discrete competing-risk probe."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from latent_timing_duplex.silence import fit_softmax_probe, fixed_width_projection
from latent_timing_duplex.silence.outcomes import first_observed_event

try:
    from experiments.silence_graft.compile_r2_dataset import _exact_authenticated_package
    from experiments.silence_graft.receipts import read_hmac_key, require_disjoint_output
    from experiments.silence_graft.render_clips import _jsonl
    from experiments.silence_graft.train_probe import _feature_sets, _load, _stable_seed
except ModuleNotFoundError:  # Direct script execution from the repository root.
    from compile_r2_dataset import _exact_authenticated_package
    from receipts import read_hmac_key, require_disjoint_output
    from render_clips import _jsonl
    from train_probe import _feature_sets, _load, _stable_seed

BIN_EDGES_S = np.arange(0.0, 9.0, 1.0)
CLASSES = np.asarray(["no_event", "user_resume", "recorded_partner_entry"])
_DATASET_SCHEMA = "semantic-silence-r2-dataset-receipt/v2"
_FEATURE_SCHEMA = "semantic-silence-r2-feature-receipt/v1"
_OUTCOME_SCHEMA = "semantic-silence-observed-outcome/v2"


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"outcome {name} must be numeric")
    value = float(value)
    if not np.isfinite(value):
        raise ValueError(f"outcome {name} must be finite")
    return value


def _normalize_outcome(row: dict[str, Any]) -> dict[str, Any]:
    """Validate the compiled tie/censor contract and retain likelihood fields."""
    if row.get("schema") != _OUTCOME_SCHEMA or row.get("observation_horizon_s") != 8.0:
        raise ValueError("outcome schema or fixed eight-second horizon is invalid")
    duration = _finite(row.get("observed_duration_s"), "observed_duration_s")
    if not 0.0 < duration <= 8.0:
        raise ValueError("outcome duration must be in (0, 8]")
    delays: dict[str, float | None] = {}
    for cause, prefix in (
        ("user_resume", "user_resume"),
        ("recorded_partner_entry", "recorded_partner_entry"),
    ):
        value = row.get(f"{prefix}_delay_s")
        delays[cause] = None if value is None else _finite(value, f"{prefix}_delay_s")
        if delays[cause] is not None and not 0.0 < delays[cause] <= 8.0:
            raise ValueError("observed event delay must be in (0, 8]")
    user, partner = delays["user_resume"], delays["recorded_partner_entry"]
    event, _ = first_observed_event(
        row.get("user_resume_s"), row.get("recorded_partner_entry_s"), row.get("sample_rate")
    )
    if row.get("first_event") != event:
        raise ValueError("outcome violates the user-resumption tie rule")
    expected_duration = delays[event] if event is not None else (
        _finite(row.get("administrative_censor_s"), "administrative_censor_s")
        - _finite(row.get("model_cutoff_s"), "model_cutoff_s")
    )
    if abs(duration - float(expected_duration)) > 1e-8:
        raise ValueError("outcome duration does not match its event or censor time")
    expected_flags = {
        "user_resume_event": event == "user_resume",
        "recorded_partner_entry_event": event == "recorded_partner_entry",
        "administrative_censor_event": event is None,
        "right_censored": event is None,
    }
    if any(row.get(name) is not expected for name, expected in expected_flags.items()):
        raise ValueError("outcome event and censor indicators are inconsistent")
    return {
        "item_id": str(row.get("item_id", "")),
        "source": str(row.get("source", "")),
        "lineage_id": str(row.get("lineage_id", "")),
        "split": str(row.get("split", "")),
        "event": event,
        "duration_s": duration,
    }


def _load_features(feature_root: Path, hmac_key: Path) -> tuple[dict[str, np.ndarray], str, str]:
    _, receipt, outputs = _exact_authenticated_package(
        Path(feature_root),
        receipt_name="feature_receipt.json",
        receipt_schema=_FEATURE_SCHEMA,
        key=read_hmac_key(hmac_key),
        where="R2 feature package",
    )
    if set(outputs) != {"features.npz"}:
        raise ValueError("R2 feature package has an invalid output set")
    hashes = receipt.get("sha256")
    if not isinstance(hashes, dict) or not isinstance(hashes.get("dataset_receipt"), str):
        raise ValueError("R2 feature receipt does not bind its dataset receipt")
    path = Path(feature_root) / "features.npz"
    return _load(path), hashlib.sha256(path.read_bytes()).hexdigest(), hashes["dataset_receipt"]


def _load_outcomes(dataset_root: Path, hmac_key: Path) -> tuple[list[dict[str, Any]], str, int]:
    receipt_payload, receipt, outputs = _exact_authenticated_package(
        Path(dataset_root),
        receipt_name="dataset_receipt.json",
        receipt_schema=_DATASET_SCHEMA,
        key=read_hmac_key(hmac_key),
        where="R2 dataset package",
    )
    if receipt.get("splits") != ["train", "development"]:
        raise ValueError("R2 dataset must retain the locked train/development splits")
    if receipt.get("sampling_weight_identified") is not False:
        raise ValueError("R2 pilot sampling weights must remain explicitly unidentified")
    limitations = receipt.get("limitations")
    if not isinstance(limitations, list) or "recorded partner entry is outcome metadata, not normative supervision" not in limitations:
        raise ValueError("R2 dataset must keep recorded partner entry observational")
    rows: list[dict[str, Any]] = []
    for split in ("train", "development"):
        relative = f"future_outcomes/{split}.jsonl"
        if relative not in outputs:
            raise ValueError(f"R2 dataset is missing {relative}")
        payload = (Path(dataset_root) / relative).read_bytes()
        for row in (_jsonl(payload, relative) if payload.strip() else []):
            normalized = _normalize_outcome(row)
            if normalized["split"] != split:
                raise ValueError("outcome is stored under the wrong protected split")
            rows.append(normalized)
    counts = receipt.get("counts")
    excluded = counts.get("excluded_majority_unratable") if isinstance(counts, dict) else None
    if isinstance(excluded, bool) or not isinstance(excluded, int) or excluded < 0:
        raise ValueError("R2 dataset receipt requires an exclusion count")
    return rows, hashlib.sha256(receipt_payload).hexdigest(), excluded


def _align_outcomes(
    data: dict[str, np.ndarray], rows: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], set[str]]:
    by_item: dict[str, dict[str, Any]] = {}
    for row in rows:
        item = row["item_id"]
        if not item or item in by_item:
            raise ValueError("outcomes contain an empty or duplicate item_id")
        by_item[item] = row
    aligned = []
    for index, item in enumerate(data["item_id"].astype(str)):
        if item not in by_item:
            raise ValueError(f"feature item {item!r} has no observed outcome")
        row = by_item[item]
        for name in ("source", "lineage_id", "split"):
            if row[name] != str(data[name][index]):
                raise ValueError(f"item {item!r} has inconsistent {name} lineage")
        aligned.append(row)
    split = data["split"].astype(str)
    if set(split) != {"train", "development"}:
        raise ValueError("screening requires only, and both, train/development rows")
    lineage = data["lineage_id"].astype(str)
    if set(lineage[split == "train"]) & set(lineage[split == "development"]):
        raise ValueError("train/development lineages overlap")
    return aligned, set(by_item) - set(data["item_id"].astype(str))


def _person_periods(
    features: np.ndarray,
    outcomes: list[dict[str, Any]],
    edges: np.ndarray = BIN_EDGES_S,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Expand items into fully observed no-event bins plus any observed event bin."""
    x = np.asarray(features, dtype=np.float64)
    edges = np.asarray(edges, dtype=np.float64)
    if x.ndim != 2 or len(outcomes) != len(x):
        raise ValueError("features and outcomes must align")
    if edges.ndim != 1 or len(edges) < 2 or edges[0] != 0 or np.any(np.diff(edges) <= 0):
        raise ValueError("bin edges must increase from zero")
    time_basis = np.eye(len(edges) - 1)
    rows: list[np.ndarray] = []
    labels: list[str] = []
    items: list[int] = []
    for item, outcome in enumerate(outcomes):
        duration, event = float(outcome["duration_s"]), outcome["event"]
        if not 0.0 < duration <= edges[-1] + 1e-9:
            raise ValueError("outcome lies outside fixed bins")
        if event is None:
            count = int(np.sum(edges[1:] <= duration + 1e-9))
            bins, item_labels = range(count), ["no_event"] * count
        else:
            event_bin = int(np.searchsorted(edges, duration, side="left") - 1)
            event_bin = max(0, min(event_bin, len(time_basis) - 1))
            bins = range(event_bin + 1)
            item_labels = ["no_event"] * event_bin + [str(event)]
        for bin_index, label in zip(bins, item_labels):
            rows.append(np.concatenate([x[item], time_basis[bin_index]]))
            labels.append(label)
            items.append(item)
    width = x.shape[1] + len(time_basis)
    return (
        np.asarray(rows, dtype=np.float64).reshape(-1, width),
        np.asarray(labels),
        np.asarray(items, dtype=np.int64),
    )


def _item_losses(
    labels: np.ndarray, probabilities: np.ndarray, item_index: np.ndarray, n_items: int
) -> tuple[np.ndarray, np.ndarray]:
    class_index = {name: i for i, name in enumerate(CLASSES.tolist())}
    yi = np.asarray([class_index[name] for name in labels.tolist()])
    row_loss = -np.log(np.clip(probabilities[np.arange(len(labels)), yi], 1e-12, 1.0))
    counts = np.bincount(item_index, minlength=n_items)
    losses = np.bincount(item_index, weights=row_loss, minlength=n_items)
    return losses, counts


def _paired_bootstrap(
    model_loss: np.ndarray,
    baseline_loss: np.ndarray,
    lineage: np.ndarray,
    informative: np.ndarray,
    *,
    n_boot: int,
    seed: int,
) -> dict[str, float | int]:
    if n_boot < 1:
        raise ValueError("n_boot must be positive")
    groups = np.unique(lineage[informative])
    if len(groups) < 2:
        raise ValueError("paired comparison needs at least two informative development lineages")
    indices = {group: np.flatnonzero(informative & (lineage == group)) for group in groups}
    delta = baseline_loss - model_loss  # Positive means the named model has lower NLL.
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_boot):
        take = np.concatenate([indices[group] for group in rng.choice(groups, len(groups), replace=True)])
        draws.append(float(np.mean(delta[take])))
    return {
        "nll_improvement": float(np.mean(delta[informative])),
        "ci_low": float(np.quantile(draws, 0.025)),
        "ci_high": float(np.quantile(draws, 0.975)),
        "n_boot": n_boot,
    }


def run_screen(
    data: dict[str, np.ndarray],
    outcomes: list[dict[str, Any]],
    *,
    width: int = 128,
    steps: int = 1000,
    learning_rate: float = 0.1,
    n_boot: int = 10000,
    seed: int = 20261009,
    expected_omitted_outcomes: int = 0,
) -> dict[str, Any]:
    outcomes, omitted_outcomes = _align_outcomes(data, outcomes)
    if len(omitted_outcomes) != expected_omitted_outcomes:
        raise ValueError("outcomes without feature rows do not match authenticated exclusions")
    feature_sets, _ = _feature_sets(data, seed)
    split = data["split"].astype(str)
    train, development = split == "train", split == "development"
    train_outcomes = [outcomes[i] for i in np.flatnonzero(train)]
    development_outcomes = [outcomes[i] for i in np.flatnonzero(development)]
    development_lineage = data["lineage_id"].astype(str)[development]
    event_counts = {
        split_name: {
            cause: sum(row["event"] == cause for row in selected)
            for cause in ("user_resume", "recorded_partner_entry", None)
        }
        for split_name, selected in (
            ("train", train_outcomes), ("development", development_outcomes)
        )
    }
    if any(
        event_counts[split_name][cause] == 0
        for split_name in ("train", "development")
        for cause in ("user_resume", "recorded_partner_entry")
    ):
        raise ValueError("competing-risk screen is non-estimable without both causes in train and development")
    if np.unique(development_lineage).size < 2:
        raise ValueError("competing-risk screen needs at least two development lineages")
    results: dict[str, Any] = {}
    losses: dict[str, np.ndarray] = {}
    informative: np.ndarray | None = None
    n_train_informative = n_train_risk_intervals = n_development_risk_intervals = 0
    projection_seed = _stable_seed(seed, "competing_risk_shared_projection")
    for name, raw in feature_sets.items():
        projected = fixed_width_projection(raw, width, seed=projection_seed)
        train_x, train_y, train_item = _person_periods(projected[train], train_outcomes)
        development_x, development_y, development_item = _person_periods(
            projected[development], development_outcomes
        )
        probe = fit_softmax_probe(
            train_x,
            train_y,
            classes=CLASSES,
            steps=steps,
            learning_rate=learning_rate,
            class_balance=False,
        )
        probabilities = probe.predict_proba(development_x)
        item_loss, counts = _item_losses(
            development_y, probabilities, development_item, len(development_outcomes)
        )
        losses[name] = item_loss
        current_informative = counts > 0
        if informative is None:
            informative = current_informative
            n_train_informative = int(np.unique(train_item).size)
            n_train_risk_intervals = len(train_y)
            n_development_risk_intervals = len(development_y)
        elif not np.array_equal(informative, current_informative):
            raise AssertionError("feature sets produced different risk intervals")
        bin_index = np.argmax(development_x[:, -len(BIN_EDGES_S) + 1 :], axis=1)
        event_mass = {}
        for index in range(len(BIN_EDGES_S) - 1):
            rows = bin_index == index
            event_mass[f"{BIN_EDGES_S[index]:g}-{BIN_EDGES_S[index + 1]:g}s"] = {
                "n_intervals": int(rows.sum()),
                "observed_fraction": {
                    cause: float(np.mean(development_y[rows] == cause)) if rows.any() else None
                    for cause in CLASSES.tolist()
                },
                "predicted_mean_probability": {
                    cause: float(np.mean(probabilities[rows, cause_index])) if rows.any() else None
                    for cause_index, cause in enumerate(CLASSES.tolist())
                },
            }
        results[name] = {
            "raw_width": int(raw.shape[1]),
            "projected_feature_width": width,
            "time_basis_width": len(BIN_EDGES_S) - 1,
            "probe_width": int(train_x.shape[1]),
            "mean_item_nll": float(np.mean(item_loss[current_informative])),
            "mean_risk_interval_nll": float(np.sum(item_loss) / np.sum(counts)),
            "development_event_mass_by_bin": event_mass,
        }
    assert informative is not None
    baselines = ("duration_acoustic", "duration_energy_change_onset", "hidden")
    pairs = [
        (name, baseline)
        for name in feature_sets
        for baseline in baselines
        if name != baseline
    ]
    comparisons = {
        f"{model}_minus_{baseline}": _paired_bootstrap(
            losses[model], losses[baseline], development_lineage, informative,
            n_boot=n_boot, seed=_stable_seed(seed, f"risk:{model}:{baseline}"),
        )
        for model, baseline in pairs
    }
    candidate = "hidden_predicted_latent" if "hidden_predicted_latent" in losses else "hidden"
    diagnostic_pairs = [(candidate, baseline) for baseline in baselines if candidate != baseline]
    development_source = data["source"].astype(str)[development]
    development_elapsed = np.asarray(data["elapsed_s"][development], dtype=np.float64)

    def diagnostics(mask: np.ndarray, group: str) -> dict[str, Any]:
        selected = informative & mask
        if selected.sum() == 0 or np.unique(development_lineage[selected]).size < 2:
            return {"status": "insufficient_lineages", "n_items": int(selected.sum())}
        return {
            "status": "estimated",
            "n_items": int(selected.sum()),
            **{
                f"{model}_minus_{baseline}": _paired_bootstrap(
                    losses[model], losses[baseline], development_lineage, selected,
                    n_boot=n_boot, seed=_stable_seed(seed, f"risk:{group}:{model}:{baseline}"),
                )
                for model, baseline in diagnostic_pairs
            },
        }

    elapsed_diagnostics = {
        name: diagnostics(mask, name)
        for name, mask in {
            "short_elapsed_lt_1s": development_elapsed < 1.0,
            "medium_elapsed_1_to_lt_3s": (development_elapsed >= 1.0) & (development_elapsed < 3.0),
            "long_elapsed_ge_3s": development_elapsed >= 3.0,
        }.items()
    }
    source_diagnostics = {
        source: diagnostics(development_source == source, f"source:{source}")
        for source in np.unique(development_source)
    }
    for counts in event_counts.values():
        counts["right_censored"] = counts.pop(None)
    return {
        "schema": "semantic-silence-competing-risk-probe-report/v1",
        "analysis_status": "development_only_exploratory",
        "event_roles": {
            "user_resume": "observed first-event cause",
            "recorded_partner_entry": "observed first-event cause; not normative assistant timing",
            "no_event": "event-free continuation through a fully observed at-risk interval",
        },
        "bin_edges_s": BIN_EDGES_S.tolist(),
        "censoring": "partial terminal censor intervals omitted; completed no-event intervals retained",
        "tie_rule": "user resumption wins when events round to the same source-sample frame",
        "seed": seed,
        "outcomes_without_features": {
            "count": len(omitted_outcomes),
            "reason": "majority-unratable items are excluded from causal labels and features",
        },
        "n_train": int(train.sum()),
        "n_train_informative": n_train_informative,
        "n_train_risk_intervals": n_train_risk_intervals,
        "n_development": int(development.sum()),
        "n_development_informative": int(informative.sum()),
        "n_development_risk_intervals": n_development_risk_intervals,
        "event_counts": event_counts,
        "results": results,
        "required_elapsed_strata": elapsed_diagnostics,
        "source_diagnostics": source_diagnostics,
        "paired_lineage_bootstrap_nll": comparisons,
        "comparison_direction": "positive nll_improvement means the feature set before '_minus_' has lower mean item NLL",
        "limitations": [
            "Observational first-event screening only; recorded partner entry is not normative assistant timing.",
            "Deterministic pilot sampling does not identify population incidence or calibration.",
            "Acceptable-onset and premature-interruption efficacy require replay and are not computed.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--hmac-key", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261009)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--n-boot", type=int, default=10000)
    args = parser.parse_args()
    require_disjoint_output(args.output, [args.feature_root, args.dataset_root])
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite existing report: {args.output}")
    data, feature_sha256, bound_dataset_sha256 = _load_features(
        args.feature_root, args.hmac_key
    )
    outcomes, dataset_receipt_sha256, excluded_outcomes = _load_outcomes(
        args.dataset_root, args.hmac_key
    )
    if bound_dataset_sha256 != dataset_receipt_sha256:
        raise ValueError("feature and outcome packages bind different R2 datasets")
    report = run_screen(
        data, outcomes, width=args.width, steps=args.steps,
        learning_rate=args.learning_rate, n_boot=args.n_boot, seed=args.seed,
        expected_omitted_outcomes=excluded_outcomes,
    )
    report["feature_sha256"] = feature_sha256
    report["dataset_receipt_sha256"] = dataset_receipt_sha256
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")


if __name__ == "__main__":
    main()
