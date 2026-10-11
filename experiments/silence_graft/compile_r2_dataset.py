#!/usr/bin/env python3
"""Compile authenticated labels and outcomes into the private pilot R2 dataset."""

from __future__ import annotations

import argparse
import hmac
import math
from collections import Counter
from pathlib import Path
from typing import Any

from latent_timing_duplex.silence.outcomes import first_observed_event

try:
    from experiments.silence_graft.build_assignments import (
        _publish_packages,
        _sha256,
        _verify_output_map,
    )
    from experiments.silence_graft.compile_annotations import (
        _exact_assignment_packages,
        _load_assignment_map,
    )
    from experiments.silence_graft.receipts import (
        read_hmac_key,
        receipt_bytes,
        require_digest,
        require_disjoint_output,
        sign_receipt,
        verify_receipt,
    )
    from experiments.silence_graft.render_clips import _json, _jsonl, _lines, _require_ready
except ModuleNotFoundError:  # Direct script execution from the repository root.
    from build_assignments import _publish_packages, _sha256, _verify_output_map
    from compile_annotations import _exact_assignment_packages, _load_assignment_map
    from receipts import read_hmac_key, receipt_bytes, require_digest, require_disjoint_output, sign_receipt, verify_receipt
    from render_clips import _json, _jsonl, _lines, _require_ready

_ADJUDICATION_SCHEMA = "semantic-silence-adjudication-receipt/v1"
_LABEL_SCHEMA = "semantic-silence-causal-model-label/v1"
_OUTCOME_SCHEMA = "semantic-silence-observed-outcome/v2"
_OUTCOME_RECEIPT_SCHEMA = "semantic-silence-outcomes-receipt/v2"
_DATASET_RECEIPT_SCHEMA = "semantic-silence-r2-dataset-receipt/v2"
_CAUSAL_SCHEMA = "semantic-silence-r2-causal-label/v1"
_EXCLUDED_SPLITS = {"test", "challenge"}
_PILOT_SPLITS = ("train", "development")
_ACTIONS = {"wait", "ack", "respond", "repair"}
_WEIGHT_REASON = "deterministic balanced pilot selection does not identify inclusion probabilities"
_LABEL_FIELDS = {
    "schema", "protocol_version", "item_id", "action", "action_distribution",
    "onset_guidance", "tags", "quality", "confidence", "split", "source",
    "lineage_id", "elapsed_s", "model_cutoff_s",
}
_OUTCOME_FIELDS = {
    "schema", "protocol_version", "item_id", "session_id", "episode_id", "source",
    "lineage_id", "split", "sample_rate", "onset_s", "model_cutoff_s", "checkpoint_elapsed_s",
    "observation_horizon_s", "observed_duration_s", "administrative_censor_s",
    "user_resume_s", "user_resume_delay_s", "user_resume_observed",
    "recorded_partner_entry_s", "recorded_partner_entry_delay_s",
    "recorded_partner_entry_observed", "first_event", "first_event_s",
    "user_resume_event", "recorded_partner_entry_event", "administrative_censor_event",
    "right_censored", "overlap_duration_s", "recorded_overlap_at_user_resume",
}


def _finite(value: Any, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{where} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{where} must be finite")
    return result


def _exact_authenticated_package(
    root: Path,
    *,
    receipt_name: str,
    receipt_schema: str,
    key: bytes,
    where: str,
) -> tuple[bytes, dict[str, Any], dict[str, str]]:
    _require_ready(root, where)
    receipt_path = root / receipt_name
    receipt_payload = receipt_path.read_bytes()
    receipt = _json(receipt_payload, f"{where} receipt")
    verify_receipt(receipt, key, f"{where} receipt")
    if receipt.get("schema") != receipt_schema:
        raise ValueError(f"invalid {where} receipt schema")
    outputs = _verify_output_map(root, receipt.get("outputs"), f"{where} receipt")
    if any(path.is_symlink() for path in root.rglob("*")):
        raise ValueError(f"{where} must not contain symlinks")
    actual = {str(path.relative_to(root)) for path in root.rglob("*") if path.is_file()}
    if actual != set(outputs) | {receipt_name, "READY"}:
        raise ValueError(f"{where} contains unrecorded or missing files")
    return receipt_payload, receipt, outputs


def _optional_jsonl(payload: bytes, where: str) -> list[dict[str, Any]]:
    return [] if not payload.strip() else _jsonl(payload, where)


def _load_splits(payload: bytes, protocol: str) -> dict[str, str]:
    data = _json(payload, "manifest splits")
    if data.get("schema") != "semantic-silence-config/v1" or data.get("protocol_version") != protocol:
        raise ValueError("manifest splits schema or protocol does not match dataset inputs")
    if isinstance(data.get("split_seed"), bool) or not isinstance(data.get("split_seed"), int):
        raise ValueError("manifest splits requires an integer split_seed")
    lineages = data.get("lineages")
    if not isinstance(lineages, dict) or not lineages:
        raise ValueError("manifest splits requires lineage assignments")
    if any(not isinstance(lineage, str) or not lineage or split not in {*_PILOT_SPLITS, *_EXCLUDED_SPLITS} for lineage, split in lineages.items()):
        raise ValueError("manifest splits contains an invalid lineage or split")
    return lineages


def _sampling_fields() -> dict[str, Any]:
    return {
        "sampling_probability": None,
        "sampling_weight": 1.0,
        "sampling_weight_identified": False,
        "sampling_weight_status": "not_identifiable",
        "sampling_weight_reason": _WEIGHT_REASON,
        "sampling_design": "deterministic_balanced_selection",
    }


def compile_r2_dataset(
    adjudication_root: Path,
    outcomes_root: Path,
    annotator_root: Path,
    coordinator_root: Path,
    manifest_splits: Path,
    manifest_lock: Path,
    hmac_key: Path,
    output_dir: Path,
) -> Path:
    """Publish split causal labels, future outcomes, and a private joined table."""
    adjudication_root, outcomes_root, annotator_root, coordinator_root = map(
        Path, (adjudication_root, outcomes_root, annotator_root, coordinator_root)
    )
    manifest_splits, manifest_lock, output_dir = map(Path, (manifest_splits, manifest_lock, output_dir))
    require_disjoint_output(
        output_dir, [adjudication_root, outcomes_root, annotator_root, coordinator_root]
    )
    key = read_hmac_key(hmac_key)

    assignment_payload, _, coordinator_outputs = _exact_assignment_packages(
        annotator_root, coordinator_root, key
    )
    assignment_receipt = _json(assignment_payload, "assignment receipt")
    protocol = assignment_receipt.get("protocol_version")
    if not isinstance(protocol, str) or not protocol:
        raise ValueError("assignment receipt requires protocol_version")
    if assignment_receipt.get("capabilities") != ["sample-grid-outcomes/v1"]:
        raise ValueError("assignment package lacks authenticated sample-grid outcome support")
    assignment_rows, _, private_payload, reveal_payload = _load_assignment_map(
        coordinator_root,
        coordinator_outputs,
        {
            "annotation": {
                "pilot_splits": list(_PILOT_SPLITS),
                "pilot_review_range_episodes": [60, 90],
                "pilot_min_lineages": 30,
            },
            "silence": {"max_episodes_per_lineage": 5},
        },
    )

    adjudication_payload, adjudication, adjudication_outputs = _exact_authenticated_package(
        adjudication_root,
        receipt_name="adjudication_receipt.json",
        receipt_schema=_ADJUDICATION_SCHEMA,
        key=key,
        where="adjudication package",
    )
    outcomes_receipt_payload, outcomes_receipt, outcome_outputs = _exact_authenticated_package(
        outcomes_root,
        receipt_name="outcomes_receipt.json",
        receipt_schema=_OUTCOME_RECEIPT_SCHEMA,
        key=key,
        where="outcomes package",
    )
    if adjudication.get("protocol_version") != protocol or outcomes_receipt.get("protocol_version") != protocol:
        raise ValueError("receipt protocols do not match assignments")

    adjudication_hashes = adjudication.get("sha256")
    if not isinstance(adjudication_hashes, dict):
        raise ValueError("adjudication receipt requires its agreement chain")
    for name in ("agreement_receipt", "validated_annotations", "assignment_receipt", "config"):
        require_digest(adjudication_hashes.get(name), f"adjudication receipt sha256.{name}")
    if adjudication_hashes["assignment_receipt"] != _sha256(assignment_payload):
        raise ValueError("adjudication receipt is not chained to these assignments")
    assignment_hashes = assignment_receipt.get("sha256")
    if (
        not isinstance(assignment_hashes, dict)
        or adjudication_hashes["config"] != assignment_hashes.get("config")
    ):
        raise ValueError("adjudication and assignments do not share the unchanged protocol config")
    if adjudication.get("majority_unratable_disposition") != "excluded":
        raise ValueError("adjudication package has an invalid unratable disposition")

    labels_payload = (adjudication_root / "adjudicated.jsonl").read_bytes()
    excluded_payload = (adjudication_root / "excluded_majority_unratable.jsonl").read_bytes()
    if adjudication_outputs.get("adjudicated.jsonl") != _sha256(labels_payload) or adjudication_outputs.get(
        "excluded_majority_unratable.jsonl"
    ) != _sha256(excluded_payload):
        raise ValueError("adjudication outputs do not match receipt")
    labels_rows = _optional_jsonl(labels_payload, "adjudicated labels")
    excluded_rows = _optional_jsonl(excluded_payload, "excluded adjudication items")

    outcome_payload = (outcomes_root / "outcomes.jsonl").read_bytes()
    if outcome_outputs != {"outcomes.jsonl": _sha256(outcome_payload)}:
        raise ValueError("outcomes receipt output set is invalid")
    outcome_hashes = outcomes_receipt.get("sha256")
    if not isinstance(outcome_hashes, dict) or outcome_hashes.get("assignment_receipt") != _sha256(assignment_payload):
        raise ValueError("outcomes receipt is not chained to these assignments")
    if outcome_hashes.get("config") != assignment_hashes.get("config"):
        raise ValueError("outcomes and assignments do not share the unchanged protocol config")
    if (
        outcome_hashes.get("private_assignment_source_map") != _sha256(private_payload)
        or outcome_hashes.get("coordinator_reveal_release_map") != _sha256(reveal_payload)
    ):
        raise ValueError("outcomes receipt is not chained to the private assignment maps")
    if outcomes_receipt.get("recorded_partner_entry_role") != "outcome_only_not_normative_supervision":
        raise ValueError("outcomes receipt must keep recorded partner entry observational")
    lock_payload, splits_payload = manifest_lock.read_bytes(), manifest_splits.read_bytes()
    if (
        outcome_hashes.get("manifest_lock") != _sha256(lock_payload)
        or assignment_hashes.get("manifest_lock") != _sha256(lock_payload)
    ):
        raise ValueError("assignments and outcomes are not chained to this manifest lock")
    manifest_receipt = _json(lock_payload, "manifest lock receipt")
    verify_receipt(manifest_receipt, key, "manifest lock receipt")
    if manifest_receipt.get("schema") != "semantic-silence-config/v1" or manifest_receipt.get("protocol_version") != protocol:
        raise ValueError("manifest lock schema or protocol does not match dataset inputs")
    manifest_hashes = manifest_receipt.get("sha256")
    if not isinstance(manifest_hashes, dict) or not hmac.compare_digest(
        require_digest(manifest_hashes.get("splits"), "manifest lock sha256.splits"),
        _sha256(splits_payload),
    ):
        raise ValueError("manifest splits do not match manifest lock")
    splits = _load_splits(splits_payload, protocol)
    manifest_counts = manifest_receipt.get("counts")
    if (
        not isinstance(manifest_counts, dict)
        or manifest_counts.get("input_lineages") != len(splits)
        or manifest_counts.get("split_lineages") != dict(sorted(Counter(splits.values()).items()))
    ):
        raise ValueError("manifest splits do not match manifest lock counts")

    item_metadata: dict[str, dict[str, Any]] = {}
    session_bindings: dict[str, tuple[str, str, str]] = {}
    lineage_bindings: dict[str, tuple[str, str]] = {}
    for row in assignment_rows:
        if row["repeat"]:
            continue
        item = row["item_id"]
        metadata = {field: row[field] for field in (
            "item_id", "session_id", "episode_id", "source", "lineage_id", "split",
            "onset_s", "elapsed_s", "model_cutoff_s", "sample_rate",
        )}
        previous = item_metadata.setdefault(item, metadata)
        if previous != metadata:
            raise ValueError(f"item {item!r} has mixed assignment identity")
        if splits.get(row["lineage_id"]) != row["split"]:
            raise ValueError(f"lineage {row['lineage_id']!r} moved from its locked split")
        if row["split"] in _EXCLUDED_SPLITS or row["split"] not in _PILOT_SPLITS:
            raise ValueError("test or challenge data entered pilot compilation")
        session_scope = (row["lineage_id"], row["source"], row["split"])
        if session_bindings.setdefault(row["session_id"], session_scope) != session_scope:
            raise ValueError("a session has mixed lineage, source, or split bindings")
        lineage_scope = (row["source"], row["split"])
        if lineage_bindings.setdefault(row["lineage_id"], lineage_scope) != lineage_scope:
            raise ValueError("a lineage has mixed source or split bindings")

    labels: dict[str, dict[str, Any]] = {}
    for number, row in enumerate(labels_rows, 1):
        where = f"adjudicated label row {number}"
        if set(row) != _LABEL_FIELDS or row.get("schema") != _LABEL_SCHEMA or row.get("protocol_version") != protocol:
            raise ValueError(f"{where} has invalid fields, schema, or protocol")
        item = row.get("item_id")
        if not isinstance(item, str) or item not in item_metadata or item in labels:
            raise ValueError(f"{where} has unknown or duplicate item_id")
        metadata = item_metadata[item]
        for field in ("source", "lineage_id", "split"):
            if row.get(field) != metadata[field]:
                raise ValueError(f"{where} has a mixed {field} binding")
        for field in ("elapsed_s", "model_cutoff_s"):
            if _finite(row.get(field), f"{where} {field}") != float(metadata[field]):
                raise ValueError(f"{where} has inconsistent {field}")
        if row.get("action") not in _ACTIONS or not isinstance(row.get("onset_guidance"), dict):
            raise ValueError(f"{where} has an invalid action or onset guidance")
        labels[item] = row

    excluded: set[str] = set()
    for number, row in enumerate(excluded_rows, 1):
        if set(row) != {"item_id", "disposition"} or row.get("disposition") != "excluded_majority_unratable":
            raise ValueError(f"excluded adjudication row {number} is invalid")
        item = row.get("item_id")
        if not isinstance(item, str) or item not in item_metadata or item in excluded or item in labels:
            raise ValueError(f"excluded adjudication row {number} has unknown or duplicate item_id")
        excluded.add(item)
    counts = adjudication.get("counts")
    if not isinstance(counts, dict) or counts.get("input_items") != len(item_metadata) or counts.get("items") != len(labels):
        raise ValueError("adjudication counts do not match assignment coverage")
    if set(item_metadata) != set(labels) | excluded:
        raise ValueError("adjudication labels and exclusions do not exactly cover assigned items")

    outcomes: dict[str, dict[str, Any]] = {}
    for number, row in enumerate(_jsonl(outcome_payload, "observed outcomes"), 1):
        where = f"observed outcome row {number}"
        if set(row) != _OUTCOME_FIELDS or row.get("schema") != _OUTCOME_SCHEMA or row.get("protocol_version") != protocol:
            raise ValueError(f"{where} has invalid fields, schema, or protocol")
        item = row.get("item_id")
        if not isinstance(item, str) or item not in item_metadata or item in outcomes:
            raise ValueError(f"{where} has unknown or duplicate item_id")
        metadata = item_metadata[item]
        for field in ("session_id", "episode_id", "source", "lineage_id", "split", "sample_rate"):
            if row.get(field) != metadata[field]:
                raise ValueError(f"{where} has a mixed {field} binding")
        for field in ("onset_s", "model_cutoff_s", "checkpoint_elapsed_s"):
            expected_field = "elapsed_s" if field == "checkpoint_elapsed_s" else field
            if _finite(row.get(field), f"{where} {field}") != float(metadata[expected_field]):
                raise ValueError(f"{where} has inconsistent {field}")
        cutoff = float(row["model_cutoff_s"])
        duration = _finite(row.get("observed_duration_s"), f"{where} observed_duration_s")
        censor = _finite(row.get("administrative_censor_s"), f"{where} administrative_censor_s")
        overlap = _finite(row.get("overlap_duration_s"), f"{where} overlap_duration_s")
        window_duration = censor - cutoff
        if row.get("observation_horizon_s") != 8.0 or not 0 < duration <= window_duration <= 8.0:
            raise ValueError(f"{where} exceeds its locked observation boundary")
        if not 0 <= overlap <= window_duration:
            raise ValueError(f"{where} has invalid overlap duration")
        for prefix in ("user_resume", "recorded_partner_entry"):
            event_time, delay, observed = row.get(f"{prefix}_s"), row.get(f"{prefix}_delay_s"), row.get(f"{prefix}_observed")
            if not isinstance(observed, bool) or (event_time is None) != (not observed) or (delay is None) != (not observed):
                raise ValueError(f"{where} has inconsistent {prefix} indicators")
            if observed and (
                not cutoff < _finite(event_time, f"{where} {prefix}_s") <= censor
                or abs(float(event_time) - cutoff - _finite(delay, f"{where} {prefix}_delay_s")) > 1e-9
            ):
                raise ValueError(f"{where} has a future event outside the observed boundary")
        user_time, partner_time = row["user_resume_s"], row["recorded_partner_entry_s"]
        expected_event, expected_time = first_observed_event(
            user_time, partner_time, row["sample_rate"]
        )
        if row.get("first_event") != expected_event or row.get("first_event_s") != expected_time:
            raise ValueError(f"{where} violates the locked first-event or tie rule")
        expected_duration = (expected_time if expected_time is not None else censor) - cutoff
        if abs(duration - expected_duration) > 1e-9:
            raise ValueError(f"{where} has inconsistent observed duration")
        flags = {
            "user_resume_event": expected_event == "user_resume",
            "recorded_partner_entry_event": expected_event == "recorded_partner_entry",
            "administrative_censor_event": expected_event is None,
            "right_censored": expected_event is None,
        }
        if any(not isinstance(row.get(name), bool) or row[name] is not value for name, value in flags.items()):
            raise ValueError(f"{where} has inconsistent event indicators")
        if not isinstance(row.get("recorded_overlap_at_user_resume"), bool):
            raise ValueError(f"{where} requires a boolean overlap-at-resumption indicator")
        outcomes[item] = row
    if set(outcomes) != set(item_metadata):
        raise ValueError("outcomes must exactly cover every assigned item")
    outcome_counts = outcomes_receipt.get("counts")
    if (
        not isinstance(outcome_counts, dict)
        or outcome_counts.get("items") != len(outcomes)
        or outcome_counts.get("lineages") != len({row["lineage_id"] for row in outcomes.values()})
    ):
        raise ValueError("outcomes receipt counts do not match outcome rows")

    causal_by_split: dict[str, list[dict[str, Any]]] = {split: [] for split in _PILOT_SPLITS}
    outcomes_by_split: dict[str, list[dict[str, Any]]] = {split: [] for split in _PILOT_SPLITS}
    joined_by_split: dict[str, list[dict[str, Any]]] = {split: [] for split in _PILOT_SPLITS}
    for item in sorted(outcomes):
        outcomes_by_split[outcomes[item]["split"]].append(outcomes[item])
    for item in sorted(labels):
        label, metadata, outcome = labels[item], item_metadata[item], outcomes[item]
        sampling = _sampling_fields()
        causal = {
            "schema": _CAUSAL_SCHEMA,
            "protocol_version": protocol,
            "item_id": item,
            "source": label["source"],
            "lineage_id": label["lineage_id"],
            "split": label["split"],
            "action": label["action"],
            "action_distribution": label["action_distribution"],
            "confidence": label["confidence"],
            "tags": label["tags"],
            "quality": label["quality"],
            "elapsed_s": label["elapsed_s"],
            "model_cutoff_s": label["model_cutoff_s"],
            "onset_guidance": label["onset_guidance"],
            **sampling,
        }
        if {"session_id", "episode_id", "annotator_id"} & causal.keys():
            raise AssertionError("causal model labels contain raw private identities")
        outcome_values = {
            key: value for key, value in outcome.items()
            if key not in {"schema", "protocol_version", "item_id", "session_id", "episode_id", "source", "lineage_id", "split"}
        }
        joined = {
            "schema": "semantic-silence-r2-example/v1",
            "protocol_version": protocol,
            "item_id": item,
            "session_id": metadata["session_id"],
            "source": metadata["source"],
            "lineage_id": metadata["lineage_id"],
            "split": metadata["split"],
            "action": label["action"],
            "action_distribution": label["action_distribution"],
            "confidence": label["confidence"],
            "tags": label["tags"],
            "quality": label["quality"],
            "elapsed_s": label["elapsed_s"],
            "model_cutoff_s": label["model_cutoff_s"],
            "onset_guidance": label["onset_guidance"],
            "outcome": outcome_values,
            **sampling,
        }
        causal_by_split[label["split"]].append(causal)
        joined_by_split[label["split"]].append(joined)

    files: dict[str, bytes] = {}
    for split in _PILOT_SPLITS:
        files[f"causal_labels/{split}.jsonl"] = _lines(causal_by_split[split]) if causal_by_split[split] else b""
        files[f"future_outcomes/{split}.jsonl"] = _lines(outcomes_by_split[split]) if outcomes_by_split[split] else b""
        files[f"research_joined/{split}.jsonl"] = _lines(joined_by_split[split]) if joined_by_split[split] else b""
    output_hashes = {name: _sha256(payload) for name, payload in sorted(files.items())}
    receipt = sign_receipt({
        "schema": _DATASET_RECEIPT_SCHEMA,
        "protocol_version": protocol,
        "splits": list(_PILOT_SPLITS),
        "excluded_splits": sorted(_EXCLUDED_SPLITS),
        "sampling_weight_identified": False,
        "sampling_weight_reason": _WEIGHT_REASON,
        "limitations": [
            "pilot prevalence is not recoverable",
            "population calibration is not recoverable",
            "population hazard incidence is not recoverable",
            "recorded partner entry is outcome metadata, not normative supervision",
        ],
        "counts": {
            "assigned_items": len(item_metadata),
            "causal_labels": len(labels),
            "future_outcomes": len(outcomes),
            "excluded_majority_unratable": len(excluded),
            "by_split": {
                split: {
                    "causal_labels": len(causal_by_split[split]),
                    "future_outcomes": len(outcomes_by_split[split]),
                }
                for split in _PILOT_SPLITS
            },
        },
        "sha256": {
            "adjudication_receipt": _sha256(adjudication_payload),
            "adjudicated": _sha256(labels_payload),
            "excluded_majority_unratable": _sha256(excluded_payload),
            "outcomes_receipt": _sha256(outcomes_receipt_payload),
            "outcomes": _sha256(outcome_payload),
            "assignment_receipt": _sha256(assignment_payload),
            "private_assignment_source_map": _sha256(private_payload),
            "coordinator_reveal_release_map": _sha256(reveal_payload),
            "manifest_splits": _sha256(splits_payload),
            "manifest_lock": _sha256(lock_payload),
        },
        "outputs": output_hashes,
    }, key)
    files["dataset_receipt.json"] = receipt_bytes(receipt)
    return _publish_packages({output_dir: files})[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adjudication-root", type=Path, required=True)
    parser.add_argument("--outcomes-root", type=Path, required=True)
    parser.add_argument("--annotator-root", type=Path, required=True)
    parser.add_argument("--coordinator-root", type=Path, required=True)
    parser.add_argument("--manifest-splits", type=Path, required=True)
    parser.add_argument("--manifest-lock", type=Path, required=True)
    parser.add_argument("--hmac-key", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    compile_r2_dataset(
        args.adjudication_root, args.outcomes_root, args.annotator_root,
        args.coordinator_root, args.manifest_splits, args.manifest_lock,
        args.hmac_key, args.output_dir,
    )


if __name__ == "__main__":
    main()
