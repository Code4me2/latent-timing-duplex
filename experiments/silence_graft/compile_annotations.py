#!/usr/bin/env python3
"""Compile, report, and adjudicate locked semantic-silence pilot responses."""

from __future__ import annotations

import argparse
import math
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from latent_timing_duplex.silence.agreement import (
    assigned_agreement_report,
    lineage_bootstrap_ci,
    nominal_krippendorff_alpha,
)
from latent_timing_duplex.silence.assigned_annotations import (
    load_assigned_annotations,
    parse_assigned_annotation,
)
from latent_timing_duplex.silence.types import AssignedSilenceAnnotation, SilenceAction

try:
    from experiments.silence_graft.build_assignments import (
        _publish_packages,
        _roots_overlap,
        _sha256,
        _verify_output_map,
    )
    from experiments.silence_graft.protocol_v1 import load_protocol_v1, parse_protocol_v1
    from experiments.silence_graft.receipts import (
        read_hmac_key,
        receipt_bytes,
        require_digest,
        sign_receipt,
        verify_receipt,
    )
    from experiments.silence_graft.render_clips import (
        _json,
        _jsonl,
        _lines,
        _require_ready,
    )
except ModuleNotFoundError:  # Direct script execution from the repository root.
    from build_assignments import _publish_packages, _roots_overlap, _sha256, _verify_output_map
    from protocol_v1 import load_protocol_v1, parse_protocol_v1
    from receipts import read_hmac_key, receipt_bytes, require_digest, sign_receipt, verify_receipt
    from render_clips import _json, _jsonl, _lines, _require_ready

_ASSIGNMENT_RECEIPT_SCHEMA = "semantic-silence-assignment-receipt/v1"
_COMPLETION_RECEIPT_SCHEMA = "semantic-silence-prefix-completion-receipt/v1"
_VALIDATED_SCHEMA = "semantic-silence-validated-private-annotation/v1"
_AGREEMENT_REPORT_SCHEMA = "semantic-silence-pilot-agreement-report/v1"
_AGREEMENT_RECEIPT_SCHEMA = "semantic-silence-pilot-agreement-receipt/v1"
_EXPERT_RECEIPT_SCHEMA = "semantic-silence-expert-decisions-receipt/v1"
_LABEL_SCHEMA = "semantic-silence-causal-model-label/v1"
_ADJUDICATION_RECEIPT_SCHEMA = "semantic-silence-adjudication-receipt/v1"
_RATABLE = {action.value for action in SilenceAction if action is not SilenceAction.UNRATABLE}
_ACTIONS = tuple(action.value for action in SilenceAction)
_VALIDATED_FIELDS = {
    "schema", "protocol_version", "assignment_id", "presentation_id", "annotator_id",
    "item_id", "repeat", "source", "lineage_id", "split", "onset_s", "elapsed_s",
    "model_cutoff_s", "prefix_sha256", "annotation_sha256", "action", "confidence",
    "also_acceptable", "tags", "acceptable_onset", "quality", "ratable",
}


def _exact_assignment_packages(
    annotator_root: Path, coordinator_root: Path, key: bytes
) -> tuple[bytes, dict[str, str], dict[str, str]]:
    if _roots_overlap(annotator_root, coordinator_root):
        raise ValueError("annotator and coordinator assignment roots must be separate")
    _require_ready(annotator_root, "annotator assignment package")
    _require_ready(coordinator_root, "coordinator assignment package")
    receipt_path = coordinator_root / "assignment_receipt.json"
    payload = receipt_path.read_bytes()
    receipt = _json(payload, "assignment receipt")
    verify_receipt(receipt, key, "assignment receipt")
    if receipt.get("schema") != _ASSIGNMENT_RECEIPT_SCHEMA:
        raise ValueError("invalid assignment receipt schema")
    annotator_outputs = _verify_output_map(
        annotator_root, receipt.get("annotator_outputs"), "assignment receipt annotator package"
    )
    coordinator_outputs = _verify_output_map(
        coordinator_root, receipt.get("outputs"), "assignment receipt coordinator package"
    )
    for root, outputs, extras, where in (
        (annotator_root, annotator_outputs, {"READY"}, "annotator assignment package"),
        (
            coordinator_root,
            coordinator_outputs,
            {"assignment_receipt.json", "READY"},
            "coordinator assignment package",
        ),
    ):
        if any(path.is_symlink() for path in root.rglob("*")):
            raise ValueError(f"{where} must not contain symlinks")
        actual = {str(path.relative_to(root)) for path in root.rglob("*") if path.is_file()}
        if actual != set(outputs) | extras:
            raise ValueError(f"{where} contains unrecorded or missing files")
    return payload, annotator_outputs, coordinator_outputs


def _config(payload: bytes, path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    cfg = parse_protocol_v1(payload, path)
    return cfg, cfg["annotation"]


def _finite(row: Mapping[str, Any], field: str, where: str, *, nonnegative: bool = True) -> float:
    value = row.get(field)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{where} requires finite {field}")
    result = float(value)
    if nonnegative and result < 0:
        raise ValueError(f"{where} requires non-negative {field}")
    return result


def _load_assignment_map(
    coordinator_root: Path,
    outputs: Mapping[str, str],
    cfg: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], bytes, bytes]:
    private_name, reveal_name = "private_assignment_source_map.jsonl", "coordinator_reveal_release_map.jsonl"
    private_payload = (coordinator_root / private_name).read_bytes()
    reveal_payload = (coordinator_root / reveal_name).read_bytes()
    if outputs.get(private_name) != _sha256(private_payload) or outputs.get(reveal_name) != _sha256(reveal_payload):
        raise ValueError("private assignment maps are not locked by assignment receipt")
    rows = _jsonl(private_payload, "private assignment source map")
    reveal_rows = _jsonl(reveal_payload, "coordinator reveal-release map")
    reveals: dict[str, dict[str, Any]] = {}
    presentations: set[str] = set()
    for number, row in enumerate(reveal_rows, 1):
        assignment = row.get("assignment_id")
        presentation = row.get("presentation_id")
        if not isinstance(assignment, str) or not assignment or assignment in reveals:
            raise ValueError(f"invalid or duplicate reveal assignment at row {number}")
        if not isinstance(presentation, str) or not presentation or presentation in presentations:
            raise ValueError(f"invalid or duplicate presentation at row {number}")
        require_digest(row.get("prefix_sha256"), f"reveal row {number} prefix_sha256")
        presentations.add(presentation)
        reveals[assignment] = row

    assignments: set[str] = set()
    map_presentations: set[str] = set()
    originals: dict[str, list[dict[str, Any]]] = {}
    original_keys: set[tuple[str, str]] = set()
    repeat_keys: list[tuple[str, str]] = []
    item_metadata: dict[str, tuple[Any, ...]] = {}
    lineage_bindings: dict[str, tuple[str, str]] = {}
    session_bindings: dict[str, tuple[str, str, str]] = {}
    pilot_splits = set(cfg["annotation"].get("pilot_splits", ()))
    for number, row in enumerate(rows, 1):
        where = f"private assignment map row {number}"
        assignment, presentation = row.get("assignment_id"), row.get("presentation_id")
        annotator, item = row.get("annotator_id"), row.get("item_id")
        if not isinstance(assignment, str) or not assignment or assignment in assignments:
            raise ValueError(f"{where} has invalid or duplicate assignment_id")
        if not isinstance(presentation, str) or not presentation or presentation in map_presentations:
            raise ValueError(f"{where} has invalid or duplicate presentation_id")
        if any(not isinstance(value, str) or not value for value in (annotator, item)):
            raise ValueError(f"{where} requires annotator_id and item_id")
        if not isinstance(row.get("repeat"), bool):
            raise ValueError(f"{where} repeat must be a boolean")
        for field in ("source", "lineage_id", "session_id", "split"):
            if not isinstance(row.get(field), str) or not row[field]:
                raise ValueError(f"{where} requires {field}")
        onset = _finite(row, "onset_s", where)
        elapsed = _finite(row, "elapsed_s", where)
        checkpoint = _finite(row, "checkpoint_elapsed_s", where)
        cutoff = _finite(row, "model_cutoff_s", where)
        if abs(elapsed - checkpoint) > 1e-9 or abs(cutoff - onset - elapsed) > 1e-6:
            raise ValueError(f"{where} has inconsistent onset/elapsed/model cutoff")
        if row["split"] not in pilot_splits:
            raise ValueError(f"{where} uses an excluded split")
        binding = (row["source"], row["split"])
        previous_binding = lineage_bindings.setdefault(row["lineage_id"], binding)
        if previous_binding != binding:
            raise ValueError(f"lineage {row['lineage_id']!r} has inconsistent source or overlaps splits")
        session_binding = (row["source"], row["lineage_id"], row["split"])
        if session_bindings.setdefault(row["session_id"], session_binding) != session_binding:
            raise ValueError(f"session {row['session_id']!r} has inconsistent source, lineage, or split")
        metadata = (row["source"], row["lineage_id"], row["session_id"], row["split"], onset, elapsed, cutoff)
        previous = item_metadata.setdefault(item, metadata)
        if previous != metadata:
            raise ValueError(f"item {item!r} has inconsistent private metadata")
        reveal = reveals.get(assignment)
        if reveal is None or any(
            reveal.get(field) != row.get(field)
            for field in ("assignment_id", "presentation_id", "annotator_id", "item_id", "repeat")
        ):
            raise ValueError(f"{where} does not match reveal-release map")
        key = (annotator, item)
        if row["repeat"]:
            repeat_keys.append(key)
        else:
            if key in original_keys:
                raise ValueError(f"item {item!r} has duplicate original annotator")
            original_keys.add(key)
            originals.setdefault(item, []).append(row)
        assignments.add(assignment)
        map_presentations.add(presentation)
    if assignments != set(reveals):
        raise ValueError("private assignment and reveal-release maps differ")
    for item, group in originals.items():
        if len(group) != 3 or len({row["annotator_id"] for row in group}) != 3:
            raise ValueError(f"item {item!r} does not have exactly 3 distinct original annotators")
    minimum_items, maximum_items = cfg["annotation"]["pilot_review_range_episodes"]
    if not minimum_items <= len(originals) <= maximum_items:
        raise ValueError(
            f"production pilot requires {minimum_items}–{maximum_items} items; found {len(originals)}"
        )
    items_by_lineage = Counter(group[0]["lineage_id"] for group in originals.values())
    if len(items_by_lineage) < cfg["annotation"]["pilot_min_lineages"]:
        raise ValueError("production pilot does not meet the lineage minimum")
    if any(count > cfg["silence"]["max_episodes_per_lineage"] for count in items_by_lineage.values()):
        raise ValueError("production pilot exceeds the per-lineage item cap")
    if any(key not in original_keys for key in repeat_keys):
        raise ValueError("repeat assignment has no matching original annotator/item")
    return rows, reveals, private_payload, reveal_payload


def _onset(annotation: AssignedSilenceAnnotation) -> dict[str, Any] | None:
    value = annotation.acceptable_onset
    return None if value is None else {
        "earliest_elapsed_s": value.earliest_elapsed_s,
        "latest_elapsed_s": value.latest_elapsed_s,
        "wait_for_user_resume": value.wait_for_user_resume,
    }


def _validated_row(
    annotation: AssignedSilenceAnnotation, metadata: Mapping[str, Any], prefix_sha256: str
) -> dict[str, Any]:
    return {
        "schema": _VALIDATED_SCHEMA,
        "protocol_version": annotation.protocol_version,
        "assignment_id": annotation.assignment_id,
        "presentation_id": annotation.presentation_id,
        "annotator_id": annotation.annotator_id,
        "item_id": metadata["item_id"],
        "repeat": metadata["repeat"],
        "source": metadata["source"],
        "lineage_id": metadata["lineage_id"],
        "split": metadata["split"],
        "onset_s": metadata["onset_s"],
        "elapsed_s": metadata["elapsed_s"],
        "model_cutoff_s": metadata["model_cutoff_s"],
        "prefix_sha256": prefix_sha256,
        "annotation_sha256": annotation.annotation_sha256,
        "action": annotation.action.value,
        "confidence": annotation.confidence,
        "also_acceptable": [value.value for value in annotation.also_acceptable],
        "tags": list(annotation.tags),
        "acceptable_onset": _onset(annotation),
        "quality": annotation.quality.value,
        "ratable": annotation.ratable,
    }


def report(
    annotator_root: Path,
    coordinator_root: Path,
    raw_responses: Path,
    hmac_key: Path,
    config: str | Path,
    output_dir: Path,
    *,
    bootstrap_replicates: int | None = None,
) -> Path:
    """Validate every assigned response and publish completion/agreement artifacts."""
    annotator_root, coordinator_root, raw_responses, output_dir = map(
        Path, (annotator_root, coordinator_root, raw_responses, output_dir)
    )
    if any(_roots_overlap(output_dir, root) for root in (annotator_root, coordinator_root)):
        raise ValueError("report output must be separate from assignment packages")
    key = read_hmac_key(hmac_key)
    assignment_payload, annotator_outputs, coordinator_outputs = _exact_assignment_packages(
        annotator_root, coordinator_root, key
    )
    config_path, config_payload, cfg = load_protocol_v1(config)
    annotation_cfg = cfg["annotation"]
    assignment_receipt = _json(assignment_payload, "assignment receipt")
    if assignment_receipt.get("protocol_version") != cfg.get("protocol_version"):
        raise ValueError("assignment receipt protocol does not match config")
    assignment_hashes = assignment_receipt.get("sha256")
    if not isinstance(assignment_hashes, dict) or assignment_hashes.get("config") != _sha256(config_payload):
        raise ValueError("config SHA-256 does not match assignment receipt")
    assignment_rows, reveals, private_payload, reveal_payload = _load_assignment_map(
        coordinator_root, coordinator_outputs, cfg
    )
    raw_payload = raw_responses.read_bytes()
    counts = assignment_receipt.get("counts")
    if not isinstance(counts, dict) or counts.get("prefix_assignments") != len(assignment_rows) or counts.get("items") != len({row["item_id"] for row in assignment_rows}):
        raise ValueError("assignment receipt counts do not match private assignment map")
    annotations = load_assigned_annotations(raw_payload)
    metadata = {row["assignment_id"]: row for row in assignment_rows}
    if {row.assignment_id for row in annotations} != set(metadata):
        missing = set(metadata) - {row.assignment_id for row in annotations}
        unexpected = {row.assignment_id for row in annotations} - set(metadata)
        raise ValueError(f"raw responses have {len(missing)} missing and {len(unexpected)} unexpected assignments")
    for row in annotations:
        expected = metadata[row.assignment_id]
        if row.presentation_id != expected["presentation_id"]:
            raise ValueError(f"assignment {row.assignment_id!r} has mismatched presentation_id")
        if row.annotator_id != expected["annotator_id"]:
            raise ValueError(f"assignment {row.assignment_id!r} has mismatched annotator_id")

    validated = [
        _validated_row(row, metadata[row.assignment_id], reveals[row.assignment_id]["prefix_sha256"])
        for row in annotations
    ]
    validated_payload = _lines(validated)
    completions = [
        {
            "assignment_id": row.assignment_id,
            "presentation_id": row.presentation_id,
            "annotator_id": row.annotator_id,
            "item_id": metadata[row.assignment_id]["item_id"],
            "prefix_sha256": reveals[row.assignment_id]["prefix_sha256"],
            "annotation_sha256": row.annotation_sha256,
        }
        for row in annotations
    ]
    completion_payload = _lines(completions)
    completion_receipt = sign_receipt(
        {
            "schema": _COMPLETION_RECEIPT_SCHEMA,
            "protocol_version": cfg["protocol_version"],
            "sha256": {
                "assignment_receipt": _sha256(assignment_payload),
                "completions": _sha256(completion_payload),
            },
        },
        key,
    )
    completion_receipt_payload = receipt_bytes(completion_receipt)
    bootstrap = annotation_cfg["bootstrap"]
    confirmatory = bootstrap_replicates is None
    n_boot = bootstrap["replicates"] if confirmatory else bootstrap_replicates
    if isinstance(n_boot, bool) or not isinstance(n_boot, int) or n_boot < 1:
        raise ValueError("bootstrap_replicates must be a positive integer")
    metrics = assigned_agreement_report(
        annotations,
        assignment_rows,
        boundaries_s=cfg["silence"]["duration_strata_s"],
        stratum_labels=cfg["silence"]["duration_strata_labels"],
        n_boot=n_boot,
        seed=bootstrap["seed"],
    )
    metrics["bootstrap"]["confirmatory"] = confirmatory
    metrics["bootstrap"]["nonconfirmatory_override"] = not confirmatory
    metrics["bootstrap"]["configured_replicates"] = bootstrap["replicates"]
    alpha = metrics["overall"]["nominal_krippendorff_alpha"]
    threshold = float(annotation_cfg["agreement_gate"])
    original_rows = [row for row in validated if not row["repeat"]]
    by_item: dict[str, list[dict[str, Any]]] = {}
    for row in original_rows:
        by_item.setdefault(row["item_id"], []).append(row)
    pairable = {
        item for item, rows in by_item.items() if sum(row["ratable"] for row in rows) >= 2
    }
    pairable_lineages = {by_item[item][0]["lineage_id"] for item in pairable}
    pairable_fraction = len(pairable) / len(by_item)
    alpha_bootstrap = metrics["overall"]["lineage_bootstrap_95_ci"].get(
        "nominal_krippendorff_alpha", {}
    )
    valid_replicates = alpha_bootstrap.get("valid_replicates", 0)
    valid_fraction = valid_replicates / n_boot
    coverage = {
        "items": len(by_item),
        "allowed_item_range": annotation_cfg["pilot_review_range_episodes"],
        "lineages": len({row["lineage_id"] for row in original_rows}),
        "required_lineages": annotation_cfg["pilot_min_lineages"],
        "max_items_per_lineage": max(Counter(rows[0]["lineage_id"] for rows in by_item.values()).values()),
        "allowed_max_items_per_lineage": cfg["silence"]["max_episodes_per_lineage"],
        "pairable_items": len(pairable),
        "pairable_fraction": pairable_fraction,
        "required_pairable_fraction": annotation_cfg["pilot_min_pairable_fraction"],
        "pairable_lineages": len(pairable_lineages),
        "required_pairable_lineages": annotation_cfg["pilot_min_pairable_lineages"],
        "valid_alpha_bootstrap_replicates": valid_replicates,
        "requested_alpha_bootstrap_replicates": n_boot,
        "valid_alpha_bootstrap_fraction": valid_fraction,
        "required_valid_alpha_bootstrap_fraction": bootstrap["min_valid_fraction"],
    }
    coverage_passed = (
        coverage["allowed_item_range"][0] <= coverage["items"] <= coverage["allowed_item_range"][1]
        and coverage["lineages"] >= coverage["required_lineages"]
        and coverage["max_items_per_lineage"] <= coverage["allowed_max_items_per_lineage"]
        and pairable_fraction >= coverage["required_pairable_fraction"]
        and len(pairable_lineages) >= coverage["required_pairable_lineages"]
        and valid_fraction >= coverage["required_valid_alpha_bootstrap_fraction"]
    )
    coverage["passed"] = coverage_passed
    gate_passed = confirmatory and coverage_passed and alpha is not None and alpha >= threshold
    decision = (
        "nonconfirmatory"
        if not confirmatory
        else "retain_original_taxonomy" if gate_passed else "new_collapsed_pilot_required"
    )
    input_hashes = {
        "assignment_receipt": _sha256(assignment_payload),
        "config": _sha256(config_payload),
        "private_assignment_source_map": _sha256(private_payload),
        "coordinator_reveal_release_map": _sha256(reveal_payload),
        "raw_responses": _sha256(raw_payload),
    }
    output_hashes = {
        "validated_annotations.jsonl": _sha256(validated_payload),
        "completions.jsonl": _sha256(completion_payload),
        "completion_receipt.json": _sha256(completion_receipt_payload),
    }
    agreement_report = sign_receipt({
        "schema": _AGREEMENT_REPORT_SCHEMA,
        "protocol_version": cfg["protocol_version"],
        "counts": {
            "assignments": len(annotations),
            "items": len({row["item_id"] for row in assignment_rows if not row["repeat"]}),
            "original_annotations": sum(not row["repeat"] for row in assignment_rows),
            "repeat_annotations": sum(row["repeat"] for row in assignment_rows),
            "pairable_items": len(pairable),
            "pairable_lineages": len(pairable_lineages),
        },
        **metrics,
        "coverage": coverage,
        "taxonomy": annotation_cfg["primary_actions"],
        "threshold": threshold,
        "gate_passed": gate_passed,
        "collapse_decision": decision,
        "collapse_map": annotation_cfg["collapse_map"],
        "collapsed_reliability_requires_new_pilot": True,
        "collapse_statement": (
            "Nonconfirmatory bootstrap override; no gate or taxonomy decision is permitted."
            if not confirmatory
            else "Original taxonomy passed; adjudication may proceed."
            if gate_passed
            else "Agreement or coverage gate failed: a new collapsed pilot is required; never relabel the current pilot."
        ),
        "sha256": input_hashes,
        "outputs": output_hashes,
        "assignment_annotator_outputs": annotator_outputs,
        "assignment_coordinator_outputs": coordinator_outputs,
    }, key)
    report_payload = receipt_bytes(agreement_report)
    all_outputs = {**output_hashes, "agreement_report.json": _sha256(report_payload)}
    agreement_receipt = sign_receipt(
        {
            "schema": _AGREEMENT_RECEIPT_SCHEMA,
            "protocol_version": cfg["protocol_version"],
            "counts": agreement_report["counts"],
            **metrics,
            "coverage": coverage,
            "taxonomy": annotation_cfg["primary_actions"],
            "threshold": threshold,
            "gate_passed": gate_passed,
            "collapse_decision": decision,
            "collapse_map": annotation_cfg["collapse_map"],
            "collapsed_reliability_requires_new_pilot": True,
            "collapse_statement": agreement_report["collapse_statement"],
            "sha256": input_hashes,
            "assignment_annotator_outputs": annotator_outputs,
            "assignment_coordinator_outputs": coordinator_outputs,
            "outputs": all_outputs,
        },
        key,
    )
    files = {
        "validated_annotations.jsonl": validated_payload,
        "completions.jsonl": completion_payload,
        "completion_receipt.json": completion_receipt_payload,
        "agreement_report.json": report_payload,
        "agreement_receipt.json": receipt_bytes(agreement_receipt),
    }
    return _publish_packages({output_dir: files})[0]


def _load_validated(payload: bytes, protocol: str) -> list[dict[str, Any]]:
    rows = _jsonl(payload, "validated annotations")
    assignments: set[str] = set()
    presentations: set[str] = set()
    for number, row in enumerate(rows, 1):
        where = f"validated annotation row {number}"
        if set(row) != _VALIDATED_FIELDS:
            raise ValueError(f"{where} has missing or unknown fields")
        if row.get("schema") != _VALIDATED_SCHEMA or row.get("protocol_version") != protocol:
            raise ValueError(f"{where} has invalid schema or protocol")
        for field in ("assignment_id", "presentation_id", "annotator_id", "item_id", "source", "lineage_id", "split"):
            if not isinstance(row.get(field), str) or not row[field]:
                raise ValueError(f"{where} requires {field}")
        if row["assignment_id"] in assignments or row["presentation_id"] in presentations:
            raise ValueError(f"{where} duplicates assignment or presentation")
        assignments.add(row["assignment_id"])
        presentations.add(row["presentation_id"])
        require_digest(row.get("annotation_sha256"), f"{where} annotation_sha256")
        require_digest(row.get("prefix_sha256"), f"{where} prefix_sha256")
        if not isinstance(row.get("repeat"), bool) or not isinstance(row.get("ratable"), bool):
            raise ValueError(f"{where} requires boolean repeat and ratable")
        try:
            parse_assigned_annotation({
                "schema": "semantic-silence-prefix-response/v1",
                "protocol_version": row["protocol_version"],
                "assignment_id": row["assignment_id"],
                "presentation_id": row["presentation_id"],
                "annotator_id": row["annotator_id"],
                "action": row["action"],
                "confidence": row["confidence"],
                "also_acceptable": row["also_acceptable"],
                "tags": row["tags"],
                "acceptable_onset": row["acceptable_onset"],
                "quality": row["quality"],
                "ratable": row["ratable"],
            })
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{where} has invalid annotation: {exc}") from exc
        for field in ("onset_s", "elapsed_s", "model_cutoff_s"):
            _finite(row, field, where)
        if abs(float(row["model_cutoff_s"]) - float(row["onset_s"]) - float(row["elapsed_s"])) > 1e-6:
            raise ValueError(f"{where} has inconsistent causal times")
        onset = row["acceptable_onset"]
        if onset is not None and onset["earliest_elapsed_s"] is not None and onset["earliest_elapsed_s"] + 1e-9 < row["elapsed_s"]:
            raise ValueError(f"{where} acceptable onset precedes the checkpoint")
    return rows


def _expert_actions(
    unresolved: set[str],
    observed_actions: Mapping[str, set[str]],
    decisions_path: Path | None,
    receipt_path: Path | None,
    *,
    agreement_receipt_payload: bytes,
    validated_payload: bytes,
    protocol: str,
    key: bytes,
) -> tuple[dict[str, str], dict[str, str]]:
    if not unresolved:
        if decisions_path is not None or receipt_path is not None:
            raise ValueError("expert decisions are forbidden when no items require them")
        return {}, {}
    if decisions_path is None or receipt_path is None:
        raise ValueError("all-three-different or no-majority items require authenticated expert decisions")
    decision_payload, lock_payload = decisions_path.read_bytes(), receipt_path.read_bytes()
    lock = _json(lock_payload, "expert decisions receipt")
    verify_receipt(lock, key, "expert decisions receipt")
    if lock.get("schema") != _EXPERT_RECEIPT_SCHEMA or lock.get("protocol_version") != protocol:
        raise ValueError("invalid expert decisions receipt schema or protocol")
    hashes = lock.get("sha256")
    expected = {
        "agreement_receipt": _sha256(agreement_receipt_payload),
        "validated_annotations": _sha256(validated_payload),
        "expert_decisions": _sha256(decision_payload),
    }
    if not isinstance(hashes, dict) or hashes != expected:
        raise ValueError("expert decisions receipt hashes do not exactly match inputs")
    decisions: dict[str, str] = {}
    for number, row in enumerate(_jsonl(decision_payload, "expert decisions"), 1):
        if set(row) != {"item_id", "action"}:
            raise ValueError(f"expert decision row {number} must contain only item_id and action")
        item, action = row["item_id"], row["action"]
        if (
            not isinstance(item, str) or not item or item in decisions
            or action not in _RATABLE or action not in observed_actions.get(item, set())
        ):
            raise ValueError(f"invalid or unobserved expert decision row {number}")
        decisions[item] = action
    if set(decisions) != unresolved:
        raise ValueError("expert decisions must cover exactly every unresolved item")
    return decisions, {
        "expert_decisions": _sha256(decision_payload),
        "expert_decisions_receipt": _sha256(lock_payload),
    }


def _mean(values: list[float]) -> float | None:
    return None if not values else sum(values) / len(values)


def _adjudicated_row(item: str, rows: list[dict[str, Any]], action: str, protocol: str) -> dict[str, Any]:
    matching = [row for row in rows if row["ratable"] and row["action"] == action]
    if not matching:
        raise ValueError("adjudicated action was not observed for the item")
    onsets = [row["acceptable_onset"] for row in matching]
    windows = [value for value in onsets if not value["wait_for_user_resume"]]
    onset_guidance = {
        "wait_for_user_resume_fraction": sum(value["wait_for_user_resume"] for value in onsets) / len(onsets),
        "earliest_elapsed_s": _mean([float(value["earliest_elapsed_s"]) for value in windows]),
        "latest_elapsed_s": _mean([float(value["latest_elapsed_s"]) for value in windows]),
    }
    tags = dict(sorted(Counter(tag for row in matching for tag in row["tags"]).items()))
    quality = dict(sorted(Counter(row["quality"] for row in matching).items()))
    first = rows[0]
    distribution = Counter(row["action"] for row in rows)
    return {
        "schema": _LABEL_SCHEMA,
        "protocol_version": protocol,
        "item_id": item,
        "action": action,
        "action_distribution": {name: distribution.get(name, 0) for name in _ACTIONS},
        "onset_guidance": onset_guidance,
        "tags": tags,
        "quality": quality,
        "confidence": _mean([float(row["confidence"]) for row in rows]),
        "split": first["split"],
        "source": first["source"],
        "lineage_id": first["lineage_id"],
        "elapsed_s": first["elapsed_s"],
        "model_cutoff_s": first["model_cutoff_s"],
    }


def _validate_gate_receipt(
    lock: Mapping[str, Any], rows: list[dict[str, Any]]
) -> tuple[dict[str, list[dict[str, Any]]], set[str]]:
    expected_taxonomy = ["wait", "ack", "respond", "repair", "unratable"]
    expected_collapse = {"ack": "respond", "wait": "wait", "respond": "respond", "repair": "repair"}
    if lock.get("threshold") != 0.80 or lock.get("taxonomy") != expected_taxonomy:
        raise ValueError("agreement receipt threshold or taxonomy is not protocol v1")
    if lock.get("collapse_map") != expected_collapse or lock.get("collapsed_reliability_requires_new_pilot") is not True:
        raise ValueError("agreement receipt collapse rule is not protocol v1")
    bootstrap = lock.get("bootstrap")
    if not isinstance(bootstrap, dict) or any(
        bootstrap.get(field) != value for field, value in {
            "unit": "lineage", "seed": 20261009, "replicates": 10000,
            "requested_replicates": 10000, "configured_replicates": 10000,
            "confirmatory": True, "nonconfirmatory_override": False,
        }.items()
    ):
        raise ValueError("agreement receipt is nonconfirmatory or has invalid bootstrap settings")

    originals = [row for row in rows if not row["repeat"]]
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in originals:
        grouped.setdefault(row["item_id"], []).append(row)
    item_count = len(grouped)
    if not 60 <= item_count <= 90:
        raise ValueError("adjudication requires the locked 60–90-item pilot sample")
    lineage_counts = Counter(group[0]["lineage_id"] for group in grouped.values())
    if len(lineage_counts) < 30 or max(lineage_counts.values()) > 5:
        raise ValueError("adjudication sample violates lineage coverage or cap")
    if any(row["split"] not in {"train", "development"} for row in originals):
        raise ValueError("adjudication sample contains an excluded split")
    if any(len(group) != 3 or len({row["annotator_id"] for row in group}) != 3 for group in grouped.values()):
        raise ValueError("adjudication requires exactly 3 distinct original annotators per item")

    ratings = [[row["action"] for row in group] for group in grouped.values()]
    alpha = nominal_krippendorff_alpha(ratings)
    by_lineage: dict[str, list[tuple[str, ...]]] = {}
    for group in grouped.values():
        by_lineage.setdefault(group[0]["lineage_id"], []).append(
            tuple(row["action"] for row in group)
        )
    alpha_ci = lineage_bootstrap_ci(
        by_lineage, nominal_krippendorff_alpha, n_boot=10000, seed=20261009
    )
    overall = lock.get("overall")
    unratable_rate = sum(not row["ratable"] for row in originals) / len(originals)
    if (
        not isinstance(overall, dict)
        or overall.get("items") != len(grouped)
        or overall.get("annotations") != len(originals)
        or overall.get("unratable_rate") != unratable_rate
        or overall.get("nominal_krippendorff_alpha") != alpha
    ):
        raise ValueError("agreement receipt metrics are inconsistent with validated rows")
    receipt_ci = overall.get("lineage_bootstrap_95_ci")
    if not isinstance(receipt_ci, dict) or receipt_ci.get("nominal_krippendorff_alpha") != alpha_ci:
        raise ValueError("agreement receipt alpha bootstrap is inconsistent with validated rows")

    pairable = {item for item, group in grouped.items() if sum(row["ratable"] for row in group) >= 2}
    pairable_lineages = {grouped[item][0]["lineage_id"] for item in pairable}
    valid = alpha_ci["valid_replicates"]
    coverage = {
        "items": item_count, "allowed_item_range": [60, 90],
        "lineages": len(lineage_counts), "required_lineages": 30,
        "max_items_per_lineage": max(lineage_counts.values()),
        "allowed_max_items_per_lineage": 5,
        "pairable_items": len(pairable),
        "pairable_fraction": len(pairable) / item_count,
        "required_pairable_fraction": 0.80,
        "pairable_lineages": len(pairable_lineages),
        "required_pairable_lineages": 30,
        "valid_alpha_bootstrap_replicates": valid,
        "requested_alpha_bootstrap_replicates": 10000,
        "valid_alpha_bootstrap_fraction": valid / 10000,
        "required_valid_alpha_bootstrap_fraction": 0.90,
    }
    coverage["passed"] = (
        coverage["pairable_fraction"] >= 0.80
        and coverage["pairable_lineages"] >= 30
        and coverage["valid_alpha_bootstrap_fraction"] >= 0.90
    )
    counts = lock.get("counts")
    expected_counts = {
        "assignments": len(rows), "items": item_count,
        "original_annotations": len(originals),
        "repeat_annotations": len(rows) - len(originals),
        "pairable_items": len(pairable), "pairable_lineages": len(pairable_lineages),
    }
    if counts != expected_counts or lock.get("coverage") != coverage:
        raise ValueError("agreement receipt counts or coverage are inconsistent with validated rows")
    gate = coverage["passed"] and alpha is not None and alpha >= 0.80
    if lock.get("gate_passed") is not gate:
        raise ValueError("agreement receipt gate result is inconsistent with validated rows")
    if lock.get("collapse_decision") != ("retain_original_taxonomy" if gate else "new_collapsed_pilot_required"):
        raise ValueError("agreement receipt collapse decision is inconsistent with its gate")
    if not gate:
        raise ValueError("agreement gate did not pass; adjudication is forbidden")
    return grouped, pairable


def adjudicate(
    validated_annotations: Path,
    agreement_receipt: Path,
    hmac_key: Path,
    output_dir: Path,
    *,
    expert_decisions: Path | None = None,
    expert_receipt: Path | None = None,
) -> Path:
    """Publish causal model labels after a passing authenticated agreement gate."""
    validated_annotations, agreement_receipt, output_dir = map(
        Path, (validated_annotations, agreement_receipt, output_dir)
    )
    key = read_hmac_key(hmac_key)
    agreement_payload = agreement_receipt.read_bytes()
    lock = _json(agreement_payload, "agreement receipt")
    verify_receipt(lock, key, "agreement receipt")
    if lock.get("schema") != _AGREEMENT_RECEIPT_SCHEMA:
        raise ValueError("invalid agreement receipt schema")
    validated_payload = validated_annotations.read_bytes()
    outputs = lock.get("outputs")
    if not isinstance(outputs, dict) or outputs.get("validated_annotations.jsonl") != _sha256(validated_payload):
        raise ValueError("validated annotations do not exactly match agreement receipt")
    agreement_hashes = lock.get("sha256")
    if not isinstance(agreement_hashes, dict):
        raise ValueError("agreement receipt requires its input hash chain")
    assignment_hash = require_digest(
        agreement_hashes.get("assignment_receipt"),
        "agreement receipt sha256.assignment_receipt",
    )
    config_hash = require_digest(agreement_hashes.get("config"), "agreement receipt sha256.config")
    protocol = lock.get("protocol_version")
    if not isinstance(protocol, str) or not protocol:
        raise ValueError("agreement receipt requires protocol_version")
    rows = _load_validated(validated_payload, protocol)
    originals = [row for row in rows if not row["repeat"]]
    lineage_bindings: dict[str, tuple[str, str]] = {}
    item_identity: dict[str, tuple[Any, ...]] = {}
    for row in originals:
        identity = (row["source"], row["lineage_id"], row["split"], row["elapsed_s"], row["model_cutoff_s"])
        if item_identity.setdefault(row["item_id"], identity) != identity:
            raise ValueError(f"item {row['item_id']!r} has inconsistent metadata")
        binding = (row["source"], row["split"])
        if lineage_bindings.setdefault(row["lineage_id"], binding) != binding:
            raise ValueError(f"lineage {row['lineage_id']!r} has inconsistent source or overlaps splits")
    grouped, _ = _validate_gate_receipt(lock, rows)
    selected: dict[str, str] = {}
    unresolved: set[str] = set()
    excluded: set[str] = set()
    observed: dict[str, set[str]] = {}
    for item, group in grouped.items():
        if sum(not row["ratable"] for row in group) >= 2:
            excluded.add(item)
            continue
        counts = Counter(row["action"] for row in group if row["ratable"])
        observed[item] = set(counts)
        majority = [action for action, count in counts.items() if count >= 2]
        if len(majority) == 1:
            selected[item] = majority[0]
        else:
            unresolved.add(item)
    experts, expert_hashes = _expert_actions(
        unresolved,
        observed,
        Path(expert_decisions) if expert_decisions is not None else None,
        Path(expert_receipt) if expert_receipt is not None else None,
        agreement_receipt_payload=agreement_payload,
        validated_payload=validated_payload,
        protocol=protocol,
        key=key,
    )
    selected.update(experts)
    labels = [_adjudicated_row(item, grouped[item], selected[item], protocol) for item in sorted(selected)]
    labels_payload = _lines(labels)
    excluded_payload = _lines([
        {"item_id": item, "disposition": "excluded_majority_unratable"}
        for item in sorted(excluded)
    ]) if excluded else b""
    receipt = sign_receipt(
        {
            "schema": _ADJUDICATION_RECEIPT_SCHEMA,
            "protocol_version": protocol,
            "counts": {
                "input_items": len(grouped), "items": len(labels),
                "expert_decisions": len(experts), "majority_unratable_excluded": len(excluded),
            },
            "majority_unratable_disposition": "excluded",
            "sha256": {
                "agreement_receipt": _sha256(agreement_payload),
                "validated_annotations": _sha256(validated_payload),
                "assignment_receipt": assignment_hash,
                "config": config_hash,
                **expert_hashes,
            },
            "outputs": {
                "adjudicated.jsonl": _sha256(labels_payload),
                "excluded_majority_unratable.jsonl": _sha256(excluded_payload),
            },
        },
        key,
    )
    return _publish_packages(
        {output_dir: {
            "adjudicated.jsonl": labels_payload,
            "excluded_majority_unratable.jsonl": excluded_payload,
            "adjudication_receipt.json": receipt_bytes(receipt),
        }}
    )[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="operation", required=True)
    compile_parser = commands.add_parser("report")
    compile_parser.add_argument("--annotator-root", type=Path, required=True)
    compile_parser.add_argument("--coordinator-root", type=Path, required=True)
    compile_parser.add_argument("--raw-responses", type=Path, required=True)
    compile_parser.add_argument("--hmac-key", type=Path, required=True)
    compile_parser.add_argument("--config", default="silence.yaml")
    compile_parser.add_argument("--output-dir", type=Path, required=True)
    adjudicate_parser = commands.add_parser("adjudicate")
    adjudicate_parser.add_argument("--validated-annotations", type=Path, required=True)
    adjudicate_parser.add_argument("--agreement-receipt", type=Path, required=True)
    adjudicate_parser.add_argument("--hmac-key", type=Path, required=True)
    adjudicate_parser.add_argument("--output-dir", type=Path, required=True)
    adjudicate_parser.add_argument("--expert-decisions", type=Path)
    adjudicate_parser.add_argument("--expert-receipt", type=Path)
    args = parser.parse_args()
    if args.operation == "report":
        report(
            args.annotator_root, args.coordinator_root, args.raw_responses,
            args.hmac_key, args.config, args.output_dir,
        )
    else:
        adjudicate(
            args.validated_annotations, args.agreement_receipt, args.hmac_key,
            args.output_dir, expert_decisions=args.expert_decisions,
            expert_receipt=args.expert_receipt,
        )


if __name__ == "__main__":
    main()
