#!/usr/bin/env python3
"""Extract authenticated, outcome-only observations after pilot cutoffs."""

from __future__ import annotations

import argparse
import hmac
from collections import Counter
from pathlib import Path
from typing import Any

from latent_timing_duplex.config import repo_root
from latent_timing_duplex.silence import SilenceContext, extract_silence_checkpoints

try:
    from experiments.silence_graft.build_assignments import _publish_packages, _sha256
    from experiments.silence_graft.build_manifest import _sessions, assign_lineage_splits, elapsed_stratum
    from experiments.silence_graft.compile_annotations import (
        _exact_assignment_packages,
        _load_assignment_map,
    )
    from experiments.silence_graft.protocol_v1 import load_protocol_v1
    from experiments.silence_graft.receipts import (
        read_hmac_key,
        receipt_bytes,
        require_digest,
        sign_receipt,
        verify_receipt,
    )
    from experiments.silence_graft.render_clips import _json, _lines
except ModuleNotFoundError:  # Direct script execution from the repository root.
    from build_assignments import _publish_packages, _sha256
    from build_manifest import _sessions, assign_lineage_splits, elapsed_stratum
    from compile_annotations import _exact_assignment_packages, _load_assignment_map
    from protocol_v1 import load_protocol_v1
    from receipts import read_hmac_key, receipt_bytes, require_digest, sign_receipt, verify_receipt
    from render_clips import _json, _lines

_OUTCOME_SCHEMA = "semantic-silence-observed-outcome/v1"
_RECEIPT_SCHEMA = "semantic-silence-outcomes-receipt/v1"
_PROTOCOL_DOC = "DATA_LABELING_PROTOCOL.md"
_EPSILON = 1e-9


def _active(intervals: list[tuple[float, float]], time_s: float) -> bool:
    return any(start <= time_s < end for start, end in intervals)


def _first_start(intervals: list[tuple[float, float]], cutoff_s: float, end_s: float) -> float | None:
    return next((start for start, _ in intervals if cutoff_s < start <= end_s), None)


def _overlap_duration(
    user: list[tuple[float, float]],
    assistant: list[tuple[float, float]],
    start_s: float,
    end_s: float,
) -> float:
    return sum(
        max(0.0, min(user_end, assistant_end, end_s) - max(user_start, assistant_start, start_s))
        for user_start, user_end in user
        for assistant_start, assistant_end in assistant
    )


def _verify_manifest_lock(
    payload: bytes,
    *,
    vad_payload: bytes,
    config_payload: bytes,
    protocol_payload: bytes,
    cfg: dict[str, Any],
    key: bytes,
) -> dict[str, Any]:
    lock = _json(payload, "manifest lock receipt")
    verify_receipt(lock, key, "manifest lock receipt")
    for field in ("schema", "protocol_version", "label_schema"):
        if lock.get(field) != cfg[field]:
            raise ValueError(f"manifest lock {field} does not match config")
    hashes = lock.get("sha256")
    if not isinstance(hashes, dict):
        raise ValueError("manifest lock requires SHA-256 commitments")
    expected = {
        "input_vad": _sha256(vad_payload),
        "config": _sha256(config_payload),
        "protocol_doc": _sha256(protocol_payload),
    }
    for name, digest in expected.items():
        if not hmac.compare_digest(require_digest(hashes.get(name), f"manifest lock sha256.{name}"), digest):
            raise ValueError(f"{name} SHA-256 does not match manifest lock")
    for name in ("full_manifest", "pilot_manifest", "splits"):
        require_digest(hashes.get(name), f"manifest lock sha256.{name}")
    return lock


def extract_outcomes(
    vad: Path,
    manifest_lock: Path,
    annotator_root: Path,
    coordinator_root: Path,
    hmac_key: Path,
    output_dir: Path,
    config: str | Path = "silence.yaml",
) -> Path:
    """Publish one authenticated eight-second outcome row per assigned item."""
    vad, manifest_lock, annotator_root, coordinator_root, output_dir = map(
        Path, (vad, manifest_lock, annotator_root, coordinator_root, output_dir)
    )
    key = read_hmac_key(hmac_key)
    _, config_payload, cfg = load_protocol_v1(config)
    protocol_payload = (repo_root() / "docs" / _PROTOCOL_DOC).read_bytes()
    vad_payload, lock_payload = vad.read_bytes(), manifest_lock.read_bytes()
    manifest_receipt = _verify_manifest_lock(
        lock_payload,
        vad_payload=vad_payload,
        config_payload=config_payload,
        protocol_payload=protocol_payload,
        cfg=cfg,
        key=key,
    )

    assignment_payload, _, coordinator_outputs = _exact_assignment_packages(
        annotator_root, coordinator_root, key
    )
    assignment_receipt = _json(assignment_payload, "assignment receipt")
    if assignment_receipt.get("protocol_version") != cfg["protocol_version"]:
        raise ValueError("assignment receipt protocol does not match config")
    assignment_hashes = assignment_receipt.get("sha256")
    if (
        not isinstance(assignment_hashes, dict)
        or assignment_hashes.get("config") != _sha256(config_payload)
        or assignment_hashes.get("manifest_lock") != _sha256(lock_payload)
    ):
        raise ValueError("assignment receipt does not bind this config and manifest lock")
    assignment_rows, _, private_payload, reveal_payload = _load_assignment_map(
        coordinator_root, coordinator_outputs, cfg
    )

    sessions = _sessions(vad_payload)
    by_session = {session["session_id"]: session for session in sessions}
    lineage_scope: dict[str, tuple[str, str]] = {}
    session_scope: dict[str, tuple[str, str]] = {}
    for session in sessions:
        status = "challenge" if session["challenge"] else "ordinary"
        if lineage_scope.setdefault(session["lineage_id"], (session["source"], status)) != (session["source"], status):
            raise ValueError("original VAD has a lineage spanning sources or challenge status")
        if session_scope.setdefault(session["session_id"], (session["lineage_id"], session["source"])) != (
            session["lineage_id"], session["source"]
        ):
            raise ValueError("original VAD has an inconsistent session identity")
    splits = assign_lineage_splits(
        (session["lineage_id"] for session in sessions),
        cfg["silence"]["split_seed"],
        (session["lineage_id"] for session in sessions if session["challenge"]),
    )
    manifest_counts = manifest_receipt.get("counts")
    expected_input_counts = {
        "input_sessions": len(sessions),
        "input_lineages": len(splits),
        "split_lineages": dict(sorted(Counter(splits.values()).items())),
    }
    if not isinstance(manifest_counts, dict) or any(
        manifest_counts.get(name) != value for name, value in expected_input_counts.items()
    ):
        raise ValueError("original VAD identities do not match manifest lock counts")

    reconstructed: dict[str, list[Any]] = {}
    for session in sessions:
        for checkpoint in extract_silence_checkpoints(
            session["session_id"],
            session["duration_s"],
            session["user"],
            session["assistant"],
            checkpoints_s=cfg["silence"]["checkpoints_s"],
            source=session["source"],
            lineage_id=session["lineage_id"],
        ):
            reconstructed.setdefault(checkpoint.episode_id, []).append(checkpoint)

    originals: dict[str, dict[str, Any]] = {}
    for row in assignment_rows:
        if row["repeat"]:
            continue
        item = row["item_id"]
        identity = {
            field: row[field]
            for field in (
                "item_id", "session_id", "episode_id", "source", "lineage_id", "split",
                "onset_s", "model_cutoff_s", "checkpoint_elapsed_s",
            )
        }
        if item in originals and originals[item] != identity:
            raise ValueError(f"item {item!r} has ambiguous assignment identities")
        originals[item] = identity
    expected_pilot_counts = {
        "pilot_episodes": len(originals),
        "pilot_checkpoints": len(originals),
        "pilot_lineages": len({row["lineage_id"] for row in originals.values()}),
        "pilot_by_source": dict(sorted(Counter(row["source"] for row in originals.values()).items())),
        "pilot_by_elapsed_stratum": dict(sorted(Counter(
            elapsed_stratum(
                float(row["checkpoint_elapsed_s"]),
                cfg["silence"]["duration_strata_s"],
                cfg["silence"]["duration_strata_labels"],
            )
            for row in originals.values()
        ).items())),
    }
    if any(manifest_counts.get(name) != value for name, value in expected_pilot_counts.items()):
        raise ValueError("assigned pilot coverage does not match manifest lock counts")

    horizon_s = float(cfg["pilot_media"]["reveal_s"])
    if horizon_s != 8.0:
        raise ValueError("observed-outcome horizon must remain locked at 8 seconds")
    rows: list[dict[str, Any]] = []
    for item in sorted(originals):
        identity = originals[item]
        session = by_session.get(identity["session_id"])
        if session is None:
            raise ValueError(f"item {item!r} references a session absent from original VAD")
        candidates = [
            row for row in reconstructed.get(identity["episode_id"], ())
            if abs(row.model_cutoff_s - float(identity["model_cutoff_s"])) <= _EPSILON
        ]
        if len(candidates) != 1:
            raise ValueError(f"item {item!r} has an ambiguous or missing original-VAD join")
        checkpoint = candidates[0]
        expected = (
            checkpoint.session_id,
            checkpoint.source,
            checkpoint.lineage_id,
            checkpoint.onset_s,
            checkpoint.model_cutoff_s,
            checkpoint.elapsed_s,
            splits.get(checkpoint.lineage_id or checkpoint.session_id),
        )
        observed = (
            identity["session_id"], identity["source"], identity["lineage_id"],
            float(identity["onset_s"]), float(identity["model_cutoff_s"]),
            float(identity["checkpoint_elapsed_s"]), identity["split"],
        )
        if expected != observed:
            raise ValueError(f"item {item!r} does not match original VAD identity or protected split")
        if identity["split"] not in cfg["annotation"]["pilot_splits"]:
            raise ValueError("a protected split entered observed outcomes")
        cutoff = checkpoint.model_cutoff_s
        if checkpoint.context is not SilenceContext.JOINT or _active(session["assistant"], cutoff):
            raise ValueError(f"item {item!r} is not joint silence at cutoff")
        observation_end = min(session["duration_s"], cutoff + horizon_s)
        if observation_end <= cutoff:
            raise ValueError(f"item {item!r} has no post-cutoff observation interval")

        user_resume = None if checkpoint.right_censored else checkpoint.silence_end_s
        if user_resume is not None and not cutoff < user_resume <= observation_end:
            user_resume = None
        partner_entry = _first_start(session["assistant"], cutoff, observation_end)
        if user_resume is not None and user_resume <= cutoff + _EPSILON:
            raise ValueError("user-resumption outcome must be strictly after cutoff")
        if partner_entry is not None and partner_entry <= cutoff + _EPSILON:
            raise ValueError("recorded-partner outcome must be strictly after cutoff")

        if user_resume is not None and (partner_entry is None or user_resume <= partner_entry):
            first_event, first_event_s = "user_resume", user_resume
        elif partner_entry is not None:
            first_event, first_event_s = "recorded_partner_entry", partner_entry
        else:
            first_event = first_event_s = None
        right_censored = first_event is None
        rows.append({
            "schema": _OUTCOME_SCHEMA,
            "protocol_version": cfg["protocol_version"],
            "item_id": item,
            "session_id": identity["session_id"],
            "episode_id": identity["episode_id"],
            "source": identity["source"],
            "lineage_id": identity["lineage_id"],
            "split": identity["split"],
            "onset_s": checkpoint.onset_s,
            "model_cutoff_s": cutoff,
            "checkpoint_elapsed_s": checkpoint.elapsed_s,
            "observation_horizon_s": horizon_s,
            "observed_duration_s": (first_event_s if first_event_s is not None else observation_end) - cutoff,
            "administrative_censor_s": observation_end,
            "user_resume_s": user_resume,
            "user_resume_delay_s": None if user_resume is None else user_resume - cutoff,
            "user_resume_observed": user_resume is not None,
            "recorded_partner_entry_s": partner_entry,
            "recorded_partner_entry_delay_s": None if partner_entry is None else partner_entry - cutoff,
            "recorded_partner_entry_observed": partner_entry is not None,
            "first_event": first_event,
            "first_event_s": first_event_s,
            "user_resume_event": first_event == "user_resume",
            "recorded_partner_entry_event": first_event == "recorded_partner_entry",
            "administrative_censor_event": right_censored,
            "right_censored": right_censored,
            "overlap_duration_s": _overlap_duration(
                session["user"], session["assistant"], cutoff, observation_end
            ),
            "recorded_overlap_at_user_resume": (
                user_resume is not None and _active(session["assistant"], user_resume)
            ),
        })
    if set(originals) != {row["item_id"] for row in rows}:
        raise ValueError("observed outcomes do not exactly cover assigned items")

    outcomes_payload = _lines(rows)
    receipt = sign_receipt({
        "schema": _RECEIPT_SCHEMA,
        "protocol_version": cfg["protocol_version"],
        "observation_horizon_s": horizon_s,
        "recorded_partner_entry_role": "outcome_only_not_normative_supervision",
        "counts": {"items": len(rows), "lineages": len({row["lineage_id"] for row in rows})},
        "sha256": {
            "input_vad": _sha256(vad_payload),
            "manifest_lock": _sha256(lock_payload),
            "assignment_receipt": _sha256(assignment_payload),
            "private_assignment_source_map": _sha256(private_payload),
            "coordinator_reveal_release_map": _sha256(reveal_payload),
            "config": _sha256(config_payload),
            "protocol_doc": _sha256(protocol_payload),
        },
        "outputs": {"outcomes.jsonl": _sha256(outcomes_payload)},
    }, key)
    return _publish_packages({output_dir: {
        "outcomes.jsonl": outcomes_payload,
        "outcomes_receipt.json": receipt_bytes(receipt),
    }})[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vad", type=Path, required=True)
    parser.add_argument("--manifest-lock", type=Path, required=True)
    parser.add_argument("--annotator-root", type=Path, required=True)
    parser.add_argument("--coordinator-root", type=Path, required=True)
    parser.add_argument("--hmac-key", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", default="silence.yaml")
    args = parser.parse_args()
    extract_outcomes(
        args.vad, args.manifest_lock, args.annotator_root, args.coordinator_root,
        args.hmac_key, args.output_dir, args.config,
    )


if __name__ == "__main__":
    main()
