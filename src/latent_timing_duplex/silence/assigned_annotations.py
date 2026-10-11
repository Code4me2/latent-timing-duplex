"""Strict parsing for blinded pilot prefix responses."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .annotations import ALLOWED_TAGS
from .types import (
    ASSIGNED_PREFIX_SCHEMA,
    SILENCE_PILOT_PROTOCOL,
    AcceptableOnset,
    AssignedSilenceAnnotation,
    SilenceAction,
    SilenceQuality,
)

_FIELDS = frozenset(
    {
        "schema",
        "protocol_version",
        "assignment_id",
        "presentation_id",
        "annotator_id",
        "action",
        "confidence",
        "also_acceptable",
        "tags",
        "acceptable_onset",
        "quality",
        "ratable",
    }
)
_ONSET_FIELDS = frozenset(
    {"earliest_elapsed_s", "latest_elapsed_s", "wait_for_user_resume"}
)


def parse_assigned_annotation(obj: Mapping[str, Any]) -> AssignedSilenceAnnotation:
    """Parse one response without coercing JSON values or accepting extra fields."""
    if not isinstance(obj, Mapping):
        raise ValueError("row must be a JSON object")
    keys = set(obj)
    missing, unknown = sorted(_FIELDS - keys), sorted(keys - _FIELDS)
    if missing:
        raise ValueError(f"assigned response missing fields: {missing}")
    if unknown:
        raise ValueError(f"unknown assigned response fields: {unknown}")
    if obj["schema"] != ASSIGNED_PREFIX_SCHEMA:
        raise ValueError(f"schema must be {ASSIGNED_PREFIX_SCHEMA!r}")
    if obj["protocol_version"] != SILENCE_PILOT_PROTOCOL:
        raise ValueError(f"protocol_version must be {SILENCE_PILOT_PROTOCOL!r}")
    for field in ("assignment_id", "presentation_id", "annotator_id"):
        if not isinstance(obj[field], str) or not obj[field].strip():
            raise ValueError(f"{field} must be a nonempty string")

    confidence = obj["confidence"]
    if isinstance(confidence, bool) or not isinstance(confidence, int):
        raise ValueError("confidence must be an integer")
    if not isinstance(obj["ratable"], bool):
        raise ValueError("ratable must be a boolean")

    tags_obj = obj["tags"]
    if not isinstance(tags_obj, list) or any(not isinstance(tag, str) for tag in tags_obj):
        raise ValueError("tags must be a JSON array of strings")
    tags = tuple(tags_obj)
    unknown_tags = sorted(set(tags) - ALLOWED_TAGS)
    if unknown_tags:
        raise ValueError(f"unknown silence tags: {unknown_tags}")

    alternatives_obj = obj["also_acceptable"]
    if not isinstance(alternatives_obj, list) or len(alternatives_obj) > 1:
        raise ValueError("also_acceptable must be a JSON array of zero or one action")
    try:
        action = SilenceAction(_string(obj["action"], "action"))
        alternatives = tuple(
            SilenceAction(_string(value, "also_acceptable action"))
            for value in alternatives_obj
        )
        quality = SilenceQuality(_string(obj["quality"], "quality"))
    except ValueError as exc:
        raise ValueError(f"invalid assigned response enum: {exc}") from exc

    onset = _parse_onset(obj["acceptable_onset"])
    canonical = json.dumps(
        dict(obj), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return AssignedSilenceAnnotation(
        assignment_id=obj["assignment_id"],
        presentation_id=obj["presentation_id"],
        annotator_id=obj["annotator_id"],
        action=action,
        confidence=confidence,
        also_acceptable=alternatives,
        tags=tags,
        acceptable_onset=onset,
        quality=quality,
        ratable=obj["ratable"],
        schema=obj["schema"],
        protocol_version=obj["protocol_version"],
        row_sha256=hashlib.sha256(canonical).hexdigest(),
    )


def load_assigned_annotations_bytes(payload: bytes) -> list[AssignedSilenceAnnotation]:
    """Load exact UTF-8 JSONL bytes and retain each canonical-object SHA-256."""
    if not isinstance(payload, bytes):
        raise TypeError("payload must be bytes")
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"assigned responses are not valid UTF-8: {exc}") from exc
    rows: list[AssignedSilenceAnnotation] = []
    for line_number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            raise ValueError(f"assigned responses:{line_number}: blank rows are forbidden")
        try:
            obj = json.loads(
                line,
                object_pairs_hook=_unique_object,
                parse_constant=_reject_json_constant,
            )
            rows.append(parse_assigned_annotation(obj))
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            raise ValueError(f"assigned responses:{line_number}: {exc}") from exc
    if not rows:
        raise ValueError("assigned responses contain no rows")
    seen: set[str] = set()
    for row in rows:
        if row.assignment_id in seen:
            raise ValueError(f"duplicate assignment_id {row.assignment_id!r}")
        seen.add(row.assignment_id)
    return rows


def load_assigned_annotations(
    source: bytes | str | Path,
) -> list[AssignedSilenceAnnotation]:
    """Load strict assigned responses from exact bytes or a binary path read."""
    payload = source if isinstance(source, bytes) else Path(source).read_bytes()
    return load_assigned_annotations_bytes(payload)


parse_assigned_prefix_annotation = parse_assigned_annotation
load_assigned_prefix_annotations = load_assigned_annotations


def _parse_onset(value: Any) -> AcceptableOnset | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError("acceptable_onset must be an object or null")
    keys = set(value)
    if keys != _ONSET_FIELDS:
        missing, unknown = sorted(_ONSET_FIELDS - keys), sorted(keys - _ONSET_FIELDS)
        raise ValueError(
            f"acceptable_onset has missing fields {missing} and unknown fields {unknown}"
        )
    wait = value["wait_for_user_resume"]
    if not isinstance(wait, bool):
        raise ValueError("wait_for_user_resume must be a boolean")
    earliest = _optional_finite_number(value["earliest_elapsed_s"], "earliest_elapsed_s")
    latest = _optional_finite_number(value["latest_elapsed_s"], "latest_elapsed_s")
    return AcceptableOnset(earliest, latest, wait)


def _optional_finite_number(value: Any, field: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a number or null")
    if not math.isfinite(value):
        raise ValueError(f"{field} must be finite")
    return float(value)


def _string(value: Any, field: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string")
    return value


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> Any:
    raise ValueError(f"non-finite JSON number {value!r} is forbidden")
