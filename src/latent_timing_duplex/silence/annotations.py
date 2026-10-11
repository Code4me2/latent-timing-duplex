"""Strict JSONL validation for human silence annotations."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from .types import AcceptableOnset, SilenceAction, SilenceAnnotation, SilenceQuality

ALLOWED_TAGS = frozenset(
    {
        "hesitation",
        "floor_holding",
        "deliberation",
        "turn_yielding",
        "response_solicitation",
        "affective_stance",
        "disengagement",
        "comprehension_difficulty",
        "technical_uncertainty",
    }
)


def parse_annotation(obj: Mapping[str, Any]) -> SilenceAnnotation:
    tags = tuple(str(tag) for tag in obj.get("tags", ()))
    unknown = sorted(set(tags) - ALLOWED_TAGS)
    if unknown:
        raise ValueError(f"unknown silence tags: {unknown}")
    onset_obj = obj.get("acceptable_onset")
    if onset_obj is None:
        onset = None
    elif isinstance(onset_obj, Mapping):
        onset = AcceptableOnset(
            earliest_elapsed_s=_optional_float(onset_obj.get("earliest_elapsed_s")),
            latest_elapsed_s=_optional_float(onset_obj.get("latest_elapsed_s")),
            wait_for_user_resume=bool(onset_obj.get("wait_for_user_resume", False)),
        )
    else:
        raise ValueError("acceptable_onset must be an object or null")
    try:
        action = SilenceAction(str(obj["action"]))
        quality = SilenceQuality(str(obj["quality"]))
        alternatives = tuple(
            SilenceAction(str(value)) for value in obj.get("also_acceptable", ())
        )
    except KeyError as exc:
        raise ValueError(f"annotation missing {exc.args[0]!r}") from exc
    except ValueError as exc:
        raise ValueError(f"invalid annotation enum: {exc}") from exc
    confidence = obj.get("confidence")
    return SilenceAnnotation(
        episode_id=str(obj.get("episode_id", "")),
        checkpoint_elapsed_s=float(obj.get("checkpoint_elapsed_s", -1.0)),
        annotator=str(obj.get("annotator", "")),
        action=action,
        tags=tags,
        acceptable_onset=onset,
        quality=quality,
        ratable=bool(obj.get("ratable", True)),
        confidence=None if confidence is None else int(confidence),
        also_acceptable=alternatives,
        schema=str(obj.get("schema", "semantic-silence-labels/v1")),
        clip_sha256=(
            None if obj.get("clip_sha256") is None else str(obj["clip_sha256"])
        ),
    )


def load_annotations(path: str | Path) -> list[SilenceAnnotation]:
    rows: list[SilenceAnnotation] = []
    for line_number, line in enumerate(
        Path(path).read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
            if not isinstance(obj, dict):
                raise ValueError("row must be a JSON object")
            rows.append(parse_annotation(obj))
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            raise ValueError(f"{path}:{line_number}: {exc}") from exc
    _reject_duplicates(rows)
    return rows


def validate_annotation_coverage(
    annotations: Iterable[SilenceAnnotation],
    expected_keys: Iterable[tuple[str, float]],
    *,
    annotators_per_item: int = 3,
) -> None:
    """Require the locked number of independent rows for each expected item."""
    if annotators_per_item < 1:
        raise ValueError("annotators_per_item must be positive")
    counts: dict[tuple[str, float], int] = {}
    expected = set(expected_keys)
    for row in annotations:
        key = (row.episode_id, row.checkpoint_elapsed_s)
        if key not in expected:
            raise ValueError(f"annotation references unknown checkpoint {key!r}")
        counts[key] = counts.get(key, 0) + 1
    missing = sorted(key for key in expected if counts.get(key, 0) != annotators_per_item)
    if missing:
        raise ValueError(
            f"{len(missing)} checkpoints do not have exactly "
            f"{annotators_per_item} annotations"
        )


def _reject_duplicates(rows: Iterable[SilenceAnnotation]) -> None:
    seen: set[tuple[str, float, str]] = set()
    for row in rows:
        key = (row.episode_id, row.checkpoint_elapsed_s, row.annotator)
        if key in seen:
            raise ValueError(f"duplicate annotation {key!r}")
        seen.add(key)


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)
