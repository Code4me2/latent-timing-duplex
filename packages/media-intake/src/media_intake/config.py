"""JSON source configuration shared by the CLI and batch API."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .models import CaptionPolicy, MediaFormatPolicy, YouTubeSource

SCHEMA = "media-intake-sources/v1"


def _object(value: Any, name: str) -> dict:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    return value


def load_config(path: Path) -> tuple[list[YouTubeSource], MediaFormatPolicy, CaptionPolicy]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema") != SCHEMA:
        raise ValueError(f"expected {SCHEMA}")
    rows = value.get("sources")
    if not isinstance(rows, list) or not rows:
        raise ValueError("sources must be a nonempty list")
    sources = []
    for index, raw in enumerate(rows):
        row = _object(raw, f"sources[{index}]")
        missing = [key for key in ("source_id", "url", "language") if key not in row]
        if missing:
            raise ValueError(f"sources[{index}] is missing {', '.join(missing)}")
        try:
            sources.append(YouTubeSource(row["source_id"], row["url"], row["language"]))
        except (AttributeError, TypeError) as exc:
            raise ValueError(f"sources[{index}] has invalid field types") from exc
    if len({row.source_id for row in sources}) != len(sources):
        raise ValueError("source IDs must be unique")

    policies_raw = value.get("policies", {})
    policies = _object(policies_raw, "policies")
    media_raw = policies.get("media", {})
    captions_value = policies.get("captions", {})
    media_values = _object(media_raw, "policies.media")
    captions_raw = dict(_object(captions_value, "policies.captions"))
    if "automatic_translations" in captions_raw:
        translations = captions_raw["automatic_translations"]
        if not isinstance(translations, list):
            raise ValueError("policies.captions.automatic_translations must be a list")
        captions_raw["automatic_translations"] = tuple(translations)
    try:
        media = MediaFormatPolicy(**media_values)
        captions = CaptionPolicy(**captions_raw)
    except (AttributeError, TypeError) as exc:
        raise ValueError("policy has unknown fields or invalid field types") from exc
    for source in sources:
        if source.language != media.language or source.language != captions.language:
            raise ValueError("source and policy languages must match")
    return sources, media, captions
