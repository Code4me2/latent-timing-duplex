"""Host-specific metadata-only podcast audit.

This compatibility command delegates generic extraction to ``media_intake`` and
adds this research project's lineage/eligibility fields. It never downloads
media or grants pilot eligibility.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile

from media_intake import YouTubeSource
from media_intake.adapters import YtDlpAdapter
from media_intake.orchestrator import summarize_metadata as generic_summary

SOURCE_SCHEMA = "podcast-sources/v1"


def _generic_source(source: dict) -> YouTubeSource:
    return YouTubeSource(source["source_id"], source["url"], source["expected_language"])


def validate_sources(document: dict) -> list[dict]:
    if not isinstance(document, dict) or document.get("schema") != SOURCE_SCHEMA:
        raise ValueError("Expected podcast-sources/v1")
    sources = document.get("sources")
    if not isinstance(sources, list) or not sources:
        raise ValueError("Expected a nonempty sources list")
    seen: set[str] = set()
    for source in sources:
        if not isinstance(source, dict):
            raise ValueError("Source must be an object")
        if not isinstance(source.get("expected_language"), str) or not source["expected_language"]:
            raise ValueError("An explicit expected_language is required")
        parsed = _generic_source(source)
        if parsed.source_id in seen:
            raise ValueError("Source IDs must match video IDs and be unique")
        seen.add(parsed.source_id)
    return sources


def summarize_metadata(source: dict, metadata: dict) -> dict:
    """Add app policy fields without putting them in the reusable package."""
    result = generic_summary(_generic_source(source), metadata)
    result.update({
        "rights_status": source.get("rights_status", "unreviewed"),
        "speaker_channels_status": source.get("speaker_channels_status", "unaudited"),
        "pilot_eligible": False,
    })
    return result


def inspect_source(source: dict) -> dict:
    adapter = YtDlpAdapter()
    with tempfile.TemporaryDirectory(prefix="podcast-metadata-") as folder:
        root = Path(folder)
        metadata = adapter.metadata(_generic_source(source), root / "source.info.json",
                                    root / "metadata.log")
    return summarize_metadata(source, metadata)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", required=True, help="Public source inventory JSON")
    args = parser.parse_args(argv)
    with open(args.sources, encoding="utf-8") as handle:
        sources = validate_sources(json.load(handle))
    rows = []
    failed = False
    for source in sources:
        try:
            rows.append(inspect_source(source))
        except Exception as exc:
            failed = True
            rows.append({"source_id": source["source_id"], "error": type(exc).__name__})
    print(json.dumps({
        "schema": "podcast-intake-audit/v1",
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "media_downloaded": False,
        "sources": rows,
    }, indent=2, ensure_ascii=False, allow_nan=False))
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
