"""Backwards-compatible project wrapper around the standalone media_intake package.

Project policy (permission confirmation, Git exclusion, and pilot eligibility)
stays here. Generic acquisition, validation, provenance, and retries live in the
extractable package under ``packages/media-intake``.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import uuid

from media_intake import CaptionPolicy, MediaFormatPolicy, MediaIntake
from media_intake.adapters import validate_probe
from media_intake.adapters.yt_dlp import select_original_audio
from media_intake.locking import CANONICAL_LOCK_NAME, OutputLock
from media_intake.models import YouTubeSource
from media_intake.orchestrator import select_caption_tracks, validate_vtt
from media_intake.orchestrator import verify_original_selection as _verify_original_selection
from media_intake.provenance import artifact, secure_directory, secure_root, sha256_file, utc_now, write_json_new

from .podcasts import summarize_metadata, validate_sources

BASE = ["yt-dlp", "--ignore-config", "--no-playlist", "--no-progress",
        "--no-overwrites", "--socket-timeout", "30", "--retries", "3",
        "--fragment-retries", "3", "--retry-sleep", "2"]


def now() -> str:
    return utc_now()


def write_json(path: Path, value: dict) -> None:
    write_json_new(path, value)


def sha256(path: Path) -> str:
    return sha256_file(path)


def run_logged(command: list[str], log: Path, *, stdout: Path | None = None,
               timeout: int = 7200) -> None:
    """Legacy helper retained for callers; commands are always shell-free."""
    log_descriptor = os.open(log, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    with os.fdopen(log_descriptor, "ab") as errors:
        if stdout is None:
            subprocess.run(command, stdout=errors, stderr=errors, check=True,
                           timeout=timeout, shell=False)
        else:
            descriptor = os.open(stdout, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(descriptor, "wb") as output:
                subprocess.run(command, stdout=output, stderr=errors, check=True,
                               timeout=timeout, shell=False)


def _source(source: dict) -> YouTubeSource:
    return YouTubeSource(source["source_id"], source["url"], source["expected_language"])


def _media_policy(source: dict) -> MediaFormatPolicy:
    return MediaFormatPolicy(maximum_height=1080, language=source["expected_language"],
                             video_codec="h264", audio_codec="aac", container="mp4")


def select_audio(source: dict, metadata: dict) -> str:
    return select_original_audio(_source(source), metadata, _media_policy(source))


def caption_tracks(metadata: dict, language: str) -> list[tuple[str, str]]:
    tracks, _ = select_caption_tracks(metadata, CaptionPolicy(language=language))
    return [(track.kind, track.language_key) for track in tracks]


def verify_probe(probe: dict, duration: float) -> None:
    validate_probe(probe, duration, MediaFormatPolicy())


def verify_original_selection(source: dict, info: dict) -> None:
    _verify_original_selection(_source(source), info, _media_policy(source))


def acquire(source: dict, root: Path, work: Path, tools: dict,
            adopt_existing: bool) -> dict:
    """Compatibility API preserving legacy paths and receipt schemas."""
    secure_directory(work)
    client = MediaIntake()
    result = client.acquire(
        _source(source), root, _media_policy(source),
        CaptionPolicy(language=source["expected_language"]),
        adopt_existing=adopt_existing, attempt_dir=work,
        receipt_schema="podcast-media-receipt/v1",
        record_schema="podcast-acquisition/v1",
        caption_parent="", record_name="intake-record.json",
    )
    # Keep the historical aggregate log path while retaining stage-specific logs.
    aggregate = work / "download.log"
    payload = b"".join(path.read_bytes() for path in
                       (work / "metadata.log", work / "media-download.log", work / "ffprobe.log")
                       if path.exists() and path.is_file() and not path.is_symlink())
    descriptor = os.open(aggregate, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)
    # Preserve the host's richer curated publisher provenance at this boundary.
    fresh = json.loads((work / "source.info.json").read_text(encoding="utf-8"))
    result.update({key: fresh.get(key) for key in (
        "channel_id", "channel_url", "uploader", "uploader_id", "uploader_url",
        "timestamp", "release_timestamp", "description", "chapters", "categories", "tags",
    )})
    # App-only policy and eligibility assertions never enter the generic package.
    result.update({
        "rights_status": source.get("rights_status", "unreviewed"),
        "speaker_channels_status": source.get("speaker_channels_status", "unaudited"),
        "pilot_eligible": False,
    })
    write_json_new(work / "record.json", result)
    return result


def _inside_git(path: Path) -> bool:
    absolute = path.expanduser().absolute()
    return any((parent / ".git").exists() for parent in (absolute, *absolute.parents))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--adopt-existing", action="store_true",
                        help="Verify and receipt existing MP4s with matching yt-dlp info JSON")
    args = parser.parse_args(argv)
    document = json.loads(Path(args.sources).read_text(encoding="utf-8"))
    sources = validate_sources(document)
    if any(source.get("rights_status") != "user_confirmed" for source in sources):
        parser.error("Every source must have rights_status=user_confirmed")
    requested_root = Path(args.output_dir)
    if _inside_git(requested_root):
        parser.error("Media and transcripts must be stored outside a Git working tree")
    for name in ("yt-dlp", "ffmpeg", "ffprobe"):
        if not shutil.which(name):
            parser.error(f"Required executable not found: {name}")
    root = secure_root(requested_root)
    with OutputLock(root / CANONICAL_LOCK_NAME):
        run = secure_directory(root / "acquisition-runs" / uuid.uuid4().hex, root=root)
        write_json_new(run / "sources.json", document)
        results = []
        for source in sources:
            work = run / source["source_id"]
            try:
                record = acquire(source, root, work, {}, args.adopt_existing)
                results.append({"source_id": source["source_id"], "status": record["status"],
                                "record": str((work / "record.json").relative_to(root))})
            except Exception as exc:
                results.append({"source_id": source["source_id"], "status": "failed",
                                "error": type(exc).__name__, "detail": str(exc)[:500]})
        report = {"schema": "podcast-acquisition-run/v1", "completed_at": now(),
                  "sources": results}
        write_json_new(run / "run.json", report)
        print(json.dumps({"report": str(run / "run.json"), **report}, indent=2))
        return int(any(row["status"] != "ok" for row in results))


if __name__ == "__main__":
    raise SystemExit(main())
