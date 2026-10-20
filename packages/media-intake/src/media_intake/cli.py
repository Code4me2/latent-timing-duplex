"""Command line interface for independent and coordinated intake stages."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

from .config import load_config
from .locking import CANONICAL_LOCK_NAME, OutputLock
from .models import CaptionPolicy, MediaFormatPolicy, YouTubeSource
from .orchestrator import MediaIntake
from .provenance import secure_root


def _source(args: argparse.Namespace) -> YouTubeSource:
    return YouTubeSource(args.source_id, args.url, args.language)


def _media_policy(args: argparse.Namespace) -> MediaFormatPolicy:
    return MediaFormatPolicy(args.maximum_height, args.language, args.video_codec,
                             args.audio_codec, args.container)


def _caption_policy(args: argparse.Namespace) -> CaptionPolicy:
    return CaptionPolicy(args.language, not args.no_publisher, not args.no_automatic_original,
                         tuple(args.automatic_translation))


def _add_source(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--source-id", required=True)
    parser.add_argument("--url", required=True)
    parser.add_argument("--language", required=True)


def _add_output(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--output-dir", required=True, type=Path)


def _add_media(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--maximum-height", type=int, default=1080)
    parser.add_argument("--video-codec", choices=("h264", "vp9", "av1"), default="h264")
    parser.add_argument("--audio-codec", choices=("aac", "opus"), default="aac")
    parser.add_argument("--container", choices=("mp4", "webm", "mkv"), default="mp4")
    parser.add_argument("--adopt-existing", action="store_true")


def _add_captions(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--no-publisher", action="store_true")
    parser.add_argument("--no-automatic-original", action="store_true")
    parser.add_argument("--automatic-translation", action="append", default=[], metavar="LANG",
                        help="explicitly opt in to an automatic translated caption language")


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="media-intake")
    commands = root.add_subparsers(dest="command", required=True)
    metadata = commands.add_parser("metadata", help="inspect metadata only")
    _add_source(metadata)
    media = commands.add_parser("media", help="acquire and verify media only")
    _add_source(media); _add_output(media); _add_media(media)
    captions = commands.add_parser("captions", help="acquire captions only")
    _add_source(captions); _add_output(captions); _add_captions(captions)
    acquire = commands.add_parser("acquire", help="coordinate media and caption acquisition")
    _add_source(acquire); _add_output(acquire); _add_media(acquire); _add_captions(acquire)
    batch = commands.add_parser("batch", help="run selected stages for a JSON source list")
    batch.add_argument("--sources", required=True, type=Path)
    batch.add_argument("--output-dir", type=Path)
    batch.add_argument("--stages", default="media,captions",
                       help="comma-separated metadata,media,captions")
    batch.add_argument("--adopt-existing", action="store_true")
    return root


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    client = MediaIntake()
    try:
        if args.command == "metadata":
            result = client.inspect_metadata(_source(args))
        elif args.command == "batch":
            return _batch(client, args)
        else:
            source = _source(args)
            root = secure_root(args.output_dir)
            with OutputLock(root / CANONICAL_LOCK_NAME):
                if args.command == "media":
                    result = client.acquire_media(source, root, _media_policy(args),
                                                  adopt_existing=args.adopt_existing)
                elif args.command == "captions":
                    result = client.acquire_captions(source, root, _caption_policy(args))
                else:
                    result = client.acquire(source, root, _media_policy(args), _caption_policy(args),
                                            adopt_existing=args.adopt_existing)
        print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
        return int(result.get("status") == "partial")
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
        print(json.dumps({"status": "failed", "error": type(exc).__name__,
                          "detail": str(exc)[:500]}), file=sys.stderr)
        return 2


def _batch(client: MediaIntake, args: argparse.Namespace) -> int:
    sources, media_policy, caption_policy = load_config(args.sources)
    stages = tuple(item.strip() for item in args.stages.split(",") if item.strip())
    if not stages or len(set(stages)) != len(stages) or any(
            item not in {"metadata", "media", "captions"} for item in stages):
        raise ValueError("stages must be unique metadata,media,captions values")
    if set(stages) != {"metadata"} and args.output_dir is None:
        raise ValueError("--output-dir is required for media or captions")
    root = secure_root(args.output_dir) if args.output_dir else None
    rows = []
    lock = OutputLock(root / CANONICAL_LOCK_NAME) if root else _NullLock()
    with lock:
        for source in sources:
            try:
                if set(stages) == {"metadata"}:
                    value = client.inspect_metadata(source)
                elif "media" in stages and "captions" in stages:
                    value = client.acquire(source, root, media_policy, caption_policy,
                                           adopt_existing=args.adopt_existing)
                elif "media" in stages:
                    value = client.acquire_media(source, root, media_policy,
                                                 adopt_existing=args.adopt_existing)
                else:
                    value = client.acquire_captions(source, root, caption_policy)
                rows.append({"source_id": source.source_id, "status": value.get("status", "ok"),
                             "result": value})
            except Exception as exc:
                rows.append({"source_id": source.source_id, "status": "failed",
                             "error": type(exc).__name__, "detail": str(exc)[:500]})
    report = {"schema": "media-intake-batch/v1", "stages": list(stages), "sources": rows}
    print(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False))
    return int(any(row["status"] != "ok" for row in rows))


class _NullLock:
    def __enter__(self) -> "_NullLock": return self
    def __exit__(self, *_: object) -> None: return None
