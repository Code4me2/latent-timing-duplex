"""Small orchestration API with independent metadata, media, and caption stages."""
from __future__ import annotations

from dataclasses import asdict
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any

from .adapters import FFprobeAdapter, YtDlpAdapter, original_audio_formats, select_original_audio, validate_probe
from .models import CaptionPolicy, CaptionTrack, MediaFormatPolicy, YouTubeSource
from .provenance import (
    artifact, new_attempt, private_file, secure_directory, secure_root, sha256_file, utc_now,
    write_json_new,
)

_RECEIPT_SCHEMAS = {"media-intake-media-receipt/v1", "podcast-media-receipt/v1"}
_VIDEO_FORMAT = {"h264": ("avc1", "mp4"), "vp9": ("vp9", "webm"), "av1": ("av01", "webm")}


def _source_value(source: YouTubeSource) -> dict:
    return asdict(source)


def _policy_value(policy: MediaFormatPolicy | CaptionPolicy) -> dict:
    value = asdict(policy)
    if isinstance(policy, CaptionPolicy):
        value["automatic_translations"] = list(policy.automatic_translations)
    return value


def _promote_new(source: Path, destination: Path) -> None:
    """Atomically publish one staged file without replacing existing evidence."""
    os.link(source, destination, follow_symlinks=False)
    source.unlink()


def summarize_metadata(source: YouTubeSource, metadata: dict) -> dict:
    formats = original_audio_formats(source, metadata)
    return {
        "source_id": source.source_id,
        "url": source.url,
        "title": metadata.get("title"),
        "channel": metadata.get("channel"),
        "duration_s": metadata.get("duration"),
        "upload_date": metadata.get("upload_date"),
        "availability": metadata.get("availability"),
        "reported_license": metadata.get("license"),
        "original_audio_formats": formats,
        "format_check": "original_candidates_found" if formats else "manual_review_required",
    }


def select_caption_tracks(metadata: dict, policy: CaptionPolicy) -> tuple[list[CaptionTrack], list[dict]]:
    primary = policy.language.split("-")[0]
    tracks: list[CaptionTrack] = []
    unavailable: list[dict] = []
    published = metadata.get("subtitles") or {}
    if policy.publisher:
        key = next((key for key in dict.fromkeys([policy.language, primary, *sorted(published)])
                    if key.split("-")[0] == primary and published.get(key)), None)
        if key is None:
            unavailable.append({"kind": "publisher", "status": "unavailable",
                                "reason": "no matching publisher track"})
        else:
            tracks.append(CaptionTrack("publisher", key))
    automatic = metadata.get("automatic_captions") or {}
    if policy.automatic_original:
        key = next((key for key in (policy.language + "-orig", primary + "-orig")
                    if automatic.get(key)), None)
        if key is None:
            unavailable.append({"kind": "automatic", "status": "unavailable",
                                "reason": "no explicitly original automatic track; translations were not substituted"})
        else:
            tracks.append(CaptionTrack("automatic", key))
    for language in policy.automatic_translations:
        candidates = (language, language.split("-")[0])
        key = next((item for item in candidates if automatic.get(item)), None)
        if key is None:
            unavailable.append({"kind": "automatic_translation", "requested_language": language,
                                "status": "unavailable", "reason": "requested automatic translation unavailable"})
        else:
            tracks.append(CaptionTrack("automatic_translation", key))
    for track in tracks:
        if not re.fullmatch(r"[A-Za-z0-9_-]+", track.language_key):
            raise ValueError("unsafe caption language key")
    return tracks, unavailable


def validate_vtt(path: Path) -> None:
    if path.is_symlink() or not path.is_file():
        raise ValueError("caption output is not a regular file")
    text = path.read_text(encoding="utf-8-sig")
    if not text.startswith("WEBVTT") or not re.search(
        r"(?m)^\s*(?:\d+:)?\d{2}:\d{2}\.\d{3}\s+-->\s+", text
    ):
        raise ValueError("caption file is not WebVTT with timed cues")


def verify_original_selection(source: YouTubeSource, info: dict, policy: MediaFormatPolicy) -> None:
    valid = set()
    for row in info.get("formats", []):
        try:
            candidate = select_original_audio(source, {"id": info.get("id"), "formats": [row]}, policy)
        except ValueError:
            continue
        valid.add(candidate)
    ids = str(info.get("format_id", "")).split("+")
    selected = [row for row in info.get("formats", []) if row.get("format_id") in ids]
    if len(ids) != 2 or len(set(ids)) != 2 or len(selected) != 2:
        raise ValueError("expected two identifiable selected media formats")
    audio = [row for row in selected if row.get("vcodec") == "none"]
    video = [row for row in selected if row.get("acodec") == "none"]
    video_prefix, video_ext = _VIDEO_FORMAT[policy.video_codec]
    height = video[0].get("height") if len(video) == 1 else None
    if len(audio) != 1 or audio[0].get("format_id") not in valid:
        raise ValueError("acquisition metadata does not establish original audio selection")
    if (len(video) != 1 or not str(video[0].get("vcodec") or "").startswith(video_prefix)
            or video[0].get("ext") != video_ext or isinstance(height, bool)
            or not isinstance(height, (int, float)) or not 0 < height <= policy.maximum_height):
        raise ValueError("acquisition metadata does not match the selected video policy")


class MediaIntake:
    """Orchestrate stages. Adapters can be replaced by deterministic test doubles."""

    def __init__(self, *, yt_dlp: Any | None = None, ffprobe: Any | None = None) -> None:
        self.yt_dlp = yt_dlp or YtDlpAdapter()
        self.ffprobe = ffprobe or FFprobeAdapter()

    def inspect_metadata(self, source: YouTubeSource) -> dict:
        with tempfile.TemporaryDirectory(prefix="media-intake-metadata-") as folder:
            path = Path(folder)
            metadata = self.yt_dlp.metadata(source, path / "source.info.json", path / "metadata.log")
        result = summarize_metadata(source, metadata)
        result["tools"] = {"yt-dlp": self._adapter_version(self.yt_dlp)}
        result["checked_at"] = utc_now()
        return result

    def acquire_media(self, source: YouTubeSource, output_root: Path,
                      policy: MediaFormatPolicy, *, adopt_existing: bool = False,
                      attempt_dir: Path | None = None, metadata: dict | None = None,
                      metadata_path: Path | None = None,
                      receipt_schema: str = "media-intake-media-receipt/v1") -> dict:
        if policy.language != source.language:
            raise ValueError("media policy language must match source language")
        if receipt_schema not in _RECEIPT_SCHEMAS:
            raise ValueError("unsupported media receipt schema")
        root = secure_root(Path(output_root))
        attempt = self._attempt(root, source, attempt_dir)
        record_path = attempt / "media-record.json"
        source_value, policy_value = _source_value(source), _policy_value(policy)
        try:
            fresh, fresh_path = self._metadata(source, attempt, metadata, metadata_path)
            audio_id = select_original_audio(source, fresh, policy)
            tools = self._media_versions()
            media = root / f"{source.media_id}.{policy.container}"
            info_path = root / f"{source.media_id}.info.json"
            receipt_path = root / f"{source.media_id}.media-receipt.json"
            for candidate in (media, info_path, receipt_path):
                if candidate.is_symlink():
                    raise ValueError("refusing symlinked acquisition artifact")
            reused = media.exists()
            probe_path = attempt / "ffprobe.json"
            if reused:
                if not media.is_file():
                    raise ValueError("media output is not a regular file")
                if not info_path.is_file():
                    raise ValueError("media acquisition metadata is missing or not a regular file")
                if receipt_path.exists():
                    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
                    if not isinstance(receipt, dict):
                        raise ValueError("existing media receipt must be a JSON object")
                    if receipt.get("schema") != receipt_schema:
                        raise ValueError("existing media receipt schema mismatch; use its compatible command or migrate explicitly")
                    if receipt.get("source_id") != source.source_id:
                        raise ValueError("existing media receipt identity mismatch")
                    if receipt.get("source") not in (None, source_value):
                        raise ValueError("existing media receipt source mismatch")
                    if receipt.get("media_policy") not in (None, policy_value):
                        raise ValueError("existing media receipt policy mismatch")
                    if (receipt.get("media", {}).get("sha256") != sha256_file(media)
                            or receipt.get("acquisition_metadata", {}).get("sha256") != sha256_file(info_path)):
                        raise ValueError("existing media/metadata changed; quarantine and review before retry")
                elif not adopt_existing:
                    raise ValueError("unreceipted media exists; explicit adoption is required")
                acquisition = json.loads(info_path.read_text(encoding="utf-8"))
                if not isinstance(acquisition, dict):
                    raise ValueError("acquisition metadata must be a JSON object")
                verify_original_selection(source, acquisition, policy)
                probe = self.ffprobe.inspect(media, probe_path, attempt / "ffprobe.log")
                validate_probe(probe, float(acquisition["duration"]), policy)
            else:
                if receipt_path.exists():
                    raise ValueError("receipted media is missing; restore or quarantine before retry")
                if info_path.exists():
                    raise ValueError("unreceipted acquisition sidecar exists without media; quarantine it before retry")
                staging = secure_directory(attempt / "media-staging", root=root)
                staged_media = staging / media.name
                staged_info = staging / info_path.name
                self.yt_dlp.download_media(source, fresh_path, staging, policy, audio_id,
                                           attempt / "media-download.log")
                if not staged_media.exists() or not staged_info.exists():
                    raise ValueError("media adapter did not produce expected media and info JSON")
                private_file(staged_media)
                private_file(staged_info)
                acquisition = json.loads(staged_info.read_text(encoding="utf-8"))
                if not isinstance(acquisition, dict):
                    raise ValueError("acquisition metadata must be a JSON object")
                verify_original_selection(source, acquisition, policy)
                probe = self.ffprobe.inspect(staged_media, probe_path, attempt / "ffprobe.log")
                validate_probe(probe, float(acquisition["duration"]), policy)
                promoted: list[Path] = []
                try:
                    _promote_new(staged_media, media)
                    promoted.append(media)
                    _promote_new(staged_info, info_path)
                    promoted.append(info_path)
                except Exception:
                    for item in promoted:
                        item.unlink(missing_ok=True)
                    raise
            media_item = artifact(media, root)
            if not receipt_path.exists():
                checked = utc_now()
                write_json_new(receipt_path, {
                    "schema": receipt_schema, "source_id": source.source_id,
                    "source": source_value, "media_policy": policy_value,
                    "verified_at": checked, "adopted_existing": reused,
                    "tools_at_verification": tools,
                    # Unknown original download times are deliberately left null on adoption.
                    "acquisition_time": None if reused else checked,
                    "media": media_item, "acquisition_metadata": artifact(info_path, root),
                })
            result = {
                "stage": "media", "status": "ok", "source_id": source.source_id,
                "source": source_value, "media_policy": policy_value,
                "verified_at": utc_now(), "tools": tools,
                "media_reused": reused, "media": media_item,
                "media_receipt": artifact(receipt_path, root),
                "acquisition_metadata": artifact(info_path, root),
                "refreshed_metadata": artifact(fresh_path, root), "probe": artifact(probe_path, root),
            }
            write_json_new(record_path, result)
            return result
        except Exception as exc:
            self._failure_record(record_path, source, "media", exc, policy)
            raise

    def acquire_captions(self, source: YouTubeSource, output_root: Path,
                         policy: CaptionPolicy, *, attempt_dir: Path | None = None,
                         metadata: dict | None = None, metadata_path: Path | None = None,
                         caption_parent: str = "captions") -> dict:
        if policy.language != source.language:
            raise ValueError("caption policy language must match source language")
        root = secure_root(Path(output_root))
        attempt = self._attempt(root, source, attempt_dir)
        record_path = attempt / "captions-record.json"
        try:
            fresh, fresh_path = self._metadata(source, attempt, metadata, metadata_path)
            tools = {"yt-dlp": self._adapter_version(self.yt_dlp)}
            tracks, statuses = select_caption_tracks(fresh, policy)
            if caption_parent not in ("", "captions"):
                raise ValueError("unsupported caption layout")
            for track in tracks:
                folder = secure_directory(attempt / caption_parent / track.kind, root=root)
                log = folder / "download.log"
                try:
                    path = self.yt_dlp.download_caption(source, fresh_path, folder, track, log)
                    validate_vtt(path)
                    private_file(path)
                    statuses.append({"kind": track.kind, "language_key": track.language_key,
                                     "status": "downloaded", "artifact": artifact(path, root)})
                except Exception as exc:
                    statuses.append({"kind": track.kind, "language_key": track.language_key,
                                     "status": "failed", "error": type(exc).__name__,
                                     "detail": str(exc)[:500], "log": str(log.relative_to(root))})
            status = "partial" if any(row["status"] == "failed" for row in statuses) else "ok"
            result = {"stage": "captions", "status": status, "source_id": source.source_id,
                      "source": _source_value(source), "caption_policy": _policy_value(policy),
                      "verified_at": utc_now(),
                      "tools": tools,
                      "refreshed_metadata": artifact(fresh_path, root),
                      "captions": statuses}
            write_json_new(record_path, result)
            return result
        except Exception as exc:
            self._failure_record(record_path, source, "captions", exc, policy)
            raise

    def acquire(self, source: YouTubeSource, output_root: Path, media_policy: MediaFormatPolicy,
                caption_policy: CaptionPolicy, *, adopt_existing: bool = False,
                attempt_dir: Path | None = None,
                receipt_schema: str = "media-intake-media-receipt/v1",
                record_schema: str = "media-intake-acquisition/v1",
                caption_parent: str = "captions",
                record_name: str = "record.json") -> dict:
        root = secure_root(Path(output_root))
        attempt = self._attempt(root, source, attempt_dir)
        metadata, metadata_path = self._metadata(source, attempt, None, None)
        media = self.acquire_media(source, root, media_policy, adopt_existing=adopt_existing,
                                   attempt_dir=attempt, metadata=metadata,
                                   metadata_path=metadata_path, receipt_schema=receipt_schema)
        captions = self.acquire_captions(source, root, caption_policy, attempt_dir=attempt,
                                         metadata=metadata, metadata_path=metadata_path,
                                         caption_parent=caption_parent)
        result = summarize_metadata(source, metadata)
        result.update({
            "schema": record_schema,
            "source": _source_value(source),
            "media_policy": _policy_value(media_policy),
            "caption_policy": _policy_value(caption_policy),
            "status": "partial" if captions["status"] == "partial" else "ok",
            "checked_at": utc_now(),
            "tools": media["tools"],
            "media_reused": media["media_reused"],
            "media": media["media"],
            "media_receipt": media["media_receipt"],
            "acquisition_metadata": media["acquisition_metadata"],
            "refreshed_metadata": media["refreshed_metadata"],
            "probe": media["probe"],
            "captions": captions["captions"],
        })
        if record_name not in ("record.json", "intake-record.json"):
            raise ValueError("unsupported coordinated record name")
        write_json_new(attempt / record_name, result)
        return result

    def _attempt(self, root: Path, source: YouTubeSource, supplied: Path | None) -> Path:
        if supplied is None:
            return new_attempt(root, source.source_id)
        return secure_directory(Path(supplied), root=root)

    def _metadata(self, source: YouTubeSource, attempt: Path, metadata: dict | None,
                  metadata_path: Path | None) -> tuple[dict, Path]:
        if metadata is not None:
            if metadata_path is None or not metadata_path.is_file() or metadata_path.is_symlink():
                raise ValueError("provided metadata requires its regular provenance file")
            if metadata.get("id") != source.media_id:
                raise ValueError("returned video identity does not match source")
            return metadata, metadata_path
        path = attempt / "source.info.json"
        return self.yt_dlp.metadata(source, path, attempt / "metadata.log"), path

    def _media_versions(self) -> dict[str, str]:
        ffmpeg = getattr(self.yt_dlp, "ffmpeg_version", None)
        return {
            "yt-dlp": self._adapter_version(self.yt_dlp),
            "ffmpeg": ffmpeg() if ffmpeg is not None else f"injected:{type(self.yt_dlp).__name__}",
            "ffprobe": self._adapter_version(self.ffprobe),
        }

    @staticmethod
    def _adapter_version(adapter: Any) -> str:
        method = getattr(adapter, "version", None)
        return method() if method is not None else f"injected:{type(adapter).__name__}"

    @staticmethod
    def _failure_record(path: Path, source: YouTubeSource, stage: str, exc: Exception,
                        policy: MediaFormatPolicy | CaptionPolicy) -> None:
        if path.exists():
            return
        policy_key = "media_policy" if isinstance(policy, MediaFormatPolicy) else "caption_policy"
        write_json_new(path, {
            "stage": stage, "status": "failed", "source_id": source.source_id,
            "source": _source_value(source), policy_key: _policy_value(policy),
            "failed_at": utc_now(), "error": type(exc).__name__, "detail": str(exc)[:500],
        })
