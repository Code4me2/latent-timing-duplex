"""yt-dlp adapter: metadata, media, and captions are separate operations."""
from __future__ import annotations

import json
from pathlib import Path
import re

from ..models import CaptionTrack, MediaFormatPolicy, YouTubeSource
from ..provenance import write_bytes_new
from ..runner import Runner, SubprocessRunner

_SAFE_FORMAT_ID = re.compile(r"[A-Za-z0-9_-]+")
_SAFE_TRACK = re.compile(r"[A-Za-z0-9_-]+")
_CODEC = {
    "h264": ("avc1", "mp4"), "vp9": ("vp9", "webm"), "av1": ("av01", "webm"),
    "aac": ("mp4a.", "m4a"), "opus": ("opus", "webm"),
}


class YtDlpAdapter:
    def __init__(self, runner: Runner | None = None, executable: str = "yt-dlp") -> None:
        self.runner = runner or SubprocessRunner()
        self.executable = executable

    @property
    def base(self) -> list[str]:
        return [self.executable, "--ignore-config", "--no-playlist", "--no-progress",
                "--no-overwrites", "--socket-timeout", "30", "--retries", "3",
                "--fragment-retries", "3", "--retry-sleep", "2"]

    def metadata(self, source: YouTubeSource, destination: Path, log: Path) -> dict:
        result = self.runner.run(self.base + ["--skip-download", "--dump-single-json", source.url],
                                 timeout=180, log=log)
        write_bytes_new(destination, result.stdout)
        value = json.loads(result.stdout.decode("utf-8"))
        if value.get("id") != source.media_id:
            raise ValueError("returned video identity does not match source")
        return value

    def download_media(self, source: YouTubeSource, metadata_path: Path, output_root: Path,
                       policy: MediaFormatPolicy, audio_format_id: str, log: Path) -> None:
        if not _SAFE_FORMAT_ID.fullmatch(audio_format_id):
            raise ValueError("unsafe audio format identifier")
        video_prefix, video_ext = _CODEC[policy.video_codec]
        expression = (f"bv[ext={video_ext}][vcodec^={video_prefix}]"
                      f"[height<={policy.maximum_height}]+{audio_format_id}")
        self.runner.run(self.base + ["--load-info-json", str(metadata_path), "-f", expression,
                        "--merge-output-format", policy.container, "--write-info-json",
                        "-o", str(output_root / "%(id)s.%(ext)s")], timeout=7200, log=log)

    def download_caption(self, source: YouTubeSource, metadata_path: Path, output_dir: Path,
                         track: CaptionTrack, log: Path) -> Path:
        if not _SAFE_TRACK.fullmatch(track.language_key):
            raise ValueError("unsafe caption language key")
        automatic = track.kind != "publisher"
        command = self.base + ["--load-info-json", str(metadata_path), "--skip-download",
                   "--write-auto-subs" if automatic else "--write-subs",
                   "--sub-langs", track.language_key, "--sub-format", "vtt",
                   "--sleep-subtitles", "2", "-o", str(output_dir / "%(id)s.%(ext)s")]
        self.runner.run(command, timeout=300, log=log)
        return output_dir / f"{source.media_id}.{track.language_key}.vtt"

    def version(self) -> str:
        result = self.runner.run([self.executable, "--version"], timeout=15)
        return result.stdout.decode("utf-8", "replace").splitlines()[0]

    def ffmpeg_version(self) -> str:
        result = self.runner.run(["ffmpeg", "-version"], timeout=15)
        return result.stdout.decode("utf-8", "replace").splitlines()[0]


def select_original_audio(source: YouTubeSource, metadata: dict,
                          policy: MediaFormatPolicy) -> str:
    if metadata.get("id") != source.media_id:
        raise ValueError("returned video identity does not match source")
    codec_prefix, extension = _CODEC[policy.audio_codec]
    candidates = []
    for item in metadata.get("formats", []):
        note = str(item.get("format_note") or "").lower()
        if (item.get("vcodec") == "none" and item.get("acodec") not in (None, "none")
                and str(item.get("acodec")).startswith(codec_prefix)
                and item.get("ext") == extension and item.get("language") == policy.language
                and "original" in note and "dubbed" not in note):
            candidates.append(item)
    if not candidates:
        raise ValueError("no explicitly original requested-language audio matches the format policy")
    selected = max(candidates, key=lambda row: row.get("abr") or 0).get("format_id")
    if not isinstance(selected, str) or not _SAFE_FORMAT_ID.fullmatch(selected):
        raise ValueError("unsafe audio format identifier")
    return selected


def original_audio_formats(source: YouTubeSource, metadata: dict) -> list[dict]:
    if metadata.get("id") != source.media_id:
        raise ValueError("returned video identity does not match source")
    answer = []
    for item in metadata.get("formats", []):
        note = str(item.get("format_note") or "").lower()
        if (item.get("vcodec") == "none" and item.get("acodec") not in (None, "none")
                and item.get("language") == source.language
                and "original" in note and "dubbed" not in note):
            answer.append({key: item.get(key) for key in (
                "format_id", "ext", "acodec", "audio_channels", "asr", "abr", "language", "format_note"
            )})
    return answer
