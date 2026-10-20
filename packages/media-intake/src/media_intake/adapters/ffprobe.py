"""ffprobe adapter and typed-policy verification."""
from __future__ import annotations

import json
import math
from pathlib import Path

from ..models import MediaFormatPolicy
from ..provenance import write_bytes_new
from ..runner import Runner, SubprocessRunner

_PROBE_CODEC = {"h264": "h264", "vp9": "vp9", "av1": "av1", "aac": "aac", "opus": "opus"}
_CONTAINER_FORMATS = {
    "mp4": {"mov", "mp4", "m4a", "3gp", "3g2", "mj2"},
    # ffprobe uses one demuxer name for both Matroska and WebM.
    "webm": {"matroska", "webm"},
    "mkv": {"matroska", "webm"},
}


class FFprobeAdapter:
    def __init__(self, runner: Runner | None = None, executable: str = "ffprobe") -> None:
        self.runner = runner or SubprocessRunner()
        self.executable = executable

    def inspect(self, media: Path, destination: Path, log: Path) -> dict:
        result = self.runner.run([self.executable, "-v", "error", "-show_streams",
                                  "-show_format", "-of", "json", str(media)],
                                 timeout=120, log=log)
        write_bytes_new(destination, result.stdout)
        return json.loads(result.stdout.decode("utf-8"))

    def version(self) -> str:
        result = self.runner.run([self.executable, "-version"], timeout=15)
        return result.stdout.decode("utf-8", "replace").splitlines()[0]


def validate_probe(probe: dict, expected_duration: float, policy: MediaFormatPolicy) -> None:
    format_names = {
        item.strip().lower()
        for item in str(probe.get("format", {}).get("format_name") or "").split(",")
        if item.strip()
    }
    if not format_names.intersection(_CONTAINER_FORMATS[policy.container]):
        raise ValueError("media container does not match the format policy")
    streams = probe.get("streams", [])
    video = [row for row in streams if row.get("codec_type") == "video"]
    audio = [row for row in streams if row.get("codec_type") == "audio"]
    if (len(video) != 1 or len(audio) != 1
            or video[0].get("codec_name") != _PROBE_CODEC[policy.video_codec]
            or audio[0].get("codec_name") != _PROBE_CODEC[policy.audio_codec]
            or not 0 < video[0].get("height", 0) <= policy.maximum_height):
        raise ValueError("media streams do not match the format policy")
    measured = float(probe.get("format", {}).get("duration", 0))
    if (not math.isfinite(expected_duration) or expected_duration <= 0
            or not math.isfinite(measured)
            or abs(measured - expected_duration) > max(2.0, expected_duration * 0.001)):
        raise ValueError("media duration does not match source metadata")
