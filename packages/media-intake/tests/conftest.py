import json
from pathlib import Path

import pytest

from media_intake import YouTubeSource

SOURCE = YouTubeSource("youtube-kWPvdAZGd3A", "https://www.youtube.com/watch?v=kWPvdAZGd3A", "en-US")
AUDIO = {"format_id": "140-7", "vcodec": "none", "acodec": "mp4a.40.2", "ext": "m4a",
         "language": "en-US", "format_note": "English original", "abr": 129}
VIDEO = {"format_id": "137", "vcodec": "avc1.640028", "acodec": "none", "ext": "mp4",
         "height": 1080}
META = {"id": "kWPvdAZGd3A", "duration": 10, "format_id": "137+140-7",
        "formats": [VIDEO, AUDIO], "automatic_captions": {"en-orig": [{"ext": "vtt"}]}}
PROBE = {"streams": [{"codec_type": "video", "codec_name": "h264", "height": 1080},
                     {"codec_type": "audio", "codec_name": "aac", "channels": 2}],
         "format": {"duration": "10.01", "format_name": "mov,mp4,m4a,3gp,3g2,mj2"}}
VTT = "WEBVTT\n\n00:00:01.000 --> 00:00:02.000\nExample\n"


class FakeYtDlp:
    def __init__(self, metadata=None):
        self.value = metadata or META
        self.calls = []
        self.caption_failure = False

    def metadata(self, source, destination, log):
        self.calls.append("metadata")
        destination.write_text(json.dumps(self.value))
        return self.value

    def download_media(self, source, metadata_path, output_root, policy, audio_id, log):
        self.calls.append("media")
        (output_root / f"{source.media_id}.{policy.container}").write_bytes(b"media")
        info = output_root / f"{source.media_id}.info.json"
        if not info.exists():
            info.write_text(json.dumps(self.value))

    def download_caption(self, source, metadata_path, output_dir, track, log):
        self.calls.append(("caption", track.kind, track.language_key))
        if self.caption_failure:
            raise RuntimeError("offline failure")
        path = output_dir / f"{source.media_id}.{track.language_key}.vtt"
        path.write_text(VTT)
        return path

    def version(self): return "fake-yt-dlp"


class FakeProbe:
    def __init__(self): self.calls = 0
    def inspect(self, media, destination, log):
        self.calls += 1
        destination.write_text(json.dumps(PROBE))
        return PROBE
    def version(self): return "fake-ffprobe"


@pytest.fixture
def source(): return SOURCE
