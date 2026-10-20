"""Validated inputs and deliberately narrow acquisition policies."""
from __future__ import annotations

from dataclasses import dataclass, field
import re
from urllib.parse import urlsplit

_VIDEO_ID = re.compile(r"[A-Za-z0-9_-]{11}")
_SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}")
_LANGUAGE = re.compile(r"[A-Za-z]{2,8}(?:-[A-Za-z0-9]{1,8})*")


@dataclass(frozen=True)
class YouTubeSource:
    source_id: str
    url: str
    language: str

    def __post_init__(self) -> None:
        if not _SAFE_ID.fullmatch(self.source_id):
            raise ValueError("source_id must be a safe identifier")
        if not _LANGUAGE.fullmatch(self.language):
            raise ValueError("language must be a simple BCP-47 language tag")
        parts = urlsplit(self.url)
        if (parts.scheme, parts.netloc, parts.path) != ("https", "www.youtube.com", "/watch"):
            raise ValueError("use a canonical HTTPS www.youtube.com watch URL")
        if parts.fragment or not parts.query.startswith("v=") or "&" in parts.query:
            raise ValueError("YouTube URL must contain only a video ID")
        if not _VIDEO_ID.fullmatch(parts.query[2:]):
            raise ValueError("invalid YouTube video ID")
        if self.source_id != "youtube-" + parts.query[2:]:
            raise ValueError("source_id must be youtube-<video-id>")

    @property
    def media_id(self) -> str:
        return self.url.rsplit("=", 1)[1]


@dataclass(frozen=True)
class MediaFormatPolicy:
    maximum_height: int = 1080
    language: str = "en-US"
    video_codec: str = "h264"
    audio_codec: str = "aac"
    container: str = "mp4"

    def __post_init__(self) -> None:
        if isinstance(self.maximum_height, bool) or not isinstance(self.maximum_height, int):
            raise ValueError("maximum_height must be an integer")
        if not 144 <= self.maximum_height <= 4320:
            raise ValueError("maximum_height must be between 144 and 4320")
        if not _LANGUAGE.fullmatch(self.language):
            raise ValueError("language must be a simple BCP-47 language tag")
        combinations = {
            ("h264", "aac", "mp4"),
            ("vp9", "opus", "webm"),
            ("av1", "opus", "webm"),
            ("h264", "aac", "mkv"),
            ("vp9", "opus", "mkv"),
            ("av1", "opus", "mkv"),
        }
        if (self.video_codec, self.audio_codec, self.container) not in combinations:
            raise ValueError("unsupported video/audio/container policy combination")


@dataclass(frozen=True)
class CaptionPolicy:
    language: str = "en-US"
    publisher: bool = True
    automatic_original: bool = True
    automatic_translations: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not _LANGUAGE.fullmatch(self.language):
            raise ValueError("language must be a simple BCP-47 language tag")
        if type(self.publisher) is not bool or type(self.automatic_original) is not bool:
            raise ValueError("caption selection flags must be booleans")
        if isinstance(self.automatic_translations, (str, bytes)):
            raise ValueError("automatic_translations must be a sequence of language tags")
        try:
            translations = tuple(self.automatic_translations)
        except TypeError as exc:
            raise ValueError("automatic_translations must be a sequence of language tags") from exc
        if any(not _LANGUAGE.fullmatch(tag) for tag in translations):
            raise ValueError("translation languages must be simple BCP-47 tags")
        if len(set(translations)) != len(translations):
            raise ValueError("translation languages must be unique")
        object.__setattr__(self, "automatic_translations", translations)


@dataclass(frozen=True)
class CaptionTrack:
    kind: str
    language_key: str
