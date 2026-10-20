"""Offline compatibility tests for the project-specific intake wrapper."""
import json
from pathlib import Path

import pytest

from latent_timing_duplex.data import podcasts

SOURCE = {
    "source_id": "youtube-kWPvdAZGd3A",
    "url": "https://www.youtube.com/watch?v=kWPvdAZGd3A",
    "expected_language": "en-US",
}
ORIGINAL = {
    "format_id": "251", "vcodec": "none", "acodec": "opus",
    "language": "en-US", "audio_channels": 2,
    "format_note": "English (US) original (default), medium",
    "url": "https://example.invalid/private-signed-url",
}


def metadata(*formats):
    return {"id": "kWPvdAZGd3A", "formats": list(formats), "license": None}


def test_registered_sources():
    path = Path(__file__).resolve().parents[1] / "configs/podcast_sources.json"
    assert len(podcasts.validate_sources(json.loads(path.read_text()))) == 3


@pytest.mark.parametrize("update", [
    {"url": "https://example.com/watch?v=kWPvdAZGd3A"},
    {"url": SOURCE["url"] + "&list=playlist"},
    {"url": "--exec=bad-command"}, {"source_id": "wrong"}, {"expected_language": ""},
])
def test_invalid_source(update):
    with pytest.raises(ValueError):
        podcasts.validate_sources({"schema": "podcast-sources/v1", "sources": [SOURCE | update]})


def test_duplicate_source():
    with pytest.raises(ValueError):
        podcasts.validate_sources({"schema": "podcast-sources/v1", "sources": [SOURCE, SOURCE]})


def test_host_adds_policy_fields_but_generic_summary_omits_signed_urls():
    result = podcasts.summarize_metadata(SOURCE, metadata(
        ORIGINAL, ORIGINAL | {"format_note": "English dubbed-auto"},
        ORIGINAL | {"language": "fr-FR"}, ORIGINAL | {"vcodec": "avc1"},
    ))
    assert len(result["original_audio_formats"]) == 1
    assert "url" not in result["original_audio_formats"][0]
    assert result["rights_status"] == "unreviewed"
    assert result["speaker_channels_status"] == "unaudited"
    assert result["pilot_eligible"] is False


def test_wrong_video_rejected():
    with pytest.raises(ValueError): podcasts.summarize_metadata(SOURCE, {"id": "wrong"})


def test_inspection_delegates_metadata_only(monkeypatch):
    calls = []
    class Adapter:
        def metadata(self, source, destination, log):
            calls.append((source, destination, log))
            destination.write_text(json.dumps(metadata(ORIGINAL)))
            return metadata(ORIGINAL)
    monkeypatch.setattr(podcasts, "YtDlpAdapter", Adapter)
    result = podcasts.inspect_source(SOURCE)
    assert result["source_id"] == SOURCE["source_id"] and len(calls) == 1
