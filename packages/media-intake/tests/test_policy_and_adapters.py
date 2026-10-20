import json
from pathlib import Path
import subprocess

import pytest

from media_intake import CaptionPolicy, CaptionTrack, MediaFormatPolicy, YouTubeSource
from media_intake.adapters.ffprobe import validate_probe
from media_intake.adapters.yt_dlp import YtDlpAdapter, select_original_audio
from media_intake.config import load_config
from media_intake.runner import CommandResult, SubprocessRunner
from conftest import AUDIO, META, PROBE, SOURCE


@pytest.mark.parametrize("kwargs", [
    {"maximum_height": 0}, {"maximum_height": True}, {"language": "../en"},
    {"video_codec": "anything"}, {"container": "mp4", "audio_codec": "opus"},
])
def test_media_policy_rejects_unvalidated_expressions(kwargs):
    with pytest.raises(ValueError): MediaFormatPolicy(**kwargs)


def test_caption_translation_is_explicit_and_validated():
    assert CaptionPolicy().automatic_translations == ()
    assert CaptionPolicy(automatic_translations=("fr",)).automatic_translations == ("fr",)
    with pytest.raises(ValueError): CaptionPolicy(automatic_translations=("../fr",))


@pytest.mark.parametrize("kwargs", [
    {"publisher": "false"}, {"publisher": 1},
    {"automatic_original": None}, {"automatic_translations": "fr"},
])
def test_caption_policy_rejects_non_boolean_flags_and_malformed_translations(kwargs):
    with pytest.raises(ValueError):
        CaptionPolicy(**kwargs)


@pytest.mark.parametrize("source_id,url", [
    ("../bad", SOURCE.url),
    (SOURCE.source_id, "https://www.youtube.com/watch?v=kWPvdAZGd3A&list=bad"),
    (SOURCE.source_id, "--exec=bad"),
])
def test_source_validation_blocks_injection_and_traversal(source_id, url):
    with pytest.raises(ValueError): YouTubeSource(source_id, url, "en-US")


class RecordingRunner:
    def __init__(self): self.calls = []
    def run(self, command, *, timeout, log=None):
        self.calls.append((list(command), timeout, log))
        return CommandResult(b'{"id":"kWPvdAZGd3A"}', b"")


def test_metadata_adapter_delegates_shell_safe_arguments(tmp_path):
    runner = RecordingRunner()
    adapter = YtDlpAdapter(runner)
    adapter.metadata(SOURCE, tmp_path / "info.json", tmp_path / "log")
    command = runner.calls[0][0]
    assert command[-1] == SOURCE.url
    assert "--skip-download" in command and "--ignore-config" in command
    assert all(isinstance(item, str) for item in command)


def test_media_and_caption_adapter_arguments_and_output_names(tmp_path):
    runner = RecordingRunner()
    adapter = YtDlpAdapter(runner)
    policy = MediaFormatPolicy(language="en-US")
    metadata = tmp_path / "source.info.json"
    adapter.download_media(SOURCE, metadata, tmp_path, policy, "140-7", tmp_path / "media.log")
    media_command = runner.calls[0][0]
    assert media_command[media_command.index("-f") + 1] == "bv[ext=mp4][vcodec^=avc1][height<=1080]+140-7"
    assert media_command[media_command.index("--merge-output-format") + 1] == "mp4"
    assert media_command[media_command.index("-o") + 1] == str(tmp_path / "%(id)s.%(ext)s")
    track = CaptionTrack("automatic", "en-orig")
    output = adapter.download_caption(SOURCE, metadata, tmp_path, track, tmp_path / "caption.log")
    caption_command = runner.calls[1][0]
    assert "--write-auto-subs" in caption_command and "--skip-download" in caption_command
    assert caption_command[caption_command.index("--sub-langs") + 1] == "en-orig"
    assert output == tmp_path / "kWPvdAZGd3A.en-orig.vtt"


def test_probe_rejects_missing_or_mismatched_container():
    policy = MediaFormatPolicy(language="en-US")
    validate_probe(PROBE, 10, policy)
    for name in (None, "matroska,webm"):
        changed = PROBE | {"format": {"duration": "10.01", "format_name": name}}
        with pytest.raises(ValueError, match="container"):
            validate_probe(changed, 10, policy)


def test_original_language_rejects_dub_and_wrong_language():
    policy = MediaFormatPolicy(language="en-US")
    assert select_original_audio(SOURCE, META, policy) == "140-7"
    for changed in [AUDIO | {"format_note": "English dubbed"}, AUDIO | {"language": "fr"}]:
        with pytest.raises(ValueError):
            select_original_audio(SOURCE, META | {"formats": [changed]}, policy)


@pytest.mark.parametrize("document", [
    {"schema": "media-intake-sources/v1", "sources": [{}]},
    {"schema": "media-intake-sources/v1", "sources": [1]},
    {"schema": "media-intake-sources/v1", "sources": [{
        "source_id": SOURCE.source_id, "url": SOURCE.url, "language": "en-US",
    }], "policies": []},
    {"schema": "media-intake-sources/v1", "sources": [{
        "source_id": SOURCE.source_id, "url": SOURCE.url, "language": "en-US",
    }], "policies": {"captions": {"publisher": "false"}}},
])
def test_load_config_reports_malformed_rows_and_policies_as_value_error(tmp_path, document):
    path = tmp_path / "sources.json"
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError):
        load_config(path)


def test_runner_retries_bounded_and_never_uses_shell(monkeypatch):
    calls = []
    def run(*args, **kwargs):
        calls.append((args, kwargs))
        if len(calls) == 1: raise subprocess.CalledProcessError(1, args[0])
        return subprocess.CompletedProcess(args[0], 0, b"ok", b"")
    monkeypatch.setattr(subprocess, "run", run)
    result = SubprocessRunner(attempts=2, retry_delay=0).run(["tool", "arg"], timeout=1)
    assert result.stdout == b"ok" and len(calls) == 2
    assert all(call[1]["shell"] is False for call in calls)
