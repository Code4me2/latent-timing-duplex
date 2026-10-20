"""Compatibility tests: generic behavior is tested in packages/media-intake/tests."""
import json
from pathlib import Path

import pytest

from latent_timing_duplex.data import podcast_download as download
from media_intake.locking import OutputLock

SOURCE = {"source_id": "youtube-kWPvdAZGd3A",
          "url": "https://www.youtube.com/watch?v=kWPvdAZGd3A",
          "expected_language": "en-US", "rights_status": "user_confirmed"}
AUDIO = {"format_id": "140-7", "vcodec": "none", "acodec": "mp4a.40.2",
         "ext": "m4a", "language": "en-US", "format_note": "English original", "abr": 129}
VIDEO = {"format_id": "137", "vcodec": "avc1", "acodec": "none", "ext": "mp4", "height": 1080}
META = {"id": "kWPvdAZGd3A", "duration": 10, "format_id": "137+140-7",
        "formats": [VIDEO, AUDIO], "automatic_captions": {"en-orig": [{"ext": "vtt"}]}}
PROBE = {"streams": [{"codec_type": "video", "codec_name": "h264", "height": 1080},
                     {"codec_type": "audio", "codec_name": "aac", "channels": 2}],
         "format": {"duration": "10.01", "format_name": "mov,mp4,m4a,3gp,3g2,mj2"}}
VTT = "WEBVTT\n\n00:00:01.000 --> 00:00:02.000\nExample\n"


def test_legacy_helpers_keep_original_audio_and_caption_contract():
    assert download.CANONICAL_LOCK_NAME == ".media-intake.lock"
    assert download.select_audio(SOURCE, META) == "140-7"
    with pytest.raises(ValueError):
        download.select_audio(SOURCE, META | {"formats": [VIDEO, AUDIO | {"format_note": "dubbed"}]})
    assert download.caption_tracks(META, "en-US") == [("automatic", "en-orig")]
    assert download.caption_tracks({"automatic_captions": {"en": [{}]}}, "en-US") == []
    download.verify_original_selection(SOURCE, META)
    download.verify_probe(PROBE, 10)


def test_compatibility_wrapper_uses_legacy_schemas_and_host_policy(tmp_path, monkeypatch):
    calls = []
    class Client:
        def acquire(self, source, root, media_policy, caption_policy, **kwargs):
            calls.append(kwargs)
            kwargs["attempt_dir"].mkdir(parents=True, exist_ok=True)
            (kwargs["attempt_dir"] / "source.info.json").write_text(json.dumps(META))
            return {"schema": kwargs["record_schema"], "source_id": source.source_id,
                    "status": "ok", "media": {"path": "kWPvdAZGd3A.mp4"}, "captions": []}
    monkeypatch.setattr(download, "MediaIntake", Client)
    work = tmp_path / "acquisition-runs" / "run" / SOURCE["source_id"]
    result = download.acquire(SOURCE, tmp_path, work, {"yt-dlp": "ignored"}, False)
    assert calls[0]["receipt_schema"] == "podcast-media-receipt/v1"
    assert calls[0]["record_schema"] == "podcast-acquisition/v1"
    assert calls[0]["caption_parent"] == ""
    assert result["pilot_eligible"] is False
    persisted = json.loads((work / "record.json").read_text())
    assert persisted["rights_status"] == "user_confirmed"


def test_compatibility_cli_is_blocked_by_canonical_standalone_lock(tmp_path, monkeypatch):
    root = tmp_path / "root"; root.mkdir()
    config = tmp_path / "sources.json"
    config.write_text(json.dumps({"schema": "podcast-sources/v1", "sources": [SOURCE]}))
    monkeypatch.setattr(download, "_inside_git", lambda path: False)
    monkeypatch.setattr(download.shutil, "which", lambda name: f"/bin/{name}")
    with OutputLock(root / download.CANONICAL_LOCK_NAME):
        with pytest.raises(RuntimeError, match="another acquisition"):
            download.main(["--sources", str(config), "--output-dir", str(root)])


def test_host_generated_run_rejects_symlinked_parent(tmp_path, monkeypatch):
    root = tmp_path / "root"; root.mkdir()
    elsewhere = tmp_path / "elsewhere"; elsewhere.mkdir()
    (root / "acquisition-runs").symlink_to(elsewhere, target_is_directory=True)
    config = tmp_path / "sources.json"
    config.write_text(json.dumps({"schema": "podcast-sources/v1", "sources": [SOURCE]}))
    monkeypatch.setattr(download, "_inside_git", lambda path: False)
    monkeypatch.setattr(download.shutil, "which", lambda name: f"/bin/{name}")
    with pytest.raises(ValueError, match="symlink"):
        download.main(["--sources", str(config), "--output-dir", str(root)])
    assert not list(elsewhere.iterdir())


def test_real_generic_flow_writes_and_reuses_legacy_receipt(tmp_path, monkeypatch):
    generic = download.MediaIntake

    class Adapter:
        def metadata(self, source, destination, log):
            destination.write_text(json.dumps(META))
            return META
        def download_media(self, source, metadata_path, output_root, policy, audio_id, log):
            (output_root / f"{source.media_id}.mp4").write_bytes(b"media")
            (output_root / f"{source.media_id}.info.json").write_text(json.dumps(META))
        def download_caption(self, source, metadata_path, output_dir, track, log):
            path = output_dir / f"{source.media_id}.{track.language_key}.vtt"
            path.write_text(VTT)
            return path
        def version(self): return "fake-yt-dlp"
        def ffmpeg_version(self): return "fake-ffmpeg"

    class Probe:
        def inspect(self, media, destination, log):
            destination.write_text(json.dumps(PROBE))
            return PROBE
        def version(self): return "fake-ffprobe"

    monkeypatch.setattr(download, "MediaIntake", lambda: generic(yt_dlp=Adapter(), ffprobe=Probe()))
    first_work = tmp_path / "acquisition-runs" / "first" / SOURCE["source_id"]
    second_work = tmp_path / "acquisition-runs" / "second" / SOURCE["source_id"]
    first = download.acquire(SOURCE, tmp_path, first_work, {}, False)
    second = download.acquire(SOURCE, tmp_path, second_work, {}, False)
    receipt = json.loads((tmp_path / "kWPvdAZGd3A.media-receipt.json").read_text())
    assert receipt["schema"] == "podcast-media-receipt/v1"
    assert first["media_reused"] is False and second["media_reused"] is True
    assert json.loads((second_work / "record.json").read_text())["source"]["url"] == SOURCE["url"]


def test_legacy_logged_runner_is_shell_free(tmp_path, monkeypatch):
    seen = []
    def run(command, **kwargs):
        seen.append(kwargs)
    monkeypatch.setattr(download.subprocess, "run", run)
    download.run_logged(["tool", "--arg"], tmp_path / "log")
    assert seen[0]["shell"] is False
