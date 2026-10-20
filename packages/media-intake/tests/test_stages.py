import json
from pathlib import Path

import pytest

from media_intake import CaptionPolicy, MediaFormatPolicy, MediaIntake
from conftest import FakeProbe, FakeYtDlp, META, PROBE, SOURCE


class BombProbe:
    def inspect(self, *args, **kwargs): raise AssertionError("ffprobe must not run")


def test_metadata_only_needs_no_media_or_probe(tmp_path):
    yt = FakeYtDlp()
    result = MediaIntake(yt_dlp=yt, ffprobe=BombProbe()).inspect_metadata(SOURCE)
    assert result["source_id"] == SOURCE.source_id
    assert yt.calls == ["metadata"]
    assert not list(tmp_path.iterdir())


def test_captions_only_needs_no_mp4_or_probe(tmp_path):
    yt = FakeYtDlp()
    result = MediaIntake(yt_dlp=yt, ffprobe=BombProbe()).acquire_captions(
        SOURCE, tmp_path, CaptionPolicy(language="en-US"))
    assert result["status"] == "ok"
    assert not list(tmp_path.glob("*.mp4"))
    assert ("caption", "automatic", "en-orig") in yt.calls


def test_new_download_and_verified_reuse_refresh_captions(tmp_path):
    yt, probe = FakeYtDlp(), FakeProbe()
    client = MediaIntake(yt_dlp=yt, ffprobe=probe)
    first = client.acquire(SOURCE, tmp_path, MediaFormatPolicy(language="en-US"),
                           CaptionPolicy(language="en-US"))
    second = client.acquire(SOURCE, tmp_path, MediaFormatPolicy(language="en-US"),
                            CaptionPolicy(language="en-US"))
    assert first["status"] == second["status"] == "ok"
    assert yt.calls.count("media") == 1
    assert len([row for row in yt.calls if isinstance(row, tuple)]) == 2
    assert second["media_reused"] is True


def test_explicit_legacy_adoption_preserves_unknown_download_time(tmp_path):
    (tmp_path / "kWPvdAZGd3A.mp4").write_bytes(b"legacy")
    (tmp_path / "kWPvdAZGd3A.info.json").write_text(json.dumps(META))
    client = MediaIntake(yt_dlp=FakeYtDlp(), ffprobe=FakeProbe())
    with pytest.raises(ValueError, match="adoption"):
        client.acquire_media(SOURCE, tmp_path, MediaFormatPolicy(language="en-US"))
    result = client.acquire_media(SOURCE, tmp_path, MediaFormatPolicy(language="en-US"),
                                  adopt_existing=True)
    receipt = json.loads((tmp_path / "kWPvdAZGd3A.media-receipt.json").read_text())
    assert result["media_reused"] and receipt["acquisition_time"] is None


def test_adoption_rejects_mismatched_container_even_when_streams_match(tmp_path):
    (tmp_path / "kWPvdAZGd3A.mp4").write_bytes(b"legacy")
    (tmp_path / "kWPvdAZGd3A.info.json").write_text(json.dumps(META))

    class MatroskaProbe(FakeProbe):
        def inspect(self, media, destination, log):
            value = PROBE | {"format": {"duration": "10.01", "format_name": "matroska,webm"}}
            destination.write_text(json.dumps(value))
            return value

    with pytest.raises(ValueError, match="container"):
        MediaIntake(yt_dlp=FakeYtDlp(), ffprobe=MatroskaProbe()).acquire_media(
            SOURCE, tmp_path, MediaFormatPolicy(language="en-US"), adopt_existing=True)
    assert not (tmp_path / "kWPvdAZGd3A.media-receipt.json").exists()


def test_receipt_detects_changed_media(tmp_path):
    client = MediaIntake(yt_dlp=FakeYtDlp(), ffprobe=FakeProbe())
    client.acquire_media(SOURCE, tmp_path, MediaFormatPolicy(language="en-US"))
    (tmp_path / "kWPvdAZGd3A.mp4").write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed"):
        client.acquire_media(SOURCE, tmp_path, MediaFormatPolicy(language="en-US"))


def test_media_failure_propagates_and_leaves_failure_status(tmp_path):
    class BrokenMedia(FakeYtDlp):
        def download_media(self, *args, **kwargs):
            raise RuntimeError("media failed")
    client = MediaIntake(yt_dlp=BrokenMedia(), ffprobe=FakeProbe())
    with pytest.raises(RuntimeError, match="media failed"):
        client.acquire_media(SOURCE, tmp_path, MediaFormatPolicy(language="en-US"))
    records = list(tmp_path.glob("attempts/*/*/media-record.json"))
    assert len(records) == 1 and json.loads(records[0].read_text())["status"] == "failed"


def test_caption_failure_status_and_later_isolated_retry(tmp_path):
    yt = FakeYtDlp(); yt.caption_failure = True
    client = MediaIntake(yt_dlp=yt, ffprobe=BombProbe())
    failed = client.acquire_captions(SOURCE, tmp_path, CaptionPolicy(language="en-US"))
    assert failed["status"] == "partial"
    assert any(row["status"] == "failed" for row in failed["captions"])
    yt.caption_failure = False
    assert client.acquire_captions(SOURCE, tmp_path, CaptionPolicy(language="en-US"))["status"] == "ok"


def test_auto_translation_not_selected_unless_requested(tmp_path):
    metadata = META | {"automatic_captions": {"en": [{}], "fr": [{}]}}
    yt = FakeYtDlp(metadata)
    first = MediaIntake(yt_dlp=yt, ffprobe=BombProbe()).acquire_captions(
        SOURCE, tmp_path, CaptionPolicy(language="en-US"))
    assert all(row.get("kind") != "automatic_translation" for row in first["captions"])
    second = MediaIntake(yt_dlp=yt, ffprobe=BombProbe()).acquire_captions(
        SOURCE, tmp_path, CaptionPolicy(language="en-US", automatic_translations=("fr",)))
    assert any(row.get("kind") == "automatic_translation" and row["status"] == "downloaded"
               for row in second["captions"])


def test_symlink_output_and_artifacts_are_rejected(tmp_path):
    real = tmp_path / "real"; real.mkdir()
    link = tmp_path / "link"; link.symlink_to(real, target_is_directory=True)
    client = MediaIntake(yt_dlp=FakeYtDlp(), ffprobe=FakeProbe())
    with pytest.raises(ValueError, match="symlink"):
        client.acquire_media(SOURCE, link, MediaFormatPolicy(language="en-US"))
    target = tmp_path / "target"; target.write_bytes(b"x")
    (real / "kWPvdAZGd3A.mp4").symlink_to(target)
    with pytest.raises(ValueError, match="symlink"):
        client.acquire_media(SOURCE, real, MediaFormatPolicy(language="en-US"), adopt_existing=True)


def test_symlinked_attempt_parent_is_rejected(tmp_path):
    root = tmp_path / "root"; root.mkdir()
    elsewhere = tmp_path / "elsewhere"; elsewhere.mkdir()
    (root / "attempts").symlink_to(elsewhere, target_is_directory=True)
    client = MediaIntake(yt_dlp=FakeYtDlp(), ffprobe=FakeProbe())
    attempt = root / "attempts" / "run" / SOURCE.source_id
    with pytest.raises(ValueError, match="symlink"):
        client.acquire_media(SOURCE, root, MediaFormatPolicy(language="en-US"),
                             attempt_dir=attempt)
    with pytest.raises(ValueError, match="symlink"):
        client.acquire_media(SOURCE, root, MediaFormatPolicy(language="en-US"))
    assert not list(elsewhere.iterdir())


def test_stale_sidecar_without_media_is_never_reused_for_new_download(tmp_path):
    stale = META | {
        "format_id": "399+140-7",
        "formats": [
            {"format_id": "399", "vcodec": "av01", "acodec": "none", "ext": "webm",
             "height": 1080},
            META["formats"][1],
        ],
    }
    (tmp_path / "kWPvdAZGd3A.info.json").write_text(json.dumps(stale))
    yt = FakeYtDlp()
    with pytest.raises(ValueError, match="sidecar"):
        MediaIntake(yt_dlp=yt, ffprobe=FakeProbe()).acquire_media(
            SOURCE, tmp_path, MediaFormatPolicy(language="en-US"))
    assert "media" not in yt.calls
    assert not (tmp_path / "kWPvdAZGd3A.mp4").exists()


def test_records_include_source_and_normalized_policies(tmp_path):
    result = MediaIntake(yt_dlp=FakeYtDlp(), ffprobe=FakeProbe()).acquire(
        SOURCE, tmp_path, MediaFormatPolicy(language="en-US"),
        CaptionPolicy(language="en-US", automatic_translations=("fr",)))
    assert result["source"]["url"] == SOURCE.url
    assert result["media_policy"]["container"] == "mp4"
    assert result["caption_policy"]["automatic_translations"] == ["fr"]
    receipt = json.loads((tmp_path / "kWPvdAZGd3A.media-receipt.json").read_text())
    assert receipt["source"]["language"] == "en-US"
    assert receipt["media_policy"]["video_codec"] == "h264"


def test_receipt_schema_must_match_calling_compatibility_surface(tmp_path):
    client = MediaIntake(yt_dlp=FakeYtDlp(), ffprobe=FakeProbe())
    client.acquire_media(SOURCE, tmp_path, MediaFormatPolicy(language="en-US"))
    with pytest.raises(ValueError, match="schema mismatch"):
        client.acquire_media(SOURCE, tmp_path, MediaFormatPolicy(language="en-US"),
                             receipt_schema="podcast-media-receipt/v1")
