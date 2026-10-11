"""Security and media-boundary checks for the semantic-silence pilot."""

from __future__ import annotations

import hashlib
import json
import wave
from collections import Counter
from pathlib import Path

import pytest

from experiments.silence_graft import render_clips as renderer_module
from experiments.silence_graft.build_assignments import _sentinels, build_assignments, release
from experiments.silence_graft.receipts import receipt_bytes, sign_receipt, verify_receipt
from experiments.silence_graft.render_clips import render_clips

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "silence.yaml"
PROTOCOL = ROOT / "docs" / "DATA_LABELING_PROTOCOL.md"
KEY_BYTES = b"pilot-test-key-material-is-at-least-32-bytes"


def _pcm24(value: int) -> bytes:
    return value.to_bytes(3, "little", signed=True)


def _write_wav(path: Path, *, channels: int = 2) -> None:
    frames = b"".join(
        b"".join(_pcm24(value) for value in ((100 + frame, -100 - frame) if channels == 2 else (100 + frame,)))
        for frame in range(8)
    )
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(3)
        handle.setframerate(4)
        handle.writeframes(frames)


def _jsonl(rows) -> bytes:
    return ("\n".join(json.dumps(row, sort_keys=True, separators=(",", ":")) for row in rows) + "\n").encode()


def _rows(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _inputs(tmp_path: Path, *, media_updates=None, manifest_updates=None, key_bytes=KEY_BYTES):
    tmp_path.mkdir(parents=True, exist_ok=True)
    wav = tmp_path / "private.wav"
    _write_wav(wav)
    manifest_rows = [
        {
            "context": "joint_silence", "elapsed_s": 0.48,
            "episode_id": "source-session:episode-1", "lineage_id": "lineage-1",
            "model_cutoff_s": 0.625, "onset_s": 0.145, "right_censored": False,
            "session_id": "source-session", "silence_end_s": 1.75,
            "source": "test-corpus", "split": "train",
        },
        {
            "context": "joint_silence", "elapsed_s": 0.48,
            "episode_id": "source-session:episode-2", "lineage_id": "lineage-1",
            "model_cutoff_s": 1.125, "onset_s": 0.645, "right_censored": False,
            "session_id": "source-session", "silence_end_s": 1.75,
            "source": "test-corpus", "split": "development",
        },
    ]
    if manifest_updates:
        manifest_rows[0].update(manifest_updates)
    manifest_payload = _jsonl(manifest_rows)
    manifest = tmp_path / "manifest.pilot.jsonl"
    manifest.write_bytes(manifest_payload)
    lock = tmp_path / "manifest.jsonl.lock.json"
    lock.write_text(json.dumps({
        "schema": "semantic-silence-config/v1",
        "protocol_version": "semantic-silence-pilot/v1",
        "label_schema": "semantic-silence-labels/v1",
        "sha256": {
            "pilot_manifest": hashlib.sha256(manifest_payload).hexdigest(),
            "config": hashlib.sha256(CONFIG.read_bytes()).hexdigest(),
            "protocol_doc": hashlib.sha256(PROTOCOL.read_bytes()).hexdigest(),
        },
    }), encoding="utf-8")
    media_row = {
        "assistant_channel": 0, "session_id": "source-session", "user_channel": 1,
        "wav_path": str(wav.resolve()), "wav_sha256": hashlib.sha256(wav.read_bytes()).hexdigest(),
    }
    if media_updates:
        media_row.update(media_updates)
    media = tmp_path / "media.jsonl"
    media.write_bytes(_jsonl([media_row]))
    key = tmp_path / "hmac.key"
    key.write_bytes(key_bytes)
    key.chmod(0o600)
    return manifest, lock, media, key, wav


def _render(tmp_path: Path):
    manifest, lock, media, key, wav = _inputs(tmp_path)
    prefix, coordinator = render_clips(
        manifest, lock, media, tmp_path / "prefix-package", tmp_path / "coordinator-package", key
    )
    return prefix, coordinator, key, wav


def _audition(coordinator: Path, key: Path, path: Path, *, session_updates=None, sentinel_payload=None) -> Path:
    index = coordinator / "coordinator_media_index.jsonl"
    source_map = coordinator / "private_source_mapping.jsonl"
    renderer_receipt = coordinator / "renderer_receipt.json"
    source = _rows(source_map)[0]
    session = {
        "assistant_channel": source["assistant_channel"],
        "assistant_channel_auditioned": True,
        "session_id": source["session_id"],
        "user_channel": source["user_channel"],
        "user_channel_auditioned": True,
        "wav_sha256": source["wav_sha256"],
    }
    if session_updates:
        session.update(session_updates)
    hashes = {
        "config": hashlib.sha256(CONFIG.read_bytes()).hexdigest(),
        "coordinator_media_index": hashlib.sha256(index.read_bytes()).hexdigest(),
        "private_source_mapping": hashlib.sha256(source_map.read_bytes()).hexdigest(),
        "renderer_receipt": hashlib.sha256(renderer_receipt.read_bytes()).hexdigest(),
    }
    if sentinel_payload is not None:
        hashes["sentinel_list"] = hashlib.sha256(sentinel_payload).hexdigest()
    signed = sign_receipt({
        "audited_at": "2026-10-09T12:00:00Z",
        "auditor_identity": "auditor-1",
        "protocol_version": "semantic-silence-pilot/v1",
        "schema": "semantic-silence-channel-audition/v1",
        "sessions": [session],
        "sha256": hashes,
    }, key.read_bytes())
    path.write_bytes(receipt_bytes(signed))
    return path


def _assignment_inputs(tmp_path: Path, *, audition_updates=None):
    prefix, coordinator, key, _ = _render(tmp_path)
    annotators = tmp_path / "annotators.json"
    annotators.write_text(json.dumps(["alice", "bob", "carol"]), encoding="utf-8")
    audition = _audition(coordinator, key, tmp_path / "audition.json", session_updates=audition_updates)
    return prefix, coordinator, annotators, key, audition


def _build_assignments(tmp_path: Path) -> tuple[Path, Path]:
    prefix, coordinator, annotators, key, audition = _assignment_inputs(tmp_path)
    return build_assignments(
        prefix, coordinator, annotators, key, audition,
        tmp_path / "annotator-assignments", tmp_path / "coordinator-assignments",
    )


def _queue_paths(root: Path) -> list[Path]:
    return sorted(root.glob("annotators/*/queue.jsonl"))


def _completion_files(tmp_path: Path, coordinator_assignments: Path, key: Path, rows):
    completions = tmp_path / "completions.jsonl"
    completions.write_bytes(_jsonl(rows))
    receipt = tmp_path / "completions.receipt.json"
    signed = sign_receipt({
        "schema": "semantic-silence-prefix-completion-receipt/v1",
        "protocol_version": "semantic-silence-pilot/v1",
        "sha256": {
            "assignment_receipt": hashlib.sha256(
                (coordinator_assignments / "assignment_receipt.json").read_bytes()
            ).hexdigest(),
            "completions": hashlib.sha256(completions.read_bytes()).hexdigest(),
        },
    }, key.read_bytes())
    receipt.write_bytes(receipt_bytes(signed))
    return completions, receipt


def test_renders_separate_frame_exact_signed_packages(tmp_path: Path) -> None:
    prefix, coordinator, key, _ = _render(tmp_path)
    assert (prefix / "READY").is_file() and (coordinator / "READY").is_file()
    assert not list(prefix.rglob("*reveal*"))
    assert {path.name for path in prefix.iterdir()} == {"prefix", "READY"}

    mapping = _rows(coordinator / "private_source_mapping.jsonl")
    item = mapping[0]["item_id"]
    with wave.open(str(prefix / "prefix" / f"{item}.wav"), "rb") as clip:
        assert (clip.getnchannels(), clip.getsampwidth(), clip.getframerate(), clip.getnframes()) == (2, 3, 4, 3)
        assert clip.readframes(3) == b"".join(_pcm24(-100 - frame) + _pcm24(100 + frame) for frame in range(3))
    with wave.open(str(coordinator / "reveal_user" / f"{item}.wav"), "rb") as clip:
        assert (clip.getnchannels(), clip.getnframes()) == (1, 5)
        assert clip.readframes(5) == b"".join(_pcm24(-100 - frame) for frame in range(3, 8))

    receipt = json.loads((coordinator / "renderer_receipt.json").read_text())
    verify_receipt(receipt, key.read_bytes(), "renderer receipt")
    assert receipt["key_fingerprint"] == hashlib.sha256(key.read_bytes()).hexdigest()
    assert receipt["sources"] == [{
        "assistant_channel": 0, "session_id": "source-session", "user_channel": 1,
        "wav_sha256": mapping[0]["wav_sha256"],
    }]
    assert mapping[0]["lineage_id"] == "lineage-1"


def test_media_hash_and_post_read_mutation_are_rejected(tmp_path: Path, monkeypatch) -> None:
    manifest, lock, media, key, wav = _inputs(tmp_path / "bad-hash", media_updates={"wav_sha256": "0" * 64})
    with pytest.raises(ValueError, match="source WAV SHA-256"):
        render_clips(manifest, lock, media, tmp_path / "p1", tmp_path / "c1", key)

    manifest, lock, media, key, wav = _inputs(tmp_path / "mutation")
    original = renderer_module._read_channels
    changed = False

    def mutate(*args, **kwargs):
        nonlocal changed
        result = original(*args, **kwargs)
        if not changed:
            with wav.open("ab") as handle:
                handle.write(b"x")
            changed = True
        return result

    monkeypatch.setattr(renderer_module, "_read_channels", mutate)
    with pytest.raises(ValueError, match="changed during rendering"):
        render_clips(manifest, lock, media, tmp_path / "p2", tmp_path / "c2", key)
    assert not (tmp_path / "p2").exists() and not (tmp_path / "c2").exists()


@pytest.mark.parametrize("missing", ["schema", "protocol_version", "label_schema", "pilot_manifest", "config", "protocol_doc"])
def test_lock_fields_and_digests_are_mandatory(tmp_path: Path, missing: str) -> None:
    manifest, lock, media, key, _ = _inputs(tmp_path)
    value = json.loads(lock.read_text())
    (value["sha256"] if missing in value["sha256"] else value).pop(missing)
    lock.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="lock receipt"):
        render_clips(manifest, lock, media, tmp_path / "prefix", tmp_path / "coordinator", key)


def test_exact_media_sessions_manifest_lineage_and_strong_key_are_required(tmp_path: Path) -> None:
    manifest, lock, media, key, _ = _inputs(tmp_path / "extra")
    row = json.loads(media.read_text().splitlines()[0])
    media.write_bytes(_jsonl([row, {**row, "session_id": "extra"}]))
    with pytest.raises(ValueError, match="non-pilot"):
        render_clips(manifest, lock, media, tmp_path / "p", tmp_path / "c", key)

    manifest, lock, media, key, _ = _inputs(tmp_path / "lineage", manifest_updates={"lineage_id": ""})
    with pytest.raises(ValueError, match="lineage_id"):
        render_clips(manifest, lock, media, tmp_path / "p2", tmp_path / "c2", key)

    manifest, lock, media, key, _ = _inputs(tmp_path / "weak", key_bytes=b"weak")
    with pytest.raises(ValueError, match="at least 32 bytes"):
        render_clips(manifest, lock, media, tmp_path / "p3", tmp_path / "c3", key)

    manifest, lock, media, key, _ = _inputs(tmp_path / "permissions")
    key.chmod(0o644)
    with pytest.raises(ValueError, match="group or other users"):
        render_clips(manifest, lock, media, tmp_path / "p4", tmp_path / "c4", key)


def test_renderer_rejects_elapsed_silence_beyond_prefix(tmp_path: Path) -> None:
    manifest, lock, media, key, _ = _inputs(
        tmp_path,
        manifest_updates={
            "elapsed_s": 20.5,
            "onset_s": 0.0,
            "model_cutoff_s": 20.5,
            "silence_end_s": 21.0,
        },
    )
    with pytest.raises(ValueError, match="exceeds the prefix duration"):
        render_clips(manifest, lock, media, tmp_path / "prefix", tmp_path / "coordinator", key)


def test_builds_private_qc_map_and_minimal_aliased_public_queues(tmp_path: Path) -> None:
    first_public, first_private = _build_assignments(tmp_path / "first")
    second_public, _ = _build_assignments(tmp_path / "second")
    assert [path.read_bytes() for path in _queue_paths(first_public)] == [
        path.read_bytes() for path in _queue_paths(second_public)
    ]
    private = _rows(first_private / "private_assignment_source_map.jsonl")
    originals = [row for row in private if not row["repeat"]]
    assert all(len({row["annotator_id"] for row in originals if row["item_id"] == item}) == 3 for item in {row["item_id"] for row in originals})
    loads = Counter(row["annotator_id"] for row in originals)
    assert max(loads.values()) - min(loads.values()) <= 1

    allowed = {"assignment_id", "prefix_clip", "presentation_id", "protocol_version", "schema"}
    public = [row for path in _queue_paths(first_public) for row in _rows(path)]
    assert all(set(row) == allowed for row in public)
    assert len({row["prefix_clip"] for row in public}) == len(public)
    assert all((path.parent / row["prefix_clip"]).is_file() for path in _queue_paths(first_public) for row in _rows(path))
    forbidden = ("item_id", "repeat", "sentinel", "annotator_id", "source", "episode_id", "reveal", "sha256")
    assert not any(term in json.dumps(public) for term in forbidden)
    assert {path.name for path in first_public.iterdir()} == {"annotators", "READY"}
    assert {path.name for path in first_private.iterdir()} == {
        "assignment_receipt.json", "coordinator_reveal_release_map.jsonl",
        "private_assignment_source_map.jsonl", "READY",
    }
    receipt = json.loads((first_private / "assignment_receipt.json").read_text())
    assert set(receipt["annotator_outputs"]) == {
        str(path.relative_to(first_public)) for path in first_public.rglob("*") if path.is_file() and path.name != "READY"
    }
    assert not list(first_private.rglob("*.wav"))
    for rows in ({row["annotator_id"]: [] for row in private},):
        for row in private:
            rows[row["annotator_id"]].append(row)
        for annotator_rows in rows.values():
            repeat = next(row for row in annotator_rows if row["repeat"])
            original = next(row for row in annotator_rows if row["item_id"] == repeat["item_id"] and not row["repeat"])
            assert repeat["prefix_alias"] != original["prefix_alias"]
            assert (first_public / repeat["prefix_alias"]).read_bytes() == (first_public / original["prefix_alias"]).read_bytes()


def test_assignment_requires_authenticated_exact_session_audition(tmp_path: Path) -> None:
    prefix, coordinator, annotators, key, audition = _assignment_inputs(
        tmp_path, audition_updates={"user_channel": 0, "assistant_channel": 1}
    )
    with pytest.raises(ValueError, match="mapping mismatch"):
        build_assignments(
            prefix, coordinator, annotators, key, audition,
            tmp_path / "annotator-assignments", tmp_path / "coordinator-assignments",
        )
    assert not (tmp_path / "annotator-assignments").exists()
    assert not (tmp_path / "coordinator-assignments").exists()

    audition = _audition(coordinator, key, tmp_path / "tampered.json")
    value = json.loads(audition.read_text())
    value["auditor_identity"] = "other"
    audition.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="HMAC verification"):
        build_assignments(
            prefix, coordinator, annotators, key, audition,
            tmp_path / "tampered-public", tmp_path / "tampered-private",
        )


def test_repeat_and_sentinel_config_are_enforced(tmp_path: Path) -> None:
    prefix, coordinator, annotators, key, audition = _assignment_inputs(tmp_path)
    bad_config = tmp_path / "bad.yaml"
    bad_config.write_text(CONFIG.read_text().replace("repeat_fraction: 0.05", "repeat_fraction: 0.10"), encoding="utf-8")
    with pytest.raises(ValueError, match="exactly 0.05"):
        build_assignments(
            prefix, coordinator, annotators, key, audition,
            tmp_path / "bad-repeat-public", tmp_path / "bad-repeat-private", bad_config,
        )
    sentinel = tmp_path / "sentinels.json"
    sentinel.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="forbidden"):
        build_assignments(
            prefix, coordinator, annotators, key, audition,
            tmp_path / "bad-sentinel-public", tmp_path / "bad-sentinel-private", sentinel_list=sentinel,
        )
    digest = hashlib.sha256(sentinel.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="required"):
        _sentinels(None, digest, {"sha256": {"sentinel_list": digest}})
    with pytest.raises(ValueError, match="authenticated audition"):
        _sentinels(sentinel, digest, {"sha256": {}})
    assert _sentinels(sentinel, digest, {"sha256": {"sentinel_list": digest}}) == (set(), sentinel.read_bytes())


def test_renderer_and_assignment_receipts_must_authenticate(tmp_path: Path) -> None:
    prefix, coordinator, annotators, key, audition = _assignment_inputs(tmp_path)
    renderer_receipt = coordinator / "renderer_receipt.json"
    value = json.loads(renderer_receipt.read_text())
    value["counts"]["items"] += 1
    renderer_receipt.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="HMAC verification"):
        build_assignments(
            prefix, coordinator, annotators, key, audition,
            tmp_path / "bad-renderer-public", tmp_path / "bad-renderer-private",
        )

    annotator_assignments, coordinator_assignments = _build_assignments(tmp_path / "assignment")
    assignment_receipt = coordinator_assignments / "assignment_receipt.json"
    value = json.loads(assignment_receipt.read_text())
    value["counts"]["items"] += 1
    assignment_receipt.write_text(json.dumps(value), encoding="utf-8")
    completions = tmp_path / "empty.jsonl"
    completions.write_text("{}\n", encoding="utf-8")
    completion_receipt = tmp_path / "empty.receipt.json"
    completion_receipt.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="HMAC verification"):
        release(
            annotator_assignments, coordinator_assignments, completions, completion_receipt,
            tmp_path / "assignment" / "hmac.key", tmp_path / "bad-assignment",
        )


def test_release_requires_annotation_hash_signed_receipts_and_aliases(tmp_path: Path) -> None:
    annotator_assignments, coordinator_assignments = _build_assignments(tmp_path)
    key = tmp_path / "hmac.key"
    expected = _rows(coordinator_assignments / "coordinator_reveal_release_map.jsonl")
    fields = ("assignment_id", "presentation_id", "annotator_id", "item_id", "prefix_sha256")
    completed = [{**{field: row[field] for field in fields}, "annotation_sha256": "a" * 64} for row in expected]

    missing_hash = [dict(row) for row in completed]
    del missing_hash[0]["annotation_sha256"]
    completions, receipt = _completion_files(tmp_path, coordinator_assignments, key, missing_hash)
    with pytest.raises(ValueError, match="annotation_sha256"):
        release(
            annotator_assignments, coordinator_assignments, completions, receipt, key, tmp_path / "missing-hash"
        )

    completions, receipt = _completion_files(tmp_path, coordinator_assignments, key, completed)
    released = release(
        annotator_assignments, coordinator_assignments, completions, receipt, key, tmp_path / "released"
    )
    reveal_rows = [row for path in _queue_paths(released) for row in _rows(path)]
    allowed = {"assignment_id", "presentation_id", "protocol_version", "reveal_clip", "schema"}
    assert all(set(row) == allowed for row in reveal_rows)
    assert len({row["reveal_clip"] for row in reveal_rows}) == len(reveal_rows)
    assert all((path.parent / row["reveal_clip"]).is_file() for path in _queue_paths(released) for row in _rows(path))
    forbidden = ("item_id", "repeat", "sentinel", "annotator_id", "source", "sha256")
    assert not any(term in json.dumps(reveal_rows) for term in forbidden)
    assert {path.name for path in released.iterdir()} == {"annotators", "release_receipt.json", "READY"}
    release_receipt = json.loads((released / "release_receipt.json").read_text())
    verify_receipt(release_receipt, key.read_bytes(), "release receipt")


def test_release_verifies_completion_hmac_and_exact_reveal_map_hash(tmp_path: Path) -> None:
    annotator_assignments, coordinator_assignments = _build_assignments(tmp_path)
    key = tmp_path / "hmac.key"
    expected = _rows(coordinator_assignments / "coordinator_reveal_release_map.jsonl")
    fields = ("assignment_id", "presentation_id", "annotator_id", "item_id", "prefix_sha256")
    completed = [{**{field: row[field] for field in fields}, "annotation_sha256": "b" * 64} for row in expected]
    completions, receipt = _completion_files(tmp_path, coordinator_assignments, key, completed)
    signed = json.loads(receipt.read_text())
    signed["sha256"]["completions"] = "0" * 64
    receipt.write_text(json.dumps(signed), encoding="utf-8")
    with pytest.raises(ValueError, match="HMAC verification"):
        release(
            annotator_assignments, coordinator_assignments, completions, receipt, key, tmp_path / "bad-completion"
        )

    completions, receipt = _completion_files(tmp_path, coordinator_assignments, key, completed)
    with (coordinator_assignments / "coordinator_reveal_release_map.jsonl").open("ab") as handle:
        handle.write(b"\n")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        release(annotator_assignments, coordinator_assignments, completions, receipt, key, tmp_path / "bad-map")


def test_existing_targets_are_never_overwritten(tmp_path: Path) -> None:
    manifest, lock, media, key, _ = _inputs(tmp_path)
    prefix = tmp_path / "prefix"
    prefix.mkdir()
    coordinator = tmp_path / "coordinator"
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        render_clips(manifest, lock, media, prefix, coordinator, key)
    assert prefix.is_dir() and not coordinator.exists()

    annotator_assignments, coordinator_assignments = _build_assignments(tmp_path / "assignment-case")
    output = tmp_path / "reserved"
    output.mkdir()
    expected = _rows(coordinator_assignments / "coordinator_reveal_release_map.jsonl")
    fields = ("assignment_id", "presentation_id", "annotator_id", "item_id", "prefix_sha256")
    rows = [{**{field: row[field] for field in fields}, "annotation_sha256": "c" * 64} for row in expected]
    assignment_key = tmp_path / "assignment-case" / "hmac.key"
    completions, receipt = _completion_files(tmp_path, coordinator_assignments, assignment_key, rows)
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        release(annotator_assignments, coordinator_assignments, completions, receipt, assignment_key, output)


def test_assignment_roots_refuse_nesting_overwrite_and_unrecorded_public_files(tmp_path: Path) -> None:
    prefix, coordinator, annotators, key, audition = _assignment_inputs(tmp_path / "inputs")
    root = tmp_path / "nested"
    with pytest.raises(ValueError, match="must be separate"):
        build_assignments(prefix, coordinator, annotators, key, audition, root, root / "coordinator")

    reserved = tmp_path / "reserved-annotators"
    reserved.mkdir()
    private = tmp_path / "uncreated-coordinator"
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        build_assignments(prefix, coordinator, annotators, key, audition, reserved, private)
    assert reserved.is_dir() and not private.exists()

    annotator_root, coordinator_root = _build_assignments(tmp_path / "exact-files")
    (annotator_root / "unrecorded").write_text("private metadata", encoding="utf-8")
    with pytest.raises(ValueError, match="unrecorded or missing"):
        release(
            annotator_root, coordinator_root, tmp_path / "unused-completions",
            tmp_path / "unused-receipt", key, tmp_path / "unused-release",
        )
