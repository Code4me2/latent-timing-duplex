"""Security and media-boundary checks for the semantic-silence pilot."""

from __future__ import annotations

import hashlib
import json
import wave
from collections import Counter
from pathlib import Path

import pytest

from experiments.silence_graft import build_assignments as assignments_module
from experiments.silence_graft import render_clips as renderer_module
from experiments.silence_graft.build_assignments import _sentinels, build_assignments, release
from experiments.silence_graft.build_features import _pcm, _window_stats, build_features
from experiments.silence_graft.build_manifest import assign_lineage_splits
from experiments.silence_graft.compile_annotations import adjudicate, report
from experiments.silence_graft.compile_r2_dataset import compile_r2_dataset
from experiments.silence_graft.extract_outcomes import extract_outcomes
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


@pytest.mark.parametrize(
    ("width", "negative", "positive", "scale"),
    [(1, 64, 192, 128), (2, -16384, 16384, 32768), (3, -4194304, 4194304, 8388608), (4, -1073741824, 1073741824, 2147483648)],
)
def test_exact_cutoff_pcm_widths_and_trailing_boundaries(
    tmp_path: Path, width: int, negative: int, positive: int, scale: int
) -> None:
    path = tmp_path / f"pcm-{width}.wav"
    if width == 1:
        payload = bytes([negative, positive] * 4)
    else:
        payload = b"".join(
            value.to_bytes(width, "little", signed=True)
            for _ in range(4)
            for value in (negative, positive)
        )
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(2)
        handle.setsampwidth(width)
        handle.setframerate(1000)
        handle.writeframes(payload)
    samples, rate, observed_width = _pcm(path)
    assert (samples.shape, rate, observed_width) == ((4, 2), 1000, width)
    expected_negative = (negative - 128) / scale if width == 1 else negative / scale
    expected_positive = (positive - 128) / scale if width == 1 else positive / scale
    np = pytest.importorskip("numpy")
    np.testing.assert_allclose(samples, [[expected_negative, expected_positive]] * 4)
    acoustic, change_onset, truncated = _window_stats(samples, rate)
    assert acoustic.shape == (42,) and change_onset.shape == (18,) and truncated.shape == (6,)
    assert truncated.tolist() == [1.0] * 6


def _rows(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _inputs(tmp_path: Path, *, media_updates=None, manifest_updates=None, key_bytes=KEY_BYTES):
    tmp_path.mkdir(parents=True, exist_ok=True)
    wav = tmp_path / "private.wav"
    _write_wav(wav)
    candidates = [f"lineage-{index:03d}" for index in range(1000)]
    candidate_splits = assign_lineage_splits(candidates, 20261009)
    lineage_ids = (
        [lineage for lineage in candidates if candidate_splits[lineage] == "train"][:15]
        + [lineage for lineage in candidates if candidate_splits[lineage] == "development"][:15]
    )
    lineage_splits = {lineage: candidate_splits[lineage] for lineage in lineage_ids}
    vad_rows = []
    manifest_rows = []
    for index in range(75):
        session = f"source-session-{index:03d}"
        lineage = lineage_ids[index % 30]
        case = index % 4
        resume = None if case == 3 else 1.0 if case == 2 else 1.5
        user = [[0.0, 0.145], *([] if resume is None else [[resume, 2.0]])]
        assistant = [[0.99 if case == 2 else 1.0, 1.1]] if case in {1, 2} else []
        vad_rows.append({
            "session_id": session, "lineage_id": lineage,
            "source": "test-corpus", "duration_s": 2.0,
            "user_vad": user, "assistant_vad": assistant,
        })
        manifest_rows.append({
            "context": "joint_silence", "elapsed_s": 0.48,
            "episode_id": f"{session}:00000",
            "lineage_id": lineage,
            "model_cutoff_s": 0.625, "onset_s": 0.145, "right_censored": resume is None,
            "session_id": session, "silence_end_s": 2.0 if resume is None else resume,
            "source": "test-corpus", "split": lineage_splits[lineage],
        })
    if manifest_updates:
        manifest_rows[0].update(manifest_updates)
    vad_payload = _jsonl(vad_rows)
    (tmp_path / "vad.jsonl").write_bytes(vad_payload)
    manifest_payload = _jsonl(manifest_rows)
    manifest = tmp_path / "manifest.pilot.jsonl"
    manifest.write_bytes(manifest_payload)
    splits_payload = (json.dumps({
        "schema": "semantic-silence-config/v1",
        "protocol_version": "semantic-silence-pilot/v1",
        "split_seed": 20261009,
        "lineages": lineage_splits,
    }, indent=2, sort_keys=True) + "\n").encode()
    (tmp_path / "manifest.splits.json").write_bytes(splits_payload)
    lock = tmp_path / "manifest.jsonl.lock.json"
    lock.write_bytes(receipt_bytes(sign_receipt({
        "schema": "semantic-silence-config/v1",
        "protocol_version": "semantic-silence-pilot/v1",
        "label_schema": "semantic-silence-labels/v1",
        "sha256": {
            "input_vad": hashlib.sha256(vad_payload).hexdigest(),
            "full_manifest": "2" * 64,
            "splits": hashlib.sha256(splits_payload).hexdigest(),
            "pilot_manifest": hashlib.sha256(manifest_payload).hexdigest(),
            "config": hashlib.sha256(CONFIG.read_bytes()).hexdigest(),
            "protocol_doc": hashlib.sha256(PROTOCOL.read_bytes()).hexdigest(),
        },
        "counts": {
            "input_sessions": 75, "input_lineages": 30,
            "split_lineages": {"development": 15, "train": 15},
            "full_episodes": 75, "full_checkpoints": 75,
            "pilot_episodes": 75, "pilot_checkpoints": 75, "pilot_lineages": 30,
            "pilot_by_source": {"test-corpus": 75},
            "pilot_by_elapsed_stratum": {"<1s": 75},
        },
    }, key_bytes if len(key_bytes) >= 32 else KEY_BYTES)))
    media_rows = [{
        "assistant_channel": 0, "session_id": row["session_id"], "user_channel": 1,
        "wav_path": str(wav.resolve()), "wav_sha256": hashlib.sha256(wav.read_bytes()).hexdigest(),
    } for row in manifest_rows]
    if media_updates:
        media_rows[0].update(media_updates)
    media = tmp_path / "media.jsonl"
    media.write_bytes(_jsonl(media_rows))
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
    sources = _rows(source_map)
    sessions = [{
        "assistant_channel": source["assistant_channel"],
        "assistant_channel_auditioned": True,
        "session_id": source["session_id"],
        "user_channel": source["user_channel"],
        "user_channel_auditioned": True,
        "wav_sha256": source["wav_sha256"],
    } for source in sources]
    if session_updates:
        sessions[0].update(session_updates)
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
        "sessions": sessions,
        "sha256": hashes,
    }, key.read_bytes())
    path.write_bytes(receipt_bytes(signed))
    return path


def _assignment_inputs(tmp_path: Path, *, audition_updates=None):
    prefix, coordinator, key, _ = _render(tmp_path)
    annotators = tmp_path / "annotators.json"
    annotators.write_text(
        json.dumps(["alice", "bob", "carol", "dana", "erin", "frank", "grace", "heidi", "ivan"]),
        encoding="utf-8",
    )
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
    assert len(receipt["sources"]) == 75
    assert receipt["sources"][0] == {
        "assistant_channel": 0, "sample_rate": 4, "session_id": "source-session-000",
        "user_channel": 1, "wav_sha256": mapping[0]["wav_sha256"],
    }
    assert mapping[0]["lineage_id"].startswith("lineage-")


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
    render_clips(manifest, lock, media, tmp_path / "p2", tmp_path / "c2", key)
    assert (tmp_path / "p2" / "READY").is_file()
    assert (tmp_path / "c2" / "READY").is_file()


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
    rows = _rows(media)
    media.write_bytes(_jsonl([*rows, {**rows[0], "session_id": "extra"}]))
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
    with pytest.raises(ValueError, match="mode 0600"):
        render_clips(manifest, lock, media, tmp_path / "p4", tmp_path / "c4", key)


def test_renderer_rejects_source_and_key_symlinks(tmp_path: Path) -> None:
    manifest, lock, media, key, wav = _inputs(tmp_path / "source")
    linked_wav = tmp_path / "source" / "linked.wav"
    linked_wav.symlink_to(wav)
    rows = _rows(media)
    for row in rows:
        row["wav_path"] = str(linked_wav.absolute())
    media.write_bytes(_jsonl(rows))
    with pytest.raises(ValueError, match="non-symlink"):
        render_clips(manifest, lock, media, tmp_path / "source-prefix", tmp_path / "source-coordinator", key)

    manifest, lock, media, key, _ = _inputs(tmp_path / "key")
    linked_key = tmp_path / "key" / "linked.key"
    linked_key.symlink_to(key)
    with pytest.raises(ValueError, match="non-symlink"):
        render_clips(manifest, lock, media, tmp_path / "key-prefix", tmp_path / "key-coordinator", linked_key)


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
    assert all(
        row["checkpoint_elapsed_s"] == row["elapsed_s"] == 0.48
        and row["model_cutoff_s"] - row["onset_s"] == pytest.approx(row["elapsed_s"])
        for row in private
    )
    originals = [row for row in private if not row["repeat"]]
    assert all(len({row["annotator_id"] for row in originals if row["item_id"] == item}) == 3 for item in {row["item_id"] for row in originals})
    loads = Counter(row["annotator_id"] for row in originals)
    assert max(loads.values()) - min(loads.values()) <= 1
    assert all(
        len(rows) == len({row["lineage_id"] for row in rows})
        for rows in (
            [row for row in originals if row["annotator_id"] == annotator]
            for annotator in loads
        )
    )

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
    annotators.write_text(json.dumps(["alice", "bob", "carol"]), encoding="utf-8")
    with pytest.raises(ValueError, match="lineage-independent"):
        build_assignments(
            prefix, coordinator, annotators, key, audition,
            tmp_path / "small-pool-public", tmp_path / "small-pool-private",
        )
    annotators.write_text(
        json.dumps(["alice", "bob", "carol", "dana", "erin", "frank", "grace", "heidi", "ivan"]),
        encoding="utf-8",
    )
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


def test_annotation_report_release_and_adjudication_end_to_end(tmp_path: Path) -> None:
    annotator_root, coordinator_root = _build_assignments(tmp_path)
    key = tmp_path / "hmac.key"
    assignments = _rows(coordinator_root / "private_assignment_source_map.jsonl")
    actions = ("wait", "ack", "respond", "repair")
    item_actions = {
        item: actions[index % len(actions)]
        for index, item in enumerate(sorted({row["item_id"] for row in assignments}))
    }
    responses = []
    for row in assignments:
        action = item_actions[row["item_id"]]
        responses.append({
            "schema": "semantic-silence-prefix-response/v1",
            "protocol_version": "semantic-silence-pilot/v1",
            "assignment_id": row["assignment_id"],
            "presentation_id": row["presentation_id"],
            "annotator_id": row["annotator_id"],
            "action": action,
            "confidence": 4,
            "also_acceptable": [],
            "tags": ["floor_holding"],
            "acceptable_onset": (
                {
                    "earliest_elapsed_s": None,
                    "latest_elapsed_s": None,
                    "wait_for_user_resume": True,
                }
                if action == "wait"
                else {
                    "earliest_elapsed_s": 0.48,
                    "latest_elapsed_s": 1.0,
                    "wait_for_user_resume": False,
                }
            ),
            "quality": "clean",
            "ratable": True,
        })
    raw = tmp_path / "raw.jsonl"
    raw.write_bytes(_jsonl(responses))
    report_root = report(
        annotator_root, coordinator_root, raw, key, CONFIG,
        tmp_path / "annotation-report",
    )
    agreement = json.loads((report_root / "agreement_receipt.json").read_text())
    verify_receipt(agreement, key.read_bytes(), "agreement receipt")
    assert agreement["gate_passed"] is True

    released = release(
        annotator_root, coordinator_root, report_root / "completions.jsonl",
        report_root / "completion_receipt.json", key, tmp_path / "released-from-compiler",
    )
    assert (released / "READY").is_file()

    labels_root = adjudicate(
        report_root / "validated_annotations.jsonl",
        report_root / "agreement_receipt.json", key, tmp_path / "adjudicated",
    )
    labels = _rows(labels_root / "adjudicated.jsonl")
    assert {row["action"] for row in labels} == {"wait", "ack", "respond", "repair"}
    forbidden = {"episode_id", "session_id", "silence_end_s", "annotator_id", "onset_s"}
    assert all(not forbidden.intersection(row) for row in labels)
    adjudication_receipt = json.loads((labels_root / "adjudication_receipt.json").read_text())
    verify_receipt(adjudication_receipt, key.read_bytes(), "adjudication receipt")
    assert adjudication_receipt["sha256"]["assignment_receipt"] == hashlib.sha256(
        (coordinator_root / "assignment_receipt.json").read_bytes()
    ).hexdigest()

    nested_outcomes = coordinator_root / "nested-outcomes"
    with pytest.raises(ValueError, match="separate from authenticated input packages"):
        extract_outcomes(
            tmp_path / "vad.jsonl", tmp_path / "manifest.jsonl.lock.json",
            annotator_root, coordinator_root, key, nested_outcomes, CONFIG,
        )
    assert not nested_outcomes.exists()

    outcomes_root = extract_outcomes(
        tmp_path / "vad.jsonl", tmp_path / "manifest.jsonl.lock.json",
        annotator_root, coordinator_root, key, tmp_path / "outcomes", CONFIG,
    )
    outcomes = _rows(outcomes_root / "outcomes.jsonl")
    assert len(outcomes) == 75
    assert {row["first_event"] for row in outcomes} == {
        None, "user_resume", "recorded_partner_entry"
    }
    tie = next(row for row in outcomes if row["session_id"] == "source-session-002")
    assert tie["recorded_partner_entry_s"] < tie["user_resume_s"]
    assert tie["sample_rate"] == 4 and tie["first_event"] == "user_resume"
    assert tie["recorded_overlap_at_user_resume"] is True
    assert tie["overlap_duration_s"] == pytest.approx(0.1)

    different_lock = json.loads((tmp_path / "manifest.jsonl.lock.json").read_text())
    different_lock.pop("hmac_sha256")
    different_lock.pop("key_fingerprint")
    different_lock["replacement_lock"] = True
    different_lock_path = tmp_path / "different-manifest.lock.json"
    different_lock_path.write_bytes(receipt_bytes(sign_receipt(different_lock, key.read_bytes())))
    with pytest.raises(ValueError, match="bind this config and manifest lock"):
        extract_outcomes(
            tmp_path / "vad.jsonl", different_lock_path, annotator_root, coordinator_root,
            key, tmp_path / "rejected-different-lock", CONFIG,
        )

    nested_dataset = outcomes_root / "nested-dataset"
    with pytest.raises(ValueError, match="separate from authenticated input packages"):
        compile_r2_dataset(
            labels_root, outcomes_root, annotator_root, coordinator_root,
            tmp_path / "manifest.splits.json", tmp_path / "manifest.jsonl.lock.json",
            key, nested_dataset,
        )
    assert not nested_dataset.exists()

    dataset = compile_r2_dataset(
        labels_root, outcomes_root, annotator_root, coordinator_root,
        tmp_path / "manifest.splits.json", tmp_path / "manifest.jsonl.lock.json",
        key, tmp_path / "r2-dataset",
    )
    causal = [
        row
        for split in ("train", "development")
        for row in _rows(dataset / "causal_labels" / f"{split}.jsonl")
    ]
    assert len(causal) == 75
    assert all(row["sampling_weight"] == 1.0 for row in causal)
    assert all(row["sampling_weight_identified"] is False for row in causal)
    assert all(
        row["action"] in row["action_distribution"]
        and set(row["action_distribution"]) == {"wait", "ack", "respond", "repair", "unratable"}
        for row in causal
    )
    assert all({"confidence", "tags", "quality"} <= row.keys() for row in causal)
    assert all(not {"session_id", "episode_id", "annotator_id"} & row.keys() for row in causal)
    verify_receipt(
        json.loads((dataset / "dataset_receipt.json").read_text()),
        key.read_bytes(), "dataset receipt",
    )

    renderer_root = tmp_path / "coordinator-package"
    prefix_root = tmp_path / "prefix-package"
    render = {
        row["item_id"]: row
        for row in _rows(renderer_root / "render_records.jsonl")
        if row["package"] == "prefix"
    }
    media_index = {row["item_id"]: row for row in _rows(renderer_root / "coordinator_media_index.jsonl")}
    items = sorted(row["item_id"] for row in causal)
    causal_by_item = {row["item_id"]: row for row in causal}
    frozen_path = tmp_path / "frozen.npz"
    import numpy as np
    np.savez(
        frozen_path,
        item_id=np.asarray(items),
        prefix_sha256=np.asarray([media_index[item]["prefix_sha256"] for item in items]),
        available_at_s=np.asarray([causal_by_item[item]["model_cutoff_s"] for item in items]),
        model_cutoff_s=np.asarray([causal_by_item[item]["model_cutoff_s"] for item in items]),
        hidden=np.arange(len(items) * 3, dtype=float).reshape(len(items), 3),
        hidden_previous=np.arange(len(items) * 3, dtype=float).reshape(len(items), 3) - 1.0,
        predicted_latent=np.arange(len(items) * 2, dtype=float).reshape(len(items), 2),
    )
    renderer_digest = hashlib.sha256((renderer_root / "renderer_receipt.json").read_bytes()).hexdigest()
    frozen_receipt = tmp_path / "frozen.receipt.json"
    frozen_receipt.write_bytes(receipt_bytes(sign_receipt({
        "schema": "semantic-silence-frozen-feature-receipt/v1",
        "protocol_version": "semantic-silence-pilot/v1",
        "frozen": True,
        "causal_input": "authenticated_prefix_clip_only",
        "model_id": "fake-model",
        "model_revision": "test-revision",
        "layer": "layer-1",
        "checkpoint_sha256": "1" * 64,
        "extractor_code_sha256": "2" * 64,
        "feature_npz_sha256": hashlib.sha256(frozen_path.read_bytes()).hexdigest(),
        "renderer_receipt_sha256": renderer_digest,
        "config_sha256": hashlib.sha256(CONFIG.read_bytes()).hexdigest(),
        "context_start": "authenticated_render_record_start_frame",
        "state_reset": "per_prefix_clip",
        "history_truncation": "renderer_prefix_or_session_start",
    }, key.read_bytes())))
    nested_features = prefix_root / "nested-features"
    with pytest.raises(ValueError, match="separate from authenticated input packages"):
        build_features(
            dataset, labels_root, prefix_root, renderer_root, annotator_root, coordinator_root,
            frozen_path, frozen_receipt, key, nested_features, CONFIG,
        )
    assert not nested_features.exists()

    feature_package = build_features(
        dataset, labels_root, prefix_root, renderer_root, annotator_root, coordinator_root,
        frozen_path, frozen_receipt, key, tmp_path / "r2-features", CONFIG,
    )
    with np.load(feature_package / "features.npz", allow_pickle=False) as archive:
        assert set(("item_id", "source", "energy_change", "sampling_weight_status")) <= set(archive.files)
        assert np.all(archive["model_cutoff_s"] == 0.625)
        assert np.all(archive["render_boundary_s"] == 0.75)
        assert np.all(archive["available_at_s"] <= archive["model_cutoff_s"])
        assert np.any(archive["history_truncated"] == 1.0)
        np.testing.assert_allclose(archive["hidden_velocity"], 1.0)
    second = build_features(
        dataset, labels_root, prefix_root, renderer_root, annotator_root, coordinator_root,
        frozen_path, frozen_receipt, key, tmp_path / "r2-features-copy", CONFIG,
    )
    assert (feature_package / "features.npz").read_bytes() == (second / "features.npz").read_bytes()

    original_frozen = frozen_path.read_bytes()
    with frozen_path.open("ab") as handle:
        handle.write(b"tamper")
    with pytest.raises(ValueError, match="NPZ hash mismatch"):
        build_features(
            dataset, labels_root, prefix_root, renderer_root, annotator_root, coordinator_root,
            frozen_path, frozen_receipt, key, tmp_path / "tampered-features", CONFIG,
        )
    frozen_path.write_bytes(original_frozen)
    with np.load(frozen_path, allow_pickle=False) as archive:
        future_arrays = {name: archive[name] for name in archive.files}
    future_arrays["available_at_s"] = future_arrays["available_at_s"].copy()
    future_arrays["available_at_s"][0] += 1.0
    future_path = tmp_path / "future-frozen.npz"
    np.savez(future_path, **future_arrays)
    future_lock = json.loads(frozen_receipt.read_text())
    future_lock.pop("hmac_sha256")
    future_lock.pop("key_fingerprint")
    future_lock["feature_npz_sha256"] = hashlib.sha256(future_path.read_bytes()).hexdigest()
    future_receipt = tmp_path / "future-frozen.receipt.json"
    future_receipt.write_bytes(receipt_bytes(sign_receipt(future_lock, key.read_bytes())))
    with pytest.raises(ValueError, match="observes beyond"):
        build_features(
            dataset, labels_root, prefix_root, renderer_root, annotator_root, coordinator_root,
            future_path, future_receipt, key, tmp_path / "future-features", CONFIG,
        )

    outcome_path = outcomes_root / "outcomes.jsonl"
    outcome_receipt_path = outcomes_root / "outcomes_receipt.json"
    original_outcomes, original_outcome_receipt = outcome_path.read_bytes(), outcome_receipt_path.read_bytes()

    def replace_outcomes(rows, target: str) -> None:
        payload = _jsonl(rows)
        outcome_path.write_bytes(payload)
        receipt = json.loads(original_outcome_receipt)
        receipt.pop("hmac_sha256")
        receipt["outputs"]["outcomes.jsonl"] = hashlib.sha256(payload).hexdigest()
        receipt["counts"]["items"] = len(rows)
        outcome_receipt_path.write_bytes(receipt_bytes(sign_receipt(receipt, key.read_bytes())))
        with pytest.raises(ValueError, match=target):
            compile_r2_dataset(
                labels_root, outcomes_root, annotator_root, coordinator_root,
                tmp_path / "manifest.splits.json", tmp_path / "manifest.jsonl.lock.json",
                key, tmp_path / f"rejected-{target.replace(' ', '-')}",
            )
        outcome_path.write_bytes(original_outcomes)
        outcome_receipt_path.write_bytes(original_outcome_receipt)

    future_rows = _rows(outcome_path)
    future_rows[0]["observed_duration_s"] = 8.1
    future_rows[0]["administrative_censor_s"] = future_rows[0]["model_cutoff_s"] + 8.1
    replace_outcomes(future_rows, "observation boundary")
    replace_outcomes(_rows(outcome_path)[:-1], "exactly cover")
    replace_outcomes([*_rows(outcome_path), _rows(outcome_path)[0]], "duplicate item_id")

    splits_path, lock_path = tmp_path / "manifest.splits.json", tmp_path / "manifest.jsonl.lock.json"
    original_splits, original_lock = splits_path.read_bytes(), lock_path.read_bytes()
    moved = json.loads(original_splits)
    moved_lineage = next(iter(moved["lineages"]))
    moved["lineages"][moved_lineage] = (
        "development" if moved["lineages"][moved_lineage] == "train" else "train"
    )
    moved_payload = (json.dumps(moved, indent=2, sort_keys=True) + "\n").encode()
    splits_path.write_bytes(moved_payload)
    moved_lock = json.loads(original_lock)
    moved_lock.pop("hmac_sha256")
    moved_lock["sha256"]["splits"] = hashlib.sha256(moved_payload).hexdigest()
    moved_lock["counts"]["split_lineages"] = dict(sorted(Counter(moved["lineages"].values()).items()))
    lock_path.write_bytes(receipt_bytes(sign_receipt(moved_lock, key.read_bytes())))
    moved_outcome_receipt = json.loads(original_outcome_receipt)
    moved_outcome_receipt.pop("hmac_sha256")
    moved_outcome_receipt["sha256"]["manifest_lock"] = hashlib.sha256(lock_path.read_bytes()).hexdigest()
    outcome_receipt_path.write_bytes(receipt_bytes(sign_receipt(moved_outcome_receipt, key.read_bytes())))
    with pytest.raises(ValueError, match="not chained to this manifest lock"):
        compile_r2_dataset(
            labels_root, outcomes_root, annotator_root, coordinator_root,
            splits_path, lock_path, key, tmp_path / "rejected-split-move",
        )
    splits_path.write_bytes(original_splits)
    lock_path.write_bytes(original_lock)
    outcome_receipt_path.write_bytes(original_outcome_receipt)

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        extract_outcomes(
            tmp_path / "vad.jsonl", tmp_path / "manifest.jsonl.lock.json",
            annotator_root, coordinator_root, key, outcomes_root, CONFIG,
        )
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        compile_r2_dataset(
            labels_root, outcomes_root, annotator_root, coordinator_root,
            splits_path, lock_path, key, dataset,
        )
    with (tmp_path / "vad.jsonl").open("ab") as handle:
        handle.write(b" ")
    with pytest.raises(ValueError, match="input_vad SHA-256"):
        extract_outcomes(
            tmp_path / "vad.jsonl", tmp_path / "manifest.jsonl.lock.json",
            annotator_root, coordinator_root, key, tmp_path / "tampered-outcomes", CONFIG,
        )


def test_failed_gate_and_all_different_require_new_pilot_or_expert(tmp_path: Path) -> None:
    annotator_root, coordinator_root = _build_assignments(tmp_path)
    key = tmp_path / "hmac.key"
    assignments = _rows(coordinator_root / "private_assignment_source_map.jsonl")
    by_item = {}
    for row in assignments:
        if not row["repeat"]:
            by_item.setdefault(row["item_id"], []).append(row["annotator_id"])
    action = {
        (item, annotator): ["wait", "ack", "respond"][index]
        for item, annotators in by_item.items()
        for index, annotator in enumerate(sorted(annotators))
    }
    responses = [{
        "schema": "semantic-silence-prefix-response/v1",
        "protocol_version": "semantic-silence-pilot/v1",
        "assignment_id": row["assignment_id"],
        "presentation_id": row["presentation_id"],
        "annotator_id": row["annotator_id"],
        "action": action[(row["item_id"], row["annotator_id"])],
        "confidence": 3,
        "also_acceptable": [],
        "tags": [],
        "acceptable_onset": {
            "earliest_elapsed_s": None,
            "latest_elapsed_s": None,
            "wait_for_user_resume": True,
        },
        "quality": "clean",
        "ratable": True,
    } for row in assignments]
    raw = tmp_path / "raw.jsonl"
    raw.write_bytes(_jsonl(responses))
    report_root = report(
        annotator_root, coordinator_root, raw, key, CONFIG,
        tmp_path / "failed-report", bootstrap_replicates=2,
    )
    receipt_path = report_root / "agreement_receipt.json"
    failed = json.loads(receipt_path.read_text())
    assert failed["gate_passed"] is False
    assert failed["bootstrap"]["nonconfirmatory_override"] is True
    assert failed["collapse_decision"] == "nonconfirmatory"
    with pytest.raises(ValueError, match="nonconfirmatory|gate did not pass"):
        adjudicate(
            report_root / "validated_annotations.jsonl", receipt_path, key,
            tmp_path / "forbidden-adjudication",
        )

    failed.pop("hmac_sha256")
    failed["gate_passed"] = True
    receipt_path.write_bytes(receipt_bytes(sign_receipt(failed, key.read_bytes())))
    with pytest.raises(ValueError, match="nonconfirmatory"):
        adjudicate(
            report_root / "validated_annotations.jsonl", receipt_path, key,
            tmp_path / "missing-expert",
        )


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


def test_multi_package_publication_removes_ready_markers_on_late_failure(
    tmp_path: Path, monkeypatch
) -> None:
    manifest, lock, media, key, _ = _inputs(tmp_path / "renderer-input")
    original_render_publish = renderer_module._publish_staging
    calls = 0

    def fail_second_render(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("injected publication failure")
        return original_render_publish(*args, **kwargs)

    monkeypatch.setattr(renderer_module, "_publish_staging", fail_second_render)
    prefix, coordinator = tmp_path / "failed-prefix", tmp_path / "failed-coordinator"
    with pytest.raises(RuntimeError, match="injected"):
        render_clips(manifest, lock, media, prefix, coordinator, key)
    assert not (prefix / "READY").exists()
    assert not (coordinator / "READY").exists()

    first, second = tmp_path / "first-package", tmp_path / "second-package"
    original_assignment_publish = assignments_module._publish_staging
    calls = 0

    def fail_second_assignment(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("injected assignment publication failure")
        return original_assignment_publish(*args, **kwargs)

    monkeypatch.setattr(assignments_module, "_publish_staging", fail_second_assignment)
    with pytest.raises(RuntimeError, match="injected"):
        assignments_module._publish_packages({first: {"one": b"1"}, second: {"two": b"2"}})
    assert not (first / "READY").exists()
    assert not (second / "READY").exists()


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
