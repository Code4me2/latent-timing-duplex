"""Locked silence manifest construction checks."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path

import pytest

from experiments.silence_graft.build_manifest import (
    _publish_artifacts,
    _records,
    _sessions,
    assign_lineage_splits,
    build_manifests,
    companion_paths,
    elapsed_stratum,
    select_pilot_items,
)
from experiments.silence_graft.receipts import verify_receipt
from latent_timing_duplex.silence import extract_silence_checkpoints, sample_episodes

ROOT = Path(__file__).resolve().parents[1]



def test_vad_json_rejects_duplicate_keys_and_nonfinite_numbers() -> None:
    with pytest.raises(ValueError, match="duplicate JSON key"):
        _records(b'{"session_id":"one","session_id":"two"}\n')
    with pytest.raises(ValueError, match="non-finite JSON number"):
        _records(b'{"duration_s":NaN}\n')


def test_lineage_cap_spans_sessions() -> None:
    rows = []
    for session in ("s1", "s2"):
        rows.extend(
            extract_silence_checkpoints(
                session,
                6.0,
                [(0, 1), (2, 3), (4, 5)],
                [],
                checkpoints_s=[0.24, 0.48],
                lineage_id="shared",
            )
        )
    selected = sample_episodes(rows, max_per_lineage=2, seed=7)
    assert len({row.episode_id for row in selected}) == 2
    assert {row.session_id for row in selected} <= {"s1", "s2"}


def test_pilot_selection_is_deterministic_disjoint_and_uses_elapsed_strata() -> None:
    rows = []
    splits = {}
    for index, split in enumerate(
        ["train", "development", "train", "development", "test", "challenge"]
    ):
        lineage = f"lineage-{index}"
        rows.extend(
            extract_silence_checkpoints(
                f"session-{index}",
                10.0,
                [(0.0, 1.0)],
                [],
                checkpoints_s=[0.48, 3.0],
                source=f"source-{index % 2}",
                lineage_id=lineage,
            )
        )
        splits[lineage] = split
    kwargs = dict(
        target_episodes=4,
        min_lineages=3,
        allowed_splits=["train", "development"],
        excluded_splits=["test", "challenge"],
        boundaries_s=[1.0, 3.0],
        stratum_labels=["<1s", "1-<3s", ">=3s"],
        seed=11,
    )
    first = select_pilot_items(rows, splits, **kwargs)
    second = select_pilot_items(rows, splits, **kwargs)
    assert first == second
    assert len(first) == len({row.episode_id for row in first}) == 4
    assert len({row.lineage_id for row in first}) >= 3
    assert {splits[row.lineage_id] for row in first} <= {"train", "development"}
    assert all(row.silence_end_s - row.onset_s > 3 for row in first)
    assert {
        elapsed_stratum(row.elapsed_s, [1.0, 3.0], ["<1s", "1-<3s", ">=3s"])
        for row in first
    } == {"<1s", ">=3s"}
    with pytest.raises(ValueError, match="requires 5 eligible episodes"):
        select_pilot_items(rows, splits, **{**kwargs, "target_episodes": 5})


def _valid_record(**updates):
    row = {
        "session_id": "s1",
        "lineage_id": "l1",
        "source": "source",
        "duration_s": 5.0,
        "user_vad": [[0.0, 1.0], [2.0, 3.0]],
        "assistant_vad": [[0.5, 1.5]],
    }
    row.update(updates)
    return row


@pytest.mark.parametrize(
    "payload,match",
    [
        ("{\n", "malformed JSON"),
        (json.dumps(["not", "an", "object"]) + "\n", "must be an object"),
        (json.dumps({key: value for key, value in _valid_record().items() if key != "session_id"}) + "\n", "session_id"),
        (json.dumps({key: value for key, value in _valid_record().items() if key != "lineage_id"}) + "\n", "lineage_id"),
        (json.dumps({key: value for key, value in _valid_record().items() if key != "source"}) + "\n", "source"),
        (json.dumps({key: value for key, value in _valid_record().items() if key != "duration_s"}) + "\n", "duration_s"),
        (json.dumps({key: value for key, value in _valid_record().items() if key != "user_vad"}) + "\n", "user VAD requires exactly one"),
        (json.dumps({**_valid_record(), "user": []}) + "\n", "user VAD requires exactly one"),
        (json.dumps({key: value for key, value in _valid_record().items() if key != "assistant_vad"}) + "\n", "assistant VAD requires exactly one"),
        (json.dumps({**_valid_record(), "assistant": []}) + "\n", "assistant VAD requires exactly one"),
        (json.dumps(_valid_record(challenge="false")) + "\n", "challenge must be a boolean"),
        (json.dumps(_valid_record(duration_s=float("nan"))) + "\n", "finite"),
        (json.dumps(_valid_record(duration_s=0.0)) + "\n", "positive"),
        (json.dumps(_valid_record(user_vad=[[0.0, float("nan")]])) + "\n", "finite"),
        (json.dumps(_valid_record(user_vad=[[-0.1, 1.0]])) + "\n", "within duration_s"),
        (json.dumps(_valid_record(user_vad=[[1.0, 1.0]])) + "\n", "start must precede"),
        (json.dumps(_valid_record(user_vad=[[0.0, 2.0], [1.0, 3.0]])) + "\n", "must not overlap"),
        (json.dumps(_valid_record()) + "\n" + json.dumps(_valid_record()) + "\n", "duplicate session_id"),
    ],
)
def test_vad_input_is_strict(tmp_path: Path, payload: str, match: str) -> None:
    path = tmp_path / "vad.jsonl"
    path.write_text(payload, encoding="utf-8")
    with pytest.raises(ValueError, match=match):
        _sessions(path.read_bytes())


def _corpus() -> str:
    lines = []
    for lineage_index in range(50):
        for session_index in range(2):
            lines.append(
                json.dumps(
                    {
                        "session_id": f"session-{lineage_index:02d}-{session_index}",
                        "lineage_id": f"lineage-{lineage_index:02d}",
                        "source": f"source-{lineage_index % 2}",
                        "challenge": lineage_index == 49,
                        "duration_s": 11.0,
                        "user_vad": [[0, 1], [2, 3], [6, 7]],
                        "assistant_vad": [[3.5, 4.0]],
                    },
                    sort_keys=True,
                )
            )
    return "\n".join(lines) + "\n"


def _read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_locked_outputs_are_deterministic_hashed_and_never_overwritten(tmp_path: Path) -> None:
    outputs = []
    for directory in (tmp_path / "first", tmp_path / "second"):
        directory.mkdir()
        vad = directory / "vad.jsonl"
        vad.write_text(_corpus(), encoding="utf-8")
        key = directory / "hmac.key"
        key.write_bytes(b"manifest-test-key-material-at-least-32-bytes")
        key.chmod(0o600)
        outputs.append(build_manifests(vad, directory / "manifest.jsonl", hmac_key=key))

    for first, second in zip(outputs[0], outputs[1]):
        assert first.read_bytes() == second.read_bytes()

    full_path, pilot_path, splits_path, lock_path = outputs[0]
    full, pilot = _read_jsonl(full_path), _read_jsonl(pilot_path)
    split_document = json.loads(splits_path.read_text(encoding="utf-8"))
    splits = split_document["lineages"]
    assert set(splits) == {f"lineage-{index:02d}" for index in range(50)}
    assert set(splits.values()) <= {"train", "development", "test", "challenge"}
    assert splits["lineage-49"] == "challenge"
    assert all(row["split"] == splits[row["lineage_id"]] for row in full)
    assert {row["split"] for row in pilot} <= {"train", "development"}
    assert "lineage-49" not in {row["lineage_id"] for row in pilot}
    assert len(pilot) == len({row["episode_id"] for row in pilot}) == 75
    assert len({row["lineage_id"] for row in pilot}) >= 30

    episodes_by_lineage = defaultdict(set)
    for row in full:
        episodes_by_lineage[row["lineage_id"]].add(row["episode_id"])
    assert max(map(len, episodes_by_lineage.values())) <= 5

    receipt = json.loads(lock_path.read_text(encoding="utf-8"))
    verify_receipt(receipt, (tmp_path / "first" / "hmac.key").read_bytes(), "manifest lock")
    hashes = receipt["sha256"]
    assert hashes["input_vad"] == hashlib.sha256((tmp_path / "first" / "vad.jsonl").read_bytes()).hexdigest()
    assert hashes["config"] == hashlib.sha256((ROOT / "configs" / "silence.yaml").read_bytes()).hexdigest()
    assert hashes["protocol_doc"] == hashlib.sha256((ROOT / "docs" / "DATA_LABELING_PROTOCOL.md").read_bytes()).hexdigest()
    assert hashes["full_manifest"] == hashlib.sha256(full_path.read_bytes()).hexdigest()
    assert hashes["pilot_manifest"] == hashlib.sha256(pilot_path.read_bytes()).hexdigest()
    assert hashes["splits"] == hashlib.sha256(splits_path.read_bytes()).hexdigest()
    assert receipt["counts"]["pilot_episodes"] == 75
    assert receipt["counts"]["pilot_lineages"] >= 30

    assert companion_paths(full_path) == outputs[0][1:]
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        build_manifests(
            tmp_path / "first" / "vad.jsonl", full_path,
            hmac_key=tmp_path / "first" / "hmac.key",
        )


def test_split_assignment_is_lineage_total_and_deterministic() -> None:
    lineages = ["b", "a", "b", "c"]
    first = assign_lineage_splits(lineages, 19)
    assert first == assign_lineage_splits(reversed(lineages), 19)
    assert set(first) == {"a", "b", "c"}
    assert all(split in {"train", "development", "test"} for split in first.values())


def test_locked_config_cannot_select_test(tmp_path: Path) -> None:
    vad = tmp_path / "vad.jsonl"
    vad.write_text(_corpus(), encoding="utf-8")
    config = tmp_path / "silence.yaml"
    source = (ROOT / "configs" / "silence.yaml").read_text(encoding="utf-8")
    config.write_text(
        source.replace(
            "pilot_splits: [train, development]", "pilot_splits: [test]"
        ),
        encoding="utf-8",
    )
    key = tmp_path / "hmac.key"
    key.write_bytes(b"manifest-test-key-material-at-least-32-bytes")
    key.chmod(0o600)
    with pytest.raises(ValueError, match="pilot_splits"):
        build_manifests(vad, tmp_path / "manifest.jsonl", config, hmac_key=key)


def test_failed_publication_removes_partial_outputs(tmp_path: Path, monkeypatch) -> None:
    import experiments.silence_graft.build_manifest as module

    original = module.os.link
    calls = 0

    def fail_second_link(source, destination, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("simulated publication failure")
        return original(source, destination, **kwargs)

    monkeypatch.setattr(module.os, "link", fail_second_link)
    artifacts = [
        (tmp_path / "one.json", b"one"),
        (tmp_path / "two.json", b"two"),
    ]
    with pytest.raises(OSError, match="simulated"):
        _publish_artifacts(artifacts)
    assert not any(path.exists() for path, _ in artifacts)
    assert not list(tmp_path.glob(".*.tmp"))
