"""Causal silence episode, annotation, and gate checks."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from latent_timing_duplex.silence import (
    CausalFeatureRecord,
    GateState,
    SilenceAction,
    SilenceCheckpoint,
    SilenceContext,
    SilenceDecision,
    SilenceGate,
    extract_silence_checkpoints,
    load_annotations,
    parse_annotation,
    sample_episodes,
)


ROOT = Path(__file__).resolve().parents[1]


def test_labeling_protocol_locks_prefix_before_reveal() -> None:
    protocol = (ROOT / "docs" / "DATA_LABELING_PROTOCOL.md").read_text()
    normalized = " ".join(protocol.split()).lower()
    assert "future user resumption" in normalized
    assert "after the primary answer is submitted and locked" in normalized
    assert "never revise or overwrite the prefix answer" in normalized
    assert "splits by `lineage_id`" in normalized


def test_semantic_silence_documentation_lock() -> None:
    protocol = (ROOT / "docs" / "DATA_LABELING_PROTOCOL.md").read_text()
    graft = (ROOT / "docs" / "SILENCE_GRAFT.md").read_text()
    roadmap = (ROOT / "docs" / "RESEARCH_ROADMAP.md").read_text()
    experiment = (ROOT / "experiments" / "silence_graft" / "README.md").read_text()
    readme = (ROOT / "README.md").read_text()

    for text in (protocol, graft, roadmap, experiment, readme):
        assert "R1" in text
    protocol_line = " ".join(protocol.split())
    graft_line = " ".join(graft.split())
    roadmap_line = " ".join(roadmap.split())
    readme_line = " ".join(readme.split())
    gate_status = "gate remains open until a real pilot agreement receipt passes"
    assert gate_status in graft_line and gate_status in readme_line
    assert "R1 is scaffolded but its gate is open" in roadmap_line
    assert "analysis-lock receipt" in protocol_line
    assert "checkpoint elapsed time, never realized silence" in protocol_line
    assert "Target 75 episodes" in protocol_line
    assert "60–90" in protocol_line and "at least 30 source lineages" in protocol_line
    assert "train/development episodes only" in protocol_line
    assert "fixed 20-second prefix" in protocol_line
    assert "fixed 8 seconds of the user channel" in protocol_line
    assert "nearest-sample rounding" in protocol_line
    assert "gain_db: 0.0" in protocol_line and "show no transcript" in protocol_line
    assert "ACK     -> RESPOND" in protocol
    assert "UNRATABLE -> missing for nominal action alpha" in protocol_line
    assert "cannot be relabeled or reused" in protocol_line
    assert "nominal Krippendorff's alpha" in protocol_line
    assert "per-class positive agreement" in protocol_line
    assert "onset endpoint disagreement" in graft_line
    assert "Repeat 5%" in graft_line
    assert "seed `20261009` and 10,000 replicates" in protocol_line
    assert "Do not insert automatic sentinels" in protocol_line
    assert "normative, interval-censored assistant-entry supervision" in protocol_line
    assert "user-resumption/assistant-entry tie" in protocol_line
    assert "PR #" not in roadmap and "future PR" not in roadmap


def _decision(action: SilenceAction, probability: float = 0.8) -> SilenceDecision:
    probabilities = {
        SilenceAction.WAIT: 1.0 - probability,
        SilenceAction.ACK: 0.0,
        SilenceAction.RESPOND: 0.0,
        SilenceAction.REPAIR: 0.0,
    }
    probabilities[action] = probability
    return SilenceDecision(probabilities, 0.2, 0.7)


def test_extracts_causal_joint_and_listen_checkpoints() -> None:
    rows = extract_silence_checkpoints(
        "s1",
        4.0,
        user_intervals=[(0.0, 1.0), (2.0, 3.0)],
        assistant_intervals=[(1.4, 1.8)],
        checkpoints_s=[0.24, 0.48, 0.96],
    )
    first = [row for row in rows if row.episode_id == "s1:00000"]
    assert [row.context for row in first] == [
        SilenceContext.JOINT,
        SilenceContext.LISTEN,
        SilenceContext.JOINT,
    ]
    assert all(row.model_cutoff_s == pytest.approx(row.onset_s + row.elapsed_s) for row in rows)
    assert rows[-1].right_censored


def test_future_activity_cannot_change_an_existing_checkpoint() -> None:
    common = dict(
        session_id="s1",
        duration_s=4.0,
        user_intervals=[(0.0, 1.0), (3.0, 3.5)],
        checkpoints_s=[0.48],
    )
    before = extract_silence_checkpoints(assistant_intervals=[], **common)[0]
    after = extract_silence_checkpoints(assistant_intervals=[(2.0, 2.5)], **common)[0]
    assert before.model_cutoff_s == after.model_cutoff_s == pytest.approx(1.48)
    assert before.context == after.context == SilenceContext.JOINT


def test_checkpoint_at_user_resumption_is_not_silence() -> None:
    rows = extract_silence_checkpoints(
        "s1",
        3.0,
        [(0.0, 1.0), (2.0, 2.5)],
        [],
        checkpoints_s=[1.0],
    )
    assert rows == []


def test_sampling_keeps_whole_episodes_and_is_deterministic() -> None:
    rows = extract_silence_checkpoints(
        "s1",
        7.0,
        [(0, 1), (2, 3), (4, 5)],
        [],
        checkpoints_s=[0.24, 0.48],
    )
    first = sample_episodes(rows, max_per_lineage=2, seed=7)
    second = sample_episodes(rows, max_per_lineage=2, seed=7)
    assert first == second
    assert len({row.episode_id for row in first}) == 2
    counts = {episode_id: 0 for episode_id in {row.episode_id for row in first}}
    for row in first:
        counts[row.episode_id] += 1
    assert set(counts.values()) == {2}


def test_acausal_and_nonfinite_features_are_rejected() -> None:
    with pytest.raises(ValueError, match="acausal"):
        CausalFeatureRecord("e1", 2.0, 2.1, (1.0,), "future-surprise")
    with pytest.raises(ValueError, match="timestamps must be finite"):
        CausalFeatureRecord("e1", 2.0, float("nan"), (1.0,), "nan-time")
    with pytest.raises(ValueError, match="values must be non-empty and finite"):
        CausalFeatureRecord("e1", 2.0, 2.0, (float("nan"),), "nan-value")
    for cutoff, available in ((float("nan"), 1.0), (1.0, float("inf"))):
        with pytest.raises(ValueError, match="finite"):
            CausalFeatureRecord("e1", cutoff, available, (1.0,), "invalid-time")


def test_checkpoint_rejects_non_finite_timestamps() -> None:
    with pytest.raises(ValueError, match="finite"):
        SilenceCheckpoint(
            "s1",
            "e1",
            0.0,
            2.0,
            float("nan"),
            1.0,
            SilenceContext.JOINT,
        )


def test_annotation_schema_accepts_wait_and_rejects_future_contract_errors() -> None:
    row = parse_annotation(
        {
            "episode_id": "e1",
            "checkpoint_elapsed_s": 0.96,
            "annotator": "a1",
            "action": "wait",
            "confidence": 4,
            "also_acceptable": ["ack"],
            "tags": ["floor_holding"],
            "acceptable_onset": {
                "earliest_elapsed_s": None,
                "latest_elapsed_s": None,
                "wait_for_user_resume": True,
            },
            "quality": "clean",
            "ratable": True,
        }
    )
    assert row.action is SilenceAction.WAIT
    assert row.confidence == 4
    assert row.also_acceptable == (SilenceAction.ACK,)
    with pytest.raises(ValueError, match="cannot precede"):
        parse_annotation(
            {
                "episode_id": "e1",
                "checkpoint_elapsed_s": 2.0,
                "annotator": "a1",
                "action": "respond",
                "tags": [],
                "acceptable_onset": {
                    "earliest_elapsed_s": 1.0,
                    "latest_elapsed_s": 1.5,
                },
                "quality": "clean",
            }
        )


def test_unratable_annotation_has_no_onset_target() -> None:
    row = parse_annotation(
        {
            "episode_id": "e1",
            "checkpoint_elapsed_s": 0.96,
            "annotator": "a1",
            "action": "unratable",
            "tags": ["technical_uncertainty"],
            "acceptable_onset": None,
            "quality": "dropout_mute_or_edit",
            "ratable": False,
        }
    )
    assert row.acceptable_onset is None


def test_annotation_loader_rejects_duplicate_annotator(tmp_path) -> None:
    obj = {
        "episode_id": "e1",
        "checkpoint_elapsed_s": 0.48,
        "annotator": "a1",
        "action": "wait",
        "tags": [],
        "acceptable_onset": {"wait_for_user_resume": True},
        "quality": "clean",
    }
    path = tmp_path / "annotations.jsonl"
    path.write_text(json.dumps(obj) + "\n" + json.dumps(obj) + "\n")
    with pytest.raises(ValueError, match="duplicate"):
        load_annotations(path)


def test_gate_requires_sustained_evidence_and_user_resumption_cancels() -> None:
    gate = SilenceGate(threshold=0.6, consecutive_frames=2)
    respond = _decision(SilenceAction.RESPOND)
    assert gate.step(respond, user_active=False, assistant_active=False) is GateState.HOLD
    assert gate.step(respond, user_active=False, assistant_active=False) is GateState.RESPOND
    assert gate.step(respond, user_active=True, assistant_active=False) is GateState.LISTEN


def test_gate_holds_on_uncertain_decision() -> None:
    gate = SilenceGate(threshold=0.7)
    decision = _decision(SilenceAction.ACK, probability=0.6)
    assert gate.step(decision, user_active=False, assistant_active=False) is GateState.HOLD
