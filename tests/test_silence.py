"""Causal silence episode, annotation, and gate checks."""

from __future__ import annotations

import json

import pytest

from latent_timing_duplex.silence import (
    CausalFeatureRecord,
    GateState,
    SilenceAction,
    SilenceContext,
    SilenceDecision,
    SilenceGate,
    extract_silence_checkpoints,
    load_annotations,
    parse_annotation,
    sample_episodes,
)


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
    first = sample_episodes(rows, max_per_session=2, seed=7)
    second = sample_episodes(rows, max_per_session=2, seed=7)
    assert first == second
    assert len({row.episode_id for row in first}) == 2
    counts = {episode_id: 0 for episode_id in {row.episode_id for row in first}}
    for row in first:
        counts[row.episode_id] += 1
    assert set(counts.values()) == {2}


def test_acausal_feature_is_rejected() -> None:
    with pytest.raises(ValueError, match="acausal"):
        CausalFeatureRecord("e1", 2.0, 2.1, (1.0,), "future-surprise")


def test_annotation_schema_accepts_wait_and_rejects_future_contract_errors() -> None:
    row = parse_annotation(
        {
            "episode_id": "e1",
            "checkpoint_elapsed_s": 0.96,
            "annotator": "a1",
            "action": "wait",
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
