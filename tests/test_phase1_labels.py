"""Transcript / VAD proxies and gold JSONL event sources."""

from __future__ import annotations

import json

from latent_timing_duplex.phase1.labels import (
    JsonlEventSource,
    TranscriptProxySource,
    VadProxySource,
    TranscriptTurn,
    events_from_intervals,
    events_from_turns,
    normalize_session_id,
)


def test_normalize_session_id_strips_window_suffix() -> None:
    assert normalize_session_id("abc:mid180") == "abc"
    assert normalize_session_id("abc:first180") == "abc"
    assert normalize_session_id("abc") == "abc"


def test_events_from_intervals_classifies_three_kinds() -> None:
    intervals = [
        ("user", 0.0, 2.0),
        ("assistant", 2.1, 3.5),  # turn_shift after user ends
        ("user", 2.4, 2.7),  # short overlap → backchannel
        ("user", 4.0, 6.0),  # floor was assistant → turn_shift
        ("assistant", 4.5, 6.5),  # long overlap → barge_in
    ]
    events = events_from_intervals(intervals)
    kinds = [(round(e.t, 2), e.kind) for e in events]
    assert (2.1, "turn_shift") in kinds
    assert (2.4, "backchannel") in kinds
    assert (4.0, "turn_shift") in kinds
    assert (4.5, "barge_in") in kinds


def test_events_from_turns_matches_intervals() -> None:
    turns = [
        TranscriptTurn("A", 0.0, 1.0),
        TranscriptTurn("B", 1.2, 2.0),
    ]
    events = events_from_turns(turns)
    assert len(events) == 1
    assert events[0].kind == "turn_shift"
    assert events[0].speaker == "B"


def test_jsonl_event_source(tmp_path) -> None:
    path = tmp_path / "events.jsonl"
    path.write_text(
        json.dumps(
            {
                "session_id": "s1:mid180",
                "events": [{"t": 1.5, "kind": "turn_shift", "speaker": "user"}],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    src = JsonlEventSource.from_path(path)
    rows = src.events_for("s1")
    assert len(rows) == 1
    assert rows[0].kind == "turn_shift"
    assert rows[0].t == 1.5


def test_transcript_and_vad_sources(tmp_path) -> None:
    tr = tmp_path / "turns.json"
    tr.write_text(
        json.dumps(
            {
                "session_id": "conv",
                "turns": [
                    {"speaker": "user", "start": 0.0, "end": 1.0},
                    {"speaker": "assistant", "start": 1.2, "end": 2.0},
                ],
            }
        ),
        encoding="utf-8",
    )
    vad = tmp_path / "vad.jsonl"
    vad.write_text(
        json.dumps(
            {
                "uuid": "conv",
                "user_vad": [[0.0, 1.0]],
                "assistant_vad": [[1.2, 2.0]],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    t_events = TranscriptProxySource.from_path(tr).events_for("conv")
    v_events = VadProxySource.from_path(vad).events_for("conv")
    assert [e.kind for e in t_events] == ["turn_shift"]
    assert [e.kind for e in v_events] == ["turn_shift"]
