"""Pluggable turn-event labels plus transcript / VAD proxies.

Gold annotations (JSONL of ``TurnEvent`` rows) are preferred. When those are
missing, this module derives *proxies* from transcript turns or VAD intervals:

* ``turn_shift`` — other-speaker onset after the current speaker has ended
* ``barge_in`` — long other-speaker onset while the current speaker is active
* ``backchannel`` — short other-speaker burst (default ≤ 0.6 s) during or
  immediately after the current speaker

These proxies are **not** CANDOR / DuplexChat official turn annotations.
Overlap rules, backchannel duration, and missing speaker-identity on
transcripts all leak into the labels. Report them as such.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from latent_timing_duplex.types import TURN_EVENT_KINDS, DualChannelSession, TurnEvent, TurnEventKind

DEFAULT_BACKCHANNEL_MAX_S = 0.6
DEFAULT_BACKCHANNEL_GAP_S = 0.25
TURN_KEYS = ("turns", "utterances", "segments", "items")
SPEAKER_KEYS = ("speaker", "spk", "role", "channel", "name")
START_KEYS = ("start", "start_s", "t_start", "startTime", "begin", "onset")
END_KEYS = ("end", "end_s", "t_end", "endTime", "stop", "offset")
KIND_ALIASES: dict[str, TurnEventKind] = {
    "turn_shift": "turn_shift",
    "turn-shift": "turn_shift",
    "shift": "turn_shift",
    "turn": "turn_shift",
    "backchannel": "backchannel",
    "back-channel": "backchannel",
    "bc": "backchannel",
    "barge_in": "barge_in",
    "barge-in": "barge_in",
    "bargein": "barge_in",
    "overlap": "barge_in",
}


class TurnEventSource(Protocol):
    """Resolve labeled events for one session id."""

    def events_for(self, session_id: str) -> list[TurnEvent]: ...


@dataclass(frozen=True)
class TranscriptTurn:
    speaker: str
    start: float
    end: float
    text: str = ""

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


@dataclass
class InMemoryEventSource:
    """Test / CLI helper: a dict of session_id → events."""

    events: dict[str, list[TurnEvent]]
    source_name: str = "memory"

    def events_for(self, session_id: str) -> list[TurnEvent]:
        key = normalize_session_id(session_id)
        if session_id in self.events:
            return list(self.events[session_id])
        if key in self.events:
            return list(self.events[key])
        raise KeyError(f"no events for session {session_id!r} (normalized {key!r})")


@dataclass
class JsonlEventSource:
    """Gold (or pre-derived) events. One JSON object per line or a JSON list."""

    path: Path
    _index: dict[str, list[TurnEvent]]

    @classmethod
    def from_path(cls, path: str | Path) -> JsonlEventSource:
        root = Path(path)
        if not root.is_file():
            raise FileNotFoundError(
                f"event label file {path!r} is missing. Pass --labels with a "
                "JSONL of session events, or use --transcripts / --vad for proxies."
            )
        return cls(path=root, _index=_load_event_index(root))

    def events_for(self, session_id: str) -> list[TurnEvent]:
        direct = self._index.get(session_id)
        if direct is not None:
            return list(direct)
        key = normalize_session_id(session_id)
        if key in self._index:
            return list(self._index[key])
        raise KeyError(f"{self.path} has no events for {session_id!r}")


@dataclass
class TranscriptProxySource:
    """Speaker-change proxies from transcript turns."""

    path: Path
    _turns: dict[str, list[TranscriptTurn]]
    backchannel_max_s: float = DEFAULT_BACKCHANNEL_MAX_S
    backchannel_gap_s: float = DEFAULT_BACKCHANNEL_GAP_S

    @classmethod
    def from_path(
        cls,
        path: str | Path,
        backchannel_max_s: float = DEFAULT_BACKCHANNEL_MAX_S,
        backchannel_gap_s: float = DEFAULT_BACKCHANNEL_GAP_S,
    ) -> TranscriptProxySource:
        root = Path(path)
        if not root.is_file():
            raise FileNotFoundError(
                f"transcript file {path!r} is missing. Expected JSON/JSONL turns "
                "with speaker + start/end seconds."
            )
        return cls(
            path=root,
            _turns=_load_turn_index(root),
            backchannel_max_s=backchannel_max_s,
            backchannel_gap_s=backchannel_gap_s,
        )

    def events_for(self, session_id: str) -> list[TurnEvent]:
        turns = _lookup(self._turns, session_id)
        return events_from_turns(
            turns,
            backchannel_max_s=self.backchannel_max_s,
            backchannel_gap_s=self.backchannel_gap_s,
        )


@dataclass
class VadProxySource:
    """Silence→speech / overlap proxies from per-speaker VAD intervals."""

    path: Path
    _intervals: dict[str, list[tuple[str, float, float]]]
    backchannel_max_s: float = DEFAULT_BACKCHANNEL_MAX_S
    backchannel_gap_s: float = DEFAULT_BACKCHANNEL_GAP_S

    @classmethod
    def from_path(
        cls,
        path: str | Path,
        backchannel_max_s: float = DEFAULT_BACKCHANNEL_MAX_S,
        backchannel_gap_s: float = DEFAULT_BACKCHANNEL_GAP_S,
    ) -> VadProxySource:
        root = Path(path)
        if not root.is_file():
            raise FileNotFoundError(
                f"VAD file {path!r} is missing. Expected JSON/JSONL with "
                "user/assistant (or LEFT/RIGHT) interval lists."
            )
        return cls(
            path=root,
            _intervals=_load_vad_index(root),
            backchannel_max_s=backchannel_max_s,
            backchannel_gap_s=backchannel_gap_s,
        )

    def events_for(self, session_id: str) -> list[TurnEvent]:
        intervals = _lookup(self._intervals, session_id)
        return events_from_intervals(
            intervals,
            backchannel_max_s=self.backchannel_max_s,
            backchannel_gap_s=self.backchannel_gap_s,
        )


@dataclass
class CompositeEventSource:
    """Prefer gold labels; fall back to the first source that has the session."""

    sources: list[TurnEventSource]

    def events_for(self, session_id: str) -> list[TurnEvent]:
        errors: list[str] = []
        for src in self.sources:
            try:
                return src.events_for(session_id)
            except KeyError as exc:
                errors.append(str(exc))
        raise KeyError(
            f"no turn-event source has {session_id!r}: " + " | ".join(errors)
        )


def normalize_session_id(session_id: str) -> str:
    """Strip equal-length window suffixes (``:mid180``, ``:first180``, …)."""
    name = str(session_id)
    for mode in ("mid", "first"):
        for window in (180, 300, 120, 60):
            suffix = f":{mode}{window}"
            if name.endswith(suffix):
                return name[: -len(suffix)]
    if ":" in name:
        head, tail = name.rsplit(":", 1)
        if tail in {"mid180", "first180", "mid", "first"}:
            return head
    return name


def events_from_turns(
    turns: list[TranscriptTurn],
    *,
    backchannel_max_s: float = DEFAULT_BACKCHANNEL_MAX_S,
    backchannel_gap_s: float = DEFAULT_BACKCHANNEL_GAP_S,
) -> list[TurnEvent]:
    """Map transcript turns onto the three Phase 0 event kinds (proxy)."""
    intervals = [(t.speaker, t.start, t.end) for t in turns if t.end > t.start]
    return events_from_intervals(
        intervals,
        backchannel_max_s=backchannel_max_s,
        backchannel_gap_s=backchannel_gap_s,
    )


def events_from_intervals(
    intervals: list[tuple[str, float, float]],
    *,
    backchannel_max_s: float = DEFAULT_BACKCHANNEL_MAX_S,
    backchannel_gap_s: float = DEFAULT_BACKCHANNEL_GAP_S,
) -> list[TurnEvent]:
    """Classify speaker intervals as turn-shift / backchannel / barge-in proxies.

    Limitations (honest):
    - No lexical backchannel detector (uh-huh vs. real content).
    - A short overlapping burst is always ``backchannel``, never ``barge_in``.
    - Simultaneous onsets prefer the earlier interval as "current" speaker.
    - Speaker strings are compared case-insensitively after strip.
    """
    cleaned: list[tuple[str, float, float]] = []
    for speaker, start, end in intervals:
        spk = str(speaker).strip() or "unknown"
        a, b = float(start), float(end)
        if b <= a:
            continue
        cleaned.append((spk, a, b))
    cleaned.sort(key=lambda row: (row[1], row[2], row[0]))
    events: list[TurnEvent] = []
    for i, (spk, start, end) in enumerate(cleaned):
        others = [
            row
            for j, row in enumerate(cleaned)
            if j != i and _norm_spk(row[0]) != _norm_spk(spk)
        ]
        overlapping = [row for row in others if row[1] < end and row[2] > start]
        covering = [row for row in overlapping if row[1] <= start < row[2]]
        recent = [
            row
            for row in others
            if row[2] <= start and (start - row[2]) <= backchannel_gap_s
        ]
        dur = end - start
        if covering and dur <= backchannel_max_s:
            events.append(TurnEvent(t=start, kind="backchannel", speaker=spk))
            continue
        if covering and dur > backchannel_max_s:
            events.append(TurnEvent(t=start, kind="barge_in", speaker=spk))
            continue
        if recent and dur <= backchannel_max_s and not covering:
            events.append(TurnEvent(t=start, kind="backchannel", speaker=spk))
            continue
        predecessors = [row for row in cleaned[:i] if row[1] < start]
        if not predecessors:
            # Conversation-initial onset is not a mid-dialogue shift.
            continue
        floor = max(predecessors, key=lambda row: (row[2], row[1]))
        if _norm_spk(floor[0]) != _norm_spk(spk):
            events.append(TurnEvent(t=start, kind="turn_shift", speaker=spk))
    events.sort(key=lambda e: (e.t, e.kind))
    return events


def attach_events(
    session: DualChannelSession,
    events: list[TurnEvent],
    *,
    source_note: str = "phase1-labels",
) -> DualChannelSession:
    """Return a copy of ``session`` with ``events`` (window-relative times)."""
    valid = [e for e in events if 0.0 <= e.t < session.duration_s]
    return DualChannelSession(
        session_id=session.session_id,
        duration_s=session.duration_s,
        sample_rate=session.sample_rate,
        user_audio=session.user_audio,
        assistant_audio=session.assistant_audio,
        events=valid,
        source=session.source,
        notes=(session.notes + f" | labels={source_note}").strip(" |"),
    )


def parse_event(obj: dict[str, Any]) -> TurnEvent:
    kind = _coerce_kind(obj.get("kind") or obj.get("event") or obj.get("type"))
    if kind is None:
        raise ValueError(f"cannot parse event kind from {obj!r}")
    t = obj.get("t", obj.get("time", obj.get("t_s", obj.get("onset"))))
    if t is None:
        raise ValueError(f"event missing time: {obj!r}")
    speaker = obj.get("speaker")
    if speaker is not None:
        speaker = str(speaker)
    return TurnEvent(t=float(t), kind=kind, speaker=speaker)


def _norm_spk(name: str) -> str:
    return str(name).strip().lower()


def _coerce_kind(raw: Any) -> TurnEventKind | None:
    if raw is None:
        return None
    key = str(raw).strip().lower().replace(" ", "_")
    if key in TURN_EVENT_KINDS:
        return key  # type: ignore[return-value]
    return KIND_ALIASES.get(key)


def _lookup(index: dict[str, Any], session_id: str) -> Any:
    if session_id in index:
        return index[session_id]
    key = normalize_session_id(session_id)
    if key in index:
        return index[key]
    raise KeyError(f"no entry for session {session_id!r}")


def _iter_json_records(path: Path) -> list[Any]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if path.suffix in {".jsonl", ".ndjson"} or "\n{" in text or text.startswith("{") and "\n" in text:
        rows: list[Any] = []
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return parsed
            if isinstance(parsed, dict):
                return [parsed]
        except json.JSONDecodeError:
            pass
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
        return rows
    parsed = json.loads(text)
    if isinstance(parsed, list):
        return parsed
    return [parsed]


def _session_id_from(obj: dict[str, Any]) -> str | None:
    for key in (
        "session_id",
        "conversation_id",
        "uuid",
        "clip_id",
        "episode_id",
        "id",
        "name",
    ):
        if key in obj and obj[key] is not None:
            return str(obj[key])
    return None


def _load_event_index(path: Path) -> dict[str, list[TurnEvent]]:
    index: dict[str, list[TurnEvent]] = {}
    for rec in _iter_json_records(path):
        if not isinstance(rec, dict):
            continue
        sid = _session_id_from(rec)
        payload = rec.get("events")
        if payload is None and "kind" in rec:
            if sid is None:
                continue
            index.setdefault(sid, []).append(parse_event(rec))
            continue
        if sid is None or payload is None:
            continue
        index.setdefault(sid, []).extend(parse_event(e) for e in payload)
    for sid, rows in list(index.items()):
        rows.sort(key=lambda e: (e.t, e.kind))
        index[normalize_session_id(sid)] = rows
    return index


def _first_key(obj: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in obj and obj[key] is not None:
            return obj[key]
    return None


def _parse_turn(obj: dict[str, Any], default_speaker: str = "unknown") -> TranscriptTurn | None:
    start = _first_key(obj, START_KEYS)
    end = _first_key(obj, END_KEYS)
    if start is None or end is None:
        return None
    speaker = _first_key(obj, SPEAKER_KEYS)
    text = obj.get("text") or obj.get("utterance") or ""
    return TranscriptTurn(
        speaker=str(speaker) if speaker is not None else default_speaker,
        start=float(start),
        end=float(end),
        text=str(text),
    )


def _load_turn_index(path: Path) -> dict[str, list[TranscriptTurn]]:
    index: dict[str, list[TranscriptTurn]] = {}
    for rec in _iter_json_records(path):
        if not isinstance(rec, dict):
            continue
        sid = _session_id_from(rec)
        turns_raw: Any = None
        for key in TURN_KEYS:
            if key in rec:
                turns_raw = rec[key]
                break
        if turns_raw is None and _first_key(rec, START_KEYS) is not None:
            if sid is None:
                sid = "unknown"
            parsed = _parse_turn(rec)
            if parsed is not None:
                index.setdefault(sid, []).append(parsed)
            continue
        if sid is None or turns_raw is None:
            continue
        parsed_list = [t for t in (_parse_turn(x) for x in turns_raw) if t is not None]
        index.setdefault(sid, []).extend(parsed_list)
    for sid, rows in list(index.items()):
        rows.sort(key=lambda t: (t.start, t.end))
        index[normalize_session_id(sid)] = rows
    return index


def _as_interval_list(raw: Any, speaker: str) -> list[tuple[str, float, float]]:
    out: list[tuple[str, float, float]] = []
    if raw is None:
        return out
    for item in raw:
        if isinstance(item, dict):
            start = _first_key(item, START_KEYS)
            end = _first_key(item, END_KEYS)
            if start is None or end is None:
                continue
            out.append((speaker, float(start), float(end)))
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            out.append((speaker, float(item[0]), float(item[1])))
    return out


def _load_vad_index(path: Path) -> dict[str, list[tuple[str, float, float]]]:
    index: dict[str, list[tuple[str, float, float]]] = {}
    for rec in _iter_json_records(path):
        if not isinstance(rec, dict):
            continue
        sid = _session_id_from(rec) or "unknown"
        intervals: list[tuple[str, float, float]] = []
        for key, speaker in (
            ("user_vad", "user"),
            ("user", "user"),
            ("left", "user"),
            ("LEFT", "user"),
            ("assistant_vad", "assistant"),
            ("assistant", "assistant"),
            ("agent_vad", "assistant"),
            ("right", "assistant"),
            ("RIGHT", "assistant"),
        ):
            if key in rec:
                intervals.extend(_as_interval_list(rec[key], speaker))
        if "intervals" in rec:
            for item in rec["intervals"]:
                if isinstance(item, dict):
                    parsed = _parse_turn(item)
                    if parsed is not None:
                        intervals.append((parsed.speaker, parsed.start, parsed.end))
        if not intervals:
            continue
        index.setdefault(sid, []).extend(intervals)
    for sid, rows in list(index.items()):
        rows.sort(key=lambda r: (r[1], r[2]))
        index[normalize_session_id(sid)] = rows
    return index


def describe_label_limitations() -> list[str]:
    return [
        "Transcript / VAD labels are proxies, not official CANDOR or DuplexChat "
        "turn annotations.",
        "backchannel = short other-speaker burst (default ≤ 0.6 s); there is no "
        "lexical filter.",
        "barge_in = longer other-speaker onset while the current speaker is still "
        "active; short overlaps are classified as backchannel instead.",
        "turn_shift = other-speaker onset after the previous speaker has ended; "
        "conversation-initial onsets are not counted.",
        "Prefer --labels gold JSONL when annotations exist; proxies are a fallback.",
    ]
