"""Shared dataclasses for dual-channel dialogue, chunk signals, and events."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Sequence

TurnEventKind = Literal["turn_shift", "backchannel", "barge_in"]
TURN_EVENT_KINDS: tuple[TurnEventKind, ...] = (
    "turn_shift",
    "backchannel",
    "barge_in",
)


@dataclass(frozen=True)
class TurnEvent:
    """A labeled turn-taking event at time ``t`` seconds."""

    t: float
    kind: TurnEventKind
    # Optional speaker that initiates the event (user / assistant).
    speaker: str | None = None


@dataclass(frozen=True)
class ChunkSignal:
    """One scalar per audio chunk. The harness does not care what produced it.

    Typical Phase 0 sources (once filled in): frozen user-channel NLL, VAP
    state probability, or a hand-crafted control. Phase 1 will add JEPA
    prediction error ("surprise") on the same interface.
    """

    t_start: float
    t_end: float
    value: float
    name: str = "signal"


@dataclass(frozen=True)
class HorizonScore:
    """How well a per-chunk signal predicts one event kind at one horizon."""

    horizon_s: float
    event_kind: TurnEventKind
    n_chunks: int
    n_positives: int
    n_negatives: int
    auroc: float | None
    average_precision: float | None


@dataclass
class DualChannelSession:
    """Speaker-separated dual-channel dialogue metadata.

    Phase 0 does not ship audio. ``user_audio`` / ``assistant_audio`` stay
    ``None`` unless a later pipeline loads them from a local path you provide.
    """

    session_id: str
    duration_s: float
    sample_rate: int | None = None
    user_audio: object | None = None
    assistant_audio: object | None = None
    events: list[TurnEvent] = field(default_factory=list)
    source: str = "unknown"
    notes: str = ""

    def events_of(self, kind: TurnEventKind) -> list[TurnEvent]:
        return [e for e in self.events if e.kind == kind]


def iter_chunks(
    duration_s: float,
    chunk_duration_s: float,
    values: Sequence[float] | None = None,
    name: str = "signal",
) -> list[ChunkSignal]:
    """Build a uniform chunk grid. ``values`` must match the chunk count if given."""
    if chunk_duration_s <= 0:
        raise ValueError("chunk_duration_s must be positive")
    n = int(duration_s / chunk_duration_s)
    if n <= 0:
        raise ValueError("duration_s must cover at least one chunk")
    if values is not None and len(values) != n:
        raise ValueError(f"expected {n} values, got {len(values)}")
    chunks: list[ChunkSignal] = []
    for i in range(n):
        value = float(values[i]) if values is not None else 0.0
        chunks.append(
            ChunkSignal(
                t_start=i * chunk_duration_s,
                t_end=(i + 1) * chunk_duration_s,
                value=value,
                name=name,
            )
        )
    return chunks
