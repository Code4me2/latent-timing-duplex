"""In-memory synthetic dual-channel timelines for the eval harness.

No waveforms are written to disk. Events are placed on a timeline and an
optional per-chunk scalar is peaked near those events so the harness has
something predictive to score against a random control.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from latent_timing_duplex.types import (
    TURN_EVENT_KINDS,
    ChunkSignal,
    DualChannelSession,
    TurnEvent,
    TurnEventKind,
    iter_chunks,
)

DEFAULT_CHUNK_S = 0.08


@dataclass(frozen=True)
class SyntheticBundle:
    session: DualChannelSession
    predictive: list[ChunkSignal]
    random: list[ChunkSignal]


def generate_synthetic_session(
    duration_s: float = 120.0,
    chunk_duration_s: float = DEFAULT_CHUNK_S,
    seed: int = 0,
    n_turn_shifts: int = 24,
    n_backchannels: int = 18,
    n_barge_ins: int = 8,
) -> SyntheticBundle:
    """Build a labeled session plus predictive and random chunk signals."""
    rng = np.random.default_rng(seed)
    counts: dict[TurnEventKind, int] = {
        "turn_shift": n_turn_shifts,
        "backchannel": n_backchannels,
        "barge_in": n_barge_ins,
    }
    events: list[TurnEvent] = []
    # Keep events away from the very start/end so every horizon has room.
    lo, hi = 1.0, max(1.5, duration_s - 2.5)
    for kind in TURN_EVENT_KINDS:
        times = rng.uniform(lo, hi, size=counts[kind])
        for t in sorted(times.tolist()):
            speaker = "user" if kind != "backchannel" else "assistant"
            events.append(TurnEvent(t=float(t), kind=kind, speaker=speaker))
    events.sort(key=lambda e: e.t)

    session = DualChannelSession(
        session_id=f"synthetic-{seed}",
        duration_s=duration_s,
        sample_rate=None,
        events=events,
        source="synthetic",
        notes="In-memory timeline only. No audio files.",
    )
    predictive = _predictive_signal(session, chunk_duration_s)
    n = len(predictive)
    random_values = rng.normal(size=n).tolist()
    random = iter_chunks(
        duration_s=n * chunk_duration_s,
        chunk_duration_s=chunk_duration_s,
        values=random_values,
        name="random",
    )
    return SyntheticBundle(session=session, predictive=predictive, random=random)


def _predictive_signal(
    session: DualChannelSession,
    chunk_duration_s: float,
    half_life_s: float = 0.35,
) -> list[ChunkSignal]:
    """Higher values on chunks whose end is close to a future/near event."""
    n = int(session.duration_s / chunk_duration_s)
    values = np.full(n, 0.05, dtype=np.float64)
    event_times = np.array([e.t for e in session.events], dtype=np.float64)
    for i in range(n):
        t_end = (i + 1) * chunk_duration_s
        future = event_times[event_times > t_end]
        if future.size == 0:
            continue
        dt = float(future.min() - t_end)
        if dt <= 2.0:
            values[i] = float(np.exp(-dt / half_life_s))
    return iter_chunks(
        duration_s=n * chunk_duration_s,
        chunk_duration_s=chunk_duration_s,
        values=values.tolist(),
        name="synthetic_salience",
    )
