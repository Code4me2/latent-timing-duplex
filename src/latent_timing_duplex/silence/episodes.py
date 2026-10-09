"""Extract causal checkpoints after user speech from two-channel VAD."""

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Iterable, Sequence

from .types import SilenceCheckpoint, SilenceContext

Interval = tuple[float, float]
DEFAULT_CHECKPOINTS_S = (0.24, 0.48, 0.96, 2.0, 3.0, 5.0, 10.0)


def extract_silence_checkpoints(
    session_id: str,
    duration_s: float,
    user_intervals: Iterable[Interval],
    assistant_intervals: Iterable[Interval],
    *,
    checkpoints_s: Sequence[float] = DEFAULT_CHECKPOINTS_S,
    source: str = "unknown",
    lineage_id: str | None = None,
) -> list[SilenceCheckpoint]:
    """Emit checkpoints between each user offset and the next user onset.

    ``assistant_intervals`` determines whether a checkpoint is joint silence or
    an ordinary LISTEN period. No future observation is placed in model inputs;
    ``silence_end_s`` and ``right_censored`` are outcome metadata only.
    """
    if duration_s <= 0:
        raise ValueError("duration_s must be positive")
    checkpoints = sorted({float(x) for x in checkpoints_s})
    if not checkpoints or checkpoints[0] <= 0:
        raise ValueError("checkpoints_s must contain positive values")
    user = _merge_intervals(user_intervals, duration_s)
    assistant = _merge_intervals(assistant_intervals, duration_s)
    rows: list[SilenceCheckpoint] = []
    for index, (_, onset) in enumerate(user):
        next_onset = user[index + 1][0] if index + 1 < len(user) else duration_s
        if next_onset <= onset:
            continue
        episode_id = f"{session_id}:{index:05d}"
        censored = index + 1 == len(user)
        for elapsed in checkpoints:
            cutoff = onset + elapsed
            if cutoff >= next_onset - 1e-9:
                break
            context = (
                SilenceContext.LISTEN
                if _active_at(assistant, cutoff)
                else SilenceContext.JOINT
            )
            rows.append(
                SilenceCheckpoint(
                    session_id=session_id,
                    episode_id=episode_id,
                    onset_s=onset,
                    silence_end_s=next_onset,
                    model_cutoff_s=cutoff,
                    elapsed_s=elapsed,
                    context=context,
                    right_censored=censored,
                    source=source,
                    lineage_id=lineage_id or session_id,
                )
            )
    return rows


def sample_episodes(
    rows: Sequence[SilenceCheckpoint],
    *,
    max_per_session: int,
    seed: int,
) -> list[SilenceCheckpoint]:
    """Deterministically cap whole episodes without dropping their checkpoints."""
    if max_per_session < 1:
        raise ValueError("max_per_session must be positive")
    grouped: dict[str, dict[str, list[SilenceCheckpoint]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in rows:
        grouped[row.session_id][row.episode_id].append(row)
    selected: list[SilenceCheckpoint] = []
    for session_id in sorted(grouped):
        episodes = grouped[session_id]
        ids = sorted(episodes)
        if len(ids) > max_per_session:
            rng = random.Random(f"{seed}:{session_id}")
            ids = sorted(rng.sample(ids, max_per_session))
        for episode_id in ids:
            selected.extend(sorted(episodes[episode_id], key=lambda row: row.elapsed_s))
    return selected


def _merge_intervals(intervals: Iterable[Interval], duration_s: float) -> list[Interval]:
    cleaned: list[Interval] = []
    for start, end in intervals:
        a = max(0.0, float(start))
        b = min(duration_s, float(end))
        if b > a:
            cleaned.append((a, b))
    cleaned.sort()
    merged: list[list[float]] = []
    for start, end in cleaned:
        if merged and start <= merged[-1][1] + 1e-9:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(start, end) for start, end in merged]


def _active_at(intervals: Sequence[Interval], t: float) -> bool:
    return any(start <= t < end for start, end in intervals)
