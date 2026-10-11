"""Extract causal checkpoints after user speech from two-channel VAD."""

from __future__ import annotations

import hashlib
import math
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
    duration_s = float(duration_s)
    if not math.isfinite(duration_s) or duration_s <= 0:
        raise ValueError("duration_s must be finite and positive")
    checkpoints = sorted({float(x) for x in checkpoints_s})
    if not checkpoints or any(not math.isfinite(x) or x <= 0 for x in checkpoints):
        raise ValueError("checkpoints_s must contain finite positive values")
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
    max_per_lineage: int,
    seed: int,
) -> list[SilenceCheckpoint]:
    """Deterministically cap whole episodes across every session in a lineage."""
    if max_per_lineage < 1:
        raise ValueError("max_per_lineage must be positive")
    grouped: dict[str, dict[str, list[SilenceCheckpoint]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in rows:
        lineage_id = row.lineage_id or row.session_id
        episode = grouped[lineage_id][row.episode_id]
        if episode and episode[0].session_id != row.session_id:
            raise ValueError(f"episode_id {row.episode_id!r} spans sessions")
        episode.append(row)
    selected: list[SilenceCheckpoint] = []
    for lineage_id in sorted(grouped):
        episodes = grouped[lineage_id]
        ranked = sorted(
            episodes,
            key=lambda episode_id: (
                hashlib.sha256(
                    f"{seed}:{lineage_id}:{episode_id}".encode()
                ).digest(),
                episode_id,
            ),
        )[:max_per_lineage]
        for episode_id in sorted(ranked):
            selected.extend(sorted(episodes[episode_id], key=lambda row: row.elapsed_s))
    return selected


def _merge_intervals(intervals: Iterable[Interval], duration_s: float) -> list[Interval]:
    cleaned: list[Interval] = []
    for start, end in intervals:
        start, end = float(start), float(end)
        if not math.isfinite(start) or not math.isfinite(end):
            raise ValueError("VAD timestamps must be finite")
        if start < 0 or end > duration_s:
            raise ValueError("VAD timestamps must lie within the session duration")
        if start >= end:
            raise ValueError("VAD interval start must precede end")
        cleaned.append((start, end))
    cleaned.sort()
    merged: list[list[float]] = []
    for start, end in cleaned:
        if merged and start < merged[-1][1]:
            raise ValueError("within-channel VAD intervals must not overlap")
        if merged and start == merged[-1][1]:
            merged[-1][1] = end
        else:
            merged.append([start, end])
    return [(start, end) for start, end in merged]


def _active_at(intervals: Sequence[Interval], t: float) -> bool:
    return any(start <= t < end for start, end in intervals)
