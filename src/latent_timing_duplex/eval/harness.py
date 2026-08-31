"""Score any per-chunk scalar as a predictor of upcoming turn events.

For each chunk ending at ``t_end`` and each horizon ``h``, the binary label
is whether an event of the requested kind occurs in ``(t_end, t_end + h]``.
Chunks whose horizon window would run past the session duration are dropped
so every label is fully observed.

The harness is the one Phase 0 component that is implemented, not stubbed:
it is the gate for "is this implicit signal predictive?"
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from latent_timing_duplex.eval.metrics import average_precision, auroc
from latent_timing_duplex.types import (
    TURN_EVENT_KINDS,
    ChunkSignal,
    DualChannelSession,
    HorizonScore,
    TurnEventKind,
)

DEFAULT_HORIZONS_S = (0.16, 0.32, 0.50, 1.00, 2.00)


def _event_times(session: DualChannelSession, kind: TurnEventKind) -> np.ndarray:
    return np.array([e.t for e in session.events if e.kind == kind], dtype=np.float64)


def _labels_for_horizon(
    t_ends: np.ndarray,
    event_times: np.ndarray,
    horizon_s: float,
    duration_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (valid_mask, y) for one horizon."""
    valid = t_ends + horizon_s <= duration_s
    if event_times.size == 0:
        return valid, np.zeros(t_ends.shape, dtype=np.int32)
    # Event in (t_end, t_end + horizon]
    lo = t_ends[:, None]
    hi = lo + horizon_s
    inside = (event_times[None, :] > lo) & (event_times[None, :] <= hi)
    y = inside.any(axis=1).astype(np.int32)
    return valid, y


def score_signal(
    session: DualChannelSession,
    signal: Sequence[ChunkSignal],
    horizons_s: Sequence[float] = DEFAULT_HORIZONS_S,
    event_kinds: Sequence[TurnEventKind] = TURN_EVENT_KINDS,
) -> list[HorizonScore]:
    """Map one per-chunk signal onto turn-event scores at multiple horizons."""
    if not signal:
        raise ValueError("signal must contain at least one chunk")
    t_ends = np.array([c.t_end for c in signal], dtype=np.float64)
    values = np.array([c.value for c in signal], dtype=np.float64)
    scores: list[HorizonScore] = []
    for kind in event_kinds:
        times = _event_times(session, kind)
        for horizon in horizons_s:
            valid, y = _labels_for_horizon(t_ends, times, float(horizon), session.duration_s)
            y_v = y[valid]
            s_v = values[valid]
            n_pos = int(y_v.sum()) if y_v.size else 0
            n_neg = int(y_v.size - n_pos) if y_v.size else 0
            scores.append(
                HorizonScore(
                    horizon_s=float(horizon),
                    event_kind=kind,
                    n_chunks=int(y_v.size),
                    n_positives=n_pos,
                    n_negatives=n_neg,
                    auroc=auroc(y_v, s_v) if y_v.size else None,
                    average_precision=average_precision(y_v, s_v) if y_v.size else None,
                )
            )
    return scores


def score_session_bundle(
    session: DualChannelSession,
    signals: dict[str, Sequence[ChunkSignal]],
    horizons_s: Sequence[float] = DEFAULT_HORIZONS_S,
    event_kinds: Sequence[TurnEventKind] = TURN_EVENT_KINDS,
) -> dict[str, list[HorizonScore]]:
    """Score several named signals on the same session and event set."""
    return {
        name: score_signal(session, sig, horizons_s=horizons_s, event_kinds=event_kinds)
        for name, sig in signals.items()
    }


def format_score_table(named: dict[str, list[HorizonScore]]) -> str:
    """Plain-text table for the CLI."""
    header = (
        f"{'signal':<22} {'event':<13} {'h(s)':>6} "
        f"{'n+':>5} {'n-':>5} {'AUROC':>7} {'AP':>7}"
    )
    lines = [header, "-" * len(header)]
    for name, rows in named.items():
        for row in rows:
            auroc_s = "   n/a" if row.auroc is None else f"{row.auroc:7.3f}"
            ap_s = "   n/a" if row.average_precision is None else f"{row.average_precision:7.3f}"
            lines.append(
                f"{name:<22} {row.event_kind:<13} {row.horizon_s:6.2f} "
                f"{row.n_positives:5d} {row.n_negatives:5d} {auroc_s} {ap_s}"
            )
    return "\n".join(lines)
