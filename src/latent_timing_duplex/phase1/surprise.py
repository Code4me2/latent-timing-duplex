"""Latent surprise = prediction error, wrapped for the Phase 0 harness.

``surprise_values`` is a per-chunk scalar. ``surprise_to_chunks`` builds
``ChunkSignal`` rows. ``score_surprise`` calls ``eval.harness.score_signal``
so JEPA surprise, Moshi NLL, and VAP share the same turn-event metrics.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from latent_timing_duplex.eval.harness import DEFAULT_HORIZONS_S, score_signal, score_session_bundle
from latent_timing_duplex.phase1.horizons import CHUNK_DURATION_S, pair_indices
from latent_timing_duplex.types import ChunkSignal, DualChannelSession, HorizonScore, iter_chunks


def surprise_values(
    pred: np.ndarray,
    target: np.ndarray,
    kind: str = "mse",
) -> np.ndarray:
    """Per-row surprise. ``pred`` / ``target`` are ``[N, D]``."""
    p = np.asarray(pred, dtype=np.float64)
    t = np.asarray(target, dtype=np.float64)
    if p.shape != t.shape:
        raise ValueError(f"pred shape {p.shape} != target shape {t.shape}")
    if p.ndim != 2:
        raise ValueError(f"expected [N, D], got {p.shape}")
    if kind == "mse":
        return np.mean((p - t) ** 2, axis=1)
    if kind == "l2":
        return np.sqrt(np.sum((p - t) ** 2, axis=1))
    raise ValueError(f"unknown surprise kind {kind!r}; expected 'mse' or 'l2'")


def surprise_to_chunks(
    values: Sequence[float],
    chunk_duration_s: float = CHUNK_DURATION_S,
    name: str = "jepa:surprise",
    t0: float = 0.0,
) -> list[ChunkSignal]:
    """Wrap a surprise series as harness ``ChunkSignal``s.

    Values are aligned to the *source* chunk end times (the moment the
    prediction is made). ``t0`` shifts the grid (equal-length crops).
    """
    n = len(values)
    if n == 0:
        raise ValueError("surprise values must be non-empty")
    if t0 == 0.0:
        return iter_chunks(
            duration_s=n * chunk_duration_s,
            chunk_duration_s=chunk_duration_s,
            values=[float(v) for v in values],
            name=name,
        )
    chunks: list[ChunkSignal] = []
    for i, value in enumerate(values):
        chunks.append(
            ChunkSignal(
                t_start=t0 + i * chunk_duration_s,
                t_end=t0 + (i + 1) * chunk_duration_s,
                value=float(value),
                name=name,
            )
        )
    return chunks


def surprise_from_sequences(
    hidden: np.ndarray,
    target: np.ndarray,
    predict_fn,
    horizon_s: float,
    chunk_duration_s: float = CHUNK_DURATION_S,
    kind: str = "mse",
    name: str = "jepa:surprise",
) -> list[ChunkSignal]:
    """Predict ``z_{t+h}`` from ``h_t`` and emit source-aligned surprise."""
    src, tgt = pair_indices(len(hidden), horizon_s, chunk_duration_s)
    if src.size == 0:
        raise ValueError("not enough chunks for this horizon")
    pred = np.asarray(predict_fn(np.asarray(hidden)[src]), dtype=np.float64)
    values = surprise_values(pred, np.asarray(target)[tgt], kind=kind)
    return surprise_to_chunks(values.tolist(), chunk_duration_s=chunk_duration_s, name=name)


def score_surprise(
    session: DualChannelSession,
    surprise: Sequence[ChunkSignal],
    horizons_s: Sequence[float] = DEFAULT_HORIZONS_S,
) -> list[HorizonScore]:
    """Hook surprise into the existing turn-event eval harness."""
    return score_signal(session, surprise, horizons_s=horizons_s)


def score_surprise_bundle(
    session: DualChannelSession,
    signals: dict[str, Sequence[ChunkSignal]],
    horizons_s: Sequence[float] = DEFAULT_HORIZONS_S,
) -> dict[str, list[HorizonScore]]:
    """Score surprise plus Phase 0 baselines (NLL, VAP) on one session."""
    return score_session_bundle(session, signals, horizons_s=horizons_s)
