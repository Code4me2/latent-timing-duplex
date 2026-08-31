"""Ranking metrics with no sklearn dependency."""

from __future__ import annotations

import numpy as np


def auroc(y_true: np.ndarray, y_score: np.ndarray) -> float | None:
    """ROC AUC via pairwise comparison. ``None`` if a class is missing."""
    y_true = np.asarray(y_true, dtype=np.int32)
    y_score = np.asarray(y_score, dtype=np.float64)
    pos = y_score[y_true == 1]
    neg = y_score[y_true == 0]
    if pos.size == 0 or neg.size == 0:
        return None
    # Broadcast pairwise: ties count as 0.5.
    gt = (pos[:, None] > neg[None, :]).sum()
    eq = (pos[:, None] == neg[None, :]).sum()
    return float((gt + 0.5 * eq) / (pos.size * neg.size))


def average_precision(y_true: np.ndarray, y_score: np.ndarray) -> float | None:
    """Average precision. ``None`` if there are no positives."""
    y_true = np.asarray(y_true, dtype=np.int32)
    y_score = np.asarray(y_score, dtype=np.float64)
    n_pos = int(y_true.sum())
    if n_pos == 0 or y_true.size == n_pos:
        return None
    order = np.argsort(-y_score, kind="mergesort")
    ranked = y_true[order]
    hits = np.cumsum(ranked)
    precision_at_hit = hits[ranked == 1] / (np.flatnonzero(ranked == 1) + 1)
    return float(precision_at_hit.mean())
