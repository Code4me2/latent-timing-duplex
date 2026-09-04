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


def precision_recall_at_threshold(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float,
) -> tuple[float | None, float | None, float | None]:
    """Precision, recall, F1 at a score threshold (predict positive if ``>=``)."""
    y_true = np.asarray(y_true, dtype=np.int32)
    y_score = np.asarray(y_score, dtype=np.float64)
    if y_true.size == 0:
        return None, None, None
    pred = y_score >= threshold
    tp = int(np.logical_and(pred, y_true == 1).sum())
    fp = int(np.logical_and(pred, y_true == 0).sum())
    fn = int(np.logical_and(~pred, y_true == 1).sum())
    prec = float(tp / (tp + fp)) if (tp + fp) else None
    rec = float(tp / (tp + fn)) if (tp + fn) else None
    if prec is None or rec is None or (prec + rec) == 0.0:
        f1 = None
    else:
        f1 = float(2.0 * prec * rec / (prec + rec))
    return prec, rec, f1


def f1max_operating_point(
    y_true: np.ndarray,
    y_score: np.ndarray,
) -> tuple[float | None, float | None, float | None, float | None]:
    """Best-F1 threshold and the (precision, recall, F1) at that cut.

    Thresholds are unique scores (plus ``+inf`` so an all-negative cut exists).
    Ties keep the first (highest-score) threshold that achieves the max F1.
    """
    y_true = np.asarray(y_true, dtype=np.int32)
    y_score = np.asarray(y_score, dtype=np.float64)
    if y_true.size == 0 or y_true.min() == y_true.max():
        return None, None, None, None
    thresholds = np.unique(y_score)[::-1]
    best: tuple[float, float, float, float] | None = None
    for thr in thresholds:
        prec, rec, f1 = precision_recall_at_threshold(y_true, y_score, float(thr))
        if f1 is None:
            continue
        if best is None or f1 > best[3]:
            best = (float(thr), float(prec or 0.0), float(rec or 0.0), f1)
    if best is None:
        return None, None, None, None
    return best


def precision_at_recall(
    y_true: np.ndarray,
    y_score: np.ndarray,
    target_recall: float,
) -> float | None:
    """Highest precision whose recall is at least ``target_recall``.

    Walks the ranked list (high score first). ``None`` if that recall is
    unreachable (too few positives).
    """
    if not 0.0 < target_recall <= 1.0:
        raise ValueError(f"target_recall must be in (0, 1], got {target_recall}")
    y_true = np.asarray(y_true, dtype=np.int32)
    y_score = np.asarray(y_score, dtype=np.float64)
    n_pos = int(y_true.sum())
    if n_pos == 0:
        return None
    order = np.argsort(-y_score, kind="mergesort")
    ranked = y_true[order]
    hits = np.cumsum(ranked)
    recall = hits / n_pos
    precision = hits / np.arange(1, ranked.size + 1)
    ok = recall >= target_recall
    if not ok.any():
        return None
    return float(precision[ok].max())


def efron_bootstrap_mean_ci(
    values: np.ndarray,
    *,
    n_boot: int = 10000,
    seed: int = 20260903,
    alpha: float = 0.05,
    weights: np.ndarray | None = None,
) -> tuple[float, float] | None:
    """Efron percentile CI for a (weighted) mean of session-level metrics.

    Non-finite values are dropped. ``None`` if fewer than two finite units.
    Resamples *units* (conversations / episodes), not frames.
    """
    x = np.asarray(values, dtype=np.float64).reshape(-1)
    if weights is None:
        w = np.ones_like(x)
    else:
        w = np.asarray(weights, dtype=np.float64).reshape(-1)
        if w.shape != x.shape:
            raise ValueError("weights must match values")
    ok = np.isfinite(x) & np.isfinite(w) & (w > 0)
    x = x[ok]
    w = w[ok]
    if x.size < 2:
        return None
    if n_boot < 1:
        return None
    rng = np.random.default_rng(int(seed))
    idx = rng.integers(0, x.size, size=(int(n_boot), x.size))
    xb = x[idx]
    wb = w[idx]
    means = np.sum(xb * wb, axis=1) / np.sum(wb, axis=1)
    lo = float(np.quantile(means, alpha / 2.0))
    hi = float(np.quantile(means, 1.0 - alpha / 2.0))
    return lo, hi
