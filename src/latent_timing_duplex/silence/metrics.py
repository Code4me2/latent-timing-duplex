"""Held-out action metrics and paired lineage bootstrap."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from latent_timing_duplex.eval.metrics import average_precision


def multiclass_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    classes: np.ndarray,
) -> dict[str, object]:
    y = np.asarray(labels)
    p = np.asarray(probabilities, dtype=np.float64)
    classes = np.asarray(classes)
    if p.shape != (y.size, classes.size):
        raise ValueError("probabilities must have shape [rows, classes]")
    if y.size == 0 or not np.isfinite(p).all():
        raise ValueError("metrics require non-empty finite probabilities")
    index = {value: i for i, value in enumerate(classes.tolist())}
    try:
        yi = np.asarray([index[value] for value in y.tolist()], dtype=np.int64)
    except KeyError as exc:
        raise ValueError(f"unknown label {exc.args[0]!r}") from exc
    per_class: dict[str, float | None] = {}
    valid = []
    for i, label in enumerate(classes):
        score = average_precision((yi == i).astype(np.int32), p[:, i])
        per_class[str(label)] = score
        if score is not None:
            valid.append(score)
    predicted = classes[np.argmax(p, axis=1)]
    return {
        "macro_auprc": float(np.mean(valid)) if len(valid) == classes.size else None,
        "per_class_auprc": per_class,
        "accuracy": float(np.mean(predicted == y)),
        "nll": float(-np.log(np.clip(p[np.arange(y.size), yi], 1e-12, 1.0)).mean()),
    }


def paired_lineage_bootstrap_delta(
    labels: np.ndarray,
    probabilities_a: np.ndarray,
    probabilities_b: np.ndarray,
    classes: np.ndarray,
    lineage_ids: np.ndarray,
    *,
    n_boot: int = 2000,
    seed: int = 20261009,
    statistic: Callable[[np.ndarray, np.ndarray, np.ndarray], float | None] | None = None,
    min_valid_fraction: float = 0.9,
) -> dict[str, float | int | None]:
    """Bootstrap A-B by conversation/podcast lineage, never by frame."""
    y = np.asarray(labels)
    a = np.asarray(probabilities_a)
    b = np.asarray(probabilities_b)
    groups = np.asarray(lineage_ids)
    if y.size != groups.size or a.shape != b.shape or a.shape[0] != y.size:
        raise ValueError("labels, predictions, and lineage_ids must align")
    if n_boot < 1:
        raise ValueError("n_boot must be positive")
    if not 0.0 < min_valid_fraction <= 1.0:
        raise ValueError("min_valid_fraction must be in (0, 1]")
    unique = np.unique(groups)
    if unique.size < 2:
        raise ValueError("paired bootstrap needs at least two lineages")

    if statistic is None:
        statistic = lambda yy, pp, cc: multiclass_metrics(yy, pp, cc)["macro_auprc"]  # type: ignore[return-value]
    observed_a = statistic(y, a, classes)
    observed_b = statistic(y, b, classes)
    if observed_a is None or observed_b is None:
        return {
            "delta": None, "ci_low": None, "ci_high": None, "n_boot": 0,
            "n_boot_requested": n_boot, "n_boot_valid": 0, "valid_fraction": 0.0,
        }

    indices = {group: np.flatnonzero(groups == group) for group in unique}
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(n_boot):
        sampled = rng.choice(unique, size=unique.size, replace=True)
        take = np.concatenate([indices[group] for group in sampled])
        value_a = statistic(y[take], a[take], classes)
        value_b = statistic(y[take], b[take], classes)
        if value_a is not None and value_b is not None:
            deltas.append(value_a - value_b)
    valid_fraction = len(deltas) / n_boot
    interval_valid = valid_fraction >= min_valid_fraction
    return {
        "delta": float(observed_a - observed_b),
        "ci_low": float(np.quantile(deltas, 0.025)) if interval_valid else None,
        "ci_high": float(np.quantile(deltas, 0.975)) if interval_valid else None,
        "n_boot": len(deltas),
        "n_boot_requested": n_boot,
        "n_boot_valid": len(deltas),
        "valid_fraction": valid_fraction,
    }
