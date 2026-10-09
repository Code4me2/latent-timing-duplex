"""Small NumPy softmax probe for frozen silence representations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SoftmaxProbe:
    classes: np.ndarray
    mean: np.ndarray
    scale: np.ndarray
    weights: np.ndarray
    bias: np.ndarray

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        x = _as_matrix(features)
        z = (x - self.mean) / self.scale
        logits = z @ self.weights + self.bias
        logits -= logits.max(axis=1, keepdims=True)
        exp = np.exp(logits)
        return exp / exp.sum(axis=1, keepdims=True)


def fit_softmax_probe(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    classes: np.ndarray | None = None,
    steps: int = 1000,
    learning_rate: float = 0.1,
    l2: float = 1e-4,
) -> SoftmaxProbe:
    """Fit a class-balanced linear softmax probe with full-batch descent."""
    x = _as_matrix(features)
    y = np.asarray(labels)
    if x.shape[0] != y.size or y.size == 0:
        raise ValueError("features and non-empty labels must have matching rows")
    if steps < 1 or learning_rate <= 0 or l2 < 0:
        raise ValueError("invalid optimizer settings")
    classes = np.asarray(np.unique(y) if classes is None else classes)
    class_index = {value: index for index, value in enumerate(classes.tolist())}
    try:
        yi = np.asarray([class_index[value] for value in y.tolist()], dtype=np.int64)
    except KeyError as exc:
        raise ValueError(f"label {exc.args[0]!r} is absent from classes") from exc
    counts = np.bincount(yi, minlength=classes.size)
    if np.any(counts == 0):
        raise ValueError("every requested class needs at least one training row")

    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale < 1e-8] = 1.0
    z = (x - mean) / scale
    weights = np.zeros((z.shape[1], classes.size), dtype=np.float64)
    bias = np.zeros(classes.size, dtype=np.float64)
    sample_weight = y.size / (classes.size * counts[yi])
    target = np.eye(classes.size, dtype=np.float64)[yi]

    for _ in range(steps):
        logits = z @ weights + bias
        logits -= logits.max(axis=1, keepdims=True)
        exp = np.exp(logits)
        probabilities = exp / exp.sum(axis=1, keepdims=True)
        error = (probabilities - target) * sample_weight[:, None] / y.size
        weights -= learning_rate * (z.T @ error + l2 * weights)
        bias -= learning_rate * error.sum(axis=0)

    return SoftmaxProbe(classes, mean, scale, weights, bias)


def fixed_width_projection(
    features: np.ndarray,
    width: int,
    *,
    seed: int,
) -> np.ndarray:
    """Give probe comparisons the same input width without learned projection."""
    x = _as_matrix(features)
    if width < 1:
        raise ValueError("width must be positive")
    rng = np.random.default_rng(seed)
    projection = rng.normal(0.0, 1.0 / np.sqrt(width), size=(x.shape[1], width))
    return x @ projection


def _as_matrix(features: np.ndarray) -> np.ndarray:
    x = np.asarray(features, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    if x.ndim != 2 or not np.isfinite(x).all():
        raise ValueError("features must be a finite 2-D array")
    return x
