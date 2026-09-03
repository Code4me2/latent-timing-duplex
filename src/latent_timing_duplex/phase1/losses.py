"""Prediction MSE plus a JEPA-style isotropic-Gaussian regularizer.

The head predicts the next user-chunk embedding. Targets are stop-grad
(the caller simply does not differentiate them). Collapse is discouraged
by matching the *predicted* batch to ``N(0, I)``:

* ``R_mean`` — batch mean near 0
* ``R_var``  — per-dimension variance near 1
* ``R_cov``  — off-diagonal covariance near 0
* ``R_energy`` — ``0.5 * mean(||z||^2)``, the ``-log N(z; 0, I)`` quadratic

``phase1_loss = MSE + λ (R_mean + R_var + R_cov)``. ``R_energy`` is reported
and is the term differentiated by the numpy train stub (same isotropic
direction, cheaper / stabler grads than the full covariance).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class GaussianRegularizer:
    """Components of the isotropic-Gaussian penalty on ``[B, D]`` embeddings."""

    total: float
    mean: float
    var: float
    cov: float
    energy: float


@dataclass(frozen=True)
class Phase1Loss:
    """Scalar training objective plus the pieces tests assert on."""

    total: float
    mse: float
    regularizer: GaussianRegularizer
    lambda_reg: float


def prediction_mse(pred: np.ndarray, target: np.ndarray) -> float:
    """Mean squared error. ``pred`` and ``target`` must share shape ``[B, D]``."""
    p = np.asarray(pred, dtype=np.float64)
    t = np.asarray(target, dtype=np.float64)
    if p.shape != t.shape:
        raise ValueError(f"pred shape {p.shape} != target shape {t.shape}")
    if p.ndim != 2:
        raise ValueError(f"expected [B, D], got {p.shape}")
    return float(np.mean((p - t) ** 2))


def isotropic_gaussian_regularizer(
    z: np.ndarray,
    eps: float = 1e-8,
) -> GaussianRegularizer:
    """Moment-match ``z`` (``[B, D]``) toward ``N(0, I)``.

    ``B == 1`` still yields a defined energy and mean term; variance and
    covariance are 0 (a single point has no spread).
    """
    del eps
    x = np.asarray(z, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"expected [B, D], got {x.shape}")
    batch, dim = x.shape
    if batch == 0 or dim == 0:
        raise ValueError(f"z must be non-empty [B, D], got {x.shape}")

    mean_vec = x.mean(axis=0)
    r_mean = float(np.sum(mean_vec**2))
    r_energy = float(0.5 * np.mean(np.sum(x**2, axis=1)))

    if batch < 2:
        r_var = 0.0
        r_cov = 0.0
    else:
        # Population moments so a perfect N(0, I) sample is near 0 at large B.
        centered = x - mean_vec
        var = (centered**2).mean(axis=0)
        r_var = float(np.mean((var - 1.0) ** 2))
        cov = (centered.T @ centered) / batch
        off = cov.copy()
        np.fill_diagonal(off, 0.0)
        r_cov = float(np.sum(off**2) / dim)

    total = r_mean + r_var + r_cov
    return GaussianRegularizer(
        total=total,
        mean=r_mean,
        var=r_var,
        cov=r_cov,
        energy=r_energy,
    )


def phase1_loss(
    pred: np.ndarray,
    target: np.ndarray,
    lambda_reg: float = 1.0,
) -> Phase1Loss:
    """``MSE(pred, target) + λ R_iso(pred)``. ``lambda_reg`` must be >= 0."""
    if lambda_reg < 0:
        raise ValueError(f"lambda_reg must be >= 0, got {lambda_reg}")
    mse = prediction_mse(pred, target)
    reg = isotropic_gaussian_regularizer(pred)
    total = mse + float(lambda_reg) * reg.total
    return Phase1Loss(total=total, mse=mse, regularizer=reg, lambda_reg=float(lambda_reg))


def mse_grad(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    """``d MSE / d pred`` with mean reduction, shape ``[B, D]``."""
    p = np.asarray(pred, dtype=np.float64)
    t = np.asarray(target, dtype=np.float64)
    return 2.0 * (p - t) / p.size


def energy_grad(pred: np.ndarray) -> np.ndarray:
    """``d (0.5 mean_i ||z_i||^2) / d z``, shape ``[B, D]``."""
    p = np.asarray(pred, dtype=np.float64)
    return p / p.shape[0]
