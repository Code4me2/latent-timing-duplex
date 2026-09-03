"""Gaussian regularizer and Phase 1 loss (no torch, no weights)."""

from __future__ import annotations

import numpy as np
import pytest

from latent_timing_duplex.phase1.losses import (
    isotropic_gaussian_regularizer,
    phase1_loss,
    prediction_mse,
)


def test_mse_zero_and_known() -> None:
    pred = np.zeros((4, 3))
    target = np.zeros((4, 3))
    assert prediction_mse(pred, target) == 0.0
    target[:, 0] = 2.0
    # mean of 4*3 squares: four 4's and eight 0's → 16/12
    assert prediction_mse(pred, target) == pytest.approx(16.0 / 12.0)


def test_mse_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="shape"):
        prediction_mse(np.zeros((2, 3)), np.zeros((2, 4)))
    with pytest.raises(ValueError, match="expected"):
        prediction_mse(np.zeros(3), np.zeros(3))


def test_isotropic_near_standard_normal() -> None:
    rng = np.random.default_rng(0)
    z = rng.standard_normal((4000, 8))
    reg = isotropic_gaussian_regularizer(z)
    assert reg.mean < 0.05
    assert reg.var < 0.05
    assert reg.cov < 0.05
    assert reg.total == pytest.approx(reg.mean + reg.var + reg.cov)
    assert reg.energy == pytest.approx(0.5 * np.mean(np.sum(z**2, axis=1)))


def test_constant_embeddings_are_penalized() -> None:
    z = np.ones((32, 4)) * 3.0
    reg = isotropic_gaussian_regularizer(z)
    # Mean ||μ||^2 = 4 * 9 = 36; variance is 0 so (0-1)^2 = 1; cov is 0.
    assert reg.mean == pytest.approx(36.0)
    assert reg.var == pytest.approx(1.0)
    assert reg.cov == pytest.approx(0.0)
    assert reg.energy == pytest.approx(0.5 * 4 * 9.0)
    assert reg.total > isotropic_gaussian_regularizer(np.random.default_rng(1).standard_normal((32, 4))).total


def test_correlated_dims_raise_cov() -> None:
    rng = np.random.default_rng(2)
    x = rng.standard_normal((2000, 1))
    z = np.concatenate([x, x], axis=1)
    # Standardize so mean/var terms are small; covariance should remain.
    z = (z - z.mean(axis=0)) / z.std(axis=0)
    reg = isotropic_gaussian_regularizer(z)
    iid = isotropic_gaussian_regularizer(rng.standard_normal((2000, 2)))
    assert reg.cov > 0.4
    assert reg.cov > iid.cov


def test_single_row_defined() -> None:
    reg = isotropic_gaussian_regularizer(np.array([[1.0, 0.0, 0.0]]))
    assert reg.var == 0.0
    assert reg.cov == 0.0
    assert reg.mean == pytest.approx(1.0)
    assert reg.energy == pytest.approx(0.5)


def test_phase1_loss_lambda_zero_is_mse() -> None:
    pred = np.ones((5, 2))
    target = np.zeros((5, 2))
    loss = phase1_loss(pred, target, lambda_reg=0.0)
    assert loss.mse == pytest.approx(prediction_mse(pred, target))
    assert loss.total == pytest.approx(loss.mse)
    assert loss.lambda_reg == 0.0


def test_phase1_loss_adds_regularizer() -> None:
    pred = np.ones((5, 2)) * 2.0
    target = np.zeros((5, 2))
    a = phase1_loss(pred, target, lambda_reg=0.0)
    b = phase1_loss(pred, target, lambda_reg=2.0)
    assert b.total == pytest.approx(a.mse + 2.0 * b.regularizer.total)
    assert b.regularizer.total > 0


def test_negative_lambda_rejected() -> None:
    with pytest.raises(ValueError, match="lambda_reg"):
        phase1_loss(np.zeros((2, 2)), np.zeros((2, 2)), lambda_reg=-0.1)
