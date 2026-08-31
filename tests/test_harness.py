"""Harness: per-chunk signal in, turn-event scores out."""

from __future__ import annotations

import numpy as np

from latent_timing_duplex.data.synthetic import generate_synthetic_session
from latent_timing_duplex.eval.harness import score_signal, score_session_bundle
from latent_timing_duplex.eval.metrics import average_precision, auroc


def test_auroc_perfect_and_chance() -> None:
    y = np.array([0, 0, 1, 1])
    assert auroc(y, np.array([0.1, 0.2, 0.8, 0.9])) == 1.0
    assert auroc(y, np.array([0.9, 0.8, 0.2, 0.1])) == 0.0
    assert auroc(np.array([1, 1, 1]), np.array([0.1, 0.2, 0.3])) is None


def test_average_precision_perfect() -> None:
    y = np.array([0, 1, 0, 1])
    assert average_precision(y, np.array([0.1, 0.9, 0.2, 0.8])) == 1.0
    assert average_precision(np.array([0, 0]), np.array([0.1, 0.2])) is None


def test_synthetic_salience_beats_random() -> None:
    bundle = generate_synthetic_session(duration_s=120.0, seed=0)
    named = score_session_bundle(
        bundle.session,
        {"synthetic_salience": bundle.predictive, "random": bundle.random},
        horizons_s=(0.50, 1.00),
        event_kinds=("turn_shift",),
    )
    pred = {row.horizon_s: row for row in named["synthetic_salience"]}
    rand = {row.horizon_s: row for row in named["random"]}
    for h in (0.50, 1.00):
        assert pred[h].n_positives >= 5
        assert pred[h].auroc is not None
        assert rand[h].auroc is not None
        assert pred[h].auroc > 0.7
        assert pred[h].auroc > rand[h].auroc


def test_score_signal_covers_all_event_kinds() -> None:
    bundle = generate_synthetic_session(duration_s=90.0, seed=1)
    rows = score_signal(bundle.session, bundle.predictive)
    kinds = {row.event_kind for row in rows}
    horizons = {row.horizon_s for row in rows}
    assert kinds == {"turn_shift", "backchannel", "barge_in"}
    assert 0.16 in horizons and 2.0 in horizons
