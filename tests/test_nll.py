"""Delay-NaN mask, nan-safe NLL, and FrozenNLLExtractor (no Moshi weights)."""

from __future__ import annotations

import numpy as np
import pytest

from latent_timing_duplex.exceptions import WeightsNotBundled
from latent_timing_duplex.extract.nll import (
    MOSHI_DEFAULT_DELAYS,
    FrozenNLLExtractor,
    delay_nan_mask,
    log_softmax,
    masked_nll_from_logits,
    mean_nll,
    nll_from_lm_output,
    perplexity,
    reduce_clip_nlls,
    undelay_sequence,
)
from latent_timing_duplex.models.moshi import MoshiWrapper
from latent_timing_duplex.types import DualChannelSession


class _FakeDuplex:
    model_id = "fake-duplex"

    def user_channel_nll(self, session: DualChannelSession) -> list[float]:
        n = int(session.duration_s / 0.08)
        return [float(i) for i in range(n)]


def test_delay_nan_mask_matches_moshi_undelay_tail() -> None:
    delays = (0, 0, 1, 1)
    mask = delay_nan_mask(delays, n_timesteps=5, batch=1)
    assert mask.shape == (1, 4, 5)
    assert mask[0, 0].all()
    assert mask[0, 1].all()
    assert not bool(mask[0, 2, -1])
    assert mask[0, 2, :-1].all()


def test_undelay_fills_tail_with_nan() -> None:
    rng = np.random.default_rng(0)
    tensor = rng.normal(size=(1, 3, 6, 4))
    delayed, mask = undelay_sequence(tensor, [0, 1, 2])
    assert mask[0, 0].all()
    assert not mask[0, 1, -1]
    assert not mask[0, 2, -2:].any()
    assert np.isnan(delayed[0, 1, -1]).all()
    assert np.isnan(delayed[0, 2, -2:]).all()
    # First codebook is delay-0: unchanged.
    np.testing.assert_allclose(delayed[0, 0], tensor[0, 0])


def test_naive_mean_is_nan_masked_mean_is_finite() -> None:
    """Hypothesis (non-binding): delay-NaN fill poisons an unmasked mean."""
    logits = np.zeros((1, 2, 4, 3), dtype=np.float64)
    logits[0, 1, -1] = np.nan
    targets = np.zeros((1, 2, 4), dtype=np.int64)
    mask = delay_nan_mask([0, 1], 4)
    nll, valid = masked_nll_from_logits(logits, targets, mask)
    assert np.isnan(nll.mean())
    assert np.isfinite(mean_nll(nll, valid))
    assert not valid[0, 1, -1]


def test_masked_nll_is_cross_entropy() -> None:
    logits = np.array([[[[10.0, 0.0, 0.0], [0.0, 10.0, 0.0]]]])  # [1, 1, 2, 3]
    targets = np.array([[[0, 1]]])
    nll, valid = masked_nll_from_logits(logits, targets, np.ones((1, 1, 2), dtype=bool))
    assert valid.all()
    assert nll[0, 0, 0] < 0.01
    assert nll[0, 0, 1] < 0.01


def test_log_softmax_all_nan_stays_nan() -> None:
    x = np.full((2, 3), np.nan)
    out = log_softmax(x, axis=-1)
    assert np.isnan(out).all()


def test_reduce_clip_nlls_unweighted_vs_duration() -> None:
    # Two clips: short/easy and long/hard — same pattern as Spark (0.407 vs 0.488).
    red = reduce_clip_nlls([0.2, 0.8], [10.0, 90.0], n_tokens=[100, 900])
    assert red.n_clips == 2
    assert red.n_tokens == 1000
    assert red.duration_s == 100.0
    assert red.unweighted == pytest.approx(0.5)
    assert red.duration_weighted == pytest.approx(0.2 * 0.1 + 0.8 * 0.9)


def test_perplexity_matches_bayling_spark_digits() -> None:
    assert int(round(perplexity(9.759206))) == 17313
    assert int(round(perplexity(7.167710))) == 1297


def test_nll_from_lm_output_per_step() -> None:
    card = 4
    logits = np.zeros((1, 2, 3, card))
    logits[..., 1] = 5.0
    targets = np.ones((1, 2, 3), dtype=np.int64)
    mask = np.ones((1, 2, 3), dtype=bool)
    mask[0, 1, 2] = False
    result = nll_from_lm_output(logits, targets, mask)
    assert len(result.audio_per_step) == 3
    assert all(np.isfinite(v) for v in result.audio_per_step)
    assert result.audio.n_tokens == 5  # 2*3 - 1 masked


def test_extractor_wraps_fake_model() -> None:
    session = DualChannelSession(session_id="fake", duration_s=0.24)
    chunks = FrozenNLLExtractor(_FakeDuplex()).extract(session)
    assert [c.value for c in chunks] == [0.0, 1.0, 2.0]
    assert chunks[0].name == "nll:fake-duplex"


def test_extractor_unloaded_moshi_raises() -> None:
    session = DualChannelSession(session_id="x", duration_s=1.0)
    extractor = FrozenNLLExtractor(MoshiWrapper())
    with pytest.raises(WeightsNotBundled, match="not loaded"):
        extractor.extract(session)
    wrapped = extractor.wrap_values(session, [1.0, 2.0, 3.0])
    assert [c.value for c in wrapped] == [1.0, 2.0, 3.0]


def test_moshi_default_delays_match_upstream() -> None:
    assert MOSHI_DEFAULT_DELAYS[0] == 0  # text
    assert MOSHI_DEFAULT_DELAYS[1] == 0  # first audio
    assert MOSHI_DEFAULT_DELAYS[2] == 1
    assert len(MOSHI_DEFAULT_DELAYS) == 17
