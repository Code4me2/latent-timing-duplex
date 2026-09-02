"""VAP pooling and LEFT=user p(shift) without loading ErikEkstedt weights."""

from __future__ import annotations

import numpy as np
import pytest

from latent_timing_duplex.baselines.vap import (
    VAP_SPARK_P_NOW,
    VAP_SPARK_P_SHIFT,
    VAPBaseline,
    duration_weighted_mean,
    p_shift_from_p_now,
    pool_to_chunk_grid,
    stereo_from_session,
    vap_probs_to_chunks,
)
from latent_timing_duplex.exceptions import WeightsNotBundled
from latent_timing_duplex.types import DualChannelSession


class _FakeVAP:
    def score_session(self, session: DualChannelSession) -> dict[str, np.ndarray]:
        n = int(session.duration_s * 50)
        p_now = np.full(n, 0.372166)
        return {"p_now": p_now, "p_future": np.full(n, 0.412835)}


def test_p_shift_is_one_minus_p_now_for_speaker_zero() -> None:
    p_now = np.array([0.372166, 0.5])
    shift = p_shift_from_p_now(p_now, speaker=0)
    np.testing.assert_allclose(shift, 1.0 - p_now)
    np.testing.assert_allclose(p_shift_from_p_now(p_now, speaker=1), p_now)


def test_spark_p_shift_identity() -> None:
    assert VAP_SPARK_P_SHIFT == pytest.approx(1.0 - VAP_SPARK_P_NOW)


def test_pool_four_vap_frames_per_chunk() -> None:
    # 50 Hz → 80 ms: 4 frames / chunk
    values = np.arange(8, dtype=np.float64)
    pooled = pool_to_chunk_grid(values)
    assert pooled.tolist() == [1.5, 5.5]


def test_vap_probs_to_chunks_shift() -> None:
    p_now = np.full(8, 0.25)
    chunks = vap_probs_to_chunks(p_now, chunk_duration_s=0.08)
    assert len(chunks) == 2
    assert chunks[0].value == pytest.approx(0.75)
    assert chunks[0].name == "vap:p_shift"


def test_duration_weighted_mean() -> None:
    assert duration_weighted_mean([0.2, 0.8], [10.0, 90.0]) == pytest.approx(0.74)


def test_vap_load_requires_local_checkpoint() -> None:
    with pytest.raises(WeightsNotBundled, match="ErikEkstedt/VAP"):
        VAPBaseline().load()


def test_vap_missing_file() -> None:
    with pytest.raises(FileNotFoundError, match="not a file"):
        VAPBaseline().load(local_checkpoint="/tmp/definitely-not-a-vap.pt")


def test_vap_score_with_injected_model() -> None:
    vap = VAPBaseline()
    vap.load(model=_FakeVAP())
    session = DualChannelSession(session_id="s", duration_s=0.32)
    chunks = vap.score_session(session)
    assert len(chunks) == 4  # 0.32 s / 0.08 s
    assert chunks[0].value == pytest.approx(1.0 - 0.372166)
    agg = vap.clip_aggregates(session)
    assert agg["p_now"] == pytest.approx(0.372166)
    assert agg["p_shift"] == pytest.approx(1.0 - 0.372166)


def test_score_session_without_load() -> None:
    session = DualChannelSession(session_id="s", duration_s=1.0)
    with pytest.raises(WeightsNotBundled, match="not loaded"):
        VAPBaseline().score_session(session)


def test_stereo_left_is_user() -> None:
    session = DualChannelSession(
        session_id="st",
        duration_s=1.0,
        sample_rate=16000,
        user_audio=np.ones(8),
        assistant_audio=np.full(8, 2.0),
    )
    stereo = stereo_from_session(session)
    assert stereo.shape == (2, 8)
    np.testing.assert_allclose(stereo[0], 1.0)
    np.testing.assert_allclose(stereo[1], 2.0)
