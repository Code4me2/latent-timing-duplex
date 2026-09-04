"""Every public module must import in a bare Phase 0 install."""

from __future__ import annotations


def test_package_metadata() -> None:
    import latent_timing_duplex as ltd

    assert ltd.__version__ == "0.1.0"
    assert ltd.__phase__ == 1


def test_submodule_imports() -> None:
    from latent_timing_duplex import baselines, data, eval, extract, models, phase1
    from latent_timing_duplex.baselines.vap import VAPBaseline
    from latent_timing_duplex.data.candor import CandorPipeline
    from latent_timing_duplex.data.duplexchat import DuplexChatPipeline
    from latent_timing_duplex.data.synthetic import generate_synthetic_session
    from latent_timing_duplex.eval.harness import score_signal
    from latent_timing_duplex.extract.nll import FrozenNLLExtractor, delay_nan_mask
    from latent_timing_duplex.spark_slice import SLICE_DURATION_S
    from latent_timing_duplex.models.bayling_duplex import BayLingDuplexWrapper
    from latent_timing_duplex.models.moshi import MoshiWrapper

    assert callable(score_signal)
    assert callable(generate_synthetic_session)
    assert MoshiWrapper is not None
    assert BayLingDuplexWrapper is not None
    assert VAPBaseline is not None
    assert CandorPipeline is not None
    assert DuplexChatPipeline is not None
    assert FrozenNLLExtractor is not None
    assert callable(delay_nan_mask)
    assert SLICE_DURATION_S == 1383.44
    assert models.__doc__
    assert data.__doc__
    assert baselines.__doc__
    assert eval.__doc__
    assert extract.__doc__
    assert phase1.__doc__
    assert phase1.PHASE1_HORIZONS_S == (0.08, 1.00, 5.00)
    assert phase1.SPARK_TRAINED_HORIZON_FRAMES == (1, 12, 62)
    assert callable(phase1.run_turn_event_eval)
    assert callable(phase1.score_aligned_signals)
