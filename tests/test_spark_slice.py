"""Spark 10-clip reference numbers are documented constants, not re-measured."""

from __future__ import annotations

from latent_timing_duplex.config import load_config
from latent_timing_duplex.spark_slice import (
    BAYLING_DURATION_WEIGHTED_NLL,
    BAYLING_DURATION_WEIGHTED_PPL,
    BAYLING_N_TOKENS,
    BAYLING_UNWEIGHTED_NLL,
    BAYLING_UNWEIGHTED_PPL,
    BAYLING_VOCAB_SIZE,
    MOSHI_DURATION_WEIGHTED_AUDIO_NLL,
    MOSHI_DURATION_WEIGHTED_TEXT_NLL,
    MOSHI_UNWEIGHTED_AUDIO_NLL,
    MOSHI_UNWEIGHTED_TEXT_NLL,
    SLICE_DURATION_S,
    VAP_DURATION_WEIGHTED_P_FUTURE,
    VAP_DURATION_WEIGHTED_P_NOW,
    VAP_DURATION_WEIGHTED_P_SHIFT,
    reference_table,
)


def test_spark_constants_are_the_measured_digits() -> None:
    assert SLICE_DURATION_S == 1383.44
    assert MOSHI_UNWEIGHTED_AUDIO_NLL == 0.407020
    assert MOSHI_UNWEIGHTED_TEXT_NLL == 0.023778
    assert MOSHI_DURATION_WEIGHTED_AUDIO_NLL == 0.488034
    assert MOSHI_DURATION_WEIGHTED_TEXT_NLL == 0.025773
    assert BAYLING_UNWEIGHTED_NLL == 9.759206
    assert BAYLING_UNWEIGHTED_PPL == 17313
    assert BAYLING_DURATION_WEIGHTED_NLL == 7.167710
    assert BAYLING_DURATION_WEIGHTED_PPL == 1297
    assert BAYLING_N_TOKENS == 17280
    assert BAYLING_VOCAB_SIZE == 168960
    assert VAP_DURATION_WEIGHTED_P_NOW == 0.372166
    assert VAP_DURATION_WEIGHTED_P_FUTURE == 0.412835
    assert VAP_DURATION_WEIGHTED_P_SHIFT == 0.627834


def test_yaml_matches_module() -> None:
    cfg = load_config("spark_slice.yaml")
    assert cfg["slice"]["do_not_rerun_spark"] is True
    assert cfg["slice"]["duration_s"] == SLICE_DURATION_S
    assert cfg["moshi"]["unweighted"]["audio_nll"] == MOSHI_UNWEIGHTED_AUDIO_NLL
    assert cfg["bayling_duplex"]["vocab_size"] == BAYLING_VOCAB_SIZE
    assert cfg["bayling_duplex"]["comparable_to_moshi_codebook_nll"] is False
    assert cfg["vap"]["device"] == "cpu"
    assert cfg["vap"]["duration_weighted"]["p_shift"] == VAP_DURATION_WEIGHTED_P_SHIFT
    assert cfg["hypothesis"]["binding"] is False


def test_reference_table_refuses_comparability() -> None:
    table = reference_table()
    assert table["do_not_rerun_spark"] is True
    assert table["bayling_duplex"]["comparable_to_moshi_codebook_nll"] is False
    assert "delay" in str(table["hypothesis"]).lower()
