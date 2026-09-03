"""Configs load and stay aligned with the documented public ids."""

from __future__ import annotations

from latent_timing_duplex.config import load_config
from latent_timing_duplex.models.bayling_duplex import (
    BAYLING_DECODER_ID,
    BAYLING_TOKENIZER_ID,
    BAYLING_WEIGHTS_ID,
)
from latent_timing_duplex.models.moshi import MOSHI_WEIGHTS


def test_default_config_ids() -> None:
    cfg = load_config("default.yaml")
    assert cfg["project"]["phase"] == 0
    assert cfg["models"]["moshi"]["weights"]["moshiko"] == MOSHI_WEIGHTS["moshiko"]
    assert cfg["models"]["moshi"]["weights"]["moshika"] == MOSHI_WEIGHTS["moshika"]
    assert cfg["models"]["moshi"]["download_in_phase0"] is False
    bay = cfg["models"]["bayling_duplex"]
    assert bay["weights"] == BAYLING_WEIGHTS_ID
    assert bay["tokenizer"] == BAYLING_TOKENIZER_ID
    assert bay["decoder"] == BAYLING_DECODER_ID
    assert bay["n_safetensor_shards"] == 4
    assert bay["download_in_phase0"] is False
    assert cfg["models"]["vap"]["device"] == "cpu"


def test_eval_config_horizons() -> None:
    cfg = load_config("eval.yaml")
    assert cfg["eval"]["horizons_s"] == [0.16, 0.32, 0.50, 1.00, 2.00]
    assert "turn_shift" in cfg["eval"]["event_kinds"]


def test_phase1_config() -> None:
    cfg = load_config("phase1.yaml")
    assert cfg["project"]["phase"] == 1
    assert cfg["phase1"]["horizons_s"] == [0.08, 1.00, 5.00]
    assert cfg["phase1"]["train"]["freeze_backbone"] is True
    assert cfg["phase1"]["eval"]["prefer_fixed_windows"] is True
    assert cfg["phase1"]["eval"]["window_mode"] == "mid"
    assert "phase2_finetune" in cfg["phase1"]["out_of_scope"]
    assert cfg["phase1"]["spark_paths"]["phase1_root"] == "/home/velvet/cs199-phase1-work"
