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


def test_silence_protocol_config_lock() -> None:
    cfg = load_config("silence.yaml")
    assert cfg["schema"] == "semantic-silence-config/v1"
    assert cfg["protocol_version"] == "semantic-silence-pilot/v1"
    assert cfg["label_schema"] == "semantic-silence-labels/v1"
    silence = cfg["silence"]
    assert silence["duration_strata_field"] == "checkpoint_elapsed_s"
    assert silence["duration_strata_s"] == [1.0, 3.0]
    assert silence["duration_strata_labels"] == ["<1s", "1–<3s", ">=3s"]
    assert silence["use_realized_duration_for_strata"] is False
    assert silence["max_episodes_per_lineage"] == 5
    assert "max_episodes_per_session" not in silence

    annotation = cfg["annotation"]
    assert annotation["pilot_target_episodes"] == 75
    assert annotation["pilot_review_range_episodes"] == [60, 90]
    assert annotation["pilot_min_lineages"] == 30
    assert annotation["pilot_splits"] == ["train", "development"]
    assert annotation["pilot_excluded_sets"] == ["test", "challenge"]
    assert annotation["alpha_metric"] == "krippendorff_nominal"
    assert annotation["unratable_for_action_alpha"] == "missing"
    assert annotation["collapse_map"] == {
        "ack": "respond",
        "wait": "wait",
        "respond": "respond",
        "repair": "repair",
    }
    assert annotation["collapsed_reliability_requires_new_pilot"] is True
    assert annotation["repeat_fraction"] == 0.05
    assert annotation["automatic_sentinels"] is False
    assert annotation["coordinator_locked_sentinel_list"] is None
    assert annotation["agreement_reports"] == [
        "per_class_positive_agreement",
        "onset_endpoint_disagreement",
        "repeat_consistency",
    ]
    assert annotation["bootstrap"] == {
        "unit": "lineage",
        "seed": 20261009,
        "replicates": 10000,
    }

    media = cfg["pilot_media"]
    assert media == {
        "audio_only": True,
        "prefix_s": 20.0,
        "reveal_s": 8.0,
        "reveal_channel": "user",
        "reveal_clip_at_session_end": True,
        "boundary_rounding": "nearest_sample",
        "gain_scope": "conversation_or_session",
        "gain_db": 0.0,
        "transcript": False,
    }
    assert cfg["acceptable_onset"] == {
        "supervision": "normative_interval_censored_assistant_entry",
        "recorded_partner_onset": "outcome_only",
        "user_resumption_tie": "cancel_assistant_entry",
    }
    assert cfg["gates"] == {
        "r1_requires_pilot_agreement_receipt": True,
        "test_evaluation_requires_analysis_lock_receipt": True,
    }


def test_phase1_config() -> None:
    cfg = load_config("phase1.yaml")
    assert cfg["project"]["phase"] == 1
    assert cfg["phase1"]["horizons_s"] == [0.08, 1.00, 5.00]
    assert cfg["phase1"]["train"]["freeze_backbone"] is True
    assert cfg["phase1"]["eval"]["prefer_fixed_windows"] is True
    assert cfg["phase1"]["eval"]["window_mode"] == "mid"
    assert cfg["phase1"]["eval"]["predictor_horizon_frames"] == [1, 12, 62]
    assert cfg["phase1"]["eval"]["lambda_primary"] == 0.01
    assert cfg["phase1"]["eval"]["lambda_reference"] == 0.0
    assert cfg["phase1"]["eval"]["seed"] == 20260903
    assert cfg["phase1"]["eval"]["no_phase2"] is True
    assert "phase2_finetune" in cfg["phase1"]["out_of_scope"]
    assert cfg["phase1"]["spark_paths"]["phase1_root"] == "/home/velvet/cs199-phase1-work"
