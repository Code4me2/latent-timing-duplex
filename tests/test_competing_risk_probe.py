"""Synthetic check for the observational competing-risk screen."""

from __future__ import annotations

import numpy as np
import pytest

from experiments.silence_graft.train_competing_risk_probe import (
    _normalize_outcome,
    _person_periods,
    run_screen,
)


def test_competing_risk_signal_beats_baseline_and_handles_censoring_and_ties() -> None:
    rows = []
    outcomes = []
    hidden_values = {
        "user_resume": [3.0, 0.0],
        "recorded_partner_entry": [0.0, 3.0],
        None: [-3.0, -3.0],
    }
    item = 0
    for split, n_lineages in (("train", 20), ("development", 10)):
        for lineage_number in range(n_lineages):
            lineage = f"{split}-lineage-{lineage_number}"
            for event in ("user_resume", "recorded_partner_entry", None):
                rows.append((f"item-{item}", lineage, split, event))
                outcomes.append({
                    "item_id": f"item-{item}",
                    "source": "synthetic",
                    "lineage_id": lineage,
                    "split": split,
                    "event": event,
                    "duration_s": 8.0 if event is None else 0.5,
                })
                item += 1
    data = {
        "item_id": np.asarray([row[0] for row in rows]),
        "session_id": np.asarray([f"session-{i}" for i in range(len(rows))]),
        "source": np.asarray(["synthetic"] * len(rows)),
        "lineage_id": np.asarray([row[1] for row in rows]),
        "split": np.asarray([row[2] for row in rows]),
        "action": np.asarray(["wait"] * len(rows)),
        "context": np.asarray(["joint_silence"] * len(rows)),
        "elapsed_s": np.asarray([0.48] * len(rows)),
        "model_cutoff_s": np.asarray([1.0] * len(rows)),
        "available_at_s": np.asarray([1.0] * len(rows)),
        "acoustic": np.zeros((len(rows), 1)),
        "energy_change": np.zeros((len(rows), 1)),
        "history_truncated": np.zeros((len(rows), 1)),
        "sampling_weight_status": np.asarray(["not_identifiable"] * len(rows)),
        "hidden": np.asarray([hidden_values[row[3]] for row in rows]),
    }
    report = run_screen(data, outcomes, width=4, steps=600, n_boot=50, seed=7)
    comparison = report["paired_lineage_bootstrap_nll"]["hidden_minus_duration_acoustic"]
    assert comparison["nll_improvement"] > 0.2
    assert comparison["ci_low"] > 0.0

    tie = _normalize_outcome({
        "schema": "semantic-silence-observed-outcome/v1",
        "observation_horizon_s": 8.0,
        "item_id": "tie",
        "source": "synthetic",
        "lineage_id": "tie-lineage",
        "split": "development",
        "observed_duration_s": 0.5,
        "administrative_censor_s": 9.0,
        "model_cutoff_s": 1.0,
        "user_resume_delay_s": 0.5,
        "recorded_partner_entry_delay_s": 0.5,
        "first_event": "user_resume",
        "user_resume_event": True,
        "recorded_partner_entry_event": False,
        "administrative_censor_event": False,
        "right_censored": False,
    })
    censored = {
        "item_id": "censor", "source": "synthetic", "lineage_id": "censor-lineage",
        "split": "development", "event": None, "duration_s": 1.5,
    }
    _, labels, item_index = _person_periods(np.zeros((2, 1)), [tie, censored])
    assert labels.tolist() == ["user_resume", "no_event"]
    assert item_index.tolist() == [0, 1]  # The censor's incomplete 1–2 s bin is omitted.

    invalid_tie = {**{
        "schema": "semantic-silence-observed-outcome/v1",
        "observation_horizon_s": 8.0,
        "item_id": "bad-tie",
        "source": "synthetic",
        "lineage_id": "tie-lineage",
        "split": "development",
        "observed_duration_s": 0.5,
        "administrative_censor_s": 9.0,
        "model_cutoff_s": 1.0,
        "user_resume_delay_s": 0.5,
        "recorded_partner_entry_delay_s": 0.5,
        "user_resume_event": False,
        "recorded_partner_entry_event": True,
        "administrative_censor_event": False,
        "right_censored": False,
    }, "first_event": "recorded_partner_entry"}
    with pytest.raises(ValueError, match="tie rule"):
        _normalize_outcome(invalid_tie)
