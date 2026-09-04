"""NLL / VAP JSONL loaders and same-timeline intersection."""

from __future__ import annotations

import json

import numpy as np
import pytest

from latent_timing_duplex.phase1.horizons import CHUNK_DURATION_S
from latent_timing_duplex.exceptions import Phase1EvalInputMissing
from latent_timing_duplex.phase1.series import (
    crop_values_to_window,
    intersect_by_frame,
    load_nll_jsonl,
    load_vap_jsonl,
    maybe_pool_50hz,
    values_to_aligned,
)
from latent_timing_duplex.types import ChunkSignal


def test_crop_mid180_from_longer_series() -> None:
    # 200 s → 2500 frames; mid-180 starts at 10 s → index 125.
    values = np.arange(2500, dtype=np.float64)
    cropped = crop_values_to_window(values, duration_s=200.0, window_s=180.0, mode="mid")
    assert cropped.shape == (2250,)
    assert cropped[0] == 125
    assert cropped[-1] == 125 + 2249


def test_crop_already_windowed_is_noop() -> None:
    values = np.ones(2250)
    out = crop_values_to_window(values, duration_s=180.0, window_s=180.0, mode="mid")
    assert out.shape == (2250,)
    assert np.array_equal(out, values)


def test_maybe_pool_50hz() -> None:
    # 1.6 s → 80 frames at 50 Hz → 20 chunks at 80 ms.
    values = np.arange(80, dtype=np.float64)
    pooled = maybe_pool_50hz(values, duration_s=1.6)
    assert pooled.shape == (20,)
    assert pooled[0] == pytest.approx(1.5)


def test_load_nll_and_vap_jsonl(tmp_path) -> None:
    nll_path = tmp_path / "nll.jsonl"
    vap_path = tmp_path / "vap.jsonl"
    nll = np.linspace(0.2, 0.8, 40).tolist()
    p_now = np.linspace(0.1, 0.4, 40).tolist()
    nll_path.write_text(
        json.dumps(
            {
                "session_id": "ep1",
                "audio_nll_per_step": nll,
                "duration_s": 3.2,
                "window": "mid180",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    vap_path.write_text(
        json.dumps({"uuid": "ep1", "p_now": p_now, "duration_s": 3.2, "window": "mid"})
        + "\n",
        encoding="utf-8",
    )
    nll_idx = load_nll_jsonl(nll_path, crop=True)
    vap_idx = load_vap_jsonl(vap_path, crop=True)
    nll_s = nll_idx.get("ep1")
    vap_s = vap_idx.get("ep1")
    assert nll_s.values.shape == (40,)
    assert nll_s.values[0] == pytest.approx(0.2)
    assert vap_s.name == "vap:p_shift"
    assert vap_s.values[0] == pytest.approx(0.9)


def test_intersect_by_frame_trims_to_shared_t_end() -> None:
    long = values_to_aligned("s", "nll:moshi", np.arange(10, dtype=np.float64)).to_chunks()
    short_vals = np.arange(8, dtype=np.float64) + 100
    short = [
        ChunkSignal(
            t_start=i * CHUNK_DURATION_S,
            t_end=(i + 1) * CHUNK_DURATION_S,
            value=float(v),
            name="jepa:surprise",
        )
        for i, v in enumerate(short_vals)
    ]
    aligned = intersect_by_frame({"nll:moshi": long, "jepa:surprise": short})
    assert len(aligned["nll:moshi"]) == 8
    assert len(aligned["jepa:surprise"]) == 8
    assert aligned["nll:moshi"][-1].t_end == pytest.approx(0.64)


def test_duration_sec_alias_with_per_step(tmp_path) -> None:
    path = tmp_path / "nll.jsonl"
    values = np.linspace(0.1, 0.5, 20).tolist()
    path.write_text(
        json.dumps(
            {
                "uuid": "u1",
                "audio_nll_per_step": values,
                "duration_sec": 1.6,
                "window": "mid180",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    series = load_nll_jsonl(path).get("u1")
    assert series.values.shape == (20,)


def test_aggregate_only_nll_raises_clear_error(tmp_path) -> None:
    path = tmp_path / "agg.jsonl"
    path.write_text(
        json.dumps(
            {
                "session_id": "u1",
                "audio_nll": 0.55,
                "p_shift_mean": 0.48,
                "duration_sec": 180.0,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(Phase1EvalInputMissing, match="per-step") as exc:
        load_nll_jsonl(path)
    msg = str(exc.value)
    assert "audio_nll_per_step" in msg
    assert "surprise-only" in msg
    assert "AUROC cannot be computed from clip-level means" in msg


def test_aggregate_only_vap_raises(tmp_path) -> None:
    path = tmp_path / "vap_agg.jsonl"
    path.write_text(
        json.dumps({"session_id": "u1", "p_shift_mean": 0.5, "duration_sec": 180.0})
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(Phase1EvalInputMissing, match="p_shift_per_step"):
        load_vap_jsonl(path)
