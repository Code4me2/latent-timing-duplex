"""Turn-event compare: surprise vs NLL vs VAP on synthetic tensors."""

from __future__ import annotations

import json

import numpy as np

from latent_timing_duplex.phase1.checkpoint import save_mlp_checkpoint
from latent_timing_duplex.phase1.compare import (
    EvalConfig,
    EvalPaths,
    compare_session,
    format_compare_report,
    run_synthetic_compare,
    run_turn_event_eval,
    score_aligned_signals,
    write_report,
)
from latent_timing_duplex.phase1.heads import MLPPredictor
from latent_timing_duplex.phase1.horizons import CHUNK_DURATION_S, PROTOCOL_SEED
from latent_timing_duplex.phase1.precompute import write_hidden_npz, write_target_npz
from latent_timing_duplex.phase1.series import chunks_from_values
from latent_timing_duplex.types import DualChannelSession, TurnEvent


def _predictive_session(n_chunks: int = 80, seed: int = 0) -> DualChannelSession:
    duration = n_chunks * CHUNK_DURATION_S
    rng = np.random.default_rng(seed)
    events = [
        TurnEvent(t=float(t), kind="turn_shift", speaker="user")
        for t in rng.uniform(1.0, duration - 2.0, size=10)
    ]
    return DualChannelSession(
        session_id="synth-a",
        duration_s=duration,
        events=sorted(events, key=lambda e: e.t),
        source="test",
    )


def test_score_aligned_signals_shared_timeline() -> None:
    session = _predictive_session(64, seed=1)
    n = 64
    event_t = np.array([e.t for e in session.events])
    scores = np.zeros(n)
    for i in range(n):
        t_end = (i + 1) * CHUNK_DURATION_S
        future = event_t[event_t > t_end]
        if future.size:
            dt = float(future.min() - t_end)
            if dt < 1.0:
                scores[i] = np.exp(-dt / 0.3)
    signals = {
        "jepa:surprise": chunks_from_values(scores[: 64 - 12].tolist(), "jepa:surprise"),
        "nll:moshi": chunks_from_values(scores.tolist(), "nll:moshi"),
        "vap:p_shift": chunks_from_values((0.5 * scores + 0.1).tolist(), "vap:p_shift"),
        "random": chunks_from_values(np.random.default_rng(2).normal(size=n).tolist(), "random"),
    }
    rows = score_aligned_signals(
        session,
        signals,
        horizons_s=(0.50, 1.00),
        event_kinds=("turn_shift",),
        predictor_horizon_frames=12,
        lambda_reg=0.01,
    )
    by = {(r.signal, r.horizon_s): r for r in rows}
    assert by[("jepa:surprise", 0.50)].n_chunks == by[("nll:moshi", 0.50)].n_chunks
    assert by[("jepa:surprise", 0.50)].n_chunks == 52  # 64-12, some tail dropped by horizon
    # Predictive scores should beat random when both are defined.
    pred = by[("jepa:surprise", 1.00)]
    rand = by[("random", 1.00)]
    assert pred.auroc is not None and rand.auroc is not None
    assert pred.auroc > rand.auroc
    assert pred.auprc is not None


def test_compare_session_covers_event_kinds() -> None:
    session = _predictive_session(40, seed=3)
    session.events.append(TurnEvent(t=1.2, kind="backchannel", speaker="assistant"))
    session.events.append(TurnEvent(t=2.4, kind="barge_in", speaker="user"))
    n = 40
    vals = np.linspace(0.0, 1.0, n).tolist()
    result = compare_session(
        session,
        {"jepa:surprise": chunks_from_values(vals, "jepa:surprise")},
        config=EvalConfig(eval_horizons_s=(0.50,), horizon_frames=1, bootstrap_b=8),
    )
    kinds = {r.event_kind for r in result.rows}
    assert kinds == {"turn_shift", "backchannel", "barge_in"}


def test_run_turn_event_eval_from_checkpoint_and_jsonl(tmp_path) -> None:
    rng = np.random.default_rng(4)
    hidden = rng.normal(size=(48, 8))
    proj = rng.normal(size=(8, 4))
    target = hidden @ proj
    head = MLPPredictor(hidden_dim=8, embed_dim=4, width=8, n_layers=1, seed=5)
    ckpt = tmp_path / "h1_lam0.01" / "checkpoint.npz"
    save_mlp_checkpoint(ckpt, head, horizon_frames=1, lambda_reg=0.01)
    hid = tmp_path / "hidden"
    tgt = tmp_path / "target"
    hid.mkdir()
    tgt.mkdir()
    write_hidden_npz(hid / "sess.npz", hidden, "sess")
    write_target_npz(tgt / "sess.npz", target, "sess")

    nll = (np.linspace(0.1, 0.9, 48) ** 2).tolist()
    vap = np.linspace(0.2, 0.8, 48).tolist()
    nll_path = tmp_path / "nll.jsonl"
    vap_path = tmp_path / "vap.jsonl"
    labels_path = tmp_path / "labels.jsonl"
    nll_path.write_text(
        json.dumps({"session_id": "sess", "audio_nll_per_step": nll, "window": "mid180"})
        + "\n",
        encoding="utf-8",
    )
    vap_path.write_text(
        json.dumps({"session_id": "sess", "p_shift": vap, "window": "mid180"}) + "\n",
        encoding="utf-8",
    )
    labels_path.write_text(
        json.dumps(
            {
                "session_id": "sess",
                "events": [
                    {"t": 1.1, "kind": "turn_shift"},
                    {"t": 2.0, "kind": "backchannel"},
                    {"t": 2.8, "kind": "barge_in"},
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    report = run_turn_event_eval(
        EvalPaths(
            checkpoint=ckpt,
            hidden_dir=hid,
            target_dir=tgt,
            nll_jsonl=nll_path,
            vap_jsonl=vap_path,
            labels=labels_path,
        ),
        EvalConfig(
            window_s=48 * CHUNK_DURATION_S,
            window_mode="mid",
            eval_horizons_s=(0.50, 1.00),
            seed=PROTOCOL_SEED,
            bootstrap_b=32,
            horizon_frames=1,
            lambda_reg=0.01,
            lambda_role="primary",
        ),
        session_ids=["sess"],
    )
    assert "jepa:surprise" in report.signals
    assert "nll:moshi" in report.signals
    assert "vap:p_shift" in report.signals
    assert report.predictor_horizon_frames == 1
    assert report.lambda_reg == 0.01
    assert report.seed == PROTOCOL_SEED
    assert any("proxies" in line.lower() or "proxy" in line.lower() for line in report.limitations)
    out = tmp_path / "report.json"
    write_report(report, out)
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["claim"].startswith("Code + metrics only")
    assert payload["protocol"] == "EVAL_PROTOCOL_PHASE1"


def test_synthetic_compare_cli_table() -> None:
    report = run_synthetic_compare(duration_s=20.0, seed=20260903, horizon_frames=1, bootstrap_b=16)
    text = format_compare_report(report)
    assert "jepa:surprise" in text
    assert "nll:moshi" in text
    assert "vap:p_shift" in text
    assert "No empirical claim" in text
    assert report.sessions[0].n_events > 0
