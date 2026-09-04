"""CLI entry points used in the README."""

from __future__ import annotations

from latent_timing_duplex.cli import main


def test_status(capsys) -> None:
    assert main(["status"]) == 0
    out = capsys.readouterr().out
    assert "Phase 0" in out or "phase 0" in out
    assert "BayLing-Models/BayLing-Duplex" in out
    assert "JEPA" in out


def test_check(capsys) -> None:
    assert main(["check"]) == 0
    out = capsys.readouterr().out
    assert "imports ok" in out
    assert "configs ok" in out
    assert "spark_H=[1, 12, 62]" in out
    assert "lambda_primary=0.01" in out


def test_reference(capsys) -> None:
    assert main(["reference"]) == 0
    out = capsys.readouterr().out
    assert "0.407020" in out
    assert "9.759206" in out
    assert "0.627834" in out
    assert "Do not re-run Spark" in out
    assert "168960" in out


def test_harness(capsys) -> None:
    assert main(["harness", "--duration", "40", "--seed", "2"]) == 0
    out = capsys.readouterr().out
    assert "synthetic_salience" in out
    assert "turn_shift" in out
    assert "AUROC" in out


def test_phase1(capsys) -> None:
    assert main(["phase1", "--steps", "4", "--seed", "0"]) == 0
    out = capsys.readouterr().out
    assert "jepa:surprise" in out
    assert "freeze_backbone=True" in out
    assert "PHASE0_INTERIM_FINDINGS" in out
    assert "Phase 2" in out


def test_phase1_eval_synthetic(capsys) -> None:
    assert main(["phase1-eval", "--synthetic", "--horizon-frames", "1", "--seed", "20260903"]) == 0
    out = capsys.readouterr().out
    assert "jepa:surprise" in out
    assert "nll:moshi" in out
    assert "vap:p_shift" in out
    assert "AUPRC" in out
    assert "No empirical claim" in out


def test_phase1_eval_alias(capsys) -> None:
    assert main(["phase1", "eval", "--synthetic", "--horizon-frames", "1"]) == 0
    out = capsys.readouterr().out
    assert "jepa:surprise" in out


def test_phase1_eval_missing_paths(capsys) -> None:
    assert main(["phase1-eval", "--horizon-frames", "12"]) == 2
    err = capsys.readouterr().err
    assert "missing Spark input" in err
    assert "--nll-jsonl" in err


def test_phase1_eval_help(capsys) -> None:
    raised = False
    try:
        main(["phase1-eval", "--help"])
    except SystemExit as exc:
        raised = True
        assert exc.code == 0
    assert raised
    out = capsys.readouterr().out
    assert "mlp_state_dict" in out
    assert "candor_" in out
    assert "duration_sec" in out
    assert "surprise-only" in out
    assert "extract*/transcription" in out or "transcription" in out


def test_phase1_export_series_schema(capsys) -> None:
    assert main(["phase1-export-series", "--print-schema"]) == 0
    out = capsys.readouterr().out
    assert "audio_nll_per_step" in out
    assert "p_shift_per_step" in out
    assert "FrozenNLLExtractor" in out
