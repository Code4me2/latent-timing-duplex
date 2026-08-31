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


def test_harness(capsys) -> None:
    assert main(["harness", "--duration", "40", "--seed", "2"]) == 0
    out = capsys.readouterr().out
    assert "synthetic_salience" in out
    assert "turn_shift" in out
    assert "AUROC" in out
