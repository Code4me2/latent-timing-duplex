"""Lock the Phase 1 RQ2 interim note and the docs that point at it."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTE = ROOT / "docs" / "PHASE1_RQ2_INTERIM.md"


def test_rq2_interim_note_exists_and_does_not_overclaim() -> None:
    text = NOTE.read_text(encoding="utf-8")
    assert "2026-09-03" in text
    assert "spark-61dd" in text
    assert "n=44" in text
    assert "CANDOR-only" in text
    assert "jepa:surprise" in text
    assert "0.632" in text and "0.106" in text
    assert "nll:moshi" in text
    assert "0.448" in text and "0.049" in text
    assert "vap:p_shift" in text
    assert "0.513" in text and "0.056" in text
    assert "/home/velvet/cs199-phase1-work/eval/windows/mid180/" in text
    assert "Proxy labels ≠ gold" in text or "proxy labels ≠ gold" in text.lower()
    assert "Not cross-corpus" in text or "not cross-corpus" in text
    assert "Not Phase 2" in text or "not Phase 2" in text
    assert "different question" in text
    assert "weakly" in text.lower()


def test_readme_and_plan_link_rq2_interim() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    plan = (ROOT / "docs" / "PHASE1_PLAN.md").read_text(encoding="utf-8")
    protocol = (ROOT / "docs" / "EVAL_PROTOCOL_PHASE1.md").read_text(encoding="utf-8")
    assert "PHASE1_RQ2_INTERIM.md" in readme
    assert "PHASE1_RQ2_INTERIM.md" in plan
    assert "PHASE1_RQ2_INTERIM.md" in protocol
