"""Lock the Phase 1 silence/collapse note and the docs that point at it."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTE = ROOT / "docs" / "PHASE1_SILENCE_COLLAPSE.md"


def test_silence_collapse_note_exists_and_does_not_overclaim() -> None:
    text = NOTE.read_text(encoding="utf-8")
    assert "2026-09-03/04" in text
    assert "spark-61dd" in text
    assert "57" in text
    assert "mid-180" in text
    assert "channel-energy" in text.lower()
    assert "12.5 Hz" in text
    assert "NOT MET" in text
    assert "0/57" in text
    assert "0.337" in text and "0.345" in text and "0.975" in text
    assert "0.376" in text
    assert "0.295" in text
    assert "0.180" in text and "0.187" in text and "0.964" in text
    assert "0.211" in text
    assert "0.145" in text
    assert "H=12" in text or "H=12 frames" in text
    assert "0.01" in text
    assert "mutual_silence" in text
    assert "near_turn_edge" in text
    assert "≥2" in text or ">=2" in text
    assert (
        "/home/velvet/cs199-phase1-work/eval/windows/mid180/"
        "SILENCE_COLLAPSE.md"
    ) in text
    assert (
        "/home/velvet/cs199-phase1-work/eval/windows/mid180/"
        "SILENCE_COLLAPSE.json"
    ) in text
    assert "proxy" in text.lower()
    assert "Not Phase 2" in text or "not Phase 2" in text
    assert "not a timing win" in text.lower()
    assert "flatline" in text.lower()
    assert "PHASE1_RQ2_INTERIM.md" in text
    # Must not upgrade the null into a timing or Phase 2 claim.
    lowered = text.lower()
    assert "timing control works" not in lowered
    assert "phase 2 start" not in lowered


def test_readme_and_plan_link_silence_collapse() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    plan = (ROOT / "docs" / "PHASE1_PLAN.md").read_text(encoding="utf-8")
    protocol = (ROOT / "docs" / "EVAL_PROTOCOL_PHASE1.md").read_text(encoding="utf-8")
    rq2 = (ROOT / "docs" / "PHASE1_RQ2_INTERIM.md").read_text(encoding="utf-8")
    spark = (ROOT / "docs" / "SPARK.md").read_text(encoding="utf-8")
    assert "PHASE1_SILENCE_COLLAPSE.md" in readme
    assert "PHASE1_SILENCE_COLLAPSE.md" in plan
    assert "PHASE1_SILENCE_COLLAPSE.md" in protocol
    assert "PHASE1_SILENCE_COLLAPSE.md" in rq2
    assert "PHASE1_SILENCE_COLLAPSE.md" in spark
    assert "0/57" in readme
    assert "0/57" in plan
    assert "not a timing win" in readme.lower() or "timing-win" in readme.lower()
    assert "not a timing win" in plan.lower()
    assert "not a timing win" in rq2.lower()
