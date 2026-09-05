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
    # Evening-PT cross-corpus channel-energy VAD addendum (not gold).
    assert "2026-09-03 evening PT" in text
    assert "cross-corpus proxy" in text.lower()
    assert "channel-energy" in text.lower()
    assert "CompositeEventSource" in text
    assert (
        "/home/velvet/cs199-phase1-work/eval/"
        "duplexchat_mid180_channel_energy_vad.jsonl"
    ) in text
    assert "25 ms" in text and "10 ms" in text
    assert "median + 3" in text and "MAD" in text
    assert "n≈56" in text
    assert "0.645" in text and "0.115" in text
    assert "0.463" in text and "0.050" in text
    assert "0.511" in text and "0.054" in text
    assert "0.694" in text and "0.148" in text
    assert "0.522" in text and "0.054" in text
    assert "0.503" in text and "0.046" in text
    assert "0.643" in text and "0.125" in text
    assert "VAD ≠ gold" in text or "vad ≠ gold" in text.lower()
    assert "n_DC=12" in text
    assert "not asr" in text.lower()
    assert text.find("## Primary table") < text.find("## Addendum")
    assert text.find("0.632") < text.find("0.645")
    # Pooled EVENT_HORIZON_GRID CIs (after addendum; not gold, not a freeze).
    assert "EVENT_HORIZON_GRID" in text
    assert (
        "/home/velvet/cs199-phase1-work/eval/windows/mid180/"
        "EVENT_HORIZON_GRID.md"
    ) in text
    assert "0.653" in text and "0.624" in text and "0.681" in text
    assert "0.375" in text and "0.451" in text
    assert "0.473" in text and "0.549" in text
    assert "0.612" in text and "0.676" in text
    assert "0.431" in text and "0.497" in text
    assert "0.476" in text and "0.545" in text
    assert "0.583" in text and "0.554" in text and "0.610" in text
    assert "0.460" in text and "0.515" in text
    assert "0.474" in text and "0.538" in text
    assert "B=10000" in text
    assert "about **0.63**" in text or "about 0.63" in text
    assert "about **0.60**" in text or "about 0.60" in text
    assert "barge_in" in text and "backchannel" in text
    assert "0.5–1.0" in text or "0.5-1.0" in text
    assert "weakens" in text.lower()
    assert "2.0" in text
    assert text.find("## Addendum") < text.find("## EVENT_HORIZON_GRID")
    assert text.find("0.645") < text.find("0.653")


def test_readme_and_plan_link_rq2_interim() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    plan = (ROOT / "docs" / "PHASE1_PLAN.md").read_text(encoding="utf-8")
    protocol = (ROOT / "docs" / "EVAL_PROTOCOL_PHASE1.md").read_text(encoding="utf-8")
    assert "PHASE1_RQ2_INTERIM.md" in readme
    assert "PHASE1_RQ2_INTERIM.md" in plan
    assert "PHASE1_RQ2_INTERIM.md" in protocol
    assert "channel-energy" in readme.lower()
    assert "n≈56" in readme
    assert "channel-energy" in plan.lower()
    assert "VAD ≠ gold" in plan or "vad ≠ gold" in plan.lower()
    assert "EVENT_HORIZON_GRID" in readme
    assert "EVENT_HORIZON_GRID" in plan
