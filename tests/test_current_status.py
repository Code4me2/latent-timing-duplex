"""Lock the consolidated research status and exact Phase 2 receipts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_current_status_names_the_real_target_and_non_claims() -> None:
    status = (ROOT / "docs" / "CURRENT_STATUS.md").read_text(encoding="utf-8")
    phase2 = (ROOT / "docs" / "PHASE2_EXPLORATORY.md").read_text(encoding="utf-8")
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    assert "future hidden-state prediction error" in status
    assert "not yet future user-audio latent prediction" in " ".join(status.split())
    assert "0.669 [0.642, 0.696]" in status
    assert "No Phase 2 efficacy gate has\npassed" in phase2
    assert "CURRENT_STATUS.md" in readme
    assert "PHASE2_EXPLORATORY.md" in readme


def test_phase2_receipt_hashes_and_gate_kinds() -> None:
    root = ROOT / "experiments" / "phase2"
    manifest = json.loads((root / "run-manifest.json").read_text(encoding="utf-8"))

    assert "future Moshi transformer_out" in manifest["target_semantics"]
    assert manifest["gates"]["c1_live"] == "execution_only"
    assert manifest["gates"]["c2"] == "execution_only"
    assert manifest["gates"]["phase2_efficacy"] == "not_passed"

    for record in manifest["scripts"]:
        digest = hashlib.sha256((root / record["path"]).read_bytes()).hexdigest()
        assert digest == record["sha256"]
