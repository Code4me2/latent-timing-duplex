"""Documented Spark paths for Phase 0 artifacts and Phase 1 caches.

These are constants for ``spark-61dd`` (``/home/velvet/cs199-*``). This
module does not check that the directories exist and does not download
anything. CI must not require them.
"""

from __future__ import annotations

from typing import Final

# Phase 0 read-only work trees (do not re-run those jobs).
SPARK_CANDOR_WORK: Final = "/home/velvet/cs199-candor-work"
SPARK_DUPLEXCHAT_WORK: Final = "/home/velvet/cs199-duplexchat-work"
SPARK_MOSHI_WORK: Final = "/home/velvet/cs199-moshi-work"
SPARK_BAYLING_WORK: Final = "/home/velvet/cs199-bayling-work"
SPARK_VAP_WORK: Final = "/home/velvet/cs199-vap-work"

# Phase 1 writable tree (create on Spark; not present in CI).
SPARK_PHASE1_ROOT: Final = "/home/velvet/cs199-phase1-work"

HIDDEN_SUBDIR: Final = "hidden/moshi"
TARGET_SUBDIR: Final = "targets/user_chunk"
HEAD_SUBDIR: Final = "heads"
EVAL_SUBDIR: Final = "eval/windows"

PRIMARY_SLICES: Final = (
    "duplexchat_expanded",
    "part001_excl_pilot",
)
