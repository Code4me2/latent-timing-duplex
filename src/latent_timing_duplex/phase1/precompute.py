"""Local cache helpers for frozen hidden states and target embeddings.

Does not run Moshi. Writes / reads ``.npz`` so Spark jobs and tests share
one layout. Paths on spark-61dd are documented in ``phase1.paths``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from latent_timing_duplex.phase1.horizons import CHUNK_DURATION_S
from latent_timing_duplex.phase1.paths import (
    HIDDEN_SUBDIR,
    SPARK_PHASE1_ROOT,
    TARGET_SUBDIR,
)


def spark_hidden_path(slice_name: str, session_id: str, root: str = SPARK_PHASE1_ROOT) -> Path:
    return Path(root) / HIDDEN_SUBDIR / slice_name / f"{session_id}.npz"


def spark_target_path(slice_name: str, session_id: str, root: str = SPARK_PHASE1_ROOT) -> Path:
    return Path(root) / TARGET_SUBDIR / slice_name / f"{session_id}.npz"


def write_hidden_npz(
    path: str | Path,
    hidden: np.ndarray,
    session_id: str,
    chunk_duration_s: float = CHUNK_DURATION_S,
) -> Path:
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(hidden, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"hidden must be [T, D], got {arr.shape}")
    np.savez(
        dest,
        hidden=arr,
        session_id=np.asarray(session_id),
        chunk_duration_s=np.asarray(chunk_duration_s),
        t_end=(np.arange(arr.shape[0], dtype=np.float64) + 1.0) * chunk_duration_s,
    )
    return dest


def write_target_npz(
    path: str | Path,
    target: np.ndarray,
    session_id: str,
    chunk_duration_s: float = CHUNK_DURATION_S,
) -> Path:
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(target, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"target must be [T, E], got {arr.shape}")
    np.savez(
        dest,
        target=arr,
        session_id=np.asarray(session_id),
        chunk_duration_s=np.asarray(chunk_duration_s),
    )
    return dest
