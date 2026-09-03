"""Frozen hidden-state extractor. Moshi first; backbone never trained.

Real extract needs local Moshi weights (``WeightsNotBundled`` otherwise).
Tests use ``FakeHiddenStateExtractor``, which returns deterministic
``[T, D]`` arrays and does not load checkpoints.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np

from latent_timing_duplex.exceptions import Phase1NotImplemented, Phase2OutOfScope, WeightsNotBundled
from latent_timing_duplex.phase1.heads import DEFAULT_HIDDEN_DIM
from latent_timing_duplex.phase1.horizons import CHUNK_DURATION_S
from latent_timing_duplex.types import DualChannelSession


class FrozenHiddenStateExtractor(Protocol):
    """Per-chunk hidden states from a frozen duplex backbone."""

    model_id: str
    hidden_dim: int

    def extract(self, session: DualChannelSession) -> np.ndarray:
        """Return ``[T, hidden_dim]``. Must not train the backbone."""


class FakeHiddenStateExtractor:
    """Deterministic hidden states for CPU tests. No weights."""

    model_id = "fake-moshi-hidden"

    def __init__(self, hidden_dim: int = 16, seed: int = 0) -> None:
        self.hidden_dim = hidden_dim
        self.seed = seed

    def extract(self, session: DualChannelSession) -> np.ndarray:
        n = int(session.duration_s / CHUNK_DURATION_S)
        if n < 1:
            raise ValueError("session is shorter than one chunk")
        rng = np.random.default_rng(self.seed + _stable_id(session.session_id))
        # Mild structure so a linear head can fit in the train-stub test.
        t = np.linspace(0.0, 1.0, n, dtype=np.float64)[:, None]
        base = rng.normal(size=(1, self.hidden_dim))
        return np.asarray(base + 0.15 * t, dtype=np.float64)


class MoshiHiddenStateExtractor:
    """Interface over a loaded ``MoshiWrapper``. Frozen forward only.

    ``extract`` is reserved until local weights are attached. Prefer
    ``load_precomputed`` for Spark caches under
    ``/home/velvet/cs199-phase1-work/hidden/moshi/``.
    """

    model_id = "kyutai/moshiko-pytorch-bf16"

    def __init__(self, hidden_dim: int = DEFAULT_HIDDEN_DIM) -> None:
        self.hidden_dim = hidden_dim
        self._wrapper = None

    def attach_wrapper(self, wrapper: object) -> None:
        """Use an already-loaded ``MoshiWrapper``. Does not download."""
        if getattr(wrapper, "_lm", None) is None:
            raise WeightsNotBundled(
                "MoshiHiddenStateExtractor.attach_wrapper needs a MoshiWrapper "
                "that already called load(local_dir=...). This extractor will "
                "not fetch weights."
            )
        self._wrapper = wrapper
        self.model_id = getattr(wrapper, "model_id", self.model_id)

    def extract(self, session: DualChannelSession) -> np.ndarray:
        if self._wrapper is None:
            raise WeightsNotBundled(
                "Moshi hidden states need a locally loaded MoshiWrapper "
                f"before extract({session.session_id!r}). Pass local_dir to "
                "MoshiWrapper.load, then attach_wrapper. Or load a Spark "
                "cache with load_precomputed()."
            )
        raise Phase1NotImplemented(
            "Live Moshi hidden-state extract is reserved for spark-61dd "
            "(frozen LMModel.forward, NO_CUDA_GRAPH=1, NO_TORCH_COMPILE=1). "
            "Use FakeHiddenStateExtractor in tests or load_precomputed()."
        )

    def load_precomputed(self, path: str | Path) -> np.ndarray:
        """Load a cached ``[T, D]`` array from a local ``.npz`` / ``.npy``."""
        root = Path(path)
        if not root.is_file():
            raise FileNotFoundError(
                f"precomputed hidden-state file {path!r} is missing. "
                "On Spark write it under /home/velvet/cs199-phase1-work/hidden/moshi/."
            )
        if root.suffix == ".npz":
            blob = np.load(root)
            if "hidden" not in blob:
                raise KeyError(f"{path} has no 'hidden' array")
            hidden = np.asarray(blob["hidden"], dtype=np.float64)
        else:
            hidden = np.asarray(np.load(root), dtype=np.float64)
        if hidden.ndim != 2:
            raise ValueError(f"hidden must be [T, D], got {hidden.shape}")
        if hidden.shape[1] != self.hidden_dim:
            raise ValueError(
                f"hidden dim {hidden.shape[1]} != extractor hidden_dim {self.hidden_dim}"
            )
        return hidden


def refuse_unfreeze() -> None:
    """Phase 1 must not open a backbone optimizer."""
    raise Phase2OutOfScope(
        "Unfreezing Moshi or BayLing is Phase 2 and is out of scope. "
        "Train the predictor head only."
    )


def _stable_id(session_id: str) -> int:
    return sum(ord(c) for c in session_id) % 10_000
