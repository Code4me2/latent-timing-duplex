"""Per-chunk user-channel NLL from a frozen duplex model.

Phase 0 gate: extract NLL from Moshi / BayLing-Duplex (once those wrappers
load local weights) and score the resulting scalar on the eval harness.
Whether that implicit next-token signal is predictive of turn events is a
reportable result even if it is weak.

This module is a stub. It does not run a model.
"""

from __future__ import annotations

from latent_timing_duplex.exceptions import Phase0NotImplemented
from latent_timing_duplex.models.base import FrozenDuplexModel
from latent_timing_duplex.types import ChunkSignal, DualChannelSession, iter_chunks


class FrozenNLLExtractor:
    """Reserved extractor: frozen model + dual-channel session → chunk NLL."""

    def __init__(self, model: FrozenDuplexModel, chunk_duration_s: float = 0.08) -> None:
        self.model = model
        self.chunk_duration_s = chunk_duration_s

    def extract(self, session: DualChannelSession) -> list[ChunkSignal]:
        raise Phase0NotImplemented(
            "Frozen user-channel NLL extraction is reserved. After Moshi / "
            "BayLing-Duplex load() works on local weights, call "
            "model.user_channel_nll(session) and wrap the floats with "
            f"iter_chunks(..., chunk_duration_s={self.chunk_duration_s}). "
            f"Model id on this extractor: {getattr(self.model, 'model_id', '?')}. "
            f"Session: {session.session_id!r}."
        )

    def wrap_values(self, session: DualChannelSession, values: list[float]) -> list[ChunkSignal]:
        """Helper the future implementation can call. Available now for tests."""
        return iter_chunks(
            duration_s=len(values) * self.chunk_duration_s,
            chunk_duration_s=self.chunk_duration_s,
            values=values,
            name=f"nll:{getattr(self.model, 'model_id', 'model')}",
        )
