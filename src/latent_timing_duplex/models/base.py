"""Protocol for frozen duplex models used by NLL extraction."""

from __future__ import annotations

from typing import Protocol

from latent_timing_duplex.types import DualChannelSession


class FrozenDuplexModel(Protocol):
    """Minimal surface the Phase 0 extractor will call.

    Implementations must not download weights. ``load`` only accepts a local
    directory the caller already populated.
    """

    model_id: str

    def load(self, local_dir: str | None = None) -> None:
        """Attach already-downloaded weights. Must not fetch."""

    def user_channel_nll(self, session: DualChannelSession) -> list[float]:
        """Per-chunk negative log-likelihood on the user audio channel."""
