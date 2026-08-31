"""Voice Activity Projection (VAP) baseline stub.

Ekstedt & Skantze, "Voice Activity Projection: Self-supervised Learning of
Turn-taking Events", Interspeech 2022, arXiv:2205.09812.

Official stereo implementation: https://github.com/ErikEkstedt/VAP

VAP consumes speaker-separated stereo audio and projects future voice
activity (turn shifts, backchannels, holds). It is the Phase 0 baseline the
eval harness will compare against frozen-model NLL and, later, JEPA surprise.

This stub does not download the example state dict from that repo and does
not run CPC / VAP inference.
"""

from __future__ import annotations

from latent_timing_duplex.exceptions import Phase0NotImplemented, WeightsNotBundled
from latent_timing_duplex.types import ChunkSignal, DualChannelSession

VAP_PAPER = "https://arxiv.org/abs/2205.09812"
VAP_CODE = "https://github.com/ErikEkstedt/VAP"


class VAPBaseline:
    """Reserved wrapper around ErikEkstedt/VAP."""

    def __init__(self) -> None:
        self.model_id = "ErikEkstedt/VAP"

    def load(self, local_checkpoint: str | None = None) -> None:
        if local_checkpoint is None:
            raise WeightsNotBundled(
                "VAP checkpoints are not bundled. The upstream repo "
                f"({VAP_CODE}) ships an example state dict under examples/. "
                "Download or train it yourself, then pass local_checkpoint=. "
                "This skeleton will not fetch or guess a path."
            )
        raise Phase0NotImplemented(
            f"VAP local load from {local_checkpoint!r} is a Phase 0 fill-in. "
            "No inference is implemented in the skeleton."
        )

    def score_session(self, session: DualChannelSession) -> list[ChunkSignal]:
        """Per-chunk VAP-derived salience. Reserved."""
        raise Phase0NotImplemented(
            f"VAP scoring for session {session.session_id!r} is reserved. "
            "Implement after a working local load()."
        )
