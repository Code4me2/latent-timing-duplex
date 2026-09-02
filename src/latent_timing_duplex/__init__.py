"""Latent predictive objectives for timing control in full-duplex dialogue.

Phase 0: synthetic eval harness, Moshi delay-NaN NLL, frozen VAP (CPU).
No model weights, corpora, or Phase 1/2 heads ship in this package.
"""

from latent_timing_duplex.exceptions import Phase0NotImplemented, WeightsNotBundled
from latent_timing_duplex.types import (
    ChunkSignal,
    DualChannelSession,
    HorizonScore,
    TurnEvent,
    TurnEventKind,
)

__version__ = "0.1.0"
__phase__ = 0

__all__ = [
    "ChunkSignal",
    "DualChannelSession",
    "HorizonScore",
    "Phase0NotImplemented",
    "TurnEvent",
    "TurnEventKind",
    "WeightsNotBundled",
    "__phase__",
    "__version__",
]
