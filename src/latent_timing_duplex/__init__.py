"""Latent predictive objectives for timing control in full-duplex dialogue.

Phase 0 (frozen): synthetic eval harness, Moshi delay-NaN NLL, frozen VAP.
Phase 1 (scaffolding): JEPA-style predictor heads on frozen Moshi hidden
states. No Phase 2 fine-tuning. No model weights or corpora ship here.
"""

from latent_timing_duplex.exceptions import (
    Phase0NotImplemented,
    Phase1EvalInputMissing,
    Phase1NotImplemented,
    Phase2OutOfScope,
    WeightsNotBundled,
)
from latent_timing_duplex.types import (
    ChunkSignal,
    DualChannelSession,
    HorizonScore,
    TurnEvent,
    TurnEventKind,
)

__version__ = "0.1.0"
__phase__ = 1

__all__ = [
    "ChunkSignal",
    "DualChannelSession",
    "HorizonScore",
    "Phase0NotImplemented",
    "Phase1EvalInputMissing",
    "Phase1NotImplemented",
    "Phase2OutOfScope",
    "TurnEvent",
    "TurnEventKind",
    "WeightsNotBundled",
    "__phase__",
    "__version__",
]
