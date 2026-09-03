"""Phase 1: JEPA-style latent predictor heads on frozen duplex backbones.

Moshi stays frozen. Only the small head is trained. Phase 2 fine-tuning is
out of scope (``Phase2OutOfScope``). Tests and ``ltd phase1`` run on CPU
without weights.
"""

from latent_timing_duplex.phase1.heads import (
    DEFAULT_EMBED_DIM,
    DEFAULT_HIDDEN_DIM,
    DEFAULT_WIDTH,
    MLPPredictor,
    TinyTransformerPredictor,
    count_mlp_parameters,
    count_parameters,
)
from latent_timing_duplex.phase1.horizons import (
    CHUNK_DURATION_S,
    PHASE1_HORIZONS_S,
    horizon_steps,
    pair_indices,
    target_index,
)
from latent_timing_duplex.phase1.losses import (
    Phase1Loss,
    isotropic_gaussian_regularizer,
    phase1_loss,
    prediction_mse,
)
from latent_timing_duplex.phase1.surprise import score_surprise, surprise_to_chunks, surprise_values
from latent_timing_duplex.phase1.train import TrainConfig, train_head_step, train_loop

__all__ = [
    "CHUNK_DURATION_S",
    "DEFAULT_EMBED_DIM",
    "DEFAULT_HIDDEN_DIM",
    "DEFAULT_WIDTH",
    "MLPPredictor",
    "PHASE1_HORIZONS_S",
    "Phase1Loss",
    "TinyTransformerPredictor",
    "TrainConfig",
    "count_mlp_parameters",
    "count_parameters",
    "horizon_steps",
    "isotropic_gaussian_regularizer",
    "pair_indices",
    "phase1_loss",
    "prediction_mse",
    "score_surprise",
    "surprise_to_chunks",
    "surprise_values",
    "target_index",
    "train_head_step",
    "train_loop",
]
