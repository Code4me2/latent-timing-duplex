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
    SPARK_TRAINED_HORIZON_FRAMES,
    horizon_steps,
    pair_indices,
    pair_indices_frames,
    target_index,
)
from latent_timing_duplex.phase1.losses import (
    Phase1Loss,
    isotropic_gaussian_regularizer,
    phase1_loss,
    prediction_mse,
)
from latent_timing_duplex.phase1.compare import (
    CompareReport,
    EvalConfig,
    EvalPaths,
    compare_session,
    run_synthetic_compare,
    run_turn_event_eval,
    score_aligned_signals,
)
from latent_timing_duplex.phase1.surprise import (
    score_surprise,
    surprise_from_horizon_frames,
    surprise_to_chunks,
    surprise_values,
)
from latent_timing_duplex.phase1.train import TrainConfig, train_head_step, train_loop

__all__ = [
    "CHUNK_DURATION_S",
    "CompareReport",
    "DEFAULT_EMBED_DIM",
    "DEFAULT_HIDDEN_DIM",
    "DEFAULT_WIDTH",
    "EvalConfig",
    "EvalPaths",
    "MLPPredictor",
    "PHASE1_HORIZONS_S",
    "Phase1Loss",
    "SPARK_TRAINED_HORIZON_FRAMES",
    "TinyTransformerPredictor",
    "TrainConfig",
    "compare_session",
    "count_mlp_parameters",
    "count_parameters",
    "horizon_steps",
    "isotropic_gaussian_regularizer",
    "pair_indices",
    "pair_indices_frames",
    "phase1_loss",
    "prediction_mse",
    "run_synthetic_compare",
    "run_turn_event_eval",
    "score_aligned_signals",
    "score_surprise",
    "surprise_from_horizon_frames",
    "surprise_to_chunks",
    "surprise_values",
    "target_index",
    "train_head_step",
    "train_loop",
]
