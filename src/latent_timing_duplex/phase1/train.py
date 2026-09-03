"""Head-only training stub. Backbone stays frozen. No Phase 2 fine-tune.

Numpy SGD on ``MSE + λ * energy``. The moment-matching regularizer is
computed and logged every step. Runs on CPU; Spark GB10 is optional for
cached ``.npz`` minibatches (see ``docs/PHASE1_PLAN.md``).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from latent_timing_duplex.exceptions import Phase1NotImplemented, Phase2OutOfScope
from latent_timing_duplex.phase1.heads import MLPPredictor
from latent_timing_duplex.phase1.horizons import CHUNK_DURATION_S, pair_indices
from latent_timing_duplex.phase1.losses import energy_grad, mse_grad, phase1_loss


@dataclass
class TrainConfig:
    horizon_s: float = 0.08
    lambda_reg: float = 1.0
    lr: float = 1e-2
    batch_size: int = 32
    max_steps: int = 8
    seed: int = 0
    freeze_backbone: bool = True
    device: str = "cpu"
    chunk_duration_s: float = CHUNK_DURATION_S


@dataclass
class TrainStep:
    step: int
    loss: float
    mse: float
    regularizer: float
    energy: float


@dataclass
class TrainResult:
    steps: list[TrainStep] = field(default_factory=list)
    head: MLPPredictor | None = None

    @property
    def final_mse(self) -> float | None:
        return self.steps[-1].mse if self.steps else None


def require_frozen_backbone(freeze_backbone: bool) -> None:
    if not freeze_backbone:
        raise Phase2OutOfScope(
            "Phase 1 trains the predictor head only. Setting "
            "freeze_backbone=False is Phase 2 fine-tuning and is out of scope."
        )


def aligned_pairs(
    hidden: np.ndarray,
    target: np.ndarray,
    horizon_s: float,
    chunk_duration_s: float = CHUNK_DURATION_S,
) -> tuple[np.ndarray, np.ndarray]:
    """Slice ``[T, H]`` / ``[T, E]`` into source/target rows for one horizon."""
    h = np.asarray(hidden, dtype=np.float64)
    z = np.asarray(target, dtype=np.float64)
    if h.ndim != 2 or z.ndim != 2:
        raise ValueError(f"hidden and target must be 2-D, got {h.shape} / {z.shape}")
    if h.shape[0] != z.shape[0]:
        raise ValueError(f"T mismatch: hidden {h.shape[0]} vs target {z.shape[0]}")
    src, tgt = pair_indices(h.shape[0], horizon_s, chunk_duration_s)
    if src.size == 0:
        raise ValueError(
            f"sequence length {h.shape[0]} is too short for horizon {horizon_s}s"
        )
    return h[src], z[tgt]


def train_head_step(
    head: MLPPredictor,
    hidden_batch: np.ndarray,
    target_batch: np.ndarray,
    config: TrainConfig | None = None,
) -> TrainStep:
    """One SGD step on the head. Differentiates MSE + energy regularizer."""
    cfg = config or TrainConfig()
    require_frozen_backbone(cfg.freeze_backbone)
    pred = head.forward(hidden_batch)
    loss = phase1_loss(pred, target_batch, lambda_reg=cfg.lambda_reg)
    grad = mse_grad(pred, target_batch) + cfg.lambda_reg * energy_grad(pred)
    head.apply_grads(head.backward(grad), lr=cfg.lr)
    return TrainStep(
        step=-1,
        loss=loss.total,
        mse=loss.mse,
        regularizer=loss.regularizer.total,
        energy=loss.regularizer.energy,
    )


def train_loop(
    hidden: np.ndarray,
    target: np.ndarray,
    config: TrainConfig | None = None,
    head: MLPPredictor | None = None,
) -> TrainResult:
    """Run ``max_steps`` of numpy SGD on cached pairs. No GPU, no weights."""
    cfg = config or TrainConfig()
    require_frozen_backbone(cfg.freeze_backbone)
    if cfg.device != "cpu":
        raise Phase1NotImplemented(
            f"train_loop device={cfg.device!r} is not in the CPU stub. "
            "Head training on Spark GB10 (torch) is documented in "
            "docs/PHASE1_PLAN.md; this in-repo loop stays numpy/CPU so CI "
            "needs no CUDA."
        )
    src, tgt = aligned_pairs(hidden, target, cfg.horizon_s, cfg.chunk_duration_s)
    if head is None:
        head = MLPPredictor(
            hidden_dim=src.shape[1],
            embed_dim=tgt.shape[1],
            width=min(32, max(8, src.shape[1])),
            n_layers=2,
            seed=cfg.seed,
        )
    rng = np.random.default_rng(cfg.seed)
    n = src.shape[0]
    batch = min(cfg.batch_size, n)
    result = TrainResult(head=head)
    for step in range(cfg.max_steps):
        idx = rng.choice(n, size=batch, replace=n < batch)
        logged = train_head_step(head, src[idx], tgt[idx], cfg)
        result.steps.append(
            TrainStep(
                step=step,
                loss=logged.loss,
                mse=logged.mse,
                regularizer=logged.regularizer,
                energy=logged.energy,
            )
        )
    return result
