"""Eval harness: per-chunk signal in, turn-event scores out."""

from latent_timing_duplex.eval.harness import (
    frames_and_labels,
    score_signal,
    score_session_bundle,
)
from latent_timing_duplex.eval.metrics import (
    average_precision,
    auroc,
    efron_bootstrap_mean_ci,
    f1max_operating_point,
    precision_at_recall,
)

__all__ = [
    "average_precision",
    "auroc",
    "efron_bootstrap_mean_ci",
    "f1max_operating_point",
    "frames_and_labels",
    "precision_at_recall",
    "score_session_bundle",
    "score_signal",
]
