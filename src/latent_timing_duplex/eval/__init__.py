"""Eval harness: per-chunk signal in, turn-event scores out."""

from latent_timing_duplex.eval.harness import score_signal, score_session_bundle
from latent_timing_duplex.eval.metrics import average_precision, auroc

__all__ = ["average_precision", "auroc", "score_session_bundle", "score_signal"]
