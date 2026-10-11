"""Causal semantic-silence data and gating primitives."""

from .annotations import ALLOWED_TAGS, load_annotations, parse_annotation
from .episodes import DEFAULT_CHECKPOINTS_S, extract_silence_checkpoints, sample_episodes
from .gate import SilenceGate
from .metrics import multiclass_metrics, paired_lineage_bootstrap_delta
from .probe import SoftmaxProbe, fit_softmax_probe, fixed_width_projection
from .types import (
    AcceptableOnset,
    CausalFeatureRecord,
    GateState,
    SilenceAction,
    SilenceAnnotation,
    SilenceCheckpoint,
    SilenceContext,
    SilenceDecision,
    SilenceQuality,
)

__all__ = [
    "ALLOWED_TAGS",
    "AcceptableOnset",
    "CausalFeatureRecord",
    "DEFAULT_CHECKPOINTS_S",
    "GateState",
    "SilenceAction",
    "SilenceAnnotation",
    "SilenceCheckpoint",
    "SilenceContext",
    "SilenceDecision",
    "SilenceGate",
    "SilenceQuality",
    "SoftmaxProbe",
    "extract_silence_checkpoints",
    "fit_softmax_probe",
    "fixed_width_projection",
    "load_annotations",
    "multiclass_metrics",
    "paired_lineage_bootstrap_delta",
    "parse_annotation",
    "sample_episodes",
]
