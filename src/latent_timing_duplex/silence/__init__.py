"""Causal semantic-silence data and gating primitives."""

from .annotations import ALLOWED_TAGS, load_annotations, parse_annotation
from .episodes import DEFAULT_CHECKPOINTS_S, extract_silence_checkpoints, sample_episodes
from .gate import SilenceGate
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
    "extract_silence_checkpoints",
    "load_annotations",
    "parse_annotation",
    "sample_episodes",
]
