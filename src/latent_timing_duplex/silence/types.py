"""Causal silence-episode, annotation, and controller types."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Mapping


class SilenceAction(str, Enum):
    WAIT = "wait"
    ACK = "ack"
    RESPOND = "respond"
    REPAIR = "repair"
    UNRATABLE = "unratable"


class SilenceContext(str, Enum):
    JOINT = "joint_silence"
    LISTEN = "assistant_speaking"


class SilenceQuality(str, Enum):
    UNREVIEWED = "unreviewed"
    CLEAN = "clean"
    NONLEXICAL = "breath_or_nonlexical"
    NOISY = "noise_or_clipping"
    TECHNICAL = "dropout_mute_or_edit"


class GateState(str, Enum):
    LISTEN = "listen"
    HOLD = "hold"
    ACK = "ack"
    RESPOND = "respond"
    REPAIR = "repair"


@dataclass(frozen=True)
class SilenceCheckpoint:
    """One causal decision point during a user-inactive interval."""

    session_id: str
    episode_id: str
    onset_s: float
    silence_end_s: float
    model_cutoff_s: float
    elapsed_s: float
    context: SilenceContext
    right_censored: bool = False
    quality: SilenceQuality = SilenceQuality.UNREVIEWED
    source: str = "unknown"
    lineage_id: str | None = None

    def __post_init__(self) -> None:
        if not self.session_id or not self.episode_id:
            raise ValueError("session_id and episode_id are required")
        if self.onset_s < 0 or self.silence_end_s <= self.onset_s:
            raise ValueError("silence interval must have positive duration")
        if not self.onset_s <= self.model_cutoff_s <= self.silence_end_s + 1e-9:
            raise ValueError("model_cutoff_s must lie inside the silence interval")
        expected = self.model_cutoff_s - self.onset_s
        if abs(expected - self.elapsed_s) > 1e-6:
            raise ValueError("elapsed_s must equal model_cutoff_s - onset_s")


@dataclass(frozen=True)
class AcceptableOnset:
    """Acceptable assistant-entry window, in seconds after silence onset."""

    earliest_elapsed_s: float | None
    latest_elapsed_s: float | None
    wait_for_user_resume: bool = False

    def __post_init__(self) -> None:
        if self.wait_for_user_resume:
            if self.earliest_elapsed_s is not None or self.latest_elapsed_s is not None:
                raise ValueError("wait_for_user_resume cannot include an onset window")
            return
        if self.earliest_elapsed_s is None or self.latest_elapsed_s is None:
            raise ValueError("an onset window requires earliest and latest elapsed seconds")
        if self.earliest_elapsed_s < 0 or self.latest_elapsed_s < self.earliest_elapsed_s:
            raise ValueError("invalid acceptable-onset window")


@dataclass(frozen=True)
class SilenceAnnotation:
    episode_id: str
    checkpoint_elapsed_s: float
    annotator: str
    action: SilenceAction
    tags: tuple[str, ...]
    acceptable_onset: AcceptableOnset | None
    quality: SilenceQuality
    ratable: bool = True
    confidence: int | None = None
    also_acceptable: tuple[SilenceAction, ...] = ()
    schema: str = "semantic-silence-labels/v1"
    clip_sha256: str | None = None

    def __post_init__(self) -> None:
        if not self.episode_id or not self.annotator:
            raise ValueError("episode_id and annotator are required")
        if self.checkpoint_elapsed_s < 0:
            raise ValueError("checkpoint_elapsed_s must be non-negative")
        if self.ratable == (self.action is SilenceAction.UNRATABLE):
            raise ValueError("unratable rows must use action='unratable' and vice versa")
        if self.ratable != (self.acceptable_onset is not None):
            raise ValueError("ratable rows require onset guidance; unratable rows cannot have it")
        if self.confidence is not None and not 1 <= self.confidence <= 5:
            raise ValueError("confidence must be in [1, 5]")
        if self.action in self.also_acceptable or SilenceAction.UNRATABLE in self.also_acceptable:
            raise ValueError("also_acceptable must contain distinct ratable alternatives")
        if self.clip_sha256 is not None and (
            len(self.clip_sha256) != 64
            or any(char not in "0123456789abcdef" for char in self.clip_sha256.lower())
        ):
            raise ValueError("clip_sha256 must be a 64-character hexadecimal digest")
        earliest = (
            self.acceptable_onset.earliest_elapsed_s
            if self.acceptable_onset is not None
            else None
        )
        if earliest is not None and earliest + 1e-9 < self.checkpoint_elapsed_s:
            raise ValueError("acceptable onset cannot precede the model checkpoint")


@dataclass(frozen=True)
class CausalFeatureRecord:
    """Feature vector with an explicit availability contract."""

    episode_id: str
    model_cutoff_s: float
    available_at_s: float
    values: tuple[float, ...]
    name: str

    def __post_init__(self) -> None:
        if self.available_at_s > self.model_cutoff_s + 1e-9:
            raise ValueError(
                f"feature {self.name!r} is acausal: available at "
                f"{self.available_at_s}s after cutoff {self.model_cutoff_s}s"
            )


@dataclass(frozen=True)
class SilenceDecision:
    action_probabilities: Mapping[SilenceAction, float]
    user_resume_hazard: float
    assistant_entry_hazard: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.user_resume_hazard <= 1.0:
            raise ValueError("user_resume_hazard must be in [0, 1]")
        if not 0.0 <= self.assistant_entry_hazard <= 1.0:
            raise ValueError("assistant_entry_hazard must be in [0, 1]")
        values = [float(v) for v in self.action_probabilities.values()]
        if not values or any(v < 0.0 or v > 1.0 for v in values):
            raise ValueError("action probabilities must be non-empty and in [0, 1]")
        if abs(sum(values) - 1.0) > 1e-6:
            raise ValueError("action probabilities must sum to 1")
