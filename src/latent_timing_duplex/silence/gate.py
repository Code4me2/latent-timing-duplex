"""Small, model-independent floor gate for replay before model integration."""

from __future__ import annotations

from dataclasses import dataclass

from .types import GateState, SilenceAction, SilenceDecision

_ACTION_TO_STATE = {
    SilenceAction.ACK: GateState.ACK,
    SilenceAction.RESPOND: GateState.RESPOND,
    SilenceAction.REPAIR: GateState.REPAIR,
}


@dataclass
class SilenceGate:
    """Require sustained evidence before leaving HOLD; speech resets the gate."""

    threshold: float = 0.6
    consecutive_frames: int = 2
    state: GateState = GateState.LISTEN
    _candidate: SilenceAction | None = None
    _candidate_frames: int = 0

    def __post_init__(self) -> None:
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError("threshold must be in [0, 1]")
        if self.consecutive_frames < 1:
            raise ValueError("consecutive_frames must be positive")

    def step(
        self,
        decision: SilenceDecision,
        *,
        user_active: bool,
        assistant_active: bool,
    ) -> GateState:
        if user_active or assistant_active:
            self.reset()
            return self.state

        if self.state is GateState.LISTEN:
            self.state = GateState.HOLD

        ranked = [
            (probability, action)
            for action, probability in decision.action_probabilities.items()
            if action not in {SilenceAction.WAIT, SilenceAction.UNRATABLE}
        ]
        probability, candidate = max(ranked, default=(0.0, SilenceAction.WAIT))
        if probability < self.threshold:
            self._candidate = None
            self._candidate_frames = 0
            self.state = GateState.HOLD
            return self.state

        if candidate == self._candidate:
            self._candidate_frames += 1
        else:
            self._candidate = candidate
            self._candidate_frames = 1
        if self._candidate_frames >= self.consecutive_frames:
            self.state = _ACTION_TO_STATE[candidate]
        return self.state

    def reset(self) -> None:
        self.state = GateState.LISTEN
        self._candidate = None
        self._candidate_frames = 0
