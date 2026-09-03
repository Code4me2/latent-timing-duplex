"""Equal-length window crops for Phase 0-comparable eval.

Phase 0 freeze: prefer fixed windows (first-W and mid-W). Full-length Moshi
contrasts are descriptive / length-confounded. See
``docs/EVAL_PROTOCOL_PHASE0.md``. This helper crops *metadata and attached
waveforms* so surprise, NLL, and VAP can share a window.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from latent_timing_duplex.types import DualChannelSession, TurnEvent

WindowMode = Literal["first", "mid"]
DEFAULT_WINDOW_S = 180.0


def window_bounds(
    duration_s: float,
    window_s: float = DEFAULT_WINDOW_S,
    mode: WindowMode = "mid",
) -> tuple[float, float]:
    """Return ``[start, end]`` seconds for an equal-length crop.

    Sessions shorter than ``window_s`` raise ``ValueError`` (Phase 0 exact-W
    pairing drops those episodes; do not silently pad).
    """
    if window_s <= 0:
        raise ValueError("window_s must be positive")
    if duration_s + 1e-9 < window_s:
        raise ValueError(
            f"session duration {duration_s}s is shorter than window {window_s}s; "
            "drop it under the Phase 0 exact-W rule"
        )
    if mode == "first":
        return 0.0, float(window_s)
    if mode == "mid":
        start = 0.5 * (duration_s - window_s)
        return float(start), float(start + window_s)
    raise ValueError(f"unknown window mode {mode!r}; expected 'first' or 'mid'")


def crop_session(
    session: DualChannelSession,
    window_s: float = DEFAULT_WINDOW_S,
    mode: WindowMode = "mid",
) -> DualChannelSession:
    """Return a new session covering ``[start, start+window_s)``.

    Event times are shifted to be relative to the crop. Audio, if present,
    is sliced when ``sample_rate`` is set.
    """
    start, end = window_bounds(session.duration_s, window_s, mode)
    events = [
        TurnEvent(t=e.t - start, kind=e.kind, speaker=e.speaker)
        for e in session.events
        if start <= e.t < end
    ]
    user = _slice_audio(session.user_audio, session.sample_rate, start, end)
    assistant = _slice_audio(session.assistant_audio, session.sample_rate, start, end)
    return DualChannelSession(
        session_id=f"{session.session_id}:{mode}{int(window_s)}",
        duration_s=float(window_s),
        sample_rate=session.sample_rate,
        user_audio=user,
        assistant_audio=assistant,
        events=events,
        source=session.source,
        notes=f"equal-length {mode} window {window_s}s from {session.session_id}",
    )


def _slice_audio(
    audio: object | None,
    sample_rate: int | None,
    start_s: float,
    end_s: float,
) -> object | None:
    if audio is None or sample_rate is None:
        return audio
    arr = np.asarray(audio)
    lo = int(round(start_s * sample_rate))
    hi = int(round(end_s * sample_rate))
    if arr.ndim == 1:
        return arr[lo:hi]
    if arr.ndim == 2:
        # codes [K, T] at 12.5 Hz are *not* waveform samples; leave untouched
        # unless the last axis looks like waveform length at sample_rate.
        return arr
    return audio
