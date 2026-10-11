"""Shared sample-grid ordering for observed post-cutoff events."""

from __future__ import annotations

import math


def nearest_frame(seconds: float, sample_rate: int) -> int:
    if isinstance(seconds, bool) or not isinstance(seconds, (int, float)):
        raise ValueError("event time must be numeric")
    if not math.isfinite(seconds) or seconds < 0:
        raise ValueError("event time must be finite and non-negative")
    if isinstance(sample_rate, bool) or not isinstance(sample_rate, int) or sample_rate <= 0:
        raise ValueError("sample_rate must be a positive integer")
    return int(math.floor(seconds * sample_rate + 0.5))


def first_observed_event(
    user_resume_s: float | None,
    partner_entry_s: float | None,
    sample_rate: int,
) -> tuple[str | None, float | None]:
    """Order events on the authenticated source sample grid; user wins a frame tie."""
    if isinstance(sample_rate, bool) or not isinstance(sample_rate, int) or sample_rate <= 0:
        raise ValueError("sample_rate must be a positive integer")
    user_frame = None if user_resume_s is None else nearest_frame(user_resume_s, sample_rate)
    partner_frame = None if partner_entry_s is None else nearest_frame(partner_entry_s, sample_rate)
    if user_frame is not None and (
        partner_frame is None or user_frame <= partner_frame
    ):
        return "user_resume", user_resume_s
    if partner_entry_s is not None:
        return "recorded_partner_entry", partner_entry_s
    return None, None
