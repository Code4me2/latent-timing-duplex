"""Chunked stereo dataset. LEFT=user, RIGHT=assistant.

Works from in-memory numpy waveforms so tests need no audio files. Spark
sessions are the same ``DualChannelSession`` objects once a local reconstruct
is attached. Horizon pairing uses ``phase1.horizons.pair_indices``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from latent_timing_duplex.phase1.horizons import CHUNK_DURATION_S, pair_indices
from latent_timing_duplex.types import DualChannelSession


@dataclass(frozen=True)
class StereoChunk:
    """One 80 ms (default) stereo frame. ``left`` is always the user."""

    session_id: str
    index: int
    t_start: float
    t_end: float
    left: np.ndarray
    right: np.ndarray
    sample_rate: int

    @property
    def user(self) -> np.ndarray:
        return self.left

    @property
    def assistant(self) -> np.ndarray:
        return self.right


@dataclass(frozen=True)
class HorizonPair:
    """Aligned source (context) and target (future user-chunk) indices."""

    source: StereoChunk
    target: StereoChunk
    horizon_s: float
    source_index: int
    target_index: int


def _as_mono(audio: object, name: str) -> np.ndarray:
    if audio is None:
        raise ValueError(f"{name} audio is missing; Phase 1 chunks need stereo")
    arr = np.asarray(audio)
    if arr.ndim == 2:
        arr = arr[0] if arr.shape[0] <= arr.shape[1] else arr[:, 0]
    if arr.ndim != 1:
        raise ValueError(f"{name} expected mono waveform, got shape {arr.shape}")
    return np.asarray(arr, dtype=np.float32)


class ChunkedStereoDataset:
    """Uniform grid of LEFT=user / RIGHT=assistant chunks from one session."""

    def __init__(
        self,
        session: DualChannelSession,
        chunk_duration_s: float = CHUNK_DURATION_S,
    ) -> None:
        if chunk_duration_s <= 0:
            raise ValueError("chunk_duration_s must be positive")
        if session.sample_rate is None or session.sample_rate <= 0:
            raise ValueError("session.sample_rate is required to chunk waveforms")
        left = _as_mono(session.user_audio, "user/LEFT")
        right = _as_mono(session.assistant_audio, "assistant/RIGHT")
        n = min(left.size, right.size)
        self.session = session
        self.session_id = session.session_id
        self.chunk_duration_s = float(chunk_duration_s)
        self.sample_rate = int(session.sample_rate)
        self._left = left[:n]
        self._right = right[:n]
        samples = int(round(self.chunk_duration_s * self.sample_rate))
        if samples < 1:
            raise ValueError("chunk is shorter than one sample")
        self.samples_per_chunk = samples
        self.n_chunks = n // samples
        if self.n_chunks < 1:
            raise ValueError("waveform is shorter than one chunk")

    def __len__(self) -> int:
        return self.n_chunks

    def __getitem__(self, index: int) -> StereoChunk:
        if index < 0 or index >= self.n_chunks:
            raise IndexError(index)
        lo = index * self.samples_per_chunk
        hi = lo + self.samples_per_chunk
        t0 = index * self.chunk_duration_s
        return StereoChunk(
            session_id=self.session_id,
            index=index,
            t_start=t0,
            t_end=t0 + self.chunk_duration_s,
            left=self._left[lo:hi],
            right=self._right[lo:hi],
            sample_rate=self.sample_rate,
        )

    def stereo_matrix(self) -> np.ndarray:
        """``[2, n_chunks, samples_per_chunk]`` with channel 0 = LEFT=user."""
        n = self.n_chunks * self.samples_per_chunk
        left = self._left[:n].reshape(self.n_chunks, self.samples_per_chunk)
        right = self._right[:n].reshape(self.n_chunks, self.samples_per_chunk)
        return np.stack([left, right], axis=0)

    def pairs_for_horizon(self, horizon_s: float) -> list[HorizonPair]:
        src, tgt = pair_indices(self.n_chunks, horizon_s, self.chunk_duration_s)
        return [
            HorizonPair(
                source=self[int(i)],
                target=self[int(j)],
                horizon_s=float(horizon_s),
                source_index=int(i),
                target_index=int(j),
            )
            for i, j in zip(src.tolist(), tgt.tolist())
        ]


def make_synthetic_stereo_session(
    duration_s: float = 2.0,
    sample_rate: int = 24000,
    seed: int = 0,
    session_id: str = "phase1-synth",
) -> DualChannelSession:
    """In-memory stereo session for tests. LEFT ≠ RIGHT (channel sanity)."""
    rng = np.random.default_rng(seed)
    n = int(round(duration_s * sample_rate))
    left = rng.normal(size=n).astype(np.float32)
    right = (rng.normal(size=n) + 0.25).astype(np.float32)
    return DualChannelSession(
        session_id=session_id,
        duration_s=float(duration_s),
        sample_rate=sample_rate,
        user_audio=left,
        assistant_audio=right,
        source="synthetic-stereo",
        notes="Phase 1 test waveform. LEFT=user.",
    )
