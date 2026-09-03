"""Frozen target embeddings of the next user (LEFT) audio chunk.

The predictor is trained to match these. The embedder itself is not trained
in Phase 1 (stop-grad on targets). Tests use ``FakeTargetEmbedder``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np

from latent_timing_duplex.exceptions import Phase1NotImplemented, WeightsNotBundled
from latent_timing_duplex.phase1.dataset import ChunkedStereoDataset, StereoChunk
from latent_timing_duplex.phase1.heads import DEFAULT_EMBED_DIM


class TargetEmbedder(Protocol):
    embed_dim: int

    def embed_chunks(self, chunks: list[StereoChunk]) -> np.ndarray:
        """Return ``[T, embed_dim]`` for LEFT=user chunks."""


class FakeTargetEmbedder:
    """Deterministic user-chunk embeddings for CPU tests."""

    def __init__(self, embed_dim: int = 8, seed: int = 1) -> None:
        self.embed_dim = embed_dim
        self.seed = seed

    def embed_chunks(self, chunks: list[StereoChunk]) -> np.ndarray:
        if not chunks:
            return np.zeros((0, self.embed_dim), dtype=np.float64)
        out = np.zeros((len(chunks), self.embed_dim), dtype=np.float64)
        for i, chunk in enumerate(chunks):
            rng = np.random.default_rng(self.seed + chunk.index)
            # Energy of LEFT=user is a cheap, frozen feature; pad with noise.
            energy = float(np.mean(chunk.left.astype(np.float64) ** 2))
            noise = rng.normal(size=self.embed_dim)
            out[i] = noise
            out[i, 0] += energy
        return out

    def embed_dataset(self, dataset: ChunkedStereoDataset) -> np.ndarray:
        return self.embed_chunks([dataset[i] for i in range(len(dataset))])


class FrozenUserChunkEmbedder:
    """Reserved frozen user-chunk encoder (Mimi latents on Spark).

    Without local weights this raises ``WeightsNotBundled``. Tests should
    use ``FakeTargetEmbedder`` or ``load_precomputed``.
    """

    def __init__(self, embed_dim: int = DEFAULT_EMBED_DIM) -> None:
        self.embed_dim = embed_dim
        self._ready = False

    def mark_loaded(self) -> None:
        """Caller already has a frozen encoder on disk / in memory."""
        self._ready = True

    def embed_chunks(self, chunks: list[StereoChunk]) -> np.ndarray:
        if not self._ready:
            raise WeightsNotBundled(
                "Frozen user-chunk embeddings need a local Mimi / Moshi "
                "encoder. This class does not download. Use "
                "FakeTargetEmbedder in tests or load_precomputed()."
            )
        raise Phase1NotImplemented(
            f"Live target embed of {len(chunks)} chunks is reserved for "
            "spark-61dd (frozen Mimi on LEFT=user). Load a cache instead."
        )

    def load_precomputed(self, path: str | Path) -> np.ndarray:
        root = Path(path)
        if not root.is_file():
            raise FileNotFoundError(
                f"precomputed target file {path!r} is missing. "
                "On Spark write it under /home/velvet/cs199-phase1-work/targets/user_chunk/."
            )
        if root.suffix == ".npz":
            blob = np.load(root)
            if "target" not in blob:
                raise KeyError(f"{path} has no 'target' array")
            z = np.asarray(blob["target"], dtype=np.float64)
        else:
            z = np.asarray(np.load(root), dtype=np.float64)
        if z.ndim != 2 or z.shape[1] != self.embed_dim:
            raise ValueError(f"target must be [T, {self.embed_dim}], got {z.shape}")
        return z
