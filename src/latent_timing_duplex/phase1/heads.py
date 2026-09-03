"""Small predictor heads: MLP (trainable) and a tiny Transformer (shapes).

Default MLP is ``4096 → 512 → 512 → 256`` (~2.49M params), under the
single-digit-million budget. Tests construct tiny widths. Numpy only; no
torch and no backbone weights.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

DEFAULT_HIDDEN_DIM = 4096
DEFAULT_WIDTH = 512
DEFAULT_EMBED_DIM = 256
DEFAULT_N_LAYERS = 2
MAX_PARAMS = 10_000_000


def _xavier(rng: np.random.Generator, n_in: int, n_out: int) -> np.ndarray:
    scale = math_sqrt(6.0 / (n_in + n_out))
    return rng.uniform(-scale, scale, size=(n_in, n_out)).astype(np.float64)


def math_sqrt(x: float) -> float:
    return float(np.sqrt(x))


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)


def count_linear_params(n_in: int, n_out: int) -> int:
    return n_in * n_out + n_out


def count_mlp_parameters(
    hidden_dim: int,
    embed_dim: int,
    width: int = DEFAULT_WIDTH,
    n_layers: int = DEFAULT_N_LAYERS,
) -> int:
    """Parameter count for ``MLPPredictor`` (weights + biases)."""
    if n_layers < 1:
        raise ValueError("n_layers must be >= 1")
    dims = [hidden_dim]
    dims.extend([width] * n_layers)
    dims.append(embed_dim)
    return sum(count_linear_params(a, b) for a, b in zip(dims, dims[1:]))


def count_parameters(module: object) -> int:
    """Sum sizes of ``.parameters()`` if present, else 0."""
    params = getattr(module, "parameters", None)
    if params is None:
        return 0
    values = params() if callable(params) else params
    return int(sum(np.asarray(p).size for p in values))


@dataclass
class _LinearCache:
    x: np.ndarray
    w: np.ndarray
    b: np.ndarray


class Linear:
    """Dense layer with cached activations for a backward pass."""

    def __init__(self, n_in: int, n_out: int, rng: np.random.Generator) -> None:
        self.n_in = n_in
        self.n_out = n_out
        self.w = _xavier(rng, n_in, n_out)
        self.b = np.zeros(n_out, dtype=np.float64)
        self._cache: _LinearCache | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        x64 = np.asarray(x, dtype=np.float64)
        if x64.ndim != 2 or x64.shape[1] != self.n_in:
            raise ValueError(f"expected [B, {self.n_in}], got {x64.shape}")
        self._cache = _LinearCache(x=x64, w=self.w, b=self.b)
        return x64 @ self.w + self.b

    def backward(self, dy: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self._cache is None:
            raise RuntimeError("forward must run before backward")
        g = np.asarray(dy, dtype=np.float64)
        dw = self._cache.x.T @ g
        db = g.sum(axis=0)
        dx = g @ self._cache.w.T
        return dx, dw, db

    def parameters(self) -> list[np.ndarray]:
        return [self.w, self.b]


class MLPPredictor:
    """ReLU MLP: ``hidden_dim → width × n_layers → embed_dim``."""

    def __init__(
        self,
        hidden_dim: int = DEFAULT_HIDDEN_DIM,
        embed_dim: int = DEFAULT_EMBED_DIM,
        width: int = DEFAULT_WIDTH,
        n_layers: int = DEFAULT_N_LAYERS,
        seed: int = 0,
    ) -> None:
        if n_layers < 1:
            raise ValueError("n_layers must be >= 1")
        n_params = count_mlp_parameters(hidden_dim, embed_dim, width, n_layers)
        if n_params > MAX_PARAMS:
            raise ValueError(
                f"MLP would have {n_params} params; Phase 1 budget is < {MAX_PARAMS}"
            )
        self.hidden_dim = hidden_dim
        self.embed_dim = embed_dim
        self.width = width
        self.n_layers = n_layers
        rng = np.random.default_rng(seed)
        dims = [hidden_dim] + [width] * n_layers + [embed_dim]
        self.layers = [Linear(a, b, rng) for a, b in zip(dims, dims[1:])]
        self._relu_masks: list[np.ndarray] = []

    def forward(self, hidden: np.ndarray) -> np.ndarray:
        """Map ``[B, hidden_dim]`` to ``[B, embed_dim]``."""
        x = np.asarray(hidden, dtype=np.float64)
        if x.ndim == 1:
            x = x[None, :]
        self._relu_masks = []
        for i, layer in enumerate(self.layers):
            x = layer.forward(x)
            if i < len(self.layers) - 1:
                mask = x > 0
                self._relu_masks.append(mask)
                x = x * mask
        return x

    def backward(self, d_out: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
        """Return ``[(dw, db), ...]`` per layer (input to output order)."""
        g = np.asarray(d_out, dtype=np.float64)
        grads: list[tuple[np.ndarray, np.ndarray]] = []
        for i in range(len(self.layers) - 1, -1, -1):
            if i < len(self.layers) - 1:
                g = g * self._relu_masks[i]
            dx, dw, db = self.layers[i].backward(g)
            grads.append((dw, db))
            g = dx
        grads.reverse()
        return grads

    def parameters(self) -> list[np.ndarray]:
        out: list[np.ndarray] = []
        for layer in self.layers:
            out.extend(layer.parameters())
        return out

    def apply_grads(
        self,
        grads: list[tuple[np.ndarray, np.ndarray]],
        lr: float,
    ) -> None:
        if lr < 0:
            raise ValueError("lr must be >= 0")
        if len(grads) != len(self.layers):
            raise ValueError(f"expected {len(self.layers)} grad pairs, got {len(grads)}")
        for layer, (dw, db) in zip(self.layers, grads):
            layer.w = layer.w - lr * dw
            layer.b = layer.b - lr * db


class TinyTransformerPredictor:
    """Forward-only tiny Transformer for the architecture ablation.

    A single current hidden state is projected to ``d_model``, passed through
    ``n_layers`` residual blocks (self-attention over a length-1 or short
    context is well-defined), then projected to ``embed_dim``. No backward
    pass in this skeleton; train the MLP first.
    """

    def __init__(
        self,
        hidden_dim: int = DEFAULT_HIDDEN_DIM,
        embed_dim: int = DEFAULT_EMBED_DIM,
        d_model: int = 256,
        n_layers: int = 2,
        n_heads: int = 4,
        seed: int = 0,
    ) -> None:
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.hidden_dim = hidden_dim
        self.embed_dim = embed_dim
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        rng = np.random.default_rng(seed)
        self.in_proj = Linear(hidden_dim, d_model, rng)
        self.out_proj = Linear(d_model, embed_dim, rng)
        self.blocks: list[dict[str, Linear]] = []
        for _ in range(n_layers):
            self.blocks.append(
                {
                    "q": Linear(d_model, d_model, rng),
                    "k": Linear(d_model, d_model, rng),
                    "v": Linear(d_model, d_model, rng),
                    "o": Linear(d_model, d_model, rng),
                    "ff1": Linear(d_model, 2 * d_model, rng),
                    "ff2": Linear(2 * d_model, d_model, rng),
                }
            )

    def forward(self, hidden: np.ndarray) -> np.ndarray:
        """``hidden`` is ``[B, H]`` or ``[B, T, H]`` (mean-pooled over T)."""
        x = np.asarray(hidden, dtype=np.float64)
        if x.ndim == 1:
            x = x[None, :]
        if x.ndim == 3:
            x = x.mean(axis=1)
        if x.ndim != 2:
            raise ValueError(f"expected [B, H] or [B, T, H], got {x.shape}")
        h = self.in_proj.forward(x)[:, None, :]  # [B, 1, D]
        for block in self.blocks:
            h = h + self._attn(block, h)
            ff = relu(block["ff1"].forward(h[:, 0]))
            h = h + block["ff2"].forward(ff)[:, None, :]
        return self.out_proj.forward(h[:, 0])

    def _attn(self, block: dict[str, Linear], h: np.ndarray) -> np.ndarray:
        batch, seq, dim = h.shape
        q = block["q"].forward(h.reshape(batch * seq, dim)).reshape(batch, seq, dim)
        k = block["k"].forward(h.reshape(batch * seq, dim)).reshape(batch, seq, dim)
        v = block["v"].forward(h.reshape(batch * seq, dim)).reshape(batch, seq, dim)
        scale = 1.0 / np.sqrt(dim / self.n_heads)
        scores = np.matmul(q, np.swapaxes(k, 1, 2)) * scale
        scores = scores - scores.max(axis=-1, keepdims=True)
        weights = np.exp(scores)
        weights = weights / weights.sum(axis=-1, keepdims=True)
        ctx = np.matmul(weights, v).reshape(batch * seq, dim)
        return block["o"].forward(ctx).reshape(batch, seq, dim)

    def parameters(self) -> list[np.ndarray]:
        out = self.in_proj.parameters() + self.out_proj.parameters()
        for block in self.blocks:
            for layer in block.values():
                out.extend(layer.parameters())
        return out
