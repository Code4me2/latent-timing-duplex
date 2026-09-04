"""Load Phase 1 predictor checkpoints without baking Spark absolute paths.

Preferred on-disk format is an in-repo numpy ``.npz`` written by
``save_mlp_checkpoint`` (weights are ``[in, out]`` like ``MLPPredictor``).
Spark directories ``h{H}_lam{λ}/`` are resolved from a caller-supplied
``--ablations-root`` (default documented path is only a constant).

``SELECTION_LOCKED.json`` is optional. When present it names the preferred
λ and H grid. When absent the protocol defaults apply: H∈{1,12,62},
λ=0.01 primary, λ=0 reconstruction reference.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from latent_timing_duplex.exceptions import Phase1EvalInputMissing
from latent_timing_duplex.phase1.heads import MLPPredictor
from latent_timing_duplex.phase1.horizons import (
    PRIMARY_LAMBDA,
    PROTOCOL_SEED,
    REFERENCE_LAMBDA,
    SPARK_TRAINED_HORIZON_FRAMES,
)
from latent_timing_duplex.phase1.paths import SELECTION_LOCKED_NAME

CHECKPOINT_BASENAMES = (
    "checkpoint.npz",
    "head.npz",
    "model.npz",
    "predictor.npz",
    "checkpoint.pt",
    "head.pt",
    "model.pt",
)
DIR_RE = re.compile(
    r"^h(?P<h>\d+)_lam(?P<lam>[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class SelectionLock:
    """Resolved ablation preference. Never claims a winner."""

    seed: int = PROTOCOL_SEED
    window_mode: str = "mid"
    window_s: float = 180.0
    horizon_frames: tuple[int, ...] = SPARK_TRAINED_HORIZON_FRAMES
    primary_lambda: float = PRIMARY_LAMBDA
    reference_lambda: float = REFERENCE_LAMBDA
    checkpoints: dict[str, str] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def preferred_pairs(self) -> list[tuple[int, float]]:
        pairs = [(h, self.primary_lambda) for h in self.horizon_frames]
        if self.reference_lambda != self.primary_lambda:
            pairs.extend((h, self.reference_lambda) for h in self.horizon_frames)
        return pairs


@dataclass
class LoadedCheckpoint:
    head: MLPPredictor
    horizon_frames: int
    lambda_reg: float
    path: Path
    format: str
    extra: dict[str, Any] = field(default_factory=dict)


def format_run_dirname(horizon_frames: int, lambda_reg: float) -> str:
    """Spark ablation folder name: ``h12_lam0.01``, ``h1_lam0``."""
    if float(lambda_reg) == 0.0:
        lam = "0"
    else:
        lam = f"{lambda_reg:g}"
    return f"h{int(horizon_frames)}_lam{lam}"


def parse_run_dirname(name: str) -> tuple[int, float] | None:
    match = DIR_RE.match(name.strip())
    if match is None:
        return None
    return int(match.group("h")), float(match.group("lam"))


def default_selection() -> SelectionLock:
    return SelectionLock()


def load_selection_lock(path: str | Path | None) -> SelectionLock:
    """Read ``SELECTION_LOCKED.json`` or return protocol defaults."""
    if path is None:
        return default_selection()
    root = Path(path)
    if root.is_dir():
        root = root / SELECTION_LOCKED_NAME
    if not root.is_file():
        return default_selection()
    data = json.loads(root.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{root} must be a JSON object")
    horizons = data.get("horizon_frames") or data.get("H") or data.get("horizons")
    if horizons is None:
        horizon_frames = SPARK_TRAINED_HORIZON_FRAMES
    else:
        horizon_frames = tuple(int(x) for x in horizons)
    window = data.get("window") or data.get("window_mode") or "mid"
    window_s = float(data.get("window_s") or data.get("W") or 180.0)
    if isinstance(window, str) and window.endswith("180"):
        window_mode = "mid" if window.startswith("mid") else "first"
    else:
        window_mode = str(window)
    ckpts = data.get("checkpoints") or data.get("preferred") or {}
    if not isinstance(ckpts, dict):
        ckpts = {}
    return SelectionLock(
        seed=int(data.get("seed", PROTOCOL_SEED)),
        window_mode=window_mode,
        window_s=window_s,
        horizon_frames=horizon_frames,
        primary_lambda=float(data.get("primary_lambda", data.get("lambda_primary", PRIMARY_LAMBDA))),
        reference_lambda=float(
            data.get("reference_lambda", data.get("lambda_reference", REFERENCE_LAMBDA))
        ),
        checkpoints={str(k): str(v) for k, v in ckpts.items()},
        extra={k: v for k, v in data.items() if k not in {"checkpoints", "preferred"}},
    )


def resolve_checkpoint_path(
    ablations_root: str | Path,
    horizon_frames: int,
    lambda_reg: float,
    *,
    selection: SelectionLock | None = None,
) -> Path:
    """Find a checkpoint file under ``ablations_root`` / ``h*_lam*``."""
    root = Path(ablations_root)
    if not root.is_dir():
        raise Phase1EvalInputMissing(
            f"ablations root {root} does not exist. On Spark this is typically "
            "/home/velvet/cs199-phase1-work/ablations (pass --ablations-root). "
            "CI does not need this directory."
        )
    lock = selection or default_selection()
    key = format_run_dirname(horizon_frames, lambda_reg)
    hinted = lock.checkpoints.get(key) or lock.checkpoints.get(f"h{horizon_frames}")
    candidates: list[Path] = []
    if hinted:
        hinted_path = Path(hinted)
        candidates.append(hinted_path if hinted_path.is_absolute() else root / hinted_path)
    run_dir = root / key
    # Also accept h12_lam0.010 / h12_lam0.0 variants by scanning.
    if run_dir.is_dir():
        candidates.append(run_dir)
        for base in CHECKPOINT_BASENAMES:
            candidates.append(run_dir / base)
    else:
        for child in sorted(root.iterdir()):
            parsed = parse_run_dirname(child.name)
            if parsed is None:
                continue
            h, lam = parsed
            if h == int(horizon_frames) and abs(lam - float(lambda_reg)) < 1e-12:
                candidates.append(child)
                for base in CHECKPOINT_BASENAMES:
                    candidates.append(child / base)
    for path in candidates:
        if path.is_file():
            return path
    raise Phase1EvalInputMissing(
        f"no checkpoint for H={horizon_frames} λ={lambda_reg} under {root}. "
        f"Expected {key}/checkpoint.npz (or head.npz / model.pt). "
        "Pass --checkpoint explicitly if the layout differs."
    )


def save_mlp_checkpoint(
    path: str | Path,
    head: MLPPredictor,
    *,
    horizon_frames: int,
    lambda_reg: float,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Write an in-repo numpy checkpoint (``[in, out]`` weights)."""
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "format": np.asarray("ltd-phase1-mlp-v1"),
        "layout": np.asarray("numpy"),
        "horizon_frames": np.asarray(int(horizon_frames)),
        "lambda_reg": np.asarray(float(lambda_reg)),
        "hidden_dim": np.asarray(head.hidden_dim),
        "embed_dim": np.asarray(head.embed_dim),
        "width": np.asarray(head.width),
        "n_layers": np.asarray(head.n_layers),
        "n_weight_tensors": np.asarray(len(head.layers)),
    }
    for i, layer in enumerate(head.layers):
        payload[f"w{i}"] = np.asarray(layer.w, dtype=np.float64)
        payload[f"b{i}"] = np.asarray(layer.b, dtype=np.float64)
    if extra:
        payload["extra_json"] = np.asarray(json.dumps(extra))
    np.savez(dest, **payload)
    return dest


def load_mlp_checkpoint(path: str | Path) -> LoadedCheckpoint:
    """Load ``.npz`` (preferred) or a torch ``.pt`` if torch is installed."""
    root = Path(path)
    if not root.is_file():
        raise Phase1EvalInputMissing(
            f"checkpoint {path!r} is not a file. Pass --checkpoint or "
            "--ablations-root with h*_lam*/checkpoint.npz."
        )
    parsed = parse_run_dirname(root.parent.name)
    if root.suffix == ".npz":
        return _load_npz(root, fallback=parsed)
    if root.suffix in {".pt", ".pth", ".ckpt"}:
        return _load_torch(root, fallback=parsed)
    raise ValueError(f"unsupported checkpoint suffix {root.suffix} ({root})")


def surprise_from_checkpoint(
    checkpoint: LoadedCheckpoint,
    hidden: np.ndarray,
    target: np.ndarray,
    *,
    kind: str = "mse",
) -> np.ndarray:
    """Per-source-frame surprise from a loaded head and cached tensors."""
    from latent_timing_duplex.phase1.surprise import surprise_values
    from latent_timing_duplex.phase1.horizons import pair_indices_frames

    h = np.asarray(hidden, dtype=np.float64)
    z = np.asarray(target, dtype=np.float64)
    if h.ndim != 2 or z.ndim != 2:
        raise ValueError(f"hidden/target must be 2-D, got {h.shape} / {z.shape}")
    if h.shape[0] != z.shape[0]:
        raise ValueError(f"T mismatch: hidden {h.shape[0]} vs target {z.shape[0]}")
    src, tgt = pair_indices_frames(h.shape[0], checkpoint.horizon_frames)
    if src.size == 0:
        raise ValueError(
            f"T={h.shape[0]} is too short for H={checkpoint.horizon_frames}"
        )
    pred = checkpoint.head.forward(h[src])
    return surprise_values(pred, z[tgt], kind=kind)


def _load_npz(path: Path, fallback: tuple[int, float] | None) -> LoadedCheckpoint:
    blob = np.load(path, allow_pickle=True)
    hidden_dim = int(blob["hidden_dim"]) if "hidden_dim" in blob else None
    embed_dim = int(blob["embed_dim"]) if "embed_dim" in blob else None
    width = int(blob["width"]) if "width" in blob else None
    n_layers = int(blob["n_layers"]) if "n_layers" in blob else None
    weights: list[np.ndarray] = []
    biases: list[np.ndarray] = []
    i = 0
    while f"w{i}" in blob:
        weights.append(np.asarray(blob[f"w{i}"], dtype=np.float64))
        biases.append(np.asarray(blob[f"b{i}"], dtype=np.float64))
        i += 1
    if not weights:
        raise KeyError(f"{path} has no w0/b0 tensors")
    if hidden_dim is None:
        hidden_dim = int(weights[0].shape[0])
    if embed_dim is None:
        embed_dim = int(weights[-1].shape[1] if weights[-1].ndim == 2 else weights[-1].shape[0])
    if n_layers is None:
        n_layers = max(1, len(weights) - 1)
    if width is None:
        width = int(weights[0].shape[1]) if len(weights) > 1 else hidden_dim
    head = MLPPredictor(
        hidden_dim=hidden_dim,
        embed_dim=embed_dim,
        width=width,
        n_layers=n_layers,
        seed=0,
    )
    if len(head.layers) != len(weights):
        raise ValueError(
            f"{path} has {len(weights)} weight tensors; reconstructed MLP has "
            f"{len(head.layers)} layers"
        )
    for layer, w, b in zip(head.layers, weights, biases):
        w = _coerce_weight(w, layer.n_in, layer.n_out)
        b = np.asarray(b, dtype=np.float64).reshape(-1)
        if b.shape[0] != layer.n_out:
            raise ValueError(f"bias {b.shape} != n_out {layer.n_out}")
        layer.w = w
        layer.b = b
    horizon = int(blob["horizon_frames"]) if "horizon_frames" in blob else (fallback[0] if fallback else 1)
    lam = float(blob["lambda_reg"]) if "lambda_reg" in blob else (fallback[1] if fallback else 0.0)
    extra: dict[str, Any] = {}
    if "extra_json" in blob:
        extra = json.loads(str(blob["extra_json"]))
    fmt = str(blob["format"]) if "format" in blob else "npz"
    return LoadedCheckpoint(
        head=head,
        horizon_frames=horizon,
        lambda_reg=lam,
        path=path,
        format=fmt,
        extra=extra,
    )


def _coerce_weight(w: np.ndarray, n_in: int, n_out: int) -> np.ndarray:
    arr = np.asarray(w, dtype=np.float64)
    if arr.shape == (n_in, n_out):
        return arr
    if arr.shape == (n_out, n_in):
        return arr.T
    raise ValueError(f"weight shape {arr.shape} is neither ({n_in}, {n_out}) nor torch ({n_out}, {n_in})")


def _load_torch(path: Path, fallback: tuple[int, float] | None) -> LoadedCheckpoint:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            f"torch is required to read {path}. Convert the Spark checkpoint "
            "to checkpoint.npz with save_mlp_checkpoint, or install torch."
        ) from exc
    blob = torch.load(path, map_location="cpu", weights_only=False)
    state: dict[str, Any]
    meta: dict[str, Any] = {}
    if isinstance(blob, dict) and "state_dict" in blob:
        state = dict(blob["state_dict"])
        meta = {k: v for k, v in blob.items() if k != "state_dict"}
    elif isinstance(blob, dict):
        state = blob
    else:
        raise ValueError(f"cannot interpret torch checkpoint {path}")
    weight_items = []
    bias_items = []
    for key, value in state.items():
        if not hasattr(value, "detach") and not isinstance(value, np.ndarray):
            continue
        arr = value.detach().float().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)
        lname = key.lower()
        if arr.ndim == 2 and ("weight" in lname or lname.startswith("w")):
            weight_items.append((key, arr))
        elif arr.ndim == 1 and "bias" in lname:
            bias_items.append((key, arr))
    if not weight_items:
        raise ValueError(f"{path} has no 2-D weight tensors")
    weight_items.sort(key=lambda kv: kv[0])
    bias_items.sort(key=lambda kv: kv[0])
    w0 = weight_items[0][1]
    w_last = weight_items[-1][1]
    # Infer numpy vs torch layout from first layer: torch is [out, in].
    if w0.shape[0] < w0.shape[1]:
        hidden_dim, width = int(w0.shape[1]), int(w0.shape[0])
        embed_dim = int(w_last.shape[0])
        layout_torch = True
    else:
        hidden_dim, width = int(w0.shape[0]), int(w0.shape[1])
        embed_dim = int(w_last.shape[1])
        layout_torch = False
    n_layers = max(1, len(weight_items) - 1)
    head = MLPPredictor(
        hidden_dim=hidden_dim,
        embed_dim=embed_dim,
        width=width,
        n_layers=n_layers,
        seed=0,
    )
    if len(head.layers) != len(weight_items):
        raise ValueError(
            f"{path} yielded {len(weight_items)} weights; MLP reconstruct has "
            f"{len(head.layers)} layers. Save an in-repo .npz instead."
        )
    for layer, (_k, w), (_bk, b) in zip(head.layers, weight_items, bias_items or [(None, None)] * len(weight_items)):
        if layout_torch:
            w = np.asarray(w, dtype=np.float64).T
        layer.w = _coerce_weight(w, layer.n_in, layer.n_out)
        if b is not None:
            layer.b = np.asarray(b, dtype=np.float64).reshape(-1)
    horizon = int(meta.get("horizon_frames", fallback[0] if fallback else 1))
    lam = float(meta.get("lambda_reg", fallback[1] if fallback else 0.0))
    return LoadedCheckpoint(
        head=head,
        horizon_frames=horizon,
        lambda_reg=lam,
        path=path,
        format="torch",
        extra={"keys": [k for k, _ in weight_items]},
    )
