"""Load YAML configs from the repo ``configs/`` directory."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

_PACKAGE_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _PACKAGE_DIR.parents[1]


def repo_root() -> Path:
    """Best-effort repo root (editable install) or CWD fallback."""
    candidate = _REPO_ROOT
    if (candidate / "configs").is_dir():
        return candidate
    cwd = Path.cwd()
    if (cwd / "configs").is_dir():
        return cwd
    return candidate


def find_config(name: str = "default.yaml") -> Path:
    for base in (Path.cwd(), repo_root()):
        path = base / "configs" / name
        if path.is_file():
            return path
    raise FileNotFoundError(
        f"config {name!r} not found under ./configs or the repo configs/ directory"
    )


def load_config(name: str = "default.yaml") -> dict[str, Any]:
    path = find_config(name)
    with path.open() as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return data
