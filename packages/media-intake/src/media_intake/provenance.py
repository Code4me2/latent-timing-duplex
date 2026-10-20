"""Private storage, hashes, immutable JSON records, and artifact validation."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import uuid


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def secure_root(path: Path) -> Path:
    try:
        return secure_directory(path)
    except ValueError as exc:
        raise ValueError(f"unsafe output root: {exc}") from exc


def secure_directory(path: Path, *, root: Path | None = None) -> Path:
    """Create a directory without traversing an existing symlink component."""
    raw = path.expanduser()
    absolute = Path(os.path.abspath(os.fspath(raw)))
    if root is not None:
        root_absolute = Path(os.path.abspath(os.fspath(root.expanduser())))
        try:
            absolute.relative_to(root_absolute)
        except ValueError as exc:
            raise ValueError("directory escapes output root") from exc
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current = current / part
        if current.is_symlink():
            raise ValueError(f"refusing symlinked directory component: {current}")
        if current.exists():
            if not current.is_dir():
                raise ValueError(f"refusing unsafe directory: {current}")
        else:
            current.mkdir(mode=0o700)
    if root is not None:
        try:
            absolute.resolve(strict=True).relative_to(root_absolute.resolve(strict=True))
        except ValueError as exc:
            raise ValueError("directory escapes output root") from exc
    return absolute


def write_json_new(path: Path, value: dict) -> None:
    secure_directory(path.parent)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(path, flags, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
            handle.write("\n")
    except BaseException:
        try:
            os.close(descriptor)
        except OSError:
            pass
        raise


def write_bytes_new(path: Path, payload: bytes) -> None:
    secure_directory(path.parent)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)


def private_file(path: Path) -> None:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"expected regular, non-symlink file: {path.name}")
    try:
        path.chmod(0o600)
    except NotImplementedError:
        pass


def sha256_file(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"expected regular, non-symlink file: {path.name}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact(path: Path, root: Path) -> dict:
    root_absolute = Path(os.path.abspath(os.fspath(root)))
    candidate = Path(os.path.abspath(os.fspath(path)))
    current = Path(candidate.anchor)
    for part in candidate.parts[1:]:
        current = current / part
        if current.is_symlink():
            raise ValueError(f"refusing symlinked artifact path component: {current}")
    try:
        relative = candidate.resolve(strict=True).relative_to(root_absolute.resolve(strict=True))
    except ValueError as exc:
        raise ValueError("artifact escapes output root") from exc
    return {"path": str(relative), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def new_attempt(root: Path, source_id: str, *, parent: str = "attempts") -> Path:
    # source_id was validated before this boundary; UUIDs are generated here.
    return secure_directory(root / parent / uuid.uuid4().hex / source_id, root=root)
