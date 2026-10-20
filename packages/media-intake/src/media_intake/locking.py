"""Small cross-platform non-blocking output-directory lock."""
from __future__ import annotations

import os
from pathlib import Path
import stat

from .provenance import secure_directory

CANONICAL_LOCK_NAME = ".media-intake.lock"


class OutputLock:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.handle = None

    def __enter__(self) -> "OutputLock":
        secure_directory(self.path.parent)
        if self.path.is_symlink():
            raise ValueError("refusing symlinked output lock")
        flags = os.O_RDWR | os.O_CREAT
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            descriptor = os.open(self.path, flags, 0o600)
        except OSError as exc:
            if self.path.is_symlink():
                raise ValueError("refusing symlinked output lock") from exc
            raise
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            os.close(descriptor)
            raise ValueError("output lock must be a regular file")
        self.handle = os.fdopen(descriptor, "a+b")
        try:
            if os.name == "nt":
                import msvcrt
                self.handle.seek(0)
                if self.handle.read(1) == b"":
                    self.handle.write(b"0")
                    self.handle.flush()
                self.handle.seek(0)
                msvcrt.locking(self.handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            self.handle.close()
            self.handle = None
            raise RuntimeError("another acquisition process is using this output directory") from exc
        return self

    def __exit__(self, *_: object) -> None:
        if self.handle is None:
            return
        if os.name == "nt":
            import msvcrt
            self.handle.seek(0)
            msvcrt.locking(self.handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            fcntl.flock(self.handle, fcntl.LOCK_UN)
        self.handle.close()
