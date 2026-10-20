from pathlib import Path

import pytest

from media_intake.locking import CANONICAL_LOCK_NAME, OutputLock


def test_lock_rejects_symlink_file(tmp_path):
    target = tmp_path / "target"
    target.write_bytes(b"")
    lock = tmp_path / CANONICAL_LOCK_NAME
    lock.symlink_to(target)
    with pytest.raises(ValueError, match="symlink"):
        with OutputLock(lock):
            pass


def test_canonical_lock_serializes_users(tmp_path):
    path = tmp_path / CANONICAL_LOCK_NAME
    with OutputLock(path):
        with pytest.raises(RuntimeError, match="another acquisition"):
            with OutputLock(path):
                pass
