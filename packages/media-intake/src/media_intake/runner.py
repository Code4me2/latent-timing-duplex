"""Bounded, shell-free subprocess execution."""
from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import subprocess
import time
from typing import Protocol, Sequence


@dataclass(frozen=True)
class CommandResult:
    stdout: bytes
    stderr: bytes


class Runner(Protocol):
    def run(self, command: Sequence[str], *, timeout: float, log: Path | None = None) -> CommandResult: ...


class SubprocessRunner:
    def __init__(self, *, attempts: int = 1, retry_delay: float = 1.0) -> None:
        if not 1 <= attempts <= 5 or not 0 <= retry_delay <= 60:
            raise ValueError("runner attempts/delay are outside safe bounds")
        self.attempts = attempts
        self.retry_delay = retry_delay

    def run(self, command: Sequence[str], *, timeout: float, log: Path | None = None) -> CommandResult:
        if not command or timeout <= 0 or timeout > 24 * 60 * 60:
            raise ValueError("invalid command or timeout")
        last: Exception | None = None
        for attempt in range(self.attempts):
            try:
                result = subprocess.run(
                    list(command), capture_output=True, check=True, timeout=timeout,
                    shell=False,
                )
                if log is not None:
                    _append_log(log, result.stdout + result.stderr)
                return CommandResult(result.stdout, result.stderr)
            except (subprocess.SubprocessError, OSError) as exc:
                last = exc
                if log is not None:
                    payload = getattr(exc, "stdout", b"") or b""
                    payload += getattr(exc, "stderr", b"") or b""
                    _append_log(log, payload)
                if attempt + 1 < self.attempts:
                    time.sleep(self.retry_delay)
        assert last is not None
        raise last


def _append_log(path: Path, payload: bytes | str) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    data = payload.encode("utf-8", "replace") if isinstance(payload, str) else payload
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    with os.fdopen(descriptor, "ab") as handle:
        handle.write(data)
