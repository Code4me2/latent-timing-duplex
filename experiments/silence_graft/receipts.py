"""Canonical HMAC helpers for private semantic-silence receipts."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import stat
from collections.abc import Mapping
from pathlib import Path
from typing import Any

_SHA256 = re.compile(r"[0-9a-f]{64}")
_SIGNATURE_FIELD = "hmac_sha256"


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_digest(value: Any, where: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{where} must be a lowercase SHA-256 digest")
    return value


def read_hmac_key(path: str | Path) -> bytes:
    """Read one immutable key file descriptor, rejecting links and loose modes."""
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    except OSError as error:
        raise ValueError("HMAC key must be a readable regular non-symlink file") from error
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode):
            raise ValueError("HMAC key must be a regular non-symlink file")
        if stat.S_IMODE(info.st_mode) != 0o600:
            raise ValueError("HMAC key must have mode 0600")
        chunks = []
        while chunk := os.read(descriptor, 1024 * 1024):
            chunks.append(chunk)
        key = b"".join(chunks)
    finally:
        os.close(descriptor)
    if len(key) < 32:
        raise ValueError("HMAC key must be at least 32 bytes")
    return key


def key_fingerprint(key: bytes) -> str:
    if len(key) < 32:
        raise ValueError("HMAC key must be at least 32 bytes")
    return sha256_bytes(key)


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def sign_receipt(receipt: Mapping[str, Any], key: bytes) -> dict[str, Any]:
    """Return a copy signed over its canonical JSON without ``hmac_sha256``."""
    if _SIGNATURE_FIELD in receipt:
        raise ValueError("receipt is already signed")
    signed = dict(receipt)
    signed["key_fingerprint"] = key_fingerprint(key)
    signed[_SIGNATURE_FIELD] = hmac.new(key, canonical_json(signed), hashlib.sha256).hexdigest()
    return signed


def verify_receipt(receipt: Mapping[str, Any], key: bytes, where: str = "receipt") -> None:
    """Verify a canonical receipt HMAC and its nonsecret key fingerprint."""
    if not isinstance(receipt, Mapping):
        raise ValueError(f"{where} must be a JSON object")
    signature = require_digest(receipt.get(_SIGNATURE_FIELD), f"{where} HMAC")
    if receipt.get("key_fingerprint") != key_fingerprint(key):
        raise ValueError(f"{where} key fingerprint mismatch")
    unsigned = dict(receipt)
    del unsigned[_SIGNATURE_FIELD]
    expected = hmac.new(key, canonical_json(unsigned), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(signature, expected):
        raise ValueError(f"{where} HMAC verification failed")


def receipt_bytes(receipt: Mapping[str, Any]) -> bytes:
    return (json.dumps(receipt, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True) + "\n").encode("utf-8")
