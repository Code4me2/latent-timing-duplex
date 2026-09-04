"""Spark filename / id adapters. No absolute paths are opened here.

Hidden and target caches on spark-61dd are named ``candor_<uuid>.pt`` and
``dc_<uuid>.pt`` (torch), not ``{session_id}.npz``. Session ids in JSONLs
and transcripts may omit those prefixes.
"""

from __future__ import annotations

from pathlib import Path

from latent_timing_duplex.phase1.labels import normalize_session_id

SESSION_FILE_PREFIXES: tuple[str, ...] = (
    "candor_",
    "dc_",
    "duplexchat_",
    "duplex_",
)
ARRAY_SUFFIXES: tuple[str, ...] = (".npz", ".npy", ".pt", ".pth")
TRANSCRIPT_SUFFIXES: tuple[str, ...] = (".csv", ".json", ".jsonl", ".ndjson")


def strip_session_prefix(name: str) -> str:
    """Drop a leading ``candor_`` / ``dc_`` (any documented prefix)."""
    stem = str(name).strip()
    lower = stem.lower()
    for prefix in SESSION_FILE_PREFIXES:
        if lower.startswith(prefix):
            return stem[len(prefix) :]
    return stem


def session_id_aliases(session_id: str) -> list[str]:
    """Filename stems that may refer to the same conversation / episode."""
    raw = str(session_id).strip()
    norm = normalize_session_id(raw)
    stripped = strip_session_prefix(norm)
    aliases = [raw, norm, stripped, strip_session_prefix(raw)]
    for prefix in SESSION_FILE_PREFIXES:
        aliases.append(f"{prefix}{stripped}")
        aliases.append(f"{prefix}{norm}")
    seen: set[str] = set()
    out: list[str] = []
    for item in aliases:
        if item and item not in seen:
            seen.add(item)
            out.append(item)
    return out


def ids_match(left: str, right: str) -> bool:
    a = set(session_id_aliases(left))
    b = set(session_id_aliases(right))
    return bool(a & b)


def session_id_from_filename(path: str | Path) -> str:
    """Stem without suffix, then strip ``candor_`` / ``dc_``."""
    stem = Path(path).stem
    for extra in ("_transcript", "_transcription", "_turns", "_vad"):
        if stem.endswith(extra):
            stem = stem[: -len(extra)]
    return strip_session_prefix(stem)


def resolve_named_file(
    directory: str | Path,
    session_id: str,
    suffixes: tuple[str, ...] = ARRAY_SUFFIXES,
) -> Path | None:
    """Find ``{alias}{suffix}`` or a stem that matches after prefix strip."""
    root = Path(directory)
    if not root.is_dir():
        return None
    wanted = set(session_id_aliases(session_id))
    for alias in session_id_aliases(session_id):
        for suffix in suffixes:
            candidate = root / f"{alias}{suffix}"
            if candidate.is_file():
                return candidate
    for child in sorted(root.iterdir()):
        if not child.is_file() or child.suffix not in suffixes:
            continue
        stem = child.stem
        if stem in wanted or strip_session_prefix(stem) in wanted:
            return child
        if session_id_from_filename(child) in wanted:
            return child
    return None


def list_session_ids_in_dir(
    directory: str | Path,
    suffixes: tuple[str, ...] = ARRAY_SUFFIXES,
) -> list[str]:
    """Unique stripped session ids from files in ``directory`` (non-recursive)."""
    root = Path(directory)
    if not root.is_dir():
        return []
    seen: set[str] = set()
    out: list[str] = []
    for child in sorted(root.iterdir()):
        if not child.is_file() or child.suffix not in suffixes:
            continue
        sid = session_id_from_filename(child)
        if sid not in seen:
            seen.add(sid)
            out.append(sid)
    return out
