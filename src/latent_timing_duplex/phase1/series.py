"""Load and align Moshi NLL / VAP / surprise series onto the 80 ms grid.

Phase 0 JSONLs live under Spark work trees. This module never opens those
absolute paths unless the caller passes them. Records are matched by
``session_id`` (window suffixes stripped). Full-length series are cropped
with the same first-W / mid-W rule as ``phase1.windows``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from latent_timing_duplex.baselines.vap import pool_to_chunk_grid
from latent_timing_duplex.exceptions import Phase1EvalInputMissing
from latent_timing_duplex.phase1.horizons import CHUNK_DURATION_S, MID180_N_FRAMES
from latent_timing_duplex.phase1.labels import normalize_session_id
from latent_timing_duplex.phase1.windows import WindowMode, window_bounds
from latent_timing_duplex.types import ChunkSignal, iter_chunks

SESSION_ID_KEYS = (
    "session_id",
    "conversation_id",
    "uuid",
    "clip_id",
    "episode_id",
    "id",
)
# Per-step only. Scalars such as ``audio_nll`` / ``p_shift_mean`` are aggregates
# and must not be expanded into a fake 1-frame series.
NLL_STEP_KEYS = (
    "audio_nll_per_step",
    "audio_per_step",
    "nll_audio_per_step",
    "user_nll_per_step",
    "user_channel_nll",
    "nll_per_step",
    "values",
)
NLL_MAYBE_SERIES_KEYS = (
    "nll_audio",
    "audio_nll",
    "user_nll",
    "nll",
)
NLL_AGGREGATE_KEYS = (
    "audio_nll",
    "nll_audio",
    "nll",
    "audio_nll_mean",
    "duration_weighted_nll",
    "audio_nll_dw",
)
VAP_STEP_KEYS = (
    "p_shift_per_step",
    "p_now_per_step",
    "p_future_per_step",
    "vap_p_shift_per_step",
)
VAP_MAYBE_SERIES_KEYS = (
    "p_shift",
    "p(shift)",
    "vap_p_shift",
    "p_now",
    "vap_p_now",
    "p_future",
    "vap_p_future",
)
VAP_AGGREGATE_KEYS = (
    "p_shift_mean",
    "p_now_mean",
    "p_future_mean",
    "p_shift_dw",
    "p_now_dw",
    "vap_p_shift",
    "p_shift",
    "p_now",
)
VAP_SHIFT_KEYS = ("p_shift_per_step", "p_shift", "p(shift)", "vap_p_shift")
VAP_NOW_KEYS = ("p_now_per_step", "p_now", "vap_p_now")
VAP_FUTURE_KEYS = ("p_future_per_step", "p_future", "vap_p_future")
MIN_SERIES_LEN = 2
# Keep old name so existing imports do not break; it is step-preferring now.
NLL_VALUE_KEYS = NLL_STEP_KEYS + NLL_MAYBE_SERIES_KEYS
SURPRISE_VALUE_KEYS = (
    "surprise",
    "jepa_surprise",
    "values",
    "mse",
    "prediction_error",
)
T_END_KEYS = ("t_end", "t_ends", "t_end_s")
DURATION_KEYS = (
    "duration_s",
    "duration_sec",
    "duration",
    "clip_duration_s",
    "length_s",
)


@dataclass
class AlignedSeries:
    """One named per-chunk scalar on a uniform 80 ms grid."""

    session_id: str
    name: str
    values: np.ndarray
    t_end: np.ndarray
    chunk_duration_s: float = CHUNK_DURATION_S
    notes: str = ""

    def to_chunks(self) -> list[ChunkSignal]:
        chunks: list[ChunkSignal] = []
        for t_end, value in zip(self.t_end.tolist(), self.values.tolist()):
            chunks.append(
                ChunkSignal(
                    t_start=float(t_end - self.chunk_duration_s),
                    t_end=float(t_end),
                    value=float(value),
                    name=self.name,
                )
            )
        return chunks


@dataclass
class SeriesIndex:
    """session_id → aligned series (normalized ids also registered)."""

    name: str
    records: dict[str, AlignedSeries] = field(default_factory=dict)

    def get(self, session_id: str) -> AlignedSeries:
        if session_id in self.records:
            return self.records[session_id]
        key = normalize_session_id(session_id)
        if key in self.records:
            return self.records[key]
        raise KeyError(f"no {self.name} series for session {session_id!r}")

    def session_ids(self) -> list[str]:
        seen: set[str] = set()
        out: list[str] = []
        for key in self.records:
            norm = normalize_session_id(key)
            if norm not in seen:
                seen.add(norm)
                out.append(norm)
        return out


def load_jsonl_records(path: str | Path) -> list[dict[str, Any]]:
    root = Path(path)
    if not root.is_file():
        raise FileNotFoundError(
            f"{path!r} is not a file. Pass a Spark Phase 0 JSONL or a "
            "synthetic fixture; this repo does not bake work-tree paths."
        )
    text = root.read_text(encoding="utf-8").strip()
    if not text:
        return []
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return [r for r in parsed if isinstance(r, dict)]
        if isinstance(parsed, dict):
            if "records" in parsed and isinstance(parsed["records"], list):
                return [r for r in parsed["records"] if isinstance(r, dict)]
            return [parsed]
    except json.JSONDecodeError:
        pass
    rows: list[dict[str, Any]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        if isinstance(obj, dict):
            rows.append(obj)
    return rows


def session_id_of(record: dict[str, Any]) -> str | None:
    for key in SESSION_ID_KEYS:
        if key in record and record[key] is not None:
            return str(record[key])
    return None


def _first_array(record: dict[str, Any], keys: tuple[str, ...]) -> np.ndarray | None:
    for key in keys:
        if key in record and record[key] is not None:
            return np.asarray(record[key], dtype=np.float64).reshape(-1)
    return None


def _as_step_series(
    record: dict[str, Any],
    step_keys: tuple[str, ...],
    maybe_keys: tuple[str, ...],
) -> np.ndarray | None:
    """Return a per-step vector, or None if the record is aggregate-only.

    A scalar or length-1 array is treated as an aggregate (``audio_nll``,
    ``p_shift_mean``). We never broadcast that into a fake T-length series.
    """
    for key in step_keys + maybe_keys:
        if key not in record or record[key] is None:
            continue
        arr = np.asarray(record[key], dtype=np.float64).reshape(-1)
        if arr.size >= MIN_SERIES_LEN:
            return arr
    return None


def record_has_aggregate(record: dict[str, Any], keys: tuple[str, ...]) -> bool:
    return any(k in record and record[k] is not None for k in keys)


def per_step_schema_help(kind: str) -> str:
    if kind == "nll":
        return (
            "Moshi NLL per-step JSONL schema (one object per session):\n"
            "  session_id / uuid / conversation_id\n"
            "  audio_nll_per_step: [float, ...]   # one value per 80 ms frame\n"
            "  duration_s or duration_sec: float\n"
            "  window: mid180 | first180 | full   # optional\n"
            "Aggregates such as audio_nll / duration_sec alone are not enough.\n"
            "Regenerate with: ltd phase1-export-series --print-schema\n"
            "or FrozenNLLExtractor.extract(crop_session(session, 180, 'mid'))."
        )
    return (
        "VAP per-step JSONL schema (one object per session):\n"
        "  session_id / uuid\n"
        "  p_shift_per_step or p_now_per_step: [float, ...]\n"
        "    (p_shift list, or p_now list inverted as 1-p_now when LEFT=user)\n"
        "    50 Hz series are pooled to 80 ms. Length-1 / p_shift_mean is aggregate.\n"
        "  duration_s or duration_sec: float\n"
        "Regenerate with: ltd phase1-export-series --print-schema\n"
        "or VAPBaseline.score_session(crop_session(session, 180, 'mid'))."
    )


def aggregate_only_error(kind: str, path: str | Path, n_records: int) -> Phase1EvalInputMissing:
    label = "Moshi NLL" if kind == "nll" else "VAP"
    return Phase1EvalInputMissing(
        f"{path} has {n_records} {label} record(s) but no per-step series "
        f"(found aggregates such as audio_nll / p_shift_mean / duration_sec). "
        f"AUROC cannot be computed from clip-level means. "
        f"Omit --{kind}-jsonl and pass --surprise-only for a surprise-only "
        f"dry run (not the primary RQ2 comparison), or re-emit per-step "
        f"JSONLs.\n\n{per_step_schema_help(kind)}"
    )


def _duration_of(record: dict[str, Any], n_values: int, chunk_s: float) -> float:
    for key in DURATION_KEYS:
        if key in record and record[key] is not None:
            return float(record[key])
    return float(n_values) * chunk_s


def crop_values_to_window(
    values: np.ndarray,
    *,
    duration_s: float,
    window_s: float = 180.0,
    mode: WindowMode = "mid",
    chunk_duration_s: float = CHUNK_DURATION_S,
) -> np.ndarray:
    """Slice a full-session 80 ms series to an equal-length window.

    Already-cropped series (length == ``window_s / chunk``) are returned as-is.
    """
    x = np.asarray(values, dtype=np.float64).reshape(-1)
    n_window = int(round(window_s / chunk_duration_s))
    if x.size == n_window:
        return x
    start, _end = window_bounds(duration_s, window_s, mode)
    i0 = int(round(start / chunk_duration_s))
    if i0 < 0:
        raise ValueError("window start rounded negative")
    if i0 + n_window > x.size:
        raise ValueError(
            f"cannot crop {x.size} frames to {mode}{int(window_s)} "
            f"(need index [{i0}:{i0 + n_window}]) from duration {duration_s}s"
        )
    return x[i0 : i0 + n_window]


def maybe_pool_50hz(
    values: np.ndarray,
    *,
    duration_s: float,
    chunk_duration_s: float = CHUNK_DURATION_S,
    src_hz: float = 50.0,
) -> np.ndarray:
    """Pool a 50 Hz VAP series onto the 80 ms grid when length looks like 50 Hz."""
    x = np.asarray(values, dtype=np.float64).reshape(-1)
    expected_80 = max(1, int(round(duration_s / chunk_duration_s)))
    expected_50 = max(1, int(round(duration_s * src_hz)))
    if x.size == expected_80 or x.size == MID180_N_FRAMES:
        return x
    closer_to_50 = abs(x.size - expected_50) <= abs(x.size - expected_80)
    if closer_to_50 and x.size >= 2 * expected_80:
        return pool_to_chunk_grid(x, src_hz=src_hz, chunk_duration_s=chunk_duration_s)
    return x


def values_to_aligned(
    session_id: str,
    name: str,
    values: np.ndarray,
    *,
    t_end: np.ndarray | None = None,
    chunk_duration_s: float = CHUNK_DURATION_S,
    t0: float = 0.0,
    notes: str = "",
) -> AlignedSeries:
    x = np.asarray(values, dtype=np.float64).reshape(-1)
    if x.size == 0:
        raise ValueError(f"{name} series for {session_id!r} is empty")
    if t_end is None:
        ends = t0 + (np.arange(x.size, dtype=np.float64) + 1.0) * chunk_duration_s
    else:
        ends = np.asarray(t_end, dtype=np.float64).reshape(-1)
        if ends.size != x.size:
            raise ValueError(f"t_end length {ends.size} != values {x.size}")
    return AlignedSeries(
        session_id=session_id,
        name=name,
        values=x,
        t_end=ends,
        chunk_duration_s=chunk_duration_s,
        notes=notes,
    )


def load_nll_jsonl(
    path: str | Path,
    *,
    window_s: float = 180.0,
    window_mode: WindowMode = "mid",
    name: str = "nll:moshi",
    crop: bool = True,
) -> SeriesIndex:
    """Load Phase 0 Moshi user-channel NLL series (per-step audio NLL)."""
    index = SeriesIndex(name=name)
    records = load_jsonl_records(path)
    n_aggregate = 0
    for rec in records:
        sid = session_id_of(rec)
        raw = _as_step_series(rec, NLL_STEP_KEYS, NLL_MAYBE_SERIES_KEYS)
        if raw is None:
            if sid is not None or record_has_aggregate(rec, NLL_AGGREGATE_KEYS + DURATION_KEYS):
                n_aggregate += 1
            continue
        if sid is None:
            continue
        duration = _duration_of(rec, raw.size, CHUNK_DURATION_S)
        values = raw
        notes = "nll-jsonl"
        if crop:
            already = rec.get("window") or rec.get("window_mode")
            already_mid = isinstance(already, str) and "mid" in already.lower()
            if values.size != int(round(window_s / CHUNK_DURATION_S)) and not already_mid:
                values = crop_values_to_window(
                    values,
                    duration_s=duration,
                    window_s=window_s,
                    mode=window_mode,
                )
                notes = f"nll-jsonl cropped {window_mode}{int(window_s)}"
            else:
                notes = "nll-jsonl already-windowed"
        t_end = _first_array(rec, T_END_KEYS)
        if t_end is not None and t_end.size != values.size:
            t_end = None
        series = values_to_aligned(sid, name, values, t_end=t_end, notes=notes)
        _store(index, sid, series)
    if not index.records:
        raise aggregate_only_error("nll", path, n_aggregate or len(records))
    return index


def load_vap_jsonl(
    path: str | Path,
    *,
    window_s: float = 180.0,
    window_mode: WindowMode = "mid",
    prefer: str = "p_shift",
    crop: bool = True,
) -> SeriesIndex:
    """Load Phase 0 VAP series. Prefers ``p_shift``; can invert ``p_now``."""
    name = "vap:p_shift" if prefer == "p_shift" else f"vap:{prefer}"
    index = SeriesIndex(name=name)
    records = load_jsonl_records(path)
    n_aggregate = 0
    for rec in records:
        sid = session_id_of(rec)
        shift = _as_step_series(rec, ("p_shift_per_step",), ("p_shift", "p(shift)", "vap_p_shift"))
        now = _as_step_series(rec, ("p_now_per_step",), ("p_now", "vap_p_now"))
        if prefer == "p_now":
            raw = now if now is not None else (1.0 - shift if shift is not None else None)
            name_i = "vap:p_now"
        else:
            if shift is not None:
                raw = shift
            elif now is not None:
                raw = 1.0 - now
            else:
                raw = None
            name_i = "vap:p_shift"
        if raw is None:
            if sid is not None or record_has_aggregate(rec, VAP_AGGREGATE_KEYS + DURATION_KEYS):
                n_aggregate += 1
            continue
        if sid is None:
            continue
        duration = _duration_of(rec, raw.size, CHUNK_DURATION_S)
        values = maybe_pool_50hz(raw, duration_s=duration)
        notes = "vap-jsonl"
        if crop:
            already = rec.get("window") or rec.get("window_mode")
            already_mid = isinstance(already, str) and "mid" in already.lower()
            n_window = int(round(window_s / CHUNK_DURATION_S))
            if values.size != n_window and not already_mid:
                values = crop_values_to_window(
                    values,
                    duration_s=duration,
                    window_s=window_s,
                    mode=window_mode,
                )
                notes = f"vap-jsonl cropped {window_mode}{int(window_s)}"
            else:
                notes = "vap-jsonl already-windowed"
        series = values_to_aligned(sid, name_i, values, notes=notes)
        _store(index, sid, series)
    if not index.records:
        raise aggregate_only_error("vap", path, n_aggregate or len(records))
    return index


def load_surprise_jsonl(
    path: str | Path,
    *,
    name: str = "jepa:surprise",
) -> SeriesIndex:
    """Load a precomputed surprise series (one record per session)."""
    index = SeriesIndex(name=name)
    for rec in load_jsonl_records(path):
        sid = session_id_of(rec)
        raw = _first_array(rec, SURPRISE_VALUE_KEYS)
        if sid is None or raw is None:
            continue
        series = values_to_aligned(sid, name, raw, t_end=_first_array(rec, T_END_KEYS), notes="surprise-jsonl")
        _store(index, sid, series)
    return index


def load_array_series(
    path: str | Path,
    session_id: str,
    name: str,
    *,
    key: str | None = None,
) -> AlignedSeries:
    """Load one ``.npy`` / ``.npz`` surprise or baseline series."""
    root = Path(path)
    if not root.is_file():
        raise FileNotFoundError(f"series file {path!r} is missing")
    if root.suffix == ".npz":
        blob = np.load(root)
        if key is not None:
            arr = np.asarray(blob[key], dtype=np.float64)
        else:
            for candidate in ("values", "surprise", "nll", "p_shift", "hidden"):
                if candidate in blob:
                    arr = np.asarray(blob[candidate], dtype=np.float64)
                    break
            else:
                raise KeyError(f"{path} has no values/surprise/nll/p_shift array")
        t_end = np.asarray(blob["t_end"], dtype=np.float64) if "t_end" in blob else None
    else:
        arr = np.asarray(np.load(root), dtype=np.float64)
        t_end = None
    if arr.ndim != 1:
        arr = np.asarray(arr, dtype=np.float64).reshape(-1)
    return values_to_aligned(session_id, name, arr, t_end=t_end, notes=str(root))


def intersect_by_frame(
    signals: dict[str, list[ChunkSignal]],
    *,
    chunk_duration_s: float = CHUNK_DURATION_S,
) -> dict[str, list[ChunkSignal]]:
    """Keep only frames whose ``t_end`` index is present in every signal.

    Surprise is often ``T − H`` (source-aligned). NLL / VAP are often ``T``.
    Scoring uses the intersection so every metric shares a timeline.
    """
    if not signals:
        return {}
    index_sets: list[set[int]] = []
    by_name: dict[str, dict[int, ChunkSignal]] = {}
    for name, chunks in signals.items():
        mapping: dict[int, ChunkSignal] = {}
        for chunk in chunks:
            idx = int(round(chunk.t_end / chunk_duration_s))
            mapping[idx] = chunk
        by_name[name] = mapping
        index_sets.append(set(mapping))
    common = set.intersection(*index_sets) if index_sets else set()
    ordered = sorted(common)
    out: dict[str, list[ChunkSignal]] = {}
    for name, mapping in by_name.items():
        out[name] = [mapping[i] for i in ordered]
    return out


def chunks_from_values(
    values: Iterable[float],
    name: str,
    *,
    chunk_duration_s: float = CHUNK_DURATION_S,
    t0: float = 0.0,
) -> list[ChunkSignal]:
    seq = [float(v) for v in values]
    if t0 == 0.0:
        return iter_chunks(
            duration_s=len(seq) * chunk_duration_s,
            chunk_duration_s=chunk_duration_s,
            values=seq,
            name=name,
        )
    return values_to_aligned("local", name, np.asarray(seq), t0=t0).to_chunks()


def _store(index: SeriesIndex, session_id: str, series: AlignedSeries) -> None:
    index.records[session_id] = series
    index.records[normalize_session_id(session_id)] = series
