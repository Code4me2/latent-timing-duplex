"""Re-emit per-step Moshi NLL and VAP JSONLs on mid-180 windows.

Phase 0 Spark JSONLs today are often **aggregate-only** (``audio_nll``,
``p_shift_mean``, ``duration_sec``). Those cannot be scored as per-frame
predictors. This helper writes the schema the compare scorer needs, using
the existing Phase 0 extractors:

* ``extract.nll.FrozenNLLExtractor`` + a locally loaded Moshi wrapper
* ``baselines.vap.VAPBaseline.score_session``

It does **not** invent a T-length series from a clip mean. Audio + local
weights are required; CI never runs this path.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from latent_timing_duplex.phase1.series import per_step_schema_help
from latent_timing_duplex.phase1.windows import DEFAULT_WINDOW_S, WindowMode, crop_session
from latent_timing_duplex.types import ChunkSignal, DualChannelSession


def schema_text() -> str:
    return (
        per_step_schema_help("nll")
        + "\n\n"
        + per_step_schema_help("vap")
        + "\n\n"
        "Python (Spark, local weights, already-loaded stereo session):\n"
        "  from latent_timing_duplex.phase1.export_series import (\n"
        "      nll_record_from_extractor, vap_record_from_baseline, write_jsonl,\n"
        "  )\n"
        "  from latent_timing_duplex.phase1.windows import crop_session\n"
        "  cropped = crop_session(session, window_s=180, mode='mid')\n"
        "  write_jsonl(nll_out, [nll_record_from_extractor(cropped, nll_ext)])\n"
        "  write_jsonl(vap_out, [vap_record_from_baseline(cropped, vap)])\n"
        "\n"
        "CLI: ltd phase1-export-series --print-schema\n"
        "Flags for a real export (Spark only):\n"
        "  --moshi-dir DIR --audio is already on the DualChannelSession\n"
        "  --vap-checkpoint FILE --nll-out FILE --vap-out FILE\n"
        "Same env as Phase 0 NLL: NO_CUDA_GRAPH=1 NO_TORCH_COMPILE=1. No GB10 fork.\n"
    )


def nll_record_from_values(
    session: DualChannelSession,
    values: Iterable[float],
    *,
    window: str = "mid180",
) -> dict[str, Any]:
    seq = [float(v) for v in values]
    return {
        "session_id": session.session_id,
        "audio_nll_per_step": seq,
        "duration_s": float(session.duration_s),
        "window": window,
    }


def vap_record_from_chunks(
    session: DualChannelSession,
    chunks: list[ChunkSignal],
    *,
    window: str = "mid180",
    field: str = "p_shift_per_step",
) -> dict[str, Any]:
    return {
        "session_id": session.session_id,
        field: [float(c.value) for c in chunks],
        "duration_s": float(session.duration_s),
        "window": window,
    }


def nll_record_from_extractor(
    session: DualChannelSession,
    extractor: Any,
    *,
    window_s: float = DEFAULT_WINDOW_S,
    mode: WindowMode = "mid",
    crop: bool = True,
) -> dict[str, Any]:
    """Run ``FrozenNLLExtractor.extract`` on an equal-length crop."""
    cropped = crop_session(session, window_s=window_s, mode=mode) if crop else session
    chunks = extractor.extract(cropped)
    return nll_record_from_values(
        cropped, [c.value for c in chunks], window=f"{mode}{int(window_s)}"
    )


def vap_record_from_baseline(
    session: DualChannelSession,
    vap: Any,
    *,
    window_s: float = DEFAULT_WINDOW_S,
    mode: WindowMode = "mid",
    crop: bool = True,
) -> dict[str, Any]:
    """Run ``VAPBaseline.score_session`` on an equal-length crop."""
    cropped = crop_session(session, window_s=window_s, mode=mode) if crop else session
    chunks = vap.score_session(cropped)
    return vap_record_from_chunks(
        cropped, list(chunks), window=f"{mode}{int(window_s)}"
    )


def write_jsonl(path: str | Path, records: Iterable[dict[str, Any]]) -> Path:
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec) + "\n")
    return dest
