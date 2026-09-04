"""Turn-event comparison: JEPA surprise vs Moshi NLL vs VAP on one timeline.

This is the Phase 1 RQ2 scorer. It does **not** claim a winner. It writes
AUROC / AUPRC / precision-recall operating points for every named signal
on the same mid-180 (default) windows and the same event labels.

Inputs are all caller-supplied paths. CI uses synthetic tensors.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from latent_timing_duplex.eval.harness import DEFAULT_HORIZONS_S, frames_and_labels
from latent_timing_duplex.eval.metrics import (
    average_precision,
    auroc,
    efron_bootstrap_mean_ci,
    f1max_operating_point,
    precision_at_recall,
)
from latent_timing_duplex.exceptions import Phase1EvalInputMissing
from latent_timing_duplex.phase1.checkpoint import (
    LoadedCheckpoint,
    SelectionLock,
    load_mlp_checkpoint,
    load_selection_lock,
    resolve_checkpoint_path,
    surprise_from_checkpoint,
)
from latent_timing_duplex.phase1.horizons import (
    CHUNK_DURATION_S,
    PRIMARY_LAMBDA,
    PROTOCOL_SEED,
    REFERENCE_LAMBDA,
    SPARK_TRAINED_HORIZON_FRAMES,
)
from latent_timing_duplex.phase1.labels import (
    CompositeEventSource,
    JsonlEventSource,
    TranscriptProxySource,
    TurnEventSource,
    VadProxySource,
    attach_events,
    describe_label_limitations,
    normalize_session_id,
)
from latent_timing_duplex.phase1.series import (
    SeriesIndex,
    chunks_from_values,
    intersect_by_frame,
    load_nll_jsonl,
    load_surprise_jsonl,
    load_vap_jsonl,
)
from latent_timing_duplex.phase1.windows import WindowMode, crop_session
from latent_timing_duplex.types import (
    TURN_EVENT_KINDS,
    ChunkSignal,
    DualChannelSession,
    TurnEventKind,
)

RECALL_TARGETS = (0.30, 0.50, 0.70)


@dataclass(frozen=True)
class CompareRow:
    """One signal × event kind × eval-horizon metric block."""

    signal: str
    event_kind: TurnEventKind
    horizon_s: float
    n_chunks: int
    n_positives: int
    n_negatives: int
    auroc: float | None
    auprc: float | None
    f1_max: float | None
    precision_at_f1max: float | None
    recall_at_f1max: float | None
    threshold_at_f1max: float | None
    precision_at_r30: float | None
    precision_at_r50: float | None
    precision_at_r70: float | None
    session_id: str | None = None
    predictor_horizon_frames: int | None = None
    lambda_reg: float | None = None


@dataclass
class SessionCompare:
    session_id: str
    duration_s: float
    n_events: int
    rows: list[CompareRow] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


@dataclass
class AggregateRow:
    signal: str
    event_kind: TurnEventKind
    horizon_s: float
    n_sessions: int
    mean_auroc: float | None
    mean_auprc: float | None
    auroc_ci: tuple[float, float] | None
    auprc_ci: tuple[float, float] | None
    mean_f1_max: float | None
    predictor_horizon_frames: int | None = None
    lambda_reg: float | None = None
    reduction: str = "unweighted_session_mean"


@dataclass
class CompareReport:
    """Machine-readable Phase 1 compare output. No winner field."""

    protocol: str = "EVAL_PROTOCOL_PHASE1"
    seed: int = PROTOCOL_SEED
    window_mode: str = "mid"
    window_s: float = 180.0
    eval_horizons_s: tuple[float, ...] = DEFAULT_HORIZONS_S
    predictor_horizon_frames: int | None = None
    lambda_reg: float | None = None
    lambda_role: str = ""
    signals: list[str] = field(default_factory=list)
    sessions: list[SessionCompare] = field(default_factory=list)
    aggregate: list[AggregateRow] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "protocol": self.protocol,
            "seed": self.seed,
            "window_mode": self.window_mode,
            "window_s": self.window_s,
            "eval_horizons_s": list(self.eval_horizons_s),
            "predictor_horizon_frames": self.predictor_horizon_frames,
            "lambda_reg": self.lambda_reg,
            "lambda_role": self.lambda_role,
            "signals": list(self.signals),
            "sessions": [_session_json(s) for s in self.sessions],
            "aggregate": [_agg_json(a) for a in self.aggregate],
            "limitations": list(self.limitations),
            "notes": list(self.notes),
            "claim": (
                "Code + metrics only. Do not treat this file as evidence that "
                "surprise beats NLL or VAP."
            ),
        }


@dataclass
class EvalConfig:
    window_s: float = 180.0
    window_mode: WindowMode = "mid"
    eval_horizons_s: Sequence[float] = DEFAULT_HORIZONS_S
    event_kinds: Sequence[TurnEventKind] = TURN_EVENT_KINDS
    seed: int = PROTOCOL_SEED
    bootstrap_b: int = 10000
    horizon_frames: int = 12
    lambda_reg: float = PRIMARY_LAMBDA
    lambda_role: str = "primary"
    surprise_kind: str = "mse"


@dataclass
class EvalPaths:
    """All optional except what a given mode needs. No baked Spark paths."""

    ablations_root: Path | None = None
    selection_locked: Path | None = None
    checkpoint: Path | None = None
    hidden: Path | None = None
    hidden_dir: Path | None = None
    target: Path | None = None
    target_dir: Path | None = None
    surprise_jsonl: Path | None = None
    nll_jsonl: Path | None = None
    vap_jsonl: Path | None = None
    labels: Path | None = None
    transcripts: Path | None = None
    vad: Path | None = None
    output: Path | None = None


def score_aligned_signals(
    session: DualChannelSession,
    signals: dict[str, Sequence[ChunkSignal]],
    *,
    horizons_s: Sequence[float] = DEFAULT_HORIZONS_S,
    event_kinds: Sequence[TurnEventKind] = TURN_EVENT_KINDS,
    predictor_horizon_frames: int | None = None,
    lambda_reg: float | None = None,
    align: bool = True,
) -> list[CompareRow]:
    """AUROC / AUPRC / PR operating points on an already-aligned session."""
    named: dict[str, list[ChunkSignal]]
    if align:
        named = intersect_by_frame({k: list(v) for k, v in signals.items()})
    else:
        named = {k: list(v) for k, v in signals.items()}
    rows: list[CompareRow] = []
    for name, chunks in named.items():
        for kind in event_kinds:
            for horizon in horizons_s:
                _t, y, s = frames_and_labels(session, chunks, float(horizon), kind)
                prec30 = precision_at_recall(y, s, 0.30) if y.size else None
                prec50 = precision_at_recall(y, s, 0.50) if y.size else None
                prec70 = precision_at_recall(y, s, 0.70) if y.size else None
                thr, p_f1, r_f1, f1 = (
                    f1max_operating_point(y, s) if y.size else (None, None, None, None)
                )
                n_pos = int(y.sum()) if y.size else 0
                rows.append(
                    CompareRow(
                        signal=name,
                        event_kind=kind,
                        horizon_s=float(horizon),
                        n_chunks=int(y.size),
                        n_positives=n_pos,
                        n_negatives=int(y.size - n_pos) if y.size else 0,
                        auroc=auroc(y, s) if y.size else None,
                        auprc=average_precision(y, s) if y.size else None,
                        f1_max=f1,
                        precision_at_f1max=p_f1,
                        recall_at_f1max=r_f1,
                        threshold_at_f1max=thr,
                        precision_at_r30=prec30,
                        precision_at_r50=prec50,
                        precision_at_r70=prec70,
                        session_id=session.session_id,
                        predictor_horizon_frames=predictor_horizon_frames,
                        lambda_reg=lambda_reg,
                    )
                )
    return rows


def aggregate_session_rows(
    sessions: Sequence[SessionCompare],
    *,
    seed: int = PROTOCOL_SEED,
    n_boot: int = 10000,
    predictor_horizon_frames: int | None = None,
    lambda_reg: float | None = None,
) -> list[AggregateRow]:
    """Unweighted mean of per-session metrics + Efron CI over sessions."""
    buckets: dict[tuple[str, TurnEventKind, float], list[CompareRow]] = {}
    for sess in sessions:
        for row in sess.rows:
            key = (row.signal, row.event_kind, row.horizon_s)
            buckets.setdefault(key, []).append(row)
    out: list[AggregateRow] = []
    for (signal, kind, horizon), rows in sorted(buckets.items()):
        aurocs = np.array([_or_nan(r.auroc) for r in rows], dtype=np.float64)
        aps = np.array([_or_nan(r.auprc) for r in rows], dtype=np.float64)
        f1s = np.array([_or_nan(r.f1_max) for r in rows], dtype=np.float64)
        n_ok_auroc = int(np.isfinite(aurocs).sum())
        out.append(
            AggregateRow(
                signal=signal,
                event_kind=kind,
                horizon_s=horizon,
                n_sessions=n_ok_auroc,
                mean_auroc=_nanmean(aurocs),
                mean_auprc=_nanmean(aps),
                auroc_ci=efron_bootstrap_mean_ci(aurocs, n_boot=n_boot, seed=seed),
                auprc_ci=efron_bootstrap_mean_ci(aps, n_boot=n_boot, seed=seed),
                mean_f1_max=_nanmean(f1s),
                predictor_horizon_frames=predictor_horizon_frames,
                lambda_reg=lambda_reg,
            )
        )
    return out


def build_label_source(paths: EvalPaths) -> TurnEventSource:
    sources: list[TurnEventSource] = []
    if paths.labels is not None:
        sources.append(JsonlEventSource.from_path(paths.labels))
    if paths.transcripts is not None:
        sources.append(TranscriptProxySource.from_path(paths.transcripts))
    if paths.vad is not None:
        sources.append(VadProxySource.from_path(paths.vad))
    if not sources:
        raise Phase1EvalInputMissing(
            "no turn-event labels. Pass --labels (gold JSONL), or "
            "--transcripts / --vad for speaker-change / onset proxies. "
            "See docs/EVAL_PROTOCOL_PHASE1.md."
        )
    return sources[0] if len(sources) == 1 else CompositeEventSource(sources)


def load_hidden_target(
    session_id: str,
    paths: EvalPaths,
) -> tuple[np.ndarray, np.ndarray]:
    hidden_path = _resolve_array(session_id, paths.hidden, paths.hidden_dir)
    target_path = _resolve_array(session_id, paths.target, paths.target_dir)
    hidden = _load_matrix(hidden_path, ("hidden", "arr_0"))
    target = _load_matrix(target_path, ("target", "arr_0"))
    if hidden.shape[0] != target.shape[0]:
        raise ValueError(
            f"{session_id}: hidden T={hidden.shape[0]} vs target T={target.shape[0]}"
        )
    return hidden, target


def surprise_chunks_for_session(
    session_id: str,
    *,
    checkpoint: LoadedCheckpoint | None,
    paths: EvalPaths,
    surprise_index: SeriesIndex | None,
    kind: str = "mse",
) -> list[ChunkSignal]:
    if surprise_index is not None:
        try:
            return surprise_index.get(session_id).to_chunks()
        except KeyError:
            pass
    if checkpoint is None:
        raise Phase1EvalInputMissing(
            f"no surprise series and no checkpoint for {session_id!r}. "
            "Pass --surprise-jsonl or --checkpoint / --ablations-root plus "
            "matching --hidden-dir and --target-dir."
        )
    hidden, target = load_hidden_target(session_id, paths)
    values = surprise_from_checkpoint(checkpoint, hidden, target, kind=kind)
    return chunks_from_values(values.tolist(), "jepa:surprise")


def compare_session(
    session: DualChannelSession,
    signals: dict[str, Sequence[ChunkSignal]],
    *,
    config: EvalConfig | None = None,
) -> SessionCompare:
    cfg = config or EvalConfig()
    rows = score_aligned_signals(
        session,
        signals,
        horizons_s=cfg.eval_horizons_s,
        event_kinds=cfg.event_kinds,
        predictor_horizon_frames=cfg.horizon_frames,
        lambda_reg=cfg.lambda_reg,
    )
    return SessionCompare(
        session_id=session.session_id,
        duration_s=session.duration_s,
        n_events=len(session.events),
        rows=rows,
    )


def run_synthetic_compare(
    *,
    duration_s: float = 24.0,
    seed: int = PROTOCOL_SEED,
    horizon_frames: int = 1,
    lambda_reg: float = PRIMARY_LAMBDA,
    bootstrap_b: int = 64,
) -> CompareReport:
    """CPU demo: synthetic events + three fake series. No weights, no files."""
    from latent_timing_duplex.data.synthetic import generate_synthetic_session

    bundle = generate_synthetic_session(duration_s=duration_s, seed=seed)
    session = crop_session(bundle.session, window_s=min(16.0, duration_s), mode="mid")
    n = int(session.duration_s / CHUNK_DURATION_S)
    rng = np.random.default_rng(seed)
    # Re-peak salience on the cropped event times so the demo table is defined.
    from latent_timing_duplex.data.synthetic import _predictive_signal

    pred = _predictive_signal(session, CHUNK_DURATION_S)
    nll_vals = 0.4 + 0.15 * np.array([c.value for c in pred]) + 0.05 * rng.normal(size=n)
    vap_vals = 0.5 + 0.2 * np.array([c.value for c in pred]) + 0.08 * rng.normal(size=n)
    rand_vals = rng.normal(size=n)
    surprise = pred[: n - horizon_frames] if horizon_frames < n else pred
    signals = {
        "jepa:surprise": [
            ChunkSignal(c.t_start, c.t_end, c.value, name="jepa:surprise") for c in surprise
        ],
        "nll:moshi": chunks_from_values(nll_vals.tolist(), "nll:moshi"),
        "vap:p_shift": chunks_from_values(vap_vals.tolist(), "vap:p_shift"),
        "random": chunks_from_values(rand_vals.tolist(), "random"),
    }
    cfg = EvalConfig(
        window_s=session.duration_s,
        window_mode="mid",
        eval_horizons_s=(0.50, 1.00),
        seed=seed,
        bootstrap_b=bootstrap_b,
        horizon_frames=horizon_frames,
        lambda_reg=lambda_reg,
        lambda_role="synthetic",
    )
    compared = compare_session(session, signals, config=cfg)
    compared.notes.append("synthetic demo; not a scientific result")
    report = CompareReport(
        seed=seed,
        window_mode="mid",
        window_s=session.duration_s,
        eval_horizons_s=tuple(cfg.eval_horizons_s),
        predictor_horizon_frames=horizon_frames,
        lambda_reg=lambda_reg,
        lambda_role="synthetic",
        signals=list(signals),
        sessions=[compared],
        aggregate=aggregate_session_rows(
            [compared],
            seed=seed,
            n_boot=bootstrap_b,
            predictor_horizon_frames=horizon_frames,
            lambda_reg=lambda_reg,
        ),
        limitations=describe_label_limitations()
        + ["This run used generate_synthetic_session, not CANDOR / DuplexChat."],
        notes=[
            "synthetic compare — do not report these numbers as Phase 1 findings",
            f"mid-window crop {session.duration_s}s (protocol demo; Spark uses W=180)",
        ],
    )
    return report


def run_turn_event_eval(
    paths: EvalPaths,
    config: EvalConfig | None = None,
    *,
    session_ids: Sequence[str] | None = None,
    sessions: Sequence[DualChannelSession] | None = None,
) -> CompareReport:
    """Score surprise vs NLL vs VAP. ``sessions`` injects in-memory timelines (tests)."""
    cfg = config or EvalConfig()
    selection = load_selection_lock(paths.selection_locked or paths.ablations_root)
    checkpoint = _maybe_load_checkpoint(paths, cfg, selection)
    surprise_index = (
        load_surprise_jsonl(paths.surprise_jsonl) if paths.surprise_jsonl is not None else None
    )
    nll_index = (
        load_nll_jsonl(paths.nll_jsonl, window_s=cfg.window_s, window_mode=cfg.window_mode)
        if paths.nll_jsonl is not None
        else None
    )
    vap_index = (
        load_vap_jsonl(paths.vap_jsonl, window_s=cfg.window_s, window_mode=cfg.window_mode)
        if paths.vap_jsonl is not None
        else None
    )
    labels = build_label_source(paths) if (paths.labels or paths.transcripts or paths.vad) else None

    if sessions is None:
        ids = list(session_ids or [])
        if not ids:
            ids = _infer_session_ids(paths, surprise_index, nll_index, vap_index)
        if not ids:
            raise Phase1EvalInputMissing(
                "no session ids. Pass --session-id, or provide JSONLs / "
                "hidden-dir whose filenames are session ids."
            )
        if labels is None:
            raise Phase1EvalInputMissing(
                "real sessions need --labels or --transcripts/--vad. "
                "Use --synthetic for the CPU demo."
            )
        built: list[DualChannelSession] = []
        for sid in ids:
            events = labels.events_for(sid)
            sess = DualChannelSession(
                session_id=sid,
                duration_s=cfg.window_s,
                events=events,
                source="phase1-eval",
                notes=f"equal-length {cfg.window_mode}{int(cfg.window_s)}",
            )
            # Events from full-session transcripts must be cropped.
            needs_crop = any(e.t >= cfg.window_s for e in events) or any(e.t < 0 for e in events)
            if needs_crop:
                # Caller should pass full duration via labels; without it we
                # only keep events already inside [0, W).
                sess = DualChannelSession(
                    session_id=sid,
                    duration_s=cfg.window_s,
                    events=[e for e in events if 0.0 <= e.t < cfg.window_s],
                    source="phase1-eval",
                    notes="events already treated as window-relative "
                    "(full-session crop needs duration_s on the session)",
                )
            built.append(sess)
        sessions = built

    compared: list[SessionCompare] = []
    notes = [
        f"protocol seed {cfg.seed}; bootstrap B={cfg.bootstrap_b} over sessions",
        f"predictor H={cfg.horizon_frames} frames, λ={cfg.lambda_reg} ({cfg.lambda_role})",
        "mid-180 is primary; this run uses "
        f"{cfg.window_mode} W={cfg.window_s}s",
    ]
    for session in sessions:
        sid = normalize_session_id(session.session_id)
        labeled = session
        if labels is not None and not session.events:
            labeled = attach_events(session, labels.events_for(sid))
        signals: dict[str, list[ChunkSignal]] = {}
        signals["jepa:surprise"] = surprise_chunks_for_session(
            sid,
            checkpoint=checkpoint,
            paths=paths,
            surprise_index=surprise_index,
            kind=cfg.surprise_kind,
        )
        if nll_index is not None:
            signals["nll:moshi"] = nll_index.get(sid).to_chunks()
        if vap_index is not None:
            signals["vap:p_shift"] = vap_index.get(sid).to_chunks()
        result = compare_session(labeled, signals, config=cfg)
        compared.append(result)

    missing_baselines = []
    if nll_index is None:
        missing_baselines.append("nll:moshi (--nll-jsonl)")
    if vap_index is None:
        missing_baselines.append("vap:p_shift (--vap-jsonl)")
    if missing_baselines:
        notes.append("missing baselines (scored without them): " + ", ".join(missing_baselines))

    report = CompareReport(
        seed=cfg.seed,
        window_mode=cfg.window_mode,
        window_s=cfg.window_s,
        eval_horizons_s=tuple(float(h) for h in cfg.eval_horizons_s),
        predictor_horizon_frames=cfg.horizon_frames,
        lambda_reg=cfg.lambda_reg,
        lambda_role=cfg.lambda_role,
        signals=sorted({row.signal for sess in compared for row in sess.rows}),
        sessions=compared,
        aggregate=aggregate_session_rows(
            compared,
            seed=cfg.seed,
            n_boot=cfg.bootstrap_b,
            predictor_horizon_frames=cfg.horizon_frames,
            lambda_reg=cfg.lambda_reg,
        ),
        limitations=describe_label_limitations(),
        notes=notes,
    )
    return report


def format_compare_report(report: CompareReport) -> str:
    """Plain-text table: signal × event × horizon with AUROC / AUPRC / F1."""
    header = (
        f"{'signal':<16} {'event':<13} {'h(s)':>6} "
        f"{'n+':>5} {'n-':>5} {'AUROC':>7} {'AUPRC':>7} {'F1max':>7} "
        f"{'P@R50':>7}"
    )
    lines = [
        f"Phase 1 turn-event compare  protocol={report.protocol}",
        f"window={report.window_mode}{int(report.window_s)}  seed={report.seed}  "
        f"H={report.predictor_horizon_frames}  λ={report.lambda_reg} "
        f"({report.lambda_role})",
        "No empirical claim: tables are metrics only.",
        "",
        header,
        "-" * len(header),
    ]
    # Prefer aggregate when several sessions exist; else print session rows.
    if report.aggregate and len(report.sessions) > 1:
        for row in report.aggregate:
            lines.append(
                f"{row.signal:<16} {row.event_kind:<13} {row.horizon_s:6.2f} "
                f"{row.n_sessions:5d} {'':>5} "
                f"{_fmt(row.mean_auroc)} {_fmt(row.mean_auprc)} "
                f"{_fmt(row.mean_f1_max)} {'':>7}"
            )
        lines.append("")
        lines.append("aggregate = unweighted mean of per-session metrics; n+ column is n_sessions")
    else:
        for sess in report.sessions:
            for row in sess.rows:
                lines.append(
                    f"{row.signal:<16} {row.event_kind:<13} {row.horizon_s:6.2f} "
                    f"{row.n_positives:5d} {row.n_negatives:5d} "
                    f"{_fmt(row.auroc)} {_fmt(row.auprc)} {_fmt(row.f1_max)} "
                    f"{_fmt(row.precision_at_r50)}"
                )
    if report.notes:
        lines.append("")
        lines.extend(report.notes)
    return "\n".join(lines)


def write_report(report: CompareReport, path: str | Path) -> Path:
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(report.to_jsonable(), indent=2) + "\n", encoding="utf-8")
    return dest


def protocol_defaults() -> dict[str, Any]:
    return {
        "window_mode": "mid",
        "window_s": 180.0,
        "seed": PROTOCOL_SEED,
        "horizon_frames": list(SPARK_TRAINED_HORIZON_FRAMES),
        "lambda_primary": PRIMARY_LAMBDA,
        "lambda_reference": REFERENCE_LAMBDA,
        "eval_horizons_s": list(DEFAULT_HORIZONS_S),
        "no_phase2": True,
        "plan_half_up_frames": {0.08: 1, 1.00: 13, 5.00: 63},
        "spark_trained_frames": list(SPARK_TRAINED_HORIZON_FRAMES),
    }


def _maybe_load_checkpoint(
    paths: EvalPaths,
    cfg: EvalConfig,
    selection: SelectionLock,
) -> LoadedCheckpoint | None:
    if paths.checkpoint is not None:
        return load_mlp_checkpoint(paths.checkpoint)
    if paths.ablations_root is not None:
        ckpt = resolve_checkpoint_path(
            paths.ablations_root,
            cfg.horizon_frames,
            cfg.lambda_reg,
            selection=selection,
        )
        return load_mlp_checkpoint(ckpt)
    return None


def _infer_session_ids(
    paths: EvalPaths,
    surprise_index: SeriesIndex | None,
    nll_index: SeriesIndex | None,
    vap_index: SeriesIndex | None,
) -> list[str]:
    ids: list[str] = []
    for index in (surprise_index, nll_index, vap_index):
        if index is not None:
            ids.extend(index.session_ids())
    if paths.hidden_dir is not None and paths.hidden_dir.is_dir():
        for child in sorted(paths.hidden_dir.glob("*.npz")):
            ids.append(child.stem)
        for child in sorted(paths.hidden_dir.glob("*.npy")):
            ids.append(child.stem)
    # unique, stable
    seen: set[str] = set()
    out: list[str] = []
    for sid in ids:
        key = normalize_session_id(sid)
        if key not in seen:
            seen.add(key)
            out.append(key)
    return out


def _resolve_array(session_id: str, single: Path | None, directory: Path | None) -> Path:
    if single is not None:
        if not single.is_file():
            raise Phase1EvalInputMissing(f"array file {single} is missing")
        return single
    if directory is None:
        raise Phase1EvalInputMissing(
            f"need --hidden/--target or --hidden-dir/--target-dir for {session_id}"
        )
    for suffix in (".npz", ".npy"):
        candidate = directory / f"{session_id}{suffix}"
        if candidate.is_file():
            return candidate
        candidate = directory / f"{normalize_session_id(session_id)}{suffix}"
        if candidate.is_file():
            return candidate
    raise Phase1EvalInputMissing(
        f"no hidden/target array for {session_id} under {directory}"
    )


def _load_matrix(path: Path, keys: tuple[str, ...]) -> np.ndarray:
    if path.suffix == ".npz":
        blob = np.load(path)
        for key in keys:
            if key in blob:
                arr = np.asarray(blob[key], dtype=np.float64)
                break
        else:
            raise KeyError(f"{path} has none of {keys}")
    else:
        arr = np.asarray(np.load(path), dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"{path} must be [T, D], got {arr.shape}")
    return arr


def _or_nan(value: float | None) -> float:
    return float("nan") if value is None else float(value)


def _nanmean(values: np.ndarray) -> float | None:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return None
    return float(finite.mean())


def _fmt(value: float | None) -> str:
    return "   n/a" if value is None else f"{value:7.3f}"


def _session_json(sess: SessionCompare) -> dict[str, Any]:
    return {
        "session_id": sess.session_id,
        "duration_s": sess.duration_s,
        "n_events": sess.n_events,
        "notes": sess.notes,
        "rows": [asdict(r) for r in sess.rows],
    }


def _agg_json(row: AggregateRow) -> dict[str, Any]:
    data = asdict(row)
    if row.auroc_ci is not None:
        data["auroc_ci"] = [row.auroc_ci[0], row.auroc_ci[1]]
    if row.auprc_ci is not None:
        data["auprc_ci"] = [row.auprc_ci[0], row.auprc_ci[1]]
    return data
