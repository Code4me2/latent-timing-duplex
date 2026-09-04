"""CLI: Phase 0 harness / reference plus Phase 1 synthetic head stub."""

from __future__ import annotations

import argparse
import sys

import numpy as np

from latent_timing_duplex import __phase__, __version__
from latent_timing_duplex.config import load_config
from latent_timing_duplex.data.synthetic import generate_synthetic_session
from latent_timing_duplex.eval.harness import format_score_table, score_session_bundle


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ltd",
        description=(
            "Phase 0 freeze + Phase 1 scaffolding for latent timing control. "
            "Synthetic harness always; Moshi NLL / VAP / hidden extract need "
            "local weights. No downloads, no Spark re-run, no Phase 2 FT."
        ),
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"latent-timing-duplex {__version__} (phase {__phase__})",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("status", help="What this skeleton implements vs. reserves")
    sub.add_parser("check", help="Import every public module and load configs")
    sub.add_parser(
        "reference",
        help="Print the Spark 10-clip DuplexChat EN reference numbers (do not re-run)",
    )
    harness = sub.add_parser(
        "harness",
        help="Run the eval harness on an in-memory synthetic stereo timeline",
    )
    harness.add_argument("--duration", type=float, default=120.0)
    harness.add_argument("--seed", type=int, default=0)
    phase1 = sub.add_parser(
        "phase1",
        help="Synthetic Phase 1 head step + surprise harness hook (no weights)",
    )
    phase1.add_argument("--horizon", type=float, default=0.08)
    phase1.add_argument("--lambda-reg", type=float, default=1.0)
    phase1.add_argument("--steps", type=int, default=8)
    phase1.add_argument("--seed", type=int, default=0)
    phase1_eval = sub.add_parser(
        "phase1-eval",
        help=(
            "Turn-event compare: surprise vs Moshi NLL vs VAP on mid-180 "
            "windows. Paths are caller-supplied; --synthetic needs no files."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Spark-61dd artifact layout (nothing opened unless passed):\n"
            "  /home/velvet/cs199-phase1-work/ablations/\n"
            "    SELECTION_LOCKED.json   # horizons.H_set or H_set: [1,12,62]\n"
            "    h12_lam0.01/checkpoint.pt   # mlp_state_dict / net.0.weight\n"
            "  hidden/moshi/<slice>/candor_<uuid>.pt  or  dc_<uuid>.pt\n"
            "  targets/user_chunk/<slice>/candor_<uuid>.pt\n"
            "  CANDOR extract*/transcription/*.csv   →  --transcripts-dir\n"
            "Phase 0 JSONLs that only have audio_nll / p_shift_mean /\n"
            "duration_sec are aggregate-only. RQ2 needs per-step series:\n"
            "  ltd phase1-export-series --print-schema\n"
            "Surprise-only dry run (not RQ2): add --surprise-only and omit\n"
            "the NLL/VAP JSONLs. Do not invent per-step series from means.\n"
        ),
    )
    phase1_eval.add_argument(
        "--synthetic",
        action="store_true",
        help="CPU demo on fake tensors (no Spark paths, no weights)",
    )
    phase1_eval.add_argument("--ablations-root", type=str, default=None)
    phase1_eval.add_argument("--selection-locked", type=str, default=None)
    phase1_eval.add_argument("--checkpoint", type=str, default=None)
    phase1_eval.add_argument("--hidden", type=str, default=None)
    phase1_eval.add_argument("--hidden-dir", type=str, default=None)
    phase1_eval.add_argument("--target", type=str, default=None)
    phase1_eval.add_argument("--target-dir", type=str, default=None)
    phase1_eval.add_argument("--surprise-jsonl", type=str, default=None)
    phase1_eval.add_argument("--nll-jsonl", type=str, default=None)
    phase1_eval.add_argument("--vap-jsonl", type=str, default=None)
    phase1_eval.add_argument("--labels", type=str, default=None)
    phase1_eval.add_argument(
        "--transcripts",
        type=str,
        default=None,
        help="JSON/JSONL turns, or a directory of CANDOR-style CSVs",
    )
    phase1_eval.add_argument(
        "--transcripts-dir",
        type=str,
        default=None,
        help="Directory of CANDOR extract*/transcription/*.csv (proxies)",
    )
    phase1_eval.add_argument("--vad", type=str, default=None)
    phase1_eval.add_argument(
        "--surprise-only",
        action="store_true",
        help=(
            "Score jepa:surprise without NLL/VAP (dry run). Primary RQ2 "
            "requires per-step --nll-jsonl and --vap-jsonl."
        ),
    )
    phase1_eval.add_argument("--session-id", action="append", default=None)
    phase1_eval.add_argument(
        "--horizon-frames",
        type=int,
        default=12,
        help="Predictor H in frames. Spark grid is 1, 12, 62 (default 12).",
    )
    phase1_eval.add_argument(
        "--lambda-reg",
        type=float,
        default=0.01,
        help="Which λ checkpoint to load (primary default 0.01).",
    )
    phase1_eval.add_argument(
        "--lambda-role",
        type=str,
        default="primary",
        choices=("primary", "reference", "other", "synthetic"),
    )
    phase1_eval.add_argument("--window-s", type=float, default=180.0)
    phase1_eval.add_argument(
        "--window-mode",
        type=str,
        default="mid",
        choices=("mid", "first"),
    )
    phase1_eval.add_argument(
        "--eval-horizons",
        type=str,
        default="0.16,0.32,0.50,1.00,2.00",
        help="Turn-event label horizons in seconds (comma-separated).",
    )
    phase1_eval.add_argument("--seed", type=int, default=20260903)
    phase1_eval.add_argument(
        "--bootstrap",
        type=int,
        default=10000,
        help="Session-level Efron bootstrap B (protocol default 10000).",
    )
    phase1_eval.add_argument("--output", type=str, default=None)
    export = sub.add_parser(
        "phase1-export-series",
        help=(
            "Print or write per-step Moshi NLL / VAP JSONL schema. "
            "Does not invent series from Phase 0 aggregates."
        ),
    )
    export.add_argument(
        "--print-schema",
        action="store_true",
        help="Print required per-step fields and the Spark Python snippet",
    )
    return parser


def cmd_status() -> int:
    print(
        f"latent-timing-duplex {__version__}  |  phase {__phase__} skeleton\n"
        "CCSF CS 199 Fall 2026  ·  Vel Moon Reichman  ·  advisor Indika Walimuni\n"
        "\n"
        "Implemented now\n"
        "  - package imports and YAML configs\n"
        "  - synthetic dual-channel timelines (metadata only, no audio files)\n"
        "  - eval harness: per-chunk signal → turn-event scores at multiple horizons\n"
        "  - extract.nll: Moshi LMModel.forward + delay-NaN mask (local weights)\n"
        "  - baselines.vap: frozen ErikEkstedt/VAP, CPU-ok (local checkpoint)\n"
        "  - Spark 10-clip DuplexChat EN reference numbers (ltd reference)\n"
        "  - phase1: frozen-backbone JEPA head stub (MLP, Gaussian regularizer,\n"
        "        surprise → harness). CPU numpy; no weights.\n"
        "  - phase1-eval: surprise vs Moshi NLL vs VAP on the same mid-180\n"
        "        windows (AUROC / AUPRC / PR). Paths from Spark; --synthetic\n"
        "        needs no files. See docs/EVAL_PROTOCOL_PHASE1.md.\n"
        "\n"
        "Local weights only (no download)\n"
        "  - models.moshi.MoshiWrapper              kyutai/moshiko-pytorch-bf16\n"
        "  - models.bayling_duplex.BayLingDuplexWrapper\n"
        "        BayLing-Models/BayLing-Duplex + zai-org/glm-4-voice-{tokenizer,decoder}\n"
        "  - baselines.vap.VAPBaseline              github.com/ErikEkstedt/VAP\n"
        "\n"
        "Reserved stubs\n"
        "  - data.candor.CandorPipeline             BetterUp request portal\n"
        "  - data.duplexchat.DuplexChatPipeline     reconstruct-from-podcasts\n"
        "  - phase1 Moshi hidden extract / Mimi targets (Spark caches)\n"
        "\n"
        "Not in this repo\n"
        "  - weights, corpora, caches\n"
        "  - Phase 2 policy conditioning or backbone fine-tuning\n"
        "  - Spark job re-runs (see docs/SPARK.md)\n"
        "\n"
        "See README.md, PHASE0.md, docs/PHASE1_PLAN.md,\n"
        "docs/EVAL_PROTOCOL_PHASE1.md, and docs/SPARK.md.\n"
        "Phase 0 numbers: docs/PHASE0_INTERIM_FINDINGS.md (do not invent)."
    )
    return 0


def cmd_check() -> int:
    from latent_timing_duplex import baselines, data, eval as ev, extract, models, phase1
    from latent_timing_duplex.baselines.vap import VAPBaseline
    from latent_timing_duplex.data.candor import CandorPipeline
    from latent_timing_duplex.data.duplexchat import DuplexChatPipeline
    from latent_timing_duplex.extract.nll import FrozenNLLExtractor
    from latent_timing_duplex.models.bayling_duplex import BayLingDuplexWrapper
    from latent_timing_duplex.models.moshi import MoshiWrapper
    from latent_timing_duplex.phase1.heads import MLPPredictor, count_mlp_parameters
    from latent_timing_duplex.phase1.horizons import PHASE1_HORIZONS_S, SPARK_TRAINED_HORIZON_FRAMES

    default = load_config("default.yaml")
    eval_cfg = load_config("eval.yaml")
    spark_cfg = load_config("spark_slice.yaml")
    phase1_cfg = load_config("phase1.yaml")
    _ = (
        baselines,
        data,
        ev,
        extract,
        models,
        phase1,
        MoshiWrapper,
        BayLingDuplexWrapper,
        VAPBaseline,
        CandorPipeline,
        DuplexChatPipeline,
        FrozenNLLExtractor,
        MLPPredictor,
        default,
        eval_cfg,
        spark_cfg,
        phase1_cfg,
    )
    n_params = count_mlp_parameters(
        phase1_cfg["phase1"]["head"]["hidden_dim"],
        phase1_cfg["phase1"]["head"]["embed_dim"],
        phase1_cfg["phase1"]["head"]["width"],
        phase1_cfg["phase1"]["head"]["n_layers"],
    )
    print("imports ok")
    print(
        f"configs ok  (phase0_default={default['project']['phase']}, "
        f"phase1={phase1_cfg['project']['phase']}, "
        f"eval horizons={eval_cfg['eval']['horizons_s']}, "
        f"phase1 horizons={list(PHASE1_HORIZONS_S)}, "
        f"spark_H={list(SPARK_TRAINED_HORIZON_FRAMES)}, "
        f"lambda_primary={phase1_cfg['phase1']['eval']['lambda_primary']}, "
        f"mlp_params={n_params}, "
        f"spark_slice={spark_cfg['slice']['id']})"
    )
    return 0


def cmd_reference() -> int:
    from latent_timing_duplex.spark_slice import (
        BAYLING_DURATION_WEIGHTED_NLL,
        BAYLING_DURATION_WEIGHTED_PPL,
        BAYLING_N_TOKENS,
        BAYLING_UNWEIGHTED_NLL,
        BAYLING_UNWEIGHTED_PPL,
        BAYLING_VOCAB_SIZE,
        DELAY_NAN_HYPOTHESIS,
        MOSHI_DURATION_WEIGHTED_AUDIO_NLL,
        MOSHI_DURATION_WEIGHTED_TEXT_NLL,
        MOSHI_UNWEIGHTED_AUDIO_NLL,
        MOSHI_UNWEIGHTED_TEXT_NLL,
        SLICE_DURATION_S,
        SLICE_SOURCE,
        VAP_DURATION_WEIGHTED_P_FUTURE,
        VAP_DURATION_WEIGHTED_P_NOW,
        VAP_DURATION_WEIGHTED_P_SHIFT,
    )

    print(
        f"Spark reference  |  {SLICE_SOURCE}  |  {SLICE_DURATION_S}s\n"
        "Do not re-run Spark jobs. These are measured numbers, not estimates.\n"
        "\n"
        "Moshi  LMModel.forward  nan-safe delay mask  "
        "NO_CUDA_GRAPH=1 NO_TORCH_COMPILE=1\n"
        f"  unweighted         audio {MOSHI_UNWEIGHTED_AUDIO_NLL:.6f}  "
        f"text {MOSHI_UNWEIGHTED_TEXT_NLL:.6f}\n"
        f"  duration-weighted  audio {MOSHI_DURATION_WEIGHTED_AUDIO_NLL:.6f}  "
        f"text {MOSHI_DURATION_WEIGHTED_TEXT_NLL:.6f}\n"
        "\n"
        f"BayLing-Duplex  vocab {BAYLING_VOCAB_SIZE}  tokens {BAYLING_N_TOKENS}  "
        "(not comparable to Moshi codebook NLL)\n"
        f"  unweighted         nll {BAYLING_UNWEIGHTED_NLL:.6f}  "
        f"ppl {BAYLING_UNWEIGHTED_PPL}\n"
        f"  duration-weighted  nll {BAYLING_DURATION_WEIGHTED_NLL:.6f}  "
        f"ppl {BAYLING_DURATION_WEIGHTED_PPL}\n"
        "\n"
        "VAP  ErikEkstedt/VAP  CPU  LEFT=user  p(shift)=1-p_now\n"
        f"  duration-weighted  p_now {VAP_DURATION_WEIGHTED_P_NOW:.6f}  "
        f"p_future {VAP_DURATION_WEIGHTED_P_FUTURE:.6f}  "
        f"p(shift) {VAP_DURATION_WEIGHTED_P_SHIFT:.6f}\n"
        "\n"
        f"{DELAY_NAN_HYPOTHESIS}\n"
        "See docs/SPARK.md."
    )
    return 0


def cmd_harness(duration: float, seed: int) -> int:
    bundle = generate_synthetic_session(duration_s=duration, seed=seed)
    named = score_session_bundle(
        bundle.session,
        {"synthetic_salience": bundle.predictive, "random": bundle.random},
    )
    print(
        f"synthetic session {bundle.session.session_id}  "
        f"duration={bundle.session.duration_s:.1f}s  "
        f"events={len(bundle.session.events)}"
    )
    print(format_score_table(named))
    print(
        "\nGate reminder: a later real signal (frozen NLL, VAP, JEPA surprise) "
        "plugs into the same harness. Weak or null predictiveness is still a result."
    )
    return 0


def cmd_phase1(horizon_s: float, lambda_reg: float, steps: int, seed: int) -> int:
    """CPU-only Phase 1 demo: fake hidden/targets → head SGD → harness."""
    from latent_timing_duplex.phase1.heads import MLPPredictor
    from latent_timing_duplex.phase1.hidden import FakeHiddenStateExtractor
    from latent_timing_duplex.phase1.surprise import (
        score_surprise_bundle,
        surprise_from_sequences,
    )
    from latent_timing_duplex.phase1.train import TrainConfig, train_loop
    from latent_timing_duplex.phase1.windows import crop_session

    bundle = generate_synthetic_session(duration_s=16.0, seed=seed)
    cropped = crop_session(bundle.session, window_s=8.0, mode="mid")
    hidden = FakeHiddenStateExtractor(hidden_dim=16, seed=seed).extract(cropped)
    rng = np.random.default_rng(seed + 1)
    # Frozen linear target of the *future* hidden state — head can fit this.
    proj = rng.normal(size=(16, 8))
    target = hidden @ proj
    cfg = TrainConfig(
        horizon_s=horizon_s,
        lambda_reg=lambda_reg,
        lr=5e-2,
        batch_size=16,
        max_steps=steps,
        seed=seed,
        freeze_backbone=True,
        device="cpu",
    )
    head = MLPPredictor(hidden_dim=16, embed_dim=8, width=16, n_layers=2, seed=seed)
    result = train_loop(hidden, target, config=cfg, head=head)
    first = result.steps[0].mse if result.steps else float("nan")
    last = result.steps[-1].mse if result.steps else float("nan")
    surprise = surprise_from_sequences(
        hidden,
        target,
        head.forward,
        horizon_s=horizon_s,
    )
    from latent_timing_duplex.types import iter_chunks

    random_chunks = iter_chunks(
        duration_s=len(surprise) * 0.08,
        chunk_duration_s=0.08,
        values=rng.normal(size=len(surprise)).tolist(),
        name="random",
    )
    named = score_surprise_bundle(
        cropped,
        {"jepa:surprise": surprise, "random": random_chunks},
        horizons_s=(0.50, 1.00),
    )
    print(
        f"phase1 synthetic  horizon={horizon_s}s  lambda={lambda_reg}  "
        f"steps={steps}  freeze_backbone=True\n"
        f"head MLP 16→16→16→8  mse {first:.4f} → {last:.4f}\n"
        "Equal-length mid-8s crop (protocol demo; Spark uses W=180).\n"
        "Phase 0 numbers: docs/PHASE0_INTERIM_FINDINGS.md (not invented here)."
    )
    print(format_score_table(named))
    print(
        "\nBackbone stays frozen. Phase 2 fine-tune is out of scope. "
        "See docs/PHASE1_PLAN.md."
    )
    return 0


def cmd_phase1_eval(args: argparse.Namespace) -> int:
    """Compare surprise / NLL / VAP. ``--synthetic`` is the CI / smoke path."""
    from pathlib import Path

    from latent_timing_duplex.exceptions import Phase1EvalInputMissing
    from latent_timing_duplex.phase1.compare import (
        EvalConfig,
        EvalPaths,
        format_compare_report,
        run_synthetic_compare,
        run_turn_event_eval,
        write_report,
    )

    horizons = tuple(float(x) for x in str(args.eval_horizons).split(",") if x.strip())
    if args.synthetic:
        report = run_synthetic_compare(
            seed=int(args.seed),
            horizon_frames=int(args.horizon_frames),
            lambda_reg=float(args.lambda_reg),
            bootstrap_b=min(int(args.bootstrap), 64),
        )
        print(format_compare_report(report))
        if args.output:
            write_report(report, args.output)
            print(f"\nwrote {args.output}")
        print(
            "\nSpark real run needs --ablations-root (or --checkpoint + "
            "--hidden-dir/--target-dir or --surprise-jsonl), --nll-jsonl, "
            "--vap-jsonl, and --labels or --transcripts/--vad. "
            "See docs/EVAL_PROTOCOL_PHASE1.md and "
            "docs/PHASE1_RQ2_INTERIM.md. Do not invent findings."
        )
        return 0

    def _p(value: str | None) -> Path | None:
        return Path(value) if value else None

    paths = EvalPaths(
        ablations_root=_p(args.ablations_root),
        selection_locked=_p(args.selection_locked),
        checkpoint=_p(args.checkpoint),
        hidden=_p(args.hidden),
        hidden_dir=_p(args.hidden_dir),
        target=_p(args.target),
        target_dir=_p(args.target_dir),
        surprise_jsonl=_p(args.surprise_jsonl),
        nll_jsonl=_p(args.nll_jsonl),
        vap_jsonl=_p(args.vap_jsonl),
        labels=_p(args.labels),
        transcripts=_p(args.transcripts),
        transcripts_dir=_p(getattr(args, "transcripts_dir", None)),
        vad=_p(args.vad),
        output=_p(args.output),
        surprise_only=bool(getattr(args, "surprise_only", False)),
    )
    cfg = EvalConfig(
        window_s=float(args.window_s),
        window_mode=args.window_mode,
        eval_horizons_s=horizons,
        seed=int(args.seed),
        bootstrap_b=int(args.bootstrap),
        horizon_frames=int(args.horizon_frames),
        lambda_reg=float(args.lambda_reg),
        lambda_role=str(args.lambda_role),
        surprise_only=bool(getattr(args, "surprise_only", False)),
    )
    try:
        report = run_turn_event_eval(
            paths,
            cfg,
            session_ids=args.session_id,
        )
    except Phase1EvalInputMissing as exc:
        print(f"phase1-eval: missing Spark input: {exc}", file=sys.stderr)
        print(
            "Required on spark-61dd (not in CI):\n"
            "  --ablations-root /home/velvet/cs199-phase1-work/ablations\n"
            "      (h1_lam0.01/, h12_lam0.01/, h62_lam0.01/ + optional "
            "SELECTION_LOCKED.json)\n"
            "  --hidden-dir and --target-dir  (candor_<uuid>.pt / dc_<uuid>.pt "
            "or .npz)  OR --surprise-jsonl\n"
            "  --nll-jsonl   per-step Moshi NLL (not audio_nll aggregates)\n"
            "  --vap-jsonl   per-step VAP (not p_shift_mean aggregates)\n"
            "  --labels | --transcripts-dir   gold JSONL or CANDOR CSVs\n"
            "  --surprise-only   omit NLL/VAP (dry run only)\n"
            "See docs/EVAL_PROTOCOL_PHASE1.md. "
            "ltd phase1-export-series --print-schema",
            file=sys.stderr,
        )
        return 2
    print(format_compare_report(report))
    if args.output:
        write_report(report, args.output)
        print(f"\nwrote {args.output}")
    print(
        "\nMetrics only. Do not claim surprise beats NLL or VAP from this table. "
        "See docs/EVAL_PROTOCOL_PHASE1.md."
    )
    return 0


def cmd_phase1_export_series(args: argparse.Namespace) -> int:
    from latent_timing_duplex.phase1.export_series import schema_text

    print(schema_text())
    if not args.print_schema:
        print(
            "This command does not run Moshi/VAP without local weights. "
            "Use --print-schema (always printed) and the Python snippet on Spark."
        )
    return 0


def main(argv: list[str] | None = None) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)
    if raw[:2] == ["phase1", "eval"]:
        raw = ["phase1-eval", *raw[2:]]
    if raw[:2] == ["phase1", "export-series"]:
        raw = ["phase1-export-series", *raw[2:]]
    parser = build_parser()
    args = parser.parse_args(raw)
    if args.command == "status":
        return cmd_status()
    if args.command == "check":
        return cmd_check()
    if args.command == "reference":
        return cmd_reference()
    if args.command == "harness":
        return cmd_harness(duration=args.duration, seed=args.seed)
    if args.command == "phase1":
        return cmd_phase1(
            horizon_s=args.horizon,
            lambda_reg=args.lambda_reg,
            steps=args.steps,
            seed=args.seed,
        )
    if args.command == "phase1-eval":
        return cmd_phase1_eval(args)
    if args.command == "phase1-export-series":
        return cmd_phase1_export_series(args)
    parser.error(f"unknown command {args.command}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
