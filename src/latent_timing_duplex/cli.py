"""CLI for the Phase 0 skeleton: status, import check, synthetic harness."""

from __future__ import annotations

import argparse
import sys

from latent_timing_duplex import __phase__, __version__
from latent_timing_duplex.config import load_config
from latent_timing_duplex.data.synthetic import generate_synthetic_session
from latent_timing_duplex.eval.harness import format_score_table, score_session_bundle


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ltd",
        description=(
            "Phase 0 for latent timing control in full-duplex dialogue. "
            "Synthetic harness always; Moshi NLL / VAP need local weights. "
            "No downloads, no Spark re-run."
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
        "\n"
        "Not in this repo\n"
        "  - weights, corpora, caches\n"
        "  - Phase 1 JEPA predictor heads\n"
        "  - Phase 2 policy conditioning\n"
        "  - Spark job re-runs (see docs/SPARK.md)\n"
        "\n"
        "See README.md, PHASE0.md, and docs/SPARK.md."
    )
    return 0


def cmd_check() -> int:
    from latent_timing_duplex import baselines, data, eval as ev, extract, models
    from latent_timing_duplex.baselines.vap import VAPBaseline
    from latent_timing_duplex.data.candor import CandorPipeline
    from latent_timing_duplex.data.duplexchat import DuplexChatPipeline
    from latent_timing_duplex.extract.nll import FrozenNLLExtractor
    from latent_timing_duplex.models.bayling_duplex import BayLingDuplexWrapper
    from latent_timing_duplex.models.moshi import MoshiWrapper

    default = load_config("default.yaml")
    eval_cfg = load_config("eval.yaml")
    spark_cfg = load_config("spark_slice.yaml")
    _ = (
        baselines,
        data,
        ev,
        extract,
        models,
        MoshiWrapper,
        BayLingDuplexWrapper,
        VAPBaseline,
        CandorPipeline,
        DuplexChatPipeline,
        FrozenNLLExtractor,
        default,
        eval_cfg,
        spark_cfg,
    )
    print("imports ok")
    print(
        f"configs ok  (phase={default['project']['phase']}, "
        f"eval horizons={eval_cfg['eval']['horizons_s']}, "
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


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "status":
        return cmd_status()
    if args.command == "check":
        return cmd_check()
    if args.command == "reference":
        return cmd_reference()
    if args.command == "harness":
        return cmd_harness(duration=args.duration, seed=args.seed)
    parser.error(f"unknown command {args.command}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
