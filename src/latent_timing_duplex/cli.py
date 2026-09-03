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
        "See README.md, PHASE0.md, docs/PHASE1_PLAN.md, and docs/SPARK.md.\n"
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
    from latent_timing_duplex.phase1.horizons import PHASE1_HORIZONS_S

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
    if args.command == "phase1":
        return cmd_phase1(
            horizon_s=args.horizon,
            lambda_reg=args.lambda_reg,
            steps=args.steps,
            seed=args.seed,
        )
    parser.error(f"unknown command {args.command}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
