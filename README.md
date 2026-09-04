# Latent Predictive Objectives for Timing Control in Full-Duplex Dialogue Models

**CCSF CS 199 independent study · Fall 2026 · 2–3 units**

| | |
|---|---|
| Student | Vel Moon Reichman (Velvet) |
| Instructor | Indika Walimuni |
| Owner | [code4me2](https://github.com/code4me2) |
| License | Apache-2.0 |

Full-duplex spoken dialogue models (Moshi, SyncLLM, BayLing-Duplex) bury *speak* vs. *listen* inside next-token sampling. This project tests a JEPA-style latent predictor of the next user-audio chunk, and treats prediction error (“surprise”) as an explicit salience signal for the dialogue policy.

**This repository is Phase 0 (frozen) plus Phase 1 scaffolding and the Phase 1 turn-event scorer.** It does not download weights or corpora, does not implement Phase 2 policy conditioning, and does not re-run Spark jobs. Phase 0 numbers live in [docs/PHASE0_INTERIM_FINDINGS.md](docs/PHASE0_INTERIM_FINDINGS.md) — do not invent them. A 10-clip DuplexChat EN slice is also recorded as reference constants (`ltd reference`, [docs/SPARK.md](docs/SPARK.md)).

Whether the *implicit* next-token signal is already predictive of turn events is a reportable Phase 0 result, including a negative one.

## Three gated phases

| Phase | Scope | In this repo now |
|---|---|---|
| **0** (weeks 1–5) | Inference wrappers, VAP baseline, 200–500 h working subset, eval harness, frozen user-channel NLL | **Frozen.** Synthetic harness, Moshi delay-NaN NLL, CPU VAP, Spark 10-clip reference. Findings: [docs/PHASE0_INTERIM_FINDINGS.md](docs/PHASE0_INTERIM_FINDINGS.md). Protocol: [docs/EVAL_PROTOCOL_PHASE0.md](docs/EVAL_PROTOCOL_PHASE0.md). |
| **1** | JEPA-style latent predictor on the next user-audio chunk; surprise as salience | **Scaffolding + turn-event scorer.** Plan, CPU head/loss/dataset, and `ltd phase1-eval` (surprise vs Moshi NLL vs VAP on mid-180). Spark RQ2 interim (CANDOR-only proxy first; evening-PT DC channel-energy VAD addendum; pooled EVENT_HORIZON_GRID CIs, no overclaim): [docs/PHASE1_RQ2_INTERIM.md](docs/PHASE1_RQ2_INTERIM.md). Silence/collapse diagnostic (null; 0/57; not a timing win): [docs/PHASE1_SILENCE_COLLAPSE.md](docs/PHASE1_SILENCE_COLLAPSE.md). No Phase 2. See [docs/PHASE1_PLAN.md](docs/PHASE1_PLAN.md) and [docs/EVAL_PROTOCOL_PHASE1.md](docs/EVAL_PROTOCOL_PHASE1.md). |
| **2** | Condition the dialogue policy on that salience signal | Out of scope. `Phase2OutOfScope` if a caller tries to unfreeze a backbone. |

Phase 0 is frozen (equal-length windows; prefer fixed-W). Phase 1 trains **small heads only** on frozen Moshi hidden states. Velvet approved starting Phase 1 after that freeze.

## Current scope (Phase 0 freeze + Phase 1 scaffolding)

Implemented:

- Installable Python package `latent-timing-duplex` (`import latent_timing_duplex`)
- Eval harness: any per-chunk scalar → turn-event scores (turn shifts, backchannels, barge-ins) at multiple horizons
- In-memory synthetic dual-channel *timelines* so the harness can be run without audio
- Moshi frozen user-channel NLL: `LMModel.forward` + delay-NaN mask (`extract/nll.py`); `load(local_dir=...)` only
- Frozen ErikEkstedt/VAP baseline, CPU-ok (`baselines/vap.py`); `load(local_checkpoint=...)` only
- Spark 10-clip DuplexChat EN reference numbers (`ltd reference`) — do not re-measure
- Phase 1 skeleton (`latent_timing_duplex.phase1`): chunked stereo dataset (LEFT=user), frozen Moshi hidden-state interface, target-embedding interface, small MLP / tiny Transformer head, MSE + isotropic-Gaussian regularizer, CPU train-loop stub, surprise → existing harness. `ltd phase1` runs the synthetic train path. `ltd phase1-eval` scores surprise vs Moshi NLL vs VAP on the same mid-180 windows (AUROC / AUPRC / PR).

Not implemented (intentionally):

- Downloading Moshi / BayLing-Duplex / GLM-4-Voice / VAP weights (local paths only)
- Reconstructing DuplexChat from podcasts
- Accessing or converting CANDOR
- Re-running Phase 0 Spark jobs
- Live Moshi hidden-state extract (reserved for `spark-61dd` caches)
- Phase 2 policy conditioning or backbone fine-tuning

## Model and data ids (honest, not installed)

These are the public artifacts the stubs point at. **Do not treat any of them as present on disk after `pip install`.**

| Artifact | Id / URL | Notes |
|---|---|---|
| Moshi paper | [arXiv:2410.00037](https://arxiv.org/abs/2410.00037) | Défossez et al., 2024 |
| Moshi code | [github.com/kyutai-labs/moshi](https://github.com/kyutai-labs/moshi) | Official inference |
| Moshi weights | `kyutai/moshiko-pytorch-bf16`, `kyutai/moshika-pytorch-bf16` | Public Hugging Face; CC-BY 4.0 |
| BayLing-Duplex paper | [arXiv:2606.14528](https://arxiv.org/abs/2606.14528) | Fang, Guo, Feng, 2026 |
| BayLing-Duplex code | [github.com/BayLing-Models/BayLing-Duplex](https://github.com/BayLing-Models/BayLing-Duplex) | Official inference |
| BayLing-Duplex weights | `BayLing-Models/BayLing-Duplex` | **Public.** 4 safetensors shards (`model-0000k-of-00004`), ~19.1 GB. Hugging Face card “516k params” is a display bug; the index is ~9.54B BF16 parameters. |
| Speech tokenizer | `zai-org/glm-4-voice-tokenizer` | Required by BayLing-Duplex |
| Speech decoder | `zai-org/glm-4-voice-decoder` | Required by BayLing-Duplex |
| VAP | [arXiv:2205.09812](https://arxiv.org/abs/2205.09812), [ErikEkstedt/VAP](https://github.com/ErikEkstedt/VAP) | Stereo turn-taking baseline |
| CANDOR | [arXiv:2203.00674](https://arxiv.org/abs/2203.00674), [BetterUp request portal](https://betterup-data-requests.herokuapp.com/) | Not public dump; request access |
| DuplexChat | [arXiv:2607.04941](https://arxiv.org/abs/2607.04941), `sarulab-speech/DuplexChat` | **Manifest-only.** Reconstruct locally from podcasts via [sarulab-speech/DuplexChat](https://github.com/sarulab-speech/DuplexChat) |

This repo will not invent local install paths for unpublished checkpoints.

## Hardware note

Frozen-model inference is *planned* on two NVIDIA DGX Spark systems (128 GB unified memory each). Later work is *planned* on 2× RTX 5090. **Those machines are not attached to this repository** and are not required to install or test the skeleton. The skeleton runs on CPU.

## Install the skeleton

Python 3.10+ (3.12 is fine). From the repo root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Sanity checks (`ltd` is also available as `python -m latent_timing_duplex`):

```bash
ltd status
ltd check
ltd reference
ltd harness
ltd phase1
ltd phase1-eval --synthetic
pytest
```

`ltd harness` scores a synthetic salience signal (and a random control) for turn shifts, backchannels, and barge-ins at 0.16–2.0 s horizons. That path needs no weights.

`ltd phase1` trains a tiny CPU head on fake hidden states, wraps prediction error as surprise, and scores it on the same harness (equal-length mid-window demo). No GPU and no checkpoints.

`ltd phase1-eval --synthetic` runs the full compare table (surprise / NLL / VAP / random) on fake mid-window tensors. That is a CLI smoke test, not a finding.

`ltd reference` prints the measured Spark 10-clip numbers (Moshi NLL, BayLing-Duplex token NLL, VAP CPU). Do not re-run Spark.

Moshi / VAP inference still requires weights **you** place on disk. Spark install notes (aarch64 cu130, no flash-attn, sphn sdist, tiktoken/torchaudio extras, no GB10 kernel fork) are in [docs/SPARK.md](docs/SPARK.md).

## Package layout

```
src/latent_timing_duplex/
  data/          CANDOR + DuplexChat pipelines (license notes, no data files)
  models/        Moshi + BayLing-Duplex wrappers (stubs + public ids)
  baselines/     frozen VAP (CPU, local checkpoint)
  eval/          harness: per-chunk signal in, turn-event scores out
  extract/       Moshi LMModel.forward + delay-NaN user-channel NLL
  phase1/        JEPA head + turn-event compare (dataset, hidden, targets, loss,
                 train, surprise, labels, series, checkpoint, compare)
configs/         YAML defaults + spark_slice.yaml + phase1.yaml
docs/            SPARK.md, Phase 0 freeze, PHASE1_PLAN.md, EVAL_PROTOCOL_PHASE1.md,
                 PHASE1_RQ2_INTERIM.md, PHASE1_SILENCE_COLLAPSE.md
tests/           harness, delay-NaN NLL, VAP pooling, Spark reference, Phase 1 shapes/losses/compare
```

Local `./data`, `./weights`, and `./cache` are gitignored. Create them yourself after you have access.

## Phase 0 → Phase 1

Phase 0 asked whether frozen user-channel NLL (and VAP) already predict turn events. That work is **frozen**: equal-length windows, documented slices, no invented digits. Read [docs/PHASE0_INTERIM_FINDINGS.md](docs/PHASE0_INTERIM_FINDINGS.md) and [docs/EVAL_PROTOCOL_PHASE0.md](docs/EVAL_PROTOCOL_PHASE0.md) before comparing any new signal to those baselines. Prefer fixed windows (mid-180 primary).

Phase 1 (this repo now) precomputes frozen Moshi hidden states and user-chunk embeddings, then trains a **small** predictor head (single-digit millions of params) with a JEPA-style isotropic-Gaussian regularizer. Surprise is the per-chunk prediction error and plugs into the same harness. Moshi / BayLing stay frozen. Operational steps, Spark paths (`/home/velvet/cs199-*`), ablations, and success criteria: [docs/PHASE1_PLAN.md](docs/PHASE1_PLAN.md). Turn-event protocol (mid-180, seed `20260903`, H∈{1,12,62}, λ=0.01 primary / λ=0 reference): [docs/EVAL_PROTOCOL_PHASE1.md](docs/EVAL_PROTOCOL_PHASE1.md). Spark primary numbers and caveats (CANDOR-only proxy n=44 first; evening-PT pooled n≈56 channel-energy VAD addendum): [docs/PHASE1_RQ2_INTERIM.md](docs/PHASE1_RQ2_INTERIM.md). Silence/collapse on 57 mid-180 sessions (channel-energy masks; criterion not met, 0/57): [docs/PHASE1_SILENCE_COLLAPSE.md](docs/PHASE1_SILENCE_COLLAPSE.md). Do not upgrade either note into a gold-label, Phase 2, or timing-win claim.

### Spark: what `ltd phase1-eval` still needs at runtime

This repo **replaces** the ad-hoc `MISSING_HARNESS.md` gap. The scorer and flags live here; Spark must still pass local files (nothing is downloaded, and CI does not require these paths):

| Flag | Typical spark-61dd location | What it is |
|---|---|---|
| `--ablations-root` | `/home/velvet/cs199-phase1-work/ablations` | `h{1,12,62}_lam{0,0.01}/` + optional `SELECTION_LOCKED.json` |
| `--hidden-dir` / `--target-dir` | `.../hidden/moshi/<slice>/`, `.../targets/user_chunk/<slice>/` | mid-180 `[T=2250, D=4096]` as `candor_<uuid>.pt` / `dc_<uuid>.pt` or `.npz` |
| *or* `--surprise-jsonl` | under `cs199-phase1-work/` | precomputed surprise series (skips the head) |
| `--nll-jsonl` | Phase 0 work trees | **per-step** Moshi NLL (`audio_nll_per_step`). Aggregates (`audio_nll`, `duration_sec`) are rejected |
| `--vap-jsonl` | Phase 0 VAP JSONL | **per-step** `p_shift` / `p_now`. `p_shift_mean` is rejected |
| `--labels` *or* `--transcripts-dir` / `--vad` | gold JSONL, or CANDOR `extract*/transcription/*.csv` | gold events, else speaker-change / onset proxies |

Primary: `--horizon-frames 12 --lambda-reg 0.01 --window-mode mid --window-s 180 --seed 20260903 --bootstrap 10000`. Repeat H∈{1,62} and λ=0. Alias: `ltd phase1 eval …`. Do not claim wins from the table.

Phase 2 (policy conditioning / backbone fine-tune) is **not** started. `freeze_backbone=False` raises `Phase2OutOfScope`.

## Citation

If you use this skeleton, cite the independent-study title and the upstream papers in the table above.

```
Vel Moon Reichman. Latent Predictive Objectives for Timing Control
in Full-Duplex Dialogue Models. CCSF CS 199, Fall 2026.
```
