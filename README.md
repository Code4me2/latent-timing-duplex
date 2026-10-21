# Latent Predictive Objectives for Timing Control in Full-Duplex Dialogue Models

**CCSF CS 199 independent study · Fall 2026 · 2–3 units**

| | |
|---|---|
| Student | Vel Moon Reichman (Velvet) |
| Instructor | Indika Walimuni |
| Owner | [code4me2](https://github.com/code4me2) |
| License | Apache-2.0 |

Full-duplex spoken dialogue models (Moshi, SyncLLM, BayLing-Duplex) bury *speak* vs. *listen* inside next-token sampling. This project tests whether predictive latent error (“surprise”) exposes turn timing and, separately, whether a predicted latent can steer generation. The planned user-audio target and the realized future-transformer-state target are kept distinct below.

**Current index:** [docs/CURRENT_STATUS.md](docs/CURRENT_STATUS.md) (evidence through 2026-09-25). The forward plan is [docs/RESEARCH_ROADMAP.md](docs/RESEARCH_ROADMAP.md): a causal silence-state/hazard sidecar followed, only after offline gates, by characterized valid Mimi silence-code generation plus text padding. The R1 data contract is [docs/SILENCE_GRAFT.md](docs/SILENCE_GRAFT.md), with the collection manual in [docs/DATA_LABELING_PROTOCOL.md](docs/DATA_LABELING_PROTOCOL.md). **R1 is scaffolded and its gate remains open until a real pilot agreement receipt passes; test evaluation also requires an analysis-lock receipt.** Phase 0 is frozen; Phase 1's scorer remains the supported package surface. Later C0/C1/C2 latent-graft runs are preserved as **exploratory receipts**, not a supported Phase 2 API or an efficacy result: [docs/PHASE2_EXPLORATORY.md](docs/PHASE2_EXPLORATORY.md).

This repository does not download weights or reconstruct research corpora and does not re-run Spark jobs. A separate, permission-confirmed podcast acquisition CLI can download local engineering-demo media (see below). Phase 0 numbers live in [docs/PHASE0_INTERIM_FINDINGS.md](docs/PHASE0_INTERIM_FINDINGS.md) — do not invent them. A 10-clip DuplexChat EN slice is also recorded as reference constants (`ltd reference`, [docs/SPARK.md](docs/SPARK.md)).

Whether the *implicit* next-token signal is already predictive of turn events is a reportable Phase 0 result, including a negative one.

## Three gated phases

| Phase | Scope | In this repo now |
|---|---|---|
| **0** (weeks 1–5) | Inference wrappers, VAP baseline, 200–500 h working subset, eval harness, frozen user-channel NLL | **Frozen.** Synthetic harness, Moshi delay-NaN NLL, CPU VAP, Spark 10-clip reference. Findings: [docs/PHASE0_INTERIM_FINDINGS.md](docs/PHASE0_INTERIM_FINDINGS.md). Protocol: [docs/EVAL_PROTOCOL_PHASE0.md](docs/EVAL_PROTOCOL_PHASE0.md). |
| **1** | Predict a future latent; use prediction error as salience | **Scaffolding + turn-event scorer + Spark proxy runs.** The realized Spark target was a future Moshi transformer-state mean, not an independent user-audio embedding. The silence/collapse diagnostic was null (0/57; not a timing win). See [docs/CURRENT_STATUS.md](docs/CURRENT_STATUS.md), [docs/PHASE1_RQ2_INTERIM.md](docs/PHASE1_RQ2_INTERIM.md), [docs/PHASE1_SILENCE_COLLAPSE.md](docs/PHASE1_SILENCE_COLLAPSE.md), [docs/PHASE1_PLAN.md](docs/PHASE1_PLAN.md), and [docs/EVAL_PROTOCOL_PHASE1.md](docs/EVAL_PROTOCOL_PHASE1.md). |
| **2** | Condition generation on the predicted latent | **Exploratory receipts only.** C0/C1/C2 ran out-of-tree; no efficacy gate passed. The package still raises `Phase2OutOfScope` if a caller tries to unfreeze a backbone. |

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
- Supported Phase 2 policy conditioning or backbone fine-tuning (historical exploratory scripts are receipts only)

## Podcast intake demo

The metadata-only audit, repeatable MP4/caption downloader, and next labeling
steps are documented in [docs/PODCAST_INTAKE.md](docs/PODCAST_INTAKE.md). The
acquisition source of truth is now the standalone
[Code4me2/media-intake](https://github.com/Code4me2/media-intake) repository
(initially private). [`packages/media-intake`](packages/media-intake/README.md)
is a commit-pinned vendored snapshot, so public installs require no upstream
credentials. See [upstream/update policy](docs/MEDIA_INTAKE_UPSTREAM.md).
The project modules remain thin policy/compatibility wrappers. The three episodes in
`configs/podcast_sources.json` have user-confirmed acquisition permissions but
remain engineering-demo inputs, not approved pilot data.

```bash
.venv/bin/python -m latent_timing_duplex.data.podcast_download \
  --sources configs/podcast_sources.json \
  --output-dir "$HOME/Videos/latent-timing-duplex"
```

Requires `yt-dlp`, `ffmpeg`, and `ffprobe`. Downloads original-audio MP4s,
available timestamped captions, and publisher/date metadata outside Git;
records checksums and per-source failure statuses. Reuses verified media on
reruns while independently retrying caption acquisition.

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

Python 3.10+ (3.12 is fine). The supported workspace setup uses the committed,
generated `uv.lock`. The pinned acquisition snapshot remains a workspace member
for integration tests; acquisition development lives in its standalone repository:

```bash
uv sync --locked --extra dev
source .venv/bin/activate
```

A normal `pip install .` or parent wheel installation is self-contained: the
parent wheel bundles the dependency-light `media_intake` import package rather
than declaring an unavailable index dependency. The standalone distribution has
its own lock and install instructions in
[`packages/media-intake/README.md`](packages/media-intake/README.md).

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
docs/            CURRENT_STATUS.md, frozen protocols/findings, Phase 1 null diagnostics,
                 exploratory Phase 2 note
experiments/      exact exploratory script receipts; no weights/audio/checkpoints
tests/           harness, delay-NaN NLL, VAP pooling, Spark reference, Phase 1 shapes/losses/compare
```

Local `./data`, `./weights`, and `./cache` are gitignored. Create them yourself after you have access.

## Phase 0 → Phase 1

Phase 0 asked whether frozen user-channel NLL (and VAP) already predict turn events. That work is **frozen**: equal-length windows, documented slices, no invented digits. Read [docs/PHASE0_INTERIM_FINDINGS.md](docs/PHASE0_INTERIM_FINDINGS.md) and [docs/EVAL_PROTOCOL_PHASE0.md](docs/EVAL_PROTOCOL_PHASE0.md) before comparing any new signal to those baselines. Prefer fixed windows (mid-180 primary).

The Phase 1 scaffold supports frozen hidden states plus a target-embedding interface. The realized Spark ablation instead trained on future Moshi transformer-state means; this distinction is locked in [docs/CURRENT_STATUS.md](docs/CURRENT_STATUS.md). Surprise is the per-chunk prediction error and plugs into the same harness. Moshi / BayLing stayed frozen. Operational steps and historical intent: [docs/PHASE1_PLAN.md](docs/PHASE1_PLAN.md). Turn-event protocol (mid-180, seed `20260903`, explicit Spark H∈{1,12,62}, λ=0.01 primary / λ=0 reference): [docs/EVAL_PROTOCOL_PHASE1.md](docs/EVAL_PROTOCOL_PHASE1.md). The first Spark note remains [docs/PHASE1_RQ2_INTERIM.md](docs/PHASE1_RQ2_INTERIM.md) (CANDOR first, then the historical n≈56 channel-energy VAD proxy pool and pooled `EVENT_HORIZON_GRID` CIs); the expanded proxy result and limitations are indexed in CURRENT_STATUS. Silence/collapse on 57 mid-180 sessions was null (criterion not met, 0/57): [docs/PHASE1_SILENCE_COLLAPSE.md](docs/PHASE1_SILENCE_COLLAPSE.md). Do not upgrade any of these into a gold-label, control, Phase 2, or timing-win claim.

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

The supported package still has no Phase 2 training entry point: `freeze_backbone=False` raises `Phase2OutOfScope`. Exploratory 2026-09-25 latent-graft runs are documented in [docs/PHASE2_EXPLORATORY.md](docs/PHASE2_EXPLORATORY.md), with exact script snapshots under [`experiments/phase2/`](experiments/phase2/). Their execution gates are not efficacy gates.

## Citation

If you use this skeleton, cite the independent-study title and the upstream papers in the table above.

```
Vel Moon Reichman. Latent Predictive Objectives for Timing Control
in Full-Duplex Dialogue Models. CCSF CS 199, Fall 2026.
```
