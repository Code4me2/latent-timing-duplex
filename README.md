# Latent Predictive Objectives for Timing Control in Full-Duplex Dialogue Models

**CCSF CS 199 independent study · Fall 2026 · 2–3 units**

| | |
|---|---|
| Student | Vel Moon Reichman (Velvet) |
| Instructor | Indika Walimuni |
| Owner | [code4me2](https://github.com/code4me2) |
| License | Apache-2.0 |

Full-duplex spoken dialogue models (Moshi, SyncLLM, BayLing-Duplex) bury *speak* vs. *listen* inside next-token sampling. This project tests a JEPA-style latent predictor of the next user-audio chunk, and treats prediction error (“surprise”) as an explicit salience signal for the dialogue policy.

**This repository is Phase 0.** It does not download weights or corpora, does not implement Phase 1 predictor heads or Phase 2 policy conditioning, and does not re-run Spark jobs. A 10-clip DuplexChat EN slice was already measured on Spark; those numbers are reference constants (`ltd reference`, [docs/SPARK.md](docs/SPARK.md)).

Whether the *implicit* next-token signal is already predictive of turn events is a reportable Phase 0 result, including a negative one.

## Three gated phases

| Phase | Scope | In this repo now |
|---|---|---|
| **0** (weeks 1–5) | Inference wrappers, VAP baseline, 200–500 h working subset, eval harness, frozen user-channel NLL | Synthetic harness, Moshi delay-NaN NLL, CPU VAP, Spark 10-clip reference. See [PHASE0.md](PHASE0.md) and [docs/SPARK.md](docs/SPARK.md). |
| **1** | JEPA-style latent predictor on the next user-audio chunk; surprise as salience | One-line pointer only. Not implemented. |
| **2** | Condition the dialogue policy on that salience signal | One-line pointer only. Not implemented. |

Phase 1/2 work starts only after the Phase 0 gate: the harness exists, a frozen implicit signal can be scored, and a VAP baseline is on the same metric.

## Current scope (Phase 0 only)

Implemented:

- Installable Python package `latent-timing-duplex` (`import latent_timing_duplex`)
- Eval harness: any per-chunk scalar → turn-event scores (turn shifts, backchannels, barge-ins) at multiple horizons
- In-memory synthetic dual-channel *timelines* so the harness can be run without audio
- Moshi frozen user-channel NLL: `LMModel.forward` + delay-NaN mask (`extract/nll.py`); `load(local_dir=...)` only
- Frozen ErikEkstedt/VAP baseline, CPU-ok (`baselines/vap.py`); `load(local_checkpoint=...)` only
- Spark 10-clip DuplexChat EN reference numbers (`ltd reference`) — do not re-measure

Not implemented (intentionally):

- Downloading Moshi / BayLing-Duplex / GLM-4-Voice / VAP weights (local paths only)
- Reconstructing DuplexChat from podcasts
- Accessing or converting CANDOR
- Re-running the Spark 10-clip job
- Training loops, notebooks-as-product, Phase 1 heads, Phase 2 policy code

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
pytest
```

`ltd harness` scores a synthetic salience signal (and a random control) for turn shifts, backchannels, and barge-ins at 0.16–2.0 s horizons. That path needs no weights.

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
configs/         YAML defaults + spark_slice.yaml reference numbers
docs/SPARK.md    aarch64 cu130 notes; do not re-run Spark
tests/           harness, delay-NaN NLL, VAP pooling, Spark reference
```

Local `./data`, `./weights`, and `./cache` are gitignored. Create them yourself after you have access.

## What comes later

Phase 1 adds a JEPA-style latent predictor and uses its error as salience. Phase 2 conditions the dialogue policy on that signal. Neither is implemented here.

## Citation

If you use this skeleton, cite the independent-study title and the upstream papers in the table above.

```
Vel Moon Reichman. Latent Predictive Objectives for Timing Control
in Full-Duplex Dialogue Models. CCSF CS 199, Fall 2026.
```
