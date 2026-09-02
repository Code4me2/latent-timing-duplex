# Phase 0 — weeks 1–5

**Gate:** Can any per-chunk implicit signal (starting with frozen user-channel NLL) predict turn shifts, backchannels, and barge-ins at multiple horizons? A weak or null result is still reportable.

This file is the working checklist for the skeleton in this repo. Nothing below downloads data or weights.

## Five work items

1. **Inference wrappers** for Moshi and BayLing-Duplex (stubs exist).
2. **VAP baseline** (Ekstedt & Skantze; stub exists).
3. **Working subset, 200–500 h**, speaker-separated dual-channel dialogue: CANDOR (after access) + DuplexChat (reconstruct locally).
4. **Eval harness:** any per-chunk signal → turn-event prediction at multiple horizons (**implemented** on synthetic timelines).
5. **Extract** per-chunk user-channel NLL from frozen models; score on the harness (stub exists).

## Suggested order

Do these in this order. Each step should leave a test or a CLI command greener than before.

| Step | Work | Why this order | Status in this skeleton |
|---|---|---|---|
| 1 | Harness on synthetic stereo timelines | Proves the metric before any model or corpus | **Done** — `ltd harness` |
| 2 | Moshi smoke test | Public code + public HF weights; smallest honest “can we load *something*” path. Still requires you to download weights yourself. | **Local load + NLL** — `models/moshi.py`, `extract/nll.py`. Spark 10-clip numbers in `docs/SPARK.md` (do not re-run). |
| 3 | VAP baseline | Stereo turn-taking scores on the same harness, no 9B LLM | **CPU path** — `baselines/vap.py` (local checkpoint, LEFT=user). |
| 4 | Small DuplexChat reconstruct | Manifest-only on HF; reconstruct a *tiny* English slice first (hours, not 200 h) via the official script | Stub: `data/duplexchat.py` |
| 5 | CANDOR after access | Human-subjects corpus; do not start conversion until the BetterUp request is approved | Stub: `data/candor.py` |
| 6 | Frozen user-channel NLL | Moshi first, then BayLing-Duplex once local 4-shard weights + GLM-4-Voice tokenizer/decoder are on disk | **Moshi done** — `LMModel.forward` + delay-NaN mask. BayLing load still local-only stub. |

BayLing-Duplex is deliberately *after* a Moshi smoke test: ~19 GB, ~9.5B BF16, plus `zai-org/glm-4-voice-tokenizer` and `zai-org/glm-4-voice-decoder`. The Hugging Face card’s “516k params” is a display bug (four `model-0000k-of-00004.safetensors` shards, ~9.54B parameters).

## TODOs (fill these in; do not skip the blockers)

### Wrappers

- [x] Wire `MoshiWrapper.load(local_dir=...)` to [kyutai-labs/moshi](https://github.com/kyutai-labs/moshi) using a directory *you* populated from `kyutai/moshiko-pytorch-bf16` or `kyutai/moshika-pytorch-bf16`. Local files only; no Hub download. Spark: aarch64 cu130, no flash-attn, sphn sdist, extra tiktoken/torchaudio, no GB10 kernel fork — see [docs/SPARK.md](docs/SPARK.md).
- [ ] Wire `BayLingDuplexWrapper.load(...)` to [BayLing-Models/BayLing-Duplex](https://github.com/BayLing-Models/BayLing-Duplex) using local copies of `BayLing-Models/BayLing-Duplex`, `zai-org/glm-4-voice-tokenizer`, and `zai-org/glm-4-voice-decoder`. Token NLL from the 10-clip Spark job is recorded as a reference only (vocab 168960; not comparable to Moshi codebook NLL).
- [x] Do not add a default cache path. If `local_dir` is missing, keep raising `WeightsNotBundled`.

### VAP

- [x] Load a local VAP state dict from [ErikEkstedt/VAP](https://github.com/ErikEkstedt/VAP) (their `examples/` checkpoint or one you train). CPU-ok.
- [x] Emit `list[ChunkSignal]` at the same 80 ms grid the harness uses (`p(shift) = 1 - p_now` for LEFT=user).

### Data

- [ ] **Blocker — DuplexChat reconstruct-from-podcasts.** `sarulab-speech/DuplexChat` is a manifest. Use the official reconstruct path in [sarulab-speech/DuplexChat](https://github.com/sarulab-speech/DuplexChat). Podcast audio stays with the rightsholders; do not commit shards.
- [ ] Start with a small English reconstruct (enough to debug stereo I/O), then grow toward the 200–500 h working set.
- [ ] **Blocker — CANDOR BetterUp request.** Request access at https://betterup-data-requests.herokuapp.com/ before writing any CANDOR loader body. Follow their data-use terms. Do not commit raw media.
- [ ] After access: speaker-separated dual-channel conversion and a documented subset recipe (target 200–500 h combined with DuplexChat).

### Harness and NLL

- [x] Keep the harness model-agnostic. New signals should be `list[ChunkSignal]`, not new metrics modules.
- [x] Implement `FrozenNLLExtractor.extract` as a thin adapter over `model.user_channel_nll`.
- [x] Moshi path: `LMModel.forward` + delay-NaN mask (`extract/nll.py`). Spark 10-clip reference in `ltd reference` / [docs/SPARK.md](docs/SPARK.md). Do not re-run Spark.
- [ ] Score frozen NLL vs. VAP vs. a random control on the same sessions and horizons (`configs/eval.yaml`) once a local reconstruct exists.
- [ ] Write up the gate: predictive, weakly predictive, or not predictive. All three are valid outcomes.

Hypothesis (non-binding): Moshi NaNs came from the acoustic delay pattern (`_undelay_sequence(..., fill_value=NaN)`).

## What this phase is not

- No Phase 1 JEPA encoder/predictor.
- No Phase 2 policy conditioning.
- No training scripts.
- No unpublished weight URLs.

## Hardware (planned, not attached)

Two DGX Spark boxes (128 GB unified) are the intended frozen-inference hosts. 2× RTX 5090 is intended later. Neither is provisioned by this repository.
