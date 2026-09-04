# Phase 1 plan — latent predictor heads (scaffolding)

CS 199 · Latent Predictive Objectives for Timing Control · 2026-09-03 PT

**Status:** Phase 1 scaffolding after Phase 0 freeze. Velvet approved starting
Phase 1. This document is the operational plan for *this* repo. It does **not**
implement Phase 2 fine-tuning and it does **not** unfreeze Moshi or BayLing.

Phase 0 numbers live in [PHASE0_INTERIM_FINDINGS.md](PHASE0_INTERIM_FINDINGS.md).
Do not copy or invent them here. Comparisons use the equal-length window
protocol in [EVAL_PROTOCOL_PHASE0.md](EVAL_PROTOCOL_PHASE0.md) (prefer fixed
windows; full-length Moshi gaps are descriptive / length-confounded only).
Spark RQ2 interim (CANDOR-only proxy first; evening-PT DC channel-energy
VAD addendum, H=12 λ=0.01, mid-180):
[PHASE1_RQ2_INTERIM.md](PHASE1_RQ2_INTERIM.md). That note is not a freeze
and is not Phase 2. VAD ≠ gold. Silence/collapse diagnostic (57 mid-180
sessions, channel-energy masks; criterion **not met**, 0/57; not a
timing win): [PHASE1_SILENCE_COLLAPSE.md](PHASE1_SILENCE_COLLAPSE.md).

## Goal

Precompute **frozen-model** hidden states and **target user-chunk embeddings**.
Train **small predictor heads** (single-digit millions of parameters) to predict
the next user-chunk embedding from the current hidden state, with an
**isotropic-Gaussian regularizer** (JEPA-style collapse control). Treat
prediction error as **latent surprise** and score it on the existing turn-event
harness against Phase 0 Moshi user-channel NLL and VAP.

```
h_t  = frozen Moshi hidden state at chunk t          (backbone never trained)
z_{t+h} = frozen embedding of LEFT=user audio at t+h
hat{z}_{t+h} = Head(h_t)                             (only this is trained)
L = ||hat{z} - sg(z)||^2 + λ R_iso(hat{z})
surprise_t = ||hat{z}_{t+h} - z_{t+h}||^2            → ChunkSignal → harness
```

Horizons to ablate: **80 ms, 1 s, 5 s**. Regularization strengths to ablate:
**λ ∈ {0, 0.1, 1.0, 10.0}**.

## What is in-repo now vs. reserved for Spark

| Piece | In this PR | Needs Spark / local weights |
|---|---|---|
| Plan + Spark paths | yes | — |
| Chunked stereo dataset (LEFT=user) | yes (numpy / synthetic) | real CANDOR / DuplexChat waveforms |
| Frozen hidden-state interface (Moshi first) | yes | `MoshiWrapper.load(local_dir=...)` + cache `.npz` |
| Target embedding interface | yes (fake / mean-pool) | frozen Mimi user-chunk latents |
| Small MLP / tiny Transformer head | yes (numpy) | optional torch on Spark |
| Loss: MSE + isotropic-Gaussian regularizer | yes | — |
| Train loop stub (head only) | yes (CPU numpy SGD) | full cache + more steps |
| Surprise → existing `score_signal` harness | yes | real sessions + events |
| Phase 2 policy / backbone FT | **out of scope** | do not start |

BayLing-Duplex stays a Phase 0 frozen baseline only. Do not train a BayLing
head until Moshi Phase 1 caches and one Moshi-head ablation exist.

## Spark paths (`spark-61dd`, aarch64 GB10)

Phase 0 artifacts (read-only; do not re-run those jobs):

| Tree | Role |
|---|---|
| `/home/velvet/cs199-candor-work/` | CANDOR part_001 NLL / VAP / window / bootstrap summaries |
| `/home/velvet/cs199-duplexchat-work/` | DuplexChat10 + expanded summaries |
| `/home/velvet/cs199-moshi-work/` | prior DuplexChat10 Moshi JSONL |
| `/home/velvet/cs199-bayling-work/` | prior DuplexChat10 BayLing JSONL |
| `/home/velvet/cs199-vap-work/` | prior DuplexChat10 VAP JSONL |

Phase 1 working tree (create; do not write into Phase 0 dirs):

```
/home/velvet/cs199-phase1-work/
  hidden/moshi/{duplexchat_expanded,part001_excl_pilot}/<session_id>.npz
  targets/user_chunk/{duplexchat_expanded,part001_excl_pilot}/<session_id>.npz
  heads/{mlp,transformer}/h{80ms,1s,5s}/lam{0,0.1,1,10}/
  eval/windows/{first180,mid180}/
  logs/
```

Constants: `latent_timing_duplex.phase1.paths`. This repo still does not
download CANDOR, podcasts, or weights.

## Concrete steps

Do these in order. Each step should leave a test, a cache manifest, or a
CLI command greener than before.

### 1. Confirm Phase 0 freeze inputs

- Read [PHASE0_INTERIM_FINDINGS.md](PHASE0_INTERIM_FINDINGS.md) and
  [EVAL_PROTOCOL_PHASE0.md](EVAL_PROTOCOL_PHASE0.md).
- Primary contrast remains **DuplexChat expanded** vs **part001 excluding
  pilot (n=45)** unless a duration-matched or **fixed-window** subset is
  specified.
- Prefer **fixed windows** (first-W and mid-W, W=180 s, same n=12 pairing
  rule as Phase 0) when surprise is compared to Moshi NLL or VAP.
- Do not treat full-length Moshi CANDOR-worse numbers as the scientific
  baseline.

### 2. Precompute frozen Moshi hidden states

On `spark-61dd`, same flags as Phase 0 NLL (`NO_CUDA_GRAPH=1`,
`NO_TORCH_COMPILE=1`, no flash-attn, no GB10 kernel fork; see
[SPARK.md](SPARK.md)):

1. Load Moshi locally (`MoshiWrapper.load(local_dir=...)`).
2. Pack stereo **LEFT=user, RIGHT=agent** (Phase 0 spot-check PASS; no remap).
3. Teacher-forced frozen forward; cache per-chunk hidden states `[T, H]`
   (Moshi transformer width, documented default H=4096).
4. Write `/home/velvet/cs199-phase1-work/hidden/moshi/.../<session>.npz`
   with `hidden`, `t_end`, `session_id`, `chunk_duration_s=0.08`.
5. Keep the backbone in `eval()`; never register it with an optimizer.

`MoshiHiddenStateExtractor` in this repo is the interface. Without local
weights it raises `WeightsNotBundled`. Tests use `FakeHiddenStateExtractor`.

### 3. Precompute target user-chunk embeddings

1. For each 80 ms LEFT=user chunk, embed with a **frozen** encoder
   (preferred: Mimi continuous latents already computed during Moshi
   encode; optional fixed random projection for smoke tests).
2. Cache `[T, E]` with E=256 (default) next to the hidden-state files
   under `targets/user_chunk/`.
3. Do not train the target encoder in Phase 1. Stop-grad on `z` in the loss.

### 4. Train small heads (CPU or Spark)

Default MLP (~2.5M params): `4096 → 512 → 512 → 256`. Tiny Transformer is
the architecture ablation (project 4096→256, two layers, then 256).

| Item | Value |
|---|---|
| Trainable | predictor head only |
| Frozen | Moshi, BayLing, Mimi, VAP, target embedder |
| Device | CPU is enough for the head; Spark GB10 128 GB unified is optional |
| Envelope | <10M params; numpy SGD in-repo; torch AdamW optional on Spark |
| Batch | 32 cached `(h_t, z_{t+h})` pairs |
| Steps | start with 1k–5k on a smoke cache; scale after one horizon works |

`freeze_backbone=False` raises `Phase2OutOfScope`. There is no Phase 2
entry point.

Training may run on Spark or on a CPU box that only sees the cached
`.npz` files. Precompute is the expensive step (frozen Moshi forward);
head SGD is the cheap step.

### 5. Surprise and eval harness

In-repo scorer: `ltd phase1-eval` / `phase1.compare.run_turn_event_eval`.
Protocol: [EVAL_PROTOCOL_PHASE1.md](EVAL_PROTOCOL_PHASE1.md).

1. `surprise_t = ||hat{z}_{t+h} - z_{t+h}||^2` (MSE; optional Gaussian NLL).
2. Wrap as `ChunkSignal` (`name="jepa:surprise"`) on the 80 ms grid.
3. Align Phase 0 Moshi NLL JSONL and VAP `p(shift)` / `p_now` on the **same**
   `t_end` indices (intersect when surprise is `T−H`).
4. Labels: gold `--labels` JSONL, or transcript / VAD proxies
   (`phase1.labels`). Proxies are not official annotations.
5. Metrics per signal × event × eval horizon: AUROC, AUPRC, F1-max,
   precision at recall 0.30 / 0.50 / 0.70. Session-level Efron bootstrap,
   seed `20260903`, B=10000.
6. Turn-event eval horizons stay `{0.16, 0.32, 0.50, 1.00, 2.00}` s.
   Predictor H on Spark is **{1, 12, 62}** frames (not the plan half-up
   13 / 63). λ primary = 0.01; λ = 0 is the reconstruction reference.

`ltd phase1-eval --synthetic` is the CPU smoke. Real Spark paths are
caller-supplied (see the protocol). Do not invent findings.

Equal-length protocol (required for Phase 0 comparisons):

- Prefer W=180 s exact-W pairs (n=12 rule: drop DuplexChat episodes
  shorter than W).
- Report **mid-window** as the primary equal-length cut; keep first-W
  as a position check (Phase 0: first-180 reverse does not hold mid).
- Conversation/episode bootstrap B=10000, seed `20260903`, Efron 2.5/97.5.
- Duration-weighted within conversation; unweighted across units.
- Do not rank BayLing token NLL against Moshi codebook NLL or against
  surprise.

### 6. Ablations

| Factor | Grid | Notes |
|---|---|---|
| Predictor horizon | 80 ms / 1 s / 5 s | 1 frame / 13 frames / 63 frames at 12.5 Hz (half-up; see `phase1.horizons`) |
| λ regularizer | 0, 0.1, 1.0, 10.0 | λ=0 is the collapse-control control |
| Head | MLP (default), tiny Transformer | same embed dim |
| Window | mid-180 (primary), first-180 | protocol, not a model knob |

One change at a time. Land 80 ms + λ=1.0 + MLP + mid-180 first.

## Success criteria

Phase 1 scaffolding (this PR) is done when:

1. `docs/PHASE1_PLAN.md` exists and points at the Phase 0 freeze docs
   without inventing numbers.
2. `latent_timing_duplex.phase1` imports on a CPU-only install
   (numpy + PyYAML; no torch, no weights).
3. Tests lock horizon indexing, regularizer behavior, head shapes /
   param budget, dataset LEFT=user packing, and surprise→harness.
4. `ltd phase1` runs a synthetic train-step + harness hook.
5. `ltd phase1-eval --synthetic` runs the compare table on fake tensors.
6. Phase 2 fine-tune paths raise `Phase2OutOfScope`.

Phase 1 *science* (later, on Spark) is done when:

1. Hidden-state and target caches exist for DuplexChat expanded and
   CANDOR part001 excluding pilot (or the equal-length n=12 subset).
2. At least the 3×4 horizon × λ grid is trained for the MLP head.
3. Surprise, Moshi NLL, and VAP are scored on the **same** fixed
   windows and the same turn-event labels.
4. A short write-up states whether surprise is predictive, weakly
   predictive, or not predictive relative to those frozen baselines.
   A null is still a result. Interim: [PHASE1_RQ2_INTERIM.md](PHASE1_RQ2_INTERIM.md)
   (weakly predictive on a CANDOR transcript proxy; evening-PT addendum
   is a channel-energy VAD pool, not gold, not Phase 2). Silence/collapse
   (different question): [PHASE1_SILENCE_COLLAPSE.md](PHASE1_SILENCE_COLLAPSE.md)
   — criterion not met (0/57); not a timing win by itself.
5. Moshi and BayLing weights were never trained.

## Out of scope (do not do in Phase 1)

- Phase 2: condition the dialogue policy on surprise; fine-tune Moshi /
  BayLing; LoRA / full FT; generation-time steering.
- Unfreezing any backbone “just to see.”
- Downloading CANDOR parts 002–034.
- Re-running Phase 0 Spark NLL / VAP jobs.
- Comparing BayLing PPL to Moshi codebook NLL or to surprise.
- Claiming timing-objective success from VAP `p(shift)` alone when CIs
  overlap (Phase 0 protocol).

## Hardware envelope

| Job | Where | Memory / notes |
|---|---|---|
| Frozen Moshi hidden extract | spark-61dd GB10, 128 GB unified | same as Phase 0 NLL; cu130; no flash-attn |
| Target Mimi embed | same box, during extract | cheap vs. LM forward |
| Head training (~2.5M) | Spark **or** CPU with `.npz` | fits in a few hundred MB |
| Eval harness | CPU | already Phase 0 |

This repository is not attached to Spark. CI and `pytest` stay CPU-only.

## Package map

```
src/latent_timing_duplex/phase1/
  horizons.py   80 ms / 1 s / 5 s frame offsets
  dataset.py    chunked stereo, LEFT=user
  hidden.py     frozen Moshi hidden-state interface
  targets.py    frozen user-chunk embedding interface
  heads.py      small MLP / tiny Transformer
  losses.py     MSE + isotropic-Gaussian regularizer
  train.py      head-only loop stub
  surprise.py   surprise metric + harness hook
  labels.py     gold JSONL + transcript / VAD / CANDOR CSV proxies
  series.py     NLL / VAP / surprise JSONL alignment (reject aggregates)
  artifacts.py  candor_/dc_ .pt filename aliases
  checkpoint.py h*_lam* + H_set lock + nested mlp_state_dict
  export_series.py  per-step NLL/VAP JSONL schema (no invented series)
  compare.py    surprise vs NLL vs VAP on one timeline
  windows.py    equal-length first-W / mid-W crops
  paths.py      spark-61dd path constants
```

CLI: `ltd phase1` (synthetic train demo), `ltd phase1-eval` (turn-event
compare; `--synthetic` or Spark paths), `ltd check` (imports `phase1`).
