# Phase 1 turn-event evaluation protocol

CS 199 · Latent Predictive Objectives for Timing Control · 2026-09-03 PT

**Status:** in-repo scorer + tests. This document is the protocol Spark must
follow when scoring trained heads. It does **not** report empirical results
and it does **not** start Phase 2.

Phase 0 freeze inputs stay in [PHASE0_INTERIM_FINDINGS.md](PHASE0_INTERIM_FINDINGS.md)
and [EVAL_PROTOCOL_PHASE0.md](EVAL_PROTOCOL_PHASE0.md). Do not invent NLL / VAP
digits here. Prefer the equal-length window rule from Phase 0.

## RQ2 (this protocol)

Does explicit latent surprise predict turn events (turn shifts, backchannels,
barge-ins) better than (a) Moshi user-channel token/codebook NLL and (b) VAP,
**on the same mid-180 s windows**?

The scorer emits AUROC, AUPRC, and precision-recall operating points for every
named signal. The Spark interim write-up is
[PHASE1_RQ2_INTERIM.md](PHASE1_RQ2_INTERIM.md) (weakly predictive on a
CANDOR transcript proxy first; evening-PT addendum is a channel-energy
VAD pool, not gold, not Phase 2). Silence/collapse on the same mid-180
pool is a **different question** (representation flatline in mutual
silence, not turn-event AUROC):
[PHASE1_SILENCE_COLLAPSE.md](PHASE1_SILENCE_COLLAPSE.md) (criterion not
met, 0/57; not a timing win). This protocol file still does not
invent digits.

## Locked settings

| Item | Value |
|---|---|
| Window | **mid-180 only** for the primary table (center crop, exact-W). first-180 is a position check, not the headline. |
| Frame grid | 80 ms (12.5 Hz). mid-180 → **T = 2250** |
| Hidden | frozen Moshi `[T=2250, D=4096]` |
| Predictor H | **{1, 12, 62}** frames (Spark-trained). See horizon note below. |
| λ primary | **0.01** |
| λ reference | **0** (reconstruction / collapse-control control) |
| Seed | **20260903** |
| Bootstrap | conversation/episode-level, B=10000, Efron 2.5 / 97.5 |
| Eval horizons | `{0.16, 0.32, 0.50, 1.00, 2.00}` s (Phase 0 turn-event grid) |
| Signals | `jepa:surprise`, `nll:moshi`, `vap:p_shift` (optional `random`) |
| Phase 2 | **out of scope** |

Aggregation: duration-weighted *within* a conversation when a clip is split;
**unweighted across** conversations/episodes (same unit rule as Phase 0).
The in-repo aggregator bootstraps the session-level AUROC / AUPRC means.

Do not rank losses that are not on the same scale (BayLing token NLL vs Moshi
codebook NLL vs surprise MSE).

## Horizon note (plan table vs Spark grid)

`phase1.horizons.horizon_steps` still uses **half-up** rounding for the
written plan: 1 s → 13 frames, 5 s → 63 frames.

Spark ablations were trained at **H ∈ {1, 12, 62}** (floor of the same
seconds; 12 × 80 ms = 0.96 s, 62 × 80 ms = 4.96 s). The scorer takes an
**explicit frame offset**. Do not convert `horizon_s=1.0` through half-up
when loading `h12_lam0.01`.

| Seconds (nominal) | Plan half-up | Spark-trained H |
|---|---|---|
| 80 ms | 1 | 1 |
| 1 s | 13 | 12 |
| 5 s | 63 | 62 |

## Same-timeline rule

For each session:

1. Crop metadata / series to the mid-180 window (or load already-cropped
   caches). Sessions shorter than 180 s are **dropped** (Phase 0 exact-W).
2. Compute surprise at source frames `t = 0 … T−H−1`
   (`surprise_t = ||hat z_{t+H} − z_{t+H}||^2`).
3. Align Moshi NLL and VAP `p(shift)` (or `1 − p_now`) onto the **same**
   `t_end` indices. If surprise is shorter by H, intersect frames.
4. Labels: an event of kind *k* in `(t_end, t_end + h_eval]` is the binary
   target. Chunks whose horizon would run past the window are dropped.
5. Score every named signal with the same `y`.

VAP at 50 Hz is mean-pooled to 80 ms (4 frames) before the crop, matching
Phase 0.

## Labels

Preferred: gold / existing annotations via `--labels` JSONL:

```json
{"session_id": "uuid-or-episode", "events": [{"t": 12.4, "kind": "turn_shift", "speaker": "user"}]}
```

Kinds: `turn_shift`, `backchannel`, `barge_in`.

Fallback (mark as proxy in the report):

* `--transcripts` — speaker turns with `start` / `end` → speaker-change /
  short-burst / overlap heuristics (`phase1.labels`).
* `--vad` — per-speaker intervals (`user_vad` / `assistant_vad` or LEFT/RIGHT).

Limitations of the fallback are listed by `describe_label_limitations()` and
copied into every JSON report. Do not treat proxy labels as official CANDOR
or DuplexChat turn annotations.

## What Spark must pass

Nothing under `/home/velvet/...` is opened unless you pass it. Typical
spark-61dd invocation (λ=0.01 primary, one H; repeat for 1 and 62, and for
λ=0):

```bash
ltd phase1-eval \
  --ablations-root /home/velvet/cs199-phase1-work/ablations \
  --selection-locked /home/velvet/cs199-phase1-work/ablations/SELECTION_LOCKED.json \
  --hidden-dir /home/velvet/cs199-phase1-work/hidden/moshi/part001_excl_pilot \
  --target-dir /home/velvet/cs199-phase1-work/targets/user_chunk/part001_excl_pilot \
  --nll-jsonl  /home/velvet/cs199-candor-work/<mid180-moshi-nll>.jsonl \
  --vap-jsonl  /home/velvet/cs199-vap-work/<mid180-vap>.jsonl \
  --labels     /home/velvet/cs199-phase1-work/eval/labels/<slice>.jsonl \
  --horizon-frames 12 \
  --lambda-reg 0.01 \
  --lambda-role primary \
  --window-mode mid --window-s 180 \
  --seed 20260903 \
  --bootstrap 10000 \
  --output /home/velvet/cs199-phase1-work/eval/windows/mid180/h12_lam0.01.json
```

Repeat for DuplexChat (`duplexchat_expanded` hidden/target dirs and the
matching Phase 0 JSONLs). Primary corpora remain **45 CANDOR part001
excluding pilot** and **12 DuplexChat** exact-W pairs.

`SELECTION_LOCKED.json` is optional. If missing, defaults are H∈{1,12,62},
λ_primary=0.01, λ_reference=0, seed 20260903, mid-180.

Checkpoint layout expected under `--ablations-root`:

```
h1_lam0/        h1_lam0.01/
h12_lam0/       h12_lam0.01/
h62_lam0/       h62_lam0.01/
SELECTION_LOCKED.json   # optional
grid_results.jsonl      # training log; not read by the scorer
```

Each run dir should contain `checkpoint.npz` (preferred; in-repo numpy
`[in, out]` weights) or `head.npz` / `model.pt`. Convert a torch state dict
to `.npz` with `save_mlp_checkpoint` if load fails.

`--surprise-jsonl` skips the head forward when surprise is already cached.

This CLI **replaces** the spirit of Spark’s `MISSING_HARNESS.md`: the gap
was “no in-repo scorer / flags.” The scorer is `ltd phase1-eval`. Spark
still supplies the local files listed above.

Smoke (no Spark files):

```bash
ltd phase1-eval --synthetic
```

`ltd phase1` remains the tiny train-step demo.

## JSONL field aliases

Moshi NLL (first match wins): `audio_nll_per_step`, `audio_per_step`,
`nll_audio`, `user_channel_nll`, `nll`, `values`.

VAP: `p_shift` / `p(shift)` or `p_now` (inverted to `1 − p_now` when
LEFT=user). 50 Hz series are pooled.

Session id: `session_id`, `conversation_id`, `uuid`, `clip_id`, `episode_id`.
Window suffixes (`:mid180`) are stripped for joins.

Full-length series are cropped with `phase1.windows.window_bounds`. Series
that are already length 2250 are treated as windowed.

## Metrics

Per session × signal × event kind × eval horizon:

* AUROC (pairwise; `None` if a class is missing)
* AUPRC (average precision)
* F1-max operating point (precision, recall, threshold)
* Precision at recall ∈ {0.30, 0.50, 0.70}

Across sessions: unweighted mean + Efron bootstrap CI (seed 20260903).

## Spark artifact adapters (spark-61dd)

Morpheus’s Phase 1 / Phase 0 trees do **not** match the first-cut CLI
assumptions. The scorer now adapts these layouts. Absolute paths are still
never opened unless passed.

| Artifact | Spark layout | Adapter |
|---|---|---|
| Selection lock | `/home/velvet/cs199-phase1-work/ablations/SELECTION_LOCKED.json` with `{"horizons":{"H_set":[1,12,62]}}` or top-level `H_set` | `coerce_horizon_frames` accepts a list **or** a dict (`H_set` / `horizons` / `H`) |
| Checkpoints | `h12_lam0.01/checkpoint.pt` with weights under `mlp_state_dict` (`net.0.weight`, …) | Recursive unwrap of `state_dict` / `mlp_state_dict` / `model_state_dict` / `predictor`. `.pt` / `.pth` / `.npz` |
| Hidden / targets | `candor_<uuid>.pt`, `dc_<uuid>.pt` (torch `[T,D]`) | Stem match + strip `candor_` / `dc_`; also `.npz`/`.npy` |
| Phase 0 NLL / VAP JSONL | **Aggregate-only today**: `audio_nll`, `p_shift_mean`, `duration_sec` | `duration_sec` ↔ `duration_s`. **No fake per-step series.** Missing per-step fields raise with the schema below. |
| Labels | CANDOR `extract*/transcription/*.csv` (no gold JSONL yet) | `--transcripts-dir` (or `--transcripts` pointing at a directory). Proxies. |

### Per-step series are the scientific blocker

Clip-level `audio_nll` / `p_shift_mean` cannot yield AUROC against turn
events. Do **not** broadcast a mean across T=2250. On Spark, re-emit
per-step JSONLs from existing mid-180 audio with the Phase 0 extractors:

```bash
ltd phase1-export-series --print-schema
```

```python
from latent_timing_duplex.phase1.export_series import (
    nll_record_from_extractor,
    vap_record_from_baseline,
    write_jsonl,
)
from latent_timing_duplex.phase1.windows import crop_session
from latent_timing_duplex.extract.nll import FrozenNLLExtractor, prepare_moshi_forward_env

prepare_moshi_forward_env()  # NO_CUDA_GRAPH=1 NO_TORCH_COMPILE=1
cropped = crop_session(session, window_s=180.0, mode="mid")
write_jsonl(nll_out, [nll_record_from_extractor(cropped, nll_ext, crop=False)])
write_jsonl(vap_out, [vap_record_from_baseline(cropped, vap, crop=False)])
```

Required per-step fields: `audio_nll_per_step` (or a list-valued `nll` /
`values`); `p_shift_per_step` or a list-valued `p_shift` / `p_now`.
`--surprise-only` skips baselines for a dry run and is **not** the RQ2 table.

### CANDOR transcription CSV columns (proxies)

Header required. First match wins:

| Role | Aliases |
|---|---|
| speaker | `speaker`, `speakerId`, `spk`, `role` |
| start (s) | `start`, `startTime`, `start_s` |
| end (s) | `end`, `stopTime`, `endTime`, `stop` |
| text | `text`, `utterance` (optional) |

Filename stem is the session id (`candor_` / `dc_` stripped). Values `> 1e4`
are treated as milliseconds. These are **not** official CANDOR turn labels.

## Out of scope

- Phase 2 policy fine-tune / unfreezing Moshi or BayLing
- Downloading CANDOR parts 002–034
- Re-running Phase 0 NLL / VAP jobs
- GB10 kernel forks or GPU in CI
- Claiming a win from this protocol document or from `ltd phase1-eval --synthetic`
