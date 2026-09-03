# CS 199 Phase 0 interim findings
**Latent Predictive Objectives for Timing Control in Full-Duplex Dialogue Models**  
Velvet Reichman · Indika Walimuni · CCSF CS 199 · draft 2026-09-03 (PT)

## Status
Phase 0 frozen baseline pipeline is operational on `spark-61dd` (Linux aarch64, GB10/`sm_121`, CUDA 13). Engineering validation is complete for Moshi, BayLing-Duplex, and VAP on stereo duplex audio. Full-length Moshi corpus differences are descriptive only: they do not survive equal-length first-W windows. Scientific claims remain provisional; mid-window pending.

## What ran
| Slice | n (conversations) | Duration | Notes |
|---|---|---|---|
| DuplexChat EN reconstruct (DuplexChat10) | 5 episodes / 10 clips | 1383.44 s (~23.1 min) | Stereo 24 kHz DialogueSidon reconstruct; original slice, kept separate |
| DuplexChat EN expanded | 18 new episodes / 40 clips | ~4873 s | New EN reconstruct episodes; reported separately from DuplexChat10 |
| CANDOR5 (pilot) | 5 | 9213.01 s (~2.56 h) | First 5 UUIDs from part_001; not used for primary contrast |
| CANDOR20 (predeclared) | 20 new UUIDs | 47158.95 s (~13.1 h) | Part_001 only |
| CANDOR rest | 25 | 48903.4 s (13.58 h) | Remaining part_001 UUIDs; no exclusions |
| Pooled part001 excl. pilot (CANDOR20+rest) | 45 | — | Primary CANDOR pool; excludes CANDOR5 |
| Full part001 (includes CANDOR5) | 50 | — | All part_001 UUIDs; no parts 002–034 |
| CANDOR20 resample 48→24 kHz | 20 (same UUIDs) | — | Moshi NLL only; sample-rate confound check |
| Duration-matched (18 shortest CANDOR) | 18 vs DC expanded n=18 | median 1681 s vs DC 230 s | 1:1 NN w/o replacement; supports DISJOINT |
| Fixed-window first-W=180 s (t=0) | exact-W pairs n=12 | 180 s each | 6 DC <180 s excluded; CANDOR 24 kHz; Moshi NLL |
| Fixed-window first-W=300 s (t=0) | n=5 | 300 s each | Same verdict as W=180; Moshi audio |

Fixed settings: Moshi frozen NLL (`NO_CUDA_GRAPH=1`, `NO_TORCH_COMPILE=1`, nan-safe forward); BayLing-Duplex frozen token NLL (SDPA, no flash-attn); VAP CPU-only. Channel convention **LEFT=user, RIGHT=agent** validated 2026-09-03: 3/3 CANDOR and 3/3 DuplexChat spot-checks PASS; no remap. CANDOR: channel_map ObjectIds match speakers/transcript; exclusive-turn energy 7–27× prefers mapped channel. DuplexChat: packing LEFT=stream0/speaker A; SPEAKER_XX labels do not bind per-turn to L/R, but channels are distinct and both active (medium-high confidence).

Uncertainty: conversation/episode-level bootstrap, B=10000, seed `20260903`, Efron 2.5/97.5 percentiles. DuplexChat10 primary unit = episode (n=5); DuplexChat expanded primary unit = episode (n=18); within-episode metrics duration-weighted before bootstrap. CANDOR primary unit = conversation UUID (CANDOR20 n=20; rest n=25; pooled excl. pilot n=45; full part001 n=50).

## Full-length results (descriptive; duration-weighted, 95% CI)

Full-length and duration-NN Moshi contrasts are reported here as descriptive. They do **not** survive equal-length windows (see fixed-window section) and are not the primary scientific claim.

### Moshi codebook NLL
| Corpus | Audio NLL | Text NLL |
|---|---|---|
| DuplexChat10 | 0.4880 [0.3749, 0.5208] | 0.0258 [0.0187, 0.0304] |
| DuplexChat expanded | 0.549 [0.483, 0.605] | 0.0314 [0.0242, 0.0381] |
| CANDOR20 | 0.6161 [0.6032, 0.6282] | 0.0386 [0.0360, 0.0412] |
| CANDOR rest | 0.623 [0.607, 0.639] | — |
| Pooled part001 excl. pilot | 0.619 [0.609, 0.630] | — |
| Full part001 | 0.622 [0.613, 0.632] | — |

### VAP timing
| Corpus | p_now | p_future | p(shift) |
|---|---|---|---|
| DuplexChat10 | 0.372 [0.273, 0.438] | 0.413 [0.342, 0.460] | 0.628 [0.562, 0.727] |
| DuplexChat expanded | — | — | 0.506 [0.408, 0.608] |
| CANDOR20 | 0.506 [0.475, 0.536] | 0.494 [0.474, 0.513] | 0.494 [0.464, 0.525] |
| CANDOR rest | — | — | 0.443 [0.409, 0.478] |
| Pooled part001 excl. pilot | — | — | 0.468 [0.443, 0.493] |
| Full part001 | — | — | 0.458 [0.433, 0.482] |

### Moshi sample-rate confound (CANDOR20)
CANDOR20 same 20 UUIDs resampled 48→24 kHz; Moshi NLL only.

| Rate | Audio NLL (dw) |
|---|---|
| 48 kHz | 0.616 [0.603, 0.628] |
| 24 kHz | 0.618 [0.606, 0.629] |

Pairwise Δ 24k−48k +0.0021 [−0.0016, +0.0056] (includes 0). Gap vs DuplexChat expanded 0.549 [0.483, 0.605] does not shrink (~0.067–0.069). Conclusion: sample rate is not the Moshi NLL gap; reverse DuplexChat→48 kHz skipped because 24k≈48k.

### Duration-matched reanalysis
Rule: 1:1 nearest-neighbor without replacement, `part001_excluding_pilot` n=45 vs DuplexChat expanded; DC sorted duration ascending then UUID; ties UUID lex; bootstrap B=10000, seed `20260903`.

Duration supports **DISJOINT**: CANDOR min 1538 s > DC max 904 s; sensitivity window empty; matched = 18 shortest CANDOR (median 1681 s vs DC 230 s).

| Corpus | Moshi audio NLL (dw) | VAP p(shift) (dw) |
|---|---|---|
| DuplexChat expanded | 0.549 [0.483, 0.605] | 0.506 [0.408, 0.608] |
| Matched CANDOR (18 shortest) | 0.625 [0.605, 0.644] | 0.484 [0.435, 0.532] |

Moshi audio CIs still fail to overlap (barely); the gap slightly widens. VAP `p(shift)` still overlaps. This contrast remains length-confounded (supports DISJOINT); equal-length first-W windows reverse the claim (below).

### Fixed-window Moshi NLL (equal length)
W=180 s from t=0; exact-W pairs n=12 (6 DuplexChat episodes <180 s excluded); CANDOR 24 kHz.

| Corpus | Moshi audio NLL |
|---|---|
| CANDOR (first 180 s) | 0.558 [0.524, 0.594] |
| DuplexChat (first 180 s) | 0.619 [0.579, 0.665] |

CIs overlap; the point estimate **reverses** (CANDOR lower, not higher). Text NLL also overlaps. W=300 s n=5 same verdict: CANDOR 0.570 [0.528, 0.613] vs DuplexChat 0.654 [0.593, 0.712]. Conclusion: the full-length / duration-NN Moshi “gap” was length-confounded; equal-length first-W windows show no durable CANDOR-higher-NLL effect. Mid-window pending.

### BayLing (not comparable to Moshi)
BayLing token NLL / perplexity uses a different vocabulary and objective. Report only as a separate frozen baseline: DuplexChat10 dw NLL ~7.17 (PPL ~1297); DuplexChat expanded token NLL 9.283 [8.192, 10.454]; CANDOR20 dw NLL ~8.80 (PPL ~6667); CANDOR rest dw 9.035 [8.463, 9.475]. Do **not** rank Moshi vs BayLing on these numbers.

## Interpretation
1. **Pipeline validation succeeded.** End-to-end stereo extract → Moshi/BayLing NLL → VAP works on Spark without GB10 kernel forks.
2. **CANDOR5 was misleading on timing.** Pilot `p(shift)=0.350` was not representative; CANDOR20 centers near 0.494.
3. **Full-length Moshi “gap” is descriptive, not a durable contrast.** At full length, CANDOR part001 CIs (rest 0.623 [0.607, 0.639]; pooled excl. pilot 0.619 [0.609, 0.630]; full 0.622 [0.613, 0.632]) do not overlap DuplexChat expanded 0.549 [0.483, 0.605]. That contrast does **not** survive equal-length windows and is demoted from the primary scientific claim. VAP `p(shift)` CIs overlap DuplexChat expanded 0.506 [0.408, 0.608] throughout expanded comparisons. The earlier DuplexChat10 non-overlap on `p(shift)` was fragile at n=5.
4. **Sample rate is not the full-length Moshi NLL difference.** CANDOR20 48→24 kHz audio dw 0.616 [0.603, 0.628] vs 0.618 [0.606, 0.629]; pairwise Δ +0.0021 [−0.0016, +0.0056] includes 0. Reverse DuplexChat→48 kHz skipped because 24k≈48k.
5. **Duration NN cannot equalize support.** Supports are DISJOINT (CANDOR min 1538 s > DC max 904 s). On the 18 shortest CANDOR (median 1681 s vs DC 230 s), full-length Moshi audio dw 0.625 [0.605, 0.644] vs DC 0.549 [0.483, 0.605]; CIs still fail to overlap (barely). VAP `p(shift)` overlaps: matched 0.484 [0.435, 0.532] vs DC 0.506 [0.408, 0.608]. This remains a length-confounded comparison.
6. **Equal-length first-W windows: no durable CANDOR-higher-NLL effect.** W=180 s from t=0, exact-W pairs n=12 (6 DC <180 s excluded), CANDOR 24 kHz: Moshi audio CANDOR 0.558 [0.524, 0.594] vs DuplexChat 0.619 [0.579, 0.665]; CIs overlap and the point estimate reverses. Text overlaps. W=300 s n=5 same verdict (CANDOR 0.570 [0.528, 0.613] vs DC 0.654 [0.593, 0.712]). The full-length / duration-NN Moshi “gap” was length-confounded.
7. **Caveats.** DuplexChat10 n=5 episodes is small and CIs are wide; expanded n=18 still has a wide `p(shift)` interval; W=180 keeps n=12 and W=300 n=5. Corpora still differ in genre and reconstruction path. Channel packing passed automated spot-check but remains a documented convention (especially DuplexChat SPEAKER_XX unbound to L/R). Phase 0 engineering succeeded; there is **no** robust Moshi cross-corpus effect under equal length. This is **not** confirmation of a latent predictive timing objective. Next: mid-window equal-length Moshi NLL, then freeze.

## Artifact paths (spark-61dd)
- CANDOR: `/home/velvet/cs199-candor-work/` (`nll_candor20_moshi.jsonl`, `nll_candor20_bayling.jsonl`, `vap_candor20.jsonl`, `candor20_summary.json`, `candor_rest_summary.json`, `candor_part001_pooled_summary.json`, `nll_candor25_*.jsonl`, `vap_candor25.jsonl`, `candor20_24khz_summary.json`, `nll_candor20_24khz_moshi.jsonl`, `duration_matched_summary.json`, `duration_matched_summary.md`, `duration_matched_pairs.json`, `window180_summary.json`, `bootstrap_ci.py`)
- DuplexChat: `/home/velvet/cs199-duplexchat-work/` (`duplexchat10_summary.json`, `bootstrap_ci_duplexchat10.py`, `duplexchat_expanded_*`)
- Prior DuplexChat JSONLs: `/home/velvet/cs199-moshi-work/nll_duplexchat10.jsonl`, `/home/velvet/cs199-bayling-work/nll_duplexchat10.jsonl`, `/home/velvet/cs199-vap-work/vap_duplexchat10.jsonl`
- Repo: GitHub `Code4me2/latent-timing-duplex` `main` @ `7366e5f` (Origin mirror may lag)

## Recommended next steps (before Phase 1/2)
1. ~~Finish channel/speaker spot-check~~ **done** (clean; no remap).
2. ~~Expand DuplexChat English reconstruct for tighter episode-level CIs~~ **done** (18 new episodes / 40 clips).
3. ~~Remaining part_001 CANDOR UUIDs~~ **done** (rest n=25; pooled excl. pilot n=45; full part001 n=50).
4. ~~Controlled sample-rate check (CANDOR 48 kHz vs DuplexChat 24 kHz)~~ **done** (Moshi NLL; 24k≈48k; reverse DC→48k skipped).
5. ~~Duration-matched reanalysis~~ **done** (supports DISJOINT; full-length Moshi difference is length-confounded). Matched protocol filed in `docs/EVAL_PROTOCOL_PHASE0.md`.
6. ~~Fixed-length first-W-seconds Moshi NLL~~ **done** (W=180 n=12 and W=300 n=5; CIs overlap; point estimate reverses; no durable CANDOR-higher-NLL effect).
7. Mid-window equal-length Moshi NLL, then Phase 0 freeze (engineering success + no robust Moshi cross-corpus effect under equal length; VAP overlaps throughout expanded comparisons).
8. Only then start latent predictive objective training (Phase 1), under a separate training protocol.

## Channel spot-check artifacts
`/home/velvet/cs199-candor-work/channel_spotcheck.json` and `.md` (copies under duplexchat work dir).

## Not done yet
- Phase 1/2 latent objectives
- Mid-window equal-length Moshi NLL
- Full CANDOR dump (parts 002–034)
- BayLing `load()` harness completion in-repo (BayLing smoke/NLL ran out-of-tree successfully)
