# Phase 0 matched evaluation protocol (FREEZE)
CS 199 · Latent Predictive Objectives for Timing Control · 2026-09-03 PT

**Status: Phase 0 FREEZE** — freeze criteria met pending merge to `main`. Phase 1 requires a separate go/no-go.

## Purpose
Freeze how frozen Moshi / BayLing / VAP baselines are computed and compared across DuplexChat and CANDOR before Phase 1 changes models or objectives.

## Fixed settings (do not change without a new protocol version)
- Hardware: spark-61dd only (Linux aarch64, GB10/`sm_121`, CUDA 13)
- Moshi: frozen NLL, nan-safe forward, `NO_CUDA_GRAPH=1`, `NO_TORCH_COMPILE=1`, LEFT=user RIGHT=agent
- BayLing-Duplex: frozen token NLL, SDPA, no flash-attn; **not comparable** to Moshi codebook NLL
- VAP: CPU-only ErikEkstedt checkpoint; report p_now, p_future, p(shift)
- Uncertainty: conversation/episode-level bootstrap, B=10000, seed `20260903`, Efron 2.5/97.5
- Aggregation: unweighted and duration-weighted; primary reporting = duration-weighted

## Corpora
| Name | Definition | Primary unit |
|---|---|---|
| DuplexChat10 | Original 5 episodes / 10 EN clips | episode |
| DuplexChat expanded | 18 new EN episodes / 40 clips (original 10 kept separate) | episode |
| CANDOR5 | Pilot first 5 UUIDs from part_001 | conversation |
| CANDOR20 | Next 20 predeclared UUIDs | conversation |
| CANDOR rest | Remaining 25 part_001 UUIDs | conversation |
| part001 excluding pilot | CANDOR20 + rest (n=45) | conversation |
| part001 all | CANDOR5 + CANDOR20 + rest (n=50) | conversation |

Primary cross-corpus contrast: **DuplexChat expanded** vs **part001 excluding pilot (n=45)** unless a duration-matched or fixed-window subset is specified.

## Confounds checked
1. Channel packing LEFT=user: spot-check PASS (no remap)
2. Sample rate 48→24 kHz on CANDOR20: Moshi Δ≈0; full-length gap to DuplexChat holds
3. Duration NN matching: supports disjoint; full-length Moshi gap holds on 18 shortest CANDOR; VAP overlaps
4. Fixed-length first-W window Moshi NLL: **done** (gap vanishes; CIs overlap; first-180 point estimate reverses)
5. Mid-window equal-length Moshi NLL: **done** (same n=12 center crop; CIs overlap; first-180 reverse does not hold mid-conversation). Random-offset CIs also overlap.

## Do not do under this protocol
- Download CANDOR parts 002–034 unless a new protocol version says so
- Start Phase 1/2 latent objective training
- Compare BayLing PPL directly to Moshi NLL
- Claim timing-objective success from VAP p(shift) alone when CIs overlap

## Phase 0 freeze criterion
**FREEZE status:** criteria met pending merge to `main`. Engineering success + no robust Moshi cross-corpus effect under equal length; VAP overlaps throughout expanded comparisons. Documented: (a) findings doc reflects the above slices, (b) channel + sample-rate + duration + first-W checks, (c) mid-window equal-length Moshi check, (d) this protocol filed in `docs/`. Phase 1 requires a separate go/no-go and must not start under this protocol.
