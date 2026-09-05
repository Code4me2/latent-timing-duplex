# CS 199 Phase 1 silence / collapse diagnostic
**Latent Predictive Objectives for Timing Control in Full-Duplex Dialogue Models**  
Velvet Reichman · Indika Walimuni · CCSF CS 199 · 2026-09-03/04 (PT)

## Status
**Null diagnostic** (not a Phase 1 freeze, not Phase 2, not a timing win).
Spark-side check on `spark-61dd` of whether JEPA surprise *flatlines* in
mutual silence on the same mid-180 windows used for RQ2. The turn-event
ranking stays in [PHASE1_RQ2_INTERIM.md](PHASE1_RQ2_INTERIM.md). This note
does not upgrade that ranking.

**Collapse criterion: NOT MET (0/57).**

## What ran
| Item | Value |
|---|---|
| Hardware | `spark-61dd` |
| Date | 2026-09-03/04 PT |
| Window | mid-180 (center crop, exact-W, T=2250 @ 80 ms / **12.5 Hz**) |
| Sessions | **57** mid-180 sessions |
| Activity masks | stereo **channel-energy** VAD (same proxy family as the RQ2 evening-PT addendum; not gold, not ASR) |
| Align | surprise series aligned to the 12.5 Hz chunk grid |
| Predictor | Spark H=12 frames (nominal 1 s) |
| λ | 0.01 primary; λ=0 reconstruction / collapse-control reference |

The question is whether mean surprise in **mutual silence** is substantially
lower than in **active** speech (a representation flatline / collapse
proxy). Near-turn-edge and long-silence (≥2 s) bins are reported as
context, not as extra primaries.

## Primary table — mean surprise by activity bin
H=12, mid-180, n=57, channel-energy masks.

| Bin | λ=0.01 | λ=0 |
|---|---|---|
| mutual_silence | 0.337 | 0.180 |
| active | 0.345 | 0.187 |
| silence / active ratio | **0.975** | **0.964** |
| near_turn_edge | 0.376 | 0.211 |
| long silence (≥2 s) | 0.295 | 0.145 |

Sessions meeting a collapse / flatline criterion: **0 / 57**.

## Interpretation
1. **No evidence of representation flatline in mutual silence** under this
   proxy. Silence/active ratios sit just below 1 (0.975 at λ=0.01; 0.964
   at λ=0). Surprise in mutual silence is *almost the same* as in active
   speech, not a collapse.
2. **Long silence is lower, not a flatline.** ≥2 s silence means (0.295 /
   0.145) are below the mutual-silence means but still far from a
   zeroed-out representation. Do not read that as collapse.
3. **Surprise is slightly elevated near turn edges** (0.376 vs active
   0.345 at λ=0.01; 0.211 vs 0.187 at λ=0). That is a directional
   observation on this proxy, not a turn-event metric and not a detector.
4. **λ=0 vs λ=0.01.** Absolute surprise is smaller at λ=0 (reconstruction
   reference). The *ratio* is the same story at both λ. This is not a
   regularizer win and is not the RQ2 AUROC table.

## Explicit non-claims
- **Proxies ≠ gold.** Channel-energy activity masks are stereo RMS
  intervals, not official turn annotations and not ASR transcripts.
  Mutual silence / active / near-edge bins inherit that.
- **Not Phase 2.** Heads only; Moshi / BayLing stay frozen. No policy
  conditioning, no LoRA, no generation-time steering.
- **Not a timing win by itself.** A null collapse check does not license
  a timing-control claim, a detector, or an upgrade of the RQ2 *weakly
  predictive on this proxy* ranking.
- **Different question from RQ2.** Turn-event AUROC/AUPRC live in
  [PHASE1_RQ2_INTERIM.md](PHASE1_RQ2_INTERIM.md). Reconstruction /
  collapse-control at λ=0 was already flagged there as a different
  question; this note is that diagnostic, not a second primary.
- **Not a Phase 1 science freeze.** Point estimates on one proxy mask,
  one window, two λ. CIs and per-session dumps live on Spark; this note
  does not invent them.

## Related — EVENT_HORIZON_GRID (RQ2, not collapse)
Pooled H=12 λ=0.01 `turn_shift` AUROC CIs (B=10000; proxies not gold)
live in [PHASE1_RQ2_INTERIM.md](PHASE1_RQ2_INTERIM.md) under
**EVENT_HORIZON_GRID**. Strongest offline edge at 0.5–1.0 s; weakens at
2.0 s. That grid is a turn-event ranking, **not** this collapse
diagnostic, and is **not a timing win**.

External sibling (not vendored):

```
/home/velvet/cs199-phase1-work/eval/windows/mid180/EVENT_HORIZON_GRID.md
```

## Artifact paths (spark-61dd, external)
This repository does not vendor the JSON or the Spark write-up. CI must
not open these paths.

```
/home/velvet/cs199-phase1-work/eval/windows/mid180/SILENCE_COLLAPSE.md
/home/velvet/cs199-phase1-work/eval/windows/mid180/SILENCE_COLLAPSE.json
```
