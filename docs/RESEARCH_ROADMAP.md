# Research roadmap: semantic silence in full-duplex dialogue

**Direction locked:** 2026-10-09  
**Goal:** make deliberate waiting an explicit, causal dialogue action so an
existing duplex model can distinguish a user holding the floor from yielding,
requesting acknowledgment, or needing repair.

Silence duration is evidence, not meaning. Short/medium/long are reporting
strata. The primary construct is the action appropriate at a causal checkpoint.

## Architecture path

```text
frozen duplex state h_t + elapsed silence + causal activity/quality
                              │
                              ▼
             silence-state and competing-hazard sidecar
              WAIT / ACK / RESPOND / REPAIR + hazards
                              │
                              ▼
 characterized Mimi silence codes + text padding (first) → bounded adapter (later)
```

The model continues consuming both dialogue channels while waiting. Moshi has
no dedicated audio silence token: the actuator must emit a characterized valid
Mimi silence-code sequence plus text padding rather than generate speech and
mute the waveform. A pending response is cancelled if the user resumes. No response is
pre-generated and buffered across a pause.

## Evidence phases and gates

### R0 — historical receipts

Phase 0/1 and C0/C1/C2 remain immutable evidence. Phase 1 surprise is
retrospective because it uses realized future states; C1/C2 did not establish a
semantic or aligned generation effect.

### R1 — construct and data gate

Deliver:

- silence episode/checkpoint schema;
- causal cutoff and feature-availability guards;
- action annotation schema: `WAIT`, `ACK`, `RESPOND`, `REPAIR`, `UNRATABLE`;
- interactional tags and technical-quality labels;
- lineage-disjoint split manifest;
- deterministic gate simulator and synthetic tests.

R1 is scaffolded but its gate is open until a real pilot agreement receipt
passes. Target 75 train/development episodes (allowed review range 60–90) from
at least 30 lineages, capped at five episodes per lineage; untouched test and
challenge items are excluded. The audio-only pilot uses a 20-second prefix and
8-second user-only reveal with no transcript or gain change.

Continue only if nominal primary-action Krippendorff α (with `UNRATABLE`
missing) is at least 0.80. Report per-class positive agreement, onset endpoint
disagreement, 5% repeat consistency, and 10,000 lineage-bootstrap replicates
with seed `20261009`. If needed, collapse `ACK→RESPOND` while preserving
`WAIT`, `RESPOND`, and `REPAIR`, and test reliability on a new pilot; the failed
five-class pilot cannot be reused.

### R2 — frozen representation gate

On causal checkpoints compare equal-protocol probes over:

1. elapsed silence + VAD/energy;
2. energy change/onset;
3. hidden-state velocity;
4. frozen `h_t + elapsed`;
5. `h_t + predicted_latent + elapsed`;
6. untrained, target-shuffled, and duration-bin-shuffled latent controls.

Primary: macro-AUPRC over actionable labels. Timing: competing-risk log loss,
acceptable-onset hit rate, premature interruption, and late/missed response.
Use paired bootstrap by conversation/podcast episode and report corpus and
checkpoint-elapsed-time strata (`<1s`, `1–<3s`, `>=3s`), never strata based on
realized silence duration. Test evaluation requires an analysis-lock receipt.

Representation GO requires all of:

- macro-AUPRC +0.05 over `h_t + elapsed` and the strongest acoustic/duration
  baseline, paired 95% CI lower bound > 0;
- acceptable-onset hit rate +0.05 over `h_t` and shuffled-latent controls;
- premature interruption worsens by no more than 0.02 absolute;
- positive effects in short and long strata;
- causal-cutoff, annotation-reliability, and technical-silence checks pass.

If the predicted latent adds no value over `h_t + elapsed`, remove it.

### R3 — oracle and predicted native gate

Created only after R1/R2 gates pass. Replay held-out user audio with identical
seeds under:

- base model;
- oracle annotated gate;
- duration-only gate;
- predicted semantic gate;
- shuffled semantic gate.

First actuator: characterized valid Mimi silence-code generation plus text
padding while `WAIT`. Do not post-hoc mute decoded audio. Acceptable onset is normative interval-censored
assistant-entry supervision; recorded partner onset is outcome only, and a
user-resumption tie cancels assistant entry. Continue only if oracle gating
improves acceptable onset without collapse or content degradation, then require
predicted gating to beat duration-only and shuffled controls.

### R4 — bounded pre-depformer adapter

Only if native gating works but transition or action-style quality is limiting.
Use a zero-initialized, norm-capped low-rank adapter on the depformer copy of
`h_t`; keep backbone, recurrent state, and text path frozen. Compare aligned,
off, shuffled, duration-only, and scale-matched random conditions.

### R5 — LoRA or native training

LoRA is allowed only after R4 demonstrates a specific aligned effect and a
measured capacity ceiling. The longer-term native architecture jointly trains
content generation, floor/action state, user-resumption hazard, assistant-entry
hazard, and technical-silence detection.

## Compute plan

### spark-61dd

- verify checkpoint/session lineage;
- reuse existing caches;
- extract missing causal Moshi states serially;
- run Moshi replay only after R2 passes.

### gx10

- verify CUDA/storage before use;
- train frozen probes and controls;
- run calibration and paired bootstraps;
- independently replay/evaluate after protocol lock.

Do not transfer restricted CANDOR media unless its access terms permit it.
Annotation quality, not accelerator time, is the first bottleneck.

## Stop conditions

Stop scaling if action labels are unreliable, causal probes do not beat simple
duration/acoustic controls, predicted latent adds no value over `h_t`, or
aligned gating does not beat shuffled/off controls. Preserve negative results;
do not substitute larger models for a failed construct.
