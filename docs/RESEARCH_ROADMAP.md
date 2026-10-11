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
       native silence/no-text gate (first) → bounded adapter (later)
```

The model continues consuming both dialogue channels while waiting. It emits
model-native silence/no-text tokens rather than generating speech and muting the
waveform. A pending response is cancelled if the user resumes. No response is
pre-generated and buffered across a pause.

## Evidence phases and gates

### R0 — historical receipts

Phase 0/1 and C0/C1/C2 remain immutable evidence. Phase 1 surprise is
retrospective because it uses realized future states; C1/C2 did not establish a
semantic or aligned generation effect.

### R1 — construct and data gate (PR #9)

Deliver:

- silence episode/checkpoint schema;
- causal cutoff and feature-availability guards;
- action annotation schema: `WAIT`, `ACK`, `RESPOND`, `REPAIR`, `UNRATABLE`;
- interactional tags and technical-quality labels;
- conversation/episode-disjoint split manifest;
- deterministic gate simulator and synthetic tests.

Pilot 60–90 episodes with three annotators. Continue only if primary action
labels reach Krippendorff α ≥ 0.80. If not, collapse once to
`WAIT / RESPOND / REPAIR`; stop finer semantic claims if reliability remains
below the threshold.

### R2 — frozen representation gate (PR #10)

On causal checkpoints compare equal-protocol probes over:

1. elapsed silence + VAD/energy;
2. energy change/onset;
3. hidden-state velocity;
4. frozen `h_t + elapsed`;
5. `h_t + predicted_latent + elapsed`;
6. untrained, target-shuffled, and duration-bin-shuffled latent controls.

Primary: macro-AUPRC over actionable labels. Timing: competing-risk log loss,
acceptable-onset hit rate, premature interruption, and late/missed response.
Use paired bootstrap by conversation/podcast episode and report short, medium,
long, and corpus strata.

Representation GO requires all of:

- macro-AUPRC +0.05 over `h_t + elapsed` and the strongest acoustic/duration
  baseline, paired 95% CI lower bound > 0;
- acceptable-onset hit rate +0.05 over `h_t` and shuffled-latent controls;
- premature interruption worsens by no more than 0.02 absolute;
- positive effects in short and long strata;
- causal-cutoff, annotation-reliability, and technical-silence checks pass.

If the predicted latent adds no value over `h_t + elapsed`, remove it.

### R3 — oracle and predicted native gate (future PR #11)

Created only after R1/R2 gates pass. Replay held-out user audio with identical
seeds under:

- base model;
- oracle annotated gate;
- duration-only gate;
- predicted semantic gate;
- shuffled semantic gate.

First actuator: native silence/no-text token generation while `WAIT`. Do not
post-hoc mute decoded audio. Continue only if oracle gating improves acceptable
onset without collapse or content degradation, then require predicted gating to
beat duration-only and shuffled controls.

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
