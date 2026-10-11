# Causal semantic-silence graft protocol

This protocol is the R1 scaffold from
[RESEARCH_ROADMAP.md](RESEARCH_ROADMAP.md). The operational collection manual
is [DATA_LABELING_PROTOCOL.md](DATA_LABELING_PROTOCOL.md). The R1 gate remains
open until a real pilot agreement receipt passes; this scaffold does not claim
that silence semantics are already recoverable.

## Unit of analysis

A silence episode begins at a user-speech offset and ends at the next user
onset or session boundary. Decisions are evaluated at causal checkpoints from
`configs/silence.yaml`.

Each checkpoint records:

- `model_cutoff_s`: the latest time visible to the model;
- `elapsed_s`: evidence about duration, never a semantic label;
- `context`: `joint_silence` or `assistant_speaking`;
- source and conversation/podcast lineage;
- right-censoring and technical quality.

`silence_end_s` is outcome metadata. It may be used to score a prediction, but
must never enter checkpoint features. Every extracted feature must declare
`available_at_s <= model_cutoff_s`. Reporting strata use only checkpoint elapsed
time: `<1s`, `1–<3s`, and `>=3s`; realized silence duration never determines a
stratum.

Assistant-speaking checkpoints are retained as LISTEN controls but excluded
from joint-silence semantic metrics.

## Annotation record

One JSON object per annotator and checkpoint:

```json
{
  "episode_id": "conversation:00012",
  "checkpoint_elapsed_s": 0.96,
  "annotator": "a1",
  "action": "wait",
  "tags": ["floor_holding", "hesitation"],
  "acceptable_onset": {
    "earliest_elapsed_s": null,
    "latest_elapsed_s": null,
    "wait_for_user_resume": true
  },
  "quality": "clean",
  "ratable": true
}
```

Primary actions:

- `wait`: user plausibly retains the floor; do not start a full response;
- `ack`: a brief acknowledgment is appropriate without taking the floor;
- `respond`: the assistant may take the floor;
- `repair`: check comprehension, connection, or whether the user is present;
- `unratable`: insufficient or corrupted evidence.

Secondary tags are defined in
`latent_timing_duplex.silence.annotations.ALLOWED_TAGS`. They are descriptive,
not additional primary classes.

The primary action must be submitted from prefix-only context and locked before
later user audio is revealed. Future context may support separately named
retrospective tags, but cannot revise the primary action or enter model
features. Acceptable onset is normative interval-censored assistant-entry
supervision, not recorded-partner imitation. Recorded partner onset is outcome
metadata only. `wait_for_user_resume=true` is mutually exclusive with an onset
interval; otherwise the earliest acceptable onset must not precede the
annotated checkpoint. At the same nearest-sample boundary, user resumption
cancels assistant entry.

## Pilot and split lock

Target 75 episodes (allowed coordinator review range 60–90) from at least 30
lineages, with at most five episodes per lineage. Select one joint-silence
checkpoint per episode and balance those items across corpus and checkpoint-
elapsed-time strata. Draw only from train/development; untouched test and
challenge sets are excluded. Three annotators label every item. Lock partitions by `lineage_id` before model training. A source
conversation or podcast episode may appear in only one partition.

The pilot is audio-only: fixed 20-second prefix, fixed 8-second user-only reveal
clipped at session end, no transcript, a rounded cutoff plus exact rounded
duration frame counts, and no gain change (`gain_db: 0.0` at
conversation/session scope). Prefixes and coordinator-only reveals are rendered
to separate new-only, `READY`-marked roots from source-WAV hashes. Repeat 5%
of assignments (exactly 0.05 in v1). A null sentinel lock forbids a list; otherwise its
exact SHA-256 is required in config and the authenticated channel-audition
receipt.

Renderer, channel-audition, assignment, completion, and release records use a
private HMAC key of at least 32 bytes and publish its nonsecret SHA-256
fingerprint. Public queues use a distinct copied media alias per presentation
and expose no item, repeat, sentinel, source, episode, or raw-annotator linkage.
Release requires authenticated completion rows with locked annotation hashes.

Use nominal Krippendorff α for primary action with `UNRATABLE` missing, plus
per-class positive agreement, onset endpoint disagreement, repeat consistency,
and lineage-bootstrap intervals (seed `20261009`, 10,000 replicates). Continue
with the original taxonomy only if α is at least 0.80. Otherwise collapse
exactly as `ACK→RESPOND`, `WAIT→WAIT`, `RESPOND→RESPOND`, and `REPAIR→REPAIR`.
A failed five-class pilot cannot be reused for collapsed reliability; run a new
pilot. Do not respond to persistent disagreement by scaling the model.

## Gate simulator

`SilenceGate` starts every joint-silence interval in HOLD. A non-WAIT action
requires a configured probability threshold for consecutive frames. Any user
or assistant activity resets it to LISTEN. This is a deterministic replay
component, not a learned controller and not yet a Moshi actuator.

## Frozen-probe screening

`experiments/silence_graft/train_probe.py` compares acoustic/duration, frozen
hidden state, hidden velocity, predicted latent, and available negative-control
features using a fixed-width linear probe. It trains on `train`, evaluates
`development` by default, and bootstraps paired differences by lineage. Input
archives with `available_at_s > model_cutoff_s` are rejected.

The probe report cannot by itself pass the R2 efficacy gate: acceptable-onset
and interruption outcomes still require replay. Opening the untouched test
split requires both an analysis-lock receipt and the explicit test flag.

## Explicit exclusions

The current code does not include a Moshi token override, residual adapter,
LoRA, annotation UI, or live decoder. Those remain gated by the roadmap.
