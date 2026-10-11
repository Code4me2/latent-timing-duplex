# Causal semantic-silence graft protocol

This protocol implements R1 of [RESEARCH_ROADMAP.md](RESEARCH_ROADMAP.md).
The operational collection manual is
[DATA_LABELING_PROTOCOL.md](DATA_LABELING_PROTOCOL.md). It does not claim that
silence semantics are already recoverable.

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
`available_at_s <= model_cutoff_s`.

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
features. `wait_for_user_resume=true` is mutually exclusive with an onset
interval; otherwise the earliest acceptable onset must not precede the
annotated checkpoint.

## Pilot and split lock

Sample 60–90 episodes, balanced across corpus and duration strata, with at most
five episodes from one source conversation. Three annotators label every item.
Lock development and untouched test partitions by `lineage_id` before model
training. A source conversation or podcast episode may appear in only one
partition.

Continue with the five actions only if Krippendorff α for primary action is at
least 0.80. Otherwise collapse once to `WAIT / RESPOND / REPAIR` and repeat the
reliability check. Do not respond to persistent disagreement by scaling the
model.

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
split requires an explicit flag after the analysis is locked.

## Explicit exclusions

The current code does not include a Moshi token override, residual adapter,
LoRA, annotation UI, or live decoder. Those remain gated by the roadmap.
