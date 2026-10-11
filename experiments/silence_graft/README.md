# Semantic-silence experiments

This directory follows [the locked protocol](../../docs/SILENCE_GRAFT.md) and
[data-labeling manual](../../docs/DATA_LABELING_PROTOCOL.md). R1 is scaffolded,
but its gate remains open until a real pilot agreement receipt passes. No
model-control efficacy result exists yet.

## R1: build a checkpoint manifest

```bash
python experiments/silence_graft/build_manifest.py \
  --vad /path/to/two_channel_vad.jsonl \
  --output /path/to/silence_manifest.jsonl
```

Each VAD row needs `session_id`, `duration_s`, and user/assistant interval lists
such as `user_vad: [[start_s, end_s], ...]`. `lineage_id` should identify the
source conversation or podcast; it defaults to `session_id`.

The command writes JSONL plus `.lock.json` hashes. Raw media, private paths, and
annotation exports are not committed. Pilot selection targets 75 episodes
(review range 60–90) from at least 30 train/development lineages, capped at five
episodes per lineage; untouched test and challenge items are excluded. Duration
strata use checkpoint elapsed time (`<1s`, `1–<3s`, `>=3s`), never realized
silence duration.

Pilot media is audio-only: a 20-second prefix and an 8-second user-only reveal
clipped at session end, nearest-sample boundaries, conversation/session
`gain_db: 0.0`, and no transcript. Exact agreement, repeat, collapse, onset, and
receipt rules are locked in `configs/silence.yaml` and the labeling manual.

## R2: screen frozen representations

Create a NumPy `.npz` with aligned arrays:

```text
session_id, lineage_id, split, action, context, elapsed_s,
model_cutoff_s, available_at_s, acoustic, hidden
```

Optional arrays are `hidden_velocity`, `predicted_latent`, `untrained_latent`,
and `shuffled_target_latent`. The archive is rejected if any feature is
available after its model cutoff.

```bash
python experiments/silence_graft/train_probe.py \
  --features /path/to/locked_features.npz \
  --output /path/to/development_report.json
```

The default evaluates only `development`. Do not use `--evaluate-test` until an
analysis-lock receipt exists; opening the untouched test split then also
requires that explicit flag. All feature sets use the same fixed-width linear
softmax probe; uncertainty is paired by `lineage_id`.

This script is representation screening, not a graft efficacy result. It does
not satisfy acceptable-onset or interruption gates without replay.

Moshi replay remains explicitly deferred until the annotation and
representation gates pass.
