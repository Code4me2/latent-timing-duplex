# Semantic-silence experiments

This directory follows [the locked protocol](../../docs/SILENCE_GRAFT.md) and
[data-labeling manual](../../docs/DATA_LABELING_PROTOCOL.md). No model-control
efficacy result exists yet.

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
annotation exports are not committed.

R2 frozen-probe code will be added on the next stacked branch. Moshi replay is
explicitly deferred until the annotation and representation gates pass.
