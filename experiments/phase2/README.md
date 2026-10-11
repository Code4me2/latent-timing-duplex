# Phase 2 experiment receipts

This directory preserves the exact Python scripts used for the exploratory
2026-09-25 C0/C1/C2 runs. They are snapshots, not supported package commands:
they contain `spark-61dd` absolute paths and require local Moshi, audio, cached
hidden states, and checkpoints.

- `snapshots/c0_train.py`: train the 256-D future-state predictor.
- `snapshots/c0_export_surprise.py`: export C0 prediction error.
- `snapshots/c1_train.py`: train the frozen-C0 residual injector/readout.
- `snapshots/c1_offline_eval.py`: inject/shuffle/none teacher-forced comparison.
- `snapshots/c1_live_decode.py`: inject/shuffle/none live Moshi decode.
- `snapshots/c2_train_and_live.py`: block-31 LoRA training and live comparison.

`run-manifest.json` records source paths, hashes, headline results, and whether
a gate measured execution or efficacy. Do not commit generated checkpoints,
audio, model weights, logs, or full JSONL outputs here.

Read [`docs/PHASE2_EXPLORATORY.md`](../../docs/PHASE2_EXPLORATORY.md) before
reusing any result. New experiments should be written repo-first with caller-
supplied paths and a held-out efficacy gate; do not extend these snapshots.
