# Semantic-silence experiments

This directory follows [the locked protocol](../../docs/SILENCE_GRAFT.md) and
[data-labeling manual](../../docs/DATA_LABELING_PROTOCOL.md). R1 is scaffolded,
but its gate remains open until a real pilot agreement receipt passes. No
model-control efficacy result exists yet.

## R1: build a checkpoint manifest

```bash
python experiments/silence_graft/build_manifest.py \
  --vad /path/to/two_channel_vad.jsonl \
  --output /path/to/silence_manifest.jsonl \
  --hmac-key /private/keys/pilot-hmac.key
```

Each VAD row needs explicit `session_id`, `lineage_id`, `source`, `duration_s`,
and user/assistant interval lists such as
`user_vad: [[start_s, end_s], ...]`. Pass `--config PATH_OR_NAME` to use a
non-default config.

The command writes the full JSONL plus derived `.pilot.jsonl`, `.splits.json`,
and HMAC-authenticated `.jsonl.lock.json` artifacts, and refuses to overwrite
any of them. The mode-`0600`, non-symlink HMAC key is mandatory. Raw
media, private paths, and annotation exports are not committed. Pilot selection
targets 75 episodes (review range 60–90) from at least 30 train/development
lineages, capped at five episodes per lineage. It selects one joint-silence
checkpoint per episode and balances that checkpoint's elapsed-time stratum
(`<1s`, `1–<3s`, `>=3s`), never realized silence duration. Untouched test and
challenge lineages are excluded.

Pilot media is audio-only: a 20-second prefix and an 8-second user-only reveal
clipped at session end, nearest-sample boundaries, conversation/session
`gain_db: 0.0`, and no transcript. Exact agreement, repeat, collapse, onset, and
receipt rules are locked in `configs/silence.yaml` and the labeling manual.

## R1: render blinded pilot clips

Keep the media index, HMAC key, and both output packages outside Git. The HMAC
key must be cryptographically random per pilot, contain at least 32 bytes, and
be a regular non-symlink file with no group/other permissions (`chmod 600`).
The UTF-8 JSONL media index has exactly one row for every pilot session and no
other rows:

```json
{"session_id":"conversation-001","wav_path":"/private/audio/conversation-001.wav","wav_sha256":"<lowercase SHA-256>","user_channel":0,"assistant_channel":1}
```

Paths are absolute. Channel numbers are zero-based. The renderer opens each unique regular source without following symlinks,
copies and hashes it once into a private mode-`0600` snapshot, and renders only
that immutable snapshot. It writes prefix channels in user/assistant order.

```bash
python experiments/silence_graft/render_clips.py \
  --pilot-manifest /private/data/silence_manifest.pilot.jsonl \
  --lock-receipt /private/data/silence_manifest.jsonl.lock.json \
  --media-index /private/data/pilot_media.jsonl \
  --prefix-output-dir /private/data/pilot-prefix \
  --coordinator-output-dir /private/data/pilot-coordinator \
  --hmac-key /private/keys/pilot-hmac.key
```

Both target directories must not exist. They are exclusively reserved mode
`0700`, populated with mode-`0600` files from staging, and receive `READY`
last. Consumers reject
packages without `READY`. Receipt HMACs use the stdlib-only helpers in
`receipts.py`: sorted compact UTF-8 JSON over the receipt without
`hmac_sha256`, with `key_fingerprint` included. `pilot-prefix/` contains only prefix WAVs and
`READY`; it never contains reveals or source metadata. `pilot-coordinator/`
contains user-only reveals, the coordinator index, private source/channel map,
render records, authenticated renderer receipt, and `READY`. Prefix/reveal
boundaries use the rounded cutoff frame and exact rounded configured duration
in frames. Every receipt includes the nonsecret SHA-256 key fingerprint.

## R1: build assignments and release reveals

Before assignment, the coordinator must listen to both declared channels for
every pilot session. Sign the private receipt with `sign_receipt` from
`experiments.silence_graft.receipts` using the renderer key:

```json
{
  "schema": "semantic-silence-channel-audition/v1",
  "protocol_version": "semantic-silence-pilot/v1",
  "auditor_identity": "coordinator-opaque-id",
  "audited_at": "2026-10-09T12:00:00Z",
  "sha256": {
    "config": "...",
    "coordinator_media_index": "...",
    "private_source_mapping": "...",
    "renderer_receipt": "..."
  },
  "sessions": [{
    "session_id": "conversation-001",
    "wav_sha256": "...",
    "user_channel": 0,
    "assistant_channel": 1,
    "user_channel_auditioned": true,
    "assistant_channel_auditioned": true
  }],
  "key_fingerprint": "...",
  "hmac_sha256": "..."
}
```

The annotator file is a JSON string list (or `{"annotators": [...]}`) with at
least three distinct private IDs and enough reviewers to keep each reviewer to one original item per lineage. Reuse the renderer key:

```bash
python experiments/silence_graft/build_assignments.py build \
  --prefix-root /private/data/pilot-prefix \
  --coordinator-root /private/data/pilot-coordinator \
  --annotator-ids /private/data/annotators.json \
  --hmac-key /private/keys/pilot-hmac.key \
  --audition-receipt /private/data/channel-audition.json \
  --annotator-output-dir /distribution/pilot-assignments \
  --coordinator-output-dir /private/data/pilot-assignments
```

The builder verifies both ready renderer packages, renderer/audition HMACs,
every output hash, source/channel binding, config/protocol link, and opaque item
HMAC. It assigns three reviewers per item while preventing any reviewer from seeing two original items from one lineage. The two new assignment roots must not be equal or nested. The annotator
root contains only `annotators/<opaque>/queue.jsonl`, local presentation media,
and `READY`; each opaque directory is independently distributable. Public
queues contain only `assignment_id`, `presentation_id`, a subpackage-relative
presentation alias/path, and schema/protocol—no media hash. Every presentation,
including a repeat, gets a distinct copied WAV alias. Item, repeat, sentinel,
source, episode, hash, and raw annotator linkage stays in the coordinator maps.
The coordinator root has those maps, the authenticated assignment receipt, and
`READY`, but no media. The receipt authenticates separate `annotator_outputs`
and coordinator `outputs` file maps. Protocol v1 requires a repeat fraction of
0.05. A null sentinel config forbids a list; otherwise the config must be its
exact lowercase SHA-256 and the authenticated audition receipt must bind the
same list.

Each completion row contains the exact private `assignment_id`,
`presentation_id`, `annotator_id`, `item_id`, and `prefix_sha256`, plus a
lowercase `annotation_sha256` for the locked response. Its completion receipt
hashes the records and assignment receipt and is HMAC-signed with the same key.

```bash
python experiments/silence_graft/build_assignments.py release \
  --annotator-root /distribution/pilot-assignments \
  --coordinator-root /private/data/pilot-assignments \
  --completions /private/data/completions.jsonl \
  --completion-receipt /private/data/completions.receipt.json \
  --hmac-key /private/keys/pilot-hmac.key \
  --output-dir /private/data/reveals
```

Release verifies both assignment roots, both exact receipt-bound file sets,
all receipt HMACs, and the exact private reveal-map hash before use. It fails
unless every presentation is complete, then emits opaque per-annotator
subpackages with local presentation-specific reveal aliases. Public reveal
queues expose no item/QC linkage or reveal hash; the release receipt binds the
media. All package targets are new-only and receive `READY` last. There is
deliberately no annotation UI.

## R1: compile, report, and adjudicate annotations

Collect one strict `semantic-silence-prefix-response/v1` JSONL row for every
presentation, then compile it against both unchanged assignment packages:

```bash
python experiments/silence_graft/compile_annotations.py report \
  --annotator-root /distribution/pilot-assignments \
  --coordinator-root /private/data/pilot-assignments \
  --raw-responses /private/data/raw-responses.jsonl \
  --hmac-key /private/keys/pilot-hmac.key \
  --config silence.yaml \
  --output-dir /private/data/pilot-report
```

The report command verifies both `READY` packages, their exact authenticated
file maps, and every assignment/presentation/annotator join. It emits validated
private annotations, release-compatible completion records and receipt, the
locked agreement report, and an authenticated agreement receipt. Production uses the configured 10,000 lineage bootstraps. The gate additionally
requires 60–90 items (target 75), at least 30 lineages, no more than five items per
lineage, at least 80% pairable items across at least 30 lineages, and at least
90% valid alpha bootstrap replicates. A Python bootstrap-count override is
explicitly nonconfirmatory and can never pass the gate; the CLI has no override.
A failed gate records `new_collapsed_pilot_required`; it never rewrites this
pilot.

After a passing gate:

```bash
python experiments/silence_graft/compile_annotations.py adjudicate \
  --validated-annotations /private/data/pilot-report/validated_annotations.jsonl \
  --agreement-receipt /private/data/pilot-report/agreement_receipt.json \
  --hmac-key /private/keys/pilot-hmac.key \
  --output-dir /private/data/adjudicated
```

For each no-majority item, also pass `--expert-decisions` and
`--expert-receipt`. The decision JSONL has exactly `item_id` and a ratable
`action`. Its signed `semantic-silence-expert-decisions-receipt/v1` receipt has
`sha256` entries for `expert_decisions`, `agreement_receipt`, and
`validated_annotations`. Decisions must cover exactly the unresolved items and select an action observed
for that item. Majority-`UNRATABLE` items are authenticated as excluded. A
role-separated expert signing key remains a future operational requirement;
this patch does not treat receipt timestamps as proof of pre-reveal ordering.
The model-label output contains only causal item/source/lineage/split timing and
aggregated annotation targets; raw annotator, episode/session, reveal, and
outcome fields are excluded.

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

The default evaluates only `development`. `--evaluate-test` currently fails closed; enable it only when authenticated analysis-lock receipt verification is implemented. All feature sets use the same fixed-width linear
softmax probe; uncertainty is paired by `lineage_id`.

This script is representation screening, not a graft efficacy result. It does
not satisfy acceptable-onset or interruption gates without replay.

Moshi replay remains explicitly deferred until the annotation and
representation gates pass.
