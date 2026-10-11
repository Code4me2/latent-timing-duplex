# Semantic-silence dataset and labeling protocol

**Status:** draft for review; do not begin the full annotation run until the
pilot section is approved.  
**Protocol version:** `semantic-silence-labels/v1`  
**Related:** [SILENCE_GRAFT.md](SILENCE_GRAFT.md),
[RESEARCH_ROADMAP.md](RESEARCH_ROADMAP.md)

## 1. Objective

At a causal checkpoint during user inactivity, label the action appropriate for
a duplex assistant:

- `WAIT`: continue listening;
- `ACK`: make a brief acknowledgment without taking the floor;
- `RESPOND`: take the floor with a contentful response;
- `REPAIR`: check comprehension, connection, or presence;
- `UNRATABLE`: evidence is technically or contextually unusable.

The primary label is an action judgment, not a claim about the user's internal
mental state. Silence duration is an observed covariate, never a semantic label.

## 2. Causal labeling rule

Primary action labels must use only information available through
`model_cutoff_s`. Future user resumption and the recorded partner's behavior are
not visible until after the primary answer is submitted and locked.

Use two passes:

1. **Prefix pass:** label action, confidence, acceptable onset, prefix tags, and
   technical quality from causal context.
2. **Reveal pass:** show later **user-channel** audio to collect retrospective
   tags and check quality. Never revise or overwrite the prefix answer.

If an annotator handles multiple checkpoints from one episode, all prefix-pass
answers for that episode must be locked before any continuation is revealed.
The primary model target is always the prefix-pass action.

## 3. Source eligibility and governance

Use naturally occurring speaker-separated dialogue for the primary dataset.
Synthetic pauses may be used for software tests, not as semantic training or
test examples.

Before extraction, record for each source:

- corpus and version;
- license/access terms and allowed derived artifacts;
- language;
- channel/speaker mapping;
- sample rate and timing reference;
- transcript and VAD provenance;
- source conversation or podcast lineage;
- known edits, dropouts, or channel contamination.

Do not transfer or commit restricted media. Do not place names or other direct
identifiers in item IDs. Do not infer sensitive demographic attributes.
Human-human corpora and human-assistant corpora must remain separately
reportable because appropriate partner behavior may differ.

## 4. Candidate construction

A candidate begins at a user-speech offset and ends at the next user onset or
the session boundary. Generate decision checkpoints from
`configs/silence.yaml`:

```text
0.24, 0.48, 0.96, 2.0, 3.0, 5.0, 10.0 seconds
```

Do not strip breaths, fillers, laughter, or hesitation sounds before candidate
construction. They may be interactionally meaningful.

Input two-channel VAD JSONL:

```json
{
  "session_id": "conversation-001",
  "lineage_id": "source-conversation-001",
  "source": "duplexchat",
  "duration_s": 92.4,
  "user_vad": [[0.4, 3.7], [4.5, 8.1]],
  "assistant_vad": [[9.0, 12.2]]
}
```

Build the locked manifest:

```bash
python experiments/silence_graft/build_manifest.py \
  --vad /private/data/vad.jsonl \
  --output /private/data/silence_manifest.jsonl
```

The builder writes a companion lock file containing input/output hashes and
sampling parameters. Annotate only `context == "joint_silence"` rows.
Assistant-speaking checkpoints remain LISTEN controls.

`silence_end_s` and `right_censored` are outcome metadata. They must not be
shown during the prefix pass or included in model features.

## 5. Split and sample before annotation

Assign splits by `lineage_id`, not by clip, checkpoint, or session window:

```text
train        70%
development  15%
test         15%
```

Requirements:

- one source conversation/podcast appears in one split only;
- sample no more than five episodes from a source lineage;
- balance the pilot across corpus and duration strata;
- retain a natural-prevalence untouched test set;
- report any long-pause or rare-action enrichment as a separate challenge set;
- keep test media and labels unavailable to the model-development process.

Multiple checkpoints from one episode are correlated. Evaluation and
uncertainty must group by source lineage, not treat checkpoints as independent.

## 6. Annotation media

### 6.1 Prefix clip

Render approximately 15–30 seconds of context ending exactly at
`model_cutoff_s`:

- preserve both channels and speaker identity;
- include the full elapsed silence up to the checkpoint;
- include transcript text only through the cutoff, if transcript is shown;
- do not expose numeric pause duration, next-speaker identity, or future text;
- do not trim VAD gaps;
- do not normalize each clip independently or remove quiet interactional cues.

Use one consistent conversation-level gain policy. The interface should mark
which channel is the prospective user and which is the assistant/partner.

### 6.2 Reveal clip

After all applicable prefix labels are locked, reveal the user channel from the
checkpoint through the next 3–10 seconds or the next user utterance. Initially
mask the recorded partner's future response so annotators do not imitate it.
Recorded assistant onset is measured automatically from the source timeline.

### 6.3 Assignment and blinding

Randomize items and hide corpus filenames, source IDs, model scores, observed
outcomes, and other annotators' answers. Avoid assigning adjacent checkpoints
from one episode consecutively. Record the clip hash and protocol version with
each response.

## 7. Primary decision instructions

Prompt:

> At the marked instant, what should a helpful duplex assistant do?

| Action | Operational definition |
|---|---|
| `WAIT` | A full response now would likely interrupt the user; continue listening. |
| `ACK` | A short continuer or acknowledgment is appropriate without claiming the floor. |
| `RESPOND` | It is appropriate to take the floor with content. |
| `REPAIR` | Evidence supports checking comprehension, connection, or whether the user is present. |
| `UNRATABLE` | Missing context, corrupted audio, or another technical issue prevents a meaningful judgment. |

Rules:

- judge appropriate behavior, not what the recorded partner eventually did;
- do not map short pauses automatically to `WAIT`;
- do not map long pauses automatically to `REPAIR`;
- use low confidence or an alternate acceptable action for ordinary ambiguity;
- reserve `UNRATABLE` for unusable evidence, not difficult but valid examples;
- `ACK` must be plausibly brief and non-floor-taking; otherwise choose
  `RESPOND`.

Collect confidence on a five-point scale and, optionally, one
`also_acceptable` action. The forced primary action remains the evaluation
label; confidence and alternatives preserve genuine ambiguity.

## 8. Acceptable onset

For a ratable item, record either:

- earliest and latest acceptable assistant onset, in seconds after silence
  onset; or
- `wait_for_user_resume: true` when speaking before resumption would be
  inappropriate.

An onset window cannot begin before the annotated checkpoint. A `WAIT` label
may still have a later acceptable onset window if waiting is appropriate now
but indefinite waiting is not. `UNRATABLE` items have no onset target.

Example:

```json
{
  "earliest_elapsed_s": 1.4,
  "latest_elapsed_s": 2.3,
  "wait_for_user_resume": false
}
```

## 9. Secondary tags

Prefix-pass tags are multi-label and exploratory:

```text
hesitation
floor_holding
deliberation
turn_yielding
response_solicitation
affective_stance
disengagement
comprehension_difficulty
technical_uncertainty
```

The reveal pass may collect the same tags under a separate
`retrospective_tags` field. Never replace prefix tags with retrospective tags.
Semantic tags are secondary analyses; action is the primary supervised target.

## 10. Quality labels

Use the existing primary quality values:

```text
clean
breath_or_nonlexical
noise_or_clipping
dropout_mute_or_edit
```

Also retain detailed flags outside the primary enum when available:

- transcript mismatch;
- speaker/channel contamination;
- uncertain VAD boundary;
- language mismatch;
- missing preceding context.

Technical dropout must not automatically become semantic `REPAIR`.

## 11. Raw annotation record

One JSON object per annotator and checkpoint:

```json
{
  "schema": "semantic-silence-labels/v1",
  "episode_id": "conversation-001:00012",
  "checkpoint_elapsed_s": 0.96,
  "annotator": "a1",
  "action": "wait",
  "confidence": 4,
  "also_acceptable": ["ack"],
  "tags": ["floor_holding", "hesitation"],
  "acceptable_onset": {
    "earliest_elapsed_s": null,
    "latest_elapsed_s": null,
    "wait_for_user_resume": true
  },
  "quality": "clean",
  "ratable": true,
  "clip_sha256": "..."
}
```

Store reveal-pass judgments separately or under explicitly named
`retrospective_*` fields. Store automatically measured user resumption,
assistant onset, overlap, and censoring in `outcomes.jsonl`, not in the semantic
annotation record.

## 12. Annotator preparation and quality control

Annotators must be fluent in the conversation language and use headphones.
Before the pilot:

1. review the action definitions and boundary cases;
2. complete 15–20 training examples with explanations;
3. pass audio/channel and obvious-action sentinel checks;
4. confirm they understand that the prefix answer cannot use future context.

During labeling, include sparse repeated items to measure intra-annotator
consistency and sentinel items to detect playback or attention failures. Do not
use model predictions as hidden gold answers.

## 13. Pilot, agreement, and collapse rule

Pilot 60–90 episodes from at least 30 source lineages, with every selected
checkpoint labeled independently by three annotators. This will usually yield
roughly 150–250 checkpoint items.

Before adjudication, report:

- Krippendorff's alpha for primary action;
- primary-action confusion matrix;
- per-class positive agreement;
- onset-window disagreement;
- agreement by corpus and duration stratum;
- intra-annotator consistency and unratable rate.

Continue with five actions only if primary-action alpha is at least 0.80. If it
is lower, revise examples and collapse once to `WAIT / RESPOND / REPAIR`.
Repeat the reliability check on a new pilot sample. If agreement remains below
the threshold, stop semantic scaling rather than compensating with a larger
model.

Preserve every raw judgment. Produce adjudicated labels only after raw
agreement is reported. When all three primary labels differ, require expert
adjudication; otherwise retain the majority label. Preserve the empirical label
distribution for later soft-target experiments.

## 14. Main collection and power

Use pilot prevalence and lineage-bootstrap variance to size the main dataset.
Do not select a large arbitrary checkpoint count. A plausible starting range is
600–1,000 episodes from at least 200 source lineages, including 150–200
naturalistic test episodes, but the pilot power analysis decides the final
number.

Rare `ACK` and `REPAIR` cases may be enriched in training and in a separately
reported challenge set. Do not silently balance the natural test set; report
natural prevalence and any sampling weights.

## 15. Private dataset layout

```text
semantic-silence-data/
  manifest.jsonl
  manifest.jsonl.lock.json
  splits.json
  clips/
    prefix/
    reveal_user/
  assignments/
  annotations/
    raw/
      annotator-a.jsonl
      annotator-b.jsonl
      annotator-c.jsonl
    adjudicated.jsonl
  outcomes.jsonl
  DATASET_CARD.md
```

Media, private paths, annotations containing restricted source identifiers, and
untouched test labels remain outside Git. Commit only schemas, protocol text,
aggregate statistics, and cryptographic receipts permitted by source terms.

## 16. Handoff acceptance checklist

Data is ready for frozen-probe work only when:

- [ ] source rights and dataset card are complete;
- [ ] manifest and clip hashes validate;
- [ ] all primary labels were collected prefix-only;
- [ ] three independent labels exist per pilot/evaluation item;
- [ ] raw agreement was computed before adjudication;
- [ ] the five-class or collapsed taxonomy passed the locked agreement gate;
- [ ] lineage-disjoint splits are verified;
- [ ] test labels are sealed;
- [ ] outcomes are stored separately from causal features;
- [ ] technical/unratable exclusions are documented;
- [ ] corpus and duration-stratum counts are reported.
