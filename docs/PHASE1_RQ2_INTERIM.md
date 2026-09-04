# CS 199 Phase 1 RQ2 interim findings
**Latent Predictive Objectives for Timing Control in Full-Duplex Dialogue Models**  
Velvet Reichman · Indika Walimuni · CCSF CS 199 · 2026-09-03 (PT)

## Status
**Interim only** (not a Phase 1 freeze, not Phase 2). Primary Spark compare on
`spark-61dd` (Linux aarch64, GB10/`sm_121`) using the in-repo scorer from
harness [PR #4](https://github.com/Code4me2/latent-timing-duplex/pull/4)
(`main` @ `c4710a5`). Protocol:
[EVAL_PROTOCOL_PHASE1.md](EVAL_PROTOCOL_PHASE1.md). Phase 0 freeze numbers stay
in [PHASE0_INTERIM_FINDINGS.md](PHASE0_INTERIM_FINDINGS.md) and are not restated
here.

On this **CANDOR-only transcript-proxy** slice, JEPA surprise is *weakly*
above Moshi user-channel NLL and VAP `p(shift)` for `turn_shift` at
`h=1.0` s. That is a proxy ranking, not a gold-label or cross-corpus result.
The **CANDOR-only table below remains the first primary.** An evening-PT
**cross-corpus proxy addendum** (channel-energy VAD, not gold) is appended
after the original non-claims.

## What ran
| Item | Value |
|---|---|
| Hardware | `spark-61dd` |
| Window | mid-180 (center crop, exact-W, T=2250 @ 80 ms) |
| Seed / bootstrap | `20260903` / B=10000, session-level Efron (protocol) |
| Predictor | Spark H=12 frames (nominal 1 s; not plan half-up 13) |
| λ | **0.01 primary** (λ=0 reconstruction reference also scored) |
| Signals | `jepa:surprise`, `nll:moshi`, `vap:p_shift` on one `t_end` grid |
| Labels | CANDOR `extract*/transcription/*.csv` **proxies** (`phase1.labels`) |
| Labeled eval | **CANDOR-only n=44** |
| Dropped | one CANDOR session with **0** mid-window proxy events |
| DuplexChat | 12 sessions in the precompute tree; **no on-disk transcripts**, so **not** in the first primary table |

Gold `--labels` JSONL was not used. The first primary stays CANDOR-only.
DuplexChat joins only the evening-PT addendum, via channel-energy VAD
proxies (not gold, not ASR transcripts).

## Primary table — `turn_shift`, h=1.0 s
CANDOR-only proxy labels, n=44, H=12, λ=0.01, mid-180.

| Signal | AUROC | AUPRC |
|---|---|---|
| `jepa:surprise` | 0.632 | 0.106 |
| `nll:moshi` | 0.448 | 0.049 |
| `vap:p_shift` | 0.513 | 0.056 |

Chance AUROC is 0.5. AUPRC is prevalence-sensitive; 0.106 is still a rare-event
regime, not a high-precision detector. Moshi NLL is *below* chance on this
proxy. VAP is near chance.

## Other cells (brief; not the headline)
The same run scored the Phase 0 eval grid, including **h=0.5 s**, and the
other kinds **`barge_in`** and **`backchannel`**. Those cells are in the Spark
JSON, not copied here. They do not change the headline: this note claims only
the `turn_shift` / `h=1.0` s ranking above.

Horizon × λ grid on this same proxy (ranking only; no extra digits):

- **H=12 > H=1** (nominal 1 s head beats the 80 ms head).
- **λ=0.01 ≈ λ=0** for the turn-event ranking. The isotropic regularizer does
  not move this proxy table. Reconstruction error is smallest at λ=0; that is
  a **different question** (see non-claims).

## Interpretation
1. **Weak ranking on a proxy.** Surprise beats the two frozen baselines on
   `turn_shift` at 1.0 s (AUROC 0.632 vs 0.448 / 0.513). Call this
   *weakly predictive on this proxy*, not “timing control works.”
2. **AUPRC stays low.** 0.106 vs 0.049 / 0.056 is a gap, not a usable
   operating point. Do not advertise a detector.
3. **NLL is not a strong turn-shift score here.** 0.448 AUROC on this proxy
   is a null (or slightly anti-aligned) result for implicit next-token
   surprise. That is allowed; it is not a Moshi failure mode for generation.
4. **VAP is near chance** on the same labels (0.513). Same-timeline rule
   still holds; the labels are the weak part.
5. **Grid.** Prefer the locked primary (H=12, λ=0.01). H=1 is worse. λ=0.01
   and λ=0 agree enough that this table is not a regularizer win.

## Explicit non-claims
- **Proxy labels ≠ gold.** Speaker-change / short-burst / overlap heuristics
  from CANDOR CSVs are not official turn, backchannel, or barge-in
  annotations (`describe_label_limitations()`).
- **Not cross-corpus** (this first primary). DuplexChat (12 precomputed)
  has no on-disk transcripts in the CANDOR-only table. The evening-PT
  addendum is a separate **cross-corpus proxy** ranking, not a gold join.
- **Not Phase 2.** Heads only; Moshi / BayLing stay frozen. No policy
  conditioning, no LoRA, no generation-time steering.
- **Reconstruction at λ=0 is a different question.** Best embed-MSE at λ=0
  does not license a turn-event claim, and λ=0.01 ≈ λ=0 here anyway.
- **Not a Phase 1 science freeze.** n=44, one corpus, proxy events, one
  primary cell published as point estimates. CIs and the full grid live on
  Spark; this note does not invent them.

## Artifact paths (spark-61dd, external)
This repository does not vendor the JSON. CI must not open these paths.

```
/home/velvet/cs199-phase1-work/eval/windows/mid180/
```

Typical siblings (H / λ repeats): `h12_lam0.01.json`, `h12_lam0.json`,
`h1_lam0.01.json`, and the rest of the locked grid. Scorer:
`ltd phase1-eval` (see [EVAL_PROTOCOL_PHASE1.md](EVAL_PROTOCOL_PHASE1.md)).

Evening-PT DuplexChat channel-energy VAD (external; not vendored):

```
/home/velvet/cs199-phase1-work/eval/duplexchat_mid180_channel_energy_vad.jsonl
```

## Addendum — 2026-09-03 evening PT (cross-corpus proxy)

DuplexChat labels were unblocked with **stereo channel-energy VAD
proxies**. These are **not gold** and **not ASR transcripts**. The
CANDOR-only table above remains the first primary. This addendum is a
pooled / split **proxy** ranking on the same `turn_shift` / `h=1.0` s /
H=12 / λ=0.01 cell.

### Channel-energy VAD (DuplexChat)

| Item | Value |
|---|---|
| Method | per-channel RMS energy VAD (not gold, not ASR) |
| Frame / hop | 25 ms frame, 10 ms hop |
| Threshold | median + 3·MAD (p80 if MAD ≈ 0) |
| Post | merge gaps < 0.10 s; drop intervals < 0.08 s |
| Channels | LEFT=user, RIGHT=assistant |
| Spark path | `/home/velvet/cs199-phase1-work/eval/duplexchat_mid180_channel_energy_vad.jsonl` |
| Pool | 45 CANDOR transcript CSV proxy + 12 DC VAD via `CompositeEventSource` |
| Labeled eval | pooled **n≈56** (CANDOR n=44 + DC n=12) |

Gold `--labels` JSONL was still not used. `CompositeEventSource` prefers
the first source that has the session (CANDOR CSV proxy, then DC VAD).

### Addendum table — `turn_shift`, h=1.0 s, H=12, λ=0.01

Pooled mid-180 (CANDOR transcript proxy + DC channel-energy VAD).

| Slice | n | `jepa:surprise` AUROC/AUPRC | `nll:moshi` | `vap:p_shift` |
|---|---|---|---|---|
| pooled | ≈56 | 0.645 / 0.115 | 0.463 / 0.050 | 0.511 / 0.054 |
| CANDOR | 44 | 0.632 / 0.106 | 0.448 / 0.049 | 0.513 / 0.056 |
| DuplexChat | 12 | 0.694 / 0.148 | 0.522 / 0.054 | 0.503 / 0.046 |

λ=0 pooled surprise is 0.643 / 0.125. NLL and VAP at λ=0 match the
λ=0.01 baseline columns (those signals do not depend on λ).

The CANDOR n=44 row restates the first primary. DC n=12 is a small
exact-W slice; do not treat 0.694 as a stable corpus effect.

### Addendum interpretation

The pooled ranking is still *weakly predictive on this proxy*: surprise
0.645 / 0.115 vs NLL 0.463 / 0.050 and VAP 0.511 / 0.054. AUPRC remains
a rare-event regime. This does not upgrade the first primary and does
not license a detector or a Phase 2 start.

### Addendum non-claims

- **VAD ≠ gold.** Channel-energy intervals are stereo RMS proxies, not
  official DuplexChat turn annotations and not ASR transcripts.
- **Small n_DC=12.** The DuplexChat split is underpowered; the 0.694
  AUROC is a point estimate on twelve exact-W sessions.
- **Not Phase 2.** Heads only; Moshi / BayLing stay frozen. No policy
  conditioning, no LoRA, no generation-time steering.
- **Still weakly predictive language.** Pooled surprise is above the
  two frozen baselines on this proxy and is not “timing control works.”

Silence/collapse on the same mid-180 pool is a **different question**
(representation flatline in mutual silence, not turn-event AUROC). That
proxy diagnostic is in [PHASE1_SILENCE_COLLAPSE.md](PHASE1_SILENCE_COLLAPSE.md):
criterion not met (0/57). It does not upgrade this ranking and is not a
timing win by itself.

## Recommended next
1. Gold or audited labels (do not treat CSV proxies or channel-energy
   VAD as final).
2. Treat the DC n=12 VAD split as a small-n check, not a second primary.
3. Publish CIs from the Spark JSON if a freeze is declared.
4. Do **not** start Phase 2 from this ranking.
5. Do **not** treat the silence/collapse null
   ([PHASE1_SILENCE_COLLAPSE.md](PHASE1_SILENCE_COLLAPSE.md)) as a
   timing-control result.
