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
| DuplexChat | 12 sessions in the precompute tree; **no on-disk transcripts**, so **not** in the labeled table |

Gold `--labels` JSONL was not used. DuplexChat cannot join this table until
transcripts (or gold events) exist on disk.

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
- **Not cross-corpus.** DuplexChat (12 precomputed) has no on-disk
  transcripts in this run, so there is no DuplexChat or pooled ranking.
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

## Recommended next
1. Gold or audited labels (do not treat CSV proxies as final).
2. DuplexChat transcripts on disk, then the same mid-180 table (n=12
   exact-W), still not a Phase 0 NLL re-run.
3. Publish CIs from the Spark JSON if a freeze is declared.
4. Do **not** start Phase 2 from this ranking.
