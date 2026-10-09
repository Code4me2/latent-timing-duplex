# Current research status

**Cutoff:** 2026-09-25 PT  
**Evidence index:** Phase 0 freeze + Phase 1 proxy evaluation + exploratory Phase 2 C0/C1/C2

This is the current index. Earlier phase documents remain historical records and
should not be rewritten to make later work look predeclared.

## Scientific question

Can a predictive latent derived from a frozen full-duplex model expose upcoming
turn events, and can that latent causally improve live response timing?

The first half has a promising proxy result. The second is not established.

## What was actually trained

The realized Spark target is **not an independent next-user-audio embedding**.
It is the mean of future Moshi transformer states:

```text
h_t = Moshi forward_text transformer_out[t]
y_t = mean(transformer_out[t+H : t+2H])
surprise_t = prediction_error(head(h_t), y_t)
```

During the original precompute, LEFT carried user audio and the assistant input
was silence. Optional user codec targets were saved but were not the primary
`target` tensors consumed by the ablation training. The supported interpretation
is therefore **future hidden-state prediction error**, not yet future user-audio
latent prediction.

## Evidence ledger

| Result | Evidence | Use | Current interpretation |
|---|---|---|---|
| Moshi, BayLing-Duplex, and VAP run on the stereo pipeline | Replayed engineering runs | Required dependency | Validated |
| Full-length CANDOR has higher Moshi NLL than DuplexChat | Numerical observation with disjoint duration support | Historical diagnostic | Length-confounded; not a primary claim |
| Equal-length cross-corpus Moshi NLL difference | Mid-180 n=12 comparison | Decision-relevant | No significant durable gap |
| H=12 surprise predicts proxy turn events | Session-bootstrap numerical observation | Decision-relevant | Promising proxy signal |
| Isotropic regularization improves timing prediction | λ ablation | Decision-relevant | Unsupported; λ=0.01 ≈ λ=0 in event ranking |
| C1 offline residual improves future-VAD BCE | Held-out proxy readout | Diagnostic | Predictive information reaches the injector |
| Live residual/LoRA improves duplex timing | Twelve-session live decode | Decision-relevant | Inconclusive; no JEPA-specific control result |

## Latest Phase 1 proxy result

Mid-180, H=12 (0.96 s), λ=0.01, CANDOR45 + DuplexChat41, seed
`20260903`, session bootstrap B=10000:

| Signal | turn-shift AUROC at 1 s | AUPRC |
|---|---:|---:|
| JEPA surprise | **0.669 [0.642, 0.696]** | 0.144 |
| Moshi NLL | 0.479 [0.455, 0.503] | 0.054 |
| VAP p(shift) | 0.501 [0.478, 0.525] | 0.054 |

Other proxy-event AUROCs at 1 s: barge-in 0.660; backchannel 0.611.
DuplexChat labels are channel-energy VAD proxies and CANDOR labels are
transcript-derived proxies, not gold annotations. The pooled evaluation also
contains training sessions; its bootstrap does not correct for that reuse.

Source artifact:
`/home/velvet/cs199-phase1-work/eval/overnight_2026-09-12/`

## Latest Phase 2 exploratory result

The 2026-09-25 sequence tested:

- **C0:** 4096→512→512→256 future-state predictor.
- **C1:** `h' = h + g(JEPA(h))` before the depformer.
- **C2:** C1 residual plus rank-8 LoRA on Moshi block 31.

C0 retained proxy predictiveness (pooled turn-shift AUROC 0.647). C1's
offline future-VAD BCE improved from 0.457 (none) / 0.463 (shuffle) to
0.439 (inject). Live C1 collapsed on 11/12 sessions in every condition.
C2 reduced collapse with LoRA, including when the JEPA residual was absent:

| C2 live condition | Speak | Overlap | Gap | Collapse |
|---|---:|---:|---:|---:|
| inject | 0.164 | 0.050 | 0.114 | 4/12 |
| shuffled latent | 0.107 | 0.040 | 0.067 | 2/12 |
| no latent, LoRA on | 0.160 | 0.072 | 0.088 | 5/12 |
| LoRA off | 0.211 | 0.056 | 0.155 | 11/12 |

This points to a generic LoRA/activity effect, not a demonstrated
JEPA-specific timing effect. The historical C1/C2 `PASS` files are execution
gates: they certify that conditions completed, not that efficacy improved.
See [PHASE2_EXPLORATORY.md](PHASE2_EXPLORATORY.md).

## Reconciled experiment settings

| Topic | Planned scaffold | Realized Spark work |
|---|---|---|
| Predictor offsets | half-up 1/13/63 | explicit **1/12/62** frames |
| Main predictor | 4096→512→512→256 | Phase 1 ablation used 4096→512→4096; C0 used 4096→512→512→256 |
| Regularizer grid | 0/0.1/1/10 | **0/0.01/0.1/1**; event primary 0.01, reference 0 |
| Target | frozen user-chunk embedding | future Moshi transformer-state mean |
| Proposed control | scalar surprise conditions policy | C1/C2 inject the predicted 256-D latent |

The package remains Phase 1-only and still raises `Phase2OutOfScope` for
backbone training. Phase 2 is retained as an external exploratory experiment,
not a supported package API.

## Decision

Pause further control scaling. The next experiment should first test whether
the signal is specifically predictive rather than a proxy for generic acoustic
or representation change.

Predeclare an untouched evaluation over the DuplexChat sessions added after
the original head was trained. Compare surprise with raw energy, energy
change/onset, hidden-state velocity, an untrained head, and shuffled targets.
Use paired session bootstrap and report held-out-only results. Gold or audited
turn labels remain the next data-quality requirement.
