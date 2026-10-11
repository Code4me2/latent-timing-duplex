# Exploratory Phase 2 latent graft

**Run date:** 2026-09-25 PT  
**Host:** `spark-61dd`  
**Status:** exploratory receipt; not a Phase 2 efficacy freeze

This work happened after the committed Phase 1 protocol. It does not change
the frozen Phase 0 findings or turn the package's Phase 1 API into a supported
fine-tuning API. Exact-run script snapshots are under
[`experiments/phase2/snapshots/`](../experiments/phase2/snapshots/).

## C0 — predictive head

- Head: `4096 → 512 → 512 → 256`, ReLU, 2,491,648 parameters.
- Target: future 4096-D Moshi transformer-state mean projected by a frozen
  seeded QR matrix to 256 dimensions.
- H=12, λ=0.01, 74 train sessions, 12 validation sessions.
- Best validation MSE 1.045 at step 500; stopped at step 3500.
- No Moshi load and no residual injection.

Proxy turn-shift AUROC at 1 s:

| Slice | AUROC [95% CI] |
|---|---:|
| pooled | 0.647 [0.617, 0.677] |
| CANDOR | 0.591 [0.554, 0.628] |
| DuplexChat | 0.706 [0.667, 0.746] |

The exported pooled table includes train and validation sessions. `C0 PASS`
meant that proxy AUROC cleared its exploratory gate and no offline surprise
collapse was detected.

Artifacts:
`/home/velvet/cs199-phase1-work/eval/phase2_c0_2026-09-25/`

## C1 — residual injector

Offline C1 froze C0 and trained:

```text
s = JEPA(h)
h' = h + g(s)
g: 256 → 512 → 4096
loss = BCE(readout(h'), future_user_VAD_proxy) + 1e-4 mean(g(s)^2)
```

Held-out offline BCE:

| Condition | BCE |
|---|---:|
| inject | **0.439** |
| none | 0.457 |
| shuffled latent | 0.463 |

This is a teacher-forced proxy readout, not generation.

The live run inserted the residual after `forward_text` and before the
depformer. On 12 locked validation sessions, all three conditions had 11/12
collapse flags. Injection reduced speak mass relative to no injection and did
not improve overlap. The live result does not establish timing control.

Artifacts:

- `/home/velvet/cs199-phase1-work/eval/phase2_c1_2026-09-25/`
- `/home/velvet/cs199-phase1-work/eval/phase2_c1_live_2026-09-25/`

## C2 — one-block LoRA plus residual

- LoRA: rank 8, scaling 2, Moshi block 31, 532,480 trainable LoRA parameters.
- Training: 8 sessions, first 30 s, 200 steps, 48-frame chunks.
- Live evaluation: the same 12 locked validation sessions.

| Condition | Speak | Overlap | Gap | Collapse |
|---|---:|---:|---:|---:|
| inject | 0.164 | 0.050 | 0.114 | 4/12 |
| shuffled latent | 0.107 | 0.040 | 0.067 | 2/12 |
| no latent, LoRA on | 0.160 | 0.072 | 0.088 | 5/12 |
| LoRA off | 0.211 | 0.056 | 0.155 | 11/12 |

LoRA reduced collapse even without the JEPA residual. Correctly aligned
injection did not beat no injection on overlap or gap, and shuffled injection
had fewer collapse flags. No paired confidence interval for these deltas was
produced. This is inconclusive for control and does not show a JEPA-specific
effect.

Artifact:
`/home/velvet/cs199-phase1-work/eval/phase2_c2_2026-09-25/`

## Gate terminology

- **Execution gate:** all planned conditions completed, outputs exist, and
  errors/collapse diagnostics were recorded.
- **Efficacy gate:** a predeclared paired improvement over both no-injection
  and shuffled controls, without unacceptable collapse or quality loss.

C1-live and C2 `PASS` were execution passes. No Phase 2 efficacy gate has
passed.

## Non-claims

- Proxy VAD/transcript labels are not gold turns.
- C0 pooled AUROC is not held-out-only.
- C1 offline BCE is not live conversational quality.
- Lower collapse alone is not better turn timing.
- The injected vector is the predicted latent, not scalar surprise.
- No result supports deployment or further control scaling yet.
