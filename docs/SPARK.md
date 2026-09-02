# Spark notes (DGX Spark / GB10, aarch64)

Frozen Moshi / BayLing-Duplex / VAP numbers in this repo come from a **finished**
Spark job on a 10-clip DuplexChat English slice. **Do not re-run Spark jobs
from this repository.** Re-use the reference numbers in
`latent_timing_duplex.spark_slice` and `configs/spark_slice.yaml`.

## Machine

| | |
|---|---|
| Host | NVIDIA DGX Spark (GB10), 128 GB unified memory |
| Arch | `aarch64` |
| CUDA | **cu130** (CUDA 13.0 PyTorch wheel) |
| flash-attn | **Do not install.** No usable aarch64/cu130 wheel; Moshi NLL uses `LMModel.forward`, not flash-attn kernels. |
| GB10 kernel fork | **Do not use.** Stay on the public PyTorch cu130 wheel. No custom GB10 kernel tree. |

## Upstream gaps (install these yourself on Spark)

Kyutai `moshi` and ErikEkstedt `VAP` requirement files omit packages that the
import graph still needs:

| Package | Why it is missing | Spark action |
|---|---|---|
| `sphn` | No aarch64 wheel on PyPI for the versions Moshi pins | Install from **sdist**: `pip install sphn --no-binary=sphn` |
| `tiktoken` | Missing from upstream Moshi extras | `pip install tiktoken` |
| `torchaudio` | Missing from upstream Moshi / VAP extras | Install the **cu130** torchaudio wheel that matches the torch build |

Do not add those as hard dependencies of this Phase 0 package. The skeleton
stays CPU-installable with numpy + PyYAML.

## Moshi NLL flags

NaN-safe `LMModel.forward` on Spark used:

```bash
export NO_CUDA_GRAPH=1
export NO_TORCH_COMPILE=1
```

`LMGen` CUDA graphs and `torch.compile` are generation paths. Phase 0 NLL is
teacher-forced `LMModel.forward` plus the delay-NaN mask. Graphs/compile are
disabled so aarch64 compile/graph bugs cannot poison the reduction.

## Delay-NaN (non-binding hypothesis)

Moshi’s `_undelay_sequence` rolls delayed codebooks back and fills the vacated
tail with `NaN` (`moshi/models/lm.py`, `fill_value=float('NaN')`). Default
delays are `[0, 0, 1, 1, …]` — text and the first audio codebook are delay-0;
later acoustic codebooks are delay-1. A naive `mean` over `[B, K, T]` NLL is
NaN. The extractor masks those positions (and any remaining non-finite logits)
before reducing.

Hypothesis, **not a claim**: that delay fill is what produced Moshi NaNs before
the mask.

## Measured 10-clip DuplexChat EN slice (LEFT=user, same tar)

Document only. Do not invent more digits or extra clips.

**Moshi** frozen user-channel NLL (`LMModel.forward`, nan-safe, flags above),
1383.44 s:

| Reduction | audio NLL | text NLL |
|---|---|---|
| Unweighted (mean of per-clip means) | 0.407020 | 0.023778 |
| Duration-weighted | 0.488034 | 0.025773 |

**BayLing-Duplex** token NLL, 17280 tokens, vocab **168960** (not comparable
to Moshi codebook NLL):

| Reduction | NLL | PPL |
|---|---|---|
| Unweighted | 9.759206 | 17313 |
| Duration-weighted | 7.167710 | 1297 |

**VAP** frozen ErikEkstedt/VAP, **CPU**:

| | duration-weighted |
|---|---|
| p_now | 0.372166 |
| p_future | 0.412835 |
| p(shift) = 1 − p_now (LEFT=user / speaker 0) | 0.627834 |

## What this repo will not do

- Re-run the Spark job
- Download CANDOR
- Implement Phase 1 JEPA heads or Phase 2 policy conditioning
- Fetch Moshi / BayLing / VAP weights
