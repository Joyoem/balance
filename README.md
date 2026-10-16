# balance

Discovery experiments for resource-efficient open-ended
language-conditioned segmentation on edge devices.

## Current research question

Why are existing open-ended language-conditioned segmentation
models still difficult to deploy efficiently on constrained
edge hardware?

The current stage is observation / profiling, not method design.

## Anchor reproduction

DETRIS-B on RefCOCO+ val, RTX 3060, FP32:

| Metric | Result |
|---|---:|
| Referring objects | 3,805 |
| Expressions | 10,758 |
| mIoU | 68.9882 |
| oIoU | 65.7851 |
| Pr@50 | 79.2712 |
| Model latency mean | 78.73 ms/query |
| Model latency p50 | 78.65 ms/query |
| Model latency p95 | 79.27 ms/query |
| Query pipeline p50 | 82.48 ms/query |
| Peak allocated VRAM | 1529 MB |
| Wall throughput | 11.96 queries/s |

The diagnostic evaluator reproduces the official DETRIS result.

## Repository

- `analysis/`: CPU-side analysis and diagnostic scripts
- `experiments/DETRIS/`: derived experimental data and summaries

Raw datasets, LMDB files and model checkpoints are intentionally excluded.
