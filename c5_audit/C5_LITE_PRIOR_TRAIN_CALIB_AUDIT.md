# C5-lite-prior train_calib audit

## Decision

- Status: `NO_PROMOTION`
- Classification: `C5-lite-prior train_calib negative`
- Primary baseline: `C4_final = C4-r2-cal-v2.1 v21_00444`
- Official val used: `false`
- Post-val adjustment: `false`
- Recommendation: do not freeze a C5-lite-prior scorer and do not run official val.

The true temporal-prior path was implemented and validated, but it did not produce a safe train_calib improvement. Of 659 deduplicated configurations, only the zero-residual control kept all six primary VCMR metrics nonnegative. Therefore the experiment does not verify retrieval evidence → localization calibration under this frozen protocol.

## Input and implementation checks

- `P_b` and `P_e`: softmax of the frozen CONQUER begin/end logits.
- `P_ctx`: frozen QAL query-to-video attention.
- Row-level scores were not used to fabricate temporal priors.
- train_fit: 78,045 queries, 779,142 query-video groups, 15,609,000 candidate rows.
- train_calib: 8,739 queries, 87,230 query-video groups, 1,747,800 candidate rows.
- All temporal arrays were finite, nonnegative, positive-sum, and normalized; effective lengths were 2–100.
- Term means/stds were frozen from train_fit only.
- Dense feature aggregation ran in GPU chunks and matched the ragged reference to maximum absolute error `5.960464477539063e-08`.
- C4_final remained the selection baseline; C4-lite and C3.1 were secondary references only.

## C4_final and selected strict-safe result

The strict-safe selection is `c5p_00000`, the zero-residual control, so its metrics equal C4_final exactly.

| Metric | C4_final | Selected | Delta |
|---|---:|---:|---:|
| 0.5 R@1 | 39.375215 | 39.375215 | 0.000000 |
| 0.5 R@5 | 62.284014 | 62.284014 | 0.000000 |
| 0.5 R@10 | 69.481634 | 69.481634 | 0.000000 |
| 0.5 R@100 | 76.221536 | 76.221536 | 0.000000 |
| 0.7 R@1 | 26.376016 | 26.376016 | 0.000000 |
| 0.7 R@5 | 48.666896 | 48.666896 | 0.000000 |
| 0.7 R@10 | 56.230690 | 56.230690 | 0.000000 |
| 0.7 R@100 | 65.247740 | 65.247740 | 0.000000 |

Relative to C4-lite, the zero-residual control retains the already frozen C4_final gains: `[+0.045772, +0.011443, +0.068658, +0.034329, +0.022886, +0.034329, +0.160201, +0.034329]` in the table's metric order. Relative to C3.1 the corresponding deltas are `[+1.624900, +0.595034, +0.663691, +0.137315, +0.869665, +2.197048, +2.300034, +0.732349]`.

## Highest-scoring nonzero residual

`c5p_00138` applies the boundary-agreement term at the grid boundary (`lambda=0.125`, term weight `0.075`). Its selection-score delta is `+0.00143037`, but it fails the primary constraint:

| Metric | Delta vs C4_final |
|---|---:|
| 0.5 R@1 | -0.057215 |
| 0.5 R@5 | 0.000000 |
| 0.5 R@10 | -0.068658 |
| 0.5 R@100 | +0.034329 |
| 0.7 R@1 | +0.034329 |
| 0.7 R@5 | 0.000000 |
| 0.7 R@10 | +0.022886 |
| 0.7 R@100 | +0.068658 |

Movement itself was small (`top1_changed_ratio=0.010642`, Pearson `0.999958`, hard-positive top-100 exits/entries `40/82`), but safe movement cannot override the two primary regressions. It also sits on a grid boundary with no passing neighbor.

## Localization diagnostics for the strict-safe selection

- Oracle-video R@1@0.5 delta: `0.0`
- Oracle-video R@1@0.7 delta: `0.0`
- Selected-span mIoU delta: `0.0`
- Mean best-IoU span-rank delta: `0.0`
- Positive-span upward/downward ratio: `0.0 / 0.0`
- Hard-negative downward/upward ratio: `0.0 / 0.0`
- Same-video ordering changed ratio: `0.0`
- Selected prior-peak-inside ratio: `0.880421`

The prior is structurally meaningful, but the only VCMR-safe selection does not change ranking, so localization feedback is not verified.

## Frozen boundaries

- `effective_top_n=100`, `NMS=0.7`, `max_after_nms=100` unchanged.
- `model/conquer.py`, begin/end logits, candidate generation, and evaluator unchanged.
- C4 was not retuned; C5-main and C6 were not run.
- No official-val scores, cache, submission, grid, or evaluator run was produced.
- No freeze review was generated because the promotion gate failed.
