# C7-B1 official-val one-shot

- Status: `C7_B1_R1_TRADEOFF_NO_PROMOTION`
- Fixed config: `S1_span_level_bounded_residual`, `mu=0.1`, `eta=0.1`
- Official val used: `true`
- Evaluator calls in this runner: `1`
- Training / config search / post-val adjustment / second official val: `false / false / false / false`
- evaluator_modified / nms_modified: `false / false`
- invalid_span_count / duplicate_span_count_after_nms: `0 / 0`

## Metrics

| system | 0.5-r1 | 0.5-r5 | 0.5-r10 | 0.5-r100 | 0.7-r1 | 0.7-r5 | 0.7-r10 | 0.7-r100 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| C6-B1-lite official | 14.92 | 28.66 | 34.93 | 46.56 | 8.39 | 18.82 | 24.82 | 35.88 |
| C6-B2 official | 14.98 | 28.66 | 34.90 | 46.54 | 8.45 | 18.78 | 24.76 | 35.82 |
| C7-B1 official | 15.24 | 28.82 | 34.89 | 46.54 | 8.62 | 19.43 | 25.32 | 35.81 |
| Delta vs C6-B1 | +0.32 | +0.16 | -0.04 | -0.02 | +0.23 | +0.61 | +0.50 | -0.07 |
| Delta vs C6-B2 | +0.26 | +0.16 | -0.01 | +0.00 | +0.17 | +0.65 | +0.56 | -0.01 |

## Decision Rule

```json
{
  "r1_vs_C6_B2_positive": true,
  "r5_r10_vs_C6_B1_nonnegative": false,
  "r5_r10_vs_C6_B1_negative": true,
  "rule": "Promote only if 0.7-r1 and 0.5-r1 beat C6-B2 and all R@5/R@10 deltas vs C6-B1 are non-negative; otherwise positive R@1 with negative R@5/R@10 is marked tradeoff."
}
```

Stop here and wait for human review.
