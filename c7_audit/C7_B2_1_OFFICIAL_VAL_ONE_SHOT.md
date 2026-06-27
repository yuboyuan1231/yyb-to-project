# C7-B2.1 official-val one-shot

- Status: `C7_B2_1_OFFICIAL_PROMOTED`
- selected_variant: `A4_video_residual_only`
- official_val_used: `true`
- official evaluator calls: `1`
- post_val_adjustment / second_official_val: `false / false`
- evaluator_modified / nms_modified: `false / false`
- fixed_pool_invariant_on_official: `True`
- hard_positive exits/entries/ratio: `0 / 0 / 0.0`
- invalid_span_count / duplicate_span_count_after_nms: `0 / 0`

## Metrics

| system | 0.5-r1 | 0.5-r5 | 0.5-r10 | 0.5-r100 | 0.7-r1 | 0.7-r5 | 0.7-r10 | 0.7-r100 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| C6-B1 official | 14.92 | 28.66 | 34.93 | 46.56 | 8.39 | 18.82 | 24.82 | 35.88 |
| C6-B2 official | 14.98 | 28.66 | 34.90 | 46.54 | 8.45 | 18.78 | 24.76 | 35.82 |
| C7-B1 official archived | 15.24 | 28.82 | 34.89 | 46.54 | 8.62 | 19.43 | 25.32 | 35.81 |
| C7-B2.1 A4 official | 15.45 | 28.80 | 34.80 | 46.54 | 8.70 | 19.49 | 25.17 | 35.81 |
| Delta vs C6-B1 | +0.53 | +0.14 | -0.13 | -0.02 | +0.31 | +0.67 | +0.35 | -0.07 |
| Delta vs C6-B2 | +0.47 | +0.14 | -0.10 | +0.00 | +0.25 | +0.71 | +0.41 | -0.01 |
| Delta vs C7-B1 | +0.21 | -0.02 | -0.09 | +0.00 | +0.08 | +0.06 | -0.15 | +0.00 |

Stop here and wait for human review.
