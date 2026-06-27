# C7-B6 official decision

# C7-B6 official-val one-shot

- Status: `C7_B6_OFFICIAL_PROMOTED`
- method: `R1-oriented selective top1 override`
- selected_config: `{"K_guard": 10, "model": "GateMLP_15_32_16_3", "tau": 0.20000000000000007}`
- official_val_used: `true`
- official evaluator calls: `1`
- post_val_adjustment / second_official_val: `false / false`
- evaluator_modified / nms_modified: `false / false`
- fixed_pool_invariant_on_official: `True`
- override count/rate: `272 / 0.024965580541532813`
- top1/top5/top10 changed rate: `0.024965580541532813 / 0.0056906837999082145 / 0.0`
- hard_positive exits/entries/ratio: `0 / 0 / 0.0`
- invalid_span_count / duplicate_span_count_after_nms: `0 / 0`

## Metrics

| system | 0.5-r1 | 0.5-r5 | 0.5-r10 | 0.5-r100 | 0.7-r1 | 0.7-r5 | 0.7-r10 | 0.7-r100 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| C6-B1 official | 14.92 | 28.66 | 34.93 | 46.56 | 8.39 | 18.82 | 24.82 | 35.88 |
| C6-B2 official | 14.98 | 28.66 | 34.90 | 46.54 | 8.45 | 18.78 | 24.76 | 35.82 |
| C7-B1 official archived | 15.24 | 28.82 | 34.89 | 46.54 | 8.62 | 19.43 | 25.32 | 35.81 |
| C7-B2.1 A4 official | 15.45 | 28.80 | 34.80 | 46.54 | 8.70 | 19.49 | 25.17 | 35.81 |
| C7-B6 official | 15.51 | 28.79 | 34.81 | 46.54 | 8.75 | 19.50 | 25.18 | 35.81 |
| Delta vs C6-B1 | +0.59 | +0.13 | -0.12 | -0.02 | +0.36 | +0.68 | +0.36 | -0.07 |
| Delta vs C6-B2 | +0.53 | +0.13 | -0.09 | +0.00 | +0.30 | +0.72 | +0.42 | -0.01 |
| Delta vs C7-B1 | +0.27 | -0.03 | -0.08 | +0.00 | +0.13 | +0.07 | -0.14 | +0.00 |
| Delta vs C7-B2.1 A4 | +0.06 | -0.01 | +0.01 | +0.00 | +0.05 | +0.01 | +0.01 | +0.00 |

Stop here and wait for human review.
