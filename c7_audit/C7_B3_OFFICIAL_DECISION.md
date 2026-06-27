# C7-B3 official decision

# C7-B3 official-val one-shot

- Status: `C7_B3_OFFICIAL_NO_GAIN_KEEP_C7_B2_1`
- selected_candidate: `calibrated_A4`
- selected_config: `{"T": 0.5, "clip": 3.0, "lambda": 1.0, "normalization": "per_query_z"}`
- official_val_used: `true`
- official evaluator calls: `1`
- post_val_adjustment / second_official_val: `false / false`
- evaluator_modified / nms_modified: `false / false`
- C7_B4_SN_Audit_run / MINUTE_style_scoring_added: `false / false`
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
| C7-B3 official | 12.05 | 20.64 | 26.04 | 46.54 | 6.98 | 14.45 | 18.77 | 35.81 |
| Delta vs C6-B1 | -2.87 | -8.02 | -8.89 | -0.02 | -1.41 | -4.37 | -6.05 | -0.07 |
| Delta vs C6-B2 | -2.93 | -8.02 | -8.86 | +0.00 | -1.47 | -4.33 | -5.99 | -0.01 |
| Delta vs C7-B1 | -3.19 | -8.18 | -8.85 | +0.00 | -1.64 | -4.98 | -6.55 | +0.00 |
| Delta vs C7-B2.1 A4 | -3.40 | -8.16 | -8.76 | +0.00 | -1.72 | -5.04 | -6.40 | +0.00 |

Stop here and wait for human review.
