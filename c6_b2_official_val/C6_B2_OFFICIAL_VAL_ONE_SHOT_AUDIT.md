# C6-B2 official-val one-shot audit

- Status: `C6_B2_OFFICIAL_POSITIVE`
- official_val_one_shot: `true`
- selected_config: `pairwise_main_0005`
- successful_metric_result_count: `1`
- learned_config_count_on_val: `0`
- score_grid_on_val / temperature_search_on_val / post_val_adjustment: `false / false / false`
- C6_C_used / C4_final_modified / C6_B1_lite_modified / evaluator_modified: `false / false / false / false`

## Metrics

| system | 0.5-r1 | 0.5-r5 | 0.5-r10 | 0.5-r100 | 0.7-r1 | 0.7-r5 | 0.7-r10 | 0.7-r100 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| C4_final v21_00444 | 14.81 | 28.57 | 34.81 | 46.45 | 8.35 | 18.64 | 24.63 | 35.70 |
| C6-B1-lite c6b1r1_0057 | 14.92 | 28.66 | 34.93 | 46.56 | 8.39 | 18.82 | 24.82 | 35.88 |
| C6-B2 pairwise_main_0005 | 14.98 | 28.66 | 34.90 | 46.54 | 8.45 | 18.78 | 24.76 | 35.82 |
| Δ vs C6-B1 | +0.06 | +0.00 | -0.03 | -0.02 | +0.06 | -0.04 | -0.06 | -0.06 |
| Δ vs C4_final | +0.17 | +0.09 | +0.09 | +0.09 | +0.10 | +0.14 | +0.13 | +0.12 |

## Movement

- replacement_rate_top_slots: `0.006773749426342359`
- candidate_replacement_count: `369`
- top1_changed_ratio_vs_C6_B1: `0.03129876089949518`
- top1_changed_ratio_vs_C4_final: `0.01101422670949977`
- hard-positive exits/entries vs C6-B1: `9 / 6`
- hard-positive exits/entries vs C4_final: `1 / 10`
- video_slot_drift / video_multiset_drift: `0.0 / 0.0`
- invalid_span_count / duplicate_span_count_after_nms: `0 / 0`

No post-val adjustment was performed. No second official-val run is authorized.
