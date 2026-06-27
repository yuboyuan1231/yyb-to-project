# C6-B1-lite official-val one-shot audit

- Status: `C6_B1_LITE_OFFICIAL_VAL_STRONG_POSITIVE`
- official_val_one_shot: `true`
- selected_config: `c6b1r1_0057`
- learned_config_count_on_val: `0`
- score_grid_on_val / temperature_search_on_val / post_val_adjustment: `false / false / false`
- C6_C_used / C4_final_modified / evaluator_modified: `false / false / false`

## Metrics

| system | 0.5-r1 | 0.5-r5 | 0.5-r10 | 0.5-r100 | 0.7-r1 | 0.7-r5 | 0.7-r10 | 0.7-r100 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| C4_final v21_00444 | 14.81 | 28.57 | 34.81 | 46.45 | 8.35 | 18.64 | 24.63 | 35.70 |
| C6-B1-lite c6b1r1_0057 | 14.92 | 28.66 | 34.93 | 46.56 | 8.39 | 18.82 | 24.82 | 35.88 |
| delta | +0.11 | +0.09 | +0.12 | +0.11 | +0.04 | +0.18 | +0.19 | +0.18 |

## Movement / structure

- replacement_rate_top_slots: `0.014838610983631635`
- candidate_replacement_count: `485`
- top1_changed_ratio: `0.03267553923818265`
- hard-positive top100 exits/entries: `3 / 15`
- video_slot_drift / video_multiset_drift: `0.0 / 0.0`
- invalid_span_count / duplicate_span_count: `0 / 0`

No post-val adjustment was performed. No second official-val run is authorized.
