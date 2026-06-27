# C5-main-inference-A3b-wide official-val one-shot audit

- Status: `OFFICIAL_C5_NEGATIVE`
- official_val_one_shot: `true`
- learned_config_count: `1`
- selected_config: `c5ma_0214`
- score_grid_on_val / temperature_search_on_val / post_val_adjustment: `false / false / false`
- successful_metric_result_count: `1`

## Official-val metrics

| system | 0.5-r1 | 0.5-r5 | 0.5-r10 | 0.5-r100 | 0.7-r1 | 0.7-r5 | 0.7-r10 | 0.7-r100 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| C0 | 13.68 | 27.21 | 33.75 | 46.23 | 7.76 | 17.22 | 22.49 | 35.17 |
| C3.1 | 14.27 | 28.12 | 34.42 | 46.31 | 8.09 | 18.03 | 23.58 | 35.28 |
| C4-lite | 14.81 | 28.56 | 34.73 | 46.44 | 8.35 | 18.63 | 24.61 | 35.69 |
| C4_final v21_00444 | 14.81 | 28.57 | 34.81 | 46.45 | 8.35 | 18.64 | 24.63 | 35.70 |
| C5 A3b-wide c5ma_0214 | 14.79 | 28.66 | 34.81 | 46.47 | 8.35 | 18.74 | 24.64 | 35.73 |

## Deltas vs C4_final

| metric | delta |
|---|---:|
| 0.5-r1 | -0.02 |
| 0.5-r5 | +0.09 |
| 0.5-r10 | +0.00 |
| 0.5-r100 | +0.02 |
| 0.7-r1 | +0.00 |
| 0.7-r5 | +0.10 |
| 0.7-r10 | +0.01 |
| 0.7-r100 | +0.03 |

## Movement / safety

- top1_changed_ratio: `0.008352455254703992`
- hard-positive top100 exits/entries: `20 / 13`
- Pearson(base, S_loc): `0.9983578201351995`
- mean within-query Spearman: `0.999948155289472`
- video_multiset_drift_query_ratio: `0.0`
- video_slot_sequence_drift_query_ratio: `0.0`
- quota_violation_count / duplicate_span_count: `0 / 0`

No post-val adjustment was performed. No second official-val run is authorized or implied.

Outcome note: R@100 safety was nonnegative, but the front-rank gate did not pass because `0.5-r1` decreased and neither R@1 metric was positive; therefore this is classified as `OFFICIAL_C5_NEGATIVE`, not recall tradeoff.

Next step: C5 A3b-wide did not generalize sufficiently to official val. Close fixed-candidate C5-main and do not run more official val.
