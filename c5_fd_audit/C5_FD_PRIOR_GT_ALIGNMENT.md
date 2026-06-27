# C5 failure decomposition: prior–GT alignment

Read-only train_calib diagnostic. GT is used only for diagnosis, never to fit or select a scorer.

## Coverage and peaks

| Diagnostic | Value |
|---|---:|
| `query_count` | 8739 |
| `gt_video_present_query_count` | 7043 |
| `gt_video_present_ratio` | 0.80592745 |
| `p_ctx_peak_inside_gt_span_ratio` | 0.19352549 |
| `p_bd_peak_inside_gt_span_ratio` | 0.87107767 |
| `p_retloc_peak_inside_gt_span_ratio` | 0.79710351 |

## Prior-only localization

| Diagnostic | Value |
|---|---:|
| `gt_span_mass_above_best_hard_negative_ratio` | 0.37125091 |
| `prior_only_oracle_video_r1_05` | 30.20019878 |
| `prior_only_oracle_video_r1_07` | 11.55757490 |
| `prior_only_selected_span_miou` | 0.38793636 |

## P_retloc mass distributions

| Diagnostic | Value |
|---|---:|
| `GT mass mean` | 0.38591738 |
| `GT mass median` | 0.37055403 |
| `best hard-negative mean` | 0.45087738 |
| `best hard-negative median` | 0.45223382 |
