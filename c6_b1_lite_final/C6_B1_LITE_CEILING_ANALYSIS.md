# C6-B1-lite ceiling analysis

All variants below are diagnostic/oracle analyses over frozen official-val artifacts. No official evaluator was called.

| variant | 0.5-r1 | 0.5-r5 | 0.5-r10 | 0.5-r100 | 0.7-r1 | 0.7-r5 | 0.7-r10 | 0.7-r100 | repl |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A_selector_current_actual | 14.92 | 28.66 | 34.93 | 46.55 | 8.37 | 18.84 | 24.85 | 35.87 | 485 |
| B_current_candidate_pool_oracle_apply3_max2 | 22.51 | 33.39 | 37.54 | 47.20 | 17.50 | 26.02 | 28.85 | 37.28 | 5981 |
| C_boundary_candidate_only_oracle_apply3_max2 | 18.79 | 33.33 | 37.49 | 47.21 | 12.36 | 25.35 | 28.45 | 36.95 | 6784 |
| D_original_candidate_only | 14.81 | 28.57 | 34.81 | 46.44 | 8.33 | 18.67 | 24.66 | 35.70 | 0 |
| E_hybrid_pool_greedy_slots5_max2_same_video_slots | 20.11 | 34.34 | 38.07 | 47.30 | 14.27 | 27.30 | 29.77 | 37.60 | 7211 |
| F_hybrid_pool_greedy_slots5_max5_same_video_slots | 23.47 | 34.43 | 38.12 | 47.30 | 18.90 | 27.50 | 29.93 | 37.68 | 11933 |

## Wrong-video-limited residual

- gt_video_missing_from_c4_top100: `4095` / `10895`
- gt_video_missing_rate: `0.3759`
- gt_video_not_top1_rate: `0.7030`

## Interpretation

- r1_07_oracle_gap_slots5_max2_vs_actual: `5.9018`
- r100_07_oracle_gap_slots5_max2_vs_actual: `1.7347`
- selector_conservative: `True`
- harmful_replacement_low: `True`
- wrong_video_major: `False`
- recommendation: `Recommend C6-B2` — 同视频 hybrid candidate pool 的 oracle 仍高于 actual，且 replacement harmful rate 低；优先开 C6-B2 higher-ceiling selector protocol。
