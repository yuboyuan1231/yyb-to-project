# C6-C1 arm comparison

- Status: `C6_C1_R1_TRADEOFF`
- Official val used: `false`
- Grid size: `160`
- Feasible count: `0`

## Best config

```json
{
  "config": {
    "alpha": 0.85,
    "beta": 0.2,
    "gamma": 0.2,
    "rho": 0.05,
    "gate_threshold": 0.08673583984375,
    "max_video_slot_changes": 1,
    "lambda_span": 0.15,
    "video_weight": 1.0,
    "span_anchor": "B2"
  },
  "metrics": {
    "0.5-r1": 39.82148987298318,
    "0.5-r5": 48.930083533585076,
    "0.5-r10": 59.13720105275203,
    "0.5-r100": 76.29019338597094,
    "0.7-r1": 27.051150017164435,
    "0.7-r5": 41.652362970591604,
    "0.7-r10": 50.93260098409429,
    "0.7-r100": 65.5223709806614
  },
  "video_metrics": {
    "GT_video_R@1": 48.621123698363654,
    "GT_video_R@5": 51.75649387801808,
    "GT_video_R@10": 62.03226913834535,
    "GT_video_R@100": 80.37532898500973
  },
  "movement": {
    "positive_video_entries": 0,
    "positive_video_exits": 0,
    "hard_positive_top100_query_exits": 0,
    "hard_positive_top100_query_entries": 0,
    "hard_positive_exit_ratio": 0.0,
    "video_slot_drift_rate": 0.2970784225378833,
    "video_multiset_drift_rate": 0.2970784225378833,
    "invalid_span_count": 0,
    "duplicate_span_count_after_nms": 0
  },
  "official_val_used": false,
  "span_anchor": "B2",
  "config_id": "mavr_pr_mil_0061",
  "delta_vs_C6_B1_lite": {
    "0.5-r1": 0.24030209406110714,
    "0.5-r5": -13.502689094862113,
    "0.5-r10": -10.470305526948167,
    "0.5-r100": -0.022885913720102735,
    "0.7-r1": 0.38906053324178913,
    "0.7-r5": -7.197619864973106,
    "0.7-r10": -5.504062249685319,
    "0.7-r100": 0.05721478430027105
  },
  "delta_vs_C6_B2": {
    "0.5-r1": 0.0,
    "0.5-r5": -13.491246138002062,
    "0.5-r10": -10.481748483808218,
    "0.5-r100": 0.0,
    "0.7-r1": 0.0,
    "0.7-r5": -7.243391692413319,
    "0.7-r10": -5.561277033985576,
    "0.7-r100": 0.0
  },
  "selection_score": 6.5373934850030295,
  "feasible": false
}
```
