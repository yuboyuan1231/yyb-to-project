# C6-B1-lite R1-safe selector audit

- Status: `C6_B1_LITE_R1_SAFE_POSITIVE`
- Scope: `train_fit/train_calib only`
- Official val used: `false`
- Primary focus: `R@1`; safety focus: `R@5`.
- Model: `4` layers, hidden `384`, candidate examples `3121800`.

## Best config

```json
{
  "config_id": "c6b1r1_0057",
  "apply_slots": 3,
  "threshold0": 0.4,
  "threshold_rest": 1.8,
  "max_replacements": 2,
  "metrics": {
    "0.5-r1": 39.58118777892207,
    "0.5-r5": 62.43277262844719,
    "0.5-r10": 69.6075065797002,
    "0.5-r100": 76.31307929969104,
    "0.7-r1": 26.662089483922646,
    "0.7-r5": 48.84998283556471,
    "0.7-r10": 56.43666323377961,
    "0.7-r100": 65.46515619636114
  },
  "delta_vs_pseudo_C4_final": {
    "0.5-r1": 0.20597322348094593,
    "0.5-r5": 0.148758439180682,
    "0.5-r10": 0.12587252546057925,
    "0.5-r100": 0.09154365488042515,
    "0.7-r1": 0.28607392150131616,
    "0.7-r5": 0.20597322348094593,
    "0.7-r10": 0.19453026662089457,
    "0.7-r100": 0.16020139604073336
  },
  "selection_score": 0.3241217530609896,
  "r1_positive": true,
  "r5_safe": true,
  "feasible": true,
  "movement": {
    "replacement_rate_top_slots": 0.01643971468894229,
    "replacements": 431,
    "top1_changed_ratio": 0.03936377159858107,
    "hard_positive_top100_query_exits": 0,
    "hard_positive_top100_query_entries": 8,
    "video_slot_drift": 0.0,
    "video_multiset_drift": 0.0
  }
}
```

## Baseline pseudo C4_final

```json
{
  "metrics": {
    "0.5-r1": 39.375214555441126,
    "0.5-r5": 62.28401418926651,
    "0.5-r10": 69.48163405423962,
    "0.5-r100": 76.22153564481061,
    "0.7-r1": 26.37601556242133,
    "0.7-r5": 48.644009612083764,
    "0.7-r10": 56.24213296715872,
    "0.7-r100": 65.3049548003204
  },
  "replacement_rate_top_slots": 0.0,
  "replacements": 0,
  "top1_changed_ratio": 0.0,
  "hard_positive_top100_query_exits": 0,
  "hard_positive_top100_query_entries": 0,
  "video_slot_drift": 0.0,
  "video_multiset_drift": 0.0
}
```
