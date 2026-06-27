# C6-B1-lite-v2 listwise slot selector audit (pairwise)

- Status: `C6_B1_LITE_V2_LISTWISE_PAIRWISE_R1_SAFE_NEGATIVE`
- Scope: `train_fit/train_calib only`
- Official val used: `false`
- Model: hidden `384`, transformer layers `2`, heads `8`.
- Grid size: `78`, feasible: `0`.

## Best config

```json
{
  "config_id": "c6b1r1_0007",
  "apply_slots": 1,
  "threshold0": 1.5,
  "threshold_rest": 999.0,
  "max_replacements": 1,
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
  "delta_vs_pseudo_C4_final": {
    "0.5-r1": 0.0,
    "0.5-r5": 0.0,
    "0.5-r10": 0.0,
    "0.5-r100": 0.0,
    "0.7-r1": 0.0,
    "0.7-r5": 0.0,
    "0.7-r10": 0.0,
    "0.7-r100": 0.0
  },
  "selection_score": 0.0,
  "r1_positive": false,
  "r5_safe": true,
  "feasible": false,
  "movement": {
    "replacement_rate_top_slots": 0.0,
    "replacements": 0,
    "top1_changed_ratio": 0.0,
    "hard_positive_top100_query_exits": 0,
    "hard_positive_top100_query_entries": 0,
    "video_slot_drift": 0.0,
    "video_multiset_drift": 0.0
  }
}
```
