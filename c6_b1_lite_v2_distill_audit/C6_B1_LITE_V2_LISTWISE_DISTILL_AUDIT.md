# C6-B1-lite-v2 listwise slot selector audit (distill)

- Status: `C6_B1_LITE_V2_LISTWISE_DISTILL_R1_SAFE_POSITIVE`
- Scope: `train_fit/train_calib only`
- Official val used: `false`
- Model: hidden `384`, transformer layers `2`, heads `8`.
- Grid size: `78`, feasible: `22`.

## Best config

```json
{
  "config_id": "c6b1r1_0008",
  "apply_slots": 3,
  "threshold0": -0.25,
  "threshold_rest": 0.2,
  "max_replacements": 1,
  "metrics": {
    "0.5-r1": 39.44387229660144,
    "0.5-r5": 62.30690010298661,
    "0.5-r10": 69.50451996795972,
    "0.5-r100": 76.23297860167067,
    "0.7-r1": 26.456116260441696,
    "0.7-r5": 48.72411031010413,
    "0.7-r10": 56.322233665179084,
    "0.7-r100": 65.37361254148072
  },
  "delta_vs_pseudo_C4_final": {
    "0.5-r1": 0.06865774116031531,
    "0.5-r5": 0.022885913720102735,
    "0.5-r10": 0.022885913720102735,
    "0.5-r100": 0.011442956860051368,
    "0.7-r1": 0.08010069802036668,
    "0.7-r5": 0.08010069802036668,
    "0.7-r10": 0.08010069802036668,
    "0.7-r100": 0.06865774116032242
  },
  "selection_score": 0.0975512072319466,
  "r1_positive": true,
  "r5_safe": true,
  "feasible": true,
  "movement": {
    "replacement_rate_top_slots": 0.0008391501697371935,
    "replacements": 22,
    "top1_changed_ratio": 0.00251745050921158,
    "hard_positive_top100_query_exits": 0,
    "hard_positive_top100_query_entries": 1,
    "video_slot_drift": 0.0,
    "video_multiset_drift": 0.0
  }
}
```
