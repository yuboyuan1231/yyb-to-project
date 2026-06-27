# C6-B1-lite-v2 listwise slot selector audit

- Status: `C6_B1_LITE_V2_LISTWISE_R1_SAFE_POSITIVE`
- Scope: `train_fit/train_calib only`
- Official val used: `false`
- Model: hidden `384`, transformer layers `2`, heads `8`.
- Grid size: `78`, feasible: `22`.

## Best config

```json
{
  "config_id": "c6b1r1_0009",
  "apply_slots": 3,
  "threshold0": -0.25,
  "threshold_rest": 0.2,
  "max_replacements": 2,
  "metrics": {
    "0.5-r1": 39.44387229660144,
    "0.5-r5": 62.31834305984666,
    "0.5-r10": 69.50451996795972,
    "0.5-r100": 76.24442155853072,
    "0.7-r1": 26.42178738986154,
    "0.7-r5": 48.689781439523976,
    "0.7-r10": 56.28790479459892,
    "0.7-r100": 65.35072662776061
  },
  "delta_vs_pseudo_C4_final": {
    "0.5-r1": 0.06865774116031531,
    "0.5-r5": 0.0343288705801541,
    "0.5-r10": 0.022885913720102735,
    "0.5-r100": 0.022885913720102735,
    "0.7-r1": 0.04577182744020902,
    "0.7-r5": 0.045771827440212576,
    "0.7-r10": 0.04577182744020547,
    "0.7-r100": 0.04577182744020547
  },
  "selection_score": 0.07466529351184174,
  "r1_positive": true,
  "r5_safe": true,
  "feasible": true,
  "movement": {
    "replacement_rate_top_slots": 0.007323492390433688,
    "replacements": 192,
    "top1_changed_ratio": 0.005263760155624213,
    "hard_positive_top100_query_exits": 0,
    "hard_positive_top100_query_entries": 2,
    "video_slot_drift": 0.0,
    "video_multiset_drift": 0.0
  }
}
```
