# C6-B2 P0 pool/oracle sanity audit

```json
{
  "status": "C6_B2_P0_POOL_AUDIT",
  "official_val_used": false,
  "source_pool": {
    "path": "results/rlem_c6_b1_lite_r1_safe/train_calib_candidate_pool_top_slots.npz",
    "exists": true,
    "size": 12624246,
    "sha256": "08029943dece218fa5c1cbf016fe3bb9b4330dd83a05d759e2dea2d85ecceb15"
  },
  "queries": 8739,
  "slots": 5,
  "alts": 8,
  "same_video_only": true,
  "alt_count": 8,
  "original_query_oracle": {
    "0.5_top1_slot_recall": 39.37521576881409,
    "0.7_top1_slot_recall": 26.376014947891235,
    "0.5_top5_slot_recall": 56.56253695487976,
    "0.7_top5_slot_recall": 44.62753236293793,
    "best_iou_mean_top5_slots": 0.5165146589279175
  },
  "p0_oracle_with_alternatives": {
    "0.5_top1_slot_recall": 47.32806980609894,
    "0.7_top1_slot_recall": 42.087194323539734,
    "0.5_top5_slot_recall": 61.677539348602295,
    "0.7_top5_slot_recall": 54.98340725898743,
    "best_iou_mean_top5_slots": 0.602300226688385
  },
  "oracle_delta_vs_original": {
    "0.5_top1_slot_recall": 7.952854037284851,
    "0.7_top1_slot_recall": 15.711179375648499,
    "0.5_top5_slot_recall": 5.115002393722534,
    "0.7_top5_slot_recall": 10.3558748960495,
    "best_iou_mean_top5_slots": 0.08578556776046753
  }
}
```
