# C6-B0 Candidate Generation Diagnostic

- Status: `C6_B0_NEGATIVE`
- Scope: `train_calib only`
- Official val used: `false`
- B0 policy: R@1 is primary; R@5 is the safety/secondary check.
- C4_final remains the main frozen result; B0 is not a final VCMR result.

## Candidate oracle metrics

```json
{
  "original_candidates": {
    "gt_video_oracle_r05": 77.59469046801694,
    "gt_video_oracle_r07": 68.90948621123698,
    "best_iou_mean": 0.6706794479043179,
    "best_iou_median": 0.8288888931274414,
    "best_iou_candidate_rank_mean": 5.720005679397984,
    "best_iou_candidate_rank_median": 3.0,
    "nms_retained_pre_nms_best_iou_rate": 80.59274516535073,
    "selected_span_miou_oracle_upper_bound": 0.6706794479043179,
    "gt_video_missing_count": 1696,
    "gt_video_present_query_count": 7043,
    "candidate_count_min": 0,
    "candidate_count_max": 144,
    "candidate_count_mean": 16.237555784414692,
    "candidate_count_mean_when_gt_video_present": 20.147664347579155,
    "topk_proposal_recall": {
      "top1_r05": 61.986497310905136,
      "top1_r07": 40.14189266506465,
      "top5_r05": 73.25780981805698,
      "top5_r07": 60.54468474653851,
      "top10_r05": 75.98123355074951,
      "top10_r07": 66.01441812564367,
      "top20_r05": 77.2399588053553,
      "top20_r07": 68.47465385055499,
      "top100_r05": 77.59469046801694,
      "top100_r07": 69.22988900331846
    }
  },
  "boundary_oracle_candidates": {
    "gt_video_oracle_r05": 78.96784529122326,
    "gt_video_oracle_r07": 71.19807758324751,
    "best_iou_mean": 0.6624897332391616,
    "best_iou_median": 0.8066561222076416,
    "best_iou_candidate_rank_mean": 3.0350702825500497,
    "best_iou_candidate_rank_median": 2.0,
    "nms_retained_pre_nms_best_iou_rate": 48.02608994164092,
    "selected_span_miou_oracle_upper_bound": 0.6624897332391616,
    "gt_video_missing_count": 1696,
    "gt_video_present_query_count": 7043,
    "candidate_count_min": 0,
    "candidate_count_max": 144,
    "candidate_count_mean": 16.237555784414692,
    "candidate_count_mean_when_gt_video_present": 20.147664347579155,
    "topk_proposal_recall": {
      "top1_r05": 63.08502116947019,
      "top1_r07": 42.3046115116146,
      "top5_r05": 76.57626730747225,
      "top5_r07": 65.7397871610024,
      "top10_r05": 78.36136857764046,
      "top10_r07": 69.93935232864172,
      "top20_r05": 78.85341572262273,
      "top20_r07": 71.17519166952741,
      "top100_r05": 78.96784529122326,
      "top100_r07": 71.46126559102872
    },
    "delta_vs_original_candidates": {
      "gt_video_oracle_r05": 1.3731548232063204,
      "gt_video_oracle_r07": 2.2885913720105293,
      "best_iou_mean": -0.008189714665156278,
      "best_iou_median": -0.022232770919799805,
      "best_iou_candidate_rank_mean": -2.684935396847934,
      "nms_retained_pre_nms_best_iou_rate": -32.566655223709816
    }
  },
  "hybrid_candidates": {
    "gt_video_oracle_r05": 78.96784529122326,
    "gt_video_oracle_r07": 71.20952054010756,
    "best_iou_mean": 0.6625250584546817,
    "best_iou_median": 0.8066666722297668,
    "best_iou_candidate_rank_mean": 3.0369160868947893,
    "best_iou_candidate_rank_median": 2.0,
    "nms_retained_pre_nms_best_iou_rate": 80.59274516535073,
    "selected_span_miou_oracle_upper_bound": 0.6625250584546817,
    "gt_video_missing_count": 1696,
    "gt_video_present_query_count": 7043,
    "candidate_count_min": 0,
    "candidate_count_max": 144,
    "candidate_count_mean": 16.237555784414692,
    "candidate_count_mean_when_gt_video_present": 20.147664347579155,
    "topk_proposal_recall": {
      "top1_r05": 63.08502116947019,
      "top1_r07": 42.3046115116146,
      "top5_r05": 76.57626730747225,
      "top5_r07": 65.75123011786246,
      "top10_r05": 78.36136857764046,
      "top10_r07": 69.95079528550177,
      "top20_r05": 78.85341572262273,
      "top20_r07": 71.18663462638746,
      "top100_r05": 78.96784529122326,
      "top100_r07": 71.47270854788877
    },
    "delta_vs_original_candidates": {
      "gt_video_oracle_r05": 1.3731548232063204,
      "gt_video_oracle_r07": 2.3000343288705807,
      "best_iou_mean": -0.008154389449636201,
      "best_iou_median": -0.02222222089767456,
      "best_iou_candidate_rank_mean": -2.6830895925031943,
      "nms_retained_pre_nms_best_iou_rate": 0.0
    }
  }
}
```

## Fixed-video pseudo VCMR

```json
{
  "original_candidates": {
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
    "candidate_replacement_rate_top100_slots": 0.0,
    "video_multiset_drift_top100_slots": 0.0,
    "video_slot_drift_top100_slots": 0.0
  },
  "boundary_oracle_candidates": {
    "metrics": {
      "0.5-r1": 39.55830186520197,
      "0.5-r5": 58.473509554868976,
      "0.5-r10": 65.64824350612199,
      "0.5-r100": 78.28126787962009,
      "0.7-r1": 26.99393523286417,
      "0.7-r5": 47.53404279665865,
      "0.7-r10": 55.58988442613571,
      "0.7-r100": 70.29408399130335
    },
    "candidate_replacement_rate_top100_slots": 0.8840794141206088,
    "video_multiset_drift_top100_slots": 0.0,
    "video_slot_drift_top100_slots": 0.0,
    "delta_vs_pseudo_C4_final_fixed_video_baseline": {
      "0.5-r1": 0.1830873097608432,
      "0.5-r5": -3.8105046343975317,
      "0.5-r10": -3.8333905481176345,
      "0.5-r100": 2.0597322348094735,
      "0.7-r1": 0.6179196704428414,
      "0.7-r5": -1.1099668154251106,
      "0.7-r10": -0.6522485410230061,
      "0.7-r100": 4.9891291909829505
    },
    "r1_improved": true,
    "r5_collapse": true,
    "hard_positive_top100_query_exits": 3,
    "hard_positive_top100_query_entries": 183,
    "top1_changed_ratio": 0.12049433573635428
  },
  "hybrid_candidates": {
    "metrics": {
      "0.5-r1": 39.55830186520197,
      "0.5-r5": 58.473509554868976,
      "0.5-r10": 65.64824350612199,
      "0.5-r100": 78.28126787962009,
      "0.7-r1": 26.99393523286417,
      "0.7-r5": 47.53404279665865,
      "0.7-r10": 55.58988442613571,
      "0.7-r100": 70.29408399130335
    },
    "candidate_replacement_rate_top100_slots": 0.8840771255292368,
    "video_multiset_drift_top100_slots": 0.0,
    "video_slot_drift_top100_slots": 0.0,
    "delta_vs_pseudo_C4_final_fixed_video_baseline": {
      "0.5-r1": 0.1830873097608432,
      "0.5-r5": -3.8105046343975317,
      "0.5-r10": -3.8333905481176345,
      "0.5-r100": 2.0597322348094735,
      "0.7-r1": 0.6179196704428414,
      "0.7-r5": -1.1099668154251106,
      "0.7-r10": -0.6522485410230061,
      "0.7-r100": 4.9891291909829505
    },
    "r1_improved": true,
    "r5_collapse": true,
    "hard_positive_top100_query_exits": 3,
    "hard_positive_top100_query_entries": 183,
    "top1_changed_ratio": 0.12049433573635428
  }
}
```

## Decision

```json
{
  "status": "C6_B0_NEGATIVE",
  "r1_primary_r5_safety_policy": "R@1 is the primary observation; R@5 is a safety/secondary constraint. B0 is not a final VCMR result.",
  "variant_decisions": {
    "boundary_oracle_candidates": {
      "candidate_generation_oracle_positive": false,
      "r1_primary_delta_min": 0.1830873097608432,
      "r5_safety_delta_min": -3.8105046343975317,
      "no_severe_r5_collapse": false,
      "recommend_c6_b1_review": false
    },
    "hybrid_candidates": {
      "candidate_generation_oracle_positive": false,
      "r1_primary_delta_min": 0.1830873097608432,
      "r5_safety_delta_min": -3.8105046343975317,
      "no_severe_r5_collapse": false,
      "recommend_c6_b1_review": false
    }
  },
  "official_val_used": false,
  "C6_B1_used": false,
  "C6_C_used": false
}
```
