# C7-B4 selection decision

```json
{
  "status": "C7_B4_SN_AUDIT_ONLY",
  "official_val_used": false,
  "second_official_val": false,
  "post_val_adjustment": false,
  "C7_B3_official_decision_modified": false,
  "C7_B2_1_promoted_decision_modified": false,
  "NMS_modified": false,
  "evaluator_modified": false,
  "CONQUER_QDF_QAL_original_ML_VR_frozen": true,
  "full_fine_tuning": false,
  "primary_baseline": "C7-B2.1 A4 train-only reconstruction",
  "selection_rule": [
    "official-like stress split 0.7-r1/r5/r10 >= C7-B2.1 A4",
    "standard train_calib not materially below C7-B2.1 A4",
    "R@100 unchanged; hard exits=0; invalid/duplicate=0",
    "corr(final, anchor) > corr(final, residual); topK change rates controlled"
  ],
  "selected_candidate": {
    "name": "anchor_safe_{'alpha': 0.2, 'normalization': 'bounded_minmax', 'clip': 1.0}",
    "query_count": 3732,
    "metrics": {
      "0.5-r1": 3.1082529474812435,
      "0.5-r5": 26.956055734190784,
      "0.5-r10": 38.236870310825296,
      "0.5-r100": 49.59807073954984,
      "0.7-r1": 1.7684887459807075,
      "0.7-r5": 17.122186495176848,
      "0.7-r10": 27.679528403001072,
      "0.7-r100": 41.88102893890675
    },
    "movement": {
      "hard_positive_top100_query_exits": 0,
      "hard_positive_top100_query_entries": 0,
      "hard_positive_exit_ratio": 0.0,
      "invalid_span_count": 0,
      "duplicate_span_count_after_nms": 0,
      "fixed_pool_invariant": true
    },
    "gain_loss_vs_C7_B2_1_A4": {
      "0.5-r1": {
        "gain_queries": 0,
        "loss_queries": 184
      },
      "0.5-r5": {
        "gain_queries": 0,
        "loss_queries": 448
      },
      "0.5-r10": {
        "gain_queries": 0,
        "loss_queries": 316
      },
      "0.7-r1": {
        "gain_queries": 0,
        "loss_queries": 108
      },
      "0.7-r5": {
        "gain_queries": 0,
        "loss_queries": 344
      },
      "0.7-r10": {
        "gain_queries": 0,
        "loss_queries": 267
      }
    },
    "dominance": {
      "corr_final_anchor": 0.988694392397588,
      "corr_final_residual": -0.20407557813039606,
      "top1_changed_rate": 0.10262593783494105,
      "top5_set_changed_rate": 0.392550911039657,
      "top10_set_changed_rate": 0.6197749196141479,
      "rank_displacement_p50": 3.0,
      "rank_displacement_p90": 12.0,
      "rank_displacement_p99": 30.0
    },
    "config": {
      "alpha": 0.2,
      "normalization": "bounded_minmax",
      "clip": 1.0
    },
    "baseline_metrics": {
      "0.5-r1": 8.038585209003216,
      "0.5-r5": 38.960342979635584,
      "0.5-r10": 46.70418006430868,
      "0.5-r100": 49.59807073954984,
      "0.7-r1": 4.662379421221865,
      "0.7-r5": 26.339764201500536,
      "0.7-r10": 34.833869239013936,
      "0.7-r100": 41.88102893890675
    },
    "delta_vs_C7_B2_1_A4_stress": {
      "0.5-r1": -4.930332261521972,
      "0.5-r5": -12.0042872454448,
      "0.5-r10": -8.467309753483384,
      "0.5-r100": 0.0,
      "0.7-r1": -2.8938906752411575,
      "0.7-r5": -9.217577706323688,
      "0.7-r10": -7.154340836012864,
      "0.7-r100": 0.0
    },
    "standard_train_calib": {
      "metrics": {
        "0.5-r1": 47.79723080443987,
        "0.5-r5": 67.18159972536904,
        "0.5-r10": 72.77720562993477,
        "0.5-r100": 78.36136857764046,
        "0.7-r1": 29.32829843231491,
        "0.7-r5": 53.72468245794713,
        "0.7-r10": 62.57008811076782,
        "0.7-r100": 71.99908456345119
      },
      "delta_vs_C7_B2_1_A4": {
        "0.5-r1": -3.055269481634049,
        "0.5-r5": -5.33241789678452,
        "0.5-r10": -3.9134912461380083,
        "0.5-r100": 0.0,
        "0.7-r1": -1.705000572147842,
        "0.7-r5": -4.828927794942217,
        "0.7-r10": -3.8562764618377443,
        "0.7-r100": 0.0
      },
      "movement": {
        "hard_positive_top100_query_exits": 0,
        "hard_positive_top100_query_entries": 0,
        "hard_positive_exit_ratio": 0.0,
        "invalid_span_count": 0,
        "duplicate_span_count_after_nms": 0,
        "fixed_pool_invariant": true
      },
      "dominance": {
        "corr_final_anchor": 0.989210247193325,
        "corr_final_residual": 0.16999206737401013,
        "top1_changed_rate": 0.055612770339855816,
        "top5_set_changed_rate": 0.27863599954228174,
        "top10_set_changed_rate": 0.4751115688293855,
        "rank_displacement_p50": 1.0,
        "rank_displacement_p90": 9.0,
        "rank_displacement_p99": 24.0
      }
    },
    "hard_constraints_pass": false
  },
  "recommendation": "Stop here. Do not run official val automatically."
}
```
