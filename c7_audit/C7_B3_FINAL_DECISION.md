# C7-B3 final decision

```json
{
  "status": "C7_B3_FREEZE_REVIEW_PASS",
  "official_val_used": false,
  "official_val_run": false,
  "post_val_adjustment": false,
  "second_official_val": false,
  "selected_candidate": "calibrated_A4",
  "selected_summary": {
    "median_delta_0.7_r1": 7.171281433921884,
    "median_delta_0.5_r1": 14.034075708935259,
    "median_delta_0.7_r5": 8.959340513313755,
    "median_delta_0.7_r10": 4.927034819952077,
    "worst_delta_0.7_r1": 6.378454996456412,
    "worst_delta_0.5_r1": 12.402551381998578,
    "instability_0.7_r1": 0.7512235374000495,
    "hard_exits": 0,
    "invalid": 0,
    "duplicate": 0
  },
  "worst_group_delta_0.7_r1": 0.0,
  "train_calib_final_review": {
    "name": "calibrated_A4_train_calib_final_review",
    "query_count": 8739,
    "metrics": {
      "0.5-r1": 64.77857878475798,
      "0.5-r5": 77.40016020139603,
      "0.5-r10": 78.22405309531983,
      "0.5-r100": 78.36136857764046,
      "0.7-r1": 38.65430827325781,
      "0.7-r5": 66.6323377960865,
      "0.7-r10": 71.46126559102872,
      "0.7-r100": 71.99908456345119
    },
    "movement": {
      "hard_positive_top100_query_exits": 0,
      "hard_positive_top100_query_entries": 0,
      "hard_positive_exit_ratio": 0.0,
      "invalid_span_count": 0,
      "duplicate_span_count_after_nms": 0,
      "fixed_pool_invariant": true
    },
    "rank_gain_loss_vs_anchor": {
      "0.5-r1": {
        "gain_queries": 1916,
        "loss_queries": 5
      },
      "0.5-r5": {
        "gain_queries": 1065,
        "loss_queries": 8
      },
      "0.5-r10": {
        "gain_queries": 528,
        "loss_queries": 7
      },
      "0.7-r1": {
        "gain_queries": 1074,
        "loss_queries": 0
      },
      "0.7-r5": {
        "gain_queries": 1334,
        "loss_queries": 1
      },
      "0.7-r10": {
        "gain_queries": 951,
        "loss_queries": 5
      }
    }
  },
  "baseline_A4_train_calib": {
    "name": "A4_train_calib_final_review",
    "query_count": 8739,
    "metrics": {
      "0.5-r1": 50.85250028607392,
      "0.5-r5": 72.51401762215356,
      "0.5-r10": 76.69069687607278,
      "0.5-r100": 78.36136857764046,
      "0.7-r1": 31.03329900446275,
      "0.7-r5": 58.55361025288935,
      "0.7-r10": 66.42636457260556,
      "0.7-r100": 71.99908456345119
    },
    "movement": {
      "hard_positive_top100_query_exits": 0,
      "hard_positive_top100_query_entries": 0,
      "hard_positive_exit_ratio": 0.0,
      "invalid_span_count": 0,
      "duplicate_span_count_after_nms": 0,
      "fixed_pool_invariant": true
    },
    "rank_gain_loss_vs_anchor": {
      "0.5-r1": {
        "gain_queries": 694,
        "loss_queries": 0
      },
      "0.5-r5": {
        "gain_queries": 630,
        "loss_queries": 0
      },
      "0.5-r10": {
        "gain_queries": 387,
        "loss_queries": 0
      },
      "0.7-r1": {
        "gain_queries": 408,
        "loss_queries": 0
      },
      "0.7-r5": {
        "gain_queries": 627,
        "loss_queries": 0
      },
      "0.7-r10": {
        "gain_queries": 506,
        "loss_queries": 0
      }
    }
  },
  "gate_pass": true,
  "fixed_pool_invariant": true,
  "R100_unchanged": true,
  "hard_exits_invalid_duplicate_clean": true,
  "evaluator_modified": false,
  "nms_modified": false,
  "C4_C6_artifacts_modified": false,
  "enter_C7_C": false,
  "enter_C8": false,
  "request": "Stop and request human review before any official-val one-shot."
}
```
