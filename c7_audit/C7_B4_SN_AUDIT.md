# C7-B4 SN audit

```json
{
  "status": "C7_B4_SN_AUDIT_COMPLETE",
  "official_val_used": false,
  "fixed_pool_only": true,
  "candidate_added": false,
  "NMS_modified": false,
  "evaluator_modified": false,
  "raw_start_end_logit_source": "results/rlem_c3_minimal/train_calib_evidence.jsonl.gz",
  "raw_start_end_logit_status": "raw_logit_partial_or_unavailable",
  "needed_candidate_keys": 871992,
  "matched_candidate_keys": 871910,
  "compared_systems": {
    "C7_B2_1_A4": {
      "name": "C7_B2_1_A4_stress",
      "query_count": 3732,
      "metrics": {
        "0.5-r1": 8.038585209003216,
        "0.5-r5": 38.960342979635584,
        "0.5-r10": 46.70418006430868,
        "0.5-r100": 49.59807073954984,
        "0.7-r1": 4.662379421221865,
        "0.7-r5": 26.339764201500536,
        "0.7-r10": 34.833869239013936,
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
          "loss_queries": 0
        },
        "0.5-r5": {
          "gain_queries": 0,
          "loss_queries": 0
        },
        "0.5-r10": {
          "gain_queries": 0,
          "loss_queries": 0
        },
        "0.7-r1": {
          "gain_queries": 0,
          "loss_queries": 0
        },
        "0.7-r5": {
          "gain_queries": 0,
          "loss_queries": 0
        },
        "0.7-r10": {
          "gain_queries": 0,
          "loss_queries": 0
        }
      },
      "dominance": {
        "corr_final_anchor": 0.9705625850035347,
        "corr_final_residual": -0.12539604149354613,
        "top1_changed_rate": 0.0,
        "top5_set_changed_rate": 0.0,
        "top10_set_changed_rate": 0.0,
        "rank_displacement_p50": 0.0,
        "rank_displacement_p90": 0.0,
        "rank_displacement_p99": 0.0
      }
    },
    "C7_B3_calibrated_A4": {
      "name": "C7_B3_calibrated_A4_stress",
      "query_count": 3732,
      "metrics": {
        "0.5-r1": 36.68274383708467,
        "0.5-r5": 48.4994640943194,
        "0.5-r10": 49.356913183279744,
        "0.5-r100": 49.59807073954984,
        "0.7-r1": 20.310825294748124,
        "0.7-r5": 38.531618435155416,
        "0.7-r10": 41.345123258306536,
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
          "gain_queries": 1074,
          "loss_queries": 5
        },
        "0.5-r5": {
          "gain_queries": 363,
          "loss_queries": 7
        },
        "0.5-r10": {
          "gain_queries": 105,
          "loss_queries": 6
        },
        "0.7-r1": {
          "gain_queries": 584,
          "loss_queries": 0
        },
        "0.7-r5": {
          "gain_queries": 456,
          "loss_queries": 1
        },
        "0.7-r10": {
          "gain_queries": 248,
          "loss_queries": 5
        }
      },
      "dominance": {
        "corr_final_anchor": 0.47054275077889285,
        "corr_final_residual": 0.5642708226138102,
        "top1_changed_rate": 0.7918006430868167,
        "top5_set_changed_rate": 0.9608788853161844,
        "top10_set_changed_rate": 0.9906216505894962,
        "rank_displacement_p50": 16.0,
        "rank_displacement_p90": 45.0,
        "rank_displacement_p99": 74.0
      }
    },
    "anchor_preserving_residual_best": {
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
    "MINUTE_like_raw_logit_scores": [
      {
        "name": "MINUTE_like_l_start_l_end_weak_fusion_stress",
        "query_count": 3732,
        "metrics": {
          "0.5-r1": 0.4287245444801715,
          "0.5-r5": 24.008574490889604,
          "0.5-r10": 37.540192926045016,
          "0.5-r100": 49.59807073954984,
          "0.7-r1": 0.02679528403001072,
          "0.7-r5": 15.112540192926046,
          "0.7-r10": 26.232583065380492,
          "0.7-r100": 41.934619506966776
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
            "loss_queries": 284
          },
          "0.5-r5": {
            "gain_queries": 0,
            "loss_queries": 558
          },
          "0.5-r10": {
            "gain_queries": 0,
            "loss_queries": 342
          },
          "0.7-r1": {
            "gain_queries": 0,
            "loss_queries": 173
          },
          "0.7-r5": {
            "gain_queries": 0,
            "loss_queries": 419
          },
          "0.7-r10": {
            "gain_queries": 1,
            "loss_queries": 322
          }
        },
        "dominance": {
          "corr_final_anchor": 0.9992839142276895,
          "corr_final_residual": -0.29342912871908994,
          "top1_changed_rate": 0.1385316184351554,
          "top5_set_changed_rate": 0.49517684887459806,
          "top10_set_changed_rate": 0.719989281886388,
          "rank_displacement_p50": 4.0,
          "rank_displacement_p90": 16.0,
          "rank_displacement_p99": 33.0
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
          "0.5-r1": -7.609860664523044,
          "0.5-r5": -14.95176848874598,
          "0.5-r10": -9.163987138263664,
          "0.5-r100": 0.0,
          "0.7-r1": -4.635584137191854,
          "0.7-r5": -11.22722400857449,
          "0.7-r10": -8.601286173633444,
          "0.7-r100": 0.05359056806002371
        },
        "hard_constraints_pass": false
      }
    ]
  }
}
```
