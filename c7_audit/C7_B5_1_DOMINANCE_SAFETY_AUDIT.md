# C7-B5.1 dominance safety audit

```json
{
  "status": "C7_B5_1_DOMINANCE_SAFETY_AUDIT_COMPLETE",
  "official_val_used": false,
  "systems": {
    "C7_B1_anchor": {
      "dominance_safety": {
        "corr_final_anchor": 0.9999999999999999,
        "corr_final_S_MINUTE_like": 0.65554229307353,
        "corr_final_residual": -0.29027328260647345,
        "top1_changed_rate": 0.1345123258306538,
        "top5_set_changed_rate": 0.47883172561629156,
        "top10_set_changed_rate": 0.7092711682743837,
        "rank_displacement_p50": 4.0,
        "rank_displacement_p90": 16.0,
        "rank_displacement_p99": 34.0
      },
      "movement": {
        "hard_positive_top100_query_exits": 0,
        "hard_positive_top100_query_entries": 0,
        "hard_positive_exit_ratio": 0.0,
        "invalid_span_count": 0,
        "duplicate_span_count_after_nms": 0,
        "fixed_pool_invariant": true
      },
      "R100_delta": {
        "0.5-r100": 0.0,
        "0.7-r100": 0.0
      },
      "hard_safety_pass": true,
      "weak_residual_extra_pass": null
    },
    "C7_B2_1_A4": {
      "dominance_safety": {
        "corr_final_anchor": 0.9705625850035347,
        "corr_final_S_MINUTE_like": 0.6476974934859345,
        "corr_final_residual": -0.12539604149354613,
        "top1_changed_rate": 0.0,
        "top5_set_changed_rate": 0.0,
        "top10_set_changed_rate": 0.0,
        "rank_displacement_p50": 0.0,
        "rank_displacement_p90": 0.0,
        "rank_displacement_p99": 0.0
      },
      "movement": {
        "hard_positive_top100_query_exits": 0,
        "hard_positive_top100_query_entries": 0,
        "hard_positive_exit_ratio": 0.0,
        "invalid_span_count": 0,
        "duplicate_span_count_after_nms": 0,
        "fixed_pool_invariant": true
      },
      "R100_delta": {
        "0.5-r100": 0.0,
        "0.7-r100": 0.0
      },
      "hard_safety_pass": true,
      "weak_residual_extra_pass": null
    },
    "C7_B3_calibrated_A4_diagnostic": {
      "dominance_safety": {
        "corr_final_anchor": 0.47054275077889285,
        "corr_final_S_MINUTE_like": 0.33886567783906385,
        "corr_final_residual": 0.5642708226138102,
        "top1_changed_rate": 0.7918006430868167,
        "top5_set_changed_rate": 0.9608788853161844,
        "top10_set_changed_rate": 0.9906216505894962,
        "rank_displacement_p50": 16.0,
        "rank_displacement_p90": 45.0,
        "rank_displacement_p99": 74.0
      },
      "movement": {
        "hard_positive_top100_query_exits": 0,
        "hard_positive_top100_query_entries": 0,
        "hard_positive_exit_ratio": 0.0,
        "invalid_span_count": 0,
        "duplicate_span_count_after_nms": 0,
        "fixed_pool_invariant": true
      },
      "R100_delta": {
        "0.5-r100": 0.0,
        "0.7-r100": 0.0
      },
      "hard_safety_pass": true,
      "weak_residual_extra_pass": null
    },
    "C7_B4_best_anchor_safe_diagnostic": {
      "dominance_safety": {
        "corr_final_anchor": 0.9698484862355712,
        "corr_final_S_MINUTE_like": 0.6317385542675453,
        "corr_final_residual": -0.08018836718252766,
        "top1_changed_rate": 0.09860664523043944,
        "top5_set_changed_rate": 0.3595927116827438,
        "top10_set_changed_rate": 0.5656484458735263,
        "rank_displacement_p50": 2.0,
        "rank_displacement_p90": 7.0,
        "rank_displacement_p99": 16.0
      },
      "movement": {
        "hard_positive_top100_query_exits": 0,
        "hard_positive_top100_query_entries": 0,
        "hard_positive_exit_ratio": 0.0,
        "invalid_span_count": 0,
        "duplicate_span_count_after_nms": 0,
        "fixed_pool_invariant": true
      },
      "R100_delta": {
        "0.5-r100": 0.0,
        "0.7-r100": 0.0
      },
      "hard_safety_pass": true,
      "weak_residual_extra_pass": null
    },
    "S_MINUTE_like": {
      "dominance_safety": {
        "corr_final_anchor": 0.6555422930735301,
        "corr_final_S_MINUTE_like": 1.0,
        "corr_final_residual": -0.2115802746868933,
        "top1_changed_rate": 0.6069131832797428,
        "top5_set_changed_rate": 0.934887459807074,
        "top10_set_changed_rate": 0.9860664523043944,
        "rank_displacement_p50": 17.0,
        "rank_displacement_p90": 46.0,
        "rank_displacement_p99": 71.0
      },
      "movement": {
        "hard_positive_top100_query_exits": 1,
        "hard_positive_top100_query_entries": 0,
        "hard_positive_exit_ratio": 0.0002679528403001072,
        "invalid_span_count": 0,
        "duplicate_span_count_after_nms": 0,
        "fixed_pool_invariant": true
      },
      "R100_delta": {
        "0.5-r100": -0.026795284030008304,
        "0.7-r100": -0.026795284030008304
      },
      "hard_safety_pass": false,
      "weak_residual_extra_pass": null
    },
    "S_logprob_like_diagnostic": {
      "dominance_safety": {
        "corr_final_anchor": 0.7396611510465213,
        "corr_final_S_MINUTE_like": 0.3064157673709828,
        "corr_final_residual": 0.015976417257723568,
        "top1_changed_rate": 0.4370310825294748,
        "top5_set_changed_rate": 0.9364951768488746,
        "top10_set_changed_rate": 0.9959807073954984,
        "rank_displacement_p50": 12.0,
        "rank_displacement_p90": 37.0,
        "rank_displacement_p99": 66.0
      },
      "movement": {
        "hard_positive_top100_query_exits": 1,
        "hard_positive_top100_query_entries": 0,
        "hard_positive_exit_ratio": 0.0002679528403001072,
        "invalid_span_count": 0,
        "duplicate_span_count_after_nms": 0,
        "fixed_pool_invariant": true
      },
      "R100_delta": {
        "0.5-r100": -0.026795284030008304,
        "0.7-r100": -0.026795284030008304
      },
      "hard_safety_pass": false,
      "weak_residual_extra_pass": null
    },
    "S_MINUTE_like_plus_weak_residual_beta_0.02": {
      "dominance_safety": {
        "corr_final_anchor": 0.6551481714258419,
        "corr_final_S_MINUTE_like": 0.9999957198021279,
        "corr_final_residual": -0.20899274050442251,
        "top1_changed_rate": 0.6082529474812433,
        "top5_set_changed_rate": 0.9354233654876741,
        "top10_set_changed_rate": 0.9860664523043944,
        "rank_displacement_p50": 17.0,
        "rank_displacement_p90": 46.0,
        "rank_displacement_p99": 71.0
      },
      "movement": {
        "hard_positive_top100_query_exits": 1,
        "hard_positive_top100_query_entries": 0,
        "hard_positive_exit_ratio": 0.0002679528403001072,
        "invalid_span_count": 0,
        "duplicate_span_count_after_nms": 0,
        "fixed_pool_invariant": true
      },
      "R100_delta": {
        "0.5-r100": -0.026795284030008304,
        "0.7-r100": -0.026795284030008304
      },
      "hard_safety_pass": false,
      "weak_residual_extra_pass": false
    },
    "S_MINUTE_like_plus_weak_residual_beta_0.05": {
      "dominance_safety": {
        "corr_final_anchor": 0.6545466283472374,
        "corr_final_S_MINUTE_like": 0.9999732589041015,
        "corr_final_residual": -0.20510405614743193,
        "top1_changed_rate": 0.6117363344051447,
        "top5_set_changed_rate": 0.9372990353697749,
        "top10_set_changed_rate": 0.9863344051446945,
        "rank_displacement_p50": 17.0,
        "rank_displacement_p90": 46.0,
        "rank_displacement_p99": 71.0
      },
      "movement": {
        "hard_positive_top100_query_exits": 1,
        "hard_positive_top100_query_entries": 0,
        "hard_positive_exit_ratio": 0.0002679528403001072,
        "invalid_span_count": 0,
        "duplicate_span_count_after_nms": 0,
        "fixed_pool_invariant": true
      },
      "R100_delta": {
        "0.5-r100": -0.026795284030008304,
        "0.7-r100": -0.026795284030008304
      },
      "hard_safety_pass": false,
      "weak_residual_extra_pass": false
    },
    "S_MINUTE_like_plus_weak_residual_beta_0.1": {
      "dominance_safety": {
        "corr_final_anchor": 0.6535146578988659,
        "corr_final_S_MINUTE_like": 0.9998928387302697,
        "corr_final_residual": -0.1986035100778987,
        "top1_changed_rate": 0.6187031082529475,
        "top5_set_changed_rate": 0.937566988210075,
        "top10_set_changed_rate": 0.987406216505895,
        "rank_displacement_p50": 17.0,
        "rank_displacement_p90": 46.0,
        "rank_displacement_p99": 71.0
      },
      "movement": {
        "hard_positive_top100_query_exits": 1,
        "hard_positive_top100_query_entries": 0,
        "hard_positive_exit_ratio": 0.0002679528403001072,
        "invalid_span_count": 0,
        "duplicate_span_count_after_nms": 0,
        "fixed_pool_invariant": true
      },
      "R100_delta": {
        "0.5-r100": -0.026795284030008304,
        "0.7-r100": -0.026795284030008304
      },
      "hard_safety_pass": false,
      "weak_residual_extra_pass": false
    }
  }
}
```
