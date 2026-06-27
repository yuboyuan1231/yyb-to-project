# C7-B3 fixed-pool LambdaRank

```json
{
  "status": "C7_B3_FIXED_POOL_LAMBDARANK_COMPLETE",
  "official_val_used": false,
  "pair_count": 428672,
  "feature_names": [
    "C7_B1_score",
    "C7_B2_video_residual",
    "proposal_confidence",
    "span_quality_logit",
    "rank_position",
    "score_margin",
    "video_rank",
    "within_video_span_rank",
    "same_video_candidate_count"
  ],
  "records": {
    "train_core": {
      "name": "lambdarank_train_core",
      "query_count": 4268,
      "metrics": {
        "0.5-r1": 47.72727272727273,
        "0.5-r5": 76.4526710402999,
        "0.5-r10": 78.11621368322399,
        "0.5-r100": 78.25679475164011,
        "0.7-r1": 17.432052483598877,
        "0.7-r5": 58.36457357075914,
        "0.7-r10": 71.22774133083412,
        "0.7-r100": 72.65698219306466
      },
      "movement": {
        "hard_positive_top100_query_exits": 4,
        "hard_positive_top100_query_entries": 3,
        "hard_positive_exit_ratio": 0.0009372071227741331,
        "invalid_span_count": 0,
        "duplicate_span_count_after_nms": 0,
        "fixed_pool_invariant": true
      },
      "rank_gain_loss_vs_anchor": {
        "0.5-r1": {
          "gain_queries": 827,
          "loss_queries": 589
        },
        "0.5-r5": {
          "gain_queries": 527,
          "loss_queries": 49
        },
        "0.5-r10": {
          "gain_queries": 271,
          "loss_queries": 6
        },
        "0.7-r1": {
          "gain_queries": 417,
          "loss_queries": 801
        },
        "0.7-r5": {
          "gain_queries": 697,
          "loss_queries": 404
        },
        "0.7-r10": {
          "gain_queries": 504,
          "loss_queries": 61
        }
      }
    },
    "calib_A": {
      "name": "lambdarank_calib_A",
      "query_count": 1411,
      "metrics": {
        "0.5-r1": 46.13749114103473,
        "0.5-r5": 76.25797306874557,
        "0.5-r10": 77.95889440113395,
        "0.5-r100": 78.24238128986535,
        "0.7-r1": 18.001417434443656,
        "0.7-r5": 57.76045357902197,
        "0.7-r10": 69.87951807228916,
        "0.7-r100": 71.36782423812899
      },
      "movement": {
        "hard_positive_top100_query_exits": 3,
        "hard_positive_top100_query_entries": 3,
        "hard_positive_exit_ratio": 0.002126151665485471,
        "invalid_span_count": 0,
        "duplicate_span_count_after_nms": 0,
        "fixed_pool_invariant": true
      },
      "rank_gain_loss_vs_anchor": {
        "0.5-r1": {
          "gain_queries": 255,
          "loss_queries": 234
        },
        "0.5-r5": {
          "gain_queries": 178,
          "loss_queries": 16
        },
        "0.5-r10": {
          "gain_queries": 89,
          "loss_queries": 3
        },
        "0.7-r1": {
          "gain_queries": 143,
          "loss_queries": 255
        },
        "0.7-r5": {
          "gain_queries": 242,
          "loss_queries": 148
        },
        "0.7-r10": {
          "gain_queries": 169,
          "loss_queries": 26
        }
      }
    },
    "calib_B": {
      "name": "lambdarank_calib_B",
      "query_count": 1344,
      "metrics": {
        "0.5-r1": 47.32142857142857,
        "0.5-r5": 77.5297619047619,
        "0.5-r10": 79.01785714285714,
        "0.5-r100": 79.16666666666667,
        "0.7-r1": 19.12202380952381,
        "0.7-r5": 57.217261904761905,
        "0.7-r10": 70.61011904761905,
        "0.7-r100": 71.72619047619048
      },
      "movement": {
        "hard_positive_top100_query_exits": 3,
        "hard_positive_top100_query_entries": 1,
        "hard_positive_exit_ratio": 0.002232142857142857,
        "invalid_span_count": 0,
        "duplicate_span_count_after_nms": 0,
        "fixed_pool_invariant": true
      },
      "rank_gain_loss_vs_anchor": {
        "0.5-r1": {
          "gain_queries": 247,
          "loss_queries": 186
        },
        "0.5-r5": {
          "gain_queries": 164,
          "loss_queries": 17
        },
        "0.5-r10": {
          "gain_queries": 81,
          "loss_queries": 4
        },
        "0.7-r1": {
          "gain_queries": 150,
          "loss_queries": 255
        },
        "0.7-r5": {
          "gain_queries": 225,
          "loss_queries": 153
        },
        "0.7-r10": {
          "gain_queries": 156,
          "loss_queries": 22
        }
      }
    },
    "calib_C": {
      "name": "lambdarank_calib_C",
      "query_count": 1246,
      "metrics": {
        "0.5-r1": 46.54895666131621,
        "0.5-r5": 75.92295345104334,
        "0.5-r10": 77.20706260032102,
        "0.5-r100": 77.4478330658106,
        "0.7-r1": 17.656500802568218,
        "0.7-r5": 57.9454253611557,
        "0.7-r10": 69.26163723916532,
        "0.7-r100": 70.9470304975923
      },
      "movement": {
        "hard_positive_top100_query_exits": 2,
        "hard_positive_top100_query_entries": 4,
        "hard_positive_exit_ratio": 0.0016051364365971107,
        "invalid_span_count": 0,
        "duplicate_span_count_after_nms": 0,
        "fixed_pool_invariant": true
      },
      "rank_gain_loss_vs_anchor": {
        "0.5-r1": {
          "gain_queries": 241,
          "loss_queries": 195
        },
        "0.5-r5": {
          "gain_queries": 155,
          "loss_queries": 10
        },
        "0.5-r10": {
          "gain_queries": 64,
          "loss_queries": 2
        },
        "0.7-r1": {
          "gain_queries": 119,
          "loss_queries": 219
        },
        "0.7-r5": {
          "gain_queries": 201,
          "loss_queries": 103
        },
        "0.7-r10": {
          "gain_queries": 127,
          "loss_queries": 14
        }
      }
    },
    "stress_split": {
      "name": "lambdarank_stress_split",
      "query_count": 1748,
      "metrics": {
        "0.5-r1": 54.63386727688787,
        "0.5-r5": 89.75972540045767,
        "0.5-r10": 91.81922196796339,
        "0.5-r100": 91.99084668192219,
        "0.7-r1": 24.71395881006865,
        "0.7-r5": 68.82151029748283,
        "0.7-r10": 82.66590389016018,
        "0.7-r100": 84.66819221967964
      },
      "movement": {
        "hard_positive_top100_query_exits": 1,
        "hard_positive_top100_query_entries": 8,
        "hard_positive_exit_ratio": 0.0005720823798627002,
        "invalid_span_count": 0,
        "duplicate_span_count_after_nms": 0,
        "fixed_pool_invariant": true
      },
      "rank_gain_loss_vs_anchor": {
        "0.5-r1": {
          "gain_queries": 463,
          "loss_queries": 258
        },
        "0.5-r5": {
          "gain_queries": 325,
          "loss_queries": 22
        },
        "0.5-r10": {
          "gain_queries": 172,
          "loss_queries": 2
        },
        "0.7-r1": {
          "gain_queries": 275,
          "loss_queries": 318
        },
        "0.7-r5": {
          "gain_queries": 382,
          "loss_queries": 194
        },
        "0.7-r10": {
          "gain_queries": 284,
          "loss_queries": 44
        }
      }
    },
    "train_calib_final_review": {
      "name": "lambdarank_train_calib_final_review",
      "query_count": 8739,
      "metrics": {
        "0.5-r1": 47.579814624098866,
        "0.5-r5": 76.33596521341114,
        "0.5-r10": 78.15539535415951,
        "0.5-r100": 78.31559675020026,
        "0.7-r1": 18.022657054582904,
        "0.7-r5": 57.76404622954571,
        "0.7-r10": 70.82046000686577,
        "0.7-r100": 72.28515848495252
      },
      "movement": {
        "hard_positive_top100_query_exits": 16,
        "hard_positive_top100_query_entries": 12,
        "hard_positive_exit_ratio": 0.001830873097608422,
        "invalid_span_count": 0,
        "duplicate_span_count_after_nms": 0,
        "fixed_pool_invariant": true
      },
      "rank_gain_loss_vs_anchor": {
        "0.5-r1": {
          "gain_queries": 1706,
          "loss_queries": 1298
        },
        "0.5-r5": {
          "gain_queries": 1071,
          "loss_queries": 107
        },
        "0.5-r10": {
          "gain_queries": 531,
          "loss_queries": 16
        },
        "0.7-r1": {
          "gain_queries": 896,
          "loss_queries": 1625
        },
        "0.7-r5": {
          "gain_queries": 1461,
          "loss_queries": 903
        },
        "0.7-r10": {
          "gain_queries": 1025,
          "loss_queries": 135
        }
      }
    }
  },
  "summary": {
    "median_delta_0.7_r1": -12.270050838800392,
    "median_delta_0.5_r1": -3.641997641881943,
    "median_delta_0.7_r5": 0.7983610306078219,
    "median_delta_0.7_r10": 4.061700489049009,
    "worst_delta_0.7_r1": -12.5,
    "worst_delta_0.5_r1": -5.81148121899362,
    "instability_0.7_r1": 0.19339181342003958,
    "hard_exits": 9,
    "invalid": 0,
    "duplicate": 0
  }
}
```
