# C7 B2 1 EVICTION AWARE UTILITY

```json
{
  "official_val_used": false,
  "records": [
    {
      "name": "evict_lam0.05_tail0.1_beyond5",
      "metrics": {
        "0.5-r1": 47.62558645153908,
        "0.5-r5": 69.8478086737613,
        "0.5-r10": 75.67227371552809,
        "0.5-r100": 78.36136857764046,
        "0.7-r1": 29.2825266048747,
        "0.7-r5": 55.00629362627303,
        "0.7-r10": 64.12633024373498,
        "0.7-r100": 71.99908456345119
      },
      "movement": {
        "hard_positive_top100_query_exits": 0,
        "hard_positive_top100_query_entries": 0,
        "hard_positive_exit_ratio": 0.0,
        "video_slot_drift_rate": 0.40985697212096117,
        "invalid_span_count": 0,
        "duplicate_span_count_after_nms": 0
      },
      "rank_gain_loss_vs_C7_B1_frozen": {
        "0.5-r1": {
          "gain_queries": 412,
          "loss_queries": 0
        },
        "0.5-r5": {
          "gain_queries": 397,
          "loss_queries": 0
        },
        "0.5-r10": {
          "gain_queries": 298,
          "loss_queries": 0
        },
        "0.7-r1": {
          "gain_queries": 255,
          "loss_queries": 0
        },
        "0.7-r5": {
          "gain_queries": 321,
          "loss_queries": 4
        },
        "0.7-r10": {
          "gain_queries": 309,
          "loss_queries": 4
        }
      },
      "top100_lost_query_count": 0,
      "top100_lost_query_hash": "4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945",
      "config": {
        "lambda_evict": 0.05,
        "tail_coverage_weight": 0.1,
        "apply_only_beyond_rank": 5
      },
      "delta_vs_C6_B1": {
        "0.5-r1": 8.044398672617007,
        "0.5-r5": 7.4150360453141175,
        "0.5-r10": 6.064767135827893,
        "0.5-r100": 2.048289277949422,
        "0.7-r1": 2.6204371209520545,
        "0.7-r5": 6.156310790708318,
        "0.7-r10": 7.689667009955372,
        "0.7-r100": 6.533928367090056
      },
      "delta_vs_C6_B2": {
        "0.5-r1": 7.8040965785558996,
        "0.5-r5": 7.426479002174169,
        "0.5-r10": 6.053324178967841,
        "0.5-r100": 2.071175191669525,
        "0.7-r1": 2.2313765877102654,
        "0.7-r5": 6.110538963268105,
        "0.7-r10": 7.632452225655115,
        "0.7-r100": 6.476713582789785
      },
      "delta_vs_C7_B1_frozen": {
        "0.5-r1": 4.714498226341689,
        "0.5-r5": 4.5428538734409045,
        "0.5-r10": 3.4100011442956912,
        "0.5-r100": 0.0,
        "0.7-r1": 2.917953999313422,
        "0.7-r5": 3.6274173246366814,
        "0.7-r10": 3.490101842316058,
        "0.7-r100": 0.0
      },
      "status_label": "C7_B2_1_SAFE_BUT_R1_DROP"
    },
    {
      "name": "evict_lam0.05_tail0.1_beyond10",
      "metrics": {
        "0.5-r1": 47.62558645153908,
        "0.5-r5": 69.79059388946104,
        "0.5-r10": 75.61505893122784,
        "0.5-r100": 78.36136857764046,
        "0.7-r1": 29.2825266048747,
        "0.7-r5": 54.97196475569287,
        "0.7-r10": 64.10344433001488,
        "0.7-r100": 71.99908456345119
      },
      "movement": {
        "hard_positive_top100_query_exits": 0,
        "hard_positive_top100_query_entries": 0,
        "hard_positive_exit_ratio": 0.0,
        "video_slot_drift_rate": 0.40985697212096117,
        "invalid_span_count": 0,
        "duplicate_span_count_after_nms": 0
      },
      "rank_gain_loss_vs_C7_B1_frozen": {
        "0.5-r1": {
          "gain_queries": 412,
          "loss_queries": 0
        },
        "0.5-r5": {
          "gain_queries": 392,
          "loss_queries": 0
        },
        "0.5-r10": {
          "gain_queries": 294,
          "loss_queries": 1
        },
        "0.7-r1": {
          "gain_queries": 255,
          "loss_queries": 0
        },
        "0.7-r5": {
          "gain_queries": 314,
          "loss_queries": 0
        },
        "0.7-r10": {
          "gain_queries": 309,
          "loss_queries": 6
        }
      },
      "top100_lost_query_count": 0,
      "top100_lost_query_hash": "4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945",
      "config": {
        "lambda_evict": 0.05,
        "tail_coverage_weight": 0.1,
        "apply_only_beyond_rank": 10
      },
      "delta_vs_C6_B1": {
        "0.5-r1": 8.044398672617007,
        "0.5-r5": 7.3578212610138465,
        "0.5-r10": 6.007552351527636,
        "0.5-r100": 2.048289277949422,
        "0.7-r1": 2.6204371209520545,
        "0.7-r5": 6.121981920128164,
        "0.7-r10": 7.666781096235269,
        "0.7-r100": 6.533928367090056
      },
      "delta_vs_C6_B2": {
        "0.5-r1": 7.8040965785558996,
        "0.5-r5": 7.369264217873898,
        "0.5-r10": 5.9961093946675845,
        "0.5-r100": 2.071175191669525,
        "0.7-r1": 2.2313765877102654,
        "0.7-r5": 6.076210092687951,
        "0.7-r10": 7.609566311935012,
        "0.7-r100": 6.476713582789785
      },
      "delta_vs_C7_B1_frozen": {
        "0.5-r1": 4.714498226341689,
        "0.5-r5": 4.4856390891406335,
        "0.5-r10": 3.3527863599954344,
        "0.5-r100": 0.0,
        "0.7-r1": 2.917953999313422,
        "0.7-r5": 3.5930884540565273,
        "0.7-r10": 3.467215928595955,
        "0.7-r100": 0.0
      },
      "status_label": "C7_B2_1_SAFE_BUT_R1_DROP"
    },
    {
      "name": "evict_lam0.05_tail0.1_beyond20",
      "metrics": {
        "0.5-r1": 47.62558645153908,
        "0.5-r5": 69.79059388946104,
        "0.5-r10": 75.66083075866804,
        "0.5-r100": 78.36136857764046,
        "0.7-r1": 29.2825266048747,
        "0.7-r5": 54.97196475569287,
        "0.7-r10": 64.14921615745509,
        "0.7-r100": 71.99908456345119
      },
      "movement": {
        "hard_positive_top100_query_exits": 0,
        "hard_positive_top100_query_entries": 0,
        "hard_positive_exit_ratio": 0.0,
        "video_slot_drift_rate": 0.40985697212096117,
        "invalid_span_count": 0,
        "duplicate_span_count_after_nms": 0
      },
      "rank_gain_loss_vs_C7_B1_frozen": {
        "0.5-r1": {
          "gain_queries": 412,
          "loss_queries": 0
        },
        "0.5-r5": {
          "gain_queries": 392,
          "loss_queries": 0
        },
        "0.5-r10": {
          "gain_queries": 297,
          "loss_queries": 0
        },
        "0.7-r1": {
          "gain_queries": 255,
          "loss_queries": 0
        },
        "0.7-r5": {
          "gain_queries": 314,
          "loss_queries": 0
        },
        "0.7-r10": {
          "gain_queries": 308,
          "loss_queries": 1
        }
      },
      "top100_lost_query_count": 0,
      "top100_lost_query_hash": "4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945",
      "config": {
        "lambda_evict": 0.05,
        "tail_coverage_weight": 0.1,
        "apply_only_beyond_rank": 20
      },
      "delta_vs_C6_B1": {
        "0.5-r1": 8.044398672617007,
        "0.5-r5": 7.3578212610138465,
        "0.5-r10": 6.053324178967841,
        "0.5-r100": 2.048289277949422,
        "0.7-r1": 2.6204371209520545,
        "0.7-r5": 6.121981920128164,
        "0.7-r10": 7.712552923675474,
        "0.7-r100": 6.533928367090056
      },
      "delta_vs_C6_B2": {
        "0.5-r1": 7.8040965785558996,
        "0.5-r5": 7.369264217873898,
        "0.5-r10": 6.04188122210779,
        "0.5-r100": 2.071175191669525,
        "0.7-r1": 2.2313765877102654,
        "0.7-r5": 6.076210092687951,
        "0.7-r10": 7.655338139375218,
        "0.7-r100": 6.476713582789785
      },
      "delta_vs_C7_B1_frozen": {
        "0.5-r1": 4.714498226341689,
        "0.5-r5": 4.4856390891406335,
        "0.5-r10": 3.39855818743564,
        "0.5-r100": 0.0,
        "0.7-r1": 2.917953999313422,
        "0.7-r5": 3.5930884540565273,
        "0.7-r10": 3.5129877560361606,
        "0.7-r100": 0.0
      },
      "status_label": "C7_B2_1_SAFE_BUT_R1_DROP"
    },
    {
      "name": "evict_lam0.05_tail0.2_beyond5",
      "metrics": {
        "0.5-r1": 47.62558645153908,
        "0.5-r5": 69.8478086737613,
        "0.5-r10": 75.67227371552809,
        "0.5-r100": 78.36136857764046,
        "0.7-r1": 29.2825266048747,
        "0.7-r5": 55.00629362627303,
        "0.7-r10": 64.12633024373498,
        "0.7-r100": 71.99908456345119
      },
      "movement": {
        "hard_positive_top100_query_exits": 0,
        "hard_positive_top100_query_entries": 0,
        "hard_positive_exit_ratio": 0.0,
        "video_slot_drift_rate": 0.40985697212096117,
        "invalid_span_count": 0,
        "duplicate_span_count_after_nms": 0
      },
      "rank_gain_loss_vs_C7_B1_frozen": {
        "0.5-r1": {
          "gain_queries": 412,
          "loss_queries": 0
        },
        "0.5-r5": {
          "gain_queries": 397,
          "loss_queries": 0
        },
        "0.5-r10": {
          "gain_queries": 298,
          "loss_queries": 0
        },
        "0.7-r1": {
          "gain_queries": 255,
          "loss_queries": 0
        },
        "0.7-r5": {
          "gain_queries": 321,
          "loss_queries": 4
        },
        "0.7-r10": {
          "gain_queries": 309,
          "loss_queries": 4
        }
      },
      "top100_lost_query_count": 0,
      "top100_lost_query_hash": "4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945",
      "config": {
        "lambda_evict": 0.05,
        "tail_coverage_weight": 0.2,
        "apply_only_beyond_rank": 5
      },
      "delta_vs_C6_B1": {
        "0.5-r1": 8.044398672617007,
        "0.5-r5": 7.4150360453141175,
        "0.5-r10": 6.064767135827893,
        "0.5-r100": 2.048289277949422,
        "0.7-r1": 2.6204371209520545,
        "0.7-r5": 6.156310790708318,
        "0.7-r10": 7.689667009955372,
        "0.7-r100": 6.533928367090056
      },
      "delta_vs_C6_B2": {
        "0.5-r1": 7.8040965785558996,
        "0.5-r5": 7.426479002174169,
        "0.5-r10": 6.053324178967841,
        "0.5-r100": 2.071175191669525,
        "0.7-r1": 2.2313765877102654,
        "0.7-r5": 6.110538963268105,
        "0.7-r10": 7.632452225655115,
        "0.7-r100": 6.476713582789785
      },
      "delta_vs_C7_B1_frozen": {
        "0.5-r1": 4.714498226341689,
        "0.5-r5": 4.5428538734409045,
        "0.5-r10": 3.4100011442956912,
        "0.5-r100": 0.0,
        "0.7-r1": 2.917953999313422,
        "0.7-r5": 3.6274173246366814,
        "0.7-r10": 3.490101842316058,
        "0.7-r100": 0.0
      },
      "status_label": "C7_B2_1_SAFE_BUT_R1_DROP"
    },
    {
      "name": "evict_lam0.05_tail0.2_beyond10",
      "metrics": {
        "0.5-r1": 47.62558645153908,
        "0.5-r5": 69.79059388946104,
        "0.5-r10": 75.61505893122784,
        "0.5-r100": 78.36136857764046,
        "0.7-r1": 29.2825266048747,
        "0.7-r5": 54.97196475569287,
        "0.7-r10": 64.10344433001488,
        "0.7-r100": 71.99908456345119
      },
      "movement": {
        "hard_positive_top100_query_exits": 0,
        "hard_positive_top100_query_entries": 0,
        "hard_positive_exit_ratio": 0.0,
        "video_slot_drift_rate": 0.40985697212096117,
        "invalid_span_count": 0,
        "duplicate_span_count_after_nms": 0
      },
      "rank_gain_loss_vs_C7_B1_frozen": {
        "0.5-r1": {
          "gain_queries": 412,
          "loss_queries": 0
        },
        "0.5-r5": {
          "gain_queries": 392,
          "loss_queries": 0
        },
        "0.5-r10": {
          "gain_queries": 294,
          "loss_queries": 1
        },
        "0.7-r1": {
          "gain_queries": 255,
          "loss_queries": 0
        },
        "0.7-r5": {
          "gain_queries": 314,
          "loss_queries": 0
        },
        "0.7-r10": {
          "gain_queries": 309,
          "loss_queries": 6
        }
      },
      "top100_lost_query_count": 0,
      "top100_lost_query_hash": "4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945",
      "config": {
        "lambda_evict": 0.05,
        "tail_coverage_weight": 0.2,
        "apply_only_beyond_rank": 10
      },
      "delta_vs_C6_B1": {
        "0.5-r1": 8.044398672617007,
        "0.5-r5": 7.3578212610138465,
        "0.5-r10": 6.007552351527636,
        "0.5-r100": 2.048289277949422,
        "0.7-r1": 2.6204371209520545,
        "0.7-r5": 6.121981920128164,
        "0.7-r10": 7.666781096235269,
        "0.7-r100": 6.533928367090056
      },
      "delta_vs_C6_B2": {
        "0.5-r1": 7.8040965785558996,
        "0.5-r5": 7.369264217873898,
        "0.5-r10": 5.9961093946675845,
        "0.5-r100": 2.071175191669525,
        "0.7-r1": 2.2313765877102654,
        "0.7-r5": 6.076210092687951,
        "0.7-r10": 7.609566311935012,
        "0.7-r100": 6.476713582789785
      },
      "delta_vs_C7_B1_frozen": {
        "0.5-r1": 4.714498226341689,
        "0.5-r5": 4.4856390891406335,
        "0.5-r10": 3.3527863599954344,
        "0.5-r100": 0.0,
        "0.7-r1": 2.917953999313422,
        "0.7-r5": 3.5930884540565273,
        "0.7-r10": 3.467215928595955,
        "0.7-r100": 0.0
      },
      "status_label": "C7_B2_1_SAFE_BUT_R1_DROP"
    },
    {
      "name": "evict_lam0.05_tail0.2_beyond20",
      "metrics": {
        "0.5-r1": 47.62558645153908,
        "0.5-r5": 69.79059388946104,
        "0.5-r10": 75.66083075866804,
        "0.5-r100": 78.36136857764046,
        "0.7-r1": 29.2825266048747,
        "0.7-r5": 54.97196475569287,
        "0.7-r10": 64.14921615745509,
        "0.7-r100": 71.99908456345119
      },
      "movement": {
        "hard_positive_top100_query_exits": 0,
        "hard_positive_top100_query_entries": 0,
        "hard_positive_exit_ratio": 0.0,
        "video_slot_drift_rate": 0.40985697212096117,
        "invalid_span_count": 0,
        "duplicate_span_count_after_nms": 0
      },
      "rank_gain_loss_vs_C7_B1_frozen": {
        "0.5-r1": {
          "gain_queries": 412,
          "loss_queries": 0
        },
        "0.5-r5": {
          "gain_queries": 392,
          "loss_queries": 0
        },
        "0.5-r10": {
          "gain_queries": 297,
          "loss_queries": 0
        },
        "0.7-r1": {
          "gain_queries": 255,
          "loss_queries": 0
        },
        "0.7-r5": {
          "gain_queries": 314,
          "loss_queries": 0
        },
        "0.7-r10": {
          "gain_queries": 308,
          "loss_queries": 1
        }
      },
      "top100_lost_query_count": 0,
      "top100_lost_query_hash": "4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945",
      "config": {
        "lambda_evict": 0.05,
        "tail_coverage_weight": 0.2,
        "apply_only_beyond_rank": 20
      },
      "delta_vs_C6_B1": {
        "0.5-r1": 8.044398672617007,
        "0.5-r5": 7.3578212610138465,
        "0.5-r10": 6.053324178967841,
        "0.5-r100": 2.048289277949422,
        "0.7-r1": 2.6204371209520545,
        "0.7-r5": 6.121981920128164,
        "0.7-r10": 7.712552923675474,
        "0.7-r100": 6.533928367090056
      },
      "delta_vs_C6_B2": {
        "0.5-r1": 7.8040965785558996,
        "0.5-r5": 7.369264217873898,
        "0.5-r10": 6.04188122210779,
        "0.5-r100": 2.071175191669525,
        "0.7-r1": 2.2313765877102654,
        "0.7-r5": 6.076210092687951,
        "0.7-r10": 7.655338139375218,
        "0.7-r100": 6.476713582789785
      },
      "delta_vs_C7_B1_frozen": {
        "0.5-r1": 4.714498226341689,
        "0.5-r5": 4.4856390891406335,
        "0.5-r10": 3.39855818743564,
        "0.5-r100": 0.0,
        "0.7-r1": 2.917953999313422,
        "0.7-r5": 3.5930884540565273,
        "0.7-r10": 3.5129877560361606,
        "0.7-r100": 0.0
      },
      "status_label": "C7_B2_1_SAFE_BUT_R1_DROP"
    },
    {
      "name": "evict_lam0.05_tail0.5_beyond5",
      "metrics": {
        "0.5-r1": 47.62558645153908,
        "0.5-r5": 69.8478086737613,
        "0.5-r10": 75.67227371552809,
        "0.5-r100": 78.36136857764046,
        "0.7-r1": 29.2825266048747,
        "0.7-r5": 55.00629362627303,
        "0.7-r10": 64.12633024373498,
        "0.7-r100": 71.99908456345119
      },
      "movement": {
        "hard_positive_top100_query_exits": 0,
        "hard_positive_top100_query_entries": 0,
        "hard_positive_exit_ratio": 0.0,
        "video_slot_drift_rate": 0.40985697212096117,
        "invalid_span_count": 0,
        "duplicate_span_count_after_nms": 0
      },
      "rank_gain_loss_vs_C7_B1_frozen": {
        "0.5-r1": {
          "gain_queries": 412,
          "loss_queries": 0
        },
        "0.5-r5": {
          "gain_queries": 397,
          "loss_queries": 0
        },
        "0.5-r10": {
          "gain_queries": 298,
          "loss_queries": 0
        },
        "0.7-r1": {
          "gain_queries": 255,
          "loss_queries": 0
        },
        "0.7-r5": {
          "gain_queries": 321,
          "loss_queries": 4
        },
        "0.7-r10": {
          "gain_queries": 309,
          "loss_queries": 4
        }
      },
      "top100_lost_query_count": 0,
      "top100_lost_query_hash": "4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945",
      "config": {
        "lambda_evict": 0.05,
        "tail_coverage_weight": 0.5,
        "apply_only_beyond_rank": 5
      },
      "delta_vs_C6_B1": {
        "0.5-r1": 8.044398672617007,
        "0.5-r5": 7.4150360453141175,
        "0.5-r10": 6.064767135827893,
        "0.5-r100": 2.048289277949422,
        "0.7-r1": 2.6204371209520545,
        "0.7-r5": 6.156310790708318,
        "0.7-r10": 7.689667009955372,
        "0.7-r100": 6.533928367090056
      },
      "delta_vs_C6_B2": {
        "0.5-r1": 7.8040965785558996,
        "0.5-r5": 7.426479002174169,
        "0.5-r10": 6.053324178967841,
        "0.5-r100": 2.071175191669525,
        "0.7-r1": 2.2313765877102654,
        "0.7-r5": 6.110538963268105,
        "0.7-r10": 7.632452225655115,
        "0.7-r100": 6.476713582789785
      },
      "delta_vs_C7_B1_frozen": {
        "0.5-r1": 4.714498226341689,
        "0.5-r5": 4.5428538734409045,
        "0.5-r10": 3.4100011442956912,
        "0.5-r100": 0.0,
        "0.7-r1": 2.917953999313422,
        "0.7-r5": 3.6274173246366814,
        "0.7-r10": 3.490101842316058,
        "0.7-r100": 0.0
      },
      "status_label": "C7_B2_1_SAFE_BUT_R1_DROP"
    },
    {
      "name": "evict_lam0.05_tail0.5_beyond10",
      "metrics": {
        "0.5-r1": 47.62558645153908,
        "0.5-r5": 69.79059388946104,
        "0.5-r10": 75.61505893122784,
        "0.5-r100": 78.36136857764046,
        "0.7-r1": 29.2825266048747,
        "0.7-r5": 54.97196475569287,
        "0.7-r10": 64.11488728687493,
        "0.7-r100": 71.99908456345119
      },
      "movement": {
        "hard_positive_top100_query_exits": 0,
        "hard_positive_top100_query_entries": 0,
        "hard_positive_exit_ratio": 0.0,
        "video_slot_drift_rate": 0.40985697212096117,
        "invalid_span_count": 0,
        "duplicate_span_count_after_nms": 0
      },
      "rank_gain_loss_vs_C7_B1_frozen": {
        "0.5-r1": {
          "gain_queries": 412,
          "loss_queries": 0
        },
        "0.5-r5": {
          "gain_queries": 392,
          "loss_queries": 0
        },
        "0.5-r10": {
          "gain_queries": 294,
          "loss_queries": 1
        },
        "0.7-r1": {
          "gain_queries": 255,
          "loss_queries": 0
        },
        "0.7-r5": {
          "gain_queries": 314,
          "loss_queries": 0
        },
        "0.7-r10": {
          "gain_queries": 309,
          "loss_queries": 5
        }
      },
      "top100_lost_query_count": 0,
      "top100_lost_query_hash": "4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945",
      "config": {
        "lambda_evict": 0.05,
        "tail_coverage_weight": 0.5,
        "apply_only_beyond_rank": 10
      },
      "delta_vs_C6_B1": {
        "0.5-r1": 8.044398672617007,
        "0.5-r5": 7.3578212610138465,
        "0.5-r10": 6.007552351527636,
        "0.5-r100": 2.048289277949422,
        "0.7-r1": 2.6204371209520545,
        "0.7-r5": 6.121981920128164,
        "0.7-r10": 7.67822405309532,
        "0.7-r100": 6.533928367090056
      },
      "delta_vs_C6_B2": {
        "0.5-r1": 7.8040965785558996,
        "0.5-r5": 7.369264217873898,
        "0.5-r10": 5.9961093946675845,
        "0.5-r100": 2.071175191669525,
        "0.7-r1": 2.2313765877102654,
        "0.7-r5": 6.076210092687951,
        "0.7-r10": 7.6210092687950635,
        "0.7-r100": 6.476713582789785
      },
      "delta_vs_C7_B1_frozen": {
        "0.5-r1": 4.714498226341689,
        "0.5-r5": 4.4856390891406335,
        "0.5-r10": 3.3527863599954344,
        "0.5-r100": 0.0,
        "0.7-r1": 2.917953999313422,
        "0.7-r5": 3.5930884540565273,
        "0.7-r10": 3.4786588854560065,
        "0.7-r100": 0.0
      },
      "status_label": "C7_B2_1_SAFE_BUT_R1_DROP"
    },
    {
      "name": "evict_lam0.05_tail0.5_beyond20",
      "metrics": {
        "0.5-r1": 47.62558645153908,
        "0.5-r5": 69.79059388946104,
        "0.5-r10": 75.66083075866804,
        "0.5-r100": 78.36136857764046,
        "0.7-r1": 29.2825266048747,
        "0.7-r5": 54.97196475569287,
        "0.7-r10": 64.14921615745509,
        "0.7-r100": 71.99908456345119
      },
      "movement": {
        "hard_positive_top100_query_exits": 0,
        "hard_positive_top100_query_entries": 0,
        "hard_positive_exit_ratio": 0.0,
        "video_slot_drift_rate": 0.40985697212096117,
        "invalid_span_count": 0,
        "duplicate_span_count_after_nms": 0
      },
      "rank_gain_loss_vs_C7_B1_frozen": {
        "0.5-r1": {
          "gain_queries": 412,
          "loss_queries": 0
        },
        "0.5-r5": {
          "gain_queries": 392,
          "loss_queries": 0
        },
        "0.5-r10": {
          "gain_queries": 297,
          "loss_queries": 0
        },
        "0.7-r1": {
          "gain_queries": 255,
          "loss_queries": 0
        },
        "0.7-r5": {
          "gain_queries": 314,
          "loss_queries": 0
        },
        "0.7-r10": {
          "gain_queries": 308,
          "loss_queries": 1
        }
      },
      "top100_lost_query_count": 0,
      "top100_lost_query_hash": "4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945",
      "config": {
        "lambda_evict": 0.05,
        "tail_coverage_weight": 0.5,
        "apply_only_beyond_rank": 20
      },
      "delta_vs_C6_B1": {
        "0.5-r1": 8.044398672617007,
        "0.5-r5": 7.3578212610138465,
        "0.5-r10": 6.053324178967841,
        "0.5-r100": 2.048289277949422,
        "0.7-r1": 2.6204371209520545,
        "0.7-r5": 6.121981920128164,
        "0.7-r10": 7.712552923675474,
        "0.7-r100": 6.533928367090056
      },
      "delta_vs_C6_B2": {
        "0.5-r1": 7.8040965785558996,
        "0.5-r5": 7.369264217873898,
        "0.5-r10": 6.04188122210779,
        "0.5-r100": 2.071175191669525,
        "0.7-r1": 2.2313765877102654,
        "0.7-r5": 6.076210092687951,
        "0.7-r10": 7.655338139375218,
        "0.7-r100": 6.476713582789785
      },
      "delta_vs_C7_B1_frozen": {
        "0.5-r1": 4.714498226341689,
        "0.5-r5": 4.4856390891406335,
        "0.5-r10": 3.39855818743564,
        "0.5-r100": 0.0,
        "0.7-r1": 2.917953999313422,
        "0.7-r5": 3.5930884540565273,
        "0.7-r10": 3.5129877560361606,
        "0.7-r100": 0.0
      },
      "status_label": "C7_B2_1_SAFE_BUT_R1_DROP"
    },
    {
      "name": "evict_lam0.1_tail0.1_beyond5",
      "metrics": {
        "0.5-r1": 47.62558645153908,
        "0.5-r5": 69.85925163062136,
        "0.5-r10": 75.68371667238814,
        "0.5-r100": 78.36136857764046,
        "0.7-r1": 29.2825266048747,
        "0.7-r5": 55.01773658313308,
        "0.7-r10": 64.09200137315483,
        "0.7-r100": 71.99908456345119
      },
      "movement": {
        "hard_positive_top100_query_exits": 0,
        "hard_positive_top100_query_entries": 0,
        "hard_positive_exit_ratio": 0.0,
        "video_slot_drift_rate": 0.40985697212096117,
        "invalid_span_count": 0,
        "duplicate_span_count_after_nms": 0
      },
      "rank_gain_loss_vs_C7_B1_frozen": {
        "0.5-r1": {
          "gain_queries": 412,
          "loss_queries": 0
        },
        "0.5-r5": {
          "gain_queries": 399,
          "loss_queries": 1
        },
        "0.5-r10": {
          "gain_queries": 299,
          "loss_queries": 0
        },
        "0.7-r1": {
          "gain_queries": 255,
          "loss_queries": 0
        },
        "0.7-r5": {
          "gain_queries": 325,
          "loss_queries": 7
        },
        "0.7-r10": {
          "gain_queries": 309,
          "loss_queries": 7
        }
      },
      "top100_lost_query_count": 0,
      "top100_lost_query_hash": "4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945",
      "config": {
        "lambda_evict": 0.1,
        "tail_coverage_weight": 0.1,
        "apply_only_beyond_rank": 5
      },
      "delta_vs_C6_B1": {
        "0.5-r1": 8.044398672617007,
        "0.5-r5": 7.426479002174169,
        "0.5-r10": 6.076210092687944,
        "0.5-r100": 2.048289277949422,
        "0.7-r1": 2.6204371209520545,
        "0.7-r5": 6.167753747568369,
        "0.7-r10": 7.655338139375218,
        "0.7-r100": 6.533928367090056
      },
      "delta_vs_C6_B2": {
        "0.5-r1": 7.8040965785558996,
        "0.5-r5": 7.43792195903422,
        "0.5-r10": 6.064767135827893,
        "0.5-r100": 2.071175191669525,
        "0.7-r1": 2.2313765877102654,
        "0.7-r5": 6.121981920128157,
        "0.7-r10": 7.598123355074961,
        "0.7-r100": 6.476713582789785
      },
      "delta_vs_C7_B1_frozen": {
        "0.5-r1": 4.714498226341689,
        "0.5-r5": 4.554296830300956,
        "0.5-r10": 3.4214441011557426,
        "0.5-r100": 0.0,
        "0.7-r1": 2.917953999313422,
        "0.7-r5": 3.638860281496733,
        "0.7-r10": 3.455772971735904,
        "0.7-r100": 0.0
      },
      "status_label": "C7_B2_1_SAFE_BUT_R1_DROP"
    },
    {
      "name": "evict_lam0.1_tail0.1_beyond10",
      "metrics": {
        "0.5-r1": 47.62558645153908,
        "0.5-r5": 69.79059388946104,
        "0.5-r10": 75.56928710378762,
        "0.5-r100": 78.36136857764046,
        "0.7-r1": 29.2825266048747,
        "0.7-r5": 54.97196475569287,
        "0.7-r10": 64.09200137315483,
        "0.7-r100": 71.99908456345119
      },
      "movement": {
        "hard_positive_top100_query_exits": 0,
        "hard_positive_top100_query_entries": 0,
        "hard_positive_exit_ratio": 0.0,
        "video_slot_drift_rate": 0.40985697212096117,
        "invalid_span_count": 0,
        "duplicate_span_count_after_nms": 0
      },
      "rank_gain_loss_vs_C7_B1_frozen": {
        "0.5-r1": {
          "gain_queries": 412,
          "loss_queries": 0
        },
        "0.5-r5": {
          "gain_queries": 392,
          "loss_queries": 0
        },
        "0.5-r10": {
          "gain_queries": 292,
          "loss_queries": 3
        },
        "0.7-r1": {
          "gain_queries": 255,
          "loss_queries": 0
        },
        "0.7-r5": {
          "gain_queries": 314,
          "loss_queries": 0
        },
        "0.7-r10": {
          "gain_queries": 311,
          "loss_queries": 9
        }
      },
      "top100_lost_query_count": 0,
      "top100_lost_query_hash": "4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945",
      "config": {
        "lambda_evict": 0.1,
        "tail_coverage_weight": 0.1,
        "apply_only_beyond_rank": 10
      },
      "delta_vs_C6_B1": {
        "0.5-r1": 8.044398672617007,
        "0.5-r5": 7.3578212610138465,
        "0.5-r10": 5.961780524087416,
        "0.5-r100": 2.048289277949422,
        "0.7-r1": 2.6204371209520545,
        "0.7-r5": 6.121981920128164,
        "0.7-r10": 7.655338139375218,
        "0.7-r100": 6.533928367090056
      },
      "delta_vs_C6_B2": {
        "0.5-r1": 7.8040965785558996,
        "0.5-r5": 7.369264217873898,
        "0.5-r10": 5.950337567227365,
        "0.5-r100": 2.071175191669525,
        "0.7-r1": 2.2313765877102654,
        "0.7-r5": 6.076210092687951,
        "0.7-r10": 7.598123355074961,
        "0.7-r100": 6.476713582789785
      },
      "delta_vs_C7_B1_frozen": {
        "0.5-r1": 4.714498226341689,
        "0.5-r5": 4.4856390891406335,
        "0.5-r10": 3.3070145325552147,
        "0.5-r100": 0.0,
        "0.7-r1": 2.917953999313422,
        "0.7-r5": 3.5930884540565273,
        "0.7-r10": 3.455772971735904,
        "0.7-r100": 0.0
      },
      "status_label": "C7_B2_1_SAFE_BUT_R1_DROP"
    },
    {
      "name": "evict_lam0.1_tail0.1_beyond20",
      "metrics": {
        "0.5-r1": 47.62558645153908,
        "0.5-r5": 69.79059388946104,
        "0.5-r10": 75.66083075866804,
        "0.5-r100": 78.36136857764046,
        "0.7-r1": 29.2825266048747,
        "0.7-r5": 54.97196475569287,
        "0.7-r10": 64.14921615745509,
        "0.7-r100": 71.99908456345119
      },
      "movement": {
        "hard_positive_top100_query_exits": 0,
        "hard_positive_top100_query_entries": 0,
        "hard_positive_exit_ratio": 0.0,
        "video_slot_drift_rate": 0.40985697212096117,
        "invalid_span_count": 0,
        "duplicate_span_count_after_nms": 0
      },
      "rank_gain_loss_vs_C7_B1_frozen": {
        "0.5-r1": {
          "gain_queries": 412,
          "loss_queries": 0
        },
        "0.5-r5": {
          "gain_queries": 392,
          "loss_queries": 0
        },
        "0.5-r10": {
          "gain_queries": 297,
          "loss_queries": 0
        },
        "0.7-r1": {
          "gain_queries": 255,
          "loss_queries": 0
        },
        "0.7-r5": {
          "gain_queries": 314,
          "loss_queries": 0
        },
        "0.7-r10": {
          "gain_queries": 308,
          "loss_queries": 1
        }
      },
      "top100_lost_query_count": 0,
      "top100_lost_query_hash": "4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945",
      "config": {
        "lambda_evict": 0.1,
        "tail_coverage_weight": 0.1,
        "apply_only_beyond_rank": 20
      },
      "delta_vs_C6_B1": {
        "0.5-r1": 8.044398672617007,
        "0.5-r5": 7.3578212610138465,
        "0.5-r10": 6.053324178967841,
        "0.5-r100": 2.048289277949422,
        "0.7-r1": 2.6204371209520545,
        "0.7-r5": 6.121981920128164,
        "0.7-r10": 7.712552923675474,
        "0.7-r100": 6.533928367090056
      },
      "delta_vs_C6_B2": {
        "0.5-r1": 7.8040965785558996,
        "0.5-r5": 7.369264217873898,
        "0.5-r10": 6.04188122210779,
        "0.5-r100": 2.071175191669525,
        "0.7-r1": 2.2313765877102654,
        "0.7-r5": 6.076210092687951,
        "0.7-r10": 7.655338139375218,
        "0.7-r100": 6.476713582789785
      },
      "delta_vs_C7_B1_frozen": {
        "0.5-r1": 4.714498226341689,
        "0.5-r5": 4.4856390891406335,
        "0.5-r10": 3.39855818743564,
        "0.5-r100": 0.0,
        "0.7-r1": 2.917953999313422,
        "0.7-r5": 3.5930884540565273,
        "0.7-r10": 3.5129877560361606,
        "0.7-r100": 0.0
      },
      "status_label": "C7_B2_1_SAFE_BUT_R1_DROP"
    },
    {
      "name": "evict_lam0.1_tail0.2_beyond5",
      "metrics": {
        "0.5-r1": 47.62558645153908,
        "0.5-r5": 69.85925163062136,
        "0.5-r10": 75.68371667238814,
        "0.5-r100": 78.36136857764046,
        "0.7-r1": 29.2825266048747,
        "0.7-r5": 55.01773658313308,
        "0.7-r10": 64.09200137315483,
        "0.7-r100": 71.99908456345119
      },
      "movement": {
        "hard_positive_top100_query_exits": 0,
        "hard_positive_top100_query_entries": 0,
        "hard_positive_exit_ratio": 0.0,
        "video_slot_drift_rate": 0.40985697212096117,
        "invalid_span_count": 0,
        "duplicate_span_count_after_nms": 0
      },
      "rank_gain_loss_vs_C7_B1_frozen": {
        "0.5-r1": {
          "gain_queries": 412,
          "loss_queries": 0
        },
        "0.5-r5": {
          "gain_queries": 399,
          "loss_queries": 1
        },
        "0.5-r10": {
          "gain_queries": 299,
          "loss_queries": 0
        },
        "0.7-r1": {
          "gain_queries": 255,
          "loss_queries": 0
        },
        "0.7-r5": {
          "gain_queries": 325,
          "loss_queries": 7
        },
        "0.7-r10": {
          "gain_queries": 309,
          "loss_queries": 7
        }
      },
      "top100_lost_query_count": 0,
      "top100_lost_query_hash": "4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945",
      "config": {
        "lambda_evict": 0.1,
        "tail_coverage_weight": 0.2,
        "apply_only_beyond_rank": 5
      },
      "delta_vs_C6_B1": {
        "0.5-r1": 8.044398672617007,
        "0.5-r5": 7.426479002174169,
        "0.5-r10": 6.076210092687944,
        "0.5-r100": 2.048289277949422,
        "0.7-r1": 2.6204371209520545,
        "0.7-r5": 6.167753747568369,
        "0.7-r10": 7.655338139375218,
        "0.7-r100": 6.533928367090056
      },
      "delta_vs_C6_B2": {
        "0.5-r1": 7.8040965785558996,
        "0.5-r5": 7.43792195903422,
        "0.5-r10": 6.064767135827893,
        "0.5-r100": 2.071175191669525,
        "0.7-r1": 2.2313765877102654,
        "0.7-r5": 6.121981920128157,
        "0.7-r10": 7.598123355074961,
        "0.7-r100": 6.476713582789785
      },
      "delta_vs_C7_B1_frozen": {
        "0.5-r1": 4.714498226341689,
        "0.5-r5": 4.554296830300956,
        "0.5-r10": 3.4214441011557426,
        "0.5-r100": 0.0,
        "0.7-r1": 2.917953999313422,
        "0.7-r5": 3.638860281496733,
        "0.7-r10": 3.455772971735904,
        "0.7-r100": 0.0
      },
      "status_label": "C7_B2_1_SAFE_BUT_R1_DROP"
    },
    {
      "name": "evict_lam0.1_tail0.2_beyond10",
      "metrics": {
        "0.5-r1": 47.62558645153908,
        "0.5-r5": 69.79059388946104,
        "0.5-r10": 75.56928710378762,
        "0.5-r100": 78.36136857764046,
        "0.7-r1": 29.2825266048747,
        "0.7-r5": 54.97196475569287,
        "0.7-r10": 64.09200137315483,
        "0.7-r100": 71.99908456345119
      },
      "movement": {
        "hard_positive_top100_query_exits": 0,
        "hard_positive_top100_query_entries": 0,
        "hard_positive_exit_ratio": 0.0,
        "video_slot_drift_rate": 0.40985697212096117,
        "invalid_span_count": 0,
        "duplicate_span_count_after_nms": 0
      },
      "rank_gain_loss_vs_C7_B1_frozen": {
        "0.5-r1": {
          "gain_queries": 412,
          "loss_queries": 0
        },
        "0.5-r5": {
          "gain_queries": 392,
          "loss_queries": 0
        },
        "0.5-r10": {
          "gain_queries": 292,
          "loss_queries": 3
        },
        "0.7-r1": {
          "gain_queries": 255,
          "loss_queries": 0
        },
        "0.7-r5": {
          "gain_queries": 314,
          "loss_queries": 0
        },
        "0.7-r10": {
          "gain_queries": 311,
          "loss_queries": 9
        }
      },
      "top100_lost_query_count": 0,
      "top100_lost_query_hash": "4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945",
      "config": {
        "lambda_evict": 0.1,
        "tail_coverage_weight": 0.2,
        "apply_only_beyond_rank": 10
      },
      "delta_vs_C6_B1": {
        "0.5-r1": 8.044398672617007,
        "0.5-r5": 7.3578212610138465,
        "0.5-r10": 5.961780524087416,
        "0.5-r100": 2.048289277949422,
        "0.7-r1": 2.6204371209520545,
        "0.7-r5": 6.121981920128164,
        "0.7-r10": 7.655338139375218,
        "0.7-r100": 6.533928367090056
      },
      "delta_vs_C6_B2": {
        "0.5-r1": 7.8040965785558996,
        "0.5-r5": 7.369264217873898,
        "0.5-r10": 5.950337567227365,
        "0.5-r100": 2.071175191669525,
        "0.7-r1": 2.2313765877102654,
        "0.7-r5": 6.076210092687951,
        "0.7-r10": 7.598123355074961,
        "0.7-r100": 6.476713582789785
      },
      "delta_vs_C7_B1_frozen": {
        "0.5-r1": 4.714498226341689,
        "0.5-r5": 4.4856390891406335,
        "0.5-r10": 3.3070145325552147,
        "0.5-r100": 0.0,
        "0.7-r1": 2.917953999313422,
        "0.7-r5": 3.5930884540565273,
        "0.7-r10": 3.455772971735904,
        "0.7-r100": 0.0
      },
      "status_label": "C7_B2_1_SAFE_BUT_R1_DROP"
    },
    {
      "name": "evict_lam0.1_tail0.2_beyond20",
      "metrics": {
        "0.5-r1": 47.62558645153908,
        "0.5-r5": 69.79059388946104,
        "0.5-r10": 75.66083075866804,
        "0.5-r100": 78.36136857764046,
        "0.7-r1": 29.2825266048747,
        "0.7-r5": 54.97196475569287,
        "0.7-r10": 64.14921615745509,
        "0.7-r100": 71.99908456345119
      },
      "movement": {
        "hard_positive_top100_query_exits": 0,
        "hard_positive_top100_query_entries": 0,
        "hard_positive_exit_ratio": 0.0,
        "video_slot_drift_rate": 0.40985697212096117,
        "invalid_span_count": 0,
        "duplicate_span_count_after_nms": 0
      },
      "rank_gain_loss_vs_C7_B1_frozen": {
        "0.5-r1": {
          "gain_queries": 412,
          "loss_queries": 0
        },
        "0.5-r5": {
          "gain_queries": 392,
          "loss_queries": 0
        },
        "0.5-r10": {
          "gain_queries": 297,
          "loss_queries": 0
        },
        "0.7-r1": {
          "gain_queries": 255,
          "loss_queries": 0
        },
        "0.7-r5": {
          "gain_queries": 314,
          "loss_queries": 0
        },
        "0.7-r10": {
          "gain_queries": 308,
          "loss_queries": 1
        }
      },
      "top100_lost_query_count": 0,
      "top100_lost_query_hash": "4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945",
      "config": {
        "lambda_evict": 0.1,
        "tail_coverage_weight": 0.2,
        "apply_only_beyond_rank": 20
      },
      "delta_vs_C6_B1": {
        "0.5-r1": 8.044398672617007,
        "0.5-r5": 7.3578212610138465,
        "0.5-r10": 6.053324178967841,
        "0.5-r100": 2.048289277949422,
        "0.7-r1": 2.6204371209520545,
        "0.7-r5": 6.121981920128164,
        "0.7-r10": 7.712552923675474,
        "0.7-r100": 6.533928367090056
      },
      "delta_vs_C6_B2": {
        "0.5-r1": 7.8040965785558996,
        "0.5-r5": 7.369264217873898,
        "0.5-r10": 6.04188122210779,
        "0.5-r100": 2.071175191669525,
        "0.7-r1": 2.2313765877102654,
        "0.7-r5": 6.076210092687951,
        "0.7-r10": 7.655338139375218,
        "0.7-r100": 6.476713582789785
      },
      "delta_vs_C7_B1_frozen": {
        "0.5-r1": 4.714498226341689,
        "0.5-r5": 4.4856390891406335,
        "0.5-r10": 3.39855818743564,
        "0.5-r100": 0.0,
        "0.7-r1": 2.917953999313422,
        "0.7-r5": 3.5930884540565273,
        "0.7-r10": 3.5129877560361606,
        "0.7-r100": 0.0
      },
      "status_label": "C7_B2_1_SAFE_BUT_R1_DROP"
    },
    {
      "name": "evict_lam0.1_tail0.5_beyond5",
      "metrics": {
        "0.5-r1": 47.62558645153908,
        "0.5-r5": 69.85925163062136,
        "0.5-r10": 75.68371667238814,
        "0.5-r100": 78.36136857764046,
        "0.7-r1": 29.2825266048747,
        "0.7-r5": 55.00629362627303,
        "0.7-r10": 64.09200137315483,
        "0.7-r100": 71.99908456345119
      },
      "movement": {
        "hard_positive_top100_query_exits": 0,
        "hard_positive_top100_query_entries": 0,
        "hard_positive_exit_ratio": 0.0,
        "video_slot_drift_rate": 0.40985697212096117,
        "invalid_span_count": 0,
        "duplicate_span_count_after_nms": 0
      },
      "rank_gain_loss_vs_C7_B1_frozen": {
        "0.5-r1": {
          "gain_queries": 412,
          "loss_queries": 0
        },
        "0.5-r5": {
          "gain_queries": 399,
          "loss_queries": 1
        },
        "0.5-r10": {
          "gain_queries": 299,
          "loss_queries": 0
        },
        "0.7-r1": {
          "gain_queries": 255,
          "loss_queries": 0
        },
        "0.7-r5": {
          "gain_queries": 324,
          "loss_queries": 7
        },
        "0.7-r10": {
          "gain_queries": 309,
          "loss_queries": 7
        }
      },
      "top100_lost_query_count": 0,
      "top100_lost_query_hash": "4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945",
      "config": {
        "lambda_evict": 0.1,
        "tail_coverage_weight": 0.5,
        "apply_only_beyond_rank": 5
      },
      "delta_vs_C6_B1": {
        "0.5-r1": 8.044398672617007,
        "0.5-r5": 7.426479002174169,
        "0.5-r10": 6.076210092687944,
        "0.5-r100": 2.048289277949422,
        "0.7-r1": 2.6204371209520545,
        "0.7-r5": 6.156310790708318,
        "0.7-r10": 7.655338139375218,
        "0.7-r100": 6.533928367090056
      },
      "delta_vs_C6_B2": {
        "0.5-r1": 7.8040965785558996,
        "0.5-r5": 7.43792195903422,
        "0.5-r10": 6.064767135827893,
        "0.5-r100": 2.071175191669525,
        "0.7-r1": 2.2313765877102654,
        "0.7-r5": 6.110538963268105,
        "0.7-r10": 7.598123355074961,
        "0.7-r100": 6.476713582789785
      },
      "delta_vs_C7_B1_frozen": {
        "0.5-r1": 4.714498226341689,
        "0.5-r5": 4.554296830300956,
        "0.5-r10": 3.4214441011557426,
        "0.5-r100": 0.0,
        "0.7-r1": 2.917953999313422,
        "0.7-r5": 3.6274173246366814,
        "0.7-r10": 3.455772971735904,
        "0.7-r100": 0.0
      },
      "status_label": "C7_B2_1_SAFE_BUT_R1_DROP"
    },
    {
      "name": "evict_lam0.1_tail0.5_beyond10",
      "metrics": {
        "0.5-r1": 47.62558645153908,
        "0.5-r5": 69.79059388946104,
        "0.5-r10": 75.58073006064767,
        "0.5-r100": 78.36136857764046,
        "0.7-r1": 29.2825266048747,
        "0.7-r5": 54.97196475569287,
        "0.7-r10": 64.10344433001488,
        "0.7-r100": 71.99908456345119
      },
      "movement": {
        "hard_positive_top100_query_exits": 0,
        "hard_positive_top100_query_entries": 0,
        "hard_positive_exit_ratio": 0.0,
        "video_slot_drift_rate": 0.40985697212096117,
        "invalid_span_count": 0,
        "duplicate_span_count_after_nms": 0
      },
      "rank_gain_loss_vs_C7_B1_frozen": {
        "0.5-r1": {
          "gain_queries": 412,
          "loss_queries": 0
        },
        "0.5-r5": {
          "gain_queries": 392,
          "loss_queries": 0
        },
        "0.5-r10": {
          "gain_queries": 293,
          "loss_queries": 3
        },
        "0.7-r1": {
          "gain_queries": 255,
          "loss_queries": 0
        },
        "0.7-r5": {
          "gain_queries": 314,
          "loss_queries": 0
        },
        "0.7-r10": {
          "gain_queries": 312,
          "loss_queries": 9
        }
      },
      "top100_lost_query_count": 0,
      "top100_lost_query_hash": "4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945",
      "config": {
        "lambda_evict": 0.1,
        "tail_coverage_weight": 0.5,
        "apply_only_beyond_rank": 10
      },
      "delta_vs_C6_B1": {
        "0.5-r1": 8.044398672617007,
        "0.5-r5": 7.3578212610138465,
        "0.5-r10": 5.973223480947468,
        "0.5-r100": 2.048289277949422,
        "0.7-r1": 2.6204371209520545,
        "0.7-r5": 6.121981920128164,
        "0.7-r10": 7.666781096235269,
        "0.7-r100": 6.533928367090056
      },
      "delta_vs_C6_B2": {
        "0.5-r1": 7.8040965785558996,
        "0.5-r5": 7.369264217873898,
        "0.5-r10": 5.961780524087416,
        "0.5-r100": 2.071175191669525,
        "0.7-r1": 2.2313765877102654,
        "0.7-r5": 6.076210092687951,
        "0.7-r10": 7.609566311935012,
        "0.7-r100": 6.476713582789785
      },
      "delta_vs_C7_B1_frozen": {
        "0.5-r1": 4.714498226341689,
        "0.5-r5": 4.4856390891406335,
        "0.5-r10": 3.318457489415266,
        "0.5-r100": 0.0,
        "0.7-r1": 2.917953999313422,
        "0.7-r5": 3.5930884540565273,
        "0.7-r10": 3.467215928595955,
        "0.7-r100": 0.0
      },
      "status_label": "C7_B2_1_SAFE_BUT_R1_DROP"
    },
    {
      "name": "evict_lam0.1_tail0.5_beyond20",
      "metrics": {
        "0.5-r1": 47.62558645153908,
        "0.5-r5": 69.79059388946104,
        "0.5-r10": 75.66083075866804,
        "0.5-r100": 78.36136857764046,
        "0.7-r1": 29.2825266048747,
        "0.7-r5": 54.97196475569287,
        "0.7-r10": 64.14921615745509,
        "0.7-r100": 71.99908456345119
      },
      "movement": {
        "hard_positive_top100_query_exits": 0,
        "hard_positive_top100_query_entries": 0,
        "hard_positive_exit_ratio": 0.0,
        "video_slot_drift_rate": 0.40985697212096117,
        "invalid_span_count": 0,
        "duplicate_span_count_after_nms": 0
      },
      "rank_gain_loss_vs_C7_B1_frozen": {
        "0.5-r1": {
          "gain_queries": 412,
          "loss_queries": 0
        },
        "0.5-r5": {
          "gain_queries": 392,
          "loss_queries": 0
        },
        "0.5-r10": {
          "gain_queries": 297,
          "loss_queries": 0
        },
        "0.7-r1": {
          "gain_queries": 255,
          "loss_queries": 0
        },
        "0.7-r5": {
          "gain_queries": 314,
          "loss_queries": 0
        },
        "0.7-r10": {
          "gain_queries": 308,
          "loss_queries": 1
        }
      },
      "top100_lost_query_count": 0,
      "top100_lost_query_hash": "4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945",
      "config": {
        "lambda_evict": 0.1,
        "tail_coverage_weight": 0.5,
        "apply_only_beyond_rank": 20
      },
      "delta_vs_C6_B1": {
        "0.5-r1": 8.044398672617007,
        "0.5-r5": 7.3578212610138465,
        "0.5-r10": 6.053324178967841,
        "0.5-r100": 2.048289277949422,
        "0.7-r1": 2.6204371209520545,
        "0.7-r5": 6.121981920128164,
        "0.7-r10": 7.712552923675474,
        "0.7-r100": 6.533928367090056
      },
      "delta_vs_C6_B2": {
        "0.5-r1": 7.8040965785558996,
        "0.5-r5": 7.369264217873898,
        "0.5-r10": 6.04188122210779,
        "0.5-r100": 2.071175191669525,
        "0.7-r1": 2.2313765877102654,
        "0.7-r5": 6.076210092687951,
        "0.7-r10": 7.655338139375218,
        "0.7-r100": 6.476713582789785
      },
      "delta_vs_C7_B1_frozen": {
        "0.5-r1": 4.714498226341689,
        "0.5-r5": 4.4856390891406335,
        "0.5-r10": 3.39855818743564,
        "0.5-r100": 0.0,
        "0.7-r1": 2.917953999313422,
        "0.7-r5": 3.5930884540565273,
        "0.7-r10": 3.5129877560361606,
        "0.7-r100": 0.0
      },
      "status_label": "C7_B2_1_SAFE_BUT_R1_DROP"
    },
    {
      "name": "evict_lam0.2_tail0.1_beyond5",
      "metrics": {
        "0.5-r1": 47.62558645153908,
        "0.5-r5": 69.83636571690124,
        "0.5-r10": 75.62650188808789,
        "0.5-r100": 78.36136857764046,
        "0.7-r1": 29.2825266048747,
        "0.7-r5": 55.02917953999314,
        "0.7-r10": 64.05767250257466,
        "0.7-r100": 71.99908456345119
      },
      "movement": {
        "hard_positive_top100_query_exits": 0,
        "hard_positive_top100_query_entries": 0,
        "hard_positive_exit_ratio": 0.0,
        "video_slot_drift_rate": 0.40985697212096117,
        "invalid_span_count": 0,
        "duplicate_span_count_after_nms": 0
      },
      "rank_gain_loss_vs_C7_B1_frozen": {
        "0.5-r1": {
          "gain_queries": 412,
          "loss_queries": 0
        },
        "0.5-r5": {
          "gain_queries": 399,
          "loss_queries": 3
        },
        "0.5-r10": {
          "gain_queries": 294,
          "loss_queries": 0
        },
        "0.7-r1": {
          "gain_queries": 255,
          "loss_queries": 0
        },
        "0.7-r5": {
          "gain_queries": 329,
          "loss_queries": 10
        },
        "0.7-r10": {
          "gain_queries": 309,
          "loss_queries": 10
        }
      },
      "top100_lost_query_count": 0,
      "top100_lost_query_hash": "4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945",
      "config": {
        "lambda_evict": 0.2,
        "tail_coverage_weight": 0.1,
        "apply_only_beyond_rank": 5
      },
      "delta_vs_C6_B1": {
        "0.5-r1": 8.044398672617007,
        "0.5-r5": 7.403593088454052,
        "0.5-r10": 6.018995308387687,
        "0.5-r100": 2.048289277949422,
        "0.7-r1": 2.6204371209520545,
        "0.7-r5": 6.179196704428428,
        "0.7-r10": 7.621009268795049,
        "0.7-r100": 6.533928367090056
      },
      "delta_vs_C6_B2": {
        "0.5-r1": 7.8040965785558996,
        "0.5-r5": 7.415036045314103,
        "0.5-r10": 6.007552351527636,
        "0.5-r100": 2.071175191669525,
        "0.7-r1": 2.2313765877102654,
        "0.7-r5": 6.133424876988215,
        "0.7-r10": 7.563794484494792,
        "0.7-r100": 6.476713582789785
      },
      "delta_vs_C7_B1_frozen": {
        "0.5-r1": 4.714498226341689,
        "0.5-r5": 4.531410916580839,
        "0.5-r10": 3.3642293168554858,
        "0.5-r100": 0.0,
        "0.7-r1": 2.917953999313422,
        "0.7-r5": 3.6503032383567913,
        "0.7-r10": 3.4214441011557355,
        "0.7-r100": 0.0
      },
      "status_label": "C7_B2_1_SAFE_BUT_R1_DROP"
    },
    {
      "name": "evict_lam0.2_tail0.1_beyond10",
      "metrics": {
        "0.5-r1": 47.62558645153908,
        "0.5-r5": 69.79059388946104,
        "0.5-r10": 75.63794484494794,
        "0.5-r100": 78.36136857764046,
        "0.7-r1": 29.2825266048747,
        "0.7-r5": 54.97196475569287,
        "0.7-r10": 64.10344433001488,
        "0.7-r100": 71.99908456345119
      },
      "movement": {
        "hard_positive_top100_query_exits": 0,
        "hard_positive_top100_query_entries": 0,
        "hard_positive_exit_ratio": 0.0,
        "video_slot_drift_rate": 0.40985697212096117,
        "invalid_span_count": 0,
        "duplicate_span_count_after_nms": 0
      },
      "rank_gain_loss_vs_C7_B1_frozen": {
        "0.5-r1": {
          "gain_queries": 412,
          "loss_queries": 0
        },
        "0.5-r5": {
          "gain_queries": 392,
          "loss_queries": 0
        },
        "0.5-r10": {
          "gain_queries": 299,
          "loss_queries": 4
        },
        "0.7-r1": {
          "gain_queries": 255,
          "loss_queries": 0
        },
        "0.7-r5": {
          "gain_queries": 314,
          "loss_queries": 0
        },
        "0.7-r10": {
          "gain_queries": 317,
          "loss_queries": 14
        }
      },
      "top100_lost_query_count": 0,
      "top100_lost_query_hash": "4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945",
      "config": {
        "lambda_evict": 0.2,
        "tail_coverage_weight": 0.1,
        "apply_only_beyond_rank": 10
      },
      "delta_vs_C6_B1": {
        "0.5-r1": 8.044398672617007,
        "0.5-r5": 7.3578212610138465,
        "0.5-r10": 6.030438265247739,
        "0.5-r100": 2.048289277949422,
        "0.7-r1": 2.6204371209520545,
        "0.7-r5": 6.121981920128164,
        "0.7-r10": 7.666781096235269,
        "0.7-r100": 6.533928367090056
      },
      "delta_vs_C6_B2": {
        "0.5-r1": 7.8040965785558996,
        "0.5-r5": 7.369264217873898,
        "0.5-r10": 6.018995308387687,
        "0.5-r100": 2.071175191669525,
        "0.7-r1": 2.2313765877102654,
        "0.7-r5": 6.076210092687951,
        "0.7-r10": 7.609566311935012,
        "0.7-r100": 6.476713582789785
      },
      "delta_vs_C7_B1_frozen": {
        "0.5-r1": 4.714498226341689,
        "0.5-r5": 4.4856390891406335,
        "0.5-r10": 3.375672273715537,
        "0.5-r100": 0.0,
        "0.7-r1": 2.917953999313422,
        "0.7-r5": 3.5930884540565273,
        "0.7-r10": 3.467215928595955,
        "0.7-r100": 0.0
      },
      "status_label": "C7_B2_1_SAFE_BUT_R1_DROP"
    },
    {
      "name": "evict_lam0.2_tail0.1_beyond20",
      "metrics": {
        "0.5-r1": 47.62558645153908,
        "0.5-r5": 69.79059388946104,
        "0.5-r10": 75.66083075866804,
        "0.5-r100": 78.36136857764046,
        "0.7-r1": 29.2825266048747,
        "0.7-r5": 54.97196475569287,
        "0.7-r10": 64.14921615745509,
        "0.7-r100": 71.99908456345119
      },
      "movement": {
        "hard_positive_top100_query_exits": 0,
        "hard_positive_top100_query_entries": 0,
        "hard_positive_exit_ratio": 0.0,
        "video_slot_drift_rate": 0.40985697212096117,
        "invalid_span_count": 0,
        "duplicate_span_count_after_nms": 0
      },
      "rank_gain_loss_vs_C7_B1_frozen": {
        "0.5-r1": {
          "gain_queries": 412,
          "loss_queries": 0
        },
        "0.5-r5": {
          "gain_queries": 392,
          "loss_queries": 0
        },
        "0.5-r10": {
          "gain_queries": 297,
          "loss_queries": 0
        },
        "0.7-r1": {
          "gain_queries": 255,
          "loss_queries": 0
        },
        "0.7-r5": {
          "gain_queries": 314,
          "loss_queries": 0
        },
        "0.7-r10": {
          "gain_queries": 308,
          "loss_queries": 1
        }
      },
      "top100_lost_query_count": 0,
      "top100_lost_query_hash": "4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945",
      "config": {
        "lambda_evict": 0.2,
        "tail_coverage_weight": 0.1,
        "apply_only_beyond_rank": 20
      },
      "delta_vs_C6_B1": {
        "0.5-r1": 8.044398672617007,
        "0.5-r5": 7.3578212610138465,
        "0.5-r10": 6.053324178967841,
        "0.5-r100": 2.048289277949422,
        "0.7-r1": 2.6204371209520545,
        "0.7-r5": 6.121981920128164,
        "0.7-r10": 7.712552923675474,
        "0.7-r100": 6.533928367090056
      },
      "delta_vs_C6_B2": {
        "0.5-r1": 7.8040965785558996,
        "0.5-r5": 7.369264217873898,
        "0.5-r10": 6.04188122210779,
        "0.5-r100": 2.071175191669525,
        "0.7-r1": 2.2313765877102654,
        "0.7-r5": 6.076210092687951,
        "0.7-r10": 7.655338139375218,
        "0.7-r100": 6.476713582789785
      },
      "delta_vs_C7_B1_frozen": {
        "0.5-r1": 4.714498226341689,
        "0.5-r5": 4.4856390891406335,
        "0.5-r10": 3.39855818743564,
        "0.5-r100": 0.0,
        "0.7-r1": 2.917953999313422,
        "0.7-r5": 3.5930884540565273,
        "0.7-r10": 3.5129877560361606,
        "0.7-r100": 0.0
      },
      "status_label": "C7_B2_1_SAFE_BUT_R1_DROP"
    },
    {
      "name": "evict_lam0.2_tail0.2_beyond5",
      "metrics": {
        "0.5-r1": 47.62558645153908,
        "0.5-r5": 69.83636571690124,
        "0.5-r10": 75.62650188808789,
        "0.5-r100": 78.36136857764046,
        "0.7-r1": 29.2825266048747,
        "0.7-r5": 55.02917953999314,
        "0.7-r10": 64.05767250257466,
        "0.7-r100": 71.99908456345119
      },
      "movement": {
        "hard_positive_top100_query_exits": 0,
        "hard_positive_top100_query_entries": 0,
        "hard_positive_exit_ratio": 0.0,
        "video_slot_drift_rate": 0.40985697212096117,
        "invalid_span_count": 0,
        "duplicate_span_count_after_nms": 0
      },
      "rank_gain_loss_vs_C7_B1_frozen": {
        "0.5-r1": {
          "gain_queries": 412,
          "loss_queries": 0
        },
        "0.5-r5": {
          "gain_queries": 399,
          "loss_queries": 3
        },
        "0.5-r10": {
          "gain_queries": 294,
          "loss_queries": 0
        },
        "0.7-r1": {
          "gain_queries": 255,
          "loss_queries": 0
        },
        "0.7-r5": {
          "gain_queries": 329,
          "loss_queries": 10
        },
        "0.7-r10": {
          "gain_queries": 309,
          "loss_queries": 10
        }
      },
      "top100_lost_query_count": 0,
      "top100_lost_query_hash": "4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945",
      "config": {
        "lambda_evict": 0.2,
        "tail_coverage_weight": 0.2,
        "apply_only_beyond_rank": 5
      },
      "delta_vs_C6_B1": {
        "0.5-r1": 8.044398672617007,
        "0.5-r5": 7.403593088454052,
        "0.5-r10": 6.018995308387687,
        "0.5-r100": 2.048289277949422,
        "0.7-r1": 2.6204371209520545,
        "0.7-r5": 6.179196704428428,
        "0.7-r10": 7.621009268795049,
        "0.7-r100": 6.533928367090056
      },
      "delta_vs_C6_B2": {
        "0.5-r1": 7.8040965785558996,
        "0.5-r5": 7.415036045314103,
        "0.5-r10": 6.007552351527636,
        "0.5-r100": 2.071175191669525,
        "0.7-r1": 2.2313765877102654,
        "0.7-r5": 6.133424876988215,
        "0.7-r10": 7.563794484494792,
        "0.7-r100": 6.476713582789785
      },
      "delta_vs_C7_B1_frozen": {
        "0.5-r1": 4.714498226341689,
        "0.5-r5": 4.531410916580839,
        "0.5-r10": 3.3642293168554858,
        "0.5-r100": 0.0,
        "0.7-r1": 2.917953999313422,
        "0.7-r5": 3.6503032383567913,
        "0.7-r10": 3.4214441011557355,
        "0.7-r100": 0.0
      },
      "status_label": "C7_B2_1_SAFE_BUT_R1_DROP"
    },
    {
      "name": "evict_lam0.2_tail0.2_beyond10",
      "metrics": {
        "0.5-r1": 47.62558645153908,
        "0.5-r5": 69.79059388946104,
        "0.5-r10": 75.63794484494794,
        "0.5-r100": 78.36136857764046,
        "0.7-r1": 29.2825266048747,
        "0.7-r5": 54.97196475569287,
        "0.7-r10": 64.09200137315483,
        "0.7-r100": 71.99908456345119
      },
      "movement": {
        "hard_positive_top100_query_exits": 0,
        "hard_positive_top100_query_entries": 0,
        "hard_positive_exit_ratio": 0.0,
        "video_slot_drift_rate": 0.40985697212096117,
        "invalid_span_count": 0,
        "duplicate_span_count_after_nms": 0
      },
      "rank_gain_loss_vs_C7_B1_frozen": {
        "0.5-r1": {
          "gain_queries": 412,
          "loss_queries": 0
        },
        "0.5-r5": {
          "gain_queries": 392,
          "loss_queries": 0
        },
        "0.5-r10": {
          "gain_queries": 299,
          "loss_queries": 4
        },
        "0.7-r1": {
          "gain_queries": 255,
          "loss_queries": 0
        },
        "0.7-r5": {
          "gain_queries": 314,
          "loss_queries": 0
        },
        "0.7-r10": {
          "gain_queries": 316,
          "loss_queries": 14
        }
      },
      "top100_lost_query_count": 0,
      "top100_lost_query_hash": "4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945",
      "config": {
        "lambda_evict": 0.2,
        "tail_coverage_weight": 0.2,
        "apply_only_beyond_rank": 10
      },
      "delta_vs_C6_B1": {
        "0.5-r1": 8.044398672617007,
        "0.5-r5": 7.3578212610138465,
        "0.5-r10": 6.030438265247739,
        "0.5-r100": 2.048289277949422,
        "0.7-r1": 2.6204371209520545,
        "0.7-r5": 6.121981920128164,
        "0.7-r10": 7.655338139375218,
        "0.7-r100": 6.533928367090056
      },
      "delta_vs_C6_B2": {
        "0.5-r1": 7.8040965785558996,
        "0.5-r5": 7.369264217873898,
        "0.5-r10": 6.018995308387687,
        "0.5-r100": 2.071175191669525,
        "0.7-r1": 2.2313765877102654,
        "0.7-r5": 6.076210092687951,
        "0.7-r10": 7.598123355074961,
        "0.7-r100": 6.476713582789785
      },
      "delta_vs_C7_B1_frozen": {
        "0.5-r1": 4.714498226341689,
        "0.5-r5": 4.4856390891406335,
        "0.5-r10": 3.375672273715537,
        "0.5-r100": 0.0,
        "0.7-r1": 2.917953999313422,
        "0.7-r5": 3.5930884540565273,
        "0.7-r10": 3.455772971735904,
        "0.7-r100": 0.0
      },
      "status_label": "C7_B2_1_SAFE_BUT_R1_DROP"
    },
    {
      "name": "evict_lam0.2_tail0.2_beyond20",
      "metrics": {
        "0.5-r1": 47.62558645153908,
        "0.5-r5": 69.79059388946104,
        "0.5-r10": 75.66083075866804,
        "0.5-r100": 78.36136857764046,
        "0.7-r1": 29.2825266048747,
        "0.7-r5": 54.97196475569287,
        "0.7-r10": 64.14921615745509,
        "0.7-r100": 71.99908456345119
      },
      "movement": {
        "hard_positive_top100_query_exits": 0,
        "hard_positive_top100_query_entries": 0,
        "hard_positive_exit_ratio": 0.0,
        "video_slot_drift_rate": 0.40985697212096117,
        "invalid_span_count": 0,
        "duplicate_span_count_after_nms": 0
      },
      "rank_gain_loss_vs_C7_B1_frozen": {
        "0.5-r1": {
          "gain_queries": 412,
          "loss_queries": 0
        },
        "0.5-r5": {
          "gain_queries": 392,
          "loss_queries": 0
        },
        "0.5-r10": {
          "gain_queries": 297,
          "loss_queries": 0
        },
        "0.7-r1": {
          "gain_queries": 255,
          "loss_queries": 0
        },
        "0.7-r5": {
          "gain_queries": 314,
          "loss_queries": 0
        },
        "0.7-r10": {
          "gain_queries": 308,
          "loss_queries": 1
        }
      },
      "top100_lost_query_count": 0,
      "top100_lost_query_hash": "4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945",
      "config": {
        "lambda_evict": 0.2,
        "tail_coverage_weight": 0.2,
        "apply_only_beyond_rank": 20
      },
      "delta_vs_C6_B1": {
        "0.5-r1": 8.044398672617007,
        "0.5-r5": 7.3578212610138465,
        "0.5-r10": 6.053324178967841,
        "0.5-r100": 2.048289277949422,
        "0.7-r1": 2.6204371209520545,
        "0.7-r5": 6.121981920128164,
        "0.7-r10": 7.712552923675474,
        "0.7-r100": 6.533928367090056
      },
      "delta_vs_C6_B2": {
        "0.5-r1": 7.8040965785558996,
        "0.5-r5": 7.369264217873898,
        "0.5-r10": 6.04188122210779,
        "0.5-r100": 2.071175191669525,
        "0.7-r1": 2.2313765877102654,
        "0.7-r5": 6.076210092687951,
        "0.7-r10": 7.655338139375218,
        "0.7-r100": 6.476713582789785
      },
      "delta_vs_C7_B1_frozen": {
        "0.5-r1": 4.714498226341689,
        "0.5-r5": 4.4856390891406335,
        "0.5-r10": 3.39855818743564,
        "0.5-r100": 0.0,
        "0.7-r1": 2.917953999313422,
        "0.7-r5": 3.5930884540565273,
        "0.7-r10": 3.5129877560361606,
        "0.7-r100": 0.0
      },
      "status_label": "C7_B2_1_SAFE_BUT_R1_DROP"
    },
    {
      "name": "evict_lam0.2_tail0.5_beyond5",
      "metrics": {
        "0.5-r1": 47.62558645153908,
        "0.5-r5": 69.82492276004119,
        "0.5-r10": 75.62650188808789,
        "0.5-r100": 78.36136857764046,
        "0.7-r1": 29.2825266048747,
        "0.7-r5": 55.04062249685319,
        "0.7-r10": 64.05767250257466,
        "0.7-r100": 71.99908456345119
      },
      "movement": {
        "hard_positive_top100_query_exits": 0,
        "hard_positive_top100_query_entries": 0,
        "hard_positive_exit_ratio": 0.0,
        "video_slot_drift_rate": 0.40985697212096117,
        "invalid_span_count": 0,
        "duplicate_span_count_after_nms": 0
      },
      "rank_gain_loss_vs_C7_B1_frozen": {
        "0.5-r1": {
          "gain_queries": 412,
          "loss_queries": 0
        },
        "0.5-r5": {
          "gain_queries": 398,
          "loss_queries": 3
        },
        "0.5-r10": {
          "gain_queries": 294,
          "loss_queries": 0
        },
        "0.7-r1": {
          "gain_queries": 255,
          "loss_queries": 0
        },
        "0.7-r5": {
          "gain_queries": 330,
          "loss_queries": 10
        },
        "0.7-r10": {
          "gain_queries": 309,
          "loss_queries": 10
        }
      },
      "top100_lost_query_count": 0,
      "top100_lost_query_hash": "4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945",
      "config": {
        "lambda_evict": 0.2,
        "tail_coverage_weight": 0.5,
        "apply_only_beyond_rank": 5
      },
      "delta_vs_C6_B1": {
        "0.5-r1": 8.044398672617007,
        "0.5-r5": 7.392150131594001,
        "0.5-r10": 6.018995308387687,
        "0.5-r100": 2.048289277949422,
        "0.7-r1": 2.6204371209520545,
        "0.7-r5": 6.190639661288479,
        "0.7-r10": 7.621009268795049,
        "0.7-r100": 6.533928367090056
      },
      "delta_vs_C6_B2": {
        "0.5-r1": 7.8040965785558996,
        "0.5-r5": 7.403593088454052,
        "0.5-r10": 6.007552351527636,
        "0.5-r100": 2.071175191669525,
        "0.7-r1": 2.2313765877102654,
        "0.7-r5": 6.1448678338482665,
        "0.7-r10": 7.563794484494792,
        "0.7-r100": 6.476713582789785
      },
      "delta_vs_C7_B1_frozen": {
        "0.5-r1": 4.714498226341689,
        "0.5-r5": 4.519967959720788,
        "0.5-r10": 3.3642293168554858,
        "0.5-r100": 0.0,
        "0.7-r1": 2.917953999313422,
        "0.7-r5": 3.6617461952168426,
        "0.7-r10": 3.4214441011557355,
        "0.7-r100": 0.0
      },
      "status_label": "C7_B2_1_SAFE_BUT_R1_DROP"
    },
    {
      "name": "evict_lam0.2_tail0.5_beyond10",
      "metrics": {
        "0.5-r1": 47.62558645153908,
        "0.5-r5": 69.79059388946104,
        "0.5-r10": 75.66083075866804,
        "0.5-r100": 78.36136857764046,
        "0.7-r1": 29.2825266048747,
        "0.7-r5": 54.97196475569287,
        "0.7-r10": 64.10344433001488,
        "0.7-r100": 71.99908456345119
      },
      "movement": {
        "hard_positive_top100_query_exits": 0,
        "hard_positive_top100_query_entries": 0,
        "hard_positive_exit_ratio": 0.0,
        "video_slot_drift_rate": 0.40985697212096117,
        "invalid_span_count": 0,
        "duplicate_span_count_after_nms": 0
      },
      "rank_gain_loss_vs_C7_B1_frozen": {
        "0.5-r1": {
          "gain_queries": 412,
          "loss_queries": 0
        },
        "0.5-r5": {
          "gain_queries": 392,
          "loss_queries": 0
        },
        "0.5-r10": {
          "gain_queries": 300,
          "loss_queries": 3
        },
        "0.7-r1": {
          "gain_queries": 255,
          "loss_queries": 0
        },
        "0.7-r5": {
          "gain_queries": 314,
          "loss_queries": 0
        },
        "0.7-r10": {
          "gain_queries": 317,
          "loss_queries": 14
        }
      },
      "top100_lost_query_count": 0,
      "top100_lost_query_hash": "4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945",
      "config": {
        "lambda_evict": 0.2,
        "tail_coverage_weight": 0.5,
        "apply_only_beyond_rank": 10
      },
      "delta_vs_C6_B1": {
        "0.5-r1": 8.044398672617007,
        "0.5-r5": 7.3578212610138465,
        "0.5-r10": 6.053324178967841,
        "0.5-r100": 2.048289277949422,
        "0.7-r1": 2.6204371209520545,
        "0.7-r5": 6.121981920128164,
        "0.7-r10": 7.666781096235269,
        "0.7-r100": 6.533928367090056
      },
      "delta_vs_C6_B2": {
        "0.5-r1": 7.8040965785558996,
        "0.5-r5": 7.369264217873898,
        "0.5-r10": 6.04188122210779,
        "0.5-r100": 2.071175191669525,
        "0.7-r1": 2.2313765877102654,
        "0.7-r5": 6.076210092687951,
        "0.7-r10": 7.609566311935012,
        "0.7-r100": 6.476713582789785
      },
      "delta_vs_C7_B1_frozen": {
        "0.5-r1": 4.714498226341689,
        "0.5-r5": 4.4856390891406335,
        "0.5-r10": 3.39855818743564,
        "0.5-r100": 0.0,
        "0.7-r1": 2.917953999313422,
        "0.7-r5": 3.5930884540565273,
        "0.7-r10": 3.467215928595955,
        "0.7-r100": 0.0
      },
      "status_label": "C7_B2_1_SAFE_BUT_R1_DROP"
    },
    {
      "name": "evict_lam0.2_tail0.5_beyond20",
      "metrics": {
        "0.5-r1": 47.62558645153908,
        "0.5-r5": 69.79059388946104,
        "0.5-r10": 75.66083075866804,
        "0.5-r100": 78.36136857764046,
        "0.7-r1": 29.2825266048747,
        "0.7-r5": 54.97196475569287,
        "0.7-r10": 64.14921615745509,
        "0.7-r100": 71.99908456345119
      },
      "movement": {
        "hard_positive_top100_query_exits": 0,
        "hard_positive_top100_query_entries": 0,
        "hard_positive_exit_ratio": 0.0,
        "video_slot_drift_rate": 0.40985697212096117,
        "invalid_span_count": 0,
        "duplicate_span_count_after_nms": 0
      },
      "rank_gain_loss_vs_C7_B1_frozen": {
        "0.5-r1": {
          "gain_queries": 412,
          "loss_queries": 0
        },
        "0.5-r5": {
          "gain_queries": 392,
          "loss_queries": 0
        },
        "0.5-r10": {
          "gain_queries": 297,
          "loss_queries": 0
        },
        "0.7-r1": {
          "gain_queries": 255,
          "loss_queries": 0
        },
        "0.7-r5": {
          "gain_queries": 314,
          "loss_queries": 0
        },
        "0.7-r10": {
          "gain_queries": 308,
          "loss_queries": 1
        }
      },
      "top100_lost_query_count": 0,
      "top100_lost_query_hash": "4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945",
      "config": {
        "lambda_evict": 0.2,
        "tail_coverage_weight": 0.5,
        "apply_only_beyond_rank": 20
      },
      "delta_vs_C6_B1": {
        "0.5-r1": 8.044398672617007,
        "0.5-r5": 7.3578212610138465,
        "0.5-r10": 6.053324178967841,
        "0.5-r100": 2.048289277949422,
        "0.7-r1": 2.6204371209520545,
        "0.7-r5": 6.121981920128164,
        "0.7-r10": 7.712552923675474,
        "0.7-r100": 6.533928367090056
      },
      "delta_vs_C6_B2": {
        "0.5-r1": 7.8040965785558996,
        "0.5-r5": 7.369264217873898,
        "0.5-r10": 6.04188122210779,
        "0.5-r100": 2.071175191669525,
        "0.7-r1": 2.2313765877102654,
        "0.7-r5": 6.076210092687951,
        "0.7-r10": 7.655338139375218,
        "0.7-r100": 6.476713582789785
      },
      "delta_vs_C7_B1_frozen": {
        "0.5-r1": 4.714498226341689,
        "0.5-r5": 4.4856390891406335,
        "0.5-r10": 3.39855818743564,
        "0.5-r100": 0.0,
        "0.7-r1": 2.917953999313422,
        "0.7-r5": 3.5930884540565273,
        "0.7-r10": 3.5129877560361606,
        "0.7-r100": 0.0
      },
      "status_label": "C7_B2_1_SAFE_BUT_R1_DROP"
    },
    {
      "name": "evict_lam0.5_tail0.1_beyond5",
      "metrics": {
        "0.5-r1": 47.62558645153908,
        "0.5-r5": 69.76770797574093,
        "0.5-r10": 75.63794484494794,
        "0.5-r100": 78.36136857764046,
        "0.7-r1": 29.2825266048747,
        "0.7-r5": 55.00629362627303,
        "0.7-r10": 64.0004577182744,
        "0.7-r100": 71.99908456345119
      },
      "movement": {
        "hard_positive_top100_query_exits": 0,
        "hard_positive_top100_query_entries": 0,
        "hard_positive_exit_ratio": 0.0,
        "video_slot_drift_rate": 0.40985697212096117,
        "invalid_span_count": 0,
        "duplicate_span_count_after_nms": 0
      },
      "rank_gain_loss_vs_C7_B1_frozen": {
        "0.5-r1": {
          "gain_queries": 412,
          "loss_queries": 0
        },
        "0.5-r5": {
          "gain_queries": 399,
          "loss_queries": 9
        },
        "0.5-r10": {
          "gain_queries": 296,
          "loss_queries": 1
        },
        "0.7-r1": {
          "gain_queries": 255,
          "loss_queries": 0
        },
        "0.7-r5": {
          "gain_queries": 339,
          "loss_queries": 22
        },
        "0.7-r10": {
          "gain_queries": 310,
          "loss_queries": 16
        }
      },
      "top100_lost_query_count": 0,
      "top100_lost_query_hash": "4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945",
      "config": {
        "lambda_evict": 0.5,
        "tail_coverage_weight": 0.1,
        "apply_only_beyond_rank": 5
      },
      "delta_vs_C6_B1": {
        "0.5-r1": 8.044398672617007,
        "0.5-r5": 7.334935347293744,
        "0.5-r10": 6.030438265247739,
        "0.5-r100": 2.048289277949422,
        "0.7-r1": 2.6204371209520545,
        "0.7-r5": 6.156310790708318,
        "0.7-r10": 7.563794484494792,
        "0.7-r100": 6.533928367090056
      },
      "delta_vs_C6_B2": {
        "0.5-r1": 7.8040965785558996,
        "0.5-r5": 7.346378304153795,
        "0.5-r10": 6.018995308387687,
        "0.5-r100": 2.071175191669525,
        "0.7-r1": 2.2313765877102654,
        "0.7-r5": 6.110538963268105,
        "0.7-r10": 7.506579700194536,
        "0.7-r100": 6.476713582789785
      },
      "delta_vs_C7_B1_frozen": {
        "0.5-r1": 4.714498226341689,
        "0.5-r5": 4.462753175420531,
        "0.5-r10": 3.375672273715537,
        "0.5-r100": 0.0,
        "0.7-r1": 2.917953999313422,
        "0.7-r5": 3.6274173246366814,
        "0.7-r10": 3.3642293168554787,
        "0.7-r100": 0.0
      },
      "status_label": "C7_B2_1_SAFE_BUT_R1_DROP"
    },
    {
      "name": "evict_lam0.5_tail0.1_beyond10",
      "metrics": {
        "0.5-r1": 47.62558645153908,
        "0.5-r5": 69.79059388946104,
        "0.5-r10": 75.56928710378762,
        "0.5-r100": 78.36136857764046,
        "0.7-r1": 29.2825266048747,
        "0.7-r5": 54.97196475569287,
        "0.7-r10": 64.01190067513446,
        "0.7-r100": 71.99908456345119
      },
      "movement": {
        "hard_positive_top100_query_exits": 0,
        "hard_positive_top100_query_entries": 0,
        "hard_positive_exit_ratio": 0.0,
        "video_slot_drift_rate": 0.40985697212096117,
        "invalid_span_count": 0,
        "duplicate_span_count_after_nms": 0
      },
      "rank_gain_loss_vs_C7_B1_frozen": {
        "0.5-r1": {
          "gain_queries": 412,
          "loss_queries": 0
        },
        "0.5-r5": {
          "gain_queries": 392,
          "loss_queries": 0
        },
        "0.5-r10": {
          "gain_queries": 298,
          "loss_queries": 9
        },
        "0.7-r1": {
          "gain_queries": 255,
          "loss_queries": 0
        },
        "0.7-r5": {
          "gain_queries": 314,
          "loss_queries": 0
        },
        "0.7-r10": {
          "gain_queries": 319,
          "loss_queries": 24
        }
      },
      "top100_lost_query_count": 0,
      "top100_lost_query_hash": "4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945",
      "config": {
        "lambda_evict": 0.5,
        "tail_coverage_weight": 0.1,
        "apply_only_beyond_rank": 10
      },
      "delta_vs_C6_B1": {
        "0.5-r1": 8.044398672617007,
        "0.5-r5": 7.3578212610138465,
        "0.5-r10": 5.961780524087416,
        "0.5-r100": 2.048289277949422,
        "0.7-r1": 2.6204371209520545,
        "0.7-r5": 6.121981920128164,
        "0.7-r10": 7.575237441354844,
        "0.7-r100": 6.533928367090056
      },
      "delta_vs_C6_B2": {
        "0.5-r1": 7.8040965785558996,
        "0.5-r5": 7.369264217873898,
        "0.5-r10": 5.950337567227365,
        "0.5-r100": 2.071175191669525,
        "0.7-r1": 2.2313765877102654,
        "0.7-r5": 6.076210092687951,
        "0.7-r10": 7.518022657054587,
        "0.7-r100": 6.476713582789785
      },
      "delta_vs_C7_B1_frozen": {
        "0.5-r1": 4.714498226341689,
        "0.5-r5": 4.4856390891406335,
        "0.5-r10": 3.3070145325552147,
        "0.5-r100": 0.0,
        "0.7-r1": 2.917953999313422,
        "0.7-r5": 3.5930884540565273,
        "0.7-r10": 3.37567227371553,
        "0.7-r100": 0.0
      },
      "status_label": "C7_B2_1_SAFE_BUT_R1_DROP"
    },
    {
      "name": "evict_lam0.5_tail0.1_beyond20",
      "metrics": {
        "0.5-r1": 47.62558645153908,
        "0.5-r5": 69.79059388946104,
        "0.5-r10": 75.66083075866804,
        "0.5-r100": 78.36136857764046,
        "0.7-r1": 29.2825266048747,
        "0.7-r5": 54.97196475569287,
        "0.7-r10": 64.14921615745509,
        "0.7-r100": 71.99908456345119
      },
      "movement": {
        "hard_positive_top100_query_exits": 0,
        "hard_positive_top100_query_entries": 0,
        "hard_positive_exit_ratio": 0.0,
        "video_slot_drift_rate": 0.40985697212096117,
        "invalid_span_count": 0,
        "duplicate_span_count_after_nms": 0
      },
      "rank_gain_loss_vs_C7_B1_frozen": {
        "0.5-r1": {
          "gain_queries": 412,
          "loss_queries": 0
        },
        "0.5-r5": {
          "gain_queries": 392,
          "loss_queries": 0
        },
        "0.5-r10": {
          "gain_queries": 297,
          "loss_queries": 0
        },
        "0.7-r1": {
          "gain_queries": 255,
          "loss_queries": 0
        },
        "0.7-r5": {
          "gain_queries": 314,
          "loss_queries": 0
        },
        "0.7-r10": {
          "gain_queries": 308,
          "loss_queries": 1
        }
      },
      "top100_lost_query_count": 0,
      "top100_lost_query_hash": "4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945",
      "config": {
        "lambda_evict": 0.5,
        "tail_coverage_weight": 0.1,
        "apply_only_beyond_rank": 20
      },
      "delta_vs_C6_B1": {
        "0.5-r1": 8.044398672617007,
        "0.5-r5": 7.3578212610138465,
        "0.5-r10": 6.053324178967841,
        "0.5-r100": 2.048289277949422,
        "0.7-r1": 2.6204371209520545,
        "0.7-r5": 6.121981920128164,
        "0.7-r10": 7.712552923675474,
        "0.7-r100": 6.533928367090056
      },
      "delta_vs_C6_B2": {
        "0.5-r1": 7.8040965785558996,
        "0.5-r5": 7.369264217873898,
        "0.5-r10": 6.04188122210779,
        "0.5-r100": 2.071175191669525,
        "0.7-r1": 2.2313765877102654,
        "0.7-r5": 6.076210092687951,
        "0.7-r10": 7.655338139375218,
        "0.7-r100": 6.476713582789785
      },
      "delta_vs_C7_B1_frozen": {
        "0.5-r1": 4.714498226341689,
        "0.5-r5": 4.4856390891406335,
        "0.5-r10": 3.39855818743564,
        "0.5-r100": 0.0,
        "0.7-r1": 2.917953999313422,
        "0.7-r5": 3.5930884540565273,
        "0.7-r10": 3.5129877560361606,
        "0.7-r100": 0.0
      },
      "status_label": "C7_B2_1_SAFE_BUT_R1_DROP"
    },
    {
      "name": "evict_lam0.5_tail0.2_beyond5",
      "metrics": {
        "0.5-r1": 47.62558645153908,
        "0.5-r5": 69.75626501888088,
        "0.5-r10": 75.63794484494794,
        "0.5-r100": 78.36136857764046,
        "0.7-r1": 29.2825266048747,
        "0.7-r5": 55.02917953999314,
        "0.7-r10": 64.0004577182744,
        "0.7-r100": 71.99908456345119
      },
      "movement": {
        "hard_positive_top100_query_exits": 0,
        "hard_positive_top100_query_entries": 0,
        "hard_positive_exit_ratio": 0.0,
        "video_slot_drift_rate": 0.40985697212096117,
        "invalid_span_count": 0,
        "duplicate_span_count_after_nms": 0
      },
      "rank_gain_loss_vs_C7_B1_frozen": {
        "0.5-r1": {
          "gain_queries": 412,
          "loss_queries": 0
        },
        "0.5-r5": {
          "gain_queries": 398,
          "loss_queries": 9
        },
        "0.5-r10": {
          "gain_queries": 296,
          "loss_queries": 1
        },
        "0.7-r1": {
          "gain_queries": 255,
          "loss_queries": 0
        },
        "0.7-r5": {
          "gain_queries": 340,
          "loss_queries": 21
        },
        "0.7-r10": {
          "gain_queries": 310,
          "loss_queries": 16
        }
      },
      "top100_lost_query_count": 0,
      "top100_lost_query_hash": "4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945",
      "config": {
        "lambda_evict": 0.5,
        "tail_coverage_weight": 0.2,
        "apply_only_beyond_rank": 5
      },
      "delta_vs_C6_B1": {
        "0.5-r1": 8.044398672617007,
        "0.5-r5": 7.323492390433692,
        "0.5-r10": 6.030438265247739,
        "0.5-r100": 2.048289277949422,
        "0.7-r1": 2.6204371209520545,
        "0.7-r5": 6.179196704428428,
        "0.7-r10": 7.563794484494792,
        "0.7-r100": 6.533928367090056
      },
      "delta_vs_C6_B2": {
        "0.5-r1": 7.8040965785558996,
        "0.5-r5": 7.334935347293744,
        "0.5-r10": 6.018995308387687,
        "0.5-r100": 2.071175191669525,
        "0.7-r1": 2.2313765877102654,
        "0.7-r5": 6.133424876988215,
        "0.7-r10": 7.506579700194536,
        "0.7-r100": 6.476713582789785
      },
      "delta_vs_C7_B1_frozen": {
        "0.5-r1": 4.714498226341689,
        "0.5-r5": 4.451310218560479,
        "0.5-r10": 3.375672273715537,
        "0.5-r100": 0.0,
        "0.7-r1": 2.917953999313422,
        "0.7-r5": 3.6503032383567913,
        "0.7-r10": 3.3642293168554787,
        "0.7-r100": 0.0
      },
      "status_label": "C7_B2_1_SAFE_BUT_R1_DROP"
    },
    {
      "name": "evict_lam0.5_tail0.2_beyond10",
      "metrics": {
        "0.5-r1": 47.62558645153908,
        "0.5-r5": 69.79059388946104,
        "0.5-r10": 75.56928710378762,
        "0.5-r100": 78.36136857764046,
        "0.7-r1": 29.2825266048747,
        "0.7-r5": 54.97196475569287,
        "0.7-r10": 64.0004577182744,
        "0.7-r100": 71.99908456345119
      },
      "movement": {
        "hard_positive_top100_query_exits": 0,
        "hard_positive_top100_query_entries": 0,
        "hard_positive_exit_ratio": 0.0,
        "video_slot_drift_rate": 0.40985697212096117,
        "invalid_span_count": 0,
        "duplicate_span_count_after_nms": 0
      },
      "rank_gain_loss_vs_C7_B1_frozen": {
        "0.5-r1": {
          "gain_queries": 412,
          "loss_queries": 0
        },
        "0.5-r5": {
          "gain_queries": 392,
          "loss_queries": 0
        },
        "0.5-r10": {
          "gain_queries": 298,
          "loss_queries": 9
        },
        "0.7-r1": {
          "gain_queries": 255,
          "loss_queries": 0
        },
        "0.7-r5": {
          "gain_queries": 314,
          "loss_queries": 0
        },
        "0.7-r10": {
          "gain_queries": 318,
          "loss_queries": 24
        }
      },
      "top100_lost_query_count": 0,
      "top100_lost_query_hash": "4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945",
      "config": {
        "lambda_evict": 0.5,
        "tail_coverage_weight": 0.2,
        "apply_only_beyond_rank": 10
      },
      "delta_vs_C6_B1": {
        "0.5-r1": 8.044398672617007,
        "0.5-r5": 7.3578212610138465,
        "0.5-r10": 5.961780524087416,
        "0.5-r100": 2.048289277949422,
        "0.7-r1": 2.6204371209520545,
        "0.7-r5": 6.121981920128164,
        "0.7-r10": 7.563794484494792,
        "0.7-r100": 6.533928367090056
      },
      "delta_vs_C6_B2": {
        "0.5-r1": 7.8040965785558996,
        "0.5-r5": 7.369264217873898,
        "0.5-r10": 5.950337567227365,
        "0.5-r100": 2.071175191669525,
        "0.7-r1": 2.2313765877102654,
        "0.7-r5": 6.076210092687951,
        "0.7-r10": 7.506579700194536,
        "0.7-r100": 6.476713582789785
      },
      "delta_vs_C7_B1_frozen": {
        "0.5-r1": 4.714498226341689,
        "0.5-r5": 4.4856390891406335,
        "0.5-r10": 3.3070145325552147,
        "0.5-r100": 0.0,
        "0.7-r1": 2.917953999313422,
        "0.7-r5": 3.5930884540565273,
        "0.7-r10": 3.3642293168554787,
        "0.7-r100": 0.0
      },
      "status_label": "C7_B2_1_SAFE_BUT_R1_DROP"
    },
    {
      "name": "evict_lam0.5_tail0.2_beyond20",
      "metrics": {
        "0.5-r1": 47.62558645153908,
        "0.5-r5": 69.79059388946104,
        "0.5-r10": 75.66083075866804,
        "0.5-r100": 78.36136857764046,
        "0.7-r1": 29.2825266048747,
        "0.7-r5": 54.97196475569287,
        "0.7-r10": 64.14921615745509,
        "0.7-r100": 71.99908456345119
      },
      "movement": {
        "hard_positive_top100_query_exits": 0,
        "hard_positive_top100_query_entries": 0,
        "hard_positive_exit_ratio": 0.0,
        "video_slot_drift_rate": 0.40985697212096117,
        "invalid_span_count": 0,
        "duplicate_span_count_after_nms": 0
      },
      "rank_gain_loss_vs_C7_B1_frozen": {
        "0.5-r1": {
          "gain_queries": 412,
          "loss_queries": 0
        },
        "0.5-r5": {
          "gain_queries": 392,
          "loss_queries": 0
        },
        "0.5-r10": {
          "gain_queries": 297,
          "loss_queries": 0
        },
        "0.7-r1": {
          "gain_queries": 255,
          "loss_queries": 0
        },
        "0.7-r5": {
          "gain_queries": 314,
          "loss_queries": 0
        },
        "0.7-r10": {
          "gain_queries": 308,
          "loss_queries": 1
        }
      },
      "top100_lost_query_count": 0,
      "top100_lost_query_hash": "4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945",
      "config": {
        "lambda_evict": 0.5,
        "tail_coverage_weight": 0.2,
        "apply_only_beyond_rank": 20
      },
      "delta_vs_C6_B1": {
        "0.5-r1": 8.044398672617007,
        "0.5-r5": 7.3578212610138465,
        "0.5-r10": 6.053324178967841,
        "0.5-r100": 2.048289277949422,
        "0.7-r1": 2.6204371209520545,
        "0.7-r5": 6.121981920128164,
        "0.7-r10": 7.712552923675474,
        "0.7-r100": 6.533928367090056
      },
      "delta_vs_C6_B2": {
        "0.5-r1": 7.8040965785558996,
        "0.5-r5": 7.369264217873898,
        "0.5-r10": 6.04188122210779,
        "0.5-r100": 2.071175191669525,
        "0.7-r1": 2.2313765877102654,
        "0.7-r5": 6.076210092687951,
        "0.7-r10": 7.655338139375218,
        "0.7-r100": 6.476713582789785
      },
      "delta_vs_C7_B1_frozen": {
        "0.5-r1": 4.714498226341689,
        "0.5-r5": 4.4856390891406335,
        "0.5-r10": 3.39855818743564,
        "0.5-r100": 0.0,
        "0.7-r1": 2.917953999313422,
        "0.7-r5": 3.5930884540565273,
        "0.7-r10": 3.5129877560361606,
        "0.7-r100": 0.0
      },
      "status_label": "C7_B2_1_SAFE_BUT_R1_DROP"
    },
    {
      "name": "evict_lam0.5_tail0.5_beyond5",
      "metrics": {
        "0.5-r1": 47.62558645153908,
        "0.5-r5": 69.76770797574093,
        "0.5-r10": 75.63794484494794,
        "0.5-r100": 78.36136857764046,
        "0.7-r1": 29.2825266048747,
        "0.7-r5": 55.05206545371324,
        "0.7-r10": 64.0004577182744,
        "0.7-r100": 71.99908456345119
      },
      "movement": {
        "hard_positive_top100_query_exits": 0,
        "hard_positive_top100_query_entries": 0,
        "hard_positive_exit_ratio": 0.0,
        "video_slot_drift_rate": 0.40985697212096117,
        "invalid_span_count": 0,
        "duplicate_span_count_after_nms": 0
      },
      "rank_gain_loss_vs_C7_B1_frozen": {
        "0.5-r1": {
          "gain_queries": 412,
          "loss_queries": 0
        },
        "0.5-r5": {
          "gain_queries": 399,
          "loss_queries": 9
        },
        "0.5-r10": {
          "gain_queries": 296,
          "loss_queries": 1
        },
        "0.7-r1": {
          "gain_queries": 255,
          "loss_queries": 0
        },
        "0.7-r5": {
          "gain_queries": 341,
          "loss_queries": 20
        },
        "0.7-r10": {
          "gain_queries": 310,
          "loss_queries": 16
        }
      },
      "top100_lost_query_count": 0,
      "top100_lost_query_hash": "4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945",
      "config": {
        "lambda_evict": 0.5,
        "tail_coverage_weight": 0.5,
        "apply_only_beyond_rank": 5
      },
      "delta_vs_C6_B1": {
        "0.5-r1": 8.044398672617007,
        "0.5-r5": 7.334935347293744,
        "0.5-r10": 6.030438265247739,
        "0.5-r100": 2.048289277949422,
        "0.7-r1": 2.6204371209520545,
        "0.7-r5": 6.2020826181485305,
        "0.7-r10": 7.563794484494792,
        "0.7-r100": 6.533928367090056
      },
      "delta_vs_C6_B2": {
        "0.5-r1": 7.8040965785558996,
        "0.5-r5": 7.346378304153795,
        "0.5-r10": 6.018995308387687,
        "0.5-r100": 2.071175191669525,
        "0.7-r1": 2.2313765877102654,
        "0.7-r5": 6.156310790708318,
        "0.7-r10": 7.506579700194536,
        "0.7-r100": 6.476713582789785
      },
      "delta_vs_C7_B1_frozen": {
        "0.5-r1": 4.714498226341689,
        "0.5-r5": 4.462753175420531,
        "0.5-r10": 3.375672273715537,
        "0.5-r100": 0.0,
        "0.7-r1": 2.917953999313422,
        "0.7-r5": 3.673189152076894,
        "0.7-r10": 3.3642293168554787,
        "0.7-r100": 0.0
      },
      "status_label": "C7_B2_1_SAFE_BUT_R1_DROP"
    },
    {
      "name": "evict_lam0.5_tail0.5_beyond10",
      "metrics": {
        "0.5-r1": 47.62558645153908,
        "0.5-r5": 69.79059388946104,
        "0.5-r10": 75.56928710378762,
        "0.5-r100": 78.36136857764046,
        "0.7-r1": 29.2825266048747,
        "0.7-r5": 54.97196475569287,
        "0.7-r10": 63.98901476141435,
        "0.7-r100": 71.99908456345119
      },
      "movement": {
        "hard_positive_top100_query_exits": 0,
        "hard_positive_top100_query_entries": 0,
        "hard_positive_exit_ratio": 0.0,
        "video_slot_drift_rate": 0.40985697212096117,
        "invalid_span_count": 0,
        "duplicate_span_count_after_nms": 0
      },
      "rank_gain_loss_vs_C7_B1_frozen": {
        "0.5-r1": {
          "gain_queries": 412,
          "loss_queries": 0
        },
        "0.5-r5": {
          "gain_queries": 392,
          "loss_queries": 0
        },
        "0.5-r10": {
          "gain_queries": 298,
          "loss_queries": 9
        },
        "0.7-r1": {
          "gain_queries": 255,
          "loss_queries": 0
        },
        "0.7-r5": {
          "gain_queries": 314,
          "loss_queries": 0
        },
        "0.7-r10": {
          "gain_queries": 317,
          "loss_queries": 24
        }
      },
      "top100_lost_query_count": 0,
      "top100_lost_query_hash": "4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945",
      "config": {
        "lambda_evict": 0.5,
        "tail_coverage_weight": 0.5,
        "apply_only_beyond_rank": 10
      },
      "delta_vs_C6_B1": {
        "0.5-r1": 8.044398672617007,
        "0.5-r5": 7.3578212610138465,
        "0.5-r10": 5.961780524087416,
        "0.5-r100": 2.048289277949422,
        "0.7-r1": 2.6204371209520545,
        "0.7-r5": 6.121981920128164,
        "0.7-r10": 7.552351527634741,
        "0.7-r100": 6.533928367090056
      },
      "delta_vs_C6_B2": {
        "0.5-r1": 7.8040965785558996,
        "0.5-r5": 7.369264217873898,
        "0.5-r10": 5.950337567227365,
        "0.5-r100": 2.071175191669525,
        "0.7-r1": 2.2313765877102654,
        "0.7-r5": 6.076210092687951,
        "0.7-r10": 7.495136743334484,
        "0.7-r100": 6.476713582789785
      },
      "delta_vs_C7_B1_frozen": {
        "0.5-r1": 4.714498226341689,
        "0.5-r5": 4.4856390891406335,
        "0.5-r10": 3.3070145325552147,
        "0.5-r100": 0.0,
        "0.7-r1": 2.917953999313422,
        "0.7-r5": 3.5930884540565273,
        "0.7-r10": 3.3527863599954273,
        "0.7-r100": 0.0
      },
      "status_label": "C7_B2_1_SAFE_BUT_R1_DROP"
    },
    {
      "name": "evict_lam0.5_tail0.5_beyond20",
      "metrics": {
        "0.5-r1": 47.62558645153908,
        "0.5-r5": 69.79059388946104,
        "0.5-r10": 75.66083075866804,
        "0.5-r100": 78.36136857764046,
        "0.7-r1": 29.2825266048747,
        "0.7-r5": 54.97196475569287,
        "0.7-r10": 64.14921615745509,
        "0.7-r100": 71.99908456345119
      },
      "movement": {
        "hard_positive_top100_query_exits": 0,
        "hard_positive_top100_query_entries": 0,
        "hard_positive_exit_ratio": 0.0,
        "video_slot_drift_rate": 0.40985697212096117,
        "invalid_span_count": 0,
        "duplicate_span_count_after_nms": 0
      },
      "rank_gain_loss_vs_C7_B1_frozen": {
        "0.5-r1": {
          "gain_queries": 412,
          "loss_queries": 0
        },
        "0.5-r5": {
          "gain_queries": 392,
          "loss_queries": 0
        },
        "0.5-r10": {
          "gain_queries": 297,
          "loss_queries": 0
        },
        "0.7-r1": {
          "gain_queries": 255,
          "loss_queries": 0
        },
        "0.7-r5": {
          "gain_queries": 314,
          "loss_queries": 0
        },
        "0.7-r10": {
          "gain_queries": 308,
          "loss_queries": 1
        }
      },
      "top100_lost_query_count": 0,
      "top100_lost_query_hash": "4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945",
      "config": {
        "lambda_evict": 0.5,
        "tail_coverage_weight": 0.5,
        "apply_only_beyond_rank": 20
      },
      "delta_vs_C6_B1": {
        "0.5-r1": 8.044398672617007,
        "0.5-r5": 7.3578212610138465,
        "0.5-r10": 6.053324178967841,
        "0.5-r100": 2.048289277949422,
        "0.7-r1": 2.6204371209520545,
        "0.7-r5": 6.121981920128164,
        "0.7-r10": 7.712552923675474,
        "0.7-r100": 6.533928367090056
      },
      "delta_vs_C6_B2": {
        "0.5-r1": 7.8040965785558996,
        "0.5-r5": 7.369264217873898,
        "0.5-r10": 6.04188122210779,
        "0.5-r100": 2.071175191669525,
        "0.7-r1": 2.2313765877102654,
        "0.7-r5": 6.076210092687951,
        "0.7-r10": 7.655338139375218,
        "0.7-r100": 6.476713582789785
      },
      "delta_vs_C7_B1_frozen": {
        "0.5-r1": 4.714498226341689,
        "0.5-r5": 4.4856390891406335,
        "0.5-r10": 3.39855818743564,
        "0.5-r100": 0.0,
        "0.7-r1": 2.917953999313422,
        "0.7-r5": 3.5930884540565273,
        "0.7-r10": 3.5129877560361606,
        "0.7-r100": 0.0
      },
      "status_label": "C7_B2_1_SAFE_BUT_R1_DROP"
    },
    {
      "name": "evict_lam1.0_tail0.1_beyond5",
      "metrics": {
        "0.5-r1": 47.62558645153908,
        "0.5-r5": 69.91646641492162,
        "0.5-r10": 75.66083075866804,
        "0.5-r100": 78.36136857764046,
        "0.7-r1": 29.2825266048747,
        "0.7-r5": 55.258038677194186,
        "0.7-r10": 63.89747110653393,
        "0.7-r100": 71.99908456345119
      },
      "movement": {
        "hard_positive_top100_query_exits": 0,
        "hard_positive_top100_query_entries": 0,
        "hard_positive_exit_ratio": 0.0,
        "video_slot_drift_rate": 0.40985697212096117,
        "invalid_span_count": 0,
        "duplicate_span_count_after_nms": 0
      },
      "rank_gain_loss_vs_C7_B1_frozen": {
        "0.5-r1": {
          "gain_queries": 412,
          "loss_queries": 0
        },
        "0.5-r5": {
          "gain_queries": 420,
          "loss_queries": 17
        },
        "0.5-r10": {
          "gain_queries": 299,
          "loss_queries": 2
        },
        "0.7-r1": {
          "gain_queries": 255,
          "loss_queries": 0
        },
        "0.7-r5": {
          "gain_queries": 387,
          "loss_queries": 48
        },
        "0.7-r10": {
          "gain_queries": 319,
          "loss_queries": 34
        }
      },
      "top100_lost_query_count": 0,
      "top100_lost_query_hash": "4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945",
      "config": {
        "lambda_evict": 1.0,
        "tail_coverage_weight": 0.1,
        "apply_only_beyond_rank": 5
      },
      "delta_vs_C6_B1": {
        "0.5-r1": 8.044398672617007,
        "0.5-r5": 7.483693786474426,
        "0.5-r10": 6.053324178967841,
        "0.5-r100": 2.048289277949422,
        "0.7-r1": 2.6204371209520545,
        "0.7-r5": 6.408055841629476,
        "0.7-r10": 7.460807872754316,
        "0.7-r100": 6.533928367090056
      },
      "delta_vs_C6_B2": {
        "0.5-r1": 7.8040965785558996,
        "0.5-r5": 7.495136743334477,
        "0.5-r10": 6.04188122210779,
        "0.5-r100": 2.071175191669525,
        "0.7-r1": 2.2313765877102654,
        "0.7-r5": 6.362284014189264,
        "0.7-r10": 7.403593088454059,
        "0.7-r100": 6.476713582789785
      },
      "delta_vs_C7_B1_frozen": {
        "0.5-r1": 4.714498226341689,
        "0.5-r5": 4.611511614601213,
        "0.5-r10": 3.39855818743564,
        "0.5-r100": 0.0,
        "0.7-r1": 2.917953999313422,
        "0.7-r5": 3.87916237555784,
        "0.7-r10": 3.261242705115002,
        "0.7-r100": 0.0
      },
      "status_label": "C7_B2_1_SAFE_BUT_R1_DROP"
    },
    {
      "name": "evict_lam1.0_tail0.1_beyond10",
      "metrics": {
        "0.5-r1": 47.62558645153908,
        "0.5-r5": 69.81347980318114,
        "0.5-r10": 75.36331388030668,
        "0.5-r100": 78.36136857764046,
        "0.7-r1": 29.2825266048747,
        "0.7-r5": 54.97196475569287,
        "0.7-r10": 63.725826753633136,
        "0.7-r100": 71.99908456345119
      },
      "movement": {
        "hard_positive_top100_query_exits": 0,
        "hard_positive_top100_query_entries": 0,
        "hard_positive_exit_ratio": 0.0,
        "video_slot_drift_rate": 0.40985697212096117,
        "invalid_span_count": 0,
        "duplicate_span_count_after_nms": 0
      },
      "rank_gain_loss_vs_C7_B1_frozen": {
        "0.5-r1": {
          "gain_queries": 412,
          "loss_queries": 0
        },
        "0.5-r5": {
          "gain_queries": 395,
          "loss_queries": 1
        },
        "0.5-r10": {
          "gain_queries": 284,
          "loss_queries": 13
        },
        "0.7-r1": {
          "gain_queries": 255,
          "loss_queries": 0
        },
        "0.7-r5": {
          "gain_queries": 314,
          "loss_queries": 0
        },
        "0.7-r10": {
          "gain_queries": 310,
          "loss_queries": 40
        }
      },
      "top100_lost_query_count": 0,
      "top100_lost_query_hash": "4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945",
      "config": {
        "lambda_evict": 1.0,
        "tail_coverage_weight": 0.1,
        "apply_only_beyond_rank": 10
      },
      "delta_vs_C6_B1": {
        "0.5-r1": 8.044398672617007,
        "0.5-r5": 7.380707174733949,
        "0.5-r10": 5.755807300606477,
        "0.5-r100": 2.048289277949422,
        "0.7-r1": 2.6204371209520545,
        "0.7-r5": 6.121981920128164,
        "0.7-r10": 7.289163519853524,
        "0.7-r100": 6.533928367090056
      },
      "delta_vs_C6_B2": {
        "0.5-r1": 7.8040965785558996,
        "0.5-r5": 7.392150131594001,
        "0.5-r10": 5.744364343746426,
        "0.5-r100": 2.071175191669525,
        "0.7-r1": 2.2313765877102654,
        "0.7-r5": 6.076210092687951,
        "0.7-r10": 7.231948735553267,
        "0.7-r100": 6.476713582789785
      },
      "delta_vs_C7_B1_frozen": {
        "0.5-r1": 4.714498226341689,
        "0.5-r5": 4.508525002860736,
        "0.5-r10": 3.101041309074276,
        "0.5-r100": 0.0,
        "0.7-r1": 2.917953999313422,
        "0.7-r5": 3.5930884540565273,
        "0.7-r10": 3.0895983522142103,
        "0.7-r100": 0.0
      },
      "status_label": "C7_B2_1_SAFE_BUT_R1_DROP"
    },
    {
      "name": "evict_lam1.0_tail0.1_beyond20",
      "metrics": {
        "0.5-r1": 47.62558645153908,
        "0.5-r5": 69.79059388946104,
        "0.5-r10": 75.66083075866804,
        "0.5-r100": 78.36136857764046,
        "0.7-r1": 29.2825266048747,
        "0.7-r5": 54.97196475569287,
        "0.7-r10": 64.14921615745509,
        "0.7-r100": 71.99908456345119
      },
      "movement": {
        "hard_positive_top100_query_exits": 0,
        "hard_positive_top100_query_entries": 0,
        "hard_positive_exit_ratio": 0.0,
        "video_slot_drift_rate": 0.40985697212096117,
        "invalid_span_count": 0,
        "duplicate_span_count_after_nms": 0
      },
      "rank_gain_loss_vs_C7_B1_frozen": {
        "0.5-r1": {
          "gain_queries": 412,
          "loss_queries": 0
        },
        "0.5-r5": {
          "gain_queries": 392,
          "loss_queries": 0
        },
        "0.5-r10": {
          "gain_queries": 297,
          "loss_queries": 0
        },
        "0.7-r1": {
          "gain_queries": 255,
          "loss_queries": 0
        },
        "0.7-r5": {
          "gain_queries": 314,
          "loss_queries": 0
        },
        "0.7-r10": {
          "gain_queries": 308,
          "loss_queries": 1
        }
      },
      "top100_lost_query_count": 0,
      "top100_lost_query_hash": "4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945",
      "config": {
        "lambda_evict": 1.0,
        "tail_coverage_weight": 0.1,
        "apply_only_beyond_rank": 20
      },
      "delta_vs_C6_B1": {
        "0.5-r1": 8.044398672617007,
        "0.5-r5": 7.3578212610138465,
        "0.5-r10": 6.053324178967841,
        "0.5-r100": 2.048289277949422,
        "0.7-r1": 2.6204371209520545,
        "0.7-r5": 6.121981920128164,
        "0.7-r10": 7.712552923675474,
        "0.7-r100": 6.533928367090056
      },
      "delta_vs_C6_B2": {
        "0.5-r1": 7.8040965785558996,
        "0.5-r5": 7.369264217873898,
        "0.5-r10": 6.04188122210779,
        "0.5-r100": 2.071175191669525,
        "0.7-r1": 2.2313765877102654,
        "0.7-r5": 6.076210092687951,
        "0.7-r10": 7.655338139375218,
        "0.7-r100": 6.476713582789785
      },
      "delta_vs_C7_B1_frozen": {
        "0.5-r1": 4.714498226341689,
        "0.5-r5": 4.4856390891406335,
        "0.5-r10": 3.39855818743564,
        "0.5-r100": 0.0,
        "0.7-r1": 2.917953999313422,
        "0.7-r5": 3.5930884540565273,
        "0.7-r10": 3.5129877560361606,
        "0.7-r100": 0.0
      },
      "status_label": "C7_B2_1_SAFE_BUT_R1_DROP"
    },
    {
      "name": "evict_lam1.0_tail0.2_beyond5",
      "metrics": {
        "0.5-r1": 47.62558645153908,
        "0.5-r5": 69.90502345806156,
        "0.5-r10": 75.66083075866804,
        "0.5-r100": 78.36136857764046,
        "0.7-r1": 29.2825266048747,
        "0.7-r5": 55.246595720334135,
        "0.7-r10": 63.89747110653393,
        "0.7-r100": 71.99908456345119
      },
      "movement": {
        "hard_positive_top100_query_exits": 0,
        "hard_positive_top100_query_entries": 0,
        "hard_positive_exit_ratio": 0.0,
        "video_slot_drift_rate": 0.40985697212096117,
        "invalid_span_count": 0,
        "duplicate_span_count_after_nms": 0
      },
      "rank_gain_loss_vs_C7_B1_frozen": {
        "0.5-r1": {
          "gain_queries": 412,
          "loss_queries": 0
        },
        "0.5-r5": {
          "gain_queries": 419,
          "loss_queries": 17
        },
        "0.5-r10": {
          "gain_queries": 299,
          "loss_queries": 2
        },
        "0.7-r1": {
          "gain_queries": 255,
          "loss_queries": 0
        },
        "0.7-r5": {
          "gain_queries": 385,
          "loss_queries": 47
        },
        "0.7-r10": {
          "gain_queries": 319,
          "loss_queries": 34
        }
      },
      "top100_lost_query_count": 0,
      "top100_lost_query_hash": "4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945",
      "config": {
        "lambda_evict": 1.0,
        "tail_coverage_weight": 0.2,
        "apply_only_beyond_rank": 5
      },
      "delta_vs_C6_B1": {
        "0.5-r1": 8.044398672617007,
        "0.5-r5": 7.472250829614374,
        "0.5-r10": 6.053324178967841,
        "0.5-r100": 2.048289277949422,
        "0.7-r1": 2.6204371209520545,
        "0.7-r5": 6.396612884769425,
        "0.7-r10": 7.460807872754316,
        "0.7-r100": 6.533928367090056
      },
      "delta_vs_C6_B2": {
        "0.5-r1": 7.8040965785558996,
        "0.5-r5": 7.483693786474426,
        "0.5-r10": 6.04188122210779,
        "0.5-r100": 2.071175191669525,
        "0.7-r1": 2.2313765877102654,
        "0.7-r5": 6.3508410573292124,
        "0.7-r10": 7.403593088454059,
        "0.7-r100": 6.476713582789785
      },
      "delta_vs_C7_B1_frozen": {
        "0.5-r1": 4.714498226341689,
        "0.5-r5": 4.600068657741161,
        "0.5-r10": 3.39855818743564,
        "0.5-r100": 0.0,
        "0.7-r1": 2.917953999313422,
        "0.7-r5": 3.8677194186977886,
        "0.7-r10": 3.261242705115002,
        "0.7-r100": 0.0
      },
      "status_label": "C7_B2_1_SAFE_BUT_R1_DROP"
    },
    {
      "name": "evict_lam1.0_tail0.2_beyond10",
      "metrics": {
        "0.5-r1": 47.62558645153908,
        "0.5-r5": 69.81347980318114,
        "0.5-r10": 75.36331388030668,
        "0.5-r100": 78.36136857764046,
        "0.7-r1": 29.2825266048747,
        "0.7-r5": 54.97196475569287,
        "0.7-r10": 63.725826753633136,
        "0.7-r100": 71.99908456345119
      },
      "movement": {
        "hard_positive_top100_query_exits": 0,
        "hard_positive_top100_query_entries": 0,
        "hard_positive_exit_ratio": 0.0,
        "video_slot_drift_rate": 0.40985697212096117,
        "invalid_span_count": 0,
        "duplicate_span_count_after_nms": 0
      },
      "rank_gain_loss_vs_C7_B1_frozen": {
        "0.5-r1": {
          "gain_queries": 412,
          "loss_queries": 0
        },
        "0.5-r5": {
          "gain_queries": 395,
          "loss_queries": 1
        },
        "0.5-r10": {
          "gain_queries": 284,
          "loss_queries": 13
        },
        "0.7-r1": {
          "gain_queries": 255,
          "loss_queries": 0
        },
        "0.7-r5": {
          "gain_queries": 314,
          "loss_queries": 0
        },
        "0.7-r10": {
          "gain_queries": 310,
          "loss_queries": 40
        }
      },
      "top100_lost_query_count": 0,
      "top100_lost_query_hash": "4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945",
      "config": {
        "lambda_evict": 1.0,
        "tail_coverage_weight": 0.2,
        "apply_only_beyond_rank": 10
      },
      "delta_vs_C6_B1": {
        "0.5-r1": 8.044398672617007,
        "0.5-r5": 7.380707174733949,
        "0.5-r10": 5.755807300606477,
        "0.5-r100": 2.048289277949422,
        "0.7-r1": 2.6204371209520545,
        "0.7-r5": 6.121981920128164,
        "0.7-r10": 7.289163519853524,
        "0.7-r100": 6.533928367090056
      },
      "delta_vs_C6_B2": {
        "0.5-r1": 7.8040965785558996,
        "0.5-r5": 7.392150131594001,
        "0.5-r10": 5.744364343746426,
        "0.5-r100": 2.071175191669525,
        "0.7-r1": 2.2313765877102654,
        "0.7-r5": 6.076210092687951,
        "0.7-r10": 7.231948735553267,
        "0.7-r100": 6.476713582789785
      },
      "delta_vs_C7_B1_frozen": {
        "0.5-r1": 4.714498226341689,
        "0.5-r5": 4.508525002860736,
        "0.5-r10": 3.101041309074276,
        "0.5-r100": 0.0,
        "0.7-r1": 2.917953999313422,
        "0.7-r5": 3.5930884540565273,
        "0.7-r10": 3.0895983522142103,
        "0.7-r100": 0.0
      },
      "status_label": "C7_B2_1_SAFE_BUT_R1_DROP"
    },
    {
      "name": "evict_lam1.0_tail0.2_beyond20",
      "metrics": {
        "0.5-r1": 47.62558645153908,
        "0.5-r5": 69.79059388946104,
        "0.5-r10": 75.66083075866804,
        "0.5-r100": 78.36136857764046,
        "0.7-r1": 29.2825266048747,
        "0.7-r5": 54.97196475569287,
        "0.7-r10": 64.14921615745509,
        "0.7-r100": 71.99908456345119
      },
      "movement": {
        "hard_positive_top100_query_exits": 0,
        "hard_positive_top100_query_entries": 0,
        "hard_positive_exit_ratio": 0.0,
        "video_slot_drift_rate": 0.40985697212096117,
        "invalid_span_count": 0,
        "duplicate_span_count_after_nms": 0
      },
      "rank_gain_loss_vs_C7_B1_frozen": {
        "0.5-r1": {
          "gain_queries": 412,
          "loss_queries": 0
        },
        "0.5-r5": {
          "gain_queries": 392,
          "loss_queries": 0
        },
        "0.5-r10": {
          "gain_queries": 297,
          "loss_queries": 0
        },
        "0.7-r1": {
          "gain_queries": 255,
          "loss_queries": 0
        },
        "0.7-r5": {
          "gain_queries": 314,
          "loss_queries": 0
        },
        "0.7-r10": {
          "gain_queries": 308,
          "loss_queries": 1
        }
      },
      "top100_lost_query_count": 0,
      "top100_lost_query_hash": "4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945",
      "config": {
        "lambda_evict": 1.0,
        "tail_coverage_weight": 0.2,
        "apply_only_beyond_rank": 20
      },
      "delta_vs_C6_B1": {
        "0.5-r1": 8.044398672617007,
        "0.5-r5": 7.3578212610138465,
        "0.5-r10": 6.053324178967841,
        "0.5-r100": 2.048289277949422,
        "0.7-r1": 2.6204371209520545,
        "0.7-r5": 6.121981920128164,
        "0.7-r10": 7.712552923675474,
        "0.7-r100": 6.533928367090056
      },
      "delta_vs_C6_B2": {
        "0.5-r1": 7.8040965785558996,
        "0.5-r5": 7.369264217873898,
        "0.5-r10": 6.04188122210779,
        "0.5-r100": 2.071175191669525,
        "0.7-r1": 2.2313765877102654,
        "0.7-r5": 6.076210092687951,
        "0.7-r10": 7.655338139375218,
        "0.7-r100": 6.476713582789785
      },
      "delta_vs_C7_B1_frozen": {
        "0.5-r1": 4.714498226341689,
        "0.5-r5": 4.4856390891406335,
        "0.5-r10": 3.39855818743564,
        "0.5-r100": 0.0,
        "0.7-r1": 2.917953999313422,
        "0.7-r5": 3.5930884540565273,
        "0.7-r10": 3.5129877560361606,
        "0.7-r100": 0.0
      },
      "status_label": "C7_B2_1_SAFE_BUT_R1_DROP"
    },
    {
      "name": "evict_lam1.0_tail0.5_beyond5",
      "metrics": {
        "0.5-r1": 47.62558645153908,
        "0.5-r5": 69.93935232864172,
        "0.5-r10": 75.66083075866804,
        "0.5-r100": 78.36136857764046,
        "0.7-r1": 29.2825266048747,
        "0.7-r5": 55.22370980661403,
        "0.7-r10": 63.89747110653393,
        "0.7-r100": 71.99908456345119
      },
      "movement": {
        "hard_positive_top100_query_exits": 0,
        "hard_positive_top100_query_entries": 0,
        "hard_positive_exit_ratio": 0.0,
        "video_slot_drift_rate": 0.40985697212096117,
        "invalid_span_count": 0,
        "duplicate_span_count_after_nms": 0
      },
      "rank_gain_loss_vs_C7_B1_frozen": {
        "0.5-r1": {
          "gain_queries": 412,
          "loss_queries": 0
        },
        "0.5-r5": {
          "gain_queries": 421,
          "loss_queries": 16
        },
        "0.5-r10": {
          "gain_queries": 299,
          "loss_queries": 2
        },
        "0.7-r1": {
          "gain_queries": 255,
          "loss_queries": 0
        },
        "0.7-r5": {
          "gain_queries": 381,
          "loss_queries": 45
        },
        "0.7-r10": {
          "gain_queries": 319,
          "loss_queries": 34
        }
      },
      "top100_lost_query_count": 0,
      "top100_lost_query_hash": "4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945",
      "config": {
        "lambda_evict": 1.0,
        "tail_coverage_weight": 0.5,
        "apply_only_beyond_rank": 5
      },
      "delta_vs_C6_B1": {
        "0.5-r1": 8.044398672617007,
        "0.5-r5": 7.5065797001945285,
        "0.5-r10": 6.053324178967841,
        "0.5-r100": 2.048289277949422,
        "0.7-r1": 2.6204371209520545,
        "0.7-r5": 6.373726971049322,
        "0.7-r10": 7.460807872754316,
        "0.7-r100": 6.533928367090056
      },
      "delta_vs_C6_B2": {
        "0.5-r1": 7.8040965785558996,
        "0.5-r5": 7.51802265705458,
        "0.5-r10": 6.04188122210779,
        "0.5-r100": 2.071175191669525,
        "0.7-r1": 2.2313765877102654,
        "0.7-r5": 6.32795514360911,
        "0.7-r10": 7.403593088454059,
        "0.7-r100": 6.476713582789785
      },
      "delta_vs_C7_B1_frozen": {
        "0.5-r1": 4.714498226341689,
        "0.5-r5": 4.6343975283213155,
        "0.5-r10": 3.39855818743564,
        "0.5-r100": 0.0,
        "0.7-r1": 2.917953999313422,
        "0.7-r5": 3.844833504977686,
        "0.7-r10": 3.261242705115002,
        "0.7-r100": 0.0
      },
      "status_label": "C7_B2_1_SAFE_BUT_R1_DROP"
    },
    {
      "name": "evict_lam1.0_tail0.5_beyond10",
      "metrics": {
        "0.5-r1": 47.62558645153908,
        "0.5-r5": 69.81347980318114,
        "0.5-r10": 75.39764275088683,
        "0.5-r100": 78.36136857764046,
        "0.7-r1": 29.2825266048747,
        "0.7-r5": 54.97196475569287,
        "0.7-r10": 63.77159858107335,
        "0.7-r100": 71.99908456345119
      },
      "movement": {
        "hard_positive_top100_query_exits": 0,
        "hard_positive_top100_query_entries": 0,
        "hard_positive_exit_ratio": 0.0,
        "video_slot_drift_rate": 0.40985697212096117,
        "invalid_span_count": 0,
        "duplicate_span_count_after_nms": 0
      },
      "rank_gain_loss_vs_C7_B1_frozen": {
        "0.5-r1": {
          "gain_queries": 412,
          "loss_queries": 0
        },
        "0.5-r5": {
          "gain_queries": 395,
          "loss_queries": 1
        },
        "0.5-r10": {
          "gain_queries": 286,
          "loss_queries": 12
        },
        "0.7-r1": {
          "gain_queries": 255,
          "loss_queries": 0
        },
        "0.7-r5": {
          "gain_queries": 314,
          "loss_queries": 0
        },
        "0.7-r10": {
          "gain_queries": 312,
          "loss_queries": 38
        }
      },
      "top100_lost_query_count": 0,
      "top100_lost_query_hash": "4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945",
      "config": {
        "lambda_evict": 1.0,
        "tail_coverage_weight": 0.5,
        "apply_only_beyond_rank": 10
      },
      "delta_vs_C6_B1": {
        "0.5-r1": 8.044398672617007,
        "0.5-r5": 7.380707174733949,
        "0.5-r10": 5.7901361711866315,
        "0.5-r100": 2.048289277949422,
        "0.7-r1": 2.6204371209520545,
        "0.7-r5": 6.121981920128164,
        "0.7-r10": 7.334935347293737,
        "0.7-r100": 6.533928367090056
      },
      "delta_vs_C6_B2": {
        "0.5-r1": 7.8040965785558996,
        "0.5-r5": 7.392150131594001,
        "0.5-r10": 5.77869321432658,
        "0.5-r100": 2.071175191669525,
        "0.7-r1": 2.2313765877102654,
        "0.7-r5": 6.076210092687951,
        "0.7-r10": 7.27772056299348,
        "0.7-r100": 6.476713582789785
      },
      "delta_vs_C7_B1_frozen": {
        "0.5-r1": 4.714498226341689,
        "0.5-r5": 4.508525002860736,
        "0.5-r10": 3.13537017965443,
        "0.5-r100": 0.0,
        "0.7-r1": 2.917953999313422,
        "0.7-r5": 3.5930884540565273,
        "0.7-r10": 3.135370179654423,
        "0.7-r100": 0.0
      },
      "status_label": "C7_B2_1_SAFE_BUT_R1_DROP"
    },
    {
      "name": "evict_lam1.0_tail0.5_beyond20",
      "metrics": {
        "0.5-r1": 47.62558645153908,
        "0.5-r5": 69.79059388946104,
        "0.5-r10": 75.66083075866804,
        "0.5-r100": 78.36136857764046,
        "0.7-r1": 29.2825266048747,
        "0.7-r5": 54.97196475569287,
        "0.7-r10": 64.14921615745509,
        "0.7-r100": 71.99908456345119
      },
      "movement": {
        "hard_positive_top100_query_exits": 0,
        "hard_positive_top100_query_entries": 0,
        "hard_positive_exit_ratio": 0.0,
        "video_slot_drift_rate": 0.40985697212096117,
        "invalid_span_count": 0,
        "duplicate_span_count_after_nms": 0
      },
      "rank_gain_loss_vs_C7_B1_frozen": {
        "0.5-r1": {
          "gain_queries": 412,
          "loss_queries": 0
        },
        "0.5-r5": {
          "gain_queries": 392,
          "loss_queries": 0
        },
        "0.5-r10": {
          "gain_queries": 297,
          "loss_queries": 0
        },
        "0.7-r1": {
          "gain_queries": 255,
          "loss_queries": 0
        },
        "0.7-r5": {
          "gain_queries": 314,
          "loss_queries": 0
        },
        "0.7-r10": {
          "gain_queries": 308,
          "loss_queries": 1
        }
      },
      "top100_lost_query_count": 0,
      "top100_lost_query_hash": "4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945",
      "config": {
        "lambda_evict": 1.0,
        "tail_coverage_weight": 0.5,
        "apply_only_beyond_rank": 20
      },
      "delta_vs_C6_B1": {
        "0.5-r1": 8.044398672617007,
        "0.5-r5": 7.3578212610138465,
        "0.5-r10": 6.053324178967841,
        "0.5-r100": 2.048289277949422,
        "0.7-r1": 2.6204371209520545,
        "0.7-r5": 6.121981920128164,
        "0.7-r10": 7.712552923675474,
        "0.7-r100": 6.533928367090056
      },
      "delta_vs_C6_B2": {
        "0.5-r1": 7.8040965785558996,
        "0.5-r5": 7.369264217873898,
        "0.5-r10": 6.04188122210779,
        "0.5-r100": 2.071175191669525,
        "0.7-r1": 2.2313765877102654,
        "0.7-r5": 6.076210092687951,
        "0.7-r10": 7.655338139375218,
        "0.7-r100": 6.476713582789785
      },
      "delta_vs_C7_B1_frozen": {
        "0.5-r1": 4.714498226341689,
        "0.5-r5": 4.4856390891406335,
        "0.5-r10": 3.39855818743564,
        "0.5-r100": 0.0,
        "0.7-r1": 2.917953999313422,
        "0.7-r5": 3.5930884540565273,
        "0.7-r10": 3.5129877560361606,
        "0.7-r100": 0.0
      },
      "status_label": "C7_B2_1_SAFE_BUT_R1_DROP"
    }
  ]
}
```
