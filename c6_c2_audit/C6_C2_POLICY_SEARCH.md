# C6-C2 policy search

- Scope: `train_calib only`
- Official val used: `false`
- A0 fallback: `always available`

## Summary

```json
{
  "result_count": 1296,
  "feasible_count": 0,
  "nonnoop_count": 816,
  "best_feasible": null,
  "best_nonnoop": {
    "config": {
      "config_id": "c6c2_policy_00435",
      "allowed_actions": [
        "A1_MAVR_soft_rerank",
        "A2_PREM_PRVR_MIL_promote",
        "A3_topK_preserving_promote",
        "A5_BMN_proposal_confidence_boost"
      ],
      "enable_threshold": 0.17919921875,
      "utility_margin": 0.0,
      "max_enabled_query_rate": 0.02,
      "max_video_slot_change_rate": 0.05,
      "max_actions_per_query": 1,
      "fallback": "A0_noop_B2"
    },
    "metrics": {
      "0.5-r1": 41.58370522943129,
      "0.5-r5": 65.38505549834078,
      "0.5-r10": 72.29660144181257,
      "0.5-r100": 78.34992562078041,
      "0.7-r1": 25.677995193958118,
      "0.7-r5": 49.93706373726971,
      "0.7-r10": 59.07998626845177,
      "0.7-r100": 71.99908456345119
    },
    "delta_vs_C6_B1": {
      "0.5-r1": 0.27463096464126835,
      "0.5-r5": 0.05721478430027105,
      "0.5-r10": -0.011442956860051368,
      "0.5-r100": -0.022885913720102735,
      "0.7-r1": 0.37761757638173776,
      "0.7-r5": 0.045771827440212576,
      "0.7-r10": 0.02288591372010984,
      "0.7-r100": 0.022885913720102735
    },
    "delta_vs_C6_B2": {
      "0.5-r1": 0.02288591372010984,
      "0.5-r5": -0.011442956860051368,
      "0.5-r10": 0.0,
      "0.5-r100": 0.011442956860051368,
      "0.7-r1": 0.011442956860051368,
      "0.7-r5": -0.011442956860051368,
      "0.7-r10": 0.0,
      "0.7-r100": 0.011442956860051368
    },
    "movement": {
      "enabled_queries": 125,
      "enabled_query_rate": 0.014303696075065797,
      "enabled_non_noop_queries": 125,
      "action_distribution": {
        "A0_noop_B2": 8614,
        "A1_MAVR_soft_rerank": 0,
        "A2_PREM_PRVR_MIL_promote": 20,
        "A3_topK_preserving_promote": 0,
        "A4_raw200_video_insertion": 0,
        "A5_BMN_proposal_confidence_boost": 105
      },
      "video_slot_drift_rate": 0.00018766449250486326,
      "positive_video_entries": 0,
      "positive_video_exits": 0,
      "wrong_video_to_correct_video": 1,
      "correct_video_to_wrong_video": 0,
      "hard_positive_top100_query_entries": 1,
      "hard_positive_top100_query_exits": 0,
      "hard_positive_exit_ratio": 0.0,
      "invalid_span_count": 0,
      "duplicate_span_count_after_nms": 0,
      "R5_loss_queries_07": 2,
      "R10_loss_queries_07": 1,
      "R1_gain_queries_07": 5,
      "R1_loss_queries_07": 4
    },
    "video_metrics": {
      "GT_video_R@1": 25.677995193958118,
      "GT_video_R@100_entry_proxy": 0.011442956860052637
    },
    "feasible": false,
    "no_op_equivalent": false,
    "selection_score": 1.0853358507838458,
    "official_val_used": false
  },
  "oracle_upper_bound_train_calib_diagnostic_only": {
    "config": {
      "diagnostic_only": true,
      "uses_labels": true,
      "promotion_allowed": false
    },
    "metrics": {
      "0.5-r1": 47.12209634969676,
      "0.5-r5": 66.98706945874814,
      "0.5-r10": 73.0861654651562,
      "0.5-r100": 78.38425449136056,
      "0.7-r1": 34.76370294083991,
      "0.7-r5": 52.21421215242019,
      "0.7-r10": 60.235724911317085,
      "0.7-r100": 72.2165007437922
    },
    "delta_vs_C6_B1": {
      "0.5-r1": 5.813022084906741,
      "0.5-r5": 1.659228744707633,
      "0.5-r10": 0.7781210664835783,
      "0.5-r100": 0.011442956860051368,
      "0.7-r1": 9.463325323263533,
      "0.7-r5": 2.3229202425906905,
      "0.7-r10": 1.1786245565854259,
      "0.7-r100": 0.24030209406110714
    },
    "delta_vs_C6_B2": {
      "0.5-r1": 5.561277033985583,
      "0.5-r5": 1.5905710035473106,
      "0.5-r10": 0.7895640233436296,
      "0.5-r100": 0.04577182744020547,
      "0.7-r1": 9.097150703741846,
      "0.7-r5": 2.2657054582904266,
      "0.7-r10": 1.155738642865316,
      "0.7-r100": 0.22885913720105577
    },
    "movement": {
      "enabled_queries": 1151,
      "enabled_query_rate": 0.13170843345920585,
      "enabled_non_noop_queries": 1151,
      "action_distribution": {
        "A0_noop_B2": 7588,
        "A1_MAVR_soft_rerank": 0,
        "A2_PREM_PRVR_MIL_promote": 117,
        "A3_topK_preserving_promote": 0,
        "A4_raw200_video_insertion": 0,
        "A5_BMN_proposal_confidence_boost": 1034
      },
      "video_slot_drift_rate": 0.001392607849868406,
      "positive_video_entries": 0,
      "positive_video_exits": 0,
      "wrong_video_to_correct_video": 117,
      "correct_video_to_wrong_video": 0,
      "hard_positive_top100_query_entries": 4,
      "hard_positive_top100_query_exits": 0,
      "hard_positive_exit_ratio": 0.0,
      "invalid_span_count": 0,
      "duplicate_span_count_after_nms": 950,
      "R5_loss_queries_07": 0,
      "R10_loss_queries_07": 0,
      "R1_gain_queries_07": 795,
      "R1_loss_queries_07": 0
    },
    "video_metrics": {
      "GT_video_R@1": 27.005378189724226,
      "GT_video_R@100_entry_proxy": 0.04577182744021055
    },
    "feasible": false,
    "no_op_equivalent": false,
    "selection_score": 65.95916901247283,
    "official_val_used": false
  }
}
```
