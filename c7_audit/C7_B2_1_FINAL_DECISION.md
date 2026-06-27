# C7-B2.1 final decision

- Status: `C7_B2_1_R1_PRESERVED_COVERAGE_REPAIRED`
- Official val used: `false`
- Best: `A4_video_residual_only`

## Best metrics

```json
{
  "name": "A4_video_residual_only",
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
    "video_slot_drift_rate": 0.0,
    "invalid_span_count": 0,
    "duplicate_span_count_after_nms": 0
  },
  "rank_gain_loss_vs_C7_B1_frozen": {
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
  },
  "top100_lost_query_count": 0,
  "top100_lost_query_hash": "4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945",
  "delta_vs_C6_B1": {
    "0.5-r1": 11.271312507151848,
    "0.5-r5": 10.08124499370637,
    "0.5-r10": 7.083190296372578,
    "0.5-r100": 2.048289277949422,
    "0.7-r1": 4.371209520540106,
    "0.7-r5": 9.70362741732464,
    "0.7-r10": 9.989701338825952,
    "0.7-r100": 6.533928367090056
  },
  "delta_vs_C6_B2": {
    "0.5-r1": 11.03101041309074,
    "0.5-r5": 10.092687950566422,
    "0.5-r10": 7.071747339512527,
    "0.5-r100": 2.071175191669525,
    "0.7-r1": 3.9821489872983165,
    "0.7-r5": 9.657855589884427,
    "0.7-r10": 9.932486554525696,
    "0.7-r100": 6.476713582789785
  },
  "delta_vs_C7_B1_frozen": {
    "0.5-r1": 7.94141206087653,
    "0.5-r5": 7.209062821833157,
    "0.5-r10": 4.428424304840377,
    "0.5-r100": 0.0,
    "0.7-r1": 4.668726398901473,
    "0.7-r5": 7.174733951253003,
    "0.7-r10": 5.790136171186639,
    "0.7-r100": 0.0
  },
  "status_label": "C7_B2_1_R1_PRESERVED_COVERAGE_REPAIRED"
}
```

Request human review before any official-val one-shot.
