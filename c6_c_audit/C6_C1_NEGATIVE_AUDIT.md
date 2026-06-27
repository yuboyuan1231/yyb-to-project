# C6-C1 negative audit

```json
{
  "status": "C6_C1_R1_TRADEOFF",
  "official_val_used": false,
  "official_val_authorized": false,
  "post_val_adjustment": false,
  "primary_safety_anchor": "C6-B1-lite c6b1r1_0057",
  "current_promoted_report_anchor": "C6-B2 pairwise_main_0005",
  "C6_B2_role": "evidence_only_not_video_anchor",
  "best_config": {
    "config": {
      "alpha": 0.85,
      "beta": 0.2,
      "gamma": 0.2,
      "rho": 0.05,
      "gate_threshold": 0.08673583984375,
      "max_video_slot_changes": 1,
      "lambda_span": 0.15,
      "video_weight": 1.0,
      "span_anchor": "B2"
    },
    "metrics": {
      "0.5-r1": 39.82148987298318,
      "0.5-r5": 48.930083533585076,
      "0.5-r10": 59.13720105275203,
      "0.5-r100": 76.29019338597094,
      "0.7-r1": 27.051150017164435,
      "0.7-r5": 41.652362970591604,
      "0.7-r10": 50.93260098409429,
      "0.7-r100": 65.5223709806614
    },
    "video_metrics": {
      "GT_video_R@1": 48.621123698363654,
      "GT_video_R@5": 51.75649387801808,
      "GT_video_R@10": 62.03226913834535,
      "GT_video_R@100": 80.37532898500973
    },
    "movement": {
      "positive_video_entries": 0,
      "positive_video_exits": 0,
      "hard_positive_top100_query_exits": 0,
      "hard_positive_top100_query_entries": 0,
      "hard_positive_exit_ratio": 0.0,
      "video_slot_drift_rate": 0.2970784225378833,
      "video_multiset_drift_rate": 0.2970784225378833,
      "invalid_span_count": 0,
      "duplicate_span_count_after_nms": 0
    },
    "official_val_used": false,
    "span_anchor": "B2",
    "config_id": "mavr_pr_mil_0061",
    "delta_vs_C6_B1_lite": {
      "0.5-r1": 0.24030209406110714,
      "0.5-r5": -13.502689094862113,
      "0.5-r10": -10.470305526948167,
      "0.5-r100": -0.022885913720102735,
      "0.7-r1": 0.38906053324178913,
      "0.7-r5": -7.197619864973106,
      "0.7-r10": -5.504062249685319,
      "0.7-r100": 0.05721478430027105
    },
    "delta_vs_C6_B2": {
      "0.5-r1": 0.0,
      "0.5-r5": -13.491246138002062,
      "0.5-r10": -10.481748483808218,
      "0.5-r100": 0.0,
      "0.7-r1": 0.0,
      "0.7-r5": -7.243391692413319,
      "0.7-r10": -5.561277033985576,
      "0.7-r100": 0.0
    },
    "selection_score": 6.5373934850030295,
    "feasible": false
  },
  "artifacts": {
    "start_state": {
      "path": "c6_c_audit/C6_C_START_STATE.json",
      "exists": true,
      "size": 1589,
      "sha256": "4f28509e66af44cb0a496fa669034d6fe4abd00223f9f4522e8893c94344899b"
    },
    "c0_oracle": {
      "path": "c6_c_audit/C6_C0_VIDEO_RESIDUAL_ORACLE.json",
      "exists": true,
      "size": 132311,
      "sha256": "06d6dcd3ac06ec8f22281f35b9f3b390b8f997e736f216542f6c1e9a36b6c14b"
    },
    "feature_audit": {
      "path": "c6_c_audit/C6_C1_FEATURE_AUDIT.json",
      "exists": true,
      "size": 739,
      "sha256": "cb7d6b1f156bbc9bf878b3e2cbcea7f3b089d258020c4cbc0fa54406850c6870"
    },
    "data_audit": {
      "path": "c6_c_audit/C6_C1_DATA_AUDIT.json",
      "exists": true,
      "size": 2380,
      "sha256": "1f33261c470b1570d1cf6669844c3eaf200a7ab59c326ac2c20d0f4b0c649d63"
    },
    "arm_comparison": {
      "path": "c6_c_audit/C6_C1_ARM_COMPARISON.json",
      "exists": true,
      "size": 335209,
      "sha256": "ec7dcffde9146b297fadc8efddd468ee4586b3e2f92795ddad9a40389afb9e0f"
    },
    "training_history": {
      "path": "results/rlem_c6_c/arms/mavr_pr_mil/history.json",
      "exists": true,
      "size": 705,
      "sha256": "010f8c3cfc89e8fad85ac21ef9adba3b1c88e3627676876663a8d3e378d28cad"
    },
    "model": {
      "path": "results/rlem_c6_c/arms/mavr_pr_mil/model_best.pt",
      "exists": true,
      "size": 3306859,
      "sha256": "8a43ac5889c7a2282689d50775dbc86c8db7311306172a6a56d4df8acf50b960"
    },
    "best_config": {
      "path": "results/rlem_c6_c/best_config.json",
      "exists": true,
      "size": 1810,
      "sha256": "1448d3a289b898636f6c3114911aa3766c79651cb50f2ab674c691f53536d3c8"
    },
    "gen_span_note": {
      "path": "c6_c_audit/C6_D_GENSPAN_LITE_DESIGN_NOTE.md",
      "exists": true,
      "size": 1351,
      "sha256": "6c4773aa6a328e4ae4a67c7c30f7615206454d31252a92435903dae03c7831c9"
    }
  }
}
```
