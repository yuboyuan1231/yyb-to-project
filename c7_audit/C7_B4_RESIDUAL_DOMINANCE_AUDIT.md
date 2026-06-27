# C7-B4 residual dominance audit

```json
{
  "status": "C7_B4_RESIDUAL_DOMINANCE_AUDIT_COMPLETE",
  "official_val_used": false,
  "hard_constraints": {
    "corr_final_anchor_gt_corr_final_residual": true,
    "top1_changed_rate_lte": 0.2,
    "top5_set_changed_rate_lte": 0.35,
    "top10_set_changed_rate_lte": 0.5,
    "R100_unchanged": true,
    "hard_exits": 0,
    "invalid_duplicate": 0
  },
  "candidate_count": 36,
  "hard_constraint_pass_count": 0,
  "candidates": [
    {
      "config": {
        "alpha": 0.02,
        "normalization": "none",
        "clip": 0.5
      },
      "metrics": {
        "0.5-r1": 0.4287245444801715,
        "0.5-r5": 23.713826366559484,
        "0.5-r10": 37.433011789924976,
        "0.5-r100": 49.59807073954984,
        "0.7-r1": 0.02679528403001072,
        "0.7-r5": 14.871382636655948,
        "0.7-r10": 26.178992497320472,
        "0.7-r100": 41.88102893890675
      },
      "delta_vs_C7_B2_1_A4_stress": {
        "0.5-r1": -7.609860664523044,
        "0.5-r5": -15.2465166130761,
        "0.5-r10": -9.271168274383704,
        "0.5-r100": 0.0,
        "0.7-r1": -4.635584137191854,
        "0.7-r5": -11.468381564844588,
        "0.7-r10": -8.654876741693464,
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
        "corr_final_anchor": 0.9999956022178944,
        "corr_final_residual": -0.28905173521748123,
        "top1_changed_rate": 0.13344051446945338,
        "top5_set_changed_rate": 0.4769560557341908,
        "top10_set_changed_rate": 0.7068595927116827,
        "rank_displacement_p50": 4.0,
        "rank_displacement_p90": 15.0,
        "rank_displacement_p99": 34.0
      },
      "hard_constraints_pass": false
    },
    {
      "config": {
        "alpha": 0.02,
        "normalization": "none",
        "clip": 1.0
      },
      "metrics": {
        "0.5-r1": 0.4287245444801715,
        "0.5-r5": 23.713826366559484,
        "0.5-r10": 37.433011789924976,
        "0.5-r100": 49.59807073954984,
        "0.7-r1": 0.02679528403001072,
        "0.7-r5": 14.871382636655948,
        "0.7-r10": 26.178992497320472,
        "0.7-r100": 41.88102893890675
      },
      "delta_vs_C7_B2_1_A4_stress": {
        "0.5-r1": -7.609860664523044,
        "0.5-r5": -15.2465166130761,
        "0.5-r10": -9.271168274383704,
        "0.5-r100": 0.0,
        "0.7-r1": -4.635584137191854,
        "0.7-r5": -11.468381564844588,
        "0.7-r10": -8.654876741693464,
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
        "corr_final_anchor": 0.9999956022178944,
        "corr_final_residual": -0.28905173521748123,
        "top1_changed_rate": 0.13344051446945338,
        "top5_set_changed_rate": 0.4769560557341908,
        "top10_set_changed_rate": 0.7068595927116827,
        "rank_displacement_p50": 4.0,
        "rank_displacement_p90": 15.0,
        "rank_displacement_p99": 34.0
      },
      "hard_constraints_pass": false
    },
    {
      "config": {
        "alpha": 0.02,
        "normalization": "none",
        "clip": 2.0
      },
      "metrics": {
        "0.5-r1": 0.4287245444801715,
        "0.5-r5": 23.713826366559484,
        "0.5-r10": 37.433011789924976,
        "0.5-r100": 49.59807073954984,
        "0.7-r1": 0.02679528403001072,
        "0.7-r5": 14.871382636655948,
        "0.7-r10": 26.178992497320472,
        "0.7-r100": 41.88102893890675
      },
      "delta_vs_C7_B2_1_A4_stress": {
        "0.5-r1": -7.609860664523044,
        "0.5-r5": -15.2465166130761,
        "0.5-r10": -9.271168274383704,
        "0.5-r100": 0.0,
        "0.7-r1": -4.635584137191854,
        "0.7-r5": -11.468381564844588,
        "0.7-r10": -8.654876741693464,
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
        "corr_final_anchor": 0.9999956022178944,
        "corr_final_residual": -0.28905173521748123,
        "top1_changed_rate": 0.13344051446945338,
        "top5_set_changed_rate": 0.4769560557341908,
        "top10_set_changed_rate": 0.7068595927116827,
        "rank_displacement_p50": 4.0,
        "rank_displacement_p90": 15.0,
        "rank_displacement_p99": 34.0
      },
      "hard_constraints_pass": false
    },
    {
      "config": {
        "alpha": 0.02,
        "normalization": "rank_norm",
        "clip": 0.5
      },
      "metrics": {
        "0.5-r1": 0.4555198285101822,
        "0.5-r5": 23.740621650589496,
        "0.5-r10": 37.459807073954984,
        "0.5-r100": 49.59807073954984,
        "0.7-r1": 0.05359056806002144,
        "0.7-r5": 14.871382636655948,
        "0.7-r10": 26.15219721329046,
        "0.7-r100": 41.88102893890675
      },
      "delta_vs_C7_B2_1_A4_stress": {
        "0.5-r1": -7.583065380493034,
        "0.5-r5": -15.219721329046088,
        "0.5-r10": -9.244372990353696,
        "0.5-r100": 0.0,
        "0.7-r1": -4.608788853161844,
        "0.7-r5": -11.468381564844588,
        "0.7-r10": -8.681672025723476,
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
        "corr_final_anchor": 0.9999716452820074,
        "corr_final_residual": -0.28751844913761837,
        "top1_changed_rate": 0.13344051446945338,
        "top5_set_changed_rate": 0.4769560557341908,
        "top10_set_changed_rate": 0.7081993569131833,
        "rank_displacement_p50": 4.0,
        "rank_displacement_p90": 15.0,
        "rank_displacement_p99": 34.0
      },
      "hard_constraints_pass": false
    },
    {
      "config": {
        "alpha": 0.02,
        "normalization": "rank_norm",
        "clip": 1.0
      },
      "metrics": {
        "0.5-r1": 0.48231511254019294,
        "0.5-r5": 23.84780278670954,
        "0.5-r10": 37.48660235798499,
        "0.5-r100": 49.59807073954984,
        "0.7-r1": 0.08038585209003216,
        "0.7-r5": 14.95176848874598,
        "0.7-r10": 26.312968917470524,
        "0.7-r100": 41.88102893890675
      },
      "delta_vs_C7_B2_1_A4_stress": {
        "0.5-r1": -7.556270096463023,
        "0.5-r5": -15.112540192926044,
        "0.5-r10": -9.217577706323688,
        "0.5-r100": 0.0,
        "0.7-r1": -4.581993569131833,
        "0.7-r5": -11.387995712754556,
        "0.7-r10": -8.520900321543412,
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
        "corr_final_anchor": 0.9999459952838843,
        "corr_final_residual": -0.28616690546043966,
        "top1_changed_rate": 0.13317256162915328,
        "top5_set_changed_rate": 0.4753483386923901,
        "top10_set_changed_rate": 0.7063236870310825,
        "rank_displacement_p50": 4.0,
        "rank_displacement_p90": 15.0,
        "rank_displacement_p99": 34.0
      },
      "hard_constraints_pass": false
    },
    {
      "config": {
        "alpha": 0.02,
        "normalization": "rank_norm",
        "clip": 2.0
      },
      "metrics": {
        "0.5-r1": 0.48231511254019294,
        "0.5-r5": 23.84780278670954,
        "0.5-r10": 37.48660235798499,
        "0.5-r100": 49.59807073954984,
        "0.7-r1": 0.08038585209003216,
        "0.7-r5": 14.95176848874598,
        "0.7-r10": 26.312968917470524,
        "0.7-r100": 41.88102893890675
      },
      "delta_vs_C7_B2_1_A4_stress": {
        "0.5-r1": -7.556270096463023,
        "0.5-r5": -15.112540192926044,
        "0.5-r10": -9.217577706323688,
        "0.5-r100": 0.0,
        "0.7-r1": -4.581993569131833,
        "0.7-r5": -11.387995712754556,
        "0.7-r10": -8.520900321543412,
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
        "corr_final_anchor": 0.9999459952838843,
        "corr_final_residual": -0.28616690546043966,
        "top1_changed_rate": 0.13317256162915328,
        "top5_set_changed_rate": 0.4753483386923901,
        "top10_set_changed_rate": 0.7063236870310825,
        "rank_displacement_p50": 4.0,
        "rank_displacement_p90": 15.0,
        "rank_displacement_p99": 34.0
      },
      "hard_constraints_pass": false
    },
    {
      "config": {
        "alpha": 0.02,
        "normalization": "bounded_minmax",
        "clip": 0.5
      },
      "metrics": {
        "0.5-r1": 0.4555198285101822,
        "0.5-r5": 23.767416934619508,
        "0.5-r10": 37.459807073954984,
        "0.5-r100": 49.59807073954984,
        "0.7-r1": 0.05359056806002144,
        "0.7-r5": 14.89817792068596,
        "0.7-r10": 26.232583065380492,
        "0.7-r100": 41.88102893890675
      },
      "delta_vs_C7_B2_1_A4_stress": {
        "0.5-r1": -7.583065380493034,
        "0.5-r5": -15.192926045016076,
        "0.5-r10": -9.244372990353696,
        "0.5-r100": 0.0,
        "0.7-r1": -4.608788853161844,
        "0.7-r5": -11.441586280814576,
        "0.7-r10": -8.601286173633444,
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
        "corr_final_anchor": 0.9999667546056675,
        "corr_final_residual": -0.2858284812140727,
        "top1_changed_rate": 0.13317256162915328,
        "top5_set_changed_rate": 0.47481243301178994,
        "top10_set_changed_rate": 0.7023043944265809,
        "rank_displacement_p50": 4.0,
        "rank_displacement_p90": 15.0,
        "rank_displacement_p99": 34.0
      },
      "hard_constraints_pass": false
    },
    {
      "config": {
        "alpha": 0.02,
        "normalization": "bounded_minmax",
        "clip": 1.0
      },
      "metrics": {
        "0.5-r1": 0.48231511254019294,
        "0.5-r5": 23.90139335476956,
        "0.5-r10": 37.540192926045016,
        "0.5-r100": 49.59807073954984,
        "0.7-r1": 0.08038585209003216,
        "0.7-r5": 14.92497320471597,
        "0.7-r10": 26.312968917470524,
        "0.7-r100": 41.88102893890675
      },
      "delta_vs_C7_B2_1_A4_stress": {
        "0.5-r1": -7.556270096463023,
        "0.5-r5": -15.058949624866024,
        "0.5-r10": -9.163987138263664,
        "0.5-r100": 0.0,
        "0.7-r1": -4.581993569131833,
        "0.7-r5": -11.414790996784566,
        "0.7-r10": -8.520900321543412,
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
        "corr_final_anchor": 0.999886585401792,
        "corr_final_residual": -0.28180295836678354,
        "top1_changed_rate": 0.13156484458735263,
        "top5_set_changed_rate": 0.4691854233654877,
        "top10_set_changed_rate": 0.6964094319399786,
        "rank_displacement_p50": 4.0,
        "rank_displacement_p90": 15.0,
        "rank_displacement_p99": 34.0
      },
      "hard_constraints_pass": false
    },
    {
      "config": {
        "alpha": 0.02,
        "normalization": "bounded_minmax",
        "clip": 2.0
      },
      "metrics": {
        "0.5-r1": 0.48231511254019294,
        "0.5-r5": 23.90139335476956,
        "0.5-r10": 37.540192926045016,
        "0.5-r100": 49.59807073954984,
        "0.7-r1": 0.08038585209003216,
        "0.7-r5": 14.92497320471597,
        "0.7-r10": 26.312968917470524,
        "0.7-r100": 41.88102893890675
      },
      "delta_vs_C7_B2_1_A4_stress": {
        "0.5-r1": -7.556270096463023,
        "0.5-r5": -15.058949624866024,
        "0.5-r10": -9.163987138263664,
        "0.5-r100": 0.0,
        "0.7-r1": -4.581993569131833,
        "0.7-r5": -11.414790996784566,
        "0.7-r10": -8.520900321543412,
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
        "corr_final_anchor": 0.999886585401792,
        "corr_final_residual": -0.28180295836678354,
        "top1_changed_rate": 0.13156484458735263,
        "top5_set_changed_rate": 0.4691854233654877,
        "top10_set_changed_rate": 0.6964094319399786,
        "rank_displacement_p50": 4.0,
        "rank_displacement_p90": 15.0,
        "rank_displacement_p99": 34.0
      },
      "hard_constraints_pass": false
    },
    {
      "config": {
        "alpha": 0.05,
        "normalization": "none",
        "clip": 0.5
      },
      "metrics": {
        "0.5-r1": 0.4555198285101822,
        "0.5-r5": 23.874598070739548,
        "0.5-r10": 37.48660235798499,
        "0.5-r100": 49.59807073954984,
        "0.7-r1": 0.05359056806002144,
        "0.7-r5": 14.92497320471597,
        "0.7-r10": 26.286173633440516,
        "0.7-r100": 41.88102893890675
      },
      "delta_vs_C7_B2_1_A4_stress": {
        "0.5-r1": -7.583065380493034,
        "0.5-r5": -15.085744908896036,
        "0.5-r10": -9.217577706323688,
        "0.5-r100": 0.0,
        "0.7-r1": -4.608788853161844,
        "0.7-r5": -11.414790996784566,
        "0.7-r10": -8.54769560557342,
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
        "corr_final_anchor": 0.9999725362666396,
        "corr_final_residual": -0.2872153655917118,
        "top1_changed_rate": 0.13290460878885316,
        "top5_set_changed_rate": 0.4742765273311897,
        "top10_set_changed_rate": 0.7047159699892819,
        "rank_displacement_p50": 4.0,
        "rank_displacement_p90": 15.0,
        "rank_displacement_p99": 34.0
      },
      "hard_constraints_pass": false
    },
    {
      "config": {
        "alpha": 0.05,
        "normalization": "none",
        "clip": 1.0
      },
      "metrics": {
        "0.5-r1": 0.4555198285101822,
        "0.5-r5": 23.874598070739548,
        "0.5-r10": 37.48660235798499,
        "0.5-r100": 49.59807073954984,
        "0.7-r1": 0.05359056806002144,
        "0.7-r5": 14.92497320471597,
        "0.7-r10": 26.286173633440516,
        "0.7-r100": 41.88102893890675
      },
      "delta_vs_C7_B2_1_A4_stress": {
        "0.5-r1": -7.583065380493034,
        "0.5-r5": -15.085744908896036,
        "0.5-r10": -9.217577706323688,
        "0.5-r100": 0.0,
        "0.7-r1": -4.608788853161844,
        "0.7-r5": -11.414790996784566,
        "0.7-r10": -8.54769560557342,
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
        "corr_final_anchor": 0.9999725362666396,
        "corr_final_residual": -0.2872153655917118,
        "top1_changed_rate": 0.13290460878885316,
        "top5_set_changed_rate": 0.4742765273311897,
        "top10_set_changed_rate": 0.7047159699892819,
        "rank_displacement_p50": 4.0,
        "rank_displacement_p90": 15.0,
        "rank_displacement_p99": 34.0
      },
      "hard_constraints_pass": false
    },
    {
      "config": {
        "alpha": 0.05,
        "normalization": "none",
        "clip": 2.0
      },
      "metrics": {
        "0.5-r1": 0.4555198285101822,
        "0.5-r5": 23.874598070739548,
        "0.5-r10": 37.48660235798499,
        "0.5-r100": 49.59807073954984,
        "0.7-r1": 0.05359056806002144,
        "0.7-r5": 14.92497320471597,
        "0.7-r10": 26.286173633440516,
        "0.7-r100": 41.88102893890675
      },
      "delta_vs_C7_B2_1_A4_stress": {
        "0.5-r1": -7.583065380493034,
        "0.5-r5": -15.085744908896036,
        "0.5-r10": -9.217577706323688,
        "0.5-r100": 0.0,
        "0.7-r1": -4.608788853161844,
        "0.7-r5": -11.414790996784566,
        "0.7-r10": -8.54769560557342,
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
        "corr_final_anchor": 0.9999725362666396,
        "corr_final_residual": -0.2872153655917118,
        "top1_changed_rate": 0.13290460878885316,
        "top5_set_changed_rate": 0.4742765273311897,
        "top10_set_changed_rate": 0.7047159699892819,
        "rank_displacement_p50": 4.0,
        "rank_displacement_p90": 15.0,
        "rank_displacement_p99": 34.0
      },
      "hard_constraints_pass": false
    },
    {
      "config": {
        "alpha": 0.05,
        "normalization": "rank_norm",
        "clip": 0.5
      },
      "metrics": {
        "0.5-r1": 0.5091103965702036,
        "0.5-r5": 23.928188638799572,
        "0.5-r10": 37.540192926045016,
        "0.5-r100": 49.59807073954984,
        "0.7-r1": 0.10718113612004287,
        "0.7-r5": 14.95176848874598,
        "0.7-r10": 26.286173633440516,
        "0.7-r100": 41.88102893890675
      },
      "delta_vs_C7_B2_1_A4_stress": {
        "0.5-r1": -7.529474812433012,
        "0.5-r5": -15.032154340836012,
        "0.5-r10": -9.163987138263664,
        "0.5-r100": 0.0,
        "0.7-r1": -4.555198285101822,
        "0.7-r5": -11.387995712754556,
        "0.7-r10": -8.54769560557342,
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
        "corr_final_anchor": 0.999822662909353,
        "corr_final_residual": -0.2833603626500277,
        "top1_changed_rate": 0.13210075026795284,
        "top5_set_changed_rate": 0.47106109324758844,
        "top10_set_changed_rate": 0.7044480171489818,
        "rank_displacement_p50": 4.0,
        "rank_displacement_p90": 15.0,
        "rank_displacement_p99": 34.0
      },
      "hard_constraints_pass": false
    },
    {
      "config": {
        "alpha": 0.05,
        "normalization": "rank_norm",
        "clip": 1.0
      },
      "metrics": {
        "0.5-r1": 0.5627009646302251,
        "0.5-r5": 24.169346195069668,
        "0.5-r10": 37.59378349410504,
        "0.5-r100": 49.59807073954984,
        "0.7-r1": 0.1607717041800643,
        "0.7-r5": 15.139335476956056,
        "0.7-r10": 26.55412647374062,
        "0.7-r100": 41.934619506966776
      },
      "delta_vs_C7_B2_1_A4_stress": {
        "0.5-r1": -7.475884244372991,
        "0.5-r5": -14.790996784565916,
        "0.5-r10": -9.11039657020364,
        "0.5-r100": 0.0,
        "0.7-r1": -4.501607717041801,
        "0.7-r5": -11.20042872454448,
        "0.7-r10": -8.279742765273316,
        "0.7-r100": 0.05359056806002371
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
        "corr_final_anchor": 0.999662051191443,
        "corr_final_residual": -0.2799569316433312,
        "top1_changed_rate": 0.12861736334405144,
        "top5_set_changed_rate": 0.46811361200428725,
        "top10_set_changed_rate": 0.7020364415862809,
        "rank_displacement_p50": 4.0,
        "rank_displacement_p90": 15.0,
        "rank_displacement_p99": 33.0
      },
      "hard_constraints_pass": false
    },
    {
      "config": {
        "alpha": 0.05,
        "normalization": "rank_norm",
        "clip": 2.0
      },
      "metrics": {
        "0.5-r1": 0.5627009646302251,
        "0.5-r5": 24.169346195069668,
        "0.5-r10": 37.59378349410504,
        "0.5-r100": 49.59807073954984,
        "0.7-r1": 0.1607717041800643,
        "0.7-r5": 15.139335476956056,
        "0.7-r10": 26.55412647374062,
        "0.7-r100": 41.934619506966776
      },
      "delta_vs_C7_B2_1_A4_stress": {
        "0.5-r1": -7.475884244372991,
        "0.5-r5": -14.790996784565916,
        "0.5-r10": -9.11039657020364,
        "0.5-r100": 0.0,
        "0.7-r1": -4.501607717041801,
        "0.7-r5": -11.20042872454448,
        "0.7-r10": -8.279742765273316,
        "0.7-r100": 0.05359056806002371
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
        "corr_final_anchor": 0.999662051191443,
        "corr_final_residual": -0.2799569316433312,
        "top1_changed_rate": 0.12861736334405144,
        "top5_set_changed_rate": 0.46811361200428725,
        "top10_set_changed_rate": 0.7020364415862809,
        "rank_displacement_p50": 4.0,
        "rank_displacement_p90": 15.0,
        "rank_displacement_p99": 33.0
      },
      "hard_constraints_pass": false
    },
    {
      "config": {
        "alpha": 0.05,
        "normalization": "bounded_minmax",
        "clip": 0.5
      },
      "metrics": {
        "0.5-r1": 0.5091103965702036,
        "0.5-r5": 23.981779206859592,
        "0.5-r10": 37.540192926045016,
        "0.5-r100": 49.59807073954984,
        "0.7-r1": 0.10718113612004287,
        "0.7-r5": 14.95176848874598,
        "0.7-r10": 26.339764201500536,
        "0.7-r100": 41.88102893890675
      },
      "delta_vs_C7_B2_1_A4_stress": {
        "0.5-r1": -7.529474812433012,
        "0.5-r5": -14.978563772775992,
        "0.5-r10": -9.163987138263664,
        "0.5-r100": 0.0,
        "0.7-r1": -4.555198285101822,
        "0.7-r5": -11.387995712754556,
        "0.7-r10": -8.4941050375134,
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
        "corr_final_anchor": 0.9997921050757178,
        "corr_final_residual": -0.2791339370087791,
        "top1_changed_rate": 0.13183279742765272,
        "top5_set_changed_rate": 0.4654340836012862,
        "top10_set_changed_rate": 0.6934619506966774,
        "rank_displacement_p50": 4.0,
        "rank_displacement_p90": 15.0,
        "rank_displacement_p99": 33.0
      },
      "hard_constraints_pass": false
    },
    {
      "config": {
        "alpha": 0.05,
        "normalization": "bounded_minmax",
        "clip": 1.0
      },
      "metrics": {
        "0.5-r1": 0.5894962486602358,
        "0.5-r5": 24.437299035369776,
        "0.5-r10": 37.59378349410504,
        "0.5-r100": 49.59807073954984,
        "0.7-r1": 0.18756698821007503,
        "0.7-r5": 15.30010718113612,
        "0.7-r10": 26.527331189710612,
        "0.7-r100": 41.88102893890675
      },
      "delta_vs_C7_B2_1_A4_stress": {
        "0.5-r1": -7.44908896034298,
        "0.5-r5": -14.523043944265808,
        "0.5-r10": -9.11039657020364,
        "0.5-r100": 0.0,
        "0.7-r1": -4.47481243301179,
        "0.7-r5": -11.039657020364416,
        "0.7-r10": -8.306538049303324,
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
        "corr_final_anchor": 0.9992897390135734,
        "corr_final_residual": -0.2690016213710984,
        "top1_changed_rate": 0.12593783494105038,
        "top5_set_changed_rate": 0.4445337620578778,
        "top10_set_changed_rate": 0.6803322615219721,
        "rank_displacement_p50": 4.0,
        "rank_displacement_p90": 14.0,
        "rank_displacement_p99": 32.0
      },
      "hard_constraints_pass": false
    },
    {
      "config": {
        "alpha": 0.05,
        "normalization": "bounded_minmax",
        "clip": 2.0
      },
      "metrics": {
        "0.5-r1": 0.5894962486602358,
        "0.5-r5": 24.437299035369776,
        "0.5-r10": 37.59378349410504,
        "0.5-r100": 49.59807073954984,
        "0.7-r1": 0.18756698821007503,
        "0.7-r5": 15.30010718113612,
        "0.7-r10": 26.527331189710612,
        "0.7-r100": 41.88102893890675
      },
      "delta_vs_C7_B2_1_A4_stress": {
        "0.5-r1": -7.44908896034298,
        "0.5-r5": -14.523043944265808,
        "0.5-r10": -9.11039657020364,
        "0.5-r100": 0.0,
        "0.7-r1": -4.47481243301179,
        "0.7-r5": -11.039657020364416,
        "0.7-r10": -8.306538049303324,
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
        "corr_final_anchor": 0.9992897390135734,
        "corr_final_residual": -0.2690016213710984,
        "top1_changed_rate": 0.12593783494105038,
        "top5_set_changed_rate": 0.4445337620578778,
        "top10_set_changed_rate": 0.6803322615219721,
        "rank_displacement_p50": 4.0,
        "rank_displacement_p90": 14.0,
        "rank_displacement_p99": 32.0
      },
      "hard_constraints_pass": false
    },
    {
      "config": {
        "alpha": 0.1,
        "normalization": "none",
        "clip": 0.5
      },
      "metrics": {
        "0.5-r1": 0.5359056806002144,
        "0.5-r5": 24.222936763129688,
        "0.5-r10": 37.540192926045016,
        "0.5-r100": 49.59807073954984,
        "0.7-r1": 0.1339764201500536,
        "0.7-r5": 15.058949624866024,
        "0.7-r10": 26.366559485530548,
        "0.7-r100": 41.88102893890675
      },
      "delta_vs_C7_B2_1_A4_stress": {
        "0.5-r1": -7.502679528403002,
        "0.5-r5": -14.737406216505896,
        "0.5-r10": -9.163987138263664,
        "0.5-r100": 0.0,
        "0.7-r1": -4.528403001071811,
        "0.7-r5": -11.280814576634512,
        "0.7-r10": -8.467309753483388,
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
        "corr_final_anchor": 0.9998901407582493,
        "corr_final_residual": -0.2841442559311205,
        "top1_changed_rate": 0.13156484458735263,
        "top5_set_changed_rate": 0.4683815648445874,
        "top10_set_changed_rate": 0.7001607717041801,
        "rank_displacement_p50": 4.0,
        "rank_displacement_p90": 15.0,
        "rank_displacement_p99": 33.0
      },
      "hard_constraints_pass": false
    },
    {
      "config": {
        "alpha": 0.1,
        "normalization": "none",
        "clip": 1.0
      },
      "metrics": {
        "0.5-r1": 0.5359056806002144,
        "0.5-r5": 24.222936763129688,
        "0.5-r10": 37.540192926045016,
        "0.5-r100": 49.59807073954984,
        "0.7-r1": 0.1339764201500536,
        "0.7-r5": 15.058949624866024,
        "0.7-r10": 26.366559485530548,
        "0.7-r100": 41.88102893890675
      },
      "delta_vs_C7_B2_1_A4_stress": {
        "0.5-r1": -7.502679528403002,
        "0.5-r5": -14.737406216505896,
        "0.5-r10": -9.163987138263664,
        "0.5-r100": 0.0,
        "0.7-r1": -4.528403001071811,
        "0.7-r5": -11.280814576634512,
        "0.7-r10": -8.467309753483388,
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
        "corr_final_anchor": 0.9998901407582493,
        "corr_final_residual": -0.2841442559311205,
        "top1_changed_rate": 0.13156484458735263,
        "top5_set_changed_rate": 0.4683815648445874,
        "top10_set_changed_rate": 0.7001607717041801,
        "rank_displacement_p50": 4.0,
        "rank_displacement_p90": 15.0,
        "rank_displacement_p99": 33.0
      },
      "hard_constraints_pass": false
    },
    {
      "config": {
        "alpha": 0.1,
        "normalization": "none",
        "clip": 2.0
      },
      "metrics": {
        "0.5-r1": 0.5359056806002144,
        "0.5-r5": 24.222936763129688,
        "0.5-r10": 37.540192926045016,
        "0.5-r100": 49.59807073954984,
        "0.7-r1": 0.1339764201500536,
        "0.7-r5": 15.058949624866024,
        "0.7-r10": 26.366559485530548,
        "0.7-r100": 41.88102893890675
      },
      "delta_vs_C7_B2_1_A4_stress": {
        "0.5-r1": -7.502679528403002,
        "0.5-r5": -14.737406216505896,
        "0.5-r10": -9.163987138263664,
        "0.5-r100": 0.0,
        "0.7-r1": -4.528403001071811,
        "0.7-r5": -11.280814576634512,
        "0.7-r10": -8.467309753483388,
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
        "corr_final_anchor": 0.9998901407582493,
        "corr_final_residual": -0.2841442559311205,
        "top1_changed_rate": 0.13156484458735263,
        "top5_set_changed_rate": 0.4683815648445874,
        "top10_set_changed_rate": 0.7001607717041801,
        "rank_displacement_p50": 4.0,
        "rank_displacement_p90": 15.0,
        "rank_displacement_p99": 33.0
      },
      "hard_constraints_pass": false
    },
    {
      "config": {
        "alpha": 0.1,
        "normalization": "rank_norm",
        "clip": 0.5
      },
      "metrics": {
        "0.5-r1": 0.6162915326902465,
        "0.5-r5": 24.276527331189712,
        "0.5-r10": 37.59378349410504,
        "0.5-r100": 49.59807073954984,
        "0.7-r1": 0.18756698821007503,
        "0.7-r5": 15.139335476956056,
        "0.7-r10": 26.420150053590568,
        "0.7-r100": 41.88102893890675
      },
      "delta_vs_C7_B2_1_A4_stress": {
        "0.5-r1": -7.42229367631297,
        "0.5-r5": -14.683815648445872,
        "0.5-r10": -9.11039657020364,
        "0.5-r100": 0.0,
        "0.7-r1": -4.47481243301179,
        "0.7-r5": -11.20042872454448,
        "0.7-r10": -8.413719185423368,
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
        "corr_final_anchor": 0.999289599803635,
        "corr_final_residual": -0.2763694559740928,
        "top1_changed_rate": 0.1267416934619507,
        "top5_set_changed_rate": 0.46489817792068594,
        "top10_set_changed_rate": 0.7033762057877814,
        "rank_displacement_p50": 4.0,
        "rank_displacement_p90": 14.0,
        "rank_displacement_p99": 33.0
      },
      "hard_constraints_pass": false
    },
    {
      "config": {
        "alpha": 0.1,
        "normalization": "rank_norm",
        "clip": 1.0
      },
      "metrics": {
        "0.5-r1": 0.8306538049303323,
        "0.5-r5": 24.624866023579848,
        "0.5-r10": 37.72775991425509,
        "0.5-r100": 49.59807073954984,
        "0.7-r1": 0.3483386923901393,
        "0.7-r5": 15.64844587352626,
        "0.7-r10": 26.741693461950696,
        "0.7-r100": 41.90782422293676
      },
      "delta_vs_C7_B2_1_A4_stress": {
        "0.5-r1": -7.207931404072884,
        "0.5-r5": -14.335476956055736,
        "0.5-r10": -8.976420150053592,
        "0.5-r100": 0.0,
        "0.7-r1": -4.314040728831726,
        "0.7-r5": -10.691318327974276,
        "0.7-r10": -8.09217577706324,
        "0.7-r100": 0.026795284030008304
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
        "corr_final_anchor": 0.9986458985451456,
        "corr_final_residual": -0.26948963204135,
        "top1_changed_rate": 0.11870310825294748,
        "top5_set_changed_rate": 0.4552518756698821,
        "top10_set_changed_rate": 0.687566988210075,
        "rank_displacement_p50": 3.0,
        "rank_displacement_p90": 14.0,
        "rank_displacement_p99": 32.0
      },
      "hard_constraints_pass": false
    },
    {
      "config": {
        "alpha": 0.1,
        "normalization": "rank_norm",
        "clip": 2.0
      },
      "metrics": {
        "0.5-r1": 0.8306538049303323,
        "0.5-r5": 24.624866023579848,
        "0.5-r10": 37.72775991425509,
        "0.5-r100": 49.59807073954984,
        "0.7-r1": 0.3483386923901393,
        "0.7-r5": 15.64844587352626,
        "0.7-r10": 26.741693461950696,
        "0.7-r100": 41.90782422293676
      },
      "delta_vs_C7_B2_1_A4_stress": {
        "0.5-r1": -7.207931404072884,
        "0.5-r5": -14.335476956055736,
        "0.5-r10": -8.976420150053592,
        "0.5-r100": 0.0,
        "0.7-r1": -4.314040728831726,
        "0.7-r5": -10.691318327974276,
        "0.7-r10": -8.09217577706324,
        "0.7-r100": 0.026795284030008304
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
        "corr_final_anchor": 0.9986458985451456,
        "corr_final_residual": -0.26948963204135,
        "top1_changed_rate": 0.11870310825294748,
        "top5_set_changed_rate": 0.4552518756698821,
        "top10_set_changed_rate": 0.687566988210075,
        "rank_displacement_p50": 3.0,
        "rank_displacement_p90": 14.0,
        "rank_displacement_p99": 32.0
      },
      "hard_constraints_pass": false
    },
    {
      "config": {
        "alpha": 0.1,
        "normalization": "bounded_minmax",
        "clip": 0.5
      },
      "metrics": {
        "0.5-r1": 0.6162915326902465,
        "0.5-r5": 24.437299035369776,
        "0.5-r10": 37.62057877813505,
        "0.5-r100": 49.59807073954984,
        "0.7-r1": 0.18756698821007503,
        "0.7-r5": 15.32690246516613,
        "0.7-r10": 26.580921757770632,
        "0.7-r100": 41.88102893890675
      },
      "delta_vs_C7_B2_1_A4_stress": {
        "0.5-r1": -7.42229367631297,
        "0.5-r5": -14.523043944265808,
        "0.5-r10": -9.083601286173632,
        "0.5-r100": 0.0,
        "0.7-r1": -4.47481243301179,
        "0.7-r5": -11.012861736334406,
        "0.7-r10": -8.252947481243304,
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
        "corr_final_anchor": 0.9991673528337583,
        "corr_final_residual": -0.26791325147955347,
        "top1_changed_rate": 0.1264737406216506,
        "top5_set_changed_rate": 0.4437299035369775,
        "top10_set_changed_rate": 0.6771168274383709,
        "rank_displacement_p50": 3.0,
        "rank_displacement_p90": 14.0,
        "rank_displacement_p99": 32.0
      },
      "hard_constraints_pass": false
    },
    {
      "config": {
        "alpha": 0.1,
        "normalization": "bounded_minmax",
        "clip": 1.0
      },
      "metrics": {
        "0.5-r1": 1.045016077170418,
        "0.5-r5": 25.267952840300108,
        "0.5-r10": 37.78135048231511,
        "0.5-r100": 49.59807073954984,
        "0.7-r1": 0.4555198285101822,
        "0.7-r5": 15.889603429796356,
        "0.7-r10": 26.956055734190784,
        "0.7-r100": 41.88102893890675
      },
      "delta_vs_C7_B2_1_A4_stress": {
        "0.5-r1": -6.993569131832798,
        "0.5-r5": -13.692390139335476,
        "0.5-r10": -8.922829581993568,
        "0.5-r100": 0.0,
        "0.7-r1": -4.206859592711683,
        "0.7-r5": -10.45016077170418,
        "0.7-r10": -7.877813504823152,
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
        "corr_final_anchor": 0.9971547665469969,
        "corr_final_residual": -0.24746381517363056,
        "top1_changed_rate": 0.11039657020364416,
        "top5_set_changed_rate": 0.39308681672025725,
        "top10_set_changed_rate": 0.6162915326902465,
        "rank_displacement_p50": 3.0,
        "rank_displacement_p90": 13.0,
        "rank_displacement_p99": 31.0
      },
      "hard_constraints_pass": false
    },
    {
      "config": {
        "alpha": 0.1,
        "normalization": "bounded_minmax",
        "clip": 2.0
      },
      "metrics": {
        "0.5-r1": 1.045016077170418,
        "0.5-r5": 25.267952840300108,
        "0.5-r10": 37.78135048231511,
        "0.5-r100": 49.59807073954984,
        "0.7-r1": 0.4555198285101822,
        "0.7-r5": 15.889603429796356,
        "0.7-r10": 26.956055734190784,
        "0.7-r100": 41.88102893890675
      },
      "delta_vs_C7_B2_1_A4_stress": {
        "0.5-r1": -6.993569131832798,
        "0.5-r5": -13.692390139335476,
        "0.5-r10": -8.922829581993568,
        "0.5-r100": 0.0,
        "0.7-r1": -4.206859592711683,
        "0.7-r5": -10.45016077170418,
        "0.7-r10": -7.877813504823152,
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
        "corr_final_anchor": 0.9971547665469969,
        "corr_final_residual": -0.24746381517363056,
        "top1_changed_rate": 0.11039657020364416,
        "top5_set_changed_rate": 0.39308681672025725,
        "top10_set_changed_rate": 0.6162915326902465,
        "rank_displacement_p50": 3.0,
        "rank_displacement_p90": 13.0,
        "rank_displacement_p99": 31.0
      },
      "hard_constraints_pass": false
    },
    {
      "config": {
        "alpha": 0.2,
        "normalization": "none",
        "clip": 0.5
      },
      "metrics": {
        "0.5-r1": 0.6966773847802786,
        "0.5-r5": 24.571275455519828,
        "0.5-r10": 37.67416934619507,
        "0.5-r100": 49.59807073954984,
        "0.7-r1": 0.24115755627009647,
        "0.7-r5": 15.541264737406216,
        "0.7-r10": 26.607717041800644,
        "0.7-r100": 41.88102893890675
      },
      "delta_vs_C7_B2_1_A4_stress": {
        "0.5-r1": -7.341907824222937,
        "0.5-r5": -14.389067524115756,
        "0.5-r10": -9.030010718113608,
        "0.5-r100": 0.0,
        "0.7-r1": -4.421221864951769,
        "0.7-r5": -10.79849946409432,
        "0.7-r10": -8.226152197213292,
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
        "corr_final_anchor": 0.9995605002443235,
        "corr_final_residual": -0.2779653643371079,
        "top1_changed_rate": 0.12781350482315113,
        "top5_set_changed_rate": 0.4590032154340836,
        "top10_set_changed_rate": 0.6891747052518756,
        "rank_displacement_p50": 4.0,
        "rank_displacement_p90": 14.0,
        "rank_displacement_p99": 32.0
      },
      "hard_constraints_pass": false
    },
    {
      "config": {
        "alpha": 0.2,
        "normalization": "none",
        "clip": 1.0
      },
      "metrics": {
        "0.5-r1": 0.6966773847802786,
        "0.5-r5": 24.571275455519828,
        "0.5-r10": 37.67416934619507,
        "0.5-r100": 49.59807073954984,
        "0.7-r1": 0.24115755627009647,
        "0.7-r5": 15.541264737406216,
        "0.7-r10": 26.607717041800644,
        "0.7-r100": 41.88102893890675
      },
      "delta_vs_C7_B2_1_A4_stress": {
        "0.5-r1": -7.341907824222937,
        "0.5-r5": -14.389067524115756,
        "0.5-r10": -9.030010718113608,
        "0.5-r100": 0.0,
        "0.7-r1": -4.421221864951769,
        "0.7-r5": -10.79849946409432,
        "0.7-r10": -8.226152197213292,
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
        "corr_final_anchor": 0.9995605002443235,
        "corr_final_residual": -0.2779653643371079,
        "top1_changed_rate": 0.12781350482315113,
        "top5_set_changed_rate": 0.4590032154340836,
        "top10_set_changed_rate": 0.6891747052518756,
        "rank_displacement_p50": 4.0,
        "rank_displacement_p90": 14.0,
        "rank_displacement_p99": 32.0
      },
      "hard_constraints_pass": false
    },
    {
      "config": {
        "alpha": 0.2,
        "normalization": "none",
        "clip": 2.0
      },
      "metrics": {
        "0.5-r1": 0.6966773847802786,
        "0.5-r5": 24.571275455519828,
        "0.5-r10": 37.67416934619507,
        "0.5-r100": 49.59807073954984,
        "0.7-r1": 0.24115755627009647,
        "0.7-r5": 15.541264737406216,
        "0.7-r10": 26.607717041800644,
        "0.7-r100": 41.88102893890675
      },
      "delta_vs_C7_B2_1_A4_stress": {
        "0.5-r1": -7.341907824222937,
        "0.5-r5": -14.389067524115756,
        "0.5-r10": -9.030010718113608,
        "0.5-r100": 0.0,
        "0.7-r1": -4.421221864951769,
        "0.7-r5": -10.79849946409432,
        "0.7-r10": -8.226152197213292,
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
        "corr_final_anchor": 0.9995605002443235,
        "corr_final_residual": -0.2779653643371079,
        "top1_changed_rate": 0.12781350482315113,
        "top5_set_changed_rate": 0.4590032154340836,
        "top10_set_changed_rate": 0.6891747052518756,
        "rank_displacement_p50": 4.0,
        "rank_displacement_p90": 14.0,
        "rank_displacement_p99": 32.0
      },
      "hard_constraints_pass": false
    },
    {
      "config": {
        "alpha": 0.2,
        "normalization": "rank_norm",
        "clip": 0.5
      },
      "metrics": {
        "0.5-r1": 1.0182207931404073,
        "0.5-r5": 24.973204715969988,
        "0.5-r10": 37.72775991425509,
        "0.5-r100": 49.59807073954984,
        "0.7-r1": 0.4287245444801715,
        "0.7-r5": 15.728831725616292,
        "0.7-r10": 26.661307609860664,
        "0.7-r100": 41.88102893890675
      },
      "delta_vs_C7_B2_1_A4_stress": {
        "0.5-r1": -7.020364415862809,
        "0.5-r5": -13.987138263665596,
        "0.5-r10": -8.976420150053592,
        "0.5-r100": 0.0,
        "0.7-r1": -4.233654876741694,
        "0.7-r5": -10.610932475884244,
        "0.7-r10": -8.172561629153272,
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
        "corr_final_anchor": 0.9971545440355607,
        "corr_final_residual": -0.2621840490839995,
        "top1_changed_rate": 0.11334405144694534,
        "top5_set_changed_rate": 0.4515005359056806,
        "top10_set_changed_rate": 0.6854233654876741,
        "rank_displacement_p50": 3.0,
        "rank_displacement_p90": 13.0,
        "rank_displacement_p99": 32.0
      },
      "hard_constraints_pass": false
    },
    {
      "config": {
        "alpha": 0.2,
        "normalization": "rank_norm",
        "clip": 1.0
      },
      "metrics": {
        "0.5-r1": 2.465166130760986,
        "0.5-r5": 26.04501607717042,
        "0.5-r10": 37.88853161843515,
        "0.5-r100": 49.59807073954984,
        "0.7-r1": 1.3665594855305465,
        "0.7-r5": 16.505894962486604,
        "0.7-r10": 27.30439442658092,
        "0.7-r100": 41.90782422293676
      },
      "delta_vs_C7_B2_1_A4_stress": {
        "0.5-r1": -5.57341907824223,
        "0.5-r5": -12.915326902465164,
        "0.5-r10": -8.815648445873528,
        "0.5-r100": 0.0,
        "0.7-r1": -3.2958199356913185,
        "0.7-r5": -9.833869239013932,
        "0.7-r10": -7.5294748124330155,
        "0.7-r100": 0.026795284030008304
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
        "corr_final_anchor": 0.9945929131521364,
        "corr_final_residual": -0.24823324244772796,
        "top1_changed_rate": 0.11254019292604502,
        "top5_set_changed_rate": 0.4442658092175777,
        "top10_set_changed_rate": 0.6806002143622722,
        "rank_displacement_p50": 3.0,
        "rank_displacement_p90": 13.0,
        "rank_displacement_p99": 31.0
      },
      "hard_constraints_pass": false
    },
    {
      "config": {
        "alpha": 0.2,
        "normalization": "rank_norm",
        "clip": 2.0
      },
      "metrics": {
        "0.5-r1": 2.465166130760986,
        "0.5-r5": 26.04501607717042,
        "0.5-r10": 37.88853161843515,
        "0.5-r100": 49.59807073954984,
        "0.7-r1": 1.3665594855305465,
        "0.7-r5": 16.505894962486604,
        "0.7-r10": 27.30439442658092,
        "0.7-r100": 41.90782422293676
      },
      "delta_vs_C7_B2_1_A4_stress": {
        "0.5-r1": -5.57341907824223,
        "0.5-r5": -12.915326902465164,
        "0.5-r10": -8.815648445873528,
        "0.5-r100": 0.0,
        "0.7-r1": -3.2958199356913185,
        "0.7-r5": -9.833869239013932,
        "0.7-r10": -7.5294748124330155,
        "0.7-r100": 0.026795284030008304
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
        "corr_final_anchor": 0.9945929131521364,
        "corr_final_residual": -0.24823324244772796,
        "top1_changed_rate": 0.11254019292604502,
        "top5_set_changed_rate": 0.4442658092175777,
        "top10_set_changed_rate": 0.6806002143622722,
        "rank_displacement_p50": 3.0,
        "rank_displacement_p90": 13.0,
        "rank_displacement_p99": 31.0
      },
      "hard_constraints_pass": false
    },
    {
      "config": {
        "alpha": 0.2,
        "normalization": "bounded_minmax",
        "clip": 0.5
      },
      "metrics": {
        "0.5-r1": 1.045016077170418,
        "0.5-r5": 25.455519828510184,
        "0.5-r10": 37.78135048231511,
        "0.5-r100": 49.59807073954984,
        "0.7-r1": 0.4555198285101822,
        "0.7-r5": 15.996784565916398,
        "0.7-r10": 27.009646302250804,
        "0.7-r100": 41.88102893890675
      },
      "delta_vs_C7_B2_1_A4_stress": {
        "0.5-r1": -6.993569131832798,
        "0.5-r5": -13.5048231511254,
        "0.5-r10": -8.922829581993568,
        "0.5-r100": 0.0,
        "0.7-r1": -4.206859592711683,
        "0.7-r5": -10.342979635584138,
        "0.7-r10": -7.824222936763132,
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
        "corr_final_anchor": 0.9966689351404121,
        "corr_final_residual": -0.24529060741243447,
        "top1_changed_rate": 0.11120042872454448,
        "top5_set_changed_rate": 0.3917470525187567,
        "top10_set_changed_rate": 0.610128617363344,
        "rank_displacement_p50": 3.0,
        "rank_displacement_p90": 13.0,
        "rank_displacement_p99": 31.0
      },
      "hard_constraints_pass": false
    },
    {
      "config": {
        "alpha": 0.2,
        "normalization": "bounded_minmax",
        "clip": 1.0
      },
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
      "movement": {
        "hard_positive_top100_query_exits": 0,
        "hard_positive_top100_query_entries": 0,
        "hard_positive_exit_ratio": 0.0,
        "invalid_span_count": 0,
        "duplicate_span_count_after_nms": 0,
        "fixed_pool_invariant": true
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
      "hard_constraints_pass": false
    },
    {
      "config": {
        "alpha": 0.2,
        "normalization": "bounded_minmax",
        "clip": 2.0
      },
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
      "movement": {
        "hard_positive_top100_query_exits": 0,
        "hard_positive_top100_query_entries": 0,
        "hard_positive_exit_ratio": 0.0,
        "invalid_span_count": 0,
        "duplicate_span_count_after_nms": 0,
        "fixed_pool_invariant": true
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
      "hard_constraints_pass": false
    }
  ]
}
```
