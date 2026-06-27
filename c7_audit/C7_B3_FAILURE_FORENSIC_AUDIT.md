# C7-B3 official failure forensic audit

Status: `C7_B3_FAILURE_FORENSIC_AUDIT_COMPLETE`

Scope: diagnostic only. No official val, no second official val, no post-val adjustment, no parameter tuning, no C7-B4/C7-C/C8.

## Final diagnosis

Labels: `RESIDUAL_OVER_DOMINATES_ANCHOR, PER_QUERY_Z_OVER_AMPLIFICATION, TRAIN_ONLY_DISTRIBUTION_SHIFT, POSITIVE_ENRICHED_TRAIN_POOL, OFFICIAL_RESIDUAL_GENERALIZATION_FAILURE, NO_INFRA_BUG_BUT_CALIBRATION_FAILED`

Summary:
- Fixed-pool invariant held, so R@100 stayed unchanged, but topK order changed aggressively.
- `corr(final, residual_z)` = `0.8117` vs `corr(final, anchor)` = `0.5151`.
- top1/top5/top10 changed rates = `0.4951` / `0.7281` / `0.8475`.
- official C7-B3 lost many more queries than it gained vs C7-B2.1 A4, especially R@5/R@10.

## Audit A: score-row alignment

```json
{
  "status": "PASS",
  "join_key_for_candidate_identity": "desc_id + video_idx + start_idx + end_idx + candidate_row_id + group_id",
  "residual_lookup_key": "group_id (query-video residual; span-invariant by design)",
  "unique_candidate_key_duplicate_count": 0,
  "unique_candidate_key_duplicate_examples": [],
  "missing_residual_group_count": 0,
  "external_score_row_position_based_join": false,
  "in_memory_position_based_assignment_after_keyed_lookup": true,
  "HIGH_RISK_ALIGNMENT_BUG": false,
  "C7_B2_1_A4_score_shape": [
    10895,
    100
  ],
  "C7_B3_score_shape": [
    10895,
    100
  ],
  "score_shape_match": true,
  "candidate_order_mismatch_expected": "yes, reranking intentionally changes rank order within the fixed candidate identity set",
  "score_vector_candidate_identity_one_to_one": true,
  "b21_prediction_identity_mismatch_queries": 2,
  "b3_prediction_identity_mismatch_queries": 0,
  "b21_prediction_score_mismatch_queries": 0,
  "b3_prediction_score_mismatch_queries": 0,
  "post_nms_identity_set_changed_queries": 0,
  "post_nms_identity_set_changed_rate": 0.0,
  "alignment_judgment": "C7-B3 archived output reconstructs with zero identity/score mismatch; two C7-B2.1 reference mismatches are tie/rounding-level reconstruction artifacts and do not explain C7-B3's loss."
}
```

## Audit B: score dominance

```json
{
  "corr_final_score_anchor_score_spearman": 0.5151279729126973,
  "corr_final_score_residual_z_score_spearman": 0.8116531005882719,
  "corr_final_score_anchor_score_kendall": 0.3644379183838775,
  "corr_final_score_residual_z_score_kendall": 0.6247687871987422,
  "corr_C7_B2_1_rank_C7_B3_rank_spearman": 0.602456914989342,
  "corr_C7_B2_1_rank_C7_B3_rank_kendall": 0.4518260621608114,
  "top1_changed_rate": 0.4950894905920147,
  "top5_set_changed_rate": 0.728132170720514,
  "top10_set_changed_rate": 0.8474529600734282,
  "mean_rank_displacement": 18.561853408813477,
  "rank_displacement_p50": 13.0,
  "rank_displacement_p90": 45.0,
  "rank_displacement_p99": 74.0,
  "dominance_judgment": "residual_z_score_dominates"
}
```

## Audit C: official loss query analysis

```json
{
  "0.5-r1": {
    "gain_queries": 227,
    "loss_queries": 597,
    "loss_examples": [
      {
        "query_index": 26,
        "desc_id": "89823",
        "anchor_best_positive_rank": 1,
        "C7_B2_1_best_positive_rank": 1,
        "C7_B3_best_positive_rank": 82,
        "residual_z_best_positive_rank": 82
      },
      {
        "query_index": 63,
        "desc_id": "93977",
        "anchor_best_positive_rank": 1,
        "C7_B2_1_best_positive_rank": 1,
        "C7_B3_best_positive_rank": 10,
        "residual_z_best_positive_rank": 74
      },
      {
        "query_index": 78,
        "desc_id": "90362",
        "anchor_best_positive_rank": 1,
        "C7_B2_1_best_positive_rank": 1,
        "C7_B3_best_positive_rank": 90,
        "residual_z_best_positive_rank": 90
      },
      {
        "query_index": 84,
        "desc_id": "94356",
        "anchor_best_positive_rank": 1,
        "C7_B2_1_best_positive_rank": 1,
        "C7_B3_best_positive_rank": 70,
        "residual_z_best_positive_rank": 70
      },
      {
        "query_index": 96,
        "desc_id": "92408",
        "anchor_best_positive_rank": 1,
        "C7_B2_1_best_positive_rank": 1,
        "C7_B3_best_positive_rank": 40,
        "residual_z_best_positive_rank": 40
      },
      {
        "query_index": 137,
        "desc_id": "89011",
        "anchor_best_positive_rank": 1,
        "C7_B2_1_best_positive_rank": 1,
        "C7_B3_best_positive_rank": 24,
        "residual_z_best_positive_rank": 86
      },
      {
        "query_index": 180,
        "desc_id": "88638",
        "anchor_best_positive_rank": 1,
        "C7_B2_1_best_positive_rank": 1,
        "C7_B3_best_positive_rank": 11,
        "residual_z_best_positive_rank": 49
      },
      {
        "query_index": 213,
        "desc_id": "90574",
        "anchor_best_positive_rank": 1,
        "C7_B2_1_best_positive_rank": 1,
        "C7_B3_best_positive_rank": 43,
        "residual_z_best_positive_rank": 81
      },
      {
        "query_index": 224,
        "desc_id": "96533",
        "anchor_best_positive_rank": 1,
        "C7_B2_1_best_positive_rank": 1,
        "C7_B3_best_positive_rank": 2,
        "residual_z_best_positive_rank": 60
      },
      {
        "query_index": 225,
        "desc_id": "89208",
        "anchor_best_positive_rank": 1,
        "C7_B2_1_best_positive_rank": 1,
        "C7_B3_best_positive_rank": 21,
        "residual_z_best_positive_rank": 68
      },
      {
        "query_index": 272,
        "desc_id": "92855",
        "anchor_best_positive_rank": 1,
        "C7_B2_1_best_positive_rank": 1,
        "C7_B3_best_positive_rank": 2,
        "residual_z_best_positive_rank": 40
      },
      {
        "query_index": 326,
        "desc_id": "93955",
        "anchor_best_positive_rank": 1,
        "C7_B2_1_best_positive_rank": 1,
        "C7_B3_best_positive_rank": 70,
        "residual_z_best_positive_rank": 70
      },
      {
        "query_index": 332,
        "desc_id": "88836",
        "anchor_best_positive_rank": 1,
        "C7_B2_1_best_positive_rank": 1,
        "C7_B3_best_positive_rank": 10,
        "residual_z_best_positive_rank": 64
      },
      {
        "query_index": 336,
        "desc_id": "92062",
        "anchor_best_positive_rank": 1,
        "C7_B2_1_best_positive_rank": 1,
        "C7_B3_best_positive_rank": 88,
        "residual_z_best_positive_rank": 88
      },
      {
        "query_index": 340,
        "desc_id": "89966",
        "anchor_best_positive_rank": 1,
        "C7_B2_1_best_positive_rank": 1,
        "C7_B3_best_positive_rank": 27,
        "residual_z_best_positive_rank": 33
      },
      {
        "query_index": 347,
        "desc_id": "92428",
        "anchor_best_positive_rank": 1,
        "C7_B2_1_best_positive_rank": 1,
        "C7_B3_best_positive_rank": 19,
        "residual_z_best_positive_rank": 79
      },
      {
        "query_index": 350,
        "desc_id": "95125",
        "anchor_best_positive_rank": 1,
        "C7_B2_1_best_positive_rank": 1,
        "C7_B3_best_positive_rank": 19,
        "residual_z_best_positive_rank": 76
      },
      {
        "query_index": 390,
        "desc_id": "93008",
        "anchor_best_positive_rank": 1,
        "C7_B2_1_best_positive_rank": 1,
        "C7_B3_best_positive_rank": 19,
        "residual_z_best_positive_rank": 41
      },
      {
        "query_index": 400,
        "desc_id": "88131",
        "anchor_best_positive_rank": 1,
        "C7_B2_1_best_positive_rank": 1,
        "C7_B3_best_positive_rank": 80,
        "residual_z_best_positive_rank": 80
      },
      {
        "query_index": 419,
        "desc_id": "96838",
        "anchor_best_positive_rank": 1,
        "C7_B2_1_best_positive_rank": 1,
        "C7_B3_best_positive_rank": 19,
        "residual_z_best_positive_rank": 49
      },
      {
        "query_index": 424,
        "desc_id": "94535",
        "anchor_best_positive_rank": 1,
        "C7_B2_1_best_positive_rank": 1,
        "C7_B3_best_positive_rank": 13,
        "residual_z_best_positive_rank": 52
      },
      {
        "query_index": 514,
        "desc_id": "97549",
        "anchor_best_positive_rank": 1,
        "C7_B2_1_best_positive_rank": 1,
        "C7_B3_best_positive_rank": 30,
        "residual_z_best_positive_rank": 30
      },
      {
        "query_index": 532,
        "desc_id": "89275",
        "anchor_best_positive_rank": 1,
        "C7_B2_1_best_positive_rank": 1,
        "C7_B3_best_positive_rank": 12,
        "residual_z_best_positive_rank": 31
      },
      {
        "query_index": 547,
        "desc_id": "94841",
        "anchor_best_positive_rank": 1,
        "C7_B2_1_best_positive_rank": 1,
        "C7_B3_best_positive_rank": 10,
        "residual_z_best_positive_rank": 65
      },
      {
        "query_index": 584,
        "desc_id": "94047",
        "anchor_best_positive_rank": 1,
        "C7_B2_1_best_positive_rank": 1,
        "C7_B3_best_positive_rank": 18,
        "residual_z_best_positive_rank": 85
      }
    ]
  },
  "0.5-r5": {
    "gain_queries": 199,
    "loss_queries": 1018,
    "loss_examples": [
      {
        "query_index": 2,
        "desc_id": "89063",
        "anchor_best_positive_rank": 3,
        "C7_B2_1_best_positive_rank": 3,
        "C7_B3_best_positive_rank": 23,
        "residual_z_best_positive_rank": 85
      },
      {
        "query_index": 5,
        "desc_id": "94410",
        "anchor_best_positive_rank": 2,
        "C7_B2_1_best_positive_rank": 2,
        "C7_B3_best_positive_rank": 28,
        "residual_z_best_positive_rank": 57
      },
      {
        "query_index": 8,
        "desc_id": "92752",
        "anchor_best_positive_rank": 2,
        "C7_B2_1_best_positive_rank": 2,
        "C7_B3_best_positive_rank": 6,
        "residual_z_best_positive_rank": 6
      },
      {
        "query_index": 14,
        "desc_id": "95283",
        "anchor_best_positive_rank": 5,
        "C7_B2_1_best_positive_rank": 4,
        "C7_B3_best_positive_rank": 8,
        "residual_z_best_positive_rank": 54
      },
      {
        "query_index": 26,
        "desc_id": "89823",
        "anchor_best_positive_rank": 1,
        "C7_B2_1_best_positive_rank": 1,
        "C7_B3_best_positive_rank": 82,
        "residual_z_best_positive_rank": 82
      },
      {
        "query_index": 36,
        "desc_id": "91164",
        "anchor_best_positive_rank": 2,
        "C7_B2_1_best_positive_rank": 2,
        "C7_B3_best_positive_rank": 8,
        "residual_z_best_positive_rank": 37
      },
      {
        "query_index": 50,
        "desc_id": "87480",
        "anchor_best_positive_rank": 2,
        "C7_B2_1_best_positive_rank": 2,
        "C7_B3_best_positive_rank": 28,
        "residual_z_best_positive_rank": 28
      },
      {
        "query_index": 63,
        "desc_id": "93977",
        "anchor_best_positive_rank": 1,
        "C7_B2_1_best_positive_rank": 1,
        "C7_B3_best_positive_rank": 10,
        "residual_z_best_positive_rank": 74
      },
      {
        "query_index": 78,
        "desc_id": "90362",
        "anchor_best_positive_rank": 1,
        "C7_B2_1_best_positive_rank": 1,
        "C7_B3_best_positive_rank": 90,
        "residual_z_best_positive_rank": 90
      },
      {
        "query_index": 84,
        "desc_id": "94356",
        "anchor_best_positive_rank": 1,
        "C7_B2_1_best_positive_rank": 1,
        "C7_B3_best_positive_rank": 70,
        "residual_z_best_positive_rank": 70
      },
      {
        "query_index": 89,
        "desc_id": "91117",
        "anchor_best_positive_rank": 5,
        "C7_B2_1_best_positive_rank": 5,
        "C7_B3_best_positive_rank": 10,
        "residual_z_best_positive_rank": 11
      },
      {
        "query_index": 93,
        "desc_id": "93535",
        "anchor_best_positive_rank": 4,
        "C7_B2_1_best_positive_rank": 4,
        "C7_B3_best_positive_rank": 38,
        "residual_z_best_positive_rank": 54
      },
      {
        "query_index": 96,
        "desc_id": "92408",
        "anchor_best_positive_rank": 1,
        "C7_B2_1_best_positive_rank": 1,
        "C7_B3_best_positive_rank": 40,
        "residual_z_best_positive_rank": 40
      },
      {
        "query_index": 137,
        "desc_id": "89011",
        "anchor_best_positive_rank": 1,
        "C7_B2_1_best_positive_rank": 1,
        "C7_B3_best_positive_rank": 24,
        "residual_z_best_positive_rank": 86
      },
      {
        "query_index": 139,
        "desc_id": "87808",
        "anchor_best_positive_rank": 3,
        "C7_B2_1_best_positive_rank": 3,
        "C7_B3_best_positive_rank": 62,
        "residual_z_best_positive_rank": 68
      },
      {
        "query_index": 145,
        "desc_id": "91328",
        "anchor_best_positive_rank": 5,
        "C7_B2_1_best_positive_rank": 5,
        "C7_B3_best_positive_rank": 6,
        "residual_z_best_positive_rank": 6
      },
      {
        "query_index": 165,
        "desc_id": "90698",
        "anchor_best_positive_rank": 2,
        "C7_B2_1_best_positive_rank": 2,
        "C7_B3_best_positive_rank": 16,
        "residual_z_best_positive_rank": 58
      },
      {
        "query_index": 166,
        "desc_id": "94183",
        "anchor_best_positive_rank": 3,
        "C7_B2_1_best_positive_rank": 2,
        "C7_B3_best_positive_rank": 33,
        "residual_z_best_positive_rank": 33
      },
      {
        "query_index": 180,
        "desc_id": "88638",
        "anchor_best_positive_rank": 1,
        "C7_B2_1_best_positive_rank": 1,
        "C7_B3_best_positive_rank": 11,
        "residual_z_best_positive_rank": 49
      },
      {
        "query_index": 182,
        "desc_id": "97427",
        "anchor_best_positive_rank": 4,
        "C7_B2_1_best_positive_rank": 5,
        "C7_B3_best_positive_rank": 53,
        "residual_z_best_positive_rank": 71
      },
      {
        "query_index": 188,
        "desc_id": "88964",
        "anchor_best_positive_rank": 4,
        "C7_B2_1_best_positive_rank": 4,
        "C7_B3_best_positive_rank": 13,
        "residual_z_best_positive_rank": 13
      },
      {
        "query_index": 213,
        "desc_id": "90574",
        "anchor_best_positive_rank": 1,
        "C7_B2_1_best_positive_rank": 1,
        "C7_B3_best_positive_rank": 43,
        "residual_z_best_positive_rank": 81
      },
      {
        "query_index": 225,
        "desc_id": "89208",
        "anchor_best_positive_rank": 1,
        "C7_B2_1_best_positive_rank": 1,
        "C7_B3_best_positive_rank": 21,
        "residual_z_best_positive_rank": 68
      },
      {
        "query_index": 233,
        "desc_id": "96338",
        "anchor_best_positive_rank": 7,
        "C7_B2_1_best_positive_rank": 5,
        "C7_B3_best_positive_rank": 29,
        "residual_z_best_positive_rank": 29
      },
      {
        "query_index": 251,
        "desc_id": "92981",
        "anchor_best_positive_rank": 3,
        "C7_B2_1_best_positive_rank": 3,
        "C7_B3_best_positive_rank": 9,
        "residual_z_best_positive_rank": 68
      }
    ]
  },
  "0.5-r10": {
    "gain_queries": 222,
    "loss_queries": 1151,
    "loss_examples": [
      {
        "query_index": 2,
        "desc_id": "89063",
        "anchor_best_positive_rank": 3,
        "C7_B2_1_best_positive_rank": 3,
        "C7_B3_best_positive_rank": 23,
        "residual_z_best_positive_rank": 85
      },
      {
        "query_index": 4,
        "desc_id": "90309",
        "anchor_best_positive_rank": 9,
        "C7_B2_1_best_positive_rank": 9,
        "C7_B3_best_positive_rank": 12,
        "residual_z_best_positive_rank": 22
      },
      {
        "query_index": 5,
        "desc_id": "94410",
        "anchor_best_positive_rank": 2,
        "C7_B2_1_best_positive_rank": 2,
        "C7_B3_best_positive_rank": 28,
        "residual_z_best_positive_rank": 57
      },
      {
        "query_index": 26,
        "desc_id": "89823",
        "anchor_best_positive_rank": 1,
        "C7_B2_1_best_positive_rank": 1,
        "C7_B3_best_positive_rank": 82,
        "residual_z_best_positive_rank": 82
      },
      {
        "query_index": 50,
        "desc_id": "87480",
        "anchor_best_positive_rank": 2,
        "C7_B2_1_best_positive_rank": 2,
        "C7_B3_best_positive_rank": 28,
        "residual_z_best_positive_rank": 28
      },
      {
        "query_index": 78,
        "desc_id": "90362",
        "anchor_best_positive_rank": 1,
        "C7_B2_1_best_positive_rank": 1,
        "C7_B3_best_positive_rank": 90,
        "residual_z_best_positive_rank": 90
      },
      {
        "query_index": 84,
        "desc_id": "94356",
        "anchor_best_positive_rank": 1,
        "C7_B2_1_best_positive_rank": 1,
        "C7_B3_best_positive_rank": 70,
        "residual_z_best_positive_rank": 70
      },
      {
        "query_index": 93,
        "desc_id": "93535",
        "anchor_best_positive_rank": 4,
        "C7_B2_1_best_positive_rank": 4,
        "C7_B3_best_positive_rank": 38,
        "residual_z_best_positive_rank": 54
      },
      {
        "query_index": 96,
        "desc_id": "92408",
        "anchor_best_positive_rank": 1,
        "C7_B2_1_best_positive_rank": 1,
        "C7_B3_best_positive_rank": 40,
        "residual_z_best_positive_rank": 40
      },
      {
        "query_index": 98,
        "desc_id": "92784",
        "anchor_best_positive_rank": 7,
        "C7_B2_1_best_positive_rank": 10,
        "C7_B3_best_positive_rank": 29,
        "residual_z_best_positive_rank": 36
      },
      {
        "query_index": 137,
        "desc_id": "89011",
        "anchor_best_positive_rank": 1,
        "C7_B2_1_best_positive_rank": 1,
        "C7_B3_best_positive_rank": 24,
        "residual_z_best_positive_rank": 86
      },
      {
        "query_index": 139,
        "desc_id": "87808",
        "anchor_best_positive_rank": 3,
        "C7_B2_1_best_positive_rank": 3,
        "C7_B3_best_positive_rank": 62,
        "residual_z_best_positive_rank": 68
      },
      {
        "query_index": 141,
        "desc_id": "92789",
        "anchor_best_positive_rank": 7,
        "C7_B2_1_best_positive_rank": 10,
        "C7_B3_best_positive_rank": 60,
        "residual_z_best_positive_rank": 60
      },
      {
        "query_index": 165,
        "desc_id": "90698",
        "anchor_best_positive_rank": 2,
        "C7_B2_1_best_positive_rank": 2,
        "C7_B3_best_positive_rank": 16,
        "residual_z_best_positive_rank": 58
      },
      {
        "query_index": 166,
        "desc_id": "94183",
        "anchor_best_positive_rank": 3,
        "C7_B2_1_best_positive_rank": 2,
        "C7_B3_best_positive_rank": 33,
        "residual_z_best_positive_rank": 33
      },
      {
        "query_index": 179,
        "desc_id": "91094",
        "anchor_best_positive_rank": 6,
        "C7_B2_1_best_positive_rank": 6,
        "C7_B3_best_positive_rank": 46,
        "residual_z_best_positive_rank": 60
      },
      {
        "query_index": 180,
        "desc_id": "88638",
        "anchor_best_positive_rank": 1,
        "C7_B2_1_best_positive_rank": 1,
        "C7_B3_best_positive_rank": 11,
        "residual_z_best_positive_rank": 49
      },
      {
        "query_index": 182,
        "desc_id": "97427",
        "anchor_best_positive_rank": 4,
        "C7_B2_1_best_positive_rank": 5,
        "C7_B3_best_positive_rank": 53,
        "residual_z_best_positive_rank": 71
      },
      {
        "query_index": 188,
        "desc_id": "88964",
        "anchor_best_positive_rank": 4,
        "C7_B2_1_best_positive_rank": 4,
        "C7_B3_best_positive_rank": 13,
        "residual_z_best_positive_rank": 13
      },
      {
        "query_index": 204,
        "desc_id": "88024",
        "anchor_best_positive_rank": 7,
        "C7_B2_1_best_positive_rank": 9,
        "C7_B3_best_positive_rank": 21,
        "residual_z_best_positive_rank": 21
      },
      {
        "query_index": 213,
        "desc_id": "90574",
        "anchor_best_positive_rank": 1,
        "C7_B2_1_best_positive_rank": 1,
        "C7_B3_best_positive_rank": 43,
        "residual_z_best_positive_rank": 81
      },
      {
        "query_index": 225,
        "desc_id": "89208",
        "anchor_best_positive_rank": 1,
        "C7_B2_1_best_positive_rank": 1,
        "C7_B3_best_positive_rank": 21,
        "residual_z_best_positive_rank": 68
      },
      {
        "query_index": 233,
        "desc_id": "96338",
        "anchor_best_positive_rank": 7,
        "C7_B2_1_best_positive_rank": 5,
        "C7_B3_best_positive_rank": 29,
        "residual_z_best_positive_rank": 29
      },
      {
        "query_index": 261,
        "desc_id": "89522",
        "anchor_best_positive_rank": 3,
        "C7_B2_1_best_positive_rank": 3,
        "C7_B3_best_positive_rank": 45,
        "residual_z_best_positive_rank": 66
      },
      {
        "query_index": 267,
        "desc_id": "96285",
        "anchor_best_positive_rank": 11,
        "C7_B2_1_best_positive_rank": 6,
        "C7_B3_best_positive_rank": 26,
        "residual_z_best_positive_rank": 51
      }
    ]
  },
  "0.7-r1": {
    "gain_queries": 120,
    "loss_queries": 306,
    "loss_examples": [
      {
        "query_index": 26,
        "desc_id": "89823",
        "anchor_best_positive_rank": 1,
        "C7_B2_1_best_positive_rank": 1,
        "C7_B3_best_positive_rank": 82,
        "residual_z_best_positive_rank": 82
      },
      {
        "query_index": 213,
        "desc_id": "90574",
        "anchor_best_positive_rank": 1,
        "C7_B2_1_best_positive_rank": 1,
        "C7_B3_best_positive_rank": 43,
        "residual_z_best_positive_rank": 81
      },
      {
        "query_index": 400,
        "desc_id": "88131",
        "anchor_best_positive_rank": 1,
        "C7_B2_1_best_positive_rank": 1,
        "C7_B3_best_positive_rank": 80,
        "residual_z_best_positive_rank": 80
      },
      {
        "query_index": 419,
        "desc_id": "96838",
        "anchor_best_positive_rank": 1,
        "C7_B2_1_best_positive_rank": 1,
        "C7_B3_best_positive_rank": 19,
        "residual_z_best_positive_rank": 49
      },
      {
        "query_index": 532,
        "desc_id": "89275",
        "anchor_best_positive_rank": 1,
        "C7_B2_1_best_positive_rank": 1,
        "C7_B3_best_positive_rank": 12,
        "residual_z_best_positive_rank": 31
      },
      {
        "query_index": 547,
        "desc_id": "94841",
        "anchor_best_positive_rank": 1,
        "C7_B2_1_best_positive_rank": 1,
        "C7_B3_best_positive_rank": 10,
        "residual_z_best_positive_rank": 65
      },
      {
        "query_index": 584,
        "desc_id": "94047",
        "anchor_best_positive_rank": 1,
        "C7_B2_1_best_positive_rank": 1,
        "C7_B3_best_positive_rank": 18,
        "residual_z_best_positive_rank": 85
      },
      {
        "query_index": 629,
        "desc_id": "93134",
        "anchor_best_positive_rank": 1,
        "C7_B2_1_best_positive_rank": 1,
        "C7_B3_best_positive_rank": 13,
        "residual_z_best_positive_rank": 80
      },
      {
        "query_index": 641,
        "desc_id": "89281",
        "anchor_best_positive_rank": 1,
        "C7_B2_1_best_positive_rank": 1,
        "C7_B3_best_positive_rank": 36,
        "residual_z_best_positive_rank": 78
      },
      {
        "query_index": 645,
        "desc_id": "93878",
        "anchor_best_positive_rank": 1,
        "C7_B2_1_best_positive_rank": 1,
        "C7_B3_best_positive_rank": 13,
        "residual_z_best_positive_rank": 53
      },
      {
        "query_index": 647,
        "desc_id": "92305",
        "anchor_best_positive_rank": 1,
        "C7_B2_1_best_positive_rank": 1,
        "C7_B3_best_positive_rank": 3,
        "residual_z_best_positive_rank": 55
      },
      {
        "query_index": 653,
        "desc_id": "97768",
        "anchor_best_positive_rank": 1,
        "C7_B2_1_best_positive_rank": 1,
        "C7_B3_best_positive_rank": 18,
        "residual_z_best_positive_rank": 41
      },
      {
        "query_index": 663,
        "desc_id": "88166",
        "anchor_best_positive_rank": 1,
        "C7_B2_1_best_positive_rank": 1,
        "C7_B3_best_positive_rank": 2,
        "residual_z_best_positive_rank": 54
      },
      {
        "query_index": 664,
        "desc_id": "91793",
        "anchor_best_positive_rank": 1,
        "C7_B2_1_best_positive_rank": 1,
        "C7_B3_best_positive_rank": 12,
        "residual_z_best_positive_rank": 54
      },
      {
        "query_index": 691,
        "desc_id": "88620",
        "anchor_best_positive_rank": 1,
        "C7_B2_1_best_positive_rank": 1,
        "C7_B3_best_positive_rank": 9,
        "residual_z_best_positive_rank": 50
      },
      {
        "query_index": 721,
        "desc_id": "90481",
        "anchor_best_positive_rank": 1,
        "C7_B2_1_best_positive_rank": 1,
        "C7_B3_best_positive_rank": 2,
        "residual_z_best_positive_rank": 35
      },
      {
        "query_index": 731,
        "desc_id": "96024",
        "anchor_best_positive_rank": 1,
        "C7_B2_1_best_positive_rank": 1,
        "C7_B3_best_positive_rank": 4,
        "residual_z_best_positive_rank": 70
      },
      {
        "query_index": 785,
        "desc_id": "88760",
        "anchor_best_positive_rank": 1,
        "C7_B2_1_best_positive_rank": 1,
        "C7_B3_best_positive_rank": 6,
        "residual_z_best_positive_rank": 55
      },
      {
        "query_index": 787,
        "desc_id": "91843",
        "anchor_best_positive_rank": 1,
        "C7_B2_1_best_positive_rank": 1,
        "C7_B3_best_positive_rank": 22,
        "residual_z_best_positive_rank": 42
      },
      {
        "query_index": 803,
        "desc_id": "94358",
        "anchor_best_positive_rank": 1,
        "C7_B2_1_best_positive_rank": 1,
        "C7_B3_best_positive_rank": 15,
        "residual_z_best_positive_rank": 88
      },
      {
        "query_index": 863,
        "desc_id": "92975",
        "anchor_best_positive_rank": 1,
        "C7_B2_1_best_positive_rank": 1,
        "C7_B3_best_positive_rank": 2,
        "residual_z_best_positive_rank": 73
      },
      {
        "query_index": 875,
        "desc_id": "93979",
        "anchor_best_positive_rank": 1,
        "C7_B2_1_best_positive_rank": 1,
        "C7_B3_best_positive_rank": 15,
        "residual_z_best_positive_rank": 56
      },
      {
        "query_index": 943,
        "desc_id": "96874",
        "anchor_best_positive_rank": 1,
        "C7_B2_1_best_positive_rank": 1,
        "C7_B3_best_positive_rank": 84,
        "residual_z_best_positive_rank": 84
      },
      {
        "query_index": 1015,
        "desc_id": "93759",
        "anchor_best_positive_rank": 1,
        "C7_B2_1_best_positive_rank": 1,
        "C7_B3_best_positive_rank": 14,
        "residual_z_best_positive_rank": 14
      },
      {
        "query_index": 1027,
        "desc_id": "91944",
        "anchor_best_positive_rank": 1,
        "C7_B2_1_best_positive_rank": 1,
        "C7_B3_best_positive_rank": 27,
        "residual_z_best_positive_rank": 27
      }
    ]
  },
  "0.7-r5": {
    "gain_queries": 145,
    "loss_queries": 643,
    "loss_examples": [
      {
        "query_index": 8,
        "desc_id": "92752",
        "anchor_best_positive_rank": 2,
        "C7_B2_1_best_positive_rank": 2,
        "C7_B3_best_positive_rank": 6,
        "residual_z_best_positive_rank": 6
      },
      {
        "query_index": 26,
        "desc_id": "89823",
        "anchor_best_positive_rank": 1,
        "C7_B2_1_best_positive_rank": 1,
        "C7_B3_best_positive_rank": 82,
        "residual_z_best_positive_rank": 82
      },
      {
        "query_index": 36,
        "desc_id": "91164",
        "anchor_best_positive_rank": 2,
        "C7_B2_1_best_positive_rank": 2,
        "C7_B3_best_positive_rank": 8,
        "residual_z_best_positive_rank": 37
      },
      {
        "query_index": 48,
        "desc_id": "90143",
        "anchor_best_positive_rank": 3,
        "C7_B2_1_best_positive_rank": 3,
        "C7_B3_best_positive_rank": 9,
        "residual_z_best_positive_rank": 22
      },
      {
        "query_index": 63,
        "desc_id": "93977",
        "anchor_best_positive_rank": 2,
        "C7_B2_1_best_positive_rank": 2,
        "C7_B3_best_positive_rank": 14,
        "residual_z_best_positive_rank": 75
      },
      {
        "query_index": 93,
        "desc_id": "93535",
        "anchor_best_positive_rank": 4,
        "C7_B2_1_best_positive_rank": 4,
        "C7_B3_best_positive_rank": 38,
        "residual_z_best_positive_rank": 54
      },
      {
        "query_index": 96,
        "desc_id": "92408",
        "anchor_best_positive_rank": 4,
        "C7_B2_1_best_positive_rank": 4,
        "C7_B3_best_positive_rank": 43,
        "residual_z_best_positive_rank": 43
      },
      {
        "query_index": 137,
        "desc_id": "89011",
        "anchor_best_positive_rank": 2,
        "C7_B2_1_best_positive_rank": 4,
        "C7_B3_best_positive_rank": 60,
        "residual_z_best_positive_rank": 87
      },
      {
        "query_index": 165,
        "desc_id": "90698",
        "anchor_best_positive_rank": 2,
        "C7_B2_1_best_positive_rank": 2,
        "C7_B3_best_positive_rank": 16,
        "residual_z_best_positive_rank": 58
      },
      {
        "query_index": 166,
        "desc_id": "94183",
        "anchor_best_positive_rank": 4,
        "C7_B2_1_best_positive_rank": 4,
        "C7_B3_best_positive_rank": 34,
        "residual_z_best_positive_rank": 34
      },
      {
        "query_index": 188,
        "desc_id": "88964",
        "anchor_best_positive_rank": 4,
        "C7_B2_1_best_positive_rank": 4,
        "C7_B3_best_positive_rank": 13,
        "residual_z_best_positive_rank": 13
      },
      {
        "query_index": 213,
        "desc_id": "90574",
        "anchor_best_positive_rank": 1,
        "C7_B2_1_best_positive_rank": 1,
        "C7_B3_best_positive_rank": 43,
        "residual_z_best_positive_rank": 81
      },
      {
        "query_index": 224,
        "desc_id": "96533",
        "anchor_best_positive_rank": 2,
        "C7_B2_1_best_positive_rank": 2,
        "C7_B3_best_positive_rank": 8,
        "residual_z_best_positive_rank": 61
      },
      {
        "query_index": 225,
        "desc_id": "89208",
        "anchor_best_positive_rank": 2,
        "C7_B2_1_best_positive_rank": 2,
        "C7_B3_best_positive_rank": 22,
        "residual_z_best_positive_rank": 69
      },
      {
        "query_index": 233,
        "desc_id": "96338",
        "anchor_best_positive_rank": 7,
        "C7_B2_1_best_positive_rank": 5,
        "C7_B3_best_positive_rank": 29,
        "residual_z_best_positive_rank": 29
      },
      {
        "query_index": 261,
        "desc_id": "89522",
        "anchor_best_positive_rank": 5,
        "C7_B2_1_best_positive_rank": 5,
        "C7_B3_best_positive_rank": 49,
        "residual_z_best_positive_rank": 67
      },
      {
        "query_index": 272,
        "desc_id": "92855",
        "anchor_best_positive_rank": 3,
        "C7_B2_1_best_positive_rank": 3,
        "C7_B3_best_positive_rank": 7,
        "residual_z_best_positive_rank": 42
      },
      {
        "query_index": 326,
        "desc_id": "93955",
        "anchor_best_positive_rank": 3,
        "C7_B2_1_best_positive_rank": 4,
        "C7_B3_best_positive_rank": 71,
        "residual_z_best_positive_rank": 71
      },
      {
        "query_index": 332,
        "desc_id": "88836",
        "anchor_best_positive_rank": 2,
        "C7_B2_1_best_positive_rank": 2,
        "C7_B3_best_positive_rank": 13,
        "residual_z_best_positive_rank": 65
      },
      {
        "query_index": 336,
        "desc_id": "92062",
        "anchor_best_positive_rank": 4,
        "C7_B2_1_best_positive_rank": 4,
        "C7_B3_best_positive_rank": 91,
        "residual_z_best_positive_rank": 91
      },
      {
        "query_index": 347,
        "desc_id": "92428",
        "anchor_best_positive_rank": 5,
        "C7_B2_1_best_positive_rank": 5,
        "C7_B3_best_positive_rank": 28,
        "residual_z_best_positive_rank": 82
      },
      {
        "query_index": 350,
        "desc_id": "95125",
        "anchor_best_positive_rank": 2,
        "C7_B2_1_best_positive_rank": 2,
        "C7_B3_best_positive_rank": 77,
        "residual_z_best_positive_rank": 77
      },
      {
        "query_index": 377,
        "desc_id": "92346",
        "anchor_best_positive_rank": 5,
        "C7_B2_1_best_positive_rank": 5,
        "C7_B3_best_positive_rank": 82,
        "residual_z_best_positive_rank": 85
      },
      {
        "query_index": 390,
        "desc_id": "93008",
        "anchor_best_positive_rank": 2,
        "C7_B2_1_best_positive_rank": 2,
        "C7_B3_best_positive_rank": 22,
        "residual_z_best_positive_rank": 42
      },
      {
        "query_index": 398,
        "desc_id": "96939",
        "anchor_best_positive_rank": 3,
        "C7_B2_1_best_positive_rank": 3,
        "C7_B3_best_positive_rank": 21,
        "residual_z_best_positive_rank": 39
      }
    ]
  },
  "0.7-r10": {
    "gain_queries": 172,
    "loss_queries": 841,
    "loss_examples": [
      {
        "query_index": 2,
        "desc_id": "89063",
        "anchor_best_positive_rank": 8,
        "C7_B2_1_best_positive_rank": 8,
        "C7_B3_best_positive_rank": 90,
        "residual_z_best_positive_rank": 90
      },
      {
        "query_index": 4,
        "desc_id": "90309",
        "anchor_best_positive_rank": 9,
        "C7_B2_1_best_positive_rank": 9,
        "C7_B3_best_positive_rank": 12,
        "residual_z_best_positive_rank": 22
      },
      {
        "query_index": 5,
        "desc_id": "94410",
        "anchor_best_positive_rank": 6,
        "C7_B2_1_best_positive_rank": 6,
        "C7_B3_best_positive_rank": 60,
        "residual_z_best_positive_rank": 60
      },
      {
        "query_index": 26,
        "desc_id": "89823",
        "anchor_best_positive_rank": 1,
        "C7_B2_1_best_positive_rank": 1,
        "C7_B3_best_positive_rank": 82,
        "residual_z_best_positive_rank": 82
      },
      {
        "query_index": 63,
        "desc_id": "93977",
        "anchor_best_positive_rank": 2,
        "C7_B2_1_best_positive_rank": 2,
        "C7_B3_best_positive_rank": 14,
        "residual_z_best_positive_rank": 75
      },
      {
        "query_index": 84,
        "desc_id": "94356",
        "anchor_best_positive_rank": 6,
        "C7_B2_1_best_positive_rank": 7,
        "C7_B3_best_positive_rank": 73,
        "residual_z_best_positive_rank": 73
      },
      {
        "query_index": 93,
        "desc_id": "93535",
        "anchor_best_positive_rank": 4,
        "C7_B2_1_best_positive_rank": 4,
        "C7_B3_best_positive_rank": 38,
        "residual_z_best_positive_rank": 54
      },
      {
        "query_index": 96,
        "desc_id": "92408",
        "anchor_best_positive_rank": 4,
        "C7_B2_1_best_positive_rank": 4,
        "C7_B3_best_positive_rank": 43,
        "residual_z_best_positive_rank": 43
      },
      {
        "query_index": 98,
        "desc_id": "92784",
        "anchor_best_positive_rank": 7,
        "C7_B2_1_best_positive_rank": 10,
        "C7_B3_best_positive_rank": 29,
        "residual_z_best_positive_rank": 36
      },
      {
        "query_index": 137,
        "desc_id": "89011",
        "anchor_best_positive_rank": 2,
        "C7_B2_1_best_positive_rank": 4,
        "C7_B3_best_positive_rank": 60,
        "residual_z_best_positive_rank": 87
      },
      {
        "query_index": 139,
        "desc_id": "87808",
        "anchor_best_positive_rank": 8,
        "C7_B2_1_best_positive_rank": 8,
        "C7_B3_best_positive_rank": 73,
        "residual_z_best_positive_rank": 73
      },
      {
        "query_index": 141,
        "desc_id": "92789",
        "anchor_best_positive_rank": 7,
        "C7_B2_1_best_positive_rank": 10,
        "C7_B3_best_positive_rank": 60,
        "residual_z_best_positive_rank": 60
      },
      {
        "query_index": 165,
        "desc_id": "90698",
        "anchor_best_positive_rank": 2,
        "C7_B2_1_best_positive_rank": 2,
        "C7_B3_best_positive_rank": 16,
        "residual_z_best_positive_rank": 58
      },
      {
        "query_index": 166,
        "desc_id": "94183",
        "anchor_best_positive_rank": 4,
        "C7_B2_1_best_positive_rank": 4,
        "C7_B3_best_positive_rank": 34,
        "residual_z_best_positive_rank": 34
      },
      {
        "query_index": 179,
        "desc_id": "91094",
        "anchor_best_positive_rank": 9,
        "C7_B2_1_best_positive_rank": 9,
        "C7_B3_best_positive_rank": 60,
        "residual_z_best_positive_rank": 63
      },
      {
        "query_index": 180,
        "desc_id": "88638",
        "anchor_best_positive_rank": 7,
        "C7_B2_1_best_positive_rank": 7,
        "C7_B3_best_positive_rank": 24,
        "residual_z_best_positive_rank": 51
      },
      {
        "query_index": 182,
        "desc_id": "97427",
        "anchor_best_positive_rank": 10,
        "C7_B2_1_best_positive_rank": 9,
        "C7_B3_best_positive_rank": 74,
        "residual_z_best_positive_rank": 74
      },
      {
        "query_index": 188,
        "desc_id": "88964",
        "anchor_best_positive_rank": 4,
        "C7_B2_1_best_positive_rank": 4,
        "C7_B3_best_positive_rank": 13,
        "residual_z_best_positive_rank": 13
      },
      {
        "query_index": 204,
        "desc_id": "88024",
        "anchor_best_positive_rank": 7,
        "C7_B2_1_best_positive_rank": 9,
        "C7_B3_best_positive_rank": 21,
        "residual_z_best_positive_rank": 21
      },
      {
        "query_index": 213,
        "desc_id": "90574",
        "anchor_best_positive_rank": 1,
        "C7_B2_1_best_positive_rank": 1,
        "C7_B3_best_positive_rank": 43,
        "residual_z_best_positive_rank": 81
      },
      {
        "query_index": 225,
        "desc_id": "89208",
        "anchor_best_positive_rank": 2,
        "C7_B2_1_best_positive_rank": 2,
        "C7_B3_best_positive_rank": 22,
        "residual_z_best_positive_rank": 69
      },
      {
        "query_index": 233,
        "desc_id": "96338",
        "anchor_best_positive_rank": 7,
        "C7_B2_1_best_positive_rank": 5,
        "C7_B3_best_positive_rank": 29,
        "residual_z_best_positive_rank": 29
      },
      {
        "query_index": 261,
        "desc_id": "89522",
        "anchor_best_positive_rank": 5,
        "C7_B2_1_best_positive_rank": 5,
        "C7_B3_best_positive_rank": 49,
        "residual_z_best_positive_rank": 67
      },
      {
        "query_index": 267,
        "desc_id": "96285",
        "anchor_best_positive_rank": 11,
        "C7_B2_1_best_positive_rank": 6,
        "C7_B3_best_positive_rank": 26,
        "residual_z_best_positive_rank": 51
      },
      {
        "query_index": 312,
        "desc_id": "94623",
        "anchor_best_positive_rank": 6,
        "C7_B2_1_best_positive_rank": 6,
        "C7_B3_best_positive_rank": 60,
        "residual_z_best_positive_rank": 65
      }
    ]
  }
}
```

## Audit D: distribution shift

```json
{
  "train_only": {
    "residual_mean": -0.017948858235335693,
    "residual_std": 0.18782813874253174,
    "residual_skew": 0.8690945844814877,
    "residual_kurtosis": 0.40974860951605496,
    "per_query_residual_std": {
      "p10": 0.09895718246698379,
      "p50": 0.1648409217596054,
      "p90": 0.22809382677078247
    },
    "residual_top1_margin": {
      "p10": 0.0,
      "p50": 0.0,
      "p90": 0.0
    },
    "z_score_max_distribution": {
      "p10": 0.7725428342819214,
      "p50": 2.0035665035247803,
      "p90": 2.9122548580169676
    },
    "z_score_min_distribution": {
      "p10": -1.8795482635498046,
      "p50": -1.2750211954116821,
      "p90": -0.608163595199585
    },
    "clip_saturation_ratio_mean": 0.006274173246366862,
    "clip_saturation_ratio_p90": 0.0,
    "residual_entropy_mean": 4.589293466407424,
    "residual_entropy_p10_p50_p90": {
      "p10": 4.576711370923517,
      "p50": 4.589926396088173,
      "p90": 4.600326174825474
    },
    "top_residual_candidate_anchor_rank": {
      "p10": 1.0,
      "p50": 2.0,
      "p90": 23.0,
      "rank1_rate": 0.4990273486668955,
      "rank_gt10_rate": 0.22336651790822748
    },
    "positive_pool_rate_05_or_07": 0.7864744249914177,
    "anchor_hit_rate_r100_05_or_07": 0.7864744249914177
  },
  "official": {
    "residual_mean": -0.0462261353646775,
    "residual_std": 0.09764212433394416,
    "residual_skew": 0.13974426835092765,
    "residual_kurtosis": 0.9075729275576969,
    "per_query_residual_std": {
      "p10": 0.029801084473729138,
      "p50": 0.07847888767719269,
      "p90": 0.11467834711074831
    },
    "residual_top1_margin": {
      "p10": 0.0,
      "p50": 0.0,
      "p90": 0.0
    },
    "z_score_max_distribution": {
      "p10": 0.4997227728366852,
      "p50": 1.1738351583480835,
      "p90": 2.47847843170166
    },
    "z_score_min_distribution": {
      "p10": -2.499989318847656,
      "p50": -1.4331319332122803,
      "p90": -0.6753527641296386
    },
    "clip_saturation_ratio_mean": 0.00586048646167967,
    "clip_saturation_ratio_p90": 0.0,
    "residual_entropy_mean": 4.601665297590881,
    "residual_entropy_p10_p50_p90": {
      "p10": 4.598421163504117,
      "p50": 4.602148301063982,
      "p90": 4.604724829996073
    },
    "top_residual_candidate_anchor_rank": {
      "p10": 1.0,
      "p50": 4.0,
      "p90": 34.0,
      "rank1_rate": 0.304726938962827,
      "rank_gt10_rate": 0.3044515832950895
    },
    "positive_pool_rate_05_or_07": 0.4792106470858192,
    "anchor_hit_rate_r100_05_or_07": 0.4792106470858192
  },
  "official_minus_train": {
    "residual_mean": -0.028277277129341805,
    "residual_std": -0.09018601440858758,
    "residual_entropy_mean": 0.012371831183457083,
    "positive_pool_rate_05_or_07": -0.30726377790559856,
    "anchor_hit_rate_r100_05_or_07": -0.30726377790559856
  }
}
```

## Audit E: per_query_z formula

```json
{
  "z_q_residual_over_T_equivalent_to_z_q_residual": true,
  "max_abs_difference_T_0_5_vs_T_1_0_after_z": 0.0,
  "T_effect": "cancelled_by_per_query_z_except_float_noise",
  "clip_after_z_score": true,
  "lambda": 1.0,
  "lambda_strength_judgment": "strong_for_unit_variance_z_added_to_raw_anchor_score",
  "anchor_score_vs_z_residual_std_ratio_p10_p50_p90": {
    "p10": 1.3693139155522729,
    "p50": 2.1597543236299503,
    "p90": 3.3103229857333334
  },
  "final_score_formula": "final_score = anchor_score + lambda * clip(z_q(residual / T), -clip, clip)",
  "risk": "per_query_z normalizes small raw residual differences to unit-scale perturbations, so lambda=1 can dominate topK ordering."
}
```

## Audit F: train-only optimism

```json
{
  "train_split_manifest": {
    "train_core": {
      "query_count": 4268,
      "positive_pool_rate": 0.7846766710281372,
      "anchor_hit_rate_r100_05_or_07": 0.7828022492970946,
      "query_length_mean": 12.231255531311035,
      "query_length_p50": 11.0,
      "query_length_p90": 17.0,
      "hash": "f2ca830763802b974113afc29efea42a31ba7f619137978c0bb0c51acaaeca7f"
    },
    "calib_A": {
      "query_count": 1411,
      "positive_pool_rate": 0.7859674096107483,
      "anchor_hit_rate_r100_05_or_07": 0.7824238128986535,
      "query_length_mean": 12.225372314453125,
      "query_length_p50": 11.0,
      "query_length_p90": 17.0,
      "hash": "e29ae5beaa7352a610c381693dd814ead3058443f7b58048f045e5ff3c60ff8a"
    },
    "calib_B": {
      "query_count": 1344,
      "positive_pool_rate": 0.7953869104385376,
      "anchor_hit_rate_r100_05_or_07": 0.7931547619047619,
      "query_length_mean": 12.290922164916992,
      "query_length_p50": 11.0,
      "query_length_p90": 17.0,
      "hash": "3497de880b934b51cb70d5a183fc5fb5f283d51899264da9cea3aea3bb29cb3d"
    },
    "calib_C": {
      "query_count": 1246,
      "positive_pool_rate": 0.7800962924957275,
      "anchor_hit_rate_r100_05_or_07": 0.7728731942215089,
      "query_length_mean": 12.101123809814453,
      "query_length_p50": 11.0,
      "query_length_p90": 17.0,
      "hash": "24ff013688413f4533bc9bc50d766d524b3847d069d2c8191fd860ec01b1d193"
    },
    "stress_split": {
      "query_count": 1748,
      "positive_pool_rate": 0.9210526347160339,
      "anchor_hit_rate_r100_05_or_07": 0.915903890160183,
      "query_length_mean": 12.455377578735352,
      "query_length_p50": 11.0,
      "query_length_p90": 17.0,
      "hash": "1726a24e91f2536e43ae3da5c17ffe30e643ae97fcc7210bf5ebe75b085e352d"
    },
    "train_calib_final_review": {
      "query_count": 8739,
      "positive_pool_rate": 0.7864744067192078,
      "anchor_hit_rate_r100_05_or_07": 0.7836136857764047,
      "query_length_mean": 12.231719970703125,
      "query_length_p50": 11.0,
      "query_length_p90": 17.0,
      "hash": "1ba9829704c299bcd5e24917a03709d24f124106d9aee4edfbddd625fda3efb5"
    }
  },
  "official_positive_pool_rate_05_or_07": 0.4792106470858192,
  "official_anchor_hit_rate_r100_05_or_07": 0.4792106470858192,
  "train_final_review_positive_pool_rate_05_or_07": 0.7864744067192078,
  "train_final_review_anchor_hit_rate_r100_05_or_07": 0.7836136857764047,
  "stress_split_positive_pool_rate": 0.9210526347160339,
  "stress_split_is_positive_rich": true,
  "candidate_construction_GT_forced_or_positive_enriched": "positive_enriched_relative_to_official; no evidence of explicit GT-forced official candidate construction in C7-B3 inference path",
  "train_split_candidate_generation_supervision_bias": "likely distribution/selection bias: train_calib fixed pool has much higher positive-pool/R100 rate than official",
  "diagnostic_label_used_as_scoring_feature": false,
  "diagnostic_global_A4_gt_anchor_R1_flag_found_in_diagnostics": true,
  "diagnostic_global_A4_gt_anchor_R1_flag_inference_use": false,
  "label_leakage_judgment": "not_suspected_from_code_and_freeze_leakage_audit"
}
```

## Recommendations

- Keep C7-B2.1 A4 as current promoted system.
- Archive C7-B3 as official no-gain/failure.
- Do not tune C7-B3 after official.
- If continuing, open C7-B4-SN-Audit based on MINUTE-style raw logit/shared-norm scoring, not C7-B3 per_query_z repair.
