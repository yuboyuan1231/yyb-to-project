# C7-B5 strict alignment audit

```json
{
  "status": "C7_B5_SN_ALIGNMENT_INFEASIBLE",
  "official_val_used": false,
  "candidate_unique_key": "desc_id + video_idx + start_idx + end_idx + candidate_row_id + group_id",
  "total_candidate_keys": 873900,
  "matched_candidate_keys": 871873,
  "missing_candidate_keys": 2027,
  "duplicate_candidate_keys": 0,
  "extra_logit_keys": 0,
  "source_rows_outside_fixed_pool": 873900,
  "candidate_identity_set_matches_C7_B2_1_A4_fixed_pool": true,
  "score_vector_candidate_identity_one_to_one": false,
  "position_based_join": false,
  "rounding_mismatch": false,
  "start_end_unit_mismatch": false,
  "NMS_pre_post_index_mismatch": false,
  "candidate_row_id_unstable": false,
  "group_id_video_level_only_issue": false,
  "bad_row_identity_count": 2027,
  "strict_alignment": false
}
```
