# C7-B2.1 leakage audit

```json
{
  "status": "PASS",
  "official_val_used": false,
  "inference_artifacts": [
    "train_calib_video_scores.npz model predictions",
    "C7-B1 frozen flat candidate scores"
  ],
  "feature_names": [
    "c7_prop_conf_max",
    "c7_prop_conf_mean",
    "c7_span_quality_max",
    "c7_span_quality_mean",
    "c7_span_hit05_max",
    "c7_span_hit07_max",
    "c7_span_iou_max",
    "c7_rank_inv_max",
    "c7_prop_count_norm",
    "c6c_relevance",
    "c6c_moment",
    "c6c_iou",
    "c6c_risk",
    "c6c_gate",
    "base_rank_inv",
    "base_rank_norm",
    "video_feature_0",
    "video_feature_1",
    "video_feature_2",
    "video_feature_3",
    "video_feature_4",
    "video_feature_5",
    "video_feature_6",
    "video_feature_7",
    "video_feature_8",
    "video_feature_9",
    "video_feature_10",
    "video_feature_11",
    "video_feature_12",
    "video_feature_13",
    "video_feature_14",
    "video_feature_15",
    "video_feature_16",
    "video_feature_17",
    "video_feature_18",
    "video_feature_19",
    "video_feature_20",
    "video_feature_21",
    "video_feature_22",
    "video_feature_23",
    "video_feature_24",
    "video_feature_25",
    "video_feature_26"
  ],
  "banned_feature_name_matches": [],
  "allowed_predicted_iou_like_features": [
    "c6c_iou"
  ],
  "contains_GT_IoU_as_inference_feature": false,
  "contains_hit_label_as_inference_feature": false,
  "contains_train_calib_label_as_inference_feature": false,
  "contains_official_val_prediction_as_inference_feature": false,
  "contains_oracle_decision_as_inference_feature": false,
  "training_labels_allowed_only_for_training": true
}
```
