# C6-C2 action dataset audit

- Scope: `train_fit/train_calib only`
- Official val used: `false`

## train_fit

```json
{
  "queries": 78045,
  "samples": 468270,
  "feature_dim": 31,
  "number_of_actions": 6,
  "action_names": [
    "A0_noop_B2",
    "A1_MAVR_soft_rerank",
    "A2_PREM_PRVR_MIL_promote",
    "A3_topK_preserving_promote",
    "A4_raw200_video_insertion",
    "A5_BMN_proposal_confidence_boost"
  ],
  "positive_action_rate": 0.015407777391374111,
  "bad_action_rate": 0.09225446730852127,
  "no_op_rate": 0.7338245029576953,
  "actions_with_R1_gain_but_R5_loss": 0,
  "actions_with_video_correction": 1857,
  "actions_with_video_regression": 31699,
  "changed_nonnoop_actions": 124642,
  "action_distribution": {
    "A0_noop_B2": 78045,
    "A1_MAVR_soft_rerank": 78045,
    "A2_PREM_PRVR_MIL_promote": 78045,
    "A3_topK_preserving_promote": 78045,
    "A4_raw200_video_insertion": 78045,
    "A5_BMN_proposal_confidence_boost": 78045
  },
  "feature_leakage_guard": {
    "gt_iou_as_feature": false,
    "hit_label_as_feature": false,
    "train_calib_label_as_feature": false,
    "official_val_prediction_as_feature": false,
    "oracle_decision_as_feature": false
  }
}
```
## train_calib

```json
{
  "queries": 8739,
  "samples": 52434,
  "feature_dim": 31,
  "number_of_actions": 6,
  "action_names": [
    "A0_noop_B2",
    "A1_MAVR_soft_rerank",
    "A2_PREM_PRVR_MIL_promote",
    "A3_topK_preserving_promote",
    "A4_raw200_video_insertion",
    "A5_BMN_proposal_confidence_boost"
  ],
  "positive_action_rate": 0.015161917544901371,
  "bad_action_rate": 0.09144829958677292,
  "no_op_rate": 0.7344661860624786,
  "actions_with_R1_gain_but_R5_loss": 0,
  "actions_with_video_correction": 180,
  "actions_with_video_regression": 3492,
  "changed_nonnoop_actions": 13923,
  "action_distribution": {
    "A0_noop_B2": 8739,
    "A1_MAVR_soft_rerank": 8739,
    "A2_PREM_PRVR_MIL_promote": 8739,
    "A3_topK_preserving_promote": 8739,
    "A4_raw200_video_insertion": 8739,
    "A5_BMN_proposal_confidence_boost": 8739
  },
  "feature_leakage_guard": {
    "gt_iou_as_feature": false,
    "hit_label_as_feature": false,
    "train_calib_label_as_feature": false,
    "official_val_prediction_as_feature": false,
    "oracle_decision_as_feature": false
  }
}
```
