# C13 Evidence Packet - 2026-07-04

Purpose: make C13 handoff auditable from GitHub without relying on chat memory. This packet answers the specific concerns about whether C13 artifacts are real, where they live, which branch contains them, why C12 ranker work stopped, what C7-B6 remains promoted, and whether Path B is practically feasible.

## Executive Conclusion

- C13 artifacts are real generated files on branch `c12-clean-native-vcmr`, not only a plan.
- `main` remains at commit `31a7023` and does not contain the C12-5T/U/V or C13 additions.
- `c12-clean-native-vcmr` contains the new C12-5T/U/V and C13 evidence commit `9849ed8` before this evidence-packet commit.
- C13 selected `Path B`: `C13_STRONG_FEATURE_EXTRACTION_THEN_NATIVE_RETRAIN`.
- C12 span-ranker line is stopped; C12-6 is not allowed.
- Current promoted system remains `C7-B6 R1SelectiveTop1` / C7C6-B Region-lite official candidate, not C12 or C13.

## Branch and Repository State

### git branch -vv
```
  c10-postofficial-repair-diagnostics 31a7023 Archive CONQUER-RLEM promoted C7-B6 system
* c12-clean-native-vcmr               9849ed8 [origin/c12-clean-native-vcmr] Archive C12 span pivot and C13 decision
  main                                31a7023 [origin/main] Archive CONQUER-RLEM promoted C7-B6 system
```

### git log --oneline --decorate -5
```
9849ed8 (HEAD -> c12-clean-native-vcmr, origin/c12-clean-native-vcmr) Archive C12 span pivot and C13 decision
31a7023 (origin/main, main, c10-postofficial-repair-diagnostics) Archive CONQUER-RLEM promoted C7-B6 system
0954258 Initial commit
```

### Remote Heads
```
9849ed889d26597ed12343ab540382c09456dc39	refs/heads/c12-clean-native-vcmr
31a7023b3677c791503e73cade8b659788643c40	refs/heads/main
```

### Default Branch
```
ref: refs/heads/main	HEAD
31a7023b3677c791503e73cade8b659788643c40	HEAD
```

### Difference: main..c12-clean-native-vcmr

The work branch adds C12-5T/C12-5U/C12-5V/C13 files and scripts. Before this evidence packet, the exact diff from `main` was the C12/C13 package committed at `9849ed8`.
```
A	c12_5t_best_span_promotion/C12_5T_0_MARKER_QUARANTINE.json
A	c12_5t_best_span_promotion/C12_5T_0_MARKER_QUARANTINE.md
A	c12_5t_best_span_promotion/C12_5T_A_FEATURE_SCHEMA.json
A	c12_5t_best_span_promotion/C12_5T_A_FEATURE_SEMANTICS_FIX.md
A	c12_5t_best_span_promotion/C12_5T_A_RETRIEVER_SCORE_AUDIT.json
A	c12_5t_best_span_promotion/C12_5T_B_GENERATED_POOL_AUDIT.json
A	c12_5t_best_span_promotion/C12_5T_B_GENERATED_POOL_LOCK.md
A	c12_5t_best_span_promotion/C12_5T_B_PROMOTION_DATASET_AUDIT.json
A	c12_5t_best_span_promotion/C12_5T_B_PROMOTION_DATASET_AUDIT.md
A	c12_5t_best_span_promotion/C12_5T_C_TRAINING_RESULTS.json
A	c12_5t_best_span_promotion/C12_5T_DECISION.json
A	c12_5t_best_span_promotion/C12_5T_DECISION.md
A	c12_5t_best_span_promotion/C12_5T_D_CALIBRATION_RESULTS.json
A	c12_5t_best_span_promotion/C12_5T_E_DURATION_BREAKDOWN.json
A	c12_5t_best_span_promotion/C12_5T_E_FAILURE_CASES.json
A	c12_5t_best_span_promotion/C12_5T_E_QUERY_TYPE_BREAKDOWN.json
A	c12_5t_best_span_promotion/C12_5T_E_RESULTS.json
A	c12_5t_best_span_promotion/C12_5T_E_RESULTS.md
A	c12_5t_best_span_promotion/C12_5T_MANIFEST.json
A	c12_5t_best_span_promotion/quarantine_official_markers/C9_OFFICIAL_VAL_AUTHORIZED.stale_20260703
A	c12_5t_best_span_promotion/quarantine_official_markers/OFFICIAL_VAL_AUTHORIZED.stale_20260703
A	c12_5u_topk_calibration_repair/C12_5U_A_SCORE_DECOMPOSITION_AUDIT.json
A	c12_5u_topk_calibration_repair/C12_5U_A_SCORE_DECOMPOSITION_AUDIT.md
A	c12_5u_topk_calibration_repair/C12_5U_B_TWO_HEAD_RANKER_RESULTS.json
A	c12_5u_topk_calibration_repair/C12_5U_B_TWO_HEAD_RANKER_RESULTS.md
A	c12_5u_topk_calibration_repair/C12_5U_C_CALIBRATION_REPAIR_RESULTS.json
A	c12_5u_topk_calibration_repair/C12_5U_C_CALIBRATION_REPAIR_RESULTS.md
A	c12_5u_topk_calibration_repair/C12_5U_DECISION.json
A	c12_5u_topk_calibration_repair/C12_5U_DECISION.md
A	c12_5u_topk_calibration_repair/C12_5U_D_CANDIDATE_VIDEO_SCORE_SEMANTICS.md
A	c12_5u_topk_calibration_repair/C12_5U_D_FEATURE_SCHEMA_HASH.json
A	c12_5u_topk_calibration_repair/C12_5U_E_CALIBRATION_METRICS.json
A	c12_5u_topk_calibration_repair/C12_5U_E_DURATION_BREAKDOWN.json
A	c12_5u_topk_calibration_repair/C12_5U_E_FAILURE_CASES.json
A	c12_5u_topk_calibration_repair/C12_5U_E_QUERY_TYPE_BREAKDOWN.json
A	c12_5u_topk_calibration_repair/C12_5U_E_RESULTS.json
A	c12_5u_topk_calibration_repair/C12_5U_E_RESULTS.md
A	c12_5u_topk_calibration_repair/C12_5U_MANIFEST.json
A	c12_5u_topk_calibration_repair/C12_5U_PROMOTION_DATASET_AUDIT.json
A	c12_5v_conquer_hidden_ranker/C12_5V_A_FORWARD_HOOK_AUDIT.json
A	c12_5v_conquer_hidden_ranker/C12_5V_A_HIDDEN_FEATURE_AVAILABILITY.md
A	c12_5v_conquer_hidden_ranker/C12_5V_A_HIDDEN_FEATURE_SCHEMA.json
A	c12_5v_conquer_hidden_ranker/C12_5V_A_MISSING_NONFINITE_AUDIT.json
A	c12_5v_conquer_hidden_ranker/C12_5V_B_FEATURE_BUILDER_HASH.json
A	c12_5v_conquer_hidden_ranker/C12_5V_B_SHARED_FEATURE_BUILDER.md
A	c12_5v_conquer_hidden_ranker/C12_5V_B_TRAIN_INFERENCE_SCHEMA_AUDIT.json
A	c12_5v_conquer_hidden_ranker/C12_5V_C_HIDDEN_RANKER_TRAINING_RESULTS.json
A	c12_5v_conquer_hidden_ranker/C12_5V_C_TRAINING_DATASET_AUDIT.json
A	c12_5v_conquer_hidden_ranker/C12_5V_DECISION.json
A	c12_5v_conquer_hidden_ranker/C12_5V_DECISION.md
A	c12_5v_conquer_hidden_ranker/C12_5V_D_CALIBRATION_METRICS.json
A	c12_5v_conquer_hidden_ranker/C12_5V_D_DURATION_BREAKDOWN.json
A	c12_5v_conquer_hidden_ranker/C12_5V_D_FAILURE_CASES.json
A	c12_5v_conquer_hidden_ranker/C12_5V_D_QUERY_TYPE_BREAKDOWN.json
A	c12_5v_conquer_hidden_ranker/C12_5V_D_RESULTS.json
A	c12_5v_conquer_hidden_ranker/C12_5V_D_RESULTS.md
A	c12_5v_conquer_hidden_ranker/C12_5V_MANIFEST.json
A	c13_0_c12_stop_archive/C13_0_C12_FINAL_STATE.json
A	c13_0_c12_stop_archive/C13_0_C12_FINAL_STATE.md
A	c13_0_c12_stop_archive/C13_0_NO_C12_6_DECISION.md
A	c13_0_c12_stop_archive/C13_0_PROMOTED_SYSTEM_STATUS.md
A	c13_1_strong_feature_feasibility/C13_1_EXTRACTION_COST_ESTIMATE.json
A	c13_1_strong_feature_feasibility/C13_1_FEATURE_CANDIDATES.md
A	c13_1_strong_feature_feasibility/C13_1_FEATURE_DECISION.json
A	c13_1_strong_feature_feasibility/C13_1_MULTISCALE_EVENT_FEATURE_PLAN.md
A	c13_1_strong_feature_feasibility/C13_1_SAMPLE_COVERAGE_AUDIT.json
A	c13_1_strong_feature_feasibility/C13_1_TEXT_SUBTITLE_FEATURE_PLAN.md
A	c13_1_strong_feature_feasibility/C13_1_VISUAL_FEATURE_PLAN.md
A	c13_2_baseline_architecture_feasibility/C13_2_BASELINE_DECISION.json
A	c13_2_baseline_architecture_feasibility/C13_2_BOUNDARY_ARCHITECTURE_OPTIONS.md
A	c13_2_baseline_architecture_feasibility/C13_2_EVENTFORMER_FEASIBILITY.md
A	c13_2_baseline_architecture_feasibility/C13_2_MAVR_FEASIBILITY.md
A	c13_2_baseline_architecture_feasibility/C13_2_MINUTE_FEASIBILITY.md
A	c13_2_baseline_architecture_feasibility/C13_2_PREM_FEASIBILITY.md
A	c13_3_next_step_decision/C13_3_MANIFEST.json
A	c13_3_next_step_decision/C13_3_NEXT_STEP_DECISION.json
A	c13_3_next_step_decision/C13_3_NEXT_STEP_DECISION.md
A	run_c12_5t_best_span_promotion.py
A	run_c12_5u_topk_calibration_repair.py
A	run_c12_5v_conquer_hidden_ranker.py
A	run_c13_feature_baseline_pivot.py
```

## C13 Directory Inventory
```
c13_0_c12_stop_archive/C13_0_C12_FINAL_STATE.json
c13_0_c12_stop_archive/C13_0_C12_FINAL_STATE.md
c13_0_c12_stop_archive/C13_0_NO_C12_6_DECISION.md
c13_0_c12_stop_archive/C13_0_PROMOTED_SYSTEM_STATUS.md
c13_1_strong_feature_feasibility/C13_1_EXTRACTION_COST_ESTIMATE.json
c13_1_strong_feature_feasibility/C13_1_FEATURE_CANDIDATES.md
c13_1_strong_feature_feasibility/C13_1_FEATURE_DECISION.json
c13_1_strong_feature_feasibility/C13_1_MULTISCALE_EVENT_FEATURE_PLAN.md
c13_1_strong_feature_feasibility/C13_1_SAMPLE_COVERAGE_AUDIT.json
c13_1_strong_feature_feasibility/C13_1_TEXT_SUBTITLE_FEATURE_PLAN.md
c13_1_strong_feature_feasibility/C13_1_VISUAL_FEATURE_PLAN.md
c13_2_baseline_architecture_feasibility/C13_2_BASELINE_DECISION.json
c13_2_baseline_architecture_feasibility/C13_2_BOUNDARY_ARCHITECTURE_OPTIONS.md
c13_2_baseline_architecture_feasibility/C13_2_EVENTFORMER_FEASIBILITY.md
c13_2_baseline_architecture_feasibility/C13_2_MAVR_FEASIBILITY.md
c13_2_baseline_architecture_feasibility/C13_2_MINUTE_FEASIBILITY.md
c13_2_baseline_architecture_feasibility/C13_2_PREM_FEASIBILITY.md
c13_3_next_step_decision/C13_3_MANIFEST.json
c13_3_next_step_decision/C13_3_NEXT_STEP_DECISION.json
c13_3_next_step_decision/C13_3_NEXT_STEP_DECISION.md
```

## C13 Artifact Hashes
```
8d8af3d380f4467fa84370f3e80931313e5bbefd3537abb283fbcbe3b68fd44c  c13_0_c12_stop_archive/C13_0_C12_FINAL_STATE.json
c95d2f94f927579d2fd5f4fa7416db62f091026a1139d23e90f17fc25aaeb02a  c13_0_c12_stop_archive/C13_0_C12_FINAL_STATE.md
6cda268d3048ddfb1a2e6ade92c1a80260306b07dc6069bba5e9640ebb13a3f5  c13_0_c12_stop_archive/C13_0_NO_C12_6_DECISION.md
c44a8b7ce286941869102fef970f6362bcb211598ba2073738a218d4968790ea  c13_0_c12_stop_archive/C13_0_PROMOTED_SYSTEM_STATUS.md
53de72b905193fe4f237092c2cc89a998ec8ee4693290b5ad852f93a796b95f8  c13_1_strong_feature_feasibility/C13_1_EXTRACTION_COST_ESTIMATE.json
88ba8ae6dba73d672677838b1973588328b84fe1b668b4db0b133bb62d5b1558  c13_1_strong_feature_feasibility/C13_1_FEATURE_CANDIDATES.md
723482d3650b8acee91e64824481d3f9381c8168d47bc5b79a38698fd9755b1a  c13_1_strong_feature_feasibility/C13_1_FEATURE_DECISION.json
7396f7722385c51e28d6924c5f320bb6ad38cd43a896916b2f147054a64b93fa  c13_1_strong_feature_feasibility/C13_1_MULTISCALE_EVENT_FEATURE_PLAN.md
ead733e60da0134b9032cec557ccb49eca8f25c904b868c27484117328b23f02  c13_1_strong_feature_feasibility/C13_1_SAMPLE_COVERAGE_AUDIT.json
c8e914f1a1a2cf6b41741c8aa33cff84c591875e40c37b880ff5ec840b824b7b  c13_1_strong_feature_feasibility/C13_1_TEXT_SUBTITLE_FEATURE_PLAN.md
633bba9607d2c089b8392f6bf6d77e81795c4b728b55d5616bb16377e0355c96  c13_1_strong_feature_feasibility/C13_1_VISUAL_FEATURE_PLAN.md
3cfd8d65eb7cebcc5f26649b1b33819eafaf84eb0582f41f52c0f1c6498efd6b  c13_2_baseline_architecture_feasibility/C13_2_BASELINE_DECISION.json
e9ff956831d08e5f37f82a3a157fc2e9068636693dd037e3658ed40fc908d641  c13_2_baseline_architecture_feasibility/C13_2_BOUNDARY_ARCHITECTURE_OPTIONS.md
a211e4df8e3e6c9782b8b31700552e812d783cf6b80e975d9143c781a8e201b2  c13_2_baseline_architecture_feasibility/C13_2_EVENTFORMER_FEASIBILITY.md
6f1f79ffbe9833b47fe56fe62735bbe6b365e2e3ff3ab22b07fd1f6f72ab1443  c13_2_baseline_architecture_feasibility/C13_2_MAVR_FEASIBILITY.md
2098710be3e14ad6893f287b45f2f36222605006926552b2ff5bdfb46f8a2cb3  c13_2_baseline_architecture_feasibility/C13_2_MINUTE_FEASIBILITY.md
1aed3f8652d4e002c1198651e94e0bc1db98d26a6a779a3ecf888222921a7452  c13_2_baseline_architecture_feasibility/C13_2_PREM_FEASIBILITY.md
59aa42aa953293913a6175961f8d78153476df48e0dd66413b3b61149313d122  c13_3_next_step_decision/C13_3_MANIFEST.json
fc3a042449baf22867417aa6b91292e830d1713da0bfa5e951e2a33007bc9c21  c13_3_next_step_decision/C13_3_NEXT_STEP_DECISION.json
4229b2f49cbb38d70c9802110045e0693041e28ef1a3622fd428317e135b42c3  c13_3_next_step_decision/C13_3_NEXT_STEP_DECISION.md
```

## Full C13 Artifact Contents

### c13_0_c12_stop_archive/C13_0_C12_FINAL_STATE.md
```markdown
# C13-0 C12 Final State

Final status: `C12_SPAN_RANKER_LINE_STOP_NEED_STRONGER_FEATURE_OR_BASELINE`

Current promoted system: `C7-B6 R1SelectiveTop1`

Required state facts:

1. C12 independent native retriever is weak: best C12-4 holdout GT-video top100 is 6.7512, while CONQUER reference is 100.0000.
2. C12-4R uses CONQUER-warm / zero-delta replay bridge: status `C12_RETRIEVER_CONQUER_WARM_READY_CONTINUE_TO_C12_5`, best repaired retriever `zero_delta_replay`.
3. C12 M1000 span generation has oracle headroom: T2 generated M1000 IoU@0.7 is 85.0461.
4. C12 top100 span ranking was partially repaired by T2: IoU@0.7 top100 55.0000, top50 38.8710.
5. C12 calibration / PQ-IoU correlation remains unreliable: T2 Spearman -0.1992, U2 Spearman -0.1481, V1 Spearman -0.1896.
6. C12 hidden ranker regresses: V1 IoU@0.7 top100 37.7650, top50 26.9585.
7. C12-6 is not allowed.
8. Current promoted system remains `C7-B6 R1SelectiveTop1`.

official_val_used = false
evaluator_modified = false
nms_modified = false
```

### c13_0_c12_stop_archive/C13_0_C12_FINAL_STATE.json
```json
{
  "allow_enter_c12_6": false,
  "c12_span_ranker_line_stopped": true,
  "create_c12_official_candidate": false,
  "current_promoted_system": "C7-B6 R1SelectiveTop1",
  "evaluator_modified": false,
  "evaluator_sha256": "7ba192993f461e92235614f5033dc5700155ca2cf61d2d0b8f98cbc211786f3a",
  "evidence": {
    "c12_4r_conquer_warm_replay": {
      "GT_video_in_top100_rate": 100.0,
      "best_repaired_retriever_variant": "zero_delta_replay",
      "learned_delta_repair_effective": false,
      "status": "C12_RETRIEVER_CONQUER_WARM_READY_CONTINUE_TO_C12_5"
    },
    "c12_calibration_unreliable": {
      "t2_pq_spearman": -0.199218575616094,
      "u2_pq_spearman": -0.14811460730177597,
      "v1_pq_spearman": -0.18961203992882372
    },
    "c12_hidden_ranker_regression": {
      "best_variant": "V1_hidden_boundary",
      "hidden_export_available": true,
      "iou07_top100": 37.764976958525345,
      "iou07_top50": 26.95852534562212,
      "shared_feature_count": 55,
      "status": "C12_HIDDEN_RANKER_NO_GAIN_NEED_STRONGER_FEATURES"
    },
    "c12_independent_native_retriever": {
      "GT_video_in_top100_rate": 6.751152073732719,
      "best_variant": "teacher_distilled_fusion",
      "reference_CONQUER_top100": 100.0,
      "status": "C12_RETRIEVER_COMPLEMENTARY_BUT_WEAK"
    },
    "c12_m1000_span_generation": {
      "holdout_iou07_top100": "Selection used calib_select only. Holdout is report-only. Selected teacher_distilled; holdout IoU@0.7_top100=43.0876, B6 train-reference IoU@0.7_top100=66.0156, query_type_collapse=False.",
      "oracle_headroom_from_t2_generated_m1000_iou07": 85.04608294930875,
      "selected_variant": "teacher_distilled",
      "status": "C12_LOCALIZER_WEAK_REVISE_SPAN_HEAD"
    },
    "c12_top100_span_ranking_partial_repair": {
      "best_t_variant": "T2_listwise_soft_iou",
      "iou07_top100": 55.0,
      "iou07_top50": 38.87096774193548
    }
  },
  "nms_modified": false,
  "nms_sha256": "f490e62ebd98c1bb6d0054d147a7df5a5ead508cea5b26be82d1cd8ac88619cd",
  "official_val_used": false,
  "root_official_authorization_markers_present": [],
  "stage": "C13-0",
  "status": "C12_SPAN_RANKER_LINE_STOP_NEED_STRONGER_FEATURE_OR_BASELINE"
}
```

### c13_0_c12_stop_archive/C13_0_NO_C12_6_DECISION.md
```markdown
# C13-0 No C12-6 Decision

Decision: do not enter C12-6 and do not create a C12 official candidate.

Reason: C12-5T partially recovers top100 ranking, but all score-calibration evidence remains unreliable and C12-5V hidden features regress. The span-ranker line should stop rather than receive a C12-5W patch.

Status: `C12_SPAN_RANKER_LINE_STOP_NEED_STRONGER_FEATURE_OR_BASELINE`

official_val_used = false
```

### c13_0_c12_stop_archive/C13_0_PROMOTED_SYSTEM_STATUS.md
```markdown
# C13-0 Promoted System Status

current_promoted_system = `C7-B6 R1SelectiveTop1`

C12 did not supersede the promoted system. C7-B6 / CONQUER may be used as baseline, teacher, and hard-negative source for C13, but C7-B6 fixed prediction pools must not be copied as final candidates.

official_val_used = false
```

### c13_1_strong_feature_feasibility/C13_1_FEATURE_CANDIDATES.md
```markdown
# C13-1 Feature Candidates

Status: `C13_STRONG_MULTISCALE_FEATURE_PROMISING`

Candidate feature families:

| family | candidates | feature dim | timestamp alignment | risk |
|---|---|---:|---|---|
| Text / subtitle | sentence-transformer style embedding, DeBERTa/RoBERTa/BERT re-encoding, token-level query-subtitle similarity | 768 typical | sentence offsets and clip centers must be preserved | low_to_medium |
| Visual | CLIP / EVA-CLIP / SigLIP image-video clip features, DINOv2 frame features | 768-1152 typical | feasible through sampled frame timestamps | medium_to_high |
| Motion | VideoMAE / InternVideo style motion tokens | 768-1408 typical | feasible with fixed clip windows | high |
| Multi-scale / event | clip tokens, event pooled tokens, inside/outside span context tokens | N x 768/1024 | best fit for boundary/localizer redesign | medium |

Small-sample audit requirements are represented in `C13_1_SAMPLE_COVERAGE_AUDIT.json`. The only uncovered natural bucket is `positive_not_in_b6_top100`, because the C12 CONQUER/B6 replay cache has 100% train-only GT-video top100 coverage; use native C12 missed-top100 aggregate as a hard-case proxy.

official_val_used = false
```

### c13_1_strong_feature_feasibility/C13_1_TEXT_SUBTITLE_FEATURE_PLAN.md
```markdown
# C13-1 Text Subtitle Feature Plan

Decision: strong text/subtitle features are worth a pilot.

Plan:

1. Re-encode queries and subtitle sentences with a stronger text encoder.
2. Build sentence-level and token-level query-subtitle similarity features.
3. Preserve subtitle timestamps, sentence ids, and clip-center alignment.
4. Evaluate only on train_fit/calib_select/calib_holdout sample buckets first.

Expected value: highest for t/vt queries and for failures where current score assigns high confidence to wrong boundary spans.

Risk level: low_to_medium.

official_val_used = false
```

### c13_1_strong_feature_feasibility/C13_1_VISUAL_FEATURE_PLAN.md
```markdown
# C13-1 Visual Feature Plan

Decision: strong visual features are worth a pilot, but not full extraction yet.

Plan:

1. Extract CLIP/SigLIP/EVA-CLIP style frame or short-clip embeddings for a small video sample.
2. Optionally compare DINOv2 frame descriptors when video-language models are unavailable.
3. Record frame timestamp, clip window, video id, and normalized time index.
4. Compare VR recall sample and span-ranking sample against C12 current visual bridge.

Expected value: strongest for v queries and for B6 top1 wrong cases.

Risk level: medium_to_high because decoding and dense storage dominate cost.

official_val_used = false
```

### c13_1_strong_feature_feasibility/C13_1_MULTISCALE_EVENT_FEATURE_PLAN.md
```markdown
# C13-1 Multiscale Event Feature Plan

Decision: multiscale/event features are the most promising pivot target.

Plan:

1. Pool frame/clip tokens into 8-32 event tokens per video.
2. Build span inside/outside context features over event tokens.
3. Keep both clip-level and event-level representations so boundary models can attend across scales.
4. Use C12 M1000 failure cases to test whether event tokens move high-IoU spans upward.

Expected value: directly targets the C12 failure mode where oracle span exists in M1000 but rank/calibration does not find it.

Risk level: medium.

official_val_used = false
```

### c13_1_strong_feature_feasibility/C13_1_EXTRACTION_COST_ESTIMATE.json
```json
{
  "dataset_scope": {
    "calib_holdout_queries": 4340,
    "calib_select_queries": 8677,
    "pseudo_official_holdout_used_for_selection": false,
    "train_fit_queries": 69428,
    "train_release_queries": 87175,
    "train_video_hours_estimate": 369.0386611111721,
    "train_videos": 17435
  },
  "extraction_time_estimate": {
    "full_train_visual_or_motion": "multi-day GPU job; should wait for pilot signal",
    "motion_model_pilot_gpu": "4-12 GPU hours; higher if decoding is slow",
    "text_subtitle_pilot_gpu": "minutes to <1 hour",
    "visual_clip_siglip_pilot_gpu": "1-4 GPU hours depending frame rate"
  },
  "feature_dim": {
    "clip_siglip_visual_768_to_1152": "768-1152",
    "deberta_roberta_text_768": 768,
    "event_multiscale_tokens": "N x 768/1024",
    "sentence_transformer_text_768": 768,
    "videomae_internvideo_motion_768_to_1408": "768-1408"
  },
  "gpu_cpu_cost": {
    "multiscale_event": "GPU for raw extraction, CPU ok for pooling/indexing",
    "text": "1 GPU optional; CPU possible but slower",
    "visual": "GPU recommended; CPU not practical for video/frame models"
  },
  "official_val_used": false,
  "pilot_scope": {
    "purpose": "feasibility and signal audit only; no full extraction",
    "sample_queries": 96,
    "sample_videos": 64
  },
  "risk_level": {
    "multiscale_event": "medium",
    "text_subtitle": "low_to_medium",
    "visual": "medium_to_high"
  },
  "span_ranking_sample_feasibility": "yes: reuse C12 M1000 generated pools and C12-5V failure IDs for pilot-only scoring",
  "stage": "C13-1",
  "storage_estimate": {
    "event_tokens_fp16_full_train": "smaller than dense frame tokens if pooled to 8-32 events/video",
    "subtitle_sentence_embeddings_fp16_order": "depends on subtitle sentence segmentation; expected low single-digit GiB for TVR train",
    "text_query_sentence_embeddings_fp16_all_train": "127.7 MiB",
    "visual_clip_tokens_fp16_full_train": "tens of GiB if dense frame/clip tokens are kept; small pilot <2 GiB"
  },
  "timestamp_alignment_feasibility": {
    "existing_tv_feature_stride": "current TVR features use 1.5s style clip alignment",
    "new_feature_alignment": "feasible if extraction records frame timestamp, clip center, and video duration normalization",
    "risk": "medium: subtitle sentence boundaries and video frame sampling must be canonicalized"
  },
  "vr_recall_sample_feasibility": "yes: use train_fit/calib_select/calib_holdout only, with C7-B6/CONQUER as teacher or hard-negative source"
}
```

### c13_1_strong_feature_feasibility/C13_1_FEATURE_DECISION.json
```json
{
  "full_extraction_now": false,
  "multiscale_event_worth_doing": true,
  "official_val_used": false,
  "pilot_first": true,
  "reason": "Current TVR features are insufficient for independent retrieval and reliable span scoring, while C12 M1000 has large oracle headroom. A small pilot of stronger text/subtitle, visual, and multiscale event tokens is feasible and lower risk than another C12 ranker patch.",
  "sample_coverage_audit": "C13_1_SAMPLE_COVERAGE_AUDIT.json",
  "stage": "C13-1",
  "status": "C13_STRONG_MULTISCALE_FEATURE_PROMISING",
  "strong_text_subtitle_worth_doing": true,
  "strong_visual_worth_doing": true
}
```

### c13_1_strong_feature_feasibility/C13_1_SAMPLE_COVERAGE_AUDIT.json
```json
{
  "coverage_by_duration": {
    "long": 1,
    "medium": 5,
    "short": 4
  },
  "coverage_by_query_type": {
    "t": 2,
    "v": 7,
    "vt": 1
  },
  "official_val_used": false,
  "sample_count": 12,
  "sample_policy": "train-release splits only; pseudo_official_holdout is not used for selection",
  "samples": {
    "b6_top1_correct_proxy": {
      "coverage": true,
      "desc_id": 365,
      "duration_bucket": "medium",
      "moment_duration": 8.91,
      "note": "CONQUER/B6 top1_correct proxy",
      "query_type": "v",
      "source": "first_stage_calib_holdout_top128",
      "video": "met_s03e03_seg02_clip_11"
    },
    "b6_top1_wrong_proxy": {
      "coverage": true,
      "desc_id": 366,
      "duration_bucket": "short",
      "moment_duration": 1.6499999999999986,
      "note": "CONQUER/B6 top1_wrong proxy",
      "query_type": "v",
      "source": "first_stage_calib_holdout_top128",
      "video": "met_s03e03_seg02_clip_11"
    },
    "c12_span_ranker_high_score_low_iou": {
      "coverage": true,
      "desc_id": 365,
      "duration_bucket": "medium",
      "moment_duration": 8.91,
      "note": "hidden ranker high-score low-IoU failure",
      "query_type": "v",
      "source": "C12_5V_D_FAILURE_CASES",
      "video": "met_s03e03_seg02_clip_11"
    },
    "c12_span_ranker_low_score_high_iou": {
      "coverage": true,
      "desc_id": 365,
      "duration_bucket": "medium",
      "moment_duration": 8.91,
      "note": "hidden ranker low-score high-IoU failure",
      "query_type": "v",
      "source": "C12_5V_D_FAILURE_CASES",
      "video": "met_s03e03_seg02_clip_11"
    },
    "duration_long": {
      "coverage": true,
      "desc_id": 24,
      "duration_bucket": "long",
      "moment_duration": 25.049999999999997,
      "note": "moment duration coverage",
      "query_type": "t",
      "source": "train_release",
      "video": "castle_s06e04_seg02_clip_02"
    },
    "duration_medium": {
      "coverage": true,
      "desc_id": 0,
      "duration_bucket": "medium",
      "moment_duration": 8.099999999999998,
      "note": "moment duration coverage",
      "query_type": "v",
      "source": "train_release",
      "video": "castle_s06e04_seg02_clip_06"
    },
    "duration_short": {
      "coverage": true,
      "desc_id": 1,
      "duration_bucket": "short",
      "moment_duration": 2.25,
      "note": "moment duration coverage",
      "query_type": "v",
      "source": "train_release",
      "video": "castle_s06e04_seg02_clip_06"
    },
    "native_retriever_missed_top100_proxy": {
      "coverage": true,
      "native_best_GT_video_in_top100_rate": 6.751152073732719,
      "native_best_median_rank": 1578.5,
      "note": "aggregate proxy for positive not in native top100",
      "source": "C12_4_RETRIEVER_DECISION"
    },
    "positive_not_in_b6_top100": {
      "coverage": false,
      "desc_id": null,
      "note": "blocked_by_reference_top100_replay; C12-4R replay has 100% train-only GT-video top100 coverage",
      "source": "first_stage_calib_holdout_top128"
    },
    "query_type_t": {
      "coverage": true,
      "desc_id": 4,
      "duration_bucket": "short",
      "moment_duration": 3.1499999999999986,
      "note": "query type coverage",
      "query_type": "t",
      "source": "train_release",
      "video": "castle_s06e04_seg02_clip_06"
    },
    "query_type_v": {
      "coverage": true,
      "desc_id": 0,
      "duration_bucket": "medium",
      "moment_duration": 8.099999999999998,
      "note": "query type coverage",
      "query_type": "v",
      "source": "train_release",
      "video": "castle_s06e04_seg02_clip_06"
    },
    "query_type_vt": {
      "coverage": true,
      "desc_id": 3,
      "duration_bucket": "short",
      "moment_duration": 4.5,
      "note": "query type coverage",
      "query_type": "vt",
      "source": "train_release",
      "video": "castle_s06e04_seg02_clip_06"
    }
  },
  "stage": "C13-1-sample"
}
```

### c13_2_baseline_architecture_feasibility/C13_2_PREM_FEASIBILITY.md
```markdown
# C13-2 PREM Feasibility

Status: not directly reproducible now.

Evidence:

- Local repository search found no PREM checkout.
- Public PREM repo: https://github.com/hdy007007/PREM
- Verified repo HEAD: `0cc6bd8fceae88cb89e79f71d2b22a1fb72640da`.
- The public repository README currently says: `The code is coming soon...`, so it is a README-level placeholder rather than runnable training/evaluation code.
- Paper direction is relevant: partial relevance enhancement, modality-specific pooling for VR, and focus-then-fuse localizer.

Decision: do not choose `C13_PREM_REPRO_FIRST`. Borrow PREM module ideas only.

official_val_used = false
```

### c13_2_baseline_architecture_feasibility/C13_2_MAVR_FEASIBILITY.md
```markdown
# C13-2 MA-VR Feasibility

Status: borrowable.

Local evidence:

- `c9_2_mavr_bridge/C9_2_SMOKE_RESULTS.md` reports `C9_2_MAVR_BRIDGE_SMOKE_PASS`.
- Moment-aware score tensors and gradients were already smoke-tested locally.

Decision: reuse the idea as a module candidate, not as a full mainline baseline.

official_val_used = false
```

### c13_2_baseline_architecture_feasibility/C13_2_EVENTFORMER_FEASIBILITY.md
```markdown
# C13-2 EventFormer Feasibility

Status: borrowable event-level modeling idea only.

Evidence:

- Event-aware VCMR is aligned with C12 failures because C12 has span oracle headroom but weak ranking/calibration.
- No verified runnable local/public EventFormer repository was found in this audit.

Decision: use event-level pooled tokens and event-query interaction as design input for C13 feature pilot.

official_val_used = false
```

### c13_2_baseline_architecture_feasibility/C13_2_MINUTE_FEASIBILITY.md
```markdown
# C13-2 MINUTE Feasibility

Status: borrowable training objective idea.

Evidence:

- MINUTE targets moment prediction bias and shared normalization mismatch across multiple retrieved videos.
- Reference: https://arxiv.org/abs/2301.13606
- C12 rankers still over-trust high-score low-IoU spans, so shared-normalization-style multi-video ranking is relevant.

Decision: borrow shared normalization and multimodal clue mining ideas after strong feature pilot.

official_val_used = false
```

### c13_2_baseline_architecture_feasibility/C13_2_BOUNDARY_ARCHITECTURE_OPTIONS.md
```markdown
# C13-2 Boundary Architecture Options

Need: yes, but after strong feature pilot.

Options:

1. Focus-then-fuse localizer with modality gates over subtitle, visual, and event tokens.
2. Event-token boundary decoder with inside/outside contrast and duration-aware priors.
3. Shared-normalized multi-video span ranker trained across retrieved videos.
4. Two-stage design: strong VR candidate generator, then event-aware boundary/localizer.

Do not continue with C12-5W ranker patches. C12-5V showed that adding CONQUER hidden scalar summaries to the same ranker family regresses.

official_val_used = false
```

### c13_2_baseline_architecture_feasibility/C13_2_BASELINE_DECISION.json
```json
{
  "boundary_architecture_redesign_needed": true,
  "eventformer_borrow": true,
  "mavr_borrow": true,
  "minute_borrow": true,
  "official_val_used": false,
  "prem_module_borrow": true,
  "prem_reproducible_now": false,
  "reason": "PREM public repo is not runnable from available code; MA-VR/EventFormer/MINUTE ideas are useful, but the next controllable step is native strong-feature pilot before retraining or redesigning the localizer.",
  "sources": {
    "eventformer_reference": "no verified runnable local/public repo found in this audit; borrow event-level modeling idea only",
    "local_mavr_bridge": "c9_2_mavr_bridge/C9_2_SMOKE_RESULTS.md",
    "mavr_reference": "https://doi.org/10.1109/access.2025.3542720",
    "minute_paper": "https://arxiv.org/abs/2301.13606",
    "prem_paper": "https://arxiv.org/abs/2402.13576",
    "prem_readme_observation": "README says: The code is coming soon...",
    "prem_repo": "https://github.com/hdy007007/PREM",
    "prem_repo_head_checked": "0cc6bd8fceae88cb89e79f71d2b22a1fb72640da"
  },
  "stage": "C13-2",
  "status": "C13_NATIVE_WITH_STRONG_FEATURE_FIRST"
}
```

### c13_3_next_step_decision/C13_3_NEXT_STEP_DECISION.md
```markdown
# C13-3 Next Step Decision

Selected mainline: `Path B`

Status: `C13_STRONG_FEATURE_EXTRACTION_THEN_NATIVE_RETRAIN`

Rationale:

- C12 is stopped: `C12_SPAN_RANKER_LINE_STOP_NEED_STRONGER_FEATURE_OR_BASELINE`.
- Strong text/subtitle features are worth a pilot.
- Strong visual features are worth a pilot, with higher extraction/storage risk.
- Multiscale/event features are the highest-value pivot because C12 M1000 has oracle headroom but rank/calibration fails.
- PREM is not directly reproducible from available public code, so Path A is not selected.
- MA-VR/EventFormer/MINUTE are borrowable module/objective ideas.
- Boundary/localizer redesign is likely needed, but should be guided by strong-feature pilot results.

official_val_used = false
evaluator_modified = false
nms_modified = false
```

### c13_3_next_step_decision/C13_3_NEXT_STEP_DECISION.json
```json
{
  "boundary_localizer_redesign_needed": true,
  "c12_stopped": true,
  "current_promoted_system": "C7-B6 R1SelectiveTop1",
  "evaluator_modified": false,
  "mavr_eventformer_minute_borrowable": true,
  "multiscale_event_worth_doing": true,
  "nms_modified": false,
  "official_val_used": false,
  "prem_reproducible_now": false,
  "reason": "Strong feature pilot is feasible and directly targets the observed C12 bottleneck; PREM cannot be run directly, and a full architecture rewrite should wait for pilot signal.",
  "selected_path": "Path B",
  "stage": "C13-3",
  "status": "C13_STRONG_FEATURE_EXTRACTION_THEN_NATIVE_RETRAIN",
  "strong_text_subtitle_worth_doing": true,
  "strong_visual_worth_doing": true
}
```

### c13_3_next_step_decision/C13_3_MANIFEST.json
```json
{
  "artifact_hashes": {
    "c13_0_c12_stop_archive/C13_0_C12_FINAL_STATE.json": "8d8af3d380f4467fa84370f3e80931313e5bbefd3537abb283fbcbe3b68fd44c",
    "c13_0_c12_stop_archive/C13_0_C12_FINAL_STATE.md": "c95d2f94f927579d2fd5f4fa7416db62f091026a1139d23e90f17fc25aaeb02a",
    "c13_0_c12_stop_archive/C13_0_NO_C12_6_DECISION.md": "6cda268d3048ddfb1a2e6ade92c1a80260306b07dc6069bba5e9640ebb13a3f5",
    "c13_0_c12_stop_archive/C13_0_PROMOTED_SYSTEM_STATUS.md": "c44a8b7ce286941869102fef970f6362bcb211598ba2073738a218d4968790ea",
    "c13_1_strong_feature_feasibility/C13_1_EXTRACTION_COST_ESTIMATE.json": "53de72b905193fe4f237092c2cc89a998ec8ee4693290b5ad852f93a796b95f8",
    "c13_1_strong_feature_feasibility/C13_1_FEATURE_CANDIDATES.md": "88ba8ae6dba73d672677838b1973588328b84fe1b668b4db0b133bb62d5b1558",
    "c13_1_strong_feature_feasibility/C13_1_FEATURE_DECISION.json": "723482d3650b8acee91e64824481d3f9381c8168d47bc5b79a38698fd9755b1a",
    "c13_1_strong_feature_feasibility/C13_1_MULTISCALE_EVENT_FEATURE_PLAN.md": "7396f7722385c51e28d6924c5f320bb6ad38cd43a896916b2f147054a64b93fa",
    "c13_1_strong_feature_feasibility/C13_1_SAMPLE_COVERAGE_AUDIT.json": "ead733e60da0134b9032cec557ccb49eca8f25c904b868c27484117328b23f02",
    "c13_1_strong_feature_feasibility/C13_1_TEXT_SUBTITLE_FEATURE_PLAN.md": "c8e914f1a1a2cf6b41741c8aa33cff84c591875e40c37b880ff5ec840b824b7b",
    "c13_1_strong_feature_feasibility/C13_1_VISUAL_FEATURE_PLAN.md": "633bba9607d2c089b8392f6bf6d77e81795c4b728b55d5616bb16377e0355c96",
    "c13_2_baseline_architecture_feasibility/C13_2_BASELINE_DECISION.json": "3cfd8d65eb7cebcc5f26649b1b33819eafaf84eb0582f41f52c0f1c6498efd6b",
    "c13_2_baseline_architecture_feasibility/C13_2_BOUNDARY_ARCHITECTURE_OPTIONS.md": "e9ff956831d08e5f37f82a3a157fc2e9068636693dd037e3658ed40fc908d641",
    "c13_2_baseline_architecture_feasibility/C13_2_EVENTFORMER_FEASIBILITY.md": "a211e4df8e3e6c9782b8b31700552e812d783cf6b80e975d9143c781a8e201b2",
    "c13_2_baseline_architecture_feasibility/C13_2_MAVR_FEASIBILITY.md": "6f1f79ffbe9833b47fe56fe62735bbe6b365e2e3ff3ab22b07fd1f6f72ab1443",
    "c13_2_baseline_architecture_feasibility/C13_2_MINUTE_FEASIBILITY.md": "2098710be3e14ad6893f287b45f2f36222605006926552b2ff5bdfb46f8a2cb3",
    "c13_2_baseline_architecture_feasibility/C13_2_PREM_FEASIBILITY.md": "1aed3f8652d4e002c1198651e94e0bc1db98d26a6a779a3ecf888222921a7452",
    "c13_3_next_step_decision/C13_3_MANIFEST.json": "ef6e2e18f84b0b56f6dd875d214c535456d2efc6104cbb2d740df54fc5cb6e9a",
    "c13_3_next_step_decision/C13_3_NEXT_STEP_DECISION.json": "fc3a042449baf22867417aa6b91292e830d1713da0bfa5e951e2a33007bc9c21",
    "c13_3_next_step_decision/C13_3_NEXT_STEP_DECISION.md": "4229b2f49cbb38d70c9802110045e0693041e28ef1a3622fd428317e135b42c3"
  },
  "evaluator_modified": false,
  "nms_modified": false,
  "official_val_used": false,
  "stage": "C13",
  "status": "C13_STRONG_FEATURE_EXTRACTION_THEN_NATIVE_RETRAIN"
}
```

## C12-5T Evidence: Why Stop Ranker Tuning and Pivot to C13

C12-5T T2 is the best topK selector from the C12 span-ranker line, but its score calibration remains negative and it fails the C12-6 gate. The full files are uploaded on this branch:

- `c12_5t_best_span_promotion/C12_5T_DECISION.md`
- `c12_5t_best_span_promotion/C12_5T_DECISION.json`
- `c12_5t_best_span_promotion/C12_5T_E_RESULTS.json`
- `c12_5t_best_span_promotion/C12_5T_E_RESULTS.md`

### C12_5T_DECISION.md
```markdown
# C12-5T Best-Span Promotion Decision

status = C12_BEST_SPAN_PROMOTION_STILL_WEAK_NEED_STRONGER_BOUNDARY_MODEL

best_T_variant = T2_listwise_soft_iou

best IoU@0.5 top100 = 81.2212
best IoU@0.7 top100 = 55.0000
best IoU@0.7 top50 = 38.8710
short moment IoU@0.7 top100 = 45.0909

delta_vs_c12_5s_iou07_top100 = 6.6820
allow_enter_c12_6 = false

Reason:
Selected on calib_select only. Holdout top100 IoU@0.7=55.0000; C12-5S ref=48.3180; C12-5 baseline in this run=43.0876; generated M1000 remains 85.0461; Spearman=-0.199218575616094; AUC@0.7=0.547285.

official_val_used = false
evaluator_modified = false
nms_modified = false
```

### C12_5T_DECISION.json
```json
{
  "allow_enter_c12_6": false,
  "auc_iou05": 0.534175,
  "auc_iou07": 0.547285,
  "best_iou05_top100": 81.22119815668202,
  "best_iou07_top100": 55.0,
  "best_iou07_top50": 38.87096774193548,
  "best_iou07_top500": 82.46543778801843,
  "best_iou_span_mean_rank": 194.03755760368662,
  "best_iou_span_median_rank": 119.0,
  "best_iou_span_top100_rate": 45.87557603686636,
  "best_iou_span_top10_rate": 9.67741935483871,
  "best_iou_span_top50_rate": 29.493087557603687,
  "best_ranker_variant": "T2_listwise_soft_iou",
  "candidate_video_retriever_score_semantics_fixed": true,
  "conquer_hidden_status": "C12_CONQUER_HIDDEN_NOT_AVAILABLE_USE_RAW_LOGIT_FEATURES",
  "delta_vs_c12_5s_iou05_top100": 9.562198156682015,
  "delta_vs_c12_5s_iou07_top100": 6.682000000000002,
  "evaluator_modified": false,
  "generated_M1000_iou07": 85.04608294930875,
  "nms_modified": false,
  "official_val_used": false,
  "pq_iou_pearson": -0.06382325187093174,
  "pq_iou_spearman": -0.199218575616094,
  "reason": "Selected on calib_select only. Holdout top100 IoU@0.7=55.0000; C12-5S ref=48.3180; C12-5 baseline in this run=43.0876; generated M1000 remains 85.0461; Spearman=-0.199218575616094; AUC@0.7=0.547285.",
  "short_iou07_top100": 45.09090909090909,
  "stage": "C12-5T-F",
  "status": "C12_BEST_SPAN_PROMOTION_STILL_WEAK_NEED_STRONGER_BOUNDARY_MODEL"
}
```

### C12_5T_E_RESULTS.json summary
```json
{
  "calib_holdout": {
    "C12_5_teacher_distilled_baseline": {
      "auc_iou07": 0.53702,
      "best_iou_span_top100_rate": 30.99078341013825,
      "best_iou_span_top50_rate": 18.82488479262673,
      "generated_M1000_iou07": 85.04608294930875,
      "iou05_top100": 62.995391705069125,
      "iou07_top100": 43.08755760368663,
      "iou07_top50": 30.368663594470046,
      "query_count": 4340,
      "short_iou07_top100": 30.747474747474747,
      "spearman": 0.07283213290138169
    },
    "T2_listwise_soft_iou": {
      "auc_iou07": 0.547285,
      "best_iou_span_top100_rate": 45.87557603686636,
      "best_iou_span_top50_rate": 29.493087557603687,
      "generated_M1000_iou07": 85.04608294930875,
      "iou05_top100": 81.22119815668202,
      "iou07_top100": 55.0,
      "iou07_top50": 38.87096774193548,
      "query_count": 4340,
      "short_iou07_top100": 45.09090909090909,
      "spearman": -0.199218575616094
    }
  },
  "calib_select": {
    "C12_5_teacher_distilled_baseline": {
      "auc_iou07": 0.53051,
      "best_iou_span_top100_rate": 30.4022127463409,
      "best_iou_span_top50_rate": 18.18600898928201,
      "generated_M1000_iou07": 86.03203872306096,
      "iou05_top100": 62.556183012561945,
      "iou07_top100": 42.97568283969114,
      "iou07_top50": 29.756828396911374,
      "query_count": 8677,
      "short_iou07_top100": 31.66872174270448,
      "spearman": 0.052626643010555856
    },
    "T2_listwise_soft_iou": {
      "auc_iou07": 0.621095,
      "best_iou_span_top100_rate": 46.5944450847067,
      "best_iou_span_top50_rate": 30.851676846836465,
      "generated_M1000_iou07": 86.03203872306096,
      "iou05_top100": 82.04448542122853,
      "iou07_top100": 56.58637778033883,
      "iou07_top50": 40.3826207214475,
      "query_count": 8677,
      "short_iou07_top100": 46.691327579120426,
      "spearman": -0.18574054260746647
    }
  }
}
```

## C7-B6 / C7C6-B Promoted System Evidence

Important: C12 and C13 results are train-only / feasibility artifacts. The promoted official system remains the archived C7 system. Selected small evidence files are uploaded with this packet; raw official prediction JSON files are not uploaded because they are large (`C7_OFFICIAL_PREDICTION_RAW.json` is about 206 MiB, `C7_OFFICIAL_PREDICTION_NMS.json` is about 96 MiB). Their SHA256 hashes are preserved in `C7_OFFICIAL_ONE_SHOT_RESULT.json` and `C7_OFFICIAL_ARTIFACT_HASHES.json`.

### c7_final_archive/C7_FINAL_DECISION.md
```markdown
# C7 Final Decision

```json
{
  "status": "C7_FINAL_ARCHIVED",
  "official_candidate_used": "C7C6-B Region-lite",
  "official_val_used": true,
  "official_val_count": 1,
  "post_val_adjustment": false,
  "second_official_val": false,
  "evaluator_modified": false,
  "nms_modified": false,
  "official_metrics": {
    "VCMR": {
      "0.5-r1": 14.25,
      "0.5-r5": 27.38,
      "0.5-r10": 34.06,
      "0.5-r100": 57.01,
      "0.7-r1": 8.1,
      "0.7-r5": 17.26,
      "0.7-r10": 22.71,
      "0.7-r100": 42.68
    },
    "full": {
      "VCMR": {
        "0.5-r1": 14.25,
        "0.5-r5": 27.38,
        "0.5-r10": 34.06,
        "0.5-r100": 57.01,
        "0.7-r1": 8.1,
        "0.7-r5": 17.26,
        "0.7-r10": 22.71,
        "0.7-r100": 42.68
      },
      "VCMR_by_type": {
        "v-0.5-r1": 13.1,
        "v-0.5-r5": 25.73,
        "v-0.5-r10": 32.22,
        "v-0.5-r100": 54.97,
        "v-0.7-r1": 7.29,
        "v-0.7-r5": 15.77,
        "v-0.7-r10": 21.06,
        "v-0.7-r100": 40.42,
        "t-0.5-r1": 13.8,
        "t-0.5-r5": 25.83,
        "t-0.5-r10": 31.95,
        "t-0.5-r100": 55.81,
        "t-0.7-r1": 7.68,
        "t-0.7-r5": 18.15,
        "t-0.7-r10": 23.24,
        "t-0.7-r100": 44.19,
        "vt-0.5-r1": 19.52,
        "vt-0.5-r5": 35.5,
        "vt-0.5-r10": 43.29,
        "vt-0.5-r100": 66.63,
        "vt-0.7-r1": 11.89,
        "vt-0.7-r5": 23.39,
        "vt-0.7-r10": 29.72,
        "vt-0.7-r100": 51.85,
        "desc_type_ratio": "v 74.32 t 8.85 vt 16.83"
      }
    }
  }
}
```
```

### c7_final_archive/C7_FINAL_DECISION.json
```json
{
  "evaluator_modified": false,
  "nms_modified": false,
  "official_candidate_used": "C7C6-B Region-lite",
  "official_metrics": {
    "VCMR": {
      "0.5-r1": 14.25,
      "0.5-r10": 34.06,
      "0.5-r100": 57.01,
      "0.5-r5": 27.38,
      "0.7-r1": 8.1,
      "0.7-r10": 22.71,
      "0.7-r100": 42.68,
      "0.7-r5": 17.26
    },
    "full": {
      "VCMR": {
        "0.5-r1": 14.25,
        "0.5-r10": 34.06,
        "0.5-r100": 57.01,
        "0.5-r5": 27.38,
        "0.7-r1": 8.1,
        "0.7-r10": 22.71,
        "0.7-r100": 42.68,
        "0.7-r5": 17.26
      },
      "VCMR_by_type": {
        "desc_type_ratio": "v 74.32 t 8.85 vt 16.83",
        "t-0.5-r1": 13.8,
        "t-0.5-r10": 31.95,
        "t-0.5-r100": 55.81,
        "t-0.5-r5": 25.83,
        "t-0.7-r1": 7.68,
        "t-0.7-r10": 23.24,
        "t-0.7-r100": 44.19,
        "t-0.7-r5": 18.15,
        "v-0.5-r1": 13.1,
        "v-0.5-r10": 32.22,
        "v-0.5-r100": 54.97,
        "v-0.5-r5": 25.73,
        "v-0.7-r1": 7.29,
        "v-0.7-r10": 21.06,
        "v-0.7-r100": 40.42,
        "v-0.7-r5": 15.77,
        "vt-0.5-r1": 19.52,
        "vt-0.5-r10": 43.29,
        "vt-0.5-r100": 66.63,
        "vt-0.5-r5": 35.5,
        "vt-0.7-r1": 11.89,
        "vt-0.7-r10": 29.72,
        "vt-0.7-r100": 51.85,
        "vt-0.7-r5": 23.39
      }
    }
  },
  "official_val_count": 1,
  "official_val_used": true,
  "post_val_adjustment": false,
  "second_official_val": false,
  "status": "C7_FINAL_ARCHIVED"
}
```

### c7_official_one_shot/C7_OFFICIAL_ONE_SHOT_RESULT.md
```markdown
# C7 Official One-Shot Result

```json
{
  "status": "C7_OFFICIAL_ONE_SHOT_SUCCESS",
  "official_candidate_used": "C7C6-B Region-lite",
  "fallback_used": false,
  "official_val_used": true,
  "official_val_count": 1,
  "post_val_adjustment": false,
  "second_official_val": false,
  "evaluator_modified": false,
  "nms_modified": false,
  "config_modified": false,
  "checkpoint_modified": false,
  "query_count": 10895,
  "metrics": {
    "0.5-r1": 14.25,
    "0.5-r5": 27.38,
    "0.5-r10": 34.06,
    "0.5-r100": 57.01,
    "0.7-r1": 8.1,
    "0.7-r5": 17.26,
    "0.7-r10": 22.71,
    "0.7-r100": 42.68
  },
  "top1_changed_rate_vs_anchor": 0.16888480954566315,
  "invalid_span_count": 0,
  "duplicate_span_count_after_nms": 0,
  "prediction_files": {
    "raw": "/home/a/yybwork/yybcounter/CONQUER-RLEM-c2c3/c7_official_one_shot/C7_OFFICIAL_PREDICTION_RAW.json",
    "nms": "/home/a/yybwork/yybcounter/CONQUER-RLEM-c2c3/c7_official_one_shot/C7_OFFICIAL_PREDICTION_NMS.json"
  },
  "metrics_file": "/home/a/yybwork/yybcounter/CONQUER-RLEM-c2c3/c7_official_one_shot/C7_OFFICIAL_METRICS.json",
  "artifact_hashes": {
    "raw_prediction": {
      "path": "/home/a/yybwork/yybcounter/CONQUER-RLEM-c2c3/c7_official_one_shot/C7_OFFICIAL_PREDICTION_RAW.json",
      "sha256": "df1e40a2591ea82d094d9fcc412b6b8fc982ee27d322430a7311768174620b0f"
    },
    "nms_prediction": {
      "path": "/home/a/yybwork/yybcounter/CONQUER-RLEM-c2c3/c7_official_one_shot/C7_OFFICIAL_PREDICTION_NMS.json",
      "sha256": "98725b0a204d833db42d8aaf8ca28fdf38eb818b0f08654be49bc3f0e41b11be"
    },
    "metrics": {
      "path": "/home/a/yybwork/yybcounter/CONQUER-RLEM-c2c3/c7_official_one_shot/C7_OFFICIAL_METRICS.json",
      "sha256": "ae3edfbba6cecaaf92ca2bdac2dc59db961b12769d12b665a6189d1f70f5358c"
    },
    "primary_checkpoint": {
      "path": "c7_c6_region_lite_audit/C7_C6_B_REGION_LITE_FINAL_MODEL.pt",
      "sha256": "26de62547ab6190dc08c9bdbe34366c16e6a9662a4c2d9597a1ac223433c9f51"
    },
    "evaluator": {
      "path": "standalone_eval/eval.py",
      "sha256": "7ba192993f461e92235614f5033dc5700155ca2cf61d2d0b8f98cbc211786f3a"
    },
    "nms": {
      "path": "utils/inference_utils.py",
      "sha256": "f490e62ebd98c1bb6d0054d147a7df5a5ead508cea5b26be82d1cd8ac88619cd"
    }
  },
  "diagnostics": {
    "nan_or_inf": false,
    "region_mean_prob_mean": 0.061124505806661296
  }
}
```
```

### c7_official_one_shot/C7_OFFICIAL_ONE_SHOT_RESULT.json
```json
{
  "artifact_hashes": {
    "evaluator": {
      "path": "standalone_eval/eval.py",
      "sha256": "7ba192993f461e92235614f5033dc5700155ca2cf61d2d0b8f98cbc211786f3a"
    },
    "metrics": {
      "path": "/home/a/yybwork/yybcounter/CONQUER-RLEM-c2c3/c7_official_one_shot/C7_OFFICIAL_METRICS.json",
      "sha256": "ae3edfbba6cecaaf92ca2bdac2dc59db961b12769d12b665a6189d1f70f5358c"
    },
    "nms": {
      "path": "utils/inference_utils.py",
      "sha256": "f490e62ebd98c1bb6d0054d147a7df5a5ead508cea5b26be82d1cd8ac88619cd"
    },
    "nms_prediction": {
      "path": "/home/a/yybwork/yybcounter/CONQUER-RLEM-c2c3/c7_official_one_shot/C7_OFFICIAL_PREDICTION_NMS.json",
      "sha256": "98725b0a204d833db42d8aaf8ca28fdf38eb818b0f08654be49bc3f0e41b11be"
    },
    "primary_checkpoint": {
      "path": "c7_c6_region_lite_audit/C7_C6_B_REGION_LITE_FINAL_MODEL.pt",
      "sha256": "26de62547ab6190dc08c9bdbe34366c16e6a9662a4c2d9597a1ac223433c9f51"
    },
    "raw_prediction": {
      "path": "/home/a/yybwork/yybcounter/CONQUER-RLEM-c2c3/c7_official_one_shot/C7_OFFICIAL_PREDICTION_RAW.json",
      "sha256": "df1e40a2591ea82d094d9fcc412b6b8fc982ee27d322430a7311768174620b0f"
    }
  },
  "checkpoint_modified": false,
  "config_modified": false,
  "diagnostics": {
    "nan_or_inf": false,
    "region_mean_prob_mean": 0.061124505806661296
  },
  "duplicate_span_count_after_nms": 0,
  "evaluator_modified": false,
  "fallback_used": false,
  "invalid_span_count": 0,
  "metrics": {
    "0.5-r1": 14.25,
    "0.5-r10": 34.06,
    "0.5-r100": 57.01,
    "0.5-r5": 27.38,
    "0.7-r1": 8.1,
    "0.7-r10": 22.71,
    "0.7-r100": 42.68,
    "0.7-r5": 17.26
  },
  "metrics_file": "/home/a/yybwork/yybcounter/CONQUER-RLEM-c2c3/c7_official_one_shot/C7_OFFICIAL_METRICS.json",
  "nms_modified": false,
  "official_candidate_used": "C7C6-B Region-lite",
  "official_val_count": 1,
  "official_val_used": true,
  "post_val_adjustment": false,
  "prediction_files": {
    "nms": "/home/a/yybwork/yybcounter/CONQUER-RLEM-c2c3/c7_official_one_shot/C7_OFFICIAL_PREDICTION_NMS.json",
    "raw": "/home/a/yybwork/yybcounter/CONQUER-RLEM-c2c3/c7_official_one_shot/C7_OFFICIAL_PREDICTION_RAW.json"
  },
  "query_count": 10895,
  "second_official_val": false,
  "status": "C7_OFFICIAL_ONE_SHOT_SUCCESS",
  "top1_changed_rate_vs_anchor": 0.16888480954566315
}
```

### c7_official_one_shot/C7_OFFICIAL_METRICS.json
```json
{
  "VCMR": {
    "0.5-r1": 14.25,
    "0.5-r10": 34.06,
    "0.5-r100": 57.01,
    "0.5-r5": 27.38,
    "0.7-r1": 8.1,
    "0.7-r10": 22.71,
    "0.7-r100": 42.68,
    "0.7-r5": 17.26
  },
  "full": {
    "VCMR": {
      "0.5-r1": 14.25,
      "0.5-r10": 34.06,
      "0.5-r100": 57.01,
      "0.5-r5": 27.38,
      "0.7-r1": 8.1,
      "0.7-r10": 22.71,
      "0.7-r100": 42.68,
      "0.7-r5": 17.26
    },
    "VCMR_by_type": {
      "desc_type_ratio": "v 74.32 t 8.85 vt 16.83",
      "t-0.5-r1": 13.8,
      "t-0.5-r10": 31.95,
      "t-0.5-r100": 55.81,
      "t-0.5-r5": 25.83,
      "t-0.7-r1": 7.68,
      "t-0.7-r10": 23.24,
      "t-0.7-r100": 44.19,
      "t-0.7-r5": 18.15,
      "v-0.5-r1": 13.1,
      "v-0.5-r10": 32.22,
      "v-0.5-r100": 54.97,
      "v-0.5-r5": 25.73,
      "v-0.7-r1": 7.29,
      "v-0.7-r10": 21.06,
      "v-0.7-r100": 40.42,
      "v-0.7-r5": 15.77,
      "vt-0.5-r1": 19.52,
      "vt-0.5-r10": 43.29,
      "vt-0.5-r100": 66.63,
      "vt-0.5-r5": 35.5,
      "vt-0.7-r1": 11.89,
      "vt-0.7-r10": 29.72,
      "vt-0.7-r100": 51.85,
      "vt-0.7-r5": 23.39
    }
  }
}
```

### c7_official_one_shot/C7_OFFICIAL_ARTIFACT_HASHES.json
```json
{
  "evaluator": {
    "path": "standalone_eval/eval.py",
    "sha256": "7ba192993f461e92235614f5033dc5700155ca2cf61d2d0b8f98cbc211786f3a"
  },
  "metrics": {
    "path": "/home/a/yybwork/yybcounter/CONQUER-RLEM-c2c3/c7_official_one_shot/C7_OFFICIAL_METRICS.json",
    "sha256": "ae3edfbba6cecaaf92ca2bdac2dc59db961b12769d12b665a6189d1f70f5358c"
  },
  "nms": {
    "path": "utils/inference_utils.py",
    "sha256": "f490e62ebd98c1bb6d0054d147a7df5a5ead508cea5b26be82d1cd8ac88619cd"
  },
  "nms_prediction": {
    "path": "/home/a/yybwork/yybcounter/CONQUER-RLEM-c2c3/c7_official_one_shot/C7_OFFICIAL_PREDICTION_NMS.json",
    "sha256": "98725b0a204d833db42d8aaf8ca28fdf38eb818b0f08654be49bc3f0e41b11be"
  },
  "primary_checkpoint": {
    "path": "c7_c6_region_lite_audit/C7_C6_B_REGION_LITE_FINAL_MODEL.pt",
    "sha256": "26de62547ab6190dc08c9bdbe34366c16e6a9662a4c2d9597a1ac223433c9f51"
  },
  "raw_prediction": {
    "path": "/home/a/yybwork/yybcounter/CONQUER-RLEM-c2c3/c7_official_one_shot/C7_OFFICIAL_PREDICTION_RAW.json",
    "sha256": "df1e40a2591ea82d094d9fcc412b6b8fc982ee27d322430a7311768174620b0f"
  }
}
```

### c7_final_freeze_package/C7_FINAL_DATA_PROTOCOL.md
```markdown
# C7 Final Data Protocol

- dataset: Original TVR / VCMR
- train_fit query count: 2048 for final Region checkpoint/larger confirmation
- train_calib query count: 1024 for larger confirmation
- first-stage topK videos: 100
- top-M proposals: 200
- candidate generation mode: train-only first-stage top100 CONQUER candidate pool plus integrated RLEM/Region scoring, followed by fixed R1-priority composition
- NMS threshold: 0.7
- feature release: /home/a/yybwork/data/yyb/tvr_feature_release
- base checkpoint: results/tvr-conquer_c0_repro_20260621/model.ckpt
- official evaluator path: standalone_eval/eval.py
- official split path: /home/a/yybwork/data/yyb/tvr_feature_release/data/tvr_val_release.jsonl

Train-only result is not official result. Official val has not been run in this package.
```

### c7_final_freeze_package/C7_FINAL_PRIMARY_CONFIG.json
```json
{
  "base_system": "C7C3-B checkpoint-refined RLEM candidate",
  "candidate": "C7C6-B Region-lite",
  "composition": "fixed R1-priority anchor-preserving composition",
  "official_role": "primary freeze-review candidate",
  "r1_priority_config": {
    "anchor_margin_threshold": 0.3032871469855309,
    "anchor_margin_threshold_name": "p65",
    "anchor_rank_limit": 5,
    "integrated_topn": 5,
    "max_video_promotion": 3,
    "require_in_anchor": false,
    "same_top1_video_only": true,
    "topK_gate_mode": "top5_allowed",
    "video_rank_limit": 5
  },
  "region_branch": "C7C6-B Region-lite",
  "region_lite": {
    "boundary_modulation": "disabled",
    "evaluator_modification": "disabled",
    "nms_modification": "disabled",
    "region_usage": "Region feature only -> C_prop",
    "start_end_logits_modification": "disabled",
    "variant": "C7C6-B"
  }
}
```

### c7_final_freeze_package/C7_FINAL_FREEZE_PACKAGE.md
```markdown
# C7 Final Freeze Package

Status: `C7_FINAL_FREEZE_PACKAGE_COMPLETE`

Primary: `C7C6-B Region-lite`.
Fallback: `C7C3-B checkpoint-refined RLEM candidate`.

official_val_used = false; evaluator_modified = false; nms_modified = false; official_locked = true
```

## C12-4R First-Stage Replay / zero_delta Evidence

These files prevent confusing C12/C13 train-only experiments with the promoted official system. They show C12-5 used CONQUER-warm / zero-delta replay as candidate generator, not the C7-B6 fixed prediction pool as final candidates.

### c12_4r_retriever_repair/C12_4R_A_CONQUER_FIRST_STAGE_REPLAY.md
```markdown
# C12-4R-A CONQUER First-Stage Replay

Status: `C12_4R_CONQUER_FIRST_STAGE_REPLAY_PASS`

The replay uses train first-stage ranklists as CONQUER/HERO retrieval output
over the train corpus. It does not use C7-B6 fixed prediction pools and does
not read official predictions.

```json
{
  "stage": "C12-4R-A",
  "status": "C12_4R_CONQUER_FIRST_STAGE_REPLAY_PASS",
  "calib_select_metrics": {
    "query_count": 8677,
    "missing_rank_count": 0,
    "VR_R@1": 29.66463063270716,
    "VR_R@5": 65.14924513080558,
    "VR_R@10": 79.00195920248935,
    "VR_R@100": 100.0,
    "GT_video_in_top100_rate": 100.0,
    "GT_video_median_rank": 3.0,
    "GT_video_mean_rank": 7.844531520110637
  },
  "calib_holdout_metrics": {
    "query_count": 4340,
    "missing_rank_count": 0,
    "VR_R@1": 29.101382488479263,
    "VR_R@5": 64.49308755760369,
    "VR_R@10": 79.19354838709677,
    "VR_R@100": 100.0,
    "GT_video_in_top100_rate": 100.0,
    "GT_video_median_rank": 3.0,
    "GT_video_mean_rank": 7.9633640552995395
  },
  "official_val_used": false,
  "uses_c7_b6_fixed_prediction_pool": false,
  "source": "train first-stage CONQUER/HERO ranklist LMDB"
}
```
```

### c12_4r_retriever_repair/C12_4R_A_CONQUER_FIRST_STAGE_REPLAY.json
```json
{
  "calib_holdout_metrics": {
    "GT_video_in_top100_rate": 100.0,
    "GT_video_mean_rank": 7.9633640552995395,
    "GT_video_median_rank": 3.0,
    "VR_R@1": 29.101382488479263,
    "VR_R@10": 79.19354838709677,
    "VR_R@100": 100.0,
    "VR_R@5": 64.49308755760369,
    "missing_rank_count": 0,
    "query_count": 4340
  },
  "calib_select_metrics": {
    "GT_video_in_top100_rate": 100.0,
    "GT_video_mean_rank": 7.844531520110637,
    "GT_video_median_rank": 3.0,
    "VR_R@1": 29.66463063270716,
    "VR_R@10": 79.00195920248935,
    "VR_R@100": 100.0,
    "VR_R@5": 65.14924513080558,
    "missing_rank_count": 0,
    "query_count": 8677
  },
  "official_val_used": false,
  "source": "train first-stage CONQUER/HERO ranklist LMDB",
  "stage": "C12-4R-A",
  "status": "C12_4R_CONQUER_FIRST_STAGE_REPLAY_PASS",
  "uses_c7_b6_fixed_prediction_pool": false
}
```

### c12_4r_retriever_repair/C12_4R_B_CONQUER_WARM_RETRIEVER.md
```markdown
# C12-4R-B CONQUER-Warm Retriever

Best repaired scheme: `zero_delta_replay`

The warm retriever score is:

```text
S_c12(q,v) = S_conquer_init(q,v) + alpha * delta_c12(q,v)
```

`S_conquer_init` comes from train first-stage CONQUER/HERO ranklists. The
delta head is trained only on train_fit with calib_select/calib_holdout
reporting. No official data is used.

```json
{
  "config": {
    "alpha": 0.0,
    "training": "none"
  },
  "calib_select_metrics": {
    "query_count": 8677,
    "missing_rank_count": 0,
    "VR_R@1": 29.66463063270716,
    "VR_R@5": 65.14924513080558,
    "VR_R@10": 79.00195920248935,
    "VR_R@100": 100.0,
    "GT_video_in_top100_rate": 100.0,
    "GT_video_median_rank": 3.0,
    "GT_video_mean_rank": 7.844531520110637
  },
  "calib_holdout_metrics": {
    "query_count": 4340,
    "missing_rank_count": 0,
    "VR_R@1": 29.101382488479263,
    "VR_R@5": 64.49308755760369,
    "VR_R@10": 79.19354838709677,
    "VR_R@100": 100.0,
    "GT_video_in_top100_rate": 100.0,
    "GT_video_median_rank": 3.0,
    "GT_video_mean_rank": 7.9633640552995395
  }
}
```
```

### c12_4r_retriever_repair/C12_4R_RETRIEVER_REPAIR_DECISION.md
```markdown
# C12-4R Retriever Repair Decision

Status: `C12_RETRIEVER_CONQUER_WARM_READY_CONTINUE_TO_C12_5`

Best repaired retriever variant: `zero_delta_replay`

Best learned delta variant: `delta_regularized_alpha002`

| scheme | R@1 | R@5 | R@10 | R@100 | median | teacher top1 agreement |
|---|---:|---:|---:|---:|---:|---:|
| `delta_regularized_alpha002` | 29.10 | 64.49 | 79.17 | 100.00 | 3.0 | 99.98 |
| `listwise_teacher_topK_alpha005` | 29.17 | 64.47 | 79.15 | 99.98 | 3.0 | 99.75 |
| `qtype_balanced_alpha002` | 29.10 | 64.49 | 79.17 | 100.00 | 3.0 | 99.98 |
| `stronger_teacher_topK_alpha001` | 29.10 | 64.49 | 79.17 | 100.00 | 3.0 | 100.00 |
| `zero_delta_replay` | 29.10 | 64.49 | 79.19 | 100.00 | 3.0 | 100.00 |

Interpretation:

- CONQUER first-stage replay passes and restores train-only top100 recall.
- Learned delta repair is not reliably better than zero-delta replay.
- C12-5 is allowed only if the video candidate generator is the CONQUER-warm/replay native corpus retriever, not the weak mean-pooled C12 retriever and not a C7-B6 fixed prediction pool.
- Stronger feature work remains necessary for an independent C12 retriever.

```json
{
  "stage": "C12-4R-E",
  "status": "C12_RETRIEVER_CONQUER_WARM_READY_CONTINUE_TO_C12_5",
  "conquer_first_stage_replay_pass": true,
  "conquer_warm_retriever_implemented": true,
  "best_repaired_retriever_variant": "zero_delta_replay",
  "best_learned_delta_variant": "delta_regularized_alpha002",
  "best_metrics": {
    "GT_video_in_top100_rate": 100.0,
    "GT_video_mean_rank": 7.9633640552995395,
    "GT_video_median_rank": 3.0,
    "VR_R@1": 29.101382488479263,
    "VR_R@10": 79.19354838709677,
    "VR_R@100": 100.0,
    "VR_R@5": 64.49308755760369,
    "missing_rank_count": 0,
    "query_count": 4340
  },
  "zero_delta_replay_metrics": {
    "GT_video_in_top100_rate": 100.0,
    "GT_video_mean_rank": 7.9633640552995395,
    "GT_video_median_rank": 3.0,
    "VR_R@1": 29.101382488479263,
    "VR_R@10": 79.19354838709677,
    "VR_R@100": 100.0,
    "VR_R@5": 64.49308755760369,
    "missing_rank_count": 0,
    "query_count": 4340
  },
  "best_learned_delta_metrics": {
    "GT_video_in_top100_rate": 100.0,
    "GT_video_mean_rank": 7.964516129032258,
    "GT_video_median_rank": 3.0,
    "VR_R@1": 29.101382488479263,
    "VR_R@10": 79.17050691244239,
    "VR_R@100": 100.0,
    "VR_R@5": 64.49308755760369,
    "missing_rank_count": 0,
    "query_count": 4340,
    "teacher_top1_agreement_rate": 99.97695852534562
  },
  "learned_delta_repair_effective": false,
  "teacher_distillation_repair_effective": true,
  "allow_enter_c12_5": true,
  "c12_5_entry_condition": "Use CONQUER-warm/replay native corpus video retriever as video candidate generator; do not use C7-B6 fixed prediction pool.",
  "need_stronger_features": true,
  "feature_repair_status": "C12_FEATURE_REPAIR_NOT_READY",
  "official_val_used": false,
  "evaluator_modified": false,
  "nms_modified": false,
  "notes": [
    "Replay preserves CONQUER first-stage train-only reference exactly/near-exactly.",
    "Small learned deltas did not produce robust improvement over zero-delta replay.",
    "C12-5 may proceed only with CONQUER-warm/replay video retrieval, while stronger feature work remains necessary for an independent retriever."
  ]
}
```

No official val was run, no official prediction pool was read, and evaluator/NMS were not modified.
```

### c12_4r_retriever_repair/C12_4R_RETRIEVER_REPAIR_DECISION.json
```json
{
  "allow_enter_c12_5": true,
  "best_learned_delta_metrics": {
    "GT_video_in_top100_rate": 100.0,
    "GT_video_mean_rank": 7.964516129032258,
    "GT_video_median_rank": 3.0,
    "VR_R@1": 29.101382488479263,
    "VR_R@10": 79.17050691244239,
    "VR_R@100": 100.0,
    "VR_R@5": 64.49308755760369,
    "missing_rank_count": 0,
    "query_count": 4340,
    "teacher_top1_agreement_rate": 99.97695852534562
  },
  "best_learned_delta_variant": "delta_regularized_alpha002",
  "best_metrics": {
    "GT_video_in_top100_rate": 100.0,
    "GT_video_mean_rank": 7.9633640552995395,
    "GT_video_median_rank": 3.0,
    "VR_R@1": 29.101382488479263,
    "VR_R@10": 79.19354838709677,
    "VR_R@100": 100.0,
    "VR_R@5": 64.49308755760369,
    "missing_rank_count": 0,
    "query_count": 4340
  },
  "best_repaired_retriever_variant": "zero_delta_replay",
  "c12_5_entry_condition": "Use CONQUER-warm/replay native corpus video retriever as video candidate generator; do not use C7-B6 fixed prediction pool.",
  "conquer_first_stage_replay_pass": true,
  "conquer_warm_retriever_implemented": true,
  "evaluator_modified": false,
  "feature_repair_status": "C12_FEATURE_REPAIR_NOT_READY",
  "learned_delta_repair_effective": false,
  "need_stronger_features": true,
  "nms_modified": false,
  "notes": [
    "Replay preserves CONQUER first-stage train-only reference exactly/near-exactly.",
    "Small learned deltas did not produce robust improvement over zero-delta replay.",
    "C12-5 may proceed only with CONQUER-warm/replay video retrieval, while stronger feature work remains necessary for an independent retriever."
  ],
  "official_val_used": false,
  "stage": "C12-4R-E",
  "status": "C12_RETRIEVER_CONQUER_WARM_READY_CONTINUE_TO_C12_5",
  "teacher_distillation_repair_effective": true,
  "zero_delta_replay_metrics": {
    "GT_video_in_top100_rate": 100.0,
    "GT_video_mean_rank": 7.9633640552995395,
    "GT_video_median_rank": 3.0,
    "VR_R@1": 29.101382488479263,
    "VR_R@10": 79.19354838709677,
    "VR_R@100": 100.0,
    "VR_R@5": 64.49308755760369,
    "missing_rank_count": 0,
    "query_count": 4340
  }
}
```

## Path B Feasibility: Reality Constraints

### GPU
```
NVIDIA GeForce RTX 3090, 24576 MiB, 24097 MiB, 590.48.01
NVIDIA GeForce RTX 3090, 24576 MiB, 24116 MiB, 590.48.01
```

### Disk
```
Filesystem      Size  Used Avail Use% Mounted on
/dev/sda1       1.8T  1.2T  613G  65% /home
/dev/sda1       1.8T  1.2T  613G  65% /home
/dev/nvme0n1p2  468G  104G  341G  24% /
```

### Data Availability
```
/home/a/yybwork/data/yyb/tvr_feature_release/first_stage_trained_model/model.ckpt
/home/a/yybwork/data/yyb/tvr_feature_release/data/train_select100_top2000/data.mdb
/home/a/yybwork/data/yyb/tvr_feature_release/data/train_select100_top2000/lock.mdb
/home/a/yybwork/data/yyb/tvr_feature_release/data/tvr_test_public_release.jsonl
/home/a/yybwork/data/yyb/tvr_feature_release/data/tvr_train_select100_release.jsonl
/home/a/yybwork/data/yyb/tvr_feature_release/data/tvr_train_release.jsonl
/home/a/yybwork/data/yyb/tvr_feature_release/data/tvr_val_top100_hero/data.mdb
/home/a/yybwork/data/yyb/tvr_feature_release/data/tvr_val_top100_hero/lock.mdb
/home/a/yybwork/data/yyb/tvr_feature_release/data/tvr_train_selected_release.jsonl
/home/a/yybwork/data/yyb/tvr_feature_release/data/tvr_val_release.jsonl
/home/a/yybwork/data/yyb/tvr_feature_release/data/tvr_test_public_top100_hero/data.mdb
/home/a/yybwork/data/yyb/tvr_feature_release/data/tvr_test_public_top100_hero/lock.mdb
/home/a/yybwork/data/yyb/tvr_feature_release/data/tvr_video2dur_idx.json
/home/a/yybwork/data/yyb/tvr_feature_release/sub_query_feature/tvr_query_pretrained_w_sub_query/data.mdb
/home/a/yybwork/data/yyb/tvr_feature_release/sub_query_feature/tvr_query_pretrained_w_sub_query/lock.mdb
/home/a/yybwork/data/yyb/tvr_feature_release/sub_query_feature/tvr_sub_pretrained_w_sub_query_max_cl-1.5/data.mdb
/home/a/yybwork/data/yyb/tvr_feature_release/sub_query_feature/tvr_sub_pretrained_w_sub_query_max_cl-1.5/lock.mdb
/home/a/yybwork/data/yyb/tvr_feature_release/video_feature/resnet_slowfast_1.5/data.mdb
/home/a/yybwork/data/yyb/tvr_feature_release/video_feature/resnet_slowfast_1.5/lock.mdb
```

### Raw Video File Check
```

```
No raw video files were found in the checked TVR data tree. Current local assets appear to be TVR jsonl metadata, subtitle/query LMDB features, ResNet/SlowFast LMDB visual features, and first-stage ranklist LMDB/cache files. Strong visual extraction from raw video therefore requires locating/downloading original videos or adapting Path B to pre-extracted feature augmentation first.

### Example TVR Train JSONL Row
```json
{
  "vid_name": "friends_s09e23-24_seg02_clip_33",
  "duration": 61.03,
  "ts": [
    0,
    2.75
  ],
  "desc": "Phoebe picks up a drink from the bar.",
  "type": "v",
  "desc_id": 28592
}
```

### Network Download Check
```
https://huggingface.co 200
https://github.com/openai/CLIP 200
```

## PREM / MA-VR / EventFormer / MINUTE Evidence

- PREM public repository checked: `https://github.com/hdy007007/PREM`, HEAD `0cc6bd8fceae88cb89e79f71d2b22a1fb72640da`, README says `The code is coming soon...`; therefore Path A is not currently selected.
- PREM paper: `https://arxiv.org/abs/2402.13576`.
- MA-VR local material exists in `c9_2_mavr_bridge/`, but only the bridge smoke was previously local; it is borrowable, not a promoted system.
- MINUTE paper: `https://arxiv.org/abs/2301.13606`.
- EventFormer is treated as event-level modeling idea only; no verified runnable local/public repo was found in this audit.

## What Is Uploaded Versus Not Uploaded

Uploaded or to upload with this packet:

- Complete C13 directories.
- C12-5T decision/results files.
- Selected C7 final/archive/freeze/official metrics and hash files.
- C12-4R first-stage replay and zero-delta evidence files.
- This evidence packet.

Not uploaded:

- `c7_official_one_shot/C7_OFFICIAL_PREDICTION_RAW.json` (~206 MiB).
- `c7_official_one_shot/C7_OFFICIAL_PREDICTION_NMS.json` (~96 MiB).
- Model weights and cache binaries ignored by `.gitignore` (`*.pt`, `*.ckpt`, `*.pkl`, `*.npz`, LMDB files).

## Final Handoff Interpretation

The correct execution mouth for a new window is not “prepare to enter C13”; it is “C13 has already selected Path B, but Path B must start with a constrained strong-feature pilot because raw videos are not currently visible locally.” C12-5T remains important evidence because it is the best C12 topK selector, but its negative Spearman and failed C12-6 gate are exactly why the line stopped. C7 remains the only promoted official system; C12/C13 should not be mistaken for promoted official output.
