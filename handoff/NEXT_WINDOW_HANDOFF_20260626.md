# CONQUER-RLEM current handoff for next window

Date: 2026-06-26  
Project root: `/home/a/yybwork/yybcounter/CONQUER-RLEM-c2c3`

This document is the recommended starting point for a fresh Codex window. It summarizes the current promoted result, the experimental path that led here, where the audit files live, and what must not be accidentally rerun.

## 1. Current promoted system

Current promoted system is:

```text
C6-B2 official-val one-shot
selected_config = pairwise_main_0005
status = C6_B2_OFFICIAL_POSITIVE
```

Frozen policy:

```text
arm = pairwise_main
config_id = pairwise_main_0005
apply_slots = 5
threshold0 = 4.337798070907593
threshold_rest = 6.729127645492554
max_replacements = 2
effective_top_n = 100
max_after_nms = 100
NMS = 0.7
```

Official-val metrics:

| system | 0.5-r1 | 0.5-r5 | 0.5-r10 | 0.5-r100 | 0.7-r1 | 0.7-r5 | 0.7-r10 | 0.7-r100 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| C4_final v21_00444 | 14.81 | 28.57 | 34.81 | 46.45 | 8.35 | 18.64 | 24.63 | 35.70 |
| C6-B1-lite c6b1r1_0057 | 14.92 | 28.66 | 34.93 | 46.56 | 8.39 | 18.82 | 24.82 | 35.88 |
| C6-B2 pairwise_main_0005 | 14.98 | 28.66 | 34.90 | 46.54 | 8.45 | 18.78 | 24.76 | 35.82 |
| delta vs C6-B1 | +0.06 | +0.00 | -0.03 | -0.02 | +0.06 | -0.04 | -0.06 | -0.06 |
| delta vs C4_final | +0.17 | +0.09 | +0.09 | +0.09 | +0.10 | +0.14 | +0.13 | +0.12 |

Interpretation:

- C6-B2 improves the primary R@1 metrics over C6-B1-lite.
- R@5/R@10/R@100 have only tiny declines vs C6-B1-lite, not collapse.
- C6-B2 remains positive vs C4_final on all eight metrics.
- Classification is `C6_B2_OFFICIAL_POSITIVE`.

Official-val movement:

```text
replacement_rate_top_slots = 0.006773749426342359
candidate_replacement_count = 369
top1_changed_ratio_vs_C6_B1 = 0.03129876089949518
top1_changed_ratio_vs_C4_final = 0.01101422670949977
hard_positive exits/entries vs C6_B1 = 9 / 6
hard_positive exits/entries vs C4_final = 1 / 10
video_slot_drift = 0.0
video_multiset_drift = 0.0
invalid_span_count = 0
duplicate_span_count_after_nms = 0
```

## 2. Most important files for current promoted result


C6-B2 final promoted-system files:

```text
c6_b2_final/C6_B2_FINAL_SUMMARY.md
c6_b2_final/C6_B2_FINAL_MANIFEST.json
c6_b2_final/C6_B2_FINAL_HASHES.json
```

C6-B2 official-val one-shot audit:

```text
c6_b2_official_val/C6_B2_OFFICIAL_VAL_ONE_SHOT_AUDIT.md
c6_b2_official_val/C6_B2_OFFICIAL_VAL_ONE_SHOT_MANIFEST.json
c6_b2_official_val/C6_B2_OFFICIAL_VAL_ONE_SHOT_HASHES.json
c6_b2_official_val/C6_B2_OFFICIAL_VAL_PREFLIGHT_AUDIT.md
c6_b2_official_val/C6_B2_OFFICIAL_VAL_PREFLIGHT_AUDIT.json
```

C6-B2 official-val artifacts:

```text
results/rlem_c6_b2_official_val/official_val_scores.npz
results/rlem_c6_b2_official_val/official_val_predictions.jsonl
results/rlem_c6_b2_official_val/official_val_submission.json
results/rlem_c6_b2_official_val/official_val_metrics.json
```

C6-B2 freeze/train_calib material:

```text
c6_b2_audit/C6_B2_FINAL_SUMMARY.md
c6_b2_audit/C6_B2_FREEZE_REVIEW.md
c6_b2_audit/C6_B2_FREEZE_MANIFEST.json
c6_b2_audit/C6_B2_FREEZE_HASHES.json
c6_b2_audit/C6_B2_ARM_COMPARISON.md
c6_b2_audit/C6_B2_TRAINING_AUDIT.md
c6_b2_audit/C6_B2_SAFETY_AUDIT.md
c6_b2_audit/C6_B2_COMPLETION_AUDIT.md
c6_b2_audit/C6_B2_ROBUSTNESS_AUDIT.md
c6_b2_audit/C6_B2_P0_POOL_AUDIT.md
```

C6-B2 model/code:

```text
rlem_c6_b2/run_c6_b2_goal.py
rlem_c6_b2/run_c6_b2_official_val_one_shot.py
results/rlem_c6_b2/arms/pairwise_main/model_best.pt
results/rlem_c6_b2/train_calib_scores/pairwise_main/best_config.json
results/rlem_c6_b2/train_calib_scores/pairwise_main/grid_results.json
```

Design review / blueprint used for C6-B2:

```text
c6_b2_design_review/C6_B2_PROTOCOL_DESIGN_REVIEW.md
c6_b2_design_review/C6_B2_PROTOCOL_DESIGN_REVIEW_MANIFEST.json
c6_b2_design_review/C6_B2_PROTOCOL_DESIGN_REVIEW_HASHES.json
```

## 3. Stage history and decisions

### C3/C3.1/C3.5

- C3.1 official-val one-shot was frozen as an earlier main result.
- C3.5-QSP matched training was negative and was not promoted.

Useful files:

```text
c31_audit/C31_FROZEN_MANIFEST.json
results/rlem_c31_official_val/VAL_C31_ONE_SHOT_MANIFEST.json
c35_audit/C35_QSP_NEGATIVE_DECISION.json
c35_audit/C35_QSP_FROZEN_MANIFEST.json
```

### C4

Final C4 system is not plain C4-lite. It is:

```text
C4_final = C4-r2-cal-v2.1 v21_00444
```

C4-lite is the core mechanism and major ablation baseline. C4-r2-cal-v2.1 is the final frozen C4 output.

Frozen C4 final scorer:

```text
S_C4_final = S_C4_lite
  + 0.075 * z(rel_logit)
  + 0.050 * z(iou07_logit)
  - 0.100 * z(quality_logit)

b_iou05 = 0.0
```

Important C4 files:

```text
c4_audit/C4_STAGE_FINAL_SUMMARY.md
c4_audit/C4_STAGE_FINAL_MANIFEST.json
c4_audit/C4_STAGE_FINAL_HASHES.json
c4_audit/C4_R2_CAL_V21_FREEZE_REVIEW.md
c4_audit/C4_R2_CAL_V21_FREEZE_MANIFEST.json
c4_audit/C4_R2_CAL_V21_FREEZE_HASHES.json
c4_audit/C4_R2_CAL_V21_OFFICIAL_VAL_ONE_SHOT_AUDIT.md
results/rlem_c4_r2_cal_v21_official_val/VAL_V21_ONE_SHOT_MANIFEST.json
results/rlem_c4_r2_cal_v21_official_val/val_v21_metrics.json
results/rlem_c4_r2_cal_v21_official_val/val_v21_submission.json
```

C4-main / VS-R2 head were not used; they were left for later integrated phases, not forgotten.

### C5

C5-lite-prior was negative. Several C5-main-inference attempts explored how to safely transfer retrieval evidence into localization.

Most useful C5 files:

```text
c5_audit/C5_LITE_PRIOR_NEGATIVE_AUDIT.md
c5_audit/C5_LITE_PRIOR_NEGATIVE_MANIFEST.json
c5_fd_audit/C5_FAILURE_DECOMPOSITION_SUMMARY.md
c5_main_a_audit/C5_MAIN_A_NEGATIVE_AUDIT.md
c5_main_a2_compare/C5_MAIN_A2_COMPARISON_SUMMARY.md
c5_main_a3_compare/C5_MAIN_A3_COMPARISON_SUMMARY.md
c5_main_a3_video_slot_audit/C5_MAIN_A3_FREEZE_REVIEW.md
c5_main_a3_video_slot_audit/C5_MAIN_A3_FREEZE_MANIFEST.json
c5_main_a3_video_slot_official_val/C5_MAIN_A3_OFFICIAL_VAL_ONE_SHOT_AUDIT.md
c5_main_a3_video_slot_official_val/C5_MAIN_A3_OFFICIAL_VAL_ONE_SHOT_MANIFEST.json
```

Takeaway:

- Naive C5 prior/residual can hurt front-rank metrics.
- A3-style video-slot candidate replacement pointed toward same-video replacement as the safer bridge.

### C6-A / C6-A-R / C6-A-R2

Adapter-only / fixed-candidate reranking lines were negative for final promotion, even though boundary learning improved some oracle-localization signals.

Important files:

```text
c6_audit/C6A_EXECUTION_REPORT.md
c6_audit/C6A_NEGATIVE_AUDIT.md
c6_repair_audit/C6A_R_NEGATIVE_AUDIT.md
c6_repair2_audit/CURRENT_STAGE_SUMMARY.md
c6_repair2_audit/C6A_R2_NEGATIVE_AUDIT.md
c6_repair2_audit/C6A_R2_NEGATIVE_MANIFEST.json
```

Takeaway:

- BoundaryOracleAdapter can improve boundary/oracle-video localization.
- Fixed-candidate adapter reranking could not safely translate that into stable VCMR front-rank gains.
- This motivated C6-B0/B1/B2 candidate-generation / replacement selector work.

### C6-B0

C6-B0 was a candidate-generation diagnostic protocol. It tested whether new boundary distributions can generate better candidates under fixed video slots.

Files:

```text
c6_b0_audit/C6_B0_CANDIDATE_DIAGNOSTIC.md
c6_b0_audit/C6_B0_FINAL_SUMMARY.md
c6_b0_audit/C6_B0_MANIFEST.json
c6_b0_audit/C6_B0_HASHES.json
```

Takeaway:

- Same-video candidate replacement has real oracle headroom.
- Do not treat B0 as an official final result.

### C6-B1-lite

C6-B1-lite was the first clean official-val positive after multiple negative lines.

Official status:

```text
C6_B1_LITE_OFFICIAL_VAL_STRONG_POSITIVE
selected_config = c6b1r1_0057
```

Official-val metrics vs C4_final:

```text
0.5-r1 +0.11
0.5-r5 +0.09
0.5-r10 +0.12
0.5-r100 +0.11
0.7-r1 +0.04
0.7-r5 +0.18
0.7-r10 +0.19
0.7-r100 +0.18
```

Files:

```text
c6_b1_lite_audit/C6_B1_LITE_R1_SAFE_FREEZE_REVIEW.md
c6_b1_lite_audit/C6_B1_LITE_R1_SAFE_FREEZE_MANIFEST.json
c6_b1_lite_audit/C6_B1_LITE_R1_SAFE_FREEZE_HASHES.json
c6_b1_lite_official_val/C6_B1_LITE_OFFICIAL_VAL_ONE_SHOT_AUDIT.md
c6_b1_lite_official_val/C6_B1_LITE_OFFICIAL_VAL_ONE_SHOT_MANIFEST.json
c6_b1_lite_final/C6_B1_LITE_FINAL_SUMMARY.md
c6_b1_lite_final/C6_B1_LITE_FINAL_MANIFEST.json
```

Artifacts:

```text
results/rlem_c6_b1_lite_r1_safe/model_best.pt
results/rlem_c6_b1_lite_r1_safe/best_config.json
results/rlem_c6_b1_lite_r1_safe/official_val_scores.npz
results/rlem_c6_b1_lite_r1_safe/official_val_predictions.jsonl
results/rlem_c6_b1_lite_r1_safe/official_val_submission.json
results/rlem_c6_b1_lite_r1_safe/official_val_metrics.json
results/rlem_c6_b1_lite_r1_safe/official_val_c6_cache.npz
results/rlem_c6_b1_lite_r1_safe/official_val_candidate_pool_top_slots.npz
```

### C6-B2

C6-B2 started from the design review:

```text
c6_b2_design_review/C6_B2_PROTOCOL_DESIGN_REVIEW.md
```

Protocol philosophy:

- R@1 is the primary target, especially 0.7 R@1.
- R@5/R@10 are guardrails; modest declines are acceptable, collapse is not.
- R@100 is reported, not optimized.
- Same-video replacement only; fixed video slots; unchanged NMS/evaluator.
- If an arm underperforms, audit and allow at most two retries.

Arms tried:

```text
pairwise_main          -> selected, freeze pass
listwise_set           -> failed
listwise_set_retry1    -> failed
listwise_set_retry2    -> improved but still failed; retry budget exhausted
```

C6-B2 train_calib freeze result:

```text
selected_config = pairwise_main_0005
C6_B2_FREEZE_REVIEW_PASS
```

Train_calib delta vs C6-B1-lite:

```text
0.7-r1 +0.389061
0.5-r1 +0.240302
0.7-r5 +0.045772
0.5-r5 -0.011443
0.7-r10 +0.057215
0.5-r10 +0.011443
0.7-r100 +0.057215
0.5-r100 -0.022886
```

Official-val result is now positive as described in section 1.

## 4. Important code paths

C4:

```text
rlem/c4_protocol.py
rlem/c4_lite_utils.py
rlem/c4_score_candidates.py
rlem/build_c4_video_cache.py
rlem/grid_search_c4_lite.py
rlem/rerank_c4_lite_one_shot.py
rlem/c4_r2_dataset.py
rlem/c4_r2_model.py
rlem/train_c4_r2_calibrator.py
rlem/grid_search_c4_r2_calibrated.py
```

C5:

```text
rlem/grid_search_c5_main_a3.py
rlem_c6_b0/run_c6_b0_candidate_diagnostic.py
```

C6:

```text
rlem_c6/build_c6_cache.py
rlem_c6/c6a_utils.py
rlem_c6_b1_lite/run_c6_b1_r1_safe_selector.py
rlem_c6_b1_lite/run_c6_b1_official_val_one_shot.py
rlem_c6_b1_lite/finalize_c6_b1_lite.py
rlem_c6_b2/run_c6_b2_goal.py
rlem_c6_b2/run_c6_b2_official_val_one_shot.py
```

External evaluator:

```text
standalone_eval/eval.py
```

Do not modify evaluator unless explicitly authorized.

## 5. Current safety / boundary state

For C6-B2 official val:

```text
official_val_one_shot = true
successful_metric_result_count = 1
learned_config_count_on_val = 0
score_grid_on_val = false
temperature_search_on_val = false
post_val_adjustment = false
training_used = false
selector_retrained = false
C6_C_used = false
C4_final_modified = false
C6_B1_lite_modified = false
NMS_modified = false
evaluator_modified = false
no_second_official_val = true
```

Important: do not rerun C6-B2 official val unless the user explicitly authorizes a new run and accepts that it would no longer be the original one-shot. The existing one-shot has already succeeded.

## 6. What a new window should do first

Recommended first commands:

```bash
cd /home/a/yybwork/yybcounter/CONQUER-RLEM-c2c3
sed -n '1,220p' handoff/NEXT_WINDOW_HANDOFF_20260626.md
sed -n '1,220p' c6_b2_official_val/C6_B2_OFFICIAL_VAL_ONE_SHOT_AUDIT.md
sed -n '1,220p' c6_b2_official_val/C6_B2_OFFICIAL_VAL_ONE_SHOT_MANIFEST.json
sed -n '1,220p' c6_b2_audit/C6_B2_FINAL_SUMMARY.md
```

Optional integrity checks:

```bash
python3 - <<'PY'
import json, pathlib
for p in pathlib.Path('c6_b2_official_val').glob('*.json'):
    json.loads(p.read_text())
print('C6-B2 official-val JSON OK')
PY
find results/rlem_c6_b2_official_val c6_b2_official_val -name '*.partial'
```

Expected `.partial` result: no output.

## 7. Suggested next blueprint

Since C6-B2 is official-positive, the next stage should probably be a design-review phase rather than immediately launching an aggressive new experiment.

Reasonable next-stage options:

1. `C6-B2 finalization`: completed. See `c6_b2_final/C6_B2_FINAL_SUMMARY.md`, `c6_b2_final/C6_B2_FINAL_MANIFEST.json`, and `c6_b2_final/C6_B2_FINAL_HASHES.json`.
2. `C6-C design review`: recommended next active step. It should focus on video-aware mutual integration / retrieval-localization coupling.
3. `C6-B3 cautious extension`: only if explicitly requested. It should not rerun official val casually and would need a fresh train_calib-only design review first.

My recommendation:

```text
C6-B2 is now frozen as the current promoted system.
Next open C6-C design review focused on retrieval-localization mutualization.
Do not immediately train C6-C until the protocol is explicitly reviewed.
```

Why C6-C is the natural next conceptual step:

- C6-B0/B1/B2 prove same-video replacement can safely improve front rank.
- But the absolute official 0.7 R@1 is still 8.45, below recent-method target territory.
- Remaining headroom likely requires coupling video retrieval with moment localization, not only replacing spans inside fixed video slots.

## 8. Things not to accidentally do

Do not:

- rerun C6-B2 official val;
- grid search on official val;
- train on official val;
- do post-val adjustment;
- change `effective_top_n=100`, `max_after_nms=100`, or `NMS=0.7`;
- modify `standalone_eval/eval.py`;
- modify `model/conquer.py`;
- overwrite C4_final / C6-B1 / C6-B2 frozen artifacts;
- present train_calib-only positives as official positives;
- continue listwise retries for C6-B2 without a new protocol, because the two-retry budget was exhausted.

## 9. Useful environment details

Python used in the successful runs:

```text
/home/a/miniconda3/envs/conquer-rlem/bin/python3
```

GPU use:

- C6-B2 official-val scoring used `CUDA_VISIBLE_DEVICES=0` and `--device cuda`.
- Two RTX 3090s were available during prior runs.

Project code package history:

- C4 full scaffold zip source was `/home/a/yybwork/yybcounter/fix/CONQUER_RLEM_C4_FULL_SCAFFOLD_20260623.zip`.
- C5-lite prior code bundle was `/home/a/yybwork/yybcounter/fix/CONQUER_RLEM_C5_LITE_PRIOR_CODE_BUNDLE_20260624.zip`.

## 10. Short summary for new window

If you need the one-paragraph version:

```text
The current promoted result is finalized as C6-B2 pairwise_main_0005, an official-val one-shot positive over C6-B1-lite and C4_final. It uses a frozen pairwise replacement selector over the existing C6-B1 same-video candidate pool, fixed video slots, NMS=0.7, effective_top_n=max_after_nms=100. Official val was run exactly once for this config, with no retraining, no threshold search, no post-val adjustment, and no C6-C. It improves official 0.7 R@1 from 8.39 to 8.45 and 0.5 R@1 from 14.92 to 14.98 vs C6-B1, while staying positive on all metrics vs C4_final. Next recommended action: C6-B2 finalization is complete; open C6-C design review for retrieval-localization mutualization.
```
