# Codex instructions: C5-lite-prior implementation

## Current decision

C4 is closed as:

```text
C4_final = C4-r2-cal-v2.1 v21_00444
```

C4-lite is the core mechanism and ablation baseline, but C5 must use C4_final as its primary baseline.

Do not resume C4 grid search.  Do not enter C4-main / VS-R2 head before C5-lite-prior.

## Phase 0: merge code and syntax check

Copy the bundle files into the project root, preserving `rlem/` paths.

Then run:

```bash
python3 -m py_compile \
  rlem/score_c4_r2_v21_split.py \
  rlem/c5_prior_utils.py \
  rlem/check_c5_temporal_prior_inputs.py \
  rlem/export_c5_temporal_prior_from_jsonl.py \
  rlem/build_c5_lite_prior_features.py \
  rlem/fit_c5_lite_prior_stats.py \
  rlem/grid_search_c5_lite_prior.py \
  rlem/freeze_c5_lite_prior.py \
  rlem/rerank_c5_lite_prior_one_shot.py
```

## Phase 1: materialize C4_final train scores

Generate C4-r2-cal-v2.1 scores for train_fit and train_calib.

```bash
python3 rlem/score_c4_r2_v21_split.py \
  --cache_npz "$TRAIN_FIT_C4_CACHE_NPZ" \
  --model_ckpt "$V21_MODEL_CKPT" \
  --freeze_manifest c4_audit/C4_R2_CAL_V21_FREEZE_MANIFEST.json \
  --split_role train_fit \
  --output_logits_npz results/rlem_c5_lite_prior_prep/train_fit_v21_logits.npz \
  --output_scores_npz results/rlem_c5_lite_prior_prep/train_fit_v21_scores.npz \
  --output_audit_json c5_audit/C5_PREP_TRAIN_FIT_C4_FINAL_SCORE_AUDIT.json

python3 rlem/score_c4_r2_v21_split.py \
  --cache_npz "$TRAIN_CALIB_C4_CACHE_NPZ" \
  --model_ckpt "$V21_MODEL_CKPT" \
  --freeze_manifest c4_audit/C4_R2_CAL_V21_FREEZE_MANIFEST.json \
  --split_role train_calib \
  --output_logits_npz results/rlem_c5_lite_prior_prep/train_calib_v21_logits.npz \
  --output_scores_npz results/rlem_c5_lite_prior_prep/train_calib_v21_scores.npz \
  --output_audit_json c5_audit/C5_PREP_TRAIN_CALIB_C4_FINAL_SCORE_AUDIT.json
```

## Phase 2: C5-0 read-only temporal prior export

Export per-(query, video) temporal arrays:

```text
P_b(t), P_e(t), P_ctx(t)
```

Required final NPZs:

```text
results/rlem_c5_lite_prior_prep/train_fit_temporal_prior.npz
results/rlem_c5_lite_prior_prep/train_calib_temporal_prior.npz
```

They must match the corresponding C4 cache `group_ids_sorted_unique` order.

If exporting JSONL first, convert it using:

```bash
python3 rlem/export_c5_temporal_prior_from_jsonl.py \
  --temporal_jsonl results/rlem_c5_lite_prior_prep/train_fit_temporal_prior.jsonl.gz \
  --cache_npz "$TRAIN_FIT_C4_CACHE_NPZ" \
  --output_npz results/rlem_c5_lite_prior_prep/train_fit_temporal_prior.npz \
  --output_audit_json c5_audit/C5_0_TRAIN_FIT_TEMPORAL_PRIOR_ADAPTER_AUDIT.json \
  --split_role train_fit
```

Run the gate:

```bash
python3 rlem/check_c5_temporal_prior_inputs.py \
  --cache_npz "$TRAIN_FIT_C4_CACHE_NPZ" \
  --temporal_prior_npz results/rlem_c5_lite_prior_prep/train_fit_temporal_prior.npz \
  --output_audit_json c5_audit/C5_0_TRAIN_FIT_TEMPORAL_PRIOR_GATE.json \
  --split_role train_fit

python3 rlem/check_c5_temporal_prior_inputs.py \
  --cache_npz "$TRAIN_CALIB_C4_CACHE_NPZ" \
  --temporal_prior_npz results/rlem_c5_lite_prior_prep/train_calib_temporal_prior.npz \
  --output_audit_json c5_audit/C5_0_TRAIN_CALIB_TEMPORAL_PRIOR_GATE.json \
  --split_role train_calib
```

If this fails because P_ctx/P_b/P_e are missing, stop and patch read-only CONQUER export.  Do not fabricate temporal prior from row-level scores.

## Phase 3: aggregate temporal prior to span features

```bash
python3 rlem/build_c5_lite_prior_features.py \
  --cache_npz "$TRAIN_FIT_C4_CACHE_NPZ" \
  --c4_final_scores_npz results/rlem_c5_lite_prior_prep/train_fit_v21_scores.npz \
  --temporal_prior_npz results/rlem_c5_lite_prior_prep/train_fit_temporal_prior.npz \
  --output_features_npz results/rlem_c5_lite_prior/train_fit_prior_features.npz \
  --output_audit_json c5_audit/C5_1_TRAIN_FIT_PRIOR_FEATURE_AUDIT.json \
  --split_role train_fit \
  --clip_length 1.5

python3 rlem/build_c5_lite_prior_features.py \
  --cache_npz "$TRAIN_CALIB_C4_CACHE_NPZ" \
  --c4_final_scores_npz results/rlem_c5_lite_prior_prep/train_calib_v21_scores.npz \
  --temporal_prior_npz results/rlem_c5_lite_prior_prep/train_calib_temporal_prior.npz \
  --output_features_npz results/rlem_c5_lite_prior/train_calib_prior_features.npz \
  --output_audit_json c5_audit/C5_1_TRAIN_CALIB_PRIOR_FEATURE_AUDIT.json \
  --split_role train_calib \
  --clip_length 1.5
```

## Phase 4: freeze term stats on train_fit only

```bash
python3 rlem/fit_c5_lite_prior_stats.py \
  --features_npz results/rlem_c5_lite_prior/train_fit_prior_features.npz \
  --output_stats_json c5_audit/C5_LITE_PRIOR_TERM_STATS_FREEZE.json
```

## Phase 5: train_calib search

```bash
python3 rlem/grid_search_c5_lite_prior.py \
  --cache_npz "$TRAIN_CALIB_C4_CACHE_NPZ" \
  --features_npz results/rlem_c5_lite_prior/train_calib_prior_features.npz \
  --term_stats_json c5_audit/C5_LITE_PRIOR_TERM_STATS_FREEZE.json \
  --gt_jsonl "$TRAIN_CALIB_GT_JSONL" \
  --output_dir results/rlem_c5_lite_prior \
  --grid_mode small
```

## Phase 6: freeze review only if promotion gate passes

```bash
python3 rlem/freeze_c5_lite_prior.py \
  --train_calib_audit_json results/rlem_c5_lite_prior/c5_lite_prior_train_calib_audit.json \
  --term_stats_json c5_audit/C5_LITE_PRIOR_TERM_STATS_FREEZE.json \
  --output_review_md c5_audit/C5_LITE_PRIOR_FREEZE_REVIEW.md \
  --output_manifest_json c5_audit/C5_LITE_PRIOR_FREEZE_MANIFEST.json \
  --output_hashes_json c5_audit/C5_LITE_PRIOR_FREEZE_HASHES.json
```

Stop after this.  Do not run official val.

## Hard prohibitions

```text
No official val during C5-lite-prior train_calib development.
No C5-main.
No C6.
No CONQUER backbone modification except read-only temporal export instrumentation.
No candidate generation changes.
No NMS/evaluator changes.
No C4-r2 further tuning.
No fallback to C4-lite as primary C5 baseline.
```

## Promotion rule

C5-lite-prior can request freeze review only if:

```text
1. six primary VCMR metrics are nonnegative vs C4_final;
2. both R@100 metrics are nonnegative vs C4_final;
3. at least one localization diagnostic improves:
   - oracle-video R@1@0.5;
   - oracle-video R@1@0.7;
   - selected span mIoU;
   - best-IoU span rank;
4. movement is safe;
5. official_val_used=false;
6. C5-main/C6/CONQUER modification=false.
```
