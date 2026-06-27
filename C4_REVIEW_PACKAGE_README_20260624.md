# CONQUER-RLEM C4 Review Update (2026-06-24)

## Base package

This is an incremental review package for recipients who already have:

`CONQUER_RLEM_C4_FULL_SCAFFOLD_20260623.zip`

Expected base-package SHA256:

`d723e49c1abc98b104f1583196662edffb3e24499f892fdadd017cf6998ec2cc`

The archive preserves repository-relative paths. Reviewers can compare or overlay it on a clean checkout containing the base package. It intentionally excludes large model checkpoints, NPZ caches, scored JSONL files, and submissions; their SHA256 values are retained in the audit manifests.

## Scientific status

- Frozen C3.1 remains the upstream baseline.
- C4-lite official-val one-shot is frozen and positive.
- C4-r2-cal-v1 is retained as a weak constrained reference and was not promoted.
- C4-r2-cal-v2.1 selected `v21_00444` for robustness rather than the higher but isolated `v21_00439` score.
- `v21_00444` official-val follow-up is positive: six metrics improve and two R@1 metrics tie frozen C4-lite; both R@100 metrics improve by `+0.01`.
- No post-val adjustment, official-val grid search, temperature search, C4-main, R2/VS head, C5, or C6 was used.

## Frozen v2.1 scorer

```text
S = S_C4_lite
  + 0.075 * z(rel_logit)
  + 0.050 * z(iou07_logit)
  - 0.100 * z(quality_logit)
```

The z statistics are frozen train_calib row-expanded statistics. They are not estimated from official val.

## Modified files from the base package

- `rlem/build_c4_video_cache.py`
- `rlem/c4_lite_utils.py`
- `rlem/c4_r2_dataset.py`
- `rlem/c4_score_candidates.py`
- `rlem/c4_video_targets.py`
- `rlem/grid_search_c4_lite.py`
- `rlem/grid_search_c4_r2_calibrated.py`
- `rlem/rerank_c4_lite_one_shot.py`
- `rlem/train_c4_r2_calibrator.py`

## New C4 files after the base package

- `rlem/c4_r2_v2_model.py`
- `rlem/train_c4_r2_v2.py`
- `rlem/search_c4_r2_v2.py`
- `rlem/summarize_c4_r2_v2.py`
- `rlem/search_c4_r2_v21_weak.py`
- `rlem/rerank_c4_r2_v21_one_shot.py`

`rlem/eval_submission.py` is included as review support because it is the compatibility wrapper used for the successful bundled-evaluator run. The evaluator source itself was not modified.

## Suggested review order

1. `c4_audit/C4_LITE_TRAIN_CALIB_AUDIT.md`
2. `c4_audit/C4_LITE_OFFICIAL_VAL_ONE_SHOT_AUDIT.md`
3. `c4_audit/C4_R2_CAL_V2_FINAL_AUDIT.json`
4. `c4_audit/C4_R2_CAL_V21_WEAK_FUSION_AUDIT.json`
5. `c4_audit/C4_R2_CAL_V21_FREEZE_REVIEW.md`
6. `c4_audit/C4_R2_CAL_V21_OFFICIAL_VAL_ONE_SHOT_AUDIT.md`
7. `results/rlem_c4_r2_cal_v21_official_val/VAL_V21_ONE_SHOT_MANIFEST.json`
8. Review the new v2/v2.1 code, then the nine modified scaffold files.

## Integrity

`SHA256SUMS.txt` contains a hash for every packaged file except itself. `PACKAGE_CONTENTS.txt` is the exact sorted file list. The outer ZIP hash is reported separately next to the archive.
