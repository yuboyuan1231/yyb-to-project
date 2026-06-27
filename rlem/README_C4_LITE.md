# CONQUER-RLEM C4-lite Scaffold

## Purpose

C4-lite tests the narrow hypothesis:

```text
Can localization/span-quality evidence be aggregated into a video-level signal
that improves fixed-candidate video/cross-video ranking?
```

It does **not** train or enable a CONQUER VS/R2 head. It does **not** modify
CONQUER candidate generation, effective top-100, NMS=0.7, max-after-NMS=100, or
the evaluator.

## Stage order

1. Score fixed train-calib candidates with the frozen C3.1 checkpoint.
2. Build a C4 NPZ cache with query-video group indices.
3. Run train-calib-only C4-lite grid search.
4. Stop and report. Do not run official val without explicit authorization.

## Example commands

```bash
# 1) Create train-calib scored candidates from accepted train evidence.
python3 rlem/c4_score_candidates.py \
  --evidence_jsonl results/rlem_c1/train_evidence.jsonl.gz \
  --desc_ids_filter results/rlem_c2/splits/train_calib_desc_ids.txt \
  --ckpt results/rlem_c31_capacity/models/cap_F_wd0/model_best.pt \
  --output_jsonl results/rlem_c4_lite/train_calib_c31_scored.jsonl.gz \
  --audit_json c4_audit/C4_TRAIN_CALIB_SCORE_AUDIT.json \
  --split train \
  --device cuda \
  --score_batch_size 8192 \
  --t_joint 1.5 --t_bd 1.0 --t_fp 1.0

# 2) Cache arrays and query-video group ids.
python3 rlem/build_c4_video_cache.py \
  --scored_jsonl results/rlem_c4_lite/train_calib_c31_scored.jsonl.gz \
  --output_npz results/rlem_c4_lite/train_calib_c4_cache.npz \
  --manifest_json c4_audit/C4_CACHE_MANIFEST.json

# 3) Train-calib-only C4-lite search.
python3 rlem/grid_search_c4_lite.py \
  --cache_npz results/rlem_c4_lite/train_calib_c4_cache.npz \
  --gt_jsonl results/rlem_c3_minimal/train_calib_gt.jsonl \
  --dataset_config config/tvr_data_config.json \
  --split train \
  --output_dir results/rlem_c4_lite/search \
  --save_best_submission results/rlem_c4_lite/search/best_train_calib_submission.json \
  --effective_top_n 100 --max_after_nms 100 --nms_thd 0.7 \
  --no_desc_type
```

## Stop boundary

After train-calib search, stop. Do not run official-val C4-lite one-shot unless
it is separately authorized with a frozen `best_config.json`.
