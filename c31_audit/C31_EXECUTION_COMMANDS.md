# C3.1 accepted execution commands

All commands ran from `/home/a/yybcounter/CONQUER-RLEM-c2c3`. `CUDA_VISIBLE_DEVICES=1` selected the free RTX 3090 after the first five capacity configurations; both GPUs are the same model and no protocol setting changed.

```bash
python3 rlem/build_c31_metadata_cache.py \
  --evidence_jsonl results/rlem_c3_minimal/train_calib_evidence.jsonl.gz \
  --desc_ids results/rlem_c2/splits/train_calib_desc_ids.txt \
  --gt_jsonl results/rlem_c3_minimal/train_calib_gt.jsonl \
  --dataset_config config/tvr_data_config.json --split train \
  --output_npz results/rlem_c31_capacity/train_calib_metadata.npz \
  --output_manifest results/rlem_c31_capacity/train_calib_metadata_manifest.json
```

The same sweep command was used for capacity, loss, and FP JSON lists:

```bash
CUDA_VISIBLE_DEVICES=1 python3 rlem/run_c31_model_sweep.py \
  --configs_json <C31_CAPACITY_CONFIGS.json|C31_LOSS_CONFIGS.json|C31_FP_CONFIGS.json> \
  --cache_dir results/rlem_c2/cache \
  --output_root results/rlem_c31_capacity/models \
  --summary_json c31_audit/<stage>_SWEEP_SUMMARY.json \
  --epochs 3 --batch_size 1024 --lr 0.001 --device cuda --skip_existing
```

Every completed model was screened with:

```bash
python3 rlem/grid_search_c31_capacity.py \
  --metadata_npz results/rlem_c31_capacity/train_calib_metadata.npz \
  --prediction_cache results/rlem_c31_capacity/models/<id>/predictions_train_calib.npz \
  --model_id <id> --mode probe \
  --output_json results/rlem_c31_capacity/models/<id>/probe_results.json \
  --output_csv results/rlem_c31_capacity/models/<id>/probe_results.csv --workers 3
```

The three distinct-logit finalists used `--mode full --workers 6`; their top three feasible configurations per family then used `--mode calibrate --workers 6`. The one permitted local refinement used:

```bash
python3 rlem/grid_search_c31_capacity.py \
  --metadata_npz results/rlem_c31_capacity/train_calib_metadata.npz \
  --prediction_cache results/rlem_c31_capacity/models/cap_F_wd0/predictions_train_calib.npz \
  --model_id cap_F_wd0 --mode custom \
  --configs_json c31_audit/C31_LOCAL_REFINE_CONFIGS.json \
  --output_json results/rlem_c31_capacity/models/cap_F_wd0/local_refine_results.json \
  --output_csv results/rlem_c31_capacity/models/cap_F_wd0/local_refine_results.csv \
  --workers 8
```

Final freezing used `rlem/finalize_c31.py` with train-calib metadata, all 34 probe result files, the three full grids, the three calibration searches, and the one refinement result. It explicitly rejects any input path containing `official_val` or `val_evidence`.
