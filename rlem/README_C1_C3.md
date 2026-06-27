# CONQUER-RLEM C1-C3 Patch

This patch implements the safe first-stage CONQUER-RLEM path:

```text
C1: frozen CONQUER fixed-candidate evidence export
C2: evidence heads Q_joint / Q_bd / E_fp
C3: evidence-calibrated VCMR reranking
```

It does **not** modify CONQUER training, candidate generation, NMS, or the official evaluator. C4/C5/C6 are intentionally left for later after C3 is verified.

## 1. Export C1 evidence

Run from the repository root:

```bash
python rlem/export_conquer_evidence.py \
  --dataset_config config/tvr_data_config.json \
  --model_dir tvr-conquer_general_paper_performance \
  --eval_split_name val \
  --output_jsonl results/rlem_c1/val_evidence.jsonl.gz \
  --max_vcmr_video 10 \
  --max_before_nms 200 \
  --eval_query_bsz 5 \
  --device 0
```

For training C2/C3, also export train or train_calib evidence:

```bash
python rlem/export_conquer_evidence.py \
  --dataset_config config/tvr_data_config.json \
  --model_dir tvr-conquer_general_paper_performance \
  --eval_split_name train \
  --output_jsonl results/rlem_c1/train_evidence.jsonl.gz \
  --max_vcmr_video 10 \
  --max_before_nms 200 \
  --eval_query_bsz 5 \
  --device 0
```

The output rows are `(q,v,p)` candidates with compact scalar evidence and labels.

## 2. Train C2 evidence heads

```bash
python rlem/train_evidence_heads.py \
  --train_jsonl results/rlem_c1/train_evidence.jsonl.gz \
  --val_jsonl results/rlem_c1/val_evidence.jsonl.gz \
  --output_dir results/rlem_c2/basic_heads \
  --epochs 3 \
  --batch_size 1024 \
  --lr 1e-3 \
  --hidden_dim 128 \
  --device cuda
```

For a quick smoke test, add limits:

```bash
  --max_train_rows 200000 --max_val_rows 50000
```

The best checkpoint is saved as:

```text
results/rlem_c2/basic_heads/model_best.pt
```

## 3. C3 rerank fixed candidates

```bash
python rlem/rerank_with_evidence.py \
  --evidence_jsonl results/rlem_c1/val_evidence.jsonl.gz \
  --ckpt results/rlem_c2/basic_heads/model_best.pt \
  --dataset_config config/tvr_data_config.json \
  --split val \
  --output_json results/rlem_c3/val_rlem_submission.json \
  --base_score_mode log \
  --a_joint 1.0 \
  --b_bd 0.5 \
  --d_fp 0.5 \
  --max_before_nms 1000 \
  --max_after_nms 100 \
  --nms_thd 0.6 \
  --device cuda
```

`base_score_mode=log` preserves the base ranking order when RLEM weights are zero and keeps the base term on a comparable scale with the 0-1 evidence heads.

## 4. Evaluate

Use the bundled evaluator:

```bash
python rlem/eval_submission.py \
  --submission_json results/rlem_c3/val_rlem_submission.json \
  --gt_jsonl /path/to/tvr_val_release.jsonl \
  --output_json results/rlem_c3/val_rlem_metrics.json
```

Or use the original CONQUER evaluation command if you already have one.

## 5. Important boundaries

- Do not change CONQUER candidate generation, NMS, or eval for C1-C3.
- `r1` is only calibrated; it is not trained.
- `Q_bd` is trained only on correct-video spans through `m_bd`.
- `E_fp` is the first-stage conflict head; it is not a full conflict taxonomy.
- C3 is evidence-calibrated ranking, not full retrieval-localization mutual promotion.
- C4/C5 only start after C3 gives a stable positive result.
