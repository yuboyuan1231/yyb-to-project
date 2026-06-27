#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/a/yybwork/yybcounter/CONQUER-RLEM-c2c3"
PYTHON="/home/a/miniconda3/envs/conquer-rlem/bin/python3"
CACHE="/tmp/yyb_nvme_c35/cache/c35_qsp53"
OUTPUT="$ROOT/results/rlem_c35_qsp/matched_extended_epoch8"
LOG="$ROOT/c35_audit/C35_QSP_MATCHED_EPOCH8_TRAINING.log"
DEVICE="${C35_DEVICE:-cuda:0}"

cd "$ROOT"
if [[ -d "$OUTPUT" ]] && find "$OUTPUT" -mindepth 1 -print -quit | grep -q .; then
  echo "Refusing to overwrite existing epoch-8 output: $OUTPUT" >&2
  exit 2
fi

export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8
export PYTHONUNBUFFERED=1

"$PYTHON" rlem/run_c35_matched_training.py \
  --cache_dir "$CACHE" \
  --output_dir "$OUTPUT" \
  --hidden_dims 256,256,128 \
  --dropout 0.1 \
  --epochs 8 \
  --batch_size 1024 \
  --lr 0.001 \
  --weight_decay 0.0 \
  --seed 13 \
  --shuffle_block_rows 262144 \
  --device "$DEVICE" \
  2>&1 | tee "$LOG"

echo "[done] epoch-8 matched training complete; diagnostics/probe not yet run."
