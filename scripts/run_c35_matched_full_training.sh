#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/a/yybwork/yybcounter/CONQUER-RLEM-c2c3"
PYTHON="/home/a/miniconda3/envs/conquer-rlem/bin/python3"
CACHE="/tmp/yyb_nvme_c35/cache/c35_qsp53"
OUTPUT="$ROOT/results/rlem_c35_qsp/matched_full_training"
LOG="$ROOT/c35_audit/C35_QSP_MATCHED_FULL_TRAINING.log"
DEVICE="${C35_DEVICE:-cuda:0}"

cd "$ROOT"
mkdir -p "$ROOT/c35_audit"

if [[ ! -x "$PYTHON" ]]; then
  echo "Missing reproducible runtime: $PYTHON" >&2
  exit 2
fi
if [[ ! -f "$CACHE/cache_manifest.json" ]]; then
  echo "Missing canonical NVMe cache manifest: $CACHE/cache_manifest.json" >&2
  exit 2
fi
if [[ -d "$OUTPUT" ]] && find "$OUTPUT" -mindepth 1 -print -quit | grep -q .; then
  echo "Refusing to overwrite existing full-training output: $OUTPUT" >&2
  exit 2
fi

echo "[preflight] reproducible runtime"
"$PYTHON" scripts/check_migrated_runtime.py
echo "[preflight] GPUs"
nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv,noheader

# CPU-side index generation and library operations may use up to eight threads.
# The full feature/label tensors remain resident on the selected GPU.
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8
export PYTHONUNBUFFERED=1

echo "[train] device=$DEVICE output=$OUTPUT"
"$PYTHON" rlem/run_c35_matched_training.py \
  --cache_dir "$CACHE" \
  --output_dir "$OUTPUT" \
  --hidden_dims 256,256,128 \
  --dropout 0.1 \
  --epochs 3 \
  --batch_size 1024 \
  --lr 0.001 \
  --weight_decay 0.0 \
  --seed 13 \
  --shuffle_block_rows 262144 \
  --device "$DEVICE" \
  2>&1 | tee "$LOG"

echo "[done] C3.5 matched full training completed. No score grid or official-val stage was run."
