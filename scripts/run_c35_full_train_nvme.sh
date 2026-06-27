#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="/home/a/yybwork/yybcounter/CONQUER-RLEM-c2c3"
SOURCE_DATA="/home/a/yybwork/data/yyb/tvr_feature_release"
STAGE_ROOT="/tmp/yyb_nvme_c35"
STAGE_DATA="$STAGE_ROOT/tvr_feature_release"
STAGE_INPUT="$STAGE_ROOT/input"
STAGE_OUTPUT="$STAGE_ROOT/output"
NUM_WORKERS="${NUM_WORKERS:-8}"

cd "$PROJECT_ROOT"

FINAL_OUTPUT="results/rlem_c35_qsp/train_qsp_evidence.jsonl.gz"
PUBLISH_PARTIAL="results/rlem_c35_qsp/train_qsp_evidence.jsonl.partial.gz"
STAGE_QSP="$STAGE_OUTPUT/train_qsp_evidence.jsonl.gz"
STAGE_BASE="$STAGE_INPUT/train_evidence.jsonl.gz"
STAGE_CKPT="$STAGE_INPUT/model.ckpt"

if [[ -e "$FINAL_OUTPUT" ]]; then
  echo "Refusing to overwrite accepted output: $FINAL_OUTPUT" >&2
  exit 2
fi

if pgrep -af 'rlem/export_qsp_evidence.py' >/dev/null; then
  echo "Refusing to start while another QSP exporter is running:" >&2
  pgrep -af 'rlem/export_qsp_evidence.py' >&2
  exit 2
fi

echo "[1/7] Runtime/frozen gate"
conda run --no-capture-output -n conquer-rlem \
  python scripts/check_migrated_runtime.py

echo "[2/7] Sync train-only inputs to NVMe ($STAGE_ROOT)"
mkdir -p \
  "$STAGE_DATA/sub_query_feature" \
  "$STAGE_DATA/video_feature" \
  "$STAGE_DATA/data" \
  "$STAGE_INPUT" "$STAGE_OUTPUT" \
  results/rlem_c35_qsp c35_audit

rsync -a --partial --info=progress2 \
  "$SOURCE_DATA/sub_query_feature/tvr_query_pretrained_w_sub_query" \
  "$SOURCE_DATA/sub_query_feature/tvr_sub_pretrained_w_sub_query_max_cl-1.5" \
  "$STAGE_DATA/sub_query_feature/"

rsync -a --partial --info=progress2 \
  "$SOURCE_DATA/video_feature/resnet_slowfast_1.5" \
  "$STAGE_DATA/video_feature/"

rsync -a --partial --info=progress2 \
  "$SOURCE_DATA/data/train_select100_top2000" \
  "$STAGE_DATA/data/"

rsync -a --partial --info=progress2 \
  "$SOURCE_DATA/data/tvr_train_select100_release.jsonl" \
  "$SOURCE_DATA/data/tvr_video2dur_idx.json" \
  "$STAGE_DATA/data/"

rsync -a --partial --info=progress2 \
  results/rlem_c1/train_evidence.jsonl.gz \
  results/tvr-conquer_c0_repro_20260621/model.ckpt \
  "$STAGE_INPUT/"

echo "[3/7] Verify staged inputs"
test -d "$STAGE_DATA/sub_query_feature/tvr_query_pretrained_w_sub_query"
test -d "$STAGE_DATA/sub_query_feature/tvr_sub_pretrained_w_sub_query_max_cl-1.5"
test -d "$STAGE_DATA/video_feature/resnet_slowfast_1.5"
test -d "$STAGE_DATA/data/train_select100_top2000"
test -f "$STAGE_DATA/data/tvr_train_select100_release.jsonl"
test -f "$STAGE_DATA/data/tvr_video2dur_idx.json"
cmp -s results/rlem_c1/train_evidence.jsonl.gz "$STAGE_BASE"
cmp -s results/tvr-conquer_c0_repro_20260621/model.ckpt "$STAGE_CKPT"
echo "Staged base evidence and checkpoint are byte-identical."

echo "[4/7] Full train-only QSP export on NVMe"
set -o pipefail
conda run --no-capture-output -n conquer-rlem \
  python rlem/export_qsp_evidence.py \
    --dataset_config config/tvr_data_config_nvme_c35.json \
    --base_evidence_jsonl "$STAGE_BASE" \
    --output_jsonl "$STAGE_QSP" \
    --audit_json c35_audit/C35_QSP_FULL_TRAIN_EXPORT_AUDIT.json \
    --eval_split_name train \
    --ckpt_filepath "$STAGE_CKPT" \
    --eval_query_bsz 5 \
    --max_vcmr_video 10 \
    --max_before_nms 200 \
    --num_workers "$NUM_WORKERS" \
  2>&1 | tee c35_audit/C35_QSP_FULL_TRAIN_EXPORT.log

echo "[5/7] Full train-only audit on NVMe"
conda run --no-capture-output -n conquer-rlem \
  python rlem/audit_qsp_evidence.py \
    --qsp_jsonl "$STAGE_QSP" \
    --base_jsonl "$STAGE_BASE" \
    --output_json c35_audit/C35_QSP_FULL_TRAIN_EVIDENCE_AUDIT.json \
  2>&1 | tee c35_audit/C35_QSP_FULL_TRAIN_EVIDENCE_AUDIT.log

jq -e '
  .status == "PASS" and
  .query_count == 86784 and
  .total_rows == 17356800 and
  .identity_mismatches == 0 and
  .duplicate_candidates == 0 and
  .bad_query_count_entries == 0 and
  (.nonfinite_counts | length) == 0 and
  (.range_violations | length) == 0 and
  .official_val_used == false
' c35_audit/C35_QSP_FULL_TRAIN_EVIDENCE_AUDIT.json >/dev/null

echo "[6/7] Publish accepted bytes atomically to project results"
rsync -a --partial --info=progress2 "$STAGE_QSP" "$PUBLISH_PARTIAL"
gzip -t "$PUBLISH_PARTIAL"
cmp -s "$STAGE_QSP" "$PUBLISH_PARTIAL"
mv "$PUBLISH_PARTIAL" "$FINAL_OUTPUT"

echo "[7/7] Final integrity report"
sha256sum \
  "$FINAL_OUTPUT" \
  c35_audit/C35_QSP_FULL_TRAIN_EXPORT_AUDIT.json \
  c35_audit/C35_QSP_FULL_TRAIN_EVIDENCE_AUDIT.json
conda run --no-capture-output -n conquer-rlem \
  python scripts/check_migrated_runtime.py

echo "C3.5 full train-only QSP export and audit completed. No training or val export was run."
