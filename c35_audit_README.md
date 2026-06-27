# C3.5-QSP implementation scaffold

This patch adds implementation-only support for C3.5-QSP evidence export and training from a matched augmented cache.

It intentionally does not authorize official-val QSP export, R2, REG, AMD, C4, C5, or C6.

Recommended train-only sequence after runtime restoration:

```bash
# 1) Export train-only augmented QSP evidence; row identity checked against accepted C1 train evidence.
python3 rlem/export_qsp_evidence.py \
  --base_evidence_jsonl results/rlem_c1/train_evidence.jsonl.gz \
  --output_jsonl results/rlem_c35_qsp/train_qsp_evidence.jsonl.gz \
  --audit_json c35_audit/C35_QSP_EXPORT_AUDIT.json \
  --eval_split_name train \
  --ckpt_filepath <C0_OFFICIAL_CKPT> \
  --eval_query_bsz 5 \
  --max_vcmr_video 10 \
  --max_before_nms 200

# 2) Audit candidate identity and finite QSP ranges.
python3 rlem/audit_qsp_evidence.py \
  --qsp_jsonl results/rlem_c35_qsp/train_qsp_evidence.jsonl.gz \
  --base_jsonl results/rlem_c1/train_evidence.jsonl.gz \
  --output_json c35_audit/C35_QSP_EVIDENCE_AUDIT.json

# 3) Fit FeatureStats on train_fit only with the c35_qsp schema using train_evidence_heads.py or a small stats-only helper.
# 4) Build cache with rlem/build_c2_feature_cache.py.
# 5) Train matched heads with rlem/train_c35_qsp.py using hidden_dims=256,256,128 and the frozen C3.1 loss/seed budget.
```
