# C7-B3 final freeze package

```json
{
  "status": "C7_B3_OFFICIAL_READY",
  "official_val_used": false,
  "official_val_run": false,
  "post_val_adjustment": false,
  "second_official_val": false,
  "selected_candidate": "calibrated_A4",
  "selected_config": {
    "T": 0.5,
    "lambda": 1.0,
    "clip": 3.0,
    "normalization": "per_query_z"
  },
  "definition": "Shared-Norm-inspired Fixed-Pool Moment Score Calibration / Cross-Video Comparable Moment Reranking",
  "metrics": {
    "0.5-r1": 64.77857878475798,
    "0.5-r5": 77.40016020139603,
    "0.5-r10": 78.22405309531983,
    "0.5-r100": 78.36136857764046,
    "0.7-r1": 38.65430827325781,
    "0.7-r5": 66.6323377960865,
    "0.7-r10": 71.46126559102872,
    "0.7-r100": 71.99908456345119
  },
  "delta_vs_uncalibrated_A4": {
    "0.5-r1": 13.926078498684063,
    "0.5-r5": 4.886142579242474,
    "0.5-r10": 1.5333562192470538,
    "0.5-r100": 0.0,
    "0.7-r1": 7.62100926879506,
    "0.7-r5": 8.078727543197154,
    "0.7-r10": 5.034901018423156,
    "0.7-r100": 0.0
  },
  "fixed_pool_invariant_pass": true,
  "zero_control_pass": true,
  "uncalibrated_control_pass": true,
  "leakage_audit_pass": true,
  "deterministic_rerun_pass": true,
  "theory_alignment_recorded": true,
  "official_gate_preregistered": true,
  "C7_B4_SN_Audit_run": false,
  "MINUTE_style_scoring_added_to_C7_B3": false,
  "evaluator_modified": false,
  "nms_modified": false,
  "C4_C6_C7B1_C7B21_artifacts_modified": false,
  "enter_C7_C": false,
  "enter_C8": false,
  "request": "Stop and wait for human authorization for exactly-one official-val one-shot."
}
```
