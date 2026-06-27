# C7-B2.1 deterministic rerun

```json
{
  "status": "PASS",
  "official_val_used": false,
  "selected_variant": "A4_video_residual_only",
  "metrics": {
    "0.5-r1": 50.85250028607392,
    "0.5-r5": 72.51401762215356,
    "0.5-r10": 76.69069687607278,
    "0.5-r100": 78.36136857764046,
    "0.7-r1": 31.03329900446275,
    "0.7-r5": 58.55361025288935,
    "0.7-r10": 66.42636457260556,
    "0.7-r100": 71.99908456345119
  },
  "expected_metrics": {
    "0.5-r1": 50.85250028607392,
    "0.5-r5": 72.51401762215356,
    "0.5-r10": 76.69069687607278,
    "0.5-r100": 78.36136857764046,
    "0.7-r1": 31.03329900446275,
    "0.7-r5": 58.55361025288935,
    "0.7-r10": 66.42636457260556,
    "0.7-r100": 71.99908456345119
  },
  "metric_abs_diff_vs_C7_B2_1_FINAL_DECISION": {
    "0.5-r1": 0.0,
    "0.5-r5": 0.0,
    "0.5-r10": 0.0,
    "0.5-r100": 0.0,
    "0.7-r1": 0.0,
    "0.7-r5": 0.0,
    "0.7-r10": 0.0,
    "0.7-r100": 0.0
  },
  "metric_hash": "f407e93abee75f09ccc9be2c88f68ca0450f350be93d1f2561fbec1d3a618cf8",
  "prediction_hash": "ddfc7cb80ce348a15022882d51d48daddcac0277908e856415ced379ccf60ad2",
  "score_hash": "a9284a17b526bbadcaa0167510b32060b9a1305eb51446f7e28b460df960699e",
  "model_hash": {
    "path": "results/rlem_c7_b2/c7_b2_mil_video_residual_head.pt",
    "exists": true,
    "size": 1276979,
    "sha256": "8db8ae4b65737f166e66a023b36c50312e2ee2bc78f50088a9b87c78e00e86de"
  },
  "config_hash": "c4d0f50d5a44e027d9c9d78a4bf5c18e66c47a06a0367ea32cde21231a2336cd",
  "prediction_file": {
    "path": "results/rlem_c7_b2_1/freeze/selected_A4_predictions.jsonl",
    "exists": true,
    "size": 26812243,
    "sha256": "ddfc7cb80ce348a15022882d51d48daddcac0277908e856415ced379ccf60ad2"
  },
  "score_file": {
    "path": "results/rlem_c7_b2_1/freeze/selected_A4_scores.jsonl",
    "exists": true,
    "size": 16178547,
    "sha256": "a9284a17b526bbadcaa0167510b32060b9a1305eb51446f7e28b460df960699e"
  }
}
```
