# C7 Official One-Shot Result

```json
{
  "status": "C7_OFFICIAL_ONE_SHOT_SUCCESS",
  "official_candidate_used": "C7C6-B Region-lite",
  "fallback_used": false,
  "official_val_used": true,
  "official_val_count": 1,
  "post_val_adjustment": false,
  "second_official_val": false,
  "evaluator_modified": false,
  "nms_modified": false,
  "config_modified": false,
  "checkpoint_modified": false,
  "query_count": 10895,
  "metrics": {
    "0.5-r1": 14.25,
    "0.5-r5": 27.38,
    "0.5-r10": 34.06,
    "0.5-r100": 57.01,
    "0.7-r1": 8.1,
    "0.7-r5": 17.26,
    "0.7-r10": 22.71,
    "0.7-r100": 42.68
  },
  "top1_changed_rate_vs_anchor": 0.16888480954566315,
  "invalid_span_count": 0,
  "duplicate_span_count_after_nms": 0,
  "prediction_files": {
    "raw": "/home/a/yybwork/yybcounter/CONQUER-RLEM-c2c3/c7_official_one_shot/C7_OFFICIAL_PREDICTION_RAW.json",
    "nms": "/home/a/yybwork/yybcounter/CONQUER-RLEM-c2c3/c7_official_one_shot/C7_OFFICIAL_PREDICTION_NMS.json"
  },
  "metrics_file": "/home/a/yybwork/yybcounter/CONQUER-RLEM-c2c3/c7_official_one_shot/C7_OFFICIAL_METRICS.json",
  "artifact_hashes": {
    "raw_prediction": {
      "path": "/home/a/yybwork/yybcounter/CONQUER-RLEM-c2c3/c7_official_one_shot/C7_OFFICIAL_PREDICTION_RAW.json",
      "sha256": "df1e40a2591ea82d094d9fcc412b6b8fc982ee27d322430a7311768174620b0f"
    },
    "nms_prediction": {
      "path": "/home/a/yybwork/yybcounter/CONQUER-RLEM-c2c3/c7_official_one_shot/C7_OFFICIAL_PREDICTION_NMS.json",
      "sha256": "98725b0a204d833db42d8aaf8ca28fdf38eb818b0f08654be49bc3f0e41b11be"
    },
    "metrics": {
      "path": "/home/a/yybwork/yybcounter/CONQUER-RLEM-c2c3/c7_official_one_shot/C7_OFFICIAL_METRICS.json",
      "sha256": "ae3edfbba6cecaaf92ca2bdac2dc59db961b12769d12b665a6189d1f70f5358c"
    },
    "primary_checkpoint": {
      "path": "c7_c6_region_lite_audit/C7_C6_B_REGION_LITE_FINAL_MODEL.pt",
      "sha256": "26de62547ab6190dc08c9bdbe34366c16e6a9662a4c2d9597a1ac223433c9f51"
    },
    "evaluator": {
      "path": "standalone_eval/eval.py",
      "sha256": "7ba192993f461e92235614f5033dc5700155ca2cf61d2d0b8f98cbc211786f3a"
    },
    "nms": {
      "path": "utils/inference_utils.py",
      "sha256": "f490e62ebd98c1bb6d0054d147a7df5a5ead508cea5b26be82d1cd8ac88619cd"
    }
  },
  "diagnostics": {
    "nan_or_inf": false,
    "region_mean_prob_mean": 0.061124505806661296
  }
}
```
