# C7-B2.1 official gate prereg

```json
{
  "status": "C7_B2_1_OFFICIAL_GATE_PREREG",
  "official_val_used": false,
  "promotion_gate": {
    "core": [
      "0.7-r1 vs C6-B2 > 0",
      "0.7-r5 vs C6-B2 >= 0",
      "0.7-r10 vs C6-B2 >= 0",
      "0.5-r1 vs C6-B2 > 0",
      "at least 5/6 R@1/R@5/R@10 core metrics vs C6-B2 non-negative"
    ],
    "safety": [
      "0.7-r100 vs C7-B1 official >= -0.10",
      "0.5-r100 vs C7-B1 official >= -0.10",
      "invalid_span_count = 0",
      "duplicate_span_count_after_nms = 0",
      "hard_positive_exit_ratio = 0 or near-zero",
      "evaluator_modified = false",
      "nms_modified = false",
      "post_val_adjustment = false",
      "second_official_val = false"
    ]
  },
  "status_labels": [
    "C7_B2_1_OFFICIAL_PROMOTED",
    "C7_B2_1_OFFICIAL_CORE_POSITIVE_REVIEW",
    "C7_B2_1_R1_TRADEOFF_NO_PROMOTION",
    "C7_B2_1_OFFICIAL_NEGATIVE",
    "C7_B2_1_INFRA_FAIL"
  ]
}
```
