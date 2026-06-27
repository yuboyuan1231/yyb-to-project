# C6-C1 feature audit

```json
{
  "status": "C6_C1_FEATURE_AUDIT_PASS",
  "official_val_used": false,
  "feature_sources": [
    "C4_final train_fit/train_calib cache rows",
    "C6-B1-lite candidate selector scores as stable span evidence",
    "C6-B2 pairwise_main scores as R1 utility evidence only",
    "C5 temporal prior and C6 boundary/proposal priors",
    "C6 cache video aggregate features"
  ],
  "forbidden_features_excluded": [
    "GT IoU in feature x",
    "hit labels in feature x",
    "train_calib label in feature x",
    "official-val metrics/predictions",
    "oracle rerank result"
  ],
  "model_capacity": {
    "hidden": 512,
    "layers": 4,
    "heads": [
      "relevance",
      "moment",
      "iou",
      "risk",
      "gate"
    ]
  }
}
```
