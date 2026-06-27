# C6-B1-lite-v2 listwise start

```json
{
  "stage": "C6-B1-lite-v2 listwise slot selector (pairwise)",
  "scope": "train_fit/train_calib only",
  "official_val_used": false,
  "post_val_adjustment": false,
  "C6_C_used": false,
  "design": "listwise per-slot Transformer selector over original+boundary alternatives",
  "variant": "pairwise",
  "model": {
    "hidden": 384,
    "transformer_layers": 2,
    "heads": 8
  },
  "references": [
    "listwise learning-to-rank softmax/CE losses",
    "BMN-style proposal confidence for boundary candidates",
    "reranking precision-at-top motivation"
  ]
}
```
