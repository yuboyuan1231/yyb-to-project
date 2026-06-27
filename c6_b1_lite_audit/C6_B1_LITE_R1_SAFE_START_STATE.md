# C6-B1-lite R1-safe selector start

```json
{
  "stage": "C6-B1-lite R1-safe selector",
  "scope": "train_fit/train_calib only",
  "official_val_used": false,
  "post_val_adjustment": false,
  "C6_C_used": false,
  "C4_final_retained": true,
  "primary_metric_focus": "R@1",
  "secondary_safety_focus": "R@5",
  "design": "medium MLP candidate selector + conservative top-slot replacement gate",
  "model": {
    "hidden": 384,
    "layers": 4,
    "dropout": 0.1
  }
}
```
