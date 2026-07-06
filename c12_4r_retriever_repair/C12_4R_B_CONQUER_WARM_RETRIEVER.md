# C12-4R-B CONQUER-Warm Retriever

Best repaired scheme: `zero_delta_replay`

The warm retriever score is:

```text
S_c12(q,v) = S_conquer_init(q,v) + alpha * delta_c12(q,v)
```

`S_conquer_init` comes from train first-stage CONQUER/HERO ranklists. The
delta head is trained only on train_fit with calib_select/calib_holdout
reporting. No official data is used.

```json
{
  "config": {
    "alpha": 0.0,
    "training": "none"
  },
  "calib_select_metrics": {
    "query_count": 8677,
    "missing_rank_count": 0,
    "VR_R@1": 29.66463063270716,
    "VR_R@5": 65.14924513080558,
    "VR_R@10": 79.00195920248935,
    "VR_R@100": 100.0,
    "GT_video_in_top100_rate": 100.0,
    "GT_video_median_rank": 3.0,
    "GT_video_mean_rank": 7.844531520110637
  },
  "calib_holdout_metrics": {
    "query_count": 4340,
    "missing_rank_count": 0,
    "VR_R@1": 29.101382488479263,
    "VR_R@5": 64.49308755760369,
    "VR_R@10": 79.19354838709677,
    "VR_R@100": 100.0,
    "GT_video_in_top100_rate": 100.0,
    "GT_video_median_rank": 3.0,
    "GT_video_mean_rank": 7.9633640552995395
  }
}
```
