# C12-4R-A CONQUER First-Stage Replay

Status: `C12_4R_CONQUER_FIRST_STAGE_REPLAY_PASS`

The replay uses train first-stage ranklists as CONQUER/HERO retrieval output
over the train corpus. It does not use C7-B6 fixed prediction pools and does
not read official predictions.

```json
{
  "stage": "C12-4R-A",
  "status": "C12_4R_CONQUER_FIRST_STAGE_REPLAY_PASS",
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
  },
  "official_val_used": false,
  "uses_c7_b6_fixed_prediction_pool": false,
  "source": "train first-stage CONQUER/HERO ranklist LMDB"
}
```
