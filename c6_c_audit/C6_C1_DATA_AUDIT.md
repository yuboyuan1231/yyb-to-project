# C6-C1 data audit

```json
{
  "status": "C6_C1_DATA_READY",
  "official_val_used": false,
  "feature_policy": "No GT IoU, hit labels, train_calib labels, official metrics, or oracle outputs are included in x. Labels are stored separately for training/eval only.",
  "modality_features_unavailable": true,
  "b1_train_scores": {
    "reused": false,
    "score": {
      "path": "results/rlem_c6_c/aux/train_fit_b1_candidate_scores.npz",
      "exists": true,
      "size": 11551386,
      "sha256": "b9e45e026c5676790993298f2aa9beff54ffa3c3447ffd3844c4234070837c92"
    },
    "score_min": -7.709225654602051,
    "score_max": 6.353497505187988
  },
  "b1_calib_scores": {
    "reused": true,
    "score": {
      "path": "results/rlem_c6_b1_lite_r1_safe/train_calib_candidate_scores.npz",
      "exists": true,
      "size": 1292852,
      "sha256": "0ee937a8cf0a388d03fbd1882a1b37240eb0416d2f57405d55b7451325c6e52e"
    }
  },
  "b2_train_scores": {
    "reused": false,
    "score": {
      "path": "results/rlem_c6_c/aux/train_fit_b2_pairwise_main_scores.npz",
      "exists": true,
      "size": 20712107,
      "sha256": "883b3bc9b6f3782237de56a91369596b6dc263a2f4e37bf66b4ef5c6868f85e5"
    },
    "score_min": -13.735554695129395,
    "score_max": 10.282471656799316
  },
  "b2_calib_scores": {
    "reused": true,
    "score": {
      "path": "results/rlem_c6_b2/train_calib_scores/train_calib_pairwise_main_scores.npz",
      "exists": true,
      "size": 2319243,
      "sha256": "18c0be1cfee3de904424a1e91b4b86f5bb251fc9911e8f8c38bde3e79aa2e2aa"
    }
  },
  "train": {
    "split": "train_fit",
    "reused": false,
    "features": {
      "path": "results/rlem_c6_c/train_fit_video_features.npz",
      "exists": true,
      "size": 143095940,
      "sha256": "47d73f51d1aa7831484b11e5a4bdc010789db2cb2afb838880363695622b702f"
    },
    "examples": 779142,
    "feature_dim": 57,
    "positive_video_rate": 0.08097368478775024,
    "any07_rate": 0.06939171254634857
  },
  "calib": {
    "split": "train_calib",
    "reused": false,
    "features": {
      "path": "results/rlem_c6_c/train_calib_video_features.npz",
      "exists": true,
      "size": 16029279,
      "sha256": "9b2da5949335aeaec7a22e51d70d6b3ee2574d0f8dbf8e14bc6f13dd21f51229"
    },
    "examples": 87230,
    "feature_dim": 57,
    "positive_video_rate": 0.08074057102203369,
    "any07_rate": 0.06935687363147736
  }
}
```
