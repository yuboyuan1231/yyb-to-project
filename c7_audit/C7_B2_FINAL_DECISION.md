# C7-B2 final decision

```json
{
  "status": "C7_B2_TRAIN_CALIB_NO_PROMOTION",
  "official_val_used": false,
  "official_val_run": false,
  "post_val_adjustment": false,
  "second_official_val": false,
  "C7_B1_archived_no_adjustment": true,
  "build": {
    "proposal_scores": [
      {
        "split": "train_fit",
        "reused": true,
        "scores": {
          "path": "results/rlem_c7_b2/train_fit_c7_b1_proposal_scores.npz",
          "exists": true,
          "size": 41678195,
          "sha256": "d6286f6ee5354e5712d1a8ad4802ffa2043037c0298790c48bf4807898987e21"
        }
      },
      {
        "split": "train_calib",
        "reused": true,
        "scores": {
          "path": "results/rlem_c7_b2/train_calib_c7_b1_proposal_scores.npz",
          "exists": true,
          "size": 9030668,
          "sha256": "75a0f45612b895822f68620857bfbdc7f9b43a8783cac1bda32a10d1906087e4"
        }
      }
    ],
    "video_datasets": [
      {
        "split": "train_fit",
        "reused": true,
        "dataset": {
          "path": "results/rlem_c7_b2/train_fit_video_dataset.npz",
          "exists": true,
          "size": 108540619,
          "sha256": "642696cbe67b555ac935e0f2af81074bb004929f41224bd5089eb9119cd02b9b"
        }
      },
      {
        "split": "train_calib",
        "reused": true,
        "dataset": {
          "path": "results/rlem_c7_b2/train_calib_video_dataset.npz",
          "exists": true,
          "size": 12497184,
          "sha256": "debf5499086ffb954153111f4f8f2ae4d3fec86f3ed2ee0c5b65e81e193317f1"
        }
      }
    ]
  },
  "training": {
    "reused": true,
    "model": {
      "path": "results/rlem_c7_b2/c7_b2_mil_video_residual_head.pt",
      "exists": true,
      "size": 1276979,
      "sha256": "8db8ae4b65737f166e66a023b36c50312e2ee2bc78f50088a9b87c78e00e86de"
    },
    "history": [
      {
        "epoch": 1,
        "loss": 1.7445941120386124,
        "elapsed_sec": 1.7518038749694824,
        "device": "cuda"
      },
      {
        "epoch": 2,
        "loss": 0.5192981933554014,
        "elapsed_sec": 1.5858056545257568,
        "device": "cuda"
      },
      {
        "epoch": 3,
        "loss": 0.28982123360037804,
        "elapsed_sec": 1.5702800750732422,
        "device": "cuda"
      }
    ]
  },
  "scoring": [
    {
      "split": "train_fit",
      "reused": true,
      "scores": {
        "path": "results/rlem_c7_b2/train_fit_video_scores.npz",
        "exists": true,
        "size": 10958054,
        "sha256": "3603e53757836e50020a6f47c031e0b786f5ac7775d16f5e5ac3b34a91af92c9"
      }
    },
    {
      "split": "train_calib",
      "reused": true,
      "scores": {
        "path": "results/rlem_c7_b2/train_calib_video_scores.npz",
        "exists": true,
        "size": 1224688,
        "sha256": "f98692dbb133340dfa5935f20315100d7d12fa0d39587f8580fb1a5ad0207a9d"
      }
    }
  ],
  "best": {
    "config": {
      "alpha": 1.0,
      "beta": 0.1,
      "gamma": 0.1,
      "rho": 0.05,
      "gate_threshold": 0.32373046875,
      "max_video_slot_changes": 1,
      "lambda_span": 0.1,
      "video_weight": 0.75,
      "anchor": "C7_B1_frozen"
    },
    "metrics": {
      "0.5-r1": 45.65739787161002,
      "0.5-r5": 66.90696876072778,
      "0.5-r10": 73.02895068085593,
      "0.5-r100": 76.3474081702712,
      "0.7-r1": 30.54125185948049,
      "0.7-r5": 53.17542052866461,
      "0.7-r10": 60.72777205629935,
      "0.7-r100": 65.56814280810161
    },
    "video_metrics": {
      "GT_video_R@1": 55.807300606476716,
      "GT_video_R@5": 73.74985696303925,
      "GT_video_R@10": 78.39569744822062,
      "GT_video_R@100": 80.37532898500973
    },
    "movement": {
      "hard_positive_top100_query_exits": 179,
      "hard_positive_top100_query_entries": 3,
      "hard_positive_exit_ratio": 0.026139018691588786,
      "video_slot_drift_rate": 0.40963741136854087,
      "invalid_span_count": 0,
      "duplicate_span_count_after_nms": 0
    },
    "official_val_used": false,
    "config_id": "c7_b2_mil_video_0012",
    "delta_vs_C6_B1": {
      "0.5-r1": 6.076210092687951,
      "0.5-r5": 4.474196132280589,
      "0.5-r10": 3.4214441011557284,
      "0.5-r100": 0.0343288705801541,
      "0.7-r1": 3.8791623755578435,
      "0.7-r5": 4.3254376930999,
      "0.7-r10": 4.291108822519739,
      "0.7-r100": 0.10298661174047652
    },
    "delta_vs_C6_B2": {
      "0.5-r1": 5.835907998626844,
      "0.5-r5": 4.485639089140641,
      "0.5-r10": 3.410001144295677,
      "0.5-r100": 0.05721478430025684,
      "0.7-r1": 3.4901018423160544,
      "0.7-r5": 4.2796658656596875,
      "0.7-r10": 4.233894038219482,
      "0.7-r100": 0.04577182744020547
    },
    "delta_vs_C7_B1_frozen": {
      "0.5-r1": 2.7463096464126338,
      "0.5-r5": 1.6020139604073762,
      "0.5-r10": 0.7666781096235269,
      "0.5-r100": -2.013960407369268,
      "0.7-r1": 4.176679253919211,
      "0.7-r5": 1.7965442270282637,
      "0.7-r10": 0.09154365488042515,
      "0.7-r100": -6.430941755349579
    },
    "future_promotion_gate_pass": false,
    "selection_score": 30.31718101719832
  },
  "delta_vs_C6_B1": {
    "0.5-r1": 6.076210092687951,
    "0.5-r5": 4.474196132280589,
    "0.5-r10": 3.4214441011557284,
    "0.5-r100": 0.0343288705801541,
    "0.7-r1": 3.8791623755578435,
    "0.7-r5": 4.3254376930999,
    "0.7-r10": 4.291108822519739,
    "0.7-r100": 0.10298661174047652
  },
  "delta_vs_C6_B2": {
    "0.5-r1": 5.835907998626844,
    "0.5-r5": 4.485639089140641,
    "0.5-r10": 3.410001144295677,
    "0.5-r100": 0.05721478430025684,
    "0.7-r1": 3.4901018423160544,
    "0.7-r5": 4.2796658656596875,
    "0.7-r10": 4.233894038219482,
    "0.7-r100": 0.04577182744020547
  },
  "delta_vs_C7_B1_frozen": {
    "0.5-r1": 2.7463096464126338,
    "0.5-r5": 1.6020139604073762,
    "0.5-r10": 0.7666781096235269,
    "0.5-r100": -2.013960407369268,
    "0.7-r1": 4.176679253919211,
    "0.7-r5": 1.7965442270282637,
    "0.7-r10": 0.09154365488042515,
    "0.7-r100": -6.430941755349579
  },
  "evaluator_modified": false,
  "nms_modified": false,
  "C6_C4_artifacts_modified": false,
  "enter_C7_C": false,
  "enter_C8": false
}
```
