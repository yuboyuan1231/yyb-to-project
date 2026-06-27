# C6-C2 intervention model audit

```json
{
  "status": "C6_C2_INTERVENTION_MODEL_TRAINED",
  "official_val_used": false,
  "model_class": "C6C2InterventionUtilityGate",
  "architecture": {
    "hidden": 512,
    "layers": 4,
    "dropout": 0.12,
    "heads": [
      "enable_logit",
      "expected_gain_07_r1",
      "expected_gain_05_r1",
      "expected_r5_risk",
      "expected_r10_risk",
      "expected_r100_risk",
      "hard_exit_risk",
      "video_entry_prob",
      "video_exit_prob",
      "action_utility",
      "abstain_logit"
    ]
  },
  "loss": {
    "action_good_bce_weight": 1.5,
    "action_bad_bce_weight": 1.0,
    "gain_07_regression_weight": 1.0,
    "gain_05_regression_weight": 0.7,
    "topk_risk_bce_weight": 1.0,
    "hard_exit_bce_weight": 0.8,
    "video_entry_exit_bce_weight": 0.5,
    "abstention_calibration_weight": 0.3,
    "coverage_regularization_weight": 0.3,
    "coverage_target": 0.02
  },
  "training": {
    "reused": true,
    "model": {
      "path": "results/rlem_c6_c2/c6c2_intervention_gate.pt",
      "exists": true,
      "size": 3271499,
      "sha256": "4543f492c556bca076fac353e2ae6cb7588b2b2facd984fc560a99a68d7ed5e1"
    },
    "history": [
      {
        "epoch": 1,
        "loss": 2.3791984052493653,
        "elapsed_sec": 1.5864551067352295,
        "examples": 468270,
        "device": "cuda",
        "coverage_target": 0.02
      },
      {
        "epoch": 2,
        "loss": 1.7233103287631069,
        "elapsed_sec": 1.2000253200531006,
        "examples": 468270,
        "device": "cuda",
        "coverage_target": 0.02
      },
      {
        "epoch": 3,
        "loss": 1.6840818565467308,
        "elapsed_sec": 1.1948730945587158,
        "examples": 468270,
        "device": "cuda",
        "coverage_target": 0.02
      },
      {
        "epoch": 4,
        "loss": 1.6731162173994656,
        "elapsed_sec": 1.2180225849151611,
        "examples": 468270,
        "device": "cuda",
        "coverage_target": 0.02
      },
      {
        "epoch": 5,
        "loss": 1.6628850924557652,
        "elapsed_sec": 1.2017853260040283,
        "examples": 468270,
        "device": "cuda",
        "coverage_target": 0.02
      }
    ]
  },
  "scoring": {
    "train_fit": {
      "split": "train_fit",
      "reused": true,
      "scores": {
        "path": "results/rlem_c6_c2/train_fit_gate_scores.npz",
        "exists": true,
        "size": 10711555,
        "sha256": "2a44a1ab4c482c16c84fc76b113e3eaf2480de9e32ba56645858362ff9e1b7d3"
      }
    },
    "train_calib": {
      "split": "train_calib",
      "reused": true,
      "scores": {
        "path": "results/rlem_c6_c2/train_calib_gate_scores.npz",
        "exists": true,
        "size": 1209861,
        "sha256": "24486acaee3a824c481cc9f767cdb4c3960e46d5b9736d72df079f9a774d6245"
      }
    }
  }
}
```
