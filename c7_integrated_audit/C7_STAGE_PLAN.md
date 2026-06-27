# C7 Stage Plan

## C7-B: Frozen-Backbone Head Training

- Requires explicit authorization.
- Freeze CONQUER backbone, QDF/QAL, and original VR/ML heads.
- Train only new RLEM heads on train_fit/train_calib.
- No official val.

Acceptance: delta vs C6-B2 train_calib 0.7-r1 > 0, R@5/R@10 non-collapse, and non-zero integrated contribution.

## C7-C: Partial Unfreeze

Only if C7-B passes: unfreeze VR head, ML head, optionally late QAL layers, with small LR and short schedule.

## C7-D: Freeze Review

Only train_calib freeze review. Do not run official val automatically.

```json
{
  "stage": "C7 staged training plan",
  "official_val_used": false,
  "training_started": false,
  "phases": {
    "C7_A_source_audit_design": {
      "status": "this_step",
      "allowed": [
        "source audit",
        "tensor map",
        "integrated design",
        "training plan"
      ],
      "forbidden": [
        "training",
        "official val",
        "NMS/evaluator changes"
      ]
    },
    "C7_B_frozen_backbone_head_training": {
      "requires_authorization": true,
      "freeze": [
        "CONQUER backbone",
        "QDF",
        "QAL",
        "original VR head",
        "original ML head"
      ],
      "train": [
        "moment-aware VR residual head",
        "partial relevance MIL head",
        "proposal confidence head",
        "hard-negative residual objective"
      ],
      "data": "train_fit/train_calib only",
      "acceptance": [
        "delta_vs_C6_B2 train_calib 0.7-r1 > 0",
        "R@5/R@10 non-collapse",
        "enabled/non-noop integrated contribution > 0",
        "invalid/duplicate span count remains 0",
        "positive entries >= exits and hard exits bounded"
      ]
    },
    "C7_C_partial_unfreeze": {
      "allowed_only_if": "C7-B passes train_calib freeze review",
      "unfreeze": [
        "VR head",
        "ML head",
        "optionally late QAL layers"
      ],
      "schedule": "small learning rate, short schedule, strict train_calib selection"
    },
    "C7_D_freeze_review": {
      "allowed_only_if": "train_calib passes",
      "output": "freeze review and manifest only",
      "official_val": "not automatic; requires explicit human authorization"
    }
  },
  "stop_condition": "After this audit/design step, request authorization before implementation/training."
}
```
