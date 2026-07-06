# C12-4R Retriever Repair Decision

Status: `C12_RETRIEVER_CONQUER_WARM_READY_CONTINUE_TO_C12_5`

Best repaired retriever variant: `zero_delta_replay`

Best learned delta variant: `delta_regularized_alpha002`

| scheme | R@1 | R@5 | R@10 | R@100 | median | teacher top1 agreement |
|---|---:|---:|---:|---:|---:|---:|
| `delta_regularized_alpha002` | 29.10 | 64.49 | 79.17 | 100.00 | 3.0 | 99.98 |
| `listwise_teacher_topK_alpha005` | 29.17 | 64.47 | 79.15 | 99.98 | 3.0 | 99.75 |
| `qtype_balanced_alpha002` | 29.10 | 64.49 | 79.17 | 100.00 | 3.0 | 99.98 |
| `stronger_teacher_topK_alpha001` | 29.10 | 64.49 | 79.17 | 100.00 | 3.0 | 100.00 |
| `zero_delta_replay` | 29.10 | 64.49 | 79.19 | 100.00 | 3.0 | 100.00 |

Interpretation:

- CONQUER first-stage replay passes and restores train-only top100 recall.
- Learned delta repair is not reliably better than zero-delta replay.
- C12-5 is allowed only if the video candidate generator is the CONQUER-warm/replay native corpus retriever, not the weak mean-pooled C12 retriever and not a C7-B6 fixed prediction pool.
- Stronger feature work remains necessary for an independent C12 retriever.

```json
{
  "stage": "C12-4R-E",
  "status": "C12_RETRIEVER_CONQUER_WARM_READY_CONTINUE_TO_C12_5",
  "conquer_first_stage_replay_pass": true,
  "conquer_warm_retriever_implemented": true,
  "best_repaired_retriever_variant": "zero_delta_replay",
  "best_learned_delta_variant": "delta_regularized_alpha002",
  "best_metrics": {
    "GT_video_in_top100_rate": 100.0,
    "GT_video_mean_rank": 7.9633640552995395,
    "GT_video_median_rank": 3.0,
    "VR_R@1": 29.101382488479263,
    "VR_R@10": 79.19354838709677,
    "VR_R@100": 100.0,
    "VR_R@5": 64.49308755760369,
    "missing_rank_count": 0,
    "query_count": 4340
  },
  "zero_delta_replay_metrics": {
    "GT_video_in_top100_rate": 100.0,
    "GT_video_mean_rank": 7.9633640552995395,
    "GT_video_median_rank": 3.0,
    "VR_R@1": 29.101382488479263,
    "VR_R@10": 79.19354838709677,
    "VR_R@100": 100.0,
    "VR_R@5": 64.49308755760369,
    "missing_rank_count": 0,
    "query_count": 4340
  },
  "best_learned_delta_metrics": {
    "GT_video_in_top100_rate": 100.0,
    "GT_video_mean_rank": 7.964516129032258,
    "GT_video_median_rank": 3.0,
    "VR_R@1": 29.101382488479263,
    "VR_R@10": 79.17050691244239,
    "VR_R@100": 100.0,
    "VR_R@5": 64.49308755760369,
    "missing_rank_count": 0,
    "query_count": 4340,
    "teacher_top1_agreement_rate": 99.97695852534562
  },
  "learned_delta_repair_effective": false,
  "teacher_distillation_repair_effective": true,
  "allow_enter_c12_5": true,
  "c12_5_entry_condition": "Use CONQUER-warm/replay native corpus video retriever as video candidate generator; do not use C7-B6 fixed prediction pool.",
  "need_stronger_features": true,
  "feature_repair_status": "C12_FEATURE_REPAIR_NOT_READY",
  "official_val_used": false,
  "evaluator_modified": false,
  "nms_modified": false,
  "notes": [
    "Replay preserves CONQUER first-stage train-only reference exactly/near-exactly.",
    "Small learned deltas did not produce robust improvement over zero-delta replay.",
    "C12-5 may proceed only with CONQUER-warm/replay video retrieval, while stronger feature work remains necessary for an independent retriever."
  ]
}
```

No official val was run, no official prediction pool was read, and evaluator/NMS were not modified.
