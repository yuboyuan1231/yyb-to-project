# C4 Stage Final Summary

## Final freeze

- `C4_stage_status = COMPLETE`
- `C4_final_system = C4-r2-cal-v2.1`
- `C4_final_config = v21_00444`
- `C4_core_module = C4-lite`
- `C4_primary_baseline_for_final = frozen C4-lite c4_00394`
- `C4_secondary_baseline = frozen C3.1`

C4-lite is the core mechanism, the base score consumed by the final system, the main source of C4's gain, and the primary ablation baseline. C4-r2-cal-v2.1 `v21_00444` is the final frozen C4 output.

## Frozen C4 final scorer

```text
S_C4_final =
    S_C4_lite
  + 0.075 * z(rel_logit)
  + 0.050 * z(iou07_logit)
  - 0.100 * z(quality_logit)
```

`b_iou05 = 0.0`.

All z-score statistics are the frozen train_calib row-expanded statistics. No mean or standard deviation was estimated on official val.

| Logit | Mean | Std |
|---|---:|---:|
| rel | -3.862743616104126 | 2.063218832015991 |
| iou07 | -4.213388919830322 | 2.1755518913269043 |
| quality | -3.8732757568359375 | 1.7478158473968506 |

## Official-val result

| System | 0.5-R@1 | 0.5-R@5 | 0.5-R@10 | 0.5-R@100 | 0.7-R@1 | 0.7-R@5 | 0.7-R@10 | 0.7-R@100 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Frozen C3.1 | 14.27 | 28.12 | 34.42 | 46.31 | 8.09 | 18.03 | 23.58 | 35.28 |
| Frozen C4-lite | 14.81 | 28.56 | 34.73 | 46.44 | 8.35 | 18.63 | 24.61 | 35.69 |
| C4 final: `v21_00444` | 14.81 | 28.57 | 34.81 | 46.45 | 8.35 | 18.64 | 24.63 | 35.70 |
| C4 final vs C4-lite | +0.00 | +0.01 | +0.08 | +0.01 | +0.00 | +0.01 | +0.02 | +0.01 |
| C4 final vs C3.1 | +0.54 | +0.45 | +0.39 | +0.14 | +0.26 | +0.61 | +1.05 | +0.42 |

The decomposition shows the intended hierarchy: C4-lite supplies the main C4 gain over C3.1, while v2.1 adds a small, safe calibration gain. Relative to frozen C4-lite, v2.1 is non-negative on all eight metrics, positive on six, tied on the two R@1 metrics, and positive on both R@100 metrics.

## Final protocol state

- `C4_main_used = false`
- `R2_head_used = false`
- `VS_head_used = false`
- `model/conquer.py modified = false`
- `candidate_generation_modified = false`
- `NMS_modified = false`
- `evaluator_modified = false`
- `post_val_adjustment = false`
- `score_grid_on_val = false`
- `temperature_search_on_val = false`
- `effective_top_n = 100`
- `NMS = 0.7`
- `max_after_nms = 100`

C4-main and the VS/R2-head path are deferred, not abandoned. They remain candidates for the post-C5-lite C6/backbone-integrated phase.

## Stop decision and next stage

The C4-r2-cal-v2.1 freeze review and official-val one-shot audit are complete. Therefore C4 stops here:

- no further C4-r2 grid search;
- no further C4-r2 capacity search;
- no further C4-r2 weak-fusion probes;
- no C4-main or VS/R2-head execution in this stage;
- no post-val parameter adjustment.

The planned next stage is **C5-lite: retrieval evidence → localization calibration**. This summary does not authorize or start C5-lite; `C5_started=false`.

## Frozen evidence

- C4-lite official-val freeze: `c4_audit/C4_LITE_OFFICIAL_VAL_FREEZE_MANIFEST.json`
- v2.1 freeze review: `c4_audit/C4_R2_CAL_V21_FREEZE_REVIEW.md`
- v2.1 freeze manifest: `c4_audit/C4_R2_CAL_V21_FREEZE_MANIFEST.json`
- v2.1 official-val audit: `c4_audit/C4_R2_CAL_V21_OFFICIAL_VAL_ONE_SHOT_AUDIT.md`
- v2.1 official-val manifest: `results/rlem_c4_r2_cal_v21_official_val/VAL_V21_ONE_SHOT_MANIFEST.json`

This stage-level summary supersedes any earlier provisional wording that described C4-lite alone as the current C4 main result. Historical experiment audits remain unchanged.
