# C4-r2-cal-v2.1 Official-Val One-Shot Audit

## Outcome

- Decision: `C4-r2-cal-v2.1 follow-up official-val positive`
- Config: `v21_00444`
- `official_val_one_shot=true`
- `learned_config_count=1`
- `successful_metric_result_count=1`
- `score_grid_on_val=false`
- `temperature_search_on_val=false`
- `post_val_adjustment=false`

The frozen v2.1 candidate is non-negative on all eight official-val metrics relative to frozen C4-lite. Both R@1 metrics are tied; the other six metrics improve. No parameters or z-score statistics were changed after seeing official-val results.

## Exact frozen scorer

```text
S = S_C4_lite
  + 0.075 * z(rel_logit)
  + 0.050 * z(iou07_logit)
  - 0.100 * z(quality_logit)
```

`b_iou05=0.0`. Z-score statistics came only from the frozen train_calib row-expanded statistics:

| Logit | Mean | Std |
|---|---:|---:|
| rel | -3.862743616104126 | 2.063218832015991 |
| iou07 | -4.213388919830322 | 2.1755518913269043 |
| quality | -3.8732757568359375 | 1.7478158473968506 |

No mean or standard deviation was estimated from official val. Retrieval remained fixed at `effective_top_n=100`, `NMS=0.7`, and `max_after_nms=100`.

## Official-val metrics

| System | 0.5-R@1 | 0.5-R@5 | 0.5-R@10 | 0.5-R@100 | 0.7-R@1 | 0.7-R@5 | 0.7-R@10 | 0.7-R@100 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Frozen C3.1 | 14.27 | 28.12 | 34.42 | 46.31 | 8.09 | 18.03 | 23.58 | 35.28 |
| Frozen C4-lite | 14.81 | 28.56 | 34.73 | 46.44 | 8.35 | 18.63 | 24.61 | 35.69 |
| C4-r2-cal-v2.1 `v21_00444` | 14.81 | 28.57 | 34.81 | 46.45 | 8.35 | 18.64 | 24.63 | 35.70 |
| Delta v2.1 vs C4-lite | +0.00 | +0.01 | +0.08 | +0.01 | +0.00 | +0.01 | +0.02 | +0.01 |
| Delta v2.1 vs C3.1 | +0.54 | +0.45 | +0.39 | +0.14 | +0.26 | +0.61 | +1.05 | +0.42 |

R@100 deltas relative to frozen C4-lite are `+0.01` at IoU 0.5 and `+0.01` at IoU 0.7.

## Movement relative to frozen C4-lite

| Diagnostic | Value |
|---|---:|
| top1_changed_ratio | 0.003028912345112437 |
| hard-positive top100 exits | 17 |
| hard-positive top100 entries | 154 |
| hard-positive top100 exit ratio | 0.0003029493005435267 |
| Pearson(C4-lite, v2.1) | 0.9997433762199504 |
| mean within-query Spearman | 0.9992932112426262 |

Fast versus bundled evaluator maximum absolute difference after metric rounding was `0.0` for both frozen C4-lite and v2.1.

## Frozen artifacts and SHA256

| Artifact | SHA256 |
|---|---|
| Freeze manifest | `a4feaa3a86ba49e585c3e6b989ed10e56d598052d3f316f098d9d86464c97a9e` |
| Official-val C4 scored candidates | `5ad6b5dbb0c1c576bd19fb70626ce03cd288053afb97594fee10751bfd5f6200` |
| Official-val C4 cache | `b24f32831331f24b905eb547d66b9f4c67b3c48d17d9e9579be4e81ccd0d5796` |
| v2.1 backbone checkpoint | `e7a37e0ca761e82513feb50ff4877f25b0fb514d4b403f1bd8c173948c912d64` |
| Official-val v2.1 logits | `1ac100bac6b77db649a5ad1783e5f27387e84efeb6f6e2ad455de4c49980c51b` |
| Official-val v2.1 row scores | `3a6bdfe5b80ddfcaf054b1d5ee25af28a330f1a2ffbd52d508359489533de1fa` |
| Official-val v2.1 submission | `fea4c1f4f963faf3ce0dbde8b42027df208e87f706ae799723e0a551818fcbdc` |
| Official-val v2.1 metrics | `11a92c9aa90ed6fbd124051c886e2fdb4a6d2eaaec020fde0a11e54fcc86e04f` |

## Evaluator accounting

One infrastructure-only evaluator attempt failed before ground truth was opened because an obsolete absolute GT path from an older manifest no longer existed. It produced no metric result. The path was resolved directly from the current dataset config, after which exactly one successful metric result was produced. Scores, submission, configuration, and evaluator code were unchanged between attempts.

## Scope controls

- raw `v2_cb` scoring config evaluated: false
- `v21_00439` evaluated: false
- any other v2.1 config evaluated: false
- C4-r2-v1 evaluated: false
- A-family evaluated: false
- C4-main used: false
- R2 head used: false
- VS head used: false
- C5 used: false
- C6 used: false
- `model/conquer.py` modified: false
- evaluator modified: false

Stop after this one-shot audit. No post-val tuning is authorized.
