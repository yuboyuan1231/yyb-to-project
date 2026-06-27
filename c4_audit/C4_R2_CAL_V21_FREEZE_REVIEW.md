# C4-r2-cal-v2.1 Freeze Review

## Decision

- Status: `C4_R2_CAL_V21_FREEZE_REVIEW_PASS`
- Selected config: `v21_00444`
- Primary selection baseline: frozen C4-lite `c4_00394`
- Selection reason: `robust_non_isolated_boundary_preferred_over_higher_isolated_score`
- Recommendation: freeze candidate B (`v21_00444`) rather than the higher-score candidate A (`v21_00439`).

Candidate A has the larger selection-score delta (`+0.0801006980`) but no passing neighbor. Candidate B has a slightly smaller delta (`+0.0657970019`), two passing neighbors, smaller movement, and remains strictly positive on all eight metrics versus frozen C4-lite. B is on an active search-grid boundary because `d_quality=-0.100`, but it is not an isolated boundary point.

## Frozen scorer

```text
S = S_C4_lite
  + 0.075 * z(rel_logit)
  + 0.050 * z(iou07_logit)
  - 0.100 * z(quality_logit)
```

`b_iou05=0.0`. The z-score statistics are the row-expanded train_calib statistics already recorded by the v2.1 search:

| Logit | Mean | Std |
|---|---:|---:|
| rel | -3.862743616104126 | 2.063218832015991 |
| iou05 | -3.921426534652710 | 2.1069819927215586 |
| iou07 | -4.213388919830322 | 2.1755518913269043 |
| quality | -3.8732757568359375 | 1.7478158473968506 |

Retrieval constants remain frozen at `effective_top_n=100`, `NMS=0.7`, and `max_after_nms=100`.

## Candidate comparison

Metric order below is `0.5-R@1`, `0.5-R@5`, `0.5-R@10`, `0.5-R@100`, `0.7-R@1`, `0.7-R@5`, `0.7-R@10`, `0.7-R@100`.

### Delta versus frozen C4-lite c4_00394

| Candidate | 0.5-R@1 | 0.5-R@5 | 0.5-R@10 | 0.5-R@100 | 0.7-R@1 | 0.7-R@5 | 0.7-R@10 | 0.7-R@100 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A `v21_00439` | +0.0686577412 | +0.0114429569 | +0.0686577412 | +0.0457718274 | +0.0343288706 | +0.0457718274 | +0.1830873098 | +0.0457718274 |
| B `v21_00444` | +0.0457718274 | +0.0114429569 | +0.0686577412 | +0.0343288706 | +0.0228859137 | +0.0343288706 | +0.1602013960 | +0.0343288706 |

Both candidates are strictly positive on all eight metrics versus the primary baseline.

### Delta versus frozen C3.1

| Candidate | 0.5-R@1 | 0.5-R@5 | 0.5-R@10 | 0.5-R@100 | 0.7-R@1 | 0.7-R@5 | 0.7-R@10 | 0.7-R@100 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A `v21_00439` | +1.6477857878 | +0.5950337567 | +0.6636914979 | +0.1487584392 | +0.8811076782 | +2.2084906740 | +2.3229202426 | +0.7437921959 |
| B `v21_00444` | +1.6248998741 | +0.5950337567 | +0.6636914979 | +0.1373154823 | +0.8696647214 | +2.1970477171 | +2.3000343289 | +0.7323492390 |

### Delta versus current v2_cb

| Candidate | 0.5-R@1 | 0.5-R@5 | 0.5-R@10 | 0.5-R@100 | 0.7-R@1 | 0.7-R@5 | 0.7-R@10 | 0.7-R@100 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A `v21_00439` | +0.0343288706 | -0.0114429569 | +0.0572147843 | +0.0000000000 | +0.0114429569 | +0.0000000000 | +0.0915436549 | -0.0114429569 |
| B `v21_00444` | +0.0114429569 | -0.0114429569 | +0.0572147843 | -0.0114429569 | +0.0000000000 | -0.0114429569 | +0.0686577412 | -0.0228859137 |

### Delta versus C4-r2-v1 reference

| Candidate | 0.5-R@1 | 0.5-R@5 | 0.5-R@10 | 0.5-R@100 | 0.7-R@1 | 0.7-R@5 | 0.7-R@10 | 0.7-R@100 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A `v21_00439` | +0.0228859137 | +0.0000000000 | +0.0000000000 | -0.0228859137 | +0.0000000000 | +0.0228859137 | +0.0114429569 | -0.0114429569 |
| B `v21_00444` | +0.0000000000 | +0.0000000000 | +0.0000000000 | -0.0343288706 | -0.0114429569 | +0.0114429569 | -0.0114429569 | -0.0228859137 |

C4-r2-v1 is reference-only and is not a hard promotion baseline.

## Selection and movement diagnostics

| Diagnostic | A `v21_00439` | B `v21_00444` |
|---|---:|---:|
| selection score delta vs C4-lite | +0.0801006980 | +0.0657970019 |
| top1 changed ratio | 0.0029751688 | 0.0025174505 |
| Pearson(C4-lite, candidate) | 0.9996162449 | 0.9995508996 |
| hard-positive top100 exits | 12 | 11 |
| hard-positive top100 entries | 281 | 272 |
| hard-positive top100 exit ratio | 0.0001912381 | 0.0001753016 |
| passing neighbor count | 0 | 2 |
| passing neighbor IDs | none | `v21_00468`, `v21_00588` |
| active parameter on grid boundary | false | true |
| isolated boundary point | false | false |
| movement safe | true | true |

The source audit's `isolated_boundary_point=false` for both candidates refers to literal grid-boundary status. For robustness selection, A is unsupported because it has zero passing neighbors; B is neighbor-supported despite lying on an active parameter boundary. This distinction is why B is the safer one-shot candidate.

## Protocol review

- `official_val_used=false`
- `official_val_authorized=false`
- `post_val_adjustment=false`
- `score_grid_on_val=false`
- `model_retrained=false`
- `capacity_search=false`
- `modifies_conquer=false`
- `C4-main used=false`
- `R2 head used=false`
- `VS head used=false`
- `C5 used=false`
- `C6 used=false`

No official-val run is authorized by this review. Stop after freeze materialization and wait for explicit one-shot authorization.
