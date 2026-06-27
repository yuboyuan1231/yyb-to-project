# C6-B1-lite mechanism analysis

- changed_top1_queries: `356`
- unchanged_top1_queries: `10539`
- changed_top1_ratio: `0.03267553923818265`
- changed_top1_mean_iou_delta: `0.018041213151999712`

## Top1 changed confusion

| IoU | miss→hit | hit→miss | hit→hit | miss→miss |
|---|---:|---:|---:|---:|
| 0.5 | 21 | 9 | 26 | 300 |
| 0.7 | 15 | 11 | 10 | 320 |

## Replacement aggregate

- replacement_by_slot: `{'0': 356, '1': 74, '2': 55}`
- replacement_precision_05 / 07: `0.1649 / 0.1072`
- replacement_benefit_rate_05 / 07: `0.0928 / 0.0845`
- replacement_harmful_rate_05 / 07: `0.0247 / 0.0268`
- net_hit_conversion_count_05 / 07: `33 / 28`

## Slot contribution

| slot | replacements | precision@0.5 | precision@0.7 | benefit@0.5 | harm@0.5 | benefit@0.7 | harm@0.7 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 356 | 0.132 | 0.070 | 21 | 9 | 15 | 11 |
| 1 | 74 | 0.230 | 0.176 | 12 | 1 | 12 | 1 |
| 2 | 55 | 0.291 | 0.255 | 12 | 2 | 14 | 1 |
