# C6-B1-lite R1-safe selector independent review

## Verdict

- Review status: `PASS_FOR_TRAIN_CALIB_FREEZE_REVIEW`
- Stage status under review: `C6_B1_LITE_R1_SAFE_POSITIVE`
- Official val used: `false`
- Post-val adjustment: `false`
- C6-C used: `false`
- C4_final retained: `true`

This review does not authorize or run official val. It only checks whether the train_fit/train_calib-only result is internally consistent enough to freeze for a later one-shot decision.

## Reviewed frozen candidate

```json
{
  "config_id": "c6b1r1_0057",
  "apply_slots": 3,
  "threshold0": 0.4,
  "threshold_rest": 1.8,
  "max_replacements": 2
}
```

The policy keeps the C4_final video slots fixed and permits at most two high-confidence span replacements among the first three slots. This is a conservative candidate interface, not a broad boundary-proposal replacement.

## Baseline and best metrics

Baseline is pseudo C4_final under the same fixed-video-slot evaluator.

| metric | baseline | C6-B1-lite | delta |
|---|---:|---:|---:|
| 0.5-r1 | 39.375215 | 39.581188 | +0.205973 |
| 0.5-r5 | 62.284014 | 62.432773 | +0.148758 |
| 0.5-r10 | 69.481634 | 69.607507 | +0.125873 |
| 0.5-r100 | 76.221536 | 76.313079 | +0.091544 |
| 0.7-r1 | 26.376016 | 26.662089 | +0.286074 |
| 0.7-r5 | 48.644010 | 48.849983 | +0.205973 |
| 0.7-r10 | 56.242133 | 56.436663 | +0.194530 |
| 0.7-r100 | 65.304955 | 65.465156 | +0.160201 |

All eight deltas are positive. This is the key distinction from C6-B0, where R@1 improved but R@5 collapsed.

## Movement and safety diagnostics

```json
{
  "replacement_rate_top_slots": 0.01643971468894229,
  "replacements": 431,
  "top1_changed_ratio": 0.03936377159858107,
  "hard_positive_top100_query_exits": 0,
  "hard_positive_top100_query_entries": 8,
  "video_slot_drift": 0.0,
  "video_multiset_drift": 0.0
}
```

The corrected hard-positive diagnostics compare post-NMS candidate output against post-NMS pseudo C4_final baseline. Under that corrected definition:

- hard-positive exits are `0`;
- hard-positive entries are `8`;
- video slot/multiset drift is exactly `0.0`;
- top1 change ratio is below the `0.08` gate.

## Grid robustness

- Grid size: `78`
- Feasible count: `31`
- R1-positive count: `78`
- R5-safe count: `71`

The highest raw-selection configurations were rejected by the movement gate because they changed too many top1 spans:

- examples: `c6b1r1_0013`, `c6b1r1_0023`
- top1_changed_ratio around `0.1924`
- these are not selected despite higher raw R1 score.

The selected configuration is the best feasible policy after applying the R@5 and movement safety gates. Neighbor support is acceptable but not perfect:

- exact-neighbor support is low because the grid is coarse;
- within a broader local radius, several same-family policies remain feasible;
- top feasible policies are concentrated around `threshold0=0.4`, with `threshold_rest` in `[0.8, 1.2, 1.8]`.

This suggests the result is not a pure single-point accident, but a later freeze should record that `threshold_rest=1.8` is at the high end of the searched grid.

## Audit findings

One issue was found and fixed during review:

- previous movement diagnostics used a pre-NMS original top100 reference for hard-positive exits/entries;
- this made the baseline itself show exits;
- the script was patched to compare post-NMS candidate output against post-NMS pseudo C4_final baseline;
- metrics and selected config were unchanged after rerun;
- corrected best movement is now `hard_positive_top100_query_exits=0`, `entries=8`.

## Recommendation

Recommendation: freeze `c6b1r1_0057` for a formal one-shot review decision, but do not run official val until explicitly authorized.

Rationale:

1. It fixes the C6-B0 failure mode by keeping replacement sparse.
2. It improves R@1 while keeping R@5 positive.
3. It preserves fixed C4_final video slots.
4. It has zero corrected hard-positive exits.
5. It uses a medium selector model trained on train_fit only, with train_calib used only for policy selection.

Not recommended:

- do not broaden replacement before freeze;
- do not continue arbitrary grid expansion;
- do not enter C6-C;
- do not treat this as final VCMR without official one-shot authorization.
