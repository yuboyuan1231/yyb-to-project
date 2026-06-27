# C5 front-rank promotion gate addendum

Status: `MATERIALIZED_BEFORE_OFFICIAL_VAL`

This addendum clarifies the metric hierarchy for C5. It does not alter C4_final, rerun or reinterpret a completed C5 grid, change any frozen configuration, or authorize more than one official-val evaluation.

## Metric hierarchy

The C5 primary objective is front-rank VCMR together with localization diagnostics. The six primary front-rank metrics are `0.5-r1`, `0.5-r5`, `0.5-r10`, `0.7-r1`, `0.7-r5`, and `0.7-r10`; the core subset is `0.5-r1`, `0.7-r1`, `0.5-r5`, and `0.7-r5`.

`0.5-r100` and `0.7-r100` are recall-safety metrics rather than hard primary metrics. They remain constrained together with hard-positive top-100 exits, `top1_changed_ratio`, Pearson correlation to C4_final, mean within-query Spearman correlation, video multiset and slot-sequence drift, quota violations, and duplicate spans.

Localization diagnostics are oracle-video R@1@0.5 and R@1@0.7 deltas, selected-span mIoU delta, mean best-IoU rank delta, and the best-IoU improved/worsened ratio.

## Train-calib classification rules

- `C5_PROMOTE`: at least 4/6 front-rank deltas are positive; both R@1 deltas are nonnegative with at least one positive; at least two localization diagnostics are positive; both R@100 deltas are at least -0.05; hard-positive exit ratio is at most 0.5%; structural constraints pass; selection has `official_val_used=false`.
- `C5_STRONG_PROMOTE`: all six front-rank deltas are positive; both R@100 deltas are nonnegative; at least three localization diagnostics are positive; hard-positive exit ratio is at most 0.5%; structural constraints pass.
- `C5_FRONT_RANK_POSITIVE`: at least 3/6 front-rank deltas are positive; at least one R@1 delta is positive; localization diagnostics are positive; both R@100 deltas are at least -0.10; hard-positive exit ratio is at most 0.75%.
- `C5_LOCALIZATION_ONLY`: localization diagnostics are positive but front-rank metrics do not meet `C5_FRONT_RANK_POSITIVE`.
- `C5_RECALL_TRADEOFF`: front-rank metrics are positive but R@100 or hard-positive-exit safety fails.
- `C5_NEGATIVE`: neither localization nor front-rank improves.

## Frozen A3b-wide decision

`c5ma_0214` passes both the earlier strict all-eight-positive train_calib gate and the new front-rank gate. Therefore this addendum does not rescue a failed configuration; it only clarifies metric hierarchy.

The addendum is only for interpreting completed A2/A3 experiments and classifying the authorized one-shot official-val outcome. It cannot change C4_final, any completed C5 grid, or the frozen A3b-wide parameters. Official val remains exactly one shot with no post-val adjustment.
