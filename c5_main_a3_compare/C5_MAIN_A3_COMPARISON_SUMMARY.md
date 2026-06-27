# C5-main-inference-A3 comparison

## Final classification

`A. A3_PROMOTE`

A3b-wide passes the frozen train_calib promotion gate. The recommended frozen candidate is the robust interior configuration `c5ma_0214`; this result authorizes only freeze review, not official val.

## A3a: exact top-100 row-set lock

A3a correctly enforces its structural contract:

- top100 membership drift: `0`
- hard-positive top100 exits/entries: `0 / 0`
- best rejected nonzero R@100 deltas at IoU 0.5/0.7: `+0.02289 / +0.01144`
- oracle-video R@1@0.5/0.7 deltas: `+0.05679 / +0.32657`
- selected-span mIoU delta: `+0.000755`
- mean best-IoU rank delta: `-0.01505`

It therefore protects R@100 and preserves localization signal. However, its best nonzero configuration regresses several R@1/R@5/R@10 metrics; only zero control satisfies six-primary. A3a is `NO_PROMOTION` and remains a useful boundary analysis rather than an official-val candidate.

## A3b-wide: video-slot/quota-preserved replacement

Frozen candidate: `c5ma_0214`

```text
alpha  = 0.25
beta   = 0.05
tau    = 1.50
lambda = 0.05
gamma  = 0.05
```

Eight deltas versus C4_final are all positive:

| Metric | Delta |
|---|---:|
| 0.5 R@1 | +0.045772 |
| 0.5 R@5 | +0.045772 |
| 0.5 R@10 | +0.091544 |
| 0.5 R@100 | +0.022886 |
| 0.7 R@1 | +0.091544 |
| 0.7 R@5 | +0.068658 |
| 0.7 R@10 | +0.057215 |
| 0.7 R@100 | +0.011443 |

Safety and localization:

- video multiset drift: `0`
- video slot-sequence drift: `0`
- quota violations / duplicate spans: `0 / 0`
- row membership drift: `4.62%` of queries, allowed by wide replacement
- hard-positive exits/entries: `15 / 6`
- hard-positive exit ratio: `0.0239%`, below the `0.5%` gate
- oracle-video R@1@0.5/0.7 deltas: `+0.01420 / +0.25557`
- selected-span mIoU delta: `+0.000563`
- mean best-IoU rank delta: `-0.02499`
- Pearson / mean within-query Spearman: `0.99901 / 0.99996`

The result is robust rather than an isolated grid accident: `c5ma_0214` is interior in all five dimensions and all 10 one-step neighbors satisfy the core gate. Overall, 332 configurations satisfy the full promotion gate.

## A3b strict versus wide

For the highest-selection reference `c5ma_0098`, strict mode has zero row-membership drift and zero hard-positive exits, while wide mode changes 7.37% of row sets and has 24 exits. Strict is structurally safer. Wide provides a small positive 0.7 R@100 gain by replacing spans outside the original row set and is the predeclared primary mode. Only wide received the full grid robustness selection; strict remains a diagnostic and is not separately frozen.

## Direct answers

1. **Did A3a completely protect top100 membership?** Yes; drift and exits are exactly zero.
2. **Did A3a improve early ranks without R@100 loss?** Its best nonzero improves localization and R@100, but mixed early-rank regressions fail six-primary.
3. **Did A3b-wide improve localization while preserving video quota?** Yes. Quota/sequence drift is zero and all four localization diagnostics are positive.
4. **Does A3b-wide still produce hard-positive exits?** Yes, 15, but the `0.0239%` ratio is safely below the gate.
5. **Which A3b mode is safer?** Strict structurally; wide has the stronger demonstrated final-metric result and robust grid support.
6. **Did any scheme pass the freeze gate?** Yes, A3b-wide; 332 configs pass and `c5ma_0214` is the robust selection.
7. **Recommend official-val one-shot?** Recommend requesting explicit human authorization for exactly one frozen A3b-wide run. Do not run it automatically.
8. **Close C5-main?** No. Close A/A2/A3a grids, but retain frozen A3b-wide for review.
9. **Enter C5-main-B or candidate regeneration?** No; A3b already consumes the signal safely within fixed candidates.
10. **Enter C6?** No.

## Safety

- `official_val_used=false`
- `official_val_authorized=false`
- `post_val_adjustment=false`
- `C5-main-B_used=false`
- `candidate_regeneration_used=false`
- `C6_used=false`
- C4_final, CONQUER, candidates, NMS, effective_top_n and evaluator remain unchanged.

Execution stops at freeze review pending human authorization.
