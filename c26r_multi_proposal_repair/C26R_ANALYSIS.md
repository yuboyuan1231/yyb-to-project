# C26R Multi-Proposal Repair Analysis

## Problem Diagnosis

C26 low VCMR R@100 is primarily a proposal/localizer materialization problem, not a retriever-only failure.

- C26 did not materialize native `c26_span_start` / `c26_span_end`.
- C26 VCMR used borrowed C24H `best_bmn_span_start` / `best_bmn_span_end`.
- C24H materialization collapsed C17 BMN/T2 multi-span maps to one best span per query-video.
- Restoring C17 medium multi-span proposal space changes the C26 holdout VCMR R@100@0.5 from `8.5556` to the `34-36` range for inference routes.

## Five Requested Checks

1. Original localizer: C26 did not use the original full localizer/proposal space. C17 has multi-span BMN/T2 proposals; C26/C24H only used one best BMN span per video.
2. Proposal space: C26 one-span proposal space is not preserved. C26R restores C17 multi-span rows: median `57` spans per query-video, max `64`.
3. Span unit: seconds is still the best/consistent interpretation. On the selected C26R route, seconds gives R@100@0.5 `34.0`; `times_1p5` gives `29.1111`; `div_1p5` gives `23.3333`; rounded 1.5s grid gives `33.5556`.
4. Row-space: C26R uses the exact C26 medium query set intersected with C17 medium: 900 holdout queries and 900 select queries, seed 2026.
5. Zero-delta: multi-proposal zero-delta restores R@100@0.5 to `36.1-36.7`, but has much worse VR@100 around `66-67`; balanced C26R selection prefers guarded C26+localizer.

## Routes Tested

- R1 zero-delta multi-span:
  - C17 retriever only: holdout R@100@0.5 `36.3333`, VR@100 `67.0`.
  - C26 retriever only: holdout R@100@0.5 `36.1111`, VR@100 `67.0`.
  - C26 final video only: holdout R@100@0.5 `36.6667`, VR@100 `66.3333`.
  - Interpretation: multi-span alone exposes span headroom but video ranking is too weak.

- R2 original C17 localizer:
  - BMN only and T2 only fail alone.
  - BMN+retriever and T2+retriever recover to roughly `32-33` R@100@0.5.
  - C17 BMN/T2 hybrid reaches R@100@0.5 `33.4444`; conflict guard reaches `34.5556`.

- R3 C26 with restored C17 multi-span localizer:
  - Best by calib_select: `R3b1_C26_PREM_guarded_localizer`.
  - Holdout: R@1/5/10/100@0.5 = `2.6667 / 7.7778 / 12.0 / 34.0`, VR@100 `94.4444`.
  - Highest inference R@100@0.5 is `R1c_zero_delta_c26_final_video_multi_span` at `36.6667`, but it is not selected because VR@100 is only `66.3333`.

- R4 topM proposal variants:
  - top1 per video: R@100@0.5 `8.4444`.
  - top2: `10.2222`; top4: `13.7778`; top8: `19.7778`; top16: `25.5556`; top32: `30.1111`; top64: `34.0`.
  - Interpretation: one-span collapse is the dominant failure mode.

- R5 span unit variants:
  - seconds: R@100@0.5 `34.0`.
  - times 1.5: `29.1111`.
  - divide 1.5: `23.3333`.
  - rounded 1.5 grid: `33.5556`.
  - Interpretation: no evidence of a better unit conversion than seconds.

- R6 oracle headroom:
  - Diagnostic only, excluded from selection because it uses GT IoU.
  - Holdout R@100@0.5 `71.6667`.
  - Interpretation: C17 multi-span proposal space has substantial headroom; scoring/localizer selection remains improvable.

## Decision

`C26R_MULTI_PROPOSAL_REPAIR_PROMISING`.

The next executable route should materialize multi-span proposals as a first-class C26/C26R artifact instead of using C24H one-span evidence. The best current inference route is `R3b1_C26_PREM_guarded_localizer`; it is not promoted and does not use official validation or pseudo-official selection.
