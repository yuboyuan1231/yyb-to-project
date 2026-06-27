# C5-main-inference-A3 video_slot freeze audit

- Status: `C5_MAIN_A3_FREEZE_REVIEW_PASS`
- Selected config: `c5ma_0214`
- Promotion candidates: `332`
- Passing one-step neighbors: `10`
- Official val used/authorized: `false / false`
- Variant: `A3b-wide video-slot/quota-preserved replacement`
- Frozen stats: train_fit-only A2a `Delta_center` statistics
- Config: `alpha=0.25, beta=0.05, tau=1.5, lambda=0.05, gamma=0.05`

## Selected deltas vs C4_final
- `0.5-r1`: `0.045771827440212576`
- `0.5-r5`: `0.045771827440212576`
- `0.5-r10`: `0.09154365488041094`
- `0.7-r1`: `0.0915436548804216`
- `0.7-r5`: `0.06865774116031531`
- `0.7-r10`: `0.05721478430026394`
- `0.5-r100`: `0.022885913720102735`
- `0.7-r100`: `0.011442956860051368`

## Structure
- `top100_membership_drift_query_ratio`: `0.04622954571461266`
- `top100_membership_symmetric_difference_rows`: `820`
- `video_multiset_drift_query_ratio`: `0.0`
- `video_slot_sequence_drift_query_ratio`: `0.0`
- `quota_violation_count`: `0`
- `duplicate_span_count`: `0`
- `constraint_satisfied`: `True`

## Localization and movement

- Oracle-video R@1@0.5 / R@1@0.7 deltas: `+0.01420 / +0.25557`
- Selected-span mIoU delta: `+0.000563`
- Mean best-IoU rank delta: `-0.02499`
- Best-IoU rank improved/worsened: `5.54% / 3.38%`
- Hard-positive top100 exits/entries: `15 / 6`
- Hard-positive exit ratio: `0.0239%`
- Pearson / mean within-query Spearman: `0.99901 / 0.99996`

The configuration is an interior robust point: all 10 one-step neighbors pass the core gate. This review freezes train_calib selection only; official val remains unauthorized.

See `results/rlem_c5_main_a3_video_slot/train_calib_search_audit.json` for complete movement/localization and strict/wide diagnostics.
