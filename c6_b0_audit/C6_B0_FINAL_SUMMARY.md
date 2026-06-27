# C6-B0 final summary

- Status: `C6_B0_NEGATIVE`
- Scope: `train_calib-only candidate-generation diagnostic`; not a final VCMR result.
- Official val used: `false`; C6_B1 used: `false`; C6_C used: `false`.
- C4_final retained: `C4-r2-cal-v2.1 v21_00444`.
- Readout policy: `R@1` is primary because recent papers emphasize R1; `R@5` is the main safety/secondary check.

## Key observations

- Fixed-video coverage gap: `1696` / `8739` train_calib queries have no GT video in the fixed candidate video list, so B0 cannot rescue them by design.
- Boundary candidates vs original GT-video oracle: R@0.5 `+1.373155`, R@0.7 `+2.288591`, best-IoU mean `-0.008190`.
- Hybrid candidates vs original GT-video oracle: R@0.5 `+1.373155`, R@0.7 `+2.300034`, best-IoU mean `-0.008154`.
- Fixed-video pseudo VCMR boundary/hybrid delta: 0.5-R@1 `+0.183087`, 0.7-R@1 `+0.617920`.
- R@5 safety failed: 0.5-R@5 `-3.810505`, 0.7-R@5 `-1.109967`.
- R@100 improves, but that does not override R@5 collapse: 0.5-R@100 `+2.059732`, 0.7-R@100 `+4.989129`.
- Candidate replacement rate: boundary `0.884079`, hybrid `0.884077`; video multiset/slot drift is `0.0` by construction.

## Decision

- Do not enter C6-B1 automatically.
- Keep C4_final as the main frozen result.
- B0 shows the boundary distribution can move R@1 upward, but the same replacement causes a large R@5/R@10 front-rank collapse under fixed video slots.
- The likely next idea, if manually authorized later, should not be another fixed-candidate adapter tuning pass; it would need a safer candidate-generation interface that preserves front-rank diversity.

## Decision JSON

```json
{
  "C6_B1_used": false,
  "C6_C_used": false,
  "official_val_used": false,
  "r1_primary_r5_safety_policy": "R@1 is the primary observation; R@5 is a safety/secondary constraint. B0 is not a final VCMR result.",
  "status": "C6_B0_NEGATIVE",
  "variant_decisions": {
    "boundary_oracle_candidates": {
      "candidate_generation_oracle_positive": false,
      "no_severe_r5_collapse": false,
      "r1_primary_delta_min": 0.1830873097608432,
      "r5_safety_delta_min": -3.8105046343975317,
      "recommend_c6_b1_review": false
    },
    "hybrid_candidates": {
      "candidate_generation_oracle_positive": false,
      "no_severe_r5_collapse": false,
      "r1_primary_delta_min": 0.1830873097608432,
      "r5_safety_delta_min": -3.8105046343975317,
      "recommend_c6_b1_review": false
    }
  }
}
```
