# C6-A-R2 negative audit

- Status: `C6A_R2_NEGATIVE`
- Selected config: `c6a_r2_safe`
- Selected epoch: `3`
- Selection score: `-0.6180705429905908`
- Official val used: `false`
- Post-val adjustment: `false`

## Interpretation

Boundary initialization reproduced the passed boundary-oracle behavior, but ranking-aware continuation failed to convert it into safe fixed-candidate front-rank gains. Default/large produced strong R@1 gains with severe R@5/R@10 collapse; conservative safe scaling reduced but did not remove the negative front-rank pattern. Therefore freeze gate failed and official val is not authorized.

## Selected deltas vs C4_final
- `0.5-r1`: `-0.011442956860051368`
- `0.5-r10`: `-1.0298661174047368`
- `0.5-r100`: `-0.05721478430025684`
- `0.5-r5`: `-1.006980203684634`
- `0.7-r1`: `-0.08010069802037023`
- `0.7-r10`: `-0.17164435290078472`
- `0.7-r100`: `0.22885913720104156`
- `0.7-r5`: `-0.30895983522142245`

## Selected freeze gates
- `front_positive_count`: `0`
- `r1_gate`: `False`
- `r100_gate`: `False`
- `localization_positive_count`: `3`
- `movement_gate`: `True`
- `boundary_retention_gate`: `True`
- `best_iou_rank_gate`: `True`
- `core_gate`: `False`
- `strong_gate`: `False`
