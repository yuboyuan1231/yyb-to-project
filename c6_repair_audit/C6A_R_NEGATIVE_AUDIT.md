# C6-A-R negative audit

- Status: `C6A_R_NEGATIVE`
- Stop phase: `Phase 3 A3b teacher distillation`
- Official val used: `false`
- Post-val adjustment: `false`

## Summary

C6-A-R confirmed that the boundary-only adapter can improve GT-video localization, but the A3b teacher-distilled adapter did not inherit the teacher endpoint-delta direction. Therefore Phase 4 ranking/fusion was not authorized or executed.

## Phase 1 diagnostics

### c6a_default
- endpoint_delta stats: `{'max_abs': 0.10437822341918945, 'mean': -0.003008940561902759, 'std': 0.026863338209343047}`
- rank_residual stats: `{'max_abs': 1.0180023908615112, 'mean': 0.5341375407392848, 'std': 0.17256083601302877}`
- endpoint_delta vs A3b z correlation: `{'pearson': 0.26677290360087774, 'spearman': 0.2730326393591653}`
- grad norms: `{'BoundaryAdapter': 0.23420390860802257, 'FinalFusionHead': 12.317531931408203, 'SpanQualityHead': 11.251524642594697, 'TemporalPriorEncoder': 5.006981713883046, 'VideoGate': 0.00829122297775643}`

### c6a_large
- endpoint_delta stats: `{'max_abs': 0.2228851318359375, 'mean': 0.008784073241950897, 'std': 0.03933125611368055}`
- rank_residual stats: `{'max_abs': 0.5106446743011475, 'mean': -0.09267647334202965, 'std': 0.09789223938123334}`
- endpoint_delta vs A3b z correlation: `{'pearson': -0.270472389277007, 'spearman': -0.27062967824826817}`
- grad norms: `{'BoundaryAdapter': 0.4029485700146408, 'FinalFusionHead': 16.182698746194518, 'SpanQualityHead': 14.629385249498371, 'TemporalPriorEncoder': 6.906406425138589, 'VideoGate': 0.002944923200944918}`

### c6a_regularized
- endpoint_delta stats: `{'max_abs': 0.2406296730041504, 'mean': -0.006017755223735826, 'std': 0.04298708679152546}`
- rank_residual stats: `{'max_abs': 0.4286092221736908, 'mean': 0.06046180924913836, 'std': 0.11307450236800277}`
- endpoint_delta vs A3b z correlation: `{'pearson': 0.09272188639622554, 'spearman': 0.08689395144805621}`
- grad norms: `{'BoundaryAdapter': 0.44172235705523516, 'FinalFusionHead': 12.430321164006887, 'SpanQualityHead': 11.803573937694658, 'TemporalPriorEncoder': 3.872127041933481, 'VideoGate': 0.006329939810355827}`

## Phase 2 boundary oracle

- Status: `PASS`
- Key deltas vs frozen:
  - `best_iou_rank_delta`: `-0.4789152349850916`
  - `end_abs_error_delta`: `-0.021155757489706106`
  - `end_ce_delta`: `-0.37566782245836805`
  - `oracle_video_r1_05_delta`: `0.0018458043447394656`
  - `oracle_video_r1_07_delta`: `0.0028396989919068916`
  - `peak_inside_gt_delta`: `0.004543518387050971`
  - `selected_span_iou_delta`: `0.0011520798655415554`
  - `start_abs_error_delta`: `-0.0029816839415022223`
  - `start_ce_delta`: `-0.3860126394049188`

## Phase 3 teacher adapter

- Status: `FAIL`
- Endpoint delta correlation with A3b raw: `{'pearson': -0.47737492034476703, 'spearman': -0.40910982150592745}`
- Key deltas vs frozen:
  - `best_iou_rank_delta`: `-0.23100951299162364`
  - `end_abs_error_delta`: `-0.0002839698991905504`
  - `end_ce_delta`: `-0.2759473412681088`
  - `oracle_video_r1_05_delta`: `-0.0002839698991907724`
  - `oracle_video_r1_07_delta`: `0.0002839698991907169`
  - `peak_inside_gt_delta`: `0.0012778645463580873`
  - `selected_span_iou_delta`: `-0.00010243305323154317`
  - `start_abs_error_delta`: `0.0004259548487859366`
  - `start_ce_delta`: `-0.29488264926552565`

- Key deltas vs A3b:
  - `best_iou_rank_delta`: `-0.22916370864688407`
  - `end_abs_error_delta`: `0.0009938946471674814`
  - `end_ce_delta`: `-0.45881926050851307`
  - `oracle_video_r1_05_delta`: `-0.0002839698991907724`
  - `oracle_video_r1_07_delta`: `0.0001419849495953862`
  - `peak_inside_gt_delta`: `0.0005679397983813783`
  - `selected_span_iou_delta`: `-0.00010011541171450755`
  - `start_abs_error_delta`: `0.0008519096975720952`
  - `start_ce_delta`: `-0.48816304944015787`

## Decision

Do not proceed to C6-A-R final ranking-aware training, do not run official val, and do not enter C6-B automatically. The repair suggests boundary learning is possible, but teacher-guided endpoint residual alignment is still unresolved under the current adapter formulation.
