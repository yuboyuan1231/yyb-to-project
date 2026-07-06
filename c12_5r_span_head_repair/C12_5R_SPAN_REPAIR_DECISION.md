# C12-5R Span Repair Decision

status = C12_LOCALIZER_PQ_RANKING_BOTTLENECK

best_repair_variant = r1_duration_conditioned_dense

best IoU@0.5 top100 = 63.3180
best IoU@0.7 top100 = 43.4562
short moment IoU@0.7 top100 = 31.3535

allow_enter_c12_6 = false

Reason:
Selected on calib_select only. Holdout IoU@0.7 top100=43.4562; short=31.3535; gap_recovered=0.0144; bottleneck=proposal_quality_ranking.

official_val_used = false
evaluator_modified = false
nms_modified = false
