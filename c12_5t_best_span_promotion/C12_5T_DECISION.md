# C12-5T Best-Span Promotion Decision

status = C12_BEST_SPAN_PROMOTION_STILL_WEAK_NEED_STRONGER_BOUNDARY_MODEL

best_T_variant = T2_listwise_soft_iou

best IoU@0.5 top100 = 81.2212
best IoU@0.7 top100 = 55.0000
best IoU@0.7 top50 = 38.8710
short moment IoU@0.7 top100 = 45.0909

delta_vs_c12_5s_iou07_top100 = 6.6820
allow_enter_c12_6 = false

Reason:
Selected on calib_select only. Holdout top100 IoU@0.7=55.0000; C12-5S ref=48.3180; C12-5 baseline in this run=43.0876; generated M1000 remains 85.0461; Spearman=-0.199218575616094; AUC@0.7=0.547285.

official_val_used = false
evaluator_modified = false
nms_modified = false
