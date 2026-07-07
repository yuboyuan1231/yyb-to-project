# C12-5R-A Coverage Alignment Audit

same_query_comparison_available = true

C12 holdout query count = 4340
C7-B6 reference query count = 1024
same-query intersection count = 48
aligned_delta_iou05_top100 = -31.25
aligned_delta_iou07_top100 = -25.0

Reason for B6 reference rows = C8 exact-B6 train feature re-export contains a 1024-row train-only teacher/comparison subset.

If intersection is small/empty, B6 comparison remains reference-only, not a strict gate.
official_val_used = false
