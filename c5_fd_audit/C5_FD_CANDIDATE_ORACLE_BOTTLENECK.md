# C5 failure decomposition: candidate/oracle bottleneck

The frozen C4_final ordering and unchanged NMS=0.7 are replayed read-only. c5p_00138 is used only to explain failure.

## Candidate availability

| Diagnostic | Value |
|---|---:|
| `gt_video_present_query_count` | 7043 |
| `gt_video_has_iou05_candidate_ratio` | 0.96265796 |
| `gt_video_has_iou07_candidate_ratio` | 0.85616925 |

## Best-IoU rank

| Diagnostic | Value |
|---|---:|
| `C4_final mean` | 5.72256141 |
| `prior-only mean` | 12.50163283 |
| `rejected c5p_00138 mean` | 5.71461025 |
| `c5p_00138 improved ratio` | 0.02087179 |
| `c5p_00138 worsened ratio` | 0.01249468 |

## Top100/NMS attribution

| Diagnostic | Value |
|---|---:|
| `best_iou_candidate_in_pre_nms_top100_ratio` | 0.94107625 |
| `best_iou_candidate_survives_nms_ratio` | 0.56637796 |
| `best_iou_candidate_removed_by_nms_ratio` | 0.37469828 |
| `better_than_c4_selected_span_removed_by_nms_ratio` | 0.23044157 |

## Localization oracle

| Diagnostic | Value |
|---|---:|
| `raw-200 mIoU` | 0.83218340 |
| `raw-200 R1@0.5` | 96.26579583 |
| `raw-200 R1@0.7` | 85.61692461 |
| `post-NMS mIoU` | 0.78770353 |
| `post-NMS R1@0.5` | 94.57617493 |
| `post-NMS R1@0.7` | 80.93142127 |
