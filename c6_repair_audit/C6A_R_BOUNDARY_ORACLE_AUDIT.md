# C6-A-R boundary oracle audit

- Status: `PASS`
- Official val used: `false`
- Train positive groups: `63090`
- Train_calib positive groups: `7043`

## Metrics

```json
{
  "frozen": {
    "start_ce": 3.2613417647918044,
    "end_ce": 3.328884197438914,
    "start_abs_error": 1.19664915518955,
    "end_abs_error": 1.5675138435325855,
    "peak_inside_gt": 0.29078517677126225,
    "oracle_video_r1_05": 0.7694164418571632,
    "oracle_video_r1_07": 0.499787022575607,
    "selected_span_iou": 0.6585090231282887,
    "best_iou_rank": 5.6808178333096695
  },
  "oracle_adapter": {
    "start_ce": 2.8753291253868856,
    "end_ce": 2.953216374980546,
    "start_abs_error": 1.1936674712480477,
    "end_abs_error": 1.5463580860428794,
    "peak_inside_gt": 0.2953286951583132,
    "oracle_video_r1_05": 0.7712622462019026,
    "oracle_video_r1_07": 0.5026267215675139,
    "selected_span_iou": 0.6596611029938303,
    "best_iou_rank": 5.201902598324578
  },
  "a3b_teacher": {
    "start_ce": 3.4546221649664366,
    "end_ce": 3.5117561166793183,
    "start_abs_error": 1.1962232003407638,
    "end_abs_error": 1.5662359789862275,
    "peak_inside_gt": 0.29149510151923896,
    "oracle_video_r1_05": 0.7694164418571632,
    "oracle_video_r1_07": 0.4999290075252023,
    "selected_span_iou": 0.6585067054867717,
    "best_iou_rank": 5.67897202896493
  },
  "oracle_adapter_vs_frozen": {
    "start_ce_delta": -0.3860126394049188,
    "end_ce_delta": -0.37566782245836805,
    "start_abs_error_delta": -0.0029816839415022223,
    "end_abs_error_delta": -0.021155757489706106,
    "peak_inside_gt_delta": 0.004543518387050971,
    "oracle_video_r1_05_delta": 0.0018458043447394656,
    "oracle_video_r1_07_delta": 0.0028396989919068916,
    "selected_span_iou_delta": 0.0011520798655415554,
    "best_iou_rank_delta": -0.4789152349850916
  },
  "oracle_adapter_vs_a3b": {
    "start_ce_delta": -0.579293039579551,
    "end_ce_delta": -0.5585397416987723,
    "start_abs_error_delta": -0.0025557290927160636,
    "end_abs_error_delta": -0.019877892943348074,
    "peak_inside_gt_delta": 0.003833593639074262,
    "oracle_video_r1_05_delta": 0.0018458043447394656,
    "oracle_video_r1_07_delta": 0.002697714042311561,
    "selected_span_iou_delta": 0.001154397507058591,
    "best_iou_rank_delta": -0.477069430640352
  }
}
```
