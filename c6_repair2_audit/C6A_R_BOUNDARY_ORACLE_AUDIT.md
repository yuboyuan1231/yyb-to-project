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
    "start_ce": 2.873217773602668,
    "end_ce": 2.9551083139090903,
    "start_abs_error": 1.178475081641346,
    "end_abs_error": 1.559846656254437,
    "peak_inside_gt": 0.29674854465426664,
    "oracle_video_r1_05": 0.7715462161010933,
    "oracle_video_r1_07": 0.5013488570211557,
    "selected_span_iou": 0.6593668753657838,
    "best_iou_rank": 5.2550049694732355
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
    "start_ce_delta": -0.3881239911891363,
    "end_ce_delta": -0.3737758835298237,
    "start_abs_error_delta": -0.018174073548203884,
    "end_abs_error_delta": -0.007667187278148413,
    "peak_inside_gt_delta": 0.005963367883004389,
    "oracle_video_r1_05_delta": 0.002129774243930127,
    "oracle_video_r1_07_delta": 0.0015618344455487487,
    "selected_span_iou_delta": 0.0008578522374950159,
    "best_iou_rank_delta": -0.425812863836434
  },
  "oracle_adapter_vs_a3b": {
    "start_ce_delta": -0.5814043913637685,
    "end_ce_delta": -0.556647802770228,
    "start_abs_error_delta": -0.017748118699417725,
    "end_abs_error_delta": -0.006389322731790381,
    "peak_inside_gt_delta": 0.00525344313502768,
    "oracle_video_r1_05_delta": 0.002129774243930127,
    "oracle_video_r1_07_delta": 0.001419849495953418,
    "selected_span_iou_delta": 0.0008601698790120516,
    "best_iou_rank_delta": -0.42396705949169444
  }
}
```
