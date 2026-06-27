# C6-A-R A3b teacher audit

- Status: `FAIL`
- Official val used: `false`
- Endpoint delta correlation with A3b raw: `{'pearson': -0.47737492034476703, 'spearman': -0.40910982150592745}`

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
    "start_ce": 2.9664591155262787,
    "end_ce": 3.0529368561708052,
    "start_abs_error": 1.1970751100383359,
    "end_abs_error": 1.567229873633395,
    "peak_inside_gt": 0.29206304131762034,
    "oracle_video_r1_05": 0.7691324719579724,
    "oracle_video_r1_07": 0.5000709924747977,
    "selected_span_iou": 0.6584065900750572,
    "best_iou_rank": 5.449808320318046
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
  "teacher_adapter_vs_frozen": {
    "start_ce_delta": -0.29488264926552565,
    "end_ce_delta": -0.2759473412681088,
    "start_abs_error_delta": 0.0004259548487859366,
    "end_abs_error_delta": -0.0002839698991905504,
    "peak_inside_gt_delta": 0.0012778645463580873,
    "oracle_video_r1_05_delta": -0.0002839698991907724,
    "oracle_video_r1_07_delta": 0.0002839698991907169,
    "selected_span_iou_delta": -0.00010243305323154317,
    "best_iou_rank_delta": -0.23100951299162364
  },
  "teacher_adapter_vs_a3b": {
    "start_ce_delta": -0.48816304944015787,
    "end_ce_delta": -0.45881926050851307,
    "start_abs_error_delta": 0.0008519096975720952,
    "end_abs_error_delta": 0.0009938946471674814,
    "peak_inside_gt_delta": 0.0005679397983813783,
    "oracle_video_r1_05_delta": -0.0002839698991907724,
    "oracle_video_r1_07_delta": 0.0001419849495953862,
    "selected_span_iou_delta": -0.00010011541171450755,
    "best_iou_rank_delta": -0.22916370864688407
  }
}
```
