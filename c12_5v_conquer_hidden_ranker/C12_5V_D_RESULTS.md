# C12-5V-D Results

## Reference Baselines

| variant | queries | IoU@0.5 top100 | IoU@0.7 top100 | IoU@0.7 top50 | short IoU@0.7 top100 | best-IoU top100 | best-IoU top50 | PQ Spearman | AUC@0.7 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| C12_5S_S2_span_context_contrast | 4340 | 71.6590 | 48.3180 | 34.8848 | 32.5657 | 35.1613 | 22.7189 | -0.146034 | 0.535025 |
| C12_5T_T2_listwise_soft_iou | 4340 | 81.2442 | 55.0000 | 38.8710 | 45.0909 | 45.8756 | 29.4931 | -0.193095 | 0.549555 |
| C12_5_teacher_distilled_baseline | 4340 | 62.9954 | 43.0876 | 30.3687 | 30.7475 | 30.9908 | 18.8249 | 0.072880 | 0.536050 |
| U1_two_head_select | 4340 | 70.1382 | 42.6037 | 28.8018 | 32.4444 | 32.7650 | 21.3594 | -0.148115 | 0.583480 |
| U2_two_head_quality | 4340 | 77.0507 | 53.3180 | 38.9631 | 38.9899 | 42.7880 | 27.8341 | -0.148115 | 0.583480 |
| U3_select_then_quality_top100 | 4340 | 70.1382 | 42.6037 | 34.1244 | 32.4444 | 32.7650 | 23.3180 | -0.148115 | 0.583480 |
| U4_duration_bucket_quality | 4340 | 77.0507 | 53.3180 | 38.9631 | 38.9899 | 42.7880 | 27.8341 | -0.212815 | 0.493107 |

## C12-5V Hidden Rankers

| variant | queries | IoU@0.5 top100 | IoU@0.7 top100 | IoU@0.7 top50 | short IoU@0.7 top100 | best-IoU top100 | best-IoU top50 | PQ Spearman | AUC@0.7 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| V1_hidden_boundary | 4340 | 57.3963 | 37.7650 | 26.9585 | 27.2727 | 30.2995 | 20.8756 | -0.189612 | 0.493787 |
| V2_span_context_hidden | 4340 | 48.0415 | 31.7281 | 21.9585 | 22.1414 | 25.5069 | 15.6452 | -0.221136 | 0.474045 |
| V3_two_head_hidden_quality | 4340 | 51.5668 | 35.1152 | 22.6267 | 23.9192 | 28.9401 | 16.9124 | -0.206491 | 0.509077 |
| V3_two_head_hidden_select | 4340 | 51.8433 | 33.5253 | 22.6959 | 24.6465 | 27.0737 | 16.8203 | -0.223262 | 0.470360 |
| V3_two_head_hidden_weighted | 4340 | 55.3226 | 37.0276 | 23.4793 | 26.1010 | 30.0230 | 17.3963 | -0.200423 | 0.488410 |
| V4_short_hidden_specialist | 4340 | 52.3963 | 33.8940 | 24.0092 | 23.4343 | 26.5207 | 18.0876 | -0.225851 | 0.475480 |

## Decision Summary

- hidden_export_available: true
- shared_feature_builder: C12_5V_SHARED_FEATURE_BUILDER_PASS
- trained_variants: V1_hidden_boundary, V2_span_context_hidden, V3_two_head_hidden, V4_short_hidden_specialist
- best_variant: V1_hidden_boundary
- allow_enter_c12_6: false
- status: C12_HIDDEN_RANKER_NO_GAIN_NEED_STRONGER_FEATURES
- official_val_used: false
- evaluator_modified: false
- nms_modified: false
