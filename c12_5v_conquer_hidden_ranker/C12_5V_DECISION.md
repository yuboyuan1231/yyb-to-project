# C12-5V Decision

1. hidden_export_available = true
2. shared_feature_builder = C12_5V_SHARED_FEATURE_BUILDER_PASS
3. trained_variants = V1_hidden_boundary, V2_span_context_hidden, V3_two_head_hidden, V4_short_hidden_specialist
4. best_variant = V1_hidden_boundary
5. best IoU@0.7 top100 = 37.7650
6. best IoU@0.7 top50 = 26.9585
7. short IoU@0.7 top100 = 27.2727
8. PQ Spearman / AUC@0.7 = -0.189612 / 0.493787
9. best-IoU span top100/top50 = 30.2995 / 20.8756
10. allow_enter_c12_6 = false
11. stop_reason = hidden ranker no gain; Spearman remains negative and topK metrics regress vs C12-5T/C12-5U, so the bottleneck is feature/architecture rather than promotion calibration.
12. official_val_used = false
13. evaluator_modified = false; nms_modified = false

status = C12_HIDDEN_RANKER_NO_GAIN_NEED_STRONGER_FEATURES
