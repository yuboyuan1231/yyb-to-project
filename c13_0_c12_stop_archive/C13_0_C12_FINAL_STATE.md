# C13-0 C12 Final State

Final status: `C12_SPAN_RANKER_LINE_STOP_NEED_STRONGER_FEATURE_OR_BASELINE`

Current promoted system: `C7-B6 R1SelectiveTop1`

Required state facts:

1. C12 independent native retriever is weak: best C12-4 holdout GT-video top100 is 6.7512, while CONQUER reference is 100.0000.
2. C12-4R uses CONQUER-warm / zero-delta replay bridge: status `C12_RETRIEVER_CONQUER_WARM_READY_CONTINUE_TO_C12_5`, best repaired retriever `zero_delta_replay`.
3. C12 M1000 span generation has oracle headroom: T2 generated M1000 IoU@0.7 is 85.0461.
4. C12 top100 span ranking was partially repaired by T2: IoU@0.7 top100 55.0000, top50 38.8710.
5. C12 calibration / PQ-IoU correlation remains unreliable: T2 Spearman -0.1992, U2 Spearman -0.1481, V1 Spearman -0.1896.
6. C12 hidden ranker regresses: V1 IoU@0.7 top100 37.7650, top50 26.9585.
7. C12-6 is not allowed.
8. Current promoted system remains `C7-B6 R1SelectiveTop1`.

official_val_used = false
evaluator_modified = false
nms_modified = false
