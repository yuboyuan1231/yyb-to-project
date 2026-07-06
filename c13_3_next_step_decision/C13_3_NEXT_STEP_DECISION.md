# C13-3 Next Step Decision

Selected mainline: `Path B`

Status: `C13_STRONG_FEATURE_EXTRACTION_THEN_NATIVE_RETRAIN`

Rationale:

- C12 is stopped: `C12_SPAN_RANKER_LINE_STOP_NEED_STRONGER_FEATURE_OR_BASELINE`.
- Strong text/subtitle features are worth a pilot.
- Strong visual features are worth a pilot, with higher extraction/storage risk.
- Multiscale/event features are the highest-value pivot because C12 M1000 has oracle headroom but rank/calibration fails.
- PREM is not directly reproducible from available public code, so Path A is not selected.
- MA-VR/EventFormer/MINUTE are borrowable module/objective ideas.
- Boundary/localizer redesign is likely needed, but should be guided by strong-feature pilot results.

official_val_used = false
evaluator_modified = false
nms_modified = false
