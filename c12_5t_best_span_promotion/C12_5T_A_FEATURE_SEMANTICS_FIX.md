# C12-5T-A Feature Semantics Fix

status = C12_5T_FEATURE_SEMANTICS_PASS

CONQUER hidden status = C12_CONQUER_HIDDEN_NOT_AVAILABLE_USE_RAW_LOGIT_FEATURES

`retriever_score_z` now means candidate-video first-stage score z-normalized within the query ranklist.
In GT-video oracle evaluation, the GT video is passed as the candidate video, preserving comparability without using a GT-specific feature builder.

No GT/IoU/label feature is used at inference.
No official prediction pool is read.
C7-B6 fixed span pool is not used as final candidates.

official_val_used = false
evaluator_modified = false
nms_modified = false
