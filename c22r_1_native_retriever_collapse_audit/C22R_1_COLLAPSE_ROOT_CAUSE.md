# C22R-1 Collapse Root Cause

Status: `C22R_COLLAPSE_ROOT_CAUSE_FOUND`.

C22 collapsed because the native retriever was allowed to replace the very strong first-stage retriever over the full train video library. The release-feature global/pooled native scorer is weak, while first-stage top128 already gives 100% GT-video coverage on the medium holdout. Alignment checks did not find a blocking key/join/schema mismatch; the main failure is candidate-space misuse plus an unconstrained residual/replacement policy.
