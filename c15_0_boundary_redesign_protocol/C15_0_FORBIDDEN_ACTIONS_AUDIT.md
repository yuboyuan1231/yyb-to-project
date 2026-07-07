# C15-0 Forbidden Actions Audit

C15 has no official-validation path. It reads train/calib splits, C12/C14
train-only artifacts, existing feature caches, and C12 train-only checkpoints.

It does not read official prediction pools, does not modify evaluator/NMS, does
not use pseudo_official_holdout for model selection, and does not enter C12-6.

official was not run.
