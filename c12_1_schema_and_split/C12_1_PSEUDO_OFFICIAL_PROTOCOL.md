# C12-1 Pseudo-Official Protocol

`pseudo_official_holdout` is a train-only holdout that mimics official discipline:

1. Do not train on it.
2. Do not select features, thresholds, alpha, safety rules, or checkpoints on it.
3. Evaluate each large branch on it once after the branch is frozen.
4. If the frozen branch fails, archive the failure and revise the next branch
   from train_fit/calib_select/calib_holdout evidence only.
5. Never use pseudo-official feedback to tune the same candidate.

This protocol exists to reduce the train-only optimism that appeared before C9.
