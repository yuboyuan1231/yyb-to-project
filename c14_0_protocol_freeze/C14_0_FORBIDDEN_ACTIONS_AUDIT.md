# C14-0 Forbidden Actions Audit

No official validation path exists in C14. The script only reads train-release
metadata, train/calib splits, existing C12/C13 artifacts, feature LMDB/NPZ
caches, and local model checkpoints for train-only small-sample pilots.

Root stale authorization markers: `[]`.

If root stale markers appear, quarantine should be recommended, but historical
`OFFICIAL_VAL_ALREADY_RUN` markers inside official output directories must not be
deleted.

official was not run.
