# C6-A-R2 protocol

C6-A-R2 uses the passed BoundaryOracleAdapter as initialization and removes A3b teacher loss from the main training objective. A3b is retained only as a diagnostic/reference. Training remains train_fit-only with train_calib selection; official val, candidate regeneration, C6-B, C6-C, full backbone finetuning, and post-val adjustment are forbidden.
