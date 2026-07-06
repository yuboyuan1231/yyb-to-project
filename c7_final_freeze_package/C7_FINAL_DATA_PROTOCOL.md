# C7 Final Data Protocol

- dataset: Original TVR / VCMR
- train_fit query count: 2048 for final Region checkpoint/larger confirmation
- train_calib query count: 1024 for larger confirmation
- first-stage topK videos: 100
- top-M proposals: 200
- candidate generation mode: train-only first-stage top100 CONQUER candidate pool plus integrated RLEM/Region scoring, followed by fixed R1-priority composition
- NMS threshold: 0.7
- feature release: /home/a/yybwork/data/yyb/tvr_feature_release
- base checkpoint: results/tvr-conquer_c0_repro_20260621/model.ckpt
- official evaluator path: standalone_eval/eval.py
- official split path: /home/a/yybwork/data/yyb/tvr_feature_release/data/tvr_val_release.jsonl

Train-only result is not official result. Official val has not been run in this package.
