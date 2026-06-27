# C3.5-QSP matched control/treatment smoke audit

Status: `PASS`

- Scope: `train_fit/train_calib` only
- Full training started: `false`
- Official val used: `false`
- Canonical cache: `/tmp/yyb_nvme_c35/cache/c35_qsp53`
- Cache rows: train_fit `15,609,000`; train_calib `1,747,800`
- Cache shapes: `53` features and `6` labels; all finite
- GPU-resident cache: `3.814888 GiB`

## Matched gates

- Epoch-0 q_joint logit max absolute difference: `0.0` for control31, qsp52, qsp53
- CPU-memmap vs GPU-resident input max absolute difference: `0.0`
- Post-training state max absolute difference: `2.384185791015625e-07` (gate `1e-6`)
- Post-training metric max absolute difference: `4.3655745685100555e-10` (gate `1e-8`)

The post-training tolerance is required because distinct CUDA memory/gather paths can
change floating-point reduction order. Inputs remain bit-identical; the observed model
and metric differences are below the frozen smoke tolerances.

## One-epoch smoke results

| Variant | Data path | Train loss | Calib loss | Runtime (s) |
|---|---|---:|---:|---:|
| control31_cpu | CPU memmap to CUDA | 0.1110464341 | 0.0987353619 | 1.143 |
| control31 | GPU resident | 0.1110464337 | 0.0987353621 | 1.045 |
| qsp52_drop_coverage | GPU resident | 0.1106693340 | 0.0998587459 | 1.009 |
| qsp53_with_coverage | GPU resident | 0.1105806928 | 0.0997718090 | 1.015 |

These losses establish execution integrity only. One epoch on the smoke subset is not
a model-selection result: both QSP variants have slightly lower train loss but slightly
higher calibration loss than control31.

## CPU concurrency policy

- CPU-heavy parsing/vectorization defaults to multiple ordered workers (up to 12 on this 16-thread host).
- Independent cache integrity scans run concurrently.
- A single gzip stream still has a serial decompression producer; future repeated processing should use pre-sharded uncompressed/cache inputs so workers do not wait on it.
- Model training should use the verified GPU-resident cache path when memory permits.

Runtime checker after smoke: `dataset_paths_ok=true`, `frozen_artifacts_ok=true`,
`runtime_dependencies_ok=true`.

Stop condition honored: no full C3.5 training, score search, or official-val work was run.
