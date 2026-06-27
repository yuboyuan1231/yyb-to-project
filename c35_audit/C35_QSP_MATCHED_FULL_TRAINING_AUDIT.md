# C3.5-QSP matched full training audit

Status: `PASS`

- Scope: full `train_fit` training with `train_calib` model selection only
- Matched epoch-0 differences: `0.0` for all three output heads and all variants
- Architecture: `[256, 256, 128]`, dropout `0.1`
- Training: 3 epochs, batch size 1024, AdamW lr `1e-3`, weight decay `0`, seed 13
- Cache: GPU-resident, `3.814888 GiB`
- Runtime: `261.731 s`
- Official val used: `false`
- Score grid started: `false`

## Best train_calib loss

| Variant | Best epoch | Train loss at epoch 3 | Best train_calib loss | Delta vs control |
|---|---:|---:|---:|---:|
| control31 | 3 | 0.0878625317 | 0.0902953030 | 0 |
| qsp52_drop_coverage | 3 | 0.0874305994 | 0.0913944199 | +0.0010991169 |
| qsp53_with_coverage | 3 | 0.0871633040 | 0.0912088289 | +0.0009135259 |

The control31 result exactly reproduces the frozen C3.1 `cap_F_wd0` best
train_calib loss. Both QSP treatments reduce train loss but increase
train_calib loss, indicating mild overfitting or multi-head feature conflict.
QSP53 is better than QSP52 by `0.0001855910` train_calib loss.

## Epoch-3 train_calib components

| Variant | Joint regression | Joint binary | Boundary | False-positive |
|---|---:|---:|---:|---:|
| control31 | 0.0075384859 | 0.1025053621 | 0.0183847317 | 0.0774914898 |
| qsp52_drop_coverage | 0.0076699963 | 0.1038481352 | 0.0190974972 | 0.0773297853 |
| qsp53_with_coverage | 0.0076102429 | 0.1035797299 | 0.0191373615 | 0.0771325840 |

QSP improves the false-positive head, especially with coverage, but degrades
joint and boundary calibration. Aggregate loss alone therefore does not justify
an official-val evaluation.

## Integrity

- All checkpoints load successfully and record `official_val_used=false`.
- No `.partial` artifacts remain.
- Frozen runtime checker after training: `PASS` with dataset paths, frozen
  artifacts, and runtime dependencies all valid.
- No official-val evidence, score search, or post-training calibration was run.

## Recommended next gate

Run train_calib-only prediction diagnostics and matched score calibration for
control31, QSP52, and QSP53. Proceed to an official-val one-shot only if a QSP
variant demonstrates stable downstream ranking gains over both control31 and
the frozen C3.1 configuration. Do not select a variant from training loss alone.
