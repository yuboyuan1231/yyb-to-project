# C3.5-QSP matched epoch-8 final audit

Status: `PASS`  
Decision: `STOP_C35_QSP_NEGATIVE`  
Scope: `train_fit/train_calib only`  
Official val used: `false`

## Frozen protocol

- Variants: control31, QSP52 without `qsp_sub_coverage`, QSP53 with coverage
- Matched seed-13 initialization; epoch-0 logits exactly identical
- Hidden dimensions `[256, 256, 128]`, dropout `0.1`
- AdamW, learning rate `0.001`, weight decay `0`
- Batch size `1024`; maximum epoch `8`
- Checkpoint selection used train_calib only
- Fixed C3.1 score probe only; no score grid or temperature search

## Loss curves and selected checkpoints

| Epoch | control calib | QSP52 calib | QSP53 calib |
|---:|---:|---:|---:|
| 1 | 0.09197088 | 0.09267575 | 0.09292179 |
| 2 | 0.09161764 | 0.09211962 | 0.09189881 |
| 3 | 0.09029530 | 0.09139442 | 0.09120883 |
| 4 | 0.09084936 | 0.09154246 | 0.09128936 |
| 5 | 0.08996411 | **0.09035582** | **0.09046468** |
| 6 | 0.09098865 | 0.09139915 | 0.09143677 |
| 7 | **0.08990812** | 0.09168366 | 0.09060368 |
| 8 | 0.09055401 | 0.09113078 | 0.09164348 |

Selected epochs are control31 `7`, QSP52 `5`, and QSP53 `5`. QSP52 and
QSP53 train_calib loss deltas versus control are `+0.0004477061` and
`+0.0005565598`; lower is better. The QSP variants overfit after epoch 5, so
their deficit is not caused by a three-epoch budget.

## Fixed C3.1 score probe

Frozen score family: `C_centered_gated_boundary`; weights
`base/joint/boundary/fp = 0.75/1.75/1.0/0.5`; temperatures
`joint/boundary/fp = 1.5/1.0/1.0`; effective top-100 and NMS `0.7`.

| Variant | 0.5 R1 | R5 | R10 | R100 | 0.7 R1 | R5 | R10 | R100 | Selection score |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| control31 | 37.83 | 61.72 | 68.83 | 76.07 | 25.74 | 46.53 | 54.01 | 64.60 | **4.769329** |
| QSP52 | 37.90 | 61.51 | 68.68 | 76.07 | 25.60 | 46.16 | 53.88 | 64.47 | 4.599115 |
| QSP53 | 37.91 | 61.52 | 68.61 | 76.05 | 25.43 | 46.15 | 53.93 | 64.46 | 4.564309 |

Selection-score deltas versus control are QSP52 `-0.1702139833` and QSP53
`-0.2050196437`. Both treatments satisfy safety constraints but neither
outperforms the matched control.

## Final decision

The current 52/53-feature QSP MLP formulation is frozen as a negative result.
No official-val QSP one-shot is authorized. Score grid, temperature search,
epoch 9+, QSP capacity search, R2, REG, AMD, C4, C5, and C6 are not authorized.
C3.1 frozen artifacts remain unchanged.

