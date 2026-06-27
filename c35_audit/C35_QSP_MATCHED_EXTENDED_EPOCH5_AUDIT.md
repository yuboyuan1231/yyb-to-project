# C3.5 matched extended-epoch check

Status: `PASS`  
Scope: `train_fit/train_calib only`  
Official val used: `false`  
Score grid / temperature search: `false / false`

All three variants were retrained from the same matched seed-13 initialization.
The architecture, dropout, losses, optimizer, learning rate, batch size, data
splits, and shuffle protocol are identical to the 3-epoch run.

## Train/calib loss curves

| Epoch | control train | control calib | QSP52 train | QSP52 calib | QSP53 train | QSP53 calib |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.09183025 | 0.09197088 | 0.09148642 | 0.09267575 | 0.09126078 | 0.09292179 |
| 2 | 0.08866731 | 0.09161764 | 0.08826710 | 0.09211962 | 0.08808532 | 0.09189881 |
| 3 | 0.08786253 | 0.09029530 | 0.08743060 | 0.09139442 | 0.08716330 | 0.09120883 |
| 4 | 0.08735833 | 0.09084936 | 0.08684748 | 0.09154246 | 0.08658337 | 0.09128936 |
| 5 | 0.08691546 | **0.08996411** | 0.08634860 | **0.09035582** | 0.08615299 | **0.09046468** |

All variants select epoch 5. Train loss decreases monotonically. Calib loss
briefly rebounds at epoch 4, then reaches a new minimum at epoch 5. Relative to
epoch 3, epoch-5 calib loss improves by `0.00033119` for control, `0.00103860`
for QSP52, and `0.00074415` for QSP53.

At epoch 5, QSP52/QSP53 aggregate calib loss remains above control by
`0.00039171`/`0.00050056` respectively.

## Head diagnostics

| Run | Variant | Qjoint Pearson | Qjoint Spearman | Qjoint AUC@0.5 | Qbd Pearson | Qbd Spearman | Efp AUC |
|---|---|---:|---:|---:|---:|---:|---:|
| epoch 3 | control31 | 0.532552 | 0.353684 | 0.910759 | 0.632886 | 0.619865 | 0.987945 |
| epoch 3 | QSP52 | 0.523135 | 0.352307 | 0.907796 | 0.614126 | 0.601572 | 0.988024 |
| epoch 3 | QSP53 | 0.526539 | 0.351881 | 0.907993 | 0.613631 | 0.602446 | 0.988079 |
| epoch 5 | control31 | 0.525670 | 0.348467 | 0.910263 | 0.634372 | 0.622316 | 0.988150 |
| epoch 5 | QSP52 | 0.524042 | 0.348007 | 0.908500 | 0.628742 | 0.617611 | **0.988185** |
| epoch 5 | QSP53 | 0.524613 | 0.348513 | 0.908889 | 0.626911 | 0.614867 | 0.988165 |

No head collapses and no R2 feature is enabled. QSP closes much of the boundary
diagnostic gap by epoch 5 and gives a tiny Efp AUC gain, but control retains the
best Qjoint AUC and boundary correlations.

## Fixed C3.1 score probe

Exactly one frozen configuration was evaluated for each checkpoint:
`C_centered_gated_boundary`, weights `0.75/1.75/1.0/0.5`, temperatures
`1.5/1.0/1.0`, effective top-100, NMS 0.7. No score or temperature search was
performed. The boundary center is the corresponding model's train_calib mean.

### Epoch-5 metrics

| Variant | 0.5 R1 | R5 | R10 | R100 | 0.7 R1 | R5 | R10 | R100 | Selection score |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| control31 | 37.82 | 61.63 | 68.78 | 76.04 | 25.49 | 46.16 | 53.80 | 64.46 | 4.587672 |
| QSP52 | **37.90** | 61.51 | 68.68 | **76.07** | **25.60** | 46.16 | 53.88 | **64.47** | **4.599115** |
| QSP53 | 37.91 | 61.52 | 68.61 | 76.05 | 25.43 | 46.15 | **53.93** | 64.46 | 4.564309 |

All epoch-5 probes pass the frozen safety constraints.

### Epoch-3 to epoch-5 selection score

| Variant | Epoch 3 | Epoch 5 | Delta |
|---|---:|---:|---:|
| control31 | 4.678739 | 4.587672 | -0.091067 |
| QSP52 | 4.502804 | 4.599115 | +0.096312 |
| QSP53 | 4.466091 | 4.564309 | +0.098219 |

QSP52 exceeds the epoch-5 control by `+0.011443` selection score, with five
metrics higher, one tied, and two lower. It does **not** exceed the frozen
epoch-3 control/C3.1 probe (`4.678739`). QSP53 does not exceed the epoch-5
control.

## Decision

There is still an under-training indication: every variant selects the final
epoch, epoch 5 establishes a new calib-loss minimum, and both QSP fixed probes
continue to improve materially from epoch 3. The calib curves are not strictly
monotonic because all three rebound at epoch 4, so the evidence is suggestive
rather than conclusive.

Recommendation: request authorization for one matched epoch-8 check covering
all three variants under the identical protocol. Epoch 8 was **not** started.
No official-val evidence was read and no official-val metric was used.

Runtime/frozen-artifact checker after completion: `PASS`. No partial artifacts
remain.
