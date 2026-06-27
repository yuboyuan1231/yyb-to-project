# C3.5 matched extended epoch-8 audit

Status: `PASS`  
Scope: `train_fit/train_calib only`  
Official val used: `false`  
Score grid / temperature search: `false / false`

The three variants were retrained from the same seed-13 matched initialization
for eight epochs. Architecture, features, losses, optimizer, batch size, split,
and shuffle protocol remain unchanged.

## Train/calib loss curves

| Epoch | control train | control calib | QSP52 train | QSP52 calib | QSP53 train | QSP53 calib |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.09183025 | 0.09197088 | 0.09148642 | 0.09267575 | 0.09126078 | 0.09292179 |
| 2 | 0.08866731 | 0.09161764 | 0.08826710 | 0.09211962 | 0.08808532 | 0.09189881 |
| 3 | 0.08786253 | 0.09029530 | 0.08743060 | 0.09139442 | 0.08716330 | 0.09120883 |
| 4 | 0.08735833 | 0.09084936 | 0.08684748 | 0.09154246 | 0.08658337 | 0.09128936 |
| 5 | 0.08691546 | 0.08996411 | 0.08634860 | **0.09035582** | 0.08615299 | **0.09046468** |
| 6 | 0.08658714 | 0.09098865 | 0.08598895 | 0.09139915 | 0.08568233 | 0.09143677 |
| 7 | 0.08633659 | **0.08990812** | 0.08570266 | 0.09168366 | 0.08544691 | 0.09060368 |
| 8 | 0.08602116 | 0.09055401 | 0.08534747 | 0.09113078 | 0.08503353 | 0.09164348 |

Best epochs are control31 `7`, QSP52 `5`, and QSP53 `5`. Training loss keeps
falling, while train_calib loss worsens after the selected checkpoint. The
epoch-5 QSP checkpoints selected by the epoch-8 run are tensor-identical to the
previous epoch-5 checkpoints (`max_abs_diff=0`). This resolves the earlier
under-training concern for both QSP variants.

Best train_calib loss is control31 `0.08990812`, QSP52 `0.09035582`, and QSP53
`0.09046468`. QSP52/QSP53 remain worse than control by `0.00044771` and
`0.00055656`.

## Selected-checkpoint head diagnostics

| Variant (epoch) | Qjoint Pearson | Qjoint Spearman | Qjoint AUC@0.5 | Qbd Pearson | Qbd Spearman | Efp AUC |
|---|---:|---:|---:|---:|---:|---:|
| control31 (7) | 0.523927 | **0.352475** | **0.912233** | **0.632227** | **0.620304** | **0.988290** |
| QSP52 (5) | 0.524042 | 0.348007 | 0.908500 | 0.628742 | 0.617611 | 0.988185 |
| QSP53 (5) | **0.524613** | 0.348513 | 0.908889 | 0.626911 | 0.614867 | 0.988165 |

No head collapses and no R2 feature is enabled. Apart from a tiny Qjoint
Pearson edge for QSP53, control is stronger on the ranking-relevant AUC and
boundary diagnostics.

## Fixed C3.1 score probe

Exactly one frozen score configuration was evaluated per selected checkpoint:
`C_centered_gated_boundary`, weights `0.75/1.75/1.0/0.5`, temperatures
`1.5/1.0/1.0`, effective top-100, NMS 0.7. No score grid or temperature search
was performed.

| Variant | 0.5 R1 | R5 | R10 | R100 | 0.7 R1 | R5 | R10 | R100 | Selection score |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| control31 epoch 7 | 37.83 | **61.72** | **68.83** | **76.07** | **25.74** | **46.53** | **54.01** | **64.60** | **4.769329** |
| QSP52 epoch 5 | 37.90 | 61.51 | 68.68 | **76.07** | 25.60 | 46.16 | 53.88 | 64.47 | 4.599115 |
| QSP53 epoch 5 | **37.91** | 61.52 | 68.61 | 76.05 | 25.43 | 46.15 | 53.93 | 64.46 | 4.564309 |

All three probes satisfy the frozen safety constraints. QSP52 and QSP53 trail
control by `0.170214` and `0.205020` selection score. Neither QSP treatment
exceeds control after fair checkpoint selection.

## Decision

The QSP deficit is not explained by a three-epoch training budget. QSP improves
training loss but begins to overfit after epoch 5 and does not improve the fixed
downstream score probe. The matched QSP treatment should therefore be recorded
as a negative result under the current 52/53-feature MLP formulation.

Recommendation: stop C3.5-QSP here and do not request an official-val QSP
one-shot. The epoch-7 scalar control is a separate train_calib-only observation;
it does not modify or replace frozen C3.1 artifacts and is not authorization for
another official-val evaluation.

Runtime/frozen-artifact checker: `PASS`. No partial artifacts remain. Epoch 9+
training was not started.
