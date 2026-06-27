# C5-lite-prior negative audit

## Frozen conclusion

- Status: `NO_PROMOTION`
- Classification: `C5-lite-prior train_calib negative`
- Primary baseline: `C4_final = C4-r2-cal-v2.1 v21_00444`
- Selected strict-safe configuration: `c5p_00000`, the zero-residual control
- Nonzero residual promoted: `false`
- Official val used/authorized: `false / false`
- Post-val adjustment: `false`

The temporal-prior path is a real frozen-model export, not a row-level reconstruction. `P_b` and `P_e` are softmax distributions from the frozen CONQUER begin/end logits; `P_ctx` is the frozen QAL query-to-video temporal attention. Labels were not used to produce these priors, and term statistics were frozen from train_fit only.

## Scale and integrity

- train_fit: 78,045 queries, 779,142 query-video groups, 15,609,000 fixed candidate rows.
- train_calib: 8,739 queries, 87,230 query-video groups, 1,747,800 fixed candidate rows.
- All exported distributions are finite, nonnegative and normalized, with effective temporal lengths from 2 to 100.
- GPU-vectorized span aggregation matches the reference implementation to maximum absolute error `5.960464477539063e-08`.

## Why promotion failed

Among 659 deduplicated configurations, the zero-residual control was the only selection that kept all six primary VCMR metrics nonnegative relative to C4_final. Its eight deltas and all localization movements are exactly zero, so it cannot verify retrieval → localization feedback.

The highest selection-score nonzero residual, `c5p_00138`, had selection delta `+0.00143037` and positive R@100 movement, but regressed:

- `0.5 R@1`: `-0.05721478`
- `0.5 R@10`: `-0.06865774`

It therefore fails the frozen promotion gate and is not a usable configuration. It must not be taken to official val.

## Closure

No C5-lite-prior freeze review was generated and no official-val artifact or evaluator run was produced. C4_final was not changed; CONQUER, begin/end logits, candidate generation, NMS and evaluator remain unchanged. C5-main and C6 were not run.

Conclusion: fixed-candidate external prior calibration did not verify retrieval evidence → localization calibration under the frozen protocol.
