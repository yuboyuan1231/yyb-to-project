# C5-main-inference-A negative audit

## Decision

- Status: `NO_PROMOTION`
- Classification: `C5-main-inference-A train_calib negative`
- Primary baseline: `C4_final = C4-r2-cal-v2.1 v21_00444`
- Selected strict-safe configuration: `c5ma_0000`, zero injection
- Official val used/authorized: `false / false`
- Recommendation: do not freeze a nonzero A scorer and do not run official val.

## Protocol and capacity

The final implementation uses the full frozen begin/end temporal distributions, separate start/end roles, weak QAL context gating, a retrieval-confidence gate, and a five-dimensional inference grid (`alpha`, `beta`, `tau`, `lambda`, `gamma`). It evaluates 432 nonzero configurations plus one control. No trainable MLP was added because training/fine-tuning was explicitly prohibited; an untrained network would not provide meaningful capacity.

Because the frozen export identifies begin/end probabilities rather than arbitrary raw-logit constants, the final injection is gauge-invariant:

```text
P_s_prior = Normalize[P_b^(1+alpha) * (P_ctx+eps)^beta]
P_e_prior = Normalize[P_e^(1+alpha) * (P_ctx+eps)^beta]
gate      = clip(2*sigmoid(R_video_z/tau), 0.25, 1.75)

new_P_b = Normalize[P_b * P_s_prior^(lambda*gate)]
new_P_e = Normalize[P_e * P_e_prior^(lambda*gate)]

Delta_bd = log(new_P_b[i]/P_b[i]) + log(new_P_e[j]/P_e[j])
S_A      = S_C4_final + gamma*z_trainfit(Delta_bd)
```

All `Delta_bd` and retrieval statistics were frozen on train_fit only. The 15,609,000 train_fit candidate rows produced 108 finite Delta-stat groups. Candidate times were exact 1.5-second grid multiples; no endpoint index required clipping.

An initial raw-logit implementation check was discarded because normalized-prior constants leaked video-level offsets into cross-video scores. It was not used for the final selection and did not access official val. The correction changes probability semantics, not model capacity or the search range.

## Train_calib result

Among all 433 effective configurations:

- six-primary-nonnegative count: `1`
- both-R@100-nonnegative count: `1`
- core-promotion count: `0`
- promotion-candidate count: `0`

The sole strict-safe configuration is zero injection, whose eight deltas and localization movement are exactly zero.

The best rejected nonzero configuration is `c5ma_0111`:

```text
alpha=0.0, beta=0.10, tau=1.0, lambda=0.10, gamma=0.075
```

| Metric | Delta vs C4_final |
|---|---:|
| 0.5 R@1 | -0.091544 |
| 0.5 R@5 | +0.274631 |
| 0.5 R@10 | +0.286074 |
| 0.5 R@100 | +0.011443 |
| 0.7 R@1 | +0.034329 |
| 0.7 R@5 | -0.080101 |
| 0.7 R@10 | -0.251745 |
| 0.7 R@100 | -0.240302 |

It fails six-primary and R@100 constraints.

## Localization signal versus ranking safety

`c5ma_0111` confirms that the role-specific model has enough capacity to move localization in the intended direction:

- Oracle-video R@1@0.5 delta: `+0.01420`
- Oracle-video R@1@0.7 delta: `+0.31237`
- Selected-span mIoU delta: `+0.000665`
- Mean best-IoU rank delta: `-0.02868`
- Best-IoU rank improved/worsened ratio: `6.39% / 3.98%`

However, its cross-video movement is unsafe for promotion:

- Hard-positive top-100 exits/entries: `1030 / 81`
- Hard-positive top-100 exit ratio: `1.641%` (gate limit `0.5%`)
- Top-1 changed ratio: `2.483%`
- Pearson / mean within-query Spearman: `0.99653 / 0.99450`

The result is therefore a real localization-side signal with an unacceptable final-ranking trade-off—not a null result caused by an overly simple inference module.

## Closure

No freeze review and no official-val artifact were generated. C4_final, the CONQUER checkpoint/source, persistent begin/end logits, candidate generation, NMS, effective_top_n and evaluator remain unchanged. C5-main-B, C6, C4-main and VS/R2 heads were not run.

Execution stops at the train_calib negative result pending human review.
