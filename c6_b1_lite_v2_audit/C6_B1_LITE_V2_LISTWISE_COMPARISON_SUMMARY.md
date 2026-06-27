# C6-B1-lite-v2 listwise selector comparison summary

## Verdict

- V2 status: `C6_B1_LITE_V2_LISTWISE_R1_SAFE_POSITIVE`
- Official val used: `false`
- Post-val adjustment: `false`
- C6-C used: `false`

V2 confirms that a listwise per-slot selector can produce safe positive deltas, but it does not outperform the current V1 independent selector on train_calib. V1 remains the stronger freeze candidate.

## V2 best config

```json
{
  "config_id": "c6b1r1_0009",
  "apply_slots": 3,
  "threshold0": -0.25,
  "threshold_rest": 0.2,
  "max_replacements": 2
}
```

## V1 vs V2 deltas vs pseudo C4_final

| metric | V1 c6b1r1_0057 | V2 c6b1r1_0009 |
|---|---:|---:|
| 0.5-r1 | +0.205973 | +0.068658 |
| 0.5-r5 | +0.148758 | +0.034329 |
| 0.5-r10 | +0.125873 | +0.022886 |
| 0.5-r100 | +0.091544 | +0.022886 |
| 0.7-r1 | +0.286074 | +0.045772 |
| 0.7-r5 | +0.205973 | +0.045772 |
| 0.7-r10 | +0.194530 | +0.045772 |
| 0.7-r100 | +0.160201 | +0.045772 |

## Movement comparison

| diagnostic | V1 | V2 |
|---|---:|---:|
| replacement_rate_top_slots | 0.016440 | 0.007323 |
| replacements | 431 | 192 |
| top1_changed_ratio | 0.039364 | 0.005264 |
| hard_positive_top100_query_exits | 0 | 0 |
| hard_positive_top100_query_entries | 8 | 2 |
| video_slot_drift | 0.0 | 0.0 |

## Interpretation

V2 is much more conservative. This is good for safety but weak for R@1. The model learned a reliable “do not replace unless very obvious” behavior. That is useful evidence, but it does not beat V1.

Likely causes:

1. listwise CE is dominated by no-replacement / original-preserving groups;
2. threshold selection picks extremely sparse replacement;
3. pairwise replacement margin is too weak to unlock more R@1-positive alternatives;
4. V2 scoring is calibrated differently, so V1-style thresholds are not ideal.

## Recommendation

Do not promote V2 over V1.

Keep `c6b1r1_0057` as the current best C6-B1-lite train_calib freeze candidate. If continuing development, the next model should be a hybrid:

- use V2 listwise architecture;
- add stronger pairwise original-vs-boundary replacement loss;
- distill or ensemble with V1 scores;
- preserve the same R@5 / movement gates.

Do not run official val from V2 without a separate freeze review.
