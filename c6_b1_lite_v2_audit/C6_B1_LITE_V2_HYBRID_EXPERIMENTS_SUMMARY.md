# C6-B1-lite-v2 hybrid experiments summary

## Scope

Two follow-up variants were tested after the base listwise selector:

1. `v2_pairwise`: listwise encoder with stronger original-vs-boundary pairwise replacement loss.
2. `v2_distill`: listwise encoder distilled from V1 candidate scores.

Both experiments are train_fit/train_calib only.

- Official val used: `false`
- Post-val adjustment: `false`
- C6-C used: `false`
- C4_final retained: `true`

## Result table

| system | status | best config | selection | feasible | 0.5-r1 | 0.5-r5 | 0.7-r1 | 0.7-r5 | repl. rate | top1 changed |
|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|
| V1 independent | POSITIVE | c6b1r1_0057 | 0.324122 | true | +0.205973 | +0.148758 | +0.286074 | +0.205973 | 0.016440 | 0.039364 |
| V2 base listwise | POSITIVE | c6b1r1_0009 | 0.074665 | true | +0.068658 | +0.034329 | +0.045772 | +0.045772 | 0.007323 | 0.005264 |
| V2 pairwise | NEGATIVE | c6b1r1_0007 | 0.000000 | false | +0.000000 | +0.000000 | +0.000000 | +0.000000 | 0.000000 | 0.000000 |
| V2 distill | POSITIVE | c6b1r1_0008 | 0.097551 | true | +0.068658 | +0.022886 | +0.080101 | +0.080101 | 0.000839 | 0.002517 |

## Detailed interpretation

### V2 pairwise

The stronger pairwise loss did not help. The safest selected policy became the no-replacement baseline:

```json
{
  "selection_score": 0.0,
  "feasible": false,
  "replacements": 0
}
```

Likely reason: the pairwise loss changed score calibration so that replacement margins became unusable under the existing conservative threshold grid. It may also have over-penalized ambiguous positive alternatives.

### V2 distill

Distillation from V1 improved over base V2 on 0.7 metrics, but remained much more conservative than V1:

- replacements: `22`
- replacement_rate_top_slots: `0.000839`
- hard-positive exits: `0`
- hard-positive entries: `1`

This confirms that V1 contains useful candidate-level signal, but naive distillation into the listwise selector mostly transfers safety, not the full R@1 gain.

## Final recommendation

Do not promote either v2 follow-up over V1.

Current best train_calib freeze candidate remains:

```json
{
  "system": "C6-B1-lite R1-safe selector",
  "config_id": "c6b1r1_0057",
  "apply_slots": 3,
  "threshold0": 0.4,
  "threshold_rest": 1.8,
  "max_replacements": 2
}
```

V1 remains best because it is the only candidate with materially positive R@1 while still keeping R@5 positive and hard-positive exits at zero.

## If continuing model design

The next improvement should not be another direct pairwise or naive distill run. Better options:

1. Keep V1 as teacher and use a calibrated regression target on V1 replacement margin rather than full softmax distillation.
2. Add a train_fit inner gate split so threshold selection is not entirely train_calib-driven.
3. Use V2 architecture only as a re-ranker on V1-proposed replacement candidates, not as a full replacement scorer.
4. Add a replacement prior/head that explicitly predicts “replace vs keep original” before selecting the alternative.

No official-val run is recommended from these v2 variants.
