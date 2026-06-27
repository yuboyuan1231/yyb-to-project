# C6-B1-lite R1-safe selector freeze review

## Freeze decision

- Freeze review status: `PASS`
- Frozen system: `C6-B1-lite R1-safe selector`
- Frozen config: `c6b1r1_0057`
- Official val used: `false`
- Official val authorized: `false`
- Post-val adjustment: `false`
- C6-C used: `false`
- C4_final retained: `true`

This freeze review is train_fit/train_calib-only. It does not authorize official-val execution.

## Frozen config

```json
{
  "config_id": "c6b1r1_0057",
  "apply_slots": 3,
  "threshold0": 0.4,
  "threshold_rest": 1.8,
  "max_replacements": 2,
  "effective_top_n": 100,
  "NMS": 0.7,
  "max_after_nms": 100
}
```

## Frozen interface

For each query:

1. keep C4_final video slots fixed;
2. consider only the first `apply_slots=3` slots for replacement;
3. for each eligible slot, compare original span against boundary-generated alternatives;
4. use the trained V1 independent candidate selector score;
5. replace only if:
   - best alternative is not original;
   - score margin for slot 0 exceeds `threshold0=0.4`;
   - score margin for slots 1-2 exceeds `threshold_rest=1.8`;
   - total replacements in the query do not exceed `max_replacements=2`;
6. run the same temporal NMS threshold `0.7` and keep at most `100`.

This is a sparse replacement interface, not a broad candidate-generation rewrite.

## Baseline

Primary baseline:

```text
pseudo_C4_final_fixed_video_slots
```

This baseline uses the same fixed-video-slot pseudo evaluator as the frozen candidate. It is not official val.

## Metrics

| metric | baseline | frozen C6-B1-lite | delta |
|---|---:|---:|---:|
| 0.5-r1 | 39.375215 | 39.581188 | +0.205973 |
| 0.5-r5 | 62.284014 | 62.432773 | +0.148758 |
| 0.5-r10 | 69.481634 | 69.607507 | +0.125873 |
| 0.5-r100 | 76.221536 | 76.313079 | +0.091544 |
| 0.7-r1 | 26.376016 | 26.662089 | +0.286074 |
| 0.7-r5 | 48.644010 | 48.849983 | +0.205973 |
| 0.7-r10 | 56.242133 | 56.436663 | +0.194530 |
| 0.7-r100 | 65.304955 | 65.465156 | +0.160201 |

All eight train_calib pseudo deltas are positive.

## Movement diagnostics

Movement reference:

```text
post-NMS candidate output vs post-NMS pseudo_C4_final baseline
```

```json
{
  "replacement_rate_top_slots": 0.01643971468894229,
  "replacements": 431,
  "top1_changed_ratio": 0.03936377159858107,
  "hard_positive_top100_query_exits": 0,
  "hard_positive_top100_query_entries": 8,
  "video_slot_drift": 0.0,
  "video_multiset_drift": 0.0
}
```

The corrected movement diagnostics show no hard-positive top100 query exits.

## Grid diagnostics

- Grid size: `78`
- Feasible count: `31`
- R1-positive count: `78`
- R5-safe count: `71`
- Selected config is the best feasible config after R@5 and movement gates.

Neighbor support:

- exact same grid point: `1`
- local radius 0.35: `2`
- local radius 0.8: `6`
- local radius 1.5: `12`

The selected config is not a broad plateau, but it is also not unsupported. Because official val has not been used, the exact thresholds must now be frozen.

## Comparison against alternatives

| system | result | key observation |
|---|---|---|
| C6-B0 boundary/hybrid candidate generation | negative | R@1 improved but R@5 collapsed; replacement rate about 0.884 |
| C6-B1-lite V1 independent selector | positive | best balance of R@1 gain and R@5 safety |
| C6-B1-lite V2 base listwise | positive but weaker | safer but too conservative |
| C6-B1-lite V2 pairwise | negative | degenerated to no replacement |
| C6-B1-lite V2 distill | positive but weaker | very sparse replacement, smaller gains |

V1 `c6b1r1_0057` remains the best freeze candidate.

## Artifacts under freeze

- selector model: `results/rlem_c6_b1_lite_r1_safe/model_best.pt`
- best config: `results/rlem_c6_b1_lite_r1_safe/best_config.json`
- grid results: `results/rlem_c6_b1_lite_r1_safe/grid_results.json`
- train_fit candidate pool: `results/rlem_c6_b1_lite_r1_safe/train_fit_candidate_pool_top_slots.npz`
- train_calib candidate pool: `results/rlem_c6_b1_lite_r1_safe/train_calib_candidate_pool_top_slots.npz`
- train_calib scores: `results/rlem_c6_b1_lite_r1_safe/train_calib_candidate_scores.npz`
- implementation: `rlem_c6_b1_lite/run_c6_b1_r1_safe_selector.py`

## Review notes

One diagnostic issue was found during independent review and fixed before this freeze:

- prior hard-positive movement diagnostics used a pre-NMS original reference;
- fixed reference is now post-NMS candidate output vs post-NMS pseudo C4_final;
- metrics and selected config did not change after the fix;
- corrected movement shows `hard_positive_top100_query_exits=0`.

## Final recommendation

Freeze `c6b1r1_0057`.

Do not continue C6-B1-lite grid/capacity/search before official-val decision. Do not run official val unless explicitly authorized as exactly-one one-shot.
