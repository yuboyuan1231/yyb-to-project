# C6-B2 protocol/design review: R1-first multi-arm plan

Status: `DESIGN_REVIEW_ONLY`

This document does not authorize training by itself. It is intended as the input protocol for the next GOAL-mode window. It does not run C6-B2, C6-C, official val, second official val, threshold search on val, post-val adjustment, C4_final modification, C6-B1-lite modification, NMS modification, or evaluator modification.

## 1. Current state and problem framing

Current promoted system:

- `C6-B1-lite c6b1r1_0057`
- previous promoted system: `C4_final v21_00444`
- official-val status: `C6_B1_LITE_OFFICIAL_VAL_STRONG_POSITIVE`
- official 0.7 R@1: `8.39`
- delta vs C4_final at 0.7 R@1: `+0.04`

C6-B1-lite is valuable because it is the first clean official-val positive after the negative C5/C6-A fixed-candidate lines. But it is intentionally conservative:

- replacement_rate_top_slots: `1.48%`
- top1_changed_ratio: `3.27%`
- hard-positive top100 exits / entries: `3 / 15`
- harmful replacement rate: `2.47%` at IoU 0.5 and `2.68%` at IoU 0.7

The same-video candidate pool has much higher diagnostic ceiling:

| variant | 0.7 R@1 | 0.7 R@5 | replacements |
|---|---:|---:|---:|
| current actual | 8.37 | 18.84 | 485 |
| apply3/max2 oracle | 17.50 | 26.02 | 5981 |
| boundary-only oracle | 12.36 | 25.35 | 6784 |
| slots5/max2 greedy oracle | 14.27 | 27.30 | 7211 |
| slots5/max5 greedy oracle | 18.90 | 27.50 | 11933 |

Interpretation: C6-B2 should not be a small tweak to thresholds. It should learn a stronger replacement selector that safely consumes more of the same-video candidate pool. Since R@100 is no longer a priority, the protocol should prioritize 0.7 R@1, then 0.5 R@1, with R@5/R@10 as collapse guards rather than optimization objectives.

## 2. External method signals

Recent VCMR methods point to two lessons:

1. Better high-IoU R1 needs moment-aware localization and candidate generation, not only row-level reranking.
2. To reach the recent 10--12+ R@1/IoU=0.7 range, video retrieval and moment localization likely need coupling eventually, but C6-B2 is still worth doing first because the same-video oracle gap is large.

References used for design orientation:

- CONQUER: VCMR requires ranking localized moments from a large corpus; the paper notes difficulty around simultaneously training ML and VS heads, supporting staged integration rather than an immediate unconstrained end-to-end jump. Source: https://www.cs.cityu.edu.hk/~wkchan/papers/mm2021-hou%2Bngo%2Bchan.pdf
- PREM: frames VCMR as partial relevance and separates video retrieval and moment localization enhancements. This supports adding partial-relevance features in C6-B2 without changing video slots. Source: https://arxiv.org/html/2402.13576v1
- GenSpan: reports TVR VCMR R@1/IoU=0.7 `12.12` and emphasizes generated temporal/motion span priors. This supports expanding candidate span priors and selector quality as an intermediate step before C6-C. Source: https://arxiv.org/html/2603.22121v2
- MA-VR-style moment-aware video retrieval reports high TVR VCMR R1@0.7 and motivates eventual retrieval-localization coupling. Source: https://ieeexplore.ieee.org/document/10902083/

## 3. Scope and hard boundaries for next GOAL-mode run

Allowed in C6-B2 GOAL mode:

- train_fit / train_calib only;
- train multiple C6-B2 selector arms;
- build cached pairwise/listwise candidate datasets;
- reuse C6-B1 same-video hybrid pool and BoundaryOracleAdapter priors;
- optionally expand same-video candidates within train_fit/train_calib only;
- search train_calib policies, budgets, and thresholds;
- output freeze review if a candidate passes.

Forbidden:

- official val;
- official-val evaluator;
- official-val predictions/scores/cache;
- second official val;
- post-val adjustment;
- modifying `C6-B1-lite c6b1r1_0057` frozen artifacts;
- modifying C4_final;
- modifying NMS/evaluator;
- full backbone fine-tuning;
- C6-C;
- using official-val results for model/threshold/epoch/policy selection;
- writing train_calib positive as official positive.

## 4. Metrics and selection philosophy

Primary metric:

- `0.7 R@1`

Secondary metric:

- `0.5 R@1`

Guardrails:

- `0.7 R@5`, `0.5 R@5`: mild decline acceptable, collapse not acceptable;
- `0.7 R@10`, `0.5 R@10`: diagnostic only;
- R@100: record for safety, but do not optimize; tolerate small decline if R1 improves cleanly;
- hard-positive exits, harmful replacement rate, top1_changed_ratio, invalid spans, duplicate spans, video_slot_drift, video_multiset_drift.

Primary baseline:

- `C6-B1-lite c6b1r1_0057`

Secondary baseline:

- `C4_final v21_00444`

R1-first selection score for train_calib:

```text
score =
  4.0 * delta_0.7_r1
+ 2.0 * delta_0.5_r1
+ 0.6 * min(delta_0.7_r5, 0.20)
+ 0.4 * min(delta_0.5_r5, 0.20)
+ 0.2 * min(delta_0.7_r10, 0.20)
- 2.5 * max(0, -delta_0.7_r5 - 0.10)
- 1.5 * max(0, -delta_0.5_r5 - 0.10)
- 1.0 * max(0, hard_positive_exit_ratio - 0.0075)
- 1.0 * harmful_replacement_penalty
- 0.5 * max(0, replacement_rate - 0.07)
```

R@100 is reported but not included unless it catastrophically collapses.

## 5. Data products to build first

Before any arm training, build durable cached datasets:

### 5.1 Pairwise replacement dataset

One row per `(query, video slot, original span, alternative span)`.

Fields:

- query_id, video_id, slot_id;
- original span start/end/length/score;
- alternative span start/end/length/score;
- alt - original endpoint delta;
- span IoU with original;
- boundary prior score;
- C4_final score and rank/slot features;
- C6-B1 selector score and margin;
- slot embedding index;
- query-level score distribution stats;
- candidate source: original/boundary/local-refine;
- labels from train_fit/train_calib only:
  - `gain_05`, `gain_07`;
  - `harmful_05`, `harmful_07`;
  - `replace_better_05`, `replace_better_07`;
  - `keep_better`;
  - continuous IoU delta.

### 5.2 Listwise slot dataset

One sample per `(query, slot)` containing original + alternatives as a set.

Shape target:

- `alt_count=8` first;
- optional `alt_count=16` after pool audit passes.

### 5.3 Policy/query dataset

One sample per query, containing top slot action candidates and predicted utilities, used for budget/abstention calibration.

## 6. Candidate pool plan

Use a staged pool expansion to prevent uncontrolled noise.

### Pool P0: current C6-B1 pool

- slots_for_selector_pool = 5;
- alt_count = 8;
- same-video only;
- original + BoundaryOracleAdapter alternatives.

### Pool P1: local refinement pool

Add only train_fit/train_calib local variants:

- start/end ±1, ±2, ±4 clips;
- expand/shrink around original and boundary proposals;
- neighbor-stabilized candidates.

P1 gate before training arms:

- invalid_span_count = 0;
- duplicate span count controlled after de-dup;
- oracle 0.7 R@1 improves over P0 by at least `+0.20` on train_calib;
- runtime/memory acceptable.

If P1 fails, do not use it in main arms.

### Pool P2: alt_count=16

Only if P1 or additional boundary alternatives improve oracle. Otherwise do not spend GPU on larger sets.

## 7. Arms to run

The attachment suggests seven arms. I recommend condensing them into four main arms plus two diagnostic arms. This keeps the experiment broad enough to learn, but not so broad that we burn two GPUs on variants that answer the same question.

### Arm A: Pairwise Utility + Abstention + Top-K Guard

This is the main baseline for B2.

Model: `C6B2PairwiseUtilitySelector`

Inputs:

- original features;
- alternative features;
- diff features;
- slot/rank embeddings;
- C4 score gap and query score statistics;
- endpoint/boundary prior features;
- overlap and length-change features;
- C6-B1 selector score/margin.

Outputs:

- `gain_07_logit`;
- `gain_05_logit`;
- `risk_07_logit`;
- `risk_05_logit`;
- `replace_logit`;
- `abstain_logit`;
- scalar utility.

Loss:

```text
L =
  1.5 * BCE(gain_07)
+ 0.8 * BCE(gain_05)
+ 0.7 * BCE(replace_better)
+ 0.7 * BCE(risk_07/harmful_07)
+ 0.4 * BCE(risk_05/harmful_05)
+ 0.3 * abstention calibration
+ 0.2 * top5 positive preservation
+ 0.1 * budget regularization
```

Default capacity:

- hidden_dim = 512;
- layers = 4;
- pair_interaction_layers = 2;
- GELU + LayerNorm + dropout 0.10;
- AMP enabled.

### Arm B: Set/Listwise Slot Selector

This is the most important arm after A.

Motivation: alternatives in the same slot compete. Independent pairwise scoring can over-select duplicate/noisy boundary variants.

Model:

- OriginalEncoder;
- AlternativeSetEncoder;
- CrossAttention(original, alternatives);
- ListwiseActionHead: `keep`, `replace_alt_i`, `abstain`;
- optional proposal confidence head.

Loss:

```text
listwise CE(best action for 0.7-first utility)
+ risk penalty
+ top5 guard
+ abstain calibration
```

Capacity:

- hidden_dim = 512 or 768;
- 4--6 layers;
- 8 attention heads for large;
- dropout 0.10;
- AMP.

This arm should get one GPU if two GPUs are available.

### Arm C: Safe Policy / Budget Selector

This arm wraps Arm A or B outputs with query-level budget control.

Outputs:

- query_budget_score;
- slot_budget_score;
- risk_budget;
- replacement budget class: `{0, 1, 2, 3, 5}`.

Policy search on train_calib:

- target_replacement_rate in `{0.03, 0.05}`;
- max_replacements in `{2, 3, 5}`;
- apply_slots in `{3, 5}`;
- slot0 strictness: `{high, medium}`;
- slot1/2 strictness: `{medium, low}`.

This arm exists to answer whether B1's low replacement rate is the bottleneck.

### Arm D: Temporal Coverage Guard / MMR

This is a policy layer, not necessarily a separate heavy model.

Purpose: protect R@5/R@10 while improving R@1.

Mechanisms:

- temporal overlap penalty with already kept candidates;
- at most one replacement per temporal neighborhood;
- keep C4 top5 positives when risk is high;
- MMR score: `utility - lambda * overlap_with_selected`.

Search:

- lambda in `{0.05, 0.10, 0.20}`;
- only train_calib;
- no official val.

### Diagnostic Arm E: Local Refinement Candidate Pool

This is not a model arm. Run pool audit first.

If P1 improves oracle, feed P1 into Arms A/B. If not, stop E.

### Diagnostic Arm F: Moment-aware-lite Feature Injection

This is feature injection into A/B, not a standalone final system.

Features:

- moment-level evidence aggregate;
- span-to-video evidence ratio;
- local relevance vs global video relevance;
- partial relevance score;
- subtitle/visual modality evidence if already available in frozen artifacts.

No video ranking changes.

## 8. What I would not run first

I would not initially run all seven attachment arms as independent full trainings.

Reasons:

- B2-context and B2-listwise are overlapping; start with listwise as the stronger formulation.
- B2-coverage is a policy guard and can wrap A/B; no need for a separate full model first.
- B2-safe-policy should initially be a policy head/search on A/B outputs, not a third unrelated model unless A/B pass base gates.
- Candidate expansion should be pool-audited before any expensive model training.

## 9. Training and efficiency plan for next GOAL mode

Use two GPUs as follows:

- GPU 0: Arm A pairwise utility selector.
- GPU 1: Arm B listwise/set selector.

After A/B finish:

- run C/D policy wrappers on cached scores, mostly CPU/GPU-light;
- run P1 pool audit before any P1 model training;
- only train P1 variants if oracle gain passes the P1 gate.

Efficiency requirements:

- cache pairwise/listwise datasets to `results/rlem_c6_b2/caches/`;
- memory-map large feature arrays where practical;
- batch feature construction, avoid row-wise Python hot paths;
- DataLoader with `pin_memory=True`, `persistent_workers=True`;
- AMP training;
- record examples/sec, GPU memory, epoch time;
- save logits/scores for every arm so policy search does not rerun the model.

## 10. Freeze gate for B2

Baseline: `C6-B1-lite c6b1r1_0057`.

Minimum freeze-review gate:

```text
delta_0.7_r1 > 0
delta_0.5_r1 >= 0
delta_0.7_r5 >= -0.10
delta_0.5_r5 >= -0.10
delta_0.7_r10 >= -0.15
delta_0.5_r10 >= -0.15
hard_positive_exit_ratio <= 0.75%
video_slot_drift = 0
video_multiset_drift = 0
invalid_span_count = 0
duplicate_span_count = 0
harmful_replacement_rate <= B1_harmful_rate + 0.03
replacement_rate <= 0.07
official_val_used = false
```

Strong freeze gate:

```text
delta_0.7_r1 >= +0.15 on train_calib
delta_0.5_r1 >= +0.10
0.7-r5 >= 0 or within -0.03 with large R1 gain
0.5-r5 >= 0 or within -0.03 with large R1 gain
not isolated: neighboring budgets/policies also pass
replacement_rate in 3%--5% preferred, <=7% absolute cap
harmful replacement rate low
```

Tradeoff labels:

- `B2_R1_TRADEOFF`: R1 positive but R@5/R@10 collapse beyond gate.
- `B2_LOW_IOU_ONLY`: 0.5 R@1 positive but 0.7 R@1 not positive.
- `B2_NEGATIVE`: no arm passes minimum gate.
- `B2_FREEZE_REVIEW_PASS`: one candidate passes and is robust.

## 11. Final outputs for next GOAL mode

Expected directories:

```text
rlem_c6_b2/
results/rlem_c6_b2/
c6_b2_audit/
```

Expected audit files:

```text
c6_b2_audit/C6_B2_START_STATE.md/json
c6_b2_audit/C6_B2_PROTOCOL.md
c6_b2_audit/C6_B2_DATA_AUDIT.md/json
c6_b2_audit/C6_B2_ARM_COMPARISON.md/json
c6_b2_audit/C6_B2_TRAINING_AUDIT.md/json
c6_b2_audit/C6_B2_FINAL_DECISION.md/json
c6_b2_audit/C6_B2_HASHES.json
```

If freeze pass:

```text
c6_b2_audit/C6_B2_FREEZE_REVIEW.md
c6_b2_audit/C6_B2_FREEZE_MANIFEST.json
c6_b2_audit/C6_B2_FREEZE_HASHES.json
```

If fail/tradeoff:

```text
c6_b2_audit/C6_B2_NEGATIVE_AUDIT.md
c6_b2_audit/C6_B2_NEGATIVE_MANIFEST.json
c6_b2_audit/C6_B2_NEGATIVE_HASHES.json
```

## 12. Decision tree after B2

If B2 passes freeze review:

- stop;
- do not run official val automatically;
- request human review for exactly-one official-val one-shot.

If B2 is negative or R1 tradeoff:

- keep `C6-B1-lite c6b1r1_0057` as promoted system;
- do not continue same-video selector expansion blindly;
- move to `C6-C design review` focused on video-aware mutual integration.

If B2 improves R1 but still far below 10--12 target:

- freeze if safe;
- then plan C6-C as the route to address video-ranking residual errors.

## 13. My recommended first GOAL-mode command intent

Start with:

1. Phase 0 start state;
2. build pairwise/listwise cached datasets from train_fit/train_calib;
3. run P0 oracle sanity against existing B1 numbers;
4. train Arm A on GPU0 and Arm B on GPU1;
5. materialize train_calib scores;
6. run budget/policy wrappers C/D;
7. optionally run P1 pool audit;
8. produce arm comparison and final decision;
9. stop.

Do not run official val.
