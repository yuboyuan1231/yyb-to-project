# C3-strong train_calib-only weight-search audit — 2026-06-22

## Decision

```text
C3_STRONG_PASS
```

The search used train_calib only. It evaluated 800 predeclared coarse-grid
configurations and one deterministic 80-configuration local refinement. The
selected gated-boundary configuration improves all eight metrics over both
S_base and C3-minimal default while passing every frozen stability constraint.

Official val evidence was not read, and no official-val learned submission was
generated.

## Instruction review and additions

The requested search families and grid are reasonable. The original selection
language contained subjective terms such as "明显下降" and "异常过高", so the
following thresholds were frozen before inspecting grid results:

```text
all six R@1/R@5/R@10 deltas > 0
at least 6/8 metric deltas > 0
each R@100 delta >= -0.25
hard-positive top-100 exit ratio <= 0.5%
top-1 changed ratio <= 40%
Pearson(base, C3) >= 0.98
mean within-query Spearman(base, C3) >= 0.90
Base weighted std share >= 60% and is the largest component
boundary weighted std share <= 20%
```

The composite selection score is the mean of the six R@1/R@5/R@10 deltas plus
0.25 times the mean of the two R@100 deltas. If the best Family B score is
within 0.10 points of Family A, Family B is preferred. This rule was not changed
after results were observed.

## Search command

```bash
python3 rlem/grid_search_c3_weights.py \
  --evidence_jsonl results/rlem_c3_minimal/train_calib_evidence.jsonl.gz \
  --desc_ids_filter results/rlem_c2/splits/train_calib_desc_ids.txt \
  --ckpt results/rlem_c2/basic_heads/model_best.pt \
  --dataset_config config/tvr_data_config.json \
  --split train \
  --gt_jsonl results/rlem_c3_minimal/train_calib_gt.jsonl \
  --output_dir results/rlem_c3_strong \
  --effective_top_n 100 \
  --max_after_nms 100 \
  --nms_thd 0.7 \
  --rows_per_query 200 \
  --score_batch_size 8192 \
  --device cuda \
  --workers 8 \
  --auto_refine
```

The evidence model was evaluated once. Frozen candidate scores were then reused
across every grid configuration. Each configuration independently performed
full 200-row ranking, effective top-100 selection, NMS=0.7, metric calculation,
and stability diagnostics.

## Search size and refinement

```text
coarse configurations    = 800
local refinement         = 80
total configurations     = 880
feasible configurations  = 803
runtime                   = 273.64 seconds
```

The coarse top-10 concentrated at `base_scale=0.75`, `a_joint=1.5`, and
`b_bd=0.75`, triggering the single authorized refinement. No second refinement
was performed even though a refined coefficient remained at a local edge.

## Rejected intermediate evaluation

The first fast-grid implementation used C1's exported `y_joint_05/07` threshold
labels to compute headline metrics. Final evaluator verification found small
0.01–0.05-point discrepancies. Investigation found 34 IoU0.5 and 65 IoU0.7
boundary rows, out of 1,747,800, whose export labels differed from the bundled
evaluator's float32 timestamp semantics.

Those results were rejected before grid artifacts were accepted. Candidate
correctness was recomputed with the bundled evaluator's exact float32 IoU
semantics, and the complete 880-configuration search was rerun. Final S_base,
C3-minimal, and selected-best metrics are exact to the original evaluator; the
best submission is also object-exact to the original NMS path.

## Selected configuration

```text
family      = B_gated_boundary
base_scale  = 0.75
a_joint     = 1.625
b_bd        = 0.75
d_fp        = 0.5

S = 0.75 * log(S_base)
  + 1.625 * Q_joint
  + 0.75 * Q_joint * Q_bd
  - 0.5 * E_fp
```

Family B was not selected merely by preference: its selection score is
4.124709 versus 4.026967 for the best Family A configuration. It also reduces
the mean wrong-video boundary bonus from 0.098121 to 0.009731 and reduces
hard-positive exits from 103 to 92.

## Metrics

| Metric | S_base | C3-minimal | C3-strong | vs base | vs minimal |
|---|---:|---:|---:|---:|---:|
| IoU0.5 R@1 | 30.63 | 35.29 | **37.30** | **+6.67** | **+2.01** |
| IoU0.5 R@5 | 57.68 | 59.97 | **61.06** | **+3.38** | **+1.09** |
| IoU0.5 R@10 | 66.67 | 67.87 | **68.35** | **+1.68** | **+0.48** |
| IoU0.5 R@100 | 75.81 | 75.92 | **76.04** | **+0.23** | **+0.12** |
| IoU0.7 R@1 | 20.73 | 23.92 | **25.06** | **+4.33** | **+1.14** |
| IoU0.7 R@5 | 41.41 | 44.06 | **45.97** | **+4.55** | **+1.91** |
| IoU0.7 R@10 | 49.67 | 51.79 | **53.32** | **+3.65** | **+1.53** |
| IoU0.7 R@100 | 63.84 | 64.08 | **64.26** | **+0.42** | **+0.18** |

All eight deltas are positive relative to both reference configurations.

## Stability diagnostics

```text
top-1 changed query ratio              = 22.66%
hard-positive top-100 exits            = 92
hard-positive top-100 entries          = 5,600
hard-positive exit ratio               = 0.1466%
y_fp candidates moving downward        = 62.78%
y_joint_05 candidates moving downward  = 9.86%
wrong-video gated-boundary bonus mean   = 0.009731
Pearson(base, C3)                       = 0.980298
mean within-query Spearman              = 0.985278
```

Weighted component standard-deviation shares:

| Component | Share |
|---|---:|
| Base | 77.54% |
| Q_joint | 10.70% |
| gated Q_joint×Q_bd | 3.13% |
| negative E_fp | 8.62% |

The base score remains dominant. Gating reduces the unconstrained wrong-video
Q_bd contribution by roughly 10× relative to the best additive configuration.
Hard-positive entries outnumber exits by more than 60×.

## Artifacts

- `results/rlem_c3_strong/train_calib_grid_results.json`
- `results/rlem_c3_strong/train_calib_grid_results.csv`
- `results/rlem_c3_strong/best_config.json`
- `results/rlem_c3_strong/best_train_calib_submission.json`
- `results/rlem_c3_strong/best_train_calib_metrics.json`
- `results/rlem_c3_strong/c3_strong_diagnostics.json`
- `results/rlem_c3_strong/c3_strong_diagnostics.md`
- `c3_strong_audit/C3_STRONG_FROZEN_MANIFEST.json`

Every JSON grid record and CSV row contains the requested metric, movement,
component-scale, correlation, and safety fields. Artifact hashes are frozen in
the manifest.

## Integrity and stage boundary

- C1 train and official-val evidence hashes remain unchanged.
- The C2 checkpoint and train_calib split hashes remain unchanged.
- Candidate generation, effective top-100, NMS=0.7, max-after-NMS=100, and the
  bundled evaluator remain unchanged.
- `official_val_used = false`.
- No QSP, REG, AMD, C4, C5, or C6 operation was run.

## Recommendation

The selected configuration is materially better than C3-minimal and passes all
stability constraints. A strictly one-shot official-val evaluation is therefore
recommended as the next scientific gate. Because both checkpoint and weights
were selected on train_calib, that one-shot must use this exact frozen config,
with no post-val adjustment or second attempt.

Recommendation is not authorization:

```text
recommend_official_val_one_shot = true
official_val_one_shot_executed = false
next_stage_authorized = false
```

Stop here.
