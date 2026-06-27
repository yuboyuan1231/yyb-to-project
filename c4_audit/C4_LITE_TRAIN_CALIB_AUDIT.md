# C4-lite train_calib audit

Date: 2026-06-24  
Status: `C4_LITE_TRAIN_CALIB_PASS`  
Scope: `train_calib_only`  
Official val used: `false`  
Post-val adjustment: `false`

## Scaffold and compile gate

- Source package: `CONQUER_RLEM_C4_FULL_SCAFFOLD_20260623.zip`
- Package SHA256: `d723e49c1abc98b104f1583196662edffb3e24499f892fdadd017cf6998ec2cc`
- The old lite scaffold and `c4_lite_scaffold.patch` were not used.
- All 16 requested Python entrypoints passed `py_compile` under the
  `conquer-rlem` Python 3.9.19 environment.
- Package defects fixed before/during the authorized run were limited to C4
  files: exact frozen-C3.1 baseline construction, frozen checkpoint/config
  validation, atomic cache output, evaluator desc-id typing, bundled metric
  flattening, float32 evaluator-label parity, movement diagnostics, and removal
  of parameter combinations that produced identical scores because a family did
  not consume those parameters.

## Scored candidate audit

Artifact: `c4_audit/C4_TRAIN_CALIB_SCORE_AUDIT.json`

```text
status                     = PASS
queries                    = 8,739
rows                       = 1,747,800
rows/query                 = 200 for all queries
nonfinite_rows             = 0
missing/extra query IDs    = 0 / 0
feature_count              = 31
r2_features_disabled       = true
official_val_used          = false
C3.1 checkpoint SHA256     = eb3559ff35853a7702b34a06cea1da66de594315b1bc40c89aabcd713ea55c42
```

## Cache manifest

Artifact: `c4_audit/C4_CACHE_MANIFEST.json`

```text
status                     = PASS
queries                    = 8,739
rows                       = 1,747,800
bad_query_counts           = 0
duplicate_candidates       = 0
nonfinite_values           = 0
video_groups               = 87,230
official_val_used          = false
```

## Grid search

- Grid configurations evaluated: `594` distinct effective configurations.
- Feasible configurations: `354`.
- Workers: `8`.
- Frozen controls: `effective_top_n=100`, `NMS=0.7`,
  `max_after_nms=100`.
- Fast evaluator versus bundled evaluator maximum difference after evaluator
  rounding: `0.0` for both frozen C3.1 baseline and selected C4-lite best.
- Results: `results/rlem_c4_lite/search/grid_results.csv/json` and
  `results/rlem_c4_lite/search/best_config.json`.

Selected configuration:

```text
config_id       = c4_00394
family          = B_video_span
alpha_bd        = 1.25
gamma_fp        = 0.5
video_gain      = 1.0
span_gain       = 0.5
base_scale      = 0.75
effective_top_n = 100
NMS             = 0.7
max_after_nms   = 100
```

## Exact frozen formulas

With `eps=1e-8`, frozen temperatures `T_joint/T_bd/T_fp=1.5/1.0/1.0`,
and the train_calib-only constant
`mean_qbd=0.3990069627761841`:

```text
base_log = log(max(S_base, eps))
qj       = sigmoid(q_joint_logit / 1.5)
qb       = sigmoid(q_bd_logit / 1.0)
efp      = sigmoid(e_fp_logit / 1.0)

S_C31 = 0.75 * base_log
      + 1.75 * qj
      + 1.00 * qj * (qb - mean_qbd)
      - 0.50 * efp

G_span(q,v,p)  = qj + 1.25 * qj * (qb - mean_qbd) - 0.50 * efp
G_video(q,v)   = max_{p in fixed candidates for (q,v)} G_span(q,v,p)
mu_video_rows  = 0.13249324262142181
mu_span_rows   = 0.03713987395167351

S_C4_lite(q,v,p) = 0.75 * S_C31(q,v,p)
                   + 1.00 * (G_video(q,v) - mu_video_rows)
                   + 0.50 * (G_span(q,v,p) - mu_span_rows)
```

The two means are computed once from train_calib and use the exact row-expanded
arrays consumed by the frozen search. The baseline is `S_C31`; it is not
`S_base` or `base_log`.

Selection is also relative to frozen C3.1:

```text
delta_m = metric_m(S_C4_lite) - metric_m(S_C31)
selection_score_delta = mean(delta for six R@1/R@5/R@10 metrics)
                        + 0.25 * mean(delta for two R@100 metrics)
```

The table below uses the bundled evaluator's two-decimal display values. Exact
selection uses the frozen C3.1 raw recalls stored as `base_metrics_raw` in
`grid_results.json`: `37.7503146813, 61.6889804325, 68.8179425564,
76.0842201625, 25.5063508411, 46.4698478087, 53.9306556814,
64.5153907770`.

## Metrics versus frozen C3.1 control

| Metric | C3.1 | C4-lite | Delta |
|---|---:|---:|---:|
| 0.5 R@1 | 37.75 | 39.33 | +1.5791 |
| 0.5 R@5 | 61.69 | 62.27 | +0.5836 |
| 0.5 R@10 | 68.82 | 69.41 | +0.5950 |
| 0.5 R@100 | 76.08 | 76.19 | +0.1030 |
| 0.7 R@1 | 25.51 | 26.35 | +0.8468 |
| 0.7 R@5 | 46.47 | 48.63 | +2.1627 |
| 0.7 R@10 | 53.93 | 56.07 | +2.1398 |
| 0.7 R@100 | 64.52 | 65.21 | +0.6980 |

Both R@100 metrics improve; neither declines.

Movement diagnostics:

```text
top1_changed_ratio                  = 0.1206087653
Pearson(C3.1, C4-lite)              = 0.9641599398
hard_positive_top100_exits          = 134
hard_positive_top100_entries        = 4,784
hard_positive_top100_exit_ratio     = 0.0021343697
GT-video mean score delta           = -0.8804171681
wrong-video mean score delta        = -1.1759283543
```

The selected global score rescales C3.1, so both absolute score-delta means are
negative; GT videos are suppressed substantially less than wrong videos. A
stricter A-family `C3.1 + video feedback only` configuration with
`base_scale=1.0` also improves all eight metrics, with selection gain `+0.9364`,
top-1 change `5.16%`, Pearson `0.9873`, and both R@100 metrics positive. This
supports a genuine video-feedback signal rather than a span-only explanation.

Exact A-family robustness diagnostic:

```text
config_id       = c4_00083
family          = A_video_feedback
alpha_bd        = 1.25
gamma_fp        = 0.25
video_gain      = 1.0
span_gain       = 0.0
base_scale      = 1.0
selection gain  = +0.9364153030 vs frozen C3.1
metrics         = 38.7344 / 62.2039 / 69.4244 / 76.2330
                  25.9984 / 47.6370 / 55.3038 / 65.0074
top1 changed    = 0.0516077354
Pearson         = 0.9873288244
R@100 deltas    = +0.1487584392 / +0.4920471450
```

## Frozen artifact hashes

```text
package
d723e49c1abc98b104f1583196662edffb3e24499f892fdadd017cf6998ec2cc

C4 scored candidates
a7b26b7fc6ba51d8b460834d218a5f5a2a86ea4c5d773a2a27c88e027bd46104

C4 cache
bcbd0caf12d194b31a4460fd78276832a0b2408434c7e190af6b8c511edff1cf

C3.1 checkpoint
eb3559ff35853a7702b34a06cea1da66de594315b1bc40c89aabcd713ea55c42

best_config.json
9364d5a6bfce032bace814e9953ea4b45ca433f594f848a344735d0fcda5f5d4

best_train_calib_submission.json
0912a42c543b12422f883763a9665a4d016c69aecb8551eb450ab1bb67814398

grid_results.csv
bb22b394cce6c4e5543c6545d26f8abce1ec54921f8a79a59b2dc23f2f16d985

grid_results.json
3a91667b9bd25727bd5c49245b735c2f622740f861b0a3c2762b4190d75167c8
```

## Independent review checklist

```text
baseline is frozen S_C31, not S_base/base_log              = PASS
selection delta is C4-lite vs frozen C3.1                  = PASS
movement diagnostics are relative to frozen C3.1           = PASS
fast/bundled evaluator parity for baseline and best        = PASS (0.0/0.0)
official-val evidence opened                               = false
official-val QSP/C4 file generated                         = false
C1/C2/C3/C3.1/C3.5 frozen artifacts changed               = false
model/conquer.py changed                                    = false
residual .partial files                                     = false
official_val_used                                           = false
post_val_adjustment                                         = false
C4-r2-cal/C4-main/R2/VS/C5/C6 used                         = false
```

## Decision and stop boundary

Freeze decision: `C4_LITE_TRAIN_CALIB_FROZEN`.

Independent review decision: `PASS`.

Recommendation retained for a future, separately authorized stage:
`RECOMMEND_C4_LITE_OFFICIAL_VAL_ONE_SHOT`.

No official-val access is authorized by this report. Freeze and independent
review of the exact one-shot config are required before any official-val run.

The following were not run: C4-lite official val, C4-r2-cal dataset/training/
search, C4-main, VS/R2-head training, QSP/REG/AMD continuation, C5, or C6.
No CONQUER model source, candidate generation, evaluator, or C1/C2/C3/C3.1/C3.5
frozen artifact was modified.
