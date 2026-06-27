# C4-lite official-val one-shot audit

Date: 2026-06-24  
Status: `C4_LITE_OFFICIAL_VAL_ONE_SHOT_PASS`  
Official val one-shot: `true`  
Official val result run count: `1`  
Post-val adjustment: `false`

## Frozen inputs

```text
package SHA256
d723e49c1abc98b104f1583196662edffb3e24499f892fdadd017cf6998ec2cc

C3.1 checkpoint SHA256
eb3559ff35853a7702b34a06cea1da66de594315b1bc40c89aabcd713ea55c42

train_calib frozen best_config.json SHA256
9364d5a6bfce032bace814e9953ea4b45ca433f594f848a344735d0fcda5f5d4

C4-lite freeze manifest SHA256
f59aaf23c37563a83a3f37fa518a6f5a77c8c88d4e1d62440c693f5360e35a9a
```

Exactly one learned configuration was used: `c4_00394 / B_video_span` with
`alpha_bd=1.25`, `gamma_fp=0.5`, `video_gain=1.0`, `span_gain=0.5`,
`base_scale=0.75`, effective top-100, NMS `0.7`, and max-after-NMS `100`.
The frozen train_calib constants were used verbatim:

```text
mean_qbd     = 0.3990069627761841
mu_video_rows= 0.13249324262142181
mu_span      = 0.03713987395167351
```

No constant was estimated from official val. The exact scoring formula is the
formula frozen in `C4_LITE_FREEZE_MANIFEST.json`; the one-shot implementation
was checked against the naive frozen formula with max absolute difference
`4.76837158203125e-07` before official-val access.

## Official-val artifacts

```text
scored candidates
5ad6b5dbb0c1c576bd19fb70626ce03cd288053afb97594fee10751bfd5f6200

C4 cache
b24f32831331f24b905eb547d66b9f4c67b3c48d17d9e9579be4e81ccd0d5796

C4-lite submission
ccd71e6c104667e58c18f29bacd0b8ee7eebc05d808a2f89a8c2151ce4f764b7

C4-lite metrics
435ee34e1a433e2e5bb086dccfa38cb03c28260d0abb829deee3b131a27904cb
```

Scoring and cache audits both pass: `10,895` queries, `2,179,000` rows,
exactly `200` rows/query, no nonfinite row, no duplicate candidate, and
`107,513` query-video groups.

## Official-val metrics

The C0 and reconstructed C1 baselines are identical under the frozen candidate
and NMS protocol.

| Metric | C0/C1 | Frozen C3.1 | C4-lite | C4-C3.1 | C4-C0 |
|---|---:|---:|---:|---:|---:|
| 0.5 R@1 | 13.68 | 14.27 | 14.81 | +0.54 | +1.13 |
| 0.5 R@5 | 27.21 | 28.12 | 28.56 | +0.44 | +1.35 |
| 0.5 R@10 | 33.75 | 34.42 | 34.73 | +0.31 | +0.98 |
| 0.5 R@100 | 46.23 | 46.31 | 46.44 | +0.13 | +0.21 |
| 0.7 R@1 | 7.76 | 8.09 | 8.35 | +0.26 | +0.59 |
| 0.7 R@5 | 17.22 | 18.03 | 18.63 | +0.60 | +1.41 |
| 0.7 R@10 | 22.49 | 23.58 | 24.61 | +1.03 | +2.12 |
| 0.7 R@100 | 35.17 | 35.28 | 35.69 | +0.41 | +0.52 |

All eight deltas versus frozen C3.1 are positive. Both R@100 metrics improve.
C4-lite versus C0 is shown only as an auxiliary reference and was not used for
configuration selection or post-val adjustment.

## Movement relative to frozen C3.1

```text
top1_changed_ratio              = 0.08581918311151905
hard_positive_top100_exits      = 161
hard_positive_top100_entries    = 1475
Pearson(C3.1, C4-lite)          = 0.9908818898039154
mean within-query Spearman      = 0.9905127902635381
```

The fast fixed-candidate/NMS evaluator exactly matches the bundled evaluator
after its standard two-decimal rounding for both frozen C3.1 and C4-lite:
maximum absolute difference `0.0 / 0.0`.

## Evaluator runtime note

The locked evaluator command first terminated before producing a metrics file
because the unchanged evaluator refers to the removed NumPy 1.26 alias
`np.bool`. The same frozen submission was then evaluated successfully after
process-local compatibility injection `np.bool=np.bool_`; evaluator source,
submission, formula, config, and semantics were unchanged.

```text
evaluator_attempt_count              = 2
failed_infrastructure_attempt_count  = 1
successful_evaluator_result_count    = 1
official_val_rerun_count             = 1
```

The failed attempt and compatibility retry are frozen in
`EVALUATOR_RUNTIME_AMENDMENT.json` and their separate logs.

## Protocol closure

```text
official_val_one_shot       = true
learned_config_count        = 1
official_val_rerun_count    = 1
post_val_adjustment         = false
score_grid_on_val           = false
temperature_search_on_val  = false
C4-r2-cal used              = false
C4-main used                = false
R2 used                     = false
VS head used                = false
C5 used                     = false
C6 used                     = false
```

No A-family or alternative B-family official-val run was performed. No C1/C2/
C3/C3.1/C3.5 frozen artifact, CONQUER candidate generation, `model/conquer.py`,
effective top-N, NMS, max-after-NMS, or evaluator source was modified.

Stop decision: `STOP_AFTER_C4_LITE_OFFICIAL_VAL_ONE_SHOT_PASS`.
