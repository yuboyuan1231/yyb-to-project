# C3.1 Capacity + Calibration Strengthening Audit

## Decision

`C31_PASS`

C3.1 used only frozen 31-scalar evidence from `train_fit` and `train_calib`. It did not read official-val evidence, did not run an official-val submission, and did not use QSP, R2, REG, or AMD. No later stage is authorized by this report.

The scalar plateau was broken under the pre-frozen threshold: the stable train-calib selection score increased from `4.124709` for C3-strong to `4.678739` for C3.1 (`+0.554030`, threshold `+0.10`). An official-val one-shot is recommended, but was not executed.

## Search executed

- Capacity: 6 architectures x 3 weight decays = 18 full models.
- Loss weights: 12 full models on the capacity-stage winner.
- FP imbalance handling: BCE, automatic positive weight, focal gamma 1, focal gamma 2 = 4 full models.
- Total: 34 full models; every model has a checkpoint, history, diagnostics, cached train-calib logits, and a three-family fixed-weight probe.
- Full score grid: 3 distinct-logit finalists x 1200 configurations.
- Calibration: top 3 feasible configurations per family and finalist, 32 temperature tuples per seed.
- Local refinement: one 80-configuration round because the selected coarse weights touched upper grid boundaries.
- Total recorded score evaluations: 4646.

The exact-logit deduplication amendment was frozen before the full grid. It skipped redundant full-grid evaluation only when prediction-cache SHA256 values were byte-identical; all duplicate trained models remain in the 34-model audit.

## Capacity probe summary

| Architecture | Best capacity config | Parameters | Probe selection score |
|---|---|---:|---:|
| A `[128]` | `cap_A_wd0` | 4,739 | 3.987394 |
| B `[128,128]` | `cap_B_wd0` | 21,507 | 4.125186 |
| C `[256]` | `cap_C_wd0` | 9,475 | 4.033642 |
| D `[256,128]` | `cap_D_wd1e4` | 42,243 | 4.131384 |
| E `[256,256]` | `cap_E_wd1e4` | 75,779 | 4.243430 |
| F `[256,256,128]` | `cap_F_wd0` | 108,547 | 4.283003 |

The staged capacity winner was E with weight decay `1e-4`: it was within 0.10 of the raw best F score and had fewer parameters. The later global finalist selection still retained F because all 34 model probes were ranked before the bounded full grid.

## Loss and FP findings

- Best loss-weight probe: `lambda_joint_reg=1`, `lambda_joint_bin=0.5`, `lambda_bd=1`, `lambda_fp=1`, BCE; score `4.244860`.
- Best FP variant: ordinary BCE, score `4.244860`.
- Focal gamma 1/2 scored `4.170004` / `4.053667`.
- Automatic positive weighting scored `3.259336`; it was clearly harmful under the stable probe.

Complete per-model and per-score-configuration results are frozen in `results/rlem_c31_capacity/grid_results.json` and `.csv`.

## Frozen best configuration

```text
model       = cap_F_wd0
hidden_dims = [256, 256, 128]
dropout     = 0.1
parameters  = 108547

family      = C_centered_gated_boundary
base_scale  = 0.75
a_joint     = 1.75
b_bd        = 1.0
d_fp        = 0.5
T_joint     = 1.5
T_bd        = 1.0
T_fp        = 1.0
```

Non-unit joint temperature was accepted because its selection gain exceeded the pre-frozen `+0.05` threshold. The single local-refinement round did not beat this configuration.

## Train-calib metrics

| Metric | S_base | C3-minimal | C3-strong | C3.1 | Delta vs C3-strong |
|---|---:|---:|---:|---:|---:|
| IoU0.5 R@1 | 30.63 | 35.29 | 37.30 | 37.75 | +0.45 |
| IoU0.5 R@5 | 57.68 | 59.97 | 61.06 | 61.69 | +0.63 |
| IoU0.5 R@10 | 66.67 | 67.87 | 68.35 | 68.82 | +0.47 |
| IoU0.5 R@100 | 75.81 | 75.92 | 76.04 | 76.08 | +0.04 |
| IoU0.7 R@1 | 20.73 | 23.92 | 25.06 | 25.51 | +0.45 |
| IoU0.7 R@5 | 41.41 | 44.06 | 45.97 | 46.47 | +0.50 |
| IoU0.7 R@10 | 49.67 | 51.79 | 53.32 | 53.93 | +0.61 |
| IoU0.7 R@100 | 63.84 | 64.08 | 64.26 | 64.52 | +0.26 |

## Safety diagnostics

```text
top1_changed_ratio                 = 0.231605
hard_positive_top100_exits         = 100
hard_positive_top100_entries       = 6456
hard_positive_top100_exit_ratio    = 0.001594
y_fp_down_move_ratio               = 0.632994
y_joint_05_down_move_ratio         = 0.082540
wrong_video_boundary_bonus_mean    = 0.005767
Pearson(base, C3.1)                = 0.980054
mean within-query Spearman         = 0.979693
base weighted std share            = 0.766818
boundary weighted std share        = 0.019258
```

All frozen safety constraints pass. Fast grid metrics match the bundled evaluator, and the best submission's NMS output matches the frozen official NMS implementation exactly.

## Scope closure

```text
official_val_used                       = false
official_val_metrics_used_for_selection = false
official_val_one_shot_executed          = false
post_val_adjustment                     = false
QSP / R2 / REG / AMD used               = false / false / false / false
next_stage_authorized                    = false
```

C3.1 stops here. Do not run official val or C3.5-QSP without new user authorization.
