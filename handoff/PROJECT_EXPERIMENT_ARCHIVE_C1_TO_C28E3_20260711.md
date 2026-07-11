# Project Experiment Archive: C1 to C28E-3

Date: 2026-07-11

Current branch: `c28e-code-review-clip-late-interaction`

Latest implementation/recovery commit: `963b95d Recover unlogged epoch finalization`

Latest PR: `https://github.com/yuboyuan1231/yyb-to-project/pull/9`

Current promoted official system: `C7-B6 R1SelectiveTop1`

This document summarizes what was tested from C1 through the completed C28E-3 state, what the measured result was, and what the result means for the next stage. It is a handoff archive, not a new experiment.

## Executive Summary

The project has two very different evidence regimes.

First, C1-C7 produced a clean official-promoted system. C1 reproduced the original CONQUER baseline exactly. C2-C3.1 built train-only evidence heads and stronger scalar reranking. C4-C6 explored safe calibration, priors, and selective replacements. C7-B6 then became the promoted official system through a one-shot official validation run. Its official VCMR metrics are:

| system | 0.5-R@1 | 0.5-R@5 | 0.5-R@10 | 0.5-R@100 | 0.7-R@1 | 0.7-R@5 | 0.7-R@10 | 0.7-R@100 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| C7-B6 official | 15.51 | 28.79 | 34.81 | 46.54 | 8.75 | 19.50 | 25.18 | 35.81 |

Second, C8-C28E-3 explored stronger train-only/native/PREM-style directions, but none has replaced C7-B6. The repeated bottleneck is not only span coverage; it is the coupling of video retrieval, span localization, and wrong-video suppression. Several branches recover high VR R@100, but VCMR remains low because wrong-video high-score/top1 rates stay high or span proposals collapse.

The latest result, C26S, is a negative/partial repair result:

- C26R fixed a major C26 one-span proposal collapse: VCMR R@100@0.5 improved from C26 single-span `8.5556` to C26R selected multi-span `34.0`.
- C26S tested PREM-style multi-proposal retriever-localizer mutual promotion on the available C26R row-space.
- C26S did not improve over C26R; final selected integration falls back to the C26R selected route.
- C26S row-space is partial because C26R joined multi-span materialization only covers `calib_select` and `calib_holdout`, not `train_fit`.
- C26S final decision: `C26S_CONTINUE_MUTUAL_PROMOTION_REPAIR`.

C28E-3 then executed the repaired full end-to-end late-interaction design at full configured scale. This is a completed engineering run but a negative promotion result:

- Eight epochs completed on 69,428 `train_fit` queries with 200 final candidates, 1,000 broad candidates, target length 64, hidden dimension 384, and effective batch size 32.
- The run used student broad retrieval plus clip-level late reranking; it did not use the historical static top-128 hard gate.
- Automatic selection chose epoch 7 with selection score `-31.8428`, VR R@100 `15.6390`, and VCMR R@100@0.7 `2.9273` on `calib_select`.
- The primary VCMR R@1@0.7 metric peaked at epoch 6 with only `0.0461`; epoch 7 regressed to `0.0230` on that metric.
- Epoch 7 wrong-video top1 rate remained `99.4353`, so front-rank video retrieval is the dominant failure.
- All declared promotion gates failed. C28E-3 must not replace C7-B6 and must not advance to official validation.
- No official validation or `calib_holdout` result was produced; all reported C28E-3 selection metrics are from `calib_select`.

## Global Safety and Interpretation Rules

1. Only C7-B6 is currently official-promoted.
2. C9 ran an official one-shot and failed; it did not replace C7-B6.
3. Most C12-C26 results are train-only, medium-split, diagnostic, or partial materialization evidence.
4. `pseudo_official_holdout` was repeatedly protected from selection in later stages.
5. Evaluator and NMS modifications are consistently recorded as false in the accepted decision artifacts.
6. C24F/C24G/C24H/C26/C26S evidence should not be described as official-val performance.
7. C26S should not be described as full training ready because `train_fit` multi-span C26R/C26S row-space is absent.
8. C28E-3 completed full training, but its metrics are `calib_select` selection evidence, not official-val or `calib_holdout` evidence.
9. C28E-3 automatic best means best under the configured composite selection score; it does not mean best on the primary R1@0.7 metric.
10. C28E-3 did not pass any promotion gate and does not replace the promoted C7-B6 system.

## C1: Evidence Export and Baseline Reconstruction

Source: `c1_audit/C1_EVIDENCE_EXPORT_AUDIT_20260621.md`

Status: `C1_PASS`

What was tested:

- Exported frozen CONQUER `(q,v,p)` evidence for train and val.
- Audited the complete artifacts through gzip EOF.
- Reconstructed the C0 baseline using `S_base = r1 * P_b(i) * P_e(j)`.
- Verified candidate rows/query, NMS input, NMS threshold, clip length, and schema.

Important protocol:

- Train evidence: 86,784 queries, 17,356,800 rows.
- Val evidence: 10,895 queries, 2,179,000 rows.
- Candidate rows/query: 200.
- Effective NMS input: score top-100.
- NMS threshold: 0.7.
- Clip length: 1.5 seconds.

Result:

- The reconstructed val submission was object-for-object identical to the C0 official VCMR list.
- Reproduced C0 official metrics:
  - IoU 0.5: `13.68 / 27.21 / 33.75 / 46.23`
  - IoU 0.7: `7.76 / 17.22 / 22.49 / 35.17`

Interpretation:

C1 established the frozen evidence substrate and exact baseline reconstruction. This is the root trust anchor for later reranking work.

## C2: Evidence Heads Training

Source: `c2_audit/C2_EVIDENCE_HEADS_AUDIT_20260622.md`

Status: `C2_PASS`

What was tested:

- Trained evidence heads on `train_fit`.
- Selected and diagnosed on `train_calib`.
- Did not use official val.
- Fixed preprocessing leakage by fitting feature stats only on `train_fit`.
- Removed unavailable `r2_*` features from the active schema instead of fabricating zeros.
- Materialized cached NumPy arrays to remove CPU streaming bottleneck.

Split:

| split | queries | rows |
|---|---:|---:|
| train_fit | 78,045 | 15,609,000 |
| train_calib | 8,739 | 1,747,800 |

Training:

- Device: CUDA.
- Epochs: 3.
- Batch size: 1024.
- Hidden dim: 128.
- Selection: minimum train_calib total loss.

Result:

- Best checkpoint: epoch 3.
- Train-calib diagnostics:
  - Q_joint vs y_joint Pearson: `0.535325`
  - Q_joint vs y_joint Spearman: `0.355889`
  - Q_joint vs y_joint_05 AUC: `0.912462`
  - Q_bd vs y_bd Pearson on `m_bd=1`: `0.639778`
  - E_fp vs y_fp AUC: `0.987998`

Interpretation:

C2 showed that learned evidence heads contain useful train-calib signal without official-val access. It did not authorize official reranking by itself.

## C3: Minimal and Strong Train-Calib Reranking

Sources:

- `c3_minimal_audit/C3_MINIMAL_TRAIN_CALIB_AUDIT_20260622.md`
- `c3_strong_audit/C3_STRONG_TRAIN_CALIB_AUDIT_20260622.md`

### C3-Minimal

Status: `C3_MINIMAL_PASS`

What was tested:

- Fixed default rerank using C2 heads.
- No weight search.
- Train_calib only.
- Same candidates, effective top-100 input, NMS=0.7, evaluator unchanged.

Result on train_calib:

| Metric | S_base | C3-default | Delta |
|---|---:|---:|---:|
| IoU0.5 R@1 | 30.63 | 35.29 | +4.66 |
| IoU0.5 R@5 | 57.68 | 59.97 | +2.29 |
| IoU0.5 R@10 | 66.67 | 67.87 | +1.20 |
| IoU0.5 R@100 | 75.81 | 75.92 | +0.11 |
| IoU0.7 R@1 | 20.73 | 23.92 | +3.19 |
| IoU0.7 R@5 | 41.41 | 44.06 | +2.65 |
| IoU0.7 R@10 | 49.67 | 51.79 | +2.12 |
| IoU0.7 R@100 | 63.84 | 64.08 | +0.24 |

Interpretation:

The evidence heads can safely perturb the base score and improve train_calib. This was not an independent generalization result because C2 checkpoint selection also used train_calib loss.

### C3-Strong

Status: `C3_STRONG_PASS`

What was tested:

- 800 predeclared coarse-grid configs plus 80 local refinement configs.
- Selected a gated-boundary formulation:
  - `base_scale = 0.75`
  - `a_joint = 1.625`
  - `b_bd = 0.75`
  - `d_fp = 0.5`
  - `S = 0.75*log(S_base) + 1.625*Q_joint + 0.75*Q_joint*Q_bd - 0.5*E_fp`

Result on train_calib:

| Metric | S_base | C3-minimal | C3-strong |
|---|---:|---:|---:|
| IoU0.5 R@1 | 30.63 | 35.29 | 37.30 |
| IoU0.5 R@5 | 57.68 | 59.97 | 61.06 |
| IoU0.5 R@10 | 66.67 | 67.87 | 68.35 |
| IoU0.5 R@100 | 75.81 | 75.92 | 76.04 |
| IoU0.7 R@1 | 20.73 | 23.92 | 25.06 |
| IoU0.7 R@5 | 41.41 | 44.06 | 45.97 |
| IoU0.7 R@10 | 49.67 | 51.79 | 53.32 |
| IoU0.7 R@100 | 63.84 | 64.08 | 64.26 |

Interpretation:

C3-strong improved all eight train_calib metrics and stayed within stability constraints. It recommended, but did not execute, a one-shot official-val gate.

## C3.1 and C3.5: Capacity Calibration and QSP Branch

Sources:

- `c31_audit/C31_CAPACITY_CALIBRATION_AUDIT_20260622.md`
- `c35_audit/C35_QSP_NEGATIVE_DECISION.json`
- `c35_audit/C35_QSP_MATCHED_FULL_TRAINING_AUDIT.md`

### C3.1

Status: `C31_PASS`

What was tested:

- Larger MLP capacity over frozen 31-scalar evidence.
- 34 full models.
- 4,646 score evaluations.
- Temperature/calibration search on train_calib only.

Best configuration:

- Model: `cap_F_wd0`
- Hidden dims: `[256, 256, 128]`
- Family: `C_centered_gated_boundary`

Result on train_calib:

| Metric | C3-strong | C3.1 | Delta |
|---|---:|---:|---:|
| IoU0.5 R@1 | 37.30 | 37.75 | +0.45 |
| IoU0.5 R@5 | 61.06 | 61.69 | +0.63 |
| IoU0.5 R@10 | 68.35 | 68.82 | +0.47 |
| IoU0.5 R@100 | 76.04 | 76.08 | +0.04 |
| IoU0.7 R@1 | 25.06 | 25.51 | +0.45 |
| IoU0.7 R@5 | 45.97 | 46.47 | +0.50 |
| IoU0.7 R@10 | 53.32 | 53.93 | +0.61 |
| IoU0.7 R@100 | 64.26 | 64.52 | +0.26 |

Interpretation:

C3.1 broke the scalar plateau on train_calib, but still remained train-only and did not execute official val.

### C3.5-QSP

Decision: `STOP_C35_QSP_NEGATIVE`

What was tested:

- Matched QSP52 and QSP53 feature variants against the control31 architecture.
- Full train_fit training with train_calib model selection.

Result:

- QSP variants reduced train loss but increased train_calib loss.
- QSP did not beat matched control31 after matched epoch-8 training and fixed C3.1 score probing.

Interpretation:

QSP was not promoted; it suggested feature conflict/overfitting rather than robust downstream gain.

## C4: Lite Calibration and Official One-Shot

Source: `c4_audit/C4_STAGE_FINAL_SUMMARY.md`

Status: `C4_stage_status = COMPLETE`

Final system: `C4-r2-cal-v2.1`, config `v21_00444`

What was tested:

- C4-lite as the main mechanism.
- A small final calibration residual over C4-lite:
  - `+ 0.075 * z(rel_logit)`
  - `+ 0.050 * z(iou07_logit)`
  - `- 0.100 * z(quality_logit)`
- Official-val one-shot, no post-val adjustment.

Official-val result:

| System | 0.5-R@1 | 0.5-R@5 | 0.5-R@10 | 0.5-R@100 | 0.7-R@1 | 0.7-R@5 | 0.7-R@10 | 0.7-R@100 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Frozen C3.1 | 14.27 | 28.12 | 34.42 | 46.31 | 8.09 | 18.03 | 23.58 | 35.28 |
| Frozen C4-lite | 14.81 | 28.56 | 34.73 | 46.44 | 8.35 | 18.63 | 24.61 | 35.69 |
| C4 final v21_00444 | 14.81 | 28.57 | 34.81 | 46.45 | 8.35 | 18.64 | 24.63 | 35.70 |

Interpretation:

C4 improved over C3.1 on official val, mostly via C4-lite. It was useful but not the final promoted stage after later C6/C7 work.

## C5: Prior and Failure Decomposition

Source: `c5_fd_audit/C5_FAILURE_DECOMPOSITION_SUMMARY.md`

Final classification: `B. C5_MAIN_CANDIDATE`

What was tested:

- Frozen temporal priors and their alignment with GT.
- Whether span mass and residual priors could improve localization.
- Candidate/NMS bottleneck diagnostics.

Key findings:

- `P_bd` peak inside GT: `87.11%`.
- `P_retloc` peak inside GT: `79.71%`.
- GT-window mass exceeds best hard negative only `37.13%`.
- Post-NMS top-100 oracle R@1@0.5 / R@1@0.7: `94.58% / 80.93%`.
- Best-IoU candidate survives NMS in `56.64%`.

Interpretation:

Boundary distributions contained signal, but the external span-mass/residual interface was weak and duration-sensitive. C5-lite-prior was closed under the fixed-candidate residual protocol. C5-main was suggested for human review, not run/promoted.

## C6: Selective Replacement and Safe Policies

Sources:

- `c6_b2_audit/C6_B2_FINAL_DECISION.md`
- `c6_c_audit/C6_C11_FINAL_DECISION.md`

### C6-B2

Status: `C6_B2_FREEZE_REVIEW_PASS`

Selected arm: `pairwise_main`

Best config: `pairwise_main_0005`

Train-only metrics:

| metric | value |
|---|---:|
| 0.5-r1 | 39.8215 |
| 0.5-r5 | 62.4213 |
| 0.5-r10 | 69.6189 |
| 0.5-r100 | 76.2902 |
| 0.7-r1 | 27.0512 |
| 0.7-r5 | 48.8958 |
| 0.7-r10 | 56.4939 |
| 0.7-r100 | 65.5224 |

Interpretation:

C6-B2 passed train-only freeze review but did not automatically run official val.

### C6-C1.1

Status: `C6_C11_SAFE_B2_EQUIVALENT_NO_PROMOTION`

Interpretation:

The C6-C branch did not create new video gain and was not promoted.

## C7: Selective Top1 Override and Official Promotion

Sources:

- `c7_audit/C7_B6_FINAL_DECISION.md`
- `c7_audit/C7_B6_OFFICIAL_DECISION.md`
- `c7_final_archive/C7_FINAL_DECISION.md`

Status before official: `C7_B6_R1_SELECTIVE_PROMISING`

Official status: `C7_B6_OFFICIAL_PROMOTED`

What was tested:

- R1-oriented selective top1 override.
- Guarded replacement, fixed pool invariant.
- No full top100 rerank.
- No evaluator or NMS modification.
- One official-val call, no post-val adjustment, no second official run.

Selected config:

- `K_guard = 10`
- `tau = 0.2`
- Model: `GateMLP_15_32_16_3`

Official-val result:

| system | 0.5-r1 | 0.5-r5 | 0.5-r10 | 0.5-r100 | 0.7-r1 | 0.7-r5 | 0.7-r10 | 0.7-r100 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| C6-B1 official | 14.92 | 28.66 | 34.93 | 46.56 | 8.39 | 18.82 | 24.82 | 35.88 |
| C6-B2 official | 14.98 | 28.66 | 34.90 | 46.54 | 8.45 | 18.78 | 24.76 | 35.82 |
| C7-B2.1 A4 official | 15.45 | 28.80 | 34.80 | 46.54 | 8.70 | 19.49 | 25.17 | 35.81 |
| C7-B6 official | 15.51 | 28.79 | 34.81 | 46.54 | 8.75 | 19.50 | 25.18 | 35.81 |

Movement:

- Override count/rate: `272 / 2.4966%`.
- Top1/top5/top10 changed rate: `2.4966% / 0.5691% / 0.0%`.
- Hard-positive exits/entries: `0 / 0`.

Interpretation:

C7-B6 is the currently promoted official system. Later branches must beat it cleanly before promotion.

## C8: Dual-Track Attempts After C7

Source: `c8_12_dual_track_decision/C8_12_DUAL_TRACK_DECISION.md`

Status: `C8_NO_GAIN_KEEP_C7_B6`

What was tested:

- Exact B6 identity features.
- Safe branch and R1 branch.
- Prototype compression/fusion routes.

Result:

- Exact B6 identity features ready.
- Exact union features not ready.
- Safe branch produced no gain or unsafe behavior.
- R1 branch not ready.

Interpretation:

C8 did not produce a promotable candidate. C7-B6 stayed promoted.

## C9: Integrated CONQUER Ideas and Official Negative

Sources:

- `c9_10d_final_decision_report/C9_10D_FINAL_DECISION_REPORT.md`
- `c9_post_official_failure_audit/C9_FINAL_PROJECT_DECISION.md`
- `c12_system_reliability_audit/C9_TO_C12_SYSTEM_RELIABILITY_AUDIT.md`

Pre-official status: `C9_STAGEB_SAFETY_IMPROVED_OFFICIAL_ONE_SHOT_READY_LOCKED`

Final status: `C9_OFFICIAL_NEGATIVE_KEEP_C7_B6`

What was tested:

- C9 internal audit and several module ideas: MA-VR bridge, multi-video shared norm, partial relevance gate, boundary proposal head.
- StageA/StageB adapter training.
- Safety improved StageB candidate.
- One official-val run.

Result:

- C9 did not beat C7-B6 official.
- C9 was not promoted.
- C7-B6 remained promoted.

Important failure diagnosis:

- A real feature schema mismatch was later found:
  - feature index 10, `video_slot_or_group_id_norm`.
  - train mean around `448.8255`, official mean around `0.02745`.
- This makes C9 train-only optimism unreliable as official readiness evidence.

Interpretation:

C9 is official-negative. It is also a cautionary example: future train/inference pipelines must share the exact feature builder, schema hash, and distribution checks.

## C10: Post-Official Diagnostic Repair

Source: `c10_postofficial_repair_diagnostics/C10_POSTOFFICIAL_REPAIR_DECISION.md`

Status: `C10_POSTOFFICIAL_REPAIR_NO_PROMOTABLE_GAIN_KEEP_C7_B6`

What was tested:

- Exact C7-B6 control.
- Locked C9 reproduction with original feature-10 bug.
- Schema repair.
- Alpha shrinkage.
- Rank-band top1 guards.
- Within-video and same-video-only interventions.
- Strict safe gate.

Main result:

- No tested repaired variant beat exact C7-B6 on the main metric.
- Best non-identity diagnostic variant:
  - `E2_group_id_strict_safe_k5_z1`
  - `0.5-r1 +0.06` vs C7-B6
  - `0.7-r1 -0.05` vs C7-B6
  - top1_changed_rate `0.0390`

Interpretation:

C10 is diagnostic-only because it followed an official failure. It did not create clean promotion evidence. It recommended a clean train-only C11-style branch, but the repo does not contain a separate complete `c11_*` mainline directory; the actual next broad line is C12.

## C12: Native VCMR, Retriever Repair, and Span-Ranker Line

Sources:

- `c12_system_reliability_audit/C9_TO_C12_SYSTEM_RELIABILITY_AUDIT.md`
- `c12_4_native_retriever/C12_4_RETRIEVER_DECISION.md`
- `c12_4r_retriever_repair/C12_4R_RETRIEVER_REPAIR_DECISION.md`
- `c12_5_native_span_generation/C12_5_LOCALIZER_DECISION.md`
- `c12_5r_span_head_repair/C12_5R_SPAN_REPAIR_DECISION.md`
- `c12_5s_rich_span_ranker/C12_5S_RICH_RANKER_DECISION.md`
- `c12_5t_best_span_promotion/C12_5T_DECISION.md`
- `c12_5u_topk_calibration_repair/C12_5U_DECISION.md`
- `c12_5v_conquer_hidden_ranker/C12_5V_DECISION.md`

### C12-4 Native Retriever

Status: `C12_RETRIEVER_COMPLEMENTARY_BUT_WEAK`

Best variant: `teacher_distilled_fusion`

Holdout VR:

| metric | C12 native best | CONQUER reference |
|---|---:|---:|
| VR_R@1 | 0.0691 | 29.1014 |
| VR_R@5 | 0.3917 | 64.4931 |
| VR_R@10 | 0.8525 | 79.1935 |
| VR_R@100 | 6.7512 | 100.0 |

Interpretation:

The fully native retriever was far too weak.

### C12-4R Retriever Repair

Status: `C12_RETRIEVER_CONQUER_WARM_READY_CONTINUE_TO_C12_5`

Best repaired retriever: `zero_delta_replay`

Result:

- CONQUER first-stage replay restored train-only top100 recall.
- Learned delta variants did not robustly beat zero-delta replay.
- C12-5 was allowed only with CONQUER-warm/replay video retrieval.

### C12-5 Localizer and Repairs

C12-5 status: `C12_LOCALIZER_WEAK_REVISE_SPAN_HEAD`

- Best localizer: `teacher_distilled`.
- Holdout IoU@0.7 top100: `43.0876`.
- B6 train-reference IoU@0.7 top100: `66.0156`.
- C12-6 not allowed.

C12-5R status: `C12_LOCALIZER_PQ_RANKING_BOTTLENECK`

- Best repair: `r1_duration_conditioned_dense`.
- IoU@0.7 top100: `43.4562`.
- Short moment IoU@0.7 top100: `31.3535`.
- C12-6 not allowed.

C12-5S status: `C12_SPAN_RANKER_PARTIALLY_REPAIRED_CONTINUE_WITH_CAUTION`

- Best ranker: `S2_span_context_contrast`.
- IoU@0.5 top100: `71.6590`.
- IoU@0.7 top100: `48.3180`.
- Generated M1000 IoU@0.7 remained `85.0461`.
- PQ/IoU calibration remained weak/negative.
- C12-6 not allowed.

C12-5T status: `C12_BEST_SPAN_PROMOTION_STILL_WEAK_NEED_STRONGER_BOUNDARY_MODEL`

- Best variant: `T2_listwise_soft_iou`.
- IoU@0.5 top100: `81.2212`.
- IoU@0.7 top100: `55.0000`.
- IoU@0.7 top50: `38.8710`.
- Short IoU@0.7 top100: `45.0909`.
- Spearman: `-0.1992`.
- AUC@0.7: `0.5473`.
- C12-6 not allowed.

C12-5U status: `C12_SPAN_SCORE_STILL_UNRELIABLE_NEED_CONQUER_HIDDEN`

- Best variant: `U2_two_head_quality`.
- IoU@0.7 top100: `53.3180`.
- IoU@0.7 top50: `38.9631`.
- PQ Spearman: `-0.1481`.
- C12-6 not allowed.

C12-5V status: `C12_HIDDEN_RANKER_NO_GAIN_NEED_STRONGER_FEATURES`

- Best variant: `V1_hidden_boundary`.
- IoU@0.7 top100: `37.7650`.
- PQ Spearman / AUC@0.7: `-0.1896 / 0.4938`.
- Regressed vs C12-5T/C12-5U.
- C12-6 not allowed.

Interpretation:

C12 demonstrated span oracle headroom but could not produce a reliable full VCMR path. The core failure was rank/calibration and independent feature strength, not merely candidate coverage.

## C13: Pivot Decision

Source: `c13_3_next_step_decision/C13_3_NEXT_STEP_DECISION.md`

Status: `C13_STRONG_FEATURE_EXTRACTION_THEN_NATIVE_RETRAIN`

Selected mainline: `Path B`

What was decided:

- Stop C12 span-ranker line.
- Strong text/subtitle features were worth a pilot.
- Strong visual features were worth a pilot but had extraction/storage risk.
- Multiscale/event features were likely the highest-value pivot.
- PREM was not directly reproducible from available public code at that time.

Interpretation:

C13 is a strategic pivot away from repeatedly repairing the same span ranker and toward stronger features / architecture / event modeling.

## C14: Strong Feature and Event Pilot

Sources:

- `c14_2_text_subtitle_pilot/C14_2_TEXT_SUBTITLE_DECISION.md`
- `c14_3_multiscale_event_pilot/C14_3_EVENT_DECISION.md`
- `c14_3_multiscale_event_pilot/C14_INTERIM_DECISION.json`

Statuses:

- `C14_TEXT_SUBTITLE_HARMFUL`
- `C14_EVENT_FEATURE_HARMFUL`
- Interim decision: `C14_NEED_BOUNDARY_ARCHITECTURE_REDESIGN`

What was tested:

- Text/subtitle pilot features.
- Multiscale event pilot features.
- Same-sample comparison against C12 T2 score.

Result:

Both pilots were harmful relative to C12 T2 in the tested setup.

Interpretation:

Naive feature additions did not solve the bottleneck. The next move was boundary/localizer architecture redesign.

## C15: Boundary Architecture Redesign and BMN MVP

Sources:

- `c15_1_localizer_failure_audit/C15_1_LOCALIZER_FAILURE_AUDIT.md`
- `c15_2_bmn_span_confidence_mvp/C15_2_BMN_DECISION.json`

C15-1 status: `C15_FAILURE_SCORE_MAP_REQUIRED`

Failure audit:

- T2 IoU@0.7 top100: `52.6667`.
- T2 generated oracle IoU@0.7: `62.3333`.
- T2 PQ/IoU Spearman: `-0.1331`.
- Raw start peak close within 2 clips: `24.0`.
- Raw end peak close within 2 clips: `18.3333`.

C15-2 status: `C15_BMN_MVP_PROMISING`

Best holdout summary:

- Variant: `B3_query_aware_map`.
- IoU@0.5 top100: `89.0`.
- IoU@0.7 top100: `72.0`.
- IoU@0.7 top50: `51.6667`.
- PQ/IoU Spearman: `0.0849`.
- AUC@0.7: `0.6059`.

Interpretation:

BMN-style dense span confidence maps were promising for localization. They did not yet solve full VCMR integration.

## C16: Full-Scale BMN

Sources:

- `c16_2_fullscale_bmn_training/C16_2_BMN_BEST_MODEL_DECISION.md`
- `c16_5_freeze_review/C16_5_NEXT_STEP_DECISION.json`

C16-2 status: `C16_BMN_FULLSCALE_PROMISING`

Best variant: `B7_full_query_aware_rankloss_map_seed2026`

Result:

- IoU@0.7 top100: `76.0`.
- IoU@0.7 top50: `53.5`.
- Short IoU@0.7 top100: `59.4982`.
- PQ/IoU Spearman: `0.0920`.
- AUC@0.7: `0.6614`.
- Delta vs C12-5T T2 top100: `+19.6`.

C16-5 status: `C16_CONTINUE_BMN_T2_HYBRID_TRAIN_ONLY`

Interpretation:

Full-scale BMN greatly improved train-only localization quality, but it still needed hybrid calibration/integration with retriever evidence.

## C17: BMN/T2 Hybrid and Native VCMR Integration

Sources:

- `c17_3_bmn_t2_hybrid_calibration/C17_3_HYBRID_DECISION.md`
- `c17_6_freeze_review/C17_6_NEXT_STEP_DECISION.json`

Status: `C17_CONTINUE_HYBRID_TRAIN_ONLY`

Best hybrid formula: `G_all_a0.55_b0.25_g0.2`

Result:

- Holdout VCMR R@100 IoU0.7: `22.2`.
- T2-only holdout VCMR R@100 IoU0.7: `1.4`.

Interpretation:

BMN/T2 hybrid was much better than T2-only for VCMR R@100@0.7 in train-only testing, but not official-ready.

## C18 and C19: Full Hybrid Replay and Blocker Closure

Sources:

- `c18_6_freeze_review/C18_6_NEXT_STEP_DECISION.json`
- `c19_5_final_blocker_decision/C19_5_FINAL_BLOCKER_DECISION.json`

C18 status: `C18_CONTINUE_FULL_HYBRID_TRAIN_ONLY`

C19 status: `C19_CONTINUE_TRAIN_ONLY_BLOCKER_CLOSURE`

What was tested:

- Full-score readiness.
- Baseline comparable replay.
- Front-rank hybrid calibration.
- Pseudo-official one-look diagnostics.
- Full consistency/robustness.
- C19 blocker closure and readiness assembly.

Result:

- No promotion.
- Current promoted system remained C7-B6.
- Work continued because train-only hybrid evidence was useful but not clean enough for promotion.

Interpretation:

C18/C19 were consolidation and blocker-closure stages. They did not resolve the full evidence/materialization and front-rank safety requirements.

## C20: Top1 Safe Policy

Source: `c20_6_final_decision/C20_6_FINAL_DECISION.json`

Status: `C20_NEED_RETRIEVER_LOCALIZER_CALIBRATION`

What was tested:

- Anchor availability.
- R1 failure decomposition.
- Same-video span correction.
- Selective cross-video replacement.
- Top1 safe policy assembly.

Final metrics:

| metric | value |
|---|---:|
| VCMR_R@1 IoU0.5 | 3.10 |
| VCMR_R@5 IoU0.5 | 8.00 |
| VCMR_R@10 IoU0.5 | 11.70 |
| VCMR_R@100 IoU0.5 | 37.45 |
| VCMR_R@1 IoU0.7 | 1.40 |
| VCMR_R@5 IoU0.7 | 4.00 |
| VCMR_R@10 IoU0.7 | 6.35 |
| VCMR_R@100 IoU0.7 | 22.65 |
| wrong_video_top1_rate | 71.75 |

Interpretation:

R@100 had useful evidence, but front-rank/top1 remained dominated by wrong-video errors.

## C21: Front-Rank Pairwise Promotion

Source: `c21_6_final_decision/C21_6_FINAL_DECISION.json`

Status: `C21_NEED_EVENTFORMER_PREM_STYLE_COUPLING`

What was tested:

- Front-rank pair dataset.
- Rule-based promotion.
- Learned pairwise promotion.
- Joint front-rank policy.

Final metrics:

- VCMR R@100@0.5: `37.45`.
- VCMR R@100@0.7: `22.65`.
- VCMR R@1@0.5: `2.95`.
- VCMR R@1@0.7: `1.40`.
- wrong_video_top1_rate: `71.75`.
- promotion_precision: `0.0`.
- promotion_recall: `0.0`.

Interpretation:

Front-rank promotion alone did not fix wrong-video top1. The next direction needed stronger retriever/localizer coupling.

## C22 and C22R: Native Partial Relevance and Coupling Repair

Sources:

- `c22_6_final_decision/C22_6_FINAL_DECISION.json`
- `c22r_6_final_decision/C22R_6_FINAL_DECISION.json`

### C22

Status: `C22_NEED_FULL_TRAINFIT_NATIVE_COUPLING`

Result:

- Native VR collapsed:
  - VR_R@1: `0.0`
  - VR_R@10: `0.2`
  - VR_R@100: `1.7`
  - wrong_video_top1_rate: `100.0`
- VCMR remained similar to earlier weak front-rank results because retriever evidence failed.

Interpretation:

C22 native retriever/coupling was broken or misaligned enough to require repair.

### C22R

Status: `C22R_READY_FOR_C23_FULL_TRAINFIT_NATIVE_COUPLING`

Repair result:

- VR_R@1: `28.4`.
- VR_R@5: `63.9`.
- VR_R@10: `78.6`.
- VR_R@100: `100.0`.
- Wrong_video_top1_rate: `71.6`.

Interpretation:

C22R recovered first-stage retriever behavior. It did not solve wrong-video top1, but it made C23 full-trainfit native coupling feasible.

## C23: Full Trainfit First-Stage Preserving Native Coupling

Source: `c23_6_final_decision/C23_6_FINAL_DECISION.json`

Status: `C23_NEED_STRONGER_BMN_EVENT_INTEGRATION`

What was tested:

- Full trainfit readiness.
- Full residual retriever.
- Full evidence materialization.
- Safe native coupling.
- Robustness/onelook firewall.

Final metrics:

| metric | value |
|---|---:|
| VCMR_R@1 IoU0.5 | 2.85 |
| VCMR_R@5 IoU0.5 | 8.25 |
| VCMR_R@10 IoU0.5 | 12.50 |
| VCMR_R@100 IoU0.5 | 36.05 |
| VCMR_R@1 IoU0.7 | 1.30 |
| VCMR_R@5 IoU0.7 | 3.75 |
| VCMR_R@10 IoU0.7 | 6.15 |
| VCMR_R@100 IoU0.7 | 21.05 |
| VR_R@100 | 90.8 |
| wrong_video_high_score_rate | 72.1 |

Interpretation:

C23 preserved some retriever capability but still had severe wrong-video/high-score false positives. Stronger BMN/event integration was required.

## C24: Evidence Strengthening and Full Evidence Repairs

Sources:

- `c24_6_final_decision/C24_6_FINAL_DECISION.json`
- `c24f_6_final_decision/C24F_6_FINAL_DECISION.json`
- `c24g_6_final_decision/C24G_6_FINAL_DECISION.json`
- `c24h_6_final_decision/C24H_6_FINAL_DECISION.json`

### C24

Status: `C24_NEED_FULL_EVIDENCE_MATERIALIZATION_FIRST`

Final metrics:

- VCMR_R@100@0.5: `37.55`.
- VCMR_R@100@0.7: `22.65`.
- VR_R@100: `85.05`.
- wrong_video_high_score_rate: `71.75`.

Interpretation:

C24 improved some R@100 evidence, but the direct event/BMN/T2 evidence columns were not fully materialized. The blocker was evidence materialization, not just scoring.

### C24F

Status: `C24F_NEED_MORE_FULL_EVIDENCE_REPAIR`

What was tested:

- Direct event materialization repair.
- Full evidence signal re-audit.
- Integration replay.

Final metrics degraded sharply:

- VCMR_R@100@0.5: `4.4807`.
- VCMR_R@100@0.7: `1.8330`.
- VR_R@100: `50.9165`.
- wrong_video_high_score_rate: `83.4521`.

Interpretation:

C24F fixed part of the event evidence path but was not a successful VCMR route. It exposed that more evidence repair was needed.

### C24G

Status: `C24G_NEED_MORE_BMN_T2_MATERIALIZATION`

What was tested:

- BMN/T2 source recovery.
- Trainfit materialization checks.
- Full evidence table rebuild attempt.

Result:

- Final VCMR metrics empty.
- BMN/T2 materialization was not sufficient.

Interpretation:

C24G honestly stopped on the missing/incomplete BMN/T2 train_fit evidence problem rather than pretending the full evidence table was ready.

### C24H

Status: `C24H_READY_FOR_C25_RAW_FRAME_STRONG_FEATURE_AUDIT`

What was tested:

- Builder/dependency audit.
- BMN/T2 trainfit rebuild execution.
- Full evidence table closure.
- Signal integration test.

Final metrics:

| metric | value |
|---|---:|
| VCMR_R@100 IoU0.5 | 9.1475 |
| VCMR_R@100 IoU0.7 | 4.3548 |
| VCMR_R@1 IoU0.5 | 3.1567 |
| VCMR_R@1 IoU0.7 | 1.3364 |
| VR_R@1 | 29.2166 |
| VR_R@5 | 64.2166 |
| VR_R@10 | 78.3641 |
| VR_R@100 | 99.6313 |
| wrong_video_top1_rate | 70.7834 |
| high_score_false_positive_rate | 99.6958 |

Interpretation:

C24H rebuilt BMN/T2 evidence enough to close the materialization blocker, but the final VCMR route was still weak. It therefore moved toward C25 raw-frame strong feature audit instead of promotion.

## C25P: Raw-Frame Strong Feature Preflight

Source: `c25p_6_final_readiness_packet/C25P_6_FINAL_DECISION.json`

Final decision: `C25P_ONLY_RELEASE_FEATURES_AVAILABLE`

What was tested:

- Local data/feature inventory.
- Alignment feasibility.
- Model/environment feasibility.
- Feature extraction pilot planning/results for CLIP, DINOv2, VideoMAE, InternVideo.
- Cost/readiness estimate.

Interpretation:

C25P established that only release features were available for the immediate path. Raw-frame strong feature extraction was not yet practically available as the main route.

## C26: PREM-Faithful Release Feature Processing

Sources:

- `c26_6_final_decision/C26_6_FINAL_DECISION.json`
- `c26_7_vcmr_r100_sanity_check/C26_7_VCMR_R100_SANITY_AUDIT.md`

Status: `C26_READY_FOR_C25_STRONG_FEATURE_WHEN_FRAMES_AVAILABLE`

What was tested:

- Release feature alignment.
- Relevant content mining.
- Multimodal collaborative retriever.
- Focus-then-fuse localizer.
- Robustness/route decision.

Final metrics:

| metric | value |
|---|---:|
| VCMR_R@1 IoU0.5 | 2.7778 |
| VCMR_R@5 IoU0.5 | 6.7778 |
| VCMR_R@10 IoU0.5 | 7.5556 |
| VCMR_R@100 IoU0.5 | 8.5556 |
| VCMR_R@1 IoU0.7 | 1.0000 |
| VCMR_R@5 IoU0.7 | 2.4444 |
| VCMR_R@10 IoU0.7 | 2.7778 |
| VCMR_R@100 IoU0.7 | 3.2222 |
| VR_R@100 | 99.8889 |
| wrong_video_top1_rate | 72.5556 |

Interpretation:

C26 had a suspiciously large VR-vs-VCMR gap: VR_R@100 was almost perfect while VCMR_R@100@0.5 was only `8.5556`. This triggered the C26R sanity/repair work.

## C26R: Multi-Proposal Repair

Source: `c26r_multi_proposal_repair/C26R_FINAL_DECISION.md`

Status: `C26R_MULTI_PROPOSAL_REPAIR_PROMISING`

What was tested:

- C26 single-span vs multi-span proposal space.
- C24H baseline replay on same subset.
- C26 retriever plus C24H localizer/span.
- C24H retriever plus C26 focus-fuse span.
- Span unit sanity.
- TopM proposal curve.

Main finding:

The low C26 VCMR was largely caused by proposal/localizer collapse, especially single-span output behavior. The retriever itself had useful recall.

Result:

- Best selected inference method by calib_select: `R3b1_C26_PREM_guarded_localizer`.
- Holdout VCMR R@100@0.5: `34.0`.
- C26 single-span VCMR R@100@0.5: `8.5556`.
- Highest inference holdout R@100@0.5: `R1c_zero_delta_c26_final_video_multi_span = 36.6667`.
- Diagnostic-only oracle headroom R@100@0.5: `71.6667`; excluded because it uses GT IoU.

Interpretation:

C26R is a promising repair, but not a promotion. It proves multi-span proposal space matters and that the C26 low R@100 was not purely a retriever failure.

## C26S: PREM-Style Multi-Proposal Mutual Promotion

Sources:

- `c26s_6_final_decision/C26S_6_FINAL_DECISION.json`
- `c26s_6_final_decision/C26S_6_FINAL_DECISION.md`
- `run_c26s_prem_multiproposal_mutual_promotion.py`
- `tools/c26s/`

Status: `C26S_CONTINUE_MUTUAL_PROMOTION_REPAIR`

What was tested:

- Canonical C26R multi-span row-space.
- PREM-style span relevance features.
- Span-to-video feedback features.
- Mutual promotion integration formulas.
- Robustness and route decision.

Important row-space limitation:

- Available C26R joined multi-span table covers:
  - `calib_select`: 900 queries.
  - `calib_holdout`: 900 queries.
- It does not cover `train_fit`.
- Therefore C26S row-space status is `C26S_MULTISPAN_ROWSPACE_PARTIAL`.

Selected components:

- Span ranker: `S4_false_positive_guard`.
- Feedback variant: `F5_BMN_T2_event_agreement`.
- Selected integration: `C_C26R_multi_span_selected_R3b1`.

Final metrics:

| metric | value |
|---|---:|
| VCMR_R@1 IoU0.5 | 2.6667 |
| VCMR_R@5 IoU0.5 | 7.7778 |
| VCMR_R@10 IoU0.5 | 12.0000 |
| VCMR_R@100 IoU0.5 | 34.0000 |
| VCMR_R@1 IoU0.7 | 1.1111 |
| VCMR_R@5 IoU0.7 | 3.5556 |
| VCMR_R@10 IoU0.7 | 5.6667 |
| VCMR_R@100 IoU0.7 | 19.5556 |
| VR_R@1 | 26.7778 |
| VR_R@5 | 57.7778 |
| VR_R@10 | 68.7778 |
| VR_R@100 | 94.4444 |
| wrong_video_high_score_rate | 73.2222 |

Delta vs C26R multi-span:

- All key deltas are `0.0` because the final selected route fell back to C26R selected multi-span baseline.

Interpretation:

C26S did not improve over C26R. It is useful as an audit showing that the tested mutual-promotion features were not enough to reduce wrong-video risk or improve front-rank VCMR. The correct next status is continuing repair, not full training readiness and not promotion.

## C28E-3: Full End-to-End Clip Late-Interaction Training

Sources:

- `blueprint_e2e_v2/configs/c28e_code_review.yaml`
- `blueprint_e2e_v2/engine/train.py`
- `blueprint_e2e_v2/reports/c28e_3_training/C28E_3_FULL_E2E_TRAINING_DECISION.json`
- `blueprint_e2e_v2/reports/c28e_3_training/C28E3_formal_cuda0_20260709_stream.log`
- `blueprint_e2e_v2/reports/c28e_3_training/C28C_FULL_medium_seed2026.training_log.json`

Run status: `C28E_FULL_E2E_TRAINING_COMPLETE`

Archive conclusion: `C28E-3_NO_PROMOTION_CONTINUE_RETRIEVAL_REPAIR`

What was tested:

- A clean `blueprint_e2e_v2` full-model path instead of extending historical Cxx runners.
- Current-student broad video retrieval followed by clip-level late-interaction reranking.
- Query-token MaxSim, pooled video score, and late clip score fusion.
- Dynamic candidates refreshed every epoch.
- Full multi-loss coupling for video retrieval, span localization, joint VCMR ranking, hard negatives, retrieval/localization feedback, partial relevance, region loss, consistency, and no-regression terms.
- Teacher candidates used only as training curriculum support; evaluation candidates were student-only.
- Step-level crash recovery with deterministic train order, optimizer/RNG state, dynamic candidate stores, and 100-step checkpoints.

Scale and execution:

| item | value |
|---|---:|
| train queries | 69,428 |
| selection queries | 8,677 |
| video bank | 17,435 |
| epochs | 8 (0 through 7) |
| final candidates/query | 200 |
| broad candidates/query | 1,000 |
| max spans/video | 64 |
| target sequence length | 64 |
| hidden dimension | 384 |
| effective batch size | 32 |
| CPU/GPU microbatch | 2 / 2 |
| seed | 2026 |
| device | physical GPU0 (`cuda:0`) |

Mechanism-fidelity audit:

- `candidate_count_min = candidate_count_max = 200`.
- `broad_topk = 1000`.
- Candidate source: `pooled_broad_plus_clip_late_rerank`.
- Late candidate mining was requested and used.
- Video and clip banks were encoded once per refresh on GPU.
- The historical static top-128 hard gate was not used.
- Training used effective in-batch retrieval size 32 even though memory-safe microbatches were size 2.
- Proposal/candidate/model scale was not reduced to handle memory or runtime.

Safety/protocol audit:

- Official validation used: false.
- Pseudo-official holdout used for selection: false.
- Official prediction pool used: false.
- Evaluator modified: false.
- NMS modified: false.
- Selection split: `calib_select`.
- Although the config names `calib_holdout` as `final_report_split`, this run did not execute or report a final holdout evaluation.

Per-epoch selection trajectory on `calib_select`:

| epoch | train loss | select score | VCMR R@1@0.7 | R@5@0.7 | R@10@0.7 | R@100@0.7 | VR R@100 | wrong-video top1 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 44.7158 | -47.1361 | 0.0115 | 0.0230 | 0.0346 | 0.1383 | 2.6507 | 99.9424 |
| 1 | 45.1482 | -45.8165 | 0.0000 | 0.0115 | 0.0346 | 0.2881 | 4.0682 | 99.9770 |
| 2 | 44.7955 | -46.3294 | 0.0000 | 0.0230 | 0.0461 | 0.2305 | 3.4920 | 99.9654 |
| 3 | 43.4786 | -45.0271 | 0.0000 | 0.0000 | 0.0461 | 0.4725 | 4.8750 | 99.9885 |
| 4 | 43.3751 | -43.5865 | 0.0000 | 0.0461 | 0.0807 | 0.5301 | 6.0620 | 99.8963 |
| 5 | 42.4838 | -38.9939 | 0.0115 | 0.0691 | 0.1729 | 1.1870 | 10.2801 | 99.7465 |
| 6 | 41.4286 | -37.5014 | **0.0461** | 0.1959 | 0.3918 | 1.9362 | 10.7756 | 99.6658 |
| 7 | **39.7513** | **-31.8428** | 0.0230 | **0.3457** | **0.5532** | **2.9273** | **15.6390** | **99.4353** |

Automatic best checkpoint, epoch 7:

| metric | value |
|---|---:|
| VCMR R@1 IoU0.5 | 0.1613 |
| VCMR R@5 IoU0.5 | 0.8183 |
| VCMR R@10 IoU0.5 | 1.1986 |
| VCMR R@100 IoU0.5 | 5.5664 |
| VCMR R@1 IoU0.7 | 0.0230 |
| VCMR R@5 IoU0.7 | 0.3457 |
| VCMR R@10 IoU0.7 | 0.5532 |
| VCMR R@100 IoU0.7 | 2.9273 |
| VR R@1 | 0.5647 |
| VR R@5 | 2.3395 |
| VR R@10 | 3.9530 |
| VR R@100 | 15.6390 |
| wrong-video top1 rate | 99.4353 |
| high-score false-positive rate | 99.4353 |
| top1 mean IoU | 0.00224 |

Primary-metric checkpoint distinction:

- Epoch 6 is the measured peak for VCMR R@1@0.7: `0.0461`.
- Epoch 7 is the composite-selection winner: `-31.8428`, but its R@1@0.7 falls to `0.0230`.
- The configured selection score rewards R@5/R@10, VR R@100, and wrong-video reduction strongly enough to prefer epoch 7.
- The rolling checkpoint policy retained epoch 7 as both latest and automatic best; the epoch 6 weights were not retained as a separate primary-metric-best artifact.

Promotion-gate audit using epoch 7 automatic best:

| gate | required | observed | result |
|---|---:|---:|---|
| VCMR R@1@0.7 | >= 3.0 | 0.0230 | FAIL |
| VCMR R@5@0.7 | >= 8.0 | 0.3457 | FAIL |
| VCMR R@10@0.7 | >= 12.0 | 0.5532 | FAIL |
| VR R@100 | >= 80.0 | 15.6390 | FAIL |
| wrong-video top1 | <= 95.0 | 99.4353 | FAIL |
| high-score false-positive | <= 95.0 | 99.4353 | FAIL |

Recovery and completion evidence:

- Server failures interrupted the run more than once.
- The added step recovery saved model, optimizer, RNG, deterministic train order, loss accumulators, cursor, and per-epoch candidate store every 100 steps.
- Recovery was verified at epoch 3 step 1,000 and later used to resume epoch 6 at step 1,900 without restarting the epoch.
- A second recovery repair detected an epoch checkpoint whose evaluation had not yet been written, reran only epoch finalization/evaluation, and preserved the completed training work.
- Final log marker: `C28E_FULL_E2E_TRAINING_COMPLETE`.
- Final invocation elapsed time: `17,665.68` seconds; this is not the total wall-clock duration across server crashes.
- Epoch 7 latest/best checkpoint size: 97,704,933 bytes.
- Epoch 7 latest/best SHA-256: `fa0bafc6b5c1a0d049d462bcd0b77a073222715af0713ae291c05ea8631d13ab`.
- Model weights and large candidate/cache files remain local and are not committed to GitHub.

Analysis:

1. Optimization was real but did not solve front-rank retrieval.
   - Loss fell from `44.7158` at epoch 0 to `39.7513` at epoch 7.
   - VR R@100 rose from `2.6507` to `15.6390` and VCMR R@100@0.7 rose from `0.1383` to `2.9273`.
   - R@1@0.7 stayed near zero and was non-monotonic, so the model learned broad ranking signal without learning a reliable top result.

2. Wrong-video ranking is the dominant top1 failure.
   - Epoch 7 wrong-video top1 rate is `99.4353`.
   - Correct-video/wrong-span rate is only `0.4034`, while R@1@0.5 is `0.1613`.
   - Therefore nearly all top1 failures occur before span quality matters: the selected video is wrong.

3. The candidate curriculum has a material train/eval distribution gap.
   - Epochs 6 and 7 training report teacher warm-start use on 100% of queries with `teacher_warm_topk = 40`.
   - Evaluation uses 0% teacher warm start and no GT insertion.
   - `teacher_curriculum_phase3_topk = 0` is configured, but phase 3 starts after epoch 8 while this run contains epochs 0-7. The run therefore never trained a teacher-free final phase.
   - This is a plausible contributor to weak student-only evaluation retrieval and should be repaired before another long run.

4. The automatic selection objective is not sufficiently aligned with the stated primary metric.
   - Epoch 7 correctly wins the configured composite score because its deeper recall is better.
   - Epoch 6 is twice as high on R@1@0.7, though still far below an acceptable level.
   - Future runs need both composite-best and primary-R1-best checkpoint retention, plus a lexicographic or hard R1 gate.

5. Full proposal scale and late interaction did not rescue the model by themselves.
   - The run preserved 200/1,000 candidate scale and used the intended late-interaction path.
   - The negative result therefore cannot be attributed to a speed shortcut, reduced proposal count, or fallback to the old static candidate gate.
   - The remaining problem is the learned ranking behavior and curriculum, not missing mechanism names.

6. This result is selection evidence only.
   - No `calib_holdout` or official validation metric exists for C28E-3.
   - Because every promotion gate already fails on `calib_select`, spending the protected holdout or official validation is not justified.

Artifact caveat:

- The raw `training_log.json` retains a stale top-level `status: running` value because the per-epoch writer never rewrites it to complete after the loop.
- The authoritative decision JSON and stream log both record `C28E_FULL_E2E_TRAINING_COMPLETE`, and all eight epoch records are present.
- This is a metadata-finalization defect, not an incomplete training run.

Final interpretation:

C28E-3 is valuable because it closes the implementation-fidelity question: the full dynamic, late-interaction, multi-loss model can train at intended scale, survive crashes, and improve deeper recall. It does not close the quality gap. The run fails every declared gate, leaves wrong-video top1 above 99%, and never reaches a teacher-free curriculum phase. C28E-3 is therefore not promotable and does not replace C7-B6.

## Current Technical Diagnosis

The most reliable diagnosis across C20-C28E-3 is:

1. Video retrieval can be strong under CONQUER-warm / repaired / preserved routes.
   - Examples: C22R VR_R@100 `100.0`, C24H VR_R@100 `99.6313`, C26 VR_R@100 `99.8889`.

2. VCMR remains much weaker than VR.
   - C26: VR_R@100 `99.8889`, but VCMR_R@100@0.5 `8.5556`.
   - C24H: VR_R@100 `99.6313`, but VCMR_R@100@0.5 `9.1475`.
   - C26R/C26S repair improves VCMR_R@100@0.5 to `34.0`, but wrong-video high-score rate remains `73.2222`.

3. One-span or weak proposal materialization can destroy VCMR even when retriever recall is high.
   - C26R proved this by lifting R@100@0.5 from `8.5556` to `34.0` using multi-span proposal repair.

4. Span/localizer signal has headroom but calibration is fragile.
   - C12 generated M1000 IoU@0.7 headroom reached `85.0461`, but ranked top100 was much lower.
   - C15/C16 BMN maps improved localization substantially, but integration did not become official-safe.

5. Naive event/text additions were harmful or insufficient.
   - C14 text/subtitle and event pilots were harmful.
   - C24F direct-event materialization repair did not produce a successful route.

6. The next clean improvement likely needs one of:
   - Full train_fit multi-span materialization for C26R/C26S-style proposal space.
   - Better wrong-video suppression tied to span evidence, not just video score.
   - Stronger raw-frame or stronger pretrained visual features if C25 becomes available.
   - A localizer/retriever joint model that keeps multiple proposals per video and avoids single-span collapse.

7. C28E-3 confirms that mechanism completeness and full candidate scale are not sufficient.
   - It used 1,000 broad candidates, 200 late-reranked candidates, 64 spans/video, full-scale training, and all intended coupled losses.
   - Despite this, epoch 7 wrong-video top1 remained `99.4353` and VR R@1 was only `0.5647`.

8. The immediate C28E-specific defect is curriculum and objective alignment.
   - Training remained teacher-warm through the final epoch, while evaluation was teacher-free.
   - The configured teacher-free phase was unreachable within eight epochs.
   - Composite checkpoint selection preferred deeper recall over the primary R1@0.7 peak.

## Recommended Next Actions

Immediate safe next action:

1. Do not promote C26S.
2. Keep C7-B6 as promoted official.
3. Treat C26R as the best current diagnostic repair for the C26 low-VCMR issue.
4. If continuing the release-feature route, rebuild C26R/C26S multi-span row-space for `train_fit`, not only `calib_select/calib_holdout`.
5. After full row-space exists, train/test mutual-promotion or wrong-video suppression on train_fit -> calib_select, and only report calib_holdout once.

For the C28E route:

1. Do not run C28E-3 on `calib_holdout` or official validation; it already fails all `calib_select` gates.
2. Repair the curriculum so the final training phase is genuinely teacher-free, either by moving phase 3 inside the eight-epoch schedule or by predeclaring enough epochs to reach it.
3. Add stage-wise retrieval diagnostics: broad-top1000 GT recall, late-top200 GT recall, final VR R@1/R@10/R@100, and teacher-vs-student candidate overlap.
4. Distill teacher ranking into the student score itself rather than relying on teacher candidate injection throughout training.
5. Make primary R1@0.7 a hard or lexicographic checkpoint criterion, while retaining a separate composite-best checkpoint.
6. Save immutable per-epoch or metric-specific checkpoints so a primary-metric peak such as epoch 6 cannot be overwritten.
7. Repair final metadata writing so a completed run cannot leave `training_log.json` marked `running`.
8. Require a short preflight to demonstrate strong teacher-free VR before another full eight-epoch run.

If raw frames / stronger features become available:

1. Resume the C25 strong-feature route.
2. Prioritize features that can discriminate wrong-video false positives and improve local span evidence.
3. Preserve multi-proposal output; do not regress to one-span localizer output.

Do not do:

1. Do not claim C26S is full training ready.
2. Do not use GT/oracle overlap as inference feature.
3. Do not use pseudo_official_holdout for model selection.
4. Do not run additional official validation without explicit authorization and a branch-local one-shot guard.
5. Do not modify evaluator or NMS to create gains.
6. Do not add more epochs under the same unreachable teacher-free schedule without first fixing the curriculum transition.
7. Do not treat epoch 7 composite-best as evidence that the primary R1@0.7 improved monotonically.

## Source Index

Key source files used for this archive:

- `c1_audit/C1_EVIDENCE_EXPORT_AUDIT_20260621.md`
- `c2_audit/C2_EVIDENCE_HEADS_AUDIT_20260622.md`
- `c3_minimal_audit/C3_MINIMAL_TRAIN_CALIB_AUDIT_20260622.md`
- `c3_strong_audit/C3_STRONG_TRAIN_CALIB_AUDIT_20260622.md`
- `c31_audit/C31_CAPACITY_CALIBRATION_AUDIT_20260622.md`
- `c35_audit/C35_QSP_NEGATIVE_DECISION.json`
- `c4_audit/C4_STAGE_FINAL_SUMMARY.md`
- `c5_fd_audit/C5_FAILURE_DECOMPOSITION_SUMMARY.md`
- `c6_b2_audit/C6_B2_FINAL_DECISION.md`
- `c7_audit/C7_B6_OFFICIAL_DECISION.md`
- `c7_final_archive/C7_FINAL_DECISION.md`
- `c8_12_dual_track_decision/C8_12_DUAL_TRACK_DECISION.md`
- `c9_post_official_failure_audit/C9_FINAL_PROJECT_DECISION.md`
- `c10_postofficial_repair_diagnostics/C10_POSTOFFICIAL_REPAIR_DECISION.md`
- `c12_system_reliability_audit/C9_TO_C12_SYSTEM_RELIABILITY_AUDIT.md`
- `c12_5t_best_span_promotion/C12_5T_DECISION.md`
- `c12_5v_conquer_hidden_ranker/C12_5V_DECISION.md`
- `c13_3_next_step_decision/C13_3_NEXT_STEP_DECISION.md`
- `c14_3_multiscale_event_pilot/C14_INTERIM_DECISION.json`
- `c15_2_bmn_span_confidence_mvp/C15_2_BMN_DECISION.json`
- `c16_2_fullscale_bmn_training/C16_2_BMN_BEST_MODEL_DECISION.md`
- `c17_3_bmn_t2_hybrid_calibration/C17_3_HYBRID_DECISION.md`
- `c20_6_final_decision/C20_6_FINAL_DECISION.json`
- `c21_6_final_decision/C21_6_FINAL_DECISION.json`
- `c22_6_final_decision/C22_6_FINAL_DECISION.json`
- `c22r_6_final_decision/C22R_6_FINAL_DECISION.json`
- `c23_6_final_decision/C23_6_FINAL_DECISION.json`
- `c24_6_final_decision/C24_6_FINAL_DECISION.json`
- `c24f_6_final_decision/C24F_6_FINAL_DECISION.json`
- `c24g_6_final_decision/C24G_6_FINAL_DECISION.json`
- `c24h_6_final_decision/C24H_6_FINAL_DECISION.json`
- `c25p_6_final_readiness_packet/C25P_6_FINAL_DECISION.json`
- `c26_6_final_decision/C26_6_FINAL_DECISION.json`
- `c26r_multi_proposal_repair/C26R_FINAL_DECISION.md`
- `c26s_6_final_decision/C26S_6_FINAL_DECISION.json`
- `blueprint_e2e_v2/configs/c28e_code_review.yaml`
- `blueprint_e2e_v2/engine/train.py`
- `blueprint_e2e_v2/reports/c28e_3_training/C28E_3_FULL_E2E_TRAINING_DECISION.json`
- `blueprint_e2e_v2/reports/c28e_3_training/C28E3_formal_cuda0_20260709_stream.log`
- `blueprint_e2e_v2/reports/c28e_3_training/C28C_FULL_medium_seed2026.training_log.json`
