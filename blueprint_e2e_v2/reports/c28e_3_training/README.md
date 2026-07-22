# C28E-3 Result Package

Run status: `C28E_FULL_E2E_TRAINING_COMPLETE`

Archive conclusion: `C28E-3_NO_PROMOTION_CONTINUE_RETRIEVAL_REPAIR`

## Evidence Files

- `C28E_3_FULL_E2E_TRAINING_DECISION.json`: authoritative completion decision, safety flags, full training records, and checkpoint manifests.
- `C28C_FULL_medium_seed2026.training_log.json`: raw per-epoch loss, selection metrics, candidate audits, and score-scale audits.
- `C28E3_formal_cuda0_20260709_stream.log`: append-only execution log across crash recovery and final completion.
- `../../configs/c28e_code_review.yaml`: exact scale, curriculum, loss, and selection settings.
- `../../engine/train.py`: full training and step-recovery implementation.
- `../../../handoff/PROJECT_EXPERIMENT_ARCHIVE_C1_TO_C28E3_20260711.md`: integrated project history, C28E-3 conclusion, and next-step analysis.

## Result Summary

All reported metrics are from `calib_select` (8,677 queries). No `calib_holdout` or official validation result was produced.

| checkpoint interpretation | epoch | VCMR R@1@0.7 | VCMR R@100@0.7 | VR R@100 | wrong-video top1 | select score |
|---|---:|---:|---:|---:|---:|---:|
| primary-R1 peak (metrics only) | 6 | 0.0461 | 1.9362 | 10.7756 | 99.6658 | -37.5014 |
| automatic composite best | 7 | 0.0230 | 2.9273 | 15.6390 | 99.4353 | -31.8428 |

Epoch 7 fails every configured promotion gate. The promoted system remains `C7-B6 R1SelectiveTop1`.

## Artifact Notes

- The rolling checkpoint policy retained epoch 7 as both latest and automatic best. Epoch 6 weights were not retained separately.
- Latest/best checkpoint size: 97,704,933 bytes.
- Latest/best checkpoint SHA-256: `fa0bafc6b5c1a0d049d462bcd0b77a073222715af0713ae291c05ea8631d13ab`.
- Checkpoints, candidate stores, and feature caches remain local and are intentionally excluded from GitHub.
- The raw training JSON has a stale top-level `status: running`; the authoritative decision JSON and stream log record completion, and all epochs 0-7 are present.

## Main Diagnosis

The full mechanism ran at configured scale (broad top-1,000, final top-200, hidden 384, target length 64, effective batch 32). The failure is not caused by reduced proposal scale or a static-candidate fallback.

The dominant error is front-rank video retrieval: epoch 7 wrong-video top1 is 99.4353%. Training also remained teacher-warm through the last epoch, while evaluation was teacher-free. The configured zero-teacher phase is unreachable in an eight-epoch run. Before another full run, repair that curriculum transition, add stage-wise student retrieval recall diagnostics, and retain separate composite-best and primary-R1-best checkpoints.
