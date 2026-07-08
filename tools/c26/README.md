# C26 PREM-style Release-feature Processing

C26 implements a PREM-inspired train-only path over existing TVR release
features. It does not reproduce official PREM code, does not require raw frames,
and never runs official validation.

Modules:

- `prem_feature_dataset.py`: first-stage top128 canonical rows, release-feature
  manifests, training arrays, and token/time alignment audits.
- `prem_modules.py`: modality-specific query pooling, visual/subtitle/joint
  partial relevance, modality gates, clipped residual retriever, and lightweight
  focus-then-fuse span scorer.
- `prem_losses.py`: video rank, partial relevance, strong/weak, hard negative,
  residual norm, and no-regression losses.
- `prem_train.py`: GPU training/scoring over continuous cached arrays.
- `prem_eval.py`: VR/VCMR proxy metrics, selection score, and breakdowns.
- `prem_integration.py`: first-stage-preserving formula evaluation with C24H
  auxiliary evidence kept separate from PREM partial relevance.

Large arrays, score tables, and checkpoints stay under
`/tmp/c26_score_cache/CONQUER-RLEM-c2c3`.
