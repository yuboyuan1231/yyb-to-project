# C2 Evidence Heads Training Audit — 2026-06-22

## Decision

```text
C2_PASS
```

C2 has completed within its frozen boundary. The model was trained on
`train_fit`, selected and diagnosed on `train_calib`, and did not read official
val for checkpoint selection or learned reranking. C3 is not authorized by this
report.

## Recommendation review and enforced corrections

The supplied C2 procedure was technically sound after three corrections:

1. `FeatureStats.fit` is filtered by `train_fit_desc_ids`; fitting statistics on
   the full train evidence would leak `train_calib` into preprocessing.
2. All unavailable r2 fields are removed from the numeric feature schema rather
   than converting null values into apparently real zeros.
3. Both checkpoint selection and diagnostics use the accepted train evidence
   filtered by query-level split IDs. Official val is not a fallback validation
   source.

## Deterministic query split

Method: SHA256 with salt `conquer-rlem-c2-v1`; a query goes to calib when the
first 64-bit SHA256 integer is congruent to zero modulo ten. Every query keeps
all 200 candidate rows in one split.

| Split | Queries | Rows | m_bd+ | y_joint_05+ | y_joint_07+ | y_fp+ |
|---|---:|---:|---:|---:|---:|---:|
| train_fit | 78,045 | 15,609,000 | 1,268,031 | 564,348 | 257,190 | 1,323,568 |
| train_calib | 8,739 | 1,747,800 | 141,900 | 62,745 | 28,283 | 148,469 |

The split has zero overlapping queries and its union is all 86,784 train
queries. Split files and hashes are frozen in `C2_FROZEN_MANIFEST.json`.

## Feature contract

The accepted schema contains 31 features. These unavailable r2 features are
explicitly disabled:

```text
r2_raw, r2_prob, r2_tilde, rank_r2, r_abs_gap
```

Statistics were fitted on all 15,609,000 `train_fit` rows only. All means and
standard deviations are finite. Q_bd loss is computed only where `m_bd > 0.5`;
wrong-video rows therefore cannot contribute to that loss.

## Performance correction

The original streaming path repeatedly decompressed gzip, parsed one JSON
object, performed Python dictionary lookups, and allocated tiny tensors for
every row. The 31-to-128 MLP was starved by CPU input work, explaining both the
slow run and the low observed GPU occupancy.

Before full training, the same normalized float32 features and labels were
materialized once into memory-mapped NumPy arrays. Training now performs block
shuffle and vectorized batch gathering. Model architecture, loss functions,
labels, normalization, batch size, optimizer, and float32 precision were not
reduced or approximated. A direct comparison of 100 fit and 100 calib cache
rows against `FeatureStats.transform_row` had maximum absolute difference 0.0;
all cache tensors are finite and their label counts exactly match the split
audit.

| Run | Train rows | Calib rows | Wall time |
|---|---:|---:|---:|
| Streaming smoke | 200,000 | 50,000 | 25.44 s |
| Cached smoke | 200,000 | 50,000 | 1.94 s |
| Cached full, 3 epochs | 15,609,000/epoch | 1,747,800/epoch | 83.91 s |

The smoke speedup was 13.11x. The incomplete pre-optimization full run is
preserved at `results/rlem_c2/basic_heads_aborted_preopt_20260622` for audit but
is explicitly rejected: it completed no epoch and produced no accepted
checkpoint.

## Full training and checkpoint selection

Frozen settings:

```text
device       = cuda
epochs       = 3
batch_size   = 1024
lr           = 1e-3
hidden_dim   = 128
seed         = 13
selection    = minimum train_calib total loss
```

| Epoch | Train loss | Train-calib loss | Joint reg | Joint bin | Boundary | False-positive |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.093133 | 0.091989 | 0.007662 | 0.104026 | 0.019085 | 0.078474 |
| 2 | 0.089937 | 0.091007 | 0.007541 | 0.103309 | 0.018884 | 0.077510 |
| 3 | 0.089280 | **0.089721** | 0.007501 | 0.101986 | 0.018089 | 0.077268 |

Epoch 3 is the accepted checkpoint. The checkpoint embeds its exact feature
statistics, training arguments, model configuration, epoch, and metrics.

## Complete train_calib diagnostics

Diagnostics cover all 1,747,800 calib rows.

| Diagnostic | Value |
|---|---:|
| Q_joint vs y_joint Pearson | 0.535325 |
| Q_joint vs y_joint Spearman | 0.355889 |
| Q_joint vs y_joint_05 AUC | 0.912462 |
| Q_bd vs y_bd Pearson, m_bd=1 only | 0.639778 |
| Q_bd vs y_bd Spearman, m_bd=1 only | 0.625726 |
| E_fp vs y_fp AUC | 0.987998 |

Q_joint, Q_bd, and E_fp all have non-zero spread and passed the collapse check.
Detailed means, standard deviations, extrema, percentiles, and label ratios are
in the JSON/Markdown diagnostic artifacts.

## Integrity and boundary checks

- Accepted C1 train evidence SHA256 remains
  `d49aaf49c44eda6e74ce936f31ff1a3c01a9d093e9e4f5c146840cf9548a6f48`.
- Accepted C1 val evidence SHA256 remains
  `ce9184e6b9e87581328850e6479231832e471caabae8881db9d6bbe1047f9884`.
- `train_args.json` records `val_jsonl: null` and uses the two train-derived
  split ID files plus the train-derived cache.
- The diagnostic artifact records `scope: train_calib_only`,
  `official_val_used: false`, and an empty `r2_features_enabled` list.
- All modified/new Python files pass `py_compile`.
- No official-val learned rerank, weight search, C3, C3-strong, QSP, REG, AMD,
  C4, C5, or C6 operation was run.

## Frozen artifacts

- `results/rlem_c2/basic_heads/model_best.pt`
- `results/rlem_c2/basic_heads/feature_stats.json`
- `results/rlem_c2/basic_heads/history.json`
- `results/rlem_c2/basic_heads/train_log.txt`
- `results/rlem_c2/basic_heads/diagnostics_train_calib.json`
- `results/rlem_c2/basic_heads/diagnostics_train_calib.md`
- `results/rlem_c2/cache/cache_manifest.json`
- `results/rlem_c2/splits/split_summary.json`
- `c2_audit/C2_FROZEN_MANIFEST.json`

## Gate

All mandatory C2 conditions pass. The artifacts are technically eligible for
designing a C3-minimal rerank whose search remains train_calib-only. This is
eligibility, not authorization:

```text
next_stage_authorized = false
```

Stop here. Do not run official-val C3.
