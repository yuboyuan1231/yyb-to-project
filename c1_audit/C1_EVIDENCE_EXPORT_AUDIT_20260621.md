# CONQUER-RLEM C1 Evidence Export Audit

Date: 2026-06-21

Status: `C1_PASS`

## Scope

C1 only: export frozen CONQUER `(q,v,p)` evidence for train and val, audit the
complete artifacts, and reconstruct the C0 baseline using `S_base` only. No C2
model was trained and no learned reranking was run.

## Accepted protocol

```text
checkpoint          = C0 official epoch-9 checkpoint
eval_query_bsz      = 5
max_vcmr_video      = 10
candidate rows/query = 200
effective NMS input  = score top-100
nms_thd             = 0.7
clip_length         = 1.5
```

The original inference code generates 200 rows but mutates its raw result to
top-100 before NMS. C1 retains all 200 rows for later reranking; exact baseline
reconstruction uses the effective top-100 input and NMS 0.7.

## Accepted evidence artifacts

| Split | Queries | Rows | Compressed bytes | SHA256 |
|---|---:|---:|---:|---|
| train | 86,784 | 17,356,800 | 2,793,579,331 | `d49aaf49c44eda6e74ce936f31ff1a3c01a9d093e9e4f5c146840cf9548a6f48` |
| val | 10,895 | 2,179,000 | 344,042,257 | `ce9184e6b9e87581328850e6479231832e471caabae8881db9d6bbe1047f9884` |

Paths:

- `results/rlem_c1/train_evidence.jsonl.gz`
- `results/rlem_c1/val_evidence.jsonl.gz`
- `c1_audit/train_evidence_audit.json`
- `c1_audit/val_evidence_audit.json`

## Full-stream audit

Both artifacts were decompressed and parsed from first row through gzip EOF.

- every source query appears once and in the exact source order;
- every query has exactly 200 rows and ranks 1 through 200;
- all rows share the same 55-field schema;
- required evidence and label fields are present;
- no duplicate `(desc_id, video_name, start_idx, end_idx)` candidates;
- no invalid spans, NaN, infinity, or non-finite scalar values;
- `S_base = r1 * P_b(i) * P_e(j)` within float32 relative tolerance;
- `m_bd` and positive `y_bd` never occur on wrong-video rows;
- modality and QAL context evidence are non-null on every row.

The accepted general-similarity checkpoint has no optional VS/r2 head. The
`r2_raw`, `r2_prob`, `r2_tilde`, `rank_r2`, and `r_abs_gap` keys are retained
as null rather than fabricating unavailable evidence.

## Label counts

| Count | Train | Val |
|---|---:|---:|
| GT-video rows / `m_bd=1` | 1,409,931 | 253,437 |
| `y_joint_05=1` | 627,093 | 56,115 |
| `y_joint_07=1` | 285,473 | 22,303 |
| `y_fp=1` | 1,472,037 | 192,463 |
| wrong-video `m_bd` | 0 | 0 |
| wrong-video positive `y_bd` | 0 | 0 |

## Baseline reconstruction gate

`results/rlem_c1/val_base_reconstructed_nms_0.7.json` was generated using only
`S_base`, effective score top-100 input, and frozen NMS 0.7.

It is object-for-object identical to the C0 official VCMR list for all 10,895
queries, including video ids, spans, scores, ordering, and variable post-NMS
list lengths. Re-evaluation reproduces all eight official VCMR metrics:

```text
IoU 0.5: 13.68 / 27.21 / 33.75 / 46.23
IoU 0.7:  7.76 / 17.22 / 22.49 / 35.17
```

## Corrected implementation defects

1. The supplied QAL code accepted `return_attention=True` but still returned
   one tensor. It now returns QAL plus the Q2V temporal attention only in the
   explicit intermediate-output path. Default C0 outputs remain bit-identical.
2. Full exports now write to `.partial.gz` and atomically replace the final
   path only after clean completion.
3. Added `rlem/reconstruct_base_submission.py` for the mandatory `S_base` gate.
4. Added `rlem/audit_c1_evidence.py` for complete streaming integrity checks.

## Rejected intermediate run

An initial batch-size-16 export produced the same headline metrics but changed
candidate identities for 93 of 10,895 val queries relative to C0. Those
artifacts were rejected and atomically replaced. The accepted train and val
artifacts both use the frozen batch size 5 and the final val reconstruction is
fully identical to C0.

## Gate decision

`C1_PASS`. Stop here. C2 is not authorized by this report.
