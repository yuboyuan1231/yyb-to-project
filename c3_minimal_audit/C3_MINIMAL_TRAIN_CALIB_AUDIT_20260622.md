# C3-minimal train_calib-only rerank audit — 2026-06-22

## Decision

```text
C3_MINIMAL_PASS
```

The fixed default rerank improves all eight train_calib VCMR metrics without
changing candidates, the effective top-100 input, NMS=0.7, or the evaluator.
Official val was not read or scored. No weight search and no C3-strong run was
performed.

## Instruction assessment

The instruction is technically reasonable with three explicit qualifications:

1. C2 checkpoint selection already used train_calib loss. Therefore this C3
   result is a mechanism and scale check on the same calib set, not an
   independent generalization result.
2. The default weights are treated as predeclared and were run exactly once.
   They were not adjusted after observing metrics.
3. Q_bd was trained only where `m_bd=1`, so its output on wrong-video rows is
   unconstrained. The audit therefore measures the wrong-video bonus directly.

These qualifications do not invalidate C3-minimal, but they prohibit describing
the result as an official-val or final result.

## Frozen commands

S_base reconstruction and one-time train_calib derivative extraction:

```bash
python3 rlem/reconstruct_base_submission.py \
  --evidence_jsonl results/rlem_c1/train_evidence.jsonl.gz \
  --desc_ids_filter results/rlem_c2/splits/train_calib_desc_ids.txt \
  --dataset_config config/tvr_data_config.json \
  --split train \
  --effective_top_n 100 \
  --max_after_nms 100 \
  --nms_thd 0.7 \
  --output_json results/rlem_c3_minimal/train_calib_base_reconstructed_nms_0.7.json \
  --save_filtered_evidence results/rlem_c3_minimal/train_calib_evidence.jsonl.gz \
  --save_gt_jsonl results/rlem_c3_minimal/train_calib_gt.jsonl
```

Fixed C3-default rerank:

```bash
python3 rlem/rerank_with_evidence.py \
  --evidence_jsonl results/rlem_c3_minimal/train_calib_evidence.jsonl.gz \
  --desc_ids_filter results/rlem_c2/splits/train_calib_desc_ids.txt \
  --ckpt results/rlem_c2/basic_heads/model_best.pt \
  --dataset_config config/tvr_data_config.json \
  --split train \
  --output_json results/rlem_c3_minimal/train_calib_c3_default_submission.json \
  --base_score_mode log \
  --base_scale 1.0 \
  --a_joint 1.0 \
  --b_bd 0.5 \
  --d_fp 0.5 \
  --effective_top_n 100 \
  --max_after_nms 100 \
  --nms_thd 0.7 \
  --device cuda \
  --score_batch_size 8192 \
  --save_scored_jsonl results/rlem_c3_minimal/train_calib_c3_default_scored.jsonl.gz \
  --compact_scored_jsonl
```

Both submissions were evaluated against the same train_calib GT derived from
the accepted C1 train evidence. `--no_desc_type` affects only the optional
per-type breakdown; the VCMR headline evaluator and recall definitions are
unchanged.

## Cardinality and protocol gates

```text
queries                    = 8,739
candidate rows/query       = 200
total candidate rows       = 1,747,800
effective pre-NMS input    = score top-100
NMS threshold              = 0.7
maximum after NMS          = 100
```

All S_base values are finite and strictly positive, so `log(S_base)` is
monotonic and introduces no baseline ties through clipping. Both base and C3
submissions are object-exact to an independent top-100 plus NMS recomputation.
The largest actual post-NMS list contains 78 predictions.

## Train_calib metrics

| Metric | S_base | S_C3-default | Delta |
|---|---:|---:|---:|
| IoU0.5 R@1 | 30.63 | **35.29** | **+4.66** |
| IoU0.5 R@5 | 57.68 | **59.97** | **+2.29** |
| IoU0.5 R@10 | 66.67 | **67.87** | **+1.20** |
| IoU0.5 R@100 | 75.81 | **75.92** | **+0.11** |
| IoU0.7 R@1 | 20.73 | **23.92** | **+3.19** |
| IoU0.7 R@5 | 41.41 | **44.06** | **+2.65** |
| IoU0.7 R@10 | 49.67 | **51.79** | **+2.12** |
| IoU0.7 R@100 | 63.84 | **64.08** | **+0.24** |

## Score-scale diagnostics

| Component | Mean | Std | P5 | P50 | P95 |
|---|---:|---:|---:|---:|---:|
| log BaseScore | 6.028519 | 1.367120 | 4.080629 | 5.909234 | 8.456167 |
| Q_joint | 0.036587 | 0.087105 | 0.000120 | 0.005447 | 0.186911 |
| Q_bd | 0.397859 | 0.158716 | 0.145156 | 0.394940 | 0.665264 |
| E_fp | 0.082336 | 0.227995 | 0.000000 | 0.000040 | 0.777572 |
| S_C3 | 6.222868 | 1.363361 | 4.245133 | 6.126823 | 8.553007 |

Weighted standard-deviation shares are Base 82.98%, Q_joint 5.29%, Q_bd
4.82%, and negative E_fp 6.92%. The learned terms perturb rather than replace
the base score. The score formula was checked on all 1,747,800 rows with maximum
absolute error 0.0.

Pearson(Base, C3) is 0.993871 and Spearman is 0.995092. The top-1 candidate
changes for 1,258/8,739 queries (14.40%), which is non-zero but not an abnormal
pool rewrite.

## Movement diagnostics

- A GT-video candidate exists for 7,043 queries. Its best rank improves for
  1,670 queries, is unchanged for 4,486, and degrades for 887.
- Within the GT video, the best-IoU span rank improves for 754, is unchanged for
  5,520, and degrades for 769. This component is approximately balanced; most
  headline gain comes from better video/cross-video ordering.
- Of 148,469 `y_fp=1` candidates, 52.08% move down, with a mean rank change of
  +1.25 positions (`C3 - base`).
- Of 62,745 `y_joint_05=1` candidates, only 9.60% move down; the mean movement is
  -8.96 positions, so hard positives move strongly upward overall.
- Hard-positive top-100 movement is 3,264 entries versus 107 exits. The exit
  ratio is only 0.17%, so E_fp is not mass-suppressing hard positives.
- The E_fp penalty exceeds the combined positive auxiliary boost on 5.98% of
  candidates. Its weighted P95 is 0.3888 versus base-score std 1.3671.
- Wrong-video Q_bd has mean 0.3925, corresponding to a +0.1962 default bonus.
  This is a real limitation because Q_bd is unconstrained there, but it does not
  dominate the score and the observed retrieval metrics improve.

## Integrity

- C1 train evidence hash remains
  `d49aaf49c44eda6e74ce936f31ff1a3c01a9d093e9e4f5c146840cf9548a6f48`.
- C1 official-val evidence hash remains
  `ce9184e6b9e87581328850e6479231832e471caabae8881db9d6bbe1047f9884`.
- C2 accepted checkpoint hash remains
  `e4a8944d836075139211202eac5caf0cb708de878447893b10dd8af34ac6bda9`.
- The train_calib ID hash remains
  `293f4540f23d28f70384b41d225bf395d6a55267e4e666b7448ec0c3fecc569a`.
- All C3 Python files pass `py_compile`.
- `official_val_used = false`.

## Gate and stopping boundary

All C3-minimal gates pass. The positive fixed-default result is technically
eligible for a separately authorized C3-strong train_calib-only weight-search
protocol. That future search must be predeclared and must still not inspect
official val.

Eligibility is not authorization:

```text
eligible_for_c3_strong_train_calib_search = true
next_stage_authorized = false
```

Stop here. No C3-strong or official-val run was executed.
