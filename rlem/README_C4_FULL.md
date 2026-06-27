# CONQUER-RLEM C4 Full Scaffold

## Scope

This package contains the full C4 code scaffold, but execution is stage-gated.

- **C4-lite** is fully runnable now: fixed-candidate span-quality to video-level calibration.  It does not train R2/VS and does not modify CONQUER.
- **C4-r2-cal** is fully implemented as an external video-level residual MLP calibrator over frozen C4 video features.  It does not modify CONQUER.
- **C4-main** is intentionally guarded.  Dry-run inspection code is included, but backbone VS/R2-head training is refused until C4-lite or C4-r2-cal passes and a separate protocol replaces the guarded stub.

## Non-negotiable protocol

```text
fixed candidates only until C4-main is separately authorized
effective_top_n = 100
NMS = 0.7
max_after_nms = 100
no official-val search
no R2/VS head during C4-lite
no CONQUER source modification during C4-lite or C4-r2-cal
no fp16 approximation
no approximate NMS/evaluator
```

## Stage order

### Stage 0: compile only

```bash
python3 -m py_compile \
  rlem/c4_protocol.py \
  rlem/c4_lite_utils.py \
  rlem/c4_score_candidates.py \
  rlem/build_c4_video_cache.py \
  rlem/grid_search_c4_lite.py \
  rlem/rerank_c4_lite_one_shot.py \
  rlem/c4_video_targets.py \
  rlem/c4_r2_dataset.py \
  rlem/c4_r2_model.py \
  rlem/train_c4_r2_calibrator.py \
  rlem/grid_search_c4_r2_calibrated.py \
  rlem/c4_main_patch_utils.py \
  rlem/train_c4_vs_head.py \
  rlem/export_c4_main_evidence.py \
  rlem/eval_c4_main.py \
  scripts/run_c4_stage.py
```

### Stage 1: C4-lite train_calib only

Score candidates, build cache, and search C4-lite only.  Stop afterward.

### Stage 2: C4-lite official-val one-shot

Requires explicit separate authorization and frozen C4-lite config.

### Stage 3: C4-r2-cal train_fit/train_calib

If C4-lite passes, build train_fit/train_calib video datasets and train the external residual MLP `VideoR2Calibrator`.

This model is not intentionally simplistic:

```text
input video-level frozen features
LayerNorm
Linear -> GELU -> Dropout
2 residual MLP blocks
LayerNorm
video relevance head + auxiliary quality head
```

### Stage 4: C4-r2-cal official-val one-shot

Requires explicit separate authorization.

### Stage 5+: C4-main

Guarded only.  Do not train or patch CONQUER until a separate C4-main protocol is accepted.

## Efficiency design

- Candidate scoring is batched on GPU.
- Candidate JSONL is converted to NPZ cache once.
- Query-video group IDs and offsets are precomputed.
- Video aggregation uses vectorized `numpy.maximum.reduceat`.
- R2-cal uses video-level NPZ datasets and batch GPU training.
- No candidate truncation, fp16, approximate NMS, or evaluator approximation is used.
