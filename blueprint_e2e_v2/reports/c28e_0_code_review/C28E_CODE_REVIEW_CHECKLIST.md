# C28E Code Review Checklist

- PR base must be `c28c-cleanroom-e2e-blueprint-v2`, not `main`.
- No checkpoint, raw prediction, large cache, official output, or NMS/evaluator change is committed.
- Holdout is final-report only; checkpoint and config selection must use `calib_select`.
- `late_interaction_enabled` must execute real clip-level visual/subtitle/joint scoring.
- Clip masks must exclude padded release-feature positions from pooled, partial-relevance, late, and token scores.
- Training loss must score raw candidate clips through the trainable video encoder.
- Raw candidate scoring must be chunked by candidate, preserving `candidate_topk_train` instead of lowering proposal/candidate count.
- Two-stage retrieval must be current-student pooled broad retrieval followed by clip-level reranking.
- Broad pooled recall must be reported separately from late rerank recall.
- Candidate curriculum must be explicit and audited per epoch.
- Late retriever score must enter the full model before retrieval-guided localizer, span-to-video feedback, and joint VCMR scoring.
- Teacher rankings are supervision/diagnostic only, never final static hard gates.
- Full mutual experiments remain gated until retriever passes; the full-model code path must still be present for review.
