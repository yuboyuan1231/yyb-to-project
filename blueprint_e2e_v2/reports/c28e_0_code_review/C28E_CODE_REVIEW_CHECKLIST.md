# C28E Code Review Checklist

- PR base must be `c28c-cleanroom-e2e-blueprint-v2`, not `main`.
- No checkpoint, raw prediction, large cache, official output, or NMS/evaluator change is committed.
- Holdout is final-report only; checkpoint and config selection must use `calib_select`.
- `late_interaction_enabled` must execute real clip-level visual/subtitle/joint scoring.
- Clip masks must exclude padded release-feature positions before temporal mixing and inside pooled, PREM/REG/AMD, late, token, and span-pooling scores.
- Long videos must use duration-aware 64-bin resampling and GT span insertion must use the same grid.
- C28E-3 must run full E2E training through `compute_full_loss`, not retriever-only distillation.
- Full train/eval candidate refresh must use broad pooled recall followed by clip-level late reranking.
- Training loss must score raw candidate clips through the trainable video encoder.
- Raw candidate scoring must be chunked by candidate, preserving `candidate_topk_train` instead of lowering proposal/candidate count.
- Two-stage retrieval must be current-student pooled broad retrieval followed by clip-level reranking.
- Broad pooled recall must be reported separately from late rerank recall.
- Candidate curriculum must be explicit and audited per epoch.
- ActiveMoment must be conditioned on each candidate video.
- Late retriever score must replace the retrieval evidence consumed by retrieval-guided localizer, span-to-video feedback, and joint VCMR scoring.
- Teacher rankings are supervision/diagnostic only, never final static hard gates.
- `calib_select` best checkpoint selection must be implemented for the full model; holdout remains `--allow_holdout_final` only.
