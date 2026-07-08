# C28E Code Review Checklist

- PR base must be `c28c-cleanroom-e2e-blueprint-v2`, not `main`.
- No checkpoint, raw prediction, large cache, official output, or NMS/evaluator change is committed.
- Holdout is final-report only; checkpoint and config selection must use `calib_select`.
- `late_interaction_enabled` must execute real clip-level visual/subtitle/joint scoring.
- Training loss must score raw candidate clips through the trainable video encoder.
- Two-stage retrieval must be current-student pooled broad retrieval followed by clip-level reranking.
- Teacher rankings are supervision/diagnostic only, never final static hard gates.
- Full mutual model remains gated until retriever passes the C28E retriever gate.
