# C6-D GenSpan-lite design note

Status: `DESIGN_NOTE_ONLY`

No C6-D training, no text-to-video generation, and no official val are authorized in C6-C1.

## Proposed GenSpan-lite approximation

1. Query event decomposition: split multi-verb queries into ordered event clauses using lightweight text parsing or a future approved LLM pass.
2. Subtitle cue selection: select candidate-video subtitle windows with lexical/entity overlap to each event clause.
3. Event-order temporal prior: convert ordered clauses into monotonic soft windows over candidate spans.
4. Generated-video-free approximation: use subtitle/action cue order and existing temporal priors only; do not synthesize video.
5. Optional future text-to-video prior: allowed only in a separate protocol with cost, privacy, and hallucination controls.
6. Token selector approximation: keep candidate spans whose boundary/proposal evidence aligns with event order and subtitle cue positions.
7. Feed into C6-C: add event-order agreement, cue coverage, and order-risk features to the video reranker and proposal confidence head.
8. Engineering risk: subtitle sparsity, parser error, character aliasing, overfitting to multi-verb heuristics, and higher latency.

This note follows GenSpan's idea of motion/order priors but intentionally avoids generator dependence in the current C6-C1 stage.
