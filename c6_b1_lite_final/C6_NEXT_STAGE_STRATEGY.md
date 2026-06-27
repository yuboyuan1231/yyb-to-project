# C6 next-stage strategy after C6-B1-lite

Status: `STRATEGY_ANALYSIS_ONLY`

This document is not a training authorization. It does not start C6-B2, C6-C, another official-val run, threshold search, selector retraining, or post-val adjustment.

## Current promoted system

- current_promoted_system: `C6-B1-lite c6b1r1_0057`
- previous_promoted_system: `C4_final v21_00444`
- official-val status: `C6_B1_LITE_OFFICIAL_VAL_STRONG_POSITIVE`
- 0.7 R@1: `8.39`
- delta vs C4_final at 0.7 R@1: `+0.04`
- all eight official-val metrics vs C4_final: positive

## What the finalization analysis says

C6-B1-lite is a real positive, but it is deliberately conservative:

- replacement_rate_top_slots: `1.48%`
- top1_changed_ratio: `3.27%`
- hard-positive top100 exits / entries: `3 / 15`
- harmful replacement rate is low: `2.47%` at IoU 0.5 and `2.68%` at IoU 0.7

The ceiling diagnostic shows the same-video hybrid candidate pool still has large unrealized headroom:

- actual diagnostic 0.7 R@1: `8.37`
- greedy slots5/max2 oracle diagnostic 0.7 R@1: `14.27`
- greedy slots5/max5 oracle diagnostic 0.7 R@1: `18.90`
- C4 top100 misses the GT video for `37.59%` of queries, so same-video replacement cannot solve everything.

Interpretation: the immediate bottleneck is not that boundary candidates are useless; the current selector is too conservative and under-consumes a high-ceiling candidate pool. However, a large residual error remains in video ranking / video slot selection, so a pure fixed-video replacement line cannot plausibly close the full gap to recent R1@0.7 targets by itself.

## External method signals

Recent VCMR work points toward mutual retrieval-localization coupling rather than pure late-stage reranking.

- CONQUER already noted that top-k video pool effects saturate quickly for VCMR and that ML/VS joint heads can be difficult to train simultaneously; this supports cautious protocol staging rather than jumping straight into unconstrained end-to-end coupling. Source: https://www.cs.cityu.edu.hk/~wkchan/papers/mm2021-hou%2Bngo%2Bchan.pdf
- PREM frames VCMR as partial relevance: only a small part of an untrimmed video is query-relevant, and it separately strengthens video retrieval and moment localization with modality-aware mechanisms. Source: https://arxiv.org/html/2402.13576v1
- MA-VR reports TVR VCMR R@1@0.7 around `10.04` by explicitly feeding moment-level evidence back into video retrieval, which is conceptually closer to C6-C than C6-B1. Source: https://ieeexplore.ieee.org/document/10902083/
- GenSpan reports TVR VCMR R@1@0.7 `12.12` and emphasizes generated temporal/motion priors, selector quality, and ranking/localization coupling. Source: https://arxiv.org/html/2603.22121v2

## Recommendation

Recommended next action: `C6-B2 protocol/design review first`, not training yet.

Why not jump directly to C6-C?

1. C6-B1-lite just produced the first clean official-val positive after several negative fixed-candidate lines.
2. The oracle pool has a very large same-video R1@0.7 ceiling, so it is wasteful to skip the higher-ceiling selector stage.
3. C6-C/full mutual integration is higher-risk: it can disturb video ranking, candidate generation, and C4_final stability simultaneously.

Why C6-C is still likely necessary later:

1. The gap from current 0.7 R@1 `8.39` to recent `10–12+` results is too large for conservative slot replacement alone.
2. `37.59%` of official-val queries miss the GT video in C4 top100, which no same-video replacement selector can fix.
3. External high-performing methods emphasize moment-aware video retrieval, partial relevance, and retrieval-localization coupling.

## Proposed safe sequence

### Step 1: C6-B2 design review only

Do not train yet. Design a protocol with:

- no official val;
- train_fit/train_calib only;
- primary target: 0.7 R@1;
- secondary guardrail: 0.5/0.7 R@5 non-collapse;
- explicit replacement budget schedule around 3% and 5%;
- stronger selector architecture than B1-lite, but still fixed-video and same-video-slot constrained;
- oracle-guided loss, pairwise original-vs-boundary replacement loss, and calibrated abstention.

### Step 2: C6-B2 train_calib gate

Only if authorized later:

- train a higher-ceiling selector;
- require R1@0.7 positive and R5 safe vs C6-B1-lite;
- require no video slot drift and low hard-positive exits;
- freeze review before any official val.

### Step 3: C6-C mutual integration

If C6-B2 cannot convert enough of the oracle space, move to C6-C:

- video-aware integration;
- moment-aware video retrieval feedback;
- candidate generation and video ranking coupling;
- no full unconstrained end-to-end at first; start with adapter/gated mutualization to protect C6-B1 gains.

## Boundary

This strategy document does not authorize:

- C6-B2 training;
- C6-C;
- official val;
- second official val;
- threshold search;
- selector retraining;
- post-val adjustment.

