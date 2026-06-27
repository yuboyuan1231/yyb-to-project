# C6-B2 final promoted-system summary

Status: `C6_B2_FINAL_PROMOTED_SYSTEM`

The current promoted system is now `C6-B2 pairwise_main_0005`. It supersedes `C6-B1-lite c6b1r1_0057` as the active result, while retaining `C4_final v21_00444` as the secondary baseline.

## Frozen config

- arm: `pairwise_main`
- config_id: `pairwise_main_0005`
- apply_slots: `5`
- threshold0: `4.337798070907593`
- threshold_rest: `6.729127645492554`
- max_replacements: `2`
- effective_top_n: `100`
- NMS: `0.7`
- max_after_nms: `100`

## Official-val metrics

| system | 0.5-r1 | 0.5-r5 | 0.5-r10 | 0.5-r100 | 0.7-r1 | 0.7-r5 | 0.7-r10 | 0.7-r100 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| C4_final v21_00444 | 14.81 | 28.57 | 34.81 | 46.45 | 8.35 | 18.64 | 24.63 | 35.70 |
| C6-B1-lite c6b1r1_0057 | 14.92 | 28.66 | 34.93 | 46.56 | 8.39 | 18.82 | 24.82 | 35.88 |
| C6-B2 pairwise_main_0005 | 14.98 | 28.66 | 34.90 | 46.54 | 8.45 | 18.78 | 24.76 | 35.82 |
| Δ vs C6-B1 | +0.06 | +0.00 | -0.03 | -0.02 | +0.06 | -0.04 | -0.06 | -0.06 |
| Δ vs C4_final | +0.17 | +0.09 | +0.09 | +0.09 | +0.10 | +0.14 | +0.13 | +0.12 |

## Movement and safety

- replacement_rate_top_slots: `0.006773749426342359`
- candidate_replacement_count: `369`
- video_slot_drift: `0.0`
- video_multiset_drift: `0.0`
- invalid_span_count: `0`
- duplicate_span_count_after_nms: `0`
- top1_changed_ratio_vs_C6_B1: `0.03129876089949518`
- top1_changed_ratio_vs_C4_final: `0.01101422670949977`
- hard_positive exits/entries vs C6_B1: `9 / 6`
- hard_positive exits/entries vs C4_final: `1 / 10`

## Finalization decision

- Promote `C6-B2 pairwise_main_0005` as the current system.
- C6-B2 official val was run exactly once for the frozen config.
- No retraining, no official-val grid search, no temperature search, no post-val adjustment.
- C4_final, C6-B1-lite, NMS, evaluator, and `model/conquer.py` were not modified.
- Do not run second official val without explicit new authorization.

## Recommended next stage

Next recommended step is `C6-C design review`, focused on retrieval-localization mutualization / video-aware integration. Do not start training C6-C until a protocol is written and explicitly approved.
