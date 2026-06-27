# C6-B1-lite final summary

Status: `C6_B1_LITE_FINALIZED`

- current_promoted_system: `C6-B1-lite c6b1r1_0057`
- previous_promoted_system: `C4_final v21_00444`
- promotion_type: `low-drift candidate-selector positive`
- official_val_one_shot: `true`
- no_second_official_val: `true`
- post_val_adjustment / score_grid_on_val / temperature_search_on_val: `false / false / false`

## Official metric table

| system | 0.5-r1 | 0.5-r5 | 0.5-r10 | 0.5-r100 | 0.7-r1 | 0.7-r5 | 0.7-r10 | 0.7-r100 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| C4_final v21_00444 | 14.81 | 28.57 | 34.81 | 46.45 | 8.35 | 18.64 | 24.63 | 35.70 |
| C6-B1-lite c6b1r1_0057 | 14.92 | 28.66 | 34.93 | 46.56 | 8.39 | 18.82 | 24.82 | 35.88 |
| delta | +0.11 | +0.09 | +0.12 | +0.11 | +0.04 | +0.18 | +0.19 | +0.18 |

## Train-calib positive vs official positive

Train-calib showed larger pseudo gains because it was a controlled fixed-video-slot diagnostic. Official val still confirms all eight metrics positive vs C4_final, but at smaller magnitude.

## Movement / structure

- replacement_rate_top_slots: `0.014838610983631635`
- candidate_replacement_count: `485`
- top1_changed_ratio: `0.03267553923818265`
- hard-positive top100 exits/entries: `3 / 15`
- video_slot_drift / video_multiset_drift: `0.0 / 0.0`
- invalid_span_count / duplicate_span_count: `0 / 0`

## Negative-result chain

C5-lite-prior, C5-main A/A2, C6-A, C6-A-R, and C6-A-R2 all showed that direct score residuals or fixed-candidate adapter reranking were unsafe or failed to transfer front-rank gains. C6-B0 then showed boundary distributions could be useful only if consumed as candidate generation/replacement. C6-B1-lite is the first official-val positive consumer of that signal.

## Next-stage recommendation

`Recommend C6-B2`: 同视频 hybrid candidate pool 的 oracle 仍高于 actual，且 replacement harmful rate 低；优先开 C6-B2 higher-ceiling selector protocol。

Do not start C6-B2/C6-C automatically. This is a review recommendation only.
