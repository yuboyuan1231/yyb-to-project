# C6-B2 final summary

Status: `C6_B2_FREEZE_REVIEW_PASS`

Selected train_calib freeze candidate: `pairwise_main_0005`. This is a train_fit/train_calib-only result; no official val was run or read.

## Selected policy

- arm: `pairwise_main`
- config_id: `pairwise_main_0005`
- apply_slots: `5`
- threshold0: `4.337798070907593`
- threshold_rest: `6.729127645492554`
- max_replacements: `2`
- effective_top_n / max_after_nms / NMS: `100 / 100 / 0.7`

## Metrics vs C6-B1-lite c6b1r1_0057

| metric | C6-B2 | delta vs C6-B1 |
|---|---:|---:|
| 0.7-r1 | 27.051150 | +0.389061 |
| 0.5-r1 | 39.821490 | +0.240302 |
| 0.7-r5 | 48.895755 | +0.045772 |
| 0.5-r5 | 62.421330 | -0.011443 |
| 0.7-r10 | 56.493878 | +0.057215 |
| 0.5-r10 | 69.618950 | +0.011443 |
| 0.7-r100 | 65.522371 | +0.057215 |
| 0.5-r100 | 76.290193 | -0.022886 |

## Movement and safety

- replacement_rate_top_slots: `0.04638974711065339`
- replacements: `2027`
- top1_changed_ratio: `0.05000572147843003`
- hard_positive_top100_query_exits: `1`
- hard_positive_top100_query_entries: `7`
- video_slot_drift: `0.0`
- video_multiset_drift: `0.0`
- invalid_span_count_after_nms: `0`
- duplicate_span_count_after_nms: `0`
- pre_nms_duplicate_candidate_count_all_top100: `1908`
- replacement_duplicate_count_within_query: `418`
- harmful_replacement_rate_iou05_binary: `0.010853478046373951`
- harmful_replacement_rate_iou07_binary: `0.027627035027133696`
- hard_positive_exit_ratio: `0.00011442956860052638`

## Arm comparison

| arm | feasible | 0.7 R@1 Δ | 0.5 R@1 Δ | 0.7 R@5 Δ | 0.5 R@5 Δ | replacements | selection score |
|---|---:|---:|---:|---:|---:|---:|---:|
| listwise_set | False | -1.121410 | -1.190068 | -0.480604 | -0.137315 | 1088 | -8.333265 |
| listwise_set_retry1 | False | -0.961208 | -0.698020 | -0.423389 | -0.102987 | 1084 | -6.431445 |
| listwise_set_retry2 | False | -0.331846 | -0.034329 | -0.366175 | -0.160201 | 1158 | -2.501934 |
| pairwise_main | True | +0.389061 | +0.240302 | +0.045772 | -0.011443 | 2027 | 2.071175 |

## Interpretation

- Pairwise Utility + Abstention/Top-K Guard is the only arm that passes the C6-B2 train_calib freeze gate.
- Listwise/set selector was not abandoned immediately: it was audited and retried twice after poor initial metrics. Both retries remained below the freeze gate, so this line is stopped for this run.
- The selected pairwise policy is R1-positive at both IoU 0.7 and IoU 0.5, keeps video slots fixed, and stays under the 7% replacement cap.
- Pre-NMS duplicate pressure exists because some replacement spans overlap existing same-video slots, but after the unchanged NMS=0.7 final list has zero duplicate spans.
- This is not an official-val result. The next valid step is human review before any exactly-one official-val one-shot authorization.

## Boundary flags

- official_val_used: `false`
- official_val_authorized: `false`
- post_val_adjustment: `false`
- C6_C_used: `false`
- C4_final_modified: `false`
- NMS_modified / evaluator_modified: `false / false`
