# C7 current stage summary

Current promoted system: `C7-B6 R1SelectiveTop1`

Status: `C7_B6_OFFICIAL_PROMOTED`
Archive date: `2026-06-27`

## Official Result

C7-B6 was promoted from the exactly-one official-val one-shot.

```text
official_val_used = true
official_evaluator_call_count = 1
post_val_adjustment = false
second_official_val = false
evaluator_modified = false
nms_modified = false
```

## Method

`R1-oriented selective top1 override` over the C7-B2.1 A4 fixed-pool ranking.

Selected config:

```json
{
  "K_guard": 10,
  "tau": 0.20000000000000007,
  "model": "GateMLP_15_32_16_3"
}
```

This is not a top100 full rerank and not direct S_MINUTE-like replacement. It only applies the frozen selective gate to decide whether to move the SN top candidate to rank1 within the same fixed candidate pool.

## Official Metrics

| metric | C7-B6 | delta vs C7-B2.1 A4 | delta vs C6-B2 |
|---|---:|---:|---:|
| `0.5-r1` | 15.51 | +0.06 | +0.53 |
| `0.5-r5` | 28.79 | -0.01 | +0.13 |
| `0.5-r10` | 34.81 | +0.01 | -0.09 |
| `0.5-r100` | 46.54 | +0.00 | +0.00 |
| `0.7-r1` | 8.75 | +0.05 | +0.30 |
| `0.7-r5` | 19.50 | +0.01 | +0.72 |
| `0.7-r10` | 25.18 | +0.01 | +0.42 |
| `0.7-r100` | 35.81 | +0.00 | -0.01 |

## Safety

```text
fixed_pool_invariant_on_official = true
override_query_count = 272
override_query_rate = 0.024965580541532813
top5_set_changed_rate = 0.0056906837999082145
top10_set_changed_rate = 0.0
hard_positive_top100_query_exits = 0
invalid_span_count = 0
duplicate_span_count_after_nms = 0
```

## Artifact Pointers

- Official one-shot: `c7_audit/C7_B6_OFFICIAL_VAL_ONE_SHOT.json`
- Official decision: `c7_audit/C7_B6_OFFICIAL_DECISION.json`
- Official manifest: `c7_audit/C7_B6_OFFICIAL_MANIFEST.json`
- Current promoted manifest: `c7_audit/CURRENT_PROMOTED_SYSTEM.json`
- Submission: `results/rlem_c7_b6_official_val/official_val_submission.json`
- Gate model: `c7_models/c7_b6_r1_selective_gate.pt`

Stop here. No further official val is authorized by this archive.
