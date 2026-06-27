# CONQUER-RLEM C5-lite-prior Code Bundle

## Purpose

This bundle supersedes the earlier row-level C5-lite sketch.

The corrected C5 stage starts with a read-only per-video temporal prior export:

```text
P_b(q,v,t), P_e(q,v,t), P_ctx(q,v,t)
```

Then it builds retrieval-conditioned temporal priors and aggregates them to the existing fixed candidate spans.  This keeps the current protocol clean while aligning C5 with the blueprint requirement that retrieval evidence should condition localization evidence, not just add another row-level residual.

## Current stage definition

```text
C4_final = C4-r2-cal-v2.1 v21_00444
C5-lite-prior baseline = C4_final
C5-main = not used
CONQUER modified = false
candidate generation modified = false
NMS modified = false
```

## Files

```text
rlem/score_c4_r2_v21_split.py
rlem/c5_prior_utils.py
rlem/check_c5_temporal_prior_inputs.py
rlem/export_c5_temporal_prior_from_jsonl.py
rlem/build_c5_lite_prior_features.py
rlem/fit_c5_lite_prior_stats.py
rlem/grid_search_c5_lite_prior.py
rlem/freeze_c5_lite_prior.py
rlem/rerank_c5_lite_prior_one_shot.py

CODEX_C5_LITE_PRIOR_PHASED_INSTRUCTIONS.md
C5_LITE_PRIOR_EXPECTED_TEMPORAL_EXPORT_SCHEMA.md
```

## Key correction versus previous C5 code

The previous C5-lite sketch used row-level terms such as `R_video * Q_bd`.  That can change same-video span ordering, but it does not explicitly establish a temporal prior over every video.

The corrected implementation requires C5-0:

```text
per-(query, video) temporal prior export
```

If temporal arrays are missing, `check_c5_temporal_prior_inputs.py` and `build_c5_lite_prior_features.py` fail explicitly.  They do not fabricate P_ctx/P_b/P_e from row-level scores.

## Required temporal prior NPZ schema

The C5-0 export must produce one NPZ per split with:

```text
group_id: shape [G], exactly equal to C4 cache group_ids_sorted_unique
p_ctx: dense [G,T] or ragged flat [sum_T]
p_b:   dense [G,T] or ragged flat [sum_T]
p_e:   dense [G,T] or ragged flat [sum_T]
```

For ragged format add:

```text
temporal_offsets: shape [G+1]
```

Optional but recommended:

```text
clip_start_time
clip_end_time
```

If clip times are absent, aggregation falls back to `clip_length`, default 1.5 seconds.

## No official val during C5-lite-prior development

Only run official-val scripts after a separate freeze review and explicit one-shot authorization.

