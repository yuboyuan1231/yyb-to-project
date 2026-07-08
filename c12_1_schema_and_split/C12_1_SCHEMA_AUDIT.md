# C12-1 Schema Audit

Status: `schema_audit_pass = true`

The C12 schema is explicit about the C9 failure mode: `video_slot` and
`group_id` are different fields. C12 forbids using one as a fallback for the
other, forbids silent zero-fill, and requires keyed candidate joins.

Important candidate key:

```text
query_id + candidate_row_id + video_id + start + end
```

The old fixed-pool paths may be used only for teacher / hard-negative analysis.
C12 native inference must generate candidates with a C12 retriever and localizer.

```json
{
  "stage": "C12-1",
  "schema_version": "c12_native_vcmr_schema_v1",
  "schema_audit_pass": true,
  "current_promoted_system": "C7-B6 R1SelectiveTop1",
  "official_val_used": false,
  "required_fields": {
    "query_id": {
      "source": "desc_id",
      "type": "int",
      "required": true
    },
    "query_text": {
      "source": "desc",
      "type": "string",
      "required": true
    },
    "query_text_group": {
      "source": "sha256(normalized desc)[:16]",
      "type": "string",
      "required": true
    },
    "query_type": {
      "source": "type",
      "type": "enum[v,t,vt,unknown]",
      "required": true
    },
    "video_id": {
      "source": "vid_name / video2idx mapping",
      "type": "string_or_int",
      "required": true
    },
    "video_slot": {
      "source": "rank position within generated C12 topK video list",
      "type": "int",
      "required_for_candidates": true
    },
    "group_id": {
      "source": "stable video/query group id; never a fallback for video_slot",
      "type": "int_or_string",
      "required_for_grouped_split": true
    },
    "candidate_row_id": {
      "source": "monotonic per-query candidate identity",
      "type": "int",
      "required_for_candidates": true
    },
    "start": {
      "source": "seconds",
      "type": "float",
      "required_for_candidates": true
    },
    "end": {
      "source": "seconds",
      "type": "float",
      "required_for_candidates": true
    },
    "duration": {
      "source": "video metadata",
      "type": "float",
      "required": true
    },
    "clip_index": {
      "source": "floor/ceil time with clip_length=1.5",
      "type": "int_pair",
      "required_for_span_features": true
    },
    "raw_video_score": {
      "source": "C12 retriever output",
      "type": "float",
      "required_for_candidates": true
    },
    "raw_moment_score": {
      "source": "C12 localizer score",
      "type": "float",
      "required_for_candidates": true
    },
    "proposal_score": {
      "source": "C12 proposal quality head",
      "type": "float",
      "required_for_candidates": true
    },
    "boundary_features": {
      "source": "C12 localizer logits/probs/margins",
      "type": "array",
      "required_for_candidates": true
    },
    "subtitle_features": {
      "source": "subtitle encoder/cache",
      "type": "array",
      "required_if_available": true
    },
    "visual_features": {
      "source": "visual encoder/cache",
      "type": "array",
      "required_if_available": true
    },
    "rank_features": {
      "source": "derived ranks within C12 candidate set",
      "type": "array",
      "required_for_candidates": true
    }
  },
  "forbidden_patterns": [
    "train fallback group_id/100 while inference uses video_slot/100",
    "silent zero-fill for missing required fields",
    "duplicate candidate overwrite",
    "position-based join",
    "schema semantics mismatch across train/holdout/inference"
  ],
  "join_key_policy": [
    "query_id",
    "candidate_row_id",
    "video_id",
    "start",
    "end"
  ],
  "known_schema_risk_from_c9": {
    "feature10_train": "group_id/100 when video_slot absent",
    "feature10_official_runner": "video_slot/100",
    "c12_policy": "group_id and video_slot are distinct fields and cannot substitute for each other"
  }
}
```
