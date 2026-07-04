# C12-4R-D Feature Repair Sample Audit

Status: `C12_FEATURE_REPAIR_NOT_READY`

No full strong-feature extraction was run in C12-4R.

```json
{
  "stage": "C12-4R-D",
  "status": "C12_FEATURE_REPAIR_NOT_READY",
  "official_val_used": false,
  "sample_query_count": 0,
  "sample_video_count": 0,
  "query_type_coverage": [
    "v",
    "t",
    "vt"
  ],
  "feature_candidates": {
    "Feature-A_stronger_subtitle_query_text": {
      "decision": "promising_need_sample_extraction",
      "why": "Current subtitle-only remains near-zero; stronger same-encoder text/subtitle embeddings are needed.",
      "estimated_dim": "768-1024",
      "timestamp_alignment": "preserved for subtitle clip features if extracted per 1.5s clip"
    },
    "Feature-B_stronger_visual_clip": {
      "decision": "highest_priority_for_v_query",
      "why": "Visual bridge improves v query relative to subtitle but remains far below reference; current visual mean is insufficient.",
      "estimated_dim": "512-1408 depending on CLIP/VideoMAE/InternVideo",
      "timestamp_alignment": "must preserve TVR 1.5s clip index or provide deterministic resampling map"
    },
    "Feature-C_multiscale_event": {
      "decision": "design_only",
      "why": "Do after retriever recall improves; event features will not solve corpus-level video recall alone."
    }
  },
  "storage_estimate": {
    "current_visual_mean_cache": "291MB",
    "full_clip_visual_stronger_feature_1024d_float16_estimate": "about 3.4GB for 17k videos x ~100 clips",
    "text_subtitle_1024d_float16_estimate": "about 3.4GB for subtitle clips plus query embeddings"
  },
  "extraction_time_estimate": "requires separate sample job; no full extraction in C12-4R"
}
```
