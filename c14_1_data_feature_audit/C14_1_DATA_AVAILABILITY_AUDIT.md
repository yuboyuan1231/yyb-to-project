# C14-1 Data Availability Audit

status = `C14_EVENT_FEATURE_FEASIBLE_FROM_EXISTING_FEATURES`

- raw_video_exists: `False`
- strong_visual_status: `C14_STRONG_VISUAL_EXTRACTION_BLOCKED_BY_RAW_VIDEO_ABSENCE`
- query_count: `86784`
- video_count: `17435`
- query_feature_dim: `768`
- subtitle_feature_dim: `768`
- visual_feature_dim: `4352`
- sample_count: `300`
- positive_not_in_b6_top100_blocked: `True`

If raw video is absent, C14 records
`C14_STRONG_VISUAL_EXTRACTION_BLOCKED_BY_RAW_VIDEO_ABSENCE` and does not fake
CLIP/SigLIP/DINOv2/VideoMAE/InternVideo extraction.

official was not run.
