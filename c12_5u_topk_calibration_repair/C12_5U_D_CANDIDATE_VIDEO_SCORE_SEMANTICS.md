# C12-5U-D Candidate-Video Score Semantics

status = C12_5U_CANDIDATE_VIDEO_SCORE_SEMANTICS_PASS

`retriever_score_z` uses candidate-video first-stage score. In GT-video-only oracle evaluation, candidate video = GT video.

No `fs_score_for_gt` feature builder is used in the 5U inference path. Training and inference share `span_feature_row_from_context` and `candidate_video_score_z`.

feature_schema_hash = e58877d5b7fa0fa051b09dd2ec7a7341d603ebcfe9dd821a9a296a88c18f5d02

official_val_used = false
