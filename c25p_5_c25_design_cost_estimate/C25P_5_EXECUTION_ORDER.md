# C25P-5 Execution Order

1. Confirm TVQA/TVR data authorization and frame/raw availability.
2. Build canonical frame manifest.
3. Run tiny CLIP/DINOv2 pilot under `/tmp/c25p_feature_cache/CONQUER-RLEM-c2c3`.
4. Validate annotation/top128/frame alignment.
5. Scale extraction by video shards only after C24H completion or resource clearance.
6. Integrate features into first-stage-preserving top128 rerank pilot.
