# C14 Instruction Review

The C14 instruction is directionally correct and was executed on branch
`c14-strong-feature-pilot`.

Three execution clarifications were required:

1. `positive not in B6 top100` is not naturally available in the current
   CONQUER/B6 replay caches. Both checked first-stage caches have 100% train-only
   top100 coverage, so C14 records this as blocked and uses native C12 retriever
   missed-top100 aggregate evidence as the proxy.
2. Raw TVR video files are not visible under the local TVR data tree. C14
   therefore records
   `C14_STRONG_VISUAL_EXTRACTION_BLOCKED_BY_RAW_VIDEO_ABSENCE` and does not fake
   CLIP/SigLIP/DINOv2/VideoMAE/InternVideo extraction.
3. The C12 M1000 span pool is not a stable full materialized artifact in this
   workspace. C14 reuses the same C12 localizer and feature builder to generate
   sample candidate pools, records schema hashes, and writes parquet feature
   outputs for the sampled candidates.

No official validation was run.
