# C13-1 Visual Feature Plan

Decision: strong visual features are worth a pilot, but not full extraction yet.

Plan:

1. Extract CLIP/SigLIP/EVA-CLIP style frame or short-clip embeddings for a small video sample.
2. Optionally compare DINOv2 frame descriptors when video-language models are unavailable.
3. Record frame timestamp, clip window, video id, and normalized time index.
4. Compare VR recall sample and span-ranking sample against C12 current visual bridge.

Expected value: strongest for v queries and for B6 top1 wrong cases.

Risk level: medium_to_high because decoding and dense storage dominate cost.

official_val_used = false
