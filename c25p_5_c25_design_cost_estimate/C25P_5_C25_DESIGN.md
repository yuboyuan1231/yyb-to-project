# C25P-5 C25 Design

Plan A: frames available. Extract CLIP and DINOv2 frame features, optionally VideoMAE clip features, aggregate into TVR clip-level features, preserve first-stage top128, pilot rerank first, then train_fit full.

Plan B: raw videos available but frames missing. Use ffmpeg/decord to extract 3FPS or timestamp-driven frames, build frame manifest, then run Plan A.

Plan C: no raw/frames. Use only existing release-feature enhancement. This is not equivalent to CLIP/DINO/VideoMAE/InternVideo strong-feature extraction.
