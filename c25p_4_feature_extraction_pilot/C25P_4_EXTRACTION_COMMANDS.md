# C25P-4 Extraction Commands

Dry-run manifest:

```bash
python tools/c25p/build_frame_manifest.py --input /path/to/frames --output /tmp/c25p_feature_cache/CONQUER-RLEM-c2c3/frame_manifest.json --max_videos 2
python tools/c25p/probe_frame_decode.py --manifest /tmp/c25p_feature_cache/CONQUER-RLEM-c2c3/frame_manifest.json --max_videos 2 --max_frames_per_video 16
python tools/c25p/extract_clip_frame_features.py --manifest /tmp/c25p_feature_cache/CONQUER-RLEM-c2c3/frame_manifest.json --output /tmp/c25p_feature_cache/CONQUER-RLEM-c2c3/clip_pilot --dry_run --max_videos 2 --max_frames_per_video 16
python tools/c25p/validate_feature_alignment.py --manifest /tmp/c25p_feature_cache/CONQUER-RLEM-c2c3/frame_manifest.json --features /tmp/c25p_feature_cache/CONQUER-RLEM-c2c3/clip_pilot --dry_run
```
