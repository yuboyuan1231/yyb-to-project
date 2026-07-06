# Raw/Frame/Strong Feature Packet

- Raw videos: False
- Frames: False
- Subtitles: False
- TVR annotations: True
- First-stage top128: True
- CLIP feasibility: {'available': False, 'expected_speed': 'fast frame-level', 'expected_storage': 'frames * dim * dtype', 'feature_dim': '512 or 768', 'model': 'CLIP ViT-B/32 or ViT-L/14'}
- DINOv2 feasibility: {'available': False, 'expected_speed': 'medium frame-level', 'expected_storage': 'frames * dim * dtype', 'feature_dim': '768 or 1024', 'model': 'DINOv2 ViT-B/14 or ViT-L/14'}
- VideoMAE feasibility: {'available': False, 'expected_speed': 'slower clip-level', 'expected_storage': 'clips * dim * dtype', 'feature_dim': '768 typical', 'model': 'VideoMAE/VideoMAEv2'}
- InternVideo feasibility: {'available': False, 'expected_speed': 'heavy video-language', 'expected_storage': 'clips * dim * dtype', 'feature_dim': '768-1024 typical', 'model': 'InternVideo/InternVideo2'}
- Official run: false
- Official pool read: false
- C24H artifacts touched: false
