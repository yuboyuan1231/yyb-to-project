from __future__ import annotations

import argparse
import json
from pathlib import Path


def dry_feature_main(model_name: str, feature_dim: int) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dry_run", action="store_true", default=True)
    parser.add_argument("--max_videos", type=int, default=2)
    parser.add_argument("--max_frames_per_video", type=int, default=16)
    parser.add_argument("--no_gpu", action="store_true")
    parser.add_argument("--allow_gpu_pilot", action="store_true")
    args = parser.parse_args()
    Path(args.output).mkdir(parents=True, exist_ok=True)
    result = {
        "model_name": model_name,
        "feature_dim": feature_dim,
        "dry_run": args.dry_run,
        "gpu_allowed": args.allow_gpu_pilot and not args.no_gpu,
        "full_extraction_run": False,
        "max_videos": min(args.max_videos, 2),
        "max_frames_per_video": min(args.max_frames_per_video, 16),
        "status": "dry_run_ok",
    }
    (Path(args.output) / "pilot_result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))

