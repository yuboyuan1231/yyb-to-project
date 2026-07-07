#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--max_videos", type=int, default=2)
    parser.add_argument("--max_frames_per_video", type=int, default=16)
    parser.add_argument("--dry_run", action="store_true", default=True)
    args = parser.parse_args()
    data = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    rows = data.get("rows", [])[: args.max_videos]
    result = {
        "dry_run": args.dry_run,
        "videos_checked": len(rows),
        "max_frames_per_video": min(args.max_frames_per_video, 16),
        "decode_attempted": False,
        "status": "dry_run_ok",
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

