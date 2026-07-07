#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

FRAME_EXTS = {".jpg", ".jpeg", ".png"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--fps", type=float, default=3.0)
    parser.add_argument("--max_videos", type=int, default=2)
    args = parser.parse_args()
    root = Path(args.input)
    rows = []
    if root.exists():
        for entry in sorted(os.scandir(root), key=lambda x: x.name)[: args.max_videos]:
            p = Path(entry.path)
            if p.is_dir():
                frames = [x for x in sorted(p.iterdir()) if x.suffix.lower() in FRAME_EXTS]
                rows.append({
                    "video_id": p.name,
                    "raw_video_path": None,
                    "frame_dir": str(p),
                    "fps": args.fps,
                    "frame_count": len(frames),
                    "duration_sec": len(frames) / args.fps if args.fps else None,
                    "timestamp_unit": "seconds",
                    "frame_index_start": 0,
                    "frame_index_end": max(0, len(frames) - 1),
                    "subtitle_path": None,
                    "split_coverage": [],
                    "feature_ready": bool(frames),
                    "missing_reason": "" if frames else "no_frames",
                })
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps({"rows": rows}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"rows": len(rows), "output": args.output}, ensure_ascii=False))


if __name__ == "__main__":
    main()

