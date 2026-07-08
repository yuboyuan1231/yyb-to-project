#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frame_features", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--method", default="mean")
    parser.add_argument("--dry_run", action="store_true", default=True)
    args = parser.parse_args()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    result = {"dry_run": args.dry_run, "method": args.method, "full_aggregation_run": False, "status": "dry_run_ok"}
    Path(args.output).write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()

