#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=int, required=True)
    parser.add_argument("--feature_dim", type=int, default=512)
    args = parser.parse_args()
    result = {
        "frames": args.frames,
        "feature_dim": args.feature_dim,
        "float32_gb": args.frames * args.feature_dim * 4 / (1024 ** 3),
        "float16_gb": args.frames * args.feature_dim * 2 / (1024 ** 3),
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

