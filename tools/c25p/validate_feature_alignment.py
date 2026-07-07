#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--features", required=True)
    parser.add_argument("--dry_run", action="store_true", default=True)
    args = parser.parse_args()
    manifest = Path(args.manifest)
    result = {
        "dry_run": args.dry_run,
        "manifest_exists": manifest.exists(),
        "features_path": args.features,
        "official_used": False,
        "status": "dry_run_ok" if manifest.exists() else "missing_manifest",
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

