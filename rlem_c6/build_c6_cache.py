#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from rlem_c6.c6a_utils import build_c6_label_cache


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--split_name", choices=["train_fit", "train_calib"], required=True)
    p.add_argument("--evidence_jsonl", required=True)
    p.add_argument("--desc_ids_filter", required=True)
    p.add_argument("--c5_features_npz", required=True)
    p.add_argument("--temporal_prior_npz", required=True)
    p.add_argument("--video_dataset_npz", required=True)
    p.add_argument("--output_npz", required=True)
    p.add_argument("--manifest_json", required=True)
    args = p.parse_args()
    manifest = build_c6_label_cache(**vars(args))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
