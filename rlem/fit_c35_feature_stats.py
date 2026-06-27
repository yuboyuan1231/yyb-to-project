#!/usr/bin/env python3

"""Fit C3.5 feature normalization on frozen train_fit IDs only."""

import argparse
import json
import math
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from rlem.evidence_dataset import FeatureStats
from rlem.feature_schema import C35_MATCHED_FEATURE_SETS


def load_ids(path):
    with open(path, "r", encoding="utf-8") as file:
        return {line.strip() for line in file if line.strip()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence_jsonl", required=True)
    parser.add_argument("--fit_desc_ids", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--schema", choices=sorted(C35_MATCHED_FEATURE_SETS), default="qsp53_with_coverage")
    parser.add_argument("--expected_rows", type=int, required=True)
    args = parser.parse_args()
    if "val" in Path(args.evidence_jsonl).name.lower():
        raise ValueError("C3.5 stats fitting is train-only")
    feature_names = C35_MATCHED_FEATURE_SETS[args.schema]
    stats = FeatureStats.fit(
        [args.evidence_jsonl], feature_names,
        allowed_desc_ids=load_ids(args.fit_desc_ids),
    )
    if stats.count != args.expected_rows:
        raise ValueError(f"stats row mismatch: {stats.count} != {args.expected_rows}")
    if not all(math.isfinite(value) for value in stats.mean + stats.std):
        raise ValueError("non-finite feature statistics")
    payload = stats.to_dict() | {
        "status": "PASS",
        "schema": args.schema,
        "source_evidence": str(Path(args.evidence_jsonl).resolve()),
        "fit_desc_ids": str(Path(args.fit_desc_ids).resolve()),
        "official_val_used": False,
    }
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = str(output) + ".partial"
    with open(partial, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)
    os.replace(partial, output)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
