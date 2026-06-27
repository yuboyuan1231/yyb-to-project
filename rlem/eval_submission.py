#!/usr/bin/env python

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

"""Evaluate a RLEM/CONQUER submission JSON with the bundled TVR evaluator."""

import argparse
import json
from pathlib import Path

import numpy as np
if not hasattr(np, "bool"):
    np.bool = np.bool_

from standalone_eval.eval import eval_retrieval, load_json, load_jsonl


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--submission_json", required=True)
    parser.add_argument("--gt_jsonl", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--not_verbose", action="store_true")
    parser.add_argument("--no_desc_type", action="store_true", help="Use this for datasets without TVR desc type field")
    return parser.parse_args()


def main():
    args = parse_args()
    submission = load_json(args.submission_json)
    gt = load_jsonl(args.gt_jsonl)
    metrics = eval_retrieval(
        submission,
        gt,
        iou_thds=(0.5, 0.7),
        verbose=not args.not_verbose,
        match_number=True,
        use_desc_type=not args.no_desc_type,
    )
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
