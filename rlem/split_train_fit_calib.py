#!/usr/bin/env python

"""Create a deterministic query-level C2 train_fit/train_calib split."""

import argparse
import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from rlem.io_utils import iter_jsonl, write_json


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence_jsonl", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--salt", default="conquer-rlem-c2-v1")
    parser.add_argument("--calib_modulus", type=int, default=10)
    parser.add_argument("--calib_bucket", type=int, default=0)
    parser.add_argument("--expected_rows_per_query", type=int, default=200)
    return parser.parse_args()


def split_for(desc_id, salt: str, modulus: int, calib_bucket: int) -> str:
    digest = hashlib.sha256(f"{salt}:{desc_id}".encode("utf-8")).digest()
    bucket = int.from_bytes(digest[:8], "big") % modulus
    return "train_calib" if bucket == calib_bucket else "train_fit"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    split_ids = {"train_fit": [], "train_calib": []}
    split_rows = Counter()
    label_counts = {name: Counter() for name in split_ids}
    current_id = None
    current_split = None
    current_rows = 0
    seen = set()

    def close_query():
        if current_id is None:
            return
        if current_rows != args.expected_rows_per_query:
            raise ValueError(
                f"desc_id={current_id} has {current_rows} rows; "
                f"expected {args.expected_rows_per_query}"
            )

    for row in iter_jsonl(args.evidence_jsonl):
        desc_id = row["desc_id"]
        if current_id is None or desc_id != current_id:
            close_query()
            if desc_id in seen:
                raise ValueError(f"Non-contiguous duplicate desc_id={desc_id}")
            seen.add(desc_id)
            current_id = desc_id
            current_split = split_for(
                desc_id, args.salt, args.calib_modulus, args.calib_bucket
            )
            split_ids[current_split].append(str(desc_id))
            current_rows = 0
        current_rows += 1
        split_rows[current_split] += 1
        for key in ["is_gt_video", "m_bd", "y_joint_05", "y_joint_07", "y_fp"]:
            label_counts[current_split][f"{key}_positive"] += int(float(row[key]) > 0)
        label_counts[current_split]["rows"] += 1
    close_query()

    id_paths = {}
    for name, ids in split_ids.items():
        path = output_dir / f"{name}_desc_ids.txt"
        with path.open("w", encoding="utf-8") as f:
            for desc_id in ids:
                f.write(desc_id + "\n")
        id_paths[name] = {
            "path": str(path),
            "sha256": sha256_file(path),
            "queries": len(ids),
            "rows": split_rows[name],
            "label_counts": dict(label_counts[name]),
        }

    fit_set = set(split_ids["train_fit"])
    calib_set = set(split_ids["train_calib"])
    summary = {
        "status": "PASS",
        "method": "sha256_prefix_u64_modulo",
        "salt": args.salt,
        "calib_modulus": args.calib_modulus,
        "calib_bucket": args.calib_bucket,
        "expected_rows_per_query": args.expected_rows_per_query,
        "source_evidence": os.path.abspath(args.evidence_jsonl),
        "total_queries": len(seen),
        "total_rows": sum(split_rows.values()),
        "overlap_queries": len(fit_set & calib_set),
        "union_queries": len(fit_set | calib_set),
        "splits": id_paths,
    }
    write_json(str(output_dir / "split_summary.json"), summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
