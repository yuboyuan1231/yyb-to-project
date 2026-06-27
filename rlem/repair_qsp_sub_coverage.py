#!/usr/bin/env python3

"""Repair only qsp_sub_coverage from raw train subtitle support.

The input JSONL is preserved byte-for-byte except for the numeric value of the
qsp_sub_coverage field. Candidate identities, labels, and every other QSP field
remain untouched. Output is written atomically through a .partial file.
"""

import argparse
import fcntl
import gzip
import hashlib
import io
import json
import math
import os
import re
from functools import lru_cache
from pathlib import Path

import lmdb
import numpy as np


COVERAGE_PATTERN = re.compile(
    r'("qsp_sub_coverage"\s*:\s*)'
    r'[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?'
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_jsonl", required=True)
    parser.add_argument("--output_jsonl", required=True)
    parser.add_argument("--subtitle_lmdb", required=True)
    parser.add_argument("--audit_json", required=True)
    parser.add_argument("--coverage_eps", type=float, default=1e-8)
    parser.add_argument("--cache_size", type=int, default=50000)
    parser.add_argument("--max_ctx_len", type=int, default=100)
    parser.add_argument("--expected_input_sha256", default=None)
    parser.add_argument("--expected_rows", type=int, default=None)
    parser.add_argument("--max_rows", type=int, default=None)
    return parser.parse_args()


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def update_moments(state, value):
    state["count"] += 1
    delta = value - state["mean"]
    state["mean"] += delta / state["count"]
    state["m2"] += delta * (value - state["mean"])
    state["min"] = min(state["min"], value)
    state["max"] = max(state["max"], value)


def finish_moments(state):
    count = state["count"]
    return {
        "count": count,
        "mean": state["mean"] if count else None,
        "std": math.sqrt(max(0.0, state["m2"] / count)) if count else None,
        "min": state["min"] if count else None,
        "max": state["max"] if count else None,
    }


def main():
    args = parse_args()
    input_path = Path(args.input_jsonl).resolve()
    output_path = Path(args.output_jsonl).resolve()
    if input_path == output_path:
        raise ValueError("coverage repair refuses to overwrite its input")
    if "val" in input_path.name.lower():
        raise ValueError("coverage repair is train-only and refuses val input")
    input_sha256 = sha256_file(input_path)
    if args.expected_input_sha256 and input_sha256 != args.expected_input_sha256:
        raise ValueError(
            f"input SHA256 mismatch: actual={input_sha256} expected={args.expected_input_sha256}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    lock_file = open(str(output_path) + ".lock", "w", encoding="utf-8")
    try:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise RuntimeError(f"another repair owns output lock: {output_path}.lock") from exc
    lock_file.write(f"pid={os.getpid()}\n")
    lock_file.flush()

    env = lmdb.open(
        str(Path(args.subtitle_lmdb).resolve()), readonly=True, create=False,
        readahead=False, max_readers=4096,
    )
    txn = env.begin(buffers=True)

    @lru_cache(maxsize=args.cache_size)
    def subtitle_support(video_name):
        value = txn.get(video_name.encode())
        if value is None:
            raise KeyError(f"subtitle feature missing for video {video_name}")
        with io.BytesIO(bytes(value)) as reader:
            features = np.load(reader, allow_pickle=True)["features"][:args.max_ctx_len]
        return (
            np.linalg.norm(features.astype(np.float32, copy=False), axis=-1) > args.coverage_eps
        ).astype(np.float32)

    partial = (
        Path(str(output_path)[:-3] + ".partial.gz")
        if str(output_path).endswith(".gz") else Path(str(output_path) + ".partial")
    )
    old_stats = {"count": 0, "mean": 0.0, "m2": 0.0, "min": float("inf"), "max": float("-inf")}
    new_stats = {"count": 0, "mean": 0.0, "m2": 0.0, "min": float("inf"), "max": float("-inf")}
    rows = changed_rows = zero_rows = full_rows = 0

    with gzip.open(input_path, "rt", encoding="utf-8") as source, gzip.open(partial, "wt", encoding="utf-8") as target:
        for line in source:
            if not line.strip():
                continue
            row = json.loads(line)
            old_value = float(row["qsp_sub_coverage"])
            support = subtitle_support(str(row["video_name"]))
            start = int(row["start_idx"])
            end = int(row["end_idx"])
            if not 0 <= start <= end < len(support):
                raise ValueError(
                    f"candidate span outside raw subtitle timeline: video={row['video_name']} "
                    f"span=({start},{end}) subtitle_len={len(support)}"
                )
            if end >= start:
                support_sum = support[start:end + 1].sum(dtype=np.float32)
                new_value = float(support_sum / np.float32(end - start + 1))
            else:
                new_value = 0.0
            if not math.isfinite(new_value) or not 0.0 <= new_value <= 1.0:
                raise ValueError(f"invalid repaired coverage {new_value}")
            replacement = r"\g<1>" + repr(new_value)
            repaired, replacements = COVERAGE_PATTERN.subn(replacement, line, count=1)
            if replacements != 1:
                raise ValueError("expected exactly one qsp_sub_coverage field per row")
            target.write(repaired)
            rows += 1
            changed_rows += int(new_value != old_value)
            zero_rows += int(new_value == 0.0)
            full_rows += int(new_value == 1.0)
            update_moments(old_stats, old_value)
            update_moments(new_stats, new_value)
            if args.max_rows is not None and rows >= args.max_rows:
                break

    if args.expected_rows is not None and rows != args.expected_rows:
        raise ValueError(f"row count mismatch: repaired={rows} expected={args.expected_rows}")
    os.replace(partial, output_path)
    audit = {
        "status": "PASS",
        "stage": "C3.5 QSP subtitle coverage semantic repair",
        "input_jsonl": str(input_path),
        "input_sha256": input_sha256,
        "output_jsonl": str(output_path),
        "rows": rows,
        "changed_rows": changed_rows,
        "zero_coverage_rows": zero_rows,
        "full_coverage_rows": full_rows,
        "coverage_definition": "mean(raw subtitle row L2 norm > 1e-8) over inclusive candidate span",
        "coverage_eps": args.coverage_eps,
        "max_ctx_len": args.max_ctx_len,
        "subtitle_lmdb": str(Path(args.subtitle_lmdb).resolve()),
        "subtitle_lmdb_entries": int(env.stat()["entries"]),
        "old_coverage": finish_moments(old_stats),
        "new_coverage": finish_moments(new_stats),
        "cache_info": str(subtitle_support.cache_info()),
        "candidate_or_other_feature_mutations": 0,
        "official_val_used": False,
        "output_sha256": sha256_file(output_path),
    }
    Path(args.audit_json).parent.mkdir(parents=True, exist_ok=True)
    audit_partial = str(args.audit_json) + ".partial"
    with open(audit_partial, "w", encoding="utf-8") as file:
        json.dump(audit, file, indent=2)
    os.replace(audit_partial, args.audit_json)
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
