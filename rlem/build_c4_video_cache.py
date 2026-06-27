#!/usr/bin/env python
"""Build the fixed-candidate C4 cache with bounded, preallocated memory."""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from rlem.c4_lite_utils import iter_query_groups, load_desc_ids, write_json  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--scored_jsonl", required=True)
    p.add_argument("--desc_ids_filter", required=True)
    p.add_argument("--output_npz", required=True)
    p.add_argument("--manifest_json", required=True)
    p.add_argument("--expected_rows_per_query", type=int, default=200)
    p.add_argument("--split", choices=["train", "val"], default="train")
    p.add_argument("--allow_official_val", action="store_true")
    p.add_argument("--uncompressed", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    if args.split == "val" and not args.allow_official_val:
        raise ValueError("Refusing official val C4 cache without --allow_official_val")
    allowed = load_desc_ids(args.desc_ids_filter)
    max_queries = len(allowed)
    max_rows = max_queries * args.expected_rows_per_query
    specs = {
        "query_index": np.int32, "video_idx": np.int32, "row_group_id": np.int32,
        "rank_base": np.int32, "start_time": np.float32, "end_time": np.float32,
        "s_base": np.float32, "base_log": np.float32, "log_r1": np.float32,
        "log_boundary": np.float32, "q_joint": np.float32, "q_bd": np.float32,
        "e_fp": np.float32, "s_c31": np.float32, "iou": np.float32,
        "y_fp": np.float32, "y_joint_05": np.float32, "is_gt_video": np.float32,
    }
    arrays = {key: np.empty(max_rows, dtype=dtype) for key, dtype in specs.items()}
    desc_ids = np.empty(max_queries, dtype=object)
    desc_text = np.empty(max_queries, dtype=object)
    desc_offsets = np.empty(max_queries + 1, dtype=np.int64)
    desc_offsets[0] = 0
    video_group_keys = []
    observed_ids = set()
    row_index = bad_counts = duplicate_candidates = 0
    query_count = 0

    groups = iter_query_groups(args.scored_jsonl, allowed)
    for qid, group in enumerate(tqdm(groups, total=max_queries, desc="build C4 cache")):
        if qid >= max_queries:
            raise ValueError("More query groups than the desc-id filter")
        first = group[0]
        desc_id = str(first["desc_id"])
        observed_ids.add(desc_id)
        desc_ids[qid] = first["desc_id"]
        desc_text[qid] = first.get("desc", "")
        if len(group) != args.expected_rows_per_query:
            bad_counts += 1
        end_index = row_index + len(group)
        if end_index > max_rows:
            raise ValueError("Candidate rows exceed preallocated protocol bound")
        arrays["query_index"][row_index:end_index] = qid
        seen = set()
        local_video_to_group = {}
        for row in group:
            key = (row.get("video_idx"), row.get("start_time"), row.get("end_time"))
            duplicate_candidates += int(key in seen)
            seen.add(key)
            vid = int(row["video_idx"])
            if vid not in local_video_to_group:
                local_video_to_group[vid] = len(video_group_keys)
                video_group_keys.append((qid, vid))
            arrays["video_idx"][row_index] = vid
            arrays["row_group_id"][row_index] = local_video_to_group[vid]
            arrays["rank_base"][row_index] = int(row.get("rank_base", row_index))
            arrays["start_time"][row_index] = float(row["start_time"])
            arrays["end_time"][row_index] = float(row["end_time"])
            arrays["s_base"][row_index] = float(row["s_base"])
            arrays["base_log"][row_index] = float(row["base_log"])
            arrays["log_r1"][row_index] = float(row["log_r1"])
            arrays["log_boundary"][row_index] = float(row["log_boundary"])
            arrays["q_joint"][row_index] = float(row["q_joint_cal"])
            arrays["q_bd"][row_index] = float(row["q_bd_cal"])
            arrays["e_fp"][row_index] = float(row["e_fp_cal"])
            arrays["s_c31"][row_index] = float(row["s_c31"])
            arrays["iou"][row_index] = float(row.get("iou", -1.0) if row.get("iou") is not None else -1.0)
            arrays["y_fp"][row_index] = float(row.get("y_fp", 0.0) or 0.0)
            arrays["y_joint_05"][row_index] = float(row.get("y_joint_05", 0.0) or 0.0)
            arrays["is_gt_video"][row_index] = float(row.get("is_gt_video", 0.0) or 0.0)
            row_index += 1
        query_count = qid + 1
        desc_offsets[query_count] = row_index

    missing_ids = allowed - observed_ids
    extra_ids = observed_ids - allowed
    arrays = {key: value[:row_index] for key, value in arrays.items()}
    arrays["desc_offsets"] = desc_offsets[:query_count + 1]
    arrays["desc_ids"] = desc_ids[:query_count]
    arrays["desc_text"] = desc_text[:query_count]
    arrays["video_group_keys"] = np.asarray(video_group_keys, dtype=np.int64)
    nonfinite = sum(int((~np.isfinite(value)).sum()) for value in arrays.values() if value.dtype.kind in "fc")
    sort_idx = np.argsort(arrays["row_group_id"], kind="stable")
    sorted_groups = arrays["row_group_id"][sort_idx]
    starts = np.r_[0, np.flatnonzero(sorted_groups[1:] != sorted_groups[:-1]) + 1]
    arrays["group_sort_idx"] = sort_idx.astype(np.int64)
    arrays["group_offsets"] = starts.astype(np.int64)
    arrays["group_ids_sorted_unique"] = sorted_groups[starts].astype(np.int32)

    output = Path(args.output_npz)
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = Path(str(output) + ".partial.npz")
    saver = np.savez if args.uncompressed else np.savez_compressed
    saver(partial, **arrays)
    os.replace(partial, output)
    manifest = {
        "status": "PASS" if not (bad_counts or duplicate_candidates or nonfinite or missing_ids or extra_ids) else "FAIL",
        "scored_jsonl": args.scored_jsonl, "output_npz": args.output_npz,
        "queries": query_count, "rows": row_index,
        "rows_per_query_expected": args.expected_rows_per_query,
        "bad_query_counts": bad_counts, "duplicate_candidates": duplicate_candidates,
        "nonfinite_values": nonfinite, "missing_filtered_queries": len(missing_ids),
        "extra_filtered_queries": len(extra_ids), "video_groups": len(video_group_keys),
        "preallocated_arrays": True, "compressed": not args.uncompressed,
        "official_val_used": bool(args.split == "val" and args.allow_official_val),
    }
    write_json(args.manifest_json, manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
