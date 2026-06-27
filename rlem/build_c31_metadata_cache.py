#!/usr/bin/env python

"""Build reusable train_calib candidate metadata for C3.1 score searches."""

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
if "bool" not in np.__dict__:
    np.bool = np.bool_
from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from rlem.io_utils import iter_jsonl, read_json, write_json
from standalone_eval.eval import compute_temporal_iou_batch, load_jsonl


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence_jsonl", required=True)
    parser.add_argument("--desc_ids", required=True)
    parser.add_argument("--gt_jsonl", required=True)
    parser.add_argument("--dataset_config", required=True)
    parser.add_argument("--split", choices=["train"], required=True)
    parser.add_argument("--output_npz", required=True)
    parser.add_argument("--output_manifest", required=True)
    parser.add_argument("--rows_per_query", type=int, default=200)
    return parser.parse_args()


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    args = parse_args()
    allowed = {line.strip() for line in open(args.desc_ids, encoding="utf-8") if line.strip()}
    n_queries = len(allowed)
    n_rows = n_queries * args.rows_per_query
    arrays = {
        "base_log": np.empty(n_rows, dtype=np.float64),
        "s_base": np.empty(n_rows, dtype=np.float64),
        "video_idx": np.empty(n_rows, dtype=np.int32),
        "start": np.empty(n_rows, dtype=np.float32),
        "end": np.empty(n_rows, dtype=np.float32),
        "rank_base": np.empty(n_rows, dtype=np.int16),
        "is_gt": np.empty(n_rows, dtype=np.bool_),
        "y05_export": np.empty(n_rows, dtype=np.bool_),
        "y07_export": np.empty(n_rows, dtype=np.bool_),
        "y_fp": np.empty(n_rows, dtype=np.bool_),
    }
    desc_ids = []
    descs = []
    current = None
    current_count = 0
    offset = 0
    for row in tqdm(iter_jsonl(args.evidence_jsonl), total=n_rows, desc="C3.1 metadata"):
        sid = str(row["desc_id"])
        if sid not in allowed:
            raise ValueError(f"Unexpected desc_id={sid}")
        if current is None or sid != current:
            if current is not None and current_count != args.rows_per_query:
                raise ValueError(f"desc_id={current} rows={current_count}")
            current = sid
            current_count = 0
            desc_ids.append(row["desc_id"])
            descs.append(row.get("desc", ""))
        current_count += 1
        s_base = float(row["s_base"])
        if s_base <= 0 or not math.isfinite(s_base):
            raise ValueError(f"Invalid s_base={s_base}")
        values = {
            "base_log": math.log(s_base), "s_base": s_base,
            "video_idx": int(row["video_idx"]), "start": float(row["start_time"]),
            "end": float(row["end_time"]), "rank_base": int(row["rank_base"]),
            "is_gt": bool(row["is_gt_video"]), "y05_export": bool(row["y_joint_05"]),
            "y07_export": bool(row["y_joint_07"]), "y_fp": bool(row["y_fp"]),
        }
        for key, value in values.items():
            arrays[key][offset] = value
        offset += 1
    if offset != n_rows or current_count != args.rows_per_query or len(desc_ids) != n_queries:
        raise ValueError(f"Cardinality mismatch rows={offset}, queries={len(desc_ids)}")
    shape = (n_queries, args.rows_per_query)
    arrays = {key: value.reshape(shape) for key, value in arrays.items()}
    gt = {str(item["desc_id"]): item for item in load_jsonl(args.gt_jsonl)}
    cfg = read_json(args.dataset_config)
    duration_path = cfg["video_duration_idx_path"]
    if not os.path.isabs(duration_path):
        duration_path = os.path.join(cfg["root_path"], duration_path)
    video2idx = {name: int(value[1]) for name, value in read_json(duration_path)[args.split].items()}
    eval_y05 = np.zeros(shape, dtype=np.bool_)
    eval_y07 = np.zeros(shape, dtype=np.bool_)
    for qi, desc_id in enumerate(desc_ids):
        item = gt[str(desc_id)]
        match = arrays["video_idx"][qi] == video2idx[item["vid_name"]]
        predictions = np.stack([arrays["start"][qi], arrays["end"][qi]], axis=1)
        timestamp = np.asarray(item["ts"], dtype=np.float32)
        iou = compute_temporal_iou_batch(predictions, timestamp) * match
        eval_y05[qi] = iou >= 0.5
        eval_y07[qi] = iou >= 0.7
    arrays["eval_y05"] = eval_y05
    arrays["eval_y07"] = eval_y07
    arrays["desc_ids"] = np.asarray(desc_ids, dtype=np.int64)
    partial = str(args.output_npz) + ".partial"
    Path(args.output_npz).parent.mkdir(parents=True, exist_ok=True)
    with open(partial, "wb") as f:
        np.savez(f, **arrays)
    os.replace(partial, args.output_npz)
    manifest = {
        "status": "PASS", "scope": "train_calib_only", "official_val_used": False,
        "evidence": args.evidence_jsonl, "evidence_sha256": sha256_file(args.evidence_jsonl),
        "desc_ids_sha256": sha256_file(args.desc_ids), "gt_sha256": sha256_file(args.gt_jsonl),
        "queries": n_queries, "rows": n_rows, "rows_per_query": args.rows_per_query,
        "y05_evaluator_disagreements_vs_export": int(np.sum(eval_y05 != arrays["y05_export"])),
        "y07_evaluator_disagreements_vs_export": int(np.sum(eval_y07 != arrays["y07_export"])),
        "desc_text_path": str(Path(args.output_npz).with_suffix(".descs.json")),
    }
    write_json(Path(args.output_npz).with_suffix(".descs.json"), {
        "desc_ids": desc_ids, "descs": descs,
    })
    write_json(args.output_manifest, manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
