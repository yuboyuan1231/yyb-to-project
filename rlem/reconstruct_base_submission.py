#!/usr/bin/env python

"""Reconstruct a fixed-candidate CONQUER VCMR submission from C1 evidence.

This is a C1 integrity gate.  It uses only ``s_base`` and the frozen CONQUER
NMS implementation; no learned RLEM score is involved.
"""

import argparse
import gzip
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from rlem.io_utils import iter_jsonl, read_json, write_json
from utils.inference_utils import filter_vcmr_by_nms


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence_jsonl", required=True)
    parser.add_argument("--dataset_config", required=True)
    parser.add_argument("--split", default="val", choices=["train", "val", "test_public"])
    parser.add_argument("--output_json", required=True)
    parser.add_argument(
        "--desc_ids_filter",
        default=None,
        help="Optional newline-delimited desc_id allowlist (for train_calib-only reconstruction)",
    )
    parser.add_argument(
        "--save_filtered_evidence",
        default=None,
        help="Optional .jsonl.gz derivative containing exactly the selected evidence rows",
    )
    parser.add_argument(
        "--save_gt_jsonl",
        default=None,
        help="Optional evaluator GT JSONL derived only from selected evidence labels",
    )
    parser.add_argument(
        "--max_before_nms",
        "--effective_top_n",
        dest="max_before_nms",
        type=int,
        default=100,
        help="Effective official limit: original inference truncates to top-100 before NMS",
    )
    parser.add_argument("--max_after_nms", type=int, default=100)
    parser.add_argument("--nms_thd", type=float, default=0.7)
    parser.add_argument("--no_nms", action="store_true")
    return parser.parse_args()


def load_desc_ids(path: Optional[str]) -> Optional[Set[str]]:
    if not path:
        return None
    with open(path, "r", encoding="utf-8") as f:
        ids = {line.strip() for line in f if line.strip()}
    if not ids:
        raise ValueError(f"No desc_ids found in {path}")
    return ids


def selected_rows(
    evidence_jsonl: str,
    allowed_ids: Optional[Set[str]],
    save_filtered_evidence: Optional[str],
):
    writer = None
    partial = None
    if save_filtered_evidence:
        if not str(save_filtered_evidence).endswith(".jsonl.gz"):
            raise ValueError("--save_filtered_evidence must end in .jsonl.gz")
        Path(save_filtered_evidence).parent.mkdir(parents=True, exist_ok=True)
        partial = str(save_filtered_evidence) + ".partial.gz"
        writer = gzip.open(partial, "wt", encoding="utf-8", compresslevel=6)
    try:
        for row in iter_jsonl(evidence_jsonl):
            if allowed_ids is not None and str(row.get("desc_id")) not in allowed_ids:
                continue
            if writer is not None:
                writer.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            yield row
    except BaseException:
        if writer is not None:
            writer.close()
        if partial and os.path.exists(partial):
            os.remove(partial)
        raise
    else:
        if writer is not None:
            writer.close()
            os.replace(partial, save_filtered_evidence)


def load_video2idx(dataset_config: str, split: str) -> Dict[str, int]:
    cfg = read_json(dataset_config)
    path = cfg["video_duration_idx_path"]
    if not os.path.isabs(path):
        path = os.path.join(cfg["root_path"], path)
    duration_idx = read_json(path)
    return {name: int(value[1]) for name, value in duration_idx[split].items()}


def build_query_prediction(rows: List[Dict], args) -> Dict:
    if not rows:
        raise ValueError("Cannot reconstruct an empty query group")
    rows = sorted(rows, key=lambda row: (-float(row["s_base"]), int(row["rank_base"])))
    predictions = [
        [
            int(row["video_idx"]),
            float(row["start_time"]),
            float(row["end_time"]),
            float(row["s_base"]),
        ]
        for row in rows[: args.max_before_nms]
    ]
    if args.no_nms:
        predictions = predictions[: args.max_after_nms]
    else:
        predictions = filter_vcmr_by_nms(
            predictions,
            nms_threshold=args.nms_thd,
            max_before_nms=args.max_before_nms,
            max_after_nms=args.max_after_nms,
        )
    first = rows[0]
    return {
        "desc_id": first["desc_id"],
        "desc": first.get("desc", ""),
        "predictions": predictions,
    }


def iter_query_groups(rows: Iterable[Dict]):
    current_id = None
    current_rows = []
    completed = set()
    for row in rows:
        desc_id = row["desc_id"]
        if current_id is None:
            current_id = desc_id
        if desc_id != current_id:
            completed.add(current_id)
            yield current_rows
            if desc_id in completed:
                raise ValueError(f"Non-contiguous evidence rows for desc_id={desc_id}")
            current_id = desc_id
            current_rows = []
        current_rows.append(row)
    if current_rows:
        yield current_rows


def main():
    args = parse_args()
    allowed_ids = load_desc_ids(args.desc_ids_filter)
    gt_writer = None
    gt_partial = None
    if args.save_gt_jsonl:
        Path(args.save_gt_jsonl).parent.mkdir(parents=True, exist_ok=True)
        gt_partial = args.save_gt_jsonl + ".partial"
        gt_writer = open(gt_partial, "w", encoding="utf-8")
    vcmr = []
    found_ids = set()
    row_counts = Counter()
    total_rows = 0
    try:
        groups = iter_query_groups(
            selected_rows(args.evidence_jsonl, allowed_ids, args.save_filtered_evidence)
        )
        for group in groups:
            first = group[0]
            found_ids.add(str(first["desc_id"]))
            row_counts[len(group)] += 1
            total_rows += len(group)
            vcmr.append(build_query_prediction(group, args))
            if gt_writer is not None:
                item = {
                    "desc_id": first["desc_id"],
                    "desc": first.get("desc", ""),
                    "vid_name": first["gt_vid_name"],
                    "ts": first["gt_ts"],
                }
                gt_writer.write(
                    json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n"
                )
    except BaseException:
        if gt_writer is not None:
            gt_writer.close()
        if gt_partial and os.path.exists(gt_partial):
            os.remove(gt_partial)
        raise
    else:
        if gt_writer is not None:
            gt_writer.close()
            os.replace(gt_partial, args.save_gt_jsonl)
    if allowed_ids is not None:
        if found_ids != allowed_ids:
            raise ValueError(
                f"Filtered desc_id mismatch: found={len(found_ids)} expected={len(allowed_ids)}"
            )
    if row_counts != Counter({200: len(vcmr)}):
        raise ValueError(f"Unexpected selected rows/query distribution: {dict(row_counts)}")
    submission = {
        "video2idx": load_video2idx(args.dataset_config, args.split),
        "VCMR": vcmr,
    }
    write_json(args.output_json, submission, pretty=False)
    print(
        f"Saved base reconstruction to {args.output_json}; "
        f"queries={len(vcmr)}; rows={total_rows}; "
        f"rows_per_query={dict(row_counts)}; nms={not args.no_nms}; nms_thd={args.nms_thd}"
    )


if __name__ == "__main__":
    main()
