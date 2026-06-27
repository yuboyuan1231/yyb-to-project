#!/usr/bin/env python

"""Stream-audit a complete C1 evidence artifact without materializing it."""

import argparse
import json
import math
import os
import sys
from collections import Counter
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from rlem.io_utils import iter_jsonl, write_json


REQUIRED_FIELDS = {
    "desc_id", "video_name", "video_idx", "start_idx", "end_idx",
    "start_time", "end_time", "rank_base", "rank_r1", "r1", "r1_tilde",
    "r2_raw", "r2_prob", "p_b_i", "p_e_j", "l_logit", "l_prob", "l_prod",
    "u_b", "u_e", "u_bd", "sharp_b", "sharp_e", "margin_b", "margin_e",
    "mu_v", "mu_s", "m_ctx", "mean_ctx", "max_ctx", "peak_in_ctx", "h_ctx",
    "s_base", "iou", "is_gt_video", "y_joint", "y_joint_05", "y_joint_07",
    "m_bd", "y_bd", "y_fp",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence_jsonl", required=True)
    parser.add_argument("--source_jsonl", required=True)
    parser.add_argument("--expected_rows_per_query", type=int, default=200)
    parser.add_argument("--output_json", required=True)
    return parser.parse_args()


def source_desc_ids(path):
    return [row["desc_id"] for row in iter_jsonl(path)]


def main():
    args = parse_args()
    expected_source_ids = source_desc_ids(args.source_jsonl)
    rows = 0
    query_ids = []
    query_counts = []
    schema = None
    schema_mismatch_rows = 0
    missing_required = Counter()
    nonfinite_values = 0
    invalid_spans = 0
    duplicate_candidates = 0
    invalid_rank_groups = 0
    max_sbase_relative_error = 0.0
    counts = Counter()
    current_id = None
    current_count = 0
    current_candidates = set()
    current_ranks = set()

    def close_group():
        nonlocal duplicate_candidates, invalid_rank_groups
        if current_id is None:
            return
        query_ids.append(current_id)
        query_counts.append(current_count)
        duplicate_candidates += current_count - len(current_candidates)
        if current_ranks != set(range(1, args.expected_rows_per_query + 1)):
            invalid_rank_groups += 1

    for row in iter_jsonl(args.evidence_jsonl):
        desc_id = row["desc_id"]
        if current_id is None:
            current_id = desc_id
        elif desc_id != current_id:
            close_group()
            current_id = desc_id
            current_count = 0
            current_candidates = set()
            current_ranks = set()

        rows += 1
        current_count += 1
        current_candidates.add((row["video_name"], row["start_idx"], row["end_idx"]))
        current_ranks.add(int(row["rank_base"]))

        fields = set(row)
        if schema is None:
            schema = fields
        elif fields != schema:
            schema_mismatch_rows += 1
        for field in REQUIRED_FIELDS - fields:
            missing_required[field] += 1

        for value in row.values():
            if isinstance(value, float) and not math.isfinite(value):
                nonfinite_values += 1

        if not (0 <= row["start_idx"] <= row["end_idx"] < 100):
            invalid_spans += 1
        span_len = int(row["end_idx"]) - int(row["start_idx"]) + 1
        if span_len < 1 or span_len > 24:
            invalid_spans += 1

        calculated = float(row["r1"]) * float(row["p_b_i"]) * float(row["p_e_j"])
        rel_error = abs(float(row["s_base"]) - calculated) / max(abs(calculated), 1e-12)
        max_sbase_relative_error = max(max_sbase_relative_error, rel_error)

        for key in ["is_gt_video", "m_bd", "y_joint_05", "y_joint_07", "y_fp"]:
            counts[f"{key}_positive"] += int(float(row[key]) > 0)
        counts["r2_raw_nonnull"] += int(row.get("r2_raw") is not None)
        counts["r2_prob_nonnull"] += int(row.get("r2_prob") is not None)
        counts["modality_nonnull"] += int(row.get("mu_v") is not None and row.get("mu_s") is not None)
        counts["context_nonnull"] += int(row.get("m_ctx") is not None)
        counts["m_bd_wrong_video"] += int(float(row["m_bd"]) > 0 and not row["is_gt_video"])
        counts["y_bd_wrong_video"] += int(float(row["y_bd"]) > 0 and not row["is_gt_video"])

    close_group()
    query_count_distribution = Counter(query_counts)
    expected_rows = len(expected_source_ids) * args.expected_rows_per_query
    result = {
        "status": "PASS",
        "evidence_jsonl": os.path.abspath(args.evidence_jsonl),
        "compressed_bytes": os.path.getsize(args.evidence_jsonl),
        "rows": rows,
        "queries": len(query_ids),
        "expected_rows": expected_rows,
        "source_query_count": len(expected_source_ids),
        "source_query_order_exact": query_ids == expected_source_ids,
        "rows_per_query_distribution": {str(k): v for k, v in sorted(query_count_distribution.items())},
        "field_count": len(schema or []),
        "fields": sorted(schema or []),
        "missing_required": dict(missing_required),
        "schema_mismatch_rows": schema_mismatch_rows,
        "duplicate_candidates": duplicate_candidates,
        "invalid_rank_groups": invalid_rank_groups,
        "invalid_spans": invalid_spans,
        "nonfinite_values": nonfinite_values,
        "max_sbase_relative_error": max_sbase_relative_error,
        "counts": dict(counts),
    }
    failures = []
    if rows != expected_rows:
        failures.append("row_count")
    if query_ids != expected_source_ids:
        failures.append("source_query_order")
    if query_count_distribution != {args.expected_rows_per_query: len(expected_source_ids)}:
        failures.append("rows_per_query")
    for name, value in [
        ("missing_required", sum(missing_required.values())),
        ("schema_mismatch_rows", schema_mismatch_rows),
        ("duplicate_candidates", duplicate_candidates),
        ("invalid_rank_groups", invalid_rank_groups),
        ("invalid_spans", invalid_spans),
        ("nonfinite_values", nonfinite_values),
        ("m_bd_wrong_video", counts["m_bd_wrong_video"]),
        ("y_bd_wrong_video", counts["y_bd_wrong_video"]),
    ]:
        if value:
            failures.append(name)
    if max_sbase_relative_error > 2e-6:
        failures.append("s_base_formula")
    result["failures"] = failures
    result["status"] = "PASS" if not failures else "FAIL"
    write_json(args.output_json, result)
    print(json.dumps(result, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
