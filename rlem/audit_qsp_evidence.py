#!/usr/bin/env python

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

"""Audit C3.5-QSP evidence JSONL without training."""

import argparse
import json
import math
from collections import Counter, defaultdict

import numpy as np
from scipy.stats import spearmanr
from tqdm import tqdm

from rlem.io_utils import iter_jsonl, write_json
from rlem.qsp_features import QSP_FEATURES

IDENTITY_FIELDS = ["desc_id", "video_name", "rank_base", "start_idx", "end_idx"]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--qsp_jsonl", required=True)
    parser.add_argument("--base_jsonl", default=None, help="Optional accepted C1 evidence for row identity comparison")
    parser.add_argument("--expected_rows_per_query", type=int, default=200)
    parser.add_argument(
        "--percentile_sample_stride", type=int, default=87,
        help="Deterministic full-file sampling stride for percentile and Spearman estimates.",
    )
    parser.add_argument(
        "--require_base_exhaustion", action="store_true",
        help="Deprecated compatibility flag; full exhaustion is now the default.",
    )
    parser.add_argument(
        "--allow_base_prefix", action="store_true",
        help="Allow the base JSONL to contain trailing rows (smoke audits only).",
    )
    parser.add_argument("--output_json", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    total = 0
    per_query = Counter()
    nonfinite = Counter()
    values = defaultdict(list)
    full_stats = {
        name: {"count": 0, "sum": 0.0, "sum_sq": 0.0, "min": float("inf"), "max": float("-inf")}
        for name in QSP_FEATURES
    }
    gate_ctx = {"count": 0, "sum_x": 0.0, "sum_y": 0.0, "sum_xx": 0.0, "sum_yy": 0.0, "sum_xy": 0.0}
    gate_ctx_sample_x = []
    gate_ctx_sample_y = []
    base_iter = iter_jsonl(args.base_jsonl) if args.base_jsonl else None
    identity_mismatches = 0
    duplicate_candidate_count = 0
    current_desc_id = None
    current_query_keys = set()
    completed_desc_ids = set()
    noncontiguous_query_segments = 0
    for row in tqdm(iter_jsonl(args.qsp_jsonl), desc="audit QSP evidence"):
        total += 1
        desc_id = str(row.get("desc_id"))
        per_query[desc_id] += 1
        if desc_id != current_desc_id:
            if current_desc_id is not None:
                completed_desc_ids.add(current_desc_id)
            if desc_id in completed_desc_ids:
                noncontiguous_query_segments += 1
            current_desc_id = desc_id
            current_query_keys.clear()
        key = (str(row.get("video_name")), int(row.get("start_idx")), int(row.get("end_idx")))
        if key in current_query_keys:
            duplicate_candidate_count += 1
        current_query_keys.add(key)
        if base_iter is not None:
            try:
                base = next(base_iter)
            except StopIteration:
                identity_mismatches += 1
                base = None
            if base is not None:
                for field in IDENTITY_FIELDS:
                    if str(row.get(field)) != str(base.get(field)):
                        identity_mismatches += 1
                        break
        row_values = {}
        for name in QSP_FEATURES:
            try:
                val = float(row.get(name))
            except (TypeError, ValueError):
                nonfinite[name] += 1
                continue
            if not math.isfinite(val):
                nonfinite[name] += 1
            else:
                row_values[name] = val
                state = full_stats[name]
                state["count"] += 1
                state["sum"] += val
                state["sum_sq"] += val * val
                state["min"] = min(state["min"], val)
                state["max"] = max(state["max"], val)
                if total % args.percentile_sample_stride == 0:
                    values[name].append(val)
        try:
            gate = row_values["qsp_mass_gate"]
            ctx = float(row.get("m_ctx"))
            if not math.isfinite(ctx):
                raise ValueError
            gate_ctx["count"] += 1
            gate_ctx["sum_x"] += gate
            gate_ctx["sum_y"] += ctx
            gate_ctx["sum_xx"] += gate * gate
            gate_ctx["sum_yy"] += ctx * ctx
            gate_ctx["sum_xy"] += gate * ctx
            if total % args.percentile_sample_stride == 0:
                gate_ctx_sample_x.append(gate)
                gate_ctx_sample_y.append(ctx)
        except (KeyError, TypeError, ValueError):
            pass
    base_has_trailing_rows = False
    require_base_exhaustion = not args.allow_base_prefix
    if base_iter is not None:
        try:
            next(base_iter)
            base_has_trailing_rows = True
            if require_base_exhaustion:
                identity_mismatches += 1
        except StopIteration:
            pass
    bad_query_counts = {qid: cnt for qid, cnt in per_query.items() if cnt != args.expected_rows_per_query}
    stats = {}
    for name in QSP_FEATURES:
        arr = np.asarray(values.get(name, []), dtype=np.float64)
        state = full_stats[name]
        count = state["count"]
        if count == 0:
            stats[name] = {"count": 0, "percentile_sample_count": 0}
        else:
            mean = state["sum"] / count
            variance = max(0.0, state["sum_sq"] / count - mean * mean)
            stats[name] = {
                "count": count,
                "mean": mean,
                "std": math.sqrt(variance),
                "min": state["min"],
                "p5": float(np.percentile(arr, 5)),
                "p50": float(np.percentile(arr, 50)),
                "p95": float(np.percentile(arr, 95)),
                "max": state["max"],
                "percentile_sample_count": int(arr.size),
            }
    bounded_01 = {
        name for name in QSP_FEATURES
        if not name.startswith("qsp_entropy_") and name not in {"qsp_js_vs", "qsp_modality_entropy"}
    }
    range_violations = {}
    collapsed_features = []
    for name, summary in stats.items():
        if not summary.get("count"):
            range_violations[name] = "no finite samples"
            continue
        lo, hi = summary["min"], summary["max"]
        if lo < -1e-6:
            range_violations[name] = f"negative minimum {lo}"
        elif name in bounded_01 and hi > 1.0 + 1e-6:
            range_violations[name] = f"maximum above one: {hi}"
        elif name in {"qsp_js_vs", "qsp_modality_entropy"} and hi > math.log(2.0) + 1e-6:
            range_violations[name] = f"maximum above log(2): {hi}"
        elif name.startswith("qsp_entropy_") and hi > math.log(100.0) + 1e-5:
            range_violations[name] = f"maximum above log(100): {hi}"
        if summary.get("std", 0.0) <= 1e-12:
            collapsed_features.append(name)
    near_collapsed_features = [
        name for name, summary in stats.items()
        if summary.get("count") and summary.get("std", 0.0) <= 1e-6
    ]
    collapse_fail_threshold = max(2, math.ceil(len(QSP_FEATURES) * 0.25))
    collapse_gate_pass = len(collapsed_features) < collapse_fail_threshold
    passed = (
        not bad_query_counts and not nonfinite and identity_mismatches == 0
        and duplicate_candidate_count == 0 and not range_violations and collapse_gate_pass
        and noncontiguous_query_segments == 0
    )
    rows_per_query_distribution = {
        str(rows): count for rows, count in sorted(Counter(per_query.values()).items())
    }
    query_row_counts = np.asarray(list(per_query.values()), dtype=np.float64)
    rows_per_query_summary = {
        "min": int(query_row_counts.min()) if query_row_counts.size else None,
        "mean": float(query_row_counts.mean()) if query_row_counts.size else None,
        "p5": float(np.percentile(query_row_counts, 5)) if query_row_counts.size else None,
        "p50": float(np.percentile(query_row_counts, 50)) if query_row_counts.size else None,
        "p95": float(np.percentile(query_row_counts, 95)) if query_row_counts.size else None,
        "max": int(query_row_counts.max()) if query_row_counts.size else None,
    }
    n_corr = gate_ctx["count"]
    if n_corr:
        numerator = n_corr * gate_ctx["sum_xy"] - gate_ctx["sum_x"] * gate_ctx["sum_y"]
        denominator = math.sqrt(max(0.0, n_corr * gate_ctx["sum_xx"] - gate_ctx["sum_x"] ** 2))
        denominator *= math.sqrt(max(0.0, n_corr * gate_ctx["sum_yy"] - gate_ctx["sum_y"] ** 2))
        pearson = numerator / denominator if denominator > 0 else None
    else:
        pearson = None
    sampled_spearman = None
    if len(gate_ctx_sample_x) >= 2:
        value = float(spearmanr(gate_ctx_sample_x, gate_ctx_sample_y).statistic)
        sampled_spearman = value if math.isfinite(value) else None
    report = {
        "status": "PASS" if passed else "FAIL",
        "rows": total,
        "queries": len(per_query),
        "total_rows": total,
        "query_count": len(per_query),
        "expected_rows_per_query": args.expected_rows_per_query,
        "rows_per_query_distribution": rows_per_query_distribution,
        "rows_per_query_summary": rows_per_query_summary,
        "bad_query_count_entries": len(bad_query_counts),
        "nonfinite_counts": dict(nonfinite),
        "identity_mismatches": identity_mismatches,
        "duplicate_candidate_keys": duplicate_candidate_count,
        "duplicate_candidates": duplicate_candidate_count,
        "noncontiguous_query_segments": noncontiguous_query_segments,
        "base_has_trailing_rows": base_has_trailing_rows,
        "base_evidence_full_consumption_required": require_base_exhaustion,
        "range_violations": range_violations,
        "collapsed_features": collapsed_features,
        "near_collapsed_features_std_le_1e-6": near_collapsed_features,
        "collapse_gate": {
            "pass": collapse_gate_pass,
            "collapsed_feature_count": len(collapsed_features),
            "fail_threshold": collapse_fail_threshold,
            "qsp_sub_coverage_non_informative": "qsp_sub_coverage" in collapsed_features,
            "training_protocol_action": (
                "add drop_sub_coverage ablation" if "qsp_sub_coverage" in collapsed_features else "none"
            ),
        },
        "feature_stats_sampled": stats,
        "percentile_sampling": {
            "method": "deterministic full-file stride",
            "stride": args.percentile_sample_stride,
        },
        "p_gate_p_ctx_correlation": {
            "definition": "span-level proxy: qsp_mass_gate versus C1 m_ctx for the same fixed candidate span",
            "count": n_corr,
            "pearson_full": pearson,
            "spearman_sampled": sampled_spearman,
            "spearman_sample_count": len(gate_ctx_sample_x),
        },
        "qsp_sub_coverage_full_train_std": stats["qsp_sub_coverage"].get("std"),
        "qsp_modality_weight_distribution": {
            "qsp_q_v": stats["qsp_q_v"],
            "qsp_q_s": stats["qsp_q_s"],
        },
        "qsp_js_vs_distribution": stats["qsp_js_vs"],
        "official_val_used": False,
        "official_val_used_for_selection": False,
    }
    write_json(args.output_json, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
