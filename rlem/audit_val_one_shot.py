#!/usr/bin/env python

"""Audit the single authorized C3-strong official-val rerank."""

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from rlem.io_utils import iter_jsonl, read_json, write_json
from utils.inference_utils import filter_vcmr_by_nms


METRIC_KEYS = [
    "0.5-r1", "0.5-r5", "0.5-r10", "0.5-r100",
    "0.7-r1", "0.7-r5", "0.7-r10", "0.7-r100",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scored_jsonl", required=True)
    parser.add_argument("--base_submission", required=True)
    parser.add_argument("--learned_submission", required=True)
    parser.add_argument("--c0_metrics", required=True)
    parser.add_argument("--c1_metrics", required=True)
    parser.add_argument("--learned_metrics", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_md", required=True)
    parser.add_argument("--score_family", choices=["gated_boundary"], required=True)
    parser.add_argument("--base_scale", type=float, required=True)
    parser.add_argument("--a_joint", type=float, required=True)
    parser.add_argument("--b_bd", type=float, required=True)
    parser.add_argument("--d_fp", type=float, required=True)
    parser.add_argument("--expected_queries", type=int, required=True)
    parser.add_argument("--rows_per_query", type=int, default=200)
    parser.add_argument("--effective_top_n", type=int, default=100)
    parser.add_argument("--max_after_nms", type=int, default=100)
    parser.add_argument("--nms_thd", type=float, default=0.7)
    return parser.parse_args()


def iter_groups(path):
    current = None
    rows = []
    completed = set()
    for row in iter_jsonl(path):
        desc_id = row["desc_id"]
        if current is None:
            current = desc_id
        if desc_id != current:
            completed.add(current)
            yield current, rows
            if desc_id in completed:
                raise ValueError(f"Non-contiguous desc_id={desc_id}")
            current, rows = desc_id, []
        rows.append(row)
    if rows:
        yield current, rows


def distribution(values):
    values = np.asarray(values, dtype=np.float64)
    ps = [0, 1, 5, 25, 50, 75, 95, 99, 100]
    return {
        "mean": float(np.mean(values)), "std": float(np.std(values)),
        "min": float(np.min(values)), "max": float(np.max(values)),
        "percentiles": {str(p): float(v) for p, v in zip(ps, np.percentile(values, ps))},
    }


def prediction_equal(left, right):
    return len(left) == len(right) and all(
        len(a) == len(b) and all(float(x) == float(y) for x, y in zip(a, b))
        for a, b in zip(left, right)
    )


def length_summary(submission):
    lengths = [len(item["predictions"]) for item in submission["VCMR"]]
    return {
        "queries": len(lengths), "min": min(lengths), "max": max(lengths),
        "mean": float(np.mean(lengths)),
        "distribution": {str(k): int(v) for k, v in sorted(Counter(lengths).items())},
    }


def main():
    args = parse_args()
    expected_rows = args.expected_queries * args.rows_per_query
    arrays = {
        key: np.empty(expected_rows, dtype=np.float64)
        for key in ["base_log", "q_joint", "q_bd", "gated_boundary", "e_fp", "s_c3"]
    }
    base_submission = read_json(args.base_submission)
    learned_submission = read_json(args.learned_submission)
    base_by_id = {str(item["desc_id"]): item for item in base_submission["VCMR"]}
    learned_by_id = {str(item["desc_id"]): item for item in learned_submission["VCMR"]}
    if set(base_by_id) != set(learned_by_id):
        raise ValueError("Base/learned desc_id sets differ")

    top1_changed = hard_exits = hard_entries = 0
    fp_down = fp_total = hard_down = hard_total = 0
    wrong_boundary_sum = 0.0
    wrong_count = 0
    formula_max_error = 0.0
    base_nms_exact = learned_nms_exact = True
    query_count = offset = 0
    scored_ids = set()

    for desc_id, rows in iter_groups(args.scored_jsonl):
        if len(rows) != args.rows_per_query:
            raise ValueError(f"desc_id={desc_id} rows={len(rows)}")
        sid = str(desc_id)
        if sid not in base_by_id or sid not in learned_by_id:
            raise ValueError(f"desc_id={desc_id} absent from submission")
        scored_ids.add(sid)
        query_count += 1
        base_order = sorted(
            range(len(rows)), key=lambda i: (-float(rows[i]["s_base"]), int(rows[i]["rank_base"]))
        )
        learned_order = sorted(
            range(len(rows)), key=lambda i: (-float(rows[i]["s_rlem"]), int(rows[i]["rank_base"]))
        )
        base_rank = np.empty(len(rows), dtype=np.int32)
        learned_rank = np.empty(len(rows), dtype=np.int32)
        for rank, idx in enumerate(base_order):
            base_rank[idx] = rank
        for rank, idx in enumerate(learned_order):
            learned_rank[idx] = rank
        top1_changed += int(base_order[0] != learned_order[0])

        base_log = np.asarray([float(row["base_score_used"]) for row in rows])
        q_joint = np.asarray([float(row["q_joint_pred"]) for row in rows])
        q_bd = np.asarray([float(row["q_bd_pred"]) for row in rows])
        boundary = q_joint * q_bd
        e_fp = np.asarray([float(row["e_fp_pred"]) for row in rows])
        s_c3 = np.asarray([float(row["s_rlem"]) for row in rows])
        expected = (
            args.base_scale * base_log + args.a_joint * q_joint
            + args.b_bd * boundary - args.d_fp * e_fp
        )
        formula_max_error = max(formula_max_error, float(np.max(np.abs(expected - s_c3))))
        end = offset + len(rows)
        for key, values in [
            ("base_log", base_log), ("q_joint", q_joint), ("q_bd", q_bd),
            ("gated_boundary", boundary), ("e_fp", e_fp), ("s_c3", s_c3),
        ]:
            arrays[key][offset:end] = values
        offset = end

        for i, row in enumerate(rows):
            if int(row["y_fp"]) == 1:
                fp_total += 1
                fp_down += int(learned_rank[i] > base_rank[i])
            if int(row["y_joint_05"]) == 1:
                hard_total += 1
                hard_down += int(learned_rank[i] > base_rank[i])
                hard_exits += int(base_rank[i] < args.effective_top_n <= learned_rank[i])
                hard_entries += int(learned_rank[i] < args.effective_top_n <= base_rank[i])
            if int(row["is_gt_video"]) == 0:
                wrong_boundary_sum += args.b_bd * boundary[i]
                wrong_count += 1

        base_pre = [[
            int(rows[i]["video_idx"]), float(rows[i]["start_time"]),
            float(rows[i]["end_time"]), float(rows[i]["s_base"]),
        ] for i in base_order[:args.effective_top_n]]
        learned_pre = [[
            int(rows[i]["video_idx"]), float(rows[i]["start_time"]),
            float(rows[i]["end_time"]), float(rows[i]["s_rlem"]),
        ] for i in learned_order[:args.effective_top_n]]
        expected_base = filter_vcmr_by_nms(
            base_pre, args.nms_thd, args.effective_top_n, args.max_after_nms
        )
        expected_learned = filter_vcmr_by_nms(
            learned_pre, args.nms_thd, args.effective_top_n, args.max_after_nms
        )
        base_nms_exact &= prediction_equal(expected_base, base_by_id[sid]["predictions"])
        learned_nms_exact &= prediction_equal(expected_learned, learned_by_id[sid]["predictions"])

    if query_count != args.expected_queries or offset != expected_rows:
        raise ValueError(f"Cardinality mismatch queries={query_count}, rows={offset}")
    if scored_ids != set(base_by_id):
        raise ValueError("Scored/submission query sets differ")
    if not all(np.isfinite(values).all() for values in arrays.values()):
        raise ValueError("Non-finite score component")

    c0 = read_json(args.c0_metrics)["VCMR"]
    c1 = read_json(args.c1_metrics)["VCMR"]
    learned = read_json(args.learned_metrics)["VCMR"]
    if c0 != c1:
        raise ValueError("C0/C1 baseline metrics differ")
    delta_c0 = {key: round(learned[key] - c0[key], 6) for key in METRIC_KEYS}
    delta_c1 = {key: round(learned[key] - c1[key], 6) for key in METRIC_KEYS}
    weighted = {
        "base": args.base_scale * arrays["base_log"],
        "q_joint": args.a_joint * arrays["q_joint"],
        "gated_boundary": args.b_bd * arrays["gated_boundary"],
        "negative_e_fp": -args.d_fp * arrays["e_fp"],
    }
    stds = {key: float(np.std(value)) for key, value in weighted.items()}
    total_std = sum(stds.values())
    shares = {key: value / total_std for key, value in stds.items()}
    result = {
        "status": "PASS",
        "stage": "C3-strong official-val one-shot",
        "official_val_one_shot": True,
        "post_val_adjustment": False,
        "score_family": args.score_family,
        "weights": {
            "base_scale": args.base_scale, "a_joint": args.a_joint,
            "b_bd": args.b_bd, "d_fp": args.d_fp,
        },
        "queries": query_count,
        "rows": offset,
        "c0_official_metrics": c0,
        "c1_reconstructed_metrics": c1,
        "c3_strong_one_shot_metrics": learned,
        "delta_vs_c0": delta_c0,
        "delta_vs_c1": delta_c1,
        "top1_changed_queries": top1_changed,
        "top1_changed_query_ratio": top1_changed / query_count,
        "hard_positive_top100_exits": hard_exits,
        "hard_positive_top100_entries": hard_entries,
        "hard_positive_top100_exit_ratio": hard_exits / max(hard_total, 1),
        "y_fp_down_move_ratio": fp_down / max(fp_total, 1),
        "y_joint_05_down_move_ratio": hard_down / max(hard_total, 1),
        "wrong_video_gated_boundary_bonus_mean": wrong_boundary_sum / max(wrong_count, 1),
        "score_component_distributions": {
            key: distribution(values) for key, values in arrays.items()
        },
        "weighted_component_distributions": {
            key: distribution(values) for key, values in weighted.items()
        },
        "weighted_component_std_share": shares,
        "score_correlations": {
            "pearson_base_vs_c3": float(pearsonr(arrays["base_log"], arrays["s_c3"]).statistic),
            "spearman_base_vs_c3": float(spearmanr(arrays["base_log"], arrays["s_c3"]).statistic),
        },
        "formula_max_abs_error": formula_max_error,
        "nms_audit": {
            "effective_top_n": args.effective_top_n,
            "nms_thd": args.nms_thd,
            "max_after_nms": args.max_after_nms,
            "base_exact": base_nms_exact,
            "learned_exact": learned_nms_exact,
        },
        "prediction_lengths": {
            "base": length_summary(base_submission),
            "learned": length_summary(learned_submission),
        },
        "next_stage_authorized": False,
    }
    write_json(args.output_json, result)
    rows_md = "\n".join(
        f"| {key} | {c0[key]:.2f} | {c1[key]:.2f} | {learned[key]:.2f} | {delta_c0[key]:+.2f} |"
        for key in METRIC_KEYS
    )
    components_md = "\n".join(
        f"| {key} | {value['mean']:.6f} | {value['std']:.6f} | {value['percentiles']['5']:.6f} | {value['percentiles']['50']:.6f} | {value['percentiles']['95']:.6f} |"
        for key, value in result["score_component_distributions"].items()
    )
    md = f"""# C3-strong official-val one-shot delta

`official_val_one_shot = true`  
`post_val_adjustment = false`

## Metrics

| Metric | C0 official | C1 reconstructed | C3-strong | Delta vs C0/C1 |
|---|---:|---:|---:|---:|
{rows_md}

## Movement

- Top-1 changed: {top1_changed}/{query_count} ({100*top1_changed/query_count:.2f}%)
- Hard-positive top-100 exits/entries: {hard_exits}/{hard_entries}
- y_fp down-move ratio: {100*result['y_fp_down_move_ratio']:.2f}%
- y_joint_05 down-move ratio: {100*result['y_joint_05_down_move_ratio']:.2f}%

## Score components

| Component | Mean | Std | P5 | P50 | P95 |
|---|---:|---:|---:|---:|---:|
{components_md}

Formula max absolute error: {formula_max_error:.3g}.  
Base/learned top-100 plus NMS exact: `{str(base_nms_exact).lower()}/{str(learned_nms_exact).lower()}`.

No post-val adjustment was made. Stop here.
"""
    Path(args.output_md).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_md).write_text(md, encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
