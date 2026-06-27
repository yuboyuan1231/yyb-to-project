#!/usr/bin/env python

"""Audit fixed-weight C3-minimal reranking on train_calib only."""

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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scored_jsonl", required=True)
    parser.add_argument("--base_submission", required=True)
    parser.add_argument("--c3_submission", required=True)
    parser.add_argument("--base_metrics", required=True)
    parser.add_argument("--c3_metrics", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_md", required=True)
    parser.add_argument("--expected_queries", type=int, required=True)
    parser.add_argument("--rows_per_query", type=int, default=200)
    parser.add_argument("--effective_top_n", type=int, default=100)
    parser.add_argument("--max_after_nms", type=int, default=100)
    parser.add_argument("--nms_thd", type=float, default=0.7)
    parser.add_argument("--base_scale", type=float, default=1.0)
    parser.add_argument("--a_joint", type=float, default=1.0)
    parser.add_argument("--b_bd", type=float, default=0.5)
    parser.add_argument("--d_fp", type=float, default=0.5)
    return parser.parse_args()


def iter_groups(path):
    current_id = None
    rows = []
    completed = set()
    for row in iter_jsonl(path):
        desc_id = row["desc_id"]
        if current_id is None:
            current_id = desc_id
        if desc_id != current_id:
            completed.add(current_id)
            yield current_id, rows
            if desc_id in completed:
                raise ValueError(f"Non-contiguous scored rows for desc_id={desc_id}")
            current_id = desc_id
            rows = []
        rows.append(row)
    if rows:
        yield current_id, rows


def identity(row):
    return (int(row["video_idx"]), float(row["start_time"]), float(row["end_time"]))


def distribution(values):
    values = np.asarray(values, dtype=np.float64)
    percentiles = [0, 1, 5, 25, 50, 75, 95, 99, 100]
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "percentiles": {
            str(p): float(v) for p, v in zip(percentiles, np.percentile(values, percentiles))
        },
    }


def movement_summary(counter, rank_deltas, top100_exits, top100_entries, total):
    return {
        "candidates": int(total),
        "rank_degraded": int(counter["degraded"]),
        "rank_unchanged": int(counter["unchanged"]),
        "rank_improved": int(counter["improved"]),
        "rank_degraded_ratio": float(counter["degraded"] / max(total, 1)),
        "mean_rank_delta_c3_minus_base": float(np.mean(rank_deltas)) if rank_deltas else None,
        "median_rank_delta_c3_minus_base": float(np.median(rank_deltas)) if rank_deltas else None,
        "top100_exits": int(top100_exits),
        "top100_entries": int(top100_entries),
        "top100_exit_ratio": float(top100_exits / max(total, 1)),
    }


def rank_movement_summary(counter, deltas, eligible):
    return {
        "eligible_queries": int(eligible),
        "improved": int(counter["improved"]),
        "unchanged": int(counter["unchanged"]),
        "degraded": int(counter["degraded"]),
        "improved_ratio": float(counter["improved"] / max(eligible, 1)),
        "degraded_ratio": float(counter["degraded"] / max(eligible, 1)),
        "mean_rank_gain_base_minus_c3": float(np.mean(deltas)) if deltas else None,
        "median_rank_gain_base_minus_c3": float(np.median(deltas)) if deltas else None,
    }


def prediction_length_summary(submission):
    lengths = [len(item["predictions"]) for item in submission["VCMR"]]
    counts = Counter(lengths)
    return {
        "queries": len(lengths),
        "min": min(lengths),
        "max": max(lengths),
        "mean": float(np.mean(lengths)),
        "distribution": {str(k): int(v) for k, v in sorted(counts.items())},
    }


def prediction_equal(left, right):
    if len(left) != len(right):
        return False
    for a, b in zip(left, right):
        if len(a) != len(b):
            return False
        if any(float(x) != float(y) for x, y in zip(a, b)):
            return False
    return True


def main():
    args = parse_args()
    expected_rows = args.expected_queries * args.rows_per_query
    components = {
        name: np.empty(expected_rows, dtype=np.float64)
        for name in ["base_score", "q_joint", "q_bd", "e_fp", "s_c3"]
    }
    base_submission = read_json(args.base_submission)
    c3_submission = read_json(args.c3_submission)
    base_by_id = {str(item["desc_id"]): item for item in base_submission["VCMR"]}
    c3_by_id = {str(item["desc_id"]): item for item in c3_submission["VCMR"]}
    if set(base_by_id) != set(c3_by_id):
        raise ValueError("Base/C3 submission desc_id sets differ")

    top1_changed = 0
    gt_video = Counter()
    gt_video_deltas = []
    gt_span = Counter()
    gt_span_deltas = []
    fp_counter, hard_counter = Counter(), Counter()
    fp_deltas, hard_deltas = [], []
    fp_exits = fp_entries = hard_exits = hard_entries = 0
    fp_total = hard_total = 0
    nms_exact_base = nms_exact_c3 = True
    query_count = 0
    offset = 0
    qbd_wrong_sum = qbd_gt_sum = 0.0
    qbd_wrong_count = qbd_gt_count = 0
    penalty_over_positive_boost = 0
    scored_ids = set()
    score_formula_max_abs_error = 0.0
    min_raw_s_base = float("inf")
    nonpositive_raw_s_base = 0

    for desc_id, rows in iter_groups(args.scored_jsonl):
        if len(rows) != args.rows_per_query:
            raise ValueError(f"desc_id={desc_id} has {len(rows)} scored rows")
        sid = str(desc_id)
        scored_ids.add(sid)
        if sid not in base_by_id or sid not in c3_by_id:
            raise ValueError(f"desc_id={desc_id} absent from submissions")
        query_count += 1

        base_order = sorted(range(len(rows)), key=lambda i: (-float(rows[i]["s_base"]), int(rows[i]["rank_base"])))
        c3_order = sorted(range(len(rows)), key=lambda i: (-float(rows[i]["s_rlem"]), int(rows[i]["rank_base"])))
        base_rank = np.empty(len(rows), dtype=np.int32)
        c3_rank = np.empty(len(rows), dtype=np.int32)
        for rank, idx in enumerate(base_order, start=1):
            base_rank[idx] = rank
        for rank, idx in enumerate(c3_order, start=1):
            c3_rank[idx] = rank

        if identity(rows[base_order[0]]) != identity(rows[c3_order[0]]):
            top1_changed += 1

        gt_indices = [i for i, row in enumerate(rows) if int(row["is_gt_video"]) == 1]
        if gt_indices:
            base_best = min(int(base_rank[i]) for i in gt_indices)
            c3_best = min(int(c3_rank[i]) for i in gt_indices)
            gain = base_best - c3_best
            gt_video_deltas.append(gain)
            gt_video["improved" if gain > 0 else "degraded" if gain < 0 else "unchanged"] += 1

            target = max(gt_indices, key=lambda i: (float(rows[i]["iou"]), -int(rows[i]["rank_base"])))
            gt_base_order = sorted(gt_indices, key=lambda i: (-float(rows[i]["s_base"]), int(rows[i]["rank_base"])))
            gt_c3_order = sorted(gt_indices, key=lambda i: (-float(rows[i]["s_rlem"]), int(rows[i]["rank_base"])))
            base_span_rank = gt_base_order.index(target) + 1
            c3_span_rank = gt_c3_order.index(target) + 1
            span_gain = base_span_rank - c3_span_rank
            gt_span_deltas.append(span_gain)
            gt_span["improved" if span_gain > 0 else "degraded" if span_gain < 0 else "unchanged"] += 1

        for i, row in enumerate(rows):
            delta = int(c3_rank[i]) - int(base_rank[i])
            state = "degraded" if delta > 0 else "improved" if delta < 0 else "unchanged"
            if int(row["y_fp"]) == 1:
                fp_total += 1
                fp_counter[state] += 1
                fp_deltas.append(delta)
                fp_exits += int(base_rank[i] <= args.effective_top_n < c3_rank[i])
                fp_entries += int(c3_rank[i] <= args.effective_top_n < base_rank[i])
            if int(row["y_joint_05"]) == 1:
                hard_total += 1
                hard_counter[state] += 1
                hard_deltas.append(delta)
                hard_exits += int(base_rank[i] <= args.effective_top_n < c3_rank[i])
                hard_entries += int(c3_rank[i] <= args.effective_top_n < base_rank[i])

        qj = np.asarray([float(row["q_joint_pred"]) for row in rows])
        qb = np.asarray([float(row["q_bd_pred"]) for row in rows])
        efp = np.asarray([float(row["e_fp_pred"]) for row in rows])
        base = np.asarray([float(row["base_score_used"]) for row in rows])
        s_c3 = np.asarray([float(row["s_rlem"]) for row in rows])
        end = offset + len(rows)
        components["base_score"][offset:end] = base
        components["q_joint"][offset:end] = qj
        components["q_bd"][offset:end] = qb
        components["e_fp"][offset:end] = efp
        components["s_c3"][offset:end] = s_c3
        offset = end
        gt_mask = np.asarray([int(row["is_gt_video"]) == 1 for row in rows])
        qbd_gt_sum += float(np.sum(qb[gt_mask]))
        qbd_wrong_sum += float(np.sum(qb[~gt_mask]))
        qbd_gt_count += int(np.sum(gt_mask))
        qbd_wrong_count += int(np.sum(~gt_mask))
        positive_boost = args.a_joint * qj + args.b_bd * qb
        penalty_over_positive_boost += int(np.sum(args.d_fp * efp > positive_boost))
        expected_s_c3 = args.base_scale * base + positive_boost - args.d_fp * efp
        score_formula_max_abs_error = max(
            score_formula_max_abs_error, float(np.max(np.abs(expected_s_c3 - s_c3)))
        )
        raw_s_base = np.asarray([float(row["s_base"]) for row in rows])
        min_raw_s_base = min(min_raw_s_base, float(np.min(raw_s_base)))
        nonpositive_raw_s_base += int(np.sum(raw_s_base <= 0))

        base_pre = [
            [int(rows[i]["video_idx"]), float(rows[i]["start_time"]), float(rows[i]["end_time"]), float(rows[i]["s_base"])]
            for i in base_order[:args.effective_top_n]
        ]
        c3_pre = [
            [int(rows[i]["video_idx"]), float(rows[i]["start_time"]), float(rows[i]["end_time"]), float(rows[i]["s_rlem"])]
            for i in c3_order[:args.effective_top_n]
        ]
        expected_base = filter_vcmr_by_nms(
            base_pre, args.nms_thd, args.effective_top_n, args.max_after_nms
        )
        expected_c3 = filter_vcmr_by_nms(
            c3_pre, args.nms_thd, args.effective_top_n, args.max_after_nms
        )
        nms_exact_base &= prediction_equal(expected_base, base_by_id[sid]["predictions"])
        nms_exact_c3 &= prediction_equal(expected_c3, c3_by_id[sid]["predictions"])

    if query_count != args.expected_queries or offset != expected_rows:
        raise ValueError(
            f"Scored cardinality mismatch: queries={query_count}/{args.expected_queries}, rows={offset}/{expected_rows}"
        )
    if set(base_by_id) != scored_ids:
        raise ValueError("Scored/submission query sets differ")
    if not all(np.isfinite(values).all() for values in components.values()):
        raise ValueError("Non-finite score component")

    base_metrics = read_json(args.base_metrics)["VCMR"]
    c3_metrics = read_json(args.c3_metrics)["VCMR"]
    metric_delta = {key: float(c3_metrics[key] - base_metrics[key]) for key in base_metrics}
    weighted = {
        "base": args.base_scale * components["base_score"],
        "q_joint": args.a_joint * components["q_joint"],
        "q_bd": args.b_bd * components["q_bd"],
        "negative_e_fp": -args.d_fp * components["e_fp"],
    }
    stds = {key: float(np.std(value)) for key, value in weighted.items()}
    std_sum = sum(stds.values())
    std_shares = {key: value / std_sum for key, value in stds.items()}
    hard_movement = movement_summary(
        hard_counter, hard_deltas, hard_exits, hard_entries, hard_total
    )
    gates = {
        "base_reconstruction_complete": len(base_by_id) == args.expected_queries,
        "c3_submission_complete": len(c3_by_id) == args.expected_queries,
        "no_cliff_drop_ge_5_points": min(metric_delta.values()) >= -5.0,
        "all_components_finite": True,
        "score_formula_exact_within_1e_6": score_formula_max_abs_error <= 1e-6,
        "raw_s_base_strictly_positive_for_log": nonpositive_raw_s_base == 0,
        "no_component_std_share_ge_0_98": max(std_shares.values()) < 0.98,
        "top1_changed_nonzero_and_below_95pct": 0 < top1_changed < 0.95 * query_count,
        "hard_positive_rank_degradation_below_75pct": hard_movement["rank_degraded_ratio"] < 0.75,
        "hard_positive_top100_exit_below_25pct": hard_movement["top100_exit_ratio"] < 0.25,
        "effective_top100_exact": nms_exact_base and nms_exact_c3,
        "max_after_nms_100": (
            prediction_length_summary(base_submission)["max"] <= args.max_after_nms
            and prediction_length_summary(c3_submission)["max"] <= args.max_after_nms
        ),
        "official_val_not_used": True,
    }
    status = "PASS" if all(gates.values()) else "FAIL"
    result = {
        "status": status,
        "stage": "C3-minimal",
        "scope": "train_calib_only",
        "official_val_used": False,
        "queries": query_count,
        "rows": offset,
        "protocol": {
            "base_score_mode": "log",
            "base_scale": args.base_scale,
            "a_joint": args.a_joint,
            "b_bd": args.b_bd,
            "d_fp": args.d_fp,
            "candidate_rows_per_query": args.rows_per_query,
            "effective_top_n": args.effective_top_n,
            "nms_thd": args.nms_thd,
            "max_after_nms": args.max_after_nms,
        },
        "base_metrics": base_metrics,
        "c3_default_metrics": c3_metrics,
        "metric_delta_c3_minus_base": metric_delta,
        "component_distributions": {
            key: distribution(value) for key, value in components.items()
        },
        "weighted_component_std": stds,
        "weighted_component_std_share": std_shares,
        "score_formula_audit": {
            "max_abs_error": score_formula_max_abs_error,
            "min_raw_s_base": min_raw_s_base,
            "nonpositive_raw_s_base": nonpositive_raw_s_base,
        },
        "score_correlations": {
            "pearson_base_vs_c3": float(pearsonr(components["base_score"], components["s_c3"]).statistic),
            "spearman_base_vs_c3": float(spearmanr(components["base_score"], components["s_c3"]).statistic),
        },
        "top1_changed_queries": top1_changed,
        "top1_changed_query_ratio": top1_changed / query_count,
        "gt_video_best_rank_movement": rank_movement_summary(
            gt_video, gt_video_deltas, len(gt_video_deltas)
        ),
        "gt_video_best_iou_span_rank_movement": rank_movement_summary(
            gt_span, gt_span_deltas, len(gt_span_deltas)
        ),
        "false_positive_movement": movement_summary(
            fp_counter, fp_deltas, fp_exits, fp_entries, fp_total
        ),
        "hard_positive_y_joint_05_movement": hard_movement,
        "e_fp_penalty": {
            "weighted_penalty_distribution": distribution(args.d_fp * components["e_fp"]),
            "penalty_exceeds_total_positive_boost_count": penalty_over_positive_boost,
            "penalty_exceeds_total_positive_boost_ratio": penalty_over_positive_boost / offset,
        },
        "q_bd_wrong_video_check": {
            "gt_video_mean": qbd_gt_sum / qbd_gt_count,
            "wrong_video_mean": qbd_wrong_sum / qbd_wrong_count,
            "weighted_wrong_video_mean_bonus": args.b_bd * qbd_wrong_sum / qbd_wrong_count,
        },
        "prediction_lengths": {
            "base": prediction_length_summary(base_submission),
            "c3_default": prediction_length_summary(c3_submission),
        },
        "nms_audit": {
            "pre_nms_input_rows_per_query": args.effective_top_n,
            "base_submission_exact_to_recomputed_top100_nms": nms_exact_base,
            "c3_submission_exact_to_recomputed_top100_nms": nms_exact_c3,
            "post_nms_max": args.max_after_nms,
        },
        "gates": gates,
    }
    result["eligible_for_c3_strong_train_calib_search"] = bool(
        status == "PASS" and max(metric_delta.values()) > 0
    )
    write_json(args.output_json, result)

    keys = list(base_metrics)
    metric_rows = "\n".join(
        f"| {key} | {base_metrics[key]:.2f} | {c3_metrics[key]:.2f} | {metric_delta[key]:+.2f} |"
        for key in keys
    )
    component_rows = "\n".join(
        f"| {key} | {value['mean']:.6f} | {value['std']:.6f} | {value['min']:.6f} | {value['max']:.6f} |"
        for key, value in result["component_distributions"].items()
    )
    md = f"""# C3-minimal train_calib diagnostics

Status: `{status}`  
Scope: `train_calib_only`  
Official val used: `false`

## Metrics

| Metric | S_base | S_C3-default | Delta |
|---|---:|---:|---:|
{metric_rows}

## Score components

| Component | Mean | Std | Min | Max |
|---|---:|---:|---:|---:|
{component_rows}

- Pearson(S_base, S_C3): {result['score_correlations']['pearson_base_vs_c3']:.6f}
- Spearman(S_base, S_C3): {result['score_correlations']['spearman_base_vs_c3']:.6f}
- Top-1 changed queries: {top1_changed}/{query_count} ({100 * top1_changed / query_count:.2f}%)
- GT-video best-rank improved/degraded: {gt_video['improved']}/{gt_video['degraded']}
- GT-video best-IoU span rank improved/degraded: {gt_span['improved']}/{gt_span['degraded']}
- y_fp rank degraded ratio: {100 * result['false_positive_movement']['rank_degraded_ratio']:.2f}%
- y_joint_05 rank degraded ratio: {100 * hard_movement['rank_degraded_ratio']:.2f}%
- y_joint_05 top-100 exit ratio: {100 * hard_movement['top100_exit_ratio']:.2f}%
- Weighted wrong-video Q_bd mean bonus: {result['q_bd_wrong_video_check']['weighted_wrong_video_mean_bonus']:.6f}
- Base/C3 top-100 + NMS exact: {str(nms_exact_base).lower()}/{str(nms_exact_c3).lower()}
- Post-NMS maximum list length: {max(result['prediction_lengths']['base']['max'], result['prediction_lengths']['c3_default']['max'])}

Full percentiles, movement counts, penalty diagnostics, prediction-length distributions,
and machine-readable gates are in the JSON artifact.
"""
    Path(args.output_md).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_md).write_text(md, encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
