#!/usr/bin/env python

"""Audit the single authorized C3.1 official-val one-shot."""

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr

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
    parser.add_argument("--c3_metrics", required=True)
    parser.add_argument("--learned_metrics", required=True)
    parser.add_argument("--frozen_config", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_md", required=True)
    parser.add_argument("--expected_queries", type=int, required=True)
    parser.add_argument("--rows_per_query", type=int, default=200)
    parser.add_argument("--effective_top_n", type=int, default=100)
    parser.add_argument("--max_after_nms", type=int, default=100)
    parser.add_argument("--nms_thd", type=float, default=0.7)
    return parser.parse_args()


def iter_groups(path):
    current, rows, completed = None, [], set()
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
    ps = [0, 5, 50, 95, 100]
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


def metric_obj(path):
    obj = read_json(path)
    return obj["VCMR"] if "VCMR" in obj else obj


def delta(left, right):
    return {key: round(left[key] - right[key], 6) for key in METRIC_KEYS}


def main():
    args = parse_args()
    frozen = read_json(args.frozen_config)
    cfg = frozen["score_config"]
    mean_qbd = float(frozen["diagnostics"]["mean_qbd_train_calib"])
    expected_rows = args.expected_queries * args.rows_per_query
    array_names = ["base_log", "q_joint", "q_bd", "e_fp", "boundary", "s_c31"]
    arrays = {key: np.empty(expected_rows, dtype=np.float64) for key in array_names}
    base_submission = read_json(args.base_submission)
    learned_submission = read_json(args.learned_submission)
    base_by_id = {str(item["desc_id"]): item for item in base_submission["VCMR"]}
    learned_by_id = {str(item["desc_id"]): item for item in learned_submission["VCMR"]}
    if set(base_by_id) != set(learned_by_id):
        raise ValueError("Base/learned query sets differ")

    top1_changed = hard_exits = hard_entries = 0
    fp_down = fp_total = hard_down = hard_total = 0
    wrong_boundary_sum = 0.0
    wrong_count = query_count = offset = 0
    formula_error = calibration_error = centered_error = mean_error = 0.0
    base_nms_exact = learned_nms_exact = True
    spearman_values = []
    scored_ids = set()
    for desc_id, rows in iter_groups(args.scored_jsonl):
        if len(rows) != args.rows_per_query:
            raise ValueError(f"desc_id={desc_id} rows={len(rows)}")
        sid = str(desc_id)
        if sid not in base_by_id or sid not in learned_by_id:
            raise ValueError(f"Unknown desc_id={desc_id}")
        scored_ids.add(sid)
        query_count += 1
        base_order = sorted(range(len(rows)), key=lambda i: (-float(rows[i]["s_base"]), int(rows[i]["rank_base"])))
        learned_order = sorted(range(len(rows)), key=lambda i: (-float(rows[i]["s_c31"]), int(rows[i]["rank_base"])))
        base_rank = np.empty(len(rows), dtype=np.int32)
        learned_rank = np.empty(len(rows), dtype=np.int32)
        for rank, idx in enumerate(base_order): base_rank[idx] = rank
        for rank, idx in enumerate(learned_order): learned_rank[idx] = rank
        top1_changed += int(base_order[0] != learned_order[0])
        rank_diff = base_rank.astype(np.float64) - learned_rank.astype(np.float64)
        n = len(rows)
        spearman_values.append(1.0 - 6.0 * np.sum(rank_diff * rank_diff) / (n * (n * n - 1.0)))

        base_log = np.asarray([float(row["base_score_used"]) for row in rows])
        qj = np.asarray([float(row["q_joint_cal"]) for row in rows])
        qb = np.asarray([float(row["q_bd_cal"]) for row in rows])
        efp = np.asarray([float(row["e_fp_cal"]) for row in rows])
        boundary = np.asarray([float(row["centered_gated_boundary"]) for row in rows])
        score = np.asarray([float(row["s_c31"]) for row in rows])
        logits = {
            "qj": np.asarray([float(row["q_joint_logit"]) for row in rows]),
            "qb": np.asarray([float(row["q_bd_logit"]) for row in rows]),
            "efp": np.asarray([float(row["e_fp_logit"]) for row in rows]),
        }
        expected_qj = 1.0 / (1.0 + np.exp(-logits["qj"] / cfg["t_joint"]))
        expected_qb = 1.0 / (1.0 + np.exp(-logits["qb"] / cfg["t_bd"]))
        expected_efp = 1.0 / (1.0 + np.exp(-logits["efp"] / cfg["t_fp"]))
        calibration_error = max(calibration_error, float(np.max(np.abs(expected_qj-qj))), float(np.max(np.abs(expected_qb-qb))), float(np.max(np.abs(expected_efp-efp))))
        centered_error = max(centered_error, float(np.max(np.abs(qj * (qb - mean_qbd) - boundary))))
        mean_error = max(mean_error, max(abs(float(row["mean_qbd_train_calib"]) - mean_qbd) for row in rows))
        expected_score = cfg["base_scale"]*base_log + cfg["a_joint"]*qj + cfg["b_bd"]*boundary - cfg["d_fp"]*efp
        formula_error = max(formula_error, float(np.max(np.abs(expected_score-score))))
        end = offset + n
        for key, values in [("base_log",base_log),("q_joint",qj),("q_bd",qb),("e_fp",efp),("boundary",boundary),("s_c31",score)]: arrays[key][offset:end] = values
        offset = end

        for i, row in enumerate(rows):
            if int(row["y_fp"]) == 1:
                fp_total += 1; fp_down += int(learned_rank[i] > base_rank[i])
            if int(row["y_joint_05"]) == 1:
                hard_total += 1; hard_down += int(learned_rank[i] > base_rank[i])
                hard_exits += int(base_rank[i] < args.effective_top_n <= learned_rank[i])
                hard_entries += int(learned_rank[i] < args.effective_top_n <= base_rank[i])
            if int(row["is_gt_video"]) == 0:
                wrong_boundary_sum += cfg["b_bd"] * boundary[i]; wrong_count += 1

        base_pre = [[int(rows[i]["video_idx"]),float(rows[i]["start_time"]),float(rows[i]["end_time"]),float(rows[i]["s_base"])] for i in base_order[:args.effective_top_n]]
        learned_pre = [[int(rows[i]["video_idx"]),float(rows[i]["start_time"]),float(rows[i]["end_time"]),float(rows[i]["s_c31"])] for i in learned_order[:args.effective_top_n]]
        expected_base = filter_vcmr_by_nms(base_pre,args.nms_thd,args.effective_top_n,args.max_after_nms)
        expected_learned = filter_vcmr_by_nms(learned_pre,args.nms_thd,args.effective_top_n,args.max_after_nms)
        base_nms_exact &= prediction_equal(expected_base,base_by_id[sid]["predictions"])
        learned_nms_exact &= prediction_equal(expected_learned,learned_by_id[sid]["predictions"])

    if query_count != args.expected_queries or offset != expected_rows or scored_ids != set(base_by_id):
        raise ValueError("One-shot cardinality/query-set mismatch")
    if not all(np.isfinite(value).all() for value in arrays.values()):
        raise ValueError("Non-finite score components")
    c0, c1, c3, c31 = map(metric_obj,[args.c0_metrics,args.c1_metrics,args.c3_metrics,args.learned_metrics])
    if c0 != c1:
        raise ValueError("C0 and C1 reconstructed baseline metrics differ")
    weighted = {
        "base": cfg["base_scale"]*arrays["base_log"],
        "q_joint": cfg["a_joint"]*arrays["q_joint"],
        "centered_boundary": cfg["b_bd"]*arrays["boundary"],
        "negative_e_fp": -cfg["d_fp"]*arrays["e_fp"],
    }
    stds = {key: float(np.std(value)) for key,value in weighted.items()}
    total = sum(stds.values())
    shares = {key: value/total for key,value in stds.items()}
    result = {
        "status":"PASS","stage":"C3.1 official-val one-shot","official_val_one_shot":True,
        "post_val_adjustment":False,"learned_config_count":1,"official_val_calibration_performed":False,
        "frozen_config":cfg,"mean_qbd_train_calib":mean_qbd,"queries":query_count,"rows":offset,
        "metrics":{"c0_official":c0,"c1_reconstructed":c1,"c3_strong":c3,"c31_one_shot":c31},
        "deltas":{"c31_vs_c0":delta(c31,c0),"c31_vs_c1":delta(c31,c1),"c31_vs_c3_strong":delta(c31,c3)},
        "top1_changed_queries":top1_changed,"top1_changed_query_ratio":top1_changed/query_count,
        "hard_positive_top100_exits":hard_exits,"hard_positive_top100_entries":hard_entries,
        "hard_positive_top100_exit_ratio":hard_exits/max(hard_total,1),
        "y_fp_down_move_ratio":fp_down/max(fp_total,1),"y_joint_05_down_move_ratio":hard_down/max(hard_total,1),
        "wrong_video_centered_boundary_bonus_mean":wrong_boundary_sum/max(wrong_count,1),
        "score_correlations":{"pearson_base_vs_c31":float(pearsonr(arrays["base_log"],arrays["s_c31"]).statistic),"spearman_mean_within_query":float(np.mean(spearman_values))},
        "weighted_component_std":stds,"weighted_component_std_share":shares,
        "score_component_distributions":{key:distribution(value) for key,value in arrays.items()},
        "score_formula_max_abs_error":formula_error,"calibration_max_abs_error":calibration_error,
        "centered_boundary_max_abs_error":centered_error,"frozen_mean_max_abs_error":mean_error,
        "top100_plus_nms_exactness":{"base_exact":base_nms_exact,"learned_exact":learned_nms_exact,"effective_top_n":args.effective_top_n,"nms_thd":args.nms_thd,"max_after_nms":args.max_after_nms},
        "prediction_length":{"base_max":max(len(x["predictions"]) for x in base_submission["VCMR"]),"learned_max":max(len(x["predictions"]) for x in learned_submission["VCMR"])},
        "next_stage_authorized":False,
    }
    if formula_error > 1e-6 or calibration_error > 1e-6 or centered_error > 1e-6 or mean_error > 1e-12 or not base_nms_exact or not learned_nms_exact:
        result["status"] = "FAIL"
    write_json(args.output_json,result)
    metric_rows = "\n".join(f"| {k} | {c0[k]:.2f} | {c1[k]:.2f} | {c3[k]:.2f} | {c31[k]:.2f} | {c31[k]-c0[k]:+.2f} | {c31[k]-c3[k]:+.2f} |" for k in METRIC_KEYS)
    md = f"""# C3.1 official-val one-shot delta

`official_val_one_shot = true`  
`post_val_adjustment = false`

| Metric | C0 | C1 reconstructed | C3-strong | C3.1 | C3.1-C0 | C3.1-C3 |
|---|---:|---:|---:|---:|---:|---:|
{metric_rows}

- Top-1 changed: {top1_changed}/{query_count} ({100*top1_changed/query_count:.2f}%)
- Hard-positive top-100 exits/entries: {hard_exits}/{hard_entries}; exit ratio {100*result['hard_positive_top100_exit_ratio']:.4f}%
- y_fp / y_joint_05 down-move: {100*result['y_fp_down_move_ratio']:.2f}% / {100*result['y_joint_05_down_move_ratio']:.2f}%
- Wrong-video centered-boundary bonus mean: {result['wrong_video_centered_boundary_bonus_mean']:.8f}
- Pearson / mean within-query Spearman: {result['score_correlations']['pearson_base_vs_c31']:.6f} / {result['score_correlations']['spearman_mean_within_query']:.6f}
- Formula max absolute error: {formula_error:.3g}
- Base/learned top-100 plus NMS exact: `{str(base_nms_exact).lower()}/{str(learned_nms_exact).lower()}`

No post-val adjustment was made. Stop here.
"""
    Path(args.output_md).parent.mkdir(parents=True,exist_ok=True)
    Path(args.output_md).write_text(md,encoding="utf-8")
    print(json.dumps(result,indent=2))


if __name__ == "__main__":
    main()
