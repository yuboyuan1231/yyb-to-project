#!/usr/bin/env python
"""Exact train-calib-only rank-preserving residual search for one V2 group."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from rlem.c4_lite_utils import ALL_METRICS, PRIMARY_METRICS, selection_score, write_json  # noqa: E402
from rlem.c4_protocol import C4Protocol, write_protocol_manifest  # noqa: E402
from rlem.c4_video_targets import frozen_c4_lite_row_scores  # noqa: E402
from rlem.grid_search_c4_lite import attach_eval_labels, eval_score, fast_metrics, fast_selected, _matrix  # noqa: E402

_CACHE = _ARGS = _C4 = _C4_METRICS = _C4_ORDER = _ZS = _BASE_MOMENTS = None


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--group", choices=["v2_ab", "v2_cb", "v2_acb"], required=True)
    p.add_argument("--cache_npz", required=True)
    p.add_argument("--logits_npz", required=True)
    p.add_argument("--gt_jsonl", required=True)
    p.add_argument("--dataset_config", required=True)
    p.add_argument("--split", choices=["train"], required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--audit_json", required=True)
    p.add_argument("--save_best_submission", required=True)
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--effective_top_n", type=int, default=100)
    p.add_argument("--max_after_nms", type=int, default=100)
    p.add_argument("--nms_thd", type=float, default=0.7)
    p.add_argument("--no_desc_type", action="store_true")
    return p.parse_args()


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""): h.update(chunk)
    return h.hexdigest()


def configs(group):
    tau_values = [0.025, 0.05, 0.075, 0.10]
    a_values = [0.0, 0.05, 0.075, 0.10]
    b_values = [0.0] if group == "v2_ab" else [0.0, 0.05, 0.075, 0.10]
    c_values = [0.0] if group == "v2_ab" else [0.0, 0.025, 0.05]
    d_values = [-0.05, -0.025, 0.0, 0.025]
    seen = set(); out = []
    for ti, tau in enumerate(tau_values):
        for ai, a in enumerate(a_values):
            for bi, b in enumerate(b_values):
                for ci, c in enumerate(c_values):
                    for di, d in enumerate(d_values):
                        signature = ("zero",) if a == b == c == d == 0.0 else (tau, a, b, c, d)
                        if signature in seen: continue
                        seen.add(signature)
                        out.append({
                            "config_id": f"{group}_{len(out):05d}", "group": group,
                            "tau": tau, "a_rel": a, "b_iou05": b,
                            "c_iou07": c, "d_quality": d,
                            "grid_index": [ti, ai, bi, ci, di],
                            "zero_residual_control": a == b == c == d == 0.0,
                        })
    ranges = {"tau": tau_values, "a_rel": a_values, "b_iou05": b_values, "c_iou07": c_values, "d_quality": d_values}
    raw_size = len(tau_values) * len(a_values) * len(b_values) * len(c_values) * len(d_values)
    return out, ranges, raw_size


def _score(cfg):
    inner = (
        cfg["a_rel"] * _ZS["rel"] + cfg["b_iou05"] * _ZS["iou05"]
        + cfg["c_iou07"] * _ZS["iou07"] + cfg["d_quality"] * _ZS["quality"]
    )
    return (_C4 + np.float32(cfg["tau"]) * np.tanh(inner)).astype(np.float32)


def _pearson_with_base(score):
    n, sx, sxx = _BASE_MOMENTS
    sy = float(np.sum(score, dtype=np.float64)); syy = float(np.sum(score.astype(np.float64) ** 2))
    sxy = float(np.sum(_C4.astype(np.float64) * score.astype(np.float64)))
    cov = sxy / n - (sx / n) * (sy / n)
    vx = sxx / n - (sx / n) ** 2; vy = syy / n - (sy / n) ** 2
    return float(cov / math.sqrt(max(vx * vy, 1e-30)))


def evaluate_config(cfg):
    score = _score(cfg); metrics, _, order = fast_metrics(_CACHE, score, _ARGS)
    deltas = {k: float(metrics[k] - _C4_METRICS[k]) for k in ALL_METRICS}
    top1 = float(np.mean(order[:, 0] != _C4_ORDER[:, 0])); pearson = _pearson_with_base(score)
    all_primary_nonnegative = all(deltas[k] >= 0.0 for k in PRIMARY_METRICS)
    r100_nonnegative = deltas["0.5-r100"] >= 0.0 and deltas["0.7-r100"] >= 0.0
    hard = all_primary_nonnegative and r100_nonnegative and top1 <= 0.08 and pearson >= 0.985
    return {
        **cfg, "metrics": metrics, "deltas_vs_frozen_c4_lite": deltas,
        "selection_score_delta_vs_frozen_c4_lite": selection_score(metrics, _C4_METRICS),
        "all_eight_strictly_positive": all(deltas[k] > 0.0 for k in ALL_METRICS),
        "all_six_primary_nonnegative": all_primary_nonnegative,
        "r100_both_nonnegative": r100_nonnegative,
        "top1_changed_ratio": top1, "pearson_c4_lite_v2": pearson,
        "hard_constraints_pass": hard, "official_val_used": False,
    }


def add_neighbors(results, ranges):
    lookup = {tuple(r["grid_index"]): r for r in results}
    dims = [len(ranges[k]) for k in ["tau", "a_rel", "b_iou05", "c_iou07", "d_quality"]]
    for rec in results:
        idx = rec["grid_index"]; passing = []
        for dim, size in enumerate(dims):
            for delta in (-1, 1):
                other = list(idx); other[dim] += delta
                if 0 <= other[dim] < size:
                    candidate = lookup.get(tuple(other))
                    if candidate and candidate["hard_constraints_pass"] and not candidate["zero_residual_control"]:
                        passing.append(candidate["config_id"])
        active_boundary = rec["grid_index"][0] in {0, dims[0] - 1}
        for dim, key in enumerate(["a_rel", "b_iou05", "c_iou07", "d_quality"], start=1):
            if rec[key] != 0.0 and rec["grid_index"][dim] in {0, dims[dim] - 1}:
                active_boundary = True
        rec["passing_neighbor_ids"] = sorted(set(passing))
        rec["passing_neighbor_count"] = len(rec["passing_neighbor_ids"])
        rec["active_parameter_on_grid_boundary"] = active_boundary
        rec["robust_neighbor_supported"] = len(passing) > 0
        rec["isolated_boundary_point"] = active_boundary and len(passing) == 0


def best_key(rec):
    return (
        int(rec["all_eight_strictly_positive"]),
        int(rec["robust_neighbor_supported"]),
        rec["passing_neighbor_count"],
        rec["selection_score_delta_vs_frozen_c4_lite"],
    )


def best_movement(cache, base, score, args):
    base_scores, base_order, _ = fast_selected(cache, base, args); scores, order, _ = fast_selected(cache, score, args)
    base_inv = np.empty_like(base_order, dtype=np.int16); inv = np.empty_like(order, dtype=np.int16)
    ranks = np.broadcast_to(np.arange(200, dtype=np.int16), order.shape)
    np.put_along_axis(base_inv, base_order, ranks, axis=1); np.put_along_axis(inv, order, ranks, axis=1)
    hard = cache["eval_y05"]; exits = int(np.sum(hard & (base_inv < 100) & (inv >= 100))); entries = int(np.sum(hard & (base_inv >= 100) & (inv < 100)))
    return {"relative_to": "frozen_C4_lite_c4_00394", "queries": len(order),
        "top1_changed_ratio": float(np.mean(base_order[:, 0] != order[:, 0])),
        "pearson_c4_lite_v2": float(np.corrcoef(base_scores.ravel(), scores.ravel())[0, 1]),
        "hard_positive_top100_exits": exits, "hard_positive_top100_entries": entries,
        "hard_positive_top100_exit_ratio": float(exits / max(int(hard.sum()), 1))}


def main():
    global _CACHE, _ARGS, _C4, _C4_METRICS, _C4_ORDER, _ZS, _BASE_MOMENTS
    args = parse_args(); protocol = C4Protocol(stage="c4_r2_grid_search", split=args.split, effective_top_n=args.effective_top_n, max_after_nms=args.max_after_nms, nms_thd=args.nms_thd); protocol.validate()
    with np.load(args.cache_npz, allow_pickle=True) as z: cache = {k: z[k] for k in z.files}
    attach_eval_labels(cache, args.gt_jsonl); c31 = cache["s_c31"].astype(np.float32); c4, _, _, _ = frozen_c4_lite_row_scores(cache)
    c31_metrics, _, _ = fast_metrics(cache, c31, args); c4_metrics, _, c4_order = fast_metrics(cache, c4, args)
    _, _, bundled_c31 = eval_score(cache, c31, args); _, _, bundled_c4 = eval_score(cache, c4, args)
    parity = {"c31": max(abs(round(c31_metrics[k], 2) - bundled_c31[k]) for k in ALL_METRICS), "c4_lite": max(abs(round(c4_metrics[k], 2) - bundled_c4[k]) for k in ALL_METRICS)}
    if max(parity.values()) > 1e-9: raise ValueError(f"Control evaluator mismatch: {parity}")
    with np.load(args.logits_npz, allow_pickle=True) as z:
        if not np.array_equal(z["group_id"].astype(np.int64), cache["group_ids_sorted_unique"].astype(np.int64)): raise ValueError("Logit/cache group mismatch")
        dense = {k: z[k].astype(np.float32) for k in ["rel_logit", "iou05_logit", "iou07_logit", "quality_logit"]}
    row_gid = cache["row_group_id"].astype(np.int64); zs = {}; stats = {}
    for short, key in [("rel", "rel_logit"), ("iou05", "iou05_logit"), ("iou07", "iou07_logit"), ("quality", "quality_logit")]:
        row = dense[key][row_gid]; mean, std = float(row.mean()), float(row.std())
        zs[short] = ((row - mean) / std).astype(np.float32); stats[key] = {"row_expanded_mean": mean, "row_expanded_std": std}
    cfgs, ranges, raw_grid = configs(args.group)
    _CACHE, _ARGS, _C4, _C4_METRICS, _C4_ORDER, _ZS = cache, args, c4, c4_metrics, c4_order, zs
    _BASE_MOMENTS = (len(c4), float(np.sum(c4, dtype=np.float64)), float(np.sum(c4.astype(np.float64) ** 2)))
    if args.workers <= 1: results = [evaluate_config(c) for c in tqdm(cfgs, desc=f"{args.group} rank-preserving search")]
    else:
        with mp.get_context("fork").Pool(args.workers) as pool:
            results = list(tqdm(pool.imap_unordered(evaluate_config, cfgs, chunksize=1), total=len(cfgs), desc=f"{args.group} rank-preserving search"))
    results.sort(key=lambda r: r["config_id"]); add_neighbors(results, ranges)
    eligible = [r for r in results if r["hard_constraints_pass"] and not r["zero_residual_control"]]
    best = max(eligible, key=best_key) if eligible else max(results, key=lambda r: r["selection_score_delta_vs_frozen_c4_lite"])
    best_score = _score(best); movement = best_movement(cache, c4, best_score, args); best["movement_vs_frozen_c4_lite"] = movement; best["z_stats_train_calib"] = stats
    submission, _, bundled_best = eval_score(cache, best_score, args); parity["best"] = max(abs(round(best["metrics"][k], 2) - bundled_best[k]) for k in ALL_METRICS)
    best["deltas_vs_frozen_c31"] = {k: best["metrics"][k] - c31_metrics[k] for k in ALL_METRICS}
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    write_json(str(out / "grid_results.json"), {"status": "PASS", "scope": "train_calib_only", "group": args.group, "raw_grid_size": raw_grid, "deduplicated_grid_size": len(results), "ranges": ranges, "controls": {"frozen_c31": c31_metrics, "frozen_c4_lite": c4_metrics}, "best": best, "results": results, "official_val_used": False})
    fields = ["config_id", "tau", "a_rel", "b_iou05", "c_iou07", "d_quality", "selection_score_delta_vs_frozen_c4_lite", "hard_constraints_pass", "all_eight_strictly_positive", "top1_changed_ratio", "pearson_c4_lite_v2", "passing_neighbor_count", "active_parameter_on_grid_boundary"] + ALL_METRICS
    with open(out / "grid_results.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(fields)
        for r in results: w.writerow([r["metrics"].get(k, r.get(k, "")) if k in ALL_METRICS else r.get(k, "") for k in fields])
    write_json(str(out / "best_config.json"), best); write_json(args.save_best_submission, submission, pretty=False)
    robust_pass = best["hard_constraints_pass"] and best["robust_neighbor_supported"] and not best["isolated_boundary_point"]
    audit = {"status": "PASS", "group": args.group, "scope": "train_calib_only", "raw_grid_size": raw_grid, "deduplicated_grid_size": len(results), "workers": args.workers, "hard_constraint_pass_count": sum(r["hard_constraints_pass"] and not r["zero_residual_control"] for r in results), "all_eight_positive_count": sum(r["all_eight_strictly_positive"] for r in results), "baseline_primary": "frozen_C4_lite_c4_00394", "baseline_secondary": "frozen_C3.1", "raw_base_log_selection": False, "best_config": best, "movement_vs_frozen_c4_lite": movement, "robustness_pass": robust_pass, "fast_bundled_parity": parity, "logits_sha256": sha256(args.logits_npz), "recommend_freeze_review": robust_pass and best["all_eight_strictly_positive"], "official_val_used": False, "post_val_adjustment": False, "future_stages_used": {"official_val": False, "c4_main": False, "r2_head": False, "vs_head": False, "c5": False, "c6": False}}
    write_protocol_manifest(args.audit_json, protocol, audit); print(json.dumps(audit, indent=2))


if __name__ == "__main__": main()
