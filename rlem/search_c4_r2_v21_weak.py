#!/usr/bin/env python
"""C4-r2-cal-v2.1 train-calib-only linear weak-fusion probe."""

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

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rlem.c4_lite_utils import ALL_METRICS, PRIMARY_METRICS, selection_score, write_json  # noqa: E402
from rlem.c4_protocol import C4Protocol, write_protocol_manifest  # noqa: E402
from rlem.c4_video_targets import frozen_c4_lite_row_scores  # noqa: E402
from rlem.grid_search_c4_lite import attach_eval_labels, eval_score, fast_metrics, fast_selected  # noqa: E402

_CACHE = _ARGS = _C4 = _C4_METRICS = _C4_ORDER = _ZS = _MOMENTS = None


def parse_args():
    p = argparse.ArgumentParser()
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
    p.add_argument("--reuse_grid_results", action="store_true")
    return p.parse_args()


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""): h.update(chunk)
    return h.hexdigest()


def make_configs():
    ranges = {
        "a_rel": [0.0, 0.025, 0.05, 0.075, 0.10, 0.125],
        "b_iou05": [0.0, 0.025, 0.05, 0.075, 0.10, 0.125],
        "c_iou07": [0.0, 0.025, 0.05, 0.075],
        "d_quality": [-0.10, -0.075, -0.05, -0.025, 0.0, 0.025],
    }
    out = []; seen = set()
    for ai, a in enumerate(ranges["a_rel"]):
        for bi, b in enumerate(ranges["b_iou05"]):
            for ci, c in enumerate(ranges["c_iou07"]):
                for di, d in enumerate(ranges["d_quality"]):
                    signature = (a, b, c, d)
                    if signature in seen: continue
                    seen.add(signature)
                    out.append({
                        "config_id": f"v21_{len(out):05d}", "a_rel": a,
                        "b_iou05": b, "c_iou07": c, "d_quality": d,
                        "grid_index": [ai, bi, ci, di],
                        "zero_residual_control": a == b == c == d == 0.0,
                        "v1_style_prior": a == 0.075 and b == 0.0 and c == 0.0 and d == -0.05,
                    })
    return out, ranges


def score_for(cfg):
    return (_C4 + cfg["a_rel"] * _ZS["rel"] + cfg["b_iou05"] * _ZS["iou05"] + cfg["c_iou07"] * _ZS["iou07"] + cfg["d_quality"] * _ZS["quality"]).astype(np.float32)


def pearson(score):
    n, sx, sxx = _MOMENTS; sy = float(np.sum(score, dtype=np.float64)); syy = float(np.sum(score.astype(np.float64) ** 2)); sxy = float(np.sum(_C4.astype(np.float64) * score.astype(np.float64)))
    cov = sxy / n - sx * sy / (n * n); vx = sxx / n - (sx / n) ** 2; vy = syy / n - (sy / n) ** 2
    return float(cov / math.sqrt(max(vx * vy, 1e-30)))


def evaluate(cfg):
    score = score_for(cfg); metrics, _, order = fast_metrics(_CACHE, score, _ARGS)
    delta = {k: float(metrics[k] - _C4_METRICS[k]) for k in ALL_METRICS}
    top1 = float(np.mean(order[:, 0] != _C4_ORDER[:, 0])); corr = pearson(score)
    primary_ok = all(delta[k] >= 0.0 for k in PRIMARY_METRICS)
    r100_ok = delta["0.5-r100"] >= 0.0 and delta["0.7-r100"] >= 0.0
    movement_ok = top1 <= 0.08 and corr >= 0.985
    return {
        **cfg, "metrics": metrics, "deltas_vs_frozen_c4_lite": delta,
        "selection_score_delta_vs_frozen_c4_lite": selection_score(metrics, _C4_METRICS),
        "all_eight_strictly_positive": all(delta[k] > 0.0 for k in ALL_METRICS),
        "six_primary_nonnegative": primary_ok, "r100_both_nonnegative": r100_ok,
        "top1_changed_ratio": top1, "pearson_c4_lite_candidate": corr,
        "movement_safe": movement_ok, "hard_constraints_pass": primary_ok and r100_ok and movement_ok,
        "official_val_used": False,
    }


def attach_neighbors(results, ranges):
    lookup = {tuple(r["grid_index"]): r for r in results}; sizes = [len(ranges[k]) for k in ["a_rel", "b_iou05", "c_iou07", "d_quality"]]
    for rec in results:
        passing = []
        for dim, size in enumerate(sizes):
            for step in (-1, 1):
                idx = list(rec["grid_index"]); idx[dim] += step
                if 0 <= idx[dim] < size:
                    other = lookup.get(tuple(idx))
                    if other and other["hard_constraints_pass"] and not other["zero_residual_control"]: passing.append(other["config_id"])
        active_boundary = False
        for dim, key in enumerate(["a_rel", "b_iou05", "c_iou07", "d_quality"]):
            if rec[key] != 0.0 and rec["grid_index"][dim] in {0, sizes[dim] - 1}: active_boundary = True
        rec["passing_neighbor_ids"] = sorted(set(passing)); rec["passing_neighbor_count"] = len(rec["passing_neighbor_ids"])
        rec["active_parameter_on_grid_boundary"] = active_boundary
        rec["isolated_boundary_point"] = active_boundary and not passing
        rec["robust_neighbor_supported"] = bool(passing)


def movement(cache, base, score, args):
    bs, bo, _ = fast_selected(cache, base, args); ns, no, _ = fast_selected(cache, score, args)
    bi = np.empty_like(bo, dtype=np.int16); ni = np.empty_like(no, dtype=np.int16); ranks = np.broadcast_to(np.arange(200, dtype=np.int16), bo.shape)
    np.put_along_axis(bi, bo, ranks, axis=1); np.put_along_axis(ni, no, ranks, axis=1)
    hard = cache["eval_y05"]; exits = int(np.sum(hard & (bi < 100) & (ni >= 100))); entries = int(np.sum(hard & (bi >= 100) & (ni < 100)))
    return {"relative_to": "frozen_C4_lite_c4_00394", "top1_changed_ratio": float(np.mean(bo[:, 0] != no[:, 0])), "pearson_c4_lite_candidate": float(np.corrcoef(bs.ravel(), ns.ravel())[0, 1]), "hard_positive_top100_exits": exits, "hard_positive_top100_entries": entries, "hard_positive_top100_exit_ratio": float(exits / max(int(hard.sum()), 1))}


def enrich(candidate, c31_metrics, c4_score, v1, current, cache, args):
    candidate["deltas_vs_frozen_c31"] = {k: candidate["metrics"][k] - c31_metrics[k] for k in ALL_METRICS}
    candidate["deltas_vs_c4_r2_v1_reference"] = {k: candidate["metrics"][k] - v1[k] for k in ALL_METRICS}
    candidate["deltas_vs_current_v2_cb"] = {k: candidate["metrics"][k] - current[k] for k in ALL_METRICS}
    candidate["movement_vs_frozen_c4_lite"] = movement(cache, c4_score, score_for(candidate), args)
    candidate["strictly_dominates_current_v2_cb"] = all(candidate["metrics"][k] >= current[k] for k in ALL_METRICS) and any(candidate["metrics"][k] > current[k] for k in ALL_METRICS)
    return candidate


def main():
    global _CACHE, _ARGS, _C4, _C4_METRICS, _C4_ORDER, _ZS, _MOMENTS
    args = parse_args(); protocol = C4Protocol(stage="c4_r2_grid_search", split=args.split, effective_top_n=args.effective_top_n, max_after_nms=args.max_after_nms, nms_thd=args.nms_thd); protocol.validate()
    with np.load(args.cache_npz, allow_pickle=True) as z: cache = {k: z[k] for k in z.files}
    attach_eval_labels(cache, args.gt_jsonl); c31 = cache["s_c31"].astype(np.float32); c4, _, _, _ = frozen_c4_lite_row_scores(cache)
    c31_metrics, _, _ = fast_metrics(cache, c31, args); c4_metrics, _, c4_order = fast_metrics(cache, c4, args)
    _, _, bc31 = eval_score(cache, c31, args); _, _, bc4 = eval_score(cache, c4, args)
    parity = {"c31": max(abs(round(c31_metrics[k], 2) - bc31[k]) for k in ALL_METRICS), "c4_lite": max(abs(round(c4_metrics[k], 2) - bc4[k]) for k in ALL_METRICS)}
    if max(parity.values()) > 1e-9: raise ValueError(parity)
    with np.load(args.logits_npz, allow_pickle=True) as z:
        if not np.array_equal(z["group_id"].astype(np.int64), cache["group_ids_sorted_unique"].astype(np.int64)): raise ValueError("logit/cache mismatch")
        logits = {k: z[k].astype(np.float32) for k in ["rel_logit", "iou05_logit", "iou07_logit", "quality_logit"]}
    row_gid = cache["row_group_id"].astype(np.int64); zs = {}; zstats = {}
    for short, key in [("rel", "rel_logit"), ("iou05", "iou05_logit"), ("iou07", "iou07_logit"), ("quality", "quality_logit")]:
        row = logits[key][row_gid]; mean, std = float(row.mean()), float(row.std()); zs[short] = ((row - mean) / std).astype(np.float32); zstats[key] = {"row_expanded_mean": mean, "row_expanded_std": std}
    _CACHE, _ARGS, _C4, _C4_METRICS, _C4_ORDER, _ZS = cache, args, c4, c4_metrics, c4_order, zs
    _MOMENTS = (len(c4), float(np.sum(c4, dtype=np.float64)), float(np.sum(c4.astype(np.float64) ** 2)))
    configs, ranges = make_configs(); out = Path(args.output_dir)
    if args.reuse_grid_results:
        existing = out / "grid_results.json"
        if not existing.exists(): raise FileNotFoundError(existing)
        results = json.load(open(existing, "r", encoding="utf-8"))["results"]
    else:
        with mp.get_context("fork").Pool(args.workers) as pool:
            results = list(tqdm(pool.imap_unordered(evaluate, configs, chunksize=1), total=len(configs), desc="C4-r2-cal-v2.1 weak fusion"))
    results.sort(key=lambda r: r["config_id"]); attach_neighbors(results, ranges)
    strict_pool = [r for r in results if r["hard_constraints_pass"] and r["all_eight_strictly_positive"] and not r["zero_residual_control"]]
    if not strict_pool: strict_pool = [r for r in results if r["hard_constraints_pass"] and not r["zero_residual_control"]]
    strict_best = max(strict_pool, key=lambda r: r["selection_score_delta_vs_frozen_c4_lite"])
    robust_pool = [r for r in strict_pool if r["robust_neighbor_supported"]]
    robust_best = max(robust_pool, key=lambda r: (r["passing_neighbor_count"], r["selection_score_delta_vs_frozen_c4_lite"])) if robust_pool else None
    trade_pool = [r for r in results if r["movement_safe"] and not r["zero_residual_control"]]
    trade_best = max(trade_pool, key=lambda r: r["selection_score_delta_vs_frozen_c4_lite"])
    v1 = json.load(open(ROOT / "c4_audit/C4_R2_CAL_CONSTRAINED_SEARCH_AUDIT.json"))["best_config"]["metrics"]
    current = json.load(open(ROOT / "c4_audit/C4_R2_CAL_V2_V2_CB_SEARCH_AUDIT.json"))["best_config"]["metrics"]
    strict_best = enrich(strict_best, c31_metrics, c4, v1, current, cache, args); trade_best = enrich(trade_best, c31_metrics, c4, v1, current, cache, args)
    if robust_best is not None: robust_best = enrich(robust_best, c31_metrics, c4, v1, current, cache, args)
    submission, _, bundled = eval_score(cache, score_for(strict_best), args); parity["strict_best"] = max(abs(round(strict_best["metrics"][k], 2) - bundled[k]) for k in ALL_METRICS)
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "grid_results.json", {"status": "PASS", "scope": "train_calib_only", "grid_size": len(results), "ranges": ranges, "z_stats": zstats, "controls": {"frozen_c31": c31_metrics, "frozen_c4_lite": c4_metrics, "c4_r2_v1_reference": v1, "current_v2_cb": current}, "best_strict_safe": strict_best, "best_robust_strict_safe": robust_best, "best_trade_off": trade_best, "results": results, "official_val_used": False})
    fields = ["config_id", "a_rel", "b_iou05", "c_iou07", "d_quality", "selection_score_delta_vs_frozen_c4_lite", "hard_constraints_pass", "all_eight_strictly_positive", "top1_changed_ratio", "pearson_c4_lite_candidate", "passing_neighbor_count", "active_parameter_on_grid_boundary"] + ALL_METRICS
    with open(out / "grid_results.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(fields)
        for r in results: w.writerow([r["metrics"].get(k, r.get(k, "")) if k in ALL_METRICS else r.get(k, "") for k in fields])
    write_json(out / "best_config.json", strict_best); write_json(out / "best_robust_strict_config.json", robust_best); write_json(out / "best_trade_off_config.json", trade_best); write_json(args.save_best_submission, submission, pretty=False)
    improves_selection = strict_best["selection_score_delta_vs_frozen_c4_lite"] > selection_score(current, c4_metrics)
    v1_prior = next(r for r in results if r["v1_style_prior"])
    audit = {"status": "PASS", "stage": "C4-r2-cal-v2.1 weak-fusion probe", "scope": "train_calib_only", "model_retrained": False, "capacity_search": False, "grid_size": len(results), "hard_constraint_pass_count": sum(r["hard_constraints_pass"] for r in results), "nonzero_hard_constraint_pass_count": sum(r["hard_constraints_pass"] and not r["zero_residual_control"] for r in results), "all_eight_positive_count": sum(r["all_eight_strictly_positive"] for r in results), "v1_style_prior_diagnostic": v1_prior, "best_strict_safe_config": strict_best, "best_robust_strict_safe_config": robust_best, "best_trade_off_config": trade_best, "strict_best_improves_selection_over_current_v2_cb": improves_selection, "strict_best_strictly_dominates_current_v2_cb": strict_best["strictly_dominates_current_v2_cb"], "weak_fusion_v2_improves_over_current_v2_cb": improves_selection and strict_best["strictly_dominates_current_v2_cb"], "fast_bundled_parity": parity, "logits_sha256": sha256(args.logits_npz), "official_val_used": False, "official_val_evidence_opened": False, "post_val_adjustment": False, "future_stages_used": {"official_val": False, "c4_main": False, "r2_head": False, "vs_head": False, "c5": False, "c6": False}, "stop_boundary": "STOP_AFTER_C4_R2_CAL_V21_WEAK_FUSION_TRAIN_CALIB"}
    write_protocol_manifest(args.audit_json, protocol, audit); print(json.dumps(audit, indent=2))


if __name__ == "__main__": main()
