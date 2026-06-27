#!/usr/bin/env python
"""Train-calib C5-lite-prior residual search.

This search uses C4_final (C4-r2-cal-v2.1) as the primary baseline and only
chooses weights on train_calib.  It reports both VCMR metrics and localization-
specific diagnostics; without localization improvement it cannot be promoted as
retrieval -> localization.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import multiprocessing as mp
import os
import sys
from pathlib import Path
from typing import Dict, Iterator, List

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from rlem.c5_prior_utils import (  # noqa: E402
    ALL_METRICS,
    PRIMARY_METRICS,
    C5_PRIOR_TERM_NAMES,
    attach_eval_labels,
    fast_metrics,
    load_cache,
    localization_diagnostics,
    metric_deltas,
    movement_diagnostics,
    prior_diagnostics,
    promotion_flags,
    score_with_config,
    selection_score,
    sha256,
    standardized_term_matrix,
    write_json,
)
from rlem.c4_video_targets import frozen_c4_lite_row_scores  # noqa: E402


_CACHE = _BASELINE = _Z_TERMS = _BASE_METRICS = _ARGS = None


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cache_npz", required=True)
    p.add_argument("--features_npz", required=True)
    p.add_argument("--term_stats_json", required=True)
    p.add_argument("--gt_jsonl", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--effective_top_n", type=int, default=100)
    p.add_argument("--max_after_nms", type=int, default=100)
    p.add_argument("--nms_thd", type=float, default=0.7)
    p.add_argument("--grid_mode", choices=["small", "medium"], default="small")
    p.add_argument("--workers", type=int, default=max(1, min(12, os.cpu_count() or 1)))
    return p.parse_args()


def load_features(path: str) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as payload:
        return {key: payload[key] for key in payload.files}


def generate_configs(mode: str) -> Iterator[Dict[str, float]]:
    """Generate a bounded, interpretable family grid.

    Families are intentionally low-dimensional.  This is not a capacity search;
    it is a small train_calib selection over prior-feature residuals.
    """
    lambdas = [0.05, 0.075, 0.10, 0.125] if mode == "small" else [0.025, 0.05, 0.075, 0.10, 0.125, 0.15]
    pos = [0.0, 0.05, 0.075, 0.10]
    neg = [0.0, 0.05, 0.075]
    families = []
    # Family A: context prior mass/peak/boundary.
    for a, b, c in itertools.product(pos, pos[:3], pos[:3]):
        cfg = {f"w_{name}": 0.0 for name in C5_PRIOR_TERM_NAMES}
        cfg.update({"w_retctx_mass": a, "w_retctx_peak_inside": b, "w_retctx_boundary_agree": c})
        cfg["family"] = "ctx_prior"
        families.append(cfg)
    # Family B: boundary prior mass/peak/boundary.
    for a, b, c in itertools.product(pos, pos[:3], pos[:3]):
        cfg = {f"w_{name}": 0.0 for name in C5_PRIOR_TERM_NAMES}
        cfg.update({"w_retbd_mass": a, "w_retbd_peak_inside": b, "w_retbd_boundary_agree": c})
        cfg["family"] = "bd_prior"
        families.append(cfg)
    # Family C: combined context+boundary with false-positive suppression.
    for a, b, c, d in itertools.product(pos[:3], pos[:3], neg, neg[:2]):
        cfg = {f"w_{name}": 0.0 for name in C5_PRIOR_TERM_NAMES}
        cfg.update({
            "w_retctx_mass": a,
            "w_retbd_mass": b,
            "w_low_conf_efp_neg": c,
            "w_low_conf_unc_neg": d,
        })
        cfg["family"] = "ctx_bd_suppress"
        families.append(cfg)
    # Family D: center/boundary agreement + retrieval rank signals.
    for a, b, c, d in itertools.product(pos[:3], pos[:3], pos[:3], pos[:2]):
        cfg = {f"w_{name}": 0.0 for name in C5_PRIOR_TERM_NAMES}
        cfg.update({
            "w_retctx_center_proximity": a,
            "w_retbd_center_proximity": b,
            "w_margin_z_x_retbd_mass": c,
            "w_rank_inv_x_retbd_mass": d,
        })
        cfg["family"] = "center_rank"
        families.append(cfg)
    # Zero control.
    zero = {f"w_{name}": 0.0 for name in C5_PRIOR_TERM_NAMES}
    zero["family"] = "zero_control"
    families.append(zero)
    seen_base = set()
    seen_effective = set()
    idx = 0
    for base in families:
        key = tuple((name, float(base.get(f"w_{name}", 0.0))) for name in C5_PRIOR_TERM_NAMES)
        if key in seen_base:
            continue
        seen_base.add(key)
        for lam in lambdas:
            effective = tuple(round(float(lam) * float(base.get(f"w_{name}", 0.0)), 12) for name in C5_PRIOR_TERM_NAMES)
            if effective in seen_effective:
                continue
            seen_effective.add(effective)
            cfg = dict(base)
            cfg["lambda_scale"] = float(lam)
            cfg["center_residual"] = False
            cfg["effective_signature"] = list(effective)
            cfg["config_id"] = f"c5p_{idx:05d}"
            idx += 1
            yield cfg


def _fast_record(cfg):
    score = score_with_config(_BASELINE, _Z_TERMS, cfg)
    metrics, _, _ = fast_metrics(
        _CACHE, score, _ARGS.effective_top_n, _ARGS.max_after_nms, _ARGS.nms_thd
    )
    deltas = metric_deltas(metrics, _BASE_METRICS)
    move = movement_diagnostics(_CACHE, _BASELINE, score)
    six = all(deltas.get(key, -1e9) >= -1e-9 for key in PRIMARY_METRICS)
    r100 = all(deltas.get(key, -1e9) >= -1e-9 for key in ("0.5-r100", "0.7-r100"))
    safe = (
        move["hard_positive_top100_exit_ratio"] <= 0.005
        and move["top1_changed_ratio"] <= 0.25
        and move["pearson_base_candidate"] >= 0.98
    )
    return {
        "config": cfg,
        "metrics": metrics,
        "deltas": deltas,
        "movement": move,
        "selection_score_delta_vs_c4_final": selection_score(metrics, _BASE_METRICS),
        "six_primary_nonnegative": bool(six),
        "r100_both_nonnegative": bool(r100),
        "movement_safe": bool(safe),
        "vcmr_movement_gate": bool(six and r100 and safe),
    }


def _detailed_record(rec, cache, baseline, z_terms, features):
    score = score_with_config(baseline, z_terms, rec["config"])
    loc = localization_diagnostics(cache, baseline, score)
    prior = prior_diagnostics(cache, features, baseline, score)
    oracle_positive = loc["oracle_video_r1_05_delta"] > 0.0 or loc["oracle_video_r1_07_delta"] > 0.0
    rank_improved = loc["best_iou_span_rank_delta_mean"] < 0.0
    positive_trend = loc["positive_span_upward_ratio"] > loc["positive_span_downward_ratio"]
    negative_trend = loc["hard_negative_downward_ratio"] > loc["hard_negative_upward_ratio"]
    rec.update({
        "localization": loc,
        "prior": prior,
        "oracle_localization_positive": bool(oracle_positive),
        "best_iou_span_rank_improved": bool(rank_improved),
        "positive_span_upward_trend": bool(positive_trend),
        "hard_negative_downward_trend": bool(negative_trend),
        "localization_gate": bool(oracle_positive and rank_improved and positive_trend and negative_trend),
    })
    rec["core_promotion_gate"] = bool(rec["vcmr_movement_gate"] and rec["localization_gate"])
    loc_bonus = (
        0.25 * max(0.0, loc["oracle_video_r1_05_delta"])
        + 0.25 * max(0.0, loc["oracle_video_r1_07_delta"])
        + 25.0 * max(0.0, loc["selected_span_miou_delta"])
        + 0.05 * max(0.0, -loc["best_iou_span_rank_delta_mean"])
    )
    rec["selection_score_with_localization_bonus"] = float(
        rec["selection_score_delta_vs_c4_final"] + loc_bonus
    )
    return rec


def _attach_robustness(records):
    core = [rec for rec in records if rec.get("core_promotion_gate")]
    for rec in records:
        cfg = rec["config"]
        signature = np.asarray(cfg["effective_signature"], dtype=np.float64)
        candidates = []
        for other in core:
            if other is rec or other["config"]["family"] != cfg["family"]:
                continue
            distance = float(np.sum(np.abs(signature - np.asarray(other["config"]["effective_signature"], dtype=np.float64))))
            if distance > 0:
                candidates.append((distance, other["config"]["config_id"]))
        candidates.sort()
        nearest = candidates[:4]
        min_distance = nearest[0][0] if nearest else None
        neighbors = [cid for dist, cid in nearest if min_distance is not None and dist <= min_distance + 1e-12]
        active_weights = [abs(float(cfg.get(f"w_{name}", 0.0))) for name in C5_PRIOR_TERM_NAMES]
        active_boundary = (
            cfg["lambda_scale"] in ({0.05, 0.125} if _ARGS.grid_mode == "small" else {0.025, 0.15})
            or any(value >= 0.10 - 1e-12 for value in active_weights)
        )
        rec["passing_neighbor_ids"] = neighbors
        rec["passing_neighbor_count"] = len(neighbors)
        rec["active_parameter_on_grid_boundary"] = bool(active_boundary)
        rec["isolated_boundary_point"] = bool(active_boundary and not neighbors)
        rec["promotion_candidate"] = bool(rec.get("core_promotion_gate") and not rec["isolated_boundary_point"])


def _legacy_main_unused():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache = load_cache(args.cache_npz)
    attach_eval_labels(cache, args.gt_jsonl)
    features = load_features(args.features_npz)
    stats_manifest = json.loads(Path(args.term_stats_json).read_text(encoding="utf-8"))
    if stats_manifest.get("official_val_used") is not False:
        raise ValueError("C5-lite-prior stats must come from train_fit only")
    term_stats = stats_manifest["term_stats"]
    baseline = features["s_c4_final"].astype(np.float32)
    z_terms = standardized_term_matrix(features, term_stats)
    base_metrics, _, _ = fast_metrics(cache, baseline, args.effective_top_n, args.max_after_nms, args.nms_thd)
    base_loc = localization_diagnostics(cache, baseline, baseline)
    rows: List[Dict] = []
    best = None
    for cfg in generate_configs(args.grid_mode):
        score = score_with_config(baseline, z_terms, cfg)
        metrics, _, _ = fast_metrics(cache, score, args.effective_top_n, args.max_after_nms, args.nms_thd)
        loc = localization_diagnostics(cache, baseline, score)
        move = movement_diagnostics(cache, baseline, score)
        prior = prior_diagnostics(cache, features, baseline, score)
        flags = promotion_flags(metrics, base_metrics, loc, move)
        deltas = metric_deltas(metrics, base_metrics)
        sel = selection_score(metrics, base_metrics)
        loc_bonus = 0.0
        loc_bonus += 0.25 * max(0.0, loc.get("oracle_video_r1_05_delta", 0.0))
        loc_bonus += 0.25 * max(0.0, loc.get("oracle_video_r1_07_delta", 0.0))
        loc_bonus += 25.0 * max(0.0, loc.get("selected_span_miou_delta", 0.0))
        # Selection prioritizes non-degradation + localization evidence.
        score_select = float(sel + loc_bonus)
        rec = {
            "config_id": cfg["config_id"],
            "family": cfg["family"],
            "lambda_scale": cfg["lambda_scale"],
            "selection_score_delta_vs_c4_final": sel,
            "selection_score_with_localization_bonus": score_select,
            **{f"delta_{k}": v for k, v in deltas.items()},
            **{f"metric_{k}": float(metrics[k]) for k in metrics},
            **flags,
            "top1_changed_ratio": move["top1_changed_ratio"],
            "hard_positive_top100_exit_ratio": move["hard_positive_top100_exit_ratio"],
            "pearson_base_candidate": move["pearson_base_candidate"],
            "oracle_video_r1_05_delta": loc["oracle_video_r1_05_delta"],
            "oracle_video_r1_07_delta": loc["oracle_video_r1_07_delta"],
            "selected_span_miou_delta": loc["selected_span_miou_delta"],
            "best_iou_span_rank_delta_mean": loc["best_iou_span_rank_delta_mean"],
            "positive_span_upward_ratio": loc["positive_span_upward_ratio"],
            "positive_span_downward_ratio": loc["positive_span_downward_ratio"],
            "hard_negative_downward_ratio": loc["hard_negative_downward_ratio"],
            "selected_retloc_peak_inside_ratio": prior["selected_retloc_peak_inside_ratio"],
        }
        for name in C5_PRIOR_TERM_NAMES:
            rec[f"w_{name}"] = float(cfg.get(f"w_{name}", 0.0))
        rows.append(rec)
        if flags["promotion_candidate"]:
            if best is None or score_select > best["selection_score_with_localization_bonus"]:
                best = {"record": rec, "config": cfg, "metrics": metrics, "deltas": deltas, "localization": loc, "movement": move, "prior": prior, "flags": flags}
    if best is None:
        # Keep the best diagnostic config, but mark as not promotable.
        rows_sorted = sorted(rows, key=lambda r: r["selection_score_with_localization_bonus"], reverse=True)
        top = rows_sorted[0]
        cfg = {"config_id": top["config_id"], "family": top["family"], "lambda_scale": top["lambda_scale"], "center_residual": True}
        for name in C5_PRIOR_TERM_NAMES:
            cfg[f"w_{name}"] = float(top.get(f"w_{name}", 0.0))
        score = score_with_config(baseline, z_terms, cfg)
        metrics, _, _ = fast_metrics(cache, score, args.effective_top_n, args.max_after_nms, args.nms_thd)
        best = {
            "record": top,
            "config": cfg,
            "metrics": metrics,
            "deltas": metric_deltas(metrics, base_metrics),
            "localization": localization_diagnostics(cache, baseline, score),
            "movement": movement_diagnostics(cache, baseline, score),
            "prior": prior_diagnostics(cache, features, baseline, score),
            "flags": promotion_flags(metrics, base_metrics, localization_diagnostics(cache, baseline, score), movement_diagnostics(cache, baseline, score)),
        }
    grid_csv = out_dir / "c5_lite_prior_grid_results.csv"
    if rows:
        fields = list(rows[0].keys())
        with open(grid_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    grid_json = out_dir / "c5_lite_prior_grid_results.json"
    best_json = out_dir / "best_config.json"
    audit_json = out_dir / "c5_lite_prior_train_calib_audit.json"
    write_json(str(grid_json), rows)
    write_json(str(best_json), best["config"])
    audit = {
        "status": "PASS" if best["flags"].get("promotion_candidate") else "NO_PROMOTION",
        "stage": "C5-lite-prior train_calib search",
        "scope": "train_calib_only",
        "official_val_used": False,
        "post_val_adjustment": False,
        "primary_baseline": "C4_final = C4-r2-cal-v2.1 v21_00444",
        "baseline_metrics": base_metrics,
        "baseline_localization": base_loc,
        "best_config": best["config"],
        "best_metrics": best["metrics"],
        "best_deltas_vs_c4_final": best["deltas"],
        "best_localization": best["localization"],
        "best_movement": best["movement"],
        "best_prior_diagnostics": best["prior"],
        "promotion_flags": best["flags"],
        "grid_size": len(rows),
        "artifacts": {
            "cache_npz": {"path": args.cache_npz, "sha256": sha256(args.cache_npz)},
            "features_npz": {"path": args.features_npz, "sha256": sha256(args.features_npz)},
            "term_stats_json": {"path": args.term_stats_json, "sha256": sha256(args.term_stats_json)},
            "gt_jsonl": {"path": args.gt_jsonl, "sha256": sha256(args.gt_jsonl)},
            "grid_csv": {"path": str(grid_csv), "sha256": sha256(str(grid_csv))},
            "grid_json": {"path": str(grid_json), "sha256": sha256(str(grid_json))},
            "best_config_json": {"path": str(best_json), "sha256": sha256(str(best_json))},
        },
        "retrieval_constants": {"effective_top_n": args.effective_top_n, "nms_thd": args.nms_thd, "max_after_nms": args.max_after_nms},
        "used": {"C5-lite-prior": True, "C5-main": False, "C6": False, "CONQUER_modified": False},
    }
    write_json(str(audit_json), audit)
    print(json.dumps({"status": audit["status"], "best_config": best["config"].get("config_id"), "audit": str(audit_json)}, indent=2))


def main():
    global _CACHE, _BASELINE, _Z_TERMS, _BASE_METRICS, _ARGS
    args = parse_args()
    if args.effective_top_n != 100 or args.max_after_nms != 100 or abs(args.nms_thd - 0.7) > 1e-12:
        raise ValueError("C5-lite-prior requires top_n=100, NMS=0.7, max_after_nms=100")
    out_dir = Path(args.output_dir)
    targets = [
        out_dir / "c5_lite_prior_grid_results.csv",
        out_dir / "c5_lite_prior_grid_results.json",
        out_dir / "best_config.json",
        out_dir / "c5_lite_prior_train_calib_audit.json",
        out_dir / "localization_diagnostics.json",
        out_dir / "movement_diagnostics.json",
    ]
    existing = [str(path) for path in targets if path.exists()]
    if existing:
        raise FileExistsError(existing)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache = load_cache(args.cache_npz)
    attach_eval_labels(cache, args.gt_jsonl)
    features = load_features(args.features_npz)
    if not np.array_equal(features["row_group_id"].astype(np.int64), cache["row_group_id"].astype(np.int64)):
        raise ValueError("C5 features/cache row alignment mismatch")
    stats_manifest = json.loads(Path(args.term_stats_json).read_text(encoding="utf-8"))
    if stats_manifest.get("scope") != "train_fit_only" or stats_manifest.get("official_val_used") is not False:
        raise ValueError("C5-lite-prior stats must be frozen on train_fit only")
    baseline = features["s_c4_final"].astype(np.float32)
    z_terms = standardized_term_matrix(features, stats_manifest["term_stats"])
    base_metrics, _, _ = fast_metrics(cache, baseline, 100, 100, 0.7)
    c4_lite_score, _, _, _ = frozen_c4_lite_row_scores(cache)
    c4_lite_metrics, _, _ = fast_metrics(cache, c4_lite_score, 100, 100, 0.7)
    c31_metrics, _, _ = fast_metrics(cache, cache["s_c31"].astype(np.float32), 100, 100, 0.7)
    base_loc = localization_diagnostics(cache, baseline, baseline)
    configs = list(generate_configs(args.grid_mode))
    _CACHE, _BASELINE, _Z_TERMS, _BASE_METRICS, _ARGS = cache, baseline, z_terms, base_metrics, args
    if args.workers <= 1:
        records = [_fast_record(cfg) for cfg in configs]
    else:
        with mp.get_context("fork").Pool(args.workers) as pool:
            records = list(pool.imap_unordered(_fast_record, configs, chunksize=1))
    records.sort(key=lambda rec: rec["config"]["config_id"])

    detailed = [rec for rec in records if rec["vcmr_movement_gate"]]
    if not detailed:
        detailed = sorted(records, key=lambda rec: rec["selection_score_delta_vs_c4_final"], reverse=True)[:10]
    for rec in detailed:
        _detailed_record(rec, cache, baseline, z_terms, features)
    _attach_robustness(records)
    promotable = [rec for rec in records if rec.get("promotion_candidate")]
    if promotable:
        best = max(promotable, key=lambda rec: (
            rec.get("passing_neighbor_count", 0),
            rec.get("selection_score_with_localization_bonus", -1e9),
        ))
        status = "PASS"
    else:
        best = max(detailed, key=lambda rec: rec.get("selection_score_with_localization_bonus", -1e9))
        status = "NO_PROMOTION"

    best["deltas_vs_c4_lite"] = metric_deltas(best["metrics"], c4_lite_metrics)
    best["deltas_vs_c31"] = metric_deltas(best["metrics"], c31_metrics)
    best_flags = {
        "six_primary_nonnegative": best["six_primary_nonnegative"],
        "r100_both_nonnegative": best["r100_both_nonnegative"],
        "oracle_localization_positive": best.get("oracle_localization_positive", False),
        "best_iou_span_rank_improved": best.get("best_iou_span_rank_improved", False),
        "positive_span_upward_trend": best.get("positive_span_upward_trend", False),
        "hard_negative_downward_trend": best.get("hard_negative_downward_trend", False),
        "movement_safe": best["movement_safe"],
        "isolated_boundary_point": best.get("isolated_boundary_point", True),
        "promotion_candidate": best.get("promotion_candidate", False),
    }

    rows: List[Dict] = []
    for rec in records:
        cfg, move = rec["config"], rec["movement"]
        row = {
            "config_id": cfg["config_id"], "family": cfg["family"],
            "lambda_scale": cfg["lambda_scale"],
            "selection_score_delta_vs_c4_final": rec["selection_score_delta_vs_c4_final"],
            "selection_score_with_localization_bonus": rec.get("selection_score_with_localization_bonus"),
            "six_primary_nonnegative": rec["six_primary_nonnegative"],
            "r100_both_nonnegative": rec["r100_both_nonnegative"],
            "movement_safe": rec["movement_safe"],
            "localization_gate": rec.get("localization_gate"),
            "promotion_candidate": rec.get("promotion_candidate", False),
            "passing_neighbor_count": rec.get("passing_neighbor_count", 0),
            "active_parameter_on_grid_boundary": rec.get("active_parameter_on_grid_boundary", False),
            "isolated_boundary_point": rec.get("isolated_boundary_point", False),
            "top1_changed_ratio": move["top1_changed_ratio"],
            "hard_positive_top100_exits": move["hard_positive_top100_exits"],
            "hard_positive_top100_entries": move["hard_positive_top100_entries"],
            "pearson_base_candidate": move["pearson_base_candidate"],
            **{f"delta_{key}": value for key, value in rec["deltas"].items()},
            **{f"metric_{key}": value for key, value in rec["metrics"].items()},
        }
        loc = rec.get("localization", {})
        for key in (
            "oracle_video_r1_05_delta", "oracle_video_r1_07_delta",
            "selected_span_miou_delta", "best_iou_span_rank_delta_mean",
            "positive_span_upward_ratio", "positive_span_downward_ratio",
            "hard_negative_downward_ratio", "hard_negative_upward_ratio",
            "affected_same_video_query_ratio",
        ):
            row[key] = loc.get(key)
        for name in C5_PRIOR_TERM_NAMES:
            row[f"w_{name}"] = float(cfg.get(f"w_{name}", 0.0))
        rows.append(row)

    grid_csv, grid_json, best_json, audit_json, loc_json, move_json = targets
    fields = list(rows[0].keys())
    with open(grid_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    write_json(str(grid_json), {
        "status": status, "grid_mode": args.grid_mode,
        "raw_grid_note": "equivalent effective coefficients deduplicated",
        "deduplicated_grid_size": len(records),
        "detailed_localization_config_count": len(detailed),
        "records": rows,
    })
    write_json(str(best_json), best["config"])
    write_json(str(loc_json), best.get("localization", {}))
    write_json(str(move_json), best["movement"])
    audit = {
        "status": status,
        "stage": "C5-lite-prior train_calib search",
        "scope": "train_calib_only",
        "official_val_used": False,
        "post_val_adjustment": False,
        "score_grid_on_val": False,
        "primary_baseline": "C4_final = C4-r2-cal-v2.1 v21_00444",
        "secondary_baselines": {"C4_lite": c4_lite_metrics, "C3.1": c31_metrics},
        "baseline_metrics": base_metrics,
        "baseline_localization": base_loc,
        "best_config": best["config"],
        "best_metrics": best["metrics"],
        "best_deltas_vs_c4_final": best["deltas"],
        "best_deltas_vs_c4_lite": best["deltas_vs_c4_lite"],
        "best_deltas_vs_c31": best["deltas_vs_c31"],
        "best_localization": best.get("localization"),
        "best_movement": best["movement"],
        "best_prior_diagnostics": best.get("prior"),
        "neighbor_robustness": {
            "passing_neighbor_count": best.get("passing_neighbor_count", 0),
            "passing_neighbor_ids": best.get("passing_neighbor_ids", []),
            "active_parameter_on_grid_boundary": best.get("active_parameter_on_grid_boundary", False),
            "isolated_boundary_point": best.get("isolated_boundary_point", True),
        },
        "promotion_flags": best_flags,
        "deduplicated_grid_size": len(records),
        "detailed_localization_config_count": len(detailed),
        "workers": args.workers,
        "artifacts": {
            "cache_npz": {"path": args.cache_npz, "sha256": sha256(args.cache_npz)},
            "features_npz": {"path": args.features_npz, "sha256": sha256(args.features_npz)},
            "term_stats_json": {"path": args.term_stats_json, "sha256": sha256(args.term_stats_json)},
            "gt_jsonl": {"path": args.gt_jsonl, "sha256": sha256(args.gt_jsonl)},
            "grid_csv": {"path": str(grid_csv), "sha256": sha256(str(grid_csv))},
            "grid_json": {"path": str(grid_json), "sha256": sha256(str(grid_json))},
            "best_config_json": {"path": str(best_json), "sha256": sha256(str(best_json))},
        },
        "retrieval_constants": {"effective_top_n": 100, "nms_thd": 0.7, "max_after_nms": 100},
        "used": {"C5-lite-prior": True, "C5-main": False, "C6": False, "CONQUER_modified": False},
    }
    write_json(str(audit_json), audit)
    print(json.dumps({"status": status, "best_config": best["config"]["config_id"], "audit": str(audit_json)}, indent=2))


if __name__ == "__main__":
    main()
