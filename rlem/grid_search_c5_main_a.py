#!/usr/bin/env python
"""Train-calib-only search for C5-main-inference-A endpoint injection."""

from __future__ import annotations

import argparse
import csv
import json
import math
import multiprocessing as mp
import os
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from rlem.c4_lite_utils import PRIMARY_METRICS, selection_score  # noqa: E402
from rlem.c4_video_targets import frozen_c4_lite_row_scores  # noqa: E402
from rlem.c5_main_a_utils import (  # noqa: E402
    ALPHAS, BETAS, GAMMAS, LAMBDAS, TAUS, EndpointPriorEngine, artifact,
    exact_candidate_indices, group_max, iter_effective_configs,
    load_c4_final_scores, load_npz_arrays, retrieval_gate, retrieval_z,
    setting_id, standardized_delta,
)
from rlem.c5_prior_utils import (  # noqa: E402
    attach_eval_labels, fast_metrics,
    load_cache, localization_diagnostics, metric_deltas,
    movement_diagnostics, write_json,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cache_npz", required=True)
    p.add_argument("--c4_final_scores_npz", required=True)
    p.add_argument("--temporal_prior_npz", required=True)
    p.add_argument("--stats_json", required=True)
    p.add_argument("--gt_jsonl", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--split_role", choices=["train_calib"], default="train_calib")
    p.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    p.add_argument("--chunk_rows", type=int, default=500_000)
    p.add_argument("--effective_top_n", type=int, default=100)
    p.add_argument("--max_after_nms", type=int, default=100)
    p.add_argument("--nms_thd", type=float, default=.7)
    p.add_argument("--workers", type=int, default=max(1, min(12, os.cpu_count() or 1)))
    return p.parse_args()


_CACHE = _BASELINE = _Z_BY_SETTING = _BASE_METRICS = None


def _fast_record(cfg: Dict[str, float]) -> Dict:
    if cfg["family"] == "zero_control":
        score = _BASELINE
    else:
        z = _Z_BY_SETTING[setting_id(cfg["alpha"], cfg["beta"], cfg["tau"], cfg["lambda"])]
        score = (_BASELINE + np.float32(cfg["gamma"]) * z).astype(np.float32)
    metrics, _, _ = fast_metrics(_CACHE, score, 100, 100, .7)
    deltas = metric_deltas(metrics, _BASE_METRICS)
    move = movement_diagnostics(_CACHE, _BASELINE, score)
    six = all(deltas.get(k, -1e9) >= -1e-9 for k in PRIMARY_METRICS)
    r100 = all(deltas.get(k, -1e9) >= -1e-9 for k in ("0.5-r100", "0.7-r100"))
    rec = {
        "config": cfg, "metrics": metrics, "deltas_vs_c4_final": deltas,
        "movement": move, "six_primary_nonnegative": bool(six),
        "r100_both_nonnegative": bool(r100), "movement_safe": _movement_safe(move),
        "selection_score_delta_vs_c4_final": float(selection_score(metrics, _BASE_METRICS)),
    }
    rec["vcmr_movement_gate"] = bool(six and r100 and rec["movement_safe"])
    return rec


def _extra_rank_diagnostics(cache: Dict[str, np.ndarray], base: np.ndarray, new: np.ndarray) -> Dict[str, float]:
    offsets = cache["desc_offsets"].astype(np.int64)
    row_gid = cache["row_group_id"].astype(np.int64)
    is_gt = cache["is_gt_video"].astype(np.float32) > .5
    iou = cache["eval_iou"].reshape(-1)
    improved = worsened = unchanged = 0
    deltas = []
    for q in range(len(offsets) - 1):
        s, e = int(offsets[q]), int(offsets[q + 1])
        local = np.arange(s, e, dtype=np.int64)
        gt = local[is_gt[s:e]]
        if not gt.size:
            continue
        best = int(gt[np.argmax(iou[gt])]); gid = int(row_gid[best])
        same = gt[row_gid[gt] == gid]
        if not same.size:
            continue
        b_order = np.argsort(-base[same], kind="stable")
        n_order = np.argsort(-new[same], kind="stable")
        bp = int(np.where(same[b_order] == best)[0][0]) + 1
        np_ = int(np.where(same[n_order] == best)[0][0]) + 1
        delta = np_ - bp; deltas.append(delta)
        improved += int(delta < 0); worsened += int(delta > 0); unchanged += int(delta == 0)
    n = max(len(deltas), 1)
    return {
        "best_iou_rank_improved_ratio": float(improved / n),
        "best_iou_rank_worsened_ratio": float(worsened / n),
        "best_iou_rank_unchanged_ratio": float(unchanged / n),
        "best_iou_rank_delta_mean": float(np.mean(deltas)) if deltas else float("nan"),
        "delta_bd_improves_more_often_than_worsens": bool(improved > worsened),
    }


def _movement_safe(move: Dict[str, float]) -> bool:
    return bool(
        move["top1_changed_ratio"] <= .08
        and move["pearson_base_candidate"] >= .985
        and move["hard_positive_top100_exit_ratio"] <= .005
    )


def _adjacent(a: Dict[str, float], b: Dict[str, float]) -> bool:
    ranges = (ALPHAS, BETAS, TAUS, LAMBDAS, GAMMAS)
    names = ("alpha", "beta", "tau", "lambda", "gamma")
    distance = 0
    for name, values in zip(names, ranges):
        ia, ib = values.index(float(a[name])), values.index(float(b[name]))
        distance += abs(ia - ib)
    return distance == 1


def _boundary(cfg: Dict[str, float]) -> bool:
    return bool(
        cfg["alpha"] in (ALPHAS[0], ALPHAS[-1])
        or cfg["beta"] in (BETAS[0], BETAS[-1])
        or cfg["tau"] in (TAUS[0], TAUS[-1])
        or cfg["lambda"] in (LAMBDAS[0], LAMBDAS[-1])
        or cfg["gamma"] in (GAMMAS[0], GAMMAS[-1])
    )


def main():
    args = parse_args()
    if (args.effective_top_n, args.max_after_nms, args.nms_thd) != (100, 100, .7):
        raise ValueError("C5-main-A frozen retrieval constants must remain 100/100/0.7")
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    paths = {
        "grid_csv": out / "grid_results.csv", "grid_json": out / "grid_results.json",
        "best": out / "best_config.json", "audit": out / "train_calib_search_audit.json",
        "loc": out / "localization_diagnostics.json", "move": out / "movement_diagnostics.json",
    }
    existing = [str(p) for p in paths.values() if p.exists()]
    if existing:
        raise FileExistsError(existing)

    stats = json.loads(Path(args.stats_json).read_text(encoding="utf-8"))
    if stats.get("stats_source") != "train_fit_only" or stats.get("official_val_used") is not False:
        raise ValueError("C5-main-A requires frozen train_fit-only statistics")
    cache = load_cache(args.cache_npz); attach_eval_labels(cache, args.gt_jsonl)
    rows = len(cache["row_group_id"])
    baseline = load_c4_final_scores(args.c4_final_scores_npz, rows)
    row_gid = cache["row_group_id"].astype(np.int64)
    group_count = len(cache["group_ids_sorted_unique"])
    temporal_length = load_npz_arrays(args.temporal_prior_npz, ("temporal_length",))["temporal_length"].astype(np.int64)
    start_idx, end_idx, index_audit = exact_candidate_indices(
        cache["start_time"], cache["end_time"], row_gid, temporal_length
    )
    group_score = group_max(baseline, row_gid, group_count)
    rz = retrieval_z(group_score, stats["retrieval_score_stats"])
    gates = {tau: retrieval_gate(rz, tau) for tau in TAUS}

    engine = EndpointPriorEngine.load(args.temporal_prior_npz, args.device)
    z_by_setting = {}
    for alpha in ALPHAS:
        for beta in BETAS:
            log_s, log_e, _ = engine.role_log_priors(alpha, beta)
            for tau in TAUS:
                for lambda_ in LAMBDAS:
                    key = setting_id(alpha, beta, tau, lambda_)
                    delta = engine.injected_endpoint_delta(
                        log_s, log_e, np.float32(lambda_) * gates[tau],
                        row_gid, start_idx, end_idx, chunk_rows=args.chunk_rows,
                    )
                    z_by_setting[key] = standardized_delta(delta, stats["delta_stats"][key])
                    del delta
            del log_s, log_e

    base_metrics, _, _ = fast_metrics(cache, baseline, 100, 100, .7)
    c4_lite, _, _, _ = frozen_c4_lite_row_scores(cache)
    c4_lite_metrics, _, _ = fast_metrics(cache, c4_lite, 100, 100, .7)
    c31_metrics, _, _ = fast_metrics(cache, cache["s_c31"].astype(np.float32), 100, 100, .7)

    global _CACHE, _BASELINE, _Z_BY_SETTING, _BASE_METRICS
    _CACHE, _BASELINE, _Z_BY_SETTING, _BASE_METRICS = cache, baseline, z_by_setting, base_metrics
    configs = list(iter_effective_configs())
    if args.workers <= 1:
        records = [_fast_record(cfg) for cfg in configs]
    else:
        with mp.get_context("fork").Pool(args.workers) as pool:
            records = list(pool.imap_unordered(_fast_record, configs, chunksize=1))
    records.sort(key=lambda r: r["config"]["config_id"])

    detailed = [r for r in records if r["vcmr_movement_gate"]]
    if not detailed:
        detailed = sorted(records, key=lambda r: r["selection_score_delta_vs_c4_final"], reverse=True)[:10]
    for rec in detailed:
        cfg = rec["config"]
        if cfg["family"] == "zero_control": score = baseline
        else:
            z = z_by_setting[setting_id(cfg["alpha"], cfg["beta"], cfg["tau"], cfg["lambda"])]
            score = (baseline + np.float32(cfg["gamma"]) * z).astype(np.float32)
        loc = localization_diagnostics(cache, baseline, score)
        extra = _extra_rank_diagnostics(cache, baseline, score)
        loc.update(extra); rec["localization"] = loc
        oracle = loc["oracle_video_r1_05_delta"] > 0 or loc["oracle_video_r1_07_delta"] > 0
        rank = loc["best_iou_span_rank_delta_mean"] < 0 and loc["delta_bd_improves_more_often_than_worsens"]
        miou = loc["selected_span_miou_delta"] >= -1e-12
        rec["localization_gate"] = bool(oracle and rank and miou)
        rec["core_promotion_gate"] = bool(rec["vcmr_movement_gate"] and rec["localization_gate"])
        rec["selection_with_localization_bonus"] = float(
            rec["selection_score_delta_vs_c4_final"]
            + .25 * max(0.0, loc["oracle_video_r1_05_delta"])
            + .25 * max(0.0, loc["oracle_video_r1_07_delta"])
            + 25.0 * max(0.0, loc["selected_span_miou_delta"])
            + .05 * max(0.0, -loc["best_iou_span_rank_delta_mean"])
        )

    core = [r for r in detailed if r.get("core_promotion_gate") and r["config"]["family"] != "zero_control"]
    for rec in records:
        cfg = rec["config"]
        neighbors = []
        if cfg["family"] != "zero_control":
            neighbors = [r["config"]["config_id"] for r in core if r is not rec and _adjacent(cfg, r["config"])]
        rec["passing_neighbor_ids"] = neighbors
        rec["passing_neighbor_count"] = len(neighbors)
        rec["active_parameter_on_grid_boundary"] = bool(cfg["family"] != "zero_control" and _boundary(cfg))
        rec["isolated_boundary_point"] = bool(rec["active_parameter_on_grid_boundary"] and not neighbors)
        rec["promotion_candidate"] = bool(rec.get("core_promotion_gate", False) and not rec["isolated_boundary_point"])

    promotable = [r for r in records if r["promotion_candidate"]]
    if promotable:
        best = max(promotable, key=lambda r: (r["passing_neighbor_count"], r.get("selection_with_localization_bonus", -1e9)))
        status = "PASS"
    else:
        best = max(detailed, key=lambda r: r.get("selection_with_localization_bonus", r["selection_score_delta_vs_c4_final"]))
        status = "NO_PROMOTION"
    best["deltas_vs_c4_lite"] = metric_deltas(best["metrics"], c4_lite_metrics)
    best["deltas_vs_c31"] = metric_deltas(best["metrics"], c31_metrics)

    rows_out: List[Dict] = []
    for r in records:
        cfg, mv = r["config"], r["movement"]
        row = {
            **cfg,
            "selection_score_delta_vs_c4_final": r["selection_score_delta_vs_c4_final"],
            "six_primary_nonnegative": r["six_primary_nonnegative"],
            "r100_both_nonnegative": r["r100_both_nonnegative"],
            "movement_safe": r["movement_safe"],
            "localization_gate": r.get("localization_gate"),
            "promotion_candidate": r["promotion_candidate"],
            "passing_neighbor_count": r["passing_neighbor_count"],
            "active_parameter_on_grid_boundary": r["active_parameter_on_grid_boundary"],
            "isolated_boundary_point": r["isolated_boundary_point"],
            "top1_changed_ratio": mv["top1_changed_ratio"],
            "hard_positive_top100_exits": mv["hard_positive_top100_exits"],
            "hard_positive_top100_entries": mv["hard_positive_top100_entries"],
            "pearson": mv["pearson_base_candidate"],
            "mean_within_query_spearman": mv["mean_within_query_spearman"],
            **{f"delta_{k}": v for k, v in r["deltas_vs_c4_final"].items()},
            **{f"metric_{k}": v for k, v in r["metrics"].items()},
        }
        for k, v in (r.get("localization") or {}).items(): row[f"loc_{k}"] = v
        rows_out.append(row)
    fields = []
    for row in rows_out:
        for key in row:
            if key not in fields: fields.append(key)
    with open(paths["grid_csv"], "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows_out)
    write_json(str(paths["grid_json"]), {
        "status": status, "effective_grid_size": len(records),
        "role_endpoint_grid_size": 432,
        "zero_control_count": 1, "records": rows_out,
    })
    write_json(str(paths["best"]), best["config"])
    write_json(str(paths["loc"]), best.get("localization", {}))
    write_json(str(paths["move"]), best["movement"])
    flags = {
        "six_primary_nonnegative": best["six_primary_nonnegative"],
        "r100_both_nonnegative": best["r100_both_nonnegative"],
        "movement_safe": best["movement_safe"],
        "best_iou_rank_improved": (best.get("localization") or {}).get("best_iou_span_rank_delta_mean", 0) < 0,
        "best_iou_improved_more_often_than_worsened": (best.get("localization") or {}).get("delta_bd_improves_more_often_than_worsens", False),
        "oracle_video_localization_positive": ((best.get("localization") or {}).get("oracle_video_r1_05_delta", 0) > 0 or (best.get("localization") or {}).get("oracle_video_r1_07_delta", 0) > 0),
        "selected_span_miou_nonnegative": (best.get("localization") or {}).get("selected_span_miou_delta", -1) >= -1e-12,
        "isolated_boundary_point": best["isolated_boundary_point"],
        "promotion_candidate": best["promotion_candidate"],
    }
    audit = {
        "status": status,
        "stage": "C5-main-inference-A train_calib search",
        "scope": "train_calib_only",
        "official_val_used": False, "post_val_adjustment": False, "score_grid_on_val": False,
        "training_used": False,
        "primary_baseline": "C4_final = C4-r2-cal-v2.1 v21_00444",
        "secondary_baselines": {"C4_lite": c4_lite_metrics, "C3.1": c31_metrics},
        "baseline_metrics": base_metrics,
        "best_config": best["config"], "best_metrics": best["metrics"],
        "best_deltas_vs_c4_final": best["deltas_vs_c4_final"],
        "best_deltas_vs_c4_lite": best["deltas_vs_c4_lite"], "best_deltas_vs_c31": best["deltas_vs_c31"],
        "best_movement": best["movement"], "best_localization": best.get("localization"),
        "promotion_flags": flags,
        "neighbor_robustness": {
            "passing_neighbor_count": best["passing_neighbor_count"],
            "passing_neighbor_ids": best["passing_neighbor_ids"],
            "active_parameter_on_grid_boundary": best["active_parameter_on_grid_boundary"],
            "isolated_boundary_point": best["isolated_boundary_point"],
        },
        "effective_grid_size": len(records),
        "workers": int(args.workers),
        "protocol_corrections": {
            "gauge_invariant_endpoint_posterior": True,
            "delta": "change in normalized begin/end endpoint log-probability",
            "gate": "clip(2*sigmoid(R_video_z/tau), 0.25, 1.75)",
            "discarded_raw_logit_attempt_count": 1,
        },
        "endpoint_index_audit": index_audit,
        "counts": {
            "six_primary_nonnegative": int(sum(r["six_primary_nonnegative"] for r in records)),
            "r100_both_nonnegative": int(sum(r["r100_both_nonnegative"] for r in records)),
            "movement_safe": int(sum(r["movement_safe"] for r in records)),
            "core_promotion_gate": int(sum(bool(r.get("core_promotion_gate")) for r in records)),
            "promotion_candidate": int(sum(r["promotion_candidate"] for r in records)),
        },
        "artifacts": {
            "cache_npz": artifact(args.cache_npz), "c4_final_scores_npz": artifact(args.c4_final_scores_npz),
            "temporal_prior_npz": artifact(args.temporal_prior_npz), "stats_json": artifact(args.stats_json),
            "gt_jsonl": artifact(args.gt_jsonl), "grid_csv": artifact(str(paths["grid_csv"])),
            "grid_json": artifact(str(paths["grid_json"])), "best_config": artifact(str(paths["best"])),
        },
        "retrieval_constants": {"effective_top_n": 100, "nms_thd": .7, "max_after_nms": 100},
        "used": {"C5_main_inference_A": True, "C5_main_B": False, "C6": False, "C4_main": False, "VS_R2_head": False},
    }
    write_json(str(paths["audit"]), audit)
    print(json.dumps({"status": status, "best_config": best["config"]["config_id"], "audit": str(paths["audit"])}, indent=2))


if __name__ == "__main__":
    main()
