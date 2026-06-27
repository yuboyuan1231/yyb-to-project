#!/usr/bin/env python
"""Train-calib-only fixed scorer search for C4-r2-cal.

The control is the exact frozen C4-lite c4_00394 score.  Frozen C3.1 is also
evaluated as a named secondary control.  No raw base-log control is used.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from rlem.c4_protocol import C4Protocol, write_protocol_manifest  # noqa: E402
from rlem.c4_lite_utils import ALL_METRICS, PRIMARY_METRICS, selection_score, write_json  # noqa: E402
from rlem.c4_video_targets import frozen_c4_lite_row_scores  # noqa: E402
from rlem.grid_search_c4_lite import attach_eval_labels, eval_score, fast_metrics, fast_selected, _matrix  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cache_npz", required=True)
    p.add_argument("--r2_logits_npz", required=True)
    p.add_argument("--gt_jsonl", required=True)
    p.add_argument("--dataset_config", required=True)
    p.add_argument("--split", choices=["train"], required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--audit_json", required=True)
    p.add_argument("--save_best_submission", required=True)
    p.add_argument("--effective_top_n", type=int, default=100)
    p.add_argument("--max_after_nms", type=int, default=100)
    p.add_argument("--nms_thd", type=float, default=0.7)
    p.add_argument("--video_gain_values", default="0.0,0.025,0.05,0.1,0.2,0.4")
    p.add_argument("--quality_gain_values", default="0.0,0.025,0.05,0.1,0.2")
    p.add_argument(
        "--required_nonnegative_metrics",
        default="",
        help="Comma-separated deltas vs frozen C4-lite that must be >=0.",
    )
    p.add_argument("--no_desc_type", action="store_true")
    return p.parse_args()


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _floats(text):
    return [float(v) for v in text.split(",") if v.strip()]


def _deltas(metrics, control):
    return {key: float(metrics[key] - control[key]) for key in ALL_METRICS}


def _movement(cache, control_score, new_score, args):
    control_matrix, control_order, _ = fast_selected(cache, control_score, args)
    new_matrix, new_order, _ = fast_selected(cache, new_score, args)
    control_inv = np.empty_like(control_order, dtype=np.int16)
    new_inv = np.empty_like(new_order, dtype=np.int16)
    ranks = np.broadcast_to(np.arange(200, dtype=np.int16), control_order.shape)
    np.put_along_axis(control_inv, control_order, ranks, axis=1)
    np.put_along_axis(new_inv, new_order, ranks, axis=1)
    hard = cache["eval_y05"]
    exits = int(np.sum(hard & (control_inv < 100) & (new_inv >= 100)))
    entries = int(np.sum(hard & (control_inv >= 100) & (new_inv < 100)))
    return {
        "relative_to": "frozen_C4_lite_c4_00394",
        "queries": int(control_order.shape[0]),
        "top1_changed_ratio": float(np.mean(control_order[:, 0] != new_order[:, 0])),
        "pearson_c4_lite_c4_r2": float(np.corrcoef(control_matrix.ravel(), new_matrix.ravel())[0, 1]),
        "hard_positive_top100_exits": exits,
        "hard_positive_top100_entries": entries,
        "hard_positive_top100_exit_ratio": float(exits / max(int(hard.sum()), 1)),
    }


def main():
    args = parse_args()
    protocol = C4Protocol(
        stage="c4_r2_grid_search", split=args.split,
        effective_top_n=args.effective_top_n, max_after_nms=args.max_after_nms, nms_thd=args.nms_thd,
    )
    protocol.validate()
    if args.effective_top_n != 100 or args.max_after_nms != 100 or not math.isclose(args.nms_thd, 0.7):
        raise ValueError("Frozen retrieval controls are top_n=100, NMS=0.7, max_after_nms=100")
    with np.load(args.cache_npz, allow_pickle=True) as loaded:
        cache = {key: loaded[key] for key in loaded.files}
    attach_eval_labels(cache, args.gt_jsonl)
    c31_score = cache["s_c31"].astype(np.float32)
    c4_score, _, _, _ = frozen_c4_lite_row_scores(cache)

    c31_metrics, _, _ = fast_metrics(cache, c31_score, args)
    c4_metrics, _, _ = fast_metrics(cache, c4_score, args)
    _, _, bundled_c31 = eval_score(cache, c31_score, args)
    _, _, bundled_c4 = eval_score(cache, c4_score, args)
    parity_c31 = max(abs(round(c31_metrics[k], 2) - bundled_c31[k]) for k in ALL_METRICS)
    parity_c4 = max(abs(round(c4_metrics[k], 2) - bundled_c4[k]) for k in ALL_METRICS)
    if max(parity_c31, parity_c4) > 1e-9:
        raise ValueError(f"Fast/bundled control mismatch: C31={parity_c31}, C4={parity_c4}")

    with np.load(args.r2_logits_npz, allow_pickle=True) as logits:
        group_id = logits["group_id"].astype(np.int64)
        video_logit = logits["video_logit"].astype(np.float32)
        quality_logit = logits["quality_logit"].astype(np.float32)
    expected_groups = cache["group_ids_sorted_unique"].astype(np.int64)
    if not np.array_equal(group_id, expected_groups):
        raise ValueError("train_calib logits do not align exactly with cache video groups")
    row_gid = cache["row_group_id"].astype(np.int64)
    row_video = video_logit[row_gid]
    row_quality = quality_logit[row_gid]
    video_mean, video_std = float(row_video.mean()), float(row_video.std())
    quality_mean, quality_std = float(row_quality.mean()), float(row_quality.std())
    if min(video_std, quality_std) <= 0:
        raise ValueError("Degenerate C4-r2 logits")
    video_z = ((row_video - video_mean) / video_std).astype(np.float32)
    quality_z = ((row_quality - quality_mean) / quality_std).astype(np.float32)

    configs = []
    idx = 0
    for vg in _floats(args.video_gain_values):
        for qg in _floats(args.quality_gain_values):
            configs.append({
                "config_id": f"c4r2_{idx:05d}", "family": "C4lite_plus_R2_video_quality",
                "c4_lite_scale": 1.0, "video_logit_z_gain": vg, "quality_logit_z_gain": qg,
            })
            idx += 1
    results = []
    required_nonnegative = [v.strip() for v in args.required_nonnegative_metrics.split(",") if v.strip()]
    unknown_constraints = sorted(set(required_nonnegative) - set(ALL_METRICS))
    if unknown_constraints:
        raise ValueError(f"Unknown constrained metrics: {unknown_constraints}")
    best = None
    best_score = None
    for cfg in tqdm(configs, desc="C4-r2-cal train_calib fixed scorer search"):
        score = (c4_score + cfg["video_logit_z_gain"] * video_z + cfg["quality_logit_z_gain"] * quality_z).astype(np.float32)
        metrics, _, _ = fast_metrics(cache, score, args)
        d_c4 = _deltas(metrics, c4_metrics)
        d_c31 = _deltas(metrics, c31_metrics)
        rec = {
            **cfg, "metrics": metrics,
            "deltas_vs_frozen_c4_lite": d_c4,
            "deltas_vs_frozen_c31": d_c31,
            "selection_score_delta_vs_frozen_c4_lite": selection_score(metrics, c4_metrics),
            "selection_score_delta_vs_frozen_c31": selection_score(metrics, c31_metrics),
            "primary_positive_count_vs_c4_lite": int(sum(d_c4[k] > 0 for k in PRIMARY_METRICS)),
            "r100_min_delta_vs_c4_lite": float(min(d_c4["0.5-r100"], d_c4["0.7-r100"])),
            "official_val_used": False,
        }
        rec["feasible"] = rec["primary_positive_count_vs_c4_lite"] >= 4 and rec["r100_min_delta_vs_c4_lite"] >= -0.25
        rec["hard_constraints_pass"] = all(d_c4[key] >= 0.0 for key in required_nonnegative)
        rec["all_eight_strictly_positive_vs_c4_lite"] = all(d_c4[key] > 0.0 for key in ALL_METRICS)
        rec["all_six_primary_strictly_positive_vs_c4_lite"] = all(d_c4[key] > 0.0 for key in PRIMARY_METRICS)
        results.append(rec)
        eligible = rec["hard_constraints_pass"] if required_nonnegative else rec["feasible"]
        if eligible and (best is None or rec["selection_score_delta_vs_frozen_c4_lite"] > best["selection_score_delta_vs_frozen_c4_lite"]):
            best, best_score = rec, score
    if best is None:
        best = max(results, key=lambda item: item["selection_score_delta_vs_frozen_c4_lite"])
        best_score = (c4_score + best["video_logit_z_gain"] * video_z + best["quality_logit_z_gain"] * quality_z).astype(np.float32)

    best_submission, _, bundled_best = eval_score(cache, best_score, args)
    parity_best = max(abs(round(best["metrics"][k], 2) - bundled_best[k]) for k in ALL_METRICS)
    if parity_best > 1e-9:
        raise ValueError(f"Fast/bundled best mismatch: {parity_best}")
    movement = _movement(cache, c4_score, best_score, args)
    best["movement_vs_frozen_c4_lite"] = movement
    best["normalization_from_train_calib_rows"] = {
        "video_logit_row_expanded_mean": video_mean, "video_logit_row_expanded_std": video_std,
        "quality_logit_row_expanded_mean": quality_mean, "quality_logit_row_expanded_std": quality_std,
    }
    best["effective_top_n"] = 100
    best["nms_thd"] = 0.7
    best["max_after_nms"] = 100
    best["post_val_adjustment"] = False

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_json(str(out / "grid_results.json"), {
        "status": "PASS", "scope": "train_calib_only", "grid_size": len(results),
        "required_nonnegative_metrics": required_nonnegative,
        "all_eight_strictly_positive_exists": any(rec["all_eight_strictly_positive_vs_c4_lite"] for rec in results),
        "all_six_primary_strictly_positive_exists": any(rec["all_six_primary_strictly_positive_vs_c4_lite"] for rec in results),
        "controls": {"frozen_c31": c31_metrics, "frozen_c4_lite_c4_00394": c4_metrics},
        "best": best, "results": results, "official_val_used": False,
    })
    fields = [
        "config_id", "family", "c4_lite_scale", "video_logit_z_gain", "quality_logit_z_gain",
        "selection_score_delta_vs_frozen_c4_lite", "selection_score_delta_vs_frozen_c31",
        "primary_positive_count_vs_c4_lite", "r100_min_delta_vs_c4_lite", "feasible",
    ] + ALL_METRICS
    with open(out / "grid_results.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(fields)
        for rec in results:
            writer.writerow([rec["metrics"].get(k, rec.get(k, "")) if k in ALL_METRICS else rec.get(k, "") for k in fields])
    write_json(str(out / "best_config.json"), best)
    write_json(args.save_best_submission, best_submission, pretty=False)
    exceeds = all(best["deltas_vs_frozen_c4_lite"][k] > 0 for k in PRIMARY_METRICS)
    r100_safe = min(best["deltas_vs_frozen_c4_lite"]["0.5-r100"], best["deltas_vs_frozen_c4_lite"]["0.7-r100"]) >= 0
    nonzero_hard_pass_count = sum(
        rec["hard_constraints_pass"]
        and (rec["video_logit_z_gain"] != 0.0 or rec["quality_logit_z_gain"] != 0.0)
        for rec in results
    )
    constrained_robust = (not required_nonnegative) or nonzero_hard_pass_count >= 2
    audit = {
        "status": "PASS",
        "scope": "train_calib_only",
        "grid_size": len(results),
        "required_nonnegative_metrics": required_nonnegative,
        "hard_constraint_pass_count": int(sum(rec["hard_constraints_pass"] for rec in results)),
        "nonzero_hard_constraint_pass_count": int(nonzero_hard_pass_count),
        "constrained_robustness_pass": bool(constrained_robust),
        "all_eight_strictly_positive_exists": any(rec["all_eight_strictly_positive_vs_c4_lite"] for rec in results),
        "all_six_primary_strictly_positive_exists": any(rec["all_six_primary_strictly_positive_vs_c4_lite"] for rec in results),
        "capacity_search": False,
        "scorer_family_count": 1,
        "baseline_control_primary": "frozen_C4_lite_c4_00394",
        "baseline_control_secondary": "frozen_C3.1",
        "raw_base_log_used_as_control": False,
        "exact_scorer_formula": "S_C4_r2 = S_C4_lite_c4_00394 + video_logit_z_gain*z(video_logit) + quality_logit_z_gain*z(quality_logit)",
        "frozen_c31_metrics": c31_metrics,
        "frozen_c4_lite_metrics": c4_metrics,
        "best_config": best,
        "train_calib_rerank_metrics": best["metrics"],
        "deltas_vs_frozen_c31": best["deltas_vs_frozen_c31"],
        "deltas_vs_frozen_c4_lite": best["deltas_vs_frozen_c4_lite"],
        "r100_deltas_vs_frozen_c4_lite": {k: best["deltas_vs_frozen_c4_lite"][k] for k in ["0.5-r100", "0.7-r100"]},
        "movement_vs_frozen_c4_lite": movement,
        "fast_bundled_evaluator_parity": {"frozen_c31": parity_c31, "frozen_c4_lite": parity_c4, "best_c4_r2": parity_best},
        "c4_r2_cal_exceeds_c4_lite_train_calib_all_primary": exceeds,
        "r100_safe": r100_safe,
        "recommend_future_official_val_one_shot": bool(exceeds and r100_safe and constrained_robust and movement["hard_positive_top100_exits"] <= movement["hard_positive_top100_entries"]),
        "r2_logits_sha256": sha256_file(args.r2_logits_npz),
        "best_submission_sha256": sha256_file(args.save_best_submission),
        "official_val_used": False,
        "official_val_evidence_opened": False,
        "score_grid_on_val": False,
        "post_val_adjustment": False,
        "future_stages_used": {"c4_r2_official_val": False, "c4_main": False, "r2_head": False, "vs_head": False, "c5": False, "c6": False},
    }
    write_protocol_manifest(args.audit_json, protocol, audit)
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
