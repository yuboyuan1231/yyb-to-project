#!/usr/bin/env python
"""Apply one frozen C4-lite config to a scored candidate JSONL.

This script is for a later explicit one-shot only.  It refuses official val
unless --allow_official_val is provided and never searches weights.
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from rlem.c4_lite_utils import (  # noqa: E402
    iter_query_groups, load_desc_ids, row_span_quality, submission_from_groups,
    write_json, get_row_q_scores, infer_log_r1,
)


EXPECTED_ONE_SHOT = {
    "config_id": "c4_00394", "family": "B_video_span",
    "alpha_bd": 1.25, "gamma_fp": 0.5, "video_gain": 1.0,
    "span_gain": 0.5, "base_scale": 0.75,
    "effective_top_n": 100, "nms_thd": 0.7, "max_after_nms": 100,
    "t_joint": 1.5, "t_bd": 1.0, "t_fp": 1.0,
    "mean_qbd_train_calib": 0.3990069627761841,
    "mean_g_video_row_expanded_train_calib": 0.13249324262142181,
    "mean_g_span_train_calib": 0.03713987395167351,
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--scored_jsonl", required=True)
    p.add_argument("--frozen_config", required=True)
    p.add_argument("--dataset_config", required=True)
    p.add_argument("--split", choices=["train", "val"], required=True)
    p.add_argument("--output_json", required=True)
    p.add_argument("--audit_json", required=True)
    p.add_argument("--desc_ids_filter", default=None)
    p.add_argument("--allow_official_val", action="store_true")
    p.add_argument("--effective_top_n", type=int, default=100)
    p.add_argument("--max_after_nms", type=int, default=100)
    p.add_argument("--nms_thd", type=float, default=0.7)
    return p.parse_args()


def apply_group(rows, cfg):
    q = [get_row_q_scores(r, cfg.get("t_joint", 1.0), cfg.get("t_bd", 1.0), cfg.get("t_fp", 1.0)) for r in rows]
    mean_qbd = float(cfg["mean_qbd_train_calib"])
    g_span = np.asarray([
        row_span_quality(qj, qb, efp, cfg["alpha_bd"], cfg["gamma_fp"], mean_qbd=mean_qbd, centered_bd=True)
        for qj, qb, efp in q
    ], dtype=np.float32)
    # Max span-quality per video.
    per_video = {}
    for i, r in enumerate(rows):
        vid = int(r["video_idx"])
        per_video[vid] = max(per_video.get(vid, -1e30), float(g_span[i]))
    g_video = np.asarray([per_video[int(r["video_idx"])] for r in rows], dtype=np.float32)
    g_video_c = g_video - float(cfg["mean_g_video_row_expanded_train_calib"])
    g_span_c = g_span - float(cfg["mean_g_span_train_calib"])
    for i, r in enumerate(rows):
        base_log = float(r.get("base_log", math.log(max(float(r.get("s_base", 0.0)), 1e-8))))
        log_r1 = float(r.get("log_r1", infer_log_r1(r)))
        log_boundary = float(r.get("log_boundary", base_log - log_r1))
        fam = cfg["family"]
        if fam == "A_video_feedback":
            score = cfg["base_scale"] * float(r["s_c31"]) + cfg["video_gain"] * g_video_c[i]
        elif fam == "B_video_span":
            score = cfg["base_scale"] * float(r["s_c31"]) + cfg["video_gain"] * g_video_c[i] + cfg["span_gain"] * g_span_c[i]
        elif fam == "C_split_base":
            score = cfg["video_scale"] * log_r1 + cfg["boundary_scale"] * log_boundary + cfg["video_gain"] * g_video_c[i] + cfg["span_gain"] * g_span_c[i]
        else:
            raise ValueError(fam)
        r["s_c4"] = float(score)
    return rows


def validate_exact_config(cfg):
    for key, expected in EXPECTED_ONE_SHOT.items():
        if key not in cfg:
            raise ValueError(f"Frozen one-shot config lacks {key}")
        actual = cfg[key]
        if isinstance(expected, float):
            if not math.isclose(float(actual), expected, rel_tol=0.0, abs_tol=1e-12):
                raise ValueError(f"Frozen one-shot {key}={actual}, expected {expected}")
        elif actual != expected:
            raise ValueError(f"Frozen one-shot {key}={actual}, expected {expected}")


def movement_diagnostics(groups):
    base_all, new_all = [], []
    changed_top1 = hard_exits = hard_entries = 0
    spearman = []
    for rows in groups:
        n = len(rows)
        base = np.asarray([float(r["s_c31"]) for r in rows], dtype=np.float64)
        new = np.asarray([float(r["s_c4"]) for r in rows], dtype=np.float64)
        rank_base = np.asarray([int(r.get("rank_base", i)) for i, r in enumerate(rows)], dtype=np.int64)
        base_order = np.lexsort((rank_base, -base))
        new_order = np.lexsort((rank_base, -new))
        changed_top1 += int(int(base_order[0]) != int(new_order[0]))
        base_inv = np.empty(n, dtype=np.int32); base_inv[base_order] = np.arange(n)
        new_inv = np.empty(n, dtype=np.int32); new_inv[new_order] = np.arange(n)
        hard = np.asarray([bool(r.get("y_joint_05", False)) for r in rows])
        hard_exits += int(np.sum(hard & (base_inv < 100) & (new_inv >= 100)))
        hard_entries += int(np.sum(hard & (base_inv >= 100) & (new_inv < 100)))
        diff = base_inv.astype(np.float64) - new_inv.astype(np.float64)
        spearman.append(1.0 - 6.0 * float(np.sum(diff * diff)) / (n * (n * n - 1.0)))
        base_all.append(base); new_all.append(new)
    base_flat = np.concatenate(base_all); new_flat = np.concatenate(new_all)
    return {
        "reference": "frozen_c31",
        "queries": len(groups),
        "top1_changed_ratio": changed_top1 / max(len(groups), 1),
        "hard_positive_top100_exits": hard_exits,
        "hard_positive_top100_entries": hard_entries,
        "pearson_c31_c4": float(np.corrcoef(base_flat, new_flat)[0, 1]),
        "mean_within_query_spearman": float(np.mean(spearman)),
    }


def main():
    args = parse_args()
    if args.split == "val" and not args.allow_official_val:
        raise ValueError("Refusing official val C4-lite one-shot without explicit --allow_official_val")
    if args.effective_top_n != 100 or args.max_after_nms != 100 or not math.isclose(args.nms_thd, 0.7):
        raise ValueError("C4-lite protocol requires effective_top_n=100, max_after_nms=100, NMS=0.7")
    if Path(args.output_json).exists() or Path(args.audit_json).exists():
        raise FileExistsError("C4-lite one-shot output/audit already exists; rerun is forbidden")
    frozen = json.load(open(args.frozen_config, "r", encoding="utf-8"))
    if args.split == "val":
        if frozen.get("status") != "FROZEN" or frozen.get("decision") != "C4_LITE_TRAIN_CALIB_FROZEN":
            raise ValueError("Official-val one-shot requires the accepted C4 freeze manifest")
        cfg = frozen["exact_one_shot_config"]
    else:
        cfg = frozen.get("exact_one_shot_config", frozen.get("best", frozen))
    validate_exact_config(cfg)
    allowed = load_desc_ids(args.desc_ids_filter)
    groups = []
    for group in iter_query_groups(args.scored_jsonl, allowed):
        groups.append(apply_group(group, cfg))
    if not groups or any(len(group) != 200 for group in groups):
        raise ValueError("C4-lite one-shot requires exactly 200 rows for every query")
    movement = movement_diagnostics(groups)
    submission = submission_from_groups(groups, "s_c4", args.dataset_config, args.split, args.effective_top_n, args.max_after_nms, args.nms_thd)
    row_counts = submission.pop("_row_counts", {})
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    partial = args.output_json + ".partial"
    with open(partial, "w", encoding="utf-8") as f:
        json.dump(submission, f, ensure_ascii=False, separators=(",", ":"))
    os.replace(partial, args.output_json)
    audit = {
        "status": "PASS", "scope": args.split,
        "official_val_used": bool(args.split == "val" and args.allow_official_val),
        "config": cfg,
        "queries": len(groups), "rows_per_query_distribution": row_counts,
        "movement_relative_to_c31": movement,
        "official_val_one_shot": bool(args.split == "val" and args.allow_official_val),
        "learned_config_count": 1,
        "score_grid_on_val": False,
        "temperature_search_on_val": False,
        "post_val_adjustment": False,
        "used": {"c4_r2_cal": False, "c4_main": False, "r2": False, "vs_head": False, "c5": False, "c6": False},
    }
    write_json(args.audit_json, audit)
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
