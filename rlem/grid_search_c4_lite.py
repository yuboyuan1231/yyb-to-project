#!/usr/bin/env python
"""Train-calib-only C4-lite video-level feedback search.

C4-lite tests whether localization/span-quality evidence can improve video-level
ranking by aggregating span quality per (query, video).  It never trains a VS/R2
head and never changes the fixed candidate pool.
"""

import argparse
import itertools
import json
import math
import multiprocessing as mp
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from rlem.c4_lite_utils import (  # noqa: E402
    PRIMARY_METRICS, ALL_METRICS, flatten_eval_metrics, selection_score,
    submission_from_groups, write_json,
)
from standalone_eval.eval import eval_retrieval, load_jsonl  # noqa: E402
from rlem.grid_search_c3_weights import greedy_nms_indices, raw_metrics_from_selected  # noqa: E402


_CACHE = None
_ARGS = None
_BASELINE_FAST = None


def parse_float_list(text: str) -> List[float]:
    return [float(x) for x in text.split(",") if x.strip()]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cache_npz", required=True)
    p.add_argument("--gt_jsonl", required=True)
    p.add_argument("--dataset_config", required=True)
    p.add_argument("--split", choices=["train"], required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--effective_top_n", type=int, default=100)
    p.add_argument("--max_after_nms", type=int, default=100)
    p.add_argument("--nms_thd", type=float, default=0.7)
    p.add_argument("--alpha_bd_values", default="0.75,1.0,1.25")
    p.add_argument("--gamma_fp_values", default="0.25,0.5,0.75")
    p.add_argument("--video_gain_values", default="-0.25,0.0,0.25,0.5,0.75,1.0")
    p.add_argument("--span_gain_values", default="0.0,0.25,0.5")
    p.add_argument("--base_scale_values", default="0.75,1.0")
    p.add_argument("--video_scale_values", default="1.0")
    p.add_argument("--boundary_scale_values", default="1.0")
    p.add_argument("--families", default="A_video_feedback,B_video_span,C_split_base")
    p.add_argument("--max_configs", type=int, default=None)
    p.add_argument("--save_best_submission", default=None)
    p.add_argument("--no_desc_type", action="store_true")
    p.add_argument("--workers", type=int, default=max(1, min(8, os.cpu_count() or 1)))
    return p.parse_args()


def load_cache(path):
    # Materialize once. Re-indexing an NpzFile inside every grid configuration
    # would repeatedly decompress the same 1.7M-row arrays.
    with np.load(path, allow_pickle=True) as payload:
        return {key: payload[key] for key in payload.files}


def aggregate_video_signal(g_span: np.ndarray, cache, mode="max"):
    sort_idx = cache["group_sort_idx"]
    offsets = cache["group_offsets"]
    unique = cache["group_ids_sorted_unique"]
    sorted_values = g_span[sort_idx]
    if mode == "max":
        group_values = np.maximum.reduceat(sorted_values, offsets)
    elif mode == "mean":
        sums = np.add.reduceat(sorted_values, offsets)
        counts = np.diff(np.r_[offsets, len(sorted_values)])
        group_values = sums / np.maximum(counts, 1)
    else:
        raise ValueError(mode)
    # unique ids are dense because assigned incrementally, but keep explicit map.
    dense = np.zeros(int(np.max(unique)) + 1, dtype=np.float32)
    dense[unique] = group_values.astype(np.float32)
    return dense[cache["row_group_id"]]


def score_config(cache, cfg: Dict):
    qj = cache["q_joint"].astype(np.float32)
    qb = cache["q_bd"].astype(np.float32)
    efp = cache["e_fp"].astype(np.float32)
    mean_qbd = float(np.mean(qb))
    g_span = qj + cfg["alpha_bd"] * qj * (qb - mean_qbd) - cfg["gamma_fp"] * efp
    g_video = aggregate_video_signal(g_span, cache, mode="max")
    g_video_centered = g_video - float(np.mean(g_video))
    g_span_centered = g_span - float(np.mean(g_span))
    family = cfg["family"]
    if family == "A_video_feedback":
        score = cfg["base_scale"] * cache["s_c31"] + cfg["video_gain"] * g_video_centered
    elif family == "B_video_span":
        score = cfg["base_scale"] * cache["s_c31"] + cfg["video_gain"] * g_video_centered + cfg["span_gain"] * g_span_centered
    elif family == "C_split_base":
        score = (
            cfg["video_scale"] * cache["log_r1"] + cfg["boundary_scale"] * cache["log_boundary"]
            + cfg["video_gain"] * g_video_centered + cfg["span_gain"] * g_span_centered
        )
    else:
        raise ValueError(family)
    return score.astype(np.float32), {
        "mean_qbd": mean_qbd,
        "mean_g_video": float(np.mean(g_video)),
        "mean_g_span": float(np.mean(g_span)),
        "std_share_base": float(np.std(cache["s_c31"]) / max(np.std(score), 1e-12)),
        "std_g_video": float(np.std(g_video_centered)),
        "std_g_span": float(np.std(g_span_centered)),
    }


def groups_for_submission(cache, score: np.ndarray, score_key="s_c4"):
    desc_offsets = cache["desc_offsets"]
    desc_ids = cache["desc_ids"]
    desc_text = cache["desc_text"]
    rows_groups = []
    for q in range(len(desc_ids)):
        start, end = int(desc_offsets[q]), int(desc_offsets[q + 1])
        rows = []
        for i in range(start, end):
            rows.append({
                "desc_id": int(desc_ids[q]), "desc": str(desc_text[q]),
                "video_idx": int(cache["video_idx"][i]),
                "start_time": float(cache["start_time"][i]),
                "end_time": float(cache["end_time"][i]),
                "rank_base": int(cache["rank_base"][i]),
                "base_log": float(cache["base_log"][i]),
                score_key: float(score[i]),
                "y_fp": float(cache["y_fp"][i]),
                "y_joint_05": float(cache["y_joint_05"][i]),
                "iou": float(cache["iou"][i]),
            })
        rows_groups.append(rows)
    return rows_groups


def eval_score(cache, score, args):
    groups = groups_for_submission(cache, score)
    submission = submission_from_groups(
        groups, "s_c4", args.dataset_config, args.split,
        effective_top_n=args.effective_top_n,
        max_after_nms=args.max_after_nms,
        nms_thd=args.nms_thd,
    )
    submission.pop("_row_counts", None)
    metrics = eval_retrieval(
        submission,
        load_jsonl(args.gt_jsonl),
        iou_thds=(0.5, 0.7),
        verbose=False,
        match_number=True,
        use_desc_type=not args.no_desc_type,
    )
    return submission, metrics, flatten_eval_metrics(metrics)


def movement_diagnostics(cache, base_score, new_score):
    changed = 0
    queries = len(cache["desc_offsets"]) - 1
    for q in range(queries):
        start, end = int(cache["desc_offsets"][q]), int(cache["desc_offsets"][q + 1])
        ranks = cache["rank_base"][start:end]
        base_order = np.lexsort((ranks, -base_score[start:end]))
        new_order = np.lexsort((ranks, -new_score[start:end]))
        changed += int(int(base_order[0]) != int(new_order[0]))
    if np.std(base_score) == 0 or np.std(new_score) == 0:
        pearson = float("nan")
    else:
        pearson = float(np.corrcoef(base_score, new_score)[0, 1])
    return {
        "queries": queries,
        "top1_changed_ratio": changed / max(queries, 1),
        "pearson_c31_c4": pearson,
    }


def config_iter(args):
    families = [x.strip() for x in args.families.split(",") if x.strip()]
    count = 0
    alpha_values = parse_float_list(args.alpha_bd_values)
    gamma_values = parse_float_list(args.gamma_fp_values)
    video_gain_values = parse_float_list(args.video_gain_values)
    span_gain_values = parse_float_list(args.span_gain_values)
    base_scale_values = parse_float_list(args.base_scale_values)
    video_scale_values = parse_float_list(args.video_scale_values)
    boundary_scale_values = parse_float_list(args.boundary_scale_values)
    for family in families:
        # Do not enumerate parameters that a family does not consume. The
        # package version silently repeated hundreds of byte-identical scores.
        if family == "A_video_feedback":
            products = itertools.product(alpha_values, gamma_values, video_gain_values, [0.0], base_scale_values, [1.0], [1.0])
        elif family == "B_video_span":
            products = itertools.product(alpha_values, gamma_values, video_gain_values, span_gain_values, base_scale_values, [1.0], [1.0])
        elif family == "C_split_base":
            products = itertools.product(alpha_values, gamma_values, video_gain_values, span_gain_values, [1.0], video_scale_values, boundary_scale_values)
        else:
            raise ValueError(family)
        for values in products:
            cfg = {
                "config_id": f"c4_{count:05d}", "family": family,
                "alpha_bd": values[0], "gamma_fp": values[1],
                "video_gain": values[2], "span_gain": values[3],
                "base_scale": values[4], "video_scale": values[5], "boundary_scale": values[6],
            }
            count += 1
            yield cfg
            if args.max_configs is not None and count >= args.max_configs:
                return


def _matrix(cache, key):
    rows = int(cache["desc_offsets"][-1])
    queries = len(cache["desc_offsets"]) - 1
    if rows != queries * 200 or not np.all(np.diff(cache["desc_offsets"]) == 200):
        raise ValueError("C4 fast grid requires exactly 200 contiguous rows/query")
    return np.asarray(cache[key]).reshape(queries, 200)


def fast_selected(cache, score, args):
    scores = np.asarray(score, dtype=np.float32).reshape(-1, 200)
    order = np.argsort(-scores, axis=1, kind="stable")
    video_idx = _matrix(cache, "video_idx")
    starts = _matrix(cache, "start_time")
    ends = _matrix(cache, "end_time")
    selected = [
        greedy_nms_indices(
            order[q, :args.effective_top_n], scores[q], video_idx[q],
            starts[q], ends[q], args.nms_thd, args.max_after_nms,
        )
        for q in range(scores.shape[0])
    ]
    return scores, order, selected


def fast_metrics(cache, score, args):
    scores, order, selected = fast_selected(cache, score, args)
    raw = raw_metrics_from_selected(selected, cache["eval_y05"], cache["eval_y07"])
    return {key: float(value) for key, value in raw.items()}, scores, order


def attach_eval_labels(cache, gt_jsonl):
    gt_by_id = {int(row["desc_id"]): row for row in load_jsonl(gt_jsonl)}
    desc_ids = [int(value) for value in cache["desc_ids"]]
    if set(desc_ids) != set(gt_by_id):
        raise ValueError("C4 cache/GT desc_id sets differ")
    # Match standalone_eval exactly: it casts predictions and GT timestamps to
    # float32 before temporal-IoU thresholding.
    gt_ts = np.asarray([gt_by_id[desc_id]["ts"] for desc_id in desc_ids], dtype=np.float32)
    starts = _matrix(cache, "start_time").astype(np.float32)
    ends = _matrix(cache, "end_time").astype(np.float32)
    gt_start = gt_ts[:, 0, None]
    gt_end = gt_ts[:, 1, None]
    intersection = np.maximum(0.0, np.minimum(ends, gt_end) - np.maximum(starts, gt_start))
    union = np.maximum(ends, gt_end) - np.minimum(starts, gt_start)
    temporal_iou = np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)
    is_gt = _matrix(cache, "is_gt_video") > 0.5
    cache["eval_y05"] = is_gt & (temporal_iou >= 0.5)
    cache["eval_y07"] = is_gt & (temporal_iou >= 0.7)


def evaluate_fast_config(cfg):
    cache, args, baseline = _CACHE, _ARGS, _BASELINE_FAST
    score, stats = score_config(cache, cfg)
    flat, score_matrix_values, order = fast_metrics(cache, score, args)
    deltas = {key: flat[key] - baseline["metrics"][key] for key in ALL_METRICS}
    base_scores = baseline["scores"]
    base_order = baseline["order"]
    inv = np.empty_like(order, dtype=np.int16)
    base_inv = np.empty_like(base_order, dtype=np.int16)
    rank_values = np.broadcast_to(np.arange(200, dtype=np.int16), order.shape)
    np.put_along_axis(inv, order, rank_values, axis=1)
    np.put_along_axis(base_inv, base_order, rank_values, axis=1)
    hard = (_matrix(cache, "is_gt_video") > 0.5) & (_matrix(cache, "iou") >= 0.5)
    hard_total = int(np.sum(hard))
    score_delta = score_matrix_values - base_scores
    is_gt = _matrix(cache, "is_gt_video") > 0.5
    movement = {
        "queries": int(order.shape[0]),
        "top1_changed_ratio": float(np.mean(order[:, 0] != base_order[:, 0])),
        "pearson_c31_c4": float(np.corrcoef(base_scores.ravel(), score_matrix_values.ravel())[0, 1]),
        "hard_positive_top100_exits": int(np.sum(hard & (base_inv < 100) & (inv >= 100))),
        "hard_positive_top100_entries": int(np.sum(hard & (base_inv >= 100) & (inv < 100))),
        "hard_positive_top100_exit_ratio": float(np.sum(hard & (base_inv < 100) & (inv >= 100)) / max(hard_total, 1)),
        "gt_video_score_delta_mean": float(np.mean(score_delta[is_gt])),
        "wrong_video_score_delta_mean": float(np.mean(score_delta[~is_gt])),
    }
    primary_positive_count = int(sum(1 for key in PRIMARY_METRICS if deltas[key] > 0))
    r100_min_delta = float(min(deltas["0.5-r100"], deltas["0.7-r100"]))
    return {
        **cfg,
        "feasible": primary_positive_count >= 4 and r100_min_delta >= -0.25,
        "metrics": flat,
        "selection_score_delta_vs_base": selection_score(flat, baseline["metrics"]),
        "stats": stats,
        "official_val_used": False,
        "post_val_adjustment": False,
        "movement_vs_c31": movement,
        "deltas_vs_base": deltas,
        "primary_positive_count": primary_positive_count,
        "r100_min_delta": r100_min_delta,
    }


def evaluate_many(configs, workers):
    if workers <= 1:
        return [evaluate_fast_config(cfg) for cfg in tqdm(configs, desc="C4-lite grid")]
    with mp.get_context("fork").Pool(workers) as pool:
        return list(tqdm(
            pool.imap_unordered(evaluate_fast_config, configs, chunksize=1),
            total=len(configs), desc="C4-lite grid",
        ))


def main():
    global _CACHE, _ARGS, _BASELINE_FAST
    args = parse_args()
    if args.effective_top_n != 100 or args.max_after_nms != 100 or not math.isclose(args.nms_thd, 0.7):
        raise ValueError("C4-lite protocol requires effective_top_n=100, max_after_nms=100, NMS=0.7")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache = load_cache(args.cache_npz)
    attach_eval_labels(cache, args.gt_jsonl)

    # Baseline is the exact frozen C3.1 score, not the original CONQUER base score.
    c31_score = cache["s_c31"].astype(np.float32)
    base_submission, base_metrics_raw, base_flat = eval_score(cache, c31_score, args)
    fast_base_flat, fast_base_scores, fast_base_order = fast_metrics(cache, c31_score, args)
    fast_diff = max(abs(round(fast_base_flat[key], 2) - base_flat[key]) for key in ALL_METRICS)
    if fast_diff > 1e-9:
        raise ValueError(f"Fast/bundled evaluator baseline mismatch: {fast_diff}")
    _CACHE, _ARGS = cache, args
    _BASELINE_FAST = {"metrics": fast_base_flat, "scores": fast_base_scores, "order": fast_base_order}
    configs = list(config_iter(args))
    results = evaluate_many(configs, args.workers)
    results.sort(key=lambda item: item["config_id"])
    best = None
    best_submission = None
    for rec in results:
        if rec["feasible"] and (best is None or rec["selection_score_delta_vs_base"] > best["selection_score_delta_vs_base"]):
            best = rec
    if best is None:
        best = max(results, key=lambda x: x["selection_score_delta_vs_base"])
    best_score, _ = score_config(cache, best)
    best_submission, bundled_best_raw, bundled_best_flat = eval_score(cache, best_score, args)
    best_fast_diff = max(abs(bundled_best_flat[key] - round(best["metrics"][key], 2)) for key in ALL_METRICS)
    if best_fast_diff > 1e-9:
        raise ValueError(f"Fast/bundled evaluator best mismatch: {best_fast_diff}")
    write_json(str(out_dir / "c4_lite_grid_results.json"), {
        "status": "PASS", "scope": "train_calib_only", "official_val_used": False,
        "post_val_adjustment": False, "baseline": "frozen_c31",
        "grid_size": len(results), "workers": args.workers,
        "fast_vs_bundled_baseline_max_abs_diff": fast_diff,
        "fast_vs_bundled_best_max_abs_diff": best_fast_diff,
        "cache_npz": args.cache_npz, "gt_jsonl": args.gt_jsonl,
        "base_metrics": base_flat, "best": best, "results": results,
    })
    # CSV summary.
    with open(out_dir / "c4_lite_grid_results.csv", "w", encoding="utf-8") as f:
        header = ["config_id", "family", "alpha_bd", "gamma_fp", "video_gain", "span_gain", "base_scale", "selection_score_delta_vs_base", "primary_positive_count", "r100_min_delta", "feasible"] + ALL_METRICS
        f.write(",".join(header) + "\n")
        for rec in results:
            row = [rec.get(k, "") for k in header[:11]] + [rec["metrics"].get(k, "") for k in ALL_METRICS]
            f.write(",".join(str(x) for x in row) + "\n")
    write_json(str(out_dir / "best_config.json"), best)
    if args.save_best_submission and best_submission is not None:
        write_json(args.save_best_submission, best_submission, pretty=False)
    print(json.dumps({"status": "PASS", "configs": len(results), "best": best}, indent=2))


if __name__ == "__main__":
    main()
