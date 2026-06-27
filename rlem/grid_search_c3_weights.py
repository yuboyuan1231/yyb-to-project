#!/usr/bin/env python

"""C3-strong train_calib-only weight search over frozen C2 evidence heads."""

import argparse
import csv
import hashlib
import itertools
import json
import math
import multiprocessing as mp
import os
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
if "bool" not in np.__dict__:
    np.bool = np.bool_
from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from rlem.evidence_dataset import FeatureStats
from rlem.evidence_model import EvidenceMLP
from rlem.io_utils import iter_jsonl, read_json, write_json
from standalone_eval.eval import compute_temporal_iou_batch, eval_retrieval, load_jsonl
from utils.inference_utils import filter_vcmr_by_nms


METRIC_KEYS = [
    "0.5-r1", "0.5-r5", "0.5-r10", "0.5-r100",
    "0.7-r1", "0.7-r5", "0.7-r10", "0.7-r100",
]
PRIMARY_KEYS = ["0.5-r1", "0.5-r5", "0.5-r10", "0.7-r1", "0.7-r5", "0.7-r10"]
R100_KEYS = ["0.5-r100", "0.7-r100"]
FAMILIES = ["A_additive", "B_gated_boundary"]
COARSE_GRID = {
    "base_scale": [0.75, 1.0, 1.25, 1.5],
    "a_joint": [0.25, 0.5, 0.75, 1.0, 1.5],
    "b_bd": [0.0, 0.25, 0.5, 0.75],
    "d_fp": [0.0, 0.25, 0.5, 0.75, 1.0],
}
SELECTION_RULE = {
    "primary_metrics_all_positive": True,
    "positive_delta_count_min": 6,
    "r100_delta_floor": -0.25,
    "hard_positive_top100_exit_ratio_max": 0.005,
    "top1_changed_ratio_max": 0.40,
    "pearson_min": 0.98,
    "spearman_mean_within_query_min": 0.90,
    "base_component_std_share_min": 0.60,
    "boundary_component_std_share_max": 0.20,
    "selection_score": "mean six R@1/R@5/R@10 deltas + 0.25 * mean two R@100 deltas",
    "family_b_preference_tolerance": 0.10,
    "refine_once_if_boundary_count_in_top10_min": 3,
}

_DATA = None
_BASELINE = None
_ARGS = None


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence_jsonl", required=True)
    parser.add_argument("--desc_ids_filter", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--dataset_config", required=True)
    parser.add_argument("--split", required=True, choices=["train"])
    parser.add_argument("--gt_jsonl", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--effective_top_n", type=int, default=100)
    parser.add_argument("--max_after_nms", type=int, default=100)
    parser.add_argument("--nms_thd", type=float, default=0.7)
    parser.add_argument("--rows_per_query", type=int, default=200)
    parser.add_argument("--score_batch_size", type=int, default=8192)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--workers", type=int, default=max(1, min(8, os.cpu_count() or 1)))
    parser.add_argument("--auto_refine", action="store_true")
    parser.add_argument("--max_configs", type=int, default=None, help="Smoke-test limiter; omit for accepted run")
    parser.add_argument("--seed", type=int, default=13)
    return parser.parse_args()


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_ids(path):
    with open(path, "r", encoding="utf-8") as f:
        ids = {line.strip() for line in f if line.strip()}
    if not ids:
        raise ValueError(f"No desc_ids in {path}")
    return ids


def load_model(path, device):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    stats = FeatureStats.from_dict(ckpt["feature_stats"])
    cfg = ckpt["model_cfg"]
    model = EvidenceMLP(cfg["input_dim"], cfg["hidden_dim"], cfg["dropout"])
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    return model, stats, ckpt


@torch.no_grad()
def load_and_score(args):
    allowed_ids = load_ids(args.desc_ids_filter)
    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    model, stats, ckpt = load_model(args.ckpt, device)
    rows_per_query = args.rows_per_query
    pred_chunks = {"q_joint": [], "q_bd": [], "e_fp": []}
    desc_ids = []
    descs = []
    arrays = {key: [] for key in [
        "s_base", "video_idx", "start", "end", "is_gt", "y05", "y07", "y_fp", "rank_base"
    ]}
    batch_rows = []
    current_id = None
    query_rows = 0
    completed = set()

    def flush():
        if not batch_rows:
            return
        x = np.stack([stats.transform_row(row) for row in batch_rows])
        pred = model.predict_scores(torch.from_numpy(x).float().to(device))
        pred_chunks["q_joint"].append(pred["q_joint"].cpu().numpy().astype(np.float32))
        pred_chunks["q_bd"].append(pred["q_bd"].cpu().numpy().astype(np.float32))
        pred_chunks["e_fp"].append(pred["e_fp"].cpu().numpy().astype(np.float32))
        batch_rows.clear()

    for row in tqdm(iter_jsonl(args.evidence_jsonl), desc="load+score train_calib evidence"):
        sid = str(row["desc_id"])
        if sid not in allowed_ids:
            raise ValueError(f"Evidence contains desc_id outside train_calib allowlist: {sid}")
        if current_id is None:
            current_id = sid
            desc_ids.append(row["desc_id"])
            descs.append(row.get("desc", ""))
        elif sid != current_id:
            if query_rows != rows_per_query:
                raise ValueError(f"desc_id={current_id} rows={query_rows}, expected={rows_per_query}")
            completed.add(current_id)
            if sid in completed:
                raise ValueError(f"Non-contiguous desc_id={sid}")
            current_id = sid
            query_rows = 0
            desc_ids.append(row["desc_id"])
            descs.append(row.get("desc", ""))
        query_rows += 1
        s_base = float(row["s_base"])
        if not math.isfinite(s_base) or s_base <= 0:
            raise ValueError(f"Invalid s_base={s_base} for desc_id={sid}")
        arrays["s_base"].append(s_base)
        arrays["video_idx"].append(int(row["video_idx"]))
        arrays["start"].append(float(row["start_time"]))
        arrays["end"].append(float(row["end_time"]))
        arrays["is_gt"].append(int(row["is_gt_video"]))
        arrays["y05"].append(int(row["y_joint_05"]))
        arrays["y07"].append(int(row["y_joint_07"]))
        arrays["y_fp"].append(int(row["y_fp"]))
        arrays["rank_base"].append(int(row["rank_base"]))
        batch_rows.append(row)
        if len(batch_rows) >= args.score_batch_size:
            flush()
    flush()
    if current_id is not None and query_rows != rows_per_query:
        raise ValueError(f"desc_id={current_id} rows={query_rows}, expected={rows_per_query}")
    found_ids = {str(value) for value in desc_ids}
    if found_ids != allowed_ids:
        raise ValueError(f"Evidence/filter desc_id mismatch: {len(found_ids)} vs {len(allowed_ids)}")
    n_queries = len(desc_ids)
    n_rows = n_queries * rows_per_query
    q_joint = np.concatenate(pred_chunks["q_joint"])
    q_bd = np.concatenate(pred_chunks["q_bd"])
    e_fp = np.concatenate(pred_chunks["e_fp"])
    if len(q_joint) != n_rows:
        raise ValueError(f"Scored rows={len(q_joint)}, expected={n_rows}")
    shaped = {
        "base": np.log(np.asarray(arrays["s_base"], dtype=np.float64)).reshape(n_queries, rows_per_query),
        "s_base_raw": np.asarray(arrays["s_base"], dtype=np.float64).reshape(n_queries, rows_per_query),
        "q_joint": q_joint.reshape(n_queries, rows_per_query),
        "q_bd": q_bd.reshape(n_queries, rows_per_query),
        "e_fp": e_fp.reshape(n_queries, rows_per_query),
        "video_idx": np.asarray(arrays["video_idx"], dtype=np.int32).reshape(n_queries, rows_per_query),
        "start": np.asarray(arrays["start"], dtype=np.float32).reshape(n_queries, rows_per_query),
        "end": np.asarray(arrays["end"], dtype=np.float32).reshape(n_queries, rows_per_query),
        "is_gt": np.asarray(arrays["is_gt"], dtype=np.bool_).reshape(n_queries, rows_per_query),
        "y05": np.asarray(arrays["y05"], dtype=np.bool_).reshape(n_queries, rows_per_query),
        "y07": np.asarray(arrays["y07"], dtype=np.bool_).reshape(n_queries, rows_per_query),
        "y_fp": np.asarray(arrays["y_fp"], dtype=np.bool_).reshape(n_queries, rows_per_query),
        "rank_base": np.asarray(arrays["rank_base"], dtype=np.int32).reshape(n_queries, rows_per_query),
        "desc_ids": desc_ids,
        "descs": descs,
        "checkpoint_epoch": int(ckpt["epoch"]),
        "feature_names": list(stats.feature_names),
    }
    expected_ranks = np.arange(1, rows_per_query + 1, dtype=np.int32)
    if not np.all(np.sort(shaped["rank_base"], axis=1) == expected_ranks):
        raise ValueError("rank_base is not a per-query permutation 1..200")
    del model, pred_chunks, batch_rows, arrays
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return shaped


def score_matrix(config, data):
    boundary = data["q_bd"] if config["family"] == "A_additive" else data["q_joint"] * data["q_bd"]
    scores = (
        config["base_scale"] * data["base"]
        + config["a_joint"] * data["q_joint"]
        + config["b_bd"] * boundary
        - config["d_fp"] * data["e_fp"]
    )
    return scores, boundary


def greedy_nms_indices(order, scores, video_idx, starts, ends, threshold, max_after):
    groups = {}
    for idx in order:
        groups.setdefault(int(video_idx[idx]), []).append(int(idx))
    kept = []
    for indices in groups.values():
        local_kept = []
        for idx in indices:
            suppress = False
            for prior in local_kept:
                intersection = max(0.0, min(float(ends[prior]), float(ends[idx])) - max(float(starts[prior]), float(starts[idx])))
                union = max(float(ends[prior]), float(ends[idx])) - min(float(starts[prior]), float(starts[idx]))
                iou = intersection / union if union > 0 else 0.0
                if iou > threshold:
                    suppress = True
                    break
            if not suppress:
                local_kept.append(idx)
        kept.extend(local_kept)
    kept.sort(key=lambda idx: float(scores[idx]), reverse=True)
    return kept[:max_after]


def raw_metrics_from_selected(selected, y05, y07):
    correct = {key: 0 for key in METRIC_KEYS}
    for qi, indices in enumerate(selected):
        labels = {"0.5": y05[qi, indices], "0.7": y07[qi, indices]}
        for iou in ["0.5", "0.7"]:
            for k in [1, 5, 10, 100]:
                correct[f"{iou}-r{k}"] += int(np.any(labels[iou][:k]))
    n = len(selected)
    return {key: 100.0 * value / n for key, value in correct.items()}


def evaluate_config(config, return_selected=False):
    data, baseline, args = _DATA, _BASELINE, _ARGS
    scores, boundary = score_matrix(config, data)
    order = np.argsort(-scores, axis=1, kind="stable")
    inv = np.empty_like(order, dtype=np.int32)
    rank_values = np.broadcast_to(np.arange(args.rows_per_query, dtype=np.int32), order.shape)
    np.put_along_axis(inv, order, rank_values, axis=1)
    selected = []
    for qi in range(order.shape[0]):
        selected.append(greedy_nms_indices(
            order[qi, :args.effective_top_n], scores[qi], data["video_idx"][qi],
            data["start"][qi], data["end"][qi], args.nms_thd, args.max_after_nms,
        ))
    raw_metrics = raw_metrics_from_selected(selected, data["eval_y05"], data["eval_y07"])
    delta = {key: raw_metrics[key] - baseline["raw_metrics"][key] for key in METRIC_KEYS}
    base_inv = baseline["inv"]
    hard = data["y05"]
    fp = data["y_fp"]
    hard_exits = int(np.sum(hard & (base_inv < args.effective_top_n) & (inv >= args.effective_top_n)))
    hard_entries = int(np.sum(hard & (base_inv >= args.effective_top_n) & (inv < args.effective_top_n)))
    wrong = ~data["is_gt"]
    weighted_components = {
        "base": config["base_scale"] * data["base"],
        "q_joint": config["a_joint"] * data["q_joint"],
        "boundary": config["b_bd"] * boundary,
        "negative_e_fp": -config["d_fp"] * data["e_fp"],
    }
    component_std = {key: float(np.std(value)) for key, value in weighted_components.items()}
    std_total = sum(component_std.values())
    std_share = {key: value / std_total for key, value in component_std.items()}
    base_flat = data["base"].ravel()
    score_flat = scores.ravel()
    pearson = float(np.corrcoef(base_flat, score_flat)[0, 1])
    rank_diff = baseline["inv"].astype(np.float64) - inv.astype(np.float64)
    n = args.rows_per_query
    per_query_spearman = 1.0 - 6.0 * np.sum(rank_diff * rank_diff, axis=1) / (n * (n * n - 1.0))
    metrics = {key: round(raw_metrics[key], 2) for key in METRIC_KEYS}
    primary_deltas = [delta[key] for key in PRIMARY_KEYS]
    r100_deltas = [delta[key] for key in R100_KEYS]
    result = {
        **config,
        "metrics": metrics,
        "raw_metrics": raw_metrics,
        "delta_vs_s_base": {key: round(delta[key], 6) for key in METRIC_KEYS},
        "mean_delta_vs_s_base": float(np.mean(list(delta.values()))),
        "primary_mean_delta": float(np.mean(primary_deltas)),
        "r100_mean_delta": float(np.mean(r100_deltas)),
        "min_primary_delta": float(min(primary_deltas)),
        "positive_delta_count": int(sum(value > 0 for value in delta.values())),
        "selection_score": float(np.mean(primary_deltas) + 0.25 * np.mean(r100_deltas)),
        "top1_changed_ratio": float(np.mean(order[:, 0] != baseline["order"][:, 0])),
        "hard_positive_top100_exits": hard_exits,
        "hard_positive_top100_entries": hard_entries,
        "hard_positive_top100_exit_ratio": hard_exits / int(np.sum(hard)),
        "y_fp_down_move_ratio": float(np.mean(inv[fp] > base_inv[fp])),
        "y_joint_05_down_move_ratio": float(np.mean(inv[hard] > base_inv[hard])),
        "wrong_video_qbd_bonus_mean": float(np.mean((config["b_bd"] * boundary)[wrong])),
        "wrong_video_qbd_bonus_mean_normalized_by_base_scale": float(
            np.mean((config["b_bd"] * boundary)[wrong]) / config["base_scale"]
        ),
        "weighted_component_std": component_std,
        "weighted_component_std_share": std_share,
        "pearson_base_vs_c3": pearson,
        "spearman_mean_within_query": float(np.mean(per_query_spearman)),
    }
    constraints = {
        "primary_metrics_all_positive": all(value > 0 for value in primary_deltas),
        "positive_delta_count_min_6": result["positive_delta_count"] >= 6,
        "r100_delta_floor": min(r100_deltas) >= SELECTION_RULE["r100_delta_floor"],
        "hard_positive_exit_ratio": result["hard_positive_top100_exit_ratio"] <= SELECTION_RULE["hard_positive_top100_exit_ratio_max"],
        "top1_changed_ratio": result["top1_changed_ratio"] <= SELECTION_RULE["top1_changed_ratio_max"],
        "pearson": pearson >= SELECTION_RULE["pearson_min"],
        "spearman": result["spearman_mean_within_query"] >= SELECTION_RULE["spearman_mean_within_query_min"],
        "base_component_dominant": (
            std_share["base"] >= SELECTION_RULE["base_component_std_share_min"]
            and std_share["base"] == max(std_share.values())
        ),
        "boundary_component_not_dominant": std_share["boundary"] <= SELECTION_RULE["boundary_component_std_share_max"],
    }
    result["constraints"] = constraints
    result["feasible"] = bool(all(constraints.values()))
    if return_selected:
        return result, selected, scores, order
    return result


def build_baseline(data, args):
    scores = data["base"]
    order = np.argsort(-scores, axis=1, kind="stable")
    inv = np.empty_like(order, dtype=np.int32)
    ranks = np.broadcast_to(np.arange(args.rows_per_query, dtype=np.int32), order.shape)
    np.put_along_axis(inv, order, ranks, axis=1)
    selected = [
        greedy_nms_indices(
            order[qi, :args.effective_top_n], scores[qi], data["video_idx"][qi],
            data["start"][qi], data["end"][qi], args.nms_thd, args.max_after_nms,
        )
        for qi in range(order.shape[0])
    ]
    return {
        "order": order,
        "inv": inv,
        "selected": selected,
        "raw_metrics": raw_metrics_from_selected(selected, data["eval_y05"], data["eval_y07"]),
    }


def generate_coarse_configs(max_configs=None):
    configs = []
    index = 0
    for family, base_scale, a_joint, b_bd, d_fp in itertools.product(
        FAMILIES, COARSE_GRID["base_scale"], COARSE_GRID["a_joint"],
        COARSE_GRID["b_bd"], COARSE_GRID["d_fp"],
    ):
        configs.append({
            "config_id": f"coarse_{index:04d}", "phase": "coarse", "family": family,
            "base_scale": base_scale, "a_joint": a_joint, "b_bd": b_bd, "d_fp": d_fp,
        })
        index += 1
    return configs[:max_configs] if max_configs is not None else configs


def result_sort_key(result):
    return (
        result["selection_score"], result["min_primary_delta"],
        -result["hard_positive_top100_exit_ratio"], result["spearman_mean_within_query"],
        -result["top1_changed_ratio"],
    )


def select_best(results):
    feasible = [result for result in results if result["feasible"]]
    if not feasible:
        return None, {"reason": "no_feasible_config"}
    leaders = {}
    for family in FAMILIES:
        family_results = [result for result in feasible if result["family"] == family]
        if family_results:
            leaders[family] = max(family_results, key=result_sort_key)
    best_a = leaders.get("A_additive")
    best_b = leaders.get("B_gated_boundary")
    prefer_b = bool(
        best_b is not None and (
            best_a is None
            or best_b["selection_score"] >= best_a["selection_score"] - SELECTION_RULE["family_b_preference_tolerance"]
        )
    )
    chosen_family = "B_gated_boundary" if prefer_b else "A_additive"
    chosen = max(
        [result for result in feasible if result["family"] == chosen_family],
        key=result_sort_key,
    )
    return chosen, {
        "reason": "family_b_within_tolerance" if prefer_b else "family_a_exceeds_b_tolerance",
        "family_leaders": leaders,
    }


def boundary_trigger(results, selected):
    feasible = sorted([result for result in results if result["feasible"]], key=result_sort_key, reverse=True)[:10]
    triggers = []
    for parameter, values in COARSE_GRID.items():
        low, high = min(values), max(values)
        for boundary_name, boundary_value in [("low", low), ("high", high)]:
            if boundary_name == "low" and parameter in {"b_bd", "d_fp"} and boundary_value == 0:
                continue
            count = sum(math.isclose(result[parameter], boundary_value) for result in feasible)
            if math.isclose(selected[parameter], boundary_value) and count >= SELECTION_RULE["refine_once_if_boundary_count_in_top10_min"]:
                triggers.append({"parameter": parameter, "boundary": boundary_name, "value": boundary_value, "top10_count": count})
    return triggers


def generate_refined_configs(selected, existing):
    half_step = {"base_scale": 0.125, "a_joint": 0.125, "b_bd": 0.125, "d_fp": 0.125}
    value_sets = {}
    for parameter, step in half_step.items():
        center = float(selected[parameter])
        values = sorted({round(max(0.0, center - step), 6), round(center, 6), round(center + step, 6)})
        if parameter == "base_scale":
            values = [value for value in values if value > 0]
        value_sets[parameter] = values
    existing_keys = {
        (item["family"], item["base_scale"], item["a_joint"], item["b_bd"], item["d_fp"])
        for item in existing
    }
    refined = []
    for values in itertools.product(
        value_sets["base_scale"], value_sets["a_joint"], value_sets["b_bd"], value_sets["d_fp"]
    ):
        key = (selected["family"], *values)
        if key in existing_keys:
            continue
        refined.append({
            "config_id": f"refine_{len(refined):04d}", "phase": "refine_once",
            "family": selected["family"], "base_scale": values[0], "a_joint": values[1],
            "b_bd": values[2], "d_fp": values[3],
        })
    return refined, value_sets


def evaluate_many(configs, workers):
    if workers <= 1:
        return [evaluate_config(config) for config in tqdm(configs, desc="grid configs")]
    ctx = mp.get_context("fork")
    with ctx.Pool(processes=workers) as pool:
        return list(tqdm(
            pool.imap_unordered(evaluate_config, configs, chunksize=1),
            total=len(configs), desc="grid configs",
        ))


def load_video2idx(dataset_config, split):
    cfg = read_json(dataset_config)
    path = cfg["video_duration_idx_path"]
    if not os.path.isabs(path):
        path = os.path.join(cfg["root_path"], path)
    return {name: int(value[1]) for name, value in read_json(path)[split].items()}


def attach_evaluator_labels(data, gt, dataset_config, split):
    """Precompute correctness with the bundled evaluator's float32 semantics."""
    video2idx = load_video2idx(dataset_config, split)
    gt_by_id = {str(item["desc_id"]): item for item in gt}
    shape = data["video_idx"].shape
    eval_y05 = np.zeros(shape, dtype=np.bool_)
    eval_y07 = np.zeros(shape, dtype=np.bool_)
    for qi, desc_id in enumerate(data["desc_ids"]):
        item = gt_by_id[str(desc_id)]
        video_match = data["video_idx"][qi] == video2idx[item["vid_name"]]
        predictions = np.stack([data["start"][qi], data["end"][qi]], axis=1).astype(np.float32)
        timestamps = item["ts"]
        if len(timestamps) >= 4:
            correct05 = []
            correct07 = []
            for timestamp in timestamps:
                iou = compute_temporal_iou_batch(
                    predictions, np.asarray(timestamp, dtype=np.float32)
                ) * video_match
                correct05.append(iou >= 0.5)
                correct07.append(iou >= 0.7)
            eval_y05[qi] = np.sum(correct05, axis=0) >= 2
            eval_y07[qi] = np.sum(correct07, axis=0) >= 2
        else:
            iou = compute_temporal_iou_batch(
                predictions, np.asarray(timestamps, dtype=np.float32)
            ) * video_match
            eval_y05[qi] = iou >= 0.5
            eval_y07[qi] = iou >= 0.7
    data["eval_y05"] = eval_y05
    data["eval_y07"] = eval_y07
    data["evaluator_label_audit"] = {
        "y05_disagreements_vs_c1_export_label": int(np.sum(eval_y05 != data["y05"])),
        "y07_disagreements_vs_c1_export_label": int(np.sum(eval_y07 != data["y07"])),
        "rows": int(eval_y05.size),
        "semantics": "bundled evaluator float32 timestamp and IoU threshold semantics",
    }


def build_submission(config, selected, scores, data, args):
    vcmr = []
    for qi, indices in enumerate(selected):
        predictions = [
            [
                int(data["video_idx"][qi, idx]), float(data["start"][qi, idx]),
                float(data["end"][qi, idx]), float(scores[qi, idx]),
            ]
            for idx in indices
        ]
        vcmr.append({"desc_id": data["desc_ids"][qi], "desc": data["descs"][qi], "predictions": predictions})
    return {"video2idx": load_video2idx(args.dataset_config, args.split), "VCMR": vcmr}


def build_official_submission(config, data, args):
    scores, _ = score_matrix(config, data)
    order = np.argsort(-scores, axis=1, kind="stable")
    vcmr = []
    for qi in range(order.shape[0]):
        predictions = [
            [
                int(data["video_idx"][qi, idx]), float(data["start"][qi, idx]),
                float(data["end"][qi, idx]), float(scores[qi, idx]),
            ]
            for idx in order[qi, :args.effective_top_n]
        ]
        predictions = filter_vcmr_by_nms(
            predictions, nms_threshold=args.nms_thd,
            max_before_nms=args.effective_top_n, max_after_nms=args.max_after_nms,
        )
        vcmr.append({"desc_id": data["desc_ids"][qi], "desc": data["descs"][qi], "predictions": predictions})
    return {"video2idx": load_video2idx(args.dataset_config, args.split), "VCMR": vcmr}


def write_csv(path, results):
    fields = [
        "config_id", "phase", "family", "base_scale", "a_joint", "b_bd", "d_fp",
        *METRIC_KEYS, "mean_delta_vs_S_base", "primary_mean_delta", "r100_mean_delta",
        "selection_score", "positive_delta_count", "top1_changed_ratio",
        "hard_positive_top100_exits", "hard_positive_top100_entries",
        "hard_positive_top100_exit_ratio", "y_fp_down_move_ratio",
        "y_joint_05_down_move_ratio", "wrong_video_qbd_bonus_mean",
        "base_std_share", "q_joint_std_share", "boundary_std_share", "negative_e_fp_std_share",
        "pearson_base_vs_c3", "spearman_mean_within_query", "feasible",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for result in results:
            row = {key: result.get(key) for key in fields}
            row.update(result["metrics"])
            shares = result["weighted_component_std_share"]
            row.update({
                "base_std_share": shares["base"], "q_joint_std_share": shares["q_joint"],
                "boundary_std_share": shares["boundary"],
                "negative_e_fp_std_share": shares["negative_e_fp"],
            })
            writer.writerow(row)


def main():
    global _DATA, _BASELINE, _ARGS
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    forbidden = (Path("results/rlem_c1/val_evidence.jsonl.gz").resolve())
    if Path(args.evidence_jsonl).resolve() == forbidden:
        raise ValueError("Official val evidence is forbidden for C3-strong")

    started = time.time()
    data = load_and_score(args)
    gt = load_jsonl(args.gt_jsonl)
    gt_ids = {str(item["desc_id"]) for item in gt}
    if gt_ids != {str(value) for value in data["desc_ids"]}:
        raise ValueError("GT/evidence desc_id mismatch")
    attach_evaluator_labels(data, gt, args.dataset_config, args.split)
    _DATA, _ARGS = data, args
    baseline = build_baseline(data, args)
    _BASELINE = baseline

    coarse_configs = generate_coarse_configs(args.max_configs)
    coarse_results = evaluate_many(coarse_configs, args.workers)
    preliminary, preliminary_meta = select_best(coarse_results)
    if preliminary is None:
        raise RuntimeError("No feasible coarse-grid configuration")
    triggers = boundary_trigger(coarse_results, preliminary) if args.max_configs is None else []
    refined_configs, refined_values, refined_results = [], {}, []
    if args.auto_refine and triggers:
        refined_configs, refined_values = generate_refined_configs(preliminary, coarse_configs)
        refined_results = evaluate_many(refined_configs, args.workers)
    all_results = sorted(coarse_results + refined_results, key=lambda item: item["config_id"])
    best, selection_meta = select_best(all_results)
    if best is None:
        raise RuntimeError("No feasible final configuration")

    best_result, best_selected, best_scores, _ = evaluate_config(best, return_selected=True)
    best_submission = build_submission(best, best_selected, best_scores, data, args)
    best_submission_path = output_dir / "best_train_calib_submission.json"
    write_json(best_submission_path, best_submission, pretty=False)
    official_best_submission = build_official_submission(best, data, args)
    if official_best_submission["VCMR"] != best_submission["VCMR"]:
        raise ValueError("Fast NMS best submission differs from frozen official NMS")
    official_metrics = eval_retrieval(
        best_submission, gt, iou_thds=(0.5, 0.7), verbose=False,
        match_number=True, use_desc_type=False,
    )["VCMR"]
    if official_metrics != best_result["metrics"]:
        raise ValueError(f"Fast/official best metrics differ: {best_result['metrics']} vs {official_metrics}")

    base_config = {
        "family": "A_additive", "base_scale": 1.0, "a_joint": 0.0, "b_bd": 0.0, "d_fp": 0.0,
    }
    default_config = {
        "config_id": "c3_minimal_default", "phase": "reference", "family": "A_additive",
        "base_scale": 1.0, "a_joint": 1.0, "b_bd": 0.5, "d_fp": 0.5,
    }
    base_submission = build_official_submission(base_config, data, args)
    default_submission = build_official_submission(default_config, data, args)
    base_official_metrics = eval_retrieval(
        base_submission, gt, iou_thds=(0.5, 0.7), verbose=False,
        match_number=True, use_desc_type=False,
    )["VCMR"]
    default_official_metrics = eval_retrieval(
        default_submission, gt, iou_thds=(0.5, 0.7), verbose=False,
        match_number=True, use_desc_type=False,
    )["VCMR"]
    baseline_metrics = {key: round(value, 2) for key, value in baseline["raw_metrics"].items()}
    if base_official_metrics != baseline_metrics:
        raise ValueError("Fast baseline metrics differ from official evaluator")
    default_grid = next((
        result for result in coarse_results
        if result["family"] == "A_additive" and all(
            math.isclose(result[key], default_config[key])
            for key in ["base_scale", "a_joint", "b_bd", "d_fp"]
        )
    ), None)
    if default_grid is None:
        default_grid = evaluate_config(default_config)
    if default_official_metrics != default_grid["metrics"]:
        raise ValueError("Fast default metrics differ from official evaluator")

    best_vs_default = {
        key: round(best_result["metrics"][key] - default_official_metrics[key], 6)
        for key in METRIC_KEYS
    }
    best_config = {
        "status": "PASS",
        "stage": "C3-strong",
        "scope": "train_calib_only",
        "official_val_used": False,
        "selection_rule": SELECTION_RULE,
        "selection_metadata": selection_meta,
        "config": {key: best_result[key] for key in ["config_id", "phase", "family", "base_scale", "a_joint", "b_bd", "d_fp"]},
        "metrics": best_result["metrics"],
        "delta_vs_s_base": best_result["delta_vs_s_base"],
        "delta_vs_c3_minimal_default": best_vs_default,
        "diagnostics": {key: best_result[key] for key in [
            "selection_score", "top1_changed_ratio", "hard_positive_top100_exits",
            "hard_positive_top100_entries", "hard_positive_top100_exit_ratio",
            "y_fp_down_move_ratio", "y_joint_05_down_move_ratio",
            "wrong_video_qbd_bonus_mean", "weighted_component_std",
            "weighted_component_std_share", "pearson_base_vs_c3",
            "spearman_mean_within_query", "constraints", "feasible",
        ]},
        "checkpoint_sha256": sha256_file(args.ckpt),
        "evidence_sha256": sha256_file(args.evidence_jsonl),
        "desc_ids_sha256": sha256_file(args.desc_ids_filter),
        "submission_path": str(best_submission_path),
    }
    write_json(output_dir / "best_config.json", best_config)

    grid_obj = {
        "status": "PASS",
        "stage": "C3-strong",
        "scope": "train_calib_only",
        "official_val_used": False,
        "coarse_grid": COARSE_GRID,
        "selection_rule": SELECTION_RULE,
        "coarse_config_count": len(coarse_results),
        "boundary_refinement_trigger": triggers,
        "refinement_values": refined_values,
        "refined_config_count": len(refined_results),
        "total_config_count": len(all_results),
        "baseline_metrics": base_official_metrics,
        "evaluator_label_audit": data["evaluator_label_audit"],
        "c3_minimal_default_metrics": default_official_metrics,
        "preliminary_selection": preliminary_meta,
        "final_selection": selection_meta,
        "results": all_results,
    }
    write_json(output_dir / "train_calib_grid_results.json", grid_obj)
    write_csv(output_dir / "train_calib_grid_results.csv", all_results)

    family_leaders = selection_meta.get("family_leaders", {})
    c3_pass = bool(
        best_result["feasible"]
        and (
            best_result["selection_score"] >= default_grid["selection_score"] - 0.10
            or best_result["hard_positive_top100_exit_ratio"] < default_grid["hard_positive_top100_exit_ratio"]
        )
    )
    diagnostics = {
        "status": "PASS" if c3_pass else "FAIL",
        "stage": "C3-strong",
        "scope": "train_calib_only",
        "official_val_used": False,
        "weight_search_performed_on": "train_calib_only",
        "coarse_config_count": len(coarse_results),
        "refined_config_count": len(refined_results),
        "total_config_count": len(all_results),
        "boundary_refinement_trigger": triggers,
        "baseline_metrics": base_official_metrics,
        "evaluator_label_audit": data["evaluator_label_audit"],
        "c3_minimal_default_metrics": default_official_metrics,
        "best_metrics": best_result["metrics"],
        "best_delta_vs_s_base": best_result["delta_vs_s_base"],
        "best_delta_vs_c3_minimal_default": best_vs_default,
        "best_config": best_config["config"],
        "best_diagnostics": best_config["diagnostics"],
        "family_leaders": family_leaders,
        "fast_grid_evaluator_validation": {
            "baseline_exact_to_official_evaluator": True,
            "c3_minimal_default_exact_to_official_evaluator": True,
            "best_exact_to_official_evaluator": True,
            "best_submission_exact_to_official_nms": True,
        },
        "runtime_seconds": time.time() - started,
        "recommend_official_val_one_shot": bool(c3_pass),
        "official_val_one_shot_executed": False,
        "next_stage_authorized": False,
    }
    write_json(output_dir / "c3_strong_diagnostics.json", diagnostics)
    metric_rows = "\n".join(
        f"| {key} | {base_official_metrics[key]:.2f} | {default_official_metrics[key]:.2f} | {best_result['metrics'][key]:.2f} | {best_result['metrics'][key]-base_official_metrics[key]:+.2f} | {best_vs_default[key]:+.2f} |"
        for key in METRIC_KEYS
    )
    md = f"""# C3-strong train_calib-only diagnostics

Status: `{'PASS' if c3_pass else 'FAIL'}`  
Official val used: `false`  
Official-val one-shot executed: `false`

## Search

- Coarse configs: {len(coarse_results)}
- Refined configs: {len(refined_results)}
- Total configs: {len(all_results)}
- Boundary refinement trigger: `{json.dumps(triggers, ensure_ascii=False)}`
- Best family: `{best_result['family']}`
- Best weights: base={best_result['base_scale']}, joint={best_result['a_joint']}, boundary={best_result['b_bd']}, fp={best_result['d_fp']}

## Metrics

| Metric | S_base | C3-minimal | C3-strong | vs base | vs minimal |
|---|---:|---:|---:|---:|---:|
{metric_rows}

## Best stability diagnostics

- Selection score: {best_result['selection_score']:.6f}
- Top-1 changed ratio: {100*best_result['top1_changed_ratio']:.2f}%
- Hard-positive top-100 exits/entries: {best_result['hard_positive_top100_exits']}/{best_result['hard_positive_top100_entries']}
- Hard-positive exit ratio: {100*best_result['hard_positive_top100_exit_ratio']:.4f}%
- y_fp down-move ratio: {100*best_result['y_fp_down_move_ratio']:.2f}%
- y_joint_05 down-move ratio: {100*best_result['y_joint_05_down_move_ratio']:.2f}%
- Wrong-video boundary bonus mean: {best_result['wrong_video_qbd_bonus_mean']:.6f}
- Base component std share: {100*best_result['weighted_component_std_share']['base']:.2f}%
- Boundary component std share: {100*best_result['weighted_component_std_share']['boundary']:.2f}%
- Pearson(base, C3): {best_result['pearson_base_vs_c3']:.6f}
- Mean within-query Spearman(base, C3): {best_result['spearman_mean_within_query']:.6f}

The fast grid evaluator was checked exactly against the frozen NMS and bundled evaluator
for S_base, C3-minimal default, and the selected best configuration.

`recommend_official_val_one_shot = {str(bool(c3_pass)).lower()}` is a recommendation only.
`next_stage_authorized = false`. Stop here.
"""
    (output_dir / "c3_strong_diagnostics.md").write_text(md, encoding="utf-8")
    print(json.dumps(diagnostics, indent=2))


if __name__ == "__main__":
    main()
