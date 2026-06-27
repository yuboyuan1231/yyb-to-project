#!/usr/bin/env python

"""Fast train-calib-only C3.1 score and temperature search.

The script consumes only the frozen train_calib metadata cache and cached
logits from one C3.1 model.  It deliberately has no evidence-jsonl argument,
which makes accidental official-val use impossible through this entrypoint.
"""

import argparse
import csv
import itertools
import json
import math
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from rlem.io_utils import write_json
from rlem.grid_search_c3_weights import greedy_nms_indices, raw_metrics_from_selected


METRIC_KEYS = [
    "0.5-r1", "0.5-r5", "0.5-r10", "0.5-r100",
    "0.7-r1", "0.7-r5", "0.7-r10", "0.7-r100",
]
PRIMARY_KEYS = ["0.5-r1", "0.5-r5", "0.5-r10", "0.7-r1", "0.7-r5", "0.7-r10"]
R100_KEYS = ["0.5-r100", "0.7-r100"]
FAMILIES = ["A_additive", "B_gated_boundary", "C_centered_gated_boundary"]
GRID = {
    "base_scale": [0.5, 0.75, 1.0, 1.25],
    "a_joint": [0.75, 1.0, 1.25, 1.5, 1.75],
    "b_bd": [0.0, 0.25, 0.5, 0.75, 1.0],
    "d_fp": [0.25, 0.5, 0.75, 1.0],
}
TEMPERATURES = {
    "t_joint": [0.75, 1.0, 1.25, 1.5],
    "t_fp": [0.75, 1.0, 1.25, 1.5],
    "t_bd": [1.0, 1.25],
}
SAFETY = {
    "r100_delta_floor": -0.25,
    "hard_positive_top100_exit_ratio_max": 0.005,
    "top1_changed_ratio_max": 0.40,
    "pearson_min": 0.98,
    "spearman_mean_within_query_min": 0.90,
    "base_component_std_share_min": 0.60,
    "boundary_component_std_share_max": 0.20,
}

_DATA = None
_BASELINE = None
_ARGS = None


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata_npz", required=True)
    parser.add_argument("--prediction_cache", required=True)
    parser.add_argument("--model_id", required=True)
    parser.add_argument("--mode", choices=["probe", "full", "calibrate", "custom"], required=True)
    parser.add_argument("--configs_json", help="Seed configs for calibrate/custom modes")
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--workers", type=int, default=max(1, min(8, os.cpu_count() or 1)))
    parser.add_argument("--effective_top_n", type=int, default=100)
    parser.add_argument("--max_after_nms", type=int, default=100)
    parser.add_argument("--nms_thd", type=float, default=0.7)
    parser.add_argument("--rows_per_query", type=int, default=200)
    parser.add_argument("--max_configs", type=int, default=None)
    return parser.parse_args()


def sigmoid_temperature(logit, temperature):
    value = np.clip(logit.astype(np.float64) / float(temperature), -40.0, 40.0)
    return (1.0 / (1.0 + np.exp(-value))).astype(np.float32)


def load_data(args):
    with np.load(args.metadata_npz, allow_pickle=False) as meta:
        data = {key: meta[key] for key in meta.files}
    with np.load(args.prediction_cache, allow_pickle=False) as pred:
        logits = {key: pred[key] for key in pred.files}
    required_meta = {
        "base_log", "s_base", "video_idx", "start", "end", "rank_base",
        "is_gt", "y05_export", "y_fp", "eval_y05", "eval_y07", "desc_ids",
    }
    required_pred = {"q_joint_logit", "q_bd_logit", "e_fp_logit"}
    if not required_meta <= set(data) or not required_pred <= set(logits):
        raise ValueError("Incomplete C3.1 metadata or prediction cache")
    shape = data["base_log"].shape
    if len(shape) != 2 or shape[1] != args.rows_per_query:
        raise ValueError(f"Unexpected metadata shape {shape}")
    for key, value in logits.items():
        if value.size != np.prod(shape):
            raise ValueError(f"Prediction rows mismatch for {key}")
        logits[key] = value.reshape(shape)
    data["logits"] = logits
    return data


def calibrated_heads(config, data):
    return (
        sigmoid_temperature(data["logits"]["q_joint_logit"], config.get("t_joint", 1.0)),
        sigmoid_temperature(data["logits"]["q_bd_logit"], config.get("t_bd", 1.0)),
        sigmoid_temperature(data["logits"]["e_fp_logit"], config.get("t_fp", 1.0)),
    )


def score_matrix(config, data):
    q_joint, q_bd, e_fp = calibrated_heads(config, data)
    if config["family"] == "A_additive":
        boundary = q_bd
    elif config["family"] == "B_gated_boundary":
        boundary = q_joint * q_bd
    elif config["family"] == "C_centered_gated_boundary":
        boundary = q_joint * (q_bd - float(np.mean(q_bd)))
    else:
        raise ValueError(f"Unknown family: {config['family']}")
    scores = (
        float(config["base_scale"]) * data["base_log"]
        + float(config["a_joint"]) * q_joint
        + float(config["b_bd"]) * boundary
        - float(config["d_fp"]) * e_fp
    )
    return scores, q_joint, q_bd, e_fp, boundary


def build_baseline(data, args):
    scores = data["base_log"]
    order = np.argsort(-scores, axis=1, kind="stable")
    inv = np.empty_like(order, dtype=np.int16)
    ranks = np.broadcast_to(np.arange(args.rows_per_query, dtype=np.int16), order.shape)
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


def evaluate_config(config, return_selected=False):
    data, baseline, args = _DATA, _BASELINE, _ARGS
    scores, q_joint, q_bd, e_fp, boundary = score_matrix(config, data)
    order = np.argsort(-scores, axis=1, kind="stable")
    inv = np.empty_like(order, dtype=np.int16)
    ranks = np.broadcast_to(np.arange(args.rows_per_query, dtype=np.int16), order.shape)
    np.put_along_axis(inv, order, ranks, axis=1)
    selected = [
        greedy_nms_indices(
            order[qi, :args.effective_top_n], scores[qi], data["video_idx"][qi],
            data["start"][qi], data["end"][qi], args.nms_thd, args.max_after_nms,
        )
        for qi in range(order.shape[0])
    ]
    raw = raw_metrics_from_selected(selected, data["eval_y05"], data["eval_y07"])
    delta = {key: raw[key] - baseline["raw_metrics"][key] for key in METRIC_KEYS}
    hard, fp = data["y05_export"], data["y_fp"]
    hard_exits = int(np.sum(hard & (baseline["inv"] < args.effective_top_n) & (inv >= args.effective_top_n)))
    hard_entries = int(np.sum(hard & (baseline["inv"] >= args.effective_top_n) & (inv < args.effective_top_n)))
    components = {
        "base": float(config["base_scale"]) * data["base_log"],
        "q_joint": float(config["a_joint"]) * q_joint,
        "boundary": float(config["b_bd"]) * boundary,
        "negative_e_fp": -float(config["d_fp"]) * e_fp,
    }
    component_std = {key: float(np.std(value)) for key, value in components.items()}
    std_sum = sum(component_std.values())
    std_share = {key: value / std_sum for key, value in component_std.items()}
    pearson = float(np.corrcoef(data["base_log"].ravel(), scores.ravel())[0, 1])
    rank_diff = baseline["inv"].astype(np.float32) - inv.astype(np.float32)
    n = args.rows_per_query
    spearman = 1.0 - 6.0 * np.sum(rank_diff * rank_diff, axis=1) / (n * (n * n - 1.0))
    primary = [delta[key] for key in PRIMARY_KEYS]
    r100 = [delta[key] for key in R100_KEYS]
    result = {
        **config,
        "model_id": args.model_id,
        "metrics": {key: round(raw[key], 2) for key in METRIC_KEYS},
        "raw_metrics": raw,
        "delta_vs_s_base": {key: round(delta[key], 6) for key in METRIC_KEYS},
        "positive_delta_count": int(sum(value > 0 for value in delta.values())),
        "min_primary_delta": float(min(primary)),
        "primary_mean_delta": float(np.mean(primary)),
        "r100_mean_delta": float(np.mean(r100)),
        "selection_score": float(np.mean(primary) + 0.25 * np.mean(r100)),
        "top1_changed_ratio": float(np.mean(order[:, 0] != baseline["order"][:, 0])),
        "hard_positive_top100_exits": hard_exits,
        "hard_positive_top100_entries": hard_entries,
        "hard_positive_top100_exit_ratio": hard_exits / int(np.sum(hard)),
        "y_fp_down_move_ratio": float(np.mean(inv[fp] > baseline["inv"][fp])),
        "y_joint_05_down_move_ratio": float(np.mean(inv[hard] > baseline["inv"][hard])),
        "wrong_video_boundary_bonus_mean": float(np.mean((float(config["b_bd"]) * boundary)[~data["is_gt"]])),
        "mean_qbd_train_calib": float(np.mean(q_bd)),
        "weighted_component_std": component_std,
        "weighted_component_std_share": std_share,
        "pearson_base_vs_c3": pearson,
        "spearman_mean_within_query": float(np.mean(spearman)),
    }
    constraints = {
        "primary_metrics_all_positive": all(value > 0 for value in primary),
        "positive_delta_count_min_6": result["positive_delta_count"] >= 6,
        "r100_delta_floor": min(r100) >= SAFETY["r100_delta_floor"],
        "hard_positive_exit_ratio": result["hard_positive_top100_exit_ratio"] <= SAFETY["hard_positive_top100_exit_ratio_max"],
        "top1_changed_ratio": result["top1_changed_ratio"] <= SAFETY["top1_changed_ratio_max"],
        "pearson": pearson >= SAFETY["pearson_min"],
        "spearman": result["spearman_mean_within_query"] >= SAFETY["spearman_mean_within_query_min"],
        "base_component_dominant": std_share["base"] >= SAFETY["base_component_std_share_min"] and std_share["base"] == max(std_share.values()),
        "boundary_component_not_dominant": std_share["boundary"] <= SAFETY["boundary_component_std_share_max"],
    }
    result["constraints"] = constraints
    result["feasible"] = bool(all(constraints.values()))
    if return_selected:
        return result, selected, scores
    return result


def load_seed_configs(path):
    if not path:
        raise ValueError("--configs_json is required for this mode")
    obj = json.load(open(path, encoding="utf-8"))
    if isinstance(obj, dict):
        obj = obj.get("configs", obj.get("results"))
    if not isinstance(obj, list):
        raise ValueError("configs JSON must contain a list")
    return obj


def generate_configs(args):
    if args.mode == "probe":
        return [
            {
                "config_id": f"probe_{family}", "phase": "fixed_probe", "family": family,
                "base_scale": 0.75, "a_joint": 1.625, "b_bd": 0.75, "d_fp": 0.5,
                "t_joint": 1.0, "t_bd": 1.0, "t_fp": 1.0,
            }
            for family in FAMILIES
        ]
    if args.mode == "full":
        configs = []
        for family, values in itertools.product(FAMILIES, itertools.product(
            GRID["base_scale"], GRID["a_joint"], GRID["b_bd"], GRID["d_fp"]
        )):
            configs.append({
                "config_id": f"grid_{len(configs):04d}", "phase": "full_grid", "family": family,
                "base_scale": values[0], "a_joint": values[1], "b_bd": values[2], "d_fp": values[3],
                "t_joint": 1.0, "t_bd": 1.0, "t_fp": 1.0,
            })
        return configs
    seeds = load_seed_configs(args.configs_json)
    seeds = [seed for seed in seeds if seed.get("model_id", args.model_id) == args.model_id]
    if not seeds:
        raise ValueError(f"No configs apply to model_id={args.model_id}")
    if args.mode == "custom":
        return seeds
    configs = []
    for seed_index, seed in enumerate(seeds):
        for tj, tfp, tbd in itertools.product(
            TEMPERATURES["t_joint"], TEMPERATURES["t_fp"], TEMPERATURES["t_bd"]
        ):
            config = {
                key: seed[key]
                for key in ["family", "base_scale", "a_joint", "b_bd", "d_fp"]
            }
            config.update({
                "config_id": f"cal_{seed_index:02d}_{len(configs):04d}",
                "source_config_id": seed.get("config_id", f"seed_{seed_index}"),
                "phase": "temperature_calibration", "t_joint": tj, "t_bd": tbd, "t_fp": tfp,
            })
            configs.append(config)
    return configs


def evaluate_many(configs, workers):
    if workers <= 1:
        return [evaluate_config(config) for config in tqdm(configs, desc="C3.1 configs")]
    with mp.get_context("fork").Pool(workers) as pool:
        return list(tqdm(
            pool.imap_unordered(evaluate_config, configs, chunksize=1),
            total=len(configs), desc="C3.1 configs",
        ))


def result_sort_key(result):
    return (
        result["selection_score"], result["min_primary_delta"],
        -result["hard_positive_top100_exit_ratio"], result["spearman_mean_within_query"],
        -result["top1_changed_ratio"],
    )


def write_csv(path, results):
    fields = [
        "model_id", "config_id", "source_config_id", "phase", "family",
        "base_scale", "a_joint", "b_bd", "d_fp", "t_joint", "t_bd", "t_fp",
        *METRIC_KEYS, "selection_score", "primary_mean_delta", "r100_mean_delta",
        "positive_delta_count", "top1_changed_ratio", "hard_positive_top100_exits",
        "hard_positive_top100_entries", "hard_positive_top100_exit_ratio",
        "y_fp_down_move_ratio", "y_joint_05_down_move_ratio",
        "wrong_video_boundary_bonus_mean", "pearson_base_vs_c3",
        "spearman_mean_within_query", "base_std_share", "q_joint_std_share",
        "boundary_std_share", "negative_e_fp_std_share", "feasible",
    ]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for result in results:
            row = {key: result.get(key) for key in fields}
            row.update(result["metrics"])
            for key, value in result["weighted_component_std_share"].items():
                row[f"{key}_std_share"] = value
            writer.writerow(row)


def main():
    global _DATA, _BASELINE, _ARGS
    args = parse_args()
    started = time.time()
    if "val_evidence" in str(Path(args.metadata_npz)) or "official_val" in str(Path(args.metadata_npz)):
        raise ValueError("C3.1 score search forbids official-val inputs")
    _ARGS = args
    _DATA = load_data(args)
    _BASELINE = build_baseline(_DATA, args)
    configs = generate_configs(args)
    if args.max_configs is not None:
        configs = configs[:args.max_configs]
    results = evaluate_many(configs, args.workers)
    results.sort(key=lambda item: item["config_id"])
    feasible = [result for result in results if result["feasible"]]
    best = max(feasible, key=result_sort_key) if feasible else None
    payload = {
        "status": "PASS" if best is not None else "NO_FEASIBLE_CONFIG",
        "stage": "C3.1 capacity and calibration strengthening",
        "scope": "train_calib_only",
        "official_val_used": False,
        "model_id": args.model_id,
        "mode": args.mode,
        "config_count": len(results),
        "feasible_count": len(feasible),
        "grid": GRID if args.mode == "full" else None,
        "temperatures": TEMPERATURES if args.mode == "calibrate" else None,
        "safety_constraints": SAFETY,
        "baseline_metrics": {key: round(value, 2) for key, value in _BASELINE["raw_metrics"].items()},
        "best": best,
        "results": results,
        "runtime_seconds": time.time() - started,
    }
    write_json(args.output_json, payload)
    write_csv(args.output_csv, results)
    print(json.dumps({key: payload[key] for key in [
        "status", "model_id", "mode", "config_count", "feasible_count", "runtime_seconds"
    ]} | {"best": best}, indent=2))


if __name__ == "__main__":
    main()
