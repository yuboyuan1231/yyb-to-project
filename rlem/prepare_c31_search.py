#!/usr/bin/env python

"""Deterministic stage transitions for the bounded C3.1 search."""

import argparse
import glob
import hashlib
import json
import math
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from rlem.io_utils import write_json


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["finalists", "calibration_seeds", "select_calibration", "refine"], required=True)
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--top_models", type=int, default=3)
    parser.add_argument("--top_per_family", type=int, default=3)
    parser.add_argument("--temperature_gain_min", type=float, default=0.05)
    return parser.parse_args()


def result_key(item):
    return (
        item["selection_score"], item["min_primary_delta"],
        -item["hard_positive_top100_exit_ratio"], item["spearman_mean_within_query"],
        -item["top1_changed_ratio"],
    )


def load(path):
    return json.load(open(path, encoding="utf-8"))


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def finalists(args):
    records = []
    for pattern in args.inputs:
        for path in sorted(glob.glob(pattern)):
            obj = load(path)
            if obj.get("mode") != "probe" or not obj.get("best"):
                continue
            best = obj["best"]
            if not best["feasible"]:
                continue
            model_dir = Path(path).parent
            train = load(model_dir / "train_summary.json")
            prediction_cache = model_dir / "predictions_train_calib.npz"
            records.append({
                "model_id": obj["model_id"], "model_dir": str(model_dir),
                "prediction_cache": str(prediction_cache),
                "prediction_cache_sha256": sha256_file(prediction_cache),
                "checkpoint": str(model_dir / "model_best.pt"),
                "parameter_count": train["parameter_count"], "probe_best": best,
            })
    if len(records) < args.top_models:
        raise RuntimeError(f"Only {len(records)} feasible probed models")
    records.sort(key=lambda item: result_key(item["probe_best"]), reverse=True)
    selected = []
    duplicate_outputs = []
    seen_predictions = {}
    for item in records:
        digest = item["prediction_cache_sha256"]
        if digest in seen_predictions:
            duplicate_outputs.append({
                "model_id": item["model_id"],
                "identical_to": seen_predictions[digest],
                "prediction_cache_sha256": digest,
            })
            continue
        seen_predictions[digest] = item["model_id"]
        selected.append(item)
        if len(selected) == args.top_models:
            break
    if len(selected) < args.top_models:
        raise RuntimeError(f"Only {len(selected)} distinct feasible prediction caches")
    return {
        "status": "PASS", "scope": "train_calib_only", "official_val_used": False,
        "selection": "top distinct models by frozen three-family probe",
        "candidate_model_count": len(records), "selected_count": len(selected),
        "deduplication": "exact prediction-cache SHA256",
        "duplicate_outputs_skipped_before_top_k": duplicate_outputs,
        "finalists": selected, "all_model_ranking": records,
    }


def calibration_seeds(args):
    seeds = []
    source_files = []
    for path in args.inputs:
        obj = load(path)
        if obj.get("mode") != "full":
            raise ValueError(f"Not a full-grid result: {path}")
        source_files.append(path)
        for family in ["A_additive", "B_gated_boundary", "C_centered_gated_boundary"]:
            candidates = [
                item for item in obj["results"] if item["family"] == family and item["feasible"]
            ]
            candidates.sort(key=result_key, reverse=True)
            for item in candidates[:args.top_per_family]:
                seeds.append({
                    key: item[key] for key in [
                        "config_id", "family", "base_scale", "a_joint", "b_bd", "d_fp"
                    ]
                } | {"model_id": obj["model_id"], "unit_temperature_selection_score": item["selection_score"]})
    return {
        "status": "PASS", "scope": "train_calib_only", "official_val_used": False,
        "source_full_grids": source_files, "top_per_family": args.top_per_family,
        "seed_count": len(seeds), "configs": seeds,
    }


def select_calibration(args):
    accepted = []
    rejected_nonunit = []
    all_results = []
    for path in args.inputs:
        obj = load(path)
        if obj.get("mode") != "calibrate":
            raise ValueError(f"Not calibration results: {path}")
        results = [item for item in obj["results"] if item["feasible"]]
        all_results.extend(results)
        grouped = {}
        for item in results:
            grouped.setdefault(item["source_config_id"], []).append(item)
        for source, group in grouped.items():
            unit = next((item for item in group if all(
                math.isclose(float(item[key]), 1.0) for key in ["t_joint", "t_bd", "t_fp"]
            )), None)
            if unit is None:
                raise RuntimeError(f"Missing T=1 candidate for {source}")
            best = max(group, key=result_key)
            if best["selection_score"] >= unit["selection_score"] + args.temperature_gain_min:
                accepted.append(best)
            else:
                accepted.append(unit)
                if best["config_id"] != unit["config_id"]:
                    rejected_nonunit.append({
                        "source_config_id": source, "unit": unit, "nonunit_best": best,
                        "gain": best["selection_score"] - unit["selection_score"],
                    })
    if not accepted:
        raise RuntimeError("No accepted calibrated configuration")
    family_best = {}
    for family in ["A_additive", "B_gated_boundary", "C_centered_gated_boundary"]:
        values = [item for item in accepted if item["family"] == family]
        if values:
            family_best[family] = max(values, key=result_key)
    raw_best = max(accepted, key=result_key)
    safe = [family_best[key] for key in ["B_gated_boundary", "C_centered_gated_boundary"] if key in family_best]
    safe_best = max(safe, key=result_key) if safe else None
    best = safe_best if safe_best and safe_best["selection_score"] >= raw_best["selection_score"] - 0.10 else raw_best
    return {
        "status": "PASS", "scope": "train_calib_only", "official_val_used": False,
        "temperature_gain_min": args.temperature_gain_min,
        "family_safety_preference_tolerance": 0.10,
        "accepted_source_count": len(accepted), "accepted": accepted,
        "rejected_nonunit": rejected_nonunit, "family_best": family_best,
        "raw_best": raw_best, "best": best,
    }


def refine(args):
    if len(args.inputs) != 1:
        raise ValueError("refine mode requires exactly one selected-calibration JSON")
    selected = load(args.inputs[0])["best"]
    sets = {}
    for key in ["base_scale", "a_joint", "b_bd", "d_fp"]:
        center = float(selected[key])
        values = {round(max(0.0, center - 0.125), 6), round(center, 6), round(center + 0.125, 6)}
        if key == "base_scale":
            values = {value for value in values if value > 0}
        sets[key] = sorted(values)
    configs = []
    for values in __import__("itertools").product(
        sets["base_scale"], sets["a_joint"], sets["b_bd"], sets["d_fp"]
    ):
        if all(math.isclose(values[i], float(selected[key])) for i, key in enumerate(
            ["base_scale", "a_joint", "b_bd", "d_fp"]
        )):
            continue
        configs.append({
            "config_id": f"refine_{len(configs):03d}", "phase": "local_refinement_once",
            "family": selected["family"], "base_scale": values[0], "a_joint": values[1],
            "b_bd": values[2], "d_fp": values[3],
            "t_joint": selected["t_joint"], "t_bd": selected["t_bd"], "t_fp": selected["t_fp"],
        })
    return {
        "status": "PASS", "scope": "train_calib_only", "official_val_used": False,
        "source_best": selected, "value_sets": sets, "config_count": len(configs), "configs": configs,
    }


def main():
    args = parse_args()
    function = {
        "finalists": finalists, "calibration_seeds": calibration_seeds,
        "select_calibration": select_calibration, "refine": refine,
    }[args.mode]
    payload = function(args)
    write_json(args.output_json, payload)
    print(json.dumps({key: payload.get(key) for key in [
        "status", "candidate_model_count", "selected_count", "seed_count",
        "accepted_source_count", "config_count"
    ]}, indent=2))


if __name__ == "__main__":
    main()
