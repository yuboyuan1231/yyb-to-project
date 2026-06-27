#!/usr/bin/env python

"""Freeze the completed train-calib-only C3.1 experiment."""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from rlem.io_utils import read_json, write_json
import rlem.grid_search_c31_capacity as search
from standalone_eval.eval import eval_retrieval, load_jsonl
from utils.inference_utils import filter_vcmr_by_nms


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata_npz", required=True)
    parser.add_argument("--descs_json", required=True)
    parser.add_argument("--models_root", required=True)
    parser.add_argument("--selected_calibration", required=True)
    parser.add_argument("--refinement_results")
    parser.add_argument("--search_results", nargs="+", required=True)
    parser.add_argument("--dataset_config", required=True)
    parser.add_argument("--gt_jsonl", required=True)
    parser.add_argument("--c3_strong_best", required=True)
    parser.add_argument("--c3_strong_grid", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--manifest_json", required=True)
    parser.add_argument("--effective_top_n", type=int, default=100)
    parser.add_argument("--max_after_nms", type=int, default=100)
    parser.add_argument("--nms_thd", type=float, default=0.7)
    return parser.parse_args()


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def key(item):
    return (
        item["selection_score"], item["min_primary_delta"],
        -item["hard_positive_top100_exit_ratio"], item["spearman_mean_within_query"],
        -item["top1_changed_ratio"],
    )


def load_video2idx(dataset_config):
    cfg = read_json(dataset_config)
    path = cfg["video_duration_idx_path"]
    if not os.path.isabs(path):
        path = os.path.join(cfg["root_path"], path)
    return {name: int(value[1]) for name, value in read_json(path)["train"].items()}


def build_submissions(best, selected, scores, data, descs, video2idx, args):
    fast_rows = []
    official_rows = []
    order = np.argsort(-scores, axis=1, kind="stable")
    for qi, indices in enumerate(selected):
        fast_predictions = [[
            int(data["video_idx"][qi, idx]), float(data["start"][qi, idx]),
            float(data["end"][qi, idx]), float(scores[qi, idx]),
        ] for idx in indices]
        candidates = [[
            int(data["video_idx"][qi, idx]), float(data["start"][qi, idx]),
            float(data["end"][qi, idx]), float(scores[qi, idx]),
        ] for idx in order[qi, :args.effective_top_n]]
        official_predictions = filter_vcmr_by_nms(
            candidates, nms_threshold=args.nms_thd,
            max_before_nms=args.effective_top_n, max_after_nms=args.max_after_nms,
        )
        row = {"desc_id": int(data["desc_ids"][qi]), "desc": descs[qi]}
        fast_rows.append(row | {"predictions": fast_predictions})
        official_rows.append(row | {"predictions": official_predictions})
    fast = {"video2idx": video2idx, "VCMR": fast_rows}
    official = {"video2idx": video2idx, "VCMR": official_rows}
    if fast["VCMR"] != official["VCMR"]:
        raise ValueError("Fast C3.1 NMS differs from frozen official NMS")
    return fast


def main():
    args = parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if any("official_val" in value or "val_evidence" in value for value in [
        args.metadata_npz, args.gt_jsonl, *args.search_results
    ]):
        raise ValueError("Official-val input is forbidden in C3.1 finalization")

    calibrated = read_json(args.selected_calibration)
    candidates = [calibrated["best"]]
    if args.refinement_results:
        refined = read_json(args.refinement_results)
        feasible = [item for item in refined["results"] if item["feasible"]]
        if feasible:
            candidates.append(max(feasible, key=key))
    best = max(candidates, key=key)
    model_id = best["model_id"]
    model_dir = Path(args.models_root) / model_id
    prediction_cache = model_dir / "predictions_train_calib.npz"
    checkpoint = model_dir / "model_best.pt"
    runtime_args = SimpleNamespace(
        metadata_npz=args.metadata_npz, prediction_cache=str(prediction_cache),
        rows_per_query=200, model_id=model_id, effective_top_n=args.effective_top_n,
        max_after_nms=args.max_after_nms, nms_thd=args.nms_thd,
    )
    search._ARGS = runtime_args
    search._DATA = search.load_data(runtime_args)
    search._BASELINE = search.build_baseline(search._DATA, runtime_args)
    best_result, selected, scores = search.evaluate_config(best, return_selected=True)
    if not best_result["feasible"]:
        raise RuntimeError("Final C3.1 candidate is not feasible")
    desc_obj = read_json(args.descs_json)
    if [int(value) for value in desc_obj["desc_ids"]] != [int(value) for value in search._DATA["desc_ids"]]:
        raise ValueError("Description metadata order mismatch")
    submission = build_submissions(
        best_result, selected, scores, search._DATA, desc_obj["descs"],
        load_video2idx(args.dataset_config), args,
    )
    gt = load_jsonl(args.gt_jsonl)
    official_metrics = eval_retrieval(
        submission, gt, iou_thds=(0.5, 0.7), verbose=False,
        match_number=True, use_desc_type=False,
    )["VCMR"]
    if official_metrics != best_result["metrics"]:
        raise ValueError(f"Fast/official metrics mismatch: {best_result['metrics']} vs {official_metrics}")

    c3_best = read_json(args.c3_strong_best)
    c3_grid = read_json(args.c3_strong_grid)
    c3_score = float(c3_best["diagnostics"]["selection_score"])
    gain = best_result["selection_score"] - c3_score
    breakthrough = gain >= 0.10
    retained = "C3.1 candidate" if breakthrough else "frozen C3-strong"
    train_args = read_json(model_dir / "train_args.json")
    config_payload = {
        "status": "C31_PASS", "scope": "train_fit_train_calib_only",
        "official_val_used": False, "official_val_metrics_used_for_selection": False,
        "post_val_adjustment": False, "scalar_plateau_broken": breakthrough,
        "retained_candidate": retained, "model_id": model_id,
        "model_architecture": {
            "hidden_dims": train_args["hidden_dims_parsed"], "dropout": train_args["dropout"],
            "parameter_count": train_args["parameter_count"],
        },
        "training": {key: train_args[key] for key in [
            "weight_decay", "lambda_joint_reg", "lambda_joint_bin", "lambda_bd",
            "lambda_fp", "fp_loss_type", "focal_gamma", "epochs", "batch_size", "lr", "seed"
        ]},
        "score_config": {key: best_result[key] for key in [
            "family", "base_scale", "a_joint", "b_bd", "d_fp", "t_joint", "t_bd", "t_fp"
        ]},
        "metrics": best_result["metrics"], "delta_vs_s_base": best_result["delta_vs_s_base"],
        "selection_score": best_result["selection_score"],
        "selection_gain_vs_frozen_c3_strong": gain,
        "breakthrough_threshold": 0.10,
        "diagnostics": {key: best_result[key] for key in [
            "top1_changed_ratio", "hard_positive_top100_exits", "hard_positive_top100_entries",
            "hard_positive_top100_exit_ratio", "y_fp_down_move_ratio",
            "y_joint_05_down_move_ratio", "wrong_video_boundary_bonus_mean",
            "mean_qbd_train_calib", "weighted_component_std", "weighted_component_std_share",
            "pearson_base_vs_c3", "spearman_mean_within_query", "constraints", "feasible"
        ]},
        "checkpoint": str(checkpoint), "checkpoint_sha256": sha256_file(checkpoint),
        "prediction_cache": str(prediction_cache), "prediction_cache_sha256": sha256_file(prediction_cache),
        "recommend_official_val_one_shot": breakthrough,
        "official_val_one_shot_executed": False, "next_stage_authorized": False,
    }
    write_json(output / "best_config.json", config_payload)
    write_json(output / "best_train_calib_submission.json", submission, pretty=False)
    metrics_payload = {
        "scope": "train_calib_only", "official_val_used": False,
        "s_base": c3_grid["baseline_metrics"],
        "c3_minimal": c3_grid["c3_minimal_default_metrics"],
        "c3_strong": c3_best["metrics"], "c31": best_result["metrics"],
        "c31_delta_vs_c3_strong": {
            key: round(best_result["metrics"][key] - c3_best["metrics"][key], 6)
            for key in search.METRIC_KEYS
        },
        "selection_score_c3_strong": c3_score,
        "selection_score_c31": best_result["selection_score"],
        "selection_gain_vs_c3_strong": gain,
    }
    write_json(output / "best_train_calib_metrics.json", metrics_payload)

    merged_results = []
    source_summaries = []
    for path in args.search_results:
        obj = read_json(path)
        merged_results.extend(obj.get("results", []))
        source_summaries.append({
            "path": path, "sha256": sha256_file(path), "model_id": obj.get("model_id"),
            "mode": obj.get("mode"), "config_count": obj.get("config_count"),
        })
    grid_payload = {
        "status": "PASS", "scope": "train_calib_only", "official_val_used": False,
        "source_searches": source_summaries, "total_evaluated_configs": len(merged_results),
        "results": merged_results,
    }
    write_json(output / "grid_results.json", grid_payload)
    search.write_csv(output / "grid_results.csv", merged_results)

    diagnostics = {
        "status": "C31_PASS", "reason": "scalar_capacity_improved" if breakthrough else "scalar_capacity_near_plateau",
        "scope": "train_fit_train_calib_only", "official_val_used": False,
        "official_val_metrics_used_for_selection": False, "post_val_adjustment": False,
        "model_count": len(list(Path(args.models_root).glob("*/train_summary.json"))),
        "score_config_count": len(merged_results), "best_config": config_payload,
        "comparison": metrics_payload, "frozen_artifacts_modified": False,
        "qsp_used": False, "r2_used": False, "reg_used": False, "amd_used": False,
        "recommend_official_val_one_shot": breakthrough,
        "next_stage_authorized": False,
    }
    write_json(output / "c31_diagnostics.json", diagnostics)
    rows = "\n".join(
        f"| {metric} | {metrics_payload['s_base'][metric]:.2f} | {metrics_payload['c3_minimal'][metric]:.2f} | {metrics_payload['c3_strong'][metric]:.2f} | {metrics_payload['c31'][metric]:.2f} | {metrics_payload['c31_delta_vs_c3_strong'][metric]:+.2f} |"
        for metric in search.METRIC_KEYS
    )
    md = f"""# C3.1 Capacity + Calibration Strengthening

Status: `C31_PASS`  
Scope: `train_fit/train_calib only`  
Official val used: `false`  
QSP/R2 used: `false/false`

| Metric | S_base | C3-minimal | C3-strong | C3.1 | C3.1 - C3-strong |
|---|---:|---:|---:|---:|---:|
{rows}

- Model: `{model_id}`; hidden={train_args['hidden_dims_parsed']}; parameters={train_args['parameter_count']}
- Family: `{best_result['family']}`
- Temperatures: joint={best_result['t_joint']}, boundary={best_result['t_bd']}, fp={best_result['t_fp']}
- Selection score: C3-strong={c3_score:.6f}, C3.1={best_result['selection_score']:.6f}, delta={gain:+.6f}
- Scalar plateau broken at +0.10 threshold: `{str(breakthrough).lower()}`
- Retained candidate: `{retained}`
- Recommend official-val one-shot: `{str(breakthrough).lower()}`
- Next stage authorized: `false`
"""
    (output / "c31_diagnostics.md").write_text(md, encoding="utf-8")

    immutable_inputs = {
        "train_metadata": args.metadata_npz,
        "c2_cache_manifest": "results/rlem_c2/cache/cache_manifest.json",
        "c2_model": "results/rlem_c2/basic_heads/model_best.pt",
        "c3_strong_best": args.c3_strong_best,
    }
    manifest = {
        "status": "C31_FROZEN", "scope": "train_fit_train_calib_only",
        "official_val_used": False, "official_val_one_shot_executed": False,
        "post_val_adjustment": False, "qsp_used": False, "r2_used": False,
        "immutable_inputs": {
            name: {"path": path, "sha256": sha256_file(path)}
            for name, path in immutable_inputs.items()
        },
        "selected_checkpoint_sha256": sha256_file(checkpoint),
        "outputs": {}, "next_stage_authorized": False,
    }
    for name in [
        "grid_results.json", "grid_results.csv", "best_config.json",
        "best_train_calib_submission.json", "best_train_calib_metrics.json",
        "c31_diagnostics.json", "c31_diagnostics.md",
    ]:
        manifest["outputs"][name] = sha256_file(output / name)
    write_json(args.manifest_json, manifest)
    print(json.dumps({
        "status": "C31_PASS", "model_id": model_id, "selection_gain_vs_c3_strong": gain,
        "scalar_plateau_broken": breakthrough, "official_val_used": False,
    }, indent=2))


if __name__ == "__main__":
    main()
