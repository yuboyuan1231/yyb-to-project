#!/usr/bin/env python

"""Select a C3.1 stage winner and freeze the next staged model list."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from rlem.io_utils import write_json


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_root", required=True)
    parser.add_argument("--configs_json", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--next_stage", choices=["none", "loss", "fp"], default="none")
    parser.add_argument("--next_configs_json")
    parser.add_argument("--simplicity_tolerance", type=float, default=0.10)
    return parser.parse_args()


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_next_configs(stage, winner_config):
    common = {
        "hidden_dims": winner_config["hidden_dims"],
        "dropout": winner_config["dropout"],
        "weight_decay": winner_config["weight_decay"],
        "lambda_joint_reg": 1.0,
    }
    configs = []
    if stage == "loss":
        for jbin in [0.5, 1.0]:
            for boundary in [0.5, 1.0]:
                for fp in [0.5, 1.0, 1.5]:
                    configs.append({
                        "config_id": f"loss_jb{jbin:g}_bd{boundary:g}_fp{fp:g}".replace(".", "p"),
                        **common, "lambda_joint_bin": jbin, "lambda_bd": boundary,
                        "lambda_fp": fp, "fp_loss_type": "bce", "focal_gamma": 2.0,
                    })
    elif stage == "fp":
        common.update({
            "lambda_joint_bin": winner_config["lambda_joint_bin"],
            "lambda_bd": winner_config["lambda_bd"],
            "lambda_fp": winner_config["lambda_fp"],
        })
        configs = [
            {"config_id": "fp_bce", **common, "fp_loss_type": "bce", "focal_gamma": 2.0},
            {"config_id": "fp_pos_weight_auto", **common, "fp_loss_type": "bce_pos_weight", "focal_gamma": 2.0},
            {"config_id": "fp_focal_gamma1", **common, "fp_loss_type": "focal", "focal_gamma": 1.0},
            {"config_id": "fp_focal_gamma2", **common, "fp_loss_type": "focal", "focal_gamma": 2.0},
        ]
    return configs


def main():
    args = parse_args()
    configs = json.load(open(args.configs_json, encoding="utf-8"))
    by_id = {item["config_id"]: item for item in configs}
    records = []
    for config_id, config in by_id.items():
        directory = Path(args.model_root) / config_id
        required = [
            directory / "train_summary.json", directory / "diagnostics_train_calib.json",
            directory / "probe_results.json", directory / "model_best.pt",
        ]
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise FileNotFoundError(f"Incomplete model {config_id}: {missing}")
        train = json.load(open(required[0], encoding="utf-8"))
        diagnostics = json.load(open(required[1], encoding="utf-8"))
        probe = json.load(open(required[2], encoding="utf-8"))
        if train["status"] != "PASS" or diagnostics["status"] != "PASS" or probe["status"] != "PASS":
            raise RuntimeError(f"Failed C3.1 model: {config_id}")
        best = probe["best"]
        records.append({
            "config_id": config_id,
            "config": config,
            "parameter_count": train["parameter_count"],
            "best_epoch": train["best_epoch"],
            "best_val_loss": train["best_val_loss"],
            "probe_best": best,
            "probe_selection_score": best["selection_score"],
            "probe_family": best["family"],
            "checkpoint_sha256": sha256_file(required[3]),
        })
    feasible = [item for item in records if item["probe_best"]["feasible"]]
    if not feasible:
        raise RuntimeError("No feasible model in C3.1 stage")
    top_score = max(item["probe_selection_score"] for item in feasible)
    within = [
        item for item in feasible
        if item["probe_selection_score"] >= top_score - args.simplicity_tolerance
    ]
    winner = min(
        within,
        key=lambda item: (
            item["parameter_count"], -item["probe_selection_score"],
            item["best_val_loss"], item["config_id"],
        ),
    )
    payload = {
        "status": "PASS", "scope": "train_calib_only", "official_val_used": False,
        "model_root": args.model_root, "configs_json": args.configs_json,
        "model_count": len(records), "simplicity_tolerance": args.simplicity_tolerance,
        "top_raw_selection_score": top_score,
        "models_within_simplicity_tolerance": [item["config_id"] for item in within],
        "selection_rule": "lowest parameter count within 0.10 of best probe; then score, loss, id",
        "winner": winner, "records": records,
    }
    write_json(args.output_json, payload)
    if args.next_stage != "none":
        if not args.next_configs_json:
            raise ValueError("--next_configs_json is required when --next_stage is not none")
        next_configs = make_next_configs(args.next_stage, winner["config"])
        write_json(args.next_configs_json, next_configs)
        print(json.dumps({"winner": winner["config_id"], "next_stage": args.next_stage, "next_count": len(next_configs)}, indent=2))
    else:
        print(json.dumps({"winner": winner["config_id"]}, indent=2))


if __name__ == "__main__":
    main()
