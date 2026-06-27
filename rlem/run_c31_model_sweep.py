#!/usr/bin/env python

"""Sequentially train and diagnose a frozen list of C3.1 model configs."""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs_json", required=True)
    parser.add_argument("--cache_dir", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--summary_json", default=None)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--max_train_rows", type=int, default=None)
    parser.add_argument("--max_val_rows", type=int, default=None)
    parser.add_argument("--skip_existing", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    configs = json.load(open(args.configs_json, encoding="utf-8"))
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    results = []
    for index, config in enumerate(configs):
        config_id = config["config_id"]
        output_dir = output_root / config_id
        summary_path = output_dir / "train_summary.json"
        if args.skip_existing and summary_path.exists():
            summary = json.load(open(summary_path, encoding="utf-8"))
            if summary.get("status") == "PASS" and (output_dir / "diagnostics_train_calib.json").exists():
                results.append({"config_id": config_id, "status": "REUSED", "summary": summary})
                continue
        output_dir.mkdir(parents=True, exist_ok=True)
        train_cmd = [
            sys.executable, "rlem/train_c31_capacity.py",
            "--config_id", config_id, "--cache_dir", args.cache_dir,
            "--output_dir", str(output_dir), "--hidden_dims", ",".join(map(str, config["hidden_dims"])),
            "--dropout", str(config["dropout"]), "--weight_decay", str(config["weight_decay"]),
            "--lambda_joint_reg", str(config.get("lambda_joint_reg", 1.0)),
            "--lambda_joint_bin", str(config.get("lambda_joint_bin", 0.25)),
            "--lambda_bd", str(config.get("lambda_bd", 1.0)),
            "--lambda_fp", str(config.get("lambda_fp", 0.5)),
            "--fp_loss_type", config.get("fp_loss_type", "bce"),
            "--focal_gamma", str(config.get("focal_gamma", 2.0)),
            "--epochs", str(args.epochs), "--batch_size", str(args.batch_size),
            "--lr", str(args.lr), "--device", args.device,
        ]
        if args.max_train_rows is not None:
            train_cmd.extend(["--max_train_rows", str(args.max_train_rows)])
        if args.max_val_rows is not None:
            train_cmd.extend(["--max_val_rows", str(args.max_val_rows)])
        started = time.time()
        with open(output_dir / "train.log", "w", encoding="utf-8") as log:
            subprocess.run(train_cmd, stdout=log, stderr=subprocess.STDOUT, check=True)
        diagnose_cmd = [
            sys.executable, "rlem/diagnose_c31_capacity.py",
            "--ckpt", str(output_dir / "model_best.pt"), "--cache_dir", args.cache_dir,
            "--output_json", str(output_dir / "diagnostics_train_calib.json"),
            "--output_md", str(output_dir / "diagnostics_train_calib.md"),
            "--prediction_cache", str(output_dir / "predictions_train_calib.npz"),
            "--batch_size", "8192", "--device", args.device,
        ]
        with open(output_dir / "diagnose.log", "w", encoding="utf-8") as log:
            subprocess.run(diagnose_cmd, stdout=log, stderr=subprocess.STDOUT, check=True)
        summary = json.load(open(summary_path, encoding="utf-8"))
        diagnostics = json.load(open(output_dir / "diagnostics_train_calib.json", encoding="utf-8"))
        if summary["status"] != "PASS" or diagnostics["status"] != "PASS":
            raise RuntimeError(f"C3.1 config failed: {config_id}")
        results.append({
            "config_id": config_id, "status": "PASS",
            "runtime_seconds": time.time() - started,
            "summary": summary, "diagnostic_metrics": diagnostics["metrics"],
        })
        print(f"[{index + 1}/{len(configs)}] PASS {config_id}", flush=True)
    manifest = {
        "status": "PASS", "official_val_used": False,
        "configs_json": args.configs_json, "config_count": len(configs), "results": results,
    }
    summary_path = Path(args.summary_json) if args.summary_json else output_root / "sweep_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
