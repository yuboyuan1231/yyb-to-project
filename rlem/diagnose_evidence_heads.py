#!/usr/bin/env python

"""Diagnose C2 evidence heads on a query-filtered train_calib partition."""

import argparse
import json
import math
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from rlem.evidence_dataset import FeatureStats, StreamingEvidenceDataset, evidence_collate
from rlem.evidence_model import EvidenceMLP
from rlem.io_utils import read_json, write_json
from rlem.train_evidence_heads import iter_cached_batches, load_cache


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence_jsonl", required=True)
    parser.add_argument("--desc_ids", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--cache_dir", default=None)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_md", required=True)
    parser.add_argument("--batch_size", type=int, default=8192)
    parser.add_argument("--max_rows", type=int, default=None)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    return parser.parse_args()


def load_ids(path):
    with open(path, "r", encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip()}


def safe_corr(func, x, y):
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return None
    value = float(func(x, y).statistic)
    return value if math.isfinite(value) else None


def safe_auc(y, score):
    if len(np.unique(y)) < 2:
        return None
    value = float(roc_auc_score(y, score))
    return value if math.isfinite(value) else None


def distribution(x):
    percentiles = [0, 1, 5, 25, 50, 75, 95, 99, 100]
    values = np.percentile(x, percentiles)
    return {
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
        "percentiles": {str(p): float(v) for p, v in zip(percentiles, values)},
    }


def main():
    args = parse_args()
    allowed_ids = load_ids(args.desc_ids)
    ckpt = torch.load(args.ckpt, map_location="cpu")
    stats = FeatureStats.from_dict(ckpt["feature_stats"])
    cfg = ckpt["model_cfg"]
    model = EvidenceMLP(
        input_dim=cfg["input_dim"],
        hidden_dim=cfg.get("hidden_dim", 128),
        dropout=cfg.get("dropout", 0.0),
    )
    model.load_state_dict(ckpt["model_state"])
    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    if args.cache_dir:
        cache = load_cache(args.cache_dir, stats.feature_names)
        loader = iter_cached_batches(
            cache["train_calib"]["x"], cache["train_calib"]["y"],
            args.batch_size, train=False, seed=999,
            max_rows=args.max_rows,
        )
    else:
        dataset = StreamingEvidenceDataset(
            args.evidence_jsonl,
            stats=stats,
            max_rows=args.max_rows,
            shuffle_buffer=0,
            allowed_desc_ids=allowed_ids,
        )
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            collate_fn=evidence_collate,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
        )
    collected = {key: [] for key in [
        "q_joint", "q_bd", "e_fp", "y_joint", "y_joint_05", "y_joint_07", "y_bd", "m_bd", "y_fp"
    ]}
    with torch.no_grad():
        for batch in tqdm(loader, desc="diagnose train_calib"):
            x = batch["x"].to(device)
            pred = model.predict_scores(x)
            for key in ["q_joint", "q_bd", "e_fp"]:
                collected[key].append(pred[key].cpu().numpy().astype(np.float32))
            for key in ["y_joint", "y_joint_05", "y_joint_07", "y_bd", "m_bd", "y_fp"]:
                collected[key].append(batch[key].numpy().astype(np.float32))
    arrays = {key: np.concatenate(parts) for key, parts in collected.items()}
    n = len(arrays["q_joint"])
    if any(len(value) != n for value in arrays.values()):
        raise ValueError("Diagnostic arrays have inconsistent lengths")
    bd_mask = arrays["m_bd"] > 0.5
    head_stats = {key: distribution(arrays[key]) for key in ["q_joint", "q_bd", "e_fp"]}
    collapse = {
        key: bool(value["std"] < 1e-4 or value["percentiles"]["99"] - value["percentiles"]["1"] < 1e-3)
        for key, value in head_stats.items()
    }
    result = {
        "status": "PASS" if not any(collapse.values()) else "FAIL",
        "scope": "train_calib_only",
        "official_val_used": False,
        "checkpoint": os.path.abspath(args.ckpt),
        "checkpoint_epoch": int(ckpt.get("epoch", -1)),
        "rows": n,
        "desc_id_count": len(allowed_ids),
        "feature_count": len(stats.feature_names),
        "feature_names": stats.feature_names,
        "r2_features_enabled": [name for name in stats.feature_names if name.startswith("r2_") or name in {"rank_r2", "r_abs_gap"}],
        "metrics": {
            "q_joint_y_joint_pearson": safe_corr(pearsonr, arrays["q_joint"], arrays["y_joint"]),
            "q_joint_y_joint_spearman": safe_corr(spearmanr, arrays["q_joint"], arrays["y_joint"]),
            "q_joint_y_joint_05_auc": safe_auc(arrays["y_joint_05"], arrays["q_joint"]),
            "q_bd_y_bd_pearson_m_bd": safe_corr(pearsonr, arrays["q_bd"][bd_mask], arrays["y_bd"][bd_mask]),
            "q_bd_y_bd_spearman_m_bd": safe_corr(spearmanr, arrays["q_bd"][bd_mask], arrays["y_bd"][bd_mask]),
            "e_fp_y_fp_auc": safe_auc(arrays["y_fp"], arrays["e_fp"]),
        },
        "head_distributions": head_stats,
        "collapse": collapse,
        "label_counts": {
            "y_joint_05_positive": int(np.sum(arrays["y_joint_05"] > 0.5)),
            "y_joint_07_positive": int(np.sum(arrays["y_joint_07"] > 0.5)),
            "y_fp_positive": int(np.sum(arrays["y_fp"] > 0.5)),
            "m_bd_positive": int(np.sum(bd_mask)),
        },
        "label_ratios": {
            "y_joint_05_positive": float(np.mean(arrays["y_joint_05"] > 0.5)),
            "y_joint_07_positive": float(np.mean(arrays["y_joint_07"] > 0.5)),
            "y_fp_positive": float(np.mean(arrays["y_fp"] > 0.5)),
            "m_bd_positive": float(np.mean(bd_mask)),
        },
    }
    write_json(args.output_json, result)
    lines = [
        "# C2 train_calib diagnostics",
        "",
        f"Status: `{result['status']}`",
        "",
        f"Rows: {n:,}; query IDs: {len(allowed_ids):,}; checkpoint epoch: {result['checkpoint_epoch']}",
        "",
        "## Metrics",
        "",
    ]
    for key, value in result["metrics"].items():
        lines.append(f"- `{key}`: {value}")
    lines += ["", "## Head distributions", ""]
    for key, value in head_stats.items():
        lines.append(f"- `{key}`: mean={value['mean']:.6f}, std={value['std']:.6f}, min={value['min']:.6f}, max={value['max']:.6f}, collapse={collapse[key]}")
    lines += ["", "Official val was not read or scored.", ""]
    Path(args.output_md).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_md).write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(result, indent=2))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
