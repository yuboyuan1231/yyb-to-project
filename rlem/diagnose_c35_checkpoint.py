#!/usr/bin/env python3

"""Train-calib-only head diagnostics for a C3.5 subset checkpoint."""

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
from tqdm import tqdm

from rlem.evidence_model import FlexibleEvidenceMLP
from rlem.io_utils import read_json, write_json
from rlem.train_evidence_heads import CACHE_LABEL_NAMES


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--cache_dir", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_md", required=True)
    parser.add_argument("--prediction_cache", required=True)
    parser.add_argument("--batch_size", type=int, default=8192)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def distribution(values):
    percentiles = [0, 5, 50, 95, 100]
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "percentiles": {
            str(p): float(value)
            for p, value in zip(percentiles, np.percentile(values, percentiles))
        },
    }


def safe_corr(fn, x, y):
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return None
    return float(fn(x, y).statistic)


def sigmoid(logits):
    values = np.clip(logits.astype(np.float64), -40.0, 40.0)
    return (1.0 / (1.0 + np.exp(-values))).astype(np.float32)


def main():
    args = parse_args()
    cache_dir = Path(args.cache_dir).resolve()
    manifest = read_json(cache_dir / "cache_manifest.json")
    if manifest.get("status") != "PASS" or manifest.get("official_val_used") is not False:
        raise ValueError("diagnostics require the PASS train-only canonical cache")
    source_name = Path(manifest.get("source_evidence", "")).name.lower()
    if "train" not in source_name or source_name.startswith("val"):
        raise ValueError("official-val cache is forbidden")

    checkpoint = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    if checkpoint.get("model_type") != "FlexibleEvidenceMLP":
        raise ValueError("C3.5 diagnostics require FlexibleEvidenceMLP")
    if checkpoint.get("official_val_used") is not False:
        raise ValueError("checkpoint is not explicitly train-only")
    cfg = checkpoint["model_cfg"]
    feature_names = checkpoint["feature_stats"]["feature_names"]
    canonical = manifest["feature_names"]
    if any(name not in canonical for name in feature_names):
        raise ValueError("checkpoint feature absent from canonical cache")
    columns = [canonical.index(name) for name in feature_names]
    if cfg["input_dim"] != len(columns):
        raise ValueError("checkpoint input dimension mismatch")

    model = FlexibleEvidenceMLP(cfg["input_dim"], cfg["hidden_dims"], cfg["dropout"])
    model.load_state_dict(checkpoint["model_state"])
    device = torch.device(args.device)
    model.to(device).eval()
    entry = manifest["splits"]["train_calib"]
    x = np.load(entry["x_path"], mmap_mode="r")
    y = np.load(entry["y_path"], mmap_mode="r")
    outputs = {key: [] for key in ("q_joint_logit", "q_bd_logit", "e_fp_logit")}
    labels = {key: [] for key in CACHE_LABEL_NAMES}
    with torch.no_grad():
        for start in tqdm(
            range(0, len(x), args.batch_size),
            total=math.ceil(len(x) / args.batch_size),
            desc=f"diagnose {len(columns)}d",
        ):
            end = min(start + args.batch_size, len(x))
            xb = np.array(x[start:end, columns], dtype=np.float32, copy=True)
            yb = np.array(y[start:end], dtype=np.float32, copy=True)
            logits = model(torch.from_numpy(xb).to(device))
            for key in outputs:
                outputs[key].append(logits[key].cpu().numpy().astype(np.float32))
            for column, key in enumerate(CACHE_LABEL_NAMES):
                labels[key].append(yb[:, column])
    outputs = {key: np.concatenate(value) for key, value in outputs.items()}
    labels = {key: np.concatenate(value) for key, value in labels.items()}
    scores = {
        "q_joint": sigmoid(outputs["q_joint_logit"]),
        "q_bd": sigmoid(outputs["q_bd_logit"]),
        "e_fp": sigmoid(outputs["e_fp_logit"]),
    }
    mask = labels["m_bd"] > 0.5
    hard = labels["y_joint_05"] > 0.5
    wrong = ~mask
    metrics = {
        "q_joint_y_joint_pearson": safe_corr(pearsonr, scores["q_joint"], labels["y_joint"]),
        "q_joint_y_joint_spearman": safe_corr(spearmanr, scores["q_joint"], labels["y_joint"]),
        "q_joint_y_joint_05_auc": float(roc_auc_score(labels["y_joint_05"], scores["q_joint"])),
        "q_bd_y_bd_pearson_m_bd": safe_corr(pearsonr, scores["q_bd"][mask], labels["y_bd"][mask]),
        "q_bd_y_bd_spearman_m_bd": safe_corr(spearmanr, scores["q_bd"][mask], labels["y_bd"][mask]),
        "e_fp_y_fp_auc": float(roc_auc_score(labels["y_fp"], scores["e_fp"])),
    }
    diagnostics = {
        "status": "PASS",
        "stage": "C3.5 matched extended-epoch head diagnostics",
        "scope": "train_calib_only",
        "official_val_used": False,
        "checkpoint": str(Path(args.ckpt).resolve()),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "rows": len(x),
        "feature_count": len(feature_names),
        "feature_names": feature_names,
        "r2_features_enabled": [
            name for name in feature_names
            if name.startswith("r2_") or name in {"rank_r2", "r_abs_gap"}
        ],
        "metrics": metrics,
        "head_distributions": {key: distribution(value) for key, value in scores.items()},
        "collapse": {key: bool(np.std(value) < 1e-5) for key, value in scores.items()},
        "wrong_video_q_bd": distribution(scores["q_bd"][wrong]),
        "hard_positive": {
            "rows": int(np.sum(hard)),
            **{key: distribution(value[hard]) for key, value in scores.items()},
        },
    }
    if diagnostics["r2_features_enabled"] or any(diagnostics["collapse"].values()):
        diagnostics["status"] = "FAIL"
    write_json(args.output_json, diagnostics)
    Path(args.output_md).parent.mkdir(parents=True, exist_ok=True)
    metric_rows = "\n".join(f"| {key} | {value:.8f} |" for key, value in metrics.items())
    Path(args.output_md).write_text(
        f"""# C3.5 extended-epoch head diagnostics

Status: `{diagnostics['status']}`  
Scope: `train_calib_only`  
Official val used: `false`

| Metric | Value |
|---|---:|
{metric_rows}

- Checkpoint epoch: {diagnostics['checkpoint_epoch']}
- Features: {len(feature_names)}
- Collapsed heads: {[key for key, value in diagnostics['collapse'].items() if value]}
- Wrong-video Q_bd mean/std: {diagnostics['wrong_video_q_bd']['mean']:.8f}/{diagnostics['wrong_video_q_bd']['std']:.8f}
""",
        encoding="utf-8",
    )
    output = Path(args.prediction_cache)
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = Path(str(output) + ".partial")
    with open(partial, "wb") as file:
        np.savez(file, **outputs)
    os.replace(partial, output)
    print(json.dumps(diagnostics, indent=2))


if __name__ == "__main__":
    main()
