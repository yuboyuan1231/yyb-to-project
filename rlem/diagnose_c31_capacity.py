#!/usr/bin/env python

"""Train-calib diagnostics and reusable logits for one C3.1 checkpoint."""

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from rlem.evidence_model import FlexibleEvidenceMLP
from rlem.io_utils import write_json
from rlem.train_evidence_heads import CACHE_LABEL_NAMES, iter_cached_batches, load_cache


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--cache_dir", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_md", required=True)
    parser.add_argument("--prediction_cache", required=True)
    parser.add_argument("--batch_size", type=int, default=8192)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    return parser.parse_args()


def distribution(values):
    ps = [0, 5, 50, 95, 100]
    return {
        "mean": float(np.mean(values)), "std": float(np.std(values)),
        "min": float(np.min(values)), "max": float(np.max(values)),
        "percentiles": {str(p): float(v) for p, v in zip(ps, np.percentile(values, ps))},
    }


def safe_corr(fn, x, y):
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return None
    return float(fn(x, y).statistic)


def main():
    args = parse_args()
    checkpoint = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    if checkpoint.get("model_type") != "FlexibleEvidenceMLP":
        raise ValueError("C3.1 diagnostics require FlexibleEvidenceMLP checkpoint")
    cfg = checkpoint["model_cfg"]
    model = FlexibleEvidenceMLP(cfg["input_dim"], cfg["hidden_dims"], cfg["dropout"])
    model.load_state_dict(checkpoint["model_state"])
    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    feature_names = checkpoint["feature_stats"]["feature_names"]
    cache = load_cache(args.cache_dir, feature_names)
    x, y = cache["train_calib"]["x"], cache["train_calib"]["y"]
    outputs = {key: [] for key in ["q_joint_logit", "q_bd_logit", "e_fp_logit"]}
    label_chunks = {key: [] for key in CACHE_LABEL_NAMES}
    with torch.no_grad():
        batches = iter_cached_batches(x, y, args.batch_size, train=False, seed=1013)
        for batch in tqdm(batches, total=math.ceil(len(x) / args.batch_size), desc="C3.1 diagnose"):
            logits = model(batch["x"].to(device))
            for key in outputs:
                outputs[key].append(logits[key].cpu().numpy().astype(np.float32))
            for key in label_chunks:
                label_chunks[key].append(batch[key].numpy().astype(np.float32))
    outputs = {key: np.concatenate(value) for key, value in outputs.items()}
    labels = {key: np.concatenate(value) for key, value in label_chunks.items()}
    scores = {
        "q_joint": 1.0 / (1.0 + np.exp(-outputs["q_joint_logit"])),
        "q_bd": 1.0 / (1.0 + np.exp(-outputs["q_bd_logit"])),
        "e_fp": 1.0 / (1.0 + np.exp(-outputs["e_fp_logit"])),
    }
    mask = labels["m_bd"] > 0.5
    hard = labels["y_joint_05"] > 0.5
    wrong = ~mask
    diagnostics = {
        "status": "PASS",
        "stage": "C3.1 capacity",
        "scope": "train_calib_only",
        "official_val_used": False,
        "checkpoint": str(Path(args.ckpt).resolve()),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "rows": len(x),
        "feature_count": len(feature_names),
        "r2_features_enabled": [
            name for name in feature_names
            if name.startswith("r2_") or name in {"rank_r2", "r_abs_gap"}
        ],
        "metrics": {
            "q_joint_y_joint_pearson": safe_corr(pearsonr, scores["q_joint"], labels["y_joint"]),
            "q_joint_y_joint_spearman": safe_corr(spearmanr, scores["q_joint"], labels["y_joint"]),
            "q_joint_y_joint_05_auc": float(roc_auc_score(labels["y_joint_05"], scores["q_joint"])),
            "q_bd_y_bd_pearson_m_bd": safe_corr(pearsonr, scores["q_bd"][mask], labels["y_bd"][mask]),
            "q_bd_y_bd_spearman_m_bd": safe_corr(spearmanr, scores["q_bd"][mask], labels["y_bd"][mask]),
            "e_fp_y_fp_auc": float(roc_auc_score(labels["y_fp"], scores["e_fp"])),
        },
        "head_distributions": {key: distribution(value) for key, value in scores.items()},
        "collapse": {key: bool(np.std(value) < 1e-5) for key, value in scores.items()},
        "label_ratios": {
            "y_joint_05_positive": float(np.mean(labels["y_joint_05"] > 0.5)),
            "y_joint_07_positive": float(np.mean(labels["y_joint_07"] > 0.5)),
            "y_fp_positive": float(np.mean(labels["y_fp"] > 0.5)),
            "m_bd_positive": float(np.mean(mask)),
        },
        "wrong_video_q_bd": distribution(scores["q_bd"][wrong]),
        "hard_positive": {
            "rows": int(np.sum(hard)),
            "q_joint": distribution(scores["q_joint"][hard]),
            "q_bd": distribution(scores["q_bd"][hard]),
            "e_fp": distribution(scores["e_fp"][hard]),
        },
        "feature_stats_finite": bool(all(
            math.isfinite(value)
            for value in checkpoint["feature_stats"]["mean"] + checkpoint["feature_stats"]["std"]
        )),
    }
    if (
        diagnostics["r2_features_enabled"] or any(diagnostics["collapse"].values())
        or not diagnostics["feature_stats_finite"]
    ):
        diagnostics["status"] = "FAIL"
    write_json(args.output_json, diagnostics)
    rows_md = "\n".join(
        f"| {key} | {value:.6f} |" for key, value in diagnostics["metrics"].items()
    )
    md = f"""# C3.1 model diagnostics

Status: `{diagnostics['status']}`  
Scope: `train_calib_only`  
Official val used: `false`

| Metric | Value |
|---|---:|
{rows_md}

- Rows: {len(x)}
- Wrong-video Q_bd mean/std: {diagnostics['wrong_video_q_bd']['mean']:.6f}/{diagnostics['wrong_video_q_bd']['std']:.6f}
- Collapsed heads: {[key for key, value in diagnostics['collapse'].items() if value]}
- R2 features enabled: {diagnostics['r2_features_enabled']}
"""
    Path(args.output_md).write_text(md, encoding="utf-8")
    partial = str(args.prediction_cache) + ".partial"
    Path(args.prediction_cache).parent.mkdir(parents=True, exist_ok=True)
    with open(partial, "wb") as f:
        np.savez(
            f,
            q_joint_logit=outputs["q_joint_logit"],
            q_bd_logit=outputs["q_bd_logit"],
            e_fp_logit=outputs["e_fp_logit"],
        )
    os.replace(partial, args.prediction_cache)
    print(json.dumps(diagnostics, indent=2))


if __name__ == "__main__":
    main()
