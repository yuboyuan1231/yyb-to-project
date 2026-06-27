#!/usr/bin/env python

"""Single frozen C3.1 rerank path for the authorized official-val one-shot."""

import argparse
import gzip
import hashlib
import json
import math
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from rlem.evidence_dataset import FeatureStats
from rlem.evidence_model import FlexibleEvidenceMLP
from rlem.io_utils import iter_jsonl, read_json, write_json
from utils.inference_utils import filter_vcmr_by_nms


EPS = 1e-8
EXPECTED_CONFIG = {
    "family": "C_centered_gated_boundary",
    "base_scale": 0.75,
    "a_joint": 1.75,
    "b_bd": 1.0,
    "d_fp": 0.5,
    "t_joint": 1.5,
    "t_bd": 1.0,
    "t_fp": 1.0,
}
COMPACT_FIELDS = [
    "desc_id", "desc", "video_idx", "video_name", "start_time", "end_time",
    "s_base", "rank_base", "is_gt_video", "iou", "y_fp", "y_joint_05",
    "q_joint_logit", "q_bd_logit", "e_fp_logit", "q_joint_cal", "q_bd_cal",
    "e_fp_cal", "mean_qbd_train_calib", "centered_gated_boundary",
    "base_score_used", "s_c31",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence_jsonl", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--frozen_config", required=True)
    parser.add_argument("--train_calib_prediction_cache", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--save_scored_jsonl", required=True)
    parser.add_argument("--dataset_config", required=True)
    parser.add_argument("--split", choices=["val"], required=True)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--score_batch_size", type=int, default=8192)
    parser.add_argument("--effective_top_n", type=int, default=100)
    parser.add_argument("--max_after_nms", type=int, default=100)
    parser.add_argument("--nms_thd", type=float, default=0.7)
    parser.add_argument("--expected_rows_per_query", type=int, default=200)
    return parser.parse_args()


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sigmoid_temperature(logit, temperature):
    value = np.clip(logit.astype(np.float64) / float(temperature), -40.0, 40.0)
    return (1.0 / (1.0 + np.exp(-value))).astype(np.float32)


def load_frozen(args, device):
    frozen = read_json(args.frozen_config)
    if frozen.get("status") != "C31_PASS" or frozen.get("official_val_used") is not False:
        raise ValueError("Not an accepted train-calib-only C3.1 frozen config")
    config = frozen["score_config"]
    for key, expected in EXPECTED_CONFIG.items():
        actual = config.get(key)
        if isinstance(expected, float):
            if not math.isclose(float(actual), expected, rel_tol=0.0, abs_tol=1e-12):
                raise ValueError(f"Frozen {key}={actual}, expected={expected}")
        elif actual != expected:
            raise ValueError(f"Frozen {key}={actual}, expected={expected}")
    if sha256_file(args.ckpt) != frozen["checkpoint_sha256"]:
        raise ValueError("Checkpoint SHA256 differs from frozen C3.1 best_config")
    if sha256_file(args.train_calib_prediction_cache) != frozen["prediction_cache_sha256"]:
        raise ValueError("Train-calib prediction cache SHA256 differs from frozen C3.1 best_config")
    checkpoint = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    if checkpoint.get("model_type") != "FlexibleEvidenceMLP":
        raise ValueError("C3.1 one-shot requires FlexibleEvidenceMLP")
    cfg = checkpoint["model_cfg"]
    model = FlexibleEvidenceMLP(cfg["input_dim"], cfg["hidden_dims"], cfg["dropout"])
    model.load_state_dict(checkpoint["model_state"])
    model.to(device).eval()
    stats = FeatureStats.from_dict(checkpoint["feature_stats"])
    forbidden = {"r2_raw", "r2_prob", "r2_tilde", "rank_r2", "r_abs_gap"}
    if len(stats.feature_names) != 31 or forbidden & set(stats.feature_names):
        raise ValueError("C3.1 one-shot requires frozen 31-feature no-R2 schema")
    mean_frozen = float(frozen["diagnostics"]["mean_qbd_train_calib"])
    with np.load(args.train_calib_prediction_cache, allow_pickle=False) as cache:
        mean_recomputed = float(np.mean(sigmoid_temperature(cache["q_bd_logit"], config["t_bd"])))
    if not math.isclose(mean_frozen, mean_recomputed, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(f"Frozen/recomputed train-calib Q_bd mean mismatch: {mean_frozen} vs {mean_recomputed}")
    return model, stats, frozen, config, mean_frozen, mean_recomputed


def load_video2idx(dataset_config, split):
    cfg = read_json(dataset_config)
    path = cfg["video_duration_idx_path"]
    if not os.path.isabs(path):
        path = os.path.join(cfg["root_path"], path)
    return {name: int(value[1]) for name, value in read_json(path)[split].items()}


def atomic_scored_writer(path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    partial = path[:-3] + ".partial.gz" if path.endswith(".gz") else path + ".partial"
    writer = gzip.open(partial, "wt", encoding="utf-8", compresslevel=6) if path.endswith(".gz") else open(partial, "w", encoding="utf-8")
    return writer, partial


@torch.no_grad()
def score_batch(rows, model, stats, device, config, mean_qbd):
    x = np.stack([stats.transform_row(row) for row in rows])
    logits = model(torch.from_numpy(x).float().to(device))
    values = {key: value.detach().cpu().numpy().astype(np.float32) for key, value in logits.items()}
    q_joint = sigmoid_temperature(values["q_joint_logit"], config["t_joint"])
    q_bd = sigmoid_temperature(values["q_bd_logit"], config["t_bd"])
    e_fp = sigmoid_temperature(values["e_fp_logit"], config["t_fp"])
    boundary = q_joint * (q_bd - mean_qbd)
    base = np.asarray([math.log(max(float(row["s_base"]), EPS)) for row in rows], dtype=np.float64)
    score = (
        config["base_scale"] * base + config["a_joint"] * q_joint
        + config["b_bd"] * boundary - config["d_fp"] * e_fp
    )
    output = []
    for i, row in enumerate(rows):
        new = dict(row)
        new.update({
            "q_joint_logit": float(values["q_joint_logit"][i]),
            "q_bd_logit": float(values["q_bd_logit"][i]),
            "e_fp_logit": float(values["e_fp_logit"][i]),
            "q_joint_cal": float(q_joint[i]), "q_bd_cal": float(q_bd[i]),
            "e_fp_cal": float(e_fp[i]), "mean_qbd_train_calib": mean_qbd,
            "centered_gated_boundary": float(boundary[i]),
            "base_score_used": float(base[i]), "s_c31": float(score[i]),
        })
        output.append({key: new.get(key) for key in COMPACT_FIELDS})
    return output


def main():
    args = parse_args()
    if args.effective_top_n != 100 or args.max_after_nms != 100 or not math.isclose(args.nms_thd, 0.7):
        raise ValueError("C3.1 one-shot protocol requires top_n=100, max_after=100, NMS=0.7")
    if Path(args.output_json).exists() or Path(args.save_scored_jsonl).exists():
        raise FileExistsError("C3.1 one-shot outputs already exist; rerun is forbidden")
    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    model, stats, frozen, config, mean_qbd, mean_recomputed = load_frozen(args, device)
    groups = defaultdict(list)
    writer, partial = atomic_scored_writer(args.save_scored_jsonl)
    try:
        batch = []
        for row in tqdm(iter_jsonl(args.evidence_jsonl), desc="C3.1 one-shot read+score"):
            batch.append(row)
            if len(batch) >= args.score_batch_size:
                for scored in score_batch(batch, model, stats, device, config, mean_qbd):
                    groups[scored["desc_id"]].append(scored)
                    writer.write(json.dumps(scored, ensure_ascii=False, separators=(",", ":")) + "\n")
                batch.clear()
        if batch:
            for scored in score_batch(batch, model, stats, device, config, mean_qbd):
                groups[scored["desc_id"]].append(scored)
                writer.write(json.dumps(scored, ensure_ascii=False, separators=(",", ":")) + "\n")
        writer.close()
        counts = Counter(len(rows) for rows in groups.values())
        if counts != Counter({args.expected_rows_per_query: len(groups)}):
            raise ValueError(f"Unexpected rows/query distribution: {dict(counts)}")
        os.replace(partial, args.save_scored_jsonl)
    except BaseException:
        writer.close()
        if os.path.exists(partial):
            os.remove(partial)
        raise

    vcmr = []
    for desc_id, rows in tqdm(groups.items(), desc="C3.1 one-shot submission"):
        order = sorted(range(len(rows)), key=lambda i: (-float(rows[i]["s_c31"]), int(rows[i]["rank_base"])))
        candidates = [[
            int(rows[i]["video_idx"]), float(rows[i]["start_time"]),
            float(rows[i]["end_time"]), float(rows[i]["s_c31"]),
        ] for i in order[:args.effective_top_n]]
        predictions = filter_vcmr_by_nms(
            candidates, nms_threshold=args.nms_thd,
            max_before_nms=args.effective_top_n, max_after_nms=args.max_after_nms,
        )
        vcmr.append({"desc_id": desc_id, "desc": rows[0].get("desc", ""), "predictions": predictions})
    write_json(args.output_json, {"video2idx": load_video2idx(args.dataset_config, args.split), "VCMR": vcmr}, pretty=False)
    print(json.dumps({
        "status": "ONE_SHOT_RERANK_COMPLETE", "queries": len(vcmr), "rows_per_query": dict(counts),
        "model_id": frozen["model_id"], "score_config": config,
        "mean_qbd_train_calib_frozen": mean_qbd,
        "mean_qbd_train_calib_recomputed": mean_recomputed,
        "official_val_calibration_performed": False,
    }, indent=2))


if __name__ == "__main__":
    main()
