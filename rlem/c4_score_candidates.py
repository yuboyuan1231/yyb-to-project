#!/usr/bin/env python
"""Score fixed candidates with a frozen C3.1 evidence-head checkpoint.

This creates the compact scored candidate JSONL consumed by C4-lite.  It does
not change candidates and refuses official val by default unless explicitly
allowed for a separately authorized one-shot.
"""

import argparse
import gzip
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from rlem.evidence_dataset import FeatureStats  # noqa: E402
from rlem.evidence_model import EvidenceMLP, FlexibleEvidenceMLP  # noqa: E402
from rlem.io_utils import iter_jsonl, read_json, write_json  # noqa: E402
from rlem.c4_lite_utils import (  # noqa: E402
    EPS, atomic_text_writer, load_desc_ids, replace_atomic, sigmoid_temperature,
)

COMPACT_KEEP = [
    "desc_id", "desc", "video_idx", "video_name", "start_idx", "end_idx",
    "start_time", "end_time", "rank_base", "s_base", "r1", "is_gt_video",
    "gt_vid_name", "gt_ts", "iou", "y_joint", "y_joint_05", "y_joint_07",
    "y_bd", "m_bd", "y_fp",
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--evidence_jsonl", required=True)
    p.add_argument("--desc_ids_filter", default=None)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--frozen_config", default="results/rlem_c31_capacity/best_config.json")
    p.add_argument("--output_jsonl", required=True)
    p.add_argument("--audit_json", required=True)
    p.add_argument("--split", choices=["train", "val"], required=True)
    p.add_argument("--allow_official_val", action="store_true")
    p.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    p.add_argument("--score_batch_size", type=int, default=8192)
    p.add_argument("--gzip_compresslevel", type=int, default=1)
    p.add_argument("--t_joint", type=float, default=1.5)
    p.add_argument("--t_bd", type=float, default=1.0)
    p.add_argument("--t_fp", type=float, default=1.0)
    p.add_argument("--expected_rows_per_query", type=int, default=200)
    p.add_argument("--max_rows", type=int, default=None)
    return p.parse_args()


def build_model_from_ckpt(ckpt: Dict):
    cfg = ckpt.get("model_cfg", {})
    input_dim = int(cfg.get("input_dim"))
    if ckpt.get("model_type") == "FlexibleEvidenceMLP" or "hidden_dims" in cfg:
        model = FlexibleEvidenceMLP(input_dim, cfg.get("hidden_dims", [256, 256, 128]), cfg.get("dropout", 0.1))
    else:
        model = EvidenceMLP(input_dim=input_dim, hidden_dim=cfg.get("hidden_dim", 128), dropout=cfg.get("dropout", 0.1))
    model.load_state_dict(ckpt["model_state"])
    return model


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_frozen_c31(path: str, ckpt_path: str, args):
    frozen = read_json(path)
    if frozen.get("status") != "C31_PASS" or frozen.get("official_val_used") is not False:
        raise ValueError("C4-lite requires the accepted train-calib-only C3.1 config")
    if sha256_file(ckpt_path) != frozen.get("checkpoint_sha256"):
        raise ValueError("C4-lite checkpoint hash differs from frozen C3.1")
    config = frozen["score_config"]
    expected = {
        "family": "C_centered_gated_boundary",
        "base_scale": 0.75,
        "a_joint": 1.75,
        "b_bd": 1.0,
        "d_fp": 0.5,
        "t_joint": args.t_joint,
        "t_bd": args.t_bd,
        "t_fp": args.t_fp,
    }
    for key, value in expected.items():
        actual = config.get(key)
        if isinstance(value, float):
            if not math.isclose(float(actual), value, rel_tol=0.0, abs_tol=1e-12):
                raise ValueError(f"Frozen C3.1 {key}={actual}, requested {value}")
        elif actual != value:
            raise ValueError(f"Frozen C3.1 {key}={actual}, expected {value}")
    return frozen, config, float(frozen["diagnostics"]["mean_qbd_train_calib"])


def load_model(ckpt_path: str, device):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    stats = FeatureStats.from_dict(ckpt["feature_stats"])
    forbidden = {"r2_raw", "r2_prob", "r2_tilde", "rank_r2", "r_abs_gap"}
    if forbidden & set(stats.feature_names):
        raise ValueError("C4-lite candidate scoring requires no-R2 feature stats")
    model = build_model_from_ckpt(ckpt)
    model.to(device).eval()
    return model, stats, ckpt


@torch.no_grad()
def score_batch(rows: List[Dict], model, stats, device, args, c31_config, mean_qbd):
    xs = np.stack([stats.transform_row(row) for row in rows]).astype(np.float32)
    logits = model(torch.from_numpy(xs).to(device))
    qjl = logits["q_joint_logit"].detach().cpu().numpy().astype(np.float32)
    qbl = logits["q_bd_logit"].detach().cpu().numpy().astype(np.float32)
    efl = logits["e_fp_logit"].detach().cpu().numpy().astype(np.float32)
    qj = sigmoid_temperature(qjl, args.t_joint)
    qb = sigmoid_temperature(qbl, args.t_bd)
    efp = sigmoid_temperature(efl, args.t_fp)
    out = []
    for i, row in enumerate(rows):
        item = {key: row.get(key) for key in COMPACT_KEEP if key in row}
        # Preserve base ranking evidence.
        item["base_log"] = float(math.log(max(float(row.get("s_base", 0.0)), EPS)))
        item["log_r1"] = float(math.log(max(float(row.get("r1", row.get("s_base", 0.0))), EPS)))
        item["log_boundary"] = float(item["base_log"] - item["log_r1"])
        item["q_joint_logit"] = float(qjl[i])
        item["q_bd_logit"] = float(qbl[i])
        item["e_fp_logit"] = float(efl[i])
        item["q_joint_cal"] = float(qj[i])
        item["q_bd_cal"] = float(qb[i])
        item["e_fp_cal"] = float(efp[i])
        boundary = float(qj[i]) * (float(qb[i]) - mean_qbd)
        item["mean_qbd_train_calib"] = mean_qbd
        item["centered_gated_boundary"] = boundary
        item["s_c31"] = float(
            c31_config["base_scale"] * item["base_log"]
            + c31_config["a_joint"] * float(qj[i])
            + c31_config["b_bd"] * boundary
            - c31_config["d_fp"] * float(efp[i])
        )
        item["official_val_used_in_scoring"] = bool(args.split == "val" and args.allow_official_val)
        out.append(item)
    return out


def main():
    args = parse_args()
    if args.split == "val" and not args.allow_official_val:
        raise ValueError("Refusing official val candidate scoring without --allow_official_val")
    if Path(args.output_jsonl).exists():
        raise FileExistsError(f"Output already exists: {args.output_jsonl}")
    allowed = load_desc_ids(args.desc_ids_filter)
    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    model, stats, ckpt = load_model(args.ckpt, device)
    frozen, c31_config, mean_qbd = load_frozen_c31(args.frozen_config, args.ckpt, args)
    writer, partial = atomic_text_writer(
        args.output_jsonl,
        gzip_output=args.output_jsonl.endswith(".gz"),
        gzip_compresslevel=args.gzip_compresslevel,
    )
    total = 0
    q_counts = {}
    bad_nonfinite = 0
    try:
        batch = []
        for row in tqdm(iter_jsonl(args.evidence_jsonl, max_rows=args.max_rows), desc="C4 score fixed candidates"):
            desc_id = str(row["desc_id"])
            if allowed is not None and desc_id not in allowed:
                continue
            batch.append(row)
            if len(batch) >= args.score_batch_size:
                for scored in score_batch(batch, model, stats, device, args, c31_config, mean_qbd):
                    q_counts[scored["desc_id"]] = q_counts.get(scored["desc_id"], 0) + 1
                    vals = [scored[k] for k in ["base_log", "log_r1", "log_boundary", "q_joint_cal", "q_bd_cal", "e_fp_cal", "s_c31"]]
                    bad_nonfinite += int(not np.all(np.isfinite(vals)))
                    writer.write(json.dumps(scored, ensure_ascii=False, separators=(",", ":")) + "\n")
                    total += 1
                batch.clear()
        if batch:
            for scored in score_batch(batch, model, stats, device, args, c31_config, mean_qbd):
                q_counts[scored["desc_id"]] = q_counts.get(scored["desc_id"], 0) + 1
                vals = [scored[k] for k in ["base_log", "log_r1", "log_boundary", "q_joint_cal", "q_bd_cal", "e_fp_cal", "s_c31"]]
                bad_nonfinite += int(not np.all(np.isfinite(vals)))
                writer.write(json.dumps(scored, ensure_ascii=False, separators=(",", ":")) + "\n")
                total += 1
        writer.close()
        replace_atomic(partial, args.output_jsonl)
    except BaseException:
        writer.close()
        if os.path.exists(partial):
            os.remove(partial)
        raise
    dist = {}
    for count in q_counts.values():
        dist[str(count)] = dist.get(str(count), 0) + 1
    observed_ids = {str(value) for value in q_counts}
    missing_ids = sorted(allowed - observed_ids) if allowed is not None else []
    extra_ids = sorted(observed_ids - allowed) if allowed is not None else []
    audit = {
        "status": "PASS" if bad_nonfinite == 0 and q_counts and set(q_counts.values()) == {args.expected_rows_per_query} and not missing_ids and not extra_ids else "FAIL",
        "scope": (
            "train_fit_only" if args.split == "train" and args.desc_ids_filter and "train_fit" in Path(args.desc_ids_filter).name
            else "train_calib_only" if args.split == "train" and args.desc_ids_filter and "train_calib" in Path(args.desc_ids_filter).name
            else f"{args.split}_only"
        ),
        "official_val_used": False if args.split == "train" else bool(args.allow_official_val),
        "rows": total,
        "queries": len(q_counts),
        "rows_per_query_distribution": dist,
        "nonfinite_rows": bad_nonfinite,
        "missing_filtered_queries": len(missing_ids),
        "extra_filtered_queries": len(extra_ids),
        "feature_count": len(stats.feature_names),
        "feature_names": stats.feature_names,
        "r2_features_disabled": True,
        "frozen_c31_config": c31_config,
        "frozen_c31_checkpoint_sha256": frozen["checkpoint_sha256"],
        "mean_qbd_train_calib": mean_qbd,
        "temperature": {"joint": args.t_joint, "bd": args.t_bd, "fp": args.t_fp},
    }
    write_json(args.audit_json, audit)
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
