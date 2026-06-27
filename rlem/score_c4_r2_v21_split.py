#!/usr/bin/env python
"""Score train_fit/train_calib fixed candidates with frozen C4-r2-cal-v2.1.

This is a C5-prep helper.  It materializes C4-final row scores for train splits
so C5-lite can use C4-r2-cal-v2.1, not C4-lite, as its baseline.

It is not an official-val script.  For official val use the existing audited
`rerank_c4_r2_v21_one_shot.py`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from rlem.c4_lite_utils import write_json  # noqa: E402
from rlem.c4_r2_dataset import ArrayStandardizer  # noqa: E402
from rlem.c4_r2_v2_model import build_v2_model  # noqa: E402
from rlem.c4_video_targets import VIDEO_FEATURE_NAMES, build_video_dataset_from_cache, frozen_c4_lite_row_scores  # noqa: E402

CONFIG_ID = "v21_00444"
EXPECTED_FREEZE_STATUS = "C4_R2_CAL_V21_FREEZE_REVIEW_PASS"
EXPECTED_SELECTION_REASON = "robust_non_isolated_boundary_preferred_over_higher_isolated_score"
WEIGHTS = {"rel_logit": 0.075, "iou05_logit": 0.0, "iou07_logit": 0.05, "quality_logit": -0.1}
Z_STATS = {
    "rel_logit": {"mean": -3.862743616104126, "std": 2.063218832015991},
    "iou05_logit": {"mean": -3.92142653465271, "std": 2.1069819927215576},
    "iou07_logit": {"mean": -4.213388919830322, "std": 2.1755518913269043},
    "quality_logit": {"mean": -3.8732757568359375, "std": 1.7478158473968506},
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cache_npz", required=True)
    p.add_argument(
        "--video_dataset_npz",
        default=None,
        help="Optional existing C4 27-d query-video dataset; avoids recomputing exact features.",
    )
    p.add_argument("--model_ckpt", required=True)
    p.add_argument("--freeze_manifest", required=True)
    p.add_argument("--split_role", choices=["train_fit", "train_calib"], required=True)
    p.add_argument("--output_logits_npz", required=True)
    p.add_argument("--output_scores_npz", required=True)
    p.add_argument("--output_audit_json", required=True)
    p.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    p.add_argument("--batch_size", type=int, default=8192)
    return p.parse_args()


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def require_freeze(path):
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    if manifest.get("status") != EXPECTED_FREEZE_STATUS:
        raise ValueError("Durable v2.1 freeze status mismatch")
    if manifest.get("selected_config") != CONFIG_ID:
        raise ValueError("Only v21_00444 is authorized for C4-final scoring")
    if manifest.get("selection_reason") != EXPECTED_SELECTION_REASON:
        raise ValueError("Freeze selection reason mismatch")
    actual_weights = manifest.get("weights", {})
    expected_weights = {"a_rel": 0.075, "b_iou05": 0.0, "c_iou07": 0.05, "d_quality": -0.1}
    if actual_weights != expected_weights:
        raise ValueError(f"Frozen weights mismatch: {actual_weights}")
    for key, expected in Z_STATS.items():
        actual = manifest["z_stats"][key]
        for field in ("mean", "std"):
            if not math.isclose(float(actual[field]), expected[field], rel_tol=0.0, abs_tol=1e-12):
                raise ValueError(f"Frozen {key} {field} mismatch")
    return manifest


def atomic_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = Path(str(path) + ".partial")
    with open(partial, "wb") as f:
        np.savez_compressed(f, **arrays)
    os.replace(partial, path)


def infer_logits(features, checkpoint, device, batch_size):
    cfg = checkpoint["model_config"]
    if cfg.get("group") != "v2_cb" or cfg.get("input_dim") != len(VIDEO_FEATURE_NAMES):
        raise ValueError("Frozen v21 backbone checkpoint mismatch")
    standardizer = ArrayStandardizer.from_dict(checkpoint["standardizer"])
    if standardizer.feature_names != VIDEO_FEATURE_NAMES:
        raise ValueError("Feature names do not match frozen model standardizer")
    x = standardizer.transform(features)
    model = build_v2_model("v2_cb", x.shape[1]).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    chunks = {key: [] for key in ("rel_logit", "iou05_logit", "iou07_logit", "quality_logit")}
    with torch.inference_mode():
        for start in range(0, len(x), batch_size):
            batch = torch.from_numpy(x[start:start + batch_size]).to(device, non_blocking=True)
            out = model(batch)
            for key in chunks:
                chunks[key].append(out[key].detach().cpu().numpy().astype(np.float32, copy=False))
    return {key: np.concatenate(value).astype(np.float32, copy=False) for key, value in chunks.items()}


def main():
    args = parse_args()
    outputs = [args.output_logits_npz, args.output_scores_npz, args.output_audit_json]
    existing = [path for path in outputs if Path(path).exists()]
    if existing:
        raise FileExistsError(f"Refusing overwrite: {existing}")
    freeze = require_freeze(args.freeze_manifest)
    with np.load(args.cache_npz, allow_pickle=True) as payload:
        cache = {key: payload[key] for key in payload.files}
    if args.video_dataset_npz:
        with np.load(args.video_dataset_npz, allow_pickle=True) as payload:
            required = ["features", "feature_names", "group_id", "query_index", "video_idx"]
            missing = [key for key in required if key not in payload.files]
            if missing:
                raise KeyError(f"Existing video dataset lacks {missing}")
            video_data = {key: payload[key] for key in required}
        names = [str(value) for value in video_data["feature_names"].tolist()]
        if names != VIDEO_FEATURE_NAMES:
            raise ValueError("Existing video dataset feature schema mismatch")
    else:
        video_data = build_video_dataset_from_cache(cache, device=args.device)
    if video_data["features"].shape[1] != len(VIDEO_FEATURE_NAMES):
        raise ValueError("C4-final video feature dim mismatch")
    if not np.array_equal(video_data["group_id"].astype(np.int64), cache["group_ids_sorted_unique"].astype(np.int64)):
        raise ValueError("Video features/cache group IDs mismatch")
    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.model_ckpt, map_location="cpu", weights_only=False)
    logits = infer_logits(video_data["features"], checkpoint, device, args.batch_size)
    atomic_npz(
        args.output_logits_npz,
        group_id=video_data["group_id"].astype(np.int32),
        query_index=video_data["query_index"].astype(np.int32),
        video_idx=video_data["video_idx"].astype(np.int32),
        **logits,
    )
    row_gid = cache["row_group_id"].astype(np.int64)
    c4_score, _, _, _ = frozen_c4_lite_row_scores(cache)
    residual = np.zeros_like(c4_score, dtype=np.float32)
    for key, weight in WEIGHTS.items():
        if weight == 0.0:
            continue
        stat = Z_STATS[key]
        row_logit = logits[key][row_gid]
        z = ((row_logit - np.float32(stat["mean"])) / np.float32(stat["std"])).astype(np.float32)
        residual += np.float32(weight) * z
    s_v21 = (c4_score + residual).astype(np.float32)
    if not np.all(np.isfinite(s_v21)):
        raise ValueError("Non-finite C4-final scores")
    atomic_npz(args.output_scores_npz, row_group_id=cache["row_group_id"].astype(np.int32), s_c4_lite=c4_score, s_v21=s_v21)
    audit = {
        "status": "PASS",
        "stage": "C4-r2-cal-v2.1 train split scoring for C5-lite",
        "split_role": args.split_role,
        "official_val_used": False,
        "post_val_adjustment": False,
        "score_grid_on_val": False,
        "config_id": CONFIG_ID,
        "queries": int(len(cache["desc_ids"])),
        "candidate_rows": int(len(cache["row_group_id"])),
        "video_groups": int(len(video_data["group_id"])),
        "feature_dim": int(video_data["features"].shape[1]),
        "weights": freeze["weights"],
        "z_stats": Z_STATS,
        "artifacts": {
            "cache_npz": {"path": args.cache_npz, "sha256": sha256(args.cache_npz)},
            "video_dataset_npz": (
                {"path": args.video_dataset_npz, "sha256": sha256(args.video_dataset_npz)}
                if args.video_dataset_npz else None
            ),
            "model_ckpt": {"path": args.model_ckpt, "sha256": sha256(args.model_ckpt)},
            "freeze_manifest": {"path": args.freeze_manifest, "sha256": sha256(args.freeze_manifest)},
            "output_logits_npz": {"path": args.output_logits_npz, "sha256": sha256(args.output_logits_npz)},
            "output_scores_npz": {"path": args.output_scores_npz, "sha256": sha256(args.output_scores_npz)},
        },
        "used": {"C4-final": True, "C5-lite": False, "C5-main": False, "C6": False},
    }
    write_json(args.output_audit_json, audit)
    print(json.dumps({"status": "PASS", "scores": args.output_scores_npz}, indent=2))


if __name__ == "__main__":
    main()
