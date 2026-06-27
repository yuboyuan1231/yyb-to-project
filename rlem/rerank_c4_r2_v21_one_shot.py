#!/usr/bin/env python
"""Run exactly the frozen C4-r2-cal-v2.1 v21_00444 official-val scorer.

This entry point has no search surface.  It consumes the frozen C4 cache, the
already-trained v2_cb backbone used by v2.1, and train_calib z statistics from
the durable freeze manifest.  It refuses official val unless explicitly
authorized and refuses to overwrite any output.
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

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rlem.c4_lite_utils import submission_from_groups, write_json  # noqa: E402
from rlem.c4_protocol import C4Protocol  # noqa: E402
from rlem.c4_r2_dataset import ArrayStandardizer  # noqa: E402
from rlem.c4_r2_v2_model import build_v2_model  # noqa: E402
from rlem.c4_video_targets import (  # noqa: E402
    VIDEO_FEATURE_NAMES,
    build_video_dataset_from_cache,
    frozen_c4_lite_row_scores,
)


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
    p.add_argument("--scored_candidates", required=True)
    p.add_argument("--model_ckpt", required=True)
    p.add_argument("--freeze_manifest", required=True)
    p.add_argument("--dataset_config", required=True)
    p.add_argument("--output_logits_npz", required=True)
    p.add_argument("--output_scores_npz", required=True)
    p.add_argument("--output_submission_json", required=True)
    p.add_argument("--output_audit_json", required=True)
    p.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    p.add_argument("--batch_size", type=int, default=8192)
    p.add_argument("--effective_top_n", type=int, default=100)
    p.add_argument("--max_after_nms", type=int, default=100)
    p.add_argument("--nms_thd", type=float, default=0.7)
    p.add_argument("--allow_official_val", action="store_true")
    return p.parse_args()


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def require_exact_freeze(path):
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    if manifest.get("status") != EXPECTED_FREEZE_STATUS:
        raise ValueError("Durable v2.1 freeze status mismatch")
    if manifest.get("selected_config") != CONFIG_ID:
        raise ValueError("Only v21_00444 is authorized")
    if manifest.get("selection_reason") != EXPECTED_SELECTION_REASON:
        raise ValueError("Freeze selection reason mismatch")
    if manifest.get("official_val_authorized") is not False:
        raise ValueError("Durable train_calib freeze manifest must precede authorization")
    actual_weights = manifest.get("weights", {})
    expected_weights = {"a_rel": 0.075, "b_iou05": 0.0, "c_iou07": 0.05, "d_quality": -0.1}
    if actual_weights != expected_weights:
        raise ValueError(f"Frozen weights mismatch: {actual_weights}")
    for key, expected in Z_STATS.items():
        actual = manifest["z_stats"][key]
        for field in ("mean", "std"):
            if not math.isclose(float(actual[field]), expected[field], rel_tol=0.0, abs_tol=1e-12):
                raise ValueError(f"Frozen {key} {field} mismatch")
    constants = manifest.get("retrieval_constants", {})
    if constants != {"effective_top_n": 100, "nms_thd": 0.7, "max_after_nms": 100}:
        raise ValueError("Frozen retrieval constants mismatch")
    return manifest


def atomic_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = Path(str(path) + ".partial")
    with open(partial, "wb") as f:
        np.savez_compressed(f, **arrays)
    os.replace(partial, path)


def atomic_json(path, obj, *, compact=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = Path(str(path) + ".partial")
    with open(partial, "w", encoding="utf-8") as f:
        if compact:
            json.dump(obj, f, ensure_ascii=False, separators=(",", ":"))
        else:
            json.dump(obj, f, ensure_ascii=False, indent=2, sort_keys=True)
            f.write("\n")
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


def movement(cache, base_score, new_score):
    offsets = cache["desc_offsets"].astype(np.int64)
    if not np.all(np.diff(offsets) == 200):
        raise ValueError("One-shot requires exactly 200 rows/query")
    queries = len(offsets) - 1
    base = base_score.reshape(queries, 200)
    new = new_score.reshape(queries, 200)
    base_order = np.argsort(-base, axis=1, kind="stable")
    new_order = np.argsort(-new, axis=1, kind="stable")
    ranks = np.broadcast_to(np.arange(200, dtype=np.int16), base_order.shape)
    base_inv = np.empty_like(base_order, dtype=np.int16)
    new_inv = np.empty_like(new_order, dtype=np.int16)
    np.put_along_axis(base_inv, base_order, ranks, axis=1)
    np.put_along_axis(new_inv, new_order, ranks, axis=1)
    hard = cache["y_joint_05"].reshape(queries, 200) > 0.5
    exits = int(np.sum(hard & (base_inv < 100) & (new_inv >= 100)))
    entries = int(np.sum(hard & (base_inv >= 100) & (new_inv < 100)))
    diff = base_inv.astype(np.float64) - new_inv.astype(np.float64)
    spearman = 1.0 - 6.0 * np.sum(diff * diff, axis=1) / (200.0 * (200.0 * 200.0 - 1.0))
    return {
        "relative_to": "frozen_C4_lite_c4_00394",
        "queries": queries,
        "top1_changed_ratio": float(np.mean(base_order[:, 0] != new_order[:, 0])),
        "hard_positive_top100_exits": exits,
        "hard_positive_top100_entries": entries,
        "hard_positive_top100_exit_ratio": float(exits / max(int(hard.sum()), 1)),
        "pearson_c4_lite_candidate": float(np.corrcoef(base_score.astype(np.float64), new_score.astype(np.float64))[0, 1]),
        "mean_within_query_spearman": float(np.mean(spearman)),
    }


def group_generator(cache, score):
    offsets = cache["desc_offsets"]
    for q in range(len(cache["desc_ids"])):
        start, end = int(offsets[q]), int(offsets[q + 1])
        yield [
            {
                "desc_id": int(cache["desc_ids"][q]),
                "desc": str(cache["desc_text"][q]),
                "video_idx": int(cache["video_idx"][i]),
                "start_time": float(cache["start_time"][i]),
                "end_time": float(cache["end_time"][i]),
                "rank_base": int(cache["rank_base"][i]),
                "s_v21": float(score[i]),
            }
            for i in range(start, end)
        ]


def main():
    args = parse_args()
    protocol = C4Protocol(
        stage="c4_r2_one_shot", split="val",
        official_val_allowed=args.allow_official_val,
        reads_official_val=True,
        effective_top_n=args.effective_top_n,
        max_after_nms=args.max_after_nms,
        nms_thd=args.nms_thd,
    )
    protocol.validate()
    if not args.allow_official_val:
        raise ValueError("Explicit official-val one-shot authorization is required")
    outputs = [args.output_logits_npz, args.output_scores_npz, args.output_submission_json, args.output_audit_json]
    existing = [path for path in outputs if Path(path).exists()]
    if existing:
        raise FileExistsError(f"One-shot rerun is forbidden; outputs exist: {existing}")
    freeze = require_exact_freeze(args.freeze_manifest)

    with np.load(args.cache_npz, allow_pickle=True) as payload:
        cache = {key: payload[key] for key in payload.files}
    if len(cache["desc_ids"]) != 10895 or len(cache["row_group_id"]) != 2179000:
        raise ValueError("Official-val cache shape mismatch")
    video_data = build_video_dataset_from_cache(cache, device=args.device)
    if video_data["features"].shape != (len(cache["group_ids_sorted_unique"]), 27):
        raise ValueError("Official-val video feature shape mismatch")
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
    v21_score = (c4_score + residual).astype(np.float32)
    if not np.all(np.isfinite(v21_score)):
        raise ValueError("Non-finite v21 scores")
    atomic_npz(
        args.output_scores_npz,
        row_group_id=cache["row_group_id"].astype(np.int32),
        s_c4_lite=c4_score,
        s_v21=v21_score,
    )

    movement_diag = movement(cache, c4_score, v21_score)
    submission = submission_from_groups(
        group_generator(cache, v21_score), "s_v21", args.dataset_config, "val",
        effective_top_n=args.effective_top_n,
        max_after_nms=args.max_after_nms,
        nms_thd=args.nms_thd,
    )
    row_counts = submission.pop("_row_counts")
    if row_counts != {200: 10895}:
        raise ValueError(f"Unexpected rows/query: {row_counts}")
    atomic_json(args.output_submission_json, submission, compact=True)

    audit = {
        **protocol.to_manifest(),
        "status": "PASS",
        "stage": "C4-r2-cal-v2.1 official-val one-shot rerank",
        "config_id": CONFIG_ID,
        "official_val_one_shot": True,
        "learned_config_count": 1,
        "raw_v2_cb_config_evaluated": False,
        "score_grid_on_val": False,
        "temperature_search_on_val": False,
        "post_val_adjustment": False,
        "official_val_mean_std_estimated": False,
        "z_stats_source": "train_calib_row_expanded_frozen",
        "weights": freeze["weights"],
        "z_stats": Z_STATS,
        "queries": 10895,
        "candidate_rows": 2179000,
        "video_groups": int(len(video_data["group_id"])),
        "feature_dim": 27,
        "inference_device": str(device),
        "movement_relative_to_frozen_c4_lite": movement_diag,
        "artifacts": {
            "official_val_scored_candidates": {"path": args.scored_candidates, "sha256": sha256(args.scored_candidates)},
            "official_val_c4_cache": {"path": args.cache_npz, "sha256": sha256(args.cache_npz)},
            "v21_model_checkpoint": {"path": args.model_ckpt, "sha256": sha256(args.model_ckpt)},
            "freeze_manifest": {"path": args.freeze_manifest, "sha256": sha256(args.freeze_manifest)},
            "official_val_v21_logits": {"path": args.output_logits_npz, "sha256": sha256(args.output_logits_npz)},
            "official_val_v21_row_scores": {"path": args.output_scores_npz, "sha256": sha256(args.output_scores_npz)},
            "official_val_v21_submission": {"path": args.output_submission_json, "sha256": sha256(args.output_submission_json)},
        },
        "successful_metric_result_count": 0,
        "used": {"C4-main": False, "R2": False, "VS": False, "C5": False, "C6": False},
    }
    write_json(args.output_audit_json, audit)
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
