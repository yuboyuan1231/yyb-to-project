#!/usr/bin/env python
"""Build frozen-C4 video-level train datasets for C4-r2-cal.

All score features are label-free and use the already frozen C3.1/C4-lite
constants.  Targets are read only from train evidence caches.  Aggregation is
vectorized; top-k segment statistics use bounded GPU chunks when CUDA is
available, with an exact NumPy fallback.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from rlem.c4_protocol import C4Protocol, write_protocol_manifest  # noqa: E402

FROZEN_MEAN_QBD = 0.3990069627761841
FROZEN_ALPHA_BD = 1.25
FROZEN_GAMMA_FP = 0.5
FROZEN_MU_VIDEO_ROWS = 0.13249324262142181
FROZEN_MU_SPAN_ROWS = 0.03713987395167351

VIDEO_FEATURE_NAMES = [
    "log_r1_max",
    "base_log_max", "base_log_mean",
    "log_boundary_max", "log_boundary_mean",
    "s_c31_max", "s_c31_mean", "s_c31_top3_mean", "s_c31_std",
    "s_c4_lite_max", "s_c4_lite_mean", "s_c4_lite_top3_mean", "s_c4_lite_std",
    "g_video", "g_span_mean", "g_span_top3_mean",
    "q_joint_max", "q_joint_mean", "q_joint_top3_mean",
    "q_bd_centered_max", "q_bd_centered_mean", "q_bd_centered_top3_mean",
    "e_fp_min", "e_fp_mean", "e_fp_bottom3_mean",
    "best_rank_inv", "span_count_norm",
]


def _reduce_mean(values: np.ndarray, sort_idx: np.ndarray, offsets: np.ndarray, counts: np.ndarray) -> np.ndarray:
    return (np.add.reduceat(values[sort_idx].astype(np.float64), offsets) / counts).astype(np.float32)


def _reduce_max(values: np.ndarray, sort_idx: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    return np.maximum.reduceat(values[sort_idx], offsets).astype(np.float32)


def _reduce_min(values: np.ndarray, sort_idx: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    return np.minimum.reduceat(values[sort_idx], offsets).astype(np.float32)


def _topk_segment_mean(
    values: np.ndarray,
    sort_idx: np.ndarray,
    offsets: np.ndarray,
    counts: np.ndarray,
    *,
    largest: bool,
    device: torch.device,
    groups_per_chunk: int = 100_000,
) -> np.ndarray:
    """Exact mean of the largest/smallest three values in every segment."""
    sorted_values = values[sort_idx].astype(np.float32, copy=False)
    out = np.empty(len(offsets), dtype=np.float32)
    total_rows = len(sort_idx)
    for g0 in range(0, len(offsets), groups_per_chunk):
        g1 = min(g0 + groups_per_chunk, len(offsets))
        r0 = int(offsets[g0])
        r1 = int(offsets[g1]) if g1 < len(offsets) else total_rows
        local_counts_np = counts[g0:g1].astype(np.int64, copy=False)
        max_count = int(local_counts_np.max())
        local_values = torch.from_numpy(sorted_values[r0:r1]).to(device)
        local_counts = torch.from_numpy(local_counts_np).to(device)
        group = torch.repeat_interleave(torch.arange(g1 - g0, device=device), local_counts)
        starts = torch.cumsum(local_counts, dim=0) - local_counts
        pos = torch.arange(r1 - r0, device=device) - torch.repeat_interleave(starts, local_counts)
        fill = -torch.inf if largest else torch.inf
        padded = torch.full((g1 - g0, max_count), fill, dtype=torch.float32, device=device)
        padded[group, pos] = local_values
        k = min(3, max_count)
        top = torch.topk(padded, k=k, dim=1, largest=largest).values
        valid = torch.minimum(local_counts, torch.tensor(k, device=device)).to(torch.float32)
        top = torch.where(torch.isfinite(top), top, torch.zeros_like(top))
        out[g0:g1] = (top.sum(dim=1) / valid).cpu().numpy()
        del local_values, local_counts, group, starts, pos, padded, top
    return out


def frozen_c4_lite_row_scores(cache: np.lib.npyio.NpzFile):
    qj = cache["q_joint"].astype(np.float32)
    qb_centered = cache["q_bd"].astype(np.float32) - np.float32(FROZEN_MEAN_QBD)
    efp = cache["e_fp"].astype(np.float32)
    g_span = qj + np.float32(FROZEN_ALPHA_BD) * qj * qb_centered - np.float32(FROZEN_GAMMA_FP) * efp
    group_max = np.full(len(cache["group_ids_sorted_unique"]), -np.inf, dtype=np.float32)
    np.maximum.at(group_max, cache["row_group_id"], g_span)
    g_video_rows = group_max[cache["row_group_id"]]
    s_c4 = (
        np.float32(0.75) * cache["s_c31"].astype(np.float32)
        + g_video_rows - np.float32(FROZEN_MU_VIDEO_ROWS)
        + np.float32(0.5) * (g_span - np.float32(FROZEN_MU_SPAN_ROWS))
    ).astype(np.float32)
    return s_c4, g_span.astype(np.float32), group_max, qb_centered


def build_video_dataset_from_cache(cache: np.lib.npyio.NpzFile, device: str = "cuda") -> Dict[str, np.ndarray]:
    sort_idx = cache["group_sort_idx"].astype(np.int64)
    offsets = cache["group_offsets"].astype(np.int64)
    total_rows = len(sort_idx)
    counts = np.diff(np.r_[offsets, total_rows]).astype(np.int64)
    if np.any(counts <= 0):
        raise ValueError("C4 cache contains empty video groups")
    torch_device = torch.device("cuda" if device == "cuda" and torch.cuda.is_available() else "cpu")

    s_c4, g_span, g_video, qb_centered = frozen_c4_lite_row_scores(cache)
    qj = cache["q_joint"].astype(np.float32)
    efp = cache["e_fp"].astype(np.float32)
    s_c31 = cache["s_c31"].astype(np.float32)

    c31_mean = _reduce_mean(s_c31, sort_idx, offsets, counts)
    c4_mean = _reduce_mean(s_c4, sort_idx, offsets, counts)
    c31_sq_mean = _reduce_mean(s_c31 * s_c31, sort_idx, offsets, counts)
    c4_sq_mean = _reduce_mean(s_c4 * s_c4, sort_idx, offsets, counts)
    rank_min = _reduce_min(cache["rank_base"].astype(np.float32), sort_idx, offsets)

    columns = [
        _reduce_max(cache["log_r1"], sort_idx, offsets),
        _reduce_max(cache["base_log"], sort_idx, offsets), _reduce_mean(cache["base_log"], sort_idx, offsets, counts),
        _reduce_max(cache["log_boundary"], sort_idx, offsets), _reduce_mean(cache["log_boundary"], sort_idx, offsets, counts),
        _reduce_max(s_c31, sort_idx, offsets), c31_mean,
        _topk_segment_mean(s_c31, sort_idx, offsets, counts, largest=True, device=torch_device),
        np.sqrt(np.maximum(c31_sq_mean - c31_mean * c31_mean, 0.0)).astype(np.float32),
        _reduce_max(s_c4, sort_idx, offsets), c4_mean,
        _topk_segment_mean(s_c4, sort_idx, offsets, counts, largest=True, device=torch_device),
        np.sqrt(np.maximum(c4_sq_mean - c4_mean * c4_mean, 0.0)).astype(np.float32),
        g_video.astype(np.float32), _reduce_mean(g_span, sort_idx, offsets, counts),
        _topk_segment_mean(g_span, sort_idx, offsets, counts, largest=True, device=torch_device),
        _reduce_max(qj, sort_idx, offsets), _reduce_mean(qj, sort_idx, offsets, counts),
        _topk_segment_mean(qj, sort_idx, offsets, counts, largest=True, device=torch_device),
        _reduce_max(qb_centered, sort_idx, offsets), _reduce_mean(qb_centered, sort_idx, offsets, counts),
        _topk_segment_mean(qb_centered, sort_idx, offsets, counts, largest=True, device=torch_device),
        _reduce_min(efp, sort_idx, offsets), _reduce_mean(efp, sort_idx, offsets, counts),
        _topk_segment_mean(efp, sort_idx, offsets, counts, largest=False, device=torch_device),
        (1.0 / np.maximum(rank_min, 1.0)).astype(np.float32),
        (counts.astype(np.float32) / np.float32(200.0)),
    ]
    features = np.column_stack(columns).astype(np.float32)
    representative = sort_idx[offsets]
    best_iou = _reduce_max(cache["iou"].astype(np.float32), sort_idx, offsets)
    label_relevant = (_reduce_max(cache["is_gt_video"].astype(np.float32), sort_idx, offsets) > 0.5).astype(np.float32)
    best_iou = np.where(label_relevant > 0.5, np.maximum(best_iou, 0.0), 0.0).astype(np.float32)
    return {
        "features": features,
        "label_relevant": label_relevant,
        "label_best_iou": best_iou,
        "label_any_05": (best_iou >= 0.5).astype(np.float32),
        "label_any_07": (best_iou >= 0.7).astype(np.float32),
        "group_id": cache["group_ids_sorted_unique"].astype(np.int32),
        "query_index": cache["query_index"][representative].astype(np.int32),
        "video_idx": cache["video_idx"][representative].astype(np.int32),
        "feature_names": np.asarray(VIDEO_FEATURE_NAMES, dtype=object),
        "frozen_mean_qbd": np.asarray([FROZEN_MEAN_QBD], dtype=np.float64),
        "frozen_mu_video_rows": np.asarray([FROZEN_MU_VIDEO_ROWS], dtype=np.float64),
        "frozen_mu_span_rows": np.asarray([FROZEN_MU_SPAN_ROWS], dtype=np.float64),
    }


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cache_npz", required=True)
    p.add_argument("--output_npz", required=True)
    p.add_argument("--manifest_json", required=True)
    p.add_argument("--split", choices=["train"], default="train")
    p.add_argument("--split_role", choices=["train_fit", "train_calib"], required=True)
    p.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    return p.parse_args()


def main():
    args = parse_args()
    protocol = C4Protocol(stage="c4_r2_build_video_dataset", split=args.split)
    protocol.validate()
    with np.load(args.cache_npz, allow_pickle=True) as cache:
        out = build_video_dataset_from_cache(cache, device=args.device)
        candidate_rows = int(len(cache["row_group_id"]))
        query_count = int(len(cache["desc_ids"]))
    path = Path(args.output_npz)
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **out)
    rel = out["label_relevant"] > 0.5
    manifest = {
        "status": "PASS" if np.all(np.isfinite(out["features"])) else "FAIL",
        "split_role": args.split_role,
        "cache_npz": args.cache_npz,
        "output_npz": args.output_npz,
        "queries": query_count,
        "candidate_rows": candidate_rows,
        "video_rows": int(len(rel)),
        "feature_dim": int(out["features"].shape[1]),
        "positive_video_rows": int(rel.sum()),
        "negative_video_rows": int((~rel).sum()),
        "iou05_positive_video_rows": int(out["label_any_05"].sum()),
        "iou07_positive_video_rows": int(out["label_any_07"].sum()),
        "nonfinite_values": int((~np.isfinite(out["features"])).sum()),
        "aggregation_device": str("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu"),
        "baseline_features": ["frozen_s_c31", "frozen_c4_lite_c4_00394"],
        "frozen_constants": {
            "mean_qbd": FROZEN_MEAN_QBD, "alpha_bd": FROZEN_ALPHA_BD,
            "gamma_fp": FROZEN_GAMMA_FP, "mu_video_rows": FROZEN_MU_VIDEO_ROWS,
            "mu_span_rows": FROZEN_MU_SPAN_ROWS,
        },
        "targets_from_train_evidence_only": True,
        "official_val_used": False,
    }
    write_protocol_manifest(args.manifest_json, protocol, manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
