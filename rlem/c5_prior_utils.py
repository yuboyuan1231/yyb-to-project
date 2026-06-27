#!/usr/bin/env python
"""Utilities for CONQUER-RLEM C5-lite-prior.

C5-lite-prior is the fixed-candidate, train_fit/train_calib-only bridge from
C4 fixed-candidate ranking to the blueprint's retrieval -> localization branch.

It explicitly separates two concepts:
  1. per-(query, video) temporal priors P_b/P_e/P_ctx/P_retloc(t);
  2. per-(query, video, span) prior features aggregated from those temporal
     priors.

The code intentionally refuses to fabricate a temporal prior from row-level
candidate scores.  If P_ctx/P_b/P_e are not exported, run the C5-0 read-only
export patch first.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from rlem.c4_lite_utils import (  # noqa: E402
    ALL_METRICS,
    PRIMARY_METRICS,
    flatten_eval_metrics,
    selection_score,
    submission_from_groups,
    write_json,
)
from standalone_eval.eval import eval_retrieval, load_jsonl  # noqa: E402
from rlem.grid_search_c3_weights import greedy_nms_indices, raw_metrics_from_selected  # noqa: E402

EPS = 1e-8
DEFAULT_CLIP_LENGTH = 1.5

# These are the row-level features used by the train-calib search.  They are
# all derived from temporal priors or from retrieval confidence x temporal prior
# interactions; they are not direct video-only score bonuses.
C5_PRIOR_TERM_NAMES = [
    "retctx_mass",
    "retbd_mass",
    "retctx_peak_inside",
    "retbd_peak_inside",
    "retctx_boundary_agree",
    "retbd_boundary_agree",
    "retctx_center_proximity",
    "retbd_center_proximity",
    "low_conf_efp_neg",
    "low_conf_unc_neg",
    "margin_z_x_retbd_mass",
    "rank_inv_x_retbd_mass",
]

# Raw span prior features that are saved for diagnostics.
C5_PRIOR_RAW_FEATURES = [
    "ctx_mass", "ctx_mean", "ctx_max", "ctx_peak_inside", "ctx_boundary_agree", "ctx_center_proximity",
    "bd_mass", "bd_mean", "bd_max", "bd_peak_inside", "bd_boundary_agree", "bd_center_proximity",
    "retctx_mass", "retbd_mass", "retloc_mass", "retloc_mean", "retloc_max", "retloc_peak_inside",
    "r_video_z", "r_video_sig", "margin_z", "rank_inv",
]

ALIAS = {
    "group_id": ["group_id", "group_ids", "row_group_id", "video_group_id"],
    "p_ctx": ["p_ctx", "P_ctx", "ctx_prior", "qal_ctx_prior", "p_qal"],
    "p_b": ["p_b", "P_b", "start_prob", "pb", "start_prior"],
    "p_e": ["p_e", "P_e", "end_prob", "pe", "end_prior"],
    "offsets": ["temporal_offsets", "prior_offsets", "clip_offsets", "offsets"],
    "clip_start_time": ["clip_start_time", "clip_start_times", "start_times", "clip_starts"],
    "clip_end_time": ["clip_end_time", "clip_end_times", "end_times", "clip_ends"],
}


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path: str) -> Dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def atomic_npz(path: str, **arrays) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = Path(str(path) + ".partial")
    with open(partial, "wb") as f:
        np.savez_compressed(f, **arrays)
    os.replace(partial, path)


def load_cache(path: str) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as payload:
        return {key: payload[key] for key in payload.files}


def require_200_rows(cache: Dict[str, np.ndarray]) -> Tuple[int, int]:
    offsets = np.asarray(cache["desc_offsets"], dtype=np.int64)
    queries = len(offsets) - 1
    rows = int(offsets[-1])
    if rows != queries * 200 or not np.all(np.diff(offsets) == 200):
        raise ValueError("C5-lite-prior requires exactly 200 contiguous rows/query")
    return queries, rows


def matrix(cache: Dict[str, np.ndarray], key: str) -> np.ndarray:
    queries, _ = require_200_rows(cache)
    return np.asarray(cache[key]).reshape(queries, 200)


def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x.astype(np.float64), -40.0, 40.0)
    return (1.0 / (1.0 + np.exp(-x))).astype(np.float32)


def zscore(values: np.ndarray, mean: float, std: float) -> np.ndarray:
    return ((values.astype(np.float32) - np.float32(mean)) / np.float32(max(float(std), EPS))).astype(np.float32)


def normalize_distribution(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    arr = np.maximum(arr, 0.0)
    total = float(arr.sum())
    if total <= EPS:
        if arr.size == 0:
            raise ValueError("Cannot normalize empty temporal prior")
        return np.full_like(arr, 1.0 / float(arr.size), dtype=np.float32)
    return (arr / total).astype(np.float32)


def _first_key(files: Sequence[str], canonical: str) -> Optional[str]:
    names = set(files)
    for key in ALIAS[canonical]:
        if key in names:
            return key
    return None


@dataclass
class TemporalPriorStore:
    """Access per-video temporal arrays from a dense or ragged NPZ export."""

    path: str
    group_id: np.ndarray
    p_ctx: np.ndarray
    p_b: np.ndarray
    p_e: np.ndarray
    offsets: Optional[np.ndarray] = None
    temporal_length: Optional[np.ndarray] = None
    clip_start_time: Optional[np.ndarray] = None
    clip_end_time: Optional[np.ndarray] = None
    dense: bool = False

    @classmethod
    def load(cls, path: str) -> "TemporalPriorStore":
        with np.load(path, allow_pickle=True) as payload:
            files = list(payload.files)
            gid_key = _first_key(files, "group_id")
            ctx_key = _first_key(files, "p_ctx")
            pb_key = _first_key(files, "p_b")
            pe_key = _first_key(files, "p_e")
            off_key = _first_key(files, "offsets")
            cst_key = _first_key(files, "clip_start_time")
            cet_key = _first_key(files, "clip_end_time")
            missing = [name for name, key in [("group_id", gid_key), ("p_ctx", ctx_key), ("p_b", pb_key), ("p_e", pe_key)] if key is None]
            if missing:
                raise KeyError(
                    "Temporal prior NPZ is missing required arrays: "
                    + ", ".join(missing)
                    + ". Do not fabricate P_ctx/P_b/P_e from row scores; run the C5-0 read-only temporal export patch."
                )
            group_id = payload[gid_key].astype(np.int64)
            p_ctx = payload[ctx_key].astype(np.float32)
            p_b = payload[pb_key].astype(np.float32)
            p_e = payload[pe_key].astype(np.float32)
            offsets = payload[off_key].astype(np.int64) if off_key is not None else None
            temporal_length = payload["temporal_length"].astype(np.int64) if "temporal_length" in files else None
            clip_start = payload[cst_key].astype(np.float32) if cst_key is not None else None
            clip_end = payload[cet_key].astype(np.float32) if cet_key is not None else None
        dense = p_ctx.ndim == 2
        if dense:
            if p_b.shape != p_ctx.shape or p_e.shape != p_ctx.shape:
                raise ValueError("Dense p_ctx/p_b/p_e shapes differ")
            if len(group_id) != p_ctx.shape[0]:
                raise ValueError("Dense temporal prior group_id length differs from number of rows")
            if offsets is not None:
                raise ValueError("Dense temporal prior should not also provide ragged offsets")
            if temporal_length is not None:
                if temporal_length.shape != (len(group_id),):
                    raise ValueError("Dense temporal_length must have one value per group")
                if np.any(temporal_length <= 0) or np.any(temporal_length > p_ctx.shape[1]):
                    raise ValueError("Dense temporal_length values must be in [1, temporal_width]")
        else:
            if offsets is None:
                raise ValueError("Ragged temporal prior requires temporal_offsets/prior_offsets")
            if len(offsets) != len(group_id) + 1:
                raise ValueError("Ragged temporal prior offsets length must equal group_count + 1")
            if p_b.ndim != 1 or p_e.ndim != 1 or p_ctx.ndim != 1:
                raise ValueError("Ragged p_ctx/p_b/p_e must be flat 1D arrays")
            if int(offsets[-1]) != len(p_ctx) or len(p_b) != len(p_ctx) or len(p_e) != len(p_ctx):
                raise ValueError("Ragged temporal prior flat lengths/offsets mismatch")
        return cls(
            path=str(path), group_id=group_id, p_ctx=p_ctx, p_b=p_b, p_e=p_e,
            offsets=offsets, temporal_length=temporal_length,
            clip_start_time=clip_start, clip_end_time=clip_end, dense=dense,
        )

    def group_count(self) -> int:
        return int(len(self.group_id))

    def arrays_for_pos(self, pos: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        if self.dense:
            length = int(self.temporal_length[pos]) if self.temporal_length is not None else int(self.p_ctx.shape[1])
            ctx = self.p_ctx[pos, :length]
            pb = self.p_b[pos, :length]
            pe = self.p_e[pos, :length]
            cst = self.clip_start_time[pos, :length] if self.clip_start_time is not None and self.clip_start_time.ndim == 2 else None
            cet = self.clip_end_time[pos, :length] if self.clip_end_time is not None and self.clip_end_time.ndim == 2 else None
            return ctx, pb, pe, cst, cet
        s, e = int(self.offsets[pos]), int(self.offsets[pos + 1])
        cst = self.clip_start_time[s:e] if self.clip_start_time is not None and self.clip_start_time.ndim == 1 else None
        cet = self.clip_end_time[s:e] if self.clip_end_time is not None and self.clip_end_time.ndim == 1 else None
        return self.p_ctx[s:e], self.p_b[s:e], self.p_e[s:e], cst, cet


def validate_temporal_store(cache: Dict[str, np.ndarray], store: TemporalPriorStore) -> Dict[str, object]:
    if "group_ids_sorted_unique" not in cache:
        raise KeyError("C4 cache lacks group_ids_sorted_unique")
    expected = np.asarray(cache["group_ids_sorted_unique"], dtype=np.int64)
    if len(expected) != store.group_count():
        raise ValueError(f"Temporal prior group count {store.group_count()} != cache group count {len(expected)}")
    if not np.array_equal(expected, store.group_id.astype(np.int64)):
        raise ValueError("Temporal prior group_id order does not match C4 cache group_ids_sorted_unique")
    lengths = []
    bad = 0
    negative = 0
    zero_sum = 0
    normalization_required = 0
    sum_min = {"p_ctx": float("inf"), "p_b": float("inf"), "p_e": float("inf")}
    sum_max = {"p_ctx": float("-inf"), "p_b": float("-inf"), "p_e": float("-inf")}
    for pos in range(store.group_count()):
        ctx, pb, pe, cst, cet = store.arrays_for_pos(pos)
        lengths.append(int(len(ctx)))
        bad += int(len(ctx) == 0 or len(pb) != len(ctx) or len(pe) != len(ctx))
        bad += int((~np.isfinite(ctx)).sum() + (~np.isfinite(pb)).sum() + (~np.isfinite(pe)).sum())
        negative_here = int((ctx < 0).sum() + (pb < 0).sum() + (pe < 0).sum())
        negative += negative_here
        bad += negative_here
        for name, values in (("p_ctx", ctx), ("p_b", pb), ("p_e", pe)):
            total = float(np.sum(values, dtype=np.float64))
            sum_min[name] = min(sum_min[name], total)
            sum_max[name] = max(sum_max[name], total)
            if total <= EPS:
                zero_sum += 1
                bad += 1
            elif not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-4):
                normalization_required += 1
        if cst is not None or cet is not None:
            if cst is None or cet is None or len(cst) != len(ctx) or len(cet) != len(ctx):
                bad += 1
            else:
                bad += int((~np.isfinite(cst)).sum() + (~np.isfinite(cet)).sum())
    return {
        "group_count": int(store.group_count()),
        "dense": bool(store.dense),
        "min_temporal_len": int(min(lengths) if lengths else 0),
        "max_temporal_len": int(max(lengths) if lengths else 0),
        "mean_temporal_len": float(np.mean(lengths) if lengths else 0.0),
        "bad_group_or_nonfinite_count": int(bad),
        "negative_value_count": int(negative),
        "zero_sum_distribution_count": int(zero_sum),
        "normalization_required_count": int(normalization_required),
        "sum_ranges": {
            name: {"min": float(sum_min[name]), "max": float(sum_max[name])}
            for name in sum_min if math.isfinite(sum_min[name])
        },
    }


def load_c4_final_score(
    cache: Dict[str, np.ndarray],
    c4_final_scores_npz: str,
    *,
    score_key: str = "s_v21",
) -> Tuple[np.ndarray, Dict[str, object]]:
    if not c4_final_scores_npz:
        raise ValueError("C5-lite-prior requires C4_final scores from C4-r2-cal-v2.1; do not fall back to C4-lite.")
    with np.load(c4_final_scores_npz, allow_pickle=True) as payload:
        if score_key not in payload.files:
            raise KeyError(f"{score_key} not found in {c4_final_scores_npz}; keys={payload.files}")
        score = payload[score_key].astype(np.float32)
        if score.shape[0] != cache["row_group_id"].shape[0]:
            raise ValueError("C4 final score length does not match cache rows")
        if "row_group_id" in payload.files and not np.array_equal(
            payload["row_group_id"].astype(np.int64), cache["row_group_id"].astype(np.int64)
        ):
            raise ValueError("C4 final score row_group_id does not match cache")
    if not np.all(np.isfinite(score)):
        raise ValueError("Non-finite C4_final scores")
    return score, {"path": c4_final_scores_npz, "score_key": score_key, "sha256": sha256(c4_final_scores_npz)}


def group_max(values: np.ndarray, row_group_id: np.ndarray, group_count: int) -> np.ndarray:
    out = np.full(group_count, -np.inf, dtype=np.float32)
    np.maximum.at(out, row_group_id.astype(np.int64), values.astype(np.float32))
    return out


def group_mean(values: np.ndarray, row_group_id: np.ndarray, group_count: int) -> np.ndarray:
    sums = np.bincount(row_group_id.astype(np.int64), weights=values.astype(np.float64), minlength=group_count)
    counts = np.bincount(row_group_id.astype(np.int64), minlength=group_count)
    return (sums / np.maximum(counts, 1)).astype(np.float32)


def querywise_group_z(group_values: np.ndarray, group_query: np.ndarray) -> np.ndarray:
    q_count = int(np.max(group_query)) + 1
    sums = np.bincount(group_query.astype(np.int64), weights=group_values.astype(np.float64), minlength=q_count)
    counts = np.bincount(group_query.astype(np.int64), minlength=q_count)
    means = sums / np.maximum(counts, 1)
    sq = np.bincount(group_query.astype(np.int64), weights=(group_values.astype(np.float64) ** 2), minlength=q_count) / np.maximum(counts, 1)
    stds = np.sqrt(np.maximum(sq - means * means, 0.0))
    return ((group_values - means[group_query]) / np.maximum(stds[group_query], EPS)).astype(np.float32)


def querywise_group_rank_inv(group_values: np.ndarray, group_query: np.ndarray) -> np.ndarray:
    out = np.zeros_like(group_values, dtype=np.float32)
    for q in np.unique(group_query.astype(np.int64)):
        idx = np.flatnonzero(group_query == q)
        order = idx[np.argsort(-group_values[idx], kind="stable")]
        out[order] = 1.0 / (np.arange(len(order), dtype=np.float32) + 1.0)
    return out


def querywise_group_margin(group_values: np.ndarray, group_query: np.ndarray) -> np.ndarray:
    out = np.zeros_like(group_values, dtype=np.float32)
    for q in np.unique(group_query.astype(np.int64)):
        idx = np.flatnonzero(group_query == q)
        vals = group_values[idx]
        order = np.argsort(-vals, kind="stable")
        if len(order) <= 1:
            out[idx] = 0.0
            continue
        top1, top2 = float(vals[order[0]]), float(vals[order[1]])
        local = vals - top1
        local[order[0]] = top1 - top2
        out[idx] = local.astype(np.float32)
    return out


def build_video_retrieval_confidence(cache: Dict[str, np.ndarray], baseline_score: np.ndarray) -> Dict[str, np.ndarray]:
    row_gid = cache["row_group_id"].astype(np.int64)
    group_count = int(np.max(row_gid)) + 1
    video_group_keys = cache.get("video_group_keys")
    if video_group_keys is None or len(video_group_keys) < group_count:
        raise ValueError("cache lacks video_group_keys required by C5-lite-prior")
    group_query = video_group_keys[:group_count, 0].astype(np.int64)
    group_score = group_max(baseline_score, row_gid, group_count)
    r_z_group = querywise_group_z(group_score, group_query)
    r_sig_group = sigmoid(r_z_group)
    margin_group = querywise_group_margin(group_score, group_query)
    margin_z_group = querywise_group_z(margin_group, group_query)
    rank_inv_group = querywise_group_rank_inv(group_score, group_query)
    return {
        "group_score": group_score.astype(np.float32),
        "group_query": group_query.astype(np.int32),
        "r_video_z_group": r_z_group.astype(np.float32),
        "r_video_sig_group": r_sig_group.astype(np.float32),
        "margin_z_group": margin_z_group.astype(np.float32),
        "rank_inv_group": rank_inv_group.astype(np.float32),
        "r_video_z_row": r_z_group[row_gid].astype(np.float32),
        "r_video_sig_row": r_sig_group[row_gid].astype(np.float32),
        "margin_z_row": margin_z_group[row_gid].astype(np.float32),
        "rank_inv_row": rank_inv_group[row_gid].astype(np.float32),
    }


def span_clip_indices(
    start_time: float,
    end_time: float,
    temporal_len: int,
    *,
    clip_start: Optional[np.ndarray],
    clip_end: Optional[np.ndarray],
    clip_length: float = DEFAULT_CLIP_LENGTH,
) -> np.ndarray:
    if temporal_len <= 0:
        raise ValueError("Empty temporal prior")
    if clip_start is not None and clip_end is not None:
        cs = np.asarray(clip_start, dtype=np.float32)
        ce = np.asarray(clip_end, dtype=np.float32)
        if len(cs) != temporal_len or len(ce) != temporal_len:
            raise ValueError("clip time length differs from temporal prior length")
        # Select clips overlapping the candidate span.  If none overlap, fall back
        # to midpoint containment to avoid empty aggregation from boundary rounding.
        mask = (ce > float(start_time)) & (cs < float(end_time))
        idx = np.flatnonzero(mask)
        if idx.size:
            return idx.astype(np.int64)
        mid = 0.5 * (cs + ce)
        idx = np.flatnonzero((mid >= float(start_time)) & (mid <= float(end_time)))
        if idx.size:
            return idx.astype(np.int64)
    # Fallback: TVR/CONQUER normally uses 1.5s clips; caller must pass the exact
    # value from dataset config if different.
    st = int(math.floor(max(float(start_time), 0.0) / float(clip_length)))
    ed = int(math.ceil(max(float(end_time), float(start_time) + 1e-6) / float(clip_length))) - 1
    st = max(0, min(st, temporal_len - 1))
    ed = max(st, min(ed, temporal_len - 1))
    return np.arange(st, ed + 1, dtype=np.int64)


def boundary_indices(indices: np.ndarray) -> Tuple[int, int]:
    if len(indices) == 0:
        return 0, 0
    return int(indices[0]), int(indices[-1])


def center_proximity(indices: np.ndarray, peak_idx: int, temporal_len: int) -> float:
    if len(indices) == 0:
        return 0.0
    center = 0.5 * (float(indices[0]) + float(indices[-1]))
    dist = abs(center - float(peak_idx)) / max(float(temporal_len - 1), 1.0)
    return float(1.0 - min(dist, 1.0))


def aggregate_one_prior(prior: np.ndarray, indices: np.ndarray) -> Tuple[float, float, float, float, float, float]:
    values = prior[indices]
    mass = float(np.sum(values))
    mean = float(np.mean(values)) if len(values) else 0.0
    maxv = float(np.max(values)) if len(values) else 0.0
    peak = int(np.argmax(prior))
    peak_inside = float(peak in set(int(i) for i in indices.tolist()))
    bi, bj = boundary_indices(indices)
    boundary_agree = float(0.5 * (prior[bi] + prior[bj]))
    prox = center_proximity(indices, peak, len(prior))
    return mass, mean, maxv, peak_inside, boundary_agree, prox


def _normalize_dense_rows(values: np.ndarray, chunk_groups: int = 100_000) -> np.ndarray:
    """Normalize dense temporal evidence without a full-size float64 temporary."""
    out = np.empty(values.shape, dtype=np.float32)
    width = values.shape[1]
    for start in range(0, len(values), chunk_groups):
        end = min(start + chunk_groups, len(values))
        block = np.nan_to_num(values[start:end].astype(np.float32, copy=True), nan=0.0, posinf=0.0, neginf=0.0)
        np.maximum(block, 0.0, out=block)
        sums = block.sum(axis=1, keepdims=True, dtype=np.float32)
        zero = sums[:, 0] <= EPS
        block /= np.maximum(sums, np.float32(EPS))
        if np.any(zero):
            block[zero] = np.float32(1.0 / width)
        out[start:end] = block
    return out


def _aggregate_dense_torch(prior, row_gid, start_idx, end_idx, positions):
    values = prior[row_gid]
    mask = (positions >= start_idx[:, None]) & (positions <= end_idx[:, None])
    masked = values * mask
    mass = masked.sum(dim=1)
    length = (end_idx - start_idx + 1).to(torch.float32)
    mean = mass / length
    maxv = values.masked_fill(~mask, -torch.inf).max(dim=1).values
    peak = values.argmax(dim=1)
    peak_inside = ((peak >= start_idx) & (peak <= end_idx)).to(torch.float32)
    row = torch.arange(len(row_gid), device=values.device)
    boundary = 0.5 * (values[row, start_idx] + values[row, end_idx])
    center = 0.5 * (start_idx.to(torch.float32) + end_idx.to(torch.float32))
    proximity = 1.0 - torch.clamp(torch.abs(center - peak.to(torch.float32)) / max(values.shape[1] - 1, 1), max=1.0)
    return mass, mean, maxv, peak_inside, boundary, proximity


def _build_dense_c5_prior_features(
    cache: Dict[str, np.ndarray],
    temporal: TemporalPriorStore,
    baseline_score: np.ndarray,
    *,
    clip_length: float,
    a_ctx: float,
    a_bd: float,
    a_ctx_ret: float,
    a_bd_ret: float,
    device: str,
    row_chunk_size: int,
) -> Dict[str, np.ndarray]:
    """Exact chunked GPU/CPU aggregation for the dense export used in C5-0."""
    rows = len(cache["row_group_id"])
    group_count, temporal_len = temporal.p_ctx.shape
    conf = build_video_retrieval_confidence(cache, baseline_score)
    p_ctx = _normalize_dense_rows(temporal.p_ctx)
    p_b = _normalize_dense_rows(temporal.p_b)
    p_e = _normalize_dense_rows(temporal.p_e)
    p_bd = _normalize_dense_rows(0.5 * p_b + 0.5 * p_e)
    del p_b, p_e
    r_sig_group = conf["r_video_sig_group"].astype(np.float32)
    # High retrieval confidence increases semantic-context weight; low
    # confidence increases boundary-prior weight. Unlike scalar multiplication
    # followed by normalization, this changes the temporal mixture itself.
    p_retloc = np.empty_like(p_ctx)
    for start in range(0, group_count, 100_000):
        end = min(start + 100_000, group_count)
        r = r_sig_group[start:end, None]
        mixed = (
            (np.float32(a_ctx) + np.float32(a_ctx_ret) * r) * p_ctx[start:end]
            + (np.float32(a_bd) + np.float32(a_bd_ret) * (1.0 - r)) * p_bd[start:end]
        )
        p_retloc[start:end] = _normalize_dense_rows(mixed, chunk_groups=len(mixed))

    out = {key: np.empty(rows, dtype=np.float32) for key in C5_PRIOR_RAW_FEATURES}
    terms = {key: np.empty(rows, dtype=np.float32) for key in C5_PRIOR_TERM_NAMES}
    torch_device = torch.device("cuda" if device == "cuda" and torch.cuda.is_available() else "cpu")
    p_ctx_t = torch.from_numpy(p_ctx).to(torch_device)
    p_bd_t = torch.from_numpy(p_bd).to(torch_device)
    p_retloc_t = torch.from_numpy(p_retloc).to(torch_device)
    positions = torch.arange(temporal_len, device=torch_device)
    row_gid_all = cache["row_group_id"].astype(np.int64)
    starts_all = np.floor(np.maximum(cache["start_time"].astype(np.float64), 0.0) / float(clip_length)).astype(np.int64)
    ends_all = (
        np.ceil(
            np.maximum(cache["end_time"].astype(np.float64), cache["start_time"].astype(np.float64) + 1e-6)
            / float(clip_length)
        ).astype(np.int64) - 1
    )
    np.clip(starts_all, 0, temporal_len - 1, out=starts_all)
    ends_all = np.maximum(ends_all, starts_all)
    np.clip(ends_all, 0, temporal_len - 1, out=ends_all)

    for start in range(0, rows, row_chunk_size):
        end = min(start + row_chunk_size, rows)
        gid = torch.from_numpy(row_gid_all[start:end]).to(torch_device)
        st = torch.from_numpy(starts_all[start:end]).to(torch_device)
        ed = torch.from_numpy(ends_all[start:end]).to(torch_device)
        ctx = _aggregate_dense_torch(p_ctx_t, gid, st, ed, positions)
        bd = _aggregate_dense_torch(p_bd_t, gid, st, ed, positions)
        retloc = _aggregate_dense_torch(p_retloc_t, gid, st, ed, positions)
        r_sig = torch.from_numpy(conf["r_video_sig_row"][start:end]).to(torch_device)

        def put(name, value):
            out[name][start:end] = value.detach().cpu().numpy().astype(np.float32, copy=False)

        for prefix, values in (("ctx", ctx), ("bd", bd)):
            for suffix, value in zip(("mass", "mean", "max", "peak_inside", "boundary_agree", "center_proximity"), values):
                put(f"{prefix}_{suffix}", value)
        for suffix, value in zip(("mass", "mean", "max", "peak_inside"), retloc[:4]):
            put(f"retloc_{suffix}", value)
        put("retctx_mass", r_sig * ctx[0])
        put("retbd_mass", r_sig * bd[0])
        for name in ("r_video_z", "r_video_sig", "margin_z", "rank_inv"):
            out[name][start:end] = conf[f"{name}_row"][start:end]

        term_values = {
            "retctx_mass": r_sig * ctx[0],
            "retbd_mass": r_sig * bd[0],
            "retctx_peak_inside": r_sig * ctx[3],
            "retbd_peak_inside": r_sig * bd[3],
            "retctx_boundary_agree": r_sig * ctx[4],
            "retbd_boundary_agree": r_sig * bd[4],
            "retctx_center_proximity": r_sig * ctx[5],
            "retbd_center_proximity": r_sig * bd[5],
        }
        low = 1.0 - r_sig
        efp = torch.from_numpy(cache["e_fp"][start:end].astype(np.float32)).to(torch_device)
        qb = torch.from_numpy(cache["q_bd"][start:end].astype(np.float32)).to(torch_device)
        margin = torch.from_numpy(conf["margin_z_row"][start:end]).to(torch_device)
        rank_inv = torch.from_numpy(conf["rank_inv_row"][start:end]).to(torch_device)
        term_values.update({
            "low_conf_efp_neg": -low * efp,
            "low_conf_unc_neg": -low * (1.0 - qb),
            "margin_z_x_retbd_mass": margin * r_sig * bd[0],
            "rank_inv_x_retbd_mass": rank_inv * r_sig * bd[0],
        })
        for name, value in term_values.items():
            terms[name][start:end] = value.detach().cpu().numpy().astype(np.float32, copy=False)

    for key, arr in {**out, **terms}.items():
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"Non-finite C5 prior feature: {key}")
    return {**out, **{f"term_{key}": value for key, value in terms.items()}}


def build_c5_prior_features(
    cache: Dict[str, np.ndarray],
    temporal: TemporalPriorStore,
    baseline_score: np.ndarray,
    *,
    clip_length: float = DEFAULT_CLIP_LENGTH,
    a_ctx: float = 1.0,
    a_bd: float = 1.0,
    a_ctx_ret: float = 1.0,
    a_bd_ret: float = 1.0,
    device: str = "cuda",
    row_chunk_size: int = 262_144,
) -> Dict[str, np.ndarray]:
    """Aggregate per-video temporal priors to candidate-span features."""
    validate_temporal_store(cache, temporal)
    require_200_rows(cache)
    if temporal.dense and temporal.clip_start_time is None and temporal.clip_end_time is None:
        return _build_dense_c5_prior_features(
            cache, temporal, baseline_score,
            clip_length=clip_length,
            a_ctx=a_ctx, a_bd=a_bd,
            a_ctx_ret=a_ctx_ret, a_bd_ret=a_bd_ret,
            device=device, row_chunk_size=row_chunk_size,
        )
    rows = len(cache["row_group_id"])
    group_count = temporal.group_count()
    conf = build_video_retrieval_confidence(cache, baseline_score)
    row_gid = cache["row_group_id"].astype(np.int64)
    # Allocate raw features and final search terms.
    out = {key: np.zeros(rows, dtype=np.float32) for key in C5_PRIOR_RAW_FEATURES}
    terms = {key: np.zeros(rows, dtype=np.float32) for key in C5_PRIOR_TERM_NAMES}

    group_offsets = np.asarray(cache["group_offsets"], dtype=np.int64)
    group_sort_idx = np.asarray(cache["group_sort_idx"], dtype=np.int64)
    # group_offsets does not include a final sentinel in C4 cache; construct one.
    offsets = np.r_[group_offsets, len(group_sort_idx)]
    for pos in range(group_count):
        gid = int(temporal.group_id[pos])
        if pos >= len(offsets) - 1:
            raise ValueError("group_offsets shorter than temporal groups")
        row_indices = group_sort_idx[int(offsets[pos]):int(offsets[pos + 1])]
        if row_indices.size == 0:
            continue
        ctx_raw, pb_raw, pe_raw, clip_start, clip_end = temporal.arrays_for_pos(pos)
        p_ctx = normalize_distribution(ctx_raw)
        p_b = normalize_distribution(pb_raw)
        p_e = normalize_distribution(pe_raw)
        p_bd = normalize_distribution(0.5 * p_b + 0.5 * p_e)
        r_sig = float(conf["r_video_sig_group"][gid])
        r_z = float(conf["r_video_z_group"][gid])
        margin_z = float(conf["margin_z_group"][gid])
        rank_inv = float(conf["rank_inv_group"][gid])
        # Read-only retrieval-conditioned temporal prior.  This is not injected
        # into CONQUER; it is aggregated to candidate span features below.
        p_retloc = normalize_distribution(
            np.float32(a_ctx + a_ctx_ret * r_sig) * p_ctx
            + np.float32(a_bd + a_bd_ret * (1.0 - r_sig)) * p_bd
        )
        # Keep these as explicit retrieval x temporal interactions. Normalizing
        # a scalar multiple would cancel retrieval confidence completely.
        p_retctx = np.float32(r_sig) * p_ctx
        p_retbd = np.float32(r_sig) * p_bd
        for ri in row_indices:
            st = float(cache["start_time"][ri])
            en = float(cache["end_time"][ri])
            idx = span_clip_indices(st, en, len(p_ctx), clip_start=clip_start, clip_end=clip_end, clip_length=clip_length)
            ctx = aggregate_one_prior(p_ctx, idx)
            bd = aggregate_one_prior(p_bd, idx)
            retloc = aggregate_one_prior(p_retloc, idx)
            retctx = aggregate_one_prior(p_retctx, idx)
            retbd = aggregate_one_prior(p_retbd, idx)
            out["ctx_mass"][ri], out["ctx_mean"][ri], out["ctx_max"][ri], out["ctx_peak_inside"][ri], out["ctx_boundary_agree"][ri], out["ctx_center_proximity"][ri] = ctx
            out["bd_mass"][ri], out["bd_mean"][ri], out["bd_max"][ri], out["bd_peak_inside"][ri], out["bd_boundary_agree"][ri], out["bd_center_proximity"][ri] = bd
            out["retloc_mass"][ri], out["retloc_mean"][ri], out["retloc_max"][ri], out["retloc_peak_inside"][ri], _, _ = retloc
            out["retctx_mass"][ri] = retctx[0]
            out["retbd_mass"][ri] = retbd[0]
            out["r_video_z"][ri] = r_z
            out["r_video_sig"][ri] = r_sig
            out["margin_z"][ri] = margin_z
            out["rank_inv"][ri] = rank_inv
            qb = float(cache["q_bd"][ri])
            efp = float(cache["e_fp"][ri])
            low = 1.0 - r_sig
            terms["retctx_mass"][ri] = retctx[0]
            terms["retbd_mass"][ri] = retbd[0]
            terms["retctx_peak_inside"][ri] = r_sig * ctx[3]
            terms["retbd_peak_inside"][ri] = r_sig * bd[3]
            terms["retctx_boundary_agree"][ri] = retctx[4]
            terms["retbd_boundary_agree"][ri] = retbd[4]
            terms["retctx_center_proximity"][ri] = r_sig * ctx[5]
            terms["retbd_center_proximity"][ri] = r_sig * bd[5]
            terms["low_conf_efp_neg"][ri] = -low * efp
            terms["low_conf_unc_neg"][ri] = -low * (1.0 - qb)
            terms["margin_z_x_retbd_mass"][ri] = margin_z * retbd[0]
            terms["rank_inv_x_retbd_mass"][ri] = rank_inv * retbd[0]
    for key, arr in {**out, **terms}.items():
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"Non-finite C5 prior feature: {key}")
    return {**out, **{f"term_{k}": v for k, v in terms.items()}}


def fit_term_stats(feature_payload: Dict[str, np.ndarray]) -> Dict[str, Dict[str, float]]:
    stats = {}
    for name in C5_PRIOR_TERM_NAMES:
        key = f"term_{name}"
        values = np.asarray(feature_payload[key], dtype=np.float64)
        stats[name] = {"mean": float(np.mean(values)), "std": float(max(np.std(values), EPS))}
    return stats


def standardized_term_matrix(feature_payload: Dict[str, np.ndarray], stats: Dict[str, Dict[str, float]]) -> np.ndarray:
    cols = []
    for name in C5_PRIOR_TERM_NAMES:
        key = f"term_{name}"
        if name not in stats:
            raise KeyError(f"Missing C5 prior term stat: {name}")
        cols.append(zscore(feature_payload[key], stats[name]["mean"], stats[name]["std"]))
    return np.column_stack(cols).astype(np.float32)


def score_with_config(baseline_score: np.ndarray, z_terms: np.ndarray, cfg: Dict[str, float]) -> np.ndarray:
    weights = np.asarray([float(cfg.get(f"w_{name}", 0.0)) for name in C5_PRIOR_TERM_NAMES], dtype=np.float32)
    residual = z_terms @ weights
    if bool(cfg.get("center_residual", False)):
        if "residual_center_frozen" not in cfg:
            raise ValueError("Residual centering requires an explicitly frozen train-only constant")
        residual = residual - np.float32(cfg["residual_center_frozen"])
    scale = float(cfg.get("lambda_scale", 1.0))
    return (baseline_score.astype(np.float32) + np.float32(scale) * residual.astype(np.float32)).astype(np.float32)


def groups_for_submission(cache: Dict[str, np.ndarray], score: np.ndarray, score_key: str = "s_c5") -> Iterator[List[Dict]]:
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
                score_key: float(score[i]),
            }
            for i in range(start, end)
        ]


def attach_eval_labels(cache: Dict[str, np.ndarray], gt_jsonl: str) -> None:
    gt_by_id = {int(row["desc_id"]): row for row in load_jsonl(gt_jsonl)}
    desc_ids = [int(value) for value in cache["desc_ids"]]
    if set(desc_ids) != set(gt_by_id):
        raise ValueError("C5 cache/GT desc_id sets differ")
    gt_ts = np.asarray([gt_by_id[desc_id]["ts"] for desc_id in desc_ids], dtype=np.float32)
    starts = matrix(cache, "start_time").astype(np.float32)
    ends = matrix(cache, "end_time").astype(np.float32)
    gt_start = gt_ts[:, 0, None]
    gt_end = gt_ts[:, 1, None]
    inter = np.maximum(0.0, np.minimum(ends, gt_end) - np.maximum(starts, gt_start))
    union = np.maximum(ends, gt_end) - np.minimum(starts, gt_start)
    temporal_iou = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
    is_gt = matrix(cache, "is_gt_video") > 0.5
    cache["eval_y05"] = is_gt & (temporal_iou >= 0.5)
    cache["eval_y07"] = is_gt & (temporal_iou >= 0.7)
    cache["eval_iou"] = np.where(is_gt, temporal_iou, 0.0).astype(np.float32)


def eval_score(cache: Dict[str, np.ndarray], score: np.ndarray, dataset_config: str, split: str, effective_top_n: int, max_after_nms: int, nms_thd: float, gt_jsonl: str, *, no_desc_type: bool = False) -> Tuple[Dict, Dict, Dict[str, float]]:
    submission = submission_from_groups(
        groups_for_submission(cache, score),
        "s_c5",
        dataset_config,
        split,
        effective_top_n=effective_top_n,
        max_after_nms=max_after_nms,
        nms_thd=nms_thd,
    )
    submission.pop("_row_counts", None)
    metrics = eval_retrieval(
        submission,
        load_jsonl(gt_jsonl),
        iou_thds=(0.5, 0.7),
        verbose=False,
        match_number=True,
        use_desc_type=not no_desc_type,
    )
    return submission, metrics, flatten_eval_metrics(metrics)


def fast_selected(cache: Dict[str, np.ndarray], score: np.ndarray, effective_top_n: int, max_after_nms: int, nms_thd: float):
    scores = np.asarray(score, dtype=np.float32).reshape(-1, 200)
    order = np.argsort(-scores, axis=1, kind="stable")
    video_idx = matrix(cache, "video_idx")
    starts = matrix(cache, "start_time")
    ends = matrix(cache, "end_time")
    selected = [
        greedy_nms_indices(
            order[q, :effective_top_n], scores[q], video_idx[q],
            starts[q], ends[q], nms_thd, max_after_nms,
        )
        for q in range(scores.shape[0])
    ]
    return scores, order, selected


def fast_metrics(cache: Dict[str, np.ndarray], score: np.ndarray, effective_top_n: int, max_after_nms: int, nms_thd: float) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
    scores, order, selected = fast_selected(cache, score, effective_top_n, max_after_nms, nms_thd)
    raw = raw_metrics_from_selected(selected, cache["eval_y05"], cache["eval_y07"])
    return {key: float(value) for key, value in raw.items()}, scores, order


def movement_diagnostics(cache: Dict[str, np.ndarray], base_score: np.ndarray, new_score: np.ndarray) -> Dict[str, float]:
    queries = len(cache["desc_offsets"]) - 1
    base = base_score.reshape(queries, 200)
    new = new_score.reshape(queries, 200)
    base_order = np.argsort(-base, axis=1, kind="stable")
    new_order = np.argsort(-new, axis=1, kind="stable")
    ranks = np.broadcast_to(np.arange(200, dtype=np.int16), base_order.shape)
    base_inv = np.empty_like(base_order, dtype=np.int16)
    new_inv = np.empty_like(new_order, dtype=np.int16)
    np.put_along_axis(base_inv, base_order, ranks, axis=1)
    np.put_along_axis(new_inv, new_order, ranks, axis=1)
    hard = np.asarray(cache.get("eval_y05", (matrix(cache, "is_gt_video") > 0.5) & (matrix(cache, "iou") >= 0.5)), dtype=bool)
    hard_total = int(hard.sum())
    diff = base_inv.astype(np.float64) - new_inv.astype(np.float64)
    spearman = 1.0 - 6.0 * np.sum(diff * diff, axis=1) / (200.0 * (200.0 * 200.0 - 1.0))
    pearson = float(np.corrcoef(base_score.astype(np.float64), new_score.astype(np.float64))[0, 1])
    return {
        "queries": int(queries),
        "top1_changed_ratio": float(np.mean(base_order[:, 0] != new_order[:, 0])),
        "hard_positive_top100_exits": int(np.sum(hard & (base_inv < 100) & (new_inv >= 100))),
        "hard_positive_top100_entries": int(np.sum(hard & (base_inv >= 100) & (new_inv < 100))),
        "hard_positive_top100_exit_ratio": float(np.sum(hard & (base_inv < 100) & (new_inv >= 100)) / max(hard_total, 1)),
        "pearson_base_candidate": pearson,
        "mean_within_query_spearman": float(np.mean(spearman)),
    }


def localization_diagnostics(cache: Dict[str, np.ndarray], base_score: np.ndarray, new_score: np.ndarray) -> Dict[str, float]:
    offsets = cache["desc_offsets"].astype(np.int64)
    gid_all = cache["row_group_id"].astype(np.int64)
    is_gt_all = cache["is_gt_video"].astype(np.float32) > 0.5
    iou_all = np.asarray(cache.get("eval_iou", cache["iou"]), dtype=np.float32).reshape(-1)
    n = len(offsets) - 1
    oracle05_base = oracle05_new = 0
    oracle07_base = oracle07_new = 0
    miou_base, miou_new = [], []
    best_rank_base, best_rank_new = [], []
    pos_up = pos_down = pos_total = 0
    neg_down = neg_up = neg_total = 0
    affected_same_video_queries = 0
    for q in range(n):
        s, e = int(offsets[q]), int(offsets[q + 1])
        local = np.arange(s, e)
        gt_local = local[is_gt_all[s:e]]
        if gt_local.size == 0:
            continue
        best_global = int(gt_local[np.argmax(iou_all[gt_local])])
        gt_group = int(gid_all[best_global])
        same = local[gid_all[s:e] == gt_group]
        if same.size == 0:
            continue
        b_scores = base_score[same]
        n_scores = new_score[same]
        b_order = np.argsort(-b_scores, kind="stable")
        n_order = np.argsort(-n_scores, kind="stable")
        b_top = int(same[b_order[0]])
        n_top = int(same[n_order[0]])
        biou = float(max(iou_all[b_top], 0.0))
        niou = float(max(iou_all[n_top], 0.0))
        miou_base.append(biou)
        miou_new.append(niou)
        oracle05_base += int(biou >= 0.5)
        oracle05_new += int(niou >= 0.5)
        oracle07_base += int(biou >= 0.7)
        oracle07_new += int(niou >= 0.7)
        base_inv = np.empty(len(same), dtype=np.int16)
        new_inv = np.empty(len(same), dtype=np.int16)
        ranks = np.arange(len(same), dtype=np.int16)
        base_inv[b_order] = ranks
        new_inv[n_order] = ranks
        best_local_pos = int(np.where(same == best_global)[0][0])
        best_rank_base.append(int(base_inv[best_local_pos]) + 1)
        best_rank_new.append(int(new_inv[best_local_pos]) + 1)
        if not np.array_equal(base_inv, new_inv):
            affected_same_video_queries += 1
        pos_mask = iou_all[same] >= 0.5
        neg_mask = iou_all[same] < 0.3
        if np.any(pos_mask):
            pos_total += int(np.sum(pos_mask))
            pos_up += int(np.sum(new_inv[pos_mask] < base_inv[pos_mask]))
            pos_down += int(np.sum(new_inv[pos_mask] > base_inv[pos_mask]))
        if np.any(neg_mask):
            hard_neg = neg_mask & (base_inv <= 10)
            neg_total += int(np.sum(hard_neg))
            neg_down += int(np.sum(new_inv[hard_neg] > base_inv[hard_neg]))
            neg_up += int(np.sum(new_inv[hard_neg] < base_inv[hard_neg]))
    denom = max(len(miou_base), 1)
    base_r05 = 100.0 * oracle05_base / denom
    new_r05 = 100.0 * oracle05_new / denom
    base_r07 = 100.0 * oracle07_base / denom
    new_r07 = 100.0 * oracle07_new / denom
    b_rank = np.asarray(best_rank_base, dtype=np.float64) if best_rank_base else np.asarray([np.nan])
    n_rank = np.asarray(best_rank_new, dtype=np.float64) if best_rank_new else np.asarray([np.nan])
    return {
        "gt_video_queries": int(denom),
        "affected_same_video_query_ratio": float(affected_same_video_queries / denom),
        "oracle_video_r1_05_base": float(base_r05),
        "oracle_video_r1_05_new": float(new_r05),
        "oracle_video_r1_05_delta": float(new_r05 - base_r05),
        "oracle_video_r1_07_base": float(base_r07),
        "oracle_video_r1_07_new": float(new_r07),
        "oracle_video_r1_07_delta": float(new_r07 - base_r07),
        "selected_span_miou_base": float(np.mean(miou_base)) if miou_base else float("nan"),
        "selected_span_miou_new": float(np.mean(miou_new)) if miou_new else float("nan"),
        "selected_span_miou_delta": float(np.mean(miou_new) - np.mean(miou_base)) if miou_base else float("nan"),
        "best_iou_span_rank_base_mean": float(np.nanmean(b_rank)),
        "best_iou_span_rank_new_mean": float(np.nanmean(n_rank)),
        "best_iou_span_rank_delta_mean": float(np.nanmean(n_rank - b_rank)),
        "positive_span_upward_ratio": float(pos_up / max(pos_total, 1)),
        "positive_span_downward_ratio": float(pos_down / max(pos_total, 1)),
        "hard_negative_downward_ratio": float(neg_down / max(neg_total, 1)),
        "hard_negative_upward_ratio": float(neg_up / max(neg_total, 1)),
        "positive_span_count": int(pos_total),
        "hard_negative_count": int(neg_total),
    }


def prior_diagnostics(cache: Dict[str, np.ndarray], features: Dict[str, np.ndarray], base_score: np.ndarray, new_score: np.ndarray) -> Dict[str, float]:
    queries = len(cache["desc_offsets"]) - 1
    new = new_score.reshape(queries, 200)
    order = np.argsort(-new, axis=1, kind="stable")
    peak_inside = np.asarray(features.get("retloc_peak_inside", np.zeros_like(new_score)), dtype=np.float32).reshape(queries, 200)
    retloc_mass = np.asarray(features.get("retloc_mass", np.zeros_like(new_score)), dtype=np.float32).reshape(queries, 200)
    top = order[:, 0]
    return {
        "selected_retloc_peak_inside_ratio": float(np.mean(peak_inside[np.arange(queries), top])),
        "selected_retloc_mass_mean": float(np.mean(retloc_mass[np.arange(queries), top])),
        "all_rows_retloc_peak_inside_ratio": float(np.mean(peak_inside)),
        "all_rows_retloc_mass_mean": float(np.mean(retloc_mass)),
    }


def metric_deltas(candidate: Dict[str, float], baseline: Dict[str, float]) -> Dict[str, float]:
    return {k: float(candidate.get(k, float("nan")) - baseline.get(k, float("nan"))) for k in ALL_METRICS if k in candidate and k in baseline}


def promotion_flags(metrics: Dict[str, float], base_metrics: Dict[str, float], loc: Dict[str, float], move: Dict[str, float]) -> Dict[str, bool]:
    deltas = metric_deltas(metrics, base_metrics)
    six_nonneg = all(deltas.get(k, -1e9) >= -1e-9 for k in PRIMARY_METRICS)
    r100_nonneg = all(deltas.get(k, -1e9) >= -1e-9 for k in ["0.5-r100", "0.7-r100"])
    loc_positive = (
        loc.get("oracle_video_r1_05_delta", 0.0) > 0.0
        or loc.get("oracle_video_r1_07_delta", 0.0) > 0.0
        or loc.get("selected_span_miou_delta", 0.0) > 0.0
        or loc.get("best_iou_span_rank_delta_mean", 0.0) < 0.0
    )
    movement_safe = (
        move.get("hard_positive_top100_exit_ratio", 1.0) <= 0.005
        and move.get("top1_changed_ratio", 1.0) <= 0.25
        and move.get("pearson_base_candidate", 0.0) >= 0.98
    )
    return {
        "six_primary_nonnegative": bool(six_nonneg),
        "r100_both_nonnegative": bool(r100_nonneg),
        "localization_diagnostic_positive": bool(loc_positive),
        "movement_safe": bool(movement_safe),
        "promotion_candidate": bool(six_nonneg and r100_nonneg and loc_positive and movement_safe),
    }
