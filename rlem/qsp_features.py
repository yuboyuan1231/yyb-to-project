#!/usr/bin/env python

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

"""Query-specific prior (QSP) utilities for CONQUER-RLEM C3.5.

The functions in this file deliberately produce bounded scalar evidence from
CONQUER intermediate tensors.  They do not change candidate generation, NMS,
CONQUER logits, or labels.  They are intended for C3.5 fixed-candidate evidence
export only.
"""

from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F

EPS = 1e-8


QSP_FEATURES = [
    "qsp_mass_v", "qsp_mean_v", "qsp_max_v", "qsp_peak_in_v", "qsp_entropy_v",
    "qsp_mass_s", "qsp_mean_s", "qsp_max_s", "qsp_peak_in_s", "qsp_entropy_s",
    "qsp_mass_gate", "qsp_mean_gate", "qsp_max_gate", "qsp_peak_in_gate", "qsp_entropy_gate",
    "qsp_js_vs", "qsp_gate_confidence", "qsp_sub_coverage",
    "qsp_q_v", "qsp_q_s", "qsp_modality_gap", "qsp_modality_entropy",
]


def _safe_float(value, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if np.isfinite(out) else default


def _normalize_distribution(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=np.float64)
    p = np.where(np.isfinite(p), p, 0.0)
    p = np.maximum(p, 0.0)
    total = float(p.sum())
    if total <= EPS:
        if len(p) == 0:
            return p.astype(np.float32)
        return np.full_like(p, 1.0 / len(p), dtype=np.float64).astype(np.float32)
    return (p / total).astype(np.float32)


def entropy(p: np.ndarray) -> float:
    p = _normalize_distribution(p).astype(np.float64)
    return float(-(p * np.log(p + EPS)).sum())


def js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    p = _normalize_distribution(p).astype(np.float64)
    q = _normalize_distribution(q).astype(np.float64)
    m = 0.5 * (p + q)
    kl_pm = float((p * (np.log(p + EPS) - np.log(m + EPS))).sum())
    kl_qm = float((q * (np.log(q + EPS) - np.log(m + EPS))).sum())
    return max(0.0, 0.5 * (kl_pm + kl_qm))


def span_prior_stats(prior: Optional[np.ndarray], start_idx: int, end_idx: int) -> Dict[str, float]:
    if prior is None:
        return {"mass": 0.0, "mean": 0.0, "max": 0.0, "peak_in_span": 0.0, "entropy": 0.0}
    p = _normalize_distribution(prior)
    if len(p) == 0:
        return {"mass": 0.0, "mean": 0.0, "max": 0.0, "peak_in_span": 0.0, "entropy": 0.0}
    i = max(0, int(start_idx))
    j = min(len(p) - 1, int(end_idx))
    if j < i:
        return {"mass": 0.0, "mean": 0.0, "max": 0.0, "peak_in_span": 0.0, "entropy": entropy(p)}
    segment = p[i:j + 1]
    peak = int(np.argmax(p))
    return {
        "mass": float(segment.sum()),
        "mean": float(segment.mean()) if len(segment) else 0.0,
        "max": float(segment.max()) if len(segment) else 0.0,
        "peak_in_span": float(i <= peak <= j),
        "entropy": entropy(p),
    }


def _masked_temporal_softmax(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.bool()
    logits = logits.masked_fill(~mask, -1e4)
    probs = F.softmax(logits, dim=-1)
    probs = probs * mask.float()
    probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(EPS)
    return probs


@torch.no_grad()
def modality_temporal_prior(
    query_feature: torch.Tensor,
    video_feature: torch.Tensor,
    video_mask: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Return P_mod(t) from query/video hidden states.

    Args:
        query_feature: [N, D]
        video_feature: [N, T, D]
        video_mask: [N, T]
    """
    if query_feature.dim() == 3:
        # Conservative fallback for token-level query representations.
        query_feature = query_feature.mean(dim=1)
    q = F.normalize(query_feature.float(), dim=-1)
    v = F.normalize(video_feature.float(), dim=-1)
    logits = torch.einsum("nd,ntd->nt", q, v) / max(float(temperature), EPS)
    return _masked_temporal_softmax(logits, video_mask)


@torch.no_grad()
def build_qsp_priors_from_intermediates(
    intermediates: Dict,
    topk: int,
    visual_key: str = "visual",
    subtitle_key: str = "sub",
    temperature: float = 1.0,
) -> Dict[str, Optional[np.ndarray]]:
    """Build P_v, P_s and P_gate arrays from CONQUER intermediates.

    The returned arrays use the VCMR convention [qbs, topk, T], excluding the
    SVMR/GT slot at shared index 0.
    """
    qbs = int(intermediates["query_batch"])
    shared = int(intermediates["shared_video_num"])
    video_len = int(intermediates["video_len"])
    query_repeated = intermediates["query_feature_repeated"]
    video_mask = intermediates["video_mask"].bool()
    video_mask_dict = intermediates.get("video_mask_dict") or {}
    visual_mask = video_mask_dict.get(visual_key, video_mask).bool()
    subtitle_mask = video_mask_dict.get(subtitle_key, video_mask).bool()
    subtitle_coverage_mask = intermediates.get("sub_coverage_mask")
    video_feature_dict = intermediates.get("video_feature_dict") or {}

    out = {
        "p_v": None,
        "p_s": None,
        "p_gate": None,
        "sub_coverage": None,
        "q_v": None,
        "q_s": None,
    }

    if visual_key in video_feature_dict:
        p_v = modality_temporal_prior(query_repeated, video_feature_dict[visual_key], visual_mask, temperature)
        out["p_v"] = p_v.view(qbs, shared, video_len)[:, 1:topk + 1].detach().cpu().numpy()
    if subtitle_key in video_feature_dict:
        p_s = modality_temporal_prior(query_repeated, video_feature_dict[subtitle_key], subtitle_mask, temperature)
        out["p_s"] = p_s.view(qbs, shared, video_len)[:, 1:topk + 1].detach().cpu().numpy()
        with torch.no_grad():
            # Sequence masks only encode video length. True subtitle support is
            # supplied from the pre-encoder raw subtitle rows, where no-subtitle
            # clips are exact zero vectors.
            coverage = (
                subtitle_coverage_mask.bool()
                if subtitle_coverage_mask is not None else subtitle_mask
            ).float()
            out["sub_coverage"] = coverage.view(qbs, shared, video_len)[:, 1:topk + 1].detach().cpu().numpy()

    q_v = None
    q_s = None
    moe = intermediates.get("moe_weights_dict")
    if moe is not None and visual_key in moe and subtitle_key in moe:
        q_v = moe[visual_key].view(qbs, shared)[:, 1:topk + 1].detach().cpu().numpy()
        q_s = moe[subtitle_key].view(qbs, shared)[:, 1:topk + 1].detach().cpu().numpy()
    else:
        # Fallback: equal merge.  This is intentionally explicit and auditable.
        q_v = np.full((qbs, topk), 0.5, dtype=np.float32)
        q_s = np.full((qbs, topk), 0.5, dtype=np.float32)
    out["q_v"] = q_v
    out["q_s"] = q_s

    p_v_np, p_s_np = out["p_v"], out["p_s"]
    if p_v_np is not None and p_s_np is not None:
        denom = np.maximum(q_v + q_s, EPS)
        wv = (q_v / denom)[..., None]
        ws = (q_s / denom)[..., None]
        p_gate = wv * p_v_np + ws * p_s_np
        p_gate = p_gate / np.maximum(p_gate.sum(axis=-1, keepdims=True), EPS)
        out["p_gate"] = p_gate.astype(np.float32)
    elif p_v_np is not None:
        out["p_gate"] = p_v_np
    elif p_s_np is not None:
        out["p_gate"] = p_s_np
    return out



def _normalize_last_axis(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    arr = np.where(np.isfinite(arr), arr, 0.0)
    arr = np.maximum(arr, 0.0)
    denom = arr.sum(axis=-1, keepdims=True)
    fallback = np.full_like(arr, 1.0 / max(arr.shape[-1], 1), dtype=np.float32)
    return np.where(denom > EPS, arr / np.maximum(denom, EPS), fallback).astype(np.float32)


def _batch_entropy(prior_2d: np.ndarray) -> np.ndarray:
    p = _normalize_last_axis(prior_2d).astype(np.float64)
    return (-(p * np.log(p + EPS)).sum(axis=-1)).astype(np.float32)


def _batch_js_divergence(p_2d: Optional[np.ndarray], q_2d: Optional[np.ndarray]) -> np.ndarray:
    if p_2d is None or q_2d is None:
        length = 0 if p_2d is None and q_2d is None else (p_2d if q_2d is None else q_2d).shape[0]
        return np.zeros((length,), dtype=np.float32)
    p = _normalize_last_axis(p_2d).astype(np.float64)
    q = _normalize_last_axis(q_2d).astype(np.float64)
    m = 0.5 * (p + q)
    kl_pm = (p * (np.log(p + EPS) - np.log(m + EPS))).sum(axis=-1)
    kl_qm = (q * (np.log(q + EPS) - np.log(m + EPS))).sum(axis=-1)
    return np.maximum(0.0, 0.5 * (kl_pm + kl_qm)).astype(np.float32)


def _span_stats_vectorized(prior_2d: Optional[np.ndarray], v_idx: np.ndarray, st_idx: np.ndarray, ed_idx: np.ndarray) -> Dict[str, np.ndarray]:
    """Vectorized span statistics for one query's candidate rows.

    Args:
        prior_2d: [topk, T] temporal prior for one query, or None.
        v_idx/st_idx/ed_idx: [K] candidate arrays.

    Returns feature arrays with shape [K]. Segment max still uses a bounded
    K-loop because K is max_before_nms=200; prefix-sum mass/mean, entropy and
    peak checks are vectorized. This preserves exact bounded candidate semantics
    while removing the expensive per-row normalization/entropy/JS work.
    """
    k = int(len(v_idx))
    if prior_2d is None or k == 0:
        zeros = np.zeros((k,), dtype=np.float32)
        return {"mass": zeros, "mean": zeros, "max": zeros, "peak_in_span": zeros, "entropy": zeros}
    p = _normalize_last_axis(prior_2d)
    topk, t_len = p.shape
    v = np.clip(v_idx.astype(np.int64), 0, topk - 1)
    st = np.clip(st_idx.astype(np.int64), 0, t_len - 1)
    ed = np.clip(ed_idx.astype(np.int64), 0, t_len - 1)
    valid = ed >= st
    prefix = np.concatenate([np.zeros((topk, 1), dtype=np.float32), np.cumsum(p, axis=-1, dtype=np.float32)], axis=-1)
    mass = prefix[v, ed + 1] - prefix[v, st]
    mass = np.where(valid, mass, 0.0).astype(np.float32)
    length = np.maximum(ed - st + 1, 1).astype(np.float32)
    mean = np.where(valid, mass / length, 0.0).astype(np.float32)
    # K is bounded to 200 under the frozen protocol; this tiny loop is cheaper
    # than materializing a [K, T] boolean mask for every query and avoids any
    # numerical approximation.
    segmax = np.zeros((k,), dtype=np.float32)
    for n in range(k):
        if valid[n]:
            segmax[n] = float(p[v[n], st[n]:ed[n] + 1].max())
    peak = np.argmax(p, axis=-1)
    peak_in = ((st <= peak[v]) & (peak[v] <= ed) & valid).astype(np.float32)
    ent = _batch_entropy(p)[v].astype(np.float32)
    return {"mass": mass, "mean": mean, "max": segmax, "peak_in_span": peak_in, "entropy": ent}


def qsp_rows_features_vectorized(
    priors: Dict[str, Optional[np.ndarray]],
    q_idx: int,
    v_local: np.ndarray,
    st_idx: np.ndarray,
    ed_idx: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Return QSP feature arrays for all selected candidate rows of one query.

    The output values are float32 NumPy arrays keyed by QSP_FEATURES. It is
    designed for export-time JSON streaming: compute all features in vectorized
    blocks, then only loop to serialize rows and perform strict identity checks.
    """
    v_local = np.asarray(v_local, dtype=np.int64)
    st_idx = np.asarray(st_idx, dtype=np.int64)
    ed_idx = np.asarray(ed_idx, dtype=np.int64)
    k = int(len(v_local))
    p_v = None if priors.get("p_v") is None else priors["p_v"][q_idx]
    p_s = None if priors.get("p_s") is None else priors["p_s"][q_idx]
    p_gate = None if priors.get("p_gate") is None else priors["p_gate"][q_idx]
    sv = _span_stats_vectorized(p_v, v_local, st_idx, ed_idx)
    ss = _span_stats_vectorized(p_s, v_local, st_idx, ed_idx)
    sg = _span_stats_vectorized(p_gate, v_local, st_idx, ed_idx)

    if priors.get("q_v") is not None:
        q_v = np.asarray(priors["q_v"][q_idx, v_local], dtype=np.float32)
    else:
        q_v = np.full((k,), 0.5, dtype=np.float32)
    if priors.get("q_s") is not None:
        q_s = np.asarray(priors["q_s"][q_idx, v_local], dtype=np.float32)
    else:
        q_s = np.full((k,), 0.5, dtype=np.float32)
    denom = np.maximum(q_v + q_s, EPS).astype(np.float32)
    q_v = (q_v / denom).astype(np.float32)
    q_s = (q_s / denom).astype(np.float32)
    mod_entropy = (-(q_v * np.log(q_v + EPS) + q_s * np.log(q_s + EPS))).astype(np.float32)

    sub_cov = np.zeros((k,), dtype=np.float32)
    if priors.get("sub_coverage") is not None:
        cov = np.asarray(priors["sub_coverage"][q_idx], dtype=np.float32)
        topk, t_len = cov.shape
        v = np.clip(v_local, 0, topk - 1)
        st = np.clip(st_idx, 0, t_len - 1)
        ed = np.clip(ed_idx, 0, t_len - 1)
        valid = ed >= st
        prefix = np.concatenate([np.zeros((topk, 1), dtype=np.float32), np.cumsum(cov, axis=-1, dtype=np.float32)], axis=-1)
        mass_cov = prefix[v, ed + 1] - prefix[v, st]
        length = np.maximum(ed - st + 1, 1).astype(np.float32)
        sub_cov = np.where(valid, mass_cov / length, 0.0).astype(np.float32)

    js = np.zeros((k,), dtype=np.float32)
    if p_v is not None and p_s is not None:
        js_all = _batch_js_divergence(p_v, p_s)
        js = js_all[np.clip(v_local, 0, len(js_all) - 1)].astype(np.float32)
    gate_conf = np.zeros((k,), dtype=np.float32)
    if p_gate is not None and p_gate.shape[-1] > 1:
        gate_conf = (1.0 - sg["entropy"] / max(np.log(p_gate.shape[-1] + EPS), EPS)).astype(np.float32)

    return {
        "qsp_mass_v": sv["mass"], "qsp_mean_v": sv["mean"], "qsp_max_v": sv["max"],
        "qsp_peak_in_v": sv["peak_in_span"], "qsp_entropy_v": sv["entropy"],
        "qsp_mass_s": ss["mass"], "qsp_mean_s": ss["mean"], "qsp_max_s": ss["max"],
        "qsp_peak_in_s": ss["peak_in_span"], "qsp_entropy_s": ss["entropy"],
        "qsp_mass_gate": sg["mass"], "qsp_mean_gate": sg["mean"], "qsp_max_gate": sg["max"],
        "qsp_peak_in_gate": sg["peak_in_span"], "qsp_entropy_gate": sg["entropy"],
        "qsp_js_vs": js,
        "qsp_gate_confidence": gate_conf,
        "qsp_sub_coverage": sub_cov,
        "qsp_q_v": q_v, "qsp_q_s": q_s,
        "qsp_modality_gap": np.abs(q_v - q_s).astype(np.float32),
        "qsp_modality_entropy": mod_entropy,
    }


def qsp_row_features(priors: Dict[str, Optional[np.ndarray]], q_idx: int, v_local: int, st_idx: int, ed_idx: int) -> Dict[str, float]:
    p_v = None if priors.get("p_v") is None else priors["p_v"][q_idx, v_local]
    p_s = None if priors.get("p_s") is None else priors["p_s"][q_idx, v_local]
    p_gate = None if priors.get("p_gate") is None else priors["p_gate"][q_idx, v_local]
    cov = None if priors.get("sub_coverage") is None else priors["sub_coverage"][q_idx, v_local]
    sv = span_prior_stats(p_v, st_idx, ed_idx)
    ss = span_prior_stats(p_s, st_idx, ed_idx)
    sg = span_prior_stats(p_gate, st_idx, ed_idx)
    q_v = _safe_float(priors.get("q_v")[q_idx, v_local]) if priors.get("q_v") is not None else 0.5
    q_s = _safe_float(priors.get("q_s")[q_idx, v_local]) if priors.get("q_s") is not None else 0.5
    norm = max(q_v + q_s, EPS)
    q_v, q_s = q_v / norm, q_s / norm
    mod_entropy = float(-(q_v * np.log(q_v + EPS) + q_s * np.log(q_s + EPS)))
    sub_cov = 0.0
    if cov is not None:
        i, j = max(0, int(st_idx)), min(len(cov) - 1, int(ed_idx))
        if j >= i:
            sub_cov = float(np.asarray(cov[i:j + 1], dtype=np.float32).mean())
    return {
        "qsp_mass_v": sv["mass"], "qsp_mean_v": sv["mean"], "qsp_max_v": sv["max"],
        "qsp_peak_in_v": sv["peak_in_span"], "qsp_entropy_v": sv["entropy"],
        "qsp_mass_s": ss["mass"], "qsp_mean_s": ss["mean"], "qsp_max_s": ss["max"],
        "qsp_peak_in_s": ss["peak_in_span"], "qsp_entropy_s": ss["entropy"],
        "qsp_mass_gate": sg["mass"], "qsp_mean_gate": sg["mean"], "qsp_max_gate": sg["max"],
        "qsp_peak_in_gate": sg["peak_in_span"], "qsp_entropy_gate": sg["entropy"],
        "qsp_js_vs": js_divergence(p_v, p_s) if p_v is not None and p_s is not None else 0.0,
        "qsp_gate_confidence": 1.0 - (sg["entropy"] / max(np.log(len(p_gate) + EPS), EPS)) if p_gate is not None and len(p_gate) > 1 else 0.0,
        "qsp_sub_coverage": sub_cov,
        "qsp_q_v": q_v, "qsp_q_s": q_s,
        "qsp_modality_gap": abs(q_v - q_s),
        "qsp_modality_entropy": mod_entropy,
    }
