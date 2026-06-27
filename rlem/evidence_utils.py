
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import math
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

EPS = 1e-8


def temporal_iou_seconds(pred: Sequence[float], gt: Optional[Sequence[float]]) -> float:
    if gt is None:
        return 0.0
    inter = max(0.0, min(float(pred[1]), float(gt[1])) - max(float(pred[0]), float(gt[0])))
    union = max(float(pred[1]), float(gt[1])) - min(float(pred[0]), float(gt[0]))
    return float(inter / union) if union > 0 else 0.0


def entropy(prob: np.ndarray) -> float:
    prob = np.asarray(prob, dtype=np.float64)
    return float(-(prob * np.log(prob + EPS)).sum())


def top_margin(prob: np.ndarray) -> float:
    prob = np.asarray(prob, dtype=np.float64)
    if prob.size < 2:
        return 0.0
    top2 = np.partition(prob, -2)[-2:]
    return float(top2.max() - top2.min())


def local_sharpness(prob: np.ndarray, idx: int, radius: int = 2) -> float:
    prob = np.asarray(prob, dtype=np.float64)
    lo = max(0, int(idx) - radius)
    hi = min(prob.size, int(idx) + radius + 1)
    neighbors = [k for k in range(lo, hi) if k != int(idx)]
    if not neighbors:
        return float(prob[int(idx)])
    return float(prob[int(idx)] - prob[neighbors].mean())


def query_standardize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return (values - values.mean()) / (values.std() + EPS)


def span_stats_from_prior(prior: Optional[np.ndarray], st_idx: int, ed_idx: int) -> Dict[str, float]:
    if prior is None:
        return {
            "mass": None,
            "mean": None,
            "max": None,
            "peak_in_span": None,
            "entropy": None,
        }
    prior = np.asarray(prior, dtype=np.float64)
    st_idx = int(st_idx)
    ed_idx = int(ed_idx)
    span = prior[st_idx:ed_idx + 1]
    peak_idx = int(np.argmax(prior)) if prior.size else -1
    return {
        "mass": float(span.sum()),
        "mean": float(span.mean()) if span.size else 0.0,
        "max": float(span.max()) if span.size else 0.0,
        "peak_in_span": int(st_idx <= peak_idx <= ed_idx),
        "entropy": entropy(prior),
    }


def make_length_mask(shape: Tuple[int, ...], min_l: int, max_l: int) -> np.ndarray:
    """Same valid-span convention as CONQUER inference.generate_min_max_length_mask."""
    single_dims = (1,) * (len(shape) - 2)
    mask_shape = single_dims + shape[-2:]
    base = np.ones(mask_shape, dtype=np.float32)
    mask_triu = np.triu(base, k=min_l)
    mask_triu_reversed = 1 - np.triu(base, k=max_l)
    return mask_triu * mask_triu_reversed
