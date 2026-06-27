#!/usr/bin/env python
"""Utilities for CONQUER-RLEM C4-lite video-level calibration.

C4-lite is intentionally *not* an R2/VS-head implementation.  It consumes fixed
candidate rows and frozen C3/C3.1 evidence-head predictions, aggregates span
quality to a video-level signal, and reranks the existing 200 candidates/query
without changing candidate generation, top-100, NMS, or the evaluator.
"""

import gzip
import json
import math
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from rlem.io_utils import iter_jsonl, read_json, write_json  # noqa: E402
from utils.inference_utils import filter_vcmr_by_nms  # noqa: E402

EPS = 1e-8
R2_FORBIDDEN = {"r2_raw", "r2_prob", "r2_tilde", "rank_r2", "r_abs_gap"}

PRIMARY_METRICS = [
    "0.5-r1", "0.5-r5", "0.5-r10",
    "0.7-r1", "0.7-r5", "0.7-r10",
]
ALL_METRICS = PRIMARY_METRICS + ["0.5-r100", "0.7-r100"]


def open_text(path: str, mode: str = "rt"):
    if str(path).endswith(".gz"):
        return gzip.open(path, mode, encoding="utf-8")
    return open(path, mode, encoding="utf-8")


def atomic_text_writer(path: str, gzip_output: Optional[bool] = None, gzip_compresslevel: int = 6):
    path = str(path)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if gzip_output is None:
        gzip_output = path.endswith(".gz")
    partial = path + ".partial"
    if gzip_output:
        if not partial.endswith(".gz"):
            partial = path + ".partial.gz"
        handle = gzip.open(partial, "wt", encoding="utf-8", compresslevel=int(gzip_compresslevel))
    else:
        handle = open(partial, "w", encoding="utf-8")
    return handle, partial


def replace_atomic(partial: str, final: str) -> None:
    os.replace(partial, final)


def load_desc_ids(path: Optional[str]) -> Optional[set]:
    if not path:
        return None
    with open(path, "r", encoding="utf-8") as f:
        values = {line.strip() for line in f if line.strip()}
    if not values:
        raise ValueError(f"No desc_ids found in {path}")
    return values


def sigmoid_temperature(logits: np.ndarray, temperature: float) -> np.ndarray:
    values = np.clip(logits.astype(np.float64) / float(temperature), -40.0, 40.0)
    return (1.0 / (1.0 + np.exp(-values))).astype(np.float32)


def safe_log(value) -> float:
    return math.log(max(float(value), EPS))


def get_row_q_scores(row: Dict, t_joint: float = 1.0, t_bd: float = 1.0, t_fp: float = 1.0) -> Tuple[float, float, float]:
    """Return calibrated q_joint/q_bd/e_fp from a scored row.

    Supports either calibrated columns, sigmoid-score columns, or logit columns.
    """
    if "q_joint_cal" in row and "q_bd_cal" in row and "e_fp_cal" in row:
        return float(row["q_joint_cal"]), float(row["q_bd_cal"]), float(row["e_fp_cal"])
    if "q_joint_pred" in row and "q_bd_pred" in row and "e_fp_pred" in row:
        return float(row["q_joint_pred"]), float(row["q_bd_pred"]), float(row["e_fp_pred"])
    if "q_joint_logit" in row and "q_bd_logit" in row and "e_fp_logit" in row:
        qj = sigmoid_temperature(np.asarray([float(row["q_joint_logit"])]), t_joint)[0]
        qb = sigmoid_temperature(np.asarray([float(row["q_bd_logit"])]), t_bd)[0]
        efp = sigmoid_temperature(np.asarray([float(row["e_fp_logit"])]), t_fp)[0]
        return float(qj), float(qb), float(efp)
    raise KeyError("Row lacks q_joint/q_bd/e_fp prediction or logit columns")


def infer_log_r1(row: Dict) -> float:
    # C1 evidence generally has r1. Some compact C3 scored files may have r1_tilde
    # or only base score. If video score is unavailable, fall back to log(s_base)
    # and mark this in the cache manifest.
    for key in ["r1", "r1_raw", "r1_score"]:
        if key in row and row[key] is not None:
            return safe_log(row[key])
    return safe_log(row.get("s_base", 0.0))


def row_span_quality(q_joint: float, q_bd: float, e_fp: float, alpha_bd: float, gamma_fp: float, mean_qbd: float = 0.0, centered_bd: bool = True) -> float:
    bd_term = q_bd - float(mean_qbd) if centered_bd else q_bd
    return float(q_joint + alpha_bd * q_joint * bd_term - gamma_fp * e_fp)


def iter_query_groups(path: str, allowed_ids: Optional[set] = None) -> Iterator[List[Dict]]:
    current_id = None
    current = []
    completed = set()
    for row in iter_jsonl(path):
        desc_id = str(row["desc_id"])
        if allowed_ids is not None and desc_id not in allowed_ids:
            continue
        if current_id is None:
            current_id = desc_id
        if desc_id != current_id:
            if current_id in completed:
                raise ValueError(f"Non-contiguous rows for desc_id={current_id}")
            completed.add(current_id)
            yield current
            current = []
            current_id = desc_id
        current.append(row)
    if current:
        yield current


def load_video2idx(dataset_config: str, split: str) -> Dict[str, int]:
    cfg = read_json(dataset_config)
    path = cfg["video_duration_idx_path"]
    if not os.path.isabs(path):
        path = os.path.join(cfg["root_path"], path)
    obj = read_json(path)
    return {name: int(value[1]) for name, value in obj[split].items()}


def submission_from_groups(
    scored_groups: Iterable[List[Dict]],
    score_key: str,
    dataset_config: str,
    split: str,
    effective_top_n: int = 100,
    max_after_nms: int = 100,
    nms_thd: float = 0.7,
) -> Dict:
    vcmr = []
    row_counts = Counter()
    for rows in scored_groups:
        if not rows:
            continue
        row_counts[len(rows)] += 1
        # Stable tie-breaker preserves C1 frozen order.
        order = sorted(range(len(rows)), key=lambda i: (-float(rows[i][score_key]), int(rows[i].get("rank_base", i))))
        candidates = [
            [
                int(rows[i]["video_idx"]),
                float(rows[i]["start_time"]),
                float(rows[i]["end_time"]),
                float(rows[i][score_key]),
            ]
            for i in order[:effective_top_n]
        ]
        predictions = filter_vcmr_by_nms(
            candidates,
            nms_threshold=nms_thd,
            max_before_nms=effective_top_n,
            max_after_nms=max_after_nms,
        )
        first = rows[0]
        vcmr.append({"desc_id": first["desc_id"], "desc": first.get("desc", ""), "predictions": predictions})
    return {"video2idx": load_video2idx(dataset_config, split), "VCMR": vcmr, "_row_counts": dict(row_counts)}


def flatten_eval_metrics(metrics: Dict) -> Dict[str, float]:
    """Convert bundled evaluator output to flat 0.5-r1 style keys.

    Supports the common TVR evaluator structures seen in CONQUER artifacts.
    """
    flat = {}
    # Already flat.
    for key, value in metrics.items():
        if isinstance(value, (int, float)) and ("r" in key.lower() or "R" in key):
            flat[key.replace(" ", "").lower()] = float(value)
    # Nested VCMR/iou/rank variants.
    root = metrics.get("VCMR", metrics.get("vcmr", metrics))
    if isinstance(root, dict):
        # This repository's bundled evaluator returns
        # {"VCMR": {"0.5-r1": 37.75, ...}}.
        for metric_key, metric_value in root.items():
            normalized = str(metric_key).lower().replace(" ", "")
            if normalized in ALL_METRICS and isinstance(metric_value, (int, float)):
                flat[normalized] = float(metric_value)
        for iou_key, sub in root.items():
            iou_str = str(iou_key).replace("IoU", "").replace("iou", "").replace("@", "").strip()
            if iou_str.startswith("0.5") or iou_str == "0.5":
                prefix = "0.5"
            elif iou_str.startswith("0.7") or iou_str == "0.7":
                prefix = "0.7"
            else:
                continue
            if isinstance(sub, dict):
                for rkey, val in sub.items():
                    rnorm = str(rkey).lower().replace(" ", "").replace("@", "")
                    if rnorm in {"r1", "r@1", "recall1"}:
                        flat[f"{prefix}-r1"] = float(val)
                    elif rnorm in {"r5", "r@5", "recall5"}:
                        flat[f"{prefix}-r5"] = float(val)
                    elif rnorm in {"r10", "r@10", "recall10"}:
                        flat[f"{prefix}-r10"] = float(val)
                    elif rnorm in {"r100", "r@100", "recall100"}:
                        flat[f"{prefix}-r100"] = float(val)
    # Common keys like "R@1,IoU=0.5".
    def normalize_key(k: str) -> Optional[str]:
        s = k.lower().replace(" ", "")
        iou = None
        if "0.5" in s:
            iou = "0.5"
        elif "0.7" in s:
            iou = "0.7"
        if iou is None:
            return None
        for r in [100, 10, 5, 1]:
            if f"r@{r}" in s or f"r{r}" in s:
                return f"{iou}-r{r}"
        return None
    for key, value in metrics.items():
        nk = normalize_key(str(key))
        if nk and isinstance(value, (int, float)):
            flat[nk] = float(value)
    return flat


def selection_score(flat_metrics: Dict[str, float], baseline: Optional[Dict[str, float]] = None) -> float:
    if baseline is None:
        values = [flat_metrics[k] for k in PRIMARY_METRICS if k in flat_metrics]
        r100 = [flat_metrics[k] for k in ["0.5-r100", "0.7-r100"] if k in flat_metrics]
    else:
        values = [flat_metrics[k] - baseline[k] for k in PRIMARY_METRICS if k in flat_metrics and k in baseline]
        r100 = [flat_metrics[k] - baseline[k] for k in ["0.5-r100", "0.7-r100"] if k in flat_metrics and k in baseline]
    if len(values) != 6:
        return float("nan")
    return float(np.mean(values) + 0.25 * (np.mean(r100) if r100 else 0.0))


def safety_from_rows(rows: Sequence[Dict], base_key: str, new_key: str) -> Dict[str, float]:
    base = np.asarray([float(r[base_key]) for r in rows], dtype=np.float64)
    new = np.asarray([float(r[new_key]) for r in rows], dtype=np.float64)
    if base.size < 2 or np.std(base) == 0 or np.std(new) == 0:
        pearson = float("nan")
    else:
        pearson = float(np.corrcoef(base, new)[0, 1])
    changed_top1 = 0
    total_q = 0
    current = None
    group = []
    for r in rows:
        if current is None:
            current = r["desc_id"]
        if r["desc_id"] != current:
            total_q += 1
            changed_top1 += int(_top1_changed(group, base_key, new_key))
            group = []
            current = r["desc_id"]
        group.append(r)
    if group:
        total_q += 1
        changed_top1 += int(_top1_changed(group, base_key, new_key))
    return {"pearson_base_new": pearson, "top1_changed_ratio": changed_top1 / max(total_q, 1), "queries": total_q}


def _top1_changed(group: Sequence[Dict], base_key: str, new_key: str) -> bool:
    base_top = min(range(len(group)), key=lambda i: (-float(group[i][base_key]), int(group[i].get("rank_base", i))))
    new_top = min(range(len(group)), key=lambda i: (-float(group[i][new_key]), int(group[i].get("rank_base", i))))
    return base_top != new_top
