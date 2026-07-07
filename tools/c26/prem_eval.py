from __future__ import annotations

from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd


METRIC_KEYS = [
    "VCMR_R@1_IoU0.5", "VCMR_R@5_IoU0.5", "VCMR_R@10_IoU0.5", "VCMR_R@100_IoU0.5",
    "VCMR_R@1_IoU0.7", "VCMR_R@5_IoU0.7", "VCMR_R@10_IoU0.7", "VCMR_R@100_IoU0.7",
    "VR_R@1", "VR_R@5", "VR_R@10", "VR_R@100",
    "wrong_video_top1_rate", "high_score_false_positive_rate",
]


def zscore_by_query(df: pd.DataFrame, col: str) -> np.ndarray:
    vals = pd.to_numeric(df[col], errors="coerce").fillna(0.0).astype(np.float32)
    out = np.zeros(len(df), dtype=np.float32)
    for _qid, idx in df.groupby("query_id", sort=False).indices.items():
        x = vals.iloc[idx].to_numpy(np.float32)
        out[idx] = (x - float(x.mean())) / max(float(x.std()), 1e-6)
    return out


def minmax_by_query(df: pd.DataFrame, col: str) -> np.ndarray:
    vals = pd.to_numeric(df[col], errors="coerce").fillna(0.0).astype(np.float32)
    out = np.zeros(len(df), dtype=np.float32)
    for _qid, idx in df.groupby("query_id", sort=False).indices.items():
        x = vals.iloc[idx].to_numpy(np.float32)
        out[idx] = (x - float(x.min())) / max(float(x.max() - x.min()), 1e-6)
    return out


def iou_1d(a0: float, a1: float, b0: float, b1: float) -> float:
    inter = max(0.0, min(a1, b1) - max(a0, b0))
    union = max(a1, b1) - min(a0, b0)
    return inter / union if union > 0 else 0.0


def recall(flags: Sequence[bool]) -> float:
    return 100.0 * sum(bool(x) for x in flags) / max(1, len(flags))


def evaluate_vr_vcmr(df: pd.DataFrame, score_col: str, span_start_col: str | None = None, span_end_col: str | None = None) -> Dict[str, Any]:
    vr_hits = {1: [], 5: [], 10: [], 100: []}
    vcmr05 = {1: [], 5: [], 10: [], 100: []}
    vcmr07 = {1: [], 5: [], 10: [], 100: []}
    top1_wrong: List[bool] = []
    high_fp: List[bool] = []
    scores = pd.to_numeric(df[score_col], errors="coerce").fillna(-1e9).to_numpy(np.float32)
    work = df.copy()
    work["_score_eval"] = scores
    for _qid, g in work.groupby("query_id", sort=False):
        gg = g.sort_values("_score_eval", ascending=False).head(128)
        is_gt = gg["video_id"].astype(str).to_numpy() == gg["gt_video_id"].astype(str).to_numpy()
        top1_wrong.append(not bool(is_gt[0]) if len(is_gt) else True)
        threshold = float(np.quantile(gg["_score_eval"].to_numpy(np.float32), 0.90)) if len(gg) else 0.0
        if len(gg):
            top = gg.iloc[0]
            high_fp.append(bool(top["_score_eval"] >= threshold and str(top["video_id"]) != str(top["gt_video_id"])))
        for k in vr_hits:
            sub = gg.head(k)
            gt_video_hit = bool((sub["video_id"].astype(str) == sub["gt_video_id"].astype(str)).any())
            vr_hits[k].append(gt_video_hit)
            ok05 = False
            ok07 = False
            if span_start_col and span_end_col and span_start_col in sub.columns and span_end_col in sub.columns:
                for rec in sub.itertuples(index=False):
                    if str(getattr(rec, "video_id")) != str(getattr(rec, "gt_video_id")):
                        continue
                    ps = float(getattr(rec, span_start_col))
                    pe = float(getattr(rec, span_end_col))
                    gt_s = float(getattr(rec, "gt_start"))
                    gt_e = float(getattr(rec, "gt_end"))
                    iou = iou_1d(ps, pe, gt_s, gt_e)
                    ok05 = ok05 or iou >= 0.5
                    ok07 = ok07 or iou >= 0.7
            vcmr05[k].append(ok05)
            vcmr07[k].append(ok07)
    out = {"query_count": int(work["query_id"].nunique())}
    for k in [1, 5, 10, 100]:
        out[f"VR_R@{k}"] = recall(vr_hits[k])
        out[f"VCMR_R@{k}_IoU0.5"] = recall(vcmr05[k])
        out[f"VCMR_R@{k}_IoU0.7"] = recall(vcmr07[k])
    out["wrong_video_top1_rate"] = recall(top1_wrong)
    out["high_score_false_positive_rate"] = recall(high_fp)
    return out


def metric_delta(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, float]:
    return {k: float(a.get(k, 0.0) or 0.0) - float(b.get(k, 0.0) or 0.0) for k in METRIC_KEYS if k in a or k in b}


def selection_score(metrics: Dict[str, Any], baseline: Dict[str, Any]) -> float:
    d = metric_delta(metrics, baseline)
    return (
        3.0 * d.get("VCMR_R@1_IoU0.7", 0.0)
        + 2.0 * d.get("VCMR_R@5_IoU0.7", 0.0)
        + 1.5 * d.get("VCMR_R@10_IoU0.7", 0.0)
        + 1.0 * d.get("VCMR_R@1_IoU0.5", 0.0)
        + 0.8 * d.get("VCMR_R@5_IoU0.5", 0.0)
        - 2.0 * max(0.0, d.get("wrong_video_top1_rate", 0.0))
        - 1.0 * max(0.0, baseline.get("VR_R@100", 0.0) - metrics.get("VR_R@100", 0.0) - 0.2)
        - 0.8 * max(0.0, d.get("high_score_false_positive_rate", 0.0))
    )


def breakdowns(df: pd.DataFrame, score_col: str, span_start_col: str | None = None, span_end_col: str | None = None) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for col in ["query_type", "duration_bucket"]:
        if col not in df.columns:
            continue
        out[col] = {}
        for key, g in df.groupby(col):
            out[col][str(key)] = evaluate_vr_vcmr(g, score_col, span_start_col, span_end_col)
    return out
