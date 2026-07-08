from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
import pandas as pd

from blueprint_e2e_v2.data.temporal_grid import iou_1d


def _metric(xs: list[bool]) -> float:
    return 100.0 * sum(bool(x) for x in xs) / max(1, len(xs))


def nms_rank_query(rows: pd.DataFrame, score_col: str, threshold: float = 0.7, max_keep: int = 200) -> pd.DataFrame:
    rows = rows.sort_values(score_col, ascending=False)
    kept = []
    by_video: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for idx, r in rows.iterrows():
        vid = str(r["video_id"])
        span = (float(r["span_start"]), float(r["span_end"]))
        if any(iou_1d(span, prev) > threshold for prev in by_video[vid]):
            continue
        by_video[vid].append(span)
        kept.append(idx)
        if len(kept) >= max_keep:
            break
    return rows.loc[kept].copy()


def evaluate_score_table(df: pd.DataFrame, score_col: str = "vcmr_score", nms_threshold: float = 0.7) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    duplicate_count = int(df.duplicated(["split", "seed", "query_id", "video_id", "span_start", "span_end"]).sum())
    invalid_count = int(((df["span_end"] <= df["span_start"]) | df["span_start"].isna() | df["span_end"].isna()).sum())
    for (_split, seed, qid), qdf in df.groupby(["split", "seed", "query_id"], sort=False):
        ranked = nms_rank_query(qdf, score_col, threshold=nms_threshold)
        if ranked.empty:
            continue
        gt_video = str(ranked.iloc[0]["gt_video_id"])
        gt_ts = (float(ranked.iloc[0]["gt_start"]), float(ranked.iloc[0]["gt_end"]))
        ious = [iou_1d((float(r["span_start"]), float(r["span_end"])), gt_ts) if str(r["video_id"]) == gt_video else 0.0 for _, r in ranked.iterrows()]
        videos = [str(v) for v in ranked["video_id"].tolist()]
        unique_videos = []
        for v in videos:
            if v not in unique_videos:
                unique_videos.append(v)
        top1_video_ok = bool(videos and videos[0] == gt_video)
        top1_iou = float(ious[0]) if ious else 0.0
        rec = {
            "split": _split,
            "seed": int(seed),
            "query_id": int(qid),
            "query_type": str(ranked.iloc[0].get("query_type", "unknown")),
            "duration_bucket": str(ranked.iloc[0].get("duration_bucket", "unknown")),
            "top1_video_correct": top1_video_ok,
            "wrong_video_top1": not top1_video_ok,
            "correct_video_wrong_span_top1": bool(top1_video_ok and top1_iou < 0.5),
            "top1_iou": top1_iou,
        }
        for k in [1, 5, 10, 100]:
            rec[f"VCMR_R@{k}_IoU0.5"] = any(i >= 0.5 for i in ious[: min(k, len(ious))])
            rec[f"VCMR_R@{k}_IoU0.7"] = any(i >= 0.7 for i in ious[: min(k, len(ious))])
            rec[f"VR_R@{k}"] = gt_video in unique_videos[: min(k, len(unique_videos))]
        records.append(rec)

    def aggregate(rs: list[dict[str, Any]]) -> dict[str, Any]:
        out: dict[str, Any] = {"query_count": len(rs)}
        for k in [1, 5, 10, 100]:
            out[f"VCMR_R@{k}_IoU0.5"] = _metric([r[f"VCMR_R@{k}_IoU0.5"] for r in rs])
            out[f"VCMR_R@{k}_IoU0.7"] = _metric([r[f"VCMR_R@{k}_IoU0.7"] for r in rs])
            out[f"VR_R@{k}"] = _metric([r[f"VR_R@{k}"] for r in rs])
        out["wrong_video_top1_rate"] = _metric([r["wrong_video_top1"] for r in rs])
        out["high_score_false_positive_rate"] = out["wrong_video_top1_rate"]
        out["correct_video_wrong_span_rate"] = _metric([r["correct_video_wrong_span_top1"] for r in rs])
        out["top1_mean_iou"] = float(np.mean([float(r["top1_iou"]) for r in rs])) if rs else 0.0
        out["duplicate_span_count"] = duplicate_count
        out["invalid_span_count"] = invalid_count
        return out

    summary = aggregate(records)
    by_qtype = {}
    by_duration = {}
    if records:
        frame = pd.DataFrame(records)
        by_qtype = {str(k): aggregate(v.to_dict("records")) for k, v in frame.groupby("query_type")}
        by_duration = {str(k): aggregate(v.to_dict("records")) for k, v in frame.groupby("duration_bucket")}
    return {"summary": summary, "by_query_type": by_qtype, "by_duration": by_duration, "records": records[:2000]}


def metric_delta(a: dict[str, Any], b: dict[str, Any]) -> dict[str, float]:
    keys = sorted(set(a) | set(b))
    out: dict[str, float] = {}
    for k in keys:
        if isinstance(a.get(k), (int, float)) and isinstance(b.get(k), (int, float)):
            out[k] = float(a.get(k, 0.0) - b.get(k, 0.0))
    return out


def selection_score(delta: dict[str, float]) -> float:
    return (
        10.0 * delta.get("VCMR_R@1_IoU0.7", 0.0)
        + 4.0 * delta.get("VCMR_R@5_IoU0.7", 0.0)
        + 2.0 * delta.get("VCMR_R@10_IoU0.7", 0.0)
        + 1.5 * delta.get("VCMR_R@1_IoU0.5", 0.0)
        + 1.0 * delta.get("VCMR_R@5_IoU0.5", 0.0)
        - 3.0 * max(0.0, delta.get("wrong_video_top1_rate", 0.0))
        - 2.0 * max(0.0, -delta.get("VR_R@100", 0.0))
        - 1.5 * max(0.0, delta.get("high_score_false_positive_rate", 0.0))
    )

