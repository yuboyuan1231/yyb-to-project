from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd


def _norm_by_group(df: pd.DataFrame, col: str, keys=None) -> np.ndarray:
    keys = keys or [df["split"], df["query_id"], df["seed"]]
    vals = pd.to_numeric(df[col], errors="coerce").fillna(0.0).astype(np.float32)
    gb = vals.groupby(keys, observed=True)
    mn = gb.transform("min").astype(np.float32)
    mx = gb.transform("max").astype(np.float32)
    return ((vals - mn) / (mx - mn).replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(np.float32)


def add_span_relevance(df: pd.DataFrame) -> Dict[str, str]:
    df["visual_span_relevance_mean"] = _norm_by_group(df, "c26_visual_relevance")
    df["visual_span_relevance_max"] = np.maximum(df["visual_span_relevance_mean"].to_numpy(np.float32), _norm_by_group(df, "start_prob") if "start_prob" in df else 0.0)
    df["visual_span_relevance_topk"] = 0.7 * df["visual_span_relevance_mean"] + 0.3 * _norm_by_group(df, "actionness_score")
    df["subtitle_span_relevance_mean"] = _norm_by_group(df, "c26_subtitle_relevance")
    df["subtitle_span_relevance_max"] = np.maximum(df["subtitle_span_relevance_mean"].to_numpy(np.float32), _norm_by_group(df, "t2_score"))
    df["subtitle_span_relevance_topk"] = 0.7 * df["subtitle_span_relevance_mean"] + 0.3 * _norm_by_group(df, "t2_score")
    df["modality_gate_visual"] = df["visual_span_relevance_topk"].astype(np.float32)
    df["modality_gate_subtitle"] = df["subtitle_span_relevance_topk"].astype(np.float32)
    df["modality_gate_joint"] = 0.5 * (df["modality_gate_visual"] + df["modality_gate_subtitle"])
    bmn = _norm_by_group(df, "bmn_final_score")
    t2 = _norm_by_group(df, "t2_score")
    dur = pd.to_numeric(df["span_duration"], errors="coerce").fillna(0.0).astype(np.float32)
    dur_prior = np.exp(-np.abs((dur / dur.groupby([df["split"], df["query_id"], df["seed"]], observed=True).transform("max").replace(0.0, np.nan).fillna(1.0)).to_numpy(np.float32) - 0.14) * 6.0).astype(np.float32)
    df["prem_span_relevance"] = (
        0.25 * df["visual_span_relevance_topk"].to_numpy(np.float32)
        + 0.25 * df["subtitle_span_relevance_topk"].to_numpy(np.float32)
        + 0.25 * bmn
        + 0.15 * t2
        + 0.10 * dur_prior
    ).astype(np.float32)
    df["focus_span_agreement"] = (1.0 - np.abs(df["visual_span_relevance_topk"].to_numpy(np.float32) - df["subtitle_span_relevance_topk"].to_numpy(np.float32))).astype(np.float32)
    df["span_relevance_margin"] = df["prem_span_relevance"] - pd.Series(df["prem_span_relevance"]).groupby([df["split"], df["query_id"], df["seed"], df["video_id"]], observed=True).transform("mean").to_numpy(np.float32)
    p = pd.Series(df["prem_span_relevance"]).groupby([df["split"], df["query_id"], df["seed"], df["video_id"]], observed=True).transform(lambda x: x / max(float(x.sum()), 1e-6)).to_numpy(np.float32)
    df["span_relevance_entropy_local"] = (-p * np.log(np.maximum(p, 1e-8))).astype(np.float32)
    return {
        "visual": "C26 visual relevance projected to spans with actionness",
        "subtitle": "C26 subtitle relevance projected to spans with T2 score",
        "no_gt_inference_features": "candidate_iou/iou labels are not used",
    }


def span_score_variants(df: pd.DataFrame) -> Dict[str, np.ndarray]:
    bmn = _norm_by_group(df, "bmn_final_score")
    t2 = _norm_by_group(df, "t2_score")
    prem = df["prem_span_relevance"].to_numpy(np.float32)
    focus = df["focus_span_agreement"].to_numpy(np.float32)
    conflict = ((t2 > 0.85) & (bmn < 0.25)).astype(np.float32)
    return {
        "S0_c26r_guarded_baseline": df["R3b1_C26_PREM_guarded_localizer"].to_numpy(np.float32),
        "S1_prem_span_only": prem,
        "S2_prem_bmn_t2": 0.42 * prem + 0.34 * bmn + 0.24 * t2,
        "S3_focus_consistency": 0.40 * prem + 0.30 * bmn + 0.20 * t2 + 0.10 * focus,
        "S4_false_positive_guard": 0.42 * prem + 0.32 * bmn + 0.22 * t2 + 0.08 * focus - 0.12 * conflict,
    }

