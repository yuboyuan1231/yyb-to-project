from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd


def add_feedback_features(df: pd.DataFrame, span_score_col: str) -> pd.DataFrame:
    keys = ["split", "query_id", "seed", "video_id"]
    work = df[keys + [span_score_col, "prem_span_relevance", "focus_span_agreement", "bmn_final_score", "t2_score"]].copy()
    work["_score"] = pd.to_numeric(work[span_score_col], errors="coerce").fillna(0.0).astype(np.float32)
    work["_rank"] = work.groupby(keys, observed=True)["_score"].rank(method="first", ascending=False)
    top = work[work["_rank"] <= 8].copy()
    grp = top.groupby(keys, observed=True)
    feat = grp["_score"].agg(best_span_score="max", top8_span_mean="mean", high_score_span_count=lambda x: int((x > x.mean()).sum())).reset_index()
    top2 = top[top["_rank"] <= 2].groupby(keys, observed=True)["_score"].mean().rename("top2_span_mean").reset_index()
    top4 = top[top["_rank"] <= 4].groupby(keys, observed=True)["_score"].mean().rename("top4_span_mean").reset_index()
    lse = grp["_score"].apply(lambda x: float(np.log(np.exp(x - x.max()).sum()) + x.max())).rename("topK_span_logsumexp").reset_index()
    margin = grp["_score"].apply(lambda x: float(np.sort(x.to_numpy())[-1] - np.sort(x.to_numpy())[-2]) if len(x) >= 2 else 0.0).rename("span_score_margin").reset_index()
    entropy = grp["_score"].apply(lambda x: _entropy(x.to_numpy(np.float32))).rename("span_score_entropy").reset_index()
    agreement = grp["focus_span_agreement"].mean().rename("prem_focus_span_agreement_topK").reset_index()
    aux_agree = top.assign(_aux=1.0 - np.abs(pd.to_numeric(top["bmn_final_score"], errors="coerce").fillna(0.0) - pd.to_numeric(top["t2_score"], errors="coerce").fillna(0.0))).groupby(keys, observed=True)["_aux"].mean().rename("bmn_t2_event_agreement_topK").reset_index()
    for extra in [top2, top4, lse, margin, entropy, agreement, aux_agree]:
        feat = feat.merge(extra, on=keys, how="left")
    feat["span_score_concentration"] = (feat["best_span_score"] - feat["top8_span_mean"]).astype(np.float32)
    feat["low_confidence_span_count"] = 8 - feat["high_score_span_count"].fillna(0)
    feat["hard_negative_span_risk"] = feat["span_score_entropy"].astype(np.float32)
    feat["wrong_video_risk_score"] = feat["hard_negative_span_risk"] - feat["span_score_margin"].astype(np.float32)
    feat["span_distribution_quality"] = feat["best_span_score"] + feat["top4_span_mean"] - 0.15 * feat["span_score_entropy"]
    return df.merge(feat, on=keys, how="left", validate="many_to_one")


def _entropy(x: np.ndarray) -> float:
    if len(x) == 0:
        return 0.0
    z = x - float(np.max(x))
    p = np.exp(z)
    p = p / max(float(p.sum()), 1e-8)
    return float(-(p * np.log(np.maximum(p, 1e-8))).sum())


def feedback_variants(df: pd.DataFrame) -> Dict[str, np.ndarray]:
    base = df["c26_final_norm"].to_numpy(np.float32) if "c26_final_norm" in df else df["c26_final_F_score"].to_numpy(np.float32)
    prem = df["c26_retriever_norm"].to_numpy(np.float32) if "c26_retriever_norm" in df else base
    best = df["best_span_score"].fillna(0.0).to_numpy(np.float32)
    lse = df["topK_span_logsumexp"].fillna(0.0).to_numpy(np.float32)
    margin = df["span_score_margin"].fillna(0.0).to_numpy(np.float32)
    entropy = df["span_score_entropy"].fillna(0.0).to_numpy(np.float32)
    focus = df["prem_focus_span_agreement_topK"].fillna(0.0).to_numpy(np.float32)
    agree = df["bmn_t2_event_agreement_topK"].fillna(0.0).to_numpy(np.float32)
    risk = df["wrong_video_risk_score"].fillna(0.0).to_numpy(np.float32)
    return {
        "F0_no_feedback": base,
        "F1_best_span_only": base + 0.12 * best,
        "F2_topK_logsumexp": base + 0.10 * lse,
        "F3_margin_entropy": base + 0.14 * margin - 0.05 * entropy,
        "F4_PREM_focus_consistency": base + 0.10 * best + 0.08 * focus,
        "F5_BMN_T2_event_agreement": base + 0.10 * best + 0.08 * agree,
        "F6_wrong_risk_penalty": base + 0.12 * best - 0.08 * risk,
        "F7_full_span_to_video_feedback": base + 0.08 * prem + 0.10 * best + 0.05 * lse + 0.05 * margin + 0.04 * focus + 0.04 * agree - 0.08 * entropy - 0.06 * risk,
        "F8_no_regression_guarded_feedback": np.maximum(base - 0.03, base + 0.06 * prem + 0.10 * best + 0.04 * margin + 0.04 * focus - 0.05 * risk),
    }

