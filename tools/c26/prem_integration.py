from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from tools.c26.prem_eval import evaluate_vr_vcmr, metric_delta, selection_score, zscore_by_query
from tools.c26.prem_feature_dataset import C26_CACHE, write_json


C24H_CACHE = Path("/tmp/c24h_score_cache/CONQUER-RLEM-c2c3")


def attach_scores(df_split: pd.DataFrame, scores: Dict[str, np.ndarray]) -> pd.DataFrame:
    out = df_split.copy().reset_index(drop=True)
    if len(out) != len(next(iter(scores.values()), [])):
        raise ValueError(f"score length mismatch: rows={len(out)} scores={len(next(iter(scores.values()), []))}")
    for key, vals in scores.items():
        out[f"c26_{key}"] = vals
    out["first_stage_z"] = zscore_by_query(out, "first_stage_score")
    out["c26_prem_z"] = zscore_by_query(out, "c26_prem_score")
    out["c26_residual_z"] = zscore_by_query(out, "c26_residual")
    return out


def load_c24h_aux_for(df: pd.DataFrame) -> pd.DataFrame:
    path = C24H_CACHE / "C24H_FULL_EVIDENCE_TABLE_full.local.parquet"
    if not path.exists():
        return df
    qids = set(int(x) for x in df["query_id"].unique().tolist())
    cols = [
        "query_id", "video_id", "bmn_final_score", "t2_score", "event_relevance_score",
        "best_bmn_span_start", "best_bmn_span_end", "best_event_start", "best_event_end",
    ]
    try:
        aux = pd.read_parquet(path, columns=[c for c in cols if c])
    except Exception:
        return df
    aux = aux[aux["query_id"].astype(int).isin(qids)].copy()
    if len(aux) == 0:
        return df
    keep_cols = [c for c in cols if c in aux.columns]
    merged = df.merge(aux[keep_cols], on=["query_id", "video_id"], how="left", validate="one_to_one")
    for c in ["bmn_final_score", "t2_score", "event_relevance_score"]:
        if c in merged.columns:
            merged[f"{c}_z"] = zscore_by_query(merged, c)
        else:
            merged[f"{c}_z"] = 0.0
    if "best_bmn_span_start" not in merged.columns:
        merged["best_bmn_span_start"] = np.nan
        merged["best_bmn_span_end"] = np.nan
    return merged


def formula_space() -> List[Dict[str, Any]]:
    return [
        {"name": "A_C23_R0_first_stage_only", "alpha": 1.0, "beta": 0.0, "gamma": 0.0, "delta": 0.0, "eta": 0.0},
        {"name": "B_C19_C20_C21_hybrid_proxy", "alpha": 1.0, "beta": 0.0, "gamma": 0.0, "delta": 0.03, "eta": 0.05},
        {"name": "C_C24H_full_evidence_route", "alpha": 1.0, "beta": 0.0, "gamma": 0.0, "delta": 0.06, "eta": 0.10},
        {"name": "D_C26_PREM_retriever_only", "alpha": 1.0, "beta": 0.18, "gamma": 0.0, "delta": 0.0, "eta": 0.0},
        {"name": "E_visual_focus_only", "alpha": 1.0, "beta": 0.10, "gamma": 0.08, "focus": "visual", "delta": 0.0, "eta": 0.0},
        {"name": "F_subtitle_focus_only", "alpha": 1.0, "beta": 0.10, "gamma": 0.08, "focus": "subtitle", "delta": 0.0, "eta": 0.0},
        {"name": "G_visual_subtitle_focus", "alpha": 1.0, "beta": 0.14, "gamma": 0.12, "focus": "vs", "delta": 0.0, "eta": 0.0},
        {"name": "H_visual_subtitle_event", "alpha": 1.0, "beta": 0.12, "gamma": 0.10, "focus": "vs", "delta": 0.06, "eta": 0.0},
        {"name": "I_visual_subtitle_BMN_T2", "alpha": 1.0, "beta": 0.12, "gamma": 0.10, "focus": "vs", "delta": 0.0, "eta": 0.08},
        {"name": "J_full_focus_fuse_no_regression", "alpha": 1.0, "beta": 0.12, "gamma": 0.10, "focus": "vs", "delta": 0.04, "eta": 0.06, "no_regression": True},
    ]


def apply_formula(df: pd.DataFrame, formula: Dict[str, Any]) -> np.ndarray:
    base = df["first_stage_z"].to_numpy(np.float32) * float(formula.get("alpha", 1.0))
    prem = df["c26_residual"].to_numpy(np.float32) * float(formula.get("beta", 0.0))
    focus_name = formula.get("focus")
    if focus_name == "visual":
        focus = zscore_by_query(df, "c26_visual_relevance")
    elif focus_name == "subtitle":
        focus = zscore_by_query(df, "c26_subtitle_relevance")
    elif focus_name == "vs":
        tmp = df.copy()
        tmp["_vs_focus"] = pd.to_numeric(tmp["c26_visual_relevance"], errors="coerce").fillna(0.0) + pd.to_numeric(tmp["c26_subtitle_relevance"], errors="coerce").fillna(0.0)
        focus = zscore_by_query(tmp, "_vs_focus")
    else:
        focus = np.zeros(len(df), dtype=np.float32)
    event = df.get("event_relevance_score_z", pd.Series(np.zeros(len(df)))).to_numpy(np.float32) * float(formula.get("delta", 0.0))
    bmn = df.get("bmn_final_score_z", pd.Series(np.zeros(len(df)))).to_numpy(np.float32)
    t2 = df.get("t2_score_z", pd.Series(np.zeros(len(df)))).to_numpy(np.float32)
    aux = (0.7 * bmn + 0.3 * t2) * float(formula.get("eta", 0.0))
    score = base + prem + float(formula.get("gamma", 0.0)) * focus + event + aux
    if formula.get("no_regression"):
        score = np.maximum(score, base - 0.05)
    return score.astype(np.float32)


def select_and_eval(select_df: pd.DataFrame, holdout_df: pd.DataFrame) -> Tuple[Dict[str, Any], Dict[str, Any], List[Dict[str, Any]]]:
    select_df = load_c24h_aux_for(select_df)
    holdout_df = load_c24h_aux_for(holdout_df)
    baseline_select = evaluate_vr_vcmr(select_df.assign(_score=select_df["first_stage_z"]), "_score", "best_bmn_span_start", "best_bmn_span_end")
    baseline_holdout = evaluate_vr_vcmr(holdout_df.assign(_score=holdout_df["first_stage_z"]), "_score", "best_bmn_span_start", "best_bmn_span_end")
    rows: List[Dict[str, Any]] = []
    best = None
    best_obj = -1e18
    best_select_metrics: Dict[str, Any] = {}
    for formula in formula_space():
        sdf = select_df.copy()
        sdf["_formula_score"] = apply_formula(sdf, formula)
        m = evaluate_vr_vcmr(sdf, "_formula_score", "best_bmn_span_start", "best_bmn_span_end")
        obj = selection_score(m, baseline_select)
        rec = {"formula": formula, "select_metrics": m, "objective": obj, "delta_vs_baseline_select": metric_delta(m, baseline_select)}
        rows.append(rec)
        # A/B/C are mandatory comparison baselines, not valid C26 selections:
        # C26 must select a PREM-style formula with partial relevance present.
        selectable = str(formula.get("name", ""))[0] not in {"A", "B", "C"}
        if selectable and obj > best_obj:
            best_obj = obj
            best = formula
            best_select_metrics = m
    assert best is not None
    hdf = holdout_df.copy()
    hdf["_formula_score"] = apply_formula(hdf, best)
    hold_metrics = evaluate_vr_vcmr(hdf, "_formula_score", "best_bmn_span_start", "best_bmn_span_end")
    selected = {
        "selected_formula": best,
        "select_objective": best_obj,
        "select_metrics": best_select_metrics,
        "holdout_metrics": hold_metrics,
        "baseline_select": baseline_select,
        "baseline_holdout": baseline_holdout,
        "delta_vs_baseline_holdout": metric_delta(hold_metrics, baseline_holdout),
        "c24h_aux_used": any(float(best.get(k, 0.0)) != 0.0 for k in ["delta", "eta"]),
    }
    return selected, {"select_scored_rows": int(len(select_df)), "holdout_scored_rows": int(len(holdout_df))}, rows


def save_score_sample(df: pd.DataFrame, path: Path, n: int = 4000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = [c for c in df.columns if not c.startswith("_")]
    df[cols].head(n).to_parquet(path, index=False)
