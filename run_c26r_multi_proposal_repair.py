#!/usr/bin/env python3
"""C26R multi-proposal localizer repair audit.

The C26 sanity check showed that VCMR R@100 is dominated by a one-span
materialization bottleneck. This runner restores the C17 multi-span proposal
space for the exact C26 medium query set and tests several repair routes
without using official validation or pseudo-official selection.
"""

from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import torch

import run_c17_bmn_t2_native_vcmr_integration as c17
from run_c26_prem_faithful_release_feature_processing import filtered_df_for_arrays
from tools.c26.prem_feature_dataset import C26_CACHE, build_canonical_table, build_training_arrays, jsonable, stable_hash
from tools.c26.prem_integration import apply_formula as c26_apply_formula
from tools.c26.prem_integration import attach_scores
from tools.c26.prem_train import score_arrays


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "c26r_multi_proposal_repair"
CACHE = Path("/tmp/c26r_score_cache/CONQUER-RLEM-c2c3")
C17_TABLE = Path("/tmp/c17_score_cache/CONQUER-RLEM-c2c3/C17_2_SCORE_TABLE_medium.local.parquet")
MODE = "medium"
SEED = 2026
SPLITS = ["calib_select", "calib_holdout"]
SELECT_SPLIT = "calib_select"
HOLDOUT_SPLIT = "calib_holdout"


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(obj), indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def shaish(obj: Any) -> str:
    return stable_hash(jsonable(obj))


def load_c26_scored() -> Tuple[pd.DataFrame, Dict[str, Any]]:
    df, manifest = build_canonical_table(MODE, force=False)
    arrays_path, arrays_manifest = build_training_arrays(df, MODE, force=False)
    ckpt = C26_CACHE / f"C26_PREM_RETRIEVER_{MODE}_seed{SEED}.local.pt"
    frames: List[pd.DataFrame] = []
    final_formula = {"name": "F_subtitle_focus_only", "alpha": 1.0, "beta": 0.10, "gamma": 0.08, "focus": "subtitle", "delta": 0.0, "eta": 0.0}
    for split in SPLITS:
        scores = score_arrays(arrays_path, ckpt, split, batch_size=4096, device="cuda:0")
        sdf = attach_scores(filtered_df_for_arrays(df, split), scores)
        sdf["c26_retriever_score"] = sdf["first_stage_z"] + sdf["c26_residual"]
        sdf["c26_final_F_score"] = c26_apply_formula(sdf, final_formula)
        keep = [
            "split", "query_id", "video_id", "candidate_video_rank", "first_stage_score",
            "first_stage_z", "c26_retriever_score", "c26_final_F_score",
            "c26_visual_relevance", "c26_subtitle_relevance", "c26_residual",
        ]
        frames.append(sdf[keep].copy())
    out = pd.concat(frames, ignore_index=True)
    audit = {
        "rows": int(len(out)),
        "queries_by_split": {s: int(out[out["split"] == s]["query_id"].nunique()) for s in SPLITS},
        "checkpoint": str(ckpt),
        "canonical_manifest_hash": shaish(manifest),
        "arrays_manifest_hash": shaish(arrays_manifest),
        "formula": final_formula,
    }
    return out, audit


def load_c17_multi_span(c26_scored: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    qids_by_split = {s: set(int(x) for x in c26_scored[c26_scored["split"] == s]["query_id"].unique()) for s in SPLITS}
    cols = [
        "split", "query_id", "seed", "video_id", "span_start", "span_end", "span_duration",
        "retriever_score", "retriever_rank", "bmn_pred_iou", "bmn_p_iou_05", "bmn_p_iou_07",
        "bmn_rank_score", "bmn_final_score", "start_prob", "end_prob", "actionness_score",
        "duration_score", "t2_score", "old_c12_score", "span_map_rank", "gt_video_id",
        "gt_start", "gt_end", "query_type", "duration_bucket", "gt_video_rank",
    ]
    t0 = time.time()
    raw = pd.read_parquet(C17_TABLE, columns=cols)
    raw = raw[(raw["seed"].astype(int) == SEED) & raw["split"].isin(SPLITS)].copy()
    mask = np.zeros(len(raw), dtype=bool)
    for split, qids in qids_by_split.items():
        mask |= (raw["split"].astype(str).to_numpy() == split) & raw["query_id"].astype(int).isin(qids).to_numpy()
    raw = raw[mask].copy()
    for col in ["split", "query_type", "duration_bucket"]:
        raw[col] = raw[col].astype("category")
    for col in ["query_id", "seed", "retriever_rank", "gt_video_rank", "span_map_rank"]:
        raw[col] = pd.to_numeric(raw[col], errors="coerce").fillna(-1).astype(np.int32)
    for col in [
        "span_start", "span_end", "span_duration", "retriever_score", "bmn_pred_iou",
        "bmn_p_iou_05", "bmn_p_iou_07", "bmn_rank_score", "bmn_final_score", "start_prob",
        "end_prob", "actionness_score", "duration_score", "t2_score", "old_c12_score",
        "gt_start", "gt_end",
    ]:
        raw[col] = pd.to_numeric(raw[col], errors="coerce").astype(np.float32)
    audit = {
        "source_path": str(C17_TABLE),
        "load_seconds": time.time() - t0,
        "rows": int(len(raw)),
        "queries_by_split": {s: int(raw[raw["split"].astype(str) == s]["query_id"].nunique()) for s in SPLITS},
        "seed": SEED,
        "official_used": False,
    }
    return raw, audit


def add_c26_scores_to_spans(c17_df: pd.DataFrame, c26_scored: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    keep = [
        "split", "query_id", "video_id", "candidate_video_rank", "first_stage_z",
        "c26_retriever_score", "c26_final_F_score", "c26_visual_relevance",
        "c26_subtitle_relevance", "c26_residual",
    ]
    merged = c17_df.merge(c26_scored[keep], on=["split", "query_id", "video_id"], how="left", validate="many_to_one")
    missing = int(merged["c26_final_F_score"].isna().sum())
    merged["c26_retriever_score"] = pd.to_numeric(merged["c26_retriever_score"], errors="coerce").fillna(0.0).astype(np.float32)
    merged["c26_final_F_score"] = pd.to_numeric(merged["c26_final_F_score"], errors="coerce").fillna(0.0).astype(np.float32)
    merged["c26_visual_relevance"] = pd.to_numeric(merged["c26_visual_relevance"], errors="coerce").fillna(0.0).astype(np.float32)
    merged["c26_subtitle_relevance"] = pd.to_numeric(merged["c26_subtitle_relevance"], errors="coerce").fillna(0.0).astype(np.float32)
    merged["c26_residual"] = pd.to_numeric(merged["c26_residual"], errors="coerce").fillna(0.0).astype(np.float32)
    audit = {
        "merged_rows": int(len(merged)),
        "missing_c26_score_rows": missing,
        "missing_rate": float(missing / max(1, len(merged))),
        "duplicate_c17_span_keys": int(merged.duplicated(["split", "query_id", "seed", "video_id", "span_start", "span_end"]).sum()),
    }
    return merged, audit


def add_norms(df: pd.DataFrame) -> Dict[str, Any]:
    audit = c17.add_normalized_scores_inplace(df)
    # C26 video scores are already query-normalized, but each span row repeats a
    # video-level score. Re-normalize inside the C17 query/seed row-space so it
    # is comparable to C17 localizer columns.
    for col, dst in [
        ("c26_retriever_score", "c26_retriever_norm"),
        ("c26_final_F_score", "c26_final_norm"),
        ("c26_visual_relevance", "c26_visual_norm"),
        ("c26_subtitle_relevance", "c26_subtitle_norm"),
    ]:
        vals = df[col].astype(np.float32)
        gb = vals.groupby([df["split"], df["query_id"], df["seed"]], observed=True)
        mins = gb.transform("min").astype(np.float32)
        maxs = gb.transform("max").astype(np.float32)
        denom = (maxs - mins).replace(0.0, np.nan).astype(np.float32)
        df[dst] = ((vals - mins) / denom).replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float32)
        audit[col] = {"missing_count": int(vals.isna().sum()), "fill_policy": "C26 repeated video score renormalized inside C17 query/seed row-space"}
    return audit


def iou_array(df: pd.DataFrame, start_col: str = "span_start", end_col: str = "span_end") -> np.ndarray:
    s = df[start_col].to_numpy(np.float32)
    e = df[end_col].to_numpy(np.float32)
    gs = df["gt_start"].to_numpy(np.float32)
    ge = df["gt_end"].to_numpy(np.float32)
    same = df["video_id"].astype(str).to_numpy() == df["gt_video_id"].astype(str).to_numpy()
    inter = np.maximum(0.0, np.minimum(e, ge) - np.maximum(s, gs))
    union = np.maximum(e, ge) - np.minimum(s, gs)
    iou = inter / np.maximum(union, 1e-6)
    return np.where(same, iou, 0.0).astype(np.float32)


def assign_scores(df: pd.DataFrame, formulas: Dict[str, np.ndarray]) -> None:
    for name, arr in formulas.items():
        df[name] = np.asarray(arr, dtype=np.float32)


def score_formulas(df: pd.DataFrame) -> Dict[str, np.ndarray]:
    retr = df["retriever_norm"].to_numpy(np.float32)
    bmn = df["bmn_norm"].to_numpy(np.float32)
    t2 = df["t2_norm"].to_numpy(np.float32)
    old = df["old_norm"].to_numpy(np.float32)
    c26r = df["c26_retriever_norm"].to_numpy(np.float32)
    c26f = df["c26_final_norm"].to_numpy(np.float32)
    c26v = df["c26_visual_norm"].to_numpy(np.float32)
    c26s = df["c26_subtitle_norm"].to_numpy(np.float32)
    dur = df["duration_score"].fillna(0.0).to_numpy(np.float32)
    conflict = ((t2 > 0.85) & (bmn < 0.25)).astype(np.float32)
    agree = ((bmn > 0.35) & (t2 > 0.35)).astype(np.float32)
    return {
        "R1_zero_delta_c17_retriever_multi_span": retr,
        "R1b_zero_delta_c26_retriever_multi_span": c26r,
        "R1c_zero_delta_c26_final_video_multi_span": c26f,
        "R2a_BMN_only": bmn,
        "R2a1_BMN_plus_retriever": 0.45 * retr + 0.55 * bmn,
        "R2b_T2_only": t2,
        "R2b1_T2_plus_retriever": 0.45 * retr + 0.55 * t2,
        "R2c_C17_BMN_T2_hybrid": 0.40 * retr + 0.36 * bmn + 0.24 * t2,
        "R2c1_C17_hybrid_conflict_guard": 0.42 * retr + 0.34 * bmn + 0.24 * t2 - 0.12 * conflict,
        "R3a_C26_final_plus_BMN": 0.50 * c26f + 0.50 * bmn,
        "R3a1_C26_final_plus_BMN_T2": 0.44 * c26f + 0.34 * bmn + 0.22 * t2,
        "R3b_C26_PREM_plus_C17_localizer": 0.42 * c26r + 0.34 * bmn + 0.18 * t2 + 0.06 * (c26v + c26s),
        "R3b1_C26_PREM_guarded_localizer": 0.45 * c26r + 0.32 * bmn + 0.20 * t2 + 0.06 * agree - 0.10 * conflict,
        "R3c_duration_aware_C26_BMN_T2": 0.42 * c26f + 0.32 * bmn + 0.18 * t2 + 0.08 * dur,
    }


def evaluate_method(df: pd.DataFrame, score_col: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for split in SPLITS:
        sdf = df[df["split"].astype(str) == split]
        out[split] = c17.evaluate_vcmr(sdf, score_col)
    return out


def select_by_calib(results: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    best_name = ""
    best_obj = -1e18
    best_hold: Dict[str, Any] = {}
    for name, rec in results.items():
        if name.startswith("R6_oracle"):
            rec["selection_objective"] = None
            rec["selection_excluded_reason"] = "diagnostic_only_oracle_uses_gt_iou"
            continue
        sel = rec[SELECT_SPLIT]["summary"]
        obj = (
            4.0 * float(sel.get("VCMR_R@10_IoU0.5", 0.0))
            + 2.0 * float(sel.get("VCMR_R@10_IoU0.7", 0.0))
            + 1.0 * float(sel.get("VCMR_R@100_IoU0.5", 0.0))
            + 0.5 * float(sel.get("VCMR_R@100_IoU0.7", 0.0))
            + 0.5 * float(sel.get("VR_R@100", 0.0))
            - 0.5 * float(sel.get("wrong_video_high_score_rate", 0.0))
        )
        rec["selection_objective"] = obj
        if obj > best_obj:
            best_obj = obj
            best_name = name
            best_hold = rec[HOLDOUT_SPLIT]["summary"]
    return best_name, best_hold


def evaluate_all_methods(df: pd.DataFrame, names: Iterable[str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for name in names:
        print(f"[C26R] evaluating {name}", flush=True)
        out[name] = evaluate_method(df, name)
    return out


def limit_topm_per_video(df: pd.DataFrame, score_col: str, m: int) -> pd.DataFrame:
    tmp = df.copy()
    tmp["_localizer_rank_for_topm"] = tmp.groupby(["split", "query_id", "seed", "video_id"], observed=True)[score_col].rank(method="first", ascending=False)
    out = tmp[tmp["_localizer_rank_for_topm"] <= int(m)].copy()
    return out.drop(columns=["_localizer_rank_for_topm"])


def unit_variant_df(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    out = df.copy()
    if mode == "seconds":
        return out
    if mode == "times_1p5":
        out["span_start"] = out["span_start"] * 1.5
        out["span_end"] = out["span_end"] * 1.5
    elif mode == "div_1p5":
        out["span_start"] = out["span_start"] / 1.5
        out["span_end"] = out["span_end"] / 1.5
    elif mode == "round_1p5_grid":
        out["span_start"] = np.round(out["span_start"] / 1.5) * 1.5
        out["span_end"] = np.maximum(out["span_start"] + 0.01, np.round(out["span_end"] / 1.5) * 1.5)
    else:
        raise ValueError(mode)
    return out


def proposal_audit(df: pd.DataFrame) -> Dict[str, Any]:
    span_counts = df.groupby(["split", "query_id", "seed", "video_id"], observed=True).size()
    q_counts = df.groupby(["split", "query_id", "seed"], observed=True).size()
    gt = df[df["video_id"].astype(str) == df["gt_video_id"].astype(str)].copy()
    gt_iou = iou_array(gt) if len(gt) else np.zeros((0,), dtype=np.float32)
    return {
        "rows": int(len(df)),
        "query_count": int(df[["split", "query_id", "seed"]].drop_duplicates().shape[0]),
        "query_video_pair_count": int(df[["split", "query_id", "seed", "video_id"]].drop_duplicates().shape[0]),
        "span_count_per_query_video": {
            "min": int(span_counts.min()),
            "median": float(span_counts.median()),
            "p95": float(span_counts.quantile(0.95)),
            "max": int(span_counts.max()),
        },
        "span_rows_per_query": {
            "min": int(q_counts.min()),
            "median": float(q_counts.median()),
            "p95": float(q_counts.quantile(0.95)),
            "max": int(q_counts.max()),
        },
        "gt_video_span_oracle_coverage": {
            "gt_video_rows": int(len(gt)),
            "iou05_span_rows": int((gt_iou >= 0.5).sum()),
            "iou07_span_rows": int((gt_iou >= 0.7).sum()),
        },
    }


def route_interpretation(results: Dict[str, Any], best_name: str, single_span_r100: float) -> Dict[str, Any]:
    def hold(name: str, key: str) -> float:
        return float(results.get(name, {}).get(HOLDOUT_SPLIT, {}).get("summary", {}).get(key, 0.0))
    notes: Dict[str, Any] = {}
    notes["route1_zero_delta"] = {
        "tested": ["R1_zero_delta_c17_retriever_multi_span", "R1b_zero_delta_c26_retriever_multi_span", "R1c_zero_delta_c26_final_video_multi_span"],
        "failure_or_success": "multi-proposal without span-localizer remains weak if VCMR R@100 does not exceed single-span by a large margin",
        "holdout_r100": {n: hold(n, "VCMR_R@100_IoU0.5") for n in ["R1_zero_delta_c17_retriever_multi_span", "R1b_zero_delta_c26_retriever_multi_span", "R1c_zero_delta_c26_final_video_multi_span"]},
    }
    notes["route2_original_c17_localizer"] = {
        "tested": ["R2a_BMN_only", "R2a1_BMN_plus_retriever", "R2b_T2_only", "R2b1_T2_plus_retriever", "R2c_C17_BMN_T2_hybrid", "R2c1_C17_hybrid_conflict_guard"],
        "conclusion": "restoring original multi-span localizer/proposal space is beneficial" if max(hold(n, "VCMR_R@100_IoU0.5") for n in ["R2a1_BMN_plus_retriever", "R2b1_T2_plus_retriever", "R2c_C17_BMN_T2_hybrid", "R2c1_C17_hybrid_conflict_guard"]) > single_span_r100 + 5.0 else "C17 localizer did not recover enough; inspect proposal/evaluator alignment",
    }
    notes["route3_c26_with_multi_span"] = {
        "tested": ["R3a_C26_final_plus_BMN", "R3a1_C26_final_plus_BMN_T2", "R3b_C26_PREM_plus_C17_localizer", "R3b1_C26_PREM_guarded_localizer", "R3c_duration_aware_C26_BMN_T2"],
        "best_name": best_name if best_name.startswith("R3") else None,
        "best_holdout_r100": hold(best_name, "VCMR_R@100_IoU0.5") if best_name.startswith("R3") else None,
    }
    notes["route6_oracle_headroom"] = {
        "diagnostic_only": True,
        "excluded_from_selection": True,
        "holdout_r100": hold("R6_oracle_span_upper_bound_diagnostic", "VCMR_R@100_IoU0.5"),
        "meaning": "Upper bound using GT IoU labels; shows proposal-space headroom only, never an inference route.",
    }
    return notes


def main() -> None:
    os.environ.setdefault("OMP_NUM_THREADS", str(min(32, os.cpu_count() or 1)))
    torch.set_num_threads(min(32, os.cpu_count() or 1))
    OUT.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    local_path = CACHE / "C26R_C17_C26_JOINED_MULTI_SPAN_medium_seed2026.local.parquet"
    source_audit_path = OUT / "C26R_0_SOURCE_AND_PROPOSAL_AUDIT.json"
    if local_path.exists() and source_audit_path.exists():
        print("[C26R] resuming from joined multi-span table", flush=True)
        df = pd.read_parquet(local_path)
        formulas = {k: df[k].to_numpy(np.float32) for k in score_formulas(df).keys() if k in df.columns}
        source_audit = json.loads(source_audit_path.read_text(encoding="utf-8"))
    else:
        print("[C26R] loading C26 scores", flush=True)
        c26_scored, c26_audit = load_c26_scored()
        print("[C26R] loading C17 multi-span table", flush=True)
        c17_df, c17_audit = load_c17_multi_span(c26_scored)
        print("[C26R] joining C26 scores", flush=True)
        df, join_audit = add_c26_scores_to_spans(c17_df, c26_scored)
        norm_audit = add_norms(df)
        df["oracle_iou_score"] = iou_array(df)
        formulas = score_formulas(df)
        assign_scores(df, formulas)
        df["R6_oracle_span_upper_bound_diagnostic"] = np.where(df["oracle_iou_score"] >= 0.7, 10.0, np.where(df["oracle_iou_score"] >= 0.5, 8.0, df["retriever_norm"] * 0.01)).astype(np.float32)
        df.to_parquet(local_path, index=False)
        source_audit = {
            "c26": c26_audit,
            "c17": c17_audit,
            "join": join_audit,
            "norm": norm_audit,
            "local_joined_path": str(local_path),
            "local_joined_size_bytes": local_path.stat().st_size,
            "proposal_audit": proposal_audit(df),
            "official_val_used": False,
            "official_prediction_pool_used": False,
            "pseudo_official_holdout_used_for_selection": False,
        }
    write_json(OUT / "C26R_0_SOURCE_AND_PROPOSAL_AUDIT.json", source_audit)
    write_text(OUT / "C26R_0_SOURCE_AND_PROPOSAL_AUDIT.md", "# C26R Source And Proposal Audit\n\nC26R restores C17 medium multi-span proposals for the exact C26 medium query set. Official validation and pseudo-official selection are not used.")

    method_names = list(formulas.keys()) + ["R6_oracle_span_upper_bound_diagnostic"]
    print("[C26R] evaluating main repair routes", flush=True)
    results = evaluate_all_methods(df, method_names)
    best_name, best_hold = select_by_calib(results)
    single_span = json.loads((ROOT / "c26_7_vcmr_r100_sanity_check/C26_7_VCMR_R100_SANITY_AUDIT.json").read_text(encoding="utf-8"))
    single_span_r100 = float(single_span["same_subset_metrics"]["c26_final_F"]["VCMR_R@100_IoU0.5"])
    route_notes = route_interpretation(results, best_name, single_span_r100)
    write_json(OUT / "C26R_1_METHOD_RESULTS.json", {"method_results": results, "best_by_calib_select": best_name, "best_holdout": best_hold, "single_span_c26_r100_iou05": single_span_r100, "route_notes": route_notes})

    print("[C26R] evaluating topM proposal variants", flush=True)
    topm_results: Dict[str, Any] = {}
    topm_base = best_name if best_name != "R6_oracle_span_upper_bound_diagnostic" else "R3b1_C26_PREM_guarded_localizer"
    for m in [1, 2, 4, 8, 16, 32, 64]:
        limited = limit_topm_per_video(df, topm_base, m)
        topm_results[f"top{m}_per_video_{topm_base}"] = evaluate_method(limited, topm_base)
    write_json(OUT / "C26R_2_TOPM_PROPOSAL_RESULTS.json", {"base_method": topm_base, "results": topm_results})

    print("[C26R] evaluating span unit variants", flush=True)
    unit_results: Dict[str, Any] = {}
    for mode in ["seconds", "times_1p5", "div_1p5", "round_1p5_grid"]:
        u = unit_variant_df(df, mode)
        unit_results[mode] = evaluate_method(u, topm_base)
    write_json(OUT / "C26R_3_SPAN_UNIT_RESULTS.json", {"base_method": topm_base, "results": unit_results})

    sample_cols = [
        "split", "query_id", "seed", "video_id", "span_start", "span_end", "gt_video_id", "gt_start", "gt_end",
        "retriever_rank", "span_map_rank", "retriever_score", "bmn_final_score", "t2_score", "c26_final_F_score",
        topm_base, "oracle_iou_score",
    ]
    sample = df[df["split"].astype(str) == HOLDOUT_SPLIT].sort_values(topm_base, ascending=False)[sample_cols].head(5000)
    sample_path = OUT / "C26R_REPAIR_SCORE_SAMPLE.parquet"
    sample.to_parquet(sample_path, index=False)

    selected_summary = {
        "stage": "C26R",
        "status": "C26R_MULTI_PROPOSAL_REPAIR_PROMISING" if best_hold.get("VCMR_R@100_IoU0.5", 0.0) > single_span_r100 + 5.0 else "C26R_MULTI_PROPOSAL_REPAIR_NOT_ENOUGH",
        "best_by_calib_select": best_name,
        "best_holdout": best_hold,
        "single_span_c26_r100_iou05": single_span_r100,
        "delta_best_vs_single_span_r100_iou05": float(best_hold.get("VCMR_R@100_IoU0.5", 0.0) - single_span_r100),
        "route_notes": route_notes,
        "topm_base_method": topm_base,
        "topm_results": {k: v[HOLDOUT_SPLIT]["summary"] for k, v in topm_results.items()},
        "unit_results": {k: v[HOLDOUT_SPLIT]["summary"] for k, v in unit_results.items()},
        "sample_path": str(sample_path),
        "local_large_artifacts": {"joined_multi_span_table": str(local_path)},
        "runtime_seconds": time.time() - t0,
        "official_val_used": False,
        "official_prediction_pool_used": False,
        "pseudo_official_holdout_used_for_selection": False,
        "evaluator_modified": False,
        "nms_modified": False,
        "c26r_promoted_system": False,
    }
    write_json(OUT / "C26R_FINAL_DECISION.json", selected_summary)
    write_text(
        OUT / "C26R_FINAL_DECISION.md",
        "# C26R Final Decision\n\n"
        f"Status: `{selected_summary['status']}`.\n\n"
        f"Best method by calib_select: `{best_name}`.\n\n"
        f"Holdout VCMR R@100@0.5: `{best_hold.get('VCMR_R@100_IoU0.5')}` versus C26 single-span `{single_span_r100}`.\n\n"
        "C26R does not use official validation or pseudo-official selection and does not promote a system.\n",
    )
    print(json.dumps(jsonable({
        "status": selected_summary["status"],
        "best": best_name,
        "holdout_r100_iou05": best_hold.get("VCMR_R@100_IoU0.5"),
        "single_span_r100_iou05": single_span_r100,
        "runtime_seconds": selected_summary["runtime_seconds"],
    }), indent=2), flush=True)


if __name__ == "__main__":
    main()
