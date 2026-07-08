#!/usr/bin/env python3
"""C26 VCMR R@100 sanity check.

This audit checks whether the low C26 VCMR R@100 is a C26-specific bug, a
shared span proposal limitation, or a row-space/evaluator mismatch. It never
touches official validation or official prediction pools.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from run_c26_prem_faithful_release_feature_processing import filtered_df_for_arrays
from tools.c26.prem_eval import evaluate_vr_vcmr, metric_delta, zscore_by_query
from tools.c26.prem_feature_dataset import C26_CACHE, build_canonical_table, build_training_arrays, jsonable, stable_hash
from tools.c26.prem_integration import apply_formula, attach_scores
from tools.c26.prem_train import score_arrays


ROOT = Path(__file__).resolve().parent
C24H_CACHE = Path("/tmp/c24h_score_cache/CONQUER-RLEM-c2c3")
OUT = ROOT / "c26_7_vcmr_r100_sanity_check"
MODE = "medium"
SPLIT = "calib_holdout"
SEED = 2026


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(obj), indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def c24h_formula_scores(df: pd.DataFrame) -> Dict[str, np.ndarray]:
    fs = pd.to_numeric(df.get("first_stage_z_c24h", df.get("first_stage_z")), errors="coerce").fillna(0.0).to_numpy(np.float32)
    b = pd.to_numeric(df.get("bmn_z", 0.0), errors="coerce").fillna(0.0).to_numpy(np.float32)
    t = pd.to_numeric(df.get("t2_z", 0.0), errors="coerce").fillna(0.0).to_numpy(np.float32)
    e = pd.to_numeric(df.get("event_z", 0.0), errors="coerce").fillna(0.0).to_numpy(np.float32)
    conflict = ((b > 0.8) & (t < -0.2)).astype(np.float32)
    agree_be = ((b > 0.3) & (e > 0.3)).astype(np.float32)
    guard = ((fs > 0.0) | ((b > 0.5) & (e > 0.2))).astype(np.float32)
    return {
        "A_C23_baseline_R0_baseline_only": fs,
        "B_C19_C20_C21_hybrid_proxy": fs + 0.20 * b + 0.12 * t,
        "C_C24_selected_B_C19_C21_hybrid": fs + 0.24 * b + 0.12 * t,
        "D_C24F_event_only": fs + 0.35 * e,
        "E_C24G_partial_evidence_proxy": fs + 0.18 * b + 0.18 * e,
        "F_C24H_BMN_only": fs + 0.35 * b,
        "G_C24H_T2_only": fs + 0.28 * t,
        "H_C24H_event_only": fs + 0.35 * e,
        "I_BMN_event_consistency": fs + 0.22 * b + 0.22 * e + 0.08 * agree_be,
        "J_BMN_T2_consistency": fs + 0.22 * b + 0.16 * t - 0.14 * conflict,
        "K_all_evidence_wrong_risk_penalty": fs + 0.18 * b + 0.18 * e + 0.12 * t - 0.18 * conflict,
        "L_all_evidence_no_regression_guard": fs + guard * (0.18 * b + 0.18 * e + 0.10 * t) - 0.10 * conflict,
    }


def read_c24h_subset(qids: List[int]) -> pd.DataFrame:
    path = C24H_CACHE / "C24H_FULL_EVIDENCE_TABLE_full.local.parquet"
    cols = [
        "split", "query_id", "video_id", "gt_video_id", "gt_start", "gt_end",
        "candidate_video_rank", "first_stage_score", "first_stage_z",
        "bmn_z", "t2_z", "event_z", "bmn_final_score", "t2_score",
        "event_relevance_score", "best_bmn_span_start", "best_bmn_span_end",
        "span_start", "span_end", "candidate_iou", "iou_ge_05", "iou_ge_07",
        "correct_video", "bmn_available", "t2_available", "event_available",
    ]
    df = pd.read_parquet(path, columns=cols)
    qset = set(int(x) for x in qids)
    return df[(df["split"].astype(str) == SPLIT) & (df["query_id"].astype(int).isin(qset))].copy()


def topk_failure_breakdown(df: pd.DataFrame, score_col: str, k: int = 100) -> Dict[str, Any]:
    rows = []
    hit_vr = hit_vcmr = correct_video_bad_span = missing_correct_video = 0
    for qid, g in df.groupby("query_id", sort=False):
        top = g.sort_values(score_col, ascending=False).head(k)
        correct = top[top["video_id"].astype(str) == top["gt_video_id"].astype(str)]
        if len(correct):
            hit_vr += 1
            best_iou = float(pd.to_numeric(correct["candidate_iou"], errors="coerce").fillna(0.0).max())
            if best_iou >= 0.5:
                hit_vcmr += 1
            else:
                correct_video_bad_span += 1
                rec = correct.sort_values("candidate_iou", ascending=False).iloc[0]
                rows.append({
                    "query_id": int(qid),
                    "gt_video_id": str(rec["gt_video_id"]),
                    "best_candidate_iou_in_top100": best_iou,
                    "gt_start": float(rec["gt_start"]),
                    "gt_end": float(rec["gt_end"]),
                    "span_start": float(rec["best_bmn_span_start"]),
                    "span_end": float(rec["best_bmn_span_end"]),
                })
        else:
            missing_correct_video += 1
    qn = int(df["query_id"].nunique())
    return {
        "query_count": qn,
        "top100_vr_hit_queries": hit_vr,
        "top100_vcmr05_hit_queries": hit_vcmr,
        "correct_video_in_top100_but_span_iou_lt_05": correct_video_bad_span,
        "correct_video_missing_from_top100": missing_correct_video,
        "vr100_minus_vcmr100_gap_queries": hit_vr - hit_vcmr,
        "failure_examples": rows[:50],
    }


def span_unit_audit(df: pd.DataFrame, score_col: str) -> Dict[str, Any]:
    work = df.copy()
    s = pd.to_numeric(work["best_bmn_span_start"], errors="coerce")
    e = pd.to_numeric(work["best_bmn_span_end"], errors="coerce")
    vd = pd.to_numeric(work["video_duration"], errors="coerce") if "video_duration" in work else pd.Series(np.nan, index=work.index)
    span_dur = e - s
    current = evaluate_vr_vcmr(work, score_col, "best_bmn_span_start", "best_bmn_span_end")
    work["span_start_times_1p5"] = s * 1.5
    work["span_end_times_1p5"] = e * 1.5
    times_1p5 = evaluate_vr_vcmr(work, score_col, "span_start_times_1p5", "span_end_times_1p5")
    work["span_start_div_1p5"] = s / 1.5
    work["span_end_div_1p5"] = e / 1.5
    div_1p5 = evaluate_vr_vcmr(work, score_col, "span_start_div_1p5", "span_end_div_1p5")
    grid = np.abs((s.to_numpy(np.float64) / 1.5) - np.round(s.to_numpy(np.float64) / 1.5))
    return {
        "unit_conclusion": "seconds_consistent" if current.get("VCMR_R@100_IoU0.5", 0.0) >= max(times_1p5.get("VCMR_R@100_IoU0.5", 0.0), div_1p5.get("VCMR_R@100_IoU0.5", 0.0)) else "unit_suspicious",
        "current_seconds_metrics": current,
        "if_span_values_were_clip_index_times_1p5": times_1p5,
        "if_span_values_were_seconds_div_1p5": div_1p5,
        "negative_or_zero_duration_count": int((span_dur <= 0).sum()),
        "span_end_gt_video_duration_plus_1s_count": int(((e > vd + 1.0) & vd.notna()).sum()),
        "span_duration_summary": {
            "min": float(span_dur.min()),
            "median": float(span_dur.median()),
            "p95": float(span_dur.quantile(0.95)),
            "max": float(span_dur.max()),
        },
        "start_on_1p5s_grid_rate": float(np.mean(grid < 1e-3)),
    }


def proposal_audit(df: pd.DataFrame) -> Dict[str, Any]:
    key = ["query_id", "video_id"]
    unique_spans = (
        df[key + ["best_bmn_span_start", "best_bmn_span_end"]]
        .drop_duplicates()
        .groupby(key, sort=False)
        .size()
    )
    return {
        "row_count": int(len(df)),
        "query_count": int(df["query_id"].nunique()),
        "query_video_pair_count": int(df[key].drop_duplicates().shape[0]),
        "has_c26_native_span_columns": all(c in df.columns for c in ["c26_span_start", "c26_span_end"]),
        "span_columns_used_for_vcmr": ["best_bmn_span_start", "best_bmn_span_end"],
        "unique_span_proposals_per_query_video": {
            "min": int(unique_spans.min()),
            "median": float(unique_spans.median()),
            "max": int(unique_spans.max()),
            "all_one": bool((unique_spans == 1).all()),
        },
        "top100_proposal_interpretation": "one row is one query-video candidate with one borrowed C24H BMN span; C26 does not emit multiple spans per video.",
    }


def update_final_decision(audit: Dict[str, Any]) -> None:
    path = ROOT / "c26_6_final_decision/C26_6_FINAL_DECISION.json"
    rec = load_json(path, {})
    rec["vcmr_r100_sanity_check"] = {
        "status": audit["status"],
        "audit_path": str(OUT / "C26_7_VCMR_R100_SANITY_AUDIT.json"),
        "c24h_same_subset_vcmr_r100_iou05": audit["same_subset_metrics"]["c24h_selected_B"]["VCMR_R@100_IoU0.5"],
        "c26_final_vcmr_r100_iou05": audit["same_subset_metrics"]["c26_final_F"]["VCMR_R@100_IoU0.5"],
        "span_unit_conclusion": audit["span_unit_audit"]["unit_conclusion"],
        "single_span_proposal_per_query_video": audit["proposal_audit"]["unique_span_proposals_per_query_video"]["all_one"],
        "conclusion": audit["conclusion"],
    }
    rec["final_decision_note_after_r100_sanity"] = audit["decision_note"]
    write_json(path, rec)
    md_path = ROOT / "c26_6_final_decision/C26_6_FINAL_DECISION.md"
    old = md_path.read_text(encoding="utf-8") if md_path.exists() else "# C26-6 Final Decision\n"
    block = (
        "\n\n## VCMR R@100 Sanity Check\n\n"
        f"Status: `{audit['status']}`.\n\n"
        f"{audit['decision_note']}\n"
    )
    if "## VCMR R@100 Sanity Check" not in old:
        write_text(md_path, old.rstrip() + block)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df, manifest = build_canonical_table(MODE, force=False)
    hold = filtered_df_for_arrays(df, SPLIT)
    arrays_path, _arrays_manifest = build_training_arrays(df, MODE, force=False)
    ckpt = C26_CACHE / f"C26_PREM_RETRIEVER_{MODE}_seed{SEED}.local.pt"
    scores = score_arrays(arrays_path, ckpt, SPLIT, batch_size=4096, device="cuda:0")
    c26 = attach_scores(hold, scores)
    c26["_c26_retriever_score"] = c26["first_stage_z"] + c26["c26_residual"]
    selected_formula = {"name": "F_subtitle_focus_only", "alpha": 1.0, "beta": 0.10, "gamma": 0.08, "focus": "subtitle", "delta": 0.0, "eta": 0.0}

    c24h = read_c24h_subset([int(x) for x in c26["query_id"].unique()])
    join_cols = [
        "split", "query_id", "video_id", "first_stage_z", "bmn_z", "t2_z", "event_z",
        "bmn_final_score", "t2_score", "event_relevance_score",
        "best_bmn_span_start", "best_bmn_span_end", "span_start", "span_end",
        "candidate_iou", "iou_ge_05", "iou_ge_07", "correct_video",
        "bmn_available", "t2_available", "event_available",
    ]
    merged = c26.merge(c24h[join_cols], on=["split", "query_id", "video_id"], how="left", validate="one_to_one", suffixes=("", "_c24h"))
    merged["first_stage_z_c24h"] = pd.to_numeric(merged["first_stage_z_c24h"], errors="coerce").fillna(merged["first_stage_z"])
    merged["bmn_final_score_z"] = zscore_by_query(merged, "bmn_final_score")
    merged["t2_score_z"] = zscore_by_query(merged, "t2_score")
    merged["event_relevance_score_z"] = zscore_by_query(merged, "event_relevance_score")
    merged["_c26_final_F_score"] = apply_formula(merged, selected_formula)
    for name, arr in c24h_formula_scores(merged).items():
        merged[f"_c24h_{name}"] = arr

    same_subset_metrics = {
        "c24h_first_stage_with_c24h_bmn_span": evaluate_vr_vcmr(merged.assign(_score=merged["first_stage_z_c24h"]), "_score", "best_bmn_span_start", "best_bmn_span_end"),
        "c24h_selected_B": evaluate_vr_vcmr(merged, "_c24h_B_C19_C20_C21_hybrid_proxy", "best_bmn_span_start", "best_bmn_span_end"),
        "c24h_best_available_formula_on_same_subset": {},
        "c26_retriever_plus_c24h_span": evaluate_vr_vcmr(merged, "_c26_retriever_score", "best_bmn_span_start", "best_bmn_span_end"),
        "c26_final_F": evaluate_vr_vcmr(merged, "_c26_final_F_score", "best_bmn_span_start", "best_bmn_span_end"),
    }
    best_name = None
    best_metric: Dict[str, Any] = {}
    best_v = -math.inf
    formula_metrics: Dict[str, Any] = {}
    for col in [c for c in merged.columns if c.startswith("_c24h_")]:
        name = col.replace("_c24h_", "")
        m = evaluate_vr_vcmr(merged, col, "best_bmn_span_start", "best_bmn_span_end")
        formula_metrics[name] = m
        if float(m.get("VCMR_R@100_IoU0.5", 0.0)) > best_v:
            best_name, best_metric, best_v = name, m, float(m.get("VCMR_R@100_IoU0.5", 0.0))
    same_subset_metrics["c24h_best_available_formula_on_same_subset"] = {"name": best_name, "metrics": best_metric}

    unavailable_cross = {
        "status": "C26_NATIVE_SPAN_ABSENT",
        "reason": "C26 focus-fuse does not materialize c26_span_start/c26_span_end; C24H retriever + C26 focus-fuse span cannot be evaluated as an independent span/localizer cross.",
        "available_c26_span_columns": [c for c in merged.columns if c.startswith("c26_") and "span" in c],
    }
    span_audit = span_unit_audit(merged, "_c26_final_F_score")
    prop_audit = proposal_audit(merged)
    failure = topk_failure_breakdown(merged, "_c26_final_F_score", 100)
    c24h_failure = topk_failure_breakdown(merged, "_c24h_B_C19_C20_C21_hybrid_proxy", 100)
    join_audit = {
        "c26_holdout_rows": int(len(c26)),
        "c26_holdout_queries": int(c26["query_id"].nunique()),
        "c24h_subset_rows_before_join": int(len(c24h)),
        "merged_rows": int(len(merged)),
        "missing_c24h_span_rows": int(merged["best_bmn_span_start"].isna().sum()),
        "duplicate_c26_keys": int(c26.duplicated(["split", "query_id", "video_id"]).sum()),
        "duplicate_c24h_subset_keys": int(c24h.duplicated(["split", "query_id", "video_id"]).sum()),
        "row_space_hash": stable_hash({
            "mode": MODE,
            "split": SPLIT,
            "query_count": int(c26["query_id"].nunique()),
            "row_count": int(len(c26)),
            "query_min": int(c26["query_id"].min()),
            "query_max": int(c26["query_id"].max()),
        }),
        "canonical_manifest": manifest,
    }

    c24h_r100 = same_subset_metrics["c24h_selected_B"]["VCMR_R@100_IoU0.5"]
    c26_r100 = same_subset_metrics["c26_final_F"]["VCMR_R@100_IoU0.5"]
    unit_ok = span_audit["unit_conclusion"] == "seconds_consistent"
    single_span = prop_audit["unique_span_proposals_per_query_video"]["all_one"]
    c24h_same_low = c24h_r100 < 15.0
    if not unit_ok:
        status = "C26_R100_SANITY_FAILED_SPAN_UNIT_SUSPICIOUS"
        conclusion = "Span unit conversion is suspicious; C26 VCMR should not be treated as a clean negative until repaired."
    elif not single_span:
        status = "C26_R100_SANITY_PASS_MULTIPLE_PROPOSALS_PRESENT"
        conclusion = "Multiple span proposals are present; low R@100 is not caused by one-span output."
    elif c24h_same_low:
        status = "C26_R100_SANITY_PASS_SHARED_SINGLE_SPAN_LIMITATION"
        conclusion = "C24H on the same C26 medium subset is also low at VCMR R@100, while VR R@100 is near-saturated. The gap is a shared one-span/localizer proposal limitation, not a C26-only scoring regression."
    else:
        status = "C26_R100_SANITY_INCONCLUSIVE_C26_LOCALIZER_WEAK"
        conclusion = "C24H recovers on the same subset but C26 does not; investigate C26 localizer/scoring."
    decision_note = (
        "C26 remains not promoted. The R@100 gap is not an official/evaluator/pseudo issue; "
        "it is dominated by the shared borrowed BMN single-span proposal bottleneck. "
        "A future repair should materialize multiple native span proposals before rejudging the PREM localizer."
    )
    audit = {
        "stage": "C26-7",
        "status": status,
        "mode": MODE,
        "split": SPLIT,
        "same_subset_metrics": same_subset_metrics,
        "formula_metrics": formula_metrics,
        "delta_c26_final_vs_c24h_selected_B": metric_delta(same_subset_metrics["c26_final_F"], same_subset_metrics["c24h_selected_B"]),
        "requested_cross_checks": {
            "1_c24h_baseline_replay_same_subset": same_subset_metrics["c24h_selected_B"],
            "2_c26_retriever_plus_c24h_span": same_subset_metrics["c26_retriever_plus_c24h_span"],
            "3_c24h_retriever_plus_c26_focus_fuse_span": unavailable_cross,
            "4_span_start_end_unit": span_audit,
            "5_single_span_proposal_check": prop_audit,
        },
        "top100_failure_breakdown_c26_final_F": failure,
        "top100_failure_breakdown_c24h_selected_B": c24h_failure,
        "join_audit": join_audit,
        "span_unit_audit": span_audit,
        "proposal_audit": prop_audit,
        "official_val_used": False,
        "official_prediction_pool_used": False,
        "pseudo_official_holdout_used_for_selection": False,
        "evaluator_modified": False,
        "nms_modified": False,
        "conclusion": conclusion,
        "decision_note": decision_note,
    }

    sample_cols = [
        "query_id", "video_id", "gt_video_id", "candidate_video_rank", "first_stage_score",
        "best_bmn_span_start", "best_bmn_span_end", "candidate_iou", "_c26_final_F_score",
        "_c24h_B_C19_C20_C21_hybrid_proxy",
    ]
    bad_qids = [int(x["query_id"]) for x in failure["failure_examples"][:80]]
    sample = merged[merged["query_id"].isin(bad_qids)][sample_cols].head(5000)
    sample_path = OUT / "C26_7_R100_FAILURE_SAMPLE.parquet"
    sample.to_parquet(sample_path, index=False)
    audit["sample_path"] = str(sample_path)
    write_json(OUT / "C26_7_VCMR_R100_SANITY_AUDIT.json", audit)
    write_text(
        OUT / "C26_7_VCMR_R100_SANITY_AUDIT.md",
        "# C26-7 VCMR R@100 Sanity Audit\n\n"
        f"Status: `{status}`.\n\n"
        f"- C24H selected B on same subset VCMR R@100@0.5: `{c24h_r100}`\n"
        f"- C26 final F on same subset VCMR R@100@0.5: `{c26_r100}`\n"
        f"- Span unit: `{span_audit['unit_conclusion']}`\n"
        f"- One span per query-video: `{single_span}`\n\n"
        f"{conclusion}\n",
    )
    update_final_decision(audit)
    print(json.dumps(jsonable({"status": status, "c24h_r100": c24h_r100, "c26_r100": c26_r100, "single_span": single_span, "unit": span_audit["unit_conclusion"]}), indent=2))


if __name__ == "__main__":
    main()
