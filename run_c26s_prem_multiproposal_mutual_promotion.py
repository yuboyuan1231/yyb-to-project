#!/usr/bin/env python3
"""C26S PREM-style multi-proposal mutual promotion.

This runner never runs official validation, never reads official prediction
pools, never uses pseudo-official holdout for selection, and never modifies
evaluator/NMS. It builds on C26R multi-span evidence and tests proposal-level
PREM span ranking plus span-to-video feedback.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch

from tools.c26s.multiproposal_dataset import (
    CANONICAL,
    C26R_JOINED,
    C26S_CACHE,
    canonical_columns,
    jsonable,
    load_json,
    load_or_build_rowspace,
    proposal_distribution,
    stable_hash,
    write_json,
    write_text,
)
from tools.c26s.mutual_promotion_eval import evaluate_by_split, metric_delta, summarize_holdout
from tools.c26s.mutual_promotion_losses import loss_manifest, selection_objective
from tools.c26s.span_ranker import add_span_relevance, span_score_variants
from tools.c26s.span_to_video_feedback import add_feedback_features, feedback_variants


ROOT = Path(__file__).resolve().parent
PROMOTED = "C7-B6 R1SelectiveTop1"
OUT0 = ROOT / "c26s_0_protocol_freeze"
OUT1 = ROOT / "c26s_1_multiproposal_rowspace"
OUT2 = ROOT / "c26s_2_prem_span_relevance"
OUT3 = ROOT / "c26s_3_span_to_video_feedback"
OUT4 = ROOT / "c26s_4_mutual_promotion_integration"
OUT5 = ROOT / "c26s_5_robustness_route_decision"
OUT6 = ROOT / "c26s_6_final_decision"
SPLITS = ["calib_select", "calib_holdout"]


def sh(cmd: str) -> str:
    return subprocess.check_output(cmd, cwd=ROOT, shell=True, stderr=subprocess.STDOUT, text=True).strip()


def git_info() -> Dict[str, Any]:
    return {
        "branch": sh("git branch --show-current"),
        "commit": sh("git rev-parse --short HEAD"),
        "commit_full": sh("git rev-parse HEAD"),
        "status_short": sh("git status --short").splitlines(),
    }


def file_record(path: Path) -> Dict[str, Any]:
    return {
        "path": str(path),
        "exists": path.exists(),
        "readable": os.access(path, os.R_OK) if path.exists() else False,
        "size_bytes": path.stat().st_size if path.exists() and path.is_file() else None,
    }


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def root_contamination() -> Dict[str, Any]:
    blockers: List[str] = []
    warnings: List[str] = []
    for p in ROOT.iterdir():
        if not p.is_file():
            continue
        name = p.name.lower()
        if "official" in name and p.suffix == ".py":
            warnings.append(str(p))
        if "official" in name and any(tok in name for tok in ["prediction", "raw", "nms"]) and not p.suffix == ".py":
            blockers.append(str(p))
    return {"blocking_root_files": blockers, "official_script_warnings": warnings}


def ensure_protocol_ready() -> None:
    rec = load_json(OUT0 / "C26S_0_PROTOCOL.json", {})
    if rec.get("status") != "C26S_PROTOCOL_READY":
        raise RuntimeError(f"C26S protocol not ready: {rec.get('status')}")


def stage_c26s_0(args: argparse.Namespace) -> Dict[str, Any]:
    c26r = load_json(ROOT / "c26r_multi_proposal_repair/C26R_FINAL_DECISION.json", {})
    c26r_topm = load_json(ROOT / "c26r_multi_proposal_repair/C26R_2_TOPM_PROPOSAL_RESULTS.json", {})
    c26 = load_json(ROOT / "c26_6_final_decision/C26_6_FINAL_DECISION.json", {})
    c24h = load_json(ROOT / "c24h_6_final_decision/C24H_6_FINAL_DECISION.json", {})
    c26_stage2 = ROOT / "c26_2_relevant_content_mining/C26_2_MINING_DECISION.json"
    contamination = root_contamination()
    has_c26r_commit = sh("git merge-base --is-ancestor 3b0d1d2 HEAD >/dev/null 2>&1; echo $?") == "0"
    c26r_ok = c26r.get("status") == "C26R_MULTI_PROPOSAL_REPAIR_PROMISING" and c26r.get("best_by_calib_select") == "R3b1_C26_PREM_guarded_localizer"
    c26r_proved = abs(float(c26r.get("single_span_c26_r100_iou05", 0.0)) - 8.5556) < 0.2 and c26r.get("topm_results", {}).get("top1_per_video_R3b1_C26_PREM_guarded_localizer", {}).get("VCMR_R@100_IoU0.5", 0) < 9.0
    c24h_ok = (
        c24h.get("status") == "C24H_READY_FOR_C25_RAW_FRAME_STRONG_FEATURE_AUDIT"
        and float(c24h.get("bmn_train_fit_coverage", 0.0) or 0.0) >= 1.0
        and float(c24h.get("t2_train_fit_coverage", 0.0) or 0.0) >= 1.0
        and bool(c24h.get("event_direct_columns_present"))
    )
    if contamination["blocking_root_files"]:
        status = "C26S_PROTOCOL_BLOCKED_CONTAMINATION_RISK"
    elif not c26r_ok or not c26r_proved or not c26r_topm:
        status = "C26S_PROTOCOL_BLOCKED_MISSING_C26R_ARTIFACTS"
    elif not C26R_JOINED.exists():
        status = "C26S_PROTOCOL_BLOCKED_MISSING_MULTISPAN_TABLE"
    elif not c24h_ok:
        status = "C26S_PROTOCOL_BLOCKED_MISSING_C24H_EVIDENCE"
    elif not c26_stage2.exists():
        status = "C26S_PROTOCOL_BLOCKED_SPLIT_OR_SCHEMA_MISSING"
    else:
        status = "C26S_PROTOCOL_READY"
    rec = {
        "stage": "C26S-0",
        "status": status,
        "git": git_info(),
        "created_from_c26r_commit_3b0d1d2": has_c26r_commit,
        "c26r_final_decision": c26r.get("status"),
        "c26r_selected_route": c26r.get("best_by_calib_select"),
        "c26r_single_span_r100_iou05": c26r.get("single_span_c26_r100_iou05"),
        "c26r_multispan_table": file_record(C26R_JOINED),
        "c26r_topm_exists": bool(c26r_topm),
        "c26_final_status": c26.get("status"),
        "c24h_final_status": c24h.get("status"),
        "c24h_bmn_train_fit_coverage": c24h.get("bmn_train_fit_coverage"),
        "c24h_t2_train_fit_coverage": c24h.get("t2_train_fit_coverage"),
        "event_direct_columns_present": c24h.get("event_direct_columns_present"),
        "forbidden_actions": {
            "official_val_used": False,
            "official_prediction_pool_used": False,
            "pseudo_used_for_selection": False,
            "evaluator_modified": False,
            "nms_modified": False,
            "promoted_system": PROMOTED,
            "c26s_promoted": False,
        },
        "contamination": contamination,
    }
    write_text(OUT0 / "C26S_0_PROTOCOL.md", f"# C26S-0 Protocol\n\nStatus: `{status}`.")
    write_json(OUT0 / "C26S_0_PROTOCOL.json", rec)
    write_text(OUT0 / "C26S_0_C26R_ACCEPTANCE.md", "# C26S C26R Acceptance\n\nC26R proved one-span proposal collapse and selected `R3b1_C26_PREM_guarded_localizer` as the best inference route by calib_select.")
    write_json(OUT0 / "C26S_0_DEPENDENCY_AUDIT.json", rec)
    write_text(OUT0 / "C26S_0_FORBIDDEN_ACTIONS_AUDIT.md", "# Forbidden Actions\n\nNo official validation, no official prediction pool, no pseudo selection, no evaluator/NMS edits, no promotion.")
    write_json(OUT0 / "C26S_0_REPRODUCIBILITY_MANIFEST.json", rec)
    return rec


def stage_c26s_1(args: argparse.Namespace) -> Dict[str, Any]:
    ensure_protocol_ready()
    max_q = args.max_queries
    if args.mode == "smoke" and max_q is None:
        max_q = 20
    df, manifest = load_or_build_rowspace(args.mode, max_queries=max_q, force=args.force)
    dist = proposal_distribution(df)
    duplicate = int(df.duplicated(["split", "query_id", "seed", "video_id", "span_start", "span_end"]).sum())
    invalid = int(((df["span_end"] <= df["span_start"]) | df["span_start"].isna() | df["span_end"].isna()).sum())
    nan_inf = int(np.isnan(df.select_dtypes(include=[np.number]).to_numpy(np.float64)).sum())
    has_train_fit = bool((df["split"].astype(str) == "train_fit").any())
    multi_ok = dist["span_count_per_query_video"]["median"] > 1 and dist["span_count_per_query_video"]["max"] > 1
    status = "C26S_MULTISPAN_ROWSPACE_READY" if multi_ok and has_train_fit else ("C26S_MULTISPAN_ROWSPACE_PARTIAL" if multi_ok else "C26S_MULTISPAN_ROWSPACE_BLOCKED")
    sample = df.head(5000)
    sample_path = OUT1 / "C26S_1_ROWSPACE_SAMPLE.parquet"
    ensure_parent(sample_path)
    sample.to_parquet(sample_path, index=False)
    join_audit = {
        "position_based_join": False,
        "duplicate_span_count": duplicate,
        "invalid_span_count": invalid,
        "silent_fill": False,
        "train_fit_missing_reason": None if has_train_fit else "C26R joined multi-span table only covers calib_select/calib_holdout; train_fit multi-span rebuild is required for READY/full training.",
    }
    rec = {
        "stage": "C26S-1",
        "status": status,
        "mode": args.mode,
        "manifest": manifest,
        "coverage": {
            "query_count": int(df[["split", "query_id"]].drop_duplicates().shape[0]),
            "video_candidate_count": int(df[["split", "query_id", "video_id"]].drop_duplicates().shape[0]),
            "span_row_count": int(len(df)),
            "split_query_coverage": df.groupby("split", observed=True)["query_id"].nunique().to_dict(),
            "has_train_fit": has_train_fit,
        },
        "proposal_distribution": dist,
        "join_audit": join_audit,
        "nan_inf_numeric_count": nan_inf,
        "span_unit": "seconds",
        "sample_path": str(sample_path),
        "official_used": False,
    }
    write_text(OUT1 / "C26S_1_ROWSPACE_PLAN.md", "# C26S-1 Rowspace Plan\n\nUse C26R/C17 multi-span proposal rows as canonical row-space.")
    write_json(OUT1 / "C26S_1_SOURCE_MANIFEST.json", manifest)
    write_json(OUT1 / "C26S_1_CANONICAL_SCHEMA.json", {"columns": canonical_columns(), "schema_hash": stable_hash(canonical_columns()), "diagnostic_only": ["gt_video_id", "gt_start", "gt_end", "candidate_iou", "iou_ge_05", "iou_ge_07", "correct_video"]})
    write_json(OUT1 / "C26S_1_MULTISPAN_COVERAGE_AUDIT.json", rec["coverage"])
    write_json(OUT1 / "C26S_1_PROPOSAL_DISTRIBUTION_AUDIT.json", dist)
    write_json(OUT1 / "C26S_1_JOIN_AUDIT.json", join_audit)
    write_text(OUT1 / "C26S_1_ROWSPACE_DECISION.md", f"# C26S-1 Rowspace Decision\n\nStatus: `{status}`.")
    write_json(OUT1 / "C26S_1_ROWSPACE_DECISION.json", rec)
    return rec


def stage_c26s_2(args: argparse.Namespace) -> Dict[str, Any]:
    ensure_protocol_ready()
    max_q = args.max_queries if args.mode != "smoke" else (args.max_queries or 20)
    df, _ = load_or_build_rowspace(args.mode, max_queries=max_q, force=False)
    feature_info = add_span_relevance(df)
    scored_path = C26S_CACHE / f"C26S_SPAN_RELEVANCE_{args.mode}_seed{args.seed}.local.parquet"
    df.to_parquet(scored_path, index=False)
    visual_cov = float(pd.to_numeric(df["visual_span_relevance_mean"], errors="coerce").notna().mean())
    subtitle_cov = float(pd.to_numeric(df["subtitle_span_relevance_mean"], errors="coerce").notna().mean())
    status = "C26S_PREM_SPAN_RELEVANCE_READY" if max(visual_cov, subtitle_cov) > 0.99 and args.mode != "full" else "C26S_PREM_SPAN_RELEVANCE_PARTIAL"
    sample_path = OUT2 / "C26S_2_SPAN_RELEVANCE_SAMPLE.parquet"
    ensure_parent(sample_path)
    df.head(5000).to_parquet(sample_path, index=False)
    rec = {
        "stage": "C26S-2",
        "status": status,
        "local_path": str(scored_path),
        "feature_info": feature_info,
        "visual_coverage": visual_cov,
        "subtitle_coverage": subtitle_cov,
        "query_type_breakdown": df.groupby("query_type", observed=True)["query_id"].nunique().to_dict() if "query_type" in df else {},
        "duration_breakdown": df.groupby("duration_bucket", observed=True)["query_id"].nunique().to_dict() if "duration_bucket" in df else {},
        "diagnostic_columns_excluded_from_inference": ["candidate_iou", "iou_ge_05", "iou_ge_07", "correct_video"],
        "official_used": False,
        "sample_path": str(sample_path),
    }
    write_text(OUT2 / "C26S_2_SPAN_RELEVANCE_PLAN.md", "# C26S-2 Span Relevance Plan\n\nProject C26 PREM-style visual/subtitle relevance and C17 localizer signals to proposal spans.")
    write_json(OUT2 / "C26S_2_FEATURE_SCHEMA.json", {"features": [c for c in df.columns if c.endswith("relevance") or c.startswith("visual_span") or c.startswith("subtitle_span") or c.startswith("modality_gate")], "gt_excluded": True})
    write_json(OUT2 / "C26S_2_VISUAL_SPAN_RELEVANCE_AUDIT.json", {"coverage": visual_cov})
    write_json(OUT2 / "C26S_2_SUBTITLE_SPAN_RELEVANCE_AUDIT.json", {"coverage": subtitle_cov})
    write_json(OUT2 / "C26S_2_FOCUS_CONSISTENCY_AUDIT.json", {"mean": float(df["focus_span_agreement"].mean()), "p10": float(df["focus_span_agreement"].quantile(0.10))})
    write_json(OUT2 / "C26S_2_SPAN_RELEVANCE_FEATURES.json", rec)
    write_text(OUT2 / "C26S_2_SPAN_RELEVANCE_DECISION.md", f"# C26S-2 Span Relevance Decision\n\nStatus: `{status}`.")
    write_json(OUT2 / "C26S_2_SPAN_RELEVANCE_DECISION.json", rec)
    return rec


def _load_span_relevance(args: argparse.Namespace) -> pd.DataFrame:
    path = C26S_CACHE / f"C26S_SPAN_RELEVANCE_{args.mode}_seed{args.seed}.local.parquet"
    if path.exists():
        return pd.read_parquet(path)
    max_q = args.max_queries if args.mode != "smoke" else (args.max_queries or 20)
    df, _ = load_or_build_rowspace(args.mode, max_queries=max_q, force=False)
    add_span_relevance(df)
    df.to_parquet(path, index=False)
    return df


def stage_c26s_3(args: argparse.Namespace) -> Dict[str, Any]:
    ensure_protocol_ready()
    df = _load_span_relevance(args)
    span_variants = span_score_variants(df)
    for name, arr in span_variants.items():
        df[name] = arr.astype(np.float32)
    base_span = "S4_false_positive_guard"
    df = add_feedback_features(df, base_span)
    fvars = feedback_variants(df)
    for name, arr in fvars.items():
        df[name] = arr.astype(np.float32)
    path = C26S_CACHE / f"C26S_FEEDBACK_{args.mode}_seed{args.seed}.local.parquet"
    df.to_parquet(path, index=False)
    baseline = evaluate_by_split(df, "F0_no_feedback")
    results: Dict[str, Any] = {}
    for name in fvars:
        results[name] = evaluate_by_split(df, name)
    # Selection by video metrics only is approximated by VCMR evaluator because
    # all spans in a video share the same feedback score.
    best_name = "F0_no_feedback"
    best_obj = -1e18
    for name, res in results.items():
        obj = selection_objective(res["calib_select"]["summary"], baseline["calib_select"]["summary"])
        results[name]["selection_objective"] = obj
        if obj > best_obj:
            best_obj = obj
            best_name = name
    hold = summarize_holdout(results[best_name])
    base_hold = summarize_holdout(baseline)
    delta = metric_delta(hold, base_hold)
    harmful = delta.get("VR_R@100", 0.0) < -0.5 or delta.get("wrong_video_high_score_rate", 0.0) > 1.0
    promising = (delta.get("VCMR_R@1_IoU0.7", 0.0) > 0 or delta.get("VCMR_R@5_IoU0.7", 0.0) > 0 or delta.get("VCMR_R@10_IoU0.7", 0.0) > 0) and not harmful
    status = "C26S_SPAN_TO_VIDEO_FEEDBACK_PROMISING" if promising else ("C26S_SPAN_TO_VIDEO_FEEDBACK_HARMFUL" if harmful else "C26S_SPAN_TO_VIDEO_FEEDBACK_STABLE_NO_GAIN")
    sample_path = OUT3 / "C26S_3_FEEDBACK_SAMPLE.parquet"
    ensure_parent(sample_path)
    df.head(5000).to_parquet(sample_path, index=False)
    rec = {
        "stage": "C26S-3",
        "status": status,
        "selected_feedback_variant": best_name,
        "baseline": baseline,
        "results": results,
        "delta_vs_no_feedback_holdout": delta,
        "local_path": str(path),
        "official_used": False,
        "sample_path": str(sample_path),
    }
    write_text(OUT3 / "C26S_3_FEEDBACK_PLAN.md", "# C26S-3 Feedback Plan\n\nAggregate topK span distributions into video-level localization feedback.")
    write_json(OUT3 / "C26S_3_FEEDBACK_FEATURE_SCHEMA.json", {"features": ["best_span_score", "top2_span_mean", "top4_span_mean", "topK_span_logsumexp", "span_score_margin", "span_score_entropy", "wrong_video_risk_score"]})
    write_json(OUT3 / "C26S_3_SPAN_DISTRIBUTION_AUDIT.json", {"score_col": base_span, "rows": int(len(df))})
    write_json(OUT3 / "C26S_3_FEEDBACK_FEATURES.json", {"local_path": str(path), "local_only": True})
    write_json(OUT3 / "C26S_3_VIDEO_RERANK_RESULTS.json", results)
    write_json(OUT3 / "C26S_3_WRONG_VIDEO_SUPPRESSION_AUDIT.json", {"baseline": base_hold.get("wrong_video_high_score_rate"), "selected": hold.get("wrong_video_high_score_rate")})
    write_text(OUT3 / "C26S_3_FEEDBACK_DECISION.md", f"# C26S-3 Feedback Decision\n\nStatus: `{status}`.\nSelected: `{best_name}`.")
    write_json(OUT3 / "C26S_3_FEEDBACK_DECISION.json", rec)
    return rec


def _load_feedback(args: argparse.Namespace) -> pd.DataFrame:
    path = C26S_CACHE / f"C26S_FEEDBACK_{args.mode}_seed{args.seed}.local.parquet"
    if path.exists():
        return pd.read_parquet(path)
    stage_c26s_3(args)
    return pd.read_parquet(path)


def stage_c26s_4(args: argparse.Namespace) -> Dict[str, Any]:
    ensure_protocol_ready()
    df = _load_feedback(args)
    s3 = load_json(OUT3 / "C26S_3_FEEDBACK_DECISION.json", {})
    selected_feedback = s3.get("selected_feedback_variant", "F8_no_regression_guarded_feedback")
    span_score = df["S4_false_positive_guard"].to_numpy(np.float32)
    video = df[selected_feedback].to_numpy(np.float32)
    c26r = df["R3b1_C26_PREM_guarded_localizer"].to_numpy(np.float32)
    df["A_C24H_full_evidence_route"] = df["R2c_C17_BMN_T2_hybrid"].to_numpy(np.float32) if "R2c_C17_BMN_T2_hybrid" in df else c26r
    df["B_C26_single_span_final_proxy"] = df["c26_final_F_score"].to_numpy(np.float32)
    df["C_C26R_multi_span_selected_R3b1"] = c26r
    df["D_C26S_span_ranking_only"] = span_score
    df["E_C26S_span_to_video_feedback_only"] = video
    df["F_C26S_mutual_promotion_full"] = 0.68 * video + 0.32 * span_score
    df["G_no_regression_guarded_mutual_promotion"] = np.maximum(c26r - 0.03, 0.62 * video + 0.30 * span_score + 0.08 * c26r).astype(np.float32)
    df["H_oracle_diagnostic_only"] = np.where(df["candidate_iou"].to_numpy(np.float32) >= 0.7, 10.0, np.where(df["candidate_iou"].to_numpy(np.float32) >= 0.5, 8.0, 0.01 * c26r)).astype(np.float32)
    methods = [
        "A_C24H_full_evidence_route", "B_C26_single_span_final_proxy", "C_C26R_multi_span_selected_R3b1",
        "D_C26S_span_ranking_only", "E_C26S_span_to_video_feedback_only",
        "F_C26S_mutual_promotion_full", "G_no_regression_guarded_mutual_promotion",
        "H_oracle_diagnostic_only",
    ]
    results: Dict[str, Any] = {}
    baseline = evaluate_by_split(df, "C_C26R_multi_span_selected_R3b1")
    best = None
    best_obj = -1e18
    for name in methods:
        res = evaluate_by_split(df, name)
        if name.startswith("H_"):
            res["selectable"] = False
            res["selection_excluded_reason"] = "diagnostic_oracle_uses_gt_iou"
        else:
            res["selectable"] = True
            obj = selection_objective(res["calib_select"]["summary"], baseline["calib_select"]["summary"])
            res["selection_objective"] = obj
            if obj > best_obj:
                best = name
                best_obj = obj
        results[name] = res
    assert best is not None
    hold = results[best]["calib_holdout"]["summary"]
    base_hold = baseline["calib_holdout"]["summary"]
    delta = metric_delta(hold, base_hold)
    front_positive = delta.get("VCMR_R@1_IoU0.7", 0.0) > 0 or delta.get("VCMR_R@5_IoU0.7", 0.0) > 0 or delta.get("VCMR_R@10_IoU0.7", 0.0) > 0
    harmful = delta.get("VR_R@100", 0.0) < -0.5 or delta.get("wrong_video_high_score_rate", 0.0) > 1.0
    r100_only = delta.get("VCMR_R@100_IoU0.5", 0.0) > 0 and not front_positive
    status = "C26S_MUTUAL_PROMOTION_PROMISING" if front_positive and not harmful else ("C26S_MUTUAL_PROMOTION_HARMFUL" if harmful else ("C26S_MUTUAL_PROMOTION_R100_ONLY" if r100_only else "C26S_MUTUAL_PROMOTION_STABLE_NO_GAIN"))
    sample_path = OUT4 / "C26S_4_INTEGRATION_SCORE_SAMPLE.parquet"
    ensure_parent(sample_path)
    df.head(5000).to_parquet(sample_path, index=False)
    rec = {
        "stage": "C26S-4",
        "status": status,
        "selected_integration": best,
        "selected_feedback_variant": selected_feedback,
        "results": results,
        "delta_vs_c26r_holdout": delta,
        "sample_path": str(sample_path),
        "official_used": False,
    }
    write_text(OUT4 / "C26S_4_INTEGRATION_PLAN.md", "# C26S-4 Integration Plan\n\nFuse video feedback and proposal span ranking into VCMR scores.")
    write_text(OUT4 / "C26S_4_MODEL_ARCHITECTURE.md", "# C26S-4 Architecture\n\nfirst-stage/C26 PREM video score -> span scoring -> span-to-video feedback -> mutual VCMR score.")
    write_json(OUT4 / "C26S_4_TRAINING_CONFIGS.json", {"selection_split": "calib_select", "report_split": "calib_holdout", "loss_manifest": loss_manifest(), "pseudo_used": False})
    write_json(OUT4 / "C26S_4_INTEGRATION_RESULTS.json", results)
    write_json(OUT4 / "C26S_4_COMPONENT_ABLATION.json", results)
    write_json(OUT4 / "C26S_4_WRONG_VIDEO_AUDIT.json", {"baseline": base_hold.get("wrong_video_high_score_rate"), "selected": hold.get("wrong_video_high_score_rate")})
    write_json(OUT4 / "C26S_4_SELECTED_INTEGRATION.json", {"selected": best, "holdout": hold, "delta_vs_c26r": delta})
    write_text(OUT4 / "C26S_4_INTEGRATION_DECISION.md", f"# C26S-4 Integration Decision\n\nStatus: `{status}`.\nSelected: `{best}`.")
    write_json(OUT4 / "C26S_4_INTEGRATION_DECISION.json", rec)
    return rec


def stage_c26s_5(args: argparse.Namespace) -> Dict[str, Any]:
    s4 = load_json(OUT4 / "C26S_4_INTEGRATION_DECISION.json", {})
    s1 = load_json(OUT1 / "C26S_1_ROWSPACE_DECISION.json", {})
    selected = s4.get("selected_integration")
    results = s4.get("results", {})
    hold = results.get(selected, {}).get("calib_holdout", {}).get("summary", {}) if selected else {}
    c26r = results.get("C_C26R_multi_span_selected_R3b1", {}).get("calib_holdout", {}).get("summary", {})
    delta = metric_delta(hold, c26r)
    integration_status = s4.get("status")
    rowspace_partial = s1.get("status") != "C26S_MULTISPAN_ROWSPACE_READY"
    if integration_status == "C26S_MUTUAL_PROMOTION_PROMISING" and not rowspace_partial:
        route = "C26S_READY_FOR_C26T_FULL_TRAINING"
        status = "C26S_ROBUSTNESS_PASS"
    elif integration_status in {"C26S_MUTUAL_PROMOTION_PROMISING", "C26S_MUTUAL_PROMOTION_STABLE_NO_GAIN"}:
        route = "C26S_CONTINUE_MUTUAL_PROMOTION_REPAIR" if rowspace_partial else "C26S_READY_FOR_C26T_FULL_TRAINING"
        status = "C26S_ROBUSTNESS_PARTIAL" if rowspace_partial else "C26S_ROBUSTNESS_PASS"
    elif integration_status == "C26S_MUTUAL_PROMOTION_HARMFUL":
        route = "C26S_STOP_RELEASE_FEATURE_PREM_ROUTE"
        status = "C26S_ROBUSTNESS_FAIL"
    else:
        route = "C26S_READY_FOR_C25_STRONG_FEATURE_WHEN_FRAMES_AVAILABLE"
        status = "C26S_ROBUSTNESS_INCONCLUSIVE"
    rec = {
        "stage": "C26S-5",
        "status": status,
        "route_recommendation": route,
        "seed_robustness": {"seed2026": integration_status, "seed2027": "not_run_requires_C26S_joined_table", "seed2028": "not_run_requires_C26S_joined_table"},
        "query_duration_robustness": results.get(selected, {}) if selected else {},
        "score_distribution_audit": {"delta_vs_c26r": delta},
        "proposal_space_audit": s1.get("proposal_distribution"),
        "selection_firewall_audit": {"selection_split": "calib_select", "report_split": "calib_holdout", "pseudo_used_for_selection": False, "official_used": False},
        "span_to_video_feedback_helped": selected in {"F_C26S_mutual_promotion_full", "G_no_regression_guarded_mutual_promotion", "E_C26S_span_to_video_feedback_only"},
    }
    write_text(OUT5 / "C26S_5_ROBUSTNESS_PLAN.md", "# C26S-5 Robustness Plan\n\nCheck seed, query/duration, wrong-video, proposal sensitivity and route.")
    write_json(OUT5 / "C26S_5_SEED_ROBUSTNESS.json", rec["seed_robustness"])
    write_json(OUT5 / "C26S_5_QUERY_DURATION_ROBUSTNESS.json", rec["query_duration_robustness"])
    write_json(OUT5 / "C26S_5_D_E_F_SUBSET_AUDIT.json", {"available": False, "reason": "D/E/F subset labels not materialized in C26S medium row-space"})
    write_json(OUT5 / "C26S_5_SCORE_DISTRIBUTION_AUDIT.json", rec["score_distribution_audit"])
    write_json(OUT5 / "C26S_5_PROPOSAL_SPACE_AUDIT.json", rec["proposal_space_audit"])
    write_text(OUT5 / "C26S_5_C25_ROUTE_AUDIT.md", f"# C26S C25 Route Audit\n\nRoute: `{route}`. C26S uses release features only; C25 frames remain useful if available.")
    write_json(OUT5 / "C26S_5_PSEUDO_ONELOOK_DIAGNOSTIC.json", {"run": False, "used_for_selection": False})
    write_json(OUT5 / "C26S_5_SELECTION_FIREWALL_AUDIT.json", rec["selection_firewall_audit"])
    write_text(OUT5 / "C26S_5_ROBUSTNESS_DECISION.md", f"# C26S-5 Robustness Decision\n\nStatus: `{status}`.\nRoute: `{route}`.")
    write_json(OUT5 / "C26S_5_ROBUSTNESS_DECISION.json", rec)
    return rec


def stage_c26s_6(args: argparse.Namespace) -> Dict[str, Any]:
    recs = {
        "c26s_0": load_json(OUT0 / "C26S_0_PROTOCOL.json", {}),
        "c26s_1": load_json(OUT1 / "C26S_1_ROWSPACE_DECISION.json", {}),
        "c26s_2": load_json(OUT2 / "C26S_2_SPAN_RELEVANCE_DECISION.json", {}),
        "c26s_3": load_json(OUT3 / "C26S_3_FEEDBACK_DECISION.json", {}),
        "c26s_4": load_json(OUT4 / "C26S_4_INTEGRATION_DECISION.json", {}),
        "c26s_5": load_json(OUT5 / "C26S_5_ROBUSTNESS_DECISION.json", {}),
    }
    route = recs["c26s_5"].get("route_recommendation") or "C26S_INCONCLUSIVE_NEED_REPAIR"
    allowed = {
        "C26S_READY_FOR_C26T_FULL_TRAINING",
        "C26S_CONTINUE_MUTUAL_PROMOTION_REPAIR",
        "C26S_WAIT_FOR_C25_FRAMES",
        "C26S_READY_FOR_C25_STRONG_FEATURE_WHEN_FRAMES_AVAILABLE",
        "C26S_STOP_RELEASE_FEATURE_PREM_ROUTE",
        "C26S_INCONCLUSIVE_NEED_REPAIR",
    }
    decision = route if route in allowed else "C26S_INCONCLUSIVE_NEED_REPAIR"
    s4 = recs["c26s_4"]
    selected = s4.get("selected_integration")
    results = s4.get("results", {})
    final_metrics = results.get(selected, {}).get("calib_holdout", {}).get("summary", {}) if selected else {}
    c26r_metrics = results.get("C_C26R_multi_span_selected_R3b1", {}).get("calib_holdout", {}).get("summary", {})
    c26_single = load_json(ROOT / "c26_6_final_decision/C26_6_FINAL_DECISION.json", {}).get("final_vcmr_metrics", {})
    c24h = load_json(ROOT / "c24h_6_final_decision/C24H_6_FINAL_DECISION.json", {}).get("final_vcmr_metrics", {})
    c23 = load_json(ROOT / "c23_6_final_decision/C23_6_FINAL_DECISION.json", {}).get("final_vcmr_metrics", {})
    rec = {
        "stage": "C26S-6",
        "status": decision,
        "final_decision": decision,
        "c26s_0_protocol_status": recs["c26s_0"].get("status"),
        "c26s_1_rowspace_status": recs["c26s_1"].get("status"),
        "c26s_2_span_relevance_status": recs["c26s_2"].get("status"),
        "c26s_3_feedback_status": recs["c26s_3"].get("status"),
        "c26s_4_integration_status": recs["c26s_4"].get("status"),
        "c26s_5_robustness_status": recs["c26s_5"].get("status"),
        "selected_span_ranker": "S4_false_positive_guard",
        "selected_feedback_variant": recs["c26s_3"].get("selected_feedback_variant"),
        "selected_mutual_promotion_formula": selected,
        "final_vcmr_metrics": final_metrics,
        "final_vr_metrics": {k: final_metrics.get(k) for k in ["VR_R@1", "VR_R@5", "VR_R@10", "VR_R@100"]},
        "delta_vs_c26_single_span": metric_delta(final_metrics, c26_single),
        "delta_vs_c26r_multi_span": metric_delta(final_metrics, c26r_metrics),
        "delta_vs_c24h": metric_delta(final_metrics, c24h),
        "delta_vs_c23": metric_delta(final_metrics, c23),
        "delta_vs_c19_c21_hybrid": metric_delta(final_metrics, results.get("A_C24H_full_evidence_route", {}).get("calib_holdout", {}).get("summary", {})),
        "wrong_video_risk": final_metrics.get("wrong_video_high_score_rate"),
        "high_score_false_positive": final_metrics.get("high_score_false_positive_rate", final_metrics.get("wrong_video_high_score_rate")),
        "topm_proposal_curve": load_json(ROOT / "c26r_multi_proposal_repair/C26R_FINAL_DECISION.json", {}).get("topm_results"),
        "span_to_video_feedback_helped": recs["c26s_5"].get("span_to_video_feedback_helped"),
        "c25_frames_required": False,
        "official_val_used": False,
        "official_prediction_pool_used": False,
        "pseudo_official_holdout_used_for_selection": False,
        "evaluator_modified": False,
        "nms_modified": False,
        "c26s_promoted_system": False,
        "current_promoted_system": PROMOTED,
        "local_only_artifacts": {"cache_root": str(C26S_CACHE), "canonical": str(CANONICAL)},
    }
    packet = {"decision": decision, "selected_span_ranker": rec["selected_span_ranker"], "selected_feedback_variant": rec["selected_feedback_variant"], "selected_integration": selected, "metrics": final_metrics, "local_only_artifacts": rec["local_only_artifacts"]}
    write_text(OUT6 / "C26S_6_FINAL_DECISION.md", f"# C26S-6 Final Decision\n\nFinal decision: `{decision}`.\n\nC26S is not promoted. Current promoted official remains `{PROMOTED}`.")
    write_json(OUT6 / "C26S_6_FINAL_DECISION.json", rec)
    write_text(OUT6 / "C26S_6_MUTUAL_PROMOTION_PACKET.md", f"# C26S Mutual Promotion Packet\n\nDecision: `{decision}`.")
    write_json(OUT6 / "C26S_6_MUTUAL_PROMOTION_PACKET.json", packet)
    write_text(OUT6 / "C26S_6_RISK_REGISTER.md", "# C26S Risk Register\n\n- C26R medium multi-span row-space lacks train_fit.\n- C26S span relevance uses release-feature/PREM projections, not PREM official code.\n- No official validation was run.\n")
    write_text(OUT6 / "C26S_6_NEXT_STEP_DECISION.md", f"# C26S Next Step\n\n`{decision}`")
    write_json(OUT6 / "C26S_6_NEXT_STEP_DECISION.json", {"decision": decision, "official_forbidden": True})
    return rec


def run_all(args: argparse.Namespace) -> Dict[str, Any]:
    out = {}
    for name, fn in [
        ("c26s_0", stage_c26s_0),
        ("c26s_1", stage_c26s_1),
        ("c26s_2", stage_c26s_2),
        ("c26s_3", stage_c26s_3),
        ("c26s_4", stage_c26s_4),
        ("c26s_5", stage_c26s_5),
        ("c26s_6", stage_c26s_6),
    ]:
        print(f"[C26S] running {name}", flush=True)
        out[name] = fn(args)
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--stage", default="all", choices=["c26s_0", "c26s_1", "c26s_2", "c26s_3", "c26s_4", "c26s_5", "c26s_6", "all"])
    p.add_argument("--mode", default="medium", choices=["smoke", "medium", "full"])
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--force", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--max_queries", type=int, default=None)
    p.add_argument("--max_candidates", type=int, default=None)
    p.add_argument("--max_spans_per_video", type=int, default=None)
    p.add_argument("--chunk_size", type=int, default=256)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--device", default="auto")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    torch.set_num_threads(min(32, os.cpu_count() or 1))
    if args.stage == "all":
        rec = run_all(args)
    else:
        rec = {args.stage: globals()[f"stage_{args.stage}"](args)}
    final = rec.get("c26s_6") or load_json(OUT6 / "C26S_6_FINAL_DECISION.json", {})
    print(json.dumps(jsonable({
        "branch": sh("git branch --show-current"),
        "commit": sh("git rev-parse --short HEAD"),
        "c26s_0": load_json(OUT0 / "C26S_0_PROTOCOL.json", {}).get("status"),
        "c26s_1": load_json(OUT1 / "C26S_1_ROWSPACE_DECISION.json", {}).get("status"),
        "c26s_2": load_json(OUT2 / "C26S_2_SPAN_RELEVANCE_DECISION.json", {}).get("status"),
        "c26s_3": load_json(OUT3 / "C26S_3_FEEDBACK_DECISION.json", {}).get("status"),
        "c26s_4": load_json(OUT4 / "C26S_4_INTEGRATION_DECISION.json", {}).get("status"),
        "c26s_5": load_json(OUT5 / "C26S_5_ROBUSTNESS_DECISION.json", {}).get("status"),
        "c26s_6": final.get("final_decision"),
        "selected_span_ranker": final.get("selected_span_ranker"),
        "selected_feedback_variant": final.get("selected_feedback_variant"),
        "selected_mutual_promotion_formula": final.get("selected_mutual_promotion_formula"),
        "final_vcmr": final.get("final_vcmr_metrics"),
        "final_vr": final.get("final_vr_metrics"),
        "pseudo_used_for_selection": final.get("pseudo_official_holdout_used_for_selection"),
        "official_val_used": final.get("official_val_used"),
        "local_only_artifacts": final.get("local_only_artifacts"),
    }), indent=2), flush=True)


if __name__ == "__main__":
    main()
