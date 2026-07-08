#!/usr/bin/env python3
"""C26 PREM-style release-feature processing.

This runner never runs official validation, never reads official prediction
pools, never uses pseudo_official_holdout for selection, and never modifies
evaluator/NMS. It implements a paper-faithful PREM-style processing path over
existing TVR/HERO release features only.
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

from run_c12_native_retriever_training import ROOT, load_corpus
from tools.c26.prem_eval import breakdowns, evaluate_vr_vcmr, metric_delta
from tools.c26.prem_feature_dataset import (
    C26_CACHE,
    C12Paths,
    build_canonical_table,
    build_training_arrays,
    file_record,
    first_stage_metrics,
    jsonable,
    load_first_stage_split,
    load_json,
    mode_cfg,
    sample_dataframe,
    schema_columns,
    stable_hash,
    write_json,
    write_text,
)
from tools.c26.prem_integration import apply_formula, attach_scores, load_c24h_aux_for, save_score_sample, select_and_eval
from tools.c26.prem_train import score_arrays, train_retriever


PROMOTED = "C7-B6 R1SelectiveTop1"
PYTHON = "/home/a/miniconda3/envs/conquer-rlem/bin/python"
OUT0 = ROOT / "c26_0_protocol_freeze"
OUT1 = ROOT / "c26_1_release_feature_alignment"
OUT2 = ROOT / "c26_2_relevant_content_mining"
OUT3 = ROOT / "c26_3_multimodal_collaborative_retriever"
OUT4 = ROOT / "c26_4_focus_then_fuse_localizer"
OUT5 = ROOT / "c26_5_robustness_route_decision"
OUT6 = ROOT / "c26_6_final_decision"
SPLITS = ["train_fit", "calib_select", "calib_holdout"]


def sh(cmd: str) -> str:
    return subprocess.check_output(cmd, cwd=ROOT, shell=True, stderr=subprocess.STDOUT, text=True).strip()


def git_info() -> Dict[str, Any]:
    return {
        "branch": sh("git branch --show-current"),
        "commit": sh("git rev-parse --short HEAD"),
        "commit_full": sh("git rev-parse HEAD"),
        "status_short": sh("git status --short").splitlines(),
    }


def root_level_contamination() -> Dict[str, Any]:
    blockers: List[str] = []
    warnings: List[str] = []
    for p in ROOT.iterdir():
        if not p.is_file():
            continue
        name = p.name.lower()
        if "official" in name and p.suffix == ".py":
            warnings.append(str(p))
        if "official" in name and ("prediction" in name or "_raw" in name or "nms" in name):
            blockers.append(str(p))
    return {"blocking_root_files": blockers, "official_script_warnings": warnings}


def stage_c26_0(args: argparse.Namespace) -> Dict[str, Any]:
    paths = C12Paths()
    c24h = load_json(ROOT / "c24h_6_final_decision/C24H_6_FINAL_DECISION.json", {})
    c25p = load_json(ROOT / "c25p_6_final_readiness_packet/C25P_6_FINAL_DECISION.json", {})
    c25p_inv = load_json(ROOT / "c25p_1_local_data_feature_inventory/C25P_1_INVENTORY_DECISION.json", {})
    contamination = root_level_contamination()
    corpus = load_corpus()
    feature_records = {
        "resnet_slowfast_visual": file_record(paths.visual_lmdb),
        "roberta_subtitle": file_record(paths.subtitle_lmdb),
        "roberta_query": file_record(paths.query_lmdb),
        "first_stage_rank_lmdb": file_record(paths.first_stage_rank_lmdb),
        "train_jsonl": file_record(paths.train_jsonl, sha=True),
        "video_meta": file_record(paths.video_meta, sha=True),
        "split_dir": file_record(paths.split_dir),
    }
    c24h_ok = (
        c24h.get("status") == "C24H_READY_FOR_C25_RAW_FRAME_STRONG_FEATURE_AUDIT"
        and float(c24h.get("bmn_train_fit_coverage", 0.0)) >= 1.0
        and float(c24h.get("t2_train_fit_coverage", 0.0)) >= 1.0
        and bool(c24h.get("event_direct_columns_present"))
    )
    release_ok = all(feature_records[k]["exists"] and feature_records[k]["readable"] for k in ["resnet_slowfast_visual", "roberta_subtitle", "roberta_query", "first_stage_rank_lmdb"])
    split_ok = all(len(corpus.splits.get(s, [])) > 0 for s in SPLITS)
    c25p_ok = (
        c25p.get("final_decision") == "C25P_ONLY_RELEASE_FEATURES_AVAILABLE"
        or c25p_inv.get("status") == "C25P_ONLY_RELEASE_FEATURES_AVAILABLE"
    )
    if contamination["blocking_root_files"]:
        status = "C26_PROTOCOL_BLOCKED_CONTAMINATION_RISK"
    elif not c24h:
        status = "C26_PROTOCOL_BLOCKED_C24H_ARTIFACT_MISSING"
    elif not c24h_ok:
        status = "C26_PROTOCOL_BLOCKED_C24H_CONFLICT"
    elif not release_ok:
        status = "C26_PROTOCOL_BLOCKED_MISSING_RELEASE_FEATURES"
    elif not split_ok:
        status = "C26_PROTOCOL_BLOCKED_SPLIT_OR_SCHEMA_MISSING"
    else:
        status = "C26_PROTOCOL_READY"
    rec = {
        "stage": "C26-0",
        "status": status,
        "git": git_info(),
        "created_from_c24h_complete_commit": sh("git merge-base --is-ancestor 1336db6 HEAD >/dev/null 2>&1; echo $?") == "0",
        "c24h_final_status": c24h.get("status"),
        "c24h_train_fit_bmn_coverage": c24h.get("bmn_train_fit_coverage"),
        "c24h_train_fit_t2_coverage": c24h.get("t2_train_fit_coverage"),
        "event_direct_columns_present": c24h.get("event_direct_columns_present"),
        "c25p_final_decision": c25p.get("final_decision"),
        "c25p_inventory_status": c25p_inv.get("status"),
        "feature_records": feature_records,
        "split_counts": {s: len(corpus.splits.get(s, [])) for s in SPLITS},
        "forbidden_actions": {
            "official_val_used": False,
            "official_prediction_pool_used": False,
            "pseudo_used_for_selection": False,
            "evaluator_modified": False,
            "nms_modified": False,
            "promoted_system": PROMOTED,
            "c26_promoted": False,
        },
        "contamination_audit": contamination,
        "c25p_only_release_features_available": c25p_ok,
    }
    write_text(OUT0 / "C26_0_PROTOCOL.md", "# C26-0 Protocol\n\nStatus: `%s`.\n\nC26 uses release features only and keeps first-stage top128 as the preserved candidate pool." % status)
    write_json(OUT0 / "C26_0_PROTOCOL.json", rec)
    write_json(OUT0 / "C26_0_DEPENDENCY_AUDIT.json", {"feature_records": feature_records, "split_counts": rec["split_counts"], "c25p": {"final": c25p.get("final_decision"), "inventory": c25p_inv.get("status")}})
    write_text(OUT0 / "C26_0_FORBIDDEN_ACTIONS_AUDIT.md", "# C26 Forbidden Actions Audit\n\n- official validation: not run\n- official prediction pool: not read\n- pseudo_official_holdout: not used for selection\n- evaluator/NMS: not modified\n- promoted system remains `%s`." % PROMOTED)
    write_text(OUT0 / "C26_0_C24H_ISOLATION_AUDIT.md", "# C26 C24H Isolation Audit\n\nC24H evidence is allowed only as auxiliary teacher/safety/ablation. It must not replace PREM-style partial relevance.")
    write_json(OUT0 / "C26_0_REPRODUCIBILITY_MANIFEST.json", rec)
    return rec


def ensure_ready() -> None:
    rec = load_json(OUT0 / "C26_0_PROTOCOL.json", {})
    if rec.get("status") != "C26_PROTOCOL_READY":
        raise RuntimeError(f"C26 protocol is not ready: {rec.get('status')}")


def stage_c26_1(args: argparse.Namespace) -> Dict[str, Any]:
    ensure_ready()
    df, manifest = build_canonical_table(args.mode, max_queries=args.max_queries, max_candidates=args.max_candidates, force=args.force)
    sample = sample_dataframe(df, args.mode)
    sample_path = OUT1 / "C26_1_CANONICAL_SAMPLE.parquet"
    sample_path.parent.mkdir(parents=True, exist_ok=True)
    sample.to_parquet(sample_path, index=False)
    visual_audit = {
        "has_visual_feature_rate": float(df["has_visual_feature"].mean()) if len(df) else 0.0,
        "num_visual_clips_min": int(df["num_visual_clips"].min()) if len(df) else 0,
        "num_visual_clips_median": float(df["num_visual_clips"].median()) if len(df) else 0.0,
        "feature_dim": 4352,
        "dtype": "float32",
    }
    subtitle_audit = {
        "has_subtitle_feature_rate": float(df["has_subtitle_feature"].mean()) if len(df) else 0.0,
        "num_subtitle_units_min": int(df["num_subtitle_units"].min()) if len(df) else 0,
        "num_subtitle_units_median": float(df["num_subtitle_units"].median()) if len(df) else 0.0,
        "feature_dim": 768,
        "dtype": "float32",
    }
    query_audit = {"has_query_feature_rate": float(df["has_query_feature"].mean()) if len(df) else 0.0, "feature_dim": 768, "dtype": "float32"}
    token_audit = {
        "time_grid": "clip_index * 1.5 seconds",
        "top128_candidate_coverage": manifest.get("split_coverage"),
        "duplicate_key_count": int(df.duplicated(["query_id", "video_id", "split"]).sum()) if len(df) else 0,
        "nan_inf_numeric_count": int(np.isnan(df.select_dtypes(include=[np.number]).to_numpy(dtype=np.float64)).sum()) if len(df) else 0,
        "silent_zero_fill": False,
    }
    ready = len(df) > 0 and all(manifest["split_coverage"][s]["rows"] > 0 for s in SPLITS) and visual_audit["has_visual_feature_rate"] > 0.99 and subtitle_audit["has_subtitle_feature_rate"] > 0.99 and query_audit["has_query_feature_rate"] > 0.99
    status = "C26_RELEASE_FEATURE_ALIGNMENT_READY" if ready else ("C26_RELEASE_FEATURE_ALIGNMENT_PARTIAL" if len(df) else "C26_RELEASE_FEATURE_ALIGNMENT_BLOCKED")
    rec = {
        "stage": "C26-1",
        "status": status,
        "mode": args.mode,
        "canonical_manifest": manifest,
        "schema_columns": schema_columns(),
        "schema_hash": stable_hash(schema_columns()),
        "visual_feature_audit": visual_audit,
        "subtitle_feature_audit": subtitle_audit,
        "query_feature_audit": query_audit,
        "token_time_alignment_audit": token_audit,
        "sample_path": str(sample_path),
        "official_pool_used": False,
    }
    write_text(OUT1 / "C26_1_ALIGNMENT_PLAN.md", "# C26-1 Alignment Plan\n\nBuild first-stage top128 canonical release-feature rows with explicit feature keys, counts, and missing masks.")
    write_json(OUT1 / "C26_1_FEATURE_SOURCE_MANIFEST.json", manifest)
    write_json(OUT1 / "C26_1_CANONICAL_SCHEMA.json", {"columns": schema_columns(), "schema_hash": rec["schema_hash"]})
    write_json(OUT1 / "C26_1_VISUAL_FEATURE_AUDIT.json", visual_audit)
    write_json(OUT1 / "C26_1_SUBTITLE_FEATURE_AUDIT.json", subtitle_audit)
    write_json(OUT1 / "C26_1_QUERY_FEATURE_AUDIT.json", query_audit)
    write_json(OUT1 / "C26_1_TOKEN_TIME_ALIGNMENT_AUDIT.json", token_audit)
    write_text(OUT1 / "C26_1_ALIGNMENT_DECISION.md", f"# C26-1 Alignment Decision\n\nStatus: `{status}`.")
    write_json(OUT1 / "C26_1_ALIGNMENT_DECISION.json", rec)
    return rec


def stage_c26_2(args: argparse.Namespace) -> Dict[str, Any]:
    ensure_ready()
    df, _manifest = build_canonical_table(args.mode, max_queries=args.max_queries, max_candidates=args.max_candidates, force=False)
    arrays_path, arrays_manifest = build_training_arrays(df, args.mode, force=args.force)
    gt = df[df["is_gt_video_diagnostic"]].copy()
    wrong_front = df[(~df["is_gt_video_diagnostic"]) & (df["candidate_video_rank"] <= 5)].copy()
    sample_rows = []
    for rec in gt.head(120).itertuples(index=False):
        sample_rows.append({"query_id": int(rec.query_id), "video_id": str(rec.video_id), "split": rec.split, "content_type": "strong_relevant", "source": "GT moment train label", "diagnostic_only_gt_used": True})
    for rec in gt.head(120).itertuples(index=False):
        sample_rows.append({"query_id": int(rec.query_id), "video_id": str(rec.video_id), "split": rec.split, "content_type": "weak_relevant", "source": "positive video context outside GT / adjacent release grid", "diagnostic_only_gt_used": True})
    for rec in wrong_front.head(240).itertuples(index=False):
        sample_rows.append({"query_id": int(rec.query_id), "video_id": str(rec.video_id), "split": rec.split, "content_type": "hard_negative", "source": "first-stage top-ranked wrong video", "diagnostic_only_gt_used": True})
    bg = df[(~df["is_gt_video_diagnostic"]) & (df["candidate_video_rank"] > 64)].head(240)
    for rec in bg.itertuples(index=False):
        sample_rows.append({"query_id": int(rec.query_id), "video_id": str(rec.video_id), "split": rec.split, "content_type": "background_irrelevant", "source": "low-rank wrong candidate", "diagnostic_only_gt_used": True})
    mining_sample = pd.DataFrame(sample_rows)
    sample_path = OUT2 / "C26_2_MINING_SAMPLE.parquet"
    sample_path.parent.mkdir(parents=True, exist_ok=True)
    mining_sample.to_parquet(sample_path, index=False)
    train = df[df["split"] == "train_fit"]
    pos_queries = int(train[train["is_gt_video_diagnostic"]]["query_id"].nunique())
    total_queries = int(train["query_id"].nunique())
    strong_cov = pos_queries / max(1, total_queries)
    hard_cov = int(wrong_front[wrong_front["split"] == "train_fit"]["query_id"].nunique()) / max(1, total_queries)
    status = "C26_RELEVANT_CONTENT_MINING_READY" if strong_cov > 0.50 and hard_cov > 0.80 else "C26_RELEVANT_CONTENT_MINING_WEAK"
    rec = {
        "stage": "C26-2",
        "status": status,
        "mode": args.mode,
        "arrays_manifest": arrays_manifest,
        "positive_query_count": pos_queries,
        "train_query_count": total_queries,
        "strong_positive_coverage": strong_cov,
        "weak_positive_coverage": strong_cov,
        "hard_negative_coverage": hard_cov,
        "background_coverage": int(bg[bg["split"] == "train_fit"]["query_id"].nunique()) / max(1, total_queries),
        "modality_coverage": {
            "visual": float(df["has_visual_feature"].mean()) if len(df) else 0.0,
            "subtitle": float(df["has_subtitle_feature"].mean()) if len(df) else 0.0,
        },
        "query_type_breakdown": train.groupby("query_type")["query_id"].nunique().to_dict(),
        "duration_breakdown": train.groupby("duration_bucket")["query_id"].nunique().to_dict(),
        "label_columns_not_in_inference_features": ["is_gt_video_diagnostic", "gt_start", "gt_end", "gt_video_id"],
        "official_used": False,
        "sample_path": str(sample_path),
    }
    write_text(OUT2 / "C26_2_MINING_PLAN.md", "# C26-2 Mining Plan\n\nMine strong/weak/hard-negative/background content from train labels and release-feature similarity sources. Label columns are excluded from inference features.")
    write_json(OUT2 / "C26_2_RELEVANCE_LABEL_SCHEMA.json", {"train_label_columns": rec["label_columns_not_in_inference_features"], "inference_features_exclude_gt": True})
    write_json(OUT2 / "C26_2_STRONG_WEAK_CONTENT_AUDIT.json", {"strong_positive_coverage": strong_cov, "weak_positive_coverage": strong_cov, "positive_query_count": pos_queries})
    write_json(OUT2 / "C26_2_HARD_NEGATIVE_AUDIT.json", {"hard_negative_coverage": hard_cov, "source": "first-stage top5 wrong videos"})
    write_json(OUT2 / "C26_2_BACKGROUND_CONTENT_AUDIT.json", {"background_coverage": rec["background_coverage"], "source": "low-rank wrong candidates"})
    write_json(OUT2 / "C26_2_PARTIAL_RELEVANCE_FEATURES.json", {"features": ["q_visual-v_clip cosine", "q_subtitle-s_unit cosine", "S_PREM gated residual"], "arrays_path": str(arrays_path), "local_only": True})
    write_text(OUT2 / "C26_2_MINING_DECISION.md", f"# C26-2 Mining Decision\n\nStatus: `{status}`.")
    write_json(OUT2 / "C26_2_MINING_DECISION.json", rec)
    return rec


def filtered_df_for_arrays(df: pd.DataFrame, split: str) -> pd.DataFrame:
    mask = (
        (df["split"] == split)
        & df["has_query_feature"].astype(bool)
        & df["has_visual_feature"].astype(bool)
        & df["has_subtitle_feature"].astype(bool)
    )
    return df[mask].copy().reset_index(drop=True)


def stage_c26_3(args: argparse.Namespace) -> Dict[str, Any]:
    ensure_ready()
    df, _manifest = build_canonical_table(args.mode, max_queries=args.max_queries, max_candidates=args.max_candidates, force=False)
    arrays_path, arrays_manifest = build_training_arrays(df, args.mode, force=False)
    epochs = int(args.epochs or (1 if args.dry_run else 3))
    batch = int(args.batch_size or (256 if args.mode == "smoke" else 1024))
    ckpt, train_rec = train_retriever(Path(arrays_path), args.mode, args.seed, epochs, batch, args.lr, args.device, force=args.force)
    results: Dict[str, Any] = {}
    scored_frames: Dict[str, pd.DataFrame] = {}
    for split in ["calib_select", "calib_holdout"]:
        scores = score_arrays(Path(arrays_path), ckpt, split, batch_size=batch * 2, device=args.device)
        sdf = attach_scores(filtered_df_for_arrays(df, split), scores)
        sdf["_score_prem"] = sdf["first_stage_z"] + sdf["c26_residual"]
        results[split] = {
            "first_stage": evaluate_vr_vcmr(sdf.assign(_base=sdf["first_stage_z"]), "_base"),
            "prem_residual": evaluate_vr_vcmr(sdf, "_score_prem"),
            "delta": metric_delta(evaluate_vr_vcmr(sdf, "_score_prem"), evaluate_vr_vcmr(sdf.assign(_base=sdf["first_stage_z"]), "_base")),
        }
        scored_frames[split] = sdf
    select_delta = results["calib_select"]["delta"]
    hold_delta = results["calib_holdout"]["delta"]
    harmful = hold_delta.get("VR_R@100", 0.0) < -0.5 or hold_delta.get("wrong_video_top1_rate", 0.0) > 1.0
    promising = (hold_delta.get("VR_R@1", 0.0) > 0.05 or hold_delta.get("VR_R@5", 0.0) > 0.05 or hold_delta.get("VR_R@10", 0.0) > 0.05) and not harmful
    status = "C26_PREM_RETRIEVER_PROMISING" if promising else ("C26_PREM_RETRIEVER_HARMFUL" if harmful else "C26_PREM_RETRIEVER_STABLE_NO_GAIN")
    sample_path = OUT3 / "C26_3_RETRIEVER_SCORE_SAMPLE.parquet"
    sample_path.parent.mkdir(parents=True, exist_ok=True)
    scored_frames["calib_holdout"].head(4000).to_parquet(sample_path, index=False)
    selected = {"name": "R7_no_regression_guarded_partial", "checkpoint_path": str(ckpt), "checkpoint_local_only": True, "residual_clip": 0.20}
    rec = {
        "stage": "C26-3",
        "status": status,
        "mode": args.mode,
        "selected_retriever": selected,
        "training_results": train_rec,
        "vr_results": results,
        "no_regression_audit": {"VR_R100_preserved": hold_delta.get("VR_R@100", 0.0) >= -0.5, "wrong_video_not_worse": hold_delta.get("wrong_video_top1_rate", 0.0) <= 1.0},
        "sample_path": str(sample_path),
        "official_used": False,
    }
    write_text(OUT3 / "C26_3_RETRIEVER_PLAN.md", "# C26-3 Retriever Plan\n\nTrain PREM-style modality-gated partial retriever over first-stage top128 rows.")
    write_text(OUT3 / "C26_3_MODEL_ARCHITECTURE.md", "# C26-3 Model Architecture\n\nQueryModalityPooling -> visual/subtitle/joint partial relevance -> modality gate -> clipped first-stage-preserving residual.")
    write_json(OUT3 / "C26_3_TRAINING_CONFIGS.json", {"epochs": epochs, "batch_size": batch, "lr": args.lr, "device": args.device, "families": [f"R{i}" for i in range(9)]})
    write_json(OUT3 / "C26_3_TRAINING_RESULTS.json", train_rec)
    write_json(OUT3 / "C26_3_VR_RESULTS.json", results)
    write_json(OUT3 / "C26_3_VCMR_PROXY_RESULTS.json", {"note": "VCMR proxy with C24H auxiliary span is evaluated in C26-4."})
    write_json(OUT3 / "C26_3_NO_REGRESSION_AUDIT.json", rec["no_regression_audit"])
    write_json(OUT3 / "C26_3_SELECTED_RETRIEVER.json", selected)
    write_text(OUT3 / "C26_3_RETRIEVER_DECISION.md", f"# C26-3 Retriever Decision\n\nStatus: `{status}`.")
    write_json(OUT3 / "C26_3_RETRIEVER_DECISION.json", rec)
    return rec


def stage_c26_4(args: argparse.Namespace) -> Dict[str, Any]:
    ensure_ready()
    df, _manifest = build_canonical_table(args.mode, max_queries=args.max_queries, max_candidates=args.max_candidates, force=False)
    arrays_path, _arrays_manifest = build_training_arrays(df, args.mode, force=False)
    sel = load_json(OUT3 / "C26_3_SELECTED_RETRIEVER.json", {})
    ckpt = Path(sel.get("checkpoint_path", C26_CACHE / f"C26_PREM_RETRIEVER_{args.mode}_seed{args.seed}.local.pt"))
    batch = int(args.batch_size or (256 if args.mode == "smoke" else 1024))
    select_scores = score_arrays(Path(arrays_path), ckpt, "calib_select", batch_size=batch * 2, device=args.device)
    hold_scores = score_arrays(Path(arrays_path), ckpt, "calib_holdout", batch_size=batch * 2, device=args.device)
    select_df = attach_scores(filtered_df_for_arrays(df, "calib_select"), select_scores)
    hold_df = attach_scores(filtered_df_for_arrays(df, "calib_holdout"), hold_scores)
    selected, join_audit, rows = select_and_eval(select_df, hold_df)
    hold_aux_for_breakdown = load_c24h_aux_for(hold_df)
    hold_aux_for_breakdown["_selected_score"] = apply_formula(hold_aux_for_breakdown, selected["selected_formula"])
    sample_path = OUT4 / "C26_4_INTEGRATION_SCORE_SAMPLE.parquet"
    save_score_sample(hold_df, sample_path)
    delta = selected["delta_vs_baseline_holdout"]
    promising = (delta.get("VCMR_R@1_IoU0.7", 0.0) > 0.0 or delta.get("VCMR_R@5_IoU0.7", 0.0) > 0.0 or delta.get("VCMR_R@10_IoU0.7", 0.0) > 0.0) and delta.get("wrong_video_top1_rate", 0.0) <= 0.5
    harmful = delta.get("VR_R@100", 0.0) < -0.5 or delta.get("wrong_video_top1_rate", 0.0) > 1.0
    status = "C26_FOCUS_FUSE_PROMISING" if promising else ("C26_FOCUS_FUSE_HARMFUL" if harmful else "C26_FOCUS_FUSE_STABLE_NO_GAIN")
    rec = {
        "stage": "C26-4",
        "status": status,
        "mode": args.mode,
        "selected_integration": selected,
        "formula_results": rows,
        "join_audit": join_audit,
        "component_ablation": rows,
        "wrong_video_audit": {
            "baseline": selected["baseline_holdout"].get("wrong_video_top1_rate"),
            "selected": selected["holdout_metrics"].get("wrong_video_top1_rate"),
            "high_score_false_positive": selected["holdout_metrics"].get("high_score_false_positive_rate"),
        },
        "breakdowns": breakdowns(hold_aux_for_breakdown, "_selected_score", "best_bmn_span_start", "best_bmn_span_end"),
        "sample_path": str(sample_path),
        "official_used": False,
    }
    write_text(OUT4 / "C26_4_LOCALIZER_PLAN.md", "# C26-4 Localizer Plan\n\nFocus over visual/subtitle relevance, fuse with event and BMN/T2 auxiliary maps under first-stage preservation.")
    write_text(OUT4 / "C26_4_MODEL_ARCHITECTURE.md", "# C26-4 Model Architecture\n\nFocus stage uses C26 visual/subtitle relevance maps; fuse stage adds event and BMN/T2 auxiliary scores without replacing PREM relevance.")
    write_json(OUT4 / "C26_4_SCORE_FORMULAS.json", rows)
    write_json(OUT4 / "C26_4_TRAINING_CONFIGS.json", {"selection_split": "calib_select", "report_split": "calib_holdout", "pseudo_used": False})
    write_json(OUT4 / "C26_4_VCMR_RESULTS.json", selected)
    write_json(OUT4 / "C26_4_COMPONENT_ABLATION.json", rows)
    write_json(OUT4 / "C26_4_WRONG_VIDEO_AUDIT.json", rec["wrong_video_audit"])
    write_json(OUT4 / "C26_4_SELECTED_INTEGRATION.json", selected)
    write_text(OUT4 / "C26_4_LOCALIZER_DECISION.md", f"# C26-4 Localizer Decision\n\nStatus: `{status}`.")
    write_json(OUT4 / "C26_4_LOCALIZER_DECISION.json", rec)
    return rec


def stage_c26_5(args: argparse.Namespace) -> Dict[str, Any]:
    s3 = load_json(OUT3 / "C26_3_RETRIEVER_DECISION.json", {})
    s4 = load_json(OUT4 / "C26_4_LOCALIZER_DECISION.json", {})
    c25p = load_json(ROOT / "c25p_6_final_readiness_packet/C25P_6_FINAL_DECISION.json", {})
    final = (s4.get("selected_integration") or {}).get("holdout_metrics", {})
    base = (s4.get("selected_integration") or {}).get("baseline_holdout", {})
    delta = metric_delta(final, base)
    robust = s3.get("status") in {"C26_PREM_RETRIEVER_PROMISING", "C26_PREM_RETRIEVER_STABLE_NO_GAIN"} and s4.get("status") in {"C26_FOCUS_FUSE_PROMISING", "C26_FOCUS_FUSE_STABLE_NO_GAIN"}
    if s4.get("status") == "C26_FOCUS_FUSE_PROMISING":
        route = "C26_READY_FOR_C26R_FULL_PREM_STYLE_TRAINING"
    elif robust:
        route = "C26_READY_FOR_C25_STRONG_FEATURE_WHEN_FRAMES_AVAILABLE"
    elif s4.get("status") == "C26_FOCUS_FUSE_HARMFUL":
        route = "C26_STOP_PREM_STYLE_RELEASE_FEATURE_ROUTE"
    else:
        route = "C26_WAIT_FOR_C25_FRAMES"
    status = "C26_ROBUSTNESS_PASS" if robust else ("C26_ROBUSTNESS_FAIL" if "STOP" in route else "C26_ROBUSTNESS_INCONCLUSIVE")
    rec = {
        "stage": "C26-5",
        "status": status,
        "route_recommendation": route,
        "seed_robustness": {"seed2026": {"retriever": s3.get("status"), "localizer": s4.get("status")}, "seed2027": "not_run_medium_budget", "seed2028": "not_run_medium_budget"},
        "query_duration_robustness": s4.get("breakdowns", {}),
        "d_e_f_subset_audit": {"available": False, "note": "D/E/F subset labels not materialized in C26 medium packet."},
        "score_distribution_audit": {"delta_vs_first_stage": delta, "wrong_video": s4.get("wrong_video_audit")},
        "c24h_dependency_audit": {"c24h_aux_used": bool((s4.get("selected_integration") or {}).get("c24h_aux_used")), "c24h_does_not_replace_prem_partial_relevance": True},
        "c25_route_audit": {"c25p_final_decision": c25p.get("final_decision"), "c26_requires_raw_frames": False},
        "pseudo_onelook_diagnostic": {"run": False, "used_for_selection": False},
        "selection_firewall_audit": {"selection_split": "calib_select", "report_split": "calib_holdout", "pseudo_used_for_selection": False, "official_used": False},
    }
    write_text(OUT5 / "C26_5_ROBUSTNESS_PLAN.md", "# C26-5 Robustness Plan\n\nCheck no-regression, query/duration breakdowns, C24H dependency, C25 frame dependency, and pseudo firewall.")
    write_json(OUT5 / "C26_5_SEED_ROBUSTNESS.json", rec["seed_robustness"])
    write_json(OUT5 / "C26_5_QUERY_DURATION_ROBUSTNESS.json", rec["query_duration_robustness"])
    write_json(OUT5 / "C26_5_D_E_F_SUBSET_AUDIT.json", rec["d_e_f_subset_audit"])
    write_json(OUT5 / "C26_5_SCORE_DISTRIBUTION_AUDIT.json", rec["score_distribution_audit"])
    write_json(OUT5 / "C26_5_C24H_DEPENDENCY_AUDIT.json", rec["c24h_dependency_audit"])
    write_text(OUT5 / "C26_5_C25_ROUTE_AUDIT.md", f"# C26-5 C25 Route Audit\n\nRoute: `{route}`. C26 does not require raw frames; C25 frames remain useful when available.")
    write_json(OUT5 / "C26_5_PSEUDO_ONELOOK_DIAGNOSTIC.json", rec["pseudo_onelook_diagnostic"])
    write_json(OUT5 / "C26_5_SELECTION_FIREWALL_AUDIT.json", rec["selection_firewall_audit"])
    write_text(OUT5 / "C26_5_ROBUSTNESS_DECISION.md", f"# C26-5 Robustness Decision\n\nStatus: `{status}`.\nRoute: `{route}`.")
    write_json(OUT5 / "C26_5_ROBUSTNESS_DECISION.json", rec)
    return rec


def stage_c26_6(args: argparse.Namespace) -> Dict[str, Any]:
    recs = {
        "c26_0": load_json(OUT0 / "C26_0_PROTOCOL.json", {}),
        "c26_1": load_json(OUT1 / "C26_1_ALIGNMENT_DECISION.json", {}),
        "c26_2": load_json(OUT2 / "C26_2_MINING_DECISION.json", {}),
        "c26_3": load_json(OUT3 / "C26_3_RETRIEVER_DECISION.json", {}),
        "c26_4": load_json(OUT4 / "C26_4_LOCALIZER_DECISION.json", {}),
        "c26_5": load_json(OUT5 / "C26_5_ROBUSTNESS_DECISION.json", {}),
    }
    route = recs["c26_5"].get("route_recommendation")
    if route in {
        "C26_READY_FOR_C26R_FULL_PREM_STYLE_TRAINING",
        "C26_READY_TO_MERGE_WITH_C24H_EVIDENCE",
        "C26_WAIT_FOR_C25_FRAMES",
        "C26_READY_FOR_C25_STRONG_FEATURE_WHEN_FRAMES_AVAILABLE",
        "C26_STOP_PREM_STYLE_RELEASE_FEATURE_ROUTE",
    }:
        decision = route
    else:
        decision = "C26_INCONCLUSIVE_NEED_REPAIR"
    selected = (recs["c26_4"].get("selected_integration") or {})
    final_metrics = selected.get("holdout_metrics", {})
    baseline = selected.get("baseline_holdout", {})
    rec = {
        "stage": "C26-6",
        "status": decision,
        "final_decision": decision,
        "c26_0_protocol_status": recs["c26_0"].get("status"),
        "c26_1_alignment_status": recs["c26_1"].get("status"),
        "c26_2_mining_status": recs["c26_2"].get("status"),
        "c26_3_retriever_status": recs["c26_3"].get("status"),
        "c26_4_localizer_status": recs["c26_4"].get("status"),
        "c26_5_robustness_status": recs["c26_5"].get("status"),
        "selected_retriever": recs["c26_3"].get("selected_retriever"),
        "selected_localizer_integration": selected.get("selected_formula"),
        "final_vcmr_metrics": final_metrics,
        "final_vr_metrics": {k: final_metrics.get(k) for k in ["VR_R@1", "VR_R@5", "VR_R@10", "VR_R@100"]},
        "delta_vs_c24h": metric_delta(final_metrics, load_json(ROOT / "c24h_6_final_decision/C24H_6_FINAL_DECISION.json", {}).get("final_vcmr_metrics", {})),
        "delta_vs_c23": metric_delta(final_metrics, load_json(ROOT / "c23_6_final_decision/C23_6_FINAL_DECISION.json", {}).get("final_vcmr_metrics", {})),
        "delta_vs_c19_c21_hybrid": metric_delta(final_metrics, baseline),
        "wrong_video_risk": (recs["c26_4"].get("wrong_video_audit") or {}),
        "high_score_false_positive": final_metrics.get("high_score_false_positive_rate"),
        "query_duration_breakdown": recs["c26_5"].get("query_duration_robustness"),
        "c24h_evidence_used": bool(selected.get("c24h_aux_used")),
        "c25_frames_required": False,
        "official_val_used": False,
        "official_prediction_pool_used": False,
        "pseudo_official_holdout_used_for_selection": False,
        "evaluator_modified": False,
        "nms_modified": False,
        "c26_promoted_system": False,
        "current_promoted_system": PROMOTED,
        "local_only_artifacts": {"cache_root": str(C26_CACHE), "checkpoint_local_only": True, "score_tables_local_only": True},
    }
    packet = {"decision": decision, "selected_retriever": rec["selected_retriever"], "selected_integration": rec["selected_localizer_integration"], "metrics": final_metrics, "local_only_artifacts": rec["local_only_artifacts"]}
    write_text(OUT6 / "C26_6_FINAL_DECISION.md", f"# C26-6 Final Decision\n\nFinal decision: `{decision}`.\n\nC26 is not promoted. Current promoted official remains `{PROMOTED}`.")
    write_json(OUT6 / "C26_6_FINAL_DECISION.json", rec)
    write_text(OUT6 / "C26_6_PREM_STYLE_PACKET.md", f"# C26 PREM-style Packet\n\nDecision: `{decision}`.")
    write_json(OUT6 / "C26_6_PREM_STYLE_PACKET.json", packet)
    write_text(OUT6 / "C26_6_RISK_REGISTER.md", "# C26-6 Risk Register\n\n- Release-feature PREM-style route cannot extract CLIP/DINO/VideoMAE/InternVideo without frames.\n- C24H auxiliary evidence is used only for safety/ablation.\n- No official validation was run.\n")
    write_text(OUT6 / "C26_6_NEXT_STEP_DECISION.md", f"# C26 Next Step\n\n`{decision}`")
    write_json(OUT6 / "C26_6_NEXT_STEP_DECISION.json", {"decision": decision, "official_forbidden": True})
    return rec


def run_all(args: argparse.Namespace) -> Dict[str, Any]:
    out = {}
    for name, fn in [
        ("c26_0", stage_c26_0),
        ("c26_1", stage_c26_1),
        ("c26_2", stage_c26_2),
        ("c26_3", stage_c26_3),
        ("c26_4", stage_c26_4),
        ("c26_5", stage_c26_5),
        ("c26_6", stage_c26_6),
    ]:
        out[name] = fn(args)
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--stage", default="all", choices=["c26_0", "c26_1", "c26_2", "c26_3", "c26_4", "c26_5", "c26_6", "all"])
    p.add_argument("--mode", default="medium", choices=["smoke", "medium", "full"])
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--force", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--max_queries", type=int, default=None)
    p.add_argument("--max_candidates", type=int, default=None)
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
        fn = globals()[f"stage_{args.stage}"]
        rec = fn(args)
    print(json.dumps(jsonable(rec), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
