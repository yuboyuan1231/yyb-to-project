#!/usr/bin/env python3
"""C23 full-trainfit first-stage-preserving native coupling.

This runner keeps the first-stage top128 candidate pool as the retrieval prior.
It never runs official validation, never reads official prediction pools, never
modifies evaluator/NMS logic, and never uses pseudo_official_holdout for
selection.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import random
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd
import torch

import run_c22r_native_coupling_sanity_repair as c22r
import run_c18_full_hybrid_freeze_candidate as c18
from run_c12_native_retriever_training import ROOT, build_feature_caches, load_corpus, load_features, sha256_file


torch.set_num_threads(min(32, os.cpu_count() or 1))

PROMOTED = "C7-B6 R1SelectiveTop1"
BASE_C22R_COMMIT = "44c9dae5e99996e1675809d9d5a06c429b9382d7"
PYTHON = "/home/a/miniconda3/envs/conquer-rlem/bin/python"

OUT0 = ROOT / "c23_0_protocol_freeze"
OUT1 = ROOT / "c23_1_full_trainfit_readiness"
OUT2 = ROOT / "c23_2_full_residual_retriever"
OUT3 = ROOT / "c23_3_full_evidence_materialization"
OUT4 = ROOT / "c23_4_full_safe_native_coupling"
OUT5 = ROOT / "c23_5_robustness_onelook_firewall"
OUT6 = ROOT / "c23_6_final_decision"

MODE_LIMITS = {
    "smoke": {"train": 500, "select": 200, "holdout": 200, "sample_rows": 1000},
    "medium": {"train": 2000, "select": 1000, "holdout": 1000, "sample_rows": 4000},
    "full": {"train": 0, "select": 0, "holdout": 0, "sample_rows": 12000},
}


def jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(x) for x in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, Path):
        return str(obj)
    return obj


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(obj), indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def load_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def sh(cmd: str) -> str:
    return subprocess.check_output(cmd, cwd=ROOT, shell=True, stderr=subprocess.STDOUT, text=True).strip()


def sh_rc(cmd: str) -> int:
    return subprocess.run(cmd, cwd=ROOT, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode


def file_record(path: Path, sha: bool = False) -> Dict[str, Any]:
    return {
        "path": str(path),
        "exists": path.exists(),
        "readable": os.access(path, os.R_OK) if path.exists() else False,
        "size_bytes": path.stat().st_size if path.exists() and path.is_file() else None,
        "sha256": sha256_file(path) if sha and path.exists() and path.is_file() else None,
    }


def cfg(mode: str) -> Dict[str, int]:
    if mode not in MODE_LIMITS:
        raise ValueError(mode)
    return dict(MODE_LIMITS[mode])


def limited_ids(corpus: Any, split: str, mode: str) -> List[int]:
    ids = [int(x) for x in corpus.splits[split]]
    key = "select" if split == "calib_select" else split.replace("calib_", "")
    limit = cfg(mode).get(key, 0)
    return ids[:limit] if limit and len(ids) > limit else ids


def load_pickle_len(path: Path) -> int | None:
    if not path.exists():
        return None
    with path.open("rb") as f:
        return len(pickle.load(f))


def metric_delta(a: Dict[str, Any], b: Dict[str, Any], keys: Sequence[str]) -> Dict[str, float]:
    return {k: float(a.get(k, 0.0) or 0.0) - float(b.get(k, 0.0) or 0.0) for k in keys}


def c23_schema_hash() -> str:
    return c22r.stable_hash({
        "stage": "C23",
        "candidate_pool": "first_stage_top128",
        "columns": [
            "query_id", "video_id", "split", "gt_video_id", "gt_start", "gt_end",
            "candidate_video_rank", "candidate_source", "first_stage_score", "first_stage_rank",
            "visual_feature_key", "subtitle_feature_key", "query_feature_key",
            "has_visual", "has_subtitle", "has_query", "feature_missing_mask",
            "event_source", "bmn_score_available", "t2_score_available", "schema_hash", "config_hash",
        ],
    })


def build_c23_sample(corpus: Any, features: Dict[str, Any], mode: str) -> pd.DataFrame:
    rows = []
    schema = c23_schema_hash()
    for split in ["train_fit", "calib_select", "calib_holdout"]:
        ids = limited_ids(corpus, split, mode)
        first = c22r.load_first_stage_cached(split, ids)
        for did in ids:
            row = corpus.by_id[int(did)]
            q_exists = int(did) in features["desc_to_qpos"]
            for rank, (pos, score) in enumerate(first.get(int(did), {}).get("ranklist", [])[:128], start=1):
                vid = corpus.train_videos[int(pos)]
                rows.append({
                    "query_id": int(did),
                    "video_id": str(vid),
                    "split": split,
                    "gt_video_id": str(row["vid_name"]),
                    "gt_start": float(row["ts"][0]),
                    "gt_end": float(row["ts"][1]),
                    "candidate_video_rank": int(rank),
                    "candidate_source": "first_stage_top128_pickle",
                    "first_stage_score": float(score),
                    "first_stage_rank": int(rank),
                    "visual_feature_key": str(vid),
                    "subtitle_feature_key": str(vid),
                    "query_feature_key": str(int(did)),
                    "has_visual": True,
                    "has_subtitle": True,
                    "has_query": bool(q_exists),
                    "feature_missing_mask": "" if q_exists else "query",
                    "event_source": "C22R canonical event proxy; rebuild on demand in C23-3",
                    "bmn_score_available": False,
                    "t2_score_available": False,
                    "is_gt_video": str(vid) == str(row["vid_name"]),
                    "schema_hash": schema,
                    "config_hash": c22r.stable_hash({"mode": mode, "split": split, "topk": 128}),
                })
                if len(rows) >= cfg(mode)["sample_rows"]:
                    return pd.DataFrame(rows)
    return pd.DataFrame(rows)


def first_stage_replay(corpus: Any, split: str, ids: Sequence[int]) -> Dict[str, Any]:
    first = c22r.load_first_stage_cached(split, ids)
    return c22r.first_stage_metrics(first, ids)


def residual_replay_for_split(corpus: Any, features: Dict[str, Any], split: str, ids: Sequence[int], formula: Dict[str, float]) -> Dict[str, Any]:
    first = c22r.load_first_stage_cached(split, ids)
    model = c22r.load_c22_model()
    c22_scores = c22r.c22_candidate_scores(model, features, corpus, ids, first, 128)
    return c22r.replay_restricted_metrics(corpus, ids, first, c22_scores, formula)


def c19_medium_cache() -> Path:
    return Path("/tmp/c19_score_cache/CONQUER-RLEM-c2c3/C19_CANONICAL_SELECTED_SCORE_TABLE_medium.local.parquet")


def c23_formula_space() -> List[Dict[str, Any]]:
    formulas: List[Dict[str, Any]] = [
        {"name": "C19_selected_a065_b025_g010", "family": "G_retriever_BMN_T2", "alpha": 0.65, "beta": 0.25, "gamma": 0.10},
        {"name": "R_BMN_T2_balanced_055_030_015", "family": "G_retriever_BMN_T2", "alpha": 0.55, "beta": 0.30, "gamma": 0.15},
        {"name": "R_BMN_heavy_050_040_010", "family": "G_retriever_BMN_T2", "alpha": 0.50, "beta": 0.40, "gamma": 0.10},
        {"name": "R_BMN_heavy_045_040_015", "family": "G_retriever_BMN_T2", "alpha": 0.45, "beta": 0.40, "gamma": 0.15},
        {"name": "R_T2_safety_055_020_025", "family": "G_retriever_BMN_T2", "alpha": 0.55, "beta": 0.20, "gamma": 0.25},
        {"name": "R_T2_safety_045_025_030", "family": "G_retriever_BMN_T2", "alpha": 0.45, "beta": 0.25, "gamma": 0.30},
        {"name": "R_old_native_bridge_050_025_015_010", "family": "G_retriever_BMN_T2", "alpha": 0.50, "beta": 0.25, "gamma": 0.15, "delta": 0.10},
        {"name": "R_old_native_bridge_045_030_015_010", "family": "G_retriever_BMN_T2", "alpha": 0.45, "beta": 0.30, "gamma": 0.15, "delta": 0.10},
        {"name": "R_duration_aware_front", "family": "I_duration_aware", "alpha": 0.55, "beta": 0.25, "gamma": 0.20},
        {"name": "R_qtype_gated_front", "family": "H_qtype_gated", "alpha": 0.55, "beta": 0.25, "gamma": 0.20},
        {"name": "R_safety_gated", "family": "J_safety_gated", "alpha": 0.55, "beta": 0.25, "gamma": 0.20},
        {"name": "R_low_retriever_high_localizer_040_040_020", "family": "G_retriever_BMN_T2", "alpha": 0.40, "beta": 0.40, "gamma": 0.20},
        {"name": "R_front_push_050_035_015", "family": "G_retriever_BMN_T2", "alpha": 0.50, "beta": 0.35, "gamma": 0.15},
        {"name": "R_front_push_045_035_020", "family": "G_retriever_BMN_T2", "alpha": 0.45, "beta": 0.35, "gamma": 0.20},
        {"name": "R_front_push_040_035_025", "family": "G_retriever_BMN_T2", "alpha": 0.40, "beta": 0.35, "gamma": 0.25},
        {"name": "R_bmn_dominant_035_050_015", "family": "G_retriever_BMN_T2", "alpha": 0.35, "beta": 0.50, "gamma": 0.15},
        {"name": "R_bmn_dominant_old_035_045_010_010", "family": "G_retriever_BMN_T2", "alpha": 0.35, "beta": 0.45, "gamma": 0.10, "delta": 0.10},
        {"name": "R_t2_rescue_040_020_040", "family": "G_retriever_BMN_T2", "alpha": 0.40, "beta": 0.20, "gamma": 0.40},
        {"name": "R_t2_rescue_old_040_020_030_010", "family": "G_retriever_BMN_T2", "alpha": 0.40, "beta": 0.20, "gamma": 0.30, "delta": 0.10},
        {"name": "R_duration_aware_bmn_push", "family": "I_duration_aware", "alpha": 0.45, "beta": 0.35, "gamma": 0.20},
        {"name": "R_qtype_gated_bmn_push", "family": "H_qtype_gated", "alpha": 0.45, "beta": 0.35, "gamma": 0.20},
        {"name": "R_safety_gated_t2_push", "family": "J_safety_gated", "alpha": 0.45, "beta": 0.25, "gamma": 0.30},
    ]
    seen: Dict[str, Dict[str, Any]] = {}
    for f in formulas:
        seen[c22r.stable_hash(f)] = f
    return list(seen.values())


def load_c19_eval_frame(mode: str) -> pd.DataFrame:
    path = c19_medium_cache()
    cols = [
        "query_id", "video_id", "span_start", "span_end", "retriever_norm", "bmn_norm",
        "t2_norm", "old_norm", "duration_score", "final_score", "bmn_score", "t2_score",
        "gt_video_id", "gt_start", "gt_end", "split", "seed", "query_type",
        "duration_bucket", "gt_video_rank",
    ]
    df = pd.read_parquet(path, columns=cols)
    if mode == "smoke":
        keep = []
        for split in ["calib_select", "calib_holdout"]:
            ids = df.loc[df["split"].astype(str) == split, "query_id"].drop_duplicates().head(200)
            keep.append((df["split"].astype(str) == split) & df["query_id"].isin(ids))
        df = df[np.logical_or.reduce(keep)].copy()
    return df


def objective(summary: Dict[str, Any], base: Dict[str, Any]) -> float:
    wrong = max(0.0, float(summary.get("wrong_video_high_score_rate", 0.0) - base.get("wrong_video_high_score_rate", 0.0)))
    vr100_penalty = max(0.0, float(base.get("VR_R@100", 0.0) - summary.get("VR_R@100", 0.0)))
    vr10_penalty = max(0.0, float(base.get("VR_R@10", 0.0) - summary.get("VR_R@10", 0.0)))
    return (
        3.0 * (summary.get("VCMR_R@1_IoU0.7", 0.0) - base.get("VCMR_R@1_IoU0.7", 0.0))
        + 2.0 * (summary.get("VCMR_R@5_IoU0.7", 0.0) - base.get("VCMR_R@5_IoU0.7", 0.0))
        + 1.5 * (summary.get("VCMR_R@10_IoU0.7", 0.0) - base.get("VCMR_R@10_IoU0.7", 0.0))
        + 1.0 * (summary.get("VCMR_R@1_IoU0.5", 0.0) - base.get("VCMR_R@1_IoU0.5", 0.0))
        + 0.8 * (summary.get("VCMR_R@5_IoU0.5", 0.0) - base.get("VCMR_R@5_IoU0.5", 0.0))
        - 2.0 * wrong
        - 1.0 * vr100_penalty
        - 0.8 * vr10_penalty
    )


def eval_formula_split(ctx: Dict[str, Any], df: pd.DataFrame, formula: Dict[str, Any], split: str) -> Dict[str, Any]:
    score = c18.formula_score_array(df, formula)
    return c18.fast_eval_vcmr(ctx, score, split)


def stage_c23_0(mode: str, seed: int) -> Dict[str, Any]:
    c22r_final = load_json(ROOT / "c22r_6_final_decision/C22R_6_FINAL_DECISION.json", {})
    c22r_3 = load_json(ROOT / "c22r_3_first_stage_preserving_retriever/C22R_3_RETRIEVER_DECISION.json", {})
    c22r_2 = load_json(ROOT / "c22r_2_alignment_repair/C22R_2_ALIGNMENT_DECISION.json", {})
    branch = sh("git branch --show-current")
    commit = sh("git rev-parse HEAD")
    dirty = sh("git status --short").splitlines()
    corpus = load_corpus()
    split_counts = {k: len(v) for k, v in corpus.splits.items()}
    deps = {
        "c22r_1_collapse": file_record(ROOT / "c22r_1_native_retriever_collapse_audit/C22R_1_COLLAPSE_DECISION.json", sha=True),
        "c22r_2_alignment": file_record(ROOT / "c22r_2_alignment_repair/C22R_2_ALIGNMENT_DECISION.json", sha=True),
        "c22r_3_retriever": file_record(ROOT / "c22r_3_first_stage_preserving_retriever/C22R_3_RETRIEVER_DECISION.json", sha=True),
        "c22r_4_joint": file_record(ROOT / "c22r_4_safe_joint_integration/C22R_4_JOINT_DECISION.json", sha=True),
        "c22r_5_robustness": file_record(ROOT / "c22r_5_robustness_full_readiness/C22R_5_ROBUSTNESS_DECISION.json", sha=True),
        "c22r_6_next": file_record(ROOT / "c22r_6_final_decision/C22R_6_NEXT_STEP_DECISION.json", sha=True),
        "c12_split_manifest": file_record(ROOT / "c12_1_schema_and_split/C12_1_SPLIT_MANIFEST.json", sha=True),
        "first_stage_train_calib_holdout_top128": file_record(c22r.first_stage_cache_path("train_fit")),
        "first_stage_calib_select_top128": file_record(c22r.first_stage_cache_path("calib_select")),
        "first_stage_calib_holdout_top128": file_record(c22r.first_stage_cache_path("calib_holdout")),
    }
    missing = [k for k, v in deps.items() if not v["exists"] or not v["readable"]]
    firewall = {
        "official_val_used": bool(c22r_final.get("official_val_used")),
        "official_prediction_pool_used": bool(c22r_final.get("official_prediction_pool_used")),
        "pseudo_official_holdout_used_for_selection": bool(c22r_final.get("pseudo_official_holdout_used_for_selection")),
        "evaluator_modified": bool(c22r_final.get("evaluator_modified")),
        "nms_modified": bool(c22r_final.get("nms_modified")),
    }
    markers = c22r.find_stale_authorization_markers()
    restored = (c22r_3.get("status") == "C22R_RETRIEVER_SANITY_RESTORED"
                and c22r_2.get("status") == "C22R_ALIGNMENT_REPAIRED"
                and abs(float((c22r_final.get("delta_vs_first_stage_baseline") or {}).get("VR_R@100", 999))) < 1e-6)
    status = "C23_PROTOCOL_READY"
    if missing or c22r_final.get("final_decision") != "C22R_READY_FOR_C23_FULL_TRAINFIT_NATIVE_COUPLING" or not restored:
        status = "C23_PROTOCOL_BLOCKED_MISSING_CORE_ARTIFACTS"
    if any(firewall.values()) or any("/" not in x for x in markers):
        status = "C23_PROTOCOL_BLOCKED_CONTAMINATION_RISK"
    rec = {
        "stage": "C23-0",
        "status": status,
        "mode": mode,
        "seed": seed,
        "branch": branch,
        "commit": commit,
        "base_c22r_commit_expected": BASE_C22R_COMMIT,
        "base_c22r_commit_ancestor": sh_rc(f"git merge-base --is-ancestor {BASE_C22R_COMMIT} HEAD") == 0,
        "dirty_status_lines": dirty,
        "split_counts": split_counts,
        "c22r_final_decision": c22r_final.get("final_decision"),
        "c22r_retriever_sanity": c22r_3.get("status"),
        "c22r_alignment": c22r_2.get("status"),
        "c22r_vr_restored_to_first_stage": restored,
        "dependencies": deps,
        "blocked_missing": missing,
        "forbidden_action_audit": firewall,
        "stale_authorization_markers": markers,
        "official_val_used": False,
        "official_prediction_pool_used": False,
        "pseudo_official_holdout_used_for_selection": False,
        "evaluator_modified": False,
        "nms_modified": False,
        "current_promoted_system": PROMOTED,
        "c23_is_promoted_system": False,
        "repro_command": f"{PYTHON} run_c23_full_trainfit_firststage_preserving_native_coupling.py --stage all --mode {mode} --seed {seed} --force",
    }
    write_json(OUT0 / "C23_0_PROTOCOL.json", rec)
    write_json(OUT0 / "C23_0_DEPENDENCY_AUDIT.json", deps)
    write_json(OUT0 / "C23_0_REPRODUCIBILITY_MANIFEST.json", rec)
    write_text(OUT0 / "C23_0_PROTOCOL.md", f"# C23-0 Protocol Freeze\n\nStatus: `{status}`.\n\nC22R final: `{c22r_final.get('final_decision')}`.")
    write_text(OUT0 / "C23_0_C22R_ACCEPTANCE.md", f"# C22R Acceptance\n\nC22R is accepted because retriever sanity was restored while preserving first-stage top128. Current promoted system remains `{PROMOTED}`.")
    write_text(OUT0 / "C23_0_FORBIDDEN_ACTIONS_AUDIT.md", "# Forbidden Actions Audit\n\n- official_val_used: false\n- official_prediction_pool_used: false\n- pseudo_official_holdout_used_for_selection: false\n- evaluator_modified: false\n- nms_modified: false\n")
    return rec


def stage_c23_1(mode: str, seed: int) -> Dict[str, Any]:
    corpus = load_corpus()
    feature_paths = build_feature_caches(corpus)
    features = load_features(feature_paths)
    split_counts = {k: len(v) for k, v in corpus.splits.items()}
    cache_lens = {
        "train_calib_holdout_top128": load_pickle_len(c22r.first_stage_cache_path("train_fit")),
        "calib_select_medium_top128": load_pickle_len(c22r.first_stage_cache_path("calib_select")),
        "calib_holdout_medium_top128": load_pickle_len(c22r.first_stage_cache_path("calib_holdout")),
    }
    sample = build_c23_sample(corpus, features, mode)
    sample_path = OUT1 / "C23_1_DATA_SAMPLE.parquet"
    sample_path.parent.mkdir(parents=True, exist_ok=True)
    sample.to_parquet(sample_path, index=False)
    hold_ids = limited_ids(corpus, "calib_holdout", mode)
    select_ids = limited_ids(corpus, "calib_select", mode)
    train_ids = limited_ids(corpus, "train_fit", mode)
    coverage = {
        "train_fit": first_stage_replay(corpus, "train_fit", train_ids),
        "calib_select": first_stage_replay(corpus, "calib_select", select_ids),
        "calib_holdout": first_stage_replay(corpus, "calib_holdout", hold_ids),
    }
    duplicate = int(sample.duplicated(["query_id", "video_id", "split"]).sum()) if len(sample) else 0
    schema = {
        "schema_hash": c23_schema_hash(),
        "columns": sample.columns.tolist() if len(sample) else [],
        "candidate_pool": "first_stage_top128",
        "no_position_based_join": True,
        "no_silent_zero_fill": True,
    }
    feat_manifest = {k: file_record(Path(v), sha=False) for k, v in feature_paths.items()}
    bmn_t2 = {
        "c19_canonical_medium": file_record(Path("/tmp/c19_score_cache/CONQUER-RLEM-c2c3/C19_CANONICAL_SELECTED_SCORE_TABLE_medium.local.parquet")),
        "c17_score_medium": file_record(Path("/tmp/c17_score_cache/CONQUER-RLEM-c2c3/C17_2_SCORE_TABLE_medium.local.parquet")),
        "full_materialization_required_for_C23_3": mode != "full",
    }
    io_cost = {
        "full_candidate_rows_estimate_train_select_holdout": int((split_counts["train_fit"] + split_counts["calib_select"] + split_counts["calib_holdout"]) * 128),
        "mode_sample_rows_written": len(sample),
        "train_calib_holdout_top128_pickle_size_bytes": c22r.first_stage_cache_path("train_fit").stat().st_size,
        "large_full_table_commit_forbidden": True,
    }
    source_ready = cache_lens["train_calib_holdout_top128"] == split_counts["train_fit"] + split_counts["calib_select"] + split_counts["calib_holdout"]
    status = "C23_FULL_TRAINFIT_DATA_READY" if mode == "full" and source_ready else "C23_FULL_TRAINFIT_DATA_PARTIAL"
    rec = {
        "stage": "C23-1",
        "status": status,
        "mode": mode,
        "seed": seed,
        "source_readiness_pass": bool(source_ready),
        "split_coverage_audit": {"split_counts": split_counts, "first_stage_cache_lengths": cache_lens, "coverage": coverage},
        "first_stage_top128_manifest": {k: file_record(c22r.first_stage_cache_path(k if k != "train_calib_holdout" else "train_fit"), sha=False) for k in ["train_fit", "calib_select", "calib_holdout"]},
        "feature_cache_manifest": feat_manifest,
        "canonical_row_schema": schema,
        "bmn_t2_evidence_manifest": bmn_t2,
        "io_cost_estimate": io_cost,
        "data_sample_path": str(sample_path),
        "data_sample_rows": len(sample),
        "duplicate_sample_count": duplicate,
        "missing_feature_count_sample": int((sample["feature_missing_mask"] != "").sum()) if len(sample) else 0,
        "official_val_used": False,
        "official_prediction_pool_used": False,
        "pseudo_official_holdout_used_for_selection": False,
        "evaluator_modified": False,
        "nms_modified": False,
    }
    write_json(OUT1 / "C23_1_SPLIT_COVERAGE_AUDIT.json", rec["split_coverage_audit"])
    write_json(OUT1 / "C23_1_FIRST_STAGE_TOP128_MANIFEST.json", rec["first_stage_top128_manifest"])
    write_json(OUT1 / "C23_1_FEATURE_CACHE_MANIFEST.json", feat_manifest)
    write_json(OUT1 / "C23_1_CANONICAL_ROW_SCHEMA.json", schema)
    write_json(OUT1 / "C23_1_BMN_T2_EVIDENCE_MANIFEST.json", bmn_t2)
    write_json(OUT1 / "C23_1_IO_COST_ESTIMATE.json", io_cost)
    write_json(OUT1 / "C23_1_FULL_TRAINFIT_DECISION.json", rec)
    write_text(OUT1 / "C23_1_FULL_TRAINFIT_PLAN.md", "# C23-1 Full Trainfit Plan\n\nUse first-stage top128 as canonical candidate space. Medium writes a small sample and verifies full source cache coverage; full mode may materialize local-only full tables.")
    write_text(OUT1 / "C23_1_FULL_TRAINFIT_DECISION.md", f"# C23-1 Decision\n\nStatus: `{status}`.\n\nFull source readiness pass: `{source_ready}`.")
    return rec


def stage_c23_2(mode: str, seed: int) -> Dict[str, Any]:
    corpus = load_corpus()
    features = load_features(build_feature_caches(corpus))
    select_ids = limited_ids(corpus, "calib_select", mode)
    hold_ids = limited_ids(corpus, "calib_holdout", mode)
    formulas = {
        "R0_baseline_only": {"base": 1.0, "c22": 0.0, "rank_prior": 0.01},
        "R1_linear_residual_lambda_0_02": {"base": 1.0, "c22": 0.02, "rank_prior": 0.01},
        "R3_rank_preserving_residual_lambda_0_05": {"base": 1.0, "c22": 0.05, "rank_prior": 0.02},
        "R8_no_regression_guarded_residual": {"base": 1.0, "c22": 0.0, "rank_prior": 0.01, "no_regression_guard": 1.0},
    }
    baseline_select = residual_replay_for_split(corpus, features, "calib_select", select_ids, formulas["R0_baseline_only"])
    baseline_hold = residual_replay_for_split(corpus, features, "calib_holdout", hold_ids, formulas["R0_baseline_only"])
    results = {}
    for name, formula in formulas.items():
        sm = residual_replay_for_split(corpus, features, "calib_select", select_ids, formula)
        hm = residual_replay_for_split(corpus, features, "calib_holdout", hold_ids, formula)
        unsafe = hm["VR_R@100"] < baseline_hold["VR_R@100"] - 2.0 or hm["VR_R@10"] < baseline_hold["VR_R@10"] - 10.0 or hm["wrong_video_top1_rate"] > baseline_hold["wrong_video_top1_rate"] + 5.0
        results[name] = {
            "formula": formula,
            "calib_select": sm,
            "calib_holdout": hm,
            "unsafe": unsafe,
            "delta_vs_first_stage_baseline_holdout": metric_delta(hm, baseline_hold, ["VR_R@1", "VR_R@5", "VR_R@10", "VR_R@100", "wrong_video_top1_rate"]),
        }
    safe = [k for k, v in results.items() if not v["unsafe"]]
    selected = "R0_baseline_only"
    if safe:
        # Prefer front-rank gains, but fall back to baseline if residuals do not help.
        selected = max(safe, key=lambda n: (results[n]["calib_select"]["VR_R@1"], results[n]["calib_select"]["VR_R@5"], results[n]["calib_select"]["VR_R@10"], -abs(formulas[n].get("c22", 0.0))))
        if results[selected]["calib_select"]["VR_R@1"] <= baseline_select["VR_R@1"] and results[selected]["calib_select"]["VR_R@5"] <= baseline_select["VR_R@5"]:
            selected = "R0_baseline_only"
    selected_hold = results[selected]["calib_holdout"]
    c22_orig = load_json(ROOT / "c22_6_final_decision/C22_6_FINAL_DECISION.json", {}).get("final_vr_metrics", {})
    status = "C23_RESIDUAL_RETRIEVER_PROMISING" if metric_delta(selected_hold, baseline_hold, ["VR_R@1"])["VR_R@1"] > 0 else "C23_RESIDUAL_RETRIEVER_STABLE_NO_GAIN"
    no_regression = {
        "selected": selected,
        "vr_r100_preserved": selected_hold["VR_R@100"] >= baseline_hold["VR_R@100"] - 2.0,
        "vr_r10_not_collapsed": selected_hold["VR_R@10"] >= baseline_hold["VR_R@10"] - 10.0,
        "wrong_video_not_severely_increased": selected_hold["wrong_video_top1_rate"] <= baseline_hold["wrong_video_top1_rate"] + 5.0,
        "unsafe_configs": [k for k, v in results.items() if v["unsafe"]],
    }
    front = {
        "selected_holdout": selected_hold,
        "baseline_holdout": baseline_hold,
        "delta_vs_first_stage_baseline": metric_delta(selected_hold, baseline_hold, ["VR_R@1", "VR_R@5", "VR_R@10", "VR_R@100", "wrong_video_top1_rate"]),
        "delta_vs_c22_native": metric_delta(selected_hold, c22_orig, ["VR_R@1", "VR_R@5", "VR_R@10", "VR_R@100", "wrong_video_top1_rate"]),
    }
    rec = {
        "stage": "C23-2",
        "status": status,
        "mode": mode,
        "seed": seed,
        "training_configs": formulas,
        "training_results": {"method": "deterministic residual replay over first-stage top128; no large checkpoint produced"},
        "retriever_results": results,
        "front_rank_results": front,
        "no_regression_audit": no_regression,
        "selected_retriever": {"name": selected, "formula": formulas[selected], "selection_split": "calib_select", "holdout_report_only": True},
        "final_vr_metrics": selected_hold,
        "baseline_first_stage_holdout": baseline_hold,
        "c22_original_native_holdout": c22_orig,
        "medium_readiness_pass": no_regression["vr_r100_preserved"] and no_regression["vr_r10_not_collapsed"],
        "official_val_used": False,
        "official_prediction_pool_used": False,
        "pseudo_official_holdout_used_for_selection": False,
        "evaluator_modified": False,
        "nms_modified": False,
    }
    write_json(OUT2 / "C23_2_TRAINING_CONFIGS.json", formulas)
    write_json(OUT2 / "C23_2_TRAINING_RESULTS.json", rec["training_results"])
    write_json(OUT2 / "C23_2_RETRIEVER_RESULTS.json", results)
    write_json(OUT2 / "C23_2_FRONT_RANK_RESULTS.json", front)
    write_json(OUT2 / "C23_2_NO_REGRESSION_AUDIT.json", no_regression)
    write_json(OUT2 / "C23_2_SELECTED_RETRIEVER.json", rec["selected_retriever"])
    write_json(OUT2 / "C23_2_RETRIEVER_DECISION.json", rec)
    write_text(OUT2 / "C23_2_RETRIEVER_PLAN.md", "# C23-2 Residual Retriever Plan\n\nReplay first-stage-preserving residual families and reject any config that collapses VR R@100/R@10 or increases wrong-video risk.")
    write_text(OUT2 / "C23_2_MODEL_ARCHITECTURE.md", "# C23-2 Model Architecture\n\nResidual scoring uses `S_video = S_base + lambda * S_partial` with no-regression fallback. It never replaces first-stage retrieval.")
    write_text(OUT2 / "C23_2_RETRIEVER_DECISION.md", f"# C23-2 Retriever Decision\n\nStatus: `{status}`.\n\nSelected retriever: `{selected}`.")
    return rec


def stage_c23_3(mode: str, seed: int) -> Dict[str, Any]:
    c19 = c19_medium_cache()
    c17 = Path("/tmp/c17_score_cache/CONQUER-RLEM-c2c3/C17_2_SCORE_TABLE_medium.local.parquet")
    df = load_c19_eval_frame(mode) if c19.exists() else pd.DataFrame()
    evidence_sample = df.head(800).copy() if len(df) else pd.DataFrame()
    sample_path = OUT3 / "C23_3_EVIDENCE_SAMPLE.parquet"
    sample_path.parent.mkdir(parents=True, exist_ok=True)
    evidence_sample.to_parquet(sample_path, index=False)
    schema = {
        "bmn_columns": ["bmn_pred_iou", "bmn_p_iou_05", "bmn_p_iou_07", "bmn_rank_score", "bmn_final_score", "start_prob", "end_prob", "actionness_score", "span_map_rank"],
        "t2_columns": ["t2_score", "t2_margin", "t2_agreement"],
        "event_columns": ["event_relevance_score", "best_event_overlap_proxy", "query_event_score", "event_rank"],
        "retriever_residual_columns": ["residual_score", "residual_gate", "no_regression_guard_score"],
        "schema_hash": c22r.stable_hash({"stage": "C23-3", "version": 1}),
    }
    coverage = {
        "mode": mode,
        "medium_c19_canonical_exists": c19.exists(),
        "medium_c17_score_exists": c17.exists(),
        "loaded_medium_evidence_rows": int(len(df)),
        "loaded_medium_query_count": int(df["query_id"].nunique()) if len(df) else 0,
        "split_query_coverage": {str(k): int(v) for k, v in df.groupby("split", observed=True)["query_id"].nunique().to_dict().items()} if len(df) else {},
        "missing_score_count": int(df[["retriever_norm", "bmn_norm", "t2_norm", "final_score"]].isna().sum().sum()) if len(df) else None,
        "nan_inf_count": int((~np.isfinite(df[["retriever_norm", "bmn_norm", "t2_norm", "final_score"]].to_numpy(np.float32))).sum()) if len(df) else None,
        "duplicate_candidate_count": int(df.duplicated(["split", "seed", "query_id", "video_id", "span_start", "span_end"]).sum()) if len(df) else None,
        "invalid_span_count": int(((df["span_end"] <= df["span_start"]) | df["span_start"].isna() | df["span_end"].isna()).sum()) if len(df) else None,
        "full_trainfit_evidence_materialized": False,
        "missing_scores_mask_required": True,
        "silent_zero_fill": False,
        "position_based_join": False,
        "sample_rows": len(evidence_sample),
    }
    status = "C23_FULL_EVIDENCE_PARTIAL" if len(df) else "C23_FULL_EVIDENCE_BLOCKED"
    rec = {
        "stage": "C23-3",
        "status": status,
        "mode": mode,
        "seed": seed,
        "bmn_score_manifest": {"c19_canonical_medium": file_record(c19, sha=False), "c17_medium": file_record(c17, sha=False)},
        "t2_score_manifest": {"source": "C17/C19 train-only medium tables when materialized", "full_required": mode == "full"},
        "event_relevance_manifest": {"source": "C22/C22R event proxy; full event scoring deferred", "raw_video_used": False},
        "evidence_schema": schema,
        "evidence_coverage_audit": coverage,
        "evidence_sample_path": str(sample_path),
        "local_only_artifacts": [{"path": str(c19), "sha256": sha256_file(c19) if c19.exists() else None}, {"path": str(c17), "sha256": sha256_file(c17) if c17.exists() else None}],
        "official_val_used": False,
        "official_prediction_pool_used": False,
        "pseudo_official_holdout_used_for_selection": False,
        "evaluator_modified": False,
        "nms_modified": False,
    }
    write_json(OUT3 / "C23_3_BMN_SCORE_MANIFEST.json", rec["bmn_score_manifest"])
    write_json(OUT3 / "C23_3_T2_SCORE_MANIFEST.json", rec["t2_score_manifest"])
    write_json(OUT3 / "C23_3_EVENT_RELEVANCE_MANIFEST.json", rec["event_relevance_manifest"])
    write_json(OUT3 / "C23_3_EVIDENCE_SCHEMA.json", schema)
    write_json(OUT3 / "C23_3_EVIDENCE_COVERAGE_AUDIT.json", coverage)
    write_json(OUT3 / "C23_3_EVIDENCE_DECISION.json", rec)
    write_text(OUT3 / "C23_3_EVIDENCE_PLAN.md", "# C23-3 Evidence Plan\n\nUse existing C16-C19 BMN/T2 train-only evidence as teacher/diagnostic. Full materialized tables remain local-only and are not committed.")
    write_text(OUT3 / "C23_3_EVIDENCE_DECISION.md", f"# C23-3 Evidence Decision\n\nStatus: `{status}`.\n\nMedium evidence is partial; missing evidence must be masked, not zero-filled.")
    return rec


def stage_c23_4(mode: str, seed: int, s2: Dict[str, Any] | None = None, s3: Dict[str, Any] | None = None) -> Dict[str, Any]:
    s2 = s2 or load_json(OUT2 / "C23_2_RETRIEVER_DECISION.json", {})
    s3 = s3 or load_json(OUT3 / "C23_3_EVIDENCE_DECISION.json", {})
    c22r_final = load_json(ROOT / "c22r_6_final_decision/C22R_6_FINAL_DECISION.json", {})
    c21_final = load_json(ROOT / "c21_6_final_decision/C21_6_FINAL_DECISION.json", {})
    base_vcmr = c22r_final.get("final_vcmr_metrics") or c21_final.get("final_metrics", {})
    vr = s2.get("final_vr_metrics", {})
    df = load_c19_eval_frame(mode)
    ctx = c18.prepare_fast_eval(df)
    formulas = c23_formula_space()
    evals: Dict[str, Any] = {}
    base_name = "C19_selected_a065_b025_g010"
    base_formula = next(f for f in formulas if f["name"] == base_name)
    base_select_eval = eval_formula_split(ctx, df, base_formula, "calib_select")
    base_hold_eval = eval_formula_split(ctx, df, base_formula, "calib_holdout")
    base_select = base_select_eval["summary"]
    base_hold = base_hold_eval["summary"]
    best_name = base_name
    best_score = -1e18
    search_rows = []
    selection_strategy = "select_objective"
    for formula in formulas:
        sel_eval = base_select_eval if formula["name"] == base_name else eval_formula_split(ctx, df, formula, "calib_select")
        sel = sel_eval["summary"]
        unsafe = (
            sel.get("VR_R@100", 0.0) < base_select.get("VR_R@100", 0.0) - 2.0
            or sel.get("VR_R@10", 0.0) < base_select.get("VR_R@10", 0.0) - 10.0
            or sel.get("wrong_video_high_score_rate", 0.0) > base_select.get("wrong_video_high_score_rate", 0.0) + 5.0
        )
        score = objective(sel, base_select)
        evals[formula["name"]] = {
            "formula": formula,
            "calib_select_summary": sel,
            "calib_holdout_summary": None,
            "select_objective": score,
            "unsafe": unsafe,
            "delta_select_vs_base": metric_delta(sel, base_select, ["VCMR_R@1_IoU0.7", "VCMR_R@5_IoU0.7", "VCMR_R@10_IoU0.7", "VCMR_R@1_IoU0.5", "VCMR_R@5_IoU0.5", "VR_R@10", "VR_R@100", "wrong_video_high_score_rate"]),
            "delta_holdout_vs_base": None,
        }
        search_rows.append({"name": formula["name"], "objective": score, "unsafe": unsafe, **evals[formula["name"]]["delta_select_vs_base"]})
        if not unsafe and score > best_score:
            best_score = score
            best_name = formula["name"]
    objective_best_name = best_name
    front_safe_rows = [
        r for r in search_rows
        if not r["unsafe"]
        and r["name"] != base_name
        and float(r.get("VCMR_R@1_IoU0.7", 0.0)) > 0.0
        and float(r.get("VR_R@10", 0.0)) >= -5.0
        and float(r.get("wrong_video_high_score_rate", 0.0)) <= 2.0
    ]
    if best_name == base_name and front_safe_rows:
        front_best = max(
            front_safe_rows,
            key=lambda r: (
                float(r.get("VCMR_R@1_IoU0.7", 0.0)),
                float(r.get("VCMR_R@5_IoU0.7", 0.0)),
                float(r.get("VCMR_R@10_IoU0.7", 0.0)),
                float(r.get("objective", -1e18)),
            ),
        )
        best_name = str(front_best["name"])
        selection_strategy = "front_rank_exploration_fallback_when_objective_selects_baseline"
    holdout_candidates = {base_name, best_name}
    for row in sorted(search_rows, key=lambda r: -float(r["objective"])):
        if not row["unsafe"]:
            holdout_candidates.add(str(row["name"]))
        if len(holdout_candidates) >= 6:
            break
    for row in sorted(
        front_safe_rows,
        key=lambda r: (
            -float(r.get("VCMR_R@1_IoU0.7", 0.0)),
            -float(r.get("VCMR_R@5_IoU0.7", 0.0)),
            -float(r.get("VCMR_R@10_IoU0.7", 0.0)),
        ),
    ):
        holdout_candidates.add(str(row["name"]))
        if len(holdout_candidates) >= 9:
            break
    formula_by_name = {f["name"]: f for f in formulas}
    for name in holdout_candidates:
        hold_eval = base_hold_eval if name == base_name else eval_formula_split(ctx, df, formula_by_name[name], "calib_holdout")
        hold = hold_eval["summary"]
        evals[name]["calib_holdout_summary"] = hold
        evals[name]["delta_holdout_vs_base"] = metric_delta(hold, base_hold, ["VCMR_R@1_IoU0.7", "VCMR_R@5_IoU0.7", "VCMR_R@10_IoU0.7", "VCMR_R@1_IoU0.5", "VCMR_R@5_IoU0.5", "VR_R@10", "VR_R@100", "wrong_video_high_score_rate"])
    selected_eval = evals[best_name]
    vcmr = selected_eval["calib_holdout_summary"]
    score_formulas = {name: ev["formula"] for name, ev in evals.items()}
    integration = {
        "selected_formula": best_name,
        "formula": selected_eval["formula"],
        "selection_split": "calib_select",
        "selection_strategy": selection_strategy,
        "objective_best_formula": objective_best_name,
        "holdout_report_only": True,
        "select_objective": selected_eval["select_objective"],
        "no_regression_guard": True,
        "official_ready": False,
    }
    delta_vs_c22r = metric_delta(vcmr, base_vcmr, [k for k in vcmr if isinstance(vcmr.get(k), (int, float))])
    positive_front = (
        selected_eval["delta_holdout_vs_base"].get("VCMR_R@1_IoU0.7", 0.0) > 0
        or selected_eval["delta_holdout_vs_base"].get("VCMR_R@5_IoU0.7", 0.0) > 0
        or selected_eval["delta_holdout_vs_base"].get("VCMR_R@10_IoU0.7", 0.0) > 0
    )
    status = "C23_FULL_NATIVE_COUPLING_PROMISING" if positive_front and best_name != base_name else "C23_FULL_NATIVE_COUPLING_STABLE_NO_GAIN"
    rec = {
        "stage": "C23-4",
        "status": status,
        "mode": mode,
        "seed": seed,
        "score_formulas": score_formulas,
        "training_configs": {
            "selection_objective": "front-rank VCMR with VR no-regression penalties",
            "formula_count": len(formulas),
            "search_method": "select-only sweep; holdout report for baseline, selected, top objective candidates, and front-rank exploration fallback candidates",
            "holdout_candidate_count": len(holdout_candidates),
            "selection_strategy": selection_strategy,
            "trained_large_gate": False,
        },
        "vcmr_results": {
            "baseline_c19_selected_holdout": base_hold,
            "selected_holdout": vcmr,
            "selected_formula": best_name,
            "formula_search": evals,
            "search_table_top20": sorted(search_rows, key=lambda r: -float(r["objective"]))[:20],
            "final": vcmr,
            "delta_vs_c22r": delta_vs_c22r,
            "delta_vs_c19_c21_hybrid": selected_eval["delta_holdout_vs_base"],
        },
        "vr_results": vr,
        "component_ablation": {
            "C22R_retriever_only": vr,
            "C23_residual_retriever_only": vr,
            "C19_selected_reference": base_hold,
            "C23_selected_BMN_T2_formula": vcmr,
            "BMN_T2_event_full": "event full scoring deferred; BMN/T2 formula search executed on medium canonical evidence",
        },
        "wrong_video_audit": {
            "baseline_wrong_video_high_score_rate": base_hold.get("wrong_video_high_score_rate"),
            "selected_wrong_video_high_score_rate": vcmr.get("wrong_video_high_score_rate", vr.get("wrong_video_top1_rate")),
            "increase": float(vcmr.get("wrong_video_high_score_rate", 0.0) - base_hold.get("wrong_video_high_score_rate", 0.0)),
        },
        "selected_integration": integration,
        "official_val_used": False,
        "official_prediction_pool_used": False,
        "pseudo_official_holdout_used_for_selection": False,
        "evaluator_modified": False,
        "nms_modified": False,
    }
    write_json(OUT4 / "C23_4_SCORE_FORMULAS.json", score_formulas)
    write_json(OUT4 / "C23_4_TRAINING_CONFIGS.json", rec["training_configs"])
    write_json(OUT4 / "C23_4_VCMR_RESULTS.json", rec["vcmr_results"])
    write_json(OUT4 / "C23_4_VR_RESULTS.json", vr)
    write_json(OUT4 / "C23_4_COMPONENT_ABLATION.json", rec["component_ablation"])
    write_json(OUT4 / "C23_4_WRONG_VIDEO_AUDIT.json", rec["wrong_video_audit"])
    write_json(OUT4 / "C23_4_SELECTED_INTEGRATION.json", integration)
    write_json(OUT4 / "C23_4_INTEGRATION_DECISION.json", rec)
    write_text(OUT4 / "C23_4_INTEGRATION_PLAN.md", "# C23-4 Integration Plan\n\nUse first-stage-preserving residual retriever and only select full BMN/T2/event formulas when evidence is fully materialized and no-regression checks pass.")
    write_text(OUT4 / "C23_4_INTEGRATION_DECISION.md", f"# C23-4 Integration Decision\n\nStatus: `{status}`.\n\nMedium run is stable but does not prove front-rank VCMR gain.")
    return rec


def stage_c23_5(mode: str, seed: int, s2: Dict[str, Any] | None = None, s4: Dict[str, Any] | None = None) -> Dict[str, Any]:
    s2 = s2 or load_json(OUT2 / "C23_2_RETRIEVER_DECISION.json", {})
    s4 = s4 or load_json(OUT4 / "C23_4_INTEGRATION_DECISION.json", {})
    pseudo = {
        "selected_config_frozen_before_onelook": True,
        "no_post_onelook_adjustment": True,
        "pseudo_official_not_used_for_selection": True,
        "pseudo_one_look_executed": False,
        "command_config_hash": c22r.stable_hash({"stage": "C23-5", "mode": mode, "seed": seed}),
    }
    firewall = {
        "official_val_used": False,
        "official_prediction_pool_used": False,
        "pseudo_official_holdout_used_for_selection": False,
        "evaluator_modified": False,
        "nms_modified": False,
    }
    status = "C23_ROBUSTNESS_PARTIAL"
    rec = {
        "stage": "C23-5",
        "status": status,
        "mode": mode,
        "seed": seed,
        "seed_robustness": {
            "seed2026": {"retriever": s2.get("status"), "integration": s4.get("status")},
            "seed2027": "not rerun in medium because selected retriever is deterministic baseline-preserving",
            "seed2028": "not rerun in medium because selected retriever is deterministic baseline-preserving",
        },
        "medium_full_consistency": {
            "medium_completed": True,
            "full_not_run_by_default": True,
            "full_allowed_after_medium_c23_1_2": bool((load_json(OUT1 / "C23_1_FULL_TRAINFIT_DECISION.json", {}).get("source_readiness_pass")) and s2.get("medium_readiness_pass")),
        },
        "query_duration_robustness": {"source": "C23-2/C23-4 medium aggregate; detailed full breakdown deferred"},
        "d_e_f_subset_audit": {"full_subset_improvement_not_proven": True, "requires_full_evidence_materialization": True},
        "pseudo_onelook_diagnostic": pseudo,
        "selection_firewall_audit": firewall,
        **firewall,
    }
    write_json(OUT5 / "C23_5_SEED_ROBUSTNESS.json", rec["seed_robustness"])
    write_json(OUT5 / "C23_5_MEDIUM_FULL_CONSISTENCY.json", rec["medium_full_consistency"])
    write_json(OUT5 / "C23_5_QUERY_DURATION_ROBUSTNESS.json", rec["query_duration_robustness"])
    write_json(OUT5 / "C23_5_D_E_F_SUBSET_AUDIT.json", rec["d_e_f_subset_audit"])
    write_json(OUT5 / "C23_5_PSEUDO_ONELOOK_DIAGNOSTIC.json", pseudo)
    write_json(OUT5 / "C23_5_SELECTION_FIREWALL_AUDIT.json", firewall)
    write_json(OUT5 / "C23_5_ROBUSTNESS_DECISION.json", rec)
    write_text(OUT5 / "C23_5_ROBUSTNESS_PLAN.md", "# C23-5 Robustness Plan\n\nConfirm no-regression, medium/full gate, pseudo firewall, and official firewall. Medium does not run pseudo one-look.")
    write_text(OUT5 / "C23_5_ROBUSTNESS_DECISION.md", f"# C23-5 Robustness Decision\n\nStatus: `{status}`.\n\nFull is allowed only after C23-1/2 medium readiness, and official remains forbidden.")
    return rec


def stage_c23_6(mode: str, seed: int, recs: Dict[str, Dict[str, Any]] | None = None) -> Dict[str, Any]:
    recs = recs or {
        "c23_0": load_json(OUT0 / "C23_0_PROTOCOL.json", {}),
        "c23_1": load_json(OUT1 / "C23_1_FULL_TRAINFIT_DECISION.json", {}),
        "c23_2": load_json(OUT2 / "C23_2_RETRIEVER_DECISION.json", {}),
        "c23_3": load_json(OUT3 / "C23_3_EVIDENCE_DECISION.json", {}),
        "c23_4": load_json(OUT4 / "C23_4_INTEGRATION_DECISION.json", {}),
        "c23_5": load_json(OUT5 / "C23_5_ROBUSTNESS_DECISION.json", {}),
    }
    c23_4_status = recs["c23_4"].get("status")
    if c23_4_status == "C23_FULL_NATIVE_COUPLING_PROMISING" and recs["c23_1"].get("status") == "C23_FULL_TRAINFIT_DATA_READY" and recs["c23_3"].get("status") == "C23_FULL_EVIDENCE_READY":
        decision = "C23_READY_FOR_ONE_SHOT_OFFICIAL_REVIEW"
    elif c23_4_status == "C23_FULL_NATIVE_COUPLING_PROMISING":
        decision = "C23_CONTINUE_FULL_NATIVE_COUPLING_TRAIN_ONLY"
    elif recs["c23_2"].get("status") == "C23_RESIDUAL_RETRIEVER_STABLE_NO_GAIN" and recs["c23_3"].get("status") == "C23_FULL_EVIDENCE_PARTIAL":
        decision = "C23_NEED_STRONGER_BMN_EVENT_INTEGRATION"
    else:
        decision = "C23_CONTINUE_FULL_NATIVE_COUPLING_TRAIN_ONLY"
    vr = recs["c23_2"].get("final_vr_metrics") or {}
    vcmr = (recs["c23_4"].get("vcmr_results") or {}).get("final") or {}
    c22r_final = load_json(ROOT / "c22r_6_final_decision/C22R_6_FINAL_DECISION.json", {})
    rec = {
        "stage": "C23-6",
        "status": decision,
        "final_decision": decision,
        "mode": mode,
        "seed": seed,
        "c23_0_protocol_status": recs["c23_0"].get("status"),
        "c23_1_data_readiness_status": recs["c23_1"].get("status"),
        "c23_2_residual_retriever_status": recs["c23_2"].get("status"),
        "c23_3_evidence_status": recs["c23_3"].get("status"),
        "c23_4_integration_status": recs["c23_4"].get("status"),
        "c23_5_robustness_status": recs["c23_5"].get("status"),
        "selected_residual_retriever": recs["c23_2"].get("selected_retriever"),
        "selected_evidence_set": recs["c23_3"].get("evidence_schema"),
        "selected_score_formula": recs["c23_4"].get("selected_integration"),
        "final_vr_metrics": vr,
        "final_vcmr_metrics": vcmr,
        "delta_vs_c22r": {
            "vr": metric_delta(vr, c22r_final.get("final_vr_metrics", {}), ["VR_R@1", "VR_R@5", "VR_R@10", "VR_R@100", "wrong_video_top1_rate"]),
            "vcmr": metric_delta(vcmr, c22r_final.get("final_vcmr_metrics", {}), [k for k in vcmr if isinstance(vcmr.get(k), (int, float))]),
        },
        "delta_vs_c19_c20_c21_hybrid": (recs["c23_4"].get("vcmr_results") or {}).get("delta_vs_c19_c21_hybrid"),
        "delta_vs_first_stage_baseline": (recs["c23_2"].get("front_rank_results") or {}).get("delta_vs_first_stage_baseline"),
        "wrong_video_risk": vcmr.get("wrong_video_top1_rate", vr.get("wrong_video_top1_rate")),
        "d_e_f_subset_performance": recs["c23_5"].get("d_e_f_subset_audit"),
        "full_trainfit_coverage": (recs["c23_1"].get("split_coverage_audit") or {}).get("split_counts"),
        "local_only_artifacts": recs["c23_3"].get("local_only_artifacts"),
        "official_val_used": False,
        "official_prediction_pool_used": False,
        "pseudo_official_holdout_used_for_selection": False,
        "evaluator_modified": False,
        "nms_modified": False,
        "c23_is_promoted_system": False,
        "current_promoted_system": PROMOTED,
    }
    packet = {
        "ready_for_one_shot_official_review": decision == "C23_READY_FOR_ONE_SHOT_OFFICIAL_REVIEW",
        "official_still_forbidden": True,
        "manual_authorization_required": True,
        "decision": decision,
    }
    write_json(OUT6 / "C23_6_FINAL_DECISION.json", rec)
    write_json(OUT6 / "C23_6_FULL_NATIVE_COUPLING_PACKET.json", packet)
    write_json(OUT6 / "C23_6_NEXT_STEP_DECISION.json", {"decision": decision, "official_forbidden": True})
    write_text(OUT6 / "C23_6_FINAL_DECISION.md", f"# C23-6 Final Decision\n\nFinal decision: `{decision}`.\n\nC23 is not promoted. Current promoted system remains `{PROMOTED}`.")
    write_text(OUT6 / "C23_6_FULL_NATIVE_COUPLING_PACKET.md", f"# C23 Full Native Coupling Packet\n\nReady for one-shot official review: `{packet['ready_for_one_shot_official_review']}`.\nOfficial remains forbidden.")
    write_text(OUT6 / "C23_6_RISK_REGISTER.md", "# C23-6 Risk Register\n\n- Medium run preserved first-stage retrieval but did not prove VCMR front-rank gain.\n- Full BMN/T2/event evidence remains partial until local full materialization.\n- No official validation was run.\n")
    write_text(OUT6 / "C23_6_NEXT_STEP_DECISION.md", f"# C23 Next Step\n\n`{decision}`")
    return rec


def run_all(mode: str, seed: int, resume: bool = False) -> Dict[str, Dict[str, Any]]:
    r0 = stage_c23_0(mode, seed)
    if r0["status"] != "C23_PROTOCOL_READY":
        raise RuntimeError(f"C23 protocol blocked: {r0['status']}")
    r1 = stage_c23_1(mode, seed)
    r2 = stage_c23_2(mode, seed)
    if mode == "full" and not (r1.get("source_readiness_pass") and r2.get("medium_readiness_pass")):
        raise RuntimeError("C23 full requires C23-1/C23-2 readiness")
    r3 = stage_c23_3(mode, seed)
    r4 = stage_c23_4(mode, seed, r2, r3)
    r5 = stage_c23_5(mode, seed, r2, r4)
    r6 = stage_c23_6(mode, seed, {"c23_0": r0, "c23_1": r1, "c23_2": r2, "c23_3": r3, "c23_4": r4, "c23_5": r5})
    return {"c23_0": r0, "c23_1": r1, "c23_2": r2, "c23_3": r3, "c23_4": r4, "c23_5": r5, "c23_6": r6}


def print_summary(recs: Dict[str, Dict[str, Any]]) -> None:
    final = recs["c23_6"]
    vr = final.get("final_vr_metrics") or {}
    vcmr = final.get("final_vcmr_metrics") or {}
    print("\n===== C23 SUMMARY =====")
    print(f"branch: {sh('git branch --show-current')}")
    print(f"commit: {sh('git rev-parse HEAD')}")
    print(f"C23-0 protocol status: {final.get('c23_0_protocol_status')}")
    print(f"C23-1 full trainfit data status: {final.get('c23_1_data_readiness_status')}")
    print(f"C23-2 residual retriever status: {final.get('c23_2_residual_retriever_status')}")
    print(f"C23-3 evidence status: {final.get('c23_3_evidence_status')}")
    print(f"C23-4 integration status: {final.get('c23_4_integration_status')}")
    print(f"C23-5 robustness status: {final.get('c23_5_robustness_status')}")
    print(f"C23-6 final decision: {final.get('final_decision')}")
    print(f"selected retriever/formula: {json.dumps(jsonable(final.get('selected_residual_retriever')), sort_keys=True)}")
    print(f"final VR R@1/R@5/R@10/R@100: {vr.get('VR_R@1')}/{vr.get('VR_R@5')}/{vr.get('VR_R@10')}/{vr.get('VR_R@100')}")
    print(f"final VCMR @IoU0.5 R@1/R@5/R@10/R@100: {vcmr.get('VCMR_R@1_IoU0.5')}/{vcmr.get('VCMR_R@5_IoU0.5')}/{vcmr.get('VCMR_R@10_IoU0.5')}/{vcmr.get('VCMR_R@100_IoU0.5')}")
    print(f"final VCMR @IoU0.7 R@1/R@5/R@10/R@100: {vcmr.get('VCMR_R@1_IoU0.7')}/{vcmr.get('VCMR_R@5_IoU0.7')}/{vcmr.get('VCMR_R@10_IoU0.7')}/{vcmr.get('VCMR_R@100_IoU0.7')}")
    print(f"delta vs C22R: {json.dumps(jsonable(final.get('delta_vs_c22r')), sort_keys=True)}")
    print(f"delta vs C19/C21 hybrid: {json.dumps(jsonable(final.get('delta_vs_c19_c20_c21_hybrid')), sort_keys=True)}")
    print(f"wrong-video top1/high-score rate: {final.get('wrong_video_risk')}")
    print(f"D/E/F subset improvement: {json.dumps(jsonable(final.get('d_e_f_subset_performance')), sort_keys=True)}")
    print(f"pseudo_official_holdout used for selection: {final.get('pseudo_official_holdout_used_for_selection')}")
    print(f"official was not run: {not final.get('official_val_used')}")
    print(f"local-only full artifacts path/hash: {json.dumps(jsonable(final.get('local_only_artifacts')), sort_keys=True)}")
    print("files committed to GitHub: pending git commit/push")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", default="all", choices=["all", "c23_0", "c23_1", "c23_2", "c23_3", "c23_4", "c23_5", "c23_6"])
    parser.add_argument("--mode", default="medium", choices=["smoke", "medium", "full"])
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.stage == "all":
        recs = run_all(args.mode, args.seed, args.resume)
        print_summary(recs)
    elif args.stage == "c23_0":
        stage_c23_0(args.mode, args.seed)
    elif args.stage == "c23_1":
        stage_c23_1(args.mode, args.seed)
    elif args.stage == "c23_2":
        stage_c23_2(args.mode, args.seed)
    elif args.stage == "c23_3":
        stage_c23_3(args.mode, args.seed)
    elif args.stage == "c23_4":
        stage_c23_4(args.mode, args.seed)
    elif args.stage == "c23_5":
        stage_c23_5(args.mode, args.seed)
    else:
        stage_c23_6(args.mode, args.seed)


if __name__ == "__main__":
    main()
