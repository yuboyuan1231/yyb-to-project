#!/usr/bin/env python3
"""C18 full hybrid freeze candidate.

Train-only readiness, baseline replay, front-rank calibration, pseudo one-look
diagnostic, and freeze review over the C17 native VCMR hybrid artifacts.
No official validation is run.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import random
import shutil
import subprocess
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

import run_c17_bmn_t2_native_vcmr_integration as c17
from run_c12_native_retriever_training import ROOT, DEVICE, sha256_file


torch.set_num_threads(min(24, os.cpu_count() or 1))
warnings.filterwarnings("ignore", category=FutureWarning, module="run_c17_bmn_t2_native_vcmr_integration")

PROMOTED = "C7-B6 R1SelectiveTop1"
OUT0 = ROOT / "c18_0_protocol_freeze"
OUT1 = ROOT / "c18_1_full_score_readiness"
OUT2 = ROOT / "c18_2_baseline_comparable_replay"
OUT3 = ROOT / "c18_3_front_rank_hybrid_calibration"
OUT4 = ROOT / "c18_4_pseudo_official_onelook"
OUT5 = ROOT / "c18_5_full_consistency_robustness"
OUT6 = ROOT / "c18_6_freeze_review"

BEST_C17 = {"name": "G_all_a0.55_b0.25_g0.2", "family": "G_retriever_BMN_T2", "alpha": 0.55, "beta": 0.25, "gamma": 0.20}


def jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [jsonable(v) for v in obj]
    if isinstance(obj, tuple):
        return [jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
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


def stable_hash(obj: Any) -> str:
    return hashlib.sha256(json.dumps(jsonable(obj), sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def mode_cache(mode: str) -> Path:
    return c17.local_score_cache_path(mode)


def c18_pseudo_cache(mode: str) -> Path:
    base = Path(os.environ.get("C18_SCORE_CACHE_DIR", "/tmp/c18_score_cache")) / "CONQUER-RLEM-c2c3"
    base.mkdir(parents=True, exist_ok=True)
    return base / f"C18_PSEUDO_SCORE_TABLE_{mode}.local.parquet"


def require_ready(path: Path, ok: Sequence[str]) -> Dict[str, Any]:
    obj = load_json(path, {})
    if obj.get("status") not in set(ok) and obj.get("final_decision") not in set(ok):
        raise RuntimeError(f"{path} has status {obj.get('status') or obj.get('final_decision')}, expected one of {ok}")
    return obj


def score_cache_audit(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {"exists": False, "path": str(path)}
    cols = [
        "split", "query_id", "seed", "video_id", "span_start", "span_end",
        "retriever_score", "bmn_final_score", "t2_score", "old_c12_score",
        "query_type", "duration_bucket", "schema_hash", "config_hash",
    ]
    df = pd.read_parquet(path, columns=cols)
    numeric = df.select_dtypes(include=[np.number])
    split_cov = df.groupby("split", observed=True)["query_id"].nunique().to_dict()
    audit = {
        "exists": True,
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "row_count": int(len(df)),
        "query_count": int(df["query_id"].nunique()),
        "video_count": int(df["video_id"].nunique()),
        "candidate_count": int(len(df)),
        "split_coverage": split_cov,
        "score_columns": ["retriever_score", "bmn_final_score", "t2_score", "old_c12_score"],
        "schema_hashes": sorted([str(x) for x in df["schema_hash"].dropna().unique().tolist()])[:8],
        "config_hashes": sorted([str(x) for x in df["config_hash"].dropna().unique().tolist()])[:8],
        "missing_bmn_score_count": int(df["bmn_final_score"].isna().sum()),
        "missing_t2_score_count": int(df["t2_score"].isna().sum()),
        "missing_retriever_score_count": int(df["retriever_score"].isna().sum()),
        "duplicate_candidate_count": int(df.duplicated(["split", "seed", "query_id", "video_id", "span_start", "span_end"]).sum()),
        "invalid_span_count": int(((df["span_end"] <= df["span_start"]) | df["span_start"].isna() | df["span_end"].isna()).sum()),
        "nan_inf_count": int(np.isinf(numeric.to_numpy()).sum()) if len(numeric.columns) else 0,
        "score_distribution_by_split": df.groupby("split", observed=True)[["retriever_score", "bmn_final_score", "t2_score"]].describe().to_dict(),
        "score_distribution_by_query_type": df.groupby("query_type", observed=True)[["retriever_score", "bmn_final_score", "t2_score"]].mean(numeric_only=True).to_dict(),
        "score_distribution_by_duration_bucket": df.groupby("duration_bucket", observed=True)[["retriever_score", "bmn_final_score", "t2_score"]].mean(numeric_only=True).to_dict(),
    }
    if {"calib_select", "calib_holdout"}.issubset(set(df["split"].astype(str).unique())):
        audit["calib_select_vs_calib_holdout_distribution_shift"] = {
            "bmn_mean_delta": float(df[df["split"].astype(str) == "calib_holdout"]["bmn_final_score"].mean() - df[df["split"].astype(str) == "calib_select"]["bmn_final_score"].mean()),
            "t2_mean_delta": float(df[df["split"].astype(str) == "calib_holdout"]["t2_score"].mean() - df[df["split"].astype(str) == "calib_select"]["t2_score"].mean()),
        }
    return audit


def load_eval_table_for_c18(mode: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    df, audit = c17.load_eval_score_table(mode)
    fill = c17.add_normalized_scores_inplace(df)
    audit["fill_audit"] = fill
    return df, audit


def evaluate_formula_on_splits(df: pd.DataFrame, formula: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {"formula": formula}
    c17.assign_formula_score(df, formula, "eval_score")
    for split in ["calib_select", "calib_holdout", "pseudo_official_holdout"]:
        sdf = df[df["split"] == split]
        if len(sdf):
            out[split] = c17.evaluate_vcmr(sdf, "eval_score")
    return out


def formula_score_array(df: pd.DataFrame, formula: Dict[str, Any]) -> np.ndarray:
    a = float(formula.get("alpha", 0.0))
    b = float(formula.get("beta", 0.0))
    g = float(formula.get("gamma", 0.0))
    d = float(formula.get("delta", 0.0))
    e = float(formula.get("eta", 0.0))
    retr = df["retriever_norm"].to_numpy(np.float32)
    bmn = df["bmn_norm"].to_numpy(np.float32)
    t2 = df["t2_norm"].to_numpy(np.float32)
    old = df["old_norm"].to_numpy(np.float32)
    dur = df["duration_score"].fillna(0.0).to_numpy(np.float32)
    score = a * retr + b * bmn + g * t2 + d * old + e * dur
    family = formula.get("family")
    if family == "H_qtype_gated":
        qtype = df["query_type"].astype(str).to_numpy()
        score = score + np.where(qtype == "t", 0.15 * t2, 0.10 * bmn).astype(np.float32)
    elif family == "I_duration_aware":
        bucket = df["duration_bucket"].astype(str).to_numpy()
        score = score + np.where(bucket == "short", 0.18 * bmn, 0.06 * t2).astype(np.float32)
    elif family == "J_safety_gated":
        conflict = (t2 > 0.85) & (bmn < 0.25)
        score = np.where(conflict, 0.70 * t2 + 0.30 * retr, score).astype(np.float32)
    return np.asarray(score, dtype=np.float32)


def prepare_fast_eval(df: pd.DataFrame) -> Dict[str, Any]:
    video_cats = pd.Index(pd.concat([df["video_id"], df["gt_video_id"]], ignore_index=True).astype(str).unique())
    video_code = pd.Categorical(df["video_id"].astype(str), categories=video_cats).codes.astype(np.int32)
    gt_video_code = pd.Categorical(df["gt_video_id"].astype(str), categories=video_cats).codes.astype(np.int32)
    qtype = df["query_type"].astype(str).to_numpy()
    dur_bucket = df["duration_bucket"].astype(str).to_numpy()
    split = df["split"].astype(str).to_numpy()
    seed = df["seed"].to_numpy(np.int32)
    qid = df["query_id"].to_numpy(np.int32)
    keys = pd.MultiIndex.from_arrays([split, seed, qid])
    grouped = pd.Series(np.arange(len(df), dtype=np.int32), index=keys).groupby(level=[0, 1, 2], sort=False)
    groups = [(key, grouped.obj.iloc[pos].to_numpy(dtype=np.int32)) for key, pos in grouped.indices.items()]
    dup = int(df.duplicated(["split", "seed", "query_id", "video_id", "span_start", "span_end"]).sum())
    invalid = int(((df["span_end"] <= df["span_start"]) | df["span_start"].isna() | df["span_end"].isna()).sum())
    return {
        "groups": groups,
        "video_code": video_code,
        "gt_video_code": gt_video_code,
        "span_start": df["span_start"].to_numpy(np.float32),
        "span_end": df["span_end"].to_numpy(np.float32),
        "gt_start": df["gt_start"].to_numpy(np.float32),
        "gt_end": df["gt_end"].to_numpy(np.float32),
        "qtype": qtype,
        "duration_bucket": dur_bucket,
        "gt_video_rank": df["gt_video_rank"].to_numpy(np.int32),
        "duplicate_count": dup,
        "invalid_count": invalid,
    }


def fast_iou_arrays(st: np.ndarray, ed: np.ndarray, gt_s: float, gt_e: float) -> np.ndarray:
    inter = np.maximum(0.0, np.minimum(ed, gt_e) - np.maximum(st, gt_s))
    union = np.maximum(ed, gt_e) - np.minimum(st, gt_s)
    return inter / np.maximum(union, 1e-6)


def fast_eval_vcmr(ctx: Dict[str, Any], score: np.ndarray, split_filter: str | None = None) -> Dict[str, Any]:
    records: List[Dict[str, Any]] = []
    vcode = ctx["video_code"]
    gt_vcode = ctx["gt_video_code"]
    st_all = ctx["span_start"]
    ed_all = ctx["span_end"]
    gt_s_all = ctx["gt_start"]
    gt_e_all = ctx["gt_end"]
    qtype_all = ctx["qtype"]
    dur_all = ctx["duration_bucket"]
    gt_rank_all = ctx["gt_video_rank"]
    for (split, seed, qid), idx_obj in ctx["groups"]:
        if split_filter is not None and split != split_filter:
            continue
        idx = np.asarray(list(idx_obj), dtype=np.int32)
        if idx.size == 0:
            continue
        order = idx[np.argsort(-score[idx], kind="mergesort")]
        kept: List[int] = []
        by_video: Dict[int, List[Tuple[float, float]]] = {}
        for ridx in order:
            vid = int(vcode[ridx])
            span = (float(st_all[ridx]), float(ed_all[ridx]))
            prev = by_video.get(vid)
            if prev:
                overlaps = False
                for p in prev:
                    inter = max(0.0, min(span[1], p[1]) - max(span[0], p[0]))
                    union = max(span[1], p[1]) - min(span[0], p[0])
                    if union > 0 and inter / union > c17.NMS_THRESHOLD:
                        overlaps = True
                        break
                if overlaps:
                    continue
            by_video.setdefault(vid, []).append(span)
            kept.append(int(ridx))
            if len(kept) >= 200:
                break
        if not kept:
            continue
        kept_arr = np.asarray(kept, dtype=np.int32)
        first = kept_arr[0]
        same = vcode[kept_arr] == gt_vcode[first]
        ious = np.zeros((len(kept_arr),), dtype=np.float32)
        if same.any():
            ious[same] = fast_iou_arrays(st_all[kept_arr[same]], ed_all[kept_arr[same]], float(gt_s_all[first]), float(gt_e_all[first]))
        videos = vcode[kept_arr].tolist()
        unique_videos = []
        for v in videos:
            if v not in unique_videos:
                unique_videos.append(v)
        gt_only_ious = ious[same]
        rec: Dict[str, Any] = {
            "split": split,
            "seed": int(seed),
            "query_id": int(qid),
            "query_type": str(qtype_all[first]),
            "duration_bucket": str(dur_all[first]),
            "gt_video_rank": int(gt_rank_all[first]),
            "top1_video_correct": bool(vcode[first] == gt_vcode[first]),
            "top1_iou": float(ious[0]),
            "wrong_video_top1": bool(vcode[first] != gt_vcode[first]),
            "correct_video_wrong_span_top1": bool(vcode[first] == gt_vcode[first] and ious[0] < 0.5),
        }
        for k in [1, 5, 10, 100]:
            rec[f"VCMR_R@{k}_IoU0.5"] = bool(np.any(ious[: min(k, len(ious))] >= 0.5))
            rec[f"VCMR_R@{k}_IoU0.7"] = bool(np.any(ious[: min(k, len(ious))] >= 0.7))
            rec[f"VR_R@{k}"] = int(gt_vcode[first]) in unique_videos[: min(k, len(unique_videos))]
        for k in [50, 100]:
            rec[f"GT_VIDEO_TOP{k}_IoU0.5"] = bool(np.any(gt_only_ious[: min(k, len(gt_only_ious))] >= 0.5))
            rec[f"GT_VIDEO_TOP{k}_IoU0.7"] = bool(np.any(gt_only_ious[: min(k, len(gt_only_ious))] >= 0.7))
        records.append(rec)

    def aggregate(rs: List[Dict[str, Any]]) -> Dict[str, Any]:
        out: Dict[str, Any] = {"query_count": len(rs)}
        for k in [1, 5, 10, 100]:
            out[f"VCMR_R@{k}_IoU0.5"] = c17.metric([r[f"VCMR_R@{k}_IoU0.5"] for r in rs])
            out[f"VCMR_R@{k}_IoU0.7"] = c17.metric([r[f"VCMR_R@{k}_IoU0.7"] for r in rs])
            out[f"VR_R@{k}"] = c17.metric([r[f"VR_R@{k}"] for r in rs])
        for k in [50, 100]:
            out[f"GT_VIDEO_TOP{k}_IoU0.5"] = c17.metric([r[f"GT_VIDEO_TOP{k}_IoU0.5"] for r in rs])
            out[f"GT_VIDEO_TOP{k}_IoU0.7"] = c17.metric([r[f"GT_VIDEO_TOP{k}_IoU0.7"] for r in rs])
        out["wrong_video_high_score_rate"] = c17.metric([r["wrong_video_top1"] for r in rs])
        out["correct_video_wrong_span_rate"] = c17.metric([r["correct_video_wrong_span_top1"] for r in rs])
        out["top1_mean_iou"] = c17.mean([float(r["top1_iou"]) for r in rs])
        out["duplicate_span_count"] = ctx["duplicate_count"]
        out["invalid_span_count"] = ctx["invalid_count"]
        return out

    if records:
        frame = pd.DataFrame(records)
        qtype_breakdown = {str(k): aggregate(v.to_dict("records")) for k, v in frame.groupby("query_type", observed=True)}
        dur_breakdown = {str(k): aggregate(v.to_dict("records")) for k, v in frame.groupby("duration_bucket", observed=True)}
        def rb(x: int) -> str:
            if x <= 1:
                return "rank1"
            if x <= 5:
                return "rank2_5"
            if x <= 10:
                return "rank6_10"
            if x <= 100:
                return "rank11_100"
            return "rank_gt100"
        frame["video_rank_bucket"] = [rb(int(x)) for x in frame["gt_video_rank"]]
        rank_breakdown = {str(k): aggregate(v.to_dict("records")) for k, v in frame.groupby("video_rank_bucket", observed=True)}
    else:
        qtype_breakdown, dur_breakdown, rank_breakdown = {}, {}, {}
    return {
        "summary": aggregate(records),
        "query_type_breakdown": qtype_breakdown,
        "duration_breakdown": dur_breakdown,
        "video_rank_breakdown": rank_breakdown,
        "records_sample": records[:100],
    }


def fast_evaluate_formula(ctx: Dict[str, Any], df: pd.DataFrame, formula: Dict[str, Any]) -> Dict[str, Any]:
    score = formula_score_array(df, formula)
    return {
        "formula": formula,
        "calib_select": fast_eval_vcmr(ctx, score, "calib_select"),
        "calib_holdout": fast_eval_vcmr(ctx, score, "calib_holdout"),
        "pseudo_official_holdout": fast_eval_vcmr(ctx, score, "pseudo_official_holdout") if "pseudo_official_holdout" in set(df["split"].astype(str).unique()) else None,
    }


def front_rank_score(metrics: Dict[str, Any], base: Dict[str, Any]) -> float:
    s = metrics["summary"]
    b = base["summary"]
    wrong_increase = max(0.0, float(s.get("wrong_video_high_score_rate", 0.0) - b.get("wrong_video_high_score_rate", 0.0)))
    return (
        2.0 * (s.get("VCMR_R@1_IoU0.7", 0.0) - b.get("VCMR_R@1_IoU0.7", 0.0))
        + 1.5 * (s.get("VCMR_R@5_IoU0.7", 0.0) - b.get("VCMR_R@5_IoU0.7", 0.0))
        + 1.2 * (s.get("VCMR_R@10_IoU0.7", 0.0) - b.get("VCMR_R@10_IoU0.7", 0.0))
        + 0.8 * (s.get("VCMR_R@100_IoU0.7", 0.0) - b.get("VCMR_R@100_IoU0.7", 0.0))
        + 1.0 * (s.get("VCMR_R@1_IoU0.5", 0.0) - b.get("VCMR_R@1_IoU0.5", 0.0))
        + 0.8 * (s.get("VCMR_R@5_IoU0.5", 0.0) - b.get("VCMR_R@5_IoU0.5", 0.0))
        - wrong_increase
    )


def c18_search_space() -> List[Dict[str, Any]]:
    formulas: List[Dict[str, Any]] = [dict(BEST_C17)]
    for a in [0.45, 0.50, 0.55, 0.60, 0.65]:
        for b in [0.20, 0.25, 0.30, 0.35]:
            for g in [0.10, 0.15, 0.20, 0.25]:
                z = max(a + b + g, 1e-8)
                formulas.append({"name": f"grid_a{a}_b{b}_g{g}", "family": "G_retriever_BMN_T2", "alpha": round(a / z, 6), "beta": round(b / z, 6), "gamma": round(g / z, 6)})
    formulas.extend([
        {"name": "front_safety_gate", "family": "J_safety_gated", "alpha": 0.55, "beta": 0.25, "gamma": 0.20},
        {"name": "duration_aware_front", "family": "I_duration_aware", "alpha": 0.55, "beta": 0.25, "gamma": 0.20},
        {"name": "qtype_gated_front", "family": "H_qtype_gated", "alpha": 0.55, "beta": 0.25, "gamma": 0.20},
        {"name": "log_product_bmn_t2", "family": "G_retriever_BMN_T2", "alpha": 0.50, "beta": 0.30, "gamma": 0.20},
    ])
    seen = {}
    for f in formulas:
        seen[stable_hash(f)] = f
    return list(seen.values())


def stage_c18_0(mode: str, seed: int) -> Dict[str, Any]:
    OUT0.mkdir(parents=True, exist_ok=True)
    branch = sh("git branch --show-current")
    commit = sh("git rev-parse HEAD")
    dirty = sh("git status --short")
    deps = {
        "c17_evaluator": ROOT / "c17_1_train_only_vcmr_evaluator/C17_1_EVALUATOR_DECISION.json",
        "c17_score": ROOT / "c17_2_full_bmn_score_export/C17_2_FULL_BMN_DECISION.json",
        "c17_hybrid": ROOT / "c17_3_bmn_t2_hybrid_calibration/C17_3_HYBRID_DECISION.json",
        "c17_integration": ROOT / "c17_4_native_vcmr_full_integration/C17_4_INTEGRATION_DECISION.json",
        "c17_robustness": ROOT / "c17_5_robustness_and_consistency/C17_5_ROBUSTNESS_DECISION.json",
        "c17_next": ROOT / "c17_6_freeze_review/C17_6_NEXT_STEP_DECISION.json",
        "c12_split_manifest": ROOT / "c12_1_schema_and_split/C12_1_SPLIT_MANIFEST.json",
        "c12_schema_manifest": ROOT / "c12_1_schema_and_split/C12_1_SCHEMA_MANIFEST.json",
        "c12_t2_checkpoint": ROOT / "c12_models/c12_5t_T2_listwise_soft_iou.pt",
        "c16_checkpoint_manifest": ROOT / "c16_2_fullscale_bmn_training/C16_2_BMN_CHECKPOINT_MANIFEST.json",
        "c17_score_manifest": ROOT / "c17_2_full_bmn_score_export/C17_2_FULL_BMN_SCORE_MANIFEST.json",
    }
    missing = [k for k, p in deps.items() if not p.exists()]
    c17_1 = load_json(deps["c17_evaluator"], {})
    c17_2 = load_json(deps["c17_score"], {})
    c17_3 = load_json(deps["c17_hybrid"], {})
    c17_4 = load_json(deps["c17_integration"], {})
    c17_5 = load_json(deps["c17_robustness"], {})
    c17_6 = load_json(deps["c17_next"], {})
    root_markers = [p.name for p in [ROOT / "C9_OFFICIAL_VAL_AUTHORIZED", ROOT / "OFFICIAL_VAL_AUTHORIZED"] if p.exists()]
    checks = {
        "branch_expected": branch == "c18-full-hybrid-freeze-candidate",
        "head_contains_c17_commit": "87509b4" in sh("git log --oneline -5"),
        "c17_1_ready": c17_1.get("status") == "C17_TRAIN_ONLY_EVALUATOR_READY",
        "c17_2_partial": c17_2.get("status") == "C17_FULL_BMN_SCORE_PARTIAL",
        "c17_3_promising": c17_3.get("status") == "C17_HYBRID_PROMISING",
        "c17_4_promising": c17_4.get("status") == "C17_NATIVE_VCMR_PROMISING",
        "c17_5_pass": c17_5.get("status") == "C17_ROBUSTNESS_PASS",
        "c17_6_continue": c17_6.get("status") == "C17_CONTINUE_HYBRID_TRAIN_ONLY",
        "best_formula_parseable": bool(c17_3.get("best_formula_name") or c17_3.get("best_formula")),
        "official_false": c17_6.get("official_val_used") is False,
        "pseudo_not_selection": c17_6.get("pseudo_official_holdout_used_for_selection") is False,
        "score_cache_exists": mode_cache("medium").exists(),
        "no_root_stale_authorization_marker": not root_markers,
    }
    if missing:
        status = "C18_PROTOCOL_BLOCKED_MISSING_CORE_ARTIFACTS"
    elif root_markers:
        status = "C18_PROTOCOL_BLOCKED_CONTAMINATION_RISK"
    elif all(checks.values()):
        status = "C18_PROTOCOL_READY"
    else:
        status = "C18_PROTOCOL_BLOCKED_MISSING_CORE_ARTIFACTS"
    rec = {
        "stage": "C18-0",
        "status": status,
        "mode": mode,
        "seed": seed,
        "branch": branch,
        "commit": commit,
        "dirty_status": dirty.splitlines(),
        "checks": checks,
        "missing_core_artifacts": missing,
        "root_stale_authorization_markers": root_markers,
        "current_promoted_system": PROMOTED,
        "c18_is_promoted_system": False,
        "official_val_used": False,
        "pseudo_official_holdout_used_for_selection": False,
    }
    write_json(OUT0 / "C18_0_PROTOCOL.json", rec)
    write_json(OUT0 / "C18_0_DEPENDENCY_AUDIT.json", {k: {"path": str(p.relative_to(ROOT)), "exists": p.exists(), "sha256": sha256_file(p) if p.exists() and p.is_file() else None} for k, p in deps.items()})
    write_json(OUT0 / "C18_0_REPRODUCIBILITY_MANIFEST.json", {
        "mode": mode,
        "seed": seed,
        "device": str(DEVICE),
        "gpu": sh("nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader || true").splitlines(),
        "python": sh("/home/a/miniconda3/envs/conquer-rlem/bin/python --version"),
        "git_branch_vv": sh("git branch -vv"),
        "git_log": sh("git log --oneline --decorate -5"),
    })
    write_text(OUT0 / "C18_0_PROTOCOL.md", f"# C18-0 Protocol\n\nstatus: `{status}`\n\nC18 is train-only. official validation is forbidden and was not run.")
    write_text(OUT0 / "C18_0_C17_ACCEPTANCE.md", f"# C18-0 C17 Acceptance\n\nC17 hybrid is accepted as train-only positive evidence, not as a promoted official system.\n\nBest formula: `{c17_3.get('best_formula_name')}`")
    write_text(OUT0 / "C18_0_FORBIDDEN_ACTIONS_AUDIT.md", "# C18-0 Forbidden Actions Audit\n\nNo official validation, no official pool reads, no evaluator/NMS modification, no pseudo-official selection, and no C18 promoted-system claim.")
    return rec


def stage_c18_1(mode: str, seed: int, force: bool = False) -> Dict[str, Any]:
    require_ready(OUT0 / "C18_0_PROTOCOL.json", ["C18_PROTOCOL_READY"])
    OUT1.mkdir(parents=True, exist_ok=True)
    cache = mode_cache(mode)
    if not cache.exists() and not force:
        cache = mode_cache("medium")
    audit = score_cache_audit(cache)
    if not audit.get("exists"):
        status = "C18_FULL_SCORE_BLOCKED"
    elif mode != "full" and audit["missing_bmn_score_count"] == 0 and audit["missing_t2_score_count"] == 0:
        status = "C18_FULL_SCORE_MEDIUM_ONLY_READY"
    elif audit.get("row_count", 0) > 0:
        status = "C18_FULL_SCORE_PARTIAL_REPRODUCIBLE"
    else:
        status = "C18_FULL_SCORE_BLOCKED"
    reason = {
        "c17_2_partial_reason": [
            "mode is medium/top100 rather than full all-split export",
            "large score table is local-only and not committed to GitHub",
            "BMN/T2 score columns have audited missing values because candidate union contains spans from one source not present in the other",
        ],
        "not_full_ready_reason": "missing BMN/T2 cross-source scores remain audited rather than silently filled",
    }
    rec = {
        "stage": "C18-1",
        "status": status,
        "mode": mode,
        "score_cache_manifest": audit,
        **reason,
        "generation_command": "run_c17_bmn_t2_native_vcmr_integration.py --stage all --mode medium --seed 2026 --force",
        "pseudo_official_holdout_only_diagnostic": True,
        "official_val_used": False,
    }
    sample_cols = ["split", "query_id", "seed", "video_id", "span_start", "span_end", "retriever_score", "bmn_final_score", "t2_score", "query_type", "duration_bucket"]
    if audit.get("exists"):
        pd.read_parquet(cache, columns=sample_cols).head(5000).to_parquet(OUT1 / "C18_1_SCORE_SAMPLE.parquet", index=False)
    write_text(OUT1 / "C18_1_FULL_SCORE_PLAN.md", f"# C18-1 Full Score Readiness\n\nUse local score cache `{cache}`. The full table remains local-only; GitHub receives manifest and sample only.")
    write_json(OUT1 / "C18_1_SCORE_CACHE_MANIFEST.json", audit)
    write_json(OUT1 / "C18_1_SCORE_SCHEMA.json", {"columns": c17.SCORE_COLUMNS, "mode": mode, "status": status})
    write_json(OUT1 / "C18_1_SCORE_REPRODUCIBILITY_AUDIT.json", rec)
    write_json(OUT1 / "C18_1_SCORE_DISTRIBUTION_AUDIT.json", {k: audit.get(k) for k in ["score_distribution_by_split", "score_distribution_by_query_type", "score_distribution_by_duration_bucket", "calib_select_vs_calib_holdout_distribution_shift"]})
    write_json(OUT1 / "C18_1_FULL_SCORE_DECISION.json", rec)
    write_text(OUT1 / "C18_1_FULL_SCORE_DECISION.md", f"# C18-1 Full Score Decision\n\nstatus: `{status}`\n\nC18-1 is reproducible from the local cache, but it is not promoted to FULL_SCORE_READY because missing BMN/T2 source scores are audited rather than silently filled.")
    return rec


def stage_c18_2(mode: str, seed: int) -> Dict[str, Any]:
    require_ready(OUT1 / "C18_1_FULL_SCORE_DECISION.json", ["C18_FULL_SCORE_READY", "C18_FULL_SCORE_MEDIUM_ONLY_READY", "C18_FULL_SCORE_PARTIAL_REPRODUCIBLE"])
    OUT2.mkdir(parents=True, exist_ok=True)
    hres = load_json(ROOT / "c17_3_bmn_t2_hybrid_calibration/C17_3_HYBRID_RESULTS.json", {})
    f = hres.get("formula_results", {})
    mapping = {
        "C12_5T_T2": "B_T2_only",
        "C17_best_hybrid": hres.get("best_formula_name", BEST_C17["name"]),
        "BMN_only": "C_BMN_map_only",
        "retriever_BMN": "D_retriever_BMN",
        "retriever_T2": "E_retriever_T2",
        "retriever_BMN_T2": hres.get("best_formula_name", BEST_C17["name"]),
    }
    comparison = {}
    for name, key in mapping.items():
        if key in f:
            comparison[name] = {
                "formula_key": key,
                "calib_select": f[key].get("calib_select"),
                "calib_holdout": f[key].get("calib_holdout"),
            }
    c7_audit = {
        "status": "C18_C7_B6_REPLAY_UNAVAILABLE",
        "reason": "No compatible train-only C7-B6 prediction rows were found for the C18 candidate/evaluator schema; official prediction pool is forbidden and was not read.",
        "promoted_system_retained": PROMOTED,
    }
    status = "C18_BASELINE_REPLAY_PARTIAL" if c7_audit["status"].endswith("UNAVAILABLE") else "C18_BASELINE_REPLAY_READY"
    rec = {
        "stage": "C18-2",
        "status": status,
        "mode": mode,
        "baseline_comparison": comparison,
        "c7_b6_replay_audit": c7_audit,
        "sanity": {
            "random_score_sanity": "available in C17-4 native replay",
            "oracle_score_sanity": "available in C17-4 native replay",
            "c12_t2_replay_sanity": "C12-5T T2 formula replayed under C17 evaluator",
            "c17_hybrid_replay_sanity": "C17 best hybrid replayed under C17 evaluator",
            "timestamp_mapping_sanity": "C12 idx_to_ts/ts_to_idx reused",
            "nms_sanity": "train-only per-video temporal NMS; official NMS not modified",
        },
        "official_val_used": False,
        "pseudo_official_holdout_used_for_selection": False,
    }
    write_text(OUT2 / "C18_2_BASELINE_REPLAY_PLAN.md", "# C18-2 Baseline Comparable Replay\n\nCompare C12 T2, C17 hybrid, BMN-only, retriever+BMN, retriever+T2, and retriever+BMN+T2 in the same train-only evaluator schema.")
    write_json(OUT2 / "C18_2_C12_T2_REPLAY_RESULTS.json", comparison.get("C12_5T_T2", {}))
    write_json(OUT2 / "C18_2_C17_HYBRID_REPLAY_RESULTS.json", comparison.get("C17_best_hybrid", {}))
    write_json(OUT2 / "C18_2_C7_B6_REPLAY_AUDIT.json", c7_audit)
    write_json(OUT2 / "C18_2_BASELINE_COMPARISON_TABLE.json", comparison)
    write_json(OUT2 / "C18_2_REPLAY_SANITY_AUDIT.json", rec["sanity"])
    write_json(OUT2 / "C18_2_BASELINE_REPLAY_DECISION.json", rec)
    write_text(OUT2 / "C18_2_BASELINE_REPLAY_DECISION.md", f"# C18-2 Baseline Replay Decision\n\nstatus: `{status}`\n\nC7-B6 train-only replay is unavailable under the compatible schema; official prediction rows were not read.")
    return rec


def stage_c18_3(mode: str, seed: int) -> Dict[str, Any]:
    require_ready(OUT2 / "C18_2_BASELINE_REPLAY_DECISION.json", ["C18_BASELINE_REPLAY_READY", "C18_BASELINE_REPLAY_PARTIAL"])
    OUT3.mkdir(parents=True, exist_ok=True)
    df, load_audit = load_eval_table_for_c18(mode)
    fast_ctx = prepare_fast_eval(df)
    formulas = c18_search_space()
    c17_eval = fast_evaluate_formula(fast_ctx, df, BEST_C17)
    t2_eval = fast_evaluate_formula(fast_ctx, df, {"name": "B_T2_only", "family": "B_T2_only", "gamma": 1.0})
    results: Dict[str, Any] = {}
    best_name, best_score = "", -1e18
    base_select = c17_eval["calib_select"]
    for i, formula in enumerate(formulas, start=1):
        if i == 1 or i % 20 == 0:
            print(f"[C18-3] formula {i}/{len(formulas)} {formula['name']}", flush=True)
        res = fast_evaluate_formula(fast_ctx, df, formula)
        sel_score = front_rank_score(res["calib_select"], base_select)
        res["front_rank_selection_score"] = sel_score
        results[formula["name"]] = res
        if sel_score > best_score:
            best_name, best_score = formula["name"], sel_score
    selected = results[best_name]
    hold = selected["calib_holdout"]["summary"]
    c17_hold = c17_eval["calib_holdout"]["summary"]
    t2_hold = t2_eval["calib_holdout"]["summary"]
    r100_keep = hold["VCMR_R@100_IoU0.7"] >= c17_hold["VCMR_R@100_IoU0.7"] - 1.0
    front_ok = (
        hold["VCMR_R@1_IoU0.7"] >= c17_hold["VCMR_R@1_IoU0.7"]
        or hold["VCMR_R@5_IoU0.7"] >= c17_hold["VCMR_R@5_IoU0.7"]
        or hold["VCMR_R@10_IoU0.7"] >= c17_hold["VCMR_R@10_IoU0.7"]
    )
    wrong_ok = hold["wrong_video_high_score_rate"] <= c17_hold["wrong_video_high_score_rate"] + 0.5
    bmn_removal = hold["VCMR_R@100_IoU0.7"] - t2_hold["VCMR_R@100_IoU0.7"]
    if front_ok and r100_keep and wrong_ok and bmn_removal > 0:
        status = "C18_FRONT_RANK_CALIBRATION_PROMISING"
    elif r100_keep and not front_ok:
        status = "C18_FRONT_RANK_CALIBRATION_R100_ONLY"
    elif hold["VCMR_R@100_IoU0.7"] < t2_hold["VCMR_R@100_IoU0.7"]:
        status = "C18_FRONT_RANK_CALIBRATION_WEAK"
    else:
        status = "C18_FRONT_RANK_CALIBRATION_INCONCLUSIVE"
    decision = {
        "stage": "C18-3",
        "status": status,
        "mode": mode,
        "load_audit": load_audit,
        "selected_name": best_name,
        "selected_formula": selected["formula"],
        "selected_config_hash": stable_hash(selected["formula"]),
        "selected_on": "calib_select",
        "c17_best_formula": BEST_C17,
        "c17_replay": c17_eval,
        "t2_replay": t2_eval,
        "selected_results": selected,
        "c18_vs_c17_holdout_delta": {k: hold.get(k, 0.0) - c17_hold.get(k, 0.0) for k in hold if isinstance(hold.get(k), (int, float)) and isinstance(c17_hold.get(k), (int, float))},
        "c18_vs_t2_holdout_delta": {k: hold.get(k, 0.0) - t2_hold.get(k, 0.0) for k in hold if isinstance(hold.get(k), (int, float)) and isinstance(t2_hold.get(k), (int, float))},
        "bmn_removal_delta_R100_IoU0.7": bmn_removal,
        "t2_removal_delta_R100_IoU0.7": hold["VCMR_R@100_IoU0.7"] - fast_evaluate_formula(fast_ctx, df, {"name": "D_retriever_BMN", "family": "D_retriever_BMN", "alpha": 0.45, "beta": 0.55})["calib_holdout"]["summary"]["VCMR_R@100_IoU0.7"],
        "pseudo_official_holdout_used_for_selection": False,
        "official_val_used": False,
    }
    write_text(OUT3 / "C18_3_FRONT_RANK_OBJECTIVE.md", "# C18-3 Front-rank Objective\n\nSelection optimizes calib_select front-rank deltas versus C17 while penalizing wrong-video increases. R@100 is a guardrail, not the only objective.")
    write_json(OUT3 / "C18_3_SEARCH_SPACE.json", {"formulas": formulas})
    write_json(OUT3 / "C18_3_HYBRID_RESULTS.json", results)
    write_json(OUT3 / "C18_3_FRONT_RANK_RESULTS.json", {"selected": selected, "c17": c17_eval, "t2": t2_eval})
    write_json(OUT3 / "C18_3_WRONG_VIDEO_AUDIT.json", {"selected_holdout": hold.get("wrong_video_high_score_rate"), "c17_holdout": c17_hold.get("wrong_video_high_score_rate"), "t2_holdout": t2_hold.get("wrong_video_high_score_rate")})
    write_json(OUT3 / "C18_3_COMPONENT_CONTRIBUTION_AUDIT.json", {"bmn_removal_delta_R100_IoU0.7": decision["bmn_removal_delta_R100_IoU0.7"], "t2_removal_delta_R100_IoU0.7": decision["t2_removal_delta_R100_IoU0.7"]})
    write_json(OUT3 / "C18_3_SELECTED_CONFIG.json", {"selected_formula": selected["formula"], "selected_config_hash": decision["selected_config_hash"], "frozen_before_pseudo_onelook": True})
    write_json(OUT3 / "C18_3_CALIBRATION_DECISION.json", decision)
    write_text(OUT3 / "C18_3_CALIBRATION_DECISION.md", f"# C18-3 Calibration Decision\n\nstatus: `{status}`\n\nSelected `{best_name}` on calib_select. pseudo_official_holdout was not used for selection.")
    return decision


def stage_c18_4(mode: str, seed: int, force: bool = False) -> Dict[str, Any]:
    require_ready(OUT3 / "C18_3_CALIBRATION_DECISION.json", ["C18_FRONT_RANK_CALIBRATION_PROMISING", "C18_FRONT_RANK_CALIBRATION_R100_ONLY", "C18_FRONT_RANK_CALIBRATION_WEAK", "C18_FRONT_RANK_CALIBRATION_INCONCLUSIVE"])
    OUT4.mkdir(parents=True, exist_ok=True)
    selected = load_json(OUT3 / "C18_3_SELECTED_CONFIG.json", {})
    pseudo_cache = c18_pseudo_cache(mode)
    if not pseudo_cache.exists() or force:
        print("[C18-4] building pseudo_official_holdout score table after config freeze", flush=True)
        df, audit = c17.generate_score_table(mode, seed, ["pseudo_official_holdout"])
        df.to_parquet(pseudo_cache, index=False)
        write_json(pseudo_cache.with_suffix(".audit.json"), audit)
    df = pd.read_parquet(pseudo_cache)
    for col in ["split", "video_id", "gt_video_id", "query_type", "duration_bucket"]:
        if col in df:
            df[col] = df[col].astype("category")
    _fill = c17.add_normalized_scores_inplace(df)
    formula = selected["selected_formula"]
    fast_ctx = prepare_fast_eval(df)
    res = fast_evaluate_formula(fast_ctx, df, formula)
    hold = load_json(OUT3 / "C18_3_CALIBRATION_DECISION.json", {}).get("selected_results", {}).get("calib_holdout", {})
    pseudo = res.get("pseudo_official_holdout", {})
    gap = None
    if hold and pseudo:
        gap = pseudo["summary"].get("VCMR_R@100_IoU0.7", 0.0) - hold["summary"].get("VCMR_R@100_IoU0.7", 0.0)
    if pseudo and gap is not None and gap >= -5.0:
        status = "C18_PSEUDO_ONELOOK_STABLE"
    elif pseudo and gap is not None and gap >= -12.0:
        status = "C18_PSEUDO_ONELOOK_SHIFT_WARNING"
    elif pseudo:
        status = "C18_PSEUDO_ONELOOK_COLLAPSE"
    else:
        status = "C18_PSEUDO_ONELOOK_SKIPPED"
    rec = {
        "stage": "C18-4",
        "status": status,
        "mode": mode,
        "selected_config_before_onelook": True,
        "selected_formula": formula,
        "selected_config_hash": stable_hash(formula),
        "pseudo_cache": str(pseudo_cache),
        "pseudo_cache_sha256": sha256_file(pseudo_cache) if pseudo_cache.exists() else None,
        "results": res,
        "vs_calib_holdout_gap_R100_IoU0.7": gap,
        "selection_firewall": {
            "selected_config_before_onelook": True,
            "no_post_onelook_adjustment": True,
            "pseudo_official_not_used_for_selection": True,
        },
        "official_val_used": False,
        "pseudo_official_holdout_used_for_selection": False,
    }
    write_text(OUT4 / "C18_4_ONELOOK_PROTOCOL.md", "# C18-4 Pseudo-official One-look Protocol\n\nThe C18-3 config is frozen before this diagnostic. Results cannot change weights, thresholds, or formula.")
    write_json(OUT4 / "C18_4_ONELOOK_RESULTS.json", res)
    write_json(OUT4 / "C18_4_DISTRIBUTION_SHIFT_AUDIT.json", {"gap_R100_IoU0.7": gap})
    write_json(OUT4 / "C18_4_SELECTION_FIREWALL_AUDIT.json", rec["selection_firewall"])
    write_json(OUT4 / "C18_4_ONELOOK_DECISION.json", rec)
    write_text(OUT4 / "C18_4_ONELOOK_DECISION.md", f"# C18-4 One-look Decision\n\nstatus: `{status}`\n\nNo post-onelook adjustment was made.")
    return rec


def stage_c18_5(mode: str, seed: int) -> Dict[str, Any]:
    require_ready(OUT4 / "C18_4_ONELOOK_DECISION.json", ["C18_PSEUDO_ONELOOK_STABLE", "C18_PSEUDO_ONELOOK_SHIFT_WARNING", "C18_PSEUDO_ONELOOK_COLLAPSE", "C18_PSEUDO_ONELOOK_SKIPPED"])
    OUT5.mkdir(parents=True, exist_ok=True)
    c3 = load_json(OUT3 / "C18_3_CALIBRATION_DECISION.json", {})
    c4 = load_json(OUT4 / "C18_4_ONELOOK_DECISION.json", {})
    hold = c3.get("selected_results", {}).get("calib_holdout", {}).get("summary", {})
    c17_hold = c3.get("c17_replay", {}).get("calib_holdout", {}).get("summary", {})
    seed_rob = load_json(ROOT / "c17_5_robustness_and_consistency/C17_5_SEED_ROBUSTNESS.json", {})
    front_drop = hold.get("VCMR_R@1_IoU0.7", 0.0) < c17_hold.get("VCMR_R@1_IoU0.7", 0.0) - 0.5 and hold.get("VCMR_R@5_IoU0.7", 0.0) < c17_hold.get("VCMR_R@5_IoU0.7", 0.0) - 0.5
    wrong_worse = hold.get("wrong_video_high_score_rate", 0.0) > c17_hold.get("wrong_video_high_score_rate", 0.0) + 1.0
    if c4.get("status") == "C18_PSEUDO_ONELOOK_STABLE" and not front_drop and not wrong_worse and c3.get("status") == "C18_FRONT_RANK_CALIBRATION_PROMISING":
        status = "C18_ROBUSTNESS_PASS"
    elif c4.get("status") in {"C18_PSEUDO_ONELOOK_STABLE", "C18_PSEUDO_ONELOOK_SHIFT_WARNING"} and not wrong_worse:
        status = "C18_ROBUSTNESS_PARTIAL"
    else:
        status = "C18_ROBUSTNESS_FAIL"
    rec = {
        "stage": "C18-5",
        "status": status,
        "mode": mode,
        "medium_full_consistency": "full not run; medium cache is reproducible, so official-ready is blocked",
        "seed_robustness": seed_rob,
        "front_rank_drop": front_drop,
        "wrong_video_worse": wrong_worse,
        "pseudo_onelook_status": c4.get("status"),
        "official_val_used": False,
        "pseudo_official_holdout_used_for_selection": False,
    }
    write_json(OUT5 / "C18_5_MEDIUM_FULL_CONSISTENCY.json", {"medium": hold, "full": None, "status": "full_not_run"})
    write_json(OUT5 / "C18_5_SEED_ROBUSTNESS.json", seed_rob)
    write_json(OUT5 / "C18_5_FRONT_RANK_ROBUSTNESS.json", {"selected_holdout": hold, "c17_holdout": c17_hold, "front_rank_drop": front_drop})
    write_json(OUT5 / "C18_5_QUERY_DURATION_ROBUSTNESS.json", {"query_type": c3.get("selected_results", {}).get("calib_holdout", {}).get("query_type_breakdown", {}), "duration": c3.get("selected_results", {}).get("calib_holdout", {}).get("duration_breakdown", {})})
    write_json(OUT5 / "C18_5_SCORE_DISTRIBUTION_AUDIT.json", load_json(OUT1 / "C18_1_SCORE_DISTRIBUTION_AUDIT.json", {}))
    write_text(OUT5 / "C18_5_ERROR_CASES.md", "# C18-5 Error Cases\n\nDetailed case-level prediction rows remain in local-only score caches; no official prediction pool was read.")
    write_json(OUT5 / "C18_5_ROBUSTNESS_DECISION.json", rec)
    write_text(OUT5 / "C18_5_ROBUSTNESS_DECISION.md", f"# C18-5 Robustness Decision\n\nstatus: `{status}`\n\nFull mode was not run automatically; C18 remains train-only.")
    return rec


def stage_c18_6(mode: str, seed: int) -> Dict[str, Any]:
    OUT6.mkdir(parents=True, exist_ok=True)
    s0 = load_json(OUT0 / "C18_0_PROTOCOL.json", {})
    s1 = load_json(OUT1 / "C18_1_FULL_SCORE_DECISION.json", {})
    s2 = load_json(OUT2 / "C18_2_BASELINE_REPLAY_DECISION.json", {})
    s3 = load_json(OUT3 / "C18_3_CALIBRATION_DECISION.json", {})
    s4 = load_json(OUT4 / "C18_4_ONELOOK_DECISION.json", {})
    s5 = load_json(OUT5 / "C18_5_ROBUSTNESS_DECISION.json", {})
    ready = (
        s1.get("status") == "C18_FULL_SCORE_READY"
        and s3.get("status") == "C18_FRONT_RANK_CALIBRATION_PROMISING"
        and s4.get("status") == "C18_PSEUDO_ONELOOK_STABLE"
        and s5.get("status") in {"C18_ROBUSTNESS_PASS", "C18_ROBUSTNESS_PARTIAL"}
    )
    if ready:
        final = "C18_READY_FOR_ONE_SHOT_OFFICIAL_REVIEW"
    elif s3.get("status") == "C18_FRONT_RANK_CALIBRATION_R100_ONLY":
        final = "C18_NEED_FRONT_RANK_CALIBRATION"
    elif s3.get("status") == "C18_FRONT_RANK_CALIBRATION_PROMISING" and s1.get("status") != "C18_FULL_SCORE_READY":
        final = "C18_CONTINUE_FULL_HYBRID_TRAIN_ONLY"
    elif s4.get("status") == "C18_PSEUDO_ONELOOK_COLLAPSE":
        final = "C18_NEED_RETRIEVER_LOCALIZER_CALIBRATION"
    else:
        final = "C18_CONTINUE_FULL_HYBRID_TRAIN_ONLY"
    rec = {
        "stage": "C18-6",
        "status": final,
        "final_decision": final,
        "mode": mode,
        "seed": seed,
        "c18_0_protocol_status": s0.get("status"),
        "c18_1_full_score_status": s1.get("status"),
        "c18_2_baseline_replay_status": s2.get("status"),
        "c18_3_front_rank_status": s3.get("status"),
        "c18_4_pseudo_onelook_status": s4.get("status"),
        "c18_5_robustness_status": s5.get("status"),
        "official_val_used": False,
        "pseudo_official_holdout_used_for_selection": False,
        "evaluator_modified": False,
        "nms_modified": False,
        "schema_hash": s1.get("score_cache_manifest", {}).get("schema_hashes"),
        "candidate_pool_hash": s1.get("score_cache_manifest", {}).get("sha256"),
        "score_manifest_hash": stable_hash(s1.get("score_cache_manifest", {})),
        "selected_config": s3.get("selected_name"),
        "selected_formula": s3.get("selected_formula"),
        "command_lines": [f"run_c18_full_hybrid_freeze_candidate.py --stage all --mode {mode} --seed {seed}"],
        "seed_list": c17.seed_list(seed, mode),
        "missing_artifacts": ["full all-split score table", "compatible C7-B6 train-only replay"] if s1.get("status") != "C18_FULL_SCORE_READY" else ["compatible C7-B6 train-only replay"],
        "local_only_artifacts": [str(mode_cache(mode)), str(c18_pseudo_cache(mode))],
        "current_promoted_system": PROMOTED,
        "c18_is_promoted_system": False,
    }
    write_json(OUT6 / "C18_6_FREEZE_REVIEW.json", rec)
    write_json(OUT6 / "C18_6_OFFICIAL_READINESS_PACKET.json", {**rec, "official_may_be_run_by_this_script": False})
    write_json(OUT6 / "C18_6_NEXT_STEP_DECISION.json", rec)
    write_text(OUT6 / "C18_6_FREEZE_REVIEW.md", f"# C18-6 Freeze Review\n\nfinal_decision: `{final}`\n\nC18 remains train-only. official validation was not run.")
    write_text(OUT6 / "C18_6_OFFICIAL_READINESS_PACKET.md", f"# C18-6 Official Readiness Packet\n\ndecision: `{final}`\n\nThis packet is not authorization to run official validation.")
    write_text(OUT6 / "C18_6_RISK_REGISTER.md", "# C18-6 Risk Register\n\n- C18-1 is not FULL_SCORE_READY unless full score export has no audited missing source scores.\n- C7-B6 compatible train-only replay is unavailable.\n- Raw TVR video is absent.\n- C7-B6 remains the promoted official system.")
    write_text(OUT6 / "C18_6_NEXT_STEP_DECISION.md", f"# C18-6 Next Step Decision\n\ndecision: `{final}`\n\nDo not run official from this script.")
    return rec


def print_summary(final: Dict[str, Any]) -> None:
    c3 = load_json(OUT3 / "C18_3_CALIBRATION_DECISION.json", {})
    selected = c3.get("selected_results", {}).get("calib_holdout", {}).get("summary", {})
    t2 = c3.get("t2_replay", {}).get("calib_holdout", {}).get("summary", {})
    c17_best = c3.get("c17_replay", {}).get("calib_holdout", {}).get("summary", {})
    print("C18 SUMMARY")
    print(f"branch: {sh('git branch --show-current')}")
    print(f"commit: {sh('git rev-parse --short HEAD')}")
    print(f"C18-0 protocol status: {final.get('c18_0_protocol_status')}")
    print(f"C18-1 full score status: {final.get('c18_1_full_score_status')}")
    print(f"C18-2 baseline replay status: {final.get('c18_2_baseline_replay_status')}")
    print(f"C18-3 front-rank calibration status: {final.get('c18_3_front_rank_status')}")
    print(f"C18-4 pseudo one-look status: {final.get('c18_4_pseudo_onelook_status')}")
    print(f"C18-5 robustness status: {final.get('c18_5_robustness_status')}")
    print(f"C18-6 final decision: {final.get('final_decision')}")
    print(f"selected formula and weights: {c3.get('selected_formula')}")
    print(f"VCMR metrics: R1@0.7={selected.get('VCMR_R@1_IoU0.7')} R5@0.7={selected.get('VCMR_R@5_IoU0.7')} R10@0.7={selected.get('VCMR_R@10_IoU0.7')} R100@0.7={selected.get('VCMR_R@100_IoU0.7')}")
    print(f"vs C12-5T T2 delta R100@0.7: {selected.get('VCMR_R@100_IoU0.7', 0.0) - t2.get('VCMR_R@100_IoU0.7', 0.0)}")
    print(f"vs C17 best delta R100@0.7: {selected.get('VCMR_R@100_IoU0.7', 0.0) - c17_best.get('VCMR_R@100_IoU0.7', 0.0)}")
    print(f"wrong-video high-score rate: {selected.get('wrong_video_high_score_rate')}")
    print("pseudo_official_holdout used for selection: false")
    print("official was not run: true")
    print(f"local-only full score table path: {mode_cache(final.get('mode', 'medium'))}")
    print("files committed to GitHub: C18 code, JSON/MD manifests, and small sample parquet only")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", default="all", choices=["c18_0", "c18_1", "c18_2", "c18_3", "c18_4", "c18_5", "c18_6", "all"])
    parser.add_argument("--mode", default="medium", choices=["smoke", "medium", "full"])
    parser.add_argument("--seed", default=2026, type=int)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.stage in {"c18_0", "all"}:
        stage_c18_0(args.mode, args.seed)
    if args.stage in {"c18_1", "all"}:
        stage_c18_1(args.mode, args.seed, force=args.force)
    if args.stage in {"c18_2", "all"}:
        stage_c18_2(args.mode, args.seed)
    if args.stage in {"c18_3", "all"}:
        stage_c18_3(args.mode, args.seed)
        gc.collect()
    if args.stage in {"c18_4", "all"}:
        stage_c18_4(args.mode, args.seed, force=args.force)
        gc.collect()
    if args.stage in {"c18_5", "all"}:
        stage_c18_5(args.mode, args.seed)
    final = None
    if args.stage in {"c18_6", "all"}:
        final = stage_c18_6(args.mode, args.seed)
    if final:
        print_summary(final)


if __name__ == "__main__":
    main()
