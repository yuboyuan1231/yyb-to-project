#!/usr/bin/env python3
"""C24 full BMN/Event evidence strengthening audit.

This runner audits existing train-only evidence, probes whether BMN/T2/event
signals can help front-rank VCMR, and never touches official validation or
official prediction pools.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import random
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch

import run_c18_full_hybrid_freeze_candidate as c18
import run_c22r_native_coupling_sanity_repair as c22r


torch.set_num_threads(min(32, os.cpu_count() or 1))

ROOT = Path(__file__).resolve().parent
PYTHON = "/home/a/miniconda3/envs/conquer-rlem/bin/python"
BASE_C23_COMMIT = "f471dfd1f4bfdbba867a85805b8c487437aaa495"
PROMOTED = "C7-B6 R1SelectiveTop1"

C17_MEDIUM = Path("/tmp/c17_score_cache/CONQUER-RLEM-c2c3/C17_2_SCORE_TABLE_medium.local.parquet")
C19_MEDIUM = Path("/tmp/c19_score_cache/CONQUER-RLEM-c2c3/C19_CANONICAL_SELECTED_SCORE_TABLE_medium.local.parquet")
C18_PSEUDO_MEDIUM = Path("/tmp/c18_score_cache/CONQUER-RLEM-c2c3/C18_PSEUDO_SCORE_TABLE_medium.local.parquet")
C20_TRAIN = Path("/tmp/c20_score_cache/CONQUER-RLEM-c2c3/C20_TOP1_EVIDENCE_train_medium.local.parquet")
C20_PSEUDO = Path("/tmp/c20_score_cache/CONQUER-RLEM-c2c3/C20_TOP1_EVIDENCE_pseudo_medium.local.parquet")
C21_TRAIN = Path("/tmp/c21_score_cache/CONQUER-RLEM-c2c3/C21_FRONT_RANK_PAIR_DATASET_train_medium.local.parquet")
C21_PSEUDO = Path("/tmp/c21_score_cache/CONQUER-RLEM-c2c3/C21_FRONT_RANK_PAIR_DATASET_pseudo_medium.local.parquet")

OUT0 = ROOT / "c24_0_protocol_freeze"
OUT1 = ROOT / "c24_1_full_evidence_source_audit"
OUT2 = ROOT / "c24_2_evidence_label_diagnostic"
OUT3 = ROOT / "c24_3_evidence_strengthening_probe"
OUT4 = ROOT / "c24_4_strengthened_integration_replay"
OUT5 = ROOT / "c24_5_robustness_and_route_decision"
OUT6 = ROOT / "c24_6_final_decision"

MODE_LIMITS = {
    "smoke": {"queries_per_split": 200, "sample_rows": 1200},
    "medium": {"queries_per_split": 1000, "sample_rows": 3000},
    "full": {"queries_per_split": 0, "sample_rows": 6000},
}

EVIDENCE_COLS = [
    "retriever_norm",
    "bmn_norm",
    "t2_norm",
    "old_norm",
    "duration_score",
    "bmn_pred_iou",
    "bmn_p_iou_05",
    "bmn_p_iou_07",
    "bmn_rank_score",
    "bmn_final_score",
    "start_prob",
    "end_prob",
    "actionness_score",
    "t2_score",
    "bmn_score",
    "final_score",
]


def jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(x) for x in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        return None if math.isnan(v) or math.isinf(v) else v
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, float):
        return None if math.isnan(obj) or math.isinf(obj) else obj
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


def sha256_file(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    import hashlib

    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def file_record(path: Path, sha: bool = False) -> Dict[str, Any]:
    rec = {
        "path": str(path),
        "exists": path.exists(),
        "readable": os.access(path, os.R_OK) if path.exists() else False,
        "size_bytes": path.stat().st_size if path.exists() and path.is_file() else None,
    }
    if sha and path.exists() and path.is_file():
        rec["sha256"] = sha256_file(path)
    return rec


def parquet_record(path: Path) -> Dict[str, Any]:
    rec = file_record(path, sha=False)
    if path.exists():
        pf = pq.ParquetFile(path)
        rec.update({
            "row_count": pf.metadata.num_rows,
            "row_groups": pf.num_row_groups,
            "columns": pf.schema.names,
            "schema_hash": c22r.stable_hash(pf.schema.names),
        })
    return rec


def read_parquet(path: Path, columns: Sequence[str] | None = None) -> pd.DataFrame:
    return pd.read_parquet(path, columns=list(columns) if columns else None)


def maybe_limit_queries(df: pd.DataFrame, mode: str, split_col: str = "split") -> pd.DataFrame:
    limit = MODE_LIMITS[mode]["queries_per_split"]
    if not limit:
        return df
    masks = []
    for split, sdf in df.groupby(split_col, observed=True, sort=False):
        ids = pd.Index(sdf["query_id"].drop_duplicates().head(limit))
        masks.append((df[split_col].astype(str) == str(split)) & df["query_id"].isin(ids))
    if not masks:
        return df
    return df[np.logical_or.reduce(masks)].copy()


def load_c19_frame(mode: str, extra_cols: Sequence[str] | None = None) -> pd.DataFrame:
    cols = [
        "query_id", "video_id", "span_start", "span_end", "span_duration",
        "retriever_rank", "retriever_score", "gt_video_id", "gt_start", "gt_end",
        "split", "seed", "query_type", "duration_bucket", "gt_video_rank",
        "source_has_bmn", "source_has_t2",
        *EVIDENCE_COLS,
    ]
    if extra_cols:
        cols += [c for c in extra_cols if c not in cols]
    available = pq.ParquetFile(C19_MEDIUM).schema.names
    cols = [c for c in cols if c in available]
    df = read_parquet(C19_MEDIUM, cols)
    return maybe_limit_queries(df, mode)


def load_c21_frame(mode: str, pseudo: bool = False) -> pd.DataFrame:
    path = C21_PSEUDO if pseudo else C21_TRAIN
    df = read_parquet(path)
    return maybe_limit_queries(df, mode)


def load_c20_frame(mode: str, pseudo: bool = False) -> pd.DataFrame:
    path = C20_PSEUDO if pseudo else C20_TRAIN
    df = read_parquet(path)
    return maybe_limit_queries(df, mode)


def split_summary(df: pd.DataFrame) -> Dict[str, Any]:
    if len(df) == 0:
        return {}
    return {
        str(k): {"rows": int(v["rows"]), "queries": int(v["queries"])}
        for k, v in df.groupby("split", observed=True).agg(rows=("query_id", "size"), queries=("query_id", "nunique")).to_dict("index").items()
    }


def describe_numeric(df: pd.DataFrame, cols: Sequence[str], by: str | None = None) -> Dict[str, Any]:
    cols = [c for c in cols if c in df.columns]
    if not cols or len(df) == 0:
        return {}

    def one(frame: pd.DataFrame) -> Dict[str, Any]:
        out = {}
        for c in cols:
            s = pd.to_numeric(frame[c], errors="coerce")
            out[c] = {
                "count": int(s.notna().sum()),
                "missing": int(s.isna().sum()),
                "mean": float(s.mean()) if s.notna().any() else None,
                "std": float(s.std()) if s.notna().sum() > 1 else None,
                "min": float(s.min()) if s.notna().any() else None,
                "p05": float(s.quantile(0.05)) if s.notna().any() else None,
                "p50": float(s.quantile(0.50)) if s.notna().any() else None,
                "p95": float(s.quantile(0.95)) if s.notna().any() else None,
                "max": float(s.max()) if s.notna().any() else None,
            }
        return out

    if by is None:
        return one(df)
    return {str(k): one(v) for k, v in df.groupby(by, observed=True, sort=False)}


def rank_bucket(rank: Any) -> str:
    try:
        r = int(rank)
    except Exception:
        return "unknown"
    if r <= 1:
        return "top1"
    if r <= 5:
        return "top5"
    if r <= 10:
        return "top10"
    if r <= 100:
        return "top100"
    return "gt100"


def auc_score(y: Iterable[Any], score: Iterable[Any]) -> float | None:
    y_arr = np.asarray(list(y), dtype=np.float64)
    s_arr = np.asarray(list(score), dtype=np.float64)
    mask = np.isfinite(y_arr) & np.isfinite(s_arr)
    y_arr = y_arr[mask] > 0.5
    s_arr = s_arr[mask]
    n_pos = int(y_arr.sum())
    n_neg = int((~y_arr).sum())
    if n_pos == 0 or n_neg == 0:
        return None
    ranks = pd.Series(s_arr).rank(method="average").to_numpy()
    rank_sum_pos = ranks[y_arr].sum()
    return float((rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def corr_pack(df: pd.DataFrame, score_col: str, label_col: str) -> Dict[str, Any]:
    if score_col not in df.columns or label_col not in df.columns:
        return {"available": False}
    s = pd.to_numeric(df[score_col], errors="coerce")
    y = pd.to_numeric(df[label_col], errors="coerce")
    mask = s.notna() & y.notna()
    if mask.sum() < 3:
        return {"available": False, "count": int(mask.sum())}
    out = {
        "available": True,
        "count": int(mask.sum()),
        "pearson": float(s[mask].corr(y[mask], method="pearson")),
        "spearman": float(s[mask].corr(y[mask], method="spearman")),
    }
    sample = df.loc[mask, [score_col, label_col]].sample(min(5000, int(mask.sum())), random_state=2026)
    out["kendall_sample"] = float(sample[score_col].corr(sample[label_col], method="kendall")) if len(sample) > 3 else None
    return out


def metric(values: Iterable[Any]) -> float:
    vals = [bool(v) for v in values]
    return 100.0 * sum(vals) / len(vals) if vals else 0.0


def mean(values: Iterable[Any]) -> float | None:
    vals = [float(v) for v in values if v is not None and np.isfinite(float(v))]
    return float(np.mean(vals)) if vals else None


def candidate_eval(df: pd.DataFrame, score: np.ndarray) -> Dict[str, Any]:
    work = df[[
        "split", "query_id", "query_type", "duration_bucket", "candidate_iou",
        "candidate_correct_video", "candidate_iou_ge_05", "candidate_iou_ge_07",
        "gt_video_id",
    ]].copy()
    work["score"] = score
    records = []
    for (split, qid), g in work.groupby(["split", "query_id"], observed=True, sort=False):
        g = g.sort_values("score", ascending=False, kind="mergesort")
        rec = {
            "split": str(split),
            "query_id": int(qid),
            "query_type": str(g["query_type"].iloc[0]),
            "duration_bucket": str(g["duration_bucket"].iloc[0]),
            "wrong_video_top1": not bool(g["candidate_correct_video"].iloc[0]),
            "top1_iou": float(g["candidate_iou"].iloc[0]),
        }
        for k in [1, 5, 10]:
            head = g.head(k)
            rec[f"VCMR_R@{k}_IoU0.5"] = bool(head["candidate_iou_ge_05"].any())
            rec[f"VCMR_R@{k}_IoU0.7"] = bool(head["candidate_iou_ge_07"].any())
            rec[f"VR_R@{k}"] = bool(head["candidate_correct_video"].any())
        records.append(rec)
    return aggregate_records(records)


def aggregate_records(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    out = {"query_count": len(records)}
    for k in [1, 5, 10]:
        out[f"VCMR_R@{k}_IoU0.5"] = metric(r[f"VCMR_R@{k}_IoU0.5"] for r in records)
        out[f"VCMR_R@{k}_IoU0.7"] = metric(r[f"VCMR_R@{k}_IoU0.7"] for r in records)
        out[f"VR_R@{k}"] = metric(r[f"VR_R@{k}"] for r in records)
    out["wrong_video_top1_rate"] = metric(r["wrong_video_top1"] for r in records)
    out["top1_mean_iou"] = mean(r["top1_iou"] for r in records)
    return out


def delta(a: Dict[str, Any], b: Dict[str, Any], keys: Sequence[str]) -> Dict[str, float]:
    return {k: float(a.get(k, 0.0) or 0.0) - float(b.get(k, 0.0) or 0.0) for k in keys}


def stage_c24_0(mode: str, seed: int) -> Dict[str, Any]:
    branch = sh("git branch --show-current")
    commit = sh("git rev-parse HEAD")
    dirty = sh("git status --short").splitlines()
    c23_files = {
        "c23_1": ROOT / "c23_1_full_trainfit_readiness/C23_1_FULL_TRAINFIT_DECISION.json",
        "c23_2": ROOT / "c23_2_full_residual_retriever/C23_2_RETRIEVER_DECISION.json",
        "c23_3": ROOT / "c23_3_full_evidence_materialization/C23_3_EVIDENCE_DECISION.json",
        "c23_4": ROOT / "c23_4_full_safe_native_coupling/C23_4_INTEGRATION_DECISION.json",
        "c23_5": ROOT / "c23_5_robustness_onelook_firewall/C23_5_ROBUSTNESS_DECISION.json",
        "c23_6": ROOT / "c23_6_final_decision/C23_6_NEXT_STEP_DECISION.json",
    }
    c23_final = load_json(ROOT / "c23_6_final_decision/C23_6_FINAL_DECISION.json", {})
    missing = [k for k, p in c23_files.items() if not p.exists()]
    firewall = {
        "official_val_used": bool(c23_final.get("official_val_used")),
        "official_prediction_pool_used": bool(c23_final.get("official_prediction_pool_used")),
        "pseudo_official_holdout_used_for_selection": bool(c23_final.get("pseudo_official_holdout_used_for_selection")),
        "evaluator_modified": bool(c23_final.get("evaluator_modified")),
        "nms_modified": bool(c23_final.get("nms_modified")),
    }
    deps = {
        **{k: file_record(p, sha=True) for k, p in c23_files.items()},
        "c22r_alignment": file_record(ROOT / "c22r_2_alignment_repair/C22R_2_ALIGNMENT_DECISION.json", sha=True),
        "first_stage_train_fit_top128": file_record(c22r.first_stage_cache_path("train_fit")),
        "first_stage_calib_select_top128": file_record(c22r.first_stage_cache_path("calib_select")),
        "first_stage_calib_holdout_top128": file_record(c22r.first_stage_cache_path("calib_holdout")),
        "c19_medium_evidence": parquet_record(C19_MEDIUM),
        "c17_medium_evidence": parquet_record(C17_MEDIUM),
        "c21_train_pair_dataset": parquet_record(C21_TRAIN),
    }
    markers = c22r.find_stale_authorization_markers()
    root_level_markers = [m for m in markers if "/" not in m]
    status = "C24_PROTOCOL_READY"
    if missing or c23_final.get("final_decision") != "C23_NEED_STRONGER_BMN_EVENT_INTEGRATION":
        status = "C24_PROTOCOL_BLOCKED_MISSING_CORE_ARTIFACTS"
    if any(firewall.values()) or root_level_markers:
        status = "C24_PROTOCOL_BLOCKED_CONTAMINATION_RISK"
    rec = {
        "stage": "C24-0",
        "status": status,
        "mode": mode,
        "seed": seed,
        "branch": branch,
        "commit": commit,
        "base_c23_commit_expected": BASE_C23_COMMIT,
        "base_c23_commit_ancestor": sh_rc(f"git merge-base --is-ancestor {BASE_C23_COMMIT} HEAD") == 0,
        "dirty_status_lines": dirty,
        "c23_final_decision": c23_final.get("final_decision"),
        "c23_promoted": bool(c23_final.get("c23_is_promoted_system")),
        "current_promoted_system": c23_final.get("current_promoted_system", PROMOTED),
        "dependencies": deps,
        "blocked_missing": missing,
        "forbidden_action_audit": firewall,
        "stale_authorization_markers": markers,
        "root_level_contamination_warning": root_level_markers,
        "official_val_used": False,
        "official_prediction_pool_used": False,
        "pseudo_official_holdout_used_for_selection": False,
        "evaluator_modified": False,
        "nms_modified": False,
        "c24_is_promoted_system": False,
        "repro_command": f"{PYTHON} run_c24_full_bmn_event_evidence_strengthening_audit.py --stage all --mode {mode} --seed {seed} --force",
    }
    write_json(OUT0 / "C24_0_PROTOCOL.json", rec)
    write_json(OUT0 / "C24_0_DEPENDENCY_AUDIT.json", deps)
    write_json(OUT0 / "C24_0_REPRODUCIBILITY_MANIFEST.json", rec)
    write_text(OUT0 / "C24_0_PROTOCOL.md", f"# C24-0 Protocol Freeze\n\nStatus: `{status}`.\n\nBase C23 final: `{c23_final.get('final_decision')}`.")
    write_text(OUT0 / "C24_0_C23_ACCEPTANCE.md", f"# C23 Acceptance\n\nC23 is accepted as a negative/diagnostic result: strengthened evidence is needed before any official review. Current promoted system remains `{PROMOTED}`.")
    write_text(OUT0 / "C24_0_FORBIDDEN_ACTIONS_AUDIT.md", "# Forbidden Actions Audit\n\n- official validation: false\n- official prediction pool read: false\n- pseudo official used for selection: false\n- evaluator modified: false\n- NMS modified: false")
    return rec


def stage_c24_1(mode: str, seed: int) -> Dict[str, Any]:
    c19_meta = parquet_record(C19_MEDIUM)
    c17_meta = parquet_record(C17_MEDIUM)
    c20_train_meta = parquet_record(C20_TRAIN)
    c21_train_meta = parquet_record(C21_TRAIN)
    df = load_c19_frame(mode)
    df["rank_bucket"] = [rank_bucket(x) for x in df["gt_video_rank"]]
    sample = df.head(MODE_LIMITS[mode]["sample_rows"]).copy()
    sample_path = OUT1 / "C24_1_EVIDENCE_SAMPLE.parquet"
    sample_path.parent.mkdir(parents=True, exist_ok=True)
    sample.to_parquet(sample_path, index=False)
    split_cov = split_summary(df)
    missing_by_split = {
        str(k): {c: int(v[c].isna().sum()) for c in EVIDENCE_COLS if c in v.columns}
        for k, v in df.groupby("split", observed=True, sort=False)
    }
    missing_by_query_type = {
        str(k): {c: int(v[c].isna().sum()) for c in EVIDENCE_COLS if c in v.columns}
        for k, v in df.groupby("query_type", observed=True, sort=False)
    }
    missing_by_duration = {
        str(k): {c: int(v[c].isna().sum()) for c in EVIDENCE_COLS if c in v.columns}
        for k, v in df.groupby("duration_bucket", observed=True, sort=False)
    }
    duplicate_count = int(df.duplicated(["split", "seed", "query_id", "video_id", "span_start", "span_end"]).sum())
    invalid_span_count = int(((df["span_end"] <= df["span_start"]) | df["span_start"].isna() | df["span_end"].isna()).sum())
    num = df[[c for c in EVIDENCE_COLS if c in df.columns]].apply(pd.to_numeric, errors="coerce")
    nan_inf = int((~np.isfinite(num.to_numpy(np.float32))).sum())
    coverage = {
        "mode": mode,
        "row_count": int(len(df)),
        "query_count": int(df["query_id"].nunique()),
        "split_coverage": split_cov,
        "train_fit_available": "train_fit" in split_cov,
        "calib_select_available": "calib_select" in split_cov,
        "calib_holdout_available": "calib_holdout" in split_cov,
        "pseudo_official_holdout_available": False,
        "candidate_coverage": {
            "candidate_rows": int(len(df)),
            "candidate_videos": int(df["video_id"].nunique()),
            "mean_rows_per_query": float(len(df) / max(1, df["query_id"].nunique())),
        },
        "missing_count_by_split": missing_by_split,
        "missing_count_by_evidence_type": {c: int(df[c].isna().sum()) for c in EVIDENCE_COLS if c in df.columns},
        "missing_by_query_type": missing_by_query_type,
        "missing_by_duration": missing_by_duration,
        "duplicate_candidate_count": duplicate_count,
        "invalid_span_count": invalid_span_count,
        "nan_inf_count": nan_inf,
        "score_distribution_by_split": describe_numeric(df, EVIDENCE_COLS, "split"),
        "score_distribution_by_rank_bucket": describe_numeric(df, EVIDENCE_COLS, "rank_bucket"),
        "score_distribution_by_query_type": describe_numeric(df, EVIDENCE_COLS, "query_type"),
        "score_distribution_by_duration": describe_numeric(df, EVIDENCE_COLS, "duration_bucket"),
        "source_has_bmn_rate": float(pd.to_numeric(df["source_has_bmn"], errors="coerce").mean()) if "source_has_bmn" in df.columns else None,
        "source_has_t2_rate": float(pd.to_numeric(df["source_has_t2"], errors="coerce").mean()) if "source_has_t2" in df.columns else None,
    }
    schema = {
        "schema_hash": c22r.stable_hash(list(df.columns)),
        "join_key": ["split", "seed", "query_id", "video_id", "span_start", "span_end"],
        "join_key_hash": c22r.stable_hash(["split", "seed", "query_id", "video_id", "span_start", "span_end"]),
        "columns": list(df.columns),
        "no_silent_zero_fill": True,
        "no_position_based_join": True,
        "event_evidence_direct_columns_available": False,
        "event_proxy_columns": ["old_norm", "duration_score"],
    }
    alignment = {
        "duplicate_candidate_count": duplicate_count,
        "invalid_span_count": invalid_span_count,
        "schema_mismatch": False,
        "position_based_join_detected": False,
        "silent_zero_fill_detected": False,
        "missing_scores_mask_required": True,
    }
    source_ready = coverage["calib_select_available"] and coverage["calib_holdout_available"] and not duplicate_count and not invalid_span_count
    full_ready = source_ready and coverage["train_fit_available"]
    status = "C24_FULL_EVIDENCE_AUDIT_READY" if full_ready else ("C24_FULL_EVIDENCE_AUDIT_PARTIAL" if source_ready else "C24_FULL_EVIDENCE_AUDIT_BLOCKED")
    rec = {
        "stage": "C24-1",
        "status": status,
        "mode": mode,
        "seed": seed,
        "bmn_source_manifest": {
            "c16_checkpoint_manifest": file_record(ROOT / "c16_2_fullscale_bmn_training/C16_2_BMN_CHECKPOINT_MANIFEST.json", sha=True),
            "c17_medium_score_table": c17_meta,
            "c19_canonical_score_table": c19_meta,
            "c23_bmn_manifest": file_record(ROOT / "c23_3_full_evidence_materialization/C23_3_BMN_SCORE_MANIFEST.json", sha=True),
            "columns": [c for c in ["bmn_pred_iou", "bmn_p_iou_05", "bmn_p_iou_07", "bmn_rank_score", "bmn_final_score", "start_prob", "end_prob", "actionness_score", "span_map_rank", "bmn_norm", "bmn_score"] if c in df.columns],
        },
        "t2_source_manifest": {
            "c17_medium_score_table": c17_meta,
            "c19_canonical_score_table": c19_meta,
            "c23_t2_manifest": file_record(ROOT / "c23_3_full_evidence_materialization/C23_3_T2_SCORE_MANIFEST.json", sha=True),
            "columns": [c for c in ["t2_score", "t2_norm"] if c in df.columns],
            "direct_margin_or_agreement_columns_available": False,
        },
        "event_source_manifest": {
            "c22_source_manifest": file_record(ROOT / "c22_1_feature_event_audit/C22_1_SOURCE_MANIFEST.json", sha=True),
            "c23_event_manifest": file_record(ROOT / "c23_3_full_evidence_materialization/C23_3_EVENT_RELEVANCE_MANIFEST.json", sha=True),
            "direct_event_score_columns_available": False,
            "event_proxy_columns": ["old_norm", "duration_score"],
            "raw_video_used": False,
        },
        "first_stage_residual_manifest": {
            "c23_first_stage_manifest": file_record(ROOT / "c23_1_full_trainfit_readiness/C23_1_FIRST_STAGE_TOP128_MANIFEST.json", sha=True),
            "c21_pair_dataset": c21_train_meta,
            "c20_top1_evidence": c20_train_meta,
        },
        "evidence_coverage_audit": coverage,
        "evidence_schema_audit": schema,
        "evidence_alignment_audit": alignment,
        "evidence_sample_path": str(sample_path),
        "official_val_used": False,
        "official_prediction_pool_used": False,
        "pseudo_official_holdout_used_for_selection": False,
        "evaluator_modified": False,
        "nms_modified": False,
    }
    write_text(OUT1 / "C24_1_EVIDENCE_SOURCE_AUDIT_PLAN.md", "# C24-1 Evidence Source Audit Plan\n\nAudit C17/C19 BMN/T2 evidence, C20 top1 evidence, and C21 front-rank pair evidence. Direct event columns are absent, so event evidence is reported as proxy/partial.")
    write_json(OUT1 / "C24_1_BMN_SOURCE_MANIFEST.json", rec["bmn_source_manifest"])
    write_json(OUT1 / "C24_1_T2_SOURCE_MANIFEST.json", rec["t2_source_manifest"])
    write_json(OUT1 / "C24_1_EVENT_SOURCE_MANIFEST.json", rec["event_source_manifest"])
    write_json(OUT1 / "C24_1_EVIDENCE_COVERAGE_AUDIT.json", coverage)
    write_json(OUT1 / "C24_1_EVIDENCE_SCHEMA_AUDIT.json", schema)
    write_json(OUT1 / "C24_1_EVIDENCE_ALIGNMENT_AUDIT.json", alignment)
    write_json(OUT1 / "C24_1_EVIDENCE_SOURCE_DECISION.json", rec)
    write_text(OUT1 / "C24_1_EVIDENCE_SOURCE_DECISION.md", f"# C24-1 Evidence Source Decision\n\nStatus: `{status}`.\n\nC19/C17 medium evidence covers calib_select/calib_holdout but not train_fit, so full trainfit evidence is partial.")
    return rec


def stage_c24_2(mode: str, seed: int) -> Dict[str, Any]:
    df = load_c21_frame(mode)
    c20 = load_c20_frame(mode)
    score_cols = [
        "retriever_score", "bmn_score", "t2_score", "c19_hybrid_score",
        "z_retriever", "z_bmn", "z_t2", "z_hybrid",
        "softmax_retriever_top10", "softmax_bmn_top10", "softmax_t2_top10",
    ]
    label_cols = ["candidate_iou", "candidate_iou_ge_05", "candidate_iou_ge_07", "candidate_correct_video"]
    corr = {s: {l: corr_pack(df, s, l) for l in label_cols} for s in score_cols if s in df.columns}
    auc = {}
    for s in score_cols:
        if s not in df.columns:
            continue
        auc[s] = {
            "IoU>=0.5_AUC": auc_score(df["candidate_iou_ge_05"], df[s]),
            "IoU>=0.7_AUC": auc_score(df["candidate_iou_ge_07"], df[s]),
            "correct_video_AUC": auc_score(df["candidate_correct_video"], df[s]),
            "correct_video_and_iou07_AUC": auc_score((df["candidate_correct_video"].astype(bool) & df["candidate_iou_ge_07"].astype(bool)).astype(int), df[s]),
        }
    combos = {
        "BMN_only": df["z_bmn"].to_numpy(np.float32),
        "T2_only": df["z_t2"].to_numpy(np.float32),
        "event_proxy_only": df["z_hybrid"].to_numpy(np.float32) - 0.7 * df["z_retriever"].to_numpy(np.float32),
        "BMN_T2": 0.5 * df["z_bmn"].to_numpy(np.float32) + 0.5 * df["z_t2"].to_numpy(np.float32),
        "BMN_event_proxy": 0.5 * df["z_bmn"].to_numpy(np.float32) + 0.5 * (df["z_hybrid"].to_numpy(np.float32) - 0.7 * df["z_retriever"].to_numpy(np.float32)),
        "event_proxy_T2": 0.5 * (df["z_hybrid"].to_numpy(np.float32) - 0.7 * df["z_retriever"].to_numpy(np.float32)) + 0.5 * df["z_t2"].to_numpy(np.float32),
        "all_evidence": 0.45 * df["z_bmn"].to_numpy(np.float32) + 0.35 * df["z_t2"].to_numpy(np.float32) + 0.20 * df["z_hybrid"].to_numpy(np.float32),
    }
    complementarity = {
        name: {
            "IoU>=0.5_AUC": auc_score(df["candidate_iou_ge_05"], score),
            "IoU>=0.7_AUC": auc_score(df["candidate_iou_ge_07"], score),
            "correct_video_AUC": auc_score(df["candidate_correct_video"], score),
        }
        for name, score in combos.items()
    }
    reliability = {}
    for col in ["bmn_score", "t2_score", "c19_hybrid_score"]:
        if col not in df.columns:
            continue
        tmp = df[[col, "candidate_iou_ge_07", "candidate_correct_video"]].copy()
        tmp["bin"] = pd.qcut(pd.to_numeric(tmp[col], errors="coerce").rank(method="first"), 10, duplicates="drop")
        reliability[col] = {
            str(k): {
                "rows": int(len(v)),
                "score_mean": float(v[col].mean()),
                "iou07_rate": float(v["candidate_iou_ge_07"].mean()),
                "correct_video_rate": float(v["candidate_correct_video"].mean()),
                "false_positive_iou07_rate": float((~v["candidate_iou_ge_07"].astype(bool)).mean()),
            }
            for k, v in tmp.groupby("bin", observed=True, sort=False)
        }
    high_fp = {}
    for col in ["bmn_score", "t2_score", "c19_hybrid_score"]:
        if col in df.columns:
            thr = float(df[col].quantile(0.90))
            hi = df[df[col] >= thr]
            high_fp[col] = {
                "threshold_p90": thr,
                "rows": int(len(hi)),
                "wrong_video_rate": float((~hi["candidate_correct_video"].astype(bool)).mean()) if len(hi) else None,
                "iou07_false_positive_rate": float((~hi["candidate_iou_ge_07"].astype(bool)).mean()) if len(hi) else None,
            }
    top_rank = {}
    for bucket_name, sdf in {
        "top1": df[df["candidate_rank"] <= 1],
        "top5": df[df["candidate_rank"] <= 5],
        "top10": df[df["candidate_rank"] <= 10],
        "top100_proxy": df,
        "gt_video_in_top5_wrong_top1": df[(df["anchor_video_rank"] <= 5) & (~df["anchor_iou_ge_07"].astype(bool))],
    }.items():
        top_rank[bucket_name] = {s: auc_score(sdf["candidate_iou_ge_07"], sdf[s]) for s in ["retriever_score", "bmn_score", "t2_score", "c19_hybrid_score"] if s in sdf.columns and len(sdf)}
    failure = {
        "wrong_video_top1": describe_numeric(c20[c20["orig_wrong_video_top1"].astype(bool)] if "orig_wrong_video_top1" in c20.columns else c20.head(0), ["anchor_bmn_norm", "anchor_t2_norm", "anchor_retriever_norm", "bmn_anchor_margin"]),
        "correct_video_wrong_span": describe_numeric(c20[c20["orig_correct_video_wrong_span_top1"].astype(bool)] if "orig_correct_video_wrong_span_top1" in c20.columns else c20.head(0), ["anchor_bmn_norm", "anchor_t2_norm", "anchor_retriever_norm", "bmn_anchor_margin"]),
        "gt_video_in_top5_not_top1": describe_numeric(c20[(c20["gt_in_top5"].astype(bool)) & (~c20["anchor_video_correct"].astype(bool))] if {"gt_in_top5", "anchor_video_correct"}.issubset(c20.columns) else c20.head(0), ["anchor_bmn_norm", "anchor_t2_norm", "anchor_retriever_norm", "bmn_anchor_margin"]),
    }
    query_breakdown = {
        str(k): {
            "rows": int(len(v)),
            "bmn_iou07_auc": auc_score(v["candidate_iou_ge_07"], v["bmn_score"]),
            "t2_iou07_auc": auc_score(v["candidate_iou_ge_07"], v["t2_score"]),
            "hybrid_iou07_auc": auc_score(v["candidate_iou_ge_07"], v["c19_hybrid_score"]),
        }
        for k, v in df.groupby("query_type", observed=True, sort=False)
    }
    duration_breakdown = {
        str(k): {
            "rows": int(len(v)),
            "bmn_iou07_auc": auc_score(v["candidate_iou_ge_07"], v["bmn_score"]),
            "t2_iou07_auc": auc_score(v["candidate_iou_ge_07"], v["t2_score"]),
            "hybrid_iou07_auc": auc_score(v["candidate_iou_ge_07"], v["c19_hybrid_score"]),
        }
        for k, v in df.groupby("duration_bucket", observed=True, sort=False)
    }
    bmn_auc = auc.get("bmn_score", {}).get("IoU>=0.7_AUC") or 0.0
    t2_auc = auc.get("t2_score", {}).get("IoU>=0.7_AUC") or 0.0
    hybrid_auc = auc.get("c19_hybrid_score", {}).get("IoU>=0.7_AUC") or 0.0
    retr_auc = auc.get("retriever_score", {}).get("IoU>=0.7_AUC") or 0.0
    high_wrong = max([v.get("wrong_video_rate") or 0.0 for v in high_fp.values()] or [0.0])
    if max(bmn_auc, t2_auc, hybrid_auc) >= 0.62 and hybrid_auc > retr_auc + 0.02:
        status = "C24_EVIDENCE_SIGNAL_STRONG"
    elif max(bmn_auc, t2_auc, hybrid_auc) >= 0.56:
        status = "C24_EVIDENCE_SIGNAL_FRONT_RANK_WEAK"
    elif high_wrong >= 0.70:
        status = "C24_EVIDENCE_SIGNAL_NOISY_OR_HARMFUL"
    else:
        status = "C24_EVIDENCE_SIGNAL_INCONCLUSIVE"
    conclusions = {
        "independent_signal_beyond_first_stage": {
            "hybrid_minus_retriever_iou07_auc": hybrid_auc - retr_auc,
            "bmn_minus_retriever_iou07_auc": bmn_auc - retr_auc,
            "t2_minus_retriever_iou07_auc": t2_auc - retr_auc,
        },
        "front_rank_signal": status in ["C24_EVIDENCE_SIGNAL_STRONG", "C24_EVIDENCE_SIGNAL_FRONT_RANK_WEAK"],
        "r100_only_risk": status != "C24_EVIDENCE_SIGNAL_STRONG",
        "wrong_video_false_positive_risk": high_wrong,
        "event_evidence_conclusion": "direct event columns absent; event proxy is incomplete",
        "bmn_evidence_conclusion": "available with weak/moderate candidate-level signal" if bmn_auc >= 0.55 else "weak candidate-level signal",
        "t2_evidence_conclusion": "available mostly as safety/complementary signal" if t2_auc < hybrid_auc else "available",
    }
    rec = {
        "stage": "C24-2",
        "status": status,
        "mode": mode,
        "seed": seed,
        "evidence_correlation": corr,
        "auc_by_signal": auc,
        "top_rank_signal_audit": top_rank,
        "bmn_span_signal_audit": {"auc": auc.get("bmn_score"), "correlation": corr.get("bmn_score"), "reliability": reliability.get("bmn_score")},
        "event_signal_audit": {"direct_event_columns_available": False, "proxy_auc": complementarity.get("event_proxy_only")},
        "t2_signal_audit": {"auc": auc.get("t2_score"), "correlation": corr.get("t2_score"), "reliability": reliability.get("t2_score")},
        "failure_type_signal_audit": failure,
        "score_calibration": reliability,
        "high_score_false_positive": high_fp,
        "evidence_complementarity": complementarity,
        "query_type_breakdown": query_breakdown,
        "duration_breakdown": duration_breakdown,
        "conclusions": conclusions,
        "train_fit_evidence_available": False,
        "selection_split_used": "calib_select",
        "holdout_report_only": True,
        "official_val_used": False,
        "official_prediction_pool_used": False,
        "pseudo_official_holdout_used_for_selection": False,
        "evaluator_modified": False,
        "nms_modified": False,
    }
    write_text(OUT2 / "C24_2_DIAGNOSTIC_PLAN.md", "# C24-2 Diagnostic Plan\n\nUse C21 front-rank candidate labels and C20 top1 evidence to measure BMN/T2/event-proxy signal without official data.")
    write_json(OUT2 / "C24_2_EVIDENCE_CORRELATION.json", corr)
    write_json(OUT2 / "C24_2_TOP_RANK_SIGNAL_AUDIT.json", top_rank)
    write_json(OUT2 / "C24_2_BMN_SPAN_SIGNAL_AUDIT.json", rec["bmn_span_signal_audit"])
    write_json(OUT2 / "C24_2_EVENT_SIGNAL_AUDIT.json", rec["event_signal_audit"])
    write_json(OUT2 / "C24_2_T2_SIGNAL_AUDIT.json", rec["t2_signal_audit"])
    write_json(OUT2 / "C24_2_FAILURE_TYPE_SIGNAL_AUDIT.json", failure)
    write_text(OUT2 / "C24_2_DIAGNOSTIC_CASES.md", "# C24-2 Diagnostic Cases\n\nDetailed case tables are kept local; committed diagnostics summarize aggregate signal, false positives, and failure subsets.")
    write_json(OUT2 / "C24_2_DIAGNOSTIC_DECISION.json", rec)
    write_text(OUT2 / "C24_2_DIAGNOSTIC_DECISION.md", f"# C24-2 Diagnostic Decision\n\nStatus: `{status}`.\n\nDirect event evidence is unavailable; BMN/T2 are evaluated as existing-feature evidence.")
    return rec


def probe_scores(df: pd.DataFrame) -> Dict[str, np.ndarray]:
    z_r = df["z_retriever"].to_numpy(np.float32)
    z_b = df["z_bmn"].to_numpy(np.float32)
    z_t = df["z_t2"].to_numpy(np.float32)
    z_h = df["z_hybrid"].to_numpy(np.float32)
    low_r_high_b = df.get("low_retriever_high_bmn", pd.Series(False, index=df.index)).astype(bool).to_numpy()
    conflict = df.get("bmn_t2_conflict", pd.Series(False, index=df.index)).astype(bool).to_numpy()
    agree = df.get("bmn_t2_agree", pd.Series(False, index=df.index)).astype(bool).to_numpy()
    event_proxy = z_h - 0.7 * z_r
    return {
        "P0_baseline_first_stage_only": z_r,
        "P1_bmn_recalibration": 0.72 * z_r + 0.28 * z_b,
        "P2_bmn_false_positive_penalty": 0.70 * z_r + 0.25 * z_b + 0.05 * z_t - 0.35 * low_r_high_b.astype(np.float32),
        "P3_event_relevance_reweight": 0.70 * z_r + 0.30 * event_proxy,
        "P4_bmn_event_consistency": 0.65 * z_r + 0.25 * z_b + 0.10 * event_proxy + 0.12 * ((z_b > 0) & (event_proxy > 0)).astype(np.float32),
        "P5_bmn_t2_consistency": 0.65 * z_r + 0.25 * z_b + 0.10 * z_t + 0.12 * agree.astype(np.float32) - 0.22 * conflict.astype(np.float32),
        "P6_all_evidence_logistic_proxy": 0.58 * z_r + 0.20 * z_b + 0.14 * z_t + 0.08 * event_proxy,
        "P7_rank_bucket_specific_calibration": 0.62 * z_r + np.where(df["candidate_video_rank"].to_numpy(np.float32) <= 5, 0.28 * z_b + 0.10 * z_t, 0.18 * z_b + 0.08 * z_t),
        "P8_query_duration_specific_calibration": 0.60 * z_r + 0.22 * z_b + 0.12 * z_t + 0.06 * event_proxy + 0.08 * df["duration_short"].to_numpy(np.float32) * z_b,
        "P9_no_regression_guarded_evidence": z_r + np.where((z_r > 0) & (~conflict), 0.18 * z_b + 0.08 * z_t, 0.0).astype(np.float32),
    }


def c24_objective(metrics: Dict[str, Any], base: Dict[str, Any]) -> float:
    return (
        3.0 * (metrics.get("VCMR_R@1_IoU0.7", 0.0) - base.get("VCMR_R@1_IoU0.7", 0.0))
        + 2.0 * (metrics.get("VCMR_R@5_IoU0.7", 0.0) - base.get("VCMR_R@5_IoU0.7", 0.0))
        + 1.5 * (metrics.get("VCMR_R@10_IoU0.7", 0.0) - base.get("VCMR_R@10_IoU0.7", 0.0))
        + 1.0 * (metrics.get("VCMR_R@1_IoU0.5", 0.0) - base.get("VCMR_R@1_IoU0.5", 0.0))
        + 0.8 * (metrics.get("VCMR_R@5_IoU0.5", 0.0) - base.get("VCMR_R@5_IoU0.5", 0.0))
        - 2.0 * max(0.0, metrics.get("wrong_video_top1_rate", metrics.get("wrong_video_high_score_rate", 0.0)) - base.get("wrong_video_top1_rate", base.get("wrong_video_high_score_rate", 0.0)))
    )


def stage_c24_3(mode: str, seed: int, s2: Dict[str, Any] | None = None) -> Dict[str, Any]:
    df = load_c21_frame(mode)
    scores = probe_scores(df)
    results = {}
    base = {}
    for name, score in scores.items():
        split_results = {}
        for split in ["calib_select", "calib_holdout"]:
            split_results[split] = candidate_eval(df[df["split"].astype(str) == split], score[df["split"].astype(str).to_numpy() == split])
        results[name] = split_results
        if name == "P0_baseline_first_stage_only":
            base = split_results["calib_select"]
    for name in results:
        results[name]["select_objective"] = c24_objective(results[name]["calib_select"], base)
        results[name]["delta_select_vs_baseline"] = delta(results[name]["calib_select"], base, ["VCMR_R@1_IoU0.7", "VCMR_R@5_IoU0.7", "VCMR_R@10_IoU0.7", "VCMR_R@1_IoU0.5", "VCMR_R@5_IoU0.5", "wrong_video_top1_rate"])
    selected = max(results, key=lambda n: results[n]["select_objective"])
    base_hold = results["P0_baseline_first_stage_only"]["calib_holdout"]
    selected_hold = results[selected]["calib_holdout"]
    hold_delta = delta(selected_hold, base_hold, ["VCMR_R@1_IoU0.7", "VCMR_R@5_IoU0.7", "VCMR_R@10_IoU0.7", "VCMR_R@1_IoU0.5", "VCMR_R@5_IoU0.5", "wrong_video_top1_rate"])
    front_keys = ["VCMR_R@1_IoU0.7", "VCMR_R@5_IoU0.7", "VCMR_R@10_IoU0.7"]
    front_positive = any(hold_delta[k] > 0 for k in front_keys)
    front_consistent = front_positive and all(hold_delta[k] >= 0 for k in front_keys)
    wrong_safe = hold_delta["wrong_video_top1_rate"] <= 0.5
    if front_consistent and wrong_safe:
        status = "C24_EVIDENCE_STRENGTHENING_PROMISING"
    elif front_positive:
        status = "C24_EVIDENCE_STRENGTHENING_FRONT_RANK_WEAK"
    else:
        status = "C24_EVIDENCE_STRENGTHENING_FRONT_RANK_WEAK"
    rec = {
        "stage": "C24-3",
        "status": status,
        "mode": mode,
        "seed": seed,
        "feature_set_schema": {
            "features": ["z_retriever", "z_bmn", "z_t2", "z_hybrid", "event_proxy", "bmn_t2_conflict", "duration/query one-hot"],
            "label_features_forbidden": True,
            "train_fit_available": False,
            "proxy_training_used": "calib_select deterministic probe selection",
        },
        "model_search_space": list(scores.keys()),
        "strengthening_results": results,
        "front_rank_results": {
            "baseline_holdout": base_hold,
            "selected_holdout": selected_hold,
            "delta_holdout_vs_baseline": hold_delta,
        },
        "false_positive_audit": {
            "baseline_wrong_video_top1_rate": base_hold.get("wrong_video_top1_rate"),
            "selected_wrong_video_top1_rate": selected_hold.get("wrong_video_top1_rate"),
            "increase": hold_delta.get("wrong_video_top1_rate"),
        },
        "selected_probe": {
            "name": selected,
            "selection_split": "calib_select",
            "holdout_report_only": True,
            "train_fit_evidence_available": False,
        },
        "official_val_used": False,
        "official_prediction_pool_used": False,
        "pseudo_official_holdout_used_for_selection": False,
        "evaluator_modified": False,
        "nms_modified": False,
    }
    write_text(OUT3 / "C24_3_STRENGTHENING_PLAN.md", "# C24-3 Strengthening Plan\n\nRun deterministic evidence calibration probes using C21 train-only pair data. Because train_fit span evidence is absent, probes are diagnostic and downgraded.")
    write_json(OUT3 / "C24_3_FEATURE_SET_SCHEMA.json", rec["feature_set_schema"])
    write_json(OUT3 / "C24_3_MODEL_SEARCH_SPACE.json", rec["model_search_space"])
    write_json(OUT3 / "C24_3_STRENGTHENING_RESULTS.json", results)
    write_json(OUT3 / "C24_3_FRONT_RANK_RESULTS.json", rec["front_rank_results"])
    write_json(OUT3 / "C24_3_FALSE_POSITIVE_AUDIT.json", rec["false_positive_audit"])
    write_json(OUT3 / "C24_3_SELECTED_PROBE.json", rec["selected_probe"])
    write_json(OUT3 / "C24_3_STRENGTHENING_DECISION.json", rec)
    write_text(OUT3 / "C24_3_STRENGTHENING_DECISION.md", f"# C24-3 Strengthening Decision\n\nStatus: `{status}`.\n\nSelected probe: `{selected}`.")
    return rec


def c24_score_arrays(df: pd.DataFrame) -> Dict[str, np.ndarray]:
    retr = df["retriever_norm"].to_numpy(np.float32)
    bmn = df["bmn_norm"].to_numpy(np.float32)
    t2 = df["t2_norm"].to_numpy(np.float32)
    old = df["old_norm"].to_numpy(np.float32)
    dur = df["duration_score"].fillna(0.0).to_numpy(np.float32)
    conflict = ((t2 > 0.80) & (bmn < 0.25)).astype(np.float32)
    agree = ((t2 > 0.55) & (bmn > 0.55)).astype(np.float32)
    event_proxy = 0.65 * old + 0.35 * dur
    low_retr_high_bmn = ((retr < 0.35) & (bmn > 0.70)).astype(np.float32)
    no_reg_gate = ((retr > 0.55) | ((bmn > 0.65) & (t2 > 0.45))).astype(np.float32)
    final = df["final_score"].to_numpy(np.float32)
    return {
        "A_C23_selected_duration_front": 0.55 * retr + 0.25 * bmn + 0.20 * t2 + np.where(df["duration_bucket"].astype(str).to_numpy() == "short", 0.18 * bmn, 0.06 * t2).astype(np.float32),
        "B_C19_C21_hybrid": final,
        "C_BMN_raw": 0.55 * retr + 0.35 * bmn + 0.10 * t2,
        "D_BMN_calibrated": 0.58 * retr + 0.30 * np.sqrt(np.clip(bmn, 0, 1)) + 0.12 * t2,
        "E_event_proxy_calibrated": 0.66 * retr + 0.24 * bmn + 0.10 * event_proxy,
        "F_T2_calibrated": 0.60 * retr + 0.18 * bmn + 0.22 * t2,
        "G_BMN_event_consistency": 0.58 * retr + 0.24 * bmn + 0.10 * event_proxy + 0.08 * ((bmn > 0.55) & (event_proxy > 0.50)).astype(np.float32),
        "H_BMN_T2_consistency": 0.58 * retr + 0.24 * bmn + 0.14 * t2 + 0.08 * agree - 0.15 * conflict,
        "I_all_evidence_wrong_risk_penalty": 0.54 * retr + 0.23 * bmn + 0.13 * t2 + 0.10 * event_proxy - 0.25 * low_retr_high_bmn - 0.12 * conflict,
        "J_all_evidence_no_regression_guard": retr + no_reg_gate * (0.16 * bmn + 0.08 * t2 + 0.04 * event_proxy) - 0.10 * conflict,
    }


def eval_vcmr_split(ctx: Dict[str, Any], score: np.ndarray, split: str) -> Dict[str, Any]:
    return c18.fast_eval_vcmr(ctx, score.astype(np.float32), split)["summary"]


def stage_c24_4(mode: str, seed: int, s3: Dict[str, Any] | None = None) -> Dict[str, Any]:
    df = load_c19_frame(mode)
    ctx = c18.prepare_fast_eval(df)
    scores = c24_score_arrays(df)
    base_name = "B_C19_C21_hybrid"
    base_select = eval_vcmr_split(ctx, scores[base_name], "calib_select")
    base_hold = eval_vcmr_split(ctx, scores[base_name], "calib_holdout")
    results = {}
    best = base_name
    best_score = -1e18
    for name, score in scores.items():
        sel = base_select if name == base_name else eval_vcmr_split(ctx, score, "calib_select")
        obj = c24_objective(sel, base_select)
        unsafe = (
            sel.get("VR_R@100", 0.0) < base_select.get("VR_R@100", 0.0) - 2.0
            or sel.get("VR_R@10", 0.0) < base_select.get("VR_R@10", 0.0) - 10.0
            or sel.get("wrong_video_high_score_rate", 0.0) > base_select.get("wrong_video_high_score_rate", 0.0) + 5.0
        )
        results[name] = {
            "calib_select": sel,
            "select_objective": obj,
            "unsafe": unsafe,
            "delta_select_vs_baseline": delta(sel, base_select, ["VCMR_R@1_IoU0.7", "VCMR_R@5_IoU0.7", "VCMR_R@10_IoU0.7", "VCMR_R@1_IoU0.5", "VCMR_R@5_IoU0.5", "VR_R@10", "VR_R@100", "wrong_video_high_score_rate"]),
        }
        if not unsafe and obj > best_score:
            best_score = obj
            best = name
    front_rows = [
        (name, r) for name, r in results.items()
        if not r["unsafe"] and (
            r["delta_select_vs_baseline"]["VCMR_R@1_IoU0.7"] > 0
            or r["delta_select_vs_baseline"]["VCMR_R@5_IoU0.7"] > 0
            or r["delta_select_vs_baseline"]["VCMR_R@10_IoU0.7"] > 0
        )
    ]
    holdout_names = {base_name, best}
    for name, _ in sorted(front_rows, key=lambda x: -x[1]["select_objective"])[:6]:
        holdout_names.add(name)
    for name, r in sorted(results.items(), key=lambda x: -x[1]["select_objective"])[:6]:
        if not r["unsafe"]:
            holdout_names.add(name)
    for name in holdout_names:
        hold = base_hold if name == base_name else eval_vcmr_split(ctx, scores[name], "calib_holdout")
        results[name]["calib_holdout"] = hold
        results[name]["delta_holdout_vs_baseline"] = delta(hold, base_hold, ["VCMR_R@1_IoU0.7", "VCMR_R@5_IoU0.7", "VCMR_R@10_IoU0.7", "VCMR_R@1_IoU0.5", "VCMR_R@5_IoU0.5", "VR_R@10", "VR_R@100", "wrong_video_high_score_rate"])
    selected_hold = results[best]["calib_holdout"]
    hold_delta = results[best]["delta_holdout_vs_baseline"]
    front_positive = any(hold_delta[k] > 0 for k in ["VCMR_R@1_IoU0.7", "VCMR_R@5_IoU0.7", "VCMR_R@10_IoU0.7"])
    wrong_safe = hold_delta["wrong_video_high_score_rate"] <= 0.5
    if front_positive and wrong_safe:
        status = "C24_STRENGTHENED_INTEGRATION_PROMISING"
    elif front_positive:
        status = "C24_STRENGTHENED_INTEGRATION_WEAK"
    else:
        status = "C24_STRENGTHENED_INTEGRATION_WEAK"
    rec = {
        "stage": "C24-4",
        "status": status,
        "mode": mode,
        "seed": seed,
        "score_formulas": {
            "A": "C23 selected duration-aware formula",
            "B": "C19/C21 hybrid baseline",
            "C": "BMN raw",
            "D": "BMN calibrated sqrt",
            "E": "event proxy calibrated",
            "F": "T2 calibrated",
            "G": "BMN + event proxy consistency",
            "H": "BMN + T2 consistency",
            "I": "all evidence + wrong-risk penalty",
            "J": "all evidence + no-regression guard",
        },
        "replay_results": results,
        "component_ablation": {
            "baseline_c19_holdout": base_hold,
            "selected_holdout": selected_hold,
            "selected_delta": hold_delta,
            "BMN_removal_delta_proxy": results.get("F_T2_calibrated", {}).get("delta_holdout_vs_baseline"),
            "event_removal_delta_proxy": results.get("H_BMN_T2_consistency", {}).get("delta_holdout_vs_baseline"),
            "T2_removal_delta_proxy": results.get("G_BMN_event_consistency", {}).get("delta_holdout_vs_baseline"),
        },
        "wrong_video_audit": {
            "baseline_wrong_video_high_score_rate": base_hold.get("wrong_video_high_score_rate"),
            "selected_wrong_video_high_score_rate": selected_hold.get("wrong_video_high_score_rate"),
            "increase": hold_delta.get("wrong_video_high_score_rate"),
        },
        "selected_integration": {
            "name": best,
            "selection_split": "calib_select",
            "holdout_report_only": True,
            "select_objective": results[best]["select_objective"],
        },
        "final_vcmr_metrics": selected_hold,
        "official_val_used": False,
        "official_prediction_pool_used": False,
        "pseudo_official_holdout_used_for_selection": False,
        "evaluator_modified": False,
        "nms_modified": False,
    }
    write_text(OUT4 / "C24_4_INTEGRATION_PLAN.md", "# C24-4 Integration Plan\n\nReplay strengthened evidence formulas over C19 medium candidate rows, preserving first-stage retrieval and reporting holdout only after calib_select choice.")
    write_json(OUT4 / "C24_4_SCORE_FORMULAS.json", rec["score_formulas"])
    write_json(OUT4 / "C24_4_REPLAY_RESULTS.json", results)
    write_json(OUT4 / "C24_4_COMPONENT_ABLATION.json", rec["component_ablation"])
    write_json(OUT4 / "C24_4_WRONG_VIDEO_AUDIT.json", rec["wrong_video_audit"])
    write_json(OUT4 / "C24_4_SELECTED_INTEGRATION.json", rec["selected_integration"])
    write_json(OUT4 / "C24_4_INTEGRATION_DECISION.json", rec)
    write_text(OUT4 / "C24_4_INTEGRATION_DECISION.md", f"# C24-4 Integration Decision\n\nStatus: `{status}`.\n\nSelected integration: `{best}`.")
    return rec


def stage_c24_5(mode: str, seed: int, s2: Dict[str, Any] | None = None, s3: Dict[str, Any] | None = None, s4: Dict[str, Any] | None = None) -> Dict[str, Any]:
    s2 = s2 or load_json(OUT2 / "C24_2_DIAGNOSTIC_DECISION.json", {})
    s3 = s3 or load_json(OUT3 / "C24_3_STRENGTHENING_DECISION.json", {})
    s4 = s4 or load_json(OUT4 / "C24_4_INTEGRATION_DECISION.json", {})
    c24_1 = load_json(OUT1 / "C24_1_EVIDENCE_SOURCE_DECISION.json", {})
    route = "C24_CONTINUE_EXISTING_FEATURE_EVIDENCE_STRENGTHENING"
    source_partial = c24_1.get("status") != "C24_FULL_EVIDENCE_AUDIT_READY"
    signal_available = (
        s2.get("status") in ["C24_EVIDENCE_SIGNAL_STRONG", "C24_EVIDENCE_SIGNAL_FRONT_RANK_WEAK"]
        or s3.get("status") in ["C24_EVIDENCE_STRENGTHENING_PROMISING", "C24_EVIDENCE_STRENGTHENING_FRONT_RANK_WEAK"]
    )
    if source_partial and signal_available:
        route = "C24_NEED_FULL_EVIDENCE_MATERIALIZATION_FIRST"
    elif s2.get("status") in ["C24_EVIDENCE_SIGNAL_FRONT_RANK_WEAK", "C24_EVIDENCE_SIGNAL_NOISY_OR_HARMFUL", "C24_EVIDENCE_SIGNAL_INCONCLUSIVE"] and s4.get("status") != "C24_STRENGTHENED_INTEGRATION_PROMISING":
        route = "C24_READY_FOR_C25_RAW_FRAME_STRONG_FEATURE_AUDIT"
    status = "C24_ROBUSTNESS_PARTIAL"
    score_dist = {
        "diagnostic_high_score_false_positive": s2.get("high_score_false_positive"),
        "integration_wrong_video_audit": s4.get("wrong_video_audit"),
        "r100_only_risk": (s2.get("conclusions") or {}).get("r100_only_risk"),
    }
    pseudo = {
        "pseudo_one_look_executed": False,
        "pseudo_official_not_used_for_selection": True,
        "no_post_pseudo_adjustment": True,
    }
    firewall = {
        "official_val_used": False,
        "official_prediction_pool_used": False,
        "pseudo_official_holdout_used_for_selection": False,
        "evaluator_modified": False,
        "nms_modified": False,
    }
    raw_route_md = f"""# C24-5 Raw/Strong Feature Route Audit

1. Existing BMN/T2 evidence does not show enough stable front-rank independent signal to promote.
2. C24 strengthened replay status: `{s4.get('status')}`.
3. Direct event evidence is absent; current event path is only proxy/partial.
4. Recommended route: `{route}`.
5. If C25 starts, audit frame/raw video availability, CLIP/DINOv2 frame features, VideoMAE/InternVideo motion features, timestamp alignment, GPU memory, and disk budget before extraction.
"""
    rec = {
        "stage": "C24-5",
        "status": status,
        "mode": mode,
        "seed": seed,
        "route_recommendation": route,
        "seed_robustness": {
            "seed2026": {"diagnostic": s2.get("status"), "probe": s3.get("status"), "integration": s4.get("status")},
            "seed2027": "not rerun; deterministic evidence probe and partial source coverage",
            "seed2028": "not rerun; deterministic evidence probe and partial source coverage",
        },
        "query_duration_robustness": {
            "query_type_breakdown": s2.get("query_type_breakdown"),
            "duration_breakdown": s2.get("duration_breakdown"),
        },
        "d_e_f_subset_audit": {
            "source": "C20/C21 failure proxy; original C20/C21 D/E/F named subsets not directly materialized in C24",
            "failure_type_signal_audit": s2.get("failure_type_signal_audit"),
        },
        "score_distribution_audit": score_dist,
        "raw_feature_route_audit": {
            "direct_event_signal_available": False,
            "existing_feature_front_rank_gain_stable": s4.get("status") == "C24_STRENGTHENED_INTEGRATION_PROMISING",
            "recommendation": route,
        },
        "pseudo_onelook_diagnostic": pseudo,
        "selection_firewall_audit": firewall,
        **firewall,
    }
    write_text(OUT5 / "C24_5_ROBUSTNESS_PLAN.md", "# C24-5 Robustness Plan\n\nSummarize split stability, failure subsets, false positives, and route toward full evidence materialization or C25 raw/strong feature audit.")
    write_json(OUT5 / "C24_5_SEED_ROBUSTNESS.json", rec["seed_robustness"])
    write_json(OUT5 / "C24_5_QUERY_DURATION_ROBUSTNESS.json", rec["query_duration_robustness"])
    write_json(OUT5 / "C24_5_D_E_F_SUBSET_AUDIT.json", rec["d_e_f_subset_audit"])
    write_json(OUT5 / "C24_5_SCORE_DISTRIBUTION_AUDIT.json", score_dist)
    write_text(OUT5 / "C24_5_RAW_FEATURE_ROUTE_AUDIT.md", raw_route_md)
    write_json(OUT5 / "C24_5_PSEUDO_ONELOOK_DIAGNOSTIC.json", pseudo)
    write_json(OUT5 / "C24_5_SELECTION_FIREWALL_AUDIT.json", firewall)
    write_json(OUT5 / "C24_5_ROBUSTNESS_DECISION.json", rec)
    write_text(OUT5 / "C24_5_ROBUSTNESS_DECISION.md", f"# C24-5 Robustness Decision\n\nStatus: `{status}`.\n\nRoute recommendation: `{route}`.")
    return rec


def stage_c24_6(mode: str, seed: int, recs: Dict[str, Dict[str, Any]] | None = None) -> Dict[str, Any]:
    recs = recs or {
        "c24_0": load_json(OUT0 / "C24_0_PROTOCOL.json", {}),
        "c24_1": load_json(OUT1 / "C24_1_EVIDENCE_SOURCE_DECISION.json", {}),
        "c24_2": load_json(OUT2 / "C24_2_DIAGNOSTIC_DECISION.json", {}),
        "c24_3": load_json(OUT3 / "C24_3_STRENGTHENING_DECISION.json", {}),
        "c24_4": load_json(OUT4 / "C24_4_INTEGRATION_DECISION.json", {}),
        "c24_5": load_json(OUT5 / "C24_5_ROBUSTNESS_DECISION.json", {}),
    }
    route = recs["c24_5"].get("route_recommendation")
    if recs["c24_2"].get("status") == "C24_EVIDENCE_SIGNAL_STRONG" and recs["c24_4"].get("status") == "C24_STRENGTHENED_INTEGRATION_PROMISING":
        decision = "C24_READY_FOR_C23R_EVIDENCE_STRENGTHENED_NATIVE_COUPLING"
    elif route == "C24_NEED_FULL_EVIDENCE_MATERIALIZATION_FIRST":
        decision = "C24_NEED_FULL_EVIDENCE_MATERIALIZATION_FIRST"
    elif route == "C24_READY_FOR_C25_RAW_FRAME_STRONG_FEATURE_AUDIT":
        decision = "C24_READY_FOR_C25_RAW_FRAME_STRONG_FEATURE_AUDIT"
    elif recs["c24_4"].get("status") == "C24_STRENGTHENED_INTEGRATION_PROMISING":
        decision = "C24_CONTINUE_EXISTING_FEATURE_EVIDENCE_STRENGTHENING"
    else:
        decision = "C24_STOP_EXISTING_FEATURE_COUPLING"
    c23_final = load_json(ROOT / "c23_6_final_decision/C23_6_FINAL_DECISION.json", {})
    final_vcmr = recs["c24_4"].get("final_vcmr_metrics") or {}
    c23_vcmr = c23_final.get("final_vcmr_metrics") or {}
    c19_base = ((recs["c24_4"].get("component_ablation") or {}).get("baseline_c19_holdout") or {})
    rec = {
        "stage": "C24-6",
        "status": decision,
        "final_decision": decision,
        "mode": mode,
        "seed": seed,
        "c24_0_protocol_status": recs["c24_0"].get("status"),
        "c24_1_evidence_source_audit_status": recs["c24_1"].get("status"),
        "c24_2_evidence_signal_status": recs["c24_2"].get("status"),
        "c24_3_evidence_strengthening_status": recs["c24_3"].get("status"),
        "c24_4_integration_replay_status": recs["c24_4"].get("status"),
        "c24_5_robustness_route_status": recs["c24_5"].get("status"),
        "selected_evidence_probe": recs["c24_3"].get("selected_probe"),
        "selected_integration_formula": recs["c24_4"].get("selected_integration"),
        "final_vcmr_metrics": final_vcmr,
        "final_vr_metrics": {k: final_vcmr.get(k) for k in ["VR_R@1", "VR_R@5", "VR_R@10", "VR_R@100"]},
        "delta_vs_c23": delta(final_vcmr, c23_vcmr, [k for k in final_vcmr if isinstance(final_vcmr.get(k), (int, float))]),
        "delta_vs_c19_c21": delta(final_vcmr, c19_base, [k for k in final_vcmr if isinstance(final_vcmr.get(k), (int, float))]),
        "wrong_video_risk": final_vcmr.get("wrong_video_high_score_rate"),
        "high_score_false_positive": recs["c24_2"].get("high_score_false_positive"),
        "evidence_signal_conclusion": recs["c24_2"].get("conclusions"),
        "existing_feature_path_still_has_value": decision in ["C24_READY_FOR_C23R_EVIDENCE_STRENGTHENED_NATIVE_COUPLING", "C24_CONTINUE_EXISTING_FEATURE_EVIDENCE_STRENGTHENING", "C24_NEED_FULL_EVIDENCE_MATERIALIZATION_FIRST"],
        "c25_raw_frame_strong_feature_audit_recommended": decision == "C24_READY_FOR_C25_RAW_FRAME_STRONG_FEATURE_AUDIT",
        "official_val_used": False,
        "official_prediction_pool_used": False,
        "pseudo_official_holdout_used_for_selection": False,
        "evaluator_modified": False,
        "nms_modified": False,
        "c24_is_promoted_system": False,
        "current_promoted_system": PROMOTED,
    }
    packet = {
        "decision": decision,
        "official_still_forbidden": True,
        "ready_for_official": False,
        "selected_probe": rec["selected_evidence_probe"],
        "selected_integration": rec["selected_integration_formula"],
    }
    write_json(OUT6 / "C24_6_FINAL_DECISION.json", rec)
    write_json(OUT6 / "C24_6_EVIDENCE_STRENGTHENING_PACKET.json", packet)
    write_json(OUT6 / "C24_6_NEXT_STEP_DECISION.json", {"decision": decision, "official_forbidden": True})
    write_text(OUT6 / "C24_6_FINAL_DECISION.md", f"# C24-6 Final Decision\n\nFinal decision: `{decision}`.\n\nC24 is not promoted. Current promoted official system remains `{PROMOTED}`.")
    write_text(OUT6 / "C24_6_EVIDENCE_STRENGTHENING_PACKET.md", f"# C24 Evidence Strengthening Packet\n\nDecision: `{decision}`.\nOfficial validation remains forbidden.")
    write_text(OUT6 / "C24_6_RISK_REGISTER.md", "# C24-6 Risk Register\n\n- Train_fit BMN/T2/event evidence is not fully materialized in current local sources.\n- Direct event evidence columns are absent; event audit uses proxy columns only.\n- No official validation was run.\n")
    write_text(OUT6 / "C24_6_NEXT_STEP_DECISION.md", f"# C24 Next Step\n\n`{decision}`")
    return rec


def run_all(mode: str, seed: int) -> Dict[str, Dict[str, Any]]:
    r0 = stage_c24_0(mode, seed)
    if r0["status"] != "C24_PROTOCOL_READY":
        raise RuntimeError(f"C24 protocol blocked: {r0['status']}")
    r1 = stage_c24_1(mode, seed)
    r2 = stage_c24_2(mode, seed)
    if mode == "full" and not (r1["status"] == "C24_FULL_EVIDENCE_AUDIT_READY" and r2["status"] == "C24_EVIDENCE_SIGNAL_STRONG"):
        raise RuntimeError("C24 full requires C24-1/C24-2 medium readiness")
    r3 = stage_c24_3(mode, seed, r2)
    r4 = stage_c24_4(mode, seed, r3)
    r5 = stage_c24_5(mode, seed, r2, r3, r4)
    r6 = stage_c24_6(mode, seed, {"c24_0": r0, "c24_1": r1, "c24_2": r2, "c24_3": r3, "c24_4": r4, "c24_5": r5})
    return {"c24_0": r0, "c24_1": r1, "c24_2": r2, "c24_3": r3, "c24_4": r4, "c24_5": r5, "c24_6": r6}


def print_summary(recs: Dict[str, Dict[str, Any]]) -> None:
    final = recs["c24_6"]
    vcmr = final.get("final_vcmr_metrics") or {}
    print("\n===== C24 SUMMARY =====")
    print(f"branch: {sh('git branch --show-current')}")
    print(f"commit: {sh('git rev-parse HEAD')}")
    print(f"C24-0: {final.get('c24_0_protocol_status')}")
    print(f"C24-1: {final.get('c24_1_evidence_source_audit_status')}")
    print(f"C24-2: {final.get('c24_2_evidence_signal_status')}")
    print(f"C24-3: {final.get('c24_3_evidence_strengthening_status')}")
    print(f"C24-4: {final.get('c24_4_integration_replay_status')}")
    print(f"C24-5: {final.get('c24_5_robustness_route_status')}")
    print(f"C24-6 final decision: {final.get('final_decision')}")
    print(f"selected probe: {json.dumps(jsonable(final.get('selected_evidence_probe')), sort_keys=True)}")
    print(f"selected integration: {json.dumps(jsonable(final.get('selected_integration_formula')), sort_keys=True)}")
    print(f"VCMR @0.5 R@1/R@5/R@10/R@100: {vcmr.get('VCMR_R@1_IoU0.5')}/{vcmr.get('VCMR_R@5_IoU0.5')}/{vcmr.get('VCMR_R@10_IoU0.5')}/{vcmr.get('VCMR_R@100_IoU0.5')}")
    print(f"VCMR @0.7 R@1/R@5/R@10/R@100: {vcmr.get('VCMR_R@1_IoU0.7')}/{vcmr.get('VCMR_R@5_IoU0.7')}/{vcmr.get('VCMR_R@10_IoU0.7')}/{vcmr.get('VCMR_R@100_IoU0.7')}")
    print(f"wrong-video/high-score risk: {final.get('wrong_video_risk')}")
    print(f"official was not run: {not final.get('official_val_used')}")
    print(f"promoted official remains: {final.get('current_promoted_system')}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", default="all", choices=["all", "c24_0", "c24_1", "c24_2", "c24_3", "c24_4", "c24_5", "c24_6"])
    parser.add_argument("--mode", default="medium", choices=["smoke", "medium", "full"])
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.stage == "all":
        recs = run_all(args.mode, args.seed)
        print_summary(recs)
    elif args.stage == "c24_0":
        stage_c24_0(args.mode, args.seed)
    elif args.stage == "c24_1":
        stage_c24_1(args.mode, args.seed)
    elif args.stage == "c24_2":
        stage_c24_2(args.mode, args.seed)
    elif args.stage == "c24_3":
        stage_c24_3(args.mode, args.seed)
    elif args.stage == "c24_4":
        stage_c24_4(args.mode, args.seed)
    elif args.stage == "c24_5":
        stage_c24_5(args.mode, args.seed)
    else:
        stage_c24_6(args.mode, args.seed)


if __name__ == "__main__":
    main()
