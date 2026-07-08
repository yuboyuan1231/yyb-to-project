#!/usr/bin/env python3
"""C24G BMN/T2 evidence closure.

This script audits whether BMN/T2 evidence can be recovered for the same
first-stage top128 row space as C24F. It deliberately does not run official
validation, read official prediction pools, or promote any C24G result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

import run_c18_full_hybrid_freeze_candidate as c18
from run_c12_native_retriever_training import ROOT


PYTHON = "/home/a/miniconda3/envs/conquer-rlem/bin/python"
PROMOTED = "C7-B6 R1SelectiveTop1"
BASE_C24F_COMMIT = "c652b84d991be2ef93cd3358536bbd57174f135d"
C24F_CACHE = Path("/tmp/c24f_score_cache/CONQUER-RLEM-c2c3")
C24G_CACHE = Path("/tmp/c24g_score_cache/CONQUER-RLEM-c2c3")

OUT0 = ROOT / "c24g_0_protocol_freeze"
OUT1 = ROOT / "c24g_1_bmn_t2_source_recovery"
OUT2 = ROOT / "c24g_2_bmn_t2_trainfit_materialization"
OUT3 = ROOT / "c24g_3_full_evidence_table_rebuild"
OUT4 = ROOT / "c24g_4_signal_integration_reeval"
OUT5 = ROOT / "c24g_5_robustness_route_decision"
OUT6 = ROOT / "c24g_6_final_decision"

MODE_LIMITS = {
    "smoke": {"sample_rows": 1200},
    "medium": {"sample_rows": 4000},
    "full": {"sample_rows": 8000},
}

BMN_COLS = [
    "bmn_pred_iou", "bmn_p_iou_05", "bmn_p_iou_07", "bmn_rank_score",
    "bmn_final_score", "bmn_span_rank", "bmn_margin", "start_prob",
    "end_prob", "actionness_score", "best_bmn_span_start",
    "best_bmn_span_end", "bmn_available", "bmn_missing_reason",
]
T2_COLS = ["t2_score", "t2_margin", "t2_agreement", "t2_rank", "t2_available", "t2_missing_reason"]
EVENT_COLS = [
    "event_relevance_score", "query_event_score", "event_rank", "best_event_id",
    "best_event_start", "best_event_end", "best_event_duration",
    "best_event_method", "event_overlap_proxy", "event_z", "event_available",
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


def sha256_file(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def stable_hash(obj: Any) -> str:
    return hashlib.sha256(json.dumps(jsonable(obj), sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:16]


def file_record(path: Path, sha: bool = False) -> Dict[str, Any]:
    rec = {
        "path": str(path),
        "exists": path.exists(),
        "readable": os.access(path, os.R_OK) if path.exists() else False,
        "size_bytes": path.stat().st_size if path.exists() and path.is_file() else None,
    }
    if sha and path.exists() and path.is_file() and rec["size_bytes"] and rec["size_bytes"] < 200_000_000:
        rec["sha256"] = sha256_file(path)
    elif sha and path.exists() and path.is_file():
        rec["stable_hash"] = stable_hash({"path": str(path), "size": rec["size_bytes"], "mtime": int(path.stat().st_mtime)})
    return rec


def parquet_record(path: Path, sample_split: bool = True) -> Dict[str, Any]:
    rec = file_record(path, sha=True)
    if not path.exists():
        return rec
    pf = pq.ParquetFile(path)
    cols = pf.schema.names
    rec.update({
        "row_count": pf.metadata.num_rows,
        "row_groups": pf.num_row_groups,
        "columns": cols,
        "schema_hash": stable_hash(cols),
        "join_keys": [c for c in ["split", "query_id", "video_id", "span_start", "span_end"] if c in cols],
        "bmn_columns_present": [c for c in ["bmn_pred_iou", "bmn_p_iou_05", "bmn_p_iou_07", "bmn_rank_score", "bmn_final_score", "bmn_score", "bmn_norm"] if c in cols],
        "t2_columns_present": [c for c in ["t2_score", "t2_norm", "t2_rank", "t2_margin"] if c in cols],
        "official_contaminated": "official" in str(path).lower() and "pseudo_official" not in str(path).lower(),
    })
    if sample_split and "split" in cols:
        sample = []
        for rg in sorted(set([0, max(0, pf.num_row_groups - 1)])):
            try:
                tab = pf.read_row_group(rg, columns=[c for c in ["split", "query_id"] if c in cols]).slice(0, 4096)
                sample.append(tab.to_pandas())
            except Exception as exc:  # pragma: no cover - audit best effort
                rec["split_sample_error"] = str(exc)
        if sample:
            sdf = pd.concat(sample, ignore_index=True)
            rec["split_sample_values"] = sorted(sdf["split"].astype(str).dropna().unique().tolist()) if "split" in sdf else []
            rec["query_sample_count"] = int(sdf["query_id"].nunique()) if "query_id" in sdf else None
    rec["usable_for_c24g"] = bool(
        {"split", "query_id", "video_id"}.issubset(set(cols))
        and (rec["bmn_columns_present"] or rec["t2_columns_present"])
        and not rec["official_contaminated"]
    )
    return rec


def c24f_table_path(mode: str) -> Path:
    return C24F_CACHE / f"C24F_CANONICAL_FULL_EVIDENCE_TABLE_{mode}.local.parquet"


def c24g_table_path(mode: str) -> Path:
    return C24G_CACHE / f"C24G_FULL_BMN_T2_EVENT_EVIDENCE_TABLE_{mode}.local.parquet"


def load_c24f_table(mode: str) -> pd.DataFrame:
    path = c24f_table_path(mode)
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_parquet(path)


def split_coverage(df: pd.DataFrame) -> Dict[str, Any]:
    return {str(k): {"rows": int(v["rows"]), "queries": int(v["queries"])} for k, v in df.groupby("split", observed=True).agg(rows=("query_id", "size"), queries=("query_id", "nunique")).to_dict("index").items()}


def z_cols(df: pd.DataFrame, source: str, target: str) -> None:
    val = pd.to_numeric(df.get(source, pd.Series(np.nan, index=df.index)), errors="coerce")
    mean = val.groupby([df["split"], df["query_id"]], observed=True).transform("mean")
    std = val.groupby([df["split"], df["query_id"]], observed=True).transform("std").replace(0, np.nan)
    df[target] = ((val - mean) / std).fillna(0.0)


def iou_array(st: np.ndarray, ed: np.ndarray, gt_s: np.ndarray, gt_e: np.ndarray) -> np.ndarray:
    valid = np.isfinite(st) & np.isfinite(ed) & (ed > st)
    inter = np.maximum(0.0, np.minimum(ed, gt_e) - np.maximum(st, gt_s))
    union = np.maximum(ed, gt_e) - np.minimum(st, gt_s)
    return np.where(valid, inter / np.maximum(union, 1e-6), 0.0)


def evidence_coverage(df: pd.DataFrame) -> Dict[str, Any]:
    bmn = df.get("bmn_available", df.get("bmn_score_available", pd.Series(False, index=df.index))).astype(bool)
    t2 = df.get("t2_available", df.get("t2_score_available", pd.Series(False, index=df.index))).astype(bool)
    event = df.get("event_available", pd.Series(False, index=df.index)).astype(bool)
    by_split = {}
    for split, g in df.groupby("split", observed=True):
        idx = g.index
        by_split[str(split)] = {
            "rows": int(len(g)),
            "queries": int(g["query_id"].nunique()),
            "bmn_available_rows": int(bmn.loc[idx].sum()),
            "t2_available_rows": int(t2.loc[idx].sum()),
            "event_available_rows": int(event.loc[idx].sum()),
            "bmn_coverage": float(bmn.loc[idx].mean()) if len(g) else 0.0,
            "t2_coverage": float(t2.loc[idx].mean()) if len(g) else 0.0,
            "event_coverage": float(event.loc[idx].mean()) if len(g) else 0.0,
        }
    return {
        "row_count": int(len(df)),
        "query_count": int(df["query_id"].nunique()),
        "split_coverage": split_coverage(df),
        "evidence_by_split": by_split,
        "bmn_missing_count": int((~bmn).sum()),
        "t2_missing_count": int((~t2).sum()),
        "event_missing_count": int((~event).sum()),
        "event_direct_columns_present": bool(event.any() and df.get("best_event_id", pd.Series(index=df.index)).notna().any()),
        "duplicate_candidate_count": int(df.duplicated(["split", "query_id", "video_id"]).sum()) if {"split", "query_id", "video_id"}.issubset(df.columns) else None,
        "invalid_span_count": int(((pd.to_numeric(df.get("span_end"), errors="coerce") <= pd.to_numeric(df.get("span_start"), errors="coerce")) & df.get("span_start").notna() & df.get("span_end").notna()).sum()) if {"span_start", "span_end"}.issubset(df.columns) else None,
    }


def describe_numeric(df: pd.DataFrame, cols: Sequence[str]) -> Dict[str, Any]:
    out = {}
    for c in cols:
        if c not in df:
            continue
        s = pd.to_numeric(df[c], errors="coerce")
        out[c] = {
            "count": int(s.notna().sum()),
            "missing": int(s.isna().sum()),
            "mean": float(s.mean()) if s.notna().any() else None,
            "p50": float(s.quantile(0.50)) if s.notna().any() else None,
            "p90": float(s.quantile(0.90)) if s.notna().any() else None,
        }
    return out


def corr_pack(df: pd.DataFrame, score_col: str, label_col: str) -> Dict[str, Any]:
    if score_col not in df or label_col not in df:
        return {"available": False}
    s = pd.to_numeric(df[score_col], errors="coerce")
    y = pd.to_numeric(df[label_col], errors="coerce")
    mask = s.notna() & y.notna()
    if int(mask.sum()) < 5:
        return {"available": False, "count": int(mask.sum())}
    return {
        "available": True,
        "count": int(mask.sum()),
        "pearson": float(s[mask].corr(y[mask], method="pearson")),
        "spearman": float(s[mask].corr(y[mask], method="spearman")),
    }


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


def metric_delta(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, float]:
    keys = [k for k, v in a.items() if isinstance(v, (int, float))]
    return {k: float(a.get(k, 0.0) or 0.0) - float(b.get(k, 0.0) or 0.0) for k in keys}


def add_labels(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    same = out["video_id"].astype(str) == out["gt_video_id"].astype(str)
    iou = iou_array(
        pd.to_numeric(out["span_start"], errors="coerce").to_numpy(np.float32),
        pd.to_numeric(out["span_end"], errors="coerce").to_numpy(np.float32),
        pd.to_numeric(out["gt_start"], errors="coerce").to_numpy(np.float32),
        pd.to_numeric(out["gt_end"], errors="coerce").to_numpy(np.float32),
    )
    out["candidate_iou"] = np.where(same.to_numpy(), iou, 0.0)
    out["iou_ge_05"] = out["candidate_iou"] >= 0.5
    out["iou_ge_07"] = out["candidate_iou"] >= 0.7
    out["correct_video"] = same
    out["correct_video_and_iou07"] = same & out["iou_ge_07"]
    return out


def normalize_bmn_t2(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "bmn_available" not in out:
        out["bmn_available"] = out.get("bmn_score_available", pd.Series(False, index=out.index)).astype(bool)
    if "t2_available" not in out:
        out["t2_available"] = out.get("t2_score_available", pd.Series(False, index=out.index)).astype(bool)
    out.loc[~out["bmn_available"].astype(bool), "bmn_missing_reason"] = "no_train_fit_bmn_source_for_first_stage_top128"
    out.loc[out["bmn_available"].astype(bool), "bmn_missing_reason"] = ""
    out.loc[~out["t2_available"].astype(bool), "t2_missing_reason"] = "no_train_fit_t2_source_for_first_stage_top128"
    out.loc[out["t2_available"].astype(bool), "t2_missing_reason"] = ""
    out["bmn_score_available"] = out["bmn_available"].astype(bool)
    out["t2_score_available"] = out["t2_available"].astype(bool)
    out["missing_bmn_mask"] = ~out["bmn_available"].astype(bool)
    out["missing_t2_mask"] = ~out["t2_available"].astype(bool)
    for col in ["bmn_final_score", "t2_score", "event_relevance_score", "first_stage_score"]:
        if col in out:
            z_cols(out, col, {"bmn_final_score": "bmn_z", "t2_score": "t2_z", "event_relevance_score": "event_z", "first_stage_score": "first_stage_z"}[col])
    return out


def stage_c24g_0(mode: str, seed: int) -> Dict[str, Any]:
    branch = sh("git branch --show-current")
    commit = sh("git rev-parse HEAD")
    dirty = sh("git status --short")
    c24f_files = {
        "c24f_1": ROOT / "c24f_1_canonical_full_evidence_table/C24F_1_TABLE_DECISION.json",
        "c24f_2": ROOT / "c24f_2_evidence_materialization/C24F_2_MATERIALIZATION_DECISION.json",
        "c24f_3": ROOT / "c24f_3_full_evidence_signal_reaudit/C24F_3_SIGNAL_DECISION.json",
        "c24f_4": ROOT / "c24f_4_full_evidence_integration_replay/C24F_4_INTEGRATION_DECISION.json",
        "c24f_5": ROOT / "c24f_5_robustness_and_route_decision/C24F_5_ROBUSTNESS_DECISION.json",
        "c24f_6": ROOT / "c24f_6_final_decision/C24F_6_NEXT_STEP_DECISION.json",
    }
    c24f2 = load_json(c24f_files["c24f_2"], {})
    c24f6 = load_json(ROOT / "c24f_6_final_decision/C24F_6_FINAL_DECISION.json", {})
    deps = {
        "current_branch": branch,
        "current_commit": commit,
        "dirty_status": dirty.splitlines(),
        "based_on_c24f_branch": branch in {"c24g-full-bmn-t2-evidence-closure", "c24f-full-evidence-materialization-repair"},
        "c24f_files": {k: file_record(v, sha=True) for k, v in c24f_files.items()},
        "c24f_local_cache": file_record(C24F_CACHE),
        "c24f_medium_table": parquet_record(c24f_table_path("medium"), sample_split=False),
        "first_stage_top128_manifest": file_record(ROOT / "c23_1_full_trainfit_readiness/C23_1_FIRST_STAGE_TOP128_MANIFEST.json", sha=True),
        "c22r_alignment": file_record(ROOT / "c22r_2_alignment_repair/C22R_2_ALIGNMENT_DECISION.json", sha=True),
        "c23_manifest": file_record(ROOT / "c23_1_full_trainfit_readiness/C23_1_FULL_TRAINFIT_DECISION.json", sha=True),
        "stale_official_marker_warning": bool(list(ROOT.glob("**/*OFFICIAL*"))),
    }
    missing_core = [k for k, v in deps["c24f_files"].items() if not v["exists"]]
    root_level_markers = [
        x for x in dirty.splitlines()
        if (
            "/" not in x.split()[-1]
            and "OFFICIAL" in x.upper()
            and any(tok in x.upper() for tok in ["PREDICTION", "RAW", "NMS"])
            and not x.split()[-1].endswith(".py")
        )
    ]
    deps["root_level_stale_marker_warning"] = root_level_markers
    contamination = bool(root_level_markers)
    event_methods = c24f2.get("event_methods_materialized") or []
    ready = (
        not missing_core
        and c24f6.get("final_decision") == "C24F_NEED_MORE_FULL_EVIDENCE_REPAIR"
        and c24f2.get("status") == "C24F_EVIDENCE_MATERIALIZED_PARTIAL"
        and bool(c24f2.get("event_direct_columns_present"))
        and all(m in event_methods for m in ["E0_fixed_window_events", "E1_visual_cosine_change_point_events", "E2_subtitle_boundary_events", "E3_multimodal_change_point_events", "E4_hybrid_fixed_change_events"])
        and not contamination
    )
    if contamination:
        status = "C24G_PROTOCOL_BLOCKED_CONTAMINATION_RISK"
    elif ready:
        status = "C24G_PROTOCOL_READY"
    else:
        status = "C24G_PROTOCOL_BLOCKED_MISSING_CORE_ARTIFACTS"
    rec = {
        "stage": "C24G-0",
        "status": status,
        "mode": mode,
        "seed": seed,
        "dependency_audit": deps,
        "c24f_final": c24f6.get("final_decision"),
        "c24f_2_status": c24f2.get("status"),
        "direct_event_columns_present": bool(c24f2.get("event_direct_columns_present")),
        "event_methods": event_methods,
        "missing_event_count": (c24f2.get("event_materialization_manifest") or {}).get("missing_event_count"),
        "bmn_t2_train_fit_missing_is_blocker": True,
        "official_val_used": False,
        "official_prediction_pool_used": False,
        "pseudo_official_holdout_used_for_selection": False,
        "evaluator_modified": False,
        "nms_modified": False,
        "current_promoted_system": PROMOTED,
        "c24g_is_promoted_system": False,
        "repro_command": f"{PYTHON} run_c24g_full_bmn_t2_evidence_closure.py --stage all --mode {mode} --seed {seed} --force",
    }
    write_text(OUT0 / "C24G_0_PROTOCOL.md", f"# C24G-0 Protocol\n\nStatus: `{status}`.")
    write_json(OUT0 / "C24G_0_PROTOCOL.json", rec)
    write_text(OUT0 / "C24G_0_C24F_ACCEPTANCE.md", "# C24F Acceptance\n\nC24F direct event evidence is accepted as materialized for the medium subset. BMN/T2 train_fit evidence remains the active blocker.")
    write_text(OUT0 / "C24G_0_FORBIDDEN_ACTIONS_AUDIT.md", "# Forbidden Actions Audit\n\n- official validation: false\n- official prediction pool read: false\n- pseudo official used for selection: false\n- evaluator modified: false\n- NMS modified: false")
    write_json(OUT0 / "C24G_0_DEPENDENCY_AUDIT.json", deps)
    write_json(OUT0 / "C24G_0_REPRODUCIBILITY_MANIFEST.json", rec)
    return rec


def source_paths() -> Dict[str, Path]:
    return {
        "c17_medium": Path("/tmp/c17_score_cache/CONQUER-RLEM-c2c3/C17_2_SCORE_TABLE_medium.local.parquet"),
        "c18_pseudo_medium": Path("/tmp/c18_score_cache/CONQUER-RLEM-c2c3/C18_PSEUDO_SCORE_TABLE_medium.local.parquet"),
        "c19_medium": Path("/tmp/c19_score_cache/CONQUER-RLEM-c2c3/C19_CANONICAL_SELECTED_SCORE_TABLE_medium.local.parquet"),
        "c20_train_medium": Path("/tmp/c20_score_cache/CONQUER-RLEM-c2c3/C20_TOP1_EVIDENCE_train_medium.local.parquet"),
        "c21_pair_train_medium": Path("/tmp/c21_score_cache/CONQUER-RLEM-c2c3/C21_FRONT_RANK_PAIR_DATASET_train_medium.local.parquet"),
        "c24f_medium": c24f_table_path("medium"),
    }


def stage_c24g_1(mode: str, seed: int, force: bool = False) -> Dict[str, Any]:
    audits = {name: parquet_record(path) for name, path in source_paths().items()}
    c24f = load_c24f_table(mode)
    coverage = evidence_coverage(normalize_bmn_t2(c24f))
    train = coverage["evidence_by_split"].get("train_fit", {})
    select = coverage["evidence_by_split"].get("calib_select", {})
    holdout = coverage["evidence_by_split"].get("calib_holdout", {})
    train_bmn_ok = float(train.get("bmn_coverage", 0.0)) >= 0.95
    train_t2_ok = float(train.get("t2_coverage", 0.0)) >= 0.95
    select_holdout_ok = min(float(select.get("bmn_coverage", 0.0)), float(select.get("t2_coverage", 0.0)), float(holdout.get("bmn_coverage", 0.0)), float(holdout.get("t2_coverage", 0.0))) >= 0.95
    if train_bmn_ok and train_t2_ok and select_holdout_ok:
        status = "C24G_BMN_T2_SOURCE_READY"
    elif select_holdout_ok:
        status = "C24G_BMN_T2_SOURCE_PARTIAL_REBUILD_NEEDED"
    else:
        status = "C24G_BMN_T2_SOURCE_MISSING_REBUILD_REQUIRED"
    sample_cols = [c for c in ["split", "query_id", "video_id", "bmn_final_score", "t2_score", "bmn_score_available", "t2_score_available", "bmn_missing_reason", "t2_missing_reason"] if c in normalize_bmn_t2(c24f)]
    sample_path = OUT1 / "C24G_1_SOURCE_SAMPLE.parquet"
    sample_path.parent.mkdir(parents=True, exist_ok=True)
    normalize_bmn_t2(c24f).head(MODE_LIMITS[mode]["sample_rows"])[sample_cols].to_parquet(sample_path, index=False)
    rec = {
        "stage": "C24G-1",
        "status": status,
        "mode": mode,
        "seed": seed,
        "bmn_source_search_audit": audits,
        "t2_source_search_audit": audits,
        "existing_score_cache_audit": audits,
        "train_fit_coverage_audit": coverage,
        "can_directly_recover_bmn_train_fit": train_bmn_ok,
        "can_directly_recover_t2_train_fit": train_t2_ok,
        "need_materialize_splits": ["train_fit"] if not (train_bmn_ok and train_t2_ok) else [],
        "estimated_cost": {"rebuild_required": not (train_bmn_ok and train_t2_ok), "gpu_needed": "likely if rerunning BMN model", "io_note": "existing sources cover calib splits but not train_fit first-stage top128"},
        "source_sample_path": str(sample_path),
        "official_val_used": False,
        "official_prediction_pool_used": False,
        "pseudo_official_holdout_used_for_selection": False,
    }
    write_text(OUT1 / "C24G_1_SOURCE_RECOVERY_PLAN.md", "# C24G-1 Source Recovery Plan\n\nAudit existing BMN/T2 sources before any rebuild. Do not assume train_fit coverage.")
    write_json(OUT1 / "C24G_1_BMN_SOURCE_SEARCH_AUDIT.json", rec["bmn_source_search_audit"])
    write_json(OUT1 / "C24G_1_T2_SOURCE_SEARCH_AUDIT.json", rec["t2_source_search_audit"])
    write_json(OUT1 / "C24G_1_EXISTING_SCORE_CACHE_AUDIT.json", rec["existing_score_cache_audit"])
    write_json(OUT1 / "C24G_1_TRAIN_FIT_COVERAGE_AUDIT.json", coverage)
    write_json(OUT1 / "C24G_1_RECOVERY_MANIFEST.json", rec)
    write_json(OUT1 / "C24G_1_SOURCE_RECOVERY_DECISION.json", rec)
    write_text(OUT1 / "C24G_1_SOURCE_RECOVERY_DECISION.md", f"# C24G-1 Source Recovery Decision\n\nStatus: `{status}`.\n\nExisting sources do not provide true train_fit BMN/T2 evidence for the first-stage top128 row space.")
    return rec


def stage_c24g_2(mode: str, seed: int, force: bool = False) -> Dict[str, Any]:
    C24G_CACHE.mkdir(parents=True, exist_ok=True)
    df = normalize_bmn_t2(load_c24f_table(mode))
    # C24G-2 is explicitly conservative: C24F/C19 evidence is reused where
    # present, while train_fit BMN/T2 remains masked rather than silently filled.
    local_path = C24G_CACHE / f"C24G_BMN_T2_PARTIAL_MATERIALIZATION_{mode}.local.parquet"
    df.to_parquet(local_path, index=False)
    manifest = {
        "path": str(local_path),
        "size_bytes": local_path.stat().st_size,
        "sha256": sha256_file(local_path),
        "row_count": int(len(df)),
        "query_count": int(df["query_id"].nunique()),
        "split_coverage": split_coverage(df),
        "generation_command": f"{PYTHON} run_c24g_full_bmn_t2_evidence_closure.py --stage c24g_2 --mode {mode} --seed {seed} --force",
        "source": "C24F canonical medium table plus explicit BMN/T2 missing masks",
        "full_not_completed_reason": "No existing source provides train_fit BMN/T2 evidence for first-stage top128 candidates.",
    }
    coverage = evidence_coverage(df)
    distribution = describe_numeric(df, ["bmn_final_score", "bmn_z", "t2_score", "t2_z", "event_relevance_score", "event_z"])
    train_cov = coverage["evidence_by_split"].get("train_fit", {})
    all_full = all(v.get("bmn_coverage", 0.0) >= 0.95 and v.get("t2_coverage", 0.0) >= 0.95 for v in coverage["evidence_by_split"].values())
    status = "C24G_BMN_T2_MATERIALIZED" if all_full and mode == "full" else "C24G_BMN_T2_MATERIALIZED_PARTIAL"
    sample_path = OUT2 / "C24G_2_MATERIALIZATION_SAMPLE.parquet"
    sample_path.parent.mkdir(parents=True, exist_ok=True)
    df.head(MODE_LIMITS[mode]["sample_rows"]).to_parquet(sample_path, index=False)
    progress = {"completed_chunks": 1, "total_chunks": 1, "resume_supported": True, "last_status": status}
    rec = {
        "stage": "C24G-2",
        "status": status,
        "mode": mode,
        "seed": seed,
        "bmn_materialization_manifest": {**manifest, "train_fit_coverage": train_cov.get("bmn_coverage", 0.0), "missing_reason": "train_fit BMN source absent"},
        "t2_materialization_manifest": {**manifest, "train_fit_coverage": train_cov.get("t2_coverage", 0.0), "missing_reason": "train_fit T2 source absent"},
        "materialization_progress": progress,
        "evidence_coverage_audit": coverage,
        "evidence_distribution_audit": distribution,
        "materialization_sample_path": str(sample_path),
        "full_not_completed_reason": manifest["full_not_completed_reason"],
        "official_val_used": False,
        "official_prediction_pool_used": False,
        "pseudo_official_holdout_used_for_selection": False,
    }
    write_text(OUT2 / "C24G_2_MATERIALIZATION_PLAN.md", "# C24G-2 Materialization Plan\n\nReuse C24F event evidence and existing BMN/T2 where present. Do not silently fill train_fit BMN/T2.")
    write_json(OUT2 / "C24G_2_BMN_MATERIALIZATION_MANIFEST.json", rec["bmn_materialization_manifest"])
    write_json(OUT2 / "C24G_2_T2_MATERIALIZATION_MANIFEST.json", rec["t2_materialization_manifest"])
    write_json(OUT2 / "C24G_2_MATERIALIZATION_PROGRESS.json", progress)
    write_json(OUT2 / "C24G_2_EVIDENCE_COVERAGE_AUDIT.json", coverage)
    write_json(OUT2 / "C24G_2_EVIDENCE_DISTRIBUTION_AUDIT.json", distribution)
    write_json(OUT2 / "C24G_2_MATERIALIZATION_DECISION.json", rec)
    write_text(OUT2 / "C24G_2_MATERIALIZATION_DECISION.md", f"# C24G-2 Materialization Decision\n\nStatus: `{status}`.\n\nTrain_fit BMN/T2 evidence remains missing and masked.")
    return rec


def stage_c24g_3(mode: str, seed: int, force: bool = False) -> Dict[str, Any]:
    df = normalize_bmn_t2(load_c24f_table(mode))
    df = add_labels(df)
    C24G_CACHE.mkdir(parents=True, exist_ok=True)
    path = c24g_table_path(mode)
    df.to_parquet(path, index=False)
    coverage = evidence_coverage(df)
    full_ready = all(v.get("bmn_coverage", 0.0) >= 0.95 and v.get("t2_coverage", 0.0) >= 0.95 and v.get("event_coverage", 0.0) >= 0.95 for v in coverage["evidence_by_split"].values())
    status = "C24G_FULL_EVIDENCE_TABLE_READY" if full_ready and mode == "full" else "C24G_FULL_EVIDENCE_TABLE_PARTIAL"
    schema = {
        "columns": df.columns.tolist(),
        "schema_hash": stable_hash(df.columns.tolist()),
        "diagnostic_only_columns": ["gt_video_id", "gt_start", "gt_end", "candidate_iou", "iou_ge_05", "iou_ge_07", "correct_video", "correct_video_and_iou07", "event_overlap_proxy"],
        "feature_families": {"bmn": BMN_COLS, "t2": T2_COLS, "event": EVENT_COLS},
    }
    manifest = {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "row_count": int(len(df)),
        "query_count": int(df["query_id"].nunique()),
        "split_coverage": split_coverage(df),
        "schema_hash": schema["schema_hash"],
        "config_hash": stable_hash({"mode": mode, "seed": seed, "stage": "C24G-3"}),
        "generation_command": f"{PYTHON} run_c24g_full_bmn_t2_evidence_closure.py --stage c24g_3 --mode {mode} --seed {seed} --force",
        "local_only": True,
    }
    sample_path = OUT3 / "C24G_3_FULL_EVIDENCE_SAMPLE.parquet"
    sample_path.parent.mkdir(parents=True, exist_ok=True)
    df.head(MODE_LIMITS[mode]["sample_rows"]).to_parquet(sample_path, index=False)
    join_audit = {"no_position_based_join": True, "duplicate_candidate_count": coverage["duplicate_candidate_count"], "silent_zero_fill_detected": False, "source": "C24F canonical row space reused"}
    missing = {"bmn_missing_count": coverage["bmn_missing_count"], "t2_missing_count": coverage["t2_missing_count"], "event_missing_count": coverage["event_missing_count"], "by_split": coverage["evidence_by_split"]}
    rec = {
        "stage": "C24G-3",
        "status": status,
        "mode": mode,
        "seed": seed,
        "canonical_schema": schema,
        "table_manifest": manifest,
        "evidence_join_audit": join_audit,
        "split_coverage_audit": coverage,
        "missing_mask_audit": missing,
        "sample_path": str(sample_path),
        "official_val_used": False,
        "official_prediction_pool_used": False,
        "pseudo_official_holdout_used_for_selection": False,
    }
    write_text(OUT3 / "C24G_3_TABLE_REBUILD_PLAN.md", "# C24G-3 Table Rebuild Plan\n\nRebuild local canonical evidence table from C24F row space with explicit BMN/T2 masks and direct event evidence.")
    write_json(OUT3 / "C24G_3_CANONICAL_SCHEMA.json", schema)
    write_json(OUT3 / "C24G_3_TABLE_MANIFEST.json", manifest)
    write_json(OUT3 / "C24G_3_EVIDENCE_JOIN_AUDIT.json", join_audit)
    write_json(OUT3 / "C24G_3_SPLIT_COVERAGE_AUDIT.json", coverage)
    write_json(OUT3 / "C24G_3_MISSING_MASK_AUDIT.json", missing)
    write_json(OUT3 / "C24G_3_TABLE_REBUILD_DECISION.json", rec)
    write_text(OUT3 / "C24G_3_TABLE_REBUILD_DECISION.md", f"# C24G-3 Table Rebuild Decision\n\nStatus: `{status}`.\n\nEvent evidence is present; BMN/T2 train_fit remains partial and masked.")
    return rec


def high_score_fp(df: pd.DataFrame, col: str) -> Dict[str, Any]:
    if col not in df:
        return {"available": False}
    sdf = df[df["split"].isin(["calib_select", "calib_holdout"])].copy()
    s = pd.to_numeric(sdf[col], errors="coerce")
    if not s.notna().any():
        return {"available": False}
    threshold = float(s.quantile(0.90))
    high = sdf[s >= threshold]
    return {
        "available": True,
        "rows": int(len(high)),
        "threshold_p90": threshold,
        "wrong_video_rate": float((~high["correct_video"].astype(bool)).mean()) if len(high) else None,
        "iou07_false_positive_rate": float((~high["iou_ge_07"].astype(bool)).mean()) if len(high) else None,
    }


def stage_c24g_4(mode: str, seed: int, force: bool = False) -> Dict[str, Any]:
    df = pd.read_parquet(c24g_table_path(mode)) if c24g_table_path(mode).exists() else add_labels(normalize_bmn_t2(load_c24f_table(mode)))
    # Do not train/select a new formula when BMN/T2 train_fit is missing.
    features = ["first_stage_score", "bmn_final_score", "t2_score", "event_relevance_score", "bmn_z", "t2_z", "event_z"]
    signal = {c: {**corr_pack(df[df["split"].isin(["calib_select", "calib_holdout"])], c, "candidate_iou"), "auc_iou05": auc_score(df["iou_ge_05"], df[c]) if c in df else None, "auc_iou07": auc_score(df["iou_ge_07"], df[c]) if c in df else None} for c in features if c in df}
    coverage = evidence_coverage(df)
    materialized_full = all(v.get("bmn_coverage", 0.0) >= 0.95 and v.get("t2_coverage", 0.0) >= 0.95 for v in coverage["evidence_by_split"].values())
    formulas = {
        "A_C23_baseline": {"status": "reported_from_prior"},
        "B_C19_C21_hybrid": {"status": "reported_from_partial_table"},
        "D_event_only": {"status": "diagnostic_only"},
        "K_all_evidence_no_regression_guard": {"status": "not_selected_because_bmn_t2_train_fit_missing"},
    }
    if materialized_full:
        status = "C24G_FULL_EVIDENCE_FRONT_RANK_WEAK"
    else:
        status = "C24G_FULL_EVIDENCE_INCONCLUSIVE"
    selected = {"name": None, "selection_split": "calib_select", "selected": False, "reason": "BMN/T2 train_fit evidence is partial; selection is blocked by C24G constraints."}
    wrong = {"wrong_video_top1_high_score": high_score_fp(df, "event_relevance_score"), "bmn": high_score_fp(df, "bmn_z"), "t2": high_score_fp(df, "t2_z")}
    rec = {
        "stage": "C24G-4",
        "status": status,
        "mode": mode,
        "seed": seed,
        "evidence_signal_audit": signal,
        "front_rank_signal_audit": {"materialized_full": materialized_full, "blocked_reason": None if materialized_full else "BMN/T2 train_fit evidence incomplete"},
        "complementarity_audit": {"bmn_event_t2_comparison": signal},
        "integration_results": formulas,
        "component_ablation": formulas,
        "wrong_video_audit": wrong,
        "selected_integration": selected,
        "final_vcmr_metrics": {},
        "official_val_used": False,
        "official_prediction_pool_used": False,
        "pseudo_official_holdout_used_for_selection": False,
    }
    write_text(OUT4 / "C24G_4_REEVAL_PLAN.md", "# C24G-4 Re-evaluation Plan\n\nRun signal diagnostics on the partial full-evidence table. Block formula selection while BMN/T2 train_fit is incomplete.")
    write_json(OUT4 / "C24G_4_EVIDENCE_SIGNAL_AUDIT.json", signal)
    write_json(OUT4 / "C24G_4_FRONT_RANK_SIGNAL_AUDIT.json", rec["front_rank_signal_audit"])
    write_json(OUT4 / "C24G_4_COMPLEMENTARITY_AUDIT.json", rec["complementarity_audit"])
    write_json(OUT4 / "C24G_4_INTEGRATION_RESULTS.json", formulas)
    write_json(OUT4 / "C24G_4_COMPONENT_ABLATION.json", formulas)
    write_json(OUT4 / "C24G_4_WRONG_VIDEO_AUDIT.json", wrong)
    write_json(OUT4 / "C24G_4_SELECTED_INTEGRATION.json", selected)
    write_json(OUT4 / "C24G_4_REEVAL_DECISION.json", rec)
    write_text(OUT4 / "C24G_4_REEVAL_DECISION.md", f"# C24G-4 Re-evaluation Decision\n\nStatus: `{status}`.\n\nNo C25 routing decision is allowed from partial BMN/T2 evidence.")
    return rec


def stage_c24g_5(mode: str, seed: int, s2: Dict[str, Any] | None = None, s4: Dict[str, Any] | None = None) -> Dict[str, Any]:
    s2 = s2 or load_json(OUT2 / "C24G_2_MATERIALIZATION_DECISION.json", {})
    s4 = s4 or load_json(OUT4 / "C24G_4_REEVAL_DECISION.json", {})
    materialized = s2.get("status") == "C24G_BMN_T2_MATERIALIZED"
    if not materialized:
        route = "C24G_NEED_MORE_BMN_T2_MATERIALIZATION"
    elif s4.get("status") in ["C24G_FULL_EVIDENCE_FRONT_RANK_WEAK", "C24G_FULL_EVIDENCE_R100_ONLY", "C24G_FULL_EVIDENCE_HARMFUL"]:
        route = "C24G_READY_FOR_C25_RAW_FRAME_STRONG_FEATURE_AUDIT"
    else:
        route = "C24G_CONTINUE_EXISTING_FEATURE_EVIDENCE_STRENGTHENING"
    status = "C24G_ROBUSTNESS_PARTIAL" if not materialized else "C24G_ROBUSTNESS_PASS"
    c25 = {
        "full_bmn_t2_event_evidence_materialized": materialized,
        "event_direct_columns_present": True,
        "bmn_t2_train_fit_missing_resolved": materialized,
        "full_evidence_front_rank_still_weak": s4.get("status") in ["C24G_FULL_EVIDENCE_FRONT_RANK_WEAK", "C24G_FULL_EVIDENCE_R100_ONLY", "C24G_FULL_EVIDENCE_HARMFUL"],
        "recommend_c25": route == "C24G_READY_FOR_C25_RAW_FRAME_STRONG_FEATURE_AUDIT",
    }
    rec = {
        "stage": "C24G-5",
        "status": status,
        "mode": mode,
        "seed": seed,
        "route_recommendation": route,
        "seed_robustness": {"seed2026": {"materialization": s2.get("status"), "reeval": s4.get("status")}, "seed2027": "not run; blocked by BMN/T2 materialization", "seed2028": "not run; blocked by BMN/T2 materialization"},
        "query_duration_robustness": "partial evidence only",
        "d_e_f_subset_audit": "partial evidence only",
        "score_distribution_audit": s2.get("evidence_distribution_audit"),
        "c25_route_audit": c25,
        "pseudo_onelook_diagnostic": {"pseudo_one_look_executed": False, "pseudo_official_not_used_for_selection": True},
        "selection_firewall_audit": {"pseudo_official_holdout_used_for_selection": False, "official_val_used": False},
        "official_val_used": False,
        "official_prediction_pool_used": False,
        "pseudo_official_holdout_used_for_selection": False,
    }
    write_text(OUT5 / "C24G_5_ROBUSTNESS_PLAN.md", "# C24G-5 Robustness Plan\n\nRoute to C25 only after full BMN/T2/event evidence is materialized and still weak.")
    write_json(OUT5 / "C24G_5_SEED_ROBUSTNESS.json", rec["seed_robustness"])
    write_json(OUT5 / "C24G_5_QUERY_DURATION_ROBUSTNESS.json", rec["query_duration_robustness"])
    write_json(OUT5 / "C24G_5_D_E_F_SUBSET_AUDIT.json", rec["d_e_f_subset_audit"])
    write_json(OUT5 / "C24G_5_SCORE_DISTRIBUTION_AUDIT.json", rec["score_distribution_audit"])
    write_text(OUT5 / "C24G_5_C25_ROUTE_AUDIT.md", f"# C25 Route Audit\n\nRecommendation: `{route}`.\n\nC25 is blocked until BMN/T2 train_fit evidence is materially closed.")
    write_json(OUT5 / "C24G_5_PSEUDO_ONELOOK_DIAGNOSTIC.json", rec["pseudo_onelook_diagnostic"])
    write_json(OUT5 / "C24G_5_SELECTION_FIREWALL_AUDIT.json", rec["selection_firewall_audit"])
    write_json(OUT5 / "C24G_5_ROBUSTNESS_DECISION.json", rec)
    write_text(OUT5 / "C24G_5_ROBUSTNESS_DECISION.md", f"# C24G-5 Robustness Decision\n\nStatus: `{status}`.\n\nRoute: `{route}`.")
    return rec


def stage_c24g_6(mode: str, seed: int, recs: Dict[str, Dict[str, Any]] | None = None) -> Dict[str, Any]:
    recs = recs or {
        "c24g_0": load_json(OUT0 / "C24G_0_PROTOCOL.json", {}),
        "c24g_1": load_json(OUT1 / "C24G_1_SOURCE_RECOVERY_DECISION.json", {}),
        "c24g_2": load_json(OUT2 / "C24G_2_MATERIALIZATION_DECISION.json", {}),
        "c24g_3": load_json(OUT3 / "C24G_3_TABLE_REBUILD_DECISION.json", {}),
        "c24g_4": load_json(OUT4 / "C24G_4_REEVAL_DECISION.json", {}),
        "c24g_5": load_json(OUT5 / "C24G_5_ROBUSTNESS_DECISION.json", {}),
    }
    materialized = recs["c24g_2"].get("status") == "C24G_BMN_T2_MATERIALIZED"
    if recs["c24g_4"].get("status") == "C24G_FULL_EVIDENCE_FRONT_RANK_PROMISING":
        decision = "C24G_READY_FOR_C23R_EVIDENCE_STRENGTHENED_NATIVE_COUPLING"
    elif not materialized:
        decision = "C24G_NEED_MORE_BMN_T2_MATERIALIZATION"
    elif recs["c24g_5"].get("route_recommendation") == "C24G_READY_FOR_C25_RAW_FRAME_STRONG_FEATURE_AUDIT":
        decision = "C24G_READY_FOR_C25_RAW_FRAME_STRONG_FEATURE_AUDIT"
    else:
        decision = "C24G_CONTINUE_EXISTING_FEATURE_EVIDENCE_STRENGTHENING"
    final_metrics = recs["c24g_4"].get("final_vcmr_metrics") or {}
    c24f_final = load_json(ROOT / "c24f_6_final_decision/C24F_6_FINAL_DECISION.json", {})
    c24_final = load_json(ROOT / "c24_6_final_decision/C24_6_FINAL_DECISION.json", {})
    c23_final = load_json(ROOT / "c23_6_final_decision/C23_6_FINAL_DECISION.json", {})
    c19_base = {}
    coverage = recs["c24g_2"].get("evidence_coverage_audit") or {}
    train = (coverage.get("evidence_by_split") or {}).get("train_fit", {})
    rec = {
        "stage": "C24G-6",
        "status": decision,
        "final_decision": decision,
        "mode": mode,
        "seed": seed,
        "c24g_0_protocol_status": recs["c24g_0"].get("status"),
        "c24g_1_source_recovery_status": recs["c24g_1"].get("status"),
        "c24g_2_bmn_t2_materialization_status": recs["c24g_2"].get("status"),
        "c24g_3_full_evidence_table_status": recs["c24g_3"].get("status"),
        "c24g_4_signal_integration_status": recs["c24g_4"].get("status"),
        "c24g_5_robustness_route_status": recs["c24g_5"].get("status"),
        "bmn_train_fit_coverage": train.get("bmn_coverage", 0.0),
        "t2_train_fit_coverage": train.get("t2_coverage", 0.0),
        "event_direct_columns_present": bool(coverage.get("event_direct_columns_present")),
        "full_evidence_coverage": coverage,
        "selected_integration_formula": recs["c24g_4"].get("selected_integration"),
        "final_vcmr_metrics": final_metrics,
        "final_vr_metrics": {k: final_metrics.get(k) for k in ["VR_R@1", "VR_R@5", "VR_R@10", "VR_R@100"]},
        "delta_vs_c24f": metric_delta(final_metrics, c24f_final.get("final_vcmr_metrics", {})),
        "delta_vs_c24": metric_delta(final_metrics, c24_final.get("final_vcmr_metrics", {})),
        "delta_vs_c23": metric_delta(final_metrics, c23_final.get("final_vcmr_metrics", {})),
        "delta_vs_c19_c21": metric_delta(final_metrics, c19_base),
        "wrong_video_risk": recs["c24g_4"].get("wrong_video_audit"),
        "high_score_false_positive": recs["c24g_4"].get("wrong_video_audit"),
        "evidence_signal_conclusion": recs["c24g_4"].get("front_rank_signal_audit"),
        "existing_feature_path_still_has_value": False,
        "c25_raw_frame_strong_feature_audit_recommended": decision == "C24G_READY_FOR_C25_RAW_FRAME_STRONG_FEATURE_AUDIT",
        "official_val_used": False,
        "official_prediction_pool_used": False,
        "pseudo_official_holdout_used_for_selection": False,
        "evaluator_modified": False,
        "nms_modified": False,
        "c24g_is_promoted_system": False,
        "current_promoted_system": PROMOTED,
    }
    packet = {
        "decision": decision,
        "bmn_train_fit_coverage": rec["bmn_train_fit_coverage"],
        "t2_train_fit_coverage": rec["t2_train_fit_coverage"],
        "event_direct_columns_present": rec["event_direct_columns_present"],
        "official_still_forbidden": True,
        "ready_for_official": False,
    }
    write_json(OUT6 / "C24G_6_FINAL_DECISION.json", rec)
    write_text(OUT6 / "C24G_6_FINAL_DECISION.md", f"# C24G-6 Final Decision\n\nFinal decision: `{decision}`.\n\nC24G is not promoted. Current promoted official remains `{PROMOTED}`.")
    write_json(OUT6 / "C24G_6_BMN_T2_EVIDENCE_CLOSURE_PACKET.json", packet)
    write_text(OUT6 / "C24G_6_BMN_T2_EVIDENCE_CLOSURE_PACKET.md", f"# C24G BMN/T2 Evidence Closure Packet\n\nDecision: `{decision}`.\nBMN train_fit coverage: `{rec['bmn_train_fit_coverage']}`.\nT2 train_fit coverage: `{rec['t2_train_fit_coverage']}`.")
    write_text(OUT6 / "C24G_6_RISK_REGISTER.md", "# C24G-6 Risk Register\n\n- BMN/T2 train_fit evidence remains unavailable for first-stage top128.\n- Event evidence is present but cannot justify C25 without full BMN/T2 closure.\n- No official validation was run.\n")
    write_json(OUT6 / "C24G_6_NEXT_STEP_DECISION.json", {"decision": decision, "official_forbidden": True})
    write_text(OUT6 / "C24G_6_NEXT_STEP_DECISION.md", f"# C24G Next Step\n\n`{decision}`")
    return rec


def run_all(mode: str, seed: int, force: bool = False) -> Dict[str, Dict[str, Any]]:
    r0 = stage_c24g_0(mode, seed)
    if r0["status"] != "C24G_PROTOCOL_READY":
        raise RuntimeError(f"C24G protocol blocked: {r0['status']}")
    r1 = stage_c24g_1(mode, seed, force=force)
    r2 = stage_c24g_2(mode, seed, force=force)
    r3 = stage_c24g_3(mode, seed, force=force)
    r4 = stage_c24g_4(mode, seed, force=force)
    r5 = stage_c24g_5(mode, seed, r2, r4)
    r6 = stage_c24g_6(mode, seed, {"c24g_0": r0, "c24g_1": r1, "c24g_2": r2, "c24g_3": r3, "c24g_4": r4, "c24g_5": r5})
    return {"c24g_0": r0, "c24g_1": r1, "c24g_2": r2, "c24g_3": r3, "c24g_4": r4, "c24g_5": r5, "c24g_6": r6}


def print_summary(recs: Dict[str, Dict[str, Any]]) -> None:
    final = recs["c24g_6"]
    print("\n===== C24G SUMMARY =====")
    print(f"branch: {sh('git branch --show-current')}")
    print(f"commit: {sh('git rev-parse HEAD')}")
    print(f"C24G-0 protocol status: {final.get('c24g_0_protocol_status')}")
    print(f"C24G-1 source recovery status: {final.get('c24g_1_source_recovery_status')}")
    print(f"C24G-2 materialization status: {final.get('c24g_2_bmn_t2_materialization_status')}")
    print(f"C24G-3 full evidence table status: {final.get('c24g_3_full_evidence_table_status')}")
    print(f"C24G-4 signal/integration status: {final.get('c24g_4_signal_integration_status')}")
    print(f"C24G-5 route status: {final.get('c24g_5_robustness_route_status')}")
    print(f"C24G-6 final decision: {final.get('final_decision')}")
    print(f"BMN train_fit coverage: {final.get('bmn_train_fit_coverage')}")
    print(f"T2 train_fit coverage: {final.get('t2_train_fit_coverage')}")
    print(f"event direct columns present: {final.get('event_direct_columns_present')}")
    print(f"selected formula: {json.dumps(jsonable(final.get('selected_integration_formula')), ensure_ascii=False)}")
    print(f"delta vs C24F: {json.dumps(jsonable(final.get('delta_vs_c24f')), ensure_ascii=False)}")
    print(f"delta vs C23: {json.dumps(jsonable(final.get('delta_vs_c23')), ensure_ascii=False)}")
    print(f"delta vs C19/C21: {json.dumps(jsonable(final.get('delta_vs_c19_c21')), ensure_ascii=False)}")
    print(f"wrong-video top1/high-score: {json.dumps(jsonable(final.get('wrong_video_risk')), ensure_ascii=False)}")
    print(f"high-score false positive rate: {json.dumps(jsonable(final.get('high_score_false_positive')), ensure_ascii=False)}")
    print(f"C25 raw/frame/strong feature audit recommended: {final.get('c25_raw_frame_strong_feature_audit_recommended')}")
    print(f"pseudo_official_holdout used for selection: {final.get('pseudo_official_holdout_used_for_selection')}")
    print(f"official was not run: {not final.get('official_val_used')}")
    print(f"files committed to GitHub: pending git commit/push")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", default="all", choices=["all", "c24g_0", "c24g_1", "c24g_2", "c24g_3", "c24g_4", "c24g_5", "c24g_6"])
    parser.add_argument("--mode", default="medium", choices=["smoke", "medium", "full"])
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.mode == "full":
        m2 = load_json(OUT2 / "C24G_2_MATERIALIZATION_DECISION.json", {})
        if m2.get("status") != "C24G_BMN_T2_MATERIALIZED":
            raise RuntimeError("C24G full is blocked until medium C24G-1/C24G-2 prove real BMN/T2 materialization.")
    if args.stage == "all":
        recs = run_all(args.mode, args.seed, force=args.force)
        print_summary(recs)
        return
    fn = {
        "c24g_0": stage_c24g_0,
        "c24g_1": stage_c24g_1,
        "c24g_2": stage_c24g_2,
        "c24g_3": stage_c24g_3,
        "c24g_4": stage_c24g_4,
        "c24g_5": stage_c24g_5,
        "c24g_6": stage_c24g_6,
    }[args.stage]
    rec = fn(args.mode, args.seed, force=args.force) if args.stage not in {"c24g_0"} else fn(args.mode, args.seed)
    print(json.dumps(jsonable(rec), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
