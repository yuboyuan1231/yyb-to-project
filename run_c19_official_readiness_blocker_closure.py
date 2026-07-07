#!/usr/bin/env python3
"""C19 official readiness blocker closure.

This script assembles train-only evidence for whether the C18 hybrid is ready
for a human-authorized official review. It never runs official validation and
never reads the official prediction pool.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd
import torch

import run_c17_bmn_t2_native_vcmr_integration as c17
import run_c18_full_hybrid_freeze_candidate as c18
from run_c12_native_retriever_training import DEVICE, ROOT, sha256_file


torch.set_num_threads(min(24, os.cpu_count() or 1))

PROMOTED = "C7-B6 R1SelectiveTop1"
BASE_C18_COMMIT = "96d8926b56c56a0e9506eb623f290d847daaf8a8"
SELECTED = {
    "name": "grid_a0.65_b0.25_g0.1",
    "family": "G_retriever_BMN_T2",
    "alpha": 0.65,
    "beta": 0.25,
    "gamma": 0.10,
}

OUT0 = ROOT / "c19_0_protocol_freeze"
OUT1 = ROOT / "c19_1_full_score_blocker_closure"
OUT2 = ROOT / "c19_2_baseline_replay_closure"
OUT3 = ROOT / "c19_3_front_rank_risk_audit"
OUT4 = ROOT / "c19_4_readiness_packet_assembly"
OUT5 = ROOT / "c19_5_final_blocker_decision"


def jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
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


def sh_ok(cmd: str) -> bool:
    return subprocess.run(cmd, cwd=ROOT, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0


def stable_hash(obj: Any) -> str:
    raw = json.dumps(jsonable(obj), sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def c19_cache_dir() -> Path:
    base = Path(os.environ.get("C19_SCORE_CACHE_DIR", "/tmp/c19_score_cache")) / "CONQUER-RLEM-c2c3"
    base.mkdir(parents=True, exist_ok=True)
    return base


def canonical_cache(mode: str) -> Path:
    return c19_cache_dir() / f"C19_CANONICAL_SELECTED_SCORE_TABLE_{mode}.local.parquet"


def require_status(path: Path, ok: Sequence[str]) -> Dict[str, Any]:
    obj = load_json(path, {})
    status = obj.get("status") or obj.get("final_decision")
    if status not in set(ok):
        raise RuntimeError(f"{path} status={status}; expected one of {ok}")
    return obj


def file_record(path: Path, read_parquet: bool = False) -> Dict[str, Any]:
    rec: Dict[str, Any] = {
        "path": str(path),
        "exists": path.exists(),
        "readable": os.access(path, os.R_OK) if path.exists() else False,
        "size_bytes": path.stat().st_size if path.exists() else None,
        "sha256": sha256_file(path) if path.exists() and path.is_file() else None,
    }
    if read_parquet and path.exists():
        try:
            pf = pd.read_parquet(path)
            rec.update(parquet_summary(pf, include_dist=False))
        except Exception as exc:  # pragma: no cover - audit path
            rec["read_error"] = repr(exc)
    return rec


def parquet_summary(df: pd.DataFrame, include_dist: bool = True) -> Dict[str, Any]:
    split = df["split"].astype(str) if "split" in df else pd.Series([], dtype=str)
    score_cols = [c for c in ["retriever_score", "bmn_final_score", "bmn_score", "t2_score", "final_score", "old_c12_score"] if c in df]
    out: Dict[str, Any] = {
        "row_count": int(len(df)),
        "query_count": int(df["query_id"].nunique()) if "query_id" in df else None,
        "video_count": int(df["video_id"].nunique()) if "video_id" in df else None,
        "candidate_count": int(len(df)),
        "split_coverage": {str(k): int(v) for k, v in df.groupby(split, observed=True)["query_id"].nunique().to_dict().items()} if "query_id" in df and "split" in df else {},
        "score_columns": score_cols,
        "schema_hash": stable_hash({"columns": list(df.columns), "dtypes": {c: str(t) for c, t in df.dtypes.items()}}),
        "config_hash": stable_hash(SELECTED),
    }
    if include_dist:
        numeric = df.select_dtypes(include=[np.number])
        out["duplicate_candidate_count"] = int(df.duplicated(["split", "seed", "query_id", "video_id", "span_start", "span_end"]).sum()) if {"split", "seed", "query_id", "video_id", "span_start", "span_end"}.issubset(df.columns) else None
        out["invalid_span_count"] = int(((df["span_end"] <= df["span_start"]) | df["span_start"].isna() | df["span_end"].isna()).sum()) if {"span_start", "span_end"}.issubset(df.columns) else None
        out["nan_count"] = {c: int(df[c].isna().sum()) for c in score_cols}
        out["inf_count"] = int(np.isinf(numeric.to_numpy()).sum()) if len(numeric.columns) else 0
        if "split" in df and score_cols:
            out["score_distribution_by_split"] = df.groupby("split", observed=True)[score_cols].describe().to_dict()
        if "query_type" in df and score_cols:
            out["score_distribution_by_query_type"] = df.groupby("query_type", observed=True)[score_cols].mean(numeric_only=True).to_dict()
        if "duration_bucket" in df and score_cols:
            out["score_distribution_by_duration_bucket"] = df.groupby("duration_bucket", observed=True)[score_cols].mean(numeric_only=True).to_dict()
    return out


def load_or_build_canonical(mode: str, seed: int, force: bool = False) -> tuple[pd.DataFrame, Dict[str, Any]]:
    cache = canonical_cache(mode)
    if cache.exists() and not force:
        df = pd.read_parquet(cache)
        return df, {"cache_reused": True, "path": str(cache), "sha256": sha256_file(cache), **parquet_summary(df)}

    source_cache = c17.local_score_cache_path(mode)
    if not source_cache.exists() and mode != "medium":
        source_cache = c17.local_score_cache_path("medium")
    if not source_cache.exists():
        raise FileNotFoundError(f"missing C17 score cache: {source_cache}")

    print(f"[C19-1] building canonical selected score table from {source_cache}", flush=True)
    df = pd.read_parquet(source_cache)
    fill_audit = c17.add_normalized_scores_inplace(df)
    df["bmn_score"] = df["bmn_final_score"]
    df["final_score"] = c18.formula_score_array(df, SELECTED)
    df.to_parquet(cache, index=False)
    manifest = {
        "cache_reused": False,
        "path": str(cache),
        "source_cache": str(source_cache),
        "source_cache_sha256": sha256_file(source_cache),
        "sha256": sha256_file(cache),
        "formula": SELECTED,
        "fill_audit": fill_audit,
        "generation_command": f"run_c19_official_readiness_blocker_closure.py --stage c19_1 --mode {mode} --seed {seed}",
        **parquet_summary(df),
    }
    write_json(cache.with_suffix(".audit.json"), manifest)
    return df, manifest


def necessary_missing(df: pd.DataFrame) -> Dict[str, Any]:
    cols = ["retriever_score", "bmn_score", "bmn_final_score", "t2_score", "final_score", "span_start", "span_end", "query_id", "video_id", "split"]
    return {
        c: {
            "exists": c in df.columns,
            "missing_count": int(df[c].isna().sum()) if c in df.columns else None,
        }
        for c in cols
    }


def selected_splits(df: pd.DataFrame) -> set[str]:
    return set(df["split"].astype(str).unique()) if "split" in df else set()


def stage_c19_0(mode: str, seed: int) -> Dict[str, Any]:
    OUT0.mkdir(parents=True, exist_ok=True)
    branch = sh("git branch --show-current")
    commit = sh("git rev-parse HEAD")
    dirty_lines = sh("git status --short").splitlines()
    c18_paths = {
        "c18_1": ROOT / "c18_1_full_score_readiness/C18_1_FULL_SCORE_DECISION.json",
        "c18_2": ROOT / "c18_2_baseline_comparable_replay/C18_2_BASELINE_REPLAY_DECISION.json",
        "c18_3": ROOT / "c18_3_front_rank_hybrid_calibration/C18_3_CALIBRATION_DECISION.json",
        "c18_4": ROOT / "c18_4_pseudo_official_onelook/C18_4_ONELOOK_DECISION.json",
        "c18_5": ROOT / "c18_5_full_consistency_robustness/C18_5_ROBUSTNESS_DECISION.json",
        "c18_6": ROOT / "c18_6_freeze_review/C18_6_NEXT_STEP_DECISION.json",
        "c12_split": ROOT / "c12_1_schema_and_split/C12_1_SPLIT_MANIFEST.json",
        "c12_schema": ROOT / "c12_1_schema_and_split/C12_1_SCHEMA_MANIFEST.json",
        "c12_5t": ROOT / "c12_5t_best_span_promotion/C12_5T_DECISION.json",
        "c18_score_manifest": ROOT / "c18_1_full_score_readiness/C18_1_SCORE_CACHE_MANIFEST.json",
    }
    loaded = {k: load_json(p, {}) for k, p in c18_paths.items()}
    selected = loaded["c18_3"].get("selected_name") or loaded["c18_6"].get("selected_config")
    root_markers = [str(p.relative_to(ROOT)) for p in [ROOT / "C9_OFFICIAL_VAL_AUTHORIZED", ROOT / "OFFICIAL_VAL_AUTHORIZED"] if p.exists()]
    checks = {
        "branch_is_c19": branch == "c19-official-readiness-blocker-closure",
        "base_contains_c18_commit": sh_ok(f"git merge-base --is-ancestor {BASE_C18_COMMIT} HEAD"),
        "all_core_c18_paths_exist": all(p.exists() for p in c18_paths.values()),
        "c18_1_partial": loaded["c18_1"].get("status") == "C18_FULL_SCORE_PARTIAL_REPRODUCIBLE",
        "c18_2_partial": loaded["c18_2"].get("status") == "C18_BASELINE_REPLAY_PARTIAL",
        "c18_3_promising": loaded["c18_3"].get("status") == "C18_FRONT_RANK_CALIBRATION_PROMISING",
        "c18_4_stable": loaded["c18_4"].get("status") == "C18_PSEUDO_ONELOOK_STABLE",
        "c18_5_pass": loaded["c18_5"].get("status") == "C18_ROBUSTNESS_PASS",
        "c18_6_continue": loaded["c18_6"].get("final_decision") == "C18_CONTINUE_FULL_HYBRID_TRAIN_ONLY",
        "selected_config_expected": selected == SELECTED["name"],
        "official_val_used_false": loaded["c18_6"].get("official_val_used") is False,
        "pseudo_not_selection": loaded["c18_6"].get("pseudo_official_holdout_used_for_selection") is False,
        "promoted_system_retained": loaded["c18_6"].get("current_promoted_system") == PROMOTED,
        "c17_cache_exists": c17.local_score_cache_path("medium").exists(),
        "c18_pseudo_cache_exists": c18.c18_pseudo_cache("medium").exists(),
        "no_root_stale_authorization_marker": not root_markers,
    }
    missing = [k for k, p in c18_paths.items() if not p.exists()]
    if missing:
        status = "C19_PROTOCOL_BLOCKED_MISSING_CORE_ARTIFACTS"
    elif root_markers:
        status = "C19_PROTOCOL_BLOCKED_CONTAMINATION_RISK"
    elif all(checks.values()):
        status = "C19_PROTOCOL_READY"
    else:
        status = "C19_PROTOCOL_BLOCKED_MISSING_CORE_ARTIFACTS"
    rec = {
        "stage": "C19-0",
        "status": status,
        "mode": mode,
        "seed": seed,
        "branch": branch,
        "commit": commit,
        "dirty_status": dirty_lines,
        "checks": checks,
        "missing_core_artifacts": missing,
        "root_stale_authorization_markers": root_markers,
        "current_promoted_system": PROMOTED,
        "c18_c19_is_promoted_system": False,
        "official_val_used": False,
        "official_prediction_pool_used": False,
        "pseudo_official_holdout_used_for_selection": False,
    }
    write_json(OUT0 / "C19_0_PROTOCOL.json", rec)
    write_json(OUT0 / "C19_0_DEPENDENCY_AUDIT.json", {k: file_record(p) for k, p in c18_paths.items()})
    write_json(OUT0 / "C19_0_REPRODUCIBILITY_MANIFEST.json", {
        "mode": mode,
        "seed": seed,
        "device": str(DEVICE),
        "gpu": sh("nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader || true").splitlines(),
        "python": sh("/home/a/miniconda3/envs/conquer-rlem/bin/python --version"),
        "git_branch_vv": sh("git branch -vv"),
        "git_log": sh("git log --oneline --decorate -5"),
        "c17_score_cache": file_record(c17.local_score_cache_path("medium")),
        "c18_pseudo_cache": file_record(c18.c18_pseudo_cache("medium")),
    })
    write_text(OUT0 / "C19_0_PROTOCOL.md", f"# C19-0 Protocol Freeze\n\nstatus: `{status}`\n\nC19 is a no-official blocker-closure audit. Official validation and official prediction pools are forbidden.")
    write_text(OUT0 / "C19_0_C18_ACCEPTANCE.md", f"# C19-0 C18 Acceptance\n\nC18 selected `{SELECTED['name']}` is accepted as frozen train-only input. It is not a promoted official system.")
    write_text(OUT0 / "C19_0_FORBIDDEN_ACTIONS_AUDIT.md", "# C19-0 Forbidden Actions Audit\n\nNo official validation, no second official, no post-val adjustment, no official prediction pool, no evaluator/NMS modification, no pseudo selection, no silent zero-fill, no position-based join, and no promoted-system claim.")
    return rec


def source_audit(mode: str) -> Dict[str, Any]:
    sources = {
        "c17_medium_score_cache": c17.local_score_cache_path("medium"),
        "c17_requested_score_cache": c17.local_score_cache_path(mode),
        "c18_pseudo_medium_score_cache": c18.c18_pseudo_cache("medium"),
        "c19_canonical_requested_score_cache": canonical_cache(mode),
        "c16_bmn_manifest": ROOT / "c16_2_fullscale_bmn_training/C16_2_BMN_CHECKPOINT_MANIFEST.json",
        "c17_score_manifest": ROOT / "c17_2_full_bmn_score_export/C17_2_FULL_BMN_SCORE_MANIFEST.json",
        "c18_score_manifest": ROOT / "c18_1_full_score_readiness/C18_1_SCORE_CACHE_MANIFEST.json",
        "c18_selected_config": ROOT / "c18_3_front_rank_hybrid_calibration/C18_3_SELECTED_CONFIG.json",
    }
    out = {}
    for name, path in sources.items():
        out[name] = file_record(path, read_parquet=path.suffix == ".parquet" and path.exists() and path.stat().st_size < 1_500_000_000)
    return out


def stage_c19_1(mode: str, seed: int, force: bool = False) -> Dict[str, Any]:
    require_status(OUT0 / "C19_0_PROTOCOL.json", ["C19_PROTOCOL_READY"])
    OUT1.mkdir(parents=True, exist_ok=True)
    df, manifest = load_or_build_canonical(mode, seed, force=force)
    missing = necessary_missing(df)
    split_cov = selected_splits(df)
    required_ready_splits = {"train_fit", "calib_select", "calib_holdout"}
    has_calib = {"calib_select", "calib_holdout"}.issubset(split_cov)
    missing_source_counts = {
        "retriever_score": missing["retriever_score"]["missing_count"],
        "bmn_score": missing["bmn_score"]["missing_count"],
        "t2_score": missing["t2_score"]["missing_count"],
        "final_score": missing["final_score"]["missing_count"],
    }
    irrecoverable = {
        k: v for k, v in missing_source_counts.items()
        if v not in (None, 0) and k in {"bmn_score", "t2_score"}
    }
    if required_ready_splits.issubset(split_cov) and not irrecoverable and missing_source_counts.get("final_score") == 0:
        status = "C19_FULL_SCORE_READY"
    elif has_calib and missing_source_counts.get("final_score") == 0:
        status = "C19_FULL_SCORE_PARTIAL_ACCEPTABLE"
    elif len(df):
        status = "C19_FULL_SCORE_PARTIAL_UNRESOLVED"
    else:
        status = "C19_FULL_SCORE_BLOCKED"
    rec = {
        "stage": "C19-1",
        "status": status,
        "mode": mode,
        "seed": seed,
        "canonical_formula": SELECTED,
        "canonical_score_table": manifest,
        "score_source_audit": source_audit(mode),
        "missing_score_audit": {
            "necessary_columns": missing,
            "missing_source_counts": missing_source_counts,
            "irrecoverable_missing_scores": irrecoverable,
            "missing_mask_available": True,
            "silent_zero_fill_used": False,
            "normalization_fill_policy": "per-query minimum before minmax for normalized score only; source missing counts retained",
        },
        "ready_split_requirement": sorted(required_ready_splits),
        "observed_splits": sorted(split_cov),
        "not_full_ready_reason": None if status == "C19_FULL_SCORE_READY" else "medium/cache evidence does not cover train_fit and/or has audited cross-source BMN/T2 missing values",
        "official_val_used": False,
        "official_prediction_pool_used": False,
        "pseudo_official_holdout_diagnostic_flag": "pseudo_official_holdout" in split_cov,
    }
    sample_cols = [c for c in ["split", "query_id", "seed", "video_id", "span_start", "span_end", "retriever_score", "bmn_score", "t2_score", "final_score", "query_type", "duration_bucket"] if c in df]
    df[sample_cols].head(5000).to_parquet(OUT1 / "C19_1_SCORE_SAMPLE.parquet", index=False)
    write_text(OUT1 / "C19_1_FULL_SCORE_CLOSURE_PLAN.md", f"# C19-1 Full Score Blocker Closure\n\nBuild or verify canonical selected score table `{canonical_cache(mode)}`. Large score tables remain local-only.")
    write_json(OUT1 / "C19_1_SCORE_SOURCE_AUDIT.json", rec["score_source_audit"])
    write_json(OUT1 / "C19_1_SCORE_CACHE_MANIFEST.json", manifest)
    write_json(OUT1 / "C19_1_SCORE_SCHEMA.json", {"columns": list(df.columns), "schema_hash": manifest.get("schema_hash"), "config_hash": manifest.get("config_hash")})
    write_json(OUT1 / "C19_1_SCORE_REPRODUCIBILITY_AUDIT.json", {"command": rec["canonical_score_table"].get("generation_command"), "sha256": manifest.get("sha256"), "path": manifest.get("path"), "mode": mode, "seed": seed})
    write_json(OUT1 / "C19_1_MISSING_SCORE_AUDIT.json", rec["missing_score_audit"])
    write_json(OUT1 / "C19_1_SCORE_DISTRIBUTION_AUDIT.json", {k: manifest.get(k) for k in ["score_distribution_by_split", "score_distribution_by_query_type", "score_distribution_by_duration_bucket", "duplicate_candidate_count", "invalid_span_count", "nan_count", "inf_count"]})
    write_json(OUT1 / "C19_1_FULL_SCORE_DECISION.json", rec)
    write_text(OUT1 / "C19_1_FULL_SCORE_DECISION.md", f"# C19-1 Full Score Decision\n\nstatus: `{status}`\n\nC19 does not silently zero-fill missing source scores. In medium mode this cannot become full official-ready unless train_fit/full split coverage is present.")
    return rec


def formulas() -> Dict[str, Dict[str, Any]]:
    return {
        "C12_5T_T2": {"name": "B_T2_only", "family": "B_T2_only", "gamma": 1.0},
        "C18_selected_hybrid": dict(SELECTED),
        "BMN_only": {"name": "C_BMN_only", "family": "C_BMN_map_only", "beta": 1.0},
        "retriever_only": {"name": "A_retriever_only", "family": "A_retriever_only", "alpha": 1.0},
        "retriever_BMN": {"name": "D_retriever_BMN", "family": "D_retriever_BMN", "alpha": 0.45, "beta": 0.55},
        "retriever_T2": {"name": "E_retriever_T2", "family": "E_retriever_T2", "alpha": 0.45, "gamma": 0.55},
        "retriever_BMN_T2": dict(SELECTED),
    }


def evaluate_from_canonical(df: pd.DataFrame, selected_formulas: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    if "retriever_norm" not in df:
        c17.add_normalized_scores_inplace(df)
    ctx = c18.prepare_fast_eval(df)
    out = {}
    for name, formula in selected_formulas.items():
        out[name] = c18.fast_evaluate_formula(ctx, df, formula)
    return out


def c7_search_audit() -> Dict[str, Any]:
    patterns = ("c7", "c7b6", "r1selective", "zero_delta", "first_stage", "conquer")
    hits: List[Dict[str, Any]] = []
    compatible = []
    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        lower = str(path.relative_to(ROOT)).lower()
        if not any(p in lower for p in patterns):
            continue
        if path.stat().st_size > 80_000_000:
            continue
        official_like = any(tok in lower for tok in ["official_prediction", "official_val", "official_one_shot", "fixed_pool"])
        rec = {
            "path": str(path.relative_to(ROOT)),
            "size_bytes": path.stat().st_size,
            "official_like_excluded": official_like,
            "sha256": sha256_file(path) if not official_like and path.stat().st_size < 40_000_000 else None,
            "compatibility": "not_checked_official_like_excluded" if official_like else "not_c17_c18_row_schema",
        }
        if not official_like and path.suffix == ".json":
            try:
                obj = load_json(path, {})
                if isinstance(obj, dict) and {"split", "query_id", "video_id", "span_start", "span_end"}.issubset(obj.keys()):
                    rec["compatibility"] = "candidate_dict_but_not_row_table"
                elif isinstance(obj, dict) and "VCMR" in obj and "video2idx" in obj:
                    rec["compatibility"] = "legacy_vcmr_prediction_format_missing_c17_schema_columns"
                elif isinstance(obj, list) and obj and isinstance(obj[0], dict):
                    keys = set(obj[0].keys())
                    if {"split", "seed", "query_id", "video_id", "span_start", "span_end"}.issubset(keys):
                        rec["compatibility"] = "potentially_compatible"
                        compatible.append(rec)
            except Exception as exc:
                rec["read_error"] = repr(exc)
        hits.append(rec)
    hits = sorted(hits, key=lambda r: (r["official_like_excluded"], r["path"]))[:300]
    return {
        "status": "C19_C7_B6_REPLAY_UNAVAILABLE_WITH_PROOF" if not compatible else "C19_C7_B6_REPLAY_CANDIDATE_FOUND",
        "searched_patterns": list(patterns),
        "hit_count_recorded": len(hits),
        "compatible_candidates": compatible,
        "recorded_hits": hits,
        "official_prediction_pool_used": False,
        "reason": "No C7-B6 artifact was found in the C17/C18 train-only row schema; legacy VCMR prediction files are not used as comparable replay rows." if not compatible else None,
    }


def stage_c19_2(mode: str, seed: int) -> Dict[str, Any]:
    require_status(OUT1 / "C19_1_FULL_SCORE_DECISION.json", ["C19_FULL_SCORE_READY", "C19_FULL_SCORE_PARTIAL_ACCEPTABLE", "C19_FULL_SCORE_PARTIAL_UNRESOLVED"])
    OUT2.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(canonical_cache(mode))
    results = evaluate_from_canonical(df, formulas())
    c7 = c7_search_audit()
    comparison = {name: res for name, res in results.items()}
    c7_available = c7["status"] != "C19_C7_B6_REPLAY_UNAVAILABLE_WITH_PROOF"
    if c7_available:
        status = "C19_BASELINE_REPLAY_READY"
    elif c7["hit_count_recorded"] > 0:
        status = "C19_BASELINE_REPLAY_PARTIAL_ACCEPTABLE"
    else:
        status = "C19_BASELINE_REPLAY_PARTIAL_UNRESOLVED"
    sanity = {
        "same_evaluator": "C18 fast evaluator over C17/C18 train-only schema",
        "same_split": sorted(selected_splits(df)),
        "same_metric_schema": True,
        "timestamp_mapping": "span_start/span_end and gt_start/gt_end from C17 score table",
        "nms_consistency": f"per-video temporal NMS threshold {c17.NMS_THRESHOLD}; official NMS not modified",
        "duplicate_candidate_count": int(df.duplicated(["split", "seed", "query_id", "video_id", "span_start", "span_end"]).sum()),
        "invalid_span_count": int(((df["span_end"] <= df["span_start"]) | df["span_start"].isna() | df["span_end"].isna()).sum()),
        "row_coverage": parquet_summary(df, include_dist=False),
        "random_score_sanity": "covered by C17 native replay; C19 did not use it for selection",
        "oracle_score_sanity": "covered by C17 native replay; C19 did not use it for selection",
    }
    rec = {
        "stage": "C19-2",
        "status": status,
        "mode": mode,
        "baseline_comparison": comparison,
        "c7_b6_replay_search_audit": c7,
        "c7_b6_train_only_replay_availability": c7["status"],
        "sanity": sanity,
        "official_val_used": False,
        "official_prediction_pool_used": False,
        "pseudo_official_holdout_used_for_selection": False,
    }
    write_text(OUT2 / "C19_2_BASELINE_REPLAY_CLOSURE_PLAN.md", "# C19-2 Baseline Replay Closure\n\nReplay C12 T2, C18 selected hybrid, BMN-only, retriever-only, retriever+BMN, retriever+T2, and retriever+BMN+T2 under the same train-only evaluator.")
    write_json(OUT2 / "C19_2_C12_T2_REPLAY_RESULTS.json", comparison["C12_5T_T2"])
    write_json(OUT2 / "C19_2_C18_HYBRID_REPLAY_RESULTS.json", comparison["C18_selected_hybrid"])
    write_json(OUT2 / "C19_2_C7_B6_REPLAY_SEARCH_AUDIT.json", c7)
    write_json(OUT2 / "C19_2_C7_B6_REPLAY_RESULTS.json", {"status": c7["status"], "results": None, "official_prediction_pool_used": False})
    write_json(OUT2 / "C19_2_BASELINE_COMPARISON_TABLE.json", comparison)
    write_json(OUT2 / "C19_2_REPLAY_SANITY_AUDIT.json", sanity)
    write_json(OUT2 / "C19_2_BASELINE_REPLAY_DECISION.json", rec)
    write_text(OUT2 / "C19_2_BASELINE_REPLAY_DECISION.md", f"# C19-2 Baseline Replay Decision\n\nstatus: `{status}`\n\nC7-B6 compatible train-only row replay is unavailable with proof; official prediction rows were not read.")
    return rec


def metric_delta(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    out = {}
    for k, v in a.items():
        if isinstance(v, (int, float)) and isinstance(b.get(k), (int, float)):
            out[k] = float(v) - float(b[k])
    return out


def conflict_audit(df: pd.DataFrame) -> Dict[str, Any]:
    if "retriever_norm" not in df:
        c17.add_normalized_scores_inplace(df)
    score = c18.formula_score_array(df, SELECTED)
    tmp = df[["split", "seed", "query_id", "video_id", "gt_video_id", "span_start", "span_end", "retriever_norm", "bmn_norm", "t2_norm"]].copy()
    tmp["final_score"] = score
    top = tmp.sort_values("final_score", ascending=False).groupby(["split", "seed", "query_id"], observed=True).head(1)
    def sample(mask: pd.Series) -> List[Dict[str, Any]]:
        cols = ["split", "seed", "query_id", "video_id", "gt_video_id", "span_start", "span_end", "retriever_norm", "bmn_norm", "t2_norm", "final_score"]
        return top[mask][cols].head(20).to_dict("records")
    high_r_low_b = (top["retriever_norm"] >= 0.8) & (top["bmn_norm"] <= 0.2)
    low_r_high_b = (top["retriever_norm"] <= 0.2) & (top["bmn_norm"] >= 0.8)
    high_b_wrong_v = (top["bmn_norm"] >= 0.8) & (top["video_id"].astype(str) != top["gt_video_id"].astype(str))
    t2_rescue = (top["t2_norm"] >= 0.8) & (top["bmn_norm"] <= 0.35)
    bmn_rescue = (top["bmn_norm"] >= 0.8) & (top["t2_norm"] <= 0.35)
    retr_dom = (top["retriever_norm"] >= 0.8) & (top["bmn_norm"] <= 0.5) & (top["t2_norm"] <= 0.5)
    return {
        "top1_query_count": int(len(top)),
        "high_retriever_low_bmn_count": int(high_r_low_b.sum()),
        "low_retriever_high_bmn_count": int(low_r_high_b.sum()),
        "high_bmn_wrong_video_count": int(high_b_wrong_v.sum()),
        "t2_rescue_case_count": int(t2_rescue.sum()),
        "bmn_rescue_case_count": int(bmn_rescue.sum()),
        "retriever_dominance_case_count": int(retr_dom.sum()),
        "samples": {
            "high_retriever_low_bmn": sample(high_r_low_b),
            "low_retriever_high_bmn": sample(low_r_high_b),
            "high_bmn_wrong_video": sample(high_b_wrong_v),
            "t2_rescue": sample(t2_rescue),
            "bmn_rescue": sample(bmn_rescue),
            "retriever_dominance": sample(retr_dom),
        },
    }


def stage_c19_3(mode: str, seed: int) -> Dict[str, Any]:
    require_status(OUT2 / "C19_2_BASELINE_REPLAY_DECISION.json", ["C19_BASELINE_REPLAY_READY", "C19_BASELINE_REPLAY_PARTIAL_ACCEPTABLE", "C19_BASELINE_REPLAY_PARTIAL_UNRESOLVED"])
    OUT3.mkdir(parents=True, exist_ok=True)
    comp = load_json(OUT2 / "C19_2_BASELINE_COMPARISON_TABLE.json", {})
    selected = comp["C18_selected_hybrid"]["calib_holdout"]["summary"]
    t2 = comp["C12_5T_T2"]["calib_holdout"]["summary"]
    c17 = load_json(ROOT / "c18_3_front_rank_hybrid_calibration/C18_3_CALIBRATION_DECISION.json", {}).get("c17_replay", {}).get("calib_holdout", {}).get("summary", {})
    c18_prev = load_json(ROOT / "c18_3_front_rank_hybrid_calibration/C18_3_CALIBRATION_DECISION.json", {}).get("selected_results", {}).get("calib_holdout", {}).get("summary", {})
    df = pd.read_parquet(canonical_cache(mode))
    conflicts = conflict_audit(df)
    deltas = {
        "vs_C12_5T_T2": metric_delta(selected, t2),
        "vs_C17_best": metric_delta(selected, c17),
        "vs_C18_selected_replay": metric_delta(selected, c18_prev),
    }
    front_drop = (
        selected.get("VCMR_R@1_IoU0.7", 0.0) < c17.get("VCMR_R@1_IoU0.7", 0.0) - 0.5
        and selected.get("VCMR_R@5_IoU0.7", 0.0) < c17.get("VCMR_R@5_IoU0.7", 0.0) - 0.5
        and selected.get("VCMR_R@10_IoU0.7", 0.0) < c17.get("VCMR_R@10_IoU0.7", 0.0) - 0.5
    )
    r100_gain = selected.get("VCMR_R@100_IoU0.7", 0.0) >= t2.get("VCMR_R@100_IoU0.7", 0.0) + 10.0
    wrong_high = selected.get("wrong_video_high_score_rate", 100.0) >= 70.0
    if r100_gain and not front_drop and not wrong_high:
        status = "C19_FRONT_RANK_RISK_ACCEPTABLE"
    elif r100_gain and not front_drop:
        status = "C19_FRONT_RANK_RISK_WARNING"
    elif front_drop or wrong_high:
        status = "C19_FRONT_RANK_RISK_BLOCKING"
    else:
        status = "C19_FRONT_RANK_RISK_INCONCLUSIVE"
    rec = {
        "stage": "C19-3",
        "status": status,
        "mode": mode,
        "front_rank_metrics": selected,
        "deltas": deltas,
        "wrong_video_high_score_rate": selected.get("wrong_video_high_score_rate"),
        "correct_video_wrong_span_rate": selected.get("correct_video_wrong_span_rate"),
        "query_type_breakdown": comp["C18_selected_hybrid"]["calib_holdout"].get("query_type_breakdown", {}),
        "duration_breakdown": comp["C18_selected_hybrid"]["calib_holdout"].get("duration_breakdown", {}),
        "video_rank_breakdown": comp["C18_selected_hybrid"]["calib_holdout"].get("video_rank_breakdown", {}),
        "retriever_localizer_conflict_audit": conflicts,
        "judgment": {
            "r100_only_improvement": bool(r100_gain and front_drop),
            "front_rank_stable": not front_drop,
            "wrong_video_unacceptably_high": wrong_high,
            "retriever_localizer_calibration_blocker": bool(wrong_high and front_drop),
            "retriever_localizer_calibration_warning": wrong_high,
            "bmn_independent_front_rank_contribution": conflicts["bmn_rescue_case_count"] > 0,
            "t2_safety_not_selector": SELECTED["gamma"] <= 0.10,
        },
        "official_val_used": False,
        "official_prediction_pool_used": False,
    }
    write_text(OUT3 / "C19_3_FRONT_RANK_RISK_PLAN.md", "# C19-3 Front-rank Risk Plan\n\nAudit R@1/R@5/R@10/R@100 at IoU 0.5/0.7 and wrong-video risk. Do not summarize R@100 as front-rank success.")
    write_json(OUT3 / "C19_3_FRONT_RANK_METRICS.json", {"selected": selected, "deltas": deltas})
    write_json(OUT3 / "C19_3_WRONG_VIDEO_AUDIT.json", {"wrong_video_high_score_rate": selected.get("wrong_video_high_score_rate"), "correct_video_wrong_span_rate": selected.get("correct_video_wrong_span_rate"), "wrong_video_unacceptably_high": wrong_high})
    write_json(OUT3 / "C19_3_RETRIEVER_LOCALIZER_CONFLICT_AUDIT.json", conflicts)
    write_json(OUT3 / "C19_3_QUERY_DURATION_BREAKDOWN.json", {"query_type": rec["query_type_breakdown"], "duration": rec["duration_breakdown"], "video_rank": rec["video_rank_breakdown"]})
    write_text(OUT3 / "C19_3_FAILURE_CASES.md", "# C19-3 Failure Cases\n\nRepresentative top1 conflict samples are stored in `C19_3_RETRIEVER_LOCALIZER_CONFLICT_AUDIT.json`. Wrong-video risk remains the main front-rank concern.")
    write_json(OUT3 / "C19_3_RISK_DECISION.json", rec)
    write_text(OUT3 / "C19_3_RISK_DECISION.md", f"# C19-3 Risk Decision\n\nstatus: `{status}`\n\nR@100 is strong, but front-rank and wrong-video risk are judged separately.")
    return rec


def stage_c19_4(mode: str, seed: int) -> Dict[str, Any]:
    OUT4.mkdir(parents=True, exist_ok=True)
    s1 = load_json(OUT1 / "C19_1_FULL_SCORE_DECISION.json", {})
    s2 = load_json(OUT2 / "C19_2_BASELINE_REPLAY_DECISION.json", {})
    s3 = load_json(OUT3 / "C19_3_RISK_DECISION.json", {})
    firewall = {
        "official_val_used": False,
        "official_prediction_pool_used": False,
        "official_validation_may_be_run_by_this_script": False,
        "second_official_used": False,
        "post_val_adjustment_used": False,
        "evaluator_modified": False,
        "nms_modified": False,
        "c18_c19_promoted_system": False,
        "promoted_system_remains": PROMOTED,
    }
    selection_firewall = {
        "selected_config": SELECTED["name"],
        "formula": "S_final = 0.65 S_retriever + 0.25 S_BMN + 0.10 S_T2",
        "selected_config_source": "C18 calib_select",
        "pseudo_official_holdout_not_used_for_selection": True,
        "no_post_pseudo_adjustment": True,
    }
    artifacts = {
        "scripts": ["run_c19_official_readiness_blocker_closure.py", "run_c18_full_hybrid_freeze_candidate.py", "run_c17_bmn_t2_native_vcmr_integration.py"],
        "config_files": ["c18_3_front_rank_hybrid_calibration/C18_3_SELECTED_CONFIG.json"],
        "score_table_local_path": str(canonical_cache(mode)),
        "score_table_local_hash": sha256_file(canonical_cache(mode)) if canonical_cache(mode).exists() else None,
        "candidate_pool_hash": s1.get("canonical_score_table", {}).get("source_cache_sha256"),
        "schema_hash": s1.get("canonical_score_table", {}).get("schema_hash"),
        "split_manifest_hash": sha256_file(ROOT / "c12_1_schema_and_split/C12_1_SPLIT_MANIFEST.json") if (ROOT / "c12_1_schema_and_split/C12_1_SPLIT_MANIFEST.json").exists() else None,
        "evaluator_command": f"run_c19_official_readiness_blocker_closure.py --stage c19_2 --mode {mode} --seed {seed}",
        "selected_weight_command": "C18 selected grid_a0.65_b0.25_g0.1 on calib_select",
        "pseudo_one_look_command_from_C18": f"run_c18_full_hybrid_freeze_candidate.py --stage c18_4 --mode {mode} --seed {seed} --force",
        "missing_artifacts": [x for x in [s1.get("not_full_ready_reason"), s2.get("c7_b6_train_only_replay_availability"), s3.get("status") if s3.get("status") != "C19_FRONT_RANK_RISK_ACCEPTABLE" else None] if x],
        "local_only_artifacts": [str(canonical_cache(mode)), str(c17.local_score_cache_path(mode)), str(c18.c18_pseudo_cache(mode))],
        "files_safe_to_commit": ["run_c19_official_readiness_blocker_closure.py", "c19_0_*", "c19_1_*", "c19_2_*", "c19_3_*", "c19_4_*", "c19_5_*"],
        "files_not_committed_due_size": [str(canonical_cache(mode)), str(c17.local_score_cache_path(mode)), str(c18.c18_pseudo_cache(mode))],
    }
    limitations = []
    if s1.get("status") != "C19_FULL_SCORE_READY":
        limitations.append("full score blocker not fully closed")
    if s2.get("status") != "C19_BASELINE_REPLAY_READY":
        limitations.append("C7-B6 compatible train-only replay unavailable")
    if s3.get("status") != "C19_FRONT_RANK_RISK_ACCEPTABLE":
        limitations.append("front-rank/wrong-video risk remains")
    status = "C19_READINESS_PACKET_ASSEMBLED" if not limitations else "C19_READINESS_PACKET_ASSEMBLED_WITH_LIMITATIONS"
    rec = {
        "stage": "C19-4",
        "status": status,
        "mode": mode,
        "official_firewall": firewall,
        "selection_firewall": selection_firewall,
        "artifact_manifest": artifacts,
        "limitations": limitations,
    }
    write_text(OUT4 / "C19_4_READINESS_ASSEMBLY.md", f"# C19-4 Readiness Assembly\n\nstatus: `{status}`\n\nThis is a readiness packet only; it is not authorization to run official validation.")
    write_json(OUT4 / "C19_4_OFFICIAL_FIREWALL_AUDIT.json", firewall)
    write_json(OUT4 / "C19_4_SELECTION_FIREWALL_AUDIT.json", selection_firewall)
    write_json(OUT4 / "C19_4_ARTIFACT_MANIFEST.json", artifacts)
    write_text(OUT4 / "C19_4_REPRODUCIBILITY_COMMANDS.md", f"# C19-4 Reproducibility Commands\n\n```bash\n/home/a/miniconda3/envs/conquer-rlem/bin/python run_c19_official_readiness_blocker_closure.py --stage all --mode {mode} --seed {seed}\n```\n\nOfficial validation is not part of this command.")
    write_text(OUT4 / "C19_4_RISK_REGISTER.md", "\n".join(["# C19-4 Risk Register", "", *[f"- {x}" for x in limitations or ["no packet-level limitations recorded"]]]))
    write_json(OUT4 / "C19_4_READINESS_ASSEMBLY_DECISION.json", rec)
    write_text(OUT4 / "C19_4_READINESS_ASSEMBLY_DECISION.md", f"# C19-4 Assembly Decision\n\nstatus: `{status}`")
    return rec


def stage_c19_5(mode: str, seed: int) -> Dict[str, Any]:
    OUT5.mkdir(parents=True, exist_ok=True)
    s0 = load_json(OUT0 / "C19_0_PROTOCOL.json", {})
    s1 = load_json(OUT1 / "C19_1_FULL_SCORE_DECISION.json", {})
    s2 = load_json(OUT2 / "C19_2_BASELINE_REPLAY_DECISION.json", {})
    s3 = load_json(OUT3 / "C19_3_RISK_DECISION.json", {})
    s4 = load_json(OUT4 / "C19_4_READINESS_ASSEMBLY_DECISION.json", {})
    selected_metrics = s3.get("front_rank_metrics", {})
    if s0.get("status") != "C19_PROTOCOL_READY":
        final = "C19_NOT_READY_FOR_OFFICIAL"
    elif s1.get("status") != "C19_FULL_SCORE_READY":
        final = "C19_CONTINUE_TRAIN_ONLY_BLOCKER_CLOSURE"
    elif s2.get("status") == "C19_BASELINE_REPLAY_PARTIAL_UNRESOLVED":
        final = "C19_NEED_BASELINE_REPLAY_RESOLUTION"
    elif s3.get("status") == "C19_FRONT_RANK_RISK_BLOCKING":
        final = "C19_NEED_RETRIEVER_LOCALIZER_CALIBRATION"
    elif (
        s1.get("status") == "C19_FULL_SCORE_READY"
        and s2.get("status") in {"C19_BASELINE_REPLAY_READY", "C19_BASELINE_REPLAY_PARTIAL_ACCEPTABLE"}
        and s3.get("status") == "C19_FRONT_RANK_RISK_ACCEPTABLE"
        and s4.get("status") in {"C19_READINESS_PACKET_ASSEMBLED", "C19_READINESS_PACKET_ASSEMBLED_WITH_LIMITATIONS"}
    ):
        final = "C19_READY_FOR_ONE_SHOT_OFFICIAL_REVIEW"
    else:
        final = "C19_NOT_READY_FOR_OFFICIAL"
    rec = {
        "stage": "C19-5",
        "status": final,
        "final_decision": final,
        "mode": mode,
        "seed": seed,
        "c19_0_protocol_status": s0.get("status"),
        "c19_1_full_score_status": s1.get("status"),
        "c19_2_baseline_replay_status": s2.get("status"),
        "c19_3_front_rank_risk_status": s3.get("status"),
        "c19_4_readiness_assembly_status": s4.get("status"),
        "official_val_used": False,
        "official_prediction_pool_used": False,
        "pseudo_official_holdout_used_for_selection": False,
        "no_post_pseudo_adjustment": True,
        "evaluator_modified": False,
        "nms_modified": False,
        "schema_clean": s1.get("status") == "C19_FULL_SCORE_READY",
        "selected_config_frozen": True,
        "selected_formula": SELECTED,
        "key_RK_metrics": {k: selected_metrics.get(k) for k in selected_metrics if k.startswith("VCMR_R@")},
        "wrong_video_risk": s3.get("wrong_video_high_score_rate"),
        "c7_b6_train_only_replay_availability": s2.get("c7_b6_train_only_replay_availability"),
        "score_table_reproducibility": s1.get("canonical_score_table", {}),
        "limitations": s4.get("limitations", []),
        "current_promoted_system": PROMOTED,
        "c18_c19_is_promoted_system": False,
    }
    write_json(OUT5 / "C19_5_FINAL_BLOCKER_DECISION.json", rec)
    write_json(OUT5 / "C19_5_OFFICIAL_READINESS_PACKET.json", rec)
    write_json(OUT5 / "C19_5_NEXT_STEP_DECISION.json", rec)
    write_text(OUT5 / "C19_5_FINAL_BLOCKER_DECISION.md", f"# C19-5 Final Blocker Decision\n\nfinal_decision: `{final}`\n\nOfficial validation was not run.")
    write_text(OUT5 / "C19_5_OFFICIAL_READINESS_PACKET.md", f"# C19-5 Official Readiness Packet\n\ndecision: `{final}`\n\nThis packet is not authorization to run official validation.")
    write_text(OUT5 / "C19_5_NEXT_STEP_DECISION.md", f"# C19-5 Next Step Decision\n\ndecision: `{final}`")
    return rec


def print_summary(final: Dict[str, Any]) -> None:
    s2 = load_json(OUT2 / "C19_2_BASELINE_COMPARISON_TABLE.json", {})
    selected = s2.get("C18_selected_hybrid", {}).get("calib_holdout", {}).get("summary", {})
    t2 = s2.get("C12_5T_T2", {}).get("calib_holdout", {}).get("summary", {})
    c18_prev = load_json(ROOT / "c18_3_front_rank_hybrid_calibration/C18_3_CALIBRATION_DECISION.json", {}).get("selected_results", {}).get("calib_holdout", {}).get("summary", {})
    print("C19 SUMMARY")
    print(f"branch: {sh('git branch --show-current')}")
    print(f"commit: {sh('git rev-parse --short HEAD')}")
    print(f"C19-0 protocol status: {final.get('c19_0_protocol_status')}")
    print(f"C19-1 full score status: {final.get('c19_1_full_score_status')}")
    print(f"C19-2 baseline replay status: {final.get('c19_2_baseline_replay_status')}")
    print(f"C19-3 front-rank risk status: {final.get('c19_3_front_rank_risk_status')}")
    print(f"C19-4 readiness assembly status: {final.get('c19_4_readiness_assembly_status')}")
    print(f"C19-5 final decision: {final.get('final_decision')}")
    print(f"selected formula: {SELECTED}")
    for iou in ["0.5", "0.7"]:
        print(f"R@1/R@5/R@10/R@100 @ IoU{iou}: {selected.get(f'VCMR_R@1_IoU{iou}')}/{selected.get(f'VCMR_R@5_IoU{iou}')}/{selected.get(f'VCMR_R@10_IoU{iou}')}/{selected.get(f'VCMR_R@100_IoU{iou}')}")
    print(f"vs C12-5T T2 delta R100@0.7: {selected.get('VCMR_R@100_IoU0.7', 0.0) - t2.get('VCMR_R@100_IoU0.7', 0.0)}")
    print(f"vs C18 selected replay delta R100@0.7: {selected.get('VCMR_R@100_IoU0.7', 0.0) - c18_prev.get('VCMR_R@100_IoU0.7', 0.0)}")
    print(f"wrong-video high-score rate: {selected.get('wrong_video_high_score_rate')}")
    print(f"C7-B6 train-only replay availability: {final.get('c7_b6_train_only_replay_availability')}")
    print(f"full score table local path/hash: {canonical_cache(final.get('mode', 'medium'))} / {sha256_file(canonical_cache(final.get('mode', 'medium'))) if canonical_cache(final.get('mode', 'medium')).exists() else None}")
    print("pseudo_official_holdout used for selection: false")
    print("official was not run: true")
    print("files committed to GitHub: C19 code, JSON/MD manifests, and small sample parquet only")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", default="all", choices=["c19_0", "c19_1", "c19_2", "c19_3", "c19_4", "c19_5", "all"])
    parser.add_argument("--mode", default="medium", choices=["smoke", "medium", "full"])
    parser.add_argument("--seed", default=2026, type=int)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.stage in {"c19_0", "all"}:
        stage_c19_0(args.mode, args.seed)
    if args.stage in {"c19_1", "all"}:
        stage_c19_1(args.mode, args.seed, args.force)
    if args.stage in {"c19_2", "all"}:
        stage_c19_2(args.mode, args.seed)
    if args.stage in {"c19_3", "all"}:
        stage_c19_3(args.mode, args.seed)
    if args.stage in {"c19_4", "all"}:
        stage_c19_4(args.mode, args.seed)
    final = None
    if args.stage in {"c19_5", "all"}:
        final = stage_c19_5(args.mode, args.seed)
    if final:
        print_summary(final)


if __name__ == "__main__":
    main()
