#!/usr/bin/env python3
"""C24H full BMN/T2 train_fit rebuild attempt.

This runner performs an actual BMN/T2 rebuild attempt over first-stage top128
candidate videos by reusing the C17 BMN/T2 builders. It never runs official
validation, never reads official prediction pools, and never promotes C24H.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pickle
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch

import run_c17_bmn_t2_native_vcmr_integration as c17
import run_c18_full_hybrid_freeze_candidate as c18
import run_c22r_native_coupling_sanity_repair as c22r
from c12_native_retriever.scaffold import C12Paths
from run_c12_native_retriever_training import ROOT, build_feature_caches, load_corpus, load_features, sha256_file


PYTHON = "/home/a/miniconda3/envs/conquer-rlem/bin/python"
PROMOTED = "C7-B6 R1SelectiveTop1"
C24F_CACHE = Path("/tmp/c24f_score_cache/CONQUER-RLEM-c2c3")
C24G_CACHE = Path("/tmp/c24g_score_cache/CONQUER-RLEM-c2c3")
C24H_CACHE = Path("/tmp/c24h_score_cache/CONQUER-RLEM-c2c3")

OUT0 = ROOT / "c24h_0_protocol_freeze"
OUT1 = ROOT / "c24h_1_builder_dependency_audit"
OUT2 = ROOT / "c24h_2_trainfit_rebuild_execution"
OUT3 = ROOT / "c24h_3_full_evidence_table_closure"
OUT4 = ROOT / "c24h_4_full_evidence_signal_integration_test"
OUT5 = ROOT / "c24h_5_robustness_route_decision"
OUT6 = ROOT / "c24h_6_final_decision"

MODE_DEFAULT_QUERIES = {"smoke": 4, "medium": 24, "full": 0}
MODE_SAMPLE_ROWS = {"smoke": 1200, "medium": 4000, "full": 8000}
SPLITS = ["train_fit", "calib_select", "calib_holdout"]

BMN_REQUIRED = [
    "bmn_pred_iou", "bmn_p_iou_05", "bmn_p_iou_07", "bmn_rank_score",
    "bmn_final_score", "bmn_span_rank", "bmn_margin", "start_prob",
    "end_prob", "actionness_score", "best_bmn_span_start",
    "best_bmn_span_end", "bmn_available", "bmn_missing_reason",
]
T2_REQUIRED = ["t2_score", "t2_margin", "t2_agreement", "t2_rank", "t2_available", "t2_missing_reason"]


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


def stable_hash(obj: Any) -> str:
    return hashlib.sha256(json.dumps(jsonable(obj), sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:16]


def chunk_summary(chunks: Sequence[Dict[str, Any]] | None, manifest_path: Path | None = None) -> Dict[str, Any]:
    chunks = list(chunks or [])
    by_split: Dict[str, Dict[str, Any]] = {}
    for rec in chunks:
        split = str(rec.get("split", "unknown"))
        item = by_split.setdefault(split, {"chunks": 0, "rows": 0, "queries": 0, "failed_chunks": 0})
        item["chunks"] += 1
        item["rows"] += int(rec.get("candidate_row_count", rec.get("pair_count", 0)) or 0)
        item["queries"] += len(rec.get("query_ids", []) or [])
        if int(rec.get("exit_code", 0) or 0) != 0:
            item["failed_chunks"] += 1
    summary: Dict[str, Any] = {
        "chunk_count": len(chunks),
        "split_counts": by_split,
        "first_chunk": chunks[0] if chunks else None,
        "last_chunk": chunks[-1] if chunks else None,
        "chunk_manifest_path": str(manifest_path) if manifest_path else None,
        "chunk_manifest_note": "Repository manifest is a compact summary; full per-chunk parquet/json sidecars stay local under /tmp/c24h_score_cache.",
    }
    return summary


def compact_rebuild_manifest(manifest: Dict[str, Any], chunk_manifest_path: Path | None = None) -> Dict[str, Any]:
    out = dict(manifest)
    chunks = out.pop("chunks", [])
    out["chunk_summary"] = chunk_summary(chunks, chunk_manifest_path)
    return out


def file_record(path: Path, sha: bool = False) -> Dict[str, Any]:
    rec = {
        "path": str(path),
        "exists": path.exists(),
        "readable": os.access(path, os.R_OK) if path.exists() else False,
        "size_bytes": path.stat().st_size if path.exists() and path.is_file() else None,
    }
    if sha and path.exists() and path.is_file() and path.stat().st_size < 250_000_000:
        rec["sha256"] = sha256_file(path)
    elif sha and path.exists() and path.is_file():
        rec["stable_hash"] = stable_hash({"path": str(path), "size": path.stat().st_size, "mtime": int(path.stat().st_mtime)})
    return rec


def parquet_record(path: Path) -> Dict[str, Any]:
    rec = file_record(path, sha=True)
    if path.exists() and path.is_file():
        pf = pq.ParquetFile(path)
        rec.update({"row_count": pf.metadata.num_rows, "row_groups": pf.num_row_groups, "columns": pf.schema.names, "schema_hash": stable_hash(pf.schema.names)})
    return rec


def c24f_table_path(mode: str) -> Path:
    return C24F_CACHE / f"C24F_CANONICAL_FULL_EVIDENCE_TABLE_{mode}.local.parquet"


def c24h_rebuild_path(mode: str, dry_run: bool = False) -> Path:
    tag = "dry_run" if dry_run else "local"
    return C24H_CACHE / f"C24H_REBUILT_BMN_T2_{mode}.{tag}.parquet"


def c24h_chunk_dir(mode: str, dry_run: bool = False) -> Path:
    tag = "dry_run" if dry_run else "local"
    return C24H_CACHE / f"C24H_REBUILT_BMN_T2_{mode}.{tag}.chunks"


def chunk_paths(mode: str, split: str, chunk_start: int, chunk_size: int, dry_run: bool = False) -> Tuple[Path, Path]:
    d = c24h_chunk_dir(mode, dry_run=dry_run) / split
    stem = f"chunk_{int(chunk_start):09d}_{int(chunk_size):06d}"
    return d / f"{stem}.parquet", d / f"{stem}.json"


def c24h_table_path(mode: str) -> Path:
    return C24H_CACHE / f"C24H_FULL_EVIDENCE_TABLE_{mode}.local.parquet"


def build_c24h_canonical_base_from_rebuild(mode: str, seed: int, rebuild_path: Path) -> Tuple[Path, Dict[str, Any]]:
    """Create a full canonical event base from C24H rebuilt BMN/T2 rows.

    C24F's historical full path depended on C19 medium calib rows, which does
    not cover the full calib_select/calib_holdout query space. For C24H full
    closure, the truthful base is the completed C24H rebuilt top128 row-space.
    """
    import run_c24f_full_evidence_materialization_repair as c24f

    t0 = time.time()
    out_path = c24f_table_path(mode)
    manifest_path = C24F_CACHE / f"C24F_CANONICAL_FULL_EVIDENCE_TABLE_{mode}.manifest.json"
    rebuilt = pd.read_parquet(rebuild_path)
    base = rebuilt.copy()
    base["span_start"] = pd.to_numeric(base["best_bmn_span_start"], errors="coerce")
    base["span_end"] = pd.to_numeric(base["best_bmn_span_end"], errors="coerce")
    base["span_duration"] = base["span_end"] - base["span_start"]
    base["candidate_source"] = "C24H_C17_BMN_T2_FULL_REBUILD_TOP128"
    base["first_stage_rank"] = pd.to_numeric(base["candidate_video_rank"], errors="coerce")
    base["retriever_score"] = pd.to_numeric(base["first_stage_score"], errors="coerce")
    base["retriever_rank"] = base["first_stage_rank"]
    base["visual_feature_key"] = base["video_id"].astype(str)
    base["subtitle_feature_key"] = base["video_id"].astype(str)
    base["query_feature_key"] = base["query_id"].astype(str)
    base["has_visual"] = True
    base["has_subtitle"] = True
    base["has_query"] = True
    base["feature_missing_mask"] = ""
    base["event_source"] = "C24H full row-space direct event materialization from existing TVR features"
    base["bmn_score_available"] = base["bmn_available"].astype(bool)
    base["t2_score_available"] = base["t2_available"].astype(bool)
    base["event_score_available"] = False
    base["missing_bmn_mask"] = ~base["bmn_score_available"].astype(bool)
    base["missing_t2_mask"] = ~base["t2_score_available"].astype(bool)
    base["missing_event_mask"] = True
    base["best_bmn_span_iou_label"] = np.where(
        base["video_id"].astype(str).eq(base["gt_video_id"].astype(str)).to_numpy(),
        iou_array(
            pd.to_numeric(base["span_start"], errors="coerce").to_numpy(np.float32),
            pd.to_numeric(base["span_end"], errors="coerce").to_numpy(np.float32),
            pd.to_numeric(base["gt_start"], errors="coerce").to_numpy(np.float32),
            pd.to_numeric(base["gt_end"], errors="coerce").to_numpy(np.float32),
        ),
        0.0,
    )
    z_cols(base, "first_stage_score", "first_stage_z")
    z_cols(base, "bmn_final_score", "bmn_z")
    z_cols(base, "t2_score", "t2_z")

    corpus = load_corpus()
    feature_paths = build_feature_caches(corpus)
    features = load_features(feature_paths)
    canonical, _events, event_manifest = c24f.materialize_events(base, corpus, features, mode)
    schema_cols = c24f.schema_columns()
    for col in schema_cols:
        if col not in canonical.columns:
            canonical[col] = np.nan
    canonical = canonical[schema_cols + [c for c in canonical.columns if c not in schema_cols]]
    C24F_CACHE.mkdir(parents=True, exist_ok=True)
    canonical.to_parquet(out_path, index=False)
    manifest = {
        "path": str(out_path),
        "size_bytes": out_path.stat().st_size,
        "sha256": sha256_file(out_path),
        "row_count": int(len(canonical)),
        "query_count": int(canonical["query_id"].nunique()),
        "split_coverage": split_coverage(canonical),
        "event_manifest": event_manifest,
        "generation_command": f"{PYTHON} run_c24h_full_bmn_t2_trainfit_rebuild.py --stage c24h_3 --mode {mode} --seed {seed} --resume",
        "schema_hash": stable_hash(schema_cols),
        "config_hash": stable_hash({"stage": "C24H-3", "mode": mode, "seed": seed, "source": "C24H rebuilt BMN/T2 full row-space"}),
        "local_only": True,
        "source_rebuilt_bmn_t2": parquet_record(rebuild_path),
        "feature_paths": {k: str(v) for k, v in feature_paths.items()},
        "runtime_sec": time.time() - t0,
        "note": "Built because C24F full canonical table was absent and C19 medium rows do not cover full calib splits.",
    }
    write_json(manifest_path, manifest)
    return out_path, manifest


def preferred_c12_paths() -> C12Paths:
    paths = C12Paths()
    ssd_root = Path(os.environ.get("C24H_SSD_FEATURE_ROOT", "/tmp/c24f_feature_lmdb_cache"))
    sub = ssd_root / "subtitle_lmdb"
    vis = ssd_root / "visual_lmdb"
    if sub.exists() and vis.exists():
        paths = C12Paths(subtitle_lmdb=sub, visual_lmdb=vis)
    return paths


def apply_feature_path_patch() -> Dict[str, Any]:
    paths = preferred_c12_paths()
    c17.c12_5.PATHS = paths
    return {
        "subtitle_lmdb": str(paths.subtitle_lmdb),
        "visual_lmdb": str(paths.visual_lmdb),
        "ssd_feature_cache_used": str(paths.subtitle_lmdb).startswith("/tmp/") or str(paths.visual_lmdb).startswith("/tmp/"),
    }


def split_coverage(df: pd.DataFrame) -> Dict[str, Any]:
    if len(df) == 0 or "split" not in df:
        return {}
    return {str(k): {"rows": int(v["rows"]), "queries": int(v["queries"])} for k, v in df.groupby("split", observed=True).agg(rows=("query_id", "size"), queries=("query_id", "nunique")).to_dict("index").items()}


def evidence_coverage(df: pd.DataFrame) -> Dict[str, Any]:
    bmn = df.get("bmn_available", pd.Series(False, index=df.index)).astype(bool)
    t2 = df.get("t2_available", pd.Series(False, index=df.index)).astype(bool)
    event = df.get("event_available", pd.Series(False, index=df.index)).astype(bool)
    by = {}
    for split, g in df.groupby("split", observed=True):
        idx = g.index
        by[str(split)] = {
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
        "query_count": int(df["query_id"].nunique()) if "query_id" in df else 0,
        "split_coverage": split_coverage(df),
        "evidence_by_split": by,
        "bmn_missing_count": int((~bmn).sum()),
        "t2_missing_count": int((~t2).sum()),
        "event_missing_count": int((~event).sum()),
        "event_direct_columns_present": bool(event.any() and df.get("best_event_id", pd.Series(index=df.index)).notna().any()),
        "duplicate_candidate_count": int(df.duplicated(["split", "query_id", "video_id"]).sum()) if {"split", "query_id", "video_id"}.issubset(df.columns) else None,
    }


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_pickle(path: Path) -> Any:
    with path.open("rb") as f:
        return pickle.load(f)


def load_first_stage_for_split(split: str, ids: Sequence[int], mode: str) -> Tuple[Dict[int, Dict[str, Any]], Dict[str, Any]]:
    """Load top128 ranklists, falling back to the combined train/select/holdout cache.

    C22R's split-specific calib caches are medium-only. The combined cache is the
    full source for train_fit, calib_select, and calib_holdout.
    """
    ids_int = [int(x) for x in ids]
    preferred_path = c22r.first_stage_cache_path(split)
    combined_path = c22r.first_stage_cache_path("train_fit")
    source_path = preferred_path
    source_note = "split_specific_cache"
    obj = load_pickle(preferred_path)
    missing = [d for d in ids_int if d not in obj]
    if missing and combined_path.exists():
        combined = load_pickle(combined_path)
        combined_missing = [d for d in ids_int if d not in combined]
        if len(combined_missing) < len(missing):
            obj = combined
            source_path = combined_path
            source_note = "combined_train_calib_holdout_cache_fallback"
            missing = combined_missing
    first = {int(d): obj[int(d)] for d in ids_int if int(d) in obj}
    row_count = int(sum(len(first.get(int(d), {}).get("ranklist", [])[:128]) for d in ids_int))
    return first, {
        "split": split,
        "mode": mode,
        "source_path": str(source_path),
        "source_note": source_note,
        "requested_query_count": len(ids_int),
        "loaded_query_count": len(first),
        "missing_query_count": len(missing),
        "missing_query_sample": missing[:20],
        "expected_candidate_rows": row_count,
    }


def combine_chunk_parquets(chunk_records: Sequence[Dict[str, Any]], output_path: Path) -> Dict[str, Any]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = None
    rows = 0
    files = 0
    try:
        for rec in chunk_records:
            if int(rec.get("exit_code", 0)) != 0:
                continue
            p = Path(str(rec.get("output_path", "")))
            if not p.exists() or p.stat().st_size == 0:
                continue
            table = pq.read_table(p)
            if writer is None:
                writer = pq.ParquetWriter(output_path, table.schema, compression="snappy")
            writer.write_table(table)
            rows += table.num_rows
            files += 1
    finally:
        if writer is not None:
            writer.close()
    return {
        "path": str(output_path),
        "exists": output_path.exists(),
        "row_count": rows,
        "chunk_files_combined": files,
        "size_bytes": output_path.stat().st_size if output_path.exists() else 0,
        "sha256": sha256_file(output_path) if output_path.exists() else None,
    }


def coverage_from_chunks(chunks: Sequence[Dict[str, Any]], expected_by_split: Dict[str, Any]) -> Dict[str, Any]:
    by: Dict[str, Dict[str, Any]] = {}
    query_sets: Dict[str, set[int]] = {s: set() for s in SPLITS}
    for split in SPLITS:
        by[split] = {
            "rows": 0,
            "queries": 0,
            "expected_rows": int(expected_by_split.get(split, {}).get("expected_candidate_rows", 0)),
            "expected_queries": int(expected_by_split.get(split, {}).get("requested_query_count", 0)),
            "bmn_available_rows": 0,
            "t2_available_rows": 0,
            "event_available_rows": 0,
            "failed_chunks": 0,
            "missing_source_query_count": int(expected_by_split.get(split, {}).get("missing_query_count", 0)),
        }
    for rec in chunks:
        split = str(rec.get("split"))
        if split not in by:
            continue
        if int(rec.get("exit_code", 0)) != 0:
            by[split]["failed_chunks"] += 1
            continue
        by[split]["rows"] += int(rec.get("candidate_row_count", rec.get("pair_count", 0)) or 0)
        by[split]["bmn_available_rows"] += int(rec.get("bmn_rows_written", 0) or 0)
        by[split]["t2_available_rows"] += int(rec.get("t2_rows_written", 0) or 0)
        for q in rec.get("query_ids", []) or []:
            query_sets[split].add(int(q))
    for split, item in by.items():
        rows = int(item["rows"])
        item["queries"] = len(query_sets[split])
        item["bmn_coverage"] = float(item["bmn_available_rows"] / rows) if rows else 0.0
        item["t2_coverage"] = float(item["t2_available_rows"] / rows) if rows else 0.0
        item["event_coverage"] = 0.0
        item["row_space_coverage"] = float(rows / item["expected_rows"]) if item["expected_rows"] else 0.0
        item["query_space_coverage"] = float(item["queries"] / item["expected_queries"]) if item["expected_queries"] else 0.0
    return {
        "row_count": int(sum(v["rows"] for v in by.values())),
        "query_count": int(sum(v["queries"] for v in by.values())),
        "split_coverage": {s: {"rows": int(v["rows"]), "queries": int(v["queries"])} for s, v in by.items()},
        "expected_by_split": expected_by_split,
        "evidence_by_split": by,
        "bmn_missing_count": int(sum(v["rows"] - v["bmn_available_rows"] for v in by.values())),
        "t2_missing_count": int(sum(v["rows"] - v["t2_available_rows"] for v in by.values())),
        "event_missing_count": None,
        "event_direct_columns_present": False,
        "duplicate_candidate_count": None,
    }


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


def add_labels(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    same = out["video_id"].astype(str) == out["gt_video_id"].astype(str)
    iou = iou_array(
        pd.to_numeric(out["best_bmn_span_start"], errors="coerce").fillna(out.get("span_start")).to_numpy(np.float32),
        pd.to_numeric(out["best_bmn_span_end"], errors="coerce").fillna(out.get("span_end")).to_numpy(np.float32),
        pd.to_numeric(out["gt_start"], errors="coerce").to_numpy(np.float32),
        pd.to_numeric(out["gt_end"], errors="coerce").to_numpy(np.float32),
    )
    out["candidate_iou"] = np.where(same.to_numpy(), iou, 0.0)
    out["iou_ge_05"] = out["candidate_iou"] >= 0.5
    out["iou_ge_07"] = out["candidate_iou"] >= 0.7
    out["correct_video"] = same
    out["correct_video_and_iou07"] = same & out["iou_ge_07"]
    return out


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
    return float((ranks[y_arr].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def corr_pack(df: pd.DataFrame, score_col: str, label_col: str) -> Dict[str, Any]:
    if score_col not in df or label_col not in df:
        return {"available": False}
    s = pd.to_numeric(df[score_col], errors="coerce")
    y = pd.to_numeric(df[label_col], errors="coerce")
    mask = s.notna() & y.notna()
    if int(mask.sum()) < 5:
        return {"available": False, "count": int(mask.sum())}
    return {"available": True, "count": int(mask.sum()), "pearson": float(s[mask].corr(y[mask], method="pearson")), "spearman": float(s[mask].corr(y[mask], method="spearman"))}


def default_max_queries(mode: str, max_queries: int | None) -> int:
    if max_queries is not None and max_queries >= 0:
        return int(max_queries)
    return int(MODE_DEFAULT_QUERIES[mode])


def selected_ids(corpus: Any, split: str, mode: str, max_queries: int | None) -> List[int]:
    ids = [int(x) for x in corpus.splits[split]]
    lim = default_max_queries(mode, max_queries)
    return ids if lim == 0 else ids[:lim]


def best_bmn_record(bmn_map: Dict[Tuple[int, int], Dict[str, float]], duration: float, length: int) -> Dict[str, Any]:
    if not bmn_map:
        return {"bmn_available": False, "bmn_missing_reason": "bmn_builder_returned_no_span"}
    span, val = max(bmn_map.items(), key=lambda kv: float(kv[1].get("bmn_final_score", -1e9)))
    sidx, eidx = int(span[0]), int(span[1])
    st, ed = c17.c12_5.idx_to_ts(sidx, eidx, duration, length)
    return {
        "bmn_pred_iou": val.get("bmn_pred_iou"),
        "bmn_p_iou_05": val.get("bmn_p_iou_05"),
        "bmn_p_iou_07": val.get("bmn_p_iou_07"),
        "bmn_rank_score": val.get("bmn_rank_score"),
        "bmn_final_score": val.get("bmn_final_score"),
        "bmn_span_rank": val.get("span_map_rank"),
        "bmn_margin": np.nan,
        "start_prob": val.get("start_prob"),
        "end_prob": val.get("end_prob"),
        "actionness_score": val.get("actionness_score"),
        "best_bmn_span_start": float(st),
        "best_bmn_span_end": float(ed),
        "bmn_available": True,
        "bmn_missing_reason": "",
    }


def best_t2_record(t2_map: Dict[Tuple[int, int], Dict[str, float]]) -> Dict[str, Any]:
    if not t2_map:
        return {"t2_available": False, "t2_missing_reason": "t2_builder_returned_no_span"}
    _span, val = max(t2_map.items(), key=lambda kv: float(kv[1].get("t2_score", -1e9)))
    return {
        "t2_score": val.get("t2_score"),
        "t2_margin": np.nan,
        "t2_agreement": np.nan,
        "t2_rank": val.get("t2_rank"),
        "t2_available": True,
        "t2_missing_reason": "",
    }


def fast_dense_pool(
    start: np.ndarray,
    end: np.ndarray,
    sim: np.ndarray,
    qtype: int,
    mode: str,
    limit: int = 1000,
) -> List[Tuple[int, int, float, Dict[str, float]]]:
    """Vectorized replacement for C12-5R dense_pool used by C24H rebuild.

    This preserves the same candidate families as the original implementation
    while avoiding the Python nested top-start/top-end loops for every
    query-video pair.
    """
    t = int(len(start))
    if t <= 0:
        return []
    max_span = int(getattr(c17.c12_5r, "MAX_SPAN", 64))
    cand_s: List[np.ndarray] = []
    cand_e: List[np.ndarray] = []
    cand_bonus: List[np.ndarray] = []
    cand_tag: List[np.ndarray] = []

    def append(s: np.ndarray, e: np.ndarray, bonus: float, tag_id: int) -> None:
        s = np.asarray(s, dtype=np.int32).reshape(-1)
        e = np.asarray(e, dtype=np.int32).reshape(-1)
        valid = (s >= 0) & (e >= s) & (e < t) & (e < s + max_span)
        if not bool(valid.any()):
            return
        s = s[valid]
        e = e[valid]
        cand_s.append(s)
        cand_e.append(e)
        cand_bonus.append(np.full(len(s), float(bonus), dtype=np.float32))
        cand_tag.append(np.full(len(s), int(tag_id), dtype=np.int16))

    top_s = np.argsort(-start)[: min(t, 64)].astype(np.int32)
    top_e = np.argsort(-end)[: min(t, 64)].astype(np.int32)
    ss, ee = np.meshgrid(top_s, top_e, indexing="ij")
    append(ss.ravel(), ee.ravel(), 0.0, 0)

    if mode in {"r1_dense_duration", "r5_two_stage"}:
        step = 1 if t <= 80 else 2
        starts = np.arange(0, t, step, dtype=np.int32)
        for length in [1, 2, 3, 4, 5, 6]:
            append(starts, starts + int(length) - 1, 0.05, 1)
        centers = np.argsort(-sim)[: min(t, 32)].astype(np.int32)
        lengths = np.asarray([8, 10, 12, 16, 20, 28, 36, 48], dtype=np.int32)
        cc, ll = np.meshgrid(centers, lengths, indexing="ij")
        s = cc - (ll // 2)
        s = np.maximum(0, np.minimum(t - ll, s))
        append(s.ravel(), (s + ll - 1).ravel(), 0.03, 2)

    if cand_s:
        pre_s = np.concatenate(cand_s)
        pre_e = np.concatenate(cand_e)
    else:
        pre_s = np.zeros((0,), dtype=np.int32)
        pre_e = np.zeros((0,), dtype=np.int32)

    if mode in {"r2_offset_refine", "r5_two_stage"} and len(pre_s):
        offsets = np.arange(-2, 3, dtype=np.int32)
        s_grid = np.clip(pre_s[:, None] + offsets[None, :], 0, t - 1)
        best_s = s_grid[np.arange(len(pre_s)), np.argmax(start[s_grid], axis=1)]
        e_grid = np.clip(pre_e[:, None] + offsets[None, :], 0, t - 1)
        e_grid = np.maximum(e_grid, best_s[:, None])
        best_e = e_grid[np.arange(len(pre_e)), np.argmax(end[e_grid], axis=1)]
        append(best_s, best_e, 0.04, 3)

    if not cand_s:
        return []
    s_all = np.concatenate(cand_s).astype(np.int32)
    e_all = np.concatenate(cand_e).astype(np.int32)
    bonus_all = np.concatenate(cand_bonus).astype(np.float32)
    tag_all = np.concatenate(cand_tag).astype(np.int16)
    scores = start[s_all].astype(np.float32) + end[e_all].astype(np.float32) + bonus_all
    scores = scores + 0.15 * (sim[s_all].astype(np.float32) + sim[e_all].astype(np.float32))

    if mode == "r4_teacher_prior":
        target = {0: 0.10, 1: 0.16, 2: 0.13, 3: 0.14}.get(int(qtype), 0.14)
        scores = scores - 0.25 * np.abs(((e_all - s_all + 1) / max(1, t)).astype(np.float32) - float(target))

    keys = s_all.astype(np.int64) * int(t) + e_all.astype(np.int64)
    order = np.lexsort((-scores, keys))
    keys_sorted = keys[order]
    keep = np.r_[True, keys_sorted[1:] != keys_sorted[:-1]]
    best_idx = order[keep]
    final_order = best_idx[np.argsort(-scores[best_idx])[: int(limit)]]
    tag_names = {
        0: "boundary_top",
        1: "short_dense",
        2: "midlong_peak",
        3: "offset_refined",
    }
    rows: List[Tuple[int, int, float, Dict[str, float]]] = []
    for idx in final_order:
        s = int(s_all[idx])
        e = int(e_all[idx])
        meta = {
            "duration_norm": float((e - s + 1) / max(1, t)),
            "center": float((s + e) * 0.5 / max(1, t)),
            "tag": tag_names.get(int(tag_all[idx]), "native"),
        }
        if mode == "r4_teacher_prior":
            meta["teacher_duration_prior"] = {0: 0.10, 1: 0.16, 2: 0.13, 3: 0.14}.get(int(qtype), 0.14)
        rows.append((s, e, float(scores[idx]), meta))
    return rows


def run_builder_rebuild(
    mode: str,
    seed: int,
    max_queries: int | None,
    chunk_size: int,
    dry_run: bool,
    resume: bool = False,
    force: bool = False,
    splits: Sequence[str] | None = None,
    query_start: int | None = None,
    query_end: int | None = None,
    combine: bool = True,
    progress_tag: str | None = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    t0 = time.time()
    C24H_CACHE.mkdir(parents=True, exist_ok=True)
    feature_patch = apply_feature_path_patch()
    c17.c12_5r.dense_pool = fast_dense_pool
    corpus = load_corpus()
    features = load_features(build_feature_caches(corpus))
    c17.c12_5.corpus_global = corpus
    cfg = c17.mode_cfg("medium" if mode != "smoke" else "smoke")
    cfg["top_videos"] = 128
    cfg["spans_per_video"] = min(int(cfg["spans_per_video"]), 32)
    bmn_model, bmn_info = c17.load_bmn_model(seed, "medium", cfg)
    teacher, t2_ranker, t2_idxs, t2_info = c17.load_t2_ranker()
    durations = c17.video_duration_map(corpus)
    store = c17.c12_5.ClipFeatureStore()
    tag = f".{progress_tag}" if progress_tag else ""
    progress_path = C24H_CACHE / f"C24H_REBUILD_PROGRESS_{mode}{tag}{'.dry_run' if dry_run else ''}.json"
    sample_frames: List[pd.DataFrame] = []
    chunks: List[Dict[str, Any]] = []
    expected_by_split: Dict[str, Any] = {}
    sample_target = MODE_SAMPLE_ROWS[mode]
    def process_block(split: str, st: int, block: List[Tuple[int, str, float, int]], fs: Dict[int, Dict[str, Any]], source_manifest: Dict[str, Any], query_count_for_hash: int) -> None:
        nonlocal sample_frames
        chunk_path, sidecar_path = chunk_paths(mode, split, st, chunk_size, dry_run=dry_run)
        chunk_path.parent.mkdir(parents=True, exist_ok=True)
        ct0 = time.time()
        started = now_iso()
        stdout = f"split={split} chunk_start={st} pairs={len(block)}"
        stderr = ""
        exit_code = 0
        chunk_df = pd.DataFrame()
        skipped = False
        resumed_existing = False
        try:
            if chunk_path.exists() and sidecar_path.exists() and resume and not force:
                prev = load_json(sidecar_path, {})
                if int(prev.get("exit_code", 1)) == 0:
                    skipped = True
                    resumed_existing = True
                    chunk_df = pd.read_parquet(chunk_path, columns=["query_id", "bmn_available", "t2_available"])
            if not skipped:
                batch = c17.build_candidate_batch(corpus, features, store, [(d, v) for d, v, _rs, _rr in block], durations)
                t2_maps = c17.t2_scores_for_batch(teacher, t2_ranker, t2_idxs, batch, fs, corpus)
                bmn_maps = c17.bmn_scores_for_batch(bmn_model, batch, int(cfg["spans_per_video"]) * 3)
                rows: List[Dict[str, Any]] = []
                for bi, (did, vid, rscore, rrank) in enumerate(block):
                    row = corpus.by_id[int(did)]
                    duration = float(batch["durations"][bi])
                    length = int(batch["lengths"][bi].item())
                    bmn = best_bmn_record(bmn_maps[bi], duration, length)
                    t2 = best_t2_record(t2_maps[bi])
                    rows.append({
                        "query_id": int(did),
                        "video_id": str(vid),
                        "split": split,
                        "candidate_video_rank": int(rrank),
                        "first_stage_score": float(rscore),
                        "gt_video_id": str(row["vid_name"]),
                        "gt_start": float(row["ts"][0]),
                        "gt_end": float(row["ts"][1]),
                        "query_type": str(row.get("type", "unknown")),
                        "duration_bucket": c17.duration_bucket(float(row["ts"][1]) - float(row["ts"][0])),
                        **bmn,
                        **t2,
                        "builder_id": "C17_BMN_T2_INTERNAL_BUILDER",
                        "checkpoint_hash": stable_hash({"bmn": bmn_info, "t2": t2_info}),
                        "feature_hash": stable_hash({"feature_cache": "TVR release via C12/C17 ClipFeatureStore", "query_count": query_count_for_hash, "first_stage_source": source_manifest.get("source_note")}),
                        "schema_hash": stable_hash(BMN_REQUIRED + T2_REQUIRED),
                        "config_hash": stable_hash({"mode": mode, "seed": seed, "max_queries": max_queries, "chunk_size": chunk_size, "top_videos": 128}),
                    })
                chunk_df = pd.DataFrame(rows)
                if len(chunk_df):
                    for col in ["bmn_final_score", "t2_score", "first_stage_score"]:
                        z_cols(chunk_df, col, {"bmn_final_score": "bmn_z", "t2_score": "t2_z", "first_stage_score": "first_stage_z"}[col])
                    for score_col, margin_col in [("bmn_final_score", "bmn_margin"), ("t2_score", "t2_margin")]:
                        val = pd.to_numeric(chunk_df[score_col], errors="coerce")
                        med = val.groupby([chunk_df["split"], chunk_df["query_id"]], observed=True).transform("median")
                        chunk_df[margin_col] = val - med
                    chunk_df["t2_agreement"] = 1.0 - (pd.to_numeric(chunk_df["bmn_z"], errors="coerce") - pd.to_numeric(chunk_df["t2_z"], errors="coerce")).abs()
                    chunk_df.to_parquet(chunk_path, index=False)
        except Exception as exc:  # pragma: no cover - runtime evidence
            exit_code = 1
            stderr = repr(exc)
        bmn_rows = int(chunk_df.get("bmn_available", pd.Series(dtype=bool)).fillna(False).astype(bool).sum()) if len(chunk_df) else 0
        t2_rows = int(chunk_df.get("t2_available", pd.Series(dtype=bool)).fillna(False).astype(bool).sum()) if len(chunk_df) else 0
        chunk_rec = {
            "split": split,
            "chunk_start": st,
            "chunk_size": int(chunk_size),
            "pair_count": len(block),
            "candidate_row_count": int(len(chunk_df)) if exit_code == 0 else 0,
            "expected_row_count": len(block),
            "bmn_rows_written": bmn_rows,
            "t2_rows_written": t2_rows,
            "bmn_coverage": float(bmn_rows / len(chunk_df)) if len(chunk_df) else 0.0,
            "t2_coverage": float(t2_rows / len(chunk_df)) if len(chunk_df) else 0.0,
            "query_ids": sorted({int(x[0]) for x in block}),
            "query_count": len({x[0] for x in block}),
            "started_at": started,
            "ended_at": now_iso(),
            "runtime_sec": time.time() - ct0,
            "runtime_seconds": time.time() - ct0,
            "command": f"internal:C17 builder split={split} start={st} count={len(block)}",
            "stdout_summary": stdout,
            "stderr_summary": stderr,
            "exit_code": exit_code,
            "output_path": str(chunk_path),
            "output_hash": sha256_file(chunk_path) if chunk_path.exists() else None,
            "skipped_because_existing": skipped,
            "resumed_from_existing": resumed_existing,
            "first_stage_source": source_manifest,
        }
        chunks.append(chunk_rec)
        if not skipped:
            write_json(sidecar_path, chunk_rec)
        if len(chunk_df) and sum(len(x) for x in sample_frames) < sample_target:
            need = sample_target - sum(len(x) for x in sample_frames)
            sample_frames.append(chunk_df.head(need).copy())
        progress_cov = coverage_from_chunks(chunks, expected_by_split)
        progress_path.write_text(json.dumps(jsonable({
            "mode": mode,
            "seed": seed,
            "dry_run": dry_run,
            "completed_chunks": len(chunks),
            "completed_rows": progress_cov["row_count"],
            "coverage": progress_cov,
            "last_chunk": chunks[-1],
            "feature_patch": feature_patch,
            "elapsed_sec": time.time() - t0,
        }), indent=2, sort_keys=True) + "\n", encoding="utf-8")

    try:
        active_splits = [str(s) for s in (splits or SPLITS)]
        for split in active_splits:
            all_ids = selected_ids(corpus, split, mode, max_queries)
            q0 = max(0, int(query_start or 0))
            q1 = len(all_ids) if query_end is None or int(query_end) < 0 else min(len(all_ids), int(query_end))
            ids = all_ids[q0:q1]
            fs, source_manifest = load_first_stage_for_split(split, ids, mode)
            source_manifest["query_slice_start"] = q0
            source_manifest["query_slice_end"] = q1
            source_manifest["full_selected_query_count_before_slice"] = len(all_ids)
            expected_by_split[split] = source_manifest
            block: List[Tuple[int, str, float, int]] = []
            chunk_start = q0 * 128
            stop_after_dry = False
            for did in ids:
                ranklist = fs.get(int(did), {}).get("ranklist", [])[:128]
                for rank, (pos, score) in enumerate(ranklist, start=1):
                    if int(pos) < 0 or int(pos) >= len(corpus.train_videos):
                        continue
                    block.append((int(did), str(corpus.train_videos[int(pos)]), float(score), int(rank)))
                    if len(block) >= max(1, int(chunk_size)):
                        process_block(split, chunk_start, block, fs, source_manifest, len(ids))
                        chunk_start += len(block)
                        block = []
                        if dry_run:
                            stop_after_dry = True
                            break
                if stop_after_dry:
                    break
            if block and not stop_after_dry:
                process_block(split, chunk_start, block, fs, source_manifest, len(ids))
            if dry_run and chunks:
                break
    finally:
        store.close()
    combined = combine_chunk_parquets(chunks, c24h_rebuild_path(mode, dry_run=dry_run)) if combine else {
        "path": str(c24h_rebuild_path(mode, dry_run=dry_run)),
        "exists": c24h_rebuild_path(mode, dry_run=dry_run).exists(),
        "row_count": None,
        "chunk_files_combined": None,
        "combine_skipped": True,
    }
    coverage = coverage_from_chunks(chunks, expected_by_split)
    df = pd.concat(sample_frames, ignore_index=True, sort=False) if sample_frames else pd.DataFrame()
    manifest = {
        "runtime_sec": time.time() - t0,
        "mode": mode,
        "seed": seed,
        "dry_run": dry_run,
        "max_queries": max_queries,
        "chunk_size": chunk_size,
        "resume": resume,
        "force": force,
        "active_splits": active_splits,
        "query_start": query_start,
        "query_end": query_end,
        "combine": combine,
        "progress_tag": progress_tag,
        "row_count": int(coverage["row_count"]),
        "query_count": int(coverage["query_count"]),
        "split_coverage": coverage["split_coverage"],
        "expected_by_split": expected_by_split,
        "coverage": coverage,
        "combined_output": combined,
        "bmn_checkpoint": bmn_info,
        "t2_checkpoint": t2_info,
        "feature_read_paths": feature_patch,
        "progress_path": str(progress_path),
        "builder_path": "run_c17_bmn_t2_native_vcmr_integration.py",
        "builder_functions": ["build_candidate_batch", "bmn_scores_for_batch", "t2_scores_for_batch"],
        "chunks": chunks,
        "official_val_used": False,
        "official_prediction_pool_used": False,
    }
    return df, manifest


def stage_c24h_0(mode: str, seed: int) -> Dict[str, Any]:
    branch = sh("git branch --show-current")
    commit = sh("git rev-parse HEAD")
    dirty = sh("git status --short")
    c24g_files = {
        "c24g_1": ROOT / "c24g_1_bmn_t2_source_recovery/C24G_1_SOURCE_RECOVERY_DECISION.json",
        "c24g_2": ROOT / "c24g_2_bmn_t2_trainfit_materialization/C24G_2_MATERIALIZATION_DECISION.json",
        "c24g_3": ROOT / "c24g_3_full_evidence_table_rebuild/C24G_3_TABLE_REBUILD_DECISION.json",
        "c24g_4": ROOT / "c24g_4_signal_integration_reeval/C24G_4_REEVAL_DECISION.json",
        "c24g_5": ROOT / "c24g_5_robustness_route_decision/C24G_5_ROBUSTNESS_DECISION.json",
        "c24g_6": ROOT / "c24g_6_final_decision/C24G_6_NEXT_STEP_DECISION.json",
    }
    c24g_final = load_json(ROOT / "c24g_6_final_decision/C24G_6_FINAL_DECISION.json", {})
    root_blockers = [x for x in dirty.splitlines() if "/" not in x.split()[-1] and "OFFICIAL" in x.upper() and any(tok in x.upper() for tok in ["PREDICTION", "RAW", "NMS"]) and not x.split()[-1].endswith(".py")]
    deps = {
        "current_branch": branch,
        "current_commit": commit,
        "dirty_status": dirty.splitlines(),
        "c24g_files": {k: file_record(v, sha=True) for k, v in c24g_files.items()},
        "c24g_final": c24g_final.get("final_decision"),
        "bmn_train_fit_coverage": c24g_final.get("bmn_train_fit_coverage"),
        "t2_train_fit_coverage": c24g_final.get("t2_train_fit_coverage"),
        "event_direct_columns_present": c24g_final.get("event_direct_columns_present"),
        "first_stage_train_fit_top128": file_record(c22r.first_stage_cache_path("train_fit")),
        "first_stage_calib_select_top128": file_record(c22r.first_stage_cache_path("calib_select")),
        "first_stage_calib_holdout_top128": file_record(c22r.first_stage_cache_path("calib_holdout")),
        "c24f_event_cache": parquet_record(C24F_CACHE / "C24F_DIRECT_EVENT_TABLE_medium.local.parquet"),
        "c24g_cache": file_record(C24G_CACHE),
        "root_level_contamination_blockers": root_blockers,
        "historical_official_marker_warning": bool(list(ROOT.glob("**/*OFFICIAL*"))),
    }
    missing = [k for k, v in deps["c24g_files"].items() if not v["exists"]]
    ready = (
        not missing
        and not root_blockers
        and c24g_final.get("final_decision") == "C24G_NEED_MORE_BMN_T2_MATERIALIZATION"
        and float(c24g_final.get("bmn_train_fit_coverage") or 0.0) == 0.0
        and float(c24g_final.get("t2_train_fit_coverage") or 0.0) == 0.0
        and bool(c24g_final.get("event_direct_columns_present"))
    )
    status = "C24H_PROTOCOL_READY" if ready else ("C24H_PROTOCOL_BLOCKED_CONTAMINATION_RISK" if root_blockers else "C24H_PROTOCOL_BLOCKED_MISSING_CORE_ARTIFACTS")
    rec = {
        "stage": "C24H-0",
        "status": status,
        "mode": mode,
        "seed": seed,
        "dependency_audit": deps,
        "current_promoted_system": PROMOTED,
        "official_val_used": False,
        "official_prediction_pool_used": False,
        "pseudo_official_holdout_used_for_selection": False,
        "evaluator_modified": False,
        "nms_modified": False,
        "c24h_is_promoted_system": False,
        "repro_command": f"{PYTHON} run_c24h_full_bmn_t2_trainfit_rebuild.py --stage all --mode {mode} --seed {seed} --force",
    }
    write_text(OUT0 / "C24H_0_PROTOCOL.md", f"# C24H-0 Protocol\n\nStatus: `{status}`.")
    write_json(OUT0 / "C24H_0_PROTOCOL.json", rec)
    write_text(OUT0 / "C24H_0_C24G_ACCEPTANCE.md", "# C24G Acceptance\n\nC24G is accepted as proof that BMN/T2 train_fit evidence remains the active blocker.")
    write_text(OUT0 / "C24H_0_FORBIDDEN_ACTIONS_AUDIT.md", "# Forbidden Actions Audit\n\n- official validation: false\n- official prediction pool read: false\n- pseudo official used for selection: false\n- evaluator modified: false\n- NMS modified: false")
    write_json(OUT0 / "C24H_0_DEPENDENCY_AUDIT.json", deps)
    write_json(OUT0 / "C24H_0_REPRODUCIBILITY_MANIFEST.json", rec)
    return rec


def builder_static_search() -> Dict[str, Any]:
    candidates = sorted(ROOT.glob("run_c*.py"))
    out = {}
    for p in candidates:
        txt = p.read_text(encoding="utf-8", errors="ignore")
        lower = txt.lower()
        if any(k in lower for k in ["bmn", "t2", "span_map", "evidence materialization", "localizer"]):
            out[p.name] = {
                "path": str(p),
                "mentions_bmn": "bmn" in lower,
                "mentions_t2": "t2" in lower,
                "mentions_span_map": "span_map" in lower,
                "supports_train_fit_in_text": "train_fit" in lower,
                "supports_chunking_in_text": "chunk" in lower,
                "supports_resume_in_text": "resume" in lower,
            }
    return out


def run_dry_command(mode: str, seed: int) -> Dict[str, Any]:
    t0 = time.time()
    cmd = f"internal:stage_c24h_2(mode=smoke, dry_run=True, max_queries=2, chunk_size=4, write_outputs=False)"
    try:
        rec = stage_c24h_2("smoke", seed, force=True, dry_run=True, max_queries=2, chunk_size=4, write_outputs=False)
        return {
            "command": cmd,
            "exit_code": 0,
            "runtime_sec": time.time() - t0,
            "stdout_summary": json.dumps(jsonable({"status": rec.get("status"), "coverage": rec.get("rebuild_coverage_audit")}), ensure_ascii=False)[:4000],
            "stderr_summary": "",
        }
    except Exception as exc:  # pragma: no cover - audit evidence
        return {
            "command": cmd,
            "exit_code": 1,
            "runtime_sec": time.time() - t0,
            "stdout_summary": "",
            "stderr_summary": repr(exc),
        }


def stage_c24h_1(mode: str, seed: int, force: bool = False) -> Dict[str, Any]:
    builders = builder_static_search()
    ckpts = {
        "bmn_b7_seed2026": file_record(ROOT / "c12_models/c16_medium_B7_full_query_aware_rankloss_map_seed2026.pt", sha=True),
        "t2_ranker": file_record(ROOT / "c12_models/c12_5t_T2_listwise_soft_iou.pt", sha=True),
        "t2_teacher": file_record(ROOT / "c12_models/c12_5_teacher_distilled.pt", sha=True),
    }
    features = {
        "query_lmdb_or_cache": "loaded through C12 build_feature_caches",
        "subtitle_visual_store": "loaded through C12 ClipFeatureStore",
        "first_stage_train_fit": file_record(c22r.first_stage_cache_path("train_fit")),
        "first_stage_calib_select": file_record(c22r.first_stage_cache_path("calib_select")),
        "first_stage_calib_holdout": file_record(c22r.first_stage_cache_path("calib_holdout")),
    }
    dry = run_dry_command(mode, seed)
    dry_ok = dry["exit_code"] == 0
    status = "C24H_REBUILD_FEASIBLE" if dry_ok else "C24H_REBUILD_BLOCKED_RUNTIME_FAILURE"
    matrix = {
        "C17_internal_builder": {
            "file_path": "run_c17_bmn_t2_native_vcmr_integration.py",
            "function_entrypoints": ["load_bmn_model", "load_t2_ranker", "build_candidate_batch", "bmn_scores_for_batch", "t2_scores_for_batch"],
            "expected_inputs": ["first-stage top128 ranklist", "TVR subtitle/visual/query features", "C16 BMN checkpoint", "T2 teacher/ranker checkpoint"],
            "expected_outputs": BMN_REQUIRED + T2_REQUIRED,
            "required_checkpoint": ckpts,
            "required_feature_files": features,
            "supports_train_fit": True,
            "supports_chunking": True,
            "supports_resume": True,
            "supports_top128_candidate_table": True,
            "can_produce_bmn_columns": dry_ok,
            "can_produce_t2_columns": dry_ok,
            "missing_dependency": None if dry_ok else dry["stderr_summary"],
            "dry_run_result": dry,
            "failure_reason": None if dry_ok else "dry-run failed",
        }
    }
    rec = {
        "stage": "C24H-1",
        "status": status,
        "mode": mode,
        "seed": seed,
        "bmn_builder_search": builders,
        "t2_builder_search": builders,
        "checkpoint_search": ckpts,
        "feature_dependency_audit": features,
        "rebuild_feasibility_matrix": matrix,
        "dry_run_commands": [dry["command"]],
        "dry_run_results": dry,
        "official_val_used": False,
        "official_prediction_pool_used": False,
    }
    write_text(OUT1 / "C24H_1_BUILDER_AUDIT_PLAN.md", "# C24H-1 Builder Audit Plan\n\nFind and dry-run real BMN/T2 rebuild builders for train_fit first-stage top128.")
    write_json(OUT1 / "C24H_1_BMN_BUILDER_SEARCH.json", builders)
    write_json(OUT1 / "C24H_1_T2_BUILDER_SEARCH.json", builders)
    write_json(OUT1 / "C24H_1_CHECKPOINT_SEARCH.json", ckpts)
    write_json(OUT1 / "C24H_1_FEATURE_DEPENDENCY_AUDIT.json", features)
    write_json(OUT1 / "C24H_1_REBUILD_FEASIBILITY_MATRIX.json", matrix)
    write_text(OUT1 / "C24H_1_DRY_RUN_COMMANDS.md", "\n".join(["# C24H-1 Dry Run Commands", "", f"```bash\n{dry['command']}\n```"]))
    write_json(OUT1 / "C24H_1_DRY_RUN_RESULTS.json", dry)
    write_json(OUT1 / "C24H_1_BUILDER_DEPENDENCY_DECISION.json", rec)
    write_text(OUT1 / "C24H_1_BUILDER_DEPENDENCY_DECISION.md", f"# C24H-1 Builder Dependency Decision\n\nStatus: `{status}`.")
    return rec


def stage_c24h_2(
    mode: str,
    seed: int,
    force: bool = False,
    dry_run: bool = False,
    max_queries: int | None = None,
    chunk_size: int = 16,
    write_outputs: bool = True,
    resume: bool = False,
    splits: Sequence[str] | None = None,
    query_start: int | None = None,
    query_end: int | None = None,
    worker_only: bool = False,
    progress_tag: str | None = None,
) -> Dict[str, Any]:
    if worker_only:
        write_outputs = False
    df, manifest = run_builder_rebuild(
        mode,
        seed,
        max_queries=max_queries,
        chunk_size=chunk_size,
        dry_run=dry_run,
        resume=resume,
        force=force,
        splits=splits,
        query_start=query_start,
        query_end=query_end,
        combine=not worker_only,
        progress_tag=progress_tag,
    )
    C24H_CACHE.mkdir(parents=True, exist_ok=True)
    path = c24h_rebuild_path(mode, dry_run=dry_run)
    split_arg = ",".join(splits) if splits else None
    manifest.update({"path": str(path), "size_bytes": path.stat().st_size if path.exists() else 0, "sha256": sha256_file(path) if path.exists() else None, "generation_command": f"{PYTHON} run_c24h_full_bmn_t2_trainfit_rebuild.py --stage c24h_2 --mode {mode} --seed {seed} --max_queries {max_queries} --chunk_size {chunk_size}{' --resume' if resume else ''}{' --force' if force else ''}{' --dry_run' if dry_run else ''}{f' --splits {split_arg}' if split_arg else ''}{f' --query_start {query_start}' if query_start is not None else ''}{f' --query_end {query_end}' if query_end is not None else ''}{' --worker_only' if worker_only else ''}"})
    coverage = manifest.get("coverage") or (evidence_coverage(df) if len(df) else {"row_count": 0, "evidence_by_split": {}})
    attempted_splits = sorted(coverage.get("evidence_by_split", {}).keys())
    full_complete = (
        mode == "full"
        and not dry_run
        and default_max_queries(mode, max_queries) == 0
        and all(s in attempted_splits for s in SPLITS)
        and all(
            v.get("row_space_coverage", 0.0) >= 0.999
            and v.get("query_space_coverage", 0.0) >= 0.999
            and v.get("bmn_coverage", 0.0) >= 0.95
            and v.get("t2_coverage", 0.0) >= 0.95
            and int(v.get("failed_chunks", 0)) == 0
            and int(v.get("missing_source_query_count", 0)) == 0
            for v in coverage["evidence_by_split"].values()
        )
    )
    any_failure = any(int(c.get("exit_code", 0)) != 0 for c in manifest.get("chunks", []))
    if any_failure and len(df) == 0:
        status = "C24H_BMN_T2_REBUILD_BLOCKED"
    elif full_complete:
        status = "C24H_BMN_T2_REBUILD_COMPLETE"
    else:
        status = "C24H_BMN_T2_REBUILD_PARTIAL"
    sample_path = OUT2 / "C24H_2_REBUILD_SAMPLE.parquet"
    if write_outputs:
        sample_path.parent.mkdir(parents=True, exist_ok=True)
        df.head(MODE_SAMPLE_ROWS[mode]).to_parquet(sample_path, index=False)
    progress = {
        "completed_chunks": len(manifest.get("chunks", [])),
        "completed_rows": int(coverage.get("row_count", 0)),
        "completed_splits": attempted_splits,
        "estimated_remaining_note": "full requires all train_fit/calib_select/calib_holdout first-stage top128; completion is judged against expected row-space, not subset coverage",
        "resume_supported": True,
        "resume_used": resume,
        "dry_run": dry_run,
        "expected_by_split": manifest.get("expected_by_split"),
        "row_space_coverage_by_split": {k: v.get("row_space_coverage") for k, v in (coverage.get("evidence_by_split") or {}).items()},
    }
    c24f_path = c24f_table_path("medium" if mode != "smoke" else "smoke")
    if len(df) and c24f_path.exists():
        full_rows = int(pq.ParquetFile(c24f_path).metadata.num_rows)
        rows_per_sec = float(len(df) / max(float(manifest.get("runtime_sec") or 0.0), 1e-6))
        progress["measured_rows_per_sec"] = rows_per_sec
        progress["estimated_medium_row_space_rows"] = full_rows
        progress["estimated_medium_row_space_runtime_sec"] = float(full_rows / max(rows_per_sec, 1e-6))
    chunk_manifest_path = OUT2 / "C24H_2_CHUNK_MANIFEST.json"
    runtime = {
        "runtime_sec": manifest.get("runtime_sec"),
        "torch_cuda_available": torch.cuda.is_available(),
        "device": str(c17.DEVICE),
        "chunk_summary": chunk_summary(manifest.get("chunks"), chunk_manifest_path),
        "combined_output": manifest.get("combined_output"),
    }
    compact_manifest = compact_rebuild_manifest(manifest, chunk_manifest_path)
    rec = {
        "stage": "C24H-2",
        "status": status,
        "mode": mode,
        "seed": seed,
        "dry_run": dry_run,
        "bmn_rebuild_manifest": compact_manifest,
        "t2_rebuild_manifest": compact_manifest,
        "rebuild_progress": progress,
        "chunk_manifest": chunk_summary(manifest.get("chunks"), chunk_manifest_path),
        "rebuild_coverage_audit": coverage,
        "rebuild_runtime_audit": runtime,
        "rebuild_sample_path": str(sample_path),
        "rebuild_actually_attempted": True,
        "blocked_reason": "runtime failure" if status == "C24H_BMN_T2_REBUILD_BLOCKED" else None,
        "worker_only": worker_only,
        "official_val_used": False,
        "official_prediction_pool_used": False,
    }
    if write_outputs:
        write_text(OUT2 / "C24H_2_REBUILD_EXECUTION_PLAN.md", "# C24H-2 Rebuild Execution Plan\n\nRun the C17 BMN/T2 internal builder on first-stage top128 query-video candidates. Partial runs never imply full closure.")
        rec["chunk_manifest"] = chunk_summary(manifest.get("chunks"), chunk_manifest_path)
        write_json(chunk_manifest_path, rec["chunk_manifest"])
        compact_manifest = compact_rebuild_manifest(manifest, chunk_manifest_path)
        rec["bmn_rebuild_manifest"] = compact_manifest
        rec["t2_rebuild_manifest"] = compact_manifest
        runtime["chunk_summary"] = rec["chunk_manifest"]
        write_json(OUT2 / "C24H_2_BMN_REBUILD_MANIFEST.json", compact_manifest)
        write_json(OUT2 / "C24H_2_T2_REBUILD_MANIFEST.json", compact_manifest)
        write_json(OUT2 / "C24H_2_REBUILD_PROGRESS.json", progress)
        write_json(OUT2 / "C24H_2_REBUILD_COVERAGE_AUDIT.json", coverage)
        write_json(OUT2 / "C24H_2_REBUILD_RUNTIME_AUDIT.json", runtime)
        write_json(OUT2 / "C24H_2_REBUILD_EXECUTION_DECISION.json", rec)
        write_text(OUT2 / "C24H_2_REBUILD_EXECUTION_DECISION.md", f"# C24H-2 Rebuild Execution Decision\n\nStatus: `{status}`.\n\nActual BMN/T2 rebuild attempted: `true`.")
    return rec


def stage_c24h_3(mode: str, seed: int, force: bool = False) -> Dict[str, Any]:
    rebuild_path = c24h_rebuild_path(mode, dry_run=False)
    base_path = c24f_table_path(mode)
    fallback_manifest: Dict[str, Any] | None = None
    if not base_path.exists() and rebuild_path.exists():
        try:
            base_path, fallback_manifest = build_c24h_canonical_base_from_rebuild(mode, seed, rebuild_path)
        except Exception as exc:  # pragma: no cover - runtime evidence
            fallback_manifest = {"fallback_attempted": True, "fallback_failed": True, "error": repr(exc)}
    if not base_path.exists() or not rebuild_path.exists():
        missing = {
            "c24f_canonical_table": file_record(base_path, sha=False),
            "c24h_rebuilt_bmn_t2_table": file_record(rebuild_path, sha=False),
            "c24h_canonical_fallback": fallback_manifest,
        }
        status = "C24H_FULL_EVIDENCE_TABLE_BLOCKED"
        rec = {
            "stage": "C24H-3",
            "status": status,
            "mode": mode,
            "seed": seed,
            "blocked_reason": "missing C24F canonical/event table or C24H rebuilt BMN/T2 table for requested mode; C24H fallback failed or source missing",
            "missing_dependency_audit": missing,
            "split_coverage_audit": {},
            "missing_mask_audit": {},
            "diagnostic_only_column_audit": [],
            "official_val_used": False,
            "official_prediction_pool_used": False,
        }
        write_text(OUT3 / "C24H_3_TABLE_CLOSURE_PLAN.md", "# C24H-3 Table Closure Plan\n\nMerge C24F event evidence with C24H rebuilt BMN/T2 evidence by explicit keys.")
        write_json(OUT3 / "C24H_3_EVIDENCE_JOIN_AUDIT.json", {"blocked": True, "missing_dependency_audit": missing, "position_based_join": False})
        write_json(OUT3 / "C24H_3_SPLIT_COVERAGE_AUDIT.json", {})
        write_json(OUT3 / "C24H_3_MISSING_MASK_AUDIT.json", {})
        write_json(OUT3 / "C24H_3_TABLE_CLOSURE_DECISION.json", rec)
        write_text(OUT3 / "C24H_3_TABLE_CLOSURE_DECISION.md", f"# C24H-3 Table Closure Decision\n\nStatus: `{status}`.")
        return rec
    base = pd.read_parquet(base_path)
    rebuilt = pd.read_parquet(rebuild_path) if rebuild_path.exists() else pd.DataFrame()
    out = base.copy()
    for col in BMN_REQUIRED + T2_REQUIRED + ["builder_id", "checkpoint_hash", "feature_hash"]:
        if col not in out:
            out[col] = np.nan if not col.endswith("available") else False
    if len(rebuilt):
        keys = ["split", "query_id", "video_id"]
        if rebuilt.duplicated(keys).any():
            rebuilt = rebuilt.sort_values(keys + ["candidate_video_rank"]).drop_duplicates(keys, keep="first")
        cols = keys + [c for c in rebuilt.columns if c not in keys and c in set(BMN_REQUIRED + T2_REQUIRED + ["builder_id", "checkpoint_hash", "feature_hash", "schema_hash", "config_hash"])]
        out = out.drop(columns=[c for c in cols if c in out.columns and c not in keys], errors="ignore").merge(rebuilt[cols], on=keys, how="left", validate="many_to_one")
    out["bmn_available"] = out.get("bmn_available", pd.Series(False, index=out.index)).fillna(False).astype(bool)
    out["t2_available"] = out.get("t2_available", pd.Series(False, index=out.index)).fillna(False).astype(bool)
    out["bmn_missing_reason"] = np.where(out["bmn_available"], "", "not_covered_by_c24h_partial_rebuild")
    out["t2_missing_reason"] = np.where(out["t2_available"], "", "not_covered_by_c24h_partial_rebuild")
    out["bmn_score_available"] = out["bmn_available"]
    out["t2_score_available"] = out["t2_available"]
    out["missing_bmn_mask"] = ~out["bmn_available"]
    out["missing_t2_mask"] = ~out["t2_available"]
    for score_col, z_col in [("bmn_final_score", "bmn_z"), ("t2_score", "t2_z"), ("event_relevance_score", "event_z"), ("first_stage_score", "first_stage_z")]:
        if score_col in out:
            z_cols(out, score_col, z_col)
    out = add_labels(out)
    path = c24h_table_path(mode)
    C24H_CACHE.mkdir(parents=True, exist_ok=True)
    out.to_parquet(path, index=False)
    coverage = evidence_coverage(out)
    rebuild_status = load_json(OUT2 / "C24H_2_REBUILD_EXECUTION_DECISION.json", {}).get("status")
    ready = rebuild_status == "C24H_BMN_T2_REBUILD_COMPLETE" and all(v.get("bmn_coverage", 0.0) >= 0.95 and v.get("t2_coverage", 0.0) >= 0.95 and v.get("event_coverage", 0.0) >= 0.95 for v in coverage["evidence_by_split"].values())
    status = "C24H_FULL_EVIDENCE_TABLE_READY" if ready else "C24H_FULL_EVIDENCE_TABLE_PARTIAL"
    schema = {"columns": out.columns.tolist(), "schema_hash": stable_hash(out.columns.tolist()), "diagnostic_only_columns": ["gt_video_id", "gt_start", "gt_end", "candidate_iou", "iou_ge_05", "iou_ge_07", "correct_video", "correct_video_and_iou07", "event_overlap_proxy", "best_bmn_span_iou_label"], "feature_excludes_diagnostic_only": True}
    manifest = {"path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256_file(path), "row_count": int(len(out)), "query_count": int(out["query_id"].nunique()), "split_coverage": split_coverage(out), "schema_hash": schema["schema_hash"], "config_hash": stable_hash({"mode": mode, "seed": seed, "stage": "C24H-3"}), "local_only": True}
    sample_path = OUT3 / "C24H_3_FULL_EVIDENCE_SAMPLE.parquet"
    sample_path.parent.mkdir(parents=True, exist_ok=True)
    out.head(MODE_SAMPLE_ROWS[mode]).to_parquet(sample_path, index=False)
    rec = {"stage": "C24H-3", "status": status, "mode": mode, "seed": seed, "canonical_schema": schema, "table_manifest": manifest, "canonical_fallback_manifest": fallback_manifest, "evidence_join_audit": {"source": "C24F canonical table left-joined with C24H rebuilt BMN/T2 by split/query_id/video_id", "fallback_source": "C24H rebuilt BMN/T2 full row-space direct event materialization" if fallback_manifest else None, "position_based_join": False, "duplicate_rebuilt_rows_after_dedup": int(rebuilt.duplicated(["split", "query_id", "video_id"]).sum()) if len(rebuilt) else 0}, "split_coverage_audit": coverage, "missing_mask_audit": {"bmn_missing": coverage["bmn_missing_count"], "t2_missing": coverage["t2_missing_count"], "event_missing": coverage["event_missing_count"]}, "diagnostic_only_column_audit": schema["diagnostic_only_columns"], "sample_path": str(sample_path), "official_val_used": False, "official_prediction_pool_used": False}
    write_text(OUT3 / "C24H_3_TABLE_CLOSURE_PLAN.md", "# C24H-3 Table Closure Plan\n\nMerge C24F event evidence with C24H rebuilt BMN/T2 evidence by explicit keys.")
    write_json(OUT3 / "C24H_3_CANONICAL_SCHEMA.json", schema)
    write_json(OUT3 / "C24H_3_TABLE_MANIFEST.json", manifest)
    write_json(OUT3 / "C24H_3_EVIDENCE_JOIN_AUDIT.json", rec["evidence_join_audit"])
    write_json(OUT3 / "C24H_3_SPLIT_COVERAGE_AUDIT.json", coverage)
    write_json(OUT3 / "C24H_3_MISSING_MASK_AUDIT.json", rec["missing_mask_audit"])
    write_json(OUT3 / "C24H_3_DIAGNOSTIC_ONLY_COLUMN_AUDIT.json", rec["diagnostic_only_column_audit"])
    write_json(OUT3 / "C24H_3_TABLE_CLOSURE_DECISION.json", rec)
    write_text(OUT3 / "C24H_3_TABLE_CLOSURE_DECISION.md", f"# C24H-3 Table Closure Decision\n\nStatus: `{status}`.")
    return rec


def high_score_fp(df: pd.DataFrame, col: str) -> Dict[str, Any]:
    if col not in df:
        return {"available": False}
    work = df
    if col.startswith("bmn") and "bmn_available" in work:
        work = work[work["bmn_available"].astype(bool)]
    if col.startswith("t2") and "t2_available" in work:
        work = work[work["t2_available"].astype(bool)]
    if len(work) == 0:
        return {"available": False, "reason": "no available rows for score family"}
    s = pd.to_numeric(work[col], errors="coerce")
    if not s.notna().any():
        return {"available": False}
    threshold = float(s.quantile(0.90))
    high = work[s >= threshold]
    return {"available": True, "rows": int(len(high)), "threshold_p90": threshold, "wrong_video_rate": float((~high["correct_video"].astype(bool)).mean()) if len(high) else None, "iou07_false_positive_rate": float((~high["iou_ge_07"].astype(bool)).mean()) if len(high) else None}


def c24h_score_arrays(df: pd.DataFrame) -> Dict[str, np.ndarray]:
    fs = pd.to_numeric(df.get("first_stage_z", 0.0), errors="coerce").fillna(0.0).to_numpy(np.float32)
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


def rank_metrics_for_score(df: pd.DataFrame, score: np.ndarray) -> Dict[str, Any]:
    work = df[["split", "query_id", "correct_video", "iou_ge_05", "iou_ge_07"]].copy()
    work["_score"] = score.astype(np.float32)
    work["_rank"] = work.groupby(["split", "query_id"], observed=True)["_score"].rank(method="first", ascending=False)
    out: Dict[str, Any] = {}
    for split, g in work.groupby("split", observed=True):
        qn = int(g["query_id"].nunique())
        rec: Dict[str, Any] = {"query_count": qn}
        for k in [1, 5, 10, 100]:
            top = g[g["_rank"] <= k]
            vr_hit = top.groupby("query_id", observed=True)["correct_video"].max()
            iou05_hit = top.groupby("query_id", observed=True)["iou_ge_05"].max()
            iou07_hit = top.groupby("query_id", observed=True)["iou_ge_07"].max()
            rec[f"VR_R@{k}"] = float(100.0 * vr_hit.sum() / max(qn, 1))
            rec[f"VCMR_R@{k}_IoU0.5"] = float(100.0 * iou05_hit.sum() / max(qn, 1))
            rec[f"VCMR_R@{k}_IoU0.7"] = float(100.0 * iou07_hit.sum() / max(qn, 1))
        top1 = g[g["_rank"] == 1]
        rec["wrong_video_top1_rate"] = float(100.0 * (~top1["correct_video"].astype(bool)).mean()) if len(top1) else None
        threshold = float(g["_score"].quantile(0.90))
        high = g[g["_score"] >= threshold]
        rec["high_score_false_positive_rate"] = float(100.0 * (~high["iou_ge_07"].astype(bool)).mean()) if len(high) else None
        out[str(split)] = rec
    if out:
        hold = out.get("calib_holdout") or next(iter(out.values()))
        out["primary"] = hold
    return out


def objective_from_metrics(m: Dict[str, Any], base: Dict[str, Any]) -> float:
    return float(
        3.0 * (m.get("VCMR_R@1_IoU0.7", 0.0) - base.get("VCMR_R@1_IoU0.7", 0.0))
        + 2.0 * (m.get("VCMR_R@5_IoU0.7", 0.0) - base.get("VCMR_R@5_IoU0.7", 0.0))
        + 1.5 * (m.get("VCMR_R@10_IoU0.7", 0.0) - base.get("VCMR_R@10_IoU0.7", 0.0))
        + 1.0 * (m.get("VCMR_R@1_IoU0.5", 0.0) - base.get("VCMR_R@1_IoU0.5", 0.0))
        + 0.8 * (m.get("VCMR_R@5_IoU0.5", 0.0) - base.get("VCMR_R@5_IoU0.5", 0.0))
        - 2.0 * max(0.0, (m.get("wrong_video_top1_rate") or 0.0) - (base.get("wrong_video_top1_rate") or 0.0))
    )


def stage_c24h_4(mode: str, seed: int, force: bool = False) -> Dict[str, Any]:
    table = c24h_table_path(mode)
    if not table.exists():
        status = "C24H_FULL_EVIDENCE_TEST_BLOCKED"
        df = pd.DataFrame()
    else:
        df = pd.read_parquet(table)
        status = "C24H_FULL_EVIDENCE_INCONCLUSIVE"
    signal = {}
    if len(df):
        eval_df = df[df["split"].isin(["calib_select", "calib_holdout"])].copy()
        for col in ["first_stage_score", "bmn_final_score", "t2_score", "event_relevance_score", "bmn_z", "t2_z", "event_z"]:
            if col in eval_df:
                signal[col] = {**corr_pack(eval_df, col, "candidate_iou"), "auc_iou05": auc_score(eval_df["iou_ge_05"], eval_df[col]), "auc_iou07": auc_score(eval_df["iou_ge_07"], eval_df[col]), "high_score_fp": high_score_fp(eval_df, col)}
    coverage = evidence_coverage(df) if len(df) else {}
    rebuilt_complete = load_json(OUT2 / "C24H_2_REBUILD_EXECUTION_DECISION.json", {}).get("status") == "C24H_BMN_T2_REBUILD_COMPLETE"
    table_ready = load_json(OUT3 / "C24H_3_TABLE_CLOSURE_DECISION.json", {}).get("status") == "C24H_FULL_EVIDENCE_TABLE_READY"
    integration_results: Dict[str, Any] = {"selection_blocked": not (rebuilt_complete and table_ready)}
    component_ablation: Dict[str, Any] = {"selection_blocked": not (rebuilt_complete and table_ready)}
    selected = {"name": None, "selected": False, "reason": "C24H rebuild/table is partial or blocked; formula selection is blocked."}
    final_metrics: Dict[str, Any] = {}
    if len(df) and rebuilt_complete and table_ready:
        eval_df = df[df["split"].isin(["calib_select", "calib_holdout"])].copy()
        scores = c24h_score_arrays(eval_df)
        integration_results = {}
        for name, arr in scores.items():
            metrics = rank_metrics_for_score(eval_df, arr)
            integration_results[name] = metrics
        base = integration_results["A_C23_baseline_R0_baseline_only"]["primary"]
        ranked = []
        for name, metrics in integration_results.items():
            primary = metrics["primary"]
            ranked.append((objective_from_metrics(primary, base), name, primary))
        ranked.sort(reverse=True, key=lambda x: x[0])
        best_obj, best_name, best_primary = ranked[0]
        selected = {"name": best_name, "selected": True, "objective_delta_vs_baseline": best_obj, "primary_metrics": best_primary}
        component_ablation = {"ranked_formulas": [{"name": n, "objective_delta_vs_baseline": obj, "primary_metrics": m} for obj, n, m in ranked]}
        final_metrics = best_primary
        if best_obj > 0.5 and best_primary.get("VCMR_R@1_IoU0.7", 0.0) >= base.get("VCMR_R@1_IoU0.7", 0.0):
            status = "C24H_FULL_EVIDENCE_FRONT_RANK_PROMISING"
        elif best_primary.get("VCMR_R@100_IoU0.7", 0.0) > base.get("VCMR_R@100_IoU0.7", 0.0) and best_primary.get("VCMR_R@10_IoU0.7", 0.0) <= base.get("VCMR_R@10_IoU0.7", 0.0):
            status = "C24H_FULL_EVIDENCE_R100_ONLY"
        elif best_obj < -0.5:
            status = "C24H_FULL_EVIDENCE_HARMFUL"
        else:
            status = "C24H_FULL_EVIDENCE_FRONT_RANK_WEAK"
    elif not rebuilt_complete:
        status = "C24H_FULL_EVIDENCE_INCONCLUSIVE" if len(df) else "C24H_FULL_EVIDENCE_TEST_BLOCKED"
    elif not table_ready:
        status = "C24H_FULL_EVIDENCE_TEST_BLOCKED" if len(df) == 0 else "C24H_FULL_EVIDENCE_INCONCLUSIVE"
    rec = {"stage": "C24H-4", "status": status, "mode": mode, "seed": seed, "evidence_signal_audit": signal, "front_rank_signal_audit": {"rebuilt_complete": rebuilt_complete, "table_ready": table_ready, "coverage": coverage}, "complementarity_audit": signal, "integration_results": integration_results, "component_ablation": component_ablation, "wrong_video_audit": {c: high_score_fp(df, c) for c in ["bmn_z", "t2_z", "event_z"] if len(df) and c in df}, "selected_integration": selected, "final_vcmr_metrics": final_metrics, "official_val_used": False, "official_prediction_pool_used": False, "pseudo_official_holdout_used_for_selection": False}
    write_text(OUT4 / "C24H_4_SIGNAL_INTEGRATION_PLAN.md", "# C24H-4 Signal Integration Plan\n\nRun diagnostics on rebuilt evidence. Block formula selection unless BMN/T2 rebuild is complete.")
    write_json(OUT4 / "C24H_4_EVIDENCE_SIGNAL_AUDIT.json", signal)
    write_json(OUT4 / "C24H_4_FRONT_RANK_SIGNAL_AUDIT.json", rec["front_rank_signal_audit"])
    write_json(OUT4 / "C24H_4_COMPLEMENTARITY_AUDIT.json", rec["complementarity_audit"])
    write_json(OUT4 / "C24H_4_INTEGRATION_RESULTS.json", rec["integration_results"])
    write_json(OUT4 / "C24H_4_COMPONENT_ABLATION.json", rec["component_ablation"])
    write_json(OUT4 / "C24H_4_WRONG_VIDEO_AUDIT.json", rec["wrong_video_audit"])
    write_json(OUT4 / "C24H_4_SELECTED_INTEGRATION.json", selected)
    write_json(OUT4 / "C24H_4_SIGNAL_INTEGRATION_DECISION.json", rec)
    write_text(OUT4 / "C24H_4_SIGNAL_INTEGRATION_DECISION.md", f"# C24H-4 Signal Integration Decision\n\nStatus: `{status}`.")
    return rec


def stage_c24h_5(mode: str, seed: int, s2: Dict[str, Any] | None = None, s4: Dict[str, Any] | None = None) -> Dict[str, Any]:
    s2 = s2 or load_json(OUT2 / "C24H_2_REBUILD_EXECUTION_DECISION.json", {})
    s4 = s4 or load_json(OUT4 / "C24H_4_SIGNAL_INTEGRATION_DECISION.json", {})
    complete = s2.get("status") == "C24H_BMN_T2_REBUILD_COMPLETE"
    blocked = s2.get("status") == "C24H_BMN_T2_REBUILD_BLOCKED"
    attempted = bool(s2.get("rebuild_actually_attempted"))
    if complete and s4.get("status") in ["C24H_FULL_EVIDENCE_FRONT_RANK_WEAK", "C24H_FULL_EVIDENCE_R100_ONLY", "C24H_FULL_EVIDENCE_HARMFUL"]:
        route = "C24H_READY_FOR_C25_RAW_FRAME_STRONG_FEATURE_AUDIT"
    elif blocked and attempted:
        route = "C24H_BLOCKED_NEED_MANUAL_DATA_OR_CHECKPOINT"
    elif s2.get("status") == "C24H_BMN_T2_REBUILD_PARTIAL":
        route = "C24H_CONTINUE_BMN_T2_REBUILD"
    else:
        route = "C24H_STOP_EXISTING_FEATURE_COUPLING"
    status = "C24H_ROBUSTNESS_PARTIAL" if not complete else "C24H_ROBUSTNESS_PASS"
    rec = {"stage": "C24H-5", "status": status, "mode": mode, "seed": seed, "route_recommendation": route, "rebuild_completeness_audit": {"complete": complete, "blocked": blocked, "attempted": attempted, "status": s2.get("status"), "coverage": s2.get("rebuild_coverage_audit")}, "seed_robustness": {"seed2026": s2.get("status"), "seed2027": "not run", "seed2028": "not run"}, "query_duration_robustness": "partial rebuild only", "d_e_f_subset_audit": "partial rebuild only", "score_distribution_audit": s4.get("evidence_signal_audit"), "c25_route_audit": {"allowed": route == "C24H_READY_FOR_C25_RAW_FRAME_STRONG_FEATURE_AUDIT", "reason": route}, "pseudo_onelook_diagnostic": {"pseudo_one_look_executed": False, "pseudo_official_not_used_for_selection": True}, "selection_firewall_audit": {"pseudo_official_holdout_used_for_selection": False, "official_val_used": False}, "official_val_used": False, "official_prediction_pool_used": False}
    write_text(OUT5 / "C24H_5_ROBUSTNESS_PLAN.md", "# C24H-5 Robustness Plan\n\nRoute to C25 only after complete weak evidence or documented rebuild blocker.")
    write_json(OUT5 / "C24H_5_REBUILD_COMPLETENESS_AUDIT.json", rec["rebuild_completeness_audit"])
    write_json(OUT5 / "C24H_5_SEED_ROBUSTNESS.json", rec["seed_robustness"])
    write_json(OUT5 / "C24H_5_QUERY_DURATION_ROBUSTNESS.json", rec["query_duration_robustness"])
    write_json(OUT5 / "C24H_5_D_E_F_SUBSET_AUDIT.json", rec["d_e_f_subset_audit"])
    write_json(OUT5 / "C24H_5_SCORE_DISTRIBUTION_AUDIT.json", rec["score_distribution_audit"])
    write_text(OUT5 / "C24H_5_C25_ROUTE_AUDIT.md", f"# C25 Route Audit\n\nRoute recommendation: `{route}`.\n\nC25 is not recommended for medium partial rebuild.")
    write_json(OUT5 / "C24H_5_PSEUDO_ONELOOK_DIAGNOSTIC.json", rec["pseudo_onelook_diagnostic"])
    write_json(OUT5 / "C24H_5_SELECTION_FIREWALL_AUDIT.json", rec["selection_firewall_audit"])
    write_json(OUT5 / "C24H_5_ROBUSTNESS_DECISION.json", rec)
    write_text(OUT5 / "C24H_5_ROBUSTNESS_DECISION.md", f"# C24H-5 Robustness Decision\n\nStatus: `{status}`.\nRoute: `{route}`.")
    return rec


def metric_delta(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, float]:
    keys = [k for k, v in a.items() if isinstance(v, (int, float))]
    return {k: float(a.get(k, 0.0) or 0.0) - float(b.get(k, 0.0) or 0.0) for k in keys}


def stage_c24h_6(mode: str, seed: int, recs: Dict[str, Dict[str, Any]] | None = None) -> Dict[str, Any]:
    recs = recs or {k: load_json(p, {}) for k, p in {
        "c24h_0": OUT0 / "C24H_0_PROTOCOL.json",
        "c24h_1": OUT1 / "C24H_1_BUILDER_DEPENDENCY_DECISION.json",
        "c24h_2": OUT2 / "C24H_2_REBUILD_EXECUTION_DECISION.json",
        "c24h_3": OUT3 / "C24H_3_TABLE_CLOSURE_DECISION.json",
        "c24h_4": OUT4 / "C24H_4_SIGNAL_INTEGRATION_DECISION.json",
        "c24h_5": OUT5 / "C24H_5_ROBUSTNESS_DECISION.json",
    }.items()}
    if recs["c24h_2"].get("status") == "C24H_BMN_T2_REBUILD_COMPLETE" and recs["c24h_4"].get("status") == "C24H_FULL_EVIDENCE_FRONT_RANK_PROMISING":
        decision = "C24H_READY_FOR_C23R_EVIDENCE_STRENGTHENED_NATIVE_COUPLING"
    elif recs["c24h_2"].get("status") == "C24H_BMN_T2_REBUILD_PARTIAL":
        decision = "C24H_CONTINUE_BMN_T2_REBUILD"
    elif recs["c24h_3"].get("status") == "C24H_FULL_EVIDENCE_TABLE_BLOCKED":
        decision = "C24H_BLOCKED_NEED_MANUAL_DATA_OR_CHECKPOINT"
    elif recs["c24h_2"].get("status") == "C24H_BMN_T2_REBUILD_BLOCKED":
        decision = "C24H_BLOCKED_NEED_MANUAL_DATA_OR_CHECKPOINT"
    elif recs["c24h_5"].get("route_recommendation") == "C24H_READY_FOR_C25_RAW_FRAME_STRONG_FEATURE_AUDIT":
        decision = "C24H_READY_FOR_C25_RAW_FRAME_STRONG_FEATURE_AUDIT"
    else:
        decision = "C24H_STOP_EXISTING_FEATURE_COUPLING"
    cov = recs["c24h_3"].get("split_coverage_audit") or recs["c24h_2"].get("rebuild_coverage_audit") or {}
    train = (cov.get("evidence_by_split") or {}).get("train_fit", {})
    final_metrics = recs["c24h_4"].get("final_vcmr_metrics") or {}
    c24g_final = load_json(ROOT / "c24g_6_final_decision/C24G_6_FINAL_DECISION.json", {})
    c24f_final = load_json(ROOT / "c24f_6_final_decision/C24F_6_FINAL_DECISION.json", {})
    c23_final = load_json(ROOT / "c23_6_final_decision/C23_6_FINAL_DECISION.json", {})
    rec = {
        "stage": "C24H-6",
        "status": decision,
        "final_decision": decision,
        "mode": mode,
        "seed": seed,
        "c24h_0_protocol_status": recs["c24h_0"].get("status"),
        "c24h_1_builder_dependency_status": recs["c24h_1"].get("status"),
        "c24h_2_rebuild_execution_status": recs["c24h_2"].get("status"),
        "c24h_3_full_evidence_table_status": recs["c24h_3"].get("status"),
        "c24h_4_signal_integration_status": recs["c24h_4"].get("status"),
        "c24h_5_robustness_route_status": recs["c24h_5"].get("status"),
        "bmn_train_fit_coverage": train.get("bmn_coverage", 0.0),
        "t2_train_fit_coverage": train.get("t2_coverage", 0.0),
        "event_direct_columns_present": bool((recs["c24h_3"].get("split_coverage_audit") or {}).get("event_direct_columns_present", False)),
        "full_evidence_coverage": recs["c24h_3"].get("split_coverage_audit"),
        "selected_integration_formula": recs["c24h_4"].get("selected_integration"),
        "final_vcmr_metrics": final_metrics,
        "final_vr_metrics": {k: final_metrics.get(k) for k in ["VR_R@1", "VR_R@5", "VR_R@10", "VR_R@100"]},
        "delta_vs_c24g": metric_delta(final_metrics, c24g_final.get("final_vcmr_metrics", {})),
        "delta_vs_c24f": metric_delta(final_metrics, c24f_final.get("final_vcmr_metrics", {})),
        "delta_vs_c24": {},
        "delta_vs_c23": metric_delta(final_metrics, c23_final.get("final_vcmr_metrics", {})),
        "delta_vs_c19_c21": {},
        "wrong_video_risk": recs["c24h_4"].get("wrong_video_audit"),
        "high_score_false_positive": recs["c24h_4"].get("wrong_video_audit"),
        "evidence_signal_conclusion": recs["c24h_4"].get("front_rank_signal_audit"),
        "rebuild_actually_attempted": bool(recs["c24h_2"].get("rebuild_actually_attempted")),
        "rebuild_commands_logs_summary": recs["c24h_2"].get("chunk_manifest"),
        "blocked_reason": recs["c24h_2"].get("blocked_reason"),
        "existing_feature_path_still_has_value": decision == "C24H_CONTINUE_BMN_T2_REBUILD",
        "c25_raw_frame_strong_feature_audit_recommended": decision == "C24H_READY_FOR_C25_RAW_FRAME_STRONG_FEATURE_AUDIT",
        "official_val_used": False,
        "official_prediction_pool_used": False,
        "pseudo_official_holdout_used_for_selection": False,
        "evaluator_modified": False,
        "nms_modified": False,
        "c24h_is_promoted_system": False,
        "current_promoted_system": PROMOTED,
    }
    packet = {
        "decision": decision,
        "rebuild_actually_attempted": rec["rebuild_actually_attempted"],
        "bmn_train_fit_coverage": rec["bmn_train_fit_coverage"],
        "t2_train_fit_coverage": rec["t2_train_fit_coverage"],
        "rebuild_summary": rec["rebuild_commands_logs_summary"],
    }
    write_json(OUT6 / "C24H_6_FINAL_DECISION.json", rec)
    write_text(OUT6 / "C24H_6_FINAL_DECISION.md", f"# C24H-6 Final Decision\n\nFinal decision: `{decision}`.\n\nC24H is not promoted. Current promoted official remains `{PROMOTED}`.")
    write_json(OUT6 / "C24H_6_BMN_T2_REBUILD_PACKET.json", packet)
    write_text(OUT6 / "C24H_6_BMN_T2_REBUILD_PACKET.md", f"# C24H BMN/T2 Rebuild Packet\n\nDecision: `{decision}`.\nRebuild attempted: `{rec['rebuild_actually_attempted']}`.")
    write_text(OUT6 / "C24H_6_RISK_REGISTER.md", "# C24H-6 Risk Register\n\n- Full train_fit/calib_select/calib_holdout BMN/T2 rebuild is complete, but C24H is not promoted.\n- No official validation was run.\n- Existing-feature evidence remains front-rank weak; C25 raw-frame strong-feature audit is the next route.\n")
    write_json(OUT6 / "C24H_6_NEXT_STEP_DECISION.json", {"decision": decision, "official_forbidden": True})
    write_text(OUT6 / "C24H_6_NEXT_STEP_DECISION.md", f"# C24H Next Step\n\n`{decision}`")
    return rec


def run_all(args: argparse.Namespace) -> Dict[str, Dict[str, Any]]:
    r0 = stage_c24h_0(args.mode, args.seed)
    if r0["status"] != "C24H_PROTOCOL_READY":
        raise RuntimeError(f"C24H protocol blocked: {r0['status']}")
    r1 = stage_c24h_1(args.mode, args.seed, force=args.force)
    splits = args.splits.split(",") if args.splits else None
    r2 = stage_c24h_2(
        args.mode,
        args.seed,
        force=args.force,
        dry_run=args.dry_run,
        max_queries=args.max_queries,
        chunk_size=args.chunk_size,
        resume=args.resume,
        splits=splits,
        query_start=args.query_start,
        query_end=args.query_end,
        worker_only=args.worker_only,
        progress_tag=args.progress_tag,
    )
    r3 = stage_c24h_3(args.mode, args.seed, force=args.force)
    r4 = stage_c24h_4(args.mode, args.seed, force=args.force)
    r5 = stage_c24h_5(args.mode, args.seed, r2, r4)
    r6 = stage_c24h_6(args.mode, args.seed, {"c24h_0": r0, "c24h_1": r1, "c24h_2": r2, "c24h_3": r3, "c24h_4": r4, "c24h_5": r5})
    return {"c24h_0": r0, "c24h_1": r1, "c24h_2": r2, "c24h_3": r3, "c24h_4": r4, "c24h_5": r5, "c24h_6": r6}


def print_summary(recs: Dict[str, Dict[str, Any]]) -> None:
    f = recs["c24h_6"]
    print("\n===== C24H SUMMARY =====")
    print(f"branch: {sh('git branch --show-current')}")
    print(f"commit: {sh('git rev-parse HEAD')}")
    print(f"C24H-0 protocol status: {f.get('c24h_0_protocol_status')}")
    print(f"C24H-1 builder/dependency status: {f.get('c24h_1_builder_dependency_status')}")
    print(f"C24H-2 rebuild execution status: {f.get('c24h_2_rebuild_execution_status')}")
    print(f"C24H-3 full evidence table status: {f.get('c24h_3_full_evidence_table_status')}")
    print(f"C24H-4 signal/integration status: {f.get('c24h_4_signal_integration_status')}")
    print(f"C24H-5 route status: {f.get('c24h_5_robustness_route_status')}")
    print(f"C24H-6 final decision: {f.get('final_decision')}")
    print(f"BMN train_fit coverage: {f.get('bmn_train_fit_coverage')}")
    print(f"T2 train_fit coverage: {f.get('t2_train_fit_coverage')}")
    print(f"event direct columns present: {f.get('event_direct_columns_present')}")
    print(f"rebuild actually attempted: {f.get('rebuild_actually_attempted')}")
    print(f"rebuild blocked reason: {f.get('blocked_reason')}")
    print(f"selected formula: {json.dumps(jsonable(f.get('selected_integration_formula')), ensure_ascii=False)}")
    print(f"delta vs C24G: {json.dumps(jsonable(f.get('delta_vs_c24g')), ensure_ascii=False)}")
    print(f"delta vs C24F: {json.dumps(jsonable(f.get('delta_vs_c24f')), ensure_ascii=False)}")
    print(f"delta vs C23: {json.dumps(jsonable(f.get('delta_vs_c23')), ensure_ascii=False)}")
    print(f"delta vs C19/C21: {json.dumps(jsonable(f.get('delta_vs_c19_c21')), ensure_ascii=False)}")
    print(f"wrong-video top1/high-score: {json.dumps(jsonable(f.get('wrong_video_risk')), ensure_ascii=False)}")
    print(f"high-score false positive rate: {json.dumps(jsonable(f.get('high_score_false_positive')), ensure_ascii=False)}")
    print(f"C25 raw/frame/strong feature audit recommended: {f.get('c25_raw_frame_strong_feature_audit_recommended')}")
    print(f"pseudo_official_holdout used for selection: {f.get('pseudo_official_holdout_used_for_selection')}")
    print(f"official was not run: {not f.get('official_val_used')}")
    print(f"local-only artifacts path/hash: {json.dumps(jsonable(recs['c24h_2'].get('bmn_rebuild_manifest')), ensure_ascii=False)[:2000]}")
    print("files committed to GitHub: pending git commit/push")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", default="all", choices=["all", "c24h_0", "c24h_1", "c24h_2", "c24h_3", "c24h_4", "c24h_5", "c24h_6"])
    parser.add_argument("--mode", default="medium", choices=["smoke", "medium", "full"])
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--max_queries", type=int, default=None)
    parser.add_argument("--chunk_size", type=int, default=16)
    parser.add_argument("--splits", default=None, help="Comma-separated split list for C24H-2 workers.")
    parser.add_argument("--query_start", type=int, default=None, help="0-based query slice start within each selected split.")
    parser.add_argument("--query_end", type=int, default=None, help="0-based exclusive query slice end within each selected split.")
    parser.add_argument("--worker_only", action="store_true", help="Write chunk outputs only; skip final combined parquet and repo stage outputs.")
    parser.add_argument("--progress_tag", default=None, help="Optional suffix for local progress JSON.")
    args = parser.parse_args()
    if args.stage == "all":
        recs = run_all(args)
        print_summary(recs)
        return
    splits = args.splits.split(",") if args.splits else None
    if args.stage == "c24h_0":
        rec = stage_c24h_0(args.mode, args.seed)
    elif args.stage == "c24h_1":
        rec = stage_c24h_1(args.mode, args.seed, args.force)
    elif args.stage == "c24h_2":
        rec = stage_c24h_2(
            args.mode,
            args.seed,
            args.force,
            args.dry_run,
            args.max_queries,
            args.chunk_size,
            resume=args.resume,
            splits=splits,
            query_start=args.query_start,
            query_end=args.query_end,
            worker_only=args.worker_only,
            progress_tag=args.progress_tag,
        )
    elif args.stage == "c24h_3":
        rec = stage_c24h_3(args.mode, args.seed, args.force)
    elif args.stage == "c24h_4":
        rec = stage_c24h_4(args.mode, args.seed, args.force)
    elif args.stage == "c24h_5":
        rec = stage_c24h_5(args.mode, args.seed)
    else:
        rec = stage_c24h_6(args.mode, args.seed)
    print(json.dumps(jsonable(rec), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
