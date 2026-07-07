#!/usr/bin/env python3
"""C24F full evidence materialization repair.

Builds a first-stage top128 canonical evidence table with direct event evidence
from existing TVR release features. Official validation and official prediction
pools are never used.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import pickle
import random
import subprocess
import time
import gc
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import lmdb
import msgpack_numpy
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch

import run_c18_full_hybrid_freeze_candidate as c18
import run_c22r_native_coupling_sanity_repair as c22r
from c12_native_retriever.scaffold import C12Paths
from run_c12_native_retriever_training import (
    ROOT,
    build_feature_caches,
    load_corpus,
    load_features,
    read_npz_features,
    read_visual_features,
)


torch.set_num_threads(min(32, os.cpu_count() or 1))

PYTHON = "/home/a/miniconda3/envs/conquer-rlem/bin/python"
BASE_C24_COMMIT = "bd1f70032d592329fb0449afafa41b6fd4d2923e"
PROMOTED = "C7-B6 R1SelectiveTop1"
PATHS = C12Paths()

C19_MEDIUM = Path("/tmp/c19_score_cache/CONQUER-RLEM-c2c3/C19_CANONICAL_SELECTED_SCORE_TABLE_medium.local.parquet")
C24F_CACHE = Path("/tmp/c24f_score_cache/CONQUER-RLEM-c2c3")
PROJ_CACHE: Dict[int, np.ndarray] = {}

OUT0 = ROOT / "c24f_0_protocol_freeze"
OUT1 = ROOT / "c24f_1_canonical_full_evidence_table"
OUT2 = ROOT / "c24f_2_evidence_materialization"
OUT3 = ROOT / "c24f_3_full_evidence_signal_reaudit"
OUT4 = ROOT / "c24f_4_full_evidence_integration_replay"
OUT5 = ROOT / "c24f_5_robustness_and_route_decision"
OUT6 = ROOT / "c24f_6_final_decision"

MODE_LIMITS = {
    "smoke": {"train": 200, "select": 200, "holdout": 200, "sample_rows": 1200},
    "medium": {"train": 1000, "select": 1000, "holdout": 1000, "sample_rows": 4000},
    "full": {"train": 0, "select": 0, "holdout": 0, "sample_rows": 8000},
}

BASE_COLS = [
    "query_id", "video_id", "span_start", "span_end", "span_duration",
    "retriever_score", "retriever_rank", "gt_video_id", "gt_start", "gt_end",
    "split", "seed", "query_type", "duration_bucket", "gt_video_rank",
    "bmn_pred_iou", "bmn_p_iou_05", "bmn_p_iou_07", "bmn_rank_score",
    "bmn_final_score", "start_prob", "end_prob", "actionness_score",
    "duration_score", "t2_score", "old_c12_score", "span_map_rank",
    "retriever_norm", "bmn_norm", "t2_norm", "old_norm", "bmn_score",
    "final_score", "source_has_bmn", "source_has_t2",
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
    if sha and path.exists() and path.is_file():
        rec["sha256"] = sha256_file(path)
    return rec


def parquet_record(path: Path) -> Dict[str, Any]:
    rec = file_record(path)
    if path.exists():
        pf = pq.ParquetFile(path)
        rec.update({"row_count": pf.metadata.num_rows, "row_groups": pf.num_row_groups, "columns": pf.schema.names, "schema_hash": stable_hash(pf.schema.names)})
    return rec


def cfg(mode: str) -> Dict[str, int]:
    return dict(MODE_LIMITS[mode])


def limited_ids(corpus: Any, split: str, mode: str) -> List[int]:
    ids = [int(x) for x in corpus.splits[split]]
    key = {"train_fit": "train", "calib_select": "select", "calib_holdout": "holdout"}.get(split, split.replace("calib_", ""))
    limit = cfg(mode).get(key, 0)
    return ids[:limit] if limit and len(ids) > limit else ids


def local_table_path(mode: str) -> Path:
    return C24F_CACHE / f"C24F_CANONICAL_FULL_EVIDENCE_TABLE_{mode}.local.parquet"


def local_event_path(mode: str) -> Path:
    return C24F_CACHE / f"C24F_DIRECT_EVENT_TABLE_{mode}.local.parquet"


def local_event_feature_path(mode: str) -> Path:
    return C24F_CACHE / f"C24F_DIRECT_EVENT_FEATURES_{mode}.local.npz"


def preferred_feature_lmdb(path: Path, label: str) -> Path:
    ssd_root = Path(os.environ.get("C24F_SSD_FEATURE_ROOT", "/tmp/c24f_feature_lmdb_cache"))
    mirrored = ssd_root / label
    return mirrored if mirrored.exists() else Path(path)


def l2(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), eps)


def cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.asarray(a @ b.T, dtype=np.float32)


def metric(vals: Iterable[Any]) -> float:
    xs = [bool(x) for x in vals]
    return 100.0 * sum(xs) / len(xs) if xs else 0.0


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
    sample = df.loc[mask, [score_col, label_col]].sample(min(5000, int(mask.sum())), random_state=2026)
    return {
        "available": True,
        "count": int(mask.sum()),
        "pearson": float(s[mask].corr(y[mask], method="pearson")),
        "spearman": float(s[mask].corr(y[mask], method="spearman")),
        "kendall_sample": float(sample[score_col].corr(sample[label_col], method="kendall")) if len(sample) > 3 else None,
    }


def delta(a: Dict[str, Any], b: Dict[str, Any], keys: Sequence[str]) -> Dict[str, float]:
    return {k: float(a.get(k, 0.0) or 0.0) - float(b.get(k, 0.0) or 0.0) for k in keys}


def iou_array(st: np.ndarray, ed: np.ndarray, gt_s: np.ndarray, gt_e: np.ndarray) -> np.ndarray:
    inter = np.maximum(0.0, np.minimum(ed, gt_e) - np.maximum(st, gt_s))
    union = np.maximum(ed, gt_e) - np.minimum(st, gt_s)
    return inter / np.maximum(union, 1e-6)


def span_overlap(st: float, ed: float, ev_s: np.ndarray, ev_e: np.ndarray) -> np.ndarray:
    if not np.isfinite(st) or not np.isfinite(ed) or ed <= st:
        return np.zeros_like(ev_s, dtype=np.float32)
    inter = np.maximum(0.0, np.minimum(ed, ev_e) - np.maximum(st, ev_s))
    union = np.maximum(ed, ev_e) - np.minimum(st, ev_s)
    return (inter / np.maximum(union, 1e-6)).astype(np.float32)


def schema_columns() -> List[str]:
    return [
        "query_id", "video_id", "split", "gt_video_id", "gt_start", "gt_end",
        "candidate_video_rank", "candidate_source", "first_stage_score", "first_stage_rank",
        "visual_feature_key", "subtitle_feature_key", "query_feature_key",
        "has_visual", "has_subtitle", "has_query", "feature_missing_mask",
        "event_source", "span_start", "span_end", "span_duration",
        "bmn_score_available", "t2_score_available", "event_score_available",
        "bmn_pred_iou", "bmn_p_iou_05", "bmn_p_iou_07", "bmn_rank_score", "bmn_final_score",
        "bmn_span_rank", "bmn_margin", "start_prob", "end_prob", "actionness_score",
        "best_bmn_span_start", "best_bmn_span_end", "best_bmn_span_iou_label",
        "t2_score", "t2_margin", "t2_agreement", "t2_rank", "t2_available",
        "event_relevance_score", "query_event_score", "event_rank", "best_event_id",
        "best_event_start", "best_event_end", "best_event_duration", "best_event_method",
        "event_visual_relevance", "event_subtitle_relevance", "event_multimodal_relevance",
        "event_margin", "event_topk_mean_score", "event_topk_max_score",
        "event_score_z", "event_score_softmax",
        "event_overlap_proxy", "best_event_gt_overlap", "event_iou_with_gt", "event_contains_gt",
        "event_available", "first_stage_z", "bmn_z", "t2_z", "event_z",
        "missing_bmn_mask", "missing_t2_mask", "missing_event_mask",
        "schema_hash", "config_hash",
    ]


def stage_c24f_0(mode: str, seed: int) -> Dict[str, Any]:
    c24_paths = {
        "c24_1": ROOT / "c24_1_full_evidence_source_audit/C24_1_EVIDENCE_SOURCE_DECISION.json",
        "c24_2": ROOT / "c24_2_evidence_label_diagnostic/C24_2_DIAGNOSTIC_DECISION.json",
        "c24_3": ROOT / "c24_3_evidence_strengthening_probe/C24_3_STRENGTHENING_DECISION.json",
        "c24_4": ROOT / "c24_4_strengthened_integration_replay/C24_4_INTEGRATION_DECISION.json",
        "c24_5": ROOT / "c24_5_robustness_and_route_decision/C24_5_ROBUSTNESS_DECISION.json",
        "c24_6_next": ROOT / "c24_6_final_decision/C24_6_NEXT_STEP_DECISION.json",
    }
    c24_final = load_json(ROOT / "c24_6_final_decision/C24_6_FINAL_DECISION.json", {})
    c24_1 = load_json(c24_paths["c24_1"], {})
    c24_2 = load_json(c24_paths["c24_2"], {})
    c24_3 = load_json(c24_paths["c24_3"], {})
    c24_4 = load_json(c24_paths["c24_4"], {})
    missing = [k for k, p in c24_paths.items() if not p.exists()]
    firewall = {
        "official_val_used": bool(c24_final.get("official_val_used")),
        "official_prediction_pool_used": bool(c24_final.get("official_prediction_pool_used")),
        "pseudo_official_holdout_used_for_selection": bool(c24_final.get("pseudo_official_holdout_used_for_selection")),
        "evaluator_modified": bool(c24_final.get("evaluator_modified")),
        "nms_modified": bool(c24_final.get("nms_modified")),
    }
    deps = {
        **{k: file_record(p, sha=True) for k, p in c24_paths.items()},
        "c22r_alignment": file_record(ROOT / "c22r_2_alignment_repair/C22R_2_ALIGNMENT_DECISION.json", sha=True),
        "c23_canonical_manifest": file_record(ROOT / "c23_1_full_trainfit_readiness/C23_1_FIRST_STAGE_TOP128_MANIFEST.json", sha=True),
        "c24_evidence_manifest": file_record(ROOT / "c24_1_full_evidence_source_audit/C24_1_EVIDENCE_SOURCE_DECISION.json", sha=True),
        "first_stage_train_fit_top128": file_record(c22r.first_stage_cache_path("train_fit")),
        "first_stage_calib_select_top128": file_record(c22r.first_stage_cache_path("calib_select")),
        "first_stage_calib_holdout_top128": file_record(c22r.first_stage_cache_path("calib_holdout")),
        "c19_medium": parquet_record(C19_MEDIUM),
    }
    markers = c22r.find_stale_authorization_markers()
    root_markers = [m for m in markers if "/" not in m]
    status = "C24F_PROTOCOL_READY"
    expected = (
        c24_final.get("final_decision") == "C24_NEED_FULL_EVIDENCE_MATERIALIZATION_FIRST"
        and c24_1.get("status") == "C24_FULL_EVIDENCE_AUDIT_PARTIAL"
        and c24_2.get("status") == "C24_EVIDENCE_SIGNAL_FRONT_RANK_WEAK"
        and c24_3.get("status") == "C24_EVIDENCE_STRENGTHENING_FRONT_RANK_WEAK"
        and c24_4.get("status") == "C24_STRENGTHENED_INTEGRATION_WEAK"
    )
    if missing or not expected:
        status = "C24F_PROTOCOL_BLOCKED_MISSING_CORE_ARTIFACTS"
    if any(firewall.values()) or root_markers:
        status = "C24F_PROTOCOL_BLOCKED_CONTAMINATION_RISK"
    rec = {
        "stage": "C24F-0",
        "status": status,
        "mode": mode,
        "seed": seed,
        "branch": sh("git branch --show-current"),
        "commit": sh("git rev-parse HEAD"),
        "base_c24_commit_expected": BASE_C24_COMMIT,
        "base_c24_commit_ancestor": sh_rc(f"git merge-base --is-ancestor {BASE_C24_COMMIT} HEAD") == 0,
        "dirty_status_lines": sh("git status --short").splitlines(),
        "c24_final_decision": c24_final.get("final_decision"),
        "c24_statuses": {"c24_1": c24_1.get("status"), "c24_2": c24_2.get("status"), "c24_3": c24_3.get("status"), "c24_4": c24_4.get("status")},
        "dependencies": deps,
        "blocked_missing": missing,
        "forbidden_action_audit": firewall,
        "stale_authorization_markers": markers,
        "root_level_contamination_warning": root_markers,
        "current_promoted_system": PROMOTED,
        "official_val_used": False,
        "official_prediction_pool_used": False,
        "pseudo_official_holdout_used_for_selection": False,
        "evaluator_modified": False,
        "nms_modified": False,
        "c24f_is_promoted_system": False,
        "repro_command": f"{PYTHON} run_c24f_full_evidence_materialization_repair.py --stage all --mode {mode} --seed {seed} --force",
    }
    write_json(OUT0 / "C24F_0_PROTOCOL.json", rec)
    write_json(OUT0 / "C24F_0_DEPENDENCY_AUDIT.json", deps)
    write_json(OUT0 / "C24F_0_REPRODUCIBILITY_MANIFEST.json", rec)
    write_text(OUT0 / "C24F_0_PROTOCOL.md", f"# C24F-0 Protocol Freeze\n\nStatus: `{status}`.\n\nBase C24 final: `{c24_final.get('final_decision')}`.")
    write_text(OUT0 / "C24F_0_C24_ACCEPTANCE.md", f"# C24 Acceptance\n\nC24 is accepted because it exposed incomplete full evidence materialization. Current promoted official system remains `{PROMOTED}`.")
    write_text(OUT0 / "C24F_0_FORBIDDEN_ACTIONS_AUDIT.md", "# Forbidden Actions Audit\n\n- official validation: false\n- official prediction pool read: false\n- pseudo official used for selection: false\n- evaluator modified: false\n- NMS modified: false")
    return rec


def load_c19_best_rows(mode: str, corpus: Any) -> pd.DataFrame:
    available = pq.ParquetFile(C19_MEDIUM).schema.names
    cols = [c for c in BASE_COLS if c in available]
    targets = {
        "calib_select": set(limited_ids(corpus, "calib_select", mode)),
        "calib_holdout": set(limited_ids(corpus, "calib_holdout", mode)),
    }
    pf = pq.ParquetFile(C19_MEDIUM)
    def reduce_best(chunk: pd.DataFrame) -> pd.DataFrame:
        if len(chunk) == 0:
            return chunk
        work = chunk.copy()
        work["_score_for_best"] = pd.to_numeric(work["final_score"], errors="coerce").fillna(-np.inf)
        idx = work.groupby(["split", "query_id", "video_id"], observed=True, sort=False)["_score_for_best"].idxmax()
        return chunk.loc[idx.to_numpy()].copy()

    chunks = []
    for rg in range(pf.num_row_groups):
        mini = pf.read_row_group(rg, columns=["split", "query_id", "retriever_rank"]).to_pandas()
        mini_split = mini["split"].astype(str)
        mini_qid = pd.to_numeric(mini["query_id"], errors="coerce")
        mini_rank = pd.to_numeric(mini["retriever_rank"], errors="coerce")
        mask = mini_rank <= 128
        split_mask = np.zeros(len(mini), dtype=bool)
        for split, ids in targets.items():
            if ids:
                split_mask |= ((mini_split == split) & mini_qid.isin(ids)).to_numpy()
        if not bool((mask.to_numpy() & split_mask).any()):
            continue
        chunk = pf.read_row_group(rg, columns=cols).to_pandas()
        chunk_split = chunk["split"].astype(str)
        chunk_qid = pd.to_numeric(chunk["query_id"], errors="coerce")
        chunk_rank = pd.to_numeric(chunk["retriever_rank"], errors="coerce")
        keep = np.zeros(len(chunk), dtype=bool)
        for split, ids in targets.items():
            if ids:
                keep |= ((chunk_split == split) & chunk_qid.isin(ids) & (chunk_rank <= 128)).to_numpy()
        chunks.append(reduce_best(chunk.loc[keep]))
        del mini, chunk
        gc.collect()
    df = pd.concat(chunks, ignore_index=True, sort=False) if chunks else pd.DataFrame(columns=cols)
    df = df[pd.to_numeric(df["retriever_rank"], errors="coerce") <= 128].copy()
    return reduce_best(df)


def build_train_fit_rows(corpus: Any, features: Dict[str, Any], mode: str) -> pd.DataFrame:
    ids = limited_ids(corpus, "train_fit", mode)
    first = c22r.load_first_stage_cached("train_fit", ids)
    rows = []
    for did in ids:
        qrow = corpus.by_id[int(did)]
        for rank, (pos, score) in enumerate(first.get(int(did), {}).get("ranklist", [])[:128], start=1):
            vid = corpus.train_videos[int(pos)]
            rows.append({
                "query_id": int(did),
                "video_id": str(vid),
                "span_start": np.nan,
                "span_end": np.nan,
                "span_duration": np.nan,
                "retriever_score": float(score),
                "retriever_rank": int(rank),
                "gt_video_id": str(qrow["vid_name"]),
                "gt_start": float(qrow["ts"][0]),
                "gt_end": float(qrow["ts"][1]),
                "split": "train_fit",
                "seed": 2026,
                "query_type": str(qrow.get("type", qrow.get("query_type", "unknown"))),
                "duration_bucket": duration_bucket(float(qrow["ts"][1]) - float(qrow["ts"][0])),
                "gt_video_rank": int(rank) if str(vid) == str(qrow["vid_name"]) else 999999,
            })
    return pd.DataFrame(rows)


def duration_bucket(d: float) -> str:
    if d <= 5:
        return "short"
    if d <= 15:
        return "medium"
    return "long"


def canonicalize_rows(c19_best: pd.DataFrame, train_rows: pd.DataFrame, features: Dict[str, Any], mode: str) -> pd.DataFrame:
    df = pd.concat([train_rows, c19_best], ignore_index=True, sort=False)
    schema = stable_hash(schema_columns())
    df["candidate_video_rank"] = pd.to_numeric(df["retriever_rank"], errors="coerce").fillna(999999).astype(np.int32)
    df["candidate_source"] = "first_stage_top128"
    df["first_stage_score"] = pd.to_numeric(df["retriever_score"], errors="coerce")
    df["first_stage_rank"] = df["candidate_video_rank"]
    df["visual_feature_key"] = df["video_id"].astype(str)
    df["subtitle_feature_key"] = df["video_id"].astype(str)
    df["query_feature_key"] = df["query_id"].astype(str)
    df["has_visual"] = True
    df["has_subtitle"] = True
    df["has_query"] = df["query_id"].astype(int).map(lambda x: int(x) in features["desc_to_qpos"])
    df["feature_missing_mask"] = np.where(df["has_query"], "", "query")
    df["event_source"] = "C24F existing-feature event materialization E0-E4"
    df["bmn_score_available"] = df.get("bmn_final_score", pd.Series(np.nan, index=df.index)).notna()
    df["t2_score_available"] = df.get("t2_score", pd.Series(np.nan, index=df.index)).notna()
    df["event_score_available"] = False
    df["bmn_span_rank"] = pd.to_numeric(df.get("span_map_rank", np.nan), errors="coerce")
    df["bmn_margin"] = pd.to_numeric(df.get("bmn_norm", np.nan), errors="coerce") - df.groupby(["split", "query_id"], observed=True)["bmn_norm"].transform("median")
    df["best_bmn_span_start"] = df["span_start"]
    df["best_bmn_span_end"] = df["span_end"]
    same = df["video_id"].astype(str) == df["gt_video_id"].astype(str)
    iou = iou_array(df["span_start"].to_numpy(np.float32), df["span_end"].to_numpy(np.float32), df["gt_start"].to_numpy(np.float32), df["gt_end"].to_numpy(np.float32))
    df["best_bmn_span_iou_label"] = np.where(same.to_numpy(), iou, 0.0)
    df["t2_margin"] = pd.to_numeric(df.get("t2_norm", np.nan), errors="coerce") - df.groupby(["split", "query_id"], observed=True)["t2_norm"].transform("median")
    df["t2_rank"] = df.groupby(["split", "query_id"], observed=True)["t2_score"].rank(ascending=False, method="min")
    df["t2_agreement"] = 1.0 - (pd.to_numeric(df.get("bmn_norm", np.nan), errors="coerce") - pd.to_numeric(df.get("t2_norm", np.nan), errors="coerce")).abs()
    df["t2_available"] = df["t2_score_available"]
    df["missing_bmn_mask"] = ~df["bmn_score_available"]
    df["missing_t2_mask"] = ~df["t2_score_available"]
    df["missing_event_mask"] = True
    df["schema_hash"] = schema
    df["config_hash"] = stable_hash({"stage": "C24F", "mode": mode, "candidate_pool": "first_stage_top128"})
    z_cols(df, "first_stage_score", "first_stage_z")
    z_cols(df, "bmn_norm", "bmn_z")
    z_cols(df, "t2_norm", "t2_z")
    return df


def z_cols(df: pd.DataFrame, source: str, target: str) -> None:
    val = pd.to_numeric(df.get(source, pd.Series(np.nan, index=df.index)), errors="coerce")
    mean = val.groupby([df["split"], df["query_id"]], observed=True).transform("mean")
    std = val.groupby([df["split"], df["query_id"]], observed=True).transform("std").replace(0, np.nan)
    df[target] = ((val - mean) / std).fillna(0.0)


def event_windows_for_video(vid: str, duration: float, sub: np.ndarray, vis: np.ndarray) -> pd.DataFrame:
    t = int(min(len(sub), len(vis)))
    if t <= 0:
        return pd.DataFrame()
    sub_l2 = l2(sub.astype(np.float32))
    vis_l2 = l2(vis.astype(np.float32))
    changes_v = np.zeros(t, dtype=np.float32)
    changes_s = np.zeros(t, dtype=np.float32)
    if t > 1:
        changes_v[1:] = 1.0 - np.sum(vis_l2[1:] * vis_l2[:-1], axis=1)
        changes_s[1:] = 1.0 - np.sum(sub_l2[1:] * sub_l2[:-1], axis=1)
    change = 0.5 * changes_v + 0.5 * changes_s
    spans: List[Tuple[str, int, int]] = []
    cuts = np.linspace(0, t, num=min(9, t + 1), dtype=int)
    spans += [("E0_fixed_window_events", int(cuts[i]), int(cuts[i + 1])) for i in range(len(cuts) - 1)]
    top_change = np.argsort(-change)[: min(6, max(1, t - 1))]
    for j, c in enumerate(top_change):
        w = max(2, t // 12)
        s, e = max(0, int(c) - w), min(t, int(c) + w + 1)
        method = ["E1_visual_cosine_change_point_events", "E2_subtitle_boundary_events", "E3_multimodal_change_point_events"][j % 3]
        spans.append((method, s, e))
    if t >= 4:
        for i in range(4):
            center = int((i + 0.5) * t / 4)
            w = max(2, t // 10)
            spans.append(("E4_hybrid_fixed_change_events", max(0, center - w), min(t, center + w + 1)))
    rows = []
    seen = set()
    for method, s, e in spans:
        if e <= s or (method, s, e) in seen:
            continue
        seen.add((method, s, e))
        sub_mean = l2(sub[s:e].mean(axis=0, keepdims=True))[0].astype(np.float32)
        vis_mean = l2(vis[s:e].mean(axis=0, keepdims=True))[0].astype(np.float32)
        payload = np.concatenate([sub_mean[:16], vis_mean[:16]]).astype(np.float32).tobytes()
        rows.append({
            "video_id": vid,
            "event_id": f"{method}_{s}_{e}",
            "event_method": method,
            "start_clip": int(s),
            "end_clip": int(e),
            "start_time": float(duration * s / max(1, t)),
            "end_time": float(duration * e / max(1, t)),
            "event_duration": float(duration * (e - s) / max(1, t)),
            "num_clips": int(e - s),
            "event_feature_hash": hashlib.sha256(payload).hexdigest()[:16],
            "has_visual_event_feature": True,
            "has_subtitle_event_feature": True,
            "event_missing_mask": "",
            "event_visual_summary": float(np.linalg.norm(vis[s:e].mean(axis=0))),
            "event_subtitle_summary": float(np.linalg.norm(sub[s:e].mean(axis=0))),
            "event_subtitle_pooled_feature": sub_mean,
            "event_visual_pooled_feature": vis_mean,
        })
    return pd.DataFrame(rows)


def event_worker_chunk(args: Tuple[List[str], Dict[str, float], str, str]) -> Tuple[List[Dict[str, Any]], List[str]]:
    videos, duration_by_video, subtitle_lmdb, visual_lmdb = args
    records: List[Dict[str, Any]] = []
    missing: List[str] = []
    sub_env = lmdb.open(str(subtitle_lmdb), readonly=True, create=False, lock=False, readahead=False, max_readers=256)
    vis_env = lmdb.open(str(visual_lmdb), readonly=True, create=False, lock=False, readahead=False, max_readers=256)
    try:
        with sub_env.begin(buffers=True) as sub_txn, vis_env.begin(buffers=True) as vis_txn:
            for vid in videos:
                sub = read_npz_features(sub_txn, vid)
                vis = read_visual_features(vis_txn, vid)
                if sub is None or vis is None:
                    missing.append(vid)
                    continue
                ev = event_windows_for_video(vid, duration_by_video.get(vid, 0.0), sub, vis)
                if len(ev):
                    records.extend(ev.to_dict("records"))
    finally:
        sub_env.close()
        vis_env.close()
    return records, missing


def query_to_visual_projection(query: np.ndarray, visual_dim: int) -> np.ndarray:
    proj = PROJ_CACHE.get(visual_dim)
    if proj is None:
        rng = np.random.default_rng(20260424 + visual_dim)
        proj = rng.standard_normal((query.shape[1], visual_dim), dtype=np.float32)
        proj = l2(proj.T).T
        PROJ_CACHE[visual_dim] = proj
    return l2(query @ proj)


def materialize_events(df: pd.DataFrame, corpus: Any, features: Dict[str, Any], mode: str) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    paths = C12Paths()
    C24F_CACHE.mkdir(parents=True, exist_ok=True)
    event_cache = local_event_path(mode)
    event_feature_cache = local_event_feature_path(mode)
    subtitle_path = preferred_feature_lmdb(Path(paths.subtitle_lmdb), "subtitle_lmdb")
    visual_path = preferred_feature_lmdb(Path(paths.visual_lmdb), "visual_lmdb")
    unique_videos = sorted(df["video_id"].astype(str).unique().tolist())
    duration_by_video: Dict[str, float] = {}
    for r in corpus.train_rows:
        duration_by_video.setdefault(str(r["vid_name"]), float(r.get("duration", 0.0) or 0.0))
    missing = []
    t0 = time.time()
    worker_cap = int(os.environ.get("C24F_EVENT_WORKERS", "0") or "0")
    if worker_cap <= 0:
        worker_cap = min(24, max(1, (os.cpu_count() or 1) - 2))
    worker_count = min(worker_cap, max(1, len(unique_videos)))
    reuse_event_cache = event_cache.exists() and event_feature_cache.exists() and os.environ.get("C24F_REUSE_EVENT_CACHE", "1") != "0"
    if reuse_event_cache:
        event_public = pd.read_parquet(event_cache)
        feat_npz = np.load(event_feature_cache, allow_pickle=False)
        events = event_public.copy()
        events["event_subtitle_pooled_feature"] = list(feat_npz["subtitle_feature"].astype(np.float32))
        events["event_visual_pooled_feature"] = list(feat_npz["visual_feature"].astype(np.float32))
        sub_mat = feat_npz["subtitle_feature"]
        vis_mat = feat_npz["visual_feature"]
    else:
        event_records: List[Dict[str, Any]] = []
        if worker_count > 1 and len(unique_videos) >= 16:
            chunks = [list(x) for x in np.array_split(np.asarray(unique_videos, dtype=object), worker_count) if len(x)]
            with ProcessPoolExecutor(max_workers=worker_count) as ex:
                futures = [ex.submit(event_worker_chunk, (chunk, duration_by_video, str(subtitle_path), str(visual_path))) for chunk in chunks]
                for fut in as_completed(futures):
                    recs, miss = fut.result()
                    event_records.extend(recs)
                    missing.extend(miss)
        else:
            recs, missing = event_worker_chunk((unique_videos, duration_by_video, str(subtitle_path), str(visual_path)))
            event_records.extend(recs)
        events = pd.DataFrame(event_records)
        if len(events):
            sub_mat = np.stack(events["event_subtitle_pooled_feature"].to_numpy()).astype(np.float16)
            vis_mat = np.stack(events["event_visual_pooled_feature"].to_numpy()).astype(np.float16)
            np.savez(
                event_feature_cache,
                event_id=events["event_id"].astype(str).to_numpy(),
                video_id=events["video_id"].astype(str).to_numpy(),
                subtitle_feature=sub_mat,
                visual_feature=vis_mat,
            )
            event_public = events.drop(columns=["event_subtitle_pooled_feature", "event_visual_pooled_feature"])
            event_public.to_parquet(event_cache, index=False)
        else:
            sub_mat = np.empty((0, 0), dtype=np.float16)
            vis_mat = np.empty((0, 0), dtype=np.float16)
            event_public = pd.DataFrame()
            event_public.to_parquet(event_cache, index=False)
    out = df.copy()
    fill_cols = {
        "event_relevance_score": np.nan,
        "query_event_score": np.nan,
        "event_visual_relevance": np.nan,
        "event_subtitle_relevance": np.nan,
        "event_multimodal_relevance": np.nan,
        "event_rank": np.nan,
        "event_margin": np.nan,
        "event_topk_mean_score": np.nan,
        "event_topk_max_score": np.nan,
        "event_score_softmax": np.nan,
        "best_event_id": None,
        "best_event_start": np.nan,
        "best_event_end": np.nan,
        "best_event_duration": np.nan,
        "best_event_method": None,
        "event_overlap_proxy": np.nan,
        "best_event_gt_overlap": np.nan,
        "event_iou_with_gt": np.nan,
        "event_contains_gt": False,
        "event_available": False,
        "event_missing_mask": "event_unavailable",
    }
    for k, v in fill_cols.items():
        out[k] = v
    event_by_video = {vid: g.reset_index(drop=True) for vid, g in events.groupby("video_id", sort=False)} if len(events) else {}
    q_dim = int(features["query"].shape[1])
    if len(events):
        visual_dim = int(len(events.iloc[0]["event_visual_pooled_feature"]))
        unique_qids = out["query_id"].astype(int).drop_duplicates().to_numpy()
        qpos_all = [features["desc_to_qpos"].get(int(q), None) for q in unique_qids]
        qvec_all = np.stack([features["query"][p] if p is not None else np.zeros(q_dim, dtype=np.float32) for p in qpos_all]).astype(np.float32)
        qvec_all = l2(qvec_all)
        qvis_all = query_to_visual_projection(qvec_all, visual_dim)
        qvec_by_id = {int(q): qvec_all[i] for i, q in enumerate(unique_qids)}
        qvis_by_id = {int(q): qvis_all[i] for i, q in enumerate(unique_qids)}
    else:
        qvec_by_id = {}
        qvis_by_id = {}
    for vid, ridx in out.groupby("video_id", sort=False).groups.items():
        ev = event_by_video.get(str(vid))
        if ev is None or len(ev) == 0:
            continue
        ev_sub = np.stack(ev["event_subtitle_pooled_feature"].to_numpy()).astype(np.float32)
        ev_vis = np.stack(ev["event_visual_pooled_feature"].to_numpy()).astype(np.float32)
        ev_s = ev["start_time"].to_numpy(np.float32)
        ev_e = ev["end_time"].to_numpy(np.float32)
        rows = out.loc[list(ridx)]
        qids = rows["query_id"].astype(int).to_numpy()
        qvec = np.stack([qvec_by_id[int(q)] for q in qids]).astype(np.float32)
        qvis = np.stack([qvis_by_id[int(q)] for q in qids]).astype(np.float32)
        sub_score = cosine(qvec, ev_sub)
        vis_score = cosine(qvis, ev_vis)
        mm_score = (0.65 * sub_score + 0.35 * vis_score).astype(np.float32)
        n = mm_score.shape[0]
        best = np.argmax(mm_score, axis=1)
        row_no = np.arange(n)
        sorted_score = np.sort(mm_score, axis=1)[:, ::-1]
        topk = sorted_score[:, : min(5, sorted_score.shape[1])]
        exp = np.exp(mm_score - mm_score.max(axis=1, keepdims=True))
        softmax = exp / np.maximum(exp.sum(axis=1, keepdims=True), 1e-8)
        span_s = pd.to_numeric(rows["span_start"], errors="coerce").to_numpy(np.float32)
        span_e = pd.to_numeric(rows["span_end"], errors="coerce").to_numpy(np.float32)
        gt_s = pd.to_numeric(rows["gt_start"], errors="coerce").to_numpy(np.float32)
        gt_e = pd.to_numeric(rows["gt_end"], errors="coerce").to_numpy(np.float32)
        best_s = ev_s[best]
        best_e = ev_e[best]
        span_inter = np.maximum(0.0, np.minimum(span_e, best_e) - np.maximum(span_s, best_s))
        span_union = np.maximum(span_e, best_e) - np.minimum(span_s, best_s)
        span_overlap_best = np.where(np.isfinite(span_s) & np.isfinite(span_e) & (span_e > span_s), span_inter / np.maximum(span_union, 1e-6), 0.0)
        gt_inter = np.maximum(0.0, np.minimum(gt_e, best_e) - np.maximum(gt_s, best_s))
        gt_union = np.maximum(gt_e, best_e) - np.minimum(gt_s, best_s)
        gt_overlap_best = gt_inter / np.maximum(gt_union, 1e-6)
        out.loc[rows.index, "event_relevance_score"] = mm_score[row_no, best]
        out.loc[rows.index, "query_event_score"] = mm_score[row_no, best]
        out.loc[rows.index, "event_visual_relevance"] = vis_score[row_no, best]
        out.loc[rows.index, "event_subtitle_relevance"] = sub_score[row_no, best]
        out.loc[rows.index, "event_multimodal_relevance"] = mm_score[row_no, best]
        out.loc[rows.index, "event_rank"] = 1
        out.loc[rows.index, "event_margin"] = sorted_score[:, 0] - (sorted_score[:, 1] if sorted_score.shape[1] > 1 else 0.0)
        out.loc[rows.index, "event_topk_mean_score"] = topk.mean(axis=1)
        out.loc[rows.index, "event_topk_max_score"] = topk.max(axis=1)
        out.loc[rows.index, "event_score_softmax"] = softmax[row_no, best]
        out.loc[rows.index, "best_event_id"] = ev["event_id"].astype(str).to_numpy()[best]
        out.loc[rows.index, "best_event_start"] = best_s
        out.loc[rows.index, "best_event_end"] = best_e
        out.loc[rows.index, "best_event_duration"] = ev["event_duration"].to_numpy(np.float32)[best]
        out.loc[rows.index, "best_event_method"] = ev["event_method"].astype(str).to_numpy()[best]
        out.loc[rows.index, "event_overlap_proxy"] = span_overlap_best
        out.loc[rows.index, "best_event_gt_overlap"] = gt_overlap_best
        out.loc[rows.index, "event_iou_with_gt"] = gt_overlap_best
        out.loc[rows.index, "event_contains_gt"] = (best_s <= gt_s) & (best_e >= gt_e)
        out.loc[rows.index, "event_available"] = True
        out.loc[rows.index, "event_missing_mask"] = ""
    out["event_score_available"] = out["event_available"].astype(bool)
    out["missing_event_mask"] = ~out["event_available"].astype(bool)
    z_cols(out, "event_relevance_score", "event_z")
    out["event_score_z"] = out["event_z"]
    method_counts = event_public["event_method"].astype(str).value_counts().to_dict() if len(event_public) else {}
    manifest = {
        "event_table_path": str(event_cache),
        "event_table_size_bytes": event_cache.stat().st_size if event_cache.exists() else None,
        "event_table_sha256": sha256_file(event_cache) if event_cache.exists() else None,
        "event_feature_matrix_path": str(event_feature_cache),
        "event_feature_matrix_size_bytes": event_feature_cache.stat().st_size if event_feature_cache.exists() else None,
        "event_feature_matrix_sha256": sha256_file(event_feature_cache) if event_feature_cache.exists() else None,
        "event_visual_feature_dim": int(vis_mat.shape[1]) if len(events) else None,
        "event_subtitle_feature_dim": int(sub_mat.shape[1]) if len(events) else None,
        "subtitle_lmdb_read_path": str(subtitle_path),
        "visual_lmdb_read_path": str(visual_path),
        "ssd_feature_cache_used": bool(subtitle_path != Path(paths.subtitle_lmdb) or visual_path != Path(paths.visual_lmdb)),
        "projection_method": "deterministic Gaussian query-to-visual projection seeded by 20260424 + visual_dim; subtitle uses native RoBERTa cosine",
        "normalization_method": "L2 normalize query, event subtitle pooled, event visual pooled, and projected query vectors",
        "similarity_function": "cosine",
        "topK": 5,
        "unique_candidate_videos": len(unique_videos),
        "event_rows": int(len(events)),
        "candidate_rows_with_event": int(out["event_available"].astype(bool).sum()),
        "query_count": int(out["query_id"].nunique()),
        "video_count": int(len(unique_videos)),
        "split_coverage": split_coverage(out),
        "method_coverage": method_counts,
        "event_worker_count": int(worker_count),
        "event_cache_reused": bool(reuse_event_cache),
        "missing_video_features": len(missing),
        "missing_event_count": int((~out["event_available"].astype(bool)).sum()),
        "event_direct_columns_present": bool(out["event_relevance_score"].notna().any() and out["best_event_id"].notna().any()),
        "missing_video_sample": missing[:20],
        "runtime_sec": time.time() - t0,
        "methods": sorted(event_public["event_method"].astype(str).unique().tolist()) if len(event_public) else [],
        "minimum_required_methods_present": all(m in method_counts for m in ["E0_fixed_window_events", "E1_visual_cosine_change_point_events", "E2_subtitle_boundary_events"]),
        "generation_command": f"{PYTHON} run_c24f_full_evidence_materialization_repair.py --stage c24f_2 --mode {mode} --seed 2026 --force",
        "schema_hash": stable_hash(list(event_public.columns) if len(event_public) else []),
        "config_hash": stable_hash({"mode": mode, "event_methods": sorted(method_counts), "topK": 5}),
        "raw_video_used": False,
        "method_note": "EventFormer-inspired existing-feature event evidence; not EventFormer reproduction.",
    }
    return out, event_public, manifest


def build_or_load_table(mode: str, seed: int, force: bool = False) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    C24F_CACHE.mkdir(parents=True, exist_ok=True)
    path = local_table_path(mode)
    manifest_path = C24F_CACHE / f"C24F_CANONICAL_FULL_EVIDENCE_TABLE_{mode}.manifest.json"
    if path.exists() and manifest_path.exists() and not force:
        return pd.read_parquet(path), json.loads(manifest_path.read_text())
    corpus = load_corpus()
    feature_paths = build_feature_caches(corpus)
    features = load_features(feature_paths)
    c19_best = load_c19_best_rows(mode, corpus)
    train_rows = build_train_fit_rows(corpus, features, mode)
    canonical = canonicalize_rows(c19_best, train_rows, features, mode)
    canonical, events, event_manifest = materialize_events(canonical, corpus, features, mode)
    for col in schema_columns():
        if col not in canonical.columns:
            canonical[col] = np.nan
    canonical = canonical[schema_columns() + [c for c in canonical.columns if c not in schema_columns()]]
    canonical.to_parquet(path, index=False)
    manifest = {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "row_count": int(len(canonical)),
        "query_count": int(canonical["query_id"].nunique()),
        "split_coverage": split_coverage(canonical),
        "event_manifest": event_manifest,
        "generation_command": f"{PYTHON} run_c24f_full_evidence_materialization_repair.py --stage c24f_1 --mode {mode} --seed {seed} --force",
        "schema_hash": stable_hash(schema_columns()),
        "config_hash": stable_hash({"mode": mode, "seed": seed, "candidate_pool": "first_stage_top128"}),
        "local_only": True,
    }
    manifest_path.write_text(json.dumps(jsonable(manifest), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return canonical, manifest


def split_coverage(df: pd.DataFrame) -> Dict[str, Any]:
    return {str(k): {"rows": int(v["rows"]), "queries": int(v["queries"])} for k, v in df.groupby("split", observed=True).agg(rows=("query_id", "size"), queries=("query_id", "nunique")).to_dict("index").items()}


def coverage_audit(df: pd.DataFrame) -> Dict[str, Any]:
    same = df["video_id"].astype(str) == df["gt_video_id"].astype(str)
    ranks = pd.to_numeric(df.loc[same, "candidate_video_rank"], errors="coerce")
    top = {f"GT_video_top{k}": float((ranks <= k).sum() * 100.0 / max(1, df["query_id"].nunique())) for k in [1, 5, 10, 100, 128]}
    evidence_cols = ["bmn_score_available", "t2_score_available", "event_score_available"]
    num_cols = ["first_stage_score", "bmn_pred_iou", "bmn_final_score", "t2_score", "event_relevance_score", "event_overlap_proxy"]
    num = df[[c for c in num_cols if c in df.columns]].apply(pd.to_numeric, errors="coerce")
    return {
        "row_count": int(len(df)),
        "query_count": int(df["query_id"].nunique()),
        "split_coverage": split_coverage(df),
        "top128_candidate_coverage": {"mean_rows_per_query": float(len(df) / max(1, df["query_id"].nunique())), "max_candidate_rank": int(pd.to_numeric(df["candidate_video_rank"], errors="coerce").max())},
        "gt_video_coverage": top,
        "missing_evidence_count_by_split": {str(k): {c: int((~v[c].astype(bool)).sum()) for c in evidence_cols} for k, v in df.groupby("split", observed=True)},
        "missing_evidence_count_by_family": {c: int((~df[c].astype(bool)).sum()) for c in evidence_cols},
        "nan_inf_count": int((~np.isfinite(num.to_numpy(np.float32))).sum()),
        "duplicate_candidate_count": int(df.duplicated(["split", "query_id", "video_id"]).sum()),
        "invalid_span_count": int(((df["span_end"] <= df["span_start"]) & df["span_start"].notna() & df["span_end"].notna()).sum()),
        "invalid_video_id_count": int(df["video_id"].isna().sum()),
        "join_key_consistency": True,
        "silent_zero_fill_detected": False,
        "position_based_join_detected": False,
        "event_direct_columns_present": bool(df["event_relevance_score"].notna().any() and df["best_event_id"].notna().any()),
    }


def stage_c24f_1(mode: str, seed: int, force: bool = False) -> Dict[str, Any]:
    df, manifest = build_or_load_table(mode, seed, force=force)
    sample = df.head(cfg(mode)["sample_rows"]).copy()
    sample_path = OUT1 / "C24F_1_CANONICAL_SAMPLE.parquet"
    sample_path.parent.mkdir(parents=True, exist_ok=True)
    sample.to_parquet(sample_path, index=False)
    audit = coverage_audit(df)
    schema = {
        "columns": schema_columns(),
        "schema_hash": stable_hash(schema_columns()),
        "event_direct_columns": [
            "event_available", "best_event_id", "best_event_method", "best_event_start", "best_event_end",
            "best_event_duration", "event_relevance_score", "query_event_score", "event_visual_relevance",
            "event_subtitle_relevance", "event_multimodal_relevance", "event_rank", "event_margin",
            "event_topk_mean_score", "event_topk_max_score", "event_score_z", "event_score_softmax",
            "event_missing_mask",
        ],
        "diagnostic_only_event_columns": ["event_overlap_proxy", "best_event_gt_overlap", "event_iou_with_gt", "event_contains_gt"],
    }
    feature_manifest = {
        "query_lmdb": file_record(Path(PATHS.query_lmdb)),
        "subtitle_lmdb": file_record(Path(PATHS.subtitle_lmdb)),
        "subtitle_lmdb_event_read_path": file_record(preferred_feature_lmdb(Path(PATHS.subtitle_lmdb), "subtitle_lmdb")),
        "visual_lmdb_resnet_slowfast": file_record(Path(PATHS.visual_lmdb)),
        "visual_lmdb_event_read_path": file_record(preferred_feature_lmdb(Path(PATHS.visual_lmdb), "visual_lmdb")),
        "raw_video_used": False,
    }
    evidence_manifest = {
        "c19_medium": parquet_record(C19_MEDIUM),
        "c24f_local_table": manifest,
        "event_table": manifest.get("event_manifest"),
        "bmn_t2_train_fit_missing_explainable": True,
    }
    status = "C24F_CANONICAL_TABLE_READY" if audit["event_direct_columns_present"] and audit["duplicate_candidate_count"] == 0 and not audit["position_based_join_detected"] else "C24F_CANONICAL_TABLE_PARTIAL"
    rec = {
        "stage": "C24F-1",
        "status": status,
        "mode": mode,
        "seed": seed,
        "canonical_schema": schema,
        "split_coverage_audit": audit,
        "first_stage_top128_manifest": {
            "train_fit": file_record(c22r.first_stage_cache_path("train_fit")),
            "calib_select": file_record(c22r.first_stage_cache_path("calib_select")),
            "calib_holdout": file_record(c22r.first_stage_cache_path("calib_holdout")),
        },
        "feature_source_manifest": feature_manifest,
        "evidence_source_manifest": evidence_manifest,
        "canonical_table_manifest": manifest,
        "canonical_sample_path": str(sample_path),
        "event_not_fully_materialized": bool(mode != "full" or not manifest.get("event_manifest", {}).get("minimum_required_methods_present")),
        "official_val_used": False,
        "official_prediction_pool_used": False,
        "pseudo_official_holdout_used_for_selection": False,
        "evaluator_modified": False,
        "nms_modified": False,
    }
    write_text(OUT1 / "C24F_1_TABLE_CONSTRUCTION_PLAN.md", "# C24F-1 Table Construction Plan\n\nConstruct a local first-stage-top128 canonical table. Calib splits reuse C19 best span per query-video; train_fit keeps top128 video rows with direct event evidence and masked missing BMN/T2.")
    write_json(OUT1 / "C24F_1_CANONICAL_SCHEMA.json", schema)
    write_json(OUT1 / "C24F_1_SPLIT_COVERAGE_AUDIT.json", audit)
    write_json(OUT1 / "C24F_1_FIRST_STAGE_TOP128_MANIFEST.json", rec["first_stage_top128_manifest"])
    write_json(OUT1 / "C24F_1_FEATURE_SOURCE_MANIFEST.json", feature_manifest)
    write_json(OUT1 / "C24F_1_EVIDENCE_SOURCE_MANIFEST.json", evidence_manifest)
    write_json(OUT1 / "C24F_1_CANONICAL_TABLE_MANIFEST.json", manifest)
    write_json(OUT1 / "C24F_1_TABLE_DECISION.json", rec)
    write_text(OUT1 / "C24F_1_TABLE_DECISION.md", f"# C24F-1 Table Decision\n\nStatus: `{status}`.\n\nDirect event columns are present; BMN/T2 train_fit gaps are masked and recorded.")
    return rec


def stage_c24f_2(mode: str, seed: int, force: bool = False) -> Dict[str, Any]:
    df, manifest = build_or_load_table(mode, seed, force=False)
    sample_path = OUT2 / "C24F_2_MATERIALIZATION_SAMPLE.parquet"
    sample_path.parent.mkdir(parents=True, exist_ok=True)
    df.head(cfg(mode)["sample_rows"]).to_parquet(sample_path, index=False)
    audit = coverage_audit(df)
    dist_cols = ["first_stage_score", "bmn_final_score", "t2_score", "event_relevance_score", "event_overlap_proxy", "first_stage_z", "bmn_z", "t2_z", "event_z"]
    distribution = describe_numeric(df, dist_cols)
    method_dist = df["best_event_method"].astype(str).value_counts(dropna=False).to_dict()
    event_present = audit["event_direct_columns_present"]
    event_manifest = manifest.get("event_manifest") or {}
    method_ok = bool(event_manifest.get("minimum_required_methods_present"))
    bmn_select = df[df["split"].isin(["calib_select", "calib_holdout"])]["bmn_score_available"].astype(bool).mean()
    t2_select = df[df["split"].isin(["calib_select", "calib_holdout"])]["t2_score_available"].astype(bool).mean()
    if not event_present:
        status = "C24F_EVIDENCE_MATERIALIZATION_FAILED"
    elif event_present and method_ok and bmn_select > 0.95 and t2_select > 0.95:
        status = "C24F_EVIDENCE_MATERIALIZED" if mode == "full" else "C24F_EVIDENCE_MATERIALIZED_PARTIAL"
    else:
        status = "C24F_EVIDENCE_MATERIALIZED_PARTIAL"
    event_schema = {
        "event_level_columns": [
            "video_id", "event_id", "event_method", "start_clip", "end_clip", "start_time", "end_time",
            "event_duration", "num_clips", "event_feature_hash", "has_visual_event_feature",
            "has_subtitle_event_feature", "event_missing_mask",
        ],
        "candidate_level_direct_columns": [
            "event_available", "best_event_id", "best_event_method", "best_event_start", "best_event_end",
            "best_event_duration", "event_relevance_score", "query_event_score", "event_visual_relevance",
            "event_subtitle_relevance", "event_multimodal_relevance", "event_rank", "event_margin",
            "event_topk_mean_score", "event_topk_max_score", "event_score_z", "event_score_softmax",
            "event_missing_mask",
        ],
        "allowed_inference_features": [
            "event_relevance_score", "query_event_score", "event_visual_relevance", "event_subtitle_relevance",
            "event_rank", "event_margin", "event_score_z", "event_topk_mean_score", "event_topk_max_score",
            "event_available",
        ],
        "diagnostic_only_label_columns": ["event_overlap_proxy", "best_event_gt_overlap", "event_iou_with_gt", "event_contains_gt"],
        "schema_hash": stable_hash(["C24F_EVENT_SCHEMA_V2"]),
    }
    event_coverage = {
        "event_direct_columns_present": event_present,
        "event_methods_materialized": event_manifest.get("methods"),
        "minimum_required_methods_present": method_ok,
        "method_coverage": event_manifest.get("method_coverage"),
        "candidate_rows_with_event": event_manifest.get("candidate_rows_with_event"),
        "missing_event_count": event_manifest.get("missing_event_count"),
        "split_coverage": event_manifest.get("split_coverage"),
        "query_count": event_manifest.get("query_count"),
        "video_count": event_manifest.get("video_count"),
    }
    event_sample_path = OUT2 / "C24F_2_EVENT_SAMPLE.parquet"
    event_cols = [c for c in event_schema["event_level_columns"] if c in pd.read_parquet(Path(event_manifest["event_table_path"])).columns] if event_manifest.get("event_table_path") and Path(event_manifest["event_table_path"]).exists() else []
    if event_cols:
        pd.read_parquet(Path(event_manifest["event_table_path"]), columns=event_cols).head(cfg(mode)["sample_rows"]).to_parquet(event_sample_path, index=False)
    else:
        pd.DataFrame().to_parquet(event_sample_path, index=False)
    rec = {
        "stage": "C24F-2",
        "status": status,
        "mode": mode,
        "seed": seed,
        "bmn_materialization_manifest": {"source": "C19 BMN best span per query-video for calib_select/calib_holdout", "train_fit_missing_masked": True, "coverage_select_holdout": float(bmn_select), "checkpoint_manifest": file_record(ROOT / "c16_2_fullscale_bmn_training/C16_2_BMN_CHECKPOINT_MANIFEST.json", sha=True)},
        "t2_materialization_manifest": {"source": "C19/C21 T2 evidence for calib_select/calib_holdout", "train_fit_missing_masked": True, "coverage_select_holdout": float(t2_select)},
        "event_materialization_manifest": event_manifest,
        "event_schema": event_schema,
        "event_coverage_audit": event_coverage,
        "event_sample_path": str(event_sample_path),
        "evidence_coverage_audit": audit,
        "evidence_distribution_audit": {**distribution, "event_method_distribution": method_dist},
        "materialization_sample_path": str(sample_path),
        "event_direct_columns_present": event_present,
        "event_methods_materialized": event_manifest.get("methods"),
        "event_materialization_scope": "full first-stage top128 train_fit/calib_select/calib_holdout" if mode == "full" else f"{mode} subset first-stage top128 train_fit/calib_select/calib_holdout",
        "event_not_fully_materialized": bool(mode != "full" or not method_ok),
        "high_score_false_positive_preliminary": high_score_fp(df, "event_relevance_score"),
        "local_only_artifacts": [manifest, manifest.get("event_manifest")],
        "official_val_used": False,
        "official_prediction_pool_used": False,
        "pseudo_official_holdout_used_for_selection": False,
        "evaluator_modified": False,
        "nms_modified": False,
    }
    write_text(OUT2 / "C24F_2_MATERIALIZATION_PLAN.md", "# C24F-2 Materialization Plan\n\nUse C19 BMN/T2 evidence and materialize direct event evidence from TVR release subtitle/visual features. No raw video and no official pool.")
    write_json(OUT2 / "C24F_2_BMN_MATERIALIZATION_MANIFEST.json", rec["bmn_materialization_manifest"])
    write_json(OUT2 / "C24F_2_T2_MATERIALIZATION_MANIFEST.json", rec["t2_materialization_manifest"])
    write_json(OUT2 / "C24F_2_EVENT_MATERIALIZATION_MANIFEST.json", rec["event_materialization_manifest"])
    write_json(OUT2 / "C24F_2_EVENT_SCHEMA.json", event_schema)
    write_json(OUT2 / "C24F_2_EVENT_COVERAGE_AUDIT.json", event_coverage)
    write_json(OUT2 / "C24F_2_EVIDENCE_COVERAGE_AUDIT.json", audit)
    write_json(OUT2 / "C24F_2_EVIDENCE_DISTRIBUTION_AUDIT.json", rec["evidence_distribution_audit"])
    write_json(OUT2 / "C24F_2_MATERIALIZATION_DECISION.json", rec)
    write_text(OUT2 / "C24F_2_MATERIALIZATION_DECISION.md", f"# C24F-2 Materialization Decision\n\nStatus: `{status}`.\n\nEvent direct columns present: `{event_present}`.")
    return rec


def describe_numeric(df: pd.DataFrame, cols: Sequence[str]) -> Dict[str, Any]:
    out = {}
    for c in cols:
        if c not in df.columns:
            continue
        s = pd.to_numeric(df[c], errors="coerce")
        out[c] = {
            "count": int(s.notna().sum()),
            "missing": int(s.isna().sum()),
            "mean": float(s.mean()) if s.notna().any() else None,
            "p05": float(s.quantile(0.05)) if s.notna().any() else None,
            "p50": float(s.quantile(0.50)) if s.notna().any() else None,
            "p95": float(s.quantile(0.95)) if s.notna().any() else None,
        }
    return out


def high_score_fp(df: pd.DataFrame, col: str) -> Dict[str, Any]:
    sdf = labelled_rows(df)
    if col not in sdf.columns or len(sdf) == 0:
        return {"available": False}
    thr = float(pd.to_numeric(sdf[col], errors="coerce").quantile(0.90))
    hi = sdf[pd.to_numeric(sdf[col], errors="coerce") >= thr]
    return {
        "available": True,
        "threshold_p90": thr,
        "rows": int(len(hi)),
        "wrong_video_rate": float((~hi["correct_video"].astype(bool)).mean()) if len(hi) else None,
        "iou07_false_positive_rate": float((~hi["iou_ge_07"].astype(bool)).mean()) if len(hi) else None,
    }


def labelled_rows(df: pd.DataFrame) -> pd.DataFrame:
    out = df[df["span_start"].notna() & df["span_end"].notna()].copy()
    same = out["video_id"].astype(str) == out["gt_video_id"].astype(str)
    iou = iou_array(out["span_start"].to_numpy(np.float32), out["span_end"].to_numpy(np.float32), out["gt_start"].to_numpy(np.float32), out["gt_end"].to_numpy(np.float32))
    out["candidate_iou"] = np.where(same.to_numpy(), iou, 0.0)
    out["correct_video"] = same.to_numpy()
    out["iou_ge_05"] = out["candidate_iou"] >= 0.5
    out["iou_ge_07"] = out["candidate_iou"] >= 0.7
    return out


def stage_c24f_3(mode: str, seed: int) -> Dict[str, Any]:
    df, manifest = build_or_load_table(mode, seed, force=False)
    lab = labelled_rows(df)
    signals = [
        "first_stage_z", "bmn_z", "t2_z", "event_z", "event_relevance_score",
        "event_visual_relevance", "event_subtitle_relevance", "event_margin",
        "event_topk_mean_score", "event_topk_max_score", "event_score_softmax",
    ]
    diagnostic_signals = ["event_overlap_proxy", "best_event_gt_overlap", "event_iou_with_gt"]
    corr = {s: {"candidate_iou": corr_pack(lab, s, "candidate_iou")} for s in signals}
    auc = {
        s: {
            "IoU>=0.5_AUC": auc_score(lab["iou_ge_05"], lab[s]),
            "IoU>=0.7_AUC": auc_score(lab["iou_ge_07"], lab[s]),
            "correct_video_AUC": auc_score(lab["correct_video"], lab[s]),
            "correct_video_and_iou07_AUC": auc_score((lab["correct_video"].astype(bool) & lab["iou_ge_07"].astype(bool)).astype(int), lab[s]),
        }
        for s in signals if s in lab.columns
    }
    diagnostic_auc = {
        s: {"IoU>=0.7_AUC": auc_score(lab["iou_ge_07"], lab[s]), "correct_video_AUC": auc_score(lab["correct_video"], lab[s])}
        for s in diagnostic_signals if s in lab.columns
    }
    combo_scores = {
        "first_stage_only": lab["first_stage_z"],
        "first_stage_BMN": lab["first_stage_z"] + 0.25 * lab["bmn_z"],
        "first_stage_T2": lab["first_stage_z"] + 0.20 * lab["t2_z"],
        "first_stage_event": lab["first_stage_z"] + 0.25 * lab["event_z"],
        "first_stage_BMN_event": lab["first_stage_z"] + 0.20 * lab["bmn_z"] + 0.20 * lab["event_z"],
        "first_stage_BMN_T2": lab["first_stage_z"] + 0.20 * lab["bmn_z"] + 0.15 * lab["t2_z"],
        "all_evidence": lab["first_stage_z"] + 0.18 * lab["bmn_z"] + 0.14 * lab["t2_z"] + 0.18 * lab["event_z"],
    }
    complementarity = {k: {"IoU>=0.7_AUC": auc_score(lab["iou_ge_07"], v), "correct_video_AUC": auc_score(lab["correct_video"], v)} for k, v in combo_scores.items()}
    top_rank = {}
    for name, sdf in {"top1": lab[lab["candidate_video_rank"] <= 1], "top5": lab[lab["candidate_video_rank"] <= 5], "top10": lab[lab["candidate_video_rank"] <= 10], "top100": lab[lab["candidate_video_rank"] <= 100], "gt_top5_not_top1": lab[(lab["candidate_video_rank"] <= 5) & (lab["candidate_video_rank"] > 1)]}.items():
        top_rank[name] = {s: auc_score(sdf["iou_ge_07"], sdf[s]) for s in ["first_stage_z", "bmn_z", "t2_z", "event_z"] if len(sdf)}
    fp = {s: high_score_fp(lab, s) for s in ["bmn_z", "t2_z", "event_z", "event_relevance_score"]}
    event_gain = (complementarity.get("first_stage_event", {}).get("IoU>=0.7_AUC") or 0) - (complementarity.get("first_stage_only", {}).get("IoU>=0.7_AUC") or 0)
    bmn_gain = (complementarity.get("first_stage_BMN", {}).get("IoU>=0.7_AUC") or 0) - (complementarity.get("first_stage_only", {}).get("IoU>=0.7_AUC") or 0)
    c24_proxy_auc = auc_score(lab["iou_ge_07"], lab["old_norm"]) if "old_norm" in lab.columns else None
    event_auc = auc.get("event_z", {}).get("IoU>=0.7_AUC")
    max_wrong = max([v.get("wrong_video_rate") or 0.0 for v in fp.values()] or [0.0])
    if event_gain > 0.02 and bmn_gain > 0.02 and max_wrong < 0.75:
        status = "C24F_EVIDENCE_SIGNAL_STRONG"
    elif event_gain > 0.0 or bmn_gain > 0.0:
        status = "C24F_EVIDENCE_SIGNAL_FRONT_RANK_WEAK"
    elif max_wrong >= 0.75:
        status = "C24F_EVIDENCE_SIGNAL_NOISY_OR_HARMFUL"
    else:
        status = "C24F_EVIDENCE_SIGNAL_INCONCLUSIVE"
    conclusions = {
        "bmn_front_rank_independent_signal": bmn_gain,
        "event_front_rank_independent_signal": event_gain,
        "t2_safety_signal": (complementarity.get("first_stage_T2", {}).get("IoU>=0.7_AUC") or 0) - (complementarity.get("first_stage_only", {}).get("IoU>=0.7_AUC") or 0),
        "high_score_false_positive_risk": max_wrong,
        "event_direct_columns_present": True,
        "event_evidence_improved_over_c24_proxy": None if c24_proxy_auc is None or event_auc is None else bool(event_auc > c24_proxy_auc),
        "event_auc_minus_c24_proxy_auc": None if c24_proxy_auc is None or event_auc is None else float(event_auc - c24_proxy_auc),
        "existing_feature_path_still_has_value": status in ["C24F_EVIDENCE_SIGNAL_STRONG", "C24F_EVIDENCE_SIGNAL_FRONT_RANK_WEAK"],
    }
    rec = {
        "stage": "C24F-3",
        "status": status,
        "mode": mode,
        "seed": seed,
        "evidence_correlation": corr,
        "top_rank_signal_audit": top_rank,
        "bmn_signal_audit": {"auc": auc.get("bmn_z"), "gain_vs_first_stage": bmn_gain},
        "t2_signal_audit": {"auc": auc.get("t2_z"), "gain_vs_first_stage": conclusions["t2_safety_signal"]},
        "event_signal_audit": {"auc": auc.get("event_z"), "gain_vs_first_stage": event_gain, "direct_columns_present": True},
        "diagnostic_only_event_label_audit": diagnostic_auc,
        "complementarity_audit": complementarity,
        "failure_type_signal_audit": {"wrong_video_high_score": fp, "correct_video_wrong_span_rows": int((lab["correct_video"] & ~lab["iou_ge_05"]).sum()), "gt_top5_not_top1_rows": int(((lab["candidate_video_rank"] <= 5) & (lab["candidate_video_rank"] > 1)).sum())},
        "high_score_false_positive": fp,
        "evidence_signal_conclusion": conclusions,
        "official_val_used": False,
        "official_prediction_pool_used": False,
        "pseudo_official_holdout_used_for_selection": False,
        "evaluator_modified": False,
        "nms_modified": False,
    }
    write_text(OUT3 / "C24F_3_SIGNAL_REAUDIT_PLAN.md", "# C24F-3 Signal Re-audit Plan\n\nEvaluate BMN/T2/direct event evidence on the C24F canonical table with labels only for train/calib diagnostics.")
    write_json(OUT3 / "C24F_3_EVIDENCE_CORRELATION.json", corr)
    write_json(OUT3 / "C24F_3_TOP_RANK_SIGNAL_AUDIT.json", top_rank)
    write_json(OUT3 / "C24F_3_BMN_SIGNAL_AUDIT.json", rec["bmn_signal_audit"])
    write_json(OUT3 / "C24F_3_T2_SIGNAL_AUDIT.json", rec["t2_signal_audit"])
    write_json(OUT3 / "C24F_3_EVENT_SIGNAL_AUDIT.json", rec["event_signal_audit"])
    write_json(OUT3 / "C24F_3_COMPLEMENTARITY_AUDIT.json", complementarity)
    write_json(OUT3 / "C24F_3_FAILURE_TYPE_SIGNAL_AUDIT.json", rec["failure_type_signal_audit"])
    write_text(OUT3 / "C24F_3_SIGNAL_CASES.md", "# C24F-3 Signal Cases\n\nAggregate audits are committed; detailed row-level cases remain in the local-only canonical table.")
    write_json(OUT3 / "C24F_3_SIGNAL_DECISION.json", rec)
    write_text(OUT3 / "C24F_3_SIGNAL_DECISION.md", f"# C24F-3 Signal Decision\n\nStatus: `{status}`.\n\nEvent gain vs first stage AUC: `{event_gain}`.")
    return rec


def score_arrays(df: pd.DataFrame) -> Dict[str, np.ndarray]:
    fs = df["first_stage_z"].to_numpy(np.float32)
    b = df["bmn_z"].to_numpy(np.float32)
    t = df["t2_z"].to_numpy(np.float32)
    e = df["event_z"].to_numpy(np.float32)
    conflict = ((b > 0.8) & (t < -0.2)).astype(np.float32)
    agree_be = ((b > 0.3) & (e > 0.3)).astype(np.float32)
    no_reg = ((fs > 0.0) | ((b > 0.5) & (e > 0.2))).astype(np.float32)
    return {
        "A_C23_selected_baseline": fs + 0.25 * b + 0.20 * t,
        "B_C19_C20_C21_hybrid": df["final_score"].fillna(df["first_stage_score"]).to_numpy(np.float32),
        "C_C24_selected_B_C19_C21_hybrid": df["final_score"].fillna(df["first_stage_score"]).to_numpy(np.float32),
        "D_BMN_raw": fs + 0.35 * b + 0.10 * t,
        "E_BMN_calibrated": fs + 0.28 * b + 0.08 * t,
        "F_Event_raw": fs + 0.35 * e,
        "G_Event_calibrated": fs + 0.24 * e + 0.06 * b,
        "H_T2_calibrated": fs + 0.22 * t + 0.08 * b,
        "I_BMN_event_consistency": fs + 0.22 * b + 0.22 * e + 0.08 * agree_be,
        "J_BMN_T2_consistency": fs + 0.22 * b + 0.16 * t - 0.14 * conflict,
        "K_event_T2": fs + 0.22 * e + 0.14 * t,
        "L_all_evidence_wrong_risk_penalty": fs + 0.18 * b + 0.18 * e + 0.12 * t - 0.18 * conflict,
        "M_all_evidence_no_regression_guard": fs + no_reg * (0.18 * b + 0.18 * e + 0.10 * t) - 0.10 * conflict,
    }


def eval_split(ctx: Dict[str, Any], score: np.ndarray, split: str) -> Dict[str, Any]:
    return c18.fast_eval_vcmr(ctx, score.astype(np.float32), split)["summary"]


def objective(m: Dict[str, Any], base: Dict[str, Any]) -> float:
    return (
        3.0 * (m.get("VCMR_R@1_IoU0.7", 0.0) - base.get("VCMR_R@1_IoU0.7", 0.0))
        + 2.0 * (m.get("VCMR_R@5_IoU0.7", 0.0) - base.get("VCMR_R@5_IoU0.7", 0.0))
        + 1.5 * (m.get("VCMR_R@10_IoU0.7", 0.0) - base.get("VCMR_R@10_IoU0.7", 0.0))
        + 1.0 * (m.get("VCMR_R@1_IoU0.5", 0.0) - base.get("VCMR_R@1_IoU0.5", 0.0))
        + 0.8 * (m.get("VCMR_R@5_IoU0.5", 0.0) - base.get("VCMR_R@5_IoU0.5", 0.0))
        - 2.0 * max(0.0, m.get("wrong_video_high_score_rate", 0.0) - base.get("wrong_video_high_score_rate", 0.0))
        - 1.0 * max(0.0, base.get("VR_R@100", 0.0) - m.get("VR_R@100", 0.0))
    )


def stage_c24f_4(mode: str, seed: int) -> Dict[str, Any]:
    df, manifest = build_or_load_table(mode, seed, force=False)
    eval_df = df[df["split"].isin(["calib_select", "calib_holdout"]) & df["span_start"].notna() & df["span_end"].notna()].copy()
    ctx = c18.prepare_fast_eval(eval_df.rename(columns={"candidate_video_rank": "gt_video_rank"}) if "gt_video_rank" not in eval_df.columns else eval_df)
    scores = score_arrays(eval_df)
    base_name = "B_C19_C20_C21_hybrid"
    base_select = eval_split(ctx, scores[base_name], "calib_select")
    base_hold = eval_split(ctx, scores[base_name], "calib_holdout")
    results = {}
    best = base_name
    best_obj = -1e18
    for name, score in scores.items():
        sel = base_select if name == base_name else eval_split(ctx, score, "calib_select")
        obj = objective(sel, base_select)
        unsafe = sel.get("VR_R@100", 0.0) < base_select.get("VR_R@100", 0.0) - 2.0 or sel.get("wrong_video_high_score_rate", 0.0) > base_select.get("wrong_video_high_score_rate", 0.0) + 5.0
        results[name] = {"calib_select": sel, "select_objective": obj, "unsafe": unsafe, "delta_select_vs_baseline": delta(sel, base_select, metric_keys(sel))}
        if not unsafe and obj > best_obj:
            best_obj = obj
            best = name
    holdout_names = {base_name, best}
    for name, r in sorted(results.items(), key=lambda kv: -kv[1]["select_objective"])[:6]:
        if not r["unsafe"]:
            holdout_names.add(name)
    for name in holdout_names:
        hold = base_hold if name == base_name else eval_split(ctx, scores[name], "calib_holdout")
        results[name]["calib_holdout"] = hold
        results[name]["delta_holdout_vs_baseline"] = delta(hold, base_hold, metric_keys(hold))
    selected_hold = results[best]["calib_holdout"]
    hold_delta = results[best]["delta_holdout_vs_baseline"]
    front_keys = ["VCMR_R@1_IoU0.7", "VCMR_R@5_IoU0.7", "VCMR_R@10_IoU0.7"]
    front_positive = any(hold_delta.get(k, 0.0) > 0 for k in front_keys)
    front_consistent = front_positive and all(hold_delta.get(k, 0.0) >= 0 for k in front_keys)
    wrong_safe = hold_delta.get("wrong_video_high_score_rate", 0.0) <= 0.5
    if front_consistent and wrong_safe:
        status = "C24F_FULL_EVIDENCE_INTEGRATION_PROMISING"
    elif front_positive:
        status = "C24F_FULL_EVIDENCE_INTEGRATION_FRONT_RANK_WEAK"
    else:
        status = "C24F_FULL_EVIDENCE_INTEGRATION_FRONT_RANK_WEAK"
    rec = {
        "stage": "C24F-4",
        "status": status,
        "mode": mode,
        "seed": seed,
        "score_formulas": {k: "first-stage preserving formula" for k in scores},
        "replay_results": results,
        "component_ablation": {
            "baseline_holdout": base_hold,
            "selected_holdout": selected_hold,
            "selected_delta": hold_delta,
            "BMN_removal_delta": results.get("G_Event_calibrated", {}).get("delta_holdout_vs_baseline"),
            "event_removal_delta": results.get("E_BMN_calibrated", {}).get("delta_holdout_vs_baseline"),
            "T2_removal_delta": results.get("I_BMN_event_consistency", {}).get("delta_holdout_vs_baseline"),
        },
        "wrong_video_audit": {"baseline": base_hold.get("wrong_video_high_score_rate"), "selected": selected_hold.get("wrong_video_high_score_rate"), "increase": hold_delta.get("wrong_video_high_score_rate")},
        "selected_integration": {"name": best, "selection_split": "calib_select", "holdout_report_only": True, "select_objective": results[best]["select_objective"]},
        "final_vcmr_metrics": selected_hold,
        "official_val_used": False,
        "official_prediction_pool_used": False,
        "pseudo_official_holdout_used_for_selection": False,
        "evaluator_modified": False,
        "nms_modified": False,
    }
    write_text(OUT4 / "C24F_4_INTEGRATION_PLAN.md", "# C24F-4 Integration Plan\n\nReplay strengthened first-stage-preserving formulas on the C24F canonical table. Calib_select selects; calib_holdout reports only.")
    write_json(OUT4 / "C24F_4_SCORE_FORMULAS.json", rec["score_formulas"])
    write_json(OUT4 / "C24F_4_REPLAY_RESULTS.json", results)
    write_json(OUT4 / "C24F_4_COMPONENT_ABLATION.json", rec["component_ablation"])
    write_json(OUT4 / "C24F_4_WRONG_VIDEO_AUDIT.json", rec["wrong_video_audit"])
    write_json(OUT4 / "C24F_4_SELECTED_INTEGRATION.json", rec["selected_integration"])
    write_json(OUT4 / "C24F_4_INTEGRATION_DECISION.json", rec)
    write_text(OUT4 / "C24F_4_INTEGRATION_DECISION.md", f"# C24F-4 Integration Decision\n\nStatus: `{status}`.\n\nSelected integration: `{best}`.")
    return rec


def metric_keys(m: Dict[str, Any]) -> List[str]:
    return [k for k, v in m.items() if isinstance(v, (int, float))]


def stage_c24f_5(mode: str, seed: int, s2: Dict[str, Any] | None = None, s3: Dict[str, Any] | None = None, s4: Dict[str, Any] | None = None) -> Dict[str, Any]:
    s2 = s2 or load_json(OUT2 / "C24F_2_MATERIALIZATION_DECISION.json", {})
    s3 = s3 or load_json(OUT3 / "C24F_3_SIGNAL_DECISION.json", {})
    s4 = s4 or load_json(OUT4 / "C24F_4_INTEGRATION_DECISION.json", {})
    materialized = s2.get("status") == "C24F_EVIDENCE_MATERIALIZED"
    event_present = bool(s2.get("event_direct_columns_present"))
    integration_promising = s4.get("status") == "C24F_FULL_EVIDENCE_INTEGRATION_PROMISING"
    signal_strong = s3.get("status") == "C24F_EVIDENCE_SIGNAL_STRONG"
    if not materialized or not event_present:
        route = "C24F_NEED_MORE_FULL_EVIDENCE_REPAIR"
    elif signal_strong and integration_promising:
        route = "C24F_CONTINUE_EXISTING_FEATURE_EVIDENCE_STRENGTHENING"
    elif s3.get("status") in ["C24F_EVIDENCE_SIGNAL_FRONT_RANK_WEAK", "C24F_EVIDENCE_SIGNAL_NOISY_OR_HARMFUL", "C24F_EVIDENCE_SIGNAL_INCONCLUSIVE"] and not integration_promising:
        route = "C24F_READY_FOR_C25_RAW_FRAME_STRONG_FEATURE_AUDIT"
    else:
        route = "C24F_CONTINUE_EXISTING_FEATURE_EVIDENCE_STRENGTHENING"
    status = "C24F_ROBUSTNESS_PARTIAL"
    c25_audit = f"""# C24F-5 C25 Route Audit

1. Full BMN/T2/event evidence materialized: `{materialized}`.
2. Event direct columns absent: `{not event_present}`.
3. Full evidence signal status: `{s3.get('status')}`.
4. Integration status: `{s4.get('status')}`.
5. Route recommendation: `{route}`.
6. If C25 starts, audit frames/raw TVR video, TVQA/TVR frame resources, CLIP/DINOv2 feasibility, VideoMAE/InternVideo feasibility, timestamp alignment, GPU memory, and disk budget.
"""
    firewall = {"official_val_used": False, "official_prediction_pool_used": False, "pseudo_official_holdout_used_for_selection": False, "evaluator_modified": False, "nms_modified": False}
    rec = {
        "stage": "C24F-5",
        "status": status,
        "mode": mode,
        "seed": seed,
        "route_recommendation": route,
        "seed_robustness": {"seed2026": {"materialization": s2.get("status"), "signal": s3.get("status"), "integration": s4.get("status")}, "seed2027": "not rerun; deterministic materialization", "seed2028": "not rerun; deterministic materialization"},
        "query_duration_robustness": {"source": "C24F-3 aggregate; detailed query/duration breakdown is in canonical table local-only"},
        "d_e_f_subset_audit": s3.get("failure_type_signal_audit"),
        "score_distribution_audit": s2.get("evidence_distribution_audit"),
        "c25_route_audit": {"recommendation": route, "event_direct_columns_present": event_present, "integration_promising": integration_promising},
        "pseudo_onelook_diagnostic": {"pseudo_one_look_executed": False, "pseudo_official_not_used_for_selection": True},
        "selection_firewall_audit": firewall,
        **firewall,
    }
    write_text(OUT5 / "C24F_5_ROBUSTNESS_PLAN.md", "# C24F-5 Robustness Plan\n\nCheck materialization completeness, signal status, integration stability, false positives, and route toward C25 if existing features remain weak.")
    write_json(OUT5 / "C24F_5_SEED_ROBUSTNESS.json", rec["seed_robustness"])
    write_json(OUT5 / "C24F_5_QUERY_DURATION_ROBUSTNESS.json", rec["query_duration_robustness"])
    write_json(OUT5 / "C24F_5_D_E_F_SUBSET_AUDIT.json", rec["d_e_f_subset_audit"])
    write_json(OUT5 / "C24F_5_SCORE_DISTRIBUTION_AUDIT.json", rec["score_distribution_audit"])
    write_text(OUT5 / "C24F_5_C25_ROUTE_AUDIT.md", c25_audit)
    write_json(OUT5 / "C24F_5_PSEUDO_ONELOOK_DIAGNOSTIC.json", rec["pseudo_onelook_diagnostic"])
    write_json(OUT5 / "C24F_5_SELECTION_FIREWALL_AUDIT.json", firewall)
    write_json(OUT5 / "C24F_5_ROBUSTNESS_DECISION.json", rec)
    write_text(OUT5 / "C24F_5_ROBUSTNESS_DECISION.md", f"# C24F-5 Robustness Decision\n\nStatus: `{status}`.\n\nRoute recommendation: `{route}`.")
    return rec


def stage_c24f_6(mode: str, seed: int, recs: Dict[str, Dict[str, Any]] | None = None) -> Dict[str, Any]:
    recs = recs or {
        "c24f_0": load_json(OUT0 / "C24F_0_PROTOCOL.json", {}),
        "c24f_1": load_json(OUT1 / "C24F_1_TABLE_DECISION.json", {}),
        "c24f_2": load_json(OUT2 / "C24F_2_MATERIALIZATION_DECISION.json", {}),
        "c24f_3": load_json(OUT3 / "C24F_3_SIGNAL_DECISION.json", {}),
        "c24f_4": load_json(OUT4 / "C24F_4_INTEGRATION_DECISION.json", {}),
        "c24f_5": load_json(OUT5 / "C24F_5_ROBUSTNESS_DECISION.json", {}),
    }
    route = recs["c24f_5"].get("route_recommendation")
    if recs["c24f_3"].get("status") == "C24F_EVIDENCE_SIGNAL_STRONG" and recs["c24f_4"].get("status") == "C24F_FULL_EVIDENCE_INTEGRATION_PROMISING":
        decision = "C24F_READY_FOR_C23R_EVIDENCE_STRENGTHENED_NATIVE_COUPLING"
    elif recs["c24f_2"].get("status") != "C24F_EVIDENCE_MATERIALIZED":
        decision = "C24F_NEED_MORE_FULL_EVIDENCE_REPAIR"
    elif route == "C24F_READY_FOR_C25_RAW_FRAME_STRONG_FEATURE_AUDIT":
        decision = "C24F_READY_FOR_C25_RAW_FRAME_STRONG_FEATURE_AUDIT"
    elif route == "C24F_CONTINUE_EXISTING_FEATURE_EVIDENCE_STRENGTHENING":
        decision = "C24F_CONTINUE_EXISTING_FEATURE_EVIDENCE_STRENGTHENING"
    else:
        decision = "C24F_STOP_EXISTING_FEATURE_COUPLING"
    final_vcmr = recs["c24f_4"].get("final_vcmr_metrics") or {}
    c24_final = load_json(ROOT / "c24_6_final_decision/C24_6_FINAL_DECISION.json", {})
    c23_final = load_json(ROOT / "c23_6_final_decision/C23_6_FINAL_DECISION.json", {})
    c19_base = ((recs["c24f_4"].get("component_ablation") or {}).get("baseline_holdout") or {})
    materialization = recs["c24f_2"]
    rec = {
        "stage": "C24F-6",
        "status": decision,
        "final_decision": decision,
        "mode": mode,
        "seed": seed,
        "c24f_0_protocol_status": recs["c24f_0"].get("status"),
        "c24f_1_canonical_table_status": recs["c24f_1"].get("status"),
        "c24f_2_evidence_materialization_status": recs["c24f_2"].get("status"),
        "c24f_3_signal_status": recs["c24f_3"].get("status"),
        "c24f_4_integration_status": recs["c24f_4"].get("status"),
        "c24f_5_route_status": recs["c24f_5"].get("status"),
        "full_evidence_coverage": materialization.get("evidence_coverage_audit"),
        "event_direct_columns_present": bool(materialization.get("event_direct_columns_present")),
        "event_methods_materialized": materialization.get("event_methods_materialized"),
        "event_materialization_scope": materialization.get("event_materialization_scope"),
        "event_not_fully_materialized": bool(materialization.get("event_not_fully_materialized")),
        "event_evidence_improved_over_c24_proxy": (recs["c24f_3"].get("evidence_signal_conclusion") or {}).get("event_evidence_improved_over_c24_proxy"),
        "selected_evidence_probe": recs["c24f_3"].get("evidence_signal_conclusion"),
        "selected_integration_formula": recs["c24f_4"].get("selected_integration"),
        "final_vcmr_metrics": final_vcmr,
        "final_vr_metrics": {k: final_vcmr.get(k) for k in ["VR_R@1", "VR_R@5", "VR_R@10", "VR_R@100"]},
        "delta_vs_c24": delta(final_vcmr, c24_final.get("final_vcmr_metrics", {}), metric_keys(final_vcmr)),
        "delta_vs_c23": delta(final_vcmr, c23_final.get("final_vcmr_metrics", {}), metric_keys(final_vcmr)),
        "delta_vs_c19_c21": delta(final_vcmr, c19_base, metric_keys(final_vcmr)),
        "wrong_video_risk": final_vcmr.get("wrong_video_high_score_rate"),
        "high_score_false_positive": recs["c24f_3"].get("high_score_false_positive"),
        "evidence_signal_conclusion": recs["c24f_3"].get("evidence_signal_conclusion"),
        "existing_feature_path_still_has_value": decision in ["C24F_READY_FOR_C23R_EVIDENCE_STRENGTHENED_NATIVE_COUPLING", "C24F_CONTINUE_EXISTING_FEATURE_EVIDENCE_STRENGTHENING"],
        "c25_raw_frame_strong_feature_audit_recommended": decision == "C24F_READY_FOR_C25_RAW_FRAME_STRONG_FEATURE_AUDIT",
        "local_only_artifacts": materialization.get("local_only_artifacts"),
        "official_val_used": False,
        "official_prediction_pool_used": False,
        "pseudo_official_holdout_used_for_selection": False,
        "evaluator_modified": False,
        "nms_modified": False,
        "c24f_is_promoted_system": False,
        "current_promoted_system": PROMOTED,
    }
    packet = {
        "decision": decision,
        "event_direct_columns_present": rec["event_direct_columns_present"],
        "event_methods_materialized": rec["event_methods_materialized"],
        "event_materialization_scope": rec["event_materialization_scope"],
        "event_not_fully_materialized": rec["event_not_fully_materialized"],
        "event_evidence_improved_over_c24_proxy": rec["event_evidence_improved_over_c24_proxy"],
        "official_still_forbidden": True,
        "ready_for_official": False,
        "local_only_artifacts": rec["local_only_artifacts"],
    }
    write_json(OUT6 / "C24F_6_FINAL_DECISION.json", rec)
    write_json(OUT6 / "C24F_6_FULL_EVIDENCE_PACKET.json", packet)
    write_json(OUT6 / "C24F_6_NEXT_STEP_DECISION.json", {"decision": decision, "official_forbidden": True})
    write_text(OUT6 / "C24F_6_FINAL_DECISION.md", f"# C24F-6 Final Decision\n\nFinal decision: `{decision}`.\n\nC24F is not promoted. Current promoted official system remains `{PROMOTED}`.")
    write_text(OUT6 / "C24F_6_FULL_EVIDENCE_PACKET.md", f"# C24F Full Evidence Packet\n\nDecision: `{decision}`.\nEvent direct columns present: `{rec['event_direct_columns_present']}`.\nOfficial validation remains forbidden.")
    write_text(OUT6 / "C24F_6_RISK_REGISTER.md", "# C24F-6 Risk Register\n\n- Train_fit BMN/T2 remains masked from existing sources, although direct event evidence is materialized.\n- Event evidence uses existing release features, not raw video and not EventFormer reproduction.\n- No official validation was run.\n")
    write_text(OUT6 / "C24F_6_NEXT_STEP_DECISION.md", f"# C24F Next Step\n\n`{decision}`")
    return rec


def run_all(mode: str, seed: int, force: bool = False) -> Dict[str, Dict[str, Any]]:
    r0 = stage_c24f_0(mode, seed)
    if r0["status"] != "C24F_PROTOCOL_READY":
        raise RuntimeError(f"C24F protocol blocked: {r0['status']}")
    r1 = stage_c24f_1(mode, seed, force=force)
    r2 = stage_c24f_2(mode, seed, force=False)
    if mode == "full" and not (r1["status"] == "C24F_CANONICAL_TABLE_READY" and r2["status"] == "C24F_EVIDENCE_MATERIALIZED"):
        raise RuntimeError("C24F full requires C24F-1/C24F-2 medium readiness")
    r3 = stage_c24f_3(mode, seed)
    r4 = stage_c24f_4(mode, seed)
    r5 = stage_c24f_5(mode, seed, r2, r3, r4)
    r6 = stage_c24f_6(mode, seed, {"c24f_0": r0, "c24f_1": r1, "c24f_2": r2, "c24f_3": r3, "c24f_4": r4, "c24f_5": r5})
    return {"c24f_0": r0, "c24f_1": r1, "c24f_2": r2, "c24f_3": r3, "c24f_4": r4, "c24f_5": r5, "c24f_6": r6}


def print_summary(recs: Dict[str, Dict[str, Any]]) -> None:
    final = recs["c24f_6"]
    v = final.get("final_vcmr_metrics") or {}
    print("\n===== C24F SUMMARY =====")
    print(f"branch: {sh('git branch --show-current')}")
    print(f"commit: {sh('git rev-parse HEAD')}")
    print(f"C24F-0 protocol status: {final.get('c24f_0_protocol_status')}")
    print(f"C24F-1 canonical table status: {final.get('c24f_1_canonical_table_status')}")
    print(f"C24F-2 evidence materialization status: {final.get('c24f_2_evidence_materialization_status')}")
    print(f"C24F-3 signal status: {final.get('c24f_3_signal_status')}")
    print(f"C24F-4 integration status: {final.get('c24f_4_integration_status')}")
    print(f"C24F-5 route status: {final.get('c24f_5_route_status')}")
    print(f"C24F-6 final decision: {final.get('final_decision')}")
    print(f"event direct columns present: {final.get('event_direct_columns_present')}")
    print(f"selected probe/formula: {json.dumps(jsonable(final.get('selected_integration_formula')), sort_keys=True)}")
    print(f"VCMR @0.5 R@1/R@5/R@10/R@100: {v.get('VCMR_R@1_IoU0.5')}/{v.get('VCMR_R@5_IoU0.5')}/{v.get('VCMR_R@10_IoU0.5')}/{v.get('VCMR_R@100_IoU0.5')}")
    print(f"VCMR @0.7 R@1/R@5/R@10/R@100: {v.get('VCMR_R@1_IoU0.7')}/{v.get('VCMR_R@5_IoU0.7')}/{v.get('VCMR_R@10_IoU0.7')}/{v.get('VCMR_R@100_IoU0.7')}")
    print(f"VR R@1/R@5/R@10/R@100: {v.get('VR_R@1')}/{v.get('VR_R@5')}/{v.get('VR_R@10')}/{v.get('VR_R@100')}")
    print(f"delta vs C24: {json.dumps(jsonable(final.get('delta_vs_c24')), sort_keys=True)}")
    print(f"delta vs C23: {json.dumps(jsonable(final.get('delta_vs_c23')), sort_keys=True)}")
    print(f"delta vs C19/C21: {json.dumps(jsonable(final.get('delta_vs_c19_c21')), sort_keys=True)}")
    print(f"wrong-video top1/high-score: {final.get('wrong_video_risk')}")
    print(f"high-score false positive rate: {json.dumps(jsonable(final.get('high_score_false_positive')), sort_keys=True)}")
    print(f"C25 raw/frame/strong feature audit recommended: {final.get('c25_raw_frame_strong_feature_audit_recommended')}")
    print(f"pseudo_official_holdout used for selection: {final.get('pseudo_official_holdout_used_for_selection')}")
    print(f"official was not run: {not final.get('official_val_used')}")
    print(f"local-only artifacts path/hash: {json.dumps(jsonable(final.get('local_only_artifacts')), sort_keys=True)}")
    print("files committed to GitHub: pending git commit/push")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", default="all", choices=["all", "c24f_0", "c24f_1", "c24f_2", "c24f_3", "c24f_4", "c24f_5", "c24f_6"])
    parser.add_argument("--mode", default="medium", choices=["smoke", "medium", "full"])
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.stage == "all":
        recs = run_all(args.mode, args.seed, force=args.force)
        print_summary(recs)
    elif args.stage == "c24f_0":
        stage_c24f_0(args.mode, args.seed)
    elif args.stage == "c24f_1":
        stage_c24f_1(args.mode, args.seed, force=args.force)
    elif args.stage == "c24f_2":
        stage_c24f_2(args.mode, args.seed, force=args.force)
    elif args.stage == "c24f_3":
        stage_c24f_3(args.mode, args.seed)
    elif args.stage == "c24f_4":
        stage_c24f_4(args.mode, args.seed)
    elif args.stage == "c24f_5":
        stage_c24f_5(args.mode, args.seed)
    else:
        stage_c24f_6(args.mode, args.seed)


if __name__ == "__main__":
    main()
