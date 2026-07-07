from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

import run_c17_bmn_t2_native_vcmr_integration as c17


ROOT = Path(__file__).resolve().parents[2]
C26R_CACHE = Path("/tmp/c26r_score_cache/CONQUER-RLEM-c2c3")
C26S_CACHE = Path("/tmp/c26s_score_cache/CONQUER-RLEM-c2c3")
C26R_JOINED = C26R_CACHE / "C26R_C17_C26_JOINED_MULTI_SPAN_medium_seed2026.local.parquet"
CANONICAL = C26S_CACHE / "C26S_CANONICAL_MULTISPAN_medium_seed2026.local.parquet"


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
        return None if np.isnan(v) or np.isinf(v) else v
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def stable_hash(obj: Any) -> str:
    return hashlib.sha256(json.dumps(jsonable(obj), sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:16]


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


def canonical_columns() -> list[str]:
    return [
        "query_id", "video_id", "split", "candidate_video_rank", "first_stage_score", "first_stage_rank",
        "span_id", "span_start", "span_end", "span_duration", "span_source", "span_rank_within_video",
        "proposal_score_raw", "visual_feature_key", "subtitle_feature_key", "query_feature_key",
        "prem_video_score", "prem_visual_relevance_available", "prem_subtitle_relevance_available",
        "bmn_available", "t2_available", "event_available", "gt_video_id", "gt_start", "gt_end",
        "candidate_iou", "iou_ge_05", "iou_ge_07", "correct_video", "schema_hash", "config_hash",
    ]


def _prepare(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["first_stage_rank"] = pd.to_numeric(out.get("retriever_rank", out.get("candidate_video_rank", -1)), errors="coerce").fillna(-1).astype(np.int32)
    out["candidate_video_rank"] = pd.to_numeric(out.get("candidate_video_rank", out["first_stage_rank"]), errors="coerce").fillna(out["first_stage_rank"]).astype(np.int32)
    out["span_rank_within_video"] = out.groupby(["split", "query_id", "seed", "video_id"], observed=True)["R3b1_C26_PREM_guarded_localizer"].rank(method="first", ascending=False).astype(np.int32) if "R3b1_C26_PREM_guarded_localizer" in out else out.groupby(["split", "query_id", "seed", "video_id"], observed=True).cumcount().astype(np.int32) + 1
    out["span_id"] = out.groupby(["split", "query_id", "seed", "video_id"], observed=True).cumcount().astype(np.int32)
    out["span_source"] = "C17_C26R_multi_span"
    out["proposal_score_raw"] = pd.to_numeric(out.get("bmn_final_score"), errors="coerce").fillna(pd.to_numeric(out.get("t2_score"), errors="coerce")).fillna(0.0).astype(np.float32)
    out["visual_feature_key"] = out["video_id"].astype(str)
    out["subtitle_feature_key"] = out["video_id"].astype(str)
    out["query_feature_key"] = out["query_id"].astype(str)
    out["prem_video_score"] = pd.to_numeric(out.get("c26_final_F_score"), errors="coerce").fillna(0.0).astype(np.float32)
    out["prem_visual_relevance_available"] = pd.to_numeric(out.get("c26_visual_relevance"), errors="coerce").notna()
    out["prem_subtitle_relevance_available"] = pd.to_numeric(out.get("c26_subtitle_relevance"), errors="coerce").notna()
    out["bmn_available"] = pd.to_numeric(out.get("bmn_final_score"), errors="coerce").notna()
    out["t2_available"] = pd.to_numeric(out.get("t2_score"), errors="coerce").notna()
    out["event_available"] = "event_relevance_score" in out and pd.to_numeric(out.get("event_relevance_score"), errors="coerce").notna()
    out["candidate_iou"] = c17.iou_array(out["span_start"].to_numpy(np.float32), out["span_end"].to_numpy(np.float32), out["gt_start"].to_numpy(np.float32), out["gt_end"].to_numpy(np.float32)) if hasattr(c17, "iou_array") else _iou(out)
    same = out["video_id"].astype(str).to_numpy() == out["gt_video_id"].astype(str).to_numpy()
    out["candidate_iou"] = np.where(same, out["candidate_iou"].to_numpy(np.float32), 0.0).astype(np.float32)
    out["iou_ge_05"] = out["candidate_iou"] >= 0.5
    out["iou_ge_07"] = out["candidate_iou"] >= 0.7
    out["correct_video"] = same
    schema_hash = stable_hash(canonical_columns())
    out["schema_hash"] = schema_hash
    out["config_hash"] = stable_hash({"stage": "C26S-1", "source": str(C26R_JOINED)})
    return out


def _iou(df: pd.DataFrame) -> np.ndarray:
    s = df["span_start"].to_numpy(np.float32)
    e = df["span_end"].to_numpy(np.float32)
    gs = df["gt_start"].to_numpy(np.float32)
    ge = df["gt_end"].to_numpy(np.float32)
    inter = np.maximum(0.0, np.minimum(e, ge) - np.maximum(s, gs))
    union = np.maximum(e, ge) - np.minimum(s, gs)
    return inter / np.maximum(union, 1e-6)


def load_or_build_rowspace(mode: str, max_queries: int | None = None, force: bool = False) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    if mode != "medium":
        # Smoke/full currently reuse the C26R medium row-space with optional
        # query slicing; full requires a future full C17 multi-span build.
        pass
    C26S_CACHE.mkdir(parents=True, exist_ok=True)
    if CANONICAL.exists() and not force:
        df = pd.read_parquet(CANONICAL)
    else:
        df = pd.read_parquet(C26R_JOINED)
        df = _prepare(df)
        df.to_parquet(CANONICAL, index=False)
    if max_queries is not None and max_queries > 0:
        keep = []
        for split in sorted(df["split"].astype(str).unique()):
            qids = sorted(int(x) for x in df[df["split"].astype(str) == split]["query_id"].unique())[: int(max_queries)]
            keep.extend((split, q) for q in qids)
        pairs = set(keep)
        mask = [(str(s), int(q)) in pairs for s, q in zip(df["split"], df["query_id"])]
        df = df[mask].copy()
    manifest = {
        "source": str(C26R_JOINED),
        "local_path": str(CANONICAL),
        "row_count": int(len(df)),
        "query_count": int(df[["split", "query_id"]].drop_duplicates().shape[0]),
        "split_coverage": df.groupby("split", observed=True)["query_id"].nunique().to_dict(),
        "has_train_fit": bool((df["split"].astype(str) == "train_fit").any()) if len(df) else False,
        "schema_hash": stable_hash(canonical_columns()),
        "local_only": True,
    }
    return df, manifest


def proposal_distribution(df: pd.DataFrame) -> Dict[str, Any]:
    span_counts = df.groupby(["split", "query_id", "seed", "video_id"], observed=True).size()
    q_counts = df.groupby(["split", "query_id", "seed"], observed=True).size()
    topm = {}
    for m in [1, 2, 4, 8, 16, 32, 64]:
        topm[f"top{m}_per_video_rows"] = int(np.minimum(span_counts.to_numpy(), m).sum())
    return {
        "rows": int(len(df)),
        "query_video_pairs": int(span_counts.shape[0]),
        "span_count_per_query_video": {
            "min": int(span_counts.min()) if len(span_counts) else 0,
            "median": float(span_counts.median()) if len(span_counts) else 0.0,
            "p95": float(span_counts.quantile(0.95)) if len(span_counts) else 0.0,
            "max": int(span_counts.max()) if len(span_counts) else 0,
        },
        "span_rows_per_query": {
            "min": int(q_counts.min()) if len(q_counts) else 0,
            "median": float(q_counts.median()) if len(q_counts) else 0.0,
            "p95": float(q_counts.quantile(0.95)) if len(q_counts) else 0.0,
            "max": int(q_counts.max()) if len(q_counts) else 0,
        },
        "topm": topm,
    }

