from __future__ import annotations

import hashlib
import io
import json
import math
import os
import pickle
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import lmdb
import msgpack
import msgpack_numpy
import numpy as np
import pandas as pd

from c12_native_retriever.scaffold import C12Paths
from run_c12_native_retriever_training import ROOT, build_feature_caches, load_corpus, load_features, sha256_file


PROMOTED = "C7-B6 R1SelectiveTop1"
C26_CACHE = Path("/tmp/c26_score_cache/CONQUER-RLEM-c2c3")
SPLITS = ["train_fit", "calib_select", "calib_holdout"]
MODE_LIMITS = {
    "smoke": {"train_fit": 64, "calib_select": 32, "calib_holdout": 32, "top": 32, "sample_rows": 1024},
    "medium": {"train_fit": 1800, "calib_select": 900, "calib_holdout": 900, "top": 128, "sample_rows": 6000},
    "full": {"train_fit": 0, "calib_select": 0, "calib_holdout": 0, "top": 128, "sample_rows": 12000},
}


def jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
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


def stable_hash(obj: Any) -> str:
    raw = json.dumps(jsonable(obj), sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


def l2(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), eps)


def file_record(path: Path, sha: bool = False) -> Dict[str, Any]:
    rec = {
        "path": str(path),
        "exists": path.exists(),
        "readable": os.access(path, os.R_OK) if path.exists() else False,
        "size_bytes": path.stat().st_size if path.exists() and path.is_file() else None,
    }
    if sha and path.exists() and path.is_file() and path.stat().st_size < 64 * 1024 * 1024:
        rec["sha256"] = sha256_file(path)
    return rec


def mode_cfg(mode: str) -> Dict[str, int]:
    if mode not in MODE_LIMITS:
        raise ValueError(f"unknown C26 mode: {mode}")
    return dict(MODE_LIMITS[mode])


def limited_ids(corpus: Any, split: str, mode: str, max_queries: int | None = None) -> List[int]:
    ids = [int(x) for x in corpus.splits[split]]
    limit = int(max_queries or 0) if max_queries else int(mode_cfg(mode).get(split, 0))
    return ids[:limit] if limit and len(ids) > limit else ids


def first_stage_cache_candidates() -> List[Path]:
    d = ROOT / "results/c12_feature_cache"
    return [
        d / "first_stage_train_calib_holdout_top128.pkl",
        d / "first_stage_c17_medium_calib_select_top128.pkl",
        d / "first_stage_c17_medium_calib_holdout_top128.pkl",
    ]


def load_pickle(path: Path) -> Any:
    with path.open("rb") as f:
        return pickle.load(f)


def load_first_stage_split(corpus: Any, split: str, ids: Sequence[int], top: int) -> Tuple[Dict[int, Dict[str, Any]], Dict[str, Any]]:
    ids = [int(x) for x in ids]
    wanted = set(ids)
    candidates = first_stage_cache_candidates()
    preferred = {
        "train_fit": candidates[0],
        "calib_select": candidates[1],
        "calib_holdout": candidates[2],
    }.get(split, candidates[0])
    ordered = [preferred] + [p for p in candidates if p != preferred]
    best_obj: Dict[int, Any] = {}
    best_path: Path | None = None
    best_missing = len(wanted)
    for path in ordered:
        if not path.exists():
            continue
        obj = load_pickle(path)
        keys = {int(k) for k in obj.keys()}
        missing = len(wanted - keys)
        if missing < best_missing:
            best_obj = {int(k): v for k, v in obj.items()}
            best_path = path
            best_missing = missing
        if missing == 0:
            break
    if best_path is None:
        from run_c12_native_retriever_training import load_first_stage

        best_obj = load_first_stage(corpus, ids, top_keep=max(128, top), cache_name=f"first_stage_c26_{split}_top128.pkl")
        best_path = ROOT / "results/c12_feature_cache" / f"first_stage_c26_{split}_top128.pkl"
        best_missing = 0
    out = {int(d): best_obj[int(d)] for d in ids if int(d) in best_obj}
    row_count = int(sum(len(out.get(int(d), {}).get("ranklist", [])[:top]) for d in ids))
    return out, {
        "split": split,
        "source_path": str(best_path),
        "requested_query_count": len(ids),
        "loaded_query_count": len(out),
        "missing_query_count": int(best_missing),
        "missing_query_sample": [int(x) for x in sorted(wanted - set(out.keys()))[:20]],
        "top": int(top),
        "candidate_row_count": row_count,
    }


def rank_of_gt(corpus: Any, did: int, ranklist: Sequence[Tuple[int, float]]) -> int | None:
    gt = str(corpus.by_id[int(did)]["vid_name"])
    for i, (pos, _score) in enumerate(ranklist, start=1):
        if str(corpus.train_videos[int(pos)]) == gt:
            return i
    return None


def metrics_from_ranks(ranks: Sequence[int | None]) -> Dict[str, Any]:
    total = len(ranks)
    valid = [int(r) for r in ranks if isinstance(r, int)]

    def rec(k: int) -> float:
        return 100.0 * sum(1 for r in valid if r <= k) / max(1, total)

    return {
        "query_count": total,
        "missing_rank_count": total - len(valid),
        "VR_R@1": rec(1),
        "VR_R@5": rec(5),
        "VR_R@10": rec(10),
        "VR_R@100": rec(100),
        "GT_video_median_rank": float(np.median(valid)) if valid else None,
        "GT_video_mean_rank": float(np.mean(valid)) if valid else None,
        "wrong_video_top1_rate": 100.0 - rec(1),
    }


def first_stage_metrics(corpus: Any, first: Dict[int, Dict[str, Any]], ids: Sequence[int], top: int = 128) -> Dict[str, Any]:
    ranks = [rank_of_gt(corpus, int(d), first.get(int(d), {}).get("ranklist", [])[:top]) for d in ids]
    return metrics_from_ranks(ranks)


def duration_bucket(seconds: float) -> str:
    if seconds <= 5.0:
        return "short"
    if seconds <= 15.0:
        return "medium"
    return "long"


def qtype_id(qtype: str) -> int:
    return {"v": 0, "t": 1, "vt": 2}.get(str(qtype), 3)


def schema_columns() -> List[str]:
    return [
        "query_id", "video_id", "split", "candidate_video_rank", "first_stage_score", "first_stage_rank",
        "gt_video_id", "gt_start", "gt_end", "video_duration", "clip_length",
        "num_visual_clips", "num_subtitle_units", "visual_feature_key", "subtitle_feature_key", "query_feature_key",
        "has_visual_feature", "has_subtitle_feature", "has_query_feature", "feature_missing_mask",
        "time_alignment_status", "query_type", "duration_bucket", "is_gt_video_diagnostic",
        "schema_hash", "config_hash",
    ]


@dataclass
class FeatureArrays:
    query: np.ndarray
    sub_mean: np.ndarray
    sub_max: np.ndarray
    visual_mean: np.ndarray
    desc_to_qpos: Dict[int, int]
    video_to_pos: Dict[str, int]


def load_release_arrays(corpus: Any) -> Tuple[Dict[str, Path], FeatureArrays]:
    paths = build_feature_caches(corpus)
    raw = load_features(paths)
    return paths, FeatureArrays(
        query=raw["query"].astype(np.float32),
        sub_mean=raw["sub_mean"].astype(np.float32),
        sub_max=raw["sub_max"].astype(np.float32),
        visual_mean=raw["visual_mean"].astype(np.float32),
        desc_to_qpos={int(k): int(v) for k, v in raw["desc_to_qpos"].items()},
        video_to_pos={str(v): int(i) for i, v in enumerate(corpus.train_videos)},
    )


class TokenFeatureStore:
    def __init__(self, paths: C12Paths | None = None, cache_size: int = 2048) -> None:
        self.paths = paths or C12Paths()
        self.cache_size = cache_size
        self.query_env = lmdb.open(str(self.paths.query_lmdb), readonly=True, create=False, lock=False, readahead=False)
        self.subtitle_env = lmdb.open(str(self.paths.subtitle_lmdb), readonly=True, create=False, lock=False, readahead=False)
        self.visual_env = lmdb.open(str(self.paths.visual_lmdb), readonly=True, create=False, lock=False, readahead=False)
        self._q: Dict[int, np.ndarray] = {}
        self._s: Dict[str, np.ndarray] = {}
        self._v: Dict[str, np.ndarray] = {}

    def close(self) -> None:
        self.query_env.close()
        self.subtitle_env.close()
        self.visual_env.close()

    @staticmethod
    def _read_npz(txn: lmdb.Transaction, key: str) -> np.ndarray | None:
        raw = txn.get(key.encode())
        if raw is None:
            return None
        with io.BytesIO(bytes(raw)) as reader:
            return np.load(reader, allow_pickle=True)["features"].astype(np.float32)

    @staticmethod
    def _read_visual(txn: lmdb.Transaction, key: str) -> np.ndarray | None:
        raw = txn.get(key.encode())
        if raw is None:
            return None
        return msgpack_numpy.loads(bytes(raw), raw=False)["features"].astype(np.float32)

    def _trim(self, cache: Dict[Any, np.ndarray]) -> None:
        if len(cache) > self.cache_size:
            for k in list(cache.keys())[: max(1, self.cache_size // 8)]:
                cache.pop(k, None)

    def query_tokens(self, did: int) -> np.ndarray | None:
        did = int(did)
        if did not in self._q:
            with self.query_env.begin(buffers=True) as txn:
                arr = self._read_npz(txn, str(did))
            if arr is not None:
                self._q[did] = arr
                self._trim(self._q)
        return self._q.get(did)

    def subtitle_tokens(self, vid: str) -> np.ndarray | None:
        vid = str(vid)
        if vid not in self._s:
            with self.subtitle_env.begin(buffers=True) as txn:
                arr = self._read_npz(txn, vid)
            if arr is not None:
                self._s[vid] = arr
                self._trim(self._s)
        return self._s.get(vid)

    def visual_clips(self, vid: str) -> np.ndarray | None:
        vid = str(vid)
        if vid not in self._v:
            with self.visual_env.begin(buffers=True) as txn:
                arr = self._read_visual(txn, vid)
            if arr is not None:
                self._v[vid] = arr
                self._trim(self._v)
        return self._v.get(vid)


def build_canonical_table(
    mode: str,
    max_queries: int | None = None,
    max_candidates: int | None = None,
    force: bool = False,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    C26_CACHE.mkdir(parents=True, exist_ok=True)
    corpus = load_corpus()
    feature_paths, arrays = load_release_arrays(corpus)
    top = int(max_candidates or mode_cfg(mode)["top"])
    out_path = C26_CACHE / f"C26_CANONICAL_{mode}.local.parquet"
    if out_path.exists() and not force:
        df = pd.read_parquet(out_path)
        return df, load_json(C26_CACHE / f"C26_CANONICAL_{mode}.manifest.json", {})
    rows: List[Dict[str, Any]] = []
    split_audits: Dict[str, Any] = {}
    schema_hash = stable_hash(schema_columns())
    store = TokenFeatureStore(cache_size=256)
    query_probe: Dict[int, Tuple[bool, int]] = {}
    video_probe: Dict[str, Tuple[bool, int, bool, int]] = {}
    try:
        for split in SPLITS:
            ids = limited_ids(corpus, split, mode, max_queries=max_queries)
            first, audit = load_first_stage_split(corpus, split, ids, top)
            split_audits[split] = audit
            for did in ids:
                row = corpus.by_id[int(did)]
                gt = str(row["vid_name"])
                gt_start, gt_end = float(row["ts"][0]), float(row["ts"][1])
                if int(did) not in query_probe:
                    q_tokens = store.query_tokens(int(did))
                    query_probe[int(did)] = (q_tokens is not None and int(did) in arrays.desc_to_qpos, int(q_tokens.shape[0]) if q_tokens is not None else 0)
                has_query, _q_len = query_probe[int(did)]
                qtype = str(row.get("type", "unknown"))
                for rank, (pos, score) in enumerate(first.get(int(did), {}).get("ranklist", [])[:top], start=1):
                    vid = str(corpus.train_videos[int(pos)])
                    missing: List[str] = []
                    if vid not in video_probe:
                        v_clips = store.visual_clips(vid)
                        s_units = store.subtitle_tokens(vid)
                        video_probe[vid] = (
                            v_clips is not None,
                            int(v_clips.shape[0]) if v_clips is not None else 0,
                            s_units is not None,
                            int(s_units.shape[0]) if s_units is not None else 0,
                        )
                    has_v, num_v, has_s, num_s = video_probe[vid]
                    if not has_v:
                        missing.append("visual")
                    if not has_s:
                        missing.append("subtitle")
                    if not has_query:
                        missing.append("query")
                    dur = float(row.get("duration", 0.0) or max(gt_end + 1.0, 1.0))
                    rows.append({
                        "query_id": int(did),
                        "video_id": vid,
                        "split": split,
                        "candidate_video_rank": int(rank),
                        "first_stage_score": float(score),
                        "first_stage_rank": int(rank),
                        "gt_video_id": gt,
                        "gt_start": gt_start,
                        "gt_end": gt_end,
                        "video_duration": dur,
                        "clip_length": 1.5,
                        "num_visual_clips": num_v,
                        "num_subtitle_units": num_s,
                        "visual_feature_key": vid,
                        "subtitle_feature_key": vid,
                        "query_feature_key": str(int(did)),
                        "has_visual_feature": has_v,
                        "has_subtitle_feature": has_s,
                        "has_query_feature": has_query,
                        "feature_missing_mask": "|".join(missing),
                        "time_alignment_status": "clip_index_x_1.5s_release_grid" if not missing else "masked_missing_feature",
                        "query_type": qtype,
                        "duration_bucket": duration_bucket(gt_end - gt_start),
                        "is_gt_video_diagnostic": vid == gt,
                        "schema_hash": schema_hash,
                        "config_hash": stable_hash({"stage": "C26-1", "mode": mode, "top": top}),
                    })
    finally:
        store.close()
    df = pd.DataFrame(rows)
    df.to_parquet(out_path, index=False)
    manifest = {
        "path": str(out_path),
        "row_count": int(len(df)),
        "query_count": int(df["query_id"].nunique()) if len(df) else 0,
        "video_count": int(df["video_id"].nunique()) if len(df) else 0,
        "mode": mode,
        "top": top,
        "split_coverage": {s: {"rows": int((df["split"] == s).sum()), "queries": int(df[df["split"] == s]["query_id"].nunique())} for s in SPLITS},
        "feature_paths": {k: file_record(Path(v), sha=True) for k, v in feature_paths.items()},
        "first_stage_sources": split_audits,
        "schema_hash": schema_hash,
        "config_hash": stable_hash({"stage": "C26-1", "mode": mode, "top": top, "max_queries": max_queries}),
        "local_only": True,
    }
    write_json(C26_CACHE / f"C26_CANONICAL_{mode}.manifest.json", manifest)
    return df, manifest


def build_training_arrays(df: pd.DataFrame, mode: str, force: bool = False) -> Tuple[Path, Dict[str, Any]]:
    C26_CACHE.mkdir(parents=True, exist_ok=True)
    out = C26_CACHE / f"C26_TRAINING_ARRAYS_{mode}.local.npz"
    manifest_path = C26_CACHE / f"C26_TRAINING_ARRAYS_{mode}.manifest.json"
    if out.exists() and not force:
        return out, load_json(manifest_path, {})
    corpus = load_corpus()
    _paths, arrays = load_release_arrays(corpus)
    q_idx, v_idx, y, qtype, first, rank, query_ids, video_ids, split_ids = [], [], [], [], [], [], [], [], []
    split_map = {"train_fit": 0, "calib_select": 1, "calib_holdout": 2}
    for rec in df.itertuples(index=False):
        did = int(rec.query_id)
        vid = str(rec.video_id)
        if did not in arrays.desc_to_qpos or vid not in arrays.video_to_pos:
            continue
        q_idx.append(arrays.desc_to_qpos[did])
        v_idx.append(arrays.video_to_pos[vid])
        y.append(1.0 if bool(rec.is_gt_video_diagnostic) else 0.0)
        qtype.append(qtype_id(str(rec.query_type)))
        first.append(float(rec.first_stage_score))
        rank.append(int(rec.candidate_video_rank))
        query_ids.append(did)
        video_ids.append(vid)
        split_ids.append(split_map[str(rec.split)])
    np.savez(
        out,
        q_idx=np.asarray(q_idx, dtype=np.int64),
        v_idx=np.asarray(v_idx, dtype=np.int64),
        y=np.asarray(y, dtype=np.float32),
        qtype=np.asarray(qtype, dtype=np.int64),
        first_stage=np.asarray(first, dtype=np.float32),
        rank=np.asarray(rank, dtype=np.int64),
        query_ids=np.asarray(query_ids, dtype=np.int64),
        video_ids=np.asarray(video_ids, dtype=object),
        split_ids=np.asarray(split_ids, dtype=np.int64),
        query=arrays.query.astype(np.float32),
        visual_mean=arrays.visual_mean.astype(np.float32),
        sub_mean=arrays.sub_mean.astype(np.float32),
        sub_max=arrays.sub_max.astype(np.float32),
    )
    manifest = {
        "path": str(out),
        "row_count": int(len(q_idx)),
        "query_count": int(len(set(query_ids))),
        "positive_rows": int(sum(1 for z in y if z > 0.5)),
        "splits": {"train_fit": int(sum(1 for s in split_ids if s == 0)), "calib_select": int(sum(1 for s in split_ids if s == 1)), "calib_holdout": int(sum(1 for s in split_ids if s == 2))},
        "feature_dims": {"query": 768, "subtitle": 768, "visual": 4352},
        "local_only": True,
    }
    write_json(manifest_path, manifest)
    return out, manifest


def sample_dataframe(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    n = min(int(mode_cfg(mode)["sample_rows"]), len(df))
    return df.head(n).copy()


def contamination_files() -> Dict[str, Any]:
    blockers = []
    warnings = []
    for p in ROOT.rglob("*official*"):
        if ".git" in p.parts:
            continue
        name = p.name.lower()
        if p.is_file() and ("prediction" in name or name.endswith("_raw.json") or "nms" in name):
            blockers.append(str(p))
        elif p.is_file() and p.suffix == ".py":
            warnings.append(str(p))
    return {"blocking_official_prediction_or_raw_nms_files": blockers[:100], "root_official_script_warnings": warnings[:100]}
