#!/usr/bin/env python3
"""C12 native retriever scaffold.

No official files are read here. C7-B6 artifacts are optional teacher or
hard-negative sources, never the final candidate pool.
"""

from __future__ import annotations

import hashlib
import io
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import lmdb
import msgpack
import msgpack_numpy
import numpy as np


DATA_ROOT = Path("/home/a/yybwork/data/yyb/tvr_feature_release")
REPO_ROOT = Path("/home/a/yybwork/yybcounter/CONQUER-RLEM-c2c3")


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _l2(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), eps)


def _video_group(vid_name: str) -> str:
    m = re.match(r"(.+?_s\d+e[\de\-]+)", vid_name)
    if m:
        return m.group(1)
    return vid_name.rsplit("_clip_", 1)[0]


@dataclass(frozen=True)
class C12Paths:
    train_jsonl: Path = DATA_ROOT / "data/tvr_train_select100_release.jsonl"
    video_meta: Path = DATA_ROOT / "data/tvr_video2dur_idx.json"
    first_stage_rank_lmdb: Path = DATA_ROOT / "data/train_select100_top2000"
    query_lmdb: Path = DATA_ROOT / "sub_query_feature/tvr_query_pretrained_w_sub_query"
    subtitle_lmdb: Path = DATA_ROOT / "sub_query_feature/tvr_sub_pretrained_w_sub_query_max_cl-1.5"
    visual_lmdb: Path = DATA_ROOT / "video_feature/resnet_slowfast_1.5"
    split_dir: Path = REPO_ROOT / "c12_1_schema_and_split/splits"
    b6_train_teacher_subset: Path = REPO_ROOT / "results/c8_8_exact_b6_feature_reexport/C8_8_EXACT_B6_TRAIN_NMS.json"


class C12SplitLoader:
    """Loads C12 train-only splits.

    The pseudo-official split is exposed with an explicit name so callers can
    lock it until branch-level one-shot validation.
    """

    def __init__(self, paths: C12Paths = C12Paths()) -> None:
        self.paths = paths

    def load_ids(self, split: str) -> List[int]:
        allowed = {"train_fit", "calib_select", "calib_holdout", "pseudo_official_holdout"}
        if split not in allowed:
            raise ValueError(f"Unknown C12 split: {split}")
        path = self.paths.split_dir / f"{split}_desc_ids.txt"
        return [int(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]

    def load_records(self, split: str) -> List[Dict[str, Any]]:
        ids = set(self.load_ids(split))
        return [r for r in _load_jsonl(self.paths.train_jsonl) if int(r["desc_id"]) in ids]


class TVRFeatureStore:
    """Read-only TVR feature store for native C12 retriever experiments."""

    def __init__(self, paths: C12Paths = C12Paths()) -> None:
        self.paths = paths
        self.query_env = lmdb.open(str(paths.query_lmdb), readonly=True, create=False, lock=False, readahead=False)
        self.sub_env = lmdb.open(str(paths.subtitle_lmdb), readonly=True, create=False, lock=False, readahead=False)
        self.visual_env = lmdb.open(str(paths.visual_lmdb), readonly=True, create=False, lock=False, readahead=False)

    def close(self) -> None:
        self.query_env.close()
        self.sub_env.close()
        self.visual_env.close()

    @staticmethod
    def _read_npz(txn: lmdb.Transaction, key: str) -> np.ndarray:
        raw = txn.get(key.encode())
        if raw is None:
            raise KeyError(key)
        with io.BytesIO(bytes(raw)) as reader:
            return np.load(reader, allow_pickle=True)["features"].astype(np.float32)

    @staticmethod
    def _read_msgpack_features(txn: lmdb.Transaction, key: str) -> np.ndarray:
        raw = txn.get(key.encode())
        if raw is None:
            raise KeyError(key)
        return msgpack_numpy.loads(bytes(raw), raw=False)["features"].astype(np.float32)

    def query_feature(self, desc_id: int) -> np.ndarray:
        with self.query_env.begin(buffers=True) as txn:
            return self._read_npz(txn, str(desc_id))

    def subtitle_feature(self, vid_name: str) -> np.ndarray:
        with self.sub_env.begin(buffers=True) as txn:
            return self._read_npz(txn, vid_name)

    def visual_feature(self, vid_name: str) -> np.ndarray:
        with self.visual_env.begin(buffers=True) as txn:
            return self._read_msgpack_features(txn, vid_name)

    def pooled_query(self, desc_id: int) -> np.ndarray:
        return _l2(self.query_feature(desc_id).mean(axis=0, keepdims=True))[0]

    def pooled_subtitle(self, vid_name: str) -> np.ndarray:
        return _l2(self.subtitle_feature(vid_name).mean(axis=0, keepdims=True))[0]


class QueryTypeAwareRouter:
    """Chooses feature streams without changing schema semantics."""

    def route(self, query_type: str) -> Dict[str, float]:
        if query_type == "t":
            return {"subtitle": 1.0, "visual": 0.0}
        if query_type == "vt":
            return {"subtitle": 0.7, "visual": 0.3}
        if query_type == "v":
            return {"subtitle": 0.3, "visual": 0.7}
        return {"subtitle": 0.5, "visual": 0.5}


class HardNegativeSampler:
    """Train-only hard negative sampler for C12 native retriever training."""

    def __init__(self, paths: C12Paths = C12Paths(), seed: int = 1203) -> None:
        self.paths = paths
        self.rng = random.Random(seed)
        meta = json.loads(paths.video_meta.read_text(encoding="utf-8"))
        self.train_videos = list(meta["train"].keys())
        self.idx2video = {int(v[1]): k for k, v in meta["train"].items()}
        self.video_groups: Dict[str, List[str]] = {}
        for vid in self.train_videos:
            self.video_groups.setdefault(_video_group(vid), []).append(vid)

    def first_stage_wrong_top(self, desc_id: int, gt_vid: str, k: int = 20) -> List[str]:
        env = lmdb.open(str(self.paths.first_stage_rank_lmdb), readonly=True, create=False, lock=False, readahead=False)
        out: List[str] = []
        with env.begin(buffers=True) as txn:
            raw = txn.get(str(desc_id).encode())
            if raw is not None:
                ranklist = msgpack.loads(bytes(raw), raw=False)
                for item in ranklist:
                    vid = self.idx2video.get(int(item[0]))
                    if vid is not None and vid != gt_vid:
                        out.append(vid)
                    if len(out) >= k:
                        break
        env.close()
        return out

    def same_group(self, gt_vid: str, k: int = 10) -> List[str]:
        pool = [v for v in self.video_groups.get(_video_group(gt_vid), []) if v != gt_vid]
        self.rng.shuffle(pool)
        return pool[:k]

    def random_negatives(self, gt_vid: str, k: int = 20) -> List[str]:
        pool = [v for v in self.train_videos if v != gt_vid]
        return self.rng.sample(pool, min(k, len(pool)))

    def sample(self, desc_id: int, gt_vid: str, k: int = 64) -> List[str]:
        ordered = []
        for vid in (
            self.first_stage_wrong_top(desc_id, gt_vid, k=k // 2)
            + self.same_group(gt_vid, k=max(4, k // 8))
            + self.random_negatives(gt_vid, k=k)
        ):
            if vid != gt_vid and vid not in ordered:
                ordered.append(vid)
            if len(ordered) >= k:
                break
        return ordered


class NativeRetrieverBatchBuilder:
    """Builds native C12 retriever batches from corpus videos, not fixed pools."""

    def __init__(
        self,
        split_loader: C12SplitLoader | None = None,
        feature_store: TVRFeatureStore | None = None,
        neg_sampler: HardNegativeSampler | None = None,
    ) -> None:
        self.split_loader = split_loader or C12SplitLoader()
        self.feature_store = feature_store or TVRFeatureStore()
        self.neg_sampler = neg_sampler or HardNegativeSampler()

    def build_train_item(self, record: Dict[str, Any], negative_k: int = 64) -> Dict[str, Any]:
        desc_id = int(record["desc_id"])
        gt_vid = record["vid_name"]
        negs = self.neg_sampler.sample(desc_id, gt_vid, negative_k)
        return {
            "query_id": desc_id,
            "query_type": record.get("type", "unknown"),
            "query_feature": self.feature_store.pooled_query(desc_id),
            "positive_video": gt_vid,
            "negative_videos": negs,
            "candidate_videos_generated_by": "C12 native hard-negative sampler over train corpus",
            "uses_c7_b6_fixed_candidate_pool": False,
        }

    def build_split_preview(self, split: str, n: int = 8) -> List[Dict[str, Any]]:
        return [self.build_train_item(r, negative_k=16) for r in self.split_loader.load_records(split)[:n]]


def scaffold_manifest() -> Dict[str, Any]:
    paths = C12Paths()
    return {
        "official_loader_present": False,
        "uses_c7_b6_fixed_candidate_pool": False,
        "candidate_videos_generated_from_corpus": True,
        "query_encoder_input": str(paths.query_lmdb),
        "subtitle_feature_loader": str(paths.subtitle_lmdb),
        "visual_feature_loader": str(paths.visual_lmdb),
        "split_loader": str(paths.split_dir),
        "hard_negative_sources": [
            "CONQUER/HERO train first-stage wrong top videos",
            "same-show / same-video-group negatives",
            "random train corpus negatives",
            "optional C7-B6 train teacher subset for diagnostics only",
        ],
        "schema_hash": hashlib.sha256(Path(REPO_ROOT / "c12_1_schema_and_split/C12_1_SCHEMA_MANIFEST.json").read_bytes()).hexdigest()
        if Path(REPO_ROOT / "c12_1_schema_and_split/C12_1_SCHEMA_MANIFEST.json").exists()
        else None,
    }
