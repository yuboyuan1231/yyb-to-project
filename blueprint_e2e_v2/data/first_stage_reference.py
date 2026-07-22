from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import lmdb
import msgpack

from blueprint_e2e_v2.data.feature_registry import FeaturePaths
from blueprint_e2e_v2.data.feature_registry import TMP_ROOT
from blueprint_e2e_v2.utils.hashing import stable_hash


@dataclass(frozen=True)
class TeacherRankItem:
    video_index: int
    rank: int
    score: float | None
    raw_item_len: int


class FirstStageReference:
    """CONQUER first-stage rank source for training warm-start only."""

    def __init__(self, paths: FeaturePaths | None = None) -> None:
        self.paths = paths or FeaturePaths()
        meta = json.loads(self.paths.video_meta.read_text(encoding="utf-8"))
        self.idx2video = {int(v[1]): k for k, v in meta["train"].items()}
        self.env = lmdb.open(str(self.paths.first_stage_rank_lmdb), readonly=True, create=False, lock=False, readahead=False, max_readers=2048)

    def close(self) -> None:
        self.env.close()

    def top_indices(self, query_id: int, video_to_idx: dict[str, int], k: int = 64) -> list[int]:
        out: list[int] = []
        with self.env.begin(buffers=True) as txn:
            raw = txn.get(str(int(query_id)).encode())
            if raw is None:
                return out
            ranklist = msgpack.loads(bytes(raw), raw=False)
            for item in ranklist:
                vid = self.idx2video.get(int(item[0]))
                if vid is None or vid not in video_to_idx:
                    continue
                idx = int(video_to_idx[vid])
                if idx not in out:
                    out.append(idx)
                if len(out) >= k:
                    break
        return out

    @staticmethod
    def _decode_score(item: Any) -> tuple[int, float | None, int]:
        if isinstance(item, (list, tuple)):
            raw_len = len(item)
            video_ref = int(item[0])
            score = None
            if len(item) > 1:
                try:
                    score = float(item[1])
                except (TypeError, ValueError):
                    score = None
            return video_ref, score, raw_len
        return int(item), None, 1

    def top_items(self, query_id: int, video_to_idx: dict[str, int], k: int = 64) -> list[TeacherRankItem]:
        out: list[TeacherRankItem] = []
        seen: set[int] = set()
        with self.env.begin(buffers=True) as txn:
            raw = txn.get(str(int(query_id)).encode())
            if raw is None:
                return out
            ranklist = msgpack.loads(bytes(raw), raw=False)
            for raw_rank, item in enumerate(ranklist, start=1):
                video_ref, score, raw_item_len = self._decode_score(item)
                vid = self.idx2video.get(video_ref)
                if vid is None or vid not in video_to_idx:
                    continue
                idx = int(video_to_idx[vid])
                if idx in seen:
                    continue
                seen.add(idx)
                out.append(TeacherRankItem(video_index=idx, rank=len(out) + 1, score=score, raw_item_len=raw_item_len))
                if len(out) >= k:
                    break
        return out

    def bulk_top_indices(self, query_ids: list[int], video_to_idx: dict[str, int], k: int = 64) -> dict[int, list[int]]:
        TMP_ROOT.mkdir(parents=True, exist_ok=True)
        cache_key = stable_hash({"query_ids": query_ids, "video_count": len(video_to_idx), "k": k})
        cache_path = TMP_ROOT / f"C28C_FIRST_STAGE_TEACHER_TOP{int(k)}_{cache_key}.pkl"
        if cache_path.exists():
            with cache_path.open("rb") as f:
                return pickle.load(f)
        out: dict[int, list[int]] = {}
        with self.env.begin(buffers=True) as txn:
            for n, query_id in enumerate(query_ids, start=1):
                vals: list[int] = []
                raw = txn.get(str(int(query_id)).encode())
                if raw is not None:
                    ranklist = msgpack.loads(bytes(raw), raw=False)
                    for item in ranklist:
                        vid = self.idx2video.get(int(item[0]))
                        if vid is None or vid not in video_to_idx:
                            continue
                        idx = int(video_to_idx[vid])
                        if idx not in vals:
                            vals.append(idx)
                        if len(vals) >= k:
                            break
                out[int(query_id)] = vals
                if n % 10000 == 0:
                    print(f"C28C teacher warm-start cache: loaded {n}/{len(query_ids)} queries", flush=True)
        with cache_path.open("wb") as f:
            pickle.dump(out, f, protocol=pickle.HIGHEST_PROTOCOL)
        return out

    def bulk_top_items(self, query_ids: list[int], video_to_idx: dict[str, int], k: int = 64) -> dict[int, list[TeacherRankItem]]:
        TMP_ROOT.mkdir(parents=True, exist_ok=True)
        cache_key = stable_hash({"query_ids": query_ids, "video_count": len(video_to_idx), "k": k, "schema": "items_v1"})
        cache_path = TMP_ROOT / f"C28E_FIRST_STAGE_TEACHER_ITEMS_TOP{int(k)}_{cache_key}.pkl"
        if cache_path.exists():
            with cache_path.open("rb") as f:
                return pickle.load(f)
        out = {int(query_id): self.top_items(int(query_id), video_to_idx, k=k) for query_id in query_ids}
        with cache_path.open("wb") as f:
            pickle.dump(out, f, protocol=pickle.HIGHEST_PROTOCOL)
        return out

    def score_schema_audit(self, query_ids: list[int], sample_k: int = 8) -> dict[str, Any]:
        item_lengths: dict[str, int] = {}
        score_count = 0
        row_count = 0
        sample_rows: list[dict[str, Any]] = []
        with self.env.begin(buffers=True) as txn:
            for query_id in query_ids[: int(sample_k)]:
                raw = txn.get(str(int(query_id)).encode())
                if raw is None:
                    sample_rows.append({"query_id": int(query_id), "present": False})
                    continue
                ranklist = msgpack.loads(bytes(raw), raw=False)
                first = ranklist[0] if ranklist else None
                first_ref = None
                first_score = None
                first_len = 0
                if first is not None:
                    first_ref, first_score, first_len = self._decode_score(first)
                    item_lengths[str(first_len)] = item_lengths.get(str(first_len), 0) + 1
                    score_count += int(first_score is not None)
                row_count += 1
                sample_rows.append({
                    "query_id": int(query_id),
                    "present": True,
                    "ranklist_len": len(ranklist),
                    "first_video_ref": first_ref,
                    "first_score": first_score,
                    "first_item_len": first_len,
                })
        return {
            "sample_query_count": len(query_ids[: int(sample_k)]),
            "rank_rows_present": row_count,
            "item_length_histogram": item_lengths,
            "score_field_available_rate": 100.0 * score_count / max(1, row_count),
            "sample_rows": sample_rows,
        }

    def audit(self) -> dict[str, Any]:
        return {
            "path": str(self.paths.first_stage_rank_lmdb),
            "available": (self.paths.first_stage_rank_lmdb / "data.mdb").exists(),
            "use_policy": "training warm-start / hard-negative reference only; not final hard gate",
        }
