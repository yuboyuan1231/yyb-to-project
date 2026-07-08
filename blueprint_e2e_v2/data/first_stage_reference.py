from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import lmdb
import msgpack

from blueprint_e2e_v2.data.feature_registry import FeaturePaths
from blueprint_e2e_v2.data.feature_registry import TMP_ROOT
from blueprint_e2e_v2.utils.hashing import stable_hash


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

    def audit(self) -> dict[str, Any]:
        return {
            "path": str(self.paths.first_stage_rank_lmdb),
            "available": (self.paths.first_stage_rank_lmdb / "data.mdb").exists(),
            "use_policy": "training warm-start / hard-negative reference only; not final hard gate",
        }
