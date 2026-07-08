from __future__ import annotations

import io
from typing import Any

import lmdb
import numpy as np

from blueprint_e2e_v2.data.feature_registry import FeaturePaths


class QueryBank:
    def __init__(self, paths: FeaturePaths | None = None) -> None:
        self.paths = paths or FeaturePaths()
        self.env = lmdb.open(str(self.paths.query_lmdb), readonly=True, create=False, lock=False, readahead=False, max_readers=2048)
        self._token_cache: dict[int, np.ndarray] = {}

    def close(self) -> None:
        self.env.close()

    def tokens(self, query_id: int) -> np.ndarray:
        if int(query_id) in self._token_cache:
            return self._token_cache[int(query_id)]
        with self.env.begin(buffers=True) as txn:
            raw = txn.get(str(int(query_id)).encode())
            if raw is None:
                raise KeyError(f"query feature missing: {query_id}")
            arr = np.load(io.BytesIO(bytes(raw)), allow_pickle=True)["features"].astype(np.float32)
        self._token_cache[int(query_id)] = arr
        return arr

    def bulk_tokens(self, query_ids: list[int]) -> dict[int, np.ndarray]:
        missing = [int(q) for q in query_ids if int(q) not in self._token_cache]
        with self.env.begin(buffers=True) as txn:
            for n, query_id in enumerate(missing, start=1):
                raw = txn.get(str(int(query_id)).encode())
                if raw is None:
                    raise KeyError(f"query feature missing: {query_id}")
                self._token_cache[int(query_id)] = np.load(io.BytesIO(bytes(raw)), allow_pickle=True)["features"].astype(np.float32)
                if n % 10000 == 0:
                    print(f"C28C query cache: loaded {n}/{len(missing)} missing query features", flush=True)
        return {int(q): self._token_cache[int(q)] for q in query_ids}

    def audit_sample(self, query_ids: list[int]) -> dict[str, Any]:
        shapes = []
        for qid in query_ids[:8]:
            shapes.append(list(self.tokens(int(qid)).shape))
        return {"sample_count": len(shapes), "sample_shapes": shapes, "zero_fill": False}
