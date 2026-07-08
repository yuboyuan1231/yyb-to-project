from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import lmdb
import msgpack_numpy
import numpy as np

from blueprint_e2e_v2.data.feature_registry import FeaturePaths, TMP_ROOT
from blueprint_e2e_v2.utils.hashing import stable_hash


def l2(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), eps)


class VideoBank:
    def __init__(self, paths: FeaturePaths | None = None) -> None:
        self.paths = paths or FeaturePaths()
        self.env = lmdb.open(str(self.paths.visual_lmdb), readonly=True, create=False, lock=False, readahead=False, max_readers=2048)
        self.cache_enabled = False
        self._seq_cache: dict[str, np.ndarray] = {}
        meta = json.loads(self.paths.video_meta.read_text(encoding="utf-8"))
        self.video_ids = list(meta["train"].keys())
        self.video_to_idx = {v: i for i, v in enumerate(self.video_ids)}
        self.idx_to_video = {i: v for v, i in self.video_to_idx.items()}
        self.durations = self._durations_from_train()

    def _durations_from_train(self) -> dict[str, float]:
        out: dict[str, float] = {}
        with self.paths.train_jsonl.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                out.setdefault(str(row["vid_name"]), float(row["duration"]))
        return out

    def close(self) -> None:
        self.env.close()

    def sequence(self, video_id: str) -> np.ndarray:
        if self.cache_enabled and str(video_id) in self._seq_cache:
            return self._seq_cache[str(video_id)]
        with self.env.begin(buffers=True) as txn:
            raw = txn.get(str(video_id).encode())
            if raw is None:
                raise KeyError(f"visual feature missing: {video_id}")
            arr = msgpack_numpy.loads(bytes(raw), raw=False)["features"].astype(np.float32)
        if self.cache_enabled:
            self._seq_cache[str(video_id)] = arr
        return arr

    def enable_sequence_cache(self) -> None:
        self.cache_enabled = True

    def clear_sequence_cache(self) -> None:
        self._seq_cache.clear()

    def cache_audit(self) -> dict[str, Any]:
        return {"cache_enabled": self.cache_enabled, "cached_sequences": len(self._seq_cache)}

    def build_or_load_pooled(self, max_videos: int | None = None, force: bool = False) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        TMP_ROOT.mkdir(parents=True, exist_ok=True)
        suffix = "all" if not max_videos else f"top{int(max_videos)}"
        path = TMP_ROOT / f"C28C_VIDEO_VISUAL_BANK_{suffix}.npz"
        manifest_path = TMP_ROOT / f"C28C_VIDEO_VISUAL_BANK_{suffix}.manifest.json"
        if path.exists() and manifest_path.exists() and not force:
            arr = np.load(path, allow_pickle=True)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            return {"video_ids": arr["video_ids"], "visual_mean": arr["visual_mean"], "visual_max": arr["visual_max"]}, manifest
        vids = self.video_ids[: int(max_videos)] if max_videos and max_videos > 0 else self.video_ids
        means, maxes = [], []
        for vid in vids:
            seq = self.sequence(vid).astype(np.float32)
            means.append(l2(seq.mean(axis=0, keepdims=True))[0])
            maxes.append(l2(seq.max(axis=0, keepdims=True))[0])
        out = {
            "video_ids": np.array(vids, dtype=object),
            "visual_mean": np.stack(means).astype(np.float32),
            "visual_max": np.stack(maxes).astype(np.float32),
        }
        np.savez(path, **out)
        manifest = {
            "path": str(path),
            "video_count": len(vids),
            "visual_dim": int(out["visual_mean"].shape[1]),
            "schema_hash": stable_hash({"video_count": len(vids), "visual_dim": int(out["visual_mean"].shape[1])}),
            "zero_fill": False,
        }
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return out, manifest

    @staticmethod
    def _fit_seq(seq: np.ndarray, target_len: int = 64) -> np.ndarray:
        if seq.shape[0] == target_len:
            return seq.astype(np.float32, copy=False)
        if seq.shape[0] > target_len:
            return seq[:target_len].astype(np.float32, copy=False)
        out = np.zeros((target_len, seq.shape[1]), dtype=np.float32)
        out[: seq.shape[0]] = seq.astype(np.float32, copy=False)
        return out

    def build_or_load_sequence_bank(
        self,
        video_ids: list[str],
        target_len: int = 64,
        max_videos: int | None = None,
        force: bool = False,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        TMP_ROOT.mkdir(parents=True, exist_ok=True)
        vids = video_ids[: int(max_videos)] if max_videos and max_videos > 0 else video_ids
        suffix = "all" if not max_videos else f"top{int(max_videos)}"
        path = TMP_ROOT / f"C28C_VIDEO_VISUAL_SEQ_{suffix}_T{int(target_len)}.npy"
        manifest_path = TMP_ROOT / f"C28C_VIDEO_VISUAL_SEQ_{suffix}_T{int(target_len)}.manifest.json"
        if path.exists() and manifest_path.exists() and not force:
            arr = np.load(path)
            self.clear_sequence_cache()
            return arr, json.loads(manifest_path.read_text(encoding="utf-8"))
        first = self._fit_seq(self.sequence(vids[0]), target_len)
        old_cache_enabled = self.cache_enabled
        self.cache_enabled = False
        arr = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=(len(vids), int(target_len), first.shape[1]))
        try:
            arr[0] = first
            for i, vid in enumerate(vids[1:], start=1):
                arr[i] = self._fit_seq(self.sequence(vid), target_len)
                if (i + 1) % 1000 == 0:
                    print(f"C28C visual sequence bank: loaded {i + 1}/{len(vids)}", flush=True)
            arr.flush()
        finally:
            self.cache_enabled = old_cache_enabled
            self.clear_sequence_cache()
        manifest = {
            "path": str(path),
            "video_count": len(vids),
            "target_len": int(target_len),
            "visual_dim": int(first.shape[1]),
            "schema_hash": stable_hash({"video_count": len(vids), "target_len": int(target_len), "visual_dim": int(first.shape[1])}),
            "zero_fill": False,
            "source": "release_lmdb_preloaded_sequence_bank",
        }
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return np.load(path), manifest

    def audit_sample(self, video_ids: list[str]) -> dict[str, Any]:
        shapes = []
        for vid in video_ids[:8]:
            shapes.append(list(self.sequence(vid).shape))
        return {
            "train_video_count": len(self.video_ids),
            "sample_count": len(shapes),
            "sample_shapes": shapes,
            "zero_fill": False,
        }
