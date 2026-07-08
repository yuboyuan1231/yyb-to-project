from __future__ import annotations

import io
import json
from typing import Any

import lmdb
import numpy as np

from blueprint_e2e_v2.data.feature_registry import FeaturePaths, TMP_ROOT
from blueprint_e2e_v2.utils.hashing import stable_hash


class SubtitleBank:
    SEQ_CACHE_SCHEMA = "resample_v2"

    def __init__(self, paths: FeaturePaths | None = None) -> None:
        self.paths = paths or FeaturePaths()
        self.env = lmdb.open(str(self.paths.subtitle_lmdb), readonly=True, create=False, lock=False, readahead=False, max_readers=2048)
        self.cache_enabled = False
        self._seq_cache: dict[str, np.ndarray] = {}

    def close(self) -> None:
        self.env.close()

    def sequence(self, video_id: str) -> np.ndarray:
        if self.cache_enabled and str(video_id) in self._seq_cache:
            return self._seq_cache[str(video_id)]
        with self.env.begin(buffers=True) as txn:
            raw = txn.get(str(video_id).encode())
            if raw is None:
                raise KeyError(f"subtitle feature missing: {video_id}")
            arr = np.load(io.BytesIO(bytes(raw)), allow_pickle=True)["features"].astype(np.float32)
        if self.cache_enabled:
            self._seq_cache[str(video_id)] = arr
        return arr

    def enable_sequence_cache(self) -> None:
        self.cache_enabled = True

    def clear_sequence_cache(self) -> None:
        self._seq_cache.clear()

    def cache_audit(self) -> dict[str, Any]:
        return {"cache_enabled": self.cache_enabled, "cached_sequences": len(self._seq_cache)}

    def audit_sample(self, video_ids: list[str]) -> dict[str, Any]:
        shapes = []
        for vid in video_ids[:8]:
            shapes.append(list(self.sequence(vid).shape))
        return {"sample_count": len(shapes), "sample_shapes": shapes, "zero_fill": False}

    def build_or_load_pooled(self, video_ids: list[str], max_videos: int | None = None, force: bool = False) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        TMP_ROOT.mkdir(parents=True, exist_ok=True)
        suffix = "all" if not max_videos else f"top{int(max_videos)}"
        path = TMP_ROOT / f"C28C_VIDEO_SUBTITLE_BANK_{suffix}.npz"
        manifest_path = TMP_ROOT / f"C28C_VIDEO_SUBTITLE_BANK_{suffix}.manifest.json"
        if path.exists() and manifest_path.exists() and not force:
            arr = np.load(path, allow_pickle=True)
            return {"video_ids": arr["video_ids"], "subtitle_mean": arr["subtitle_mean"], "subtitle_max": arr["subtitle_max"]}, json.loads(manifest_path.read_text(encoding="utf-8"))
        vids = video_ids[: int(max_videos)] if max_videos and max_videos > 0 else video_ids
        means, maxes = [], []
        for vid in vids:
            seq = self.sequence(vid).astype(np.float32)
            means.append(seq.mean(axis=0))
            maxes.append(seq.max(axis=0))
        out = {"video_ids": np.array(vids, dtype=object), "subtitle_mean": np.stack(means).astype(np.float32), "subtitle_max": np.stack(maxes).astype(np.float32)}
        np.savez(path, **out)
        manifest = {
            "path": str(path),
            "video_count": len(vids),
            "subtitle_dim": int(out["subtitle_mean"].shape[1]),
            "schema_hash": stable_hash({"video_count": len(vids), "subtitle_dim": int(out["subtitle_mean"].shape[1])}),
            "zero_fill": False,
        }
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return out, manifest

    @staticmethod
    def _fit_seq(seq: np.ndarray, target_len: int = 64) -> np.ndarray:
        if seq.shape[0] == target_len:
            return seq.astype(np.float32, copy=False)
        if seq.shape[0] > target_len:
            out = np.zeros((target_len, seq.shape[1]), dtype=np.float32)
            edges = np.linspace(0, seq.shape[0], int(target_len) + 1)
            for i in range(int(target_len)):
                st = int(np.floor(edges[i]))
                ed = int(np.ceil(edges[i + 1]))
                ed = max(ed, st + 1)
                out[i] = seq[st: min(ed, seq.shape[0])].mean(axis=0)
            return out.astype(np.float32, copy=False)
        out = np.zeros((target_len, seq.shape[1]), dtype=np.float32)
        out[: seq.shape[0]] = seq.astype(np.float32, copy=False)
        return out

    @staticmethod
    def _fit_mask(seq_len: int, target_len: int = 64) -> np.ndarray:
        valid = int(target_len) if int(seq_len) > int(target_len) else min(int(seq_len), int(target_len))
        out = np.zeros((int(target_len),), dtype=np.bool_)
        out[:valid] = True
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
        path = TMP_ROOT / f"C28C_VIDEO_SUBTITLE_SEQ_{suffix}_T{int(target_len)}_{self.SEQ_CACHE_SCHEMA}.npy"
        manifest_path = TMP_ROOT / f"C28C_VIDEO_SUBTITLE_SEQ_{suffix}_T{int(target_len)}_{self.SEQ_CACHE_SCHEMA}.manifest.json"
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
                    print(f"C28C subtitle sequence bank: loaded {i + 1}/{len(vids)}", flush=True)
            arr.flush()
        finally:
            self.cache_enabled = old_cache_enabled
            self.clear_sequence_cache()
        manifest = {
            "path": str(path),
            "video_count": len(vids),
            "target_len": int(target_len),
            "subtitle_dim": int(first.shape[1]),
            "schema_hash": stable_hash({"video_count": len(vids), "target_len": int(target_len), "subtitle_dim": int(first.shape[1]), "sequence_cache_schema": self.SEQ_CACHE_SCHEMA}),
            "zero_fill": False,
            "source": "release_lmdb_preloaded_sequence_bank",
            "sequence_cache_schema": self.SEQ_CACHE_SCHEMA,
            "long_sequence_policy": "mean-bin-resample-to-target-len",
        }
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return np.load(path), manifest

    def build_or_load_sequence_mask(
        self,
        video_ids: list[str],
        target_len: int = 64,
        max_videos: int | None = None,
        force: bool = False,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        TMP_ROOT.mkdir(parents=True, exist_ok=True)
        vids = video_ids[: int(max_videos)] if max_videos and max_videos > 0 else video_ids
        suffix = "all" if not max_videos else f"top{int(max_videos)}"
        path = TMP_ROOT / f"C28C_VIDEO_SUBTITLE_SEQ_MASK_{suffix}_T{int(target_len)}_{self.SEQ_CACHE_SCHEMA}.npy"
        manifest_path = TMP_ROOT / f"C28C_VIDEO_SUBTITLE_SEQ_MASK_{suffix}_T{int(target_len)}_{self.SEQ_CACHE_SCHEMA}.manifest.json"
        if path.exists() and manifest_path.exists() and not force:
            return np.load(path), json.loads(manifest_path.read_text(encoding="utf-8"))
        arr = np.lib.format.open_memmap(path, mode="w+", dtype=np.bool_, shape=(len(vids), int(target_len)))
        old_cache_enabled = self.cache_enabled
        self.cache_enabled = False
        try:
            for i, vid in enumerate(vids):
                arr[i] = self._fit_mask(self.sequence(vid).shape[0], target_len)
                if (i + 1) % 1000 == 0:
                    print(f"C28C subtitle sequence mask: loaded {i + 1}/{len(vids)}", flush=True)
            arr.flush()
        finally:
            self.cache_enabled = old_cache_enabled
            self.clear_sequence_cache()
        manifest = {
            "path": str(path),
            "video_count": len(vids),
            "target_len": int(target_len),
            "schema_hash": stable_hash({"video_count": len(vids), "target_len": int(target_len), "kind": "subtitle_sequence_mask", "sequence_cache_schema": self.SEQ_CACHE_SCHEMA}),
            "valid_clip_min": int(arr.sum(axis=1).min()) if len(vids) else 0,
            "valid_clip_max": int(arr.sum(axis=1).max()) if len(vids) else 0,
            "source": "release_lmdb_sequence_lengths",
            "sequence_cache_schema": self.SEQ_CACHE_SCHEMA,
            "long_sequence_policy": "mean-bin-resampled-long-videos-mark-all-target-clips-valid",
        }
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return np.load(path), manifest
