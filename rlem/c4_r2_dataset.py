#!/usr/bin/env python
"""Dataset and standardization helpers for C4-r2-cal."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

EPS = 1e-6


@dataclass
class ArrayStandardizer:
    mean: np.ndarray
    std: np.ndarray
    feature_names: list

    @classmethod
    def fit(cls, x: np.ndarray, feature_names=None) -> "ArrayStandardizer":
        mean = np.mean(x.astype(np.float64), axis=0).astype(np.float32)
        std = np.std(x.astype(np.float64), axis=0).astype(np.float32)
        std = np.where(std < EPS, 1.0, std).astype(np.float32)
        if feature_names is None:
            feature_names = [f"f{i}" for i in range(x.shape[1])]
        return cls(mean=mean, std=std, feature_names=list(feature_names))

    def transform(self, x: np.ndarray) -> np.ndarray:
        return ((x.astype(np.float32) - self.mean) / self.std).astype(np.float32)

    def to_dict(self) -> Dict:
        return {"mean": self.mean.tolist(), "std": self.std.tolist(), "feature_names": self.feature_names}

    @classmethod
    def from_dict(cls, obj: Dict) -> "ArrayStandardizer":
        return cls(np.asarray(obj["mean"], dtype=np.float32), np.asarray(obj["std"], dtype=np.float32), list(obj["feature_names"]))


class VideoR2Dataset(Dataset):
    def __init__(self, npz_path: str, standardizer: ArrayStandardizer | None = None):
        data = np.load(npz_path, allow_pickle=True)
        x = data["features"].astype(np.float32)
        names = [str(v) for v in data["feature_names"].tolist()]
        if standardizer is None:
            standardizer = ArrayStandardizer.fit(x, names)
        self.standardizer = standardizer
        self.x = torch.from_numpy(standardizer.transform(x))
        self.y_rel = torch.from_numpy(data["label_relevant"].astype(np.float32))
        self.y_iou = torch.from_numpy(data["label_best_iou"].astype(np.float32))
        self.y_05 = torch.from_numpy(data["label_any_05"].astype(np.float32))
        self.y_07 = torch.from_numpy(data["label_any_07"].astype(np.float32))
        self.group_id = torch.from_numpy(data["group_id"].astype(np.int64))
        self.query_index = torch.from_numpy(data["query_index"].astype(np.int64))
        self.video_idx = torch.from_numpy(data["video_idx"].astype(np.int64))

    def __len__(self):
        return int(self.x.shape[0])

    def __getitem__(self, idx):
        return {
            "x": self.x[idx],
            "y_rel": self.y_rel[idx],
            "y_iou": self.y_iou[idx],
            "y_05": self.y_05[idx],
            "y_07": self.y_07[idx],
            "group_id": self.group_id[idx],
            "query_index": self.query_index[idx],
            "video_idx": self.video_idx[idx],
        }


def load_npz_features(npz_path: str):
    data = np.load(npz_path, allow_pickle=True)
    return data["features"].astype(np.float32), [str(v) for v in data["feature_names"].tolist()]
