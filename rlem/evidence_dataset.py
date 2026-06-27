
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

"""Datasets and normalization stats for C2/C3 evidence-head training."""

import math
import random
from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, List, Optional, Set

import numpy as np
import torch
from torch.utils.data import IterableDataset

from rlem.io_utils import iter_jsonl


EPS = 1e-8


def _as_float(value, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(out):
        return default
    return out


@dataclass
class FeatureStats:
    feature_names: List[str]
    mean: List[float]
    std: List[float]
    count: int
    label_counts: Dict[str, float]

    @classmethod
    def fit(
        cls,
        paths: Iterable[str],
        feature_names: List[str],
        max_rows: Optional[int] = None,
        allowed_desc_ids: Optional[Set[str]] = None,
    ):
        n_feat = len(feature_names)
        count = 0
        mean = np.zeros(n_feat, dtype=np.float64)
        m2 = np.zeros(n_feat, dtype=np.float64)
        label_counts = {
            "rows": 0,
            "y_joint_positive": 0,
            "y_joint_05_positive": 0,
            "y_joint_07_positive": 0,
            "m_bd_positive": 0,
            "y_fp_positive": 0,
        }
        remaining = max_rows
        for path in paths:
            for row in iter_jsonl(path):
                if allowed_desc_ids is not None and str(row.get("desc_id")) not in allowed_desc_ids:
                    continue
                x = np.array([_as_float(row.get(k)) for k in feature_names], dtype=np.float64)
                count += 1
                delta = x - mean
                mean += delta / count
                delta2 = x - mean
                m2 += delta * delta2
                label_counts["rows"] += 1
                label_counts["y_joint_positive"] += int(_as_float(row.get("y_joint")) > 0)
                label_counts["y_joint_05_positive"] += int(_as_float(row.get("y_joint_05")) > 0)
                label_counts["y_joint_07_positive"] += int(_as_float(row.get("y_joint_07")) > 0)
                label_counts["m_bd_positive"] += int(_as_float(row.get("m_bd")) > 0)
                label_counts["y_fp_positive"] += int(_as_float(row.get("y_fp")) > 0)
                if remaining is not None:
                    remaining -= 1
                    if remaining <= 0:
                        break
            if remaining is not None and remaining <= 0:
                break
        if count == 0:
            raise ValueError("No evidence rows found while fitting feature stats.")
        var = np.maximum(m2 / max(count - 1, 1), 0.0)
        std = np.sqrt(var)
        std[~np.isfinite(std) | (std < 1e-6)] = 1.0
        return cls(
            feature_names=list(feature_names),
            mean=mean.astype(float).tolist(),
            std=std.astype(float).tolist(),
            count=count,
            label_counts=label_counts,
        )

    @classmethod
    def from_dict(cls, obj: Dict):
        return cls(
            feature_names=list(obj["feature_names"]),
            mean=list(obj["mean"]),
            std=list(obj["std"]),
            count=int(obj.get("count", 0)),
            label_counts=dict(obj.get("label_counts", {})),
        )

    def to_dict(self) -> Dict:
        return {
            "feature_names": self.feature_names,
            "mean": self.mean,
            "std": self.std,
            "count": self.count,
            "label_counts": self.label_counts,
        }

    def transform_row(self, row: Dict) -> np.ndarray:
        x = np.array([_as_float(row.get(k)) for k in self.feature_names], dtype=np.float32)
        mean = np.asarray(self.mean, dtype=np.float32)
        std = np.asarray(self.std, dtype=np.float32)
        return (x - mean) / (std + EPS)


class StreamingEvidenceDataset(IterableDataset):
    """Streaming JSONL dataset for large C1 evidence artifacts.

    The dataset intentionally avoids loading all training rows into memory.  Set
    ``shuffle_buffer`` to a few thousand rows to get approximate per-epoch
    shuffling without materializing the full file.
    """

    def __init__(
        self,
        jsonl_path: str,
        stats: FeatureStats,
        max_rows: Optional[int] = None,
        shuffle_buffer: int = 0,
        seed: int = 13,
        neg_keep_prob: float = 1.0,
        allowed_desc_ids: Optional[Set[str]] = None,
    ):
        super().__init__()
        self.jsonl_path = jsonl_path
        self.stats = stats
        self.max_rows = max_rows
        self.shuffle_buffer = int(shuffle_buffer)
        self.seed = int(seed)
        self.neg_keep_prob = float(neg_keep_prob)
        self.allowed_desc_ids = allowed_desc_ids

    def _row_to_item(self, row: Dict) -> Dict[str, torch.Tensor]:
        x = self.stats.transform_row(row)
        return {
            "x": torch.from_numpy(x).float(),
            "y_joint": torch.tensor(_as_float(row.get("y_joint")), dtype=torch.float32),
            "y_joint_05": torch.tensor(_as_float(row.get("y_joint_05")), dtype=torch.float32),
            "y_joint_07": torch.tensor(_as_float(row.get("y_joint_07")), dtype=torch.float32),
            "y_bd": torch.tensor(_as_float(row.get("y_bd")), dtype=torch.float32),
            "m_bd": torch.tensor(_as_float(row.get("m_bd")), dtype=torch.float32),
            "y_fp": torch.tensor(_as_float(row.get("y_fp")), dtype=torch.float32),
        }

    def _iter_items_unshuffled(self) -> Iterator[Dict[str, torch.Tensor]]:
        rng = random.Random(self.seed)
        yielded = 0
        for row in iter_jsonl(self.jsonl_path):
            if self.allowed_desc_ids is not None and str(row.get("desc_id")) not in self.allowed_desc_ids:
                continue
            if self.neg_keep_prob < 1.0:
                is_any_positive = (
                    _as_float(row.get("y_joint")) > 0
                    or _as_float(row.get("m_bd")) > 0
                    or _as_float(row.get("y_fp")) > 0
                )
                if (not is_any_positive) and rng.random() > self.neg_keep_prob:
                    continue
            yield self._row_to_item(row)
            yielded += 1
            if self.max_rows is not None and yielded >= self.max_rows:
                break

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        if self.shuffle_buffer <= 1:
            yield from self._iter_items_unshuffled()
            return
        rng = random.Random(self.seed)
        buffer = []
        for item in self._iter_items_unshuffled():
            buffer.append(item)
            if len(buffer) >= self.shuffle_buffer:
                idx = rng.randrange(len(buffer))
                yield buffer.pop(idx)
        rng.shuffle(buffer)
        for item in buffer:
            yield item


def evidence_collate(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    keys = batch[0].keys()
    return {k: torch.stack([item[k] for item in batch], dim=0) for k in keys}
