from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from blueprint_e2e_v2.data.feature_registry import FeaturePaths
from blueprint_e2e_v2.utils.io import read_ids


class SplitManager:
    allowed = {"train_fit", "calib_select", "calib_holdout", "pseudo_official_holdout"}

    def __init__(self, paths: FeaturePaths | None = None) -> None:
        self.paths = paths or FeaturePaths()
        self._records: dict[int, dict[str, Any]] | None = None

    def ids(self, split: str, max_queries: int | None = None) -> list[int]:
        if split not in self.allowed:
            raise ValueError(f"unknown split {split}")
        out = read_ids(self.paths.split_dir / f"{split}_desc_ids.txt")
        if max_queries and max_queries > 0:
            out = out[: int(max_queries)]
        return out

    def records_by_id(self) -> dict[int, dict[str, Any]]:
        if self._records is None:
            rows: dict[int, dict[str, Any]] = {}
            with self.paths.train_jsonl.open("r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    rows[int(row["desc_id"])] = row
            self._records = rows
        return self._records

    def records(self, split: str, max_queries: int | None = None) -> list[dict[str, Any]]:
        by_id = self.records_by_id()
        return [by_id[int(i)] for i in self.ids(split, max_queries=max_queries) if int(i) in by_id]

    def audit(self) -> dict[str, Any]:
        counts = {s: len(self.ids(s)) for s in sorted(self.allowed)}
        return {
            "split_counts": counts,
            "pseudo_official_holdout_used_for_selection": False,
            "split_dir": str(Path(self.paths.split_dir)),
        }

