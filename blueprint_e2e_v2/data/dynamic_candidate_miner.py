from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch


@dataclass
class CandidateSet:
    query_id: int
    video_indices: list[int]
    scores: list[float]
    gt_video_index: int
    gt_inserted: bool


class CompactCandidateStore(Mapping[int, CandidateSet]):
    """Matrix-backed candidate store for full-scale C28E refreshes.

    The train split carries roughly 69k queries x 200 candidates. Keeping that
    as per-query Python lists costs several extra GB and can be killed by the
    OS after refresh. This store preserves the same candidate contents while
    materializing Python lists only for the sampled query.
    """

    def __init__(
        self,
        query_ids: np.ndarray,
        video_indices: np.ndarray,
        scores: np.ndarray,
        gt_video_indices: np.ndarray,
        gt_inserted: np.ndarray,
    ) -> None:
        self.query_ids = np.asarray(query_ids, dtype=np.int64)
        self.video_indices = np.asarray(video_indices, dtype=np.int32)
        self.scores = np.asarray(scores, dtype=np.float32)
        self.gt_video_indices = np.asarray(gt_video_indices, dtype=np.int32)
        self.gt_inserted = np.asarray(gt_inserted, dtype=np.bool_)
        self._row = {int(q): int(i) for i, q in enumerate(self.query_ids.tolist())}

    def __len__(self) -> int:
        return int(self.query_ids.shape[0])

    def __iter__(self) -> Iterator[int]:
        return (int(q) for q in self.query_ids)

    def __getitem__(self, query_id: int) -> CandidateSet:
        row = self._row[int(query_id)]
        return CandidateSet(
            int(query_id),
            self.video_indices[row].astype(np.int64, copy=False).tolist(),
            self.scores[row].astype(np.float32, copy=False).tolist(),
            int(self.gt_video_indices[row]),
            bool(self.gt_inserted[row]),
        )

    def arrays(self, query_id: int) -> tuple[np.ndarray, np.ndarray, int, bool]:
        row = self._row[int(query_id)]
        return (
            self.video_indices[row],
            self.scores[row],
            int(self.gt_video_indices[row]),
            bool(self.gt_inserted[row]),
        )

    def candidate_count_min(self) -> int:
        return int(self.video_indices.shape[1]) if self.video_indices.ndim == 2 and len(self) else 0

    def candidate_count_max(self) -> int:
        return int(self.video_indices.shape[1]) if self.video_indices.ndim == 2 and len(self) else 0

    def inserted_count(self) -> int:
        return int(self.gt_inserted.sum())


class DynamicCandidateMiner:
    """Legacy pooled miner retained for diagnostics.

    C28E full train/eval uses ``engine.refresh_hard_negatives.refresh_candidates``,
    which performs broad pooled recall followed by clip late-interaction rerank.
    """

    def __init__(self, video_ids: list[str], video_to_idx: dict[str, int], dynamic_topk: int = 200, chunk_size: int = 256) -> None:
        self.video_ids = video_ids
        self.video_to_idx = video_to_idx
        self.dynamic_topk = int(dynamic_topk)
        self.chunk_size = int(chunk_size)

    @torch.no_grad()
    def mine(
        self,
        model: torch.nn.Module,
        query_tokens: torch.Tensor,
        qtypes: torch.Tensor,
        visual_bank: torch.Tensor,
        subtitle_bank: torch.Tensor | None,
        query_ids: list[int],
        gt_video_ids: list[str],
        device: torch.device,
        query_mask: torch.Tensor | None = None,
    ) -> dict[int, CandidateSet]:
        model.eval()
        q = model.encode_query(query_tokens.to(device), qtypes.to(device), query_mask.to(device) if query_mask is not None else None)
        scores_all = []
        for st in range(0, visual_bank.shape[0], self.chunk_size):
            visual = visual_bank[st: st + self.chunk_size].to(device, non_blocking=True)
            subtitle = subtitle_bank[st: st + self.chunk_size].to(device, non_blocking=True) if subtitle_bank is not None else None
            scores = model.score_video_bank(q, visual, subtitle)
            scores_all.append(scores.float().cpu())
        scores_full = torch.cat(scores_all, dim=1)
        k = min(self.dynamic_topk, scores_full.shape[1])
        vals, idx = torch.topk(scores_full, k=k, dim=1)
        out: dict[int, CandidateSet] = {}
        for row, qid in enumerate(query_ids):
            vids = idx[row].tolist()
            vals_row = vals[row].tolist()
            gt_idx = self.video_to_idx[str(gt_video_ids[row])]
            inserted = False
            if gt_idx not in vids:
                vids[-1] = gt_idx
                vals_row[-1] = float(scores_full[row, gt_idx].item())
                inserted = True
            out[int(qid)] = CandidateSet(int(qid), [int(v) for v in vids], [float(v) for v in vals_row], int(gt_idx), inserted)
        return out

    def audit(self, candidates: dict[int, CandidateSet]) -> dict[str, Any]:
        if not candidates:
            return {"query_count": 0, "dynamic_topk": self.dynamic_topk, "gt_insert_rate": None}
        inserted = sum(1 for c in candidates.values() if c.gt_inserted)
        return {
            "query_count": len(candidates),
            "dynamic_topk": self.dynamic_topk,
            "candidate_count_min": min(len(c.video_indices) for c in candidates.values()),
            "candidate_count_max": max(len(c.video_indices) for c in candidates.values()),
            "gt_insert_rate": 100.0 * inserted / max(1, len(candidates)),
            "static_top128_hard_gate_used": False,
        }
