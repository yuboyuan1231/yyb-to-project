from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from blueprint_e2e_v2.data.dynamic_candidate_miner import CandidateSet
from blueprint_e2e_v2.data.query_bank import QueryBank
from blueprint_e2e_v2.data.split_manager import SplitManager
from blueprint_e2e_v2.data.subtitle_bank import SubtitleBank
from blueprint_e2e_v2.data.temporal_grid import TemporalGrid, iou_1d
from blueprint_e2e_v2.data.video_bank import VideoBank


class DiagnosticLabelBuilder:
    @staticmethod
    def span_labels(spans_sec: np.ndarray, gt: tuple[float, float], correct_video: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        ious = np.array([iou_1d((float(s), float(e)), gt) for s, e in spans_sec], dtype=np.float32) if correct_video else np.zeros(len(spans_sec), dtype=np.float32)
        return ious, (ious >= 0.5), (ious >= 0.7)


class MultiSpanProposalDataset(Dataset):
    def __init__(
        self,
        split: str,
        split_manager: SplitManager,
        query_bank: QueryBank,
        video_bank: VideoBank,
        subtitle_bank: SubtitleBank,
        candidates: dict[int, CandidateSet],
        max_queries: int | None = None,
        max_candidates: int = 32,
        max_spans_per_video: int = 64,
        insert_gt_for_training: bool = True,
        visual_seq_bank: np.ndarray | None = None,
        subtitle_seq_bank: np.ndarray | None = None,
    ) -> None:
        self.split = split
        self.sm = split_manager
        self.query_bank = query_bank
        self.video_bank = video_bank
        self.subtitle_bank = subtitle_bank
        self.candidates = candidates
        self.records = self.sm.records(split, max_queries=max_queries)
        self.max_candidates = int(max_candidates)
        self.max_spans_per_video = int(max_spans_per_video)
        self.insert_gt_for_training = bool(insert_gt_for_training)
        self.visual_seq_bank = visual_seq_bank
        self.subtitle_seq_bank = subtitle_seq_bank
        self.grid = TemporalGrid()

    @staticmethod
    def _fit_seq(seq: np.ndarray, target_len: int = 64) -> np.ndarray:
        if seq.shape[0] == target_len:
            return seq.astype(np.float32)
        if seq.shape[0] > target_len:
            return seq[:target_len].astype(np.float32)
        out = np.zeros((target_len, seq.shape[1]), dtype=np.float32)
        out[: seq.shape[0]] = seq.astype(np.float32)
        return out

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.records[idx]
        qid = int(row["desc_id"])
        gt_vid = str(row["vid_name"])
        cand = self.candidates[qid]
        video_indices = cand.video_indices[: self.max_candidates]
        if self.insert_gt_for_training and cand.gt_video_index not in video_indices:
            video_indices[-1] = cand.gt_video_index
        video_ids = [self.video_bank.idx_to_video[int(i)] for i in video_indices]
        duration = float(row["duration"])
        gt = (float(row["ts"][0]), float(row["ts"][1]))
        spans_clip = self.grid.grid_spans(duration, max_spans=self.max_spans_per_video)
        if self.insert_gt_for_training:
            gt_clip = np.array([[int(np.floor(gt[0] / self.grid.clip_len)), max(int(np.ceil(gt[1] / self.grid.clip_len)), int(np.floor(gt[0] / self.grid.clip_len)) + 1)]], dtype=np.int32)
            gt_clip[:, 0] = np.clip(gt_clip[:, 0], 0, self.grid.max_clips - 1)
            gt_clip[:, 1] = np.clip(gt_clip[:, 1], gt_clip[:, 0] + 1, self.grid.max_clips)
            if not ((spans_clip == gt_clip[0]).all(axis=1).any()):
                spans_clip = np.concatenate([spans_clip, gt_clip], axis=0)
        spans_sec = self.grid.clips_to_seconds(spans_clip.copy(), duration)
        labels = []
        correct = []
        for vid in video_ids:
            ious, ge05, ge07 = DiagnosticLabelBuilder.span_labels(spans_sec, gt, str(vid) == gt_vid)
            labels.append({"iou": ious, "ge05": ge05, "ge07": ge07})
            correct.append(str(vid) == gt_vid)
        if self.insert_gt_for_training and not any(correct):
            raise RuntimeError(f"C28C train target lost GT video for query {qid}: gt={gt_vid} candidates={video_ids[-5:]}")
        if self.visual_seq_bank is not None:
            visual = self.visual_seq_bank[np.asarray(video_indices, dtype=np.int64)].astype(np.float32, copy=False)
        else:
            visual = np.stack([self._fit_seq(self.video_bank.sequence(v), self.grid.max_clips) for v in video_ids]).astype(np.float32)
        if self.subtitle_seq_bank is not None:
            subtitle = self.subtitle_seq_bank[np.asarray(video_indices, dtype=np.int64)].astype(np.float32, copy=False)
        else:
            subtitle = np.stack([self._fit_seq(self.subtitle_bank.sequence(v), self.grid.max_clips) for v in video_ids]).astype(np.float32)
        return {
            "query_id": qid,
            "query_tokens": self.query_bank.tokens(qid),
            "query_type": {"v": 0, "t": 1, "vt": 2}.get(str(row.get("type", "")), 3),
            "video_ids": video_ids,
            "video_indices": np.array(video_indices, dtype=np.int64),
            "visual": visual,
            "subtitle": subtitle,
            "spans_clip": spans_clip.astype(np.int64),
            "spans_sec": spans_sec.astype(np.float32),
            "span_iou": np.stack([x["iou"] for x in labels]).astype(np.float32),
            "span_ge05": np.stack([x["ge05"] for x in labels]).astype(np.float32),
            "span_ge07": np.stack([x["ge07"] for x in labels]).astype(np.float32),
            "correct_video": np.array(correct, dtype=np.bool_),
            "gt_video_id": gt_vid,
            "gt_start": gt[0],
            "gt_end": gt[1],
            "duration": duration,
        }

    def audit(self) -> dict[str, Any]:
        return {
            "split": self.split,
            "query_count": len(self.records),
            "max_candidates": self.max_candidates,
            "max_spans_per_video": self.max_spans_per_video,
            "one_span_materialization": False,
            "gt_oracle_used_as_inference_feature": False,
            "gt_video_inserted_for_training_loss": self.insert_gt_for_training,
            "gt_aligned_span_inserted_for_training_loss": self.insert_gt_for_training,
            "preloaded_sequence_bank_used": self.visual_seq_bank is not None and self.subtitle_seq_bank is not None,
            "temporal_padding_policy": "pad/truncate release feature sequences to 64 clips for tensor batching; missing features still raise KeyError",
        }
