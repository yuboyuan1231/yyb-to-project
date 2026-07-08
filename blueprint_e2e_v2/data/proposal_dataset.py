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
        visual_seq_mask: np.ndarray | None = None,
        subtitle_seq_mask: np.ndarray | None = None,
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
        self.visual_seq_mask = visual_seq_mask
        self.subtitle_seq_mask = subtitle_seq_mask
        self.grid = TemporalGrid()
        self.video_duration_bank = dict(getattr(video_bank, "durations", {}))

    @staticmethod
    def _fit_seq(seq: np.ndarray, target_len: int = 64) -> np.ndarray:
        if seq.shape[0] == target_len:
            return seq.astype(np.float32)
        if seq.shape[0] > target_len:
            return seq[:target_len].astype(np.float32)
        out = np.zeros((target_len, seq.shape[1]), dtype=np.float32)
        out[: seq.shape[0]] = seq.astype(np.float32)
        return out

    @staticmethod
    def _fit_mask(seq_len: int, target_len: int = 64) -> np.ndarray:
        out = np.zeros((target_len,), dtype=np.bool_)
        out[: min(int(seq_len), int(target_len))] = True
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
        gt_duration = float(row["duration"])
        gt = (float(row["ts"][0]), float(row["ts"][1]))
        per_video_spans_clip = []
        per_video_spans_sec = []
        labels = []
        correct = []
        for vid in video_ids:
            is_correct_video = str(vid) == gt_vid
            duration = float(self.video_duration_bank.get(str(vid), gt_duration))
            spans_clip = self.grid.grid_spans(duration, max_spans=self.max_spans_per_video)
            if self.insert_gt_for_training and is_correct_video:
                gt_clip = np.array([[int(np.floor(gt[0] / self.grid.clip_len)), max(int(np.ceil(gt[1] / self.grid.clip_len)), int(np.floor(gt[0] / self.grid.clip_len)) + 1)]], dtype=np.int32)
                gt_clip[:, 0] = np.clip(gt_clip[:, 0], 0, self.grid.max_clips - 1)
                gt_clip[:, 1] = np.clip(gt_clip[:, 1], gt_clip[:, 0] + 1, self.grid.max_clips)
                if not ((spans_clip == gt_clip[0]).all(axis=1).any()):
                    spans_clip = np.concatenate([spans_clip, gt_clip], axis=0)
            spans_sec = self.grid.clips_to_seconds(spans_clip.copy(), duration)
            per_video_spans_clip.append(spans_clip)
            per_video_spans_sec.append(spans_sec)
            ious, ge05, ge07 = DiagnosticLabelBuilder.span_labels(spans_sec, gt, is_correct_video)
            labels.append({"iou": ious, "ge05": ge05, "ge07": ge07})
            correct.append(is_correct_video)
        if self.insert_gt_for_training and not any(correct):
            raise RuntimeError(f"C28C train target lost GT video for query {qid}: gt={gt_vid} candidates={video_ids[-5:]}")
        if self.visual_seq_bank is not None:
            visual = self.visual_seq_bank[np.asarray(video_indices, dtype=np.int64)].astype(np.float32, copy=False)
            if self.visual_seq_mask is None:
                visual_mask = np.ones(visual.shape[:2], dtype=np.bool_)
            else:
                visual_mask = self.visual_seq_mask[np.asarray(video_indices, dtype=np.int64)].astype(np.bool_, copy=False)
        else:
            visual_raw = [self.video_bank.sequence(v) for v in video_ids]
            visual = np.stack([self._fit_seq(x, self.grid.max_clips) for x in visual_raw]).astype(np.float32)
            visual_mask = np.stack([self._fit_mask(x.shape[0], self.grid.max_clips) for x in visual_raw]).astype(np.bool_)
        if self.subtitle_seq_bank is not None:
            subtitle = self.subtitle_seq_bank[np.asarray(video_indices, dtype=np.int64)].astype(np.float32, copy=False)
            if self.subtitle_seq_mask is None:
                subtitle_mask = np.ones(subtitle.shape[:2], dtype=np.bool_)
            else:
                subtitle_mask = self.subtitle_seq_mask[np.asarray(video_indices, dtype=np.int64)].astype(np.bool_, copy=False)
        else:
            subtitle_raw = [self.subtitle_bank.sequence(v) for v in video_ids]
            subtitle = np.stack([self._fit_seq(x, self.grid.max_clips) for x in subtitle_raw]).astype(np.float32)
            subtitle_mask = np.stack([self._fit_mask(x.shape[0], self.grid.max_clips) for x in subtitle_raw]).astype(np.bool_)
        clip_mask = np.logical_and(visual_mask, subtitle_mask)
        max_m = max(x.shape[0] for x in per_video_spans_clip)
        spans_clip_padded = np.zeros((len(video_ids), max_m, 2), dtype=np.int64)
        spans_sec_padded = np.zeros((len(video_ids), max_m, 2), dtype=np.float32)
        span_mask = np.zeros((len(video_ids), max_m), dtype=np.bool_)
        label_padded: dict[str, list[np.ndarray]] = {"iou": [], "ge05": [], "ge07": []}
        for i, (clip, sec, lab) in enumerate(zip(per_video_spans_clip, per_video_spans_sec, labels)):
            n = clip.shape[0]
            spans_clip_padded[i, :n] = clip.astype(np.int64)
            spans_sec_padded[i, :n] = sec.astype(np.float32)
            if n < max_m and n > 0:
                spans_clip_padded[i, n:] = clip[-1]
                spans_sec_padded[i, n:] = sec[-1]
            span_mask[i, :n] = True
            for key in label_padded:
                pad = np.zeros((max_m,), dtype=lab[key].dtype)
                pad[:n] = lab[key]
                label_padded[key].append(pad)
        return {
            "query_id": qid,
            "query_tokens": self.query_bank.tokens(qid),
            "query_type": {"v": 0, "t": 1, "vt": 2}.get(str(row.get("type", "")), 3),
            "video_ids": video_ids,
            "video_indices": np.array(video_indices, dtype=np.int64),
            "visual": visual,
            "subtitle": subtitle,
            "visual_clip_mask": visual_mask,
            "subtitle_clip_mask": subtitle_mask,
            "clip_mask": clip_mask,
            "spans_clip": spans_clip_padded.astype(np.int64),
            "spans_sec": spans_sec_padded.astype(np.float32),
            "span_mask": span_mask,
            "span_iou": np.stack(label_padded["iou"]).astype(np.float32),
            "span_ge05": np.stack(label_padded["ge05"]).astype(np.float32),
            "span_ge07": np.stack(label_padded["ge07"]).astype(np.float32),
            "correct_video": np.array(correct, dtype=np.bool_),
            "gt_video_id": gt_vid,
            "gt_start": gt[0],
            "gt_end": gt[1],
            "duration": gt_duration,
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
            "candidate_video_duration_bank_used": True,
            "candidate_proposals_use_candidate_video_duration": True,
            "preloaded_sequence_bank_used": self.visual_seq_bank is not None and self.subtitle_seq_bank is not None,
            "clip_mask_used": True,
            "temporal_padding_policy": "pad/truncate release feature sequences to 64 clips for tensor batching; missing features still raise KeyError",
        }
