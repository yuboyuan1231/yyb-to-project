from __future__ import annotations

from typing import Any

import numpy as np
import torch


def _pad_tokens(xs: list[np.ndarray]) -> tuple[torch.Tensor, torch.Tensor]:
    max_len = max(x.shape[0] for x in xs)
    dim = xs[0].shape[1]
    out = np.zeros((len(xs), max_len, dim), dtype=np.float32)
    mask = np.zeros((len(xs), max_len), dtype=np.bool_)
    for i, x in enumerate(xs):
        out[i, : x.shape[0]] = x
        mask[i, : x.shape[0]] = True
    return torch.from_numpy(out), torch.from_numpy(mask)


def c28c_collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    q, qmask = _pad_tokens([b["query_tokens"] for b in batch])
    cand_n = len(batch[0]["video_ids"])
    max_m = max(b["spans_clip"].shape[-2] for b in batch)

    def pad_candidate_spans(key: str) -> np.ndarray:
        arrs = []
        for b in batch:
            x = b[key]
            if x.ndim == 2:
                pad = np.zeros((cand_n, max_m, x.shape[-1]), dtype=x.dtype)
                n = x.shape[0]
                pad[:, :n] = x[None, :, :]
                if n < max_m and n > 0:
                    pad[:, n:] = x[-1][None, None, :]
            else:
                pad = np.zeros((x.shape[0], max_m, x.shape[-1]), dtype=x.dtype)
                n = x.shape[1]
                pad[:, :n] = x
                if n < max_m and n > 0:
                    pad[:, n:] = x[:, -1:, :]
            arrs.append(pad)
        return np.stack(arrs)

    spans_clip = torch.tensor(pad_candidate_spans("spans_clip"), dtype=torch.long)
    spans_sec = torch.tensor(pad_candidate_spans("spans_sec"), dtype=torch.float32)
    span_mask_arrs = []
    for b in batch:
        if "span_mask" in b:
            x = b["span_mask"]
            pad = np.zeros((x.shape[0], max_m), dtype=np.bool_)
            pad[:, : x.shape[1]] = x
        else:
            n = b["spans_clip"].shape[0]
            pad = np.zeros((cand_n, max_m), dtype=np.bool_)
            pad[:, :n] = True
        span_mask_arrs.append(pad)
    span_mask = torch.tensor(np.stack(span_mask_arrs), dtype=torch.bool)

    def pad_label(key: str, dtype: Any) -> torch.Tensor:
        arrs = []
        for b in batch:
            x = b[key]
            pad = np.zeros((x.shape[0], max_m), dtype=x.dtype)
            pad[:, : x.shape[1]] = x
            arrs.append(pad)
        return torch.tensor(np.stack(arrs), dtype=dtype)

    return {
        "query_ids": [int(b["query_id"]) for b in batch],
        "query_tokens": q,
        "query_mask": qmask,
        "query_type": torch.tensor([int(b["query_type"]) for b in batch], dtype=torch.long),
        "video_ids": [b["video_ids"] for b in batch],
        "video_indices": torch.tensor(np.stack([b["video_indices"] for b in batch]), dtype=torch.long),
        "visual": torch.from_numpy(np.stack([b["visual"] for b in batch]).astype(np.float32, copy=False)),
        "subtitle": torch.from_numpy(np.stack([b["subtitle"] for b in batch]).astype(np.float32, copy=False)),
        "visual_clip_mask": torch.tensor(np.stack([b["visual_clip_mask"] for b in batch]), dtype=torch.bool),
        "subtitle_clip_mask": torch.tensor(np.stack([b["subtitle_clip_mask"] for b in batch]), dtype=torch.bool),
        "clip_mask": torch.tensor(np.stack([b["clip_mask"] for b in batch]), dtype=torch.bool),
        "spans_clip": spans_clip,
        "spans_sec": spans_sec,
        "span_mask": span_mask,
        "span_iou": pad_label("span_iou", torch.float32),
        "span_ge05": pad_label("span_ge05", torch.float32),
        "span_ge07": pad_label("span_ge07", torch.float32),
        "correct_video": torch.tensor(np.stack([b["correct_video"] for b in batch]), dtype=torch.bool),
        "gt_video_id": [b["gt_video_id"] for b in batch],
        "gt_start": torch.tensor([float(b["gt_start"]) for b in batch], dtype=torch.float32),
        "gt_end": torch.tensor([float(b["gt_end"]) for b in batch], dtype=torch.float32),
        "duration": torch.tensor([float(b["duration"]) for b in batch], dtype=torch.float32),
    }
