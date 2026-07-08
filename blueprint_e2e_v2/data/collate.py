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
    max_m = max(b["spans_clip"].shape[0] for b in batch)
    cand_n = len(batch[0]["video_ids"])

    def pad_spans(key: str, fill: float = 0.0) -> np.ndarray:
        arrs = []
        for b in batch:
            x = b[key]
            pad = np.zeros((max_m,) + x.shape[1:], dtype=x.dtype)
            pad[: x.shape[0]] = x
            if x.shape[0] < max_m and x.shape[0] > 0 and key.startswith("spans"):
                pad[x.shape[0]:] = x[-1]
            arrs.append(pad)
        return np.stack(arrs)

    spans_clip = torch.tensor(pad_spans("spans_clip"), dtype=torch.long).unsqueeze(1).repeat(1, cand_n, 1, 1)
    spans_sec = torch.tensor(pad_spans("spans_sec"), dtype=torch.float32).unsqueeze(1).repeat(1, cand_n, 1, 1)
    span_mask_base = np.zeros((len(batch), max_m), dtype=np.bool_)
    for i, b in enumerate(batch):
        span_mask_base[i, : b["spans_clip"].shape[0]] = True
    span_mask = torch.tensor(span_mask_base, dtype=torch.bool).unsqueeze(1).repeat(1, cand_n, 1)

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
