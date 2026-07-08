from __future__ import annotations

import torch
import torch.nn.functional as F


def hard_negative_video_span_loss(vcmr_score: torch.Tensor, correct_video: torch.Tensor, span_mask: torch.Tensor | None = None) -> torch.Tensor:
    if span_mask is not None:
        vcmr_score = vcmr_score.masked_fill(~span_mask.bool(), -1e4)
    valid = correct_video.bool().any(dim=1) & (~correct_video.bool()).any(dim=1)
    if not bool(valid.any()):
        return vcmr_score.sum() * 0.0
    vcmr_score = vcmr_score[valid]
    correct_video = correct_video[valid]
    best = vcmr_score.max(dim=-1).values
    pos = best.masked_fill(~correct_video.bool(), -1e4).max(dim=1).values
    neg = best.masked_fill(correct_video.bool(), -1e4).max(dim=1).values
    return F.relu(1.5 - pos + neg).mean()
