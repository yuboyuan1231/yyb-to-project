from __future__ import annotations

import torch
import torch.nn.functional as F


def no_regression_loss(video_final: torch.Tensor, retriever_score: torch.Tensor, correct_video: torch.Tensor) -> torch.Tensor:
    delta = video_final - retriever_score
    has_pos = correct_video.bool().any(dim=1)
    has_neg = (~correct_video.bool()).any(dim=1)
    if bool(has_pos.any()):
        pos_delta = (delta[has_pos] * correct_video[has_pos].float()).sum(dim=1) / correct_video[has_pos].float().sum(dim=1).clamp_min(1.0)
        pos_loss = F.relu(-pos_delta).mean()
    else:
        pos_loss = delta.sum() * 0.0
    if bool(has_neg.any()):
        wrong_delta = delta[has_neg].masked_fill(correct_video[has_neg].bool(), -1e4).max(dim=1).values
        neg_loss = F.relu(wrong_delta - 0.5).mean()
    else:
        neg_loss = delta.sum() * 0.0
    return pos_loss + 0.5 * neg_loss
