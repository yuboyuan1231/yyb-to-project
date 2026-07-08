from __future__ import annotations

import torch
import torch.nn.functional as F


def video_retrieval_loss(video_score: torch.Tensor, correct_video: torch.Tensor, front_rank_weight: float = 10.0) -> torch.Tensor:
    target = correct_video.float()
    target_sum = target.sum(dim=1, keepdim=True)
    valid = target_sum.squeeze(1) > 0
    if not bool(valid.any()):
        return video_score.sum() * 0.0
    video_score = video_score[valid]
    correct_video = correct_video[valid]
    target = target[valid] / target_sum[valid].clamp_min(1.0)
    logp = F.log_softmax(video_score, dim=1)
    ce = -(target * logp).sum(dim=1).mean()
    pos = (video_score * target).sum(dim=1)
    has_neg = (~correct_video.bool()).any(dim=1)
    if bool(has_neg.any()):
        neg = video_score[has_neg].masked_fill(correct_video[has_neg].bool(), -1e4).max(dim=1).values
        margin = F.relu(1.0 - pos[has_neg] + neg).mean()
    else:
        margin = video_score.sum() * 0.0
    return ce + float(front_rank_weight) * 0.05 * margin
