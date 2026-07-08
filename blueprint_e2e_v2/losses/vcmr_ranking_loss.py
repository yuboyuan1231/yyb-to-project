from __future__ import annotations

import torch
import torch.nn.functional as F


def joint_vcmr_ranking_loss(vcmr_score: torch.Tensor, span_iou: torch.Tensor, span_ge07: torch.Tensor, correct_video: torch.Tensor, front_rank_weight: float = 10.0, span_mask: torch.Tensor | None = None) -> torch.Tensor:
    b, c, m = vcmr_score.shape
    if span_mask is not None:
        vcmr_score = vcmr_score.masked_fill(~span_mask.bool(), -1e4)
        span_iou = span_iou * span_mask.float()
        span_ge07 = span_ge07 * span_mask.float()
    flat_score = vcmr_score.reshape(b, c * m)
    soft = (span_iou * correct_video.unsqueeze(-1).float()).reshape(b, c * m)
    hard07 = span_ge07.float().reshape(b, c * m)
    target = soft + float(front_rank_weight) * hard07
    target_sum = target.sum(dim=1, keepdim=True)
    valid = target_sum.squeeze(1) > 0
    if not bool(valid.any()):
        return flat_score.sum() * 0.0
    flat_score = flat_score[valid]
    target = target[valid] / target_sum[valid].clamp_min(1.0)
    ce = -(target * F.log_softmax(flat_score, dim=1)).sum(dim=1).mean()
    has_neg = (target <= 0).any(dim=1)
    if bool(has_neg.any()):
        pos = flat_score[has_neg].masked_fill(target[has_neg] <= 0, -1e4).max(dim=1).values
        neg = flat_score[has_neg].masked_fill(target[has_neg] > 0, -1e4).max(dim=1).values
        margin = F.relu(1.0 - pos + neg).mean()
    else:
        margin = flat_score.sum() * 0.0
    return ce + 0.25 * margin
