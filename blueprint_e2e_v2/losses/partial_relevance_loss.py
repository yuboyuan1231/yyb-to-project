from __future__ import annotations

import torch


def partial_relevance_loss(prem_span: torch.Tensor, span_iou: torch.Tensor, correct_video: torch.Tensor, span_mask: torch.Tensor | None = None) -> torch.Tensor:
    target = (span_iou * correct_video.unsqueeze(-1).float()).detach()
    loss = (prem_span - target).pow(2) * (1.0 + 2.0 * target)
    if span_mask is not None:
        loss = loss * span_mask.float()
        return loss.sum() / span_mask.float().sum().clamp_min(1.0)
    return loss.mean()
