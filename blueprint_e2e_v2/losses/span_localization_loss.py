from __future__ import annotations

import torch
import torch.nn.functional as F


def span_localization_loss(span_score: torch.Tensor, span_iou: torch.Tensor, span_ge07: torch.Tensor, span_mask: torch.Tensor | None = None) -> torch.Tensor:
    target = span_iou.clamp(0.0, 1.0)
    weight = 1.0 + 4.0 * span_ge07.float()
    loss = weight * F.smooth_l1_loss(torch.sigmoid(span_score), target, reduction="none")
    if span_mask is not None:
        loss = loss * span_mask.float()
        return loss.sum() / span_mask.float().sum().clamp_min(1.0)
    return loss.mean()
