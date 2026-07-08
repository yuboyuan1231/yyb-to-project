from __future__ import annotations

import torch


def region_loss(region_span: torch.Tensor, span_iou: torch.Tensor, correct_video: torch.Tensor, span_mask: torch.Tensor | None = None) -> torch.Tensor:
    target = (span_iou >= 0.3).float() * correct_video.unsqueeze(-1).float()
    loss = torch.nn.functional.binary_cross_entropy(region_span.clamp(1e-6, 1 - 1e-6), target, reduction="none")
    if span_mask is not None:
        loss = loss * span_mask.float()
        return loss.sum() / span_mask.float().sum().clamp_min(1.0)
    return loss.mean()
