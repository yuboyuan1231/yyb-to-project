from __future__ import annotations

import torch
import torch.nn.functional as F


def active_moment_diversity_loss(anchor_centers: torch.Tensor, anchor_widths: torch.Tensor) -> torch.Tensor:
    centers = anchor_centers.sort(dim=1).values
    diffs = centers[:, 1:] - centers[:, :-1]
    return F.relu(0.05 - diffs).mean() + F.relu(0.03 - anchor_widths).mean()


def active_moment_relevance_loss(amd_span: torch.Tensor, span_iou: torch.Tensor, correct_video: torch.Tensor, span_mask: torch.Tensor | None = None) -> torch.Tensor:
    target = (span_iou * correct_video.unsqueeze(-1).float()).detach()
    loss = F.smooth_l1_loss(amd_span, target, reduction="none")
    if span_mask is not None:
        loss = loss * span_mask.float()
        return loss.sum() / span_mask.float().sum().clamp_min(1.0)
    return loss.mean()
