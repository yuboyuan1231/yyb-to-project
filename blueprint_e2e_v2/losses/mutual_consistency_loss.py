from __future__ import annotations

import torch
import torch.nn.functional as F


def retrieval_guided_localization_loss(span_score: torch.Tensor, video_score: torch.Tensor, correct_video: torch.Tensor, span_mask: torch.Tensor | None = None) -> torch.Tensor:
    if span_mask is not None:
        span_score = span_score.masked_fill(~span_mask.bool(), -1e4)
    best_span = span_score.max(dim=-1).values
    return F.smooth_l1_loss(torch.softmax(best_span, dim=1), torch.softmax(video_score.detach(), dim=1)) + 0.2 * F.binary_cross_entropy_with_logits(best_span, correct_video.float())


def localization_to_retrieval_feedback_loss(video_final: torch.Tensor, span_score: torch.Tensor, correct_video: torch.Tensor, span_mask: torch.Tensor | None = None) -> torch.Tensor:
    if span_mask is not None:
        span_score = span_score.masked_fill(~span_mask.bool(), -1e4)
    span_dist = span_score.max(dim=-1).values.detach()
    return F.smooth_l1_loss(torch.softmax(video_final, dim=1), torch.softmax(span_dist, dim=1)) + 0.2 * F.binary_cross_entropy_with_logits(video_final, correct_video.float())


def video_span_consistency_loss(video_final: torch.Tensor, span_score: torch.Tensor, span_mask: torch.Tensor | None = None) -> torch.Tensor:
    if span_mask is not None:
        span_score = span_score.masked_fill(~span_mask.bool(), -1e4)
    return F.smooth_l1_loss(video_final, span_score.max(dim=-1).values.detach())
