from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F


def prem_total_loss(out: Dict[str, torch.Tensor], labels: torch.Tensor, first_stage_z: torch.Tensor, weights: Dict[str, float] | None = None) -> Dict[str, torch.Tensor]:
    weights = weights or {}
    pos = labels.sum().clamp(min=1.0)
    neg = (1.0 - labels).sum().clamp(min=1.0)
    pos_weight = (neg / pos).clamp(1.0, 64.0)
    rank_loss = F.binary_cross_entropy_with_logits(out["video_score"], labels, pos_weight=pos_weight)
    partial_logit = out["prem_score"]
    partial_loss = F.binary_cross_entropy_with_logits(partial_logit, labels, pos_weight=pos_weight)
    strong_weak = F.margin_ranking_loss(
        out["visual_relevance"] + out["subtitle_relevance"],
        torch.zeros_like(out["visual_relevance"]),
        labels.mul(2.0).sub(1.0),
        margin=0.05,
        reduction="mean",
    )
    hard_negative = (F.relu(out["video_score"] - first_stage_z + 0.03) * (1.0 - labels)).mean()
    residual_norm = out["residual"].pow(2).mean()
    no_regression = F.relu((first_stage_z - out["video_score"]) * labels).mean()
    total = (
        weights.get("rank", 1.0) * rank_loss
        + weights.get("partial", 0.35) * partial_loss
        + weights.get("strong_weak", 0.08) * strong_weak
        + weights.get("hard_negative", 0.10) * hard_negative
        + weights.get("residual", 0.20) * residual_norm
        + weights.get("no_regression", 0.25) * no_regression
    )
    return {
        "loss": total,
        "L_video_rank": rank_loss.detach(),
        "L_partial_relevance": partial_loss.detach(),
        "L_strong_weak_contrastive": strong_weak.detach(),
        "L_hard_negative": hard_negative.detach(),
        "L_residual_norm": residual_norm.detach(),
        "L_no_regression": no_regression.detach(),
    }
