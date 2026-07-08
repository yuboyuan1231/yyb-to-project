from __future__ import annotations

import torch
import torch.nn as nn


class ActiveMoment(nn.Module):
    def __init__(self, hidden_dim: int = 384, anchors: int = 8) -> None:
        super().__init__()
        self.anchors = int(anchors)
        self.param = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, anchors * 3))

    def forward(self, q: dict[str, torch.Tensor], candidate_count: int, t: int) -> dict[str, torch.Tensor]:
        raw = self.param(q["q_joint"]).view(q["q_joint"].shape[0], self.anchors, 3)
        centers = torch.sigmoid(raw[..., 0])
        widths = torch.sigmoid(raw[..., 1]) * 0.45 + 0.04
        weights = torch.softmax(raw[..., 2], dim=-1)
        grid = torch.linspace(0.0, 1.0, t, device=q["q_joint"].device).view(1, 1, t)
        masks = torch.exp(-((grid - centers.unsqueeze(-1)) ** 2) / (2.0 * widths.unsqueeze(-1).pow(2).clamp_min(1e-4)))
        p = torch.einsum("bh,bht->bt", weights, masks)
        p = p / p.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        p = p.unsqueeze(1).expand(-1, candidate_count, -1)
        return {"P_amd": p, "anchor_centers": centers, "anchor_widths": widths, "anchor_weights": weights, "anchor_masks": masks}

