from __future__ import annotations

import torch
import torch.nn as nn


class ActiveMoment(nn.Module):
    def __init__(self, hidden_dim: int = 384, anchors: int = 8) -> None:
        super().__init__()
        self.anchors = int(anchors)
        self.param = nn.Sequential(nn.LayerNorm(hidden_dim * 3), nn.Linear(hidden_dim * 3, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, anchors * 3))

    def forward(
        self,
        q: dict[str, torch.Tensor],
        enc: dict[str, torch.Tensor] | None = None,
        candidate_count: int | None = None,
        t: int | None = None,
        clip_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if enc is None:
            if candidate_count is None or t is None:
                raise ValueError("ActiveMoment requires enc or both candidate_count and t")
            q_joint = q["q_joint"].unsqueeze(1).expand(-1, int(candidate_count), -1)
            v_joint = torch.zeros_like(q_joint)
        else:
            q_joint = q["q_joint"].unsqueeze(1).expand(-1, enc["joint_pool"].shape[1], -1)
            v_joint = enc["joint_pool"]
            t = int(enc["joint"].shape[2])
        raw = self.param(torch.cat([q_joint, v_joint, q_joint * v_joint], dim=-1)).view(q_joint.shape[0], q_joint.shape[1], self.anchors, 3)
        centers = torch.sigmoid(raw[..., 0])
        widths = torch.sigmoid(raw[..., 1]) * 0.45 + 0.04
        weights = torch.softmax(raw[..., 2], dim=-1)
        grid = torch.linspace(0.0, 1.0, int(t), device=q["q_joint"].device).view(1, 1, 1, int(t))
        masks = torch.exp(-((grid - centers.unsqueeze(-1)) ** 2) / (2.0 * widths.unsqueeze(-1).pow(2).clamp_min(1e-4)))
        p = torch.einsum("bca,bcat->bct", weights, masks)
        if clip_mask is not None:
            p = p * clip_mask.to(device=p.device, dtype=p.dtype)
        p = p / p.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        return {"P_amd": p, "anchor_centers": centers, "anchor_widths": widths, "anchor_weights": weights, "anchor_masks": masks}
