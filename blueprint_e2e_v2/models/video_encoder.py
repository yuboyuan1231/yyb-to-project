from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class VideoSubtitleEncoder(nn.Module):
    def __init__(self, visual_dim: int = 4352, subtitle_dim: int = 768, hidden_dim: int = 384, dropout: float = 0.1) -> None:
        super().__init__()
        self.visual_proj = nn.Sequential(nn.LayerNorm(visual_dim), nn.Linear(visual_dim, hidden_dim * 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim * 2, hidden_dim), nn.LayerNorm(hidden_dim))
        self.subtitle_proj = nn.Sequential(nn.LayerNorm(subtitle_dim), nn.Linear(subtitle_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.LayerNorm(hidden_dim))
        self.joint_proj = nn.Sequential(nn.LayerNorm(hidden_dim * 2), nn.Linear(hidden_dim * 2, hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.LayerNorm(hidden_dim))
        self.temporal = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim))

    def forward(self, visual: torch.Tensor, subtitle: torch.Tensor) -> dict[str, torch.Tensor]:
        b, c, t, _ = visual.shape
        vh = self.visual_proj(visual)
        sh = self.subtitle_proj(subtitle)
        jh = self.joint_proj(torch.cat([vh, sh], dim=-1))
        prev = torch.cat([jh[:, :, :1], jh[:, :, :-1]], dim=2)
        nxt = torch.cat([jh[:, :, 1:], jh[:, :, -1:]], dim=2)
        mixed = (jh + prev + nxt) / 3.0
        th = (self.temporal(mixed) + jh) * 0.5
        return {
            "visual": F.normalize(vh, dim=-1),
            "subtitle": F.normalize(sh, dim=-1),
            "joint": F.normalize(th, dim=-1),
            "visual_pool": F.normalize(vh.mean(dim=2), dim=-1),
            "subtitle_pool": F.normalize(sh.mean(dim=2), dim=-1),
            "joint_pool": F.normalize(th.mean(dim=2), dim=-1),
        }

    def encode_pooled_bank(self, visual_mean: torch.Tensor, subtitle_mean: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        vh = F.normalize(self.visual_proj(visual_mean), dim=-1)
        if subtitle_mean is None:
            sh = torch.zeros_like(vh)
        else:
            sh = F.normalize(self.subtitle_proj(subtitle_mean), dim=-1)
        jh = F.normalize(self.joint_proj(torch.cat([vh, sh], dim=-1)), dim=-1)
        return {"visual_pool": vh, "subtitle_pool": sh, "joint_pool": jh}
