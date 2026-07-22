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

    @staticmethod
    def _apply_mask(x: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        if mask is None:
            return x
        return x * mask.bool().unsqueeze(-1)

    @staticmethod
    def _masked_pool(x: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        if mask is None:
            return F.normalize(x.mean(dim=2), dim=-1)
        m = mask.to(device=x.device, dtype=x.dtype).unsqueeze(-1)
        return F.normalize((x * m).sum(dim=2) / m.sum(dim=2).clamp_min(1.0), dim=-1)

    def forward(
        self,
        visual: torch.Tensor,
        subtitle: torch.Tensor,
        visual_mask: torch.Tensor | None = None,
        subtitle_mask: torch.Tensor | None = None,
        clip_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        b, c, t, _ = visual.shape
        vh = self._apply_mask(self.visual_proj(visual), visual_mask)
        sh = self._apply_mask(self.subtitle_proj(subtitle), subtitle_mask)
        jh = self.joint_proj(torch.cat([vh, sh], dim=-1))
        if clip_mask is None:
            if visual_mask is not None and subtitle_mask is not None:
                clip_mask = torch.logical_and(visual_mask.bool(), subtitle_mask.bool())
            elif visual_mask is not None:
                clip_mask = visual_mask.bool()
            elif subtitle_mask is not None:
                clip_mask = subtitle_mask.bool()
        jh = self._apply_mask(jh, clip_mask)
        prev = torch.cat([jh[:, :, :1], jh[:, :, :-1]], dim=2)
        nxt = torch.cat([jh[:, :, 1:], jh[:, :, -1:]], dim=2)
        if clip_mask is None:
            mixed = (jh + prev + nxt) / 3.0
        else:
            m = clip_mask.to(dtype=jh.dtype, device=jh.device).unsqueeze(-1)
            prev_m = torch.cat([m[:, :, :1], m[:, :, :-1]], dim=2)
            nxt_m = torch.cat([m[:, :, 1:], m[:, :, -1:]], dim=2)
            mixed = (jh * m + prev * prev_m + nxt * nxt_m) / (m + prev_m + nxt_m).clamp_min(1.0)
        th = (self.temporal(mixed) + jh) * 0.5
        th = self._apply_mask(th, clip_mask)
        return {
            "visual": F.normalize(vh, dim=-1),
            "subtitle": F.normalize(sh, dim=-1),
            "joint": F.normalize(th, dim=-1),
            "visual_pool": self._masked_pool(vh, visual_mask),
            "subtitle_pool": self._masked_pool(sh, subtitle_mask),
            "joint_pool": self._masked_pool(th, clip_mask),
        }

    def encode_pooled_bank(self, visual_mean: torch.Tensor, subtitle_mean: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        vh = F.normalize(self.visual_proj(visual_mean), dim=-1)
        if subtitle_mean is None:
            sh = torch.zeros_like(vh)
        else:
            sh = F.normalize(self.subtitle_proj(subtitle_mean), dim=-1)
        jh = F.normalize(self.joint_proj(torch.cat([vh, sh], dim=-1)), dim=-1)
        return {"visual_pool": vh, "subtitle_pool": sh, "joint_pool": jh}
