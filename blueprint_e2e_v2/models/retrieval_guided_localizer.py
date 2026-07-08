from __future__ import annotations

import torch
import torch.nn as nn

from blueprint_e2e_v2.utils.tensor_ops import span_pool


class RetrievalGuidedLocalizer(nn.Module):
    def __init__(self, hidden_dim: int = 384, topk_feedback: int = 8) -> None:
        super().__init__()
        self.topk_feedback = int(topk_feedback)
        in_dim = hidden_dim * 3 + 14
        self.span_head = nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, hidden_dim), nn.GELU(), nn.Dropout(0.1), nn.Linear(hidden_dim, hidden_dim // 2), nn.GELU(), nn.Linear(hidden_dim // 2, 1))
        self.fp_head = nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, hidden_dim // 2), nn.GELU(), nn.Linear(hidden_dim // 2, 1))

    def forward(
        self,
        enc: dict[str, torch.Tensor],
        partial: dict[str, torch.Tensor],
        region: dict[str, torch.Tensor],
        active: dict[str, torch.Tensor],
        retr: dict[str, torch.Tensor],
        spans_clip: torch.Tensor,
        visual_mask: torch.Tensor | None = None,
        subtitle_mask: torch.Tensor | None = None,
        clip_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        vspan = span_pool(enc["visual"], spans_clip, visual_mask)
        sspan = span_pool(enc["subtitle"], spans_clip, subtitle_mask)
        jspan = span_pool(enc["joint"], spans_clip, clip_mask)
        prem = span_pool(partial["P_gate"].unsqueeze(-1), spans_clip, clip_mask).squeeze(-1)
        reg = span_pool(region["P_reg"].unsqueeze(-1), spans_clip, clip_mask).squeeze(-1)
        amd = span_pool(active["P_amd"].unsqueeze(-1), spans_clip, clip_mask).squeeze(-1)
        b, c, m, _ = vspan.shape
        scores = retr["retriever_score"]
        ranks = torch.argsort(torch.argsort(-scores, dim=1), dim=1).float() / max(1, c - 1)
        top2 = torch.topk(scores, k=min(2, c), dim=1).values
        margin = (top2[:, 0] - top2[:, -1]).unsqueeze(1).expand_as(scores)
        probs = torch.softmax(scores, dim=1)
        uncertainty = (-(probs * torch.log(probs.clamp_min(1e-8))).sum(dim=1, keepdim=True)).expand_as(scores)
        base = torch.stack([
            scores / 30.0,
            ranks,
            margin / 30.0,
            uncertainty,
            retr["wrong_video_risk"],
            retr["visual_sim"],
            retr["subtitle_sim"],
            retr["joint_sim"],
        ], dim=-1).unsqueeze(2).expand(-1, -1, m, -1)
        starts = spans_clip[..., 0].float()
        ends = spans_clip[..., 1].float()
        dur = (ends - starts).clamp_min(1.0)
        pos = torch.stack([starts / 64.0, ends / 64.0, dur / 64.0, prem, reg, amd], dim=-1)
        feats = torch.cat([vspan, sspan, jspan, base, pos], dim=-1)
        span_score = self.span_head(feats).squeeze(-1)
        fp_risk = torch.sigmoid(self.fp_head(feats).squeeze(-1))
        return {"span_score": span_score, "false_positive_risk": fp_risk, "prem_span": prem, "region_span": reg, "amd_span": amd}
