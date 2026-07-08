from __future__ import annotations

import torch
import torch.nn as nn


class SpanToVideoFeedback(nn.Module):
    def __init__(self, hidden_dim: int = 384, topk: int = 8) -> None:
        super().__init__()
        self.topk = int(topk)
        self.mlp = nn.Sequential(nn.LayerNorm(10), nn.Linear(10, hidden_dim // 4), nn.GELU(), nn.Linear(hidden_dim // 4, 1))

    def forward(self, span_score: torch.Tensor, prem: torch.Tensor, reg: torch.Tensor, amd: torch.Tensor, fp: torch.Tensor, span_mask: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        if span_mask is not None:
            span_score = span_score.masked_fill(~span_mask.bool(), -1e4)
        k = min(self.topk, span_score.shape[-1])
        vals, idx = torch.topk(span_score, k=k, dim=-1)
        p = torch.softmax(vals, dim=-1)
        entropy = -(p * torch.log(p.clamp_min(1e-8))).sum(dim=-1)
        best = vals[..., 0]
        mean = vals.mean(dim=-1)
        lse = torch.logsumexp(vals, dim=-1)
        margin = vals[..., 0] - vals[..., 1] if k > 1 else vals[..., 0] * 0.0
        concentration = best - mean
        gather = lambda x: torch.gather(x, -1, idx).mean(dim=-1)
        focus = 1.0 - (gather(prem) - gather(reg)).abs()
        amd_agree = 1.0 - (gather(amd) - gather(reg)).abs()
        high_count = (vals > mean.unsqueeze(-1)).float().sum(dim=-1) / float(k)
        hard_risk = torch.gather(fp, -1, idx).mean(dim=-1)
        feats = torch.stack([best, mean, lse, margin, entropy, concentration, focus, amd_agree, high_count, hard_risk], dim=-1)
        return {"video_feedback": self.mlp(feats).squeeze(-1), "feedback_features": feats}
