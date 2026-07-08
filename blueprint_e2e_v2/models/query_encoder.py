from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class QueryEncoder(nn.Module):
    def __init__(self, input_dim: int = 768, hidden_dim: int = 384, dropout: float = 0.1) -> None:
        super().__init__()
        self.visual_attn = nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1))
        self.subtitle_attn = nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1))
        self.joint_attn = nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))
        self.visual_proj = nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.LayerNorm(hidden_dim))
        self.subtitle_proj = nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.LayerNorm(hidden_dim))
        self.joint_proj = nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.LayerNorm(hidden_dim))
        self.gate = nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, 3))
        self.qtype_bias = nn.Embedding(4, 3)
        with torch.no_grad():
            self.qtype_bias.weight.copy_(torch.tensor([[1.2, 0.1, 0.4], [0.1, 1.2, 0.4], [0.7, 0.7, 0.8], [0.5, 0.5, 0.5]]))

    def _pool(self, tokens: torch.Tensor, mask: torch.Tensor | None, scorer: nn.Module) -> tuple[torch.Tensor, torch.Tensor]:
        logits = scorer(tokens).squeeze(-1)
        if mask is not None:
            logits = logits.masked_fill(~mask.bool(), -1e4)
        weights = torch.softmax(logits, dim=-1)
        pooled = torch.einsum("bl,bld->bd", weights, tokens)
        return pooled, weights

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor | None, qtype: torch.Tensor) -> dict[str, torch.Tensor]:
        qv_raw, av = self._pool(tokens, mask, self.visual_attn)
        qs_raw, a_s = self._pool(tokens, mask, self.subtitle_attn)
        qj_raw, aj = self._pool(tokens, mask, self.joint_attn)
        gate_source = qj_raw
        gate = torch.softmax(self.gate(gate_source) + self.qtype_bias(qtype.clamp(0, 3)), dim=-1)
        return {
            "q_visual": F.normalize(self.visual_proj(qv_raw), dim=-1),
            "q_subtitle": F.normalize(self.subtitle_proj(qs_raw), dim=-1),
            "q_joint": F.normalize(self.joint_proj(qj_raw), dim=-1),
            "gate": gate,
            "attn_visual": av,
            "attn_subtitle": a_s,
            "attn_joint": aj,
        }

