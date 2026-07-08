from __future__ import annotations

import torch
import torch.nn as nn


class RegionPrior(nn.Module):
    def __init__(self, hidden_dim: int = 384) -> None:
        super().__init__()
        self.mlp = nn.Sequential(nn.LayerNorm(hidden_dim * 2 + 3), nn.Linear(hidden_dim * 2 + 3, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1))

    def forward(self, enc: dict[str, torch.Tensor], partial: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        feats = torch.cat([
            enc["visual"],
            enc["subtitle"],
            partial["P_v"].unsqueeze(-1),
            partial["P_s"].unsqueeze(-1),
            partial["P_gate"].unsqueeze(-1),
        ], dim=-1)
        p = torch.sigmoid(self.mlp(feats).squeeze(-1))
        return {"P_reg": p}

