from __future__ import annotations

import torch
import torch.nn as nn


class PartialRelevance(nn.Module):
    def __init__(self, hidden_dim: int = 384) -> None:
        super().__init__()
        self.log_tau_v = nn.Parameter(torch.tensor(0.07).log())
        self.log_tau_s = nn.Parameter(torch.tensor(0.07).log())
        self.gate_refine = nn.Sequential(nn.LayerNorm(6), nn.Linear(6, hidden_dim // 4), nn.GELU(), nn.Linear(hidden_dim // 4, 2))

    def forward(self, q: dict[str, torch.Tensor], enc: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        tv = self.log_tau_v.exp().clamp(0.02, 1.0)
        ts = self.log_tau_s.exp().clamp(0.02, 1.0)
        pv_logits = torch.einsum("bd,bctd->bct", q["q_visual"], enc["visual"]) / tv
        ps_logits = torch.einsum("bd,bctd->bct", q["q_subtitle"], enc["subtitle"]) / ts
        pv = torch.softmax(pv_logits, dim=-1)
        ps = torch.softmax(ps_logits, dim=-1)
        ent_v = -(pv * torch.log(pv.clamp_min(1e-8))).sum(dim=-1)
        ent_s = -(ps * torch.log(ps.clamp_min(1e-8))).sum(dim=-1)
        js = 0.5 * ((pv - ps).abs().mean(dim=-1))
        g0 = q["gate"][:, :2].unsqueeze(1).expand(-1, pv.shape[1], -1)
        stats = torch.stack([ent_v, ent_s, js, pv.max(dim=-1).values, ps.max(dim=-1).values, enc["subtitle_pool"].norm(dim=-1)], dim=-1)
        refine = torch.softmax(self.gate_refine(stats) + torch.log(g0.clamp_min(1e-6)), dim=-1)
        pgate = refine[..., 0:1] * pv + refine[..., 1:2] * ps
        return {"P_v": pv, "P_s": ps, "P_gate": pgate, "prem_gate": refine, "partial_js": js}

