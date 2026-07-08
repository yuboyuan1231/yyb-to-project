from __future__ import annotations

import torch
import torch.nn as nn


class CorpusRetriever(nn.Module):
    def __init__(self, hidden_dim: int = 384) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(10.0))
        self.risk = nn.Sequential(nn.LayerNorm(5), nn.Linear(5, hidden_dim // 4), nn.GELU(), nn.Linear(hidden_dim // 4, 1))

    def score_candidates(self, q: dict[str, torch.Tensor], v: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        sv = torch.einsum("bd,bcd->bc", q["q_visual"], v["visual_pool"])
        ss = torch.einsum("bd,bcd->bc", q["q_subtitle"], v["subtitle_pool"])
        sj = torch.einsum("bd,bcd->bc", q["q_joint"], v["joint_pool"])
        g = q["gate"]
        score = self.scale.clamp(1.0, 30.0) * (g[:, 0:1] * sv + g[:, 1:2] * ss + g[:, 2:3] * sj)
        feats = torch.stack([sv, ss, sj, sv - ss, score.detach() / 30.0], dim=-1)
        wrong_risk = torch.sigmoid(self.risk(feats).squeeze(-1))
        return {"retriever_score": score, "visual_sim": sv, "subtitle_sim": ss, "joint_sim": sj, "wrong_video_risk": wrong_risk}

    def score_bank(self, q: dict[str, torch.Tensor], bank: dict[str, torch.Tensor]) -> torch.Tensor:
        sv = q["q_visual"] @ bank["visual_pool"].T
        ss = q["q_subtitle"] @ bank["subtitle_pool"].T
        sj = q["q_joint"] @ bank["joint_pool"].T
        g = q["gate"]
        return self.scale.clamp(1.0, 30.0) * (g[:, 0:1] * sv + g[:, 1:2] * ss + g[:, 2:3] * sj)

