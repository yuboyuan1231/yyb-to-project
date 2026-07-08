from __future__ import annotations

import torch
import torch.nn as nn


class JointVCMRScorer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weights = nn.Parameter(torch.tensor([1.0, 1.0, 0.4, 0.25, 0.25, -0.45], dtype=torch.float32))
        self.fb_scale = nn.Parameter(torch.tensor(0.35))
        self.risk_scale = nn.Parameter(torch.tensor(0.25))

    def forward(self, retr: dict[str, torch.Tensor], local: dict[str, torch.Tensor], feedback: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        w = self.weights
        video_final = retr["retriever_score"] + self.fb_scale * feedback["video_feedback"] - self.risk_scale * retr["wrong_video_risk"]
        score = (
            w[0] * video_final.unsqueeze(-1)
            + w[1] * local["span_score"]
            + w[2] * local["prem_span"]
            + w[3] * local["region_span"]
            + w[4] * local["amd_span"]
            + w[5] * local["false_positive_risk"]
        )
        return {"video_final": video_final, "vcmr_score": score}

