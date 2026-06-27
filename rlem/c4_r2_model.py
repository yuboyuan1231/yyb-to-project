#!/usr/bin/env python
"""External video-level calibrator model for C4-r2-cal.

This is deliberately stronger than a single linear head but still lightweight:
LayerNorm + residual MLP blocks + multitask outputs.  It is an external
calibrator over frozen C4 video features, not a CONQUER backbone VS/R2 head.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence

import torch
import torch.nn as nn


class ResidualMLPBlock(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class VideoR2Calibrator(nn.Module):
    """External video-level score head.

    Outputs:
      - video_logit: relevance logit for video-level ranking
      - quality_logit: auxiliary best-IoU / useful-video quality logit
    """

    def __init__(self, input_dim: int, hidden_dim: int = 256, num_blocks: int = 2, dropout: float = 0.1):
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_blocks = int(num_blocks)
        self.dropout = float(dropout)
        self.in_proj = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.blocks = nn.Sequential(*[ResidualMLPBlock(hidden_dim, dropout) for _ in range(num_blocks)])
        self.out_norm = nn.LayerNorm(hidden_dim)
        self.video_head = nn.Linear(hidden_dim, 1)
        self.quality_head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor):
        h = self.in_proj(x)
        h = self.blocks(h)
        h = self.out_norm(h)
        return {
            "video_logit": self.video_head(h).squeeze(-1),
            "quality_logit": self.quality_head(h).squeeze(-1),
        }


def model_config_from_instance(model: VideoR2Calibrator) -> dict:
    return {
        "input_dim": model.input_dim,
        "hidden_dim": model.hidden_dim,
        "num_blocks": model.num_blocks,
        "dropout": model.dropout,
        "model_type": "VideoR2Calibrator",
    }
