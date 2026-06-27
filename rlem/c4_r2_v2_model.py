#!/usr/bin/env python
"""Fixed external C4-r2-cal-v2 model families."""

from __future__ import annotations

import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, dim * 2), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(dim * 2, dim), nn.Dropout(dropout),
        )

    def forward(self, x):
        return x + self.net(x)


class SharedMultiTargetV2(nn.Module):
    """Lightly shared control architecture for V2-C+B."""

    def __init__(self, input_dim: int, hidden_dim: int = 256, blocks: int = 2, dropout: float = 0.1):
        super().__init__()
        self.cfg = {
            "model_type": "SharedMultiTargetV2", "input_dim": input_dim,
            "hidden_dim": hidden_dim, "shared_blocks": blocks,
            "branch_blocks": 0, "dropout": dropout,
        }
        self.input = nn.Sequential(
            nn.LayerNorm(input_dim), nn.Linear(input_dim, hidden_dim),
            nn.GELU(), nn.Dropout(dropout),
        )
        self.blocks = nn.Sequential(*[ResidualBlock(hidden_dim, dropout) for _ in range(blocks)])
        self.norm = nn.LayerNorm(hidden_dim)
        self.rel_head = nn.Linear(hidden_dim, 1)
        self.iou05_head = nn.Linear(hidden_dim, 1)
        self.iou07_head = nn.Linear(hidden_dim, 1)
        self.quality_head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        h = self.norm(self.blocks(self.input(x)))
        return {
            "rel_logit": self.rel_head(h).squeeze(-1),
            "iou05_logit": self.iou05_head(h).squeeze(-1),
            "iou07_logit": self.iou07_head(h).squeeze(-1),
            "quality_logit": self.quality_head(h).squeeze(-1),
        }


class DecoupledMultiTargetV2(nn.Module):
    """One shared block followed by independent relevance/quality branches."""

    def __init__(
        self, input_dim: int, hidden_dim: int = 256,
        shared_blocks: int = 1, branch_blocks: int = 2, dropout: float = 0.1,
    ):
        super().__init__()
        self.cfg = {
            "model_type": "DecoupledMultiTargetV2", "input_dim": input_dim,
            "hidden_dim": hidden_dim, "shared_blocks": shared_blocks,
            "branch_blocks": branch_blocks, "dropout": dropout,
        }
        self.input = nn.Sequential(
            nn.LayerNorm(input_dim), nn.Linear(input_dim, hidden_dim),
            nn.GELU(), nn.Dropout(dropout),
        )
        self.shared = nn.Sequential(*[ResidualBlock(hidden_dim, dropout) for _ in range(shared_blocks)])
        self.rel_branch = nn.Sequential(*[ResidualBlock(hidden_dim, dropout) for _ in range(branch_blocks)])
        self.quality_branch = nn.Sequential(*[ResidualBlock(hidden_dim, dropout) for _ in range(branch_blocks)])
        self.rel_norm = nn.LayerNorm(hidden_dim)
        self.quality_norm = nn.LayerNorm(hidden_dim)
        self.rel_head = nn.Linear(hidden_dim, 1)
        self.iou05_head = nn.Linear(hidden_dim, 1)
        self.iou07_head = nn.Linear(hidden_dim, 1)
        self.quality_head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        shared = self.shared(self.input(x))
        rel = self.rel_norm(self.rel_branch(shared))
        quality = self.quality_norm(self.quality_branch(shared))
        return {
            "rel_logit": self.rel_head(rel).squeeze(-1),
            "iou05_logit": self.iou05_head(rel).squeeze(-1),
            "iou07_logit": self.iou07_head(rel).squeeze(-1),
            "quality_logit": self.quality_head(quality).squeeze(-1),
        }


def build_v2_model(group: str, input_dim: int):
    if group == "v2_cb":
        return SharedMultiTargetV2(input_dim, hidden_dim=256, blocks=2, dropout=0.1)
    if group in {"v2_ab", "v2_acb"}:
        return DecoupledMultiTargetV2(
            input_dim, hidden_dim=256, shared_blocks=1, branch_blocks=2, dropout=0.1,
        )
    raise ValueError(group)


def model_config(model, group: str):
    return {
        **model.cfg, "group": group, "activation": "GELU",
        "normalization": "LayerNorm", "precision": "float32", "seed": 13,
        "loss_family": "original_rel_quality" if group == "v2_ab" else "iou05_aware_multi_target",
    }

