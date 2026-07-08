from __future__ import annotations

import torch
import torch.nn as nn


class MultiSpanProposal(nn.Module):
    """Proposal holder for architecture completeness; concrete spans come from TemporalGrid."""

    def forward(self, spans_clip: torch.Tensor) -> torch.Tensor:
        if spans_clip.shape[2] <= 1:
            raise RuntimeError("one-span materialization is forbidden")
        return spans_clip

