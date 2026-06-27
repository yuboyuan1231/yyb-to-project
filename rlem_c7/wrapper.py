from __future__ import annotations

from typing import Any, Dict, Optional

import torch
from torch import nn


class ProposalConfidenceHead(nn.Module):
    """Frozen-backbone C7-B1 proposal confidence/span-quality head."""

    def __init__(self, in_dim: int, hidden: int = 512, layers: int = 4, dropout: float = 0.12):
        super().__init__()
        mods = []
        d = in_dim
        for _ in range(layers):
            mods += [nn.Linear(d, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(dropout)]
            d = hidden
        self.backbone = nn.Sequential(*mods)
        self.proposal_confidence = nn.Linear(hidden, 1)
        self.proposal_iou05_logit = nn.Linear(hidden, 1)
        self.proposal_iou07_logit = nn.Linear(hidden, 1)
        self.span_quality_logit = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        h = self.backbone(x)
        return {
            "proposal_confidence": self.proposal_confidence(h).squeeze(-1),
            "proposal_iou05_logit": self.proposal_iou05_logit(h).squeeze(-1),
            "proposal_iou07_logit": self.proposal_iou07_logit(h).squeeze(-1),
            "span_quality_logit": self.span_quality_logit(h).squeeze(-1),
        }


class CONQUER_RLEM_Wrapper(nn.Module):
    """C7 wrapper that leaves base CONQUER untouched when disabled.

    The wrapper is intentionally thin: C7-B0/B1 can freeze the base model and
    use exported intermediates, while disabled mode delegates to the original
    CONQUER scoring path exactly.
    """

    def __init__(
        self,
        conquer_model: nn.Module,
        proposal_head: Optional[ProposalConfidenceHead] = None,
        enabled: bool = False,
    ):
        super().__init__()
        self.conquer = conquer_model
        self.proposal_head = proposal_head
        self.enabled = bool(enabled)
        self.freeze_conquer()

    def freeze_conquer(self) -> None:
        for param in self.conquer.parameters():
            param.requires_grad_(False)
        self.conquer.eval()

    def forward_disabled(self, batch: Dict[str, Any]):
        return self.conquer.get_pred_from_raw_query(batch, return_intermediates=False)

    def forward_with_intermediates(self, batch: Dict[str, Any]):
        return self.conquer.get_pred_from_raw_query(batch, return_intermediates=True)

    def forward(self, batch: Dict[str, Any]):
        if not self.enabled or self.proposal_head is None:
            return self.forward_disabled(batch)
        video_score, begin, end, intermediates = self.forward_with_intermediates(batch)
        return {
            "video_similarity_score": video_score,
            "begin_score_distribution": begin,
            "end_score_distribution": end,
            "intermediates": intermediates,
        }
