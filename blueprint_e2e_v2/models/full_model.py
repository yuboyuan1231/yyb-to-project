from __future__ import annotations

import torch
import torch.nn as nn

from blueprint_e2e_v2.models.active_moment import ActiveMoment
from blueprint_e2e_v2.models.partial_relevance import PartialRelevance
from blueprint_e2e_v2.models.proposal_generator import MultiSpanProposal
from blueprint_e2e_v2.models.query_encoder import QueryEncoder
from blueprint_e2e_v2.models.region_prior import RegionPrior
from blueprint_e2e_v2.models.retrieval_guided_localizer import RetrievalGuidedLocalizer
from blueprint_e2e_v2.models.retriever import CorpusRetriever
from blueprint_e2e_v2.models.span_to_video_feedback import SpanToVideoFeedback
from blueprint_e2e_v2.models.vcmr_scorer import JointVCMRScorer
from blueprint_e2e_v2.models.video_encoder import VideoSubtitleEncoder


class C28CFullModel(nn.Module):
    def __init__(self, query_dim: int = 768, subtitle_dim: int = 768, visual_dim: int = 4352, hidden_dim: int = 384) -> None:
        super().__init__()
        self.query_encoder = QueryEncoder(query_dim, hidden_dim)
        self.video_encoder = VideoSubtitleEncoder(visual_dim, subtitle_dim, hidden_dim)
        self.retriever = CorpusRetriever(hidden_dim)
        self.partial = PartialRelevance(hidden_dim)
        self.region = RegionPrior(hidden_dim)
        self.active = ActiveMoment(hidden_dim)
        self.proposals = MultiSpanProposal()
        self.localizer = RetrievalGuidedLocalizer(hidden_dim)
        self.feedback = SpanToVideoFeedback(hidden_dim)
        self.scorer = JointVCMRScorer()

    def encode_query(self, query_tokens: torch.Tensor, qtypes: torch.Tensor, query_mask: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        return self.query_encoder(query_tokens, query_mask, qtypes)

    def score_video_bank(self, q: dict[str, torch.Tensor], visual_mean: torch.Tensor, subtitle_mean: torch.Tensor | None = None) -> torch.Tensor:
        bank = self.video_encoder.encode_pooled_bank(visual_mean, subtitle_mean)
        return self.retriever.score_bank(q, bank)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        q = self.query_encoder(batch["query_tokens"], batch.get("query_mask"), batch["query_type"])
        enc = self.video_encoder(batch["visual"], batch["subtitle"])
        retr = self.retriever.score_candidates(q, enc)
        partial = self.partial(q, enc)
        region = self.region(enc, partial)
        active = self.active(q, candidate_count=batch["visual"].shape[1], t=batch["visual"].shape[2])
        spans = self.proposals(batch["spans_clip"])
        local = self.localizer(enc, partial, region, active, retr, spans)
        feedback = self.feedback(local["span_score"], local["prem_span"], local["region_span"], local["amd_span"], local["false_positive_risk"], batch.get("span_mask"))
        score = self.scorer(retr, local, feedback)
        return {"query": q, "enc": enc, "retr": retr, "partial": partial, "region": region, "active": active, "local": local, "feedback": feedback, "score": score}
