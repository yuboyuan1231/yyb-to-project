from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

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


def _safe_soft_topk(values: torch.Tensor, k: int, temperature: float, mask: torch.Tensor | None = None) -> torch.Tensor:
    if mask is not None:
        values = values.masked_fill(~mask.bool(), -1e4)
    k = min(int(k), values.shape[-1])
    vals = torch.topk(values, k=k, dim=-1).values
    weights = torch.softmax(vals / max(float(temperature), 1e-6), dim=-1)
    return (weights * vals).sum(dim=-1)


class C28CFullModel(nn.Module):
    def __init__(
        self,
        query_dim: int = 768,
        subtitle_dim: int = 768,
        visual_dim: int = 4352,
        hidden_dim: int = 384,
        late_interaction_enabled: bool = False,
        late_soft_topk: int = 8,
        late_temperature: float = 0.07,
        token_maxsim_weight: float = 0.0,
        pooled_score_weight: float = 1.0,
        late_score_weight: float = 0.0,
    ) -> None:
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
        self.late_interaction_enabled = bool(late_interaction_enabled)
        self.late_soft_topk = int(late_soft_topk)
        self.late_temperature = float(late_temperature)
        self.token_maxsim_weight = float(token_maxsim_weight)
        self.pooled_score_weight = float(pooled_score_weight)
        self.late_score_weight = float(late_score_weight)

    def encode_query(self, query_tokens: torch.Tensor, qtypes: torch.Tensor, query_mask: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        return self.query_encoder(query_tokens, query_mask, qtypes)

    def score_video_bank(self, q: dict[str, torch.Tensor], visual_mean: torch.Tensor, subtitle_mean: torch.Tensor | None = None) -> torch.Tensor:
        bank = self.video_encoder.encode_pooled_bank(visual_mean, subtitle_mean)
        return self.retriever.score_bank(q, bank)

    @staticmethod
    def _masked_pool(x: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        if mask is None:
            return F.normalize(x.mean(dim=2), dim=-1)
        m = mask.to(dtype=x.dtype).unsqueeze(-1)
        return F.normalize((x * m).sum(dim=2) / m.sum(dim=2).clamp_min(1.0), dim=-1)

    def _apply_clip_masks(self, enc: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        visual_mask = batch.get("visual_clip_mask")
        subtitle_mask = batch.get("subtitle_clip_mask")
        joint_mask = batch.get("clip_mask")
        if visual_mask is None and subtitle_mask is None and joint_mask is None:
            return enc
        out = dict(enc)
        if visual_mask is not None:
            out["visual"] = out["visual"] * visual_mask.bool().unsqueeze(-1)
            out["visual_pool"] = self._masked_pool(out["visual"], visual_mask)
        if subtitle_mask is not None:
            out["subtitle"] = out["subtitle"] * subtitle_mask.bool().unsqueeze(-1)
            out["subtitle_pool"] = self._masked_pool(out["subtitle"], subtitle_mask)
        if joint_mask is not None:
            out["joint"] = out["joint"] * joint_mask.bool().unsqueeze(-1)
            out["joint_pool"] = self._masked_pool(out["joint"], joint_mask)
        return out

    def _apply_late_interaction(self, q: dict[str, torch.Tensor], enc: dict[str, torch.Tensor], batch: dict[str, torch.Tensor], retr: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        visual_mask = batch.get("visual_clip_mask")
        subtitle_mask = batch.get("subtitle_clip_mask")
        joint_mask = batch.get("clip_mask")
        sv_t = torch.einsum("bd,bctd->bct", q["q_visual"], enc["visual"])
        ss_t = torch.einsum("bd,bctd->bct", q["q_subtitle"], enc["subtitle"])
        sj_t = torch.einsum("bd,bctd->bct", q["q_joint"], enc["joint"])
        sv = _safe_soft_topk(sv_t, self.late_soft_topk, self.late_temperature, visual_mask)
        ss = _safe_soft_topk(ss_t, self.late_soft_topk, self.late_temperature, subtitle_mask)
        sj = _safe_soft_topk(sj_t, self.late_soft_topk, self.late_temperature, joint_mask)
        gate = q["gate"]
        late_score = self.retriever.scale.clamp(1.0, 30.0) * (gate[:, 0:1] * sv + gate[:, 1:2] * ss + gate[:, 2:3] * sj)
        token_score = late_score.new_zeros(late_score.shape)
        if self.token_maxsim_weight > 0.0:
            tokens = batch["query_tokens"]
            token_h = F.normalize(self.query_encoder.joint_proj(tokens), dim=-1)
            token_sim = torch.einsum("bld,bctd->blct", token_h, enc["joint"])
            if joint_mask is not None:
                token_sim = token_sim.masked_fill(~joint_mask.bool().unsqueeze(1), -1e4)
            token_sim = token_sim.max(dim=-1).values
            qmask = batch.get("query_mask")
            if qmask is None:
                token_score = token_sim.mean(dim=1)
            else:
                mask = qmask.float().unsqueeze(-1)
                token_score = (token_sim * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
            token_score = self.retriever.scale.clamp(1.0, 30.0) * token_score
        combined = self.pooled_score_weight * retr["retriever_score"] + self.late_score_weight * late_score + self.token_maxsim_weight * token_score
        out = dict(retr)
        out["retriever_score_pooled"] = retr["retriever_score"]
        out["retriever_score_late"] = late_score
        out["retriever_score_token"] = token_score
        out["retriever_score"] = combined
        out["late_visual_sim"] = sv
        out["late_subtitle_sim"] = ss
        out["late_joint_sim"] = sj
        return out

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        q = self.query_encoder(batch["query_tokens"], batch.get("query_mask"), batch["query_type"])
        enc = self._apply_clip_masks(self.video_encoder(batch["visual"], batch["subtitle"]), batch)
        retr = self.retriever.score_candidates(q, enc)
        if self.late_interaction_enabled:
            retr = self._apply_late_interaction(q, enc, batch, retr)
        partial = self.partial(q, enc, batch.get("visual_clip_mask"), batch.get("subtitle_clip_mask"))
        region = self.region(enc, partial)
        active = self.active(q, candidate_count=batch["visual"].shape[1], t=batch["visual"].shape[2])
        spans = self.proposals(batch["spans_clip"])
        local = self.localizer(enc, partial, region, active, retr, spans)
        feedback = self.feedback(local["span_score"], local["prem_span"], local["region_span"], local["amd_span"], local["false_positive_risk"], batch.get("span_mask"))
        score = self.scorer(retr, local, feedback)
        return {"query": q, "enc": enc, "retr": retr, "partial": partial, "region": region, "active": active, "local": local, "feedback": feedback, "score": score}
