from __future__ import annotations

from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class QueryModalityPooling(nn.Module):
    """PREM-style modality-specific query pooling.

    The module supports token sequences [B, L, 768]. For global cached query
    features [B, 768], callers can pass a single-token sequence. Even in that
    fallback path, visual/subtitle/joint representations are produced by
    separate learned attention/projection heads and are not the same vector.
    """

    def __init__(self, input_dim: int = 768, hidden_dim: int = 384, dropout: float = 0.08) -> None:
        super().__init__()
        self.visual_attn = nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1))
        self.subtitle_attn = nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1))
        self.joint_attn = nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))
        self.visual_proj = nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.LayerNorm(hidden_dim))
        self.subtitle_proj = nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.LayerNorm(hidden_dim))
        self.joint_proj = nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.LayerNorm(hidden_dim))
        self.gate = nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, 3))
        self.qtype_bias = nn.Embedding(4, 3)
        with torch.no_grad():
            self.qtype_bias.weight.copy_(torch.tensor([
                [1.2, 0.1, 0.5],  # visual query
                [0.1, 1.2, 0.5],  # subtitle query
                [0.8, 0.8, 0.9],  # visual+text query
                [0.5, 0.5, 0.5],
            ]))

    @staticmethod
    def _pool(tokens: torch.Tensor, scorer: nn.Module) -> Tuple[torch.Tensor, torch.Tensor]:
        weights = torch.softmax(scorer(tokens).squeeze(-1), dim=-1)
        pooled = torch.einsum("bl,bld->bd", weights, tokens)
        return pooled, weights

    def forward(self, query_tokens: torch.Tensor, qtype: torch.Tensor) -> Dict[str, torch.Tensor]:
        if query_tokens.dim() == 2:
            query_tokens = query_tokens.unsqueeze(1)
        qv_raw, av = self._pool(query_tokens, self.visual_attn)
        qs_raw, a_s = self._pool(query_tokens, self.subtitle_attn)
        qj_raw, aj = self._pool(query_tokens, self.joint_attn)
        qv = F.normalize(self.visual_proj(qv_raw), dim=-1)
        qs = F.normalize(self.subtitle_proj(qs_raw), dim=-1)
        qj = F.normalize(self.joint_proj(qj_raw), dim=-1)
        gate_source = query_tokens.mean(dim=1)
        gate = torch.softmax(self.gate(gate_source) + self.qtype_bias(qtype.clamp(0, 3)), dim=-1)
        return {
            "q_visual": qv,
            "q_subtitle": qs,
            "q_joint": qj,
            "gate": gate,
            "attn_visual": av,
            "attn_subtitle": a_s,
            "attn_joint": aj,
        }


class SoftTopKPool(nn.Module):
    def __init__(self, temperature: float = 0.07) -> None:
        super().__init__()
        self.log_temperature = nn.Parameter(torch.tensor(float(temperature)).log())

    def forward(self, scores: torch.Tensor) -> torch.Tensor:
        temp = self.log_temperature.exp().clamp(0.02, 1.0)
        w = torch.softmax(scores / temp, dim=-1)
        return torch.sum(w * scores, dim=-1)


class PREMCollaborativeRetriever(nn.Module):
    def __init__(
        self,
        query_dim: int = 768,
        subtitle_dim: int = 768,
        visual_dim: int = 4352,
        hidden_dim: int = 384,
        residual_clip: float = 0.20,
        dropout: float = 0.08,
    ) -> None:
        super().__init__()
        self.query_pool = QueryModalityPooling(query_dim, hidden_dim, dropout=dropout)
        self.visual_proj = nn.Sequential(nn.LayerNorm(visual_dim), nn.Linear(visual_dim, hidden_dim * 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim * 2, hidden_dim), nn.LayerNorm(hidden_dim))
        self.subtitle_proj = nn.Sequential(nn.LayerNorm(subtitle_dim), nn.Linear(subtitle_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.LayerNorm(hidden_dim))
        self.joint_proj = nn.Sequential(nn.LayerNorm(subtitle_dim + visual_dim), nn.Linear(subtitle_dim + visual_dim, hidden_dim * 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim * 2, hidden_dim), nn.LayerNorm(hidden_dim))
        self.residual_head = nn.Sequential(nn.LayerNorm(4), nn.Linear(4, 32), nn.GELU(), nn.Linear(32, 1))
        self.scale = nn.Parameter(torch.tensor(10.0))
        self.residual_clip = float(residual_clip)

    def partial_scores(self, query_tokens: torch.Tensor, qtype: torch.Tensor, visual: torch.Tensor, subtitle: torch.Tensor) -> Dict[str, torch.Tensor]:
        q = self.query_pool(query_tokens, qtype)
        vh = F.normalize(self.visual_proj(visual), dim=-1)
        sh = F.normalize(self.subtitle_proj(subtitle), dim=-1)
        jh = F.normalize(self.joint_proj(torch.cat([subtitle, visual], dim=-1)), dim=-1)
        s_v = torch.sum(q["q_visual"] * vh, dim=-1)
        s_s = torch.sum(q["q_subtitle"] * sh, dim=-1)
        s_j = torch.sum(q["q_joint"] * jh, dim=-1)
        prem = q["gate"][:, 0] * s_v + q["gate"][:, 1] * s_s + q["gate"][:, 2] * s_j
        return {
            **q,
            "visual_relevance": s_v,
            "subtitle_relevance": s_s,
            "joint_relevance": s_j,
            "prem_score": self.scale.clamp(1.0, 30.0) * prem,
        }

    def forward(self, query_tokens: torch.Tensor, qtype: torch.Tensor, visual: torch.Tensor, subtitle: torch.Tensor, first_stage_z: torch.Tensor) -> Dict[str, torch.Tensor]:
        out = self.partial_scores(query_tokens, qtype, visual, subtitle)
        feats = torch.stack([
            out["visual_relevance"],
            out["subtitle_relevance"],
            out["joint_relevance"],
            first_stage_z,
        ], dim=-1)
        raw_residual = self.residual_head(feats).squeeze(-1)
        residual = torch.tanh(raw_residual).clamp(-1.0, 1.0) * self.residual_clip
        video_score = first_stage_z + residual
        out.update({"raw_residual": raw_residual, "residual": residual, "video_score": video_score})
        return out


class FocusThenFuseLocalizer(nn.Module):
    """Lightweight focus-then-fuse span scorer over fixed release-feature summaries."""

    def __init__(self, hidden_dim: int = 384, residual_clip: float = 0.20) -> None:
        super().__init__()
        self.span_head = nn.Sequential(nn.LayerNorm(6), nn.Linear(6, hidden_dim // 4), nn.GELU(), nn.Linear(hidden_dim // 4, 1))
        self.residual_clip = float(residual_clip)

    def forward(
        self,
        visual_rel: torch.Tensor,
        subtitle_rel: torch.Tensor,
        event_aux: torch.Tensor,
        bmn_aux: torch.Tensor,
        t2_aux: torch.Tensor,
        rank_prior: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        feats = torch.stack([visual_rel, subtitle_rel, event_aux, bmn_aux, t2_aux, rank_prior], dim=-1)
        raw = self.span_head(feats).squeeze(-1)
        span_score = torch.tanh(raw).clamp(-1.0, 1.0) * self.residual_clip
        return {"focus_fuse_span_score": span_score, "focus_raw": raw}


def diagnostics_from_batch(out: Dict[str, torch.Tensor]) -> Dict[str, Any]:
    gate = out["gate"].detach().float().cpu()
    residual = out.get("residual", torch.empty(0)).detach().float().cpu()
    return {
        "gate_mean": gate.mean(dim=0).tolist() if gate.numel() else [],
        "gate_std": gate.std(dim=0).tolist() if gate.numel() else [],
        "residual_abs_mean": float(residual.abs().mean().item()) if residual.numel() else None,
        "residual_max_abs": float(residual.abs().max().item()) if residual.numel() else None,
    }
