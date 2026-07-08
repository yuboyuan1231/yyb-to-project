from __future__ import annotations

from typing import Any

import torch

from blueprint_e2e_v2.losses.active_moment_loss import active_moment_diversity_loss, active_moment_relevance_loss
from blueprint_e2e_v2.losses.hard_negative_loss import hard_negative_video_span_loss
from blueprint_e2e_v2.losses.mutual_consistency_loss import localization_to_retrieval_feedback_loss, retrieval_guided_localization_loss, video_span_consistency_loss
from blueprint_e2e_v2.losses.no_regression_loss import no_regression_loss
from blueprint_e2e_v2.losses.partial_relevance_loss import partial_relevance_loss
from blueprint_e2e_v2.losses.region_loss import region_loss
from blueprint_e2e_v2.losses.retrieval_loss import video_retrieval_loss
from blueprint_e2e_v2.losses.span_localization_loss import span_localization_loss
from blueprint_e2e_v2.losses.vcmr_ranking_loss import joint_vcmr_ranking_loss


def compute_full_loss(out: dict[str, Any], batch: dict[str, torch.Tensor], cfg: dict[str, Any]) -> tuple[torch.Tensor, dict[str, float]]:
    retr = out["retr"]
    local = out["local"]
    active = out["active"]
    score = out["score"]
    span_iou = batch["span_iou"].to(score["vcmr_score"].device)
    span_ge07 = batch["span_ge07"].to(score["vcmr_score"].device)
    correct_video = batch["correct_video"].to(score["vcmr_score"].device)
    span_mask = batch.get("span_mask")
    span_mask = span_mask.to(score["vcmr_score"].device) if span_mask is not None else None
    parts = {
        "L_video_retrieval": video_retrieval_loss(retr["retriever_score"], correct_video, cfg.get("front_rank_weight", 10.0)),
        "L_span_localization": span_localization_loss(local["span_score"], span_iou, span_ge07, span_mask),
        "L_joint_vcmr_ranking": joint_vcmr_ranking_loss(score["vcmr_score"], span_iou, span_ge07, correct_video, cfg.get("front_rank_weight", 10.0), span_mask),
        "L_partial_relevance": partial_relevance_loss(local["prem_span"], span_iou, correct_video, span_mask),
        "L_region": region_loss(local["region_span"], span_iou, correct_video, span_mask),
        "L_amd_diversity": active_moment_diversity_loss(active["anchor_centers"], active["anchor_widths"]),
        "L_amd_relevance": active_moment_relevance_loss(local["amd_span"], span_iou, correct_video, span_mask),
        "L_retrieval_guided_localization": retrieval_guided_localization_loss(local["span_score"], retr["retriever_score"], correct_video, span_mask),
        "L_localization_to_retrieval_feedback": localization_to_retrieval_feedback_loss(score["video_final"], local["span_score"], correct_video, span_mask),
        "L_hard_negative_video_span": hard_negative_video_span_loss(score["vcmr_score"], correct_video, span_mask),
        "L_video_span_consistency": video_span_consistency_loss(score["video_final"], local["span_score"], span_mask),
        "L_no_regression": no_regression_loss(score["video_final"], retr["retriever_score"], correct_video),
    }
    weights = {
        "L_span_localization": cfg.get("lambda_span", 2.0),
        "L_joint_vcmr_ranking": cfg.get("lambda_vcmr", 4.0),
        "L_partial_relevance": cfg.get("lambda_pr", 0.5),
        "L_region": cfg.get("lambda_reg", 0.4),
        "L_amd_diversity": cfg.get("lambda_amd_div", 0.2),
        "L_amd_relevance": cfg.get("lambda_amd_rel", 0.5),
        "L_retrieval_guided_localization": cfg.get("lambda_rg", 0.5),
        "L_localization_to_retrieval_feedback": cfg.get("lambda_fb", 0.8),
        "L_hard_negative_video_span": cfg.get("lambda_hn", 1.2),
        "L_video_span_consistency": cfg.get("lambda_cons", 0.5),
        "L_no_regression": cfg.get("lambda_ng", 0.8),
    }
    total = parts["L_video_retrieval"]
    for name, weight in weights.items():
        total = total + float(weight) * parts[name]
    metrics = {k: float(v.detach().cpu().item()) for k, v in parts.items()}
    metrics["L_total"] = float(total.detach().cpu().item())
    return total, metrics
