from __future__ import annotations

import torch
import torch.nn.functional as F


def video_retrieval_loss(video_score: torch.Tensor, correct_video: torch.Tensor, front_rank_weight: float = 10.0) -> torch.Tensor:
    target = correct_video.float()
    target_sum = target.sum(dim=1, keepdim=True)
    valid = target_sum.squeeze(1) > 0
    if not bool(valid.any()):
        return video_score.sum() * 0.0
    video_score = video_score[valid]
    correct_video = correct_video[valid]
    target = target[valid] / target_sum[valid].clamp_min(1.0)
    logp = F.log_softmax(video_score, dim=1)
    ce = -(target * logp).sum(dim=1).mean()
    pos = (video_score * target).sum(dim=1)
    has_neg = (~correct_video.bool()).any(dim=1)
    if bool(has_neg.any()):
        neg = video_score[has_neg].masked_fill(correct_video[has_neg].bool(), -1e4).max(dim=1).values
        margin = F.relu(1.0 - pos[has_neg] + neg).mean()
    else:
        margin = video_score.sum() * 0.0
    return ce + float(front_rank_weight) * 0.05 * margin


def inbatch_video_retrieval_loss(
    q: dict[str, torch.Tensor],
    enc: dict[str, torch.Tensor],
    correct_video: torch.Tensor,
    video_indices: torch.Tensor | None = None,
) -> torch.Tensor:
    valid = correct_video.bool().any(dim=1)
    if int(valid.sum().item()) < 2:
        return enc["joint_pool"].sum() * 0.0
    valid_idx = torch.nonzero(valid, as_tuple=False).squeeze(1)
    pos_idx = correct_video[valid].float().argmax(dim=1)
    visual_pos = enc["visual_pool"][valid_idx, pos_idx]
    subtitle_pos = enc["subtitle_pool"][valid_idx, pos_idx]
    joint_pos = enc["joint_pool"][valid_idx, pos_idx]
    qv = q["q_visual"][valid]
    qs = q["q_subtitle"][valid]
    qj = q["q_joint"][valid]
    gate = q["gate"][valid]
    sv = qv @ visual_pos.T
    ss = qs @ subtitle_pos.T
    sj = qj @ joint_pos.T
    scores = 10.0 * (gate[:, 0:1] * sv + gate[:, 1:2] * ss + gate[:, 2:3] * sj)
    if video_indices is None:
        target = torch.eye(pos_idx.shape[0], device=scores.device, dtype=scores.dtype)
    else:
        pos_video = video_indices.to(device=pos_idx.device)[valid, pos_idx]
        target = pos_video[:, None].eq(pos_video[None, :]).to(dtype=scores.dtype)
    target = target / target.sum(dim=1, keepdim=True).clamp_min(1.0)
    logp = F.log_softmax(scores, dim=1)
    return -(target * logp).sum(dim=1).mean()
