from __future__ import annotations

import torch
import torch.nn.functional as F

from rlem_c6_repair.losses_repair import gaussian_targets


def boundary_loss_relevant(batch, out, stability: float = 0.05):
    rel = batch["label_relevant"] > 0.5
    if not bool(rel.any()):
        return out["score"].new_tensor(0.0), {"boundary_ce": 0.0, "boundary_kl": 0.0}
    _, t = batch["p_b"].shape
    target_b = gaussian_targets(t, batch["gt_start_idx"].clamp(0, t - 1), batch["time_mask"])[rel]
    target_e = gaussian_targets(t, batch["gt_end_idx"].clamp(0, t - 1), batch["time_mask"])[rel]
    pb_new, pe_new = out["p_b_new"][rel], out["p_e_new"][rel]
    pb_old, pe_old = batch["p_b"][rel], batch["p_e"][rel]
    mask = batch["time_mask"][rel]
    ce = -(target_b * pb_new.clamp_min(1e-8).log()).sum(1).mean()
    ce = ce - (target_e * pe_new.clamp_min(1e-8).log()).sum(1).mean()
    kl = (pb_new * (pb_new.clamp_min(1e-8).log() - pb_old.clamp_min(1e-8).log())).masked_fill(~mask, 0).sum(1).mean()
    kl = kl + (pe_new * (pe_new.clamp_min(1e-8).log() - pe_old.clamp_min(1e-8).log())).masked_fill(~mask, 0).sum(1).mean()
    return ce + float(stability) * kl, {"boundary_ce": float(ce.detach().cpu()), "boundary_kl": float(kl.detach().cpu())}


def listwise_iou_loss(score, iou, mask):
    target = iou.clamp_min(0).masked_fill(~mask, 0)
    denom = target.sum(1, keepdim=True)
    valid = denom[:, 0] > 1e-6
    if not bool(valid.any()):
        return score.new_tensor(0.0)
    t = target[valid] / denom[valid].clamp_min(1e-6)
    return -(t * F.log_softmax(score[valid].masked_fill(~mask[valid], -1e9), dim=1)).sum(1).mean()


def iou_quality_loss(batch, out):
    mask = batch["span_mask"]
    q = F.smooth_l1_loss(torch.sigmoid(out["quality_logit"][mask]), batch["iou"][mask].clamp(0, 1))
    b05 = F.binary_cross_entropy_with_logits(out["iou05_logit"][mask], batch["y05"][mask])
    b07 = F.binary_cross_entropy_with_logits(out["iou07_logit"][mask], batch["y07"][mask])
    return q + b05 + 0.5 * b07


def query_front_rank_proxy(batch, final_score):
    group_score = final_score.masked_fill(~batch["span_mask"], -1e9).max(dim=1).values
    return F.binary_cross_entropy_with_logits(group_score - batch["s_c4"].masked_fill(~batch["span_mask"], -1e9).max(dim=1).values.detach(), batch["label_relevant"])

