from __future__ import annotations

import torch


def gaussian_targets(length: int, centers: torch.Tensor, mask: torch.Tensor, sigma: float = 1.5) -> torch.Tensor:
    idx = torch.arange(length, device=centers.device)[None, :]
    out = torch.exp(-0.5 * ((idx - centers[:, None].float()) / sigma) ** 2) * mask
    return out / out.sum(1, keepdim=True).clamp_min(1e-8)


def boundary_ce_loss(batch, out, stability: float = 0.1):
    _, t = batch["p_b"].shape
    target_b = gaussian_targets(t, batch["gt_start_idx"].clamp(0, t - 1), batch["time_mask"])
    target_e = gaussian_targets(t, batch["gt_end_idx"].clamp(0, t - 1), batch["time_mask"])
    ce = -(target_b * out["p_b_new"].clamp_min(1e-8).log()).sum(1).mean()
    ce = ce - (target_e * out["p_e_new"].clamp_min(1e-8).log()).sum(1).mean()
    kl_b = (out["p_b_new"] * (out["p_b_new"].clamp_min(1e-8).log() - batch["p_b"].clamp_min(1e-8).log())).masked_fill(~batch["time_mask"], 0).sum(1).mean()
    kl_e = (out["p_e_new"] * (out["p_e_new"].clamp_min(1e-8).log() - batch["p_e"].clamp_min(1e-8).log())).masked_fill(~batch["time_mask"], 0).sum(1).mean()
    return ce + float(stability) * (kl_b + kl_e), {"boundary_ce": float(ce.detach().cpu()), "kl_b": float(kl_b.detach().cpu()), "kl_e": float(kl_e.detach().cpu())}

