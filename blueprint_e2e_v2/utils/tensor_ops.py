from __future__ import annotations

import torch


def masked_mean(x: torch.Tensor, mask: torch.Tensor | None, dim: int) -> torch.Tensor:
    if mask is None:
        return x.mean(dim=dim)
    m = mask.to(dtype=x.dtype)
    while m.dim() < x.dim():
        m = m.unsqueeze(-1)
    return (x * m).sum(dim=dim) / m.sum(dim=dim).clamp_min(1.0)


def sequence_mask(lengths: torch.Tensor, max_len: int) -> torch.Tensor:
    return torch.arange(max_len, device=lengths.device).unsqueeze(0) < lengths.unsqueeze(1)


def span_pool(seq: torch.Tensor, spans: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    """Mean-pool [B,C,T,D] sequence features over [B,C,M,2] inclusive-exclusive clip spans."""
    b, c, t, d = seq.shape
    m = spans.shape[2]
    starts = spans[..., 0].clamp(0, t - 1)
    ends = spans[..., 1].clamp(1, t)
    if mask is None:
        weighted = seq
        denom_prefix = None
    else:
        m_float = mask.to(device=seq.device, dtype=seq.dtype).unsqueeze(-1)
        weighted = seq * m_float
        denom_prefix = torch.cat([seq.new_zeros(b, c, 1, 1), m_float.cumsum(dim=2)], dim=2)
    prefix = torch.cat([seq.new_zeros(b, c, 1, d), weighted.cumsum(dim=2)], dim=2)
    flat_prefix = prefix.reshape(b * c, t + 1, d)
    offset = (torch.arange(b * c, device=seq.device) * (t + 1)).view(b, c, 1)
    flat = flat_prefix.reshape((b * c) * (t + 1), d)
    s_idx = starts + offset
    e_idx = ends + offset
    pooled = flat[e_idx.reshape(-1)] - flat[s_idx.reshape(-1)]
    if denom_prefix is None:
        denom = (ends - starts).clamp_min(1).reshape(-1, 1).to(seq.dtype)
    else:
        denom_flat = denom_prefix.reshape((b * c) * (t + 1), 1)
        denom = denom_flat[e_idx.reshape(-1)] - denom_flat[s_idx.reshape(-1)]
        denom = denom.clamp_min(1.0)
    return (pooled / denom).reshape(b, c, m, d)
