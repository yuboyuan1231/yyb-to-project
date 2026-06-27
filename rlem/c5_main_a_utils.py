#!/usr/bin/env python
"""Utilities for C5-main-inference-A role-specific endpoint injection.

The module is inference-only. It never edits CONQUER or regenerates candidates;
it evaluates frozen begin/end/context distributions at the exact endpoints of
the already fixed candidate spans.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, List, Tuple

import numpy as np
import torch

from rlem.c5_prior_utils import EPS, sha256


ALPHAS = (0.0, 0.25, 0.50)
BETAS = (0.0, 0.05, 0.10)
TAUS = (1.0, 1.5, 2.0)
LAMBDAS = (0.025, 0.05, 0.075, 0.10)
GAMMAS = (0.025, 0.05, 0.075, 0.10)
G_MIN = 0.25
G_MAX = 1.75
CLIP_LENGTH = 1.5


def setting_id(alpha: float, beta: float, tau: float, lambda_: float | None = None) -> str:
    text = f"a{alpha:.2f}_b{beta:.2f}_t{tau:.2f}"
    if lambda_ is not None:
        text += f"_l{lambda_:.3f}"
    return text.replace(".", "p")


def config_id(index: int) -> str:
    return f"c5ma_{index:04d}"


def load_npz_arrays(path: str, names: Iterable[str]) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as z:
        missing = [name for name in names if name not in z.files]
        if missing:
            raise KeyError(f"{path} missing arrays: {missing}")
        return {name: z[name] for name in names}


def load_c4_final_scores(path: str, expected_rows: int) -> np.ndarray:
    with np.load(path, allow_pickle=False) as z:
        if "s_v21" not in z.files:
            raise KeyError(f"{path} lacks s_v21")
        score = z["s_v21"].astype(np.float32)
    if score.shape != (expected_rows,) or not np.all(np.isfinite(score)):
        raise ValueError("Invalid C4_final score array")
    return score


def exact_candidate_indices(
    start_time: np.ndarray,
    end_time: np.ndarray,
    row_group_id: np.ndarray,
    temporal_length: np.ndarray,
    clip_length: float = CLIP_LENGTH,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    """Recover exact fixed-candidate endpoint indices from grid-aligned times."""
    st_units = np.asarray(start_time, dtype=np.float64) / float(clip_length)
    en_units = np.asarray(end_time, dtype=np.float64) / float(clip_length)
    st_round, en_round = np.rint(st_units), np.rint(en_units)
    st_err = float(np.max(np.abs(st_units - st_round)))
    en_err = float(np.max(np.abs(en_units - en_round)))
    if st_err > 1e-6 or en_err > 1e-6:
        raise ValueError(f"Candidate times are not exact {clip_length}s grid multiples: {st_err}, {en_err}")
    start_idx = st_round.astype(np.int64)
    end_idx = en_round.astype(np.int64) - 1
    gid = np.asarray(row_group_id, dtype=np.int64)
    max_valid = np.asarray(temporal_length, dtype=np.int64)[gid] - 1
    if np.any(start_idx < 0) or np.any(end_idx < start_idx):
        raise ValueError("Invalid candidate endpoint indices")
    # A candidate ending exactly at the padded video boundary is clipped to the
    # last real temporal token; count this explicitly rather than silently.
    clipped_start = int(np.sum(start_idx > max_valid))
    clipped_end = int(np.sum(end_idx > max_valid))
    start_idx = np.minimum(start_idx, max_valid)
    end_idx = np.minimum(np.maximum(end_idx, start_idx), max_valid)
    return start_idx.astype(np.int16), end_idx.astype(np.int16), {
        "clip_length": float(clip_length),
        "start_grid_max_abs_error": st_err,
        "end_grid_max_abs_error": en_err,
        "start_index_clipped_count": clipped_start,
        "end_index_clipped_count": clipped_end,
    }


def group_max(values: np.ndarray, row_group_id: np.ndarray, group_count: int) -> np.ndarray:
    out = np.full(group_count, -np.inf, dtype=np.float32)
    np.maximum.at(out, np.asarray(row_group_id, dtype=np.int64), np.asarray(values, dtype=np.float32))
    if not np.all(np.isfinite(out)):
        raise ValueError("Non-finite or empty group retrieval score")
    return out


def fit_retrieval_stats(group_score: np.ndarray) -> Dict[str, float]:
    x = np.asarray(group_score, dtype=np.float64)
    return {"mean": float(np.mean(x)), "std": float(max(np.std(x), EPS)), "count": int(len(x))}


def retrieval_z(group_score: np.ndarray, stats: Dict[str, float]) -> np.ndarray:
    return ((np.asarray(group_score, dtype=np.float32) - np.float32(stats["mean"])) / np.float32(stats["std"])).astype(np.float32)


def retrieval_gate(z: np.ndarray, tau: float) -> np.ndarray:
    """Symmetric gate centered at one; clip bounds are both reachable."""
    x = np.asarray(z, dtype=np.float32) / np.float32(tau)
    gate = 2.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))
    return np.clip(gate, G_MIN, G_MAX).astype(np.float32)


@dataclass
class EndpointPriorEngine:
    log_b: torch.Tensor
    log_e: torch.Tensor
    log_ctx: torch.Tensor
    temporal_length: torch.Tensor
    valid: torch.Tensor
    device: torch.device

    @classmethod
    def load(cls, temporal_prior_npz: str, device: str = "cuda") -> "EndpointPriorEngine":
        arrays = load_npz_arrays(temporal_prior_npz, ("p_b", "p_e", "p_ctx", "temporal_length"))
        if arrays["p_b"].shape != arrays["p_e"].shape or arrays["p_b"].shape != arrays["p_ctx"].shape:
            raise ValueError("Temporal prior shapes differ")
        dev = torch.device("cuda" if device == "cuda" and torch.cuda.is_available() else "cpu")
        pb = torch.from_numpy(arrays["p_b"].astype(np.float32, copy=False)).to(dev)
        pe = torch.from_numpy(arrays["p_e"].astype(np.float32, copy=False)).to(dev)
        pc = torch.from_numpy(arrays["p_ctx"].astype(np.float32, copy=False)).to(dev)
        length = torch.from_numpy(arrays["temporal_length"].astype(np.int64, copy=False)).to(dev)
        positions = torch.arange(pb.shape[1], device=dev)[None, :]
        valid = positions < length[:, None]
        for name, arr in (("p_b", pb), ("p_e", pe), ("p_ctx", pc)):
            if not bool(torch.isfinite(arr).all()) or bool((arr < 0).any()):
                raise ValueError(f"Invalid {name}")
        # Cache logs once: all nine alpha/beta settings reuse them.
        log_b = torch.log(torch.clamp(pb, min=EPS))
        log_e = torch.log(torch.clamp(pe, min=EPS))
        log_ctx = torch.log(torch.clamp(pc, min=EPS))
        del pb, pe, pc
        return cls(log_b, log_e, log_ctx, length, valid, dev)

    @property
    def group_count(self) -> int:
        return int(self.log_b.shape[0])

    def role_log_priors(self, alpha: float, beta: float) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
        # Clamp only for logs. Padded locations are excluded from logsumexp and
        # can never be gathered because candidate indices are length-clipped.
        raw_s = (1.0 + float(alpha)) * self.log_b + float(beta) * self.log_ctx
        raw_e = (1.0 + float(alpha)) * self.log_e + float(beta) * self.log_ctx
        neg_inf = torch.tensor(-torch.inf, device=self.device, dtype=torch.float32)
        raw_s = torch.where(self.valid, raw_s, neg_inf)
        raw_e = torch.where(self.valid, raw_e, neg_inf)
        log_s = raw_s - torch.logsumexp(raw_s, dim=1, keepdim=True)
        log_e = raw_e - torch.logsumexp(raw_e, dim=1, keepdim=True)
        ps = torch.exp(log_s.masked_fill(~self.valid, -torch.inf))
        pe = torch.exp(log_e.masked_fill(~self.valid, -torch.inf))
        entropy_s = -(ps * log_s.masked_fill(~self.valid, 0.0)).sum(dim=1)
        entropy_e = -(pe * log_e.masked_fill(~self.valid, 0.0)).sum(dim=1)
        sanity = {
            "p_s_sum_max_abs_error": float(torch.max(torch.abs(ps.sum(dim=1) - 1.0)).item()),
            "p_e_sum_max_abs_error": float(torch.max(torch.abs(pe.sum(dim=1) - 1.0)).item()),
            "p_s_peak_mean": float(ps.max(dim=1).values.mean().item()),
            "p_e_peak_mean": float(pe.max(dim=1).values.mean().item()),
            "p_s_entropy_mean": float(entropy_s.mean().item()),
            "p_e_entropy_mean": float(entropy_e.mean().item()),
            "nonfinite_valid_values": int((~torch.isfinite(log_s[self.valid])).sum().item() + (~torch.isfinite(log_e[self.valid])).sum().item()),
        }
        return log_s, log_e, sanity

    def endpoint_values(
        self,
        log_s: torch.Tensor,
        log_e: torch.Tensor,
        row_group_id: np.ndarray,
        start_idx: np.ndarray,
        end_idx: np.ndarray,
        *,
        chunk_rows: int = 1_000_000,
    ) -> np.ndarray:
        rows = len(row_group_id)
        out = np.empty(rows, dtype=np.float32)
        for start in range(0, rows, chunk_rows):
            end = min(start + chunk_rows, rows)
            gid = torch.from_numpy(np.asarray(row_group_id[start:end], dtype=np.int64)).to(self.device)
            si = torch.from_numpy(np.asarray(start_idx[start:end], dtype=np.int64)).to(self.device)
            ei = torch.from_numpy(np.asarray(end_idx[start:end], dtype=np.int64)).to(self.device)
            value = log_s[gid, si] + log_e[gid, ei]
            out[start:end] = value.detach().cpu().numpy().astype(np.float32, copy=False)
        if not np.all(np.isfinite(out)):
            raise ValueError("Non-finite endpoint log-prior values")
        return out

    def injected_endpoint_delta(
        self,
        log_s: torch.Tensor,
        log_e: torch.Tensor,
        lambda_eff_group: np.ndarray,
        row_group_id: np.ndarray,
        start_idx: np.ndarray,
        end_idx: np.ndarray,
        *,
        chunk_rows: int = 1_000_000,
    ) -> np.ndarray:
        """Gauge-invariant endpoint log-probability change.

        Frozen exports contain probabilities, not identifiable raw-logit
        constants. Therefore inference must normalize the injected posterior
        within each video before comparing endpoint scores:
          new P_b ∝ P_b * P_s_prior^lambda_eff
          new P_e ∝ P_e * P_e_prior^lambda_eff.
        """
        leff = torch.from_numpy(np.asarray(lambda_eff_group, dtype=np.float32)).to(self.device)
        base_b = torch.where(self.valid, self.log_b, torch.tensor(-torch.inf, device=self.device))
        base_e = torch.where(self.valid, self.log_e, torch.tensor(-torch.inf, device=self.device))
        base_b = base_b - torch.logsumexp(base_b, dim=1, keepdim=True)
        base_e = base_e - torch.logsumexp(base_e, dim=1, keepdim=True)
        new_b = base_b + leff[:, None] * log_s
        new_e = base_e + leff[:, None] * log_e
        new_b = torch.where(self.valid, new_b, torch.tensor(-torch.inf, device=self.device))
        new_e = torch.where(self.valid, new_e, torch.tensor(-torch.inf, device=self.device))
        new_b = new_b - torch.logsumexp(new_b, dim=1, keepdim=True)
        new_e = new_e - torch.logsumexp(new_e, dim=1, keepdim=True)
        ratio_b, ratio_e = new_b - base_b, new_e - base_e
        rows = len(row_group_id)
        out = np.empty(rows, dtype=np.float32)
        for start in range(0, rows, chunk_rows):
            end = min(start + chunk_rows, rows)
            gid = torch.from_numpy(np.asarray(row_group_id[start:end], dtype=np.int64)).to(self.device)
            si = torch.from_numpy(np.asarray(start_idx[start:end], dtype=np.int64)).to(self.device)
            ei = torch.from_numpy(np.asarray(end_idx[start:end], dtype=np.int64)).to(self.device)
            value = ratio_b[gid, si] + ratio_e[gid, ei]
            out[start:end] = value.detach().cpu().numpy().astype(np.float32, copy=False)
        if not np.all(np.isfinite(out)):
            raise ValueError("Non-finite injected endpoint Delta_bd")
        return out


def fit_delta_stats(delta_values: np.ndarray) -> Dict[str, float]:
    delta = np.asarray(delta_values, dtype=np.float64)
    return {
        "mean": float(np.mean(delta)),
        "std": float(max(np.std(delta), EPS)),
        "min": float(np.min(delta)),
        "max": float(np.max(delta)),
        "count": int(len(delta)),
        "nonfinite": int((~np.isfinite(delta)).sum()),
    }


def standardized_delta(
    delta_values: np.ndarray,
    stats: Dict[str, float],
) -> np.ndarray:
    delta = np.asarray(delta_values, dtype=np.float32)
    z = (delta - np.float32(stats["mean"])) / np.float32(stats["std"])
    if not np.all(np.isfinite(z)):
        raise ValueError("Non-finite standardized Delta_bd")
    return z.astype(np.float32)


def iter_effective_configs() -> Iterator[Dict[str, float]]:
    idx = 0
    yield {
        "config_id": config_id(idx), "family": "zero_control", "gamma": 0.0,
        "alpha": 0.0, "beta": 0.0, "tau": 1.0, "lambda": 0.0,
    }
    idx += 1
    for alpha in ALPHAS:
        for beta in BETAS:
            for tau in TAUS:
                for lambda_ in LAMBDAS:
                    for gamma in GAMMAS:
                        yield {
                            "config_id": config_id(idx), "family": "role_endpoint_injection",
                            "alpha": float(alpha), "beta": float(beta), "tau": float(tau),
                            "lambda": float(lambda_), "gamma": float(gamma),
                        }
                        idx += 1


def artifact(path: str) -> Dict[str, str]:
    return {"path": str(path), "sha256": sha256(str(path))}
