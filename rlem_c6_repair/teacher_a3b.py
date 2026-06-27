from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from rlem.c5_main_a_utils import (  # noqa: E402
    EndpointPriorEngine,
    group_max,
    retrieval_gate,
    retrieval_z,
    setting_id,
    standardized_delta,
)


A3B_CONFIG = {
    "config_id": "c5ma_0214",
    "alpha": 0.25,
    "beta": 0.05,
    "tau": 1.5,
    "lambda": 0.05,
    "gamma": 0.05,
}


def load_stats(stats_json: str) -> Dict:
    import json
    return json.loads(Path(stats_json).read_text(encoding="utf-8"))


def teacher_setting_key() -> str:
    return setting_id(A3B_CONFIG["alpha"], A3B_CONFIG["beta"], A3B_CONFIG["tau"], A3B_CONFIG["lambda"])


def compute_a3b_teacher(
    *,
    cache_npz: str,
    temporal_npz: str,
    stats_json: str,
    device: str = "cuda",
    chunk_rows: int = 500_000,
) -> Dict[str, np.ndarray | Dict]:
    """Compute frozen A3b-wide teacher endpoint posterior and row delta.

    Returns raw endpoint delta, train_fit-standardized delta, and score residual
    gamma*z_delta.  All stats are loaded from train_fit-only C5 A2/A3 freeze
    material; no official-val statistics are estimated here.
    """
    cache = np.load(cache_npz, allow_pickle=True)
    row_gid = cache["row_group_id"].astype(np.int64)
    group_count = int(row_gid.max()) + 1
    base = cache["s_c4_final"].astype(np.float32)
    stats = load_stats(stats_json)
    group_score = group_max(base, row_gid, group_count)
    rz = retrieval_z(group_score, stats["retrieval_score_stats"])
    gate = retrieval_gate(rz, A3B_CONFIG["tau"])
    eng = EndpointPriorEngine.load(temporal_npz, device)
    log_s, log_e, sanity = eng.role_log_priors(A3B_CONFIG["alpha"], A3B_CONFIG["beta"])
    raw_delta = eng.injected_endpoint_delta(
        log_s,
        log_e,
        np.float32(A3B_CONFIG["lambda"]) * gate,
        row_gid,
        cache["start_idx"].astype(np.int64),
        cache["end_idx"].astype(np.int64),
        chunk_rows=chunk_rows,
    )
    key = teacher_setting_key()
    z = standardized_delta(raw_delta, stats["delta_stats"][key])
    score_residual = np.float32(A3B_CONFIG["gamma"]) * z
    # Dense teacher distributions for boundary CE diagnostics/training.
    dev = eng.device
    leff = torch.from_numpy((np.float32(A3B_CONFIG["lambda"]) * gate).astype(np.float32)).to(dev)
    valid = eng.valid
    base_b = torch.where(valid, eng.log_b, torch.tensor(-torch.inf, device=dev))
    base_e = torch.where(valid, eng.log_e, torch.tensor(-torch.inf, device=dev))
    base_b = base_b - torch.logsumexp(base_b, dim=1, keepdim=True)
    base_e = base_e - torch.logsumexp(base_e, dim=1, keepdim=True)
    new_b = torch.where(valid, base_b + leff[:, None] * log_s, torch.tensor(-torch.inf, device=dev))
    new_e = torch.where(valid, base_e + leff[:, None] * log_e, torch.tensor(-torch.inf, device=dev))
    new_b = new_b - torch.logsumexp(new_b, dim=1, keepdim=True)
    new_e = new_e - torch.logsumexp(new_e, dim=1, keepdim=True)
    p_b_teacher = torch.exp(new_b.masked_fill(~valid, -torch.inf)).detach().cpu().numpy().astype(np.float32)
    p_e_teacher = torch.exp(new_e.masked_fill(~valid, -torch.inf)).detach().cpu().numpy().astype(np.float32)
    return {
        "raw_delta": raw_delta.astype(np.float32),
        "z_delta": z.astype(np.float32),
        "score_residual": score_residual.astype(np.float32),
        "p_b_teacher": p_b_teacher,
        "p_e_teacher": p_e_teacher,
        "sanity": sanity,
        "config": dict(A3B_CONFIG),
        "stats_source": stats.get("stats_source"),
    }

