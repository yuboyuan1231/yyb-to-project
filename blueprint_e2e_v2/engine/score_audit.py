from __future__ import annotations

import math
from typing import Any

import torch


class ScoreScaleAccumulator:
    def __init__(self, cfg: dict[str, Any]) -> None:
        self.cfg = cfg
        self._stats: dict[str, dict[str, float]] = {}

    def update(self, out: dict[str, Any]) -> None:
        retr = out.get("retr", {})
        combined = retr.get("retriever_score")
        tensors = {
            "pooled_score": retr.get("retriever_score_pooled", combined),
            "late_score": retr.get("retriever_score_late"),
            "token_score": retr.get("retriever_score_token"),
            "combined_score": combined,
        }
        for name, tensor in tensors.items():
            if tensor is None:
                continue
            self._update_tensor(name, tensor)

    def _update_tensor(self, name: str, tensor: torch.Tensor) -> None:
        x = tensor.detach().float()
        finite = torch.isfinite(x)
        if not bool(finite.any()):
            return
        vals = x[finite]
        stat = self._stats.setdefault(
            name,
            {"count": 0.0, "sum": 0.0, "sumsq": 0.0, "min": math.inf, "max": -math.inf},
        )
        stat["count"] += float(vals.numel())
        stat["sum"] += float(vals.sum().item())
        stat["sumsq"] += float((vals * vals).sum().item())
        stat["min"] = min(stat["min"], float(vals.min().item()))
        stat["max"] = max(stat["max"], float(vals.max().item()))

    def summary(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "late_interaction_enabled": bool(self.cfg.get("late_interaction_enabled", False)),
            "pooled_score_weight": float(self.cfg.get("pooled_score_weight", 1.0)),
            "late_score_weight": float(self.cfg.get("late_score_weight", 0.0)),
            "token_maxsim_weight": float(self.cfg.get("token_maxsim_weight", 0.0)),
        }
        out["score_weight_sum"] = out["pooled_score_weight"] + out["late_score_weight"] + out["token_maxsim_weight"]
        for name, stat in self._stats.items():
            count = max(1.0, stat["count"])
            mean = stat["sum"] / count
            var = max(0.0, stat["sumsq"] / count - mean * mean)
            out[f"{name}_count"] = int(stat["count"])
            out[f"{name}_mean"] = mean
            out[f"{name}_std"] = math.sqrt(var)
            out[f"{name}_min"] = stat["min"]
            out[f"{name}_max"] = stat["max"]
        pooled_std = float(out.get("pooled_score_std", 0.0) or 0.0)
        late_std = float(out.get("late_score_std", 0.0) or 0.0)
        combined_std = float(out.get("combined_score_std", 0.0) or 0.0)
        out["late_to_pooled_std_ratio"] = late_std / max(pooled_std, 1e-8)
        out["combined_to_late_std_ratio"] = combined_std / max(late_std, 1e-8)
        out["late_minus_pooled_mean"] = float(out.get("late_score_mean", 0.0) or 0.0) - float(out.get("pooled_score_mean", 0.0) or 0.0)
        return out


def loss_coupling_audit(loss_avg: dict[str, float], score_audit: dict[str, Any]) -> dict[str, float]:
    l_inbatch = float(loss_avg.get("L_inbatch_retrieval", 0.0))
    l_video = float(loss_avg.get("L_video_retrieval", 0.0))
    l_vcmr = float(loss_avg.get("L_joint_vcmr_ranking", 0.0))
    return {
        "L_inbatch_to_video_retrieval": l_inbatch / max(abs(l_video), 1e-8),
        "L_inbatch_to_joint_vcmr": l_inbatch / max(abs(l_vcmr), 1e-8),
        "late_to_pooled_std_ratio": float(score_audit.get("late_to_pooled_std_ratio", 0.0) or 0.0),
        "combined_to_late_std_ratio": float(score_audit.get("combined_to_late_std_ratio", 0.0) or 0.0),
    }
