
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

"""Small C2/C3 evidence heads."""

from typing import Dict

import torch
from torch import nn


class EvidenceMLP(nn.Module):
    """Predict Q_joint, Q_bd and E_fp from compact scalar evidence."""

    def __init__(self, input_dim: int, hidden_dim: int = 128, dropout: float = 0.10):
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.dropout = float(dropout)
        self.net = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
        )
        self.q_joint = nn.Linear(self.hidden_dim, 1)
        self.q_bd = nn.Linear(self.hidden_dim, 1)
        self.e_fp = nn.Linear(self.hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        h = self.net(x)
        return {
            "q_joint_logit": self.q_joint(h).squeeze(-1),
            "q_bd_logit": self.q_bd(h).squeeze(-1),
            "e_fp_logit": self.e_fp(h).squeeze(-1),
        }

    @torch.no_grad()
    def predict_scores(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        out = self.forward(x)
        return {
            "q_joint": torch.sigmoid(out["q_joint_logit"]),
            "q_bd": torch.sigmoid(out["q_bd_logit"]),
            "e_fp": torch.sigmoid(out["e_fp_logit"]),
        }


class FlexibleEvidenceMLP(nn.Module):
    """C3.1 evidence heads with an explicit hidden-dimension sequence."""

    def __init__(self, input_dim: int, hidden_dims, dropout: float = 0.10):
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dims = [int(value) for value in hidden_dims]
        self.dropout = float(dropout)
        if not self.hidden_dims:
            raise ValueError("hidden_dims must contain at least one layer")
        layers = []
        previous = self.input_dim
        for hidden in self.hidden_dims:
            layers.extend([nn.Linear(previous, hidden), nn.LayerNorm(hidden), nn.GELU()])
            if self.dropout > 0:
                layers.append(nn.Dropout(self.dropout))
            previous = hidden
        self.net = nn.Sequential(*layers)
        self.q_joint = nn.Linear(previous, 1)
        self.q_bd = nn.Linear(previous, 1)
        self.e_fp = nn.Linear(previous, 1)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        h = self.net(x)
        return {
            "q_joint_logit": self.q_joint(h).squeeze(-1),
            "q_bd_logit": self.q_bd(h).squeeze(-1),
            "e_fp_logit": self.e_fp(h).squeeze(-1),
        }

    @torch.no_grad()
    def predict_scores(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        out = self.forward(x)
        return {
            "q_joint": torch.sigmoid(out["q_joint_logit"]),
            "q_bd": torch.sigmoid(out["q_bd_logit"]),
            "e_fp": torch.sigmoid(out["e_fp_logit"]),
        }
