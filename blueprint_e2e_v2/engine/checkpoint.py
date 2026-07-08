from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def save_checkpoint(path: Path, model: torch.nn.Module, optimizer: torch.optim.Optimizer, epoch: int, metrics: dict[str, Any]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    core = model.module if hasattr(model, "module") else model
    torch.save({"model": core.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch, "metrics": metrics}, path)
    return {"path": str(path), "epoch": epoch, "metrics": metrics, "size_bytes": path.stat().st_size}


def load_checkpoint(path: Path, model: torch.nn.Module, optimizer: torch.optim.Optimizer | None = None) -> dict[str, Any]:
    ckpt = torch.load(path, map_location="cpu")
    core = model.module if hasattr(model, "module") else model
    core.load_state_dict(ckpt["model"])
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt

