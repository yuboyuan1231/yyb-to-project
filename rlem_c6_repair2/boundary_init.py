from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict

import torch

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def load_boundary_init(model, checkpoint_path: str) -> Dict[str, int]:
    """Load matching TemporalPriorEncoder + BoundaryAdapter weights.

    R2-large may have a separately trained large boundary-oracle checkpoint.
    If a mismatched checkpoint is accidentally supplied, this function refuses
    to silently half-initialize the boundary path.
    """
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    src = ckpt["model_state"]
    dst = model.state_dict()
    prefixes = ("inp.", "blocks.", "boundary.")
    selected = {}
    skipped = []
    for key, value in src.items():
        if key.startswith(prefixes):
            if key in dst and tuple(dst[key].shape) == tuple(value.shape):
                selected[key] = value
            else:
                skipped.append(key)
    required = [key for key in dst if key.startswith(prefixes)]
    missing = [key for key in required if key not in selected]
    if missing or skipped:
        raise ValueError(
            f"Boundary init incompatible: loaded={len(selected)} missing={len(missing)} skipped={len(skipped)}; "
            f"first_missing={missing[:3]} first_skipped={skipped[:3]}"
        )
    dst.update(selected)
    model.load_state_dict(dst)
    return {"loaded_tensors": len(selected), "missing_tensors": len(missing), "skipped_tensors": len(skipped)}

