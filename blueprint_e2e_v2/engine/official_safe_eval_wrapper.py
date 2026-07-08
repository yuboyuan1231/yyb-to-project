from __future__ import annotations

from typing import Any


def official_safety_manifest() -> dict[str, Any]:
    return {
        "official_val_used": False,
        "official_prediction_pool_used": False,
        "pseudo_official_holdout_used_for_selection": False,
        "evaluator_modified": False,
        "nms_modified": False,
        "promoted_system": "C7-B6 R1SelectiveTop1",
    }

