from __future__ import annotations

from typing import Any, Dict


def selection_objective(metrics: Dict[str, Any], baseline: Dict[str, Any]) -> float:
    """C26S front-rank-oriented selection score.

    This mirrors the requested objective. It uses calib_select metrics only in
    the runner and never uses diagnostic oracle rows for selection.
    """
    def d(key: str) -> float:
        return float(metrics.get(key, 0.0) or 0.0) - float(baseline.get(key, 0.0) or 0.0)

    wrong_inc = max(0.0, d("wrong_video_high_score_rate"))
    high_fp_inc = max(0.0, d("high_score_false_positive_rate"))
    vr100_drop = max(0.0, float(baseline.get("VR_R@100", 0.0) or 0.0) - float(metrics.get("VR_R@100", 0.0) or 0.0) - 0.5)
    r100_only = max(0.0, d("VCMR_R@100_IoU0.5") - max(0.0, d("VCMR_R@1_IoU0.5") + d("VCMR_R@5_IoU0.5") + d("VCMR_R@10_IoU0.5")))
    return (
        3.5 * d("VCMR_R@1_IoU0.7")
        + 2.5 * d("VCMR_R@5_IoU0.7")
        + 1.5 * d("VCMR_R@10_IoU0.7")
        + 1.0 * d("VCMR_R@1_IoU0.5")
        + 0.8 * d("VCMR_R@5_IoU0.5")
        - 2.0 * wrong_inc
        - 1.2 * vr100_drop
        - 0.8 * high_fp_inc
        - 0.5 * r100_only
    )


def loss_manifest() -> Dict[str, Any]:
    return {
        "L_span_rank": "proposal-level ranking over BMN/T2/PREM span scores",
        "L_span_to_video": "topK span distribution feedback consistency",
        "L_no_regression": "guard against VR_R100 collapse and wrong-video increase",
        "oracle_excluded": True,
    }

