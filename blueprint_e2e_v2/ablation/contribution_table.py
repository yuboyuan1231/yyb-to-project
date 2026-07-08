from __future__ import annotations

from typing import Any


def contribution_summary(results: dict[str, Any]) -> dict[str, Any]:
    def r(name: str, key: str) -> float:
        return float(results.get(name, {}).get("summary", {}).get(key, 0.0))
    return {
        "retrieval_to_localization_helped": r("A5_retrieval_guided_localizer", "VCMR_R@1_IoU0.7") > r("A4_retriever_localizer_no_feedback", "VCMR_R@1_IoU0.7"),
        "localization_to_retrieval_helped": r("A6_localization_to_retrieval_feedback", "VCMR_R@1_IoU0.7") > r("A4_retriever_localizer_no_feedback", "VCMR_R@1_IoU0.7"),
        "full_beats_one_direction": r("A12_full_c28c", "VCMR_R@1_IoU0.7") >= max(r("A5_retrieval_guided_localizer", "VCMR_R@1_IoU0.7"), r("A6_localization_to_retrieval_feedback", "VCMR_R@1_IoU0.7")),
    }

