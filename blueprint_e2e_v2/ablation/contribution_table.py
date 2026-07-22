from __future__ import annotations

from typing import Any


def contribution_summary(results: dict[str, Any]) -> dict[str, Any]:
    required = ["A4_retriever_localizer_no_feedback", "A5_retrieval_guided_localizer", "A6_localization_to_retrieval_feedback", "A12_full_c28c"]
    missing = [name for name in required if name not in results]
    if missing:
        return {
            "sufficient_ablation_evidence": False,
            "claimed_contributions": False,
            "missing_ablation_results": missing,
            "retrieval_to_localization_helped": None,
            "localization_to_retrieval_helped": None,
            "full_beats_one_direction": None,
        }

    def r(name: str, key: str) -> float:
        return float(results.get(name, {}).get("summary", {}).get(key, 0.0))
    return {
        "sufficient_ablation_evidence": True,
        "claimed_contributions": True,
        "missing_ablation_results": [],
        "retrieval_to_localization_helped": r("A5_retrieval_guided_localizer", "VCMR_R@1_IoU0.7") > r("A4_retriever_localizer_no_feedback", "VCMR_R@1_IoU0.7"),
        "localization_to_retrieval_helped": r("A6_localization_to_retrieval_feedback", "VCMR_R@1_IoU0.7") > r("A4_retriever_localizer_no_feedback", "VCMR_R@1_IoU0.7"),
        "full_beats_one_direction": r("A12_full_c28c", "VCMR_R@1_IoU0.7") >= max(r("A5_retrieval_guided_localizer", "VCMR_R@1_IoU0.7"), r("A6_localization_to_retrieval_feedback", "VCMR_R@1_IoU0.7")),
    }
