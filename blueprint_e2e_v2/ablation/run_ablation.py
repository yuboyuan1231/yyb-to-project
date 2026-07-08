from __future__ import annotations

from typing import Any

from blueprint_e2e_v2.ablation.contribution_table import contribution_summary


def run_post_performance_ablation(full_eval: dict[str, Any]) -> dict[str, Any]:
    # The concrete inference toggles are staged here so full-model performance
    # remains the route owner. The first implementation records the required
    # ablation contract and full-model anchor; retrained/toggled ablations can
    # be launched after the first complete C28C run.
    results = {
        "A12_full_c28c": full_eval.get("summary", {}),
        "ablation_mode": "post_full_model_contract_ready",
        "full_model_remains_final_route": True,
    }
    return {"status": "C28C_ABLATION_INCONCLUSIVE", "results": results, "contribution": contribution_summary({})}

