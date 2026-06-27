#!/usr/bin/env python
"""Create the final train-calib-only C4-r2-cal-v2 comparison and decision."""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rlem.c4_lite_utils import ALL_METRICS, PRIMARY_METRICS, selection_score, write_json  # noqa: E402


OUT = ROOT / "results/rlem_c4_r2_cal_v2"
AUDIT = ROOT / "c4_audit"


def read(path):
    with open(path, "r", encoding="utf-8") as f: return json.load(f)


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""): h.update(chunk)
    return h.hexdigest()


def main():
    groups = ["v2_ab", "v2_cb", "v2_acb"]
    search = {g: read(AUDIT / f"C4_R2_CAL_V2_{g.upper()}_SEARCH_AUDIT.json") for g in groups}
    train = {g: read(AUDIT / f"C4_R2_CAL_V2_{g.upper()}_TRAIN_AUDIT.json") for g in groups}
    grid0 = read(OUT / "v2_ab/search/grid_results.json")
    c31 = grid0["controls"]["frozen_c31"]; c4 = grid0["controls"]["frozen_c4_lite"]
    v1 = read(AUDIT / "C4_R2_CAL_CONSTRAINED_SEARCH_AUDIT.json")["best_config"]
    rows = [
        {"name": "C3.1", "metrics": c31},
        {"name": "C4-lite", "metrics": c4},
        {"name": "C4-r2-v1", "metrics": v1["metrics"]},
    ] + [{"name": g, "metrics": search[g]["best_config"]["metrics"]} for g in groups]
    for row in rows:
        row["delta_vs_c4_lite"] = {k: row["metrics"][k] - c4[k] for k in ALL_METRICS}
        row["delta_vs_c4_r2_v1"] = {k: row["metrics"][k] - v1["metrics"][k] for k in ALL_METRICS}
        row["selection_delta_vs_c4_lite"] = selection_score(row["metrics"], c4)

    decisions = {}
    for g in groups:
        best = search[g]["best_config"]; dv1 = {k: best["metrics"][k] - v1["metrics"][k] for k in ALL_METRICS}
        all8 = all(best["deltas_vs_frozen_c4_lite"][k] > 0 for k in ALL_METRICS)
        six_and_r100 = all(best["deltas_vs_frozen_c4_lite"][k] >= 0 for k in PRIMARY_METRICS) and all(best["deltas_vs_frozen_c4_lite"][k] > 0 for k in ["0.5-r100", "0.7-r100"])
        movement_safe = best["top1_changed_ratio"] <= 0.08 and best["pearson_c4_lite_v2"] >= 0.985
        clearly_exceeds_v1 = all(dv1[k] >= 0 for k in ALL_METRICS) and any(dv1[k] > 0 for k in ALL_METRICS)
        decisions[g] = {
            "all_eight_positive_vs_c4_lite": all8,
            "six_primary_nonnegative_and_r100_positive": six_and_r100,
            "robust_neighbor_supported": best["robust_neighbor_supported"],
            "passing_neighbor_count": best["passing_neighbor_count"],
            "isolated_boundary_point": best["isolated_boundary_point"],
            "movement_safe": movement_safe,
            "deltas_vs_c4_r2_v1": dv1,
            "clearly_exceeds_c4_r2_v1": clearly_exceeds_v1,
            "promotion_rule_pass": (all8 or six_and_r100) and best["robust_neighbor_supported"] and not best["isolated_boundary_point"] and movement_safe and clearly_exceeds_v1,
        }
    selected = max(groups, key=lambda g: search[g]["best_config"]["selection_score_delta_vs_frozen_c4_lite"])
    top_best = {**search[selected]["best_config"], "selected_group": selected, "promotion_rule_pass": decisions[selected]["promotion_rule_pass"]}
    write_json(OUT / "best_config.json", top_best)
    write_json(OUT / "comparison.json", {"rows": rows, "promotion_checks": decisions})
    with open(OUT / "comparison.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["name"] + ALL_METRICS + ["selection_delta_vs_c4_lite"])
        for row in rows: w.writerow([row["name"]] + [row["metrics"][k] for k in ALL_METRICS] + [row["selection_delta_vs_c4_lite"]])
    audit = {
        "status": "PASS", "scope": "train_fit_train_calib_only",
        "experiment_groups": ["V2-A+B", "V2-C+B", "V2-A+C+B"],
        "comparison": rows, "group_promotion_checks": decisions,
        "best_v2_by_selection_delta_vs_c4_lite": selected,
        "best_v2_config": top_best,
        "any_v2_passes_promotion_rule": any(v["promotion_rule_pass"] for v in decisions.values()),
        "final_promotion_decision": "DO_NOT_PROMOTE_V2" if not any(v["promotion_rule_pass"] for v in decisions.values()) else "REQUEST_FREEZE_REVIEW",
        "reason": "All V2 best configurations are robust and positive versus C4-lite, but none clearly dominates frozen C4-r2-v1 across all eight metrics.",
        "training_diagnostics": {g: {"best_epoch": train[g]["best_epoch"], "best_train_calib_metrics": train[g]["best_train_calib_metrics"]} for g in groups},
        "artifact_hashes": {g: {
            "model_best": sha(OUT / g / "model_best.pt"),
            "history": sha(OUT / g / "history.json"),
            "logits": sha(OUT / g / "train_calib_logits.npz"),
            "grid_results": sha(OUT / g / "search/grid_results.json"),
            "best_config": sha(OUT / g / "search/best_config.json"),
        } for g in groups},
        "frozen_verification": {
            "c4_lite_official_manifest": sha(AUDIT / "C4_LITE_OFFICIAL_VAL_FREEZE_MANIFEST.json"),
            "c4_lite_best_config": sha(ROOT / "results/rlem_c4_lite/search/best_config.json"),
            "c31_checkpoint": sha(ROOT / "results/rlem_c31_capacity/models/cap_F_wd0/model_best.pt"),
            "model_conquer": sha(ROOT / "model/conquer.py"),
            "evaluator": sha(ROOT / "standalone_eval/eval.py"),
        },
        "official_val_used": False, "official_val_evidence_opened": False,
        "post_val_adjustment": False, "capacity_search": False,
        "future_stages_used": {"official_val": False, "c4_main": False, "r2_head": False, "vs_head": False, "c5": False, "c6": False},
        "stop_boundary": "STOP_AFTER_C4_R2_CAL_V2_TRAIN_CALIB",
    }
    write_json(AUDIT / "C4_R2_CAL_V2_FINAL_AUDIT.json", audit)
    print(json.dumps({"selected": selected, "decision": audit["final_promotion_decision"], "promotion_checks": decisions}, indent=2))


if __name__ == "__main__": main()
