#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rlem_c7.run_c7_b0_b1 import atomic_json, atomic_text, load_npz, metric_delta  # noqa: E402
from rlem_c7.run_c7_b5_sn_scoring import (  # noqa: E402
    METRIC_KEYS,
    build_queries,
    eval_system,
    md,
    normalize_safe,
    sha256_file,
    split_indices,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--cache", default="results/rlem_c6a/cache/train_calib_c6_cache.npz")
    p.add_argument("--anchor_states", default="results/rlem_c7_b2/train_calib_c7_b1_anchor_states.json")
    p.add_argument("--video_scores", default="results/rlem_c7_b2/train_calib_video_scores.npz")
    p.add_argument("--b2_eval", default="c7_audit/C7_B2_TRAIN_CALIB_EVAL.json")
    p.add_argument("--b4_decision", default="c7_audit/C7_B4_SELECTION_DECISION.json")
    p.add_argument("--fixed_pool_logits_npz", default="results/rlem_c7_b5_1/train_calib_fixed_pool_raw_logits.npz")
    p.add_argument("--export_audit", default="c7_audit/C7_B5_1_FIXED_POOL_RAW_LOGIT_EXPORT.json")
    p.add_argument("--audit_dir", default="c7_audit")
    p.add_argument("--effective_top_n", type=int, default=100)
    p.add_argument("--max_after_nms", type=int, default=100)
    p.add_argument("--nms_thd", type=float, default=0.7)
    return p.parse_args()


def validate_fixed_pool_logits(queries: List[Dict[str, Any]], logits: Dict[str, np.ndarray]) -> Dict[str, Any]:
    total = len(queries) * 100
    mismatches: List[Dict[str, Any]] = []
    mismatch_count = 0
    finite_keys = ["b_start_logit", "e_end_logit", "p_b_i", "p_e_j", "r1"]
    nan_counts = {k: int(np.isnan(logits[k]).sum()) for k in finite_keys}
    for qd in queries:
        q = int(qd["q"])
        for j, c in enumerate(qd["anchor"][:100]):
            i = q * 100 + j
            ok = (
                int(logits["query_index"][i]) == q
                and int(logits["group_id"][i]) == int(c[0])
                and int(logits["video_idx"][i]) == int(c[1])
                and int(logits["start_idx"][i]) == int(c[2])
                and int(logits["end_idx"][i]) == int(c[3])
                and int(logits["candidate_row_id"][i]) == int(c[5])
                and int(logits["fixed_pool_rank"][i]) == j + 1
            )
            if not ok:
                mismatch_count += 1
                if len(mismatches) < 20:
                    mismatches.append({
                        "q": q,
                        "rank": j + 1,
                        "expected": [int(c[0]), int(c[1]), int(c[2]), int(c[3]), int(c[5])],
                        "actual": [
                            int(logits["group_id"][i]), int(logits["video_idx"][i]), int(logits["start_idx"][i]),
                            int(logits["end_idx"][i]), int(logits["candidate_row_id"][i]),
                        ],
                    })
    strict = (
        int(logits["query_index"].shape[0]) == total
        and mismatch_count == 0
        and all(v == 0 for v in nan_counts.values())
        and int((logits["video_slot"] < 0).sum()) == 0
    )
    return {
        "status": "C7_B5_1_STRICT_ALIGNMENT_PASS" if strict else "C7_B5_1_STRICT_ALIGNMENT_FAIL",
        "official_val_used": False,
        "total_candidate_keys": int(total),
        "matched_candidate_keys": int(total - mismatch_count) if all(v == 0 for v in nan_counts.values()) else int(total - sum(nan_counts.values())),
        "missing_candidate_keys": 0 if strict else None,
        "identity_mismatch_count": int(mismatch_count),
        "identity_mismatch_examples": mismatches,
        "nan_counts": nan_counts,
        "negative_video_slot_count": int((logits["video_slot"] < 0).sum()),
        "position_based_join": False,
        "strict_alignment": bool(strict),
        "candidate_identity_set_matches_C7_B2_1_A4_fixed_pool": bool(strict),
    }


def main() -> None:
    global args
    args = parse_args()
    audit = Path(args.audit_dir)
    audit.mkdir(parents=True, exist_ok=True)
    export_audit = json.loads(Path(args.export_audit).read_text(encoding="utf-8"))
    data = build_queries(args)
    queries = data["queries"]
    logits = dict(load_npz(args.fixed_pool_logits_npz, allow_pickle=False))
    alignment = validate_fixed_pool_logits(queries, logits)

    inventory = {
        "status": "C7_B5_1_RAW_LOGITS_AVAILABLE" if alignment["strict_alignment"] else "C7_B5_1_RAW_LOGITS_INVALID",
        "official_val_used": False,
        "source": args.fixed_pool_logits_npz,
        "source_sha256": sha256_file(args.fixed_pool_logits_npz),
        "raw_start_logits": True,
        "raw_end_logits": True,
        "post_softmax_start_prob": True,
        "post_softmax_end_prob": True,
        "video_score_r1_available": True,
        "fixed_pool_export_status": export_audit.get("status"),
        "export_matched_candidates": export_audit.get("matched_candidates"),
        "export_missing_candidates": export_audit.get("missing_candidates"),
        "tensor_shape": {k: list(v.shape) for k, v in logits.items()},
    }
    atomic_json(audit / "C7_B5_1_RAW_LOGIT_SOURCE_INVENTORY.json", inventory)
    atomic_text(audit / "C7_B5_1_RAW_LOGIT_SOURCE_INVENTORY.md", md("C7-B5.1 raw logit source inventory", inventory))
    atomic_json(audit / "C7_B5_1_STRICT_ALIGNMENT_AUDIT.json", alignment)
    atomic_text(audit / "C7_B5_1_STRICT_ALIGNMENT_AUDIT.md", md("C7-B5.1 strict alignment audit", alignment))

    construction = {
        "status": "C7_B5_1_SN_SCORE_CONSTRUCTION_COMPLETE" if alignment["strict_alignment"] else "C7_B5_1_SN_ALIGNMENT_FAIL",
        "official_val_used": False,
        "strict_alignment_required": True,
        "strict_alignment": alignment["strict_alignment"],
        "S_MINUTE_like": "log(r1) + b_start_logit + e_end_logit",
        "S_logprob_like": "log(r1) + log(p_b_i) + log(p_e_j); diagnostic only",
        "fixed_pool_only": True,
        "candidate_added": False,
        "candidate_removed": False,
        "NMS_modified": False,
        "evaluator_modified": False,
    }
    atomic_json(audit / "C7_B5_1_SN_SCORE_CONSTRUCTION.json", construction)
    atomic_text(audit / "C7_B5_1_SN_SCORE_CONSTRUCTION.md", md("C7-B5.1 SN score construction", construction))
    if not alignment["strict_alignment"]:
        final = {
            "status": "C7_B5_1_INFRA_FAIL",
            "official_val_used": False,
            "post_val_adjustment": False,
            "second_official_val": False,
            "selected_candidate": None,
            "reason": "fixed-pool raw-logit cache failed strict identity alignment",
        }
        atomic_json(audit / "C7_B5_1_FINAL_DECISION.json", final)
        atomic_text(audit / "C7_B5_1_FINAL_DECISION.md", md("C7-B5.1 final decision", final))
        print(json.dumps(final, indent=2, ensure_ascii=False))
        return

    def sl(qd: Dict[str, Any]) -> slice:
        q = int(qd["q"])
        return slice(q * 100, q * 100 + len(qd["anchor"]))

    def anchor_fn(qd: Dict[str, Any]) -> np.ndarray:
        return qd["base"]

    def baseline_fn(qd: Dict[str, Any]) -> np.ndarray:
        return qd["base"] + qd["residual"]

    def b3_fn(qd: Dict[str, Any]) -> np.ndarray:
        return qd["base"] + qd["z"]

    b4 = json.loads(Path(args.b4_decision).read_text(encoding="utf-8"))
    b4_cfg = b4.get("selected_candidate", {}).get("config", {"alpha": 0.2, "normalization": "bounded_minmax", "clip": 1.0})

    def b4_fn(qd: Dict[str, Any]) -> np.ndarray:
        return qd["base"] + float(b4_cfg["alpha"]) * normalize_safe(qd["residual"])

    def sn_raw(qd: Dict[str, Any]) -> np.ndarray:
        x = sl(qd)
        return (np.log(np.maximum(logits["r1"][x].astype(np.float32), 1e-12)) + logits["b_start_logit"][x] + logits["e_end_logit"][x]).astype(np.float32)

    def logprob(qd: Dict[str, Any]) -> np.ndarray:
        x = sl(qd)
        return (
            np.log(np.maximum(logits["r1"][x].astype(np.float32), 1e-12))
            + np.log(np.maximum(logits["p_b_i"][x], 1e-12))
            + np.log(np.maximum(logits["p_e_j"][x], 1e-12))
        ).astype(np.float32)

    systems: Dict[str, Callable[[Dict[str, Any]], np.ndarray]] = {
        "C7_B1_anchor": anchor_fn,
        "C7_B2_1_A4": baseline_fn,
        "C7_B3_calibrated_A4_diagnostic": b3_fn,
        "C7_B4_best_anchor_safe_diagnostic": b4_fn,
        "S_MINUTE_like": sn_raw,
        "S_logprob_like_diagnostic": logprob,
    }
    for beta in [0.02, 0.05, 0.1]:
        systems[f"S_MINUTE_like_plus_weak_residual_beta_{beta}"] = lambda qd, beta=beta: sn_raw(qd) + beta * normalize_safe(qd["residual"])

    splits = split_indices(queries)
    split_results: Dict[str, Dict[str, Any]] = {}
    for split_name, idx in splits.items():
        split_results[split_name] = {}
        for name, fn in systems.items():
            split_results[split_name][name] = eval_system(queries, idx, fn, args, name, baseline_fn=baseline_fn, sn_fn=sn_raw)
        base_metrics = split_results[split_name]["C7_B2_1_A4"]["metrics"]
        for name in split_results[split_name]:
            split_results[split_name][name]["delta_vs_C7_B2_1_A4"] = metric_delta(split_results[split_name][name]["metrics"], base_metrics)

    train_audit = {
        "status": "C7_B5_1_SN_TRAIN_ONLY_AUDIT_COMPLETE",
        "official_val_used": False,
        "splits": split_results,
    }
    atomic_json(audit / "C7_B5_1_SN_TRAIN_ONLY_AUDIT.json", train_audit)
    atomic_text(audit / "C7_B5_1_SN_TRAIN_ONLY_AUDIT.md", md("C7-B5.1 SN train-only audit", train_audit))

    bias_systems = ["C7_B2_1_A4", "S_MINUTE_like", "S_MINUTE_like_plus_weak_residual_beta_0.02", "S_MINUTE_like_plus_weak_residual_beta_0.05", "S_MINUTE_like_plus_weak_residual_beta_0.1"]
    bias = {
        "status": "C7_B5_1_MOMENT_PREDICTION_BIAS_DIAGNOSTIC_COMPLETE",
        "official_val_used": False,
        "standard_train_calib": {k: split_results["standard_train_calib"][k]["first_correct_video_rank_distribution"] for k in bias_systems},
        "official_like_stress": {k: split_results["official_like_stress"][k]["first_correct_video_rank_distribution"] for k in bias_systems},
    }
    atomic_json(audit / "C7_B5_1_MOMENT_PREDICTION_BIAS_DIAGNOSTIC.json", bias)
    atomic_text(audit / "C7_B5_1_MOMENT_PREDICTION_BIAS_DIAGNOSTIC.md", md("C7-B5.1 moment prediction bias diagnostic", bias))

    safety = {"status": "C7_B5_1_DOMINANCE_SAFETY_AUDIT_COMPLETE", "official_val_used": False, "systems": {}}
    for name in systems:
        rec = split_results["official_like_stress"][name]
        delta = rec["delta_vs_C7_B2_1_A4"]
        d = rec["dominance_safety"]
        m = rec["movement"]
        weak_residual = name.startswith("S_MINUTE_like_plus_weak_residual")
        safety["systems"][name] = {
            "dominance_safety": d,
            "movement": m,
            "R100_delta": {"0.5-r100": delta["0.5-r100"], "0.7-r100": delta["0.7-r100"]},
            "hard_safety_pass": bool(
                m["fixed_pool_invariant"]
                and abs(delta["0.5-r100"]) < 1e-9
                and abs(delta["0.7-r100"]) < 1e-9
                and m["hard_positive_top100_query_exits"] == 0
                and m["invalid_span_count"] == 0
                and m["duplicate_span_count_after_nms"] == 0
            ),
            "weak_residual_extra_pass": None if not weak_residual else bool(
                d["corr_final_S_MINUTE_like"] >= d["corr_final_residual"]
                and d["top1_changed_rate"] <= 0.25
                and d["top5_set_changed_rate"] <= 0.45
                and d["top10_set_changed_rate"] <= 0.60
            ),
        }
    atomic_json(audit / "C7_B5_1_DOMINANCE_SAFETY_AUDIT.json", safety)
    atomic_text(audit / "C7_B5_1_DOMINANCE_SAFETY_AUDIT.md", md("C7-B5.1 dominance safety audit", safety))

    candidates = [k for k in systems if k.startswith("S_MINUTE_like")]
    promising: List[str] = []
    weak: List[str] = []
    for name in candidates:
        stress = split_results["official_like_stress"][name]
        full = split_results["standard_train_calib"][name]
        safe = safety["systems"][name]
        d = stress["delta_vs_C7_B2_1_A4"]
        df = full["delta_vs_C7_B2_1_A4"]
        no_res_over = True
        if name.startswith("S_MINUTE_like_plus"):
            no_res_over = bool(safe["weak_residual_extra_pass"])
        ok = (
            d["0.7-r1"] >= 0
            and d["0.7-r5"] >= 0
            and d["0.7-r10"] >= 0
            and df["0.5-r1"] >= 0
            and df["0.7-r1"] >= 0
            and safe["hard_safety_pass"]
            and no_res_over
        )
        if ok:
            promising.append(name)
        elif safe["hard_safety_pass"]:
            weak.append(name)
    if promising:
        status = "C7_B5_1_SN_SCORING_PROMISING"
        selected = sorted(promising, key=lambda n: sum(split_results["official_like_stress"][n]["delta_vs_C7_B2_1_A4"][k] for k in ["0.7-r1", "0.7-r5", "0.7-r10"]), reverse=True)[0]
    elif weak:
        status = "C7_B5_1_SN_SCORING_WEAK"
        selected = sorted(weak, key=lambda n: sum(split_results["official_like_stress"][n]["delta_vs_C7_B2_1_A4"][k] for k in ["0.7-r1", "0.7-r5", "0.7-r10"]), reverse=True)[0]
    else:
        status = "C7_B5_1_NEGATIVE"
        selected = None

    final = {
        "status": status,
        "official_val_used": False,
        "second_official_val": False,
        "post_val_adjustment": False,
        "C7_B2_1_promoted_decision_modified": False,
        "C7_B3_official_decision_modified": False,
        "C7_B4_archived_decision_modified": False,
        "NMS_modified": False,
        "evaluator_modified": False,
        "CONQUER_QDF_QAL_original_ML_VR_frozen": True,
        "full_fine_tuning": False,
        "primary_baseline": "C7-B2.1 A4 train-only reconstruction",
        "strict_alignment": True,
        "selected_candidate": selected,
        "selected_stress_record": split_results["official_like_stress"].get(selected) if selected else None,
        "recommendation": "Stop here. Do not run official val automatically.",
    }
    atomic_json(audit / "C7_B5_1_FINAL_DECISION.json", final)
    atomic_text(audit / "C7_B5_1_FINAL_DECISION.md", md("C7-B5.1 final decision", final))

    manifest = {
        **final,
        "outputs": {
            "fixed_pool_export": "c7_audit/C7_B5_1_FIXED_POOL_RAW_LOGIT_EXPORT.json",
            "inventory": "c7_audit/C7_B5_1_RAW_LOGIT_SOURCE_INVENTORY.json",
            "alignment": "c7_audit/C7_B5_1_STRICT_ALIGNMENT_AUDIT.json",
            "score_construction": "c7_audit/C7_B5_1_SN_SCORE_CONSTRUCTION.json",
            "train_only_audit": "c7_audit/C7_B5_1_SN_TRAIN_ONLY_AUDIT.json",
            "bias": "c7_audit/C7_B5_1_MOMENT_PREDICTION_BIAS_DIAGNOSTIC.json",
            "safety": "c7_audit/C7_B5_1_DOMINANCE_SAFETY_AUDIT.json",
            "decision": "c7_audit/C7_B5_1_FINAL_DECISION.json",
        },
    }
    atomic_json(audit / "C7_B5_1_MANIFEST.json", manifest)
    hash_paths = [audit / p for p in [
        "C7_B5_1_FIXED_POOL_RAW_LOGIT_EXPORT.json",
        "C7_B5_1_RAW_LOGIT_SOURCE_INVENTORY.json",
        "C7_B5_1_STRICT_ALIGNMENT_AUDIT.json",
        "C7_B5_1_SN_SCORE_CONSTRUCTION.json",
        "C7_B5_1_SN_TRAIN_ONLY_AUDIT.json",
        "C7_B5_1_MOMENT_PREDICTION_BIAS_DIAGNOSTIC.json",
        "C7_B5_1_DOMINANCE_SAFETY_AUDIT.json",
        "C7_B5_1_FINAL_DECISION.json",
        "C7_B5_1_MANIFEST.json",
    ]]
    hashes = {
        "status": "C7_B5_1_HASHES",
        "official_val_used": False,
        "artifacts": {str(p): sha256_file(p) for p in hash_paths},
        "runner_sha256": sha256_file(__file__),
        "fixed_pool_logits_sha256": sha256_file(args.fixed_pool_logits_npz),
    }
    atomic_json(audit / "C7_B5_1_HASHES.json", hashes)
    print(json.dumps({
        "status": status,
        "strict_alignment": True,
        "selected_candidate": selected,
        "selected_stress_delta": final["selected_stress_record"]["delta_vs_C7_B2_1_A4"] if selected else None,
        "official_val_used": False,
        "post_val_adjustment": False,
        "second_official_val": False,
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
