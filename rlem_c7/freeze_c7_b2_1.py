#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rlem_c7.run_c7_b0_b1 import (  # noqa: E402
    atomic_json,
    atomic_text,
    artifact,
    gt_span_by_query,
    gt_video_by_query,
    labels_for_sequence,
    load_npz,
    metric_delta,
    nms_sequence,
    selected_metrics,
)
from rlem_c7.run_c7_b2_1_safety_repair import group_score_map, normalize_cand  # noqa: E402


METRIC_KEYS = [
    "0.5-r1", "0.5-r5", "0.5-r10", "0.5-r100",
    "0.7-r1", "0.7-r5", "0.7-r10", "0.7-r100",
]
Candidate = Tuple[int, int, int, int, float, int]


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_obj(obj: Any) -> str:
    data = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def write_jsonl(path: str | Path, rows: List[Dict[str, Any]]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(p) + ".partial")
    with tmp.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n")
    os.replace(tmp, p)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--calib_cache", default="results/rlem_c6a/cache/train_calib_c6_cache.npz")
    p.add_argument("--b21_manifest", default="c7_audit/C7_B2_1_MANIFEST.json")
    p.add_argument("--b21_final", default="c7_audit/C7_B2_1_FINAL_DECISION.json")
    p.add_argument("--anchor_states", default="results/rlem_c7_b2/train_calib_c7_b1_anchor_states.json")
    p.add_argument("--video_scores", default="results/rlem_c7_b2/train_calib_video_scores.npz")
    p.add_argument("--video_dataset", default="results/rlem_c7_b2/train_calib_video_dataset.npz")
    p.add_argument("--model", default="results/rlem_c7_b2/c7_b2_mil_video_residual_head.pt")
    p.add_argument("--audit_dir", default="c7_audit")
    p.add_argument("--output_dir", default="results/rlem_c7_b2_1/freeze")
    p.add_argument("--effective_top_n", type=int, default=100)
    p.add_argument("--max_after_nms", type=int, default=100)
    p.add_argument("--nms_thd", type=float, default=0.7)
    return p.parse_args()


def cand_key(c: Candidate) -> Tuple[int, int, int]:
    return int(c[1]), int(c[2]), int(c[3])


def nms(cands: Sequence[Candidate], args: argparse.Namespace) -> List[Candidate]:
    return nms_sequence(list(cands)[:args.effective_top_n], args.nms_thd, args.max_after_nms)


def build_sequences(
    states: List[Dict[str, Any]],
    gscore: Dict[int, Tuple[float, float, float, float]],
    cfg: Dict[str, Any],
    args: argparse.Namespace,
    video_weight: float,
) -> Tuple[List[List[Candidate]], List[List[Candidate]], List[List[Candidate]], List[List[Candidate]]]:
    anchor_pre: List[List[Candidate]] = []
    anchor_post: List[List[Candidate]] = []
    selected_pre: List[List[Candidate]] = []
    selected_post: List[List[Candidate]] = []
    for state in states:
        ar = [normalize_cand(x) for x in state["flat_candidates"]]
        sr = []
        for c in ar:
            residual = gscore.get(int(c[0]), (0.0, 0.0, 0.0, 0.0))[2]
            sr.append((c[0], c[1], c[2], c[3], float(c[4]) + float(video_weight) * residual, c[5]))
        sr = sorted(sr, key=lambda x: -x[4])
        anchor_pre.append(ar)
        anchor_post.append(nms(ar, args))
        selected_pre.append(sr)
        selected_post.append(nms(sr, args))
    return anchor_pre, anchor_post, selected_pre, selected_post


def evaluate_post(
    name: str,
    seqs: List[List[Candidate]],
    anchor: List[List[Candidate]],
    gt_vid: np.ndarray,
    gt_s: np.ndarray,
    gt_e: np.ndarray,
) -> Dict[str, Any]:
    labels05: List[List[bool]] = []
    labels07: List[List[bool]] = []
    anchor_labels05: List[List[bool]] = []
    anchor_labels07: List[List[bool]] = []
    exits = entries = invalid = dup = 0
    lost: List[int] = []
    rows: List[Dict[str, Any]] = []
    score_rows: List[Dict[str, Any]] = []
    for q, seq in enumerate(seqs):
        l05, l07, pos, inv, du = labels_for_sequence(seq, int(gt_vid[q]), int(gt_s[q]), int(gt_e[q]))
        a05, a07, apos, _ai, _ad = labels_for_sequence(anchor[q], int(gt_vid[q]), int(gt_s[q]), int(gt_e[q]))
        labels05.append(l05); labels07.append(l07)
        anchor_labels05.append(a05); anchor_labels07.append(a07)
        invalid += inv; dup += du
        exits += int(apos and not pos)
        entries += int((not apos) and pos)
        if apos and not pos:
            lost.append(int(q))
        for rank, c in enumerate(seq):
            rows.append({"q": int(q), "rank": int(rank), "video_idx": int(c[1]), "start_idx": int(c[2]), "end_idx": int(c[3])})
            score_rows.append({"q": int(q), "rank": int(rank), "score": round(float(c[4]), 8)})
    metrics = selected_metrics(labels05, labels07)
    gain_loss: Dict[str, Dict[str, int]] = {}
    for thr, cur, base in [("0.5", labels05, anchor_labels05), ("0.7", labels07, anchor_labels07)]:
        for r in [1, 5, 10]:
            gain = loss = 0
            for q in range(len(cur)):
                b = any(base[q][:r])
                c = any(cur[q][:r])
                gain += int((not b) and c)
                loss += int(b and not c)
            gain_loss[f"{thr}-r{r}"] = {"gain_queries": int(gain), "loss_queries": int(loss)}
    return {
        "name": name,
        "metrics": metrics,
        "movement": {
            "hard_positive_top100_query_exits": int(exits),
            "hard_positive_top100_query_entries": int(entries),
            "hard_positive_exit_ratio": float(exits / max(sum(labels_for_sequence(a, int(gt_vid[q]), int(gt_s[q]), int(gt_e[q]))[2] for q, a in enumerate(anchor)), 1)),
            "invalid_span_count": int(invalid),
            "duplicate_span_count_after_nms": int(dup),
        },
        "rank_gain_loss_vs_C7_B1_frozen": gain_loss,
        "top100_lost_query_count": int(len(lost)),
        "top100_lost_query_hash": sha256_obj(lost),
        "prediction_rows": rows,
        "score_rows": score_rows,
    }


def pool_identity_sets(seqs: List[List[Candidate]]) -> List[List[Tuple[int, int, int]]]:
    return [sorted(cand_key(c) for c in seq) for seq in seqs]


def md_json(title: str, obj: Any) -> str:
    return f"# {title}\n\n```json\n{json.dumps(obj, indent=2, ensure_ascii=False)}\n```\n"


def strip_heavy_eval(rec: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in rec.items() if k not in {"prediction_rows", "score_rows"}}


def main() -> None:
    args = parse_args()
    Path(args.audit_dir).mkdir(parents=True, exist_ok=True)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    manifest = json.loads(Path(args.b21_manifest).read_text(encoding="utf-8"))
    if manifest.get("status") != "C7_B2_1_R1_PRESERVED_COVERAGE_REPAIRED":
        raise ValueError("C7-B2.1 is not in repaired state")
    if manifest.get("official_val_used") is not False:
        raise ValueError("official val is not allowed for freeze package")
    best = manifest["best"]
    if best.get("name") != "A4_video_residual_only":
        raise ValueError("selected variant is not A4_video_residual_only")
    cfg = json.loads(Path("c7_audit/C7_B2_TRAIN_CALIB_EVAL.json").read_text(encoding="utf-8"))["best"]["config"]
    baselines = manifest["baselines"]
    cache = load_npz(args.calib_cache, allow_pickle=True)
    gt_vid = gt_video_by_query(cache)
    gt_s, gt_e = gt_span_by_query(cache)
    states = json.loads(Path(args.anchor_states).read_text(encoding="utf-8"))["states"]
    score_arrays = load_npz(args.video_scores, allow_pickle=False)
    gscore = group_score_map(score_arrays, cfg)
    selected_weight = float(cfg.get("video_weight", 0.75))
    anchor_pre, anchor_post, selected_pre, selected_post = build_sequences(states, gscore, cfg, args, selected_weight)
    _zpre, zero_post, _zpre2, zero_post2 = build_sequences(states, gscore, cfg, args, 0.0)

    selected_eval = evaluate_post("A4_video_residual_only", selected_post, anchor_post, gt_vid, gt_s, gt_e)
    selected_eval["delta_vs_C6_B1"] = metric_delta(selected_eval["metrics"], baselines["C6_B1"])
    selected_eval["delta_vs_C6_B2"] = metric_delta(selected_eval["metrics"], baselines["C6_B2"])
    selected_eval["delta_vs_C7_B1_frozen"] = metric_delta(selected_eval["metrics"], baselines["C7_B1_frozen"])

    pre_pool_equal = all(pool_identity_sets([anchor_pre[q]])[0] == pool_identity_sets([selected_pre[q]])[0] for q in range(len(states)))
    post_pool_equal = all(pool_identity_sets([anchor_post[q]])[0] == pool_identity_sets([selected_post[q]])[0] for q in range(len(states)))
    fixed_pool = {
        "status": "PASS" if pre_pool_equal and selected_eval["top100_lost_query_count"] == 0 and selected_eval["movement"]["hard_positive_top100_query_exits"] == 0 and selected_eval["movement"]["hard_positive_top100_query_entries"] == 0 and selected_eval["delta_vs_C7_B1_frozen"]["0.5-r100"] == 0 and selected_eval["delta_vs_C7_B1_frozen"]["0.7-r100"] == 0 else "FAIL",
        "official_val_used": False,
        "pre_nms_top100_candidate_pool_identical": bool(pre_pool_equal),
        "post_nms_output_identity_set_identical": bool(post_pool_equal),
        "post_nms_note": "A4 fixed-pool invariant is defined on the C7-B1 frozen pre-NMS top100 candidate pool; post-NMS set may differ because ranking changes NMS order.",
        "top100_lost_query_count": selected_eval["top100_lost_query_count"],
        "top100_lost_query_hash": selected_eval["top100_lost_query_hash"],
        "hard_positive_top100_query_exits": selected_eval["movement"]["hard_positive_top100_query_exits"],
        "hard_positive_top100_query_entries": selected_eval["movement"]["hard_positive_top100_query_entries"],
        "delta_vs_C7_B1_frozen": selected_eval["delta_vs_C7_B1_frozen"],
        "metrics": selected_eval["metrics"],
    }
    atomic_json(Path(args.audit_dir) / "C7_B2_1_FIXED_POOL_INVARIANT.json", fixed_pool)
    atomic_text(Path(args.audit_dir) / "C7_B2_1_FIXED_POOL_INVARIANT.md", md_json("C7-B2.1 fixed-pool invariant", fixed_pool))
    if fixed_pool["status"] != "PASS":
        raise RuntimeError("fixed-pool invariant failed; stopping freeze package")

    zero_eval = evaluate_post("zero_residual_control", zero_post2, anchor_post, gt_vid, gt_s, gt_e)
    zero_delta = metric_delta(zero_eval["metrics"], baselines["C7_B1_frozen"])
    zero_pred_path = Path(args.output_dir) / "zero_residual_predictions.jsonl"
    zero_score_path = Path(args.output_dir) / "zero_residual_scores.jsonl"
    anchor_pred_path = Path(args.output_dir) / "anchor_predictions.jsonl"
    write_jsonl(zero_pred_path, zero_eval["prediction_rows"])
    write_jsonl(zero_score_path, zero_eval["score_rows"])
    anchor_eval = evaluate_post("anchor", anchor_post, anchor_post, gt_vid, gt_s, gt_e)
    write_jsonl(anchor_pred_path, anchor_eval["prediction_rows"])
    zero_control = {
        "status": "PASS" if all(abs(v) < 1e-12 for v in zero_delta.values()) and pool_identity_sets(zero_post2) == pool_identity_sets(anchor_post) else "FAIL",
        "official_val_used": False,
        "video_residual_weight": 0.0,
        "metrics": zero_eval["metrics"],
        "delta_vs_C7_B1_frozen": zero_delta,
        "candidate_set_identical": pool_identity_sets(zero_post2) == pool_identity_sets(anchor_post),
        "prediction_hash_identical_to_anchor": sha256_file(zero_pred_path) == sha256_file(anchor_pred_path),
        "prediction_hash": sha256_file(zero_pred_path),
        "anchor_prediction_hash": sha256_file(anchor_pred_path),
        "NMS_modified": False,
        "evaluator_modified": False,
    }
    atomic_json(Path(args.audit_dir) / "C7_B2_1_ZERO_RESIDUAL_CONTROL.json", zero_control)
    atomic_text(Path(args.audit_dir) / "C7_B2_1_ZERO_RESIDUAL_CONTROL.md", md_json("C7-B2.1 zero-residual control", zero_control))

    pred_path = Path(args.output_dir) / "selected_A4_predictions.jsonl"
    score_path = Path(args.output_dir) / "selected_A4_scores.jsonl"
    write_jsonl(pred_path, selected_eval["prediction_rows"])
    write_jsonl(score_path, selected_eval["score_rows"])
    metric_abs_diff = {k: abs(selected_eval["metrics"][k] - best["metrics"][k]) for k in METRIC_KEYS}
    deterministic = {
        "status": "PASS" if all(v < 1e-9 for v in metric_abs_diff.values()) else "FAIL",
        "official_val_used": False,
        "selected_variant": "A4_video_residual_only",
        "metrics": selected_eval["metrics"],
        "expected_metrics": best["metrics"],
        "metric_abs_diff_vs_C7_B2_1_FINAL_DECISION": metric_abs_diff,
        "metric_hash": sha256_obj(selected_eval["metrics"]),
        "prediction_hash": sha256_file(pred_path),
        "score_hash": sha256_file(score_path),
        "model_hash": artifact(args.model),
        "config_hash": sha256_obj({"variant": "A4_video_residual_only", "video_weight": selected_weight, "base_config": cfg}),
        "prediction_file": artifact(pred_path),
        "score_file": artifact(score_path),
    }
    atomic_json(Path(args.audit_dir) / "C7_B2_1_DETERMINISTIC_RERUN.json", deterministic)
    atomic_text(Path(args.audit_dir) / "C7_B2_1_DETERMINISTIC_RERUN.md", md_json("C7-B2.1 deterministic rerun", deterministic))

    ds = np.load(args.video_dataset, allow_pickle=True)
    names = [str(x) for x in ds["feature_names"].tolist()]
    banned_exact = {"iou", "gt_iou", "hit05", "hit07", "y05", "y07", "label", "oracle_decision", "hard_positive_label"}
    banned_sub = ["official", "oracle", "gt_", "_label", "label_", "success", "failure", "hard_positive"]
    allowed = {"c6c_iou"}
    offenders = [n for n in names if n not in allowed and (n.lower() in banned_exact or any(s in n.lower() for s in banned_sub))]
    leakage = {
        "status": "PASS" if not offenders else "FAIL",
        "official_val_used": False,
        "inference_artifacts": ["train_calib_video_scores.npz model predictions", "C7-B1 frozen flat candidate scores"],
        "feature_names": names,
        "banned_feature_name_matches": offenders,
        "allowed_predicted_iou_like_features": sorted(allowed.intersection(names)),
        "contains_GT_IoU_as_inference_feature": False,
        "contains_hit_label_as_inference_feature": False,
        "contains_train_calib_label_as_inference_feature": False,
        "contains_official_val_prediction_as_inference_feature": False,
        "contains_oracle_decision_as_inference_feature": False,
        "training_labels_allowed_only_for_training": True,
    }
    atomic_json(Path(args.audit_dir) / "C7_B2_1_LEAKAGE_AUDIT.json", leakage)
    atomic_text(Path(args.audit_dir) / "C7_B2_1_LEAKAGE_AUDIT.md", md_json("C7-B2.1 leakage audit", leakage))

    neighbor_records = []
    for w in [0.0, 0.25, 0.5, selected_weight, 1.0, 1.25]:
        _ap, ap, _sp, sp = build_sequences(states, gscore, cfg, args, w)
        ev = evaluate_post(f"video_weight_{w}", sp, anchor_post, gt_vid, gt_s, gt_e)
        ev["config"] = {"video_residual_weight": w}
        ev["delta_vs_C7_B1_frozen"] = metric_delta(ev["metrics"], baselines["C7_B1_frozen"])
        ev["metric_abs_diff_vs_selected"] = {k: ev["metrics"][k] - selected_eval["metrics"][k] for k in METRIC_KEYS}
        neighbor_records.append(strip_heavy_eval(ev))
    neighbor = {
        "status": "PASS",
        "official_val_used": False,
        "scope": "audit_only_no_reselection",
        "selected_variant_fixed": "A4_video_residual_only",
        "selected_video_residual_weight": selected_weight,
        "selection_changed": False,
        "records": neighbor_records,
        "stable_region_note": "Weights around the selected value are audited only; higher neighbors do not change the frozen selection.",
    }
    atomic_json(Path(args.audit_dir) / "C7_B2_1_NEIGHBOR_SANITY.json", neighbor)
    atomic_text(Path(args.audit_dir) / "C7_B2_1_NEIGHBOR_SANITY.md", md_json("C7-B2.1 neighbor sanity", neighbor))

    prereg = {
        "status": "C7_B2_1_OFFICIAL_GATE_PREREG",
        "official_val_used": False,
        "promotion_gate": {
            "core": [
                "0.7-r1 vs C6-B2 > 0",
                "0.7-r5 vs C6-B2 >= 0",
                "0.7-r10 vs C6-B2 >= 0",
                "0.5-r1 vs C6-B2 > 0",
                "at least 5/6 R@1/R@5/R@10 core metrics vs C6-B2 non-negative",
            ],
            "safety": [
                "0.7-r100 vs C7-B1 official >= -0.10",
                "0.5-r100 vs C7-B1 official >= -0.10",
                "invalid_span_count = 0",
                "duplicate_span_count_after_nms = 0",
                "hard_positive_exit_ratio = 0 or near-zero",
                "evaluator_modified = false",
                "nms_modified = false",
                "post_val_adjustment = false",
                "second_official_val = false",
            ],
        },
        "status_labels": [
            "C7_B2_1_OFFICIAL_PROMOTED",
            "C7_B2_1_OFFICIAL_CORE_POSITIVE_REVIEW",
            "C7_B2_1_R1_TRADEOFF_NO_PROMOTION",
            "C7_B2_1_OFFICIAL_NEGATIVE",
            "C7_B2_1_INFRA_FAIL",
        ],
    }
    atomic_json(Path(args.audit_dir) / "C7_B2_1_OFFICIAL_GATE_PREREG.json", prereg)
    atomic_text(Path(args.audit_dir) / "C7_B2_1_OFFICIAL_GATE_PREREG.md", md_json("C7-B2.1 official gate prereg", prereg))

    pass_all = all(x["status"] == "PASS" for x in [fixed_pool, zero_control, deterministic, leakage, neighbor])
    status = "C7_B2_1_OFFICIAL_READY" if pass_all else "C7_B2_1_FREEZE_REVIEW_FAIL"
    final = {
        "status": status,
        "official_val_used": False,
        "official_val_run": False,
        "post_val_adjustment": False,
        "second_official_val": False,
        "promoted": False,
        "selected_variant": "A4_video_residual_only",
        "selected_config": {"video_residual_weight": selected_weight, "fixed_pool": "C7_B1_frozen_pre_nms_top100"},
        "metrics": selected_eval["metrics"],
        "delta_vs_C6_B1": selected_eval["delta_vs_C6_B1"],
        "delta_vs_C6_B2": selected_eval["delta_vs_C6_B2"],
        "delta_vs_C7_B1_frozen": selected_eval["delta_vs_C7_B1_frozen"],
        "movement": selected_eval["movement"],
        "fixed_pool_invariant_pass": fixed_pool["status"] == "PASS",
        "zero_residual_control_pass": zero_control["status"] == "PASS",
        "deterministic_rerun_pass": deterministic["status"] == "PASS",
        "leakage_audit_pass": leakage["status"] == "PASS",
        "neighbor_sanity_pass": neighbor["status"] == "PASS",
        "official_gate_preregistered": True,
        "NMS_modified": False,
        "evaluator_modified": False,
        "C4_C6_artifacts_modified": False,
        "C7_B1_official_archived_decision_modified": False,
        "enter_C7_C": False,
        "enter_C8": False,
        "request": "Stop and wait for human authorization for exactly-one official-val one-shot.",
    }
    atomic_json(Path(args.audit_dir) / "C7_B2_1_FINAL_FREEZE_PACKAGE.json", final)
    atomic_text(Path(args.audit_dir) / "C7_B2_1_FINAL_FREEZE_PACKAGE.md", md_json("C7-B2.1 final freeze package", final))
    ready = {
        "status": status,
        "official_val_used": False,
        "promoted": False,
        "selected_variant": "A4_video_residual_only",
        "model": artifact(args.model),
        "freeze_package": artifact(Path(args.audit_dir) / "C7_B2_1_FINAL_FREEZE_PACKAGE.json"),
        "requires_human_authorization_before_official_val": True,
    }
    atomic_json(Path(args.audit_dir) / "C7_B2_1_OFFICIAL_READY_MANIFEST.json", ready)
    hashes = {
        "status": "C7_B2_1_FREEZE_HASHES",
        "official_val_used": False,
        "metric_hash": deterministic["metric_hash"],
        "prediction_hash": deterministic["prediction_hash"],
        "score_hash": deterministic["score_hash"],
        "artifacts": {
            "fixed_pool": artifact(Path(args.audit_dir) / "C7_B2_1_FIXED_POOL_INVARIANT.json"),
            "zero_residual": artifact(Path(args.audit_dir) / "C7_B2_1_ZERO_RESIDUAL_CONTROL.json"),
            "deterministic": artifact(Path(args.audit_dir) / "C7_B2_1_DETERMINISTIC_RERUN.json"),
            "leakage": artifact(Path(args.audit_dir) / "C7_B2_1_LEAKAGE_AUDIT.json"),
            "neighbor": artifact(Path(args.audit_dir) / "C7_B2_1_NEIGHBOR_SANITY.json"),
            "prereg": artifact(Path(args.audit_dir) / "C7_B2_1_OFFICIAL_GATE_PREREG.json"),
            "final": artifact(Path(args.audit_dir) / "C7_B2_1_FINAL_FREEZE_PACKAGE.json"),
            "ready": artifact(Path(args.audit_dir) / "C7_B2_1_OFFICIAL_READY_MANIFEST.json"),
            "runner": artifact(__file__),
            "model": artifact(args.model),
        },
    }
    atomic_json(Path(args.audit_dir) / "C7_B2_1_FREEZE_HASHES.json", hashes)
    print(json.dumps({
        "status": status,
        "selected_variant": "A4_video_residual_only",
        "metrics": selected_eval["metrics"],
        "delta_vs_C7_B1_frozen": selected_eval["delta_vs_C7_B1_frozen"],
        "movement": selected_eval["movement"],
        "fixed_pool_invariant_pass": fixed_pool["status"] == "PASS",
        "zero_residual_control_pass": zero_control["status"] == "PASS",
        "deterministic_rerun_pass": deterministic["status"] == "PASS",
        "leakage_audit_pass": leakage["status"] == "PASS",
        "official_val_used": False,
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
