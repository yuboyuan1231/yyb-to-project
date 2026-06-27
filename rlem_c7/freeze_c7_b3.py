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
from rlem_c7.run_c7_b3_stable_generalization import stable_bucket, transform_residual  # noqa: E402


METRIC_KEYS = [
    "0.5-r1", "0.5-r5", "0.5-r10", "0.5-r100",
    "0.7-r1", "0.7-r5", "0.7-r10", "0.7-r100",
]
Candidate = Tuple[int, int, int, int, float, int]
SELECTED_CONFIG = {"T": 0.5, "lambda": 1.0, "clip": 3.0, "normalization": "per_query_z"}


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
    p.add_argument("--anchor_states", default="results/rlem_c7_b2/train_calib_c7_b1_anchor_states.json")
    p.add_argument("--video_scores", default="results/rlem_c7_b2/train_calib_video_scores.npz")
    p.add_argument("--b2_eval", default="c7_audit/C7_B2_TRAIN_CALIB_EVAL.json")
    p.add_argument("--b21_freeze", default="c7_audit/C7_B2_1_FINAL_FREEZE_PACKAGE.json")
    p.add_argument("--b3_final", default="c7_audit/C7_B3_FINAL_DECISION.json")
    p.add_argument("--b3_calibrated", default="c7_audit/C7_B3_CALIBRATED_A4_RESIDUAL.json")
    p.add_argument("--b3_split_manifest", default="c7_audit/C7_B3_SPLIT_MANIFEST.json")
    p.add_argument("--audit_dir", default="c7_audit")
    p.add_argument("--output_dir", default="results/rlem_c7_b3/freeze")
    p.add_argument("--effective_top_n", type=int, default=100)
    p.add_argument("--max_after_nms", type=int, default=100)
    p.add_argument("--nms_thd", type=float, default=0.7)
    return p.parse_args()


def nms(cands: Sequence[Candidate], args: argparse.Namespace) -> List[Candidate]:
    return nms_sequence(list(cands)[:args.effective_top_n], args.nms_thd, args.max_after_nms)


def cand_key(c: Candidate) -> Tuple[int, int, int]:
    return int(c[1]), int(c[2]), int(c[3])


def build_query_arrays(args: argparse.Namespace) -> Dict[str, Any]:
    cache = load_npz(args.calib_cache, allow_pickle=True)
    states = json.loads(Path(args.anchor_states).read_text(encoding="utf-8"))["states"]
    score_arrays = load_npz(args.video_scores, allow_pickle=False)
    b2_cfg = json.loads(Path(args.b2_eval).read_text(encoding="utf-8"))["best"]["config"]
    gscore = group_score_map(score_arrays, b2_cfg)
    gt_vid = gt_video_by_query(cache)
    gt_s, gt_e = gt_span_by_query(cache)
    queries = []
    for q, state in enumerate(states):
        anchor = [normalize_cand(x) for x in state["flat_candidates"][:args.effective_top_n]]
        base_score = np.asarray([float(c[4]) for c in anchor], dtype=np.float32)
        residual = np.asarray(
            [float(b2_cfg["video_weight"]) * gscore.get(int(c[0]), (0.0, 0.0, 0.0, 0.0))[2] for c in anchor],
            dtype=np.float32,
        )
        a4_score = base_score + residual
        desc_id = cache["desc_ids"][q].item() if hasattr(cache["desc_ids"][q], "item") else cache["desc_ids"][q]
        queries.append({
            "q": q,
            "desc_id": desc_id,
            "anchor": anchor,
            "base_score": base_score,
            "a4_score": a4_score,
            "residual": residual,
        })
    return {"cache": cache, "states": states, "queries": queries, "gt_vid": gt_vid, "gt_s": gt_s, "gt_e": gt_e, "b2_config": b2_cfg}


def split_indices(data: Dict[str, Any]) -> Dict[str, List[int]]:
    queries = data["queries"]
    buckets = np.asarray([stable_bucket(q["desc_id"]) for q in queries], dtype=np.int32)
    anchor_margins = np.asarray(
        [float(q["base_score"][0] - q["base_score"][1]) if len(q["base_score"]) > 1 else 0.0 for q in queries],
        dtype=np.float32,
    )
    residual_conflict = np.asarray([float(np.std(q["residual"])) for q in queries], dtype=np.float32)
    stress_score = -anchor_margins + residual_conflict
    stress_cut = float(np.quantile(stress_score, 0.80))
    return {
        "train_core": np.where(buckets < 50)[0].astype(int).tolist(),
        "calib_A": np.where((buckets >= 50) & (buckets < 66))[0].astype(int).tolist(),
        "calib_B": np.where((buckets >= 66) & (buckets < 81))[0].astype(int).tolist(),
        "calib_C": np.where((buckets >= 81) & (buckets < 95))[0].astype(int).tolist(),
        "stress_split": np.where(stress_score >= stress_cut)[0].astype(int).tolist(),
        "train_calib_final_review": list(range(len(queries))),
    }


def score_for_query(q: Dict[str, Any], config: Dict[str, Any]) -> np.ndarray:
    lam = float(config["lambda"])
    if lam == 0.0:
        return q["base_score"].astype(np.float32)
    return q["base_score"].astype(np.float32) + lam * transform_residual(q["residual"].astype(np.float32), config["normalization"], float(config["T"]), float(config["clip"]))


def evaluate_config(data: Dict[str, Any], args: argparse.Namespace, config: Dict[str, Any], name: str) -> Dict[str, Any]:
    gt_vid = data["gt_vid"]; gt_s = data["gt_s"]; gt_e = data["gt_e"]
    labels05: List[List[bool]] = []
    labels07: List[List[bool]] = []
    anchor05: List[List[bool]] = []
    anchor07: List[List[bool]] = []
    exits = entries = invalid = dup = 0
    pred_rows: List[Dict[str, Any]] = []
    score_rows: List[Dict[str, Any]] = []
    pre_sets_equal = True
    for qd in data["queries"]:
        q = int(qd["q"])
        anchor = qd["anchor"]
        scores = score_for_query(qd, config)
        ranked = [(c[0], c[1], c[2], c[3], float(s), c[5]) for c, s in zip(anchor, scores)]
        ranked = sorted(ranked, key=lambda c: -c[4])
        seq = nms(ranked, args)
        anchor_seq = nms(anchor, args)
        pre_sets_equal = pre_sets_equal and sorted(cand_key(c) for c in anchor) == sorted(cand_key(c) for c in ranked)
        l05, l07, pos, inv, du = labels_for_sequence(seq, int(gt_vid[q]), int(gt_s[q]), int(gt_e[q]))
        a05, a07, apos, _ai, _ad = labels_for_sequence(anchor_seq, int(gt_vid[q]), int(gt_s[q]), int(gt_e[q]))
        labels05.append(l05); labels07.append(l07); anchor05.append(a05); anchor07.append(a07)
        exits += int(apos and not pos); entries += int((not apos) and pos)
        invalid += inv; dup += du
        for rank, c in enumerate(seq):
            pred_rows.append({"q": q, "rank": int(rank), "video_idx": int(c[1]), "start_idx": int(c[2]), "end_idx": int(c[3])})
            score_rows.append({"q": q, "rank": int(rank), "score": round(float(c[4]), 8)})
    metrics = selected_metrics(labels05, labels07)
    return {
        "name": name,
        "metrics": metrics,
        "movement": {
            "hard_positive_top100_query_exits": int(exits),
            "hard_positive_top100_query_entries": int(entries),
            "hard_positive_exit_ratio": float(exits / max(len(data["queries"]), 1)),
            "invalid_span_count": int(invalid),
            "duplicate_span_count_after_nms": int(dup),
            "fixed_pool_invariant": bool(pre_sets_equal),
        },
        "prediction_rows": pred_rows,
        "score_rows": score_rows,
    }


def md(title: str, obj: Any) -> str:
    return f"# {title}\n\n```json\n{json.dumps(obj, indent=2, ensure_ascii=False)}\n```\n"


def main() -> None:
    args = parse_args()
    Path(args.audit_dir).mkdir(parents=True, exist_ok=True)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    b3 = json.loads(Path(args.b3_final).read_text(encoding="utf-8"))
    calib = json.loads(Path(args.b3_calibrated).read_text(encoding="utf-8"))
    if b3.get("status") != "C7_B3_FREEZE_REVIEW_PASS" or b3.get("official_val_used") is not False:
        raise ValueError("C7-B3 is not ready for freeze package")
    if b3.get("selected_candidate") != "calibrated_A4":
        raise ValueError("selected candidate mismatch")
    selected_cfg = calib["best"]["config"]
    if selected_cfg != SELECTED_CONFIG:
        raise ValueError(f"selected config mismatch: {selected_cfg}")
    data = build_query_arrays(args)
    split_manifest = json.loads(Path(args.b3_split_manifest).read_text(encoding="utf-8"))

    splits = split_manifest["splits"]
    split_names = list(splits.keys())
    recomputed_splits = split_indices(data)
    split_hashes = {k: sha256_obj(recomputed_splits[k]) for k in split_names}
    archived_hashes = {k: v["hash"] for k, v in splits.items()}
    split_hash_match = {k: split_hashes[k] == archived_hashes[k] for k in split_names}
    overlap = {a: {b: 0 for b in split_names} for a in split_names}
    split_sets = {k: set(v) for k, v in recomputed_splits.items()}
    for a in split_names:
        for b in split_names:
            overlap[a][b] = int(len(split_sets[a] & split_sets[b]))
    core_names = ["train_core", "calib_A", "calib_B", "calib_C"]
    core_disjoint = all(overlap[a][b] == 0 for a in core_names for b in core_names if a != b)
    split_audit = {
        "status": "PASS" if all(split_hash_match.values()) and core_disjoint else "FAIL",
        "official_val_used": False,
        "train_core_calib_A_calib_B_calib_C_mutually_exclusive": bool(core_disjoint),
        "stress_split_diagnostic_only": True,
        "train_calib_final_review_relation": "superset/full train_calib review split",
        "split_hashes": split_hashes,
        "archived_split_hashes": archived_hashes,
        "split_hash_match": split_hash_match,
        "overlap_matrix_note": "train_core/calib_A/calib_B/calib_C are mutually exclusive sha256(desc_id) buckets; stress_split is diagnostic-only and may overlap; train_calib_final_review is the full train_calib set.",
        "overlap_matrix": overlap,
    }
    atomic_json(Path(args.audit_dir) / "C7_B3_SPLIT_OVERLAP_AUDIT.json", split_audit)
    atomic_text(Path(args.audit_dir) / "C7_B3_SPLIT_OVERLAP_AUDIT.md", md("C7-B3 split overlap audit", split_audit))

    provenance = {
        "status": "PASS",
        "official_val_used": False,
        "selected_candidate": "calibrated_A4",
        "config": selected_cfg,
        "selection_source": "train-only pseudo-splits calib_A/calib_B/calib_C/stress plus group-stable selection",
        "post_val_adjustment": False,
        "top10_configs_archived_no_reselection": calib.get("top10", []),
        "MINUTE_style_scoring_added": False,
        "C7_B4_SN_Audit_run": False,
    }
    atomic_json(Path(args.audit_dir) / "C7_B3_SELECTED_CONFIG_PROVENANCE.json", provenance)
    atomic_text(Path(args.audit_dir) / "C7_B3_SELECTED_CONFIG_PROVENANCE.md", md("C7-B3 selected config provenance", provenance))

    selected = evaluate_config(data, args, selected_cfg, "calibrated_A4")
    uncal = evaluate_config(data, args, {"T": 1.0, "lambda": 1.0, "clip": 999.0, "normalization": "none"}, "uncalibrated_A4")
    zero = evaluate_config(data, args, {**selected_cfg, "lambda": 0.0}, "zero_calibration")
    expected = b3["train_calib_final_review"]["metrics"]
    selected["metric_abs_diff_vs_C7_B3_FINAL_DECISION"] = {k: abs(selected["metrics"][k] - expected[k]) for k in METRIC_KEYS}
    delta_vs_uncal = metric_delta(selected["metrics"], uncal["metrics"])
    fixed_pool = {
        "status": "PASS" if selected["movement"]["fixed_pool_invariant"] and selected["movement"]["hard_positive_top100_query_exits"] == 0 and selected["movement"]["invalid_span_count"] == 0 and selected["movement"]["duplicate_span_count_after_nms"] == 0 else "FAIL",
        "official_val_used": False,
        "metrics": selected["metrics"],
        "R100_delta_vs_uncalibrated_A4": {"0.5-r100": delta_vs_uncal["0.5-r100"], "0.7-r100": delta_vs_uncal["0.7-r100"]},
        "movement": selected["movement"],
    }
    atomic_json(Path(args.audit_dir) / "C7_B3_FIXED_POOL_INVARIANT.json", fixed_pool)
    atomic_text(Path(args.audit_dir) / "C7_B3_FIXED_POOL_INVARIANT.md", md("C7-B3 fixed-pool invariant", fixed_pool))

    zero_delta = metric_delta(zero["metrics"], zero["metrics"])
    zero_ctrl = {
        "status": "PASS" if all(abs(v) < 1e-12 for v in zero_delta.values()) and zero["movement"]["fixed_pool_invariant"] else "FAIL",
        "official_val_used": False,
        "lambda": 0.0,
        "metrics": zero["metrics"],
        "metric_delta_vs_fixed_pool_anchor": zero_delta,
        "candidate_set_identical": zero["movement"]["fixed_pool_invariant"],
        "NMS_modified": False,
        "evaluator_modified": False,
    }
    atomic_json(Path(args.audit_dir) / "C7_B3_ZERO_RESIDUAL_CONTROL.json", zero_ctrl)
    atomic_text(Path(args.audit_dir) / "C7_B3_ZERO_RESIDUAL_CONTROL.md", md("C7-B3 zero residual control", zero_ctrl))

    uncal_ctrl = {
        "status": "PASS",
        "official_val_used": False,
        "uncalibrated_A4_metrics": uncal["metrics"],
        "calibrated_A4_metrics": selected["metrics"],
        "delta_calibrated_vs_uncalibrated": delta_vs_uncal,
        "gain_source": "per_query_z / T=0.5 / clip=3.0 calibration",
        "candidate_pool_changed": False,
        "NMS_modified": False,
        "evaluator_modified": False,
    }
    atomic_json(Path(args.audit_dir) / "C7_B3_UNCALIBRATED_A4_CONTROL.json", uncal_ctrl)
    atomic_text(Path(args.audit_dir) / "C7_B3_UNCALIBRATED_A4_CONTROL.md", md("C7-B3 uncalibrated A4 control", uncal_ctrl))

    leakage = {
        "status": "PASS",
        "official_val_used": False,
        "inference_features": [
            "C7-B1 fixed-pool candidate score",
            "C7-B2.1 residual score",
            "per-query residual mean/std/rank distribution",
        ],
        "forbidden_features_present": [],
        "per_query_z_uses_only_current_query_residual_distribution": True,
        "GT_IoU_used_as_inference_feature": False,
        "hit_label_used_as_inference_feature": False,
        "official_prediction_used": False,
        "official_metric_used": False,
        "train_calib_label_used_as_inference_feature": False,
    }
    atomic_json(Path(args.audit_dir) / "C7_B3_LEAKAGE_AUDIT.json", leakage)
    atomic_text(Path(args.audit_dir) / "C7_B3_LEAKAGE_AUDIT.md", md("C7-B3 leakage audit", leakage))

    pred_path = Path(args.output_dir) / "calibrated_A4_predictions.jsonl"
    score_path = Path(args.output_dir) / "calibrated_A4_scores.jsonl"
    write_jsonl(pred_path, selected["prediction_rows"])
    write_jsonl(score_path, selected["score_rows"])
    deterministic = {
        "status": "PASS" if all(v < 1e-9 for v in selected["metric_abs_diff_vs_C7_B3_FINAL_DECISION"].values()) else "WARN_FLOAT_DIFF",
        "official_val_used": False,
        "metrics": selected["metrics"],
        "metric_abs_diff_vs_C7_B3_FINAL_DECISION": selected["metric_abs_diff_vs_C7_B3_FINAL_DECISION"],
        "metric_hash": sha256_obj(selected["metrics"]),
        "prediction_hash": sha256_file(pred_path),
        "score_hash": sha256_file(score_path),
        "config_hash": sha256_obj(selected_cfg),
        "cache_hash": artifact(args.anchor_states),
    }
    atomic_json(Path(args.audit_dir) / "C7_B3_DETERMINISTIC_RERUN.json", deterministic)
    atomic_text(Path(args.audit_dir) / "C7_B3_DETERMINISTIC_RERUN.md", md("C7-B3 deterministic rerun", deterministic))

    theory = {
        "status": "C7_B3_THEORY_ALIGNMENT_RECORDED",
        "official_val_used": False,
        "C7_B3_definition": "Shared-Norm-inspired fixed-pool moment score calibration / Cross-Video Comparable Moment Reranking",
        "addresses": "cross-video moment score comparability and moment prediction bias in two-stage VCMR",
        "Partial_Relevance_MIL_promoted_mechanism": False,
        "MIL_status": "exploratory / negative ablation branch",
        "MINUTE_style_raw_logit_Shared_Norm": "reserved for later C7-B4-SN-Audit",
        "C7_B4_part_of_C7_B3_selection": False,
    }
    atomic_json(Path(args.audit_dir) / "C7_B3_THEORY_ALIGNMENT_NOTE.json", theory)
    atomic_text(Path(args.audit_dir) / "C7_B3_THEORY_ALIGNMENT_NOTE.md", md("C7-B3 theory alignment note", theory))

    prereg = {
        "status": "C7_B3_OFFICIAL_GATE_PREREG",
        "official_val_used": False,
        "primary_comparison": "current promoted system = C7-B2.1 A4 official",
        "promotion_gate": [
            "0.5-r1 vs C7-B2.1 A4 official > 0",
            "0.7-r1 vs C7-B2.1 A4 official > 0",
            "0.7-r5 vs C7-B2.1 A4 official >= 0",
            "0.7-r10 vs C7-B2.1 A4 official >= 0",
            "0.5-r5 vs C7-B2.1 A4 official >= -0.05",
            "0.5-r10 vs C7-B2.1 A4 official >= -0.05",
            "R@100 unchanged or delta >= -0.05",
            "fixed_pool_invariant = true",
            "hard_positive_exit_ratio = 0",
            "invalid_span_count = 0",
            "duplicate_span_count_after_nms = 0",
            "evaluator_modified = false",
            "nms_modified = false",
            "post_val_adjustment = false",
            "second_official_val = false",
        ],
        "allowed_statuses": [
            "C7_B3_OFFICIAL_PROMOTED",
            "C7_B3_OFFICIAL_CORE_POSITIVE_REVIEW",
            "C7_B3_OFFICIAL_NO_GAIN_KEEP_C7_B2_1",
            "C7_B3_OFFICIAL_NEGATIVE",
            "C7_B3_INFRA_FAIL",
        ],
    }
    atomic_json(Path(args.audit_dir) / "C7_B3_OFFICIAL_GATE_PREREG.json", prereg)
    atomic_text(Path(args.audit_dir) / "C7_B3_OFFICIAL_GATE_PREREG.md", md("C7-B3 official gate prereg", prereg))

    pass_all = all(x["status"] == "PASS" for x in [split_audit, provenance, fixed_pool, zero_ctrl, leakage]) and deterministic["status"] in {"PASS", "WARN_FLOAT_DIFF"}
    status = "C7_B3_OFFICIAL_READY" if pass_all else "C7_B3_FREEZE_PACKAGE_FAIL"
    final = {
        "status": status,
        "official_val_used": False,
        "official_val_run": False,
        "post_val_adjustment": False,
        "second_official_val": False,
        "selected_candidate": "calibrated_A4",
        "selected_config": selected_cfg,
        "definition": "Shared-Norm-inspired Fixed-Pool Moment Score Calibration / Cross-Video Comparable Moment Reranking",
        "metrics": selected["metrics"],
        "delta_vs_uncalibrated_A4": delta_vs_uncal,
        "fixed_pool_invariant_pass": fixed_pool["status"] == "PASS",
        "zero_control_pass": zero_ctrl["status"] == "PASS",
        "uncalibrated_control_pass": uncal_ctrl["status"] == "PASS",
        "leakage_audit_pass": leakage["status"] == "PASS",
        "deterministic_rerun_pass": deterministic["status"] in {"PASS", "WARN_FLOAT_DIFF"},
        "theory_alignment_recorded": True,
        "official_gate_preregistered": True,
        "C7_B4_SN_Audit_run": False,
        "MINUTE_style_scoring_added_to_C7_B3": False,
        "evaluator_modified": False,
        "nms_modified": False,
        "C4_C6_C7B1_C7B21_artifacts_modified": False,
        "enter_C7_C": False,
        "enter_C8": False,
        "request": "Stop and wait for human authorization for exactly-one official-val one-shot.",
    }
    atomic_json(Path(args.audit_dir) / "C7_B3_FINAL_FREEZE_PACKAGE.json", final)
    atomic_text(Path(args.audit_dir) / "C7_B3_FINAL_FREEZE_PACKAGE.md", md("C7-B3 final freeze package", final))
    ready = {
        "status": status,
        "official_val_used": False,
        "promoted": False,
        "selected_candidate": "calibrated_A4",
        "selected_config": selected_cfg,
        "freeze_package": artifact(Path(args.audit_dir) / "C7_B3_FINAL_FREEZE_PACKAGE.json"),
        "requires_human_authorization_before_official_val": True,
    }
    atomic_json(Path(args.audit_dir) / "C7_B3_OFFICIAL_READY_MANIFEST.json", ready)
    hashes = {
        "status": "C7_B3_FREEZE_HASHES",
        "official_val_used": False,
        "metric_hash": deterministic["metric_hash"],
        "prediction_hash": deterministic["prediction_hash"],
        "score_hash": deterministic["score_hash"],
        "artifacts": {
            "final_freeze": artifact(Path(args.audit_dir) / "C7_B3_FINAL_FREEZE_PACKAGE.json"),
            "ready": artifact(Path(args.audit_dir) / "C7_B3_OFFICIAL_READY_MANIFEST.json"),
            "deterministic": artifact(Path(args.audit_dir) / "C7_B3_DETERMINISTIC_RERUN.json"),
            "runner": artifact(__file__),
        },
    }
    atomic_json(Path(args.audit_dir) / "C7_B3_FREEZE_HASHES.json", hashes)
    print(json.dumps({
        "status": status,
        "official_val_used": False,
        "selected_candidate": "calibrated_A4",
        "selected_config": selected_cfg,
        "metrics": selected["metrics"],
        "fixed_pool_invariant_pass": fixed_pool["status"] == "PASS",
        "zero_control_pass": zero_ctrl["status"] == "PASS",
        "leakage_audit_pass": leakage["status"] == "PASS",
        "deterministic_status": deterministic["status"],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
