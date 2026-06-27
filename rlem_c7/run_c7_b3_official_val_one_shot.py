#!/usr/bin/env python
"""Exactly-one official-val one-shot for frozen C7-B3.

C7-B3 is a fixed-pool calibration of the C7-B2.1 A4 video residual:
  score = C7-B1 frozen anchor score + lambda * calibrated(video_residual)

This script only reads frozen official artifacts and calls the official
evaluator once for the generated C7-B3 submission.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rlem.c4_lite_utils import flatten_eval_metrics  # noqa: E402
from rlem_c7.run_c7_b0_b1 import (  # noqa: E402
    atomic_json,
    atomic_npz,
    atomic_text,
    artifact,
    labels_for_sequence,
    load_npz,
    metric_delta,
    nms_sequence,
)
from rlem_c7.run_c7_b2_1_safety_repair import group_score_map, normalize_cand  # noqa: E402
from rlem_c7.run_c7_b3_stable_generalization import transform_residual  # noqa: E402
from standalone_eval.eval import eval_retrieval, load_jsonl  # noqa: E402


METRIC_KEYS = [
    "0.5-r1", "0.5-r5", "0.5-r10", "0.5-r100",
    "0.7-r1", "0.7-r5", "0.7-r10", "0.7-r100",
]
CLIP = 1.5
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


def load_metrics(path: str | Path, preferred_key: str | None = None) -> Dict[str, float]:
    obj = json.loads(Path(path).read_text(encoding="utf-8"))
    if preferred_key and preferred_key in obj:
        flat = obj[preferred_key]
    elif "C7_B1_official_metrics" in obj:
        flat = obj["C7_B1_official_metrics"]
    elif "C7_B2_1_A4_official_metrics" in obj:
        flat = obj["C7_B2_1_A4_official_metrics"]
    elif "VCMR" in obj:
        flat = obj["VCMR"]
    else:
        flat = flatten_eval_metrics(obj)
    return {k: float(flat[k]) for k in METRIC_KEYS}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--allow_official_val", action="store_true", required=True)
    p.add_argument("--ready_manifest", default="c7_audit/C7_B3_OFFICIAL_READY_MANIFEST.json")
    p.add_argument("--freeze_package", default="c7_audit/C7_B3_FINAL_FREEZE_PACKAGE.json")
    p.add_argument("--b21_official_manifest", default="c7_audit/C7_B2_1_OFFICIAL_MANIFEST.json")
    p.add_argument("--b21_anchor_states", default="results/rlem_c7_b2_1_official_val/aux/official_val_c7_b1_anchor_states.json")
    p.add_argument("--b21_video_scores", default="results/rlem_c7_b2_1_official_val/aux/official_val_video_scores.npz")
    p.add_argument("--b21_submission", default="results/rlem_c7_b2_1_official_val/official_val_submission.json")
    p.add_argument("--official_cache_npz", default="results/rlem_c6_b1_lite_r1_safe/official_val_c6_cache.npz")
    p.add_argument("--c6_b1_metrics", default="results/rlem_c6_b1_lite_r1_safe/official_val_metrics.json")
    p.add_argument("--c6_b2_metrics", default="results/rlem_c6_b2_official_val/official_val_metrics.json")
    p.add_argument("--c7_b1_metrics", default="c7_audit/C7_B1_OFFICIAL_MANIFEST.json")
    p.add_argument("--b2_eval", default="c7_audit/C7_B2_TRAIN_CALIB_EVAL.json")
    p.add_argument("--gt_jsonl", default="/home/a/yybwork/data/yyb/tvr_feature_release/data/tvr_val_release.jsonl")
    p.add_argument("--output_dir", default="results/rlem_c7_b3_official_val")
    p.add_argument("--audit_dir", default="c7_audit")
    p.add_argument("--effective_top_n", type=int, default=100)
    p.add_argument("--max_after_nms", type=int, default=100)
    p.add_argument("--nms_thd", type=float, default=0.7)
    return p.parse_args()


def output_paths(args: argparse.Namespace) -> Dict[str, Path]:
    out = Path(args.output_dir)
    audit = Path(args.audit_dir)
    return {
        "scores_npz": out / "official_val_scores.npz",
        "predictions_jsonl": out / "official_val_predictions.jsonl",
        "submission_json": out / "official_val_submission.json",
        "metrics_json": out / "official_val_metrics.json",
        "one_shot_md": audit / "C7_B3_OFFICIAL_VAL_ONE_SHOT.md",
        "one_shot_json": audit / "C7_B3_OFFICIAL_VAL_ONE_SHOT.json",
        "decision_md": audit / "C7_B3_OFFICIAL_DECISION.md",
        "decision_json": audit / "C7_B3_OFFICIAL_DECISION.json",
        "manifest_json": audit / "C7_B3_OFFICIAL_MANIFEST.json",
        "hashes_json": audit / "C7_B3_OFFICIAL_HASHES.json",
    }


def ensure_no_prior_outputs(paths: Dict[str, Path]) -> None:
    keys = [
        "scores_npz", "predictions_jsonl", "submission_json", "metrics_json",
        "one_shot_md", "one_shot_json", "decision_md", "decision_json",
        "manifest_json", "hashes_json",
    ]
    existing = [str(paths[k]) for k in keys if paths[k].exists()]
    if existing:
        raise FileExistsError("refusing second C7-B3 official-val run; outputs already exist: " + ", ".join(existing))


def validate_ready(args: argparse.Namespace) -> Dict[str, Any]:
    ready = json.loads(Path(args.ready_manifest).read_text(encoding="utf-8"))
    freeze = json.loads(Path(args.freeze_package).read_text(encoding="utf-8"))
    if ready.get("status") != "C7_B3_OFFICIAL_READY" or freeze.get("status") != "C7_B3_OFFICIAL_READY":
        raise ValueError("C7-B3 status_before_run is not OFFICIAL_READY")
    if ready.get("official_val_used") is not False or freeze.get("official_val_used") is not False:
        raise ValueError("C7-B3 freeze state already indicates official_val_used")
    if ready.get("selected_candidate") != "calibrated_A4" or freeze.get("selected_candidate") != "calibrated_A4":
        raise ValueError("selected_candidate mismatch")
    if ready.get("selected_config") != SELECTED_CONFIG or freeze.get("selected_config") != SELECTED_CONFIG:
        raise ValueError("selected_config mismatch")
    b21 = json.loads(Path(args.b21_official_manifest).read_text(encoding="utf-8"))
    if b21.get("status") != "C7_B2_1_OFFICIAL_PROMOTED" or b21.get("official_val_used") is not True:
        raise ValueError("C7-B2.1 A4 official promoted anchor is not available")
    return {"ready": ready, "freeze": freeze, "b21": b21}


def cand_key(c: Candidate) -> Tuple[int, int, int]:
    return int(c[1]), int(c[2]), int(c[3])


def selected_sequences(args: argparse.Namespace, paths: Dict[str, Path], cache: Dict[str, np.ndarray], c7b21_submission: Dict[str, Any]) -> Dict[str, Any]:
    states = json.loads(Path(args.b21_anchor_states).read_text(encoding="utf-8"))["states"]
    score_arrays = load_npz(args.b21_video_scores, allow_pickle=False)
    cfg = json.loads(Path(args.b2_eval).read_text(encoding="utf-8"))["best"]["config"]
    gscore = group_score_map(score_arrays, cfg)
    video_weight = float(cfg["video_weight"])
    gt_vid = cache["video_idx"][cache["desc_offsets"][:-1]].astype(np.int64)
    gt_s = cache["gt_start_idx"][cache["desc_offsets"][:-1]].astype(np.int64)
    gt_e = cache["gt_end_idx"][cache["desc_offsets"][:-1]].astype(np.int64)
    predictions_jsonl: List[Dict[str, Any]] = []
    policy_predictions = np.full((len(states), args.max_after_nms, 4), -1, dtype=np.float32)
    final_scores = np.full((len(states), args.effective_top_n), np.nan, dtype=np.float32)
    invalid = dup = exits = entries = denom_pos = 0
    fixed_pool = True
    for q, state in enumerate(states):
        anchor = [normalize_cand(x) for x in state["flat_candidates"][:args.effective_top_n]]
        base_score = np.asarray([float(c[4]) for c in anchor], dtype=np.float32)
        residual = np.asarray(
            [video_weight * gscore.get(int(c[0]), (0.0, 0.0, 0.0, 0.0))[2] for c in anchor],
            dtype=np.float32,
        )
        b21_scored = [(c[0], c[1], c[2], c[3], float(s), c[5]) for c, s in zip(anchor, base_score + residual)]
        b21_scored = sorted(b21_scored, key=lambda c: -c[4])
        calibrated = transform_residual(
            residual,
            SELECTED_CONFIG["normalization"],
            float(SELECTED_CONFIG["T"]),
            float(SELECTED_CONFIG["clip"]),
        )
        b3_scored = [(c[0], c[1], c[2], c[3], float(s), c[5]) for c, s in zip(anchor, base_score + float(SELECTED_CONFIG["lambda"]) * calibrated)]
        b3_scored = sorted(b3_scored, key=lambda c: -c[4])
        fixed_pool = fixed_pool and sorted(cand_key(c) for c in b21_scored) == sorted(cand_key(c) for c in b3_scored)
        b21_post = nms_sequence(b21_scored[:args.effective_top_n], args.nms_thd, args.max_after_nms)
        b3_post = nms_sequence(b3_scored[:args.effective_top_n], args.nms_thd, args.max_after_nms)
        _b05, _b07, bpos, _bi, _bd = labels_for_sequence(b21_post, int(gt_vid[q]), int(gt_s[q]), int(gt_e[q]))
        _l05, _l07, pos, inv, du = labels_for_sequence(b3_post, int(gt_vid[q]), int(gt_s[q]), int(gt_e[q]))
        denom_pos += int(bpos)
        exits += int(bpos and not pos)
        entries += int((not bpos) and pos)
        invalid += inv
        dup += du
        preds = []
        for r, c in enumerate(b3_post):
            preds.append([int(c[1]), float(c[2] * CLIP), float((c[3] + 1) * CLIP), float(c[4])])
            policy_predictions[q, r] = np.asarray([c[1], c[2], c[3], c[4]], dtype=np.float32)
        for r, c in enumerate(b3_scored[:args.effective_top_n]):
            final_scores[q, r] = float(c[4])
        desc_id = cache["desc_ids"][q].item() if hasattr(cache["desc_ids"][q], "item") else cache["desc_ids"][q]
        predictions_jsonl.append({"desc_id": desc_id, "desc": str(cache["desc_text"][q]), "predictions": preds})
    submission = {"video2idx": c7b21_submission["video2idx"], "VCMR": predictions_jsonl}
    atomic_json(paths["submission_json"], submission)
    write_jsonl(paths["predictions_jsonl"], predictions_jsonl)
    atomic_npz(
        paths["scores_npz"],
        policy_predictions=policy_predictions,
        final_score=final_scores,
        T=np.asarray([SELECTED_CONFIG["T"]], dtype=np.float32),
        lambda_=np.asarray([SELECTED_CONFIG["lambda"]], dtype=np.float32),
        clip=np.asarray([SELECTED_CONFIG["clip"]], dtype=np.float32),
        video_weight=np.asarray([video_weight], dtype=np.float32),
    )
    return {
        "submission": submission,
        "fixed_pool_invariant_on_official": bool(fixed_pool),
        "hard_positive_top100_query_exits": int(exits),
        "hard_positive_top100_query_entries": int(entries),
        "hard_positive_exit_ratio": float(exits / max(denom_pos, 1)),
        "hard_positive_reference": "C7-B2.1 A4 post-NMS top100 within identical candidate pool",
        "invalid_span_count": int(invalid),
        "duplicate_span_count_after_nms": int(dup),
        "queries": int(len(states)),
        "selected_config": SELECTED_CONFIG,
    }


def classify(delta_vs_b21: Dict[str, float], structural: Dict[str, Any]) -> str:
    infra_ok = (
        structural["fixed_pool_invariant_on_official"]
        and structural["hard_positive_exit_ratio"] == 0.0
        and structural["invalid_span_count"] == 0
        and structural["duplicate_span_count_after_nms"] == 0
    )
    gate_ok = (
        delta_vs_b21["0.5-r1"] > 0.0
        and delta_vs_b21["0.7-r1"] > 0.0
        and delta_vs_b21["0.7-r5"] >= 0.0
        and delta_vs_b21["0.7-r10"] >= 0.0
        and delta_vs_b21["0.5-r5"] >= -0.05
        and delta_vs_b21["0.5-r10"] >= -0.05
        and delta_vs_b21["0.5-r100"] >= -0.05
        and delta_vs_b21["0.7-r100"] >= -0.05
    )
    if not infra_ok:
        return "C7_B3_INFRA_FAIL"
    if gate_ok:
        return "C7_B3_OFFICIAL_PROMOTED"
    if delta_vs_b21["0.5-r1"] > 0.0 or delta_vs_b21["0.7-r1"] > 0.0:
        return "C7_B3_OFFICIAL_CORE_POSITIVE_REVIEW"
    if all(delta_vs_b21[k] <= 0.0 for k in ["0.5-r1", "0.7-r1", "0.7-r5", "0.7-r10"]):
        return "C7_B3_OFFICIAL_NO_GAIN_KEEP_C7_B2_1"
    return "C7_B3_OFFICIAL_NEGATIVE"


def metric_table(rows: Dict[str, Dict[str, float]]) -> str:
    out = [
        "| system | 0.5-r1 | 0.5-r5 | 0.5-r10 | 0.5-r100 | 0.7-r1 | 0.7-r5 | 0.7-r10 | 0.7-r100 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, vals in rows.items():
        fmt = "{:+.2f}" if name.startswith("Delta") else "{:.2f}"
        out.append("| " + name + " | " + " | ".join(fmt.format(vals[k]) for k in METRIC_KEYS) + " |")
    return "\n".join(out)


def main() -> None:
    args = parse_args()
    if not args.allow_official_val:
        raise ValueError("--allow_official_val is required")
    paths = output_paths(args)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.audit_dir).mkdir(parents=True, exist_ok=True)
    ensure_no_prior_outputs(paths)
    ready_state = validate_ready(args)
    cache = load_npz(args.official_cache_npz, allow_pickle=True)
    c7b21_submission = json.loads(Path(args.b21_submission).read_text(encoding="utf-8"))
    structural = selected_sequences(args, paths, cache, c7b21_submission)

    # Official evaluator: exactly one call in this script.
    metrics_raw = eval_retrieval(
        structural["submission"],
        load_jsonl(args.gt_jsonl),
        iou_thds=(0.5, 0.7),
        verbose=False,
        match_number=True,
        use_desc_type=True,
    )
    atomic_json(paths["metrics_json"], metrics_raw)
    c7b3_metrics = {k: float(flatten_eval_metrics(metrics_raw)[k]) for k in METRIC_KEYS}
    c6b1_metrics = load_metrics(args.c6_b1_metrics)
    c6b2_metrics = load_metrics(args.c6_b2_metrics)
    c7b1_metrics = load_metrics(args.c7_b1_metrics, "C7_B1_official_metrics")
    c7b21_metrics = load_metrics(args.b21_official_manifest, "C7_B2_1_A4_official_metrics")
    delta_vs_c6b1 = metric_delta(c7b3_metrics, c6b1_metrics)
    delta_vs_c6b2 = metric_delta(c7b3_metrics, c6b2_metrics)
    delta_vs_c7b1 = metric_delta(c7b3_metrics, c7b1_metrics)
    delta_vs_b21 = metric_delta(c7b3_metrics, c7b21_metrics)
    status = classify(delta_vs_b21, structural)
    common = {
        "stage": "C7-B3",
        "status": status,
        "status_before_run": "C7_B3_OFFICIAL_READY",
        "selected_candidate": "calibrated_A4",
        "method_definition": "Shared-Norm-inspired Fixed-Pool Moment Score Calibration / Cross-Video Comparable Moment Reranking",
        "mechanism": "calibrated_A4 residual rerank within C7-B2.1 A4 fixed-pool candidate set",
        "selected_config": SELECTED_CONFIG,
        "date": datetime.now().strftime("%Y-%m-%d"),
        "official_val_used": True,
        "official_val_run": True,
        "official_evaluator_call_count": 1,
        "post_val_adjustment": False,
        "second_official_val": False,
        "training_started": False,
        "config_selection_changed": False,
        "selected_candidate_changed": False,
        "C7_B4_SN_Audit_run": False,
        "MINUTE_style_scoring_added": False,
        "enter_C7_C": False,
        "enter_C8": False,
        "evaluator_modified": False,
        "nms_modified": False,
        "C4_C6_C7B1_C7B21_artifacts_modified": False,
        "C6_B1_official_metrics": c6b1_metrics,
        "C6_B2_official_metrics": c6b2_metrics,
        "C7_B1_official_archived_metrics": c7b1_metrics,
        "C7_B2_1_A4_official_metrics": c7b21_metrics,
        "C7_B3_official_metrics": c7b3_metrics,
        "delta_vs_C6_B1": delta_vs_c6b1,
        "delta_vs_C6_B2": delta_vs_c6b2,
        "delta_vs_C7_B1": delta_vs_c7b1,
        "delta_vs_C7_B2_1_A4": delta_vs_b21,
        "fixed_pool_invariant_on_official": structural["fixed_pool_invariant_on_official"],
        "hard_positive_top100_query_exits": structural["hard_positive_top100_query_exits"],
        "hard_positive_top100_query_entries": structural["hard_positive_top100_query_entries"],
        "hard_positive_exit_ratio": structural["hard_positive_exit_ratio"],
        "invalid_span_count": structural["invalid_span_count"],
        "duplicate_span_count_after_nms": structural["duplicate_span_count_after_nms"],
        "ready_state": {
            "ready_manifest": artifact(args.ready_manifest),
            "freeze_package": artifact(args.freeze_package),
            "b21_official_manifest": artifact(args.b21_official_manifest),
        },
        "final_instruction": "Stop after exactly-one official-val one-shot and wait for human review.",
    }
    atomic_json(paths["one_shot_json"], common)
    atomic_json(paths["manifest_json"], {**common, "promoted": status == "C7_B3_OFFICIAL_PROMOTED"})
    decision = {
        "status": status,
        "promoted": status == "C7_B3_OFFICIAL_PROMOTED",
        "official_val_used": True,
        "post_val_adjustment": False,
        "second_official_val": False,
        "C7_B3_official_metrics": c7b3_metrics,
        "delta_vs_C6_B1": delta_vs_c6b1,
        "delta_vs_C6_B2": delta_vs_c6b2,
        "delta_vs_C7_B1": delta_vs_c7b1,
        "delta_vs_C7_B2_1_A4": delta_vs_b21,
        "fixed_pool_invariant_on_official": structural["fixed_pool_invariant_on_official"],
        "official_gate_reference": "C7_B3_OFFICIAL_GATE_PREREG",
        "final_instruction": "Stop after exactly-one official-val one-shot and wait for human review.",
    }
    atomic_json(paths["decision_json"], decision)
    table = metric_table({
        "C6-B1 official": c6b1_metrics,
        "C6-B2 official": c6b2_metrics,
        "C7-B1 official archived": c7b1_metrics,
        "C7-B2.1 A4 official": c7b21_metrics,
        "C7-B3 official": c7b3_metrics,
        "Delta vs C6-B1": delta_vs_c6b1,
        "Delta vs C6-B2": delta_vs_c6b2,
        "Delta vs C7-B1": delta_vs_c7b1,
        "Delta vs C7-B2.1 A4": delta_vs_b21,
    })
    md = f"""# C7-B3 official-val one-shot

- Status: `{status}`
- selected_candidate: `calibrated_A4`
- selected_config: `{json.dumps(SELECTED_CONFIG, sort_keys=True)}`
- official_val_used: `true`
- official evaluator calls: `1`
- post_val_adjustment / second_official_val: `false / false`
- evaluator_modified / nms_modified: `false / false`
- C7_B4_SN_Audit_run / MINUTE_style_scoring_added: `false / false`
- fixed_pool_invariant_on_official: `{structural['fixed_pool_invariant_on_official']}`
- hard_positive exits/entries/ratio: `{structural['hard_positive_top100_query_exits']} / {structural['hard_positive_top100_query_entries']} / {structural['hard_positive_exit_ratio']}`
- invalid_span_count / duplicate_span_count_after_nms: `{structural['invalid_span_count']} / {structural['duplicate_span_count_after_nms']}`

## Metrics

{table}

Stop here and wait for human review.
"""
    atomic_text(paths["one_shot_md"], md)
    atomic_text(paths["decision_md"], "# C7-B3 official decision\n\n" + md)
    hashes = {
        "status": "C7_B3_OFFICIAL_HASHES",
        "official_val_used": True,
        "post_val_adjustment": False,
        "second_official_val": False,
        "metric_hash": sha256_obj(c7b3_metrics),
        "prediction_hash": sha256_file(paths["predictions_jsonl"]),
        "score_hash": sha256_file(paths["scores_npz"]),
        "artifacts": {
            "one_shot_json": artifact(paths["one_shot_json"]),
            "one_shot_md": artifact(paths["one_shot_md"]),
            "decision_json": artifact(paths["decision_json"]),
            "decision_md": artifact(paths["decision_md"]),
            "manifest_json": artifact(paths["manifest_json"]),
            "scores_npz": artifact(paths["scores_npz"]),
            "predictions_jsonl": artifact(paths["predictions_jsonl"]),
            "submission_json": artifact(paths["submission_json"]),
            "metrics_json": artifact(paths["metrics_json"]),
            "runner": artifact(__file__),
            "ready_manifest": artifact(args.ready_manifest),
            "freeze_package": artifact(args.freeze_package),
            "b21_anchor_states": artifact(args.b21_anchor_states),
            "b21_video_scores": artifact(args.b21_video_scores),
        },
    }
    atomic_json(paths["hashes_json"], hashes)
    print(json.dumps({
        "status": status,
        "C6_B1_official_metrics": c6b1_metrics,
        "C6_B2_official_metrics": c6b2_metrics,
        "C7_B1_official_archived_metrics": c7b1_metrics,
        "C7_B2_1_A4_official_metrics": c7b21_metrics,
        "C7_B3_official_metrics": c7b3_metrics,
        "delta_vs_C6_B1": delta_vs_c6b1,
        "delta_vs_C6_B2": delta_vs_c6b2,
        "delta_vs_C7_B1": delta_vs_c7b1,
        "delta_vs_C7_B2_1_A4": delta_vs_b21,
        "fixed_pool_invariant_on_official": structural["fixed_pool_invariant_on_official"],
        "hard_positive_top100_query_exits": structural["hard_positive_top100_query_exits"],
        "hard_positive_top100_query_entries": structural["hard_positive_top100_query_entries"],
        "hard_positive_exit_ratio": structural["hard_positive_exit_ratio"],
        "invalid_span_count": structural["invalid_span_count"],
        "duplicate_span_count_after_nms": structural["duplicate_span_count_after_nms"],
        "evaluator_modified": False,
        "nms_modified": False,
        "official_val_used": True,
        "post_val_adjustment": False,
        "second_official_val": False,
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
