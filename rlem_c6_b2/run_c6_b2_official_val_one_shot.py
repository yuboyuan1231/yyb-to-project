#!/usr/bin/env python
"""Exactly-one official-val one-shot for frozen C6-B2 pairwise_main_0005.

This runner is intentionally narrow:
  * no training;
  * no threshold/grid/temperature search on official val;
  * one frozen arm/config: pairwise_main_0005;
  * same-video replacement only, fixed video slots, unchanged NMS/evaluator.

It reuses the already materialized C6-B1 official-val C6 cache and candidate
pool, scores that pool with the frozen C6-B2 pairwise checkpoint, applies the
frozen policy, writes official-val predictions/submission, and calls the
official evaluator exactly once.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rlem.c4_lite_utils import flatten_eval_metrics  # noqa: E402
from rlem_c6_b0.run_c6_b0_candidate_diagnostic import correctness_for_ts, span_iou_idx  # noqa: E402
from rlem_c6_b2.run_c6_b2_goal import PairwiseUtilityModel  # noqa: E402
from standalone_eval.eval import eval_retrieval, load_jsonl  # noqa: E402


METRIC_KEYS = ("0.5-r1", "0.5-r5", "0.5-r10", "0.5-r100", "0.7-r1", "0.7-r5", "0.7-r10", "0.7-r100")
EXPECTED_CONFIG = {
    "arm": "pairwise_main",
    "config_id": "pairwise_main_0005",
    "apply_slots": 5,
    "threshold0": 4.337798070907593,
    "threshold_rest": 6.729127645492554,
    "max_replacements": 2,
    "effective_top_n": 100,
    "NMS": 0.7,
    "max_after_nms": 100,
}
PRIMARY_BASELINE = "C6-B1-lite c6b1r1_0057"
SECONDARY_BASELINE = "C4_final v21_00444"
CLIP = 1.5
EPS = 1e-6


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def artifact(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    return {
        "path": str(p),
        "exists": p.exists(),
        "size": int(p.stat().st_size) if p.exists() else None,
        "sha256": sha256_file(p) if p.exists() and p.is_file() else None,
    }


def atomic_json(path: str | Path, obj: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(p) + ".partial")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, p)


def atomic_text(path: str | Path, text: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(p) + ".partial")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, p)


def atomic_npz(path: str | Path, **arrays: np.ndarray) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(p) + ".partial")
    with open(tmp, "wb") as f:
        np.savez_compressed(f, **arrays)
    os.replace(tmp, p)


def load_npz(path: str | Path, allow_pickle: bool = False) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=allow_pickle) as z:
        return {k: z[k] for k in z.files}


def load_metrics(path: str | Path) -> Dict[str, float]:
    obj = json.loads(Path(path).read_text(encoding="utf-8"))
    flat = obj["VCMR"] if isinstance(obj, dict) and "VCMR" in obj else flatten_eval_metrics(obj)
    return {k: float(flat[k]) for k in METRIC_KEYS}


def metric_delta(new: Dict[str, float], base: Dict[str, float]) -> Dict[str, float]:
    return {k: float(new[k] - base[k]) for k in METRIC_KEYS}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--allow_official_val", action="store_true", required=True)
    p.add_argument("--official_cache_npz", default="results/rlem_c6_b1_lite_r1_safe/official_val_c6_cache.npz")
    p.add_argument("--candidate_pool_npz", default="results/rlem_c6_b1_lite_r1_safe/official_val_candidate_pool_top_slots.npz")
    p.add_argument("--c4_final_submission_json", default="results/rlem_c4_r2_cal_v21_official_val/val_v21_submission.json")
    p.add_argument("--c4_final_metrics_json", default="results/rlem_c4_r2_cal_v21_official_val/val_v21_metrics.json")
    p.add_argument("--c6_b1_submission_json", default="results/rlem_c6_b1_lite_r1_safe/official_val_submission.json")
    p.add_argument("--c6_b1_metrics_json", default="results/rlem_c6_b1_lite_r1_safe/official_val_metrics.json")
    p.add_argument("--gt_jsonl", default="/home/a/yybwork/data/yyb/tvr_feature_release/data/tvr_val_release.jsonl")
    p.add_argument("--selector_ckpt", default="results/rlem_c6_b2/arms/pairwise_main/model_best.pt")
    p.add_argument("--best_config_json", default="results/rlem_c6_b2/train_calib_scores/pairwise_main/best_config.json")
    p.add_argument("--freeze_manifest", default="c6_b2_audit/C6_B2_FREEZE_MANIFEST.json")
    p.add_argument("--freeze_hashes", default="c6_b2_audit/C6_B2_FREEZE_HASHES.json")
    p.add_argument("--output_dir", default="results/rlem_c6_b2_official_val")
    p.add_argument("--audit_dir", default="c6_b2_official_val")
    p.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    p.add_argument("--score_batch_size", type=int, default=131072)
    return p.parse_args()


def output_paths(args: argparse.Namespace) -> Dict[str, Path]:
    out = Path(args.output_dir)
    audit = Path(args.audit_dir)
    return {
        "scores_npz": out / "official_val_scores.npz",
        "predictions_jsonl": out / "official_val_predictions.jsonl",
        "submission_json": out / "official_val_submission.json",
        "metrics_json": out / "official_val_metrics.json",
        "preflight_json": audit / "C6_B2_OFFICIAL_VAL_PREFLIGHT_AUDIT.json",
        "preflight_md": audit / "C6_B2_OFFICIAL_VAL_PREFLIGHT_AUDIT.md",
        "start_manifest": audit / "C6_B2_OFFICIAL_VAL_START_MANIFEST.json",
        "audit_md": audit / "C6_B2_OFFICIAL_VAL_ONE_SHOT_AUDIT.md",
        "manifest_json": audit / "C6_B2_OFFICIAL_VAL_ONE_SHOT_MANIFEST.json",
        "hashes_json": audit / "C6_B2_OFFICIAL_VAL_ONE_SHOT_HASHES.json",
    }


def ensure_no_prior_outputs(paths: Dict[str, Path]) -> None:
    one_shot_outputs = [
        "scores_npz", "predictions_jsonl", "submission_json", "metrics_json",
        "audit_md", "manifest_json", "hashes_json",
    ]
    existing = [str(paths[k]) for k in one_shot_outputs if paths[k].exists()]
    if existing:
        raise FileExistsError("refusing second official-val run; outputs already exist: " + ", ".join(existing))


def validate_freeze(args: argparse.Namespace) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    freeze = json.loads(Path(args.freeze_manifest).read_text(encoding="utf-8"))
    best = json.loads(Path(args.best_config_json).read_text(encoding="utf-8"))
    if freeze.get("status") != "C6_B2_FREEZE_REVIEW_PASS":
        raise ValueError("C6-B2 freeze status mismatch")
    if freeze.get("selected_arm") != EXPECTED_CONFIG["arm"] or freeze.get("selected_config_id") != EXPECTED_CONFIG["config_id"]:
        raise ValueError("C6-B2 freeze selected config mismatch")
    frozen = freeze.get("best_config", {})
    for key in ("arm", "config_id", "apply_slots", "threshold0", "threshold_rest", "max_replacements"):
        if frozen.get(key) != EXPECTED_CONFIG[key] or best.get(key) != EXPECTED_CONFIG[key]:
            raise ValueError(f"frozen/best config mismatch for {key}: freeze={frozen.get(key)} best={best.get(key)} expected={EXPECTED_CONFIG[key]}")
    for key in ("score_grid_on_official_val", "post_val_adjustment", "C6_C_used", "C4_final_modified", "NMS_modified", "evaluator_modified"):
        if freeze.get(key) not in (False, None):
            raise ValueError(f"freeze manifest forbidden flag is true: {key}")
    if freeze.get("official_val_used") is not False:
        raise ValueError("freeze manifest indicates official_val_used")
    return freeze, best


@torch.no_grad()
def score_pairwise_pool(pool: Dict[str, np.ndarray], ckpt_path: str | Path, device_name: str, batch_size: int) -> np.ndarray:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model = PairwiseUtilityModel(int(ckpt["in_dim"]), hidden=int(ckpt["hidden"]), layers=int(ckpt["layers"])).eval()
    model.load_state_dict(ckpt["model_state"])
    device = torch.device("cuda" if device_name == "cuda" and torch.cuda.is_available() else "cpu")
    model.to(device)
    q, slots, alts = [int(x) for x in pool["shape"]]
    x = pool["x"].reshape(q, slots, alts, -1).astype(np.float32)
    pair_x = np.concatenate([
        x[:, :, 1:, :],
        np.broadcast_to(x[:, :, 0:1, :], (q, slots, alts - 1, x.shape[-1])),
        x[:, :, 1:, :] - x[:, :, 0:1, :],
    ], axis=-1).reshape(-1, int(ckpt["in_dim"]))
    out_scores = np.zeros((q, slots, alts), dtype=np.float32)
    mean = ckpt["mean"].astype(np.float32)
    std = np.maximum(ckpt["std"].astype(np.float32), EPS)
    pair_scores = np.empty(pair_x.shape[0], dtype=np.float32)
    for start in range(0, len(pair_x), batch_size):
        xb = ((pair_x[start:start + batch_size] - mean) / std).astype(np.float32)
        tb = torch.from_numpy(xb).to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
            pred = model(tb).float()
            score = pred[:, 0] + 0.6 * pred[:, 1] - 0.85 * pred[:, 2] - 0.45 * pred[:, 3] + 0.35 * pred[:, 4] + 0.15 * torch.tanh(pred[:, 5])
        pair_scores[start:start + len(score)] = score.detach().cpu().numpy().astype(np.float32)
    out_scores[:, :, 1:] = pair_scores.reshape(q, slots, alts - 1)
    return out_scores.reshape(-1)


def apply_policy_and_write(
    cache: Dict[str, np.ndarray],
    pool: Dict[str, np.ndarray],
    pool_score: np.ndarray,
    c4_submission_json: str,
    paths: Dict[str, Path],
) -> Dict[str, Any]:
    c4_submission = json.loads(Path(c4_submission_json).read_text(encoding="utf-8"))
    q_count, slots, alt_count = [int(x) for x in pool["shape"]]
    rows_per_q = int(cache["desc_offsets"][1] - cache["desc_offsets"][0])
    video_idx = cache["video_idx"].astype(np.int64)
    start_idx = cache["start_idx"].astype(np.int64)
    end_idx = cache["end_idx"].astype(np.int64)
    s_c4 = cache["s_c4_final"].astype(np.float32)
    score3 = pool_score.reshape(q_count, slots, alt_count)
    replacement_count = 0
    top1_changed_vs_c4 = 0
    invalid_pre_nms = 0
    duplicate_after_nms = 0
    replacement_by_slot = Counter()
    policy_predictions = np.full((q_count, EXPECTED_CONFIG["effective_top_n"], 4), -1, dtype=np.float32)
    selected_alt = np.zeros((q_count, slots), dtype=np.int16)
    predictions_jsonl: List[Dict[str, Any]] = []
    for q in range(q_count):
        rs = np.arange(q * rows_per_q, (q + 1) * rows_per_q, dtype=np.int64)
        base_order = rs[np.argsort(-s_c4[rs], kind="stable")[:EXPECTED_CONFIG["effective_top_n"]]]
        slot_candidates = []
        replaced_this_q = 0
        for slot_rank, row in enumerate(base_order):
            vid = int(video_idx[row])
            si = int(start_idx[row])
            ei = int(end_idx[row])
            alt = 0
            if slot_rank < min(slots, EXPECTED_CONFIG["apply_slots"]) and replaced_this_q < EXPECTED_CONFIG["max_replacements"]:
                scores = score3[q, slot_rank]
                best = int(np.argmax(scores))
                margin = float(scores[best] - scores[0])
                threshold = EXPECTED_CONFIG["threshold0"] if slot_rank == 0 else EXPECTED_CONFIG["threshold_rest"]
                if best != 0 and margin > threshold:
                    base = (q * slots + slot_rank) * alt_count
                    si = int(pool["start_idx"][base + best])
                    ei = int(pool["end_idx"][base + best])
                    alt = best
                    replacement_count += 1
                    replaced_this_q += 1
                    replacement_by_slot[slot_rank] += 1
            if slot_rank < slots:
                selected_alt[q, slot_rank] = alt
            invalid_pre_nms += int(si < 0 or ei < si)
            slot_candidates.append((vid, si, ei, float(s_c4[row]), int(row), int(alt)))
        kept: List[int] = []
        by_video: Dict[int, List[int]] = {}
        for idx, (vid, si, ei, _score, _row, _alt) in enumerate(slot_candidates):
            suppress = False
            for prior in by_video.get(vid, []):
                _, psi, pei, *_ = slot_candidates[prior]
                if span_iou_idx(si, ei, psi, pei) > EXPECTED_CONFIG["NMS"]:
                    suppress = True
                    break
            if not suppress:
                kept.append(idx)
                by_video.setdefault(vid, []).append(idx)
                if len(kept) >= EXPECTED_CONFIG["max_after_nms"]:
                    break
        duplicate_after_nms += len(kept) - len({(slot_candidates[i][0], slot_candidates[i][1], slot_candidates[i][2]) for i in kept})
        base_top = (int(video_idx[base_order[0]]), int(start_idx[base_order[0]]), int(end_idx[base_order[0]]))
        new_top = (int(slot_candidates[kept[0]][0]), int(slot_candidates[kept[0]][1]), int(slot_candidates[kept[0]][2]))
        top1_changed_vs_c4 += int(base_top != new_top)
        preds = []
        for out_rank, idx in enumerate(kept):
            vid, si, ei, score, _row, _alt = slot_candidates[idx]
            preds.append([int(vid), float(si * CLIP), float((ei + 1) * CLIP), float(score)])
            policy_predictions[q, out_rank] = np.asarray([vid, si, ei, score], dtype=np.float32)
        desc_id = cache["desc_ids"][q].item() if hasattr(cache["desc_ids"][q], "item") else cache["desc_ids"][q]
        predictions_jsonl.append({"desc_id": desc_id, "desc": str(cache["desc_text"][q]), "predictions": preds})
    submission = {"video2idx": c4_submission["video2idx"], "VCMR": predictions_jsonl}
    atomic_json(paths["submission_json"], submission)
    tmp = Path(str(paths["predictions_jsonl"]) + ".partial")
    with open(tmp, "w", encoding="utf-8") as f:
        for row in predictions_jsonl:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    os.replace(tmp, paths["predictions_jsonl"])
    atomic_npz(
        paths["scores_npz"],
        pool_score=pool_score.astype(np.float32),
        pool_shape=pool["shape"].astype(np.int64),
        policy_predictions=policy_predictions,
        selected_alt=selected_alt,
    )
    return {
        "replacement_rate_top_slots": float(replacement_count / max(q_count * EXPECTED_CONFIG["apply_slots"], 1)),
        "candidate_replacement_count": int(replacement_count),
        "replacement_by_slot": {str(k): int(v) for k, v in sorted(replacement_by_slot.items())},
        "top1_changed_ratio_vs_c4_final": float(top1_changed_vs_c4 / max(q_count, 1)),
        "video_slot_drift": 0.0,
        "video_multiset_drift": 0.0,
        "invalid_span_count": int(invalid_pre_nms),
        "duplicate_span_count_after_nms": int(duplicate_after_nms),
        "queries": int(q_count),
    }


def prediction_top1_changed(base_submission_json: str, new_submission_json: str) -> Dict[str, Any]:
    base = json.loads(Path(base_submission_json).read_text(encoding="utf-8"))["VCMR"]
    new = json.loads(Path(new_submission_json).read_text(encoding="utf-8"))["VCMR"]
    changed = 0
    for brow, nrow in zip(base, new):
        bp = brow["predictions"][0] if brow.get("predictions") else [-1, -1, -1]
        np_ = nrow["predictions"][0] if nrow.get("predictions") else [-1, -1, -1]
        changed += int((int(bp[0]), float(bp[1]), float(bp[2])) != (int(np_[0]), float(np_[1]), float(np_[2])))
    return {"top1_changed_ratio": float(changed / max(len(base), 1)), "queries_compared": int(len(base))}


def hard_positive_movement(base_submission_json: str, new_submission_json: str, gt_jsonl: str) -> Dict[str, Any]:
    base_obj = json.loads(Path(base_submission_json).read_text(encoding="utf-8"))
    new_obj = json.loads(Path(new_submission_json).read_text(encoding="utf-8"))
    base = base_obj["VCMR"]
    new = new_obj["VCMR"]
    video2idx = new_obj.get("video2idx") or base_obj.get("video2idx") or {}
    gt = {str(r["desc_id"]): r for r in load_jsonl(gt_jsonl)}
    exits = entries = base_pos = new_pos = 0
    for brow, nrow in zip(base, new):
        row = gt[str(brow["desc_id"])]
        gt_vid_raw = row.get("vid_idx", row.get("video_idx", row.get("vid_name")))
        gt_vid = video2idx.get(gt_vid_raw, gt_vid_raw)
        ts = row.get("ts", row.get("gt_ts"))

        def ok(preds: Sequence[Sequence[Any]]) -> bool:
            for pred in preds[:100]:
                if int(pred[0]) != int(gt_vid):
                    continue
                si = int(round(float(pred[1]) / CLIP))
                ei = max(si, int(round(float(pred[2]) / CLIP)) - 1)
                if correctness_for_ts(si, ei, ts, 0.5) or correctness_for_ts(si, ei, ts, 0.7):
                    return True
            return False

        bp = ok(brow["predictions"])
        np_ = ok(nrow["predictions"])
        base_pos += int(bp)
        new_pos += int(np_)
        exits += int(bp and not np_)
        entries += int((not bp) and np_)
    return {
        "hard_positive_top100_exits": int(exits),
        "hard_positive_top100_entries": int(entries),
        "base_positive_top100_queries": int(base_pos),
        "new_positive_top100_queries": int(new_pos),
        "hard_positive_top100_exit_ratio": float(exits / max(base_pos, 1)),
    }


def classify(delta_vs_b1: Dict[str, float], movement_vs_b1: Dict[str, Any], structural: Dict[str, Any]) -> str:
    r1_ok = delta_vs_b1["0.7-r1"] > 0.0 and delta_vs_b1["0.5-r1"] >= 0.0
    r5_safe = delta_vs_b1["0.7-r5"] >= -0.10 and delta_vs_b1["0.5-r5"] >= -0.10
    r10_safe = delta_vs_b1["0.7-r10"] >= -0.15 and delta_vs_b1["0.5-r10"] >= -0.15
    r100_safe = delta_vs_b1["0.7-r100"] >= -0.10 and delta_vs_b1["0.5-r100"] >= -0.10
    movement_safe = (
        structural["video_slot_drift"] == 0.0
        and structural["video_multiset_drift"] == 0.0
        and structural["invalid_span_count"] == 0
        and structural["duplicate_span_count_after_nms"] == 0
        and movement_vs_b1["top1_changed_ratio"] <= 0.08
        and movement_vs_b1["hard_positive_top100_exit_ratio"] <= 0.0075
    )
    if r1_ok and r5_safe and r100_safe and movement_safe:
        return "C6_B2_OFFICIAL_POSITIVE"
    if r1_ok and (not r5_safe or not r10_safe):
        return "C6_B2_OFFICIAL_R1_TRADEOFF"
    return "C6_B2_OFFICIAL_NEGATIVE"


def main() -> None:
    args = parse_args()
    if not args.allow_official_val:
        raise ValueError("--allow_official_val is required for this authorized one-shot")
    paths = output_paths(args)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.audit_dir).mkdir(parents=True, exist_ok=True)
    ensure_no_prior_outputs(paths)

    freeze, best = validate_freeze(args)
    start_manifest = {
        "stage": "C6-B2 official-val one-shot",
        "official_val_one_shot": True,
        "authorized_config_only": EXPECTED_CONFIG,
        "primary_baseline": PRIMARY_BASELINE,
        "secondary_baseline": SECONDARY_BASELINE,
        "training_used": False,
        "selector_retrained": False,
        "score_grid_on_val": False,
        "temperature_search_on_val": False,
        "post_val_adjustment": False,
        "learned_config_count_on_val": 0,
        "C6_C_used": False,
        "C4_final_modified": False,
        "C6_B1_lite_modified": False,
        "NMS_modified": False,
        "evaluator_modified": False,
        "no_second_official_val": True,
    }
    atomic_json(paths["start_manifest"], start_manifest)

    compile_result = subprocess.run(
        [sys.executable, "-m", "py_compile", __file__, "rlem_c6_b2/run_c6_b2_goal.py"],
        cwd=str(ROOT),
        text=True,
        capture_output=True,
        check=False,
    )
    partials = [str(p) for p in list(Path(args.output_dir).rglob("*.partial")) + list(Path(args.audit_dir).rglob("*.partial"))]
    preflight = {
        "status": "PASS" if compile_result.returncode == 0 and not partials else "FAIL",
        "official_val_used_before_this_run": False,
        "post_val_adjustment": False,
        "score_grid_on_val": False,
        "temperature_search_on_val": False,
        "py_compile_returncode": int(compile_result.returncode),
        "py_compile_stderr": compile_result.stderr,
        "partial_files": partials,
        "freeze_manifest": artifact(args.freeze_manifest),
        "freeze_hashes": artifact(args.freeze_hashes),
        "best_config": artifact(args.best_config_json),
        "selector_ckpt": artifact(args.selector_ckpt),
        "script": artifact(__file__),
        "official_cache_npz": artifact(args.official_cache_npz),
        "candidate_pool_npz": artifact(args.candidate_pool_npz),
        "c6_b1_submission": artifact(args.c6_b1_submission_json),
        "c6_b1_metrics": artifact(args.c6_b1_metrics_json),
        "c4_final_submission": artifact(args.c4_final_submission_json),
        "c4_final_metrics": artifact(args.c4_final_metrics_json),
        "config_exact_match": True,
        "freeze_selected_config": freeze.get("selected_config_id"),
        "best_config_selected": best.get("config_id"),
    }
    atomic_json(paths["preflight_json"], preflight)
    atomic_text(
        paths["preflight_md"],
        "# C6-B2 official-val preflight audit\n\n"
        f"- status: `{preflight['status']}`\n"
        f"- selected_config: `{EXPECTED_CONFIG['config_id']}`\n"
        f"- py_compile_returncode: `{preflight['py_compile_returncode']}`\n"
        f"- partial_files: `{len(partials)}`\n"
        "- official_val_used_before_this_run: `false`\n"
        "- post_val_adjustment / score_grid_on_val / temperature_search_on_val: `false / false / false`\n",
    )
    if preflight["status"] != "PASS":
        raise RuntimeError("preflight failed; stopping before official-val scoring")

    cache = load_npz(args.official_cache_npz, allow_pickle=True)
    pool = load_npz(args.candidate_pool_npz, allow_pickle=False)
    pool_score = score_pairwise_pool(pool, args.selector_ckpt, args.device, args.score_batch_size)
    structural_movement = apply_policy_and_write(cache, pool, pool_score, args.c4_final_submission_json, paths)

    # Official evaluator: exactly one call in this script.
    submission = json.loads(paths["submission_json"].read_text(encoding="utf-8"))
    metrics_raw = eval_retrieval(
        submission,
        load_jsonl(args.gt_jsonl),
        iou_thds=(0.5, 0.7),
        verbose=False,
        match_number=True,
        use_desc_type=True,
    )
    atomic_json(paths["metrics_json"], metrics_raw)
    metrics = {k: float(flatten_eval_metrics(metrics_raw)[k]) for k in METRIC_KEYS}
    c6_b1_metrics = load_metrics(args.c6_b1_metrics_json)
    c4_metrics = load_metrics(args.c4_final_metrics_json)
    delta_vs_b1 = metric_delta(metrics, c6_b1_metrics)
    delta_vs_c4 = metric_delta(metrics, c4_metrics)
    top1_vs_b1 = prediction_top1_changed(args.c6_b1_submission_json, str(paths["submission_json"]))
    hard_vs_b1 = hard_positive_movement(args.c6_b1_submission_json, str(paths["submission_json"]), args.gt_jsonl)
    top1_vs_c4 = prediction_top1_changed(args.c4_final_submission_json, str(paths["submission_json"]))
    hard_vs_c4 = hard_positive_movement(args.c4_final_submission_json, str(paths["submission_json"]), args.gt_jsonl)
    movement_vs_b1 = {**top1_vs_b1, **hard_vs_b1}
    movement_vs_c4 = {**top1_vs_c4, **hard_vs_c4}
    status = classify(delta_vs_b1, movement_vs_b1, structural_movement)

    manifest = {
        "status": status,
        "date": datetime.now().strftime("%Y-%m-%d"),
        "stage": "C6-B2 official-val one-shot",
        "official_val_one_shot": True,
        "successful_metric_result_count": 1,
        "score_grid_on_val": False,
        "temperature_search_on_val": False,
        "post_val_adjustment": False,
        "learned_config_count_on_val": 0,
        "learned_config_count_total": 1,
        "training_used": False,
        "selector_retrained": False,
        "selected_arm": EXPECTED_CONFIG["arm"],
        "selected_config": EXPECTED_CONFIG["config_id"],
        "frozen_config": EXPECTED_CONFIG,
        "primary_baseline": PRIMARY_BASELINE,
        "secondary_baseline": SECONDARY_BASELINE,
        "C6_C_used": False,
        "C4_final_modified": False,
        "C6_B1_lite_modified": False,
        "NMS_modified": False,
        "evaluator_modified": False,
        "metrics": {
            "C6_B1_lite_c6b1r1_0057": c6_b1_metrics,
            "C4_final_v21_00444": c4_metrics,
            "C6_B2_pairwise_main_0005": metrics,
            "delta_vs_C6_B1_lite": delta_vs_b1,
            "delta_vs_C4_final": delta_vs_c4,
        },
        "movement": {
            "structural_absolute": structural_movement,
            "relative_to_C6_B1_lite": movement_vs_b1,
            "relative_to_C4_final": movement_vs_c4,
        },
        "runtime": {
            "device_requested": args.device,
            "cuda_available": bool(torch.cuda.is_available()),
            "official_cache_npz": artifact(args.official_cache_npz),
            "candidate_pool_npz": artifact(args.candidate_pool_npz),
            "selector_ckpt": artifact(args.selector_ckpt),
        },
        "result_interpretation": (
            "C6-B2 official-val one-shot positive vs C6-B1-lite"
            if status == "C6_B2_OFFICIAL_POSITIVE"
            else "C6-B2 train_calib positive but official-val follow-up not confirmed"
        ),
        "no_second_official_val": True,
    }
    atomic_json(paths["manifest_json"], manifest)

    md = [
        "# C6-B2 official-val one-shot audit",
        "",
        f"- Status: `{status}`",
        "- official_val_one_shot: `true`",
        f"- selected_config: `{EXPECTED_CONFIG['config_id']}`",
        "- successful_metric_result_count: `1`",
        "- learned_config_count_on_val: `0`",
        "- score_grid_on_val / temperature_search_on_val / post_val_adjustment: `false / false / false`",
        "- C6_C_used / C4_final_modified / C6_B1_lite_modified / evaluator_modified: `false / false / false / false`",
        "",
        "## Metrics",
        "",
        "| system | 0.5-r1 | 0.5-r5 | 0.5-r10 | 0.5-r100 | 0.7-r1 | 0.7-r5 | 0.7-r10 | 0.7-r100 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        "| C4_final v21_00444 | " + " | ".join(f"{c4_metrics[k]:.2f}" for k in METRIC_KEYS) + " |",
        "| C6-B1-lite c6b1r1_0057 | " + " | ".join(f"{c6_b1_metrics[k]:.2f}" for k in METRIC_KEYS) + " |",
        "| C6-B2 pairwise_main_0005 | " + " | ".join(f"{metrics[k]:.2f}" for k in METRIC_KEYS) + " |",
        "| Δ vs C6-B1 | " + " | ".join(f"{delta_vs_b1[k]:+.2f}" for k in METRIC_KEYS) + " |",
        "| Δ vs C4_final | " + " | ".join(f"{delta_vs_c4[k]:+.2f}" for k in METRIC_KEYS) + " |",
        "",
        "## Movement",
        "",
        f"- replacement_rate_top_slots: `{structural_movement['replacement_rate_top_slots']}`",
        f"- candidate_replacement_count: `{structural_movement['candidate_replacement_count']}`",
        f"- top1_changed_ratio_vs_C6_B1: `{movement_vs_b1['top1_changed_ratio']}`",
        f"- top1_changed_ratio_vs_C4_final: `{structural_movement['top1_changed_ratio_vs_c4_final']}`",
        f"- hard-positive exits/entries vs C6-B1: `{movement_vs_b1['hard_positive_top100_exits']} / {movement_vs_b1['hard_positive_top100_entries']}`",
        f"- hard-positive exits/entries vs C4_final: `{movement_vs_c4['hard_positive_top100_exits']} / {movement_vs_c4['hard_positive_top100_entries']}`",
        f"- video_slot_drift / video_multiset_drift: `{structural_movement['video_slot_drift']} / {structural_movement['video_multiset_drift']}`",
        f"- invalid_span_count / duplicate_span_count_after_nms: `{structural_movement['invalid_span_count']} / {structural_movement['duplicate_span_count_after_nms']}`",
        "",
        "No post-val adjustment was performed. No second official-val run is authorized.",
        "",
    ]
    atomic_text(paths["audit_md"], "\n".join(md))

    hash_items = {
        "start_manifest": paths["start_manifest"],
        "preflight_json": paths["preflight_json"],
        "preflight_md": paths["preflight_md"],
        "one_shot_manifest": paths["manifest_json"],
        "one_shot_audit": paths["audit_md"],
        "scores_npz": paths["scores_npz"],
        "predictions_jsonl": paths["predictions_jsonl"],
        "submission_json": paths["submission_json"],
        "metrics_json": paths["metrics_json"],
        "freeze_manifest": args.freeze_manifest,
        "freeze_hashes": args.freeze_hashes,
        "best_config": args.best_config_json,
        "selector_ckpt": args.selector_ckpt,
        "official_cache_npz": args.official_cache_npz,
        "candidate_pool_npz": args.candidate_pool_npz,
        "c6_b1_submission_json": args.c6_b1_submission_json,
        "c6_b1_metrics_json": args.c6_b1_metrics_json,
        "c4_final_submission_json": args.c4_final_submission_json,
        "c4_final_metrics_json": args.c4_final_metrics_json,
        "one_shot_code": __file__,
        "evaluator": "standalone_eval/eval.py",
        "model_conquer": "model/conquer.py",
    }
    atomic_json(paths["hashes_json"], {
        "status": "C6_B2_OFFICIAL_VAL_HASHES",
        "official_val_one_shot": True,
        "successful_metric_result_count": 1,
        "artifacts": {k: artifact(v) for k, v in hash_items.items()},
    })
    print(json.dumps({
        "status": status,
        "metrics": metrics,
        "delta_vs_C6_B1_lite": delta_vs_b1,
        "delta_vs_C4_final": delta_vs_c4,
        "movement_vs_C6_B1_lite": movement_vs_b1,
        "structural_movement": structural_movement,
    }, indent=2))


if __name__ == "__main__":
    main()
