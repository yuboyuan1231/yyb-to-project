#!/usr/bin/env python
"""Exactly-one official-val one-shot for frozen C6-B1-lite R1-safe selector.

This script is intentionally narrow:
  * no training;
  * no grid/search/temperature tuning;
  * no post-val adjustment;
  * one frozen config: c6b1r1_0057.

It materializes the official-val cache needed by the frozen selector from
already-frozen C4/C5 artifacts, runs the BoundaryOracleAdapter once, applies the
frozen selector to same-video span alternatives, writes a submission, and calls
the official evaluator once.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rlem.c4_lite_utils import flatten_eval_metrics  # noqa: E402
from rlem.c4_video_targets import (  # noqa: E402
    VIDEO_FEATURE_NAMES,
    _reduce_max,
    _reduce_mean,
    _reduce_min,
    _topk_segment_mean,
    frozen_c4_lite_row_scores,
)
from rlem.c5_main_a_utils import exact_candidate_indices  # noqa: E402
from rlem_c6.c6a_utils import write_json  # noqa: E402
from rlem_c6_b0.run_c6_b0_candidate_diagnostic import (  # noqa: E402
    artifact,
    correctness_for_ts,
    infer_boundary_prior,
    load_gt_ts,
    span_iou_idx,
)
from rlem_c6_b1_lite.run_c6_b1_r1_safe_selector import (  # noqa: E402
    boundary_alternatives,
    make_feature,
    score_pool,
)
from standalone_eval.eval import eval_retrieval, load_jsonl  # noqa: E402


METRIC_KEYS = ("0.5-r1", "0.5-r5", "0.5-r10", "0.5-r100", "0.7-r1", "0.7-r5", "0.7-r10", "0.7-r100")
EXPECTED_CONFIG = {
    "config_id": "c6b1r1_0057",
    "apply_slots": 3,
    "threshold0": 0.4,
    "threshold_rest": 1.8,
    "max_replacements": 2,
    "effective_top_n": 100,
    "NMS": 0.7,
    "max_after_nms": 100,
}
SLOTS = 5
ALT_COUNT = 8
TOP_ENDPOINT = 16
CLIP = 1.5
EPS = 1e-12


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--allow_official_val", action="store_true", required=True)
    p.add_argument("--cache_npz", default="results/rlem_c4_lite_official_val/val_c4_cache.npz")
    p.add_argument("--c4_final_scores_npz", default="results/rlem_c4_r2_cal_v21_official_val/val_v21_row_scores.npz")
    p.add_argument("--c4_final_submission_json", default="results/rlem_c4_r2_cal_v21_official_val/val_v21_submission.json")
    p.add_argument("--c4_final_metrics_json", default="results/rlem_c4_r2_cal_v21_official_val/val_v21_metrics.json")
    p.add_argument("--temporal_prior_npz", default="results/rlem_c5_main_a3_video_slot/official_val_temporal_prior.npz")
    p.add_argument("--boundary_ckpt", default="results/rlem_c6a_repair/boundary_oracle/model_best.pt")
    p.add_argument("--selector_ckpt", default="results/rlem_c6_b1_lite_r1_safe/model_best.pt")
    p.add_argument("--best_config_json", default="results/rlem_c6_b1_lite_r1_safe/best_config.json")
    p.add_argument("--freeze_manifest", default="c6_b1_lite_audit/C6_B1_LITE_R1_SAFE_FREEZE_MANIFEST.json")
    p.add_argument("--freeze_review", default="c6_b1_lite_audit/C6_B1_LITE_R1_SAFE_FREEZE_REVIEW.md")
    p.add_argument("--freeze_hashes", default="c6_b1_lite_audit/C6_B1_LITE_R1_SAFE_FREEZE_HASHES.json")
    p.add_argument("--dataset_config", default="config/tvr_data_config.json")
    p.add_argument("--gt_jsonl", default="/home/a/yybwork/data/yyb/tvr_feature_release/data/tvr_val_release.jsonl")
    p.add_argument("--output_dir", default="results/rlem_c6_b1_lite_r1_safe")
    p.add_argument("--audit_dir", default="c6_b1_lite_official_val")
    p.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    p.add_argument("--selector_batch_size", type=int, default=65536)
    p.add_argument("--boundary_batch_groups", type=int, default=512)
    p.add_argument("--resume_from_scores_after_infra_crash", action="store_true")
    return p.parse_args()


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path: str | Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(path) + ".partial")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def atomic_npz(path: str | Path, **arrays: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(path) + ".partial")
    with open(tmp, "wb") as f:
        np.savez_compressed(f, **arrays)
    os.replace(tmp, path)


def load_metrics(path: str | Path) -> Dict[str, float]:
    obj = json.loads(Path(path).read_text(encoding="utf-8"))
    flat = obj["VCMR"] if "VCMR" in obj else flatten_eval_metrics(obj)
    return {k: float(flat[k]) for k in METRIC_KEYS}


def metric_delta(new: Dict[str, float], base: Dict[str, float]) -> Dict[str, float]:
    return {k: float(new[k] - base[k]) for k in METRIC_KEYS}


def ensure_no_outputs(outputs: Dict[str, Path], resume: bool) -> None:
    existing = {k: p for k, p in outputs.items() if p.exists()}
    if resume:
        allowed = {
            "scores_npz", "candidate_pool_npz", "boundary_prior_npz", "official_cache_npz",
            "predictions_jsonl", "submission_json", "metrics_json", "start_manifest",
            "preflight_md", "preflight_json",
        }
        bad = {k: p for k, p in existing.items() if k not in allowed}
        if bad:
            raise FileExistsError("resume allows only frozen score/cache artifacts; found " + ", ".join(str(p) for p in bad.values()))
        if not outputs["scores_npz"].exists():
            raise FileNotFoundError("resume requested but official_val_scores.npz is absent")
    elif existing:
        raise FileExistsError("one-shot outputs already exist; refusing rerun: " + ", ".join(str(p) for p in existing.values()))


def validate_freeze(args: argparse.Namespace) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    freeze = json.loads(Path(args.freeze_manifest).read_text(encoding="utf-8"))
    best = json.loads(Path(args.best_config_json).read_text(encoding="utf-8"))
    if freeze.get("status") != "C6_B1_LITE_R1_SAFE_FREEZE_REVIEW_PASS":
        raise ValueError("freeze status mismatch")
    if freeze.get("frozen_config_id") != EXPECTED_CONFIG["config_id"]:
        raise ValueError("freeze selected config mismatch")
    frozen_cfg = freeze.get("frozen_config", {})
    for k in ("apply_slots", "threshold0", "threshold_rest", "max_replacements", "effective_top_n", "NMS", "max_after_nms"):
        if frozen_cfg.get(k) != EXPECTED_CONFIG[k]:
            raise ValueError(f"freeze config mismatch for {k}: {frozen_cfg.get(k)}")
    for k in ("config_id", "apply_slots", "threshold0", "threshold_rest", "max_replacements"):
        if best.get(k) != EXPECTED_CONFIG[k]:
            raise ValueError(f"best_config mismatch for {k}: {best.get(k)}")
    if freeze.get("official_val_used") is not False or freeze.get("post_val_adjustment") is not False:
        raise ValueError("freeze manifest indicates prior official val or post-val adjustment")
    return freeze, best


def build_video_features_label_free(cache: Dict[str, np.ndarray], device_name: str) -> Dict[str, np.ndarray]:
    sort_idx = cache["group_sort_idx"].astype(np.int64)
    offsets = cache["group_offsets"].astype(np.int64)
    total_rows = len(sort_idx)
    counts = np.diff(np.r_[offsets, total_rows]).astype(np.int64)
    torch_device = torch.device("cuda" if device_name == "cuda" and torch.cuda.is_available() else "cpu")
    s_c4, g_span, g_video, qb_centered = frozen_c4_lite_row_scores(cache)
    qj = cache["q_joint"].astype(np.float32)
    efp = cache["e_fp"].astype(np.float32)
    s_c31 = cache["s_c31"].astype(np.float32)
    c31_mean = _reduce_mean(s_c31, sort_idx, offsets, counts)
    c4_mean = _reduce_mean(s_c4, sort_idx, offsets, counts)
    c31_sq_mean = _reduce_mean(s_c31 * s_c31, sort_idx, offsets, counts)
    c4_sq_mean = _reduce_mean(s_c4 * s_c4, sort_idx, offsets, counts)
    rank_min = _reduce_min(cache["rank_base"].astype(np.float32), sort_idx, offsets)
    cols = [
        _reduce_max(cache["log_r1"], sort_idx, offsets),
        _reduce_max(cache["base_log"], sort_idx, offsets), _reduce_mean(cache["base_log"], sort_idx, offsets, counts),
        _reduce_max(cache["log_boundary"], sort_idx, offsets), _reduce_mean(cache["log_boundary"], sort_idx, offsets, counts),
        _reduce_max(s_c31, sort_idx, offsets), c31_mean,
        _topk_segment_mean(s_c31, sort_idx, offsets, counts, largest=True, device=torch_device),
        np.sqrt(np.maximum(c31_sq_mean - c31_mean * c31_mean, 0.0)).astype(np.float32),
        _reduce_max(s_c4, sort_idx, offsets), c4_mean,
        _topk_segment_mean(s_c4, sort_idx, offsets, counts, largest=True, device=torch_device),
        np.sqrt(np.maximum(c4_sq_mean - c4_mean * c4_mean, 0.0)).astype(np.float32),
        g_video.astype(np.float32), _reduce_mean(g_span, sort_idx, offsets, counts),
        _topk_segment_mean(g_span, sort_idx, offsets, counts, largest=True, device=torch_device),
        _reduce_max(qj, sort_idx, offsets), _reduce_mean(qj, sort_idx, offsets, counts),
        _topk_segment_mean(qj, sort_idx, offsets, counts, largest=True, device=torch_device),
        _reduce_max(qb_centered, sort_idx, offsets), _reduce_mean(qb_centered, sort_idx, offsets, counts),
        _topk_segment_mean(qb_centered, sort_idx, offsets, counts, largest=True, device=torch_device),
        _reduce_min(efp, sort_idx, offsets), _reduce_mean(efp, sort_idx, offsets, counts),
        _topk_segment_mean(efp, sort_idx, offsets, counts, largest=False, device=torch_device),
        (1.0 / np.maximum(rank_min, 1.0)).astype(np.float32),
        (counts.astype(np.float32) / np.float32(200.0)),
    ]
    features = np.column_stack(cols).astype(np.float32)
    representative = sort_idx[offsets]
    return {
        "features": features,
        "group_id": cache["group_ids_sorted_unique"].astype(np.int32),
        "query_index": cache["query_index"][representative].astype(np.int32),
        "video_idx": cache["video_idx"][representative].astype(np.int32),
        "feature_names": np.asarray(VIDEO_FEATURE_NAMES, dtype=object),
    }


def build_official_c6_cache(args: argparse.Namespace, output_npz: Path) -> Dict[str, Any]:
    if output_npz.exists():
        return {"reused": True, "output": artifact(output_npz)}
    t0 = time.time()
    with np.load(args.cache_npz, allow_pickle=True) as payload:
        cache = {k: payload[k] for k in payload.files}
    with np.load(args.c4_final_scores_npz, allow_pickle=False) as scores_npz:
        s_c4_final = scores_npz["s_v21"].astype(np.float32)
        if len(s_c4_final) != len(cache["row_group_id"]):
            raise ValueError("C4_final score row count mismatch")
    with np.load(args.temporal_prior_npz, allow_pickle=False) as temporal:
        temporal_length = temporal["temporal_length"].astype(np.int64)
        if len(temporal_length) != len(cache["group_ids_sorted_unique"]):
            raise ValueError("temporal prior group count mismatch")
    start_idx, end_idx, endpoint_audit = exact_candidate_indices(
        cache["start_time"], cache["end_time"], cache["row_group_id"], temporal_length
    )
    video_data = build_video_features_label_free(cache, args.device)
    group_count = len(cache["group_ids_sorted_unique"])
    if not np.array_equal(video_data["group_id"], cache["group_ids_sorted_unique"].astype(np.int32)):
        raise ValueError("video feature group id mismatch")
    zeros_row = np.zeros(len(cache["row_group_id"]), dtype=np.float32)
    zeros_group = np.zeros(group_count, dtype=np.float32)
    atomic_npz(
        output_npz,
        desc_ids=cache["desc_ids"],
        desc_text=cache["desc_text"],
        desc_offsets=cache["desc_offsets"].astype(np.int64),
        row_group_id=cache["row_group_id"].astype(np.int32),
        query_index=cache["query_index"].astype(np.int32),
        video_idx=cache["video_idx"].astype(np.int32),
        start_time=cache["start_time"].astype(np.float32),
        end_time=cache["end_time"].astype(np.float32),
        start_idx=start_idx,
        end_idx=end_idx,
        s_c4_final=s_c4_final,
        iou=zeros_row,
        is_gt_video=zeros_row,
        y_joint_05=zeros_row,
        y_joint_07=zeros_row,
        gt_start_idx=np.zeros(len(zeros_row), dtype=np.int16),
        gt_end_idx=np.zeros(len(zeros_row), dtype=np.int16),
        group_sort_idx=cache["group_sort_idx"].astype(np.int64),
        group_offsets=cache["group_offsets"].astype(np.int64),
        group_ids_sorted_unique=cache["group_ids_sorted_unique"].astype(np.int32),
        video_group_keys=cache["video_group_keys"].astype(np.int64),
        video_features=video_data["features"].astype(np.float32),
        video_feature_names=video_data["feature_names"],
        label_relevant=zeros_group,
        label_best_iou=zeros_group,
        label_any_05=zeros_group,
        label_any_07=zeros_group,
        row_feature_names=np.asarray([], dtype=object),
        row_feature_mean=np.asarray([], dtype=np.float32),
        row_feature_std=np.asarray([], dtype=np.float32),
    )
    return {
        "reused": False,
        "output": artifact(output_npz),
        "queries": int(len(cache["desc_ids"])),
        "rows": int(len(cache["row_group_id"])),
        "groups": int(group_count),
        "video_feature_dim": int(video_data["features"].shape[1]),
        "endpoint_index_audit": endpoint_audit,
        "elapsed_sec": float(time.time() - t0),
    }


def normalize_distribution(x: np.ndarray) -> np.ndarray:
    y = np.maximum(np.asarray(x, dtype=np.float64), EPS)
    return (y / max(float(y.sum()), EPS)).astype(np.float32)


def build_official_pool(cache: Dict[str, np.ndarray], temporal_npz: str, prior_npz: str, output_npz: Path) -> Dict[str, Any]:
    if output_npz.exists():
        with np.load(output_npz, allow_pickle=False) as z:
            return {"reused": True, "output": artifact(output_npz), "shape": z["shape"].astype(int).tolist()}
    t0 = time.time()
    with np.load(temporal_npz, allow_pickle=False) as temporal:
        p_ctx = temporal["p_ctx"].astype(np.float32)
        temporal_length = temporal["temporal_length"].astype(np.int64)
    with np.load(prior_npz, allow_pickle=False) as prior:
        p_b = prior["p_b_new"].astype(np.float32)
        p_e = prior["p_e_new"].astype(np.float32)
    q_count = len(cache["desc_ids"])
    rows_per_q = int(cache["desc_offsets"][1] - cache["desc_offsets"][0])
    total = q_count * SLOTS * ALT_COUNT
    x = np.zeros((total, 51), dtype=np.float32)
    q_arr = np.zeros(total, dtype=np.int32)
    slot_arr = np.zeros(total, dtype=np.int16)
    alt_arr = np.zeros(total, dtype=np.int16)
    gid_arr = np.zeros(total, dtype=np.int32)
    vid_arr = np.zeros(total, dtype=np.int32)
    st_arr = np.zeros(total, dtype=np.int16)
    en_arr = np.zeros(total, dtype=np.int16)
    ost_arr = np.zeros(total, dtype=np.int16)
    oen_arr = np.zeros(total, dtype=np.int16)
    is_orig = np.zeros(total, dtype=np.bool_)
    base_score = np.zeros(total, dtype=np.float32)
    row_group = cache["row_group_id"].astype(np.int64)
    video_idx = cache["video_idx"].astype(np.int64)
    start_idx = cache["start_idx"].astype(np.int64)
    end_idx = cache["end_idx"].astype(np.int64)
    s_c4 = cache["s_c4_final"].astype(np.float32)
    video_feat = cache["video_features"].astype(np.float32)
    pos = 0
    for q in range(q_count):
        rs = np.arange(q * rows_per_q, (q + 1) * rows_per_q, dtype=np.int64)
        scores = s_c4[rs]
        order = rs[np.argsort(-scores, kind="stable")[:SLOTS]]
        q_top = float(np.max(scores)); q_mean = float(np.mean(scores)); q_std = float(np.std(scores))
        for slot_rank, row in enumerate(order):
            gid = int(row_group[row])
            t = int(temporal_length[gid])
            pb = normalize_distribution(p_b[gid, :t])
            pe = normalize_distribution(p_e[gid, :t])
            pc = normalize_distribution(p_ctx[gid, :t])
            os_, oe_ = int(start_idx[row]), int(end_idx[row])
            starts, ends, endpoint_scores = boundary_alternatives(pb, pe, t, os_, oe_, ALT_COUNT, TOP_ENDPOINT, EXPECTED_CONFIG["NMS"])
            orig_endpoint = float(endpoint_scores[0])
            for a in range(ALT_COUNT):
                si, ei = int(starts[a]), int(ends[a])
                x[pos] = make_feature(
                    si=si, ei=ei, orig_s=os_, orig_e=oe_,
                    endpoint_score=float(endpoint_scores[a]), orig_endpoint_score=orig_endpoint,
                    pb=pb, pe=pe, pc=pc, t=t, slot_rank=slot_rank,
                    s_c4=float(s_c4[row]), q_top_score=q_top, q_mean_score=q_mean, q_std_score=q_std,
                    video_feat=video_feat[gid], is_original=(a == 0),
                )
                q_arr[pos] = q; slot_arr[pos] = slot_rank; alt_arr[pos] = a
                gid_arr[pos] = gid; vid_arr[pos] = int(video_idx[row])
                st_arr[pos] = si; en_arr[pos] = ei; ost_arr[pos] = os_; oen_arr[pos] = oe_
                is_orig[pos] = a == 0; base_score[pos] = float(s_c4[row])
                pos += 1
        if (q + 1) % 5000 == 0:
            print(json.dumps({"stage": "build_official_pool", "queries": q + 1, "total": q_count}))
    atomic_npz(
        output_npz,
        x=x[:pos], query=q_arr[:pos], slot=slot_arr[:pos], alt=alt_arr[:pos],
        group_id=gid_arr[:pos], video_idx=vid_arr[:pos], start_idx=st_arr[:pos], end_idx=en_arr[:pos],
        orig_start_idx=ost_arr[:pos], orig_end_idx=oen_arr[:pos], is_original=is_orig[:pos],
        base_score=base_score[:pos], shape=np.asarray([q_count, SLOTS, ALT_COUNT], dtype=np.int64),
    )
    return {"reused": False, "output": artifact(output_npz), "shape": [q_count, SLOTS, ALT_COUNT], "elapsed_sec": float(time.time() - t0)}


def apply_policy_and_submission(
    cache: Dict[str, np.ndarray],
    pool: Dict[str, np.ndarray],
    pool_score: np.ndarray,
    c4_submission_json: str,
    outputs: Dict[str, Path],
) -> Dict[str, Any]:
    c4_submission = json.loads(Path(c4_submission_json).read_text(encoding="utf-8"))
    q_count, slots, alt_count = [int(x) for x in pool["shape"]]
    rows_per_q = int(cache["desc_offsets"][1] - cache["desc_offsets"][0])
    video_idx = cache["video_idx"].astype(np.int64)
    start_idx = cache["start_idx"].astype(np.int64)
    end_idx = cache["end_idx"].astype(np.int64)
    s_c4 = cache["s_c4_final"].astype(np.float32)
    replacement_count = 0
    top1_changed = 0
    invalid_span = 0
    duplicate_span = 0
    replacement_by_slot = Counter()
    predictions_jsonl: List[Dict[str, Any]] = []
    scores_order = np.full((q_count, EXPECTED_CONFIG["effective_top_n"], 4), -1, dtype=np.float32)
    for q in range(q_count):
        rs = np.arange(q * rows_per_q, (q + 1) * rows_per_q, dtype=np.int64)
        base_order = rs[np.argsort(-s_c4[rs], kind="stable")[:EXPECTED_CONFIG["effective_top_n"]]]
        slot_candidates = []
        replaced_this_q = 0
        for slot_rank, row in enumerate(base_order):
            vid = int(video_idx[row])
            si, ei = int(start_idx[row]), int(end_idx[row])
            replaced = False
            if slot_rank < min(slots, EXPECTED_CONFIG["apply_slots"]) and replaced_this_q < EXPECTED_CONFIG["max_replacements"]:
                base = (q * slots + slot_rank) * alt_count
                scores = pool_score[base:base + alt_count]
                best = int(np.argmax(scores))
                margin = float(scores[best] - scores[0])
                thd = EXPECTED_CONFIG["threshold0"] if slot_rank == 0 else EXPECTED_CONFIG["threshold_rest"]
                if best != 0 and margin > thd:
                    si = int(pool["start_idx"][base + best])
                    ei = int(pool["end_idx"][base + best])
                    replacement_count += 1
                    replaced_this_q += 1
                    replacement_by_slot[slot_rank] += 1
                    replaced = True
            if ei < si or si < 0:
                invalid_span += 1
            slot_candidates.append((vid, si, ei, float(s_c4[row]), int(row), replaced))
        kept: List[int] = []
        by_video: Dict[int, List[int]] = {}
        for idx, (vid, si, ei, _score, _row, _replaced) in enumerate(slot_candidates):
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
        duplicate_span += len(kept) - len({(slot_candidates[i][0], slot_candidates[i][1], slot_candidates[i][2]) for i in kept})
        base_top = (int(video_idx[base_order[0]]), int(start_idx[base_order[0]]), int(end_idx[base_order[0]]))
        new_top = (int(slot_candidates[kept[0]][0]), int(slot_candidates[kept[0]][1]), int(slot_candidates[kept[0]][2]))
        top1_changed += int(base_top != new_top)
        preds = []
        for out_rank, i in enumerate(kept):
            vid, si, ei, score, _row, _replaced = slot_candidates[i]
            preds.append([int(vid), float(si * CLIP), float((ei + 1) * CLIP), float(score)])
            scores_order[q, out_rank] = np.asarray([vid, si, ei, score], dtype=np.float32)
        predictions_jsonl.append({"desc_id": cache["desc_ids"][q].item() if hasattr(cache["desc_ids"][q], "item") else cache["desc_ids"][q], "desc": str(cache["desc_text"][q]), "predictions": preds})
    submission = {"video2idx": c4_submission["video2idx"], "VCMR": predictions_jsonl}
    atomic_json(outputs["submission_json"], submission)
    tmp = Path(str(outputs["predictions_jsonl"]) + ".partial")
    with open(tmp, "w", encoding="utf-8") as f:
        for row in predictions_jsonl:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    os.replace(tmp, outputs["predictions_jsonl"])
    atomic_npz(
        outputs["scores_npz"],
        policy_predictions=scores_order,
        pool_score=pool_score.astype(np.float32),
        pool_shape=pool["shape"].astype(np.int64),
        row_group_id=cache["row_group_id"].astype(np.int32),
        s_c4_final=s_c4.astype(np.float32),
    )
    return {
        "replacement_rate_top_slots": float(replacement_count / max(q_count * EXPECTED_CONFIG["apply_slots"], 1)),
        "candidate_replacement_count": int(replacement_count),
        "replacement_by_slot": {str(k): int(v) for k, v in sorted(replacement_by_slot.items())},
        "top1_changed_ratio": float(top1_changed / q_count),
        "video_slot_drift": 0.0,
        "video_multiset_drift": 0.0,
        "invalid_span_count": int(invalid_span),
        "duplicate_span_count": int(duplicate_span),
        "queries": int(q_count),
    }


def compute_policy_movement_only(cache: Dict[str, np.ndarray], pool: Dict[str, np.ndarray], pool_score: np.ndarray) -> Dict[str, Any]:
    q_count, slots, alt_count = [int(x) for x in pool["shape"]]
    rows_per_q = int(cache["desc_offsets"][1] - cache["desc_offsets"][0])
    video_idx = cache["video_idx"].astype(np.int64)
    start_idx = cache["start_idx"].astype(np.int64)
    end_idx = cache["end_idx"].astype(np.int64)
    s_c4 = cache["s_c4_final"].astype(np.float32)
    replacement_count = 0
    top1_changed = 0
    invalid_span = 0
    duplicate_span = 0
    replacement_by_slot = Counter()
    for q in range(q_count):
        rs = np.arange(q * rows_per_q, (q + 1) * rows_per_q, dtype=np.int64)
        base_order = rs[np.argsort(-s_c4[rs], kind="stable")[:EXPECTED_CONFIG["effective_top_n"]]]
        slot_candidates = []
        replaced_this_q = 0
        for slot_rank, row in enumerate(base_order):
            vid = int(video_idx[row])
            si, ei = int(start_idx[row]), int(end_idx[row])
            if slot_rank < min(slots, EXPECTED_CONFIG["apply_slots"]) and replaced_this_q < EXPECTED_CONFIG["max_replacements"]:
                base = (q * slots + slot_rank) * alt_count
                scores = pool_score[base:base + alt_count]
                best = int(np.argmax(scores))
                margin = float(scores[best] - scores[0])
                thd = EXPECTED_CONFIG["threshold0"] if slot_rank == 0 else EXPECTED_CONFIG["threshold_rest"]
                if best != 0 and margin > thd:
                    si = int(pool["start_idx"][base + best])
                    ei = int(pool["end_idx"][base + best])
                    replacement_count += 1
                    replaced_this_q += 1
                    replacement_by_slot[slot_rank] += 1
            if ei < si or si < 0:
                invalid_span += 1
            slot_candidates.append((vid, si, ei, float(s_c4[row]), int(row)))
        kept: List[int] = []
        by_video: Dict[int, List[int]] = {}
        for idx, (vid, si, ei, _score, _row) in enumerate(slot_candidates):
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
        duplicate_span += len(kept) - len({(slot_candidates[i][0], slot_candidates[i][1], slot_candidates[i][2]) for i in kept})
        base_top = (int(video_idx[base_order[0]]), int(start_idx[base_order[0]]), int(end_idx[base_order[0]]))
        new_top = (int(slot_candidates[kept[0]][0]), int(slot_candidates[kept[0]][1]), int(slot_candidates[kept[0]][2]))
        top1_changed += int(base_top != new_top)
    return {
        "replacement_rate_top_slots": float(replacement_count / max(q_count * EXPECTED_CONFIG["apply_slots"], 1)),
        "candidate_replacement_count": int(replacement_count),
        "replacement_by_slot": {str(k): int(v) for k, v in sorted(replacement_by_slot.items())},
        "top1_changed_ratio": float(top1_changed / q_count),
        "video_slot_drift": 0.0,
        "video_multiset_drift": 0.0,
        "invalid_span_count": int(invalid_span),
        "duplicate_span_count": int(duplicate_span),
        "queries": int(q_count),
        "recomputed_from_frozen_pool_score": True,
    }


def hard_positive_movement(c4_submission_json: str, new_submission_json: str, gt_jsonl: str) -> Dict[str, Any]:
    base_obj = json.loads(Path(c4_submission_json).read_text(encoding="utf-8"))
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
        # TVR eval submissions use numeric video_idx; val GT contains vid_idx.
        def ok(preds: Sequence[Sequence[Any]]) -> bool:
            for pred in preds[:100]:
                if int(pred[0]) != int(gt_vid):
                    continue
                si = int(round(float(pred[1]) / CLIP))
                ei = max(si, int(round(float(pred[2]) / CLIP)) - 1)
                if correctness_for_ts(si, ei, ts, 0.5) or correctness_for_ts(si, ei, ts, 0.7):
                    return True
            return False
        bp = ok(brow["predictions"]); np_ = ok(nrow["predictions"])
        base_pos += int(bp); new_pos += int(np_)
        exits += int(bp and not np_); entries += int((not bp) and np_)
    return {
        "hard_positive_top100_exits": int(exits),
        "hard_positive_top100_entries": int(entries),
        "base_positive_top100_queries": int(base_pos),
        "new_positive_top100_queries": int(new_pos),
        "hard_positive_top100_exit_ratio": float(exits / max(base_pos, 1)),
    }


def classify(delta: Dict[str, float], movement: Dict[str, Any]) -> str:
    r1_pos = delta["0.5-r1"] > 0.0 and delta["0.7-r1"] > 0.0
    r1_nonneg = delta["0.5-r1"] >= 0.0 and delta["0.7-r1"] >= 0.0
    r5_safe = delta["0.5-r5"] >= -0.05 and delta["0.7-r5"] >= -0.05
    r100_safe = delta["0.5-r100"] >= -0.05 and delta["0.7-r100"] >= -0.05
    r10_collapse = delta["0.5-r10"] < -0.10 or delta["0.7-r10"] < -0.10
    movement_safe = movement["video_slot_drift"] == 0.0 and movement["video_multiset_drift"] == 0.0 and movement["top1_changed_ratio"] <= 0.08
    all_nonneg = all(delta[k] >= 0.0 for k in METRIC_KEYS)
    if all_nonneg and r1_pos and r5_safe and r100_safe and movement_safe:
        return "C6_B1_LITE_OFFICIAL_VAL_STRONG_POSITIVE"
    if r1_pos and r5_safe and r100_safe and not r10_collapse and movement_safe:
        return "C6_B1_LITE_OFFICIAL_VAL_R1_SAFE_POSITIVE"
    if (r1_pos or r1_nonneg) and not (r5_safe and r100_safe):
        return "C6_B1_LITE_OFFICIAL_VAL_R1_TRADEOFF"
    return "C6_B1_LITE_OFFICIAL_VAL_NEGATIVE"


def main() -> None:
    args = parse_args()
    if not args.allow_official_val:
        raise ValueError("--allow_official_val is required for this authorized one-shot")
    out_dir = Path(args.output_dir)
    audit_dir = Path(args.audit_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    audit_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "official_cache_npz": out_dir / "official_val_c6_cache.npz",
        "boundary_prior_npz": out_dir / "official_val_boundary_oracle_prior.npz",
        "candidate_pool_npz": out_dir / "official_val_candidate_pool_top_slots.npz",
        "scores_npz": out_dir / "official_val_scores.npz",
        "predictions_jsonl": out_dir / "official_val_predictions.jsonl",
        "submission_json": out_dir / "official_val_submission.json",
        "metrics_json": out_dir / "official_val_metrics.json",
        "start_manifest": audit_dir / "C6_B1_LITE_OFFICIAL_VAL_START_MANIFEST.json",
        "preflight_md": audit_dir / "C6_B1_LITE_OFFICIAL_VAL_PREFLIGHT_AUDIT.md",
        "preflight_json": audit_dir / "C6_B1_LITE_OFFICIAL_VAL_PREFLIGHT_AUDIT.json",
        "audit_md": audit_dir / "C6_B1_LITE_OFFICIAL_VAL_ONE_SHOT_AUDIT.md",
        "manifest_json": audit_dir / "C6_B1_LITE_OFFICIAL_VAL_ONE_SHOT_MANIFEST.json",
        "hashes_json": audit_dir / "C6_B1_LITE_OFFICIAL_VAL_ONE_SHOT_HASHES.json",
    }
    ensure_no_outputs(outputs, args.resume_from_scores_after_infra_crash)

    freeze, best = validate_freeze(args)
    start_manifest = {
        "stage": "C6-B1-lite official-val one-shot",
        "selected_config": EXPECTED_CONFIG["config_id"],
        "apply_slots": EXPECTED_CONFIG["apply_slots"],
        "threshold0": EXPECTED_CONFIG["threshold0"],
        "threshold_rest": EXPECTED_CONFIG["threshold_rest"],
        "max_replacements": EXPECTED_CONFIG["max_replacements"],
        "official_val_one_shot": True,
        "official_val_used_before_this_run": False,
        "score_grid_on_val": False,
        "temperature_search_on_val": False,
        "post_val_adjustment": False,
        "C6_C_used": False,
        "C4_final_modified": False,
        "NMS_modified": False,
        "evaluator_modified": False,
        "freeze_path_correction": {
            "requested_by_instruction": "c6_b1_lite_freeze_audit/C6_B1_LITE_FREEZE_*.{md,json}",
            "actual_used": "c6_b1_lite_audit/C6_B1_LITE_R1_SAFE_FREEZE_*",
        },
    }
    atomic_json(outputs["start_manifest"], start_manifest)

    py = sys.executable
    compile_result = subprocess.run(
        [py, "-m", "py_compile", "rlem_c6_b1_lite/run_c6_b1_r1_safe_selector.py", __file__],
        cwd=str(ROOT),
        text=True,
        capture_output=True,
        check=False,
    )
    partials = [str(p) for p in ROOT.rglob("*.partial") if "official_val" in str(p) or "c6_b1_lite" in str(p)]
    preflight = {
        "status": "PASS" if compile_result.returncode == 0 and not partials else "FAIL",
        "python": py,
        "py_compile_returncode": int(compile_result.returncode),
        "py_compile_stderr": compile_result.stderr,
        "partial_files": partials,
        "freeze_manifest": artifact(args.freeze_manifest),
        "freeze_review": artifact(args.freeze_review),
        "freeze_hashes": artifact(args.freeze_hashes),
        "best_config": artifact(args.best_config_json),
        "selector_ckpt": artifact(args.selector_ckpt),
        "boundary_ckpt": artifact(args.boundary_ckpt),
        "selector_code": artifact("rlem_c6_b1_lite/run_c6_b1_r1_safe_selector.py"),
        "official_val_cache_source": artifact(args.cache_npz),
        "c4_final_scores": artifact(args.c4_final_scores_npz),
        "temporal_prior": artifact(args.temporal_prior_npz),
        "config_exact_match": True,
        "official_val_used_before_this_run": False,
        "post_val_adjustment": False,
        "score_grid_on_val": False,
        "temperature_search_on_val": False,
    }
    atomic_json(outputs["preflight_json"], preflight)
    outputs["preflight_md"].write_text(
        "# C6-B1-lite official-val preflight audit\n\n"
        f"- status: `{preflight['status']}`\n"
        f"- py_compile_returncode: `{preflight['py_compile_returncode']}`\n"
        f"- selected_config: `{EXPECTED_CONFIG['config_id']}`\n"
        "- path correction: instruction freeze path did not exist; used durable `c6_b1_lite_audit/C6_B1_LITE_R1_SAFE_FREEZE_*`.\n"
        f"- partial_files: `{len(partials)}`\n",
        encoding="utf-8",
    )
    if preflight["status"] != "PASS":
        raise RuntimeError("preflight failed; stopping before official val scoring")

    cache_manifest = build_official_c6_cache(args, outputs["official_cache_npz"])
    prior_manifest = infer_boundary_prior(
        cache_npz=str(outputs["official_cache_npz"]),
        temporal_npz=args.temporal_prior_npz,
        ckpt_path=args.boundary_ckpt,
        output_npz=str(outputs["boundary_prior_npz"]),
        device_name=args.device,
        batch_groups=args.boundary_batch_groups,
    )
    with np.load(outputs["official_cache_npz"], allow_pickle=True) as payload:
        cache = {k: payload[k] for k in payload.files}
    pool_manifest = build_official_pool(cache, args.temporal_prior_npz, str(outputs["boundary_prior_npz"]), outputs["candidate_pool_npz"])
    with np.load(outputs["candidate_pool_npz"], allow_pickle=False) as payload:
        pool = {k: payload[k] for k in payload.files}
    if args.resume_from_scores_after_infra_crash:
        with np.load(outputs["scores_npz"], allow_pickle=False) as z:
            pool_score = z["pool_score"].astype(np.float32)
            movement = compute_policy_movement_only(cache, pool, pool_score)
            movement["reused_scores"] = True
            movement["policy_predictions_shape"] = list(z["policy_predictions"].shape)
    else:
        pool_score = score_pool(pool, args.selector_ckpt, args.device, args.selector_batch_size)
        movement = apply_policy_and_submission(cache, pool, pool_score, args.c4_final_submission_json, outputs)

    if args.resume_from_scores_after_infra_crash and outputs["metrics_json"].exists():
        metrics_raw = json.loads(outputs["metrics_json"].read_text(encoding="utf-8"))
    else:
        submission = json.loads(outputs["submission_json"].read_text(encoding="utf-8"))
        metrics_raw = eval_retrieval(
            submission,
            load_jsonl(args.gt_jsonl),
            iou_thds=(0.5, 0.7),
            verbose=False,
            match_number=True,
            use_desc_type=True,
        )
        atomic_json(outputs["metrics_json"], metrics_raw)
    flat = {k: float(flatten_eval_metrics(metrics_raw)[k]) for k in METRIC_KEYS}
    c4_final_metrics = load_metrics(args.c4_final_metrics_json)
    delta = metric_delta(flat, c4_final_metrics)
    hard_move = hard_positive_movement(args.c4_final_submission_json, str(outputs["submission_json"]), args.gt_jsonl)
    movement.update(hard_move)
    outcome = classify(delta, movement)

    manifest = {
        "status": outcome,
        "date": datetime.now().strftime("%Y-%m-%d"),
        "stage": "C6-B1-lite official-val one-shot",
        "official_val_one_shot": True,
        "successful_metric_result_count": 1,
        "score_grid_on_val": False,
        "temperature_search_on_val": False,
        "post_val_adjustment": False,
        "learned_config_count_on_val": 0,
        "learned_config_count_total": 1,
        "failed_infrastructure_attempt_count": int(args.resume_from_scores_after_infra_crash),
        "selected_config": EXPECTED_CONFIG["config_id"],
        "frozen_config": EXPECTED_CONFIG,
        "slots_for_selector_pool": SLOTS,
        "alt_count": ALT_COUNT,
        "training_used": False,
        "selector_retrained": False,
        "C6_C_used": False,
        "C4_final_modified": False,
        "NMS_modified": False,
        "evaluator_modified": False,
        "metrics": {
            "c4_final_v21_00444": c4_final_metrics,
            "c6_b1_lite_c6b1r1_0057": flat,
            "delta_vs_c4_final": delta,
        },
        "movement_relative_to_c4_final": movement,
        "runtime": {
            "cache": cache_manifest,
            "boundary_prior": prior_manifest,
            "pool": pool_manifest,
            "device_requested": args.device,
            "cuda_available": bool(torch.cuda.is_available()),
        },
        "result_interpretation": (
            "C6-B1-lite official-val one-shot positive under R1-safe gate"
            if "POSITIVE" in outcome else
            "C6-B1-lite train_calib positive but official-val follow-up not confirmed"
        ),
        "no_second_official_val": True,
    }
    atomic_json(outputs["manifest_json"], manifest)

    md = [
        "# C6-B1-lite official-val one-shot audit",
        "",
        f"- Status: `{outcome}`",
        "- official_val_one_shot: `true`",
        "- selected_config: `c6b1r1_0057`",
        "- learned_config_count_on_val: `0`",
        "- score_grid_on_val / temperature_search_on_val / post_val_adjustment: `false / false / false`",
        "- C6_C_used / C4_final_modified / evaluator_modified: `false / false / false`",
        "",
        "## Metrics",
        "",
        "| system | 0.5-r1 | 0.5-r5 | 0.5-r10 | 0.5-r100 | 0.7-r1 | 0.7-r5 | 0.7-r10 | 0.7-r100 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        "| C4_final v21_00444 | " + " | ".join(f"{c4_final_metrics[k]:.2f}" for k in METRIC_KEYS) + " |",
        "| C6-B1-lite c6b1r1_0057 | " + " | ".join(f"{flat[k]:.2f}" for k in METRIC_KEYS) + " |",
        "| delta | " + " | ".join(f"{delta[k]:+.2f}" for k in METRIC_KEYS) + " |",
        "",
        "## Movement / structure",
        "",
        f"- replacement_rate_top_slots: `{movement.get('replacement_rate_top_slots')}`",
        f"- candidate_replacement_count: `{movement.get('candidate_replacement_count')}`",
        f"- top1_changed_ratio: `{movement.get('top1_changed_ratio')}`",
        f"- hard-positive top100 exits/entries: `{movement.get('hard_positive_top100_exits')} / {movement.get('hard_positive_top100_entries')}`",
        f"- video_slot_drift / video_multiset_drift: `{movement.get('video_slot_drift')} / {movement.get('video_multiset_drift')}`",
        f"- invalid_span_count / duplicate_span_count: `{movement.get('invalid_span_count')} / {movement.get('duplicate_span_count')}`",
        "",
        "No post-val adjustment was performed. No second official-val run is authorized.",
        "",
    ]
    outputs["audit_md"].write_text("\n".join(md), encoding="utf-8")

    hashes = {
        "status": "HASHES_MATERIALIZED_AFTER_ONE_SHOT",
        "official_val_one_shot": True,
        "successful_metric_result_count": 1,
        "artifacts": {k: artifact(v) for k, v in {
            "start_manifest": outputs["start_manifest"],
            "preflight_json": outputs["preflight_json"],
            "preflight_md": outputs["preflight_md"],
            "one_shot_manifest": outputs["manifest_json"],
            "one_shot_audit": outputs["audit_md"],
            "official_cache_npz": outputs["official_cache_npz"],
            "boundary_prior_npz": outputs["boundary_prior_npz"],
            "candidate_pool_npz": outputs["candidate_pool_npz"],
            "scores_npz": outputs["scores_npz"],
            "predictions_jsonl": outputs["predictions_jsonl"],
            "submission_json": outputs["submission_json"],
            "metrics_json": outputs["metrics_json"],
            "freeze_manifest": args.freeze_manifest,
            "best_config": args.best_config_json,
            "selector_ckpt": args.selector_ckpt,
            "boundary_ckpt": args.boundary_ckpt,
            "one_shot_code": __file__,
            "evaluator": "standalone_eval/eval.py",
            "model_conquer": "model/conquer.py",
        }.items()},
    }
    atomic_json(outputs["hashes_json"], hashes)
    print(json.dumps({"status": outcome, "metrics": flat, "delta_vs_c4_final": delta, "movement": movement}, indent=2))


if __name__ == "__main__":
    main()
