#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import math
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rlem_c6_b0.run_c6_b0_candidate_diagnostic import correctness_for_ts, load_gt_ts, span_iou_idx  # noqa: E402
from rlem_c6_b1_lite.run_c6_b1_r1_safe_selector import evaluate_policy, score_pool  # noqa: E402
from rlem_c6_b2.run_c6_b2_goal import PairwiseUtilityModel, build_pair_cache_for_split  # noqa: E402


METRIC_KEYS = [
    "0.5-r1", "0.5-r5", "0.5-r10", "0.5-r100",
    "0.7-r1", "0.7-r5", "0.7-r10", "0.7-r100",
]
CLIP = 1.5
EPS = 1e-8

_SEARCH_ARGS: argparse.Namespace | None = None
_SEARCH_SCORE_ARRAYS: Dict[str, np.ndarray] | None = None
_SEARCH_ANCHOR_BUNDLE: Dict[str, Any] | None = None
_SEARCH_B1_METRICS: Dict[str, float] | None = None
_SEARCH_B2_METRICS: Dict[str, float] | None = None


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


def load_npz(path: str | Path, allow_pickle: bool = True) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=allow_pickle) as z:
        return {k: z[k] for k in z.files}


def selected_metrics(labels05: List[List[bool]], labels07: List[List[bool]]) -> Dict[str, float]:
    q_count = len(labels05)
    out = {k: 0 for k in METRIC_KEYS}
    for q in range(q_count):
        for thr, labels in [("0.5", labels05[q]), ("0.7", labels07[q])]:
            for k in [1, 5, 10, 100]:
                out[f"{thr}-r{k}"] += int(any(labels[:k]))
    return {k: 100.0 * v / max(q_count, 1) for k, v in out.items()}


def metric_delta(new: Dict[str, float], base: Dict[str, float]) -> Dict[str, float]:
    return {k: float(new[k] - base[k]) for k in METRIC_KEYS}


def topk_mean(x: np.ndarray, k: int) -> float:
    if len(x) == 0:
        return 0.0
    y = np.sort(x)[-min(k, len(x)):]
    return float(y.mean())


def logsumexp(x: np.ndarray) -> float:
    if len(x) == 0:
        return 0.0
    m = float(np.max(x))
    return float(m + np.log(np.exp(x - m).sum()))


def entropy_from_probs(p: np.ndarray) -> float:
    p = np.asarray(p, dtype=np.float64)
    p = np.maximum(p, EPS)
    p = p / max(float(p.sum()), EPS)
    return float(-(p * np.log(p)).sum())


def row_bounds_from_gt(ts: Any) -> Tuple[float, float]:
    if isinstance(ts, dict):
        ts = ts.get("ts", ts.get("gt_ts"))
    return float(ts[0]), float(ts[1])


def load_baseline_configs(args: argparse.Namespace) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    b1 = json.loads(Path(args.b1_best_config).read_text(encoding="utf-8"))
    b2 = json.loads(Path(args.b2_best_config).read_text(encoding="utf-8"))
    return b1, b2


def ensure_b1_scores(pool_path: str, out_path: Path, model_path: str, device: str, batch_size: int) -> Dict[str, Any]:
    if out_path.exists():
        return {"reused": True, "score": artifact(out_path)}
    pool = load_npz(pool_path, allow_pickle=False)
    scores = score_pool(pool, model_path, device, batch_size)
    atomic_npz(out_path, score=scores.astype(np.float32))
    return {"reused": False, "score": artifact(out_path), "score_min": float(scores.min()), "score_max": float(scores.max())}


@torch.no_grad()
def ensure_b2_pairwise_scores(
    *,
    split: str,
    pool_path: str,
    cache_dir: Path,
    out_path: Path,
    model_path: str,
    device_name: str,
    batch_size: int,
) -> Dict[str, Any]:
    if out_path.exists():
        return {"reused": True, "score": artifact(out_path)}
    build_pair_cache_for_split(pool_path, cache_dir, split)
    labels = np.load(cache_dir / f"{split}_pair_labels.npz", allow_pickle=False)
    q, slots, alts = [int(x) for x in labels["shape"]]
    x = np.load(cache_dir / f"{split}_pair_x.npy", mmap_mode="r")
    ckpt = torch.load(model_path, map_location="cpu")
    device = torch.device("cuda" if device_name == "cuda" and torch.cuda.is_available() else "cpu")
    model = PairwiseUtilityModel(int(ckpt["in_dim"]), hidden=int(ckpt["hidden"]), layers=int(ckpt["layers"])).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    pair_scores = np.empty(len(x), dtype=np.float32)
    mean = ckpt["mean"].astype(np.float32)
    std = np.maximum(ckpt["std"].astype(np.float32), EPS)
    for start in range(0, len(x), batch_size):
        xb = (np.asarray(x[start:start + batch_size], dtype=np.float32) - mean) / std
        tb = torch.from_numpy(xb).to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
            out = model(tb).float()
            score = out[:, 0] + 0.6 * out[:, 1] - 0.85 * out[:, 2] - 0.45 * out[:, 3] + 0.35 * out[:, 4] + 0.15 * torch.tanh(out[:, 5])
        pair_scores[start:start + len(score)] = score.detach().cpu().numpy().astype(np.float32)
    pool = load_npz(pool_path, allow_pickle=False)
    full = np.zeros(len(pool["x"]), dtype=np.float32)
    full.reshape(q, slots, alts)[:, :, 1:] = pair_scores.reshape(q, slots, alts - 1)
    atomic_npz(out_path, score=full.astype(np.float32), pair_score=pair_scores.astype(np.float32))
    return {"reused": False, "score": artifact(out_path), "score_min": float(full.min()), "score_max": float(full.max())}


def aggregate_pool_evidence(pool_path: str, score_path: str, q_count: int) -> Dict[Tuple[int, int], Dict[str, float]]:
    pool = load_npz(pool_path, allow_pickle=False)
    scores = np.load(score_path, allow_pickle=False)["score"].astype(np.float32)
    q, slots, alts = [int(x) for x in pool["shape"]]
    out: Dict[Tuple[int, int], Dict[str, float]] = {}
    s3 = scores.reshape(q, slots, alts)
    for qi in range(min(q, q_count)):
        for slot in range(slots):
            base = (qi * slots + slot) * alts
            orig = float(s3[qi, slot, 0])
            margin = float(np.max(s3[qi, slot]) - orig)
            for alt in range(alts):
                idx = base + alt
                gid = int(pool["group_id"][idx])
                rec = out.setdefault((qi, gid), {"max_score": -1e9, "max_margin": -1e9, "count": 0.0})
                rec["max_score"] = max(rec["max_score"], float(scores[idx]))
                rec["max_margin"] = max(rec["max_margin"], margin)
                rec["count"] += 1.0
    return out


def gt_video_by_query(cache: Dict[str, np.ndarray]) -> np.ndarray:
    q_count = len(cache["desc_ids"])
    gt = np.full(q_count, -1, dtype=np.int64)
    keys = cache["video_group_keys"].astype(np.int64)
    rel = cache["label_relevant"].astype(np.float32)
    for gid, is_rel in enumerate(rel):
        if is_rel > 0.5:
            q = int(keys[gid, 0])
            if 0 <= q < q_count and gt[q] < 0:
                gt[q] = int(keys[gid, 1])
    return gt


def cache_video_groups_for_query(cache: Dict[str, np.ndarray], q: int) -> List[int]:
    offsets = cache["desc_offsets"].astype(np.int64)
    rows = np.arange(int(offsets[q]), int(offsets[q + 1]), dtype=np.int64)
    gids = cache["row_group_id"][rows].astype(np.int64)
    first: Dict[int, int] = {}
    s_c4 = cache["s_c4_final"].astype(np.float32)
    for row, gid in zip(rows, gids):
        old = first.get(int(gid))
        if old is None or s_c4[row] > s_c4[old]:
            first[int(gid)] = int(row)
    return [gid for gid, _row in sorted(first.items(), key=lambda kv: -float(s_c4[kv[1]]))]


def baseline_candidates(
    cache: Dict[str, np.ndarray],
    gt_ts: Dict[int, Any],
    top_n: int,
    max_after_nms: int,
    nms_thd: float,
) -> Dict[str, Any]:
    q_count = len(cache["desc_ids"])
    offsets = cache["desc_offsets"].astype(np.int64)
    video_idx = cache["video_idx"].astype(np.int64)
    start_idx = cache["start_idx"].astype(np.int64)
    end_idx = cache["end_idx"].astype(np.int64)
    s_c4 = cache["s_c4_final"].astype(np.float32)
    gt_vid = gt_video_by_query(cache)
    labels05: List[List[bool]] = []
    labels07: List[List[bool]] = []
    video_hits = {1: 0, 5: 0, 10: 0, 100: 0}
    positives = []
    for q in range(q_count):
        rows = np.arange(int(offsets[q]), int(offsets[q + 1]), dtype=np.int64)
        order = rows[np.argsort(-s_c4[rows], kind="stable")[:top_n]]
        kept: List[int] = []
        by_video: Dict[int, List[int]] = {}
        for row in order:
            vid = int(video_idx[row])
            si, ei = int(start_idx[row]), int(end_idx[row])
            suppress = False
            for prior in by_video.get(vid, []):
                if span_iou_idx(si, ei, int(start_idx[prior]), int(end_idx[prior])) > nms_thd:
                    suppress = True
                    break
            if not suppress:
                kept.append(int(row))
                by_video.setdefault(vid, []).append(int(row))
                if len(kept) >= max_after_nms:
                    break
        l05 = []
        l07 = []
        vids = [int(video_idx[row]) for row in kept]
        for row in kept:
            same = int(video_idx[row]) == int(gt_vid[q])
            l05.append(bool(same and correctness_for_ts(int(start_idx[row]), int(end_idx[row]), gt_ts[q], 0.5)))
            l07.append(bool(same and correctness_for_ts(int(start_idx[row]), int(end_idx[row]), gt_ts[q], 0.7)))
        for k in video_hits:
            video_hits[k] += int(int(gt_vid[q]) in vids[:k])
        labels05.append(l05)
        labels07.append(l07)
        positives.append(bool(any(l05[:100]) or any(l07[:100])))
    metrics = selected_metrics(labels05, labels07)
    return {
        "metrics": metrics,
        "video_metrics": {f"GT_video_R@{k}": 100.0 * v / max(q_count, 1) for k, v in video_hits.items()},
        "positive_top100_queries": int(sum(positives)),
        "positive_top100_flags": positives,
    }


def oracle_video_rerank(
    cache: Dict[str, np.ndarray],
    gt_ts: Dict[int, Any],
    mode: str,
    top_n: int,
    max_after_nms: int,
    nms_thd: float,
) -> Dict[str, Any]:
    q_count = len(cache["desc_ids"])
    offsets = cache["desc_offsets"].astype(np.int64)
    row_gid = cache["row_group_id"].astype(np.int64)
    video_idx = cache["video_idx"].astype(np.int64)
    start_idx = cache["start_idx"].astype(np.int64)
    end_idx = cache["end_idx"].astype(np.int64)
    s_c4 = cache["s_c4_final"].astype(np.float32)
    y05 = cache["y_joint_05"].astype(np.float32)
    y07 = cache["y_joint_07"].astype(np.float32)
    label_any05 = cache["label_any_05"].astype(np.float32)
    label_any07 = cache["label_any_07"].astype(np.float32)
    label_best_iou = cache["label_best_iou"].astype(np.float32)
    gt_vid = gt_video_by_query(cache)
    labels05: List[List[bool]] = []
    labels07: List[List[bool]] = []
    video_hits = {1: 0, 5: 0, 10: 0, 100: 0}
    raw200_has_gt = raw200_missing = top100_miss = low_rank_gt = 0
    for q in range(q_count):
        rows = np.arange(int(offsets[q]), int(offsets[q + 1]), dtype=np.int64)
        base_ranked = rows[np.argsort(-s_c4[rows], kind="stable")]
        base_vids = [int(video_idx[r]) for r in base_ranked]
        if int(gt_vid[q]) in base_vids:
            rank = base_vids.index(int(gt_vid[q])) + 1
            raw200_has_gt += 1
            top100_miss += int(rank > 100)
            low_rank_gt += int(rank > 10)
        else:
            raw200_missing += 1
        group_rows: Dict[int, List[int]] = {}
        for row in rows:
            group_rows.setdefault(int(row_gid[row]), []).append(int(row))
        group_scores = []
        for gid, rs in group_rows.items():
            arr = np.asarray(rs, dtype=np.int64)
            if mode == "moment_aware":
                score = 10.0 * float(label_best_iou[gid]) + float(np.max(s_c4[arr]))
            elif mode == "partial_relevance_mil":
                score = 20.0 * float(label_any07[gid]) + 8.0 * float(label_any05[gid]) + topk_mean(y07[arr] + 0.5 * y05[arr], 3)
            elif mode == "proposal_confidence":
                score = 8.0 * float(np.max(y07[arr])) + 4.0 * float(np.max(y05[arr])) + float(np.max(s_c4[arr]))
            else:
                raise ValueError(mode)
            group_scores.append((gid, score))
        ordered_groups = [gid for gid, _ in sorted(group_scores, key=lambda x: -x[1])]
        tuple_rows: List[int] = []
        for gid in ordered_groups:
            rs = np.asarray(group_rows[gid], dtype=np.int64)
            if mode == "proposal_confidence":
                order = rs[np.argsort(-(y07[rs] * 4.0 + y05[rs] * 2.0 + s_c4[rs]), kind="stable")]
            else:
                order = rs[np.argsort(-s_c4[rs], kind="stable")]
            tuple_rows.extend([int(r) for r in order])
            if len(tuple_rows) >= top_n:
                break
        kept: List[int] = []
        by_video: Dict[int, List[int]] = {}
        for row in tuple_rows[:top_n]:
            vid = int(video_idx[row])
            si, ei = int(start_idx[row]), int(end_idx[row])
            suppress = False
            for prior in by_video.get(vid, []):
                if span_iou_idx(si, ei, int(start_idx[prior]), int(end_idx[prior])) > nms_thd:
                    suppress = True
                    break
            if not suppress:
                kept.append(int(row))
                by_video.setdefault(vid, []).append(int(row))
                if len(kept) >= max_after_nms:
                    break
        l05, l07 = [], []
        vids = [int(video_idx[row]) for row in kept]
        for row in kept:
            same = int(video_idx[row]) == int(gt_vid[q])
            l05.append(bool(same and correctness_for_ts(int(start_idx[row]), int(end_idx[row]), gt_ts[q], 0.5)))
            l07.append(bool(same and correctness_for_ts(int(start_idx[row]), int(end_idx[row]), gt_ts[q], 0.7)))
        for k in video_hits:
            video_hits[k] += int(int(gt_vid[q]) in vids[:k])
        labels05.append(l05)
        labels07.append(l07)
    return {
        "mode": mode,
        "metrics": selected_metrics(labels05, labels07),
        "video_metrics": {f"GT_video_R@{k}": 100.0 * v / max(q_count, 1) for k, v in video_hits.items()},
        "raw200_has_gt_video": int(raw200_has_gt),
        "raw200_missing_gt_video": int(raw200_missing),
        "gt_video_in_raw200_but_missing_top100": int(top100_miss),
        "gt_video_low_rank_gt10": int(low_rank_gt),
    }


def write_start_state(args: argparse.Namespace) -> Dict[str, Any]:
    state = {
        "stage": "C6-C paper-grounded train_fit_train_calib_only",
        "date": "2026-06-26",
        "current_promoted_system": "C6-B2 pairwise_main_0005",
        "stable_safety_anchor_system": "C6-B1-lite c6b1r1_0057",
        "current_promoted_secondary_anchor": "C6-B2 pairwise_main_0005",
        "C6_B2_role_in_C6_C": "span_utility_evidence_only_not_primary_video_anchor",
        "C4_role_in_C6_C": "base_video_score_and_raw_candidate_source",
        "baseline_correction_from_user_attachment": {
            "attachment_claimed_stable_promoted_system": "C6-B1-lite c6b1r1_0057",
            "correct_current_promoted_system_from_final_handoff": "C6-B2 pairwise_main_0005",
            "decision": "Use C6-B1-lite as primary safety anchor to avoid compounding B2 top-k tradeoff with video-slot drift; report and guard against C6-B2 separately."
        },
        "official_val_used_for_c6c": False,
        "official_val_authorized": False,
        "post_val_adjustment": False,
        "second_official_val": False,
        "C6_B1_lite_modified": False,
        "C6_B2_modified": False,
        "C4_final_modified": False,
        "NMS_modified": False,
        "evaluator_modified": False,
        "full_backbone_finetuning": False,
        "paper_grounding": {
            "CONQUER": "two-stage VCMR and contextual query-aware ranking retained as base.",
            "MA_VR": "moment predictions condition video relevance via moment-aware evidence/gating.",
            "PREM_PRVR": "partial relevance motivates bag-of-moments MIL pooling at video level.",
            "BSN_BMN": "boundary/proposal confidence is used as feature evidence, not a new backbone.",
            "GenSpan": "design note only; no generator or C6-D training in this stage."
        },
    }
    audit_dir = Path(args.audit_dir)
    atomic_json(audit_dir / "C6_C_START_STATE.json", state)
    atomic_text(audit_dir / "C6_C_START_STATE.md", "# C6-C start state\n\n```json\n" + json.dumps(state, indent=2, ensure_ascii=False) + "\n```\n")
    return state


def run_c0(args: argparse.Namespace) -> Dict[str, Any]:
    cache = load_npz(args.calib_cache, allow_pickle=True)
    pool = load_npz(args.calib_pool, allow_pickle=False)
    gt_ts = load_gt_ts(args.gt_jsonl, cache["desc_ids"])
    b1_cfg, b2_cfg = load_baseline_configs(args)
    b1_scores = np.load(args.b1_calib_scores, allow_pickle=False)["score"].astype(np.float32)
    b2_scores = np.load(args.b2_calib_scores, allow_pickle=False)["score"].astype(np.float32)
    c4_ev = baseline_candidates(cache, gt_ts, args.effective_top_n, args.max_after_nms, args.nms_thd)
    c4_flags = c4_ev.pop("positive_top100_flags", [])
    b1_ev = evaluate_policy(
        cache=cache, pool=pool, pool_score=b1_scores, gt_ts_by_query=gt_ts,
        apply_slots=int(b1_cfg["apply_slots"]), threshold0=float(b1_cfg["threshold0"]),
        threshold_rest=float(b1_cfg["threshold_rest"]), max_replacements=int(b1_cfg["max_replacements"]),
        effective_top_n=args.effective_top_n, max_after_nms=args.max_after_nms, nms_thd=args.nms_thd,
    )
    b2_ev = evaluate_policy(
        cache=cache, pool=pool, pool_score=b2_scores, gt_ts_by_query=gt_ts,
        apply_slots=int(b2_cfg["apply_slots"]), threshold0=float(b2_cfg["threshold0"]),
        threshold_rest=float(b2_cfg["threshold_rest"]), max_replacements=int(b2_cfg["max_replacements"]),
        effective_top_n=args.effective_top_n, max_after_nms=args.max_after_nms, nms_thd=args.nms_thd,
    )
    oracles = {
        name: oracle_video_rerank(cache, gt_ts, name, 200, args.max_after_nms, args.nms_thd)
        for name in ["moment_aware", "partial_relevance_mil", "proposal_confidence"]
    }
    deltas = {
        name: {
            "vs_C6_B1_lite": metric_delta(rec["metrics"], b1_ev["metrics"]),
            "vs_C6_B2": metric_delta(rec["metrics"], b2_ev["metrics"]),
            "vs_C4_final": metric_delta(rec["metrics"], c4_ev["metrics"]),
        }
        for name, rec in oracles.items()
    }
    ma_gate = deltas["moment_aware"]["vs_C6_B1_lite"]["0.7-r1"] >= args.c0_min_oracle_delta
    mil_gate = deltas["partial_relevance_mil"]["vs_C6_B1_lite"]["0.7-r1"] >= args.c0_min_oracle_delta
    raw_gate = oracles["moment_aware"]["raw200_has_gt_video"] > 0
    status = "C6_C0_PASS_TO_C6_C1" if (ma_gate and mil_gate and raw_gate) else "C6_C0_NEGATIVE"
    payload = {
        "status": status,
        "scope": "train_calib diagnostic only",
        "official_val_used": False,
        "primary_safety_anchor": "C6-B1-lite c6b1r1_0057",
        "current_promoted_report_anchor": "C6-B2 pairwise_main_0005",
        "baselines": {
            "C4_final_pseudo": c4_ev,
            "C6_B1_lite": b1_ev,
            "C6_B2": b2_ev,
        },
        "oracles": oracles,
        "oracle_deltas": deltas,
        "gates": {
            "raw200_has_rescuable_gt_video": bool(raw_gate),
            "moment_aware_delta_0.7_r1_vs_B1_ge_min": bool(ma_gate),
            "partial_relevance_delta_0.7_r1_vs_B1_ge_min": bool(mil_gate),
            "min_oracle_delta": float(args.c0_min_oracle_delta),
            "official_val_used": False,
        },
        "diagnostic_counts": {
            "C4_positive_top100_query_count": int(sum(c4_flags)),
            "query_count": int(len(c4_flags)),
        },
    }
    audit_dir = Path(args.audit_dir)
    atomic_json(audit_dir / "C6_C0_VIDEO_RESIDUAL_ORACLE.json", payload)
    md = "# C6-C0 video residual oracle\n\n"
    md += f"- Status: `{status}`\n- Scope: `train_calib only`\n- Official val used: `false`\n"
    md += "- Primary safety anchor: `C6-B1-lite c6b1r1_0057`\n"
    md += "- C6-B2 use: `report anchor and span utility evidence, not video safety anchor`\n\n"
    for name, rec in oracles.items():
        md += f"## {name}\n\n"
        md += "Metrics:\n\n```json\n" + json.dumps(rec["metrics"], indent=2) + "\n```\n"
        md += "Delta vs C6-B1-lite:\n\n```json\n" + json.dumps(deltas[name]["vs_C6_B1_lite"], indent=2) + "\n```\n"
    atomic_text(audit_dir / "C6_C0_VIDEO_RESIDUAL_ORACLE.md", md)
    return payload


def build_feature_split(
    *,
    split: str,
    cache_path: str,
    temporal_path: str,
    pool_path: str,
    b1_score_path: str,
    b2_score_path: str,
    out_path: Path,
) -> Dict[str, Any]:
    if out_path.exists():
        return {"split": split, "reused": True, "features": artifact(out_path)}
    cache = load_npz(cache_path, allow_pickle=True)
    temporal = load_npz(temporal_path, allow_pickle=False)
    q_count = len(cache["desc_ids"])
    b1_ev = aggregate_pool_evidence(pool_path, b1_score_path, q_count)
    b2_ev = aggregate_pool_evidence(pool_path, b2_score_path, q_count)
    offsets = cache["desc_offsets"].astype(np.int64)
    row_gid = cache["row_group_id"].astype(np.int64)
    s_c4 = cache["s_c4_final"].astype(np.float32)
    start_idx = cache["start_idx"].astype(np.int64)
    end_idx = cache["end_idx"].astype(np.int64)
    video_idx_rows = cache["video_idx"].astype(np.int64)
    video_features = cache["video_features"].astype(np.float32)
    label_relevant = cache["label_relevant"].astype(np.float32)
    label_any05 = cache["label_any_05"].astype(np.float32)
    label_any07 = cache["label_any_07"].astype(np.float32)
    label_best_iou = cache["label_best_iou"].astype(np.float32)
    keys = cache["video_group_keys"].astype(np.int64)
    p_ctx = temporal["p_ctx"].astype(np.float32)
    p_b = temporal["p_b"].astype(np.float32)
    p_e = temporal["p_e"].astype(np.float32)
    tlen = temporal["temporal_length"].astype(np.int64)
    feats: List[np.ndarray] = []
    q_arr: List[int] = []
    gid_arr: List[int] = []
    vid_arr: List[int] = []
    base_rank_arr: List[int] = []
    best_row_arr: List[int] = []
    for q in range(q_count):
        rows = np.arange(int(offsets[q]), int(offsets[q + 1]), dtype=np.int64)
        order = rows[np.argsort(-s_c4[rows], kind="stable")]
        group_rank: Dict[int, int] = {}
        group_rows: Dict[int, List[int]] = {}
        for rank, row in enumerate(order):
            gid = int(row_gid[row])
            group_rank.setdefault(gid, rank)
            group_rows.setdefault(gid, []).append(int(row))
        for gid, rs_list in group_rows.items():
            rs = np.asarray(rs_list, dtype=np.int64)
            scores = s_c4[rs]
            best_local = int(rs[int(np.argmax(scores))])
            score_sorted = np.sort(scores)[::-1]
            t = int(tlen[gid])
            ctx = p_ctx[gid, :t]
            pb = p_b[gid, :t]
            pe = p_e[gid, :t]
            b1 = b1_ev.get((q, gid), {"max_score": 0.0, "max_margin": 0.0, "count": 0.0})
            b2 = b2_ev.get((q, gid), {"max_score": 0.0, "max_margin": 0.0, "count": 0.0})
            lengths = (end_idx[rs] - start_idx[rs] + 1).astype(np.float32)
            centers = 0.5 * (start_idx[rs] + end_idx[rs]).astype(np.float32)
            base_rank = int(group_rank[gid])
            engineered = np.asarray([
                float(np.max(scores)),
                float(np.mean(scores)),
                float(np.std(scores)),
                topk_mean(scores, 3),
                topk_mean(scores, 5),
                logsumexp(scores),
                float(score_sorted[0] - score_sorted[1]) if len(score_sorted) > 1 else 0.0,
                1.0 / float(base_rank + 1),
                float(base_rank) / 100.0,
                float(len(rs)) / 100.0,
                float(np.mean(lengths)) / 100.0,
                float(np.max(lengths)) / 100.0,
                float(np.mean(centers)) / 100.0,
                float(np.max(ctx)) if len(ctx) else 0.0,
                float(np.mean(ctx)) if len(ctx) else 0.0,
                entropy_from_probs(ctx) / math.log(max(t, 2)),
                float(np.max(pb)) if len(pb) else 0.0,
                float(np.max(pe)) if len(pe) else 0.0,
                entropy_from_probs(pb) / math.log(max(t, 2)),
                entropy_from_probs(pe) / math.log(max(t, 2)),
                float(b1["max_score"]),
                float(b1["max_margin"]),
                float(b1["count"]) / 40.0,
                float(b2["max_score"]),
                float(b2["max_margin"]),
                float(b2["count"]) / 40.0,
                float(b2["max_score"] - b1["max_score"]),
                float(b2["max_margin"] - b1["max_margin"]),
                float(np.max(scores) - np.mean(s_c4[rows])),
                float(np.max(scores) - np.max(s_c4[rows])),
            ], dtype=np.float32)
            feats.append(np.concatenate([engineered, video_features[gid].astype(np.float32)], axis=0))
            q_arr.append(q)
            gid_arr.append(gid)
            vid_arr.append(int(keys[gid, 1]) if gid < len(keys) else int(video_idx_rows[best_local]))
            base_rank_arr.append(base_rank)
            best_row_arr.append(best_local)
    x = np.stack(feats).astype(np.float32)
    gids = np.asarray(gid_arr, dtype=np.int32)
    atomic_npz(
        out_path,
        x=x,
        y_video=label_relevant[gids].astype(np.float32),
        y_any05=label_any05[gids].astype(np.float32),
        y_any07=label_any07[gids].astype(np.float32),
        y_best_iou=label_best_iou[gids].astype(np.float32),
        query=np.asarray(q_arr, dtype=np.int32),
        group_id=gids,
        video_idx=np.asarray(vid_arr, dtype=np.int32),
        base_rank=np.asarray(base_rank_arr, dtype=np.int16),
        best_row=np.asarray(best_row_arr, dtype=np.int32),
        feature_names=np.asarray(
            [
                "base_max", "base_mean", "base_std", "base_top3_mean", "base_top5_mean", "base_logsumexp",
                "base_top1_top2_margin", "base_rank_inv", "base_rank_norm", "span_count_norm",
                "span_length_mean", "span_length_max", "span_center_mean", "ctx_max", "ctx_mean",
                "ctx_entropy", "start_prior_max", "end_prior_max", "start_prior_entropy", "end_prior_entropy",
                "b1_score_max", "b1_margin_max", "b1_pool_count_norm", "b2_score_max", "b2_margin_max",
                "b2_pool_count_norm", "b2_minus_b1_score", "b2_minus_b1_margin", "base_vs_query_mean",
                "base_vs_query_top",
            ] + [f"video_feature_{i}" for i in range(video_features.shape[1])],
            dtype=object,
        ),
    )
    return {
        "split": split,
        "reused": False,
        "features": artifact(out_path),
        "examples": int(len(x)),
        "feature_dim": int(x.shape[1]),
        "positive_video_rate": float(label_relevant[gids].mean()),
        "any07_rate": float(label_any07[gids].mean()),
    }


class VideoFeatureDataset(Dataset):
    def __init__(self, feature_npz: str, mean: np.ndarray | None = None, std: np.ndarray | None = None):
        z = np.load(feature_npz, allow_pickle=True)
        self.x = z["x"].astype(np.float32)
        self.y_video = z["y_video"].astype(np.float32)
        self.y_any07 = z["y_any07"].astype(np.float32)
        self.y_any05 = z["y_any05"].astype(np.float32)
        self.y_best_iou = z["y_best_iou"].astype(np.float32)
        self.base_rank = z["base_rank"].astype(np.int64)
        if mean is None:
            mean = self.x.mean(axis=0)
            std = self.x.std(axis=0)
        self.mean = mean.astype(np.float32)
        self.std = np.maximum(std.astype(np.float32), EPS)

    def __len__(self) -> int:
        return int(len(self.x))

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = (self.x[idx] - self.mean) / self.std
        risk = (1.0 - self.y_video[idx]) * float(self.base_rank[idx] < 5)
        y = np.asarray([self.y_video[idx], self.y_any07[idx], self.y_any05[idx], self.y_best_iou[idx], risk], dtype=np.float32)
        return torch.from_numpy(x.astype(np.float32)), torch.from_numpy(y)


class C6CVideoReranker(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 512, layers: int = 4, dropout: float = 0.12):
        super().__init__()
        mods: List[nn.Module] = []
        d = in_dim
        for _ in range(layers):
            mods += [nn.Linear(d, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(dropout)]
            d = hidden
        self.backbone = nn.Sequential(*mods)
        self.relevance = nn.Linear(hidden, 1)
        self.moment = nn.Linear(hidden, 1)
        self.iou = nn.Linear(hidden, 1)
        self.risk = nn.Linear(hidden, 1)
        self.gate = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        h = self.backbone(x)
        return {
            "relevance": self.relevance(h).squeeze(-1),
            "moment": self.moment(h).squeeze(-1),
            "iou": self.iou(h).squeeze(-1),
            "risk": self.risk(h).squeeze(-1),
            "gate": self.gate(h).squeeze(-1),
        }


def train_video_model(args: argparse.Namespace) -> Dict[str, Any]:
    out_dir = Path(args.output_dir)
    model_path = out_dir / "arms" / "mavr_pr_mil" / "model_best.pt"
    hist_path = out_dir / "arms" / "mavr_pr_mil" / "history.json"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    if model_path.exists() and hist_path.exists():
        return {"reused": True, "model": artifact(model_path), "history": json.loads(hist_path.read_text(encoding="utf-8"))}
    ds = VideoFeatureDataset(str(out_dir / "train_fit_video_features.npz"))
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    model = C6CVideoReranker(ds.x.shape[1], hidden=args.hidden, layers=args.layers, dropout=args.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))
    pos_weight = torch.tensor([(1.0 - ds.y_video.mean()) / max(ds.y_video.mean(), EPS)], dtype=torch.float32, device=device)
    moment_weight = torch.tensor([(1.0 - ds.y_any07.mean()) / max(ds.y_any07.mean(), EPS)], dtype=torch.float32, device=device)
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        t0 = time.time()
        for xb, yb in dl:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                out = model(xb)
                rel_loss = F.binary_cross_entropy_with_logits(out["relevance"], yb[:, 0], pos_weight=pos_weight)
                mom07_loss = F.binary_cross_entropy_with_logits(out["moment"], yb[:, 1], pos_weight=moment_weight)
                mom05_loss = F.binary_cross_entropy_with_logits(out["iou"], yb[:, 2])
                iou_loss = F.smooth_l1_loss(torch.sigmoid(out["iou"]), yb[:, 3])
                risk_loss = F.binary_cross_entropy_with_logits(out["risk"], yb[:, 4])
                gate = torch.sigmoid(out["gate"])
                gate_reg = torch.relu(gate.mean() - args.gate_mean_cap) + 0.1 * gate.var()
                loss = rel_loss + 0.7 * mom07_loss + 0.35 * mom05_loss + 0.35 * iou_loss + 0.45 * risk_loss + 0.25 * gate_reg
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            losses.append(float(loss.detach().cpu()))
        rec = {
            "epoch": epoch,
            "loss": float(np.mean(losses)),
            "elapsed_sec": float(time.time() - t0),
            "examples": int(len(ds)),
            "device": str(device),
        }
        history.append(rec)
        atomic_json(hist_path, history)
        print(json.dumps({"stage": "c6c_train", **rec}), flush=True)
    torch.save({
        "model_state": model.state_dict(),
        "mean": ds.mean,
        "std": ds.std,
        "in_dim": int(ds.x.shape[1]),
        "hidden": args.hidden,
        "layers": args.layers,
        "dropout": args.dropout,
        "epochs": args.epochs,
        "pos_weight": float(pos_weight.item()),
        "moment_weight": float(moment_weight.item()),
    }, model_path)
    return {"reused": False, "model": artifact(model_path), "history": history}


@torch.no_grad()
def score_video_features(args: argparse.Namespace, split: str) -> Dict[str, Any]:
    out_dir = Path(args.output_dir)
    score_path = out_dir / f"{split}_video_scores.npz"
    if score_path.exists():
        return {"split": split, "reused": True, "scores": artifact(score_path)}
    feat_path = out_dir / f"{split}_video_features.npz"
    z = np.load(feat_path, allow_pickle=True)
    ckpt = torch.load(out_dir / "arms" / "mavr_pr_mil" / "model_best.pt", map_location="cpu")
    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    model = C6CVideoReranker(int(ckpt["in_dim"]), hidden=int(ckpt["hidden"]), layers=int(ckpt["layers"]), dropout=float(ckpt["dropout"])).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    x = z["x"].astype(np.float32)
    x = (x - ckpt["mean"]) / np.maximum(ckpt["std"], EPS)
    rel = np.empty(len(x), dtype=np.float32)
    mom = np.empty(len(x), dtype=np.float32)
    iou = np.empty(len(x), dtype=np.float32)
    risk = np.empty(len(x), dtype=np.float32)
    gate = np.empty(len(x), dtype=np.float32)
    for start in range(0, len(x), args.score_batch_size):
        xb = torch.from_numpy(x[start:start + args.score_batch_size]).to(device)
        with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
            out = model(xb)
        n = len(xb)
        rel[start:start+n] = out["relevance"].float().cpu().numpy()
        mom[start:start+n] = out["moment"].float().cpu().numpy()
        iou[start:start+n] = out["iou"].float().cpu().numpy()
        risk[start:start+n] = out["risk"].float().cpu().numpy()
        gate[start:start+n] = torch.sigmoid(out["gate"]).float().cpu().numpy()
    atomic_npz(score_path, relevance=rel, moment=mom, iou=iou, risk=risk, gate=gate, query=z["query"], group_id=z["group_id"], video_idx=z["video_idx"], base_rank=z["base_rank"])
    return {"split": split, "reused": False, "scores": artifact(score_path), "gate_mean": float(gate.mean())}


def make_anchor_candidates(
    *,
    q: int,
    cache: Dict[str, np.ndarray],
    pool: Dict[str, np.ndarray],
    pool_score: np.ndarray,
    policy_cfg: Dict[str, Any],
    effective_top_n: int,
) -> List[Tuple[int, int, int, int, float, int]]:
    offsets = cache["desc_offsets"].astype(np.int64)
    row_gid = cache["row_group_id"].astype(np.int64)
    video_idx = cache["video_idx"].astype(np.int64)
    start_idx = cache["start_idx"].astype(np.int64)
    end_idx = cache["end_idx"].astype(np.int64)
    s_c4 = cache["s_c4_final"].astype(np.float32)
    rows = np.arange(int(offsets[q]), int(offsets[q + 1]), dtype=np.int64)
    base_order = rows[np.argsort(-s_c4[rows], kind="stable")[:effective_top_n]]
    pq, slots, alts = [int(x) for x in pool["shape"]]
    score3 = pool_score.reshape(pq, slots, alts)
    out: List[Tuple[int, int, int, int, float, int]] = []
    replaced = 0
    apply_slots = int(policy_cfg.get("apply_slots", 0))
    max_repl = int(policy_cfg.get("max_replacements", 0))
    for slot_rank, row in enumerate(base_order):
        vid = int(video_idx[row])
        gid = int(row_gid[row])
        si = int(start_idx[row])
        ei = int(end_idx[row])
        if q < pq and slot_rank < min(slots, apply_slots) and replaced < max_repl:
            scores = score3[q, slot_rank]
            best = int(np.argmax(scores))
            margin = float(scores[best] - scores[0])
            threshold = float(policy_cfg["threshold0"]) if slot_rank == 0 else float(policy_cfg["threshold_rest"])
            if best != 0 and margin > threshold:
                base = (q * slots + slot_rank) * alts
                si = int(pool["start_idx"][base + best])
                ei = int(pool["end_idx"][base + best])
                replaced += 1
        out.append((gid, vid, si, ei, float(s_c4[row]), int(row)))
    return out


def precompute_anchor_candidates(
    *,
    cache: Dict[str, np.ndarray],
    pool: Dict[str, np.ndarray],
    pool_score: np.ndarray,
    policy_cfg: Dict[str, Any],
    effective_top_n: int,
) -> List[List[Tuple[int, int, int, int, float, int]]]:
    return [
        make_anchor_candidates(
            q=q,
            cache=cache,
            pool=pool,
            pool_score=pool_score,
            policy_cfg=policy_cfg,
            effective_top_n=effective_top_n,
        )
        for q in range(len(cache["desc_ids"]))
    ]


def precompute_anchor_states(
    *,
    anchor_by_q: List[List[Tuple[int, int, int, int, float, int]]],
    gt_ts: Dict[int, Any],
    gt_vid: np.ndarray,
    nms_thd: float,
    max_after_nms: int,
) -> List[Dict[str, Any]]:
    states: List[Dict[str, Any]] = []
    for q, anchor_candidates in enumerate(anchor_by_q):
        base_group_order: List[int] = []
        seen = set()
        group_candidates: Dict[int, List[Tuple[int, int, int, int, float, int]]] = {}
        for cand in anchor_candidates:
            gid = cand[0]
            if gid not in seen:
                seen.add(gid)
                base_group_order.append(gid)
            group_candidates.setdefault(gid, []).append(cand)

        base_kept: List[Tuple[int, int, int, int, float, int]] = []
        base_by_video: Dict[int, List[int]] = {}
        for cand in anchor_candidates:
            _gid, vid, si, ei, _score, _row = cand
            suppress = False
            for prior_idx in base_by_video.get(vid, []):
                prior = base_kept[prior_idx]
                if span_iou_idx(si, ei, prior[2], prior[3]) > nms_thd:
                    suppress = True
                    break
            if not suppress:
                base_by_video.setdefault(vid, []).append(len(base_kept))
                base_kept.append(cand)
                if len(base_kept) >= max_after_nms:
                    break
        base_pos = False
        for _gid, vid, si, ei, _score, _row in base_kept:
            same = int(vid) == int(gt_vid[q])
            if same and (correctness_for_ts(si, ei, gt_ts[q], 0.5) or correctness_for_ts(si, ei, gt_ts[q], 0.7)):
                base_pos = True
                break
        states.append({
            "base_group_order": base_group_order,
            "group_candidates": group_candidates,
            "base_pos": bool(base_pos),
        })
    return states


def evaluate_video_scores(args: argparse.Namespace, score_arrays: Dict[str, np.ndarray], config: Dict[str, Any], anchor_bundle: Dict[str, Any]) -> Dict[str, Any]:
    gt_ts = anchor_bundle["gt_ts"]
    gt_vid = anchor_bundle["gt_vid"]
    anchor_states = anchor_bundle["b2_anchor_states"] if config.get("span_anchor", "B1") == "B2" else anchor_bundle["b1_anchor_states"]
    q_count = len(anchor_states)
    group_score: Dict[int, Tuple[float, float]] = {}
    alpha = float(config["alpha"])
    beta = float(config["beta"])
    gamma = float(config["gamma"])
    rho = float(config["rho"])
    gate_threshold = float(config["gate_threshold"])
    gids = score_arrays["group_id"].astype(np.int64)
    base_rank = score_arrays["base_rank"].astype(np.int64)
    for i, gid in enumerate(gids):
        base_component = 1.0 / float(int(base_rank[i]) + 1)
        gate = float(score_arrays["gate"][i])
        residual = beta * gate * float(score_arrays["relevance"][i]) + gamma * gate * (float(score_arrays["moment"][i]) + 0.35 * float(score_arrays["iou"][i])) - rho * gate * float(score_arrays["risk"][i])
        if gate < gate_threshold:
            residual = 0.0
        group_score[int(gid)] = (alpha * base_component + residual, gate)
    labels05: List[List[bool]] = []
    labels07: List[List[bool]] = []
    video_hits = {1: 0, 5: 0, 10: 0, 100: 0}
    hard_base_pos = []
    hard_new_pos = []
    video_slot_changed = 0
    total_slots = 0
    invalid = 0
    dup_after_nms = 0
    for q in range(q_count):
        state = anchor_states[q]
        base_group_order = state["base_group_order"]
        group_candidates = state["group_candidates"]
        reranked = sorted(base_group_order, key=lambda gid: -group_score.get(gid, (0.0, 0.0))[0])
        max_changes = int(config["max_video_slot_changes"])
        final_groups = list(base_group_order)
        changes = 0
        for pos, gid in enumerate(reranked):
            if pos >= len(final_groups):
                break
            if final_groups[pos] == gid:
                continue
            if changes >= max_changes:
                break
            if group_score.get(gid, (0.0, 0.0))[1] < gate_threshold:
                continue
            old_pos = final_groups.index(gid)
            final_groups.pop(old_pos)
            final_groups.insert(pos, gid)
            changes += 1
        total_slots += len(base_group_order)
        video_slot_changed += sum(int(a != b) for a, b in zip(base_group_order, final_groups))
        tuple_rows = []
        for gid in final_groups:
            gscore = group_score.get(gid, (0.0, 0.0))[0]
            cands = group_candidates[gid]
            ordered = sorted(cands, key=lambda c: -((1.0 + float(config["lambda_span"])) * c[4] + float(config["video_weight"]) * gscore))
            tuple_rows.extend(ordered)
            if len(tuple_rows) >= args.effective_top_n:
                break
        kept: List[int] = []
        by_video: Dict[int, List[int]] = {}
        kept_cands: List[Tuple[int, int, int, int, float, int]] = []
        for cand in tuple_rows[:args.effective_top_n]:
            _gid, vid, si, ei, _score, _row = cand
            invalid += int(si < 0 or ei < si)
            suppress = False
            for prior in by_video.get(vid, []):
                pc = kept_cands[prior]
                if span_iou_idx(si, ei, pc[2], pc[3]) > args.nms_thd:
                    suppress = True
                    break
            if not suppress:
                kept.append(len(kept_cands))
                kept_cands.append(cand)
                by_video.setdefault(vid, []).append(len(kept_cands) - 1)
                if len(kept) >= args.max_after_nms:
                    break
        dup_after_nms += len(kept_cands) - len({(c[1], c[2], c[3]) for c in kept_cands})
        l05, l07 = [], []
        vids = [int(c[1]) for c in kept_cands]
        for _gid, vid, si, ei, _score, _row in kept_cands:
            same = int(vid) == int(gt_vid[q])
            l05.append(bool(same and correctness_for_ts(si, ei, gt_ts[q], 0.5)))
            l07.append(bool(same and correctness_for_ts(si, ei, gt_ts[q], 0.7)))
        for k in video_hits:
            video_hits[k] += int(int(gt_vid[q]) in vids[:k])
        labels05.append(l05)
        labels07.append(l07)
        hard_new_pos.append(bool(any(l05[:100]) or any(l07[:100])))
        hard_base_pos.append(bool(state["base_pos"]))
    metrics = selected_metrics(labels05, labels07)
    exits = sum(int(a and not b) for a, b in zip(hard_base_pos, hard_new_pos))
    entries = sum(int((not a) and b) for a, b in zip(hard_base_pos, hard_new_pos))
    return {
        "config": config,
        "metrics": metrics,
        "video_metrics": {f"GT_video_R@{k}": 100.0 * v / max(q_count, 1) for k, v in video_hits.items()},
        "movement": {
            "positive_video_entries": int(entries),
            "positive_video_exits": int(exits),
            "hard_positive_top100_query_exits": int(exits),
            "hard_positive_top100_query_entries": int(entries),
            "hard_positive_exit_ratio": float(exits / max(sum(hard_base_pos), 1)),
            "video_slot_drift_rate": float(video_slot_changed / max(total_slots, 1)),
            "video_multiset_drift_rate": float(video_slot_changed / max(total_slots, 1)),
            "invalid_span_count": int(invalid),
            "duplicate_span_count_after_nms": int(dup_after_nms),
        },
        "official_val_used": False,
        "span_anchor": config.get("span_anchor", "B1"),
    }


def selection_score(rec: Dict[str, Any], b1: Dict[str, float], b2: Dict[str, float]) -> float:
    d1 = metric_delta(rec["metrics"], b1)
    d2 = metric_delta(rec["metrics"], b2)
    mv = rec["movement"]
    return float(
        4.0 * d1["0.7-r1"]
        + 2.0 * d1["0.5-r1"]
        + 1.0 * d2["0.7-r1"]
        + 0.5 * d2["0.5-r1"]
        + 0.8 * max(0.0, rec["video_metrics"]["GT_video_R@1"])
        - 2.0 * max(0.0, -d1["0.7-r5"] - 0.10)
        - 1.5 * max(0.0, -d1["0.5-r5"] - 0.10)
        - 1.0 * max(0.0, mv["video_slot_drift_rate"] - 0.20)
        - 5.0 * mv["hard_positive_exit_ratio"]
    )


def evaluate_config_record(item: Tuple[int, Dict[str, Any]]) -> Dict[str, Any]:
    i, cfg = item
    if _SEARCH_ARGS is None or _SEARCH_SCORE_ARRAYS is None or _SEARCH_ANCHOR_BUNDLE is None or _SEARCH_B1_METRICS is None or _SEARCH_B2_METRICS is None:
        raise RuntimeError("search worker globals are not initialized")
    rec = evaluate_video_scores(_SEARCH_ARGS, _SEARCH_SCORE_ARRAYS, cfg, _SEARCH_ANCHOR_BUNDLE)
    d1 = metric_delta(rec["metrics"], _SEARCH_B1_METRICS)
    d2 = metric_delta(rec["metrics"], _SEARCH_B2_METRICS)
    rec["config_id"] = f"mavr_pr_mil_{i:04d}"
    rec["delta_vs_C6_B1_lite"] = d1
    rec["delta_vs_C6_B2"] = d2
    rec["selection_score"] = selection_score(rec, _SEARCH_B1_METRICS, _SEARCH_B2_METRICS)
    rec["feasible"] = bool(
        d1["0.7-r1"] >= _SEARCH_ARGS.freeze_min_r1_delta_vs_b1
        and d1["0.5-r1"] >= 0.0
        and d1["0.7-r5"] >= -0.10
        and d1["0.5-r5"] >= -0.10
        and rec["movement"]["positive_video_entries"] > rec["movement"]["positive_video_exits"]
        and rec["movement"]["hard_positive_exit_ratio"] <= 0.01
        and rec["movement"]["video_slot_drift_rate"] <= 0.20
        and rec["movement"]["invalid_span_count"] == 0
        and rec["movement"]["duplicate_span_count_after_nms"] == 0
    )
    return rec


def run_search(args: argparse.Namespace) -> Dict[str, Any]:
    global _SEARCH_ARGS, _SEARCH_SCORE_ARRAYS, _SEARCH_ANCHOR_BUNDLE, _SEARCH_B1_METRICS, _SEARCH_B2_METRICS
    cache = load_npz(args.calib_cache, allow_pickle=True)
    pool = load_npz(args.calib_pool, allow_pickle=False)
    gt_ts = load_gt_ts(args.gt_jsonl, cache["desc_ids"])
    b1_cfg, b2_cfg = load_baseline_configs(args)
    b1_scores = np.load(args.b1_calib_scores, allow_pickle=False)["score"].astype(np.float32)
    b2_scores = np.load(args.b2_calib_scores, allow_pickle=False)["score"].astype(np.float32)
    score_arrays = load_npz(Path(args.output_dir) / "train_calib_video_scores.npz", allow_pickle=False)
    t_anchor = time.time()
    b1_anchor_candidates = precompute_anchor_candidates(
        cache=cache, pool=pool, pool_score=b1_scores, policy_cfg=b1_cfg, effective_top_n=args.effective_top_n,
    )
    b2_anchor_candidates = precompute_anchor_candidates(
        cache=cache, pool=pool, pool_score=b2_scores, policy_cfg=b2_cfg, effective_top_n=args.effective_top_n,
    )
    gt_vid = gt_video_by_query(cache)
    anchor_bundle = {
        "pool": pool,
        "b1_scores": b1_scores,
        "b2_scores": b2_scores,
        "b1_cfg": b1_cfg,
        "b2_cfg": b2_cfg,
        "gt_ts": gt_ts,
        "gt_vid": gt_vid,
        "b1_anchor_states": precompute_anchor_states(
            anchor_by_q=b1_anchor_candidates, gt_ts=gt_ts, gt_vid=gt_vid,
            nms_thd=args.nms_thd, max_after_nms=args.max_after_nms,
        ),
        "b2_anchor_states": precompute_anchor_states(
            anchor_by_q=b2_anchor_candidates, gt_ts=gt_ts, gt_vid=gt_vid,
            nms_thd=args.nms_thd, max_after_nms=args.max_after_nms,
        ),
    }
    anchor_precompute_sec = float(time.time() - t_anchor)
    b1_ev = evaluate_policy(cache=cache, pool=pool, pool_score=b1_scores, gt_ts_by_query=gt_ts, apply_slots=int(b1_cfg["apply_slots"]), threshold0=float(b1_cfg["threshold0"]), threshold_rest=float(b1_cfg["threshold_rest"]), max_replacements=int(b1_cfg["max_replacements"]), effective_top_n=100, max_after_nms=100, nms_thd=0.7)
    b2_ev = evaluate_policy(cache=cache, pool=pool, pool_score=b2_scores, gt_ts_by_query=gt_ts, apply_slots=int(b2_cfg["apply_slots"]), threshold0=float(b2_cfg["threshold0"]), threshold_rest=float(b2_cfg["threshold_rest"]), max_replacements=int(b2_cfg["max_replacements"]), effective_top_n=100, max_after_nms=100, nms_thd=0.7)
    configs = []
    gate_values = score_arrays["gate"].astype(np.float32)
    gate_quantiles = [0.50, 0.70, 0.85, 0.93, 0.97, 0.99] if args.search_profile == "full" else [0.70, 0.85, 0.93, 0.99]
    gate_thresholds = sorted({
        float(np.quantile(gate_values, q))
        for q in gate_quantiles
    })
    # Keep a very low threshold neighbor to test broader motion, but derive
    # most thresholds from the model's actual calibration scale.
    gate_thresholds = [max(float(gate_values.min()), gate_thresholds[0] - 0.005)] + gate_thresholds
    if args.search_profile == "full":
        alpha_grid = [0.7, 0.85, 1.0]
        beta_grid = [0.05, 0.10, 0.20]
        gamma_grid = [0.05, 0.10, 0.20]
        rho_grid = [0.05, 0.10]
        max_change_grid = [1, 2, 3]
    else:
        alpha_grid = [0.85, 1.0]
        beta_grid = [0.10, 0.20]
        gamma_grid = [0.10, 0.20]
        rho_grid = [0.05]
        max_change_grid = [1, 2]
    for alpha in alpha_grid:
        for beta in beta_grid:
            for gamma in gamma_grid:
                for rho in rho_grid:
                    for gate_threshold in gate_thresholds:
                        for max_changes in max_change_grid:
                            for span_anchor in ["B1", "B2"]:
                                configs.append({
                                    "alpha": alpha, "beta": beta, "gamma": gamma, "rho": rho,
                                    "gate_threshold": gate_threshold,
                                    "max_video_slot_changes": max_changes,
                                    "lambda_span": 0.15,
                                    "video_weight": 1.0,
                                    "span_anchor": span_anchor,
                                })
    _SEARCH_ARGS = args
    _SEARCH_SCORE_ARRAYS = score_arrays
    _SEARCH_ANCHOR_BUNDLE = anchor_bundle
    _SEARCH_B1_METRICS = b1_ev["metrics"]
    _SEARCH_B2_METRICS = b2_ev["metrics"]
    if args.search_workers > 1:
        ctx = mp.get_context("fork")
        with ctx.Pool(processes=args.search_workers) as pool_mp:
            results = list(pool_mp.imap_unordered(evaluate_config_record, list(enumerate(configs)), chunksize=max(1, len(configs) // (args.search_workers * 4))))
        results = sorted(results, key=lambda r: r["config_id"])
    else:
        results = [evaluate_config_record(item) for item in enumerate(configs)]
    feasible = [r for r in results if r["feasible"]]
    best = max(feasible or results, key=lambda r: (r["feasible"], r["selection_score"], r["delta_vs_C6_B2"]["0.7-r1"]))
    if best["feasible"]:
        status = "C6_C1_FREEZE_REVIEW_PASS"
    elif best["delta_vs_C6_B1_lite"]["0.7-r1"] >= args.freeze_min_r1_delta_vs_b1 and (
        best["delta_vs_C6_B1_lite"]["0.7-r5"] < -0.10 or best["delta_vs_C6_B1_lite"]["0.5-r5"] < -0.10
    ):
        status = "C6_C1_R1_TRADEOFF"
    elif best["video_metrics"]["GT_video_R@1"] > b1_ev.get("video_metrics", {}).get("GT_video_R@1", 0.0):
        status = "C6_C1_VIDEO_ONLY_POSITIVE"
    else:
        status = "C6_C1_NEGATIVE"
    payload = {
        "status": status,
        "official_val_used": False,
        "primary_safety_anchor": "C6-B1-lite c6b1r1_0057",
        "current_promoted_report_anchor": "C6-B2 pairwise_main_0005",
        "baseline_C6_B1_lite": b1_ev,
        "baseline_C6_B2": b2_ev,
        "anchor_precompute_sec": anchor_precompute_sec,
        "gate_thresholds": gate_thresholds,
        "best": best,
        "grid_size": len(results),
        "search_profile": args.search_profile,
        "feasible_count": len(feasible),
        "results": results,
    }
    audit_dir = Path(args.audit_dir)
    atomic_json(audit_dir / "C6_C1_ARM_COMPARISON.json", payload)
    atomic_json(Path(args.output_dir) / "train_calib_grid_results.json", results)
    atomic_json(Path(args.output_dir) / "best_config.json", best)
    md = "# C6-C1 arm comparison\n\n"
    md += f"- Status: `{status}`\n- Official val used: `false`\n- Grid size: `{len(results)}`\n- Feasible count: `{len(feasible)}`\n\n"
    md += "## Best config\n\n```json\n" + json.dumps(best, indent=2, ensure_ascii=False) + "\n```\n"
    atomic_text(audit_dir / "C6_C1_ARM_COMPARISON.md", md)
    return payload


def build_data(args: argparse.Namespace) -> Dict[str, Any]:
    out_dir = Path(args.output_dir)
    aux_dir = out_dir / "aux"
    aux_dir.mkdir(parents=True, exist_ok=True)
    b1_train = ensure_b1_scores(args.train_pool, aux_dir / "train_fit_b1_candidate_scores.npz", args.b1_model, args.device, args.score_batch_size)
    b1_calib = {"reused": True, "score": artifact(args.b1_calib_scores)}
    b2_train = ensure_b2_pairwise_scores(split="train_fit", pool_path=args.train_pool, cache_dir=Path(args.b2_cache_dir), out_path=aux_dir / "train_fit_b2_pairwise_main_scores.npz", model_path=args.b2_model, device_name=args.device, batch_size=args.score_batch_size)
    b2_calib = {"reused": True, "score": artifact(args.b2_calib_scores)}
    train = build_feature_split(split="train_fit", cache_path=args.train_cache, temporal_path=args.train_temporal, pool_path=args.train_pool, b1_score_path=str(aux_dir / "train_fit_b1_candidate_scores.npz"), b2_score_path=str(aux_dir / "train_fit_b2_pairwise_main_scores.npz"), out_path=out_dir / "train_fit_video_features.npz")
    calib = build_feature_split(split="train_calib", cache_path=args.calib_cache, temporal_path=args.calib_temporal, pool_path=args.calib_pool, b1_score_path=args.b1_calib_scores, b2_score_path=args.b2_calib_scores, out_path=out_dir / "train_calib_video_features.npz")
    payload = {
        "status": "C6_C1_DATA_READY",
        "official_val_used": False,
        "feature_policy": "No GT IoU, hit labels, train_calib labels, official metrics, or oracle outputs are included in x. Labels are stored separately for training/eval only.",
        "modality_features_unavailable": True,
        "b1_train_scores": b1_train,
        "b1_calib_scores": b1_calib,
        "b2_train_scores": b2_train,
        "b2_calib_scores": b2_calib,
        "train": train,
        "calib": calib,
    }
    audit_dir = Path(args.audit_dir)
    atomic_json(audit_dir / "C6_C1_DATA_AUDIT.json", payload)
    atomic_text(audit_dir / "C6_C1_DATA_AUDIT.md", "# C6-C1 data audit\n\n```json\n" + json.dumps(payload, indent=2, ensure_ascii=False) + "\n```\n")
    feat_payload = {
        "status": "C6_C1_FEATURE_AUDIT_PASS",
        "official_val_used": False,
        "feature_sources": [
            "C4_final train_fit/train_calib cache rows",
            "C6-B1-lite candidate selector scores as stable span evidence",
            "C6-B2 pairwise_main scores as R1 utility evidence only",
            "C5 temporal prior and C6 boundary/proposal priors",
            "C6 cache video aggregate features",
        ],
        "forbidden_features_excluded": [
            "GT IoU in feature x",
            "hit labels in feature x",
            "train_calib label in feature x",
            "official-val metrics/predictions",
            "oracle rerank result",
        ],
        "model_capacity": {"hidden": args.hidden, "layers": args.layers, "heads": ["relevance", "moment", "iou", "risk", "gate"]},
    }
    atomic_json(audit_dir / "C6_C1_FEATURE_AUDIT.json", feat_payload)
    atomic_text(audit_dir / "C6_C1_FEATURE_AUDIT.md", "# C6-C1 feature audit\n\n```json\n" + json.dumps(feat_payload, indent=2, ensure_ascii=False) + "\n```\n")
    return payload


def write_gen_span_note(args: argparse.Namespace) -> Dict[str, Any]:
    note = """# C6-D GenSpan-lite design note

Status: `DESIGN_NOTE_ONLY`

No C6-D training, no text-to-video generation, and no official val are authorized in C6-C1.

## Proposed GenSpan-lite approximation

1. Query event decomposition: split multi-verb queries into ordered event clauses using lightweight text parsing or a future approved LLM pass.
2. Subtitle cue selection: select candidate-video subtitle windows with lexical/entity overlap to each event clause.
3. Event-order temporal prior: convert ordered clauses into monotonic soft windows over candidate spans.
4. Generated-video-free approximation: use subtitle/action cue order and existing temporal priors only; do not synthesize video.
5. Optional future text-to-video prior: allowed only in a separate protocol with cost, privacy, and hallucination controls.
6. Token selector approximation: keep candidate spans whose boundary/proposal evidence aligns with event order and subtitle cue positions.
7. Feed into C6-C: add event-order agreement, cue coverage, and order-risk features to the video reranker and proposal confidence head.
8. Engineering risk: subtitle sparsity, parser error, character aliasing, overfitting to multi-verb heuristics, and higher latency.

This note follows GenSpan's idea of motion/order priors but intentionally avoids generator dependence in the current C6-C1 stage.
"""
    path = Path(args.audit_dir) / "C6_D_GENSPAN_LITE_DESIGN_NOTE.md"
    atomic_text(path, note)
    return {"status": "C6_D_GENSPAN_LITE_DESIGN_NOTE_WRITTEN", "path": artifact(path), "official_val_used": False}


def finalize(args: argparse.Namespace) -> Dict[str, Any]:
    audit_dir = Path(args.audit_dir)
    best_path = Path(args.output_dir) / "best_config.json"
    best = json.loads(best_path.read_text(encoding="utf-8")) if best_path.exists() else None
    arm_path = audit_dir / "C6_C1_ARM_COMPARISON.json"
    arm_status = None
    if arm_path.exists():
        arm_status = json.loads(arm_path.read_text(encoding="utf-8")).get("status")
    status = "C6_C1_FREEZE_REVIEW_PASS" if best and best.get("feasible") else (arm_status or "C6_C1_NEGATIVE")
    payload = {
        "status": status,
        "official_val_used": False,
        "official_val_authorized": False,
        "post_val_adjustment": False,
        "primary_safety_anchor": "C6-B1-lite c6b1r1_0057",
        "current_promoted_report_anchor": "C6-B2 pairwise_main_0005",
        "C6_B2_role": "evidence_only_not_video_anchor",
        "best_config": best,
        "artifacts": {
            "start_state": artifact(audit_dir / "C6_C_START_STATE.json"),
            "c0_oracle": artifact(audit_dir / "C6_C0_VIDEO_RESIDUAL_ORACLE.json"),
            "feature_audit": artifact(audit_dir / "C6_C1_FEATURE_AUDIT.json"),
            "data_audit": artifact(audit_dir / "C6_C1_DATA_AUDIT.json"),
            "arm_comparison": artifact(audit_dir / "C6_C1_ARM_COMPARISON.json"),
            "training_history": artifact(Path(args.output_dir) / "arms" / "mavr_pr_mil" / "history.json"),
            "model": artifact(Path(args.output_dir) / "arms" / "mavr_pr_mil" / "model_best.pt"),
            "best_config": artifact(best_path),
            "gen_span_note": artifact(audit_dir / "C6_D_GENSPAN_LITE_DESIGN_NOTE.md"),
        },
    }
    if status == "C6_C1_FREEZE_REVIEW_PASS":
        atomic_json(audit_dir / "C6_C1_FREEZE_MANIFEST.json", payload)
        atomic_text(audit_dir / "C6_C1_FREEZE_REVIEW.md", "# C6-C1 freeze review\n\n```json\n" + json.dumps(payload, indent=2, ensure_ascii=False) + "\n```\n")
        hash_path = audit_dir / "C6_C1_FREEZE_HASHES.json"
    else:
        atomic_json(audit_dir / "C6_C1_NEGATIVE_MANIFEST.json", payload)
        atomic_text(audit_dir / "C6_C1_NEGATIVE_AUDIT.md", "# C6-C1 negative audit\n\n```json\n" + json.dumps(payload, indent=2, ensure_ascii=False) + "\n```\n")
        hash_path = audit_dir / "C6_C1_NEGATIVE_HASHES.json"
    targets = [Path(v["path"]) for v in payload["artifacts"].values() if v.get("exists")]
    atomic_json(hash_path, {str(p): artifact(p) for p in targets})
    atomic_json(audit_dir / "C6_C1_FINAL_DECISION.json", payload)
    atomic_text(audit_dir / "C6_C1_FINAL_DECISION.md", "# C6-C1 final decision\n\n```json\n" + json.dumps(payload, indent=2, ensure_ascii=False) + "\n```\n")
    return payload


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["start", "c0", "build_data", "train", "score", "search", "gen_note", "finalize", "all"], required=True)
    p.add_argument("--train_cache", default="results/rlem_c6a/cache/train_fit_c6_cache.npz")
    p.add_argument("--calib_cache", default="results/rlem_c6a/cache/train_calib_c6_cache.npz")
    p.add_argument("--train_temporal", default="results/rlem_c5_lite_prior_prep/train_fit_temporal_prior.npz")
    p.add_argument("--calib_temporal", default="results/rlem_c5_lite_prior_prep/train_calib_temporal_prior.npz")
    p.add_argument("--train_pool", default="results/rlem_c6_b1_lite_r1_safe/train_fit_candidate_pool_top_slots.npz")
    p.add_argument("--calib_pool", default="results/rlem_c6_b1_lite_r1_safe/train_calib_candidate_pool_top_slots.npz")
    p.add_argument("--gt_jsonl", default="results/rlem_c3_minimal/train_calib_gt.jsonl")
    p.add_argument("--b1_model", default="results/rlem_c6_b1_lite_r1_safe/model_best.pt")
    p.add_argument("--b1_calib_scores", default="results/rlem_c6_b1_lite_r1_safe/train_calib_candidate_scores.npz")
    p.add_argument("--b1_best_config", default="results/rlem_c6_b1_lite_r1_safe/best_config.json")
    p.add_argument("--b2_model", default="results/rlem_c6_b2/arms/pairwise_main/model_best.pt")
    p.add_argument("--b2_cache_dir", default="results/rlem_c6_b2/caches")
    p.add_argument("--b2_calib_scores", default="results/rlem_c6_b2/train_calib_scores/train_calib_pairwise_main_scores.npz")
    p.add_argument("--b2_best_config", default="results/rlem_c6_b2/train_calib_scores/pairwise_main/best_config.json")
    p.add_argument("--output_dir", default="results/rlem_c6_c")
    p.add_argument("--audit_dir", default="c6_c_audit")
    p.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    p.add_argument("--batch_size", type=int, default=4096)
    p.add_argument("--score_batch_size", type=int, default=65536)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--search_workers", type=int, default=16)
    p.add_argument("--search_profile", choices=["balanced", "full"], default="balanced")
    p.add_argument("--hidden", type=int, default=512)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.12)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--gate_mean_cap", type=float, default=0.35)
    p.add_argument("--effective_top_n", type=int, default=100)
    p.add_argument("--max_after_nms", type=int, default=100)
    p.add_argument("--nms_thd", type=float, default=0.7)
    p.add_argument("--c0_min_oracle_delta", type=float, default=0.30)
    p.add_argument("--freeze_min_r1_delta_vs_b1", type=float, default=0.20)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.audit_dir).mkdir(parents=True, exist_ok=True)
    if args.mode in ("start", "all"):
        print(json.dumps(write_start_state(args), ensure_ascii=False, indent=2))
    if args.mode in ("c0", "all"):
        print(json.dumps(run_c0(args), ensure_ascii=False, indent=2))
    if args.mode in ("build_data", "all"):
        print(json.dumps(build_data(args), ensure_ascii=False, indent=2))
    if args.mode in ("train", "all"):
        print(json.dumps(train_video_model(args), ensure_ascii=False, indent=2))
    if args.mode in ("score", "all"):
        print(json.dumps(score_video_features(args, "train_calib"), ensure_ascii=False, indent=2))
    if args.mode in ("search", "all"):
        print(json.dumps(run_search(args), ensure_ascii=False, indent=2))
    if args.mode in ("gen_note", "all"):
        print(json.dumps(write_gen_span_note(args), ensure_ascii=False, indent=2))
    if args.mode in ("finalize", "all"):
        print(json.dumps(finalize(args), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
