#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rlem_c6_b0.run_c6_b0_candidate_diagnostic import span_iou_idx  # noqa: E402
from rlem_c6_b1_lite.run_c6_b1_r1_safe_selector import evaluate_policy  # noqa: E402
from rlem_c6_c.run_c6_c_goal import (  # noqa: E402
    artifact,
    atomic_json,
    atomic_npz,
    atomic_text,
    load_baseline_configs,
    load_npz,
    make_anchor_candidates,
    score_video_features,
)
from rlem_c6_c.run_c6_c11_repair import nms_sequence  # noqa: E402


METRIC_KEYS = [
    "0.5-r1", "0.5-r5", "0.5-r10", "0.5-r100",
    "0.7-r1", "0.7-r5", "0.7-r10", "0.7-r100",
]

ACTION_NAMES = [
    "A0_noop_B2",
    "A1_MAVR_soft_rerank",
    "A2_PREM_PRVR_MIL_promote",
    "A3_topK_preserving_promote",
    "A4_raw200_video_insertion",
    "A5_BMN_proposal_confidence_boost",
]

EPS = 1e-8


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def metric_delta(new: Dict[str, float], base: Dict[str, float]) -> Dict[str, float]:
    return {k: float(new[k] - base[k]) for k in METRIC_KEYS}


def selected_metrics_from_hits(hits: Dict[str, np.ndarray]) -> Dict[str, float]:
    q_count = max(int(len(hits["hit_07_r1"])), 1)
    return {
        "0.5-r1": 100.0 * float(hits["hit_05_r1"].sum()) / q_count,
        "0.5-r5": 100.0 * float(hits["hit_05_r5"].sum()) / q_count,
        "0.5-r10": 100.0 * float(hits["hit_05_r10"].sum()) / q_count,
        "0.5-r100": 100.0 * float(hits["hit_05_r100"].sum()) / q_count,
        "0.7-r1": 100.0 * float(hits["hit_07_r1"].sum()) / q_count,
        "0.7-r5": 100.0 * float(hits["hit_07_r5"].sum()) / q_count,
        "0.7-r10": 100.0 * float(hits["hit_07_r10"].sum()) / q_count,
        "0.7-r100": 100.0 * float(hits["hit_07_r100"].sum()) / q_count,
    }


def entropy_from_scores(x: np.ndarray) -> float:
    if len(x) == 0:
        return 0.0
    y = np.asarray(x, dtype=np.float64)
    y = y - float(np.max(y))
    p = np.exp(y)
    p = p / max(float(p.sum()), EPS)
    return float(-(p * np.log(np.maximum(p, EPS))).sum() / math.log(max(len(p), 2)))


def topk_margin(scores: List[float], k: int) -> float:
    if not scores:
        return 0.0
    y = sorted(scores[:k], reverse=True)
    if len(y) < 2:
        return float(y[0])
    return float(y[0] - y[-1])


def gt_video_by_query(cache: Dict[str, np.ndarray]) -> np.ndarray:
    q_count = len(cache["desc_ids"])
    out = np.full(q_count, -1, dtype=np.int64)
    keys = cache["video_group_keys"].astype(np.int64)
    rel = cache["label_relevant"].astype(np.float32)
    for gid, ok in enumerate(rel):
        if ok > 0.5:
            q = int(keys[gid, 0])
            if 0 <= q < q_count and out[q] < 0:
                out[q] = int(keys[gid, 1])
    return out


def gt_span_by_query(cache: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    q_count = len(cache["desc_ids"])
    offsets = cache["desc_offsets"].astype(np.int64)
    gs = cache["gt_start_idx"].astype(np.int64)
    ge = cache["gt_end_idx"].astype(np.int64)
    out_s = np.zeros(q_count, dtype=np.int64)
    out_e = np.zeros(q_count, dtype=np.int64)
    for q in range(q_count):
        row = int(offsets[q])
        out_s[q] = int(gs[row])
        out_e[q] = int(ge[row])
    return out_s, out_e


def labels_for_sequence_idx(
    seq: List[Tuple[int, int, int, int, float, int]],
    gt_vid: int,
    gt_si: int,
    gt_ei: int,
) -> Dict[str, bool]:
    labs05: List[bool] = []
    labs07: List[bool] = []
    vids: List[int] = []
    invalid = 0
    for _gid, vid, si, ei, _score, _row in seq:
        invalid += int(si < 0 or ei < si)
        same = int(vid) == int(gt_vid)
        iou = span_iou_idx(int(si), int(ei), int(gt_si), int(gt_ei)) if same else 0.0
        labs05.append(bool(same and iou >= 0.5))
        labs07.append(bool(same and iou >= 0.7))
        vids.append(int(vid))
    out = {
        "hit_05_r1": bool(any(labs05[:1])),
        "hit_05_r5": bool(any(labs05[:5])),
        "hit_05_r10": bool(any(labs05[:10])),
        "hit_05_r100": bool(any(labs05[:100])),
        "hit_07_r1": bool(any(labs07[:1])),
        "hit_07_r5": bool(any(labs07[:5])),
        "hit_07_r10": bool(any(labs07[:10])),
        "hit_07_r100": bool(any(labs07[:100])),
        "gt_video_r1": int(gt_vid) in vids[:1],
        "gt_video_r5": int(gt_vid) in vids[:5],
        "gt_video_r10": int(gt_vid) in vids[:10],
        "gt_video_r100": int(gt_vid) in vids[:100],
        "top1_video_is_gt": bool(len(vids) > 0 and int(vids[0]) == int(gt_vid)),
        "invalid_span_count": bool(invalid > 0),
        "duplicate_span_count_after_nms": bool(len(seq) != len({(c[1], c[2], c[3]) for c in seq})),
    }
    return out


def make_score_arrays(score_npz: str | Path) -> Dict[str, np.ndarray]:
    z = load_npz(score_npz, allow_pickle=False)
    max_gid = int(z["group_id"].max())
    arr = {
        k: np.zeros(max_gid + 1, dtype=np.float32)
        for k in ["relevance", "moment", "iou", "risk", "gate", "base_rank"]
    }
    gids = z["group_id"].astype(np.int64)
    for k in ["relevance", "moment", "iou", "risk", "gate", "base_rank"]:
        arr[k][gids] = z[k].astype(np.float32)
    return arr


def c6c_score(arr: Dict[str, np.ndarray], gid: int, mode: str) -> float:
    if gid < 0 or gid >= len(arr["gate"]):
        return 0.0
    gate = float(arr["gate"][gid])
    rel = float(arr["relevance"][gid])
    mom = float(arr["moment"][gid])
    iou = float(arr["iou"][gid])
    risk = float(arr["risk"][gid])
    base = 1.0 / float(int(arr["base_rank"][gid]) + 1)
    if mode == "mavr":
        return float(base + 0.24 * gate * rel + 0.18 * gate * (mom + 0.35 * iou) - 0.08 * gate * risk)
    if mode == "mil":
        return float(0.35 * base + gate * (0.45 * mom + 0.30 * iou + 0.25 * rel) - 0.12 * risk)
    if mode == "proposal":
        return float(0.20 * base + 0.55 * mom + 0.45 * iou - 0.10 * risk + 0.15 * gate)
    return float(base)


def action_utility_label(delta07: float, delta05: float, r5_loss: float, r10_loss: float, hard_exit: float, entry: float, exit_: float) -> float:
    return float(2.0 * delta07 + 1.0 * delta05 + 0.5 * entry - 1.5 * r5_loss - 1.0 * r10_loss - 1.0 * hard_exit - 0.5 * exit_)


def raw_top200_candidates(
    q: int,
    cache: Dict[str, np.ndarray],
    limit: int,
) -> List[Tuple[int, int, int, int, float, int]]:
    offsets = cache["desc_offsets"].astype(np.int64)
    row_gid = cache["row_group_id"].astype(np.int64)
    video_idx = cache["video_idx"].astype(np.int64)
    start_idx = cache["start_idx"].astype(np.int64)
    end_idx = cache["end_idx"].astype(np.int64)
    s_c4 = cache["s_c4_final"].astype(np.float32)
    rows = np.arange(int(offsets[q]), int(offsets[q + 1]), dtype=np.int64)
    order = rows[np.argsort(-s_c4[rows], kind="stable")[:limit]]
    out: List[Tuple[int, int, int, int, float, int]] = []
    seen: set[int] = set()
    for row in order:
        gid = int(row_gid[row])
        if gid in seen:
            continue
        seen.add(gid)
        out.append((gid, int(video_idx[row]), int(start_idx[row]), int(end_idx[row]), float(s_c4[row]), int(row)))
    return out


def replace_or_promote(seq: List[Tuple[int, int, int, int, float, int]], cand: Tuple[int, int, int, int, float, int], to_rank: int) -> List[Tuple[int, int, int, int, float, int]]:
    out = list(seq)
    remove_idx = None
    for i, c in enumerate(out):
        if int(c[0]) == int(cand[0]) and int(c[1]) == int(cand[1]) and int(c[2]) == int(cand[2]) and int(c[3]) == int(cand[3]):
            remove_idx = i
            break
    if remove_idx is not None:
        item = out.pop(remove_idx)
    else:
        item = cand
    out.insert(max(0, min(to_rank, len(out))), item)
    return out[:100]


def same_video_span_replace(seq: List[Tuple[int, int, int, int, float, int]], cand: Tuple[int, int, int, int, float, int], rank: int) -> List[Tuple[int, int, int, int, float, int]]:
    out = list(seq)
    if 0 <= rank < len(out):
        out[rank] = cand
    return out[:100]


def pool_best_for_slot(
    q: int,
    slot_rank: int,
    pool: Dict[str, np.ndarray],
    pool_score: np.ndarray,
) -> Tuple[Tuple[int, int, int, int, float, int] | None, float]:
    pq, slots, alts = [int(x) for x in pool["shape"]]
    if q >= pq or slot_rank >= slots:
        return None, 0.0
    base = (q * slots + slot_rank) * alts
    scores = pool_score[base:base + alts].astype(np.float32)
    best = int(np.argmax(scores))
    margin = float(scores[best] - scores[0])
    idx = base + best
    cand = (
        int(pool["group_id"][idx]),
        int(pool["video_idx"][idx]),
        int(pool["start_idx"][idx]),
        int(pool["end_idx"][idx]),
        float(scores[best]),
        -1,
    )
    return cand, margin


@dataclass
class QueryContext:
    b1_seq: List[Tuple[int, int, int, int, float, int]]
    b2_seq: List[Tuple[int, int, int, int, float, int]]
    raw_groups: List[Tuple[int, int, int, int, float, int]]
    base_labels: Dict[str, bool]
    gt_vid: int
    gt_si: int
    gt_ei: int


@dataclass
class FastAnchorData:
    offsets: np.ndarray
    row_gid: np.ndarray
    video_idx: np.ndarray
    start_idx: np.ndarray
    end_idx: np.ndarray
    s_c4: np.ndarray
    pool_group_id: np.ndarray
    pool_video_idx: np.ndarray
    pool_start_idx: np.ndarray
    pool_end_idx: np.ndarray
    score3: np.ndarray
    pool_shape: Tuple[int, int, int]


def make_fast_anchor_data(cache: Dict[str, np.ndarray], pool: Dict[str, np.ndarray], pool_score: np.ndarray) -> FastAnchorData:
    q, slots, alts = [int(x) for x in pool["shape"]]
    return FastAnchorData(
        offsets=cache["desc_offsets"].astype(np.int64, copy=False),
        row_gid=cache["row_group_id"].astype(np.int64, copy=False),
        video_idx=cache["video_idx"].astype(np.int64, copy=False),
        start_idx=cache["start_idx"].astype(np.int64, copy=False),
        end_idx=cache["end_idx"].astype(np.int64, copy=False),
        s_c4=cache["s_c4_final"].astype(np.float32, copy=False),
        pool_group_id=pool["group_id"].astype(np.int64, copy=False),
        pool_video_idx=pool["video_idx"].astype(np.int64, copy=False),
        pool_start_idx=pool["start_idx"].astype(np.int64, copy=False),
        pool_end_idx=pool["end_idx"].astype(np.int64, copy=False),
        score3=pool_score.reshape(q, slots, alts).astype(np.float32, copy=False),
        pool_shape=(q, slots, alts),
    )


def make_anchor_candidates_fast(
    *,
    q: int,
    data: FastAnchorData,
    policy_cfg: Dict[str, Any],
    effective_top_n: int,
) -> List[Tuple[int, int, int, int, float, int]]:
    rows = np.arange(int(data.offsets[q]), int(data.offsets[q + 1]), dtype=np.int64)
    base_order = rows[np.argsort(-data.s_c4[rows], kind="stable")[:effective_top_n]]
    pq, slots, alts = data.pool_shape
    out: List[Tuple[int, int, int, int, float, int]] = []
    replaced = 0
    apply_slots = int(policy_cfg.get("apply_slots", 0))
    max_repl = int(policy_cfg.get("max_replacements", 0))
    for slot_rank, row in enumerate(base_order):
        gid = int(data.row_gid[row])
        vid = int(data.video_idx[row])
        si = int(data.start_idx[row])
        ei = int(data.end_idx[row])
        if q < pq and slot_rank < min(slots, apply_slots) and replaced < max_repl:
            scores = data.score3[q, slot_rank]
            best = int(np.argmax(scores))
            margin = float(scores[best] - scores[0])
            threshold = float(policy_cfg["threshold0"]) if slot_rank == 0 else float(policy_cfg["threshold_rest"])
            if best != 0 and margin > threshold:
                idx = (q * slots + slot_rank) * alts + best
                si = int(data.pool_start_idx[idx])
                ei = int(data.pool_end_idx[idx])
                replaced += 1
        out.append((gid, vid, si, ei, float(data.s_c4[row]), int(row)))
    return out


def raw_top200_candidates_fast(q: int, data: FastAnchorData, limit: int) -> List[Tuple[int, int, int, int, float, int]]:
    rows = np.arange(int(data.offsets[q]), int(data.offsets[q + 1]), dtype=np.int64)
    order = rows[np.argsort(-data.s_c4[rows], kind="stable")[:limit]]
    out: List[Tuple[int, int, int, int, float, int]] = []
    seen: set[int] = set()
    for row in order:
        gid = int(data.row_gid[row])
        if gid in seen:
            continue
        seen.add(gid)
        out.append((gid, int(data.video_idx[row]), int(data.start_idx[row]), int(data.end_idx[row]), float(data.s_c4[row]), int(row)))
    return out


def build_query_context(
    *,
    q: int,
    b1_anchor_data: FastAnchorData,
    b2_anchor_data: FastAnchorData,
    b1_cfg: Dict[str, Any],
    b2_cfg: Dict[str, Any],
    gt_vid: np.ndarray,
    gt_si: np.ndarray,
    gt_ei: np.ndarray,
    effective_top_n: int,
    nms_thd: float,
    max_after_nms: int,
) -> QueryContext:
    b1 = make_anchor_candidates_fast(q=q, data=b1_anchor_data, policy_cfg=b1_cfg, effective_top_n=effective_top_n)
    b2 = make_anchor_candidates_fast(q=q, data=b2_anchor_data, policy_cfg=b2_cfg, effective_top_n=effective_top_n)
    b1_seq = nms_sequence(b1, nms_thd, max_after_nms)
    b2_seq = nms_sequence(b2, nms_thd, max_after_nms)
    base_labels = labels_for_sequence_idx(b2_seq, int(gt_vid[q]), int(gt_si[q]), int(gt_ei[q]))
    return QueryContext(
        b1_seq=b1_seq,
        b2_seq=b2_seq,
        raw_groups=raw_top200_candidates_fast(q, b2_anchor_data, 200),
        base_labels=base_labels,
        gt_vid=int(gt_vid[q]),
        gt_si=int(gt_si[q]),
        gt_ei=int(gt_ei[q]),
    )


def candidate_features(
    *,
    qctx: QueryContext,
    score_arr: Dict[str, np.ndarray],
    action_id: int,
    selected: Tuple[int, int, int, int, float, int] | None,
    action_changed: bool,
    raw_insert: bool,
    slot_change_count: int,
    span_utility_margin: float,
) -> np.ndarray:
    b1_scores = [float(c[4]) for c in qctx.b1_seq]
    b2_scores = [float(c[4]) for c in qctx.b2_seq]
    top = qctx.b2_seq[0] if qctx.b2_seq else None
    sel = selected if selected is not None else top
    top_gid = int(top[0]) if top is not None else -1
    sel_gid = int(sel[0]) if sel is not None else -1
    c6_top = c6c_score(score_arr, top_gid, "mavr")
    c6_sel_mavr = c6c_score(score_arr, sel_gid, "mavr")
    c6_sel_mil = c6c_score(score_arr, sel_gid, "mil")
    c6_sel_prop = c6c_score(score_arr, sel_gid, "proposal")
    top10_gids = [int(c[0]) for c in qctx.b2_seq[:10]]
    top10_mavr = np.asarray([c6c_score(score_arr, gid, "mavr") for gid in top10_gids], dtype=np.float32)
    top10_mil = np.asarray([c6c_score(score_arr, gid, "mil") for gid in top10_gids], dtype=np.float32)
    gate = float(score_arr["gate"][sel_gid]) if 0 <= sel_gid < len(score_arr["gate"]) else 0.0
    rel = float(score_arr["relevance"][sel_gid]) if 0 <= sel_gid < len(score_arr["relevance"]) else 0.0
    mom = float(score_arr["moment"][sel_gid]) if 0 <= sel_gid < len(score_arr["moment"]) else 0.0
    iou = float(score_arr["iou"][sel_gid]) if 0 <= sel_gid < len(score_arr["iou"]) else 0.0
    risk = float(score_arr["risk"][sel_gid]) if 0 <= sel_gid < len(score_arr["risk"]) else 0.0
    onehot = np.zeros(len(ACTION_NAMES), dtype=np.float32)
    onehot[action_id] = 1.0
    base = np.asarray([
        float(b1_scores[0]) if b1_scores else 0.0,
        topk_margin(b1_scores, 5),
        topk_margin(b1_scores, 10),
        float(b2_scores[0]) if b2_scores else 0.0,
        topk_margin(b2_scores, 5),
        topk_margin(b2_scores, 10),
        c6_sel_mavr - c6_top,
        c6_sel_mavr,
        c6_sel_mil,
        c6_sel_prop,
        mom,
        iou,
        gate,
        risk,
        entropy_from_scores(np.asarray(b2_scores[:10], dtype=np.float32)),
        entropy_from_scores(top10_mavr),
        entropy_from_scores(top10_mil),
        float(top10_mavr[0] - top10_mavr[1]) if len(top10_mavr) > 1 else 0.0,
        float(len([s for s in top10_mavr if s >= np.quantile(top10_mavr, 0.75)])) / max(len(top10_mavr), 1),
        float(risk - rel),
        float(slot_change_count),
        float(raw_insert),
        float(span_utility_margin),
        float(c6_sel_mavr - (float(b2_scores[0]) if b2_scores else 0.0)),
        float(action_changed),
    ], dtype=np.float32)
    return np.concatenate([base, onehot], axis=0).astype(np.float32)


def make_action_sequences(
    *,
    q: int,
    qctx: QueryContext,
    score_arr: Dict[str, np.ndarray],
    pool: Dict[str, np.ndarray],
    b2_scores: np.ndarray,
) -> List[Dict[str, Any]]:
    base = list(qctx.b2_seq)
    actions: List[Dict[str, Any]] = []
    top = base[0] if base else None
    actions.append({
        "action_id": 0,
        "seq": base[:100],
        "selected": top,
        "changed": False,
        "raw_insert": False,
        "slot_change_count": 0,
        "span_utility_margin": 0.0,
    })

    def add_promote(action_id: int, window: int, mode: str, to_rank: int) -> None:
        if not base:
            actions.append({"action_id": action_id, "seq": base[:100], "selected": top, "changed": False, "raw_insert": False, "slot_change_count": 0, "span_utility_margin": 0.0})
            return
        limit = min(window, len(base))
        scores = [c6c_score(score_arr, int(base[i][0]), mode) for i in range(limit)]
        best_i = int(np.argmax(scores)) if scores else 0
        changed = bool(best_i > to_rank and scores[best_i] > scores[to_rank] - 1e-6)
        seq = replace_or_promote(base, base[best_i], to_rank) if changed else base[:100]
        actions.append({
            "action_id": action_id,
            "seq": seq,
            "selected": base[best_i] if base else None,
            "changed": changed,
            "raw_insert": False,
            "slot_change_count": int(best_i - to_rank) if changed else 0,
            "span_utility_margin": float(scores[best_i] - scores[to_rank]) if scores else 0.0,
        })

    add_promote(1, 10, "mavr", 0)
    add_promote(2, 20, "mil", 0)
    add_promote(3, 5, "mavr", 0)

    present_gids = {int(c[0]) for c in base}
    raw_candidates = [c for c in qctx.raw_groups[100:200] if int(c[0]) not in present_gids]
    if raw_candidates:
        raw_scores = [c6c_score(score_arr, int(c[0]), "mil") for c in raw_candidates]
        best_raw_i = int(np.argmax(raw_scores))
        cand = raw_candidates[best_raw_i]
        insert_rank = min(9, len(base))
        seq = list(base)
        seq.insert(insert_rank, cand)
        seq = seq[:100]
        actions.append({
            "action_id": 4,
            "seq": seq,
            "selected": cand,
            "changed": True,
            "raw_insert": True,
            "slot_change_count": 1,
            "span_utility_margin": float(raw_scores[best_raw_i] - (c6c_score(score_arr, int(base[-1][0]), "mil") if base else 0.0)),
        })
    else:
        actions.append({"action_id": 4, "seq": base[:100], "selected": top, "changed": False, "raw_insert": True, "slot_change_count": 0, "span_utility_margin": 0.0})

    # Proposal-confidence boost: replace the current top tuple by the best B2 span
    # alternative for the same slot/video, preserving video order by construction.
    cand, margin = pool_best_for_slot(q, 0, pool, b2_scores)
    if cand is not None and top is not None and int(cand[1]) == int(top[1]) and margin > 0:
        seq = same_video_span_replace(base, cand, 0)
        changed = bool((cand[2], cand[3]) != (top[2], top[3]))
        actions.append({
            "action_id": 5,
            "seq": seq,
            "selected": cand,
            "changed": changed,
            "raw_insert": False,
            "slot_change_count": 0,
            "span_utility_margin": float(margin),
        })
    else:
        actions.append({"action_id": 5, "seq": base[:100], "selected": top, "changed": False, "raw_insert": False, "slot_change_count": 0, "span_utility_margin": float(margin)})
    return actions


def build_action_dataset_split(
    *,
    split: str,
    cache_path: str,
    pool_path: str,
    b1_score_path: str,
    b2_score_path: str,
    c6c_score_path: str,
    b1_cfg: Dict[str, Any],
    b2_cfg: Dict[str, Any],
    out_path: Path,
    effective_top_n: int,
    max_after_nms: int,
    nms_thd: float,
) -> Dict[str, Any]:
    if out_path.exists():
        return {"split": split, "reused": True, "dataset": artifact(out_path)}
    cache = load_npz(cache_path, allow_pickle=True)
    pool = load_npz(pool_path, allow_pickle=False)
    b1_scores = np.load(b1_score_path, allow_pickle=False)["score"].astype(np.float32)
    b2_scores = np.load(b2_score_path, allow_pickle=False)["score"].astype(np.float32)
    b1_anchor_data = make_fast_anchor_data(cache, pool, b1_scores)
    b2_anchor_data = make_fast_anchor_data(cache, pool, b2_scores)
    score_arr = make_score_arrays(c6c_score_path)
    gt_vid = gt_video_by_query(cache)
    gt_si, gt_ei = gt_span_by_query(cache)
    q_count = len(cache["desc_ids"])

    xs: List[np.ndarray] = []
    query: List[int] = []
    action_id: List[int] = []
    changed: List[int] = []
    base_hits: Dict[str, List[int]] = {f"base_{k}": [] for k in ["hit_05_r1", "hit_05_r5", "hit_05_r10", "hit_05_r100", "hit_07_r1", "hit_07_r5", "hit_07_r10", "hit_07_r100"]}
    post_hits: Dict[str, List[int]] = {k: [] for k in ["hit_05_r1", "hit_05_r5", "hit_05_r10", "hit_05_r100", "hit_07_r1", "hit_07_r5", "hit_07_r10", "hit_07_r100"]}
    label_cols: Dict[str, List[float]] = {
        "delta_hit_07_r1": [], "delta_hit_05_r1": [], "delta_hit_07_r5": [], "delta_hit_05_r5": [],
        "delta_hit_07_r10": [], "delta_hit_05_r10": [], "delta_hit_07_r100": [], "delta_hit_05_r100": [],
        "hard_positive_exit": [], "positive_video_entry": [], "positive_video_exit": [],
        "wrong_video_to_correct_video": [], "correct_video_to_wrong_video": [],
        "topK_collapse_flag": [], "action_good": [], "action_bad": [], "action_utility": [],
        "invalid_span_count": [], "duplicate_span_count_after_nms": [], "video_slot_change_count": [],
    }
    t0 = time.time()
    for q in range(q_count):
        qctx = build_query_context(
            q=q, b1_anchor_data=b1_anchor_data, b2_anchor_data=b2_anchor_data,
            b1_cfg=b1_cfg, b2_cfg=b2_cfg, gt_vid=gt_vid, gt_si=gt_si, gt_ei=gt_ei,
            effective_top_n=effective_top_n, nms_thd=nms_thd, max_after_nms=max_after_nms,
        )
        actions = make_action_sequences(q=q, qctx=qctx, score_arr=score_arr, pool=pool, b2_scores=b2_scores)
        base = qctx.base_labels
        for rec in actions:
            aid = int(rec["action_id"])
            labs = labels_for_sequence_idx(rec["seq"], qctx.gt_vid, qctx.gt_si, qctx.gt_ei)
            feat = candidate_features(
                qctx=qctx, score_arr=score_arr, action_id=aid, selected=rec["selected"],
                action_changed=bool(rec["changed"]), raw_insert=bool(rec["raw_insert"]),
                slot_change_count=int(rec["slot_change_count"]), span_utility_margin=float(rec["span_utility_margin"]),
            )
            xs.append(feat)
            query.append(q)
            action_id.append(aid)
            changed.append(int(bool(rec["changed"])))
            for k in post_hits:
                post_hits[k].append(int(labs[k]))
            for k in base_hits:
                base_hits[k].append(int(base[k.replace("base_", "")]))

            d07r1 = float(int(labs["hit_07_r1"]) - int(base["hit_07_r1"]))
            d05r1 = float(int(labs["hit_05_r1"]) - int(base["hit_05_r1"]))
            d07r5 = float(int(labs["hit_07_r5"]) - int(base["hit_07_r5"]))
            d05r5 = float(int(labs["hit_05_r5"]) - int(base["hit_05_r5"]))
            d07r10 = float(int(labs["hit_07_r10"]) - int(base["hit_07_r10"]))
            d05r10 = float(int(labs["hit_05_r10"]) - int(base["hit_05_r10"]))
            d07r100 = float(int(labs["hit_07_r100"]) - int(base["hit_07_r100"]))
            d05r100 = float(int(labs["hit_05_r100"]) - int(base["hit_05_r100"]))
            hard_exit = float((base["hit_07_r100"] or base["hit_05_r100"]) and not (labs["hit_07_r100"] or labs["hit_05_r100"]))
            video_entry = float((not base["gt_video_r100"]) and labs["gt_video_r100"])
            video_exit = float(base["gt_video_r100"] and not labs["gt_video_r100"])
            wrong_to_correct = float((not base["top1_video_is_gt"]) and labs["top1_video_is_gt"])
            correct_to_wrong = float(base["top1_video_is_gt"] and not labs["top1_video_is_gt"])
            topk_collapse = float((base["hit_07_r5"] and not labs["hit_07_r5"]) or (base["hit_07_r10"] and not labs["hit_07_r10"]) or (base["hit_05_r5"] and not labs["hit_05_r5"]) or (base["hit_05_r10"] and not labs["hit_05_r10"]))
            action_good = float(d07r1 > 0 and d07r5 >= 0 and hard_exit == 0.0)
            action_bad = float(topk_collapse > 0 or hard_exit > 0 or correct_to_wrong > 0)
            utility = action_utility_label(d07r1, d05r1, float(d07r5 < 0 or d05r5 < 0), float(d07r10 < 0 or d05r10 < 0), hard_exit, video_entry, video_exit)
            for k, v in [
                ("delta_hit_07_r1", d07r1), ("delta_hit_05_r1", d05r1), ("delta_hit_07_r5", d07r5), ("delta_hit_05_r5", d05r5),
                ("delta_hit_07_r10", d07r10), ("delta_hit_05_r10", d05r10), ("delta_hit_07_r100", d07r100), ("delta_hit_05_r100", d05r100),
                ("hard_positive_exit", hard_exit), ("positive_video_entry", video_entry), ("positive_video_exit", video_exit),
                ("wrong_video_to_correct_video", wrong_to_correct), ("correct_video_to_wrong_video", correct_to_wrong),
                ("topK_collapse_flag", topk_collapse), ("action_good", action_good), ("action_bad", action_bad), ("action_utility", utility),
                ("invalid_span_count", float(labs["invalid_span_count"])), ("duplicate_span_count_after_nms", float(labs["duplicate_span_count_after_nms"])),
                ("video_slot_change_count", float(rec["slot_change_count"])),
            ]:
                label_cols[k].append(float(v))
        if (q + 1) % 10000 == 0:
            print(json.dumps({"stage": "build_action_dataset", "split": split, "queries_done": q + 1, "elapsed_sec": time.time() - t0}), flush=True)

    arrays: Dict[str, np.ndarray] = {
        "x": np.stack(xs).astype(np.float32),
        "query": np.asarray(query, dtype=np.int32),
        "action_id": np.asarray(action_id, dtype=np.int16),
        "action_changed": np.asarray(changed, dtype=np.int8),
        "feature_names": np.asarray([
            "b1_top1_score", "b1_top5_margin", "b1_top10_margin", "b2_top1_score", "b2_top5_margin",
            "b2_top10_margin", "c6c_proposed_video_delta", "mavr_score", "partial_relevance_mil_score",
            "proposal_confidence", "moment_evidence", "iou_evidence", "gate", "risk_score",
            "b2_evidence_entropy", "mavr_entropy", "mil_entropy", "top1_top2_video_margin",
            "positive_candidate_density_proxy", "hard_negative_similarity_proxy", "video_slot_change_count",
            "raw200_insertion_flag", "span_utility_margin", "modality_agreement_conflict", "action_changed_flag",
        ] + [f"action_onehot_{name}" for name in ACTION_NAMES], dtype=object),
        "action_names": np.asarray(ACTION_NAMES, dtype=object),
    }
    arrays.update({k: np.asarray(v, dtype=np.int8) for k, v in base_hits.items()})
    arrays.update({k: np.asarray(v, dtype=np.int8) for k, v in post_hits.items()})
    arrays.update({k: np.asarray(v, dtype=np.float32) for k, v in label_cols.items()})
    atomic_npz(out_path, **arrays)
    return {
        "split": split,
        "reused": False,
        "dataset": artifact(out_path),
        "queries": int(q_count),
        "actions": int(len(action_id)),
        "feature_dim": int(arrays["x"].shape[1]),
        "elapsed_sec": float(time.time() - t0),
    }


class ActionDataset(Dataset):
    def __init__(self, path: str | Path, mean: np.ndarray | None = None, std: np.ndarray | None = None):
        z = np.load(path, allow_pickle=True)
        self.x = z["x"].astype(np.float32)
        if mean is None:
            mean = self.x.mean(axis=0)
            std = self.x.std(axis=0)
        self.mean = mean.astype(np.float32)
        self.std = np.maximum(std.astype(np.float32), EPS)
        self.y = np.stack([
            z["action_good"].astype(np.float32),
            z["action_bad"].astype(np.float32),
            z["delta_hit_07_r1"].astype(np.float32),
            z["delta_hit_05_r1"].astype(np.float32),
            ((z["delta_hit_07_r5"] < 0) | (z["delta_hit_05_r5"] < 0)).astype(np.float32),
            ((z["delta_hit_07_r10"] < 0) | (z["delta_hit_05_r10"] < 0)).astype(np.float32),
            ((z["delta_hit_07_r100"] < 0) | (z["delta_hit_05_r100"] < 0)).astype(np.float32),
            z["hard_positive_exit"].astype(np.float32),
            z["positive_video_entry"].astype(np.float32),
            z["positive_video_exit"].astype(np.float32),
            z["action_utility"].astype(np.float32),
        ], axis=1).astype(np.float32)

    def __len__(self) -> int:
        return int(len(self.x))

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = (self.x[idx] - self.mean) / self.std
        return torch.from_numpy(x.astype(np.float32)), torch.from_numpy(self.y[idx])


class C6C2InterventionUtilityGate(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 512, layers: int = 4, dropout: float = 0.12):
        super().__init__()
        mods: List[nn.Module] = []
        d = in_dim
        for _ in range(layers):
            mods += [nn.Linear(d, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(dropout)]
            d = hidden
        self.backbone = nn.Sequential(*mods)
        self.enable_logit = nn.Linear(hidden, 1)
        self.expected_gain_07_r1 = nn.Linear(hidden, 1)
        self.expected_gain_05_r1 = nn.Linear(hidden, 1)
        self.expected_r5_risk = nn.Linear(hidden, 1)
        self.expected_r10_risk = nn.Linear(hidden, 1)
        self.expected_r100_risk = nn.Linear(hidden, 1)
        self.hard_exit_risk = nn.Linear(hidden, 1)
        self.video_entry_prob = nn.Linear(hidden, 1)
        self.video_exit_prob = nn.Linear(hidden, 1)
        self.action_utility = nn.Linear(hidden, 1)
        self.abstain_logit = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        h = self.backbone(x)
        return {
            "enable_logit": self.enable_logit(h).squeeze(-1),
            "expected_gain_07_r1": self.expected_gain_07_r1(h).squeeze(-1),
            "expected_gain_05_r1": self.expected_gain_05_r1(h).squeeze(-1),
            "expected_r5_risk": self.expected_r5_risk(h).squeeze(-1),
            "expected_r10_risk": self.expected_r10_risk(h).squeeze(-1),
            "expected_r100_risk": self.expected_r100_risk(h).squeeze(-1),
            "hard_exit_risk": self.hard_exit_risk(h).squeeze(-1),
            "video_entry_prob": self.video_entry_prob(h).squeeze(-1),
            "video_exit_prob": self.video_exit_prob(h).squeeze(-1),
            "action_utility": self.action_utility(h).squeeze(-1),
            "abstain_logit": self.abstain_logit(h).squeeze(-1),
        }


def train_gate(args: argparse.Namespace) -> Dict[str, Any]:
    model_path = Path(args.output_dir) / "c6c2_intervention_gate.pt"
    hist_path = Path(args.output_dir) / "c6c2_intervention_gate_history.json"
    if model_path.exists() and hist_path.exists() and not args.force_train:
        return {"reused": True, "model": artifact(model_path), "history": json.loads(hist_path.read_text(encoding="utf-8"))}
    ds = ActionDataset(Path(args.output_dir) / "train_fit_action_dataset.npz")
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=False)
    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    model = C6C2InterventionUtilityGate(ds.x.shape[1], hidden=args.hidden, layers=args.layers, dropout=args.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))
    y = ds.y
    good_pos_weight = torch.tensor([(1.0 - y[:, 0].mean()) / max(float(y[:, 0].mean()), EPS)], dtype=torch.float32, device=device).clamp(max=50)
    bad_pos_weight = torch.tensor([(1.0 - y[:, 1].mean()) / max(float(y[:, 1].mean()), EPS)], dtype=torch.float32, device=device).clamp(max=50)
    history: List[Dict[str, Any]] = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses: List[float] = []
        t0 = time.time()
        for xb, yb in dl:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                out = model(xb)
                enable_prob = torch.sigmoid(out["enable_logit"])
                abstain_target = torch.clamp(yb[:, 1] + (1.0 - yb[:, 0]) * 0.35, 0.0, 1.0)
                loss = (
                    1.5 * F.binary_cross_entropy_with_logits(out["enable_logit"], yb[:, 0], pos_weight=good_pos_weight)
                    + 1.0 * F.binary_cross_entropy_with_logits(out["abstain_logit"], yb[:, 1], pos_weight=bad_pos_weight)
                    + 1.0 * F.smooth_l1_loss(out["expected_gain_07_r1"], yb[:, 2])
                    + 0.7 * F.smooth_l1_loss(out["expected_gain_05_r1"], yb[:, 3])
                    + 1.0 * F.binary_cross_entropy_with_logits(out["expected_r5_risk"], yb[:, 4])
                    + 0.8 * F.binary_cross_entropy_with_logits(out["expected_r10_risk"], yb[:, 5])
                    + 0.5 * F.binary_cross_entropy_with_logits(out["expected_r100_risk"], yb[:, 6])
                    + 0.8 * F.binary_cross_entropy_with_logits(out["hard_exit_risk"], yb[:, 7])
                    + 0.5 * F.binary_cross_entropy_with_logits(out["video_entry_prob"], yb[:, 8])
                    + 0.5 * F.binary_cross_entropy_with_logits(out["video_exit_prob"], yb[:, 9])
                    + 0.8 * F.smooth_l1_loss(out["action_utility"], yb[:, 10])
                    + 0.3 * F.binary_cross_entropy_with_logits(out["abstain_logit"], abstain_target)
                    + 0.3 * (enable_prob.mean() - args.coverage_target).pow(2)
                )
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
            "coverage_target": float(args.coverage_target),
        }
        history.append(rec)
        atomic_json(hist_path, history)
        print(json.dumps({"stage": "train_c6c2_gate", **rec}), flush=True)
    torch.save({
        "model_state": model.state_dict(),
        "mean": ds.mean,
        "std": ds.std,
        "in_dim": int(ds.x.shape[1]),
        "hidden": args.hidden,
        "layers": args.layers,
        "dropout": args.dropout,
        "coverage_target": float(args.coverage_target),
        "feature_names": np.load(Path(args.output_dir) / "train_fit_action_dataset.npz", allow_pickle=True)["feature_names"],
    }, model_path)
    return {"reused": False, "model": artifact(model_path), "history": history}


@torch.no_grad()
def score_gate(args: argparse.Namespace, split: str) -> Dict[str, Any]:
    out_path = Path(args.output_dir) / f"{split}_gate_scores.npz"
    if out_path.exists() and not args.force_score:
        return {"split": split, "reused": True, "scores": artifact(out_path)}
    data_path = Path(args.output_dir) / f"{split}_action_dataset.npz"
    z = np.load(data_path, allow_pickle=True)
    ckpt = torch.load(Path(args.output_dir) / "c6c2_intervention_gate.pt", map_location="cpu")
    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    model = C6C2InterventionUtilityGate(int(ckpt["in_dim"]), hidden=int(ckpt["hidden"]), layers=int(ckpt["layers"]), dropout=float(ckpt["dropout"])).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    x = (z["x"].astype(np.float32) - ckpt["mean"]) / np.maximum(ckpt["std"], EPS)
    outs = {k: np.empty(len(x), dtype=np.float32) for k in [
        "enable_prob", "expected_gain_07_r1", "expected_gain_05_r1", "expected_r5_risk",
        "expected_r10_risk", "expected_r100_risk", "hard_exit_risk", "video_entry_prob",
        "video_exit_prob", "action_utility", "abstain_prob", "policy_utility",
    ]}
    for start in range(0, len(x), args.score_batch_size):
        xb = torch.from_numpy(x[start:start + args.score_batch_size]).to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
            out = model(xb)
        n = len(xb)
        enable = torch.sigmoid(out["enable_logit"]).float()
        r5 = torch.sigmoid(out["expected_r5_risk"]).float()
        r10 = torch.sigmoid(out["expected_r10_risk"]).float()
        r100 = torch.sigmoid(out["expected_r100_risk"]).float()
        hard = torch.sigmoid(out["hard_exit_risk"]).float()
        entry = torch.sigmoid(out["video_entry_prob"]).float()
        exit_ = torch.sigmoid(out["video_exit_prob"]).float()
        abstain = torch.sigmoid(out["abstain_logit"]).float()
        utility = (
            2.0 * out["expected_gain_07_r1"].float()
            + 1.0 * out["expected_gain_05_r1"].float()
            + 0.5 * entry
            - 1.5 * r5
            - 1.0 * r10
            - 1.0 * hard
            - 0.5 * exit_
            + 0.25 * out["action_utility"].float()
            - 0.20 * abstain
        )
        batch_vals = {
            "enable_prob": enable,
            "expected_gain_07_r1": out["expected_gain_07_r1"].float(),
            "expected_gain_05_r1": out["expected_gain_05_r1"].float(),
            "expected_r5_risk": r5,
            "expected_r10_risk": r10,
            "expected_r100_risk": r100,
            "hard_exit_risk": hard,
            "video_entry_prob": entry,
            "video_exit_prob": exit_,
            "action_utility": out["action_utility"].float(),
            "abstain_prob": abstain,
            "policy_utility": utility,
        }
        for k, v in batch_vals.items():
            outs[k][start:start + n] = v.detach().cpu().numpy().astype(np.float32)
    arrays = {
        **outs,
        "query": z["query"].astype(np.int32),
        "action_id": z["action_id"].astype(np.int16),
        "action_changed": z["action_changed"].astype(np.int8),
    }
    atomic_npz(out_path, **arrays)
    return {"split": split, "reused": False, "scores": artifact(out_path)}


def baseline_metrics_from_action_dataset(data: Dict[str, np.ndarray]) -> Dict[str, float]:
    q = data["query"].astype(np.int64)
    a0 = data["action_id"].astype(np.int64) == 0
    order = np.argsort(q[a0], kind="stable")
    hits = {
        "hit_05_r1": data["base_hit_05_r1"][a0][order],
        "hit_05_r5": data["base_hit_05_r5"][a0][order],
        "hit_05_r10": data["base_hit_05_r10"][a0][order],
        "hit_05_r100": data["base_hit_05_r100"][a0][order],
        "hit_07_r1": data["base_hit_07_r1"][a0][order],
        "hit_07_r5": data["base_hit_07_r5"][a0][order],
        "hit_07_r10": data["base_hit_07_r10"][a0][order],
        "hit_07_r100": data["base_hit_07_r100"][a0][order],
    }
    return selected_metrics_from_hits(hits)


def evaluate_anchor_index_policy(
    *,
    cache: Dict[str, np.ndarray],
    pool: Dict[str, np.ndarray],
    pool_score: np.ndarray,
    policy_cfg: Dict[str, Any],
    effective_top_n: int,
    max_after_nms: int,
    nms_thd: float,
) -> Dict[str, Any]:
    gt_vid = gt_video_by_query(cache)
    gt_si, gt_ei = gt_span_by_query(cache)
    q_count = len(cache["desc_ids"])
    anchor_data = make_fast_anchor_data(cache, pool, pool_score)
    hit_cols: Dict[str, List[int]] = {
        k: [] for k in [
            "hit_05_r1", "hit_05_r5", "hit_05_r10", "hit_05_r100",
            "hit_07_r1", "hit_07_r5", "hit_07_r10", "hit_07_r100",
        ]
    }
    video_hits = {1: 0, 5: 0, 10: 0, 100: 0}
    invalid = 0
    dup = 0
    for q in range(q_count):
        cands = make_anchor_candidates_fast(q=q, data=anchor_data, policy_cfg=policy_cfg, effective_top_n=effective_top_n)
        seq = nms_sequence(cands, nms_thd, max_after_nms)
        labs = labels_for_sequence_idx(seq, int(gt_vid[q]), int(gt_si[q]), int(gt_ei[q]))
        for k in hit_cols:
            hit_cols[k].append(int(labs[k]))
        vids = [int(c[1]) for c in seq]
        for k in video_hits:
            video_hits[k] += int(int(gt_vid[q]) in vids[:k])
        invalid += int(labs["invalid_span_count"])
        dup += int(labs["duplicate_span_count_after_nms"])
    hits = {k: np.asarray(v, dtype=np.int8) for k, v in hit_cols.items()}
    return {
        "metrics": selected_metrics_from_hits(hits),
        "video_metrics": {f"GT_video_R@{k}": 100.0 * float(v) / max(q_count, 1) for k, v in video_hits.items()},
        "movement": {
            "invalid_span_count": int(invalid),
            "duplicate_span_count_after_nms": int(dup),
        },
        "evaluation_basis": "index_iou_reconstruction",
        "official_val_used": False,
    }


def evaluate_selected_policy(
    *,
    data: Dict[str, np.ndarray],
    scores: Dict[str, np.ndarray],
    selected_index_by_query: np.ndarray,
    b1_metrics: Dict[str, float],
    b2_metrics: Dict[str, float],
    config: Dict[str, Any],
) -> Dict[str, Any]:
    idx = selected_index_by_query
    q_count = len(idx)
    hits = {k: data[k][idx].astype(np.int8) for k in [
        "hit_05_r1", "hit_05_r5", "hit_05_r10", "hit_05_r100",
        "hit_07_r1", "hit_07_r5", "hit_07_r10", "hit_07_r100",
    ]}
    metrics = selected_metrics_from_hits(hits)
    action_ids = data["action_id"][idx].astype(np.int64)
    changed = data["action_changed"][idx].astype(np.int64)
    enabled_mask = (action_ids != 0) & (changed > 0)
    base_top100 = ((data["base_hit_07_r100"][idx] > 0) | (data["base_hit_05_r100"][idx] > 0))
    new_top100 = ((data["hit_07_r100"][idx] > 0) | (data["hit_05_r100"][idx] > 0))
    action_dist = {ACTION_NAMES[i]: int(np.sum(action_ids == i)) for i in range(len(ACTION_NAMES))}
    movement = {
        "enabled_queries": int(enabled_mask.sum()),
        "enabled_query_rate": float(enabled_mask.sum() / max(q_count, 1)),
        "enabled_non_noop_queries": int(enabled_mask.sum()),
        "action_distribution": action_dist,
        "video_slot_drift_rate": float(data["video_slot_change_count"][idx].sum() / max(q_count * 100, 1)),
        "positive_video_entries": int(data["positive_video_entry"][idx].sum()),
        "positive_video_exits": int(data["positive_video_exit"][idx].sum()),
        "wrong_video_to_correct_video": int(data["wrong_video_to_correct_video"][idx].sum()),
        "correct_video_to_wrong_video": int(data["correct_video_to_wrong_video"][idx].sum()),
        "hard_positive_top100_query_entries": int((~base_top100 & new_top100).sum()),
        "hard_positive_top100_query_exits": int((base_top100 & ~new_top100).sum()),
        "hard_positive_exit_ratio": float(data["hard_positive_exit"][idx].sum() / max(float(base_top100.sum()), 1.0)),
        "invalid_span_count": int(data["invalid_span_count"][idx].sum()),
        "duplicate_span_count_after_nms": int(data["duplicate_span_count_after_nms"][idx].sum()),
        "R5_loss_queries_07": int(((data["base_hit_07_r5"][idx] > 0) & (data["hit_07_r5"][idx] == 0)).sum()),
        "R10_loss_queries_07": int(((data["base_hit_07_r10"][idx] > 0) & (data["hit_07_r10"][idx] == 0)).sum()),
        "R1_gain_queries_07": int(((data["base_hit_07_r1"][idx] == 0) & (data["hit_07_r1"][idx] > 0)).sum()),
        "R1_loss_queries_07": int(((data["base_hit_07_r1"][idx] > 0) & (data["hit_07_r1"][idx] == 0)).sum()),
    }
    video_metrics = {
        "GT_video_R@1": 100.0 * float((data["wrong_video_to_correct_video"][idx] + (data["base_hit_07_r1"][idx] > 0)).clip(max=1).sum()) / max(q_count, 1),
        "GT_video_R@100_entry_proxy": 100.0 * float(movement["hard_positive_top100_query_entries"]) / max(q_count, 1),
    }
    d1 = metric_delta(metrics, b1_metrics)
    d2 = metric_delta(metrics, b2_metrics)
    feasible = bool(
        d2["0.7-r1"] >= 0.05
        and d1["0.7-r1"] >= 0.20
        and d1["0.5-r1"] >= 0.0
        and d1["0.7-r5"] >= -0.10
        and d1["0.5-r5"] >= -0.10
        and d1["0.7-r10"] >= -0.15
        and d1["0.5-r10"] >= -0.15
        and d1["0.7-r100"] >= -0.15
        and d1["0.5-r100"] >= -0.15
        and movement["enabled_non_noop_queries"] >= max(50, int(math.ceil(0.005 * q_count)))
        and movement["positive_video_entries"] > movement["positive_video_exits"]
        and movement["hard_positive_exit_ratio"] <= 0.01
        and movement["video_slot_drift_rate"] <= 0.20
        and movement["invalid_span_count"] == 0
        and movement["duplicate_span_count_after_nms"] == 0
    )
    no_op_equiv = bool(
        movement["enabled_non_noop_queries"] == 0
        or all(abs(d2[k]) < 1e-9 for k in METRIC_KEYS)
        or action_dist["A0_noop_B2"] == q_count
    )
    score = float(
        4.0 * d2["0.7-r1"]
        + 1.5 * d2["0.5-r1"]
        + 2.0 * d1["0.7-r1"]
        - 8.0 * max(0.0, -d1["0.7-r5"] - 0.10)
        - 4.0 * max(0.0, -d1["0.7-r10"] - 0.15)
        - 4.0 * movement["hard_positive_exit_ratio"]
        + 0.002 * movement["enabled_non_noop_queries"]
    )
    return {
        "config": config,
        "metrics": metrics,
        "delta_vs_C6_B1": d1,
        "delta_vs_C6_B2": d2,
        "movement": movement,
        "video_metrics": video_metrics,
        "feasible": feasible,
        "no_op_equivalent": no_op_equiv,
        "selection_score": score,
        "official_val_used": False,
    }


def policy_search(args: argparse.Namespace) -> Dict[str, Any]:
    data = load_npz(Path(args.output_dir) / "train_calib_action_dataset.npz", allow_pickle=True)
    scores = load_npz(Path(args.output_dir) / "train_calib_gate_scores.npz", allow_pickle=False)
    cache = load_npz(args.calib_cache, allow_pickle=True)
    pool = load_npz(args.calib_pool, allow_pickle=False)
    b1_cfg, b2_cfg = load_baseline_configs(args)
    b1_scores = np.load(args.b1_calib_scores, allow_pickle=False)["score"].astype(np.float32)
    b2_scores = np.load(args.b2_calib_scores, allow_pickle=False)["score"].astype(np.float32)
    b1_eval = evaluate_anchor_index_policy(
        cache=cache, pool=pool, pool_score=b1_scores, policy_cfg=b1_cfg,
        effective_top_n=args.effective_top_n, max_after_nms=args.max_after_nms, nms_thd=args.nms_thd,
    )
    b2_metrics_reconstructed = baseline_metrics_from_action_dataset(data)
    b2_eval = evaluate_anchor_index_policy(
        cache=cache, pool=pool, pool_score=b2_scores, policy_cfg=b2_cfg,
        effective_top_n=args.effective_top_n, max_after_nms=args.max_after_nms, nms_thd=args.nms_thd,
    )
    q_count = len(cache["desc_ids"])
    query = data["query"].astype(np.int64)
    action_id = data["action_id"].astype(np.int64)
    changed = data["action_changed"].astype(np.int64)
    a0_idx = np.full(q_count, -1, dtype=np.int64)
    for i in np.where(action_id == 0)[0]:
        a0_idx[int(query[i])] = int(i)
    if np.any(a0_idx < 0):
        raise RuntimeError("A0 fallback missing for at least one query")

    allowed_sets = [
        (1, 2, 3),
        (1, 2, 3, 5),
        (1, 2, 3, 4, 5),
    ]
    enable_grid = sorted(set([0.20, 0.35, 0.50, 0.65, 0.80] + [float(np.quantile(scores["enable_prob"], q)) for q in [0.80, 0.90, 0.95, 0.98]]))
    utility_margins = [0.00, 0.02, 0.05, 0.10]
    max_rates = [0.01, 0.02, 0.05, 0.10]
    drift_rates = [0.05, 0.10, 0.20]
    results: List[Dict[str, Any]] = []
    cfg_id = 0
    for allowed in allowed_sets:
        allowed_mask = np.isin(action_id, np.asarray(allowed, dtype=np.int64)) & (changed > 0)
        for enable_th in enable_grid:
            for util_margin in utility_margins:
                structural_ok = (data["invalid_span_count"] == 0) & (data["duplicate_span_count_after_nms"] == 0)
                candidate_mask = allowed_mask & structural_ok & (scores["enable_prob"] >= enable_th) & (scores["policy_utility"] >= util_margin)
                # Pick the best predicted non-noop action per query, then rate-cap globally.
                best_idx = np.full(q_count, -1, dtype=np.int64)
                best_score = np.full(q_count, -1e9, dtype=np.float32)
                for i in np.where(candidate_mask)[0]:
                    q = int(query[i])
                    s = float(scores["policy_utility"][i])
                    if s > best_score[q]:
                        best_score[q] = s
                        best_idx[q] = int(i)
                nonzero_queries = np.where(best_idx >= 0)[0]
                for max_rate in max_rates:
                    cap = int(math.floor(q_count * max_rate))
                    if len(nonzero_queries) > cap:
                        keep_q = nonzero_queries[np.argsort(-best_score[nonzero_queries], kind="stable")[:cap]]
                    else:
                        keep_q = nonzero_queries
                    for drift_rate in drift_rates:
                        selected = a0_idx.copy()
                        if len(keep_q):
                            selected[keep_q] = best_idx[keep_q]
                        drift = float(data["video_slot_change_count"][selected].sum() / max(q_count * 100, 1))
                        if drift > drift_rate + 1e-12:
                            continue
                        cfg = {
                            "config_id": f"c6c2_policy_{cfg_id:05d}",
                            "allowed_actions": [ACTION_NAMES[i] for i in allowed],
                            "enable_threshold": float(enable_th),
                            "utility_margin": float(util_margin),
                            "max_enabled_query_rate": float(max_rate),
                            "max_video_slot_change_rate": float(drift_rate),
                            "max_actions_per_query": 1,
                            "fallback": "A0_noop_B2",
                        }
                        cfg_id += 1
                        rec = evaluate_selected_policy(
                            data=data, scores=scores, selected_index_by_query=selected,
                            b1_metrics=b1_eval["metrics"], b2_metrics=b2_metrics_reconstructed,
                            config=cfg,
                        )
                        results.append(rec)
    if not results:
        selected = a0_idx.copy()
        results.append(evaluate_selected_policy(
            data=data, scores=scores, selected_index_by_query=selected,
            b1_metrics=b1_eval["metrics"], b2_metrics=b2_metrics_reconstructed,
            config={"config_id": "c6c2_policy_a0_only", "fallback": "A0_noop_B2"},
        ))
    feasible = [r for r in results if r["feasible"] and not r["no_op_equivalent"]]
    nonnoop = [r for r in results if not r["no_op_equivalent"]]
    best_feasible = max(feasible, key=lambda r: (r["selection_score"], r["delta_vs_C6_B2"]["0.7-r1"])) if feasible else None
    best_nonnoop = max(nonnoop, key=lambda r: (r["selection_score"], r["delta_vs_C6_B2"]["0.7-r1"])) if nonnoop else None
    best_overall = max(results, key=lambda r: (r["feasible"], not r["no_op_equivalent"], r["selection_score"], r["delta_vs_C6_B2"]["0.7-r1"]))
    oracle = oracle_upper_bound_from_labels(data, b1_eval["metrics"], b2_metrics_reconstructed)
    return {
        "baselines": {
            "C6_B1_lite_eval": b1_eval,
            "C6_B2_eval_reconstructed_from_action_A0": {"metrics": b2_metrics_reconstructed},
            "C6_B2_eval_policy_function": b2_eval,
        },
        "result_count": len(results),
        "feasible_count": len(feasible),
        "nonnoop_count": len(nonnoop),
        "best_feasible": best_feasible,
        "best_nonnoop": best_nonnoop,
        "best_overall": best_overall,
        "top_results": sorted(results, key=lambda r: (r["feasible"], not r["no_op_equivalent"], r["selection_score"]), reverse=True)[:50],
        "oracle_upper_bound_train_calib_diagnostic_only": oracle,
        "official_val_used": False,
    }


def oracle_upper_bound_from_labels(data: Dict[str, np.ndarray], b1_metrics: Dict[str, float], b2_metrics: Dict[str, float]) -> Dict[str, Any]:
    q_count = int(data["query"].max()) + 1
    query = data["query"].astype(np.int64)
    action_id = data["action_id"].astype(np.int64)
    utility = data["action_utility"].astype(np.float32)
    changed = data["action_changed"].astype(np.int64)
    selected = np.full(q_count, -1, dtype=np.int64)
    best = np.full(q_count, -1e9, dtype=np.float32)
    for i in range(len(query)):
        q = int(query[i])
        score = float(utility[i])
        if action_id[i] == 0:
            score = max(score, -1e-6)
        if action_id[i] != 0 and changed[i] == 0:
            score -= 1.0
        if score > best[q]:
            best[q] = score
            selected[q] = i
    return evaluate_selected_policy(
        data=data, scores={}, selected_index_by_query=selected,
        b1_metrics=b1_metrics, b2_metrics=b2_metrics,
        config={"diagnostic_only": True, "uses_labels": True, "promotion_allowed": False},
    )


def action_dataset_audit(args: argparse.Namespace, build_info: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {"official_val_used": False, "splits": {}, "build_info": build_info}
    for split in ["train_fit", "train_calib"]:
        z = load_npz(Path(args.output_dir) / f"{split}_action_dataset.npz", allow_pickle=True)
        aid = z["action_id"]
        changed = z["action_changed"]
        out["splits"][split] = {
            "queries": int(z["query"].max()) + 1,
            "samples": int(len(aid)),
            "feature_dim": int(z["x"].shape[1]),
            "number_of_actions": len(ACTION_NAMES),
            "action_names": ACTION_NAMES,
            "positive_action_rate": float(z["action_good"].mean()),
            "bad_action_rate": float(z["action_bad"].mean()),
            "no_op_rate": float((changed == 0).mean()),
            "actions_with_R1_gain_but_R5_loss": int(((z["delta_hit_07_r1"] > 0) & (z["delta_hit_07_r5"] < 0)).sum()),
            "actions_with_video_correction": int(z["wrong_video_to_correct_video"].sum()),
            "actions_with_video_regression": int(z["correct_video_to_wrong_video"].sum()),
            "changed_nonnoop_actions": int(((aid != 0) & (changed > 0)).sum()),
            "action_distribution": {ACTION_NAMES[i]: int((aid == i).sum()) for i in range(len(ACTION_NAMES))},
            "feature_leakage_guard": {
                "gt_iou_as_feature": False,
                "hit_label_as_feature": False,
                "train_calib_label_as_feature": False,
                "official_val_prediction_as_feature": False,
                "oracle_decision_as_feature": False,
            },
        }
    atomic_json(Path(args.audit_dir) / "C6_C2_ACTION_DATASET_AUDIT.json", out)
    md = "# C6-C2 action dataset audit\n\n"
    md += "- Scope: `train_fit/train_calib only`\n- Official val used: `false`\n\n"
    for split, rec in out["splits"].items():
        md += f"## {split}\n\n```json\n{json.dumps(rec, indent=2, ensure_ascii=False)}\n```\n"
    atomic_text(Path(args.audit_dir) / "C6_C2_ACTION_DATASET_AUDIT.md", md)
    return out


def write_start_state(args: argparse.Namespace) -> Dict[str, Any]:
    state = {
        "stage": "C6-C2 query-level intervention utility learning",
        "date": "2026-06-26",
        "stable_safety_anchor_system": "C6-B1-lite c6b1r1_0057",
        "current_promoted_system_from_final_handoff": "C6-B2 pairwise_main_0005",
        "r1_extension_system": "C6-B2 pairwise_main_0005",
        "previous_c6c_status": "C6_C11_SAFE_B2_EQUIVALENT_NO_PROMOTION",
        "instruction_alignment": {
            "overall": "aligned",
            "correction": "Treat C6-B1-lite as safety anchor, while C6-B2 remains the current promoted/R1-extension comparison anchor.",
            "reason": "C6-C1.1 was safe only because enabled_queries=0, so C6-C2 must learn query/action utility and enforce non-zero intervention before any promotion."
        },
        "official_val_used_for_c6c2": False,
        "official_val_authorized": False,
        "post_val_adjustment": False,
        "second_official_val": False,
        "C6_B1_lite_modified": False,
        "C6_B2_modified": False,
        "C4_final_modified": False,
        "NMS_modified": False,
        "evaluator_modified": False,
        "full_backbone_finetuning": False,
    }
    atomic_json(Path(args.audit_dir) / "C6_C2_START_STATE.json", state)
    atomic_text(Path(args.audit_dir) / "C6_C2_START_STATE.md", "# C6-C2 start state\n\n```json\n" + json.dumps(state, indent=2, ensure_ascii=False) + "\n```\n")
    return state


def write_model_audit(args: argparse.Namespace, train_info: Dict[str, Any], score_info: Dict[str, Any]) -> Dict[str, Any]:
    payload = {
        "status": "C6_C2_INTERVENTION_MODEL_TRAINED",
        "official_val_used": False,
        "model_class": "C6C2InterventionUtilityGate",
        "architecture": {
            "hidden": args.hidden,
            "layers": args.layers,
            "dropout": args.dropout,
            "heads": [
                "enable_logit", "expected_gain_07_r1", "expected_gain_05_r1", "expected_r5_risk",
                "expected_r10_risk", "expected_r100_risk", "hard_exit_risk", "video_entry_prob",
                "video_exit_prob", "action_utility", "abstain_logit",
            ],
        },
        "loss": {
            "action_good_bce_weight": 1.5,
            "action_bad_bce_weight": 1.0,
            "gain_07_regression_weight": 1.0,
            "gain_05_regression_weight": 0.7,
            "topk_risk_bce_weight": 1.0,
            "hard_exit_bce_weight": 0.8,
            "video_entry_exit_bce_weight": 0.5,
            "abstention_calibration_weight": 0.3,
            "coverage_regularization_weight": 0.3,
            "coverage_target": args.coverage_target,
        },
        "training": train_info,
        "scoring": score_info,
    }
    atomic_json(Path(args.audit_dir) / "C6_C2_INTERVENTION_MODEL_AUDIT.json", payload)
    atomic_text(Path(args.audit_dir) / "C6_C2_INTERVENTION_MODEL_AUDIT.md", "# C6-C2 intervention model audit\n\n```json\n" + json.dumps(payload, indent=2, ensure_ascii=False) + "\n```\n")
    return payload


def write_policy_and_final(args: argparse.Namespace, search_result: Dict[str, Any]) -> Dict[str, Any]:
    audit_dir = Path(args.audit_dir)
    atomic_json(audit_dir / "C6_C2_POLICY_SEARCH.json", search_result)
    md = "# C6-C2 policy search\n\n"
    md += "- Scope: `train_calib only`\n- Official val used: `false`\n- A0 fallback: `always available`\n\n"
    md += "## Summary\n\n```json\n" + json.dumps({
        "result_count": search_result["result_count"],
        "feasible_count": search_result["feasible_count"],
        "nonnoop_count": search_result["nonnoop_count"],
        "best_feasible": search_result["best_feasible"],
        "best_nonnoop": search_result["best_nonnoop"],
        "oracle_upper_bound_train_calib_diagnostic_only": search_result["oracle_upper_bound_train_calib_diagnostic_only"],
    }, indent=2, ensure_ascii=False) + "\n```\n"
    atomic_text(audit_dir / "C6_C2_POLICY_SEARCH.md", md)

    best = search_result["best_feasible"] or search_result["best_nonnoop"] or search_result["best_overall"]
    if best["feasible"] and not best["no_op_equivalent"]:
        status = "C6_C2_FREEZE_REVIEW_PASS"
        promoted = True
    elif best["no_op_equivalent"]:
        status = "C6_C2_SAFE_NOOP_B2_EQUIVALENT"
        promoted = False
    elif best["delta_vs_C6_B2"]["0.7-r1"] > 0 and (
        best["delta_vs_C6_B1"]["0.7-r5"] < -0.10
        or best["delta_vs_C6_B1"]["0.7-r10"] < -0.15
        or best["delta_vs_C6_B2"]["0.7-r5"] < 0.0
        or best["delta_vs_C6_B2"]["0.7-r10"] < 0.0
        or best["movement"]["invalid_span_count"] > 0
        or best["movement"]["duplicate_span_count_after_nms"] > 0
        or best["movement"]["positive_video_entries"] <= best["movement"]["positive_video_exits"]
    ):
        status = "C6_C2_R1_TRADEOFF"
        promoted = False
    elif best["movement"]["positive_video_entries"] > best["movement"]["positive_video_exits"] and best["delta_vs_C6_B2"]["0.7-r1"] <= 0:
        status = "C6_C2_VIDEO_ONLY_POSITIVE"
        promoted = False
    else:
        status = "C6_C2_NEGATIVE"
        promoted = False
    final = {
        "status": status,
        "promoted": promoted,
        "official_val_used": False,
        "official_val_authorized": False,
        "post_val_adjustment": False,
        "second_official_val": False,
        "current_promoted_system": "C6-B2 pairwise_main_0005 remains unchanged unless human authorizes a future official-val one-shot.",
        "stable_safety_anchor_system": "C6-B1-lite c6b1r1_0057",
        "selected_policy": best,
        "hard_noop_rule": {
            "enabled_queries_zero": best["movement"]["enabled_queries"] == 0,
            "enabled_non_noop_queries_zero": best["movement"]["enabled_non_noop_queries"] == 0,
            "delta_vs_C6_B2_zero_all_metrics": all(abs(best["delta_vs_C6_B2"][k]) < 1e-9 for k in METRIC_KEYS),
            "all_selected_actions_A0": best["movement"]["action_distribution"]["A0_noop_B2"] == int(sum(best["movement"]["action_distribution"].values())),
        },
        "decision": "Stop after C6-C2; do not enter official val, C6-C3, C6-D, or post-val adjustment automatically.",
    }
    atomic_json(audit_dir / "C6_C2_FINAL_DECISION.json", final)
    atomic_text(audit_dir / "C6_C2_FINAL_DECISION.md", "# C6-C2 final decision\n\n```json\n" + json.dumps(final, indent=2, ensure_ascii=False) + "\n```\n")
    if promoted:
        freeze_manifest = {
            "status": status,
            "selected_policy": best["config"],
            "artifacts": {
                "model": artifact(Path(args.output_dir) / "c6c2_intervention_gate.pt"),
                "calib_scores": artifact(Path(args.output_dir) / "train_calib_gate_scores.npz"),
                "policy_search": artifact(audit_dir / "C6_C2_POLICY_SEARCH.json"),
                "final_decision": artifact(audit_dir / "C6_C2_FINAL_DECISION.json"),
            },
            "official_val_used": False,
        }
        atomic_json(audit_dir / "C6_C2_FREEZE_MANIFEST.json", freeze_manifest)
        hashes = {k: artifact(v["path"]) for k, v in freeze_manifest["artifacts"].items()}
        atomic_json(audit_dir / "C6_C2_FREEZE_HASHES.json", hashes)
        atomic_text(audit_dir / "C6_C2_FREEZE_REVIEW.md", "# C6-C2 freeze review\n\n```json\n" + json.dumps(freeze_manifest, indent=2, ensure_ascii=False) + "\n```\n")
    else:
        negative_manifest = {
            "status": status,
            "selected_policy": best["config"],
            "best_metrics": best["metrics"],
            "delta_vs_C6_B1": best["delta_vs_C6_B1"],
            "delta_vs_C6_B2": best["delta_vs_C6_B2"],
            "movement": best["movement"],
            "official_val_used": False,
        }
        atomic_json(audit_dir / "C6_C2_NEGATIVE_MANIFEST.json", negative_manifest)
        hashes = {
            "action_dataset_audit": artifact(audit_dir / "C6_C2_ACTION_DATASET_AUDIT.json"),
            "model_audit": artifact(audit_dir / "C6_C2_INTERVENTION_MODEL_AUDIT.json"),
            "policy_search": artifact(audit_dir / "C6_C2_POLICY_SEARCH.json"),
            "final_decision": artifact(audit_dir / "C6_C2_FINAL_DECISION.json"),
            "model": artifact(Path(args.output_dir) / "c6c2_intervention_gate.pt"),
        }
        atomic_json(audit_dir / "C6_C2_NEGATIVE_HASHES.json", hashes)
        atomic_text(audit_dir / "C6_C2_NEGATIVE_AUDIT.md", "# C6-C2 negative/no-promotion audit\n\n```json\n" + json.dumps(negative_manifest, indent=2, ensure_ascii=False) + "\n```\n")
    return final


def ensure_c6c_video_scores(args: argparse.Namespace) -> Dict[str, Any]:
    ns = SimpleNamespace(
        output_dir=args.c6c_output_dir,
        device=args.device,
        score_batch_size=args.score_batch_size,
    )
    train_fit = score_video_features(ns, "train_fit")
    train_calib = score_video_features(ns, "train_calib")
    return {"train_fit": train_fit, "train_calib": train_calib}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--train_fit_cache", default="results/rlem_c6a/cache/train_fit_c6_cache.npz")
    p.add_argument("--calib_cache", default="results/rlem_c6a/cache/train_calib_c6_cache.npz")
    p.add_argument("--train_fit_pool", default="results/rlem_c6_b1_lite_r1_safe/train_fit_candidate_pool_top_slots.npz")
    p.add_argument("--calib_pool", default="results/rlem_c6_b1_lite_r1_safe/train_calib_candidate_pool_top_slots.npz")
    p.add_argument("--train_fit_b1_scores", default="results/rlem_c6_c/aux/train_fit_b1_candidate_scores.npz")
    p.add_argument("--train_fit_b2_scores", default="results/rlem_c6_c/aux/train_fit_b2_pairwise_main_scores.npz")
    p.add_argument("--b1_calib_scores", default="results/rlem_c6_b1_lite_r1_safe/train_calib_candidate_scores.npz")
    p.add_argument("--b2_calib_scores", default="results/rlem_c6_b2/train_calib_scores/train_calib_pairwise_main_scores.npz")
    p.add_argument("--b1_best_config", default="results/rlem_c6_b1_lite_r1_safe/best_config.json")
    p.add_argument("--b2_best_config", default="results/rlem_c6_b2/train_calib_scores/pairwise_main/best_config.json")
    p.add_argument("--c6c_output_dir", default="results/rlem_c6_c")
    p.add_argument("--output_dir", default="results/rlem_c6_c2")
    p.add_argument("--audit_dir", default="c6_c2_audit")
    p.add_argument("--effective_top_n", type=int, default=100)
    p.add_argument("--max_after_nms", type=int, default=100)
    p.add_argument("--nms_thd", type=float, default=0.7)
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch_size", type=int, default=8192)
    p.add_argument("--score_batch_size", type=int, default=65536)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--hidden", type=int, default=512)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.12)
    p.add_argument("--lr", type=float, default=1.5e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--coverage_target", type=float, default=0.02)
    p.add_argument("--force_train", action="store_true")
    p.add_argument("--force_score", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.audit_dir).mkdir(parents=True, exist_ok=True)
    write_start_state(args)
    b1_cfg, b2_cfg = load_baseline_configs(args)
    c6c_scores = ensure_c6c_video_scores(args)
    build_info = {
        "c6c_video_scores": c6c_scores,
        "train_fit": build_action_dataset_split(
            split="train_fit", cache_path=args.train_fit_cache, pool_path=args.train_fit_pool,
            b1_score_path=args.train_fit_b1_scores, b2_score_path=args.train_fit_b2_scores,
            c6c_score_path=Path(args.c6c_output_dir) / "train_fit_video_scores.npz",
            b1_cfg=b1_cfg, b2_cfg=b2_cfg, out_path=Path(args.output_dir) / "train_fit_action_dataset.npz",
            effective_top_n=args.effective_top_n, max_after_nms=args.max_after_nms, nms_thd=args.nms_thd,
        ),
        "train_calib": build_action_dataset_split(
            split="train_calib", cache_path=args.calib_cache, pool_path=args.calib_pool,
            b1_score_path=args.b1_calib_scores, b2_score_path=args.b2_calib_scores,
            c6c_score_path=Path(args.c6c_output_dir) / "train_calib_video_scores.npz",
            b1_cfg=b1_cfg, b2_cfg=b2_cfg, out_path=Path(args.output_dir) / "train_calib_action_dataset.npz",
            effective_top_n=args.effective_top_n, max_after_nms=args.max_after_nms, nms_thd=args.nms_thd,
        ),
    }
    action_dataset_audit(args, build_info)
    train_info = train_gate(args)
    score_info = {
        "train_fit": score_gate(args, "train_fit"),
        "train_calib": score_gate(args, "train_calib"),
    }
    write_model_audit(args, train_info, score_info)
    search_result = policy_search(args)
    final = write_policy_and_final(args, search_result)
    print(json.dumps({
        "status": final["status"],
        "promoted": final["promoted"],
        "official_val_used": False,
        "selected_policy": final["selected_policy"]["config"],
        "metrics": final["selected_policy"]["metrics"],
        "delta_vs_C6_B1": final["selected_policy"]["delta_vs_C6_B1"],
        "delta_vs_C6_B2": final["selected_policy"]["delta_vs_C6_B2"],
        "movement": final["selected_policy"]["movement"],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
