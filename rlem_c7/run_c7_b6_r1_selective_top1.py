#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rlem_c7.run_c7_b0_b1 import (  # noqa: E402
    atomic_json,
    atomic_text,
    labels_for_sequence,
    metric_delta,
    nms_sequence,
    selected_metrics,
)
from rlem_c7.run_c7_b5_sn_scoring import (  # noqa: E402
    METRIC_KEYS,
    build_queries,
    cand_id,
    md,
    sha256_file,
    split_indices,
)


Candidate = Tuple[int, int, int, int, float, int]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--cache", default="results/rlem_c6a/cache/train_calib_c6_cache.npz")
    p.add_argument("--anchor_states", default="results/rlem_c7_b2/train_calib_c7_b1_anchor_states.json")
    p.add_argument("--video_scores", default="results/rlem_c7_b2/train_calib_video_scores.npz")
    p.add_argument("--b2_eval", default="c7_audit/C7_B2_TRAIN_CALIB_EVAL.json")
    p.add_argument("--fixed_pool_logits_npz", default="results/rlem_c7_b5_1/train_calib_fixed_pool_raw_logits.npz")
    p.add_argument("--proposal_scores", default="results/rlem_c7_b2/train_calib_c7_b1_proposal_scores.npz")
    p.add_argument("--audit_dir", default="c7_audit")
    p.add_argument("--model_dir", default="c7_models")
    p.add_argument("--effective_top_n", type=int, default=100)
    p.add_argument("--max_after_nms", type=int, default=100)
    p.add_argument("--nms_thd", type=float, default=0.7)
    p.add_argument("--seed", type=int, default=20260626)
    p.add_argument("--epochs", type=int, default=180)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def sha256_obj(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")).hexdigest()


def load_features(args: argparse.Namespace, q_count: int) -> Dict[str, np.ndarray]:
    logits = np.load(args.fixed_pool_logits_npz, allow_pickle=False)
    prop = np.load(args.proposal_scores, allow_pickle=False)
    n = q_count * 100
    for name, arr in [
        ("query_index", logits["query_index"]),
        ("proposal_query", prop["query"]),
        ("proposal_rank", prop["rank"]),
    ]:
        if len(arr) != n:
            raise ValueError(f"{name} has {len(arr)} rows, expected {n}")
    return {
        "b_start": logits["b_start_logit"].astype(np.float32).reshape(q_count, 100),
        "e_end": logits["e_end_logit"].astype(np.float32).reshape(q_count, 100),
        "p_b": logits["p_b_i"].astype(np.float32).reshape(q_count, 100),
        "p_e": logits["p_e_j"].astype(np.float32).reshape(q_count, 100),
        "r1": logits["r1"].astype(np.float32).reshape(q_count, 100),
        "proposal_confidence": prop["proposal_confidence"].astype(np.float32).reshape(q_count, 100),
        "span_quality": prop["span_quality"].astype(np.float32).reshape(q_count, 100),
    }


def rank_array(scores: np.ndarray) -> np.ndarray:
    order = np.argsort(-scores, kind="stable")
    ranks = np.empty_like(order, dtype=np.int32)
    ranks[order] = np.arange(1, len(scores) + 1, dtype=np.int32)
    return ranks


def ordered_candidates(qd: Dict[str, Any], order: Sequence[int], scores: np.ndarray) -> List[Candidate]:
    out: List[Candidate] = []
    n = len(order)
    for pos, idx in enumerate(order):
        c = qd["anchor"][int(idx)]
        # Scores are monotonic by explicit order; NMS preserves input order.
        out.append((c[0], c[1], c[2], c[3], float(n - pos) + 1e-6 * float(scores[int(idx)]), c[5]))
    return out


def move_to_front(order: np.ndarray, idx: int) -> np.ndarray:
    idx = int(idx)
    if int(order[0]) == idx:
        return order.copy()
    rest = order[order != idx]
    return np.concatenate([np.asarray([idx], dtype=order.dtype), rest])


def video_rank_in_order(qd: Dict[str, Any], order: np.ndarray, idx: int) -> int:
    target_video = int(qd["anchor"][int(idx)][1])
    seen: List[int] = []
    for oid in order:
        v = int(qd["anchor"][int(oid)][1])
        if v not in seen:
            seen.append(v)
        if v == target_video:
            return len(seen)
    return 999


def overlap_idx(a: Candidate, b: Candidate) -> float:
    if int(a[1]) != int(b[1]):
        return 0.0
    inter = max(0, min(int(a[3]), int(b[3])) - max(int(a[2]), int(b[2])) + 1)
    union = max(int(a[3]), int(b[3])) - min(int(a[2]), int(b[2])) + 1
    return float(inter / max(union, 1))


def correct(c: Candidate, qd: Dict[str, Any], thr: float) -> bool:
    labels05, labels07, *_ = labels_for_sequence([c], int(qd["gt_vid"]), int(qd["gt_s"]), int(qd["gt_e"]))
    return bool(labels07[0] if thr >= 0.7 else labels05[0])


def build_query_records(queries: List[Dict[str, Any]], feats: Dict[str, np.ndarray]) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for q, qd in enumerate(queries):
        a4 = (qd["base"] + qd["residual"]).astype(np.float32)
        residual = qd["residual"].astype(np.float32)
        sn = (np.log(np.maximum(feats["r1"][q], 1e-12)) + feats["b_start"][q] + feats["e_end"][q]).astype(np.float32)
        a4_order = np.argsort(-a4, kind="stable")
        sn_order = np.argsort(-sn, kind="stable")
        residual_order = np.argsort(-residual, kind="stable")
        a4_rank = rank_array(a4)
        sn_rank = rank_array(sn)
        residual_rank = rank_array(residual)
        rrf = 1.0 / (60.0 + a4_rank.astype(np.float32)) + 1.0 / (60.0 + sn_rank.astype(np.float32)) + 1.0 / (60.0 + residual_rank.astype(np.float32))
        rrf_order = np.argsort(-rrf, kind="stable")
        a4_top = int(a4_order[0])
        sn_top = int(sn_order[0])
        rrf_top = int(rrf_order[0])
        records.append({
            "q": q,
            "desc_id": str(qd["desc_id"]),
            "a4_scores": a4,
            "sn_scores": sn,
            "residual": residual,
            "rrf_scores": rrf,
            "a4_order": a4_order,
            "sn_order": sn_order,
            "rrf_order": rrf_order,
            "a4_rank": a4_rank,
            "sn_rank": sn_rank,
            "residual_rank": residual_rank,
            "a4_top": a4_top,
            "sn_top": sn_top,
            "rrf_top": rrf_top,
            "rank_A4_SN": int(a4_rank[sn_top]),
            "rank_A4_RRF": int(a4_rank[rrf_top]),
            "rank_SN_A4": int(sn_rank[a4_top]),
            "video_rank_SN": int(video_rank_in_order(qd, a4_order, sn_top)),
            "video_rank_A4": int(video_rank_in_order(qd, a4_order, a4_top)),
            "a4_margin": float(a4[a4_order[0]] - a4[a4_order[1]]),
            "sn_margin": float(sn[sn_order[0]] - sn[sn_order[1]]),
            "sn_delta_top": float(sn[sn_top] - sn[a4_top]),
            "a4_delta_top": float(a4[a4_top] - a4[sn_top]),
            "b_end_margin": float((feats["b_start"][q, sn_top] + feats["e_end"][q, sn_top]) - (feats["b_start"][q, a4_top] + feats["e_end"][q, a4_top])),
            "proposal_confidence": float(feats["proposal_confidence"][q, sn_top]),
            "span_quality": float(feats["span_quality"][q, sn_top]),
            "duration": int(qd["anchor"][sn_top][3] - qd["anchor"][sn_top][2] + 1),
            "overlap_SN_A4": overlap_idx(qd["anchor"][sn_top], qd["anchor"][a4_top]),
            "label_better_07": float(correct(qd["anchor"][sn_top], qd, 0.7) and not correct(qd["anchor"][a4_top], qd, 0.7)),
            "label_better_05": float(correct(qd["anchor"][sn_top], qd, 0.5) and not correct(qd["anchor"][a4_top], qd, 0.5)),
            "label_harmful_07": float(correct(qd["anchor"][a4_top], qd, 0.7) and not correct(qd["anchor"][sn_top], qd, 0.7)),
        })
    return records


def sequence_for_record(qd: Dict[str, Any], rec: Dict[str, Any], mode: str, guard: int | None = None, candidate: str = "SN") -> Tuple[List[Candidate], bool, int]:
    base_order = rec["a4_order"]
    scores = rec["a4_scores"]
    if mode == "A4":
        order = base_order
        changed = False
        replacement_rank = 1
    else:
        idx = int(rec["sn_top"] if candidate == "SN" else rec["rrf_top"])
        replacement_rank = int(rec["a4_rank"][idx])
        changed = guard is None or replacement_rank <= int(guard)
        order = move_to_front(base_order, idx) if changed else base_order
    return ordered_candidates(qd, order, scores), bool(changed), int(replacement_rank)


def eval_orders(
    name: str,
    queries: List[Dict[str, Any]],
    records: List[Dict[str, Any]],
    idx: Sequence[int],
    order_fn,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    labels05: List[List[bool]] = []
    labels07: List[List[bool]] = []
    base05: List[List[bool]] = []
    base07: List[List[bool]] = []
    top1chg = top5chg = top10chg = 0
    exits = entries = invalid = dup = 0
    changed_count = 0
    gains_by_rank = {"<=5": 0, "<=10": 0, ">10": 0}
    loss_by_rank = {"<=5": 0, "<=10": 0, ">10": 0}
    for q in idx:
        qd, rec = queries[q], records[q]
        base_seq_raw, _base_changed, _br = sequence_for_record(qd, rec, "A4")
        cur_seq_raw, changed, repl_rank = order_fn(qd, rec)
        base_ids = [cand_id(c) for c in base_seq_raw]
        cur_ids = [cand_id(c) for c in cur_seq_raw]
        top1chg += int(base_ids[:1] != cur_ids[:1])
        top5chg += int(set(base_ids[:5]) != set(cur_ids[:5]))
        top10chg += int(set(base_ids[:10]) != set(cur_ids[:10]))
        changed_count += int(changed)
        base_seq = nms_sequence(base_seq_raw[:args.effective_top_n], args.nms_thd, args.max_after_nms)
        cur_seq = nms_sequence(cur_seq_raw[:args.effective_top_n], args.nms_thd, args.max_after_nms)
        l05, l07, pos, inv, du = labels_for_sequence(cur_seq, int(qd["gt_vid"]), int(qd["gt_s"]), int(qd["gt_e"]))
        a05, a07, apos, _ai, _ad = labels_for_sequence(base_seq, int(qd["gt_vid"]), int(qd["gt_s"]), int(qd["gt_e"]))
        labels05.append(l05); labels07.append(l07); base05.append(a05); base07.append(a07)
        exits += int(apos and not pos)
        entries += int((not apos) and pos)
        invalid += inv
        dup += du
        gain07 = (not any(a07[:1])) and any(l07[:1])
        loss07 = any(a07[:1]) and (not any(l07[:1]))
        bucket = "<=5" if repl_rank <= 5 else ("<=10" if repl_rank <= 10 else ">10")
        if gain07:
            gains_by_rank[bucket] += 1
        if loss07:
            loss_by_rank[bucket] += 1
    n = max(len(idx), 1)
    metrics = selected_metrics(labels05, labels07)
    base_metrics = selected_metrics(base05, base07)
    return {
        "name": name,
        "query_count": int(len(idx)),
        "metrics": metrics,
        "delta_vs_C7_B2_1_A4": metric_delta(metrics, base_metrics),
        "changed_query_count": int(changed_count),
        "changed_query_rate": float(changed_count / n),
        "top1_changed_rate": float(top1chg / n),
        "top5_set_changed_rate": float(top5chg / n),
        "top10_set_changed_rate": float(top10chg / n),
        "gain_loss_vs_baseline": {
            f"{thr}-r{r}": {
                "gain_queries": int(sum((not any(b[:r])) and any(c[:r]) for b, c in zip(base, cur))),
                "loss_queries": int(sum(any(b[:r]) and (not any(c[:r])) for b, c in zip(base, cur))),
            }
            for thr, cur, base in [("0.5", labels05, base05), ("0.7", labels07, base07)] for r in [1, 5, 10]
        },
        "r1_07_gain_by_A4_rank_of_replacement": gains_by_rank,
        "r1_07_loss_by_A4_rank_of_replacement": loss_by_rank,
        "movement": {
            "hard_positive_top100_query_exits": int(exits),
            "hard_positive_top100_query_entries": int(entries),
            "hard_positive_exit_ratio": float(exits / n),
            "invalid_span_count": int(invalid),
            "duplicate_span_count_after_nms": int(dup),
            "fixed_pool_invariant": True,
            "R100_delta": {
                "0.5-r100": float(metrics["0.5-r100"] - base_metrics["0.5-r100"]),
                "0.7-r100": float(metrics["0.7-r100"] - base_metrics["0.7-r100"]),
            },
        },
    }


FLAG_KEYS = ["0.5-r1", "0.5-r5", "0.5-r10", "0.5-r100", "0.7-r1", "0.7-r5", "0.7-r10", "0.7-r100"]


def flags_from_labels(l05: Sequence[bool], l07: Sequence[bool]) -> np.ndarray:
    return np.asarray([
        any(l05[:1]), any(l05[:5]), any(l05[:10]), any(l05[:100]),
        any(l07[:1]), any(l07[:5]), any(l07[:10]), any(l07[:100]),
    ], dtype=bool)


def precompute_eval_cache(queries: List[Dict[str, Any]], records: List[Dict[str, Any]], args: argparse.Namespace) -> Dict[str, Any]:
    n = len(queries)
    base_flags = np.zeros((n, len(FLAG_KEYS)), dtype=bool)
    base_pos = np.zeros(n, dtype=bool)
    base_invalid = np.zeros(n, dtype=np.int16)
    base_dup = np.zeros(n, dtype=np.int16)
    variants: Dict[str, Dict[str, Any]] = {}
    for variant in ["SN", "RRF"]:
        variants[variant] = {
            "flags": np.zeros((n, len(FLAG_KEYS)), dtype=bool),
            "pos": np.zeros(n, dtype=bool),
            "invalid": np.zeros(n, dtype=np.int16),
            "dup": np.zeros(n, dtype=np.int16),
            "top1_changed": np.zeros(n, dtype=bool),
            "top5_changed": np.zeros(n, dtype=bool),
            "top10_changed": np.zeros(n, dtype=bool),
            "rank_A4_replacement": np.zeros(n, dtype=np.int16),
        }
    for q, (qd, rec) in enumerate(zip(queries, records)):
        base_raw, _c, _r = sequence_for_record(qd, rec, "A4")
        base_ids = [cand_id(c) for c in base_raw]
        base_seq = nms_sequence(base_raw[:args.effective_top_n], args.nms_thd, args.max_after_nms)
        b05, b07, bpos, binv, bdup = labels_for_sequence(base_seq, int(qd["gt_vid"]), int(qd["gt_s"]), int(qd["gt_e"]))
        base_flags[q] = flags_from_labels(b05, b07)
        base_pos[q] = bpos
        base_invalid[q] = binv
        base_dup[q] = bdup
        for variant in ["SN", "RRF"]:
            raw, _changed, repl_rank = sequence_for_record(qd, rec, "override", guard=100, candidate=variant)
            ids = [cand_id(c) for c in raw]
            seq = nms_sequence(raw[:args.effective_top_n], args.nms_thd, args.max_after_nms)
            l05, l07, pos, inv, dup = labels_for_sequence(seq, int(qd["gt_vid"]), int(qd["gt_s"]), int(qd["gt_e"]))
            v = variants[variant]
            v["flags"][q] = flags_from_labels(l05, l07)
            v["pos"][q] = pos
            v["invalid"][q] = inv
            v["dup"][q] = dup
            v["top1_changed"][q] = base_ids[:1] != ids[:1]
            v["top5_changed"][q] = set(base_ids[:5]) != set(ids[:5])
            v["top10_changed"][q] = set(base_ids[:10]) != set(ids[:10])
            v["rank_A4_replacement"][q] = repl_rank
    return {
        "base": {"flags": base_flags, "pos": base_pos, "invalid": base_invalid, "dup": base_dup},
        "variants": variants,
    }


def metrics_from_flags(flags: np.ndarray) -> Dict[str, float]:
    n = max(int(flags.shape[0]), 1)
    vals = 100.0 * flags.astype(np.float32).sum(axis=0) / float(n)
    return {k: float(v) for k, v in zip(FLAG_KEYS, vals)}


def eval_cached(
    name: str,
    cache: Dict[str, Any],
    idx: Sequence[int],
    variant: str,
    allow_mask: np.ndarray,
) -> Dict[str, Any]:
    ii = np.asarray(idx, dtype=np.int64)
    use = allow_mask[ii].astype(bool)
    base = cache["base"]
    var = cache["variants"][variant]
    flags = np.where(use[:, None], var["flags"][ii], base["flags"][ii])
    base_flags = base["flags"][ii]
    metrics = metrics_from_flags(flags)
    base_metrics = metrics_from_flags(base_flags)
    n = max(len(ii), 1)
    gains = {}
    for col, key in enumerate(FLAG_KEYS):
        gains[key] = {
            "gain_queries": int((~base_flags[:, col] & flags[:, col]).sum()),
            "loss_queries": int((base_flags[:, col] & ~flags[:, col]).sum()),
        }
    repl_rank = var["rank_A4_replacement"][ii]
    gain07 = use & (~base_flags[:, 4]) & flags[:, 4]
    loss07 = use & base_flags[:, 4] & (~flags[:, 4])
    gains_by_rank = {
        "<=5": int((gain07 & (repl_rank <= 5)).sum()),
        "<=10": int((gain07 & (repl_rank > 5) & (repl_rank <= 10)).sum()),
        ">10": int((gain07 & (repl_rank > 10)).sum()),
    }
    loss_by_rank = {
        "<=5": int((loss07 & (repl_rank <= 5)).sum()),
        "<=10": int((loss07 & (repl_rank > 5) & (repl_rank <= 10)).sum()),
        ">10": int((loss07 & (repl_rank > 10)).sum()),
    }
    pos = np.where(use, var["pos"][ii], base["pos"][ii])
    invalid = np.where(use, var["invalid"][ii], base["invalid"][ii])
    dup = np.where(use, var["dup"][ii], base["dup"][ii])
    exits = int((base["pos"][ii] & ~pos).sum())
    entries = int((~base["pos"][ii] & pos).sum())
    return {
        "name": name,
        "query_count": int(len(ii)),
        "metrics": metrics,
        "delta_vs_C7_B2_1_A4": metric_delta(metrics, base_metrics),
        "changed_query_count": int(use.sum()),
        "changed_query_rate": float(use.sum() / n),
        "top1_changed_rate": float((use & var["top1_changed"][ii]).sum() / n),
        "top5_set_changed_rate": float((use & var["top5_changed"][ii]).sum() / n),
        "top10_set_changed_rate": float((use & var["top10_changed"][ii]).sum() / n),
        "gain_loss_vs_baseline": gains,
        "r1_07_gain_by_A4_rank_of_replacement": gains_by_rank,
        "r1_07_loss_by_A4_rank_of_replacement": loss_by_rank,
        "movement": {
            "hard_positive_top100_query_exits": exits,
            "hard_positive_top100_query_entries": entries,
            "hard_positive_exit_ratio": float(exits / n),
            "invalid_span_count": int(invalid.sum()),
            "duplicate_span_count_after_nms": int(dup.sum()),
            "fixed_pool_invariant": True,
            "R100_delta": {
                "0.5-r100": float(metrics["0.5-r100"] - base_metrics["0.5-r100"]),
                "0.7-r100": float(metrics["0.7-r100"] - base_metrics["0.7-r100"]),
            },
        },
    }


def percentile_thresholds(values: np.ndarray, specs: Sequence[int]) -> Dict[str, float]:
    return {f"p{p}": float(np.quantile(values, p / 100.0)) for p in specs}


def rule_allowed(rec: Dict[str, Any], k_guard: int, sn_margin: float, a4_uncert: float, video_limit: int) -> bool:
    return (
        int(rec["rank_A4_SN"]) <= int(k_guard)
        and float(rec["sn_margin"]) >= float(sn_margin)
        and float(rec["a4_margin"]) <= float(a4_uncert)
        and int(rec["video_rank_SN"]) <= int(video_limit)
    )


def phase_b_search(records: List[Dict[str, Any]], splits: Dict[str, List[int]], eval_cache: Dict[str, Any]) -> Dict[str, Any]:
    sn_margins = np.asarray([r["sn_margin"] for r in records], dtype=np.float32)
    a4_margins = np.asarray([r["a4_margin"] for r in records], dtype=np.float32)
    sn_thr = percentile_thresholds(sn_margins, [50, 60, 70, 80])
    a4_thr = percentile_thresholds(a4_margins, [20, 30, 40, 50])
    records_out = []
    for k in [5, 10]:
        for sn_name, sn_t in sn_thr.items():
            for a4_name, a4_t in a4_thr.items():
                for vlim in [3, 5, 10]:
                    cfg = {"K_guard": k, "SN_margin_threshold_name": sn_name, "SN_margin_threshold": sn_t, "A4_uncertainty_threshold_name": a4_name, "A4_uncertainty_threshold": a4_t, "video_rank_limit": vlim}
                    allow = np.asarray([rule_allowed(r, cfg["K_guard"], cfg["SN_margin_threshold"], cfg["A4_uncertainty_threshold"], cfg["video_rank_limit"]) for r in records], dtype=bool)
                    stress = eval_cached("rule_gated_top1_override", eval_cache, splits["official_like_stress"], "SN", allow)
                    calib_a = eval_cached("rule_gated_top1_override", eval_cache, splits["calib_A"], "SN", allow)
                    calib_b = eval_cached("rule_gated_top1_override", eval_cache, splits["calib_B"], "SN", allow)
                    calib_c = eval_cached("rule_gated_top1_override", eval_cache, splits["calib_C"], "SN", allow)
                    full = eval_cached("rule_gated_top1_override", eval_cache, splits["standard_train_calib"], "SN", allow)
                    mode_limit = 0.05
                    safety = (
                        stress["delta_vs_C7_B2_1_A4"]["0.7-r10"] >= -0.10
                        and stress["movement"]["R100_delta"]["0.5-r100"] == 0.0
                        and stress["movement"]["R100_delta"]["0.7-r100"] == 0.0
                        and stress["movement"]["hard_positive_top100_query_exits"] == 0
                        and stress["movement"]["invalid_span_count"] == 0
                        and stress["movement"]["duplicate_span_count_after_nms"] == 0
                        and (stress["top5_set_changed_rate"] <= mode_limit if k == 5 else stress["top10_set_changed_rate"] <= mode_limit)
                    )
                    if k == 5:
                        safety = safety and stress["delta_vs_C7_B2_1_A4"]["0.7-r5"] >= -0.10
                    records_out.append({"config": cfg, "stress": stress, "calib_A": calib_a, "calib_B": calib_b, "calib_C": calib_c, "standard_train_calib": full, "safety_pass": bool(safety)})
    best = sorted(records_out, key=lambda r: (r["safety_pass"], r["stress"]["delta_vs_C7_B2_1_A4"]["0.7-r1"], r["stress"]["delta_vs_C7_B2_1_A4"]["0.7-r5"], -r["stress"]["top5_set_changed_rate"]), reverse=True)[0]
    positive_count = int(sum(r["stress"]["delta_vs_C7_B2_1_A4"]["0.7-r1"] > 0 for r in records_out))
    safe_count = int(sum(r["safety_pass"] for r in records_out))
    return {
        "status": "C7_B6_RULE_GATED_TOP1_OVERRIDE_COMPLETE",
        "official_val_used": False,
        "thresholds": {"SN_margin_top1_top2": sn_thr, "A4_top1_uncertainty": a4_thr},
        "grid_count": len(records_out),
        "positive_0.7_r1_on_stress_count": positive_count,
        "safety_pass_count": safe_count,
        "best_config": best["config"],
        "best_record": best,
        "all_records": records_out,
    }


class GateMLP(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, 32), nn.ReLU(), nn.Dropout(0.05), nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def make_gate_dataset(records: List[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    names = [
        "rank_A4_SN", "rank_SN_A4", "a4_margin", "sn_margin", "sn_delta_top", "a4_delta_top",
        "video_rank_SN", "video_rank_A4", "duration", "overlap_SN_A4", "b_end_margin",
        "proposal_confidence", "span_quality", "residual_SN", "residual_A4",
    ]
    x = []
    y = []
    for r in records:
        sn = int(r["sn_top"])
        a4 = int(r["a4_top"])
        x.append([
            float(r["rank_A4_SN"]), float(r["rank_SN_A4"]), float(r["a4_margin"]), float(r["sn_margin"]),
            float(r["sn_delta_top"]), float(r["a4_delta_top"]), float(r["video_rank_SN"]), float(r["video_rank_A4"]),
            float(r["duration"]), float(r["overlap_SN_A4"]), float(r["b_end_margin"]),
            float(r["proposal_confidence"]), float(r["span_quality"]), float(r["residual"][sn]), float(r["residual"][a4]),
        ])
        y.append([float(r["label_better_07"]), float(r["label_better_05"]), float(r["label_harmful_07"])])
    return np.asarray(x, dtype=np.float32), np.asarray(y, dtype=np.float32), names


def train_gate_model(records: List[Dict[str, Any]], splits: Dict[str, List[int]], eval_cache: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    x, y, names = make_gate_dataset(records)
    train_idx = np.asarray(sorted(set(splits["calib_A"] + splits["calib_B"])), dtype=np.int64)
    tune_idx = np.asarray(splits["calib_C"], dtype=np.int64)
    mu = x[train_idx].mean(axis=0)
    sig = x[train_idx].std(axis=0) + 1e-6
    xz = (x - mu) / sig
    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    model = GateMLP(x.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    xt = torch.from_numpy(xz[train_idx]).to(device)
    yt = torch.from_numpy(y[train_idx]).to(device)
    pos = yt.sum(dim=0)
    neg = yt.shape[0] - pos
    pos_weight = torch.clamp(neg / torch.clamp(pos, min=1.0), max=25.0)
    for _ in range(args.epochs):
        model.train()
        opt.zero_grad(set_to_none=True)
        logits = model(xt)
        loss = (
            F.binary_cross_entropy_with_logits(logits[:, 0], yt[:, 0], pos_weight=pos_weight[0])
            + 0.5 * F.binary_cross_entropy_with_logits(logits[:, 1], yt[:, 1], pos_weight=pos_weight[1])
            + F.binary_cross_entropy_with_logits(logits[:, 2], yt[:, 2], pos_weight=pos_weight[2])
        )
        loss.backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        probs = torch.sigmoid(model(torch.from_numpy(xz).to(device))).cpu().numpy()
    margins = probs[:, 0] - probs[:, 2]
    candidates = []
    for k_guard in [5, 10]:
        for tau in np.linspace(-0.50, 0.80, 27):
            cfg = {"K_guard": int(k_guard), "tau": float(tau), "model": "GateMLP_15_32_16_3", "train_split": "calib_A+calib_B", "tau_tune_reference": "calib_C/stress train-only"}
            allow = np.asarray([(margins[int(r["q"])] > cfg["tau"] and int(r["rank_A4_SN"]) <= cfg["K_guard"]) for r in records], dtype=bool)
            stress = eval_cached("selective_gate_model", eval_cache, splits["official_like_stress"], "SN", allow)
            calib_a = eval_cached("selective_gate_model", eval_cache, splits["calib_A"], "SN", allow)
            calib_b = eval_cached("selective_gate_model", eval_cache, splits["calib_B"], "SN", allow)
            calib_c = eval_cached("selective_gate_model", eval_cache, tune_idx.tolist(), "SN", allow)
            full = eval_cached("selective_gate_model", eval_cache, splits["standard_train_calib"], "SN", allow)
            safety = (
                stress["delta_vs_C7_B2_1_A4"]["0.7-r10"] >= -0.10
                and stress["movement"]["hard_positive_top100_query_exits"] == 0
                and stress["movement"]["invalid_span_count"] == 0
                and stress["movement"]["duplicate_span_count_after_nms"] == 0
                and stress["movement"]["R100_delta"]["0.5-r100"] == 0.0
                and stress["movement"]["R100_delta"]["0.7-r100"] == 0.0
                and (stress["top5_set_changed_rate"] <= 0.05 if k_guard == 5 else stress["top10_set_changed_rate"] <= 0.05)
            )
            candidates.append({"config": cfg, "stress": stress, "calib_A": calib_a, "calib_B": calib_b, "calib_C": calib_c, "standard_train_calib": full, "safety_pass": bool(safety)})
    best = sorted(candidates, key=lambda r: (r["safety_pass"], r["stress"]["delta_vs_C7_B2_1_A4"]["0.7-r1"], np.median([r[x]["delta_vs_C7_B2_1_A4"]["0.7-r1"] for x in ["calib_A", "calib_B", "calib_C"]]), -r["stress"]["top5_set_changed_rate"]), reverse=True)[0]
    Path(args.model_dir).mkdir(parents=True, exist_ok=True)
    model_path = Path(args.model_dir) / "c7_b6_r1_selective_gate.pt"
    torch.save({
        "state_dict": model.state_dict(),
        "feature_names": names,
        "mean": mu,
        "std": sig,
        "best_config": best["config"],
        "official_val_used": False,
    }, model_path)
    return {
        "status": "C7_B6_SELECTIVE_GATE_MODEL_COMPLETE",
        "official_val_used": False,
        "device": str(device),
        "model": {"path": str(model_path), "sha256": sha256_file(model_path), "architecture": "small MLP 15-32-16-3"},
        "feature_names": names,
        "label_positive_counts": {"replace_better_07": int(y[:, 0].sum()), "replace_better_05": int(y[:, 1].sum()), "replace_harmful_07": int(y[:, 2].sum())},
        "train_split": {"name": "calib_A+calib_B", "query_count": int(len(train_idx))},
        "tau_grid_count": len(candidates),
        "best_config": best["config"],
        "best_record": best,
        "all_records": candidates,
    }


def final_decision(phase_b: Dict[str, Any], phase_c: Dict[str, Any] | None, splits: Dict[str, List[int]]) -> Dict[str, Any]:
    source = "phase_c_gate_model" if phase_c and phase_c.get("status") == "C7_B6_SELECTIVE_GATE_MODEL_COMPLETE" else "phase_b_rule_gated"
    rec = phase_c["best_record"] if source == "phase_c_gate_model" else phase_b["best_record"]
    stress = rec["stress"]
    full = rec["standard_train_calib"]
    calib_deltas = []
    if "calib_C" in rec:
        calib_deltas = [rec[x]["delta_vs_C7_B2_1_A4"]["0.7-r1"] for x in ["calib_A", "calib_B", "calib_C"]]
    stress_r1 = stress["delta_vs_C7_B2_1_A4"]["0.7-r1"]
    r5 = stress["delta_vs_C7_B2_1_A4"]["0.7-r5"]
    r10 = stress["delta_vs_C7_B2_1_A4"]["0.7-r10"]
    bounded = r5 >= -0.10 and r10 >= -0.10
    topk_ok = stress["top5_set_changed_rate"] <= 0.05 or stress["top10_set_changed_rate"] <= 0.05
    infra_ok = (
        stress["movement"]["hard_positive_top100_query_exits"] == 0
        and stress["movement"]["invalid_span_count"] == 0
        and stress["movement"]["duplicate_span_count_after_nms"] == 0
        and stress["movement"]["R100_delta"]["0.5-r100"] == 0.0
        and stress["movement"]["R100_delta"]["0.7-r100"] == 0.0
    )
    if not infra_ok:
        status = "C7_B6_INFRA_FAIL"
    elif stress_r1 > 0 and bounded and topk_ok and full["delta_vs_C7_B2_1_A4"]["0.7-r1"] >= -0.10:
        status = "C7_B6_R1_SELECTIVE_PROMISING"
    elif stress_r1 > 0:
        status = "C7_B6_R1_SELECTIVE_R1_ONLY_TRADEOFF"
    elif bounded and topk_ok:
        status = "C7_B6_SAFE_BUT_NO_GAIN"
    else:
        status = "C7_B6_NEGATIVE"
    return {
        "status": status,
        "official_val_used": False,
        "second_official_val": False,
        "post_val_adjustment": False,
        "C7_B2_1_promoted_decision_modified": False,
        "C7_B3_C7_B4_C7_B5_1_archived_decisions_modified": False,
        "NMS_modified": False,
        "evaluator_modified": False,
        "CONQUER_QDF_QAL_original_ML_VR_frozen": True,
        "full_fine_tuning": False,
        "top100_full_rerank_promoted_candidate": False,
        "S_MINUTE_like_direct_replacement": False,
        "selected_source": source,
        "selected_config": rec["config"],
        "selected_stress_record": stress,
        "selected_standard_train_calib_record": full,
        "calib_ABC_0.7_r1_median_delta": float(np.median(calib_deltas)) if calib_deltas else None,
        "selection_notes": "Candidate is train-only only; do not run official val automatically.",
    }


def main() -> None:
    args = parse_args()
    audit = Path(args.audit_dir)
    audit.mkdir(parents=True, exist_ok=True)
    queries = build_queries(args)["queries"]
    feats = load_features(args, len(queries))
    records = build_query_records(queries, feats)
    splits = split_indices(queries)
    eval_cache = precompute_eval_cache(queries, records, args)

    counterfactual: Dict[str, Any] = {
        "status": "C7_B6_COUNTERFACTUAL_TOP1_SWAP_ORACLE_COMPLETE",
        "official_val_used": False,
        "baseline": "C7-B2.1 A4 fixed-pool ranking",
        "systems": {},
    }
    for split_name in ["standard_train_calib", "calib_A", "calib_B", "calib_C", "official_like_stress"]:
        counterfactual["systems"][split_name] = {}
        idx = splits[split_name]
        for cand in ["SN", "RRF"]:
            for label, guard in [("S0_top5_internal", 5), ("S1_top10_internal", 10), ("S2_top20_diagnostic", 20)]:
                if cand == "SN":
                    rank_field = "rank_A4_SN"
                else:
                    rank_field = "rank_A4_RRF"
                allow = np.asarray([int(r[rank_field]) <= guard for r in records], dtype=bool)
                counterfactual["systems"][split_name][f"{label}_{cand}"] = eval_cached(f"{label}_{cand}", eval_cache, idx, cand, allow)
    atomic_json(audit / "C7_B6_COUNTERFACTUAL_TOP1_SWAP_ORACLE.json", counterfactual)
    atomic_text(audit / "C7_B6_COUNTERFACTUAL_TOP1_SWAP_ORACLE.md", md("C7-B6 counterfactual top1 swap oracle", counterfactual))

    phase_b = phase_b_search(records, splits, eval_cache)
    atomic_json(audit / "C7_B6_RULE_GATED_TOP1_OVERRIDE.json", phase_b)
    atomic_text(audit / "C7_B6_RULE_GATED_TOP1_OVERRIDE.md", md("C7-B6 rule-gated top1 override", phase_b))

    if phase_b["positive_0.7_r1_on_stress_count"] > 0:
        phase_c = train_gate_model(records, splits, eval_cache, args)
    else:
        phase_c = {
            "status": "C7_B6_SELECTIVE_GATE_MODEL_SKIPPED",
            "official_val_used": False,
            "reason": "Phase B produced no positive 0.7-r1 result on official-like stress split.",
        }
    atomic_json(audit / "C7_B6_SELECTIVE_GATE_MODEL.json", phase_c)
    atomic_text(audit / "C7_B6_SELECTIVE_GATE_MODEL.md", md("C7-B6 selective gate model", phase_c))

    final = final_decision(phase_b, phase_c if phase_c.get("status") == "C7_B6_SELECTIVE_GATE_MODEL_COMPLETE" else None, splits)
    atomic_json(audit / "C7_B6_FINAL_DECISION.json", final)
    atomic_text(audit / "C7_B6_FINAL_DECISION.md", md("C7-B6 final decision", final))
    manifest = {
        **final,
        "outputs": {
            "counterfactual": "c7_audit/C7_B6_COUNTERFACTUAL_TOP1_SWAP_ORACLE.json",
            "rule_gated": "c7_audit/C7_B6_RULE_GATED_TOP1_OVERRIDE.json",
            "gate_model": "c7_audit/C7_B6_SELECTIVE_GATE_MODEL.json",
            "decision": "c7_audit/C7_B6_FINAL_DECISION.json",
        },
    }
    atomic_json(audit / "C7_B6_MANIFEST.json", manifest)
    paths = [
        audit / "C7_B6_COUNTERFACTUAL_TOP1_SWAP_ORACLE.json",
        audit / "C7_B6_RULE_GATED_TOP1_OVERRIDE.json",
        audit / "C7_B6_SELECTIVE_GATE_MODEL.json",
        audit / "C7_B6_FINAL_DECISION.json",
        audit / "C7_B6_MANIFEST.json",
    ]
    hashes = {
        "status": "C7_B6_HASHES",
        "official_val_used": False,
        "artifacts": {str(p): sha256_file(p) for p in paths},
        "runner_sha256": sha256_file(__file__),
        "fixed_pool_logits_sha256": sha256_file(args.fixed_pool_logits_npz),
        "manifest_hash": sha256_obj(manifest),
    }
    atomic_json(audit / "C7_B6_HASHES.json", hashes)
    print(json.dumps({
        "status": final["status"],
        "selected_source": final["selected_source"],
        "selected_config": final["selected_config"],
        "stress_delta": final["selected_stress_record"]["delta_vs_C7_B2_1_A4"],
        "official_val_used": False,
        "post_val_adjustment": False,
        "second_official_val": False,
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
