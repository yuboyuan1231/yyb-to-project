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

from rlem_c6_b0.run_c6_b0_candidate_diagnostic import span_iou_idx  # noqa: E402
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


METRIC_KEYS = [
    "0.5-r1", "0.5-r5", "0.5-r10", "0.5-r100",
    "0.7-r1", "0.7-r5", "0.7-r10", "0.7-r100",
]
RANKS = [1, 5, 10]
CURRENT_TARGET = {
    "0.7-r1": 30.5413,
    "0.5-r1": 45.6574,
    "0.7-r5": 53.1754,
    "0.5-r5": 66.9070,
    "0.7-r100_delta": -6.4309,
    "0.5-r100_delta": -2.0140,
    "hard_exit_ratio": 0.0261,
}

Candidate = Tuple[int, int, int, int, float, int]


def sha256_obj(obj: Any) -> str:
    data = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--calib_cache", default="results/rlem_c6a/cache/train_calib_c6_cache.npz")
    p.add_argument("--c7_b2_eval", default="c7_audit/C7_B2_TRAIN_CALIB_EVAL.json")
    p.add_argument("--c7_b2_final", default="c7_audit/C7_B2_FINAL_DECISION.json")
    p.add_argument("--anchor_states", default="results/rlem_c7_b2/train_calib_c7_b1_anchor_states.json")
    p.add_argument("--video_scores", default="results/rlem_c7_b2/train_calib_video_scores.npz")
    p.add_argument("--proposal_scores", default="results/rlem_c7_b2/train_calib_c7_b1_proposal_scores.npz")
    p.add_argument("--audit_dir", default="c7_audit")
    p.add_argument("--output_dir", default="results/rlem_c7_b2_1")
    p.add_argument("--effective_top_n", type=int, default=100)
    p.add_argument("--max_after_nms", type=int, default=100)
    p.add_argument("--nms_thd", type=float, default=0.7)
    return p.parse_args()


def normalize_cand(raw: Sequence[Any]) -> Candidate:
    return (int(raw[0]), int(raw[1]), int(raw[2]), int(raw[3]), float(raw[4]), int(raw[5]))


def cand_key(c: Candidate) -> Tuple[int, int, int]:
    return (int(c[1]), int(c[2]), int(c[3]))


def load_baseline_payload(args: argparse.Namespace) -> Dict[str, Any]:
    obj = json.loads(Path(args.c7_b2_eval).read_text(encoding="utf-8"))
    if obj.get("official_val_used") is not False:
        raise ValueError("C7-B2.1 must not consume official-val-selected state")
    return obj


def group_score_map(score_arrays: Dict[str, np.ndarray], cfg: Dict[str, Any]) -> Dict[int, Tuple[float, float, float, float]]:
    out: Dict[int, Tuple[float, float, float, float]] = {}
    gids = score_arrays["group_id"].astype(np.int64)
    base_rank = score_arrays["base_rank"].astype(np.float32)
    for i, gid in enumerate(gids):
        gate = float(score_arrays["gate"][i])
        mil = gate * (float(score_arrays["relevance"][i]) + float(score_arrays["moment"][i]) + 0.35 * float(score_arrays["iou"][i]))
        residual = (
            float(cfg["beta"]) * gate * float(score_arrays["relevance"][i])
            + float(cfg["gamma"]) * gate * (float(score_arrays["moment"][i]) + 0.35 * float(score_arrays["iou"][i]))
            - float(cfg["rho"]) * gate * float(score_arrays["risk"][i])
        )
        if gate < float(cfg["gate_threshold"]):
            residual = 0.0
        base_component = 1.0 / float(base_rank[i] + 1.0)
        total = float(cfg["alpha"]) * base_component + residual
        out[int(gid)] = (total, gate, residual, mil)
    return out


def moved_groups_for_state(state: Dict[str, Any], gscore: Dict[int, Tuple[float, float, float, float]], cfg: Dict[str, Any]) -> Tuple[set[int], float]:
    base_order = [int(x) for x in state["base_group_order"]]
    reranked = sorted(base_order, key=lambda gid: -gscore.get(gid, (0.0, 0.0, 0.0, 0.0))[0])
    final_groups = list(base_order)
    moved: set[int] = set()
    changes = 0
    for pos, gid in enumerate(reranked):
        if pos >= len(final_groups) or changes >= int(cfg["max_video_slot_changes"]):
            break
        if final_groups[pos] == gid:
            continue
        if gscore.get(gid, (0.0, 0.0, 0.0, 0.0))[1] < float(cfg["gate_threshold"]):
            continue
        old = final_groups.index(gid)
        final_groups.pop(old)
        final_groups.insert(pos, gid)
        moved.add(gid)
        changes += 1
    drift = float(sum(int(a != b) for a, b in zip(base_order, final_groups)) / max(len(final_groups), 1))
    return moved, drift


def c7b2_ranked_candidates(state: Dict[str, Any], gscore: Dict[int, Tuple[float, float, float, float]], cfg: Dict[str, Any]) -> Tuple[List[Candidate], float]:
    moved, drift = moved_groups_for_state(state, gscore, cfg)
    out: List[Candidate] = []
    for raw in state["flat_candidates"]:
        c = normalize_cand(raw)
        residual = gscore.get(int(c[0]), (0.0, 0.0, 0.0, 0.0))[2] if int(c[0]) in moved else 0.0
        out.append((c[0], c[1], c[2], c[3], float(c[4]) + float(cfg["video_weight"]) * residual, c[5]))
    return sorted(out, key=lambda c: -c[4]), drift


def mil_ranked_candidates(state: Dict[str, Any], gscore: Dict[int, Tuple[float, float, float, float]], cfg: Dict[str, Any], with_residual: bool = False) -> Tuple[List[Candidate], float]:
    base_order = [int(x) for x in state["base_group_order"]]
    reranked = sorted(base_order, key=lambda gid: -gscore.get(gid, (0.0, 0.0, 0.0, 0.0))[3])
    max_changes = int(cfg["max_video_slot_changes"]) if with_residual else 0
    final_groups = list(base_order)
    moved: set[int] = set()
    changes = 0
    for pos, gid in enumerate(reranked):
        if pos >= len(final_groups) or changes >= max_changes:
            break
        if final_groups[pos] == gid:
            continue
        if gscore.get(gid, (0.0, 0.0, 0.0, 0.0))[1] < float(cfg["gate_threshold"]):
            continue
        old = final_groups.index(gid)
        final_groups.pop(old)
        final_groups.insert(pos, gid)
        moved.add(gid)
        changes += 1
    drift = float(sum(int(a != b) for a, b in zip(base_order, final_groups)) / max(len(final_groups), 1))
    rows: List[Candidate] = []
    by_gid: Dict[int, List[Candidate]] = {}
    for raw in state["flat_candidates"]:
        c = normalize_cand(raw)
        by_gid.setdefault(int(c[0]), []).append(c)
    for gid in final_groups:
        delta = gscore.get(gid, (0.0, 0.0, 0.0, 0.0))[2] if gid in moved else 0.0
        rows.extend([(c[0], c[1], c[2], c[3], float(c[4]) + float(cfg["video_weight"]) * delta, c[5]) for c in by_gid.get(gid, [])])
        if len(rows) >= 100:
            break
    return rows, drift


def nms(cands: Sequence[Candidate], args: argparse.Namespace) -> List[Candidate]:
    return nms_sequence(list(cands)[:args.effective_top_n], args.nms_thd, args.max_after_nms)


def merge_unique(first: Sequence[Candidate], *rest: Sequence[Candidate], limit: int = 100) -> List[Candidate]:
    out: List[Candidate] = []
    seen = set()
    for seq in (first,) + rest:
        for c in seq:
            key = cand_key(c)
            if key in seen:
                continue
            seen.add(key)
            out.append(c)
            if len(out) >= limit:
                return out
    return out


def make_protection_maps(args: argparse.Namespace) -> Dict[str, np.ndarray]:
    prop = np.load(args.proposal_scores, allow_pickle=False)
    rows = prop["row"].astype(np.int64)
    max_row = int(rows.max()) + 1
    conf = np.zeros(max_row, dtype=np.float32)
    qual = np.zeros(max_row, dtype=np.float32)
    conf[rows] = prop["proposal_confidence"].astype(np.float32)
    qual[rows] = prop["span_quality"].astype(np.float32)
    qvals = {
        "q70": float(np.quantile(prop["proposal_confidence"], 0.7)),
        "q80": float(np.quantile(prop["proposal_confidence"], 0.8)),
        "q90": float(np.quantile(prop["proposal_confidence"], 0.9)),
    }
    return {"conf": conf, "qual": qual, "quantiles": qvals}


def protection_score(c: Candidate, maps: Dict[str, np.ndarray], tail_weight: float = 0.2) -> float:
    row = int(c[5])
    conf = float(maps["conf"][row]) if 0 <= row < len(maps["conf"]) else 0.0
    qual = float(maps["qual"][row]) if 0 <= row < len(maps["qual"]) else 0.0
    utility = float(c[4])
    tail = tail_weight / float(max(abs(utility), 1.0))
    return 0.55 * conf + 0.25 * np.tanh(qual) + 0.15 * np.tanh(utility / 5.0) + 0.05 * tail


def evaluate_sequences(
    name: str,
    sequences: List[List[Candidate]],
    anchor_sequences: List[List[Candidate]],
    gt_vid: np.ndarray,
    gt_s: np.ndarray,
    gt_e: np.ndarray,
    drift_values: List[float],
) -> Dict[str, Any]:
    labels05: List[List[bool]] = []
    labels07: List[List[bool]] = []
    anchor05: List[List[bool]] = []
    anchor07: List[List[bool]] = []
    invalid = 0
    dup = 0
    lost_top100: List[int] = []
    entries = exits = 0
    for q, seq in enumerate(sequences):
        l05, l07, pos, inv, du = labels_for_sequence(seq, int(gt_vid[q]), int(gt_s[q]), int(gt_e[q]))
        a05, a07, apos, _ai, _ad = labels_for_sequence(anchor_sequences[q], int(gt_vid[q]), int(gt_s[q]), int(gt_e[q]))
        labels05.append(l05)
        labels07.append(l07)
        anchor05.append(a05)
        anchor07.append(a07)
        invalid += inv
        dup += du
        exits += int(apos and not pos)
        entries += int((not apos) and pos)
        if apos and not pos:
            lost_top100.append(int(q))
    metrics = selected_metrics(labels05, labels07)
    gain_loss: Dict[str, Dict[str, int]] = {}
    for thr, cur, base in [("0.5", labels05, anchor05), ("0.7", labels07, anchor07)]:
        for r in RANKS:
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
            "hard_positive_exit_ratio": float(exits / max(sum(labels_for_sequence(a, int(gt_vid[q]), int(gt_s[q]), int(gt_e[q]))[2] for q, a in enumerate(anchor_sequences)), 1)),
            "video_slot_drift_rate": float(np.mean(drift_values)) if drift_values else 0.0,
            "invalid_span_count": int(invalid),
            "duplicate_span_count_after_nms": int(dup),
        },
        "rank_gain_loss_vs_C7_B1_frozen": gain_loss,
        "top100_lost_query_count": len(lost_top100),
        "top100_lost_query_hash": sha256_obj(lost_top100),
    }


def add_deltas(rec: Dict[str, Any], baselines: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
    rec["delta_vs_C6_B1"] = metric_delta(rec["metrics"], baselines["C6_B1"])
    rec["delta_vs_C6_B2"] = metric_delta(rec["metrics"], baselines["C6_B2"])
    rec["delta_vs_C7_B1_frozen"] = metric_delta(rec["metrics"], baselines["C7_B1_frozen"])
    return rec


def classify(rec: Dict[str, Any]) -> str:
    m = rec["metrics"]
    d7 = rec["delta_vs_C7_B1_frozen"]
    mv = rec["movement"]
    strict_r1 = m["0.7-r1"] >= CURRENT_TARGET["0.7-r1"] and m["0.5-r1"] >= CURRENT_TARGET["0.5-r1"]
    near_r1 = m["0.7-r1"] >= 30.4913 and m["0.5-r1"] >= 45.6074
    r5_ok = m["0.7-r5"] >= 53.1254 and m["0.5-r5"] >= 66.8570
    exits_reduced = mv["hard_positive_exit_ratio"] < CURRENT_TARGET["hard_exit_ratio"]
    exits_strong = mv["hard_positive_exit_ratio"] <= 0.010
    coverage_repaired = d7["0.7-r100"] > CURRENT_TARGET["0.7-r100_delta"] and d7["0.5-r100"] > CURRENT_TARGET["0.5-r100_delta"]
    clean = mv["invalid_span_count"] == 0 and mv["duplicate_span_count_after_nms"] == 0
    if strict_r1 and r5_ok and exits_reduced and coverage_repaired and clean:
        return "C7_B2_1_R1_PRESERVED_COVERAGE_REPAIRED"
    if (strict_r1 or near_r1) and r5_ok and exits_reduced and clean:
        return "C7_B2_1_R1_PRESERVED_EXIT_REDUCED"
    if m["0.7-r1"] >= CURRENT_TARGET["0.7-r1"] and not exits_reduced:
        return "C7_B2_1_HIGH_R1_UNSAFE"
    if exits_strong and not near_r1:
        return "C7_B2_1_SAFE_BUT_R1_DROP"
    return "C7_B2_1_NEGATIVE"


def lexicographic_key(rec: Dict[str, Any]) -> Tuple[Any, ...]:
    m = rec["metrics"]
    d7 = rec["delta_vs_C7_B1_frozen"]
    mv = rec["movement"]
    strict_r1 = m["0.7-r1"] >= CURRENT_TARGET["0.7-r1"] and m["0.5-r1"] >= CURRENT_TARGET["0.5-r1"]
    near_r1 = m["0.7-r1"] >= 30.4913 and m["0.5-r1"] >= 45.6074
    r5_ok = m["0.7-r5"] >= 53.1254 and m["0.5-r5"] >= 66.8570
    exits_reduced = mv["hard_positive_exit_ratio"] < CURRENT_TARGET["hard_exit_ratio"]
    coverage_repaired = d7["0.7-r100"] > CURRENT_TARGET["0.7-r100_delta"] and d7["0.5-r100"] > CURRENT_TARGET["0.5-r100_delta"]
    clean = mv["invalid_span_count"] == 0 and mv["duplicate_span_count_after_nms"] == 0
    return (
        int(strict_r1),
        int(near_r1),
        int(r5_ok),
        int(exits_reduced),
        int(coverage_repaired),
        int(clean),
        -mv["hard_positive_exit_ratio"],
        d7["0.7-r100"],
        m["0.7-r1"],
        m["0.5-r1"],
    )


def main() -> None:
    args = parse_args()
    Path(args.audit_dir).mkdir(parents=True, exist_ok=True)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    cache = load_npz(args.calib_cache, allow_pickle=True)
    gt_vid = gt_video_by_query(cache)
    gt_s, gt_e = gt_span_by_query(cache)
    payload = load_baseline_payload(args)
    cfg = payload["best"]["config"]
    baselines = {
        "C6_B1": payload["baselines"]["C6_B1"]["metrics"],
        "C6_B2": payload["baselines"]["C6_B2"]["metrics"],
        "C7_B1_frozen": payload["baselines"]["C7_B1_frozen"]["metrics"],
    }
    states = json.loads(Path(args.anchor_states).read_text(encoding="utf-8"))["states"]
    score_arrays = load_npz(args.video_scores, allow_pickle=False)
    gscore = group_score_map(score_arrays, cfg)
    maps = make_protection_maps(args)

    anchor_ranked: List[List[Candidate]] = []
    anchor_kept: List[List[Candidate]] = []
    b2_ranked: List[List[Candidate]] = []
    b2_kept: List[List[Candidate]] = []
    b2_drifts: List[float] = []
    mil_ranked: List[List[Candidate]] = []
    video_ranked: List[List[Candidate]] = []
    for state in states:
        ar = [normalize_cand(x) for x in state["flat_candidates"]]
        br, drift = c7b2_ranked_candidates(state, gscore, cfg)
        mr, _ = mil_ranked_candidates(state, gscore, cfg, with_residual=False)
        vr = [(c[0], c[1], c[2], c[3], float(c[4]) + float(cfg["video_weight"]) * gscore.get(int(c[0]), (0.0, 0.0, 0.0, 0.0))[2], c[5]) for c in ar]
        vr = sorted(vr, key=lambda c: -c[4])
        anchor_ranked.append(ar)
        anchor_kept.append(nms(ar, args))
        b2_ranked.append(br)
        b2_kept.append(nms(br, args))
        b2_drifts.append(drift)
        mil_ranked.append(mr)
        video_ranked.append(vr)

    phase_a_specs = {
        "A0_C7_B1_frozen_anchor": (anchor_kept, [0.0] * len(states)),
        "A1_C7_B2_full_current_best": (b2_kept, b2_drifts),
        "A2_span_residual_only": (anchor_kept, [0.0] * len(states)),
        "A3_MIL_score_only": ([nms(x, args) for x in mil_ranked], [0.0] * len(states)),
        "A4_video_residual_only": ([nms(x, args) for x in video_ranked], [0.0] * len(states)),
        "A5_MIL_plus_span_residual": ([nms(x, args) for x in mil_ranked], [0.0] * len(states)),
        "A6_MIL_plus_video_residual_without_tail_protection": (b2_kept, b2_drifts),
        "A7_MIL_plus_video_residual_with_tail_protection": (
            [merge_unique(b2_kept[q][:20], anchor_kept[q], b2_kept[q][20:], limit=100) for q in range(len(states))],
            b2_drifts,
        ),
    }
    phase_a = [add_deltas(evaluate_sequences(k, seqs, anchor_kept, gt_vid, gt_s, gt_e, drifts), baselines) for k, (seqs, drifts) in phase_a_specs.items()]
    atomic_json(Path(args.audit_dir) / "C7_B2_1_DECOMPOSITION_AUDIT.json", {"official_val_used": False, "records": phase_a})
    atomic_text(Path(args.audit_dir) / "C7_B2_1_DECOMPOSITION_AUDIT.md", "# C7-B2.1 decomposition audit\n\n```json\n" + json.dumps({"official_val_used": False, "records": phase_a}, indent=2, ensure_ascii=False) + "\n```\n")

    phase_b_records = []
    for top_zone in [10, 20, 30, 50]:
        for tail_start in [20, 30, 50, 70]:
            seqs = []
            for q in range(len(states)):
                prefix = b2_kept[q][:tail_start]
                seqs.append(merge_unique(prefix, anchor_kept[q], b2_kept[q][tail_start:], limit=100))
            rec = evaluate_sequences(f"top_zone_{top_zone}_tail_start_{tail_start}", seqs, anchor_kept, gt_vid, gt_s, gt_e, b2_drifts)
            rec["config"] = {"top_zone": top_zone, "tail_fill_start": tail_start, "tail_source": "C7_B1_frozen"}
            phase_b_records.append(add_deltas(rec, baselines))
    atomic_json(Path(args.audit_dir) / "C7_B2_1_TOP_ZONE_PROTECTED_TAIL.json", {"official_val_used": False, "records": phase_b_records})

    phase_c_records = []
    qmap = maps["quantiles"]
    for top_p in [10, 20, 30, 50]:
        for qname in ["q70", "q80", "q90"]:
            thr = qmap[qname]
            for rerank_top in [10, 20, 30, 50]:
                seqs = []
                for qi in range(len(states)):
                    protected = [c for c in anchor_kept[qi] if (0 <= c[5] < len(maps["conf"]) and float(maps["conf"][c[5]]) >= thr)]
                    protected = sorted(protected, key=lambda c: -protection_score(c, maps))[:top_p]
                    union_tail = sorted(merge_unique(b2_kept[qi], anchor_kept[qi], limit=200), key=lambda c: -max(float(c[4]), protection_score(c, maps)))
                    seqs.append(merge_unique(b2_kept[qi][:rerank_top], protected, union_tail, limit=100))
                rec = evaluate_sequences(f"union_p{top_p}_{qname}_top{rerank_top}", seqs, anchor_kept, gt_vid, gt_s, gt_e, b2_drifts)
                rec["config"] = {"protected_anchor_top_p": top_p, "protected_conf_threshold": thr, "threshold_quantile": qname, "rerank_top_zone": rerank_top}
                phase_c_records.append(add_deltas(rec, baselines))
    atomic_json(Path(args.audit_dir) / "C7_B2_1_ANCHOR_UNION_RERANK.json", {"official_val_used": False, "records": phase_c_records})

    phase_d_records = []
    for lam in [0.05, 0.1, 0.2, 0.5, 1.0]:
        for tail_w in [0.1, 0.2, 0.5]:
            for beyond in [5, 10, 20]:
                seqs = []
                for q in range(len(states)):
                    rescored: List[Candidate] = []
                    for r, c in enumerate(b2_kept[q]):
                        penalty = 0.0
                        if r >= beyond and r < len(anchor_kept[q]):
                            penalty = lam * protection_score(anchor_kept[q][r], maps, tail_weight=tail_w)
                        rescored.append((c[0], c[1], c[2], c[3], float(c[4]) - penalty, c[5]))
                    seqs.append(merge_unique(sorted(rescored, key=lambda c: -c[4]), anchor_kept[q], limit=100))
                rec = evaluate_sequences(f"evict_lam{lam}_tail{tail_w}_beyond{beyond}", seqs, anchor_kept, gt_vid, gt_s, gt_e, b2_drifts)
                rec["config"] = {"lambda_evict": lam, "tail_coverage_weight": tail_w, "apply_only_beyond_rank": beyond}
                phase_d_records.append(add_deltas(rec, baselines))
    atomic_json(Path(args.audit_dir) / "C7_B2_1_EVICTION_AWARE_UTILITY.json", {"official_val_used": False, "records": phase_d_records})

    all_candidates = phase_a + phase_b_records + phase_c_records + phase_d_records
    for rec in all_candidates:
        rec["status_label"] = classify(rec)
    best = max(all_candidates, key=lexicographic_key)
    best["status_label"] = classify(best)
    phase_e = {
        "status": "not_run",
        "reason": "Phase B-D produced non-training safety repair candidates; hard-exit risk head is reserved for a later train_fit-only follow-up if human review requests it.",
        "official_val_used": False,
    }
    atomic_json(Path(args.audit_dir) / "C7_B2_1_HARD_EXIT_RISK_HEAD.json", phase_e)
    atomic_text(Path(args.audit_dir) / "C7_B2_1_HARD_EXIT_RISK_HEAD.md", "# C7-B2.1 hard-exit risk head\n\n```json\n" + json.dumps(phase_e, indent=2, ensure_ascii=False) + "\n```\n")

    manifest = {
        "status": best["status_label"],
        "official_val_used": False,
        "official_val_run": False,
        "post_val_adjustment": False,
        "second_official_val": False,
        "C7_B1_official_decision_modified": False,
        "NMS_modified": False,
        "evaluator_modified": False,
        "C4_C6_artifacts_modified": False,
        "CONQUER_backbone_frozen": True,
        "QDF_QAL_original_ML_VR_frozen": True,
        "selection_rule": "lexicographic R1 -> R5 -> exit reduction -> R100 repair -> clean invalid/duplicate",
        "best": best,
        "phase_counts": {
            "A": len(phase_a),
            "B": len(phase_b_records),
            "C": len(phase_c_records),
            "D": len(phase_d_records),
            "E": 0,
        },
        "baselines": baselines,
        "request": "Request human review before any official-val one-shot.",
    }
    atomic_json(Path(args.audit_dir) / "C7_B2_1_MANIFEST.json", manifest)
    atomic_json(Path(args.audit_dir) / "C7_B2_1_FINAL_DECISION.json", manifest)
    md = "# C7-B2.1 final decision\n\n"
    md += f"- Status: `{manifest['status']}`\n- Official val used: `false`\n- Best: `{best['name']}`\n\n"
    md += "## Best metrics\n\n```json\n" + json.dumps(best, indent=2, ensure_ascii=False) + "\n```\n\n"
    md += "Request human review before any official-val one-shot.\n"
    atomic_text(Path(args.audit_dir) / "C7_B2_1_FINAL_DECISION.md", md)
    for name, data in [
        ("C7_B2_1_TOP_ZONE_PROTECTED_TAIL.md", {"official_val_used": False, "records": phase_b_records}),
        ("C7_B2_1_ANCHOR_UNION_RERANK.md", {"official_val_used": False, "records": phase_c_records}),
        ("C7_B2_1_EVICTION_AWARE_UTILITY.md", {"official_val_used": False, "records": phase_d_records}),
    ]:
        atomic_text(Path(args.audit_dir) / name, "# " + name.replace(".md", "").replace("_", " ") + "\n\n```json\n" + json.dumps(data, indent=2, ensure_ascii=False) + "\n```\n")
    hashes = {
        "status": "C7_B2_1_HASHES",
        "official_val_used": False,
        "artifacts": {
            "runner": artifact(__file__),
            "manifest": artifact(Path(args.audit_dir) / "C7_B2_1_MANIFEST.json"),
            "final": artifact(Path(args.audit_dir) / "C7_B2_1_FINAL_DECISION.json"),
            "decomposition": artifact(Path(args.audit_dir) / "C7_B2_1_DECOMPOSITION_AUDIT.json"),
            "top_zone": artifact(Path(args.audit_dir) / "C7_B2_1_TOP_ZONE_PROTECTED_TAIL.json"),
            "union": artifact(Path(args.audit_dir) / "C7_B2_1_ANCHOR_UNION_RERANK.json"),
            "eviction": artifact(Path(args.audit_dir) / "C7_B2_1_EVICTION_AWARE_UTILITY.json"),
        },
    }
    atomic_json(Path(args.audit_dir) / "C7_B2_1_HASHES.json", hashes)
    print(json.dumps({
        "status": manifest["status"],
        "official_val_used": False,
        "best": best["name"],
        "metrics": best["metrics"],
        "delta_vs_C7_B1_frozen": best["delta_vs_C7_B1_frozen"],
        "movement": best["movement"],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
