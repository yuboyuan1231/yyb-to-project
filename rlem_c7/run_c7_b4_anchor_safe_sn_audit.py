#!/usr/bin/env python
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence, Tuple

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rlem_c7.run_c7_b0_b1 import (  # noqa: E402
    atomic_json,
    atomic_text,
    gt_span_by_query,
    gt_video_by_query,
    labels_for_sequence,
    load_npz,
    metric_delta,
    nms_sequence,
    selected_metrics,
)
from rlem_c7.run_c7_b2_1_safety_repair import group_score_map, normalize_cand  # noqa: E402
from rlem_c7.run_c7_b3_stable_generalization import transform_residual  # noqa: E402


METRIC_KEYS = [
    "0.5-r1", "0.5-r5", "0.5-r10", "0.5-r100",
    "0.7-r1", "0.7-r5", "0.7-r10", "0.7-r100",
]
Candidate = Tuple[int, int, int, int, float, int]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--cache", default="results/rlem_c6a/cache/train_calib_c6_cache.npz")
    p.add_argument("--anchor_states", default="results/rlem_c7_b2/train_calib_c7_b1_anchor_states.json")
    p.add_argument("--video_scores", default="results/rlem_c7_b2/train_calib_video_scores.npz")
    p.add_argument("--b2_eval", default="c7_audit/C7_B2_TRAIN_CALIB_EVAL.json")
    p.add_argument("--c7b3_forensic", default="c7_audit/C7_B3_FAILURE_FORENSIC_AUDIT.json")
    p.add_argument("--evidence_jsonl_gz", default="results/rlem_c3_minimal/train_calib_evidence.jsonl.gz")
    p.add_argument("--audit_dir", default="c7_audit")
    p.add_argument("--effective_top_n", type=int, default=100)
    p.add_argument("--max_after_nms", type=int, default=100)
    p.add_argument("--nms_thd", type=float, default=0.7)
    return p.parse_args()


def sha256_obj(obj: Any) -> str:
    data = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def stable_hash(x: Any) -> str:
    return hashlib.sha256(str(x).encode("utf-8")).hexdigest()


def md(title: str, obj: Any) -> str:
    return f"# {title}\n\n```json\n{json.dumps(obj, indent=2, ensure_ascii=False)}\n```\n"


def cand_id(c: Candidate) -> Tuple[int, int, int, int]:
    return int(c[1]), int(c[2]), int(c[3]), int(c[5])


def corr(x: np.ndarray, y: np.ndarray, method: str = "spearman") -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) < 2:
        return 0.0
    xx = x[mask].astype(np.float64)
    yy = y[mask].astype(np.float64)
    if float(np.std(xx)) == 0.0 or float(np.std(yy)) == 0.0:
        return 0.0
    if method == "spearman":
        return float(stats.spearmanr(xx, yy).correlation)
    return float(stats.kendalltau(xx, yy).correlation)


def quantiles(values: Sequence[float], qs: Sequence[float] = (0.1, 0.5, 0.9)) -> Dict[str, float]:
    a = np.asarray(list(values), dtype=np.float64)
    if len(a) == 0:
        return {f"p{int(q * 100)}": 0.0 for q in qs}
    return {f"p{int(q * 100)}": float(np.quantile(a, q)) for q in qs}


def normalize_residual(res: np.ndarray, mode: str, clip: float) -> np.ndarray:
    r = res.astype(np.float32)
    if mode == "none":
        x = r
    elif mode == "rank_norm":
        order = np.argsort(np.argsort(r)).astype(np.float32)
        x = (order / max(len(order) - 1, 1) - 0.5) * 2.0
    elif mode == "bounded_minmax":
        lo, hi = float(np.min(r)), float(np.max(r))
        x = np.zeros_like(r) if hi - lo < 1e-8 else ((r - lo) / (hi - lo) - 0.5) * 2.0
    else:
        raise ValueError(mode)
    return np.clip(x, -clip, clip).astype(np.float32)


def ranks_for(cands: Sequence[Candidate]) -> Dict[Tuple[int, int, int, int], int]:
    return {cand_id(c): i + 1 for i, c in enumerate(cands)}


def build_queries(args: argparse.Namespace) -> Dict[str, Any]:
    cache = load_npz(args.cache, allow_pickle=True)
    states = json.loads(Path(args.anchor_states).read_text(encoding="utf-8"))["states"]
    cfg = json.loads(Path(args.b2_eval).read_text(encoding="utf-8"))["best"]["config"]
    gscore = group_score_map(load_npz(args.video_scores, allow_pickle=False), cfg)
    gt_vid = gt_video_by_query(cache)
    gt_s, gt_e = gt_span_by_query(cache)
    queries = []
    all_margins = []
    for q, state in enumerate(states):
        desc_id = cache["desc_ids"][q].item() if hasattr(cache["desc_ids"][q], "item") else cache["desc_ids"][q]
        anchor = [normalize_cand(x) for x in state["flat_candidates"][:args.effective_top_n]]
        base = np.asarray([float(c[4]) for c in anchor], dtype=np.float32)
        residual = np.asarray([float(cfg["video_weight"]) * gscore.get(int(c[0]), (0.0, 0.0, 0.0, 0.0))[2] for c in anchor], dtype=np.float32)
        anchor_order = sorted([(c[0], c[1], c[2], c[3], float(s), c[5]) for c, s in zip(anchor, base)], key=lambda c: -c[4])
        b21_order = sorted([(c[0], c[1], c[2], c[3], float(s), c[5]) for c, s in zip(anchor, base + residual)], key=lambda c: -c[4])
        z = transform_residual(residual, "per_query_z", 0.5, 3.0)
        b3_order = sorted([(c[0], c[1], c[2], c[3], float(s), c[5]) for c, s in zip(anchor, base + z)], key=lambda c: -c[4])
        labels05 = labels_for_sequence(anchor, int(gt_vid[q]), int(gt_s[q]), int(gt_e[q]))[0]
        labels07 = labels_for_sequence(anchor, int(gt_vid[q]), int(gt_s[q]), int(gt_e[q]))[1]
        anchor_ranks = ranks_for(anchor_order)
        top_res_idx = int(np.argmax(residual)) if len(residual) else 0
        top_res_rank = int(anchor_ranks[cand_id(anchor[top_res_idx])])
        margin = float(base[0] - base[1]) if len(base) > 1 else 0.0
        all_margins.append(margin)
        queries.append({
            "q": q,
            "desc_id": desc_id,
            "anchor": anchor,
            "base": base,
            "residual": residual,
            "z": z,
            "anchor_order": anchor_order,
            "b21_order": b21_order,
            "b3_order": b3_order,
            "top_res_anchor_rank": top_res_rank,
            "anchor_top1_margin": margin,
            "positive_pool": bool(any(labels05) or any(labels07)),
            "anchor_hit_r100": bool(any(labels05[:100]) or any(labels07[:100])),
            "gt_vid": int(gt_vid[q]),
            "gt_s": int(gt_s[q]),
            "gt_e": int(gt_e[q]),
        })
    margin_high = float(np.quantile(np.asarray(all_margins, dtype=np.float32), 0.75))
    return {"queries": queries, "cache": cache, "margin_high": margin_high}


def make_sequence(anchor: List[Candidate], scores: np.ndarray, args: argparse.Namespace) -> List[Candidate]:
    ranked = [(c[0], c[1], c[2], c[3], float(s), c[5]) for c, s in zip(anchor, scores.astype(np.float32))]
    ranked = sorted(ranked, key=lambda c: -c[4])
    return nms_sequence(ranked[:args.effective_top_n], args.nms_thd, args.max_after_nms)


def enforce_anchor_safe(qd: Dict[str, Any], scores: np.ndarray) -> np.ndarray:
    out = scores.astype(np.float32).copy()
    ids = [cand_id(c) for c in qd["anchor"]]
    index = {k: i for i, k in enumerate(ids)}
    anchor_ids = [cand_id(c) for c in qd["anchor_order"]]
    for protected_ids, max_rank in [(anchor_ids[:1], 5), (anchor_ids[:5], 10)]:
        order = np.argsort(-out, kind="stable")
        ranked_ids = [ids[i] for i in order]
        threshold_score = float(out[order[min(max_rank - 1, len(order) - 1)]])
        bump = 1e-5
        for pid in protected_ids:
            cur_rank = ranked_ids.index(pid) + 1
            if cur_rank > max_rank:
                out[index[pid]] = threshold_score + bump
                bump += 1e-5
    return out


def candidate_scores(qd: Dict[str, Any], config: Dict[str, Any], margin_high: float) -> np.ndarray:
    res_safe = normalize_residual(qd["residual"], config["normalization"], float(config["clip"]))
    weight = float(config["alpha"])
    if qd["top_res_anchor_rank"] > 20:
        weight *= 0.1
    if qd["anchor_top1_margin"] >= margin_high:
        weight *= 0.1
    if qd["top_res_anchor_rank"] > 10:
        weight *= 0.1
    return enforce_anchor_safe(qd, qd["base"] + weight * res_safe)


def eval_records(queries: List[Dict[str, Any]], indices: Sequence[int], score_fn: Callable[[Dict[str, Any]], np.ndarray], args: argparse.Namespace, name: str) -> Dict[str, Any]:
    labels05: List[List[bool]] = []
    labels07: List[List[bool]] = []
    base05: List[List[bool]] = []
    base07: List[List[bool]] = []
    exits = entries = invalid = dup = 0
    fixed_pool = True
    all_final, all_anchor, all_res = [], [], []
    all_base_rank, all_new_rank = [], []
    top1chg = top5chg = top10chg = 0
    for q in indices:
        qd = queries[q]
        scores = score_fn(qd)
        seq = make_sequence(qd["anchor"], scores, args)
        base_seq = nms_sequence(qd["b21_order"][:args.effective_top_n], args.nms_thd, args.max_after_nms)
        l05, l07, pos, inv, du = labels_for_sequence(seq, qd["gt_vid"], qd["gt_s"], qd["gt_e"])
        a05, a07, apos, _ai, _ad = labels_for_sequence(base_seq, qd["gt_vid"], qd["gt_s"], qd["gt_e"])
        labels05.append(l05); labels07.append(l07); base05.append(a05); base07.append(a07)
        exits += int(apos and not pos); entries += int((not apos) and pos)
        invalid += inv; dup += du
        ranked = sorted([(c[0], c[1], c[2], c[3], float(s), c[5]) for c, s in zip(qd["anchor"], scores)], key=lambda c: -c[4])
        fixed_pool = fixed_pool and sorted(cand_id(c) for c in ranked) == sorted(cand_id(c) for c in qd["b21_order"])
        b21_ids = [cand_id(c) for c in qd["b21_order"]]
        new_ids = [cand_id(c) for c in ranked]
        top1chg += int(b21_ids[:1] != new_ids[:1])
        top5chg += int(set(b21_ids[:5]) != set(new_ids[:5]))
        top10chg += int(set(b21_ids[:10]) != set(new_ids[:10]))
        br = {k: i + 1 for i, k in enumerate(b21_ids)}
        nr = {k: i + 1 for i, k in enumerate(new_ids)}
        for k in b21_ids:
            all_base_rank.append(br[k]); all_new_rank.append(nr[k])
        all_final.append(scores.astype(np.float32))
        all_anchor.append(qd["base"].astype(np.float32))
        all_res.append(normalize_residual(qd["residual"], "bounded_minmax", 2.0))
    n = max(len(indices), 1)
    disp = np.abs(np.asarray(all_new_rank, dtype=np.float32) - np.asarray(all_base_rank, dtype=np.float32))
    final = np.concatenate(all_final); anchor = np.concatenate(all_anchor); res = np.concatenate(all_res)
    metrics = selected_metrics(labels05, labels07)
    gain_loss = {}
    for thr, cur, base in [("0.5", labels05, base05), ("0.7", labels07, base07)]:
        for r in [1, 5, 10]:
            gain = loss = 0
            for i in range(len(cur)):
                b = any(base[i][:r]); c = any(cur[i][:r])
                gain += int((not b) and c); loss += int(b and not c)
            gain_loss[f"{thr}-r{r}"] = {"gain_queries": int(gain), "loss_queries": int(loss)}
    return {
        "name": name,
        "query_count": int(len(indices)),
        "metrics": metrics,
        "movement": {
            "hard_positive_top100_query_exits": int(exits),
            "hard_positive_top100_query_entries": int(entries),
            "hard_positive_exit_ratio": float(exits / n),
            "invalid_span_count": int(invalid),
            "duplicate_span_count_after_nms": int(dup),
            "fixed_pool_invariant": bool(fixed_pool),
        },
        "gain_loss_vs_C7_B2_1_A4": gain_loss,
        "dominance": {
            "corr_final_anchor": corr(final, anchor),
            "corr_final_residual": corr(final, res),
            "top1_changed_rate": float(top1chg / n),
            "top5_set_changed_rate": float(top5chg / n),
            "top10_set_changed_rate": float(top10chg / n),
            "rank_displacement_p50": float(np.quantile(disp, 0.50)),
            "rank_displacement_p90": float(np.quantile(disp, 0.90)),
            "rank_displacement_p99": float(np.quantile(disp, 0.99)),
        },
    }


def choose_official_like_split(queries: List[Dict[str, Any]]) -> Dict[str, Any]:
    positives = [q for q, x in enumerate(queries) if x["anchor_hit_r100"]]
    negatives = [q for q, x in enumerate(queries) if not x["anchor_hit_r100"]]
    target_pos = len(negatives)
    def key(q: int) -> Tuple[int, str]:
        x = queries[q]
        sparse_rank = abs(x["top_res_anchor_rank"] - 34)
        return (sparse_rank, stable_hash(x["desc_id"]))
    pos_sel = sorted(positives, key=key)[:target_pos]
    neg_sel = sorted(negatives, key=key)
    idx = sorted(pos_sel + neg_sel, key=lambda q: stable_hash(queries[q]["desc_id"]))
    stats_payload = split_stats(queries, idx)
    return {"indices": idx, "stats": stats_payload}


def split_stats(queries: List[Dict[str, Any]], idx: Sequence[int]) -> Dict[str, Any]:
    vals = [queries[q] for q in idx]
    return {
        "query_count": int(len(idx)),
        "positive_pool_rate_05_or_07": float(np.mean([x["positive_pool"] for x in vals])) if vals else 0.0,
        "anchor_hit_rate_r100_05_or_07": float(np.mean([x["anchor_hit_r100"] for x in vals])) if vals else 0.0,
        "top_residual_candidate_anchor_rank": {
            **quantiles([x["top_res_anchor_rank"] for x in vals]),
            "rank1_rate": float(np.mean([x["top_res_anchor_rank"] == 1 for x in vals])) if vals else 0.0,
            "rank_gt10_rate": float(np.mean([x["top_res_anchor_rank"] > 10 for x in vals])) if vals else 0.0,
        },
        "split_hash": sha256_obj(list(idx)),
        "desc_id_hash": sha256_obj([str(queries[q]["desc_id"]) for q in idx]),
    }


def hard_pass(rec: Dict[str, Any]) -> bool:
    d = rec["dominance"]; m = rec["movement"]
    return (
        d["corr_final_anchor"] > d["corr_final_residual"]
        and d["top1_changed_rate"] <= 0.20
        and d["top5_set_changed_rate"] <= 0.35
        and d["top10_set_changed_rate"] <= 0.50
        and abs(rec["metrics"]["0.5-r100"] - rec["baseline_metrics"]["0.5-r100"]) < 1e-9
        and abs(rec["metrics"]["0.7-r100"] - rec["baseline_metrics"]["0.7-r100"]) < 1e-9
        and m["hard_positive_top100_query_exits"] == 0
        and m["invalid_span_count"] == 0
        and m["duplicate_span_count_after_nms"] == 0
    )


def load_sn_evidence(path: str | Path, needed: set[Tuple[str, int, int, int]]) -> Dict[Tuple[str, int, int, int], Dict[str, float]]:
    out: Dict[Tuple[str, int, int, int], Dict[str, float]] = {}
    if not Path(path).exists():
        return out
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            key = (str(row["desc_id"]), int(row["video_idx"]), int(row["start_idx"]), int(row["end_idx"]))
            if key in needed:
                out[key] = {
                    "b_start_logit": float(row.get("b_start_logit", 0.0)),
                    "e_end_logit": float(row.get("e_end_logit", 0.0)),
                    "l_logit": float(row.get("l_logit", row.get("b_start_logit", 0.0) + row.get("e_end_logit", 0.0))),
                    "r1": float(row.get("r1", 0.0)),
                }
                if len(out) == len(needed):
                    break
    return out


def main() -> None:
    args = parse_args()
    Path(args.audit_dir).mkdir(parents=True, exist_ok=True)
    data = build_queries(args)
    queries = data["queries"]
    full_idx = list(range(len(queries)))
    stress = choose_official_like_split(queries)
    stress_idx = stress["indices"]
    baseline_fn = lambda qd: qd["base"] + qd["residual"]
    c7b3_fn = lambda qd: qd["base"] + qd["z"]
    baseline_stress = eval_records(queries, stress_idx, baseline_fn, args, "C7_B2_1_A4_stress")
    baseline_full = eval_records(queries, full_idx, baseline_fn, args, "C7_B2_1_A4_train_calib")
    c7b3_stress = eval_records(queries, stress_idx, c7b3_fn, args, "C7_B3_calibrated_A4_stress")
    phase_a = {
        "status": "C7_B4_OFFICIAL_LIKE_STRESS_SPLIT_READY",
        "official_val_used": False,
        "construction": "deterministic train_calib subset balancing anchor-hit positives and negatives, sorted by sha256(desc_id)",
        "target_positive_pool_rate": "0.45-0.55",
        "target_anchor_hit_rate_r100": "0.45-0.55",
        **stress["stats"],
        "baseline_C7_B2_1_A4_metrics": baseline_stress["metrics"],
    }
    atomic_json(Path(args.audit_dir) / "C7_B4_OFFICIAL_LIKE_STRESS_SPLIT.json", phase_a)
    atomic_text(Path(args.audit_dir) / "C7_B4_OFFICIAL_LIKE_STRESS_SPLIT.md", md("C7-B4 official-like stress split", phase_a))

    configs = []
    for alpha in [0.02, 0.05, 0.1, 0.2]:
        for norm in ["none", "rank_norm", "bounded_minmax"]:
            for clip in [0.5, 1.0, 2.0]:
                configs.append({"alpha": alpha, "normalization": norm, "clip": clip})
    results = []
    for cfg in configs:
        fn = lambda qd, cfg=cfg: candidate_scores(qd, cfg, data["margin_high"])
        rec_s = eval_records(queries, stress_idx, fn, args, f"anchor_safe_{cfg}")
        rec_f = eval_records(queries, full_idx, fn, args, f"anchor_safe_full_{cfg}")
        rec_s["config"] = cfg
        rec_s["baseline_metrics"] = baseline_stress["metrics"]
        rec_s["delta_vs_C7_B2_1_A4_stress"] = metric_delta(rec_s["metrics"], baseline_stress["metrics"])
        rec_s["standard_train_calib"] = {
            "metrics": rec_f["metrics"],
            "delta_vs_C7_B2_1_A4": metric_delta(rec_f["metrics"], baseline_full["metrics"]),
            "movement": rec_f["movement"],
            "dominance": rec_f["dominance"],
        }
        rec_s["hard_constraints_pass"] = hard_pass(rec_s)
        results.append(rec_s)
    def select_key(r: Dict[str, Any]) -> Tuple[Any, ...]:
        d = r["delta_vs_C7_B2_1_A4_stress"]
        return (
            int(r["hard_constraints_pass"]),
            int(d["0.7-r1"] >= 0),
            int(d["0.7-r5"] >= 0),
            int(d["0.7-r10"] >= 0),
            d["0.7-r1"] + d["0.7-r5"] + d["0.7-r10"],
            -r["dominance"]["top5_set_changed_rate"],
        )
    best_anchor = sorted(results, key=select_key, reverse=True)[0]
    phase_b = {
        "status": "C7_B4_ANCHOR_PRESERVING_RESIDUAL_COMPLETE",
        "official_val_used": False,
        "forbidden_C7_B3_strong_config_used_as_primary": False,
        "guard_rules": {
            "residual_top_candidate_anchor_rank_gt20_weight_x": 0.1,
            "anchor_top1_margin_high_weight_x": 0.1,
            "top_residual_anchor_topK_disagreement_weight_x": 0.1,
            "anchor_top1_rank_max": 5,
            "anchor_top5_rank_max": 10,
            "anchor_top1_margin_high_threshold": data["margin_high"],
        },
        "baseline_stress": baseline_stress,
        "baseline_train_calib": baseline_full,
        "candidates": results,
        "best_anchor_preserving_candidate": best_anchor,
    }
    atomic_json(Path(args.audit_dir) / "C7_B4_ANCHOR_PRESERVING_RESIDUAL.json", phase_b)
    atomic_text(Path(args.audit_dir) / "C7_B4_ANCHOR_PRESERVING_RESIDUAL.md", md("C7-B4 anchor-preserving residual", phase_b))
    phase_c = {
        "status": "C7_B4_RESIDUAL_DOMINANCE_AUDIT_COMPLETE",
        "official_val_used": False,
        "hard_constraints": {
            "corr_final_anchor_gt_corr_final_residual": True,
            "top1_changed_rate_lte": 0.20,
            "top5_set_changed_rate_lte": 0.35,
            "top10_set_changed_rate_lte": 0.50,
            "R100_unchanged": True,
            "hard_exits": 0,
            "invalid_duplicate": 0,
        },
        "candidate_count": len(results),
        "hard_constraint_pass_count": int(sum(r["hard_constraints_pass"] for r in results)),
        "candidates": [
            {
                "config": r["config"],
                "metrics": r["metrics"],
                "delta_vs_C7_B2_1_A4_stress": r["delta_vs_C7_B2_1_A4_stress"],
                "movement": r["movement"],
                "dominance": r["dominance"],
                "hard_constraints_pass": r["hard_constraints_pass"],
            }
            for r in results
        ],
    }
    atomic_json(Path(args.audit_dir) / "C7_B4_RESIDUAL_DOMINANCE_AUDIT.json", phase_c)
    atomic_text(Path(args.audit_dir) / "C7_B4_RESIDUAL_DOMINANCE_AUDIT.md", md("C7-B4 residual dominance audit", phase_c))

    # Phase D: fixed-pool SN audit from raw row evidence when available.
    needed = {(str(qd["desc_id"]), int(c[1]), int(c[2]), int(c[3])) for qd in queries for c in qd["anchor"]}
    sn_map = load_sn_evidence(args.evidence_jsonl_gz, needed)
    sn_records: List[Dict[str, Any]] = []
    sn_status = "raw_logit_fixed_pool_available" if len(sn_map) == len(needed) else "raw_logit_partial_or_unavailable"
    if sn_map:
        def sn_score(qd: Dict[str, Any]) -> np.ndarray:
            vals = []
            for c in qd["anchor"]:
                row = sn_map.get((str(qd["desc_id"]), int(c[1]), int(c[2]), int(c[3])), {"l_logit": 0.0})
                vals.append(float(row["l_logit"]))
            v = normalize_residual(np.asarray(vals, dtype=np.float32), "bounded_minmax", 1.0)
            return enforce_anchor_safe(qd, qd["base"] + 0.05 * v)
        sn_rec = eval_records(queries, stress_idx, sn_score, args, "MINUTE_like_l_start_l_end_weak_fusion_stress")
        sn_rec["baseline_metrics"] = baseline_stress["metrics"]
        sn_rec["delta_vs_C7_B2_1_A4_stress"] = metric_delta(sn_rec["metrics"], baseline_stress["metrics"])
        sn_rec["hard_constraints_pass"] = hard_pass(sn_rec)
        sn_records.append(sn_rec)
    phase_d = {
        "status": "C7_B4_SN_AUDIT_COMPLETE",
        "official_val_used": False,
        "fixed_pool_only": True,
        "candidate_added": False,
        "NMS_modified": False,
        "evaluator_modified": False,
        "raw_start_end_logit_source": args.evidence_jsonl_gz,
        "raw_start_end_logit_status": sn_status,
        "needed_candidate_keys": int(len(needed)),
        "matched_candidate_keys": int(len(sn_map)),
        "compared_systems": {
            "C7_B2_1_A4": baseline_stress,
            "C7_B3_calibrated_A4": c7b3_stress,
            "anchor_preserving_residual_best": best_anchor,
            "MINUTE_like_raw_logit_scores": sn_records,
        },
    }
    atomic_json(Path(args.audit_dir) / "C7_B4_SN_AUDIT.json", phase_d)
    atomic_text(Path(args.audit_dir) / "C7_B4_SN_AUDIT.md", md("C7-B4 SN audit", phase_d))

    promising = [
        r for r in results + sn_records
        if r.get("hard_constraints_pass")
        and r["delta_vs_C7_B2_1_A4_stress"]["0.7-r1"] >= 0
        and r["delta_vs_C7_B2_1_A4_stress"]["0.7-r5"] >= 0
        and r["delta_vs_C7_B2_1_A4_stress"]["0.7-r10"] >= 0
        and min(r.get("standard_train_calib", {"delta_vs_C7_B2_1_A4": {"0.7-r1": 0, "0.7-r5": 0, "0.7-r10": 0}})["delta_vs_C7_B2_1_A4"][k] for k in ["0.7-r1", "0.7-r5", "0.7-r10"]) >= -0.05
    ]
    safe = [r for r in results + sn_records if r.get("hard_constraints_pass")]
    unsafe_gain = [
        r for r in results + sn_records
        if not r.get("hard_constraints_pass") and sum(r["delta_vs_C7_B2_1_A4_stress"][k] for k in ["0.7-r1", "0.7-r5", "0.7-r10"]) > 0
    ]
    if promising:
        status = "C7_B4_TRAIN_ONLY_PROMISING"
        selected = sorted(promising, key=select_key, reverse=True)[0]
    elif safe:
        status = "C7_B4_SAFE_BUT_NO_GAIN"
        selected = sorted(safe, key=select_key, reverse=True)[0]
    elif unsafe_gain:
        status = "C7_B4_HIGH_GAIN_UNSAFE"
        selected = sorted(unsafe_gain, key=select_key, reverse=True)[0]
    elif sn_status != "raw_logit_fixed_pool_available" and not safe:
        status = "C7_B4_SN_AUDIT_ONLY"
        selected = best_anchor
    else:
        status = "C7_B4_NEGATIVE"
        selected = best_anchor
    phase_e = {
        "status": status,
        "official_val_used": False,
        "second_official_val": False,
        "post_val_adjustment": False,
        "C7_B3_official_decision_modified": False,
        "C7_B2_1_promoted_decision_modified": False,
        "NMS_modified": False,
        "evaluator_modified": False,
        "CONQUER_QDF_QAL_original_ML_VR_frozen": True,
        "full_fine_tuning": False,
        "primary_baseline": "C7-B2.1 A4 train-only reconstruction",
        "selection_rule": [
            "official-like stress split 0.7-r1/r5/r10 >= C7-B2.1 A4",
            "standard train_calib not materially below C7-B2.1 A4",
            "R@100 unchanged; hard exits=0; invalid/duplicate=0",
            "corr(final, anchor) > corr(final, residual); topK change rates controlled",
        ],
        "selected_candidate": selected,
        "recommendation": "Stop here. Do not run official val automatically.",
    }
    atomic_json(Path(args.audit_dir) / "C7_B4_SELECTION_DECISION.json", phase_e)
    atomic_text(Path(args.audit_dir) / "C7_B4_SELECTION_DECISION.md", md("C7-B4 selection decision", phase_e))
    print(json.dumps({
        "status": status,
        "stress_split": phase_a,
        "best_anchor_config": best_anchor.get("config"),
        "best_anchor_delta_0.7": {k: best_anchor["delta_vs_C7_B2_1_A4_stress"][k] for k in ["0.7-r1", "0.7-r5", "0.7-r10"]},
        "best_anchor_dominance": best_anchor["dominance"],
        "sn_status": sn_status,
        "official_val_used": False,
        "post_val_adjustment": False,
        "second_official_val": False,
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
