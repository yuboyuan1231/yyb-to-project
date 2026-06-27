#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rlem_c6_b0.run_c6_b0_candidate_diagnostic import correctness_for_ts, load_gt_ts, span_iou_idx  # noqa: E402
from rlem_c6_b1_lite.run_c6_b1_r1_safe_selector import evaluate_policy  # noqa: E402
from rlem_c6_c.run_c6_c_goal import load_baseline_configs, load_npz, make_anchor_candidates  # noqa: E402


METRIC_KEYS = [
    "0.5-r1", "0.5-r5", "0.5-r10", "0.5-r100",
    "0.7-r1", "0.7-r5", "0.7-r10", "0.7-r100",
]


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


def metric_delta(new: Dict[str, float], base: Dict[str, float]) -> Dict[str, float]:
    return {k: float(new[k] - base[k]) for k in METRIC_KEYS}


def selected_metrics(labels05: List[List[bool]], labels07: List[List[bool]]) -> Dict[str, float]:
    out = {k: 0 for k in METRIC_KEYS}
    q_count = len(labels05)
    for q in range(q_count):
        for thr, labs in [("0.5", labels05[q]), ("0.7", labels07[q])]:
            for k in [1, 5, 10, 100]:
                out[f"{thr}-r{k}"] += int(any(labs[:k]))
    return {k: 100.0 * v / max(q_count, 1) for k, v in out.items()}


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


def nms_sequence(seq: List[Tuple[int, int, int, int, float, int]], nms_thd: float, max_keep: int) -> List[Tuple[int, int, int, int, float, int]]:
    kept: List[Tuple[int, int, int, int, float, int]] = []
    by_video: Dict[int, List[int]] = {}
    for cand in seq:
        _gid, vid, si, ei, _score, _row = cand
        suppress = False
        for prior_idx in by_video.get(vid, []):
            prior = kept[prior_idx]
            if span_iou_idx(si, ei, prior[2], prior[3]) > nms_thd:
                suppress = True
                break
        if not suppress:
            by_video.setdefault(vid, []).append(len(kept))
            kept.append(cand)
            if len(kept) >= max_keep:
                break
    return kept


def labels_for_sequence(
    seq: List[Tuple[int, int, int, int, float, int]],
    gt_vid: int,
    gt_ts: Any,
) -> Tuple[List[bool], List[bool], int, int, int, int]:
    labs05: List[bool] = []
    labs07: List[bool] = []
    best_video_rank = 10**9
    best_span05 = 10**9
    best_span07 = 10**9
    for i, (_gid, vid, si, ei, _score, _row) in enumerate(seq, start=1):
        same_video = int(vid) == int(gt_vid)
        if same_video:
            best_video_rank = min(best_video_rank, i)
        ok05 = bool(same_video and correctness_for_ts(si, ei, gt_ts, 0.5))
        ok07 = bool(same_video and correctness_for_ts(si, ei, gt_ts, 0.7))
        if ok05:
            best_span05 = min(best_span05, i)
        if ok07:
            best_span07 = min(best_span07, i)
        labs05.append(ok05)
        labs07.append(ok07)
    return labs05, labs07, best_video_rank, best_span05, best_span07, len(seq)


def hit(labs: List[bool], k: int) -> bool:
    return bool(any(labs[:k]))


def make_score_arrays(score_npz: str) -> Dict[str, np.ndarray]:
    z = load_npz(score_npz, allow_pickle=False)
    max_gid = int(z["group_id"].max())
    arr = {k: np.zeros(max_gid + 1, dtype=np.float32) for k in ["relevance", "moment", "iou", "risk", "gate", "base_rank"]}
    gids = z["group_id"].astype(np.int64)
    for k in ["relevance", "moment", "iou", "risk", "gate", "base_rank"]:
        arr[k][gids] = z[k].astype(np.float32)
    return arr


def c6c_group_score(score_arr: Dict[str, np.ndarray], gid: int, cfg: Dict[str, Any]) -> Tuple[float, float]:
    gate = float(score_arr["gate"][gid])
    base = 1.0 / float(int(score_arr["base_rank"][gid]) + 1)
    residual = (
        float(cfg["beta"]) * gate * float(score_arr["relevance"][gid])
        + float(cfg["gamma"]) * gate * (float(score_arr["moment"][gid]) + 0.35 * float(score_arr["iou"][gid]))
        - float(cfg["rho"]) * gate * float(score_arr["risk"][gid])
    )
    return float(float(cfg["alpha"]) * base + residual), gate


def build_query_records(args: argparse.Namespace) -> Dict[str, Any]:
    cache = load_npz(args.calib_cache, allow_pickle=True)
    pool = load_npz(args.calib_pool, allow_pickle=False)
    b1_cfg, b2_cfg = load_baseline_configs(args)
    b1_scores = np.load(args.b1_calib_scores, allow_pickle=False)["score"].astype(np.float32)
    b2_scores = np.load(args.b2_calib_scores, allow_pickle=False)["score"].astype(np.float32)
    gt_ts = load_gt_ts(args.gt_jsonl, cache["desc_ids"])
    gt_vid = gt_video_by_query(cache)
    score_arr = make_score_arrays(args.c6c_score_npz)
    q_count = len(cache["desc_ids"])
    records = []
    for q in range(q_count):
        b1 = make_anchor_candidates(q=q, cache=cache, pool=pool, pool_score=b1_scores, policy_cfg=b1_cfg, effective_top_n=args.effective_top_n)
        b2 = make_anchor_candidates(q=q, cache=cache, pool=pool, pool_score=b2_scores, policy_cfg=b2_cfg, effective_top_n=args.effective_top_n)
        b1_seq = nms_sequence(b1, args.nms_thd, args.max_after_nms)
        b2_seq = nms_sequence(b2, args.nms_thd, args.max_after_nms)
        b05, b07, bv_rank, bs05, bs07, _ = labels_for_sequence(b1_seq, int(gt_vid[q]), gt_ts[q])
        records.append({
            "b1": b1,
            "b2": b2,
            "b1_seq": b1_seq,
            "b2_seq": b2_seq,
            "base_labels05": b05,
            "base_labels07": b07,
            "base_video_rank": None if bv_rank >= 10**9 else bv_rank,
            "base_span05_rank": None if bs05 >= 10**9 else bs05,
            "base_span07_rank": None if bs07 >= 10**9 else bs07,
            "gt_ts": gt_ts[q],
            "gt_vid": int(gt_vid[q]),
        })
    b1_eval = evaluate_policy(
        cache=cache, pool=pool, pool_score=b1_scores, gt_ts_by_query=gt_ts,
        apply_slots=int(b1_cfg["apply_slots"]), threshold0=float(b1_cfg["threshold0"]),
        threshold_rest=float(b1_cfg["threshold_rest"]), max_replacements=int(b1_cfg["max_replacements"]),
        effective_top_n=args.effective_top_n, max_after_nms=args.max_after_nms, nms_thd=args.nms_thd,
    )
    b2_eval = evaluate_policy(
        cache=cache, pool=pool, pool_score=b2_scores, gt_ts_by_query=gt_ts,
        apply_slots=int(b2_cfg["apply_slots"]), threshold0=float(b2_cfg["threshold0"]),
        threshold_rest=float(b2_cfg["threshold_rest"]), max_replacements=int(b2_cfg["max_replacements"]),
        effective_top_n=args.effective_top_n, max_after_nms=args.max_after_nms, nms_thd=args.nms_thd,
    )
    return {"records": records, "score_arr": score_arr, "b1_eval": b1_eval, "b2_eval": b2_eval, "cache": cache}


def compose_sequence(rec: Dict[str, Any], score_arr: Dict[str, np.ndarray], cfg: Dict[str, Any]) -> Tuple[List[Tuple[int, int, int, int, float, int]], Dict[str, Any]]:
    # B2 is same-video/span utility evidence only.  Both sequences are
    # pre-NMSed once, then C6-C1.1 only moves elements within a topK window.
    base = list(rec["b2_seq"] if cfg.get("use_b2_span", False) else rec["b1_seq"])

    window = int(cfg["window"])
    promote_to = int(cfg.get("promote_to", 0))
    base_top = base[0]
    base_score, _base_gate = c6c_group_score(score_arr, base_top[0], cfg)
    best_i = 0
    best_score = base_score
    best_gate = _base_gate
    for i in range(1, min(window, len(base))):
        s, g = c6c_group_score(score_arr, base[i][0], cfg)
        if g >= float(cfg["gate_threshold"]) and s > best_score:
            best_i = i
            best_score = s
            best_gate = g
    enabled = bool(best_i > 0 and best_score - base_score >= float(cfg["min_margin"]))
    seq = list(base)
    if enabled:
        cand = seq.pop(best_i)
        seq.insert(promote_to, cand)
    return seq[: int(cfg["max_after_nms"])], {
        "enabled": enabled,
        "promoted_from_rank": int(best_i + 1) if enabled else None,
        "promoted_to_rank": int(promote_to + 1) if enabled else None,
        "best_gate": float(best_gate),
        "score_margin": float(best_score - base_score),
    }


def evaluate_config(records: List[Dict[str, Any]], score_arr: Dict[str, np.ndarray], cfg: Dict[str, Any]) -> Dict[str, Any]:
    labels05: List[List[bool]] = []
    labels07: List[List[bool]] = []
    base05: List[List[bool]] = []
    base07: List[List[bool]] = []
    qdiag = []
    enabled_count = 0
    for q, rec in enumerate(records):
        seq, move = compose_sequence(rec, score_arr, cfg)
        b05, b07 = rec["base_labels05"], rec["base_labels07"]
        bv_rank = rec["base_video_rank"] if rec["base_video_rank"] is not None else 10**9
        bs05 = rec["base_span05_rank"] if rec["base_span05_rank"] is not None else 10**9
        bs07 = rec["base_span07_rank"] if rec["base_span07_rank"] is not None else 10**9
        a05, a07, av_rank, as05, as07, _ = labels_for_sequence(seq, rec["gt_vid"], rec["gt_ts"])
        base05.append(b05); base07.append(b07); labels05.append(a05); labels07.append(a07)
        enabled_count += int(move["enabled"])
        qdiag.append({
            "query_index": q,
            "enabled": move["enabled"],
            "top1_gain_07": (not hit(b07, 1)) and hit(a07, 1),
            "top1_loss_07": hit(b07, 1) and (not hit(a07, 1)),
            "top5_gain_07": (not hit(b07, 5)) and hit(a07, 5),
            "top5_loss_07": hit(b07, 5) and (not hit(a07, 5)),
            "top10_gain_07": (not hit(b07, 10)) and hit(a07, 10),
            "top10_loss_07": hit(b07, 10) and (not hit(a07, 10)),
            "lost_R5_but_gain_R1_07": hit(b07, 5) and (not hit(a07, 5)) and (not hit(b07, 1)) and hit(a07, 1),
            "lost_R10_but_gain_R1_07": hit(b07, 10) and (not hit(a07, 10)) and (not hit(b07, 1)) and hit(a07, 1),
            "positive_video_best_rank_before": None if bv_rank >= 10**9 else bv_rank,
            "positive_video_best_rank_after": None if av_rank >= 10**9 else av_rank,
            "positive_span05_best_rank_before": None if bs05 >= 10**9 else bs05,
            "positive_span05_best_rank_after": None if as05 >= 10**9 else as05,
            "positive_span07_best_rank_before": None if bs07 >= 10**9 else bs07,
            "positive_span07_best_rank_after": None if as07 >= 10**9 else as07,
            **move,
        })
    metrics = selected_metrics(labels05, labels07)
    base_metrics = selected_metrics(base05, base07)
    video_before = {
        "GT_video_R@1": 100.0 * sum(d["positive_video_best_rank_before"] is not None and d["positive_video_best_rank_before"] <= 1 for d in qdiag) / len(qdiag),
        "GT_video_R@5": 100.0 * sum(d["positive_video_best_rank_before"] is not None and d["positive_video_best_rank_before"] <= 5 for d in qdiag) / len(qdiag),
        "GT_video_R@10": 100.0 * sum(d["positive_video_best_rank_before"] is not None and d["positive_video_best_rank_before"] <= 10 for d in qdiag) / len(qdiag),
    }
    video_after = {
        "GT_video_R@1": 100.0 * sum(d["positive_video_best_rank_after"] is not None and d["positive_video_best_rank_after"] <= 1 for d in qdiag) / len(qdiag),
        "GT_video_R@5": 100.0 * sum(d["positive_video_best_rank_after"] is not None and d["positive_video_best_rank_after"] <= 5 for d in qdiag) / len(qdiag),
        "GT_video_R@10": 100.0 * sum(d["positive_video_best_rank_after"] is not None and d["positive_video_best_rank_after"] <= 10 for d in qdiag) / len(qdiag),
    }
    return {
        "config": cfg,
        "metrics": metrics,
        "base_metrics_reconstructed": base_metrics,
        "delta_vs_reconstructed_B1": metric_delta(metrics, base_metrics),
        "query_counts": {
            "enabled_queries": int(enabled_count),
            "top1_gain_queries": int(sum(d["top1_gain_07"] for d in qdiag)),
            "top1_loss_queries": int(sum(d["top1_loss_07"] for d in qdiag)),
            "top5_gain_queries": int(sum(d["top5_gain_07"] for d in qdiag)),
            "top5_loss_queries": int(sum(d["top5_loss_07"] for d in qdiag)),
            "top10_gain_queries": int(sum(d["top10_gain_07"] for d in qdiag)),
            "top10_loss_queries": int(sum(d["top10_loss_07"] for d in qdiag)),
            "lost_R5_but_gain_R1_queries": int(sum(d["lost_R5_but_gain_R1_07"] for d in qdiag)),
            "lost_R10_but_gain_R1_queries": int(sum(d["lost_R10_but_gain_R1_07"] for d in qdiag)),
        },
        "video_metrics_before": video_before,
        "video_metrics_after": video_after,
        "query_diagnostics": qdiag,
        "official_val_used": False,
    }


def score_for_category(rec: Dict[str, Any], official_b1_metrics: Dict[str, float], category: str) -> float:
    d = metric_delta(rec["metrics"], official_b1_metrics)
    if category == "conservative":
        r5_pen, r10_pen, r1_w = 8.0, 4.0, 3.0
    elif category == "balanced":
        r5_pen, r10_pen, r1_w = 5.0, 2.5, 4.0
    else:
        r5_pen, r10_pen, r1_w = 1.0, 0.5, 6.0
    return float(
        r1_w * d["0.7-r1"]
        + 2.0 * d["0.5-r1"]
        - r5_pen * max(0.0, -d["0.7-r5"] - 0.10)
        - r5_pen * max(0.0, -d["0.5-r5"] - 0.10)
        - r10_pen * max(0.0, -d["0.7-r10"] - 0.15)
        - r10_pen * max(0.0, -d["0.5-r10"] - 0.15)
    )


def search(records: List[Dict[str, Any]], score_arr: Dict[str, np.ndarray], b1_metrics: Dict[str, float], args: argparse.Namespace) -> Dict[str, Any]:
    gates = np.asarray(score_arr["gate"], dtype=np.float32)
    gate_grid = [float(np.quantile(gates, q)) for q in [0.85, 0.93, 0.97, 0.99]]
    categories = {
        "conservative": {"windows": [3, 5], "margins": [0.00, 0.03, 0.06], "use_b2": [False, True], "profile": "R5/R10-preserving"},
        "balanced": {"windows": [5], "margins": [0.00, 0.03, 0.06], "use_b2": [False, True], "profile": "top5-set-preserving"},
        "R1-only": {"windows": [10], "margins": [-0.02, 0.00, 0.03], "use_b2": [True], "profile": "tradeoff-allowed"},
    }
    out: Dict[str, Any] = {"categories": {}, "all_results": []}
    config_id = 0
    for cat, spec in categories.items():
        results = []
        for window in spec["windows"]:
            for margin in spec["margins"]:
                for gate in gate_grid:
                    for use_b2 in spec["use_b2"]:
                        cfg = {
                            "config_id": f"c6c11_{config_id:04d}",
                            "category": cat,
                            "window": int(window),
                            "promote_to": 0,
                            "gate_threshold": float(gate),
                            "min_margin": float(margin),
                            "use_b2_span": bool(use_b2),
                            "alpha": 1.0,
                            "beta": 0.20,
                            "gamma": 0.20,
                            "rho": 0.05,
                            "nms_thd": args.nms_thd,
                            "max_after_nms": args.max_after_nms,
                        }
                        config_id += 1
                        rec = evaluate_config(records, score_arr, cfg)
                        rec["delta_vs_C6_B1_lite"] = metric_delta(rec["metrics"], b1_metrics)
                        rec["selection_score"] = score_for_category(rec, b1_metrics, cat)
                        d = rec["delta_vs_C6_B1_lite"]
                        rec["feasible"] = bool(
                            d["0.7-r1"] >= 0.20
                            and d["0.5-r1"] >= 0.0
                            and d["0.7-r5"] >= -0.10
                            and d["0.5-r5"] >= -0.10
                            and d["0.7-r10"] >= -0.15
                            and d["0.5-r10"] >= -0.15
                        )
                        rec["tradeoff"] = bool(d["0.7-r5"] < -0.10 or d["0.7-r10"] < -0.15 or d["0.5-r5"] < -0.10 or d["0.5-r10"] < -0.15)
                        results.append(rec)
                        out["all_results"].append({k: v for k, v in rec.items() if k != "query_diagnostics"})
        best = max(results, key=lambda r: (r["feasible"], r["selection_score"], r["delta_vs_C6_B1_lite"]["0.7-r1"]))
        out["categories"][cat] = {
            "profile": spec["profile"],
            "best": {k: v for k, v in best.items() if k != "query_diagnostics"},
            "best_query_diagnostics": best["query_diagnostics"],
            "feasible_count": int(sum(r["feasible"] for r in results)),
            "result_count": len(results),
        }
    return out


def write_outputs(result: Dict[str, Any], context: Dict[str, Any], args: argparse.Namespace) -> None:
    audit = Path(args.audit_dir)
    audit.mkdir(parents=True, exist_ok=True)
    b1 = context["b1_eval"]["metrics"]
    b2 = context["b2_eval"]["metrics"]
    for cat, payload in result["categories"].items():
        payload["best"]["delta_vs_C6_B2"] = metric_delta(payload["best"]["metrics"], b2)
    balanced = result["categories"]["balanced"]["best"]
    no_new_c6c_video_gain = bool(
        balanced["query_counts"]["enabled_queries"] == 0
        and abs(balanced["delta_vs_C6_B2"]["0.7-r1"]) < 1e-9
        and abs(balanced["delta_vs_C6_B2"]["0.5-r1"]) < 1e-9
    )
    promoted = bool(
        balanced["feasible"]
        and not no_new_c6c_video_gain
        and balanced["query_counts"]["enabled_queries"] > 0
        and balanced["delta_vs_C6_B2"]["0.7-r1"] >= 0.0
        and balanced["delta_vs_C6_B2"]["0.5-r1"] >= 0.0
    )
    if promoted:
        status = "C6_C11_FREEZE_CANDIDATE_READY"
    elif no_new_c6c_video_gain:
        status = "C6_C11_SAFE_B2_EQUIVALENT_NO_PROMOTION"
    else:
        status = "C6_C11_NO_PROMOTION" if result["categories"]["balanced"]["feasible_count"] > 0 else "C6_C11_NO_FEASIBLE"

    manifest = {
        "status": status,
        "official_val_used": False,
        "official_val_authorized": False,
        "primary_safety_anchor": "C6-B1-lite c6b1r1_0057",
        "current_promoted_report_anchor": "C6-B2 pairwise_main_0005",
        "b2_role": "span utility/tie-breaker only",
        "baselines": {"C6_B1_lite": context["b1_eval"], "C6_B2": context["b2_eval"]},
        "category_best": {cat: p["best"] for cat, p in result["categories"].items()},
        "promoted": promoted,
        "no_new_c6c_video_gain": no_new_c6c_video_gain,
        "decision": "Do not enter official val unless status is C6_C11_FREEZE_CANDIDATE_READY and human approval is given.",
    }
    atomic_json(audit / "C6_C11_CONFIG_COMPARISON.json", result["all_results"])
    atomic_json(audit / "C6_C11_FINAL_DECISION.json", manifest)

    # Query manifest: keep compact but actionable.
    manifest_queries: Dict[str, Any] = {}
    for cat, payload in result["categories"].items():
        qd = payload["best_query_diagnostics"]
        manifest_queries[cat] = {
            "lost_r5_queries": [d for d in qd if d["top5_loss_07"]][:500],
            "lost_r10_queries": [d for d in qd if d["top10_loss_07"]][:500],
            "lost_R5_but_gain_R1": [d for d in qd if d["lost_R5_but_gain_R1_07"]][:500],
            "lost_R10_but_gain_R1": [d for d in qd if d["lost_R10_but_gain_R1_07"]][:500],
            "top1_gain_queries": [d for d in qd if d["top1_gain_07"]][:500],
        }
    atomic_json(audit / "c6_c11_lost_r5_r10_manifest.json", manifest_queries)

    diagnostic = {
        "status": "C6_C11_DIAGNOSTICS_COMPLETE",
        "official_val_used": False,
        "query_level_counts": {cat: p["best"]["query_counts"] for cat, p in result["categories"].items()},
        "positive_video_rank_before_after": {cat: {
            "before": p["best"]["video_metrics_before"],
            "after": p["best"]["video_metrics_after"],
        } for cat, p in result["categories"].items()},
        "proposal_confidence_oracle_note": {
            "observation": "C6-C0 proposal_confidence oracle has identical 0.7 R@1/R@5/R@10/R@100.",
            "interpretation": "This is caused by the oracle definition: it ranks with GT-derived y07/y05 proposal labels, so queries with an available 0.7-positive proposal place it at rank 1, while queries without one have no later positive. It is a single-best oracle diagnostic, not evidence of evaluator/NMS correctness and not eligible for promotion decisions.",
            "evaluator_bug_suspected": False,
            "promotion_allowed_from_oracle": False,
        },
    }
    atomic_json(audit / "C6_C11_DIAGNOSTIC_AUDIT.json", diagnostic)

    md_diag = "# C6-C1.1 diagnostic audit\n\n"
    md_diag += f"- Status: `{diagnostic['status']}`\n- Official val used: `false`\n"
    md_diag += "- C6-C0 proposal_confidence identical 0.7 recalls are an oracle-definition artifact, not a promotion signal.\n\n"
    md_diag += "## Query Counts\n\n```json\n" + json.dumps(diagnostic["query_level_counts"], indent=2, ensure_ascii=False) + "\n```\n"
    atomic_text(audit / "C6_C11_DIAGNOSTIC_AUDIT.md", md_diag)

    md_search = "# C6-C1.1 topK-preserving search\n\n"
    md_search += "- Scope: `train_calib only`\n- Official val used: `false`\n- B1 topK tuple/window preservation is enforced by construction.\n\n"
    for cat, payload in result["categories"].items():
        md_search += f"## {cat}\n\n```json\n" + json.dumps(payload["best"], indent=2, ensure_ascii=False) + "\n```\n"
    atomic_text(audit / "C6_C11_TOPK_PRESERVING_SEARCH.md", md_search)

    md_cmp = "# C6-C1.1 config comparison\n\n"
    md_cmp += f"- Status: `{status}`\n- Promoted: `{str(promoted).lower()}`\n\n"
    md_cmp += "```json\n" + json.dumps(manifest["category_best"], indent=2, ensure_ascii=False) + "\n```\n"
    atomic_text(audit / "C6_C11_CONFIG_COMPARISON.md", md_cmp)

    md_final = "# C6-C1.1 final decision\n\n"
    md_final += f"- Status: `{status}`\n- Promoted: `{str(promoted).lower()}`\n- Official val used: `false`\n"
    md_final += f"- No new C6-C video gain: `{str(no_new_c6c_video_gain).lower()}`\n\n"
    md_final += "C6-C1.1 stops here unless a freeze candidate is explicitly approved for a one-shot official-val run.\n"
    atomic_text(audit / "C6_C11_FINAL_DECISION.md", md_final)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--calib_cache", default="results/rlem_c6a/cache/train_calib_c6_cache.npz")
    p.add_argument("--calib_pool", default="results/rlem_c6_b1_lite_r1_safe/train_calib_candidate_pool_top_slots.npz")
    p.add_argument("--gt_jsonl", default="results/rlem_c3_minimal/train_calib_gt.jsonl")
    p.add_argument("--b1_calib_scores", default="results/rlem_c6_b1_lite_r1_safe/train_calib_candidate_scores.npz")
    p.add_argument("--b1_best_config", default="results/rlem_c6_b1_lite_r1_safe/best_config.json")
    p.add_argument("--b2_calib_scores", default="results/rlem_c6_b2/train_calib_scores/train_calib_pairwise_main_scores.npz")
    p.add_argument("--b2_best_config", default="results/rlem_c6_b2/train_calib_scores/pairwise_main/best_config.json")
    p.add_argument("--c6c_score_npz", default="results/rlem_c6_c/train_calib_video_scores.npz")
    p.add_argument("--audit_dir", default="c6_c_audit")
    p.add_argument("--effective_top_n", type=int, default=100)
    p.add_argument("--max_after_nms", type=int, default=100)
    p.add_argument("--nms_thd", type=float, default=0.7)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    context = build_query_records(args)
    result = search(context["records"], context["score_arr"], context["b1_eval"]["metrics"], args)
    write_outputs(result, context, args)
    balanced = result["categories"]["balanced"]["best"]
    print(json.dumps({
        "status": json.loads(Path(args.audit_dir, "C6_C11_FINAL_DECISION.json").read_text())["status"],
        "balanced_best": {
            "config": balanced["config"],
            "metrics": balanced["metrics"],
            "delta_vs_C6_B1_lite": balanced["delta_vs_C6_B1_lite"],
            "feasible": balanced["feasible"],
            "tradeoff": balanced["tradeoff"],
        },
        "official_val_used": False,
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
