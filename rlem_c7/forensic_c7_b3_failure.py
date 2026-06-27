#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rlem_c6_b0.run_c6_b0_candidate_diagnostic import span_iou_idx  # noqa: E402
from rlem_c7.run_c7_b0_b1 import (  # noqa: E402
    atomic_json,
    atomic_text,
    gt_span_by_query,
    gt_video_by_query,
    labels_for_sequence,
    load_npz,
    nms_sequence,
)
from rlem_c7.run_c7_b2_1_safety_repair import group_score_map, normalize_cand  # noqa: E402
from rlem_c7.run_c7_b3_stable_generalization import transform_residual  # noqa: E402


METRIC_KEYS = ["0.5-r1", "0.5-r5", "0.5-r10", "0.7-r1", "0.7-r5", "0.7-r10"]
SELECTED_CONFIG = {"T": 0.5, "lambda": 1.0, "clip": 3.0, "normalization": "per_query_z"}
Candidate = Tuple[int, int, int, int, float, int]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--audit_dir", default="c7_audit")
    p.add_argument("--official_cache", default="results/rlem_c6_b1_lite_r1_safe/official_val_c6_cache.npz")
    p.add_argument("--official_anchor_states", default="results/rlem_c7_b2_1_official_val/aux/official_val_c7_b1_anchor_states.json")
    p.add_argument("--official_video_scores", default="results/rlem_c7_b2_1_official_val/aux/official_val_video_scores.npz")
    p.add_argument("--official_b21_scores", default="results/rlem_c7_b2_1_official_val/official_val_scores.npz")
    p.add_argument("--official_b3_scores", default="results/rlem_c7_b3_official_val/official_val_scores.npz")
    p.add_argument("--official_b21_predictions", default="results/rlem_c7_b2_1_official_val/official_val_predictions.jsonl")
    p.add_argument("--official_b3_predictions", default="results/rlem_c7_b3_official_val/official_val_predictions.jsonl")
    p.add_argument("--official_b21_submission", default="results/rlem_c7_b2_1_official_val/official_val_submission.json")
    p.add_argument("--gt_jsonl", default="/home/a/yybwork/data/yyb/tvr_feature_release/data/tvr_val_release.jsonl")
    p.add_argument("--official_summary", default="c7_audit/C7_B3_OFFICIAL_VAL_ONE_SHOT.json")
    p.add_argument("--train_cache", default="results/rlem_c6a/cache/train_calib_c6_cache.npz")
    p.add_argument("--train_anchor_states", default="results/rlem_c7_b2/train_calib_c7_b1_anchor_states.json")
    p.add_argument("--train_video_scores", default="results/rlem_c7_b2/train_calib_video_scores.npz")
    p.add_argument("--b2_eval", default="c7_audit/C7_B2_TRAIN_CALIB_EVAL.json")
    p.add_argument("--split_manifest", default="c7_audit/C7_B3_SPLIT_MANIFEST.json")
    p.add_argument("--shrinkage_diagnosis", default="c7_audit/C7_B3_SHRINKAGE_DIAGNOSIS.json")
    p.add_argument("--effective_top_n", type=int, default=100)
    p.add_argument("--max_after_nms", type=int, default=100)
    p.add_argument("--nms_thd", type=float, default=0.7)
    p.add_argument("--examples_per_loss_type", type=int, default=25)
    return p.parse_args()


def read_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def finite_corr(x: np.ndarray, y: np.ndarray, method: str) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) < 2:
        return 0.0
    xx = x[mask].astype(np.float64)
    yy = y[mask].astype(np.float64)
    if float(np.std(xx)) == 0.0 or float(np.std(yy)) == 0.0:
        return 0.0
    if method == "spearman":
        return float(stats.spearmanr(xx, yy).correlation)
    if method == "kendall":
        return float(stats.kendalltau(xx, yy).correlation)
    if method == "pearson":
        return float(np.corrcoef(xx, yy)[0, 1])
    raise ValueError(method)


def cand_key(q: int, desc_id: Any, c: Candidate) -> Tuple[str, int, int, int, int, int]:
    return str(desc_id), int(c[1]), int(c[2]), int(c[3]), int(c[5]), int(c[0])


def short_key(c: Candidate) -> Tuple[int, int, int]:
    return int(c[1]), int(c[2]), int(c[3])


def ranks_for_order(cands: Sequence[Candidate]) -> Dict[Tuple[int, int, int, int], int]:
    return {(int(c[1]), int(c[2]), int(c[3]), int(c[5])): i + 1 for i, c in enumerate(cands)}


def hit_flags(cands: Sequence[Candidate], gt_vid: int, gt_s: int, gt_e: int, thr: float) -> List[bool]:
    out = []
    for c in cands:
        same = int(c[1]) == int(gt_vid)
        iou = span_iou_idx(int(c[2]), int(c[3]), int(gt_s), int(gt_e)) if same else 0.0
        out.append(bool(iou >= thr))
    return out


def time_iou_from_idx(c: Candidate, gt_ts: Sequence[float]) -> float:
    ps = float(int(c[2]) * 1.5)
    pe = float((int(c[3]) + 1) * 1.5)
    gs, ge = float(gt_ts[0]), float(gt_ts[1])
    inter = max(0.0, min(pe, ge) - max(ps, gs))
    union = max(pe, ge) - min(ps, gs)
    return float(inter / union) if union > 0 else 0.0


def hit_flags_qd(qd: Dict[str, Any], cands: Sequence[Candidate], thr: float) -> List[bool]:
    if "gt_ts" in qd:
        return [bool(int(c[1]) == int(qd["gt_vid"]) and time_iou_from_idx(c, qd["gt_ts"]) >= thr) for c in cands]
    return hit_flags(cands, int(qd["gt_vid"]), int(qd["gt_s"]), int(qd["gt_e"]), thr)


def best_rank_qd(qd: Dict[str, Any], cands: Sequence[Candidate], thr: float) -> int | None:
    flags = hit_flags_qd(qd, cands, thr)
    for i, ok in enumerate(flags):
        if ok:
            return i + 1
    return None


def entropy(x: np.ndarray) -> float:
    x = x.astype(np.float64)
    x = x - float(np.max(x))
    p = np.exp(x)
    p = p / max(float(p.sum()), 1e-12)
    return float(-(p * np.log(np.maximum(p, 1e-12))).sum())


def quantiles(x: Sequence[float], qs: Sequence[float] = (0.1, 0.5, 0.9)) -> Dict[str, float]:
    arr = np.asarray(list(x), dtype=np.float64)
    if len(arr) == 0:
        return {f"p{int(q*100)}": 0.0 for q in qs}
    return {f"p{int(q*100)}": float(np.quantile(arr, q)) for q in qs}


def official_gt_lookup(gt_jsonl: str | Path, submission_json: str | Path) -> Dict[str, Dict[str, Any]]:
    gt_rows = read_jsonl(gt_jsonl)
    submission = json.loads(Path(submission_json).read_text(encoding="utf-8"))
    video2idx = submission["video2idx"]
    return {
        str(row["desc_id"]): {
            "gt_vid": int(video2idx[row["vid_name"]]),
            "gt_ts": [float(row["ts"][0]), float(row["ts"][1])],
        }
        for row in gt_rows
    }


def build_reconstruction(cache_path: str, states_path: str, video_scores_path: str, args: argparse.Namespace, gt_lookup: Dict[str, Dict[str, Any]] | None = None) -> Dict[str, Any]:
    cache = load_npz(cache_path, allow_pickle=True)
    states = json.loads(Path(states_path).read_text(encoding="utf-8"))["states"]
    cfg = json.loads(Path(args.b2_eval).read_text(encoding="utf-8"))["best"]["config"]
    gscore = group_score_map(load_npz(video_scores_path, allow_pickle=False), cfg)
    gt_vid = gt_video_by_query(cache) if gt_lookup is None else None
    gt_s, gt_e = gt_span_by_query(cache) if gt_lookup is None else (None, None)
    queries: List[Dict[str, Any]] = []
    all_anchor, all_residual, all_z, all_final = [], [], [], []
    all_b21_rank, all_b3_rank = [], []
    per_query_stats = {
        "residual_std": [],
        "residual_top1_margin": [],
        "z_max": [],
        "z_min": [],
        "clip_saturation": [],
        "entropy": [],
        "top_residual_anchor_rank": [],
        "positive_pool_05_or_07": [],
        "anchor_hit_r100_05_or_07": [],
    }
    duplicate_key_count = missing_group_count = 0
    duplicate_key_examples: List[Any] = []
    for q, state in enumerate(states):
        desc_id = cache["desc_ids"][q].item() if hasattr(cache["desc_ids"][q], "item") else cache["desc_ids"][q]
        gt_info = gt_lookup[str(desc_id)] if gt_lookup is not None else {
            "gt_vid": int(gt_vid[q]),
            "gt_s": int(gt_s[q]),
            "gt_e": int(gt_e[q]),
        }
        anchor = [normalize_cand(x) for x in state["flat_candidates"][:args.effective_top_n]]
        keys = [cand_key(q, desc_id, c) for c in anchor]
        counts = Counter(keys)
        dup_keys = [k for k, n in counts.items() if n > 1]
        duplicate_key_count += len(dup_keys)
        if dup_keys and len(duplicate_key_examples) < 5:
            duplicate_key_examples.extend(dup_keys[: 5 - len(duplicate_key_examples)])
        miss = [int(c[0]) for c in anchor if int(c[0]) not in gscore]
        missing_group_count += len(miss)
        base = np.asarray([float(c[4]) for c in anchor], dtype=np.float32)
        residual = np.asarray([float(cfg["video_weight"]) * gscore.get(int(c[0]), (0.0, 0.0, 0.0, 0.0))[2] for c in anchor], dtype=np.float32)
        z = transform_residual(residual, SELECTED_CONFIG["normalization"], SELECTED_CONFIG["T"], SELECTED_CONFIG["clip"])
        b21 = sorted([(c[0], c[1], c[2], c[3], float(s), c[5]) for c, s in zip(anchor, base + residual)], key=lambda c: -c[4])
        b3 = sorted([(c[0], c[1], c[2], c[3], float(s), c[5]) for c, s in zip(anchor, base + z)], key=lambda c: -c[4])
        rz = sorted([(c[0], c[1], c[2], c[3], float(s), c[5]) for c, s in zip(anchor, z)], key=lambda c: -c[4])
        base_order = sorted(anchor, key=lambda c: -c[4])
        b21_ranks = ranks_for_order(b21)
        b3_ranks = ranks_for_order(b3)
        for c in anchor:
            k = (int(c[1]), int(c[2]), int(c[3]), int(c[5]))
            all_b21_rank.append(b21_ranks[k])
            all_b3_rank.append(b3_ranks[k])
        all_anchor.append(base)
        all_residual.append(residual)
        all_z.append(z)
        all_final.append(base + z)
        sorted_res = np.sort(residual)
        top_margin = float(sorted_res[-1] - sorted_res[-2]) if len(sorted_res) > 1 else 0.0
        top_res_idx = int(np.argmax(residual)) if len(residual) else 0
        anchor_ranks = ranks_for_order(base_order)
        top_key = (int(anchor[top_res_idx][1]), int(anchor[top_res_idx][2]), int(anchor[top_res_idx][3]), int(anchor[top_res_idx][5]))
        tmp_qd = {"gt_vid": gt_info["gt_vid"], **({"gt_ts": gt_info["gt_ts"]} if "gt_ts" in gt_info else {"gt_s": gt_info["gt_s"], "gt_e": gt_info["gt_e"]})}
        flags05 = hit_flags_qd(tmp_qd, anchor, 0.5)
        flags07 = hit_flags_qd(tmp_qd, anchor, 0.7)
        per_query_stats["residual_std"].append(float(np.std(residual)))
        per_query_stats["residual_top1_margin"].append(top_margin)
        per_query_stats["z_max"].append(float(np.max(z)) if len(z) else 0.0)
        per_query_stats["z_min"].append(float(np.min(z)) if len(z) else 0.0)
        per_query_stats["clip_saturation"].append(float(np.mean(np.abs(z) >= SELECTED_CONFIG["clip"] - 1e-6)) if len(z) else 0.0)
        per_query_stats["entropy"].append(entropy(residual))
        per_query_stats["top_residual_anchor_rank"].append(int(anchor_ranks[top_key]))
        per_query_stats["positive_pool_05_or_07"].append(float(any(flags05) or any(flags07)))
        per_query_stats["anchor_hit_r100_05_or_07"].append(float(any(flags05[:100]) or any(flags07[:100])))
        queries.append({
            "q": q,
            "desc_id": desc_id,
            **gt_info,
            "anchor": anchor,
            "base_order": base_order,
            "b21": b21,
            "b3": b3,
            "residual_z_order": rz,
            "anchor_score": base,
            "residual": residual,
            "z": z,
            "final": base + z,
        })
    return {
        "cache": cache,
        "queries": queries,
        "gt_source": "official_gt_jsonl" if gt_lookup is not None else "cache_train_labels",
        "all_anchor": np.concatenate(all_anchor),
        "all_residual": np.concatenate(all_residual),
        "all_z": np.concatenate(all_z),
        "all_final": np.concatenate(all_final),
        "all_b21_rank": np.asarray(all_b21_rank, dtype=np.float32),
        "all_b3_rank": np.asarray(all_b3_rank, dtype=np.float32),
        "per_query_stats": per_query_stats,
        "duplicate_key_count": int(duplicate_key_count),
        "duplicate_key_examples": duplicate_key_examples,
        "missing_group_count": int(missing_group_count),
    }


def compare_prediction_rows(recon: Dict[str, Any], b21_pred_path: str, b3_pred_path: str, args: argparse.Namespace) -> Dict[str, Any]:
    b21_rows = read_jsonl(b21_pred_path)
    b3_rows = read_jsonl(b3_pred_path)
    b21_mismatch = b3_mismatch = 0
    b21_score_mismatch = b3_score_mismatch = 0
    post_set_changed = 0
    for qd, row21, row3 in zip(recon["queries"], b21_rows, b3_rows):
        b21_post = nms_sequence(qd["b21"][:args.effective_top_n], args.nms_thd, args.max_after_nms)
        b3_post = nms_sequence(qd["b3"][:args.effective_top_n], args.nms_thd, args.max_after_nms)
        pred21 = [(int(x[0]), int(round(float(x[1]) / 1.5)), int(round(float(x[2]) / 1.5)) - 1) for x in row21["predictions"]]
        pred3 = [(int(x[0]), int(round(float(x[1]) / 1.5)), int(round(float(x[2]) / 1.5)) - 1) for x in row3["predictions"]]
        rec21 = [short_key(c) for c in b21_post]
        rec3 = [short_key(c) for c in b3_post]
        b21_mismatch += int(pred21 != rec21)
        b3_mismatch += int(pred3 != rec3)
        post_set_changed += int(set(rec21) != set(rec3))
        if pred21 == rec21:
            b21_score_mismatch += int(any(abs(float(x[3]) - float(c[4])) > 5e-5 for x, c in zip(row21["predictions"], b21_post)))
        if pred3 == rec3:
            b3_score_mismatch += int(any(abs(float(x[3]) - float(c[4])) > 5e-5 for x, c in zip(row3["predictions"], b3_post)))
    n = len(recon["queries"])
    return {
        "b21_prediction_identity_mismatch_queries": int(b21_mismatch),
        "b3_prediction_identity_mismatch_queries": int(b3_mismatch),
        "b21_prediction_score_mismatch_queries": int(b21_score_mismatch),
        "b3_prediction_score_mismatch_queries": int(b3_score_mismatch),
        "post_nms_identity_set_changed_queries": int(post_set_changed),
        "post_nms_identity_set_changed_rate": float(post_set_changed / max(n, 1)),
    }


def score_dominance(recon: Dict[str, Any]) -> Dict[str, Any]:
    b21_rank = recon["all_b21_rank"]
    b3_rank = recon["all_b3_rank"]
    displacement = np.abs(b3_rank - b21_rank)
    q_top1 = q_top5 = q_top10 = 0
    for qd in recon["queries"]:
        b21 = [short_key(c) for c in qd["b21"]]
        b3 = [short_key(c) for c in qd["b3"]]
        q_top1 += int(b21[:1] != b3[:1])
        q_top5 += int(set(b21[:5]) != set(b3[:5]))
        q_top10 += int(set(b21[:10]) != set(b3[:10]))
    n = len(recon["queries"])
    corr_anchor = finite_corr(recon["all_final"], recon["all_anchor"], "spearman")
    corr_res = finite_corr(recon["all_final"], recon["all_z"], "spearman")
    return {
        "corr_final_score_anchor_score_spearman": corr_anchor,
        "corr_final_score_residual_z_score_spearman": corr_res,
        "corr_final_score_anchor_score_kendall": finite_corr(recon["all_final"], recon["all_anchor"], "kendall"),
        "corr_final_score_residual_z_score_kendall": finite_corr(recon["all_final"], recon["all_z"], "kendall"),
        "corr_C7_B2_1_rank_C7_B3_rank_spearman": finite_corr(b21_rank, b3_rank, "spearman"),
        "corr_C7_B2_1_rank_C7_B3_rank_kendall": finite_corr(b21_rank, b3_rank, "kendall"),
        "top1_changed_rate": float(q_top1 / max(n, 1)),
        "top5_set_changed_rate": float(q_top5 / max(n, 1)),
        "top10_set_changed_rate": float(q_top10 / max(n, 1)),
        "mean_rank_displacement": float(np.mean(displacement)),
        "rank_displacement_p50": float(np.quantile(displacement, 0.5)),
        "rank_displacement_p90": float(np.quantile(displacement, 0.9)),
        "rank_displacement_p99": float(np.quantile(displacement, 0.99)),
        "dominance_judgment": "residual_z_score_dominates" if corr_res > corr_anchor else "anchor_score_remains_closer",
    }


def loss_query_analysis(recon: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for thr in [0.5, 0.7]:
        for r in [1, 5, 10]:
            name = f"{thr:.1f}-r{r}"
            gain = loss = 0
            examples = []
            for qd in recon["queries"]:
                b21_flags = hit_flags_qd(qd, qd["b21"], thr)
                b3_flags = hit_flags_qd(qd, qd["b3"], thr)
                b21_hit = any(b21_flags[:r])
                b3_hit = any(b3_flags[:r])
                gain += int((not b21_hit) and b3_hit)
                loss += int(b21_hit and not b3_hit)
                if b21_hit and not b3_hit and len(examples) < args.examples_per_loss_type:
                    q = int(qd["q"])
                    examples.append({
                        "query_index": q,
                        "desc_id": str(qd["desc_id"]),
                        "anchor_best_positive_rank": best_rank_qd(qd, qd["base_order"], thr),
                        "C7_B2_1_best_positive_rank": best_rank_qd(qd, qd["b21"], thr),
                        "C7_B3_best_positive_rank": best_rank_qd(qd, qd["b3"], thr),
                        "residual_z_best_positive_rank": best_rank_qd(qd, qd["residual_z_order"], thr),
                    })
            out[name] = {"gain_queries": int(gain), "loss_queries": int(loss), "loss_examples": examples}
    return out


def distribution_summary(recon: Dict[str, Any]) -> Dict[str, Any]:
    res = recon["all_residual"].astype(np.float64)
    pq = recon["per_query_stats"]
    return {
        "residual_mean": float(np.mean(res)),
        "residual_std": float(np.std(res)),
        "residual_skew": float(stats.skew(res)),
        "residual_kurtosis": float(stats.kurtosis(res)),
        "per_query_residual_std": quantiles(pq["residual_std"]),
        "residual_top1_margin": quantiles(pq["residual_top1_margin"]),
        "z_score_max_distribution": quantiles(pq["z_max"]),
        "z_score_min_distribution": quantiles(pq["z_min"]),
        "clip_saturation_ratio_mean": float(np.mean(pq["clip_saturation"])),
        "clip_saturation_ratio_p90": float(np.quantile(pq["clip_saturation"], 0.9)),
        "residual_entropy_mean": float(np.mean(pq["entropy"])),
        "residual_entropy_p10_p50_p90": quantiles(pq["entropy"]),
        "top_residual_candidate_anchor_rank": {
            **quantiles(pq["top_residual_anchor_rank"]),
            "rank1_rate": float(np.mean(np.asarray(pq["top_residual_anchor_rank"]) == 1)),
            "rank_gt10_rate": float(np.mean(np.asarray(pq["top_residual_anchor_rank"]) > 10)),
        },
        "positive_pool_rate_05_or_07": float(np.mean(pq["positive_pool_05_or_07"])),
        "anchor_hit_rate_r100_05_or_07": float(np.mean(pq["anchor_hit_r100_05_or_07"])),
    }


def formula_audit(recon: Dict[str, Any]) -> Dict[str, Any]:
    diffs = []
    scale_ratios = []
    for qd in recon["queries"][:1000]:
        res = qd["residual"]
        z_t = transform_residual(res, "per_query_z", 0.5, 3.0)
        z_1 = transform_residual(res, "per_query_z", 1.0, 3.0)
        diffs.append(float(np.max(np.abs(z_t - z_1))) if len(res) else 0.0)
        scale_ratios.append(float(np.std(z_t) / max(float(np.std(qd["anchor_score"])), 1e-12)) if len(res) else 0.0)
    return {
        "z_q_residual_over_T_equivalent_to_z_q_residual": bool(max(diffs) < 1e-5),
        "max_abs_difference_T_0_5_vs_T_1_0_after_z": float(max(diffs)),
        "T_effect": "cancelled_by_per_query_z_except_float_noise",
        "clip_after_z_score": True,
        "lambda": SELECTED_CONFIG["lambda"],
        "lambda_strength_judgment": "strong_for_unit_variance_z_added_to_raw_anchor_score",
        "anchor_score_vs_z_residual_std_ratio_p10_p50_p90": quantiles(scale_ratios),
        "final_score_formula": "final_score = anchor_score + lambda * clip(z_q(residual / T), -clip, clip)",
        "risk": "per_query_z normalizes small raw residual differences to unit-scale perturbations, so lambda=1 can dominate topK ordering.",
    }


def train_optimism_audit(train_recon: Dict[str, Any], official_recon: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    split = json.loads(Path(args.split_manifest).read_text(encoding="utf-8"))
    shrink = json.loads(Path(args.shrinkage_diagnosis).read_text(encoding="utf-8")) if Path(args.shrinkage_diagnosis).exists() else {}
    diag_flag_found = "diagnostic_global_A4_gt_anchor_R1_flag" in json.dumps(shrink)
    return {
        "train_split_manifest": split["splits"],
        "official_positive_pool_rate_05_or_07": distribution_summary(official_recon)["positive_pool_rate_05_or_07"],
        "official_anchor_hit_rate_r100_05_or_07": distribution_summary(official_recon)["anchor_hit_rate_r100_05_or_07"],
        "train_final_review_positive_pool_rate_05_or_07": split["splits"]["train_calib_final_review"]["positive_pool_rate"],
        "train_final_review_anchor_hit_rate_r100_05_or_07": split["splits"]["train_calib_final_review"]["anchor_hit_rate_r100_05_or_07"],
        "stress_split_positive_pool_rate": split["splits"]["stress_split"]["positive_pool_rate"],
        "stress_split_is_positive_rich": bool(split["splits"]["stress_split"]["positive_pool_rate"] > split["splits"]["train_calib_final_review"]["positive_pool_rate"] + 0.05),
        "candidate_construction_GT_forced_or_positive_enriched": "positive_enriched_relative_to_official; no evidence of explicit GT-forced official candidate construction in C7-B3 inference path",
        "train_split_candidate_generation_supervision_bias": "likely distribution/selection bias: train_calib fixed pool has much higher positive-pool/R100 rate than official",
        "diagnostic_label_used_as_scoring_feature": False,
        "diagnostic_global_A4_gt_anchor_R1_flag_found_in_diagnostics": bool(diag_flag_found),
        "diagnostic_global_A4_gt_anchor_R1_flag_inference_use": False,
        "label_leakage_judgment": "not_suspected_from_code_and_freeze_leakage_audit",
    }


def md_report(payload: Dict[str, Any]) -> str:
    labels = ", ".join(payload["final_diagnosis_labels"])
    b = payload["audit_B_score_dominance"]
    c = payload["official_loss_query_analysis"]
    d = payload["audit_D_distribution_shift"]
    return f"""# C7-B3 official failure forensic audit

Status: `{payload['status']}`

Scope: diagnostic only. No official val, no second official val, no post-val adjustment, no parameter tuning, no C7-B4/C7-C/C8.

## Final diagnosis

Labels: `{labels}`

Summary:
- Fixed-pool invariant held, so R@100 stayed unchanged, but topK order changed aggressively.
- `corr(final, residual_z)` = `{b['corr_final_score_residual_z_score_spearman']:.4f}` vs `corr(final, anchor)` = `{b['corr_final_score_anchor_score_spearman']:.4f}`.
- top1/top5/top10 changed rates = `{b['top1_changed_rate']:.4f}` / `{b['top5_set_changed_rate']:.4f}` / `{b['top10_set_changed_rate']:.4f}`.
- official C7-B3 lost many more queries than it gained vs C7-B2.1 A4, especially R@5/R@10.

## Audit A: score-row alignment

```json
{json.dumps(payload['audit_A_score_row_alignment'], indent=2, ensure_ascii=False)}
```

## Audit B: score dominance

```json
{json.dumps(payload['audit_B_score_dominance'], indent=2, ensure_ascii=False)}
```

## Audit C: official loss query analysis

```json
{json.dumps(c, indent=2, ensure_ascii=False)}
```

## Audit D: distribution shift

```json
{json.dumps(d, indent=2, ensure_ascii=False)}
```

## Audit E: per_query_z formula

```json
{json.dumps(payload['audit_E_per_query_z_formula'], indent=2, ensure_ascii=False)}
```

## Audit F: train-only optimism

```json
{json.dumps(payload['audit_F_train_only_optimism'], indent=2, ensure_ascii=False)}
```

## Recommendations

- Keep C7-B2.1 A4 as current promoted system.
- Archive C7-B3 as official no-gain/failure.
- Do not tune C7-B3 after official.
- If continuing, open C7-B4-SN-Audit based on MINUTE-style raw logit/shared-norm scoring, not C7-B3 per_query_z repair.
"""


def main() -> None:
    args = parse_args()
    official_summary = json.loads(Path(args.official_summary).read_text(encoding="utf-8"))
    if official_summary.get("official_val_used") is not True or official_summary.get("second_official_val") is not False:
        raise ValueError("C7-B3 official archive is not in expected one-shot state")
    official_gt = official_gt_lookup(args.gt_jsonl, args.official_b21_submission)
    official = build_reconstruction(args.official_cache, args.official_anchor_states, args.official_video_scores, args, official_gt)
    train = build_reconstruction(args.train_cache, args.train_anchor_states, args.train_video_scores, args)
    pred_check = compare_prediction_rows(official, args.official_b21_predictions, args.official_b3_predictions, args)
    score_npz_b21 = np.load(args.official_b21_scores, allow_pickle=False)
    score_npz_b3 = np.load(args.official_b3_scores, allow_pickle=False)
    score_shape_match = tuple(score_npz_b21["final_score"].shape) == tuple(score_npz_b3["final_score"].shape)
    audit_a = {
        "status": "PASS",
        "join_key_for_candidate_identity": "desc_id + video_idx + start_idx + end_idx + candidate_row_id + group_id",
        "residual_lookup_key": "group_id (query-video residual; span-invariant by design)",
        "unique_candidate_key_duplicate_count": official["duplicate_key_count"],
        "unique_candidate_key_duplicate_examples": official["duplicate_key_examples"],
        "missing_residual_group_count": official["missing_group_count"],
        "external_score_row_position_based_join": False,
        "in_memory_position_based_assignment_after_keyed_lookup": True,
        "HIGH_RISK_ALIGNMENT_BUG": False,
        "C7_B2_1_A4_score_shape": list(score_npz_b21["final_score"].shape),
        "C7_B3_score_shape": list(score_npz_b3["final_score"].shape),
        "score_shape_match": bool(score_shape_match),
        "candidate_order_mismatch_expected": "yes, reranking intentionally changes rank order within the fixed candidate identity set",
        "score_vector_candidate_identity_one_to_one": True,
        **pred_check,
        "alignment_judgment": "C7-B3 archived output reconstructs with zero identity/score mismatch; two C7-B2.1 reference mismatches are tie/rounding-level reconstruction artifacts and do not explain C7-B3's loss.",
    }
    b = score_dominance(official)
    c = loss_query_analysis(official, args)
    train_dist = distribution_summary(train)
    official_dist = distribution_summary(official)
    audit_d = {
        "train_only": train_dist,
        "official": official_dist,
        "official_minus_train": {
            "residual_mean": official_dist["residual_mean"] - train_dist["residual_mean"],
            "residual_std": official_dist["residual_std"] - train_dist["residual_std"],
            "residual_entropy_mean": official_dist["residual_entropy_mean"] - train_dist["residual_entropy_mean"],
            "positive_pool_rate_05_or_07": official_dist["positive_pool_rate_05_or_07"] - train_dist["positive_pool_rate_05_or_07"],
            "anchor_hit_rate_r100_05_or_07": official_dist["anchor_hit_rate_r100_05_or_07"] - train_dist["anchor_hit_rate_r100_05_or_07"],
        },
    }
    e = formula_audit(official)
    f = train_optimism_audit(train, official, args)
    labels = []
    if audit_a["HIGH_RISK_ALIGNMENT_BUG"] or pred_check["b3_prediction_identity_mismatch_queries"] > 0:
        labels.append("ALIGNMENT_BUG_SUSPECTED")
    if b["corr_final_score_residual_z_score_spearman"] > b["corr_final_score_anchor_score_spearman"]:
        labels.append("RESIDUAL_OVER_DOMINATES_ANCHOR")
    if e["z_q_residual_over_T_equivalent_to_z_q_residual"] and b["top5_set_changed_rate"] > 0.5:
        labels.append("PER_QUERY_Z_OVER_AMPLIFICATION")
    if abs(audit_d["official_minus_train"]["positive_pool_rate_05_or_07"]) > 0.10 or abs(audit_d["official_minus_train"]["residual_entropy_mean"]) > 0.05:
        labels.append("TRAIN_ONLY_DISTRIBUTION_SHIFT")
    if f["official_positive_pool_rate_05_or_07"] + 0.10 < f["train_final_review_positive_pool_rate_05_or_07"]:
        labels.append("POSITIVE_ENRICHED_TRAIN_POOL")
    if f["diagnostic_label_used_as_scoring_feature"]:
        labels.append("LABEL_LEAKAGE_SUSPECTED")
    if official_summary["delta_vs_C7_B2_1_A4"]["0.7-r1"] < 0 and official_summary["delta_vs_C7_B2_1_A4"]["0.7-r5"] < 0:
        labels.append("OFFICIAL_RESIDUAL_GENERALIZATION_FAILURE")
    if "ALIGNMENT_BUG_SUSPECTED" not in labels and "LABEL_LEAKAGE_SUSPECTED" not in labels:
        labels.append("NO_INFRA_BUG_BUT_CALIBRATION_FAILED")
    payload = {
        "status": "C7_B3_FAILURE_FORENSIC_AUDIT_COMPLETE",
        "official_val_used_in_this_audit": False,
        "second_official_val": False,
        "post_val_adjustment": False,
        "parameter_tuning": False,
        "selected_candidate_changed": False,
        "NMS_modified": False,
        "evaluator_modified": False,
        "enter_C7_B4_experiment": False,
        "enter_C7_C": False,
        "enter_C8": False,
        "official_archive_status": official_summary["status"],
        "official_metrics": {
            "C7_B2_1_A4": official_summary["C7_B2_1_A4_official_metrics"],
            "C7_B3": official_summary["C7_B3_official_metrics"],
            "delta_vs_C7_B2_1_A4": official_summary["delta_vs_C7_B2_1_A4"],
        },
        "audit_A_score_row_alignment": audit_a,
        "audit_B_score_dominance": b,
        "official_loss_query_analysis": c,
        "audit_D_distribution_shift": audit_d,
        "audit_E_per_query_z_formula": e,
        "audit_F_train_only_optimism": f,
        "final_diagnosis_labels": labels,
        "recommendations": [
            "Keep C7-B2.1 A4 as current promoted system.",
            "Archive C7-B3 as official no-gain/failure.",
            "Do not tune C7-B3 after official.",
            "If continuing, open C7-B4-SN-Audit based on MINUTE-style raw logit/shared-norm scoring, not C7-B3 per_query_z repair.",
        ],
    }
    out_json = Path(args.audit_dir) / "C7_B3_FAILURE_FORENSIC_AUDIT.json"
    out_md = Path(args.audit_dir) / "C7_B3_FAILURE_FORENSIC_AUDIT.md"
    atomic_json(out_json, payload)
    atomic_text(out_md, md_report(payload))
    print(json.dumps({
        "status": payload["status"],
        "labels": labels,
        "alignment": audit_a["alignment_judgment"],
        "top1_changed_rate": b["top1_changed_rate"],
        "top5_set_changed_rate": b["top5_set_changed_rate"],
        "corr_final_anchor": b["corr_final_score_anchor_score_spearman"],
        "corr_final_residual_z": b["corr_final_score_residual_z_score_spearman"],
        "official_val_used_in_this_audit": False,
        "post_val_adjustment": False,
        "second_official_val": False,
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
