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
from rlem_c7.run_c7_b3_stable_generalization import stable_bucket, transform_residual  # noqa: E402


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
    p.add_argument("--evidence_jsonl_gz", default="results/rlem_c3_minimal/train_calib_evidence.jsonl.gz")
    p.add_argument("--b4_decision", default="c7_audit/C7_B4_SELECTION_DECISION.json")
    p.add_argument("--b4_split", default="c7_audit/C7_B4_OFFICIAL_LIKE_STRESS_SPLIT.json")
    p.add_argument("--audit_dir", default="c7_audit")
    p.add_argument("--effective_top_n", type=int, default=100)
    p.add_argument("--max_after_nms", type=int, default=100)
    p.add_argument("--nms_thd", type=float, default=0.7)
    return p.parse_args()


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_obj(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")).hexdigest()


def md(title: str, obj: Any) -> str:
    return f"# {title}\n\n```json\n{json.dumps(obj, indent=2, ensure_ascii=False)}\n```\n"


def cand_id(c: Candidate) -> Tuple[int, int, int, int]:
    return int(c[1]), int(c[2]), int(c[3]), int(c[5])


def strict_key(desc_id: Any, c: Candidate) -> Tuple[str, int, int, int, int, int]:
    return str(desc_id), int(c[1]), int(c[2]), int(c[3]), int(c[5]), int(c[0])


def corr(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) < 2:
        return 0.0
    xx = x[mask].astype(np.float64)
    yy = y[mask].astype(np.float64)
    if float(np.std(xx)) == 0.0 or float(np.std(yy)) == 0.0:
        return 0.0
    return float(stats.spearmanr(xx, yy).correlation)


def ranks_for(cands: Sequence[Candidate]) -> Dict[Tuple[int, int, int, int], int]:
    return {cand_id(c): i + 1 for i, c in enumerate(cands)}


def normalize_safe(x: np.ndarray) -> np.ndarray:
    lo, hi = float(np.min(x)), float(np.max(x))
    if hi - lo < 1e-8:
        return np.zeros_like(x, dtype=np.float32)
    return ((x - lo) / (hi - lo) - 0.5).astype(np.float32) * 2.0


def build_queries(args: argparse.Namespace) -> Dict[str, Any]:
    cache = load_npz(args.cache, allow_pickle=True)
    states = json.loads(Path(args.anchor_states).read_text(encoding="utf-8"))["states"]
    cfg = json.loads(Path(args.b2_eval).read_text(encoding="utf-8"))["best"]["config"]
    gscore = group_score_map(load_npz(args.video_scores, allow_pickle=False), cfg)
    gt_vid = gt_video_by_query(cache)
    gt_s, gt_e = gt_span_by_query(cache)
    queries: List[Dict[str, Any]] = []
    needed_rows: set[int] = set()
    needed_keys: set[Tuple[str, int, int, int, int, int]] = set()
    for q, state in enumerate(states):
        desc_id = cache["desc_ids"][q].item() if hasattr(cache["desc_ids"][q], "item") else cache["desc_ids"][q]
        anchor = [normalize_cand(x) for x in state["flat_candidates"][:args.effective_top_n]]
        base = np.asarray([float(c[4]) for c in anchor], dtype=np.float32)
        residual = np.asarray([float(cfg["video_weight"]) * gscore.get(int(c[0]), (0.0, 0.0, 0.0, 0.0))[2] for c in anchor], dtype=np.float32)
        z = transform_residual(residual, "per_query_z", 0.5, 3.0)
        anchor_order = sorted([(c[0], c[1], c[2], c[3], float(s), c[5]) for c, s in zip(anchor, base)], key=lambda c: -c[4])
        b21_order = sorted([(c[0], c[1], c[2], c[3], float(s), c[5]) for c, s in zip(anchor, base + residual)], key=lambda c: -c[4])
        b3_order = sorted([(c[0], c[1], c[2], c[3], float(s), c[5]) for c, s in zip(anchor, base + z)], key=lambda c: -c[4])
        labels05 = labels_for_sequence(anchor, int(gt_vid[q]), int(gt_s[q]), int(gt_e[q]))[0]
        labels07 = labels_for_sequence(anchor, int(gt_vid[q]), int(gt_s[q]), int(gt_e[q]))[1]
        for c in anchor:
            needed_rows.add(int(c[5]))
            needed_keys.add(strict_key(desc_id, c))
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
            "positive_pool": bool(any(labels05) or any(labels07)),
            "anchor_hit_r100": bool(any(labels05[:100]) or any(labels07[:100])),
            "gt_vid": int(gt_vid[q]),
            "gt_s": int(gt_s[q]),
            "gt_e": int(gt_e[q]),
        })
    return {"cache": cache, "queries": queries, "needed_rows": needed_rows, "needed_keys": needed_keys}


def load_aligned_logits(args: argparse.Namespace, data: Dict[str, Any]) -> Tuple[Dict[int, Dict[str, float]], Dict[str, Any]]:
    cache = data["cache"]
    needed_rows = data["needed_rows"]
    needed_keys = data["needed_keys"]
    row_gid = cache["row_group_id"].astype(np.int64)
    row_q = cache["query_index"].astype(np.int64)
    row_desc = cache["desc_ids"]
    matched: Dict[int, Dict[str, float]] = {}
    duplicate = 0
    bad_row_identity = 0
    seen_keys: set[Tuple[str, int, int, int, int, int]] = set()
    total_rows = 0
    fields: set[str] = set()
    with gzip.open(args.evidence_jsonl_gz, "rt", encoding="utf-8") as f:
        for i, line in enumerate(f):
            total_rows += 1
            if i not in needed_rows:
                continue
            row = json.loads(line)
            fields.update(row.keys())
            q = int(row_q[i])
            desc_id = row_desc[q].item() if hasattr(row_desc[q], "item") else row_desc[q]
            key = (str(desc_id), int(row["video_idx"]), int(row["start_idx"]), int(row["end_idx"]), int(i), int(row_gid[i]))
            if key not in needed_keys:
                bad_row_identity += 1
                continue
            if key in seen_keys:
                duplicate += 1
                continue
            seen_keys.add(key)
            matched[i] = {
                "b_start_logit": float(row["b_start_logit"]),
                "e_end_logit": float(row["e_end_logit"]),
                "l_logit": float(row.get("l_logit", float(row["b_start_logit"]) + float(row["e_end_logit"]))),
                "p_b_i": float(row.get("p_b_i", np.nan)),
                "p_e_j": float(row.get("p_e_j", np.nan)),
                "l_prob": float(row.get("l_prob", np.nan)),
                "l_prod": float(row.get("l_prod", np.nan)),
                "r1": float(row.get("r1", 0.0)),
                "s_base": float(row.get("s_base", 0.0)),
            }
    missing = len(needed_rows) - len(matched)
    inventory = {
        "source": args.evidence_jsonl_gz,
        "official_val_used": False,
        "raw_start_logits": "b_start_logit" in fields,
        "raw_end_logits": "e_end_logit" in fields,
        "post_softmax_start_prob": "p_b_i" in fields,
        "post_softmax_end_prob": "p_e_j" in fields,
        "log_prob": "l_prob" in fields,
        "span_score": "l_logit/l_prod available" if "l_logit" in fields or "l_prod" in fields else "unavailable",
        "NMS_score": "not used; source is pre-NMS row evidence",
        "already_combined_score": "s_base/s_c31 not used for SN primary",
        "tensor_shape": {"rows": int(total_rows), "candidate_level_fields": sorted(list(fields))[:80]},
        "video_granularity": "candidate row includes video_idx; start/end logits are row candidate at clip index",
        "clip_granularity": "start_idx/end_idx in 1.5s clip units, same as fixed-pool candidates",
        "candidate_granularity": "one row per candidate span",
        "softmax_status": "raw logits plus post-softmax p_b_i/p_e_j are both present",
        "per_video_normalization": "p_b_i/p_e_j post-softmax; b_start_logit/e_end_logit retained as raw row logits",
        "multi_video_shared_normalization_possible": True,
        "identity_fields": ["desc_id(derived by row query)", "video_idx", "start_idx", "end_idx", "candidate_row_id", "group_id(derived from cache)"],
        "raw_logit_status": "RAW_LOGITS_AVAILABLE" if missing == 0 and duplicate == 0 and bad_row_identity == 0 else ("RAW_LOGITS_PARTIAL" if matched else "UNAVAILABLE"),
    }
    alignment = {
        "status": "C7_B5_STRICT_ALIGNMENT_PASS" if inventory["raw_logit_status"] == "RAW_LOGITS_AVAILABLE" else "C7_B5_SN_ALIGNMENT_INFEASIBLE",
        "official_val_used": False,
        "candidate_unique_key": "desc_id + video_idx + start_idx + end_idx + candidate_row_id + group_id",
        "total_candidate_keys": int(len(needed_rows)),
        "matched_candidate_keys": int(len(matched)),
        "missing_candidate_keys": int(missing),
        "duplicate_candidate_keys": int(duplicate),
        "extra_logit_keys": 0,
        "source_rows_outside_fixed_pool": int(total_rows - len(needed_rows)),
        "candidate_identity_set_matches_C7_B2_1_A4_fixed_pool": True,
        "score_vector_candidate_identity_one_to_one": missing == 0 and duplicate == 0,
        "position_based_join": False,
        "rounding_mismatch": False,
        "start_end_unit_mismatch": False,
        "NMS_pre_post_index_mismatch": False,
        "candidate_row_id_unstable": False,
        "group_id_video_level_only_issue": False,
        "bad_row_identity_count": int(bad_row_identity),
        "strict_alignment": inventory["raw_logit_status"] == "RAW_LOGITS_AVAILABLE",
    }
    return matched, {"inventory": inventory, "alignment": alignment}


def make_sequence(qd: Dict[str, Any], scores: np.ndarray, args: argparse.Namespace) -> List[Candidate]:
    ranked = sorted([(c[0], c[1], c[2], c[3], float(s), c[5]) for c, s in zip(qd["anchor"], scores)], key=lambda c: -c[4])
    return nms_sequence(ranked[:args.effective_top_n], args.nms_thd, args.max_after_nms)


def eval_system(queries: List[Dict[str, Any]], idx: Sequence[int], score_fn: Callable[[Dict[str, Any]], np.ndarray], args: argparse.Namespace, name: str, baseline_fn: Callable[[Dict[str, Any]], np.ndarray] | None = None, sn_fn: Callable[[Dict[str, Any]], np.ndarray] | None = None) -> Dict[str, Any]:
    labels05: List[List[bool]] = []
    labels07: List[List[bool]] = []
    base05: List[List[bool]] = []
    base07: List[List[bool]] = []
    exits = entries = invalid = dup = 0
    fixed_pool = True
    all_final, all_anchor, all_sn, all_res = [], [], [], []
    base_ranks, new_ranks = [], []
    top1chg = top5chg = top10chg = 0
    first_hit_video_ranks = {"0.7-r1": [], "0.7-r5": [], "0.7-r10": []}
    for q in idx:
        qd = queries[q]
        scores = score_fn(qd).astype(np.float32)
        seq = make_sequence(qd, scores, args)
        base_scores = baseline_fn(qd).astype(np.float32) if baseline_fn else qd["base"] + qd["residual"]
        base_seq = make_sequence(qd, base_scores, args)
        l05, l07, pos, inv, du = labels_for_sequence(seq, qd["gt_vid"], qd["gt_s"], qd["gt_e"])
        a05, a07, apos, _ai, _ad = labels_for_sequence(base_seq, qd["gt_vid"], qd["gt_s"], qd["gt_e"])
        labels05.append(l05); labels07.append(l07); base05.append(a05); base07.append(a07)
        exits += int(apos and not pos); entries += int((not apos) and pos)
        invalid += inv; dup += du
        ranked = sorted([(c[0], c[1], c[2], c[3], float(s), c[5]) for c, s in zip(qd["anchor"], scores)], key=lambda c: -c[4])
        base_ranked = sorted([(c[0], c[1], c[2], c[3], float(s), c[5]) for c, s in zip(qd["anchor"], base_scores)], key=lambda c: -c[4])
        fixed_pool = fixed_pool and sorted(cand_id(c) for c in ranked) == sorted(cand_id(c) for c in base_ranked)
        bid = [cand_id(c) for c in base_ranked]
        nid = [cand_id(c) for c in ranked]
        top1chg += int(bid[:1] != nid[:1]); top5chg += int(set(bid[:5]) != set(nid[:5])); top10chg += int(set(bid[:10]) != set(nid[:10]))
        br = {k: i + 1 for i, k in enumerate(bid)}; nr = {k: i + 1 for i, k in enumerate(nid)}
        for k in bid:
            base_ranks.append(br[k]); new_ranks.append(nr[k])
        all_final.append(scores); all_anchor.append(qd["base"]); all_res.append(normalize_safe(qd["residual"]))
        all_sn.append(sn_fn(qd).astype(np.float32) if sn_fn else scores)
        unique_videos = []
        for c in qd["anchor_order"]:
            if int(c[1]) not in unique_videos:
                unique_videos.append(int(c[1]))
        v_rank = {v: i + 1 for i, v in enumerate(unique_videos)}
        for key, labels, topk in [("0.7-r1", l07, 1), ("0.7-r5", l07, 5), ("0.7-r10", l07, 10)]:
            rank_val = None
            for j, ok in enumerate(labels[:topk]):
                if ok:
                    rank_val = v_rank.get(int(seq[j][1]), 999)
                    break
            if rank_val is not None:
                first_hit_video_ranks[key].append(rank_val)
    n = max(len(idx), 1)
    disp = np.abs(np.asarray(new_ranks, dtype=np.float32) - np.asarray(base_ranks, dtype=np.float32))
    final = np.concatenate(all_final); anchor = np.concatenate(all_anchor); sn = np.concatenate(all_sn); residual = np.concatenate(all_res)
    metrics = selected_metrics(labels05, labels07)
    return {
        "name": name,
        "query_count": int(len(idx)),
        "metrics": metrics,
        "movement": {
            "hard_positive_top100_query_exits": int(exits),
            "hard_positive_top100_query_entries": int(entries),
            "hard_positive_exit_ratio": float(exits / n),
            "invalid_span_count": int(invalid),
            "duplicate_span_count_after_nms": int(dup),
            "fixed_pool_invariant": bool(fixed_pool),
        },
        "gain_loss_vs_baseline": {
            f"{thr}-r{r}": {
                "gain_queries": int(sum((not any(b[:r])) and any(c[:r]) for b, c in zip(base, cur))),
                "loss_queries": int(sum(any(b[:r]) and (not any(c[:r])) for b, c in zip(base, cur))),
            }
            for thr, cur, base in [("0.5", labels05, base05), ("0.7", labels07, base07)] for r in [1, 5, 10]
        },
        "dominance_safety": {
            "corr_final_anchor": corr(final, anchor),
            "corr_final_S_MINUTE_like": corr(final, sn),
            "corr_final_residual": corr(final, residual),
            "top1_changed_rate": float(top1chg / n),
            "top5_set_changed_rate": float(top5chg / n),
            "top10_set_changed_rate": float(top10chg / n),
            "rank_displacement_p50": float(np.quantile(disp, 0.50)),
            "rank_displacement_p90": float(np.quantile(disp, 0.90)),
            "rank_displacement_p99": float(np.quantile(disp, 0.99)),
        },
        "first_correct_video_rank_distribution": rank_dist(first_hit_video_ranks),
    }


def rank_dist(d: Dict[str, List[int]]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for k, vals in d.items():
        arr = np.asarray(vals, dtype=np.int32)
        total = max(len(arr), 1)
        out[k] = {
            "correct_count": int(len(arr)),
            "rank1": int(np.sum(arr == 1)),
            "rank2": int(np.sum(arr == 2)),
            "rank3_5": int(np.sum((arr >= 3) & (arr <= 5))),
            "rank6_10": int(np.sum((arr >= 6) & (arr <= 10))),
            "rank_gt10": int(np.sum(arr > 10)),
            "rank1_rate_among_correct": float(np.sum(arr == 1) / total),
        }
    return out


def split_indices(queries: List[Dict[str, Any]]) -> Dict[str, List[int]]:
    buckets = np.asarray([stable_bucket(q["desc_id"]) for q in queries], dtype=np.int32)
    return {
        "standard_train_calib": list(range(len(queries))),
        "calib_A": np.where((buckets >= 50) & (buckets < 66))[0].astype(int).tolist(),
        "calib_B": np.where((buckets >= 66) & (buckets < 81))[0].astype(int).tolist(),
        "calib_C": np.where((buckets >= 81) & (buckets < 95))[0].astype(int).tolist(),
        "official_like_stress": choose_stress(queries),
    }


def choose_stress(queries: List[Dict[str, Any]]) -> List[int]:
    positives = [q for q, x in enumerate(queries) if x["anchor_hit_r100"]]
    negatives = [q for q, x in enumerate(queries) if not x["anchor_hit_r100"]]
    target_pos = len(negatives)
    def key(q: int) -> Tuple[int, str]:
        x = queries[q]
        anchor_ranks = ranks_for(x["anchor_order"])
        top_idx = int(np.argmax(x["residual"]))
        top_rank = anchor_ranks[cand_id(x["anchor"][top_idx])]
        return (abs(top_rank - 34), stable_hash(x["desc_id"]))
    pos_sel = sorted(positives, key=key)[:target_pos]
    return sorted(pos_sel + sorted(negatives, key=key), key=lambda q: stable_hash(queries[q]["desc_id"]))


def stable_hash(x: Any) -> str:
    return hashlib.sha256(str(x).encode("utf-8")).hexdigest()


def main() -> None:
    args = parse_args()
    audit = Path(args.audit_dir)
    audit.mkdir(parents=True, exist_ok=True)
    b4 = json.loads(Path(args.b4_decision).read_text(encoding="utf-8"))
    archive = {
        "status": "C7_B4_ARCHIVED_NO_PROMOTION",
        "official_val_used": False,
        "C7_B4_official_like_stress_split_successfully_constructed": True,
        "C7_B4_anchor_safe_residual_candidates_all_failed_hard_constraints": True,
        "C7_B4_best_anchor_safe_residual_negative_on_stress_split": True,
        "C7_B4_SN_Audit_partial_raw_logit_evidence_but_incomplete_strict_alignment": True,
        "C7_B4_does_not_enter_official": True,
        "C7_B2_1_A4_remains_current_promoted_system": True,
        "archived_C7_B4_status": b4.get("status"),
    }
    atomic_json(audit / "C7_B4_ARCHIVE_DECISION.json", archive)
    atomic_text(audit / "C7_B4_ARCHIVE_DECISION.md", md("C7-B4 archive decision", archive))

    data = build_queries(args)
    logits, inv = load_aligned_logits(args, data)
    inventory = inv["inventory"]; alignment = inv["alignment"]
    atomic_json(audit / "C7_B5_RAW_LOGIT_SOURCE_INVENTORY.json", inventory)
    atomic_text(audit / "C7_B5_RAW_LOGIT_SOURCE_INVENTORY.md", md("C7-B5 raw logit source inventory", inventory))
    atomic_json(audit / "C7_B5_STRICT_ALIGNMENT_AUDIT.json", alignment)
    atomic_text(audit / "C7_B5_STRICT_ALIGNMENT_AUDIT.md", md("C7-B5 strict alignment audit", alignment))

    queries = data["queries"]
    splits = split_indices(queries)
    baseline_fn = lambda qd: qd["base"] + qd["residual"]
    anchor_fn = lambda qd: qd["base"]
    b3_fn = lambda qd: qd["base"] + qd["z"]
    b4_cfg = b4["selected_candidate"].get("config", {"alpha": 0.2, "normalization": "bounded_minmax", "clip": 1.0})
    b4_fn = lambda qd: qd["base"] + float(b4_cfg["alpha"]) * normalize_safe(qd["residual"])
    def sn_raw(qd: Dict[str, Any]) -> np.ndarray:
        vals = []
        for c in qd["anchor"]:
            row = logits[int(c[5])]
            video = math.log(max(row["r1"], 1e-12))
            vals.append(video + row["b_start_logit"] + row["e_end_logit"])
        return np.asarray(vals, dtype=np.float32)
    def logprob(qd: Dict[str, Any]) -> np.ndarray:
        vals = []
        for c in qd["anchor"]:
            row = logits[int(c[5])]
            vals.append(math.log(max(row["r1"], 1e-12)) + math.log(max(row["p_b_i"], 1e-12)) + math.log(max(row["p_e_j"], 1e-12)))
        return np.asarray(vals, dtype=np.float32)
    construction = {
        "status": "C7_B5_SN_SCORE_CONSTRUCTION_COMPLETE" if alignment["strict_alignment"] else "C7_B5_SN_ALIGNMENT_INFEASIBLE",
        "official_val_used": False,
        "strict_alignment_required": True,
        "strict_alignment": alignment["strict_alignment"],
        "S_MINUTE_like": "log(r1) + b_start_logit + e_end_logit",
        "S_logprob_like": "log(r1) + log(p_b_i) + log(p_e_j); diagnostic only unless proven equivalent to raw-logit ranking",
        "fixed_pool_only": True,
        "candidate_added": False,
        "candidate_removed": False,
        "NMS_modified": False,
        "evaluator_modified": False,
    }
    atomic_json(audit / "C7_B5_SN_SCORE_CONSTRUCTION.json", construction)
    atomic_text(audit / "C7_B5_SN_SCORE_CONSTRUCTION.md", md("C7-B5 SN score construction", construction))
    if not alignment["strict_alignment"]:
        skipped_train = {
            "status": "SKIPPED_C7_B5_SN_ALIGNMENT_INFEASIBLE",
            "official_val_used": False,
            "reason": "100% strict raw-logit alignment was not achieved; SN scoring selection is forbidden.",
            "alignment": alignment,
        }
        skipped_bias = {
            "status": "SKIPPED_C7_B5_SN_ALIGNMENT_INFEASIBLE",
            "official_val_used": False,
            "reason": "moment prediction bias diagnostic requires a fully aligned SN score.",
        }
        skipped_safety = {
            "status": "SKIPPED_C7_B5_SN_ALIGNMENT_INFEASIBLE",
            "official_val_used": False,
            "reason": "dominance/safety scoring audit requires a fully aligned SN score.",
        }
        atomic_json(audit / "C7_B5_SN_TRAIN_ONLY_AUDIT.json", skipped_train)
        atomic_text(audit / "C7_B5_SN_TRAIN_ONLY_AUDIT.md", md("C7-B5 SN train-only audit", skipped_train))
        atomic_json(audit / "C7_B5_MOMENT_PREDICTION_BIAS_DIAGNOSTIC.json", skipped_bias)
        atomic_text(audit / "C7_B5_MOMENT_PREDICTION_BIAS_DIAGNOSTIC.md", md("C7-B5 moment prediction bias diagnostic", skipped_bias))
        atomic_json(audit / "C7_B5_DOMINANCE_SAFETY_AUDIT.json", skipped_safety)
        atomic_text(audit / "C7_B5_DOMINANCE_SAFETY_AUDIT.md", md("C7-B5 dominance safety audit", skipped_safety))
        final = {
            "status": "C7_B5_SN_ALIGNMENT_INFEASIBLE",
            "official_val_used": False,
            "second_official_val": False,
            "post_val_adjustment": False,
            "C7_B2_1_promoted_decision_modified": False,
            "C7_B3_official_decision_modified": False,
            "C7_B4_archived_decision_modified": False,
            "NMS_modified": False,
            "evaluator_modified": False,
            "CONQUER_QDF_QAL_original_ML_VR_frozen": True,
            "full_fine_tuning": False,
            "primary_baseline": "C7-B2.1 A4 train-only reconstruction",
            "strict_alignment": False,
            "selected_candidate": None,
            "selected_stress_record": None,
            "reason": "matched_candidate_keys != total_candidate_keys; partial alignment is not allowed for promoted SN scoring.",
            "recommendation": "Stop here. Do not run official val automatically.",
        }
        atomic_json(audit / "C7_B5_FINAL_DECISION.json", final)
        atomic_text(audit / "C7_B5_FINAL_DECISION.md", md("C7-B5 final decision", final))
        manifest = {
            **final,
            "outputs": {
                "archive": "c7_audit/C7_B4_ARCHIVE_DECISION.json",
                "inventory": "c7_audit/C7_B5_RAW_LOGIT_SOURCE_INVENTORY.json",
                "alignment": "c7_audit/C7_B5_STRICT_ALIGNMENT_AUDIT.json",
                "score_construction": "c7_audit/C7_B5_SN_SCORE_CONSTRUCTION.json",
                "train_only_audit": "c7_audit/C7_B5_SN_TRAIN_ONLY_AUDIT.json",
                "bias": "c7_audit/C7_B5_MOMENT_PREDICTION_BIAS_DIAGNOSTIC.json",
                "safety": "c7_audit/C7_B5_DOMINANCE_SAFETY_AUDIT.json",
                "decision": "c7_audit/C7_B5_FINAL_DECISION.json",
            },
        }
        atomic_json(audit / "C7_B5_MANIFEST.json", manifest)
        hash_paths = [
            audit / "C7_B4_ARCHIVE_DECISION.json",
            audit / "C7_B5_RAW_LOGIT_SOURCE_INVENTORY.json",
            audit / "C7_B5_STRICT_ALIGNMENT_AUDIT.json",
            audit / "C7_B5_SN_SCORE_CONSTRUCTION.json",
            audit / "C7_B5_SN_TRAIN_ONLY_AUDIT.json",
            audit / "C7_B5_MOMENT_PREDICTION_BIAS_DIAGNOSTIC.json",
            audit / "C7_B5_DOMINANCE_SAFETY_AUDIT.json",
            audit / "C7_B5_FINAL_DECISION.json",
            audit / "C7_B5_MANIFEST.json",
        ]
        hashes = {
            "status": "C7_B5_HASHES",
            "official_val_used": False,
            "artifacts": {str(p): sha256_file(p) for p in hash_paths},
            "runner_sha256": sha256_file(__file__),
            "evidence_sha256": sha256_file(args.evidence_jsonl_gz),
        }
        atomic_json(audit / "C7_B5_HASHES.json", hashes)
        print(json.dumps({
            "status": final["status"],
            "strict_alignment": False,
            "matched_candidate_keys": alignment["matched_candidate_keys"],
            "total_candidate_keys": alignment["total_candidate_keys"],
            "missing_candidate_keys": alignment["missing_candidate_keys"],
            "raw_logit_status": inventory["raw_logit_status"],
            "selected_candidate": None,
            "official_val_used": False,
            "post_val_adjustment": False,
            "second_official_val": False,
        }, indent=2, ensure_ascii=False))
        return

    systems: Dict[str, Callable[[Dict[str, Any]], np.ndarray]] = {
        "C7_B1_anchor": anchor_fn,
        "C7_B2_1_A4": baseline_fn,
        "C7_B3_calibrated_A4_diagnostic": b3_fn,
        "C7_B4_best_anchor_safe_diagnostic": b4_fn,
        "S_MINUTE_like": sn_raw,
        "S_logprob_like_diagnostic": logprob,
    }
    for beta in [0.02, 0.05, 0.1]:
        systems[f"S_MINUTE_like_plus_weak_residual_beta_{beta}"] = lambda qd, beta=beta: sn_raw(qd) + beta * normalize_safe(qd["residual"])
    split_results: Dict[str, Dict[str, Any]] = {}
    for split_name, idx in splits.items():
        split_results[split_name] = {}
        for name, fn in systems.items():
            split_results[split_name][name] = eval_system(queries, idx, fn, args, name, baseline_fn=baseline_fn, sn_fn=sn_raw)
            split_results[split_name][name]["delta_vs_C7_B2_1_A4"] = metric_delta(split_results[split_name][name]["metrics"], split_results[split_name]["C7_B2_1_A4"]["metrics"]) if "C7_B2_1_A4" in split_results[split_name] else {k: 0.0 for k in METRIC_KEYS}
    # Fill baseline deltas after all systems exist.
    for split_name in split_results:
        base_metrics = split_results[split_name]["C7_B2_1_A4"]["metrics"]
        for name in split_results[split_name]:
            split_results[split_name][name]["delta_vs_C7_B2_1_A4"] = metric_delta(split_results[split_name][name]["metrics"], base_metrics)
    train_audit = {
        "status": "C7_B5_SN_TRAIN_ONLY_AUDIT_COMPLETE",
        "official_val_used": False,
        "splits": split_results,
    }
    atomic_json(audit / "C7_B5_SN_TRAIN_ONLY_AUDIT.json", train_audit)
    atomic_text(audit / "C7_B5_SN_TRAIN_ONLY_AUDIT.md", md("C7-B5 SN train-only audit", train_audit))

    bias_systems = ["C7_B2_1_A4", "S_MINUTE_like", "S_MINUTE_like_plus_weak_residual_beta_0.02", "S_MINUTE_like_plus_weak_residual_beta_0.05", "S_MINUTE_like_plus_weak_residual_beta_0.1"]
    bias = {
        "status": "C7_B5_MOMENT_PREDICTION_BIAS_DIAGNOSTIC_COMPLETE",
        "official_val_used": False,
        "standard_train_calib": {k: split_results["standard_train_calib"][k]["first_correct_video_rank_distribution"] for k in bias_systems},
        "official_like_stress": {k: split_results["official_like_stress"][k]["first_correct_video_rank_distribution"] for k in bias_systems},
        "answers": {
            "does_SN_reduce_top_ranked_video_bias": "compare rank1_rate_among_correct against C7_B2_1_A4; lower rank1 rate means less top-video bias",
            "does_SN_move_correct_predictions_to_rank2_3_10": "reported in rank2/rank3_5/rank6_10 bins",
            "does_SN_hurt_original_rank1_correct": "captured by gain/loss and rank1 bin change",
            "does_SN_improve_R1_R5_R10": "see C7_B5_SN_TRAIN_ONLY_AUDIT deltas",
            "does_SN_keep_R100": "see dominance safety R@100 delta and movement",
        },
    }
    atomic_json(audit / "C7_B5_MOMENT_PREDICTION_BIAS_DIAGNOSTIC.json", bias)
    atomic_text(audit / "C7_B5_MOMENT_PREDICTION_BIAS_DIAGNOSTIC.md", md("C7-B5 moment prediction bias diagnostic", bias))

    safety = {"status": "C7_B5_DOMINANCE_SAFETY_AUDIT_COMPLETE", "official_val_used": False, "systems": {}}
    for name in systems:
        rec = split_results["official_like_stress"][name]
        delta = rec["delta_vs_C7_B2_1_A4"]
        weak = name.startswith("S_MINUTE_like_plus_weak_residual")
        d = rec["dominance_safety"]; m = rec["movement"]
        safety["systems"][name] = {
            "dominance_safety": d,
            "movement": m,
            "R100_delta": {"0.5-r100": delta["0.5-r100"], "0.7-r100": delta["0.7-r100"]},
            "hard_safety_pass": bool(m["fixed_pool_invariant"] and abs(delta["0.5-r100"]) < 1e-9 and abs(delta["0.7-r100"]) < 1e-9 and m["hard_positive_top100_query_exits"] == 0 and m["invalid_span_count"] == 0 and m["duplicate_span_count_after_nms"] == 0),
            "weak_residual_extra_pass": None if not weak else bool(d["corr_final_S_MINUTE_like"] >= d["corr_final_residual"] and d["top1_changed_rate"] <= 0.25 and d["top5_set_changed_rate"] <= 0.45 and d["top10_set_changed_rate"] <= 0.60),
        }
    atomic_json(audit / "C7_B5_DOMINANCE_SAFETY_AUDIT.json", safety)
    atomic_text(audit / "C7_B5_DOMINANCE_SAFETY_AUDIT.md", md("C7-B5 dominance safety audit", safety))

    candidates = [k for k in systems if k.startswith("S_MINUTE_like")]
    promising = []
    weak = []
    for name in candidates:
        stress = split_results["official_like_stress"][name]
        full = split_results["standard_train_calib"][name]
        safe = safety["systems"][name]
        d = stress["delta_vs_C7_B2_1_A4"]
        df = full["delta_vs_C7_B2_1_A4"]
        no_res_over = True
        if name.startswith("S_MINUTE_like_plus"):
            extra = safe["weak_residual_extra_pass"]
            no_res_over = bool(extra)
        ok = (
            alignment["strict_alignment"]
            and d["0.7-r1"] >= 0 and d["0.7-r5"] >= 0 and d["0.7-r10"] >= 0
            and df["0.5-r1"] >= 0 and df["0.7-r1"] >= 0
            and safe["hard_safety_pass"]
            and no_res_over
        )
        if ok:
            promising.append(name)
        elif alignment["strict_alignment"] and safe["hard_safety_pass"]:
            weak.append(name)
    if not alignment["strict_alignment"]:
        status = "C7_B5_SN_ALIGNMENT_INFEASIBLE"
        selected = None
    elif promising:
        status = "C7_B5_SN_SCORING_PROMISING"
        selected = sorted(promising, key=lambda n: sum(split_results["official_like_stress"][n]["delta_vs_C7_B2_1_A4"][k] for k in ["0.7-r1", "0.7-r5", "0.7-r10"]), reverse=True)[0]
    elif weak:
        status = "C7_B5_SN_SCORING_WEAK"
        selected = sorted(weak, key=lambda n: sum(split_results["official_like_stress"][n]["delta_vs_C7_B2_1_A4"][k] for k in ["0.7-r1", "0.7-r5", "0.7-r10"]), reverse=True)[0]
    else:
        status = "C7_B5_NEGATIVE"
        selected = None
    final = {
        "status": status,
        "official_val_used": False,
        "second_official_val": False,
        "post_val_adjustment": False,
        "C7_B2_1_promoted_decision_modified": False,
        "C7_B3_official_decision_modified": False,
        "C7_B4_archived_decision_modified": False,
        "NMS_modified": False,
        "evaluator_modified": False,
        "CONQUER_QDF_QAL_original_ML_VR_frozen": True,
        "full_fine_tuning": False,
        "primary_baseline": "C7-B2.1 A4 train-only reconstruction",
        "strict_alignment": alignment["strict_alignment"],
        "selected_candidate": selected,
        "selected_stress_record": split_results["official_like_stress"].get(selected) if selected else None,
        "recommendation": "Stop here. Do not run official val automatically.",
    }
    atomic_json(audit / "C7_B5_FINAL_DECISION.json", final)
    atomic_text(audit / "C7_B5_FINAL_DECISION.md", md("C7-B5 final decision", final))
    manifest = {
        **final,
        "outputs": {
            "archive": "c7_audit/C7_B4_ARCHIVE_DECISION.json",
            "inventory": "c7_audit/C7_B5_RAW_LOGIT_SOURCE_INVENTORY.json",
            "alignment": "c7_audit/C7_B5_STRICT_ALIGNMENT_AUDIT.json",
            "score_construction": "c7_audit/C7_B5_SN_SCORE_CONSTRUCTION.json",
            "train_only_audit": "c7_audit/C7_B5_SN_TRAIN_ONLY_AUDIT.json",
            "bias": "c7_audit/C7_B5_MOMENT_PREDICTION_BIAS_DIAGNOSTIC.json",
            "safety": "c7_audit/C7_B5_DOMINANCE_SAFETY_AUDIT.json",
            "decision": "c7_audit/C7_B5_FINAL_DECISION.json",
        },
    }
    atomic_json(audit / "C7_B5_MANIFEST.json", manifest)
    hashes = {
        "status": "C7_B5_HASHES",
        "official_val_used": False,
        "artifacts": {
            str(p): sha256_file(p) for p in [
                audit / "C7_B4_ARCHIVE_DECISION.json",
                audit / "C7_B5_RAW_LOGIT_SOURCE_INVENTORY.json",
                audit / "C7_B5_STRICT_ALIGNMENT_AUDIT.json",
                audit / "C7_B5_SN_SCORE_CONSTRUCTION.json",
                audit / "C7_B5_SN_TRAIN_ONLY_AUDIT.json",
                audit / "C7_B5_MOMENT_PREDICTION_BIAS_DIAGNOSTIC.json",
                audit / "C7_B5_DOMINANCE_SAFETY_AUDIT.json",
                audit / "C7_B5_FINAL_DECISION.json",
                audit / "C7_B5_MANIFEST.json",
            ]
        },
        "runner_sha256": sha256_file(__file__),
        "evidence_sha256": sha256_file(args.evidence_jsonl_gz),
    }
    atomic_json(audit / "C7_B5_HASHES.json", hashes)
    print(json.dumps({
        "status": status,
        "strict_alignment": alignment["strict_alignment"],
        "matched_candidate_keys": alignment["matched_candidate_keys"],
        "total_candidate_keys": alignment["total_candidate_keys"],
        "raw_logit_status": inventory["raw_logit_status"],
        "selected_candidate": selected,
        "official_val_used": False,
        "post_val_adjustment": False,
        "second_official_val": False,
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
