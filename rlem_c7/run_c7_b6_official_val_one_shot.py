#!/usr/bin/env python
"""Exactly-one official-val one-shot for C7-B6 R1SelectiveTop1.

Fixed mechanism:
  C7-B2.1 A4 official fixed-pool ranking is the safety anchor.
  C7-B6 may only move the S_MINUTE-like top candidate to rank1 when the
  frozen selective gate authorizes the replacement.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rlem.c4_lite_utils import flatten_eval_metrics  # noqa: E402
from rlem_c7.run_c7_b0_b1 import (  # noqa: E402
    atomic_json,
    atomic_npz,
    atomic_text,
    artifact,
    labels_for_sequence,
    load_npz,
    metric_delta,
    nms_sequence,
    selected_metrics,
)
from rlem_c7.run_c7_b2_1_safety_repair import group_score_map, normalize_cand  # noqa: E402
from rlem_c7.run_c7_b6_r1_selective_top1 import GateMLP, cand_id, overlap_idx, rank_array, video_rank_in_order  # noqa: E402
from standalone_eval.eval import eval_retrieval, load_jsonl  # noqa: E402


METRIC_KEYS = [
    "0.5-r1", "0.5-r5", "0.5-r10", "0.5-r100",
    "0.7-r1", "0.7-r5", "0.7-r10", "0.7-r100",
]
CLIP = 1.5
Candidate = Tuple[int, int, int, int, float, int]


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_obj(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")).hexdigest()


def write_jsonl(path: str | Path, rows: List[Dict[str, Any]]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(p) + ".partial")
    with tmp.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n")
    os.replace(tmp, p)


def load_metrics(path: str | Path, preferred_key: str | None = None) -> Dict[str, float]:
    obj = json.loads(Path(path).read_text(encoding="utf-8"))
    if preferred_key and preferred_key in obj:
        flat = obj[preferred_key]
    elif "C7_B1_official_metrics" in obj:
        flat = obj["C7_B1_official_metrics"]
    elif "C7_B2_1_A4_official_metrics" in obj:
        flat = obj["C7_B2_1_A4_official_metrics"]
    elif "VCMR" in obj:
        flat = obj["VCMR"]
    else:
        flat = flatten_eval_metrics(obj)
    return {k: float(flat[k]) for k in METRIC_KEYS}


def materialize_npz(obj: Any) -> Dict[str, np.ndarray]:
    if isinstance(obj, dict):
        return {k: obj[k] for k in obj.keys()}
    return {k: obj[k] for k in obj.files}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--allow_official_val", action="store_true", required=True)
    p.add_argument("--b6_final_decision", default="c7_audit/C7_B6_FINAL_DECISION.json")
    p.add_argument("--b6_manifest", default="c7_audit/C7_B6_MANIFEST.json")
    p.add_argument("--b6_model", default="c7_models/c7_b6_r1_selective_gate.pt")
    p.add_argument("--official_raw_logits", default="results/rlem_c7_b6_official_val/aux/official_val_fixed_pool_raw_logits.npz")
    p.add_argument("--official_raw_export_audit", default="c7_audit/C7_B6_OFFICIAL_FIXED_POOL_RAW_LOGIT_EXPORT.json")
    p.add_argument("--official_cache_npz", default="results/rlem_c6_b1_lite_r1_safe/official_val_c6_cache.npz")
    p.add_argument("--b21_anchor_states", default="results/rlem_c7_b2_1_official_val/aux/official_val_c7_b1_anchor_states.json")
    p.add_argument("--b21_video_scores", default="results/rlem_c7_b2_1_official_val/aux/official_val_video_scores.npz")
    p.add_argument("--b21_proposal_scores", default="results/rlem_c7_b2_1_official_val/aux/official_val_c7_b1_proposal_scores_for_video.npz")
    p.add_argument("--b21_submission", default="results/rlem_c7_b2_1_official_val/official_val_submission.json")
    p.add_argument("--b2_eval", default="c7_audit/C7_B2_TRAIN_CALIB_EVAL.json")
    p.add_argument("--c6_b1_metrics", default="results/rlem_c6_b1_lite_r1_safe/official_val_metrics.json")
    p.add_argument("--c6_b2_metrics", default="results/rlem_c6_b2_official_val/official_val_metrics.json")
    p.add_argument("--c7_b1_metrics", default="c7_audit/C7_B1_OFFICIAL_MANIFEST.json")
    p.add_argument("--b21_official_manifest", default="c7_audit/C7_B2_1_OFFICIAL_MANIFEST.json")
    p.add_argument("--gt_jsonl", default="/home/a/yybwork/data/yyb/tvr_feature_release/data/tvr_val_release.jsonl")
    p.add_argument("--output_dir", default="results/rlem_c7_b6_official_val")
    p.add_argument("--audit_dir", default="c7_audit")
    p.add_argument("--device", default="cuda")
    p.add_argument("--effective_top_n", type=int, default=100)
    p.add_argument("--max_after_nms", type=int, default=100)
    p.add_argument("--nms_thd", type=float, default=0.7)
    return p.parse_args()


def output_paths(args: argparse.Namespace) -> Dict[str, Path]:
    out = Path(args.output_dir)
    audit = Path(args.audit_dir)
    return {
        "scores_npz": out / "official_val_scores.npz",
        "predictions_jsonl": out / "official_val_predictions.jsonl",
        "submission_json": out / "official_val_submission.json",
        "metrics_json": out / "official_val_metrics.json",
        "one_shot_md": audit / "C7_B6_OFFICIAL_VAL_ONE_SHOT.md",
        "one_shot_json": audit / "C7_B6_OFFICIAL_VAL_ONE_SHOT.json",
        "decision_md": audit / "C7_B6_OFFICIAL_DECISION.md",
        "decision_json": audit / "C7_B6_OFFICIAL_DECISION.json",
        "manifest_json": audit / "C7_B6_OFFICIAL_MANIFEST.json",
        "hashes_json": audit / "C7_B6_OFFICIAL_HASHES.json",
    }


def ensure_no_prior_outputs(paths: Dict[str, Path]) -> None:
    keys = [
        "scores_npz", "predictions_jsonl", "submission_json", "metrics_json",
        "one_shot_md", "one_shot_json", "decision_md", "decision_json",
        "manifest_json", "hashes_json",
    ]
    existing = [str(paths[k]) for k in keys if paths[k].exists()]
    if existing:
        raise FileExistsError("refusing second C7-B6 official-val run; outputs already exist: " + ", ".join(existing))


def validate_ready(args: argparse.Namespace) -> Dict[str, Any]:
    decision = json.loads(Path(args.b6_final_decision).read_text(encoding="utf-8"))
    manifest = json.loads(Path(args.b6_manifest).read_text(encoding="utf-8"))
    raw_audit = json.loads(Path(args.official_raw_export_audit).read_text(encoding="utf-8"))
    if decision.get("status") != "C7_B6_R1_SELECTIVE_PROMISING":
        raise ValueError("C7-B6 train-only decision is not promising")
    if decision.get("official_val_used") is not False or manifest.get("official_val_used") is not False:
        raise ValueError("C7-B6 train-only state already indicates official_val_used")
    if decision.get("selected_source") != "phase_c_gate_model":
        raise ValueError("C7-B6 selected source mismatch")
    expected = {"K_guard": 10, "tau": 0.20000000000000007, "model": "GateMLP_15_32_16_3", "train_split": "calib_A+calib_B", "tau_tune_reference": "calib_C/stress train-only"}
    if decision.get("selected_config") != expected:
        raise ValueError("C7-B6 selected config mismatch")
    if raw_audit.get("status") != "C7_B6_OFFICIAL_FIXED_POOL_RAW_LOGIT_EXPORT_PASS":
        raise ValueError("official fixed-pool raw-logit export is not PASS")
    if raw_audit.get("matched_candidates") != raw_audit.get("total_candidates") or raw_audit.get("missing_candidates") != 0:
        raise ValueError("official raw-logit export is not complete")
    b21 = json.loads(Path(args.b21_official_manifest).read_text(encoding="utf-8"))
    if b21.get("official_val_used") is not True:
        raise ValueError("C7-B2.1 official anchor manifest is unavailable")
    return {"decision": decision, "manifest": manifest, "raw_audit": raw_audit, "b21": b21}


def validate_aligned(states: List[Dict[str, Any]], raw: Dict[str, np.ndarray], prop: Dict[str, np.ndarray]) -> Dict[str, Any]:
    n = len(states) * 100
    mismatch = 0
    prop_rows = prop["row"].astype(np.int64)
    prop_unique = len(np.unique(prop_rows)) == len(prop_rows)
    prop_row_set = set(int(x) for x in prop_rows)
    prop_missing = 0
    for q, state in enumerate(states):
        for j, x in enumerate(state["flat_candidates"][:100]):
            c = normalize_cand(x)
            i = q * 100 + j
            ok = (
                int(raw["query_index"][i]) == q
                and int(raw["group_id"][i]) == int(c[0])
                and int(raw["video_idx"][i]) == int(c[1])
                and int(raw["start_idx"][i]) == int(c[2])
                and int(raw["end_idx"][i]) == int(c[3])
                and int(raw["candidate_row_id"][i]) == int(c[5])
            )
            mismatch += int(not ok)
            prop_missing += int(int(c[5]) not in prop_row_set)
    finite = all(int(np.isnan(raw[k]).sum()) == 0 for k in ["b_start_logit", "e_end_logit", "p_b_i", "p_e_j", "r1"])
    return {
        "strict_alignment": bool(mismatch == 0 and prop_missing == 0 and prop_unique and finite and len(raw["query_index"]) == n and len(prop["query"]) == n),
        "total_candidate_keys": int(n),
        "raw_identity_mismatch_count": int(mismatch),
        "proposal_missing_by_candidate_row_id": int(prop_missing),
        "proposal_row_unique": bool(prop_unique),
        "proposal_join": "candidate_row_id keyed lookup",
        "raw_nan_free": bool(finite),
        "proposal_rows": int(len(prop["query"])),
        "raw_rows": int(len(raw["query_index"])),
    }


def move_to_front(order: np.ndarray, idx: int) -> np.ndarray:
    idx = int(idx)
    if int(order[0]) == idx:
        return order.copy()
    return np.concatenate([np.asarray([idx], dtype=order.dtype), order[order != idx]])


def rank_preserving_candidates(anchor: List[Candidate], order: np.ndarray, base_scores: np.ndarray) -> List[Candidate]:
    out: List[Candidate] = []
    n = len(order)
    for pos, idx in enumerate(order):
        c = anchor[int(idx)]
        score = float(n - pos) + 1e-6 * float(base_scores[int(idx)])
        out.append((c[0], c[1], c[2], c[3], score, c[5]))
    return out


def make_feature(rec: Dict[str, Any], ckpt: Dict[str, Any]) -> np.ndarray:
    vals = np.asarray([
        rec["rank_A4_SN"], rec["rank_SN_A4"], rec["a4_margin"], rec["sn_margin"],
        rec["sn_delta_top"], rec["a4_delta_top"], rec["video_rank_SN"], rec["video_rank_A4"],
        rec["duration"], rec["overlap_SN_A4"], rec["b_end_margin"],
        rec["proposal_confidence"], rec["span_quality"], rec["residual_SN"], rec["residual_A4"],
    ], dtype=np.float32)
    return (vals - ckpt["mean"].astype(np.float32)) / np.maximum(ckpt["std"].astype(np.float32), 1e-6)


@torch.no_grad()
def selected_sequences(args: argparse.Namespace, paths: Dict[str, Path], cache: Dict[str, np.ndarray], c7b21_submission: Dict[str, Any]) -> Dict[str, Any]:
    states = json.loads(Path(args.b21_anchor_states).read_text(encoding="utf-8"))["states"]
    raw = materialize_npz(load_npz(args.official_raw_logits, allow_pickle=False))
    prop = materialize_npz(load_npz(args.b21_proposal_scores, allow_pickle=False))
    alignment = validate_aligned(states, raw, prop)
    if not alignment["strict_alignment"]:
        raise ValueError(f"C7-B6 official alignment failed: {alignment}")
    prop_index_by_row = {int(row): i for i, row in enumerate(prop["row"].astype(np.int64))}
    score_arrays = load_npz(args.b21_video_scores, allow_pickle=False)
    cfg = json.loads(Path(args.b2_eval).read_text(encoding="utf-8"))["best"]["config"]
    gscore = group_score_map(score_arrays, cfg)
    video_weight = float(cfg["video_weight"])
    ckpt = torch.load(args.b6_model, map_location="cpu")
    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    model = GateMLP(len(ckpt["feature_names"])).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    k_guard = int(ckpt["best_config"]["K_guard"])
    tau = float(ckpt["best_config"]["tau"])
    gt_vid = cache["video_idx"][cache["desc_offsets"][:-1]].astype(np.int64)
    gt_s = cache["gt_start_idx"][cache["desc_offsets"][:-1]].astype(np.int64)
    gt_e = cache["gt_end_idx"][cache["desc_offsets"][:-1]].astype(np.int64)

    predictions_jsonl: List[Dict[str, Any]] = []
    policy_predictions = np.full((len(states), args.max_after_nms, 4), -1, dtype=np.float32)
    final_scores = np.full((len(states), args.effective_top_n), np.nan, dtype=np.float32)
    gate_margin = np.full(len(states), np.nan, dtype=np.float32)
    override_applied = np.zeros(len(states), dtype=bool)
    rank_a4_sn_all = np.full(len(states), -1, dtype=np.int16)
    top1_changed = top5_changed = top10_changed = 0
    exits = entries = invalid = dup = denom_pos = 0
    labels05: List[List[bool]] = []
    labels07: List[List[bool]] = []
    base05: List[List[bool]] = []
    base07: List[List[bool]] = []
    fixed_pool = True

    for q, state in enumerate(states):
        anchor = [normalize_cand(x) for x in state["flat_candidates"][:args.effective_top_n]]
        base_score = np.asarray([float(c[4]) for c in anchor], dtype=np.float32)
        residual = np.asarray([video_weight * gscore.get(int(c[0]), (0.0, 0.0, 0.0, 0.0))[2] for c in anchor], dtype=np.float32)
        a4_scores = base_score + residual
        x = slice(q * 100, q * 100 + len(anchor))
        sn_scores = (np.log(np.maximum(raw["r1"][x].astype(np.float32), 1e-12)) + raw["b_start_logit"][x] + raw["e_end_logit"][x]).astype(np.float32)
        a4_order = np.argsort(-a4_scores, kind="stable")
        sn_order = np.argsort(-sn_scores, kind="stable")
        a4_rank = rank_array(a4_scores)
        sn_rank = rank_array(sn_scores)
        a4_top = int(a4_order[0])
        sn_top = int(sn_order[0])
        rec = {
            "rank_A4_SN": float(a4_rank[sn_top]),
            "rank_SN_A4": float(sn_rank[a4_top]),
            "a4_margin": float(a4_scores[a4_order[0]] - a4_scores[a4_order[1]]),
            "sn_margin": float(sn_scores[sn_order[0]] - sn_scores[sn_order[1]]),
            "sn_delta_top": float(sn_scores[sn_top] - sn_scores[a4_top]),
            "a4_delta_top": float(a4_scores[a4_top] - a4_scores[sn_top]),
            "video_rank_SN": float(video_rank_in_order({"anchor": anchor}, a4_order, sn_top)),
            "video_rank_A4": float(video_rank_in_order({"anchor": anchor}, a4_order, a4_top)),
            "duration": float(anchor[sn_top][3] - anchor[sn_top][2] + 1),
            "overlap_SN_A4": float(overlap_idx(anchor[sn_top], anchor[a4_top])),
            "b_end_margin": float((raw["b_start_logit"][q * 100 + sn_top] + raw["e_end_logit"][q * 100 + sn_top]) - (raw["b_start_logit"][q * 100 + a4_top] + raw["e_end_logit"][q * 100 + a4_top])),
            "proposal_confidence": float(prop["proposal_confidence"][prop_index_by_row[int(anchor[sn_top][5])]]),
            "span_quality": float(prop["span_quality"][prop_index_by_row[int(anchor[sn_top][5])]]),
            "residual_SN": float(residual[sn_top]),
            "residual_A4": float(residual[a4_top]),
        }
        feat = torch.from_numpy(make_feature(rec, ckpt)[None]).to(device)
        probs = torch.sigmoid(model(feat)).float().cpu().numpy()[0]
        margin = float(probs[0] - probs[2])
        gate_margin[q] = margin
        rank_a4_sn_all[q] = int(a4_rank[sn_top])
        allowed = bool(margin > tau and int(a4_rank[sn_top]) <= k_guard)
        override_applied[q] = allowed
        final_order = move_to_front(a4_order, sn_top) if allowed else a4_order
        base_pre = rank_preserving_candidates(anchor, a4_order, a4_scores)
        final_pre = rank_preserving_candidates(anchor, final_order, a4_scores)
        fixed_pool = fixed_pool and sorted(cand_id(c) for c in base_pre) == sorted(cand_id(c) for c in final_pre)
        base_ids = [cand_id(c) for c in base_pre]
        final_ids = [cand_id(c) for c in final_pre]
        top1_changed += int(base_ids[:1] != final_ids[:1])
        top5_changed += int(set(base_ids[:5]) != set(final_ids[:5]))
        top10_changed += int(set(base_ids[:10]) != set(final_ids[:10]))
        base_post = nms_sequence(base_pre[:args.effective_top_n], args.nms_thd, args.max_after_nms)
        final_post = nms_sequence(final_pre[:args.effective_top_n], args.nms_thd, args.max_after_nms)
        l05, l07, pos, inv, du = labels_for_sequence(final_post, int(gt_vid[q]), int(gt_s[q]), int(gt_e[q]))
        a05, a07, apos, _ai, _ad = labels_for_sequence(base_post, int(gt_vid[q]), int(gt_s[q]), int(gt_e[q]))
        labels05.append(l05); labels07.append(l07); base05.append(a05); base07.append(a07)
        denom_pos += int(apos)
        exits += int(apos and not pos)
        entries += int((not apos) and pos)
        invalid += inv
        dup += du
        preds = []
        for r, c in enumerate(final_post):
            preds.append([int(c[1]), float(c[2] * CLIP), float((c[3] + 1) * CLIP), float(c[4])])
            policy_predictions[q, r] = np.asarray([c[1], c[2], c[3], c[4]], dtype=np.float32)
        for r, c in enumerate(final_pre[:args.effective_top_n]):
            final_scores[q, r] = float(c[4])
        desc_id = cache["desc_ids"][q].item() if hasattr(cache["desc_ids"][q], "item") else cache["desc_ids"][q]
        predictions_jsonl.append({"desc_id": desc_id, "desc": str(cache["desc_text"][q]), "predictions": preds})

    submission = {"video2idx": c7b21_submission["video2idx"], "VCMR": predictions_jsonl}
    atomic_json(paths["submission_json"], submission)
    write_jsonl(paths["predictions_jsonl"], predictions_jsonl)
    atomic_npz(
        paths["scores_npz"],
        policy_predictions=policy_predictions,
        final_score=final_scores,
        gate_margin=gate_margin,
        override_applied=override_applied.astype(np.int8),
        rank_A4_SN=rank_a4_sn_all,
        K_guard=np.asarray([k_guard], dtype=np.int16),
        tau=np.asarray([tau], dtype=np.float32),
    )
    n = max(len(states), 1)
    internal = {
        "metrics_internal": selected_metrics(labels05, labels07),
        "baseline_metrics_internal": selected_metrics(base05, base07),
        "gain_loss_vs_baseline": {
            f"{thr}-r{r}": {
                "gain_queries": int(sum((not any(b[:r])) and any(c[:r]) for b, c in zip(base, cur))),
                "loss_queries": int(sum(any(b[:r]) and (not any(c[:r])) for b, c in zip(base, cur))),
            }
            for thr, cur, base in [("0.5", labels05, base05), ("0.7", labels07, base07)] for r in [1, 5, 10]
        },
    }
    return {
        "submission": submission,
        **internal,
        "fixed_pool_invariant_on_official": bool(fixed_pool),
        "override_query_count": int(override_applied.sum()),
        "override_query_rate": float(override_applied.sum() / n),
        "top1_changed_rate": float(top1_changed / n),
        "top5_set_changed_rate": float(top5_changed / n),
        "top10_set_changed_rate": float(top10_changed / n),
        "hard_positive_top100_query_exits": int(exits),
        "hard_positive_top100_query_entries": int(entries),
        "hard_positive_exit_ratio": float(exits / max(denom_pos, 1)),
        "invalid_span_count": int(invalid),
        "duplicate_span_count_after_nms": int(dup),
        "queries": int(len(states)),
        "alignment": alignment,
        "selected_config": {"K_guard": k_guard, "tau": tau, "model": "GateMLP_15_32_16_3"},
        "device": str(device),
    }


def classify(delta_vs_b21: Dict[str, float], structural: Dict[str, Any]) -> str:
    infra_ok = (
        structural["fixed_pool_invariant_on_official"]
        and structural["invalid_span_count"] == 0
        and structural["duplicate_span_count_after_nms"] == 0
        and structural["hard_positive_top100_query_exits"] == 0
    )
    if not infra_ok:
        return "C7_B6_INFRA_FAIL"
    r1_ok = delta_vs_b21["0.7-r1"] > 0.0 and delta_vs_b21["0.5-r1"] > 0.0
    r5_r10_ok = delta_vs_b21["0.7-r5"] >= -0.10 and delta_vs_b21["0.7-r10"] >= -0.10 and delta_vs_b21["0.5-r5"] >= -0.10 and delta_vs_b21["0.5-r10"] >= -0.10
    if r1_ok and r5_r10_ok:
        return "C7_B6_OFFICIAL_PROMOTED"
    if r1_ok:
        return "C7_B6_R1_TRADEOFF_NO_PROMOTION"
    return "C7_B6_OFFICIAL_NEGATIVE"


def metric_table(rows: Dict[str, Dict[str, float]]) -> str:
    out = [
        "| system | 0.5-r1 | 0.5-r5 | 0.5-r10 | 0.5-r100 | 0.7-r1 | 0.7-r5 | 0.7-r10 | 0.7-r100 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, vals in rows.items():
        fmt = "{:+.2f}" if name.startswith("Delta") else "{:.2f}"
        out.append("| " + name + " | " + " | ".join(fmt.format(vals[k]) for k in METRIC_KEYS) + " |")
    return "\n".join(out)


def main() -> None:
    args = parse_args()
    if not args.allow_official_val:
        raise ValueError("--allow_official_val is required")
    paths = output_paths(args)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.audit_dir).mkdir(parents=True, exist_ok=True)
    ensure_no_prior_outputs(paths)
    ready_state = validate_ready(args)
    cache = load_npz(args.official_cache_npz, allow_pickle=True)
    c7b21_submission = json.loads(Path(args.b21_submission).read_text(encoding="utf-8"))
    structural = selected_sequences(args, paths, cache, c7b21_submission)

    # Official evaluator: exactly one call in this script.
    metrics_raw = eval_retrieval(
        structural["submission"],
        load_jsonl(args.gt_jsonl),
        iou_thds=(0.5, 0.7),
        verbose=False,
        match_number=True,
        use_desc_type=True,
    )
    atomic_json(paths["metrics_json"], metrics_raw)
    c7b6_metrics = {k: float(flatten_eval_metrics(metrics_raw)[k]) for k in METRIC_KEYS}
    c6b1_metrics = load_metrics(args.c6_b1_metrics)
    c6b2_metrics = load_metrics(args.c6_b2_metrics)
    c7b1_metrics = load_metrics(args.c7_b1_metrics, "C7_B1_official_metrics")
    c7b21_metrics = load_metrics(args.b21_official_manifest, "C7_B2_1_A4_official_metrics")
    delta_vs_c6b1 = metric_delta(c7b6_metrics, c6b1_metrics)
    delta_vs_c6b2 = metric_delta(c7b6_metrics, c6b2_metrics)
    delta_vs_c7b1 = metric_delta(c7b6_metrics, c7b1_metrics)
    delta_vs_b21 = metric_delta(c7b6_metrics, c7b21_metrics)
    status = classify(delta_vs_b21, structural)
    common = {
        "stage": "C7-B6",
        "status": status,
        "status_before_run": "C7_B6_R1_SELECTIVE_PROMISING",
        "method": "R1-oriented selective top1 override",
        "selected_source": "phase_c_gate_model",
        "selected_config": structural["selected_config"],
        "date": datetime.now().strftime("%Y-%m-%d"),
        "official_val_used": True,
        "official_val_run": True,
        "official_evaluator_call_count": 1,
        "post_val_adjustment": False,
        "second_official_val": False,
        "training_started": False,
        "config_selection_changed": False,
        "top100_full_rerank_promoted_candidate": False,
        "S_MINUTE_like_direct_replacement": False,
        "evaluator_modified": False,
        "nms_modified": False,
        "C7_B2_1_promoted_decision_modified": False,
        "C7_B3_C7_B4_C7_B5_1_archived_decisions_modified": False,
        "CONQUER_QDF_QAL_original_ML_VR_frozen": True,
        "C6_B1_official_metrics": c6b1_metrics,
        "C6_B2_official_metrics": c6b2_metrics,
        "C7_B1_official_archived_metrics": c7b1_metrics,
        "C7_B2_1_A4_official_metrics": c7b21_metrics,
        "C7_B6_official_metrics": c7b6_metrics,
        "delta_vs_C6_B1": delta_vs_c6b1,
        "delta_vs_C6_B2": delta_vs_c6b2,
        "delta_vs_C7_B1": delta_vs_c7b1,
        "delta_vs_C7_B2_1_A4": delta_vs_b21,
        "fixed_pool_invariant_on_official": structural["fixed_pool_invariant_on_official"],
        "override_query_count": structural["override_query_count"],
        "override_query_rate": structural["override_query_rate"],
        "top1_changed_rate": structural["top1_changed_rate"],
        "top5_set_changed_rate": structural["top5_set_changed_rate"],
        "top10_set_changed_rate": structural["top10_set_changed_rate"],
        "hard_positive_top100_query_exits": structural["hard_positive_top100_query_exits"],
        "hard_positive_top100_query_entries": structural["hard_positive_top100_query_entries"],
        "hard_positive_exit_ratio": structural["hard_positive_exit_ratio"],
        "invalid_span_count": structural["invalid_span_count"],
        "duplicate_span_count_after_nms": structural["duplicate_span_count_after_nms"],
        "gain_loss_vs_C7_B2_1_A4_internal": structural["gain_loss_vs_baseline"],
        "alignment": structural["alignment"],
        "ready_state": {
            "b6_final_decision": artifact(args.b6_final_decision),
            "b6_manifest": artifact(args.b6_manifest),
            "b6_model": artifact(args.b6_model),
            "official_raw_logits": artifact(args.official_raw_logits),
            "official_raw_export_audit": artifact(args.official_raw_export_audit),
            "b21_official_manifest": artifact(args.b21_official_manifest),
        },
        "final_instruction": "Stop after exactly-one official-val one-shot and wait for human review.",
    }
    atomic_json(paths["one_shot_json"], common)
    atomic_json(paths["manifest_json"], {**common, "promoted": status == "C7_B6_OFFICIAL_PROMOTED"})
    decision = {
        "status": status,
        "promoted": status == "C7_B6_OFFICIAL_PROMOTED",
        "official_val_used": True,
        "official_evaluator_call_count": 1,
        "post_val_adjustment": False,
        "second_official_val": False,
        "C7_B6_official_metrics": c7b6_metrics,
        "delta_vs_C6_B1": delta_vs_c6b1,
        "delta_vs_C6_B2": delta_vs_c6b2,
        "delta_vs_C7_B1": delta_vs_c7b1,
        "delta_vs_C7_B2_1_A4": delta_vs_b21,
        "fixed_pool_invariant_on_official": structural["fixed_pool_invariant_on_official"],
        "override_query_count": structural["override_query_count"],
        "top5_set_changed_rate": structural["top5_set_changed_rate"],
        "top10_set_changed_rate": structural["top10_set_changed_rate"],
        "hard_positive_top100_query_exits": structural["hard_positive_top100_query_exits"],
        "invalid_span_count": structural["invalid_span_count"],
        "duplicate_span_count_after_nms": structural["duplicate_span_count_after_nms"],
        "final_instruction": "Stop after exactly-one official-val one-shot and wait for human review.",
    }
    atomic_json(paths["decision_json"], decision)
    table = metric_table({
        "C6-B1 official": c6b1_metrics,
        "C6-B2 official": c6b2_metrics,
        "C7-B1 official archived": c7b1_metrics,
        "C7-B2.1 A4 official": c7b21_metrics,
        "C7-B6 official": c7b6_metrics,
        "Delta vs C6-B1": delta_vs_c6b1,
        "Delta vs C6-B2": delta_vs_c6b2,
        "Delta vs C7-B1": delta_vs_c7b1,
        "Delta vs C7-B2.1 A4": delta_vs_b21,
    })
    md = f"""# C7-B6 official-val one-shot

- Status: `{status}`
- method: `R1-oriented selective top1 override`
- selected_config: `{json.dumps(structural['selected_config'], sort_keys=True)}`
- official_val_used: `true`
- official evaluator calls: `1`
- post_val_adjustment / second_official_val: `false / false`
- evaluator_modified / nms_modified: `false / false`
- fixed_pool_invariant_on_official: `{structural['fixed_pool_invariant_on_official']}`
- override count/rate: `{structural['override_query_count']} / {structural['override_query_rate']}`
- top1/top5/top10 changed rate: `{structural['top1_changed_rate']} / {structural['top5_set_changed_rate']} / {structural['top10_set_changed_rate']}`
- hard_positive exits/entries/ratio: `{structural['hard_positive_top100_query_exits']} / {structural['hard_positive_top100_query_entries']} / {structural['hard_positive_exit_ratio']}`
- invalid_span_count / duplicate_span_count_after_nms: `{structural['invalid_span_count']} / {structural['duplicate_span_count_after_nms']}`

## Metrics

{table}

Stop here and wait for human review.
"""
    atomic_text(paths["one_shot_md"], md)
    atomic_text(paths["decision_md"], "# C7-B6 official decision\n\n" + md)
    hashes = {
        "status": "C7_B6_OFFICIAL_HASHES",
        "official_val_used": True,
        "post_val_adjustment": False,
        "second_official_val": False,
        "metric_hash": sha256_obj(c7b6_metrics),
        "prediction_hash": sha256_file(paths["predictions_jsonl"]),
        "score_hash": sha256_file(paths["scores_npz"]),
        "artifacts": {
            "one_shot_json": artifact(paths["one_shot_json"]),
            "one_shot_md": artifact(paths["one_shot_md"]),
            "decision_json": artifact(paths["decision_json"]),
            "decision_md": artifact(paths["decision_md"]),
            "manifest_json": artifact(paths["manifest_json"]),
            "scores_npz": artifact(paths["scores_npz"]),
            "predictions_jsonl": artifact(paths["predictions_jsonl"]),
            "submission_json": artifact(paths["submission_json"]),
            "metrics_json": artifact(paths["metrics_json"]),
            "runner": artifact(__file__),
            "b6_model": artifact(args.b6_model),
            "official_raw_logits": artifact(args.official_raw_logits),
            "official_raw_export_audit": artifact(args.official_raw_export_audit),
        },
    }
    atomic_json(paths["hashes_json"], hashes)
    print(json.dumps({
        "status": status,
        "C6_B1_official_metrics": c6b1_metrics,
        "C6_B2_official_metrics": c6b2_metrics,
        "C7_B1_official_archived_metrics": c7b1_metrics,
        "C7_B2_1_A4_official_metrics": c7b21_metrics,
        "C7_B6_official_metrics": c7b6_metrics,
        "delta_vs_C6_B1": delta_vs_c6b1,
        "delta_vs_C6_B2": delta_vs_c6b2,
        "delta_vs_C7_B1": delta_vs_c7b1,
        "delta_vs_C7_B2_1_A4": delta_vs_b21,
        "fixed_pool_invariant_on_official": structural["fixed_pool_invariant_on_official"],
        "override_query_count": structural["override_query_count"],
        "top5_set_changed_rate": structural["top5_set_changed_rate"],
        "top10_set_changed_rate": structural["top10_set_changed_rate"],
        "hard_positive_top100_query_exits": structural["hard_positive_top100_query_exits"],
        "hard_positive_top100_query_entries": structural["hard_positive_top100_query_entries"],
        "hard_positive_exit_ratio": structural["hard_positive_exit_ratio"],
        "invalid_span_count": structural["invalid_span_count"],
        "duplicate_span_count_after_nms": structural["duplicate_span_count_after_nms"],
        "evaluator_modified": False,
        "nms_modified": False,
        "official_val_used": True,
        "post_val_adjustment": False,
        "second_official_val": False,
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
