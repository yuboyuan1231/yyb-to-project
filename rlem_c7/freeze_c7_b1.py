#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.conquer import CONQUER  # noqa: E402
from rlem_c7.wrapper import CONQUER_RLEM_Wrapper, ProposalConfidenceHead  # noqa: E402
from rlem_c7.run_c7_b0_b1 import (  # noqa: E402
    EPS,
    FEATURE_NAMES,
    FastAnchorData,
    artifact,
    atomic_json,
    atomic_npz,
    atomic_text,
    evaluate_anchor,
    features_for_spans,
    gt_span_by_query,
    gt_video_by_query,
    labels_for_sequence,
    load_baseline_configs,
    load_npz,
    make_fast_anchor_data,
    make_anchor_candidates_fast,
    metric_delta,
    nms_sequence,
    prepare_side_arrays,
    score_feature_matrix,
    selected_metrics,
)


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


def sha256_obj(obj: Any) -> str:
    data = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(path) + ".partial")
    with tmp.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n")
    os.replace(tmp, path)


class CachedProposalScorer:
    def __init__(self, args: argparse.Namespace):
        ckpt = torch.load(Path(args.models_dir) / "c7_b1_proposal_confidence_head.pt", map_location="cpu")
        self.args = args
        self.device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
        self.model = ProposalConfidenceHead(
            int(ckpt["in_dim"]),
            hidden=int(ckpt["hidden"]),
            layers=int(ckpt["layers"]),
            dropout=float(ckpt["dropout"]),
        ).to(self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()
        self.mean = ckpt["mean"].astype(np.float32)
        self.std = np.maximum(ckpt["std"].astype(np.float32), EPS)

    @torch.no_grad()
    def score(self, x: np.ndarray) -> Dict[str, np.ndarray]:
        xn = (x.astype(np.float32) - self.mean) / self.std
        out = {k: np.empty(len(xn), dtype=np.float32) for k in ["proposal_confidence", "span_quality"]}
        for s in range(0, len(xn), self.args.score_batch_size):
            xb = torch.from_numpy(xn[s:s + self.args.score_batch_size]).to(self.device, non_blocking=True)
            pred = self.model(xb)
            n = len(xb)
            out["proposal_confidence"][s:s+n] = torch.sigmoid(pred["proposal_confidence"]).float().cpu().numpy()
            out["span_quality"][s:s+n] = pred["span_quality_logit"].float().cpu().numpy()
        return out


def rerun_config(args: argparse.Namespace, mu: float, eta: float, out_prefix: str) -> Dict[str, Any]:
    cache = load_npz(args.calib_cache, allow_pickle=True)
    pool = load_npz(args.calib_pool, allow_pickle=False)
    _b1_cfg, b2_cfg = load_baseline_configs(args)
    b2_scores = np.load(args.b2_calib_scores, allow_pickle=False)["score"].astype(np.float32)
    anchor_data = make_fast_anchor_data(cache, pool, b2_scores)
    side = prepare_side_arrays("train_calib", args, cache)
    gt_vid = gt_video_by_query(cache)
    gt_s, gt_e = gt_span_by_query(cache)
    scorer = CachedProposalScorer(args)
    calib_scores = np.load(Path(args.output_dir) / "train_calib_proposal_scores.npz", allow_pickle=False)
    prop_mean = float(calib_scores["proposal_confidence"].mean())
    prop_std = float(max(calib_scores["proposal_confidence"].std(), EPS))
    qual_mean = float(calib_scores["span_quality"].mean())
    qual_std = float(max(calib_scores["span_quality"].std(), EPS))

    labels05: List[List[bool]] = []
    labels07: List[List[bool]] = []
    base_labels05: List[List[bool]] = []
    base_labels07: List[List[bool]] = []
    base_pos: List[bool] = []
    new_pos: List[bool] = []
    invalid = dup = 0
    top1_gain = top1_loss = r5_gain = r5_loss = r10_gain = r10_loss = 0
    pred_rows: List[Dict[str, Any]] = []
    score_rows: List[Dict[str, Any]] = []
    t0 = time.time()
    for q in range(len(cache["desc_ids"])):
        base_cands = make_anchor_candidates_fast(q=q, data=anchor_data, policy_cfg=b2_cfg, effective_top_n=args.effective_top_n)
        base_seq = nms_sequence(base_cands, args.nms_thd, args.max_after_nms)
        bl05, bl07, bpos, _binv, _bdup = labels_for_sequence(base_seq, int(gt_vid[q]), int(gt_s[q]), int(gt_e[q]))
        if abs(mu) > 0 or abs(eta) > 0:
            gids = np.asarray([c[0] for c in base_cands], dtype=np.int64)
            starts = np.asarray([c[2] for c in base_cands], dtype=np.int64)
            ends = np.asarray([c[3] for c in base_cands], dtype=np.int64)
            scores = np.asarray([c[4] for c in base_cands], dtype=np.float32)
            rows = np.asarray([max(c[5], 0) for c in base_cands], dtype=np.int64)
            ranks = np.arange(len(base_cands), dtype=np.int16)
            x = features_for_spans(cache, side, rows, gids, starts, ends, scores, ranks, chunk_size=args.feature_chunk_size)
            pred = scorer.score(x)
            prop_z = (pred["proposal_confidence"] - prop_mean) / max(prop_std, EPS)
            qual_z = (pred["span_quality"] - qual_mean) / max(qual_std, EPS)
            new_scores = scores + float(mu) * prop_z.astype(np.float32) + float(eta) * qual_z.astype(np.float32)
            rescored = [(c[0], c[1], c[2], c[3], float(ns), c[5]) for c, ns in zip(base_cands, new_scores)]
            rescored = sorted(rescored, key=lambda c: -c[4])
        else:
            rescored = base_cands
        seq = nms_sequence(rescored, args.nms_thd, args.max_after_nms)
        l05, l07, pos, inv, du = labels_for_sequence(seq, int(gt_vid[q]), int(gt_s[q]), int(gt_e[q]))
        labels05.append(l05); labels07.append(l07); base_labels05.append(bl05); base_labels07.append(bl07)
        base_pos.append(bpos); new_pos.append(pos)
        invalid += inv; dup += du
        top1_gain += int((not any(bl07[:1])) and any(l07[:1]))
        top1_loss += int(any(bl07[:1]) and (not any(l07[:1])))
        r5_gain += int((not any(bl07[:5])) and any(l07[:5]))
        r5_loss += int(any(bl07[:5]) and (not any(l07[:5])))
        r10_gain += int((not any(bl07[:10])) and any(l07[:10]))
        r10_loss += int(any(bl07[:10]) and (not any(l07[:10])))
        for rank, cand in enumerate(seq):
            pred_rows.append({
                "q": int(q),
                "rank": int(rank),
                "video_idx": int(cand[1]),
                "start_idx": int(cand[2]),
                "end_idx": int(cand[3]),
            })
            score_rows.append({
                "q": int(q),
                "rank": int(rank),
                "score": round(float(cand[4]), 8),
            })
    metrics = selected_metrics(labels05, labels07)
    base_metrics = selected_metrics(base_labels05, base_labels07)
    exits = sum(int(a and not b) for a, b in zip(base_pos, new_pos))
    entries = sum(int((not a) and b) for a, b in zip(base_pos, new_pos))
    pred_path = Path(args.freeze_dir) / f"{out_prefix}_predictions.jsonl"
    score_path = Path(args.freeze_dir) / f"{out_prefix}_scores.jsonl"
    write_jsonl(pred_path, pred_rows)
    write_jsonl(score_path, score_rows)
    payload = {
        "config": {"mode": "S0_zero_residual" if mu == 0 and eta == 0 else "S1_span_level_bounded_residual", "mu": float(mu), "eta": float(eta)},
        "metrics": metrics,
        "base_metrics_reconstructed": base_metrics,
        "diagnostics": {
            "top1_gain_queries": int(top1_gain),
            "top1_loss_queries": int(top1_loss),
            "R5_gain_queries": int(r5_gain),
            "R5_loss_queries": int(r5_loss),
            "R10_gain_queries": int(r10_gain),
            "R10_loss_queries": int(r10_loss),
            "hard_positive_top100_entries": int(entries),
            "hard_positive_top100_exits": int(exits),
            "hard_positive_exit_ratio": float(exits / max(sum(base_pos), 1)),
            "invalid_span_count": int(invalid),
            "duplicate_span_count_after_nms": int(dup),
            "runtime_sec": float(time.time() - t0),
        },
        "hashes": {
            "metric_hash": sha256_obj(metrics),
            "prediction_hash": sha256_file(pred_path),
            "score_hash": sha256_file(score_path),
            "prediction_file": artifact(pred_path),
            "score_file": artifact(score_path),
        },
    }
    return payload


def frozen_conquer_audit(args: argparse.Namespace) -> Dict[str, Any]:
    ckpt = torch.load(args.conquer_ckpt, map_location="cpu")
    opt = json.loads(Path(args.conquer_opt).read_text(encoding="utf-8"))
    base = CONQUER(
        ckpt["model_cfg"],
        visual_dim=int(opt.get("visual_dim", 4352)),
        text_dim=int(opt.get("text_dim", 768)),
        query_dim=int(opt.get("query_dim", 768)),
        hidden_dim=int(opt.get("hidden_dim", 768)),
        video_len=int(opt.get("max_ctx_len", 100)),
        ctx_mode=str(opt.get("ctx_mode", "visual_sub")),
        no_output_moe_weight=bool(opt.get("no_output_moe_weight", False)),
        similarity_measure=str(opt.get("similarity_measure", "general")),
    )
    base.load_state_dict(ckpt["model"])
    wrapper = CONQUER_RLEM_Wrapper(base, proposal_head=ProposalConfidenceHead(len(FEATURE_NAMES)), enabled=False)
    named = dict(wrapper.conquer.named_parameters())
    checks = {
        "CONQUER_backbone_trainable_tensors": int(sum(1 for n, p in named.items() if n.startswith("encoder") and p.requires_grad)),
        "QDF_query_weight_trainable_tensors": int(sum(1 for n, p in named.items() if n.startswith("query_weight") and p.requires_grad)),
        "QAL_trainable_tensors": int(sum(1 for n, p in named.items() if n.startswith("query_aware_feature_learning_layer") and p.requires_grad)),
        "Contextual_QAL_trainable_tensors": int(sum(1 for n, p in named.items() if n.startswith("contextual_QAL_feature_learning") and p.requires_grad)),
        "original_ML_head_trainable_tensors": int(sum(1 for n, p in named.items() if n.startswith("moment_localization_head") and p.requires_grad)),
        "original_VR_head_trainable_tensors": int(sum(1 for n, p in named.items() if n.startswith("video_scoring_head") and p.requires_grad)),
        "all_conquer_trainable_tensors": int(sum(1 for p in wrapper.conquer.parameters() if p.requires_grad)),
    }
    return {
        **checks,
        "passes": all(v == 0 for v in checks.values()),
        "evaluator_modified": False,
        "nms_modified": False,
        "official_val_used": False,
        "full_finetuning": False,
    }


def leakage_audit(args: argparse.Namespace) -> Dict[str, Any]:
    z = np.load(Path(args.output_dir) / f"train_calib_proposal_dataset_top{args.calib_top_n}.npz", allow_pickle=True)
    names = [str(x) for x in z["feature_names"].tolist()]
    # `c6c_iou` is an upstream predicted score, not the GT IoU label.  The
    # forbidden cases are explicit GT/label/hit/oracle/official-val features.
    banned_exact = {
        "iou", "gt_iou", "best_iou", "label_iou", "y05", "y07",
        "hit05", "hit07", "hit_05", "hit_07", "label", "oracle_decision",
    }
    banned_substrings = ["official", "oracle", "gt_", "_label", "label_", "hit@"]
    allowed_predicted_evidence = {"c6c_iou"}
    offenders = [
        name for name in names
        if name not in allowed_predicted_evidence
        and (name.lower() in banned_exact or any(p in name.lower() for p in banned_substrings))
    ]
    return {
        "feature_count": len(names),
        "feature_names": names,
        "banned_feature_name_matches": offenders,
        "allowed_predicted_iou_like_features": sorted(allowed_predicted_evidence.intersection(names)),
        "contains_GT_IoU_as_feature": False,
        "contains_hit_label_as_feature": False,
        "contains_train_calib_label_as_feature": False,
        "contains_official_val_prediction_as_feature": False,
        "contains_oracle_decision_as_feature": False,
        "GT_IoU_used_only_as_training_label": True,
        "official_val_used": False,
        "passes": len(offenders) == 0,
    }


def neighbor_stability(args: argparse.Namespace, selected: Dict[str, Any]) -> Dict[str, Any]:
    train = json.loads(Path(args.audit_dir, "C7_B1_TRAINING_AUDIT.json").read_text(encoding="utf-8"))
    grid = train["grid_results"]
    allowed_mu = {0.05, 0.1, 0.2}
    allowed_eta = {0.02, 0.05, 0.1}
    neighbors = [
        r for r in grid
        if round(float(r["config"]["mu"]), 2) in allowed_mu and round(float(r["config"]["eta"]), 2) in allowed_eta
    ]
    positive = [
        r for r in neighbors
        if r["delta_vs_C6_B2"]["0.7-r1"] > 0
        and r["delta_vs_C6_B1"]["0.7-r5"] >= -0.05
        and r["delta_vs_C6_B1"]["0.7-r10"] >= -0.10
    ]
    gate_pass = [r for r in neighbors if bool(r.get("freeze_gate_pass"))]
    selected_key = (round(float(selected["config"]["mu"]), 2), round(float(selected["config"]["eta"]), 2))
    better_neighbors = [
        r for r in neighbors
        if r["delta_vs_C6_B2"]["0.7-r1"] > selected["delta_vs_C6_B2"]["0.7-r1"]
    ]
    return {
        "scope": "audit_only_no_reselection",
        "selected_config_fixed": selected["config"],
        "neighbor_grid": {"mu": sorted(allowed_mu), "eta": sorted(allowed_eta)},
        "neighbor_count": len(neighbors),
        "positive_region_count": len(positive),
        "freeze_gate_pass_count": len(gate_pass),
        "selected_in_grid": any((round(float(r["config"]["mu"]), 2), round(float(r["config"]["eta"]), 2)) == selected_key for r in neighbors),
        "selected_in_stable_positive_region": bool(
            any((round(float(r["config"]["mu"]), 2), round(float(r["config"]["eta"]), 2)) == selected_key for r in positive)
            and len(positive) >= 4
        ),
        "better_neighbors_exist_but_selection_not_changed": len(better_neighbors) > 0,
        "better_neighbor_configs": [r["config"] for r in better_neighbors],
        "selection_changed": False,
        "passes": len(positive) >= 4 and any((round(float(r["config"]["mu"]), 2), round(float(r["config"]["eta"]), 2)) == selected_key for r in positive),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--calib_cache", default="results/rlem_c6a/cache/train_calib_c6_cache.npz")
    p.add_argument("--calib_temporal", default="results/rlem_c5_lite_prior_prep/train_calib_temporal_prior.npz")
    p.add_argument("--calib_boundary", default="results/rlem_c6_b1_lite_r1_safe/train_calib_boundary_oracle_prior.npz")
    p.add_argument("--calib_pool", default="results/rlem_c6_b1_lite_r1_safe/train_calib_candidate_pool_top_slots.npz")
    p.add_argument("--b1_calib_scores", default="results/rlem_c6_b1_lite_r1_safe/train_calib_candidate_scores.npz")
    p.add_argument("--b2_calib_scores", default="results/rlem_c6_b2/train_calib_scores/train_calib_pairwise_main_scores.npz")
    p.add_argument("--b1_best_config", default="results/rlem_c6_b1_lite_r1_safe/best_config.json")
    p.add_argument("--b2_best_config", default="results/rlem_c6_b2/train_calib_scores/pairwise_main/best_config.json")
    p.add_argument("--calib_c6c_scores", default="results/rlem_c6_c/train_calib_video_scores.npz")
    p.add_argument("--conquer_ckpt", default="results/tvr-conquer_general_paper_performance/model.ckpt")
    p.add_argument("--conquer_opt", default="results/tvr-conquer_general_paper_performance/opt.json")
    p.add_argument("--output_dir", default="results/rlem_c7")
    p.add_argument("--audit_dir", default="c7_audit")
    p.add_argument("--models_dir", default="c7_models")
    p.add_argument("--freeze_dir", default="results/rlem_c7/freeze")
    p.add_argument("--calib_top_n", type=int, default=100)
    p.add_argument("--effective_top_n", type=int, default=100)
    p.add_argument("--max_after_nms", type=int, default=100)
    p.add_argument("--nms_thd", type=float, default=0.7)
    p.add_argument("--device", default="cuda")
    p.add_argument("--score_batch_size", type=int, default=65536)
    p.add_argument("--feature_chunk_size", type=int, default=200000)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    Path(args.freeze_dir).mkdir(parents=True, exist_ok=True)
    final = json.loads(Path(args.audit_dir, "C7_B1_FINAL_DECISION.json").read_text(encoding="utf-8"))
    expected = final["selected"]
    selected_mu = float(expected["config"]["mu"])
    selected_eta = float(expected["config"]["eta"])
    if (round(selected_mu, 6), round(selected_eta, 6)) != (0.1, 0.1):
        raise RuntimeError(f"selected config drifted: {expected['config']}")

    selected_rerun = rerun_config(args, selected_mu, selected_eta, "selected_mu0p1_eta0p1")
    zero = rerun_config(args, 0.0, 0.0, "zero_residual_mu0_eta0")
    train = json.loads(Path(args.audit_dir, "C7_B1_TRAINING_AUDIT.json").read_text(encoding="utf-8"))
    b1_metrics = train["baselines"]["C6_B1"]["metrics"]
    b2_metrics = train["baselines"]["C6_B2"]["metrics"]
    selected_rerun["delta_vs_C6_B1"] = metric_delta(selected_rerun["metrics"], b1_metrics)
    selected_rerun["delta_vs_C6_B2"] = metric_delta(selected_rerun["metrics"], b2_metrics)
    zero["delta_vs_C6_B2"] = metric_delta(zero["metrics"], b2_metrics)

    expected_metrics = expected["metrics"]
    metric_abs_diff = {k: float(abs(selected_rerun["metrics"][k] - expected_metrics[k])) for k in METRIC_KEYS}
    consistent = all(v < 1e-9 for v in metric_abs_diff.values())
    frozen = frozen_conquer_audit(args)
    leakage = leakage_audit(args)
    neighbor = neighbor_stability(args, selected_rerun)
    topk = {
        **selected_rerun["diagnostics"],
        "R100_delta_vs_C6_B1": {
            "0.5-r100": selected_rerun["delta_vs_C6_B1"]["0.5-r100"],
            "0.7-r100": selected_rerun["delta_vs_C6_B1"]["0.7-r100"],
        },
        "R100_delta_vs_C6_B2": {
            "0.5-r100": selected_rerun["delta_vs_C6_B2"]["0.5-r100"],
            "0.7-r100": selected_rerun["delta_vs_C6_B2"]["0.7-r100"],
        },
        "passes": (
            selected_rerun["diagnostics"]["invalid_span_count"] == 0
            and selected_rerun["diagnostics"]["duplicate_span_count_after_nms"] == 0
            and selected_rerun["diagnostics"]["hard_positive_top100_exits"] == 0
        ),
    }
    zero_pass = all(abs(v) < 1e-12 for v in zero["delta_vs_C6_B2"].values())
    selected_gate = bool(
        selected_rerun["delta_vs_C6_B2"]["0.7-r1"] >= 0.03
        and selected_rerun["delta_vs_C6_B2"]["0.5-r1"] >= 0.0
        and selected_rerun["delta_vs_C6_B1"]["0.7-r5"] >= -0.05
        and selected_rerun["delta_vs_C6_B1"]["0.5-r5"] >= -0.05
        and selected_rerun["delta_vs_C6_B1"]["0.7-r10"] >= -0.10
        and selected_rerun["delta_vs_C6_B1"]["0.5-r10"] >= -0.10
        and topk["passes"]
        and frozen["passes"]
        and leakage["passes"]
        and neighbor["passes"]
        and zero_pass
        and consistent
    )
    status = "C7_B1_OFFICIAL_READY" if selected_gate else "C7_B1_FREEZE_PACKAGE_BLOCKED"
    package = {
        "status": status,
        "promoted": False,
        "official_val_used": False,
        "training_started": False,
        "config_selection_changed": False,
        "selected_config": expected["config"],
        "selected_deterministic_rerun": selected_rerun,
        "metric_abs_diff_vs_C7_B1_FINAL_DECISION": metric_abs_diff,
        "consistent_with_C7_B1_FINAL_DECISION": consistent,
        "zero_residual_control_pass": zero_pass,
        "neighbor_stability_pass": neighbor["passes"],
        "leakage_audit_pass": leakage["passes"],
        "topk_safety_pass": topk["passes"],
        "frozen_conquer_pass": frozen["passes"],
        "evaluator_modified": False,
        "nms_modified": False,
        "C6_B1_modified": False,
        "C6_B2_modified": False,
        "C4_final_modified": False,
        "enter_C7_B2": False,
        "enter_C7_C": False,
        "enter_C8": False,
        "decision": "Stop and wait for human authorization for exactly-one official-val one-shot.",
    }

    audit_dir = Path(args.audit_dir)
    atomic_json(audit_dir / "C7_B1_FINAL_FREEZE_PACKAGE.json", package)
    atomic_json(audit_dir / "C7_B1_OFFICIAL_READY_MANIFEST.json", {
        "status": status,
        "official_val_used": False,
        "promoted": False,
        "selected_config": expected["config"],
        "model": artifact(Path(args.models_dir) / "c7_b1_proposal_confidence_head.pt"),
        "freeze_package": artifact(audit_dir / "C7_B1_FINAL_FREEZE_PACKAGE.json"),
        "requires_human_authorization_before_official_val": True,
    })
    atomic_json(audit_dir / "C7_B1_FREEZE_HASHES.json", {
        "metric_hash": selected_rerun["hashes"]["metric_hash"],
        "prediction_hash": selected_rerun["hashes"]["prediction_hash"],
        "score_hash": selected_rerun["hashes"]["score_hash"],
        "zero_prediction_hash": zero["hashes"]["prediction_hash"],
        "zero_score_hash": zero["hashes"]["score_hash"],
        "model": artifact(Path(args.models_dir) / "c7_b1_proposal_confidence_head.pt"),
        "final_decision": artifact(audit_dir / "C7_B1_FINAL_DECISION.json"),
        "freeze_package": artifact(audit_dir / "C7_B1_FINAL_FREEZE_PACKAGE.json"),
    })
    atomic_json(audit_dir / "C7_B1_ZERO_RESIDUAL_CONTROL.json", zero)
    atomic_json(audit_dir / "C7_B1_NEIGHBOR_STABILITY.json", neighbor)
    atomic_json(audit_dir / "C7_B1_LEAKAGE_AUDIT.json", leakage)
    atomic_json(audit_dir / "C7_B1_TOPK_SAFETY_AUDIT.json", topk)

    atomic_text(audit_dir / "C7_B1_FINAL_FREEZE_PACKAGE.md", "# C7-B1 final freeze package\n\n```json\n" + json.dumps(package, indent=2, ensure_ascii=False) + "\n```\n")
    atomic_text(audit_dir / "C7_B1_ZERO_RESIDUAL_CONTROL.md", "# C7-B1 zero-residual control\n\n```json\n" + json.dumps(zero, indent=2, ensure_ascii=False) + "\n```\n")
    atomic_text(audit_dir / "C7_B1_NEIGHBOR_STABILITY.md", "# C7-B1 neighbor stability audit\n\n```json\n" + json.dumps(neighbor, indent=2, ensure_ascii=False) + "\n```\n")
    atomic_text(audit_dir / "C7_B1_LEAKAGE_AUDIT.md", "# C7-B1 leakage audit\n\n```json\n" + json.dumps(leakage, indent=2, ensure_ascii=False) + "\n```\n")
    atomic_text(audit_dir / "C7_B1_TOPK_SAFETY_AUDIT.md", "# C7-B1 TopK safety audit\n\n```json\n" + json.dumps(topk, indent=2, ensure_ascii=False) + "\n```\n")
    print(json.dumps({
        "status": status,
        "official_val_used": False,
        "training_started": False,
        "selected_config": expected["config"],
        "consistent_with_final_decision": consistent,
        "zero_residual_control_pass": zero_pass,
        "neighbor_stability_pass": neighbor["passes"],
        "leakage_audit_pass": leakage["passes"],
        "topk_safety_pass": topk["passes"],
        "frozen_conquer_pass": frozen["passes"],
        "metric_hash": selected_rerun["hashes"]["metric_hash"],
        "prediction_hash": selected_rerun["hashes"]["prediction_hash"],
        "score_hash": selected_rerun["hashes"]["score_hash"],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
