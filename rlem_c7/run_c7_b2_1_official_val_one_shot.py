#!/usr/bin/env python
"""Exactly-one official-val one-shot for frozen C7-B2.1 A4.

Fixed mechanism:
  C7-B1-frozen official top100 candidate pool is the only candidate source.
  C7-B2 video residual scores only rerank that fixed pool.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rlem.c4_lite_utils import flatten_eval_metrics  # noqa: E402
from rlem_c7.run_c7_b0_b1 import (  # noqa: E402
    EPS,
    atomic_json,
    atomic_npz,
    atomic_text,
    artifact,
    features_for_spans,
    labels_for_sequence,
    load_npz,
    make_fast_anchor_data,
    make_anchor_candidates_fast,
    metric_delta,
    nms_sequence,
    selected_metrics,
)
from rlem_c7.run_c7_b1_official_val_one_shot import (  # noqa: E402
    CachedProposalScorer,
    build_c7_official_predictions,
    prepare_official_side,
    score_official_c6c_features,
    build_official_c6c_features,
    pool_score_array,
)
from rlem_c7.run_c7_b2_train_calib import C7B2VideoResidualHead  # noqa: E402
from rlem_c7.run_c7_b2_1_safety_repair import group_score_map, normalize_cand  # noqa: E402
from rlem_c7.wrapper import ProposalConfidenceHead  # noqa: E402
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
    data = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def write_jsonl(path: str | Path, rows: List[Dict[str, Any]]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(p) + ".partial")
    with tmp.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n")
    os.replace(tmp, p)


def load_metrics(path: str | Path) -> Dict[str, float]:
    obj = json.loads(Path(path).read_text(encoding="utf-8"))
    if "C7_B1_official_metrics" in obj:
        flat = obj["C7_B1_official_metrics"]
    elif "VCMR" in obj:
        flat = obj["VCMR"]
    else:
        flat = flatten_eval_metrics(obj)
    return {k: float(flat[k]) for k in METRIC_KEYS}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--allow_official_val", action="store_true", required=True)
    p.add_argument("--ready_manifest", default="c7_audit/C7_B2_1_OFFICIAL_READY_MANIFEST.json")
    p.add_argument("--freeze_package", default="c7_audit/C7_B2_1_FINAL_FREEZE_PACKAGE.json")
    p.add_argument("--official_cache_npz", default="results/rlem_c6_b1_lite_r1_safe/official_val_c6_cache.npz")
    p.add_argument("--official_pool_npz", default="results/rlem_c6_b1_lite_r1_safe/official_val_candidate_pool_top_slots.npz")
    p.add_argument("--official_temporal_npz", default="results/rlem_c5_main_a3_video_slot/official_val_temporal_prior.npz")
    p.add_argument("--official_boundary_npz", default="results/rlem_c6_b1_lite_r1_safe/official_val_boundary_oracle_prior.npz")
    p.add_argument("--c6_b1_scores", default="results/rlem_c6_b1_lite_r1_safe/official_val_scores.npz")
    p.add_argument("--c6_b2_scores", default="results/rlem_c6_b2_official_val/official_val_scores.npz")
    p.add_argument("--c6_b1_metrics", default="results/rlem_c6_b1_lite_r1_safe/official_val_metrics.json")
    p.add_argument("--c6_b2_metrics", default="results/rlem_c6_b2_official_val/official_val_metrics.json")
    p.add_argument("--c7_b1_metrics", default="c7_audit/C7_B1_OFFICIAL_MANIFEST.json")
    p.add_argument("--c7_b1_submission", default="results/rlem_c7_official_val/official_val_submission.json")
    p.add_argument("--c6c_model", default="results/rlem_c6_c/arms/mavr_pr_mil/model_best.pt")
    p.add_argument("--c7_model", default="c7_models/c7_b1_proposal_confidence_head.pt")
    p.add_argument("--c7_calib_scores", default="results/rlem_c7/train_calib_proposal_scores.npz")
    p.add_argument("--b2_best_config", default="results/rlem_c6_b2/train_calib_scores/pairwise_main/best_config.json")
    p.add_argument("--c7_b2_model", default="results/rlem_c7_b2/c7_b2_mil_video_residual_head.pt")
    p.add_argument("--gt_jsonl", default="/home/a/yybwork/data/yyb/tvr_feature_release/data/tvr_val_release.jsonl")
    p.add_argument("--output_dir", default="results/rlem_c7_b2_1_official_val")
    p.add_argument("--audit_dir", default="c7_audit")
    p.add_argument("--device", default="cuda")
    p.add_argument("--score_batch_size", type=int, default=65536)
    p.add_argument("--feature_chunk_size", type=int, default=200000)
    p.add_argument("--effective_top_n", type=int, default=100)
    p.add_argument("--max_after_nms", type=int, default=100)
    p.add_argument("--nms_thd", type=float, default=0.7)
    return p.parse_args()


def output_paths(args: argparse.Namespace) -> Dict[str, Path]:
    out = Path(args.output_dir)
    audit = Path(args.audit_dir)
    return {
        "aux_dir": out / "aux",
        "c6c_features": out / "aux" / "official_val_c6c_video_features.npz",
        "c6c_scores": out / "aux" / "official_val_c6c_video_scores.npz",
        "c7_prop_scores": out / "aux" / "official_val_c7_b1_proposal_scores_for_video.npz",
        "video_dataset": out / "aux" / "official_val_video_dataset.npz",
        "video_scores": out / "aux" / "official_val_video_scores.npz",
        "anchor_states": out / "aux" / "official_val_c7_b1_anchor_states.json",
        "scores_npz": out / "official_val_scores.npz",
        "predictions_jsonl": out / "official_val_predictions.jsonl",
        "submission_json": out / "official_val_submission.json",
        "metrics_json": out / "official_val_metrics.json",
        "one_shot_md": audit / "C7_B2_1_OFFICIAL_VAL_ONE_SHOT.md",
        "one_shot_json": audit / "C7_B2_1_OFFICIAL_VAL_ONE_SHOT.json",
        "decision_md": audit / "C7_B2_1_OFFICIAL_DECISION.md",
        "decision_json": audit / "C7_B2_1_OFFICIAL_DECISION.json",
        "manifest_json": audit / "C7_B2_1_OFFICIAL_MANIFEST.json",
        "hashes_json": audit / "C7_B2_1_OFFICIAL_HASHES.json",
    }


def ensure_no_prior_outputs(paths: Dict[str, Path]) -> None:
    keys = [
        "scores_npz", "predictions_jsonl", "submission_json", "metrics_json",
        "one_shot_md", "one_shot_json", "decision_md", "decision_json",
        "manifest_json", "hashes_json",
    ]
    existing = [str(paths[k]) for k in keys if paths[k].exists()]
    if existing:
        raise FileExistsError("refusing second C7-B2.1 official-val run; outputs already exist: " + ", ".join(existing))


def validate_ready(args: argparse.Namespace) -> Dict[str, Any]:
    ready = json.loads(Path(args.ready_manifest).read_text(encoding="utf-8"))
    freeze = json.loads(Path(args.freeze_package).read_text(encoding="utf-8"))
    if ready.get("status") != "C7_B2_1_OFFICIAL_READY" or freeze.get("status") != "C7_B2_1_OFFICIAL_READY":
        raise ValueError("C7-B2.1 status_before_run is not OFFICIAL_READY")
    if ready.get("official_val_used") is not False or freeze.get("official_val_used") is not False:
        raise ValueError("C7-B2.1 freeze state already indicates official_val_used")
    if ready.get("selected_variant") != "A4_video_residual_only" or freeze.get("selected_variant") != "A4_video_residual_only":
        raise ValueError("selected variant mismatch")
    return {"ready": ready, "freeze": freeze}


def aggregate_by_group(values: np.ndarray, gids: np.ndarray, n: int, op: str, default: float = 0.0) -> np.ndarray:
    if op == "max":
        out = np.full(n, -1e9, dtype=np.float32)
        np.maximum.at(out, gids, values.astype(np.float32))
        out[out < -1e8] = default
        return out
    if op == "mean":
        sums = np.bincount(gids, weights=values.astype(np.float64), minlength=n).astype(np.float32)
        cnt = np.bincount(gids, minlength=n).astype(np.float32)
        return sums / np.maximum(cnt, 1.0)
    raise ValueError(op)


@torch.no_grad()
def score_c7_proposal_full(args: argparse.Namespace, x: np.ndarray) -> Dict[str, np.ndarray]:
    ckpt = torch.load(args.c7_model, map_location="cpu")
    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    model = ProposalConfidenceHead(
        int(ckpt["in_dim"]),
        hidden=int(ckpt["hidden"]),
        layers=int(ckpt["layers"]),
        dropout=float(ckpt["dropout"]),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    xn = (x.astype(np.float32) - ckpt["mean"]) / np.maximum(ckpt["std"], EPS)
    out = {k: np.empty(len(xn), dtype=np.float32) for k in ["proposal_confidence", "proposal_iou05", "proposal_iou07", "span_quality"]}
    for start in range(0, len(xn), args.score_batch_size):
        xb = torch.from_numpy(xn[start:start + args.score_batch_size]).to(device, non_blocking=True)
        pred = model(xb)
        n = len(xb)
        out["proposal_confidence"][start:start+n] = torch.sigmoid(pred["proposal_confidence"]).float().cpu().numpy()
        out["proposal_iou05"][start:start+n] = torch.sigmoid(pred["proposal_iou05_logit"]).float().cpu().numpy()
        out["proposal_iou07"][start:start+n] = torch.sigmoid(pred["proposal_iou07_logit"]).float().cpu().numpy()
        out["span_quality"][start:start+n] = pred["span_quality_logit"].float().cpu().numpy()
    return out


@torch.no_grad()
def ensure_official_c7_proposal_scores_for_video(args: argparse.Namespace, paths: Dict[str, Path], cache: Dict[str, np.ndarray], side: Dict[str, Any]) -> Dict[str, Any]:
    out = paths["c7_prop_scores"]
    if out.exists():
        return {"reused": True, "scores": artifact(out)}
    q_count = len(cache["desc_ids"])
    offsets = cache["desc_offsets"].astype(np.int64)
    s_c4 = cache["s_c4_final"].astype(np.float32)
    rows_all: List[np.ndarray] = []
    ranks_all: List[np.ndarray] = []
    for q in range(q_count):
        rows = np.arange(int(offsets[q]), int(offsets[q + 1]), dtype=np.int64)
        order = rows[np.argsort(-s_c4[rows], kind="stable")[:args.effective_top_n]]
        rows_all.append(order)
        ranks_all.append(np.arange(len(order), dtype=np.int16))
    rows = np.concatenate(rows_all).astype(np.int64)
    ranks = np.concatenate(ranks_all).astype(np.int16)
    gids = cache["row_group_id"][rows].astype(np.int64)
    starts = cache["start_idx"][rows].astype(np.int64)
    ends = cache["end_idx"][rows].astype(np.int64)
    base_scores = cache["s_c4_final"][rows].astype(np.float32)
    x = features_for_spans(cache, side, rows, gids, starts, ends, base_scores, ranks, chunk_size=args.feature_chunk_size)
    pred = score_c7_proposal_full(args, x)
    atomic_npz(
        out,
        proposal_confidence=pred["proposal_confidence"].astype(np.float32),
        proposal_iou05=pred["proposal_iou05"].astype(np.float32),
        proposal_iou07=pred["proposal_iou07"].astype(np.float32),
        span_quality=pred["span_quality"].astype(np.float32),
        query=cache["query_index"][rows].astype(np.int32),
        row=rows.astype(np.int64),
        group_id=gids.astype(np.int32),
        rank=ranks.astype(np.int16),
    )
    return {"reused": False, "scores": artifact(out), "examples": int(len(rows))}


def build_official_video_dataset(args: argparse.Namespace, paths: Dict[str, Path], cache: Dict[str, np.ndarray], c6c_scores: Dict[str, np.ndarray]) -> Dict[str, Any]:
    out = paths["video_dataset"]
    if out.exists():
        return {"reused": True, "dataset": artifact(out)}
    prop = load_npz(paths["c7_prop_scores"], allow_pickle=False)
    n = int(cache["video_group_keys"].shape[0])
    gids_prop = prop["group_id"].astype(np.int64)
    cnt = np.bincount(gids_prop, minlength=n).astype(np.float32)
    rank_inv = 1.0 / (prop["rank"].astype(np.float32) + 1.0)
    cols = [
        aggregate_by_group(prop["proposal_confidence"], gids_prop, n, "max"),
        aggregate_by_group(prop["proposal_confidence"], gids_prop, n, "mean"),
        aggregate_by_group(prop["span_quality"], gids_prop, n, "max"),
        aggregate_by_group(prop["span_quality"], gids_prop, n, "mean"),
        aggregate_by_group(prop["proposal_iou05"], gids_prop, n, "max"),
        aggregate_by_group(prop["proposal_iou07"], gids_prop, n, "max"),
        aggregate_by_group(np.clip(prop["span_quality"], 0.0, 1.0), gids_prop, n, "max"),
        aggregate_by_group(rank_inv, gids_prop, n, "max"),
        np.minimum(cnt / 100.0, 1.0).astype(np.float32),
    ]
    c6c_idx = np.full(n, -1, dtype=np.int64)
    c6c_idx[c6c_scores["group_id"].astype(np.int64)] = np.arange(len(c6c_scores["group_id"]), dtype=np.int64)
    valid = c6c_idx >= 0
    for key in ["relevance", "moment", "iou", "risk", "gate"]:
        arr = np.zeros(n, dtype=np.float32)
        arr[valid] = c6c_scores[key].astype(np.float32)[c6c_idx[valid]]
        cols.append(arr)
    base_rank = np.full(n, 100.0, dtype=np.float32)
    base_rank[valid] = c6c_scores["base_rank"].astype(np.float32)[c6c_idx[valid]]
    cols.extend([1.0 / (base_rank + 1.0), np.clip(base_rank / 100.0, 0.0, 1.0).astype(np.float32)])
    x = np.concatenate([np.stack(cols, axis=1), cache["video_features"].astype(np.float32)], axis=1)
    atomic_npz(
        out,
        x=x.astype(np.float32),
        group_id=np.arange(n, dtype=np.int32),
        query=cache["video_group_keys"][:, 0].astype(np.int32),
        video_idx=cache["video_group_keys"][:, 1].astype(np.int32),
        base_rank=base_rank.astype(np.float32),
    )
    return {"reused": False, "dataset": artifact(out), "groups": int(n), "feature_dim": int(x.shape[1])}


@torch.no_grad()
def score_official_video_dataset(args: argparse.Namespace, paths: Dict[str, Path]) -> Dict[str, Any]:
    out = paths["video_scores"]
    if out.exists():
        return {"reused": True, "scores": artifact(out)}
    z = load_npz(paths["video_dataset"], allow_pickle=False)
    ckpt = torch.load(args.c7_b2_model, map_location="cpu")
    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    model = C7B2VideoResidualHead(int(ckpt["in_dim"]), int(ckpt["hidden"]), int(ckpt["layers"]), float(ckpt["dropout"])).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    x = (z["x"].astype(np.float32) - ckpt["mean"]) / np.maximum(ckpt["std"], EPS)
    outs = {k: np.empty(len(x), dtype=np.float32) for k in ["relevance", "moment", "iou", "risk", "gate"]}
    for start in range(0, len(x), args.score_batch_size):
        xb = torch.from_numpy(x[start:start + args.score_batch_size]).to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
            pred = model(xb)
        n = len(xb)
        outs["relevance"][start:start+n] = pred["relevance"].float().cpu().numpy()
        outs["moment"][start:start+n] = pred["moment"].float().cpu().numpy()
        outs["iou"][start:start+n] = pred["iou"].float().cpu().numpy()
        outs["risk"][start:start+n] = pred["risk"].float().cpu().numpy()
        outs["gate"][start:start+n] = torch.sigmoid(pred["gate"]).float().cpu().numpy()
    atomic_npz(out, **outs, group_id=z["group_id"], query=z["query"], video_idx=z["video_idx"], base_rank=z["base_rank"])
    return {"reused": False, "scores": artifact(out), "gate_mean": float(outs["gate"].mean())}


def build_official_anchor_states(args: argparse.Namespace, paths: Dict[str, Path], cache: Dict[str, np.ndarray], pool: Dict[str, np.ndarray], b2_pool_score: np.ndarray, side: Dict[str, Any]) -> Dict[str, Any]:
    out = paths["anchor_states"]
    if out.exists():
        return {"reused": True, "anchor_states": artifact(out)}
    b2_cfg = json.loads(Path(args.b2_best_config).read_text(encoding="utf-8"))
    calib_scores = load_npz(args.c7_calib_scores, allow_pickle=False)
    prop_mean = float(calib_scores["proposal_confidence"].mean())
    prop_std = float(max(calib_scores["proposal_confidence"].std(), EPS))
    qual_mean = float(calib_scores["span_quality"].mean())
    qual_std = float(max(calib_scores["span_quality"].std(), EPS))
    data = make_fast_anchor_data(cache, pool, b2_pool_score)
    scorer = CachedProposalScorer(args)
    states: List[Dict[str, Any]] = []
    for q in range(len(cache["desc_ids"])):
        base = make_anchor_candidates_fast(q=q, data=data, policy_cfg=b2_cfg, effective_top_n=args.effective_top_n)
        gids = np.asarray([c[0] for c in base], dtype=np.int64)
        starts = np.asarray([c[2] for c in base], dtype=np.int64)
        ends = np.asarray([c[3] for c in base], dtype=np.int64)
        scores = np.asarray([c[4] for c in base], dtype=np.float32)
        rows = np.asarray([max(c[5], 0) for c in base], dtype=np.int64)
        ranks = np.arange(len(base), dtype=np.int16)
        x = features_for_spans(cache, side, rows, gids, starts, ends, scores, ranks, chunk_size=args.feature_chunk_size)
        pred = scorer.score(x)
        prop_z = (pred["proposal_confidence"] - prop_mean) / max(prop_std, EPS)
        qual_z = (pred["span_quality"] - qual_mean) / max(qual_std, EPS)
        new_scores = scores + 0.1 * prop_z.astype(np.float32) + 0.1 * qual_z.astype(np.float32)
        rescored = [(c[0], c[1], c[2], c[3], float(ns), c[5]) for c, ns in zip(base, new_scores)]
        rescored = sorted(rescored, key=lambda c: -c[4])
        base_group_order: List[int] = []
        seen = set()
        for c in rescored:
            if int(c[0]) not in seen:
                seen.add(int(c[0]))
                base_group_order.append(int(c[0]))
        states.append({
            "base_group_order": base_group_order,
            "flat_candidates": [[int(c[0]), int(c[1]), int(c[2]), int(c[3]), float(c[4]), int(c[5])] for c in rescored],
        })
    atomic_json(out, {"states": states})
    return {"reused": False, "anchor_states": artifact(out), "queries": int(len(states))}


def selected_sequences(args: argparse.Namespace, paths: Dict[str, Path], cache: Dict[str, np.ndarray], c4_submission: Dict[str, Any]) -> Dict[str, Any]:
    states = json.loads(paths["anchor_states"].read_text(encoding="utf-8"))["states"]
    score_arrays = load_npz(paths["video_scores"], allow_pickle=False)
    cfg = json.loads(Path("c7_audit/C7_B2_TRAIN_CALIB_EVAL.json").read_text(encoding="utf-8"))["best"]["config"]
    gscore = group_score_map(score_arrays, cfg)
    video_weight = float(cfg["video_weight"])
    gt_vid = cache["video_idx"][cache["desc_offsets"][:-1]].astype(np.int64)
    gt_s = cache["gt_start_idx"][cache["desc_offsets"][:-1]].astype(np.int64)
    gt_e = cache["gt_end_idx"][cache["desc_offsets"][:-1]].astype(np.int64)
    anchor_post: List[List[Candidate]] = []
    selected_post: List[List[Candidate]] = []
    anchor_pre_sets = []
    selected_pre_sets = []
    predictions_jsonl: List[Dict[str, Any]] = []
    policy_predictions = np.full((len(states), args.max_after_nms, 4), -1, dtype=np.float32)
    final_scores = np.full((len(states), args.effective_top_n), np.nan, dtype=np.float32)
    invalid = dup = exits = entries = 0
    labels05: List[List[bool]] = []
    labels07: List[List[bool]] = []
    anchor_labels05: List[List[bool]] = []
    anchor_labels07: List[List[bool]] = []
    for q, state in enumerate(states):
        anchor = [normalize_cand(x) for x in state["flat_candidates"]]
        selected = []
        for c in anchor:
            residual = gscore.get(int(c[0]), (0.0, 0.0, 0.0, 0.0))[2]
            selected.append((c[0], c[1], c[2], c[3], float(c[4]) + video_weight * residual, c[5]))
        selected = sorted(selected, key=lambda c: -c[4])
        apost = nms_sequence(anchor[:args.effective_top_n], args.nms_thd, args.max_after_nms)
        spost = nms_sequence(selected[:args.effective_top_n], args.nms_thd, args.max_after_nms)
        anchor_post.append(apost)
        selected_post.append(spost)
        anchor_pre_sets.append(sorted((int(c[1]), int(c[2]), int(c[3])) for c in anchor))
        selected_pre_sets.append(sorted((int(c[1]), int(c[2]), int(c[3])) for c in selected))
        l05, l07, pos, inv, du = labels_for_sequence(spost, int(gt_vid[q]), int(gt_s[q]), int(gt_e[q]))
        a05, a07, apos, _ai, _ad = labels_for_sequence(apost, int(gt_vid[q]), int(gt_s[q]), int(gt_e[q]))
        labels05.append(l05); labels07.append(l07); anchor_labels05.append(a05); anchor_labels07.append(a07)
        invalid += inv; dup += du
        exits += int(apos and not pos); entries += int((not apos) and pos)
        preds = []
        for r, c in enumerate(spost):
            preds.append([int(c[1]), float(c[2] * CLIP), float((c[3] + 1) * CLIP), float(c[4])])
            policy_predictions[q, r] = np.asarray([c[1], c[2], c[3], c[4]], dtype=np.float32)
        for r, c in enumerate(selected[:args.effective_top_n]):
            final_scores[q, r] = float(c[4])
        desc_id = cache["desc_ids"][q].item() if hasattr(cache["desc_ids"][q], "item") else cache["desc_ids"][q]
        predictions_jsonl.append({"desc_id": desc_id, "desc": str(cache["desc_text"][q]), "predictions": preds})
    submission = {"video2idx": c4_submission["video2idx"], "VCMR": predictions_jsonl}
    atomic_json(paths["submission_json"], submission)
    write_jsonl(paths["predictions_jsonl"], predictions_jsonl)
    atomic_npz(paths["scores_npz"], policy_predictions=policy_predictions, final_score=final_scores, video_weight=np.asarray([video_weight], dtype=np.float32))
    fixed_pool = all(a == b for a, b in zip(anchor_pre_sets, selected_pre_sets))
    return {
        "submission": submission,
        "metrics_internal": selected_metrics(labels05, labels07),
        "anchor_metrics_internal": selected_metrics(anchor_labels05, anchor_labels07),
        "fixed_pool_invariant_on_official": bool(fixed_pool),
        "hard_positive_top100_query_exits": int(exits),
        "hard_positive_top100_query_entries": int(entries),
        "hard_positive_exit_ratio": float(exits / max(sum(labels_for_sequence(a, int(gt_vid[q]), int(gt_s[q]), int(gt_e[q]))[2] for q, a in enumerate(anchor_post)), 1)),
        "invalid_span_count": int(invalid),
        "duplicate_span_count_after_nms": int(dup),
        "queries": int(len(states)),
        "video_weight": video_weight,
    }


def classify(delta_vs_b2: Dict[str, float], delta_vs_c7b1: Dict[str, float], structural: Dict[str, Any]) -> str:
    core_keys = ["0.5-r1", "0.5-r5", "0.5-r10", "0.7-r1", "0.7-r5", "0.7-r10"]
    core_nonneg = sum(int(delta_vs_b2[k] >= 0.0) for k in core_keys)
    core_strong = delta_vs_b2["0.7-r1"] > 0 and delta_vs_b2["0.7-r5"] >= 0 and delta_vs_b2["0.7-r10"] >= 0 and delta_vs_b2["0.5-r1"] > 0 and core_nonneg >= 5
    safety = delta_vs_c7b1["0.7-r100"] >= -0.10 and delta_vs_c7b1["0.5-r100"] >= -0.10 and structural["invalid_span_count"] == 0 and structural["duplicate_span_count_after_nms"] == 0 and structural["hard_positive_exit_ratio"] <= 0.001
    if core_strong and safety:
        return "C7_B2_1_OFFICIAL_PROMOTED"
    if core_nonneg >= 5 and structural["invalid_span_count"] == 0 and structural["duplicate_span_count_after_nms"] == 0:
        return "C7_B2_1_OFFICIAL_CORE_POSITIVE_REVIEW"
    if delta_vs_b2["0.7-r1"] > 0 and (delta_vs_b2["0.7-r5"] < -0.10 or delta_vs_b2["0.7-r10"] < -0.10):
        return "C7_B2_1_R1_TRADEOFF_NO_PROMOTION"
    return "C7_B2_1_OFFICIAL_NEGATIVE"


def metric_table(c6b1: Dict[str, float], c6b2: Dict[str, float], c7b1: Dict[str, float], c7b21: Dict[str, float], d1: Dict[str, float], d2: Dict[str, float], d7: Dict[str, float]) -> str:
    rows = [
        "| system | 0.5-r1 | 0.5-r5 | 0.5-r10 | 0.5-r100 | 0.7-r1 | 0.7-r5 | 0.7-r10 | 0.7-r100 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        "| C6-B1 official | " + " | ".join(f"{c6b1[k]:.2f}" for k in METRIC_KEYS) + " |",
        "| C6-B2 official | " + " | ".join(f"{c6b2[k]:.2f}" for k in METRIC_KEYS) + " |",
        "| C7-B1 official archived | " + " | ".join(f"{c7b1[k]:.2f}" for k in METRIC_KEYS) + " |",
        "| C7-B2.1 A4 official | " + " | ".join(f"{c7b21[k]:.2f}" for k in METRIC_KEYS) + " |",
        "| Delta vs C6-B1 | " + " | ".join(f"{d1[k]:+.2f}" for k in METRIC_KEYS) + " |",
        "| Delta vs C6-B2 | " + " | ".join(f"{d2[k]:+.2f}" for k in METRIC_KEYS) + " |",
        "| Delta vs C7-B1 | " + " | ".join(f"{d7[k]:+.2f}" for k in METRIC_KEYS) + " |",
    ]
    return "\n".join(rows)


def main() -> None:
    args = parse_args()
    if not args.allow_official_val:
        raise ValueError("--allow_official_val is required")
    paths = output_paths(args)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.audit_dir).mkdir(parents=True, exist_ok=True)
    paths["aux_dir"].mkdir(parents=True, exist_ok=True)
    ensure_no_prior_outputs(paths)
    ready_state = validate_ready(args)
    cache = load_npz(args.official_cache_npz, allow_pickle=True)
    pool = load_npz(args.official_pool_npz, allow_pickle=False)
    temporal = load_npz(args.official_temporal_npz, allow_pickle=False)
    boundary = load_npz(args.official_boundary_npz, allow_pickle=False)
    b1_pool_score = pool_score_array(args.c6_b1_scores)
    b2_pool_score = pool_score_array(args.c6_b2_scores)
    c6c_feature_info = build_official_c6c_features(cache, pool, b1_pool_score, b2_pool_score, temporal, paths["c6c_features"])
    c6c_score_info = score_official_c6c_features(args, paths["c6c_features"], paths["c6c_scores"])
    c6c_scores = load_npz(paths["c6c_scores"], allow_pickle=False)
    side = prepare_official_side(cache, pool, b1_pool_score, b2_pool_score, temporal, boundary, c6c_scores)
    proposal_info = ensure_official_c7_proposal_scores_for_video(args, paths, cache, side)
    video_ds_info = build_official_video_dataset(args, paths, cache, c6c_scores)
    video_score_info = score_official_video_dataset(args, paths)
    anchor_info = build_official_anchor_states(args, paths, cache, pool, b2_pool_score, side)
    c7b1_submission = json.loads(Path(args.c7_b1_submission).read_text(encoding="utf-8"))
    structural = selected_sequences(args, paths, cache, c7b1_submission)

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
    c7b21_metrics = {k: float(flatten_eval_metrics(metrics_raw)[k]) for k in METRIC_KEYS}
    c6b1_metrics = load_metrics(args.c6_b1_metrics)
    c6b2_metrics = load_metrics(args.c6_b2_metrics)
    c7b1_metrics = load_metrics(args.c7_b1_metrics)
    delta_vs_c6b1 = metric_delta(c7b21_metrics, c6b1_metrics)
    delta_vs_c6b2 = metric_delta(c7b21_metrics, c6b2_metrics)
    delta_vs_c7b1 = metric_delta(c7b21_metrics, c7b1_metrics)
    status = classify(delta_vs_c6b2, delta_vs_c7b1, structural)
    common = {
        "stage": "C7-B2.1",
        "status": status,
        "status_before_run": "C7_B2_1_OFFICIAL_READY",
        "selected_variant": "A4_video_residual_only",
        "mechanism": "fixed-pool video residual rerank",
        "anchor": "C7-B1-frozen top100 candidate pool",
        "date": datetime.now().strftime("%Y-%m-%d"),
        "official_val_used": True,
        "official_val_run": True,
        "post_val_adjustment": False,
        "second_official_val": False,
        "training_started": False,
        "config_selection_changed": False,
        "neighbor_search": False,
        "phase_E_entered": False,
        "enter_C7_C": False,
        "enter_C8": False,
        "evaluator_modified": False,
        "nms_modified": False,
        "C4_C6_artifacts_modified": False,
        "C7_B1_official_archived_decision_modified": False,
        "C6_B1_official_metrics": c6b1_metrics,
        "C6_B2_official_metrics": c6b2_metrics,
        "C7_B1_official_archived_metrics": c7b1_metrics,
        "C7_B2_1_A4_official_metrics": c7b21_metrics,
        "delta_vs_C6_B1": delta_vs_c6b1,
        "delta_vs_C6_B2": delta_vs_c6b2,
        "delta_vs_C7_B1_official": delta_vs_c7b1,
        "fixed_pool_invariant_on_official": structural["fixed_pool_invariant_on_official"],
        "hard_positive_top100_query_exits": structural["hard_positive_top100_query_exits"],
        "hard_positive_top100_query_entries": structural["hard_positive_top100_query_entries"],
        "hard_positive_exit_ratio": structural["hard_positive_exit_ratio"],
        "invalid_span_count": structural["invalid_span_count"],
        "duplicate_span_count_after_nms": structural["duplicate_span_count_after_nms"],
        "auxiliary_fixed_inference": {
            "c6c_feature_info": c6c_feature_info,
            "c6c_score_info": c6c_score_info,
            "proposal_info": proposal_info,
            "video_dataset_info": video_ds_info,
            "video_score_info": video_score_info,
            "anchor_info": anchor_info,
        },
        "ready_state": {
            "ready_manifest": artifact(args.ready_manifest),
            "freeze_package": artifact(args.freeze_package),
        },
    }
    atomic_json(paths["one_shot_json"], common)
    atomic_json(paths["manifest_json"], {**common, "promoted": status == "C7_B2_1_OFFICIAL_PROMOTED", "official_evaluator_call_count": 1})
    decision = {
        "status": status,
        "promoted": status == "C7_B2_1_OFFICIAL_PROMOTED",
        "official_val_used": True,
        "post_val_adjustment": False,
        "second_official_val": False,
        "C7_B2_1_A4_official_metrics": c7b21_metrics,
        "delta_vs_C6_B1": delta_vs_c6b1,
        "delta_vs_C6_B2": delta_vs_c6b2,
        "delta_vs_C7_B1_official": delta_vs_c7b1,
        "fixed_pool_invariant_on_official": structural["fixed_pool_invariant_on_official"],
        "final_instruction": "Stop after exactly-one official-val one-shot and wait for human review.",
    }
    atomic_json(paths["decision_json"], decision)
    table = metric_table(c6b1_metrics, c6b2_metrics, c7b1_metrics, c7b21_metrics, delta_vs_c6b1, delta_vs_c6b2, delta_vs_c7b1)
    md = f"""# C7-B2.1 official-val one-shot

- Status: `{status}`
- selected_variant: `A4_video_residual_only`
- official_val_used: `true`
- official evaluator calls: `1`
- post_val_adjustment / second_official_val: `false / false`
- evaluator_modified / nms_modified: `false / false`
- fixed_pool_invariant_on_official: `{structural['fixed_pool_invariant_on_official']}`
- hard_positive exits/entries/ratio: `{structural['hard_positive_top100_query_exits']} / {structural['hard_positive_top100_query_entries']} / {structural['hard_positive_exit_ratio']}`
- invalid_span_count / duplicate_span_count_after_nms: `{structural['invalid_span_count']} / {structural['duplicate_span_count_after_nms']}`

## Metrics

{table}

Stop here and wait for human review.
"""
    atomic_text(paths["one_shot_md"], md)
    atomic_text(paths["decision_md"], "# C7-B2.1 official decision\n\n" + md)
    hashes = {
        "status": "C7_B2_1_OFFICIAL_HASHES",
        "official_val_used": True,
        "post_val_adjustment": False,
        "second_official_val": False,
        "metric_hash": sha256_obj(c7b21_metrics),
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
            "model": artifact(args.c7_b2_model),
            "ready_manifest": artifact(args.ready_manifest),
            "freeze_package": artifact(args.freeze_package),
        },
    }
    atomic_json(paths["hashes_json"], hashes)
    print(json.dumps({
        "status": status,
        "C6_B1_official_metrics": c6b1_metrics,
        "C6_B2_official_metrics": c6b2_metrics,
        "C7_B1_official_archived_metrics": c7b1_metrics,
        "C7_B2_1_A4_official_metrics": c7b21_metrics,
        "delta_vs_C6_B1": delta_vs_c6b1,
        "delta_vs_C6_B2": delta_vs_c6b2,
        "delta_vs_C7_B1_official": delta_vs_c7b1,
        "fixed_pool_invariant_on_official": structural["fixed_pool_invariant_on_official"],
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
