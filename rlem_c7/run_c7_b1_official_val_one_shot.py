#!/usr/bin/env python
"""Exactly-one official-val one-shot for frozen C7-B1.

This runner is intentionally narrow:
  * fixed C7-B1 S1_span_level_bounded_residual config, mu=0.1, eta=0.1;
  * no training, no grid/search, no post-val adjustment;
  * C6-B2-compatible tuple path as anchor;
  * C6-C video evidence is generated only as frozen inference input features;
  * official evaluator is called exactly once.
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
from types import SimpleNamespace
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rlem.c4_lite_utils import flatten_eval_metrics  # noqa: E402
from rlem_c6_b0.run_c6_b0_candidate_diagnostic import span_iou_idx  # noqa: E402
from rlem_c6_c.run_c6_c_goal import C6CVideoReranker  # noqa: E402
from rlem_c7.run_c7_b0_b1 import (  # noqa: E402
    EPS,
    FastAnchorData,
    FEATURE_NAMES,
    artifact,
    atomic_json,
    atomic_npz,
    atomic_text,
    features_for_spans,
    make_fast_anchor_data,
    make_anchor_candidates_fast,
    nms_sequence,
)
from rlem_c7.wrapper import ProposalConfidenceHead  # noqa: E402
from standalone_eval.eval import eval_retrieval, load_jsonl  # noqa: E402


METRIC_KEYS = [
    "0.5-r1", "0.5-r5", "0.5-r10", "0.5-r100",
    "0.7-r1", "0.7-r5", "0.7-r10", "0.7-r100",
]
CLIP = 1.5
FIXED_CONFIG = {
    "stage": "C7-B1",
    "mode": "S1_span_level_bounded_residual",
    "mu": 0.1,
    "eta": 0.1,
    "model": "c7_models/c7_b1_proposal_confidence_head.pt",
}


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_obj(obj: Any) -> str:
    data = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def load_npz(path: str | Path, allow_pickle: bool = False) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=allow_pickle) as z:
        return {k: z[k] for k in z.files}


def metric_delta(new: Dict[str, float], base: Dict[str, float]) -> Dict[str, float]:
    return {k: float(new[k] - base[k]) for k in METRIC_KEYS}


def load_metrics(path: str | Path) -> Dict[str, float]:
    obj = json.loads(Path(path).read_text(encoding="utf-8"))
    flat = obj["VCMR"] if isinstance(obj, dict) and "VCMR" in obj else flatten_eval_metrics(obj)
    return {k: float(flat[k]) for k in METRIC_KEYS}


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(path) + ".partial")
    with tmp.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    os.replace(tmp, path)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--allow_official_val", action="store_true", required=True)
    p.add_argument("--official_cache_npz", default="results/rlem_c6_b1_lite_r1_safe/official_val_c6_cache.npz")
    p.add_argument("--official_pool_npz", default="results/rlem_c6_b1_lite_r1_safe/official_val_candidate_pool_top_slots.npz")
    p.add_argument("--official_temporal_npz", default="results/rlem_c5_main_a3_video_slot/official_val_temporal_prior.npz")
    p.add_argument("--official_boundary_npz", default="results/rlem_c6_b1_lite_r1_safe/official_val_boundary_oracle_prior.npz")
    p.add_argument("--c6_b1_official_scores", default="results/rlem_c6_b1_lite_r1_safe/official_val_scores.npz")
    p.add_argument("--c6_b2_official_scores", default="results/rlem_c6_b2_official_val/official_val_scores.npz")
    p.add_argument("--c6_b1_submission_json", default="results/rlem_c6_b1_lite_r1_safe/official_val_submission.json")
    p.add_argument("--c6_b1_metrics_json", default="results/rlem_c6_b1_lite_r1_safe/official_val_metrics.json")
    p.add_argument("--c6_b2_submission_json", default="results/rlem_c6_b2_official_val/official_val_submission.json")
    p.add_argument("--c6_b2_metrics_json", default="results/rlem_c6_b2_official_val/official_val_metrics.json")
    p.add_argument("--c6c_model", default="results/rlem_c6_c/arms/mavr_pr_mil/model_best.pt")
    p.add_argument("--c7_model", default=FIXED_CONFIG["model"])
    p.add_argument("--c7_calib_scores", default="results/rlem_c7/train_calib_proposal_scores.npz")
    p.add_argument("--b2_best_config", default="results/rlem_c6_b2/train_calib_scores/pairwise_main/best_config.json")
    p.add_argument("--ready_manifest", default="c7_audit/C7_B1_OFFICIAL_READY_MANIFEST.json")
    p.add_argument("--freeze_package", default="c7_audit/C7_B1_FINAL_FREEZE_PACKAGE.json")
    p.add_argument("--gt_jsonl", default="/home/a/yybwork/data/yyb/tvr_feature_release/data/tvr_val_release.jsonl")
    p.add_argument("--output_dir", default="results/rlem_c7_official_val")
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
        "scores_npz": out / "official_val_scores.npz",
        "predictions_jsonl": out / "official_val_predictions.jsonl",
        "submission_json": out / "official_val_submission.json",
        "metrics_json": out / "official_val_metrics.json",
        "one_shot_md": audit / "C7_B1_OFFICIAL_VAL_ONE_SHOT.md",
        "one_shot_json": audit / "C7_B1_OFFICIAL_VAL_ONE_SHOT.json",
        "decision_md": audit / "C7_B1_OFFICIAL_DECISION.md",
        "decision_json": audit / "C7_B1_OFFICIAL_DECISION.json",
        "manifest_json": audit / "C7_B1_OFFICIAL_MANIFEST.json",
        "hashes_json": audit / "C7_B1_OFFICIAL_HASHES.json",
    }


def ensure_no_prior_outputs(paths: Dict[str, Path]) -> None:
    one_shot = [
        "scores_npz", "predictions_jsonl", "submission_json", "metrics_json",
        "one_shot_md", "one_shot_json", "decision_md", "decision_json",
        "manifest_json", "hashes_json",
    ]
    existing = [str(paths[k]) for k in one_shot if paths[k].exists()]
    if existing:
        raise FileExistsError("refusing second C7-B1 official-val run; outputs already exist: " + ", ".join(existing))


def validate_ready_state(args: argparse.Namespace) -> Dict[str, Any]:
    ready = json.loads(Path(args.ready_manifest).read_text(encoding="utf-8"))
    freeze = json.loads(Path(args.freeze_package).read_text(encoding="utf-8"))
    if ready.get("status") != "C7_B1_OFFICIAL_READY" or freeze.get("status") != "C7_B1_OFFICIAL_READY":
        raise ValueError("C7-B1 is not in OFFICIAL_READY state")
    if ready.get("official_val_used") is not False or freeze.get("official_val_used") is not False:
        raise ValueError("ready/freeze manifest already indicates official_val_used")
    for obj, name in [(ready, "ready"), (freeze, "freeze")]:
        cfg = obj.get("selected_config", {})
        if cfg.get("mode") != FIXED_CONFIG["mode"] or float(cfg.get("mu")) != 0.1 or float(cfg.get("eta")) != 0.1:
            raise ValueError(f"{name} selected config mismatch: {cfg}")
    return {"ready": ready, "freeze": freeze}


def topk_mean(x: np.ndarray, k: int) -> float:
    if len(x) == 0:
        return 0.0
    y = np.sort(x)[-min(k, len(x)):]
    return float(y.mean())


def logsumexp(x: np.ndarray) -> float:
    if len(x) == 0:
        return 0.0
    m = float(np.max(x))
    return float(m + np.log(np.exp(x - m).sum()))


def entropy_from_probs(p: np.ndarray) -> float:
    p = np.asarray(p, dtype=np.float64)
    p = np.maximum(p, EPS)
    p = p / max(float(p.sum()), EPS)
    return float(-(p * np.log(p)).sum())


def pool_score_array(path: str | Path) -> np.ndarray:
    z = load_npz(path, allow_pickle=False)
    if "pool_score" in z:
        return z["pool_score"].astype(np.float32)
    if "score" in z:
        return z["score"].astype(np.float32)
    raise KeyError(f"no pool_score/score in {path}")


def aggregate_pool_evidence(pool: Dict[str, np.ndarray], scores: np.ndarray, q_count: int) -> Dict[Tuple[int, int], Dict[str, float]]:
    q, slots, alts = [int(x) for x in pool["shape"]]
    s3 = scores.reshape(q, slots, alts)
    out: Dict[Tuple[int, int], Dict[str, float]] = {}
    for qi in range(min(q, q_count)):
        for slot in range(slots):
            base = (qi * slots + slot) * alts
            orig = float(s3[qi, slot, 0])
            margin = float(np.max(s3[qi, slot]) - orig)
            for alt in range(alts):
                idx = base + alt
                gid = int(pool["group_id"][idx])
                rec = out.setdefault((qi, gid), {"max_score": -1e9, "max_margin": -1e9, "count": 0.0})
                rec["max_score"] = max(rec["max_score"], float(scores[idx]))
                rec["max_margin"] = max(rec["max_margin"], margin)
                rec["count"] += 1.0
    return out


def build_official_c6c_features(
    cache: Dict[str, np.ndarray],
    pool: Dict[str, np.ndarray],
    b1_pool_score: np.ndarray,
    b2_pool_score: np.ndarray,
    temporal: Dict[str, np.ndarray],
    out_path: Path,
) -> Dict[str, Any]:
    if out_path.exists():
        return {"reused": True, "features": artifact(out_path)}
    q_count = len(cache["desc_ids"])
    b1_ev = aggregate_pool_evidence(pool, b1_pool_score, q_count)
    b2_ev = aggregate_pool_evidence(pool, b2_pool_score, q_count)
    offsets = cache["desc_offsets"].astype(np.int64)
    row_gid = cache["row_group_id"].astype(np.int64)
    s_c4 = cache["s_c4_final"].astype(np.float32)
    start_idx = cache["start_idx"].astype(np.int64)
    end_idx = cache["end_idx"].astype(np.int64)
    video_features = cache["video_features"].astype(np.float32)
    keys = cache["video_group_keys"].astype(np.int64)
    p_ctx = temporal["p_ctx"].astype(np.float32)
    p_b = temporal["p_b"].astype(np.float32)
    p_e = temporal["p_e"].astype(np.float32)
    tlen = temporal["temporal_length"].astype(np.int64)

    feats: List[np.ndarray] = []
    q_arr: List[int] = []
    gid_arr: List[int] = []
    vid_arr: List[int] = []
    rank_arr: List[int] = []
    best_row_arr: List[int] = []
    for q in range(q_count):
        rows = np.arange(int(offsets[q]), int(offsets[q + 1]), dtype=np.int64)
        order = rows[np.argsort(-s_c4[rows], kind="stable")]
        group_rank: Dict[int, int] = {}
        group_rows: Dict[int, List[int]] = {}
        for rank, row in enumerate(order):
            gid = int(row_gid[row])
            group_rank.setdefault(gid, int(rank))
            group_rows.setdefault(gid, []).append(int(row))
        for gid, rs_list in group_rows.items():
            rs = np.asarray(rs_list, dtype=np.int64)
            scores = s_c4[rs]
            best_local = int(rs[int(np.argmax(scores))])
            score_sorted = np.sort(scores)[::-1]
            t = max(int(tlen[gid]), 1)
            ctx = p_ctx[gid, :t]
            pb = p_b[gid, :t]
            pe = p_e[gid, :t]
            b1 = b1_ev.get((q, gid), {"max_score": 0.0, "max_margin": 0.0, "count": 0.0})
            b2 = b2_ev.get((q, gid), {"max_score": 0.0, "max_margin": 0.0, "count": 0.0})
            lengths = (end_idx[rs] - start_idx[rs] + 1).astype(np.float32)
            centers = 0.5 * (start_idx[rs] + end_idx[rs]).astype(np.float32)
            base_rank = int(group_rank[gid])
            engineered = np.asarray([
                float(np.max(scores)),
                float(np.mean(scores)),
                float(np.std(scores)),
                topk_mean(scores, 3),
                topk_mean(scores, 5),
                logsumexp(scores),
                float(score_sorted[0] - score_sorted[1]) if len(score_sorted) > 1 else 0.0,
                1.0 / float(base_rank + 1),
                float(base_rank) / 100.0,
                float(len(rs)) / 100.0,
                float(np.mean(lengths)) / 100.0,
                float(np.max(lengths)) / 100.0,
                float(np.mean(centers)) / 100.0,
                float(np.max(ctx)),
                float(np.mean(ctx)),
                entropy_from_probs(ctx) / math.log(max(t, 2)),
                float(np.max(pb)),
                float(np.max(pe)),
                entropy_from_probs(pb) / math.log(max(t, 2)),
                entropy_from_probs(pe) / math.log(max(t, 2)),
                float(b1["max_score"]),
                float(b1["max_margin"]),
                float(b1["count"]) / 40.0,
                float(b2["max_score"]),
                float(b2["max_margin"]),
                float(b2["count"]) / 40.0,
                float(b2["max_score"] - b1["max_score"]),
                float(b2["max_margin"] - b1["max_margin"]),
                float(np.max(scores) - np.mean(s_c4[rows])),
                float(np.max(scores) - np.max(s_c4[rows])),
            ], dtype=np.float32)
            feats.append(np.concatenate([engineered, video_features[gid].astype(np.float32)], axis=0))
            q_arr.append(q)
            gid_arr.append(gid)
            vid_arr.append(int(keys[gid, 1]))
            rank_arr.append(base_rank)
            best_row_arr.append(best_local)
    x = np.stack(feats).astype(np.float32)
    atomic_npz(
        out_path,
        x=x,
        query=np.asarray(q_arr, dtype=np.int32),
        group_id=np.asarray(gid_arr, dtype=np.int32),
        video_idx=np.asarray(vid_arr, dtype=np.int32),
        base_rank=np.asarray(rank_arr, dtype=np.int16),
        best_row=np.asarray(best_row_arr, dtype=np.int32),
    )
    return {"reused": False, "features": artifact(out_path), "examples": int(len(x)), "feature_dim": int(x.shape[1])}


@torch.no_grad()
def score_official_c6c_features(args: argparse.Namespace, feat_path: Path, out_path: Path) -> Dict[str, Any]:
    if out_path.exists():
        return {"reused": True, "scores": artifact(out_path)}
    z = load_npz(feat_path, allow_pickle=False)
    ckpt = torch.load(args.c6c_model, map_location="cpu")
    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    model = C6CVideoReranker(
        int(ckpt["in_dim"]),
        hidden=int(ckpt["hidden"]),
        layers=int(ckpt["layers"]),
        dropout=float(ckpt["dropout"]),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    x = (z["x"].astype(np.float32) - ckpt["mean"]) / np.maximum(ckpt["std"], EPS)
    rel = np.empty(len(x), dtype=np.float32)
    mom = np.empty(len(x), dtype=np.float32)
    iou = np.empty(len(x), dtype=np.float32)
    risk = np.empty(len(x), dtype=np.float32)
    gate = np.empty(len(x), dtype=np.float32)
    for start in range(0, len(x), args.score_batch_size):
        xb = torch.from_numpy(x[start:start + args.score_batch_size]).to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
            out = model(xb)
        n = len(xb)
        rel[start:start+n] = out["relevance"].float().cpu().numpy()
        mom[start:start+n] = out["moment"].float().cpu().numpy()
        iou[start:start+n] = out["iou"].float().cpu().numpy()
        risk[start:start+n] = out["risk"].float().cpu().numpy()
        gate[start:start+n] = torch.sigmoid(out["gate"]).float().cpu().numpy()
    atomic_npz(
        out_path,
        relevance=rel,
        moment=mom,
        iou=iou,
        risk=risk,
        gate=gate,
        query=z["query"],
        group_id=z["group_id"],
        video_idx=z["video_idx"],
        base_rank=z["base_rank"],
    )
    return {"reused": False, "scores": artifact(out_path), "gate_mean": float(gate.mean())}


def make_gid_index(group_id: np.ndarray, max_gid: int) -> np.ndarray:
    out = np.full(max_gid + 1, -1, dtype=np.int64)
    out[group_id.astype(np.int64)] = np.arange(len(group_id), dtype=np.int64)
    return out


def aggregate_pool_group_score(pool: Dict[str, np.ndarray], score: np.ndarray, max_gid: int) -> np.ndarray:
    out = np.full(max_gid + 1, -10.0, dtype=np.float32)
    np.maximum.at(out, pool["group_id"].astype(np.int64), score.astype(np.float32))
    out[~np.isfinite(out)] = -10.0
    return out


def prepare_official_side(
    cache: Dict[str, np.ndarray],
    pool: Dict[str, np.ndarray],
    b1_pool_score: np.ndarray,
    b2_pool_score: np.ndarray,
    temporal: Dict[str, np.ndarray],
    boundary: Dict[str, np.ndarray],
    c6c_scores: Dict[str, np.ndarray],
) -> Dict[str, Any]:
    max_gid = int(cache["video_group_keys"].shape[0] - 1)
    return {
        "temporal": temporal,
        "temporal_idx": make_gid_index(temporal["group_id"], max_gid),
        "boundary": boundary,
        "boundary_idx": make_gid_index(boundary["group_id"], max_gid),
        "b1_group_score": aggregate_pool_group_score(pool, b1_pool_score, max_gid),
        "b2_group_score": aggregate_pool_group_score(pool, b2_pool_score, max_gid),
        "c6c": c6c_scores,
        "c6c_idx": make_gid_index(c6c_scores["group_id"], max_gid),
    }


class CachedProposalScorer:
    def __init__(self, args: argparse.Namespace):
        ckpt = torch.load(args.c7_model, map_location="cpu")
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
        self.batch_size = int(args.score_batch_size)

    @torch.no_grad()
    def score(self, x: np.ndarray) -> Dict[str, np.ndarray]:
        xn = (x.astype(np.float32) - self.mean) / self.std
        prop = np.empty(len(xn), dtype=np.float32)
        qual = np.empty(len(xn), dtype=np.float32)
        for start in range(0, len(xn), self.batch_size):
            xb = torch.from_numpy(xn[start:start + self.batch_size]).to(self.device, non_blocking=True)
            pred = self.model(xb)
            n = len(xb)
            prop[start:start+n] = torch.sigmoid(pred["proposal_confidence"]).float().cpu().numpy()
            qual[start:start+n] = pred["span_quality_logit"].float().cpu().numpy()
        return {"proposal_confidence": prop, "span_quality": qual}


def build_c7_official_predictions(
    args: argparse.Namespace,
    paths: Dict[str, Path],
    cache: Dict[str, np.ndarray],
    pool: Dict[str, np.ndarray],
    b2_pool_score: np.ndarray,
    side: Dict[str, Any],
) -> Dict[str, Any]:
    b2_cfg = json.loads(Path(args.b2_best_config).read_text(encoding="utf-8"))
    data: FastAnchorData = make_fast_anchor_data(cache, pool, b2_pool_score)
    c6_b2_submission = json.loads(Path(args.c6_b2_submission_json).read_text(encoding="utf-8"))
    calib_scores = load_npz(args.c7_calib_scores, allow_pickle=False)
    prop_mean = float(calib_scores["proposal_confidence"].mean())
    prop_std = float(max(calib_scores["proposal_confidence"].std(), EPS))
    qual_mean = float(calib_scores["span_quality"].mean())
    qual_std = float(max(calib_scores["span_quality"].std(), EPS))
    scorer = CachedProposalScorer(args)

    q_count = len(cache["desc_ids"])
    policy_predictions = np.full((q_count, args.max_after_nms, 4), -1, dtype=np.float32)
    final_base_scores = np.full((q_count, args.effective_top_n), np.nan, dtype=np.float32)
    final_scores = np.full((q_count, args.effective_top_n), np.nan, dtype=np.float32)
    proposal_conf = np.full((q_count, args.effective_top_n), np.nan, dtype=np.float32)
    span_quality = np.full((q_count, args.effective_top_n), np.nan, dtype=np.float32)
    invalid_span_count = 0
    duplicate_span_count_after_nms = 0
    top1_changed_vs_b2 = 0
    predictions_jsonl: List[Dict[str, Any]] = []

    for q in range(q_count):
        base_cands = make_anchor_candidates_fast(q=q, data=data, policy_cfg=b2_cfg, effective_top_n=args.effective_top_n)
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
        new_scores = scores + FIXED_CONFIG["mu"] * prop_z.astype(np.float32) + FIXED_CONFIG["eta"] * qual_z.astype(np.float32)
        rescored = [(c[0], c[1], c[2], c[3], float(ns), c[5]) for c, ns in zip(base_cands, new_scores)]
        rescored = sorted(rescored, key=lambda c: -c[4])
        seq = nms_sequence(rescored, args.nms_thd, args.max_after_nms)
        duplicate_span_count_after_nms += len(seq) - len({(int(c[1]), int(c[2]), int(c[3])) for c in seq})

        preds: List[List[float]] = []
        for rank, cand in enumerate(seq):
            _gid, vid, si, ei, score, _row = cand
            invalid_span_count += int(si < 0 or ei < si)
            preds.append([int(vid), float(si * CLIP), float((ei + 1) * CLIP), float(score)])
            policy_predictions[q, rank] = np.asarray([vid, si, ei, score], dtype=np.float32)
        if preds and c6_b2_submission["VCMR"][q]["predictions"]:
            b2_top = c6_b2_submission["VCMR"][q]["predictions"][0]
            top1_changed_vs_b2 += int((int(preds[0][0]), float(preds[0][1]), float(preds[0][2])) != (int(b2_top[0]), float(b2_top[1]), float(b2_top[2])))
        n = min(args.effective_top_n, len(new_scores))
        final_base_scores[q, :n] = scores[:n]
        proposal_conf[q, :n] = pred["proposal_confidence"][:n]
        span_quality[q, :n] = pred["span_quality"][:n]
        final_scores[q, :n] = new_scores[:n]
        desc_id = cache["desc_ids"][q].item() if hasattr(cache["desc_ids"][q], "item") else cache["desc_ids"][q]
        predictions_jsonl.append({"desc_id": desc_id, "desc": str(cache["desc_text"][q]), "predictions": preds})

    submission = {"video2idx": c6_b2_submission["video2idx"], "VCMR": predictions_jsonl}
    atomic_json(paths["submission_json"], submission)
    write_jsonl(paths["predictions_jsonl"], predictions_jsonl)
    atomic_npz(
        paths["scores_npz"],
        policy_predictions=policy_predictions,
        base_score=final_base_scores,
        final_score=final_scores,
        proposal_confidence=proposal_conf,
        span_quality=span_quality,
        mu=np.asarray([FIXED_CONFIG["mu"]], dtype=np.float32),
        eta=np.asarray([FIXED_CONFIG["eta"]], dtype=np.float32),
    )
    return {
        "invalid_span_count": int(invalid_span_count),
        "duplicate_span_count_after_nms": int(duplicate_span_count_after_nms),
        "top1_changed_ratio_vs_C6_B2": float(top1_changed_vs_b2 / max(q_count, 1)),
        "queries": int(q_count),
        "normalization_source": "results/rlem_c7/train_calib_proposal_scores.npz",
        "proposal_confidence_mean_train_calib": prop_mean,
        "proposal_confidence_std_train_calib": prop_std,
        "span_quality_mean_train_calib": qual_mean,
        "span_quality_std_train_calib": qual_std,
    }


def classify_official(delta_vs_b1: Dict[str, float], delta_vs_b2: Dict[str, float]) -> Tuple[str, Dict[str, Any]]:
    r1_vs_b2_positive = delta_vs_b2["0.7-r1"] > 0.0 and delta_vs_b2["0.5-r1"] > 0.0
    r5_r10_vs_b1 = [delta_vs_b1[k] for k in ["0.5-r5", "0.5-r10", "0.7-r5", "0.7-r10"]]
    r5_r10_nonnegative = all(v >= 0.0 for v in r5_r10_vs_b1)
    r5_r10_negative = any(v < 0.0 for v in r5_r10_vs_b1)
    if r1_vs_b2_positive and r5_r10_nonnegative:
        status = "C7_B1_OFFICIAL_PROMOTED"
    elif r1_vs_b2_positive and r5_r10_negative:
        status = "C7_B1_R1_TRADEOFF_NO_PROMOTION"
    else:
        status = "C7_B1_OFFICIAL_NEGATIVE"
    return status, {
        "r1_vs_C6_B2_positive": bool(r1_vs_b2_positive),
        "r5_r10_vs_C6_B1_nonnegative": bool(r5_r10_nonnegative),
        "r5_r10_vs_C6_B1_negative": bool(r5_r10_negative),
        "rule": (
            "Promote only if 0.7-r1 and 0.5-r1 beat C6-B2 and all R@5/R@10 deltas vs C6-B1 are non-negative; "
            "otherwise positive R@1 with negative R@5/R@10 is marked tradeoff."
        ),
    }


def metric_table(c6_b1: Dict[str, float], c6_b2: Dict[str, float], c7: Dict[str, float],
                 d1: Dict[str, float], d2: Dict[str, float]) -> str:
    rows = [
        "| system | 0.5-r1 | 0.5-r5 | 0.5-r10 | 0.5-r100 | 0.7-r1 | 0.7-r5 | 0.7-r10 | 0.7-r100 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        "| C6-B1-lite official | " + " | ".join(f"{c6_b1[k]:.2f}" for k in METRIC_KEYS) + " |",
        "| C6-B2 official | " + " | ".join(f"{c6_b2[k]:.2f}" for k in METRIC_KEYS) + " |",
        "| C7-B1 official | " + " | ".join(f"{c7[k]:.2f}" for k in METRIC_KEYS) + " |",
        "| Delta vs C6-B1 | " + " | ".join(f"{d1[k]:+.2f}" for k in METRIC_KEYS) + " |",
        "| Delta vs C6-B2 | " + " | ".join(f"{d2[k]:+.2f}" for k in METRIC_KEYS) + " |",
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
    ready_state = validate_ready_state(args)

    cache = load_npz(args.official_cache_npz, allow_pickle=True)
    pool = load_npz(args.official_pool_npz, allow_pickle=False)
    temporal = load_npz(args.official_temporal_npz, allow_pickle=False)
    boundary = load_npz(args.official_boundary_npz, allow_pickle=False)
    b1_pool_score = pool_score_array(args.c6_b1_official_scores)
    b2_pool_score = pool_score_array(args.c6_b2_official_scores)

    c6c_feature_info = build_official_c6c_features(
        cache, pool, b1_pool_score, b2_pool_score, temporal, paths["c6c_features"]
    )
    c6c_score_info = score_official_c6c_features(args, paths["c6c_features"], paths["c6c_scores"])
    c6c_scores = load_npz(paths["c6c_scores"], allow_pickle=False)
    side = prepare_official_side(cache, pool, b1_pool_score, b2_pool_score, temporal, boundary, c6c_scores)
    structural = build_c7_official_predictions(args, paths, cache, pool, b2_pool_score, side)

    # Official evaluator: exactly one call in this script.
    submission = json.loads(paths["submission_json"].read_text(encoding="utf-8"))
    metrics_raw = eval_retrieval(
        submission,
        load_jsonl(args.gt_jsonl),
        iou_thds=(0.5, 0.7),
        verbose=False,
        match_number=True,
        use_desc_type=True,
    )
    atomic_json(paths["metrics_json"], metrics_raw)
    c7_metrics = {k: float(flatten_eval_metrics(metrics_raw)[k]) for k in METRIC_KEYS}
    c6_b1_metrics = load_metrics(args.c6_b1_metrics_json)
    c6_b2_metrics = load_metrics(args.c6_b2_metrics_json)
    delta_vs_b1 = metric_delta(c7_metrics, c6_b1_metrics)
    delta_vs_b2 = metric_delta(c7_metrics, c6_b2_metrics)
    status, rule_eval = classify_official(delta_vs_b1, delta_vs_b2)

    common = {
        "stage": "C7-B1",
        "status": status,
        "status_before_run": "C7_B1_OFFICIAL_READY",
        "date": datetime.now().strftime("%Y-%m-%d"),
        "fixed_config": FIXED_CONFIG,
        "official_val_used": True,
        "post_val_adjustment": False,
        "second_official_val": False,
        "training_started": False,
        "config_selection_changed": False,
        "mu_eta_search": False,
        "evaluator_modified": False,
        "nms_modified": False,
        "C4_modified": False,
        "C6_B1_modified": False,
        "C6_B2_modified": False,
        "enter_C7_B2": False,
        "enter_C7_C": False,
        "enter_C8": False,
        "C6_B1_official_metrics": c6_b1_metrics,
        "C6_B2_official_metrics": c6_b2_metrics,
        "C7_B1_official_metrics": c7_metrics,
        "delta_vs_C6_B1": delta_vs_b1,
        "delta_vs_C6_B2": delta_vs_b2,
        "invalid_span_count": structural["invalid_span_count"],
        "duplicate_span_count_after_nms": structural["duplicate_span_count_after_nms"],
        "structural_diagnostics": structural,
        "decision_rule_eval": rule_eval,
        "c6c_official_inference_features": {
            "purpose": "frozen C6-C evidence input for C7-B1 features only; no tuning or selection",
            "feature_info": c6c_feature_info,
            "score_info": c6c_score_info,
        },
        "ready_state_artifacts": {
            "ready_manifest": artifact(args.ready_manifest),
            "freeze_package": artifact(args.freeze_package),
        },
    }
    atomic_json(paths["one_shot_json"], common)

    manifest = {
        **common,
        "promoted": status == "C7_B1_OFFICIAL_PROMOTED",
        "one_shot_result_count": 1,
        "official_evaluator_call_count": 1,
        "ready_state_validated": True,
        "model_artifacts": {
            "c7_model": artifact(args.c7_model),
            "c6c_model": artifact(args.c6c_model),
            "c7_calib_scores": artifact(args.c7_calib_scores),
        },
        "inputs": {
            "official_cache_npz": artifact(args.official_cache_npz),
            "official_pool_npz": artifact(args.official_pool_npz),
            "official_temporal_npz": artifact(args.official_temporal_npz),
            "official_boundary_npz": artifact(args.official_boundary_npz),
            "c6_b1_official_scores": artifact(args.c6_b1_official_scores),
            "c6_b2_official_scores": artifact(args.c6_b2_official_scores),
            "c6_b1_metrics_json": artifact(args.c6_b1_metrics_json),
            "c6_b2_metrics_json": artifact(args.c6_b2_metrics_json),
        },
        "outputs": {
            "scores_npz": artifact(paths["scores_npz"]),
            "predictions_jsonl": artifact(paths["predictions_jsonl"]),
            "submission_json": artifact(paths["submission_json"]),
            "metrics_json": artifact(paths["metrics_json"]),
        },
    }
    atomic_json(paths["manifest_json"], manifest)

    decision = {
        "status": status,
        "promoted": status == "C7_B1_OFFICIAL_PROMOTED",
        "official_val_used": True,
        "post_val_adjustment": False,
        "second_official_val": False,
        "decision_rule_eval": rule_eval,
        "C7_B1_official_metrics": c7_metrics,
        "delta_vs_C6_B1": delta_vs_b1,
        "delta_vs_C6_B2": delta_vs_b2,
        "final_instruction": "Stop after exactly-one official-val one-shot and wait for human review.",
    }
    atomic_json(paths["decision_json"], decision)

    table = metric_table(c6_b1_metrics, c6_b2_metrics, c7_metrics, delta_vs_b1, delta_vs_b2)
    one_shot_md = f"""# C7-B1 official-val one-shot

- Status: `{status}`
- Fixed config: `S1_span_level_bounded_residual`, `mu=0.1`, `eta=0.1`
- Official val used: `true`
- Evaluator calls in this runner: `1`
- Training / config search / post-val adjustment / second official val: `false / false / false / false`
- evaluator_modified / nms_modified: `false / false`
- invalid_span_count / duplicate_span_count_after_nms: `{structural['invalid_span_count']} / {structural['duplicate_span_count_after_nms']}`

## Metrics

{table}

## Decision Rule

```json
{json.dumps(rule_eval, indent=2, ensure_ascii=False)}
```

Stop here and wait for human review.
"""
    atomic_text(paths["one_shot_md"], one_shot_md)
    decision_md = f"""# C7-B1 official decision

- Decision: `{status}`
- Promoted: `{str(status == 'C7_B1_OFFICIAL_PROMOTED').lower()}`
- official_val_used: `true`
- post_val_adjustment: `false`
- second_official_val: `false`

{table}
"""
    atomic_text(paths["decision_md"], decision_md)

    hashes = {
        "status": "C7_B1_OFFICIAL_HASHES",
        "official_val_used": True,
        "post_val_adjustment": False,
        "second_official_val": False,
        "metric_hash": sha256_obj(c7_metrics),
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
            "c6c_features": artifact(paths["c6c_features"]),
            "c6c_scores": artifact(paths["c6c_scores"]),
            "c7_model": artifact(args.c7_model),
            "c6c_model": artifact(args.c6c_model),
            "runner": artifact(__file__),
            "evaluator": artifact("standalone_eval/eval.py"),
        },
    }
    atomic_json(paths["hashes_json"], hashes)

    print(json.dumps({
        "status": status,
        "C6_B1_official_metrics": c6_b1_metrics,
        "C6_B2_official_metrics": c6_b2_metrics,
        "C7_B1_official_metrics": c7_metrics,
        "delta_vs_C6_B1": delta_vs_b1,
        "delta_vs_C6_B2": delta_vs_b2,
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
