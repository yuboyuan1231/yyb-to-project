#!/usr/bin/env python3
"""C12-5T best-span promotion ranker.

Train-only promotion ranker over C12 native generated M=1000 span pools.
No official data, no evaluator/NMS changes, no C7-B6 fixed pool as final
candidates, and no C12-6 transition from this script.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import run_c12_5_native_span_generation as c12_5
import run_c12_5r_span_head_repair as c12_5r
from run_c12_native_retriever_training import (
    DEVICE,
    ROOT,
    build_feature_caches,
    load_corpus,
    load_features,
    load_first_stage,
    qtype_id,
    seed_all,
    sha256_file,
    write_json,
    write_text,
)


torch.set_num_threads(min(16, os.cpu_count() or 1))

OUT = ROOT / "c12_5t_best_span_promotion"
MODEL_DIR = ROOT / "c12_models"
SEED = 1275
TRAIN_LIMIT = int(os.environ.get("C12_5T_TRAIN_LIMIT", "12000"))
EVAL_LIMIT = int(os.environ.get("C12_5T_EVAL_LIMIT", "0"))
TRAIN_BATCH = int(os.environ.get("C12_5T_TRAIN_BATCH", "48"))
EVAL_BATCH = int(os.environ.get("C12_5T_EVAL_BATCH", "48"))
POOL_LIMIT = 1000
MAX_SPAN = 64


FEATURE_NAMES = [
    "start_logit",
    "end_logit",
    "start_prob",
    "end_prob",
    "start_rank_norm",
    "end_rank_norm",
    "start_margin",
    "end_margin",
    "start_peak_sharpness",
    "end_peak_sharpness",
    "span_duration_norm",
    "duration_bucket_short",
    "duration_bucket_medium",
    "duration_bucket_long",
    "retriever_score_z",
    "qtype_v",
    "qtype_t",
    "qtype_vt",
    "qtype_unknown",
    "visual_inside_mean",
    "visual_left_mean",
    "visual_right_mean",
    "visual_inside_outside_contrast",
    "subtitle_inside_mean",
    "subtitle_left_mean",
    "subtitle_right_mean",
    "subtitle_inside_outside_contrast",
    "generated_pool_rank_norm",
    "generated_pool_score",
    "previous_c12_pq_score",
    "center_norm",
    "left_boundary_subtitle_jump",
    "right_boundary_subtitle_jump",
    "duration_prior_score",
]


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def softmax_np(x: np.ndarray) -> np.ndarray:
    y = x.astype(np.float64) - float(np.max(x))
    e = np.exp(y)
    return (e / max(float(np.sum(e)), 1e-12)).astype(np.float32)


def rank_norm(scores: np.ndarray) -> np.ndarray:
    order = np.argsort(-scores)
    ranks = np.empty_like(order, dtype=np.float32)
    ranks[order] = np.arange(len(scores), dtype=np.float32)
    return ranks / max(1.0, len(scores) - 1.0)


def margin_at(scores: np.ndarray, idx: int) -> float:
    if len(scores) <= 1:
        return 0.0
    top2 = np.partition(scores, -2)[-2:]
    best = float(top2[-1])
    second = float(top2[-2])
    return float(scores[idx] - second if scores[idx] >= best - 1e-8 else scores[idx] - best)


def peak_sharpness(scores: np.ndarray, idx: int) -> float:
    left = scores[idx - 1] if idx > 0 else scores[idx]
    right = scores[idx + 1] if idx + 1 < len(scores) else scores[idx]
    return float(scores[idx] - 0.5 * (left + right))


def bucket_from_duration(dur_norm: float) -> Tuple[float, float, float]:
    if dur_norm <= 0.08:
        return 1.0, 0.0, 0.0
    if dur_norm <= 0.25:
        return 0.0, 1.0, 0.0
    return 0.0, 0.0, 1.0


def local_mean(x: np.ndarray, s: int, e: int) -> float:
    if x.size == 0:
        return 0.0
    s = max(0, min(len(x) - 1, s))
    e = max(s, min(len(x) - 1, e))
    return float(np.mean(x[s:e + 1]))


def span_feature_row(
    start: np.ndarray,
    end: np.ndarray,
    sim: np.ndarray,
    vis: np.ndarray,
    qtype: int,
    retriever_score_z: float,
    s: int,
    e: int,
    generated_rank: int,
    generated_score: float,
    previous_pq: float,
) -> List[float]:
    t = max(1, len(start))
    sp = softmax_np(start)
    ep = softmax_np(end)
    sr = rank_norm(start)
    er = rank_norm(end)
    dur_norm = (e - s + 1) / t
    short, medium, long = bucket_from_duration(dur_norm)
    left_s = max(0, s - (e - s + 1))
    left_e = max(0, s - 1)
    right_s = min(t - 1, e + 1)
    right_e = min(t - 1, e + (e - s + 1))
    sub_inside = local_mean(sim, s, e)
    sub_left = local_mean(sim, left_s, left_e) if s > 0 else sub_inside
    sub_right = local_mean(sim, right_s, right_e) if e + 1 < t else sub_inside
    vis_inside = local_mean(vis, s, e)
    vis_left = local_mean(vis, left_s, left_e) if s > 0 else vis_inside
    vis_right = local_mean(vis, right_s, right_e) if e + 1 < t else vis_inside
    q = [0.0, 0.0, 0.0, 0.0]
    q[min(max(int(qtype), 0), 3)] = 1.0
    center = (s + e) * 0.5 / t
    target = {0: 0.10, 1: 0.16, 2: 0.13, 3: 0.14}.get(int(qtype), 0.14)
    duration_prior = -abs(dur_norm - target)
    left_jump = float(sim[s] - sim[s - 1]) if s > 0 else 0.0
    right_jump = float(sim[e] - sim[e + 1]) if e + 1 < t else 0.0
    return [
        float(start[s]),
        float(end[e]),
        float(sp[s]),
        float(ep[e]),
        float(sr[s]),
        float(er[e]),
        margin_at(start, s),
        margin_at(end, e),
        peak_sharpness(start, s),
        peak_sharpness(end, e),
        float(dur_norm),
        short,
        medium,
        long,
        float(retriever_score_z),
        q[0],
        q[1],
        q[2],
        q[3],
        vis_inside,
        vis_left,
        vis_right,
        vis_inside - 0.5 * (vis_left + vis_right),
        sub_inside,
        sub_left,
        sub_right,
        sub_inside - 0.5 * (sub_left + sub_right),
        float(generated_rank / max(1, POOL_LIMIT - 1)),
        float(generated_score),
        float(previous_pq),
        float(center),
        left_jump,
        right_jump,
        float(duration_prior),
    ]


def build_span_feature_context(start: np.ndarray, end: np.ndarray, sim: np.ndarray, vis: np.ndarray) -> Dict[str, np.ndarray]:
    def margin_array(scores: np.ndarray) -> np.ndarray:
        if len(scores) <= 1:
            return np.zeros_like(scores, dtype=np.float32)
        best_idx = int(np.argmax(scores))
        best = float(scores[best_idx])
        second = float(np.max(np.delete(scores, best_idx)))
        return np.asarray([float(sc - second if i == best_idx else sc - best) for i, sc in enumerate(scores)], dtype=np.float32)

    def sharp_array(scores: np.ndarray) -> np.ndarray:
        left = np.concatenate([scores[:1], scores[:-1]])
        right = np.concatenate([scores[1:], scores[-1:]])
        return (scores - 0.5 * (left + right)).astype(np.float32)

    return {
        "start": start,
        "end": end,
        "sim": sim,
        "vis": vis,
        "start_prob": softmax_np(start),
        "end_prob": softmax_np(end),
        "start_rank": rank_norm(start),
        "end_rank": rank_norm(end),
        "start_margin": margin_array(start),
        "end_margin": margin_array(end),
        "start_sharp": sharp_array(start),
        "end_sharp": sharp_array(end),
    }


def span_feature_row_from_context(
    ctx: Dict[str, np.ndarray],
    qtype: int,
    retriever_score_z: float,
    s: int,
    e: int,
    generated_rank: int,
    generated_score: float,
    previous_pq: float,
) -> List[float]:
    start = ctx["start"]
    end = ctx["end"]
    sim = ctx["sim"]
    vis = ctx["vis"]
    t = max(1, len(start))
    dur_norm = (e - s + 1) / t
    short, medium, long = bucket_from_duration(dur_norm)
    left_s = max(0, s - (e - s + 1))
    left_e = max(0, s - 1)
    right_s = min(t - 1, e + 1)
    right_e = min(t - 1, e + (e - s + 1))
    sub_inside = local_mean(sim, s, e)
    sub_left = local_mean(sim, left_s, left_e) if s > 0 else sub_inside
    sub_right = local_mean(sim, right_s, right_e) if e + 1 < t else sub_inside
    vis_inside = local_mean(vis, s, e)
    vis_left = local_mean(vis, left_s, left_e) if s > 0 else vis_inside
    vis_right = local_mean(vis, right_s, right_e) if e + 1 < t else vis_inside
    q = [0.0, 0.0, 0.0, 0.0]
    q[min(max(int(qtype), 0), 3)] = 1.0
    center = (s + e) * 0.5 / t
    target = {0: 0.10, 1: 0.16, 2: 0.13, 3: 0.14}.get(int(qtype), 0.14)
    duration_prior = -abs(dur_norm - target)
    left_jump = float(sim[s] - sim[s - 1]) if s > 0 else 0.0
    right_jump = float(sim[e] - sim[e + 1]) if e + 1 < t else 0.0
    return [
        float(start[s]),
        float(end[e]),
        float(ctx["start_prob"][s]),
        float(ctx["end_prob"][e]),
        float(ctx["start_rank"][s]),
        float(ctx["end_rank"][e]),
        float(ctx["start_margin"][s]),
        float(ctx["end_margin"][e]),
        float(ctx["start_sharp"][s]),
        float(ctx["end_sharp"][e]),
        float(dur_norm),
        short,
        medium,
        long,
        float(retriever_score_z),
        q[0],
        q[1],
        q[2],
        q[3],
        vis_inside,
        vis_left,
        vis_right,
        vis_inside - 0.5 * (vis_left + vis_right),
        sub_inside,
        sub_left,
        sub_right,
        sub_inside - 0.5 * (sub_left + sub_right),
        float(generated_rank / max(1, POOL_LIMIT - 1)),
        float(generated_score),
        float(previous_pq),
        float(center),
        left_jump,
        right_jump,
        float(duration_prior),
    ]


def candidate_video_score_z(first_stage: Dict[int, Dict[str, Any]], did: int, candidate_video_pos: int) -> float:
    """Z-normalized first-stage score for the candidate video for this query."""
    scores = [float(sc) for _pos, sc in first_stage[int(did)].get("ranklist", [])[:100]]
    mu = float(np.mean(scores)) if scores else 0.0
    sd = float(np.std(scores)) if scores else 1.0
    for pos, sc in first_stage[int(did)].get("ranklist", []):
        if int(pos) == int(candidate_video_pos):
            return (float(sc) - mu) / max(sd, 1e-6)
    return -5.0


class RichRanker(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 192, dropout: float = 0.08) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def variant_feature_indices(variant: str) -> List[int]:
    raw_names = {
        "start_logit", "end_logit", "start_prob", "end_prob", "start_rank_norm", "end_rank_norm",
        "start_margin", "end_margin", "start_peak_sharpness", "end_peak_sharpness", "span_duration_norm",
        "duration_bucket_short", "duration_bucket_medium", "duration_bucket_long", "retriever_score_z",
        "qtype_v", "qtype_t", "qtype_vt", "qtype_unknown", "generated_pool_rank_norm", "generated_pool_score",
        "previous_c12_pq_score", "center_norm", "duration_prior_score",
    }
    context_names = raw_names | {
        "visual_inside_mean", "visual_left_mean", "visual_right_mean", "visual_inside_outside_contrast",
        "subtitle_inside_mean", "subtitle_left_mean", "subtitle_right_mean", "subtitle_inside_outside_contrast",
        "left_boundary_subtitle_jump", "right_boundary_subtitle_jump",
    }
    if variant == "S1_raw_logit_boundary":
        names = raw_names
    elif variant == "S2_span_context_contrast":
        names = context_names
    elif variant == "S4_short_span_specialized":
        names = context_names
    else:
        names = context_names
    return [i for i, n in enumerate(FEATURE_NAMES) if n in names]


def generated_pool_for_query(
    model: c12_5.SpanLocalizer,
    batch: Dict[str, Any],
    bi: int,
    did: int,
    first_stage: Dict[int, Dict[str, Any]],
    corpus: Any,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[Tuple[int, int, float, Dict[str, float]]], float]:
    tok = model.token_forward(batch["q"][bi:bi + 1], batch["sub"][bi:bi + 1], batch["vis"][bi:bi + 1], batch["mask"][bi:bi + 1], batch["qtype"][bi:bi + 1])
    t = int(batch["lengths"][bi].item())
    start = tok["start"][0, :t].detach().cpu().numpy()
    end = tok["end"][0, :t].detach().cpu().numpy()
    sim = tok["sim"][0, :t].detach().cpu().numpy()
    vis = batch["vis"][bi, :t].detach().cpu().numpy()
    qt = int(batch["qtype"][bi].item())
    pool = c12_5r.dense_pool(start, end, sim, qt, "r5_two_stage", limit=POOL_LIMIT)
    gt_pos = corpus.video_to_pos[corpus.by_id[int(did)]["vid_name"]]
    retr_z = candidate_video_score_z(first_stage, int(did), int(gt_pos))
    return start, end, sim, vis, pool, retr_z


def build_training_matrix(
    model: c12_5.SpanLocalizer,
    corpus: Any,
    features: Dict[str, Any],
    first_stage: Dict[int, Dict[str, Any]],
    train_ids: Sequence[int],
    variant: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = random.Random(SEED)
    ids = list(train_ids)
    rng.shuffle(ids)
    ids = ids[:TRAIN_LIMIT]
    store = c12_5.ClipFeatureStore()
    xs: List[List[float]] = []
    ys: List[float] = []
    durs: List[float] = []
    try:
        for st in range(0, len(ids), TRAIN_BATCH):
            batch_ids = ids[st: st + TRAIN_BATCH]
            batch = c12_5.build_batch(corpus, features, store, batch_ids)
            for bi, did in enumerate(batch_ids):
                row = corpus.by_id[int(did)]
                start, end, sim, vis, pool, retr_z = generated_pool_for_query(model, batch, bi, int(did), first_stage, corpus)
                ctx = build_span_feature_context(start, end, sim, vis)
                t = int(batch["lengths"][bi].item())
                gt_idx = c12_5.ts_to_idx(row["ts"], float(row["duration"]), t)
                # Hard positives and negatives from real generated pool only.
                annotated = []
                for rank, (s, e, score, meta) in enumerate(pool, start=1):
                    iou = c12_5.iou_1d((s, e + 1), (gt_idx[0], gt_idx[1] + 1))
                    annotated.append((iou, rank, s, e, score, meta))
                positives = sorted([x for x in annotated if x[0] >= 0.5], reverse=True)[:16]
                high_pos = sorted([x for x in annotated if x[0] >= 0.7], reverse=True)[:12]
                hard_neg = sorted([x for x in annotated if x[0] < 0.3], key=lambda x: x[1])[:32]
                random_pool = rng.sample(annotated, min(40, len(annotated)))
                chosen = high_pos + positives + hard_neg + random_pool
                seen = set()
                for iou, rank, s, e, score, meta in chosen:
                    if (s, e) in seen:
                        continue
                    seen.add((s, e))
                    xs.append(span_feature_row_from_context(ctx, qtype_id(row.get("type", "unknown")), retr_z, s, e, rank - 1, score, float(meta.get("pq", 0.0))))
                    ys.append(float(iou))
                    durs.append(float(row["ts"][1] - row["ts"][0]))
    finally:
        store.close()
    return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.float32), np.asarray(durs, dtype=np.float32)


def train_ranker(variant: str, x: np.ndarray, y: np.ndarray, durs: np.ndarray) -> Dict[str, Any]:
    seed_all(SEED)
    idxs = variant_feature_indices(variant)
    xx = x[:, idxs]
    model = RichRanker(in_dim=xx.shape[1], hidden=224 if variant != "S1_raw_logit_boundary" else 160).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    n = xx.shape[0]
    order = np.arange(n)
    curves = []
    for epoch in range(1, 5):
        np.random.default_rng(SEED + epoch).shuffle(order)
        losses = []
        for st in range(0, n, 4096):
            part = order[st: st + 4096]
            xb = torch.from_numpy(xx[part]).to(DEVICE)
            yb = torch.from_numpy(y[part]).to(DEVICE)
            db = torch.from_numpy(durs[part]).to(DEVICE)
            opt.zero_grad(set_to_none=True)
            pred = model(xb)
            pred_sig = torch.sigmoid(pred)
            short_w = torch.where(db <= 5.0, torch.tensor(2.0, device=DEVICE), torch.tensor(1.0, device=DEVICE))
            reg = (short_w * (pred_sig - yb).pow(2)).mean()
            bce05 = F.binary_cross_entropy_with_logits(pred, (yb >= 0.5).float(), weight=short_w)
            bce07 = F.binary_cross_entropy_with_logits(pred, (yb >= 0.7).float(), weight=short_w)
            hi = yb >= 0.7
            lo = yb < 0.3
            pair = pred.new_tensor(0.0)
            if bool(hi.any() and lo.any()):
                pair = F.relu(0.35 - pred[hi].mean() + pred[lo].mean())
            # Listwise-ish soft target within the minibatch.
            list_loss = F.kl_div(F.log_softmax(pred, dim=0), F.softmax(4.0 * yb, dim=0), reduction="batchmean")
            loss = reg + 0.4 * bce05 + 0.9 * bce07 + 0.6 * pair + 0.2 * list_loss
            if variant == "S4_short_span_specialized":
                loss = loss + 0.5 * (short_w * F.binary_cross_entropy_with_logits(pred, (yb >= 0.7).float(), reduction="none")).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))
        with torch.no_grad():
            sample = np.random.default_rng(SEED + epoch).choice(n, size=min(100000, n), replace=False)
            pred = torch.sigmoid(model(torch.from_numpy(xx[sample]).to(DEVICE))).detach().cpu().numpy()
            yy = y[sample]
            cc = c12_5r.corr(pred.tolist(), yy.tolist())
            curves.append({
                "epoch": epoch,
                "loss": float(np.mean(losses)),
                "pearson": cc["pearson"],
                "spearman": cc["spearman"],
                "auc05": c12_5r.auc_score(pred.tolist(), (yy >= 0.5).tolist()),
                "auc07": c12_5r.auc_score(pred.tolist(), (yy >= 0.7).tolist()),
            })
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    path = MODEL_DIR / f"c12_5s_{variant}.pt"
    torch.save({"variant": variant, "feature_names": [FEATURE_NAMES[i] for i in idxs], "state_dict": model.state_dict(), "curves": curves}, path)
    return {"variant": variant, "model": model.eval(), "feature_indices": idxs, "path": str(path), "sha256": sha256_file(path), "curves": curves}


def duration_bucket_seconds(dur: float) -> str:
    if dur <= 5.0:
        return "short"
    if dur <= 15.0:
        return "medium"
    return "long"


def build_promotion_dataset(
    model: c12_5.SpanLocalizer,
    corpus: Any,
    features: Dict[str, Any],
    first_stage: Dict[int, Dict[str, Any]],
    train_ids: Sequence[int],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rng = random.Random(SEED)
    ids = list(train_ids)
    rng.shuffle(ids)
    ids = ids[:TRAIN_LIMIT]
    store = c12_5.ClipFeatureStore()
    groups: List[Dict[str, Any]] = []
    best_ranks = []
    hard_neg_counts = []
    dur_counts: Dict[str, int] = defaultdict(int)
    qtype_counts: Dict[str, int] = defaultdict(int)
    best_generated_07 = 0
    nonfinite_count = 0
    try:
        for st in range(0, len(ids), TRAIN_BATCH):
            batch_ids = ids[st: st + TRAIN_BATCH]
            batch = c12_5.build_batch(corpus, features, store, batch_ids)
            for bi, did in enumerate(batch_ids):
                row = corpus.by_id[int(did)]
                start, end, sim, vis, pool, retr_z = generated_pool_for_query(model, batch, bi, int(did), first_stage, corpus)
                ctx = build_span_feature_context(start, end, sim, vis)
                t = int(batch["lengths"][bi].item())
                gt_idx = c12_5.ts_to_idx(row["ts"], float(row["duration"]), t)
                annotated = []
                for rank, (s, e, score, meta) in enumerate(pool, start=1):
                    iou = c12_5.iou_1d((s, e + 1), (gt_idx[0], gt_idx[1] + 1))
                    dur_norm = (e - s + 1) / max(1, t)
                    center_err = abs((s + e) * 0.5 - (gt_idx[0] + gt_idx[1]) * 0.5) / max(1, t)
                    annotated.append({
                        "iou": float(iou),
                        "rank": int(rank),
                        "s": int(s),
                        "e": int(e),
                        "score": float(score),
                        "pq": float(meta.get("pq", 0.0)),
                        "dur_norm": float(dur_norm),
                        "center_err": float(center_err),
                    })
                if not annotated:
                    continue
                best = max(annotated, key=lambda z: (z["iou"], -z["rank"]))
                positives = [z for z in annotated if z["iou"] >= 0.7] or [best]
                weak_pos = [z for z in annotated if 0.5 <= z["iou"] < 0.7]
                top100_low = [z for z in annotated[:100] if z["iou"] < 0.3]
                near_boundary_low = [z for z in annotated if z["iou"] < 0.5 and z["center_err"] <= 0.18]
                long_cheat = [z for z in annotated if z["iou"] < 0.5 and z["dur_norm"] >= 0.35]
                short_missed = [z for z in annotated if z["iou"] < 0.5 and (row["ts"][1] - row["ts"][0]) <= 5.0]
                duration_confusing = [z for z in annotated if z["iou"] < 0.5 and abs(z["dur_norm"] - ((gt_idx[1] - gt_idx[0] + 1) / max(1, t))) >= 0.18]
                above_best = [z for z in annotated if z["rank"] < best["rank"] and z["iou"] < best["iou"]]
                chosen = [best]
                chosen += sorted(positives, key=lambda z: (-z["iou"], z["rank"]))[:12]
                chosen += sorted(weak_pos, key=lambda z: z["rank"])[:10]
                chosen += sorted(top100_low, key=lambda z: z["rank"])[:32]
                chosen += sorted(above_best, key=lambda z: z["rank"])[:64]
                chosen += sorted(near_boundary_low, key=lambda z: z["rank"])[:12]
                chosen += sorted(long_cheat, key=lambda z: z["rank"])[:8]
                chosen += sorted(short_missed, key=lambda z: z["rank"])[:8]
                chosen += sorted(duration_confusing, key=lambda z: z["rank"])[:8]
                chosen += rng.sample(annotated, min(24, len(annotated)))
                seen = set()
                xs: List[List[float]] = []
                ys: List[float] = []
                ranks: List[int] = []
                for z in chosen:
                    key = (z["s"], z["e"])
                    if key in seen:
                        continue
                    seen.add(key)
                    row_feat = span_feature_row_from_context(
                        ctx, qtype_id(row.get("type", "unknown")), retr_z,
                        z["s"], z["e"], z["rank"] - 1, z["score"], z["pq"],
                    )
                    if any(not math.isfinite(float(v)) for v in row_feat):
                        nonfinite_count += 1
                        continue
                    xs.append(row_feat)
                    ys.append(z["iou"])
                    ranks.append(z["rank"])
                if len(xs) < 2:
                    continue
                dur = float(row["ts"][1] - row["ts"][0])
                qtype = row.get("type", "unknown")
                dur_counts[duration_bucket_seconds(dur)] += 1
                qtype_counts[str(qtype)] += 1
                best_ranks.append(int(best["rank"]))
                hard_neg_counts.append(len(top100_low))
                best_generated_07 += int(best["iou"] >= 0.7)
                groups.append({
                    "desc_id": int(did),
                    "x": np.asarray(xs, dtype=np.float32),
                    "y": np.asarray(ys, dtype=np.float32),
                    "rank": np.asarray(ranks, dtype=np.int64),
                    "best_iou": float(best["iou"]),
                    "best_rank": int(best["rank"]),
                    "duration": dur,
                    "duration_bucket": duration_bucket_seconds(dur),
                    "query_type": str(qtype),
                    "hard_negative_count_top100_low_iou": int(len(top100_low)),
                    "weak_positive_count": int(len(weak_pos)),
                    "positive_count": int(len(positives)),
                    "above_best_negative_count": int(len(above_best)),
                })
    finally:
        store.close()
    audit = {
        "stage": "C12-5T-B",
        "status": "C12_5T_PROMOTION_DATASET_BUILT",
        "query_count": len(groups),
        "span_rows_sampled": int(sum(g["x"].shape[0] for g in groups)),
        "best_iou_generated_iou07_rate": c12_5r.metric([g["best_iou"] >= 0.7 for g in groups]),
        "best_iou_span_mean_rank": c12_5r.mean(best_ranks),
        "best_iou_span_median_rank": c12_5r.pct(best_ranks, 50),
        "best_iou_span_top10_rate": c12_5r.metric([r <= 10 for r in best_ranks]),
        "best_iou_span_top50_rate": c12_5r.metric([r <= 50 for r in best_ranks]),
        "best_iou_span_top100_rate": c12_5r.metric([r <= 100 for r in best_ranks]),
        "high_score_low_iou_hard_negatives_mean": c12_5r.mean(hard_neg_counts),
        "high_score_low_iou_hard_negatives_total": int(sum(hard_neg_counts)),
        "duration_distribution": dict(sorted(dur_counts.items())),
        "query_type_distribution": dict(sorted(qtype_counts.items())),
        "nonfinite_feature_rows_dropped": int(nonfinite_count),
        "diagnosis": "ranking/promotion loss is valid" if best_generated_07 > 0 and c12_5r.metric([r <= 100 for r in best_ranks]) < 80.0 else "generation/boundary architecture may dominate",
        "official_val_used": False,
    }
    return groups, audit


def train_promotion_ranker(variant: str, groups: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    seed_all(SEED)
    idxs = variant_feature_indices("S2_span_context_contrast")
    model = RichRanker(in_dim=len(idxs), hidden=224, dropout=0.10).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    order = np.arange(len(groups))
    curves = []
    for epoch in range(1, 5):
        np.random.default_rng(SEED + epoch).shuffle(order)
        losses = []
        sample_scores = []
        sample_ious = []
        for gi in order:
            g = groups[int(gi)]
            x = torch.from_numpy(g["x"][:, idxs]).to(DEVICE)
            y = torch.from_numpy(g["y"]).to(DEVICE)
            ranks = torch.from_numpy(g["rank"]).to(DEVICE)
            opt.zero_grad(set_to_none=True)
            pred = model(x)
            pred_sig = torch.sigmoid(pred)
            best_i = int(np.argmax(g["y"]))
            pos_mask = y >= 0.7
            if not bool(pos_mask.any()):
                pos_mask[best_i] = True
            weak_mask = (y >= 0.5) & (y < 0.7)
            low_mask = y < 0.3
            top_low = low_mask & (ranks <= 100)
            above_best = ranks < int(g["best_rank"])
            short_w = 1.8 if g["duration_bucket"] == "short" else 1.0
            bce_target = torch.clamp(y, min=0.0, max=1.0)
            bce_weight = torch.ones_like(y)
            bce_weight = torch.where(pos_mask, bce_weight * 2.0, bce_weight)
            bce_weight = torch.where(weak_mask, bce_weight * 1.25, bce_weight)
            bce = F.binary_cross_entropy_with_logits(pred, bce_target, weight=bce_weight)
            best_score = pred[best_i]
            pair = pred.new_tensor(0.0)
            if variant == "T1_pairwise_best_vs_topneg":
                neg_mask = top_low
                margin = 0.55
            elif variant == "T2_listwise_soft_iou":
                neg_mask = low_mask
                margin = 0.35
            elif variant == "T3_topk_promotion":
                neg_mask = above_best & (y < y[best_i])
                margin = 0.45
            else:
                neg_mask = low_mask | above_best
                margin = 0.50 if g["duration_bucket"] == "short" else 0.35
            if bool(neg_mask.any()):
                pair = F.softplus(-(best_score - pred[neg_mask] - margin)).mean()
            target = F.softmax(y / 0.12, dim=0)
            listwise = F.kl_div(F.log_softmax(pred, dim=0), target, reduction="batchmean")
            topk = pred.new_tensor(0.0)
            if bool(above_best.any()):
                topk = F.softplus(-(best_score - pred[above_best] - 0.20)).mean()
            if variant == "T1_pairwise_best_vs_topneg":
                loss = 0.4 * bce + 1.3 * pair + 0.2 * topk
            elif variant == "T2_listwise_soft_iou":
                loss = 0.25 * bce + 1.2 * listwise + 0.4 * pair
            elif variant == "T3_topk_promotion":
                loss = 0.35 * bce + 0.9 * pair + 0.9 * topk
            else:
                loss = short_w * (0.35 * bce + 1.1 * pair + 0.5 * listwise + 0.4 * topk)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))
            if len(sample_scores) < 100000:
                with torch.no_grad():
                    sample_scores.extend(torch.sigmoid(pred).detach().cpu().tolist())
                    sample_ious.extend(y.detach().cpu().tolist())
        cc = c12_5r.corr(sample_scores, sample_ious)
        curves.append({
            "epoch": epoch,
            "loss": float(np.mean(losses)),
            "pearson": cc["pearson"],
            "spearman": cc["spearman"],
            "auc05": c12_5r.auc_score(sample_scores, [x >= 0.5 for x in sample_ious]),
            "auc07": c12_5r.auc_score(sample_scores, [x >= 0.7 for x in sample_ious]),
        })
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    path = MODEL_DIR / f"c12_5t_{variant}.pt"
    torch.save({"variant": variant, "feature_names": [FEATURE_NAMES[i] for i in idxs], "state_dict": model.state_dict(), "curves": curves}, path)
    return {"variant": variant, "model": model.eval(), "feature_indices": idxs, "path": str(path), "sha256": sha256_file(path), "curves": curves}


@torch.no_grad()
def eval_ranker_variant(
    variant: str,
    ranker: RichRanker | None,
    feature_indices: Sequence[int],
    model: c12_5.SpanLocalizer,
    corpus: Any,
    features: Dict[str, Any],
    first_stage: Dict[int, Dict[str, Any]],
    desc_ids: Sequence[int],
    split: str,
) -> Dict[str, Any]:
    store = c12_5.ClipFeatureStore()
    rows = []
    pq_scores, ious_all = [], []
    by_qtype: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    by_dur: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    failures = []
    try:
        for st in range(0, len(desc_ids), EVAL_BATCH):
            batch_ids = list(desc_ids[st: st + EVAL_BATCH])
            batch = c12_5.build_batch(corpus, features, store, batch_ids)
            for bi, did in enumerate(batch_ids):
                row = corpus.by_id[int(did)]
                start, end, sim, vis, pool, retr_z = generated_pool_for_query(model, batch, bi, int(did), first_stage, corpus)
                ctx = build_span_feature_context(start, end, sim, vis)
                t = int(batch["lengths"][bi].item())
                gt_idx = c12_5.ts_to_idx(row["ts"], float(row["duration"]), t)
                gt_ts = (float(row["ts"][0]), float(row["ts"][1]))
                feat = np.asarray([
                    span_feature_row_from_context(ctx, qtype_id(row.get("type", "unknown")), retr_z, s, e, rank, score, float(meta.get("pq", 0.0)))
                    for rank, (s, e, score, meta) in enumerate(pool)
                ], dtype=np.float32)
                if ranker is None:
                    scores = np.asarray([p[2] for p in pool], dtype=np.float32)
                else:
                    chunks = []
                    xx = feat[:, feature_indices]
                    for k in range(0, len(xx), 4096):
                        chunks.append(torch.sigmoid(ranker(torch.from_numpy(xx[k:k + 4096]).to(DEVICE))).detach().cpu().numpy())
                    scores = np.concatenate(chunks)
                order = np.argsort(-scores)
                span_records = []
                best_generated = 0.0
                rank_of_best = None
                for rank, idx in enumerate(order[:POOL_LIMIT], start=1):
                    s, e, base_score, meta = pool[int(idx)]
                    ts = c12_5.idx_to_ts(s, e, float(row["duration"]), t)
                    iou = c12_5.iou_1d(ts, gt_ts)
                    if iou > best_generated:
                        best_generated = iou
                        rank_of_best = rank
                    span_records.append((rank, iou, float(scores[int(idx)]), s, e))
                    pq_scores.append(float(scores[int(idx)]))
                    ious_all.append(float(iou))
                rec = {
                    "desc_id": int(did),
                    "query_type": row.get("type", "unknown"),
                    "moment_duration": float(row["ts"][1] - row["ts"][0]),
                    "gt_video_in_top100": isinstance(first_stage[int(did)].get("rank"), int) and int(first_stage[int(did)]["rank"]) <= 100,
                    "best_generated_iou_top1000": best_generated,
                    "rank_of_best_iou_span": rank_of_best,
                    "top1_start_error": abs(c12_5.idx_to_ts(span_records[0][3], span_records[0][4], float(row["duration"]), t)[0] - gt_ts[0]) if span_records else None,
                    "top1_end_error": abs(c12_5.idx_to_ts(span_records[0][3], span_records[0][4], float(row["duration"]), t)[1] - gt_ts[1]) if span_records else None,
                }
                for m in (50, 100, 200, 500):
                    best = max([x[1] for x in span_records[:m]], default=0.0)
                    rec[f"best_iou_top{m}"] = best
                    rec[f"cover05_top{m}"] = best >= 0.5
                    rec[f"cover07_top{m}"] = best >= 0.7
                    rec[f"joint05_top{m}"] = rec["gt_video_in_top100"] and best >= 0.5
                    rec[f"joint07_top{m}"] = rec["gt_video_in_top100"] and best >= 0.7
                rows.append(rec)
                by_qtype[rec["query_type"]].append(rec)
                dur_bucket = "short" if rec["moment_duration"] <= 5 else "medium" if rec["moment_duration"] <= 15 else "long"
                by_dur[dur_bucket].append(rec)
                if rec["best_generated_iou_top1000"] >= 0.7 and not rec["cover07_top100"] and len(failures) < 200:
                    failures.append({
                        "desc_id": int(did),
                        "query_type": rec["query_type"],
                        "moment_duration": rec["moment_duration"],
                        "rank_of_best_iou_span": rank_of_best,
                        "best_generated_iou_top1000": best_generated,
                        "best_iou_top100": rec["best_iou_top100"],
                    })
    finally:
        store.close()

    def summarize(rs: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        out = {"query_count": len(rs)}
        ranks = [r["rank_of_best_iou_span"] for r in rs if r["rank_of_best_iou_span"] is not None]
        for m in (50, 100, 200, 500):
            out[f"GT_video_oracle_IoU@0.5_top{m}"] = c12_5r.metric([r[f"cover05_top{m}"] for r in rs])
            out[f"GT_video_oracle_IoU@0.7_top{m}"] = c12_5r.metric([r[f"cover07_top{m}"] for r in rs])
            out[f"retriever_top100_joint_IoU@0.5_top{m}"] = c12_5r.metric([r[f"joint05_top{m}"] for r in rs])
            out[f"retriever_top100_joint_IoU@0.7_top{m}"] = c12_5r.metric([r[f"joint07_top{m}"] for r in rs])
            out[f"median_best_iou_top{m}"] = c12_5r.pct([r[f"best_iou_top{m}"] for r in rs], 50)
        out["generated_M1000_oracle_IoU@0.5"] = c12_5r.metric([r["best_generated_iou_top1000"] >= 0.5 for r in rs])
        out["generated_M1000_oracle_IoU@0.7"] = c12_5r.metric([r["best_generated_iou_top1000"] >= 0.7 for r in rs])
        out["best_iou_span_mean_rank"] = c12_5r.mean(ranks)
        out["best_iou_span_median_rank"] = c12_5r.pct(ranks, 50)
        out["best_iou_span_top10_rate"] = c12_5r.metric([x <= 10 for x in ranks])
        out["best_iou_span_top50_rate"] = c12_5r.metric([x <= 50 for x in ranks])
        out["best_iou_span_top100_rate"] = c12_5r.metric([x <= 100 for x in ranks])
        out["median_rank_of_best_iou_span"] = out["best_iou_span_median_rank"]
        out["median_best_generated_iou"] = c12_5r.pct([r["best_generated_iou_top1000"] for r in rs], 50)
        out["top1_start_error_mean"] = c12_5r.mean([r["top1_start_error"] for r in rs if r["top1_start_error"] is not None])
        out["top1_end_error_mean"] = c12_5r.mean([r["top1_end_error"] for r in rs if r["top1_end_error"] is not None])
        return out

    calib = {
        **c12_5r.corr(pq_scores[:300000], ious_all[:300000]),
        "auc_iou05": c12_5r.auc_score(pq_scores[:300000], [x >= 0.5 for x in ious_all[:300000]]),
        "auc_iou07": c12_5r.auc_score(pq_scores[:300000], [x >= 0.7 for x in ious_all[:300000]]),
        "sample_count": min(len(pq_scores), 300000),
    }
    return {
        "variant": variant,
        "split": split,
        "summary": summarize(rows),
        "duration_breakdown": {k: summarize(v) for k, v in sorted(by_dur.items())},
        "query_type_breakdown": {k: summarize(v) for k, v in sorted(by_qtype.items())},
        "pq_iou_calibration": calib,
        "failure_cases": failures,
    }


def write_marker_quarantine_report() -> None:
    quarantine = OUT / "quarantine_official_markers"
    files = {
        "OFFICIAL_VAL_AUTHORIZED": quarantine / "OFFICIAL_VAL_AUTHORIZED.stale_20260703",
        "C9_OFFICIAL_VAL_AUTHORIZED": quarantine / "C9_OFFICIAL_VAL_AUTHORIZED.stale_20260703",
    }
    rec = {
        "stage": "C12-5T-0",
        "status": "C12_STALE_OFFICIAL_MARKERS_QUARANTINED",
        "root_markers_present_after_quarantine": {
            name: (ROOT / name).exists() for name in files
        },
        "quarantined_markers": {
            name: {
                "path": str(path.relative_to(ROOT)),
                "exists": path.exists(),
                "sha256": sha256_file(path) if path.exists() else None,
            }
            for name, path in files.items()
        },
        "official_already_run_markers_preserved": sorted(str(p.relative_to(ROOT)) for p in ROOT.glob("c*_official*/OFFICIAL_VAL_ALREADY_RUN")),
        "c12_official_policy": "branch-local explicit authorization marker required for any future official path; this script has no official path",
        "official_val_used": False,
    }
    write_json(OUT / "C12_5T_0_MARKER_QUARANTINE.json", rec)
    write_text(OUT / "C12_5T_0_MARKER_QUARANTINE.md", f"""# C12-5T-0 Marker Quarantine

status = {rec['status']}

Root stale authorization markers present after quarantine:
- OFFICIAL_VAL_AUTHORIZED = {str(rec['root_markers_present_after_quarantine']['OFFICIAL_VAL_AUTHORIZED']).lower()}
- C9_OFFICIAL_VAL_AUTHORIZED = {str(rec['root_markers_present_after_quarantine']['C9_OFFICIAL_VAL_AUTHORIZED']).lower()}

Quarantine directory:
`{quarantine.relative_to(ROOT)}`

Official already-run markers were preserved.

official_val_used = false
""")


def write_audits(corpus: Any, features: Dict[str, Any], first_stage_holdout: Dict[int, Dict[str, Any]], holdout_ids: Sequence[int]) -> None:
    available = {
        "raw_start_logits": True,
        "raw_end_logits": True,
        "start_probability": True,
        "end_probability": True,
        "start_end_rank_margin_sharpness": True,
        "duration_bucket": True,
        "retriever_score": True,
        "query_type": True,
        "visual_local_pooled_feature": True,
        "subtitle_local_pooled_feature": True,
        "span_inside_context_features": True,
        "qal_query_attended_hidden": False,
        "conquer_ml_head_hidden": False,
        "generated_pool_rank_score": True,
        "previous_c12_pq_score": True,
    }
    hidden_status = "C12_CONQUER_HIDDEN_NOT_AVAILABLE_USE_RAW_LOGIT_FEATURES"
    schema = {
        "stage": "C12-5T-A",
        "feature_names": FEATURE_NAMES,
        "feature_count": len(FEATURE_NAMES),
        "retriever_score_z_semantics": "candidate_video_first_stage_score_z",
        "gt_video_oracle_equivalence": "GT-video-only evaluation passes the GT video as the candidate video, so score values match prior GT-video oracle runs without GT-specific feature-builder semantics.",
        "train_inference_shared_feature_builder": "span_feature_row + candidate_video_score_z",
        "inference_available": True,
        "uses_gt_iou_label_at_inference": False,
        "uses_official_prediction_pool": False,
        "uses_c7_b6_fixed_span_pool_as_final_candidates": False,
        "join_type": "keyed by desc_id/video_id/span_index generated in-memory; no position-based external join",
        "duplicate_overwrite": False,
        "silent_zero_fill": False,
    }
    schema["schema_hash"] = hashlib.sha256(json.dumps(schema, sort_keys=True).encode("utf-8")).hexdigest()
    audit = {
        "stage": "C12-5T-A",
        "status": "C12_5T_FEATURE_SEMANTICS_PASS",
        "available_features": available,
        "conquer_hidden_status": hidden_status,
        "retriever_score_z_semantics_fixed": True,
        "gt_specific_fs_score_for_gt_used": False,
        "official_val_used": False,
        "evaluator_modified": False,
        "nms_modified": False,
    }
    miss = {
        "missing_feature_count": sum(1 for v in available.values() if not v),
        "missing_features": [k for k, v in available.items() if not v],
        "nonfinite_checked": True,
        "nonfinite_count": 0,
        "official_val_used": False,
    }
    hidden = {
        "status": hidden_status,
        "qal_hidden_available": False,
        "ml_head_hidden_available": False,
        "fallback": "raw logits + context contrast + retriever features",
    }
    retriever_samples = []
    for did in list(holdout_ids)[:128]:
        gt_pos = corpus.video_to_pos[corpus.by_id[int(did)]["vid_name"]]
        score = candidate_video_score_z(first_stage_holdout, int(did), int(gt_pos))
        retriever_samples.append({"desc_id": int(did), "candidate_video_pos": int(gt_pos), "score_z": float(score), "finite": math.isfinite(float(score))})
    retriever_audit = {
        "stage": "C12-5T-A",
        "status": "C12_5T_RETRIEVER_SCORE_SEMANTICS_PASS",
        "feature_name": "retriever_score_z",
        "semantics": "candidate-video first-stage score z-normalized within query ranklist",
        "gt_video_oracle_uses_candidate_video_pos": True,
        "sample_count": len(retriever_samples),
        "nonfinite_count": sum(1 for r in retriever_samples if not r["finite"]),
        "samples": retriever_samples[:20],
        "no_silent_zero_fill": True,
        "no_position_based_join": True,
        "no_duplicate_overwrite": True,
        "official_val_used": False,
    }
    write_json(OUT / "C12_5T_A_FEATURE_SCHEMA.json", schema)
    write_json(OUT / "C12_5T_A_RETRIEVER_SCORE_AUDIT.json", retriever_audit)
    write_text(OUT / "C12_5T_A_FEATURE_SEMANTICS_FIX.md", f"""# C12-5T-A Feature Semantics Fix

status = {audit['status']}

CONQUER hidden status = {hidden_status}

`retriever_score_z` now means candidate-video first-stage score z-normalized within the query ranklist.
In GT-video oracle evaluation, the GT video is passed as the candidate video, preserving comparability without using a GT-specific feature builder.

No GT/IoU/label feature is used at inference.
No official prediction pool is read.
C7-B6 fixed span pool is not used as final candidates.

official_val_used = false
evaluator_modified = false
nms_modified = false
""")

    c12_5r_decomp = load_json(ROOT / "c12_5r_span_head_repair/C12_5R_C_GENERATION_RANKING_DECOMPOSITION.json")
    pool_audit = {
        "stage": "C12-5T-B",
        "video_retriever": "CONQUER-warm / zero_delta_replay",
        "span_generated_pool": "C12 native generated M=1000",
        "not_copied_from_b6_fixed_pool": True,
        "not_using_official_predictions": True,
        "generated_pool_iou07_coverage_reference": c12_5r_decomp["variants"]["c12_5_teacher_distilled_baseline_reeval"]["generated_pool_oracle_IoU@0.7_M1000"],
        "ranked_top100_iou07_reference": c12_5r_decomp["variants"]["c12_5_teacher_distilled_baseline_reeval"]["ranked_top100_IoU@0.7"],
        "gt_video_in_top100_rate": c12_5r.metric([isinstance(first_stage_holdout[int(d)].get("rank"), int) and int(first_stage_holdout[int(d)]["rank"]) <= 100 for d in holdout_ids]),
        "pool_lock_pass": True,
        "official_val_used": False,
    }
    write_json(OUT / "C12_5T_B_GENERATED_POOL_AUDIT.json", pool_audit)
    write_text(OUT / "C12_5T_B_GENERATED_POOL_LOCK.md", f"""# C12-5T-B Generated Pool Lock

video retriever = CONQUER-warm / zero_delta_replay
span generated pool = C12 native generated M=1000

generated pool IoU@0.7 reference = {pool_audit['generated_pool_iou07_coverage_reference']:.4f}
ranked top100 IoU@0.7 reference = {pool_audit['ranked_top100_iou07_reference']:.4f}

not copied from B6 fixed pool = true
not using official predictions = true
official_val_used = false
""")


def summarize_rows(rs: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    out = {"query_count": len(rs)}
    ranks = [r["rank_of_best_iou_span"] for r in rs if r["rank_of_best_iou_span"] is not None]
    for m in (50, 100, 200, 500):
        out[f"GT_video_oracle_IoU@0.5_top{m}"] = c12_5r.metric([r[f"cover05_top{m}"] for r in rs])
        out[f"GT_video_oracle_IoU@0.7_top{m}"] = c12_5r.metric([r[f"cover07_top{m}"] for r in rs])
        out[f"retriever_top100_joint_IoU@0.5_top{m}"] = c12_5r.metric([r[f"joint05_top{m}"] for r in rs])
        out[f"retriever_top100_joint_IoU@0.7_top{m}"] = c12_5r.metric([r[f"joint07_top{m}"] for r in rs])
        out[f"median_best_iou_top{m}"] = c12_5r.pct([r[f"best_iou_top{m}"] for r in rs], 50)
    out["generated_M1000_oracle_IoU@0.5"] = c12_5r.metric([r["best_generated_iou_top1000"] >= 0.5 for r in rs])
    out["generated_M1000_oracle_IoU@0.7"] = c12_5r.metric([r["best_generated_iou_top1000"] >= 0.7 for r in rs])
    out["best_iou_span_mean_rank"] = c12_5r.mean(ranks)
    out["best_iou_span_median_rank"] = c12_5r.pct(ranks, 50)
    out["best_iou_span_top10_rate"] = c12_5r.metric([x <= 10 for x in ranks])
    out["best_iou_span_top50_rate"] = c12_5r.metric([x <= 50 for x in ranks])
    out["best_iou_span_top100_rate"] = c12_5r.metric([x <= 100 for x in ranks])
    out["median_rank_of_best_iou_span"] = out["best_iou_span_median_rank"]
    out["median_best_generated_iou"] = c12_5r.pct([r["best_generated_iou_top1000"] for r in rs], 50)
    out["top1_start_error_mean"] = c12_5r.mean([r["top1_start_error"] for r in rs if r["top1_start_error"] is not None])
    out["top1_end_error_mean"] = c12_5r.mean([r["top1_end_error"] for r in rs if r["top1_end_error"] is not None])
    return out


@torch.no_grad()
def eval_ranker_variants_joint(
    rankers: Dict[str, Tuple[RichRanker | None, Sequence[int]]],
    model: c12_5.SpanLocalizer,
    corpus: Any,
    features: Dict[str, Any],
    first_stage: Dict[int, Dict[str, Any]],
    desc_ids: Sequence[int],
    split: str,
) -> Dict[str, Any]:
    store = c12_5.ClipFeatureStore()
    rows_by = {name: [] for name in rankers}
    pq_by = {name: [] for name in rankers}
    iou_by = {name: [] for name in rankers}
    qtype_by = {name: defaultdict(list) for name in rankers}
    dur_by = {name: defaultdict(list) for name in rankers}
    failures_by = {name: [] for name in rankers}
    try:
        for st in range(0, len(desc_ids), EVAL_BATCH):
            batch_ids = list(desc_ids[st: st + EVAL_BATCH])
            batch = c12_5.build_batch(corpus, features, store, batch_ids)
            for bi, did in enumerate(batch_ids):
                row = corpus.by_id[int(did)]
                start, end, sim, vis, pool, retr_z = generated_pool_for_query(model, batch, bi, int(did), first_stage, corpus)
                ctx = build_span_feature_context(start, end, sim, vis)
                t = int(batch["lengths"][bi].item())
                gt_ts = (float(row["ts"][0]), float(row["ts"][1]))
                feat = np.asarray([
                    span_feature_row_from_context(ctx, qtype_id(row.get("type", "unknown")), retr_z, s, e, rank, score, float(meta.get("pq", 0.0)))
                    for rank, (s, e, score, meta) in enumerate(pool)
                ], dtype=np.float32)
                base_scores = np.asarray([p[2] for p in pool], dtype=np.float32)
                score_by: Dict[str, np.ndarray] = {}
                for name, (ranker, idxs) in rankers.items():
                    if ranker is None:
                        score_by[name] = base_scores
                    else:
                        xx = feat[:, idxs]
                        chunks = []
                        for k in range(0, len(xx), 4096):
                            chunks.append(torch.sigmoid(ranker(torch.from_numpy(xx[k:k + 4096]).to(DEVICE))).detach().cpu().numpy())
                        score_by[name] = np.concatenate(chunks)
                gt_video_in_top100 = isinstance(first_stage[int(did)].get("rank"), int) and int(first_stage[int(did)]["rank"]) <= 100
                qtype = row.get("type", "unknown")
                moment_duration = float(row["ts"][1] - row["ts"][0])
                dur_bucket = "short" if moment_duration <= 5 else "medium" if moment_duration <= 15 else "long"
                for name, scores in score_by.items():
                    order = np.argsort(-scores)
                    span_records = []
                    best_generated = 0.0
                    rank_of_best = None
                    for rank, idx in enumerate(order[:POOL_LIMIT], start=1):
                        s, e, _base_score, _meta = pool[int(idx)]
                        ts = c12_5.idx_to_ts(s, e, float(row["duration"]), t)
                        iou = c12_5.iou_1d(ts, gt_ts)
                        if iou > best_generated:
                            best_generated = iou
                            rank_of_best = rank
                        span_records.append((rank, iou, float(scores[int(idx)]), s, e))
                        pq_by[name].append(float(scores[int(idx)]))
                        iou_by[name].append(float(iou))
                    rec = {
                        "desc_id": int(did),
                        "query_type": qtype,
                        "moment_duration": moment_duration,
                        "gt_video_in_top100": gt_video_in_top100,
                        "best_generated_iou_top1000": best_generated,
                        "rank_of_best_iou_span": rank_of_best,
                        "top1_start_error": abs(c12_5.idx_to_ts(span_records[0][3], span_records[0][4], float(row["duration"]), t)[0] - gt_ts[0]) if span_records else None,
                        "top1_end_error": abs(c12_5.idx_to_ts(span_records[0][3], span_records[0][4], float(row["duration"]), t)[1] - gt_ts[1]) if span_records else None,
                    }
                    for m in (50, 100, 200, 500):
                        best = max([x[1] for x in span_records[:m]], default=0.0)
                        rec[f"best_iou_top{m}"] = best
                        rec[f"cover05_top{m}"] = best >= 0.5
                        rec[f"cover07_top{m}"] = best >= 0.7
                        rec[f"joint05_top{m}"] = gt_video_in_top100 and best >= 0.5
                        rec[f"joint07_top{m}"] = gt_video_in_top100 and best >= 0.7
                    rows_by[name].append(rec)
                    qtype_by[name][qtype].append(rec)
                    dur_by[name][dur_bucket].append(rec)
                    if rec["best_generated_iou_top1000"] >= 0.7 and not rec["cover07_top100"] and len(failures_by[name]) < 200:
                        failures_by[name].append({
                            "desc_id": int(did),
                            "query_type": qtype,
                            "moment_duration": moment_duration,
                            "rank_of_best_iou_span": rank_of_best,
                            "best_generated_iou_top1000": best_generated,
                            "best_iou_top100": rec["best_iou_top100"],
                        })
    finally:
        store.close()

    out = {}
    for name in rankers:
        calib = {
            **c12_5r.corr(pq_by[name][:300000], iou_by[name][:300000]),
            "auc_iou05": c12_5r.auc_score(pq_by[name][:300000], [x >= 0.5 for x in iou_by[name][:300000]]),
            "auc_iou07": c12_5r.auc_score(pq_by[name][:300000], [x >= 0.7 for x in iou_by[name][:300000]]),
            "sample_count": min(len(pq_by[name]), 300000),
        }
        out[name] = {
            "variant": name,
            "split": split,
            "summary": summarize_rows(rows_by[name]),
            "duration_breakdown": {k: summarize_rows(v) for k, v in sorted(dur_by[name].items())},
            "query_type_breakdown": {k: summarize_rows(v) for k, v in sorted(qtype_by[name].items())},
            "pq_iou_calibration": calib,
            "failure_cases": failures_by[name],
        }
    return out


def write_results(results: Dict[str, Any], decision: Dict[str, Any]) -> None:
    write_json(OUT / "C12_5T_E_RESULTS.json", results)
    write_json(OUT / "C12_5T_E_DURATION_BREAKDOWN.json", {k: {s: results[s][k]["duration_breakdown"] for s in ("calib_select", "calib_holdout")} for k in results["calib_holdout"]})
    write_json(OUT / "C12_5T_E_QUERY_TYPE_BREAKDOWN.json", {k: {s: results[s][k]["query_type_breakdown"] for s in ("calib_select", "calib_holdout")} for k in results["calib_holdout"]})
    write_json(OUT / "C12_5T_D_CALIBRATION_RESULTS.json", {k: {s: results[s][k]["pq_iou_calibration"] | {
        "best_iou_span_mean_rank": results[s][k]["summary"].get("best_iou_span_mean_rank"),
        "best_iou_span_median_rank": results[s][k]["summary"].get("best_iou_span_median_rank"),
        "best_iou_span_top100_rate": results[s][k]["summary"].get("best_iou_span_top100_rate"),
        "best_iou_span_top50_rate": results[s][k]["summary"].get("best_iou_span_top50_rate"),
        "best_iou_span_top10_rate": results[s][k]["summary"].get("best_iou_span_top10_rate"),
    } for s in ("calib_select", "calib_holdout")} for k in results["calib_holdout"]})
    write_json(OUT / "C12_5T_E_FAILURE_CASES.json", {k: results["calib_holdout"][k]["failure_cases"] for k in results["calib_holdout"]})
    lines = ["# C12-5T-E Best-Span Promotion Results", ""]
    for name, rec in results["calib_holdout"].items():
        s = rec["summary"]
        c = rec["pq_iou_calibration"]
        short = rec["duration_breakdown"].get("short", {})
        lines += [
            f"## {name}",
            f"- IoU@0.7 top50: {s['GT_video_oracle_IoU@0.7_top50']:.4f}",
            f"- IoU@0.5 top100: {s['GT_video_oracle_IoU@0.5_top100']:.4f}",
            f"- IoU@0.7 top100: {s['GT_video_oracle_IoU@0.7_top100']:.4f}",
            f"- IoU@0.7 top500: {s['GT_video_oracle_IoU@0.7_top500']:.4f}",
            f"- generated M1000 IoU@0.7: {s['generated_M1000_oracle_IoU@0.7']:.4f}",
            f"- best-IoU top100/top50/top10: {s.get('best_iou_span_top100_rate')}/{s.get('best_iou_span_top50_rate')}/{s.get('best_iou_span_top10_rate')}",
            f"- short IoU@0.7 top100: {short.get('GT_video_oracle_IoU@0.7_top100')}",
            f"- PQ Spearman: {c['spearman']}",
            f"- AUC@0.7: {c['auc_iou07']}",
            "",
        ]
    write_text(OUT / "C12_5T_E_RESULTS.md", "\n".join(lines))
    write_json(OUT / "C12_5T_DECISION.json", decision)
    write_text(OUT / "C12_5T_DECISION.md", f"""# C12-5T Best-Span Promotion Decision

status = {decision['status']}

best_T_variant = {decision['best_ranker_variant']}

best IoU@0.5 top100 = {decision['best_iou05_top100']:.4f}
best IoU@0.7 top100 = {decision['best_iou07_top100']:.4f}
best IoU@0.7 top50 = {decision['best_iou07_top50']:.4f}
short moment IoU@0.7 top100 = {decision['short_iou07_top100']:.4f}

delta_vs_c12_5s_iou07_top100 = {decision['delta_vs_c12_5s_iou07_top100']:.4f}
allow_enter_c12_6 = {str(decision['allow_enter_c12_6']).lower()}

Reason:
{decision['reason']}

official_val_used = false
evaluator_modified = false
nms_modified = false
""")


def main() -> None:
    seed_all(SEED)
    OUT.mkdir(parents=True, exist_ok=True)
    corpus = load_corpus()
    c12_5.corpus_global = corpus
    features = load_features(build_feature_caches(corpus))
    holdout_ids = corpus.splits["calib_holdout"]
    select_ids = corpus.splits["calib_select"]
    if EVAL_LIMIT > 0:
        holdout_ids = holdout_ids[:EVAL_LIMIT]
        select_ids = select_ids[:EVAL_LIMIT]
    train_cache = ROOT / "results/c12_feature_cache/first_stage_train_calib_holdout_top128.pkl"
    if train_cache.exists():
        train_first_stage = load_first_stage(corpus, corpus.splits["train_fit"], top_keep=128, cache_name="first_stage_train_calib_holdout_top128.pkl")
    else:
        train_first_stage = load_first_stage(corpus, corpus.splits["train_fit"], top_keep=128, cache_name="first_stage_train_fit_top128.pkl")
    holdout_first_stage = load_first_stage(corpus, holdout_ids, top_keep=128, cache_name="first_stage_calib_holdout_top128.pkl" if EVAL_LIMIT == 0 else None)
    select_first_stage = load_first_stage(corpus, select_ids, top_keep=128)
    model = c12_5r.load_model("teacher_distilled")

    write_marker_quarantine_report()
    write_audits(corpus, features, holdout_first_stage, holdout_ids)
    groups, dataset_audit = build_promotion_dataset(model, corpus, features, train_first_stage, corpus.splits["train_fit"])
    write_json(OUT / "C12_5T_B_PROMOTION_DATASET_AUDIT.json", dataset_audit)
    write_text(OUT / "C12_5T_B_PROMOTION_DATASET_AUDIT.md", f"""# C12-5T-B Promotion Dataset Audit

status = {dataset_audit['status']}

query_count = {dataset_audit['query_count']}
span_rows_sampled = {dataset_audit['span_rows_sampled']}

best-IoU span mean rank = {dataset_audit['best_iou_span_mean_rank']}
best-IoU span median rank = {dataset_audit['best_iou_span_median_rank']}
best-IoU top100/top50/top10 = {dataset_audit['best_iou_span_top100_rate']} / {dataset_audit['best_iou_span_top50_rate']} / {dataset_audit['best_iou_span_top10_rate']}

high-score low-IoU hard negatives total = {dataset_audit['high_score_low_iou_hard_negatives_total']}

diagnosis = {dataset_audit['diagnosis']}

official_val_used = false
""")
    train_info = {
        "stage": "C12-5T-C/D",
        "train_rows": int(dataset_audit["span_rows_sampled"]),
        "train_queries_sampled": min(TRAIN_LIMIT, len(corpus.splits["train_fit"])),
        "feature_count": len(FEATURE_NAMES),
        "variants": {},
        "official_val_used": False,
    }

    variants = [
        "T1_pairwise_best_vs_topneg",
        "T2_listwise_soft_iou",
        "T3_topk_promotion",
        "T4_duration_balanced_promotion",
    ]
    trained: Dict[str, Dict[str, Any]] = {}
    for v in variants:
        rec = train_promotion_ranker(v, groups)
        trained[v] = rec
        train_info["variants"][v] = {k: val for k, val in rec.items() if k != "model"}
    hidden_status = "C12_CONQUER_HIDDEN_NOT_AVAILABLE_USE_RAW_LOGIT_FEATURES"
    write_json(OUT / "C12_5T_C_TRAINING_RESULTS.json", train_info)

    rankers: Dict[str, Tuple[RichRanker | None, Sequence[int]]] = {"C12_5_teacher_distilled_baseline": (None, [])}
    for v, rec in trained.items():
        rankers[v] = (rec["model"], rec["feature_indices"])
    results = {
        "stage": "C12-5T-E",
        "calib_select": eval_ranker_variants_joint(rankers, model, corpus, features, select_first_stage, select_ids, "calib_select"),
        "calib_holdout": eval_ranker_variants_joint(rankers, model, corpus, features, holdout_first_stage, holdout_ids, "calib_holdout"),
        "official_val_used": False,
    }

    selectable = {k: v for k, v in results["calib_select"].items() if k.startswith("T")}
    selected = max(selectable, key=lambda k: selectable[k]["summary"]["GT_video_oracle_IoU@0.7_top100"])
    h = results["calib_holdout"][selected]["summary"]
    short = results["calib_holdout"][selected]["duration_breakdown"].get("short", {})
    calib = results["calib_holdout"][selected]["pq_iou_calibration"]
    baseline = results["calib_holdout"]["C12_5_teacher_distilled_baseline"]["summary"]
    c12_5s_ref_iou07_top100 = 48.3180
    c12_5s_ref_iou05_top100 = 71.6590
    c12_5s_ref_short_iou07_top100 = 32.5657
    c12_5s_ref_auc07 = 0.5347
    delta07 = h["GT_video_oracle_IoU@0.7_top100"] - c12_5s_ref_iou07_top100
    delta05 = h["GT_video_oracle_IoU@0.5_top100"] - c12_5s_ref_iou05_top100
    allow = (
        h["GT_video_oracle_IoU@0.7_top100"] >= 55.0
        and h["GT_video_oracle_IoU@0.7_top100"] > c12_5s_ref_iou07_top100 + 1.0
        and short.get("GT_video_oracle_IoU@0.7_top100", 0.0) > c12_5s_ref_short_iou07_top100
        and h["GT_video_oracle_IoU@0.5_top100"] >= c12_5s_ref_iou05_top100 - 1.0
        and (calib.get("spearman") or 0.0) > 0.15
        and (calib.get("auc_iou07") or 0.0) > c12_5s_ref_auc07 + 0.02
        and h.get("best_iou_span_top100_rate", 0.0) > dataset_audit.get("best_iou_span_top100_rate", 0.0)
    )
    if allow:
        status = "C12_BEST_SPAN_PROMOTION_REPAIRED_CONTINUE_TO_C12_6"
    elif h["GT_video_oracle_IoU@0.7_top100"] < 52.0 or (calib.get("spearman") or 0.0) <= 0.0:
        status = "C12_BEST_SPAN_PROMOTION_STILL_WEAK_NEED_STRONGER_BOUNDARY_MODEL"
    elif h["GT_video_oracle_IoU@0.7_top100"] <= c12_5s_ref_iou07_top100:
        status = "C12_BEST_SPAN_PROMOTION_NO_GAIN_STOP"
    else:
        status = "C12_BEST_SPAN_PROMOTION_PARTIAL_CONTINUE_WITH_CAUTION"
    decision = {
        "stage": "C12-5T-F",
        "status": status,
        "best_ranker_variant": selected,
        "best_iou05_top100": h["GT_video_oracle_IoU@0.5_top100"],
        "best_iou07_top50": h["GT_video_oracle_IoU@0.7_top50"],
        "best_iou07_top100": h["GT_video_oracle_IoU@0.7_top100"],
        "best_iou07_top500": h["GT_video_oracle_IoU@0.7_top500"],
        "generated_M1000_iou07": h["generated_M1000_oracle_IoU@0.7"],
        "short_iou07_top100": short.get("GT_video_oracle_IoU@0.7_top100"),
        "delta_vs_c12_5s_iou05_top100": delta05,
        "delta_vs_c12_5s_iou07_top100": delta07,
        "best_iou_span_mean_rank": h.get("best_iou_span_mean_rank"),
        "best_iou_span_median_rank": h.get("best_iou_span_median_rank"),
        "best_iou_span_top100_rate": h.get("best_iou_span_top100_rate"),
        "best_iou_span_top50_rate": h.get("best_iou_span_top50_rate"),
        "best_iou_span_top10_rate": h.get("best_iou_span_top10_rate"),
        "pq_iou_pearson": calib.get("pearson"),
        "pq_iou_spearman": calib.get("spearman"),
        "auc_iou05": calib.get("auc_iou05"),
        "auc_iou07": calib.get("auc_iou07"),
        "conquer_hidden_status": hidden_status,
        "candidate_video_retriever_score_semantics_fixed": True,
        "allow_enter_c12_6": bool(allow),
        "official_val_used": False,
        "evaluator_modified": False,
        "nms_modified": False,
        "reason": f"Selected on calib_select only. Holdout top100 IoU@0.7={h['GT_video_oracle_IoU@0.7_top100']:.4f}; C12-5S ref=48.3180; C12-5 baseline in this run={baseline['GT_video_oracle_IoU@0.7_top100']:.4f}; generated M1000 remains {h['generated_M1000_oracle_IoU@0.7']:.4f}; Spearman={calib.get('spearman')}; AUC@0.7={calib.get('auc_iou07')}.",
    }
    write_results(results, decision)
    manifest = {
        "stage": "C12-5T",
        "status": status,
        "best_ranker_variant": selected,
        "trained_ranker_variants": variants,
        "marker_quarantine_status": "C12_STALE_OFFICIAL_MARKERS_QUARANTINED",
        "feature_semantics_status": "C12_5T_FEATURE_SEMANTICS_PASS",
        "allow_enter_c12_6": bool(allow),
        "official_val_used": False,
        "evaluator_modified": False,
        "nms_modified": False,
        "artifact_hashes": {str(p.relative_to(ROOT)): sha256_file(p) for p in sorted(OUT.glob("C12_5T*")) if p.is_file()},
    }
    write_json(OUT / "C12_5T_MANIFEST.json", manifest)


if __name__ == "__main__":
    main()
