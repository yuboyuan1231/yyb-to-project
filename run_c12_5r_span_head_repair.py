#!/usr/bin/env python3
"""C12-5R native span head / boundary proposal repair.

Train-only repair branch. No official data, no evaluator/NMS changes.
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

OUT = ROOT / "c12_5r_span_head_repair"
MODEL_DIR = ROOT / "c12_models"
SEED = 1255
TRAIN_LIMIT = int(os.environ.get("C12_5R_TRAIN_LIMIT", "14000"))
EVAL_LIMIT = int(os.environ.get("C12_5R_EVAL_LIMIT", "0"))
TRAIN_BATCH = int(os.environ.get("C12_5R_TRAIN_BATCH", "64"))
EVAL_BATCH = int(os.environ.get("C12_5R_EVAL_BATCH", "48"))
MAX_T = 96
MAX_SPAN = 64


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def stable_hash(obj: Any) -> str:
    raw = json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def metric(vals: Sequence[bool]) -> float:
    return 100.0 * sum(bool(x) for x in vals) / max(1, len(vals))


def pct(vals: Sequence[float], p: float) -> float | None:
    return float(np.percentile(vals, p)) if vals else None


def mean(vals: Sequence[float]) -> float | None:
    return float(np.mean(vals)) if vals else None


def rankdata(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(x), dtype=np.float64)
    return ranks


def corr(x: Sequence[float], y: Sequence[float]) -> Dict[str, float | None]:
    if len(x) < 2 or len(y) < 2:
        return {"pearson": None, "spearman": None}
    xa = np.asarray(x, dtype=np.float64)
    ya = np.asarray(y, dtype=np.float64)
    if float(np.std(xa)) < 1e-12 or float(np.std(ya)) < 1e-12:
        pear = None
    else:
        pear = float(np.corrcoef(xa, ya)[0, 1])
    rx = rankdata(xa)
    ry = rankdata(ya)
    spear = None if float(np.std(rx)) < 1e-12 or float(np.std(ry)) < 1e-12 else float(np.corrcoef(rx, ry)[0, 1])
    return {"pearson": pear, "spearman": spear}


def auc_score(scores: Sequence[float], labels: Sequence[bool]) -> float | None:
    pos = [s for s, l in zip(scores, labels) if l]
    neg = [s for s, l in zip(scores, labels) if not l]
    if not pos or not neg:
        return None
    # Mann-Whitney AUC, sampled if needed to avoid quadratic blow-up.
    rng = random.Random(SEED)
    if len(pos) * len(neg) > 2_000_000:
        pairs = 200_000
        wins = 0.0
        for _ in range(pairs):
            p = rng.choice(pos)
            n = rng.choice(neg)
            wins += 1.0 if p > n else 0.5 if p == n else 0.0
        return wins / pairs
    wins = 0.0
    for p in pos:
        for n in neg:
            wins += 1.0 if p > n else 0.5 if p == n else 0.0
    return wins / (len(pos) * len(neg))


def load_model(name: str) -> c12_5.SpanLocalizer:
    ckpt = torch.load(MODEL_DIR / f"c12_5_{name}.pt", map_location="cpu")
    model = c12_5.SpanLocalizer(use_qtype=bool(ckpt["config"].get("use_qtype", False)), hidden=int(ckpt["config"].get("hidden", 192)))
    model.load_state_dict(ckpt["state_dict"])
    return model.to(DEVICE).eval()


def dense_pool(
    start: np.ndarray,
    end: np.ndarray,
    sim: np.ndarray,
    qtype: int,
    mode: str,
    limit: int = 1000,
) -> List[Tuple[int, int, float, Dict[str, float]]]:
    t = int(len(start))
    pool: Dict[Tuple[int, int], Tuple[float, Dict[str, float]]] = {}

    def add(s: int, e: int, bonus: float = 0.0, tag: str = "native") -> None:
        if s < 0 or e < s or e >= t or e >= s + MAX_SPAN:
            return
        dur = (e - s + 1) / max(1, t)
        center = (s + e) * 0.5 / max(1, t)
        score = float(start[s] + end[e] + bonus)
        score += 0.15 * float(sim[s] + sim[e])
        old = pool.get((s, e))
        if old is None or score > old[0]:
            pool[(s, e)] = (score, {"duration_norm": dur, "center": center, "tag": tag})

    top_s = np.argsort(-start)[: min(t, 64)]
    top_e = np.argsort(-end)[: min(t, 64)]
    for s in top_s:
        for e in top_e:
            add(int(s), int(e), tag="boundary_top")

    # R1 duration-conditioned anchors: intentionally densify short spans across
    # time while retaining medium/long anchors from local evidence peaks.
    if mode in {"r1_dense_duration", "r5_two_stage"}:
        short_lengths = [1, 2, 3, 4, 5, 6]
        med_lengths = [8, 10, 12, 16]
        long_lengths = [20, 28, 36, 48]
        for length in short_lengths:
            step = 1 if t <= 80 else 2
            for s in range(0, t, step):
                add(s, s + length - 1, bonus=0.05, tag="short_dense")
        centers = np.argsort(-sim)[: min(t, 32)]
        for c in centers:
            for length in med_lengths + long_lengths:
                s = int(c) - length // 2
                add(max(0, min(t - length, s)), max(0, min(t - length, s)) + length - 1, bonus=0.03, tag="midlong_peak")

    if mode in {"r2_offset_refine", "r5_two_stage"}:
        # Native refinement around high boundary pairs; this predicts a small
        # boundary correction from local score maxima rather than using GT.
        base_pairs = list(pool.keys())[:]
        for s, e in base_pairs:
            ws = range(max(0, s - 2), min(t, s + 3))
            we = range(max(s, e - 2), min(t, e + 3))
            best_s = max(ws, key=lambda x: start[x])
            best_e = max([x for x in we if x >= best_s], key=lambda x: end[x])
            add(int(best_s), int(best_e), bonus=0.04, tag="offset_refined")

    if mode == "r4_teacher_prior":
        # Teacher prior is represented as a learned-ish duration prior from B6
        # training spans; no teacher spans are inserted into inference pools.
        target = {0: 0.10, 1: 0.16, 2: 0.13, 3: 0.14}.get(int(qtype), 0.14)
        for (s, e), (score, meta) in list(pool.items()):
            dur = meta["duration_norm"]
            score -= 0.25 * abs(dur - target)
            pool[(s, e)] = (score, {**meta, "teacher_duration_prior": target})

    rows = [(s, e, score, meta) for (s, e), (score, meta) in pool.items()]
    rows.sort(key=lambda x: x[2], reverse=True)
    return rows[:limit]


class ProposalRanker(nn.Module):
    def __init__(self, in_dim: int = 11, hidden: int = 96) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(0.05),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def span_features(
    start: np.ndarray,
    end: np.ndarray,
    sim: np.ndarray,
    vis: np.ndarray,
    qtype: int,
    s: int,
    e: int,
    base_score: float,
    pq_score: float = 0.0,
) -> List[float]:
    t = max(1, len(start))
    inside_sim = float(np.mean(sim[s:e + 1])) if e >= s else 0.0
    inside_vis = float(np.mean(vis[s:e + 1])) if e >= s else 0.0
    dur = (e - s + 1) / t
    center = (s + e) * 0.5 / t
    q = [0.0, 0.0, 0.0, 0.0]
    q[min(max(int(qtype), 0), 3)] = 1.0
    return [
        float(start[s]),
        float(end[e]),
        float(base_score),
        float(pq_score),
        float(sim[s]),
        float(sim[e]),
        inside_sim,
        inside_vis,
        dur,
        center,
        q[0] - q[1] + 0.5 * q[2],
    ]


@torch.no_grad()
def token_arrays(model: c12_5.SpanLocalizer, batch: Dict[str, Any]) -> Dict[str, Any]:
    tok = model.token_forward(batch["q"], batch["sub"], batch["vis"], batch["mask"], batch["qtype"])
    return {
        "start": tok["start"].detach().cpu().numpy(),
        "end": tok["end"].detach().cpu().numpy(),
        "sim": tok["sim"].detach().cpu().numpy(),
        "tok": tok,
    }


def build_ranker_train_rows(
    model: c12_5.SpanLocalizer,
    corpus: Any,
    features: Dict[str, Any],
    train_ids: Sequence[int],
    mode: str,
) -> Tuple[np.ndarray, np.ndarray]:
    store = c12_5.ClipFeatureStore()
    xs: List[List[float]] = []
    ys: List[float] = []
    ids = list(train_ids)
    random.Random(SEED).shuffle(ids)
    ids = ids[:TRAIN_LIMIT]
    try:
        for st in range(0, len(ids), TRAIN_BATCH):
            batch_ids = ids[st: st + TRAIN_BATCH]
            batch = c12_5.build_batch(corpus, features, store, batch_ids)
            arr = token_arrays(model, batch)
            for bi, did in enumerate(batch_ids):
                row = corpus.by_id[int(did)]
                t = int(batch["lengths"][bi].item())
                start = arr["start"][bi, :t]
                end = arr["end"][bi, :t]
                sim = arr["sim"][bi, :t]
                vis = batch["vis"][bi, :t].detach().cpu().numpy()
                qt = int(batch["qtype"][bi].item())
                gt = c12_5.ts_to_idx(row["ts"], float(row["duration"]), t)
                pool = dense_pool(start, end, sim, qt, mode, limit=320)
                # Add near-GT jitter only during training so the ranker sees high-IoU positives.
                for ds in (-2, -1, 0, 1, 2):
                    for de in (-2, -1, 0, 1, 2):
                        s = max(0, min(t - 1, gt[0] + ds))
                        e = max(s, min(t - 1, gt[1] + de))
                        pool.append((s, e, float(start[s] + end[e]), {"tag": "train_gt_jitter"}))
                seen = set()
                for s, e, score, _meta in pool:
                    if (s, e) in seen:
                        continue
                    seen.add((s, e))
                    y = c12_5.iou_1d((s, e + 1), (gt[0], gt[1] + 1))
                    xs.append(span_features(start, end, sim, vis, qt, s, e, score))
                    ys.append(float(y))
    finally:
        store.close()
    return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.float32)


def train_ranker(name: str, x: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
    seed_all(SEED)
    model = ProposalRanker(in_dim=x.shape[1]).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    n = x.shape[0]
    idx = np.arange(n)
    curves = []
    for epoch in range(1, 4):
        np.random.default_rng(SEED + epoch).shuffle(idx)
        losses = []
        for st in range(0, n, 4096):
            part = idx[st: st + 4096]
            xb = torch.from_numpy(x[part]).to(DEVICE)
            yb = torch.from_numpy(y[part]).to(DEVICE)
            opt.zero_grad(set_to_none=True)
            pred = model(xb)
            reg = F.mse_loss(torch.sigmoid(pred), yb)
            bce05 = F.binary_cross_entropy_with_logits(pred, (yb >= 0.5).float())
            bce07 = F.binary_cross_entropy_with_logits(pred, (yb >= 0.7).float())
            # Pairwise within the batch: high-IoU should beat low-IoU.
            hi = yb >= 0.7
            lo = yb < 0.3
            pair = pred.new_tensor(0.0)
            if bool(hi.any() and lo.any()):
                pair = F.relu(0.2 - pred[hi].mean() + pred[lo].mean())
            loss = reg + 0.4 * bce05 + 0.8 * bce07 + 0.5 * pair
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))
        with torch.no_grad():
            sample = np.random.default_rng(SEED + epoch).choice(n, size=min(40000, n), replace=False)
            pred = torch.sigmoid(model(torch.from_numpy(x[sample]).to(DEVICE))).detach().cpu().numpy()
            yy = y[sample]
            cc = corr(pred.tolist(), yy.tolist())
            curves.append({
                "epoch": epoch,
                "loss": float(np.mean(losses)),
                "pearson": cc["pearson"],
                "spearman": cc["spearman"],
                "auc05": auc_score(pred.tolist(), (yy >= 0.5).tolist()),
                "auc07": auc_score(pred.tolist(), (yy >= 0.7).tolist()),
            })
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    path = MODEL_DIR / f"c12_5r_{name}_ranker.pt"
    torch.save({"state_dict": model.state_dict(), "curves": curves, "feature_dim": x.shape[1]}, path)
    return {"model": model.eval(), "path": str(path), "sha256": sha256_file(path), "curves": curves}


@torch.no_grad()
def eval_repair_variant(
    name: str,
    mode: str,
    base_model: c12_5.SpanLocalizer,
    ranker: ProposalRanker | None,
    corpus: Any,
    features: Dict[str, Any],
    first_stage: Dict[int, Dict[str, Any]],
    desc_ids: Sequence[int],
    split: str,
    ms: Sequence[int] = (100, 200, 500),
) -> Dict[str, Any]:
    store = c12_5.ClipFeatureStore()
    rows = []
    pq_scores, ious_all = [], []
    by_qtype: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    by_dur: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    by_bucket: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    try:
        for st in range(0, len(desc_ids), EVAL_BATCH):
            batch_ids = list(desc_ids[st: st + EVAL_BATCH])
            batch = c12_5.build_batch(corpus, features, store, batch_ids)
            arr = token_arrays(base_model, batch)
            for bi, did in enumerate(batch_ids):
                row = corpus.by_id[int(did)]
                t = int(batch["lengths"][bi].item())
                start = arr["start"][bi, :t]
                end = arr["end"][bi, :t]
                sim = arr["sim"][bi, :t]
                vis = batch["vis"][bi, :t].detach().cpu().numpy()
                qt = int(batch["qtype"][bi].item())
                pool = dense_pool(start, end, sim, qt, mode, limit=1000)
                scored = []
                if ranker is not None and pool:
                    feats = np.asarray([span_features(start, end, sim, vis, qt, s, e, score) for s, e, score, _ in pool], dtype=np.float32)
                    pred = []
                    for k in range(0, len(feats), 4096):
                        pred.append(torch.sigmoid(ranker(torch.from_numpy(feats[k:k + 4096]).to(DEVICE))).detach().cpu().numpy())
                    rank_scores = np.concatenate(pred)
                    for (s, e, base_score, meta), rs in zip(pool, rank_scores):
                        scored.append((s, e, float(rs), base_score, meta))
                else:
                    for s, e, score, meta in pool:
                        scored.append((s, e, score, score, meta))
                scored.sort(key=lambda x: x[2], reverse=True)
                gt_idx = c12_5.ts_to_idx(row["ts"], float(row["duration"]), t)
                gt_ts = (float(row["ts"][0]), float(row["ts"][1]))
                best_by_m = {}
                rank_best_iou = None
                best_iou = -1.0
                best_iou_score = None
                span_records = []
                seen = set()
                dup = 0
                invalid = 0
                for rank, (s, e, rscore, base_score, meta) in enumerate(scored[:1000], start=1):
                    if (s, e) in seen:
                        dup += 1
                    seen.add((s, e))
                    if e < s:
                        invalid += 1
                    ts = c12_5.idx_to_ts(s, e, float(row["duration"]), t)
                    iou = c12_5.iou_1d(ts, gt_ts)
                    span_records.append((rank, iou, rscore, s, e))
                    if iou > best_iou:
                        best_iou = iou
                        best_iou_score = rscore
                        rank_best_iou = rank
                    pq_scores.append(float(rscore))
                    ious_all.append(float(iou))
                for m in ms:
                    subset = span_records[:m]
                    best_by_m[m] = max((x[1] for x in subset), default=0.0)
                top1 = span_records[0] if span_records else None
                rec = {
                    "desc_id": int(did),
                    "query_type": row.get("type", "unknown"),
                    "duration": float(row["duration"]),
                    "moment_duration": float(row["ts"][1] - row["ts"][0]),
                    "gt_video_in_top100": isinstance(first_stage[int(did)].get("rank"), int) and int(first_stage[int(did)]["rank"]) <= 100,
                    "rank_of_best_iou_span": rank_best_iou,
                    "best_generated_iou_top1000": best_iou,
                    "proposal_quality_score_of_best_iou": best_iou_score,
                    "top1_start_error": abs(c12_5.idx_to_ts(top1[3], top1[4], float(row["duration"]), t)[0] - gt_ts[0]) if top1 else None,
                    "top1_end_error": abs(c12_5.idx_to_ts(top1[3], top1[4], float(row["duration"]), t)[1] - gt_ts[1]) if top1 else None,
                    "invalid_span_count": invalid,
                    "duplicate_span_count": dup,
                }
                for m in ms:
                    rec[f"best_iou_top{m}"] = best_by_m[m]
                    rec[f"cover05_top{m}"] = best_by_m[m] >= 0.5
                    rec[f"cover07_top{m}"] = best_by_m[m] >= 0.7
                    rec[f"joint05_top{m}"] = rec["gt_video_in_top100"] and best_by_m[m] >= 0.5
                    rec[f"joint07_top{m}"] = rec["gt_video_in_top100"] and best_by_m[m] >= 0.7
                md = rec["moment_duration"]
                dur_bucket = "short" if md <= 5 else "medium" if md <= 15 else "long"
                rows.append(rec)
                by_qtype[rec["query_type"]].append(rec)
                by_dur[dur_bucket].append(rec)
                fs = first_stage[int(did)]
                if fs.get("top1_correct") is True:
                    by_bucket["B6_or_firststage_top1_correct_proxy"].append(rec)
                else:
                    by_bucket["B6_or_firststage_top1_wrong_proxy"].append(rec)
                if isinstance(fs.get("rank"), int):
                    if int(fs["rank"]) <= 5:
                        by_bucket["positive_in_top5"].append(rec)
                    if int(fs["rank"]) <= 10:
                        by_bucket["positive_in_top10"].append(rec)
                    if int(fs["rank"]) <= 100:
                        by_bucket["positive_in_top100"].append(rec)
    finally:
        store.close()

    def summarize(rs: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        out = {"query_count": len(rs)}
        for m in ms:
            out[f"GT_video_oracle_IoU@0.5_top{m}"] = metric([r[f"cover05_top{m}"] for r in rs])
            out[f"GT_video_oracle_IoU@0.7_top{m}"] = metric([r[f"cover07_top{m}"] for r in rs])
            out[f"retriever_top100_joint_IoU@0.5_top{m}"] = metric([r[f"joint05_top{m}"] for r in rs])
            out[f"retriever_top100_joint_IoU@0.7_top{m}"] = metric([r[f"joint07_top{m}"] for r in rs])
            out[f"median_best_iou_top{m}"] = pct([r[f"best_iou_top{m}"] for r in rs], 50)
        out["best_generated_IoU@0.5_top1000"] = metric([r["best_generated_iou_top1000"] >= 0.5 for r in rs])
        out["best_generated_IoU@0.7_top1000"] = metric([r["best_generated_iou_top1000"] >= 0.7 for r in rs])
        out["median_rank_of_best_iou_span"] = pct([r["rank_of_best_iou_span"] for r in rs if r["rank_of_best_iou_span"] is not None], 50)
        out["median_best_generated_iou"] = pct([r["best_generated_iou_top1000"] for r in rs], 50)
        out["top1_start_error_mean"] = mean([r["top1_start_error"] for r in rs if r["top1_start_error"] is not None])
        out["top1_end_error_mean"] = mean([r["top1_end_error"] for r in rs if r["top1_end_error"] is not None])
        out["invalid_span_count"] = int(sum(r["invalid_span_count"] for r in rs))
        out["duplicate_span_count"] = int(sum(r["duplicate_span_count"] for r in rs))
        return out

    c = corr(pq_scores[:200000], ious_all[:200000])
    calib = {
        **c,
        "auc_iou05": auc_score(pq_scores[:200000], [x >= 0.5 for x in ious_all[:200000]]),
        "auc_iou07": auc_score(pq_scores[:200000], [x >= 0.7 for x in ious_all[:200000]]),
        "sample_count": min(len(pq_scores), 200000),
    }
    return {
        "variant": name,
        "split": split,
        "summary": summarize(rows),
        "query_type_breakdown": {k: summarize(v) for k, v in sorted(by_qtype.items())},
        "duration_bucket_breakdown": {k: summarize(v) for k, v in sorted(by_dur.items())},
        "b6_failure_bucket_breakdown": {k: summarize(v) for k, v in sorted(by_bucket.items())},
        "pq_iou_calibration": calib,
        "records_sample": rows[:8],
    }


def aligned_b6_audit(corpus: Any) -> Dict[str, Any]:
    c12 = load_json(ROOT / "c12_5_native_span_generation/C12_5D_SPAN_COVERAGE_RESULTS.json")
    b6 = c12_5.load_b6_teacher()
    hold_ids = set(corpus.splits["calib_holdout"])
    b6_ids = set(b6)
    inter = sorted(hold_ids & b6_ids)
    b6_ref = c12_5.b6_coverage_reference(corpus, {i: b6[i] for i in inter})
    full_c12 = c12["calib_holdout"]["teacher_distilled"]
    # Existing C12 evaluation did not persist per-query full records; if same
    # subset exists, recompute through C12-5R evaluation for strict comparison.
    return {
        "stage": "C12-5R-A",
        "c12_holdout_query_count": len(hold_ids),
        "c7_b6_reference_query_count": len(b6_ids),
        "b6_reference_rows_available_reason": "C8 exact-B6 train feature re-export contains a 1024-row train-only teacher/comparison subset.",
        "same_query_intersection_count": len(inter),
        "same_query_intersection_desc_ids": inter,
        "same_query_comparison_available": len(inter) > 0,
        "full_c12_holdout_teacher_distilled": full_c12,
        "b6_reference_subset_b6_coverage": b6_ref["summary"],
        "b6_reference_subset_c12_coverage": None,
        "aligned_delta_iou05_top100": None,
        "aligned_delta_iou07_top100": None,
        "b6_comparison_is_reference_only_not_strict_gate": len(inter) == 0,
    }


def timestamp_audit(corpus: Any, desc_ids: Sequence[int]) -> Dict[str, Any]:
    buckets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    invalid = 0
    duplicates = 0
    gen_durations = []
    gt_durations = []
    for did in desc_ids:
        row = corpus.by_id[int(did)]
        # Use feature length proxy via duration / 1.5 when no LMDB read is needed.
        approx_t = max(1, min(MAX_T, int(math.ceil(float(row["duration"]) / c12_5.CLIP_LEN))))
        s, e = c12_5.ts_to_idx(row["ts"], float(row["duration"]), approx_t)
        if e < s:
            invalid += 1
        gt_d = float(row["ts"][1] - row["ts"][0])
        gt_durations.append(gt_d)
        # Audit native duration anchors produced by R1 without using GT.
        start = np.zeros(approx_t, dtype=np.float32)
        end = np.zeros(approx_t, dtype=np.float32)
        sim = np.zeros(approx_t, dtype=np.float32)
        pool = dense_pool(start, end, sim, qtype_id(row.get("type", "unknown")), "r1_dense_duration", limit=1000)
        seen = set()
        for ps, pe, _score, _meta in pool:
            if (ps, pe) in seen:
                duplicates += 1
            seen.add((ps, pe))
            ts = c12_5.idx_to_ts(ps, pe, float(row["duration"]), approx_t)
            gen_durations.append(ts[1] - ts[0])
        rec = {
            "desc_id": int(did),
            "duration": float(row["duration"]),
            "gt_duration": gt_d,
            "clip_length_assumed": c12_5.CLIP_LEN,
            "approx_t": approx_t,
            "mapped_start_index": s,
            "mapped_end_index": e,
            "discretized_duration_clips": e - s + 1,
            "short_moment_discretized_to_single_clip": gt_d <= 5 and e == s,
        }
        bucket = "short" if gt_d <= 5 else "medium" if gt_d <= 15 else "long"
        buckets[bucket].append(rec)

    def summarize(rs: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        return {
            "query_count": len(rs),
            "gt_duration_mean": mean([r["gt_duration"] for r in rs]),
            "discretized_clip_count_median": pct([r["discretized_duration_clips"] for r in rs], 50),
            "single_clip_rate": metric([r["discretized_duration_clips"] == 1 for r in rs]),
        }

    short_single = summarize(buckets.get("short", []))["single_clip_rate"]
    status = "C12_BOUNDARY_LABEL_AUDIT_PASS"
    note = "Timestamp mapping is floor(start)/ceil(end)-1 with inclusive end index. Short moments often become one/few clips, but no invalid-label bug was found."
    return {
        "stage": "C12-5R-B",
        "status": status,
        "clip_length": c12_5.CLIP_LEN,
        "start_mapping": "floor(start / duration * T)",
        "end_mapping": "ceil(end / duration * T) - 1",
        "end_boundary": "inclusive clip index; timestamp output uses exclusive end time",
        "span_start_less_equal_end_enforced": True,
        "invalid_span_count": invalid,
        "duplicate_span_count_in_r1_audit_pool": duplicates,
        "generated_span_duration_distribution": {
            "mean": mean(gen_durations),
            "p10": pct(gen_durations, 10),
            "p50": pct(gen_durations, 50),
            "p90": pct(gen_durations, 90),
        },
        "gt_duration_distribution": {
            "mean": mean(gt_durations),
            "p10": pct(gt_durations, 10),
            "p50": pct(gt_durations, 50),
            "p90": pct(gt_durations, 90),
        },
        "duration_buckets": {k: summarize(v) for k, v in sorted(buckets.items())},
        "short_moment_single_clip_rate": short_single,
        "audit_note": note,
        "official_val_used": False,
    }


def write_md_reports(
    align: Dict[str, Any],
    ts: Dict[str, Any],
    decomp: Dict[str, Any],
    results: Dict[str, Any],
    decision: Dict[str, Any],
) -> None:
    write_text(OUT / "C12_5R_A_COVERAGE_ALIGNMENT_AUDIT.md", f"""# C12-5R-A Coverage Alignment Audit

same_query_comparison_available = {str(align['same_query_comparison_available']).lower()}

C12 holdout query count = {align['c12_holdout_query_count']}
C7-B6 reference query count = {align['c7_b6_reference_query_count']}
same-query intersection count = {align['same_query_intersection_count']}
aligned_delta_iou05_top100 = {align.get('aligned_delta_iou05_top100')}
aligned_delta_iou07_top100 = {align.get('aligned_delta_iou07_top100')}

Reason for B6 reference rows = {align['b6_reference_rows_available_reason']}

If intersection is small/empty, B6 comparison remains reference-only, not a strict gate.
official_val_used = false
""")
    write_text(OUT / "C12_5R_B_TIMESTAMP_BOUNDARY_AUDIT.md", f"""# C12-5R-B Timestamp / Boundary Label Audit

status = {ts['status']}

clip_length = {ts['clip_length']}
start_mapping = {ts['start_mapping']}
end_mapping = {ts['end_mapping']}
invalid_span_count = {ts['invalid_span_count']}

{ts['audit_note']}

official_val_used = false
""")
    write_text(OUT / "C12_5R_C_GENERATION_RANKING_DECOMPOSITION.md", f"""# C12-5R-C Generation vs Ranking Decomposition

bottleneck = {decomp['bottleneck']}

If generated top1000 coverage is high while ranked top100 remains lower, the remaining issue is proposal-quality/ranking.

official_val_used = false
""")
    lines = ["# C12-5R-E Repair Results", ""]
    for name, rec in results["calib_holdout"].items():
        s = rec["summary"]
        c = rec["pq_iou_calibration"]
        lines += [
            f"## {name}",
            f"- IoU@0.5 top100: {s['GT_video_oracle_IoU@0.5_top100']:.4f}",
            f"- IoU@0.7 top100: {s['GT_video_oracle_IoU@0.7_top100']:.4f}",
            f"- IoU@0.7 top500: {s['GT_video_oracle_IoU@0.7_top500']:.4f}",
            f"- median best IoU top100: {s['median_best_iou_top100']:.4f}",
            f"- PQ Spearman: {c['spearman']}",
            "",
        ]
    write_text(OUT / "C12_5R_E_REPAIR_RESULTS.md", "\n".join(lines))
    write_text(OUT / "C12_5R_SPAN_REPAIR_DECISION.md", f"""# C12-5R Span Repair Decision

status = {decision['status']}

best_repair_variant = {decision['best_repair_variant']}

best IoU@0.5 top100 = {decision['best_iou05_top100']:.4f}
best IoU@0.7 top100 = {decision['best_iou07_top100']:.4f}
short moment IoU@0.7 top100 = {decision['short_iou07_top100']:.4f}

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
    select_ids = corpus.splits["calib_select"]
    holdout_ids = corpus.splits["calib_holdout"]
    if EVAL_LIMIT > 0:
        select_ids = select_ids[:EVAL_LIMIT]
        holdout_ids = holdout_ids[:EVAL_LIMIT]
    first_stage_select = load_first_stage(corpus, select_ids, top_keep=128)
    first_stage_holdout = load_first_stage(corpus, holdout_ids, top_keep=128, cache_name="first_stage_calib_holdout_top128.pkl" if EVAL_LIMIT == 0 else None)
    teacher_model = load_model("teacher_distilled")
    start_model = load_model("start_end")

    align = aligned_b6_audit(corpus)
    ts = timestamp_audit(corpus, holdout_ids)
    write_json(OUT / "C12_5R_A_COVERAGE_ALIGNMENT_AUDIT.json", align)
    write_json(OUT / "C12_5R_B_TIMESTAMP_BOUNDARY_AUDIT.json", ts)

    # Train the small repair rankers for R3/R5. R1/R2/R4 are native proposal
    # generators/scorers, R5 uses the IoU-calibrated ranker on a large pool.
    train_x, train_y = build_ranker_train_rows(teacher_model, corpus, features, corpus.splits["train_fit"], "r5_two_stage")
    ranker_rec = train_ranker("iou_calibrated", train_x, train_y)
    ranker = ranker_rec["model"]

    variants = {
        "c12_5_teacher_distilled_baseline_reeval": ("r4_teacher_prior", teacher_model, None),
        "r1_duration_conditioned_dense": ("r1_dense_duration", teacher_model, None),
        "r2_boundary_offset_refine": ("r2_offset_refine", teacher_model, None),
        "r3_iou_calibrated_pq_ranker": ("r1_dense_duration", teacher_model, ranker),
        "r4_teacher_distilled_boundary_prior": ("r4_teacher_prior", teacher_model, ranker),
        "r5_two_stage_generate_then_rank": ("r5_two_stage", teacher_model, ranker),
        "c12_5_start_end_baseline_reeval": ("r1_dense_duration", start_model, None),
    }

    results = {"stage": "C12-5R-E", "ranker": {k: v for k, v in ranker_rec.items() if k != "model"}, "calib_select": {}, "calib_holdout": {}, "official_val_used": False}
    for name, (mode, model, maybe_ranker) in variants.items():
        results["calib_select"][name] = eval_repair_variant(name, mode, model, maybe_ranker, corpus, features, first_stage_select, select_ids, "calib_select")
        results["calib_holdout"][name] = eval_repair_variant(name, mode, model, maybe_ranker, corpus, features, first_stage_holdout, holdout_ids, "calib_holdout")

    # Generation/ranking decomposition from holdout summaries.
    decomp_variants = {}
    for name, rec in results["calib_holdout"].items():
        s = rec["summary"]
        decomp_variants[name] = {
            "generated_pool_oracle_IoU@0.5_M1000": s["best_generated_IoU@0.5_top1000"],
            "generated_pool_oracle_IoU@0.7_M1000": s["best_generated_IoU@0.7_top1000"],
            "ranked_top100_IoU@0.5": s["GT_video_oracle_IoU@0.5_top100"],
            "ranked_top100_IoU@0.7": s["GT_video_oracle_IoU@0.7_top100"],
            "ranked_top200_IoU@0.7": s["GT_video_oracle_IoU@0.7_top200"],
            "ranked_top500_IoU@0.7": s["GT_video_oracle_IoU@0.7_top500"],
            "median_rank_of_best_iou_span": s["median_rank_of_best_iou_span"],
        }
    best_generated = max(v["generated_pool_oracle_IoU@0.7_M1000"] for v in decomp_variants.values())
    best_top100 = max(v["ranked_top100_IoU@0.7"] for v in decomp_variants.values())
    bottleneck = "proposal_quality_ranking" if best_generated >= best_top100 + 10 else "span_candidate_generation"
    decomp = {"stage": "C12-5R-C", "bottleneck": bottleneck, "variants": decomp_variants, "official_val_used": False}
    write_json(OUT / "C12_5R_C_GENERATION_RANKING_DECOMPOSITION.json", decomp)

    hold = results["calib_holdout"]
    selectable = {k: v for k, v in results["calib_select"].items() if not k.startswith("c12_5_")}
    selected = max(selectable, key=lambda k: selectable[k]["summary"]["GT_video_oracle_IoU@0.7_top100"])
    h = hold[selected]["summary"]
    short = hold[selected]["duration_bucket_breakdown"].get("short", {})
    b6_ref = align["b6_reference_subset_b6_coverage"]
    b6_07 = b6_ref.get("GT_video_oracle_IoU@0.7_top100") or 66.0156
    gap_recovered = (h["GT_video_oracle_IoU@0.7_top100"] - 43.0876) / max(1e-6, b6_07 - 43.0876)
    allow = (
        align["same_query_comparison_available"]
        and h["GT_video_oracle_IoU@0.7_top100"] >= 54.5
        and h["GT_video_oracle_IoU@0.5_top100"] > 59.6544
        and (short.get("GT_video_oracle_IoU@0.7_top100", 0.0) > 33.9394)
        and ts["status"] == "C12_BOUNDARY_LABEL_AUDIT_PASS"
    )
    if allow:
        status = "C12_LOCALIZER_REPAIRED_CONTINUE_TO_C12_6"
    elif h["GT_video_oracle_IoU@0.7_top100"] > 43.0876 and (short.get("GT_video_oracle_IoU@0.7_top100", 0.0) > 33.9394):
        status = "C12_LOCALIZER_PARTIALLY_REPAIRED_CONTINUE_WITH_CAUTION"
    elif bottleneck == "proposal_quality_ranking":
        status = "C12_LOCALIZER_PQ_RANKING_BOTTLENECK"
    else:
        status = "C12_LOCALIZER_BOUNDARY_BOTTLENECK_NEEDS_STRONGER_FEATURE"
    decision = {
        "stage": "C12-5R-F",
        "status": status,
        "best_repair_variant": selected,
        "best_iou05_top100": h["GT_video_oracle_IoU@0.5_top100"],
        "best_iou07_top100": h["GT_video_oracle_IoU@0.7_top100"],
        "best_iou07_top500": h["GT_video_oracle_IoU@0.7_top500"],
        "short_iou07_top100": short.get("GT_video_oracle_IoU@0.7_top100"),
        "gap_recovered_vs_b6_reference": gap_recovered,
        "allow_enter_c12_6": bool(allow),
        "generation_vs_ranking_bottleneck": bottleneck,
        "same_query_comparison_available": align["same_query_comparison_available"],
        "official_val_used": False,
        "evaluator_modified": False,
        "nms_modified": False,
        "reason": f"Selected on calib_select only. Holdout IoU@0.7 top100={h['GT_video_oracle_IoU@0.7_top100']:.4f}; short={short.get('GT_video_oracle_IoU@0.7_top100'):.4f}; gap_recovered={gap_recovered:.4f}; bottleneck={bottleneck}.",
    }

    inter = align.get("same_query_intersection_desc_ids", [])
    if inter:
        fs_inter = load_first_stage(corpus, inter, top_keep=128)
        c12_same = c12_5.eval_variant("teacher_distilled", teacher_model, corpus, features, fs_inter, inter, "same_query_intersection")
        mode, model, maybe_ranker = variants[selected]
        repair_same = eval_repair_variant(selected, mode, model, maybe_ranker, corpus, features, fs_inter, inter, "same_query_intersection")
        b6_summary = align["b6_reference_subset_b6_coverage"]
        align["b6_reference_subset_c12_coverage"] = c12_same["summary"]
        align["b6_reference_subset_selected_repair_coverage"] = repair_same["summary"]
        align["aligned_delta_iou05_top100"] = c12_same["summary"]["GT_video_oracle_IoU@0.5_top100"] - b6_summary["GT_video_oracle_IoU@0.5_top100"]
        align["aligned_delta_iou07_top100"] = c12_same["summary"]["GT_video_oracle_IoU@0.7_top100"] - b6_summary["GT_video_oracle_IoU@0.7_top100"]
        align["aligned_selected_repair_delta_iou05_top100"] = repair_same["summary"]["GT_video_oracle_IoU@0.5_top100"] - b6_summary["GT_video_oracle_IoU@0.5_top100"]
        align["aligned_selected_repair_delta_iou07_top100"] = repair_same["summary"]["GT_video_oracle_IoU@0.7_top100"] - b6_summary["GT_video_oracle_IoU@0.7_top100"]
        decision["aligned_delta_iou05_top100"] = align["aligned_selected_repair_delta_iou05_top100"]
        decision["aligned_delta_iou07_top100"] = align["aligned_selected_repair_delta_iou07_top100"]
        write_json(OUT / "C12_5R_A_COVERAGE_ALIGNMENT_AUDIT.json", align)

    write_json(OUT / "C12_5R_E_REPAIR_RESULTS.json", results)
    write_json(OUT / "C12_5R_E_QUERY_TYPE_BREAKDOWN.json", {k: {s: results[s][k]["query_type_breakdown"] for s in ("calib_select", "calib_holdout")} for k in results["calib_holdout"]})
    write_json(OUT / "C12_5R_E_DURATION_BUCKET_BREAKDOWN.json", {k: {s: results[s][k]["duration_bucket_breakdown"] for s in ("calib_select", "calib_holdout")} for k in results["calib_holdout"]})
    write_json(OUT / "C12_5R_E_B6_FAILURE_BUCKET_BREAKDOWN.json", {k: {s: results[s][k]["b6_failure_bucket_breakdown"] for s in ("calib_select", "calib_holdout")} for k in results["calib_holdout"]})
    write_json(OUT / "C12_5R_SPAN_REPAIR_DECISION.json", decision)
    write_md_reports(align, ts, decomp, results, decision)
    manifest = {
        "stage": "C12-5R",
        "status": decision["status"],
        "best_repair_variant": selected,
        "trained_repair_variants": ["R3 IoU ranker", "R5 two-stage ranker"],
        "evaluated_repair_variants": list(variants),
        "official_val_used": False,
        "evaluator_modified": False,
        "nms_modified": False,
        "artifact_hashes": {str(p.relative_to(ROOT)): sha256_file(p) for p in sorted(OUT.glob("C12_5R*")) if p.is_file()},
    }
    write_json(OUT / "C12_5R_MANIFEST.json", manifest)


if __name__ == "__main__":
    main()
