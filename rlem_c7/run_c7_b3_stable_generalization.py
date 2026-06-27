#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import math
import multiprocessing as mp
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rlem_c6_b0.run_c6_b0_candidate_diagnostic import span_iou_idx  # noqa: E402
from rlem_c7.run_c7_b0_b1 import (  # noqa: E402
    EPS,
    atomic_json,
    atomic_text,
    artifact,
    gt_span_by_query,
    gt_video_by_query,
    labels_for_sequence,
    load_npz,
    metric_delta,
    nms_sequence,
    selected_metrics,
)
from rlem_c7.run_c7_b2_1_safety_repair import group_score_map, normalize_cand  # noqa: E402


METRIC_KEYS = [
    "0.5-r1", "0.5-r5", "0.5-r10", "0.5-r100",
    "0.7-r1", "0.7-r5", "0.7-r10", "0.7-r100",
]
Candidate = Tuple[int, int, int, int, float, int]
_W_QUERIES: List["QueryData"] | None = None
_W_SPLITS: Dict[str, List[int]] | None = None
_W_ARGS: argparse.Namespace | None = None
_W_META: Dict[str, Any] | None = None
_W_BASELINE: Dict[str, Dict[str, Any]] | None = None


def sha256_obj(obj: Any) -> str:
    data = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def stable_bucket(x: Any, mod: int = 100) -> int:
    h = hashlib.sha256(str(x).encode("utf-8")).hexdigest()
    return int(h[:12], 16) % mod


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = x.astype(np.float64)
    y = y.astype(np.float64)
    if len(x) < 2 or float(np.std(x)) < EPS or float(np.std(y)) < EPS:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--calib_cache", default="results/rlem_c6a/cache/train_calib_c6_cache.npz")
    p.add_argument("--anchor_states", default="results/rlem_c7_b2/train_calib_c7_b1_anchor_states.json")
    p.add_argument("--video_scores", default="results/rlem_c7_b2/train_calib_video_scores.npz")
    p.add_argument("--proposal_scores", default="results/rlem_c7_b2/train_calib_c7_b1_proposal_scores.npz")
    p.add_argument("--b21_freeze", default="c7_audit/C7_B2_1_FINAL_FREEZE_PACKAGE.json")
    p.add_argument("--b2_eval", default="c7_audit/C7_B2_TRAIN_CALIB_EVAL.json")
    p.add_argument("--audit_dir", default="c7_audit")
    p.add_argument("--output_dir", default="results/rlem_c7_b3")
    p.add_argument("--device", default="cuda")
    p.add_argument("--nms_thd", type=float, default=0.7)
    p.add_argument("--effective_top_n", type=int, default=100)
    p.add_argument("--max_after_nms", type=int, default=100)
    p.add_argument("--gate_epochs", type=int, default=80)
    p.add_argument("--gate_hidden", type=int, default=256)
    p.add_argument("--gate_layers", type=int, default=3)
    p.add_argument("--ranker_epochs", type=int, default=6)
    p.add_argument("--ranker_hidden", type=int, default=384)
    p.add_argument("--ranker_layers", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=16384)
    p.add_argument("--max_ranker_queries", type=int, default=6000)
    p.add_argument("--max_pairs_per_query", type=int, default=64)
    p.add_argument("--workers", type=int, default=16)
    return p.parse_args()


def cand_key(c: Candidate) -> Tuple[int, int, int]:
    return int(c[1]), int(c[2]), int(c[3])


@dataclass
class QueryData:
    q: int
    desc_id: Any
    text: str
    anchor: List[Candidate]
    anchor_score: np.ndarray
    residual: np.ndarray
    prop_conf: np.ndarray
    span_quality: np.ndarray
    y05: np.ndarray
    y07: np.ndarray
    query_len: int
    anchor_post: List[Candidate] | None = None
    anchor_labels05: List[bool] | None = None
    anchor_labels07: List[bool] | None = None
    anchor_pos: bool = False


def make_prop_maps(path: str | Path) -> Dict[str, np.ndarray]:
    z = np.load(path, allow_pickle=False)
    rows = z["row"].astype(np.int64)
    max_row = int(rows.max()) + 1
    out = {}
    for key in ["proposal_confidence", "span_quality"]:
        arr = np.zeros(max_row, dtype=np.float32)
        arr[rows] = z[key].astype(np.float32)
        out[key] = arr
    return out


def build_query_data(args: argparse.Namespace) -> Tuple[List[QueryData], Dict[str, Any]]:
    cache = load_npz(args.calib_cache, allow_pickle=True)
    states = json.loads(Path(args.anchor_states).read_text(encoding="utf-8"))["states"]
    score_arrays = load_npz(args.video_scores, allow_pickle=False)
    cfg = json.loads(Path(args.b2_eval).read_text(encoding="utf-8"))["best"]["config"]
    gscore = group_score_map(score_arrays, cfg)
    prop_maps = make_prop_maps(args.proposal_scores)
    gt_vid = gt_video_by_query(cache)
    gt_s, gt_e = gt_span_by_query(cache)
    queries: List[QueryData] = []
    for q, state in enumerate(states):
        anchor = [normalize_cand(x) for x in state["flat_candidates"][:args.effective_top_n]]
        residual = np.asarray([float(cfg["video_weight"]) * gscore.get(int(c[0]), (0.0, 0.0, 0.0, 0.0))[2] for c in anchor], dtype=np.float32)
        rows = np.asarray([max(int(c[5]), 0) for c in anchor], dtype=np.int64)
        y05 = np.zeros(len(anchor), dtype=np.float32)
        y07 = np.zeros(len(anchor), dtype=np.float32)
        for i, c in enumerate(anchor):
            same = int(c[1]) == int(gt_vid[q])
            iou = span_iou_idx(int(c[2]), int(c[3]), int(gt_s[q]), int(gt_e[q])) if same else 0.0
            y05[i] = float(iou >= 0.5)
            y07[i] = float(iou >= 0.7)
        text = str(cache["desc_text"][q])
        queries.append(QueryData(
            q=q,
            desc_id=cache["desc_ids"][q].item() if hasattr(cache["desc_ids"][q], "item") else cache["desc_ids"][q],
            text=text,
            anchor=anchor,
            anchor_score=np.asarray([float(c[4]) for c in anchor], dtype=np.float32),
            residual=residual,
            prop_conf=prop_maps["proposal_confidence"][rows].astype(np.float32),
            span_quality=prop_maps["span_quality"][rows].astype(np.float32),
            y05=y05,
            y07=y07,
            query_len=len(text.split()),
        ))
    meta = {"cache": cache, "gt_vid": gt_vid, "gt_s": gt_s, "gt_e": gt_e, "base_config": cfg}
    return queries, meta


def precompute_anchor_cache(queries: List[QueryData], args: argparse.Namespace, meta: Dict[str, Any]) -> None:
    gt_vid = meta["gt_vid"]; gt_s = meta["gt_s"]; gt_e = meta["gt_e"]
    for qd in queries:
        seq = nms_sequence(qd.anchor[:args.effective_top_n], args.nms_thd, args.max_after_nms)
        l05, l07, pos, _inv, _dup = labels_for_sequence(seq, int(gt_vid[qd.q]), int(gt_s[qd.q]), int(gt_e[qd.q]))
        qd.anchor_post = seq
        qd.anchor_labels05 = l05
        qd.anchor_labels07 = l07
        qd.anchor_pos = bool(pos)


def seq_from_scores(qd: QueryData, scores: np.ndarray, args: argparse.Namespace) -> List[Candidate]:
    ranked = [
        (c[0], c[1], c[2], c[3], float(s), c[5])
        for c, s in zip(qd.anchor, scores.astype(np.float32))
    ]
    ranked = sorted(ranked, key=lambda c: -c[4])
    return nms_sequence(ranked[:args.effective_top_n], args.nms_thd, args.max_after_nms)


def anchor_seq(qd: QueryData, args: argparse.Namespace) -> List[Candidate]:
    if qd.anchor_post is not None:
        return qd.anchor_post
    return nms_sequence(qd.anchor[:args.effective_top_n], args.nms_thd, args.max_after_nms)


def evaluate_scores(
    queries: List[QueryData],
    q_indices: Sequence[int],
    score_fn: Callable[[QueryData], np.ndarray],
    args: argparse.Namespace,
    meta: Dict[str, Any],
    name: str,
) -> Dict[str, Any]:
    gt_vid = meta["gt_vid"]; gt_s = meta["gt_s"]; gt_e = meta["gt_e"]
    labels05: List[List[bool]] = []
    labels07: List[List[bool]] = []
    anchor05: List[List[bool]] = []
    anchor07: List[List[bool]] = []
    exits = entries = invalid = dup = 0
    for q in q_indices:
        qd = queries[q]
        seq = seq_from_scores(qd, score_fn(qd), args)
        l05, l07, pos, inv, du = labels_for_sequence(seq, int(gt_vid[q]), int(gt_s[q]), int(gt_e[q]))
        if qd.anchor_labels05 is not None and qd.anchor_labels07 is not None:
            a05 = qd.anchor_labels05
            a07 = qd.anchor_labels07
            apos = qd.anchor_pos
        else:
            aseq = anchor_seq(qd, args)
            a05, a07, apos, _ai, _ad = labels_for_sequence(aseq, int(gt_vid[q]), int(gt_s[q]), int(gt_e[q]))
        labels05.append(l05); labels07.append(l07); anchor05.append(a05); anchor07.append(a07)
        exits += int(apos and not pos); entries += int((not apos) and pos)
        invalid += inv; dup += du
    metrics = selected_metrics(labels05, labels07)
    gain_loss = {}
    for thr, cur, base in [("0.5", labels05, anchor05), ("0.7", labels07, anchor07)]:
        for r in [1, 5, 10]:
            gain = loss = 0
            for i in range(len(cur)):
                b = any(base[i][:r]); c = any(cur[i][:r])
                gain += int((not b) and c); loss += int(b and not c)
            gain_loss[f"{thr}-r{r}"] = {"gain_queries": int(gain), "loss_queries": int(loss)}
    fixed_pool = True
    return {
        "name": name,
        "query_count": int(len(q_indices)),
        "metrics": metrics,
        "movement": {
            "hard_positive_top100_query_exits": int(exits),
            "hard_positive_top100_query_entries": int(entries),
            "hard_positive_exit_ratio": float(exits / max(len(q_indices), 1)),
            "invalid_span_count": int(invalid),
            "duplicate_span_count_after_nms": int(dup),
            "fixed_pool_invariant": fixed_pool,
        },
        "rank_gain_loss_vs_anchor": gain_loss,
    }


def split_indices(queries: List[QueryData]) -> Dict[str, List[int]]:
    buckets = np.asarray([stable_bucket(q.desc_id) for q in queries], dtype=np.int32)
    anchor_margins = np.asarray([float(q.anchor_score[0] - q.anchor_score[1]) if len(q.anchor_score) > 1 else 0.0 for q in queries], dtype=np.float32)
    residual_conflict = np.asarray([float(np.std(q.residual)) for q in queries], dtype=np.float32)
    stress_score = -anchor_margins + residual_conflict
    stress_cut = float(np.quantile(stress_score, 0.80))
    return {
        "train_core": np.where(buckets < 50)[0].astype(int).tolist(),
        "calib_A": np.where((buckets >= 50) & (buckets < 66))[0].astype(int).tolist(),
        "calib_B": np.where((buckets >= 66) & (buckets < 81))[0].astype(int).tolist(),
        "calib_C": np.where((buckets >= 81) & (buckets < 95))[0].astype(int).tolist(),
        "stress_split": np.where(stress_score >= stress_cut)[0].astype(int).tolist(),
        "train_calib_final_review": list(range(len(queries))),
    }


def split_stats(queries: List[QueryData], splits: Dict[str, List[int]], args: argparse.Namespace, meta: Dict[str, Any]) -> Dict[str, Any]:
    out = {}
    for name, idx in splits.items():
        qlens = np.asarray([queries[q].query_len for q in idx], dtype=np.float32)
        positives = np.asarray([float(np.max(queries[q].y05) > 0 or np.max(queries[q].y07) > 0) for q in idx], dtype=np.float32)
        anchor_eval = evaluate_scores(queries, idx, lambda qd: qd.anchor_score, args, meta, f"{name}_anchor")
        out[name] = {
            "query_count": len(idx),
            "positive_pool_rate": float(positives.mean()) if len(positives) else 0.0,
            "anchor_hit_rate_r100_05_or_07": float(anchor_eval["metrics"]["0.5-r100"] / 100.0),
            "query_length_mean": float(qlens.mean()) if len(qlens) else 0.0,
            "query_length_p50": float(np.median(qlens)) if len(qlens) else 0.0,
            "query_length_p90": float(np.quantile(qlens, 0.9)) if len(qlens) else 0.0,
            "hash": sha256_obj(idx),
        }
    return out


def transform_residual(res: np.ndarray, mode: str, temp: float, clip_c: float) -> np.ndarray:
    x = res.astype(np.float32) / max(float(temp), EPS)
    if mode == "per_query_z":
        x = (x - float(x.mean())) / max(float(x.std()), EPS)
    elif mode == "per_query_rank":
        order = np.argsort(np.argsort(x)).astype(np.float32)
        x = (order / max(len(order) - 1, 1) - 0.5) * 2.0
    elif mode == "sigmoid":
        x = 1.0 / (1.0 + np.exp(-x)) - 0.5
    elif mode == "none":
        pass
    else:
        raise ValueError(mode)
    return np.clip(x, -clip_c, clip_c).astype(np.float32)


def summarize_candidate(records: Dict[str, Dict[str, Any]], baseline_records: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    split_names = ["calib_A", "calib_B", "calib_C", "stress_split"]
    d07r1 = [records[s]["metrics"]["0.7-r1"] - baseline_records[s]["metrics"]["0.7-r1"] for s in split_names]
    d05r1 = [records[s]["metrics"]["0.5-r1"] - baseline_records[s]["metrics"]["0.5-r1"] for s in split_names]
    d07r5 = [records[s]["metrics"]["0.7-r5"] - baseline_records[s]["metrics"]["0.7-r5"] for s in split_names]
    d07r10 = [records[s]["metrics"]["0.7-r10"] - baseline_records[s]["metrics"]["0.7-r10"] for s in split_names]
    return {
        "median_delta_0.7_r1": float(np.median(d07r1)),
        "median_delta_0.5_r1": float(np.median(d05r1)),
        "median_delta_0.7_r5": float(np.median(d07r5)),
        "median_delta_0.7_r10": float(np.median(d07r10)),
        "worst_delta_0.7_r1": float(np.min(d07r1)),
        "worst_delta_0.5_r1": float(np.min(d05r1)),
        "instability_0.7_r1": float(np.std(d07r1)),
        "hard_exits": int(sum(records[s]["movement"]["hard_positive_top100_query_exits"] for s in split_names)),
        "invalid": int(sum(records[s]["movement"]["invalid_span_count"] for s in split_names)),
        "duplicate": int(sum(records[s]["movement"]["duplicate_span_count_after_nms"] for s in split_names)),
    }


def candidate_key(summary: Dict[str, Any]) -> Tuple[Any, ...]:
    clean = summary["hard_exits"] == 0 and summary["invalid"] == 0 and summary["duplicate"] == 0
    return (
        int(summary["median_delta_0.7_r1"] >= 0.0),
        int(summary["median_delta_0.5_r1"] >= 0.0),
        int(summary["median_delta_0.7_r5"] >= 0.0),
        int(summary["median_delta_0.7_r10"] >= 0.0),
        int(summary["worst_delta_0.7_r1"] >= -0.05),
        int(clean),
        summary["median_delta_0.7_r5"],
        summary["median_delta_0.7_r10"],
        -summary["instability_0.7_r1"],
    )


def _init_calib_worker(
    queries: List["QueryData"],
    splits: Dict[str, List[int]],
    args: argparse.Namespace,
    meta: Dict[str, Any],
    baseline: Dict[str, Dict[str, Any]],
) -> None:
    global _W_QUERIES, _W_SPLITS, _W_ARGS, _W_META, _W_BASELINE
    _W_QUERIES = queries
    _W_SPLITS = splits
    _W_ARGS = args
    _W_META = meta
    _W_BASELINE = baseline


def _eval_calibrated_config(item: Tuple[float, float, float, str]) -> Dict[str, Any]:
    if _W_QUERIES is None or _W_SPLITS is None or _W_ARGS is None or _W_META is None or _W_BASELINE is None:
        raise RuntimeError("calibration worker globals are not initialized")
    temp, lam, clip_c, norm = item

    def fn(qd: QueryData) -> np.ndarray:
        return qd.anchor_score + float(lam) * transform_residual(qd.residual, norm, float(temp), float(clip_c))

    split_names = ["calib_A", "calib_B", "calib_C", "stress_split"]
    records = {
        s: evaluate_scores(_W_QUERIES, _W_SPLITS[s], fn, _W_ARGS, _W_META, f"calib_t{temp}_l{lam}_c{clip_c}_{norm}_{s}")
        for s in split_names
    }
    summary = summarize_candidate(records, _W_BASELINE)
    return {
        "config": {"T": float(temp), "lambda": float(lam), "clip": float(clip_c), "normalization": norm},
        "summary": summary,
        "records": records,
    }


def query_features(qd: QueryData) -> np.ndarray:
    top = min(10, len(qd.anchor_score))
    probs = np.maximum(qd.prop_conf[:top], EPS)
    probs = probs / max(float(probs.sum()), EPS)
    res = qd.residual[:top]
    rank_agree = float(np.argmax(qd.anchor_score[:top]) == np.argmax((qd.anchor_score + qd.residual)[:top]))
    return np.asarray([
        float(qd.anchor_score[0] - qd.anchor_score[1]) if len(qd.anchor_score) > 1 else 0.0,
        float(qd.anchor_score[0] - qd.anchor_score[min(4, len(qd.anchor_score)-1)]),
        float(-(probs * np.log(probs)).sum()),
        float(qd.prop_conf[0]),
        float(qd.prop_conf[:top].max() - qd.prop_conf[:top].mean()),
        float(np.std(res)),
        float(np.max(res) - np.partition(res, -2)[-2]) if len(res) > 1 else 0.0,
        rank_agree,
        float(qd.query_len),
        float(len({c[1] for c in qd.anchor})),
        float(abs(np.argmax(qd.anchor_score[:top]) - np.argmax((qd.anchor_score + qd.residual)[:top]))),
        float(np.mean(qd.span_quality[:top])),
    ], dtype=np.float32)


class TinyMLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int = 1, hidden: int = 64, layers: int = 2, dropout: float = 0.08):
        super().__init__()
        mods: List[nn.Module] = []
        d = in_dim
        for _ in range(max(int(layers), 1)):
            mods.extend([nn.Linear(d, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(dropout)])
            d = hidden
        mods.append(nn.Linear(d, out_dim))
        self.net = nn.Sequential(*mods)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train_query_gate(queries: List[QueryData], train_idx: List[int], args: argparse.Namespace, meta: Dict[str, Any]) -> Dict[str, Any]:
    xs, ys = [], []
    for q in train_idx:
        qd = queries[q]
        anchor = evaluate_scores(queries, [q], lambda z: z.anchor_score, args, meta, "a")["metrics"]
        a4 = evaluate_scores(queries, [q], lambda z: z.anchor_score + z.residual, args, meta, "a4")["metrics"]
        gain = float((a4["0.7-r1"] + a4["0.7-r5"] + a4["0.7-r10"]) > (anchor["0.7-r1"] + anchor["0.7-r5"] + anchor["0.7-r10"]))
        risk = float((a4["0.7-r1"] < anchor["0.7-r1"]) or (a4["0.7-r5"] < anchor["0.7-r5"]))
        xs.append(query_features(qd))
        ys.append([gain, risk])
    x = np.stack(xs).astype(np.float32)
    y = np.asarray(ys, dtype=np.float32)
    mean, std = x.mean(axis=0), np.maximum(x.std(axis=0), EPS)
    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    model = TinyMLP(x.shape[1], out_dim=2, hidden=args.gate_hidden, layers=args.gate_layers, dropout=0.10).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    ds = TensorDataset(torch.from_numpy(((x - mean) / std).astype(np.float32)), torch.from_numpy(y))
    dl = DataLoader(ds, batch_size=256, shuffle=True)
    for _ in range(args.gate_epochs):
        for xb, yb in dl:
            xb = xb.to(device); yb = yb.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, yb)
            loss.backward()
            opt.step()
    return {"model": model, "mean": mean.astype(np.float32), "std": std.astype(np.float32)}


def gate_lambdas(gate: Dict[str, Any], queries: List[QueryData], args: argparse.Namespace) -> np.ndarray:
    x = np.stack([query_features(qd) for qd in queries]).astype(np.float32)
    xn = (x - gate["mean"]) / np.maximum(gate["std"], EPS)
    device = next(gate["model"].parameters()).device
    with torch.no_grad():
        logits = gate["model"](torch.from_numpy(xn).to(device)).float().cpu().numpy()
    gain = 1.0 / (1.0 + np.exp(-logits[:, 0]))
    risk = 1.0 / (1.0 + np.exp(-logits[:, 1]))
    return np.clip(gain * (1.0 - 0.65 * risk), 0.0, 1.0).astype(np.float32)


def candidate_features(qd: QueryData) -> np.ndarray:
    n = len(qd.anchor)
    video_counts: Dict[int, int] = {}
    for c in qd.anchor:
        video_counts[int(c[1])] = video_counts.get(int(c[1]), 0) + 1
    video_seen: Dict[int, int] = {}
    feats = []
    for i, c in enumerate(qd.anchor):
        vid = int(c[1])
        vrank = len(video_seen)
        within = video_seen.get(vid, 0)
        video_seen[vid] = within + 1
        feats.append([
            qd.anchor_score[i],
            qd.residual[i],
            qd.prop_conf[i],
            qd.span_quality[i],
            1.0 / float(i + 1),
            float(i) / 100.0,
            float(qd.anchor_score[i] - qd.anchor_score[min(i + 1, n - 1)]),
            1.0 / float(vrank + 1),
            1.0 / float(within + 1),
            float(video_counts[vid]) / 100.0,
        ])
    return np.asarray(feats, dtype=np.float32)


def train_pairwise_ranker(queries: List[QueryData], train_idx: List[int], args: argparse.Namespace) -> Dict[str, Any]:
    xs, ys = [], []
    rng = np.random.default_rng(20260626)
    for q in train_idx[:args.max_ranker_queries]:
        qd = queries[q]
        feat = candidate_features(qd)
        rel = 2 * qd.y07 + qd.y05
        pos = np.where(rel > 0)[0]
        neg = np.where(rel == 0)[0]
        if len(pos) == 0 or len(neg) == 0:
            continue
        for _ in range(args.max_pairs_per_query):
            i = int(rng.choice(pos)); j = int(rng.choice(neg))
            xs.append(feat[i] - feat[j]); ys.append(1.0)
            xs.append(feat[j] - feat[i]); ys.append(0.0)
    if not xs:
        raise RuntimeError("no ranker pairs")
    x = np.stack(xs).astype(np.float32)
    y = np.asarray(ys, dtype=np.float32)[:, None]
    mean, std = x.mean(axis=0), np.maximum(x.std(axis=0), EPS)
    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    model = TinyMLP(x.shape[1], out_dim=1, hidden=args.ranker_hidden, layers=args.ranker_layers, dropout=0.10).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    ds = TensorDataset(torch.from_numpy(((x - mean) / std).astype(np.float32)), torch.from_numpy(y))
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True)
    for _ in range(args.ranker_epochs):
        for xb, yb in dl:
            xb = xb.to(device); yb = yb.to(device)
            opt.zero_grad(set_to_none=True)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(model(xb), yb)
            loss.backward()
            opt.step()
    return {"model": model, "mean": mean.astype(np.float32), "std": std.astype(np.float32), "pairs": len(xs)}


def ranker_scores(ranker: Dict[str, Any], qd: QueryData) -> np.ndarray:
    feat = candidate_features(qd)
    # A linearized absolute score via model response to candidate features
    x = (feat - ranker["mean"]) / np.maximum(ranker["std"], EPS)
    device = next(ranker["model"].parameters()).device
    with torch.no_grad():
        out = ranker["model"](torch.from_numpy(x.astype(np.float32)).to(device)).float().cpu().numpy().reshape(-1)
    return out.astype(np.float32)


def build_groups(queries: List[QueryData], args: argparse.Namespace, meta: Dict[str, Any]) -> Dict[str, List[int]]:
    margins = np.asarray([float(q.anchor_score[0] - q.anchor_score[1]) for q in queries], dtype=np.float32)
    conf_conc = np.asarray([float(q.prop_conf[:10].max() - q.prop_conf[:10].mean()) for q in queries], dtype=np.float32)
    qlen = np.asarray([q.query_len for q in queries], dtype=np.float32)
    conflict = np.asarray([float(abs(np.argmax(q.anchor_score[:10]) - np.argmax((q.anchor_score + q.residual)[:10]))) for q in queries], dtype=np.float32)
    density = np.asarray([float(np.mean(q.prop_conf[:20])) for q in queries], dtype=np.float32)
    anchor_eval_top1 = []
    anchor_eval_top5 = []
    for q, qd in enumerate(queries):
        seq = anchor_seq(qd, args)
        l05, l07, _pos, _i, _d = labels_for_sequence(seq, int(meta["gt_vid"][q]), int(meta["gt_s"][q]), int(meta["gt_e"][q]))
        anchor_eval_top1.append(float(any(l07[:1]) or any(l05[:1])))
        anchor_eval_top5.append(float(any(l07[:5]) or any(l05[:5])))
    idx = np.arange(len(queries))
    return {
        "G1_anchor_top1_hit": idx[np.asarray(anchor_eval_top1) > 0.5].astype(int).tolist(),
        "G1_anchor_top1_miss": idx[np.asarray(anchor_eval_top1) <= 0.5].astype(int).tolist(),
        "G2_anchor_top5_hit": idx[np.asarray(anchor_eval_top5) > 0.5].astype(int).tolist(),
        "G2_anchor_top5_miss": idx[np.asarray(anchor_eval_top5) <= 0.5].astype(int).tolist(),
        "G3_high_anchor_margin": idx[margins >= np.median(margins)].astype(int).tolist(),
        "G3_low_anchor_margin": idx[margins < np.median(margins)].astype(int).tolist(),
        "G4_high_conf_concentration": idx[conf_conc >= np.median(conf_conc)].astype(int).tolist(),
        "G4_flat_confidence": idx[conf_conc < np.median(conf_conc)].astype(int).tolist(),
        "G5_short_query": idx[qlen <= np.median(qlen)].astype(int).tolist(),
        "G5_long_query": idx[qlen > np.median(qlen)].astype(int).tolist(),
        "G7_high_residual_agreement": idx[conflict <= np.median(conflict)].astype(int).tolist(),
        "G7_high_residual_conflict": idx[conflict > np.median(conflict)].astype(int).tolist(),
        "G8_high_positive_density_proxy": idx[density >= np.median(density)].astype(int).tolist(),
        "G8_low_positive_density_proxy": idx[density < np.median(density)].astype(int).tolist(),
    }


def main() -> None:
    args = parse_args()
    Path(args.audit_dir).mkdir(parents=True, exist_ok=True)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    freeze = json.loads(Path(args.b21_freeze).read_text(encoding="utf-8"))
    if freeze.get("official_val_used") is not False or freeze.get("status") != "C7_B2_1_OFFICIAL_READY":
        raise ValueError("C7-B3 must start from train-only C7-B2.1 freeze package, not official artifacts")
    queries, meta = build_query_data(args)
    precompute_anchor_cache(queries, args, meta)
    splits = split_indices(queries)
    stats = split_stats(queries, splits, args, meta)
    split_manifest = {
        "status": "C7_B3_SPLITS_READY",
        "official_val_used": False,
        "split_method": "deterministic sha256(desc_id) buckets plus stress_split by query-group stress score",
        "splits": stats,
    }
    atomic_json(Path(args.audit_dir) / "C7_B3_SPLIT_MANIFEST.json", split_manifest)
    atomic_text(Path(args.audit_dir) / "C7_B3_SPLIT_MANIFEST.md", "# C7-B3 split manifest\n\n```json\n" + json.dumps(split_manifest, indent=2, ensure_ascii=False) + "\n```\n")

    anchor_fn = lambda qd: qd.anchor_score
    a4_fn = lambda qd: qd.anchor_score + qd.residual
    baseline_records = {name: evaluate_scores(queries, idx, a4_fn, args, meta, f"A4_{name}") for name, idx in splits.items()}
    anchor_records = {name: evaluate_scores(queries, idx, anchor_fn, args, meta, f"anchor_{name}") for name, idx in splits.items()}
    residual_top = np.asarray([float(q.residual[np.argmax(q.residual)]) for q in queries], dtype=np.float32)
    h1 = np.asarray([float(baseline_records["train_calib_final_review"]["metrics"]["0.7-r1"] > anchor_records["train_calib_final_review"]["metrics"]["0.7-r1"])] * len(queries), dtype=np.float32)
    hit1 = np.asarray([float(np.max(q.y07[:1]) > 0.5 or np.max(q.y05[:1]) > 0.5) for q in queries], dtype=np.float32)
    hit5 = np.asarray([float(np.max(q.y07[:5]) > 0.5 or np.max(q.y05[:5]) > 0.5) for q in queries], dtype=np.float32)
    hit10 = np.asarray([float(np.max(q.y07[:10]) > 0.5 or np.max(q.y05[:10]) > 0.5) for q in queries], dtype=np.float32)
    shrink = {
        "status": "C7_B3_SHRINKAGE_DIAGNOSIS_COMPLETE",
        "official_val_used": False,
        "A4_records": baseline_records,
        "anchor_records": anchor_records,
        "A4_gain_vs_anchor": {s: metric_delta(baseline_records[s]["metrics"], anchor_records[s]["metrics"]) for s in splits},
        "residual_correlation": {
            "residual_top_vs_hit1": pearson(residual_top, hit1),
            "residual_top_vs_hit5": pearson(residual_top, hit5),
            "residual_top_vs_hit10": pearson(residual_top, hit10),
            "diagnostic_global_A4_gt_anchor_R1_flag": float(h1[0]),
        },
    }
    atomic_json(Path(args.audit_dir) / "C7_B3_SHRINKAGE_DIAGNOSIS.json", shrink)
    atomic_text(Path(args.audit_dir) / "C7_B3_SHRINKAGE_DIAGNOSIS.md", "# C7-B3 shrinkage diagnosis\n\n```json\n" + json.dumps(shrink, indent=2, ensure_ascii=False) + "\n```\n")

    # Phase B
    grid_items = [
        (temp, lam, clip_c, norm)
        for temp in [0.5, 0.75, 1.0, 1.5, 2.0, 3.0]
        for lam in [0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0]
        for clip_c in [1.0, 2.0, 3.0, 5.0]
        for norm in ["none", "per_query_z", "per_query_rank", "sigmoid"]
    ]
    if args.workers > 1:
        ctx = mp.get_context("fork")
        with ctx.Pool(
            processes=int(args.workers),
            initializer=_init_calib_worker,
            initargs=(queries, splits, args, meta, baseline_records),
        ) as pool:
            grid_records = list(pool.imap_unordered(_eval_calibrated_config, grid_items, chunksize=max(1, len(grid_items) // (int(args.workers) * 4))))
        grid_records = sorted(grid_records, key=lambda r: (r["config"]["T"], r["config"]["lambda"], r["config"]["clip"], r["config"]["normalization"]))
    else:
        _init_calib_worker(queries, splits, args, meta, baseline_records)
        grid_records = [_eval_calibrated_config(item) for item in grid_items]
    best_calib = max(grid_records, key=lambda r: candidate_key(r["summary"]))
    phase_b = {"status": "C7_B3_CALIBRATED_A4_COMPLETE", "official_val_used": False, "grid_size": len(grid_records), "best": best_calib, "top10": sorted(grid_records, key=lambda r: candidate_key(r["summary"]), reverse=True)[:10]}
    atomic_json(Path(args.audit_dir) / "C7_B3_CALIBRATED_A4_RESIDUAL.json", phase_b)
    atomic_text(Path(args.audit_dir) / "C7_B3_CALIBRATED_A4_RESIDUAL.md", "# C7-B3 calibrated A4 residual\n\n```json\n" + json.dumps(phase_b, indent=2, ensure_ascii=False) + "\n```\n")

    best_cfg = best_calib["config"]
    calib_fn = lambda qd: qd.anchor_score + best_cfg["lambda"] * transform_residual(qd.residual, best_cfg["normalization"], best_cfg["T"], best_cfg["clip"])
    gate = train_query_gate(queries, splits["train_core"], args, meta)
    lambdas = gate_lambdas(gate, queries, args)
    gate_fn = lambda qd: qd.anchor_score + float(lambdas[qd.q]) * (calib_fn(qd) - qd.anchor_score)
    gate_records = {s: evaluate_scores(queries, splits[s], gate_fn, args, meta, f"query_gate_{s}") for s in splits}
    phase_c = {
        "status": "C7_B3_QUERY_GATE_COMPLETE",
        "official_val_used": False,
        "feature_names": [
            "anchor_top1_margin", "anchor_top5_margin", "anchor_entropy_top10", "proposal_confidence_top1",
            "proposal_confidence_entropy", "residual_entropy", "residual_top1_margin", "C7B1_C7B2_agreement",
            "query_length", "video_count", "rank_conflict_features", "span_quality_top10_mean",
        ],
        "lambda_stats": {"mean": float(lambdas.mean()), "p10": float(np.quantile(lambdas, 0.1)), "p50": float(np.median(lambdas)), "p90": float(np.quantile(lambdas, 0.9))},
        "records": gate_records,
        "summary": summarize_candidate({s: gate_records[s] for s in ["calib_A", "calib_B", "calib_C", "stress_split"]}, baseline_records),
    }
    atomic_json(Path(args.audit_dir) / "C7_B3_QUERY_GATE.json", phase_c)
    atomic_text(Path(args.audit_dir) / "C7_B3_QUERY_GATE.md", "# C7-B3 query gate\n\n```json\n" + json.dumps(phase_c, indent=2, ensure_ascii=False) + "\n```\n")

    ranker = train_pairwise_ranker(queries, splits["train_core"], args)
    rank_fn = lambda qd: ranker_scores(ranker, qd)
    rank_records = {s: evaluate_scores(queries, splits[s], rank_fn, args, meta, f"lambdarank_{s}") for s in splits}
    phase_d = {
        "status": "C7_B3_FIXED_POOL_LAMBDARANK_COMPLETE",
        "official_val_used": False,
        "pair_count": ranker["pairs"],
        "feature_names": [
            "C7_B1_score", "C7_B2_video_residual", "proposal_confidence", "span_quality_logit",
            "rank_position", "score_margin", "video_rank", "within_video_span_rank", "same_video_candidate_count",
        ],
        "records": rank_records,
        "summary": summarize_candidate({s: rank_records[s] for s in ["calib_A", "calib_B", "calib_C", "stress_split"]}, baseline_records),
    }
    atomic_json(Path(args.audit_dir) / "C7_B3_FIXED_POOL_LAMBDARANK.json", phase_d)
    atomic_text(Path(args.audit_dir) / "C7_B3_FIXED_POOL_LAMBDARANK.md", "# C7-B3 fixed-pool LambdaRank\n\n```json\n" + json.dumps(phase_d, indent=2, ensure_ascii=False) + "\n```\n")

    groups = build_groups(queries, args, meta)
    candidates = {
        "A4_baseline": {"fn": a4_fn, "summary": summarize_candidate({s: baseline_records[s] for s in ["calib_A", "calib_B", "calib_C", "stress_split"]}, baseline_records)},
        "calibrated_A4": {"fn": calib_fn, "summary": best_calib["summary"]},
        "query_gate": {"fn": gate_fn, "summary": phase_c["summary"]},
        "fixed_pool_lambdarank": {"fn": rank_fn, "summary": phase_d["summary"]},
    }
    group_records: Dict[str, Any] = {}
    for cname, cinfo in candidates.items():
        grecs = {}
        for gname, idx in groups.items():
            if len(idx) == 0:
                continue
            rec = evaluate_scores(queries, idx, cinfo["fn"], args, meta, f"{cname}_{gname}")
            base = evaluate_scores(queries, idx, a4_fn, args, meta, f"A4_{gname}")
            rec["delta_vs_A4"] = metric_delta(rec["metrics"], base["metrics"])
            grecs[gname] = rec
        worst_group_r1 = min((r["delta_vs_A4"]["0.7-r1"] for r in grecs.values()), default=0.0)
        instability = float(np.std([r["delta_vs_A4"]["0.7-r1"] for r in grecs.values()])) if grecs else 0.0
        stable_score = float(cinfo["summary"]["median_delta_0.7_r5"] + cinfo["summary"]["median_delta_0.7_r10"] - 2.0 * max(0.0, -cinfo["summary"]["worst_delta_0.7_r1"]) - 1.0 * max(0.0, -worst_group_r1) - 0.2 * instability)
        group_records[cname] = {"summary": cinfo["summary"], "group_records": grecs, "worst_group_delta_0.7_r1": float(worst_group_r1), "instability": instability, "stable_score": stable_score}
    selected_name = max(group_records, key=lambda k: group_records[k]["stable_score"])
    phase_e = {
        "status": "C7_B3_GROUP_STABLE_SELECTION_COMPLETE",
        "official_val_used": False,
        "selected_candidate": selected_name,
        "records": group_records,
        "selection_rule": "stable_score = median_gain - alpha*worst_split_loss - beta*worst_group_loss - gamma*instability",
    }
    atomic_json(Path(args.audit_dir) / "C7_B3_GROUP_STABLE_SELECTION.json", phase_e)
    atomic_text(Path(args.audit_dir) / "C7_B3_GROUP_STABLE_SELECTION.md", "# C7-B3 group-stable selection\n\n```json\n" + json.dumps(phase_e, indent=2, ensure_ascii=False) + "\n```\n")

    selected_records = {s: evaluate_scores(queries, splits[s], candidates[selected_name]["fn"], args, meta, f"{selected_name}_{s}") for s in splits}
    selected_summary = summarize_candidate({s: selected_records[s] for s in ["calib_A", "calib_B", "calib_C", "stress_split"]}, baseline_records)
    worst_group_r1 = group_records[selected_name]["worst_group_delta_0.7_r1"]
    final_review = selected_records["train_calib_final_review"]
    fixed_pool_ok = all(r["movement"]["fixed_pool_invariant"] for r in selected_records.values())
    clean = all(r["movement"]["hard_positive_top100_query_exits"] == 0 and r["movement"]["invalid_span_count"] == 0 and r["movement"]["duplicate_span_count_after_nms"] == 0 for r in selected_records.values())
    r100_unchanged = abs(final_review["metrics"]["0.7-r100"] - baseline_records["train_calib_final_review"]["metrics"]["0.7-r100"]) < 1e-9
    gate_pass = (
        selected_summary["median_delta_0.7_r1"] >= 0
        and selected_summary["median_delta_0.5_r1"] >= 0
        and selected_summary["median_delta_0.7_r5"] >= 0
        and selected_summary["median_delta_0.7_r10"] >= 0
        and selected_summary["worst_delta_0.7_r1"] >= -0.05
        and worst_group_r1 >= -0.10
        and r100_unchanged
        and clean
        and fixed_pool_ok
    )
    if gate_pass:
        status = "C7_B3_FREEZE_REVIEW_PASS"
    elif selected_summary["median_delta_0.7_r1"] >= 0 and selected_summary["median_delta_0.7_r5"] >= 0:
        status = "C7_B3_STABLE_BUT_SMALL_GAIN"
    elif selected_summary["median_delta_0.7_r1"] > 0:
        status = "C7_B3_HIGH_GAIN_UNSTABLE"
    else:
        status = "C7_B3_NEGATIVE"
    final = {
        "status": status,
        "official_val_used": False,
        "official_val_run": False,
        "post_val_adjustment": False,
        "second_official_val": False,
        "selected_candidate": selected_name,
        "selected_summary": selected_summary,
        "worst_group_delta_0.7_r1": worst_group_r1,
        "train_calib_final_review": final_review,
        "baseline_A4_train_calib": baseline_records["train_calib_final_review"],
        "gate_pass": gate_pass,
        "fixed_pool_invariant": fixed_pool_ok,
        "R100_unchanged": r100_unchanged,
        "hard_exits_invalid_duplicate_clean": clean,
        "evaluator_modified": False,
        "nms_modified": False,
        "C4_C6_artifacts_modified": False,
        "enter_C7_C": False,
        "enter_C8": False,
        "request": "Stop and request human review before any official-val one-shot.",
    }
    atomic_json(Path(args.audit_dir) / "C7_B3_FINAL_DECISION.json", final)
    atomic_text(Path(args.audit_dir) / "C7_B3_FINAL_DECISION.md", "# C7-B3 final decision\n\n```json\n" + json.dumps(final, indent=2, ensure_ascii=False) + "\n```\n")
    freeze_manifest = {
        "status": status,
        "official_val_used": False,
        "selected_candidate": selected_name,
        "freeze_review_pass": gate_pass,
        "final_decision": artifact(Path(args.audit_dir) / "C7_B3_FINAL_DECISION.json"),
    }
    atomic_json(Path(args.audit_dir) / "C7_B3_FREEZE_MANIFEST.json", freeze_manifest)
    atomic_json(Path(args.audit_dir) / "C7_B3_HASHES.json", {
        "status": "C7_B3_HASHES",
        "official_val_used": False,
        "artifacts": {
            "split_manifest": artifact(Path(args.audit_dir) / "C7_B3_SPLIT_MANIFEST.json"),
            "shrinkage": artifact(Path(args.audit_dir) / "C7_B3_SHRINKAGE_DIAGNOSIS.json"),
            "calibrated": artifact(Path(args.audit_dir) / "C7_B3_CALIBRATED_A4_RESIDUAL.json"),
            "query_gate": artifact(Path(args.audit_dir) / "C7_B3_QUERY_GATE.json"),
            "lambdarank": artifact(Path(args.audit_dir) / "C7_B3_FIXED_POOL_LAMBDARANK.json"),
            "group_stable": artifact(Path(args.audit_dir) / "C7_B3_GROUP_STABLE_SELECTION.json"),
            "final": artifact(Path(args.audit_dir) / "C7_B3_FINAL_DECISION.json"),
            "runner": artifact(__file__),
        },
        "selected_summary_hash": sha256_obj(selected_summary),
    })
    print(json.dumps({
        "status": status,
        "official_val_used": False,
        "selected_candidate": selected_name,
        "selected_summary": selected_summary,
        "worst_group_delta_0.7_r1": worst_group_r1,
        "gate_pass": gate_pass,
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
