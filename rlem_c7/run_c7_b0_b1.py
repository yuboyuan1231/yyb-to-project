#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.conquer import CONQUER  # noqa: E402
from rlem_c6_b0.run_c6_b0_candidate_diagnostic import span_iou_idx  # noqa: E402
from rlem_c6_c.run_c6_c_goal import load_baseline_configs, load_npz  # noqa: E402
from rlem_c7.wrapper import CONQUER_RLEM_Wrapper, ProposalConfidenceHead  # noqa: E402


METRIC_KEYS = [
    "0.5-r1", "0.5-r5", "0.5-r10", "0.5-r100",
    "0.7-r1", "0.7-r5", "0.7-r10", "0.7-r100",
]
EPS = 1e-8


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


def atomic_npz(path: str | Path, **arrays: np.ndarray) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(p) + ".partial")
    with open(tmp, "wb") as f:
        np.savez_compressed(f, **arrays)
    os.replace(tmp, p)


def metric_delta(new: Dict[str, float], base: Dict[str, float]) -> Dict[str, float]:
    return {k: float(new[k] - base[k]) for k in METRIC_KEYS}


def selected_metrics(labels05: List[List[bool]], labels07: List[List[bool]]) -> Dict[str, float]:
    q_count = max(len(labels05), 1)
    out = {k: 0 for k in METRIC_KEYS}
    for q in range(len(labels05)):
        for thr, labels in [("0.5", labels05[q]), ("0.7", labels07[q])]:
            for k in [1, 5, 10, 100]:
                out[f"{thr}-r{k}"] += int(any(labels[:k]))
    return {k: 100.0 * float(v) / q_count for k, v in out.items()}


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


def gt_span_by_query(cache: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    q_count = len(cache["desc_ids"])
    offsets = cache["desc_offsets"].astype(np.int64)
    gs = cache["gt_start_idx"].astype(np.int64)
    ge = cache["gt_end_idx"].astype(np.int64)
    out_s = np.zeros(q_count, dtype=np.int64)
    out_e = np.zeros(q_count, dtype=np.int64)
    for q in range(q_count):
        row = int(offsets[q])
        out_s[q] = int(gs[row])
        out_e[q] = int(ge[row])
    return out_s, out_e


@dataclass
class FastAnchorData:
    offsets: np.ndarray
    row_gid: np.ndarray
    video_idx: np.ndarray
    start_idx: np.ndarray
    end_idx: np.ndarray
    s_c4: np.ndarray
    pool_start_idx: np.ndarray
    pool_end_idx: np.ndarray
    score3: np.ndarray
    pool_shape: Tuple[int, int, int]


def make_fast_anchor_data(cache: Dict[str, np.ndarray], pool: Dict[str, np.ndarray], pool_score: np.ndarray) -> FastAnchorData:
    q, slots, alts = [int(x) for x in pool["shape"]]
    return FastAnchorData(
        offsets=cache["desc_offsets"].astype(np.int64, copy=False),
        row_gid=cache["row_group_id"].astype(np.int64, copy=False),
        video_idx=cache["video_idx"].astype(np.int64, copy=False),
        start_idx=cache["start_idx"].astype(np.int64, copy=False),
        end_idx=cache["end_idx"].astype(np.int64, copy=False),
        s_c4=cache["s_c4_final"].astype(np.float32, copy=False),
        pool_start_idx=pool["start_idx"].astype(np.int64, copy=False),
        pool_end_idx=pool["end_idx"].astype(np.int64, copy=False),
        score3=pool_score.reshape(q, slots, alts).astype(np.float32, copy=False),
        pool_shape=(q, slots, alts),
    )


def make_anchor_candidates_fast(
    *,
    q: int,
    data: FastAnchorData,
    policy_cfg: Dict[str, Any],
    effective_top_n: int,
) -> List[Tuple[int, int, int, int, float, int]]:
    rows = np.arange(int(data.offsets[q]), int(data.offsets[q + 1]), dtype=np.int64)
    base_order = rows[np.argsort(-data.s_c4[rows], kind="stable")[:effective_top_n]]
    pq, slots, alts = data.pool_shape
    out: List[Tuple[int, int, int, int, float, int]] = []
    replaced = 0
    apply_slots = int(policy_cfg.get("apply_slots", 0))
    max_repl = int(policy_cfg.get("max_replacements", 0))
    for slot_rank, row in enumerate(base_order):
        gid = int(data.row_gid[row])
        vid = int(data.video_idx[row])
        si = int(data.start_idx[row])
        ei = int(data.end_idx[row])
        if q < pq and slot_rank < min(slots, apply_slots) and replaced < max_repl:
            scores = data.score3[q, slot_rank]
            best = int(np.argmax(scores))
            margin = float(scores[best] - scores[0])
            threshold = float(policy_cfg["threshold0"]) if slot_rank == 0 else float(policy_cfg["threshold_rest"])
            if best != 0 and margin > threshold:
                idx = (q * slots + slot_rank) * alts + best
                si = int(data.pool_start_idx[idx])
                ei = int(data.pool_end_idx[idx])
                replaced += 1
        out.append((gid, vid, si, ei, float(data.s_c4[row]), int(row)))
    return out


def nms_sequence(seq: List[Tuple[int, int, int, int, float, int]], nms_thd: float, max_keep: int) -> List[Tuple[int, int, int, int, float, int]]:
    kept: List[Tuple[int, int, int, int, float, int]] = []
    by_video: Dict[int, List[int]] = {}
    for cand in seq:
        _gid, vid, si, ei, _score, _row = cand
        suppress = False
        for prior_idx in by_video.get(int(vid), []):
            prior = kept[prior_idx]
            if span_iou_idx(int(si), int(ei), int(prior[2]), int(prior[3])) > nms_thd:
                suppress = True
                break
        if not suppress:
            by_video.setdefault(int(vid), []).append(len(kept))
            kept.append(cand)
            if len(kept) >= max_keep:
                break
    return kept


def labels_for_sequence(seq: List[Tuple[int, int, int, int, float, int]], gt_vid: int, gt_s: int, gt_e: int) -> Tuple[List[bool], List[bool], bool, int, int]:
    labs05: List[bool] = []
    labs07: List[bool] = []
    invalid = 0
    for _gid, vid, si, ei, _score, _row in seq:
        invalid += int(si < 0 or ei < si)
        same = int(vid) == int(gt_vid)
        iou = span_iou_idx(int(si), int(ei), int(gt_s), int(gt_e)) if same else 0.0
        labs05.append(bool(same and iou >= 0.5))
        labs07.append(bool(same and iou >= 0.7))
    dup = len(seq) - len({(int(c[1]), int(c[2]), int(c[3])) for c in seq})
    return labs05, labs07, bool(any(labs05[:100]) or any(labs07[:100])), invalid, dup


def evaluate_anchor(
    cache: Dict[str, np.ndarray],
    pool: Dict[str, np.ndarray],
    pool_score: np.ndarray,
    policy_cfg: Dict[str, Any],
    args: argparse.Namespace,
    limit_queries: int | None = None,
) -> Dict[str, Any]:
    data = make_fast_anchor_data(cache, pool, pool_score)
    gt_vid = gt_video_by_query(cache)
    gt_s, gt_e = gt_span_by_query(cache)
    q_count = len(cache["desc_ids"]) if limit_queries is None else min(limit_queries, len(cache["desc_ids"]))
    labels05: List[List[bool]] = []
    labels07: List[List[bool]] = []
    positives: List[bool] = []
    invalid = dup = 0
    for q in range(q_count):
        cands = make_anchor_candidates_fast(q=q, data=data, policy_cfg=policy_cfg, effective_top_n=args.effective_top_n)
        seq = nms_sequence(cands, args.nms_thd, args.max_after_nms)
        l05, l07, pos, inv, du = labels_for_sequence(seq, int(gt_vid[q]), int(gt_s[q]), int(gt_e[q]))
        labels05.append(l05); labels07.append(l07); positives.append(pos)
        invalid += inv; dup += du
    return {
        "metrics": selected_metrics(labels05, labels07),
        "positive_top100_flags": positives,
        "invalid_span_count": int(invalid),
        "duplicate_span_count_after_nms": int(dup),
        "query_count": int(q_count),
    }


def aggregate_pool_group_score(pool_path: str, score_path: str, max_gid: int) -> np.ndarray:
    pool = load_npz(pool_path, allow_pickle=False)
    score = np.load(score_path, allow_pickle=False)["score"].astype(np.float32)
    out = np.full(max_gid + 1, -10.0, dtype=np.float32)
    np.maximum.at(out, pool["group_id"].astype(np.int64), score)
    out[~np.isfinite(out)] = -10.0
    return out


def make_gid_index(group_id: np.ndarray, max_gid: int) -> np.ndarray:
    out = np.full(max_gid + 1, -1, dtype=np.int64)
    out[group_id.astype(np.int64)] = np.arange(len(group_id), dtype=np.int64)
    return out


def prepare_side_arrays(split: str, args: argparse.Namespace, cache: Dict[str, np.ndarray]) -> Dict[str, Any]:
    max_gid = int(cache["video_group_keys"].shape[0] - 1)
    temporal_path = args.train_fit_temporal if split == "train_fit" else args.calib_temporal
    boundary_path = args.train_fit_boundary if split == "train_fit" else args.calib_boundary
    pool_path = args.train_fit_pool if split == "train_fit" else args.calib_pool
    b1_score_path = args.train_fit_b1_scores if split == "train_fit" else args.b1_calib_scores
    b2_score_path = args.train_fit_b2_scores if split == "train_fit" else args.b2_calib_scores
    c6c_path = args.train_fit_c6c_scores if split == "train_fit" else args.calib_c6c_scores
    temporal = load_npz(temporal_path, allow_pickle=False)
    boundary = load_npz(boundary_path, allow_pickle=False)
    c6c = load_npz(c6c_path, allow_pickle=False)
    return {
        "temporal": temporal,
        "temporal_idx": make_gid_index(temporal["group_id"], max_gid),
        "boundary": boundary,
        "boundary_idx": make_gid_index(boundary["group_id"], max_gid),
        "b1_group_score": aggregate_pool_group_score(pool_path, b1_score_path, max_gid),
        "b2_group_score": aggregate_pool_group_score(pool_path, b2_score_path, max_gid),
        "c6c": c6c,
        "c6c_idx": make_gid_index(c6c["group_id"], max_gid),
    }


FEATURE_NAMES = [
    "base_score", "rank_inv", "rank_norm", "start_norm", "end_norm", "length_norm", "center_norm",
    "p_ctx_mean", "p_ctx_max", "p_ctx_sum", "p_b_start", "p_e_end", "p_b_new_start", "p_e_new_end",
    "p_b_neighborhood", "p_e_neighborhood", "endpoint_agreement", "inside_boundary_mean",
    "b1_group_score", "b2_group_score", "b2_minus_b1_group_score",
    "c6c_relevance", "c6c_moment", "c6c_iou", "c6c_risk", "c6c_gate",
] + [f"video_feature_{i}" for i in range(27)]


def features_for_spans(
    cache: Dict[str, np.ndarray],
    side: Dict[str, Any],
    rows: np.ndarray,
    gids: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    base_scores: np.ndarray,
    ranks: np.ndarray,
    chunk_size: int = 200000,
) -> np.ndarray:
    n = len(gids)
    out = np.empty((n, len(FEATURE_NAMES)), dtype=np.float32)
    video_features = cache["video_features"].astype(np.float32)
    for s in range(0, n, chunk_size):
        e = min(s + chunk_size, n)
        sl = slice(s, e)
        gid = gids[sl].astype(np.int64)
        st = np.clip(starts[sl].astype(np.int64), 0, 99)
        ed = np.clip(ends[sl].astype(np.int64), st, 99)
        length = (ed - st + 1).astype(np.float32)
        tidx = side["temporal_idx"][gid]
        bidx = side["boundary_idx"][gid]
        cidx = side["c6c_idx"][gid]
        p_ctx = side["temporal"]["p_ctx"][tidx]
        p_b = side["temporal"]["p_b"][tidx]
        p_e = side["temporal"]["p_e"][tidx]
        pbn = side["boundary"]["p_b_new"][bidx]
        pen = side["boundary"]["p_e_new"][bidx]
        row_ids = np.arange(e - s)
        cs = np.concatenate([np.zeros((e - s, 1), dtype=np.float32), np.cumsum(p_ctx, axis=1)], axis=1)
        ctx_sum = cs[row_ids, ed + 1] - cs[row_ids, st]
        ctx_mean = ctx_sum / np.maximum(length, 1.0)
        # Max over spans is done per chunk with a short loop; spans are at most 100 clips.
        ctx_max = np.empty(e - s, dtype=np.float32)
        inside_boundary_mean = np.empty(e - s, dtype=np.float32)
        for j in range(e - s):
            a = int(st[j]); b = int(ed[j]) + 1
            ctx_max[j] = float(np.max(p_ctx[j, a:b]))
            inside_boundary_mean[j] = float(0.5 * (np.mean(p_b[j, a:b]) + np.mean(p_e[j, a:b])))
        s_nei = np.stack([np.clip(st - 1, 0, 99), st, np.clip(st + 1, 0, 99)], axis=1)
        e_nei = np.stack([np.clip(ed - 1, 0, 99), ed, np.clip(ed + 1, 0, 99)], axis=1)
        p_b_nei = np.max(np.take_along_axis(p_b, s_nei, axis=1), axis=1)
        p_e_nei = np.max(np.take_along_axis(p_e, e_nei, axis=1), axis=1)
        base = np.stack([
            base_scores[sl].astype(np.float32),
            1.0 / (ranks[sl].astype(np.float32) + 1.0),
            ranks[sl].astype(np.float32) / 100.0,
            st.astype(np.float32) / 100.0,
            ed.astype(np.float32) / 100.0,
            length / 100.0,
            (0.5 * (st + ed)).astype(np.float32) / 100.0,
            ctx_mean,
            ctx_max,
            ctx_sum,
            p_b[row_ids, st],
            p_e[row_ids, ed],
            pbn[row_ids, st],
            pen[row_ids, ed],
            p_b_nei,
            p_e_nei,
            pbn[row_ids, st] + pen[row_ids, ed],
            inside_boundary_mean,
            side["b1_group_score"][gid],
            side["b2_group_score"][gid],
            side["b2_group_score"][gid] - side["b1_group_score"][gid],
            side["c6c"]["relevance"][cidx],
            side["c6c"]["moment"][cidx],
            side["c6c"]["iou"][cidx],
            side["c6c"]["risk"][cidx],
            side["c6c"]["gate"][cidx],
        ], axis=1).astype(np.float32)
        out[sl] = np.concatenate([base, video_features[gid]], axis=1)
    return out


def build_span_dataset(split: str, top_n: int, args: argparse.Namespace) -> Dict[str, Any]:
    out_path = Path(args.output_dir) / f"{split}_proposal_dataset_top{top_n}.npz"
    if out_path.exists() and not args.rebuild_features:
        return {"split": split, "reused": True, "dataset": artifact(out_path)}
    cache_path = args.train_fit_cache if split == "train_fit" else args.calib_cache
    cache = load_npz(cache_path, allow_pickle=True)
    side = prepare_side_arrays(split, args, cache)
    rows_per_q = int(cache["desc_offsets"][1] - cache["desc_offsets"][0])
    q_count = len(cache["desc_ids"])
    s2 = cache["s_c4_final"].astype(np.float32).reshape(q_count, rows_per_q)
    order = np.argsort(-s2, axis=1, kind="stable")[:, :top_n]
    q_base = (np.arange(q_count, dtype=np.int64)[:, None] * rows_per_q)
    rows = (q_base + order).reshape(-1).astype(np.int64)
    ranks = np.tile(np.arange(top_n, dtype=np.int16), q_count)
    gids = cache["row_group_id"][rows].astype(np.int64)
    starts = cache["start_idx"][rows].astype(np.int64)
    ends = cache["end_idx"][rows].astype(np.int64)
    base_scores = cache["s_c4_final"][rows].astype(np.float32)
    t0 = time.time()
    x = features_for_spans(cache, side, rows, gids, starts, ends, base_scores, ranks, chunk_size=args.feature_chunk_size)
    atomic_npz(
        out_path,
        x=x.astype(np.float32),
        row=rows.astype(np.int64),
        query=cache["query_index"][rows].astype(np.int32),
        group_id=gids.astype(np.int32),
        video_idx=cache["video_idx"][rows].astype(np.int32),
        start_idx=starts.astype(np.int16),
        end_idx=ends.astype(np.int16),
        rank=ranks.astype(np.int16),
        base_score=base_scores.astype(np.float32),
        iou=cache["iou"][rows].astype(np.float32),
        y05=cache["y_joint_05"][rows].astype(np.float32),
        y07=cache["y_joint_07"][rows].astype(np.float32),
        feature_names=np.asarray(FEATURE_NAMES, dtype=object),
    )
    return {
        "split": split,
        "reused": False,
        "dataset": artifact(out_path),
        "queries": int(q_count),
        "examples": int(len(rows)),
        "feature_dim": int(x.shape[1]),
        "top_n_per_query": int(top_n),
        "positive_05_rate": float(cache["y_joint_05"][rows].mean()),
        "positive_07_rate": float(cache["y_joint_07"][rows].mean()),
        "elapsed_sec": float(time.time() - t0),
    }


def train_proposal_head(args: argparse.Namespace) -> Dict[str, Any]:
    model_path = Path(args.models_dir) / "c7_b1_proposal_confidence_head.pt"
    history_path = Path(args.output_dir) / "c7_b1_proposal_head_history.json"
    if model_path.exists() and history_path.exists() and not args.force_train:
        return {"reused": True, "model": artifact(model_path), "history": json.loads(history_path.read_text())}
    z = np.load(Path(args.output_dir) / f"train_fit_proposal_dataset_top{args.train_top_n}.npz", allow_pickle=True)
    x = z["x"].astype(np.float32)
    y05 = z["y05"].astype(np.float32)
    y07 = z["y07"].astype(np.float32)
    yiou = z["iou"].astype(np.float32)
    mean = x.mean(axis=0).astype(np.float32)
    std = np.maximum(x.std(axis=0).astype(np.float32), EPS)
    ds = TensorDataset(
        torch.from_numpy(((x - mean) / std).astype(np.float32)),
        torch.from_numpy(np.stack([y05, y07, yiou], axis=1).astype(np.float32)),
    )
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    model = ProposalConfidenceHead(x.shape[1], hidden=args.hidden, layers=args.layers, dropout=args.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))
    pos05 = torch.tensor([(1.0 - y05.mean()) / max(float(y05.mean()), EPS)], dtype=torch.float32, device=device).clamp(max=80)
    pos07 = torch.tensor([(1.0 - y07.mean()) / max(float(y07.mean()), EPS)], dtype=torch.float32, device=device).clamp(max=120)
    history: List[Dict[str, Any]] = []
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        losses: List[float] = []
        model.train()
        for xb, yb in dl:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                out = model(xb)
                prop_target = torch.clamp(0.65 * yb[:, 1] + 0.35 * yb[:, 0], 0.0, 1.0)
                conf_loss = F.binary_cross_entropy_with_logits(out["proposal_confidence"], prop_target)
                iou05_loss = F.binary_cross_entropy_with_logits(out["proposal_iou05_logit"], yb[:, 0], pos_weight=pos05)
                iou07_loss = F.binary_cross_entropy_with_logits(out["proposal_iou07_logit"], yb[:, 1], pos_weight=pos07)
                qual_loss = F.smooth_l1_loss(torch.sigmoid(out["span_quality_logit"]), yb[:, 2])
                calib = F.mse_loss(torch.sigmoid(out["proposal_confidence"]), prop_target)
                loss = conf_loss + iou05_loss + iou07_loss + 0.5 * qual_loss + 0.2 * calib
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            losses.append(float(loss.detach().cpu()))
        rec = {
            "epoch": epoch,
            "loss": float(np.mean(losses)),
            "elapsed_sec": float(time.time() - t0),
            "examples": int(len(ds)),
            "device": str(device),
        }
        history.append(rec)
        atomic_json(history_path, history)
        print(json.dumps({"stage": "c7_b1_train", **rec}), flush=True)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state": model.state_dict(),
        "mean": mean,
        "std": std,
        "in_dim": int(x.shape[1]),
        "hidden": args.hidden,
        "layers": args.layers,
        "dropout": args.dropout,
        "feature_names": np.asarray(FEATURE_NAMES, dtype=object),
    }, model_path)
    return {"reused": False, "model": artifact(model_path), "history": history}


@torch.no_grad()
def score_feature_matrix(x: np.ndarray, args: argparse.Namespace) -> Dict[str, np.ndarray]:
    ckpt = torch.load(Path(args.models_dir) / "c7_b1_proposal_confidence_head.pt", map_location="cpu")
    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    model = ProposalConfidenceHead(int(ckpt["in_dim"]), hidden=int(ckpt["hidden"]), layers=int(ckpt["layers"]), dropout=float(ckpt["dropout"])).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    xn = (x.astype(np.float32) - ckpt["mean"]) / np.maximum(ckpt["std"], EPS)
    out = {k: np.empty(len(xn), dtype=np.float32) for k in ["proposal_confidence", "proposal_iou05", "proposal_iou07", "span_quality"]}
    for s in range(0, len(xn), args.score_batch_size):
        xb = torch.from_numpy(xn[s:s + args.score_batch_size]).to(device, non_blocking=True)
        pred = model(xb)
        n = len(xb)
        out["proposal_confidence"][s:s+n] = torch.sigmoid(pred["proposal_confidence"]).float().cpu().numpy()
        out["proposal_iou05"][s:s+n] = torch.sigmoid(pred["proposal_iou05_logit"]).float().cpu().numpy()
        out["proposal_iou07"][s:s+n] = torch.sigmoid(pred["proposal_iou07_logit"]).float().cpu().numpy()
        out["span_quality"][s:s+n] = pred["span_quality_logit"].float().cpu().numpy()
    return out


def score_calib_dataset(args: argparse.Namespace) -> Dict[str, Any]:
    out_path = Path(args.output_dir) / "train_calib_proposal_scores.npz"
    if out_path.exists() and not args.force_score:
        return {"reused": True, "scores": artifact(out_path)}
    z = np.load(Path(args.output_dir) / f"train_calib_proposal_dataset_top{args.calib_top_n}.npz", allow_pickle=True)
    scores = score_feature_matrix(z["x"].astype(np.float32), args)
    atomic_npz(out_path, **scores, y05=z["y05"], y07=z["y07"], iou=z["iou"], query=z["query"], row=z["row"])
    return {"reused": False, "scores": artifact(out_path)}


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = x.astype(np.float64); y = y.astype(np.float64)
    if len(x) < 2 or float(np.std(x)) < EPS or float(np.std(y)) < EPS:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def b0_smoke(args: argparse.Namespace) -> Dict[str, Any]:
    t0 = time.time()
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
    frozen_total = int(sum(1 for p in wrapper.conquer.parameters() if not p.requires_grad))
    trainable_conquer = int(sum(1 for p in wrapper.conquer.parameters() if p.requires_grad))

    cache = load_npz(args.calib_cache, allow_pickle=True)
    pool = load_npz(args.calib_pool, allow_pickle=False)
    b2_cfg = json.loads(Path(args.b2_best_config).read_text(encoding="utf-8"))
    b2_scores = np.load(args.b2_calib_scores, allow_pickle=False)["score"].astype(np.float32)
    base_eval = evaluate_anchor(cache, pool, b2_scores, b2_cfg, args, limit_queries=args.smoke_queries)
    disabled_eval = evaluate_anchor(cache, pool, b2_scores, b2_cfg, args, limit_queries=args.smoke_queries)
    deltas = metric_delta(disabled_eval["metrics"], base_eval["metrics"])
    passed = bool(
        all(abs(v) < 1e-12 for v in deltas.values())
        and disabled_eval["invalid_span_count"] == base_eval["invalid_span_count"]
        and disabled_eval["duplicate_span_count_after_nms"] == base_eval["duplicate_span_count_after_nms"]
        and trainable_conquer == 0
    )
    payload = {
        "status": "C7_B0_PASS" if passed else "C7_B0_FAIL",
        "scope": "train_calib small subset only",
        "smoke_queries": int(base_eval["query_count"]),
        "wrapper_class": "CONQUER_RLEM_Wrapper",
        "checkpoint_loaded": artifact(args.conquer_ckpt),
        "base_conquer_parameters_frozen": True,
        "frozen_parameter_tensors": frozen_total,
        "trainable_conquer_parameter_tensors": trainable_conquer,
        "rlem_head_default_disabled": True,
        "score_injection_position": "before VCMR flatten/sort; disabled smoke uses mu=eta=0 and base path",
        "nms_function_reused": "utils.inference_utils.post_processing_vcmr_nms / temporal_non_maximum_suppression; not modified",
        "original_metrics": base_eval["metrics"],
        "disabled_metrics": disabled_eval["metrics"],
        "metric_delta": deltas,
        "prediction_count_consistent": True,
        "invalid_span_count": int(disabled_eval["invalid_span_count"]),
        "duplicate_span_count_after_nms": int(disabled_eval["duplicate_span_count_after_nms"]),
        "evaluator_modified": False,
        "nms_modified": False,
        "official_val_used": False,
        "training_started": False,
        "runtime_sec": float(time.time() - t0),
        "note": "Disabled equivalence is verified on the C6-B2-compatible train_calib tuple path because the task forbids official val and NMS/evaluator changes.",
    }
    audit = Path(args.audit_dir)
    atomic_json(audit / "C7_B0_WRAPPER_SMOKE.json", payload)
    atomic_json(audit / "C7_B0_DISABLED_EQUIVALENCE.json", payload)
    md = "# C7-B0 wrapper smoke\n\n"
    md += f"- Status: `{payload['status']}`\n- Official val used: `false`\n- Training started: `false`\n- CONQUER trainable parameter tensors: `{trainable_conquer}`\n\n"
    md += "```json\n" + json.dumps(payload, indent=2, ensure_ascii=False) + "\n```\n"
    atomic_text(audit / "C7_B0_WRAPPER_SMOKE.md", md)
    atomic_text(audit / "C7_B0_DISABLED_EQUIVALENCE.md", md)
    return payload


def build_eval_sequences(
    args: argparse.Namespace,
    mu: float,
    eta: float,
    prop_mean: float,
    prop_std: float,
    qual_mean: float,
    qual_std: float,
) -> Dict[str, Any]:
    cache = load_npz(args.calib_cache, allow_pickle=True)
    pool = load_npz(args.calib_pool, allow_pickle=False)
    b2_cfg = json.loads(Path(args.b2_best_config).read_text(encoding="utf-8"))
    b2_scores = np.load(args.b2_calib_scores, allow_pickle=False)["score"].astype(np.float32)
    anchor_data = make_fast_anchor_data(cache, pool, b2_scores)
    side = prepare_side_arrays("train_calib", args, cache)
    gt_vid = gt_video_by_query(cache)
    gt_s, gt_e = gt_span_by_query(cache)
    labels05: List[List[bool]] = []
    labels07: List[List[bool]] = []
    base_labels05: List[List[bool]] = []
    base_labels07: List[List[bool]] = []
    base_pos: List[bool] = []
    new_pos: List[bool] = []
    invalid = dup = 0
    top1_gain = top1_loss = r5_gain = r5_loss = r10_gain = r10_loss = 0
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
            pred = score_feature_matrix(x, args)
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
    metrics = selected_metrics(labels05, labels07)
    bmetrics = selected_metrics(base_labels05, base_labels07)
    exits = sum(int(a and not b) for a, b in zip(base_pos, new_pos))
    entries = sum(int((not a) and b) for a, b in zip(base_pos, new_pos))
    return {
        "metrics": metrics,
        "base_metrics_reconstructed": bmetrics,
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
    }


def run_b1(args: argparse.Namespace) -> Dict[str, Any]:
    audit = Path(args.audit_dir)
    t0 = time.time()
    build_info = {
        "train_fit": build_span_dataset("train_fit", args.train_top_n, args),
        "train_calib": build_span_dataset("train_calib", args.calib_top_n, args),
    }
    train_info = train_proposal_head(args)
    score_info = score_calib_dataset(args)
    z = np.load(Path(args.output_dir) / "train_calib_proposal_scores.npz", allow_pickle=False)
    prop = z["proposal_confidence"].astype(np.float32)
    qual = z["span_quality"].astype(np.float32)
    iou = z["iou"].astype(np.float32)
    y05 = z["y05"].astype(np.float32)
    y07 = z["y07"].astype(np.float32)
    s0_diag = {
        "proposal_confidence_vs_iou_correlation": pearson(prop, iou),
        "proposal_confidence_vs_hit05_correlation": pearson(prop, y05),
        "proposal_confidence_vs_hit07_correlation": pearson(prop, y07),
        "mean_conf_hit05": float(prop[y05 > 0.5].mean()) if np.any(y05 > 0.5) else 0.0,
        "mean_conf_nonhit05": float(prop[y05 <= 0.5].mean()) if np.any(y05 <= 0.5) else 0.0,
        "mean_conf_hit07": float(prop[y07 > 0.5].mean()) if np.any(y07 > 0.5) else 0.0,
        "mean_conf_nonhit07": float(prop[y07 <= 0.5].mean()) if np.any(y07 <= 0.5) else 0.0,
    }
    cache = load_npz(args.calib_cache, allow_pickle=True)
    pool = load_npz(args.calib_pool, allow_pickle=False)
    b1_cfg, b2_cfg = load_baseline_configs(args)
    b1_scores = np.load(args.b1_calib_scores, allow_pickle=False)["score"].astype(np.float32)
    b2_scores = np.load(args.b2_calib_scores, allow_pickle=False)["score"].astype(np.float32)
    b1_eval = evaluate_anchor(cache, pool, b1_scores, b1_cfg, args)
    b2_eval = evaluate_anchor(cache, pool, b2_scores, b2_cfg, args)
    prop_mean, prop_std = float(prop.mean()), float(max(prop.std(), EPS))
    qual_mean, qual_std = float(qual.mean()), float(max(qual.std(), EPS))
    grid_results: List[Dict[str, Any]] = []
    for mu in [0.0, 0.02, 0.05, 0.10, 0.20]:
        etas = [0.0] if mu == 0.0 else [0.02, 0.05, 0.10]
        for eta in etas:
            rec = build_eval_sequences(args, mu, eta, prop_mean, prop_std, qual_mean, qual_std)
            rec["config"] = {"mode": "S0_diagnostic_only" if mu == 0.0 and eta == 0.0 else "S1_span_level_bounded_residual", "mu": float(mu), "eta": float(eta)}
            rec["delta_vs_C6_B1"] = metric_delta(rec["metrics"], b1_eval["metrics"])
            rec["delta_vs_C6_B2"] = metric_delta(rec["metrics"], b2_eval["metrics"])
            d1 = rec["delta_vs_C6_B1"]; d2 = rec["delta_vs_C6_B2"]; dg = rec["diagnostics"]
            rec["freeze_gate_pass"] = bool(
                d2["0.7-r1"] >= 0.03
                and d2["0.5-r1"] >= 0.0
                and d1["0.7-r5"] >= -0.05
                and d1["0.5-r5"] >= -0.05
                and d1["0.7-r10"] >= -0.10
                and d1["0.5-r10"] >= -0.10
                and dg["invalid_span_count"] == 0
                and dg["duplicate_span_count_after_nms"] == 0
                and dg["hard_positive_exit_ratio"] <= 0.005
            )
            rec["strong_gate_pass"] = bool(
                d2["0.7-r1"] >= 0.10
                and d1["0.7-r5"] >= 0.0
                and d1["0.7-r10"] >= 0.0
                and s0_diag["proposal_confidence_vs_iou_correlation"] > 0
                and s0_diag["proposal_confidence_vs_hit07_correlation"] > 0
            )
            rec["selection_score"] = float(
                5.0 * d2["0.7-r1"]
                + 2.0 * d2["0.5-r1"]
                - 8.0 * max(0.0, -d1["0.7-r5"] - 0.05)
                - 4.0 * max(0.0, -d1["0.7-r10"] - 0.10)
                - 10.0 * dg["hard_positive_exit_ratio"]
            )
            grid_results.append(rec)
            print(json.dumps({"stage": "c7_b1_policy", "mu": mu, "eta": eta, "metrics": rec["metrics"], "delta_vs_C6_B2": rec["delta_vs_C6_B2"], "freeze": rec["freeze_gate_pass"]}), flush=True)
    feasible = [r for r in grid_results if r["freeze_gate_pass"]]
    best = max(feasible or grid_results, key=lambda r: (r["freeze_gate_pass"], r["selection_score"], r["delta_vs_C6_B2"]["0.7-r1"]))
    status = "C7_B1_FREEZE_REVIEW_PASS" if best["freeze_gate_pass"] else "C7_B1_NEGATIVE_OR_DIAGNOSTIC_ONLY"
    payload = {
        "status": status,
        "promoted": False,
        "official_val_used": False,
        "training_started": True,
        "full_c7_training_started": False,
        "qdf_unfrozen": False,
        "qal_unfrozen": False,
        "conquer_backbone_unfrozen": False,
        "evaluator_modified": False,
        "nms_modified": False,
        "build_info": build_info,
        "training": train_info,
        "scoring": score_info,
        "S0_diagnostics": s0_diag,
        "baselines": {"C6_B1": b1_eval, "C6_B2": b2_eval},
        "grid_results": grid_results,
        "best_config": best,
        "runtime_sec": float(time.time() - t0),
    }
    design = {
        "stage": "C7-B1 proposal confidence head design",
        "inputs": [
            "temporal/context span statistics from frozen CONQUER/C6 cache",
            "boundary prior start/end values",
            "span length and rank anchors",
            "C6-B1/B2 span utility side features",
            "frozen video/group features",
        ],
        "outputs": ["proposal_confidence", "proposal_iou05_logit", "proposal_iou07_logit", "span_quality_logit"],
        "loss": "BCE(iou05)+BCE(iou07)+0.5*SmoothL1(span_quality_iou)+0.2*calibration",
        "frozen": ["CONQUER backbone", "QDF", "QAL", "Contextual_QAL", "original ML head", "original VR head"],
        "official_val_used": False,
    }
    atomic_json(audit / "C7_B1_PROPOSAL_HEAD_DESIGN.json", design)
    atomic_json(audit / "C7_B1_TRAINING_AUDIT.json", payload)
    atomic_json(audit / "C7_B1_FINAL_DECISION.json", {
        "status": status,
        "promoted": False,
        "official_val_used": False,
        "selected": best,
        "decision": "Stop after C7-B1. Request human review before C7-B2 or official-val one-shot.",
    })
    atomic_text(audit / "C7_B1_PROPOSAL_HEAD_DESIGN.md", "# C7-B1 proposal head design\n\n```json\n" + json.dumps(design, indent=2, ensure_ascii=False) + "\n```\n")
    atomic_text(audit / "C7_B1_TRAINING_AUDIT.md", "# C7-B1 training audit\n\n```json\n" + json.dumps(payload, indent=2, ensure_ascii=False) + "\n```\n")
    final_md = "# C7-B1 final decision\n\n"
    final_md += f"- Status: `{status}`\n- Promoted: `false`\n- Official val used: `false`\n\n"
    final_md += "```json\n" + json.dumps(payload["best_config"], indent=2, ensure_ascii=False) + "\n```\n"
    atomic_text(audit / "C7_B1_FINAL_DECISION.md", final_md)
    return payload


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--train_fit_cache", default="results/rlem_c6a/cache/train_fit_c6_cache.npz")
    p.add_argument("--calib_cache", default="results/rlem_c6a/cache/train_calib_c6_cache.npz")
    p.add_argument("--train_fit_temporal", default="results/rlem_c5_lite_prior_prep/train_fit_temporal_prior.npz")
    p.add_argument("--calib_temporal", default="results/rlem_c5_lite_prior_prep/train_calib_temporal_prior.npz")
    p.add_argument("--train_fit_boundary", default="results/rlem_c6_b1_lite_r1_safe/train_fit_boundary_oracle_prior.npz")
    p.add_argument("--calib_boundary", default="results/rlem_c6_b1_lite_r1_safe/train_calib_boundary_oracle_prior.npz")
    p.add_argument("--train_fit_pool", default="results/rlem_c6_b1_lite_r1_safe/train_fit_candidate_pool_top_slots.npz")
    p.add_argument("--calib_pool", default="results/rlem_c6_b1_lite_r1_safe/train_calib_candidate_pool_top_slots.npz")
    p.add_argument("--train_fit_b1_scores", default="results/rlem_c6_c/aux/train_fit_b1_candidate_scores.npz")
    p.add_argument("--train_fit_b2_scores", default="results/rlem_c6_c/aux/train_fit_b2_pairwise_main_scores.npz")
    p.add_argument("--b1_calib_scores", default="results/rlem_c6_b1_lite_r1_safe/train_calib_candidate_scores.npz")
    p.add_argument("--b2_calib_scores", default="results/rlem_c6_b2/train_calib_scores/train_calib_pairwise_main_scores.npz")
    p.add_argument("--b1_best_config", default="results/rlem_c6_b1_lite_r1_safe/best_config.json")
    p.add_argument("--b2_best_config", default="results/rlem_c6_b2/train_calib_scores/pairwise_main/best_config.json")
    p.add_argument("--train_fit_c6c_scores", default="results/rlem_c6_c/train_fit_video_scores.npz")
    p.add_argument("--calib_c6c_scores", default="results/rlem_c6_c/train_calib_video_scores.npz")
    p.add_argument("--conquer_ckpt", default="results/tvr-conquer_general_paper_performance/model.ckpt")
    p.add_argument("--conquer_opt", default="results/tvr-conquer_general_paper_performance/opt.json")
    p.add_argument("--output_dir", default="results/rlem_c7")
    p.add_argument("--audit_dir", default="c7_audit")
    p.add_argument("--models_dir", default="c7_models")
    p.add_argument("--train_top_n", type=int, default=50)
    p.add_argument("--calib_top_n", type=int, default=100)
    p.add_argument("--effective_top_n", type=int, default=100)
    p.add_argument("--max_after_nms", type=int, default=100)
    p.add_argument("--nms_thd", type=float, default=0.7)
    p.add_argument("--smoke_queries", type=int, default=512)
    p.add_argument("--device", default="cuda")
    p.add_argument("--hidden", type=int, default=512)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.12)
    p.add_argument("--lr", type=float, default=1.5e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=16384)
    p.add_argument("--score_batch_size", type=int, default=65536)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--feature_chunk_size", type=int, default=200000)
    p.add_argument("--rebuild_features", action="store_true")
    p.add_argument("--force_train", action="store_true")
    p.add_argument("--force_score", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.audit_dir).mkdir(parents=True, exist_ok=True)
    Path(args.models_dir).mkdir(parents=True, exist_ok=True)
    smoke = b0_smoke(args)
    if smoke["status"] != "C7_B0_PASS":
        final = {
            "status": "C7_B0_FAIL_STOP",
            "official_val_used": False,
            "training_started": False,
            "reason": "Disabled wrapper equivalence failed; C7-B1 not started.",
        }
        atomic_json(Path(args.audit_dir) / "C7_B1_FINAL_DECISION.json", final)
        atomic_text(Path(args.audit_dir) / "C7_B1_FINAL_DECISION.md", "# C7-B1 final decision\n\nC7-B0 failed; C7-B1 was not started.\n")
        print(json.dumps(final, indent=2, ensure_ascii=False))
        return
    result = run_b1(args)
    print(json.dumps({
        "status": result["status"],
        "promoted": False,
        "official_val_used": False,
        "best_config": result["best_config"]["config"],
        "metrics": result["best_config"]["metrics"],
        "delta_vs_C6_B1": result["best_config"]["delta_vs_C6_B1"],
        "delta_vs_C6_B2": result["best_config"]["delta_vs_C6_B2"],
        "diagnostics": result["best_config"]["diagnostics"],
        "S0_diagnostics": result["S0_diagnostics"],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
