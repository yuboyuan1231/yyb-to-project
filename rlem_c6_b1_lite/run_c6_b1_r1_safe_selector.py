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
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from rlem_c6.c6a_utils import write_json, write_text  # noqa: E402
from rlem_c6_b0.run_c6_b0_candidate_diagnostic import (  # noqa: E402
    artifact,
    correctness_for_ts,
    infer_boundary_prior,
    load_gt_ts,
    span_iou_idx,
)


METRIC_KEYS = [
    "0.5-r1", "0.5-r5", "0.5-r10", "0.5-r100",
    "0.7-r1", "0.7-r5", "0.7-r10", "0.7-r100",
]
CLIP = 1.5
EPS = 1e-12


class CandidateSelector(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 384, layers: int = 4, dropout: float = 0.10):
        super().__init__()
        mods: List[nn.Module] = []
        d = in_dim
        for _ in range(layers):
            mods += [nn.Linear(d, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(dropout)]
            d = hidden
        mods.append(nn.Linear(d, 3))
        self.net = nn.Sequential(*mods)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_cache(path: str) -> Dict[str, np.ndarray]:
    return {k: v for k, v in np.load(path, allow_pickle=True).items()}


def normalize_distribution(x: np.ndarray) -> np.ndarray:
    y = np.maximum(np.asarray(x, dtype=np.float64), EPS)
    return (y / max(float(y.sum()), EPS)).astype(np.float64)


def temporal_nms(starts: np.ndarray, ends: np.ndarray, scores: np.ndarray, thd: float, max_keep: int) -> List[int]:
    order = np.argsort(-scores, kind="stable")
    kept: List[int] = []
    for idx in order:
        si, ei = int(starts[idx]), int(ends[idx])
        bad = False
        for prior in kept:
            if span_iou_idx(si, ei, int(starts[prior]), int(ends[prior])) > thd:
                bad = True
                break
        if not bad:
            kept.append(int(idx))
            if len(kept) >= max_keep:
                break
    return kept


def boundary_alternatives(
    pb: np.ndarray,
    pe: np.ndarray,
    t: int,
    orig_s: int,
    orig_e: int,
    alt_count: int,
    top_endpoint: int,
    nms_thd: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    pb = normalize_distribution(pb[:t])
    pe = normalize_distribution(pe[:t])
    m = min(t, max(top_endpoint, int(math.ceil(math.sqrt(alt_count * 32)))))
    top_s = np.argsort(-pb, kind="stable")[:m]
    top_e = np.argsort(-pe, kind="stable")[:m]
    ss, ee = np.meshgrid(top_s, top_e, indexing="ij")
    ss = ss.reshape(-1)
    ee = ee.reshape(-1)
    valid = ss <= ee
    ss = ss[valid].astype(np.int16)
    ee = ee[valid].astype(np.int16)
    sc = (np.log(pb[ss]) + np.log(pe[ee])).astype(np.float32)
    if len(ss) < alt_count - 1:
        all_s, all_e = np.triu_indices(t)
        ss = np.concatenate([ss, all_s.astype(np.int16)])
        ee = np.concatenate([ee, all_e.astype(np.int16)])
        sc = np.concatenate([sc, (np.log(pb[all_s]) + np.log(pe[all_e])).astype(np.float32)])
    keys = ss.astype(np.int32) * 1024 + ee.astype(np.int32)
    uniq, first = np.unique(keys, return_index=True)
    ss, ee, sc = ss[first], ee[first], sc[first]
    keep = temporal_nms(ss, ee, sc, nms_thd, max(alt_count * 2, alt_count))
    # Exclude exact original from the boundary alternatives; original is always alt 0.
    filtered = [i for i in keep if not (int(ss[i]) == int(orig_s) and int(ee[i]) == int(orig_e))]
    if len(filtered) < alt_count - 1:
        more = [int(i) for i in np.argsort(-sc, kind="stable") if int(i) not in set(filtered) and not (int(ss[i]) == int(orig_s) and int(ee[i]) == int(orig_e))]
        filtered += more[:alt_count - 1 - len(filtered)]
    filtered = filtered[:alt_count - 1]
    starts = [int(orig_s)] + [int(ss[i]) for i in filtered]
    ends = [int(orig_e)] + [int(ee[i]) for i in filtered]
    scores = [float(np.log(pb[max(0, min(t - 1, orig_s))]) + np.log(pe[max(0, min(t - 1, orig_e))]))] + [float(sc[i]) for i in filtered]
    while len(starts) < alt_count:
        starts.append(int(orig_s)); ends.append(int(orig_e)); scores.append(scores[0])
    return np.asarray(starts, dtype=np.int16), np.asarray(ends, dtype=np.int16), np.asarray(scores, dtype=np.float32)


def make_feature(
    *,
    si: int,
    ei: int,
    orig_s: int,
    orig_e: int,
    endpoint_score: float,
    orig_endpoint_score: float,
    pb: np.ndarray,
    pe: np.ndarray,
    pc: np.ndarray,
    t: int,
    slot_rank: int,
    s_c4: float,
    q_top_score: float,
    q_mean_score: float,
    q_std_score: float,
    video_feat: np.ndarray,
    is_original: bool,
) -> np.ndarray:
    si = int(max(0, min(t - 1, si)))
    ei = int(max(si, min(t - 1, ei)))
    span = slice(si, ei + 1)
    length = ei - si + 1
    center = 0.5 * (si + ei)
    base = [
        float(slot_rank) / 100.0,
        float(slot_rank == 0),
        float(slot_rank < 5),
        float(is_original),
        float(not is_original),
        si / 100.0,
        ei / 100.0,
        length / 100.0,
        center / 100.0,
        (si - orig_s) / 100.0,
        (ei - orig_e) / 100.0,
        span_iou_idx(si, ei, orig_s, orig_e),
        float(endpoint_score),
        float(endpoint_score - orig_endpoint_score),
        float(pb[si]),
        float(pe[ei]),
        float(np.log(max(float(pb[si]), EPS))),
        float(np.log(max(float(pe[ei]), EPS))),
        float(pc[span].mean()) if length > 0 else 0.0,
        float(pc[span].sum()) if length > 0 else 0.0,
        float(s_c4),
        float((s_c4 - q_mean_score) / max(q_std_score, 1e-6)),
        float(s_c4 - q_top_score),
        1.0 / float(slot_rank + 1),
    ]
    return np.concatenate([np.asarray(base, dtype=np.float32), video_feat.astype(np.float32)], axis=0)


def build_pool(
    *,
    cache: Dict[str, np.ndarray],
    temporal_npz: str,
    prior_npz: str,
    gt_ts_by_query: Dict[int, Any] | None,
    slots: int,
    alt_count: int,
    top_endpoint: int,
    nms_thd: float,
    max_queries: int | None = None,
) -> Dict[str, np.ndarray]:
    with np.load(temporal_npz, allow_pickle=False) as temporal:
        p_ctx = temporal["p_ctx"].astype(np.float32)
        temporal_length = temporal["temporal_length"].astype(np.int64)
    with np.load(prior_npz, allow_pickle=False) as prior:
        p_b = prior["p_b_new"].astype(np.float32)
        p_e = prior["p_e_new"].astype(np.float32)

    q_count = len(cache["desc_ids"]) if max_queries is None else min(max_queries, len(cache["desc_ids"]))
    rows_per_q = int(cache["desc_offsets"][1] - cache["desc_offsets"][0])
    row_group = cache["row_group_id"].astype(np.int64)
    video_idx = cache["video_idx"].astype(np.int64)
    start_idx = cache["start_idx"].astype(np.int64)
    end_idx = cache["end_idx"].astype(np.int64)
    s_c4 = cache["s_c4_final"].astype(np.float32)
    label_relevant = cache["label_relevant"].astype(np.float32)
    gt_s = cache["gt_start_idx"].astype(np.int64)
    gt_e = cache["gt_end_idx"].astype(np.int64)
    video_feat = cache["video_features"].astype(np.float32)
    total = q_count * slots * alt_count
    # 24 engineered features + 27 video features.
    feat_dim = 51
    x = np.zeros((total, feat_dim), dtype=np.float32)
    y05 = np.zeros(total, dtype=np.float32)
    y07 = np.zeros(total, dtype=np.float32)
    yiou = np.zeros(total, dtype=np.float32)
    q_arr = np.zeros(total, dtype=np.int32)
    slot_arr = np.zeros(total, dtype=np.int16)
    alt_arr = np.zeros(total, dtype=np.int16)
    gid_arr = np.zeros(total, dtype=np.int32)
    vid_arr = np.zeros(total, dtype=np.int32)
    st_arr = np.zeros(total, dtype=np.int16)
    en_arr = np.zeros(total, dtype=np.int16)
    orig_st_arr = np.zeros(total, dtype=np.int16)
    orig_en_arr = np.zeros(total, dtype=np.int16)
    is_orig_arr = np.zeros(total, dtype=np.bool_)
    base_score_arr = np.zeros(total, dtype=np.float32)

    pos = 0
    for q in range(q_count):
        rs = np.arange(q * rows_per_q, (q + 1) * rows_per_q, dtype=np.int64)
        scores = s_c4[rs]
        order = rs[np.argsort(-scores, kind="stable")[:slots]]
        q_top = float(np.max(scores))
        q_mean = float(np.mean(scores))
        q_std = float(np.std(scores))
        for slot_rank, row in enumerate(order):
            gid = int(row_group[row])
            t = int(temporal_length[gid])
            pb = normalize_distribution(p_b[gid, :t]).astype(np.float32)
            pe = normalize_distribution(p_e[gid, :t]).astype(np.float32)
            pc = normalize_distribution(p_ctx[gid, :t]).astype(np.float32)
            os_, oe_ = int(start_idx[row]), int(end_idx[row])
            starts, ends, endpoint_scores = boundary_alternatives(
                pb, pe, t, os_, oe_, alt_count, top_endpoint, nms_thd
            )
            orig_endpoint = float(endpoint_scores[0])
            for a in range(alt_count):
                si, ei = int(starts[a]), int(ends[a])
                x[pos] = make_feature(
                    si=si, ei=ei, orig_s=os_, orig_e=oe_,
                    endpoint_score=float(endpoint_scores[a]),
                    orig_endpoint_score=orig_endpoint,
                    pb=pb, pe=pe, pc=pc, t=t,
                    slot_rank=slot_rank,
                    s_c4=float(s_c4[row]),
                    q_top_score=q_top, q_mean_score=q_mean, q_std_score=q_std,
                    video_feat=video_feat[gid],
                    is_original=(a == 0),
                )
                rel = bool(label_relevant[gid] > 0.5)
                if rel:
                    if gt_ts_by_query is not None:
                        iou = 1.0 if correctness_for_ts(si, ei, gt_ts_by_query[q], 0.5) else 0.0
                        iou7 = 1.0 if correctness_for_ts(si, ei, gt_ts_by_query[q], 0.7) else 0.0
                        # Quality target still uses index IoU as smooth proxy.
                        q_iou = span_iou_idx(si, ei, int(gt_s[row]), int(gt_e[row]))
                    else:
                        q_iou = span_iou_idx(si, ei, int(gt_s[row]), int(gt_e[row]))
                        iou = float(q_iou >= 0.5)
                        iou7 = float(q_iou >= 0.7)
                    y05[pos] = float(iou)
                    y07[pos] = float(iou7)
                    yiou[pos] = float(q_iou)
                q_arr[pos] = q
                slot_arr[pos] = slot_rank
                alt_arr[pos] = a
                gid_arr[pos] = gid
                vid_arr[pos] = int(video_idx[row])
                st_arr[pos] = si
                en_arr[pos] = ei
                orig_st_arr[pos] = os_
                orig_en_arr[pos] = oe_
                is_orig_arr[pos] = (a == 0)
                base_score_arr[pos] = float(s_c4[row])
                pos += 1
        if (q + 1) % 5000 == 0:
            print(json.dumps({"stage": "build_pool", "queries": q + 1, "total": q_count}))
    return {
        "x": x[:pos],
        "y05": y05[:pos],
        "y07": y07[:pos],
        "yiou": yiou[:pos],
        "query": q_arr[:pos],
        "slot": slot_arr[:pos],
        "alt": alt_arr[:pos],
        "group_id": gid_arr[:pos],
        "video_idx": vid_arr[:pos],
        "start_idx": st_arr[:pos],
        "end_idx": en_arr[:pos],
        "orig_start_idx": orig_st_arr[:pos],
        "orig_end_idx": orig_en_arr[:pos],
        "is_original": is_orig_arr[:pos],
        "base_score": base_score_arr[:pos],
        "shape": np.asarray([q_count, slots, alt_count], dtype=np.int64),
    }


def save_npz(path: str | Path, **arrays) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(p) + ".partial")
    with open(tmp, "wb") as f:
        np.savez_compressed(f, **arrays)
    os.replace(tmp, p)


def fit_selector(
    train_pool: Dict[str, np.ndarray],
    out_dir: Path,
    device_name: str,
    epochs: int,
    batch_size: int,
    hidden: int,
    layers: int,
) -> Dict[str, Any]:
    x = train_pool["x"].astype(np.float32)
    mean = x.mean(axis=0).astype(np.float32)
    std = np.maximum(x.std(axis=0), 1e-6).astype(np.float32)
    xz = ((x - mean) / std).astype(np.float32)
    y05 = train_pool["y05"].astype(np.float32)
    y07 = train_pool["y07"].astype(np.float32)
    yiou = train_pool["yiou"].astype(np.float32)
    # Top slot gets strongest weight; top5 still matters for safety.
    slot = train_pool["slot"].astype(np.int64)
    weight = np.where(slot == 0, 3.0, np.where(slot < 5, 1.5, 0.75)).astype(np.float32)
    pos_weight05 = float((len(y05) - y05.sum()) / max(float(y05.sum()), 1.0))
    pos_weight07 = float((len(y07) - y07.sum()) / max(float(y07.sum()), 1.0))
    device = torch.device("cuda" if device_name == "cuda" and torch.cuda.is_available() else "cpu")
    ds = TensorDataset(
        torch.from_numpy(xz),
        torch.from_numpy(y05),
        torch.from_numpy(y07),
        torch.from_numpy(yiou),
        torch.from_numpy(weight),
    )
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=(device.type == "cuda"))
    model = CandidateSelector(x.shape[1], hidden=hidden, layers=layers).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        t0 = time.time()
        for xb, a05, a07, aiou, ww in dl:
            xb = xb.to(device, non_blocking=True)
            a05 = a05.to(device, non_blocking=True)
            a07 = a07.to(device, non_blocking=True)
            aiou = aiou.to(device, non_blocking=True)
            ww = ww.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                out = model(xb)
                l05 = F.binary_cross_entropy_with_logits(out[:, 0], a05, reduction="none", pos_weight=torch.tensor(pos_weight05, device=device))
                l07 = F.binary_cross_entropy_with_logits(out[:, 1], a07, reduction="none", pos_weight=torch.tensor(pos_weight07, device=device))
                liou = F.smooth_l1_loss(torch.sigmoid(out[:, 2]), aiou, reduction="none")
                loss = (ww * (1.5 * l05 + 0.7 * l07 + 0.5 * liou)).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))
        rec = {"epoch": epoch, "loss": float(np.mean(losses)), "elapsed_sec": float(time.time() - t0)}
        history.append(rec)
        print(json.dumps({"stage": "train_selector", **rec}))
        write_json(out_dir / "history.json", history)
    ckpt = {
        "model_state": model.state_dict(),
        "feature_mean": mean,
        "feature_std": std,
        "in_dim": int(x.shape[1]),
        "hidden": int(hidden),
        "layers": int(layers),
        "pos_weight05": pos_weight05,
        "pos_weight07": pos_weight07,
        "epochs": int(epochs),
    }
    torch.save(ckpt, out_dir / "model_best.pt")
    return {
        "history": history,
        "model_artifact": artifact(out_dir / "model_best.pt"),
        "pos_weight05": pos_weight05,
        "pos_weight07": pos_weight07,
        "train_examples": int(len(x)),
        "feature_dim": int(x.shape[1]),
        "device": str(device),
    }


@torch.no_grad()
def score_pool(pool: Dict[str, np.ndarray], model_path: str, device_name: str, batch_size: int) -> np.ndarray:
    ckpt = torch.load(model_path, map_location="cpu")
    model = CandidateSelector(int(ckpt["in_dim"]), hidden=int(ckpt["hidden"]), layers=int(ckpt["layers"]))
    model.load_state_dict(ckpt["model_state"])
    device = torch.device("cuda" if device_name == "cuda" and torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    x = ((pool["x"].astype(np.float32) - ckpt["feature_mean"]) / ckpt["feature_std"]).astype(np.float32)
    scores = np.empty(len(x), dtype=np.float32)
    for start in range(0, len(x), batch_size):
        xb = torch.from_numpy(x[start:start + batch_size]).to(device)
        out = model(xb).float()
        # R1-biased proposal confidence with IoU0.7 as precision boost.
        sc = out[:, 0] + 0.45 * out[:, 1] + 0.25 * torch.sigmoid(out[:, 2])
        scores[start:start + len(sc)] = sc.detach().cpu().numpy().astype(np.float32)
    return scores


def selected_metrics(labels: Dict[str, List[List[bool]]]) -> Dict[str, float]:
    q_count = len(labels["0.5"])
    out = {k: 0 for k in METRIC_KEYS}
    for q in range(q_count):
        for thr in ["0.5", "0.7"]:
            labs = labels[thr][q]
            for k in [1, 5, 10, 100]:
                out[f"{thr}-r{k}"] += int(any(labs[:k]))
    return {k: 100.0 * v / q_count for k, v in out.items()}


def evaluate_policy(
    *,
    cache: Dict[str, np.ndarray],
    pool: Dict[str, np.ndarray],
    pool_score: np.ndarray,
    gt_ts_by_query: Dict[int, Any],
    apply_slots: int,
    threshold0: float,
    threshold_rest: float,
    max_replacements: int,
    effective_top_n: int,
    max_after_nms: int,
    nms_thd: float,
) -> Dict[str, Any]:
    q_count, slots, alt_count = [int(x) for x in pool["shape"]]
    rows_per_q = int(cache["desc_offsets"][1] - cache["desc_offsets"][0])
    video_idx = cache["video_idx"].astype(np.int64)
    start_idx = cache["start_idx"].astype(np.int64)
    end_idx = cache["end_idx"].astype(np.int64)
    s_c4 = cache["s_c4_final"].astype(np.float32)
    label_relevant = cache["label_relevant"].astype(np.float32)
    video_keys = cache["video_group_keys"].astype(np.int64)
    gt_video_by_q = np.full(q_count, -1, dtype=np.int64)
    for gid, rel in enumerate(label_relevant):
        if rel > 0.5:
            q = int(video_keys[gid, 0])
            if q < q_count and gt_video_by_q[q] < 0:
                gt_video_by_q[q] = int(video_keys[gid, 1])
    labels = {"0.5": [], "0.7": []}
    replacements = 0
    top1_changed = 0
    hard_exits = 0
    hard_entries = 0
    orig_positive = []
    cand_positive = []
    for q in range(q_count):
        rs = np.arange(q * rows_per_q, (q + 1) * rows_per_q, dtype=np.int64)
        base_order = rs[np.argsort(-s_c4[rs], kind="stable")[:effective_top_n]]
        slot_candidates = []
        replaced_this_q = 0
        for slot_rank, row in enumerate(base_order):
            vid = int(video_idx[row])
            si, ei = int(start_idx[row]), int(end_idx[row])
            if slot_rank < min(slots, apply_slots) and replaced_this_q < max_replacements:
                base = (q * slots + slot_rank) * alt_count
                scores = pool_score[base:base + alt_count]
                best = int(np.argmax(scores))
                margin = float(scores[best] - scores[0])
                thd = threshold0 if slot_rank == 0 else threshold_rest
                if best != 0 and margin > thd:
                    si = int(pool["start_idx"][base + best])
                    ei = int(pool["end_idx"][base + best])
                    replacements += 1
                    replaced_this_q += 1
            ts = gt_ts_by_query[q]
            is_gt = vid == int(gt_video_by_q[q])
            y05 = bool(is_gt and correctness_for_ts(si, ei, ts, 0.5))
            y07 = bool(is_gt and correctness_for_ts(si, ei, ts, 0.7))
            slot_candidates.append((vid, si, ei, float(s_c4[row]), y05, y07, int(start_idx[row]), int(end_idx[row])))
        kept: List[int] = []
        by_video: Dict[int, List[int]] = {}
        for idx, (vid, si, ei, score, *_rest) in enumerate(slot_candidates):
            suppress = False
            for prior in by_video.get(vid, []):
                _, psi, pei, *_ = slot_candidates[prior]
                if span_iou_idx(si, ei, psi, pei) > nms_thd:
                    suppress = True
                    break
            if not suppress:
                kept.append(idx)
                by_video.setdefault(vid, []).append(idx)
                if len(kept) >= max_after_nms:
                    break
        labs05 = [slot_candidates[i][4] for i in kept]
        labs07 = [slot_candidates[i][5] for i in kept]
        labels["0.5"].append(labs05)
        labels["0.7"].append(labs07)
        cand_positive.append(bool(any(labs05[:100]) or any(labs07[:100])))
        # Original positive/top1 signatures for movement diagnostics.
        orig_slots = []
        for row in base_order:
            vid = int(video_idx[row]); si = int(start_idx[row]); ei = int(end_idx[row])
            ts = gt_ts_by_query[q]
            is_gt = vid == int(gt_video_by_q[q])
            orig_slots.append((vid, si, ei, bool(is_gt and correctness_for_ts(si, ei, ts, 0.5)), bool(is_gt and correctness_for_ts(si, ei, ts, 0.7))))
        orig_kept: List[int] = []
        orig_by_video: Dict[int, List[int]] = {}
        for idx, (vid, si, ei, *_labels) in enumerate(orig_slots):
            suppress = False
            for prior in orig_by_video.get(vid, []):
                _, psi, pei, *_ = orig_slots[prior]
                if span_iou_idx(si, ei, psi, pei) > nms_thd:
                    suppress = True
                    break
            if not suppress:
                orig_kept.append(idx)
                orig_by_video.setdefault(vid, []).append(idx)
                if len(orig_kept) >= max_after_nms:
                    break
        orig_positive.append(bool(any(orig_slots[i][3] or orig_slots[i][4] for i in orig_kept[:100])))
        orig_top1 = orig_slots[orig_kept[0]] if orig_kept else (-1, -1, -1, False, False)
        cand_top1 = slot_candidates[kept[0]] if kept else (-1, -1, -1, 0.0, False, False, -1, -1)
        top1_changed += int((cand_top1[0], cand_top1[1], cand_top1[2]) != (orig_top1[0], orig_top1[1], orig_top1[2]))
    metrics = selected_metrics(labels)
    hard_exits = sum(int(a and not b) for a, b in zip(orig_positive, cand_positive))
    hard_entries = sum(int((not a) and b) for a, b in zip(orig_positive, cand_positive))
    return {
        "metrics": metrics,
        "replacement_rate_top_slots": float(replacements / max(q_count * min(slots, apply_slots), 1)),
        "replacements": int(replacements),
        "top1_changed_ratio": float(top1_changed / q_count),
        "hard_positive_top100_query_exits": int(hard_exits),
        "hard_positive_top100_query_entries": int(hard_entries),
        "video_slot_drift": 0.0,
        "video_multiset_drift": 0.0,
    }


def metric_delta(new: Dict[str, float], base: Dict[str, float]) -> Dict[str, float]:
    return {k: float(new[k] - base[k]) for k in METRIC_KEYS}


def selection_score(delta: Dict[str, float]) -> float:
    # User preference: R1 primary, R5 secondary, deeper metrics small regularizer.
    return float(
        1.0 * (delta["0.5-r1"] + delta["0.7-r1"]) / 2.0
        + 0.35 * (delta["0.5-r5"] + delta["0.7-r5"]) / 2.0
        + 0.10 * (delta["0.5-r10"] + delta["0.7-r10"]) / 2.0
    )


def grid_search_policy(cache: Dict[str, np.ndarray], pool: Dict[str, np.ndarray], scores: np.ndarray, gt_ts: Dict[int, Any], args: argparse.Namespace) -> Dict[str, Any]:
    baseline = evaluate_policy(
        cache=cache, pool=pool, pool_score=scores, gt_ts_by_query=gt_ts,
        apply_slots=0, threshold0=999.0, threshold_rest=999.0, max_replacements=0,
        effective_top_n=args.effective_top_n, max_after_nms=args.max_after_nms, nms_thd=args.nms_thd,
    )
    results = []
    # R1-focused narrow grid.  The first wide version was unnecessarily slow
    # because every config reconstructs top100 + NMS in Python.  B0 already
    # showed that broad replacement is unsafe, so this grid deliberately keeps
    # the search around top1/top3 conservative replacement policies.
    grid = [
        (1, t0, 999.0, 1)
        for t0 in [-0.25, 0.0, 0.1, 0.2, 0.4, 0.7, 1.0, 1.5]
    ]
    grid += [
        (3, t0, tr, mr)
        for t0 in [-0.25, 0.0, 0.1, 0.2, 0.4, 0.7, 1.0]
        for tr in [0.2, 0.5, 0.8, 1.2, 1.8]
        for mr in [1, 2]
    ]
    for apply_slots, threshold0, threshold_rest, max_rep in grid:
        ev = evaluate_policy(
            cache=cache, pool=pool, pool_score=scores, gt_ts_by_query=gt_ts,
            apply_slots=apply_slots, threshold0=threshold0, threshold_rest=threshold_rest,
            max_replacements=max_rep,
            effective_top_n=args.effective_top_n, max_after_nms=args.max_after_nms, nms_thd=args.nms_thd,
        )
        delta = metric_delta(ev["metrics"], baseline["metrics"])
        r5_safe = delta["0.5-r5"] >= -0.25 and delta["0.7-r5"] >= -0.25
        r1_positive = delta["0.5-r1"] > 0 or delta["0.7-r1"] > 0
        rec = {
            "config_id": f"c6b1r1_{len(results):04d}",
            "apply_slots": apply_slots,
            "threshold0": threshold0,
            "threshold_rest": threshold_rest,
            "max_replacements": max_rep,
            "metrics": ev["metrics"],
            "delta_vs_pseudo_C4_final": delta,
            "selection_score": selection_score(delta),
            "r1_positive": bool(r1_positive),
            "r5_safe": bool(r5_safe),
            "feasible": bool(r1_positive and r5_safe and ev["top1_changed_ratio"] <= 0.08),
            "movement": {k: ev[k] for k in ["replacement_rate_top_slots", "replacements", "top1_changed_ratio", "hard_positive_top100_query_exits", "hard_positive_top100_query_entries", "video_slot_drift", "video_multiset_drift"]},
        }
        results.append(rec)
    feasible = [r for r in results if r["feasible"]]
    if feasible:
        best = max(feasible, key=lambda r: (r["selection_score"], r["delta_vs_pseudo_C4_final"]["0.5-r1"] + r["delta_vs_pseudo_C4_final"]["0.7-r1"], -r["movement"]["replacement_rate_top_slots"]))
        status = "C6_B1_LITE_R1_SAFE_POSITIVE"
    else:
        best = max(results, key=lambda r: (r["selection_score"], r["r5_safe"], -r["movement"]["replacement_rate_top_slots"]))
        status = "C6_B1_LITE_R1_SAFE_NEGATIVE"
    return {"status": status, "baseline": baseline, "grid_results": results, "best_config": best}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--train_cache", default="results/rlem_c6a/cache/train_fit_c6_cache.npz")
    p.add_argument("--train_temporal", default="results/rlem_c5_lite_prior_prep/train_fit_temporal_prior.npz")
    p.add_argument("--calib_cache", default="results/rlem_c6a/cache/train_calib_c6_cache.npz")
    p.add_argument("--calib_temporal", default="results/rlem_c5_lite_prior_prep/train_calib_temporal_prior.npz")
    p.add_argument("--boundary_oracle_ckpt", default="results/rlem_c6a_repair/boundary_oracle/model_best.pt")
    p.add_argument("--calib_gt_jsonl", default="results/rlem_c3_minimal/train_calib_gt.jsonl")
    p.add_argument("--output_dir", default="results/rlem_c6_b1_lite_r1_safe")
    p.add_argument("--audit_dir", default="c6_b1_lite_audit")
    p.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    p.add_argument("--batch_groups", type=int, default=512)
    p.add_argument("--slots", type=int, default=5)
    p.add_argument("--alt_count", type=int, default=8)
    p.add_argument("--top_endpoint", type=int, default=24)
    p.add_argument("--effective_top_n", type=int, default=100)
    p.add_argument("--max_after_nms", type=int, default=100)
    p.add_argument("--nms_thd", type=float, default=0.7)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=32768)
    p.add_argument("--hidden", type=int, default=384)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--max_train_queries", type=int, default=None)
    args = p.parse_args()
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    audit_dir = Path(args.audit_dir); audit_dir.mkdir(parents=True, exist_ok=True)
    start_state = {
        "stage": "C6-B1-lite R1-safe selector",
        "scope": "train_fit/train_calib only",
        "official_val_used": False,
        "post_val_adjustment": False,
        "C6_C_used": False,
        "C4_final_retained": True,
        "primary_metric_focus": "R@1",
        "secondary_safety_focus": "R@5",
        "design": "medium MLP candidate selector + conservative top-slot replacement gate",
        "model": {"hidden": args.hidden, "layers": args.layers, "dropout": 0.10},
    }
    write_json(audit_dir / "C6_B1_LITE_R1_SAFE_START_STATE.json", start_state)
    write_text(audit_dir / "C6_B1_LITE_R1_SAFE_START_STATE.md", "# C6-B1-lite R1-safe selector start\n\n```json\n" + json.dumps(start_state, indent=2, ensure_ascii=False) + "\n```\n")

    train_prior = out_dir / "train_fit_boundary_oracle_prior.npz"
    calib_prior = out_dir / "train_calib_boundary_oracle_prior.npz"
    infer_train = infer_boundary_prior(cache_npz=args.train_cache, temporal_npz=args.train_temporal, ckpt_path=args.boundary_oracle_ckpt, output_npz=str(train_prior), device_name=args.device, batch_groups=args.batch_groups)
    infer_calib = infer_boundary_prior(cache_npz=args.calib_cache, temporal_npz=args.calib_temporal, ckpt_path=args.boundary_oracle_ckpt, output_npz=str(calib_prior), device_name=args.device, batch_groups=args.batch_groups)

    train_cache = load_cache(args.train_cache)
    calib_cache = load_cache(args.calib_cache)
    gt_calib = load_gt_ts(args.calib_gt_jsonl, calib_cache["desc_ids"])
    t0 = time.time()
    train_pool_path = out_dir / "train_fit_candidate_pool_top_slots.npz"
    if train_pool_path.exists():
        train_pool = load_cache(str(train_pool_path))
    else:
        train_pool = build_pool(cache=train_cache, temporal_npz=args.train_temporal, prior_npz=str(train_prior), gt_ts_by_query=None, slots=args.slots, alt_count=args.alt_count, top_endpoint=args.top_endpoint, nms_thd=args.nms_thd, max_queries=args.max_train_queries)
        save_npz(train_pool_path, **train_pool)
    calib_pool_path = out_dir / "train_calib_candidate_pool_top_slots.npz"
    if calib_pool_path.exists():
        calib_pool = load_cache(str(calib_pool_path))
    else:
        calib_pool = build_pool(cache=calib_cache, temporal_npz=args.calib_temporal, prior_npz=str(calib_prior), gt_ts_by_query=gt_calib, slots=args.slots, alt_count=args.alt_count, top_endpoint=args.top_endpoint, nms_thd=args.nms_thd)
        save_npz(calib_pool_path, **calib_pool)
    pool_audit = {"elapsed_sec": float(time.time() - t0), "train_examples": int(len(train_pool["x"])), "calib_examples": int(len(calib_pool["x"])), "slots": args.slots, "alt_count": args.alt_count}

    if (out_dir / "model_best.pt").exists() and (out_dir / "history.json").exists():
        train_audit = {
            "reused": True,
            "history": json.loads((out_dir / "history.json").read_text(encoding="utf-8")),
            "model_artifact": artifact(out_dir / "model_best.pt"),
            "train_examples": int(len(train_pool["x"])),
            "feature_dim": int(train_pool["x"].shape[1]),
            "device": args.device,
        }
    else:
        train_audit = fit_selector(train_pool, out_dir, args.device, args.epochs, args.batch_size, args.hidden, args.layers)
    calib_scores = score_pool(calib_pool, str(out_dir / "model_best.pt"), args.device, args.batch_size)
    save_npz(out_dir / "train_calib_candidate_scores.npz", score=calib_scores)
    search = grid_search_policy(calib_cache, calib_pool, calib_scores, gt_calib, args)
    write_json(out_dir / "grid_results.json", search["grid_results"])
    write_json(out_dir / "best_config.json", search["best_config"])
    payload = {
        "stage": "C6-B1-lite R1-safe selector",
        "status": search["status"],
        "official_val_used": False,
        "post_val_adjustment": False,
        "C6_C_used": False,
        "C4_final_retained": True,
        "references": {
            "BMN": "Boundary-Matching Network motivates boundary pair proposal confidence.",
            "BSN_DBG_family": "Boundary-sensitive/dense-boundary proposal generation motivates not trusting boundary peaks without proposal confidence.",
        },
        "runtime": {"infer_train": infer_train, "infer_calib": infer_calib, "pool": pool_audit, "training": train_audit},
        "baseline_pseudo_C4_final": search["baseline"],
        "best_config": search["best_config"],
        "grid_size": len(search["grid_results"]),
        "feasible_count": int(sum(1 for r in search["grid_results"] if r["feasible"])),
        "r1_positive_count": int(sum(1 for r in search["grid_results"] if r["r1_positive"])),
        "r5_safe_count": int(sum(1 for r in search["grid_results"] if r["r5_safe"])),
        "artifacts": {
            "model_best": artifact(out_dir / "model_best.pt"),
            "history": artifact(out_dir / "history.json"),
            "train_pool": artifact(train_pool_path),
            "calib_pool": artifact(calib_pool_path),
            "calib_scores": artifact(out_dir / "train_calib_candidate_scores.npz"),
            "grid_results": artifact(out_dir / "grid_results.json"),
            "best_config": artifact(out_dir / "best_config.json"),
            "train_prior": artifact(train_prior),
            "calib_prior": artifact(calib_prior),
        },
    }
    write_json(audit_dir / "C6_B1_LITE_R1_SAFE_AUDIT.json", payload)
    md = "# C6-B1-lite R1-safe selector audit\n\n"
    md += f"- Status: `{payload['status']}`\n- Scope: `train_fit/train_calib only`\n- Official val used: `false`\n"
    md += "- Primary focus: `R@1`; safety focus: `R@5`.\n"
    md += f"- Model: `{args.layers}` layers, hidden `{args.hidden}`, candidate examples `{pool_audit['train_examples']}`.\n\n"
    md += "## Best config\n\n```json\n" + json.dumps(search["best_config"], ensure_ascii=False, indent=2) + "\n```\n"
    md += "\n## Baseline pseudo C4_final\n\n```json\n" + json.dumps(search["baseline"], ensure_ascii=False, indent=2) + "\n```\n"
    write_text(audit_dir / "C6_B1_LITE_R1_SAFE_AUDIT.md", md)
    manifest = {
        "status": payload["status"],
        "stage": "C6-B1-lite R1-safe selector",
        "official_val_used": False,
        "official_val_authorized": False,
        "post_val_adjustment": False,
        "C6_C_used": False,
        "C4_final_retained": True,
        "R1_primary": True,
        "R5_safety": True,
        "model_scale": {"hidden": args.hidden, "layers": args.layers, "train_examples": int(pool_audit["train_examples"])},
    }
    write_json(audit_dir / "C6_B1_LITE_R1_SAFE_MANIFEST.json", manifest)
    hash_targets = [
        audit_dir / "C6_B1_LITE_R1_SAFE_START_STATE.md",
        audit_dir / "C6_B1_LITE_R1_SAFE_START_STATE.json",
        audit_dir / "C6_B1_LITE_R1_SAFE_AUDIT.md",
        audit_dir / "C6_B1_LITE_R1_SAFE_AUDIT.json",
        audit_dir / "C6_B1_LITE_R1_SAFE_MANIFEST.json",
        out_dir / "model_best.pt",
        out_dir / "history.json",
        out_dir / "best_config.json",
        out_dir / "grid_results.json",
        out_dir / "train_calib_candidate_scores.npz",
        out_dir / "train_calib_candidate_pool_top_slots.npz",
    ]
    write_json(audit_dir / "C6_B1_LITE_R1_SAFE_HASHES.json", {str(p): artifact(p) for p in hash_targets})
    print(json.dumps({"status": payload["status"], "best_config": search["best_config"], "feasible_count": payload["feasible_count"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
