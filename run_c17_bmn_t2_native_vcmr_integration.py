#!/usr/bin/env python3
"""C17 BMN + T2 native train-only VCMR integration replay.

This script builds a bounded train-only VCMR replay over first-stage retrieved
videos and candidate spans. It never runs official validation, never reads
official prediction pools, and never uses pseudo_official_holdout for model
selection.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import random
import subprocess
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

import run_c12_5_native_span_generation as c12_5
import run_c12_5r_span_head_repair as c12_5r
import run_c12_5t_best_span_promotion as c12_5t
from run_c12_native_retriever_training import (
    DEVICE,
    ROOT,
    build_feature_caches,
    load_corpus,
    load_features,
    load_first_stage,
    qtype_id,
    sha256_file,
)
from run_c16_fullscale_bmn_native_localizer import FullBMN, VARIANTS


torch.set_num_threads(min(24, os.cpu_count() or 1))

PROMOTED = "C7-B6 R1SelectiveTop1"
MODEL_DIR = ROOT / "c12_models"
OUT0 = ROOT / "c17_0_protocol_freeze"
OUT1 = ROOT / "c17_1_train_only_vcmr_evaluator"
OUT2 = ROOT / "c17_2_full_bmn_score_export"
OUT3 = ROOT / "c17_3_bmn_t2_hybrid_calibration"
OUT4 = ROOT / "c17_4_native_vcmr_full_integration"
OUT5 = ROOT / "c17_5_robustness_and_consistency"
OUT6 = ROOT / "c17_6_freeze_review"

NMS_THRESHOLD = float(os.environ.get("C17_NMS_THRESHOLD", "0.7"))
EVAL_BATCH = int(os.environ.get("C17_EVAL_BATCH", "96"))

MODE_LIMITS: Dict[str, Dict[str, int]] = {
    "smoke": {"per_split": 100, "top_videos": 10, "spans_per_video": 20, "seeds": 1, "d_max": 64, "hidden": 128},
    "medium": {"per_split": 1000, "top_videos": 100, "spans_per_video": 32, "seeds": 2, "d_max": 64, "hidden": 128},
    "full": {"per_split": 0, "top_videos": 100, "spans_per_video": 50, "seeds": 3, "d_max": 64, "hidden": 128},
}

SCORE_COLUMNS = [
    "query_id",
    "video_id",
    "span_start",
    "span_end",
    "span_duration",
    "retriever_score",
    "retriever_rank",
    "bmn_pred_iou",
    "bmn_p_iou_05",
    "bmn_p_iou_07",
    "bmn_rank_score",
    "bmn_final_score",
    "start_prob",
    "end_prob",
    "actionness_score",
    "duration_score",
    "t2_score",
    "old_c12_score",
    "span_map_rank",
    "seed",
    "split",
    "schema_hash",
    "config_hash",
    "gt_video_id",
    "gt_start",
    "gt_end",
    "query_type",
    "duration_bucket",
]


def jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [jsonable(v) for v in obj]
    if isinstance(obj, tuple):
        return [jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(obj), indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def load_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def sh(cmd: str) -> str:
    return subprocess.check_output(cmd, cwd=ROOT, shell=True, stderr=subprocess.STDOUT, text=True).strip()


def stable_hash(obj: Any) -> str:
    raw = json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def metric(xs: Sequence[bool]) -> float:
    return 100.0 * sum(bool(x) for x in xs) / max(1, len(xs))


def mean(xs: Sequence[float]) -> float | None:
    return float(np.mean(xs)) if xs else None


def pct(xs: Sequence[float], p: float) -> float | None:
    return float(np.percentile(xs, p)) if xs else None


def duration_bucket(seconds: float) -> str:
    if seconds <= 5.0:
        return "short"
    if seconds <= 15.0:
        return "medium"
    return "long"


def mode_cfg(mode: str) -> Dict[str, int]:
    if mode not in MODE_LIMITS:
        raise ValueError(f"unknown mode {mode}")
    return dict(MODE_LIMITS[mode])


def seed_list(base_seed: int, mode: str) -> List[int]:
    return [base_seed + i for i in range(mode_cfg(mode)["seeds"])]


def split_ids(corpus: Any, mode: str) -> Dict[str, List[int]]:
    cfg = mode_cfg(mode)
    out: Dict[str, List[int]] = {}
    for split in ["calib_select", "calib_holdout", "pseudo_official_holdout"]:
        ids = [int(x) for x in corpus.splits[split]]
        limit = cfg["per_split"]
        if limit and len(ids) > limit:
            ids = ids[:limit]
        out[split] = ids
    return out


def video_duration_map(corpus: Any) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for row in corpus.train_rows:
        out.setdefault(str(row["vid_name"]), float(row["duration"]))
    return out


def get_schema_hash(mode: str, cfg: Dict[str, int]) -> str:
    return stable_hash({
        "stage": "C17",
        "mode": mode,
        "top_videos": cfg["top_videos"],
        "spans_per_video": cfg["spans_per_video"],
        "d_max": cfg["d_max"],
        "candidate_columns": SCORE_COLUMNS,
        "nms_threshold": NMS_THRESHOLD,
    })


def get_config_hash(mode: str, seed: int, cfg: Dict[str, int]) -> str:
    return stable_hash({"mode": mode, "seed": seed, "cfg": cfg, "bmn_variant": "B7_full_query_aware_rankloss_map"})


def require_c17_0_ready() -> None:
    rec = load_json(OUT0 / "C17_0_PROTOCOL.json", {})
    if rec.get("status") != "C17_PROTOCOL_READY":
        raise RuntimeError(f"C17-0 is not ready: {rec.get('status')}")


def local_score_cache_path(mode: str) -> Path:
    base = Path(os.environ.get("C17_SCORE_CACHE_DIR", "/tmp/c17_score_cache")) / "CONQUER-RLEM-c2c3"
    base.mkdir(parents=True, exist_ok=True)
    return base / f"C17_2_SCORE_TABLE_{mode}.local.parquet"


def local_score_audit_path(mode: str) -> Path:
    base = Path(os.environ.get("C17_SCORE_CACHE_DIR", "/tmp/c17_score_cache")) / "CONQUER-RLEM-c2c3"
    base.mkdir(parents=True, exist_ok=True)
    return base / f"C17_2_SCORE_TABLE_{mode}.local.audit.json"


def candidate_video_score_z(first_stage: Dict[int, Dict[str, Any]], did: int, candidate_video_pos: int) -> float:
    scores = [float(sc) for _pos, sc in first_stage[int(did)].get("ranklist", [])[:100]]
    mu = float(np.mean(scores)) if scores else 0.0
    sd = float(np.std(scores)) if scores else 1.0
    for pos, sc in first_stage[int(did)].get("ranklist", []):
        if int(pos) == int(candidate_video_pos):
            return float((float(sc) - mu) / max(sd, 1e-6))
    return -5.0


def build_candidate_batch(
    corpus: Any,
    features: Dict[str, Any],
    store: c12_5.ClipFeatureStore,
    pairs: Sequence[Tuple[int, str]],
    durations: Dict[str, float],
) -> Dict[str, Any]:
    q_list, sub_list, vis_list, mask_list, qtypes, lengths = [], [], [], [], [], []
    desc_ids, videos, durations_out = [], [], []
    rows = []
    max_t = 1
    for did, vid in pairs:
        row = corpus.by_id[int(did)]
        sub = store.subtitle(vid)
        t = min(c12_5.MAX_T, sub.shape[0])
        sub = sub[:t]
        vis = store.visual_energy(vid, t)
        rows.append((int(did), str(vid), row, sub, vis, t, float(durations.get(str(vid), row["duration"]))))
        max_t = max(max_t, t)
    for did, vid, row, sub, vis, t, dur in rows:
        pad = max_t - t
        q_list.append(features["query"][features["desc_to_qpos"][did]])
        sub_list.append(np.pad(sub, ((0, pad), (0, 0)), mode="constant"))
        vis_list.append(np.pad(vis, (0, pad), mode="constant"))
        mask_list.append(np.asarray([True] * t + [False] * pad, dtype=bool))
        qtypes.append(qtype_id(row.get("type", "unknown")))
        lengths.append(t)
        desc_ids.append(did)
        videos.append(vid)
        durations_out.append(dur)
    return {
        "desc_ids": desc_ids,
        "video_ids": videos,
        "durations": durations_out,
        "q": torch.from_numpy(np.stack(q_list).astype(np.float32)).to(DEVICE),
        "sub": torch.from_numpy(np.stack(sub_list).astype(np.float32)).to(DEVICE),
        "vis": torch.from_numpy(np.stack(vis_list).astype(np.float32)).to(DEVICE),
        "mask": torch.from_numpy(np.stack(mask_list)).to(DEVICE),
        "lengths": torch.tensor(lengths, dtype=torch.long, device=DEVICE),
        "qtype": torch.tensor(qtypes, dtype=torch.long, device=DEVICE),
    }


def load_bmn_model(seed: int, mode: str, cfg: Dict[str, int]) -> Tuple[FullBMN, Dict[str, Any]]:
    path = MODEL_DIR / f"c16_{mode}_B7_full_query_aware_rankloss_map_seed{seed}.pt"
    if not path.exists() and mode != "medium":
        path = MODEL_DIR / f"c16_medium_B7_full_query_aware_rankloss_map_seed{seed}.pt"
    ckpt = torch.load(path, map_location=DEVICE)
    model = FullBMN(int(cfg["hidden"]), int(cfg["d_max"]), True, True, True).to(DEVICE)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, {"path": str(path.relative_to(ROOT)), "sha256": sha256_file(path), "seed": seed, "mode_in_checkpoint": ckpt.get("mode")}


def load_t2_ranker() -> Tuple[c12_5.SpanLocalizer, c12_5t.RichRanker, Sequence[int], Dict[str, Any]]:
    teacher = c12_5r.load_model("teacher_distilled")
    idxs = c12_5t.variant_feature_indices("T2_listwise_soft_iou")
    ranker = c12_5t.RichRanker(in_dim=len(idxs), hidden=224).to(DEVICE)
    path = MODEL_DIR / "c12_5t_T2_listwise_soft_iou.pt"
    ckpt = torch.load(path, map_location=DEVICE)
    ranker.load_state_dict(ckpt["state_dict"])
    ranker.eval()
    return teacher.eval(), ranker, idxs, {"path": str(path.relative_to(ROOT)), "sha256": sha256_file(path)}


def t2_feature_matrix_from_context(
    ctx: Dict[str, np.ndarray],
    qtype: int,
    retriever_score_z: float,
    pool: Sequence[Tuple[int, int, float, Dict[str, float]]],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not pool:
        return np.zeros((0, len(c12_5t.FEATURE_NAMES)), dtype=np.float32), np.zeros((0,), dtype=np.int32), np.zeros((0,), dtype=np.int32)
    start = ctx["start"]
    end = ctx["end"]
    sim = ctx["sim"]
    vis = ctx["vis"]
    t = max(1, len(start))
    s = np.asarray([int(x[0]) for x in pool], dtype=np.int32)
    e = np.asarray([int(x[1]) for x in pool], dtype=np.int32)
    generated_score = np.asarray([float(x[2]) for x in pool], dtype=np.float32)
    previous_pq = np.asarray([float(x[3].get("pq", 0.0)) for x in pool], dtype=np.float32)
    dur = (e - s + 1).astype(np.float32)
    dur_norm = dur / float(t)
    short = (dur_norm <= 0.08).astype(np.float32)
    medium = ((dur_norm > 0.08) & (dur_norm <= 0.25)).astype(np.float32)
    long = (dur_norm > 0.25).astype(np.float32)
    q = np.zeros((len(pool), 4), dtype=np.float32)
    q[:, min(max(int(qtype), 0), 3)] = 1.0
    center = ((s + e).astype(np.float32) * 0.5) / float(t)
    target = {0: 0.10, 1: 0.16, 2: 0.13, 3: 0.14}.get(int(qtype), 0.14)
    duration_prior = -np.abs(dur_norm - float(target))

    def range_mean(x: np.ndarray, left: np.ndarray, right: np.ndarray) -> np.ndarray:
        pref = np.concatenate([np.asarray([0.0], dtype=np.float32), np.cumsum(x.astype(np.float32))])
        left = np.clip(left, 0, t - 1)
        right = np.clip(right, 0, t - 1)
        denom = np.maximum(1, right - left + 1).astype(np.float32)
        return (pref[right + 1] - pref[left]) / denom

    left_s = np.maximum(0, s - (e - s + 1))
    left_e = np.maximum(0, s - 1)
    right_s = np.minimum(t - 1, e + 1)
    right_e = np.minimum(t - 1, e + (e - s + 1))
    sub_inside = range_mean(sim, s, e)
    sub_left = np.where(s > 0, range_mean(sim, left_s, left_e), sub_inside)
    sub_right = np.where(e + 1 < t, range_mean(sim, right_s, right_e), sub_inside)
    vis_inside = range_mean(vis, s, e)
    vis_left = np.where(s > 0, range_mean(vis, left_s, left_e), vis_inside)
    vis_right = np.where(e + 1 < t, range_mean(vis, right_s, right_e), vis_inside)
    left_jump = np.where(s > 0, sim[s] - sim[np.maximum(0, s - 1)], 0.0).astype(np.float32)
    right_jump = np.where(e + 1 < t, sim[e] - sim[np.minimum(t - 1, e + 1)], 0.0).astype(np.float32)
    generated_rank_norm = np.arange(len(pool), dtype=np.float32) / float(max(1, c12_5t.POOL_LIMIT - 1))
    feat = np.stack([
        start[s],
        end[e],
        ctx["start_prob"][s],
        ctx["end_prob"][e],
        ctx["start_rank"][s],
        ctx["end_rank"][e],
        ctx["start_margin"][s],
        ctx["end_margin"][e],
        ctx["start_sharp"][s],
        ctx["end_sharp"][e],
        dur_norm,
        short,
        medium,
        long,
        np.full(len(pool), float(retriever_score_z), dtype=np.float32),
        q[:, 0],
        q[:, 1],
        q[:, 2],
        q[:, 3],
        vis_inside,
        vis_left,
        vis_right,
        vis_inside - 0.5 * (vis_left + vis_right),
        sub_inside,
        sub_left,
        sub_right,
        sub_inside - 0.5 * (sub_left + sub_right),
        generated_rank_norm,
        generated_score,
        previous_pq,
        center,
        left_jump,
        right_jump,
        duration_prior,
    ], axis=1).astype(np.float32)
    return feat, s, e


@torch.no_grad()
def bmn_scores_for_batch(model: FullBMN, batch: Dict[str, Any], top_for_union: int) -> List[Dict[Tuple[int, int], Dict[str, float]]]:
    enc = model.encode(batch)
    out_all: List[Dict[Tuple[int, int], Dict[str, float]]] = []
    for bi in range(len(batch["desc_ids"])):
        length = int(batch["lengths"][bi].item())
        spans, out = model.span_outputs_one(enc, bi, length)
        start_p = torch.sigmoid(enc["start"][bi, :length])
        end_p = torch.sigmoid(enc["end"][bi, :length])
        action_p = torch.sigmoid(enc["action"][bi, :length])
        sidx, eidx = spans[:, 0], spans[:, 1]
        pref = torch.cat([action_p.new_zeros(1), action_p.cumsum(dim=0)], dim=0)
        act = (pref[eidx + 1] - pref[sidx]) / (eidx - sidx + 1).float().clamp_min(1.0)
        pred_iou = out["pred_iou"]
        p05 = torch.sigmoid(out["p05"])
        p07 = torch.sigmoid(out["p07"])
        final = torch.sigmoid(out["final"])
        score = final + 0.55 * pred_iou + 0.30 * p07 + 0.20 * (start_p[sidx] + end_p[eidx]) + 0.15 * act
        keep = min(int(score.numel()), int(top_for_union))
        order = torch.argsort(score, descending=True)[:keep].detach().cpu().tolist()
        row_scores: Dict[Tuple[int, int], Dict[str, float]] = {}
        for rank, idx in enumerate(order, start=1):
            s = int(sidx[idx].item())
            e = int(eidx[idx].item())
            row_scores[(s, e)] = {
                "bmn_pred_iou": float(pred_iou[idx].detach().cpu()),
                "bmn_p_iou_05": float(p05[idx].detach().cpu()),
                "bmn_p_iou_07": float(p07[idx].detach().cpu()),
                "bmn_final_score": float(score[idx].detach().cpu()),
                "bmn_rank_score": float(1.0 / rank),
                "start_prob": float(start_p[s].detach().cpu()),
                "end_prob": float(end_p[e].detach().cpu()),
                "actionness_score": float(act[idx].detach().cpu()),
                "duration_score": float((e - s + 1) / max(1, length)),
                "span_map_rank": rank,
                "bmn_union_candidate": rank <= top_for_union,
            }
        out_all.append(row_scores)
    return out_all


@torch.no_grad()
def t2_scores_for_batch(
    teacher: c12_5.SpanLocalizer,
    ranker: c12_5t.RichRanker,
    idxs: Sequence[int],
    batch: Dict[str, Any],
    first_stage: Dict[int, Dict[str, Any]],
    corpus: Any,
) -> List[Dict[Tuple[int, int], Dict[str, float]]]:
    tok = teacher.token_forward(batch["q"], batch["sub"], batch["vis"], batch["mask"], batch["qtype"])
    out_all: List[Dict[Tuple[int, int], Dict[str, float]]] = []
    for bi, did in enumerate(batch["desc_ids"]):
        length = int(batch["lengths"][bi].item())
        start = tok["start"][bi, :length].detach().cpu().numpy()
        end = tok["end"][bi, :length].detach().cpu().numpy()
        sim = tok["sim"][bi, :length].detach().cpu().numpy()
        vis = batch["vis"][bi, :length].detach().cpu().numpy()
        qt = int(batch["qtype"][bi].item())
        pool = c12_5r.dense_pool(start, end, sim, qt, "r5_two_stage", limit=c12_5t.POOL_LIMIT)
        video_pos = corpus.video_to_pos[batch["video_ids"][bi]]
        retr_z = candidate_video_score_z(first_stage, int(did), int(video_pos))
        ctx = c12_5t.build_span_feature_context(start, end, sim, vis)
        feat, s_arr, e_arr = t2_feature_matrix_from_context(ctx, qt, retr_z, pool)
        scores = np.zeros((len(pool),), dtype=np.float32)
        if len(feat):
            chunks = []
            xx = feat[:, idxs]
            chunk = 65536
            for k in range(0, len(xx), chunk):
                chunks.append(torch.sigmoid(ranker(torch.from_numpy(xx[k:k + chunk]).to(DEVICE))).detach().cpu().numpy())
            scores = np.concatenate(chunks).astype(np.float32)
        order_rank = np.empty_like(scores, dtype=np.int32)
        order_rank[np.argsort(-scores)] = np.arange(1, len(scores) + 1, dtype=np.int32)
        row_scores: Dict[Tuple[int, int], Dict[str, float]] = {}
        top_idx = np.argsort(-scores)[: min(len(scores), 96)] if len(scores) else []
        for idx in top_idx:
            _s, _e, base_score, meta = pool[int(idx)]
            s_i = int(s_arr[idx])
            e_i = int(e_arr[idx])
            row_scores[(s_i, e_i)] = {
                "t2_score": float(scores[idx]),
                "old_c12_score": float(base_score),
                "previous_c12_pq_score": float(meta.get("pq", 0.0)),
                "t2_rank": int(order_rank[idx]) if len(order_rank) else None,
                "t2_union_candidate": int(order_rank[idx]) <= 64 if len(order_rank) else False,
            }
        out_all.append(row_scores)
    return out_all


def retrieved_videos(corpus: Any, first_stage: Dict[int, Dict[str, Any]], did: int, top_videos: int) -> List[Tuple[str, float, int]]:
    ranklist = first_stage[int(did)].get("ranklist", [])[:top_videos]
    out: List[Tuple[str, float, int]] = []
    seen = set()
    for rank, (pos, score) in enumerate(ranklist, start=1):
        pos_i = int(pos)
        if pos_i < 0 or pos_i >= len(corpus.train_videos):
            continue
        vid = str(corpus.train_videos[pos_i])
        if vid in seen:
            continue
        seen.add(vid)
        out.append((vid, float(score), rank))
    gt_vid = str(corpus.by_id[int(did)]["vid_name"])
    if gt_vid not in seen and isinstance(first_stage.get(int(did), {}).get("rank"), int) and int(first_stage[int(did)]["rank"]) <= top_videos:
        out.append((gt_vid, candidate_video_score_z(first_stage, int(did), corpus.video_to_pos[gt_vid]), int(first_stage[int(did)]["rank"])))
    return out


def generate_score_table(mode: str, seed: int, splits: Sequence[str]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    cfg = mode_cfg(mode)
    corpus = load_corpus()
    c12_5.corpus_global = corpus
    features = load_features(build_feature_caches(corpus))
    ids_by = split_ids(corpus, mode)
    durations = video_duration_map(corpus)
    schema_hash = get_schema_hash(mode, cfg)
    teacher, t2_ranker, t2_idxs, t2_info = load_t2_ranker()
    seeds = seed_list(seed, mode)
    bmn_models = {s: load_bmn_model(s, mode, cfg) for s in seeds}
    rows: List[Dict[str, Any]] = []
    audits: Dict[str, Any] = {"mode": mode, "seeds": seeds, "splits": {}, "t2_checkpoint": t2_info, "bmn_checkpoints": {str(k): v[1] for k, v in bmn_models.items()}}
    store = c12_5.ClipFeatureStore()
    start_time = time.time()
    try:
        for split in splits:
            ids = ids_by[split]
            fs = load_first_stage(corpus, ids, top_keep=128, cache_name=f"first_stage_c17_{mode}_{split}_top128.pkl")
            pairs: List[Tuple[int, str, float, int]] = []
            for did in ids:
                for vid, rscore, rrank in retrieved_videos(corpus, fs, int(did), int(cfg["top_videos"])):
                    pairs.append((int(did), vid, float(rscore), int(rrank)))
            print(f"[C17-2] split={split} query_count={len(ids)} query_video_pairs={len(pairs)} batch={EVAL_BATCH}", flush=True)
            split_counts = {"query_count": len(ids), "query_video_pairs": len(pairs), "candidate_rows_before_dedup": 0}
            for st in range(0, len(pairs), EVAL_BATCH):
                block = pairs[st: st + EVAL_BATCH]
                if st == 0 or st % max(EVAL_BATCH * 20, 1) == 0:
                    elapsed = time.time() - start_time
                    done = st + len(block)
                    print(f"[C17-2] split={split} pairs={done}/{len(pairs)} rows={len(rows)} elapsed={elapsed:.1f}s", flush=True)
                batch = build_candidate_batch(corpus, features, store, [(d, v) for d, v, _rs, _rr in block], durations)
                t2_maps = t2_scores_for_batch(teacher, t2_ranker, t2_idxs, batch, fs, corpus)
                for s, (bmn_model, _info) in bmn_models.items():
                    bmn_maps = bmn_scores_for_batch(bmn_model, batch, int(cfg["spans_per_video"]) * 3)
                    config_hash = get_config_hash(mode, s, cfg)
                    for bi, (did, vid, rscore, rrank) in enumerate(block):
                        row = corpus.by_id[int(did)]
                        gt_vid = str(row["vid_name"])
                        gt_ts = (float(row["ts"][0]), float(row["ts"][1]))
                        duration = float(batch["durations"][bi])
                        length = int(batch["lengths"][bi].item())
                        bmn_map = bmn_maps[bi]
                        t2_map = t2_maps[bi]
                        bmn_top = sorted(bmn_map.items(), key=lambda kv: kv[1]["span_map_rank"])[: int(cfg["spans_per_video"])]
                        t2_top = sorted(t2_map.items(), key=lambda kv: kv[1]["t2_rank"] or 10**9)[: int(cfg["spans_per_video"])]
                        union_spans = []
                        seen = set()
                        for span, _val in bmn_top + t2_top:
                            if span not in seen:
                                seen.add(span)
                                union_spans.append(span)
                        split_counts["candidate_rows_before_dedup"] += len(union_spans)
                        for span in union_spans:
                            sidx, eidx = int(span[0]), int(span[1])
                            if eidx < sidx or sidx < 0 or eidx >= length:
                                continue
                            st_ts, ed_ts = c12_5.idx_to_ts(sidx, eidx, duration, length)
                            bmn = bmn_map.get(span)
                            t2 = t2_map.get(span)
                            rec = {
                                "query_id": int(did),
                                "video_id": vid,
                                "span_start": float(st_ts),
                                "span_end": float(ed_ts),
                                "span_duration": float(ed_ts - st_ts),
                                "retriever_score": float(rscore),
                                "retriever_rank": int(rrank),
                                "bmn_pred_iou": bmn.get("bmn_pred_iou") if bmn else None,
                                "bmn_p_iou_05": bmn.get("bmn_p_iou_05") if bmn else None,
                                "bmn_p_iou_07": bmn.get("bmn_p_iou_07") if bmn else None,
                                "bmn_rank_score": bmn.get("bmn_rank_score") if bmn else None,
                                "bmn_final_score": bmn.get("bmn_final_score") if bmn else None,
                                "start_prob": bmn.get("start_prob") if bmn else None,
                                "end_prob": bmn.get("end_prob") if bmn else None,
                                "actionness_score": bmn.get("actionness_score") if bmn else None,
                                "duration_score": bmn.get("duration_score") if bmn else float((eidx - sidx + 1) / max(1, length)),
                                "t2_score": t2.get("t2_score") if t2 else None,
                                "old_c12_score": t2.get("old_c12_score") if t2 else None,
                                "span_map_rank": bmn.get("span_map_rank") if bmn else None,
                                "seed": int(s),
                                "split": split,
                                "schema_hash": schema_hash,
                                "config_hash": config_hash,
                                "gt_video_id": gt_vid,
                                "gt_start": gt_ts[0],
                                "gt_end": gt_ts[1],
                                "query_type": row.get("type", "unknown"),
                                "duration_bucket": duration_bucket(gt_ts[1] - gt_ts[0]),
                                "gt_video_rank": int(fs[int(did)].get("rank", 10**9)) if isinstance(fs[int(did)].get("rank"), int) else None,
                                "source_has_bmn": bmn is not None,
                                "source_has_t2": t2 is not None,
                            }
                            rows.append(rec)
            audits["splits"][split] = split_counts
    finally:
        store.close()
    df = pd.DataFrame(rows)
    audits["runtime_seconds"] = time.time() - start_time
    audits["row_count"] = int(len(df))
    audits["schema_hash"] = schema_hash
    return df, audits


def add_normalized_scores(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    out = df.copy()
    fill_audit: Dict[str, Any] = {}
    for col, dst in [("retriever_score", "retriever_norm"), ("bmn_final_score", "bmn_norm"), ("t2_score", "t2_norm"), ("old_c12_score", "old_norm")]:
        out[dst] = np.nan
        missing = int(out[col].isna().sum()) if col in out else len(out)
        fill_audit[col] = {"missing_count": missing, "fill_policy": "per-query minimum before minmax; audited, not silent zero"}
        for (_split, qid, seed), idx in out.groupby(["split", "query_id", "seed"]).groups.items():
            vals = out.loc[idx, col].astype(float)
            if vals.notna().any():
                min_val = float(vals.min(skipna=True))
                max_val = float(vals.max(skipna=True))
                vals = vals.fillna(min_val)
                denom = max(max_val - min_val, 1e-8)
                out.loc[idx, dst] = (vals - min_val) / denom
            else:
                out.loc[idx, dst] = 0.0
    return out, fill_audit


def apply_formula(df: pd.DataFrame, formula: Dict[str, Any]) -> pd.Series:
    kind = formula["family"]
    a = float(formula.get("alpha", 0.0))
    b = float(formula.get("beta", 0.0))
    g = float(formula.get("gamma", 0.0))
    d = float(formula.get("delta", 0.0))
    e = float(formula.get("eta", 0.0))
    score = a * df["retriever_norm"] + b * df["bmn_norm"] + g * df["t2_norm"] + d * df["old_norm"] + e * df["duration_score"].fillna(0.0)
    if kind == "H_qtype_gated":
        score = score + np.where(df["query_type"].astype(str) == "t", 0.15 * df["t2_norm"], 0.10 * df["bmn_norm"])
    elif kind == "I_duration_aware":
        score = score + np.where(df["duration_bucket"].astype(str) == "short", 0.18 * df["bmn_norm"], 0.06 * df["t2_norm"])
    elif kind == "J_safety_gated":
        conflict = (df["t2_norm"] > 0.85) & (df["bmn_norm"] < 0.25)
        score = np.where(conflict, 0.70 * df["t2_norm"] + 0.30 * df["retriever_norm"], score)
    return pd.Series(score, index=df.index, dtype=float)


def load_eval_score_table(mode: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    cache = local_score_cache_path(mode)
    if not cache.exists():
        raise FileNotFoundError(f"missing local C17 score cache: {cache}")
    columns = [
        "split",
        "query_id",
        "seed",
        "video_id",
        "span_start",
        "span_end",
        "gt_video_id",
        "gt_start",
        "gt_end",
        "query_type",
        "duration_bucket",
        "gt_video_rank",
        "retriever_score",
        "bmn_final_score",
        "t2_score",
        "old_c12_score",
        "duration_score",
    ]
    df = pd.read_parquet(cache, columns=columns)
    for col in ["split", "query_type", "duration_bucket"]:
        df[col] = df[col].astype("category")
    video_cats = pd.Index(pd.concat([df["video_id"], df["gt_video_id"]], ignore_index=True).astype(str).unique())
    df["video_id"] = pd.Categorical(df["video_id"].astype(str), categories=video_cats)
    df["gt_video_id"] = pd.Categorical(df["gt_video_id"].astype(str), categories=video_cats)
    for col in ["query_id", "seed", "gt_video_rank"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(-1).astype(np.int32)
    for col in ["span_start", "span_end", "gt_start", "gt_end", "retriever_score", "bmn_final_score", "t2_score", "old_c12_score", "duration_score"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype(np.float32)
    audit = {
        "row_count": int(len(df)),
        "cache_path": str(cache),
        "loaded_columns": columns,
        "memory_optimized": True,
    }
    return df, audit


def add_normalized_scores_inplace(df: pd.DataFrame) -> Dict[str, Any]:
    fill_audit: Dict[str, Any] = {}
    group_keys = [df["split"], df["query_id"], df["seed"]]
    for col, dst in [("retriever_score", "retriever_norm"), ("bmn_final_score", "bmn_norm"), ("t2_score", "t2_norm"), ("old_c12_score", "old_norm")]:
        vals = df[col].astype(np.float32)
        missing = int(vals.isna().sum())
        gb = vals.groupby(group_keys, observed=True)
        mins = gb.transform("min").astype(np.float32)
        maxs = gb.transform("max").astype(np.float32)
        filled = vals.fillna(mins).fillna(0.0).astype(np.float32)
        denom = (maxs - mins).replace(0.0, np.nan).astype(np.float32)
        norm = ((filled - mins.fillna(0.0)) / denom).replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float32)
        df[dst] = norm
        fill_audit[col] = {"missing_count": missing, "fill_policy": "per-query minimum before minmax; audited, not silent zero"}
    return fill_audit


def assign_formula_score(df: pd.DataFrame, formula: Dict[str, Any], out_col: str = "eval_score") -> None:
    score = apply_formula(df, formula)
    df[out_col] = pd.to_numeric(score, errors="coerce").fillna(0.0).astype(np.float32)


def assign_oracle_score(df: pd.DataFrame, out_col: str = "eval_score") -> None:
    if hasattr(df["video_id"], "cat") and hasattr(df["gt_video_id"], "cat"):
        same_video = df["video_id"].cat.codes.to_numpy() == df["gt_video_id"].cat.codes.to_numpy()
    else:
        same_video = df["video_id"].astype(str).to_numpy() == df["gt_video_id"].astype(str).to_numpy()
    span_start = df["span_start"].to_numpy(dtype=np.float32)
    span_end = df["span_end"].to_numpy(dtype=np.float32)
    gt_start = df["gt_start"].to_numpy(dtype=np.float32)
    gt_end = df["gt_end"].to_numpy(dtype=np.float32)
    inter = np.maximum(0.0, np.minimum(span_end, gt_end) - np.maximum(span_start, gt_start))
    union = np.maximum(span_end, gt_end) - np.minimum(span_start, gt_start)
    iou = inter / np.maximum(union, 1e-6)
    df[out_col] = np.where(same_video & (iou >= 0.7), 10.0, np.where(same_video, 5.0, 0.0)).astype(np.float32)


def nms_rank_query(rows: pd.DataFrame, score_col: str, max_keep: int = 200) -> pd.DataFrame:
    rows = rows.sort_values(score_col, ascending=False)
    kept: List[int] = []
    by_video: Dict[str, List[Tuple[float, float]]] = defaultdict(list)
    for idx, r in rows.iterrows():
        vid = str(r["video_id"])
        span = (float(r["span_start"]), float(r["span_end"]))
        if any(c12_5.iou_1d(span, prev) > NMS_THRESHOLD for prev in by_video[vid]):
            continue
        by_video[vid].append(span)
        kept.append(idx)
        if len(kept) >= max_keep:
            break
    return rows.loc[kept].copy()


def evaluate_vcmr(df: pd.DataFrame, score_col: str = "final_score") -> Dict[str, Any]:
    records: List[Dict[str, Any]] = []
    duplicate_count = int(df.duplicated(["split", "seed", "query_id", "video_id", "span_start", "span_end"]).sum())
    invalid_count = int(((df["span_end"] <= df["span_start"]) | df["span_start"].isna() | df["span_end"].isna()).sum())
    for (_split, seed, qid), qdf in df.groupby(["split", "seed", "query_id"], sort=False):
        ranked = nms_rank_query(qdf, score_col)
        if ranked.empty:
            continue
        gt_video = str(ranked.iloc[0]["gt_video_id"])
        gt_ts = (float(ranked.iloc[0]["gt_start"]), float(ranked.iloc[0]["gt_end"]))
        ious = [c12_5.iou_1d((float(r["span_start"]), float(r["span_end"])), gt_ts) if str(r["video_id"]) == gt_video else 0.0 for _, r in ranked.iterrows()]
        videos = [str(v) for v in ranked["video_id"].tolist()]
        unique_videos = []
        for v in videos:
            if v not in unique_videos:
                unique_videos.append(v)
        top1_video_ok = videos[0] == gt_video if videos else False
        top1_iou = ious[0] if ious else 0.0
        gt_only = ranked[ranked["video_id"].astype(str) == gt_video]
        gt_ious = [c12_5.iou_1d((float(r["span_start"]), float(r["span_end"])), gt_ts) for _, r in gt_only.iterrows()]
        rec = {
            "split": _split,
            "seed": int(seed),
            "query_id": int(qid),
            "query_type": str(ranked.iloc[0]["query_type"]),
            "duration_bucket": str(ranked.iloc[0]["duration_bucket"]),
            "gt_video_rank": int(ranked.iloc[0]["gt_video_rank"]) if not pd.isna(ranked.iloc[0].get("gt_video_rank")) else None,
            "top1_video_correct": top1_video_ok,
            "top1_iou": float(top1_iou),
            "wrong_video_top1": not top1_video_ok,
            "correct_video_wrong_span_top1": bool(top1_video_ok and top1_iou < 0.5),
        }
        for k in [1, 5, 10, 100]:
            rec[f"VCMR_R@{k}_IoU0.5"] = any(i >= 0.5 for i in ious[: min(k, len(ious))])
            rec[f"VCMR_R@{k}_IoU0.7"] = any(i >= 0.7 for i in ious[: min(k, len(ious))])
            rec[f"VR_R@{k}"] = gt_video in unique_videos[: min(k, len(unique_videos))]
        for k in [50, 100]:
            rec[f"GT_VIDEO_TOP{k}_IoU0.5"] = any(i >= 0.5 for i in gt_ious[: min(k, len(gt_ious))])
            rec[f"GT_VIDEO_TOP{k}_IoU0.7"] = any(i >= 0.7 for i in gt_ious[: min(k, len(gt_ious))])
        records.append(rec)

    def aggregate(rs: List[Dict[str, Any]]) -> Dict[str, Any]:
        out: Dict[str, Any] = {"query_count": len(rs)}
        for k in [1, 5, 10, 100]:
            out[f"VCMR_R@{k}_IoU0.5"] = metric([r[f"VCMR_R@{k}_IoU0.5"] for r in rs])
            out[f"VCMR_R@{k}_IoU0.7"] = metric([r[f"VCMR_R@{k}_IoU0.7"] for r in rs])
            out[f"VR_R@{k}"] = metric([r[f"VR_R@{k}"] for r in rs])
        for k in [50, 100]:
            out[f"GT_VIDEO_TOP{k}_IoU0.5"] = metric([r[f"GT_VIDEO_TOP{k}_IoU0.5"] for r in rs])
            out[f"GT_VIDEO_TOP{k}_IoU0.7"] = metric([r[f"GT_VIDEO_TOP{k}_IoU0.7"] for r in rs])
        out["wrong_video_high_score_rate"] = metric([r["wrong_video_top1"] for r in rs])
        out["correct_video_wrong_span_rate"] = metric([r["correct_video_wrong_span_top1"] for r in rs])
        out["top1_mean_iou"] = mean([float(r["top1_iou"]) for r in rs])
        out["duplicate_span_count"] = duplicate_count
        out["invalid_span_count"] = invalid_count
        return out

    by_qtype: Dict[str, Any] = {}
    by_duration: Dict[str, Any] = {}
    by_video_rank: Dict[str, Any] = {}
    if records:
        frame = pd.DataFrame(records)
        by_qtype = {str(k): aggregate(v.to_dict("records")) for k, v in frame.groupby("query_type")}
        by_duration = {str(k): aggregate(v.to_dict("records")) for k, v in frame.groupby("duration_bucket")}
        def rb(x: Any) -> str:
            if x is None:
                return "missing"
            x = int(x)
            if x <= 1:
                return "rank1"
            if x <= 5:
                return "rank2_5"
            if x <= 10:
                return "rank6_10"
            if x <= 100:
                return "rank11_100"
            return "rank_gt100"
        frame["video_rank_bucket"] = [rb(x) for x in frame["gt_video_rank"].tolist()]
        by_video_rank = {str(k): aggregate(v.to_dict("records")) for k, v in frame.groupby("video_rank_bucket")}
    return {
        "summary": aggregate(records),
        "query_type_breakdown": by_qtype,
        "duration_breakdown": by_duration,
        "video_rank_breakdown": by_video_rank,
        "records_sample": records[:100],
    }


def build_formula_space() -> List[Dict[str, Any]]:
    formulas = [
        {"name": "A_retriever_old_c12", "family": "A_retriever_old_c12", "alpha": 0.60, "delta": 0.40},
        {"name": "B_T2_only", "family": "B_T2_only", "gamma": 1.0},
        {"name": "C_BMN_map_only", "family": "C_BMN_map_only", "beta": 1.0},
        {"name": "D_retriever_BMN", "family": "D_retriever_BMN", "alpha": 0.45, "beta": 0.55},
        {"name": "E_retriever_T2", "family": "E_retriever_T2", "alpha": 0.45, "gamma": 0.55},
        {"name": "F_BMN_T2", "family": "F_BMN_T2", "beta": 0.55, "gamma": 0.45},
    ]
    for alpha in [0.25, 0.40, 0.55]:
        for beta in [0.25, 0.40, 0.55]:
            gamma = round(1.0 - alpha - beta, 2)
            if gamma < 0.10:
                continue
            formulas.append({"name": f"G_all_a{alpha}_b{beta}_g{gamma}", "family": "G_retriever_BMN_T2", "alpha": alpha, "beta": beta, "gamma": gamma})
    formulas.extend([
        {"name": "H_qtype_gated", "family": "H_qtype_gated", "alpha": 0.35, "beta": 0.35, "gamma": 0.30},
        {"name": "I_duration_aware", "family": "I_duration_aware", "alpha": 0.35, "beta": 0.40, "gamma": 0.25},
        {"name": "J_safety_gated", "family": "J_safety_gated", "alpha": 0.35, "beta": 0.40, "gamma": 0.25},
    ])
    return formulas


def score_for_selection(metrics: Dict[str, Any]) -> float:
    s = metrics["summary"]
    return (
        float(s.get("VCMR_R@100_IoU0.7", 0.0))
        + 0.40 * float(s.get("VCMR_R@10_IoU0.7", 0.0))
        + 0.20 * float(s.get("VCMR_R@1_IoU0.7", 0.0))
        - 0.10 * float(s.get("wrong_video_high_score_rate", 0.0))
    )


def evaluate_formulas(df: pd.DataFrame, formulas: Sequence[Dict[str, Any]]) -> Tuple[Dict[str, Any], pd.DataFrame]:
    norm, fill_audit = add_normalized_scores(df)
    results: Dict[str, Any] = {"fill_audit": fill_audit, "formula_results": {}, "selection_split": "calib_select", "holdout_split": "calib_holdout"}
    best_name, best_score = "", -1e18
    for formula in formulas:
        col = "score_" + formula["name"]
        norm[col] = apply_formula(norm, formula)
        split_results = {}
        for split in ["calib_select", "calib_holdout"]:
            split_df = norm[norm["split"] == split].copy()
            split_df["final_score"] = split_df[col]
            split_results[split] = evaluate_vcmr(split_df, "final_score")
        sel = score_for_selection(split_results["calib_select"])
        results["formula_results"][formula["name"]] = {"formula": formula, "selection_score": sel, **split_results}
        if sel > best_score:
            best_name, best_score = formula["name"], sel
    results["best_formula_name"] = best_name
    results["best_formula"] = results["formula_results"][best_name]["formula"] if best_name else None
    results["pseudo_official_holdout_used_for_selection"] = False
    if best_name:
        norm["final_score"] = norm["score_" + best_name]
    return results, norm


def stage_c17_0(mode: str, seed: int) -> Dict[str, Any]:
    OUT0.mkdir(parents=True, exist_ok=True)
    branch = sh("git branch --show-current")
    commit = sh("git rev-parse HEAD")
    dirty = sh("git status --short")
    root_markers = [p.name for p in [ROOT / "C9_OFFICIAL_VAL_AUTHORIZED", ROOT / "OFFICIAL_VAL_AUTHORIZED"] if p.exists()]
    deps = {
        "c16_best_model": ROOT / "c16_2_fullscale_bmn_training/C16_2_BMN_BEST_MODEL_DECISION.json",
        "c16_robustness": ROOT / "c16_3_ablation_robustness/C16_3_ROBUSTNESS_DECISION.json",
        "c16_integration": ROOT / "c16_4_native_vcmr_integration/C16_4_INTEGRATION_DECISION.json",
        "c16_next_step_json": ROOT / "c16_5_freeze_review/C16_5_NEXT_STEP_DECISION.json",
        "c16_next_step_md": ROOT / "c16_5_freeze_review/C16_5_NEXT_STEP_DECISION.md",
        "c12_split_manifest": ROOT / "c12_1_schema_and_split/C12_1_SPLIT_MANIFEST.json",
        "c12_schema_manifest": ROOT / "c12_1_schema_and_split/C12_1_SCHEMA_MANIFEST.json",
        "c12_t2_checkpoint": ROOT / "c12_models/c12_5t_T2_listwise_soft_iou.pt",
        "c16_checkpoint_manifest": ROOT / "c16_2_fullscale_bmn_training/C16_2_BMN_CHECKPOINT_MANIFEST.json",
        "c16_seed2026_checkpoint": ROOT / "c12_models/c16_medium_B7_full_query_aware_rankloss_map_seed2026.pt",
        "c16_seed2027_checkpoint": ROOT / "c12_models/c16_medium_B7_full_query_aware_rankloss_map_seed2027.pt",
    }
    missing = [k for k, p in deps.items() if not p.exists()]
    c16_best = load_json(deps["c16_best_model"], {})
    c16_rob = load_json(deps["c16_robustness"], {})
    c16_int = load_json(deps["c16_integration"], {})
    c16_next = load_json(deps["c16_next_step_json"], {})
    promoted = load_json(ROOT / "c7_audit/CURRENT_PROMOTED_SYSTEM.json", {})
    checks = {
        "branch_expected": branch == "c17-bmn-t2-native-vcmr-integration",
        "head_contains_c16_commit": "f0642ef" in sh("git log --oneline -8"),
        "c16_2_promising": c16_best.get("status") == "C16_BMN_FULLSCALE_PROMISING",
        "c16_3_partial_or_pass": c16_rob.get("status") in {"C16_ROBUSTNESS_PARTIAL", "C16_ROBUSTNESS_PASS"},
        "c16_4_localizer_only": c16_int.get("status") == "C16_NATIVE_INTEGRATION_LOCALIZER_ONLY",
        "c16_5_continue_hybrid": c16_next.get("status") == "C16_CONTINUE_BMN_T2_HYBRID_TRAIN_ONLY",
        "official_val_used_false": c16_next.get("official_val_used") is False,
        "promoted_system_unchanged": "C7-B6" in json.dumps(promoted, sort_keys=True) and "R1SelectiveTop1" in json.dumps(promoted, sort_keys=True),
        "no_root_stale_authorization_marker": not root_markers,
    }
    if missing:
        status = "C17_PROTOCOL_BLOCKED_MISSING_CORE_ARTIFACTS"
    elif root_markers:
        status = "C17_PROTOCOL_BLOCKED_CONTAMINATION_RISK"
    elif all(checks.values()):
        status = "C17_PROTOCOL_READY"
    else:
        status = "C17_PROTOCOL_BLOCKED_MISSING_CORE_ARTIFACTS"
    rec = {
        "stage": "C17-0",
        "status": status,
        "mode": mode,
        "seed": seed,
        "branch": branch,
        "commit": commit,
        "dirty_status": dirty.splitlines(),
        "checks": checks,
        "missing_core_artifacts": missing,
        "root_stale_authorization_markers": root_markers,
        "current_promoted_system": PROMOTED,
        "official_val_used": False,
        "pseudo_official_holdout_used_for_selection": False,
    }
    write_json(OUT0 / "C17_0_PROTOCOL.json", rec)
    write_json(OUT0 / "C17_0_DEPENDENCY_AUDIT.json", {
        "stage": "C17-0",
        "dependencies": {k: {"path": str(p.relative_to(ROOT)), "exists": p.exists(), "sha256": sha256_file(p) if p.exists() and p.is_file() else None} for k, p in deps.items()},
        "official_val_used": False,
    })
    write_json(OUT0 / "C17_0_REPRODUCIBILITY_MANIFEST.json", {
        "stage": "C17-0",
        "mode": mode,
        "seed": seed,
        "device": str(DEVICE),
        "torch_cuda_available": torch.cuda.is_available(),
        "gpu": sh("nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader || true").splitlines(),
        "python": sh("/home/a/miniconda3/envs/conquer-rlem/bin/python --version"),
        "git_branch_vv": sh("git branch -vv"),
        "git_log": sh("git log --oneline --decorate -5"),
    })
    write_text(OUT0 / "C17_0_PROTOCOL.md", f"""# C17-0 Protocol

status: `{status}`

C17 starts from C16's accepted BMN localizer result and tests train-only native
VCMR integration. official validation is forbidden and was not run.

The promoted official system remains `{PROMOTED}`.
""")
    write_text(OUT0 / "C17_0_C16_ACCEPTANCE.md", f"""# C17-0 C16 Acceptance

- C16-2: `{c16_best.get('status')}`
- C16-3: `{c16_rob.get('status')}`
- C16-4: `{c16_int.get('status')}`
- C16-5: `{c16_next.get('status')}`

C16 is accepted as a strong localizer direction, not as a promoted official
system.
""")
    write_text(OUT0 / "C17_0_FORBIDDEN_ACTIONS_AUDIT.md", """# C17-0 Forbidden Actions Audit

- official validation: not run
- official prediction pool: not read
- evaluator/NMS official logic: not modified
- pseudo_official_holdout model selection: not used
- C7-B6 fixed prediction pool as C17 candidates: not used
- raw TVR video assumption: false
""")
    return rec


def stage_c17_1(mode: str, seed: int) -> Dict[str, Any]:
    require_c17_0_ready()
    OUT1.mkdir(parents=True, exist_ok=True)
    cfg = mode_cfg(mode)
    schema = {
        "input_columns": SCORE_COLUMNS,
        "metrics": [
            "VCMR R@1/5/10/100 @ IoU 0.5/0.7",
            "VR recall @ 1/5/10/100",
            "GT-video-only SVMR-like top50/top100 @ IoU 0.5/0.7",
            "wrong-video high-score rate",
            "correct-video wrong-span rate",
            "duplicate/invalid/timestamp counts",
            "query-type/duration/video-rank breakdowns",
        ],
        "nms_threshold": NMS_THRESHOLD,
        "official_val_used": False,
    }
    sanity = {
        "perfect_oracle_score_sanity": "implemented in evaluator by direct gt span semantics; exercised during C17-4 oracle rows",
        "random_score_sanity": "implemented in C17-4 baseline as random_uniform",
        "c12_5t_t2_replay_sanity": "implemented through T2 candidate scores over retrieved videos",
        "c16_bmn_map_only_sanity": "implemented through BMN map-only formula over retrieved videos",
        "timestamp_sanity": "idx_to_ts/ts_to_idx from C12 native span generation reused",
        "nms_sanity": f"train-only per-query/video temporal suppression at threshold {NMS_THRESHOLD}; official NMS not modified",
    }
    missing = ["C7-B6 train-only replay is only included if compatible local rows are present later"]
    if int(cfg["top_videos"]) < 100:
        status = "C17_TRAIN_ONLY_EVALUATOR_PARTIAL"
        missing.append("retriever top100 is not covered in this mode, so VR@100 is bounded by the configured replay pool")
    else:
        status = "C17_TRAIN_ONLY_EVALUATOR_READY"
    rec = {
        "stage": "C17-1",
        "status": status,
        "mode": mode,
        "evaluator_kind": "minimal native train-only VCMR evaluator",
        "missing_capability": missing,
        "metric_schema_hash": stable_hash(schema),
        "official_val_used": False,
        "pseudo_official_holdout_used_for_selection": False,
    }
    write_text(OUT1 / "C17_1_EVALUATOR_PLAN.md", """# C17-1 Evaluator Plan

Build final video-span rankings from train-only candidate rows, apply historical
temporal suppression per query/video, then compute VCMR R@K@IoU, VR recall,
GT-video-only span recall, and error breakdowns.

Medium mode replays retriever top100 videos, so VR@100 and VCMR@100 are
computed from an actual multi-video candidate pool, not GT-video localizer-only
coverage.
""")
    write_json(OUT1 / "C17_1_EVALUATOR_IMPLEMENTATION_AUDIT.json", rec)
    write_json(OUT1 / "C17_1_VCMR_METRIC_SCHEMA.json", schema)
    write_json(OUT1 / "C17_1_REPLAY_SANITY_RESULTS.json", sanity)
    write_json(OUT1 / "C17_1_TIMESTAMP_AND_NMS_AUDIT.json", {
        "timestamp_mapping": "C12 idx_to_ts/ts_to_idx reused",
        "nms_policy": "per query/video temporal suppression in train-only replay",
        "nms_threshold": NMS_THRESHOLD,
        "official_nms_modified": False,
    })
    write_json(OUT1 / "C17_1_EVALUATOR_DECISION.json", rec)
    write_text(OUT1 / "C17_1_EVALUATOR_DECISION.md", f"""# C17-1 Evaluator Decision

status: `{status}`

The evaluator can compute native train-only VCMR metrics over final video-span
rankings. Medium/full mode use retriever top100 videos; smoke remains a bounded
implementation check.
""")
    return rec


def stage_c17_2(mode: str, seed: int, force: bool = False) -> Tuple[Dict[str, Any], pd.DataFrame]:
    require_c17_0_ready()
    OUT2.mkdir(parents=True, exist_ok=True)
    cfg = mode_cfg(mode)
    cache = local_score_cache_path(mode)
    audit_cache = local_score_audit_path(mode)
    if cache.exists() and audit_cache.exists() and not force:
        df = pd.read_parquet(cache)
        audits = load_json(audit_cache, {})
    else:
        df, audits = generate_score_table(mode, seed, ["calib_select", "calib_holdout"])
        df.to_parquet(cache, index=False)
        write_json(audit_cache, audits)
    missing_bmn = int(df["bmn_final_score"].isna().sum()) if len(df) else 0
    missing_t2 = int(df["t2_score"].isna().sum()) if len(df) else 0
    invalid = int(((df["span_end"] <= df["span_start"]) | df["span_start"].isna() | df["span_end"].isna()).sum()) if len(df) else 0
    dup = int(df.duplicated(["split", "seed", "query_id", "video_id", "span_start", "span_end"]).sum()) if len(df) else 0
    status = "C17_FULL_BMN_SCORE_READY" if len(df) and missing_bmn == 0 and invalid == 0 and int(cfg["top_videos"]) >= 100 else "C17_FULL_BMN_SCORE_PARTIAL"
    audit = {
        "stage": "C17-2",
        "status": status,
        "mode": mode,
        "row_count": int(len(df)),
        "missing_bmn_score_count": missing_bmn,
        "missing_t2_score_count": missing_t2,
        "duplicate_candidate_count": dup,
        "invalid_span_count": invalid,
        "nan_inf_count": int(np.isinf(df.select_dtypes(include=[np.number]).to_numpy()).sum()) if len(df) else 0,
        "score_distribution_by_split": df.groupby("split")[["bmn_final_score", "t2_score", "retriever_score"]].describe().to_dict() if len(df) else {},
        "score_distribution_by_query_type": df.groupby("query_type")[["bmn_final_score", "t2_score"]].mean(numeric_only=True).to_dict() if len(df) else {},
        "score_distribution_by_duration_bucket": df.groupby("duration_bucket")[["bmn_final_score", "t2_score"]].mean(numeric_only=True).to_dict() if len(df) else {},
        "calib_select_vs_calib_holdout_distribution_shift": {
            "bmn_mean_delta": float(df[df.split == "calib_holdout"]["bmn_final_score"].mean() - df[df.split == "calib_select"]["bmn_final_score"].mean()) if len(df) else None,
            "t2_mean_delta": float(df[df.split == "calib_holdout"]["t2_score"].mean() - df[df.split == "calib_select"]["t2_score"].mean()) if len(df) else None,
        },
        "pseudo_official_holdout_used_for_selection": False,
        "official_val_used": False,
        "medium_topk_policy": f"{mode} exports retriever top{cfg['top_videos']} videos",
    }
    manifest = {
        "stage": "C17-2",
        "mode": mode,
        "local_score_table": str(cache),
        "local_score_table_sha256": sha256_file(cache) if cache.exists() else None,
        "committed_sample": "c17_2_full_bmn_score_export/C17_2_SCORE_SAMPLE.parquet",
        "row_count": int(len(df)),
        "schema_hash": audits.get("schema_hash"),
        "seeds": audits.get("seeds"),
        "checkpoints": audits.get("bmn_checkpoints"),
        "large_full_parquet_committed": False,
    }
    sample = df.head(min(5000, len(df))).copy()
    sample.to_parquet(OUT2 / "C17_2_SCORE_SAMPLE.parquet", index=False)
    write_text(OUT2 / "C17_2_FULL_BMN_SCORE_PLAN.md", f"""# C17-2 Full BMN Score Plan

mode: `{mode}`

Build a candidate-level train-only score table from retriever top videos,
C16 B7 BMN span maps, and C12-5T T2 scores. The local full parquet is retained
for reproducibility but should not be committed as a large prediction artifact.
""")
    write_json(OUT2 / "C17_2_FULL_BMN_RESULTS_BY_SEED.json", audits)
    write_json(OUT2 / "C17_2_FULL_BMN_SCORE_SCHEMA.json", {"columns": SCORE_COLUMNS, "schema_hash": audits.get("schema_hash"), "mode": mode})
    write_json(OUT2 / "C17_2_FULL_BMN_SCORE_MANIFEST.json", manifest)
    write_json(OUT2 / "C17_2_FULL_BMN_SCORE_AUDIT.json", audit)
    write_json(OUT2 / "C17_2_FULL_BMN_DECISION.json", audit)
    write_text(OUT2 / "C17_2_FULL_BMN_DECISION.md", f"""# C17-2 Full BMN Score Decision

status: `{status}`

Rows: `{len(df)}`. The committed parquet is a small sample only; the bounded
local table is stored at `{cache}` on the local NVMe-backed cache path when
available. It is not an official prediction and is not used for official
validation.
""")
    return audit, df


def stage_c17_3(mode: str, seed: int, df: pd.DataFrame | None = None, force: bool = False) -> Tuple[Dict[str, Any], pd.DataFrame]:
    require_c17_0_ready()
    OUT3.mkdir(parents=True, exist_ok=True)
    if df is None:
        _audit, df = stage_c17_2(mode, seed, force=force)
    formulas = build_formula_space()
    results, scored = evaluate_formulas(df, formulas)
    best = results["best_formula_name"]
    best_res = results["formula_results"][best]
    hold = best_res["calib_holdout"]["summary"]
    t2_hold = results["formula_results"]["B_T2_only"]["calib_holdout"]["summary"]
    bmn_hold = results["formula_results"]["C_BMN_map_only"]["calib_holdout"]["summary"]
    no_bmn_best = max(
        (v for k, v in results["formula_results"].items() if v["formula"].get("beta", 0.0) == 0.0),
        key=lambda x: score_for_selection(x["calib_select"]),
    )
    bmn_gain = float(hold["VCMR_R@100_IoU0.7"] - t2_hold["VCMR_R@100_IoU0.7"])
    no_bmn_drop = float(hold["VCMR_R@100_IoU0.7"] - no_bmn_best["calib_holdout"]["summary"]["VCMR_R@100_IoU0.7"])
    wrong_delta = float(hold["wrong_video_high_score_rate"] - t2_hold["wrong_video_high_score_rate"])
    if bmn_gain > 0.0 and no_bmn_drop > 0.0 and wrong_delta <= 3.0:
        status = "C17_HYBRID_PROMISING"
    elif bmn_hold["VCMR_R@100_IoU0.7"] > t2_hold["VCMR_R@100_IoU0.7"] and best_res["formula"].get("gamma", 0.0) < 0.15:
        status = "C17_BMN_LOCALIZER_ONLY"
    elif best_res["formula"].get("gamma", 0.0) >= 0.55:
        status = "C17_T2_STILL_DOMINATES"
    elif hold["VR_R@100"] < 90.0:
        status = "C17_RETRIEVER_CALIBRATION_BOTTLENECK"
    else:
        status = "C17_HYBRID_INCONCLUSIVE"
    decision = {
        "stage": "C17-3",
        "status": status,
        "mode": mode,
        "best_formula_name": best,
        "best_formula": best_res["formula"],
        "best_calib_select": best_res["calib_select"],
        "best_calib_holdout": best_res["calib_holdout"],
        "t2_only_calib_holdout": results["formula_results"]["B_T2_only"]["calib_holdout"],
        "bmn_only_calib_holdout": results["formula_results"]["C_BMN_map_only"]["calib_holdout"],
        "bmn_gain_vs_t2_R100_IoU0.7": bmn_gain,
        "hybrid_vs_no_bmn_R100_IoU0.7": no_bmn_drop,
        "wrong_video_rate_delta_vs_t2": wrong_delta,
        "pseudo_official_holdout_used_for_selection": False,
        "official_val_used": False,
    }
    write_json(OUT3 / "C17_3_HYBRID_SEARCH_SPACE.json", {"formulas": formulas})
    write_json(OUT3 / "C17_3_HYBRID_RESULTS.json", results)
    write_json(OUT3 / "C17_3_BMN_CONTRIBUTION_AUDIT.json", {
        "hybrid_vs_no_bmn_R100_IoU0.7": no_bmn_drop,
        "bmn_only_vs_t2_only_R100_IoU0.7": float(bmn_hold["VCMR_R@100_IoU0.7"] - t2_hold["VCMR_R@100_IoU0.7"]),
        "retriever_bmn_vs_retriever_t2_R100_IoU0.7": float(results["formula_results"]["D_retriever_BMN"]["calib_holdout"]["summary"]["VCMR_R@100_IoU0.7"] - results["formula_results"]["E_retriever_T2"]["calib_holdout"]["summary"]["VCMR_R@100_IoU0.7"]),
    })
    write_json(OUT3 / "C17_3_T2_DOMINANCE_AUDIT.json", {
        "best_gamma": best_res["formula"].get("gamma", 0.0),
        "drop_when_bmn_removed_R100_IoU0.7": no_bmn_drop,
        "drop_when_t2_removed_R100_IoU0.7": float(hold["VCMR_R@100_IoU0.7"] - bmn_hold["VCMR_R@100_IoU0.7"]),
    })
    scored["retriever_localizer_conflict"] = np.where((scored["retriever_norm"] > 0.80) & (scored["bmn_norm"] < 0.20), "high_retriever_low_bmn", np.where((scored["retriever_norm"] < 0.20) & (scored["bmn_norm"] > 0.80), "low_retriever_high_bmn", "none"))
    write_json(OUT3 / "C17_3_RETRIEVER_LOCALIZER_CALIBRATION_AUDIT.json", {
        "conflict_counts": scored["retriever_localizer_conflict"].value_counts().to_dict(),
        "wrong_video_high_bmn_cases_sample": scored[(scored["video_id"] != scored["gt_video_id"]) & (scored["bmn_norm"] > 0.90)].head(50)[["query_id", "video_id", "gt_video_id", "bmn_norm", "retriever_norm", "t2_norm"]].to_dict("records"),
    })
    scored.head(min(5000, len(scored))).to_parquet(OUT3 / "C17_3_SCORE_SAMPLE.parquet", index=False)
    write_json(OUT3 / "C17_3_HYBRID_DECISION.json", decision)
    write_text(OUT3 / "C17_3_HYBRID_DECISION.md", f"""# C17-3 Hybrid Decision

status: `{status}`

Best formula: `{best}`.

Holdout VCMR R@100 IoU0.7: `{hold.get('VCMR_R@100_IoU0.7')}`.
T2-only holdout VCMR R@100 IoU0.7: `{t2_hold.get('VCMR_R@100_IoU0.7')}`.

pseudo_official_holdout was not used for selection. official was not run.
""")
    return decision, scored


def stage_c17_4(mode: str, seed: int, scored: pd.DataFrame | None = None, force: bool = False) -> Dict[str, Any]:
    require_c17_0_ready()
    OUT4.mkdir(parents=True, exist_ok=True)
    hybrid = load_json(OUT3 / "C17_3_HYBRID_RESULTS.json", {})
    best = hybrid.get("best_formula_name")
    formula_results = hybrid.get("formula_results", {})
    if scored is None:
        scored, load_audit = load_eval_score_table(mode)
        fill_audit = add_normalized_scores_inplace(scored)
    else:
        load_audit = {"source": "in_memory_scored_table", "row_count": int(len(scored))}
        fill_audit = {"source": "already_normalized"}
    formula_defs = {
        "C12_5T_T2_baseline": formula_results.get("B_T2_only", {}).get("formula", {"name": "B_T2_only", "family": "B_T2_only", "gamma": 1.0}),
        "C16_BMN_map_only": formula_results.get("C_BMN_map_only", {}).get("formula", {"name": "C_BMN_map_only", "family": "C_BMN_map_only", "beta": 1.0}),
        "C17_best_hybrid": formula_results.get(best, {}).get("formula"),
        "C17_retriever_BMN": formula_results.get("D_retriever_BMN", {}).get("formula", {"name": "D_retriever_BMN", "family": "D_retriever_BMN", "alpha": 0.45, "beta": 0.55}),
        "C17_retriever_BMN_T2": formula_results.get(best, {}).get("formula"),
    }
    results: Dict[str, Any] = {}
    for name, formula in formula_defs.items():
        if not formula:
            continue
        assign_formula_score(scored, formula, "eval_score")
        results[name] = evaluate_vcmr(scored, "eval_score")
    rng = np.random.default_rng(seed)
    scored["eval_score"] = rng.random(len(scored), dtype=np.float32)
    results["random_score_sanity"] = evaluate_vcmr(scored, "eval_score")
    assign_oracle_score(scored, "eval_score")
    results["oracle_upper_bound_in_candidate_pool"] = evaluate_vcmr(scored, "eval_score")
    if formula_defs.get("C17_best_hybrid"):
        assign_formula_score(scored, formula_defs["C17_best_hybrid"], "eval_score")
    seed_robustness_from_stage4: Dict[str, Any] = {}
    for s, sdf in scored[scored["split"] == "calib_holdout"].groupby("seed", observed=True):
        seed_robustness_from_stage4[str(int(s))] = evaluate_vcmr(sdf, "eval_score")["summary"]
    t2 = results.get("C12_5T_T2_baseline", {}).get("summary", {})
    hy = results.get("C17_best_hybrid", {}).get("summary", {})
    delta = float(hy.get("VCMR_R@100_IoU0.7", 0.0) - t2.get("VCMR_R@100_IoU0.7", 0.0))
    wrong_delta = float(hy.get("wrong_video_high_score_rate", 0.0) - t2.get("wrong_video_high_score_rate", 0.0))
    if delta > 0.0 and wrong_delta <= 3.0 and hy.get("VCMR_R@1_IoU0.7", 0.0) >= t2.get("VCMR_R@1_IoU0.7", 0.0):
        status = "C17_NATIVE_VCMR_PROMISING"
    elif results.get("C16_BMN_map_only", {}).get("summary", {}).get("GT_VIDEO_TOP100_IoU0.7", 0.0) > t2.get("GT_VIDEO_TOP100_IoU0.7", 0.0) and delta <= 0.0:
        status = "C17_NATIVE_VCMR_LOCALIZER_GAIN_NOT_TRANSLATED"
    elif hy.get("VCMR_R@100_IoU0.7", 0.0) < t2.get("VCMR_R@100_IoU0.7", 0.0):
        status = "C17_NATIVE_VCMR_WEAK"
    else:
        status = "C17_NATIVE_VCMR_INCONCLUSIVE"
    improved, harmed = [], []
    t2_sample = {r["query_id"]: r for r in results.get("C12_5T_T2_baseline", {}).get("records_sample", [])}
    hy_sample = {r["query_id"]: r for r in results.get("C17_best_hybrid", {}).get("records_sample", [])}
    for qid, hrec in hy_sample.items():
        trec = t2_sample.get(qid)
        if not trec:
            continue
        if hrec.get("VCMR_R@100_IoU0.7") and not trec.get("VCMR_R@100_IoU0.7") and len(improved) < 25:
            improved.append(int(qid))
        if trec.get("VCMR_R@100_IoU0.7") and not hrec.get("VCMR_R@100_IoU0.7") and len(harmed) < 25:
            harmed.append(int(qid))
    decision = {
        "stage": "C17-4",
        "status": status,
        "mode": mode,
        "native_vcmr_results": results,
        "delta_vs_t2_R100_IoU0.7": delta,
        "wrong_video_delta_vs_t2": wrong_delta,
        "improved_query_examples": improved,
        "harmed_query_examples": harmed,
        "load_audit": load_audit,
        "fill_audit": fill_audit,
        "seed_robustness_from_stage4": seed_robustness_from_stage4,
        "official_val_used": False,
        "pseudo_official_holdout_used_for_selection": False,
    }
    write_text(OUT4 / "C17_4_NATIVE_INTEGRATION_PLAN.md", """# C17-4 Native VCMR Integration Plan

Replay the native train-only pipeline: retriever topK videos, BMN span map,
optional T2 safety signal, legal span filter, per-video temporal NMS, final
video-span ranking, and C17 train-only VCMR evaluator.
""")
    write_json(OUT4 / "C17_4_NATIVE_VCMR_RESULTS.json", results)
    write_json(OUT4 / "C17_4_BASELINE_COMPARISON.json", {
        "delta_vs_t2_R100_IoU0.7": delta,
        "wrong_video_delta_vs_t2": wrong_delta,
        "baselines": list(results.keys()),
    })
    write_json(OUT4 / "C17_4_ERROR_BREAKDOWN.json", {
        "hybrid_summary": hy,
        "t2_summary": t2,
        "improved_query_examples": improved,
        "harmed_query_examples": harmed,
    })
    write_json(OUT4 / "C17_4_VIDEO_RANK_BREAKDOWN.json", results.get("C17_best_hybrid", {}).get("video_rank_breakdown", {}))
    write_json(OUT4 / "C17_4_DURATION_QUERYTYPE_BREAKDOWN.json", {
        "duration": results.get("C17_best_hybrid", {}).get("duration_breakdown", {}),
        "query_type": results.get("C17_best_hybrid", {}).get("query_type_breakdown", {}),
    })
    write_json(OUT4 / "C17_4_INTEGRATION_DECISION.json", decision)
    write_text(OUT4 / "C17_4_INTEGRATION_DECISION.md", f"""# C17-4 Integration Decision

status: `{status}`

Delta vs C12-5T T2, VCMR R@100 IoU0.7: `{delta}`.

This is train-only native replay. official was not run.
""")
    return decision


def stage_c17_5(mode: str, seed: int) -> Dict[str, Any]:
    require_c17_0_ready()
    OUT5.mkdir(parents=True, exist_ok=True)
    hybrid = load_json(OUT3 / "C17_3_HYBRID_DECISION.json", {})
    integration = load_json(OUT4 / "C17_4_INTEGRATION_DECISION.json", {})
    results = load_json(OUT3 / "C17_3_HYBRID_RESULTS.json", {})
    best_name = results.get("best_formula_name")
    best = results.get("formula_results", {}).get(best_name, {}) if best_name else {}
    select = best.get("calib_select", {}).get("summary", {})
    hold = best.get("calib_holdout", {}).get("summary", {})
    overfit = float(select.get("VCMR_R@100_IoU0.7", 0.0) - hold.get("VCMR_R@100_IoU0.7", 0.0))
    seed_stats = integration.get("seed_robustness_from_stage4", {})
    direction = [v.get("VCMR_R@100_IoU0.7", 0.0) for v in seed_stats.values()]
    robust = len(direction) >= 2 and max(direction) - min(direction) <= 5.0
    if integration.get("status") == "C17_NATIVE_VCMR_PROMISING" and robust and overfit <= 5.0:
        status = "C17_ROBUSTNESS_PASS"
    elif integration.get("status") in {"C17_NATIVE_VCMR_PROMISING", "C17_NATIVE_VCMR_INCONCLUSIVE"}:
        status = "C17_ROBUSTNESS_PARTIAL"
    else:
        status = "C17_ROBUSTNESS_FAIL"
    rec = {
        "stage": "C17-5",
        "status": status,
        "mode": mode,
        "medium_vs_full": "full not run automatically; medium-bounded trend only",
        "seed_robustness": seed_stats,
        "select_to_holdout_overfit_R100_IoU0.7": overfit,
        "pseudo_official_holdout_used_for_selection": False,
        "pseudo_official_onelook_diagnostic": "not run",
        "official_val_used": False,
    }
    write_json(OUT5 / "C17_5_FULL_MEDIUM_CONSISTENCY.json", {"medium": hold, "full": None, "status": "not_run"})
    write_json(OUT5 / "C17_5_SEED_ROBUSTNESS.json", seed_stats)
    write_json(OUT5 / "C17_5_QUERY_DURATION_ROBUSTNESS.json", {
        "query_type": best.get("calib_holdout", {}).get("query_type_breakdown", {}),
        "duration": best.get("calib_holdout", {}).get("duration_breakdown", {}),
    })
    write_json(OUT5 / "C17_5_DISTRIBUTION_SHIFT_AUDIT.json", {"select_to_holdout_overfit_R100_IoU0.7": overfit})
    write_json(OUT5 / "C17_5_PSEUDO_OFFICIAL_ONELOOK_DIAGNOSTIC.json", {"status": "not_run", "used_for_selection": False})
    write_json(OUT5 / "C17_5_ROBUSTNESS_DECISION.json", rec)
    write_text(OUT5 / "C17_5_ROBUSTNESS_DECISION.md", f"""# C17-5 Robustness Decision

status: `{status}`

Full mode and pseudo official one-look were not run. Selection used
calib_select only; calib_holdout was report-only.
""")
    return rec


def stage_c17_6(mode: str, seed: int) -> Dict[str, Any]:
    OUT6.mkdir(parents=True, exist_ok=True)
    s0 = load_json(OUT0 / "C17_0_PROTOCOL.json", {})
    s1 = load_json(OUT1 / "C17_1_EVALUATOR_DECISION.json", {})
    s2 = load_json(OUT2 / "C17_2_FULL_BMN_DECISION.json", {})
    s3 = load_json(OUT3 / "C17_3_HYBRID_DECISION.json", {})
    s4 = load_json(OUT4 / "C17_4_INTEGRATION_DECISION.json", {})
    s5 = load_json(OUT5 / "C17_5_ROBUSTNESS_DECISION.json", {})
    ready = (
        s1.get("status") == "C17_TRAIN_ONLY_EVALUATOR_READY"
        and s2.get("status") == "C17_FULL_BMN_SCORE_READY"
        and s3.get("status") == "C17_HYBRID_PROMISING"
        and s4.get("status") == "C17_NATIVE_VCMR_PROMISING"
        and s5.get("status") in {"C17_ROBUSTNESS_PASS", "C17_ROBUSTNESS_PARTIAL"}
    )
    if ready:
        final = "C17_READY_FOR_ONE_SHOT_OFFICIAL_REVIEW"
    elif s3.get("status") == "C17_HYBRID_PROMISING" and s5.get("status") != "C17_ROBUSTNESS_PASS":
        final = "C17_CONTINUE_HYBRID_TRAIN_ONLY"
    elif s4.get("status") == "C17_NATIVE_VCMR_LOCALIZER_GAIN_NOT_TRANSLATED":
        final = "C17_NEED_RETRIEVER_LOCALIZER_CALIBRATION"
    elif s3.get("status") == "C17_BMN_LOCALIZER_ONLY":
        final = "C17_NEED_BSN_FALLBACK"
    elif s4.get("status") in {"C17_NATIVE_VCMR_WEAK", "C17_NATIVE_VCMR_INCONCLUSIVE"}:
        final = "C17_NEED_RETRIEVER_LOCALIZER_CALIBRATION"
    else:
        final = "C17_CONTINUE_HYBRID_TRAIN_ONLY"
    manifest = load_json(OUT2 / "C17_2_FULL_BMN_SCORE_MANIFEST.json", {})
    rec = {
        "stage": "C17-6",
        "final_decision": final,
        "status": final,
        "mode": mode,
        "seed": seed,
        "c17_0_status": s0.get("status"),
        "c17_1_evaluator_status": s1.get("status"),
        "c17_2_bmn_score_status": s2.get("status"),
        "c17_3_hybrid_status": s3.get("status"),
        "c17_4_native_integration_status": s4.get("status"),
        "c17_5_robustness_status": s5.get("status"),
        "official_val_used": False,
        "pseudo_official_holdout_used_for_selection": False,
        "evaluator_modified": False,
        "official_nms_modified": False,
        "schema_hash": manifest.get("schema_hash"),
        "candidate_pool_hash": manifest.get("local_score_table_sha256"),
        "score_manifest_hash": stable_hash(manifest),
        "best_config": s3.get("best_formula_name"),
        "selected_weights": s3.get("best_formula"),
        "command_lines": [f"run_c17_bmn_t2_native_vcmr_integration.py --stage all --mode {mode} --seed {seed}"],
        "seed_list": seed_list(seed, mode),
        "missing_artifacts": s1.get("missing_capability", []),
        "current_promoted_system": PROMOTED,
    }
    write_json(OUT6 / "C17_6_FREEZE_REVIEW.json", rec)
    write_json(OUT6 / "C17_6_OFFICIAL_READINESS_PACKET.json", {**rec, "official_may_be_run_by_this_script": False})
    write_json(OUT6 / "C17_6_NEXT_STEP_DECISION.json", rec)
    write_text(OUT6 / "C17_6_FREEZE_REVIEW.md", f"""# C17-6 Freeze Review

final_decision: `{final}`

C17 remains train-only. official validation was not run.
""")
    write_text(OUT6 / "C17_6_OFFICIAL_READINESS_PACKET.md", f"""# C17-6 Official Readiness Packet

decision: `{final}`

This packet is not an authorization to run official validation. C17-1 is
`{s1.get('status')}` and C17-2 is `{s2.get('status')}`; any remaining PARTIAL
status or robustness gap must be reviewed separately before official use.
""")
    write_text(OUT6 / "C17_6_RISK_REGISTER.md", """# C17-6 Risk Register

- Full mode consistency was not run automatically.
- Raw TVR video is absent.
- C7-B6 remains the promoted official system.
""")
    write_text(OUT6 / "C17_6_NEXT_STEP_DECISION.md", f"""# C17-6 Next Step Decision

decision: `{final}`

Recommended next action: continue train-only retriever/localizer calibration or
run an explicit full-scale train-only replay before any separate official
review.
""")
    return rec


def print_terminal_summary(final: Dict[str, Any]) -> None:
    s3 = load_json(OUT3 / "C17_3_HYBRID_DECISION.json", {})
    s4 = load_json(OUT4 / "C17_4_INTEGRATION_DECISION.json", {})
    hy = s4.get("native_vcmr_results", {}).get("C17_best_hybrid", {}).get("summary", {})
    t2 = s4.get("native_vcmr_results", {}).get("C12_5T_T2_baseline", {}).get("summary", {})
    print("C17 SUMMARY")
    print(f"branch: {sh('git branch --show-current')}")
    print(f"commit: {sh('git rev-parse --short HEAD')}")
    print(f"C17-0 status: {final.get('c17_0_status')}")
    print(f"C17-1 evaluator status: {final.get('c17_1_evaluator_status')}")
    print(f"C17-2 BMN score status: {final.get('c17_2_bmn_score_status')}")
    print(f"C17-3 hybrid status: {final.get('c17_3_hybrid_status')}")
    print(f"C17-4 native integration status: {final.get('c17_4_native_integration_status')}")
    print(f"C17-5 robustness status: {final.get('c17_5_robustness_status')}")
    print(f"C17-6 final decision: {final.get('final_decision')}")
    print(f"best hybrid formula/weights: {s3.get('best_formula')}")
    print(f"key VCMR train-only metrics: R@1@0.7={hy.get('VCMR_R@1_IoU0.7')} R@10@0.7={hy.get('VCMR_R@10_IoU0.7')} R@100@0.7={hy.get('VCMR_R@100_IoU0.7')}")
    print(f"vs C12-5T T2 delta R@100@0.7: {float(hy.get('VCMR_R@100_IoU0.7', 0.0) - t2.get('VCMR_R@100_IoU0.7', 0.0))}")
    print("pseudo_official_holdout used for selection: false")
    print("official was not run: true")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", default="all", choices=["c17_0", "c17_1", "c17_2", "c17_3", "c17_4", "c17_5", "c17_6", "all"])
    parser.add_argument("--mode", default="medium", choices=["smoke", "medium", "full"])
    parser.add_argument("--seed", default=2026, type=int)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    df: pd.DataFrame | None = None
    scored: pd.DataFrame | None = None
    if args.stage in {"c17_0", "all"}:
        stage_c17_0(args.mode, args.seed)
    if args.stage in {"c17_1", "all"}:
        stage_c17_1(args.mode, args.seed)
    if args.stage in {"c17_2", "all"}:
        _audit, df = stage_c17_2(args.mode, args.seed, force=args.force)
    if args.stage in {"c17_3", "all"}:
        _decision, scored = stage_c17_3(args.mode, args.seed, df=df, force=args.force)
        if args.stage == "all":
            df = None
            scored = None
            gc.collect()
    if args.stage in {"c17_4", "all"}:
        stage_c17_4(args.mode, args.seed, scored=scored, force=args.force)
    if args.stage in {"c17_5", "all"}:
        stage_c17_5(args.mode, args.seed)
    final: Dict[str, Any] | None = None
    if args.stage in {"c17_6", "all"}:
        final = stage_c17_6(args.mode, args.seed)
    if final:
        print_terminal_summary(final)


if __name__ == "__main__":
    main()
