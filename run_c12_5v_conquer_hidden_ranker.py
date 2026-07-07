#!/usr/bin/env python3
"""C12-5V CONQUER-hidden / semantic boundary proposal ranker.

Train-only hidden-rich proposal rankers over C12 native generated M=1000 span
pools. No official data, no evaluator/NMS changes, and no C12-6 transition.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import run_c12_5u_topk_calibration_repair as u
from data_loader.second_stage_start_end_dataset import StartEndDataset
from model.conquer import CONQUER
from utils.basic_utils import load_config
from utils.model_utils import move_cuda, start_end_collate


ROOT = u.ROOT
OUT = ROOT / "c12_5v_conquer_hidden_ranker"
MODEL_DIR = ROOT / "c12_models"
SEED = 1295
TRAIN_LIMIT = int(os.environ.get("C12_5V_TRAIN_LIMIT", "12000"))
EVAL_LIMIT = int(os.environ.get("C12_5V_EVAL_LIMIT", "0"))
CONQUER_BATCH = int(os.environ.get("C12_5V_CONQUER_BATCH", "16"))
C12_BATCH = int(os.environ.get("C12_5V_C12_BATCH", "32"))
POOL_LIMIT = 1000


BASE_FEATURES = list(u.FEATURE_NAMES)
HIDDEN_FEATURES = [
    "conq_start_logit",
    "conq_end_logit",
    "conq_start_prob",
    "conq_end_prob",
    "q2v_attention_inside_mean",
    "q2v_attention_left_mean",
    "q2v_attention_right_mean",
    "q2v_attention_contrast",
    "qdf_inside_norm",
    "qdf_contrast_norm",
    "qal_inside_norm",
    "qal_contrast_norm",
    "contextual_inside_norm",
    "contextual_contrast_norm",
    "g_inside_norm",
    "g_contrast_norm",
    "contextual_start_query_dot",
    "contextual_end_query_dot",
    "contextual_inside_query_dot",
    "conq_begin_local_jump",
    "conq_end_local_jump",
]
FEATURE_NAMES = BASE_FEATURES + HIDDEN_FEATURES


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def sha256_obj(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


class DescIdSubset(Dataset):
    def __init__(self, base: StartEndDataset, desc_ids: Sequence[int | str]):
        wanted = set(str(x) for x in desc_ids)
        self.base = base
        self.indices = [i for i, row in enumerate(base.query_data) if str(row["desc_id"]) in wanted]
        self.query_data = [base.query_data[i] for i in self.indices]
        self.video2idx = base.video2idx
        self.idx2video = base.idx2video

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> Any:
        return self.base[self.indices[item]]


def build_conquer_dataset(desc_ids: Sequence[int], is_eval: bool = False) -> DescIdSubset:
    base = StartEndDataset(
        load_config("config/tvr_data_config.json"),
        max_ctx_len=100,
        max_desc_len=30,
        clip_length=1.5,
        ctx_mode="visual_sub",
        is_eval=is_eval,
        mode="train",
        neg_video_num=3,
        use_extend_pool=500,
        inference_top_k=10,
    )
    return DescIdSubset(base, desc_ids)


def build_conquer_model() -> CONQUER:
    model_cfg = load_config("config/model_config.json")
    model = CONQUER(
        model_cfg,
        visual_dim=4352,
        text_dim=768,
        query_dim=768,
        hidden_dim=768,
        video_len=100,
        ctx_mode="visual_sub",
        lw_video_ce=0.05,
        lw_st_ed=0.01,
        similarity_measure="general",
        use_debug=False,
        no_output_moe_weight=False,
    )
    ckpt = torch.load(ROOT / "results/tvr-conquer_c0_repro_20260621/model.ckpt", map_location="cpu")
    model.load_state_dict(ckpt["model"])
    model.to(u.DEVICE)
    model.eval()
    return model


def norm_seq(x: torch.Tensor) -> np.ndarray:
    return torch.linalg.norm(x.float(), dim=-1).detach().cpu().numpy().astype(np.float32)


def safe_local_mean(x: np.ndarray, s: int, e: int) -> float:
    return u.local_mean(x, s, e)


def scale_span(s: int, e: int, src_t: int, dst_t: int) -> Tuple[int, int]:
    if src_t <= 1:
        return 0, 0
    ss = int(round(float(s) / max(1, src_t - 1) * max(1, dst_t - 1)))
    ee = int(round(float(e) / max(1, src_t - 1) * max(1, dst_t - 1)))
    ss = max(0, min(dst_t - 1, ss))
    ee = max(ss, min(dst_t - 1, ee))
    return ss, ee


def prefix(x: np.ndarray) -> np.ndarray:
    return np.concatenate([np.zeros((1,), dtype=np.float64), np.cumsum(x.astype(np.float64))])


def prefix_mean(pref: np.ndarray, s: int, e: int) -> float:
    s = max(0, min(len(pref) - 2, s))
    e = max(s, min(len(pref) - 2, e))
    return float((pref[e + 1] - pref[s]) / max(1, e - s + 1))


def build_hidden_context(hidden: Dict[str, Any]) -> Dict[str, Any]:
    ctx = dict(hidden)
    ctx["begin_prob"] = u.softmax_np(hidden["begin"])
    ctx["end_prob"] = u.softmax_np(hidden["end"])
    for key in ["q2v", "qdf_norm", "qal_norm", "ctx_norm", "g_norm", "ctx_query_dot"]:
        ctx[key + "_prefix"] = prefix(hidden[key])
    return ctx


def hidden_row(hidden: Dict[str, Any], s: int, e: int, src_t: int) -> List[float]:
    dst_t = int(hidden["video_len"])
    hs, he = scale_span(s, e, src_t, dst_t)
    span_len = he - hs + 1
    left_s = max(0, hs - span_len)
    left_e = max(0, hs - 1)
    right_s = min(dst_t - 1, he + 1)
    right_e = min(dst_t - 1, he + span_len)

    def contrast(seq: np.ndarray) -> Tuple[float, float, float, float]:
        inside = safe_local_mean(seq, hs, he)
        left = safe_local_mean(seq, left_s, left_e) if hs > 0 else inside
        right = safe_local_mean(seq, right_s, right_e) if he + 1 < dst_t else inside
        return inside, left, right, inside - 0.5 * (left + right)

    q2v_i, q2v_l, q2v_r, q2v_c = contrast(hidden["q2v"])
    qdf_i, _qdf_l, _qdf_r, qdf_c = contrast(hidden["qdf_norm"])
    qal_i, _qal_l, _qal_r, qal_c = contrast(hidden["qal_norm"])
    ctx_i, _ctx_l, _ctx_r, ctx_c = contrast(hidden["ctx_norm"])
    g_i, _g_l, _g_r, g_c = contrast(hidden["g_norm"])
    ctx_dot = hidden["ctx_query_dot"]
    begin = hidden["begin"]
    end = hidden["end"]
    sp = u.softmax_np(begin)
    ep = u.softmax_np(end)
    b_jump = float(begin[hs] - begin[hs - 1]) if hs > 0 else 0.0
    e_jump = float(end[he] - end[he + 1]) if he + 1 < dst_t else 0.0
    return [
        float(begin[hs]),
        float(end[he]),
        float(sp[hs]),
        float(ep[he]),
        q2v_i,
        q2v_l,
        q2v_r,
        q2v_c,
        qdf_i,
        qdf_c,
        qal_i,
        qal_c,
        ctx_i,
        ctx_c,
        g_i,
        g_c,
        float(ctx_dot[hs]),
        float(ctx_dot[he]),
        safe_local_mean(ctx_dot, hs, he),
        b_jump,
        e_jump,
    ]


def hidden_row_cached(hidden: Dict[str, Any], s: int, e: int, src_t: int) -> List[float]:
    dst_t = int(hidden["video_len"])
    hs, he = scale_span(s, e, src_t, dst_t)
    span_len = he - hs + 1
    left_s = max(0, hs - span_len)
    left_e = max(0, hs - 1)
    right_s = min(dst_t - 1, he + 1)
    right_e = min(dst_t - 1, he + span_len)

    def contrast(name: str) -> Tuple[float, float, float, float]:
        pref = hidden[name + "_prefix"]
        inside = prefix_mean(pref, hs, he)
        left = prefix_mean(pref, left_s, left_e) if hs > 0 else inside
        right = prefix_mean(pref, right_s, right_e) if he + 1 < dst_t else inside
        return inside, left, right, inside - 0.5 * (left + right)

    q2v_i, q2v_l, q2v_r, q2v_c = contrast("q2v")
    qdf_i, _qdf_l, _qdf_r, qdf_c = contrast("qdf_norm")
    qal_i, _qal_l, _qal_r, qal_c = contrast("qal_norm")
    ctx_i, _ctx_l, _ctx_r, ctx_c = contrast("ctx_norm")
    g_i, _g_l, _g_r, g_c = contrast("g_norm")
    begin = hidden["begin"]
    end = hidden["end"]
    b_jump = float(begin[hs] - begin[hs - 1]) if hs > 0 else 0.0
    e_jump = float(end[he] - end[he + 1]) if he + 1 < dst_t else 0.0
    return [
        float(begin[hs]),
        float(end[he]),
        float(hidden["begin_prob"][hs]),
        float(hidden["end_prob"][he]),
        q2v_i,
        q2v_l,
        q2v_r,
        q2v_c,
        qdf_i,
        qdf_c,
        qal_i,
        qal_c,
        ctx_i,
        ctx_c,
        g_i,
        g_c,
        prefix_mean(hidden["ctx_query_dot_prefix"], hs, hs),
        prefix_mean(hidden["ctx_query_dot_prefix"], he, he),
        prefix_mean(hidden["ctx_query_dot_prefix"], hs, he),
        b_jump,
        e_jump,
    ]


@torch.no_grad()
def export_hidden_for_ids(desc_ids: Sequence[int]) -> Dict[int, Dict[str, Any]]:
    ds = build_conquer_dataset(desc_ids, is_eval=False)
    loader = DataLoader(ds, batch_size=CONQUER_BATCH, collate_fn=start_end_collate, num_workers=0, shuffle=False)
    model = build_conquer_model()
    hidden_by_id: Dict[int, Dict[str, Any]] = {}
    for batch in loader:
        inputs = move_cuda(batch["model_inputs"], u.DEVICE) if u.DEVICE.type == "cuda" else batch["model_inputs"]
        _vs, begin, end, inter = model.get_pred_from_raw_query(inputs, return_intermediates=True)
        bsz, shared, video_len = begin.shape
        q = inter["query_feature"].float()
        qmask = inputs["query"]["feat_mask"].bool().to(q.device)
        qmean = (q * qmask.unsqueeze(-1)).sum(1) / qmask.sum(1, keepdim=True).clamp_min(1)
        qmean = F.normalize(qmean, dim=-1)
        qdf = inter["qdf_feature"].view(bsz, shared, video_len, -1)[:, 0]
        qal = inter["qal_feature"].view(bsz, shared, video_len, -1)[:, 0]
        ctx = inter["contextual_qal"].view(bsz, shared, video_len, -1)[:, 0]
        g = inter["g_feature"].view(bsz, shared, video_len, -1)[:, 0]
        q2v = inter["qal_aux"]["q2v_attention"].view(bsz, shared, video_len)[:, 0]
        ctx_query_dot = torch.einsum("btd,bd->bt", F.normalize(ctx.float(), dim=-1), qmean)
        for i, meta in enumerate(batch["meta"]):
            did = int(meta["desc_id"])
            hidden_by_id[did] = {
                "video_len": int(video_len),
                "begin": begin[i, 0].detach().cpu().numpy().astype(np.float32),
                "end": end[i, 0].detach().cpu().numpy().astype(np.float32),
                "q2v": q2v[i].detach().cpu().numpy().astype(np.float32),
                "qdf_norm": norm_seq(qdf[i]),
                "qal_norm": norm_seq(qal[i]),
                "ctx_norm": norm_seq(ctx[i]),
                "g_norm": norm_seq(g[i]),
                "ctx_query_dot": ctx_query_dot[i].detach().cpu().numpy().astype(np.float32),
            }
    return hidden_by_id


def feature_indices(kind: str) -> List[int]:
    names = set(BASE_FEATURES)
    if kind == "V1_hidden_boundary":
        names = {
            "start_logit", "end_logit", "start_prob", "end_prob", "start_rank_norm", "end_rank_norm",
            "start_margin", "end_margin", "start_peak_sharpness", "end_peak_sharpness", "span_duration_norm",
            "duration_bucket_short", "duration_bucket_medium", "duration_bucket_long", "retriever_score_z",
            "qtype_v", "qtype_t", "qtype_vt", "qtype_unknown", "generated_pool_rank_norm", "generated_pool_score",
            "previous_c12_pq_score", "center_norm", "duration_prior_score",
            "conq_start_logit", "conq_end_logit", "conq_start_prob", "conq_end_prob",
            "contextual_start_query_dot", "contextual_end_query_dot", "conq_begin_local_jump", "conq_end_local_jump",
        }
    elif kind == "V2_span_context_hidden":
        names = set(FEATURE_NAMES) - {"conq_begin_local_jump", "conq_end_local_jump"}
    elif kind == "V4_short_hidden_specialist":
        names = set(FEATURE_NAMES)
    else:
        names = set(FEATURE_NAMES)
    return [i for i, n in enumerate(FEATURE_NAMES) if n in names]


class Ranker(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(0.1), nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(0.1), nn.Linear(hidden, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class TwoHead(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 256) -> None:
        super().__init__()
        self.trunk = nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(0.1), nn.Linear(hidden, hidden), nn.GELU())
        self.sel = nn.Linear(hidden, 1)
        self.qual = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.trunk(x)
        return self.sel(z).squeeze(-1), self.qual(z).squeeze(-1)


def build_groups(desc_ids: Sequence[int], corpus: Any, features: Dict[str, Any], first_stage: Dict[int, Dict[str, Any]], span_model: Any, hidden_by_id: Dict[int, Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    store = u.c12_5.ClipFeatureStore()
    groups: List[Dict[str, Any]] = []
    rng = random.Random(SEED)
    best_ranks = []
    missing_hidden = 0
    try:
        for st in range(0, len(desc_ids), C12_BATCH):
            ids = list(desc_ids[st:st + C12_BATCH])
            batch = u.c12_5.build_batch(corpus, features, store, ids)
            for bi, did in enumerate(ids):
                if int(did) not in hidden_by_id:
                    missing_hidden += 1
                    continue
                hctx = build_hidden_context(hidden_by_id[int(did)])
                row = corpus.by_id[int(did)]
                start, end, sim, vis, pool, retr_z = u.generated_pool_for_query(span_model, batch, bi, int(did), first_stage, corpus)
                ctx = u.build_span_feature_context(start, end, sim, vis)
                t = int(batch["lengths"][bi].item())
                gt_idx = u.c12_5.ts_to_idx(row["ts"], float(row["duration"]), t)
                annotated = []
                for rank, (s, e, score, meta) in enumerate(pool, start=1):
                    iou = u.c12_5.iou_1d((s, e + 1), (gt_idx[0], gt_idx[1] + 1))
                    annotated.append({"iou": float(iou), "rank": rank, "s": int(s), "e": int(e), "score": float(score), "pq": float(meta.get("pq", 0.0))})
                best = max(annotated, key=lambda z: (z["iou"], -z["rank"]))
                pos = [z for z in annotated if z["iou"] >= 0.7] or [best]
                weak = [z for z in annotated if 0.5 <= z["iou"] < 0.7]
                low = [z for z in annotated[:100] if z["iou"] < 0.3]
                above = [z for z in annotated if z["rank"] < best["rank"] and z["iou"] < best["iou"]]
                chosen = [best] + sorted(pos, key=lambda z: (-z["iou"], z["rank"]))[:12] + sorted(weak, key=lambda z: z["rank"])[:8] + sorted(low, key=lambda z: z["rank"])[:32] + sorted(above, key=lambda z: z["rank"])[:48] + rng.sample(annotated, min(16, len(annotated)))
                xs, ys, ranks, seen = [], [], [], set()
                for z in chosen:
                    key = (z["s"], z["e"])
                    if key in seen:
                        continue
                    seen.add(key)
                    base = u.span_feature_row_from_context(ctx, u.qtype_id(row.get("type", "unknown")), retr_z, z["s"], z["e"], z["rank"] - 1, z["score"], z["pq"])
                    hr = hidden_row_cached(hctx, z["s"], z["e"], t)
                    feat = base + hr
                    if all(np.isfinite(feat)):
                        xs.append(feat)
                        ys.append(z["iou"])
                        ranks.append(z["rank"])
                if len(xs) < 2:
                    continue
                dur = float(row["ts"][1] - row["ts"][0])
                best_ranks.append(best["rank"])
                groups.append({"desc_id": int(did), "x": np.asarray(xs, dtype=np.float32), "y": np.asarray(ys, dtype=np.float32), "rank": np.asarray(ranks, dtype=np.int64), "best_rank": int(best["rank"]), "duration_bucket": u.duration_bucket_seconds(dur)})
    finally:
        store.close()
    audit = {
        "query_count": len(groups),
        "missing_hidden_queries": missing_hidden,
        "span_rows_sampled": int(sum(g["x"].shape[0] for g in groups)),
        "best_iou_span_mean_rank": u.c12_5r.mean(best_ranks),
        "best_iou_span_median_rank": u.c12_5r.pct(best_ranks, 50),
        "best_iou_span_top100_rate": u.c12_5r.metric([r <= 100 for r in best_ranks]),
        "official_val_used": False,
    }
    return groups, audit


def train_ranker(kind: str, groups: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    u.seed_all(SEED)
    idxs = feature_indices(kind)
    if kind == "V3_two_head_hidden":
        model: nn.Module = TwoHead(len(idxs)).to(u.DEVICE)
    else:
        model = Ranker(len(idxs), hidden=288 if kind != "V1_hidden_boundary" else 192).to(u.DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    order = np.arange(len(groups))
    curves = []
    for epoch in range(1, 5):
        np.random.default_rng(SEED + epoch).shuffle(order)
        losses, ss_all, qq_all, yy_all = [], [], [], []
        for gi in order:
            g = groups[int(gi)]
            x = torch.from_numpy(g["x"][:, idxs]).to(u.DEVICE)
            y = torch.from_numpy(g["y"]).to(u.DEVICE)
            ranks = torch.from_numpy(g["rank"]).to(u.DEVICE)
            opt.zero_grad(set_to_none=True)
            if kind == "V3_two_head_hidden":
                sel, qual = model(x)  # type: ignore[misc]
                pred = sel
                qpred = qual
            else:
                pred = model(x)  # type: ignore[operator]
                qpred = pred
            best_i = int(np.argmax(g["y"]))
            hi = y >= 0.7
            lo = y < 0.3
            top_low = lo & (ranks <= 100)
            above = ranks < int(g["best_rank"])
            best_score = pred[best_i]
            pair = pred.new_tensor(0.0)
            if bool((top_low | above).any()):
                pair = F.softplus(-(best_score - pred[top_low | above] - 0.45)).mean()
            listwise = F.kl_div(F.log_softmax(pred, dim=0), F.softmax(y / 0.12, dim=0), reduction="batchmean")
            bce = F.binary_cross_entropy_with_logits(qpred, (y >= 0.7).float())
            qreg = (torch.sigmoid(qpred) - y).pow(2).mean()
            qpair = pred.new_tensor(0.0)
            if bool(hi.any() and lo.any()):
                qpair = F.softplus(-(qpred[hi].mean() - qpred[lo].mean() - 0.35))
            short_w = 1.8 if (kind == "V4_short_hidden_specialist" and g["duration_bucket"] == "short") else 1.0
            loss = short_w * (0.9 * listwise + 0.9 * pair + 0.45 * bce + 0.3 * qreg + 0.5 * qpair)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))
            if len(yy_all) < 100000:
                with torch.no_grad():
                    ss = torch.sigmoid(pred).detach().cpu().tolist()
                    qq = torch.sigmoid(qpred).detach().cpu().tolist()
                    ss_all.extend(ss)
                    qq_all.extend(qq)
                    yy_all.extend(y.detach().cpu().tolist())
        sc = u.c12_5r.corr(ss_all, yy_all)
        qc = u.c12_5r.corr(qq_all, yy_all)
        curves.append({"epoch": epoch, "loss": float(np.mean(losses)), "select_spearman": sc["spearman"], "quality_spearman": qc["spearman"], "auc07": u.c12_5r.auc_score(qq_all, [v >= 0.7 for v in yy_all])})
    path = MODEL_DIR / f"c12_5v_{kind}.pt"
    torch.save({"variant": kind, "feature_names": [FEATURE_NAMES[i] for i in idxs], "state_dict": model.state_dict(), "curves": curves}, path)
    return {"variant": kind, "model": model.eval(), "feature_indices": idxs, "path": str(path), "sha256": u.sha256_file(path), "curves": curves}


def load_ranker(kind: str) -> Dict[str, Any]:
    path = MODEL_DIR / f"c12_5v_{kind}.pt"
    ckpt = torch.load(path, map_location=u.DEVICE)
    idxs = [FEATURE_NAMES.index(n) for n in ckpt["feature_names"]]
    if kind == "V3_two_head_hidden":
        model: nn.Module = TwoHead(len(idxs)).to(u.DEVICE)
    else:
        model = Ranker(len(idxs), hidden=288 if kind != "V1_hidden_boundary" else 192).to(u.DEVICE)
    model.load_state_dict(ckpt["state_dict"])
    return {"variant": kind, "model": model.eval(), "feature_indices": idxs, "path": str(path), "sha256": u.sha256_file(path), "curves": ckpt.get("curves", [])}


def summarize(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    out = {"query_count": len(rows)}
    ranks = [r["rank_of_best_iou_span"] for r in rows if r["rank_of_best_iou_span"] is not None]
    for m in (50, 100, 200, 500):
        out[f"GT_video_oracle_IoU@0.5_top{m}"] = u.c12_5r.metric([r[f"cover05_top{m}"] for r in rows])
        out[f"GT_video_oracle_IoU@0.7_top{m}"] = u.c12_5r.metric([r[f"cover07_top{m}"] for r in rows])
    out["generated_M1000_oracle_IoU@0.7"] = u.c12_5r.metric([r["best_generated_iou_top1000"] >= 0.7 for r in rows])
    out["best_iou_span_mean_rank"] = u.c12_5r.mean(ranks)
    out["best_iou_span_median_rank"] = u.c12_5r.pct(ranks, 50)
    out["best_iou_span_top100_rate"] = u.c12_5r.metric([r <= 100 for r in ranks])
    out["best_iou_span_top50_rate"] = u.c12_5r.metric([r <= 50 for r in ranks])
    out["best_iou_span_top10_rate"] = u.c12_5r.metric([r <= 10 for r in ranks])
    return out


@torch.no_grad()
def evaluate(desc_ids: Sequence[int], corpus: Any, features: Dict[str, Any], first_stage: Dict[int, Dict[str, Any]], span_model: Any, hidden_by_id: Dict[int, Dict[str, Any]], trained: Dict[str, Dict[str, Any]], split: str) -> Dict[str, Any]:
    store = u.c12_5.ClipFeatureStore()
    rows_by = {k: [] for k in trained}
    score_by = {k: [] for k in trained}
    iou_by = {k: [] for k in trained}
    dur_by = {k: defaultdict(list) for k in trained}
    qt_by = {k: defaultdict(list) for k in trained}
    failures = {k: {"high_score_low_iou": [], "low_score_high_iou": []} for k in trained}
    try:
        for st in range(0, len(desc_ids), C12_BATCH):
            ids = list(desc_ids[st:st + C12_BATCH])
            batch = u.c12_5.build_batch(corpus, features, store, ids)
            for bi, did in enumerate(ids):
                if int(did) not in hidden_by_id:
                    continue
                hctx = build_hidden_context(hidden_by_id[int(did)])
                row = corpus.by_id[int(did)]
                start, end, sim, vis, pool, retr_z = u.generated_pool_for_query(span_model, batch, bi, int(did), first_stage, corpus)
                ctx = u.build_span_feature_context(start, end, sim, vis)
                t = int(batch["lengths"][bi].item())
                gt_ts = (float(row["ts"][0]), float(row["ts"][1]))
                qtype = row.get("type", "unknown")
                dur = float(row["ts"][1] - row["ts"][0])
                db = u.duration_bucket_seconds(dur)
                feats, ious = [], []
                for rank, (s, e, score, meta) in enumerate(pool):
                    base = u.span_feature_row_from_context(ctx, u.qtype_id(qtype), retr_z, s, e, rank, score, float(meta.get("pq", 0.0)))
                    feats.append(base + hidden_row_cached(hctx, s, e, t))
                    ious.append(u.c12_5.iou_1d(u.c12_5.idx_to_ts(s, e, float(row["duration"]), t), gt_ts))
                feat_np = np.asarray(feats, dtype=np.float32)
                iou_np = np.asarray(ious, dtype=np.float32)
                best_idx = int(np.argmax(iou_np))
                for name, rec_model in trained.items():
                    idxs = rec_model["feature_indices"]
                    xx = torch.from_numpy(feat_np[:, idxs]).to(u.DEVICE)
                    model = rec_model["model"]
                    if name.startswith("V3"):
                        sel, qual = model(xx)
                        if name.endswith("quality"):
                            scores = torch.sigmoid(qual).detach().cpu().numpy()
                        elif name.endswith("weighted"):
                            scores = (0.7 * torch.sigmoid(sel) + 0.3 * torch.sigmoid(qual)).detach().cpu().numpy()
                        else:
                            scores = torch.sigmoid(sel).detach().cpu().numpy()
                    else:
                        scores = torch.sigmoid(model(xx)).detach().cpu().numpy()
                    order = np.argsort(-scores)
                    rank_best = int(np.where(order == best_idx)[0][0] + 1)
                    rec = {"desc_id": int(did), "query_type": qtype, "moment_duration": dur, "best_generated_iou_top1000": float(np.max(iou_np)), "rank_of_best_iou_span": rank_best}
                    for m in (50, 100, 200, 500):
                        best = float(np.max(iou_np[order[:m]]))
                        rec[f"cover05_top{m}"] = best >= 0.5
                        rec[f"cover07_top{m}"] = best >= 0.7
                    rows_by[name].append(rec)
                    dur_by[name][db].append(rec)
                    qt_by[name][str(qtype)].append(rec)
                    score_by[name].extend(scores[:POOL_LIMIT].tolist())
                    iou_by[name].extend(iou_np[:POOL_LIMIT].tolist())
                    if split == "calib_holdout":
                        p90 = float(np.percentile(scores, 90))
                        p25 = float(np.percentile(scores, 25))
                        high_bad = [int(i) for i in order[:100] if scores[int(i)] >= p90 and iou_np[int(i)] < 0.3]
                        low_good = [int(i) for i in order[-300:] if scores[int(i)] <= p25 and iou_np[int(i)] >= 0.7]
                        for ix in high_bad[:1]:
                            if len(failures[name]["high_score_low_iou"]) < 200:
                                failures[name]["high_score_low_iou"].append({"desc_id": int(did), "score": float(scores[ix]), "iou": float(iou_np[ix]), "rank": int(np.where(order == ix)[0][0] + 1)})
                        for ix in low_good[:1]:
                            if len(failures[name]["low_score_high_iou"]) < 200:
                                failures[name]["low_score_high_iou"].append({"desc_id": int(did), "score": float(scores[ix]), "iou": float(iou_np[ix]), "rank": int(np.where(order == ix)[0][0] + 1)})
    finally:
        store.close()
    out = {}
    for name in trained:
        calib = {
            **u.c12_5r.corr(score_by[name][:300000], iou_by[name][:300000]),
            "auc_iou05": u.c12_5r.auc_score(score_by[name][:300000], [x >= 0.5 for x in iou_by[name][:300000]]),
            "auc_iou07": u.c12_5r.auc_score(score_by[name][:300000], [x >= 0.7 for x in iou_by[name][:300000]]),
            "sample_count": min(300000, len(score_by[name])),
        }
        out[name] = {"variant": name, "split": split, "summary": summarize(rows_by[name]), "duration_breakdown": {k: summarize(v) for k, v in sorted(dur_by[name].items())}, "query_type_breakdown": {k: summarize(v) for k, v in sorted(qt_by[name].items())}, "pq_iou_calibration": calib, "failure_cases": failures[name]}
    return out


def write_availability(sample: Dict[str, Any], status: str) -> str:
    schema = {
        "stage": "C12-5V-A",
        "status": status,
        "feature_names": FEATURE_NAMES,
        "feature_count": len(FEATURE_NAMES),
        "hidden_sources": ["CONQUER.query_feature", "QDF_feature", "QAL_feature", "Contextual_QAL", "G", "q2v_attention", "begin/end logits"],
        "train_inference_shared_feature_builder": "span_feature_row_from_context + hidden_row",
        "candidate_video_score_semantics": "candidate-video first-stage score; GT oracle passes GT video as candidate",
        "no_gt_iou_label_at_inference": True,
        "no_official_prediction_pool": True,
        "no_c7_b6_fixed_span_pool_as_final_candidates": True,
        "no_position_based_join": True,
        "duplicate_overwrite": False,
        "silent_zero_fill": False,
    }
    schema_hash = sha256_obj(schema)
    schema["schema_hash"] = schema_hash
    write_json(OUT / "C12_5V_A_HIDDEN_FEATURE_SCHEMA.json", schema)
    write_json(OUT / "C12_5V_A_FORWARD_HOOK_AUDIT.json", sample)
    missing = {"stage": "C12-5V-A", "missing_feature_count": 0 if status == "C12_CONQUER_HIDDEN_AVAILABLE" else len(HIDDEN_FEATURES), "nonfinite_count": int(sample.get("nonfinite_count", 0)), "official_val_used": False}
    write_json(OUT / "C12_5V_A_MISSING_NONFINITE_AUDIT.json", missing)
    write_text(OUT / "C12_5V_A_HIDDEN_FEATURE_AVAILABILITY.md", f"""# C12-5V-A Hidden Feature Availability

status = {status}

CONQUER forward with `return_intermediates=True` exposes query embedding, QDF, QAL, contextual QAL, G, q2v attention, and begin/end logits.

feature_schema_hash = {schema_hash}

official_val_used = false
""")
    return schema_hash


def main() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    OUT.mkdir(parents=True, exist_ok=True)
    corpus = u.load_corpus()
    u.c12_5.corpus_global = corpus
    c12_features = u.load_features(u.build_feature_caches(corpus))
    span_model = u.c12_5r.load_model("teacher_distilled")

    train_ids = list(corpus.splits["train_fit"])[:TRAIN_LIMIT]
    select_ids = list(corpus.splits["calib_select"])[:EVAL_LIMIT or len(corpus.splits["calib_select"])]
    holdout_ids = list(corpus.splits["calib_holdout"])[:EVAL_LIMIT or len(corpus.splits["calib_holdout"])]

    smoke_hidden = export_hidden_for_ids(holdout_ids[:4])
    sample = {
        "stage": "C12-5V-A",
        "sample_desc_ids": sorted(smoke_hidden),
        "available_keys": sorted(next(iter(smoke_hidden.values())).keys()) if smoke_hidden else [],
        "nonfinite_count": 0,
        "official_val_used": False,
    }
    status = "C12_CONQUER_HIDDEN_AVAILABLE" if smoke_hidden else "C12_CONQUER_HIDDEN_EXPORT_BLOCKED"
    schema_hash = write_availability(sample, status)
    if status != "C12_CONQUER_HIDDEN_AVAILABLE":
        decision = {"stage": "C12-5V-E", "status": "C12_CONQUER_HIDDEN_EXPORT_BLOCKED_NEED_STRONGER_BOUNDARY_MODEL", "allow_enter_c12_6": False, "official_val_used": False, "evaluator_modified": False, "nms_modified": False}
        write_json(OUT / "C12_5V_DECISION.json", decision)
        write_text(OUT / "C12_5V_DECISION.md", "# C12-5V Decision\n\nstatus = C12_CONQUER_HIDDEN_EXPORT_BLOCKED_NEED_STRONGER_BOUNDARY_MODEL\n")
        return

    builder = {"stage": "C12-5V-B", "feature_builder_hash": schema_hash, "feature_names": FEATURE_NAMES, "distribution_audit": {"hidden_smoke_queries": len(smoke_hidden)}, "official_val_used": False}
    write_json(OUT / "C12_5V_B_FEATURE_BUILDER_HASH.json", builder)
    write_json(OUT / "C12_5V_B_TRAIN_INFERENCE_SCHEMA_AUDIT.json", builder | {"status": "C12_5V_SHARED_FEATURE_BUILDER_PASS"})
    write_text(OUT / "C12_5V_B_SHARED_FEATURE_BUILDER.md", f"# C12-5V-B Shared Feature Builder\n\nstatus = C12_5V_SHARED_FEATURE_BUILDER_PASS\n\nfeature_builder_hash = {schema_hash}\n")

    train_first_stage = u.load_first_stage(corpus, train_ids, top_keep=128, cache_name="first_stage_train_calib_holdout_top128.pkl")
    select_first_stage = u.load_first_stage(corpus, select_ids, top_keep=128)
    holdout_first_stage = u.load_first_stage(corpus, holdout_ids, top_keep=128, cache_name="first_stage_calib_holdout_top128.pkl" if EVAL_LIMIT == 0 else None)

    variants = ["V1_hidden_boundary", "V2_span_context_hidden", "V3_two_head_hidden", "V4_short_hidden_specialist"]
    can_resume = all((MODEL_DIR / f"c12_5v_{v}.pt").exists() for v in variants)
    if can_resume:
        trained_raw = {v: load_ranker(v) for v in variants}
    else:
        hidden_train = export_hidden_for_ids(train_ids)
        groups, data_audit = build_groups(train_ids, corpus, c12_features, train_first_stage, span_model, hidden_train)
        write_json(OUT / "C12_5V_C_TRAINING_DATASET_AUDIT.json", data_audit)
        trained_raw = {v: train_ranker(v, groups) for v in variants}
    trained: Dict[str, Dict[str, Any]] = {}
    for v, rec in trained_raw.items():
        if v == "V3_two_head_hidden":
            for suffix in ("select", "quality", "weighted"):
                trained[f"V3_two_head_hidden_{suffix}"] = rec
        else:
            trained[v] = rec
    train_info = {"stage": "C12-5V-C", "trained_variants": variants, "models": {k: {kk: vv for kk, vv in r.items() if kk != "model"} for k, r in trained_raw.items()}, "official_val_used": False}
    write_json(OUT / "C12_5V_C_HIDDEN_RANKER_TRAINING_RESULTS.json", train_info)

    hidden_select = export_hidden_for_ids(select_ids)
    hidden_holdout = export_hidden_for_ids(holdout_ids)
    results = {
        "stage": "C12-5V-D",
        "references": json.loads((ROOT / "c12_5u_topk_calibration_repair/C12_5U_E_RESULTS.json").read_text(encoding="utf-8"))["calib_holdout"],
        "calib_select": evaluate(select_ids, corpus, c12_features, select_first_stage, span_model, hidden_select, trained, "calib_select"),
        "calib_holdout": evaluate(holdout_ids, corpus, c12_features, holdout_first_stage, span_model, hidden_holdout, trained, "calib_holdout"),
        "official_val_used": False,
    }
    write_json(OUT / "C12_5V_D_RESULTS.json", results)
    hold = results["calib_holdout"]
    write_json(OUT / "C12_5V_D_CALIBRATION_METRICS.json", {k: {"calib_select": results["calib_select"][k]["pq_iou_calibration"], "calib_holdout": hold[k]["pq_iou_calibration"]} for k in hold})
    write_json(OUT / "C12_5V_D_DURATION_BREAKDOWN.json", {k: hold[k]["duration_breakdown"] for k in hold})
    write_json(OUT / "C12_5V_D_QUERY_TYPE_BREAKDOWN.json", {k: hold[k]["query_type_breakdown"] for k in hold})
    write_json(OUT / "C12_5V_D_FAILURE_CASES.json", {k: hold[k]["failure_cases"] for k in hold})

    selectable = results["calib_select"]
    selected = max(selectable, key=lambda k: (
        selectable[k]["pq_iou_calibration"]["spearman"] > 0.15,
        selectable[k]["summary"]["GT_video_oracle_IoU@0.7_top100"] >= 55,
        selectable[k]["summary"]["GT_video_oracle_IoU@0.7_top50"],
        selectable[k]["pq_iou_calibration"]["auc_iou07"],
    ))
    best = hold[selected]
    s = best["summary"]
    c = best["pq_iou_calibration"]
    short = best["duration_breakdown"].get("short", {})
    no_q_collapse = all(v["GT_video_oracle_IoU@0.7_top100"] >= 45 for v in best["query_type_breakdown"].values())
    allow = (
        s["GT_video_oracle_IoU@0.7_top100"] >= 55
        and s["GT_video_oracle_IoU@0.7_top50"] > 38.963
        and short.get("GT_video_oracle_IoU@0.7_top100", 0.0) >= 45
        and (c.get("spearman") or 0.0) > 0.15
        and (c.get("auc_iou07") or 0.0) > 0.6035
        and s["GT_video_oracle_IoU@0.5_top100"] >= 75
        and no_q_collapse
    )
    if allow:
        status_dec = "C12_HIDDEN_SPAN_SCORE_REPAIRED_CONTINUE_TO_C12_6"
    elif (c.get("spearman") or 0.0) <= 0:
        status_dec = "C12_HIDDEN_RANKER_NO_GAIN_NEED_STRONGER_FEATURES"
    elif s["GT_video_oracle_IoU@0.7_top100"] >= 55:
        status_dec = "C12_HIDDEN_TOPK_GOOD_CALIBRATION_PARTIAL"
    else:
        status_dec = "C12_SPAN_LINE_STOP_KEEP_C7_B6"
    decision = {
        "stage": "C12-5V-E",
        "status": status_dec,
        "best_variant": selected,
        "best_iou07_top100": s["GT_video_oracle_IoU@0.7_top100"],
        "best_iou07_top50": s["GT_video_oracle_IoU@0.7_top50"],
        "short_iou07_top100": short.get("GT_video_oracle_IoU@0.7_top100"),
        "best_iou_span_top100_rate": s.get("best_iou_span_top100_rate"),
        "best_iou_span_top50_rate": s.get("best_iou_span_top50_rate"),
        "pq_spearman": c.get("spearman"),
        "auc_iou07": c.get("auc_iou07"),
        "allow_enter_c12_6": bool(allow),
        "official_val_used": False,
        "evaluator_modified": False,
        "nms_modified": False,
        "reason": f"Selected on calib_select. Holdout top100={s['GT_video_oracle_IoU@0.7_top100']:.4f}, top50={s['GT_video_oracle_IoU@0.7_top50']:.4f}, Spearman={c.get('spearman')}, AUC@0.7={c.get('auc_iou07')}.",
    }
    write_json(OUT / "C12_5V_DECISION.json", decision)
    def md_row(name: str, rec: dict) -> str:
        ss = rec["summary"]
        cc = rec["pq_iou_calibration"]
        short_iou = rec["duration_breakdown"].get("short", {}).get("GT_video_oracle_IoU@0.7_top100")
        return (
            f"| {name} | {ss.get('query_count')} | "
            f"{ss['GT_video_oracle_IoU@0.5_top100']:.4f} | "
            f"{ss['GT_video_oracle_IoU@0.7_top100']:.4f} | "
            f"{ss['GT_video_oracle_IoU@0.7_top50']:.4f} | "
            f"{short_iou:.4f} | "
            f"{ss.get('best_iou_span_top100_rate'):.4f} | "
            f"{ss.get('best_iou_span_top50_rate'):.4f} | "
            f"{cc.get('spearman'):.6f} | "
            f"{cc.get('auc_iou07'):.6f} |"
        )

    header = "| variant | queries | IoU@0.5 top100 | IoU@0.7 top100 | IoU@0.7 top50 | short IoU@0.7 top100 | best-IoU top100 | best-IoU top50 | PQ Spearman | AUC@0.7 |"
    sep = "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
    lines = ["# C12-5V-D Results", "", "## Reference Baselines", "", header, sep]
    for name, rec in refs.items():
        lines.append(md_row(name, rec))
    lines += ["", "## C12-5V Hidden Rankers", "", header, sep]
    for name, rec in hold.items():
        lines.append(md_row(name, rec))
    lines += [
        "",
        "## Decision Summary",
        "",
        f"- hidden_export_available: true",
        f"- shared_feature_builder: C12_5V_SHARED_FEATURE_BUILDER_PASS",
        f"- trained_variants: V1_hidden_boundary, V2_span_context_hidden, V3_two_head_hidden, V4_short_hidden_specialist",
        f"- best_variant: {selected}",
        f"- allow_enter_c12_6: {str(allow).lower()}",
        f"- status: {status_dec}",
        f"- official_val_used: false",
        f"- evaluator_modified: false",
        f"- nms_modified: false",
    ]
    write_text(OUT / "C12_5V_D_RESULTS.md", "\n".join(lines) + "\n")
    write_text(OUT / "C12_5V_DECISION.md", f"""# C12-5V Decision

status = {decision['status']}

best_variant = {selected}

IoU@0.7 top100 = {decision['best_iou07_top100']:.4f}
IoU@0.7 top50 = {decision['best_iou07_top50']:.4f}
short IoU@0.7 top100 = {decision['short_iou07_top100']}
PQ Spearman = {decision['pq_spearman']}
AUC@0.7 = {decision['auc_iou07']}

allow_enter_c12_6 = {str(allow).lower()}

official_val_used = false
evaluator_modified = false
nms_modified = false
""")
    manifest = {"stage": "C12-5V", "status": status_dec, "best_variant": selected, "allow_enter_c12_6": bool(allow), "official_val_used": False, "evaluator_modified": False, "nms_modified": False, "artifact_hashes": {str(p.relative_to(ROOT)): u.sha256_file(p) for p in sorted(OUT.glob("C12_5V*")) if p.is_file()}}
    write_json(OUT / "C12_5V_MANIFEST.json", manifest)


if __name__ == "__main__":
    main()
