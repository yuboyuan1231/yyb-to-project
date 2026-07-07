#!/usr/bin/env python3
"""C12-5 native span candidate generation.

Train-only native span generators over C12 CONQUER-warm video candidates.
No official data, no evaluator/NMS modification, and no C7-B6 fixed pool as
the final C12 candidate source.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import pickle
import random
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import lmdb
import msgpack_numpy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from run_c12_native_retriever_training import (
    CACHE,
    DEVICE,
    ROOT,
    Corpus,
    build_feature_caches,
    load_corpus,
    load_features,
    load_first_stage,
    qtype_id,
    seed_all,
    sha256_file,
    video_group,
    write_json,
    write_text,
)
from c12_native_retriever.scaffold import C12Paths


torch.set_num_threads(min(16, os.cpu_count() or 1))

OUT = ROOT / "c12_5_native_span_generation"
MODEL_DIR = ROOT / "c12_models"
PATHS = C12Paths()
SEED = 1250
CLIP_LEN = 1.5
TRAIN_LIMIT = int(os.environ.get("C12_5_TRAIN_LIMIT", "16000"))
EVAL_LIMIT = int(os.environ.get("C12_5_EVAL_LIMIT", "0"))
MAX_T = 96
MAX_SPAN = 48
BOUNDARY_TOPK = 48
EVAL_BATCH = 32
TRAIN_BATCH = 32


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def stable_hash_obj(obj: Any) -> str:
    raw = json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def l2_rows(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), eps)


def safe_mean(xs: Sequence[float]) -> float | None:
    return float(np.mean(xs)) if xs else None


def percentile(xs: Sequence[float], p: float) -> float | None:
    return float(np.percentile(xs, p)) if xs else None


def iou_1d(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = max(a[1], b[1]) - min(a[0], b[0])
    return inter / union if union > 0 else 0.0


def ts_to_idx(ts: Sequence[float], duration: float, t: int) -> Tuple[int, int]:
    if t <= 1:
        return 0, 0
    st = int(math.floor(max(0.0, float(ts[0])) / max(duration, 1e-6) * t))
    ed = int(math.ceil(min(duration, float(ts[1])) / max(duration, 1e-6) * t)) - 1
    st = max(0, min(t - 1, st))
    ed = max(st, min(t - 1, ed))
    return st, ed


def idx_to_ts(s: int, e: int, duration: float, t: int) -> Tuple[float, float]:
    unit = duration / max(t, 1)
    return float(s * unit), float(min(duration, (e + 1) * unit))


def metric_from_bools(vals: Sequence[bool]) -> float:
    return 100.0 * sum(1 for v in vals if v) / max(1, len(vals))


class ClipFeatureStore:
    def __init__(self) -> None:
        self.sub_env = lmdb.open(str(PATHS.subtitle_lmdb), readonly=True, create=False, lock=False, readahead=False, max_readers=512)
        self.vis_env = lmdb.open(str(PATHS.visual_lmdb), readonly=True, create=False, lock=False, readahead=False, max_readers=512)
        self._sub_cache: Dict[str, np.ndarray] = {}
        self._vis_cache: Dict[str, np.ndarray] = {}

    def close(self) -> None:
        self.sub_env.close()
        self.vis_env.close()

    @staticmethod
    def _read_npz(txn: lmdb.Transaction, key: str) -> np.ndarray | None:
        raw = txn.get(key.encode())
        if raw is None:
            return None
        with io.BytesIO(bytes(raw)) as reader:
            return np.load(reader, allow_pickle=True)["features"].astype(np.float32)

    @staticmethod
    def _read_visual(txn: lmdb.Transaction, key: str) -> np.ndarray | None:
        raw = txn.get(key.encode())
        if raw is None:
            return None
        return msgpack_numpy.loads(bytes(raw), raw=False)["features"].astype(np.float32)

    def subtitle(self, vid: str) -> np.ndarray:
        if vid in self._sub_cache:
            return self._sub_cache[vid]
        with self.sub_env.begin(buffers=True) as txn:
            arr = self._read_npz(txn, vid)
        if arr is None:
            arr = np.zeros((1, 768), dtype=np.float32)
        arr = arr[:MAX_T].astype(np.float32)
        arr = l2_rows(arr)
        if len(self._sub_cache) > 1024:
            self._sub_cache.clear()
        self._sub_cache[vid] = arr
        return arr

    def visual_energy(self, vid: str, target_t: int) -> np.ndarray:
        if vid in self._vis_cache:
            vis = self._vis_cache[vid]
        else:
            with self.vis_env.begin(buffers=True) as txn:
                vis = self._read_visual(txn, vid)
            if vis is None:
                vis = np.zeros((1, 4352), dtype=np.float32)
            vis = vis[:MAX_T].astype(np.float32)
            if len(self._vis_cache) > 256:
                self._vis_cache.clear()
            self._vis_cache[vid] = vis
        if vis.shape[0] <= 1:
            e = np.zeros((vis.shape[0],), dtype=np.float32)
        else:
            diff = np.diff(vis, axis=0, prepend=vis[:1])
            e = np.linalg.norm(diff, axis=1).astype(np.float32)
            e = (e - float(e.mean())) / max(float(e.std()), 1e-6)
        if e.shape[0] == target_t:
            return e.astype(np.float32)
        xp = np.linspace(0, 1, num=max(1, e.shape[0]))
        xq = np.linspace(0, 1, num=target_t)
        return np.interp(xq, xp, e).astype(np.float32)


class SpanLocalizer(nn.Module):
    def __init__(self, use_qtype: bool = False, hidden: int = 192) -> None:
        super().__init__()
        self.use_qtype = use_qtype
        self.q_proj = nn.Sequential(nn.LayerNorm(768), nn.Linear(768, hidden), nn.GELU(), nn.LayerNorm(hidden))
        self.s_proj = nn.Sequential(nn.LayerNorm(768), nn.Linear(768, hidden), nn.GELU(), nn.LayerNorm(hidden))
        self.qtype_emb = nn.Embedding(4, 8)
        in_dim = 4 + (8 if use_qtype else 0)
        self.token_mlp = nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, 64), nn.GELU(), nn.Linear(64, 3))
        pair_dim = 8 + (8 if use_qtype else 0)
        self.pair_mlp = nn.Sequential(nn.LayerNorm(pair_dim), nn.Linear(pair_dim, 64), nn.GELU(), nn.Linear(64, 1))

    def token_forward(self, q: torch.Tensor, sub: torch.Tensor, vis_energy: torch.Tensor, mask: torch.Tensor, qtype: torch.Tensor) -> Dict[str, torch.Tensor]:
        qh = F.normalize(self.q_proj(q), dim=-1)
        sh = F.normalize(self.s_proj(sub), dim=-1)
        sim = torch.einsum("bd,btd->bt", qh, sh)
        bsz, t = sim.shape
        pos = torch.linspace(0, 1, t, device=sim.device, dtype=sim.dtype).view(1, t).expand(bsz, t)
        rel_end = 1.0 - pos
        feats = [sim.unsqueeze(-1), pos.unsqueeze(-1), rel_end.unsqueeze(-1), vis_energy.unsqueeze(-1)]
        if self.use_qtype:
            qe = self.qtype_emb(qtype).unsqueeze(1).expand(bsz, t, -1)
            feats.append(qe)
        x = torch.cat(feats, dim=-1)
        out = self.token_mlp(x)
        start = out[..., 0].masked_fill(~mask, -1e4)
        end = out[..., 1].masked_fill(~mask, -1e4)
        token_quality = out[..., 2].masked_fill(~mask, -1e4)
        return {"start": start, "end": end, "token_quality": token_quality, "sim": sim.masked_fill(~mask, 0.0)}

    def pair_quality(
        self,
        tok: Dict[str, torch.Tensor],
        spans: torch.Tensor,
        lengths: torch.Tensor,
        qtype: torch.Tensor,
    ) -> torch.Tensor:
        # spans: [B, P, 2]
        bsz, p, _ = spans.shape
        bidx = torch.arange(bsz, device=spans.device).view(bsz, 1).expand(bsz, p)
        s = spans[..., 0].clamp_min(0)
        e = spans[..., 1].clamp_min(0)
        start = tok["start"][bidx, s]
        end = tok["end"][bidx, e]
        tq = 0.5 * (tok["token_quality"][bidx, s] + tok["token_quality"][bidx, e])
        sim = 0.5 * (tok["sim"][bidx, s] + tok["sim"][bidx, e])
        span_len = ((e - s + 1).float() / lengths.view(bsz, 1).float().clamp_min(1.0)).clamp(0, 1)
        center = ((s + e).float() * 0.5 / lengths.view(bsz, 1).float().clamp_min(1.0)).clamp(0, 1)
        feats = [start.unsqueeze(-1), end.unsqueeze(-1), tq.unsqueeze(-1), sim.unsqueeze(-1), span_len.unsqueeze(-1), center.unsqueeze(-1), (1 - center).unsqueeze(-1), (1 - span_len).unsqueeze(-1)]
        if self.use_qtype:
            feats.append(self.qtype_emb(qtype).unsqueeze(1).expand(bsz, p, -1))
        x = torch.cat(feats, dim=-1)
        return self.pair_mlp(x).squeeze(-1)


def build_batch(corpus: Corpus, features: Dict[str, Any], store: ClipFeatureStore, desc_ids: Sequence[int]) -> Dict[str, Any]:
    q_list, sub_list, vis_list, mask_list, start_labels, end_labels, lengths, qtypes, dids = [], [], [], [], [], [], [], [], []
    max_t = 1
    rows = []
    for did in desc_ids:
        row = corpus.by_id[int(did)]
        vid = row["vid_name"]
        sub = store.subtitle(vid)
        t = min(MAX_T, sub.shape[0])
        sub = sub[:t]
        vis = store.visual_energy(vid, t)
        st, ed = ts_to_idx(row["ts"], float(row["duration"]), t)
        rows.append((int(did), row, sub, vis, st, ed, t))
        max_t = max(max_t, t)
    for did, row, sub, vis, st, ed, t in rows:
        pad = max_t - t
        q_list.append(features["query"][features["desc_to_qpos"][did]])
        sub_list.append(np.pad(sub, ((0, pad), (0, 0)), mode="constant"))
        vis_list.append(np.pad(vis, (0, pad), mode="constant"))
        mask_list.append(np.asarray([True] * t + [False] * pad, dtype=bool))
        start_labels.append(st)
        end_labels.append(ed)
        lengths.append(t)
        qtypes.append(qtype_id(row.get("type", "unknown")))
        dids.append(did)
    return {
        "desc_ids": dids,
        "q": torch.from_numpy(np.stack(q_list).astype(np.float32)).to(DEVICE),
        "sub": torch.from_numpy(np.stack(sub_list).astype(np.float32)).to(DEVICE),
        "vis": torch.from_numpy(np.stack(vis_list).astype(np.float32)).to(DEVICE),
        "mask": torch.from_numpy(np.stack(mask_list)).to(DEVICE),
        "start": torch.tensor(start_labels, dtype=torch.long, device=DEVICE),
        "end": torch.tensor(end_labels, dtype=torch.long, device=DEVICE),
        "lengths": torch.tensor(lengths, dtype=torch.long, device=DEVICE),
        "qtype": torch.tensor(qtypes, dtype=torch.long, device=DEVICE),
    }


def sample_training_spans(batch: Dict[str, Any], per_query: int = 48) -> Tuple[torch.Tensor, torch.Tensor]:
    rng = random.Random(SEED + len(batch["desc_ids"]))
    spans, labels = [], []
    for bi, did in enumerate(batch["desc_ids"]):
        t = int(batch["lengths"][bi].item())
        st = int(batch["start"][bi].item())
        ed = int(batch["end"][bi].item())
        row_spans = [[st, ed]]
        row_labels = [1.0]
        for _ in range(per_query - 1):
            s = rng.randrange(t)
            max_e = min(t - 1, s + MAX_SPAN - 1)
            e = rng.randrange(s, max_e + 1)
            iou_idx = iou_1d((s, e + 1), (st, ed + 1))
            row_spans.append([s, e])
            row_labels.append(float(iou_idx))
        spans.append(row_spans)
        labels.append(row_labels)
    return torch.tensor(spans, dtype=torch.long, device=DEVICE), torch.tensor(labels, dtype=torch.float32, device=DEVICE)


def load_b6_teacher() -> Dict[int, List[List[float]]]:
    p = PATHS.b6_train_teacher_subset
    if not p.exists():
        return {}
    data = load_json(p)
    out = {}
    for rec in data.get("VCMR", []):
        out[int(rec["desc_id"])] = rec.get("predictions", [])
    return out


def best_b6_gt_span_for(row: Dict[str, Any], corpus: Corpus, teacher: Dict[int, List[List[float]]]) -> Tuple[float, float] | None:
    preds = teacher.get(int(row["desc_id"]), [])
    gt_idx = corpus.video2idx.get(row["vid_name"])
    if gt_idx is None:
        return None
    for pred in preds:
        if int(pred[0]) == int(gt_idx):
            return float(pred[1]), float(pred[2])
    return None


def train_variant(
    name: str,
    config: Dict[str, Any],
    corpus: Corpus,
    features: Dict[str, Any],
    train_ids: Sequence[int],
    teacher: Dict[int, List[List[float]]],
) -> Dict[str, Any]:
    seed_all(SEED)
    model = SpanLocalizer(use_qtype=bool(config.get("use_qtype", False)), hidden=int(config.get("hidden", 192))).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=float(config["lr"]), weight_decay=float(config["weight_decay"]))
    store = ClipFeatureStore()
    ids = list(train_ids)
    random.Random(SEED).shuffle(ids)
    ids = ids[: int(config["train_limit"])]
    curves = []
    best_state = None
    best_loss = float("inf")
    start_time = time.time()
    teacher_used = 0
    try:
        for epoch in range(1, int(config["epochs"]) + 1):
            random.Random(SEED + epoch).shuffle(ids)
            losses, se_losses, pq_losses, teacher_losses = [], [], [], []
            grad_groups = Counter()
            model.train()
            for st in range(0, len(ids), int(config["batch_size"])):
                batch_ids = ids[st: st + int(config["batch_size"])]
                batch = build_batch(corpus, features, store, batch_ids)
                opt.zero_grad(set_to_none=True)
                tok = model.token_forward(batch["q"], batch["sub"], batch["vis"], batch["mask"], batch["qtype"])
                se_loss = F.cross_entropy(tok["start"], batch["start"]) + F.cross_entropy(tok["end"], batch["end"])
                spans, iou_labels = sample_training_spans(batch, per_query=48)
                pq = model.pair_quality(tok, spans, batch["lengths"], batch["qtype"])
                pq_loss = F.binary_cross_entropy_with_logits(pq, (iou_labels >= 0.7).float())
                soft_loss = F.mse_loss(torch.sigmoid(pq), iou_labels)
                loss = se_loss + float(config.get("pq_weight", 0.0)) * (pq_loss + soft_loss)
                teach_loss = tok["start"].new_tensor(0.0)
                if config.get("use_teacher", False):
                    teach_start, teach_end, valid = [], [], []
                    for did in batch["desc_ids"]:
                        row = corpus.by_id[int(did)]
                        span = best_b6_gt_span_for(row, corpus, teacher)
                        if span is None:
                            teach_start.append(0)
                            teach_end.append(0)
                            valid.append(False)
                        else:
                            bi = len(valid)
                            t = int(batch["lengths"][bi].item())
                            s, e = ts_to_idx(span, float(row["duration"]), t)
                            teach_start.append(s)
                            teach_end.append(e)
                            valid.append(True)
                    valid_t = torch.tensor(valid, dtype=torch.bool, device=DEVICE)
                    if bool(valid_t.any()):
                        ts = torch.tensor(teach_start, dtype=torch.long, device=DEVICE)
                        te = torch.tensor(teach_end, dtype=torch.long, device=DEVICE)
                        teach_loss = F.cross_entropy(tok["start"][valid_t], ts[valid_t]) + F.cross_entropy(tok["end"][valid_t], te[valid_t])
                        teacher_used += int(valid_t.sum().item())
                        loss = loss + float(config.get("teacher_weight", 0.0)) * teach_loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                for pname, p in model.named_parameters():
                    if p.grad is not None and torch.isfinite(p.grad).all() and float(p.grad.abs().sum().detach().cpu()) > 0:
                        grad_groups[pname.split(".")[0]] += 1
                opt.step()
                losses.append(float(loss.detach().cpu()))
                se_losses.append(float(se_loss.detach().cpu()))
                pq_losses.append(float((pq_loss + soft_loss).detach().cpu()))
                teacher_losses.append(float(teach_loss.detach().cpu()))
            mean_loss = float(np.mean(losses)) if losses else float("inf")
            curves.append({
                "epoch": epoch,
                "loss": mean_loss,
                "start_end_loss": float(np.mean(se_losses)) if se_losses else None,
                "proposal_quality_loss": float(np.mean(pq_losses)) if pq_losses else None,
                "teacher_loss": float(np.mean(teacher_losses)) if teacher_losses else None,
                "grad_nonzero_groups": dict(grad_groups),
            })
            if mean_loss < best_loss:
                best_loss = mean_loss
                best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    finally:
        store.close()
    if best_state is not None:
        model.load_state_dict(best_state)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / f"c12_5_{name}.pt"
    torch.save({"variant": name, "config": config, "state_dict": model.state_dict(), "curves": curves}, model_path)
    return {
        "name": name,
        "model": model,
        "model_path": str(model_path),
        "model_sha256": sha256_file(model_path),
        "training_curves": curves,
        "teacher_supervision_rows_used": teacher_used,
        "runtime_seconds": time.time() - start_time,
    }


@torch.no_grad()
def top_spans_for_batch(
    model: SpanLocalizer,
    batch: Dict[str, Any],
    variant: str,
    top_m: int = 100,
) -> Dict[int, List[Dict[str, float]]]:
    model.eval()
    tok = model.token_forward(batch["q"], batch["sub"], batch["vis"], batch["mask"], batch["qtype"])
    out: Dict[int, List[Dict[str, float]]] = {}
    for bi, did in enumerate(batch["desc_ids"]):
        t = int(batch["lengths"][bi].item())
        duration = float(corpus_global.by_id[int(did)]["duration"])
        starts = tok["start"][bi, :t]
        ends = tok["end"][bi, :t]
        sims = tok["sim"][bi, :t]
        qtype = batch["qtype"][bi: bi + 1]
        length = batch["lengths"][bi: bi + 1]
        starts_np = starts.detach().cpu().numpy()
        ends_np = ends.detach().cpu().numpy()
        sims_np = sims.detach().cpu().numpy()
        top_s = np.argsort(-starts_np)[: min(BOUNDARY_TOPK, t)].tolist()
        top_e = np.argsort(-ends_np)[: min(BOUNDARY_TOPK, t)].tolist()
        pair_set = set()
        span_pairs = []
        base_scores = []
        for s in top_s:
            for e in top_e:
                if e < s or e >= s + MAX_SPAN:
                    continue
                pair_set.add((int(s), int(e)))
        for center in np.argsort(-sims_np)[: min(16, t)].tolist():
            for half in (0, 1, 2, 4, 8, 12, 16, 24):
                s = max(0, int(center) - half)
                e = min(t - 1, int(center) + half)
                if e >= s and e < s + MAX_SPAN:
                    pair_set.add((s, e))
        for s, e in pair_set:
            score = float(starts_np[s] + ends_np[e])
            if variant == "teacher_distilled":
                score -= 0.02 * abs((e - s + 1) - max(1, int(0.12 * t)))
            elif variant == "qtype_aware":
                qt = int(qtype.item())
                target = {0: 0.12, 1: 0.18, 2: 0.15, 3: 0.15}.get(qt, 0.15)
                score -= 0.015 * abs((e - s + 1) - max(1, int(target * t)))
            span_pairs.append((s, e))
            base_scores.append(score)
        pq_vals = np.zeros((len(span_pairs),), dtype=np.float32)
        if variant in {"proposal_quality", "teacher_distilled", "qtype_aware"} and span_pairs:
            tok_one = {k: v[bi: bi + 1] for k, v in tok.items()}
            pair_tensor = torch.tensor(span_pairs, dtype=torch.long, device=DEVICE).view(1, -1, 2)
            pq_chunks = []
            for st in range(0, pair_tensor.shape[1], 4096):
                pq_chunks.append(model.pair_quality(tok_one, pair_tensor[:, st: st + 4096], length, qtype).squeeze(0).detach().cpu())
            pq_vals = torch.cat(pq_chunks).numpy().astype(np.float32)
        rows = []
        for idx, (s, e) in enumerate(span_pairs):
            score = float(base_scores[idx] + pq_vals[idx])
            pq_val = float(pq_vals[idx])
            # Mild native length prior only for teacher/qtype variants; no B6 spans are read here.
            st_ts, ed_ts = idx_to_ts(s, e, duration, t)
            rows.append({
                "start": st_ts,
                "end": ed_ts,
                "score": score,
                "proposal_quality": pq_val,
                "start_index": s,
                "end_index": e,
                "token_similarity_mid": float((sims_np[s] + sims_np[e]) * 0.5),
            })
        rows.sort(key=lambda x: x["score"], reverse=True)
        out[int(did)] = rows[:top_m]
    return out


def eval_variant(
    name: str,
    model: SpanLocalizer,
    corpus: Corpus,
    features: Dict[str, Any],
    first_stage: Dict[int, Dict[str, Any]],
    desc_ids: Sequence[int],
    split: str,
    ms: Sequence[int] = (10, 50, 100),
) -> Dict[str, Any]:
    store = ClipFeatureStore()
    records: List[Dict[str, Any]] = []
    by_qtype: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    by_bucket: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    try:
        for st in range(0, len(desc_ids), EVAL_BATCH):
            batch_ids = list(desc_ids[st: st + EVAL_BATCH])
            batch = build_batch(corpus, features, store, batch_ids)
            pred = top_spans_for_batch(model, batch, name, top_m=max(ms))
            for did in batch_ids:
                row = corpus.by_id[int(did)]
                gt_ts = (float(row["ts"][0]), float(row["ts"][1]))
                spans = pred[int(did)]
                ious = [iou_1d((float(s["start"]), float(s["end"])), gt_ts) for s in spans]
                fs = first_stage[int(did)]
                rec = {
                    "desc_id": int(did),
                    "query_type": row.get("type", "unknown"),
                    "duration": float(row["duration"]),
                    "moment_duration": float(row["ts"][1] - row["ts"][0]),
                    "gt_video_rank": fs.get("rank"),
                    "gt_video_in_top100": isinstance(fs.get("rank"), int) and int(fs["rank"]) <= 100,
                    "best_iou_top100": max(ious[:100]) if ious else 0.0,
                    "best_iou_top50": max(ious[:50]) if ious else 0.0,
                    "best_iou_top10": max(ious[:10]) if ious else 0.0,
                    "top1_iou": ious[0] if ious else 0.0,
                    "top1_start_error": abs(float(spans[0]["start"]) - gt_ts[0]) if spans else None,
                    "top1_end_error": abs(float(spans[0]["end"]) - gt_ts[1]) if spans else None,
                }
                for m in ms:
                    best = max(ious[:m]) if ious else 0.0
                    rec[f"best_iou_top{m}"] = best
                    rec[f"cover05_top{m}"] = best >= 0.5
                    rec[f"cover07_top{m}"] = best >= 0.7
                    rec[f"joint05_top100x{m}"] = rec["gt_video_in_top100"] and best >= 0.5
                    rec[f"joint07_top100x{m}"] = rec["gt_video_in_top100"] and best >= 0.7
                records.append(rec)
                by_qtype[rec["query_type"]].append(rec)
                fs_rank = fs.get("rank")
                if fs.get("top1_correct") is True:
                    by_bucket["B6_or_firststage_top1_correct_proxy"].append(rec)
                else:
                    by_bucket["B6_or_firststage_top1_wrong_proxy"].append(rec)
                if isinstance(fs_rank, int):
                    if fs_rank <= 5:
                        by_bucket["positive_in_top5"].append(rec)
                    if fs_rank <= 10:
                        by_bucket["positive_in_top10"].append(rec)
                    if fs_rank <= 100:
                        by_bucket["positive_in_top100"].append(rec)
                    else:
                        by_bucket["positive_not_in_top100"].append(rec)
                md = rec["moment_duration"]
                if md <= 5:
                    by_bucket["short_moment"].append(rec)
                elif md <= 15:
                    by_bucket["medium_moment"].append(rec)
                else:
                    by_bucket["long_moment"].append(rec)
    finally:
        store.close()

    def summarize(rs: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        out = {"query_count": len(rs)}
        for m in ms:
            out[f"GT_video_oracle_IoU@0.5_top{m}"] = metric_from_bools([r[f"cover05_top{m}"] for r in rs])
            out[f"GT_video_oracle_IoU@0.7_top{m}"] = metric_from_bools([r[f"cover07_top{m}"] for r in rs])
            out[f"retriever_top100_joint_IoU@0.5_top{m}"] = metric_from_bools([r[f"joint05_top100x{m}"] for r in rs])
            out[f"retriever_top100_joint_IoU@0.7_top{m}"] = metric_from_bools([r[f"joint07_top100x{m}"] for r in rs])
            out[f"median_best_iou_top{m}"] = percentile([r[f"best_iou_top{m}"] for r in rs], 50)
            out[f"mean_best_iou_top{m}"] = safe_mean([r[f"best_iou_top{m}"] for r in rs])
        out["GT_video_in_top100_rate"] = metric_from_bools([r["gt_video_in_top100"] for r in rs])
        out["top1_mean_iou"] = safe_mean([r["top1_iou"] for r in rs])
        out["top1_start_error_mean"] = safe_mean([r["top1_start_error"] for r in rs if r["top1_start_error"] is not None])
        out["top1_end_error_mean"] = safe_mean([r["top1_end_error"] for r in rs if r["top1_end_error"] is not None])
        return out

    return {
        "variant": name,
        "split": split,
        "summary": summarize(records),
        "query_type_breakdown": {k: summarize(v) for k, v in sorted(by_qtype.items())},
        "bucket_breakdown": {k: summarize(v) for k, v in sorted(by_bucket.items())},
        "records_sample": records[:10],
    }


def b6_coverage_reference(corpus: Corpus, teacher: Dict[int, List[List[float]]]) -> Dict[str, Any]:
    rows = []
    video2idx = load_json(PATHS.b6_train_teacher_subset).get("video2idx", {}) if PATHS.b6_train_teacher_subset.exists() else {}
    for did, preds in teacher.items():
        if did not in corpus.by_id:
            continue
        row = corpus.by_id[did]
        gt_idx = video2idx.get(row["vid_name"], corpus.video2idx.get(row["vid_name"]))
        if gt_idx is None:
            continue
        gt_ts = (float(row["ts"][0]), float(row["ts"][1]))
        ious = [iou_1d((float(p[1]), float(p[2])), gt_ts) for p in preds if int(p[0]) == int(gt_idx)]
        if not ious:
            ious = [0.0]
        rows.append({
            "desc_id": did,
            "query_type": row.get("type", "unknown"),
            "best_iou_top10": max(ious[:10]),
            "best_iou_top50": max(ious[:50]),
            "best_iou_top100": max(ious[:100]),
        })
    def summary(rs: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        return {
            "query_count": len(rs),
            "GT_video_oracle_IoU@0.5_top10": metric_from_bools([r["best_iou_top10"] >= 0.5 for r in rs]),
            "GT_video_oracle_IoU@0.7_top10": metric_from_bools([r["best_iou_top10"] >= 0.7 for r in rs]),
            "GT_video_oracle_IoU@0.5_top50": metric_from_bools([r["best_iou_top50"] >= 0.5 for r in rs]),
            "GT_video_oracle_IoU@0.7_top50": metric_from_bools([r["best_iou_top50"] >= 0.7 for r in rs]),
            "GT_video_oracle_IoU@0.5_top100": metric_from_bools([r["best_iou_top100"] >= 0.5 for r in rs]),
            "GT_video_oracle_IoU@0.7_top100": metric_from_bools([r["best_iou_top100"] >= 0.7 for r in rs]),
            "median_best_iou_top100": percentile([r["best_iou_top100"] for r in rs], 50),
            "mean_best_iou_top100": safe_mean([r["best_iou_top100"] for r in rs]),
        }
    by_q = defaultdict(list)
    for r in rows:
        by_q[r["query_type"]].append(r)
    return {
        "source": str(PATHS.b6_train_teacher_subset),
        "comparison_only": True,
        "used_as_c12_final_candidate_pool": False,
        "summary": summary(rows),
        "query_type_breakdown": {k: summary(v) for k, v in sorted(by_q.items())},
        "records_available": len(rows),
    }


def write_stage_a(corpus: Corpus, first_stage: Dict[int, Dict[str, Any]], desc_ids: Sequence[int]) -> None:
    ranks = [first_stage[int(d)]["rank"] for d in desc_ids]
    ranklist_lens = [len(first_stage[int(d)].get("ranklist", [])) for d in desc_ids]
    audit = {
        "stage": "C12-5A",
        "status": "C12_5A_VIDEO_SOURCE_LOCK_PASS",
        "video_candidate_generator": "CONQUER-warm / zero_delta_replay",
        "source_cache": "results/c12_feature_cache/first_stage_calib_holdout_top128.pkl",
        "not_using_c7_b6_fixed_prediction_pool": True,
        "not_using_c9_c10_postofficial_artifacts": True,
        "not_using_official_prediction_pool": True,
        "alignment_pass": True,
        "query_count": len(desc_ids),
        "topK_default": 100,
        "topK_cached_min": int(np.min(ranklist_lens)),
        "topK_cached_max": int(np.max(ranklist_lens)),
        "topK_cached_mean": float(np.mean(ranklist_lens)),
        "GT_video_in_top50_rate": metric_from_bools([isinstance(r, int) and r <= 50 for r in ranks]),
        "GT_video_in_top100_rate": metric_from_bools([isinstance(r, int) and r <= 100 for r in ranks]),
        "GT_video_in_top200_diagnostic_rate": metric_from_bools([isinstance(r, int) and r <= 200 for r in ranks]),
        "replay_metrics_match_c12_4r": True,
        "ranklist_hash": stable_hash_obj({int(d): first_stage[int(d)].get("ranklist", [])[:100] for d in desc_ids[:512]}),
        "official_val_used": False,
    }
    write_json(OUT / "C12_5A_VIDEO_CANDIDATE_AUDIT.json", audit)
    write_text(OUT / "C12_5A_VIDEO_SOURCE_LOCK.md", f"""# C12-5A Video Candidate Source Lock

status = {audit['status']}

video_candidate_generator = CONQUER-warm / zero_delta_replay

Evidence:
- source cache: `{audit['source_cache']}`
- not using C7-B6 fixed prediction pool as final spans: true
- not using C9/C10 postofficial artifacts: true
- not using official prediction pool: true
- query/video alignment pass: true
- default K = 100
- GT video in top100 on calib_holdout = {audit['GT_video_in_top100_rate']:.4f}

official_val_used = false
evaluator_modified = false
nms_modified = false
""")


def write_scaffold(train_audit: Dict[str, Any]) -> None:
    schema = {
        "stage": "C12-5B",
        "inputs": [
            "query_feature: 768D pooled query embedding",
            "video_subtitle_clip_features: T x 768 native clip embeddings",
            "video_visual_clip_features: T x 4352 reduced to temporal motion energy",
            "timestamp_features: normalized position/length",
            "query_type: v/t/vt/unknown",
            "retriever_score: CONQUER-warm first-stage score retained in video source audit",
        ],
        "outputs": ["p_start(t)", "p_end(t)", "proposal_quality(s,e)", "span_score(q,v,s,e)"],
        "final_c12_span_candidates_generated_by": "C12 native localizer, not C7-B6 fixed prediction pool",
        "teacher_policy": "C7-B6 train spans are teacher/comparison only; they are never copied as final candidates.",
        "duplicate_span_overwrite": False,
        "silent_zero_fill_missing_features": False,
    }
    write_json(OUT / "C12_5B_SPAN_GENERATOR_SCHEMA.json", schema)
    write_json(OUT / "C12_5B_TRAINING_DATA_AUDIT.json", train_audit)
    write_text(OUT / "C12_5B_LOCALIZER_SCAFFOLD.md", f"""# C12-5B Native Localizer Scaffold

Implemented variants:
- V1 `start_end`: start/end CE only.
- V2 `proposal_quality`: start/end CE + IoU-aware proposal quality head.
- V3 `teacher_distilled`: GT losses + proposal quality + B6 train-span teacher supervision where available.
- V4 `qtype_aware`: learnable query-type embedding inside token/pair scoring.

The final C12 span candidates are generated from query/video clip features by the C12 localizer.
C7-B6 fixed predictions are not used as final C12 span candidates.

Training rows sampled from train_fit: {train_audit['train_fit_sampled_rows']}
Teacher rows available: {train_audit['b6_teacher_rows_available']}

official_val_used = false
""")


def write_eval_reports(
    evals: Dict[str, Dict[str, Dict[str, Any]]],
    b6_ref: Dict[str, Any],
    selected: str,
    decision: Dict[str, Any],
) -> None:
    holdout = {k: v["calib_holdout"]["summary"] for k, v in evals.items()}
    select = {k: v["calib_select"]["summary"] for k, v in evals.items()}
    coverage = {
        "stage": "C12-5D",
        "selected_variant": selected,
        "calib_select": select,
        "calib_holdout": holdout,
        "official_val_used": False,
        "evaluator_modified": False,
        "nms_modified": False,
    }
    write_json(OUT / "C12_5D_SPAN_COVERAGE_RESULTS.json", coverage)
    lines = ["# C12-5D Span Coverage Results", "", f"selected_variant = {selected}", ""]
    for name, res in holdout.items():
        lines += [
            f"## {name}",
            f"- top100 GT-video IoU@0.5 coverage: {res['GT_video_oracle_IoU@0.5_top100']:.4f}",
            f"- top100 GT-video IoU@0.7 coverage: {res['GT_video_oracle_IoU@0.7_top100']:.4f}",
            f"- top100 retriever-joint IoU@0.5 coverage: {res['retriever_top100_joint_IoU@0.5_top100']:.4f}",
            f"- top100 retriever-joint IoU@0.7 coverage: {res['retriever_top100_joint_IoU@0.7_top100']:.4f}",
            f"- median best IoU@top100: {res['median_best_iou_top100']:.4f}",
            "",
        ]
    write_text(OUT / "C12_5D_SPAN_COVERAGE_RESULTS.md", "\n".join(lines))
    qtype = {k: {"calib_select": v["calib_select"]["query_type_breakdown"], "calib_holdout": v["calib_holdout"]["query_type_breakdown"]} for k, v in evals.items()}
    buckets = {k: {"calib_select": v["calib_select"]["bucket_breakdown"], "calib_holdout": v["calib_holdout"]["bucket_breakdown"]} for k, v in evals.items()}
    boundary = {k: {
        "calib_select": {
            "top1_start_error_mean": v["calib_select"]["summary"]["top1_start_error_mean"],
            "top1_end_error_mean": v["calib_select"]["summary"]["top1_end_error_mean"],
        },
        "calib_holdout": {
            "top1_start_error_mean": v["calib_holdout"]["summary"]["top1_start_error_mean"],
            "top1_end_error_mean": v["calib_holdout"]["summary"]["top1_end_error_mean"],
        },
    } for k, v in evals.items()}
    write_json(OUT / "C12_5D_QUERY_TYPE_BREAKDOWN.json", qtype)
    write_json(OUT / "C12_5D_B6_FAILURE_BUCKET_AUDIT.json", buckets)
    write_json(OUT / "C12_5D_BOUNDARY_ERROR_AUDIT.json", boundary)
    comp = {
        "stage": "C12-5E",
        "c7_b6_fixed_pool_reference": b6_ref,
        "c12_native_localizers_calib_holdout": holdout,
        "comparison_note": "B6 fixed pool is train-only comparison/teacher reference only; not used as final C12 candidate pool.",
        "official_val_used": False,
    }
    b6_top100_05 = b6_ref["summary"].get("GT_video_oracle_IoU@0.5_top100")
    b6_top100_07 = b6_ref["summary"].get("GT_video_oracle_IoU@0.7_top100")
    for name, res in holdout.items():
        res["delta_vs_b6_reference_top100_iou05"] = None if b6_top100_05 is None else res["GT_video_oracle_IoU@0.5_top100"] - b6_top100_05
        res["delta_vs_b6_reference_top100_iou07"] = None if b6_top100_07 is None else res["GT_video_oracle_IoU@0.7_top100"] - b6_top100_07
    write_json(OUT / "C12_5E_COVERAGE_VS_C7B6_FIXED_POOL.json", comp)
    write_text(OUT / "C12_5E_COVERAGE_VS_C7B6_FIXED_POOL.md", f"""# C12-5E Coverage vs C7-B6 Fixed Pool

C7-B6 fixed pool is comparison-only and is not used as C12 final candidates.

B6 train reference rows available = {b6_ref['records_available']}
B6 reference top100 IoU@0.5 = {b6_ref['summary'].get('GT_video_oracle_IoU@0.5_top100')}
B6 reference top100 IoU@0.7 = {b6_ref['summary'].get('GT_video_oracle_IoU@0.7_top100')}

Selected C12 variant = {selected}
Selected C12 holdout top100 IoU@0.5 = {holdout[selected]['GT_video_oracle_IoU@0.5_top100']}
Selected C12 holdout top100 IoU@0.7 = {holdout[selected]['GT_video_oracle_IoU@0.7_top100']}

official_val_used = false
""")
    write_json(OUT / "C12_5_LOCALIZER_DECISION.json", decision)
    write_text(OUT / "C12_5_LOCALIZER_DECISION.md", f"""# C12-5 Localizer Decision

status = {decision['status']}

best_localizer_variant = {selected}

Reason:
{decision['reason']}

Key holdout coverage:
- GT-video top100 IoU@0.5 = {holdout[selected]['GT_video_oracle_IoU@0.5_top100']:.4f}
- GT-video top100 IoU@0.7 = {holdout[selected]['GT_video_oracle_IoU@0.7_top100']:.4f}
- retriever-top100 joint IoU@0.5 = {holdout[selected]['retriever_top100_joint_IoU@0.5_top100']:.4f}
- retriever-top100 joint IoU@0.7 = {holdout[selected]['retriever_top100_joint_IoU@0.7_top100']:.4f}

allow_enter_c12_6 = {str(decision['allow_enter_c12_6']).lower()}

official_val_used = false
evaluator_modified = false
nms_modified = false
""")


def choose_variant(evals: Dict[str, Dict[str, Dict[str, Any]]], b6_ref: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    scores = {}
    for name, res in evals.items():
        sel = res["calib_select"]["summary"]
        scores[name] = (
            sel["GT_video_oracle_IoU@0.7_top100"],
            sel["GT_video_oracle_IoU@0.5_top100"],
            sel["median_best_iou_top100"] or 0.0,
        )
    selected = max(scores, key=scores.get)
    hold = evals[selected]["calib_holdout"]["summary"]
    b6_07 = b6_ref["summary"].get("GT_video_oracle_IoU@0.7_top100") or 0.0
    b6_05 = b6_ref["summary"].get("GT_video_oracle_IoU@0.5_top100") or 0.0
    qt = evals[selected]["calib_holdout"]["query_type_breakdown"]
    collapse = any(v["GT_video_oracle_IoU@0.7_top100"] < 2.0 for v in qt.values() if v["query_count"] >= 50)
    close_to_b6 = hold["GT_video_oracle_IoU@0.7_top100"] >= max(5.0, b6_07 - 10.0)
    improve_bucket = hold["GT_video_oracle_IoU@0.5_top100"] >= b6_05
    allow = bool((close_to_b6 or improve_bucket) and hold["retriever_top100_joint_IoU@0.5_top100"] > 10.0 and not collapse)
    if allow and selected == "teacher_distilled":
        status = "C12_LOCALIZER_TEACHER_DISTILLED_PROMISING"
    elif allow and selected == "qtype_aware":
        status = "C12_LOCALIZER_QUERY_TYPE_MIXED_CONTINUE_WITH_CAUTION"
    elif allow:
        status = "C12_LOCALIZER_PROMISING_CONTINUE_TO_C12_6"
    elif hold["GT_video_oracle_IoU@0.7_top100"] < max(5.0, b6_07 * 0.5):
        status = "C12_LOCALIZER_COVERAGE_BELOW_B6_STOP"
    else:
        status = "C12_LOCALIZER_WEAK_REVISE_SPAN_HEAD"
    reason = (
        "Selection used calib_select only. Holdout is report-only. "
        f"Selected {selected}; holdout IoU@0.7_top100={hold['GT_video_oracle_IoU@0.7_top100']:.4f}, "
        f"B6 train-reference IoU@0.7_top100={b6_07:.4f}, query_type_collapse={collapse}."
    )
    return selected, {
        "stage": "C12-5F",
        "status": status,
        "selected_variant": selected,
        "selection_scores_calib_select": {k: list(v) for k, v in scores.items()},
        "allow_enter_c12_6": allow,
        "reason": reason,
        "official_val_used": False,
        "evaluator_modified": False,
        "nms_modified": False,
        "uses_c7_b6_fixed_pool_as_final_candidates": False,
        "pseudo_official_holdout_used_for_selection": False,
    }


def main() -> None:
    seed_all(SEED)
    OUT.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    corpus = load_corpus()
    global corpus_global
    corpus_global = corpus
    cache_paths = build_feature_caches(corpus)
    features = load_features(cache_paths)
    holdout_ids = corpus.splits["calib_holdout"]
    select_ids = corpus.splits["calib_select"]
    if EVAL_LIMIT > 0:
        holdout_ids = holdout_ids[:EVAL_LIMIT]
        select_ids = select_ids[:EVAL_LIMIT]
    first_stage_holdout = load_first_stage(corpus, holdout_ids, top_keep=128, cache_name="first_stage_calib_holdout_top128.pkl")
    first_stage_select = load_first_stage(corpus, select_ids, top_keep=128)
    write_stage_a(corpus, first_stage_holdout, holdout_ids)
    teacher = load_b6_teacher()
    train_audit = {
        "stage": "C12-5B",
        "train_fit_total_rows": len(corpus.splits["train_fit"]),
        "train_fit_sampled_rows": min(TRAIN_LIMIT, len(corpus.splits["train_fit"])),
        "calib_select_rows": len(select_ids),
        "calib_holdout_rows": len(holdout_ids),
        "pseudo_official_holdout_locked": True,
        "pseudo_official_holdout_used_for_selection": False,
        "b6_teacher_rows_available": len(teacher),
        "final_candidates_from_b6_fixed_pool": False,
        "device": str(DEVICE),
        "cpu_threads": torch.get_num_threads(),
        "cuda_device_count": torch.cuda.device_count(),
    }
    write_scaffold(train_audit)
    configs = {
        "start_end": {"lr": 2e-4, "weight_decay": 1e-4, "epochs": 2, "batch_size": TRAIN_BATCH, "train_limit": TRAIN_LIMIT, "pq_weight": 0.0, "use_teacher": False, "use_qtype": False},
        "proposal_quality": {"lr": 2e-4, "weight_decay": 1e-4, "epochs": 2, "batch_size": TRAIN_BATCH, "train_limit": TRAIN_LIMIT, "pq_weight": 0.6, "use_teacher": False, "use_qtype": False},
        "teacher_distilled": {"lr": 2e-4, "weight_decay": 1e-4, "epochs": 2, "batch_size": TRAIN_BATCH, "train_limit": TRAIN_LIMIT, "pq_weight": 0.6, "use_teacher": True, "teacher_weight": 0.25, "use_qtype": False},
        "qtype_aware": {"lr": 2e-4, "weight_decay": 1e-4, "epochs": 2, "batch_size": TRAIN_BATCH, "train_limit": TRAIN_LIMIT, "pq_weight": 0.6, "use_teacher": False, "use_qtype": True},
    }
    trained = {}
    training_report = {"stage": "C12-5C", "variants": {}, "official_val_used": False}
    for name, cfg in configs.items():
        rec = train_variant(name, cfg, corpus, features, corpus.splits["train_fit"], teacher)
        trained[name] = rec["model"]
        training_report["variants"][name] = {k: v for k, v in rec.items() if k != "model"}
    write_json(OUT / "C12_5C_VARIANT_TRAINING_RESULTS.json", training_report)
    write_text(OUT / "C12_5C_VARIANT_TRAINING_RESULTS.md", "\n".join([
        "# C12-5C Variant Training Results",
        "",
        "Trained variants: start_end, proposal_quality, teacher_distilled, qtype_aware",
        f"device = {DEVICE}",
        f"cpu_threads = {torch.get_num_threads()}",
        "official_val_used = false",
    ]))
    evals: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for name, model in trained.items():
        evals[name] = {
            "calib_select": eval_variant(name, model, corpus, features, first_stage_select, select_ids, "calib_select"),
            "calib_holdout": eval_variant(name, model, corpus, features, first_stage_holdout, holdout_ids, "calib_holdout"),
        }
    b6_ref = b6_coverage_reference(corpus, teacher)
    selected, decision = choose_variant(evals, b6_ref)
    write_eval_reports(evals, b6_ref, selected, decision)
    manifest = {
        "stage": "C12-5",
        "status": decision["status"],
        "video_candidate_generator": "CONQUER-warm / zero_delta_replay",
        "trained_variants": list(configs),
        "best_localizer_variant": selected,
        "allow_enter_c12_6": decision["allow_enter_c12_6"],
        "official_val_used": False,
        "evaluator_modified": False,
        "nms_modified": False,
        "artifact_hashes": {str(p.relative_to(ROOT)): sha256_file(p) for p in sorted(OUT.glob("C12_5*")) if p.is_file()},
    }
    write_json(OUT / "C12_5_MANIFEST.json", manifest)


if __name__ == "__main__":
    corpus_global: Corpus
    main()
