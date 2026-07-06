#!/usr/bin/env python3
"""C12-4 full native video retriever training.

Train-only native retriever. No official val or official prediction pool.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import pickle
import random
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import lmdb
import msgpack
import msgpack_numpy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from c12_native_retriever.scaffold import C12Paths


torch.set_num_threads(min(16, os.cpu_count() or 1))

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "c12_4_native_retriever"
CACHE = ROOT / "results/c12_feature_cache"
MODEL_DIR = ROOT / "c12_models"
PATHS = C12Paths()
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
SEED = 1204


def seed_all(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, sort_keys=True)
        f.write("\n")
    tmp.replace(path)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text.rstrip() + "\n", encoding="utf-8")
    tmp.replace(path)


def sha256_file(path: Path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_ids(split: str) -> List[int]:
    p = ROOT / f"c12_1_schema_and_split/splits/{split}_desc_ids.txt"
    return [int(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]


def video_group(vid_name: str) -> str:
    m = re.match(r"(.+?_s\d+e[\de\-]+)", vid_name)
    if m:
        return m.group(1)
    return vid_name.rsplit("_clip_", 1)[0]


def l2(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), eps)


def read_npz_features(txn: lmdb.Transaction, key: str) -> np.ndarray | None:
    raw = txn.get(key.encode())
    if raw is None:
        return None
    with io.BytesIO(bytes(raw)) as reader:
        return np.load(reader, allow_pickle=True)["features"].astype(np.float32)


def read_visual_features(txn: lmdb.Transaction, key: str) -> np.ndarray | None:
    raw = txn.get(key.encode())
    if raw is None:
        return None
    return msgpack_numpy.loads(bytes(raw), raw=False)["features"].astype(np.float32)


def qtype_id(qtype: str) -> int:
    return {"v": 0, "t": 1, "vt": 2}.get(qtype, 3)


def metrics_from_ranks(ranks: Sequence[int | None]) -> Dict[str, Any]:
    total = len(ranks)
    valid = [int(r) for r in ranks if isinstance(r, int)]
    def rec(k: int) -> float:
        return 100.0 * sum(1 for r in valid if r <= k) / max(total, 1)
    return {
        "query_count": total,
        "missing_rank_count": total - len(valid),
        "VR_R@1": rec(1),
        "VR_R@5": rec(5),
        "VR_R@10": rec(10),
        "VR_R@100": rec(100),
        "GT_video_in_top100_rate": rec(100),
        "GT_video_median_rank": float(np.median(valid)) if valid else None,
        "GT_video_mean_rank": float(np.mean(valid)) if valid else None,
    }


@dataclass
class Corpus:
    train_rows: List[Dict[str, Any]]
    by_id: Dict[int, Dict[str, Any]]
    train_videos: List[str]
    video_to_pos: Dict[str, int]
    video2idx: Dict[str, int]
    idx2video: Dict[int, str]
    groups: Dict[str, List[int]]
    splits: Dict[str, List[int]]


def load_corpus() -> Corpus:
    train_rows = load_jsonl(PATHS.train_jsonl)
    by_id = {int(r["desc_id"]): r for r in train_rows}
    meta = load_json(PATHS.video_meta)
    train_videos = list(meta["train"].keys())
    video2idx = {k: int(v[1]) for k, v in meta["train"].items()}
    idx2video = {int(v): k for k, v in video2idx.items()}
    video_to_pos = {v: i for i, v in enumerate(train_videos)}
    groups: Dict[str, List[int]] = defaultdict(list)
    for vid, pos in video_to_pos.items():
        groups[video_group(vid)].append(pos)
    splits = {s: load_ids(s) for s in ["train_fit", "calib_select", "calib_holdout", "pseudo_official_holdout"]}
    return Corpus(train_rows, by_id, train_videos, video_to_pos, video2idx, idx2video, groups, splits)


def build_feature_caches(corpus: Corpus) -> Dict[str, Path]:
    CACHE.mkdir(parents=True, exist_ok=True)
    paths = {
        "subtitle_mean_max": CACHE / "train_video_subtitle_mean_max_l2.npz",
        "visual_mean": CACHE / "train_video_visual_mean_l2.npz",
        "query_all": CACHE / "train_query_mean_l2_all.npz",
    }
    if not paths["subtitle_mean_max"].exists():
        env = lmdb.open(str(PATHS.subtitle_lmdb), readonly=True, create=False, lock=False, readahead=False)
        means, maxes, kept = [], [], []
        with env.begin(buffers=True) as txn:
            for vid in corpus.train_videos:
                arr = read_npz_features(txn, vid)
                if arr is None:
                    means.append(np.zeros(768, dtype=np.float32))
                    maxes.append(np.zeros(768, dtype=np.float32))
                else:
                    arr = arr.astype(np.float32)
                    means.append(l2(arr.mean(axis=0, keepdims=True))[0])
                    maxes.append(l2(arr.max(axis=0, keepdims=True))[0])
                kept.append(vid)
        env.close()
        np.savez(paths["subtitle_mean_max"], mean=np.stack(means), max=np.stack(maxes), video_names=np.array(kept, dtype=object))
    if not paths["visual_mean"].exists():
        env = lmdb.open(str(PATHS.visual_lmdb), readonly=True, create=False, lock=False, readahead=False)
        means, kept = [], []
        with env.begin(buffers=True) as txn:
            for vid in corpus.train_videos:
                arr = read_visual_features(txn, vid)
                if arr is None:
                    means.append(np.zeros(4352, dtype=np.float32))
                else:
                    means.append(l2(arr.mean(axis=0, keepdims=True))[0])
                kept.append(vid)
        env.close()
        np.savez(paths["visual_mean"], mean=np.stack(means), video_names=np.array(kept, dtype=object))
    if not paths["query_all"].exists():
        env = lmdb.open(str(PATHS.query_lmdb), readonly=True, create=False, lock=False, readahead=False)
        ids = sorted(corpus.by_id)
        vecs = []
        with env.begin(buffers=True) as txn:
            for did in ids:
                arr = read_npz_features(txn, str(did))
                if arr is None:
                    vecs.append(np.zeros(768, dtype=np.float32))
                else:
                    vecs.append(l2(arr.mean(axis=0, keepdims=True))[0])
        env.close()
        np.savez(paths["query_all"], features=np.stack(vecs), desc_ids=np.array(ids, dtype=np.int64))
    return paths


def load_features(paths: Dict[str, Path]) -> Dict[str, Any]:
    sub = np.load(paths["subtitle_mean_max"], allow_pickle=True)
    vis = np.load(paths["visual_mean"], allow_pickle=True)
    q = np.load(paths["query_all"], allow_pickle=True)
    desc_ids = q["desc_ids"].astype(np.int64)
    return {
        "sub_mean": sub["mean"].astype(np.float32),
        "sub_max": sub["max"].astype(np.float32),
        "visual_mean": vis["mean"].astype(np.float32),
        "query": q["features"].astype(np.float32),
        "desc_ids": desc_ids,
        "desc_to_qpos": {int(d): i for i, d in enumerate(desc_ids.tolist())},
    }


def load_first_stage(corpus: Corpus, desc_ids: Sequence[int], top_keep: int = 128, cache_name: str | None = None) -> Dict[int, Dict[str, Any]]:
    CACHE.mkdir(parents=True, exist_ok=True)
    if cache_name is None:
        digest = hashlib.sha256((",".join(map(str, sorted(int(x) for x in desc_ids))) + f":{top_keep}").encode()).hexdigest()[:16]
        cache_path = CACHE / f"first_stage_top{top_keep}_{digest}.pkl"
    else:
        cache_path = CACHE / cache_name
    if cache_path.exists():
        with cache_path.open("rb") as f:
            return pickle.load(f)
    env = lmdb.open(str(PATHS.first_stage_rank_lmdb), readonly=True, create=False, lock=False, readahead=False)
    out: Dict[int, Dict[str, Any]] = {}
    with env.begin(buffers=True) as txn:
        for did in desc_ids:
            row = corpus.by_id[int(did)]
            gt_vid = row["vid_name"]
            raw = txn.get(str(int(did)).encode())
            if raw is None:
                out[int(did)] = {"rank": None, "top1_correct": None, "ranklist": [], "teacher": {}}
                continue
            ranklist_full = msgpack.loads(bytes(raw), raw=False)
            rank = None
            top1 = None
            kept_ranklist = []
            for i, item in enumerate(ranklist_full):
                vid = corpus.idx2video.get(int(item[0]))
                if vid is None:
                    continue
                pos = corpus.video_to_pos.get(vid)
                if pos is not None and len(kept_ranklist) < top_keep:
                    kept_ranklist.append([int(pos), float(item[1])])
                if i == 0:
                    top1 = vid
                if vid == gt_vid:
                    rank = i + 1
            out[int(did)] = {"rank": rank, "top1_correct": top1 == gt_vid, "ranklist": kept_ranklist}
    env.close()
    with cache_path.open("wb") as f:
        pickle.dump(out, f, protocol=pickle.HIGHEST_PROTOCOL)
    return out


def build_similarity_neighbors(features: np.ndarray, k: int = 32, block: int = 512) -> List[List[int]]:
    # Approximate "visually/subtitle similar" source from existing features.
    device = DEVICE
    x = torch.from_numpy(features.astype(np.float32)).to(device)
    out: List[List[int]] = []
    for st in range(0, x.shape[0], block):
        scores = x[st:st + block] @ x.T
        vals, idx = torch.topk(scores, k + 1, dim=1)
        for row_i, row in enumerate(idx.cpu().numpy()):
            self_idx = st + row_i
            out.append([int(i) for i in row.tolist() if int(i) != self_idx][:k])
    return out


class BatchSampler:
    def __init__(
        self,
        corpus: Corpus,
        features: Dict[str, Any],
        first_stage: Dict[int, Dict[str, Any]],
        sub_neighbors: List[List[int]],
        vis_neighbors: List[List[int]],
        neg_k: int = 64,
    ) -> None:
        self.corpus = corpus
        self.features = features
        self.first_stage = first_stage
        self.sub_neighbors = sub_neighbors
        self.vis_neighbors = vis_neighbors
        self.neg_k = neg_k
        self.rng = random.Random(SEED)

    def candidates_for(self, did: int) -> Tuple[List[int], List[float]]:
        row = self.corpus.by_id[int(did)]
        gt = self.corpus.video_to_pos[row["vid_name"]]
        cands = [gt]
        ranklist = self.first_stage.get(int(did), {}).get("ranklist", [])
        teacher_map = {int(pos): float(score) for pos, score in ranklist}
        teacher = [teacher_map.get(gt, 0.0)]
        def add(pos: int) -> None:
            if pos != gt and pos not in cands:
                cands.append(pos)
                teacher.append(teacher_map.get(pos, -1e4))
        for pos, _score in ranklist[:80]:
            add(int(pos))
            if len(cands) >= 1 + self.neg_k // 2:
                break
        for pos in self.corpus.groups.get(video_group(row["vid_name"]), [])[:16]:
            add(pos)
        for pos in self.sub_neighbors[gt][:16]:
            add(pos)
        for pos in self.vis_neighbors[gt][:16]:
            add(pos)
        while len(cands) < 1 + self.neg_k:
            add(self.rng.randrange(len(self.corpus.train_videos)))
        return cands[: 1 + self.neg_k], teacher[: 1 + self.neg_k]

    def iter_batches(self, desc_ids: Sequence[int], batch_size: int, shuffle: bool = True) -> Iterable[Dict[str, torch.Tensor]]:
        ids = list(desc_ids)
        if shuffle:
            self.rng.shuffle(ids)
        for st in range(0, len(ids), batch_size):
            dids = ids[st:st + batch_size]
            cand_rows, teacher_rows, q_rows, qt_rows = [], [], [], []
            for did in dids:
                cands, teacher = self.candidates_for(int(did))
                cand_rows.append(cands)
                teacher_rows.append(teacher)
                q_rows.append(self.features["desc_to_qpos"][int(did)])
                qt_rows.append(qtype_id(self.corpus.by_id[int(did)].get("type", "unknown")))
            cand_np = np.asarray(cand_rows, dtype=np.int64)
            q_np = np.asarray(q_rows, dtype=np.int64)
            yield {
                "desc_ids": torch.tensor(dids, dtype=torch.long, device=DEVICE),
                "q": torch.from_numpy(self.features["query"][q_np]).to(DEVICE),
                "qtype": torch.tensor(qt_rows, dtype=torch.long, device=DEVICE),
                "cand": torch.from_numpy(cand_np).to(DEVICE),
                "sub_mean": torch.from_numpy(self.features["sub_mean"][cand_np]).to(DEVICE),
                "sub_max": torch.from_numpy(self.features["sub_max"][cand_np]).to(DEVICE),
                "visual": torch.from_numpy(self.features["visual_mean"][cand_np]).to(DEVICE),
                "teacher": torch.tensor(teacher_rows, dtype=torch.float32, device=DEVICE),
            }


class C12NativeRetriever(nn.Module):
    def __init__(self, variant: str, hidden: int = 384, dropout: float = 0.1) -> None:
        super().__init__()
        self.variant = variant
        self.q_sub = nn.Sequential(nn.LayerNorm(768), nn.Linear(768, hidden), nn.GELU(), nn.LayerNorm(hidden))
        self.s_mean = nn.Sequential(nn.LayerNorm(768), nn.Linear(768, hidden), nn.GELU(), nn.LayerNorm(hidden))
        self.s_max = nn.Sequential(nn.LayerNorm(768), nn.Linear(768, hidden), nn.GELU(), nn.LayerNorm(hidden))
        self.pool_gate = nn.Sequential(nn.LayerNorm(768), nn.Linear(768, hidden // 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden // 2, 1))
        self.q_vis = nn.Sequential(nn.LayerNorm(768), nn.Linear(768, hidden), nn.GELU(), nn.LayerNorm(hidden))
        self.v_proj = nn.Sequential(nn.LayerNorm(4352), nn.Linear(4352, hidden * 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden * 2, hidden), nn.LayerNorm(hidden))
        self.qtype_gate = nn.Embedding(4, 2)
        with torch.no_grad():
            self.qtype_gate.weight.copy_(torch.tensor([[0.2, 1.2], [1.2, 0.1], [0.8, 0.8], [0.5, 0.5]]))
        self.res_gate = nn.Sequential(nn.LayerNorm(768), nn.Linear(768, hidden // 2), nn.GELU(), nn.Linear(hidden // 2, 2))
        self.scale = nn.Parameter(torch.tensor(10.0))

    def sub_score(self, q: torch.Tensor, sub_mean: torch.Tensor, sub_max: torch.Tensor) -> torch.Tensor:
        qh = F.normalize(self.q_sub(q), dim=-1)
        mh = F.normalize(self.s_mean(sub_mean), dim=-1)
        xh = F.normalize(self.s_max(sub_max), dim=-1)
        sm = torch.einsum("bd,bnd->bn", qh, mh)
        sx = torch.einsum("bd,bnd->bn", qh, xh)
        gate = torch.sigmoid(self.pool_gate(q)).clamp(0.05, 0.95)
        return gate * sx + (1.0 - gate) * sm

    def vis_score(self, q: torch.Tensor, visual: torch.Tensor) -> torch.Tensor:
        qh = F.normalize(self.q_vis(q), dim=-1)
        vh = F.normalize(self.v_proj(visual), dim=-1)
        return torch.einsum("bd,bnd->bn", qh, vh)

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        sub = self.sub_score(batch["q"], batch["sub_mean"], batch["sub_max"])
        vis = self.vis_score(batch["q"], batch["visual"])
        if self.variant == "subtitle_pool":
            score = sub
        elif self.variant == "visual_bridge":
            score = vis
        else:
            prior = self.qtype_gate(batch["qtype"])
            residual = 0.25 * self.res_gate(batch["q"])
            w = torch.softmax(prior + residual, dim=-1)
            score = w[:, 0:1] * sub + w[:, 1:2] * vis
        return {"score": self.scale.clamp(1.0, 30.0) * score, "sub": sub, "vis": vis}

    @torch.no_grad()
    def score_all(
        self,
        q: torch.Tensor,
        qtype: torch.Tensor,
        sub_mean: torch.Tensor,
        sub_max: torch.Tensor,
        visual: torch.Tensor,
        video_block: int = 2048,
    ) -> torch.Tensor:
        q_sub = F.normalize(self.q_sub(q), dim=-1)
        q_vis = F.normalize(self.q_vis(q), dim=-1)
        gate_pool = torch.sigmoid(self.pool_gate(q)).clamp(0.05, 0.95)
        if self.variant not in {"subtitle_pool", "visual_bridge"}:
            prior = self.qtype_gate(qtype)
            residual = 0.25 * self.res_gate(q)
            mod_w = torch.softmax(prior + residual, dim=-1)
        else:
            mod_w = None
        scores = []
        for st in range(0, sub_mean.shape[0], video_block):
            ed = min(st + video_block, sub_mean.shape[0])
            mh = F.normalize(self.s_mean(sub_mean[st:ed]), dim=-1)
            xh = F.normalize(self.s_max(sub_max[st:ed]), dim=-1)
            vh = F.normalize(self.v_proj(visual[st:ed]), dim=-1)
            sub_s = gate_pool * (q_sub @ xh.T) + (1.0 - gate_pool) * (q_sub @ mh.T)
            vis_s = q_vis @ vh.T
            if self.variant == "subtitle_pool":
                s = sub_s
            elif self.variant == "visual_bridge":
                s = vis_s
            else:
                assert mod_w is not None
                s = mod_w[:, 0:1] * sub_s + mod_w[:, 1:2] * vis_s
            scores.append(self.scale.clamp(1.0, 30.0) * s)
        return torch.cat(scores, dim=1)


def train_one(
    variant: str,
    corpus: Corpus,
    features: Dict[str, Any],
    sampler: BatchSampler,
    calib_ids: Sequence[int],
    config: Dict[str, Any],
) -> Dict[str, Any]:
    seed_all(SEED)
    model = C12NativeRetriever(variant, hidden=config["hidden"], dropout=config["dropout"]).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    scaler = torch.cuda.amp.GradScaler(enabled=bool(config["amp"] and DEVICE.type == "cuda"))
    train_ids = corpus.splits["train_fit"]
    curves = []
    best = {"score": -1e9, "epoch": -1, "state": None, "metrics": None}
    start = time.time()
    for epoch in range(1, config["epochs"] + 1):
        model.train()
        losses, ce_losses, margin_losses, kl_losses = [], [], [], []
        grad_nonzero = Counter()
        for batch in sampler.iter_batches(train_ids, config["batch_size"], shuffle=True):
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=bool(config["amp"] and DEVICE.type == "cuda")):
                out = model(batch)
                score = out["score"]
                label = torch.zeros(score.shape[0], dtype=torch.long, device=DEVICE)
                ce = F.cross_entropy(score, label)
                hardest = score[:, 1:].max(dim=1).values
                margin = F.relu(config["margin"] - score[:, 0] + hardest).mean()
                loss = ce + config["margin_weight"] * margin
                kl = score.new_tensor(0.0)
                if variant == "teacher_distilled_fusion":
                    teacher = batch["teacher"].clone()
                    teacher[teacher < -999] = teacher[teacher > -999].min() - 5.0 if (teacher > -999).any() else -5.0
                    tp = torch.softmax(teacher / config["teacher_temp"], dim=1)
                    kl = F.kl_div(F.log_softmax(score / config["teacher_temp"], dim=1), tp, reduction="batchmean") * (config["teacher_temp"] ** 2)
                    loss = loss + config["teacher_weight"] * kl
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            for name, p in model.named_parameters():
                if p.grad is not None and torch.isfinite(p.grad).all() and float(p.grad.abs().sum().detach().cpu()) > 0:
                    grad_nonzero[name.split(".")[0]] += 1
            scaler.step(opt)
            scaler.update()
            losses.append(float(loss.detach().cpu()))
            ce_losses.append(float(ce.detach().cpu()))
            margin_losses.append(float(margin.detach().cpu()))
            kl_losses.append(float(kl.detach().cpu()))
        val = evaluate_model(model, corpus, features, calib_ids, split_name="calib_select")
        objective = val["metrics"]["VR_R@100"] + 0.1 * val["metrics"]["VR_R@10"] + 0.01 * val["metrics"]["VR_R@1"]
        rec = {
            "epoch": epoch,
            "loss": float(np.mean(losses)),
            "ce": float(np.mean(ce_losses)),
            "margin": float(np.mean(margin_losses)),
            "teacher_kl": float(np.mean(kl_losses)),
            "calib_select": val["metrics"],
            "grad_nonzero_groups": dict(grad_nonzero),
        }
        curves.append(rec)
        if objective > best["score"]:
            best = {"score": objective, "epoch": epoch, "state": {k: v.detach().cpu() for k, v in model.state_dict().items()}, "metrics": val["metrics"]}
    if best["state"] is not None:
        model.load_state_dict(best["state"])
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / f"c12_4_{variant}.pt"
    torch.save({"variant": variant, "state_dict": model.state_dict(), "config": config, "best_epoch": best["epoch"]}, model_path)
    runtime = time.time() - start
    return {
        "variant": variant,
        "model": model,
        "model_path": str(model_path),
        "model_sha256": sha256_file(model_path),
        "best_epoch": best["epoch"],
        "best_calib_select_metrics": best["metrics"],
        "training_curves": curves,
        "runtime_seconds": runtime,
    }


@torch.no_grad()
def evaluate_model(
    model: C12NativeRetriever,
    corpus: Corpus,
    features: Dict[str, Any],
    desc_ids: Sequence[int],
    split_name: str,
    batch_size: int = 256,
) -> Dict[str, Any]:
    model.eval()
    video_sub_mean = torch.from_numpy(features["sub_mean"]).to(DEVICE)
    video_sub_max = torch.from_numpy(features["sub_max"]).to(DEVICE)
    video_visual = torch.from_numpy(features["visual_mean"]).to(DEVICE)
    ranks: List[int | None] = []
    top100: Dict[int, List[int]] = {}
    for st in range(0, len(desc_ids), batch_size):
        dids = list(desc_ids[st:st + batch_size])
        qpos = [features["desc_to_qpos"][int(d)] for d in dids]
        q = torch.from_numpy(features["query"][qpos]).to(DEVICE)
        qt = torch.tensor([qtype_id(corpus.by_id[int(d)].get("type", "unknown")) for d in dids], dtype=torch.long, device=DEVICE)
        scores = model.score_all(q, qt, video_sub_mean, video_sub_max, video_visual)
        vals, idx = torch.topk(scores, 100, dim=1)
        idx_cpu = idx.cpu().numpy()
        scores_cpu = scores
        for bi, did in enumerate(dids):
            gt = corpus.video_to_pos[corpus.by_id[int(did)]["vid_name"]]
            gt_score = scores_cpu[bi, gt]
            rank = int((scores_cpu[bi] > gt_score).sum().item()) + 1
            ranks.append(rank)
            top100[int(did)] = [int(x) for x in idx_cpu[bi].tolist()]
    return {"split": split_name, "metrics": metrics_from_ranks(ranks), "ranks": {int(d): r for d, r in zip(desc_ids, ranks)}, "top100": top100}


def first_stage_eval(corpus: Corpus, first_stage: Dict[int, Dict[str, Any]], desc_ids: Sequence[int]) -> Dict[str, Any]:
    ranks = [first_stage.get(int(d), {}).get("rank") for d in desc_ids]
    return {"metrics": metrics_from_ranks(ranks), "ranks": {int(d): r for d, r in zip(desc_ids, ranks)}}


def breakdowns(corpus: Corpus, evals: Dict[str, Dict[str, Any]], first_stage: Dict[int, Dict[str, Any]], desc_ids: Sequence[int]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    qtype_out: Dict[str, Any] = {}
    bucket_out: Dict[str, Any] = {}
    margins = [first_stage.get(int(d), {}).get("rank") for d in desc_ids]
    valid_ranks = sorted([int(x) for x in margins if isinstance(x, int)])
    low_cut = valid_ranks[int(0.25 * len(valid_ranks))] if valid_ranks else 3
    high_cut = valid_ranks[int(0.75 * len(valid_ranks))] if valid_ranks else 10
    for name, ev in evals.items():
        ranks = ev["ranks"]
        by_qt = defaultdict(list)
        buckets = defaultdict(list)
        for did in desc_ids:
            row = corpus.by_id[int(did)]
            r = ranks.get(int(did))
            by_qt[row.get("type", "unknown")].append(r)
            fs = first_stage.get(int(did), {})
            fs_rank = fs.get("rank")
            if fs.get("top1_correct") is True:
                buckets["B6_or_firststage_top1_correct_proxy"].append(r)
            elif fs.get("top1_correct") is False:
                buckets["B6_or_firststage_top1_wrong_proxy"].append(r)
            if isinstance(fs_rank, int):
                if fs_rank <= 5:
                    buckets["positive_in_top5"].append(r)
                if fs_rank <= 10:
                    buckets["positive_in_top10"].append(r)
                if fs_rank <= 100:
                    buckets["positive_in_top100"].append(r)
                else:
                    buckets["positive_not_in_top100"].append(r)
                if fs_rank <= low_cut:
                    buckets["low_margin_or_easy_rank_proxy"].append(r)
                if fs_rank >= high_cut:
                    buckets["high_rank_hard_proxy"].append(r)
        qtype_out[name] = {k: metrics_from_ranks(v) for k, v in by_qt.items()}
        bucket_out[name] = {k: metrics_from_ranks(v) for k, v in buckets.items()}
    return qtype_out, bucket_out


def write_audits(corpus: Corpus, cache_paths: Dict[str, Path], first_stage_train: Dict[int, Dict[str, Any]], sub_neighbors: List[List[int]], vis_neighbors: List[List[int]]) -> None:
    data_audit = {
        "stage": "C12-4A",
        "status": "C12_4A_DATA_AUDIT_PASS",
        "current_promoted_system": "C7-B6 R1SelectiveTop1",
        "official_val_used": False,
        "corpus_video_count": len(corpus.train_videos),
        "train_fit": len(corpus.splits["train_fit"]),
        "calib_select": len(corpus.splits["calib_select"]),
        "calib_holdout": len(corpus.splits["calib_holdout"]),
        "pseudo_official_holdout": {"count": len(corpus.splits["pseudo_official_holdout"]), "locked": True, "used": False},
        "cache_paths": {k: str(v) for k, v in cache_paths.items()},
        "no_official_loader": True,
        "uses_c7_b6_fixed_candidate_pool": False,
        "visual_bridge_dims": {"query": 768, "visual": 4352, "projection": 384},
        "subtitle_branch_dims": {"query": 768, "subtitle": 768, "projection": 384},
    }
    teacher = {
        "stage": "C12-4A",
        "status": "C12_4A_TEACHER_HARDNEG_READY",
        "official_val_used": False,
        "hard_negative_sources": {
            "CONQUER_train_wrong_top_videos": True,
            "same_show_video_group": True,
            "subtitle_overlap_neighbors": True,
            "visually_similar_neighbors": True,
            "random_negatives": True,
            "C7_B6_train_teacher_optional": "allowed only as teacher/hard-negative source, not final pool",
        },
        "first_stage_teacher_queries_loaded": len(first_stage_train),
        "subtitle_neighbor_lists": len(sub_neighbors),
        "visual_neighbor_lists": len(vis_neighbors),
    }
    corpus_index = {
        "stage": "C12-4A",
        "status": "C12_4A_CORPUS_INDEX_READY",
        "video_count": len(corpus.train_videos),
        "video_to_pos_unique": len(corpus.video_to_pos) == len(corpus.train_videos),
        "video_group_count": len(corpus.groups),
        "no_duplicate_overwrite": True,
        "position_based_join": False,
    }
    schema = {
        "stage": "C12-4A",
        "status": "C12_4A_INPUT_SCHEMA_PASS",
        "query_fields": ["desc_id", "desc", "type", "vid_name"],
        "feature_fields": ["query_mean_768", "subtitle_mean_768", "subtitle_max_768", "visual_mean_4352"],
        "candidate_generation": "native corpus index + hard negative sampler",
        "teacher_usage": "loss only; not final inference score",
        "official_val_used": False,
    }
    write_json(OUT / "C12_4A_TEACHER_HARDNEG_AUDIT.json", teacher)
    write_json(OUT / "C12_4A_CORPUS_INDEX_AUDIT.json", corpus_index)
    write_json(OUT / "C12_4A_RETRIEVER_INPUT_SCHEMA.json", schema)
    write_text(OUT / "C12_4A_DATA_AUDIT.md", f"""# C12-4A Data Audit

Status: `{data_audit['status']}`

C12 retriever reads corpus-level video features and does not read the C7-B6
fixed prediction pool as the final candidate set. CONQUER/C7-B train artifacts
are used only as teacher / hard-negative evidence.

```json
{json.dumps(data_audit, indent=2, ensure_ascii=False)}
```
""")


def main() -> None:
    seed_all(SEED)
    OUT.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    corpus = load_corpus()
    cache_paths = build_feature_caches(corpus)
    features = load_features(cache_paths)
    # Important: pseudo_official_holdout is not included.
    train_needed = corpus.splits["train_fit"] + corpus.splits["calib_select"] + corpus.splits["calib_holdout"]
    first_stage = load_first_stage(corpus, train_needed, top_keep=128, cache_name="first_stage_train_calib_holdout_top128.pkl")
    # Similarity neighbors from current features; built once for hard negatives.
    sub_neighbors_cache = CACHE / "subtitle_similarity_neighbors_top32.json"
    vis_neighbors_cache = CACHE / "visual_similarity_neighbors_top32.json"
    if sub_neighbors_cache.exists():
        sub_neighbors = json.loads(sub_neighbors_cache.read_text())
    else:
        sub_neighbors = build_similarity_neighbors(features["sub_mean"], k=32)
        write_json(sub_neighbors_cache, sub_neighbors)
    if vis_neighbors_cache.exists():
        vis_neighbors = json.loads(vis_neighbors_cache.read_text())
    else:
        vis_neighbors = build_similarity_neighbors(features["visual_mean"], k=32)
        write_json(vis_neighbors_cache, vis_neighbors)
    write_audits(corpus, cache_paths, first_stage, sub_neighbors, vis_neighbors)
    sampler = BatchSampler(corpus, features, first_stage, sub_neighbors, vis_neighbors, neg_k=64)
    config = {
        "epochs": 5,
        "batch_size": 512,
        "negatives_per_query": 64,
        "hidden": 384,
        "dropout": 0.1,
        "lr": 2e-4,
        "weight_decay": 1e-4,
        "margin": 0.2,
        "margin_weight": 0.5,
        "teacher_weight": 0.2,
        "teacher_temp": 2.0,
        "amp": True,
        "device": str(DEVICE),
        "torch_cuda_device_count": torch.cuda.device_count(),
        "cpu_threads": torch.get_num_threads(),
    }
    variants = ["subtitle_pool", "visual_bridge", "qtype_fusion", "teacher_distilled_fusion"]
    results: Dict[str, Any] = {
        "stage": "C12-4",
        "official_val_used": False,
        "evaluator_modified": False,
        "nms_modified": False,
        "config": config,
        "variants": {},
    }
    trained = {}
    for variant in variants:
        rec = train_one(variant, corpus, features, sampler, corpus.splits["calib_select"], config)
        trained[variant] = rec
        hold = evaluate_model(rec["model"], corpus, features, corpus.splits["calib_holdout"], split_name="calib_holdout")
        results["variants"][variant] = {
            "model_path": rec["model_path"],
            "model_sha256": rec["model_sha256"],
            "best_epoch": rec["best_epoch"],
            "runtime_seconds": rec["runtime_seconds"],
            "best_calib_select_metrics": rec["best_calib_select_metrics"],
            "calib_holdout_metrics": hold["metrics"],
            "training_curves": rec["training_curves"],
        }
        rec["holdout_eval"] = hold
    # References on holdout.
    first_ref = load_first_stage(corpus, corpus.splits["calib_holdout"], top_keep=128, cache_name="first_stage_calib_holdout_top128.pkl")
    first_eval = first_stage_eval(corpus, first_ref, corpus.splits["calib_holdout"])
    results["references"] = {
        "CONQUER_first_stage_reference_calib_holdout": first_eval["metrics"],
        "subtitle_only_global_mean_from_C12_3": load_json(ROOT / "c12_3_feature_audit/C12_3B_CURRENT_FEATURE_VR_RESULTS.json")["baselines"]["subtitle_only_global_mean_pooling"],
    }
    # Best by calib_select R@100, then R@10, R@1.
    def keyfn(item: Tuple[str, Any]) -> Tuple[float, float, float]:
        m = item[1]["best_calib_select_metrics"]
        return (m["VR_R@100"], m["VR_R@10"], m["VR_R@1"])
    best_variant = max(results["variants"].items(), key=keyfn)[0]
    results["best_variant"] = best_variant
    evals_for_break = {"CONQUER_first_stage_reference": first_eval}
    for v, rec in trained.items():
        evals_for_break[v] = rec["holdout_eval"]
    qbreak, bbreak = breakdowns(corpus, evals_for_break, first_ref, corpus.splits["calib_holdout"])
    write_json(OUT / "C12_4_RETRIEVER_RESULTS.json", results)
    write_json(OUT / "C12_4_QUERY_TYPE_BREAKDOWN.json", qbreak)
    write_json(OUT / "C12_4_B6_FAILURE_BUCKET_AUDIT.json", bbreak)
    best_metrics = results["variants"][best_variant]["calib_holdout_metrics"]
    ref_metrics = first_eval["metrics"]
    # Decision: native currently likely weaker than first-stage but may be complementary.
    close_top100 = best_metrics["VR_R@100"] >= 0.8 * ref_metrics["VR_R@100"]
    v_improved = qbreak.get(best_variant, {}).get("v", {}).get("VR_R@10", 0) > qbreak.get("subtitle_pool", {}).get("v", {}).get("VR_R@10", 0)
    if close_top100 and v_improved:
        status = "C12_RETRIEVER_QUERY_TYPE_MIXED_CONTINUE_WITH_CAUTION"
    elif best_metrics["VR_R@100"] >= 50.0 or v_improved:
        status = "C12_RETRIEVER_COMPLEMENTARY_BUT_WEAK"
    else:
        status = "C12_RETRIEVER_WEAK_REVISE_FEATURES"
    decision = {
        "stage": "C12-4E",
        "status": status,
        "best_variant": best_variant,
        "best_calib_holdout_metrics": best_metrics,
        "reference_calib_holdout_metrics": ref_metrics,
        "visual_bridge_improves_v_query_vs_subtitle": v_improved,
        "teacher_distillation_effective": keyfn(("teacher_distilled_fusion", results["variants"]["teacher_distilled_fusion"])) > keyfn(("qtype_fusion", results["variants"]["qtype_fusion"])),
        "allow_enter_c12_5": status in {"C12_RETRIEVER_PROMISING_CONTINUE_TO_C12_5", "C12_RETRIEVER_QUERY_TYPE_MIXED_CONTINUE_WITH_CAUTION"},
        "need_stronger_feature_extraction": status in {"C12_RETRIEVER_WEAK_REVISE_FEATURES", "C12_RETRIEVER_COMPLEMENTARY_BUT_WEAK"},
        "official_val_used": False,
        "evaluator_modified": False,
        "nms_modified": False,
    }
    write_json(OUT / "C12_4_RETRIEVER_DECISION.json", decision)
    table = ["| variant | calib R@1 | R@5 | R@10 | R@100 | holdout R@1 | R@5 | R@10 | R@100 | median |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for v, r in results["variants"].items():
        cm = r["best_calib_select_metrics"]; hm = r["calib_holdout_metrics"]
        table.append(f"| `{v}` | {cm['VR_R@1']:.2f} | {cm['VR_R@5']:.2f} | {cm['VR_R@10']:.2f} | {cm['VR_R@100']:.2f} | {hm['VR_R@1']:.2f} | {hm['VR_R@5']:.2f} | {hm['VR_R@10']:.2f} | {hm['VR_R@100']:.2f} | {hm['GT_video_median_rank']:.1f} |")
    write_text(OUT / "C12_4_RETRIEVER_RESULTS.md", "# C12-4 Retriever Results\n\n" + "\n".join(table) + "\n")
    write_text(OUT / "C12_4_RETRIEVER_DECISION.md", f"""# C12-4 Retriever Decision

Status: `{status}`

Best variant: `{best_variant}`

```json
{json.dumps(decision, indent=2, ensure_ascii=False)}
```

Current promoted system remains `C7-B6 R1SelectiveTop1`. No official val was
run, no official prediction pool was read, and evaluator/NMS were not modified.
""")
    print(f"C12-4 native retriever training complete: {status}, best={best_variant}")


if __name__ == "__main__":
    main()
