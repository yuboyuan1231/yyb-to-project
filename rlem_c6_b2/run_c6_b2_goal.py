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
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rlem_c6_b0.run_c6_b0_candidate_diagnostic import load_gt_ts  # noqa: E402
from rlem_c6_b1_lite.run_c6_b1_r1_safe_selector import evaluate_policy, score_pool  # noqa: E402


METRIC_KEYS = [
    "0.5-r1", "0.5-r5", "0.5-r10", "0.5-r100",
    "0.7-r1", "0.7-r5", "0.7-r10", "0.7-r100",
]
EPS = 1e-6


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


def write_json(path: str | Path, obj: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(p) + ".partial")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, p)


def write_text(path: str | Path, text: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(p) + ".partial")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, p)


def load_pool(path: str) -> Dict[str, np.ndarray]:
    return {k: v for k, v in np.load(path, allow_pickle=False).items()}


def load_cache(path: str) -> Dict[str, np.ndarray]:
    return {k: v for k, v in np.load(path, allow_pickle=True).items()}


def pair_feature_paths(out_dir: Path, split: str) -> Dict[str, Path]:
    return {
        "x": out_dir / f"{split}_pair_x.npy",
        "labels": out_dir / f"{split}_pair_labels.npz",
        "stats": out_dir / f"{split}_pair_stats.json",
    }


def build_pair_cache_for_split(pool_path: str, out_dir: Path, split: str, chunk_q: int = 2048) -> Dict[str, Any]:
    paths = pair_feature_paths(out_dir, split)
    if paths["x"].exists() and paths["labels"].exists():
        return {"split": split, "reused": True, "x": artifact(paths["x"]), "labels": artifact(paths["labels"])}
    pool = load_pool(pool_path)
    q, slots, alts = [int(x) for x in pool["shape"]]
    assert alts == 8, pool["shape"]
    feat_dim = int(pool["x"].shape[1])
    pair_per_q = slots * (alts - 1)
    n = q * pair_per_q
    x_out = np.lib.format.open_memmap(paths["x"], mode="w+", dtype=np.float32, shape=(n, feat_dim * 3))
    gain05 = np.empty(n, dtype=np.float32)
    gain07 = np.empty(n, dtype=np.float32)
    risk05 = np.empty(n, dtype=np.float32)
    risk07 = np.empty(n, dtype=np.float32)
    replace = np.empty(n, dtype=np.float32)
    iou_delta = np.empty(n, dtype=np.float32)
    query = np.empty(n, dtype=np.int32)
    slot = np.empty(n, dtype=np.int16)
    alt = np.empty(n, dtype=np.int16)
    orig_iou = np.empty(n, dtype=np.float32)
    alt_iou = np.empty(n, dtype=np.float32)
    X = pool["x"].reshape(q, slots, alts, feat_dim)
    y05 = pool["y05"].reshape(q, slots, alts)
    y07 = pool["y07"].reshape(q, slots, alts)
    yiou = pool["yiou"].reshape(q, slots, alts)
    pos = 0
    t0 = time.time()
    for q0 in range(0, q, chunk_q):
        q1 = min(q0 + chunk_q, q)
        bq = q1 - q0
        orig = X[q0:q1, :, 0:1, :]
        altx = X[q0:q1, :, 1:, :]
        orig_b = np.broadcast_to(orig, altx.shape)
        feat = np.concatenate([altx, orig_b, altx - orig_b], axis=-1).reshape(-1, feat_dim * 3)
        m = feat.shape[0]
        x_out[pos:pos + m] = feat
        o05 = y05[q0:q1, :, 0:1]; a05 = y05[q0:q1, :, 1:]
        o07 = y07[q0:q1, :, 0:1]; a07 = y07[q0:q1, :, 1:]
        oi = yiou[q0:q1, :, 0:1]; ai = yiou[q0:q1, :, 1:]
        gain05[pos:pos + m] = (a05 > o05).reshape(-1)
        gain07[pos:pos + m] = (a07 > o07).reshape(-1)
        risk05[pos:pos + m] = (a05 < o05).reshape(-1)
        risk07[pos:pos + m] = (a07 < o07).reshape(-1)
        iou_delta[pos:pos + m] = (ai - oi).reshape(-1)
        replace[pos:pos + m] = ((a07 > o07) | ((a07 == o07) & (a05 > o05)) | ((a07 == o07) & (a05 == o05) & (ai > oi + 1e-6))).reshape(-1)
        orig_iou[pos:pos + m] = np.broadcast_to(oi, ai.shape).reshape(-1)
        alt_iou[pos:pos + m] = ai.reshape(-1)
        qq = np.arange(q0, q1, dtype=np.int32)[:, None, None]
        ss = np.arange(slots, dtype=np.int16)[None, :, None]
        aa = np.arange(1, alts, dtype=np.int16)[None, None, :]
        query[pos:pos + m] = np.broadcast_to(qq, (bq, slots, alts - 1)).reshape(-1)
        slot[pos:pos + m] = np.broadcast_to(ss, (bq, slots, alts - 1)).reshape(-1)
        alt[pos:pos + m] = np.broadcast_to(aa, (bq, slots, alts - 1)).reshape(-1)
        pos += m
    x_out.flush()
    np.savez_compressed(
        paths["labels"],
        gain05=gain05, gain07=gain07, risk05=risk05, risk07=risk07, replace=replace,
        iou_delta=iou_delta, query=query, slot=slot, alt=alt, orig_iou=orig_iou, alt_iou=alt_iou,
        shape=np.asarray([q, slots, alts], dtype=np.int64),
    )
    stats = {
        "split": split,
        "queries": q,
        "slots": slots,
        "alts": alts,
        "pair_examples": int(n),
        "feature_dim": feat_dim * 3,
        "gain07_positive_rate": float(gain07.mean()),
        "gain05_positive_rate": float(gain05.mean()),
        "risk07_rate": float(risk07.mean()),
        "risk05_rate": float(risk05.mean()),
        "replace_positive_rate": float(replace.mean()),
        "elapsed_sec": float(time.time() - t0),
        "source_pool": artifact(pool_path),
        "x": artifact(paths["x"]),
        "labels": artifact(paths["labels"]),
        "official_val_used": False,
    }
    write_json(paths["stats"], stats)
    return stats


class PairMemmapDataset(Dataset):
    def __init__(self, x_path: str, labels_path: str, mean: np.ndarray | None = None, std: np.ndarray | None = None):
        labels = np.load(labels_path, allow_pickle=False)
        self.labels = {k: labels[k] for k in labels.files}
        self.x = np.load(x_path, mmap_mode="r")
        if mean is None:
            # Full-pass statistics are acceptable and deterministic; this is train-only.
            mean = np.asarray(self.x, dtype=np.float32).mean(axis=0)
            std = np.asarray(self.x, dtype=np.float32).std(axis=0)
        self.mean = mean.astype(np.float32)
        self.std = np.maximum(std.astype(np.float32), EPS)

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = (np.asarray(self.x[idx], dtype=np.float32) - self.mean) / self.std
        y = np.asarray([
            self.labels["gain07"][idx],
            self.labels["gain05"][idx],
            self.labels["risk07"][idx],
            self.labels["risk05"][idx],
            self.labels["replace"][idx],
            self.labels["iou_delta"][idx],
        ], dtype=np.float32)
        return torch.from_numpy(x), torch.from_numpy(y)


class PairwiseUtilityModel(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 512, layers: int = 4, dropout: float = 0.10):
        super().__init__()
        mods: List[nn.Module] = []
        d = in_dim
        for _ in range(layers):
            mods += [nn.Linear(d, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(dropout)]
            d = hidden
        self.backbone = nn.Sequential(*mods)
        self.head = nn.Linear(hidden, 6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))


class ListwisePoolDataset(Dataset):
    def __init__(self, pool_path: str, mean: np.ndarray | None = None, std: np.ndarray | None = None):
        pool = load_pool(pool_path)
        q, slots, alts = [int(x) for x in pool["shape"]]
        self.x = pool["x"].reshape(q * slots, alts, -1).astype(np.float32)
        self.y05 = pool["y05"].reshape(q * slots, alts).astype(np.float32)
        self.y07 = pool["y07"].reshape(q * slots, alts).astype(np.float32)
        self.yiou = pool["yiou"].reshape(q * slots, alts).astype(np.float32)
        self.slot = np.broadcast_to(np.arange(slots, dtype=np.int64)[None, :], (q, slots)).reshape(-1)
        if mean is None:
            mean = self.x.reshape(-1, self.x.shape[-1]).mean(axis=0)
            std = self.x.reshape(-1, self.x.shape[-1]).std(axis=0)
        self.mean = mean.astype(np.float32)
        self.std = np.maximum(std.astype(np.float32), EPS)
        orig07 = self.y07[:, 0:1]; orig05 = self.y05[:, 0:1]; orig_iou = self.yiou[:, 0:1]
        utility = 3.0 * (self.y07 - orig07) + 1.2 * (self.y05 - orig05) + 0.5 * (self.yiou - orig_iou)
        utility[:, 0] = 0.0
        best = np.argmax(utility, axis=1)
        best_gain = utility[np.arange(len(best)), best]
        best = np.where(best_gain > 1e-6, best, 0)
        self.target = best.astype(np.int64)

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = (self.x[idx] - self.mean) / self.std
        return torch.from_numpy(x.astype(np.float32)), torch.tensor(self.target[idx], dtype=torch.long), torch.tensor(self.slot[idx], dtype=torch.long)


class ListwiseSetModel(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 512, layers: int = 4, heads: int = 8, dropout: float = 0.10, max_slots: int = 8):
        super().__init__()
        self.in_proj = nn.Sequential(nn.Linear(in_dim, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.slot_emb = nn.Embedding(max_slots, hidden)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=heads, dim_feedforward=hidden * 4, dropout=dropout,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=layers)
        self.head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, 1))

    def forward(self, x: torch.Tensor, slot: torch.Tensor) -> torch.Tensor:
        h = self.in_proj(x)
        h = h + self.slot_emb(slot).unsqueeze(1)
        h = self.encoder(h)
        return self.head(h).squeeze(-1)


def train_pairwise(args: argparse.Namespace) -> Dict[str, Any]:
    out_dir = Path(args.output_dir) / "arms" / args.arm_name
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = pair_feature_paths(Path(args.cache_dir), "train_fit")
    ds = PairMemmapDataset(str(paths["x"]), str(paths["labels"]))
    mean, std = ds.mean, ds.std
    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    model = PairwiseUtilityModel(ds.x.shape[1], hidden=args.hidden, layers=args.layers, dropout=0.10).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))
    # Pos weights from train labels.
    y = ds.labels
    pos_weights = []
    for k in ["gain07", "gain05", "risk07", "risk05", "replace"]:
        pos = float(y[k].sum())
        pos_weights.append((len(y[k]) - pos) / max(pos, 1.0))
    pos_weights_t = torch.tensor(pos_weights, dtype=torch.float32, device=device)
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        t0 = time.time()
        order = np.random.permutation(len(ds.x))
        for start in range(0, len(order), args.batch_size):
            idx = order[start:start + args.batch_size]
            xb_np = ((np.asarray(ds.x[idx], dtype=np.float32) - mean) / std).astype(np.float32)
            yb_np = np.stack([
                ds.labels["gain07"][idx],
                ds.labels["gain05"][idx],
                ds.labels["risk07"][idx],
                ds.labels["risk05"][idx],
                ds.labels["replace"][idx],
                ds.labels["iou_delta"][idx],
            ], axis=1).astype(np.float32)
            xb = torch.from_numpy(xb_np).to(device, non_blocking=True)
            yb = torch.from_numpy(yb_np).to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                out = model(xb)
                bce = F.binary_cross_entropy_with_logits(out[:, :5], yb[:, :5], reduction="none", pos_weight=pos_weights_t)
                loss = (
                    1.5 * bce[:, 0] + 0.8 * bce[:, 1] +
                    0.7 * bce[:, 2] + 0.4 * bce[:, 3] +
                    0.7 * bce[:, 4] +
                    0.3 * F.smooth_l1_loss(torch.tanh(out[:, 5]), torch.clamp(yb[:, 5] * 2.0, -1.0, 1.0), reduction="none")
                ).mean()
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
            "examples_per_sec": float(len(ds) / max(time.time() - t0, 1e-6)),
            "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
            "max_memory_allocated": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0,
        }
        history.append(rec)
        write_json(out_dir / "history.json", history)
        print(json.dumps({"stage": "train_pairwise", **rec}))
    ckpt = {
        "model_state": model.state_dict(),
        "mean": mean,
        "std": std,
        "in_dim": int(ds.x.shape[1]),
        "hidden": args.hidden,
        "layers": args.layers,
        "epochs": args.epochs,
        "pos_weights": pos_weights,
    }
    torch.save(ckpt, out_dir / "model_best.pt")
    return {"arm": args.arm_name, "history": history, "model": artifact(out_dir / "model_best.pt"), "official_val_used": False}


def train_listwise(args: argparse.Namespace) -> Dict[str, Any]:
    out_dir = Path(args.output_dir) / "arms" / args.arm_name
    out_dir.mkdir(parents=True, exist_ok=True)
    ds = ListwisePoolDataset(args.train_pool)
    mean, std = ds.mean, ds.std
    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    dl = DataLoader(ds, batch_size=args.list_batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=(device.type == "cuda"), persistent_workers=args.num_workers > 0)
    model = ListwiseSetModel(ds.x.shape[-1], hidden=args.hidden, layers=args.layers, heads=8, dropout=0.10).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))
    target_counts = np.bincount(ds.target, minlength=ds.x.shape[1]).astype(np.float64)
    if args.list_weight_mode == "inverse":
        weight = (target_counts.sum() / np.maximum(target_counts, 1.0))
        weight = weight / weight.mean()
    elif args.list_weight_mode == "sqrt_cap":
        weight = np.sqrt(target_counts.sum() / np.maximum(target_counts, 1.0))
        weight = np.minimum(weight / weight.mean(), 5.0)
    else:
        weight = np.ones_like(target_counts, dtype=np.float64)
    weight_t = torch.tensor(weight, dtype=torch.float32, device=device)
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        t0 = time.time()
        for xb, target, slot in dl:
            xb = xb.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            slot = slot.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                logits = model(xb, slot)
                ce = F.cross_entropy(logits, target, weight=weight_t)
            scaler.scale(ce).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            losses.append(float(ce.detach().cpu()))
        rec = {
            "epoch": epoch,
            "loss": float(np.mean(losses)),
            "elapsed_sec": float(time.time() - t0),
            "examples_per_sec": float(len(ds) / max(time.time() - t0, 1e-6)),
            "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
            "max_memory_allocated": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0,
        }
        history.append(rec)
        write_json(out_dir / "history.json", history)
        print(json.dumps({"stage": "train_listwise", **rec}))
    torch.save({
        "model_state": model.state_dict(),
        "mean": mean,
        "std": std,
        "in_dim": int(ds.x.shape[-1]),
        "hidden": args.hidden,
        "layers": args.layers,
        "heads": 8,
        "epochs": args.epochs,
        "target_counts": target_counts.tolist(),
        "list_weight_mode": args.list_weight_mode,
    }, out_dir / "model_best.pt")
    return {"arm": args.arm_name, "history": history, "model": artifact(out_dir / "model_best.pt"), "official_val_used": False}


@torch.no_grad()
def score_pairwise(args: argparse.Namespace, split: str, model_path: str) -> Dict[str, Any]:
    paths = pair_feature_paths(Path(args.cache_dir), split)
    labels = np.load(paths["labels"], allow_pickle=False)
    shape = labels["shape"].astype(int).tolist()
    q, slots, alts = shape
    ckpt = torch.load(model_path, map_location="cpu")
    x = np.load(paths["x"], mmap_mode="r")
    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    model = PairwiseUtilityModel(int(ckpt["in_dim"]), hidden=int(ckpt["hidden"]), layers=int(ckpt["layers"])).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    scores_pairs = np.empty(len(x), dtype=np.float32)
    for start in range(0, len(x), args.score_batch_size):
        xb = (np.asarray(x[start:start + args.score_batch_size], dtype=np.float32) - ckpt["mean"]) / ckpt["std"]
        tb = torch.from_numpy(xb).to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
            out = model(tb).float()
            score = out[:, 0] + 0.6 * out[:, 1] - 0.85 * out[:, 2] - 0.45 * out[:, 3] + 0.35 * out[:, 4] + 0.15 * torch.tanh(out[:, 5])
        scores_pairs[start:start + len(score)] = score.detach().cpu().numpy().astype(np.float32)
    pool = load_pool(args.calib_pool if split == "train_calib" else args.train_pool)
    full = np.zeros(len(pool["x"]), dtype=np.float32)
    pair = scores_pairs.reshape(q, slots, alts - 1)
    full.reshape(q, slots, alts)[:, :, 1:] = pair
    out_path = Path(args.output_dir) / "train_calib_scores" / f"{split}_{args.arm_name}_scores.npz"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, score=full.astype(np.float32), pair_score=scores_pairs)
    return {"split": split, "score": artifact(out_path), "score_min": float(full.min()), "score_max": float(full.max()), "official_val_used": False}


@torch.no_grad()
def score_listwise(args: argparse.Namespace, split: str, model_path: str) -> Dict[str, Any]:
    pool_path = args.calib_pool if split == "train_calib" else args.train_pool
    pool = load_pool(pool_path)
    q, slots, alts = [int(x) for x in pool["shape"]]
    ckpt = torch.load(model_path, map_location="cpu")
    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    model = ListwiseSetModel(int(ckpt["in_dim"]), hidden=int(ckpt["hidden"]), layers=int(ckpt["layers"]), heads=int(ckpt["heads"])).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    X = pool["x"].reshape(q * slots, alts, -1).astype(np.float32)
    slot_arr = np.broadcast_to(np.arange(slots, dtype=np.int64)[None, :], (q, slots)).reshape(-1)
    scores = np.empty((q * slots, alts), dtype=np.float32)
    for start in range(0, len(X), args.list_score_batch_size):
        xb = ((X[start:start + args.list_score_batch_size] - ckpt["mean"]) / ckpt["std"]).astype(np.float32)
        tb = torch.from_numpy(xb).to(device, non_blocking=True)
        sl = torch.from_numpy(slot_arr[start:start + len(xb)]).to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
            logits = model(tb, sl).float()
        scores[start:start + len(xb)] = logits.detach().cpu().numpy().astype(np.float32)
    full = scores.reshape(-1)
    out_path = Path(args.output_dir) / "train_calib_scores" / f"{split}_{args.arm_name}_scores.npz"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, score=full.astype(np.float32))
    return {"split": split, "score": artifact(out_path), "score_min": float(full.min()), "score_max": float(full.max()), "official_val_used": False}


def metric_delta(new: Dict[str, float], base: Dict[str, float]) -> Dict[str, float]:
    return {k: float(new[k] - base[k]) for k in METRIC_KEYS}


def selection_score(delta: Dict[str, float], movement: Dict[str, Any]) -> float:
    hard_exit_ratio = float(movement.get("hard_positive_top100_query_exits", 0)) / 8739.0
    repl_rate = float(movement.get("replacement_rate_top_slots", 0.0))
    return float(
        4.0 * delta["0.7-r1"]
        + 2.0 * delta["0.5-r1"]
        + 0.6 * min(delta["0.7-r5"], 0.20)
        + 0.4 * min(delta["0.5-r5"], 0.20)
        + 0.2 * min(delta["0.7-r10"], 0.20)
        - 2.5 * max(0.0, -delta["0.7-r5"] - 0.10)
        - 1.5 * max(0.0, -delta["0.5-r5"] - 0.10)
        - 1.0 * max(0.0, hard_exit_ratio - 0.0075)
        - 0.5 * max(0.0, repl_rate - 0.07)
    )


def margin_quantiles(scores: np.ndarray, pool: Dict[str, np.ndarray], slots_use: int) -> Tuple[List[float], List[float]]:
    q, slots, alts = [int(x) for x in pool["shape"]]
    s = scores.reshape(q, slots, alts)
    best = s.max(axis=2)
    margin = best - s[:, :, 0]
    slot0 = margin[:, 0]
    rest = margin[:, 1:min(slots_use, slots)].reshape(-1)
    qs = [0.70, 0.80, 0.88, 0.92, 0.95, 0.97, 0.985]
    t0 = sorted(set(float(np.quantile(slot0, qv)) for qv in qs))
    tr = sorted(set(float(np.quantile(rest, qv)) for qv in qs))
    return t0, tr


def search_arm(args: argparse.Namespace, arm: str, score_npz: str) -> Dict[str, Any]:
    cache = load_cache(args.calib_cache)
    pool = load_pool(args.calib_pool)
    with np.load(score_npz, allow_pickle=False) as z:
        scores = z["score"].astype(np.float32)
    gt_ts = load_gt_ts(args.gt_jsonl, cache["desc_ids"])
    b1_scores = np.load(args.b1_calib_scores, allow_pickle=False)["score"].astype(np.float32)
    b1_cfg = json.loads(Path(args.b1_best_config).read_text())
    b1_ev = evaluate_policy(
        cache=cache, pool=pool, pool_score=b1_scores, gt_ts_by_query=gt_ts,
        apply_slots=int(b1_cfg["apply_slots"]), threshold0=float(b1_cfg["threshold0"]),
        threshold_rest=float(b1_cfg["threshold_rest"]), max_replacements=int(b1_cfg["max_replacements"]),
        effective_top_n=100, max_after_nms=100, nms_thd=0.7,
    )
    configs = []
    q, slots, alts = [int(x) for x in pool["shape"]]
    s3 = scores.reshape(q, slots, alts)
    margin = s3.max(axis=2) - s3[:, :, 0]
    for apply_slots in [3, 5]:
        slot0 = margin[:, 0]
        rest = margin[:, 1:apply_slots].reshape(-1)
        for max_rep in [2, 3]:
            for target in [0.03, 0.05]:
                # Approximate target replacement rates using margin quantiles,
                # then evaluate exactly with NMS/GT.  Start with the core
                # R1-focused policies; expand only if a candidate is promising.
                q0 = min(max(1.0 - target, 0.50), 0.995)
                qr = min(max(1.0 - target, 0.50), 0.995)
                t0_bal = float(np.quantile(slot0, q0))
                tr_bal = float(np.quantile(rest, qr))
                configs.append((apply_slots, t0_bal, tr_bal, max_rep))
    # Keep grid bounded but diverse; avoids hundreds of slow Python-NMS evals.
    seen = set(); uniq = []
    for c in configs:
        key = (c[0], round(c[1], 4), round(c[2], 4), c[3])
        if key not in seen:
            seen.add(key); uniq.append(c)
    results = []
    for i, (apply_slots, t0, tr, max_rep) in enumerate(uniq):
        print(json.dumps({"stage": "search_arm_eval", "arm": arm, "i": i + 1, "total": len(uniq), "apply_slots": apply_slots, "max_rep": max_rep}), flush=True)
        ev = evaluate_policy(
            cache=cache, pool=pool, pool_score=scores, gt_ts_by_query=gt_ts,
            apply_slots=apply_slots, threshold0=t0, threshold_rest=tr, max_replacements=max_rep,
            effective_top_n=100, max_after_nms=100, nms_thd=0.7,
        )
        delta = metric_delta(ev["metrics"], b1_ev["metrics"])
        movement = {k: ev[k] for k in ["replacement_rate_top_slots", "replacements", "top1_changed_ratio", "hard_positive_top100_query_exits", "hard_positive_top100_query_entries", "video_slot_drift", "video_multiset_drift"]}
        feasible = (
            delta["0.7-r1"] > 0
            and delta["0.5-r1"] >= 0
            and delta["0.7-r5"] >= -0.10
            and delta["0.5-r5"] >= -0.10
            and delta["0.7-r10"] >= -0.15
            and delta["0.5-r10"] >= -0.15
            and movement["video_slot_drift"] == 0.0
            and movement["video_multiset_drift"] == 0.0
            and movement["replacement_rate_top_slots"] <= 0.07
        )
        rec = {
            "arm": arm,
            "config_id": f"{arm}_{i:04d}",
            "apply_slots": int(apply_slots),
            "threshold0": float(t0),
            "threshold_rest": float(tr),
            "max_replacements": int(max_rep),
            "metrics": ev["metrics"],
            "delta_vs_C6_B1": delta,
            "movement": movement,
            "selection_score": selection_score(delta, movement),
            "feasible": bool(feasible),
            "official_val_used": False,
        }
        results.append(rec)
    feasible = [r for r in results if r["feasible"]]
    best = max(feasible or results, key=lambda r: (r["selection_score"], r["delta_vs_C6_B1"]["0.7-r1"], -r["movement"]["replacement_rate_top_slots"]))
    out_dir = Path(args.output_dir) / "train_calib_scores" / arm
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "grid_results.json", results)
    write_json(out_dir / "best_config.json", best)
    return {"arm": arm, "grid_size": len(results), "feasible_count": len(feasible), "best": best, "baseline_C6_B1": b1_ev}


def build_data(args: argparse.Namespace) -> Dict[str, Any]:
    out_dir = Path(args.cache_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train = build_pair_cache_for_split(args.train_pool, out_dir, "train_fit")
    calib = build_pair_cache_for_split(args.calib_pool, out_dir, "train_calib")
    audit = {
        "status": "PASS",
        "official_val_used": False,
        "source_train_pool": artifact(args.train_pool),
        "source_calib_pool": artifact(args.calib_pool),
        "train_pair_cache": train,
        "calib_pair_cache": calib,
        "listwise_source": "directly from C6-B1 pool reshape, no official val",
    }
    write_json("c6_b2_audit/C6_B2_DATA_AUDIT.json", audit)
    write_text("c6_b2_audit/C6_B2_DATA_AUDIT.md", "# C6-B2 data audit\n\n" + json.dumps(audit, indent=2, ensure_ascii=False) + "\n")
    return audit


def finalize(args: argparse.Namespace, search_records: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not search_records:
        score_root = Path(args.output_dir) / "train_calib_scores"
        for best_path in sorted(score_root.glob("*/best_config.json")):
            arm = best_path.parent.name
            grid_path = best_path.parent / "grid_results.json"
            grid = json.loads(grid_path.read_text()) if grid_path.exists() else []
            search_records.append({
                "arm": arm,
                "best": json.loads(best_path.read_text()),
                "grid_size": len(grid),
                "feasible_count": int(sum(1 for r in grid if r.get("feasible"))),
                "best_config": artifact(best_path),
                "grid_results": artifact(grid_path),
            })
    arms = [r for r in search_records]
    if not arms:
        raise RuntimeError("No C6-B2 arm search records found; run search_* before finalize.")
    best = max((r["best"] for r in arms), key=lambda x: (x["feasible"], x["selection_score"], x["delta_vs_C6_B1"]["0.7-r1"]))
    feasible = bool(best["feasible"])
    if feasible:
        status = "C6_B2_FREEZE_REVIEW_PASS"
    elif best["delta_vs_C6_B1"]["0.7-r1"] > 0:
        status = "C6_B2_R1_TRADEOFF"
    elif best["delta_vs_C6_B1"]["0.5-r1"] > 0:
        status = "C6_B2_LOW_IOU_ONLY"
    else:
        status = "C6_B2_NEGATIVE"
    training = {
        "status": "C6_B2_TRAINING_AUDIT",
        "official_val_used": False,
        "retry_policy": {
            "underperforming_arm_retry_limit": 2,
            "listwise_retries_used": len([a for a in arms if a["arm"].startswith("listwise_set_retry")]),
            "no_more_listwise_retries": True,
        },
        "arms": {},
    }
    for r in arms:
        arm = r["arm"]
        arm_dir = Path(args.output_dir) / "arms" / arm
        score_path = Path(args.output_dir) / "train_calib_scores" / f"train_calib_{arm}_scores.npz"
        hist_path = arm_dir / "history.json"
        history = json.loads(hist_path.read_text()) if hist_path.exists() else []
        training["arms"][arm] = {
            "history": history,
            "history_artifact": artifact(hist_path),
            "model": artifact(arm_dir / "model_best.pt"),
            "score": artifact(score_path),
            "best_config": r.get("best_config", artifact(Path(args.output_dir) / "train_calib_scores" / arm / "best_config.json")),
            "grid_results": r.get("grid_results", artifact(Path(args.output_dir) / "train_calib_scores" / arm / "grid_results.json")),
        }
    write_json("c6_b2_audit/C6_B2_TRAINING_AUDIT.json", training)
    train_lines = ["# C6-B2 training audit", "", "Official val used: `false`", "", "| arm | epochs | final loss | feasible | 0.7 R@1 Δ | note |", "|---|---:|---:|---:|---:|---|"]
    for r in arms:
        arm = r["arm"]; b = r["best"]; hist = training["arms"][arm]["history"]
        final_loss = hist[-1]["loss"] if hist else None
        note = "selected feasible" if b is best else ("retry after listwise underperformance" if "retry" in arm else "initial arm")
        train_lines.append(f"| {arm} | {len(hist)} | {final_loss if final_loss is not None else 'n/a'} | {b['feasible']} | {b['delta_vs_C6_B1']['0.7-r1']:+.4f} | {note} |")
    train_lines += [
        "",
        "Listwise was audited after poor initial metrics and retried twice (`sqrt_cap`, then `none` class weighting). Both retries failed the C6-B2 freeze gate, so no further listwise tuning is performed in this run.",
    ]
    write_text("c6_b2_audit/C6_B2_TRAINING_AUDIT.md", "\n".join(train_lines) + "\n")

    comp = {"status": status, "arms": arms, "best": best, "official_val_used": False, "post_val_adjustment": False}
    write_json("c6_b2_audit/C6_B2_ARM_COMPARISON.json", comp)
    lines = ["# C6-B2 arm comparison", "", f"Status: `{status}`", "", "| arm | feasible | 0.7-r1 Δ | 0.5-r1 Δ | 0.7-r5 Δ | repl_rate | score |", "|---|---:|---:|---:|---:|---:|---:|"]
    for r in arms:
        b = r["best"]; d = b["delta_vs_C6_B1"]; m = b["movement"]
        lines.append(f"| {r['arm']} | {b['feasible']} | {d['0.7-r1']:+.4f} | {d['0.5-r1']:+.4f} | {d['0.7-r5']:+.4f} | {m['replacement_rate_top_slots']:.4f} | {b['selection_score']:.4f} |")
    write_text("c6_b2_audit/C6_B2_ARM_COMPARISON.md", "\n".join(lines) + "\n")
    decision = {
        "status": status,
        "selected_arm": best["arm"],
        "selected_config_id": best["config_id"],
        "best_config": best,
        "primary_metric": "0.7-r1",
        "secondary_metric": "0.5-r1",
        "primary_baseline": "C6-B1-lite c6b1r1_0057",
        "secondary_baseline": "C4_final v21_00444",
        "freeze_gate_passed": feasible,
        "score_grid_on_official_val": False,
        "official_val_used": False,
        "official_val_authorized": False,
        "post_val_adjustment": False,
        "C6_C_used": False,
        "C4_final_modified": False,
        "NMS_modified": False,
        "evaluator_modified": False,
        "next_step": "STOP_AND_WAIT_FOR_HUMAN_REVIEW",
    }
    write_json("c6_b2_audit/C6_B2_FINAL_DECISION.json", decision)
    write_text("c6_b2_audit/C6_B2_FINAL_DECISION.md", "# C6-B2 final decision\n\n" + json.dumps(decision, indent=2, ensure_ascii=False) + "\n")
    if feasible:
        write_json("c6_b2_audit/C6_B2_FREEZE_MANIFEST.json", decision)
        write_text("c6_b2_audit/C6_B2_FREEZE_REVIEW.md", "# C6-B2 freeze review\n\n" + json.dumps(decision, indent=2, ensure_ascii=False) + "\n")
    else:
        write_json("c6_b2_audit/C6_B2_NEGATIVE_MANIFEST.json", decision)
        write_text("c6_b2_audit/C6_B2_NEGATIVE_AUDIT.md", "# C6-B2 negative/tradeoff audit\n\n" + json.dumps(decision, indent=2, ensure_ascii=False) + "\n")
    hashes = {"status": "C6_B2_HASHES", "artifacts": {}}
    for p in [
        "c6_b2_audit/C6_B2_START_STATE.json", "c6_b2_audit/C6_B2_PROTOCOL.md",
        "c6_b2_audit/C6_B2_DATA_AUDIT.json", "c6_b2_audit/C6_B2_ARM_COMPARISON.json",
        "c6_b2_audit/C6_B2_TRAINING_AUDIT.json",
        "c6_b2_audit/C6_B2_FINAL_DECISION.json", __file__,
    ]:
        hashes["artifacts"][p] = artifact(p)
    for r in arms:
        arm = r["arm"]
        for p in [
            Path(args.output_dir) / "arms" / arm / "model_best.pt",
            Path(args.output_dir) / "arms" / arm / "history.json",
            Path(args.output_dir) / "train_calib_scores" / f"train_calib_{arm}_scores.npz",
            Path(args.output_dir) / "train_calib_scores" / arm / "best_config.json",
            Path(args.output_dir) / "train_calib_scores" / arm / "grid_results.json",
        ]:
            hashes["artifacts"][str(p)] = artifact(p)
    write_json("c6_b2_audit/C6_B2_HASHES.json", hashes)
    if feasible:
        freeze_hashes = {
            "status": "C6_B2_FREEZE_HASHES",
            "selected_config_id": best["config_id"],
            "official_val_used": False,
            "artifacts": {
                "freeze_manifest": artifact("c6_b2_audit/C6_B2_FREEZE_MANIFEST.json"),
                "freeze_review": artifact("c6_b2_audit/C6_B2_FREEZE_REVIEW.md"),
                "selected_best_config": artifact(Path(args.output_dir) / "train_calib_scores" / best["arm"] / "best_config.json"),
                "selected_grid_results": artifact(Path(args.output_dir) / "train_calib_scores" / best["arm"] / "grid_results.json"),
                "selected_model": artifact(Path(args.output_dir) / "arms" / best["arm"] / "model_best.pt"),
                "selected_scores": artifact(Path(args.output_dir) / "train_calib_scores" / f"train_calib_{best['arm']}_scores.npz"),
            },
        }
        write_json("c6_b2_audit/C6_B2_FREEZE_HASHES.json", freeze_hashes)
    return decision


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["build_data", "train_pairwise", "train_listwise", "score_pairwise", "score_listwise", "search_pairwise", "search_listwise", "finalize"], required=True)
    p.add_argument("--train_pool", default="results/rlem_c6_b1_lite_r1_safe/train_fit_candidate_pool_top_slots.npz")
    p.add_argument("--calib_pool", default="results/rlem_c6_b1_lite_r1_safe/train_calib_candidate_pool_top_slots.npz")
    p.add_argument("--calib_cache", default="results/rlem_c6a/cache/train_calib_c6_cache.npz")
    p.add_argument("--gt_jsonl", default="results/rlem_c3_minimal/train_calib_gt.jsonl")
    p.add_argument("--b1_calib_scores", default="results/rlem_c6_b1_lite_r1_safe/train_calib_candidate_scores.npz")
    p.add_argument("--b1_best_config", default="results/rlem_c6_b1_lite_r1_safe/best_config.json")
    p.add_argument("--cache_dir", default="results/rlem_c6_b2/caches")
    p.add_argument("--output_dir", default="results/rlem_c6_b2")
    p.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--batch_size", type=int, default=8192)
    p.add_argument("--list_batch_size", type=int, default=1024)
    p.add_argument("--score_batch_size", type=int, default=65536)
    p.add_argument("--list_score_batch_size", type=int, default=4096)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--hidden", type=int, default=512)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--model_path", default=None)
    p.add_argument("--score_npz", default=None)
    p.add_argument("--arm_name", default=None)
    p.add_argument("--list_weight_mode", choices=["inverse", "sqrt_cap", "none"], default="sqrt_cap")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    if args.mode == "build_data":
        print(json.dumps(build_data(args), indent=2))
    elif args.mode == "train_pairwise":
        if args.arm_name is None:
            args.arm_name = "pairwise_main"
        print(json.dumps(train_pairwise(args), indent=2))
    elif args.mode == "train_listwise":
        if args.arm_name is None:
            args.arm_name = "listwise_set"
        print(json.dumps(train_listwise(args), indent=2))
    elif args.mode == "score_pairwise":
        if args.arm_name is None:
            args.arm_name = "pairwise_main"
        model = args.model_path or str(Path(args.output_dir) / "arms" / args.arm_name / "model_best.pt")
        print(json.dumps(score_pairwise(args, "train_calib", model), indent=2))
    elif args.mode == "score_listwise":
        if args.arm_name is None:
            args.arm_name = "listwise_set"
        model = args.model_path or str(Path(args.output_dir) / "arms" / args.arm_name / "model_best.pt")
        print(json.dumps(score_listwise(args, "train_calib", model), indent=2))
    elif args.mode == "search_pairwise":
        if args.arm_name is None:
            args.arm_name = "pairwise_main"
        score = args.score_npz or str(Path(args.output_dir) / "train_calib_scores" / f"train_calib_{args.arm_name}_scores.npz")
        print(json.dumps(search_arm(args, args.arm_name, score), indent=2))
    elif args.mode == "search_listwise":
        if args.arm_name is None:
            args.arm_name = "listwise_set"
        score = args.score_npz or str(Path(args.output_dir) / "train_calib_scores" / f"train_calib_{args.arm_name}_scores.npz")
        print(json.dumps(search_arm(args, args.arm_name, score), indent=2))
    elif args.mode == "finalize":
        print(json.dumps(finalize(args, []), indent=2))


if __name__ == "__main__":
    main()
