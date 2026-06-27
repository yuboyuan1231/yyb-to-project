#!/usr/bin/env python

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

"""Train C2/C3 CONQUER-RLEM evidence heads from C1 JSONL artifacts.

This script does not touch CONQUER weights.  It trains a compact MLP over
(q,v,p)-level scalar evidence exported by ``export_conquer_evidence.py``.
"""

import argparse
import json
import os
import math
from pathlib import Path
from typing import Dict, Optional

import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from rlem.evidence_dataset import FeatureStats, StreamingEvidenceDataset, evidence_collate
from rlem.evidence_model import EvidenceMLP
from rlem.feature_schema import R2_FEATURES, parse_feature_list
from rlem.io_utils import read_json, write_json


CACHE_LABEL_NAMES = ["y_joint", "y_joint_05", "y_joint_07", "y_bd", "m_bd", "y_fp"]


def compute_loss(out: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor], args) -> Dict[str, torch.Tensor]:
    q_joint = torch.sigmoid(out["q_joint_logit"])
    q_bd = torch.sigmoid(out["q_bd_logit"])

    loss_joint_reg = F.smooth_l1_loss(q_joint, batch["y_joint"], reduction="mean")
    loss_joint_bin = F.binary_cross_entropy_with_logits(
        out["q_joint_logit"], batch["y_joint_05"], reduction="mean"
    )

    bd_mask = batch["m_bd"] > 0.5
    if torch.any(bd_mask):
        loss_bd = F.smooth_l1_loss(q_bd[bd_mask], batch["y_bd"][bd_mask], reduction="mean")
    else:
        loss_bd = out["q_bd_logit"].sum() * 0.0

    pos_weight = torch.tensor(float(args.fp_pos_weight), device=out["e_fp_logit"].device)
    loss_fp = F.binary_cross_entropy_with_logits(
        out["e_fp_logit"], batch["y_fp"], pos_weight=pos_weight, reduction="mean"
    )

    total = (
        args.lambda_joint * loss_joint_reg
        + args.lambda_joint_bin * loss_joint_bin
        + args.lambda_bd * loss_bd
        + args.lambda_fp * loss_fp
    )
    return {
        "loss": total,
        "loss_joint_reg": loss_joint_reg.detach(),
        "loss_joint_bin": loss_joint_bin.detach(),
        "loss_bd": loss_bd.detach(),
        "loss_fp": loss_fp.detach(),
    }


@torch.no_grad()
def evaluate(model, loader, device, args, max_batches: Optional[int] = None) -> Dict[str, float]:
    model.eval()
    totals = {}
    n = 0
    for batch_idx, batch in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(batch["x"])
        losses = compute_loss(out, batch, args)
        bs = batch["x"].size(0)
        for key, val in losses.items():
            totals[key] = totals.get(key, 0.0) + float(val) * bs
        n += bs
    if n == 0:
        return {"loss": float("inf")}
    return {k: v / n for k, v in totals.items()}


def load_desc_ids(path: Optional[str]):
    if not path:
        return None
    with open(path, "r", encoding="utf-8") as f:
        ids = {line.strip() for line in f if line.strip()}
    if not ids:
        raise ValueError(f"No desc_ids found in {path}")
    return ids


def make_loader(path: str, stats: FeatureStats, args, train: bool, allowed_desc_ids):
    ds = StreamingEvidenceDataset(
        path,
        stats=stats,
        max_rows=args.max_train_rows if train else args.max_val_rows,
        shuffle_buffer=args.shuffle_buffer if train else 0,
        seed=args.seed + (0 if train else 999),
        neg_keep_prob=args.neg_keep_prob if train else 1.0,
        allowed_desc_ids=allowed_desc_ids,
    )
    return DataLoader(
        ds,
        batch_size=args.batch_size,
        collate_fn=evidence_collate,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )


def load_cache(cache_dir: str, feature_names):
    manifest = read_json(os.path.join(cache_dir, "cache_manifest.json"))
    if manifest.get("status") != "PASS":
        raise ValueError("C2 cache manifest is not PASS")
    if manifest["feature_names"] != feature_names:
        raise ValueError("C2 cache feature_names do not match requested features")
    if manifest["label_names"] != CACHE_LABEL_NAMES:
        raise ValueError("C2 cache label schema mismatch")
    cache = {"manifest": manifest}
    for split in ["train_fit", "train_calib"]:
        entry = manifest["splits"][split]
        cache[split] = {
            "x": np.load(entry["x_path"], mmap_mode="r"),
            "y": np.load(entry["y_path"], mmap_mode="r"),
        }
    return cache


def iter_cached_batches(x, y, batch_size, train, seed, max_rows=None, block_rows=262144):
    n = len(x) if max_rows is None else min(len(x), int(max_rows))
    label_cols = {name: idx for idx, name in enumerate(CACHE_LABEL_NAMES)}
    if train:
        rng = np.random.default_rng(seed)
        blocks = [(start, min(start + block_rows, n)) for start in range(0, n, block_rows)]
        for block_idx in rng.permutation(len(blocks)):
            start, end = blocks[int(block_idx)]
            indices = np.arange(start, end, dtype=np.int64)
            rng.shuffle(indices)
            for offset in range(0, len(indices), batch_size):
                idx = indices[offset:offset + batch_size]
                xb = np.asarray(x[idx], dtype=np.float32)
                yb = np.asarray(y[idx], dtype=np.float32)
                yield {
                    "x": torch.from_numpy(xb),
                    **{
                        name: torch.from_numpy(yb[:, col])
                        for name, col in label_cols.items()
                    },
                }
    else:
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            xb = np.array(x[start:end], dtype=np.float32, copy=True)
            yb = np.array(y[start:end], dtype=np.float32, copy=True)
            yield {
                "x": torch.from_numpy(xb),
                **{
                    name: torch.from_numpy(yb[:, col])
                    for name, col in label_cols.items()
                },
            }


@torch.no_grad()
def evaluate_cached(model, x, y, device, args):
    model.eval()
    totals = {}
    n = 0
    batches = iter_cached_batches(
        x, y, args.batch_size, train=False, seed=args.seed + 999,
        max_rows=args.max_val_rows, block_rows=args.shuffle_block_rows,
    )
    for batch in batches:
        batch = {k: v.to(device) for k, v in batch.items()}
        losses = compute_loss(model(batch["x"]), batch, args)
        bs = batch["x"].size(0)
        for key, val in losses.items():
            totals[key] = totals.get(key, 0.0) + float(val) * bs
        n += bs
    if n == 0:
        return {"loss": float("inf")}
    return {k: v / n for k, v in totals.items()}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_jsonl", required=True)
    parser.add_argument("--val_jsonl", default=None)
    parser.add_argument("--train_desc_ids", required=True)
    parser.add_argument("--val_desc_ids", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--cache_dir", default=None)
    parser.add_argument("--features", default="c2_no_r2", help="c2_no_r2/c3 or comma-separated feature names")
    parser.add_argument("--stats_json", default=None, help="Reuse existing stats JSON instead of fitting on train_jsonl")
    parser.add_argument("--stats_max_rows", type=int, default=None)
    parser.add_argument("--max_train_rows", type=int, default=None)
    parser.add_argument("--max_val_rows", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--shuffle_buffer", type=int, default=10000)
    parser.add_argument("--shuffle_block_rows", type=int, default=262144)
    parser.add_argument("--neg_keep_prob", type=float, default=1.0)
    parser.add_argument("--lambda_joint", type=float, default=1.0)
    parser.add_argument("--lambda_joint_bin", type=float, default=0.25)
    parser.add_argument("--lambda_bd", type=float, default=1.0)
    parser.add_argument("--lambda_fp", type=float, default=0.50)
    parser.add_argument("--fp_pos_weight", type=float, default=1.0)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--save_every_epoch", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    train_desc_ids = load_desc_ids(args.train_desc_ids)
    val_desc_ids = load_desc_ids(args.val_desc_ids)
    overlap = train_desc_ids & val_desc_ids
    if overlap:
        raise ValueError(f"train/calib desc_id overlap: {len(overlap)}")
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    feature_names = parse_feature_list(args.features)
    forbidden_r2 = sorted(set(feature_names) & set(R2_FEATURES))
    if forbidden_r2:
        raise ValueError(f"Unavailable r2 features must be disabled: {forbidden_r2}")
    if args.cache_dir:
        stats = FeatureStats.from_dict(read_json(os.path.join(args.cache_dir, "feature_stats.json")))
        if stats.feature_names != feature_names:
            raise ValueError("cache feature_names do not match --features")
    elif args.stats_json:
        stats = FeatureStats.from_dict(read_json(args.stats_json))
        if stats.feature_names != feature_names:
            raise ValueError("--stats_json feature_names do not match --features")
    else:
        stats = FeatureStats.fit(
            [args.train_jsonl],
            feature_names,
            max_rows=args.stats_max_rows,
            allowed_desc_ids=train_desc_ids,
        )
    if not all(math.isfinite(v) for v in stats.mean + stats.std):
        raise ValueError("Non-finite feature statistics")
    write_json(os.path.join(args.output_dir, "feature_stats.json"), stats.to_dict())
    write_json(os.path.join(args.output_dir, "train_args.json"), vars(args))

    cache = load_cache(args.cache_dir, feature_names) if args.cache_dir else None

    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    model = EvidenceMLP(input_dim=len(feature_names), hidden_dim=args.hidden_dim, dropout=args.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val = float("inf")
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        if cache:
            train_loader = iter_cached_batches(
                cache["train_fit"]["x"], cache["train_fit"]["y"],
                args.batch_size, train=True, seed=args.seed + epoch - 1,
                max_rows=args.max_train_rows, block_rows=args.shuffle_block_rows,
            )
            train_total = math.ceil(
                (len(cache["train_fit"]["x"]) if args.max_train_rows is None else min(len(cache["train_fit"]["x"]), args.max_train_rows))
                / args.batch_size
            )
        else:
            train_loader = make_loader(
                args.train_jsonl, stats, args, train=True, allowed_desc_ids=train_desc_ids
            )
            train_total = None
        running = {}
        n_seen = 0
        pbar = tqdm(train_loader, total=train_total, desc=f"train epoch {epoch}")
        for batch in pbar:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            out = model(batch["x"])
            losses = compute_loss(out, batch, args)
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            bs = batch["x"].size(0)
            n_seen += bs
            for key, val in losses.items():
                running[key] = running.get(key, 0.0) + float(val.detach()) * bs
            pbar.set_postfix(loss=running["loss"] / max(n_seen, 1))

        train_metrics = {f"train_{k}": v / max(n_seen, 1) for k, v in running.items()}
        val_path = args.val_jsonl or args.train_jsonl
        if cache:
            val_metrics = {
                f"val_{k}": v for k, v in evaluate_cached(
                    model, cache["train_calib"]["x"], cache["train_calib"]["y"],
                    device, args,
                ).items()
            }
            cur_val = val_metrics.get("val_loss", float("inf"))
        elif val_path:
            val_loader = make_loader(
                val_path, stats, args, train=False, allowed_desc_ids=val_desc_ids
            )
            val_metrics = {f"val_{k}": v for k, v in evaluate(model, val_loader, device, args).items()}
            cur_val = val_metrics.get("val_loss", float("inf"))
        else:
            val_metrics = {}
            cur_val = train_metrics.get("train_loss", float("inf"))

        record = {"epoch": epoch, **train_metrics, **val_metrics}
        history.append(record)
        print(json.dumps(record, indent=2))

        ckpt = {
            "model_state": model.state_dict(),
            "model_cfg": {
                "input_dim": len(feature_names),
                "hidden_dim": args.hidden_dim,
                "dropout": args.dropout,
            },
            "feature_stats": stats.to_dict(),
            "train_args": vars(args),
            "epoch": epoch,
            "metrics": record,
        }
        if args.save_every_epoch:
            torch.save(ckpt, os.path.join(args.output_dir, f"model_epoch{epoch}.pt"))
        if cur_val <= best_val:
            best_val = cur_val
            torch.save(ckpt, os.path.join(args.output_dir, "model_best.pt"))

    write_json(os.path.join(args.output_dir, "history.json"), history)
    print(f"Saved best checkpoint to {os.path.join(args.output_dir, 'model_best.pt')}")


if __name__ == "__main__":
    main()
