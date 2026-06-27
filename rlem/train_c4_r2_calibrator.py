#!/usr/bin/env python
"""Train the fixed external C4-r2-cal residual MLP on train_fit only."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from rlem.c4_protocol import C4Protocol, write_protocol_manifest  # noqa: E402
from rlem.c4_r2_dataset import ArrayStandardizer, VideoR2Dataset, load_npz_features  # noqa: E402
from rlem.c4_r2_model import VideoR2Calibrator, model_config_from_instance  # noqa: E402
from rlem.c4_lite_utils import write_json  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train_video_npz", required=True)
    p.add_argument("--calib_video_npz", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--manifest_json", required=True)
    p.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    p.add_argument("--seed", type=int, default=13)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=4096)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--num_blocks", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--lambda_quality", type=float, default=0.25)
    return p.parse_args()


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def binary_auc_safe(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y = y_true.astype(bool)
    n_pos = int(y.sum())
    n_neg = int(len(y) - n_pos)
    if not n_pos or not n_neg:
        return float("nan")
    order = np.argsort(y_score, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(order) + 1, dtype=np.float64)
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def average_precision_safe(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y = y_true.astype(bool)
    n_pos = int(y.sum())
    if not n_pos:
        return float("nan")
    order = np.argsort(-y_score, kind="mergesort")
    ranked = y[order]
    precision = np.cumsum(ranked) / np.arange(1, len(ranked) + 1)
    return float(precision[ranked].sum() / n_pos)


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def evaluate(model, loader, device, lambda_quality: float, collect: bool = False):
    model.eval()
    loss_sum = bce_sum = quality_sum = 0.0
    count = 0
    video_logits, quality_logits, yrels, yious = [], [], [], []
    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device, non_blocking=True)
            y_rel = batch["y_rel"].to(device, non_blocking=True)
            y_iou = batch["y_iou"].to(device, non_blocking=True)
            out = model(x)
            bce = F.binary_cross_entropy_with_logits(out["video_logit"], y_rel)
            quality = F.smooth_l1_loss(torch.sigmoid(out["quality_logit"]), y_iou)
            n = len(x)
            loss_sum += float((bce + lambda_quality * quality).item()) * n
            bce_sum += float(bce.item()) * n
            quality_sum += float(quality.item()) * n
            count += n
            video_logits.append(out["video_logit"].cpu().numpy())
            quality_logits.append(out["quality_logit"].cpu().numpy())
            yrels.append(y_rel.cpu().numpy())
            yious.append(y_iou.cpu().numpy())
    vl = np.concatenate(video_logits).astype(np.float32)
    ql = np.concatenate(quality_logits).astype(np.float32)
    yr = np.concatenate(yrels).astype(np.float32)
    yi = np.concatenate(yious).astype(np.float32)
    qp = 1.0 / (1.0 + np.exp(-np.clip(ql, -40, 40)))
    metrics = {
        "loss": loss_sum / count,
        "video_bce": bce_sum / count,
        "quality_smooth_l1": quality_sum / count,
        "video_auc": binary_auc_safe(yr, vl),
        "video_ap": average_precision_safe(yr, vl),
        "quality_pearson_best_iou": _corr(qp, yi),
        "quality_mae_best_iou": float(np.mean(np.abs(qp - yi))),
    }
    return (metrics, vl, ql) if collect else metrics


def main():
    args = parse_args()
    protocol = C4Protocol(stage="c4_r2_train_calibrator", split="train")
    protocol.validate()
    if (args.hidden_dim, args.num_blocks, args.dropout, args.seed) != (256, 2, 0.1, 13):
        raise ValueError("C4-r2-cal architecture is frozen at hidden=256, blocks=2, dropout=0.1, seed=13")
    set_seed(args.seed)
    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    x_train, names = load_npz_features(args.train_video_npz)
    standardizer = ArrayStandardizer.fit(x_train, names)
    train_ds = VideoR2Dataset(args.train_video_npz, standardizer)
    calib_ds = VideoR2Dataset(args.calib_video_npz, standardizer)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
        pin_memory=device.type == "cuda", persistent_workers=args.num_workers > 0,
    )
    calib_loader = DataLoader(
        calib_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
        pin_memory=device.type == "cuda", persistent_workers=args.num_workers > 0,
    )
    model = VideoR2Calibrator(len(names), 256, 2, 0.1).float().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_path = out_dir / "model_best.pt"
    history = []
    best = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        seen = 0
        for batch in tqdm(train_loader, desc=f"C4-r2-cal epoch {epoch}/{args.epochs}"):
            x = batch["x"].to(device, non_blocking=True)
            y_rel = batch["y_rel"].to(device, non_blocking=True)
            y_iou = batch["y_iou"].to(device, non_blocking=True)
            out = model(x)
            bce = F.binary_cross_entropy_with_logits(out["video_logit"], y_rel)
            quality = F.smooth_l1_loss(torch.sigmoid(out["quality_logit"]), y_iou)
            loss = bce + args.lambda_quality * quality
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            running += float(loss.item()) * len(x)
            seen += len(x)
        calib = evaluate(model, calib_loader, device, args.lambda_quality)
        rec = {"epoch": epoch, "train_loss": running / seen, **{f"calib_{k}": v for k, v in calib.items()}}
        history.append(rec)
        print(json.dumps(rec, sort_keys=True))
        if best is None or rec["calib_loss"] < best["calib_loss"]:
            best = rec
            state = {
                "model_state": model.state_dict(),
                "model_cfg": model_config_from_instance(model),
                "standardizer": standardizer.to_dict(),
                "train_args": vars(args),
                "best_epoch": epoch,
                "best_metrics": rec,
                "official_val_used": False,
                "stage": "c4_r2_cal_train_fit_select_train_calib",
            }
            partial = str(best_path) + ".partial"
            torch.save(state, partial)
            os.replace(partial, best_path)
    write_json(str(out_dir / "history.json"), history)

    checkpoint = torch.load(best_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    best_metrics, video_logits, quality_logits = evaluate(model, calib_loader, device, args.lambda_quality, collect=True)
    with np.load(args.calib_video_npz, allow_pickle=True) as calib_data:
        np.savez_compressed(
            out_dir / "train_calib_logits.npz",
            video_logit=video_logits, quality_logit=quality_logits,
            group_id=calib_data["group_id"].astype(np.int32),
            query_index=calib_data["query_index"].astype(np.int32),
            video_idx=calib_data["video_idx"].astype(np.int32),
            label_relevant=calib_data["label_relevant"].astype(np.float32),
            label_best_iou=calib_data["label_best_iou"].astype(np.float32),
        )
    audit = {
        "status": "PASS",
        "training_split": "train_fit_only",
        "checkpoint_selection_split": "train_calib_only",
        "train_video_rows": len(train_ds),
        "train_positive_video_rows": int(train_ds.y_rel.sum().item()),
        "train_negative_video_rows": int(len(train_ds) - train_ds.y_rel.sum().item()),
        "calib_video_rows": len(calib_ds),
        "calib_positive_video_rows": int(calib_ds.y_rel.sum().item()),
        "calib_negative_video_rows": int(len(calib_ds) - calib_ds.y_rel.sum().item()),
        "architecture": {**checkpoint["model_cfg"], "activation": "GELU", "normalization": "LayerNorm", "precision": "float32", "seed": 13},
        "parameter_count": int(sum(p.numel() for p in model.parameters())),
        "epochs": args.epochs,
        "best_epoch": int(checkpoint["best_epoch"]),
        "best_train_calib_metrics": best_metrics,
        "loss_curves": {
            "epoch": [rec["epoch"] for rec in history],
            "train_loss": [rec["train_loss"] for rec in history],
            "train_calib_loss": [rec["calib_loss"] for rec in history],
        },
        "history_json": str(out_dir / "history.json"),
        "model_best": str(best_path),
        "model_best_sha256": sha256_file(str(best_path)),
        "train_calib_logits": str(out_dir / "train_calib_logits.npz"),
        "train_calib_logits_sha256": sha256_file(str(out_dir / "train_calib_logits.npz")),
        "capacity_search": False,
        "official_val_used": False,
        "post_val_adjustment": False,
    }
    write_protocol_manifest(args.manifest_json, protocol, audit)
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
