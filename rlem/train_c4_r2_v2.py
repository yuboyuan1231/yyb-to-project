#!/usr/bin/env python
"""Train one fixed C4-r2-cal-v2 group on train_fit and select on train_calib."""

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

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from rlem.c4_protocol import C4Protocol, write_protocol_manifest  # noqa: E402
from rlem.c4_r2_dataset import ArrayStandardizer, VideoR2Dataset, load_npz_features  # noqa: E402
from rlem.c4_r2_v2_model import build_v2_model, model_config  # noqa: E402
from rlem.c4_lite_utils import write_json  # noqa: E402


def args_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--group", choices=["v2_ab", "v2_cb", "v2_acb"], required=True)
    p.add_argument("--train_video_npz", required=True)
    p.add_argument("--calib_video_npz", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--audit_json", required=True)
    p.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=4096)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=13)
    return p.parse_args()


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def auc(y, s):
    y = y.astype(bool); np_, nn_ = int(y.sum()), int((~y).sum())
    if not np_ or not nn_: return float("nan")
    order = np.argsort(s, kind="stable"); ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(order) + 1)
    return float((ranks[y].sum() - np_ * (np_ + 1) / 2) / (np_ * nn_))


def ap(y, s):
    y = y.astype(bool); n = int(y.sum())
    if not n: return float("nan")
    yy = y[np.argsort(-s, kind="stable")]
    return float((np.cumsum(yy)[yy] / np.arange(1, len(yy) + 1)[yy]).sum() / n)


def corr(a, b):
    return float(np.corrcoef(a, b)[0, 1]) if len(a) > 1 and np.std(a) and np.std(b) else float("nan")


def loss_parts(out, batch, group):
    rel = F.binary_cross_entropy_with_logits(out["rel_logit"], batch["y_rel"])
    quality = F.smooth_l1_loss(torch.sigmoid(out["quality_logit"]), batch["y_iou"])
    if group == "v2_ab":
        return rel + quality, {"rel_bce": rel, "quality_smooth_l1": quality}
    iou05 = F.binary_cross_entropy_with_logits(out["iou05_logit"], batch["y_05"])
    iou07 = F.binary_cross_entropy_with_logits(out["iou07_logit"], batch["y_07"])
    return rel + iou05 + 0.5 * iou07 + 0.5 * quality, {
        "rel_bce": rel, "iou05_bce": iou05, "iou07_bce": iou07,
        "quality_smooth_l1": quality,
    }


def evaluate(model, loader, device, group, collect=False):
    model.eval(); sums = {}; count = 0
    outputs = {k: [] for k in ["rel_logit", "iou05_logit", "iou07_logit", "quality_logit"]}
    labels = {k: [] for k in ["y_rel", "y_05", "y_07", "y_iou"]}
    with torch.no_grad():
        for raw in loader:
            batch = {k: v.to(device, non_blocking=True) for k, v in raw.items() if k.startswith("y_") or k == "x"}
            out = model(batch["x"]); loss, parts = loss_parts(out, batch, group); n = len(batch["x"])
            sums["loss"] = sums.get("loss", 0.0) + float(loss) * n
            for k, value in parts.items(): sums[k] = sums.get(k, 0.0) + float(value) * n
            count += n
            for k in outputs: outputs[k].append(out[k].cpu().numpy())
            for k in labels: labels[k].append(batch[k].cpu().numpy())
    outputs = {k: np.concatenate(v).astype(np.float32) for k, v in outputs.items()}
    labels = {k: np.concatenate(v).astype(np.float32) for k, v in labels.items()}
    qp = 1 / (1 + np.exp(-np.clip(outputs["quality_logit"], -40, 40)))
    metrics = {k: v / count for k, v in sums.items()}
    metrics.update({
        "rel_auc": auc(labels["y_rel"], outputs["rel_logit"]),
        "rel_ap": ap(labels["y_rel"], outputs["rel_logit"]),
        "iou05_auc": auc(labels["y_05"], outputs["iou05_logit"]) if group != "v2_ab" else None,
        "iou05_ap": ap(labels["y_05"], outputs["iou05_logit"]) if group != "v2_ab" else None,
        "iou07_auc": auc(labels["y_07"], outputs["iou07_logit"]) if group != "v2_ab" else None,
        "iou07_ap": ap(labels["y_07"], outputs["iou07_logit"]) if group != "v2_ab" else None,
        "quality_pearson_best_iou": corr(qp, labels["y_iou"]),
    })
    return (metrics, outputs, labels) if collect else metrics


def main():
    args = args_parser(); protocol = C4Protocol(stage="c4_r2_train_calibrator", split="train"); protocol.validate()
    if args.seed != 13: raise ValueError("V2 seed is frozen at 13")
    set_seed(args.seed); device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    x, names = load_npz_features(args.train_video_npz); standardizer = ArrayStandardizer.fit(x, names)
    train = VideoR2Dataset(args.train_video_npz, standardizer); calib = VideoR2Dataset(args.calib_video_npz, standardizer)
    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=device.type == "cuda", persistent_workers=args.num_workers > 0)
    calib_loader = DataLoader(calib, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=device.type == "cuda", persistent_workers=args.num_workers > 0)
    model = build_v2_model(args.group, len(names)).float().to(device); cfg = model_config(model, args.group)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    write_json(str(out_dir / "model_config.json"), cfg)
    history = []; best = None; best_path = out_dir / "model_best.pt"
    for epoch in range(1, args.epochs + 1):
        model.train(); total = 0.0; count = 0
        for raw in tqdm(train_loader, desc=f"{args.group} epoch {epoch}/{args.epochs}"):
            batch = {k: v.to(device, non_blocking=True) for k, v in raw.items() if k.startswith("y_") or k == "x"}
            output = model(batch["x"]); loss, _ = loss_parts(output, batch, args.group)
            optimizer.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); optimizer.step()
            total += float(loss) * len(batch["x"]); count += len(batch["x"])
        metrics = evaluate(model, calib_loader, device, args.group)
        rec = {"epoch": epoch, "train_loss": total / count, **{f"calib_{k}": v for k, v in metrics.items()}}
        history.append(rec); print(json.dumps(rec, sort_keys=True))
        if best is None or rec["calib_loss"] < best["calib_loss"]:
            best = rec; partial = str(best_path) + ".partial"
            torch.save({"model_state": model.state_dict(), "model_config": cfg, "standardizer": standardizer.to_dict(), "best_epoch": epoch, "best_metrics": rec, "official_val_used": False}, partial)
            os.replace(partial, best_path)
    write_json(str(out_dir / "history.json"), history)
    checkpoint = torch.load(best_path, map_location="cpu", weights_only=False); model.load_state_dict(checkpoint["model_state"])
    metrics, outputs, labels = evaluate(model, calib_loader, device, args.group, collect=True)
    with np.load(args.calib_video_npz, allow_pickle=True) as z:
        np.savez_compressed(out_dir / "train_calib_logits.npz", **outputs,
            group_id=z["group_id"].astype(np.int32), query_index=z["query_index"].astype(np.int32), video_idx=z["video_idx"].astype(np.int32),
            label_relevant=labels["y_rel"], label_any_05=labels["y_05"], label_any_07=labels["y_07"], label_best_iou=labels["y_iou"])
    audit = {
        "status": "PASS", "group": args.group, "model_config": cfg,
        "parameter_count": int(sum(p.numel() for p in model.parameters())),
        "training_split": "train_fit_only", "selection_split": "train_calib_only",
        "train_video_rows": len(train), "calib_video_rows": len(calib),
        "best_epoch": checkpoint["best_epoch"], "best_train_calib_metrics": metrics,
        "loss_curve": {"epoch": [r["epoch"] for r in history], "train": [r["train_loss"] for r in history], "calib": [r["calib_loss"] for r in history]},
        "model_best_sha256": sha256(best_path), "logits_sha256": sha256(out_dir / "train_calib_logits.npz"),
        "capacity_search": False, "pos_weight_used": False,
        "official_val_used": False, "post_val_adjustment": False,
    }
    write_protocol_manifest(args.audit_json, protocol, audit); print(json.dumps(audit, indent=2))


if __name__ == "__main__": main()
