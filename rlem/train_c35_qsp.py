#!/usr/bin/env python

"""Train one C3.5 flexible evidence-head configuration on frozen augmented caches."""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from rlem.evidence_dataset import FeatureStats
from rlem.evidence_model import FlexibleEvidenceMLP
from rlem.io_utils import read_json, write_json
from rlem.train_evidence_heads import iter_cached_batches, load_cache


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_id", required=True)
    parser.add_argument("--cache_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--hidden_dims", required=True, help="Comma-separated, e.g. 256,128")
    parser.add_argument("--dropout", type=float, required=True)
    parser.add_argument("--weight_decay", type=float, required=True)
    parser.add_argument("--lambda_joint_reg", type=float, default=1.0)
    parser.add_argument("--lambda_joint_bin", type=float, default=0.25)
    parser.add_argument("--lambda_bd", type=float, default=1.0)
    parser.add_argument("--lambda_fp", type=float, default=0.5)
    parser.add_argument("--fp_loss_type", choices=["bce", "bce_pos_weight", "focal"], default="bce")
    parser.add_argument("--focal_gamma", type=float, default=2.0)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--shuffle_block_rows", type=int, default=262144)
    parser.add_argument("--max_train_rows", type=int, default=None)
    parser.add_argument("--max_val_rows", type=int, default=None)
    return parser.parse_args()


def compute_losses(out, batch, args, fp_pos_weight):
    q_joint = torch.sigmoid(out["q_joint_logit"])
    q_bd = torch.sigmoid(out["q_bd_logit"])
    joint_reg = F.smooth_l1_loss(q_joint, batch["y_joint"])
    joint_bin = F.binary_cross_entropy_with_logits(out["q_joint_logit"], batch["y_joint_05"])
    mask = batch["m_bd"] > 0.5
    boundary = (
        F.smooth_l1_loss(q_bd[mask], batch["y_bd"][mask])
        if torch.any(mask) else out["q_bd_logit"].sum() * 0.0
    )
    if args.fp_loss_type == "bce":
        fp = F.binary_cross_entropy_with_logits(out["e_fp_logit"], batch["y_fp"])
    elif args.fp_loss_type == "bce_pos_weight":
        pos_weight = torch.tensor(fp_pos_weight, device=out["e_fp_logit"].device)
        fp = F.binary_cross_entropy_with_logits(
            out["e_fp_logit"], batch["y_fp"], pos_weight=pos_weight,
        )
    else:
        base = F.binary_cross_entropy_with_logits(
            out["e_fp_logit"], batch["y_fp"], reduction="none",
        )
        probability = torch.sigmoid(out["e_fp_logit"])
        p_t = probability * batch["y_fp"] + (1.0 - probability) * (1.0 - batch["y_fp"])
        fp = (((1.0 - p_t) ** args.focal_gamma) * base).mean()
    total = (
        args.lambda_joint_reg * joint_reg + args.lambda_joint_bin * joint_bin
        + args.lambda_bd * boundary + args.lambda_fp * fp
    )
    return {
        "loss": total,
        "loss_joint_reg": joint_reg.detach(),
        "loss_joint_bin": joint_bin.detach(),
        "loss_bd": boundary.detach(),
        "loss_fp": fp.detach(),
    }


@torch.no_grad()
def evaluate(model, x, y, device, args, fp_pos_weight):
    model.eval()
    totals = {}
    seen = 0
    for batch in iter_cached_batches(
        x, y, args.batch_size, train=False, seed=args.seed + 999,
        max_rows=args.max_val_rows, block_rows=args.shuffle_block_rows,
    ):
        batch = {key: value.to(device) for key, value in batch.items()}
        losses = compute_losses(model(batch["x"]), batch, args, fp_pos_weight)
        size = batch["x"].shape[0]
        seen += size
        for key, value in losses.items():
            totals[key] = totals.get(key, 0.0) + float(value) * size
    return {key: value / seen for key, value in totals.items()}


def atomic_torch_save(obj, path):
    partial = str(path) + ".partial"
    torch.save(obj, partial)
    os.replace(partial, path)


def main():
    args = parse_args()
    started = time.time()
    hidden_dims = [int(value) for value in args.hidden_dims.split(",") if value.strip()]
    if not hidden_dims:
        raise ValueError("Empty --hidden_dims")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stats = FeatureStats.from_dict(read_json(Path(args.cache_dir) / "feature_stats.json"))
    forbidden = {"r2_raw", "r2_prob", "r2_tilde", "rank_r2", "r_abs_gap"}
    if forbidden & set(stats.feature_names):
        raise ValueError("C3.5 keeps unavailable r2 features disabled")
    if len(stats.feature_names) <= 31:
        raise ValueError("C3.5 requires an augmented feature schema beyond the frozen 31 scalar baseline")
    cache = load_cache(args.cache_dir, stats.feature_names)
    rows = int(stats.label_counts["rows"])
    positives = int(stats.label_counts["y_fp_positive"])
    fp_pos_weight = (rows - positives) / positives
    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    model = FlexibleEvidenceMLP(len(stats.feature_names), hidden_dims, args.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    train_args = vars(args) | {
        "hidden_dims_parsed": hidden_dims,
        "parameter_count": parameter_count,
        "fp_pos_weight_auto": fp_pos_weight,
        "feature_schema": "c35_qsp",
        "official_val_used": False,
    }
    write_json(output_dir / "train_args.json", train_args)
    write_json(output_dir / "feature_stats.json", stats.to_dict())

    history = []
    best_val = float("inf")
    train_rows = len(cache["train_fit"]["x"])
    if args.max_train_rows is not None:
        train_rows = min(train_rows, args.max_train_rows)
    train_total = math.ceil(train_rows / args.batch_size)
    for epoch in range(1, args.epochs + 1):
        model.train()
        totals = {}
        seen = 0
        batches = iter_cached_batches(
            cache["train_fit"]["x"], cache["train_fit"]["y"],
            args.batch_size, train=True, seed=args.seed + epoch - 1,
            max_rows=args.max_train_rows, block_rows=args.shuffle_block_rows,
        )
        for batch in tqdm(batches, total=train_total, desc=f"{args.config_id} epoch {epoch}"):
            batch = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            losses = compute_losses(model(batch["x"]), batch, args, fp_pos_weight)
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            size = batch["x"].shape[0]
            seen += size
            for key, value in losses.items():
                totals[key] = totals.get(key, 0.0) + float(value.detach()) * size
        train_metrics = {f"train_{key}": value / seen for key, value in totals.items()}
        val_metrics = {
            f"val_{key}": value for key, value in evaluate(
                model, cache["train_calib"]["x"], cache["train_calib"]["y"],
                device, args, fp_pos_weight,
            ).items()
        }
        record = {"epoch": epoch, **train_metrics, **val_metrics}
        if not all(math.isfinite(value) for value in record.values()):
            raise ValueError(f"Non-finite training record: {record}")
        history.append(record)
        print(json.dumps(record, indent=2))
        checkpoint = {
            "model_state": model.state_dict(),
            "model_type": "FlexibleEvidenceMLP",
            "model_cfg": {
                "input_dim": len(stats.feature_names),
                "hidden_dims": hidden_dims,
                "dropout": args.dropout,
            },
            "feature_stats": stats.to_dict(),
            "train_args": train_args,
            "epoch": epoch,
            "metrics": record,
        }
        if record["val_loss"] <= best_val:
            best_val = record["val_loss"]
            atomic_torch_save(checkpoint, output_dir / "model_best.pt")
    write_json(output_dir / "history.json", history)
    summary = {
        "status": "PASS",
        "config_id": args.config_id,
        "best_epoch": min(history, key=lambda item: item["val_loss"])["epoch"],
        "best_val_loss": min(item["val_loss"] for item in history),
        "parameter_count": parameter_count,
        "runtime_seconds": time.time() - started,
        "official_val_used": False,
    }
    write_json(output_dir / "train_summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
