#!/usr/bin/env python3

"""Full train-only matched C3.5 control/treatment training.

This runner consumes only the frozen train_fit/train_calib cache.  It trains the
31-, 52-, and 53-feature variants from one matched initialization and never
loads official-val evidence or performs score calibration.
"""

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import torch
from tqdm import tqdm

from rlem.evidence_dataset import FeatureStats
from rlem.feature_schema import C35_MATCHED_FEATURE_SETS
from rlem.io_utils import read_json
from rlem.run_c35_matched_smoke import (
    atomic_json,
    iter_indices,
    make_control_state,
    matched_model,
    state_sha256,
)
from rlem.train_c35_qsp import compute_losses
from rlem.train_evidence_heads import CACHE_LABEL_NAMES


VARIANTS = ("control31", "qsp52_drop_coverage", "qsp53_with_coverage")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--hidden_dims", default="256,256,128")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--shuffle_block_rows", type=int, default=262144)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_torch_save(payload, path):
    path = Path(path)
    partial = Path(str(path) + ".partial")
    torch.save(payload, partial)
    os.replace(partial, path)


def subset_stats(stats, columns):
    return FeatureStats(
        feature_names=[stats.feature_names[index] for index in columns],
        mean=[stats.mean[index] for index in columns],
        std=[stats.std[index] for index in columns],
        count=stats.count,
        label_counts=dict(stats.label_counts),
    )


def make_batch(resident, indices, feature_index, all_features):
    row_index = torch.from_numpy(indices).to(resident["x"].device)
    xb = resident["x"].index_select(0, row_index)
    if not all_features:
        xb = xb.index_select(1, feature_index)
    yb = resident["y"].index_select(0, row_index)
    return {
        "x": xb,
        **{name: yb[:, column] for column, name in enumerate(CACHE_LABEL_NAMES)},
    }


@torch.no_grad()
def evaluate(model, resident, feature_index, all_features, args, loss_args):
    model.eval()
    totals = {}
    seen = 0
    row_count = resident["x"].shape[0]
    total_batches = math.ceil(row_count / args.batch_size)
    batches = iter_indices(
        row_count, args.batch_size, False, args.seed + 999, args.shuffle_block_rows,
    )
    for indices in tqdm(batches, total=total_batches, desc="  train_calib", leave=False):
        batch = make_batch(resident, indices, feature_index, all_features)
        losses = compute_losses(model(batch["x"]), batch, loss_args, 1.0)
        size = len(indices)
        seen += size
        for key, value in losses.items():
            totals[key] = totals.get(key, 0.0) + float(value) * size
    return {key: value / seen for key, value in totals.items()}


def train_variant(
    name, resident, columns, canonical_stats, control_state, args, device, output_root,
):
    print(f"\n[full] starting {name} ({len(columns)} features)", flush=True)
    started = time.time()
    hidden_dims = [int(value) for value in args.hidden_dims.split(",") if value]
    model = matched_model(
        len(columns), columns, hidden_dims, args.dropout, control_state,
    ).to(device)
    initial_hash = state_sha256(model.state_dict())
    feature_index = torch.tensor(columns, dtype=torch.long, device=device)
    all_features = columns == list(range(len(canonical_stats.feature_names)))
    variant_stats = subset_stats(canonical_stats, columns)
    loss_args = SimpleNamespace(
        lambda_joint_reg=1.0,
        lambda_joint_bin=0.25,
        lambda_bd=1.0,
        lambda_fp=0.5,
        fp_loss_type="bce",
        focal_gamma=2.0,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
    )
    # Reset the stochastic stream after constructing each differently shaped model.
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    run_dir = output_root / name
    run_dir.mkdir(parents=True, exist_ok=False)
    training_args = {
        "variant": name,
        "feature_names": variant_stats.feature_names,
        "hidden_dims": hidden_dims,
        "dropout": args.dropout,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "seed": args.seed,
        "shuffle_block_rows": args.shuffle_block_rows,
        "loss": {
            "lambda_joint_reg": 1.0,
            "lambda_joint_bin": 0.25,
            "lambda_bd": 1.0,
            "lambda_fp": 0.5,
            "fp_loss_type": "bce",
        },
        "matched_initialization": True,
        "gpu_resident": True,
        "official_val_used": False,
    }
    atomic_json(run_dir / "train_args.json", training_args)
    atomic_json(run_dir / "feature_stats.json", variant_stats.to_dict())

    history = []
    best_val = float("inf")
    best_epoch = None
    row_count = resident["train_fit"]["x"].shape[0]
    total_batches = math.ceil(row_count / args.batch_size)
    for epoch in range(1, args.epochs + 1):
        model.train()
        totals = {}
        seen = 0
        batches = iter_indices(
            row_count, args.batch_size, True,
            args.seed + epoch - 1, args.shuffle_block_rows,
        )
        progress = tqdm(
            batches, total=total_batches, desc=f"{name} epoch {epoch}/{args.epochs}",
        )
        for indices in progress:
            batch = make_batch(
                resident["train_fit"], indices, feature_index, all_features,
            )
            optimizer.zero_grad(set_to_none=True)
            losses = compute_losses(model(batch["x"]), batch, loss_args, 1.0)
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            size = len(indices)
            seen += size
            for key, value in losses.items():
                totals[key] = totals.get(key, 0.0) + float(value.detach()) * size
            if seen % (args.batch_size * 500) < args.batch_size:
                progress.set_postfix(loss=f"{totals['loss'] / seen:.6f}")

        train_metrics = {
            f"train_{key}": value / seen for key, value in totals.items()
        }
        val_metrics = {
            f"val_{key}": value
            for key, value in evaluate(
                model, resident["train_calib"], feature_index, all_features,
                args, loss_args,
            ).items()
        }
        record = {"epoch": epoch, **train_metrics, **val_metrics}
        if not all(math.isfinite(value) for value in record.values()):
            raise ValueError(f"non-finite training record for {name}: {record}")
        history.append(record)
        atomic_json(run_dir / "history.json", history)
        print(json.dumps(record, indent=2), flush=True)

        if record["val_loss"] <= best_val:
            best_val = record["val_loss"]
            best_epoch = epoch
            state = {
                key: value.detach().cpu()
                for key, value in model.state_dict().items()
            }
            checkpoint = {
                "model_state": state,
                "model_type": "FlexibleEvidenceMLP",
                "model_cfg": {
                    "input_dim": len(columns),
                    "hidden_dims": hidden_dims,
                    "dropout": args.dropout,
                },
                "feature_stats": variant_stats.to_dict(),
                "train_args": training_args,
                "epoch": epoch,
                "metrics": record,
                "official_val_used": False,
            }
            atomic_torch_save(checkpoint, run_dir / "model_best.pt")

    summary = {
        "status": "PASS",
        "variant": name,
        "input_dim": len(columns),
        "initial_state_sha256": initial_hash,
        "best_epoch": best_epoch,
        "best_val_loss": best_val,
        "runtime_seconds": time.time() - started,
        "train_fit_rows": int(resident["train_fit"]["x"].shape[0]),
        "train_calib_rows": int(resident["train_calib"]["x"].shape[0]),
        "full_training": True,
        "official_val_used": False,
    }
    atomic_json(run_dir / "train_summary.json", summary)
    print(f"[full] completed {name}: best epoch={best_epoch}, val_loss={best_val:.10f}", flush=True)
    del optimizer, model, feature_index
    torch.cuda.empty_cache()
    return summary, history


def main():
    args = parse_args()
    if not torch.cuda.is_available() or not args.device.startswith("cuda"):
        raise RuntimeError("full matched training requires CUDA")
    if args.epochs not in {3, 5, 8} or args.batch_size != 1024 or args.seed != 13:
        raise ValueError("authorized protocols require epochs in {3,5,8}, batch_size=1024, seed=13")
    if args.weight_decay != 0.0 or args.lr != 1e-3:
        raise ValueError("frozen protocol requires lr=1e-3 and weight_decay=0")

    cache_dir = Path(args.cache_dir).resolve()
    output_root = Path(args.output_dir).resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    manifest_path = cache_dir / "cache_manifest.json"
    manifest = read_json(manifest_path)
    canonical_features = C35_MATCHED_FEATURE_SETS["qsp53_with_coverage"]
    if manifest.get("status") != "PASS":
        raise ValueError("canonical cache manifest is not PASS")
    if manifest.get("official_val_used") is not False:
        raise ValueError("cache is not explicitly train-only")
    if manifest.get("feature_names") != canonical_features:
        raise ValueError("canonical 53-feature order mismatch")
    source_name = Path(manifest.get("source_evidence", "")).name.lower()
    if "train" not in source_name or source_name.startswith("val"):
        raise ValueError("full matched training accepts train evidence only")

    stats = FeatureStats.from_dict(read_json(cache_dir / "feature_stats.json"))
    if stats.feature_names != canonical_features or stats.count != 15609000:
        raise ValueError("canonical train_fit feature statistics mismatch")
    columns_by_variant = {
        name: [canonical_features.index(feature) for feature in C35_MATCHED_FEATURE_SETS[name]]
        for name in VARIANTS
    }
    hidden_dims = [int(value) for value in args.hidden_dims.split(",") if value]
    if hidden_dims != [256, 256, 128] or args.dropout != 0.1:
        raise ValueError("frozen architecture requires [256,256,128] and dropout=0.1")

    # Accuracy is preferred over TF32 acceleration for this matched comparison.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(args.device)
    print(f"[cache] loading canonical cache onto {device}", flush=True)
    resident = {}
    resident_bytes = 0
    for split in ("train_fit", "train_calib"):
        entry = manifest["splits"][split]
        x = np.load(entry["x_path"], mmap_mode="r")
        y = np.load(entry["y_path"], mmap_mode="r")
        resident[split] = {
            "x": torch.from_numpy(np.array(x, dtype=np.float32, copy=True)).to(device),
            "y": torch.from_numpy(np.array(y, dtype=np.float32, copy=True)).to(device),
        }
        resident_bytes += sum(
            tensor.numel() * tensor.element_size()
            for tensor in resident[split].values()
        )
        print(f"[cache] {split} resident", flush=True)

    control_state = make_control_state(hidden_dims, args.dropout, args.seed)
    init_path = output_root / "MATCHED_CONTROL31_INIT.pt"
    atomic_torch_save(
        {
            "model_state": control_state,
            "model_cfg": {"input_dim": 31, "hidden_dims": hidden_dims, "dropout": args.dropout},
            "seed": args.seed,
            "official_val_used": False,
        },
        init_path,
    )

    # All treatment-only first-layer columns are zero, so all three output heads
    # must match control exactly at epoch zero.
    probe_rows = min(4096, resident["train_calib"]["x"].shape[0])
    probe_index = np.arange(probe_rows, dtype=np.int64)
    epoch0 = {}
    control_outputs = None
    for name in VARIANTS:
        columns = columns_by_variant[name]
        model = matched_model(
            len(columns), columns, hidden_dims, args.dropout, control_state,
        ).to(device).eval()
        feature_index = torch.tensor(columns, dtype=torch.long, device=device)
        batch = make_batch(
            resident["train_calib"], probe_index, feature_index,
            columns == list(range(len(canonical_features))),
        )
        with torch.no_grad():
            outputs = {key: value.cpu() for key, value in model(batch["x"]).items()}
        if control_outputs is None:
            control_outputs = outputs
        epoch0[name] = {
            key: float((value - control_outputs[key]).abs().max())
            for key, value in outputs.items()
        }
        del model, feature_index, batch
    if any(value != 0.0 for result in epoch0.values() for value in result.values()):
        raise ValueError(f"matched epoch-0 outputs differ: {epoch0}")

    started = time.time()
    summaries = []
    histories = {}
    for name in VARIANTS:
        summary, history = train_variant(
            name, resident, columns_by_variant[name], stats, control_state,
            args, device, output_root,
        )
        summaries.append(summary)
        histories[name] = history

    overall = {
        "status": "PASS",
        "stage": "C3.5 matched full train-only control/treatment training",
        "cache_dir": str(cache_dir),
        "cache_manifest_sha256": sha256_file(manifest_path),
        "matched_init_path": str(init_path),
        "matched_init_sha256": sha256_file(init_path),
        "matched_epoch0_max_abs_diff": epoch0,
        "resident_cache_bytes": resident_bytes,
        "resident_cache_gib": resident_bytes / (1024 ** 3),
        "protocol": {
            "hidden_dims": hidden_dims,
            "dropout": args.dropout,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "seed": args.seed,
        },
        "runs": summaries,
        "runtime_seconds": time.time() - started,
        "score_grid_started": False,
        "official_val_used": False,
    }
    atomic_json(output_root / "C35_MATCHED_FULL_TRAINING_SUMMARY.json", overall)
    with open(output_root / "C35_MATCHED_FULL_TRAINING_RESULTS.csv", "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["variant", "input_dim", "best_epoch", "best_val_loss", "runtime_seconds"],
        )
        writer.writeheader()
        for summary in summaries:
            writer.writerow({key: summary[key] for key in writer.fieldnames})
    print(json.dumps(overall, indent=2), flush=True)


if __name__ == "__main__":
    main()
