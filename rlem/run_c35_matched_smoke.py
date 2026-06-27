#!/usr/bin/env python3

"""Run matched C3.5 control/treatment smoke training on frozen train splits."""

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

from rlem.evidence_dataset import FeatureStats
from rlem.evidence_model import FlexibleEvidenceMLP
from rlem.feature_schema import C35_MATCHED_FEATURE_SETS
from rlem.io_utils import read_json
from rlem.train_c35_qsp import compute_losses
from rlem.train_evidence_heads import CACHE_LABEL_NAMES


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--hidden_dims", default="256,256,128")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--max_train_rows", type=int, default=524288)
    parser.add_argument("--max_calib_rows", type=int, default=131072)
    parser.add_argument("--shuffle_block_rows", type=int, default=262144)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--state_equivalence_atol", type=float, default=1e-6)
    parser.add_argument("--history_equivalence_atol", type=float, default=1e-8)
    return parser.parse_args()


def atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = str(path) + ".partial"
    with open(partial, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)
    os.replace(partial, path)


def state_sha256(state):
    digest = hashlib.sha256()
    for key in sorted(state):
        digest.update(key.encode())
        digest.update(state[key].detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def make_control_state(hidden_dims, dropout, seed):
    torch.manual_seed(seed)
    model = FlexibleEvidenceMLP(31, hidden_dims, dropout)
    return {key: value.detach().clone() for key, value in model.state_dict().items()}


def matched_model(input_dim, feature_indices, hidden_dims, dropout, control_state):
    model = FlexibleEvidenceMLP(input_dim, hidden_dims, dropout)
    state = model.state_dict()
    for key, value in state.items():
        if key == "net.0.weight":
            value.zero_()
            for variant_col, canonical_col in enumerate(feature_indices):
                if canonical_col < 31:
                    value[:, variant_col].copy_(control_state[key][:, canonical_col])
        else:
            value.copy_(control_state[key])
    model.load_state_dict(state)
    return model


def iter_indices(n_rows, batch_size, train, seed, block_rows):
    if train:
        rng = np.random.default_rng(seed)
        blocks = [(start, min(start + block_rows, n_rows)) for start in range(0, n_rows, block_rows)]
        for block_index in rng.permutation(len(blocks)):
            start, end = blocks[int(block_index)]
            indices = np.arange(start, end, dtype=np.int64)
            rng.shuffle(indices)
            for offset in range(0, len(indices), batch_size):
                yield indices[offset:offset + batch_size]
    else:
        for start in range(0, n_rows, batch_size):
            yield np.arange(start, min(start + batch_size, n_rows), dtype=np.int64)


def batch_from_numpy(x, y, indices, feature_indices, device):
    xb = np.asarray(x[indices], dtype=np.float32)[:, feature_indices]
    yb = np.asarray(y[indices], dtype=np.float32)
    return {
        "x": torch.from_numpy(xb).to(device),
        **{
            name: torch.from_numpy(yb[:, col]).to(device)
            for col, name in enumerate(CACHE_LABEL_NAMES)
        },
    }


def batch_from_resident(resident, indices, feature_indices, device):
    index = torch.from_numpy(indices).to(device)
    features = torch.tensor(feature_indices, dtype=torch.long, device=device)
    yb = resident["y"].index_select(0, index)
    return {
        "x": resident["x"].index_select(0, index).index_select(1, features),
        **{name: yb[:, col] for col, name in enumerate(CACHE_LABEL_NAMES)},
    }


def evaluate(model, x, y, resident, feature_indices, args, device, loss_args, use_resident):
    model.eval()
    totals = {}
    seen = 0
    n_rows = min(len(x), args.max_calib_rows)
    with torch.no_grad():
        for indices in iter_indices(n_rows, args.batch_size, False, args.seed + 999, args.shuffle_block_rows):
            batch = (
                batch_from_resident(resident, indices, feature_indices, device)
                if use_resident else batch_from_numpy(x, y, indices, feature_indices, device)
            )
            losses = compute_losses(model(batch["x"]), batch, loss_args, 1.0)
            size = len(indices)
            seen += size
            for key, value in losses.items():
                totals[key] = totals.get(key, 0.0) + float(value) * size
    return {key: value / seen for key, value in totals.items()}


def run_one(name, x_fit, y_fit, x_calib, y_calib, resident, feature_indices, args, device, control_state, use_resident):
    print(
        f"[smoke] start {name}: mode={'gpu_resident' if use_resident else 'cpu_memmap_to_cuda'} ",
        f"train_rows={min(len(x_fit), args.max_train_rows)} calib_rows={min(len(x_calib), args.max_calib_rows)}",
        flush=True,
    )
    hidden_dims = [int(value) for value in args.hidden_dims.split(",")]
    model = matched_model(len(feature_indices), feature_indices, hidden_dims, args.dropout, control_state).to(device)
    initial_state_hash = state_sha256(model.state_dict())
    loss_args = SimpleNamespace(
        lambda_joint_reg=1.0, lambda_joint_bin=0.25, lambda_bd=1.0,
        lambda_fp=0.5, fp_loss_type="bce", focal_gamma=2.0,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    started = time.time()
    history = []
    n_rows = min(len(x_fit), args.max_train_rows)
    for epoch in range(1, args.epochs + 1):
        model.train()
        totals = {}
        seen = 0
        for indices in iter_indices(
            n_rows, args.batch_size, True, args.seed + epoch - 1, args.shuffle_block_rows,
        ):
            batch = (
                batch_from_resident(resident["train_fit"], indices, feature_indices, device)
                if use_resident else batch_from_numpy(x_fit, y_fit, indices, feature_indices, device)
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
        train_metrics = {f"train_{key}": value / seen for key, value in totals.items()}
        val = evaluate(
            model, x_calib, y_calib,
            None if resident is None else resident["train_calib"], feature_indices,
            args, device, loss_args, use_resident,
        )
        record = {"epoch": epoch, **train_metrics, **{f"val_{key}": value for key, value in val.items()}}
        if not all(math.isfinite(value) for value in record.values()):
            raise ValueError(f"non-finite smoke record for {name}: {record}")
        history.append(record)
    state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    result = {
        "name": name,
        "input_dim": len(feature_indices),
        "mode": "gpu_resident" if use_resident else "cpu_memmap_to_cuda",
        "initial_state_sha256": initial_state_hash,
        "final_state_sha256": state_sha256(state),
        "runtime_seconds": time.time() - started,
        "history": history,
        "state": state,
    }
    print(f"[smoke] done {name}: {result['runtime_seconds']:.3f}s", flush=True)
    return result


def main():
    args = parse_args()
    if not torch.cuda.is_available() or not args.device.startswith("cuda"):
        raise RuntimeError("matched smoke requires CUDA")
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = read_json(Path(args.cache_dir) / "cache_manifest.json")
    stats = FeatureStats.from_dict(read_json(Path(args.cache_dir) / "feature_stats.json"))
    canonical_features = C35_MATCHED_FEATURE_SETS["qsp53_with_coverage"]
    if manifest.get("status") != "PASS" or manifest["feature_names"] != canonical_features:
        raise ValueError("canonical 53-feature cache manifest mismatch")
    if stats.feature_names != canonical_features or stats.count != 15609000:
        raise ValueError("canonical feature stats mismatch")
    cache = {}
    for split in ("train_fit", "train_calib"):
        entry = manifest["splits"][split]
        cache[split] = {
            "x": np.load(entry["x_path"], mmap_mode="r"),
            "y": np.load(entry["y_path"], mmap_mode="r"),
        }
    feature_indices = {
        name: [canonical_features.index(feature) for feature in features]
        for name, features in C35_MATCHED_FEATURE_SETS.items()
    }
    hidden_dims = [int(value) for value in args.hidden_dims.split(",")]
    control_state = make_control_state(hidden_dims, args.dropout, args.seed)

    # Epoch-0 outputs must match exactly because all treatment-only columns start at zero.
    probe_indices = np.arange(min(4096, args.max_calib_rows), dtype=np.int64)
    probe_outputs = {}
    for name, columns in feature_indices.items():
        model = matched_model(len(columns), columns, hidden_dims, args.dropout, control_state).to(device).eval()
        batch = batch_from_numpy(cache["train_calib"]["x"], cache["train_calib"]["y"], probe_indices, columns, device)
        with torch.no_grad():
            probe_outputs[name] = model(batch["x"])["q_joint_logit"].cpu()
    epoch0_diff = {
        name: float((value - probe_outputs["control31"]).abs().max())
        for name, value in probe_outputs.items()
    }
    if any(value != 0.0 for value in epoch0_diff.values()):
        raise ValueError(f"epoch-0 matched logits differ: {epoch0_diff}")

    # First train control through the existing memmap->CUDA path.
    control_cpu = run_one(
        "control31_cpu", cache["train_fit"]["x"], cache["train_fit"]["y"],
        cache["train_calib"]["x"], cache["train_calib"]["y"], None,
        feature_indices["control31"], args, device, control_state, False,
    )

    # Materialize the canonical cache once on the same GPU as the model.
    resident = {}
    resident_bytes = 0
    for split in ("train_fit", "train_calib"):
        resident[split] = {
            "x": torch.from_numpy(np.array(cache[split]["x"], dtype=np.float32, copy=True)).to(device),
            "y": torch.from_numpy(np.array(cache[split]["y"], dtype=np.float32, copy=True)).to(device),
        }
        resident_bytes += sum(tensor.numel() * tensor.element_size() for tensor in resident[split].values())

    # Confirm the two data paths are numerically identical before training.
    data_path_max_abs_diff = 0.0
    for indices in list(iter_indices(args.max_train_rows, args.batch_size, True, args.seed, args.shuffle_block_rows))[:8]:
        cpu_batch = batch_from_numpy(cache["train_fit"]["x"], cache["train_fit"]["y"], indices, list(range(53)), device)
        gpu_batch = batch_from_resident(resident["train_fit"], indices, list(range(53)), device)
        for key in cpu_batch:
            data_path_max_abs_diff = max(
                data_path_max_abs_diff,
                float((cpu_batch[key] - gpu_batch[key]).abs().max()),
            )
    if data_path_max_abs_diff != 0.0:
        raise ValueError(f"CPU/GPU cache paths differ: {data_path_max_abs_diff}")

    runs = [control_cpu]
    for name in ("control31", "qsp52_drop_coverage", "qsp53_with_coverage"):
        runs.append(run_one(
            name, cache["train_fit"]["x"], cache["train_fit"]["y"],
            cache["train_calib"]["x"], cache["train_calib"]["y"], resident,
            feature_indices[name], args, device, control_state, True,
        ))

    control_gpu = next(run for run in runs if run["name"] == "control31")
    final_state_max_abs_diff = max(
        float((control_cpu["state"][key] - control_gpu["state"][key]).abs().max())
        for key in control_cpu["state"]
    )
    history_max_abs_diff = max(
        abs(control_cpu["history"][0][key] - control_gpu["history"][0][key])
        for key in control_cpu["history"][0]
        if key != "epoch"
    )
    if (
        final_state_max_abs_diff > args.state_equivalence_atol
        or history_max_abs_diff > args.history_equivalence_atol
    ):
        raise ValueError(
            f"memmap/GPU-resident training mismatch: state={final_state_max_abs_diff} history={history_max_abs_diff}"
        )

    for run in runs:
        run_dir = output_dir / run["name"]
        run_dir.mkdir(parents=True, exist_ok=True)
        state = run.pop("state")
        checkpoint = {
            "model_state": state,
            "model_cfg": {"input_dim": run["input_dim"], "hidden_dims": hidden_dims, "dropout": args.dropout},
            "smoke": True,
            "official_val_used": False,
        }
        partial = run_dir / "model_smoke.pt.partial"
        torch.save(checkpoint, partial)
        os.replace(partial, run_dir / "model_smoke.pt")
        atomic_json(run_dir / "summary.json", run)

    summary = {
        "status": "PASS",
        "stage": "C3.5 matched control/treatment smoke only",
        "cache_dir": str(Path(args.cache_dir).resolve()),
        "feature_indices": feature_indices,
        "matched_epoch0_q_joint_logit_max_abs_diff": epoch0_diff,
        "data_path_max_abs_diff": data_path_max_abs_diff,
        "memmap_vs_gpu_resident_final_state_max_abs_diff": final_state_max_abs_diff,
        "memmap_vs_gpu_resident_history_max_abs_diff": history_max_abs_diff,
        "memmap_vs_gpu_resident_state_atol": args.state_equivalence_atol,
        "memmap_vs_gpu_resident_history_atol": args.history_equivalence_atol,
        "resident_cache_bytes": resident_bytes,
        "resident_cache_gib": resident_bytes / (1024 ** 3),
        "smoke_budget": {
            "epochs": args.epochs, "batch_size": args.batch_size,
            "max_train_rows": args.max_train_rows, "max_calib_rows": args.max_calib_rows,
            "seed": args.seed,
        },
        "runs": runs,
        "full_training_started": False,
        "official_val_used": False,
    }
    atomic_json(output_dir / "C35_MATCHED_SMOKE_SUMMARY.json", summary)
    with open(output_dir / "C35_MATCHED_SMOKE_RESULTS.csv", "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["name", "input_dim", "mode", "runtime_seconds", "train_loss", "val_loss"])
        writer.writeheader()
        for run in runs:
            writer.writerow({
                "name": run["name"], "input_dim": run["input_dim"], "mode": run["mode"],
                "runtime_seconds": run["runtime_seconds"],
                "train_loss": run["history"][-1]["train_loss"],
                "val_loss": run["history"][-1]["val_loss"],
            })
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
