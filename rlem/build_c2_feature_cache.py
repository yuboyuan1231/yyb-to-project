#!/usr/bin/env python

"""Build vectorized, memory-mapped C2 tensors from accepted C1 JSONL."""

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
from tqdm import tqdm

from rlem.evidence_dataset import EPS, FeatureStats, _as_float
from rlem.io_utils import iter_jsonl, read_json, write_json


LABEL_NAMES = ["y_joint", "y_joint_05", "y_joint_07", "y_bd", "m_bd", "y_fp"]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence_jsonl", required=True)
    parser.add_argument("--fit_desc_ids", required=True)
    parser.add_argument("--calib_desc_ids", required=True)
    parser.add_argument("--split_summary", required=True)
    parser.add_argument("--stats_json", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--chunk_rows", type=int, default=32768)
    return parser.parse_args()


def load_ids(path):
    with open(path, "r", encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip()}


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def vectorize_rows(rows, feature_names, mean, std):
    n = len(rows)
    x = np.empty((n, len(feature_names)), dtype=np.float32)
    for col, name in enumerate(feature_names):
        x[:, col] = np.fromiter(
            (_as_float(row.get(name)) for row in rows), dtype=np.float32, count=n
        )
    x = (x - mean) / (std + EPS)
    y = np.empty((n, len(LABEL_NAMES)), dtype=np.float32)
    for col, name in enumerate(LABEL_NAMES):
        y[:, col] = np.fromiter(
            (_as_float(row.get(name)) for row in rows), dtype=np.float32, count=n
        )
    return x, y


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fit_ids = load_ids(args.fit_desc_ids)
    calib_ids = load_ids(args.calib_desc_ids)
    if fit_ids & calib_ids:
        raise ValueError("fit/calib desc_id overlap")
    summary = read_json(args.split_summary)
    expected = {
        "train_fit": int(summary["splits"]["train_fit"]["rows"]),
        "train_calib": int(summary["splits"]["train_calib"]["rows"]),
    }
    stats = FeatureStats.from_dict(read_json(args.stats_json))
    if stats.count != expected["train_fit"]:
        raise ValueError(f"stats count {stats.count} != train_fit rows {expected['train_fit']}")
    feature_names = stats.feature_names
    mean = np.asarray(stats.mean, dtype=np.float32)
    std = np.asarray(stats.std, dtype=np.float32)
    paths = {}
    arrays = {}
    for split in ["train_fit", "train_calib"]:
        x_partial = output_dir / f"{split}_x.partial.npy"
        y_partial = output_dir / f"{split}_y.partial.npy"
        paths[split] = {
            "x_partial": x_partial,
            "y_partial": y_partial,
            "x": output_dir / f"{split}_x.npy",
            "y": output_dir / f"{split}_y.npy",
        }
        arrays[split] = {
            "x": np.lib.format.open_memmap(
                x_partial, mode="w+", dtype=np.float32,
                shape=(expected[split], len(feature_names)),
            ),
            "y": np.lib.format.open_memmap(
                y_partial, mode="w+", dtype=np.float32,
                shape=(expected[split], len(LABEL_NAMES)),
            ),
            "offset": 0,
            "rows": [],
        }

    def flush(split):
        state = arrays[split]
        rows = state["rows"]
        if not rows:
            return
        x, y = vectorize_rows(rows, feature_names, mean, std)
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError(f"non-finite cache values in {split}")
        start = state["offset"]
        end = start + len(rows)
        if end > expected[split]:
            raise ValueError(f"cache overflow for {split}: {end} > {expected[split]}")
        state["x"][start:end] = x
        state["y"][start:end] = y
        state["offset"] = end
        state["rows"] = []

    total_rows = sum(expected.values())
    for row in tqdm(iter_jsonl(args.evidence_jsonl), total=total_rows, desc="build C2 cache"):
        desc_id = str(row.get("desc_id"))
        if desc_id in fit_ids:
            split = "train_fit"
        elif desc_id in calib_ids:
            split = "train_calib"
        else:
            raise ValueError(f"desc_id absent from split files: {desc_id}")
        arrays[split]["rows"].append(row)
        if len(arrays[split]["rows"]) >= args.chunk_rows:
            flush(split)
    for split in arrays:
        flush(split)
        if arrays[split]["offset"] != expected[split]:
            raise ValueError(
                f"cache row mismatch for {split}: {arrays[split]['offset']} != {expected[split]}"
            )
        arrays[split]["x"].flush()
        arrays[split]["y"].flush()
        del arrays[split]["x"]
        del arrays[split]["y"]
        os.replace(paths[split]["x_partial"], paths[split]["x"])
        os.replace(paths[split]["y_partial"], paths[split]["y"])

    shutil.copy2(args.stats_json, output_dir / "feature_stats.json")
    manifest = {
        "status": "PASS",
        "source_evidence": os.path.abspath(args.evidence_jsonl),
        "stats_source": os.path.abspath(args.stats_json),
        "feature_names": feature_names,
        "label_names": LABEL_NAMES,
        "dtype": "float32",
        "normalization": "identical to FeatureStats.transform_row",
        "splits": {},
    }
    for split in ["train_fit", "train_calib"]:
        x_path, y_path = paths[split]["x"], paths[split]["y"]
        x = np.load(x_path, mmap_mode="r")
        y = np.load(y_path, mmap_mode="r")
        manifest["splits"][split] = {
            "rows": int(x.shape[0]),
            "x_shape": list(x.shape),
            "y_shape": list(y.shape),
            "x_path": str(x_path),
            "y_path": str(y_path),
            "x_sha256": sha256_file(x_path),
            "y_sha256": sha256_file(y_path),
            "x_finite": bool(np.isfinite(x).all()),
            "y_finite": bool(np.isfinite(y).all()),
        }
    write_json(str(output_dir / "cache_manifest.json"), manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
