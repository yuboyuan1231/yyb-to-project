#!/usr/bin/env python3

"""Build the canonical 53-column C3.5 cache with ordered CPU multiprocessing."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import gzip
import hashlib
import json
import multiprocessing as mp
import os
import shutil
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
from tqdm import tqdm

from rlem.build_c2_feature_cache import LABEL_NAMES, load_ids, vectorize_rows
from rlem.evidence_dataset import FeatureStats
from rlem.feature_schema import C35_MATCHED_FEATURE_SETS
from rlem.io_utils import read_json


_FIT_IDS = None
_CALIB_IDS = None
_FEATURE_NAMES = None
_MEAN = None
_STD = None


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def process_line_chunk(lines):
    fit_rows = []
    calib_rows = []
    for line in lines:
        row = json.loads(line)
        desc_id = str(row.get("desc_id"))
        if desc_id in _FIT_IDS:
            fit_rows.append(row)
        elif desc_id in _CALIB_IDS:
            calib_rows.append(row)
        else:
            raise ValueError(f"desc_id absent from frozen split: {desc_id}")
    result = {"input_rows": len(lines)}
    for split, rows in (("train_fit", fit_rows), ("train_calib", calib_rows)):
        if rows:
            x, y = vectorize_rows(rows, _FEATURE_NAMES, _MEAN, _STD)
        else:
            x = np.empty((0, len(_FEATURE_NAMES)), dtype=np.float32)
            y = np.empty((0, len(LABEL_NAMES)), dtype=np.float32)
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError(f"non-finite values in {split} worker chunk")
        result[split] = (x, y)
    return result


def iter_line_chunks(path, chunk_rows):
    with gzip.open(path, "rt", encoding="utf-8") as file:
        chunk = []
        for line in file:
            if not line.strip():
                continue
            chunk.append(line)
            if len(chunk) >= chunk_rows:
                yield chunk
                chunk = []
        if chunk:
            yield chunk


def finalize_cache(output_dir, paths, expected, args):
    stats_target = output_dir / "feature_stats.json"
    if Path(args.stats_json).resolve() != stats_target.resolve():
        shutil.copy2(args.stats_json, stats_target)

    manifest = {
        "status": "PASS",
        "stage": "C3.5 canonical 53-feature cache",
        "source_evidence": str(Path(args.evidence_jsonl).resolve()),
        "feature_names": _FEATURE_NAMES,
        "label_names": LABEL_NAMES,
        "dtype": "float32",
        "builder_workers": args.num_workers,
        "ordered_chunk_pipeline": True,
        "official_val_used": False,
        "splits": {},
    }
    def inspect_array(path):
        array = np.load(path, mmap_mode="r")
        return {
            "shape": tuple(array.shape),
            "sha256": sha256_file(path),
            "finite": bool(np.isfinite(array).all()),
        }

    jobs = {}
    with ThreadPoolExecutor(max_workers=min(4, args.num_workers)) as executor:
        for split in ("train_fit", "train_calib"):
            for kind in ("x", "y"):
                path = paths[split][kind]
                if not path.is_file():
                    raise FileNotFoundError(f"missing completed cache array: {path}")
                jobs[(split, kind)] = executor.submit(inspect_array, path)
        inspected = {key: job.result() for key, job in jobs.items()}

    for split in ("train_fit", "train_calib"):
        x_path, y_path = paths[split]["x"], paths[split]["y"]
        x_info, y_info = inspected[(split, "x")], inspected[(split, "y")]
        expected_x_shape = (expected[split], len(_FEATURE_NAMES))
        expected_y_shape = (expected[split], len(LABEL_NAMES))
        if x_info["shape"] != expected_x_shape or y_info["shape"] != expected_y_shape:
            raise ValueError(
                f"completed cache shape mismatch for {split}: "
                f"x={x_info['shape']} y={y_info['shape']}, "
                f"expected={expected_x_shape}/{expected_y_shape}"
            )
        manifest["splits"][split] = {
            "rows": expected[split],
            "x_shape": list(x_info["shape"]), "y_shape": list(y_info["shape"]),
            "x_path": str(x_path), "y_path": str(y_path),
            "x_sha256": x_info["sha256"], "y_sha256": y_info["sha256"],
            "x_finite": x_info["finite"], "y_finite": y_info["finite"],
        }
    partial = output_dir / "cache_manifest.json.partial"
    with open(partial, "w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2)
    os.replace(partial, output_dir / "cache_manifest.json")
    print(json.dumps(manifest, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence_jsonl", required=True)
    parser.add_argument("--fit_desc_ids", required=True)
    parser.add_argument("--calib_desc_ids", required=True)
    parser.add_argument("--split_summary", required=True)
    parser.add_argument("--stats_json", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--chunk_rows", type=int, default=8192)
    parser.add_argument("--num_workers", type=int, default=min(12, os.cpu_count() or 1))
    parser.add_argument(
        "--finalize_existing", action="store_true",
        help="Validate and manifest already-completed arrays without reparsing evidence.",
    )
    args = parser.parse_args()
    if "val" in Path(args.evidence_jsonl).name.lower():
        raise ValueError("C3.5 cache build is train-only")

    global _FIT_IDS, _CALIB_IDS, _FEATURE_NAMES, _MEAN, _STD
    _FIT_IDS = load_ids(args.fit_desc_ids)
    _CALIB_IDS = load_ids(args.calib_desc_ids)
    if _FIT_IDS & _CALIB_IDS:
        raise ValueError("fit/calib desc_id overlap")
    summary = read_json(args.split_summary)
    expected = {
        split: int(summary["splits"][split]["rows"])
        for split in ("train_fit", "train_calib")
    }
    stats = FeatureStats.from_dict(read_json(args.stats_json))
    _FEATURE_NAMES = stats.feature_names
    if _FEATURE_NAMES != C35_MATCHED_FEATURE_SETS["qsp53_with_coverage"]:
        raise ValueError("cache requires canonical qsp53_with_coverage feature order")
    if stats.count != expected["train_fit"]:
        raise ValueError("feature stats were not fit on the complete train_fit split")
    _MEAN = np.asarray(stats.mean, dtype=np.float32)
    _STD = np.asarray(stats.std, dtype=np.float32)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for split in ("train_fit", "train_calib"):
        paths[split] = {
            "x_partial": output_dir / f"{split}_x.partial.npy",
            "y_partial": output_dir / f"{split}_y.partial.npy",
            "x": output_dir / f"{split}_x.npy",
            "y": output_dir / f"{split}_y.npy",
        }

    if args.finalize_existing:
        finalize_cache(output_dir, paths, expected, args)
        return

    arrays = {}
    for split in ("train_fit", "train_calib"):
        arrays[split] = {
            "x": np.lib.format.open_memmap(
                paths[split]["x_partial"], mode="w+", dtype=np.float32,
                shape=(expected[split], len(_FEATURE_NAMES)),
            ),
            "y": np.lib.format.open_memmap(
                paths[split]["y_partial"], mode="w+", dtype=np.float32,
                shape=(expected[split], len(LABEL_NAMES)),
            ),
            "offset": 0,
        }

    total_rows = sum(expected.values())
    context = mp.get_context("fork")
    with context.Pool(processes=args.num_workers) as pool, tqdm(total=total_rows, desc="build C3.5 cache") as progress:
        results = pool.imap(
            process_line_chunk,
            iter_line_chunks(args.evidence_jsonl, args.chunk_rows),
            chunksize=1,
        )
        for result in results:
            progress.update(result["input_rows"])
            for split in ("train_fit", "train_calib"):
                x, y = result[split]
                start = arrays[split]["offset"]
                end = start + len(x)
                if end > expected[split]:
                    raise ValueError(f"cache overflow for {split}: {end} > {expected[split]}")
                arrays[split]["x"][start:end] = x
                arrays[split]["y"][start:end] = y
                arrays[split]["offset"] = end

    for split in arrays:
        if arrays[split]["offset"] != expected[split]:
            raise ValueError(f"cache row mismatch for {split}")
        arrays[split]["x"].flush()
        arrays[split]["y"].flush()
        del arrays[split]["x"], arrays[split]["y"]
        os.replace(paths[split]["x_partial"], paths[split]["x"])
        os.replace(paths[split]["y_partial"], paths[split]["y"])

    finalize_cache(output_dir, paths, expected, args)


if __name__ == "__main__":
    main()
