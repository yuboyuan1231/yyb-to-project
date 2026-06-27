#!/usr/bin/env python
"""Convert a read-only temporal-evidence JSONL export into C5 temporal-prior NPZ.

This is an adapter for Codex if the CONQUER read-only export is easier to write
as JSONL first.  Each JSONL row must represent one (query, video) group and
contain:
  - desc_id or query_index
  - video_idx
  - group_id matching C4 cache row_group_id/group_ids_sorted_unique
  - p_b/start_prob: list[float]
  - p_e/end_prob: list[float]
  - p_ctx/qal_ctx_prior: list[float]
Optional:
  - clip_start_time, clip_end_time: list[float]

The script validates that group_id order exactly matches C4 cache
`group_ids_sorted_unique`; otherwise it refuses the export.
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from rlem.c5_prior_utils import atomic_npz, load_cache, sha256, write_json  # noqa: E402


def open_text(path: str):
    return gzip.open(path, "rt", encoding="utf-8") if str(path).endswith(".gz") else open(path, "r", encoding="utf-8")


def pick(row, keys):
    for k in keys:
        if k in row and row[k] is not None:
            return row[k]
    return None


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--temporal_jsonl", required=True)
    p.add_argument("--cache_npz", required=True)
    p.add_argument("--output_npz", required=True)
    p.add_argument("--output_audit_json", required=True)
    p.add_argument("--split_role", choices=["train_fit", "train_calib", "val"], required=True)
    p.add_argument("--allow_official_val", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    if args.split_role == "val" and not args.allow_official_val:
        raise ValueError("Refusing official-val temporal JSONL conversion without --allow_official_val")
    for path in [args.output_npz, args.output_audit_json]:
        if Path(path).exists():
            raise FileExistsError(path)
    cache = load_cache(args.cache_npz)
    expected_gid = cache["group_ids_sorted_unique"].astype(np.int64)
    by_gid = {}
    rows = 0
    with open_text(args.temporal_jsonl) as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            rows += 1
            gid = int(pick(row, ["group_id", "row_group_id", "video_group_id"]))
            p_ctx = pick(row, ["p_ctx", "P_ctx", "ctx_prior", "qal_ctx_prior"])
            p_b = pick(row, ["p_b", "P_b", "start_prob", "pb", "start_prior"])
            p_e = pick(row, ["p_e", "P_e", "end_prob", "pe", "end_prior"])
            if p_ctx is None or p_b is None or p_e is None:
                raise KeyError("Temporal JSONL row lacks p_ctx/p_b/p_e arrays")
            p_ctx = np.asarray(p_ctx, dtype=np.float32)
            p_b = np.asarray(p_b, dtype=np.float32)
            p_e = np.asarray(p_e, dtype=np.float32)
            if len(p_ctx) == 0 or len(p_b) != len(p_ctx) or len(p_e) != len(p_ctx):
                raise ValueError(f"Invalid temporal lengths for group_id={gid}")
            cst = pick(row, ["clip_start_time", "clip_start_times"])
            cet = pick(row, ["clip_end_time", "clip_end_times"])
            if cst is not None or cet is not None:
                cst = np.asarray(cst, dtype=np.float32)
                cet = np.asarray(cet, dtype=np.float32)
                if len(cst) != len(p_ctx) or len(cet) != len(p_ctx):
                    raise ValueError(f"Invalid clip time lengths for group_id={gid}")
            by_gid[gid] = (p_ctx, p_b, p_e, cst, cet)
    missing = [int(g) for g in expected_gid if int(g) not in by_gid]
    extra = [int(g) for g in by_gid if int(g) not in set(expected_gid.tolist())]
    if missing or extra:
        raise ValueError(f"Temporal JSONL group mismatch: missing={len(missing)} extra={len(extra)}")
    offsets = [0]
    p_ctx_all, p_b_all, p_e_all = [], [], []
    cst_all, cet_all = [], []
    has_times = True
    nonfinite = 0
    for gid in expected_gid:
        p_ctx, p_b, p_e, cst, cet = by_gid[int(gid)]
        p_ctx_all.append(p_ctx)
        p_b_all.append(p_b)
        p_e_all.append(p_e)
        offsets.append(offsets[-1] + len(p_ctx))
        nonfinite += int((~np.isfinite(p_ctx)).sum() + (~np.isfinite(p_b)).sum() + (~np.isfinite(p_e)).sum())
        if cst is None or cet is None:
            has_times = False
        else:
            cst_all.append(cst)
            cet_all.append(cet)
            nonfinite += int((~np.isfinite(cst)).sum() + (~np.isfinite(cet)).sum())
    arrays = {
        "group_id": expected_gid.astype(np.int64),
        "temporal_offsets": np.asarray(offsets, dtype=np.int64),
        "p_ctx": np.concatenate(p_ctx_all).astype(np.float32),
        "p_b": np.concatenate(p_b_all).astype(np.float32),
        "p_e": np.concatenate(p_e_all).astype(np.float32),
    }
    if has_times:
        arrays["clip_start_time"] = np.concatenate(cst_all).astype(np.float32)
        arrays["clip_end_time"] = np.concatenate(cet_all).astype(np.float32)
    atomic_npz(args.output_npz, **arrays)
    audit = {
        "status": "PASS" if nonfinite == 0 else "FAIL",
        "stage": "C5-0 temporal prior JSONL to NPZ adapter",
        "split_role": args.split_role,
        "official_val_used": bool(args.split_role == "val" and args.allow_official_val),
        "post_val_adjustment": False,
        "temporal_jsonl": {"path": args.temporal_jsonl, "sha256": sha256(args.temporal_jsonl)},
        "cache_npz": {"path": args.cache_npz, "sha256": sha256(args.cache_npz)},
        "output_npz": {"path": args.output_npz, "sha256": sha256(args.output_npz)},
        "input_rows": rows,
        "video_groups": int(len(expected_gid)),
        "total_temporal_positions": int(offsets[-1]),
        "has_clip_times": bool(has_times),
        "nonfinite_values": int(nonfinite),
        "used": {"C5-lite-prior": False, "C5-main": False, "C6": False, "CONQUER_modified": False},
    }
    write_json(args.output_audit_json, audit)
    print(json.dumps({"status": audit["status"], "output_npz": args.output_npz}, indent=2))


if __name__ == "__main__":
    main()
