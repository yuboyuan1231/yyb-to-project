#!/usr/bin/env python
"""Freeze C5-lite-prior term statistics on train_fit only."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from rlem.c5_prior_utils import C5_PRIOR_TERM_NAMES, fit_term_stats, sha256, write_json  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--features_npz", required=True)
    p.add_argument("--output_stats_json", required=True)
    p.add_argument("--split_role", choices=["train_fit"], default="train_fit")
    return p.parse_args()


def main():
    args = parse_args()
    if Path(args.output_stats_json).exists():
        raise FileExistsError(args.output_stats_json)
    with np.load(args.features_npz, allow_pickle=True) as payload:
        features = {key: payload[key] for key in payload.files}
    missing = [f"term_{name}" for name in C5_PRIOR_TERM_NAMES if f"term_{name}" not in features]
    if missing:
        raise KeyError(f"Missing C5 prior terms in feature NPZ: {missing}")
    stats = fit_term_stats(features)
    nonfinite = {name: int((~np.isfinite(features[f"term_{name}"])).sum()) for name in C5_PRIOR_TERM_NAMES}
    manifest = {
        "status": "PASS" if all(v == 0 for v in nonfinite.values()) else "FAIL",
        "stage": "C5-lite-prior term statistics freeze",
        "scope": "train_fit_only",
        "split_role": args.split_role,
        "official_val_used": False,
        "post_val_adjustment": False,
        "features_npz": {"path": args.features_npz, "sha256": sha256(args.features_npz)},
        "candidate_rows": int(len(features["s_c4_final"])),
        "term_names": C5_PRIOR_TERM_NAMES,
        "term_stats": stats,
        "nonfinite_by_term": nonfinite,
        "retrieval_constants": {"effective_top_n": 100, "nms_thd": 0.7, "max_after_nms": 100},
        "used": {"C5-lite-prior": True, "C5-main": False, "C6": False, "CONQUER_modified": False},
    }
    write_json(args.output_stats_json, manifest)
    print(json.dumps({"status": manifest["status"], "output": args.output_stats_json}, indent=2))


if __name__ == "__main__":
    main()
