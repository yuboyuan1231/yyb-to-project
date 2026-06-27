#!/usr/bin/env python
"""C5-0 input gate: verify per-video temporal prior export.

This script must pass before C5-lite-prior feature construction.  It refuses
row-only C4 caches as a substitute for per-video temporal priors.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from rlem.c5_prior_utils import (  # noqa: E402
    TemporalPriorStore,
    load_cache,
    sha256,
    validate_temporal_store,
    write_json,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cache_npz", required=True)
    p.add_argument("--temporal_prior_npz", required=True)
    p.add_argument("--output_audit_json", required=True)
    p.add_argument("--split_role", choices=["train_fit", "train_calib", "val"], required=True)
    p.add_argument("--allow_official_val", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    if args.split_role == "val" and not args.allow_official_val:
        raise ValueError("Refusing official-val temporal-prior gate without --allow_official_val")
    cache = load_cache(args.cache_npz)
    store = TemporalPriorStore.load(args.temporal_prior_npz)
    diag = validate_temporal_store(cache, store)
    status = "PASS" if diag["bad_group_or_nonfinite_count"] == 0 else "FAIL"
    audit = {
        "status": status,
        "stage": "C5-0 temporal prior input gate",
        "split_role": args.split_role,
        "official_val_used": bool(args.split_role == "val" and args.allow_official_val),
        "post_val_adjustment": False,
        "cache_npz": {"path": args.cache_npz, "sha256": sha256(args.cache_npz)},
        "temporal_prior_npz": {"path": args.temporal_prior_npz, "sha256": sha256(args.temporal_prior_npz)},
        "diagnostics": diag,
        "required_arrays": ["group_id", "p_ctx", "p_b", "p_e"],
        "note": "P_ctx/P_b/P_e must be exported read-only from CONQUER/QAL/ML outputs; row-level candidate scores are not accepted as temporal priors.",
        "used": {"C5-lite-prior": False, "C5-main": False, "C6": False, "CONQUER_modified": False},
    }
    write_json(args.output_audit_json, audit)
    print(json.dumps({"status": status, "output": args.output_audit_json}, indent=2))


if __name__ == "__main__":
    main()
