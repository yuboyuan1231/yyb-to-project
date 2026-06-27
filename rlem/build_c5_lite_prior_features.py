#!/usr/bin/env python
"""Build C5-lite-prior span features from per-video temporal priors.

Input contract:
  - C4 cache NPZ from the frozen fixed-candidate pipeline.
  - C4_final row scores from C4-r2-cal-v2.1 v21_00444.
  - temporal prior NPZ containing group_id, p_ctx, p_b, p_e.

Output:
  - row-aligned NPZ with raw prior span features, search terms, and C4_final baseline.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from rlem.c5_prior_utils import (  # noqa: E402
    C5_PRIOR_RAW_FEATURES,
    C5_PRIOR_TERM_NAMES,
    TemporalPriorStore,
    atomic_npz,
    build_c5_prior_features,
    load_c4_final_score,
    load_cache,
    require_200_rows,
    sha256,
    validate_temporal_store,
    write_json,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cache_npz", required=True)
    p.add_argument("--c4_final_scores_npz", required=True)
    p.add_argument("--temporal_prior_npz", required=True)
    p.add_argument("--output_features_npz", required=True)
    p.add_argument("--output_audit_json", required=True)
    p.add_argument("--split_role", choices=["train_fit", "train_calib", "val"], required=True)
    p.add_argument("--allow_official_val", action="store_true")
    p.add_argument("--c4_final_score_key", default="s_v21")
    p.add_argument("--clip_length", type=float, default=1.5)
    p.add_argument("--a_ctx", type=float, default=1.0)
    p.add_argument("--a_bd", type=float, default=1.0)
    p.add_argument("--a_ctx_ret", type=float, default=1.0)
    p.add_argument("--a_bd_ret", type=float, default=1.0)
    p.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    p.add_argument("--row_chunk_size", type=int, default=262144)
    return p.parse_args()


def main():
    args = parse_args()
    if args.split_role == "val" and not args.allow_official_val:
        raise ValueError("Refusing official-val C5 prior feature build without --allow_official_val")
    for path in [args.output_features_npz, args.output_audit_json]:
        if Path(path).exists():
            raise FileExistsError(path)
    cache = load_cache(args.cache_npz)
    queries, rows = require_200_rows(cache)
    baseline, baseline_artifact = load_c4_final_score(cache, args.c4_final_scores_npz, score_key=args.c4_final_score_key)
    temporal = TemporalPriorStore.load(args.temporal_prior_npz)
    temporal_diag = validate_temporal_store(cache, temporal)
    features = build_c5_prior_features(
        cache,
        temporal,
        baseline,
        clip_length=args.clip_length,
        a_ctx=args.a_ctx,
        a_bd=args.a_bd,
        a_ctx_ret=args.a_ctx_ret,
        a_bd_ret=args.a_bd_ret,
        device=args.device,
        row_chunk_size=args.row_chunk_size,
    )
    arrays = {
        "row_group_id": cache["row_group_id"].astype(np.int32),
        "query_index": cache["query_index"].astype(np.int32),
        "video_idx": cache["video_idx"].astype(np.int32),
        "start_time": cache["start_time"].astype(np.float32),
        "end_time": cache["end_time"].astype(np.float32),
        "s_c4_final": baseline.astype(np.float32),
        **{key: features[key].astype(np.float32) for key in C5_PRIOR_RAW_FEATURES if key in features},
        **{f"term_{key}": features[f"term_{key}"].astype(np.float32) for key in C5_PRIOR_TERM_NAMES},
    }
    atomic_npz(args.output_features_npz, **arrays)
    nonfinite = {key: int((~np.isfinite(value)).sum()) for key, value in arrays.items() if value.dtype.kind in "fc"}
    audit = {
        "status": "PASS" if all(v == 0 for v in nonfinite.values()) and temporal_diag["bad_group_or_nonfinite_count"] == 0 else "FAIL",
        "stage": "C5-lite-prior feature export",
        "split_role": args.split_role,
        "official_val_used": bool(args.split_role == "val" and args.allow_official_val),
        "post_val_adjustment": False,
        "baseline": "C4-r2-cal-v2.1 C4_final",
        "cache_npz": {"path": args.cache_npz, "sha256": sha256(args.cache_npz)},
        "c4_final_scores_npz": baseline_artifact,
        "temporal_prior_npz": {"path": args.temporal_prior_npz, "sha256": sha256(args.temporal_prior_npz)},
        "output_features_npz": {"path": args.output_features_npz, "sha256": sha256(args.output_features_npz)},
        "queries": int(queries),
        "candidate_rows": int(rows),
        "raw_feature_names": C5_PRIOR_RAW_FEATURES,
        "term_names": C5_PRIOR_TERM_NAMES,
        "temporal_diagnostics": temporal_diag,
        "clip_length": float(args.clip_length),
        "retloc_prior_weights": {"a_ctx": args.a_ctx, "a_bd": args.a_bd, "a_ctx_ret": args.a_ctx_ret, "a_bd_ret": args.a_bd_ret},
        "aggregation_runtime": {"device": args.device, "row_chunk_size": args.row_chunk_size, "vectorized_dense_path": bool(temporal.dense)},
        "nonfinite_by_array": nonfinite,
        "retrieval_constants": {"effective_top_n": 100, "nms_thd": 0.7, "max_after_nms": 100},
        "used": {"C5-lite-prior": True, "C5-main": False, "C6": False, "CONQUER_modified": False},
    }
    write_json(args.output_audit_json, audit)
    print(json.dumps({"status": audit["status"], "features": args.output_features_npz}, indent=2))


if __name__ == "__main__":
    main()
