#!/usr/bin/env python
"""Freeze C5-main-inference-A endpoint-injection statistics on train_fit."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from rlem.c5_main_a_utils import (  # noqa: E402
    ALPHAS, BETAS, LAMBDAS, TAUS, EndpointPriorEngine, artifact, exact_candidate_indices,
    fit_delta_stats, fit_retrieval_stats, group_max, load_c4_final_scores,
    load_npz_arrays, retrieval_gate, retrieval_z, setting_id,
)
from rlem.c5_prior_utils import write_json  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cache_npz", required=True)
    p.add_argument("--c4_final_scores_npz", required=True)
    p.add_argument("--temporal_prior_npz", required=True)
    p.add_argument("--output_stats_json", required=True)
    p.add_argument("--split_role", choices=["train_fit"], default="train_fit")
    p.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    p.add_argument("--chunk_rows", type=int, default=1_000_000)
    return p.parse_args()


def main():
    args = parse_args()
    out = Path(args.output_stats_json)
    if out.exists():
        raise FileExistsError(out)
    cache = load_npz_arrays(args.cache_npz, ("row_group_id", "start_time", "end_time", "group_ids_sorted_unique"))
    row_gid = cache["row_group_id"].astype(np.int64)
    group_count = len(cache["group_ids_sorted_unique"])
    if not np.array_equal(cache["group_ids_sorted_unique"].astype(np.int64), np.arange(group_count)):
        raise ValueError("C5-main-A requires dense aligned group ids")
    score = load_c4_final_scores(args.c4_final_scores_npz, len(row_gid))
    temporal_length = load_npz_arrays(args.temporal_prior_npz, ("temporal_length",))["temporal_length"].astype(np.int64)
    if len(temporal_length) != group_count:
        raise ValueError("Temporal/cache group counts differ")
    start_idx, end_idx, index_audit = exact_candidate_indices(
        cache["start_time"], cache["end_time"], row_gid, temporal_length
    )
    group_score = group_max(score, row_gid, group_count)
    retrieval_stats = fit_retrieval_stats(group_score)
    rz = retrieval_z(group_score, retrieval_stats)
    gate_stats = {}
    gates = {}
    for tau in TAUS:
        gate = retrieval_gate(rz, tau); gates[tau] = gate
        gate_stats[f"tau_{tau:.2f}"] = {
            "min": float(np.min(gate)), "max": float(np.max(gate)),
            "mean": float(np.mean(gate)), "std": float(np.std(gate)),
            "at_g_min_ratio": float(np.mean(gate <= .2500001)),
            "at_g_max_ratio": float(np.mean(gate >= 1.7499999)),
        }

    engine = EndpointPriorEngine.load(args.temporal_prior_npz, args.device)
    delta_stats, prior_sanity = {}, {}
    for alpha in ALPHAS:
        for beta in BETAS:
            log_s, log_e, sanity = engine.role_log_priors(alpha, beta)
            ab_key = f"a{alpha:.2f}_b{beta:.2f}"
            prior_sanity[ab_key] = sanity
            for tau in TAUS:
                for lambda_ in LAMBDAS:
                    key = setting_id(alpha, beta, tau, lambda_)
                    delta = engine.injected_endpoint_delta(
                        log_s, log_e, np.float32(lambda_) * gates[tau],
                        row_gid, start_idx, end_idx, chunk_rows=args.chunk_rows,
                    )
                    delta_stats[key] = {
                        "alpha": float(alpha), "beta": float(beta), "tau": float(tau), "lambda": float(lambda_),
                        **fit_delta_stats(delta),
                    }
                    del delta
            del log_s, log_e

    status = "PASS"
    if any(v["nonfinite"] for v in delta_stats.values()):
        status = "FAIL"
    if any(v["nonfinite_valid_values"] for v in prior_sanity.values()):
        status = "FAIL"
    manifest = {
        "status": status,
        "stage": "C5-main-inference-A train_fit statistics freeze",
        "scope": "train_fit_only",
        "stats_source": "train_fit_only",
        "official_val_used": False,
        "post_val_adjustment": False,
        "primary_baseline": "C4_final = C4-r2-cal-v2.1 v21_00444",
        "candidate_rows": int(len(row_gid)),
        "video_groups": int(group_count),
        "endpoint_index_audit": index_audit,
        "retrieval_score_stats": retrieval_stats,
        "retrieval_gate_definition": "clip(2*sigmoid(R_video_z/tau), 0.25, 1.75)",
        "retrieval_gate_stats": gate_stats,
        "role_prior_definition": {
            "start": "Normalize[P_b^(1+alpha) * (P_ctx+eps)^beta]",
            "end": "Normalize[P_e^(1+alpha) * (P_ctx+eps)^beta]",
            "P_ctx_role": "weak gate only; beta <= 0.10",
        },
        "delta_definition": "[log(new_P_b[i])-log(P_b[i])] + [log(new_P_e[j])-log(P_e[j])], where new_P_role is normalized after P_role * P_role_prior^(lambda*gate)",
        "logit_gauge_correction": {
            "gauge_invariant": True,
            "reason": "Frozen exports identify begin/end probabilities but not arbitrary raw-logit constants; within-video posterior normalization prevents those constants from leaking into cross-video scores.",
            "discarded_raw_formula": "lambda*gate*(log(P_s_prior[i])+log(P_e_prior[j]))"
        },
        "prior_sanity": prior_sanity,
        "delta_stats": delta_stats,
        "artifacts": {
            "cache_npz": artifact(args.cache_npz),
            "c4_final_scores_npz": artifact(args.c4_final_scores_npz),
            "temporal_prior_npz": artifact(args.temporal_prior_npz),
        },
        "runtime": {"device": str(engine.device), "chunk_rows": int(args.chunk_rows)},
        "retrieval_constants": {"effective_top_n": 100, "nms_thd": .7, "max_after_nms": 100},
        "used": {"C5_main_inference_A": True, "training": False, "C5_main_B": False, "C6": False},
    }
    write_json(str(out), manifest)
    print(json.dumps({"status": status, "output": str(out)}, indent=2))


if __name__ == "__main__":
    main()
