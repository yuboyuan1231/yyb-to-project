#!/usr/bin/env python
"""Future C5-lite-prior official-val one-shot reranker.

Do not run during train_calib development.  This script requires an explicit
--allow_official_val flag and a durable C5-lite-prior freeze manifest.
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
    load_cache,
    eval_score,
    score_with_config,
    standardized_term_matrix,
    movement_diagnostics,
    localization_diagnostics,
    prior_diagnostics,
    metric_deltas,
    sha256,
    write_json,
    atomic_npz,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cache_npz", required=True)
    p.add_argument("--features_npz", required=True)
    p.add_argument("--freeze_manifest_json", required=True)
    p.add_argument("--gt_jsonl", required=True)
    p.add_argument("--dataset_config", required=True)
    p.add_argument("--output_scores_npz", required=True)
    p.add_argument("--output_submission_json", required=True)
    p.add_argument("--output_metrics_json", required=True)
    p.add_argument("--output_audit_json", required=True)
    p.add_argument("--split", default="val")
    p.add_argument("--allow_official_val", action="store_true")
    p.add_argument("--effective_top_n", type=int, default=100)
    p.add_argument("--max_after_nms", type=int, default=100)
    p.add_argument("--nms_thd", type=float, default=0.7)
    p.add_argument("--no_desc_type", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    if not args.allow_official_val:
        raise ValueError("Refusing C5-lite-prior official-val one-shot without --allow_official_val")
    for path in [args.output_scores_npz, args.output_submission_json, args.output_metrics_json, args.output_audit_json]:
        if Path(path).exists():
            raise FileExistsError(path)
    manifest = json.loads(Path(args.freeze_manifest_json).read_text(encoding="utf-8"))
    if manifest.get("status") != "C5_LITE_PRIOR_FREEZE_REVIEW_PASS":
        raise ValueError("C5-lite-prior official val requires a PASS freeze review")
    if manifest.get("official_val_used") is not False or manifest.get("post_val_adjustment") is not False:
        raise ValueError("Invalid C5-lite-prior freeze manifest protocol flags")
    cache = load_cache(args.cache_npz)
    with np.load(args.features_npz, allow_pickle=True) as payload:
        features = {key: payload[key] for key in payload.files}
    baseline = features["s_c4_final"].astype(np.float32)
    z_terms = standardized_term_matrix(features, manifest["term_stats"])
    score = score_with_config(baseline, z_terms, manifest["score_config"])
    atomic_npz(args.output_scores_npz, row_group_id=cache["row_group_id"].astype(np.int32), s_c4_final=baseline, s_c5_prior=score)
    submission, metrics_nested, flat_metrics = eval_score(
        cache, score, args.dataset_config, args.split, args.effective_top_n, args.max_after_nms, args.nms_thd, args.gt_jsonl, no_desc_type=args.no_desc_type
    )
    Path(args.output_submission_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_submission_json).write_text(json.dumps(submission, ensure_ascii=False), encoding="utf-8")
    write_json(args.output_metrics_json, {"nested": metrics_nested, "flat": flat_metrics})
    # Official-val localization diagnostics require GT. They are diagnostic only;
    # no parameter changes are allowed after this script.
    loc = localization_diagnostics(cache, baseline, score)
    move = movement_diagnostics(cache, baseline, score)
    prior = prior_diagnostics(cache, features, baseline, score)
    audit = {
        "status": "PASS",
        "stage": "C5-lite-prior official-val one-shot",
        "official_val_one_shot": True,
        "learned_config_count": 1,
        "score_grid_on_val": False,
        "temperature_search_on_val": False,
        "post_val_adjustment": False,
        "config_id": manifest["selected_config"],
        "metrics": flat_metrics,
        "localization_diagnostics": loc,
        "movement_diagnostics": move,
        "prior_diagnostics": prior,
        "artifacts": {
            "freeze_manifest_json": {"path": args.freeze_manifest_json, "sha256": sha256(args.freeze_manifest_json)},
            "cache_npz": {"path": args.cache_npz, "sha256": sha256(args.cache_npz)},
            "features_npz": {"path": args.features_npz, "sha256": sha256(args.features_npz)},
            "scores_npz": {"path": args.output_scores_npz, "sha256": sha256(args.output_scores_npz)},
            "submission_json": {"path": args.output_submission_json, "sha256": sha256(args.output_submission_json)},
            "metrics_json": {"path": args.output_metrics_json, "sha256": sha256(args.output_metrics_json)},
        },
        "used": {"C5-lite-prior": True, "C5-main": False, "C6": False, "CONQUER_modified": False},
    }
    write_json(args.output_audit_json, audit)
    print(json.dumps({"status": "PASS", "metrics": args.output_metrics_json}, indent=2))


if __name__ == "__main__":
    main()
