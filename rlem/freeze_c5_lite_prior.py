#!/usr/bin/env python
"""Materialize C5-lite-prior freeze review/manifest after train_calib search."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from rlem.c5_prior_utils import C5_PRIOR_TERM_NAMES, sha256, write_json  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train_calib_audit_json", required=True)
    p.add_argument("--term_stats_json", required=True)
    p.add_argument("--output_review_md", required=True)
    p.add_argument("--output_manifest_json", required=True)
    p.add_argument("--output_hashes_json", required=True)
    return p.parse_args()


def main():
    args = parse_args()
    for path in [args.output_review_md, args.output_manifest_json, args.output_hashes_json]:
        if Path(path).exists():
            raise FileExistsError(path)
    audit = json.loads(Path(args.train_calib_audit_json).read_text(encoding="utf-8"))
    stats = json.loads(Path(args.term_stats_json).read_text(encoding="utf-8"))
    if audit.get("official_val_used") is not False:
        raise ValueError("C5-lite-prior freeze review cannot use official val")
    if stats.get("official_val_used") is not False:
        raise ValueError("C5-lite-prior term stats must be train_fit-only")
    status = "C5_LITE_PRIOR_FREEZE_REVIEW_PASS" if audit.get("status") == "PASS" else "C5_LITE_PRIOR_FREEZE_REVIEW_NO_PROMOTION"
    best = audit["best_config"]
    flags = audit.get("promotion_flags", {})
    manifest = {
        "status": status,
        "selected_config": best.get("config_id"),
        "selected_family": best.get("family"),
        "primary_baseline": "C4_final = C4-r2-cal-v2.1 v21_00444",
        "stage": "C5-lite-prior",
        "official_val_used": False,
        "official_val_authorized": False,
        "post_val_adjustment": False,
        "score_grid_on_val": False,
        "temperature_search_on_val": False,
        "model_retrained": False,
        "capacity_search": False,
        "modifies_conquer": False,
        "C5_main_used": False,
        "C6_used": False,
        "retrieval_constants": audit.get("retrieval_constants"),
        "term_names": C5_PRIOR_TERM_NAMES,
        "term_stats": stats.get("term_stats"),
        "score_config": best,
        "metrics_vs_c4_final": audit.get("best_deltas_vs_c4_final"),
        "localization_diagnostics": audit.get("best_localization"),
        "movement_diagnostics": audit.get("best_movement"),
        "prior_diagnostics": audit.get("best_prior_diagnostics"),
        "promotion_flags": flags,
        "artifacts": {
            "train_calib_audit_json": {"path": args.train_calib_audit_json, "sha256": sha256(args.train_calib_audit_json)},
            "term_stats_json": {"path": args.term_stats_json, "sha256": sha256(args.term_stats_json)},
        },
    }
    write_json(args.output_manifest_json, manifest)
    md = []
    md.append("# C5-lite-prior Freeze Review\n")
    md.append("## Decision\n")
    md.append(f"- Status: `{status}`\n")
    md.append(f"- Selected config: `{best.get('config_id')}`\n")
    md.append("- Primary baseline: `C4_final = C4-r2-cal-v2.1 v21_00444`\n")
    md.append("- Official val used: `false`\n")
    md.append("- CONQUER modified: `false`\n")
    md.append("\n## Promotion flags\n")
    for k, v in flags.items():
        md.append(f"- `{k}`: `{str(v).lower()}`\n")
    md.append("\n## Metric deltas vs C4_final\n")
    for k, v in (audit.get("best_deltas_vs_c4_final") or {}).items():
        md.append(f"- `{k}`: `{v}`\n")
    md.append("\n## Localization diagnostics\n")
    for k, v in (audit.get("best_localization") or {}).items():
        md.append(f"- `{k}`: `{v}`\n")
    md.append("\n## Score config\n")
    md.append("```json\n")
    md.append(json.dumps(best, indent=2, ensure_ascii=False))
    md.append("\n```\n")
    Path(args.output_review_md).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_review_md).write_text("".join(md), encoding="utf-8")
    hashes = {
        "review_md": {"path": args.output_review_md, "sha256": sha256(args.output_review_md)},
        "manifest_json": {"path": args.output_manifest_json, "sha256": sha256(args.output_manifest_json)},
        "train_calib_audit_json": {"path": args.train_calib_audit_json, "sha256": sha256(args.train_calib_audit_json)},
        "term_stats_json": {"path": args.term_stats_json, "sha256": sha256(args.term_stats_json)},
    }
    write_json(args.output_hashes_json, hashes)
    print(json.dumps({"status": status, "manifest": args.output_manifest_json}, indent=2))


if __name__ == "__main__":
    main()
