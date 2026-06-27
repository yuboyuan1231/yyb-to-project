#!/usr/bin/env python
"""Stage guard helper for the C4 full scaffold.

This helper validates the requested stage and prints the canonical command plan.
It intentionally does not execute long-running commands; Codex should run the
printed commands stepwise and stop at each audit gate.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

STAGES = [
    "compile_only",
    "c4_lite_train_calib",
    "c4_lite_official_val_one_shot",
    "c4_r2_build_datasets",
    "c4_r2_train_calibrator",
    "c4_r2_train_calib_search",
    "c4_r2_official_val_one_shot",
    "c4_main_dry_run",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=STAGES, required=True)
    ap.add_argument("--output_json", default="c4_audit/C4_STAGE_PLAN.json")
    ap.add_argument("--authorize_official_val", action="store_true")
    ap.add_argument("--authorize_c4_r2", action="store_true")
    ap.add_argument("--authorize_c4_main", action="store_true")
    args = ap.parse_args()
    if "official_val" in args.stage and not args.authorize_official_val:
        raise RuntimeError("Official-val stage requested without --authorize_official_val")
    if args.stage.startswith("c4_r2") and not args.authorize_c4_r2:
        raise RuntimeError("C4-r2-cal stage requested without --authorize_c4_r2")
    if args.stage.startswith("c4_main") and not args.authorize_c4_main:
        raise RuntimeError("C4-main stage requested without --authorize_c4_main")
    plan = {
        "stage": args.stage,
        "authorized_official_val": args.authorize_official_val,
        "authorized_c4_r2": args.authorize_c4_r2,
        "authorized_c4_main": args.authorize_c4_main,
        "status": "PLAN_PASS",
        "execute_long_running_commands": False,
    }
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(plan, f, indent=2)
        f.write("\n")
    print(json.dumps(plan, indent=2))

if __name__ == "__main__":
    main()
