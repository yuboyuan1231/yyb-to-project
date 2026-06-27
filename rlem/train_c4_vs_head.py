#!/usr/bin/env python
"""Guarded entry for future C4-main VS/R2 head training.

This scaffold refuses to train by default.  It exists so the full C4 code tree is
complete and py_compile-able while preventing accidental backbone modification.
Use C4-lite and C4-r2-cal first.  Only after those pass should a new protocol
replace this guarded stub with source-specific CONQUER training code.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_json", required=True)
    ap.add_argument("--authorize_c4_main", action="store_true")
    ap.add_argument("--dry_run", action="store_true")
    return ap.parse_args()


def main():
    args = parse_args()
    if not args.authorize_c4_main:
        raise RuntimeError(
            "C4-main VS/R2 head training is not authorized. Run C4-lite and C4-r2-cal gates first."
        )
    if not args.dry_run:
        raise RuntimeError(
            "This scaffold does not implement backbone training. Use --dry_run for inspection only, "
            "then create a separate C4-main protocol-specific trainer."
        )
    report = {
        "status": "DRY_RUN_PASS",
        "c4_main_training_executed": False,
        "modifies_conquer": False,
        "official_val_used": False,
        "message": "C4-main is guarded pending separate authorization and source-specific integration.",
    }
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
