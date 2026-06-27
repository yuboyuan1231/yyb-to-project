#!/usr/bin/env python
"""Guarded utilities for future C4-main / CONQUER VS-head integration.

This file deliberately performs inspection and dry-run checks only.  It does not
patch model/conquer.py or train a backbone head.  C4-main requires a separate
explicit protocol after C4-lite/C4-r2-cal prove positive signal.
"""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Dict, List

CANDIDATE_NAMES = ["vs_head", "video_score_head", "r2_head", "video_retrieval_head"]


def inspect_conquer_source(path: str) -> Dict:
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    tree = ast.parse(text)
    class_names = [n.name for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]
    function_names = [n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
    name_hits = {name: (name in text) for name in CANDIDATE_NAMES}
    return {
        "path": str(p),
        "classes": class_names,
        "functions_containing_pred": [name for name in function_names if "pred" in name.lower() or "forward" in name.lower()],
        "candidate_head_name_hits": name_hits,
        "contains_r2_literal": "r2" in text,
        "contains_vs_literal": "VS" in text or "vs" in text,
    }


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conquer_py", default="model/conquer.py")
    ap.add_argument("--output_json", required=True)
    ap.add_argument("--authorize_c4_main", action="store_true")
    ap.add_argument("--dry_run_only", action="store_true", default=True)
    return ap.parse_args()


def main():
    args = parse_args()
    info = inspect_conquer_source(args.conquer_py)
    info.update({
        "status": "DRY_RUN_PASS",
        "modifies_conquer": False,
        "trained_vs_head": False,
        "authorization_present": bool(args.authorize_c4_main),
        "note": "C4-main source modification is intentionally not performed by this scaffold.",
    })
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print(json.dumps(info, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
