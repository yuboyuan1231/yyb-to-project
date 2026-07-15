"""Thin six-action runner for C28F F0/F1 recovery completion."""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Mapping, Sequence

sys.dont_write_bytecode = True
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ.setdefault("OMP_NUM_THREADS", str(max(1, (os.cpu_count() or 1) - 1)))
os.environ.setdefault("MKL_NUM_THREADS", os.environ["OMP_NUM_THREADS"])

from .atomic_io import BootstrapLease
from .runtime_control import F0_A_ANALYZE_ACTION, F0_B_ANALYZE_ACTION, RUN_F0_B_ACTION
from .runtime_stages import (
    G4_ROLE_POLICY_LOCK_ACTION,
    G6_VERIFY_F1_ACTION,
    G7_FINALIZE_ACTION,
    run_f0_a_analyze,
    run_f0_b,
    run_f0_b_analyze,
    run_g4_role_policy_lock,
    run_g6_verify_f1,
    run_g7_finalize,
)


RUNTIME_ACTIONS = (
    F0_A_ANALYZE_ACTION,
    G4_ROLE_POLICY_LOCK_ACTION,
    RUN_F0_B_ACTION,
    F0_B_ANALYZE_ACTION,
    G6_VERIFY_F1_ACTION,
    G7_FINALIZE_ACTION,
)


def dispatch(action: str, bootstrap_lease: BootstrapLease) -> Mapping[str, Any]:
    handlers = {
        F0_A_ANALYZE_ACTION: run_f0_a_analyze,
        G4_ROLE_POLICY_LOCK_ACTION: run_g4_role_policy_lock,
        RUN_F0_B_ACTION: lambda: run_f0_b(bootstrap_lease),
        F0_B_ANALYZE_ACTION: lambda: run_f0_b_analyze(
            bootstrap_lease=bootstrap_lease
        ),
        G6_VERIFY_F1_ACTION: run_g6_verify_f1,
        G7_FINALIZE_ACTION: run_g7_finalize,
    }
    try:
        handler = handlers[action]
    except KeyError as exc:
        raise RuntimeError(f"unsupported runtime action: {action}") from exc
    return handler()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "C28F F0/F1 recovery completion; no F0-A replay, training, "
            "protected evaluation, official workflow, or F2+ action exists."
        )
    )
    parser.add_argument("--action", required=True, choices=RUNTIME_ACTIONS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    if sys.flags.dont_write_bytecode != 1:
        raise RuntimeError("run with python -B")
    args = build_parser().parse_args(argv)
    with BootstrapLease(args.action) as bootstrap_lease:
        result = dispatch(args.action, bootstrap_lease)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
