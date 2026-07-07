from __future__ import annotations

from typing import Any, Dict

import pandas as pd

import run_c17_bmn_t2_native_vcmr_integration as c17


SPLITS = ["calib_select", "calib_holdout"]


def evaluate_by_split(df: pd.DataFrame, score_col: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for split in SPLITS:
        sdf = df[df["split"].astype(str) == split]
        out[split] = c17.evaluate_vcmr(sdf, score_col)
    return out


def summarize_holdout(results: Dict[str, Any], split: str = "calib_holdout") -> Dict[str, Any]:
    return results.get(split, {}).get("summary", {})


def metric_delta(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, float]:
    keys = sorted(set(a) | set(b))
    return {k: float(a.get(k, 0.0) or 0.0) - float(b.get(k, 0.0) or 0.0) for k in keys if isinstance(a.get(k, b.get(k, 0.0)), (int, float))}

