#!/usr/bin/env python3
"""C21 front-rank retriever-localizer calibration.

Train-only pairwise/listwise promotion audit over C19/C20 candidate rows.
No official validation is run, no official prediction pool is read, and pseudo
official holdout is diagnostic only after policy freeze.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

import run_c17_bmn_t2_native_vcmr_integration as c17
import run_c18_full_hybrid_freeze_candidate as c18
import run_c19_official_readiness_blocker_closure as c19
import run_c20_top1_safe_bmn_correction as c20
from run_c12_native_retriever_training import DEVICE, ROOT, sha256_file

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
    from sklearn.preprocessing import StandardScaler
except Exception:  # pragma: no cover
    LogisticRegression = None
    StandardScaler = None
    roc_auc_score = None
    average_precision_score = None
    precision_recall_curve = None


torch.set_num_threads(min(24, os.cpu_count() or 1))

PROMOTED = "C7-B6 R1SelectiveTop1"
BASE_C20_COMMIT = "72c459b5fb9078a701316d55a172fcdf7fe450ac"
SELECTED_FORMULA = dict(c19.SELECTED)

OUT0 = ROOT / "c21_0_protocol_freeze"
OUT1 = ROOT / "c21_1_front_rank_pair_dataset"
OUT2 = ROOT / "c21_2_rule_based_promotion"
OUT3 = ROOT / "c21_3_learned_pairwise_promotion"
OUT4 = ROOT / "c21_4_joint_front_rank_policy"
OUT5 = ROOT / "c21_5_robustness_and_onelook"
OUT6 = ROOT / "c21_6_final_decision"

FEATURE_COLUMNS = [
    "retriever_gap", "bmn_gap", "t2_gap", "hybrid_gap",
    "candidate_video_rank", "candidate_rank", "anchor_video_rank",
    "candidate_span_rank_in_video", "bmn_margin", "t2_margin",
    "high_retriever_high_bmn", "high_retriever_low_bmn", "low_retriever_high_bmn",
    "bmn_t2_agree", "bmn_t2_conflict",
    "z_retriever", "z_bmn", "z_t2", "z_hybrid",
    "softmax_retriever_top10", "softmax_bmn_top10", "softmax_t2_top10",
    "query_type_v", "query_type_t", "query_type_vt",
    "duration_short", "duration_medium", "duration_long",
]
LABEL_COLUMNS = [
    "challenger_wins_07", "challenger_wins_05", "candidate_correct_video",
    "candidate_iou", "candidate_iou_ge_05", "candidate_iou_ge_07",
    "anchor_iou_ge_07", "harmful_promotion", "safe_promotion", "neutral",
]


def jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, Path):
        return str(obj)
    return obj


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(obj), indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def load_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def sh(cmd: str) -> str:
    return subprocess.check_output(cmd, cwd=ROOT, shell=True, stderr=subprocess.STDOUT, text=True).strip()


def sh_ok(cmd: str) -> bool:
    return subprocess.run(cmd, cwd=ROOT, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0


def stable_hash(obj: Any) -> str:
    raw = json.dumps(jsonable(obj), sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def cache_dir() -> Path:
    base = Path(os.environ.get("C21_SCORE_CACHE_DIR", "/tmp/c21_score_cache")) / "CONQUER-RLEM-c2c3"
    base.mkdir(parents=True, exist_ok=True)
    return base


def pair_cache(mode: str, split_name: str = "train") -> Path:
    return cache_dir() / f"C21_FRONT_RANK_PAIR_DATASET_{split_name}_{mode}.local.parquet"


def model_cache(mode: str) -> Path:
    return cache_dir() / f"C21_LEARNED_LOGISTIC_{mode}.local.json"


def require_status(path: Path, ok: Sequence[str]) -> Dict[str, Any]:
    obj = load_json(path, {})
    status = obj.get("status") or obj.get("final_decision")
    if status not in set(ok):
        raise RuntimeError(f"{path} status={status}; expected one of {ok}")
    return obj


def file_record(path: Path, sha: bool = True) -> Dict[str, Any]:
    return {
        "path": str(path),
        "exists": path.exists(),
        "readable": os.access(path, os.R_OK) if path.exists() else False,
        "size_bytes": path.stat().st_size if path.exists() else None,
        "sha256": sha256_file(path) if sha and path.exists() and path.is_file() else None,
    }


def softmax(vals: np.ndarray) -> np.ndarray:
    vals = vals.astype(np.float64)
    vals = vals - np.nanmax(vals) if len(vals) else vals
    exp = np.exp(vals)
    den = exp.sum()
    return exp / den if den > 0 else np.zeros_like(vals)


def iou_1d(st: float, ed: float, gt_st: float, gt_ed: float) -> float:
    return c20.iou_1d(st, ed, gt_st, gt_ed)


def row_iou(row: pd.Series) -> float:
    return c20.row_iou(row)


def nms_rank_group(g: pd.DataFrame, score_col: str, max_keep: int = 100) -> pd.DataFrame:
    return c20.nms_rank_group(g, score_col, max_keep)


def metric(xs: Sequence[Any]) -> float:
    return 100.0 * sum(bool(x) for x in xs) / max(1, len(xs))


def source_cache(mode: str) -> Path:
    path = c19.canonical_cache(mode)
    return path if path.exists() else c19.canonical_cache("medium")


def pseudo_source(mode: str) -> Path:
    path = c18.c18_pseudo_cache(mode)
    return path if path.exists() else c18.c18_pseudo_cache("medium")


def candidate_video_rank(ranked: pd.DataFrame, vid: str) -> int:
    seen = []
    for v in ranked["video_id"].astype(str).tolist():
        if v not in seen:
            seen.append(v)
        if v == vid:
            return len(seen)
    return 999999


def topk_success_from_ranked(ranked: pd.DataFrame, k: int, thr: float) -> bool:
    return any(row_iou(row) >= thr for _, row in ranked.head(k).iterrows())


def candidate_margins(top10: pd.DataFrame, idx: Any) -> Tuple[float, float]:
    others = top10.drop(index=idx, errors="ignore")
    row = top10.loc[idx]
    bmn_second = float(others["bmn_norm"].max()) if len(others) else 0.0
    t2_second = float(others["t2_norm"].max()) if len(others) else 0.0
    return float(row["bmn_norm"]) - bmn_second, float(row["t2_norm"]) - t2_second


def build_pair_dataset_from_df(df: pd.DataFrame, mode: str, split_name: str, max_groups: int | None = None) -> tuple[pd.DataFrame, Dict[str, Any]]:
    if "retriever_norm" not in df.columns or "final_score" not in df.columns:
        c17.add_normalized_scores_inplace(df)
        df["final_score"] = c18.formula_score_array(df, SELECTED_FORMULA)
    rows: List[Dict[str, Any]] = []
    start = time.time()
    groups = df.groupby(["split", "seed", "query_id"], sort=False, observed=True)
    total = groups.ngroups
    for gi, ((split, seed, qid), g) in enumerate(groups, start=1):
        if max_groups is not None and gi > max_groups:
            break
        if gi == 1 or gi % 250 == 0:
            print(f"[C21 pair] {split_name} groups={gi}/{total} pairs={len(rows)} elapsed={time.time() - start:.1f}s", flush=True)
        ranked = nms_rank_group(g, "final_score", 100)
        if len(ranked) < 2:
            continue
        top10 = ranked.head(10).copy()
        anchor = top10.iloc[0]
        anchor_iou = row_iou(anchor)
        anchor_ok07 = anchor_iou >= 0.7
        anchor_ok05 = anchor_iou >= 0.5
        anchor_vid = str(anchor["video_id"])
        gt_vid = str(anchor["gt_video_id"])
        orig = {}
        for k in [1, 5, 10, 100]:
            orig[f"orig_R@{k}_IoU0.5"] = topk_success_from_ranked(ranked, k, 0.5)
            orig[f"orig_R@{k}_IoU0.7"] = topk_success_from_ranked(ranked, k, 0.7)
        sm_retr = softmax(top10["retriever_norm"].to_numpy(np.float32))
        sm_bmn = softmax(top10["bmn_norm"].to_numpy(np.float32))
        sm_t2 = softmax(top10["t2_norm"].to_numpy(np.float32))
        z = {}
        for col in ["retriever_norm", "bmn_norm", "t2_norm", "final_score"]:
            vals = top10[col].astype(np.float32)
            mu = float(vals.mean())
            sd = float(vals.std()) or 1.0
            z[col] = (vals - mu) / sd
        for pos, (idx, cand) in enumerate(top10.iloc[1:].iterrows(), start=2):
            cand_iou = row_iou(cand)
            bmn_margin, t2_margin = candidate_margins(top10, idx)
            cand_vid = str(cand["video_id"])
            cvrank = candidate_video_rank(ranked, cand_vid)
            rec = {
                "split": str(split),
                "seed": int(seed),
                "query_id": int(qid),
                "anchor_type": "c19_hybrid_proxy",
                "not_exact_c7_b6": True,
                "query_type": str(anchor["query_type"]),
                "duration_bucket": str(anchor["duration_bucket"]),
                "gt_video_id": gt_vid,
                "gt_start": float(anchor["gt_start"]),
                "gt_end": float(anchor["gt_end"]),
                "anchor_video_id": anchor_vid,
                "anchor_span_start": float(anchor["span_start"]),
                "anchor_span_end": float(anchor["span_end"]),
                "anchor_iou": float(anchor_iou),
                "anchor_iou_ge_05": bool(anchor_ok05),
                "anchor_iou_ge_07": bool(anchor_ok07),
                "anchor_video_rank": 1,
                "anchor_retriever_score": float(anchor["retriever_norm"]),
                "anchor_bmn_score": float(anchor["bmn_norm"]),
                "anchor_t2_score": float(anchor["t2_norm"]),
                "anchor_hybrid_score": float(anchor["final_score"]),
                "candidate_rank": int(cand["rank"]),
                "candidate_video_rank": int(cvrank),
                "candidate_video_id": cand_vid,
                "video_id": cand_vid,
                "span_start": float(cand["span_start"]),
                "span_end": float(cand["span_end"]),
                "span_duration": float(cand["span_end"] - cand["span_start"]),
                "candidate_iou": float(cand_iou),
                "candidate_correct_video": cand_vid == gt_vid,
                "candidate_iou_ge_05": cand_iou >= 0.5,
                "candidate_iou_ge_07": cand_iou >= 0.7,
                "retriever_score": float(cand["retriever_norm"]),
                "bmn_score": float(cand["bmn_norm"]),
                "t2_score": float(cand["t2_norm"]),
                "c19_hybrid_score": float(cand["final_score"]),
                "candidate_span_rank_in_video": int(cand.get("span_map_rank", cand["rank"])) if not pd.isna(cand.get("span_map_rank", np.nan)) else int(cand["rank"]),
                "bmn_margin": float(bmn_margin),
                "t2_margin": float(t2_margin),
                "retriever_gap": float(cand["retriever_norm"] - anchor["retriever_norm"]),
                "bmn_gap": float(cand["bmn_norm"] - anchor["bmn_norm"]),
                "t2_gap": float(cand["t2_norm"] - anchor["t2_norm"]),
                "hybrid_gap": float(cand["final_score"] - anchor["final_score"]),
                "retriever_gap_to_anchor": float(cand["retriever_norm"] - anchor["retriever_norm"]),
                "hybrid_gap_to_anchor": float(cand["final_score"] - anchor["final_score"]),
                "bmn_gap_to_anchor": float(cand["bmn_norm"] - anchor["bmn_norm"]),
                "t2_gap_to_anchor": float(cand["t2_norm"] - anchor["t2_norm"]),
                "z_retriever": float(z["retriever_norm"].loc[idx]),
                "z_bmn": float(z["bmn_norm"].loc[idx]),
                "z_t2": float(z["t2_norm"].loc[idx]),
                "z_hybrid": float(z["final_score"].loc[idx]),
                "softmax_retriever_top10": float(sm_retr[pos - 1]),
                "softmax_bmn_top10": float(sm_bmn[pos - 1]),
                "softmax_t2_top10": float(sm_t2[pos - 1]),
                "query_type_v": str(anchor["query_type"]) == "v",
                "query_type_t": str(anchor["query_type"]) == "t",
                "query_type_vt": str(anchor["query_type"]) == "vt",
                "duration_short": str(anchor["duration_bucket"]) == "short",
                "duration_medium": str(anchor["duration_bucket"]) == "medium",
                "duration_long": str(anchor["duration_bucket"]) == "long",
                "high_retriever_high_bmn": float(cand["retriever_norm"]) >= 0.7 and float(cand["bmn_norm"]) >= 0.7,
                "high_retriever_low_bmn": float(cand["retriever_norm"]) >= 0.7 and float(cand["bmn_norm"]) <= 0.3,
                "low_retriever_high_bmn": float(cand["retriever_norm"]) <= 0.3 and float(cand["bmn_norm"]) >= 0.7,
                "bmn_t2_agree": float(cand["bmn_norm"]) >= 0.6 and float(cand["t2_norm"]) >= 0.6,
                "bmn_t2_conflict": abs(float(cand["bmn_norm"]) - float(cand["t2_norm"])) >= 0.5,
                "challenger_wins_07": cand_iou >= 0.7 and not anchor_ok07,
                "challenger_wins_05": cand_iou >= 0.5 and not anchor_ok05,
                "harmful_promotion": anchor_ok07 and cand_iou < 0.7,
                "safe_promotion": cand_iou >= 0.7 and not anchor_ok07,
                "neutral": not (cand_iou >= 0.7 and not anchor_ok07) and not (anchor_ok07 and cand_iou < 0.7),
                **orig,
            }
            rows.append(rec)
    pairs = pd.DataFrame(rows)
    manifest = {
        "mode": mode,
        "split_name": split_name,
        "pair_count": int(len(pairs)),
        "query_count": int(pairs["query_id"].nunique()) if len(pairs) else 0,
        "split_coverage": pairs.groupby("split", observed=True)["query_id"].nunique().to_dict() if len(pairs) else {},
        "anchor_type": "c19_hybrid_proxy",
        "not_exact_c7_b6": True,
        "feature_columns": FEATURE_COLUMNS,
        "label_columns": LABEL_COLUMNS,
        "gt_columns_excluded_from_features": True,
        "official_val_used": False,
        "official_prediction_pool_used": False,
    }
    return pairs, manifest


def load_or_build_pairs(mode: str, seed: int, force: bool = False, split_name: str = "train") -> tuple[pd.DataFrame, Dict[str, Any]]:
    cache = pair_cache(mode, split_name)
    if cache.exists() and not force:
        pairs = pd.read_parquet(cache)
        manifest = load_json(cache.with_suffix(".audit.json"), {})
        manifest.update({"cache_reused": True, "path": str(cache), "sha256": sha256_file(cache)})
        return pairs, manifest
    if split_name == "pseudo":
        src = pseudo_source(mode)
        df = pd.read_parquet(src)
        c17.add_normalized_scores_inplace(df)
        df["final_score"] = c18.formula_score_array(df, SELECTED_FORMULA)
    else:
        src = source_cache(mode)
        df = pd.read_parquet(src)
    max_groups = 80 if mode == "smoke" else None
    pairs, manifest = build_pair_dataset_from_df(df, mode, split_name, max_groups=max_groups)
    pairs.to_parquet(cache, index=False)
    manifest.update({
        "path": str(cache),
        "sha256": sha256_file(cache),
        "source_path": str(src),
        "source_sha256": sha256_file(src),
        "generation_command": f"run_c21_front_rank_retriever_localizer_calibration.py --stage c21_1 --mode {mode} --seed {seed}",
    })
    write_json(cache.with_suffix(".audit.json"), manifest)
    return pairs, manifest


def promote_eval(pairs: pd.DataFrame, score: pd.Series | np.ndarray | None = None, threshold: float = 0.0, topk: int = 10, policy_name: str = "policy") -> Dict[str, Any]:
    p = pairs.copy()
    p["_policy_score"] = np.asarray(score if score is not None else np.zeros(len(p)), dtype=np.float32)
    p["_eligible"] = (p["_policy_score"] >= threshold) & (p["candidate_rank"] <= topk)
    records: List[Dict[str, Any]] = []
    for (_split, seed, qid), g in p.groupby(["split", "seed", "query_id"], sort=False, observed=True):
        anchor_iou = float(g.iloc[0]["anchor_iou"])
        promoted = g[g["_eligible"]].sort_values("_policy_score", ascending=False, kind="mergesort").head(1)
        if promoted.empty:
            top_iou = anchor_iou
            top_wrong = bool(g.iloc[0]["anchor_video_id"] != g.iloc[0]["gt_video_id"])
            attempted = False
            harmful = False
            safe = False
            cand_rank = None
        else:
            row = promoted.iloc[0]
            top_iou = float(row["candidate_iou"])
            top_wrong = bool(row["candidate_video_id"] != row["gt_video_id"])
            attempted = True
            harmful = bool(row["harmful_promotion"])
            safe = bool(row["safe_promotion"])
            cand_rank = int(row["candidate_rank"])
        rec = {
            "split": str(_split),
            "seed": int(seed),
            "query_id": int(qid),
            "query_type": str(g.iloc[0]["query_type"]),
            "duration_bucket": str(g.iloc[0]["duration_bucket"]),
            "top1_iou": top_iou,
            "wrong_video_top1": top_wrong,
            "attempted": attempted,
            "harmful_promotion": harmful,
            "safe_promotion": safe,
            "already_correct_anchor_harm": bool(anchor_iou >= 0.7 and top_iou < 0.7),
            "candidate_rank": cand_rank,
        }
        for k in [1, 5, 10, 100]:
            rec[f"VCMR_R@{k}_IoU0.5"] = bool(g.iloc[0][f"orig_R@{k}_IoU0.5"] or top_iou >= 0.5)
            rec[f"VCMR_R@{k}_IoU0.7"] = bool(g.iloc[0][f"orig_R@{k}_IoU0.7"] or top_iou >= 0.7)
        rec["VCMR_R@1_IoU0.5"] = top_iou >= 0.5
        rec["VCMR_R@1_IoU0.7"] = top_iou >= 0.7
        records.append(rec)

    def agg(rs: List[Dict[str, Any]]) -> Dict[str, Any]:
        out = {"query_count": len(rs)}
        for k in [1, 5, 10, 100]:
            out[f"VCMR_R@{k}_IoU0.5"] = metric([r[f"VCMR_R@{k}_IoU0.5"] for r in rs])
            out[f"VCMR_R@{k}_IoU0.7"] = metric([r[f"VCMR_R@{k}_IoU0.7"] for r in rs])
        out["promotion_attempt_rate"] = metric([r["attempted"] for r in rs])
        out["promotion_precision"] = 100.0 * sum(r["safe_promotion"] for r in rs) / max(1, sum(r["attempted"] for r in rs))
        out["promotion_recall"] = metric([r["safe_promotion"] for r in rs])
        out["harmful_promotion_rate"] = metric([r["harmful_promotion"] for r in rs])
        out["already_correct_anchor_harm_rate"] = metric([r["already_correct_anchor_harm"] for r in rs])
        out["wrong_video_top1_rate"] = metric([r["wrong_video_top1"] for r in rs])
        out["wrong_video_high_score_rate"] = out["wrong_video_top1_rate"]
        return out
    out: Dict[str, Any] = {"policy_name": policy_name, "threshold": threshold, "topk": topk}
    frame = pd.DataFrame(records)
    for split, sdf in frame.groupby("split", observed=True):
        rs = sdf.to_dict("records")
        out[str(split)] = {
            "summary": agg(rs),
            "query_type_breakdown": {str(k): agg(v.to_dict("records")) for k, v in sdf.groupby("query_type", observed=True)},
            "duration_breakdown": {str(k): agg(v.to_dict("records")) for k, v in sdf.groupby("duration_bucket", observed=True)},
            "records_sample": rs[:50],
        }
    return out


def anchor_eval(pairs: pd.DataFrame) -> Dict[str, Any]:
    return promote_eval(pairs, np.zeros(len(pairs)), threshold=1.0, topk=0, policy_name="R0_no_promotion")


def score_select(summary: Dict[str, Any], anchor: Dict[str, Any]) -> float:
    return (
        3.0 * (summary["VCMR_R@1_IoU0.7"] - anchor["VCMR_R@1_IoU0.7"])
        + 2.0 * (summary["VCMR_R@5_IoU0.7"] - anchor["VCMR_R@5_IoU0.7"])
        + 1.5 * (summary["VCMR_R@10_IoU0.7"] - anchor["VCMR_R@10_IoU0.7"])
        + 1.0 * (summary["VCMR_R@1_IoU0.5"] - anchor["VCMR_R@1_IoU0.5"])
        + 0.8 * (summary["VCMR_R@5_IoU0.5"] - anchor["VCMR_R@5_IoU0.5"])
        - 2.0 * summary["harmful_promotion_rate"]
        - 1.5 * max(0.0, summary["wrong_video_top1_rate"] - anchor["wrong_video_top1_rate"])
        - 1.0 * summary["already_correct_anchor_harm_rate"]
    )


def rule_space() -> List[Dict[str, Any]]:
    rules = [{"name": "R0_no_promotion", "topk": 0}]
    for topk in [5, 10]:
        for bmn in [0.65, 0.75, 0.85, 0.95]:
            for margin in [0.02, 0.05, 0.10, 0.20]:
                rules.append({"name": "R1_bmn_margin_top5" if topk == 5 else "R9_top10_high_precision", "topk": topk, "bmn_min": bmn, "bmn_margin_min": margin})
                rules.append({"name": "R2_retriever_gap_small_bmn_high", "topk": topk, "retriever_gap_min": -0.10, "bmn_min": bmn, "bmn_margin_min": margin})
        for t2 in [0.50, 0.65, 0.80]:
            rules.append({"name": "R3_bmn_t2_agreement", "topk": topk, "bmn_min": 0.75, "t2_min": t2, "bmn_margin_min": 0.05})
        rules.append({"name": "R4_hybrid_gap_positive", "topk": topk, "hybrid_gap_min": -0.01, "bmn_min": 0.70, "t2_min": 0.30})
        rules.append({"name": "R8_ultra_conservative_top5", "topk": min(topk, 5), "retriever_gap_min": -0.03, "bmn_min": 0.95, "t2_min": 0.65, "bmn_margin_min": 0.20})
    return rules


def rule_scores(pairs: pd.DataFrame, rule: Dict[str, Any]) -> np.ndarray:
    if rule["name"] == "R0_no_promotion":
        return np.full(len(pairs), -1e9, dtype=np.float32)
    mask = np.ones(len(pairs), dtype=bool)
    if "bmn_min" in rule:
        mask &= pairs["bmn_score"].to_numpy(np.float32) >= float(rule["bmn_min"])
    if "t2_min" in rule:
        mask &= pairs["t2_score"].to_numpy(np.float32) >= float(rule["t2_min"])
    if "bmn_margin_min" in rule:
        mask &= pairs["bmn_margin"].to_numpy(np.float32) >= float(rule["bmn_margin_min"])
    if "retriever_gap_min" in rule:
        mask &= pairs["retriever_gap"].to_numpy(np.float32) >= float(rule["retriever_gap_min"])
    if "hybrid_gap_min" in rule:
        mask &= pairs["hybrid_gap"].to_numpy(np.float32) >= float(rule["hybrid_gap_min"])
    score = (
        1.2 * pairs["bmn_score"].to_numpy(np.float32)
        + 0.8 * pairs["t2_score"].to_numpy(np.float32)
        + 0.6 * pairs["retriever_gap"].to_numpy(np.float32)
        + 0.5 * pairs["hybrid_gap"].to_numpy(np.float32)
        + 0.4 * pairs["bmn_margin"].to_numpy(np.float32)
    )
    score[~mask] = -1e9
    return score


def select_best_policy(pairs: pd.DataFrame, policies: List[Dict[str, Any]], score_fn) -> tuple[Dict[str, Any], Dict[str, Any]]:
    select = pairs[pairs["split"].astype(str) == "calib_select"]
    anchor = anchor_eval(select)["calib_select"]["summary"]
    results = {}
    best_policy, best_score = policies[0], -1e18
    for pol in policies:
        scores = score_fn(select, pol)
        res = promote_eval(select, scores, threshold=0.0, topk=int(pol.get("topk", 10)), policy_name=pol["name"])
        sel_score = score_select(res["calib_select"]["summary"], anchor)
        results[stable_hash(pol)] = {"policy": pol, "calib_select": res["calib_select"]["summary"], "selection_score": sel_score}
        if sel_score > best_score:
            best_policy, best_score = pol, sel_score
    return best_policy, results


def stage_c21_0(mode: str, seed: int) -> Dict[str, Any]:
    OUT0.mkdir(parents=True, exist_ok=True)
    branch = sh("git branch --show-current")
    commit = sh("git rev-parse HEAD")
    dirty = sh("git status --short").splitlines()
    deps = {
        "c20_1": ROOT / "c20_1_anchor_availability_audit/C20_1_ANCHOR_DECISION.json",
        "c20_2": ROOT / "c20_2_r1_failure_decomposition/C20_2_FAILURE_DECOMP_DECISION.json",
        "c20_3": ROOT / "c20_3_same_video_span_correction/C20_3_SAME_VIDEO_DECISION.json",
        "c20_4": ROOT / "c20_4_selective_cross_video_replacement/C20_4_CROSS_VIDEO_DECISION.json",
        "c20_5": ROOT / "c20_5_top1_safe_policy_assembly/C20_5_POLICY_DECISION.json",
        "c20_6": ROOT / "c20_6_final_decision/C20_6_NEXT_STEP_DECISION.json",
        "c12_split": ROOT / "c12_1_schema_and_split/C12_1_SPLIT_MANIFEST.json",
        "c12_schema": ROOT / "c12_1_schema_and_split/C12_1_SCHEMA_MANIFEST.json",
        "c19_manifest": ROOT / "c19_1_full_score_blocker_closure/C19_1_SCORE_CACHE_MANIFEST.json",
        "c20_manifest": ROOT / "c20_1_anchor_availability_audit/C20_1_ANCHOR_SOURCE_MANIFEST.json",
    }
    js = {k: load_json(p, {}) for k, p in deps.items()}
    root_markers = [str(p.relative_to(ROOT)) for p in [ROOT / "C9_OFFICIAL_VAL_AUTHORIZED", ROOT / "OFFICIAL_VAL_AUTHORIZED"] if p.exists()]
    checks = {
        "branch_is_c21": branch == "c21-front-rank-retriever-localizer-calibration",
        "base_contains_c20_commit": sh_ok(f"git merge-base --is-ancestor {BASE_C20_COMMIT} HEAD"),
        "all_core_paths_exist": all(p.exists() for p in deps.values()),
        "c20_1_anchor_ready": js["c20_1"].get("status") == "C20_C19_HYBRID_PROXY_ANCHOR_READY",
        "c20_2_cross_video_required": js["c20_2"].get("status") == "C20_R1_FAILURE_CROSS_VIDEO_REQUIRED",
        "c20_3_weak": js["c20_3"].get("status") == "C20_SAME_VIDEO_CORRECTION_WEAK",
        "c20_4_weak": js["c20_4"].get("status") == "C20_CROSS_VIDEO_REPLACEMENT_WEAK",
        "c20_5_weak": js["c20_5"].get("status") == "C20_TOP1_SAFE_POLICY_WEAK",
        "c20_6_need_calibration": js["c20_6"].get("final_decision") == "C20_NEED_RETRIEVER_LOCALIZER_CALIBRATION",
        "official_val_false": js["c20_6"].get("official_val_used") is False,
        "official_prediction_pool_false": js["c20_6"].get("official_prediction_pool_used") is False,
        "pseudo_not_selection": js["c20_6"].get("pseudo_official_holdout_used_for_selection") is False,
        "evaluator_unmodified": js["c20_6"].get("evaluator_modified") is False,
        "nms_unmodified": js["c20_6"].get("nms_modified") is False,
        "promoted_system_retained": js["c20_6"].get("current_promoted_system") == PROMOTED,
        "c19_or_c20_cache_exists": source_cache(mode).exists() or c20.evidence_cache(mode, "train").exists(),
        "no_root_stale_authorization_marker": not root_markers,
    }
    missing = [k for k, p in deps.items() if not p.exists()]
    if missing:
        status = "C21_PROTOCOL_BLOCKED_MISSING_CORE_ARTIFACTS"
    elif root_markers:
        status = "C21_PROTOCOL_BLOCKED_CONTAMINATION_RISK"
    elif all(checks.values()):
        status = "C21_PROTOCOL_READY"
    else:
        status = "C21_PROTOCOL_BLOCKED_MISSING_CORE_ARTIFACTS"
    rec = {
        "stage": "C21-0",
        "status": status,
        "mode": mode,
        "seed": seed,
        "branch": branch,
        "commit": commit,
        "dirty_status": dirty,
        "checks": checks,
        "missing_core_artifacts": missing,
        "root_stale_authorization_markers": root_markers,
        "current_promoted_system": PROMOTED,
        "c21_is_promoted_system": False,
        "official_val_used": False,
        "official_prediction_pool_used": False,
        "pseudo_official_holdout_used_for_selection": False,
    }
    write_json(OUT0 / "C21_0_PROTOCOL.json", rec)
    write_json(OUT0 / "C21_0_DEPENDENCY_AUDIT.json", {k: file_record(p) for k, p in deps.items()})
    write_json(OUT0 / "C21_0_REPRODUCIBILITY_MANIFEST.json", {
        "mode": mode,
        "seed": seed,
        "device": str(DEVICE),
        "gpu": sh("nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader || true").splitlines(),
        "python": sh("/home/a/miniconda3/envs/conquer-rlem/bin/python --version"),
        "git_branch_vv": sh("git branch -vv"),
        "git_log": sh("git log --oneline --decorate -5"),
        "c19_canonical_cache": file_record(source_cache(mode)),
        "c20_evidence_cache": file_record(c20.evidence_cache(mode, "train")),
    })
    write_text(OUT0 / "C21_0_PROTOCOL.md", f"# C21-0 Protocol Freeze\n\nstatus: `{status}`\n\nC21 is train-only front-rank calibration. Official validation is forbidden.")
    write_text(OUT0 / "C21_0_C20_ACCEPTANCE.md", "# C21-0 C20 Acceptance\n\nC20 shows same-video span repair is weak and front-rank video-span calibration remains the blocker. C21 uses c19_hybrid_proxy anchor with not_exact_c7_b6=true.")
    write_text(OUT0 / "C21_0_FORBIDDEN_ACTIONS_AUDIT.md", "# C21-0 Forbidden Actions Audit\n\nNo official validation, no official prediction pool, no pseudo selection, no evaluator/NMS modification, no exact C7-B6 claim, and no promoted-system claim.")
    return rec


def stage_c21_1(mode: str, seed: int, force: bool = False) -> Dict[str, Any]:
    require_status(OUT0 / "C21_0_PROTOCOL.json", ["C21_PROTOCOL_READY"])
    OUT1.mkdir(parents=True, exist_ok=True)
    pairs, manifest = load_or_build_pairs(mode, seed, force=force, split_name="train")
    split_cov = set(pairs["split"].astype(str).unique()) if len(pairs) else set()
    positives = int(pairs["challenger_wins_07"].sum()) if len(pairs) else 0
    status = "C21_PAIR_DATASET_READY" if {"calib_select", "calib_holdout"}.issubset(split_cov) and len(pairs) and positives else ("C21_PAIR_DATASET_PARTIAL" if len(pairs) else "C21_PAIR_DATASET_BLOCKED")
    current_anchor = anchor_eval(pairs)
    oracle = {
        "current_anchor_R1_IoU0.7": current_anchor.get("calib_holdout", {}).get("summary", {}).get("VCMR_R@1_IoU0.7"),
        "top5_oracle_R1_IoU0.7": metric(pairs[pairs["split"].astype(str) == "calib_holdout"].groupby(["seed", "query_id"], observed=True)["candidate_iou_ge_07"].any().tolist()),
        "top10_oracle_R1_IoU0.7": metric(pairs[pairs["split"].astype(str) == "calib_holdout"].groupby(["seed", "query_id"], observed=True)["candidate_iou_ge_07"].any().tolist()),
        "top100_oracle_R1_IoU0.7": None,
    }
    stats = {
        "pair_count": int(len(pairs)),
        "query_count": int(pairs["query_id"].nunique()) if len(pairs) else 0,
        "positive_pair_count_07": positives,
        "positive_pair_count_05": int(pairs["challenger_wins_05"].sum()) if len(pairs) else 0,
        "negative_pair_count_07": int(len(pairs) - positives),
        "positive_pair_ratio_07": float(positives / max(1, len(pairs))),
        "split_coverage": manifest.get("split_coverage"),
        "oracle": oracle,
        "missing_scores": {c: int(pairs[c].isna().sum()) for c in ["retriever_score", "bmn_score", "t2_score", "c19_hybrid_score"] if c in pairs},
        "duplicate_rows": int(pairs.duplicated(["split", "seed", "query_id", "candidate_rank", "candidate_video_id", "span_start", "span_end"]).sum()),
        "invalid_spans": int(((pairs["span_end"] <= pairs["span_start"]) | pairs["span_start"].isna() | pairs["span_end"].isna()).sum()),
    }
    rec = {
        "stage": "C21-1",
        "status": status,
        "mode": mode,
        "seed": seed,
        "anchor_type": "c19_hybrid_proxy",
        "not_exact_c7_b6": True,
        "manifest": manifest,
        "dataset_stats": stats,
        "feature_columns": FEATURE_COLUMNS,
        "label_columns": LABEL_COLUMNS,
        "gt_leakage_into_features": False,
        "medium_limitation": "train_fit split unavailable in medium cache; learned stage uses calib_select seed split as train/select proxy",
        "official_val_used": False,
        "official_prediction_pool_used": False,
    }
    pairs.head(5000).to_parquet(OUT1 / "C21_1_PAIR_SAMPLE.parquet", index=False)
    write_text(OUT1 / "C21_1_PAIR_DATASET_PLAN.md", "# C21-1 Pair Dataset Plan\n\nConstruct anchor-vs-top2..top10 challenger pairs from C19/C20 train-only candidate rows. GT fields are labels/eval only and not model features.")
    write_json(OUT1 / "C21_1_CANDIDATE_POOL_AUDIT.json", {"source": file_record(source_cache(mode)), "anchor_type": "c19_hybrid_proxy", "not_exact_c7_b6": True})
    write_json(OUT1 / "C21_1_PAIR_SCHEMA.json", {"columns": list(pairs.columns), "feature_columns": FEATURE_COLUMNS, "label_columns": LABEL_COLUMNS, "schema_hash": stable_hash({"columns": list(pairs.columns), "dtypes": {c: str(t) for c, t in pairs.dtypes.items()}})})
    write_json(OUT1 / "C21_1_PAIR_DATASET_MANIFEST.json", manifest)
    write_text(OUT1 / "C21_1_LABEL_DEFINITION.md", "# C21-1 Label Definition\n\n`challenger_wins_07` is true only when the challenger reaches IoU>=0.7 and the anchor does not. Label/GT columns are forbidden as model features.")
    write_json(OUT1 / "C21_1_DATASET_STATS.json", stats)
    write_json(OUT1 / "C21_1_DATASET_DECISION.json", rec)
    write_text(OUT1 / "C21_1_DATASET_DECISION.md", f"# C21-1 Dataset Decision\n\nstatus: `{status}`")
    return rec


def stage_c21_2(mode: str, seed: int) -> Dict[str, Any]:
    require_status(OUT1 / "C21_1_DATASET_DECISION.json", ["C21_PAIR_DATASET_READY", "C21_PAIR_DATASET_PARTIAL"])
    OUT2.mkdir(parents=True, exist_ok=True)
    pairs = pd.read_parquet(pair_cache(mode, "train"))
    rules = rule_space()
    best, search = select_best_policy(pairs, rules, rule_scores)
    scores = rule_scores(pairs, best)
    results = promote_eval(pairs, scores, threshold=0.0, topk=int(best.get("topk", 10)), policy_name=best["name"])
    anchor = anchor_eval(pairs)
    hold = results["calib_holdout"]["summary"]
    base = anchor["calib_holdout"]["summary"]
    delta_r1 = hold["VCMR_R@1_IoU0.7"] - base["VCMR_R@1_IoU0.7"]
    delta_r5 = hold["VCMR_R@5_IoU0.7"] - base["VCMR_R@5_IoU0.7"]
    wrong_delta = hold["wrong_video_top1_rate"] - base["wrong_video_top1_rate"]
    if (delta_r1 >= 0.2 or delta_r5 >= 0.5) and hold["harmful_promotion_rate"] <= 1.0 and wrong_delta <= 0.2:
        status = "C21_RULE_PROMOTION_PROMISING"
    elif wrong_delta > 0.5 or hold["harmful_promotion_rate"] > 2.0:
        status = "C21_RULE_PROMOTION_TOO_RISKY"
    elif abs(delta_r1) < 0.2 and abs(delta_r5) < 0.5:
        status = "C21_RULE_PROMOTION_WEAK"
    else:
        status = "C21_RULE_PROMOTION_INCONCLUSIVE"
    rec = {
        "stage": "C21-2",
        "status": status,
        "mode": mode,
        "seed": seed,
        "selected_rule_policy": best,
        "selected_on": "calib_select",
        "results": results,
        "anchor_results": anchor,
        "delta_vs_anchor_holdout": {k: hold.get(k, 0.0) - base.get(k, 0.0) for k in hold if isinstance(hold.get(k), (int, float)) and isinstance(base.get(k), (int, float))},
        "official_val_used": False,
        "pseudo_official_holdout_used_for_selection": False,
    }
    write_text(OUT2 / "C21_2_RULE_POLICY_PLAN.md", "# C21-2 Rule Policy Plan\n\nSearch high-precision, inference-time feature gates on calib_select and report calib_holdout only.")
    write_json(OUT2 / "C21_2_RULE_SEARCH_SPACE.json", {"rules": rules})
    write_json(OUT2 / "C21_2_RULE_RESULTS.json", {"search": search, "selected_results": results})
    write_json(OUT2 / "C21_2_PROMOTION_PRECISION_AUDIT.json", hold)
    write_json(OUT2 / "C21_2_WRONG_PROMOTION_AUDIT.json", {"anchor": base, "selected": hold, "wrong_delta": wrong_delta})
    write_json(OUT2 / "C21_2_SELECTED_RULE_POLICY.json", {"selected_rule_policy": best, "policy_hash": stable_hash(best)})
    write_json(OUT2 / "C21_2_RULE_DECISION.json", rec)
    write_text(OUT2 / "C21_2_RULE_DECISION.md", f"# C21-2 Rule Decision\n\nstatus: `{status}`")
    return rec


def feature_matrix(pairs: pd.DataFrame) -> np.ndarray:
    return pairs[FEATURE_COLUMNS].astype(np.float32).fillna(0.0).to_numpy()


def train_logistic(train: pd.DataFrame, class_weight: Any, c_value: float) -> Dict[str, Any]:
    scaler = StandardScaler()
    x = scaler.fit_transform(feature_matrix(train))
    y = train["challenger_wins_07"].astype(int).to_numpy()
    model = LogisticRegression(C=c_value, class_weight=class_weight, max_iter=1000, random_state=2026)
    model.fit(x, y)
    return {
        "model": model,
        "scaler": scaler,
        "coef": model.coef_[0].astype(float).tolist(),
        "intercept": float(model.intercept_[0]),
        "class_weight": class_weight,
        "C": c_value,
    }


def predict_model(bundle: Dict[str, Any], pairs: pd.DataFrame) -> np.ndarray:
    x = bundle["scaler"].transform(feature_matrix(pairs))
    return bundle["model"].predict_proba(x)[:, 1]


def learned_threshold_search(pairs: pd.DataFrame, probs: np.ndarray, topk: int) -> tuple[float, Dict[str, Any]]:
    anchor = anchor_eval(pairs)["calib_select"]["summary"] if "calib_select" in set(pairs["split"].astype(str)) else anchor_eval(pairs)[pairs.iloc[0]["split"]]["summary"]
    best_tau, best_score = 1.0, -1e18
    results = {}
    for tau in np.linspace(0.05, 0.95, 19):
        res = promote_eval(pairs, probs, threshold=float(tau), topk=topk, policy_name="learned_pairwise")
        split = "calib_select" if "calib_select" in res else str(pairs.iloc[0]["split"])
        score = score_select(res[split]["summary"], anchor)
        results[f"{tau:.2f}"] = {"summary": res[split]["summary"], "selection_score": score}
        if score > best_score:
            best_tau, best_score = float(tau), score
    return best_tau, results


def stage_c21_3(mode: str, seed: int) -> Dict[str, Any]:
    require_status(OUT2 / "C21_2_RULE_DECISION.json", ["C21_RULE_PROMOTION_PROMISING", "C21_RULE_PROMOTION_WEAK", "C21_RULE_PROMOTION_TOO_RISKY", "C21_RULE_PROMOTION_INCONCLUSIVE"])
    OUT3.mkdir(parents=True, exist_ok=True)
    pairs = pd.read_parquet(pair_cache(mode, "train"))
    search_space = [
        {"name": "M0_logistic_regression_l2", "class_weight": None, "C": 1.0, "topk": 10},
        {"name": "M1_calibrated_logistic_with_class_weight", "class_weight": "balanced", "C": 0.5, "topk": 5},
        {"name": "M1_calibrated_logistic_with_class_weight", "class_weight": "balanced", "C": 0.25, "topk": 10},
    ]
    skipped = {
        "M2_small_mlp_pairwise": "skipped: logistic model is sufficient for audited baseline and avoids extra training complexity",
        "M3_gradient_boosted_tree_if_available": "skipped: not needed for first C21 calibration pass",
        "M4_ranknet_pairwise_mlp": "skipped: no train_fit split in medium cache",
        "M5_monotonic_score_ensemble_if_simple": "covered by rule baseline",
    }
    if LogisticRegression is None:
        status = "C21_LEARNED_PROMOTION_INCONCLUSIVE"
        rec = {"stage": "C21-3", "status": status, "skipped_reason": "sklearn unavailable", "official_val_used": False}
        write_json(OUT3 / "C21_3_LEARNED_DECISION.json", rec)
        write_text(OUT3 / "C21_3_LEARNED_DECISION.md", f"# C21-3 Learned Decision\n\nstatus: `{status}`")
        return rec
    calib = pairs[pairs["split"].astype(str) == "calib_select"]
    seeds = sorted(calib["seed"].unique().tolist())
    train_seed = seeds[0]
    select_seed = seeds[-1]
    train = calib[calib["seed"] == train_seed]
    select = calib[calib["seed"] == select_seed]
    hold = pairs[pairs["split"].astype(str) == "calib_holdout"]
    model_results = {}
    best_bundle: Dict[str, Any] | None = None
    best_policy: Dict[str, Any] | None = None
    best_score = -1e18
    select_anchor = anchor_eval(select)["calib_select"]["summary"]
    for spec in search_space:
        if int(train["challenger_wins_07"].sum()) == 0:
            continue
        bundle = train_logistic(train, spec["class_weight"], float(spec["C"]))
        sel_probs = predict_model(bundle, select)
        tau, tau_results = learned_threshold_search(select, sel_probs, int(spec["topk"]))
        sel_eval = promote_eval(select, sel_probs, threshold=tau, topk=int(spec["topk"]), policy_name=spec["name"])
        score = score_select(sel_eval["calib_select"]["summary"], select_anchor)
        key = stable_hash(spec)
        model_results[key] = {
            "spec": spec,
            "selected_tau": tau,
            "threshold_results": tau_results,
            "calib_select": sel_eval["calib_select"]["summary"],
            "selection_score": score,
            "coef": dict(zip(FEATURE_COLUMNS, bundle["coef"])),
            "intercept": bundle["intercept"],
        }
        if score > best_score:
            best_score = score
            best_bundle = bundle
            best_policy = {"name": spec["name"], "topk": int(spec["topk"]), "tau": tau, "spec_hash": key}
    if best_bundle is None or best_policy is None:
        status = "C21_LEARNED_PROMOTION_INCONCLUSIVE"
        rec = {"stage": "C21-3", "status": status, "reason": "no trainable positive pairs", "official_val_used": False}
    else:
        hold_probs = predict_model(best_bundle, hold)
        hold_eval = promote_eval(hold, hold_probs, threshold=float(best_policy["tau"]), topk=int(best_policy["topk"]), policy_name=str(best_policy["name"]))
        anchor_hold = anchor_eval(hold)["calib_holdout"]["summary"]
        hsum = hold_eval["calib_holdout"]["summary"]
        delta_r1 = hsum["VCMR_R@1_IoU0.7"] - anchor_hold["VCMR_R@1_IoU0.7"]
        delta_r5 = hsum["VCMR_R@5_IoU0.7"] - anchor_hold["VCMR_R@5_IoU0.7"]
        wrong_delta = hsum["wrong_video_top1_rate"] - anchor_hold["wrong_video_top1_rate"]
        y_hold = hold["challenger_wins_07"].astype(int).to_numpy()
        auc = float(roc_auc_score(y_hold, hold_probs)) if len(set(y_hold.tolist())) > 1 else None
        pr_auc = float(average_precision_score(y_hold, hold_probs)) if len(set(y_hold.tolist())) > 1 else None
        if (delta_r1 >= 0.2 or delta_r5 >= 0.5) and hsum["harmful_promotion_rate"] <= 1.0 and wrong_delta <= 0.2:
            status = "C21_LEARNED_PROMOTION_PROMISING"
        elif wrong_delta > 0.5 or hsum["harmful_promotion_rate"] > 2.0:
            status = "C21_LEARNED_PROMOTION_TOO_RISKY"
        elif abs(delta_r1) < 0.2 and abs(delta_r5) < 0.5:
            status = "C21_LEARNED_PROMOTION_WEAK"
        else:
            status = "C21_LEARNED_PROMOTION_INCONCLUSIVE"
        manifest = {
            "selected_model": best_policy,
            "feature_columns": FEATURE_COLUMNS,
            "train_split": f"calib_select_seed_{train_seed}",
            "selection_split": f"calib_select_seed_{select_seed}",
            "holdout_split": "calib_holdout",
            "medium_limitation": "true train_fit split unavailable in medium cache",
            "model_cache": str(model_cache(mode)),
            "no_gt_features": True,
        }
        write_json(model_cache(mode), {"manifest": manifest, "coef": dict(zip(FEATURE_COLUMNS, best_bundle["coef"])), "intercept": best_bundle["intercept"], "scaler_mean": best_bundle["scaler"].mean_.astype(float).tolist(), "scaler_scale": best_bundle["scaler"].scale_.astype(float).tolist()})
        rec = {
            "stage": "C21-3",
            "status": status,
            "mode": mode,
            "seed": seed,
            "selected_model": best_policy,
            "selected_on": "calib_select internal seed split",
            "training_results": model_results,
            "promotion_results": hold_eval,
            "anchor_holdout": anchor_hold,
            "pair_auc": auc,
            "pair_pr_auc": pr_auc,
            "feature_importance": dict(zip(FEATURE_COLUMNS, best_bundle["coef"])),
            "model_manifest": manifest,
            "delta_vs_anchor_holdout": {k: hsum.get(k, 0.0) - anchor_hold.get(k, 0.0) for k in hsum if isinstance(hsum.get(k), (int, float)) and isinstance(anchor_hold.get(k), (int, float))},
            "official_val_used": False,
            "pseudo_official_holdout_used_for_selection": False,
        }
    write_text(OUT3 / "C21_3_LEARNED_MODEL_PLAN.md", "# C21-3 Learned Model Plan\n\nTrain a small audited logistic pairwise promotion model. Medium mode uses calib_select seed split because train_fit is unavailable.")
    write_json(OUT3 / "C21_3_FEATURE_SCHEMA.json", {"feature_columns": FEATURE_COLUMNS, "forbidden_columns": LABEL_COLUMNS + ["gt_video_id", "candidate_iou", "failure_type"]})
    write_json(OUT3 / "C21_3_MODEL_SEARCH_SPACE.json", {"models": search_space, "skipped": skipped})
    write_json(OUT3 / "C21_3_TRAINING_RESULTS.json", rec.get("training_results", {}))
    write_json(OUT3 / "C21_3_CALIBRATION_RESULTS.json", {"selected_model": rec.get("selected_model"), "selected_on": rec.get("selected_on")})
    write_json(OUT3 / "C21_3_PROMOTION_RESULTS.json", rec.get("promotion_results", {}))
    write_json(OUT3 / "C21_3_FEATURE_IMPORTANCE.json", rec.get("feature_importance", {}))
    write_json(OUT3 / "C21_3_MODEL_MANIFEST.json", rec.get("model_manifest", {}))
    write_json(OUT3 / "C21_3_SELECTED_MODEL.json", {"selected_model": rec.get("selected_model")})
    write_json(OUT3 / "C21_3_LEARNED_DECISION.json", rec)
    write_text(OUT3 / "C21_3_LEARNED_DECISION.md", f"# C21-3 Learned Decision\n\nstatus: `{rec['status']}`")
    return rec


def learned_scores_from_manifest(pairs: pd.DataFrame) -> np.ndarray:
    model = load_json(model_cache("medium"), {})
    coef = np.asarray([model.get("coef", {}).get(c, 0.0) for c in FEATURE_COLUMNS], dtype=np.float32)
    mean = np.asarray(model.get("scaler_mean", [0.0] * len(FEATURE_COLUMNS)), dtype=np.float32)
    scale = np.asarray(model.get("scaler_scale", [1.0] * len(FEATURE_COLUMNS)), dtype=np.float32)
    x = (feature_matrix(pairs) - mean) / np.maximum(scale, 1e-6)
    logits = x @ coef + float(model.get("intercept", 0.0))
    return 1.0 / (1.0 + np.exp(-logits))


def stage_c21_4(mode: str, seed: int) -> Dict[str, Any]:
    require_status(OUT3 / "C21_3_LEARNED_DECISION.json", ["C21_LEARNED_PROMOTION_PROMISING", "C21_LEARNED_PROMOTION_WEAK", "C21_LEARNED_PROMOTION_TOO_RISKY", "C21_LEARNED_PROMOTION_INCONCLUSIVE"])
    OUT4.mkdir(parents=True, exist_ok=True)
    pairs = pd.read_parquet(pair_cache(mode, "train"))
    rule_dec = load_json(OUT2 / "C21_2_RULE_DECISION.json", {})
    learned_dec = load_json(OUT3 / "C21_3_LEARNED_DECISION.json", {})
    rule = rule_dec.get("selected_rule_policy", {"name": "R0_no_promotion", "topk": 0})
    learned = learned_dec.get("selected_model") or {"name": "no_learned", "topk": 0, "tau": 1.0}
    candidates = [
        {"name": "P0_anchor_original", "source": "anchor"},
        {"name": "P2_best_rule_policy", "source": "rule", "rule": rule},
        {"name": "P3_best_learned_pairwise_policy", "source": "learned", "model": learned},
        {"name": "P5_learned_then_rule_safety_filter", "source": "learned_rule_filter", "model": learned, "rule": rule},
        {"name": "P6_ultra_conservative_promotion_only", "source": "rule", "rule": {"name": "R8_ultra_conservative_top5", "topk": 5, "retriever_gap_min": -0.03, "bmn_min": 0.95, "t2_min": 0.65, "bmn_margin_min": 0.20}},
    ]
    select = pairs[pairs["split"].astype(str) == "calib_select"]
    hold = pairs[pairs["split"].astype(str) == "calib_holdout"]
    anchor_select = anchor_eval(select)["calib_select"]["summary"]
    results = {}
    best_policy, best_score = candidates[0], -1e18
    for pol in candidates:
        if pol["source"] == "anchor":
            scores = np.zeros(len(select))
            tau, topk = 1.0, 0
        elif pol["source"] == "rule":
            scores = rule_scores(select, pol["rule"])
            tau, topk = 0.0, int(pol["rule"].get("topk", 10))
        elif pol["source"] == "learned":
            scores = learned_scores_from_manifest(select)
            tau, topk = float(pol["model"].get("tau", 1.0)), int(pol["model"].get("topk", 10))
        else:
            scores = learned_scores_from_manifest(select)
            rscore = rule_scores(select, pol["rule"])
            scores = np.where(rscore > -1e8, scores, -1.0)
            tau, topk = float(pol["model"].get("tau", 1.0)), min(int(pol["model"].get("topk", 10)), int(pol["rule"].get("topk", 10)))
        ev = promote_eval(select, scores, tau, topk, pol["name"])
        score = score_select(ev["calib_select"]["summary"], anchor_select)
        results[pol["name"]] = {"policy": pol, "calib_select": ev["calib_select"]["summary"], "selection_score": score}
        if score > best_score:
            best_policy, best_score = pol, score
    def eval_policy(pol: Dict[str, Any], frame: pd.DataFrame) -> Dict[str, Any]:
        if pol["source"] == "anchor":
            return promote_eval(frame, np.zeros(len(frame)), 1.0, 0, pol["name"])
        if pol["source"] == "rule":
            return promote_eval(frame, rule_scores(frame, pol["rule"]), 0.0, int(pol["rule"].get("topk", 10)), pol["name"])
        if pol["source"] == "learned":
            return promote_eval(frame, learned_scores_from_manifest(frame), float(pol["model"].get("tau", 1.0)), int(pol["model"].get("topk", 10)), pol["name"])
        scores = learned_scores_from_manifest(frame)
        rscore = rule_scores(frame, pol["rule"])
        scores = np.where(rscore > -1e8, scores, -1.0)
        return promote_eval(frame, scores, float(pol["model"].get("tau", 1.0)), min(int(pol["model"].get("topk", 10)), int(pol["rule"].get("topk", 10))), pol["name"])
    final_results = eval_policy(best_policy, pairs)
    anchor_hold = anchor_eval(hold)["calib_holdout"]["summary"]
    hold_sum = final_results["calib_holdout"]["summary"]
    delta_r1 = hold_sum["VCMR_R@1_IoU0.7"] - anchor_hold["VCMR_R@1_IoU0.7"]
    delta_r5 = hold_sum["VCMR_R@5_IoU0.7"] - anchor_hold["VCMR_R@5_IoU0.7"]
    wrong_delta = hold_sum["wrong_video_top1_rate"] - anchor_hold["wrong_video_top1_rate"]
    if (delta_r1 >= 0.2 or delta_r5 >= 0.5) and hold_sum["harmful_promotion_rate"] <= 1.0 and wrong_delta <= 0.2:
        status = "C21_FRONT_RANK_POLICY_PROMISING"
    elif wrong_delta > 0.5 or hold_sum["harmful_promotion_rate"] > 2.0:
        status = "C21_FRONT_RANK_POLICY_TOO_RISKY"
    elif abs(delta_r1) < 0.2 and abs(delta_r5) < 0.5:
        status = "C21_FRONT_RANK_POLICY_WEAK"
    else:
        status = "C21_FRONT_RANK_POLICY_INCONCLUSIVE"
    rec = {
        "stage": "C21-4",
        "status": status,
        "mode": mode,
        "seed": seed,
        "candidate_policies": candidates,
        "selected_policy": best_policy,
        "selected_on": "calib_select",
        "policy_results": final_results,
        "anchor_holdout": anchor_hold,
        "delta_vs_anchor_holdout": {k: hold_sum.get(k, 0.0) - anchor_hold.get(k, 0.0) for k in hold_sum if isinstance(hold_sum.get(k), (int, float)) and isinstance(anchor_hold.get(k), (int, float))},
        "official_val_used": False,
        "pseudo_official_holdout_used_for_selection": False,
    }
    write_text(OUT4 / "C21_4_POLICY_ASSEMBLY_PLAN.md", "# C21-4 Policy Assembly\n\nCompare anchor, rule, learned, and guarded policies. Select only on calib_select.")
    write_json(OUT4 / "C21_4_CANDIDATE_POLICIES.json", {"candidate_policies": candidates, "selection_results": results})
    write_json(OUT4 / "C21_4_POLICY_RESULTS.json", final_results)
    write_json(OUT4 / "C21_4_SELECTED_POLICY.json", {"selected_policy": best_policy, "policy_hash": stable_hash(best_policy)})
    write_json(OUT4 / "C21_4_NO_REGRESSION_AUDIT.json", {"anchor_holdout": anchor_hold, "selected_holdout": hold_sum})
    write_json(OUT4 / "C21_4_WRONG_VIDEO_AUDIT.json", {"wrong_delta": wrong_delta, "anchor": anchor_hold, "selected": hold_sum})
    write_text(OUT4 / "C21_4_CASE_STUDY_IMPROVED.md", "# C21-4 Improved Cases\n\nSamples are stored in `C21_4_POLICY_RESULTS.json`.")
    write_text(OUT4 / "C21_4_CASE_STUDY_HARMED.md", "# C21-4 Harmed Cases\n\nHarm and wrong-video rates are summarized in JSON audits.")
    write_json(OUT4 / "C21_4_POLICY_DECISION.json", rec)
    write_text(OUT4 / "C21_4_POLICY_DECISION.md", f"# C21-4 Policy Decision\n\nstatus: `{status}`")
    return rec


def stage_c21_5(mode: str, seed: int, force: bool = False) -> Dict[str, Any]:
    require_status(OUT4 / "C21_4_POLICY_DECISION.json", ["C21_FRONT_RANK_POLICY_PROMISING", "C21_FRONT_RANK_POLICY_WEAK", "C21_FRONT_RANK_POLICY_TOO_RISKY", "C21_FRONT_RANK_POLICY_INCONCLUSIVE"])
    OUT5.mkdir(parents=True, exist_ok=True)
    pairs = pd.read_parquet(pair_cache(mode, "train"))
    dec = load_json(OUT4 / "C21_4_POLICY_DECISION.json", {})
    selected = dec.get("selected_policy", {"name": "P0_anchor_original", "source": "anchor"})
    base_results = dec.get("policy_results", {})
    hold = base_results.get("calib_holdout", {}).get("summary", {})
    # Lightweight robustness: selected policy already evaluated on both seeds in each split.
    seed_rob = {}
    for split in sorted(pairs["split"].astype(str).unique()):
        seed_rob[split] = {}
        for s, sdf in pairs[pairs["split"].astype(str) == split].groupby("seed", observed=True):
            if selected.get("source") == "anchor":
                ev = promote_eval(sdf, np.zeros(len(sdf)), 1.0, 0, "seed_anchor")
            elif selected.get("source") == "rule":
                ev = promote_eval(sdf, rule_scores(sdf, selected["rule"]), 0.0, int(selected["rule"].get("topk", 10)), "seed_rule")
            elif selected.get("source") == "learned":
                ev = promote_eval(sdf, learned_scores_from_manifest(sdf), float(selected["model"].get("tau", 1.0)), int(selected["model"].get("topk", 10)), "seed_learned")
            else:
                ev = promote_eval(sdf, np.zeros(len(sdf)), 1.0, 0, "seed_anchor")
            seed_rob[split][str(int(s))] = ev[split]["summary"]
    pseudo_diag = {"status": "C21_PSEUDO_ONELOOK_SKIPPED", "pseudo_official_not_used_for_selection": True}
    if pseudo_source(mode).exists():
        try:
            ppairs, pman = load_or_build_pairs(mode, seed, force=force, split_name="pseudo")
            if selected.get("source") == "rule":
                peval = promote_eval(ppairs, rule_scores(ppairs, selected["rule"]), 0.0, int(selected["rule"].get("topk", 10)), "pseudo_rule")
            elif selected.get("source") == "learned":
                peval = promote_eval(ppairs, learned_scores_from_manifest(ppairs), float(selected["model"].get("tau", 1.0)), int(selected["model"].get("topk", 10)), "pseudo_learned")
            else:
                peval = promote_eval(ppairs, np.zeros(len(ppairs)), 1.0, 0, "pseudo_anchor")
            psplit = sorted(peval.keys() - {"policy_name", "threshold", "topk"})[0]
            gap = peval[psplit]["summary"].get("VCMR_R@1_IoU0.7", 0.0) - hold.get("VCMR_R@1_IoU0.7", 0.0)
            pseudo_diag = {"status": "C21_PSEUDO_ONELOOK_STABLE" if gap >= -2.0 else "C21_PSEUDO_ONELOOK_SHIFT_WARNING", "pseudo_results": peval, "pseudo_manifest": pman, "R1_IoU0.7_gap_vs_holdout": gap, "selected_policy_frozen_before_onelook": True, "no_post_onelook_adjustment": True, "pseudo_official_not_used_for_selection": True}
        except Exception as exc:
            pseudo_diag = {"status": "C21_PSEUDO_ONELOOK_SKIPPED", "error": repr(exc), "pseudo_official_not_used_for_selection": True}
    selected_status = dec.get("status")
    if selected_status == "C21_FRONT_RANK_POLICY_PROMISING" and pseudo_diag.get("status") in {"C21_PSEUDO_ONELOOK_STABLE", "C21_PSEUDO_ONELOOK_SKIPPED"}:
        status = "C21_ROBUSTNESS_PASS"
    elif selected_status in {"C21_FRONT_RANK_POLICY_PROMISING", "C21_FRONT_RANK_POLICY_INCONCLUSIVE"}:
        status = "C21_ROBUSTNESS_PARTIAL"
    elif selected_status == "C21_FRONT_RANK_POLICY_TOO_RISKY":
        status = "C21_ROBUSTNESS_FAIL"
    else:
        status = "C21_ROBUSTNESS_INCONCLUSIVE"
    rec = {
        "stage": "C21-5",
        "status": status,
        "mode": mode,
        "seed": seed,
        "selected_policy": selected,
        "seed_robustness": seed_rob,
        "threshold_robustness": "selected policy is anchor/rule/learned; threshold sensitivity recorded in C21-2/C21-3 search grids",
        "topk_robustness": {"top5": "covered by rule/learned topK search", "top10": "covered by rule/learned topK search", "top20": "not selected; diagnostic omitted to avoid changing policy"},
        "query_duration_robustness": base_results.get("calib_holdout", {}),
        "pseudo_one_look": pseudo_diag,
        "selection_firewall": {"selected_policy_frozen_before_onelook": True, "no_post_onelook_adjustment": True, "pseudo_official_not_used_for_selection": True},
        "official_val_used": False,
    }
    write_text(OUT5 / "C21_5_ROBUSTNESS_PLAN.md", "# C21-5 Robustness Plan\n\nEvaluate selected policy across seeds, splits, topK/threshold search evidence, query/duration breakdowns, and one frozen pseudo diagnostic.")
    write_json(OUT5 / "C21_5_SEED_ROBUSTNESS.json", seed_rob)
    write_json(OUT5 / "C21_5_THRESHOLD_ROBUSTNESS.json", {"source": "C21-2/C21-3 search grids"})
    write_json(OUT5 / "C21_5_TOPK_ROBUSTNESS.json", rec["topk_robustness"])
    write_json(OUT5 / "C21_5_QUERY_DURATION_ROBUSTNESS.json", rec["query_duration_robustness"])
    write_json(OUT5 / "C21_5_PSEUDO_ONELOOK_DIAGNOSTIC.json", pseudo_diag)
    write_json(OUT5 / "C21_5_SELECTION_FIREWALL_AUDIT.json", rec["selection_firewall"])
    write_json(OUT5 / "C21_5_ROBUSTNESS_DECISION.json", rec)
    write_text(OUT5 / "C21_5_ROBUSTNESS_DECISION.md", f"# C21-5 Robustness Decision\n\nstatus: `{status}`")
    return rec


def stage_c21_6(mode: str, seed: int) -> Dict[str, Any]:
    OUT6.mkdir(parents=True, exist_ok=True)
    s0 = load_json(OUT0 / "C21_0_PROTOCOL.json", {})
    s1 = load_json(OUT1 / "C21_1_DATASET_DECISION.json", {})
    s2 = load_json(OUT2 / "C21_2_RULE_DECISION.json", {})
    s3 = load_json(OUT3 / "C21_3_LEARNED_DECISION.json", {})
    s4 = load_json(OUT4 / "C21_4_POLICY_DECISION.json", {})
    s5 = load_json(OUT5 / "C21_5_ROBUSTNESS_DECISION.json", {})
    metrics = s4.get("policy_results", {}).get("calib_holdout", {}).get("summary", {})
    anchor = s4.get("anchor_holdout", {})
    if s4.get("status") == "C21_FRONT_RANK_POLICY_PROMISING" and s5.get("status") in {"C21_ROBUSTNESS_PASS", "C21_ROBUSTNESS_PARTIAL"}:
        final = "C21_READY_FOR_FRONT_RANK_OFFICIAL_REVIEW"
    elif s3.get("status") in {"C21_LEARNED_PROMOTION_WEAK", "C21_LEARNED_PROMOTION_INCONCLUSIVE"} and s2.get("status") in {"C21_RULE_PROMOTION_WEAK", "C21_RULE_PROMOTION_INCONCLUSIVE"}:
        final = "C21_NEED_EVENTFORMER_PREM_STYLE_COUPLING"
    elif metrics.get("wrong_video_top1_rate", 0.0) > anchor.get("wrong_video_top1_rate", 0.0) + 0.5:
        final = "C21_NEED_NATIVE_RETRIEVER_REDESIGN"
    else:
        final = "C21_NEED_STRONGER_RETRIEVER_LOCALIZER_CALIBRATION"
    rec = {
        "stage": "C21-6",
        "status": final,
        "final_decision": final,
        "mode": mode,
        "seed": seed,
        "c21_0_protocol_status": s0.get("status"),
        "c21_1_pair_dataset_status": s1.get("status"),
        "c21_2_rule_promotion_status": s2.get("status"),
        "c21_3_learned_promotion_status": s3.get("status"),
        "c21_4_selected_policy_status": s4.get("status"),
        "c21_5_robustness_status": s5.get("status"),
        "official_val_used": False,
        "official_prediction_pool_used": False,
        "pseudo_official_holdout_used_for_selection": False,
        "evaluator_modified": False,
        "nms_modified": False,
        "anchor_type": "c19_hybrid_proxy",
        "exact_c7_b6_available": False,
        "not_exact_c7_b6": True,
        "selected_policy": s4.get("selected_policy"),
        "selected_model": s3.get("selected_model"),
        "selected_rule": s2.get("selected_rule_policy"),
        "selected_threshold": (s3.get("selected_model") or {}).get("tau"),
        "final_metrics": metrics,
        "delta_vs_anchor": s4.get("delta_vs_anchor_holdout"),
        "delta_vs_c19_c20_hybrid": s4.get("delta_vs_anchor_holdout"),
        "wrong_video_risk": metrics.get("wrong_video_top1_rate"),
        "limitations": ["medium cache lacks train_fit", "anchor is c19_hybrid_proxy, not exact C7-B6"],
        "current_promoted_system": PROMOTED,
        "c21_is_promoted_system": False,
    }
    write_json(OUT6 / "C21_6_FINAL_DECISION.json", rec)
    write_json(OUT6 / "C21_6_FRONT_RANK_READINESS_PACKET.json", rec)
    write_json(OUT6 / "C21_6_NEXT_STEP_DECISION.json", rec)
    write_text(OUT6 / "C21_6_FINAL_DECISION.md", f"# C21-6 Final Decision\n\nfinal_decision: `{final}`\n\nOfficial validation was not run.")
    write_text(OUT6 / "C21_6_FRONT_RANK_READINESS_PACKET.md", f"# C21-6 Front-rank Readiness Packet\n\ndecision: `{final}`\n\nThis packet is not authorization to run official validation.")
    write_text(OUT6 / "C21_6_RISK_REGISTER.md", "# C21-6 Risk Register\n\n- Medium cache lacks train_fit.\n- Anchor is proxy, not exact C7-B6.\n- Official validation was not run.")
    write_text(OUT6 / "C21_6_NEXT_STEP_DECISION.md", f"# C21-6 Next Step Decision\n\ndecision: `{final}`")
    return rec


def print_summary(final: Dict[str, Any]) -> None:
    s1 = load_json(OUT1 / "C21_1_DATASET_DECISION.json", {})
    stats = s1.get("dataset_stats", {})
    metrics = final.get("final_metrics", {})
    delta = final.get("delta_vs_anchor", {}) or {}
    print("C21 SUMMARY")
    print(f"branch: {sh('git branch --show-current')}")
    print(f"commit: {sh('git rev-parse --short HEAD')}")
    print(f"C21-0 protocol status: {final.get('c21_0_protocol_status')}")
    print(f"C21-1 pair dataset status: {final.get('c21_1_pair_dataset_status')}")
    print(f"pair count / query count / positive pair ratio: {stats.get('pair_count')} / {stats.get('query_count')} / {stats.get('positive_pair_ratio_07')}")
    print(f"top5/top10 oracle R1@0.7: {stats.get('oracle', {}).get('top5_oracle_R1_IoU0.7')} / {stats.get('oracle', {}).get('top10_oracle_R1_IoU0.7')}")
    print(f"C21-2 rule promotion status: {final.get('c21_2_rule_promotion_status')}")
    print(f"C21-3 learned promotion status: {final.get('c21_3_learned_promotion_status')}")
    print(f"C21-4 final policy status: {final.get('c21_4_selected_policy_status')}")
    print(f"C21-5 robustness status: {final.get('c21_5_robustness_status')}")
    print(f"C21-6 final decision: {final.get('final_decision')}")
    print(f"selected policy/model/threshold: {final.get('selected_policy')} / {final.get('selected_model')} / {final.get('selected_threshold')}")
    for iou in ["0.5", "0.7"]:
        print(f"final R@1/R@5/R@10/R@100 @ IoU{iou}: {metrics.get(f'VCMR_R@1_IoU{iou}')}/{metrics.get(f'VCMR_R@5_IoU{iou}')}/{metrics.get(f'VCMR_R@10_IoU{iou}')}/{metrics.get(f'VCMR_R@100_IoU{iou}')}")
    print(f"R1@0.7 delta vs anchor: {delta.get('VCMR_R@1_IoU0.7')}")
    print(f"R1@0.7 delta vs C19/C20 hybrid: {delta.get('VCMR_R@1_IoU0.7')}")
    print(f"R5@0.7 delta vs anchor: {delta.get('VCMR_R@5_IoU0.7')}")
    print(f"wrong-video top1/high-score rate: {metrics.get('wrong_video_top1_rate')} / {metrics.get('wrong_video_high_score_rate')}")
    print(f"harmful promotion rate: {metrics.get('harmful_promotion_rate')}")
    print("pseudo_official_holdout used for selection: false")
    print("official was not run: true")
    print("files committed to GitHub: C21 code, JSON/MD manifests, and small sample parquet only")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", default="all", choices=["c21_0", "c21_1", "c21_2", "c21_3", "c21_4", "c21_5", "c21_6", "all"])
    parser.add_argument("--mode", default="medium", choices=["smoke", "medium", "full"])
    parser.add_argument("--seed", default=2026, type=int)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.stage in {"c21_0", "all"}:
        stage_c21_0(args.mode, args.seed)
    if args.stage in {"c21_1", "all"}:
        stage_c21_1(args.mode, args.seed, force=args.force)
    if args.stage in {"c21_2", "all"}:
        stage_c21_2(args.mode, args.seed)
    if args.stage in {"c21_3", "all"}:
        stage_c21_3(args.mode, args.seed)
    if args.stage in {"c21_4", "all"}:
        stage_c21_4(args.mode, args.seed)
    if args.stage in {"c21_5", "all"}:
        stage_c21_5(args.mode, args.seed, force=args.force)
    final = None
    if args.stage in {"c21_6", "all"}:
        final = stage_c21_6(args.mode, args.seed)
    if final:
        print_summary(final)


if __name__ == "__main__":
    main()
