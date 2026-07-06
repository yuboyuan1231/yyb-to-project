#!/usr/bin/env python3
"""C20 top1-safe BMN correction.

Train-only R1-oriented decomposition and policy audit. This script never runs
official validation, never reads the official prediction pool, and never
modifies evaluator or NMS logic.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import subprocess
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

import run_c17_bmn_t2_native_vcmr_integration as c17
import run_c18_full_hybrid_freeze_candidate as c18
import run_c19_official_readiness_blocker_closure as c19
from run_c12_native_retriever_training import DEVICE, ROOT, sha256_file


torch.set_num_threads(min(24, os.cpu_count() or 1))

PROMOTED = "C7-B6 R1SelectiveTop1"
BASE_C19_COMMIT = "d1fad99f15e4f28e19c74331bdf450c74118f9dd"
SELECTED_FORMULA = dict(c19.SELECTED)

OUT0 = ROOT / "c20_0_protocol_freeze"
OUT1 = ROOT / "c20_1_anchor_availability_audit"
OUT2 = ROOT / "c20_2_r1_failure_decomposition"
OUT3 = ROOT / "c20_3_same_video_span_correction"
OUT4 = ROOT / "c20_4_selective_cross_video_replacement"
OUT5 = ROOT / "c20_5_top1_safe_policy_assembly"
OUT6 = ROOT / "c20_6_final_decision"


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
    base = Path(os.environ.get("C20_SCORE_CACHE_DIR", "/tmp/c20_score_cache")) / "CONQUER-RLEM-c2c3"
    base.mkdir(parents=True, exist_ok=True)
    return base


def evidence_cache(mode: str, split_name: str = "train") -> Path:
    return cache_dir() / f"C20_TOP1_EVIDENCE_{split_name}_{mode}.local.parquet"


def source_cache(mode: str) -> Path:
    path = c19.canonical_cache(mode)
    if path.exists():
        return path
    return c19.canonical_cache("medium")


def c18_pseudo_cache(mode: str) -> Path:
    path = c18.c18_pseudo_cache(mode)
    if path.exists():
        return path
    return c18.c18_pseudo_cache("medium")


def require_status(path: Path, ok: Sequence[str]) -> Dict[str, Any]:
    obj = load_json(path, {})
    status = obj.get("status") or obj.get("final_decision")
    if status not in set(ok):
        raise RuntimeError(f"{path} status={status}; expected one of {ok}")
    return obj


def iou_1d(st: float, ed: float, gt_st: float, gt_ed: float) -> float:
    inter = max(0.0, min(ed, gt_ed) - max(st, gt_st))
    union = max(ed, gt_ed) - min(st, gt_st)
    return float(inter / max(union, 1e-6))


def metric(xs: Sequence[Any]) -> float:
    return 100.0 * sum(bool(x) for x in xs) / max(1, len(xs))


def file_record(path: Path, sha: bool = True) -> Dict[str, Any]:
    return {
        "path": str(path),
        "exists": path.exists(),
        "readable": os.access(path, os.R_OK) if path.exists() else False,
        "size_bytes": path.stat().st_size if path.exists() else None,
        "sha256": sha256_file(path) if sha and path.exists() and path.is_file() else None,
    }


def span_tuple(row: pd.Series) -> Tuple[float, float]:
    return (float(row["span_start"]), float(row["span_end"]))


def row_iou(row: pd.Series) -> float:
    if str(row["video_id"]) != str(row["gt_video_id"]):
        return 0.0
    return iou_1d(float(row["span_start"]), float(row["span_end"]), float(row["gt_start"]), float(row["gt_end"]))


def nms_rank_group(g: pd.DataFrame, score_col: str, max_keep: int = 200) -> pd.DataFrame:
    order = g.sort_values(score_col, ascending=False, kind="mergesort")
    kept = []
    by_video: Dict[str, List[Tuple[float, float]]] = {}
    for idx, row in order.iterrows():
        vid = str(row["video_id"])
        span = span_tuple(row)
        prev = by_video.get(vid, [])
        if any(iou_1d(span[0], span[1], p[0], p[1]) > c17.NMS_THRESHOLD for p in prev):
            continue
        by_video.setdefault(vid, []).append(span)
        kept.append(idx)
        if len(kept) >= max_keep:
            break
    ranked = g.loc[kept].copy()
    ranked["rank"] = np.arange(1, len(ranked) + 1, dtype=np.int32)
    return ranked


def best_by_bmn(rows: pd.DataFrame) -> tuple[pd.Series | None, float]:
    if rows.empty:
        return None, 0.0
    ordered = rows.sort_values("bmn_norm", ascending=False, kind="mergesort")
    best = ordered.iloc[0]
    second = float(ordered.iloc[1]["bmn_norm"]) if len(ordered) > 1 else 0.0
    margin = float(best["bmn_norm"]) - second
    return best, margin


def unique_video_rank(ranked: pd.DataFrame, vid: str, limit: int = 200) -> int:
    seen = []
    for v in ranked["video_id"].astype(str).tolist()[:limit]:
        if v not in seen:
            seen.append(v)
        if v == vid:
            return len(seen)
    return 999999


def topk_success(ranked: pd.DataFrame, k: int, thr: float) -> bool:
    for _, row in ranked.head(k).iterrows():
        if row_iou(row) >= thr:
            return True
    return False


def video_pool(ranked: pd.DataFrame, k: int) -> set[str]:
    seen = []
    for v in ranked["video_id"].astype(str).tolist():
        if v not in seen:
            seen.append(v)
        if len(seen) >= k:
            break
    return set(seen)


def candidate_dict(row: pd.Series | None, prefix: str, margin: float = 0.0) -> Dict[str, Any]:
    if row is None:
        return {
            f"{prefix}_exists": False,
            f"{prefix}_video_id": None,
            f"{prefix}_span_start": None,
            f"{prefix}_span_end": None,
            f"{prefix}_iou": 0.0,
            f"{prefix}_retriever_norm": 0.0,
            f"{prefix}_bmn_norm": 0.0,
            f"{prefix}_t2_norm": 0.0,
            f"{prefix}_final_score": 0.0,
            f"{prefix}_margin": 0.0,
            f"{prefix}_retriever_gap_vs_anchor": None,
            f"{prefix}_rank": None,
        }
    return {
        f"{prefix}_exists": True,
        f"{prefix}_video_id": str(row["video_id"]),
        f"{prefix}_span_start": float(row["span_start"]),
        f"{prefix}_span_end": float(row["span_end"]),
        f"{prefix}_iou": row_iou(row),
        f"{prefix}_retriever_norm": float(row.get("retriever_norm", 0.0)),
        f"{prefix}_bmn_norm": float(row.get("bmn_norm", 0.0)),
        f"{prefix}_t2_norm": float(row.get("t2_norm", 0.0)),
        f"{prefix}_final_score": float(row.get("final_score", 0.0)),
        f"{prefix}_margin": float(margin),
        f"{prefix}_rank": int(row.get("rank", 999999)) if not pd.isna(row.get("rank", np.nan)) else None,
    }


def build_evidence_from_df(df: pd.DataFrame, mode: str, split_name: str, max_groups: int | None = None) -> tuple[pd.DataFrame, Dict[str, Any]]:
    needed_norm = {"retriever_norm", "bmn_norm", "t2_norm", "final_score"}
    if not needed_norm.issubset(df.columns):
        c17.add_normalized_scores_inplace(df)
        if "final_score" not in df:
            df["final_score"] = c18.formula_score_array(df, SELECTED_FORMULA)
    if "bmn_score" not in df:
        df["bmn_score"] = df["bmn_final_score"]
    rows: List[Dict[str, Any]] = []
    start = time.time()
    groups = df.groupby(["split", "seed", "query_id"], sort=False, observed=True)
    total = groups.ngroups
    for gi, ((split, seed, qid), g) in enumerate(groups, start=1):
        if max_groups is not None and gi > max_groups:
            break
        if gi == 1 or gi % 250 == 0:
            print(f"[C20 evidence] {split_name} groups={gi}/{total} rows={len(rows)} elapsed={time.time() - start:.1f}s", flush=True)
        ranked = nms_rank_group(g, "final_score", 200)
        if ranked.empty:
            continue
        anchor = ranked.iloc[0]
        gt_vid = str(anchor["gt_video_id"])
        anchor_vid = str(anchor["video_id"])
        anchor_iou = row_iou(anchor)
        anchor_video_rows = g[g["video_id"].astype(str) == anchor_vid]
        gt_video_rows = g[g["video_id"].astype(str) == gt_vid]
        bmn_anchor, bmn_anchor_margin = best_by_bmn(anchor_video_rows)
        bmn_gt, bmn_gt_margin = best_by_bmn(gt_video_rows)
        top5_vids = video_pool(ranked, 5)
        top10_vids = video_pool(ranked, 10)
        cross5_rows = g[g["video_id"].astype(str).isin(top5_vids - {anchor_vid})]
        cross10_rows = g[g["video_id"].astype(str).isin(top10_vids - {anchor_vid})]
        cross5, cross5_margin = best_by_bmn(cross5_rows)
        cross10, cross10_margin = best_by_bmn(cross10_rows)
        gt_rank = unique_video_rank(ranked, gt_vid, 200)
        rec: Dict[str, Any] = {
            "split": str(split),
            "seed": int(seed),
            "query_id": int(qid),
            "query_type": str(anchor["query_type"]),
            "duration_bucket": str(anchor["duration_bucket"]),
            "gt_video_id": gt_vid,
            "gt_start": float(anchor["gt_start"]),
            "gt_end": float(anchor["gt_end"]),
            "anchor_type": "c19_hybrid_proxy",
            "not_exact_c7_b6": True,
            "anchor_video_id": anchor_vid,
            "anchor_span_start": float(anchor["span_start"]),
            "anchor_span_end": float(anchor["span_end"]),
            "anchor_iou": float(anchor_iou),
            "anchor_video_correct": anchor_vid == gt_vid,
            "anchor_correct_iou05": anchor_iou >= 0.5,
            "anchor_correct_iou07": anchor_iou >= 0.7,
            "anchor_score": float(anchor["final_score"]),
            "anchor_retriever_norm": float(anchor.get("retriever_norm", 0.0)),
            "anchor_bmn_norm": float(anchor.get("bmn_norm", 0.0)),
            "anchor_t2_norm": float(anchor.get("t2_norm", 0.0)),
            "anchor_bmn_margin": float(bmn_anchor_margin),
            "gt_video_rank": int(gt_rank),
            "gt_in_top5": gt_rank <= 5,
            "gt_in_top10": gt_rank <= 10,
            "gt_in_top100": gt_rank <= 100,
            "orig_wrong_video_top1": anchor_vid != gt_vid,
            "orig_correct_video_wrong_span_top1": bool(anchor_vid == gt_vid and anchor_iou < 0.5),
            "ranked_candidate_count": int(len(ranked)),
        }
        for k in [1, 5, 10, 100]:
            rec[f"orig_R@{k}_IoU0.5"] = topk_success(ranked, k, 0.5)
            rec[f"orig_R@{k}_IoU0.7"] = topk_success(ranked, k, 0.7)
        rec.update(candidate_dict(bmn_anchor, "bmn_anchor", bmn_anchor_margin))
        rec.update(candidate_dict(bmn_gt, "bmn_gt", bmn_gt_margin))
        rec.update(candidate_dict(cross5, "cross5", cross5_margin))
        rec.update(candidate_dict(cross10, "cross10", cross10_margin))
        if rec["cross5_exists"]:
            rec["cross5_retriever_gap_vs_anchor"] = float(rec["anchor_retriever_norm"] - rec["cross5_retriever_norm"])
        if rec["cross10_exists"]:
            rec["cross10_retriever_gap_vs_anchor"] = float(rec["anchor_retriever_norm"] - rec["cross10_retriever_norm"])
        ranked_correct = ranked.copy()
        ranked_correct["tmp_iou"] = [row_iou(r) for _, r in ranked_correct.iterrows()]
        correct_rows = ranked_correct[ranked_correct["tmp_iou"] >= 0.7]
        rec["hybrid_rank_of_gt_video_span_iou07"] = int(correct_rows.iloc[0]["rank"]) if not correct_rows.empty else 999999
        rows.append(rec)
    ev = pd.DataFrame(rows)
    manifest = {
        "mode": mode,
        "split_name": split_name,
        "row_count": int(len(ev)),
        "groups_seen": int(total),
        "groups_written": int(len(ev)),
        "anchor_type": "c19_hybrid_proxy",
        "not_exact_c7_b6": True,
        "source_formula": SELECTED_FORMULA,
        "official_val_used": False,
        "official_prediction_pool_used": False,
    }
    return ev, manifest


def build_or_load_evidence(mode: str, seed: int, force: bool = False, split_name: str = "train") -> tuple[pd.DataFrame, Dict[str, Any]]:
    cache = evidence_cache(mode, split_name)
    if cache.exists() and not force:
        ev = pd.read_parquet(cache)
        manifest = load_json(cache.with_suffix(".audit.json"), {})
        manifest.update({"cache_reused": True, "path": str(cache), "sha256": sha256_file(cache)})
        return ev, manifest
    if split_name == "pseudo":
        src = c18_pseudo_cache(mode)
        if not src.exists():
            raise FileNotFoundError(src)
        print(f"[C20] loading pseudo diagnostic source {src}", flush=True)
        df = pd.read_parquet(src)
        c17.add_normalized_scores_inplace(df)
        if "final_score" not in df:
            df["final_score"] = c18.formula_score_array(df, SELECTED_FORMULA)
    else:
        src = source_cache(mode)
        if not src.exists():
            raise FileNotFoundError(src)
        print(f"[C20] loading canonical source {src}", flush=True)
        df = pd.read_parquet(src)
    max_groups = 80 if mode == "smoke" else None
    ev, manifest = build_evidence_from_df(df, mode, split_name, max_groups=max_groups)
    ev.to_parquet(cache, index=False)
    manifest.update({
        "path": str(cache),
        "sha256": sha256_file(cache),
        "source_path": str(src),
        "source_sha256": sha256_file(src),
        "generation_command": f"run_c20_top1_safe_bmn_correction.py --stage c20_1 --mode {mode} --seed {seed}",
    })
    write_json(cache.with_suffix(".audit.json"), manifest)
    return ev, manifest


def aggregate_policy(rows: pd.DataFrame, prefix: str = "") -> Dict[str, Any]:
    out: Dict[str, Any] = {"query_count": int(len(rows))}
    for k in [1, 5, 10, 100]:
        out[f"VCMR_R@{k}_IoU0.5"] = metric(rows[f"{prefix}R@{k}_IoU0.5"].tolist())
        out[f"VCMR_R@{k}_IoU0.7"] = metric(rows[f"{prefix}R@{k}_IoU0.7"].tolist())
    out["wrong_video_top1_rate"] = metric(rows[f"{prefix}wrong_video_top1"].tolist())
    out["wrong_video_high_score_rate"] = out["wrong_video_top1_rate"]
    out["correct_video_wrong_span_rate"] = metric(rows[f"{prefix}correct_video_wrong_span_top1"].tolist())
    out["attempt_rate"] = metric(rows.get(f"{prefix}attempted", pd.Series([False] * len(rows))).tolist())
    out["harm_rate"] = metric(rows.get(f"{prefix}harmed_anchor_correct", pd.Series([False] * len(rows))).tolist())
    out["repair_rate"] = metric(rows.get(f"{prefix}repaired_anchor_incorrect", pd.Series([False] * len(rows))).tolist())
    return out


def breakdowns(rows: pd.DataFrame, prefix: str = "") -> Dict[str, Any]:
    return {
        "query_type": {str(k): aggregate_policy(v, prefix) for k, v in rows.groupby("query_type", observed=True)},
        "duration": {str(k): aggregate_policy(v, prefix) for k, v in rows.groupby("duration_bucket", observed=True)},
    }


def apply_top1_policy(ev: pd.DataFrame, same: Dict[str, Any] | None = None, cross: Dict[str, Any] | None = None) -> pd.DataFrame:
    rows = ev.copy()
    same = same or {"name": "S0_anchor_original"}
    cross = cross or {"name": "X0_no_cross_video"}
    new_iou = rows["anchor_iou"].astype(float).copy()
    new_vid = rows["anchor_video_id"].astype(str).copy()
    attempted = pd.Series(False, index=rows.index)
    replacement_kind = pd.Series("anchor", index=rows.index)

    def same_mask(policy: Dict[str, Any]) -> pd.Series:
        name = policy.get("name", "S0_anchor_original")
        if name == "S0_anchor_original":
            return pd.Series(False, index=rows.index)
        if name == "S1_always_bmn_same_video":
            return rows["bmn_anchor_exists"].fillna(False)
        if name == "S2_margin_gated_bmn_same_video":
            return rows["bmn_anchor_exists"].fillna(False) & (rows["bmn_anchor_margin"].astype(float) >= float(policy.get("margin_threshold", 0.0)))
        if name == "S4_score_disagreement_gated":
            return (
                rows["bmn_anchor_exists"].fillna(False)
                & (rows["bmn_anchor_margin"].astype(float) >= float(policy.get("margin_threshold", 0.05)))
                & (rows["bmn_anchor_bmn_norm"].astype(float) >= float(policy.get("bmn_threshold", 0.55)))
                & (rows["bmn_anchor_t2_norm"].astype(float) + float(policy.get("t2_slack", 0.10)) >= rows["anchor_t2_norm"].astype(float))
            )
        if name == "S5_duration_aware_same_video":
            mask = rows["bmn_anchor_exists"].fillna(False).copy()
            for bucket, thr in policy.get("thresholds", {}).items():
                mask &= ~((rows["duration_bucket"].astype(str) == bucket) & (rows["bmn_anchor_margin"].astype(float) < float(thr)))
            return mask
        if name == "S6_query_type_aware_same_video":
            mask = rows["bmn_anchor_exists"].fillna(False).copy()
            for qtype, thr in policy.get("thresholds", {}).items():
                mask &= ~((rows["query_type"].astype(str) == qtype) & (rows["bmn_anchor_margin"].astype(float) < float(thr)))
            return mask
        return pd.Series(False, index=rows.index)

    smask = same_mask(same)
    new_iou.loc[smask] = rows.loc[smask, "bmn_anchor_iou"].astype(float)
    attempted.loc[smask] = True
    replacement_kind.loc[smask] = "same_video"

    def cross_mask(policy: Dict[str, Any]) -> tuple[pd.Series, str]:
        name = policy.get("name", "X0_no_cross_video")
        if name == "X0_no_cross_video":
            return pd.Series(False, index=rows.index), "cross5"
        pref = "cross10" if "top10" in name else "cross5"
        exists = rows[f"{pref}_exists"].fillna(False)
        gap_ok = rows[f"{pref}_retriever_gap_vs_anchor"].fillna(99.0).astype(float) <= float(policy.get("max_retriever_gap", 0.15))
        bmn_ok = rows[f"{pref}_bmn_norm"].astype(float) >= float(policy.get("bmn_threshold", 0.70))
        margin_ok = rows[f"{pref}_margin"].astype(float) >= float(policy.get("margin_threshold", 0.05))
        t2_ok = rows[f"{pref}_t2_norm"].astype(float) >= float(policy.get("t2_threshold", 0.0))
        return exists & gap_ok & bmn_ok & margin_ok & t2_ok, pref

    cmask, cpref = cross_mask(cross)
    new_iou.loc[cmask] = rows.loc[cmask, f"{cpref}_iou"].astype(float)
    new_vid.loc[cmask] = rows.loc[cmask, f"{cpref}_video_id"].astype(str)
    attempted.loc[cmask] = True
    replacement_kind.loc[cmask] = "cross_video"

    rows["policy_top1_iou"] = new_iou
    rows["policy_top1_video_id"] = new_vid
    rows["policy_wrong_video_top1"] = new_vid != rows["gt_video_id"].astype(str)
    rows["policy_correct_video_wrong_span_top1"] = (new_vid == rows["gt_video_id"].astype(str)) & (new_iou < 0.5)
    rows["policy_attempted"] = attempted
    rows["policy_replacement_kind"] = replacement_kind
    rows["policy_harmed_anchor_correct"] = rows["anchor_correct_iou07"] & (new_iou < 0.7)
    rows["policy_repaired_anchor_incorrect"] = (~rows["anchor_correct_iou07"]) & (new_iou >= 0.7)
    for k in [1, 5, 10, 100]:
        rows[f"policy_R@{k}_IoU0.5"] = rows[f"orig_R@{k}_IoU0.5"] | (new_iou >= 0.5)
        rows[f"policy_R@{k}_IoU0.7"] = rows[f"orig_R@{k}_IoU0.7"] | (new_iou >= 0.7)
    rows["policy_R@1_IoU0.5"] = new_iou >= 0.5
    rows["policy_R@1_IoU0.7"] = new_iou >= 0.7
    return rows


def anchor_rows(ev: pd.DataFrame) -> pd.DataFrame:
    rows = ev.copy()
    for k in [1, 5, 10, 100]:
        rows[f"anchor_R@{k}_IoU0.5"] = rows[f"orig_R@{k}_IoU0.5"]
        rows[f"anchor_R@{k}_IoU0.7"] = rows[f"orig_R@{k}_IoU0.7"]
    rows["anchor_wrong_video_top1"] = rows["orig_wrong_video_top1"]
    rows["anchor_correct_video_wrong_span_top1"] = rows["orig_correct_video_wrong_span_top1"]
    rows["anchor_attempted"] = False
    rows["anchor_harmed_anchor_correct"] = False
    rows["anchor_repaired_anchor_incorrect"] = False
    return rows


def policy_score(summary: Dict[str, Any], anchor: Dict[str, Any]) -> float:
    return (
        4.0 * (summary["VCMR_R@1_IoU0.7"] - anchor["VCMR_R@1_IoU0.7"])
        + 1.0 * (summary["VCMR_R@1_IoU0.5"] - anchor["VCMR_R@1_IoU0.5"])
        + 0.5 * (summary["VCMR_R@5_IoU0.7"] - anchor["VCMR_R@5_IoU0.7"])
        - 2.0 * max(0.0, summary["wrong_video_top1_rate"] - anchor["wrong_video_top1_rate"])
        - 3.0 * summary.get("harm_rate", 0.0)
    )


def split_eval(ev: pd.DataFrame, same: Dict[str, Any] | None = None, cross: Dict[str, Any] | None = None) -> Dict[str, Any]:
    out: Dict[str, Any] = {"same_policy": same or {"name": "S0_anchor_original"}, "cross_policy": cross or {"name": "X0_no_cross_video"}}
    for split in sorted(ev["split"].astype(str).unique()):
        sdf = ev[ev["split"].astype(str) == split]
        rows = apply_top1_policy(sdf, same, cross)
        out[split] = {"summary": aggregate_policy(rows, "policy_"), **breakdowns(rows, "policy_"), "records_sample": rows.head(50).to_dict("records")}
    return out


def failure_type(row: pd.Series) -> str:
    if row["anchor_video_correct"] and row["anchor_iou"] >= 0.7:
        return "A"
    if row["anchor_video_correct"] and 0.5 <= row["anchor_iou"] < 0.7:
        return "B"
    if row["anchor_video_correct"] and row["anchor_iou"] < 0.5:
        return "C"
    if (not row["anchor_video_correct"]) and row["gt_in_top5"]:
        return "D"
    if (not row["anchor_video_correct"]) and row["gt_in_top10"]:
        return "E"
    if (not row["anchor_video_correct"]) and row["gt_in_top100"]:
        return "F"
    if not row["gt_in_top100"]:
        return "G"
    return "H"


def stage_c20_0(mode: str, seed: int) -> Dict[str, Any]:
    OUT0.mkdir(parents=True, exist_ok=True)
    branch = sh("git branch --show-current")
    commit = sh("git rev-parse HEAD")
    dirty = sh("git status --short").splitlines()
    deps = {
        "c19_1": ROOT / "c19_1_full_score_blocker_closure/C19_1_FULL_SCORE_DECISION.json",
        "c19_2": ROOT / "c19_2_baseline_replay_closure/C19_2_BASELINE_REPLAY_DECISION.json",
        "c19_3": ROOT / "c19_3_front_rank_risk_audit/C19_3_RISK_DECISION.json",
        "c19_4": ROOT / "c19_4_readiness_packet_assembly/C19_4_READINESS_ASSEMBLY_DECISION.json",
        "c19_5": ROOT / "c19_5_final_blocker_decision/C19_5_NEXT_STEP_DECISION.json",
        "c12_split": ROOT / "c12_1_schema_and_split/C12_1_SPLIT_MANIFEST.json",
        "c12_schema": ROOT / "c12_1_schema_and_split/C12_1_SCHEMA_MANIFEST.json",
        "c19_manifest": ROOT / "c19_1_full_score_blocker_closure/C19_1_SCORE_CACHE_MANIFEST.json",
    }
    js = {k: load_json(p, {}) for k, p in deps.items()}
    root_markers = [str(p.relative_to(ROOT)) for p in [ROOT / "C9_OFFICIAL_VAL_AUTHORIZED", ROOT / "OFFICIAL_VAL_AUTHORIZED"] if p.exists()]
    checks = {
        "branch_is_c20": branch == "c20-top1-safe-bmn-correction",
        "base_contains_c19_commit": sh_ok(f"git merge-base --is-ancestor {BASE_C19_COMMIT} HEAD"),
        "all_core_paths_exist": all(p.exists() for p in deps.values()),
        "c19_1_partial_acceptable": js["c19_1"].get("status") == "C19_FULL_SCORE_PARTIAL_ACCEPTABLE",
        "c19_2_partial_acceptable": js["c19_2"].get("status") == "C19_BASELINE_REPLAY_PARTIAL_ACCEPTABLE",
        "c19_3_warning": js["c19_3"].get("status") == "C19_FRONT_RANK_RISK_WARNING",
        "c19_5_continue": js["c19_5"].get("final_decision") == "C19_CONTINUE_TRAIN_ONLY_BLOCKER_CLOSURE",
        "official_val_false": js["c19_5"].get("official_val_used") is False,
        "official_prediction_pool_false": js["c19_5"].get("official_prediction_pool_used") is False,
        "pseudo_not_selection": js["c19_5"].get("pseudo_official_holdout_used_for_selection") is False,
        "evaluator_unmodified": js["c19_5"].get("evaluator_modified") is False,
        "nms_unmodified": js["c19_5"].get("nms_modified") is False,
        "promoted_system_retained": js["c19_5"].get("current_promoted_system") == PROMOTED,
        "c19_local_cache_exists": source_cache(mode).exists(),
        "no_root_stale_authorization_marker": not root_markers,
    }
    missing = [k for k, p in deps.items() if not p.exists()]
    if missing:
        status = "C20_PROTOCOL_BLOCKED_MISSING_CORE_ARTIFACTS"
    elif root_markers:
        status = "C20_PROTOCOL_BLOCKED_CONTAMINATION_RISK"
    elif all(checks.values()):
        status = "C20_PROTOCOL_READY"
    else:
        status = "C20_PROTOCOL_BLOCKED_MISSING_CORE_ARTIFACTS"
    rec = {
        "stage": "C20-0",
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
        "c20_is_promoted_system": False,
        "official_val_used": False,
        "official_prediction_pool_used": False,
        "pseudo_official_holdout_used_for_selection": False,
    }
    write_json(OUT0 / "C20_0_PROTOCOL.json", rec)
    write_json(OUT0 / "C20_0_DEPENDENCY_AUDIT.json", {k: file_record(p) for k, p in deps.items()})
    write_json(OUT0 / "C20_0_REPRODUCIBILITY_MANIFEST.json", {
        "mode": mode,
        "seed": seed,
        "device": str(DEVICE),
        "gpu": sh("nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader || true").splitlines(),
        "python": sh("/home/a/miniconda3/envs/conquer-rlem/bin/python --version"),
        "git_branch_vv": sh("git branch -vv"),
        "git_log": sh("git log --oneline --decorate -5"),
        "c19_canonical_cache": file_record(source_cache(mode)),
        "c19_score_cache_dir": str(source_cache(mode).parent),
    })
    write_text(OUT0 / "C20_0_PROTOCOL.md", f"# C20-0 Protocol Freeze\n\nstatus: `{status}`\n\nC20 is top1-safe train-only correction. Official validation is forbidden.")
    write_text(OUT0 / "C20_0_C19_ACCEPTANCE.md", "# C20-0 C19 Acceptance\n\nC19 is accepted as evidence that R@100 improved while top1/front-rank risk remains. C20 uses C19 selected hybrid only as train-only proxy anchor unless exact C7-B6 train-only rows are found.")
    write_text(OUT0 / "C20_0_FORBIDDEN_ACTIONS_AUDIT.md", "# C20-0 Forbidden Actions Audit\n\nNo official validation, no official prediction pool, no pseudo selection, no evaluator/NMS modification, no C20 promoted-system claim, and no exact C7-B6 anchor claim without compatible train-only rows.")
    return rec


def search_anchor_sources() -> Dict[str, Any]:
    patterns = ("c7", "c7b6", "zero_delta", "first_stage", "conquer", "c19", "baseline")
    hits: List[Dict[str, Any]] = []
    compatible = []
    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        rel = str(path.relative_to(ROOT))
        lower = rel.lower()
        if not any(p in lower for p in patterns):
            continue
        if path.stat().st_size > 100_000_000:
            continue
        official_like = any(tok in lower for tok in ["official_prediction", "official_val", "official_one_shot", "fixed_pool"])
        rec = {
            "path": rel,
            "exists": True,
            "readable": os.access(path, os.R_OK),
            "size_bytes": path.stat().st_size,
            "official_contamination_risk": official_like,
            "contains_required_columns": False,
            "schema_compatibility": "official_like_excluded" if official_like else "not_checked",
            "can_be_converted_safely": False,
            "rejection_reason": None,
        }
        if path.suffix == ".parquet" and not official_like:
            try:
                cols = list(pd.read_parquet(path).columns)
                req = {"split", "seed", "query_id", "video_id", "span_start", "span_end"}
                rec.update({"columns": cols, "contains_required_columns": req.issubset(cols)})
                if req.issubset(cols):
                    rec["schema_compatibility"] = "c17_c18_c19_row_schema"
                    rec["can_be_converted_safely"] = True
                    compatible.append(rec)
                else:
                    rec["schema_compatibility"] = "parquet_missing_required_columns"
                    rec["rejection_reason"] = "missing C17/C18/C19 row schema columns"
            except Exception as exc:
                rec["read_error"] = repr(exc)
        elif path.suffix == ".json" and not official_like:
            try:
                obj = load_json(path, {})
                if isinstance(obj, dict) and "VCMR" in obj and "video2idx" in obj:
                    rec["schema_compatibility"] = "legacy_vcmr_prediction_format"
                    rec["rejection_reason"] = "legacy prediction format lacks split/seed/source score columns"
                elif isinstance(obj, list) and obj and isinstance(obj[0], dict):
                    cols = list(obj[0].keys())
                    req = {"split", "seed", "query_id", "video_id", "span_start", "span_end"}
                    rec.update({"columns": cols, "contains_required_columns": req.issubset(cols)})
                    if req.issubset(cols):
                        rec["schema_compatibility"] = "potential_row_json"
                        rec["can_be_converted_safely"] = True
                        compatible.append(rec)
                    else:
                        rec["schema_compatibility"] = "json_missing_required_columns"
                else:
                    rec["schema_compatibility"] = "manifest_or_metrics_not_rows"
            except Exception as exc:
                rec["read_error"] = repr(exc)
        elif official_like:
            rec["rejection_reason"] = "official-like artifact excluded by protocol"
        else:
            rec["rejection_reason"] = "not a row source"
        hits.append(rec)
    hits = sorted(hits, key=lambda x: (x["official_contamination_risk"], x["path"]))[:400]
    exact = [h for h in compatible if "c7" in h["path"].lower() and not h["official_contamination_risk"]]
    return {
        "exact_c7_b6_candidates": exact,
        "compatible_candidates": compatible,
        "recorded_hits": hits,
        "hit_count_recorded": len(hits),
        "official_prediction_pool_used": False,
    }


def stage_c20_1(mode: str, seed: int, force: bool = False) -> Dict[str, Any]:
    require_status(OUT0 / "C20_0_PROTOCOL.json", ["C20_PROTOCOL_READY"])
    OUT1.mkdir(parents=True, exist_ok=True)
    search = search_anchor_sources()
    ev, manifest = build_or_load_evidence(mode, seed, force=force, split_name="train")
    exact = bool(search["exact_c7_b6_candidates"])
    if exact:
        status = "C20_EXACT_C7_B6_ANCHOR_READY"
        anchor_type = "exact_c7_b6_train_only_anchor"
        not_exact = False
    elif len(ev):
        status = "C20_C19_HYBRID_PROXY_ANCHOR_READY"
        anchor_type = "c19_hybrid_anchor"
        not_exact = True
    else:
        status = "C20_ANCHOR_UNAVAILABLE"
        anchor_type = "no_usable_anchor"
        not_exact = True
    sample_cols = [c for c in ev.columns if c in {
        "split", "seed", "query_id", "query_type", "duration_bucket", "anchor_type", "anchor_video_id",
        "anchor_span_start", "anchor_span_end", "anchor_iou", "anchor_correct_iou07", "gt_video_rank",
        "bmn_anchor_iou", "bmn_anchor_margin", "cross5_iou", "cross10_iou",
    }]
    ev[sample_cols].head(5000).to_parquet(OUT1 / "C20_1_ANCHOR_SAMPLE.parquet", index=False)
    rec = {
        "stage": "C20-1",
        "status": status,
        "mode": mode,
        "seed": seed,
        "anchor_type": anchor_type,
        "not_exact_c7_b6": not_exact,
        "exact_c7_b6_anchored": not not_exact,
        "anchor_manifest": manifest,
        "c7_b6_anchor_search": search,
        "official_val_used": False,
        "official_prediction_pool_used": False,
    }
    write_text(OUT1 / "C20_1_ANCHOR_AUDIT_PLAN.md", "# C20-1 Anchor Audit Plan\n\nSearch exact C7-B6 train-only rows first. If absent, use clearly marked train-only proxy anchor. Official prediction pool is forbidden.")
    write_json(OUT1 / "C20_1_C7_B6_ANCHOR_SEARCH_AUDIT.json", search)
    write_json(OUT1 / "C20_1_ANCHOR_ROW_SCHEMA_AUDIT.json", {"anchor_columns": list(ev.columns), "row_count": int(len(ev)), "schema_hash": stable_hash({"columns": list(ev.columns), "dtypes": {c: str(t) for c, t in ev.dtypes.items()}})})
    write_json(OUT1 / "C20_1_ANCHOR_SOURCE_MANIFEST.json", manifest)
    write_json(OUT1 / "C20_1_ANCHOR_DECISION.json", rec)
    write_text(OUT1 / "C20_1_ANCHOR_DECISION.md", f"# C20-1 Anchor Decision\n\nstatus: `{status}`\n\nanchor_type: `{anchor_type}`\n\nnot_exact_c7_b6: `{str(not_exact).lower()}`")
    return rec


def stage_c20_2(mode: str, seed: int) -> Dict[str, Any]:
    s1 = require_status(OUT1 / "C20_1_ANCHOR_DECISION.json", ["C20_EXACT_C7_B6_ANCHOR_READY", "C20_CONQUER_PROXY_ANCHOR_READY", "C20_C19_HYBRID_PROXY_ANCHOR_READY"])
    OUT2.mkdir(parents=True, exist_ok=True)
    ev = pd.read_parquet(evidence_cache(mode, "train"))
    ev["failure_type"] = [failure_type(r) for _, r in ev.iterrows()]
    counts = ev["failure_type"].value_counts().sort_index().to_dict()
    pct = {k: 100.0 * v / max(1, len(ev)) for k, v in counts.items()}
    anchor_eval = aggregate_policy(anchor_rows(ev), "anchor_")
    same_upper = float(((ev["failure_type"].isin(["B", "C"])) & (ev["bmn_anchor_iou"] >= 0.7)).sum()) * 100.0 / max(1, len(ev))
    cross_upper = float(((ev["failure_type"].isin(["D", "E"])) & ((ev["cross5_iou"] >= 0.7) | (ev["cross10_iou"] >= 0.7))).sum()) * 100.0 / max(1, len(ev))
    total_upper = anchor_eval["VCMR_R@1_IoU0.7"] + same_upper + cross_upper
    unrecoverable = pct.get("G", 0.0)
    by_type = {}
    for typ, g in ev.groupby("failure_type", observed=True):
        by_type[str(typ)] = {
            "query_count": int(len(g)),
            "percentage": 100.0 * len(g) / max(1, len(ev)),
            "current_R1_IoU0.7_contribution": 100.0 * int(g["anchor_correct_iou07"].sum()) / max(1, len(ev)),
            "bmn_same_video_repair_upper_bound_count": int((g["bmn_anchor_iou"] >= 0.7).sum()),
            "cross_video_repair_upper_bound_count": int(((g["cross5_iou"] >= 0.7) | (g["cross10_iou"] >= 0.7)).sum()),
            "wrong_video_rate": metric(g["orig_wrong_video_top1"].tolist()),
        }
    if (pct.get("B", 0.0) + pct.get("C", 0.0)) >= (pct.get("D", 0.0) + pct.get("E", 0.0) + pct.get("F", 0.0)) and same_upper > 0.5:
        status = "C20_R1_FAILURE_SPAN_REPAIR_PROMISING"
    elif (pct.get("D", 0.0) + pct.get("E", 0.0)) >= 10.0:
        status = "C20_R1_FAILURE_CROSS_VIDEO_REQUIRED"
    elif unrecoverable >= 25.0:
        status = "C20_R1_FAILURE_RETRIEVER_LIMITED"
    else:
        status = "C20_R1_FAILURE_INCONCLUSIVE"
    rec = {
        "stage": "C20-2",
        "status": status,
        "mode": mode,
        "seed": seed,
        "anchor_type": s1.get("anchor_type"),
        "not_exact_c7_b6": s1.get("not_exact_c7_b6"),
        "failure_type_counts": counts,
        "failure_type_percentages": pct,
        "failure_type_metrics": by_type,
        "current_anchor_R1_IoU0.7": anchor_eval["VCMR_R@1_IoU0.7"],
        "same_video_repair_upper_bound": same_upper,
        "safe_cross_video_upper_bound": cross_upper,
        "total_recoverable_upper_bound": min(100.0, total_upper),
        "unrecoverable_ratio": unrecoverable,
        "breakdowns": {
            "query_type": ev.groupby(["failure_type", "query_type"], observed=True).size().to_dict(),
            "duration": ev.groupby(["failure_type", "duration_bucket"], observed=True).size().to_dict(),
            "video_rank": ev.groupby("failure_type", observed=True)["gt_video_rank"].describe().to_dict(),
        },
        "official_val_used": False,
        "official_prediction_pool_used": False,
    }
    sample_cols = ["split", "seed", "query_id", "failure_type", "query_type", "duration_bucket", "anchor_video_id", "gt_video_id", "anchor_iou", "gt_video_rank", "bmn_anchor_iou", "bmn_gt_iou", "cross5_iou", "cross10_iou"]
    ev[sample_cols].head(5000).to_parquet(OUT2 / "C20_2_FAILURE_SAMPLE.parquet", index=False)
    write_text(OUT2 / "C20_2_FAILURE_DECOMP_PLAN.md", "# C20-2 R1 Failure Decomposition\n\nClassify A-H failure types using the selected train-only anchor. Same-video and cross-video upper bounds are diagnostic, not official metrics.")
    write_json(OUT2 / "C20_2_FAILURE_TYPE_COUNTS.json", {"counts": counts, "percentages": pct})
    write_json(OUT2 / "C20_2_FAILURE_TYPE_METRICS.json", by_type)
    write_json(OUT2 / "C20_2_BMN_REPAIR_POTENTIAL.json", {"same_video_repair_upper_bound": same_upper, "total_recoverable_upper_bound": total_upper})
    write_json(OUT2 / "C20_2_RETRIEVER_RECOVERABILITY.json", {"safe_cross_video_upper_bound": cross_upper, "unrecoverable_ratio": unrecoverable})
    write_text(OUT2 / "C20_2_FAILURE_EXAMPLES.md", "# C20-2 Failure Examples\n\nRepresentative rows are stored in `C20_2_FAILURE_SAMPLE.parquet`.")
    write_json(OUT2 / "C20_2_FAILURE_DECOMP_DECISION.json", rec)
    write_text(OUT2 / "C20_2_FAILURE_DECOMP_DECISION.md", f"# C20-2 Failure Decomposition Decision\n\nstatus: `{status}`")
    return rec


def same_policy_space() -> List[Dict[str, Any]]:
    policies = [{"name": "S0_anchor_original"}, {"name": "S1_always_bmn_same_video"}]
    for thr in [0.0, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30]:
        policies.append({"name": "S2_margin_gated_bmn_same_video", "margin_threshold": thr})
    for thr in [0.02, 0.05, 0.10, 0.15]:
        for bmn in [0.45, 0.55, 0.65, 0.75]:
            policies.append({"name": "S4_score_disagreement_gated", "margin_threshold": thr, "bmn_threshold": bmn, "t2_slack": 0.10})
    policies.extend([
        {"name": "S5_duration_aware_same_video", "thresholds": {"short": 0.05, "medium": 0.10, "long": 0.15}},
        {"name": "S5_duration_aware_same_video", "thresholds": {"short": 0.02, "medium": 0.05, "long": 0.10}},
        {"name": "S6_query_type_aware_same_video", "thresholds": {"v": 0.05, "t": 0.10, "vt": 0.05}},
        {"name": "S6_query_type_aware_same_video", "thresholds": {"v": 0.02, "t": 0.05, "vt": 0.02}},
    ])
    return policies


def choose_policy(ev: pd.DataFrame, policies: List[Dict[str, Any]], cross: Dict[str, Any] | None = None) -> tuple[Dict[str, Any], Dict[str, Any]]:
    select = ev[ev["split"].astype(str) == "calib_select"]
    anchor_summary = aggregate_policy(anchor_rows(select), "anchor_")
    best, best_score = policies[0], -1e18
    results = {}
    for pol in policies:
        rows = apply_top1_policy(select, pol, cross)
        summary = aggregate_policy(rows, "policy_")
        score = policy_score(summary, anchor_summary)
        results[stable_hash(pol)] = {"policy": pol, "calib_select": summary, "selection_score": score}
        if score > best_score:
            best, best_score = pol, score
    return best, results


def stage_c20_3(mode: str, seed: int) -> Dict[str, Any]:
    require_status(OUT2 / "C20_2_FAILURE_DECOMP_DECISION.json", ["C20_R1_FAILURE_SPAN_REPAIR_PROMISING", "C20_R1_FAILURE_CROSS_VIDEO_REQUIRED", "C20_R1_FAILURE_RETRIEVER_LIMITED", "C20_R1_FAILURE_INCONCLUSIVE"])
    OUT3.mkdir(parents=True, exist_ok=True)
    ev = pd.read_parquet(evidence_cache(mode, "train"))
    policies = same_policy_space()
    selected, select_results = choose_policy(ev, policies)
    results = split_eval(ev, selected, {"name": "X0_no_cross_video"})
    anchor_hold = aggregate_policy(anchor_rows(ev[ev["split"].astype(str) == "calib_holdout"]), "anchor_")
    hold = results["calib_holdout"]["summary"]
    delta = hold["VCMR_R@1_IoU0.7"] - anchor_hold["VCMR_R@1_IoU0.7"]
    wrong_delta = hold["wrong_video_top1_rate"] - anchor_hold["wrong_video_top1_rate"]
    if delta >= 0.5 and hold["harm_rate"] <= 1.0 and wrong_delta <= 0.01:
        status = "C20_SAME_VIDEO_CORRECTION_PROMISING"
    elif delta < -0.1 or hold["harm_rate"] > 3.0:
        status = "C20_SAME_VIDEO_CORRECTION_HARMFUL"
    elif abs(delta) < 0.5:
        status = "C20_SAME_VIDEO_CORRECTION_WEAK"
    else:
        status = "C20_SAME_VIDEO_CORRECTION_INCONCLUSIVE"
    rec = {
        "stage": "C20-3",
        "status": status,
        "mode": mode,
        "seed": seed,
        "selected_policy": selected,
        "selected_on": "calib_select",
        "selected_policy_hash": stable_hash(selected),
        "results": results,
        "anchor_holdout": anchor_hold,
        "R1_IoU0.7_delta_vs_anchor_holdout": delta,
        "wrong_video_delta_vs_anchor_holdout": wrong_delta,
        "pseudo_official_holdout_used_for_selection": False,
        "official_val_used": False,
    }
    write_text(OUT3 / "C20_3_SAME_VIDEO_POLICY_PLAN.md", "# C20-3 Same-video Policy Plan\n\nKeep anchor top1 video fixed and optionally replace the span using inference-time BMN/T2/retriever features. Selection is calib_select only.")
    write_json(OUT3 / "C20_3_POLICY_SEARCH_SPACE.json", {"policies": policies})
    write_json(OUT3 / "C20_3_SAME_VIDEO_RESULTS.json", {"search": select_results, "selected_results": results})
    write_json(OUT3 / "C20_3_NO_REGRESSION_AUDIT.json", {"anchor_holdout": anchor_hold, "selected_holdout": hold, "R1_IoU0.7_delta": delta, "wrong_video_delta": wrong_delta})
    write_text(OUT3 / "C20_3_REPAIR_CASES.md", "# C20-3 Repair Cases\n\nDetailed selected-policy samples are in `C20_3_SAME_VIDEO_RESULTS.json`.")
    write_text(OUT3 / "C20_3_HARM_CASES.md", "# C20-3 Harm Cases\n\nHarm is measured as already R1@0.7-correct anchor queries that become incorrect after replacement.")
    write_json(OUT3 / "C20_3_SELECTED_POLICY.json", {"selected_policy": selected, "selected_policy_hash": stable_hash(selected), "gt_dependent": False})
    write_json(OUT3 / "C20_3_SAME_VIDEO_DECISION.json", rec)
    write_text(OUT3 / "C20_3_SAME_VIDEO_DECISION.md", f"# C20-3 Same-video Decision\n\nstatus: `{status}`\n\nSelected policy `{selected.get('name')}` on calib_select.")
    return rec


def cross_policy_space() -> List[Dict[str, Any]]:
    policies = [{"name": "X0_no_cross_video"}]
    for bmn in [0.70, 0.80, 0.90, 0.95]:
        for margin in [0.05, 0.10, 0.20, 0.30]:
            policies.append({"name": "X1_top5_high_bmn_margin", "bmn_threshold": bmn, "margin_threshold": margin, "max_retriever_gap": 0.20, "t2_threshold": 0.0})
            policies.append({"name": "X2_top5_retriever_gap_small", "bmn_threshold": bmn, "margin_threshold": margin, "max_retriever_gap": 0.05, "t2_threshold": 0.30})
    for bmn in [0.90, 0.95, 0.98]:
        policies.append({"name": "X3_top10_ultra_conservative", "bmn_threshold": bmn, "margin_threshold": 0.20, "max_retriever_gap": 0.05, "t2_threshold": 0.50})
        policies.append({"name": "X4_t2_agreement_required", "bmn_threshold": bmn, "margin_threshold": 0.15, "max_retriever_gap": 0.10, "t2_threshold": 0.70})
    return policies


def stage_c20_4(mode: str, seed: int) -> Dict[str, Any]:
    require_status(OUT3 / "C20_3_SAME_VIDEO_DECISION.json", ["C20_SAME_VIDEO_CORRECTION_PROMISING", "C20_SAME_VIDEO_CORRECTION_WEAK", "C20_SAME_VIDEO_CORRECTION_HARMFUL", "C20_SAME_VIDEO_CORRECTION_INCONCLUSIVE"])
    OUT4.mkdir(parents=True, exist_ok=True)
    ev = pd.read_parquet(evidence_cache(mode, "train"))
    same = load_json(OUT3 / "C20_3_SELECTED_POLICY.json", {}).get("selected_policy", {"name": "S0_anchor_original"})
    policies = cross_policy_space()
    selected, search = choose_policy(ev, policies, cross=None)
    # choose_policy treats policies as same policies, so evaluate cross policies explicitly.
    select = ev[ev["split"].astype(str) == "calib_select"]
    anchor_summary = aggregate_policy(apply_top1_policy(select, same, {"name": "X0_no_cross_video"}), "policy_")
    selected, best_score = policies[0], -1e18
    search = {}
    for pol in policies:
        rows = apply_top1_policy(select, same, pol)
        summary = aggregate_policy(rows, "policy_")
        score = policy_score(summary, anchor_summary)
        search[stable_hash(pol)] = {"policy": pol, "calib_select": summary, "selection_score": score}
        if score > best_score:
            selected, best_score = pol, score
    results = split_eval(ev, same, selected)
    same_results = split_eval(ev, same, {"name": "X0_no_cross_video"})
    hold = results["calib_holdout"]["summary"]
    same_hold = same_results["calib_holdout"]["summary"]
    delta = hold["VCMR_R@1_IoU0.7"] - same_hold["VCMR_R@1_IoU0.7"]
    wrong_delta = hold["wrong_video_top1_rate"] - same_hold["wrong_video_top1_rate"]
    if selected.get("name") != "X0_no_cross_video" and delta >= 0.5 and wrong_delta <= 0.2 and hold["harm_rate"] <= 1.0:
        status = "C20_CROSS_VIDEO_REPLACEMENT_PROMISING"
    elif selected.get("name") != "X0_no_cross_video" and (wrong_delta > 0.2 or hold["harm_rate"] > 2.0):
        status = "C20_CROSS_VIDEO_REPLACEMENT_TOO_RISKY"
    elif delta < 0.5:
        status = "C20_CROSS_VIDEO_REPLACEMENT_WEAK"
    else:
        status = "C20_CROSS_VIDEO_REPLACEMENT_INCONCLUSIVE"
    rec = {
        "stage": "C20-4",
        "status": status,
        "mode": mode,
        "seed": seed,
        "same_video_policy": same,
        "selected_cross_policy": selected,
        "selected_on": "calib_select",
        "results": results,
        "same_video_baseline_results": same_results,
        "R1_IoU0.7_delta_vs_same_video_holdout": delta,
        "wrong_video_delta_vs_same_video_holdout": wrong_delta,
        "official_val_used": False,
        "pseudo_official_holdout_used_for_selection": False,
    }
    write_text(OUT4 / "C20_4_CROSS_VIDEO_POLICY_PLAN.md", "# C20-4 Cross-video Policy Plan\n\nAllow top5/top10 cross-video replacement only under strict inference-time feature gates selected on calib_select.")
    write_json(OUT4 / "C20_4_CROSS_VIDEO_SEARCH_SPACE.json", {"policies": policies})
    write_json(OUT4 / "C20_4_CROSS_VIDEO_RESULTS.json", {"search": search, "selected_results": results})
    write_json(OUT4 / "C20_4_WRONG_SWAP_AUDIT.json", {"same_holdout": same_hold, "selected_holdout": hold, "R1_delta": delta, "wrong_video_delta": wrong_delta})
    write_text(OUT4 / "C20_4_REPLACEMENT_CASES.md", "# C20-4 Replacement Cases\n\nSelected-policy samples are stored in `C20_4_CROSS_VIDEO_RESULTS.json`.")
    write_text(OUT4 / "C20_4_HARM_CASES.md", "# C20-4 Harm Cases\n\nWrong swaps and already-correct harms are summarized in `C20_4_WRONG_SWAP_AUDIT.json`.")
    write_json(OUT4 / "C20_4_SELECTED_POLICY.json", {"selected_policy": selected, "selected_policy_hash": stable_hash(selected), "gt_dependent": False})
    write_json(OUT4 / "C20_4_CROSS_VIDEO_DECISION.json", rec)
    write_text(OUT4 / "C20_4_CROSS_VIDEO_DECISION.md", f"# C20-4 Cross-video Decision\n\nstatus: `{status}`\n\nSelected policy `{selected.get('name')}` on calib_select.")
    return rec


def stage_c20_5(mode: str, seed: int, force: bool = False) -> Dict[str, Any]:
    require_status(OUT4 / "C20_4_CROSS_VIDEO_DECISION.json", ["C20_CROSS_VIDEO_REPLACEMENT_PROMISING", "C20_CROSS_VIDEO_REPLACEMENT_TOO_RISKY", "C20_CROSS_VIDEO_REPLACEMENT_WEAK", "C20_CROSS_VIDEO_REPLACEMENT_INCONCLUSIVE"])
    OUT5.mkdir(parents=True, exist_ok=True)
    ev = pd.read_parquet(evidence_cache(mode, "train"))
    same = load_json(OUT3 / "C20_3_SELECTED_POLICY.json", {}).get("selected_policy", {"name": "S0_anchor_original"})
    cross_dec = load_json(OUT4 / "C20_4_CROSS_VIDEO_DECISION.json", {})
    cross_selected = cross_dec.get("selected_cross_policy", {"name": "X0_no_cross_video"})
    cross = cross_selected if cross_dec.get("status") == "C20_CROSS_VIDEO_REPLACEMENT_PROMISING" else {"name": "X0_no_cross_video"}
    final_results = split_eval(ev, same, cross)
    same_results = split_eval(ev, same, {"name": "X0_no_cross_video"})
    anchor_hold = aggregate_policy(anchor_rows(ev[ev["split"].astype(str) == "calib_holdout"]), "anchor_")
    t2 = load_json(ROOT / "c19_2_baseline_replay_closure/C19_2_BASELINE_COMPARISON_TABLE.json", {}).get("C12_5T_T2", {}).get("calib_holdout", {}).get("summary", {})
    c19_hybrid = load_json(ROOT / "c19_2_baseline_replay_closure/C19_2_BASELINE_COMPARISON_TABLE.json", {}).get("C18_selected_hybrid", {}).get("calib_holdout", {}).get("summary", {})
    hold = final_results["calib_holdout"]["summary"]
    pseudo_diag: Dict[str, Any] = {
        "status": "C20_PSEUDO_ONELOOK_SKIPPED",
        "reason": "pseudo diagnostic cache unavailable",
        "pseudo_official_holdout_used_for_selection": False,
    }
    pseudo_path = c18_pseudo_cache(mode)
    if pseudo_path.exists():
        try:
            pev, pmanifest = build_or_load_evidence(mode, seed, force=force, split_name="pseudo")
            pres = split_eval(pev, same, cross)
            pseudo_summary = pres.get("pseudo_official_holdout", pres.get("calib_holdout", {})).get("summary", {})
            gap = pseudo_summary.get("VCMR_R@1_IoU0.7", 0.0) - hold.get("VCMR_R@1_IoU0.7", 0.0)
            pseudo_diag = {
                "status": "C20_PSEUDO_ONELOOK_STABLE" if gap >= -2.0 else "C20_PSEUDO_ONELOOK_SHIFT_WARNING",
                "policy_frozen_before_pseudo": True,
                "no_post_pseudo_adjustment": True,
                "pseudo_official_holdout_used_for_selection": False,
                "pseudo_results": pres,
                "pseudo_manifest": pmanifest,
                "R1_IoU0.7_gap_vs_holdout": gap,
            }
        except Exception as exc:
            pseudo_diag = {"status": "C20_PSEUDO_ONELOOK_SKIPPED", "error": repr(exc), "pseudo_official_holdout_used_for_selection": False}
    delta_anchor = hold["VCMR_R@1_IoU0.7"] - anchor_hold["VCMR_R@1_IoU0.7"]
    delta_c19 = hold["VCMR_R@1_IoU0.7"] - c19_hybrid.get("VCMR_R@1_IoU0.7", hold["VCMR_R@1_IoU0.7"])
    if delta_anchor >= 0.5 and delta_c19 >= 0.0 and hold["harm_rate"] <= 1.0 and pseudo_diag.get("status") in {"C20_PSEUDO_ONELOOK_STABLE", "C20_PSEUDO_ONELOOK_SKIPPED"}:
        status = "C20_TOP1_SAFE_POLICY_PROMISING"
    elif hold["wrong_video_top1_rate"] > anchor_hold["wrong_video_top1_rate"] + 0.5 or hold["harm_rate"] > 2.0:
        status = "C20_TOP1_SAFE_POLICY_TOO_RISKY"
    elif abs(delta_anchor) < 0.5:
        status = "C20_TOP1_SAFE_POLICY_WEAK"
    else:
        status = "C20_TOP1_SAFE_POLICY_INCONCLUSIVE"
    rec = {
        "stage": "C20-5",
        "status": status,
        "mode": mode,
        "seed": seed,
        "final_policy": {"anchor": "c19_hybrid_proxy", "same_video_policy": same, "cross_video_policy": cross, "cross_video_policy_source": cross_dec.get("status")},
        "final_results": final_results,
        "comparison": {
            "anchor_original": anchor_hold,
            "c19_hybrid_selected_formula": c19_hybrid,
            "same_video_bmn_correction": same_results["calib_holdout"]["summary"],
            "cross_video_replacement_only": cross_dec.get("results", {}).get("calib_holdout", {}).get("summary", {}),
            "same_video_plus_selective_cross_video": hold,
            "c12_5t_t2": t2,
            "oracle_same_video_repair_upper_bound": load_json(OUT2 / "C20_2_FAILURE_DECOMP_DECISION.json", {}).get("same_video_repair_upper_bound"),
            "oracle_safe_cross_video_upper_bound": load_json(OUT2 / "C20_2_FAILURE_DECOMP_DECISION.json", {}).get("safe_cross_video_upper_bound"),
        },
        "R1_IoU0.7_delta_vs_anchor": delta_anchor,
        "R1_IoU0.7_delta_vs_C19_hybrid": delta_c19,
        "R1_IoU0.7_delta_vs_C12_5T_T2": hold["VCMR_R@1_IoU0.7"] - t2.get("VCMR_R@1_IoU0.7", hold["VCMR_R@1_IoU0.7"]),
        "pseudo_one_look": pseudo_diag,
        "policy_frozen_before_pseudo": True,
        "no_post_pseudo_adjustment": True,
        "pseudo_official_holdout_used_for_selection": False,
        "official_val_used": False,
    }
    write_text(OUT5 / "C20_5_POLICY_ASSEMBLY_PLAN.md", "# C20-5 Top1-safe Policy Assembly\n\nAssemble frozen anchor preservation, same-video BMN correction, and optional cross-video replacement. Pseudo one-look is diagnostic only after policy freeze.")
    write_json(OUT5 / "C20_5_FINAL_POLICY.json", rec["final_policy"])
    write_json(OUT5 / "C20_5_FINAL_POLICY_RESULTS.json", rec["final_results"])
    write_json(OUT5 / "C20_5_ROBUSTNESS_AUDIT.json", {"calib_select": rec["final_results"].get("calib_select", {}).get("summary"), "calib_holdout": hold, "gap_R1_IoU0.7": hold.get("VCMR_R@1_IoU0.7", 0.0) - rec["final_results"].get("calib_select", {}).get("summary", {}).get("VCMR_R@1_IoU0.7", 0.0)})
    write_json(OUT5 / "C20_5_FRONT_RANK_METRICS.json", hold)
    write_json(OUT5 / "C20_5_WRONG_VIDEO_AUDIT.json", {"anchor": anchor_hold, "final": hold})
    write_json(OUT5 / "C20_5_PSEUDO_ONELOOK_DIAGNOSTIC.json", pseudo_diag)
    write_json(OUT5 / "C20_5_POLICY_DECISION.json", rec)
    write_text(OUT5 / "C20_5_POLICY_DECISION.md", f"# C20-5 Policy Decision\n\nstatus: `{status}`\n\nFinal policy is frozen before pseudo diagnostics and does not use official validation.")
    return rec


def stage_c20_6(mode: str, seed: int) -> Dict[str, Any]:
    OUT6.mkdir(parents=True, exist_ok=True)
    s0 = load_json(OUT0 / "C20_0_PROTOCOL.json", {})
    s1 = load_json(OUT1 / "C20_1_ANCHOR_DECISION.json", {})
    s2 = load_json(OUT2 / "C20_2_FAILURE_DECOMP_DECISION.json", {})
    s3 = load_json(OUT3 / "C20_3_SAME_VIDEO_DECISION.json", {})
    s4 = load_json(OUT4 / "C20_4_CROSS_VIDEO_DECISION.json", {})
    s5 = load_json(OUT5 / "C20_5_POLICY_DECISION.json", {})
    final_metrics = s5.get("final_results", {}).get("calib_holdout", {}).get("summary", {})
    if s5.get("status") == "C20_TOP1_SAFE_POLICY_PROMISING":
        final = "C20_READY_FOR_TOP1_ORIENTED_OFFICIAL_REVIEW"
    elif s2.get("status") in {"C20_R1_FAILURE_CROSS_VIDEO_REQUIRED", "C20_R1_FAILURE_RETRIEVER_LIMITED"} and s5.get("status") != "C20_TOP1_SAFE_POLICY_PROMISING":
        final = "C20_NEED_RETRIEVER_LOCALIZER_CALIBRATION"
    elif s3.get("status") == "C20_SAME_VIDEO_CORRECTION_PROMISING" and s4.get("status") in {"C20_CROSS_VIDEO_REPLACEMENT_TOO_RISKY", "C20_CROSS_VIDEO_REPLACEMENT_WEAK"}:
        final = "C20_CONTINUE_TOP1_SAFE_TRAIN_ONLY"
    elif s5.get("status") in {"C20_TOP1_SAFE_POLICY_WEAK", "C20_TOP1_SAFE_POLICY_INCONCLUSIVE"}:
        final = "C20_NEED_EVENTFORMER_PREM_STYLE_COUPLING"
    else:
        final = "C20_CONTINUE_TOP1_SAFE_TRAIN_ONLY"
    limitations = []
    if s1.get("not_exact_c7_b6"):
        limitations.append("anchor is proxy, not exact C7-B6")
    if s5.get("status") != "C20_TOP1_SAFE_POLICY_PROMISING":
        limitations.append("final top1-safe policy is not promising enough for official review")
    rec = {
        "stage": "C20-6",
        "status": final,
        "final_decision": final,
        "mode": mode,
        "seed": seed,
        "c20_0_protocol_status": s0.get("status"),
        "c20_1_anchor_status": s1.get("status"),
        "c20_1_anchor_type": s1.get("anchor_type"),
        "c20_2_failure_decomposition_status": s2.get("status"),
        "c20_3_same_video_correction_status": s3.get("status"),
        "c20_4_cross_video_replacement_status": s4.get("status"),
        "c20_5_final_policy_status": s5.get("status"),
        "official_val_used": False,
        "official_prediction_pool_used": False,
        "pseudo_official_holdout_used_for_selection": False,
        "evaluator_modified": False,
        "nms_modified": False,
        "selected_anchor_type": s1.get("anchor_type"),
        "exact_c7_b6_anchored": s1.get("exact_c7_b6_anchored", False),
        "final_policy": s5.get("final_policy"),
        "final_metrics": final_metrics,
        "R1_IoU0.7_gain_vs_anchor": s5.get("R1_IoU0.7_delta_vs_anchor"),
        "R1_IoU0.7_gain_vs_C19_hybrid": s5.get("R1_IoU0.7_delta_vs_C19_hybrid"),
        "wrong_video_risk": final_metrics.get("wrong_video_top1_rate"),
        "limitations": limitations,
        "current_promoted_system": PROMOTED,
        "c20_is_promoted_system": False,
    }
    write_json(OUT6 / "C20_6_FINAL_DECISION.json", rec)
    write_json(OUT6 / "C20_6_TOP1_READINESS_PACKET.json", rec)
    write_json(OUT6 / "C20_6_NEXT_STEP_DECISION.json", rec)
    write_text(OUT6 / "C20_6_FINAL_DECISION.md", f"# C20-6 Final Decision\n\nfinal_decision: `{final}`\n\nOfficial validation was not run.")
    write_text(OUT6 / "C20_6_TOP1_READINESS_PACKET.md", f"# C20-6 Top1 Readiness Packet\n\ndecision: `{final}`\n\nThis packet is not authorization to run official validation.")
    write_text(OUT6 / "C20_6_RISK_REGISTER.md", "\n".join(["# C20-6 Risk Register", "", *[f"- {x}" for x in limitations or ["no major limitations recorded"]]]))
    write_text(OUT6 / "C20_6_NEXT_STEP_DECISION.md", f"# C20-6 Next Step Decision\n\ndecision: `{final}`")
    return rec


def print_summary(final: Dict[str, Any]) -> None:
    s2 = load_json(OUT2 / "C20_2_FAILURE_DECOMP_DECISION.json", {})
    metrics = final.get("final_metrics", {})
    print("C20 SUMMARY")
    print(f"branch: {sh('git branch --show-current')}")
    print(f"commit: {sh('git rev-parse --short HEAD')}")
    print(f"C20-0 protocol status: {final.get('c20_0_protocol_status')}")
    print(f"C20-1 anchor status/type: {final.get('c20_1_anchor_status')} / {final.get('selected_anchor_type')}")
    print(f"C20-2 failure decomposition status: {final.get('c20_2_failure_decomposition_status')}")
    print(f"Type A-H failure counts: {s2.get('failure_type_counts')}")
    print(f"same-video repair upper bound: {s2.get('same_video_repair_upper_bound')}")
    print(f"cross-video repair upper bound: {s2.get('safe_cross_video_upper_bound')}")
    print(f"C20-3 same-video correction status: {final.get('c20_3_same_video_correction_status')}")
    print(f"C20-4 cross-video replacement status: {final.get('c20_4_cross_video_replacement_status')}")
    print(f"C20-5 final policy status: {final.get('c20_5_final_policy_status')}")
    print(f"C20-6 final decision: {final.get('final_decision')}")
    for iou in ["0.5", "0.7"]:
        print(f"final R@1/R@5/R@10/R@100 @ IoU{iou}: {metrics.get(f'VCMR_R@1_IoU{iou}')}/{metrics.get(f'VCMR_R@5_IoU{iou}')}/{metrics.get(f'VCMR_R@10_IoU{iou}')}/{metrics.get(f'VCMR_R@100_IoU{iou}')}")
    print(f"R1@0.7 delta vs anchor original: {final.get('R1_IoU0.7_gain_vs_anchor')}")
    print(f"R1@0.7 delta vs C19 hybrid: {final.get('R1_IoU0.7_gain_vs_C19_hybrid')}")
    print(f"wrong-video top1/high-score rate: {metrics.get('wrong_video_top1_rate')} / {metrics.get('wrong_video_high_score_rate')}")
    print("pseudo_official_holdout used for selection: false")
    print("official was not run: true")
    print("files committed to GitHub: C20 code, JSON/MD manifests, and small sample parquet only")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", default="all", choices=["c20_0", "c20_1", "c20_2", "c20_3", "c20_4", "c20_5", "c20_6", "all"])
    parser.add_argument("--mode", default="medium", choices=["smoke", "medium", "full"])
    parser.add_argument("--seed", default=2026, type=int)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.stage in {"c20_0", "all"}:
        stage_c20_0(args.mode, args.seed)
    if args.stage in {"c20_1", "all"}:
        stage_c20_1(args.mode, args.seed, force=args.force)
    if args.stage in {"c20_2", "all"}:
        stage_c20_2(args.mode, args.seed)
    if args.stage in {"c20_3", "all"}:
        stage_c20_3(args.mode, args.seed)
    if args.stage in {"c20_4", "all"}:
        stage_c20_4(args.mode, args.seed)
    if args.stage in {"c20_5", "all"}:
        stage_c20_5(args.mode, args.seed, force=args.force)
    final = None
    if args.stage in {"c20_6", "all"}:
        final = stage_c20_6(args.mode, args.seed)
    if final:
        print_summary(final)


if __name__ == "__main__":
    main()
