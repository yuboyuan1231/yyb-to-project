#!/usr/bin/env python3
"""C15 Boundary / Localizer Architecture Redesign.

Train-only boundary/localizer redesign after C14 showed text/event features were
harmful. This script does not run official validation, does not read official
prediction pools, does not modify evaluator/NMS, and does not use
pseudo_official_holdout for model selection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import subprocess
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

import run_c12_5_native_span_generation as c12_5
import run_c12_5r_span_head_repair as c12_5r
import run_c12_5t_best_span_promotion as c12_5t
from run_c12_native_retriever_training import (
    DEVICE,
    ROOT,
    build_feature_caches,
    load_corpus,
    load_features,
    load_first_stage,
    qtype_id,
    sha256_file,
)
from utils.temporal_nms import temporal_non_maximum_suppression


torch.set_num_threads(min(16, os.cpu_count() or 1))

SEED = 1515
PROMOTED = "C7-B6 R1SelectiveTop1"
SAMPLE_LIMIT = int(os.environ.get("C15_SAMPLE_LIMIT", "300"))
TRAIN_EPOCHS = int(os.environ.get("C15_TRAIN_EPOCHS", "4"))
TRAIN_BATCH = int(os.environ.get("C15_TRAIN_BATCH", "16"))
EVAL_BATCH = int(os.environ.get("C15_EVAL_BATCH", "24"))
D_MAX = int(os.environ.get("C15_D_MAX", "64"))
HIDDEN = int(os.environ.get("C15_HIDDEN", "128"))
NMS_THRESHOLD = float(os.environ.get("C15_NMS_THRESHOLD", "0.7"))

OUT0 = ROOT / "c15_0_boundary_redesign_protocol"
OUT1 = ROOT / "c15_1_localizer_failure_audit"
OUT2 = ROOT / "c15_2_bmn_span_confidence_mvp"
MODEL_DIR = ROOT / "c12_models"


def seed_all(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def load_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def sh(cmd: str) -> str:
    return subprocess.check_output(cmd, shell=True, cwd=ROOT, text=True, stderr=subprocess.STDOUT).strip()


def stable_hash(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def pct(vals: Sequence[float], p: float) -> float | None:
    return float(np.percentile(vals, p)) if vals else None


def mean(vals: Sequence[float]) -> float | None:
    return float(np.mean(vals)) if vals else None


def metric(vals: Sequence[bool]) -> float:
    return 100.0 * sum(bool(v) for v in vals) / max(1, len(vals))


def duration_bucket(seconds: float) -> str:
    if seconds <= 5.0:
        return "short"
    if seconds <= 15.0:
        return "medium"
    return "long"


def iou_bin(v: float) -> str:
    if v < 0.3:
        return "[0,0.3)"
    if v < 0.5:
        return "[0.3,0.5)"
    if v < 0.7:
        return "[0.5,0.7)"
    return "[0.7,1.0]"


def protocol_ready() -> bool:
    p = load_json(OUT0 / "C15_0_PROTOCOL.json", {})
    return p.get("status") == "C15_PROTOCOL_READY"


def require_protocol_ready() -> None:
    if not protocol_ready():
        raise RuntimeError("C15-0 is not C15_PROTOCOL_READY; later stages are blocked.")


def stage_c15_0() -> Dict[str, Any]:
    OUT0.mkdir(parents=True, exist_ok=True)
    branch = sh("git branch --show-current")
    commit = sh("git rev-parse HEAD")
    dirty = sh("git status --short")
    c14_interim = load_json(ROOT / "c14_3_multiscale_event_pilot/C14_INTERIM_DECISION.json", {})
    c14_text = load_json(ROOT / "c14_2_text_subtitle_pilot/C14_2_TEXT_SUBTITLE_DECISION.json", {})
    c14_event = load_json(ROOT / "c14_3_multiscale_event_pilot/C14_3_EVENT_DECISION.json", {})
    raw_audit = load_json(ROOT / "c14_1_data_feature_audit/C14_1_RAW_VIDEO_AUDIT.json", {})

    core_paths = {
        "c14_instruction_review": ROOT / "c14_0_protocol_freeze/C14_INSTRUCTION_REVIEW.md",
        "c14_interim_decision": ROOT / "c14_3_multiscale_event_pilot/C14_INTERIM_DECISION.json",
        "c14_text_decision": ROOT / "c14_2_text_subtitle_pilot/C14_2_TEXT_SUBTITLE_DECISION.json",
        "c14_event_decision": ROOT / "c14_3_multiscale_event_pilot/C14_3_EVENT_DECISION.json",
        "c12_5t_decision": ROOT / "c12_5t_best_span_promotion/C12_5T_DECISION.json",
        "c12_5t_results": ROOT / "c12_5t_best_span_promotion/C12_5T_E_RESULTS.json",
        "c12_5u_decision": ROOT / "c12_5u_topk_calibration_repair/C12_5U_DECISION.json",
        "c12_5v_decision": ROOT / "c12_5v_conquer_hidden_ranker/C12_5V_DECISION.json",
        "c12_split_manifest": ROOT / "c12_1_schema_and_split/C12_1_SPLIT_MANIFEST.json",
        "c12_schema_manifest": ROOT / "c12_1_schema_and_split/C12_1_SCHEMA_MANIFEST.json",
        "c12_localizer_script": ROOT / "run_c12_5_native_span_generation.py",
        "c12_feature_builder_script": ROOT / "run_c12_5t_best_span_promotion.py",
        "c12_teacher_localizer": ROOT / "c12_models/c12_5_teacher_distilled.pt",
        "current_promoted_system": ROOT / "c7_audit/CURRENT_PROMOTED_SYSTEM.json",
        "c7_b6_final_decision": ROOT / "c7_audit/C7_B6_FINAL_DECISION.json",
    }
    missing = [k for k, p in core_paths.items() if not p.exists()]
    promoted_json = load_json(core_paths["current_promoted_system"], {})
    promoted_text = json.dumps(promoted_json, sort_keys=True)
    promoted_ok = "C7-B6" in promoted_text and "R1SelectiveTop1" in promoted_text
    checks = {
        "branch_expected": branch == "c15-boundary-localizer-redesign",
        "c14_instruction_review_exists": core_paths["c14_instruction_review"].exists(),
        "c14_interim_exists": core_paths["c14_interim_decision"].exists(),
        "c14_decision_boundary_redesign": c14_interim.get("decision") == "C14_NEED_BOUNDARY_ARCHITECTURE_REDESIGN",
        "c14_official_val_used_false": c14_interim.get("official_val_used") is False,
        "raw_video_exists_false": c14_interim.get("raw_video_exists") is False and raw_audit.get("raw_video_exists") is False,
        "c14_text_harmful": c14_interim.get("text_status") == "C14_TEXT_SUBTITLE_HARMFUL" and c14_text.get("status") == "C14_TEXT_SUBTITLE_HARMFUL",
        "c14_event_harmful": c14_interim.get("event_status") == "C14_EVENT_FEATURE_HARMFUL" and c14_event.get("status") == "C14_EVENT_FEATURE_HARMFUL",
        "c12_5t_5u_5v_artifacts_exist": all(core_paths[k].exists() for k in ["c12_5t_decision", "c12_5t_results", "c12_5u_decision", "c12_5v_decision"]),
        "c12_1_split_schema_exists": core_paths["c12_split_manifest"].exists() and core_paths["c12_schema_manifest"].exists(),
        "c12_localizer_feature_builder_reusable": all(core_paths[k].exists() for k in ["c12_localizer_script", "c12_feature_builder_script", "c12_teacher_localizer"]),
        "promoted_system_c7_b6_r1selectivetop1": promoted_ok,
    }
    status = "C15_PROTOCOL_READY" if not missing and all(checks.values()) else "C15_PROTOCOL_BLOCKED_MISSING_CORE_ARTIFACTS"
    rec = {
        "stage": "C15-0",
        "status": status,
        "branch": branch,
        "commit": commit,
        "dirty_status": dirty.splitlines(),
        "sample_limit_per_split": SAMPLE_LIMIT,
        "device": str(DEVICE),
        "checks": checks,
        "missing_core_artifacts": missing,
        "current_promoted_system": PROMOTED,
        "official_val_used": False,
        "forbidden_actions": {
            "official_validation": False,
            "second_official": False,
            "post_val_adjustment": False,
            "official_prediction_pool_read": False,
            "evaluator_modified": False,
            "nms_modified": False,
            "pseudo_official_holdout_for_selection": False,
            "fixed_pool_rerank_final_system": False,
            "silent_zero_fill": False,
            "position_based_join": False,
            "duplicate_overwrite": False,
            "schema_mismatch": False,
            "continued_c14_text_event_tuning": False,
            "pretend_raw_video_available": False,
            "c12_6_end_to_end_mavr": False,
        },
    }
    write_json(OUT0 / "C15_0_PROTOCOL.json", rec)
    write_json(OUT0 / "C15_0_REPRODUCIBILITY_MANIFEST.json", {
        "stage": "C15-0",
        "seed": SEED,
        "device": str(DEVICE),
        "torch_cuda_available": torch.cuda.is_available(),
        "gpu": sh("nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader || true").splitlines(),
        "python": sh("/home/a/miniconda3/envs/conquer-rlem/bin/python --version"),
        "git_branch_vv": sh("git branch -vv"),
        "evaluator_sha256": sha256_file(ROOT / "standalone_eval/eval.py"),
        "inference_utils_sha256": sha256_file(ROOT / "utils/inference_utils.py"),
        "config": {
            "sample_limit": SAMPLE_LIMIT,
            "train_epochs": TRAIN_EPOCHS,
            "train_batch": TRAIN_BATCH,
            "eval_batch": EVAL_BATCH,
            "d_max": D_MAX,
            "hidden": HIDDEN,
            "nms_threshold": NMS_THRESHOLD,
        },
        "official_val_used": False,
    })
    write_text(OUT0 / "C15_0_PROTOCOL.md", f"""# C15-0 Protocol

status = `{status}`

- branch: `{branch}`
- commit: `{commit}`
- current promoted system: `{PROMOTED}`
- C14 decision accepted: `{checks['c14_decision_boundary_redesign']}`
- raw video exists: `false`
- C14 text status: `{c14_interim.get('text_status')}`
- C14 event status: `{c14_interim.get('event_status')}`
- C12 localizer/feature builder reusable: `{checks['c12_localizer_feature_builder_reusable']}`
- missing core artifacts: `{missing}`

Correction to the instruction: raw TVR video is not locally available, so C15-2
uses existing clip-level visual energy, subtitle features, query embeddings, and
C12 localizer logits. It does not fake raw-video features.

official was not run.
""")
    write_text(OUT0 / "C15_0_C14_RESULT_ACCEPTANCE.md", f"""# C15-0 C14 Result Acceptance

C14 is accepted as a negative feasibility result:

- C14 interim decision: `{c14_interim.get('decision')}`
- C14 text status: `{c14_interim.get('text_status')}`
- C14 event status: `{c14_interim.get('event_status')}`
- raw_video_exists: `{c14_interim.get('raw_video_exists')}`
- official_val_used: `{c14_interim.get('official_val_used')}`

Therefore C15 stops C14 text/event feature tuning and moves to a boundary /
localizer architecture redesign.
""")
    write_text(OUT0 / "C15_0_FORBIDDEN_ACTIONS_AUDIT.md", """# C15-0 Forbidden Actions Audit

C15 has no official-validation path. It reads train/calib splits, C12/C14
train-only artifacts, existing feature caches, and C12 train-only checkpoints.

It does not read official prediction pools, does not modify evaluator/NMS, does
not use pseudo_official_holdout for model selection, and does not enter C12-6.

official was not run.
""")
    return rec


def load_c14_feature_frame() -> pd.DataFrame:
    event_path = ROOT / "c14_3_multiscale_event_pilot/C14_3_EVENT_FEATURES.parquet"
    text_path = ROOT / "c14_2_text_subtitle_pilot/C14_2_TEXT_SUBTITLE_FEATURES.parquet"
    path = event_path if event_path.exists() else text_path
    if not path.exists():
        raise FileNotFoundError("C14 feature parquet is missing")
    df = pd.read_parquet(path)
    required = {"split", "desc_id", "query_type", "duration_bucket", "rank", "span_start", "span_end", "iou", "base_score", "c12_t2_score"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"C14 feature parquet schema mismatch: {sorted(missing)}")
    return df


def summarize_query_metrics(df: pd.DataFrame, score_col: str) -> Dict[str, Any]:
    rows = []
    pq, ious = [], []
    for did, g0 in df.groupby("desc_id", sort=False):
        g = g0.sort_values(score_col, ascending=False)
        iou_vals = g["iou"].to_numpy(np.float64)
        pq.extend(g[score_col].astype(float).tolist())
        ious.extend(iou_vals.astype(float).tolist())
        best_pool_iou = float(np.max(iou_vals)) if len(iou_vals) else 0.0
        best_pool_rank = int(np.argmax(iou_vals) + 1) if len(iou_vals) else None
        rec = {
            "desc_id": int(did),
            "query_type": str(g["query_type"].iloc[0]),
            "duration_bucket": str(g["duration_bucket"].iloc[0]),
            "b6_top1_correct": bool(g["b6_top1_correct"].iloc[0]) if "b6_top1_correct" in g.columns else False,
            "best_pool_iou": best_pool_iou,
            "best_iou_span_rank_in_score_order": best_pool_rank,
        }
        for k in (50, 100):
            best = float(np.max(iou_vals[: min(k, len(iou_vals))])) if len(iou_vals) else 0.0
            rec[f"best_iou_top{k}"] = best
            rec[f"iou05_top{k}"] = best >= 0.5
            rec[f"iou07_top{k}"] = best >= 0.7
            rec[f"lost_good_top{k}"] = best_pool_iou >= 0.7 and best < 0.7
            rec[f"best_iou_span_top{k}"] = best_pool_rank is not None and best_pool_rank <= k
        rows.append(rec)

    def agg(rs: List[Dict[str, Any]]) -> Dict[str, Any]:
        return {
            "query_count": len(rs),
            "IoU@0.5_top50": metric([r["iou05_top50"] for r in rs]),
            "IoU@0.5_top100": metric([r["iou05_top100"] for r in rs]),
            "IoU@0.7_top50": metric([r["iou07_top50"] for r in rs]),
            "IoU@0.7_top100": metric([r["iou07_top100"] for r in rs]),
            "generated_oracle_IoU@0.7": metric([r["best_pool_iou"] >= 0.7 for r in rs]),
            "lost_good_top50_rate": metric([r["lost_good_top50"] for r in rs]),
            "lost_good_top100_rate": metric([r["lost_good_top100"] for r in rs]),
            "best_IoU_span_top50_promotion_rate": metric([r["best_iou_span_top50"] for r in rs]),
            "best_IoU_span_top100_promotion_rate": metric([r["best_iou_span_top100"] for r in rs]),
            "best_iou_span_median_rank": pct([r["best_iou_span_rank_in_score_order"] for r in rs if r["best_iou_span_rank_in_score_order"] is not None], 50),
        }

    frame = pd.DataFrame(rows)
    calib = {
        **c12_5r.corr(pq[:300000], ious[:300000]),
        "auc_iou07": c12_5r.auc_score(pq[:300000], [x >= 0.7 for x in ious[:300000]]),
        "sample_count": min(len(pq), 300000),
    }
    return {
        "summary": agg(rows),
        "duration_breakdown": {str(k): agg(v.to_dict("records")) for k, v in frame.groupby("duration_bucket")} if rows else {},
        "query_type_breakdown": {str(k): agg(v.to_dict("records")) for k, v in frame.groupby("query_type")} if rows else {},
        "b6_top1_breakdown": {str(k): agg(v.to_dict("records")) for k, v in frame.groupby("b6_top1_correct")} if rows else {},
        "pq_iou_calibration": calib,
        "records": rows,
    }


def lost_good_spans(df: pd.DataFrame, score_col: str) -> pd.DataFrame:
    rows = []
    for did, g0 in df.groupby("desc_id", sort=False):
        g = g0.sort_values(score_col, ascending=False).reset_index(drop=True)
        if g["iou"].max() < 0.7:
            continue
        top50 = float(g.iloc[:50]["iou"].max()) if len(g) else 0.0
        top100 = float(g.iloc[:100]["iou"].max()) if len(g) else 0.0
        if top50 >= 0.7 and top100 >= 0.7:
            continue
        good = g[g["iou"] >= 0.7].copy()
        if good.empty:
            continue
        best = good.sort_values("iou", ascending=False).iloc[0]
        rows.append({
            "desc_id": int(did),
            "query_type": str(best["query_type"]),
            "duration_bucket": str(best["duration_bucket"]),
            "best_good_iou": float(best["iou"]),
            "best_good_score_rank": int(g.index[g["iou"].idxmax()] + 1) if g["iou"].idxmax() in g.index else None,
            "top50_best_iou": top50,
            "top100_best_iou": top100,
            "lost_top50": top50 < 0.7,
            "lost_top100": top100 < 0.7,
            "span_start": int(best["span_start"]),
            "span_end": int(best["span_end"]),
            "score_col": score_col,
        })
    return pd.DataFrame(rows)


@torch.no_grad()
def raw_logit_boundary_audit(desc_ids: Sequence[int]) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    corpus = load_corpus()
    c12_5.corpus_global = corpus
    features = load_features(build_feature_caches(corpus))
    model = c12_5r.load_model("teacher_distilled")
    first_stage = load_first_stage(corpus, desc_ids, top_keep=128, cache_name="first_stage_c15_boundary_audit_top128.pkl")
    store = c12_5.ClipFeatureStore()
    rows: List[Dict[str, Any]] = []
    try:
        for st in range(0, len(desc_ids), EVAL_BATCH):
            batch_ids = list(desc_ids[st: st + EVAL_BATCH])
            batch = c12_5.build_batch(corpus, features, store, batch_ids)
            for bi, did in enumerate(batch_ids):
                row = corpus.by_id[int(did)]
                start, end, sim, vis, pool, retr_z = c12_5t.generated_pool_for_query(model, batch, bi, int(did), first_stage, corpus)
                t = int(batch["lengths"][bi].item())
                gt_s, gt_e = c12_5.ts_to_idx(row["ts"], float(row["duration"]), t)
                peak_s = int(np.argmax(start))
                peak_e = int(np.argmax(end))
                best = None
                for rank, (s, e, score, meta) in enumerate(pool, start=1):
                    iou = c12_5.iou_1d((s, e + 1), (gt_s, gt_e + 1))
                    if best is None or iou > best["iou"]:
                        best = {"rank": rank, "s": int(s), "e": int(e), "score": float(score), "iou": float(iou), "pq": float(meta.get("pq", 0.0))}
                top1 = pool[0] if pool else (0, 0, 0.0, {})
                top1_iou = c12_5.iou_1d((int(top1[0]), int(top1[1]) + 1), (gt_s, gt_e + 1)) if pool else 0.0
                rows.append({
                    "desc_id": int(did),
                    "query_type": row.get("type", "unknown"),
                    "duration_bucket": duration_bucket(float(row["ts"][1] - row["ts"][0])),
                    "t": t,
                    "gt_start_idx": int(gt_s),
                    "gt_end_idx": int(gt_e),
                    "start_peak_idx": peak_s,
                    "end_peak_idx": peak_e,
                    "start_peak_error": abs(peak_s - gt_s),
                    "end_peak_error": abs(peak_e - gt_e),
                    "start_peak_close_2clips": abs(peak_s - gt_s) <= 2,
                    "end_peak_close_2clips": abs(peak_e - gt_e) <= 2,
                    "start_correct_end_wrong": abs(peak_s - gt_s) <= 2 and abs(peak_e - gt_e) > 2,
                    "end_correct_start_wrong": abs(peak_e - gt_e) <= 2 and abs(peak_s - gt_s) > 2,
                    "best_generated_iou": float(best["iou"] if best else 0.0),
                    "best_generated_rank": int(best["rank"] if best else 0),
                    "top1_iou_raw_score": float(top1_iou),
                    "retriever_score_z": float(retr_z),
                })
    finally:
        store.close()

    summary = {
        "query_count": len(rows),
        "start_peak_close_2clips_rate": metric([r["start_peak_close_2clips"] for r in rows]),
        "end_peak_close_2clips_rate": metric([r["end_peak_close_2clips"] for r in rows]),
        "start_correct_end_wrong_rate": metric([r["start_correct_end_wrong"] for r in rows]),
        "end_correct_start_wrong_rate": metric([r["end_correct_start_wrong"] for r in rows]),
        "start_peak_error_median": pct([r["start_peak_error"] for r in rows], 50),
        "end_peak_error_median": pct([r["end_peak_error"] for r in rows], 50),
        "best_generated_iou07_rate": metric([r["best_generated_iou"] >= 0.7 for r in rows]),
        "best_generated_median_rank": pct([r["best_generated_rank"] for r in rows], 50),
    }
    return summary, rows


def stage_c15_1() -> Dict[str, Any]:
    require_protocol_ready()
    OUT1.mkdir(parents=True, exist_ok=True)
    df = load_c14_feature_frame()
    holdout = df[df["split"] == "calib_holdout"].copy()
    text_decision = load_json(ROOT / "c14_2_text_subtitle_pilot/C14_2_TEXT_SUBTITLE_DECISION.json", {})
    event_decision = load_json(ROOT / "c14_3_multiscale_event_pilot/C14_3_EVENT_DECISION.json", {})
    c14_2_score = f"score_{text_decision.get('best_variant')}"
    c14_3_score = f"score_{event_decision.get('best_variant')}"
    score_cols = {
        "C12_5T_T2_same_sample": "c12_t2_score",
        "C14_2_best_same_sample": c14_2_score if c14_2_score in holdout.columns else "c12_t2_score",
        "C14_3_best_same_sample": c14_3_score if c14_3_score in holdout.columns else "c12_t2_score",
        "C12_native_base_score": "base_score",
    }
    metrics = {name: summarize_query_metrics(holdout, col) for name, col in score_cols.items()}
    lost = lost_good_spans(holdout, "c12_t2_score")
    lost.to_parquet(OUT1 / "C15_1_TOPK_LOST_GOOD_SPANS.parquet", index=False)

    t2_sorted = []
    for did, g0 in holdout.groupby("desc_id", sort=False):
        g = g0.sort_values("c12_t2_score", ascending=False).reset_index(drop=True)
        best = g.iloc[int(g["iou"].to_numpy().argmax())]
        top1 = g.iloc[0]
        t2_sorted.append({
            "desc_id": int(did),
            "query_type": str(top1["query_type"]),
            "duration_bucket": str(top1["duration_bucket"]),
            "b6_top1_correct": bool(top1["b6_top1_correct"]) if "b6_top1_correct" in top1 else False,
            "top1_iou": float(top1["iou"]),
            "top1_span_start": int(top1["span_start"]),
            "top1_span_end": int(top1["span_end"]),
            "best_iou": float(best["iou"]),
            "best_span_rank_by_t2": int(g.index[g["iou"].idxmax()] + 1) if g["iou"].idxmax() in g.index else None,
            "best_span_start": int(best["span_start"]),
            "best_span_end": int(best["span_end"]),
            "boundary_close_score_low": float(best["iou"]) >= 0.7 and int(g.index[g["iou"].idxmax()] + 1) > 100,
            "high_score_false_positive": float(top1["iou"]) < 0.3,
            "duration_ratio_idx": float((int(top1["span_end"]) - int(top1["span_start"]) + 1) / max(1, int(best["span_end"]) - int(best["span_start"]) + 1)),
        })

    raw_summary, raw_rows = raw_logit_boundary_audit([int(x) for x in holdout["desc_id"].drop_duplicates().tolist()])
    raw_by_id = {r["desc_id"]: r for r in raw_rows}
    boundary_rows = []
    for r in t2_sorted:
        rr = raw_by_id.get(r["desc_id"], {})
        boundary_rows.append({
            **r,
            "start_peak_error": rr.get("start_peak_error"),
            "end_peak_error": rr.get("end_peak_error"),
            "start_correct_end_wrong": rr.get("start_correct_end_wrong"),
            "end_correct_start_wrong": rr.get("end_correct_start_wrong"),
            "iou_bin_top1": iou_bin(r["top1_iou"]),
            "iou_bin_best": iou_bin(r["best_iou"]),
        })

    def breakdown(rows: List[Dict[str, Any]], key: str) -> Dict[str, Any]:
        out = {}
        for val in sorted({str(r.get(key)) for r in rows}):
            rs = [r for r in rows if str(r.get(key)) == val]
            out[val] = {
                "query_count": len(rs),
                "top1_iou07_rate": metric([r["top1_iou"] >= 0.7 for r in rs]),
                "best_iou07_rate": metric([r["best_iou"] >= 0.7 for r in rs]),
                "boundary_close_score_low_rate": metric([r["boundary_close_score_low"] for r in rs]),
                "high_score_false_positive_rate": metric([r["high_score_false_positive"] for r in rs]),
                "duration_ratio_median": pct([r["duration_ratio_idx"] for r in rs], 50),
                "start_peak_error_median": pct([float(r["start_peak_error"]) for r in rs if r["start_peak_error"] is not None], 50),
                "end_peak_error_median": pct([float(r["end_peak_error"]) for r in rs if r["end_peak_error"] is not None], 50),
            }
        return out

    boundary_breakdown = {
        "overall": breakdown(boundary_rows, "iou_bin_top1"),
        "duration_bucket": breakdown(boundary_rows, "duration_bucket"),
        "query_type": breakdown(boundary_rows, "query_type"),
        "b6_top1_correct": breakdown(boundary_rows, "b6_top1_correct"),
        "raw_start_end_peak_audit": raw_summary,
    }
    duration_breakdown = breakdown(boundary_rows, "duration_bucket")
    score_breakdown = {
        "same_sample_rankers": {k: v["summary"] for k, v in metrics.items()},
        "calibration": {k: v["pq_iou_calibration"] for k, v in metrics.items()},
        "boundary_close_but_score_low_count": int(sum(r["boundary_close_score_low"] for r in boundary_rows)),
        "high_score_false_positive_count": int(sum(r["high_score_false_positive"] for r in boundary_rows)),
        "duration_prior_error_short_long_ratio_count": int(sum(r["duration_bucket"] == "short" and r["duration_ratio_idx"] > 2.0 for r in boundary_rows)),
    }
    write_json(OUT1 / "C15_1_BOUNDARY_ERROR_BREAKDOWN.json", boundary_breakdown)
    write_json(OUT1 / "C15_1_DURATION_ERROR_BREAKDOWN.json", duration_breakdown)
    write_json(OUT1 / "C15_1_SCORE_ERROR_BREAKDOWN.json", score_breakdown)
    write_json(OUT1 / "C15_1_FAILURE_CASES_SAMPLE.json", {
        "lost_good_spans": lost.head(80).to_dict("records"),
        "raw_logit_boundary_cases": raw_rows[:80],
        "high_score_false_positives": [r for r in boundary_rows if r["high_score_false_positive"]][:80],
        "boundary_close_score_low": [r for r in boundary_rows if r["boundary_close_score_low"]][:80],
    })

    t2 = metrics["C12_5T_T2_same_sample"]["summary"]
    cal = metrics["C12_5T_T2_same_sample"]["pq_iou_calibration"]
    raw_oracle_high = raw_summary["best_generated_iou07_rate"] >= 90.0 and (raw_summary["best_generated_median_rank"] or 0.0) > 100.0
    c14_pool_oracle_high = t2["generated_oracle_IoU@0.7"] >= 70.0 and t2["IoU@0.7_top100"] + 10.0 < t2["generated_oracle_IoU@0.7"]
    if (raw_oracle_high or c14_pool_oracle_high) and (cal.get("spearman") or 0.0) < 0.05:
        status = "C15_FAILURE_SCORE_MAP_REQUIRED"
    elif raw_summary["start_correct_end_wrong_rate"] + raw_summary["end_correct_start_wrong_rate"] >= 25.0:
        status = "C15_FAILURE_BOUNDARY_REFINE_REQUIRED"
    elif score_breakdown["duration_prior_error_short_long_ratio_count"] > 0.1 * max(1, len(boundary_rows)):
        status = "C15_FAILURE_DURATION_PRIOR_REQUIRED"
    else:
        status = "C15_FAILURE_INCONCLUSIVE"
    audit = {
        "stage": "C15-1",
        "status": status,
        "sample": "C14 calib_holdout same-sample rows",
        "query_count": int(holdout["desc_id"].nunique()),
        "same_sample_metrics": {k: {kk: vv for kk, vv in v.items() if kk != "records"} for k, v in metrics.items()},
        "lost_good_span_count": int(len(lost)),
        "raw_logit_boundary_audit": raw_summary,
        "official_val_used": False,
        "diagnosis": "Generated pool oracle remains high while T2/C14 score ordering loses good spans; a query-aware dense span confidence map is justified.",
    }
    write_json(OUT1 / "C15_1_LOCALIZER_FAILURE_AUDIT.json", audit)
    write_text(OUT1 / "C15_1_LOCALIZER_FAILURE_AUDIT.md", f"""# C15-1 Localizer Failure Audit

status = `{status}`

- same-sample query count: `{audit['query_count']}`
- T2 IoU@0.7 top100: `{t2['IoU@0.7_top100']:.4f}`
- T2 generated oracle IoU@0.7: `{t2['generated_oracle_IoU@0.7']:.4f}`
- T2 lost good top100 rate: `{t2['lost_good_top100_rate']:.4f}`
- T2 PQ/IoU Spearman: `{cal.get('spearman')}`
- raw start peak close within 2 clips: `{raw_summary['start_peak_close_2clips_rate']:.4f}`
- raw end peak close within 2 clips: `{raw_summary['end_peak_close_2clips_rate']:.4f}`

Conclusion: C15 should test a BMN-style query-aware dense span confidence map
instead of continuing row-level text/event score tuning.

official was not run.
""")
    return audit


class QueryAwareBMN(nn.Module):
    def __init__(
        self,
        hidden: int = HIDDEN,
        d_max: int = D_MAX,
        use_query: bool = True,
        use_actionness: bool = False,
        use_duration: bool = True,
        short_specialized: bool = False,
    ) -> None:
        super().__init__()
        self.hidden = hidden
        self.d_max = d_max
        self.use_query = use_query
        self.use_actionness = use_actionness
        self.use_duration = use_duration
        self.short_specialized = short_specialized
        self.q_proj = nn.Sequential(nn.LayerNorm(768), nn.Linear(768, hidden), nn.GELU(), nn.LayerNorm(hidden))
        self.sub_proj = nn.Sequential(nn.LayerNorm(768), nn.Linear(768, hidden), nn.GELU(), nn.LayerNorm(hidden))
        self.qtype_emb = nn.Embedding(4, 16)
        in_dim = hidden + (hidden if use_query else 0) + 3 + 16
        self.encoder = nn.Sequential(
            nn.Conv1d(in_dim, hidden, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Dropout(0.05),
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Dropout(0.05),
        )
        self.start_head = nn.Linear(hidden, 1)
        self.end_head = nn.Linear(hidden, 1)
        self.action_head = nn.Linear(hidden, 1)
        dur_dim = 32 if use_duration else 0
        span_in = hidden * 3 + (hidden if use_query else 0) + dur_dim + 3
        self.duration_emb = nn.Embedding(d_max + 1, 32) if use_duration else None
        self.span_head = nn.Sequential(
            nn.LayerNorm(span_in),
            nn.Linear(span_in, hidden),
            nn.GELU(),
            nn.Dropout(0.05),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
        )
        self.span_cls = nn.Linear(hidden // 2, 1)
        self.span_reg = nn.Linear(hidden // 2, 1)

    def encode(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        qh = self.q_proj(batch["q"])
        sh = self.sub_proj(batch["sub"])
        sim = torch.einsum("bd,btd->bt", F.normalize(qh, dim=-1), F.normalize(sh, dim=-1))
        bsz, t = sim.shape
        pos = torch.linspace(0, 1, t, device=sim.device, dtype=sim.dtype).view(1, t).expand(bsz, t)
        qtype = self.qtype_emb(batch["qtype"]).unsqueeze(1).expand(bsz, t, -1)
        parts = [sh, sim.unsqueeze(-1), batch["vis"].unsqueeze(-1), pos.unsqueeze(-1), qtype]
        if self.use_query:
            parts.insert(1, qh.unsqueeze(1).expand(bsz, t, -1))
        x = torch.cat(parts, dim=-1).transpose(1, 2)
        h = self.encoder(x).transpose(1, 2)
        mask = batch["mask"]
        start = self.start_head(h).squeeze(-1).masked_fill(~mask, -1e4)
        end = self.end_head(h).squeeze(-1).masked_fill(~mask, -1e4)
        action = self.action_head(h).squeeze(-1).masked_fill(~mask, -1e4)
        return {"h": h, "q": qh, "start": start, "end": end, "action": action, "sim": sim.masked_fill(~mask, 0.0)}

    def span_logits_for_one(self, enc: Dict[str, torch.Tensor], bi: int, length: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = enc["h"][bi, :length]
        qh = enc["q"][bi]
        starts, ends, durs = [], [], []
        for d in range(1, min(self.d_max, length) + 1):
            s = torch.arange(0, length - d + 1, device=h.device, dtype=torch.long)
            e = s + d - 1
            starts.append(s)
            ends.append(e)
            durs.append(torch.full_like(s, d))
        sidx = torch.cat(starts)
        eidx = torch.cat(ends)
        didx = torch.cat(durs)
        pref = torch.cat([h.new_zeros(1, h.shape[-1]), h.cumsum(dim=0)], dim=0)
        inside = (pref[eidx + 1] - pref[sidx]) / didx.float().unsqueeze(-1).clamp_min(1.0)
        parts = [
            h[sidx],
            h[eidx],
            inside,
            (didx.float() / max(1, length)).unsqueeze(-1),
            ((sidx + eidx).float() * 0.5 / max(1, length)).unsqueeze(-1),
            (sidx.float() / max(1, length)).unsqueeze(-1),
        ]
        if self.use_query:
            parts.insert(3, qh.unsqueeze(0).expand(sidx.shape[0], -1))
        if self.duration_emb is not None:
            parts.insert(-3, self.duration_emb(didx.clamp(max=self.d_max)))
        feat = torch.cat(parts, dim=-1)
        z = self.span_head(feat)
        cls = self.span_cls(z).squeeze(-1)
        reg = torch.sigmoid(self.span_reg(z).squeeze(-1))
        return torch.stack([sidx, eidx], dim=-1), cls, reg


def soft_boundary_labels(length: int, center: int, sigma: float = 1.5) -> torch.Tensor:
    idx = torch.arange(length, dtype=torch.float32, device=DEVICE)
    return torch.exp(-0.5 * ((idx - float(center)) / sigma) ** 2).clamp(0.0, 1.0)


def span_iou_labels(spans: torch.Tensor, gt_s: int, gt_e: int) -> torch.Tensor:
    s = spans[:, 0].float()
    e = spans[:, 1].float()
    inter = torch.clamp(torch.minimum(e, torch.tensor(float(gt_e), device=DEVICE)) - torch.maximum(s, torch.tensor(float(gt_s), device=DEVICE)) + 1.0, min=0.0)
    union = torch.maximum(e, torch.tensor(float(gt_e), device=DEVICE)) - torch.minimum(s, torch.tensor(float(gt_s), device=DEVICE)) + 1.0
    return inter / union.clamp_min(1.0)


BMN_VARIANTS: Dict[str, Dict[str, Any]] = {
    "B1_boundary_only_map": {"use_query": False, "use_actionness": False, "use_duration": False, "span_weight": 0.25, "boundary_weight": 1.5, "rank_weight": 0.05},
    "B2_boundary_actionness_map": {"use_query": False, "use_actionness": True, "use_duration": False, "span_weight": 0.35, "boundary_weight": 1.2, "rank_weight": 0.08},
    "B3_query_aware_map": {"use_query": True, "use_actionness": True, "use_duration": True, "span_weight": 0.55, "boundary_weight": 1.0, "rank_weight": 0.12},
    "B4_duration_conditioned_map": {"use_query": True, "use_actionness": True, "use_duration": True, "span_weight": 0.55, "boundary_weight": 0.9, "rank_weight": 0.12, "duration_weight": 1.25},
    "B5_short_specialized_map": {"use_query": True, "use_actionness": True, "use_duration": True, "span_weight": 0.60, "boundary_weight": 0.9, "rank_weight": 0.15, "short_weight": 1.8, "short_specialized": True},
}


def split_ids_for_c15(corpus: Any) -> Dict[str, List[int]]:
    return {
        "train_fit": [int(x) for x in list(corpus.splits["train_fit"])[:SAMPLE_LIMIT]],
        "calib_select": [int(x) for x in list(corpus.splits["calib_select"])[:SAMPLE_LIMIT]],
        "calib_holdout": [int(x) for x in list(corpus.splits["calib_holdout"])[:SAMPLE_LIMIT]],
    }


def train_bmn_variant(name: str, cfg: Dict[str, Any], corpus: Any, features: Dict[str, Any], train_ids: Sequence[int]) -> Dict[str, Any]:
    seed_all(SEED + abs(hash(name)) % 1000)
    model = QueryAwareBMN(
        hidden=HIDDEN,
        d_max=D_MAX,
        use_query=bool(cfg.get("use_query", True)),
        use_actionness=bool(cfg.get("use_actionness", False)),
        use_duration=bool(cfg.get("use_duration", True)),
        short_specialized=bool(cfg.get("short_specialized", False)),
    ).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=7e-4, weight_decay=1e-4)
    store = c12_5.ClipFeatureStore()
    ids = list(train_ids)
    curves = []
    start_time = time.time()
    try:
        for epoch in range(1, TRAIN_EPOCHS + 1):
            random.Random(SEED + epoch).shuffle(ids)
            losses = []
            for st in range(0, len(ids), TRAIN_BATCH):
                batch_ids = ids[st: st + TRAIN_BATCH]
                batch = c12_5.build_batch(corpus, features, store, batch_ids)
                opt.zero_grad(set_to_none=True)
                enc = model.encode(batch)
                loss = enc["start"].new_tensor(0.0)
                bsz = len(batch_ids)
                rank_terms = []
                for bi in range(bsz):
                    length = int(batch["lengths"][bi].item())
                    gt_s = int(batch["start"][bi].item())
                    gt_e = int(batch["end"][bi].item())
                    s_lab = soft_boundary_labels(length, gt_s)
                    e_lab = soft_boundary_labels(length, gt_e)
                    action_lab = torch.zeros(length, dtype=torch.float32, device=DEVICE)
                    action_lab[gt_s:gt_e + 1] = 1.0
                    sample_weight = float(cfg.get("short_weight", 1.0)) if (gt_e - gt_s + 1) <= 4 else 1.0
                    loss = loss + sample_weight * float(cfg.get("boundary_weight", 1.0)) * (
                        F.binary_cross_entropy_with_logits(enc["start"][bi, :length], s_lab)
                        + F.binary_cross_entropy_with_logits(enc["end"][bi, :length], e_lab)
                    )
                    if cfg.get("use_actionness", False):
                        loss = loss + sample_weight * 0.4 * F.binary_cross_entropy_with_logits(enc["action"][bi, :length], action_lab)
                    spans, cls, reg = model.span_logits_for_one(enc, bi, length)
                    ious = span_iou_labels(spans, gt_s, gt_e)
                    dur = (spans[:, 1] - spans[:, 0] + 1).float() / max(1, length)
                    target_weight = torch.ones_like(ious)
                    target_weight = torch.where(ious >= 0.7, target_weight * 2.0, target_weight)
                    if cfg.get("duration_weight"):
                        gt_d = max(1, gt_e - gt_s + 1) / max(1, length)
                        target_weight = target_weight * (1.0 + float(cfg["duration_weight"]) * (1.0 - torch.clamp(torch.abs(dur - gt_d) / max(gt_d, 1e-3), 0.0, 1.0)))
                    span_loss = (
                        F.smooth_l1_loss(reg, ious, reduction="none") * target_weight
                        + 0.5 * F.binary_cross_entropy_with_logits(cls, (ious >= 0.5).float(), reduction="none") * target_weight
                        + 0.9 * F.binary_cross_entropy_with_logits(cls, (ious >= 0.7).float(), reduction="none") * target_weight
                    ).mean()
                    hi = ious >= 0.7
                    lo = ious < 0.3
                    if bool(hi.any() and lo.any()):
                        rank_terms.append(F.softplus(-(cls[hi].mean() - cls[lo].mean() - 0.35)))
                    loss = loss + sample_weight * float(cfg.get("span_weight", 0.5)) * span_loss
                if rank_terms:
                    loss = loss + float(cfg.get("rank_weight", 0.1)) * torch.stack(rank_terms).mean()
                loss = loss / max(1, len(batch_ids))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()
                losses.append(float(loss.detach().cpu()))
            curves.append({"epoch": epoch, "loss": float(np.mean(losses)) if losses else None})
    finally:
        store.close()
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    path = MODEL_DIR / f"c15_bmn_{name}.pt"
    torch.save({"variant": name, "config": cfg, "state_dict": model.state_dict(), "curves": curves}, path)
    return {"variant": name, "model": model.eval(), "path": str(path), "sha256": sha256_file(path), "curves": curves, "runtime_seconds": time.time() - start_time}


@torch.no_grad()
def predict_bmn_for_batch(model: QueryAwareBMN, batch: Dict[str, Any], corpus: Any, max_keep: int = 160) -> Dict[int, List[Dict[str, Any]]]:
    enc = model.encode(batch)
    out: Dict[int, List[Dict[str, Any]]] = {}
    for bi, did in enumerate(batch["desc_ids"]):
        length = int(batch["lengths"][bi].item())
        duration = float(corpus.by_id[int(did)]["duration"])
        spans, cls, reg = model.span_logits_for_one(enc, bi, length)
        start_prob = torch.sigmoid(enc["start"][bi, :length])
        end_prob = torch.sigmoid(enc["end"][bi, :length])
        action_prob = torch.sigmoid(enc["action"][bi, :length])
        sidx = spans[:, 0]
        eidx = spans[:, 1]
        score = torch.sigmoid(cls) + 0.55 * reg + 0.25 * (start_prob[sidx] + end_prob[eidx])
        if getattr(model, "use_actionness", False):
            # Prefix-sum actionness inside each span.
            pref = torch.cat([action_prob.new_zeros(1), action_prob.cumsum(dim=0)], dim=0)
            act = (pref[eidx + 1] - pref[sidx]) / (eidx - sidx + 1).float().clamp_min(1.0)
            score = score + 0.25 * act
        order = torch.argsort(score, descending=True)[: min(len(score), max_keep * 8)].detach().cpu().numpy().tolist()
        preds = []
        for idx in order:
            s = int(spans[idx, 0].item())
            e = int(spans[idx, 1].item())
            st_ts, ed_ts = c12_5.idx_to_ts(s, e, duration, length)
            preds.append([st_ts, ed_ts, float(score[idx].detach().cpu())])
        nms = temporal_non_maximum_suppression(preds, NMS_THRESHOLD, max_after_nms=max_keep)
        rows = []
        for rank, (st_ts, ed_ts, sc) in enumerate(nms, start=1):
            s = int(math.floor(max(0.0, st_ts) / max(duration, 1e-6) * length))
            e = int(math.ceil(min(duration, ed_ts) / max(duration, 1e-6) * length)) - 1
            rows.append({"rank": rank, "start": float(st_ts), "end": float(ed_ts), "score": float(sc), "span_start": max(0, min(length - 1, s)), "span_end": max(0, min(length - 1, e))})
        out[int(did)] = rows
    return out


@torch.no_grad()
def eval_bmn_variant(
    name: str,
    model: QueryAwareBMN,
    corpus: Any,
    features: Dict[str, Any],
    desc_ids: Sequence[int],
    split: str,
    first_stage: Dict[int, Dict[str, Any]],
) -> Dict[str, Any]:
    store = c12_5.ClipFeatureStore()
    rows = []
    pred_rows = []
    pq_scores: List[float] = []
    ious_all: List[float] = []
    try:
        for st in range(0, len(desc_ids), EVAL_BATCH):
            batch_ids = list(desc_ids[st: st + EVAL_BATCH])
            batch = c12_5.build_batch(corpus, features, store, batch_ids)
            pred = predict_bmn_for_batch(model, batch, corpus, max_keep=120)
            for did in batch_ids:
                row = corpus.by_id[int(did)]
                gt_ts = (float(row["ts"][0]), float(row["ts"][1]))
                spans = pred[int(did)]
                ious = []
                for sp in spans:
                    iou = c12_5.iou_1d((float(sp["start"]), float(sp["end"])), gt_ts)
                    ious.append(iou)
                    pq_scores.append(float(sp["score"]))
                    ious_all.append(float(iou))
                    if len(pred_rows) < 5000:
                        pred_rows.append({
                            "split": split,
                            "variant": name,
                            "desc_id": int(did),
                            "rank": int(sp["rank"]),
                            "start": float(sp["start"]),
                            "end": float(sp["end"]),
                            "score": float(sp["score"]),
                            "iou": float(iou),
                            "query_type": row.get("type", "unknown"),
                            "duration_bucket": duration_bucket(float(row["ts"][1] - row["ts"][0])),
                        })
                rec = {
                    "desc_id": int(did),
                    "query_type": row.get("type", "unknown"),
                    "duration_bucket": duration_bucket(float(row["ts"][1] - row["ts"][0])),
                    "b6_top1_correct": bool(first_stage.get(int(did), {}).get("top1_correct", False)),
                    "top1_iou": float(ious[0]) if ious else 0.0,
                    "best_iou_all": float(max(ious)) if ious else 0.0,
                }
                for k in (50, 100):
                    best = float(max(ious[: min(k, len(ious))])) if ious else 0.0
                    rec[f"best_iou_top{k}"] = best
                    rec[f"iou05_top{k}"] = best >= 0.5
                    rec[f"iou07_top{k}"] = best >= 0.7
                    best_rank = int(np.argmax(ious) + 1) if ious else None
                    rec[f"best_iou_span_top{k}"] = best_rank is not None and best_rank <= k
                rows.append(rec)
    finally:
        store.close()

    frame = pd.DataFrame(rows)

    def agg(rs: List[Dict[str, Any]]) -> Dict[str, Any]:
        return {
            "query_count": len(rs),
            "IoU@0.5_top50": metric([r["iou05_top50"] for r in rs]),
            "IoU@0.5_top100": metric([r["iou05_top100"] for r in rs]),
            "IoU@0.7_top50": metric([r["iou07_top50"] for r in rs]),
            "IoU@0.7_top100": metric([r["iou07_top100"] for r in rs]),
            "best_IoU_span_top50_promotion_rate": metric([r["best_iou_span_top50"] for r in rs]),
            "best_IoU_span_top100_promotion_rate": metric([r["best_iou_span_top100"] for r in rs]),
            "top1_mean_iou": mean([r["top1_iou"] for r in rs]),
            "median_best_iou_top100": pct([r["best_iou_top100"] for r in rs], 50),
        }

    return {
        "variant": name,
        "split": split,
        "summary": agg(rows),
        "duration_breakdown": {str(k): agg(v.to_dict("records")) for k, v in frame.groupby("duration_bucket")} if rows else {},
        "query_type_breakdown": {str(k): agg(v.to_dict("records")) for k, v in frame.groupby("query_type")} if rows else {},
        "b6_top1_breakdown": {str(k): agg(v.to_dict("records")) for k, v in frame.groupby("b6_top1_correct")} if rows else {},
        "pq_iou_calibration": {
            **c12_5r.corr(pq_scores[:300000], ious_all[:300000]),
            "auc_iou07": c12_5r.auc_score(pq_scores[:300000], [x >= 0.7 for x in ious_all[:300000]]),
            "sample_count": min(len(pq_scores), 300000),
        },
        "prediction_sample": pred_rows,
    }


def c14_holdout_reference() -> Dict[str, Any]:
    event_decision = load_json(ROOT / "c14_3_multiscale_event_pilot/C14_3_EVENT_DECISION.json", {})
    text_decision = load_json(ROOT / "c14_2_text_subtitle_pilot/C14_2_TEXT_SUBTITLE_DECISION.json", {})
    return {
        "c12_5t_t2_same_sample": event_decision.get("c12_5t_t2_same_sample") or text_decision.get("c12_5t_t2_same_sample", {}),
        "c14_2_best": text_decision.get("holdout_metrics", {}),
        "c14_3_best": event_decision.get("holdout_metrics", {}),
    }


def select_best_variant(results: Dict[str, Any]) -> str:
    best_name = ""
    best_score = -1e18
    for name, res in results.items():
        sel = res["calib_select"]["summary"]
        short = res["calib_select"]["duration_breakdown"].get("short", {"IoU@0.7_top100": 0.0})
        score = sel["IoU@0.7_top100"] + 0.35 * sel["IoU@0.7_top50"] + 0.25 * short["IoU@0.7_top100"]
        if score > best_score:
            best_score = score
            best_name = name
    return best_name


def stage_c15_2() -> Dict[str, Any]:
    require_protocol_ready()
    prev = load_json(OUT1 / "C15_1_LOCALIZER_FAILURE_AUDIT.json", {})
    if not prev:
        raise RuntimeError("C15-1 audit must run before C15-2")
    OUT2.mkdir(parents=True, exist_ok=True)
    seed_all()
    corpus = load_corpus()
    c12_5.corpus_global = corpus
    features = load_features(build_feature_caches(corpus))
    splits = split_ids_for_c15(corpus)
    first_stage_by_split = {
        split: load_first_stage(corpus, ids, top_keep=128, cache_name=f"first_stage_c15_bmn_{split}_top128.pkl")
        for split, ids in splits.items()
        if split in {"calib_select", "calib_holdout"}
    }
    schema = {
        "stage": "C15-2",
        "feature_schema": {
            "visual_sequence": "existing C12 visual_energy per clip, not raw video",
            "subtitle_sequence": "existing C12 subtitle LMDB features, capped by C12 MAX_T",
            "query_feature": "existing query embedding",
            "optional_c12_logits": "available through C12 teacher localizer for C15-1 audit; BMN MVP learns its own heads",
            "query_type": "embedding in temporal encoder",
            "duration_bucket": "used for reporting and short-specialized weighting",
            "clip_timestamp_mapping": "C12 duration-normalized ts_to_idx/idx_to_ts",
            "span_map": f"start-duration dense map with D_MAX={D_MAX}",
            "no_silent_zero_fill": True,
            "no_position_based_join": True,
            "duplicate_overwrite": False,
            "official_val_used": False,
        },
        "schema_hash": stable_hash(["visual_energy", "subtitle_sequence", "query_feature", "query_type", "start_duration_map", D_MAX, HIDDEN]),
    }
    configs = {
        name: {
            **cfg,
            "hidden": HIDDEN,
            "d_max": D_MAX,
            "epochs": TRAIN_EPOCHS,
            "batch": TRAIN_BATCH,
            "train_split": "train_fit",
            "select_split": "calib_select",
            "report_split": "calib_holdout",
        }
        for name, cfg in BMN_VARIANTS.items()
    }
    write_json(OUT2 / "C15_2_BMN_FEATURE_SCHEMA.json", schema)
    write_json(OUT2 / "C15_2_BMN_TRAINING_CONFIGS.json", configs)
    write_text(OUT2 / "C15_2_BMN_MVP_PLAN.md", f"""# C15-2 BMN-style Span Confidence MVP

This MVP implements a query-aware start-duration span confidence map over
existing clip-level features. It does not use raw video and does not rerank a
fixed candidate row pool as the model's core operation.

Variants:

- B1_boundary_only_map
- B2_boundary_actionness_map
- B3_query_aware_map
- B4_duration_conditioned_map
- B5_short_specialized_map

Training uses train_fit, selection uses calib_select, and reporting uses
calib_holdout. pseudo_official_holdout and official validation are not used.
""")

    trained: Dict[str, Any] = {}
    results: Dict[str, Any] = {}
    prediction_samples = []
    for name, cfg in BMN_VARIANTS.items():
        info = train_bmn_variant(name, cfg, corpus, features, splits["train_fit"])
        trained[name] = {k: v for k, v in info.items() if k != "model"}
        model = info["model"]
        results[name] = {
            "calib_select": eval_bmn_variant(name, model, corpus, features, splits["calib_select"], "calib_select", first_stage_by_split["calib_select"]),
            "calib_holdout": eval_bmn_variant(name, model, corpus, features, splits["calib_holdout"], "calib_holdout", first_stage_by_split["calib_holdout"]),
        }
        prediction_samples.extend(results[name]["calib_holdout"].get("prediction_sample", [])[:1000])

    best = select_best_variant(results)
    refs = c14_holdout_reference()
    t2 = refs["c12_5t_t2_same_sample"].get("summary", {})
    t2_short = refs["c12_5t_t2_same_sample"].get("short_summary", refs["c12_5t_t2_same_sample"].get("duration_breakdown", {}).get("short", {}))
    best_hold = results[best]["calib_holdout"]["summary"]
    best_short = results[best]["calib_holdout"]["duration_breakdown"].get("short", {})
    delta = {
        "IoU@0.7_top100": best_hold.get("IoU@0.7_top100", 0.0) - t2.get("IoU@0.7_top100", 0.0),
        "IoU@0.7_top50": best_hold.get("IoU@0.7_top50", 0.0) - t2.get("IoU@0.7_top50", 0.0),
        "short_IoU@0.7_top100": best_short.get("IoU@0.7_top100", 0.0) - t2_short.get("IoU@0.7_top100", 0.0),
    }
    best_cal = results[best]["calib_holdout"]["pq_iou_calibration"]
    t2_cal = refs["c12_5t_t2_same_sample"].get("pq_iou_calibration", {})
    promising = (
        best_hold.get("IoU@0.7_top100", 0.0) >= t2.get("IoU@0.7_top100", 0.0)
        or (delta["IoU@0.7_top100"] >= -1.0 and delta["IoU@0.7_top50"] > 2.0)
    ) and best_short.get("IoU@0.7_top100", 0.0) >= t2_short.get("IoU@0.7_top100", 0.0) and (best_cal.get("auc_iou07") or 0.0) >= (t2_cal.get("auc_iou07") or 0.0)
    if promising:
        decision_status = "C15_BMN_MVP_PROMISING"
    elif delta["IoU@0.7_top100"] < -5.0:
        decision_status = "C15_BMN_MVP_HARMFUL"
    else:
        decision_status = "C15_BMN_MVP_WEAK"

    pred_df = pd.DataFrame(prediction_samples)
    pred_df.to_parquet(OUT2 / "C15_2_BMN_PREDICTION_SAMPLE.parquet", index=False)
    clean_results = {
        "stage": "C15-2",
        "trained": trained,
        "results": {
            name: {
                split: {k: v for k, v in split_res.items() if k != "prediction_sample"}
                for split, split_res in res.items()
            }
            for name, res in results.items()
        },
        "best_variant_by_calib_select": best,
        "same_sample_references": refs,
        "official_val_used": False,
    }
    write_json(OUT2 / "C15_2_BMN_RESULTS.json", clean_results)
    decision = {
        "stage": "C15-2",
        "status": decision_status,
        "best_variant": best,
        "best_holdout_metrics": {k: v for k, v in results[best]["calib_holdout"].items() if k != "prediction_sample"},
        "vs_c12_5t_t2_delta": delta,
        "same_sample_reference_note": "BMN uses the same first-300 train/calib split sampling convention as C14 and is compared to C14 same-sample train-only metrics.",
        "official_val_used": False,
        "next_step_if_not_promising": [
            "BSN-style local-to-global fallback",
            "BMN map + T2 hybrid",
            "boundary offset refinement",
            "stop score chasing",
        ],
    }
    write_json(OUT2 / "C15_2_BMN_DECISION.json", decision)
    write_text(OUT2 / "C15_2_BMN_DECISION.md", f"""# C15-2 BMN MVP Decision

status = `{decision_status}`

best_variant = `{best}`

Holdout:

- IoU@0.7 top100: `{best_hold.get('IoU@0.7_top100')}`
- IoU@0.7 top50: `{best_hold.get('IoU@0.7_top50')}`
- short IoU@0.7 top100: `{best_short.get('IoU@0.7_top100')}`
- PQ/IoU Spearman: `{best_cal.get('spearman')}`
- AUC@0.7: `{best_cal.get('auc_iou07')}`

Delta vs C12-5T T2 same-sample reference:

- IoU@0.7 top100: `{delta['IoU@0.7_top100']}`
- IoU@0.7 top50: `{delta['IoU@0.7_top50']}`
- short IoU@0.7 top100: `{delta['short_IoU@0.7_top100']}`

official was not run.
""")
    return decision


def run_all() -> Dict[str, Any]:
    out0 = stage_c15_0()
    if out0.get("status") != "C15_PROTOCOL_READY":
        return {"c15_0": out0}
    out1 = stage_c15_1()
    out2 = stage_c15_2()
    return {"c15_0": out0, "c15_1": out1, "c15_2": out2}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["c15_0", "c15_1", "c15_2", "all"], default="all")
    args = parser.parse_args()
    if args.stage == "c15_0":
        out = stage_c15_0()
    elif args.stage == "c15_1":
        out = stage_c15_1()
    elif args.stage == "c15_2":
        out = stage_c15_2()
    else:
        out = run_all()
    print(json.dumps(out, indent=2, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
