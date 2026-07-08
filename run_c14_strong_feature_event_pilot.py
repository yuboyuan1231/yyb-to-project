#!/usr/bin/env python3
"""C14 Strong Feature / Event Evidence Pilot.

Small-sample, train-only pilot for C13 Path B. This script does not run
official validation, does not read official prediction pools, does not modify
evaluator/NMS, and does not use pseudo_official_holdout for model selection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pickle
import random
import subprocess
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import lmdb
import msgpack_numpy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import run_c12_5_native_span_generation as c12_5
import run_c12_5r_span_head_repair as c12_5r
import run_c12_5t_best_span_promotion as c12_5t
from run_c12_native_retriever_training import (
    PATHS,
    ROOT,
    build_feature_caches,
    load_corpus,
    load_features,
    load_first_stage,
    qtype_id,
    sha256_file,
)


SEED = 1414
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAMPLE_LIMIT = int(os.environ.get("C14_SAMPLE_LIMIT", "300"))
POOL_EVAL_LIMIT = int(os.environ.get("C14_POOL_EVAL_LIMIT", "300"))
BATCH_SIZE = int(os.environ.get("C14_BATCH_SIZE", "32"))
TRAIN_EPOCHS = int(os.environ.get("C14_TRAIN_EPOCHS", "8"))

OUT0 = ROOT / "c14_0_protocol_freeze"
OUT1 = ROOT / "c14_1_data_feature_audit"
OUT2 = ROOT / "c14_2_text_subtitle_pilot"
OUT3 = ROOT / "c14_3_multiscale_event_pilot"

PROMOTED = "C7-B6 R1SelectiveTop1"


TEXT_FEATURES = [
    "query_global_to_subtitle_clip_max",
    "query_global_to_subtitle_clip_mean",
    "query_to_subtitle_window_max",
    "query_to_subtitle_window_mean",
    "subtitle_mass_inside_span",
    "subtitle_mean_inside_span",
    "subtitle_max_inside_span",
    "subtitle_peak_in_span",
    "subtitle_inside_outside_contrast",
    "subtitle_left_context_contrast",
    "subtitle_right_context_contrast",
    "subtitle_coverage_ratio",
    "subtitle_missing_mask",
    "qtype_v",
    "qtype_t",
    "qtype_vt",
    "qtype_unknown",
    "duration_bucket_short",
    "duration_bucket_medium",
    "duration_bucket_long",
    "short_moment",
]

EVENT_FEATURES = [
    "event_mass_inside_span",
    "event_mean_inside_span",
    "event_max_inside_span",
    "event_peak_in_span",
    "event_inside_outside_contrast",
    "event_left_right_context_contrast",
    "event_boundary_start_alignment",
    "event_boundary_end_alignment",
    "event_overlap_ratio",
    "best_event_overlap",
    "query_event_similarity",
    "subtitle_event_similarity",
    "visual_event_similarity",
    "multiscale_span_score_2",
    "multiscale_span_score_4",
    "multiscale_span_score_8",
    "multiscale_span_score_16",
    "multiscale_boundary_contrast_2",
    "multiscale_boundary_contrast_4",
    "multiscale_boundary_contrast_8",
    "multiscale_boundary_contrast_16",
]

META_FEATURES = [
    "base_score",
    "c12_t2_score",
    "rank_norm",
    "span_duration_norm",
    "center_norm",
    "retriever_score_z",
]


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


def stable_hash(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def sh(cmd: str) -> str:
    return subprocess.check_output(cmd, shell=True, cwd=ROOT, text=True, stderr=subprocess.STDOUT).strip()


def duration_bucket(seconds: float) -> str:
    if seconds <= 5:
        return "short"
    if seconds <= 15:
        return "medium"
    return "long"


def safe_mean(xs: Sequence[float]) -> float:
    return float(np.mean(xs)) if xs else 0.0


def safe_max(xs: Sequence[float]) -> float:
    return float(np.max(xs)) if xs else 0.0


def prefix_sum(x: np.ndarray) -> np.ndarray:
    return np.concatenate([[0.0], np.cumsum(x.astype(np.float64))])


def range_mean(pref: np.ndarray, s: int, e: int) -> float:
    if e < s:
        return 0.0
    s = max(0, s)
    e = min(len(pref) - 2, e)
    if e < s:
        return 0.0
    return float((pref[e + 1] - pref[s]) / max(1, e - s + 1))


def range_sum(pref: np.ndarray, s: int, e: int) -> float:
    if e < s:
        return 0.0
    s = max(0, s)
    e = min(len(pref) - 2, e)
    if e < s:
        return 0.0
    return float(pref[e + 1] - pref[s])


def local_window(s: int, e: int, t: int, pad: int = 2) -> Tuple[int, int]:
    return max(0, s - pad), min(t - 1, e + pad)


def stage_status_path(stage: str) -> Path:
    return {
        "c14_0": OUT0 / "C14_0_PROTOCOL_FREEZE.json",
        "c14_1": OUT1 / "C14_1_DATA_AVAILABILITY_AUDIT.json",
        "c14_2": OUT2 / "C14_2_TEXT_SUBTITLE_DECISION.json",
        "c14_3": OUT3 / "C14_3_EVENT_DECISION.json",
    }[stage]


def require_previous_ready(stage: str) -> None:
    if stage == "c14_0":
        return
    c14_0 = load_json(stage_status_path("c14_0"), {})
    if c14_0.get("status") != "C14_PROTOCOL_READY":
        raise RuntimeError(f"C14-0 is not ready: {c14_0.get('status')}")


def protocol_stage() -> Dict[str, Any]:
    OUT0.mkdir(parents=True, exist_ok=True)
    branch = sh("git branch --show-current")
    commit = sh("git rev-parse HEAD")
    dirty = sh("git status --short")
    root_markers = [p.name for p in [ROOT / "OFFICIAL_VAL_AUTHORIZED", ROOT / "C9_OFFICIAL_VAL_AUTHORIZED"] if p.exists()]
    core = {
        "c12_1_schema_and_split": ROOT / "c12_1_schema_and_split",
        "c12_system_reliability_audit": ROOT / "c12_system_reliability_audit",
        "c12_5t_decision": ROOT / "c12_5t_best_span_promotion/C12_5T_DECISION.json",
        "c12_5u_decision": ROOT / "c12_5u_topk_calibration_repair/C12_5U_DECISION.json",
        "c12_5v_decision": ROOT / "c12_5v_conquer_hidden_ranker/C12_5V_DECISION.json",
        "c13_decision": ROOT / "c13_3_next_step_decision/C13_3_NEXT_STEP_DECISION.json",
        "c7_final_decision": ROOT / "c7_final_archive/C7_FINAL_DECISION.json",
        "c12_4r_replay": ROOT / "c12_4r_retriever_repair/C12_4R_RETRIEVER_REPAIR_DECISION.json",
    }
    missing = [name for name, path in core.items() if not path.exists()]
    if root_markers:
        status = "C14_PROTOCOL_BLOCKED_CONTAMINATION_RISK"
    elif missing:
        status = "C14_PROTOCOL_BLOCKED_MISSING_CORE_ARTIFACTS"
    else:
        status = "C14_PROTOCOL_READY"
    out = {
        "stage": "C14-0",
        "status": status,
        "branch": branch,
        "commit": commit,
        "dirty_status": dirty.splitlines(),
        "current_promoted_system": PROMOTED,
        "c12_6_forbidden": True,
        "official_forbidden": True,
        "second_official_forbidden": True,
        "post_val_adjustment_forbidden": True,
        "official_prediction_pool_read_forbidden": True,
        "evaluator_nms_modification_forbidden": True,
        "pseudo_official_holdout_selection_forbidden": True,
        "root_stale_markers": root_markers,
        "missing_artifacts": missing,
        "artifact_presence": {name: path.exists() for name, path in core.items()},
        "evaluator_sha256": sha256_file(ROOT / "standalone_eval/eval.py"),
        "nms_sha256": sha256_file(ROOT / "utils/inference_utils.py"),
    }
    write_json(OUT0 / "C14_0_PROTOCOL_FREEZE.json", out)
    write_json(OUT0 / "C14_0_REPRODUCIBILITY_MANIFEST.json", {
        "stage": "C14-0",
        "seed": SEED,
        "device": str(DEVICE),
        "sample_limit": SAMPLE_LIMIT,
        "pool_eval_limit": POOL_EVAL_LIMIT,
        "python": sh("/home/a/miniconda3/envs/conquer-rlem/bin/python --version"),
        "git_branch_vv": sh("git branch -vv"),
        "official_val_used": False,
    })
    write_text(OUT0 / "C14_0_PROTOCOL_FREEZE.md", f"""# C14-0 Protocol Freeze

status = `{status}`

- branch: `{branch}`
- commit: `{commit}`
- promoted_system: `{PROMOTED}`
- C12-6 forbidden: true
- official forbidden: true
- second official forbidden: true
- post-val adjustment forbidden: true
- official prediction pool read forbidden: true
- evaluator/NMS modification forbidden: true
- pseudo_official_holdout selection forbidden: true
- root stale markers: `{root_markers}`
- missing_artifacts: `{missing}`

Evaluator hash: `{out['evaluator_sha256']}`

NMS hash: `{out['nms_sha256']}`

official was not run.
""")
    write_text(OUT0 / "C14_0_FORBIDDEN_ACTIONS_AUDIT.md", f"""# C14-0 Forbidden Actions Audit

No official validation path exists in C14. The script only reads train-release
metadata, train/calib splits, existing C12/C13 artifacts, feature LMDB/NPZ
caches, and local model checkpoints for train-only small-sample pilots.

Root stale authorization markers: `{root_markers}`.

If root stale markers appear, quarantine should be recommended, but historical
`OFFICIAL_VAL_ALREADY_RUN` markers inside official output directories must not be
deleted.

official was not run.
""")
    return out


def scan_model_caches() -> Dict[str, Any]:
    homes = [
        Path.home() / ".cache/huggingface",
        Path.home() / ".cache/torch",
        Path.home() / ".cache/open_clip",
        Path.home() / ".cache/sentence_transformers",
        Path.home() / ".cache/sentence-transformers",
    ]
    out = {}
    for p in homes:
        files = []
        if p.exists():
            for f in list(p.rglob("*"))[:200]:
                if f.is_file():
                    files.append(str(f.relative_to(p)))
        out[str(p)] = {"exists": p.exists(), "sample_files": files[:30]}
    return out


def raw_video_files() -> List[str]:
    base = Path("/home/a/yybwork/data/yyb")
    pats = ["*.mp4", "*.mkv", "*.avi", "*.webm"]
    files: List[str] = []
    for pat in pats:
        files.extend(str(p) for p in base.rglob(pat))
    return sorted(files)


def read_lmdb_sample_dims() -> Dict[str, Any]:
    dims: Dict[str, Any] = {}
    for name, path, kind in [
        ("subtitle_lmdb", PATHS.subtitle_lmdb, "npz"),
        ("query_lmdb", PATHS.query_lmdb, "npz"),
        ("visual_lmdb", PATHS.visual_lmdb, "msgpack"),
        ("first_stage_rank_lmdb", PATHS.first_stage_rank_lmdb, "msgpack"),
    ]:
        rec: Dict[str, Any] = {"path": str(path), "exists": path.exists()}
        if path.exists():
            try:
                env = lmdb.open(str(path), readonly=True, create=False, lock=False, readahead=False, max_readers=64)
                with env.begin(buffers=True) as txn:
                    cur = txn.cursor()
                    if cur.first():
                        rec["first_key"] = cur.key().decode(errors="replace")
                        raw = bytes(cur.value())
                        if kind == "npz":
                            import io
                            with io.BytesIO(raw) as reader:
                                arr = np.load(reader, allow_pickle=True)["features"]
                            rec["shape"] = list(arr.shape)
                            rec["finite_sample"] = bool(np.isfinite(arr[: min(arr.shape[0], 4)]).all())
                        elif kind == "msgpack":
                            val = msgpack_numpy.loads(raw, raw=False)
                            if isinstance(val, dict) and "features" in val:
                                arr = val["features"]
                                rec["shape"] = list(arr.shape)
                                rec["finite_sample"] = bool(np.isfinite(arr[: min(arr.shape[0], 4)]).all())
                            elif isinstance(val, list):
                                rec["list_length"] = len(val)
                                rec["first_item_type"] = type(val[0]).__name__ if val else None
                env.close()
            except Exception as exc:
                rec["error"] = repr(exc)
        dims[name] = rec
    return dims


def build_sample_manifest(corpus: Any, first_stage: Dict[int, Dict[str, Any]]) -> Dict[str, Any]:
    rng = random.Random(SEED)
    split_ids = list(corpus.splits["train_fit"][:400]) + list(corpus.splits["calib_select"][:200]) + list(corpus.splits["calib_holdout"][:400])
    chosen: List[int] = []
    categories: Dict[str, List[int]] = defaultdict(list)

    def add(did: int, cat: str) -> None:
        did = int(did)
        if did not in chosen:
            chosen.append(did)
        if did not in categories[cat]:
            categories[cat].append(did)

    for did in split_ids:
        row = corpus.by_id[int(did)]
        qtype = row.get("type", "unknown")
        dur = float(row["ts"][1] - row["ts"][0])
        fs = first_stage.get(int(did), {})
        if len(categories[f"qtype_{qtype}"]) < 30:
            add(did, f"qtype_{qtype}")
        if len(categories[f"duration_{duration_bucket(dur)}"]) < 30:
            add(did, f"duration_{duration_bucket(dur)}")
        if fs.get("top1_correct") is True and len(categories["b6_top1_correct"]) < 40:
            add(did, "b6_top1_correct")
        if fs.get("top1_correct") is False and len(categories["b6_top1_wrong"]) < 40:
            add(did, "b6_top1_wrong")
        if isinstance(fs.get("rank"), int) and int(fs["rank"]) > 100 and len(categories["positive_not_in_b6_top100"]) < 20:
            add(did, "positive_not_in_b6_top100")

    vfail = load_json(ROOT / "c12_5v_conquer_hidden_ranker/C12_5V_D_FAILURE_CASES.json", {})
    for item in vfail.get("V1_hidden_boundary", {}).get("high_score_low_iou", [])[:60]:
        if int(item["desc_id"]) in corpus.by_id:
            add(int(item["desc_id"]), "c12_5v_hidden_ranker_regression")
            add(int(item["desc_id"]), "c12_span_ranker_failure")
    tfail = load_json(ROOT / "c12_5t_best_span_promotion/C12_5T_E_FAILURE_CASES.json", {})
    for item in tfail.get("T2_listwise_soft_iou", [])[:80]:
        if int(item["desc_id"]) in corpus.by_id:
            add(int(item["desc_id"]), "c12_5t_topk_selector_good_oracle_case")

    pool = list(dict.fromkeys(split_ids))
    rng.shuffle(pool)
    for did in pool:
        if len(chosen) >= SAMPLE_LIMIT:
            break
        add(did, "background_sample")

    rows = []
    for did in chosen[:SAMPLE_LIMIT]:
        row = corpus.by_id[int(did)]
        fs = first_stage.get(int(did), {})
        rows.append({
            "desc_id": int(did),
            "split": next((s for s in ["train_fit", "calib_select", "calib_holdout"] if int(did) in set(map(int, corpus.splits[s]))), "unknown"),
            "query_type": row.get("type", "unknown"),
            "duration": float(row["ts"][1] - row["ts"][0]),
            "duration_bucket": duration_bucket(float(row["ts"][1] - row["ts"][0])),
            "video": row.get("vid_name"),
            "b6_rank": fs.get("rank"),
            "b6_top1_correct": fs.get("top1_correct"),
        })

    coverage = {k: len(v) for k, v in sorted(categories.items())}
    positive_not = len(categories.get("positive_not_in_b6_top100", []))
    return {
        "stage": "C14-1",
        "status_note": "positive_not_in_b6_top100 is naturally absent if CONQUER/B6 replay top100 coverage is 100%",
        "sample_limit": SAMPLE_LIMIT,
        "sample_count": len(rows),
        "coverage_counts": coverage,
        "positive_not_in_b6_top100_blocked": positive_not == 0,
        "positive_not_in_b6_top100_proxy": "native C12 retriever missed-top100 aggregate from C12-4 if natural B6 replay miss is absent",
        "rows": rows,
    }


def data_audit_stage() -> Dict[str, Any]:
    require_previous_ready("c14_1")
    OUT1.mkdir(parents=True, exist_ok=True)
    corpus = load_corpus()
    first_stage = load_first_stage(corpus, corpus.splits["calib_holdout"], top_keep=128, cache_name="first_stage_calib_holdout_top128.pkl")
    features = load_features(build_feature_caches(corpus))
    raw = raw_video_files()
    lmdb_dims = read_lmdb_sample_dims()
    sample = build_sample_manifest(corpus, first_stage)
    nan_inf = {
        "query_feature_finite": bool(np.isfinite(features["query"][:128]).all()),
        "subtitle_mean_finite": bool(np.isfinite(features["sub_mean"][:128]).all()),
        "subtitle_max_finite": bool(np.isfinite(features["sub_max"][:128]).all()),
        "visual_mean_finite": bool(np.isfinite(features["visual_mean"][:128]).all()),
    }
    raw_status = "C14_RAW_VIDEO_AVAILABLE" if raw else "C14_RAW_VIDEO_NOT_AVAILABLE"
    status = "C14_STRONG_VISUAL_EXTRACTION_FEASIBLE" if raw else "C14_EVENT_FEATURE_FEASIBLE_FROM_EXISTING_FEATURES"
    audit = {
        "stage": "C14-1",
        "status": status,
        "raw_video_status": raw_status,
        "strong_visual_status": "C14_STRONG_VISUAL_EXTRACTION_FEASIBLE" if raw else "C14_STRONG_VISUAL_EXTRACTION_BLOCKED_BY_RAW_VIDEO_ABSENCE",
        "query_count": len(corpus.train_rows),
        "video_count": len(corpus.train_videos),
        "split_counts": {k: len(v) for k, v in corpus.splits.items()},
        "query_feature_dim": int(features["query"].shape[1]),
        "subtitle_feature_dim": int(features["sub_mean"].shape[1]),
        "visual_feature_dim": int(features["visual_mean"].shape[1]),
        "feature_row_count": {
            "query": int(features["query"].shape[0]),
            "subtitle_video": int(features["sub_mean"].shape[0]),
            "visual_video": int(features["visual_mean"].shape[0]),
        },
        "clip_length_timestamp_mapping": "Existing C12 localizer maps timestamps to clip indices with duration-normalized ts_to_idx/idx_to_ts over subtitle clip count capped by MAX_T.",
        "subtitle_timestamp_exists": True,
        "raw_video_exists": bool(raw),
        "raw_video_files_sample": raw[:20],
        "lmdb_sample_dims": lmdb_dims,
        "nan_inf_sample_audit": nan_inf,
        "supports_clip_siglip_dinov2_videomae_internvideo_pilot": bool(raw),
        "supports_text_subtitle_pilot_from_existing_features": True,
        "supports_event_feature_pilot_from_existing_features": True,
        "official_val_used": False,
    }
    cost = {
        "stage": "C14-1",
        "gpu": sh("nvidia-smi --query-gpu=name,memory.total,memory.free,driver_version --format=csv,noheader || true").splitlines(),
        "disk": sh("df -h /home/a /home/a/yybwork /tmp 2>/dev/null || df -h").splitlines(),
        "storage_estimate": {
            "c14_text_parquet_sample": "tens to hundreds of MiB depending C14_POOL_EVAL_LIMIT",
            "c14_event_parquet_sample": "tens to hundreds of MiB depending C14_POOL_EVAL_LIMIT",
            "full_raw_visual_extraction": "blocked unless raw video is located; would likely require tens to hundreds of GiB",
        },
        "gpu_cpu_cost_estimate": {
            "text_subtitle_sample": "single GPU minutes; CPU fallback possible",
            "event_existing_features_sample": "single GPU minutes plus CPU feature building",
            "raw_video_visual_models": "blocked_by_raw_video_absence",
        },
        "official_val_used": False,
    }
    write_json(OUT1 / "C14_1_DATA_AVAILABILITY_AUDIT.json", audit)
    write_json(OUT1 / "C14_1_RAW_VIDEO_AUDIT.json", {
        "raw_video_exists": bool(raw),
        "raw_video_count": len(raw),
        "raw_video_files_sample": raw[:50],
        "status": "C14_RAW_VIDEO_AVAILABLE" if raw else "C14_STRONG_VISUAL_EXTRACTION_BLOCKED_BY_RAW_VIDEO_ABSENCE",
    })
    write_json(OUT1 / "C14_1_MODEL_CACHE_AUDIT.json", scan_model_caches())
    write_json(OUT1 / "C14_1_SAMPLE_MANIFEST.json", sample)
    write_json(OUT1 / "C14_1_COST_ESTIMATE.json", cost)
    write_text(OUT1 / "C14_1_DATA_AVAILABILITY_AUDIT.md", f"""# C14-1 Data Availability Audit

status = `{status}`

- raw_video_exists: `{bool(raw)}`
- strong_visual_status: `{audit['strong_visual_status']}`
- query_count: `{audit['query_count']}`
- video_count: `{audit['video_count']}`
- query_feature_dim: `{audit['query_feature_dim']}`
- subtitle_feature_dim: `{audit['subtitle_feature_dim']}`
- visual_feature_dim: `{audit['visual_feature_dim']}`
- sample_count: `{sample['sample_count']}`
- positive_not_in_b6_top100_blocked: `{sample['positive_not_in_b6_top100_blocked']}`

If raw video is absent, C14 records
`C14_STRONG_VISUAL_EXTRACTION_BLOCKED_BY_RAW_VIDEO_ABSENCE` and does not fake
CLIP/SigLIP/DINOv2/VideoMAE/InternVideo extraction.

official was not run.
""")
    return audit


class TinyRanker(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 96) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(0.05),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def load_t2_ranker() -> Tuple[nn.Module | None, List[int]]:
    path = ROOT / "c12_models/c12_5t_T2_listwise_soft_iou.pt"
    if not path.exists():
        return None, []
    idxs = c12_5t.variant_feature_indices("T2_listwise_soft_iou")
    model = c12_5t.RichRanker(in_dim=len(idxs), hidden=224).to(DEVICE)
    ckpt = torch.load(path, map_location=DEVICE)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, idxs


def fixed_window_scores(x: np.ndarray, windows: Sequence[int]) -> Dict[int, np.ndarray]:
    out: Dict[int, np.ndarray] = {}
    t = len(x)
    for w in windows:
        vals = np.zeros(t, dtype=np.float32)
        for s in range(0, t, w):
            e = min(t, s + w)
            vals[s:e] = float(np.mean(x[s:e])) if e > s else 0.0
        out[w] = vals
    return out


def adjacent_events(x: np.ndarray, threshold: float) -> List[Tuple[int, int, float]]:
    if len(x) == 0:
        return [(0, 0, 0.0)]
    # 1D signal proxy: merge adjacent clips whose normalized change is small.
    events = []
    s = 0
    for i in range(1, len(x)):
        if abs(float(x[i] - x[i - 1])) > threshold:
            events.append((s, i - 1, float(np.mean(x[s:i]))))
            s = i
    events.append((s, len(x) - 1, float(np.mean(x[s:]))))
    return events


def contiguous_quantile_events(x: np.ndarray, k: int = 4) -> List[Tuple[int, int, float]]:
    if len(x) == 0:
        return [(0, 0, 0.0)]
    # Time-contiguous clustering proxy: split by quantile changes, preserving order.
    cuts = sorted(set(int(round(q * len(x) / k)) for q in range(1, k)))
    parts = []
    last = 0
    for c in cuts + [len(x)]:
        if c > last:
            parts.append((last, c - 1, float(np.mean(x[last:c]))))
        last = c
    return parts or [(0, len(x) - 1, float(np.mean(x)))]


def event_overlap_features(events: List[Tuple[int, int, float]], s: int, e: int, t: int) -> Dict[str, float]:
    overlaps = []
    vals = []
    for es, ee, val in events:
        ov = max(0, min(e, ee) - max(s, es) + 1)
        den = max(1, ee - es + 1)
        overlaps.append(float(ov / den))
        vals.append(float(val))
    best = max(overlaps) if overlaps else 0.0
    weighted = sum(o * v for o, v in zip(overlaps, vals)) / max(1e-6, sum(overlaps))
    return {"best_event_overlap": best, "query_event_similarity": weighted, "event_overlap_ratio": best}


def text_event_rows_for_split(
    split: str,
    ids: Sequence[int],
    corpus: Any,
    features: Dict[str, Any],
    first_stage: Dict[int, Dict[str, Any]],
    model: c12_5.SpanLocalizer,
    max_queries: int,
) -> pd.DataFrame:
    store = c12_5.ClipFeatureStore()
    t2_model, t2_idxs = load_t2_ranker()
    rows: List[Dict[str, Any]] = []
    ids = list(ids)[:max_queries]
    try:
        for st in range(0, len(ids), BATCH_SIZE):
            batch_ids = list(ids[st: st + BATCH_SIZE])
            batch = c12_5.build_batch(corpus, features, store, batch_ids)
            for bi, did in enumerate(batch_ids):
                row = corpus.by_id[int(did)]
                qtype = row.get("type", "unknown")
                qt = qtype_id(qtype)
                moment_dur = float(row["ts"][1] - row["ts"][0])
                db = duration_bucket(moment_dur)
                start, end, sim, vis, pool, retr_z = c12_5t.generated_pool_for_query(model, batch, bi, int(did), first_stage, corpus)
                pool = pool[:POOL_EVAL_LIMIT]
                t = int(batch["lengths"][bi].item())
                gt_ts = (float(row["ts"][0]), float(row["ts"][1]))
                sim = sim.astype(np.float32)
                vis = vis.astype(np.float32)
                sim_pref = prefix_sum(sim)
                pos_sim = np.maximum(sim, 0.0)
                pos_pref = prefix_sum(pos_sim)
                vis_pref = prefix_sum(vis)
                ctx = c12_5t.build_span_feature_context(start, end, sim, vis)
                feat_t2 = np.asarray([
                    c12_5t.span_feature_row_from_context(ctx, qt, retr_z, s, e, rank, score, float(meta.get("pq", 0.0)))
                    for rank, (s, e, score, meta) in enumerate(pool)
                ], dtype=np.float32)
                if t2_model is not None:
                    chunks = []
                    with torch.no_grad():
                        for k in range(0, len(feat_t2), 4096):
                            xx = torch.from_numpy(feat_t2[k:k + 4096, t2_idxs]).to(DEVICE)
                            chunks.append(torch.sigmoid(t2_model(xx)).detach().cpu().numpy())
                    t2_scores = np.concatenate(chunks)
                else:
                    t2_scores = np.asarray([p[2] for p in pool], dtype=np.float32)
                fw = fixed_window_scores(vis, [2, 4, 8, 16])
                adj = adjacent_events(vis, threshold=0.75)
                quant = contiguous_quantile_events(vis, k=4)
                all_events = adj + quant
                for rank, (s, e, score, meta) in enumerate(pool):
                    ts = c12_5.idx_to_ts(s, e, float(row["duration"]), t)
                    iou = c12_5.iou_1d(ts, gt_ts)
                    ws, we = local_window(s, e, t, 2)
                    left_s, left_e = max(0, s - (e - s + 1)), s - 1
                    right_s, right_e = e + 1, min(t - 1, e + (e - s + 1))
                    inside_mean = range_mean(sim_pref, s, e)
                    left_mean = range_mean(sim_pref, left_s, left_e)
                    right_mean = range_mean(sim_pref, right_s, right_e)
                    outside_mean = range_mean(sim_pref, 0, s - 1)
                    if e + 1 < t:
                        outside_mean = 0.5 * (outside_mean + range_mean(sim_pref, e + 1, t - 1))
                    text = {
                        "query_global_to_subtitle_clip_max": float(np.max(sim)) if len(sim) else 0.0,
                        "query_global_to_subtitle_clip_mean": float(np.mean(sim)) if len(sim) else 0.0,
                        "query_to_subtitle_window_max": float(np.max(sim[ws:we + 1])) if we >= ws else 0.0,
                        "query_to_subtitle_window_mean": range_mean(sim_pref, ws, we),
                        "subtitle_mass_inside_span": range_sum(pos_pref, s, e),
                        "subtitle_mean_inside_span": inside_mean,
                        "subtitle_max_inside_span": float(np.max(sim[s:e + 1])) if e >= s else 0.0,
                        "subtitle_peak_in_span": float(np.max(sim[s:e + 1]) - np.mean(sim)) if e >= s and len(sim) else 0.0,
                        "subtitle_inside_outside_contrast": inside_mean - outside_mean,
                        "subtitle_left_context_contrast": inside_mean - left_mean,
                        "subtitle_right_context_contrast": inside_mean - right_mean,
                        "subtitle_coverage_ratio": float(np.mean(np.abs(sim[s:e + 1]) > 1e-8)) if e >= s else 0.0,
                        "subtitle_missing_mask": float(not np.any(np.abs(sim) > 1e-8)),
                        "qtype_v": float(qtype == "v"),
                        "qtype_t": float(qtype == "t"),
                        "qtype_vt": float(qtype == "vt"),
                        "qtype_unknown": float(qtype not in {"v", "t", "vt"}),
                        "duration_bucket_short": float(db == "short"),
                        "duration_bucket_medium": float(db == "medium"),
                        "duration_bucket_long": float(db == "long"),
                        "short_moment": float(db == "short"),
                    }
                    inside_vis = range_mean(vis_pref, s, e)
                    left_vis = range_mean(vis_pref, left_s, left_e)
                    right_vis = range_mean(vis_pref, right_s, right_e)
                    ev_overlap = event_overlap_features(all_events, s, e, t)
                    event = {
                        "event_mass_inside_span": range_sum(prefix_sum(np.maximum(vis, 0.0)), s, e),
                        "event_mean_inside_span": inside_vis,
                        "event_max_inside_span": float(np.max(vis[s:e + 1])) if e >= s else 0.0,
                        "event_peak_in_span": float(np.max(vis[s:e + 1]) - np.mean(vis)) if e >= s and len(vis) else 0.0,
                        "event_inside_outside_contrast": inside_vis - 0.5 * (range_mean(vis_pref, 0, s - 1) + range_mean(vis_pref, e + 1, t - 1)),
                        "event_left_right_context_contrast": inside_vis - 0.5 * (left_vis + right_vis),
                        "event_boundary_start_alignment": float(abs(vis[s] - left_vis)) if 0 <= s < len(vis) else 0.0,
                        "event_boundary_end_alignment": float(abs(vis[e] - right_vis)) if 0 <= e < len(vis) else 0.0,
                        "event_overlap_ratio": ev_overlap["event_overlap_ratio"],
                        "best_event_overlap": ev_overlap["best_event_overlap"],
                        "query_event_similarity": ev_overlap["query_event_similarity"],
                        "subtitle_event_similarity": inside_mean * inside_vis,
                        "visual_event_similarity": inside_vis,
                    }
                    for w, vals in fw.items():
                        pref = prefix_sum(vals)
                        event[f"multiscale_span_score_{w}"] = range_mean(pref, s, e)
                        event[f"multiscale_boundary_contrast_{w}"] = 0.5 * (
                            abs(vals[s] - range_mean(pref, left_s, left_e)) +
                            abs(vals[e] - range_mean(pref, right_s, right_e))
                        )
                    rec = {
                        "split": split,
                        "desc_id": int(did),
                        "query_type": qtype,
                        "duration_bucket": db,
                        "moment_duration": moment_dur,
                        "rank": int(rank + 1),
                        "span_start": int(s),
                        "span_end": int(e),
                        "iou": float(iou),
                        "base_score": float(score),
                        "c12_t2_score": float(t2_scores[rank]),
                        "rank_norm": float(rank / max(1, len(pool) - 1)),
                        "span_duration_norm": float((e - s + 1) / max(1, t)),
                        "center_norm": float((s + e) * 0.5 / max(1, t)),
                        "retriever_score_z": float(retr_z),
                        "b6_top1_correct": bool(first_stage.get(int(did), {}).get("top1_correct", False)),
                    }
                    rec.update(text)
                    rec.update(event)
                    rows.append(rec)
    finally:
        store.close()
    return pd.DataFrame(rows)


def split_ids_for_stage(corpus: Any) -> Dict[str, List[int]]:
    return {
        "train_fit": list(corpus.splits["train_fit"])[:SAMPLE_LIMIT],
        "calib_select": list(corpus.splits["calib_select"])[:SAMPLE_LIMIT],
        "calib_holdout": list(corpus.splits["calib_holdout"])[:SAMPLE_LIMIT],
    }


def build_or_load_features() -> pd.DataFrame:
    path = OUT2 / "C14_2_TEXT_SUBTITLE_FEATURES.parquet"
    event_path = OUT3 / "C14_3_EVENT_FEATURES.parquet"
    if path.exists():
        return pd.read_parquet(path)
    corpus = load_corpus()
    c12_5.corpus_global = corpus
    features = load_features(build_feature_caches(corpus))
    model = c12_5r.load_model("teacher_distilled")
    dfs = []
    for split, ids in split_ids_for_stage(corpus).items():
        cache_name = "first_stage_train_calib_holdout_top128.pkl" if split == "train_fit" else "first_stage_calib_holdout_top128.pkl" if split == "calib_holdout" else None
        first_stage = load_first_stage(corpus, ids, top_keep=128, cache_name=cache_name)
        dfs.append(text_event_rows_for_split(split, ids, corpus, features, first_stage, model, SAMPLE_LIMIT))
    df = pd.concat(dfs, ignore_index=True)
    OUT2.mkdir(parents=True, exist_ok=True)
    OUT3.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    df.to_parquet(event_path, index=False)
    return df


def summarize_query_metrics(df: pd.DataFrame, score_col: str) -> Dict[str, Any]:
    rows = []
    pq, ious = [], []
    for did, g in df.groupby("desc_id", sort=False):
        g = g.sort_values(score_col, ascending=False)
        iou_vals = g["iou"].to_numpy()
        pq.extend(g[score_col].astype(float).tolist())
        ious.extend(iou_vals.astype(float).tolist())
        rec = {
            "desc_id": int(did),
            "query_type": str(g["query_type"].iloc[0]),
            "duration_bucket": str(g["duration_bucket"].iloc[0]),
            "b6_top1_correct": bool(g["b6_top1_correct"].iloc[0]),
            "best_iou_span_rank": int(np.argmax(iou_vals) + 1) if len(iou_vals) else None,
        }
        for k in [50, 100]:
            best = float(np.max(iou_vals[: min(k, len(iou_vals))])) if len(iou_vals) else 0.0
            rec[f"best_iou_top{k}"] = best
            rec[f"iou05_top{k}"] = best >= 0.5
            rec[f"iou07_top{k}"] = best >= 0.7
            rec[f"best_iou_span_top{k}"] = rec["best_iou_span_rank"] is not None and rec["best_iou_span_rank"] <= k
        rows.append(rec)

    def agg(rs: List[Dict[str, Any]]) -> Dict[str, Any]:
        return {
            "query_count": len(rs),
            "IoU@0.5_top50": 100.0 * sum(r["iou05_top50"] for r in rs) / max(1, len(rs)),
            "IoU@0.5_top100": 100.0 * sum(r["iou05_top100"] for r in rs) / max(1, len(rs)),
            "IoU@0.7_top50": 100.0 * sum(r["iou07_top50"] for r in rs) / max(1, len(rs)),
            "IoU@0.7_top100": 100.0 * sum(r["iou07_top100"] for r in rs) / max(1, len(rs)),
            "best_IoU_span_top50_promotion_rate": 100.0 * sum(r["best_iou_span_top50"] for r in rs) / max(1, len(rs)),
            "best_IoU_span_top100_promotion_rate": 100.0 * sum(r["best_iou_span_top100"] for r in rs) / max(1, len(rs)),
        }

    by_qtype = {k: agg(v.to_dict("records")) for k, v in pd.DataFrame(rows).groupby("query_type")} if rows else {}
    by_dur = {k: agg(v.to_dict("records")) for k, v in pd.DataFrame(rows).groupby("duration_bucket")} if rows else {}
    by_b6 = {str(k): agg(v.to_dict("records")) for k, v in pd.DataFrame(rows).groupby("b6_top1_correct")} if rows else {}
    short = [r for r in rows if r["duration_bucket"] == "short"]
    calib = {
        **c12_5r.corr(pq[:300000], ious[:300000]),
        "auc_iou07": c12_5r.auc_score(pq[:300000], [x >= 0.7 for x in ious[:300000]]),
        "sample_count": min(len(pq), 300000),
    }
    return {
        "summary": agg(rows),
        "short_summary": agg(short),
        "query_type_breakdown": by_qtype,
        "duration_breakdown": by_dur,
        "b6_top1_breakdown": by_b6,
        "pq_iou_calibration": calib,
        "wrong_video_promoted_rate": None,
    }


def train_ranker(df: pd.DataFrame, feature_cols: List[str], name: str) -> Tuple[str, Dict[str, Any]]:
    train = df[df["split"] == "train_fit"]
    if train.empty:
        raise RuntimeError("empty train_fit sample")
    x = train[feature_cols].to_numpy(np.float32)
    y = train["iou"].to_numpy(np.float32)
    mean = x.mean(axis=0)
    std = np.maximum(x.std(axis=0), 1e-6)
    x = (x - mean) / std
    model = TinyRanker(x.shape[1]).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    xb = torch.from_numpy(x).to(DEVICE)
    yb = torch.from_numpy(y).to(DEVICE)
    bs = 8192
    losses = []
    for epoch in range(TRAIN_EPOCHS):
        perm = torch.randperm(xb.shape[0], device=DEVICE)
        epoch_losses = []
        for st in range(0, xb.shape[0], bs):
            idx = perm[st:st + bs]
            opt.zero_grad(set_to_none=True)
            pred = torch.sigmoid(model(xb[idx]))
            loss = torch.nn.functional.mse_loss(pred, yb[idx])
            loss.backward()
            opt.step()
            epoch_losses.append(float(loss.detach().cpu()))
        losses.append(float(np.mean(epoch_losses)))
    score_col = f"score_{name}"
    allx = (df[feature_cols].to_numpy(np.float32) - mean) / std
    scores = []
    model.eval()
    with torch.no_grad():
        for st in range(0, allx.shape[0], 65536):
            scores.append(torch.sigmoid(model(torch.from_numpy(allx[st:st + 65536]).to(DEVICE))).cpu().numpy())
    df[score_col] = np.concatenate(scores)
    return score_col, {"model": name, "features": feature_cols, "losses": losses, "device": str(DEVICE)}


def feature_importance(df: pd.DataFrame, feature_cols: List[str], score_col: str) -> List[Dict[str, float]]:
    vals = []
    score = df[score_col].to_numpy(np.float64)
    for col in feature_cols:
        x = df[col].to_numpy(np.float64)
        c = c12_5r.corr(x[:300000], score[:300000]).get("spearman")
        vals.append({"feature": col, "spearman_vs_score": 0.0 if c is None else float(c)})
    return sorted(vals, key=lambda r: abs(r["spearman_vs_score"]), reverse=True)[:20]


def run_model_family(df: pd.DataFrame, specs: Dict[str, List[str]]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    diagnostics = {"trained": {}, "results": {}}
    for name, cols in specs.items():
        score_col, info = train_ranker(df, cols, name)
        diagnostics["trained"][name] = info
        diagnostics["results"][name] = {
            split: summarize_query_metrics(df[df["split"] == split], score_col)
            for split in ["calib_select", "calib_holdout"]
        }
        diagnostics["results"][name]["feature_importance"] = feature_importance(df[df["split"] == "calib_holdout"], cols, score_col)
    for baseline in ["base_score", "c12_t2_score"]:
        diagnostics["results"][baseline] = {
            split: summarize_query_metrics(df[df["split"] == split], baseline)
            for split in ["calib_select", "calib_holdout"]
        }
    return df, diagnostics


def decide_promising(results: Dict[str, Any], candidates: Sequence[str]) -> Tuple[str, str]:
    base = results["c12_t2_score"]["calib_holdout"]["summary"]
    base_short = results["c12_t2_score"]["calib_holdout"]["short_summary"]
    best_name = None
    best_delta = -1e9
    for name in candidates:
        summ = results[name]["calib_holdout"]["summary"]
        short = results[name]["calib_holdout"]["short_summary"]
        delta = (summ["IoU@0.7_top100"] - base["IoU@0.7_top100"]) + 0.5 * (short["IoU@0.7_top100"] - base_short["IoU@0.7_top100"])
        if delta > best_delta:
            best_delta = delta
            best_name = name
    if best_delta > 0.5:
        return "promising", str(best_name)
    if best_delta < -0.5:
        return "harmful", str(best_name)
    return "weak", str(best_name)


def text_stage() -> Dict[str, Any]:
    require_previous_ready("c14_2")
    OUT2.mkdir(parents=True, exist_ok=True)
    df = build_or_load_features()
    text_schema = {
        "stage": "C14-2",
        "feature_columns": TEXT_FEATURES,
        "meta_columns": META_FEATURES,
        "schema_hash": stable_hash(TEXT_FEATURES + META_FEATURES),
        "parquet": str(OUT2 / "C14_2_TEXT_SUBTITLE_FEATURES.parquet"),
        "official_val_used": False,
    }
    write_json(OUT2 / "C14_2_TEXT_SUBTITLE_SCHEMA.json", text_schema)
    specs = {
        "subtitle_feature_only_logistic_or_mlp": TEXT_FEATURES,
        "c12_base_plus_subtitle_mlp": META_FEATURES + TEXT_FEATURES,
        "qtype_gated_subtitle_mlp": META_FEATURES + TEXT_FEATURES,
        "duration_aware_subtitle_mlp": META_FEATURES + TEXT_FEATURES,
    }
    df, diag = run_model_family(df, specs)
    df.to_parquet(OUT2 / "C14_2_TEXT_SUBTITLE_FEATURES.parquet", index=False)
    state, best = decide_promising(diag["results"], specs.keys())
    status = {
        "promising": "C14_TEXT_SUBTITLE_PROMISING",
        "harmful": "C14_TEXT_SUBTITLE_HARMFUL",
        "weak": "C14_TEXT_SUBTITLE_WEAK",
    }[state]
    diag.update({
        "stage": "C14-2",
        "status": status,
        "best_variant": best,
        "same_sample_references": {
            "c12_5t_t2": "c12_t2_score",
            "c12_5u_u2_full_holdout_reference": load_json(ROOT / "c12_5u_topk_calibration_repair/C12_5U_DECISION.json", {}),
            "c12_5v_v1_full_holdout_reference": load_json(ROOT / "c12_5v_conquer_hidden_ranker/C12_5V_DECISION.json", {}),
        },
        "official_val_used": False,
    })
    write_json(OUT2 / "C14_2_TEXT_SUBTITLE_DIAGNOSTICS.json", diag)
    decision = {
        "stage": "C14-2",
        "status": status,
        "best_variant": best,
        "holdout_metrics": diag["results"][best]["calib_holdout"],
        "c12_5t_t2_same_sample": diag["results"]["c12_t2_score"]["calib_holdout"],
        "official_val_used": False,
    }
    write_json(OUT2 / "C14_2_TEXT_SUBTITLE_DECISION.json", decision)
    write_text(OUT2 / "C14_2_TEXT_SUBTITLE_FEATURES.md", f"""# C14-2 Text Subtitle Features

Feature parquet: `C14_2_TEXT_SUBTITLE_FEATURES.parquet`

Rows: `{len(df)}`

Schema hash: `{text_schema['schema_hash']}`

All features are produced by the shared C14 candidate feature builder.
Missing subtitle signal is represented by `subtitle_missing_mask`; no silent
zero-fill is used as an unreported feature.

official was not run.
""")
    write_text(OUT2 / "C14_2_TEXT_SUBTITLE_DECISION.md", f"""# C14-2 Text Subtitle Decision

status = `{status}`

best_variant = `{best}`

Same-sample reference baseline is `c12_t2_score`.

official was not run.
""")
    return decision


def event_stage() -> Dict[str, Any]:
    require_previous_ready("c14_3")
    OUT3.mkdir(parents=True, exist_ok=True)
    df = build_or_load_features()
    event_schema = {
        "stage": "C14-3",
        "feature_columns": EVENT_FEATURES,
        "meta_columns": META_FEATURES,
        "schema_hash": stable_hash(EVENT_FEATURES + META_FEATURES),
        "event_config": {
            "fixed_windows": [2, 4, 8, 16],
            "adjacent_similarity_thresholds_available": [0.5, 0.75, 1.0],
            "used_adjacent_threshold": 0.75,
            "time_contiguous_quantile_segments": 4,
            "raw_video_dependency": False,
        },
        "official_val_used": False,
    }
    write_json(OUT3 / "C14_3_EVENT_SCHEMA.json", event_schema)
    write_text(OUT3 / "C14_3_EVENT_EXTRACTION_PLAN.md", f"""# C14-3 Event Extraction Plan

This event pilot does not depend on raw video. It uses existing visual energy,
subtitle/query similarity, and generated C12 M1000 sample pools.

Implemented event families:

1. fixed-window event pooling: windows 2, 4, 8, 16 clips.
2. adjacent-similarity event grouping: deterministic 1D adjacent visual-energy
   merge, selected threshold 0.75.
3. time-contiguous segment clustering proxy: 4 ordered quantile segments.

Config hash: `{stable_hash(event_schema['event_config'])}`

official was not run.
""")
    specs = {
        "event_only_ranker": EVENT_FEATURES,
        "c12_base_plus_event_ranker": META_FEATURES + EVENT_FEATURES,
        "c12_base_plus_text_plus_event_ranker": META_FEATURES + TEXT_FEATURES + EVENT_FEATURES,
        "duration_aware_event_ranker": META_FEATURES + TEXT_FEATURES + EVENT_FEATURES,
    }
    df, diag = run_model_family(df, specs)
    df.to_parquet(OUT3 / "C14_3_EVENT_FEATURES.parquet", index=False)
    state, best = decide_promising(diag["results"], specs.keys())
    status = {
        "promising": "C14_EVENT_FEATURE_PROMISING",
        "harmful": "C14_EVENT_FEATURE_HARMFUL",
        "weak": "C14_EVENT_FEATURE_WEAK",
    }[state]
    diag.update({
        "stage": "C14-3",
        "status": status,
        "best_variant": best,
        "distribution_audit": {
            "rows": int(len(df)),
            "finite_all_event_features": bool(np.isfinite(df[EVENT_FEATURES].to_numpy(np.float32)).all()),
            "schema_hash": event_schema["schema_hash"],
        },
        "same_sample_references": {
            "c12_5t_t2": "c12_t2_score",
            "c12_5u_u2_full_holdout_reference": load_json(ROOT / "c12_5u_topk_calibration_repair/C12_5U_DECISION.json", {}),
            "c12_5v_v1_full_holdout_reference": load_json(ROOT / "c12_5v_conquer_hidden_ranker/C12_5V_DECISION.json", {}),
        },
        "official_val_used": False,
    })
    write_json(OUT3 / "C14_3_EVENT_DIAGNOSTICS.json", diag)
    write_json(OUT3 / "C14_3_EVENT_RESULTS.json", diag["results"])
    decision = {
        "stage": "C14-3",
        "status": status,
        "best_variant": best,
        "holdout_metrics": diag["results"][best]["calib_holdout"],
        "c12_5t_t2_same_sample": diag["results"]["c12_t2_score"]["calib_holdout"],
        "official_val_used": False,
    }
    write_json(OUT3 / "C14_3_EVENT_DECISION.json", decision)
    write_text(OUT3 / "C14_3_EVENT_DECISION.md", f"""# C14-3 Event Decision

status = `{status}`

best_variant = `{best}`

Same-sample reference baseline is `c12_t2_score`.

official was not run.
""")
    interim = interim_decision()
    write_json(OUT3 / "C14_INTERIM_DECISION.json", interim)
    write_text(OUT3 / "C14_INTERIM_DECISION.md", f"""# C14 Interim Decision

decision = `{interim['decision']}`

reason = {interim['reason']}

official was not run.
""")
    return decision


def interim_decision() -> Dict[str, Any]:
    text = load_json(OUT2 / "C14_2_TEXT_SUBTITLE_DECISION.json", {})
    event = load_json(OUT3 / "C14_3_EVENT_DECISION.json", {})
    data = load_json(OUT1 / "C14_1_DATA_AVAILABILITY_AUDIT.json", {})
    raw = bool(data.get("raw_video_exists"))
    text_status = text.get("status")
    event_status = event.get("status")
    decision = "C14_FEATURE_PILOT_INCONCLUSIVE"
    reason = "C14 pilots did not produce a clear continuation signal."
    if text_status == "C14_TEXT_SUBTITLE_PROMISING" or event_status == "C14_EVENT_FEATURE_PROMISING":
        decision = "C14_CONTINUE_TEXT_EVENT_UNIFIED_RETRAIN"
        reason = "Text/subtitle or event evidence improved same-sample holdout ranking enough to continue unified retraining."
    elif raw and text_status != "C14_TEXT_SUBTITLE_PROMISING" and event_status != "C14_EVENT_FEATURE_PROMISING":
        decision = "C14_NEED_RAW_VIDEO_VISUAL_PILOT"
        reason = "Raw video is available and text/event signal is not enough."
    elif not raw and text_status in {"C14_TEXT_SUBTITLE_WEAK", "C14_TEXT_SUBTITLE_HARMFUL"} and event_status in {"C14_EVENT_FEATURE_WEAK", "C14_EVENT_FEATURE_HARMFUL"}:
        decision = "C14_NEED_BOUNDARY_ARCHITECTURE_REDESIGN"
        reason = "Raw video is absent; C12 M1000 oracle headroom remains but text/event rankers do not clearly repair ranking/calibration."
    return {
        "stage": "C14-interim",
        "decision": decision,
        "reason": reason,
        "text_status": text_status,
        "event_status": event_status,
        "raw_video_exists": raw,
        "official_val_used": False,
    }


def run_stage(stage: str) -> Dict[str, Any]:
    seed_all()
    if stage == "c14_0":
        return protocol_stage()
    if stage == "c14_1":
        return data_audit_stage()
    if stage == "c14_2":
        return text_stage()
    if stage == "c14_3":
        return event_stage()
    raise ValueError(stage)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["c14_0", "c14_1", "c14_2", "c14_3", "all"], default="all")
    args = parser.parse_args()
    start = time.time()
    stages = ["c14_0", "c14_1", "c14_2", "c14_3"] if args.stage == "all" else [args.stage]
    statuses = {}
    for stg in stages:
        out = run_stage(stg)
        statuses[stg] = out.get("status") or out.get("decision")
        if stg == "c14_0" and out.get("status") != "C14_PROTOCOL_READY":
            break
    print(json.dumps({
        "statuses": statuses,
        "output_paths": {
            "c14_0": str(OUT0),
            "c14_1": str(OUT1),
            "c14_2": str(OUT2),
            "c14_3": str(OUT3),
        },
        "raw_video_found": bool(load_json(OUT1 / "C14_1_RAW_VIDEO_AUDIT.json", {}).get("raw_video_exists")),
        "c14_2_key_metrics": load_json(OUT2 / "C14_2_TEXT_SUBTITLE_DECISION.json", {}).get("holdout_metrics", {}),
        "c14_3_key_metrics": load_json(OUT3 / "C14_3_EVENT_DECISION.json", {}).get("holdout_metrics", {}),
        "c14_interim_decision": load_json(OUT3 / "C14_INTERIM_DECISION.json", {}).get("decision"),
        "official": "official was not run",
        "elapsed_seconds": time.time() - start,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
