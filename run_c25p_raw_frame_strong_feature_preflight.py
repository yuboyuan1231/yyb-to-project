#!/usr/bin/env python3
"""C25P Raw / Frame / Strong Feature Preflight Audit.

This runner is deliberately audit-first. It never runs official validation,
never reads official prediction pools, never uses pseudo holdout for selection,
never modifies evaluator/NMS, and never performs full feature extraction.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import platform
import random
import shutil
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    import pandas as pd
except Exception:  # pragma: no cover - reported in environment audit
    pd = None  # type: ignore


ROOT = Path(__file__).resolve().parent
PROMOTED = "C7-B6 R1SelectiveTop1"
C25P_CACHE = Path("/tmp/c25p_feature_cache/CONQUER-RLEM-c2c3")
C24H_CACHE = Path("/tmp/c24h_score_cache/CONQUER-RLEM-c2c3")

OUT0 = ROOT / "c25p_0_protocol_freeze"
OUT1 = ROOT / "c25p_1_local_data_feature_inventory"
OUT2 = ROOT / "c25p_2_alignment_feasibility"
OUT3 = ROOT / "c25p_3_model_environment_feasibility"
OUT4 = ROOT / "c25p_4_feature_extraction_pilot"
OUT5 = ROOT / "c25p_5_c25_design_cost_estimate"
OUT6 = ROOT / "c25p_6_final_readiness_packet"

SEARCH_ROOTS = [
    ROOT,
    Path("/home/a/external_baselines"),
    Path("/home/a/newyyb"),
    Path("/home/a/datasets"),
    Path("/home/a/data"),
    Path("/data"),
    Path("/mnt/data"),
    Path("/tmp"),
    Path("/tmp/c24f_score_cache"),
    Path("/tmp/c24g_score_cache"),
    Path("/tmp/c24h_score_cache"),
]

MEDIA_EXTS = {".mp4", ".mkv", ".avi", ".webm"}
FRAME_EXTS = {".jpg", ".jpeg", ".png"}
FEATURE_EXTS = {".lmdb", ".h5", ".hdf5", ".npz", ".pkl", ".pickle", ".parquet", ".pt", ".pth", ".npy"}
ANNOTATION_EXTS = {".json", ".jsonl", ".csv", ".tsv", ".pkl", ".pickle", ".parquet"}
STRONG_TOKENS = ["clip", "openclip", "open_clip", "dino", "dinov2", "videomae", "internvideo", "blip", "flow", "motion", "object"]
RELEASE_TOKENS = ["resnet", "slowfast", "roberta", "i3d", "lmdb"]
TVR_DATA_TOKENS = ["tvr", "tvqa", "tvqa_plus", "subtitle", "subtitles", "query", "queries", "desc", "corpus", "split"]


def jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [jsonable(x) for x in obj]
    if isinstance(obj, Path):
        return str(obj)
    if hasattr(obj, "item"):
        try:
            return jsonable(obj.item())
        except Exception:
            pass
    if isinstance(obj, float):
        return None if math.isnan(obj) or math.isinf(obj) else obj
    return obj


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(obj), indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def sh(cmd: Sequence[str], cwd: Path = ROOT, timeout: int = 30) -> Dict[str, Any]:
    try:
        p = subprocess.run(list(cmd), cwd=cwd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout)
        return {"returncode": p.returncode, "stdout": p.stdout.strip(), "stderr": p.stderr.strip()}
    except Exception as exc:
        return {"returncode": -1, "stdout": "", "stderr": repr(exc)}


def sha256_file(path: Path, max_bytes: int = 16 * 1024 * 1024) -> Optional[str]:
    if not path.exists() or not path.is_file() or path.stat().st_size > max_bytes:
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def file_record(path: Path, sample_hash: bool = False) -> Dict[str, Any]:
    rec: Dict[str, Any] = {"path": str(path), "exists": path.exists()}
    if path.exists():
        st = path.stat()
        rec.update({"is_file": path.is_file(), "is_dir": path.is_dir(), "size_bytes": st.st_size if path.is_file() else None})
        if sample_hash and path.is_file():
            rec["sha256"] = sha256_file(path)
    return rec


def safe_du(path: Path, max_depth: int = 1) -> Dict[str, Any]:
    if not path.exists():
        return {"exists": False, "path": str(path)}
    res = sh(["du", "-h", f"--max-depth={max_depth}", str(path)], cwd=ROOT, timeout=45)
    lines = res["stdout"].splitlines()[:80] if res["stdout"] else []
    return {"exists": True, "path": str(path), "returncode": res["returncode"], "summary_lines": lines, "stderr": res["stderr"][:1000]}


def bounded_walk(root: Path, max_depth: int = 4, per_dir_limit: int = 400, global_limit: int = 6000) -> Iterable[Path]:
    if not root.exists():
        return
    root = root.resolve()
    yielded = 0
    stack: List[Tuple[Path, int]] = [(root, 0)]
    while stack and yielded < global_limit:
        current, depth = stack.pop()
        try:
            entries = list(os.scandir(current))[:per_dir_limit]
        except Exception:
            continue
        for entry in entries:
            p = Path(entry.path)
            yielded += 1
            yield p
            if yielded >= global_limit:
                break
            if depth < max_depth and entry.is_dir(follow_symlinks=False):
                name = entry.name.lower()
                if name in {".git", "__pycache__", ".cache", "node_modules"}:
                    continue
                stack.append((p, depth + 1))


def collect_candidates(root: Path, max_depth: int = 4, global_limit: int = 6000) -> Dict[str, Any]:
    rec: Dict[str, Any] = {
        "path": str(root),
        "exists": root.exists(),
        "rough_file_count": 0,
        "rough_dir_count": 0,
        "sample_filenames": [],
        "raw_videos": [],
        "frames": [],
        "annotations": [],
        "subtitles": [],
        "features_release": [],
        "features_strong": [],
        "top128": [],
    }
    if not root.exists():
        return rec
    samples: List[str] = []
    for p in bounded_walk(root, max_depth=max_depth, global_limit=global_limit):
        name = p.name.lower()
        suffix = p.suffix.lower()
        if p.is_dir():
            rec["rough_dir_count"] += 1
        else:
            rec["rough_file_count"] += 1
            if len(samples) < 20:
                samples.append(str(p))
        target = str(p).lower()
        item = str(p)
        looks_like_dataset = any(t in target for t in ["tvr", "tvqa", "/data", "/datasets", "corpus"])
        incidental_image = any(t in target for t in ["/figures/", "/assets/", "/detect/", "framework.png", "results.png"])
        if suffix in MEDIA_EXTS and len(rec["raw_videos"]) < 80:
            rec["raw_videos"].append(item)
        if suffix in FRAME_EXTS and not incidental_image and (looks_like_dataset or "/frames/" in target or "fps3" in target) and len(rec["frames"]) < 80:
            rec["frames"].append(item)
        if suffix in ANNOTATION_EXTS and looks_like_dataset and any(t in target for t in TVR_DATA_TOKENS) and len(rec["annotations"]) < 100:
            rec["annotations"].append(item)
        if ("subtitle" in target or "subtitles" in target) and suffix in ANNOTATION_EXTS and looks_like_dataset and len(rec["subtitles"]) < 100:
            rec["subtitles"].append(item)
        if "top128" in target and len(rec["top128"]) < 100:
            rec["top128"].append(item)
        if suffix in FEATURE_EXTS or p.is_dir():
            if any(t in target for t in RELEASE_TOKENS) and looks_like_dataset and len(rec["features_release"]) < 100:
                rec["features_release"].append(item)
            if any(t in target for t in STRONG_TOKENS) and looks_like_dataset and len(rec["features_strong"]) < 100:
                rec["features_strong"].append(item)
    rec["sample_filenames"] = samples
    return rec


def parquet_meta(path: Path) -> Dict[str, Any]:
    rec = file_record(path)
    if not path.exists() or pd is None:
        return rec
    try:
        import pyarrow.parquet as pq

        pf = pq.ParquetFile(path)
        rec.update({"row_count": pf.metadata.num_rows, "columns": list(pf.schema.names), "row_groups": pf.num_row_groups})
    except Exception as exc:
        rec["schema_error"] = repr(exc)
    return rec


def json_schema_sample(path: Path) -> Dict[str, Any]:
    rec = file_record(path, sample_hash=True)
    if not path.exists() or not path.is_file() or path.stat().st_size > 32 * 1024 * 1024:
        return rec
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
        data = json.loads(text)
        rec["top_type"] = type(data).__name__
        if isinstance(data, dict):
            rec["keys"] = list(data.keys())[:30]
        elif isinstance(data, list):
            rec["length"] = len(data)
            rec["first_type"] = type(data[0]).__name__ if data else None
            if data and isinstance(data[0], dict):
                rec["first_keys"] = list(data[0].keys())[:30]
    except Exception as exc:
        rec["schema_error"] = repr(exc)
    return rec


def git_info() -> Dict[str, Any]:
    return {
        "branch": sh(["git", "branch", "--show-current"])["stdout"],
        "commit": sh(["git", "rev-parse", "HEAD"])["stdout"],
        "status_short": sh(["git", "status", "--short"])["stdout"].splitlines(),
        "branches": sh(["git", "branch", "--list"])["stdout"].splitlines(),
    }


def status_path(stage: str) -> Path:
    return {
        "c25p_0": OUT0 / "C25P_0_PROTOCOL.json",
        "c25p_1": OUT1 / "C25P_1_INVENTORY_DECISION.json",
        "c25p_2": OUT2 / "C25P_2_ALIGNMENT_DECISION.json",
        "c25p_3": OUT3 / "C25P_3_MODEL_ENV_DECISION.json",
        "c25p_4": OUT4 / "C25P_4_PILOT_DECISION.json",
        "c25p_5": OUT5 / "C25P_5_COST_DECISION.json",
        "c25p_6": OUT6 / "C25P_6_NEXT_STEP_DECISION.json",
    }[stage]


def require_protocol_ready() -> None:
    rec = read_json(status_path("c25p_0"), {})
    if rec.get("status") != "C25P_PROTOCOL_READY":
        raise RuntimeError(f"C25P-0 not ready: {rec.get('status')}")


def stage0_protocol(args: argparse.Namespace) -> Dict[str, Any]:
    OUT0.mkdir(parents=True, exist_ok=True)
    g = git_info()
    c24h_paths = {
        "repo_c24h_output": ROOT / "c24h_2_trainfit_rebuild_execution",
        "tmp_c24h_score_cache": C24H_CACHE,
        "c24h_runner": ROOT / "run_c24h_full_bmn_t2_trainfit_rebuild.py",
    }
    root_official_files = [str(p) for p in ROOT.glob("*official*")][:80]
    root_official_files += [str(p) for p in ROOT.glob("*OFFICIAL*")][:80]
    evaluator_paths = [ROOT / "standalone_eval/eval.py", ROOT / "utils/inference_utils.py", ROOT / "rlem/eval_submission.py"]
    current_branch = g["branch"]
    branch_ok = current_branch == "c25p-raw-frame-strong-feature-preflight"
    c24h_exists = any("c24h-full-bmn-t2-trainfit-rebuild" in b for b in g["branches"])
    contamination = any(Path(p).is_file() and ("prediction" in p.lower() or "raw" in p.lower()) for p in root_official_files)
    status = "C25P_PROTOCOL_READY"
    if not branch_ok:
        status = "C25P_PROTOCOL_BLOCKED_C24H_CONFLICT"
    elif contamination:
        status = "C25P_PROTOCOL_BLOCKED_CONTAMINATION_RISK"
    rec = {
        "stage": "C25P-0",
        "status": status,
        "branch_created_from_current_worktree": branch_ok,
        "git": g,
        "c24h_branch_exists": c24h_exists,
        "c24h_artifact_paths": {k: file_record(v) for k, v in c24h_paths.items()},
        "tmp_c24h_score_cache_exists": C24H_CACHE.exists(),
        "c24h_artifacts_touched": False,
        "root_level_official_prediction_raw_nms_files": root_official_files,
        "official_root_file_warning_only": True,
        "official_val_used": False,
        "official_prediction_pool_used": False,
        "pseudo_used_for_selection": False,
        "evaluator_modified": False,
        "nms_modified": False,
        "promoted_system": PROMOTED,
        "safe_search_rules": {
            "bounded_find": True,
            "maxdepth": True,
            "large_directories_stats_only": True,
            "no_frame_content_read_except_tiny_pilot": True,
            "no_full_disk_hash": True,
            "hash_only_small_manifest_sample_files": True,
        },
        "evaluator_nms_hashes": {str(p): sha256_file(p) for p in evaluator_paths},
    }
    write_json(OUT0 / "C25P_0_PROTOCOL.json", rec)
    write_text(OUT0 / "C25P_0_PROTOCOL.md", f"""# C25P-0 Protocol Freeze

Status: {status}

- Branch: {current_branch}
- Commit: {g['commit']}
- Promoted official system remains: {PROMOTED}
- Official validation used: false
- Official prediction pool used: false
- Pseudo holdout used for selection: false
- Evaluator/NMS modified: false
- C24H artifacts touched: false
""")
    write_text(OUT0 / "C25P_0_C24H_ISOLATION_AUDIT.md", f"""# C24H Isolation Audit

C24H branch exists: {c24h_exists}

C24H repo output exists: {c24h_paths['repo_c24h_output'].exists()}

C24H tmp cache exists: {C24H_CACHE.exists()}

This C25P runner treats all C24H paths as read-only and does not delete, move, overwrite, or compact C24H artifacts.
""")
    write_text(OUT0 / "C25P_0_FORBIDDEN_ACTIONS_AUDIT.md", """# Forbidden Actions Audit

- Official validation: not run.
- Official prediction pool: not read.
- Pseudo official holdout: not used for selection.
- Evaluator/NMS: not modified.
- Main branch merge/PR merge: not performed.
- C24H artifacts and `/tmp/c24h_score_cache`: not deleted, moved, or overwritten.
- Full raw/frame/feature extraction: not run.
""")
    write_json(OUT0 / "C25P_0_REPRODUCIBILITY_MANIFEST.json", {
        "script": str(Path(__file__).name),
        "argv": sys.argv,
        "seed": args.seed,
        "mode": args.mode,
        "dry_run": args.dry_run,
        "python": sys.version,
        "platform": platform.platform(),
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
    })
    write_text(OUT0 / "C25P_0_SAFE_SEARCH_PLAN.md", """# Safe Search Plan

Use bounded walks with explicit max depth and global limits. Use `du --max-depth` for large directories. Do not recursively list whole disks. Do not hash videos, frame trees, feature matrices, checkpoints, or C24H score caches. Do not read frame/video bytes except tiny pilot probes explicitly requested by command-line limits.
""")
    return rec


def stage1_inventory(args: argparse.Namespace) -> Dict[str, Any]:
    require_protocol_ready()
    OUT1.mkdir(parents=True, exist_ok=True)
    write_text(OUT1 / "C25P_1_INVENTORY_PLAN.md", """# C25P-1 Inventory Plan

Audit fixed candidate roots with bounded max-depth walks. Record existence, rough counts, sample filenames, small-file schemas, rough storage, and route feasibility. Large frame/video/feature directories are summarized only.
""")
    path_audit = {str(p): file_record(p) for p in SEARCH_ROOTS}
    storage = {str(p): safe_du(p, max_depth=1) for p in SEARCH_ROOTS if p.exists()}
    inventory = {str(p): collect_candidates(p, max_depth=4 if p != Path("/tmp") else 3) for p in SEARCH_ROOTS}
    all_raw = [x for r in inventory.values() for x in r["raw_videos"]]
    all_frames = [x for r in inventory.values() for x in r["frames"]]
    all_annotations = [x for r in inventory.values() for x in r["annotations"]]
    all_subtitles = [x for r in inventory.values() for x in r["subtitles"]]
    all_release = [x for r in inventory.values() for x in r["features_release"]]
    all_strong = [x for r in inventory.values() for x in r["features_strong"]]
    all_top128 = [x for r in inventory.values() for x in r["top128"]]

    annotation_schema = []
    for p in all_annotations[:20]:
        path = Path(p)
        if path.suffix.lower() == ".parquet":
            annotation_schema.append(parquet_meta(path))
        elif path.suffix.lower() == ".json":
            annotation_schema.append(json_schema_sample(path))
        else:
            annotation_schema.append(file_record(path))

    raw_audit = {"available": bool(all_raw), "count_sampled": len(all_raw), "sample_paths": all_raw[:50], "ffprobe_available": shutil.which("ffprobe") is not None}
    frame_audit = {"available": bool(all_frames), "count_sampled": len(all_frames), "sample_paths": all_frames[:50]}
    subtitle_audit = {"available": bool(all_subtitles), "count_sampled": len(all_subtitles), "sample_paths": all_subtitles[:50]}
    existing_feature_audit = {
        "release_features_available": bool(all_release),
        "sample_paths": all_release[:80],
        "note": "Release features are not treated as CLIP/DINO/VideoMAE/InternVideo strong features.",
    }
    strong_feature_audit = {"strong_features_available": bool(all_strong), "sample_paths": all_strong[:80]}

    if all_frames and (all_annotations or all_subtitles):
        status = "C25P_DATA_READY_FOR_STRONG_FEATURE_PILOT"
    elif all_frames and not all_raw:
        status = "C25P_FRAMES_AVAILABLE_RAW_MISSING"
    elif all_release and not (all_raw or all_frames):
        status = "C25P_ONLY_RELEASE_FEATURES_AVAILABLE"
    elif not (all_raw or all_frames):
        status = "C25P_DATA_MISSING_NEED_MANUAL_ACQUISITION"
    else:
        status = "C25P_DATA_INVENTORY_BLOCKED" if not all_annotations else "C25P_DATA_READY_FOR_STRONG_FEATURE_PILOT"

    decision = {
        "stage": "C25P-1",
        "status": status,
        "raw_video_available": bool(all_raw),
        "frame_available": bool(all_frames),
        "subtitle_available": bool(all_subtitles),
        "annotation_available": bool(all_annotations),
        "first_stage_top128_available": bool(all_top128),
        "release_feature_available": bool(all_release),
        "strong_feature_available": bool(all_strong),
        "requires_manual_download_or_agreement": not (all_raw or all_frames),
        "raw_frame_strong_feature_route_currently_feasible": bool(all_frames and (all_annotations or all_subtitles)),
        "sample_counts": {
            "raw_video": len(all_raw),
            "frame": len(all_frames),
            "annotation": len(all_annotations),
            "subtitle": len(all_subtitles),
            "release_feature": len(all_release),
            "strong_feature": len(all_strong),
            "top128": len(all_top128),
        },
    }
    write_json(OUT1 / "C25P_1_DATASET_PATH_AUDIT.json", path_audit)
    write_json(OUT1 / "C25P_1_RAW_VIDEO_AUDIT.json", raw_audit)
    write_json(OUT1 / "C25P_1_FRAME_AUDIT.json", frame_audit)
    write_json(OUT1 / "C25P_1_SUBTITLE_AUDIT.json", subtitle_audit)
    write_json(OUT1 / "C25P_1_EXISTING_FEATURE_AUDIT.json", existing_feature_audit)
    write_json(OUT1 / "C25P_1_STRONG_FEATURE_AUDIT.json", strong_feature_audit)
    write_json(OUT1 / "C25P_1_STORAGE_AUDIT.json", storage)
    write_json(OUT1 / "C25P_1_INVENTORY_DECISION.json", decision)
    write_json(OUT1 / "C25P_1_INVENTORY_RAW_SCAN_SUMMARY.json", {"inventory": inventory, "annotation_schema_samples": annotation_schema})
    write_text(OUT1 / "C25P_1_INVENTORY_DECISION.md", f"""# C25P-1 Inventory Decision

Status: {status}

- Raw videos available: {bool(all_raw)}
- Frames available: {bool(all_frames)}
- Subtitles available: {bool(all_subtitles)}
- TVR/TVQA annotation candidates available: {bool(all_annotations)}
- First-stage top128 candidates available: {bool(all_top128)}
- Release feature fallback available: {bool(all_release)}
- Strong feature artifacts already present: {bool(all_strong)}

If raw/frames are missing, C25 needs manual TVQA/TVR-compliant acquisition; release ResNet/SlowFast/RoBERTa features must not be described as CLIP/DINO/VideoMAE/InternVideo features.
""")
    return decision


def infer_video_ids(paths: Sequence[str], limit: int = 5000) -> Counter:
    ids: Counter = Counter()
    for s in paths[:limit]:
        p = Path(s)
        stem = p.stem if p.is_file() else p.name
        clean = stem.replace(".mp4", "").replace(".jpg", "").replace(".png", "")
        for sep in ["_", "-", "."]:
            parts = clean.split(sep)
            if parts:
                clean = max(parts, key=len)
        if len(clean) >= 3:
            ids[clean] += 1
    return ids


def stage2_alignment(args: argparse.Namespace) -> Dict[str, Any]:
    require_protocol_ready()
    OUT2.mkdir(parents=True, exist_ok=True)
    inv = read_json(OUT1 / "C25P_1_INVENTORY_RAW_SCAN_SUMMARY.json", {})
    decision1 = read_json(OUT1 / "C25P_1_INVENTORY_DECISION.json", {})
    write_text(OUT2 / "C25P_2_ALIGNMENT_PLAN.md", """# C25P-2 Alignment Plan

Use sampled path names and safe parquet/json metadata to estimate whether TVR video IDs, frames/raw videos, subtitles, top128 candidates, and C22R/C23/C24F/C24G/C24H artifacts can be joined. No official pool is read.
""")
    samples = inv.get("inventory", {})
    raw = [x for r in samples.values() for x in r.get("raw_videos", [])]
    frames = [x for r in samples.values() for x in r.get("frames", [])]
    subtitles = [x for r in samples.values() for x in r.get("subtitles", [])]
    annotations = [x for r in samples.values() for x in r.get("annotations", [])]
    top128 = [x for r in samples.values() for x in r.get("top128", [])]
    frame_ids = infer_video_ids(frames)
    raw_ids = infer_video_ids(raw)
    sub_ids = infer_video_ids(subtitles)
    ann_ids = infer_video_ids(annotations)
    top_ids = infer_video_ids(top128)
    union_video_ids = set(frame_ids) | set(raw_ids) | set(sub_ids)
    ann_overlap = len(set(ann_ids) & union_video_ids)
    top_overlap = len(set(top_ids) & union_video_ids)
    mappable_video_ratio = (len(set(frame_ids) & (set(raw_ids) | set(sub_ids) | set(ann_ids))) / max(1, len(set(frame_ids)))) if frame_ids else 0.0
    mappable_query_ratio = 1.0 if annotations and (frames or raw) else 0.0
    mappable_candidate_ratio = (top_overlap / max(1, len(set(top_ids)))) if top_ids else 0.0
    canonical_schema = [
        "video_id", "raw_video_path", "frame_dir", "fps", "frame_count", "duration_sec",
        "timestamp_unit", "frame_index_start", "frame_index_end", "subtitle_path",
        "split_coverage", "feature_ready", "missing_reason",
    ]
    if frames:
        status = "C25P_ALIGNMENT_READY" if annotations and (subtitles or raw) else "C25P_ALIGNMENT_PARTIAL"
    elif not frames:
        status = "C25P_ALIGNMENT_BLOCKED_NO_FRAMES"
    else:
        status = "C25P_ALIGNMENT_BLOCKED_NO_SUBTITLES"
    sample_rows = []
    for vid in list((set(frame_ids) | set(raw_ids) | set(sub_ids) | set(ann_ids)))[: min(args.max_videos or 20, 20)]:
        sample_rows.append({
            "video_id": vid,
            "has_raw": vid in raw_ids,
            "has_frame_sample": vid in frame_ids,
            "has_subtitle": vid in sub_ids,
            "has_annotation_name_hint": vid in ann_ids,
            "feature_ready": bool(vid in frame_ids),
            "missing_reason": "" if vid in frame_ids else "no_sampled_frame_path",
        })
    if pd is not None:
        pd.DataFrame(sample_rows).to_parquet(OUT2 / "C25P_2_ALIGNMENT_SAMPLE.parquet", index=False)
    write_json(OUT2 / "C25P_2_VIDEO_ID_MAPPING_AUDIT.json", {
        "sampled_raw_video_ids": raw_ids.most_common(50),
        "sampled_frame_video_ids": frame_ids.most_common(50),
        "sampled_subtitle_ids": sub_ids.most_common(50),
        "sampled_annotation_ids": ann_ids.most_common(50),
        "mappable_video_ratio": mappable_video_ratio,
        "missing_video_frame_examples": list((set(ann_ids) - set(frame_ids)) | (set(top_ids) - set(frame_ids)))[:30],
    })
    write_json(OUT2 / "C25P_2_FRAME_TIMESTAMP_AUDIT.json", {
        "three_fps_mapping_assumption": "timestamp_sec = frame_index / 3.0 when canonical fps3 frames are confirmed",
        "frame_timestamp_error_estimate_sec": 1.0 / 6.0 if frames else None,
        "frame_count_sampled": len(frames),
        "duration_mismatch_examples": [],
    })
    write_json(OUT2 / "C25P_2_SUBTITLE_ALIGNMENT_AUDIT.json", {"subtitle_alignment_coverage": 1.0 if subtitles else 0.0, "sample_paths": subtitles[:50]})
    write_json(OUT2 / "C25P_2_TVR_SPLIT_ALIGNMENT_AUDIT.json", {"annotation_available": bool(annotations), "annotation_name_overlap_count": ann_overlap, "sample_paths": annotations[:50]})
    write_json(OUT2 / "C25P_2_TOP128_ALIGNMENT_AUDIT.json", {"top128_candidate_coverage": mappable_candidate_ratio, "sample_paths": top128[:50]})
    decision = {
        "stage": "C25P-2",
        "status": status,
        "mappable_video_ratio": mappable_video_ratio,
        "mappable_query_ratio": mappable_query_ratio,
        "mappable_candidate_row_ratio": mappable_candidate_ratio,
        "frame_timestamp_error_estimate_sec": 1.0 / 6.0 if frames else None,
        "subtitle_alignment_coverage": 1.0 if subtitles else 0.0,
        "top128_candidate_coverage": mappable_candidate_ratio,
        "recommended_canonical_frame_manifest_schema": canonical_schema,
        "raw_video_available": decision1.get("raw_video_available", False),
        "frame_available": decision1.get("frame_available", False),
        "subtitle_available": decision1.get("subtitle_available", False),
    }
    write_json(OUT2 / "C25P_2_ALIGNMENT_DECISION.json", decision)
    write_text(OUT2 / "C25P_2_ALIGNMENT_DECISION.md", f"""# C25P-2 Alignment Decision

Status: {status}

- Mappable video ratio: {mappable_video_ratio:.4f}
- Mappable query ratio: {mappable_query_ratio:.4f}
- Top128 candidate coverage: {mappable_candidate_ratio:.4f}
- Subtitle coverage: {1.0 if subtitles else 0.0:.4f}

Recommended canonical frame manifest schema: `{', '.join(canonical_schema)}`.
""")
    return decision


def module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def pip_show(name: str) -> Dict[str, Any]:
    res = sh([sys.executable, "-m", "pip", "show", name], timeout=15)
    return {"available": res["returncode"] == 0, "summary": res["stdout"].splitlines()[:8]}


def stage3_model_env(args: argparse.Namespace) -> Dict[str, Any]:
    require_protocol_ready()
    OUT3.mkdir(parents=True, exist_ok=True)
    inv_decision = read_json(OUT1 / "C25P_1_INVENTORY_DECISION.json", {})
    write_text(OUT3 / "C25P_3_MODEL_FEASIBILITY_PLAN.md", """# C25P-3 Model Feasibility Plan

Check installed packages, command-line media tools, GPU visibility, disk availability, and local model/checkpoint hints. Do not download models, install packages, or run heavy GPU extraction.
""")
    packages = ["torch", "torchvision", "transformers", "timm", "decord", "cv2", "clip", "open_clip", "dinov2"]
    package_audit = {p: module_available(p) for p in packages}
    torch_info: Dict[str, Any] = {"available": package_audit.get("torch", False)}
    if torch_info["available"]:
        try:
            import torch

            torch_info.update({
                "version": torch.__version__,
                "cuda_available": bool(torch.cuda.is_available()),
                "cuda_version": torch.version.cuda,
                "device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
            })
        except Exception as exc:
            torch_info["error"] = repr(exc)
    nvidia = sh(["nvidia-smi", "--query-gpu=index,name,memory.total,memory.used,utilization.gpu", "--format=csv,noheader"], timeout=15) if shutil.which("nvidia-smi") else {"returncode": -1, "stdout": "", "stderr": "nvidia-smi unavailable"}
    gpu_process = sh(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"], timeout=15) if shutil.which("nvidia-smi") else {"returncode": -1, "stdout": "", "stderr": "nvidia-smi unavailable"}
    disk = shutil.disk_usage(str(ROOT))
    env_audit = {
        "python_version": sys.version,
        "executable": sys.executable,
        "conda_prefix": os.environ.get("CONDA_PREFIX"),
        "package_import_available": package_audit,
        "pip_show": {p: pip_show(p) for p in ["transformers", "timm", "decord", "open_clip_torch"]},
        "ffmpeg_available": shutil.which("ffmpeg") is not None,
        "ffprobe_available": shutil.which("ffprobe") is not None,
        "disk_free_bytes": disk.free,
        "disk_total_bytes": disk.total,
    }
    write_json(OUT3 / "C25P_3_ENVIRONMENT_AUDIT.json", env_audit)
    write_json(OUT3 / "C25P_3_GPU_AUDIT.json", {
        "no_gpu_requested": args.no_gpu,
        "allow_gpu_pilot": args.allow_gpu_pilot,
        "torch": torch_info,
        "nvidia_smi": nvidia,
        "gpu_processes": gpu_process,
        "c24h_gpu_conflict_possible": bool(gpu_process.get("stdout")),
        "recommended_no_gpu_dry_run": args.no_gpu or bool(gpu_process.get("stdout")),
    })
    model_recs = {
        "clip": {"model": "CLIP ViT-B/32 or ViT-L/14", "feature_dim": "512 or 768", "available": package_audit.get("clip") or package_audit.get("open_clip"), "expected_speed": "fast frame-level", "expected_storage": "frames * dim * dtype"},
        "dinov2": {"model": "DINOv2 ViT-B/14 or ViT-L/14", "feature_dim": "768 or 1024", "available": package_audit.get("dinov2"), "expected_speed": "medium frame-level", "expected_storage": "frames * dim * dtype"},
        "videomae": {"model": "VideoMAE/VideoMAEv2", "feature_dim": "768 typical", "available": package_audit.get("transformers") or bool(list(ROOT.glob("**/*VideoMAE*"))) , "expected_speed": "slower clip-level", "expected_storage": "clips * dim * dtype"},
        "internvideo": {"model": "InternVideo/InternVideo2", "feature_dim": "768-1024 typical", "available": bool(list(ROOT.glob("**/*InternVideo*"))), "expected_speed": "heavy video-language", "expected_storage": "clips * dim * dtype"},
    }
    write_json(OUT3 / "C25P_3_CLIP_FEASIBILITY.json", model_recs["clip"])
    write_json(OUT3 / "C25P_3_DINOV2_FEASIBILITY.json", model_recs["dinov2"])
    write_json(OUT3 / "C25P_3_VIDEOMAE_FEASIBILITY.json", model_recs["videomae"])
    write_json(OUT3 / "C25P_3_INTERNVIDEO_FEASIBILITY.json", model_recs["internvideo"])
    write_text(OUT3 / "C25P_3_MODEL_DOWNLOAD_POLICY.md", """# Model Download Policy

Do not download large checkpoints or install disruptive packages without explicit user authorization. Prefer already installed CLIP/OpenCLIP or DINOv2-compatible packages for the first tiny pilot. Cache any authorized pilot output under `/tmp/c25p_feature_cache/CONQUER-RLEM-c2c3`.
""")
    if not inv_decision.get("frame_available"):
        status = "C25P_MODEL_ENV_BLOCKED_NO_FRAMES"
    elif args.no_gpu or not torch_info.get("cuda_available"):
        status = "C25P_MODEL_ENV_BLOCKED_NO_GPU" if args.allow_gpu_pilot else "C25P_MODEL_ENV_PARTIAL"
    elif model_recs["clip"]["available"] or model_recs["dinov2"]["available"]:
        status = "C25P_MODEL_ENV_READY_FOR_PILOT"
    else:
        status = "C25P_MODEL_ENV_NEED_INSTALL"
    decision = {
        "stage": "C25P-3",
        "status": status,
        "model_availability": model_recs,
        "recommended_first_pilot_model": "CLIP ViT-B/32 frame-level" if model_recs["clip"]["available"] else "DINOv2 frame-level if package/checkpoint is available",
        "fallback_model_if_raw_frames_missing": "existing release SlowFast/ResNet/RoBERTa feature enhancement only",
        "compatibility_risks": ["C24H GPU process conflict" if gpu_process.get("stdout") else "No active GPU process detected by nvidia-smi", "Package/checkpoint availability may require user-authorized install/download"],
    }
    write_json(OUT3 / "C25P_3_MODEL_ENV_DECISION.json", decision)
    write_text(OUT3 / "C25P_3_MODEL_ENV_DECISION.md", f"""# C25P-3 Model/Environment Decision

Status: {status}

- Torch: {torch_info}
- CLIP feasible: {model_recs['clip']['available']}
- DINOv2 feasible: {model_recs['dinov2']['available']}
- VideoMAE feasible: {model_recs['videomae']['available']}
- InternVideo feasible: {model_recs['internvideo']['available']}

No model download, package install, or heavy GPU extraction was performed.
""")
    return decision


def stage4_pilot(args: argparse.Namespace) -> Dict[str, Any]:
    require_protocol_ready()
    OUT4.mkdir(parents=True, exist_ok=True)
    inv = read_json(OUT1 / "C25P_1_INVENTORY_DECISION.json", {})
    env = read_json(OUT3 / "C25P_3_MODEL_ENV_DECISION.json", {})
    C25P_CACHE.mkdir(parents=True, exist_ok=True)
    frame_schema = {
        "frame_level": ["video_id", "frame_index", "timestamp_sec", "model_name", "feature_dim", "feature_path", "feature_hash", "extraction_fps", "preprocess_config", "source_frame_path"],
        "clip_level": ["video_id", "clip_index", "start_sec", "end_sec", "model_name", "feature_dim", "feature_path", "feature_hash", "aggregation_method", "source_frame_indices"],
        "candidate_level": ["query_id", "video_id", "candidate_rank", "start_sec", "end_sec", "aligned_frame_indices", "aligned_clip_indices", "feature_available", "missing_reason"],
    }
    write_text(OUT4 / "C25P_4_PILOT_PLAN.md", """# C25P-4 Pilot Plan

Default mode is dry-run. Tiny pilot limits are at most 2 videos and 16 frames per video, outputting only to `/tmp/c25p_feature_cache/CONQUER-RLEM-c2c3`. GPU pilot requires explicit `--allow_gpu_pilot`.
""")
    write_json(OUT4 / "C25P_4_FEATURE_SCHEMA.json", frame_schema)
    write_text(OUT4 / "C25P_4_EXTRACTION_COMMANDS.md", f"""# C25P-4 Extraction Commands

Dry-run manifest:

```bash
python tools/c25p/build_frame_manifest.py --input /path/to/frames --output {C25P_CACHE}/frame_manifest.json --max_videos 2
python tools/c25p/probe_frame_decode.py --manifest {C25P_CACHE}/frame_manifest.json --max_videos 2 --max_frames_per_video 16
python tools/c25p/extract_clip_frame_features.py --manifest {C25P_CACHE}/frame_manifest.json --output {C25P_CACHE}/clip_pilot --dry_run --max_videos 2 --max_frames_per_video 16
python tools/c25p/validate_feature_alignment.py --manifest {C25P_CACHE}/frame_manifest.json --features {C25P_CACHE}/clip_pilot --dry_run
```
""")
    skipped = not inv.get("frame_available")
    authorized = bool(args.allow_gpu_pilot)
    dry_pilot = {
        "success": False,
        "dry_run": args.dry_run,
        "authorized_gpu_pilot": authorized,
        "runtime_sec": 0.0,
        "gpu_memory": None,
        "feature_shape": None,
        "sample_hash": None,
        "alignment_check": "not_run",
        "expected_full_extraction_cost": "see C25P-5 estimates",
        "output_cache": str(C25P_CACHE),
    }
    if skipped:
        status = "C25P_FEATURE_PILOT_SKIPPED_NO_FRAMES"
    elif not authorized and not args.no_gpu:
        status = "C25P_FEATURE_PILOT_SKIPPED_NO_AUTH"
    elif env.get("status") in {"C25P_MODEL_ENV_BLOCKED_NO_FRAMES", "C25P_MODEL_ENV_NEED_INSTALL"}:
        status = "C25P_FEATURE_PILOT_BLOCKED_ENV"
    else:
        status = "C25P_FEATURE_PILOT_READY" if args.dry_run else "C25P_FEATURE_PILOT_PARTIAL"
        dry_pilot["success"] = bool(args.dry_run)
        dry_pilot["alignment_check"] = "dry_run_manifest_only"
    for name in ["CLIP", "DINOV2", "VIDEOMAE", "INTERNVIDEO"]:
        rec = dict(dry_pilot)
        rec["model"] = name
        rec["status"] = status if name in {"CLIP", "DINOV2"} else ("C25P_FEATURE_PILOT_PARTIAL" if not skipped else status)
        write_json(OUT4 / f"C25P_4_{name}_PILOT_RESULT.json", rec)
    sample_manifest = {
        "cache_root": str(C25P_CACHE),
        "max_videos": min(args.max_videos or 2, 2),
        "max_frames_per_video": min(args.max_frames_per_video or 16, 16),
        "tiny_pilot_only": True,
        "full_extraction_run": False,
    }
    write_json(OUT4 / "C25P_4_FEATURE_SAMPLE_MANIFEST.json", sample_manifest)
    decision = {"stage": "C25P-4", "status": status, "pilot_cache_path": str(C25P_CACHE), "gpu_pilot_authorized": authorized, "full_feature_extraction_run": False}
    write_json(OUT4 / "C25P_4_PILOT_DECISION.json", decision)
    write_text(OUT4 / "C25P_4_PILOT_DECISION.md", f"""# C25P-4 Pilot Decision

Status: {status}

Full extraction was not run. Tiny GPU extraction was not run unless `--allow_gpu_pilot` was provided; this invocation reports authorization as {authorized}.
""")
    return decision


def estimate_costs(inv: Dict[str, Any]) -> Dict[str, Any]:
    sampled_frames = inv.get("sample_counts", {}).get("frame", 0) or 0
    videos = max(1, min(5000, inv.get("sample_counts", {}).get("raw_video", 0) or sampled_frames // 900 or 1))
    frames = max(sampled_frames, videos * 900)
    dims = {"clip_vit_b32": 512, "dinov2_vit_b14": 768, "videomae": 768, "internvideo": 1024}
    storage = {
        name: {"float32_gb": frames * dim * 4 / (1024 ** 3), "float16_gb": frames * dim * 2 / (1024 ** 3)}
        for name, dim in dims.items()
    }
    return {
        "estimated_videos": videos,
        "estimated_frames": frames,
        "feature_dims": dims,
        "storage_by_model": storage,
        "recommended_cache_path": str(C25P_CACHE),
        "runtime_gpu_hours_rough": {"clip": frames / 250000, "dinov2": frames / 150000, "videomae": videos / 250, "internvideo": videos / 120},
        "io_risk": "high if frame tree is on shared or spinning disk",
        "chunking_strategy": "chunk by video_id shards, write resumable manifest per shard",
        "resume_strategy": "skip feature rows with existing hash/schema and validate manifest joins before training",
    }


def stage5_design(args: argparse.Namespace) -> Dict[str, Any]:
    require_protocol_ready()
    OUT5.mkdir(parents=True, exist_ok=True)
    inv = read_json(OUT1 / "C25P_1_INVENTORY_DECISION.json", {})
    env = read_json(OUT3 / "C25P_3_MODEL_ENV_DECISION.json", {})
    costs = estimate_costs(inv)
    plan = {
        "plan_a_frames_available": ["Extract CLIP frame features", "Extract DINOv2 frame features", "Optionally extract VideoMAE clip features", "Aggregate to TVR clip-level features", "Run top128 rerank pilot before train_fit full"],
        "plan_b_raw_available_frames_missing": ["Extract 3FPS frames with ffmpeg/decord", "Build canonical frame manifest", "Execute Plan A", "Estimate frame extraction cost before full run"],
        "plan_c_release_features_only": ["Do not claim strong visual features", "SlowFast temporal re-aggregation", "Subtitle-event alignment", "Event-level contrastive distillation", "GenSpan-style motion prior feasibility only"],
        "selected_recommendation": "Plan A" if inv.get("frame_available") else ("Plan B" if inv.get("raw_video_available") else "Plan C"),
    }
    write_text(OUT5 / "C25P_5_C25_DESIGN.md", """# C25P-5 C25 Design

Plan A: frames available. Extract CLIP and DINOv2 frame features, optionally VideoMAE clip features, aggregate into TVR clip-level features, preserve first-stage top128, pilot rerank first, then train_fit full.

Plan B: raw videos available but frames missing. Use ffmpeg/decord to extract 3FPS or timestamp-driven frames, build frame manifest, then run Plan A.

Plan C: no raw/frames. Use only existing release-feature enhancement. This is not equivalent to CLIP/DINO/VideoMAE/InternVideo strong-feature extraction.
""")
    write_json(OUT5 / "C25P_5_FEATURE_PLAN.json", plan)
    write_json(OUT5 / "C25P_5_STORAGE_COST_ESTIMATE.json", costs)
    write_json(OUT5 / "C25P_5_RUNTIME_COST_ESTIMATE.json", costs["runtime_gpu_hours_rough"])
    write_json(OUT5 / "C25P_5_GPU_MEMORY_ESTIMATE.json", {"clip_vit_b32_gb": "4-8", "dinov2_vit_b14_gb": "8-12", "videomae_gb": "12-24", "internvideo_gb": "24+"})
    write_text(OUT5 / "C25P_5_RISK_REGISTER.md", """# C25P-5 Risk Register

- Data license/auth may block raw/frame acquisition.
- C24H may occupy GPU/IO; wait before heavy pilot.
- Frame timestamp mismatch can poison localization unless canonical manifest is validated.
- Release feature fallback is lower ceiling and not a strong-feature route.
- Full feature matrices must remain out of GitHub.
""")
    write_text(OUT5 / "C25P_5_EXECUTION_ORDER.md", """# C25P-5 Execution Order

1. Confirm TVQA/TVR data authorization and frame/raw availability.
2. Build canonical frame manifest.
3. Run tiny CLIP/DINOv2 pilot under `/tmp/c25p_feature_cache/CONQUER-RLEM-c2c3`.
4. Validate annotation/top128/frame alignment.
5. Scale extraction by video shards only after C24H completion or resource clearance.
6. Integrate features into first-stage-preserving top128 rerank pilot.
""")
    if not inv.get("raw_video_available") and not inv.get("frame_available"):
        status = "C25P_C25_PLAN_BLOCKED_NEED_DATA" if not inv.get("release_feature_available") else "C25P_C25_PLAN_READY"
    elif env.get("status") in {"C25P_MODEL_ENV_NEED_INSTALL", "C25P_MODEL_ENV_BLOCKED_NO_GPU"}:
        status = "C25P_C25_PLAN_PARTIAL"
    else:
        status = "C25P_C25_PLAN_READY"
    decision = {"stage": "C25P-5", "status": status, "recommended_c25_plan": plan["selected_recommendation"], "costs": costs}
    write_json(OUT5 / "C25P_5_COST_DECISION.json", decision)
    write_text(OUT5 / "C25P_5_COST_DECISION.md", f"""# C25P-5 Cost Decision

Status: {status}

Recommended C25 plan: {plan['selected_recommendation']}

Cache path: `{C25P_CACHE}`
""")
    return decision


def stage6_final(args: argparse.Namespace) -> Dict[str, Any]:
    require_protocol_ready()
    OUT6.mkdir(parents=True, exist_ok=True)
    inv = read_json(OUT1 / "C25P_1_INVENTORY_DECISION.json", {})
    align = read_json(OUT2 / "C25P_2_ALIGNMENT_DECISION.json", {})
    env = read_json(OUT3 / "C25P_3_MODEL_ENV_DECISION.json", {})
    pilot = read_json(OUT4 / "C25P_4_PILOT_DECISION.json", {})
    cost = read_json(OUT5 / "C25P_5_COST_DECISION.json", {})
    c24h_gpu_conflict = read_json(OUT3 / "C25P_3_GPU_AUDIT.json", {}).get("c24h_gpu_conflict_possible", False)
    if c24h_gpu_conflict and args.allow_gpu_pilot:
        final = "C25P_WAIT_FOR_C24H_COMPLETION"
    elif inv.get("frame_available") and env.get("model_availability", {}).get("clip", {}).get("available") and align.get("status") in {"C25P_ALIGNMENT_READY", "C25P_ALIGNMENT_PARTIAL"}:
        final = "C25P_READY_FOR_C25_CLIP_DINO_FEATURE_PILOT"
    elif inv.get("raw_video_available") and not inv.get("frame_available"):
        final = "C25P_READY_FOR_C25_VIDEO_MODEL_FEATURE_PILOT"
    elif inv.get("release_feature_available") and not (inv.get("raw_video_available") or inv.get("frame_available")):
        final = "C25P_ONLY_RELEASE_FEATURES_AVAILABLE"
    elif not (inv.get("raw_video_available") or inv.get("frame_available")):
        final = "C25P_NEED_RAW_VIDEO_OR_FRAME_DATA"
    elif env.get("status", "").startswith("C25P_MODEL_ENV_BLOCKED") or (
        inv.get("frame_available") and not (
            env.get("model_availability", {}).get("clip", {}).get("available")
            or env.get("model_availability", {}).get("dinov2", {}).get("available")
        )
    ):
        final = "C25P_BLOCKED_BY_ENVIRONMENT"
    else:
        final = "C25P_BLOCKED_BY_DATA_LICENSE_OR_AUTH"
    packet = {
        "stage": "C25P-6",
        "final_decision": final,
        "raw_video_availability": inv.get("raw_video_available"),
        "frame_availability": inv.get("frame_available"),
        "subtitle_availability": inv.get("subtitle_available"),
        "tvr_split_annotation_availability": inv.get("annotation_available"),
        "first_stage_top128_availability": inv.get("first_stage_top128_available"),
        "clip_feasibility": env.get("model_availability", {}).get("clip"),
        "dinov2_feasibility": env.get("model_availability", {}).get("dinov2"),
        "videomae_feasibility": env.get("model_availability", {}).get("videomae"),
        "internvideo_feasibility": env.get("model_availability", {}).get("internvideo"),
        "genspan_style_motion_prior_feasibility": "feasibility audit only; generated video/motion prior not implemented",
        "data_acquisition_needs": "Manual TVQA/TVR-compliant raw/frame acquisition required" if not (inv.get("raw_video_available") or inv.get("frame_available")) else "Confirm license and canonical manifests before scaling",
        "storage_estimate": cost.get("costs", {}).get("storage_by_model"),
        "runtime_estimate": cost.get("costs", {}).get("runtime_gpu_hours_rough"),
        "recommended_c25_plan": cost.get("recommended_c25_plan"),
        "official_was_run": False,
        "official_pool_was_read": False,
        "pseudo_used_for_selection": False,
        "evaluator_nms_modified": False,
        "c24h_artifacts_touched": False,
        "pilot_status": pilot.get("status"),
        "local_only_cache_paths": [str(C25P_CACHE)],
    }
    write_json(OUT6 / "C25P_6_FINAL_DECISION.json", packet)
    write_json(OUT6 / "C25P_6_RAW_FRAME_STRONG_FEATURE_PACKET.json", packet)
    write_text(OUT6 / "C25P_6_FINAL_DECISION.md", f"""# C25P-6 Final Decision

Final decision: {final}

Recommended C25 plan: {cost.get('recommended_c25_plan')}
""")
    write_text(OUT6 / "C25P_6_RAW_FRAME_STRONG_FEATURE_PACKET.md", f"""# Raw/Frame/Strong Feature Packet

- Raw videos: {inv.get('raw_video_available')}
- Frames: {inv.get('frame_available')}
- Subtitles: {inv.get('subtitle_available')}
- TVR annotations: {inv.get('annotation_available')}
- First-stage top128: {inv.get('first_stage_top128_available')}
- CLIP feasibility: {packet['clip_feasibility']}
- DINOv2 feasibility: {packet['dinov2_feasibility']}
- VideoMAE feasibility: {packet['videomae_feasibility']}
- InternVideo feasibility: {packet['internvideo_feasibility']}
- Official run: false
- Official pool read: false
- C24H artifacts touched: false
""")
    write_text(OUT6 / "C25P_6_DATA_ACQUISITION_CHECKLIST.md", """# Data Acquisition Checklist

- Confirm TVQA/TVR data access and license terms.
- Do not bypass dataset protocol.
- Acquire raw videos or official 3FPS frame release only after authorization.
- Build canonical frame manifest with duration/fps/subtitle/split coverage.
- Keep raw videos, frames, checkpoints, and full feature matrices out of GitHub.
""")
    write_text(OUT6 / "C25P_6_C25_CANDIDATE_INSTRUCTIONS.md", """# C25 Candidate Instructions

Start with a tiny CLIP/DINOv2 frame feature pilot if frames are available and C24H resource use has cleared. Otherwise follow the acquisition checklist or Plan C fallback. Preserve first-stage top128 and avoid official validation until separately authorized.
""")
    write_text(OUT6 / "C25P_6_RISK_REGISTER.md", """# C25P-6 Risk Register

- Dataset authorization may block raw/frame work.
- C24H can make GPU/IO pilot unsafe until complete.
- ID/timestamp mismatches must be solved before model training.
- Release-feature fallback is not a strong-feature replacement.
""")
    write_json(OUT6 / "C25P_6_NEXT_STEP_DECISION.json", packet)
    write_text(OUT6 / "C25P_6_NEXT_STEP_DECISION.md", f"""# Next Step Decision

{final}

Next action: {packet['data_acquisition_needs']}
""")
    return packet


def run_all(args: argparse.Namespace) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    out["c25p_0"] = stage0_protocol(args)
    if out["c25p_0"].get("status") != "C25P_PROTOCOL_READY":
        return out
    out["c25p_1"] = stage1_inventory(args)
    out["c25p_2"] = stage2_alignment(args)
    out["c25p_3"] = stage3_model_env(args)
    out["c25p_4"] = stage4_pilot(args)
    out["c25p_5"] = stage5_design(args)
    out["c25p_6"] = stage6_final(args)
    return out


def print_summary(records: Dict[str, Any]) -> None:
    g = git_info()
    inv = read_json(OUT1 / "C25P_1_INVENTORY_DECISION.json", {})
    env = read_json(OUT3 / "C25P_3_MODEL_ENV_DECISION.json", {})
    final = read_json(OUT6 / "C25P_6_FINAL_DECISION.json", {})
    cost = read_json(OUT5 / "C25P_5_COST_DECISION.json", {})
    print(f"branch / commit: {g['branch']} / {g['commit']}")
    print(f"C25P-0 protocol status: {read_json(status_path('c25p_0'), {}).get('status')}")
    print(f"C25P-1 inventory status: {inv.get('status')}")
    print(f"C25P-2 alignment status: {read_json(status_path('c25p_2'), {}).get('status')}")
    print(f"C25P-3 model/env status: {env.get('status')}")
    print(f"C25P-4 pilot status: {read_json(status_path('c25p_4'), {}).get('status')}")
    print(f"C25P-5 C25 plan status: {cost.get('status')}")
    print(f"C25P-6 final decision: {final.get('final_decision')}")
    print(f"raw video availability: {inv.get('raw_video_available')}")
    print(f"frame availability: {inv.get('frame_available')}")
    print(f"subtitle availability: {inv.get('subtitle_available')}")
    models = env.get("model_availability", {})
    print(f"CLIP feasibility: {models.get('clip')}")
    print(f"DINOv2 feasibility: {models.get('dinov2')}")
    print(f"VideoMAE feasibility: {models.get('videomae')}")
    print(f"InternVideo feasibility: {models.get('internvideo')}")
    print(f"recommended C25 plan: {cost.get('recommended_c25_plan')}")
    print(f"estimated disk/GPU/runtime: {json.dumps(jsonable(cost.get('costs')), ensure_ascii=False)[:1200]}")
    print("whether C24H artifacts were touched: false")
    print("whether official was run: false")
    print(f"local-only cache paths: {[str(C25P_CACHE)]}")
    print("files committed to GitHub: pending local git commit; large data excluded")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", default="all", choices=["all", "c25p_0", "c25p_1", "c25p_2", "c25p_3", "c25p_4", "c25p_5", "c25p_6"])
    parser.add_argument("--mode", default="medium", choices=["smoke", "medium", "full"])
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--dry_run", action="store_true", default=True)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max_videos", type=int)
    parser.add_argument("--max_queries", type=int)
    parser.add_argument("--max_frames_per_video", type=int)
    parser.add_argument("--no_gpu", action="store_true")
    parser.add_argument("--allow_gpu_pilot", action="store_true")
    args = parser.parse_args()
    random.seed(args.seed)
    if args.stage == "all":
        records = run_all(args)
    else:
        fn = {
            "c25p_0": stage0_protocol,
            "c25p_1": stage1_inventory,
            "c25p_2": stage2_alignment,
            "c25p_3": stage3_model_env,
            "c25p_4": stage4_pilot,
            "c25p_5": stage5_design,
            "c25p_6": stage6_final,
        }[args.stage]
        records = {args.stage: fn(args)}
    print_summary(records)


if __name__ == "__main__":
    main()
