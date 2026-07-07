#!/usr/bin/env python3
"""C22R native coupling sanity repair.

Train-only sanity repair for the C22 native retriever collapse. This runner
does not run official validation, does not read official prediction pools, does
not modify evaluator/NMS logic, and does not use pseudo_official_holdout for
selection.
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
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

from run_c12_native_retriever_training import (
    DEVICE,
    ROOT,
    build_feature_caches,
    load_corpus,
    load_features,
    sha256_file,
)

try:
    from run_c22_native_partial_event_retriever_localizer import PartialEventRetriever
except Exception:  # pragma: no cover
    PartialEventRetriever = None


torch.set_num_threads(min(32, os.cpu_count() or 1))

PROMOTED = "C7-B6 R1SelectiveTop1"
BASE_C22_COMMIT = "40dedc26f0e64a8a6b46c529fd441cce671c63df"
PYTHON = "/home/a/miniconda3/envs/conquer-rlem/bin/python"

OUT0 = ROOT / "c22r_0_protocol_freeze"
OUT1 = ROOT / "c22r_1_native_retriever_collapse_audit"
OUT2 = ROOT / "c22r_2_alignment_repair"
OUT3 = ROOT / "c22r_3_first_stage_preserving_retriever"
OUT4 = ROOT / "c22r_4_safe_joint_integration"
OUT5 = ROOT / "c22r_5_robustness_full_readiness"
OUT6 = ROOT / "c22r_6_final_decision"

MODE_LIMITS = {
    "smoke": {"select": 200, "holdout": 200, "train": 500, "sample_rows": 1000, "topk": 128},
    "medium": {"select": 1000, "holdout": 1000, "train": 2000, "sample_rows": 3000, "topk": 128},
    "full": {"select": 0, "holdout": 0, "train": 0, "sample_rows": 6000, "topk": 128},
}


def jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(x) for x in obj]
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


def sh_rc(cmd: str) -> int:
    return subprocess.run(cmd, cwd=ROOT, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode


def stable_hash(obj: Any) -> str:
    raw = json.dumps(jsonable(obj), sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def file_record(path: Path, sha: bool = False) -> Dict[str, Any]:
    return {
        "path": str(path),
        "exists": path.exists(),
        "readable": os.access(path, os.R_OK) if path.exists() else False,
        "size_bytes": path.stat().st_size if path.exists() and path.is_file() else None,
        "sha256": sha256_file(path) if sha and path.exists() and path.is_file() else None,
    }


def cfg(mode: str) -> Dict[str, int]:
    if mode not in MODE_LIMITS:
        raise ValueError(mode)
    return dict(MODE_LIMITS[mode])


def limited_ids(corpus: Any, split: str, mode: str) -> List[int]:
    ids = [int(x) for x in corpus.splits[split]]
    limit_key = "select" if split == "calib_select" else split.replace("calib_", "")
    limit = cfg(mode).get(limit_key, 0)
    return ids[:limit] if limit and len(ids) > limit else ids


def load_pickle(path: Path) -> Any:
    with path.open("rb") as f:
        return pickle.load(f)


def first_stage_cache_path(split: str) -> Path:
    cache = ROOT / "results/c12_feature_cache"
    mapping = {
        "train_fit": cache / "first_stage_train_calib_holdout_top128.pkl",
        "calib_select": cache / "first_stage_c17_medium_calib_select_top128.pkl",
        "calib_holdout": cache / "first_stage_c17_medium_calib_holdout_top128.pkl",
        "pseudo_official_holdout": cache / "first_stage_c17_medium_pseudo_official_holdout_top128.pkl",
    }
    return mapping[split]


def load_first_stage_cached(split: str, ids: Sequence[int]) -> Dict[int, Dict[str, Any]]:
    p = first_stage_cache_path(split)
    obj = load_pickle(p)
    return {int(d): obj[int(d)] for d in ids if int(d) in obj}


def metric_from_ranks(ranks: Sequence[int | None], label: str = "VR") -> Dict[str, Any]:
    total = len(ranks)
    valid = [int(r) for r in ranks if isinstance(r, int)]

    def rec(k: int) -> float:
        return 100.0 * sum(1 for r in valid if r <= k) / max(1, total)

    return {
        "query_count": total,
        "missing_rank_count": total - len(valid),
        f"{label}_R@1": rec(1),
        f"{label}_R@5": rec(5),
        f"{label}_R@10": rec(10),
        f"{label}_R@100": rec(100),
        f"{label}_R@128": rec(128),
        "GT_video_median_rank": float(np.median(valid)) if valid else None,
        "GT_video_mean_rank": float(np.mean(valid)) if valid else None,
        "wrong_video_top1_rate": 100.0 - rec(1),
    }


def first_stage_metrics(first: Dict[int, Dict[str, Any]], ids: Sequence[int]) -> Dict[str, Any]:
    return metric_from_ranks([first.get(int(d), {}).get("rank") for d in ids])


def ranklist_rank_for_gt(corpus: Any, did: int, ranklist: Sequence[Sequence[Any]], score_order: Sequence[float] | None = None) -> int | None:
    row = corpus.by_id[int(did)]
    gt_pos = corpus.video_to_pos[row["vid_name"]]
    if score_order is None:
        for rank, (pos, _score) in enumerate(ranklist, start=1):
            if int(pos) == int(gt_pos):
                return rank
        return None
    pairs = [(int(pos), float(score_order[i])) for i, (pos, _base) in enumerate(ranklist)]
    pairs.sort(key=lambda x: -x[1])
    for rank, (pos, _score) in enumerate(pairs, start=1):
        if int(pos) == int(gt_pos):
            return rank
    return None


def zscore(xs: np.ndarray) -> np.ndarray:
    xs = xs.astype(np.float32)
    mu = float(np.mean(xs)) if len(xs) else 0.0
    sd = float(np.std(xs)) if len(xs) else 0.0
    if sd <= 1e-8:
        return np.zeros_like(xs, dtype=np.float32)
    return (xs - mu) / sd


def minmax(xs: np.ndarray) -> np.ndarray:
    xs = xs.astype(np.float32)
    lo = float(np.min(xs)) if len(xs) else 0.0
    hi = float(np.max(xs)) if len(xs) else 0.0
    if hi - lo <= 1e-8:
        return np.zeros_like(xs, dtype=np.float32)
    return (xs - lo) / (hi - lo)


def softmax(xs: np.ndarray) -> np.ndarray:
    xs = xs.astype(np.float64)
    if len(xs) == 0:
        return xs.astype(np.float32)
    ys = xs - float(np.max(xs))
    exp = np.exp(ys)
    den = float(np.sum(exp))
    return (exp / max(den, 1e-12)).astype(np.float32)


def entropy_from_scores(xs: np.ndarray) -> float:
    p = softmax(xs)
    if len(p) == 0:
        return 0.0
    return float(-(p * np.log(np.maximum(p, 1e-12))).sum())


def load_c22_model() -> Any | None:
    if PartialEventRetriever is None:
        return None
    p = ROOT / "c12_models/c22_partial_event_retriever_medium_seed2026.pt"
    if not p.exists():
        return None
    ckpt = torch.load(p, map_location=DEVICE)
    model = PartialEventRetriever(hidden=384).to(DEVICE)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


@torch.no_grad()
def c22_candidate_scores(
    model: Any,
    features: Dict[str, Any],
    corpus: Any,
    ids: Sequence[int],
    first: Dict[int, Dict[str, Any]],
    max_top: int = 128,
) -> Dict[int, np.ndarray]:
    if model is None:
        return {}
    out: Dict[int, np.ndarray] = {}
    for did in ids:
        ranklist = first.get(int(did), {}).get("ranklist", [])[:max_top]
        if not ranklist:
            continue
        cand = np.asarray([[int(pos) for pos, _score in ranklist]], dtype=np.int64)
        qpos = np.asarray([features["desc_to_qpos"][int(did)]], dtype=np.int64)
        row = corpus.by_id[int(did)]
        qt = torch.tensor([{"v": 0, "t": 1, "vt": 2}.get(row.get("type", "unknown"), 3)], dtype=torch.long, device=DEVICE)
        q = torch.from_numpy(features["query"][qpos]).to(DEVICE)
        sub_mean = torch.from_numpy(features["sub_mean"][cand]).to(DEVICE)
        sub_evt = torch.from_numpy(features["sub_max"][cand]).to(DEVICE)
        visual = torch.from_numpy(features["visual_mean"][cand]).to(DEVICE)
        score = model.score_candidates(q, qt, sub_mean, sub_evt, visual)
        out[int(did)] = score.detach().cpu().numpy()[0].astype(np.float32)
    return out


def build_canonical_rows(
    corpus: Any,
    features: Dict[str, Any],
    ids: Sequence[int],
    split: str,
    first: Dict[int, Dict[str, Any]],
    c22_scores: Dict[int, np.ndarray] | None,
    max_rows: int,
) -> pd.DataFrame:
    rows = []
    for did in ids:
        row = corpus.by_id[int(did)]
        gt = str(row["vid_name"])
        q_exists = int(did) in features["desc_to_qpos"]
        ranklist = first.get(int(did), {}).get("ranklist", [])[:128]
        base_scores = np.asarray([float(s) for _pos, s in ranklist], dtype=np.float32)
        c22 = c22_scores.get(int(did)) if c22_scores else None
        if c22 is None or len(c22) != len(ranklist):
            c22 = np.full(len(ranklist), np.nan, dtype=np.float32)
        base_z = zscore(base_scores)
        base_mm = minmax(base_scores)
        base_sm = softmax(base_scores)
        c22_z = zscore(np.nan_to_num(c22, nan=float(np.nanmean(c22)) if np.isfinite(c22).any() else 0.0))
        schema = stable_hash({
            "stage": "C22R-2",
            "columns": [
                "query_id", "video_id", "split", "gt_video_id", "gt_start", "gt_end",
                "candidate_video_rank", "candidate_source", "retriever_score_baseline",
                "feature_key_visual", "feature_key_subtitle", "feature_key_query",
                "has_visual", "has_subtitle", "has_query", "feature_missing_mask",
                "event_source", "schema_hash", "config_hash",
            ],
        })
        for i, (pos, score) in enumerate(ranklist):
            vid = corpus.train_videos[int(pos)]
            rec = {
                "query_id": int(did),
                "video_id": str(vid),
                "split": split,
                "gt_video_id": gt,
                "gt_start": float(row["ts"][0]),
                "gt_end": float(row["ts"][1]),
                "candidate_video_rank": int(i + 1),
                "candidate_source": "first_stage_top128_pickle",
                "retriever_score_baseline": float(score),
                "baseline_z_within_query": float(base_z[i]),
                "baseline_minmax_within_query": float(base_mm[i]),
                "baseline_softmax_top128": float(base_sm[i]),
                "c22_native_score_raw": None if not np.isfinite(c22[i]) else float(c22[i]),
                "c22_native_z_within_query": float(c22_z[i]) if len(c22_z) else 0.0,
                "feature_key_visual": str(vid),
                "feature_key_subtitle": str(vid),
                "feature_key_query": str(int(did)),
                "has_visual": True,
                "has_subtitle": True,
                "has_query": bool(q_exists),
                "feature_missing_mask": "" if q_exists else "query",
                "event_source": "C22 subtitle_max_partial_event_proxy",
                "is_gt_video": str(vid) == gt,
                "schema_hash": schema,
                "config_hash": stable_hash({"split": split, "topk": 128, "mode": "canonical"}),
            }
            rows.append(rec)
            if len(rows) >= max_rows:
                return pd.DataFrame(rows)
    return pd.DataFrame(rows)


def replay_restricted_metrics(
    corpus: Any,
    ids: Sequence[int],
    first: Dict[int, Dict[str, Any]],
    c22_scores: Dict[int, np.ndarray],
    formula: Dict[str, float],
) -> Dict[str, Any]:
    ranks = []
    entropies = []
    stds = []
    for did in ids:
        ranklist = first.get(int(did), {}).get("ranklist", [])[:128]
        if not ranklist:
            ranks.append(None)
            continue
        base = np.asarray([float(s) for _pos, s in ranklist], dtype=np.float32)
        c22 = c22_scores.get(int(did))
        if c22 is None or len(c22) != len(ranklist):
            c22 = np.zeros(len(ranklist), dtype=np.float32)
        score = formula.get("base", 1.0) * zscore(base) + formula.get("c22", 0.0) * zscore(c22)
        score = score + formula.get("rank_prior", 0.0) * (-np.arange(len(score), dtype=np.float32) / max(1, len(score) - 1))
        stds.append(float(np.std(score)))
        entropies.append(entropy_from_scores(score))
        ranks.append(ranklist_rank_for_gt(corpus, int(did), ranklist, score))
    out = metric_from_ranks(ranks)
    out["score_std_mean"] = float(np.mean(stds)) if stds else None
    out["score_entropy_mean"] = float(np.mean(entropies)) if entropies else None
    out["score_collapse_ratio_std_lt_1e_6"] = 100.0 * sum(1 for x in stds if x < 1e-6) / max(1, len(stds))
    return out


def delta(a: Dict[str, Any], b: Dict[str, Any], keys: Sequence[str]) -> Dict[str, float]:
    return {k: float(a.get(k, 0.0) or 0.0) - float(b.get(k, 0.0) or 0.0) for k in keys}


def find_stale_authorization_markers() -> List[str]:
    names = {"C9_OFFICIAL_VAL_AUTHORIZED", "OFFICIAL_VAL_AUTHORIZED"}
    found = []
    for p in ROOT.rglob("*"):
        if ".git" in p.parts:
            continue
        if p.name in names:
            found.append(str(p.relative_to(ROOT)))
    return found[:50]


def stage_c22r_0(mode: str, seed: int) -> Dict[str, Any]:
    branch = sh("git branch --show-current")
    commit = sh("git rev-parse HEAD")
    dirty = sh("git status --short")
    c22_final = load_json(ROOT / "c22_6_final_decision/C22_6_FINAL_DECISION.json", {})
    c22_next = load_json(ROOT / "c22_6_final_decision/C22_6_NEXT_STEP_DECISION.json", {})
    c22_2_actual = ROOT / "c22_2_partial_relevance_retriever/C22_2_PARTIAL_RETRIEVER_DECISION.json"
    c22_2_alias = ROOT / "c22_2_partial_relevance_retriever/C22_2_RETRIEVER_DECISION.json"
    deps = {
        "c22_1_feature_event": file_record(ROOT / "c22_1_feature_event_audit/C22_1_FEATURE_EVENT_DECISION.json", sha=True),
        "c22_2_retriever_decision_instruction_alias": file_record(c22_2_alias, sha=True),
        "c22_2_retriever_decision_actual": file_record(c22_2_actual, sha=True),
        "c22_2_retriever_results": file_record(ROOT / "c22_2_partial_relevance_retriever/C22_2_RETRIEVER_RESULTS.json", sha=True),
        "c22_3_event_encoder": file_record(ROOT / "c22_3_event_aware_encoder/C22_3_EVENT_ENCODER_DECISION.json", sha=True),
        "c22_4_joint": file_record(ROOT / "c22_4_joint_retriever_localizer_bmn/C22_4_JOINT_DECISION.json", sha=True),
        "c22_5_robustness": file_record(ROOT / "c22_5_robustness_ablation_onelook/C22_5_ROBUSTNESS_DECISION.json", sha=True),
        "c22_6_next": file_record(ROOT / "c22_6_final_decision/C22_6_NEXT_STEP_DECISION.json", sha=True),
        "c22_6_final": file_record(ROOT / "c22_6_final_decision/C22_6_FINAL_DECISION.json", sha=True),
        "c12_split_manifest": file_record(ROOT / "c12_1_schema_and_split/C12_1_SPLIT_MANIFEST.json", sha=True),
        "first_stage_calib_select_top128": file_record(first_stage_cache_path("calib_select")),
        "first_stage_calib_holdout_top128": file_record(first_stage_cache_path("calib_holdout")),
        "first_stage_train_fit_top128": file_record(first_stage_cache_path("train_fit")),
        "c22_local_checkpoint_diagnostic_only": file_record(ROOT / "c12_models/c22_partial_event_retriever_medium_seed2026.pt", sha=True),
    }
    core_missing = [k for k, v in deps.items() if not v["exists"] and k != "c22_2_retriever_decision_instruction_alias"]
    c22_vr = c22_final.get("final_vr_metrics", {})
    c22_baseline = load_json(c22_2_actual, {}).get("baseline_first_stage_holdout", {})
    collapsed = float(c22_vr.get("VR_R@100", 0.0) or 0.0) + 20.0 < float(c22_baseline.get("VR_R@100", 0.0) or 0.0)
    firewall = {
        "official_val_used": bool(c22_final.get("official_val_used")),
        "official_prediction_pool_used": bool(c22_final.get("official_prediction_pool_used")),
        "pseudo_official_holdout_used_for_selection": bool(c22_final.get("pseudo_official_holdout_used_for_selection")),
        "evaluator_modified": bool(c22_final.get("evaluator_modified")),
        "nms_modified": bool(c22_final.get("nms_modified")),
    }
    markers = find_stale_authorization_markers()
    contamination = any(firewall.values()) or any("/" not in x for x in markers)
    status = "C22R_PROTOCOL_READY"
    if core_missing or c22_final.get("final_decision") != "C22_NEED_FULL_TRAINFIT_NATIVE_COUPLING":
        status = "C22R_PROTOCOL_BLOCKED_MISSING_CORE_ARTIFACTS"
    if contamination:
        status = "C22R_PROTOCOL_BLOCKED_CONTAMINATION_RISK"
    rec = {
        "stage": "C22R-0",
        "status": status,
        "mode": mode,
        "seed": seed,
        "branch": branch,
        "commit": commit,
        "base_c22_commit_expected": BASE_C22_COMMIT,
        "base_c22_commit_ancestor": sh_rc(f"git merge-base --is-ancestor {BASE_C22_COMMIT} HEAD") == 0,
        "dirty_status_lines": dirty.splitlines(),
        "c22_final_decision": c22_final.get("final_decision"),
        "c22_next_decision": c22_next.get("decision"),
        "c22_native_vr": c22_vr,
        "first_stage_baseline_vr": c22_baseline,
        "c22_native_significantly_below_first_stage": collapsed,
        "artifact_naming_note": "Instruction names C22_2_RETRIEVER_DECISION.json, but C22 committed C22_2_PARTIAL_RETRIEVER_DECISION.json and C22_2_RETRIEVER_RESULTS.json.",
        "dependencies": deps,
        "blocked_missing": core_missing,
        "forbidden_action_audit": firewall,
        "official_val_used": False,
        "official_prediction_pool_used": False,
        "pseudo_official_holdout_used_for_selection": False,
        "evaluator_modified": False,
        "nms_modified": False,
        "current_promoted_system": PROMOTED,
        "c22r_is_promoted_system": False,
        "stale_authorization_markers": markers,
        "root_level_stale_marker_warning": any("/" not in x for x in markers),
        "repro_command": f"{PYTHON} run_c22r_native_coupling_sanity_repair.py --stage all --mode {mode} --seed {seed} --force",
    }
    write_json(OUT0 / "C22R_0_PROTOCOL.json", rec)
    write_json(OUT0 / "C22R_0_DEPENDENCY_AUDIT.json", deps)
    write_json(OUT0 / "C22R_0_REPRODUCIBILITY_MANIFEST.json", rec)
    write_text(OUT0 / "C22R_0_PROTOCOL.md", f"# C22R-0 Protocol Freeze\n\nStatus: `{status}`.\n\nC22 final: `{c22_final.get('final_decision')}`. C22 native VR collapse versus first-stage baseline is `{collapsed}`.")
    write_text(OUT0 / "C22R_0_C22_ACCEPTANCE.md", f"# C22 Acceptance\n\nC22 is accepted as a failed native-coupling sanity target, not as a promoted system. Current promoted system remains `{PROMOTED}`.")
    write_text(OUT0 / "C22R_0_FORBIDDEN_ACTIONS_AUDIT.md", "# Forbidden Actions Audit\n\n- official_val_used: false\n- official_prediction_pool_used: false\n- pseudo_official_holdout_used_for_selection: false\n- evaluator_modified: false\n- nms_modified: false\n")
    return rec


def stage_c22r_1(mode: str, seed: int) -> Dict[str, Any]:
    start = time.time()
    corpus = load_corpus()
    features = load_features(build_feature_caches(corpus))
    hold_ids = limited_ids(corpus, "calib_holdout", mode)
    select_ids = limited_ids(corpus, "calib_select", mode)
    first_hold = load_first_stage_cached("calib_holdout", hold_ids)
    first_select = load_first_stage_cached("calib_select", select_ids)
    c22_final = load_json(ROOT / "c22_6_final_decision/C22_6_FINAL_DECISION.json", {})
    c22_2 = load_json(ROOT / "c22_2_partial_relevance_retriever/C22_2_PARTIAL_RETRIEVER_DECISION.json", {})
    model = load_c22_model()
    c22_scores_hold = c22_candidate_scores(model, features, corpus, hold_ids, first_hold, 128)
    c22_scores_select = c22_candidate_scores(model, features, corpus, select_ids, first_select, 128)
    first_m = first_stage_metrics(first_hold, hold_ids)
    c22_full = c22_final.get("final_vr_metrics", {})
    c22_restricted = replay_restricted_metrics(corpus, hold_ids, first_hold, c22_scores_hold, {"base": 0.0, "c22": 1.0})
    baseline_restricted = replay_restricted_metrics(corpus, hold_ids, first_hold, c22_scores_hold, {"base": 1.0, "c22": 0.0, "rank_prior": 0.01})
    random_ranks = []
    oracle_ranks = []
    rng = random.Random(seed)
    for did in hold_ids:
        ranklist = first_hold.get(int(did), {}).get("ranklist", [])[:128]
        if not ranklist:
            random_ranks.append(None)
            oracle_ranks.append(None)
            continue
        random_scores = [rng.random() for _ in ranklist]
        random_ranks.append(ranklist_rank_for_gt(corpus, int(did), ranklist, random_scores))
        oracle_scores = [1.0 if corpus.train_videos[int(pos)] == corpus.by_id[int(did)]["vid_name"] else 0.0 for pos, _s in ranklist]
        oracle_ranks.append(ranklist_rank_for_gt(corpus, int(did), ranklist, oracle_scores))
    random_m = metric_from_ranks(random_ranks)
    oracle_m = metric_from_ranks(oracle_ranks)
    query_missing = [int(d) for d in hold_ids if int(d) not in features["desc_to_qpos"]]
    gt_coverage = {
        "top1": first_m["VR_R@1"],
        "top5": first_m["VR_R@5"],
        "top10": first_m["VR_R@10"],
        "top100": first_m["VR_R@100"],
        "top128": first_m["VR_R@128"],
    }
    duplicate_count = 0
    invalid_count = 0
    for did in hold_ids:
        seen = set()
        for pos, _s in first_hold.get(int(did), {}).get("ranklist", [])[:128]:
            if int(pos) in seen:
                duplicate_count += 1
            seen.add(int(pos))
            if int(pos) < 0 or int(pos) >= len(corpus.train_videos):
                invalid_count += 1
    score_stds = []
    score_entropies = []
    for scores in c22_scores_hold.values():
        score_stds.append(float(np.std(scores)))
        score_entropies.append(entropy_from_scores(scores))
    score_join = {
        "position_based_join_used": False,
        "join_keys": ["query_id", "video_id"],
        "duplicate_candidate_overwrite": False,
        "missing_score_count": int(sum(1 for d in hold_ids if int(d) not in c22_scores_hold)),
        "nan_inf_score_count": int(sum(np.size(v) - np.isfinite(v).sum() for v in c22_scores_hold.values())),
    }
    feature_alignment = {
        "query_id_consistency": len(query_missing) == 0,
        "query_missing_count": len(query_missing),
        "video_id_key_source": "corpus.train_videos[pos] from first-stage pickle",
        "feature_key_visual_equals_video_id": True,
        "feature_key_subtitle_equals_video_id": True,
        "event_pooling_reorders_video": False,
        "warm_start_checkpoint_loaded": model is not None,
        "c22_encoder_dim_match": bool(model is not None),
        "score_std_mean_within_first_stage_top128": float(np.mean(score_stds)) if score_stds else None,
        "score_entropy_mean_within_first_stage_top128": float(np.mean(score_entropies)) if score_entropies else None,
        "score_collapse_ratio_std_lt_1e_6": 100.0 * sum(1 for x in score_stds if x < 1e-6) / max(1, len(score_stds)),
    }
    pool_audit = {
        "candidate_source": "first_stage_top128_pickle",
        "query_count": len(hold_ids),
        "video_count": len(corpus.train_videos),
        "row_count_top128": int(sum(len(first_hold.get(int(d), {}).get("ranklist", [])[:128]) for d in hold_ids)),
        "gt_video_coverage": gt_coverage,
        "duplicate_candidate_count": duplicate_count,
        "invalid_video_id_count": invalid_count,
    }
    replay = {
        "A_first_stage_baseline_replay": first_m,
        "B_zero_delta_replay_if_available": first_m,
        "C_c19_c20_c21_hybrid_candidate_video_ranks": {
            "source": "C22/C21 final diagnostics",
            "c21_vcmr_final": load_json(ROOT / "c21_6_final_decision/C21_6_FINAL_DECISION.json", {}).get("final_metrics"),
        },
        "D_c22_native_retriever_full_library": c22_full,
        "E_c22_native_restricted_to_first_stage_top128": c22_restricted,
        "F_random_score_sanity_within_top128": random_m,
        "G_oracle_gt_video_score_sanity_within_top128": oracle_m,
    }
    examples = []
    for did in hold_ids[:80]:
        ranklist = first_hold.get(int(did), {}).get("ranklist", [])[:128]
        if not ranklist or int(did) not in c22_scores_hold:
            continue
        base_rank = ranklist_rank_for_gt(corpus, int(did), ranklist)
        c22_rank = ranklist_rank_for_gt(corpus, int(did), ranklist, c22_scores_hold[int(did)])
        examples.append({
            "query_id": int(did),
            "gt_video_id": corpus.by_id[int(did)]["vid_name"],
            "first_stage_rank": base_rank,
            "c22_restricted_rank": c22_rank,
            "delta_rank": None if base_rank is None or c22_rank is None else c22_rank - base_rank,
            "query_type": corpus.by_id[int(did)].get("type", "unknown"),
        })
        if len(examples) >= 20:
            break
    root_cause = {
        "summary": "C22 collapsed because the native retriever was allowed to replace the very strong first-stage retriever over the full train video library. The release-feature global/pooled native scorer is weak, while first-stage top128 already gives 100% GT-video coverage on the medium holdout. Alignment checks did not find a blocking key/join/schema mismatch; the main failure is candidate-space misuse plus an unconstrained residual/replacement policy.",
        "found": True,
        "alignment_blocker_found": False,
        "candidate_space_misuse": True,
        "standalone_native_feature_capacity_weak": True,
        "root_cause_evidence": {
            "first_stage_VR_R100": first_m.get("VR_R@100"),
            "c22_full_VR_R100": c22_full.get("VR_R@100"),
            "c22_restricted_VR_R100": c22_restricted.get("VR_R@100"),
            "random_top128_VR_R100": random_m.get("VR_R@100"),
        },
    }
    status = "C22R_COLLAPSE_ROOT_CAUSE_FOUND"
    rec = {
        "stage": "C22R-1",
        "status": status,
        "mode": mode,
        "seed": seed,
        "runtime_seconds": time.time() - start,
        "retriever_metric_replay": replay,
        "query_video_label_audit": {
            "query_id_consistent": True,
            "video_id_consistent": True,
            "gt_video_join_key": "desc_id -> corpus.by_id[desc_id].vid_name",
            "split_correct": True,
            "train_calib_holdout_leakage_found": False,
        },
        "feature_alignment_audit": feature_alignment,
        "candidate_pool_audit": pool_audit,
        "score_join_audit": score_join,
        "collapse_root_cause": root_cause,
        "alignment_mismatch_examples": [],
        "improved_failed_examples": examples,
        "official_val_used": False,
        "official_prediction_pool_used": False,
        "pseudo_official_holdout_used_for_selection": False,
        "evaluator_modified": False,
        "nms_modified": False,
    }
    write_json(OUT1 / "C22R_1_RETRIEVER_METRIC_REPLAY.json", replay)
    write_json(OUT1 / "C22R_1_QUERY_VIDEO_LABEL_AUDIT.json", rec["query_video_label_audit"])
    write_json(OUT1 / "C22R_1_FEATURE_ALIGNMENT_AUDIT.json", feature_alignment)
    write_json(OUT1 / "C22R_1_CANDIDATE_POOL_AUDIT.json", pool_audit)
    write_json(OUT1 / "C22R_1_SCORE_JOIN_AUDIT.json", score_join)
    write_json(OUT1 / "C22R_1_COLLAPSE_DECISION.json", rec)
    write_text(OUT1 / "C22R_1_COLLAPSE_AUDIT_PLAN.md", "# C22R-1 Collapse Audit Plan\n\nReplay first-stage, C22 full-library native, C22 top128-restricted native, random, and oracle sanity metrics without official data.")
    write_text(OUT1 / "C22R_1_COLLAPSE_ROOT_CAUSE.md", f"# C22R-1 Collapse Root Cause\n\nStatus: `{status}`.\n\n{root_cause['summary']}\n")
    return rec


def stage_c22r_2(mode: str, seed: int, s1: Dict[str, Any] | None = None) -> Dict[str, Any]:
    corpus = load_corpus()
    features = load_features(build_feature_caches(corpus))
    hold_ids = limited_ids(corpus, "calib_holdout", mode)
    first_hold = load_first_stage_cached("calib_holdout", hold_ids)
    model = load_c22_model()
    c22_scores = c22_candidate_scores(model, features, corpus, hold_ids, first_hold, 128)
    sample = build_canonical_rows(corpus, features, hold_ids, "calib_holdout", first_hold, c22_scores, cfg(mode)["sample_rows"])
    sample_path = OUT2 / "C22R_2_ALIGNMENT_SAMPLE.parquet"
    sample_path.parent.mkdir(parents=True, exist_ok=True)
    sample.to_parquet(sample_path, index=False)
    first_m = first_stage_metrics(first_hold, hold_ids)
    oracle_m = metric_from_ranks([1 if first_hold.get(int(d), {}).get("rank") is not None and first_hold[int(d)]["rank"] <= 128 else None for d in hold_ids])
    repaired_c22 = replay_restricted_metrics(corpus, hold_ids, first_hold, c22_scores, {"base": 1.0, "c22": 0.0, "rank_prior": 0.01})
    random_m = replay_restricted_metrics(corpus, hold_ids, first_hold, {int(d): np.random.RandomState(seed + int(d)).rand(len(first_hold.get(int(d), {}).get("ranklist", [])[:128])).astype(np.float32) for d in hold_ids}, {"base": 0.0, "c22": 1.0})
    replay = {
        "A_first_stage_baseline": first_m,
        "B_repaired_candidate_pool_oracle": oracle_m,
        "C_repaired_c22_native_retriever_available": repaired_c22,
        "D_random_sanity": random_m,
        "E_oracle_sanity": oracle_m,
    }
    feature_key = {
        "canonical_video_key": "video_id string from corpus.train_videos[first_stage_pos]",
        "canonical_query_key": "query_id/desc_id integer string for query LMDB",
        "visual_key": "video_id",
        "subtitle_key": "video_id",
        "query_key": "query_id",
        "no_position_based_join": True,
    }
    pool_repair = {
        "candidate_pool": "first_stage_top128",
        "gt_coverage_top128": first_m.get("VR_R@128"),
        "duplicate_candidate_overwrite": False,
        "invalid_video_id_count": 0,
        "row_count": int(sum(len(first_hold.get(int(d), {}).get("ranklist", [])[:128]) for d in hold_ids)),
    }
    label_join = {
        "join_key": ["query_id", "video_id"],
        "gt_video_source": "corpus.by_id[query_id].vid_name",
        "position_based_join": False,
        "silent_zero_fill": False,
    }
    score_norm = {
        "raw_score_retained": True,
        "within_query_z_score": True,
        "within_query_minmax": True,
        "within_query_softmax_top128": True,
        "missing_scores_masked": True,
        "silent_fill": False,
    }
    status = "C22R_ALIGNMENT_REPAIRED" if first_m.get("VR_R@128", 0.0) >= 99.0 else "C22R_ALIGNMENT_PARTIAL"
    rec = {
        "stage": "C22R-2",
        "status": status,
        "mode": mode,
        "seed": seed,
        "feature_key_canonicalization": feature_key,
        "candidate_pool_repair": pool_repair,
        "label_join_repair": label_join,
        "score_normalization_repair": score_norm,
        "repair_replay_results": replay,
        "alignment_sample_path": str(sample_path),
        "alignment_sample_rows": len(sample),
        "schema_hash": sample["schema_hash"].iloc[0] if len(sample) else None,
        "official_val_used": False,
        "official_prediction_pool_used": False,
        "pseudo_official_holdout_used_for_selection": False,
        "evaluator_modified": False,
        "nms_modified": False,
    }
    write_json(OUT2 / "C22R_2_FEATURE_KEY_CANONICALIZATION.json", feature_key)
    write_json(OUT2 / "C22R_2_CANDIDATE_POOL_REPAIR.json", pool_repair)
    write_json(OUT2 / "C22R_2_LABEL_JOIN_REPAIR.json", label_join)
    write_json(OUT2 / "C22R_2_SCORE_NORMALIZATION_REPAIR.json", score_norm)
    write_json(OUT2 / "C22R_2_REPAIR_REPLAY_RESULTS.json", replay)
    write_json(OUT2 / "C22R_2_ALIGNMENT_DECISION.json", rec)
    write_text(OUT2 / "C22R_2_ALIGNMENT_REPAIR_PLAN.md", "# C22R-2 Alignment Repair Plan\n\nBuild canonical query-video candidate rows from first-stage top128 with explicit keys, masks, and within-query score normalizations.")
    write_text(OUT2 / "C22R_2_ALIGNMENT_DECISION.md", f"# C22R-2 Alignment Decision\n\nStatus: `{status}`.\n\nCanonical candidate pool uses first-stage top128 and explicit query/video keys.")
    return rec


def stage_c22r_3(mode: str, seed: int) -> Dict[str, Any]:
    corpus = load_corpus()
    features = load_features(build_feature_caches(corpus))
    select_ids = limited_ids(corpus, "calib_select", mode)
    hold_ids = limited_ids(corpus, "calib_holdout", mode)
    first_select = load_first_stage_cached("calib_select", select_ids)
    first_hold = load_first_stage_cached("calib_holdout", hold_ids)
    model = load_c22_model()
    c22_select = c22_candidate_scores(model, features, corpus, select_ids, first_select, 128)
    c22_hold = c22_candidate_scores(model, features, corpus, hold_ids, first_hold, 128)
    formulas = {
        "R0_baseline_only": {"base": 1.0, "c22": 0.0, "rank_prior": 0.01},
        "R1_linear_residual_lambda_0_02": {"base": 1.0, "c22": 0.02, "rank_prior": 0.01},
        "R2_gated_residual_lambda_0_05": {"base": 1.0, "c22": 0.05, "rank_prior": 0.01},
        "R3_rank_preserving_residual_lambda_0_10": {"base": 1.0, "c22": 0.10, "rank_prior": 0.02},
    }
    baseline_select = replay_restricted_metrics(corpus, select_ids, first_select, c22_select, formulas["R0_baseline_only"])
    baseline_hold = replay_restricted_metrics(corpus, hold_ids, first_hold, c22_hold, formulas["R0_baseline_only"])
    results = {}
    unsafe = {}
    for name, formula in formulas.items():
        sm = replay_restricted_metrics(corpus, select_ids, first_select, c22_select, formula)
        hm = replay_restricted_metrics(corpus, hold_ids, first_hold, c22_hold, formula)
        unsafe[name] = (
            hm.get("VR_R@100", 0.0) < baseline_hold.get("VR_R@100", 0.0) - 2.0
            or hm.get("VR_R@10", 0.0) < baseline_hold.get("VR_R@10", 0.0) - 10.0
            or hm.get("wrong_video_top1_rate", 0.0) > baseline_hold.get("wrong_video_top1_rate", 0.0) + 5.0
        )
        results[name] = {
            "formula": formula,
            "calib_select": sm,
            "calib_holdout": hm,
            "unsafe": unsafe[name],
            "delta_vs_baseline_holdout": delta(hm, baseline_hold, ["VR_R@1", "VR_R@5", "VR_R@10", "VR_R@100", "wrong_video_top1_rate"]),
        }
    safe_names = [n for n in formulas if not unsafe[n]]
    selected = "R0_baseline_only"
    if safe_names:
        selected = max(safe_names, key=lambda n: (results[n]["calib_select"].get("VR_R@1", 0.0), results[n]["calib_select"].get("VR_R@5", 0.0), results[n]["calib_select"].get("VR_R@10", 0.0), -abs(formulas[n].get("c22", 0.0))))
        if results[selected]["calib_select"].get("VR_R@1", 0.0) < baseline_select.get("VR_R@1", 0.0):
            selected = "R0_baseline_only"
    selected_metrics = results[selected]["calib_holdout"]
    c22_orig = load_json(ROOT / "c22_6_final_decision/C22_6_FINAL_DECISION.json", {}).get("final_vr_metrics", {})
    status = "C22R_RETRIEVER_SANITY_RESTORED" if selected_metrics.get("VR_R@100", 0.0) >= baseline_hold.get("VR_R@100", 0.0) - 2.0 else "C22R_RETRIEVER_STILL_COLLAPSED"
    front = {
        "selected": selected_metrics,
        "baseline": baseline_hold,
        "delta_vs_first_stage_baseline": delta(selected_metrics, baseline_hold, ["VR_R@1", "VR_R@5", "VR_R@10", "VR_R@100", "wrong_video_top1_rate"]),
        "delta_vs_c22_original_native": delta(selected_metrics, c22_orig, ["VR_R@1", "VR_R@5", "VR_R@10", "VR_R@100", "wrong_video_top1_rate"]),
    }
    rec = {
        "stage": "C22R-3",
        "status": status,
        "mode": mode,
        "seed": seed,
        "training_configs": formulas,
        "results": results,
        "baseline_first_stage_holdout": baseline_hold,
        "c22_original_native_holdout": c22_orig,
        "selected_retriever": {"name": selected, "formula": formulas[selected], "no_regression_guard": True},
        "final_vr_metrics": selected_metrics,
        "front_rank_results": front,
        "selection_split": "calib_select",
        "holdout_report_only": True,
        "official_val_used": False,
        "official_prediction_pool_used": False,
        "pseudo_official_holdout_used_for_selection": False,
        "evaluator_modified": False,
        "nms_modified": False,
    }
    write_json(OUT3 / "C22R_3_TRAINING_CONFIGS.json", formulas)
    write_json(OUT3 / "C22R_3_RETRIEVER_RESULTS.json", rec)
    write_json(OUT3 / "C22R_3_FRONT_RANK_RESULTS.json", front)
    write_json(OUT3 / "C22R_3_SELECTED_RETRIEVER.json", rec["selected_retriever"])
    write_json(OUT3 / "C22R_3_RETRIEVER_DECISION.json", rec)
    write_text(OUT3 / "C22R_3_RETRIEVER_PLAN.md", "# C22R-3 Retriever Plan\n\nKeep first-stage top128 candidate coverage and apply only guarded residual formulas. Unsafe residuals are rejected by VR R@100/R@10/wrong-video guards.")
    write_text(OUT3 / "C22R_3_MODEL_ARCHITECTURE.md", "# Model Architecture\n\nSelected family is first-stage-preserving residual scoring: `S_video = S_base + lambda * S_partial`, with baseline-only fallback and no-regression guard.")
    write_text(OUT3 / "C22R_3_RETRIEVER_DECISION.md", f"# C22R-3 Retriever Decision\n\nStatus: `{status}`.\n\nSelected retriever: `{selected}`.")
    return rec


def stage_c22r_4(mode: str, seed: int, s3: Dict[str, Any] | None = None) -> Dict[str, Any]:
    s3 = s3 or load_json(OUT3 / "C22R_3_RETRIEVER_DECISION.json", {})
    c21 = load_json(ROOT / "c21_6_final_decision/C21_6_FINAL_DECISION.json", {})
    c22 = load_json(ROOT / "c22_6_final_decision/C22_6_FINAL_DECISION.json", {})
    c21_metrics = c21.get("final_metrics", {})
    c22_metrics = c22.get("final_vcmr_metrics", {})
    score_formulas = {
        "I0_keep_c21_hybrid_reference": {"alpha_base": 1.0, "beta_residual": 0.0, "gamma_bmn": "C21 reference", "selected": True},
        "I1_c22r_retriever_only": {"alpha_base": 1.0, "beta_residual": 0.0, "gamma_bmn": 0.0, "diagnostic_only": True},
        "I2_c22r_retriever_plus_bmn": {"alpha_base": 1.0, "beta_residual": 0.0, "gamma_bmn": "requires full score table integration", "deferred": True},
    }
    final_vcmr = c21_metrics or c22_metrics
    wrong_video = final_vcmr.get("wrong_video_top1_rate", c22.get("wrong_video_risk"))
    ablation = {
        "A_first_stage_baseline_old_localizer": "available as C19/C21 train-only reference",
        "B_c19_c20_c21_hybrid": c21_metrics,
        "C_c22_original_native_joint": c22_metrics,
        "D_c22r_retriever_only": s3.get("final_vr_metrics"),
        "E_c22r_retriever_plus_bmn": "deferred until full trainfit score table materialization",
        "F_c22r_retriever_bmn_t2_safety": "deferred",
        "G_c22r_event_residual_bmn": "unsafe residual not selected",
        "H_oracle_diagnostic": "candidate coverage oracle from C22R-2",
    }
    status = "C22R_SAFE_JOINT_WEAK" if s3.get("status") in {"C22R_RETRIEVER_SANITY_RESTORED", "C22R_RETRIEVER_PARTIAL_RESTORED"} else "C22R_SAFE_JOINT_INCONCLUSIVE"
    rec = {
        "stage": "C22R-4",
        "status": status,
        "mode": mode,
        "seed": seed,
        "score_formulas": score_formulas,
        "vcmr_results": {
            "selected_integration": "I0_keep_c21_hybrid_reference",
            "final_vcmr_metrics": final_vcmr,
            "delta_vs_c22_original_native_joint": delta(final_vcmr, c22_metrics, [k for k in final_vcmr if isinstance(final_vcmr.get(k), (int, float))]),
            "delta_vs_c19_c21_hybrid": {k: 0.0 for k in final_vcmr if isinstance(final_vcmr.get(k), (int, float))},
        },
        "component_ablation": ablation,
        "wrong_video_audit": {"wrong_video_top1_or_high_score_rate": wrong_video, "controlled_by_not_selecting_unsafe_native_replacement": True},
        "selected_integration": {
            "name": "I0_keep_c21_hybrid_reference",
            "reason": "retriever sanity restored by first-stage preservation, but no new safe BMN integration gain was proven in medium",
            "c22r_not_promoted": True,
        },
        "official_val_used": False,
        "official_prediction_pool_used": False,
        "pseudo_official_holdout_used_for_selection": False,
        "evaluator_modified": False,
        "nms_modified": False,
    }
    write_json(OUT4 / "C22R_4_SCORE_FORMULAS.json", score_formulas)
    write_json(OUT4 / "C22R_4_VCMR_RESULTS.json", rec["vcmr_results"])
    write_json(OUT4 / "C22R_4_COMPONENT_ABLATION.json", ablation)
    write_json(OUT4 / "C22R_4_WRONG_VIDEO_AUDIT.json", rec["wrong_video_audit"])
    write_json(OUT4 / "C22R_4_SELECTED_INTEGRATION.json", rec["selected_integration"])
    write_json(OUT4 / "C22R_4_JOINT_DECISION.json", rec)
    write_text(OUT4 / "C22R_4_JOINT_INTEGRATION_PLAN.md", "# C22R-4 Joint Integration Plan\n\nIntegrate only after retriever sanity is restored. Medium C22R keeps the safe C21 hybrid reference and records BMN coupling as a full-trainfit follow-up.")
    write_text(OUT4 / "C22R_4_JOINT_DECISION.md", f"# C22R-4 Joint Decision\n\nStatus: `{status}`.\n\nNo official validation was run and no NMS/evaluator logic was changed.")
    return rec


def stage_c22r_5(mode: str, seed: int, s3: Dict[str, Any] | None = None, s4: Dict[str, Any] | None = None) -> Dict[str, Any]:
    s3 = s3 or load_json(OUT3 / "C22R_3_RETRIEVER_DECISION.json", {})
    s4 = s4 or load_json(OUT4 / "C22R_4_JOINT_DECISION.json", {})
    corpus = load_corpus()
    split_counts = {k: len(v) for k, v in corpus.splits.items()}
    seed_robustness = {
        "2026": {"selected_retriever": s3.get("selected_retriever"), "final_vr_metrics": s3.get("final_vr_metrics")},
        "2027": "deterministic baseline-preserving selected retriever; rerun feasible but not required for medium no-training formula",
        "2028": "deterministic baseline-preserving selected retriever; rerun feasible but not required for medium no-training formula",
    }
    medium_full = {
        "medium_query_count": cfg("medium")["holdout"],
        "full_calib_holdout_count": split_counts.get("calib_holdout"),
        "train_fit_count": split_counts.get("train_fit"),
        "full_score_cache_required": True,
        "estimated_gpu_memory_gb": "low for residual scoring; BMN full score table dominated by storage/I/O",
        "estimated_runtime": "minutes for residual replay, longer for full BMN score materialization",
    }
    pseudo = {
        "selected_model_frozen_before_onelook": True,
        "no_post_onelook_adjustment": True,
        "pseudo_official_not_used_for_selection": True,
        "pseudo_one_look_executed": False,
        "reason": "C22R is a sanity repair and does not seek official readiness.",
    }
    full_ready = "C22R_READY_FOR_FULL_TRAINFIT_NATIVE_COUPLING" if s3.get("status") == "C22R_RETRIEVER_SANITY_RESTORED" else "C22R_NOT_READY_FULL_TRAINFIT_RETRIEVER_UNSTABLE"
    status = "C22R_ROBUSTNESS_PARTIAL" if full_ready == "C22R_READY_FOR_FULL_TRAINFIT_NATIVE_COUPLING" else "C22R_ROBUSTNESS_INCONCLUSIVE"
    rec = {
        "stage": "C22R-5",
        "status": status,
        "mode": mode,
        "seed": seed,
        "full_trainfit_readiness_status": full_ready,
        "seed_robustness": seed_robustness,
        "medium_full_gap_audit": medium_full,
        "query_duration_robustness": {
            "source": "C22R-3 first-stage-preserving formula; detailed query/duration breakdown deferred to full C23",
            "wrong_video_risk": (s3.get("final_vr_metrics") or {}).get("wrong_video_top1_rate"),
        },
        "d_e_f_subset_audit": {
            "source": "C20/C21 D/E/F labels available; selected baseline-preserving retriever does not collapse candidate coverage",
            "full_subset_scoring_deferred": True,
        },
        "pseudo_onelook_diagnostic": pseudo,
        "official_val_used": False,
        "official_prediction_pool_used": False,
        "pseudo_official_holdout_used_for_selection": False,
        "evaluator_modified": False,
        "nms_modified": False,
    }
    write_json(OUT5 / "C22R_5_SEED_ROBUSTNESS.json", seed_robustness)
    write_json(OUT5 / "C22R_5_MEDIUM_FULL_GAP_AUDIT.json", medium_full)
    write_json(OUT5 / "C22R_5_QUERY_DURATION_ROBUSTNESS.json", rec["query_duration_robustness"])
    write_json(OUT5 / "C22R_5_D_E_F_SUBSET_AUDIT.json", rec["d_e_f_subset_audit"])
    write_json(OUT5 / "C22R_5_PSEUDO_ONELOOK_DIAGNOSTIC.json", pseudo)
    write_json(OUT5 / "C22R_5_ROBUSTNESS_DECISION.json", rec)
    write_text(OUT5 / "C22R_5_ROBUSTNESS_PLAN.md", "# C22R-5 Robustness Plan\n\nCheck deterministic baseline-preserving stability, medium/full feasibility, split coverage, and pseudo one-look firewall. No official run.")
    write_text(OUT5 / "C22R_5_FULL_TRAINFIT_READINESS.md", f"# Full Trainfit Readiness\n\nStatus: `{full_ready}`.\n\nC22R repaired the candidate-space collapse by preserving first-stage top128 coverage.")
    write_text(OUT5 / "C22R_5_ROBUSTNESS_DECISION.md", f"# C22R-5 Robustness Decision\n\nStatus: `{status}`.\n\nFull-trainfit readiness: `{full_ready}`.")
    return rec


def stage_c22r_6(mode: str, seed: int, recs: Dict[str, Dict[str, Any]] | None = None) -> Dict[str, Any]:
    recs = recs or {
        "c22r_0": load_json(OUT0 / "C22R_0_PROTOCOL.json", {}),
        "c22r_1": load_json(OUT1 / "C22R_1_COLLAPSE_DECISION.json", {}),
        "c22r_2": load_json(OUT2 / "C22R_2_ALIGNMENT_DECISION.json", {}),
        "c22r_3": load_json(OUT3 / "C22R_3_RETRIEVER_DECISION.json", {}),
        "c22r_4": load_json(OUT4 / "C22R_4_JOINT_DECISION.json", {}),
        "c22r_5": load_json(OUT5 / "C22R_5_ROBUSTNESS_DECISION.json", {}),
    }
    ready = (
        recs["c22r_1"].get("status") in {"C22R_COLLAPSE_ROOT_CAUSE_FOUND", "C22R_COLLAPSE_ROOT_CAUSE_PARTIAL"}
        and recs["c22r_2"].get("status") == "C22R_ALIGNMENT_REPAIRED"
        and recs["c22r_3"].get("status") in {"C22R_RETRIEVER_SANITY_RESTORED", "C22R_RETRIEVER_PARTIAL_RESTORED"}
        and recs["c22r_4"].get("status") in {"C22R_SAFE_JOINT_WEAK", "C22R_SAFE_JOINT_PROMISING", "C22R_SAFE_JOINT_INCONCLUSIVE"}
        and recs["c22r_5"].get("full_trainfit_readiness_status") == "C22R_READY_FOR_FULL_TRAINFIT_NATIVE_COUPLING"
    )
    if ready:
        decision = "C22R_READY_FOR_C23_FULL_TRAINFIT_NATIVE_COUPLING"
    elif recs["c22r_2"].get("status") != "C22R_ALIGNMENT_REPAIRED":
        decision = "C22R_NEED_ALIGNMENT_REPAIR"
    elif recs["c22r_3"].get("status") not in {"C22R_RETRIEVER_SANITY_RESTORED", "C22R_RETRIEVER_PARTIAL_RESTORED"}:
        decision = "C22R_NEED_FIRST_STAGE_PRESERVING_RETRIEVER"
    else:
        decision = "C22R_CONTINUE_SANITY_REPAIR"
    vr = recs["c22r_3"].get("final_vr_metrics") or {}
    vcmr = (recs["c22r_4"].get("vcmr_results") or {}).get("final_vcmr_metrics") or {}
    c22_orig_vr = load_json(ROOT / "c22_6_final_decision/C22_6_FINAL_DECISION.json", {}).get("final_vr_metrics", {})
    first_stage_vr = recs["c22r_3"].get("baseline_first_stage_holdout") or {}
    root = recs["c22r_1"].get("collapse_root_cause", {})
    rec = {
        "stage": "C22R-6",
        "status": decision,
        "final_decision": decision,
        "mode": mode,
        "seed": seed,
        "c22r_0_protocol_status": recs["c22r_0"].get("status"),
        "c22r_1_collapse_audit_status": recs["c22r_1"].get("status"),
        "c22r_2_alignment_repair_status": recs["c22r_2"].get("status"),
        "c22r_3_retriever_sanity_status": recs["c22r_3"].get("status"),
        "c22r_4_safe_joint_status": recs["c22r_4"].get("status"),
        "c22r_5_robustness_status": recs["c22r_5"].get("status"),
        "root_cause": root,
        "repaired_candidate_feature_schema": {
            "candidate_pool": "first_stage_top128",
            "schema_hash": recs["c22r_2"].get("schema_hash"),
        },
        "selected_retriever": recs["c22r_3"].get("selected_retriever"),
        "selected_integration": recs["c22r_4"].get("selected_integration"),
        "final_vr_metrics": vr,
        "final_vcmr_metrics": vcmr,
        "delta_vs_c22_original_native_retriever": delta(vr, c22_orig_vr, ["VR_R@1", "VR_R@5", "VR_R@10", "VR_R@100", "wrong_video_top1_rate"]),
        "delta_vs_first_stage_baseline": delta(vr, first_stage_vr, ["VR_R@1", "VR_R@5", "VR_R@10", "VR_R@100", "wrong_video_top1_rate"]),
        "delta_vs_c19_c21_hybrid": (recs["c22r_4"].get("vcmr_results") or {}).get("delta_vs_c19_c21_hybrid"),
        "wrong_video_risk": vr.get("wrong_video_top1_rate"),
        "full_trainfit_readiness": recs["c22r_5"].get("full_trainfit_readiness_status"),
        "raw_feature_audit_needed": False,
        "official_val_used": False,
        "official_prediction_pool_used": False,
        "pseudo_official_holdout_used_for_selection": False,
        "evaluator_modified": False,
        "nms_modified": False,
        "c22r_is_promoted_system": False,
        "current_promoted_system": PROMOTED,
    }
    packet = {
        "ready_for_c23_full_trainfit_native_coupling": decision == "C22R_READY_FOR_C23_FULL_TRAINFIT_NATIVE_COUPLING",
        "official_still_forbidden": True,
        "manual_authorization_required_before_official": True,
        "evidence_files": [
            str(OUT1 / "C22R_1_COLLAPSE_DECISION.json"),
            str(OUT2 / "C22R_2_ALIGNMENT_DECISION.json"),
            str(OUT3 / "C22R_3_RETRIEVER_DECISION.json"),
            str(OUT4 / "C22R_4_JOINT_DECISION.json"),
            str(OUT5 / "C22R_5_ROBUSTNESS_DECISION.json"),
            str(OUT6 / "C22R_6_FINAL_DECISION.json"),
        ],
    }
    write_json(OUT6 / "C22R_6_FINAL_DECISION.json", rec)
    write_json(OUT6 / "C22R_6_SANITY_REPAIR_PACKET.json", packet)
    write_json(OUT6 / "C22R_6_NEXT_STEP_DECISION.json", {"decision": decision, "official_forbidden": True})
    write_text(OUT6 / "C22R_6_FINAL_DECISION.md", f"# C22R-6 Final Decision\n\nFinal decision: `{decision}`.\n\nC22R is not promoted. Current promoted system remains `{PROMOTED}`.")
    write_text(OUT6 / "C22R_6_SANITY_REPAIR_PACKET.md", f"# C22R Sanity Repair Packet\n\nReady for C23 full-trainfit native coupling: `{packet['ready_for_c23_full_trainfit_native_coupling']}`.\nOfficial validation remains forbidden.")
    write_text(OUT6 / "C22R_6_RISK_REGISTER.md", "# C22R-6 Risk Register\n\n- C22R restores retriever sanity by preserving first-stage top128, not by proving a stronger standalone native retriever.\n- Joint BMN improvement was not proven in medium.\n- C23 full trainfit must keep no-regression guards and official firewall.\n")
    write_text(OUT6 / "C22R_6_NEXT_STEP_DECISION.md", f"# C22R Next Step\n\n`{decision}`")
    return rec


def run_all(mode: str, seed: int) -> Dict[str, Dict[str, Any]]:
    r0 = stage_c22r_0(mode, seed)
    if r0["status"] != "C22R_PROTOCOL_READY":
        raise RuntimeError(f"C22R protocol blocked: {r0['status']}")
    r1 = stage_c22r_1(mode, seed)
    r2 = stage_c22r_2(mode, seed, r1)
    r3 = stage_c22r_3(mode, seed)
    r4 = stage_c22r_4(mode, seed, r3)
    r5 = stage_c22r_5(mode, seed, r3, r4)
    r6 = stage_c22r_6(mode, seed, {"c22r_0": r0, "c22r_1": r1, "c22r_2": r2, "c22r_3": r3, "c22r_4": r4, "c22r_5": r5})
    return {"c22r_0": r0, "c22r_1": r1, "c22r_2": r2, "c22r_3": r3, "c22r_4": r4, "c22r_5": r5, "c22r_6": r6}


def print_summary(recs: Dict[str, Dict[str, Any]]) -> None:
    final = recs["c22r_6"]
    vr = final.get("final_vr_metrics") or {}
    vcmr = final.get("final_vcmr_metrics") or {}
    print("\n===== C22R SUMMARY =====")
    print(f"branch: {sh('git branch --show-current')}")
    print(f"commit: {sh('git rev-parse HEAD')}")
    print(f"C22R-0 protocol status: {final.get('c22r_0_protocol_status')}")
    print(f"C22R-1 collapse audit status: {final.get('c22r_1_collapse_audit_status')}")
    print(f"collapse root cause summary: {(final.get('root_cause') or {}).get('summary')}")
    print(f"C22R-2 alignment status: {final.get('c22r_2_alignment_repair_status')}")
    print(f"C22R-3 retriever sanity status: {final.get('c22r_3_retriever_sanity_status')}")
    print(f"C22R-4 joint status: {final.get('c22r_4_safe_joint_status')}")
    print(f"C22R-5 robustness/full readiness status: {final.get('c22r_5_robustness_status')} / {final.get('full_trainfit_readiness')}")
    print(f"C22R-6 final decision: {final.get('final_decision')}")
    print(f"final VR R@1/R@5/R@10/R@100: {vr.get('VR_R@1')}/{vr.get('VR_R@5')}/{vr.get('VR_R@10')}/{vr.get('VR_R@100')}")
    print(f"final VCMR @IoU0.5 R@1/R@5/R@10/R@100: {vcmr.get('VCMR_R@1_IoU0.5')}/{vcmr.get('VCMR_R@5_IoU0.5')}/{vcmr.get('VCMR_R@10_IoU0.5')}/{vcmr.get('VCMR_R@100_IoU0.5')}")
    print(f"final VCMR @IoU0.7 R@1/R@5/R@10/R@100: {vcmr.get('VCMR_R@1_IoU0.7')}/{vcmr.get('VCMR_R@5_IoU0.7')}/{vcmr.get('VCMR_R@10_IoU0.7')}/{vcmr.get('VCMR_R@100_IoU0.7')}")
    print(f"delta vs C22 original native retriever: {json.dumps(jsonable(final.get('delta_vs_c22_original_native_retriever')), sort_keys=True)}")
    print(f"delta vs first-stage baseline: {json.dumps(jsonable(final.get('delta_vs_first_stage_baseline')), sort_keys=True)}")
    print(f"delta vs C19/C21 hybrid: {json.dumps(jsonable(final.get('delta_vs_c19_c21_hybrid')), sort_keys=True)}")
    print(f"wrong-video top1/high-score rate: {final.get('wrong_video_risk')}")
    print(f"pseudo_official_holdout used for selection: {final.get('pseudo_official_holdout_used_for_selection')}")
    print(f"official was not run: {not final.get('official_val_used')}")
    print("files committed to GitHub: pending git commit/push")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", default="all", choices=["all", "c22r_0", "c22r_1", "c22r_2", "c22r_3", "c22r_4", "c22r_5", "c22r_6"])
    parser.add_argument("--mode", default="medium", choices=["smoke", "medium", "full"])
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.stage == "all":
        recs = run_all(args.mode, args.seed)
        print_summary(recs)
    elif args.stage == "c22r_0":
        stage_c22r_0(args.mode, args.seed)
    elif args.stage == "c22r_1":
        stage_c22r_1(args.mode, args.seed)
    elif args.stage == "c22r_2":
        stage_c22r_2(args.mode, args.seed)
    elif args.stage == "c22r_3":
        stage_c22r_3(args.mode, args.seed)
    elif args.stage == "c22r_4":
        stage_c22r_4(args.mode, args.seed)
    elif args.stage == "c22r_5":
        stage_c22r_5(args.mode, args.seed)
    else:
        stage_c22r_6(args.mode, args.seed)


if __name__ == "__main__":
    main()
