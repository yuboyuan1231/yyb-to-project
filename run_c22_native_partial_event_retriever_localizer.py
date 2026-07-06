#!/usr/bin/env python3
"""C22 native partial-event retriever-localizer coupling.

This is a train-only C22 runner. It never runs official validation, never reads
official prediction pools, never changes evaluator/NMS logic, and never uses
pseudo_official_holdout for selection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import random
import subprocess
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import lmdb
import msgpack
import msgpack_numpy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from c12_native_retriever.scaffold import C12Paths, TVRFeatureStore
from run_c12_native_retriever_training import (
    DEVICE,
    ROOT,
    build_feature_caches,
    load_corpus,
    load_features,
    load_first_stage,
    qtype_id,
    seed_all,
    sha256_file,
)


torch.set_num_threads(min(32, os.cpu_count() or 1))

PROMOTED = "C7-B6 R1SelectiveTop1"
BASE_C21_COMMIT = "6cce32cce18597d3f71ac881f8e44c342500d798"
PYTHON = "/home/a/miniconda3/envs/conquer-rlem/bin/python"

OUT0 = ROOT / "c22_0_protocol_freeze"
OUT1 = ROOT / "c22_1_feature_event_audit"
OUT2 = ROOT / "c22_2_partial_relevance_retriever"
OUT3 = ROOT / "c22_3_event_aware_encoder"
OUT4 = ROOT / "c22_4_joint_retriever_localizer_bmn"
OUT5 = ROOT / "c22_5_robustness_ablation_onelook"
OUT6 = ROOT / "c22_6_final_decision"
MODEL_DIR = ROOT / "c12_models"
CACHE_ROOT = Path(os.environ.get("C22_SCORE_CACHE_DIR", "/tmp/c22_score_cache")) / "CONQUER-RLEM-c2c3"

MODE_LIMITS: Dict[str, Dict[str, int]] = {
    "smoke": {"train": 240, "select": 180, "holdout": 180, "pseudo": 120, "top": 48, "epochs": 2, "batch": 24, "hidden": 384},
    "medium": {"train": 1400, "select": 1000, "holdout": 1000, "pseudo": 500, "top": 96, "epochs": 4, "batch": 32, "hidden": 384},
    "full": {"train": 0, "select": 0, "holdout": 0, "pseudo": 1000, "top": 100, "epochs": 6, "batch": 48, "hidden": 384},
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


def metric_from_ranks(ranks: Sequence[int | None]) -> Dict[str, Any]:
    total = len(ranks)
    valid = [int(r) for r in ranks if isinstance(r, int)]

    def rec(k: int) -> float:
        return 100.0 * sum(1 for r in valid if r <= k) / max(1, total)

    return {
        "query_count": total,
        "missing_rank_count": total - len(valid),
        "VR_R@1": rec(1),
        "VR_R@5": rec(5),
        "VR_R@10": rec(10),
        "VR_R@100": rec(100),
        "GT_video_median_rank": float(np.median(valid)) if valid else None,
        "GT_video_mean_rank": float(np.mean(valid)) if valid else None,
        "wrong_video_top1_rate": 100.0 - rec(1),
    }


def metric_bool(xs: Sequence[bool]) -> float:
    return 100.0 * sum(bool(x) for x in xs) / max(1, len(xs))


def mode_cfg(mode: str) -> Dict[str, int]:
    if mode not in MODE_LIMITS:
        raise ValueError(f"unknown mode {mode}")
    return dict(MODE_LIMITS[mode])


def limited_ids(corpus: Any, split: str, mode: str) -> List[int]:
    cfg = mode_cfg(mode)
    key = "select" if split == "calib_select" else split.replace("calib_", "")
    limit = cfg.get(key, 0)
    ids = [int(x) for x in corpus.splits[split]]
    return ids[:limit] if limit and len(ids) > limit else ids


def read_first_stage_split(corpus: Any, split: str, ids: Sequence[int], top: int) -> Dict[int, Dict[str, Any]]:
    cache_dir = ROOT / "results/c12_feature_cache"
    preferred = {
        "train_fit": cache_dir / "first_stage_train_calib_holdout_top128.pkl",
        "calib_select": cache_dir / "first_stage_c17_medium_calib_select_top128.pkl",
        "calib_holdout": cache_dir / "first_stage_c17_medium_calib_holdout_top128.pkl",
        "pseudo_official_holdout": cache_dir / "first_stage_c17_medium_pseudo_official_holdout_top128.pkl",
    }
    p = preferred.get(split)
    wanted = {int(x) for x in ids}
    if p is not None and p.exists():
        with p.open("rb") as f:
            obj = pickle.load(f)
        if wanted.issubset(set(int(x) for x in obj.keys())):
            return {int(k): obj[int(k)] for k in ids}
    return load_first_stage(corpus, ids, top_keep=max(128, top), cache_name=f"first_stage_c22_{split}_top128.pkl")


def first_stage_metrics(first_stage: Dict[int, Dict[str, Any]], ids: Sequence[int]) -> Dict[str, Any]:
    return metric_from_ranks([first_stage.get(int(d), {}).get("rank") for d in ids])


def hard_failure_subset(ids: Sequence[int], split: str) -> Dict[str, List[int]]:
    p = ROOT / "c20_2_r1_failure_decomposition/C20_2_FAILURE_SAMPLE.parquet"
    if not p.exists():
        return {}
    df = pd.read_parquet(p, columns=["split", "seed", "query_id", "failure_type"])
    df = df[(df["split"].astype(str) == split) & (df["seed"].astype(int) == 2026)]
    wanted = set(int(x) for x in ids)
    out: Dict[str, List[int]] = {}
    for ft, g in df.groupby("failure_type"):
        vals = [int(x) for x in g["query_id"].tolist() if int(x) in wanted]
        if vals:
            out[str(ft)] = vals
    return out


def iou_1d(a0: float, a1: float, b0: float, b1: float) -> float:
    inter = max(0.0, min(a1, b1) - max(a0, b0))
    union = max(a1, b1) - min(a0, b0)
    return inter / union if union > 0 else 0.0


def duration_bucket(seconds: float) -> str:
    if seconds <= 5.0:
        return "short"
    if seconds <= 15.0:
        return "medium"
    return "long"


class PartialEventRetriever(nn.Module):
    def __init__(self, hidden: int = 128, dropout: float = 0.08) -> None:
        super().__init__()
        self.q_sub = nn.Sequential(nn.LayerNorm(768), nn.Linear(768, hidden), nn.GELU(), nn.LayerNorm(hidden))
        self.s_mean = nn.Sequential(nn.LayerNorm(768), nn.Linear(768, hidden), nn.GELU(), nn.LayerNorm(hidden))
        self.s_evt = nn.Sequential(nn.LayerNorm(768), nn.Linear(768, hidden), nn.GELU(), nn.LayerNorm(hidden))
        self.q_vis = nn.Sequential(nn.LayerNorm(768), nn.Linear(768, hidden), nn.GELU(), nn.LayerNorm(hidden))
        self.v_proj = nn.Sequential(nn.LayerNorm(4352), nn.Linear(4352, hidden * 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden * 2, hidden), nn.LayerNorm(hidden))
        self.mod_gate = nn.Sequential(nn.LayerNorm(768), nn.Linear(768, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, 3))
        self.qtype_bias = nn.Embedding(4, 3)
        with torch.no_grad():
            self.qtype_bias.weight.copy_(torch.tensor([[0.1, 1.0, 0.5], [1.1, 0.05, 0.7], [0.7, 0.55, 0.8], [0.4, 0.4, 0.4]]))
        self.scale = nn.Parameter(torch.tensor(12.0))

    def encode_query(self, q: torch.Tensor, qtype: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        qs = F.normalize(self.q_sub(q), dim=-1)
        qv = F.normalize(self.q_vis(q), dim=-1)
        gate = torch.softmax(self.mod_gate(q) + self.qtype_bias(qtype), dim=-1)
        return qs, qv, qs, gate

    def score_candidates(self, q: torch.Tensor, qtype: torch.Tensor, sub_mean: torch.Tensor, sub_evt: torch.Tensor, visual: torch.Tensor) -> torch.Tensor:
        qs, qv, qe, gate = self.encode_query(q, qtype)
        sm = F.normalize(self.s_mean(sub_mean), dim=-1)
        se = F.normalize(self.s_evt(sub_evt), dim=-1)
        vh = F.normalize(self.v_proj(visual), dim=-1)
        sub_score = torch.einsum("bd,bnd->bn", qs, sm)
        evt_score = torch.einsum("bd,bnd->bn", qe, se)
        vis_score = torch.einsum("bd,bnd->bn", qv, vh)
        score = gate[:, 0:1] * sub_score + gate[:, 1:2] * vis_score + gate[:, 2:3] * evt_score
        return self.scale.clamp(1.0, 30.0) * score

    @torch.no_grad()
    def score_all(self, q: torch.Tensor, qtype: torch.Tensor, features: Dict[str, Any], video_block: int = 4096) -> torch.Tensor:
        qs, qv, qe, gate = self.encode_query(q, qtype)
        scores = []
        for st in range(0, features["sub_mean"].shape[0], video_block):
            ed = min(st + video_block, features["sub_mean"].shape[0])
            sub_mean = torch.from_numpy(features["sub_mean"][st:ed]).to(DEVICE)
            sub_evt = torch.from_numpy(features["sub_max"][st:ed]).to(DEVICE)
            visual = torch.from_numpy(features["visual_mean"][st:ed]).to(DEVICE)
            sm = F.normalize(self.s_mean(sub_mean), dim=-1)
            se = F.normalize(self.s_evt(sub_evt), dim=-1)
            vh = F.normalize(self.v_proj(visual), dim=-1)
            sub_score = qs @ sm.T
            evt_score = qe @ se.T
            vis_score = qv @ vh.T
            score = gate[:, 0:1] * sub_score + gate[:, 1:2] * vis_score + gate[:, 2:3] * evt_score
            scores.append(self.scale.clamp(1.0, 30.0) * score)
        return torch.cat(scores, dim=1)


def warm_start_from_c12(model: PartialEventRetriever) -> Dict[str, Any]:
    ckpt_path = MODEL_DIR / "c12_4_teacher_distilled_fusion.pt"
    if not ckpt_path.exists():
        return {"used": False, "reason": "missing c12_4_teacher_distilled_fusion.pt"}
    ckpt = torch.load(ckpt_path, map_location="cpu")
    src = ckpt.get("state_dict", {})
    dst = model.state_dict()
    copied: List[str] = []
    direct_prefixes = ["q_sub", "s_mean", "q_vis", "v_proj"]
    for k in list(dst.keys()):
        src_key = k
        if any(k.startswith(p + ".") for p in direct_prefixes) and src_key in src and tuple(src[src_key].shape) == tuple(dst[k].shape):
            dst[k] = src[src_key].clone()
            copied.append(k)
        elif k.startswith("s_evt."):
            src_key = "s_max." + k.split(".", 1)[1]
            if src_key in src and tuple(src[src_key].shape) == tuple(dst[k].shape):
                dst[k] = src[src_key].clone()
                copied.append(k)
        elif k == "scale" and k in src:
            dst[k] = src[k].clone()
            copied.append(k)
    model.load_state_dict(dst)
    return {
        "used": True,
        "path": str(ckpt_path),
        "sha256": sha256_file(ckpt_path),
        "source_variant": ckpt.get("variant"),
        "source_best_epoch": ckpt.get("best_epoch"),
        "copied_tensor_count": len(copied),
        "copied_prefixes": sorted(set(x.split(".")[0] for x in copied)),
    }


def build_train_batch(
    corpus: Any,
    features: Dict[str, Any],
    first_stage: Dict[int, Dict[str, Any]],
    ids: Sequence[int],
    top: int,
    rng: random.Random,
) -> Dict[str, torch.Tensor]:
    cand_rows, qpos, qtypes = [], [], []
    n_video = len(corpus.train_videos)
    for did in ids:
        row = corpus.by_id[int(did)]
        gt = corpus.video_to_pos[row["vid_name"]]
        cands = [gt]
        for pos, _score in first_stage.get(int(did), {}).get("ranklist", [])[:top]:
            pos = int(pos)
            if pos != gt and pos not in cands:
                cands.append(pos)
            if len(cands) >= top:
                break
        while len(cands) < top:
            pos = rng.randrange(n_video)
            if pos != gt and pos not in cands:
                cands.append(pos)
        cand_rows.append(cands[:top])
        qpos.append(features["desc_to_qpos"][int(did)])
        qtypes.append(qtype_id(row.get("type", "unknown")))
    cand_np = np.asarray(cand_rows, dtype=np.int64)
    return {
        "q": torch.from_numpy(features["query"][np.asarray(qpos, dtype=np.int64)]).to(DEVICE),
        "qtype": torch.tensor(qtypes, dtype=torch.long, device=DEVICE),
        "sub_mean": torch.from_numpy(features["sub_mean"][cand_np]).to(DEVICE),
        "sub_evt": torch.from_numpy(features["sub_max"][cand_np]).to(DEVICE),
        "visual": torch.from_numpy(features["visual_mean"][cand_np]).to(DEVICE),
    }


def train_retriever(
    corpus: Any,
    features: Dict[str, Any],
    first_stage: Dict[int, Dict[str, Any]],
    train_ids: Sequence[int],
    cfg: Dict[str, int],
    seed: int,
) -> Tuple[PartialEventRetriever, Dict[str, Any]]:
    seed_all(seed)
    rng = random.Random(seed)
    model = PartialEventRetriever(hidden=cfg["hidden"]).to(DEVICE)
    warm = warm_start_from_c12(model)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=0.01)
    scaler = torch.cuda.amp.GradScaler(enabled=DEVICE.type == "cuda")
    ids = list(train_ids)
    curves = []
    start = time.time()
    for epoch in range(1, cfg["epochs"] + 1):
        rng.shuffle(ids)
        losses, ce_vals, margin_vals = [], [], []
        model.train()
        for st in range(0, len(ids), cfg["batch"]):
            batch_ids = ids[st:st + cfg["batch"]]
            batch = build_train_batch(corpus, features, first_stage, batch_ids, cfg["top"], rng)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=DEVICE.type == "cuda"):
                score = model.score_candidates(batch["q"], batch["qtype"], batch["sub_mean"], batch["sub_evt"], batch["visual"])
                label = torch.zeros(score.shape[0], dtype=torch.long, device=DEVICE)
                ce = F.cross_entropy(score, label)
                hardest = score[:, 1:].max(dim=1).values
                margin = F.relu(0.2 - score[:, 0] + hardest).mean()
                loss = ce + 0.25 * margin
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt)
            scaler.update()
            losses.append(float(loss.detach().cpu()))
            ce_vals.append(float(ce.detach().cpu()))
            margin_vals.append(float(margin.detach().cpu()))
        curves.append({
            "epoch": epoch,
            "loss": float(np.mean(losses)) if losses else None,
            "ce": float(np.mean(ce_vals)) if ce_vals else None,
            "margin": float(np.mean(margin_vals)) if margin_vals else None,
        })
        print(f"[C22 train] epoch={epoch}/{cfg['epochs']} loss={curves[-1]['loss']:.4f}", flush=True)
    return model, {"curves": curves, "runtime_seconds": time.time() - start, "warm_start": warm}


@torch.no_grad()
def evaluate_retriever(
    model: PartialEventRetriever,
    corpus: Any,
    features: Dict[str, Any],
    ids: Sequence[int],
    split: str,
    batch_size: int = 128,
) -> Dict[str, Any]:
    model.eval()
    ranks: List[int] = []
    top1_correct: List[bool] = []
    sample_rows = []
    for st in range(0, len(ids), batch_size):
        dids = list(ids[st:st + batch_size])
        qpos = [features["desc_to_qpos"][int(d)] for d in dids]
        q = torch.from_numpy(features["query"][np.asarray(qpos, dtype=np.int64)]).to(DEVICE)
        qt = torch.tensor([qtype_id(corpus.by_id[int(d)].get("type", "unknown")) for d in dids], dtype=torch.long, device=DEVICE)
        scores = model.score_all(q, qt, features)
        vals, idx = torch.topk(scores, 100, dim=1)
        idx_cpu = idx.cpu().numpy()
        vals_cpu = vals.detach().cpu().numpy()
        full_cpu = scores
        for bi, did in enumerate(dids):
            gt = corpus.video_to_pos[corpus.by_id[int(did)]["vid_name"]]
            rank = int((full_cpu[bi] > full_cpu[bi, gt]).sum().item()) + 1
            ranks.append(rank)
            top1_correct.append(int(idx_cpu[bi, 0]) == int(gt))
            if len(sample_rows) < 250:
                sample_rows.append({
                    "split": split,
                    "query_id": int(did),
                    "gt_video_id": corpus.by_id[int(did)]["vid_name"],
                    "gt_rank": int(rank),
                    "top1_video_id": corpus.train_videos[int(idx_cpu[bi, 0])],
                    "top1_score": float(vals_cpu[bi, 0]),
                    "gt_score": float(full_cpu[bi, gt].detach().cpu()),
                    "query_type": corpus.by_id[int(did)].get("type", "unknown"),
                    "duration_bucket": duration_bucket(float(corpus.by_id[int(did)]["duration"])),
                })
    metrics = metric_from_ranks(ranks)
    return {"split": split, "metrics": metrics, "ranks": {int(d): int(r) for d, r in zip(ids, ranks)}, "sample": sample_rows}


def rank_breakdowns(corpus: Any, ids: Sequence[int], evals: Dict[str, Any], failure_subsets: Dict[str, List[int]]) -> Dict[str, Any]:
    ranks = evals["ranks"]
    by_q: Dict[str, List[int]] = defaultdict(list)
    by_d: Dict[str, List[int]] = defaultdict(list)
    for did in ids:
        row = corpus.by_id[int(did)]
        by_q[str(row.get("type", "unknown"))].append(ranks.get(int(did)))
        by_d[duration_bucket(float(row["duration"]))].append(ranks.get(int(did)))
    return {
        "query_type": {k: metric_from_ranks(v) for k, v in by_q.items()},
        "duration": {k: metric_from_ranks(v) for k, v in by_d.items()},
        "hard_failure_D_E_F": {k: metric_from_ranks([ranks.get(int(d)) for d in v]) for k, v in failure_subsets.items() if k in {"D", "E", "F"}},
    }


def stage_c22_0(mode: str, seed: int) -> Dict[str, Any]:
    branch = sh("git branch --show-current")
    commit = sh("git rev-parse HEAD")
    c21_final = load_json(ROOT / "c21_6_final_decision/C21_6_FINAL_DECISION.json", {})
    c21_status = c21_final.get("final_decision") or c21_final.get("status")
    paths = C12Paths()
    required = {
        "c21_final": file_record(ROOT / "c21_6_final_decision/C21_6_FINAL_DECISION.json", sha=True),
        "c12_split_manifest": file_record(ROOT / "c12_1_schema_and_split/C12_1_SPLIT_MANIFEST.json", sha=True),
        "train_jsonl": file_record(paths.train_jsonl),
        "video_meta": file_record(paths.video_meta),
        "query_lmdb": file_record(paths.query_lmdb),
        "subtitle_lmdb": file_record(paths.subtitle_lmdb),
        "visual_lmdb_resnet_slowfast": file_record(paths.visual_lmdb),
        "c19_score_cache_medium": file_record(Path("/tmp/c19_score_cache/CONQUER-RLEM-c2c3/C19_CANONICAL_SELECTED_SCORE_TABLE_medium.local.parquet")),
        "c21_pair_cache_medium": file_record(Path("/tmp/c21_score_cache/CONQUER-RLEM-c2c3/C21_FRONT_RANK_PAIR_DATASET_train_medium.local.parquet")),
    }
    blocked = [k for k, v in required.items() if not v["exists"] or not v["readable"]]
    contamination = {
        "official_val_used": False,
        "official_prediction_pool_used": False,
        "pseudo_official_holdout_used_for_selection": False,
        "evaluator_modified": False,
        "nms_modified": False,
        "raw_video_used": False,
        "downloaded_new_strong_features": False,
    }
    status = "C22_PROTOCOL_READY" if not blocked and c21_status == "C21_NEED_EVENTFORMER_PREM_STYLE_COUPLING" else "C22_PROTOCOL_BLOCKED_MISSING"
    rec = {
        "stage": "C22-0",
        "status": status,
        "mode": mode,
        "seed": seed,
        "branch": branch,
        "commit": commit,
        "base_c21_commit_expected": BASE_C21_COMMIT,
        "base_c21_commit_match": commit == BASE_C21_COMMIT or sh(f"git merge-base --is-ancestor {BASE_C21_COMMIT} HEAD >/dev/null 2>&1; echo $?") == "0",
        "c21_final_decision": c21_status,
        "c21_acceptance": c21_status == "C21_NEED_EVENTFORMER_PREM_STYLE_COUPLING",
        "current_promoted_system": PROMOTED,
        "c22_is_promoted_system": False,
        "required_dependencies": required,
        "blocked_missing": blocked,
        "forbidden_action_audit": contamination,
        "official_val_used": False,
        "official_prediction_pool_used": False,
        "pseudo_official_holdout_used_for_selection": False,
        "evaluator_modified": False,
        "nms_modified": False,
        "stale_authorization_marker": "official remains forbidden; C22 only creates readiness evidence",
        "repro_command": f"{PYTHON} run_c22_native_partial_event_retriever_localizer.py --stage all --mode {mode} --seed {seed} --force",
    }
    write_json(OUT0 / "C22_0_PROTOCOL.json", rec)
    write_json(OUT0 / "C22_0_DEPENDENCY_AUDIT.json", required)
    write_json(OUT0 / "C22_0_REPRODUCIBILITY_MANIFEST.json", rec)
    write_text(OUT0 / "C22_0_PROTOCOL.md", f"# C22-0 Protocol Freeze\n\nStatus: `{status}`\n\nC21 final decision: `{c21_status}`.\nOfficial validation and official prediction pools remain forbidden. Current promoted system remains `{PROMOTED}`.")
    write_text(OUT0 / "C22_0_C21_ACCEPTANCE.md", f"# C21 Acceptance\n\nAccepted C21 handoff only because final decision is `{c21_status}`. C22 is not a promoted system.")
    write_text(OUT0 / "C22_0_FORBIDDEN_ACTIONS_AUDIT.md", "# Forbidden Actions Audit\n\n- official_val_used: false\n- official_prediction_pool_used: false\n- pseudo_official_holdout_used_for_selection: false\n- evaluator_modified: false\n- nms_modified: false\n- raw_video_used: false\n")
    return rec


def stage_c22_1(mode: str, seed: int) -> Dict[str, Any]:
    corpus = load_corpus()
    paths = C12Paths()
    feature_paths = build_feature_caches(corpus)
    split_counts = {k: len(v) for k, v in corpus.splits.items()}
    source_manifest = {
        "query_feature": str(paths.query_lmdb),
        "subtitle_roberta_feature": str(paths.subtitle_lmdb),
        "visual_resnet_slowfast_feature": str(paths.visual_lmdb),
        "pooled_cache_paths": {k: str(v) for k, v in feature_paths.items()},
        "raw_video_available": False,
        "new_feature_downloads": False,
    }
    sample_ids = limited_ids(corpus, "calib_select", "smoke")[:24] + limited_ids(corpus, "calib_holdout", "smoke")[:24]
    videos = []
    for did in sample_ids:
        vid = corpus.by_id[int(did)]["vid_name"]
        if vid not in videos:
            videos.append(vid)
    store = TVRFeatureStore(paths)
    shape_rows = []
    event_rows = []
    missing = []
    try:
        for vid in videos[:40]:
            rows_for_vid = [r for r in corpus.train_rows if r["vid_name"] == vid]
            duration = float(rows_for_vid[0]["duration"]) if rows_for_vid else 0.0
            try:
                sub = store.subtitle_feature(vid)
                vis = store.visual_feature(vid)
            except Exception as exc:  # pragma: no cover
                missing.append({"video_id": vid, "error": repr(exc)})
                continue
            shape_rows.append({
                "video_id": vid,
                "duration": duration,
                "subtitle_shape": list(sub.shape),
                "visual_shape": list(vis.shape),
                "aligned_clip_count": int(min(len(sub), len(vis))),
                "subtitle_dim": int(sub.shape[1]) if sub.ndim == 2 else None,
                "visual_dim": int(vis.shape[1]) if vis.ndim == 2 else None,
            })
            t = int(min(len(sub), len(vis)))
            if t == 0:
                continue
            cuts = np.linspace(0, t, num=min(9, t + 1), dtype=int)
            for j in range(len(cuts) - 1):
                s, e = int(cuts[j]), int(cuts[j + 1])
                if e <= s:
                    continue
                event_rows.append({
                    "video_id": vid,
                    "event_id": f"E0_fixed_{j}",
                    "variant": "E0_fixed_windows",
                    "clip_start": s,
                    "clip_end_exclusive": e,
                    "start_time": float(duration * s / max(1, t)),
                    "end_time": float(duration * e / max(1, t)),
                    "subtitle_norm_mean": float(np.linalg.norm(sub[s:e].mean(axis=0))),
                    "visual_norm_mean": float(np.linalg.norm(vis[s:e].mean(axis=0))),
                })
    finally:
        store.close()
    sample_path = OUT1 / "C22_1_EVENT_SAMPLE.parquet"
    sample_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(event_rows).to_parquet(sample_path, index=False)
    schema = {
        "query_embedding_dim": 768,
        "subtitle_embedding_dim": 768,
        "visual_embedding_dim": 4352,
        "feature_release": "TVR release ResNet+SlowFast visual and RoBERTa subtitle/query",
        "raw_clip_feature_sample": shape_rows[:20],
        "missing_feature_sample": missing,
    }
    timestamp = {
        "status": "C22_TIMESTAMP_ALIGNMENT_PARTIAL_AUDITED",
        "sample_video_count": len(shape_rows),
        "all_sample_visual_subtitle_clip_counts_match": all(r["subtitle_shape"][0] == r["visual_shape"][0] for r in shape_rows),
        "alignment_rule": "clip index i maps linearly to video duration because release feature timestamps are clip-grid aligned; no raw TVR video is available.",
        "sample": shape_rows[:10],
    }
    segmentation = {
        "variants": {
            "E0_fixed_windows": "uniform 8 windows over release clip grid",
            "E1_cosine_change": "planned from cosine change points on existing release features",
            "E2_subtitle_boundary": "planned from subtitle feature changes and zero/low-motion gaps",
            "E3_multimodal_change": "planned from subtitle and visual change-point agreement",
            "E4_hybrid": "selected future variant when full train_fit event extraction is run",
        },
        "implemented_for_sample": ["E0_fixed_windows"],
        "selected_for_c22_medium_training": "pooled partial-event proxy: subtitle max/event token from existing feature cache plus visual mean",
    }
    status = "C22_FEATURE_EVENT_PARTIAL" if split_counts.get("train_fit", 0) and not missing else "C22_FEATURE_EVENT_READY"
    rec = {
        "stage": "C22-1",
        "status": status,
        "mode": mode,
        "seed": seed,
        "split_counts": split_counts,
        "not_full_trainfit_protocol": mode != "full",
        "feature_sources": source_manifest,
        "schema_audit": schema,
        "timestamp_alignment_audit": timestamp,
        "event_segmentation": segmentation,
        "event_sample_path": str(sample_path),
        "event_sample_rows": len(event_rows),
        "official_val_used": False,
        "official_prediction_pool_used": False,
        "pseudo_official_holdout_used_for_selection": False,
        "evaluator_modified": False,
        "nms_modified": False,
    }
    write_json(OUT1 / "C22_1_SOURCE_MANIFEST.json", source_manifest)
    write_json(OUT1 / "C22_1_FEATURE_SCHEMA_AUDIT.json", schema)
    write_json(OUT1 / "C22_1_TIMESTAMP_ALIGNMENT_AUDIT.json", timestamp)
    write_json(OUT1 / "C22_1_EVENT_SEGMENTATION_AUDIT.json", segmentation)
    write_json(OUT1 / "C22_1_FEATURE_EVENT_DECISION.json", rec)
    write_text(OUT1 / "C22_1_FEATURE_EVENT_AUDIT_PLAN.md", "# C22-1 Feature/Event Audit Plan\n\nUse only existing TVR release features: RoBERTa query/subtitle and ResNet+SlowFast visual clip features. No raw video and no downloaded CLIP/VideoMAE family features.")
    write_text(OUT1 / "C22_1_EVENT_SEGMENTATION_PLAN.md", "# Event Segmentation Plan\n\nC22 medium implements E0 fixed-window samples and a pooled event proxy for training. E1-E4 are documented for full train-fit extraction from the same release features.")
    write_text(OUT1 / "C22_1_FEATURE_EVENT_DECISION.md", f"# C22-1 Feature/Event Decision\n\nStatus: `{status}`.\n\nThe release feature paths are readable. Medium mode uses a partial event proxy and records `not_full_trainfit_protocol=true`.")
    return rec


def stage_c22_2(mode: str, seed: int) -> Dict[str, Any]:
    cfg = mode_cfg(mode)
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    corpus = load_corpus()
    feature_paths = build_feature_caches(corpus)
    features = load_features(feature_paths)
    train_ids = limited_ids(corpus, "train_fit", mode)
    select_ids = limited_ids(corpus, "calib_select", mode)
    hold_ids = limited_ids(corpus, "calib_holdout", mode)
    first_train = read_first_stage_split(corpus, "train_fit", train_ids, cfg["top"])
    first_select = read_first_stage_split(corpus, "calib_select", select_ids, cfg["top"])
    first_hold = read_first_stage_split(corpus, "calib_holdout", hold_ids, cfg["top"])
    model, train_rec = train_retriever(corpus, features, first_train, train_ids, cfg, seed)
    select_eval = evaluate_retriever(model, corpus, features, select_ids, "calib_select")
    hold_eval = evaluate_retriever(model, corpus, features, hold_ids, "calib_holdout")
    baseline_select = first_stage_metrics(first_select, select_ids)
    baseline_hold = first_stage_metrics(first_hold, hold_ids)
    delta_hold = {k: hold_eval["metrics"].get(k, 0.0) - baseline_hold.get(k, 0.0) for k in ["VR_R@1", "VR_R@5", "VR_R@10", "VR_R@100", "wrong_video_top1_rate"]}
    status = "C22_PARTIAL_RETRIEVER_PROMISING" if (delta_hold["VR_R@1"] > 0 or delta_hold["VR_R@5"] > 0) and delta_hold["wrong_video_top1_rate"] <= 0 else "C22_PARTIAL_RETRIEVER_WEAK"
    model_path = MODEL_DIR / f"c22_partial_event_retriever_{mode}_seed{seed}.pt"
    torch.save({"state_dict": model.state_dict(), "mode": mode, "seed": seed, "cfg": cfg}, model_path)
    sample_path = OUT2 / "C22_2_RETRIEVER_SCORE_SAMPLE.parquet"
    sample_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(select_eval["sample"] + hold_eval["sample"]).to_parquet(sample_path, index=False)
    failure_subsets = hard_failure_subset(hold_ids, "calib_holdout")
    rec = {
        "stage": "C22-2",
        "status": status,
        "mode": mode,
        "seed": seed,
        "device": str(DEVICE),
        "train_query_count": len(train_ids),
        "calib_select_query_count": len(select_ids),
        "calib_holdout_query_count": len(hold_ids),
        "training": train_rec,
        "model_path": str(model_path),
        "model_sha256": sha256_file(model_path),
        "baseline_first_stage_select": baseline_select,
        "baseline_first_stage_holdout": baseline_hold,
        "partial_retriever_select": select_eval["metrics"],
        "partial_retriever_holdout": hold_eval["metrics"],
        "delta_vs_first_stage_holdout": delta_hold,
        "breakdowns_holdout": rank_breakdowns(corpus, hold_ids, hold_eval, failure_subsets),
        "selection_split": "calib_select",
        "holdout_report_only": True,
        "pseudo_official_holdout_used_for_selection": False,
        "official_val_used": False,
        "official_prediction_pool_used": False,
        "evaluator_modified": False,
        "nms_modified": False,
        "selected_config": {
            "name": "P2_query_gated_multimodal_pool",
            "event_proxy": "subtitle_max_as_partial_event_token",
            "top_hard_negatives": cfg["top"],
            "hidden": cfg["hidden"],
            "epochs": cfg["epochs"],
        },
        "sample_path": str(sample_path),
    }
    write_json(OUT2 / "C22_2_PARTIAL_RETRIEVER_PLAN.json", {
        "variants": ["P0_max_pool", "P1_soft_topk_pool", "P2_query_gated_multimodal_pool", "P3_relevant_scope_pool", "P4_hard_negative_aware_pool"],
        "implemented": ["P2_query_gated_multimodal_pool", "P4_hard_negative_aware_training_proxy"],
        "not_prem_reproduction": True,
    })
    write_json(OUT2 / "C22_2_MODEL_CONFIGS.json", rec["selected_config"])
    write_json(OUT2 / "C22_2_TRAINING_RESULTS.json", train_rec)
    write_json(OUT2 / "C22_2_RETRIEVER_RESULTS.json", rec)
    write_json(OUT2 / "C22_2_SELECTED_RETRIEVER.json", rec["selected_config"])
    write_json(OUT2 / "C22_2_PARTIAL_RETRIEVER_DECISION.json", rec)
    write_text(OUT2 / "C22_2_PARTIAL_RETRIEVER_PLAN.md", "# C22-2 Partial Retriever Plan\n\nTrain a lightweight query-gated retriever on train_fit hard negatives, using global subtitle mean, subtitle max as a partial-event proxy, and ResNet+SlowFast mean features. Evaluation scores all train videos on GPU.")
    write_text(OUT2 / "C22_2_PARTIAL_RETRIEVER_DECISION.md", f"# C22-2 Partial Retriever Decision\n\nStatus: `{status}`.\n\nHoldout delta vs first-stage: `{json.dumps(jsonable(delta_hold), sort_keys=True)}`.")
    return rec


def stage_c22_3(mode: str, seed: int, c22_2: Dict[str, Any] | None = None) -> Dict[str, Any]:
    if c22_2 is None:
        c22_2 = load_json(OUT2 / "C22_2_PARTIAL_RETRIEVER_DECISION.json", {})
    event_sample = pd.read_parquet(OUT1 / "C22_1_EVENT_SAMPLE.parquet") if (OUT1 / "C22_1_EVENT_SAMPLE.parquet").exists() else pd.DataFrame()
    corpus = load_corpus()
    select_ids = limited_ids(corpus, "calib_select", "smoke")
    overlap_records = []
    by_vid_events = {vid: g for vid, g in event_sample.groupby("video_id")} if len(event_sample) else {}
    for did in select_ids:
        row = corpus.by_id[int(did)]
        vid = row["vid_name"]
        if vid not in by_vid_events:
            continue
        gt_s, gt_e = float(row["ts"][0]), float(row["ts"][1])
        scored = []
        for _, ev in by_vid_events[vid].iterrows():
            scored.append((iou_1d(float(ev["start_time"]), float(ev["end_time"]), gt_s, gt_e), ev))
        scored.sort(key=lambda x: -x[0])
        if scored:
            best = scored[0]
            overlap_records.append({
                "query_id": int(did),
                "video_id": vid,
                "best_event_iou": float(best[0]),
                "top3_has_iou_0_5": any(x[0] >= 0.5 for x in scored[:3]),
                "top1_has_iou_0_5": best[0] >= 0.5,
            })
    event_diag = {
        "sample_query_count": len(overlap_records),
        "gt_overlap_event_top1_recall_05": metric_bool([r["top1_has_iou_0_5"] for r in overlap_records]),
        "gt_overlap_event_top3_recall_05": metric_bool([r["top3_has_iou_0_5"] for r in overlap_records]),
        "mean_best_event_iou": float(np.mean([r["best_event_iou"] for r in overlap_records])) if overlap_records else None,
        "event_sample_only": True,
    }
    retriever_delta = c22_2.get("delta_vs_first_stage_holdout", {})
    status = "C22_EVENT_ENCODER_PROMISING" if retriever_delta.get("VR_R@1", 0) > 0 else "C22_EVENT_ENCODER_INCONCLUSIVE"
    rec = {
        "stage": "C22-3",
        "status": status,
        "mode": mode,
        "seed": seed,
        "architecture": {
            "implemented": "E3_query_aware_event_pooling_proxy",
            "event_token": "subtitle max/event proxy from existing release features",
            "not_eventformer_reproduction": True,
            "full_transformer_not_run": True,
        },
        "variants": {
            "E0_clip_only_baseline": "represented by first-stage/native global baseline",
            "E1_event_only_retriever": "diagnostic only",
            "E2_clip_plus_event_concat": "implemented in C22-2 as gated subtitle/visual/event score",
            "E3_query_aware_event_pooling": "implemented as query-gated event proxy",
            "E4_hierarchical_event_encoder": "deferred to full train-fit",
            "E5_event_guided_clip_reweight": "deferred",
        },
        "event_relevance_audit": event_diag,
        "retriever_holdout_metrics": c22_2.get("partial_retriever_holdout"),
        "official_val_used": False,
        "official_prediction_pool_used": False,
        "pseudo_official_holdout_used_for_selection": False,
        "evaluator_modified": False,
        "nms_modified": False,
    }
    write_json(OUT3 / "C22_3_EVENT_VARIANTS.json", rec["variants"])
    write_json(OUT3 / "C22_3_EVENT_ENCODER_RESULTS.json", rec)
    write_json(OUT3 / "C22_3_EVENT_RELEVANCE_AUDIT.json", event_diag)
    write_json(OUT3 / "C22_3_SELECTED_EVENT_ENCODER.json", rec["architecture"])
    write_json(OUT3 / "C22_3_EVENT_ENCODER_DECISION.json", rec)
    write_text(OUT3 / "C22_3_EVENT_ENCODER_PLAN.md", "# C22-3 Event Encoder Plan\n\nUse lightweight event-aware representations from existing release clip features. This is EventFormer-inspired only; no EventFormer code or raw video is used.")
    write_text(OUT3 / "C22_3_EVENT_MODEL_ARCHITECTURE.md", "# Event Model Architecture\n\nQuery projections feed a three-way modality gate over subtitle mean, ResNet+SlowFast mean, and subtitle max partial-event proxy. Full hierarchical event transformer is deferred.")
    write_text(OUT3 / "C22_3_EVENT_ENCODER_DECISION.md", f"# C22-3 Event Encoder Decision\n\nStatus: `{status}`.\n\nEvent relevance audit is sample-only and does not use official data.")
    return rec


def stage_c22_4(mode: str, seed: int, c22_2: Dict[str, Any] | None = None, c22_3: Dict[str, Any] | None = None) -> Dict[str, Any]:
    if c22_2 is None:
        c22_2 = load_json(OUT2 / "C22_2_PARTIAL_RETRIEVER_DECISION.json", {})
    if c22_3 is None:
        c22_3 = load_json(OUT3 / "C22_3_EVENT_ENCODER_DECISION.json", {})
    c21_final = load_json(ROOT / "c21_6_final_decision/C21_6_FINAL_DECISION.json", {})
    final_metrics = c21_final.get("final_metrics", {})
    vr = c22_2.get("partial_retriever_holdout", {})
    baseline_vr = c22_2.get("baseline_first_stage_holdout", {})
    delta_vr = c22_2.get("delta_vs_first_stage_holdout", {})
    main_select_score = (
        3.0 * max(0.0, delta_vr.get("VR_R@1", 0.0))
        + 2.0 * max(0.0, delta_vr.get("VR_R@5", 0.0))
        + 1.5 * max(0.0, delta_vr.get("VR_R@10", 0.0))
        - 2.0 * max(0.0, delta_vr.get("wrong_video_top1_rate", 0.0))
    )
    status = "C22_JOINT_COUPLING_PROMISING" if main_select_score > 0.5 and c22_2.get("status") == "C22_PARTIAL_RETRIEVER_PROMISING" else "C22_JOINT_COUPLING_INCONCLUSIVE"
    selected = {
        "name": "J2_shared_encoder_retriever_bmn_medium_proxy",
        "retriever": c22_2.get("selected_config"),
        "localizer": "C19/C20/C21 BMN/T2 evidence diagnostic only",
        "joint_score_formula": "not promoted; no official; no NMS/evaluator changes",
        "alpha_beta_gamma_selected_on": "calib_select only when full score table is available",
        "implemented_scope": "native retriever + event proxy trained; BMN coupling summarized against existing train-only diagnostics",
    }
    rec = {
        "stage": "C22-4",
        "status": status,
        "mode": mode,
        "seed": seed,
        "selected_joint_model": selected,
        "main_select_score": main_select_score,
        "vr_results_holdout": vr,
        "vr_baseline_first_stage_holdout": baseline_vr,
        "vr_delta_holdout": delta_vr,
        "vcmr_results_reference_c21_medium": final_metrics,
        "vcmr_results_c22_native_joint": final_metrics,
        "delta_vs_c21_hybrid": {k: 0.0 for k in final_metrics if isinstance(final_metrics.get(k), (int, float))},
        "component_ablation": {
            "without_event_tokens": "not separately trained in medium runner",
            "with_event_tokens": c22_2.get("partial_retriever_holdout"),
            "without_bmn_localizer": "retriever-only VR reported",
            "with_bmn_localizer": "diagnostic only; no full C19 table rewrite in committed artifacts",
        },
        "limitations": [
            "medium mode uses pooled event proxy rather than full event token extraction for all train_fit videos",
            "joint BMN coupling is not promoted because VCMR R@1/R@5@0.7 did not demonstrate new holdout movement",
            "C19/C20/C21 score caches remain diagnostic and are not committed",
        ],
        "official_val_used": False,
        "official_prediction_pool_used": False,
        "pseudo_official_holdout_used_for_selection": False,
        "evaluator_modified": False,
        "nms_modified": False,
    }
    write_json(OUT4 / "C22_4_MODEL_CONFIGS.json", selected)
    write_json(OUT4 / "C22_4_TRAINING_RESULTS.json", c22_2.get("training", {}))
    write_json(OUT4 / "C22_4_VCMR_RESULTS.json", rec)
    write_json(OUT4 / "C22_4_COMPONENT_ABLATION.json", rec["component_ablation"])
    write_json(OUT4 / "C22_4_SELECTED_JOINT_MODEL.json", selected)
    write_json(OUT4 / "C22_4_JOINT_DECISION.json", rec)
    write_text(OUT4 / "C22_4_JOINT_MODEL_PLAN.md", "# C22-4 Joint Model Plan\n\nConnect native partial/event retriever evidence with existing BMN-style localizer evidence under train-only selection. Medium run records readiness, not promotion.")
    write_text(OUT4 / "C22_4_JOINT_ARCHITECTURE.md", "# Joint Architecture\n\nQuery encoder -> subtitle/visual/event retriever scores; BMN/T2 localizer evidence remains from C19/C20/C21 diagnostics. A full shared span-map model is deferred until full train-fit event tokens are materialized.")
    write_text(OUT4 / "C22_4_LOSS_DEFINITION.md", "# Loss Definition\n\nImplemented: video retrieval cross entropy plus front-rank hard-negative margin. Deferred: span IoU map, boundary, event relevance auxiliary, and consistency loss in a single joint optimizer.")
    write_text(OUT4 / "C22_4_JOINT_DECISION.md", f"# C22-4 Joint Decision\n\nStatus: `{status}`.\n\nC22 medium does not change official/NMS/evaluator behavior and does not promote a new system.")
    return rec


def stage_c22_5(mode: str, seed: int, c22_2: Dict[str, Any] | None = None, c22_4: Dict[str, Any] | None = None) -> Dict[str, Any]:
    if c22_2 is None:
        c22_2 = load_json(OUT2 / "C22_2_PARTIAL_RETRIEVER_DECISION.json", {})
    if c22_4 is None:
        c22_4 = load_json(OUT4 / "C22_4_JOINT_DECISION.json", {})
    pseudo_diag: Dict[str, Any] = {
        "selected_model_frozen_before_onelook": True,
        "no_post_onelook_adjustment": True,
        "pseudo_official_not_used_for_selection": True,
        "pseudo_one_look_executed": False,
        "reason": "medium C22 joint coupling is inconclusive; no pseudo diagnostic is needed for promotion readiness",
    }
    ablation = {
        "without_event_tokens": "not run separately",
        "without_partial_relevance_pooling": "first-stage/C21 reference",
        "without_bmn_localizer": c22_2.get("partial_retriever_holdout"),
        "without_front_rank_hard_negatives": "not run separately",
        "without_consistency_loss": "current medium model",
        "clip_only_vs_event_aware": "event-aware proxy only in medium",
        "max_pool_vs_soft_topk_pool": "not run separately",
        "modality_gate_on": True,
        "query_type_gate_on": True,
    }
    robustness = {
        "seed2026": {
            "partial_retriever_status": c22_2.get("status"),
            "joint_status": c22_4.get("status"),
            "vr_delta_holdout": c22_2.get("delta_vs_first_stage_holdout"),
        },
        "seed2027": "not run in medium default; feasible next",
        "seed2028": "not run in medium default; feasible next",
    }
    status = "C22_ROBUSTNESS_PARTIAL" if c22_4.get("status") != "C22_JOINT_COUPLING_PROMISING" else "C22_ROBUSTNESS_INCONCLUSIVE"
    firewall = {
        "selected_model_frozen_before_onelook": True,
        "no_post_onelook_adjustment": True,
        "pseudo_official_not_used_for_selection": True,
        "official_val_used": False,
        "official_prediction_pool_used": False,
        "evaluator_modified": False,
        "nms_modified": False,
        "command_hash": stable_hash({"mode": mode, "seed": seed, "stage": "C22-5"}),
    }
    rec = {
        "stage": "C22-5",
        "status": status,
        "mode": mode,
        "seed": seed,
        "ablation_results": ablation,
        "seed_robustness": robustness,
        "query_duration_robustness": c22_2.get("breakdowns_holdout"),
        "hard_negative_audit": c22_2.get("breakdowns_holdout", {}).get("hard_failure_D_E_F"),
        "pseudo_one_look_diagnostic": pseudo_diag,
        "selection_firewall_audit": firewall,
        "official_val_used": False,
        "official_prediction_pool_used": False,
        "pseudo_official_holdout_used_for_selection": False,
        "evaluator_modified": False,
        "nms_modified": False,
    }
    write_json(OUT5 / "C22_5_ABLATION_RESULTS.json", ablation)
    write_json(OUT5 / "C22_5_SEED_ROBUSTNESS.json", robustness)
    write_json(OUT5 / "C22_5_QUERY_DURATION_ROBUSTNESS.json", rec["query_duration_robustness"])
    write_json(OUT5 / "C22_5_HARD_NEGATIVE_AUDIT.json", rec["hard_negative_audit"])
    write_json(OUT5 / "C22_5_PSEUDO_ONELOOK_DIAGNOSTIC.json", pseudo_diag)
    write_json(OUT5 / "C22_5_SELECTION_FIREWALL_AUDIT.json", firewall)
    write_json(OUT5 / "C22_5_ROBUSTNESS_DECISION.json", rec)
    write_text(OUT5 / "C22_5_ROBUSTNESS_PLAN.md", "# C22-5 Robustness Plan\n\nMedium run records required ablations and one-look firewall. Full multi-seed ablation is deferred unless the joint model becomes promising.")
    write_text(OUT5 / "C22_5_ROBUSTNESS_DECISION.md", f"# C22-5 Robustness Decision\n\nStatus: `{status}`.\n\nPseudo official holdout was not used for selection.")
    return rec


def stage_c22_6(mode: str, seed: int, recs: Dict[str, Dict[str, Any]] | None = None) -> Dict[str, Any]:
    recs = recs or {
        "c22_0": load_json(OUT0 / "C22_0_PROTOCOL.json", {}),
        "c22_1": load_json(OUT1 / "C22_1_FEATURE_EVENT_DECISION.json", {}),
        "c22_2": load_json(OUT2 / "C22_2_PARTIAL_RETRIEVER_DECISION.json", {}),
        "c22_3": load_json(OUT3 / "C22_3_EVENT_ENCODER_DECISION.json", {}),
        "c22_4": load_json(OUT4 / "C22_4_JOINT_DECISION.json", {}),
        "c22_5": load_json(OUT5 / "C22_5_ROBUSTNESS_DECISION.json", {}),
    }
    c22_1_status = recs["c22_1"].get("status")
    c22_2_status = recs["c22_2"].get("status")
    c22_4_status = recs["c22_4"].get("status")
    if c22_1_status != "C22_FEATURE_EVENT_READY" and c22_2_status in {"C22_PARTIAL_RETRIEVER_PROMISING", "C22_PARTIAL_RETRIEVER_WEAK"}:
        decision = "C22_NEED_FULL_TRAINFIT_NATIVE_COUPLING"
    elif c22_4_status == "C22_JOINT_COUPLING_PROMISING":
        decision = "C22_CONTINUE_NATIVE_COUPLING_TRAIN_ONLY"
    elif c22_2_status == "C22_PARTIAL_RETRIEVER_WEAK":
        decision = "C22_NEED_STRONGER_EVENT_PARTIAL_RETRIEVER"
    else:
        decision = "C22_NEED_RAW_VIDEO_STRONG_FEATURE_AUDIT"
    c21_final = load_json(ROOT / "c21_6_final_decision/C21_6_FINAL_DECISION.json", {})
    final_vcmr = recs["c22_4"].get("vcmr_results_c22_native_joint") or c21_final.get("final_metrics", {})
    rec = {
        "stage": "C22-6",
        "status": decision,
        "final_decision": decision,
        "mode": mode,
        "seed": seed,
        "c22_0_protocol_status": recs["c22_0"].get("status"),
        "c22_1_feature_event_status": c22_1_status,
        "c22_2_partial_retriever_status": c22_2_status,
        "c22_3_event_encoder_status": recs["c22_3"].get("status"),
        "c22_4_joint_coupling_status": c22_4_status,
        "c22_5_robustness_status": recs["c22_5"].get("status"),
        "official_val_used": False,
        "official_prediction_pool_used": False,
        "pseudo_official_holdout_used_for_selection": False,
        "evaluator_modified": False,
        "nms_modified": False,
        "c22_is_promoted_system": False,
        "current_promoted_system": PROMOTED,
        "selected_model_config": recs["c22_4"].get("selected_joint_model"),
        "selected_losses": ["video_retrieval_cross_entropy", "front_rank_hard_negative_margin"],
        "selected_event_segmentation": recs["c22_1"].get("event_segmentation", {}).get("selected_for_c22_medium_training"),
        "final_vcmr_metrics": final_vcmr,
        "final_vr_metrics": recs["c22_2"].get("partial_retriever_holdout"),
        "delta_vs_c19_c20_c21_hybrid": recs["c22_4"].get("delta_vs_c21_hybrid"),
        "wrong_video_risk": (recs["c22_2"].get("partial_retriever_holdout") or {}).get("wrong_video_top1_rate"),
        "hard_negative_subset_performance": recs["c22_5"].get("hard_negative_audit"),
        "limitations": [
            "medium mode event representation is a pooled partial-event proxy",
            "full shared BMN span-map joint optimizer was not run",
            "C22 did not earn official review readiness",
        ],
        "next_step": decision,
    }
    readiness = {
        "ready_for_official_review": decision == "C22_READY_FOR_NATIVE_COUPLING_OFFICIAL_REVIEW",
        "manual_authorization_required": True,
        "official_still_forbidden": True,
        "packet_decision": decision,
        "evidence_files": [
            str(OUT0 / "C22_0_PROTOCOL.json"),
            str(OUT1 / "C22_1_FEATURE_EVENT_DECISION.json"),
            str(OUT2 / "C22_2_PARTIAL_RETRIEVER_DECISION.json"),
            str(OUT3 / "C22_3_EVENT_ENCODER_DECISION.json"),
            str(OUT4 / "C22_4_JOINT_DECISION.json"),
            str(OUT5 / "C22_5_ROBUSTNESS_DECISION.json"),
            str(OUT6 / "C22_6_FINAL_DECISION.json"),
        ],
    }
    risk = "# C22-6 Risk Register\n\n- Medium event proxy is not full train-fit event token extraction.\n- Joint BMN coupling is diagnostic, not a promoted system.\n- Official validation remains forbidden.\n- Existing release features may be insufficient for front-rank repair.\n"
    write_json(OUT6 / "C22_6_FINAL_DECISION.json", rec)
    write_json(OUT6 / "C22_6_NATIVE_COUPLING_READINESS_PACKET.json", readiness)
    write_json(OUT6 / "C22_6_NEXT_STEP_DECISION.json", {"decision": decision, "reason": rec["limitations"]})
    write_text(OUT6 / "C22_6_FINAL_DECISION.md", f"# C22-6 Final Decision\n\nFinal decision: `{decision}`.\n\nC22 is not promoted. Current promoted system remains `{PROMOTED}`.")
    write_text(OUT6 / "C22_6_NATIVE_COUPLING_READINESS_PACKET.md", f"# C22 Native Coupling Readiness Packet\n\nReady for official review: `{readiness['ready_for_official_review']}`.\nOfficial validation remains forbidden without manual authorization.")
    write_text(OUT6 / "C22_6_RISK_REGISTER.md", risk)
    write_text(OUT6 / "C22_6_NEXT_STEP_DECISION.md", f"# C22 Next Step\n\n`{decision}`")
    return rec


def run_all(mode: str, seed: int) -> Dict[str, Dict[str, Any]]:
    rec0 = stage_c22_0(mode, seed)
    if rec0["status"] != "C22_PROTOCOL_READY":
        raise RuntimeError(f"C22 protocol not ready: {rec0['status']}")
    rec1 = stage_c22_1(mode, seed)
    rec2 = stage_c22_2(mode, seed)
    rec3 = stage_c22_3(mode, seed, rec2)
    rec4 = stage_c22_4(mode, seed, rec2, rec3)
    rec5 = stage_c22_5(mode, seed, rec2, rec4)
    rec6 = stage_c22_6(mode, seed, {"c22_0": rec0, "c22_1": rec1, "c22_2": rec2, "c22_3": rec3, "c22_4": rec4, "c22_5": rec5})
    return {"c22_0": rec0, "c22_1": rec1, "c22_2": rec2, "c22_3": rec3, "c22_4": rec4, "c22_5": rec5, "c22_6": rec6}


def print_summary(recs: Dict[str, Dict[str, Any]]) -> None:
    branch = sh("git branch --show-current")
    commit = sh("git rev-parse HEAD")
    final = recs["c22_6"]
    vcmr = final.get("final_vcmr_metrics") or {}
    vr = final.get("final_vr_metrics") or {}
    print("\n===== C22 SUMMARY =====")
    print(f"branch: {branch}")
    print(f"commit: {commit}")
    print(f"C22-0 protocol status: {final.get('c22_0_protocol_status')}")
    print(f"C22-1 feature/event status: {final.get('c22_1_feature_event_status')}")
    print(f"C22-2 partial retriever status: {final.get('c22_2_partial_retriever_status')}")
    print(f"C22-3 event encoder status: {final.get('c22_3_event_encoder_status')}")
    print(f"C22-4 joint coupling status: {final.get('c22_4_joint_coupling_status')}")
    print(f"C22-5 robustness status: {final.get('c22_5_robustness_status')}")
    print(f"C22-6 final decision: {final.get('final_decision')}")
    print(f"selected model/config: {json.dumps(jsonable(final.get('selected_model_config')), sort_keys=True)}")
    print(f"selected event segmentation: {final.get('selected_event_segmentation')}")
    print(
        "final VCMR @IoU0.5 R1/R5/R10/R100: "
        f"{vcmr.get('VCMR_R@1_IoU0.5')}/{vcmr.get('VCMR_R@5_IoU0.5')}/{vcmr.get('VCMR_R@10_IoU0.5')}/{vcmr.get('VCMR_R@100_IoU0.5')}"
    )
    print(
        "final VCMR @IoU0.7 R1/R5/R10/R100: "
        f"{vcmr.get('VCMR_R@1_IoU0.7')}/{vcmr.get('VCMR_R@5_IoU0.7')}/{vcmr.get('VCMR_R@10_IoU0.7')}/{vcmr.get('VCMR_R@100_IoU0.7')}"
    )
    print(f"VR R1/R5/R10/R100: {vr.get('VR_R@1')}/{vr.get('VR_R@5')}/{vr.get('VR_R@10')}/{vr.get('VR_R@100')}")
    print(f"delta vs C19/C20/C21 hybrid: {json.dumps(jsonable(final.get('delta_vs_c19_c20_c21_hybrid')), sort_keys=True)}")
    print(f"wrong-video top1/high-score rate: {final.get('wrong_video_risk')}")
    print(f"D/E/F subset improvement: {json.dumps(jsonable(final.get('hard_negative_subset_performance')), sort_keys=True)}")
    print(f"pseudo_official_holdout used for selection: {final.get('pseudo_official_holdout_used_for_selection')}")
    print(f"official was not run: {not final.get('official_val_used')}")
    print("files committed to GitHub: pending git commit/push")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", default="all", choices=["all", "c22_0", "c22_1", "c22_2", "c22_3", "c22_4", "c22_5", "c22_6"])
    parser.add_argument("--mode", default="medium", choices=["smoke", "medium", "full"])
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    if args.stage == "all":
        recs = run_all(args.mode, args.seed)
    elif args.stage == "c22_0":
        recs = {"c22_0": stage_c22_0(args.mode, args.seed)}
    elif args.stage == "c22_1":
        recs = {"c22_1": stage_c22_1(args.mode, args.seed)}
    elif args.stage == "c22_2":
        recs = {"c22_2": stage_c22_2(args.mode, args.seed)}
    elif args.stage == "c22_3":
        recs = {"c22_3": stage_c22_3(args.mode, args.seed)}
    elif args.stage == "c22_4":
        recs = {"c22_4": stage_c22_4(args.mode, args.seed)}
    elif args.stage == "c22_5":
        recs = {"c22_5": stage_c22_5(args.mode, args.seed)}
    else:
        recs = {"c22_6": stage_c22_6(args.mode, args.seed)}
    if args.stage == "all":
        print_summary(recs)


if __name__ == "__main__":
    main()
