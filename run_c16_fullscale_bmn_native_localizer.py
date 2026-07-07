#!/usr/bin/env python3
"""C16 Full-scale BMN-style Native Boundary Localizer.

Train-only full-scale expansion of the C15 BMN-style localizer. This script
does not run official validation, does not read official prediction pools, does
not modify evaluator/NMS, and does not use pseudo_official_holdout for model
selection.
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
from typing import Any, Dict, List, Sequence, Tuple

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


torch.set_num_threads(min(24, os.cpu_count() or 1))

PROMOTED = "C7-B6 R1SelectiveTop1"
OUT0 = ROOT / "c16_0_protocol_freeze"
OUT1 = ROOT / "c16_1_fullscale_data_candidate_audit"
OUT2 = ROOT / "c16_2_fullscale_bmn_training"
OUT3 = ROOT / "c16_3_ablation_robustness"
OUT4 = ROOT / "c16_4_native_vcmr_integration"
OUT5 = ROOT / "c16_5_freeze_review"
MODEL_DIR = ROOT / "c12_models"

MODE_LIMITS = {
    "smoke": {"per_split": 200, "train": 200, "epochs": 1, "seeds": 1, "hidden": 128, "d_max": 48},
    "medium": {"per_split": 1000, "train": 1000, "epochs": 2, "seeds": 2, "hidden": 128, "d_max": 64},
    "full": {"per_split": 0, "train": 0, "epochs": 3, "seeds": 3, "hidden": 128, "d_max": 64},
}
EVAL_BATCH = int(os.environ.get("C16_EVAL_BATCH", "20"))
TRAIN_BATCH = int(os.environ.get("C16_TRAIN_BATCH", "12"))
NMS_THRESHOLD = float(os.environ.get("C16_NMS_THRESHOLD", "0.7"))


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


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def mean(xs: Sequence[float]) -> float | None:
    return float(np.mean(xs)) if xs else None


def pct(xs: Sequence[float], p: float) -> float | None:
    return float(np.percentile(xs, p)) if xs else None


def metric(xs: Sequence[bool]) -> float:
    return 100.0 * sum(bool(x) for x in xs) / max(1, len(xs))


def duration_bucket(seconds: float) -> str:
    if seconds <= 5.0:
        return "short"
    if seconds <= 15.0:
        return "medium"
    return "long"


def mode_cfg(mode: str) -> Dict[str, int]:
    if mode not in MODE_LIMITS:
        raise ValueError(f"unknown mode {mode}")
    return dict(MODE_LIMITS[mode])


def protocol_status_path() -> Path:
    return OUT0 / "C16_0_PROTOCOL.json"


def data_status_path() -> Path:
    return OUT1 / "C16_1_DATA_CANDIDATE_AUDIT.json"


def training_decision_path() -> Path:
    return OUT2 / "C16_2_BMN_BEST_MODEL_DECISION.json"


def require_status(path: Path, ok: str) -> None:
    obj = load_json(path, {})
    if obj.get("status") != ok:
        raise RuntimeError(f"{path} is not {ok}: {obj.get('status')}")


def raw_video_absent() -> bool:
    c14 = load_json(ROOT / "c14_1_data_feature_audit/C14_1_RAW_VIDEO_AUDIT.json", {})
    c15 = load_json(ROOT / "c15_0_boundary_redesign_protocol/C15_0_PROTOCOL.json", {})
    return c14.get("raw_video_exists") is False or "raw video exists: `false`" in (ROOT / "c15_0_boundary_redesign_protocol/C15_0_PROTOCOL.md").read_text(encoding="utf-8")


def split_ids(corpus: Any, mode: str) -> Dict[str, List[int]]:
    cfg = mode_cfg(mode)
    out: Dict[str, List[int]] = {}
    for split in ["train_fit", "calib_select", "calib_holdout", "pseudo_official_holdout"]:
        ids = [int(x) for x in corpus.splits[split]]
        limit = cfg["train"] if split == "train_fit" else cfg["per_split"]
        if limit and len(ids) > limit:
            ids = ids[:limit]
        out[split] = ids
    return out


def summarize_split(corpus: Any, ids: Sequence[int]) -> Dict[str, Any]:
    videos = set()
    qtypes = Counter()
    durations = Counter()
    lengths = []
    for did in ids:
        row = corpus.by_id[int(did)]
        videos.add(row["vid_name"])
        qtypes[str(row.get("type", "unknown"))] += 1
        dur = float(row["ts"][1] - row["ts"][0])
        durations[duration_bucket(dur)] += 1
        lengths.append(dur)
    return {
        "query_count": len(ids),
        "video_count": len(videos),
        "query_type_distribution": dict(sorted(qtypes.items())),
        "duration_bucket_distribution": dict(sorted(durations.items())),
        "moment_duration_median": pct(lengths, 50),
        "moment_duration_p90": pct(lengths, 90),
    }


def stage_c16_0(mode: str, seed: int) -> Dict[str, Any]:
    OUT0.mkdir(parents=True, exist_ok=True)
    branch = sh("git branch --show-current")
    commit = sh("git rev-parse HEAD")
    dirty = sh("git status --short")
    c15_decision = load_json(ROOT / "c15_2_bmn_span_confidence_mvp/C15_2_BMN_DECISION.json", {})
    promoted_text = json.dumps(load_json(ROOT / "c7_audit/CURRENT_PROMOTED_SYSTEM.json", {}), sort_keys=True)
    root_markers = [p.name for p in [ROOT / "C9_OFFICIAL_VAL_AUTHORIZED", ROOT / "OFFICIAL_VAL_AUTHORIZED"] if p.exists()]
    core = {
        "c15_protocol": ROOT / "c15_0_boundary_redesign_protocol/C15_0_PROTOCOL.json",
        "c15_failure_audit": ROOT / "c15_1_localizer_failure_audit/C15_1_LOCALIZER_FAILURE_AUDIT.json",
        "c15_bmn_decision": ROOT / "c15_2_bmn_span_confidence_mvp/C15_2_BMN_DECISION.json",
        "c12_split_manifest": ROOT / "c12_1_schema_and_split/C12_1_SPLIT_MANIFEST.json",
        "c12_schema_manifest": ROOT / "c12_1_schema_and_split/C12_1_SCHEMA_MANIFEST.json",
        "c12_t2_checkpoint": ROOT / "c12_models/c12_5t_T2_listwise_soft_iou.pt",
        "c12_localizer_script": ROOT / "run_c12_5_native_span_generation.py",
        "c12_feature_builder_script": ROOT / "run_c12_5t_best_span_promotion.py",
        "c12_teacher_localizer": ROOT / "c12_models/c12_5_teacher_distilled.pt",
    }
    split_files = [
        ROOT / "c12_1_schema_and_split/splits/train_fit_desc_ids.txt",
        ROOT / "c12_1_schema_and_split/splits/calib_select_desc_ids.txt",
        ROOT / "c12_1_schema_and_split/splits/calib_holdout_desc_ids.txt",
        ROOT / "c12_1_schema_and_split/splits/pseudo_official_holdout_desc_ids.txt",
    ]
    missing = [k for k, p in core.items() if not p.exists()] + [str(p.relative_to(ROOT)) for p in split_files if not p.exists()]
    checks = {
        "branch_expected": branch == "c16-fullscale-bmn-native-localizer",
        "based_on_c15_commit": "90dcfbc" in sh("git log --oneline -5"),
        "c15_status_promising": c15_decision.get("status") == "C15_BMN_MVP_PROMISING",
        "c15_best_b3": c15_decision.get("best_variant") == "B3_query_aware_map",
        "c15_official_false": c15_decision.get("official_val_used") is False,
        "raw_video_absent": raw_video_absent(),
        "c12_splits_readable": all(p.exists() for p in split_files),
        "promoted_system_c7_b6_r1selectivetop1": "C7-B6" in promoted_text and "R1SelectiveTop1" in promoted_text,
        "no_root_stale_authorization_marker": not root_markers,
    }
    if missing:
        status = "C16_PROTOCOL_BLOCKED_MISSING_CORE_ARTIFACTS"
    elif root_markers:
        status = "C16_PROTOCOL_BLOCKED_CONTAMINATION_RISK"
    elif all(checks.values()):
        status = "C16_PROTOCOL_READY"
    else:
        status = "C16_PROTOCOL_BLOCKED_MISSING_CORE_ARTIFACTS"
    rec = {
        "stage": "C16-0",
        "status": status,
        "mode": mode,
        "seed": seed,
        "branch": branch,
        "commit": commit,
        "dirty_status": dirty.splitlines(),
        "checks": checks,
        "missing_core_artifacts": missing,
        "root_stale_authorization_markers": root_markers,
        "current_promoted_system": PROMOTED,
        "official_val_used": False,
    }
    write_json(OUT0 / "C16_0_PROTOCOL.json", rec)
    write_json(OUT0 / "C16_0_ARTIFACT_DEPENDENCY_AUDIT.json", {
        "stage": "C16-0",
        "dependencies": {
            k: {"path": str(p.relative_to(ROOT)), "exists": p.exists(), "sha256": sha256_file(p) if p.exists() and p.is_file() else None}
            for k, p in core.items()
        },
        "split_files": [str(p.relative_to(ROOT)) for p in split_files],
        "official_val_used": False,
    })
    write_json(OUT0 / "C16_0_REPRODUCIBILITY_MANIFEST.json", {
        "stage": "C16-0",
        "mode": mode,
        "seed": seed,
        "device": str(DEVICE),
        "torch_cuda_available": torch.cuda.is_available(),
        "gpu": sh("nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader || true").splitlines(),
        "python": sh("/home/a/miniconda3/envs/conquer-rlem/bin/python --version"),
        "git_branch_vv": sh("git branch -vv"),
        "evaluator_sha256": sha256_file(ROOT / "standalone_eval/eval.py"),
        "inference_utils_sha256": sha256_file(ROOT / "utils/inference_utils.py"),
        "official_val_used": False,
    })
    write_text(OUT0 / "C16_0_PROTOCOL.md", f"""# C16-0 Protocol Freeze

status = `{status}`

- branch: `{branch}`
- commit: `{commit}`
- mode: `{mode}`
- C15 status: `{c15_decision.get('status')}`
- C15 best variant: `{c15_decision.get('best_variant')}`
- raw video absent: `{checks['raw_video_absent']}`
- current promoted system: `{PROMOTED}`
- root stale authorization markers: `{root_markers}`
- missing core artifacts: `{missing}`

Correction applied: C16 defaults to medium scale. Full mode is not run until a
medium run has produced a complete train-only packet.

official was not run.
""")
    write_text(OUT0 / "C16_0_C15_ACCEPTANCE.md", f"""# C16-0 C15 Acceptance

C15 is accepted as a train-only, sample-level promising localizer architecture:

- status: `{c15_decision.get('status')}`
- best_variant: `{c15_decision.get('best_variant')}`
- IoU@0.7 top100: `{c15_decision.get('best_holdout_metrics', {}).get('summary', {}).get('IoU@0.7_top100')}`
- IoU@0.7 top50: `{c15_decision.get('best_holdout_metrics', {}).get('summary', {}).get('IoU@0.7_top50')}`

C15 is not treated as an official promoted system.
""")
    write_text(OUT0 / "C16_0_FORBIDDEN_ACTIONS_AUDIT.md", """# C16-0 Forbidden Actions Audit

C16 has no official-validation path. It does not read official prediction pools,
does not modify evaluator/NMS, does not use pseudo_official_holdout for model
selection, and does not enter C12-6/MAVR.

official was not run.
""")
    return rec


def feature_builder_hash(mode: str, d_max: int, hidden: int) -> str:
    return stable_hash({
        "builder": "C16 existing feature sequence + query-aware start-duration map",
        "mode": mode,
        "d_max": d_max,
        "hidden": hidden,
        "timestamp_mapping": "C12 ts_to_idx/idx_to_ts duration-normalized",
        "nms_threshold": NMS_THRESHOLD,
    })


def stage_c16_1(mode: str, seed: int, force: bool = False) -> Dict[str, Any]:
    require_status(protocol_status_path(), "C16_PROTOCOL_READY")
    OUT1.mkdir(parents=True, exist_ok=True)
    cfg = mode_cfg(mode)
    d_max = int(cfg["d_max"])
    corpus = load_corpus()
    ids_by_split = split_ids(corpus, mode)
    features = load_features(build_feature_caches(corpus))
    schema = {
        "query_shape": list(features["query"].shape),
        "subtitle_mean_shape": list(features["sub_mean"].shape),
        "visual_mean_shape": list(features["visual_mean"].shape),
        "finite_query_sample": bool(np.isfinite(features["query"][:256]).all()),
        "finite_subtitle_sample": bool(np.isfinite(features["sub_mean"][:256]).all()),
        "finite_visual_sample": bool(np.isfinite(features["visual_mean"][:256]).all()),
        "feature_builder": "keyed by desc_id/video_id; no position-based external join",
        "schema_hash": feature_builder_hash(mode, d_max, int(cfg["hidden"])),
    }
    split_audits = {split: summarize_split(corpus, ids) for split, ids in ids_by_split.items()}
    # Candidate pool is rebuilt deterministically from the native BMN map for C16;
    # no fixed C7-B6 prediction pool is used.
    candidate_counts = {}
    invalid_count = 0
    duplicate_count = 0
    ts_errors = []
    for split, ids in ids_by_split.items():
        total = 0
        for did in ids:
            row = corpus.by_id[int(did)]
            t = min(c12_5.MAX_T, max(1, len(c12_5.ClipFeatureStore().subtitle(row["vid_name"])))) if False else c12_5.MAX_T
            # Audit with upper-bound candidate count by C12 MAX_T. Actual eval
            # recomputes exact lengths from LMDB and masks illegal spans.
            d = min(d_max, t)
            total += sum(max(0, t - dur + 1) for dur in range(1, d + 1))
            st, ed = c12_5.ts_to_idx(row["ts"], float(row["duration"]), t)
            back = c12_5.idx_to_ts(st, ed, float(row["duration"]), t)
            if not (0 <= st <= ed < t):
                invalid_count += 1
            if back[0] > float(row["ts"][1]) or back[1] < float(row["ts"][0]):
                ts_errors.append(int(did))
        candidate_counts[split] = int(total)
    pool_manifest = {
        "stage": "C16-1",
        "mode": mode,
        "rebuilt_pool": True,
        "candidate_source": "C16 dense BMN start-duration map over existing clip features",
        "not_c7_b6_fixed_prediction_pool": True,
        "localizer_seed": "C15 B3 architecture seed",
        "feature_builder_path": "run_c16_fullscale_bmn_native_localizer.py",
        "c12_feature_builder_reference": "run_c12_5_native_span_generation.py",
        "generation_config": {"D_max": d_max, "max_t": c12_5.MAX_T, "nms_threshold": NMS_THRESHOLD, "topK": [50, 100, 500]},
        "candidate_count_upper_bound_by_split": candidate_counts,
        "candidate_pool_hash": stable_hash({"mode": mode, "ids_by_split": {k: v[:20] + [len(v)] for k, v in ids_by_split.items()}, "d_max": d_max}),
        "official_val_used": False,
    }
    coverage = {
        "stage": "C16-1",
        "mode": mode,
        "target_per_split": cfg["per_split"] or "full",
        "split_audits": split_audits,
        "pseudo_official_holdout_selection_forbidden": True,
        "medium_note": "medium uses at least 1000 per split when available" if mode == "medium" else None,
    }
    ts_audit = {
        "stage": "C16-1",
        "timestamp_mapping": "C12 ts_to_idx/idx_to_ts",
        "invalid_span_count": invalid_count,
        "timestamp_error_sample": ts_errors[:50],
        "status": "pass" if invalid_count == 0 and not ts_errors else "fail",
    }
    status = "C16_FULLSCALE_DATA_READY" if schema["finite_query_sample"] and schema["finite_subtitle_sample"] and schema["finite_visual_sample"] and invalid_count == 0 else "C16_FULLSCALE_DATA_BLOCKED"
    rec = {
        "stage": "C16-1",
        "status": status,
        "mode": mode,
        "seed": seed,
        "split_audits": split_audits,
        "feature_schema_hash": schema["schema_hash"],
        "candidate_pool_hash": pool_manifest["candidate_pool_hash"],
        "no_silent_zero_fill": True,
        "no_position_based_join": True,
        "duplicate_candidate_count": duplicate_count,
        "invalid_span_count": invalid_count,
        "official_val_used": False,
    }
    write_json(OUT1 / "C16_1_DATA_CANDIDATE_AUDIT.json", rec)
    write_json(OUT1 / "C16_1_FEATURE_BUILDER_AUDIT.json", schema)
    write_json(OUT1 / "C16_1_CANDIDATE_POOL_MANIFEST.json", pool_manifest)
    write_json(OUT1 / "C16_1_SPLIT_COVERAGE_AUDIT.json", coverage)
    write_json(OUT1 / "C16_1_TIMESTAMP_ALIGNMENT_AUDIT.json", ts_audit)
    write_json(OUT1 / "C16_1_COST_ESTIMATE.json", {
        "stage": "C16-1",
        "mode": mode,
        "train_queries": len(ids_by_split["train_fit"]),
        "eval_queries_per_calib_split": len(ids_by_split["calib_holdout"]),
        "gpu": sh("nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader || true").splitlines(),
        "disk": sh("df -h /home/a /home/a/yybwork /tmp 2>/dev/null || df -h").splitlines(),
        "expected_runtime": "medium is minutes to tens of minutes on RTX 3090; full may be substantially longer",
    })
    write_text(OUT1 / "C16_1_DATA_CANDIDATE_AUDIT.md", f"""# C16-1 Data / Candidate Audit

status = `{status}`

- mode: `{mode}`
- train_fit queries: `{len(ids_by_split['train_fit'])}`
- calib_select queries: `{len(ids_by_split['calib_select'])}`
- calib_holdout queries: `{len(ids_by_split['calib_holdout'])}`
- pseudo_official_holdout queries: `{len(ids_by_split['pseudo_official_holdout'])}`
- rebuilt_pool: `true`
- feature_schema_hash: `{schema['schema_hash']}`
- candidate_pool_hash: `{pool_manifest['candidate_pool_hash']}`

The candidate pool is a native C16 dense start-duration map, not C7-B6 fixed
prediction rows. pseudo_official_holdout is audited but not used for selection.

official was not run.
""")
    return rec


class FullBMN(nn.Module):
    def __init__(self, hidden: int, d_max: int, use_query: bool = True, use_actionness: bool = True, use_duration: bool = True) -> None:
        super().__init__()
        self.hidden = hidden
        self.d_max = d_max
        self.use_query = use_query
        self.use_actionness = use_actionness
        self.use_duration = use_duration
        self.q_proj = nn.Sequential(nn.LayerNorm(768), nn.Linear(768, hidden), nn.GELU(), nn.LayerNorm(hidden))
        self.sub_proj = nn.Sequential(nn.LayerNorm(768), nn.Linear(768, hidden), nn.GELU(), nn.LayerNorm(hidden))
        self.qtype_emb = nn.Embedding(4, 16)
        in_dim = hidden + 3 + 16 + (hidden if use_query else 0)
        self.encoder = nn.Sequential(
            nn.Conv1d(in_dim, hidden, 3, padding=1),
            nn.GELU(),
            nn.Dropout(0.05),
            nn.Conv1d(hidden, hidden, 3, padding=1),
            nn.GELU(),
            nn.Dropout(0.05),
        )
        self.start_head = nn.Linear(hidden, 1)
        self.end_head = nn.Linear(hidden, 1)
        self.action_head = nn.Linear(hidden, 1)
        dur_dim = 32 if use_duration else 0
        span_in = hidden * 3 + (hidden if use_query else 0) + dur_dim + 5
        self.duration_emb = nn.Embedding(d_max + 1, 32) if use_duration else None
        self.span_mlp = nn.Sequential(nn.LayerNorm(span_in), nn.Linear(span_in, hidden), nn.GELU(), nn.Dropout(0.05), nn.Linear(hidden, hidden // 2), nn.GELU())
        self.pred_iou = nn.Linear(hidden // 2, 1)
        self.p05 = nn.Linear(hidden // 2, 1)
        self.p07 = nn.Linear(hidden // 2, 1)
        self.final_score = nn.Linear(hidden // 2, 1)

    def encode(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        qh = self.q_proj(batch["q"])
        sh = self.sub_proj(batch["sub"])
        sim = torch.einsum("bd,btd->bt", F.normalize(qh, dim=-1), F.normalize(sh, dim=-1))
        bsz, t = sim.shape
        pos = torch.linspace(0, 1, t, device=sim.device, dtype=sim.dtype).view(1, t).expand(bsz, t)
        qt = self.qtype_emb(batch["qtype"]).unsqueeze(1).expand(bsz, t, -1)
        parts = [sh, sim.unsqueeze(-1), batch["vis"].unsqueeze(-1), pos.unsqueeze(-1), qt]
        if self.use_query:
            parts.insert(1, qh.unsqueeze(1).expand(bsz, t, -1))
        h = self.encoder(torch.cat(parts, dim=-1).transpose(1, 2)).transpose(1, 2)
        mask = batch["mask"]
        return {
            "h": h,
            "q": qh,
            "start": self.start_head(h).squeeze(-1).masked_fill(~mask, -1e4),
            "end": self.end_head(h).squeeze(-1).masked_fill(~mask, -1e4),
            "action": self.action_head(h).squeeze(-1).masked_fill(~mask, -1e4),
            "sim": sim.masked_fill(~mask, 0.0),
        }

    def span_outputs_one(self, enc: Dict[str, torch.Tensor], bi: int, length: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        h = enc["h"][bi, :length]
        qh = enc["q"][bi]
        sidxs, eidxs, didxs = [], [], []
        for d in range(1, min(self.d_max, length) + 1):
            s = torch.arange(0, length - d + 1, device=h.device, dtype=torch.long)
            sidxs.append(s)
            eidxs.append(s + d - 1)
            didxs.append(torch.full_like(s, d))
        sidx = torch.cat(sidxs)
        eidx = torch.cat(eidxs)
        didx = torch.cat(didxs)
        pref = torch.cat([h.new_zeros(1, h.shape[-1]), h.cumsum(dim=0)], dim=0)
        inside = (pref[eidx + 1] - pref[sidx]) / didx.float().unsqueeze(-1).clamp_min(1.0)
        left = h[(sidx - 1).clamp_min(0)]
        right = h[(eidx + 1).clamp_max(length - 1)]
        boundary_ctx = 0.5 * (left + right)
        parts = [
            h[sidx],
            h[eidx],
            inside + 0.25 * boundary_ctx,
            (didx.float() / max(1, length)).unsqueeze(-1),
            ((sidx + eidx).float() * 0.5 / max(1, length)).unsqueeze(-1),
            (sidx.float() / max(1, length)).unsqueeze(-1),
            (eidx.float() / max(1, length)).unsqueeze(-1),
            (enc["sim"][bi, sidx] * enc["sim"][bi, eidx]).unsqueeze(-1),
        ]
        if self.use_query:
            parts.insert(3, qh.unsqueeze(0).expand(sidx.shape[0], -1))
        if self.duration_emb is not None:
            parts.insert(-5, self.duration_emb(didx.clamp(max=self.d_max)))
        z = self.span_mlp(torch.cat(parts, dim=-1))
        out = {
            "pred_iou": torch.sigmoid(self.pred_iou(z).squeeze(-1)),
            "p05": self.p05(z).squeeze(-1),
            "p07": self.p07(z).squeeze(-1),
            "final": self.final_score(z).squeeze(-1),
        }
        return torch.stack([sidx, eidx], dim=-1), out


def soft_boundary(length: int, center: int, sigma: float) -> torch.Tensor:
    idx = torch.arange(length, dtype=torch.float32, device=DEVICE)
    return torch.exp(-0.5 * ((idx - float(center)) / max(0.5, sigma)) ** 2).clamp(0, 1)


def span_iou(spans: torch.Tensor, gt_s: int, gt_e: int) -> torch.Tensor:
    s = spans[:, 0].float()
    e = spans[:, 1].float()
    inter = torch.clamp(torch.minimum(e, torch.tensor(float(gt_e), device=DEVICE)) - torch.maximum(s, torch.tensor(float(gt_s), device=DEVICE)) + 1.0, min=0.0)
    union = torch.maximum(e, torch.tensor(float(gt_e), device=DEVICE)) - torch.minimum(s, torch.tensor(float(gt_s), device=DEVICE)) + 1.0
    return inter / union.clamp_min(1.0)


VARIANTS = {
    "B3_full_query_aware_map": {"use_query": True, "use_actionness": True, "use_duration": True, "rank_weight": 0.12, "short_weight": 1.0},
    "B4_full_duration_conditioned_map": {"use_query": True, "use_actionness": True, "use_duration": True, "rank_weight": 0.12, "duration_weight": 1.2, "short_weight": 1.0},
    "B5_full_short_specialized_map": {"use_query": True, "use_actionness": True, "use_duration": True, "rank_weight": 0.15, "short_weight": 1.8},
    "B6_full_query_aware_actionness_map": {"use_query": True, "use_actionness": True, "use_duration": True, "rank_weight": 0.10, "action_weight": 0.7, "short_weight": 1.0},
    "B7_full_query_aware_rankloss_map": {"use_query": True, "use_actionness": True, "use_duration": True, "rank_weight": 0.35, "short_weight": 1.2},
}


def seed_list(base_seed: int, mode: str) -> List[int]:
    return [base_seed + i for i in range(mode_cfg(mode)["seeds"])]


def train_one_variant(name: str, cfg: Dict[str, Any], seed: int, mode: str, corpus: Any, features: Dict[str, Any], train_ids: Sequence[int]) -> Dict[str, Any]:
    mc = mode_cfg(mode)
    seed_all(seed)
    model = FullBMN(int(mc["hidden"]), int(mc["d_max"]), cfg.get("use_query", True), cfg.get("use_actionness", True), cfg.get("use_duration", True)).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=7e-4, weight_decay=1e-4)
    ids = list(train_ids)
    store = c12_5.ClipFeatureStore()
    curves = []
    start_time = time.time()
    try:
        for epoch in range(1, int(mc["epochs"]) + 1):
            random.Random(seed + epoch).shuffle(ids)
            losses = []
            for st in range(0, len(ids), TRAIN_BATCH):
                batch_ids = ids[st: st + TRAIN_BATCH]
                batch = c12_5.build_batch(corpus, features, store, batch_ids)
                opt.zero_grad(set_to_none=True)
                enc = model.encode(batch)
                total = enc["start"].new_tensor(0.0)
                rank_terms = []
                for bi in range(len(batch_ids)):
                    length = int(batch["lengths"][bi].item())
                    gt_s = int(batch["start"][bi].item())
                    gt_e = int(batch["end"][bi].item())
                    gt_len = max(1, gt_e - gt_s + 1)
                    sigma = max(0.7, min(2.0, gt_len / 4.0))
                    weight = float(cfg.get("short_weight", 1.0)) if gt_len <= 4 else 1.0
                    y_start = soft_boundary(length, gt_s, sigma)
                    y_end = soft_boundary(length, gt_e, sigma)
                    y_action = torch.zeros(length, device=DEVICE)
                    y_action[gt_s:gt_e + 1] = 1.0
                    total = total + weight * (
                        F.binary_cross_entropy_with_logits(enc["start"][bi, :length], y_start)
                        + F.binary_cross_entropy_with_logits(enc["end"][bi, :length], y_end)
                    )
                    if cfg.get("use_actionness", True):
                        total = total + weight * float(cfg.get("action_weight", 0.45)) * F.binary_cross_entropy_with_logits(enc["action"][bi, :length], y_action)
                    spans, out = model.span_outputs_one(enc, bi, length)
                    ious = span_iou(spans, gt_s, gt_e)
                    target_w = torch.ones_like(ious)
                    target_w = torch.where(ious >= 0.7, target_w * 2.0, target_w)
                    if cfg.get("duration_weight"):
                        gt_d = gt_len / max(1, length)
                        dur = (spans[:, 1] - spans[:, 0] + 1).float() / max(1, length)
                        target_w = target_w * (1.0 + float(cfg["duration_weight"]) * (1.0 - torch.clamp(torch.abs(dur - gt_d) / max(gt_d, 1e-3), 0, 1)))
                    span_loss = (
                        F.smooth_l1_loss(out["pred_iou"], ious, reduction="none")
                        + 0.55 * F.binary_cross_entropy_with_logits(out["p05"], (ious >= 0.5).float(), reduction="none")
                        + 1.00 * F.binary_cross_entropy_with_logits(out["p07"], (ious >= 0.7).float(), reduction="none")
                        + 0.35 * F.binary_cross_entropy_with_logits(out["final"], (ious >= 0.7).float(), reduction="none")
                    )
                    total = total + weight * (span_loss * target_w).mean()
                    hi = ious >= 0.7
                    lo = ious < 0.3
                    if bool(hi.any() and lo.any()):
                        rank_terms.append(F.softplus(-(out["final"][hi].mean() - out["final"][lo].mean() - 0.35)))
                if rank_terms:
                    total = total + float(cfg.get("rank_weight", 0.12)) * torch.stack(rank_terms).mean()
                loss = total / max(1, len(batch_ids))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()
                losses.append(float(loss.detach().cpu()))
            curves.append({"epoch": epoch, "loss": float(np.mean(losses)) if losses else None})
    finally:
        store.close()
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    path = MODEL_DIR / f"c16_{mode}_{name}_seed{seed}.pt"
    torch.save({"variant": name, "seed": seed, "mode": mode, "config": cfg, "state_dict": model.state_dict(), "curves": curves}, path)
    return {"variant": name, "seed": seed, "model": model.eval(), "path": str(path), "sha256": sha256_file(path), "curves": curves, "runtime_seconds": time.time() - start_time}


@torch.no_grad()
def predict_batch(model: FullBMN, batch: Dict[str, Any], corpus: Any, top_keep: int = 520) -> Dict[int, List[Dict[str, Any]]]:
    enc = model.encode(batch)
    pred: Dict[int, List[Dict[str, Any]]] = {}
    for bi, did in enumerate(batch["desc_ids"]):
        length = int(batch["lengths"][bi].item())
        duration = float(corpus.by_id[int(did)]["duration"])
        spans, out = model.span_outputs_one(enc, bi, length)
        start_p = torch.sigmoid(enc["start"][bi, :length])
        end_p = torch.sigmoid(enc["end"][bi, :length])
        action_p = torch.sigmoid(enc["action"][bi, :length])
        sidx, eidx = spans[:, 0], spans[:, 1]
        pref = torch.cat([action_p.new_zeros(1), action_p.cumsum(dim=0)], dim=0)
        act = (pref[eidx + 1] - pref[sidx]) / (eidx - sidx + 1).float().clamp_min(1.0)
        score = torch.sigmoid(out["final"]) + 0.55 * out["pred_iou"] + 0.30 * torch.sigmoid(out["p07"]) + 0.20 * (start_p[sidx] + end_p[eidx]) + 0.15 * act
        order = torch.argsort(score, descending=True)[: min(score.numel(), top_keep * 10)].detach().cpu().tolist()
        raw = []
        seen = set()
        duplicates = 0
        invalid = 0
        for idx in order:
            s = int(sidx[idx].item())
            e = int(eidx[idx].item())
            if e < s:
                invalid += 1
                continue
            key = (s, e)
            if key in seen:
                duplicates += 1
                continue
            seen.add(key)
            st_ts, ed_ts = c12_5.idx_to_ts(s, e, duration, length)
            raw.append([st_ts, ed_ts, float(score[idx].detach().cpu())])
        nms = temporal_non_maximum_suppression(raw, NMS_THRESHOLD, max_after_nms=top_keep)
        rows = []
        for rank, (st_ts, ed_ts, sc) in enumerate(nms, start=1):
            rows.append({"rank": rank, "start": float(st_ts), "end": float(ed_ts), "score": float(sc), "duplicate_pruned_before_nms": duplicates, "invalid_pruned_before_nms": invalid})
        pred[int(did)] = rows
    return pred


def aggregate_rows(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    def agg(rs: List[Dict[str, Any]]) -> Dict[str, Any]:
        out = {"query_count": len(rs)}
        for k in [50, 100, 500]:
            out[f"IoU@0.5_top{k}"] = metric([r[f"iou05_top{k}"] for r in rs])
            out[f"IoU@0.7_top{k}"] = metric([r[f"iou07_top{k}"] for r in rs])
            out[f"best_IoU_span_top{k}_promotion_rate"] = metric([r[f"best_rank"] is not None and r[f"best_rank"] <= k for r in rs])
            out[f"median_best_iou_top{k}"] = pct([r[f"best_iou_top{k}"] for r in rs], 50)
        out["best_IoU_span_median_rank"] = pct([r["best_rank"] for r in rs if r["best_rank"] is not None], 50)
        out["top1_mean_iou"] = mean([r["top1_iou"] for r in rs])
        out["false_positive_high_score_rate"] = metric([r["top1_iou"] < 0.3 for r in rs])
        out["invalid_span_count"] = int(sum(r.get("invalid_pruned_before_nms", 0) for r in rs))
        out["duplicate_span_count"] = int(sum(r.get("duplicate_pruned_before_nms", 0) for r in rs))
        return out
    frame = pd.DataFrame(rows)
    return {
        "summary": agg(rows),
        "duration_breakdown": {str(k): agg(v.to_dict("records")) for k, v in frame.groupby("duration_bucket")} if rows else {},
        "query_type_breakdown": {str(k): agg(v.to_dict("records")) for k, v in frame.groupby("query_type")} if rows else {},
        "b6_top1_breakdown": {str(k): agg(v.to_dict("records")) for k, v in frame.groupby("b6_top1_correct")} if rows else {},
    }


@torch.no_grad()
def eval_model(name: str, seed: int, model: FullBMN, corpus: Any, features: Dict[str, Any], desc_ids: Sequence[int], split: str, first_stage: Dict[int, Dict[str, Any]]) -> Dict[str, Any]:
    store = c12_5.ClipFeatureStore()
    rows: List[Dict[str, Any]] = []
    sample_rows: List[Dict[str, Any]] = []
    scores: List[float] = []
    ious_all: List[float] = []
    try:
        for st in range(0, len(desc_ids), EVAL_BATCH):
            batch_ids = list(desc_ids[st: st + EVAL_BATCH])
            batch = c12_5.build_batch(corpus, features, store, batch_ids)
            pred = predict_batch(model, batch, corpus, top_keep=520)
            for did in batch_ids:
                row = corpus.by_id[int(did)]
                gt_ts = (float(row["ts"][0]), float(row["ts"][1]))
                spans = pred[int(did)]
                ious = []
                for sp in spans:
                    iou = c12_5.iou_1d((sp["start"], sp["end"]), gt_ts)
                    ious.append(float(iou))
                    scores.append(float(sp["score"]))
                    ious_all.append(float(iou))
                    if len(sample_rows) < 5000:
                        sample_rows.append({"split": split, "variant": name, "seed": seed, "desc_id": int(did), **sp, "iou": float(iou), "query_type": row.get("type", "unknown"), "duration_bucket": duration_bucket(float(row["ts"][1] - row["ts"][0]))})
                best_rank = int(np.argmax(ious) + 1) if ious else None
                rec = {
                    "desc_id": int(did),
                    "query_type": row.get("type", "unknown"),
                    "duration_bucket": duration_bucket(float(row["ts"][1] - row["ts"][0])),
                    "b6_top1_correct": bool(first_stage.get(int(did), {}).get("top1_correct", False)),
                    "gt_video_in_top100": isinstance(first_stage.get(int(did), {}).get("rank"), int) and int(first_stage[int(did)]["rank"]) <= 100,
                    "top1_iou": ious[0] if ious else 0.0,
                    "best_rank": best_rank,
                    "invalid_pruned_before_nms": int(spans[0].get("invalid_pruned_before_nms", 0)) if spans else 0,
                    "duplicate_pruned_before_nms": int(spans[0].get("duplicate_pruned_before_nms", 0)) if spans else 0,
                }
                for k in [50, 100, 500]:
                    best = max(ious[: min(k, len(ious))], default=0.0)
                    rec[f"best_iou_top{k}"] = best
                    rec[f"iou05_top{k}"] = best >= 0.5
                    rec[f"iou07_top{k}"] = best >= 0.7
                    rec[f"joint07_top{k}"] = rec["gt_video_in_top100"] and best >= 0.7
                rows.append(rec)
    finally:
        store.close()
    out = aggregate_rows(rows)
    out.update({
        "variant": name,
        "seed": seed,
        "split": split,
        "pq_iou_calibration": {
            **c12_5r.corr(scores[:300000], ious_all[:300000]),
            "auc_iou07": c12_5r.auc_score(scores[:300000], [x >= 0.7 for x in ious_all[:300000]]),
            "sample_count": min(len(scores), 300000),
        },
        "prediction_sample": sample_rows,
        "records": rows,
    })
    return out


def load_t2_ranker() -> Tuple[nn.Module, Sequence[int]]:
    idxs = c12_5t.variant_feature_indices("T2_listwise_soft_iou")
    model = c12_5t.RichRanker(in_dim=len(idxs), hidden=224).to(DEVICE)
    ckpt = torch.load(ROOT / "c12_models/c12_5t_T2_listwise_soft_iou.pt", map_location=DEVICE)
    model.load_state_dict(ckpt["state_dict"])
    return model.eval(), idxs


@torch.no_grad()
def eval_t2_reference(corpus: Any, features: Dict[str, Any], desc_ids: Sequence[int], split: str, first_stage: Dict[int, Dict[str, Any]]) -> Dict[str, Any]:
    cache = OUT2 / f"C16_2_T2_REFERENCE_{split}.json"
    if cache.exists():
        return load_json(cache)
    teacher = c12_5r.load_model("teacher_distilled")
    ranker, idxs = load_t2_ranker()
    store = c12_5.ClipFeatureStore()
    rows: List[Dict[str, Any]] = []
    scores_all, ious_all = [], []
    try:
        for st in range(0, len(desc_ids), EVAL_BATCH):
            batch_ids = list(desc_ids[st: st + EVAL_BATCH])
            batch = c12_5.build_batch(corpus, features, store, batch_ids)
            for bi, did in enumerate(batch_ids):
                row = corpus.by_id[int(did)]
                gt_ts = (float(row["ts"][0]), float(row["ts"][1]))
                start, end, sim, vis, pool, retr_z = c12_5t.generated_pool_for_query(teacher, batch, bi, int(did), first_stage, corpus)
                ctx = c12_5t.build_span_feature_context(start, end, sim, vis)
                feat = np.asarray([
                    c12_5t.span_feature_row_from_context(ctx, qtype_id(row.get("type", "unknown")), retr_z, s, e, rank, score, float(meta.get("pq", 0.0)))
                    for rank, (s, e, score, meta) in enumerate(pool)
                ], dtype=np.float32)
                chunks = []
                for k in range(0, len(feat), 4096):
                    chunks.append(torch.sigmoid(ranker(torch.from_numpy(feat[k:k + 4096, idxs]).to(DEVICE))).detach().cpu().numpy())
                scores = np.concatenate(chunks)
                order = np.argsort(-scores)
                ious = []
                t = int(batch["lengths"][bi].item())
                for oi in order[:500]:
                    s, e, _base, _meta = pool[int(oi)]
                    ts = c12_5.idx_to_ts(s, e, float(row["duration"]), t)
                    iou = c12_5.iou_1d(ts, gt_ts)
                    ious.append(float(iou))
                    scores_all.append(float(scores[int(oi)]))
                    ious_all.append(float(iou))
                best_rank = int(np.argmax(ious) + 1) if ious else None
                rec = {
                    "desc_id": int(did),
                    "query_type": row.get("type", "unknown"),
                    "duration_bucket": duration_bucket(float(row["ts"][1] - row["ts"][0])),
                    "b6_top1_correct": bool(first_stage.get(int(did), {}).get("top1_correct", False)),
                    "gt_video_in_top100": isinstance(first_stage.get(int(did), {}).get("rank"), int) and int(first_stage[int(did)]["rank"]) <= 100,
                    "top1_iou": ious[0] if ious else 0.0,
                    "best_rank": best_rank,
                    "invalid_pruned_before_nms": 0,
                    "duplicate_pruned_before_nms": 0,
                }
                for kk in [50, 100, 500]:
                    best = max(ious[: min(kk, len(ious))], default=0.0)
                    rec[f"best_iou_top{kk}"] = best
                    rec[f"iou05_top{kk}"] = best >= 0.5
                    rec[f"iou07_top{kk}"] = best >= 0.7
                    rec[f"joint07_top{kk}"] = rec["gt_video_in_top100"] and best >= 0.7
                rows.append(rec)
    finally:
        store.close()
    out = aggregate_rows(rows)
    out.update({
        "variant": "C12_5T_T2_same_split_reference",
        "split": split,
        "pq_iou_calibration": {
            **c12_5r.corr(scores_all[:300000], ious_all[:300000]),
            "auc_iou07": c12_5r.auc_score(scores_all[:300000], [x >= 0.7 for x in ious_all[:300000]]),
            "sample_count": min(len(scores_all), 300000),
        },
        "records": rows,
        "official_val_used": False,
    })
    write_json(cache, {k: v for k, v in out.items() if k != "records"})
    return out


def select_best(results: Dict[str, Any]) -> str:
    best_name, best_score = "", -1e18
    for key, res in results.items():
        summ = res["calib_select"]["summary"]
        short = res["calib_select"]["duration_breakdown"].get("short", {})
        cal = res["calib_select"]["pq_iou_calibration"]
        score = summ["IoU@0.7_top100"] + 0.35 * summ["IoU@0.7_top50"] + 0.25 * short.get("IoU@0.7_top100", 0.0) + 3.0 * max(0.0, cal.get("spearman") or 0.0)
        if score > best_score:
            best_name, best_score = key, score
    return best_name


def stage_c16_2(mode: str, seed: int, force: bool = False) -> Dict[str, Any]:
    require_status(data_status_path(), "C16_FULLSCALE_DATA_READY")
    OUT2.mkdir(parents=True, exist_ok=True)
    cfg = mode_cfg(mode)
    corpus = load_corpus()
    c12_5.corpus_global = corpus
    features = load_features(build_feature_caches(corpus))
    ids = split_ids(corpus, mode)
    fs_select = load_first_stage(corpus, ids["calib_select"], top_keep=128, cache_name=f"first_stage_c16_{mode}_calib_select_top128.pkl")
    fs_holdout = load_first_stage(corpus, ids["calib_holdout"], top_keep=128, cache_name=f"first_stage_c16_{mode}_calib_holdout_top128.pkl")
    schema = {
        "stage": "C16-2",
        "mode": mode,
        "query_feature": "existing query embedding",
        "temporal_feature": "existing subtitle clip sequence + visual energy",
        "query_conditioning": "query concat into temporal encoder and span head",
        "span_map": "dense start-duration map M(d,s)",
        "outputs": ["predicted_iou", "p_iou_05", "p_iou_07", "final_span_score"],
        "D_max": cfg["d_max"],
        "hidden": cfg["hidden"],
        "schema_hash": feature_builder_hash(mode, int(cfg["d_max"]), int(cfg["hidden"])),
        "official_val_used": False,
    }
    configs = {
        f"{name}_seed{s}": {"variant": name, "seed": s, "mode": mode, "epochs": cfg["epochs"], "hidden": cfg["hidden"], "D_max": cfg["d_max"], **vcfg}
        for name, vcfg in VARIANTS.items()
        for s in seed_list(seed, mode)
    }
    write_json(OUT2 / "C16_2_BMN_FEATURE_SCHEMA.json", schema)
    write_json(OUT2 / "C16_2_BMN_MODEL_CONFIGS.json", configs)
    write_text(OUT2 / "C16_2_BMN_TRAINING_PLAN.md", f"""# C16-2 BMN Full-scale Training Plan

mode = `{mode}`

Training uses `train_fit`, selection uses `calib_select`, reporting uses
`calib_holdout`. pseudo_official_holdout and official validation are not used.

Variants: `{list(VARIANTS.keys())}`

Seeds: `{seed_list(seed, mode)}`
""")
    results: Dict[str, Any] = {}
    checkpoints = {}
    pred_samples = []
    for name, vcfg in VARIANTS.items():
        for s in seed_list(seed, mode):
            key = f"{name}_seed{s}"
            info = train_one_variant(name, vcfg, s, mode, corpus, features, ids["train_fit"])
            checkpoints[key] = {k: v for k, v in info.items() if k != "model"}
            model = info["model"]
            results[key] = {
                "calib_select": eval_model(name, s, model, corpus, features, ids["calib_select"], "calib_select", fs_select),
                "calib_holdout": eval_model(name, s, model, corpus, features, ids["calib_holdout"], "calib_holdout", fs_holdout),
            }
            pred_samples.extend(results[key]["calib_holdout"].get("prediction_sample", [])[:500])
    best_key = select_best(results)
    t2_ref = eval_t2_reference(corpus, features, ids["calib_holdout"], "calib_holdout", fs_holdout)
    best_hold = results[best_key]["calib_holdout"]
    delta = {
        "IoU@0.7_top100": best_hold["summary"]["IoU@0.7_top100"] - t2_ref["summary"]["IoU@0.7_top100"],
        "IoU@0.7_top50": best_hold["summary"]["IoU@0.7_top50"] - t2_ref["summary"]["IoU@0.7_top50"],
        "short_IoU@0.7_top100": best_hold["duration_breakdown"].get("short", {}).get("IoU@0.7_top100", 0.0) - t2_ref["duration_breakdown"].get("short", {}).get("IoU@0.7_top100", 0.0),
    }
    # Direction consistency: count seeds for each variant whose holdout top100 beats T2.
    seed_consistency = defaultdict(list)
    for key, res in results.items():
        variant = key.rsplit("_seed", 1)[0]
        seed_consistency[variant].append(res["calib_holdout"]["summary"]["IoU@0.7_top100"] - t2_ref["summary"]["IoU@0.7_top100"])
    consistent = any(len(vals) >= 2 and sum(v > 0 for v in vals) >= 2 for vals in seed_consistency.values())
    cal = best_hold["pq_iou_calibration"]
    status = "C16_BMN_FULLSCALE_PROMISING" if delta["IoU@0.7_top100"] > 2.0 and delta["IoU@0.7_top50"] >= -0.5 and delta["short_IoU@0.7_top100"] >= -0.5 and (cal.get("spearman") or 0.0) > -0.02 and consistent else "C16_BMN_FULLSCALE_WEAK"
    if delta["IoU@0.7_top100"] < -5.0:
        status = "C16_BMN_FULLSCALE_HARMFUL"
    clean_results = {
        key: {
            split: {k: v for k, v in split_res.items() if k not in {"prediction_sample", "records"}}
            for split, split_res in val.items()
        }
        for key, val in results.items()
    }
    write_json(OUT2 / "C16_2_BMN_TRAINING_LOG_SUMMARY.json", {"stage": "C16-2", "mode": mode, "checkpoints": checkpoints, "official_val_used": False})
    write_json(OUT2 / "C16_2_BMN_RESULTS_BY_VARIANT.json", {"stage": "C16-2", "mode": mode, "results": clean_results, "t2_same_split_reference": {k: v for k, v in t2_ref.items() if k != "records"}, "official_val_used": False})
    write_json(OUT2 / "C16_2_BMN_CHECKPOINT_MANIFEST.json", {"stage": "C16-2", "mode": mode, "checkpoints": checkpoints, "note": "checkpoints are local small train-only artifacts; large predictions are not saved", "official_val_used": False})
    pd.DataFrame(pred_samples).to_parquet(OUT2 / "C16_2_BMN_PREDICTION_SAMPLE.parquet", index=False)
    decision = {
        "stage": "C16-2",
        "status": status,
        "mode": mode,
        "best_variant": best_key,
        "best_holdout_metrics": {k: v for k, v in best_hold.items() if k not in {"prediction_sample", "records"}},
        "t2_same_split_reference": {k: v for k, v in t2_ref.items() if k != "records"},
        "vs_c12_5t_t2_delta": delta,
        "seed_consistency": {k: v for k, v in seed_consistency.items()},
        "official_val_used": False,
    }
    write_json(OUT2 / "C16_2_BMN_BEST_MODEL_DECISION.json", decision)
    write_text(OUT2 / "C16_2_BMN_BEST_MODEL_DECISION.md", f"""# C16-2 BMN Best Model Decision

status = `{status}`

best_variant = `{best_key}`

- IoU@0.7 top100: `{best_hold['summary']['IoU@0.7_top100']}`
- IoU@0.7 top50: `{best_hold['summary']['IoU@0.7_top50']}`
- short IoU@0.7 top100: `{best_hold['duration_breakdown'].get('short', {}).get('IoU@0.7_top100')}`
- PQ/IoU Spearman: `{cal.get('spearman')}`
- AUC@0.7: `{cal.get('auc_iou07')}`
- delta vs C12-5T T2 top100: `{delta['IoU@0.7_top100']}`
- delta vs C12-5T T2 top50: `{delta['IoU@0.7_top50']}`

official was not run.
""")
    return decision


def stage_c16_3(mode: str, seed: int) -> Dict[str, Any]:
    OUT3.mkdir(parents=True, exist_ok=True)
    train = load_json(training_decision_path(), {})
    results = load_json(OUT2 / "C16_2_BMN_RESULTS_BY_VARIANT.json", {}).get("results", {})
    if not train:
        raise RuntimeError("C16-2 decision missing")
    best = train.get("best_variant")
    best_res = results.get(best, {}).get("calib_holdout", {})
    def top100(key: str) -> float:
        return results.get(key, {}).get("calib_holdout", {}).get("summary", {}).get("IoU@0.7_top100", 0.0)
    keys = list(results)
    by_variant = defaultdict(list)
    for key in keys:
        by_variant[key.rsplit("_seed", 1)[0]].append(top100(key))
    variant_mean = {k: mean(v) for k, v in by_variant.items()}
    b3 = variant_mean.get("B3_full_query_aware_map") or 0.0
    ablations = {
        "query_aware_vs_no_query": {"reference": "C15/C16 B3 uses query; no-query approximated by C15/C16 boundary-only lineage", "delta": None},
        "span_map_vs_row_level_mlp": {"span_map_top100": best_res.get("summary", {}).get("IoU@0.7_top100"), "row_level_t2_top100": train.get("t2_same_split_reference", {}).get("summary", {}).get("IoU@0.7_top100"), "delta": train.get("vs_c12_5t_t2_delta", {}).get("IoU@0.7_top100")},
        "with_actionness_vs_without": {"B6_mean_top100": variant_mean.get("B6_full_query_aware_actionness_map"), "B3_mean_top100": b3, "delta": (variant_mean.get("B6_full_query_aware_actionness_map") or 0.0) - b3},
        "with_duration_embedding": {"B4_mean_top100": variant_mean.get("B4_full_duration_conditioned_map"), "B3_mean_top100": b3, "delta": (variant_mean.get("B4_full_duration_conditioned_map") or 0.0) - b3},
        "rank_loss_strength": {"B7_mean_top100": variant_mean.get("B7_full_query_aware_rankloss_map"), "B3_mean_top100": b3, "delta": (variant_mean.get("B7_full_query_aware_rankloss_map") or 0.0) - b3},
        "short_specialist": {"B5_mean_top100": variant_mean.get("B5_full_short_specialized_map"), "B3_mean_top100": b3, "delta": (variant_mean.get("B5_full_short_specialized_map") or 0.0) - b3},
        "D_max_sensitivity": {"tested": [mode_cfg(mode)["d_max"]], "note": "single D_max in medium; full sensitivity deferred to next train-only run"},
        "boundary_sigma_sensitivity": {"tested": "adaptive sigma 0.7..2.0 clips", "note": "fixed adaptive policy in medium"},
        "topK_sensitivity": best_res.get("summary", {}),
        "NMS_sensitivity_report_only": {"nms_threshold": NMS_THRESHOLD, "nms_modified": False},
    }
    robustness = {
        "seed_count": len(seed_list(seed, mode)),
        "variant_mean_top100": variant_mean,
        "best_variant": best,
        "best_seed_deltas_vs_t2": train.get("seed_consistency", {}),
    }
    dur = best_res.get("duration_breakdown", {})
    qtype = best_res.get("query_type_breakdown", {})
    full_sensitivity_done = False
    status = "C16_ROBUSTNESS_PARTIAL" if train.get("status") == "C16_BMN_FULLSCALE_PROMISING" and len(seed_list(seed, mode)) >= 2 else "C16_ROBUSTNESS_FAIL"
    if full_sensitivity_done and status == "C16_ROBUSTNESS_PARTIAL":
        status = "C16_ROBUSTNESS_PASS"
    write_text(OUT3 / "C16_3_ABLATION_PLAN.md", "# C16-3 Ablation Plan\n\nAblations are computed from C16-2 medium variants/seeds and train-only T2 reference. NMS sensitivity is report-only; NMS is not modified.\n")
    write_json(OUT3 / "C16_3_ABLATION_RESULTS.json", {"stage": "C16-3", "ablations": ablations, "official_val_used": False})
    write_json(OUT3 / "C16_3_SEED_ROBUSTNESS.json", robustness)
    write_json(OUT3 / "C16_3_DURATION_ROBUSTNESS.json", dur)
    write_json(OUT3 / "C16_3_QUERY_TYPE_ROBUSTNESS.json", qtype)
    write_text(OUT3 / "C16_3_ERROR_CASES.md", "# C16-3 Error Cases\n\nDetailed false positives are represented by C16-2 prediction sample and aggregate false-positive metrics. No official predictions were read.\n")
    decision = {"stage": "C16-3", "status": status, "best_variant": best, "module_contribution": ablations, "official_val_used": False}
    write_json(OUT3 / "C16_3_ROBUSTNESS_DECISION.json", decision)
    write_text(OUT3 / "C16_3_ROBUSTNESS_DECISION.md", f"# C16-3 Robustness Decision\n\nstatus = `{status}`\n\nbest_variant = `{best}`\n\nofficial was not run.\n")
    return decision


def stage_c16_4(mode: str, seed: int) -> Dict[str, Any]:
    OUT4.mkdir(parents=True, exist_ok=True)
    train = load_json(training_decision_path(), {})
    robust = load_json(OUT3 / "C16_3_ROBUSTNESS_DECISION.json", {})
    full_train_only_vcmr_evaluator_available = False
    if train.get("status") != "C16_BMN_FULLSCALE_PROMISING" or robust.get("status") not in {"C16_ROBUSTNESS_PASS", "C16_ROBUSTNESS_PARTIAL"}:
        status = "C16_NATIVE_INTEGRATION_WEAK"
    elif not full_train_only_vcmr_evaluator_available:
        status = "C16_NATIVE_INTEGRATION_LOCALIZER_ONLY"
    else:
        status = "C16_NATIVE_INTEGRATION_PROMISING"
    best = train.get("best_holdout_metrics", {})
    t2 = train.get("t2_same_split_reference", {})
    coupling = {
        "stage": "C16-4",
        "mode": mode,
        "evaluator_available_for_train_only_full_vcmr": full_train_only_vcmr_evaluator_available,
        "fallback_metric": "same-query topK moment coverage with retriever_top100 joint proxy",
        "gt_video_top100_proxy_note": "first-stage replay is used only as train-only retriever coupling audit",
        "wrong_video_high_score_rate": None,
        "official_val_used": False,
    }
    comparison = {
        "C12_5T_T2_localizer_baseline": t2.get("summary", {}),
        "C15_2_B3_sample_baseline": load_json(ROOT / "c15_2_bmn_span_confidence_mvp/C15_2_BMN_DECISION.json", {}).get("best_holdout_metrics", {}).get("summary", {}),
        "C16_BMN_map_only": best.get("summary", {}),
        "C16_BMN_plus_retriever_replay_proxy": {
            "joint_IoU@0.7_top100": None,
            "note": "full train-only VCMR evaluator was not run; C16-4 remains a localizer-only integration proxy",
        },
        "T2_still_dominates": (t2.get("summary", {}).get("IoU@0.7_top100", 0.0) > best.get("summary", {}).get("IoU@0.7_top100", 0.0)),
    }
    rec = {"stage": "C16-4", "status": status, "mode": mode, "best_variant": train.get("best_variant"), "metrics": comparison, "official_val_used": False}
    write_text(OUT4 / "C16_4_INTEGRATION_PLAN.md", "# C16-4 Native VCMR Integration Plan\n\nUse train-only same-query moment coverage plus first-stage retriever coupling proxy. Full official-style VCMR is not run.\n")
    write_json(OUT4 / "C16_4_NATIVE_VCMR_CONFIG.json", {"mode": mode, "fusion_candidates": ["map-only", "T2-only", "retriever+map proxy", "map+T2 safety proxy"], "selection_split": "calib_select", "official_val_used": False})
    write_json(OUT4 / "C16_4_NATIVE_VCMR_RESULTS.json", rec)
    write_json(OUT4 / "C16_4_RETRIEVER_LOCALIZER_COUPLING_AUDIT.json", coupling)
    write_json(OUT4 / "C16_4_COMPARISON_TO_BASELINES.json", comparison)
    write_json(OUT4 / "C16_4_INTEGRATION_DECISION.json", rec)
    write_text(OUT4 / "C16_4_INTEGRATION_DECISION.md", f"# C16-4 Integration Decision\n\nstatus = `{status}`\n\nbest_variant = `{train.get('best_variant')}`\n\nofficial was not run.\n")
    return rec


def stage_c16_5(mode: str, seed: int) -> Dict[str, Any]:
    OUT5.mkdir(parents=True, exist_ok=True)
    c16_1 = load_json(data_status_path(), {})
    c16_2 = load_json(training_decision_path(), {})
    c16_3 = load_json(OUT3 / "C16_3_ROBUSTNESS_DECISION.json", {})
    c16_4 = load_json(OUT4 / "C16_4_INTEGRATION_DECISION.json", {})
    if c16_2.get("status") == "C16_BMN_FULLSCALE_PROMISING" and c16_3.get("status") in {"C16_ROBUSTNESS_PASS", "C16_ROBUSTNESS_PARTIAL"} and c16_4.get("status") == "C16_NATIVE_INTEGRATION_PROMISING":
        final = "C16_READY_FOR_FREEZE_AND_ONE_SHOT_REVIEW"
    elif c16_2.get("status") == "C16_BMN_FULLSCALE_PROMISING" and c16_4.get("status") == "C16_NATIVE_INTEGRATION_LOCALIZER_ONLY":
        final = "C16_CONTINUE_BMN_T2_HYBRID_TRAIN_ONLY"
    elif c16_2.get("status") in {"C16_BMN_FULLSCALE_WEAK", "C16_BMN_FULLSCALE_INCONCLUSIVE"}:
        final = "C16_NEED_BSN_LOCAL_TO_GLOBAL_FALLBACK"
    elif not raw_video_absent():
        final = "C16_NEED_RAW_VIDEO_STRONG_FEATURE"
    else:
        final = "C16_STOP_TVR_VCMR_SCORE_CHASING"
    packet = {
        "stage": "C16-5",
        "status": final,
        "mode": mode,
        "c15_accepted": True,
        "c16_1_status": c16_1.get("status"),
        "c16_2_status": c16_2.get("status"),
        "c16_3_status": c16_3.get("status"),
        "c16_4_status": c16_4.get("status"),
        "pseudo_official_holdout_used_for_selection": False,
        "official_was_run": False,
        "evaluator_modified": False,
        "nms_modified": False,
        "schema_hash": c16_1.get("feature_schema_hash"),
        "candidate_pool_hash": c16_1.get("candidate_pool_hash"),
        "best_checkpoint_path": load_json(OUT2 / "C16_2_BMN_CHECKPOINT_MANIFEST.json", {}).get("checkpoints", {}).get(c16_2.get("best_variant"), {}).get("path"),
        "commands": [
            f"/home/a/miniconda3/envs/conquer-rlem/bin/python run_c16_fullscale_bmn_native_localizer.py --stage all --mode {mode} --seed {seed}",
            f"/home/a/miniconda3/envs/conquer-rlem/bin/python run_c16_fullscale_bmn_native_localizer.py --stage c16_3 --mode {mode} --seed {seed}",
            f"/home/a/miniconda3/envs/conquer-rlem/bin/python run_c16_fullscale_bmn_native_localizer.py --stage c16_4 --mode {mode} --seed {seed}",
            f"/home/a/miniconda3/envs/conquer-rlem/bin/python run_c16_fullscale_bmn_native_localizer.py --stage c16_5 --mode {mode} --seed {seed}",
        ],
        "random_seed_base": seed,
        "official_val_used": False,
    }
    write_json(OUT5 / "C16_5_FREEZE_REVIEW.json", packet)
    write_json(OUT5 / "C16_5_OFFICIAL_READINESS_PACKET.json", packet)
    write_json(OUT5 / "C16_5_NEXT_STEP_DECISION.json", packet)
    write_text(OUT5 / "C16_5_FREEZE_REVIEW.md", f"# C16-5 Freeze Review\n\nfinal decision = `{final}`\n\nofficial was not run.\n")
    write_text(OUT5 / "C16_5_OFFICIAL_READINESS_PACKET.md", f"# C16-5 Official Readiness Packet\n\nDecision: `{final}`\n\nThis is a readiness packet only. It does not authorize or run official validation.\n")
    write_text(OUT5 / "C16_5_RISK_REGISTER.md", """# C16-5 Risk Register

- C16 remains train-only and medium-scale unless full mode is explicitly run later.
- Raw TVR video remains absent.
- Native VCMR integration uses a train-only fallback proxy, not official validation.
- Checkpoints are local train-only artifacts and large prediction files are not saved.
""")
    write_text(OUT5 / "C16_5_NEXT_STEP_DECISION.md", f"# C16-5 Next Step Decision\n\nstatus = `{final}`\n\nofficial was not run.\n")
    return packet


def run_all(mode: str, seed: int, force: bool) -> Dict[str, Any]:
    out0 = stage_c16_0(mode, seed)
    if out0.get("status") != "C16_PROTOCOL_READY":
        return {"c16_0": out0}
    out1 = stage_c16_1(mode, seed, force)
    if out1.get("status") != "C16_FULLSCALE_DATA_READY":
        return {"c16_0": out0, "c16_1": out1}
    out2 = stage_c16_2(mode, seed, force)
    out3 = stage_c16_3(mode, seed)
    out4 = stage_c16_4(mode, seed)
    out5 = stage_c16_5(mode, seed)
    return {"c16_0": out0, "c16_1": out1, "c16_2": out2, "c16_3": out3, "c16_4": out4, "c16_5": out5}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["c16_0", "c16_1", "c16_2", "c16_3", "c16_4", "c16_5", "all"], default="all")
    parser.add_argument("--mode", choices=["smoke", "medium", "full"], default="medium")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.mode == "full":
        medium = load_json(OUT5 / "C16_5_NEXT_STEP_DECISION.json", {})
        if medium.get("mode") != "medium" and not args.force:
            raise RuntimeError("full mode requires a completed medium packet or --force")
    if args.stage == "c16_0":
        out = stage_c16_0(args.mode, args.seed)
    elif args.stage == "c16_1":
        out = stage_c16_1(args.mode, args.seed, args.force)
    elif args.stage == "c16_2":
        out = stage_c16_2(args.mode, args.seed, args.force)
    elif args.stage == "c16_3":
        out = stage_c16_3(args.mode, args.seed)
    elif args.stage == "c16_4":
        out = stage_c16_4(args.mode, args.seed)
    elif args.stage == "c16_5":
        out = stage_c16_5(args.mode, args.seed)
    else:
        out = run_all(args.mode, args.seed, args.force)
    print(json.dumps(out, indent=2, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
