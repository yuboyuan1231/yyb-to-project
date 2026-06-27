#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rlem_c6_b0.run_c6_b0_candidate_diagnostic import correctness_for_ts, load_gt_ts, span_iou_idx  # noqa: E402
from rlem_c6_b1_lite.run_c6_b1_r1_safe_selector import evaluate_policy  # noqa: E402
from rlem_c7.run_c7_b0_b1 import (  # noqa: E402
    EPS,
    atomic_json,
    atomic_npz,
    atomic_text,
    artifact,
    features_for_spans,
    gt_span_by_query,
    gt_video_by_query,
    labels_for_sequence,
    load_baseline_configs,
    load_npz,
    make_fast_anchor_data,
    make_anchor_candidates_fast,
    metric_delta,
    nms_sequence,
    prepare_side_arrays,
    selected_metrics,
)
from rlem_c7.freeze_c7_b1 import CachedProposalScorer  # noqa: E402


METRIC_KEYS = [
    "0.5-r1", "0.5-r5", "0.5-r10", "0.5-r100",
    "0.7-r1", "0.7-r5", "0.7-r10", "0.7-r100",
]


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--train_fit_cache", default="results/rlem_c6a/cache/train_fit_c6_cache.npz")
    p.add_argument("--calib_cache", default="results/rlem_c6a/cache/train_calib_c6_cache.npz")
    p.add_argument("--train_fit_temporal", default="results/rlem_c5_lite_prior_prep/train_fit_temporal_prior.npz")
    p.add_argument("--calib_temporal", default="results/rlem_c5_lite_prior_prep/train_calib_temporal_prior.npz")
    p.add_argument("--train_fit_boundary", default="results/rlem_c6_b1_lite_r1_safe/train_fit_boundary_oracle_prior.npz")
    p.add_argument("--calib_boundary", default="results/rlem_c6_b1_lite_r1_safe/train_calib_boundary_oracle_prior.npz")
    p.add_argument("--train_fit_pool", default="results/rlem_c6_b1_lite_r1_safe/train_fit_candidate_pool_top_slots.npz")
    p.add_argument("--calib_pool", default="results/rlem_c6_b1_lite_r1_safe/train_calib_candidate_pool_top_slots.npz")
    p.add_argument("--train_fit_b1_scores", default="results/rlem_c6_c/aux/train_fit_b1_candidate_scores.npz")
    p.add_argument("--train_fit_b2_scores", default="results/rlem_c6_c/aux/train_fit_b2_pairwise_main_scores.npz")
    p.add_argument("--b1_calib_scores", default="results/rlem_c6_b1_lite_r1_safe/train_calib_candidate_scores.npz")
    p.add_argument("--b2_calib_scores", default="results/rlem_c6_b2/train_calib_scores/train_calib_pairwise_main_scores.npz")
    p.add_argument("--b1_best_config", default="results/rlem_c6_b1_lite_r1_safe/best_config.json")
    p.add_argument("--b2_best_config", default="results/rlem_c6_b2/train_calib_scores/pairwise_main/best_config.json")
    p.add_argument("--train_fit_c6c_scores", default="results/rlem_c6_c/train_fit_video_scores.npz")
    p.add_argument("--calib_c6c_scores", default="results/rlem_c6_c/train_calib_video_scores.npz")
    p.add_argument("--gt_jsonl", default="/home/a/yybwork/data/yyb/tvr_feature_release/data/tvr_train_release.jsonl")
    p.add_argument("--c7_b1_official_manifest", default="c7_audit/C7_B1_OFFICIAL_MANIFEST.json")
    p.add_argument("--c7_b1_freeze_package", default="c7_audit/C7_B1_FINAL_FREEZE_PACKAGE.json")
    p.add_argument("--output_dir", default="results/rlem_c7_b2")
    p.add_argument("--audit_dir", default="c7_audit")
    p.add_argument("--models_dir", default="c7_models")
    p.add_argument("--device", default="cuda")
    p.add_argument("--score_batch_size", type=int, default=65536)
    p.add_argument("--feature_chunk_size", type=int, default=200000)
    p.add_argument("--effective_top_n", type=int, default=100)
    p.add_argument("--max_after_nms", type=int, default=100)
    p.add_argument("--nms_thd", type=float, default=0.7)
    p.add_argument("--hidden", type=int, default=384)
    p.add_argument("--layers", type=int, default=3)
    p.add_argument("--dropout", type=float, default=0.10)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=65536)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--search_profile", choices=["lite", "full"], default="lite")
    return p.parse_args()


def flatten_c7_b1_metrics(path: str | Path) -> Dict[str, float]:
    obj = json.loads(Path(path).read_text(encoding="utf-8"))
    selected = obj["selected_deterministic_rerun"]["metrics"]
    return {k: float(selected[k]) for k in METRIC_KEYS}


def archive_c7_b1(args: argparse.Namespace) -> Dict[str, Any]:
    manifest = json.loads(Path(args.c7_b1_official_manifest).read_text(encoding="utf-8"))
    if manifest.get("official_val_used") is not True or manifest.get("post_val_adjustment") is not False:
        raise ValueError("C7-B1 official manifest is not a completed no-adjustment one-shot")
    archive = {
        "status": "C7_B1_OFFICIAL_ARCHIVED",
        "source_status": manifest.get("status"),
        "promoted": False,
        "human_summary": (
            "official core-positive integrated proposal-confidence branch; "
            "not promoted only due strict pre-registered 0.5-r10 nonnegative rule."
        ),
        "official_val_used": True,
        "post_val_adjustment": False,
        "second_official_val": False,
        "mu_eta_adjusted": False,
        "C7_B1_retrained": False,
        "evaluator_modified": False,
        "nms_modified": False,
        "C6_C4_artifacts_modified": False,
        "next_stage": "C7-B2 train_fit/train_calib only",
        "source_manifest": artifact(args.c7_b1_official_manifest),
    }
    audit = Path(args.audit_dir)
    atomic_json(audit / "C7_B1_OFFICIAL_ARCHIVE.json", archive)
    atomic_text(
        audit / "C7_B1_OFFICIAL_ARCHIVE.md",
        "# C7-B1 official archive\n\n"
        f"- Status: `{archive['status']}`\n"
        f"- Source status: `{archive['source_status']}`\n"
        f"- Human summary: {archive['human_summary']}\n"
        "- post_val_adjustment / second_official_val: `false / false`\n"
        "- This archive does not alter the C7-B1 decision.\n",
    )
    return archive


@torch.no_grad()
def score_proposal_dataset(args: argparse.Namespace, split: str) -> Dict[str, Any]:
    src = Path("results/rlem_c7") / (f"{split}_proposal_dataset_top50.npz" if split == "train_fit" else "train_calib_proposal_dataset_top100.npz")
    out = Path(args.output_dir) / f"{split}_c7_b1_proposal_scores.npz"
    if out.exists():
        return {"split": split, "reused": True, "scores": artifact(out)}
    z = np.load(src, allow_pickle=True)
    scorer = CachedProposalScorer(args)
    scores = scorer.score(z["x"].astype(np.float32))
    atomic_npz(
        out,
        proposal_confidence=scores["proposal_confidence"].astype(np.float32),
        span_quality=scores["span_quality"].astype(np.float32),
        y05=z["y05"].astype(np.float32),
        y07=z["y07"].astype(np.float32),
        iou=z["iou"].astype(np.float32),
        query=z["query"].astype(np.int32),
        row=z["row"].astype(np.int64),
        group_id=z["group_id"].astype(np.int32),
        rank=z["rank"].astype(np.int16),
    )
    return {
        "split": split,
        "reused": False,
        "scores": artifact(out),
        "examples": int(len(z["row"])),
        "proposal_confidence_mean": float(scores["proposal_confidence"].mean()),
    }


def agg_by_group(values: np.ndarray, gids: np.ndarray, n: int, op: str, default: float = 0.0) -> np.ndarray:
    out = np.full(n, default, dtype=np.float32)
    if op == "max":
        out[:] = -1e9
        np.maximum.at(out, gids, values.astype(np.float32))
        out[out < -1e8] = default
        return out
    if op == "min":
        out[:] = 1e9
        np.minimum.at(out, gids, values.astype(np.float32))
        out[out > 1e8] = default
        return out
    if op == "mean":
        sums = np.bincount(gids, weights=values.astype(np.float64), minlength=n).astype(np.float32)
        cnt = np.bincount(gids, minlength=n).astype(np.float32)
        return sums / np.maximum(cnt, 1.0)
    raise ValueError(op)


def build_video_dataset(args: argparse.Namespace, split: str) -> Dict[str, Any]:
    out = Path(args.output_dir) / f"{split}_video_dataset.npz"
    if out.exists():
        return {"split": split, "reused": True, "dataset": artifact(out)}
    cache = load_npz(args.train_fit_cache if split == "train_fit" else args.calib_cache, allow_pickle=True)
    prop_data_path = Path("results/rlem_c7") / (f"{split}_proposal_dataset_top50.npz" if split == "train_fit" else "train_calib_proposal_dataset_top100.npz")
    prop_score_path = Path(args.output_dir) / f"{split}_c7_b1_proposal_scores.npz"
    c6c_path = args.train_fit_c6c_scores if split == "train_fit" else args.calib_c6c_scores
    prop_data = np.load(prop_data_path, allow_pickle=True)
    prop = np.load(prop_score_path, allow_pickle=False)
    c6c = np.load(c6c_path, allow_pickle=False)
    n = int(cache["video_group_keys"].shape[0])
    gids_prop = prop["group_id"].astype(np.int64)
    cnt = np.bincount(gids_prop, minlength=n).astype(np.float32)
    rank_inv = 1.0 / (prop["rank"].astype(np.float32) + 1.0)
    feat_cols = [
        agg_by_group(prop["proposal_confidence"], gids_prop, n, "max"),
        agg_by_group(prop["proposal_confidence"], gids_prop, n, "mean"),
        agg_by_group(prop["span_quality"], gids_prop, n, "max"),
        agg_by_group(prop["span_quality"], gids_prop, n, "mean"),
        agg_by_group(prop["y05"], gids_prop, n, "max"),
        agg_by_group(prop["y07"], gids_prop, n, "max"),
        agg_by_group(prop["iou"], gids_prop, n, "max"),
        agg_by_group(rank_inv, gids_prop, n, "max"),
        np.minimum(cnt / 100.0, 1.0).astype(np.float32),
    ]
    c6c_idx = np.full(n, -1, dtype=np.int64)
    c6c_idx[c6c["group_id"].astype(np.int64)] = np.arange(len(c6c["group_id"]), dtype=np.int64)
    valid = c6c_idx >= 0
    for key in ["relevance", "moment", "iou", "risk", "gate"]:
        arr = np.zeros(n, dtype=np.float32)
        arr[valid] = c6c[key].astype(np.float32)[c6c_idx[valid]]
        feat_cols.append(arr)
    base_rank = np.full(n, 100.0, dtype=np.float32)
    base_rank[valid] = c6c["base_rank"].astype(np.float32)[c6c_idx[valid]]
    feat_cols.extend([
        1.0 / (base_rank + 1.0),
        np.clip(base_rank / 100.0, 0.0, 1.0).astype(np.float32),
    ])
    x = np.concatenate([np.stack(feat_cols, axis=1), cache["video_features"].astype(np.float32)], axis=1)
    atomic_npz(
        out,
        x=x.astype(np.float32),
        group_id=np.arange(n, dtype=np.int32),
        query=cache["video_group_keys"][:, 0].astype(np.int32),
        video_idx=cache["video_group_keys"][:, 1].astype(np.int32),
        base_rank=base_rank.astype(np.float32),
        y_video=cache["label_relevant"].astype(np.float32),
        y_any05=cache["label_any_05"].astype(np.float32),
        y_any07=cache["label_any_07"].astype(np.float32),
        y_best_iou=cache["label_best_iou"].astype(np.float32),
        feature_names=np.asarray([
            "c7_prop_conf_max", "c7_prop_conf_mean", "c7_span_quality_max", "c7_span_quality_mean",
            "c7_span_hit05_max", "c7_span_hit07_max", "c7_span_iou_max", "c7_rank_inv_max",
            "c7_prop_count_norm", "c6c_relevance", "c6c_moment", "c6c_iou", "c6c_risk", "c6c_gate",
            "base_rank_inv", "base_rank_norm",
        ] + [f"video_feature_{i}" for i in range(cache["video_features"].shape[1])], dtype=object),
    )
    return {
        "split": split,
        "reused": False,
        "dataset": artifact(out),
        "groups": int(n),
        "feature_dim": int(x.shape[1]),
        "positive_video_rate": float(cache["label_relevant"].mean()),
        "partial_any07_rate": float(cache["label_any_07"].mean()),
    }


class C7B2VideoResidualHead(nn.Module):
    def __init__(self, in_dim: int, hidden: int, layers: int, dropout: float):
        super().__init__()
        mods: List[nn.Module] = []
        d = in_dim
        for _ in range(layers):
            mods += [nn.Linear(d, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(dropout)]
            d = hidden
        self.backbone = nn.Sequential(*mods)
        self.relevance = nn.Linear(hidden, 1)
        self.moment = nn.Linear(hidden, 1)
        self.iou = nn.Linear(hidden, 1)
        self.risk = nn.Linear(hidden, 1)
        self.gate = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        h = self.backbone(x)
        return {
            "relevance": self.relevance(h).squeeze(-1),
            "moment": self.moment(h).squeeze(-1),
            "iou": self.iou(h).squeeze(-1),
            "risk": self.risk(h).squeeze(-1),
            "gate": self.gate(h).squeeze(-1),
        }


def train_head(args: argparse.Namespace) -> Dict[str, Any]:
    model_path = Path(args.output_dir) / "c7_b2_mil_video_residual_head.pt"
    hist_path = Path(args.output_dir) / "c7_b2_training_history.json"
    if model_path.exists() and hist_path.exists():
        return {"reused": True, "model": artifact(model_path), "history": json.loads(hist_path.read_text(encoding="utf-8"))}
    z = np.load(Path(args.output_dir) / "train_fit_video_dataset.npz", allow_pickle=True)
    x = z["x"].astype(np.float32)
    yv = z["y_video"].astype(np.float32)
    y05 = z["y_any05"].astype(np.float32)
    y07 = z["y_any07"].astype(np.float32)
    yiou = z["y_best_iou"].astype(np.float32)
    base_rank = z["base_rank"].astype(np.float32)
    risk = ((1.0 - yv) * (base_rank < 5).astype(np.float32)).astype(np.float32)
    mean = x.mean(axis=0).astype(np.float32)
    std = np.maximum(x.std(axis=0).astype(np.float32), EPS)
    ds = TensorDataset(
        torch.from_numpy(((x - mean) / std).astype(np.float32)),
        torch.from_numpy(np.stack([yv, y05, y07, yiou, risk], axis=1).astype(np.float32)),
    )
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    model = C7B2VideoResidualHead(x.shape[1], args.hidden, args.layers, args.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))
    pos_rel = torch.tensor([(1.0 - yv.mean()) / max(float(yv.mean()), EPS)], dtype=torch.float32, device=device).clamp(max=80)
    pos_mom = torch.tensor([(1.0 - y07.mean()) / max(float(y07.mean()), EPS)], dtype=torch.float32, device=device).clamp(max=120)
    hist: List[Dict[str, Any]] = []
    for epoch in range(1, args.epochs + 1):
        losses: List[float] = []
        t0 = time.time()
        model.train()
        for xb, yb in dl:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                out = model(xb)
                rel_loss = F.binary_cross_entropy_with_logits(out["relevance"], yb[:, 0], pos_weight=pos_rel)
                mom07_loss = F.binary_cross_entropy_with_logits(out["moment"], yb[:, 2], pos_weight=pos_mom)
                mom05_loss = F.binary_cross_entropy_with_logits(out["iou"], yb[:, 1])
                iou_loss = F.smooth_l1_loss(torch.sigmoid(out["iou"]), yb[:, 3])
                risk_loss = F.binary_cross_entropy_with_logits(out["risk"], yb[:, 4])
                gate = torch.sigmoid(out["gate"])
                gate_reg = torch.relu(gate.mean() - 0.35) + 0.05 * gate.var()
                loss = rel_loss + 0.8 * mom07_loss + 0.35 * mom05_loss + 0.4 * iou_loss + 0.45 * risk_loss + 0.25 * gate_reg
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            losses.append(float(loss.detach().cpu()))
        rec = {"epoch": epoch, "loss": float(np.mean(losses)), "elapsed_sec": float(time.time() - t0), "device": str(device)}
        hist.append(rec)
        atomic_json(hist_path, hist)
        print(json.dumps({"stage": "c7_b2_train", **rec}), flush=True)
    torch.save({
        "model_state": model.state_dict(),
        "mean": mean,
        "std": std,
        "in_dim": int(x.shape[1]),
        "hidden": int(args.hidden),
        "layers": int(args.layers),
        "dropout": float(args.dropout),
        "feature_names": z["feature_names"],
    }, model_path)
    return {"reused": False, "model": artifact(model_path), "history": hist}


@torch.no_grad()
def score_video_dataset(args: argparse.Namespace, split: str) -> Dict[str, Any]:
    out_path = Path(args.output_dir) / f"{split}_video_scores.npz"
    if out_path.exists():
        return {"split": split, "reused": True, "scores": artifact(out_path)}
    z = np.load(Path(args.output_dir) / f"{split}_video_dataset.npz", allow_pickle=True)
    ckpt = torch.load(Path(args.output_dir) / "c7_b2_mil_video_residual_head.pt", map_location="cpu")
    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    model = C7B2VideoResidualHead(int(ckpt["in_dim"]), int(ckpt["hidden"]), int(ckpt["layers"]), float(ckpt["dropout"])).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    x = (z["x"].astype(np.float32) - ckpt["mean"]) / np.maximum(ckpt["std"], EPS)
    outs = {k: np.empty(len(x), dtype=np.float32) for k in ["relevance", "moment", "iou", "risk", "gate"]}
    for start in range(0, len(x), args.score_batch_size):
        xb = torch.from_numpy(x[start:start + args.score_batch_size]).to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
            pred = model(xb)
        n = len(xb)
        outs["relevance"][start:start+n] = pred["relevance"].float().cpu().numpy()
        outs["moment"][start:start+n] = pred["moment"].float().cpu().numpy()
        outs["iou"][start:start+n] = pred["iou"].float().cpu().numpy()
        outs["risk"][start:start+n] = pred["risk"].float().cpu().numpy()
        outs["gate"][start:start+n] = torch.sigmoid(pred["gate"]).float().cpu().numpy()
    atomic_npz(out_path, **outs, group_id=z["group_id"], query=z["query"], video_idx=z["video_idx"], base_rank=z["base_rank"])
    return {"split": split, "reused": False, "scores": artifact(out_path), "gate_mean": float(outs["gate"].mean())}


def build_c7b1_anchor_states(args: argparse.Namespace, cache: Dict[str, np.ndarray], pool: Dict[str, np.ndarray], b2_scores: np.ndarray, b2_cfg: Dict[str, Any]) -> Dict[str, Any]:
    out_path = Path(args.output_dir) / "train_calib_c7_b1_anchor_states.json"
    if out_path.exists():
        obj = json.loads(out_path.read_text(encoding="utf-8"))
        if obj.get("states") and "flat_candidates" not in obj["states"][0]:
            out_path.unlink()
        else:
            if "metrics" not in obj:
                gt_vid = gt_video_by_query(cache)
                gt_s, gt_e = gt_span_by_query(cache)
                labels05: List[List[bool]] = []
                labels07: List[List[bool]] = []
                for q, state in enumerate(obj["states"]):
                    flat = [tuple(x) for x in state["flat_candidates"]]
                    seq = nms_sequence(flat[:args.effective_top_n], args.nms_thd, args.max_after_nms)
                    l05, l07, _pos, _inv, _dup = labels_for_sequence(seq, int(gt_vid[q]), int(gt_s[q]), int(gt_e[q]))
                    labels05.append(l05)
                    labels07.append(l07)
                obj["metrics"] = selected_metrics(labels05, labels07)
                atomic_json(out_path, obj)
            return {"states": obj["states"], "metrics": obj["metrics"], "artifact": artifact(out_path), "reused": True}
    score_npz = np.load("results/rlem_c7/train_calib_proposal_scores.npz", allow_pickle=False)
    prop_mean = float(score_npz["proposal_confidence"].mean())
    prop_std = float(max(score_npz["proposal_confidence"].std(), EPS))
    qual_mean = float(score_npz["span_quality"].mean())
    qual_std = float(max(score_npz["span_quality"].std(), EPS))
    data = make_fast_anchor_data(cache, pool, b2_scores)
    side = prepare_side_arrays("train_calib", args, cache)
    scorer = CachedProposalScorer(args)
    gt_vid = gt_video_by_query(cache)
    gt_s, gt_e = gt_span_by_query(cache)
    anchor_states: List[Dict[str, Any]] = []
    labels05: List[List[bool]] = []
    labels07: List[List[bool]] = []
    for q in range(len(cache["desc_ids"])):
        base = make_anchor_candidates_fast(q=q, data=data, policy_cfg=b2_cfg, effective_top_n=args.effective_top_n)
        gids = np.asarray([c[0] for c in base], dtype=np.int64)
        starts = np.asarray([c[2] for c in base], dtype=np.int64)
        ends = np.asarray([c[3] for c in base], dtype=np.int64)
        scores = np.asarray([c[4] for c in base], dtype=np.float32)
        rows = np.asarray([max(c[5], 0) for c in base], dtype=np.int64)
        ranks = np.arange(len(base), dtype=np.int16)
        x = features_for_spans(cache, side, rows, gids, starts, ends, scores, ranks, chunk_size=args.feature_chunk_size)
        pred = scorer.score(x)
        prop_z = (pred["proposal_confidence"] - prop_mean) / max(prop_std, EPS)
        qual_z = (pred["span_quality"] - qual_mean) / max(qual_std, EPS)
        new_scores = scores + 0.1 * prop_z.astype(np.float32) + 0.1 * qual_z.astype(np.float32)
        rescored = [(c[0], c[1], c[2], c[3], float(ns), c[5]) for c, ns in zip(base, new_scores)]
        rescored = sorted(rescored, key=lambda c: -c[4])
        seq = nms_sequence(rescored, args.nms_thd, args.max_after_nms)
        l05, l07, pos, _inv, _dup = labels_for_sequence(seq, int(gt_vid[q]), int(gt_s[q]), int(gt_e[q]))
        labels05.append(l05)
        labels07.append(l07)
        base_group_order: List[int] = []
        seen = set()
        group_candidates: Dict[str, List[List[float]]] = {}
        for cand in rescored:
            gid = int(cand[0])
            if gid not in seen:
                seen.add(gid)
                base_group_order.append(gid)
            group_candidates.setdefault(str(gid), []).append([int(cand[0]), int(cand[1]), int(cand[2]), int(cand[3]), float(cand[4]), int(cand[5])])
        anchor_states.append({
            "base_group_order": base_group_order,
            "group_candidates": group_candidates,
            "flat_candidates": [[int(c[0]), int(c[1]), int(c[2]), int(c[3]), float(c[4]), int(c[5])] for c in rescored],
            "base_pos": bool(pos),
        })
    metrics = selected_metrics(labels05, labels07)
    atomic_json(out_path, {"states": anchor_states, "metrics": metrics})
    return {"states": anchor_states, "metrics": metrics, "artifact": artifact(out_path), "reused": False}


def evaluate_video_residual(args: argparse.Namespace, score_arrays: Dict[str, np.ndarray], cfg: Dict[str, Any], anchor_states: List[Dict[str, Any]], gt_ts: Dict[int, Any], gt_vid: np.ndarray) -> Dict[str, Any]:
    gids = score_arrays["group_id"].astype(np.int64)
    base_rank = score_arrays["base_rank"].astype(np.float32)
    group_score: Dict[int, Tuple[float, float, float]] = {}
    for i, gid in enumerate(gids):
        gate = float(score_arrays["gate"][i])
        residual = (
            float(cfg["beta"]) * gate * float(score_arrays["relevance"][i])
            + float(cfg["gamma"]) * gate * (float(score_arrays["moment"][i]) + 0.35 * float(score_arrays["iou"][i]))
            - float(cfg["rho"]) * gate * float(score_arrays["risk"][i])
        )
        if gate < float(cfg["gate_threshold"]):
            residual = 0.0
        base_component = 1.0 / float(base_rank[i] + 1.0)
        group_score[int(gid)] = (float(cfg["alpha"]) * base_component + residual, gate, residual)
    labels05: List[List[bool]] = []
    labels07: List[List[bool]] = []
    video_hits = {1: 0, 5: 0, 10: 0, 100: 0}
    base_pos: List[bool] = []
    new_pos: List[bool] = []
    invalid = dup = changed = total = 0
    for q, state in enumerate(anchor_states):
        base_order = [int(x) for x in state["base_group_order"]]
        reranked = sorted(base_order, key=lambda gid: -group_score.get(gid, (0.0, 0.0, 0.0))[0])
        final_groups = list(base_order)
        moved_groups = set()
        changes = 0
        for pos, gid in enumerate(reranked):
            if pos >= len(final_groups) or changes >= int(cfg["max_video_slot_changes"]):
                break
            if final_groups[pos] == gid:
                continue
            if group_score.get(gid, (0.0, 0.0, 0.0))[1] < float(cfg["gate_threshold"]):
                continue
            old = final_groups.index(gid)
            final_groups.pop(old)
            final_groups.insert(pos, gid)
            moved_groups.add(gid)
            changes += 1
        changed += sum(int(a != b) for a, b in zip(base_order, final_groups))
        total += len(final_groups)
        tuple_rows = []
        for raw in state["flat_candidates"]:
            cand = tuple(raw)
            gid = int(cand[0])
            delta = group_score.get(gid, (0.0, 0.0, 0.0))[2] if gid in moved_groups else 0.0
            tuple_rows.append((int(cand[0]), int(cand[1]), int(cand[2]), int(cand[3]), float(cand[4]) + float(cfg["video_weight"]) * float(delta), int(cand[5])))
        tuple_rows = sorted(tuple_rows, key=lambda c: -c[4])
        kept = nms_sequence(tuple_rows[:args.effective_top_n], args.nms_thd, args.max_after_nms)
        invalid += sum(int(c[2] < 0 or c[3] < c[2]) for c in kept)
        dup += len(kept) - len({(int(c[1]), int(c[2]), int(c[3])) for c in kept})
        l05, l07 = [], []
        vids = [int(c[1]) for c in kept]
        for _gid, vid, si, ei, _score, _row in kept:
            same = int(vid) == int(gt_vid[q])
            l05.append(bool(same and correctness_for_ts(int(si), int(ei), gt_ts[q], 0.5)))
            l07.append(bool(same and correctness_for_ts(int(si), int(ei), gt_ts[q], 0.7)))
        for k in video_hits:
            video_hits[k] += int(int(gt_vid[q]) in vids[:k])
        labels05.append(l05)
        labels07.append(l07)
        base_pos.append(bool(state["base_pos"]))
        new_pos.append(bool(any(l05[:100]) or any(l07[:100])))
    exits = sum(int(a and not b) for a, b in zip(base_pos, new_pos))
    entries = sum(int((not a) and b) for a, b in zip(base_pos, new_pos))
    return {
        "config": cfg,
        "metrics": selected_metrics(labels05, labels07),
        "video_metrics": {f"GT_video_R@{k}": 100.0 * v / max(len(anchor_states), 1) for k, v in video_hits.items()},
        "movement": {
            "hard_positive_top100_query_exits": int(exits),
            "hard_positive_top100_query_entries": int(entries),
            "hard_positive_exit_ratio": float(exits / max(sum(base_pos), 1)),
            "video_slot_drift_rate": float(changed / max(total, 1)),
            "invalid_span_count": int(invalid),
            "duplicate_span_count_after_nms": int(dup),
        },
        "official_val_used": False,
    }


def run_search(args: argparse.Namespace) -> Dict[str, Any]:
    cache = load_npz(args.calib_cache, allow_pickle=True)
    pool = load_npz(args.calib_pool, allow_pickle=False)
    b1_cfg, b2_cfg = load_baseline_configs(args)
    b1_scores = np.load(args.b1_calib_scores, allow_pickle=False)["score"].astype(np.float32)
    b2_scores = np.load(args.b2_calib_scores, allow_pickle=False)["score"].astype(np.float32)
    gt_ts = load_gt_ts(args.gt_jsonl, cache["desc_ids"])
    gt_vid = gt_video_by_query(cache)
    b1_ev = evaluate_policy(cache=cache, pool=pool, pool_score=b1_scores, gt_ts_by_query=gt_ts, apply_slots=int(b1_cfg["apply_slots"]), threshold0=float(b1_cfg["threshold0"]), threshold_rest=float(b1_cfg["threshold_rest"]), max_replacements=int(b1_cfg["max_replacements"]), effective_top_n=100, max_after_nms=100, nms_thd=0.7)
    b2_ev = evaluate_policy(cache=cache, pool=pool, pool_score=b2_scores, gt_ts_by_query=gt_ts, apply_slots=int(b2_cfg["apply_slots"]), threshold0=float(b2_cfg["threshold0"]), threshold_rest=float(b2_cfg["threshold_rest"]), max_replacements=int(b2_cfg["max_replacements"]), effective_top_n=100, max_after_nms=100, nms_thd=0.7)
    c7b1_anchor = build_c7b1_anchor_states(args, cache, pool, b2_scores, b2_cfg)
    c7b1_freeze_metrics = flatten_c7_b1_metrics(args.c7_b1_freeze_package)
    score_arrays = load_npz(Path(args.output_dir) / "train_calib_video_scores.npz", allow_pickle=False)
    gate_values = score_arrays["gate"].astype(np.float32)
    gate_thresholds = sorted({float(np.quantile(gate_values, q)) for q in ([0.65, 0.80, 0.92, 0.97] if args.search_profile == "lite" else [0.50, 0.65, 0.80, 0.92, 0.97, 0.99])})
    configs = []
    alpha_grid = [1.0] if args.search_profile == "lite" else [0.7, 0.85, 1.0]
    beta_grid = [0.05, 0.10] if args.search_profile == "lite" else [0.03, 0.05, 0.10, 0.20]
    gamma_grid = [0.05, 0.10] if args.search_profile == "lite" else [0.03, 0.05, 0.10, 0.20]
    rho_grid = [0.05] if args.search_profile == "lite" else [0.05, 0.10]
    max_change_grid = [1] if args.search_profile == "lite" else [1, 2]
    for alpha in alpha_grid:
        for beta in beta_grid:
            for gamma in gamma_grid:
                for rho in rho_grid:
                    for gate in gate_thresholds:
                        for max_changes in max_change_grid:
                            configs.append({
                                "alpha": alpha,
                                "beta": beta,
                                "gamma": gamma,
                                "rho": rho,
                                "gate_threshold": gate,
                                "max_video_slot_changes": max_changes,
                                "lambda_span": 0.10,
                                "video_weight": 0.75,
                                "anchor": "C7_B1_frozen",
                            })
    results = []
    for i, cfg in enumerate(configs):
        rec = evaluate_video_residual(args, score_arrays, cfg, c7b1_anchor["states"], gt_ts, gt_vid)
        rec["config_id"] = f"c7_b2_mil_video_{i:04d}"
        rec["delta_vs_C6_B1"] = metric_delta(rec["metrics"], b1_ev["metrics"])
        rec["delta_vs_C6_B2"] = metric_delta(rec["metrics"], b2_ev["metrics"])
        rec["delta_vs_C7_B1_frozen"] = metric_delta(rec["metrics"], c7b1_freeze_metrics)
        d7 = rec["delta_vs_C7_B1_frozen"]
        mv = rec["movement"]
        rec["future_promotion_gate_pass"] = bool(
            d7["0.7-r1"] >= 0.05
            and d7["0.5-r1"] >= 0.0
            and d7["0.7-r5"] >= -0.20
            and d7["0.5-r5"] >= -0.20
            and mv["hard_positive_exit_ratio"] <= 0.01
            and mv["video_slot_drift_rate"] <= 0.20
            and mv["invalid_span_count"] == 0
            and mv["duplicate_span_count_after_nms"] == 0
        )
        rec["selection_score"] = float(
            6.0 * d7["0.7-r1"]
            + 2.0 * d7["0.5-r1"]
            - 2.0 * max(0.0, -d7["0.7-r5"] - 0.20)
            - 1.0 * max(0.0, -d7["0.5-r5"] - 0.20)
            - 5.0 * mv["hard_positive_exit_ratio"]
            - 0.5 * max(0.0, mv["video_slot_drift_rate"] - 0.20)
        )
        results.append(rec)
    feasible = [r for r in results if r["future_promotion_gate_pass"]]
    best = max(feasible or results, key=lambda r: (r["future_promotion_gate_pass"], r["selection_score"], r["delta_vs_C7_B1_frozen"]["0.7-r1"]))
    status = "C7_B2_TRAIN_CALIB_FUTURE_GATE_PASS" if feasible else "C7_B2_TRAIN_CALIB_NO_PROMOTION"
    payload = {
        "status": status,
        "official_val_used": False,
        "official_val_run": False,
        "post_val_adjustment": False,
        "second_official_val": False,
        "future_promotion_gate_not_applied_to_C7_B1": True,
        "baselines": {
            "C6_B1": b1_ev,
            "C6_B2": b2_ev,
            "C7_B1_frozen": {"metrics": c7b1_freeze_metrics, "anchor_reconstructed_metrics": c7b1_anchor["metrics"]},
        },
        "gate": {
            "priority": "R1 first; R5 allowed only small bounded decline",
            "criteria": "delta_vs_C7_B1_frozen 0.7-r1 >= 0.05, 0.5-r1 >= 0, R5 >= -0.20, exit_ratio <= 0.01, drift <= 0.20",
        },
        "best": best,
        "feasible_count": len(feasible),
        "grid_size": len(results),
        "gate_thresholds": gate_thresholds,
        "results": results,
        "c7b1_anchor": {"artifact": c7b1_anchor["artifact"], "metric_abs_diff_vs_freeze": {k: abs(c7b1_anchor["metrics"][k] - c7b1_freeze_metrics[k]) for k in METRIC_KEYS}},
    }
    atomic_json(Path(args.audit_dir) / "C7_B2_TRAIN_CALIB_EVAL.json", payload)
    atomic_json(Path(args.output_dir) / "train_calib_grid_results.json", results)
    atomic_json(Path(args.output_dir) / "best_config.json", best)
    md = "# C7-B2 train_calib eval\n\n"
    md += f"- Status: `{status}`\n- Official val used: `false`\n- Grid size: `{len(results)}`\n- Feasible count: `{len(feasible)}`\n"
    md += "- Future gate is not applied backward to C7-B1.\n\n"
    md += "## Best\n\n```json\n" + json.dumps(best, indent=2, ensure_ascii=False) + "\n```\n"
    atomic_text(Path(args.audit_dir) / "C7_B2_TRAIN_CALIB_EVAL.md", md)
    return payload


def main() -> None:
    args = parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.audit_dir).mkdir(parents=True, exist_ok=True)
    archive = archive_c7_b1(args)
    start = {
        "status": "C7_B2_STARTED_TRAIN_FIT_TRAIN_CALIB_ONLY",
        "official_val_used": False,
        "official_val_run": False,
        "C7_B1_archived_status": archive["source_status"],
        "C7_B1_human_summary": archive["human_summary"],
        "CONQUER_backbone_frozen": True,
        "QDF_QAL_original_ML_VR_frozen": True,
        "trainable_scope": "new Partial Relevance MIL + bounded Moment-aware Video Residual head only",
        "C7_B1_proposal_confidence_role": "frozen span evidence",
        "no_C7_C_or_C8": True,
        "evaluator_modified": False,
        "nms_modified": False,
    }
    atomic_json(Path(args.audit_dir) / "C7_B2_START_STATE.json", start)
    atomic_text(Path(args.audit_dir) / "C7_B2_START_STATE.md", "# C7-B2 start state\n\n```json\n" + json.dumps(start, indent=2, ensure_ascii=False) + "\n```\n")
    build = {
        "proposal_scores": [score_proposal_dataset(args, "train_fit"), score_proposal_dataset(args, "train_calib")],
        "video_datasets": [build_video_dataset(args, "train_fit"), build_video_dataset(args, "train_calib")],
    }
    train = train_head(args)
    score = [score_video_dataset(args, "train_fit"), score_video_dataset(args, "train_calib")]
    eval_payload = run_search(args)
    final = {
        "status": eval_payload["status"],
        "official_val_used": False,
        "official_val_run": False,
        "post_val_adjustment": False,
        "second_official_val": False,
        "C7_B1_archived_no_adjustment": True,
        "build": build,
        "training": train,
        "scoring": score,
        "best": eval_payload["best"],
        "delta_vs_C6_B1": eval_payload["best"]["delta_vs_C6_B1"],
        "delta_vs_C6_B2": eval_payload["best"]["delta_vs_C6_B2"],
        "delta_vs_C7_B1_frozen": eval_payload["best"]["delta_vs_C7_B1_frozen"],
        "evaluator_modified": False,
        "nms_modified": False,
        "C6_C4_artifacts_modified": False,
        "enter_C7_C": False,
        "enter_C8": False,
    }
    atomic_json(Path(args.audit_dir) / "C7_B2_FINAL_DECISION.json", final)
    atomic_text(Path(args.audit_dir) / "C7_B2_FINAL_DECISION.md", "# C7-B2 final decision\n\n```json\n" + json.dumps(final, indent=2, ensure_ascii=False) + "\n```\n")
    atomic_json(Path(args.audit_dir) / "C7_B2_HASHES.json", {
        "status": "C7_B2_HASHES",
        "official_val_used": False,
        "artifacts": {
            "model": artifact(Path(args.output_dir) / "c7_b2_mil_video_residual_head.pt"),
            "best_config": artifact(Path(args.output_dir) / "best_config.json"),
            "eval": artifact(Path(args.audit_dir) / "C7_B2_TRAIN_CALIB_EVAL.json"),
            "final": artifact(Path(args.audit_dir) / "C7_B2_FINAL_DECISION.json"),
            "runner": artifact(__file__),
        },
    })
    print(json.dumps({
        "status": final["status"],
        "official_val_used": False,
        "best_config_id": final["best"]["config_id"],
        "metrics": final["best"]["metrics"],
        "delta_vs_C6_B1": final["delta_vs_C6_B1"],
        "delta_vs_C6_B2": final["delta_vs_C6_B2"],
        "delta_vs_C7_B1_frozen": final["delta_vs_C7_B1_frozen"],
        "movement": final["best"]["movement"],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
