#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from rlem_c6.c6a_utils import (  # noqa: E402
    C6AdapterModel,
    C6GroupDataset,
    artifact,
    c6_collate,
    c6_loss,
    move_batch,
    write_json,
    write_text,
)
from rlem_c6_repair.teacher_a3b import compute_a3b_teacher  # noqa: E402


def stats(x, mask=None):
    if torch.is_tensor(x):
        if mask is not None:
            x = x[mask]
        x = x.detach().float().cpu().numpy()
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {"mean": None, "std": None, "max_abs": None}
    return {"mean": float(x.mean()), "std": float(x.std()), "max_abs": float(np.max(np.abs(x)))}


def corr(a, b):
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    m = np.isfinite(a) & np.isfinite(b)
    a, b = a[m], b[m]
    if len(a) < 3 or np.std(a) <= 1e-12 or np.std(b) <= 1e-12:
        return {"pearson": None, "spearman": None}
    pearson = float(np.corrcoef(a, b)[0, 1])
    ar = np.argsort(np.argsort(a, kind="stable"), kind="stable").astype(np.float64)
    br = np.argsort(np.argsort(b, kind="stable"), kind="stable").astype(np.float64)
    spearman = float(np.corrcoef(ar, br)[0, 1])
    return {"pearson": pearson, "spearman": spearman}


def load_model(run_dir: Path, sample, device):
    ckpt = torch.load(run_dir / "model_best.pt", map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    model = C6AdapterModel(
        ckpt["row_feat_dim"],
        ckpt["video_feat_dim"],
        hidden_dim=int(cfg["hidden_dim"]),
        temporal_layers=int(cfg["temporal_layers"]),
        heads=int(cfg["heads"]),
        ffn_dim=int(cfg["ffn_dim"]),
        span_head_hidden=int(cfg["span_head_hidden"]),
        span_head_layers=int(cfg["span_head_layers"]),
        dropout=float(cfg["dropout"]),
        scale_b=float(cfg["scale_b"]),
        scale_e=float(cfg["scale_e"]),
        alpha_rank=float(cfg["alpha_rank"]),
        alpha_bd=float(cfg["alpha_bd"]),
        alpha_q=float(cfg["alpha_q"]),
    )
    model.load_state_dict(ckpt["model_state"])
    return model.to(device).eval(), cfg, ckpt.get("epoch")


def grad_norms(model):
    groups = {
        "TemporalPriorEncoder": ["inp", "blocks"],
        "BoundaryAdapter": ["boundary"],
        "SpanQualityHead": ["span_head"],
        "VideoGate": ["video_gate"],
        "FinalFusionHead": [],
    }
    out = {}
    for label, prefixes in groups.items():
        vals = []
        for name, p in model.named_parameters():
            if p.grad is None:
                continue
            if not prefixes or any(name.startswith(prefix) for prefix in prefixes):
                vals.append(float(p.grad.detach().norm().cpu()))
        out[label] = float(np.sqrt(np.sum(np.square(vals)))) if vals else 0.0
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--c6a_runs", required=True)
    p.add_argument("--train_cache", default="results/rlem_c6a/cache/train_fit_c6_cache.npz")
    p.add_argument("--train_features", default="results/rlem_c5_lite_prior/train_fit_c5_prior_features.npz")
    p.add_argument("--train_temporal", default="results/rlem_c5_lite_prior_prep/train_fit_temporal_prior.npz")
    p.add_argument("--train_calib_cache", required=True)
    p.add_argument("--train_calib_features", default="results/rlem_c5_lite_prior/train_calib_c5_prior_features.npz")
    p.add_argument("--train_calib_temporal", default="results/rlem_c5_lite_prior_prep/train_calib_temporal_prior.npz")
    p.add_argument("--a3b_stats_json", default="c5_main_a2_video_neutral_audit/C5_MAIN_A2A_STATS_FREEZE.json")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--audit_dir", required=True)
    p.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    args = p.parse_args()
    out_dir, audit_dir = Path(args.output_dir), Path(args.audit_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    audit_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    calib = C6GroupDataset(args.train_calib_cache, args.train_calib_features, args.train_calib_temporal, max_groups=256)
    train = C6GroupDataset(args.train_cache, args.train_features, args.train_temporal, max_groups=128)
    batch = c6_collate([calib[i] for i in range(min(64, len(calib)))])
    teacher = compute_a3b_teacher(cache_npz=args.train_calib_cache, temporal_npz=args.train_calib_temporal, stats_json=args.a3b_stats_json, device=args.device)
    rows = batch["rows"].numpy()
    teacher_batch = np.full(rows.shape, np.nan, dtype=np.float32)
    valid_rows = rows >= 0
    teacher_batch[valid_rows] = teacher["z_delta"][rows[valid_rows]]
    records = []
    for run in sorted(Path(args.c6a_runs).glob("c6a_*")):
        if not (run / "model_best.pt").exists():
            continue
        sample = c6_collate([calib[0]])
        model, cfg, epoch = load_model(run, sample, device)
        tb = {k: v for k, v in batch.items() if k != "rows"}
        with torch.no_grad():
            pred = model(move_batch(tb, device))
        span_mask = tb["span_mask"].to(device)
        rec = {
            "config": run.name,
            "best_epoch": epoch,
            "amplitude": {
                "delta_b": stats(pred["delta_b_raw"], tb["time_mask"].to(device)),
                "delta_e": stats(pred["delta_e_raw"], tb["time_mask"].to(device)),
                "gate_b": stats(pred["gate_b"], tb["time_mask"].to(device)),
                "gate_e": stats(pred["gate_e"], tb["time_mask"].to(device)),
                "endpoint_delta": stats(pred["endpoint_delta"], span_mask),
                "rank_residual": stats(pred["rank_residual"], span_mask),
                "g_video": stats(pred["g_video"]),
            },
            "teacher_correlation": {
                "endpoint_delta_vs_a3b_z": corr(pred["endpoint_delta"][span_mask].detach().cpu().numpy(), teacher_batch[valid_rows]),
                "rank_residual_vs_a3b_z": corr(pred["rank_residual"][span_mask].detach().cpu().numpy(), teacher_batch[valid_rows]),
            },
        }
        train_batch = c6_collate([train[i] for i in range(min(32, len(train)))])
        train_batch.pop("rows")
        model.train()
        model.zero_grad(set_to_none=True)
        out = model(move_batch(train_batch, device))
        loss, parts = c6_loss(move_batch(train_batch, device), out, cfg)
        loss.backward()
        rec["loss_components_one_batch"] = parts
        rec["grad_norms_one_batch"] = grad_norms(model)
        records.append(rec)
    payload = {
        "status": "PASS",
        "stage": "C6-A-R diagnostics",
        "official_val_used": False,
        "teacher": {"config": teacher["config"], "stats_source": teacher["stats_source"], "sanity": teacher["sanity"]},
        "records": records,
        "artifacts": {"train_calib_cache": artifact(args.train_calib_cache), "a3b_stats": artifact(args.a3b_stats_json)},
    }
    write_json(out_dir / "diagnostics.json", payload)
    write_json(audit_dir / "C6A_R_DIAGNOSTIC_AUDIT.json", payload)
    lines = ["# C6-A-R diagnostic audit", "", "- Status: `PASS`", "- Official val used: `false`", ""]
    for r in records:
        lines += [f"## {r['config']}", "", f"- Best epoch: `{r['best_epoch']}`", "### Amplitude"]
        for k, v in r["amplitude"].items():
            lines.append(f"- `{k}`: `{v}`")
        lines += ["### Teacher correlation"]
        lines += [f"- `{k}`: `{v}`" for k, v in r["teacher_correlation"].items()]
        lines += ["### One-batch loss components", f"- `{r['loss_components_one_batch']}`", "### Grad norms", f"- `{r['grad_norms_one_batch']}`", ""]
    write_text(audit_dir / "C6A_R_DIAGNOSTIC_AUDIT.md", "\n".join(lines) + "\n")
    print(json.dumps({"status": "PASS", "records": len(records)}, indent=2))


if __name__ == "__main__":
    main()

