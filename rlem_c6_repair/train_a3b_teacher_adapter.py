#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from rlem_c6.c6a_utils import C6AdapterModel, artifact, c6_collate, load_config, move_batch, write_json, write_text  # noqa: E402
from rlem_c6_repair.losses_repair import boundary_ce_loss  # noqa: E402
from rlem_c6_repair.teacher_a3b import compute_a3b_teacher  # noqa: E402
from rlem_c6_repair.train_boundary_oracle import BoundaryGroupDataset, delta, evaluate_model  # noqa: E402
from rlem_c6_repair.diagnose_c6a import corr  # noqa: E402


def teacher_loss(batch, out, teacher, rows, device, lambda_teacher=1.0, lambda_stability=0.05):
    gid = batch["group_id"].detach().cpu().numpy()
    pb_t = torch.from_numpy(teacher["p_b_teacher"][gid]).to(device)[:, :batch["p_b"].shape[1]]
    pe_t = torch.from_numpy(teacher["p_e_teacher"][gid]).to(device)[:, :batch["p_e"].shape[1]]
    mask = batch["time_mask"]
    kl = (pb_t * (pb_t.clamp_min(1e-8).log() - out["p_b_new"].clamp_min(1e-8).log())).masked_fill(~mask, 0).sum(1).mean()
    kl = kl + (pe_t * (pe_t.clamp_min(1e-8).log() - out["p_e_new"].clamp_min(1e-8).log())).masked_fill(~mask, 0).sum(1).mean()
    valid = rows >= 0
    target = np.zeros(rows.shape, dtype=np.float32)
    target[valid] = teacher["raw_delta"][rows[valid]]
    target_t = torch.from_numpy(target).to(device)
    mse = ((out["endpoint_delta"] - target_t) ** 2)[batch["span_mask"]].mean()
    stab = ((out["endpoint_delta"]) ** 2)[batch["span_mask"]].mean()
    return float(lambda_teacher) * (kl + mse) + float(lambda_stability) * stab, {
        "teacher_kl": float(kl.detach().cpu()),
        "teacher_mse": float(mse.detach().cpu()),
        "teacher_stability": float(stab.detach().cpu()),
    }


def endpoint_corr(model, dataset, indices, teacher, device):
    vals, tgt = [], []
    model.eval()
    with torch.no_grad():
        for start in range(0, min(len(indices), 4096), 128):
            items = [dataset[int(i)] for i in indices[start:start + 128]]
            batch = c6_collate(items)
            rows = batch.pop("rows").numpy()
            out = model(move_batch(batch, device))
            valid = rows >= 0
            vals.append(out["endpoint_delta"].detach().cpu().numpy()[valid])
            tgt.append(teacher["raw_delta"][rows[valid]])
    return corr(np.concatenate(vals), np.concatenate(tgt))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="rlem_c6/configs/c6a_adapter_default.yaml")
    p.add_argument("--train_cache", required=True)
    p.add_argument("--train_temporal", required=True)
    p.add_argument("--calib_cache", required=True)
    p.add_argument("--calib_temporal", required=True)
    p.add_argument("--a3b_stats_json", default="c5_main_a2_video_neutral_audit/C5_MAIN_A2A_STATS_FREEZE.json")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--audit_dir", required=True)
    p.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch_groups", type=int, default=128)
    p.add_argument("--lr", type=float, default=8e-5)
    p.add_argument("--lambda_teacher", type=float, default=1.0)
    p.add_argument("--lambda_stability", type=float, default=0.05)
    args = p.parse_args()
    out_dir, audit_dir = Path(args.output_dir), Path(args.audit_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    audit_dir.mkdir(parents=True, exist_ok=True)
    cfg = load_config(args.config)
    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    train_ds = BoundaryGroupDataset(args.train_cache, args.train_temporal)
    calib_ds = BoundaryGroupDataset(args.calib_cache, args.calib_temporal)
    train_pos = train_ds.positive_indices()
    calib_pos = calib_ds.positive_indices()
    print("computing train teacher", flush=True)
    teacher_train = compute_a3b_teacher(cache_npz=args.train_cache, temporal_npz=args.train_temporal, stats_json=args.a3b_stats_json, device=args.device)
    print("computing calib teacher", flush=True)
    teacher_calib = compute_a3b_teacher(cache_npz=args.calib_cache, temporal_npz=args.calib_temporal, stats_json=args.a3b_stats_json, device=args.device)
    model = C6AdapterModel(
        row_feat_dim=1,
        video_feat_dim=train_ds.video_feat_dim,
        hidden_dim=int(cfg["hidden_dim"]),
        temporal_layers=int(cfg["temporal_layers"]),
        heads=int(cfg["heads"]),
        ffn_dim=int(cfg["ffn_dim"]),
        span_head_hidden=int(cfg["span_head_hidden"]),
        span_head_layers=int(cfg["span_head_layers"]),
        dropout=float(cfg["dropout"]),
        scale_b=float(cfg["scale_b"]),
        scale_e=float(cfg["scale_e"]),
        alpha_rank=0.0,
        alpha_bd=0.0,
        alpha_q=0.0,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loader = DataLoader(Subset(train_ds, train_pos.tolist()), batch_size=args.batch_groups, shuffle=True, collate_fn=c6_collate, num_workers=0, pin_memory=(device.type == "cuda"))
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        losses = []
        for batch in loader:
            rows = batch.pop("rows").numpy()
            tb = move_batch(batch, device)
            opt.zero_grad(set_to_none=True)
            out = model(tb)
            bl, bp = boundary_ce_loss(tb, out, stability=0.1)
            tl, tp = teacher_loss(tb, out, teacher_train, rows, device, args.lambda_teacher, args.lambda_stability)
            loss = bl + tl
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))
        rec = {"epoch": epoch, "loss": float(np.mean(losses)), "elapsed_sec": time.time() - t0}
        history.append(rec)
        write_json(out_dir / "history.json", history)
        print(json.dumps(rec, indent=2), flush=True)
    torch.save({"model_state": model.state_dict(), "config": cfg, "row_feat_dim": 1, "video_feat_dim": train_ds.video_feat_dim, "epoch": args.epochs}, out_dir / "model_best.pt")
    metrics = evaluate_model(model, calib_ds, calib_pos, device, teacher=teacher_calib)
    metrics["teacher_adapter_vs_frozen"] = delta(metrics["oracle_adapter"], metrics["frozen"])
    metrics["teacher_adapter_vs_a3b"] = delta(metrics["oracle_adapter"], metrics["a3b_teacher"])
    ecorr = endpoint_corr(model, calib_ds, calib_pos, teacher_calib, device)
    # "Close to A3b" is deliberately localization-oriented: adapter must not
    # underperform A3b by more than tiny tolerances on the key GT-video span metrics.
    close_to_a3b = (
        metrics["teacher_adapter_vs_a3b"]["oracle_video_r1_07_delta"] >= -0.001
        and metrics["teacher_adapter_vs_a3b"]["selected_span_iou_delta"] >= -0.0002
        and ecorr["pearson"] is not None and ecorr["pearson"] > 0.3
    )
    payload = {
        "status": "PASS" if close_to_a3b else "FAIL",
        "stage": "C6-A-R A3b teacher distillation",
        "official_val_used": False,
        "history": history,
        "metrics": metrics,
        "endpoint_delta_correlation_with_a3b_raw": ecorr,
        "train_positive_groups": int(len(train_pos)),
        "train_calib_positive_groups": int(len(calib_pos)),
        "artifacts": {
            "model_best": artifact(out_dir / "model_best.pt"),
            "history": artifact(out_dir / "history.json"),
        },
    }
    write_json(out_dir / "metrics.json", payload)
    write_json(audit_dir / "C6A_R_TEACHER_AUDIT.json", payload)
    md = "# C6-A-R A3b teacher audit\n\n"
    md += f"- Status: `{payload['status']}`\n- Official val used: `false`\n"
    md += f"- Endpoint delta correlation with A3b raw: `{ecorr}`\n\n"
    md += "## Metrics\n\n```json\n" + json.dumps(metrics, indent=2, ensure_ascii=False) + "\n```\n"
    write_text(audit_dir / "C6A_R_TEACHER_AUDIT.md", md)
    print(json.dumps({"status": payload["status"], "endpoint_corr": ecorr, "metrics": metrics}, indent=2))


if __name__ == "__main__":
    main()

