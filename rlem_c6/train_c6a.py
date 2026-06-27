#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from rlem_c6.c6a_utils import (
    C6AdapterModel,
    C6GroupDataset,
    artifact,
    c6_collate,
    c6_loss,
    evaluate_score,
    load_config,
    move_batch,
    score_dataset,
    seed_everything,
    write_json,
)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--train_cache", required=True)
    p.add_argument("--train_features", required=True)
    p.add_argument("--train_temporal", required=True)
    p.add_argument("--calib_cache", required=True)
    p.add_argument("--calib_features", required=True)
    p.add_argument("--calib_temporal", required=True)
    p.add_argument("--calib_gt", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--epochs_override", type=int)
    p.add_argument("--max_train_groups", type=int)
    p.add_argument("--max_calib_groups", type=int)
    p.add_argument("--seed", type=int, default=13)
    args = p.parse_args()
    cfg = load_config(args.config)
    seed_everything(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if (out_dir / "history.json").exists():
        raise FileExistsError(out_dir / "history.json")
    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    train_ds = C6GroupDataset(args.train_cache, args.train_features, args.train_temporal, max_groups=args.max_train_groups)
    calib_ds = C6GroupDataset(args.calib_cache, args.calib_features, args.calib_temporal, max_groups=args.max_calib_groups)
    sample = c6_collate([train_ds[0]])
    model = C6AdapterModel(
        sample["row_feat"].shape[-1],
        sample["video_feat"].shape[-1],
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
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg["lr"]), weight_decay=float(cfg["weight_decay"]))
    scaler = torch.cuda.amp.GradScaler(enabled=bool(cfg.get("amp", True)) and device.type == "cuda")
    loader = DataLoader(
        train_ds,
        batch_size=int(cfg["batch_groups"]),
        shuffle=True,
        num_workers=args.workers,
        collate_fn=c6_collate,
        pin_memory=(device.type == "cuda"),
        persistent_workers=args.workers > 0,
    )
    baseline = np.load(args.calib_cache, allow_pickle=True)["s_c4_final"].astype(np.float32)
    epochs = int(args.epochs_override or cfg["epochs"])
    history = []
    best = None
    for epoch in range(1, epochs + 1):
        model.train()
        t0 = time.time()
        losses = []
        row_seen = 0
        for batch in loader:
            row_seen += int(batch["span_mask"].sum())
            batch.pop("rows")
            tb = move_batch(batch, device)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=bool(cfg.get("amp", True)) and device.type == "cuda"):
                pred = model(tb)
                loss, loss_parts = c6_loss(tb, pred, cfg)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["gradient_clip"]))
            scaler.step(opt)
            scaler.update()
            losses.append(loss_parts["loss"])
        score = score_dataset(
            model,
            calib_ds,
            device=args.device,
            batch_groups=max(64, int(cfg["batch_groups"])),
            amp=bool(cfg.get("amp", True)),
        )
        ev = evaluate_score(args.calib_cache, score, args.calib_gt, baseline)
        deltas = ev["deltas_vs_c4_final"]
        loc = ev.get("localization", {})
        loc_pos = sum([
            loc.get("oracle_video_r1_05_delta", 0.0) > 0,
            loc.get("oracle_video_r1_07_delta", 0.0) > 0,
            loc.get("selected_span_miou_delta", 0.0) > 0,
            loc.get("best_iou_span_rank_delta_mean", 1.0) < 0,
        ])
        score_sel = (
            2.0 * deltas.get("0.5-r1", 0.0)
            + 1.5 * deltas.get("0.7-r1", 0.0)
            + deltas.get("0.5-r5", 0.0)
            + deltas.get("0.7-r5", 0.0)
            + 0.5 * deltas.get("0.5-r10", 0.0)
            + 0.5 * deltas.get("0.7-r10", 0.0)
            + 0.5 * loc_pos
            - max(0.0, -deltas.get("0.5-r100", 0.0))
            - max(0.0, -deltas.get("0.7-r100", 0.0))
            - 0.5 * ev.get("movement", {}).get("hard_positive_top100_exit_ratio", 0.0)
        )
        elapsed = time.time() - t0
        rec = {
            "epoch": epoch,
            "train_loss_mean": float(np.mean(losses)),
            "elapsed_sec": float(elapsed),
            "rows_per_sec": float(row_seen / max(elapsed, 1e-6)),
            "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
            "selection_score": float(score_sel),
            **ev,
        }
        history.append(rec)
        write_json(out_dir / "history.json", history)
        if best is None or rec["selection_score"] > best["selection_score"]:
            best = rec
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "config": cfg,
                    "epoch": epoch,
                    "row_feat_dim": sample["row_feat"].shape[-1],
                    "video_feat_dim": sample["video_feat"].shape[-1],
                },
                out_dir / "model_best.pt",
            )
            with open(out_dir / "train_calib_scores_best.npz.partial", "wb") as f:
                np.savez_compressed(f, score=score.astype(np.float32))
            Path(out_dir / "train_calib_scores_best.npz.partial").replace(out_dir / "train_calib_scores_best.npz")
            write_json(out_dir / "best_epoch.json", rec)
        print(json.dumps({"epoch": epoch, "loss": rec["train_loss_mean"], "selection_score": rec["selection_score"]}, indent=2))
    write_json(
        out_dir / "train_audit.json",
        {
            "status": "PASS",
            "config": cfg,
            "best_epoch": best["epoch"],
            "official_val_used": False,
            "artifacts": {
                "config": artifact(args.config),
                "model_best": artifact(out_dir / "model_best.pt"),
                "history": artifact(out_dir / "history.json"),
                "scores": artifact(out_dir / "train_calib_scores_best.npz"),
            },
        },
    )


if __name__ == "__main__":
    main()
