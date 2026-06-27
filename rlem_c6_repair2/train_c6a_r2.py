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

from rlem.c5_prior_utils import attach_eval_labels, fast_metrics, localization_diagnostics, metric_deltas, movement_diagnostics  # noqa: E402
from rlem.c4_lite_utils import selection_score  # noqa: E402
from rlem_c6.c6a_utils import (  # noqa: E402
    C6AdapterModel,
    artifact,
    c6_collate,
    load_config,
    move_batch,
    write_json,
)
from rlem_c6_repair.train_boundary_oracle import BoundaryGroupDataset, delta, evaluate_model  # noqa: E402
from rlem_c6_repair2.boundary_init import load_boundary_init  # noqa: E402
from rlem_c6_repair2.losses_r2 import boundary_loss_relevant, iou_quality_loss, listwise_iou_loss, query_front_rank_proxy  # noqa: E402


def selected_indices(dataset: BoundaryGroupDataset, neg_ratio: int = 3):
    pos = dataset.positive_indices()
    gid_to_idx = {int(g): i for i, g in enumerate(dataset.group_ids)}
    # High-C4 wrong/negative videos are the relevant hard negatives.
    group_max = np.full(len(dataset.group_ids), -np.inf, dtype=np.float32)
    for i in range(len(dataset.group_offsets)):
        s = int(dataset.group_offsets[i])
        e = int(dataset.group_offsets[i + 1]) if i + 1 < len(dataset.group_offsets) else len(dataset.sort_idx)
        rows = dataset.sort_idx[s:e]
        group_max[i] = float(np.max(dataset.s_c4_final[rows]))
    neg = np.flatnonzero(dataset.label_relevant[dataset.group_ids] <= 0.5)
    neg = neg[np.argsort(-group_max[neg], kind="stable")]
    keep_neg = neg[: min(len(neg), len(pos) * neg_ratio)]
    return np.unique(np.concatenate([pos, keep_neg])).astype(np.int64), pos.astype(np.int64)


def make_model(cfg, dataset, device):
    return C6AdapterModel(
        row_feat_dim=1,
        video_feat_dim=dataset.video_feat_dim,
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


def score_batch_custom(batch, out, stats, cfg):
    z = (out["endpoint_delta"] - float(stats["mean"])) / max(float(stats["std"]), 1e-8)
    return (
        batch["s_c4"]
        + float(cfg["alpha_endpoint"]) * z
        + float(cfg["alpha_quality"]) * out["quality_logit"]
        + float(cfg["alpha_rank"]) * out["g_video"][:, None] * out["rank_residual"]
    ).masked_fill(~batch["span_mask"], -1e9)


def compute_endpoint_stats(model, dataset, device, indices=None, batch_groups=256):
    model.eval()
    vals = []
    if indices is None:
        indices = np.arange(len(dataset), dtype=np.int64)
    loader = DataLoader(Subset(dataset, indices.tolist()), batch_size=batch_groups, shuffle=False, collate_fn=c6_collate, num_workers=0, pin_memory=(device.type == "cuda"))
    with torch.no_grad():
        for batch in loader:
            batch.pop("rows")
            tb = move_batch(batch, device)
            out = model(tb)
            vals.append(out["endpoint_delta"][tb["span_mask"]].detach().float().cpu().numpy())
    x = np.concatenate(vals).astype(np.float64)
    return {"mean": float(x.mean()), "std": float(max(x.std(), 1e-8)), "count": int(x.size)}


def score_dataset_custom(model, dataset, device, stats, cfg, batch_groups=256):
    model.eval()
    score = np.empty(len(dataset.s_c4_final), dtype=np.float32)
    loader = DataLoader(dataset, batch_size=batch_groups, shuffle=False, collate_fn=c6_collate, num_workers=0, pin_memory=(device.type == "cuda"))
    with torch.no_grad():
        for batch in loader:
            rows = batch.pop("rows").numpy()
            tb = move_batch(batch, device)
            out = model(tb)
            sc = score_batch_custom(tb, out, stats, cfg).detach().float().cpu().numpy()
            for b in range(rows.shape[0]):
                valid = rows[b] >= 0
                score[rows[b, valid]] = sc[b, valid]
    return score


def eval_train_calib(cache_npz, score, gt_jsonl, baseline, boundary_metrics):
    cache = {k: v for k, v in np.load(cache_npz, allow_pickle=True).items()}
    attach_eval_labels(cache, gt_jsonl)
    metrics, _, _ = fast_metrics(cache, score, 100, 100, 0.7)
    bm, _, _ = fast_metrics(cache, baseline, 100, 100, 0.7)
    out = {
        "metrics": metrics,
        "baseline_metrics": bm,
        "deltas_vs_c4_final": metric_deltas(metrics, bm),
        "selection_delta_vs_c4_final": float(selection_score(metrics, bm)),
        "movement": movement_diagnostics(cache, baseline, score),
        "localization": localization_diagnostics(cache, baseline, score),
        "boundary_metrics": boundary_metrics,
    }
    return out


def front_selection(ev):
    d = ev["deltas_vs_c4_final"]
    loc = ev["localization"]
    loc_pos = sum([loc.get("oracle_video_r1_05_delta", 0) > 0, loc.get("oracle_video_r1_07_delta", 0) > 0, loc.get("selected_span_miou_delta", 0) > 0, loc.get("best_iou_span_rank_delta_mean", 1) < 0])
    return (
        2 * d.get("0.5-r1", 0) + 1.5 * d.get("0.7-r1", 0)
        + d.get("0.5-r5", 0) + d.get("0.7-r5", 0)
        + 0.5 * d.get("0.5-r10", 0) + 0.5 * d.get("0.7-r10", 0)
        + 0.5 * loc_pos - max(0, -d.get("0.5-r100", 0)) - max(0, -d.get("0.7-r100", 0))
        - 0.5 * ev["movement"].get("hard_positive_top100_exit_ratio", 0)
    )


def train_epoch(model, loader, opt, device, cfg, stats, stage, epoch_idx):
    model.train()
    losses = []
    rows = 0
    t0 = time.time()
    for batch in loader:
        rows += int(batch["span_mask"].sum())
        batch.pop("rows")
        tb = move_batch(batch, device)
        opt.zero_grad(set_to_none=True)
        # R2 deliberately keeps this forward in fp32.  The C6-A model showed
        # NaN span-head logits under AMP on 3090/torch-2.0.1 for this repair
        # objective, while fp32 remains stable.  This is an efficiency tradeoff,
        # not a precision shortcut.
        out = model(tb)
        score = score_batch_custom(tb, out, stats, cfg)
        bl, _ = boundary_loss_relevant(tb, out, stability=float(cfg.get("boundary_stability", 0.05)))
        ql = iou_quality_loss(tb, out)
        if stage == 1:
            loss = 1.0 * bl + 0.2 * ql + 0.05 * ((score[tb["span_mask"]] - tb["s_c4"][tb["span_mask"]]) ** 2).mean()
        else:
            wl = listwise_iou_loss(score, tb["iou"], tb["span_mask"])
            fl = query_front_rank_proxy(tb, score)
            ramp = 0.05 + 0.10 * min(max(epoch_idx - 1, 0), 4) / 4.0
            stab = ((score[tb["span_mask"]] - tb["s_c4"][tb["span_mask"]]) ** 2).mean()
            sparse = out["g_video"].mean() + out["endpoint_delta"][tb["span_mask"]].abs().mean()
            loss = 0.7 * bl + 1.0 * wl + 0.5 * ql + 0.3 * fl + ramp * stab + 0.02 * sparse
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg.get("gradient_clip", 1.0)))
        opt.step()
        losses.append(float(loss.detach().cpu()))
    return {"loss": float(np.mean(losses)), "elapsed_sec": time.time() - t0, "rows_per_sec": float(rows / max(time.time() - t0, 1e-6))}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--boundary_init", required=True)
    p.add_argument("--train_cache", required=True)
    p.add_argument("--train_temporal", required=True)
    p.add_argument("--calib_cache", required=True)
    p.add_argument("--calib_temporal", required=True)
    p.add_argument("--calib_gt", required=True)
    p.add_argument("--boundary_oracle_metrics", default="results/rlem_c6a_repair/boundary_oracle/metrics.json")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    args = p.parse_args()
    cfg = load_config(args.config)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if (out_dir / "history.json").exists():
        raise FileExistsError(out_dir / "history.json")
    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    train_ds = BoundaryGroupDataset(args.train_cache, args.train_temporal)
    calib_ds = BoundaryGroupDataset(args.calib_cache, args.calib_temporal)
    model = make_model(cfg, train_ds, device)
    init_audit = load_boundary_init(model, args.boundary_init)
    train_idx, train_pos = selected_indices(train_ds, neg_ratio=int(cfg.get("hard_negative_ratio", 3)))
    calib_pos = calib_ds.positive_indices()
    # Frozen train_fit endpoint stats after boundary initialization.
    # Frozen endpoint normalization statistics: estimate from train_fit only,
    # before R2 ranking-aware updates.  No train_calib/official-val stats.
    stats = compute_endpoint_stats(model, train_ds, device, indices=None, batch_groups=int(cfg["batch_groups"]))
    opt = torch.optim.AdamW([
        {"params": list(model.inp.parameters()) + list(model.blocks.parameters()) + list(model.boundary.parameters()), "lr": float(cfg["lr_encoder"])},
        {"params": list(model.span_head.parameters()) + list(model.video_gate.parameters()), "lr": float(cfg["lr_new_heads"])},
    ], weight_decay=float(cfg["weight_decay"]))
    loader_pos = DataLoader(Subset(train_ds, train_pos.tolist()), batch_size=int(cfg["batch_groups"]), shuffle=True, collate_fn=c6_collate, num_workers=0, pin_memory=(device.type == "cuda"))
    loader_mix = DataLoader(Subset(train_ds, train_idx.tolist()), batch_size=int(cfg["batch_groups"]), shuffle=True, collate_fn=c6_collate, num_workers=0, pin_memory=(device.type == "cuda"))
    history = []
    boundary_ref = json.load(open(args.boundary_oracle_metrics))["metrics"]["oracle_adapter"]
    best = None
    for epoch in range(1, int(cfg["stage1_epochs"]) + 1):
        rec = {"epoch": epoch, "stage": 1, **train_epoch(model, loader_pos, opt, device, cfg, stats, 1, epoch)}
        bm = evaluate_model(model, calib_ds, calib_pos, device, teacher=None)["oracle_adapter"]
        rec["boundary_metrics"] = bm
        rec["boundary_vs_boundary_oracle"] = delta(bm, boundary_ref)
        history.append(rec)
        write_json(out_dir / "history.json", history)
        print(json.dumps(rec, indent=2), flush=True)
    last_stage1 = history[-1]["boundary_vs_boundary_oracle"]
    if not (
        last_stage1["start_ce_delta"] <= 0.05
        and last_stage1["end_ce_delta"] <= 0.05
        and last_stage1["oracle_video_r1_05_delta"] >= -0.001
        and last_stage1["oracle_video_r1_07_delta"] >= -0.001
        and last_stage1["selected_span_iou_delta"] >= -0.0002
    ):
        status = "STAGE1_BOUNDARY_RETENTION_FAIL"
    else:
        status = "TRAINED"
        baseline = np.load(args.calib_cache, allow_pickle=True)["s_c4_final"].astype(np.float32)
        for i in range(1, int(cfg["stage2_epochs"]) + 1):
            epoch = int(cfg["stage1_epochs"]) + i
            rec = {"epoch": epoch, "stage": 2, **train_epoch(model, loader_mix, opt, device, cfg, stats, 2, i)}
            score = score_dataset_custom(model, calib_ds, device, stats, cfg, batch_groups=int(cfg["batch_groups"]))
            bm = evaluate_model(model, calib_ds, calib_pos, device, teacher=None)["oracle_adapter"]
            ev = eval_train_calib(args.calib_cache, score, args.calib_gt, baseline, bm)
            rec.update(ev)
            rec["boundary_vs_boundary_oracle"] = delta(bm, boundary_ref)
            rec["selection_score"] = front_selection(ev)
            history.append(rec)
            write_json(out_dir / "history.json", history)
            if best is None or rec["selection_score"] > best["selection_score"]:
                best = rec
                torch.save({"model_state": model.state_dict(), "config": cfg, "endpoint_stats": stats, "epoch": epoch, "row_feat_dim": 1, "video_feat_dim": train_ds.video_feat_dim, "init_audit": init_audit}, out_dir / "model_best.pt")
                with open(out_dir / "train_calib_scores_best.npz.partial", "wb") as f:
                    np.savez_compressed(f, score=score.astype(np.float32))
                Path(out_dir / "train_calib_scores_best.npz.partial").replace(out_dir / "train_calib_scores_best.npz")
                write_json(out_dir / "best_epoch.json", rec)
            print(json.dumps({"epoch": epoch, "selection_score": rec["selection_score"], "deltas": rec["deltas_vs_c4_final"]}, indent=2), flush=True)
    if best is None:
        best = history[-1]
    audit = {"status": status, "config": cfg, "init_audit": init_audit, "endpoint_stats_train_fit": stats, "best_epoch": best.get("epoch"), "best": best, "official_val_used": False, "post_val_adjustment": False, "artifacts": {"boundary_init": artifact(args.boundary_init), "history": artifact(out_dir / "history.json"), "model_best": artifact(out_dir / "model_best.pt")}}
    write_json(out_dir / "train_audit.json", audit)
    print(json.dumps({"status": status, "best_epoch": best.get("epoch"), "best_selection": best.get("selection_score")}, indent=2))


if __name__ == "__main__":
    main()
