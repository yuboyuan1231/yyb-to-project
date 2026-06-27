#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from rlem_c6.c6a_utils import (  # noqa: E402
    C6AdapterModel,
    artifact,
    c6_collate,
    load_config,
    move_batch,
    write_json,
    write_text,
)
from rlem_c6_repair.losses_repair import boundary_ce_loss  # noqa: E402
from rlem_c6_repair.teacher_a3b import compute_a3b_teacher  # noqa: E402


class BoundaryGroupDataset(Dataset):
    """Lightweight C6 group dataset for boundary-only training/eval.

    It avoids loading the 1.4GB C5 row-feature NPZ because boundary-only loss
    only needs temporal priors and fixed span endpoints.
    """
    def __init__(self, cache_npz: str, temporal_npz: str):
        with np.load(cache_npz, allow_pickle=True) as cache:
            self.group_offsets = cache["group_offsets"].astype(np.int64)
            self.sort_idx = cache["group_sort_idx"].astype(np.int64)
            self.group_ids = cache["group_ids_sorted_unique"].astype(np.int64)
            self.start_idx = cache["start_idx"].astype(np.int64)
            self.end_idx = cache["end_idx"].astype(np.int64)
            self.s_c4_final = cache["s_c4_final"].astype(np.float32)
            self.iou = cache["iou"].astype(np.float32)
            self.y05 = cache["y_joint_05"].astype(np.float32)
            self.y07 = cache["y_joint_07"].astype(np.float32)
            self.is_gt_video = cache["is_gt_video"].astype(np.float32)
            self.gt_start_idx = cache["gt_start_idx"].astype(np.int64)
            self.gt_end_idx = cache["gt_end_idx"].astype(np.int64)
            self.label_relevant = cache["label_relevant"].astype(np.float32)
            self.video_features = cache["video_features"].astype(np.float32)
            self.video_feat_dim = self.video_features.shape[1]
            self.row_feat_dim = 1
        with np.load(temporal_npz, allow_pickle=False) as temporal:
            self.p_b = temporal["p_b"].astype(np.float32)
            self.p_e = temporal["p_e"].astype(np.float32)
            self.p_ctx = temporal["p_ctx"].astype(np.float32)
            self.temporal_length = temporal["temporal_length"].astype(np.int64)

    def __len__(self):
        return len(self.group_offsets)

    def positive_indices(self):
        return np.flatnonzero(self.label_relevant[self.group_ids] > 0.5).astype(np.int64)

    def __getitem__(self, idx):
        s = int(self.group_offsets[idx])
        e = int(self.group_offsets[idx + 1]) if idx + 1 < len(self.group_offsets) else len(self.sort_idx)
        rows = self.sort_idx[s:e]
        gid = int(self.group_ids[idx])
        t = int(self.temporal_length[gid])
        return {
            "group_id": gid,
            "rows": rows.astype(np.int64),
            "p_b": self.p_b[gid, :t],
            "p_e": self.p_e[gid, :t],
            "p_ctx": self.p_ctx[gid, :t],
            "start_idx": self.start_idx[rows],
            "end_idx": self.end_idx[rows],
            "s_c4": self.s_c4_final[rows],
            "row_feat": np.zeros((len(rows), 1), dtype=np.float32),
            "iou": self.iou[rows],
            "y05": self.y05[rows],
            "y07": self.y07[rows],
            "is_gt_video": self.is_gt_video[rows],
            "gt_start_idx": int(self.gt_start_idx[rows[0]]),
            "gt_end_idx": int(self.gt_end_idx[rows[0]]),
            "label_relevant": float(self.label_relevant[gid]),
            "video_feat": self.video_features[gid],
        }


def gaussian_np(length, center, sigma=1.5):
    idx = np.arange(length, dtype=np.float32)
    out = np.exp(-0.5 * ((idx - float(center)) / sigma) ** 2)
    return out / max(float(out.sum()), 1e-8)


def dist_metrics_for_group(item, pb, pe):
    t = len(item["p_b"])
    gs = int(np.clip(item["gt_start_idx"], 0, t - 1))
    ge = int(np.clip(item["gt_end_idx"], 0, t - 1))
    tb, te = gaussian_np(t, gs), gaussian_np(t, ge)
    pb = np.asarray(pb[:t], dtype=np.float64)
    pe = np.asarray(pe[:t], dtype=np.float64)
    pb = pb / max(float(pb.sum()), 1e-12)
    pe = pe / max(float(pe.sum()), 1e-12)
    s_peak, e_peak = int(np.argmax(pb)), int(np.argmax(pe))
    endpoint = np.log(np.maximum(pb[item["start_idx"]], 1e-8)) + np.log(np.maximum(pe[item["end_idx"]], 1e-8))
    order = np.argsort(-endpoint, kind="stable")
    top = int(order[0])
    best = int(np.argmax(item["iou"]))
    inv = np.empty(len(order), dtype=np.int64)
    inv[order] = np.arange(len(order))
    return {
        "start_ce": float(-(tb * np.log(np.maximum(pb, 1e-8))).sum()),
        "end_ce": float(-(te * np.log(np.maximum(pe, 1e-8))).sum()),
        "start_abs_error": float(abs(s_peak - gs)),
        "end_abs_error": float(abs(e_peak - ge)),
        "peak_inside_gt": float(gs <= s_peak <= ge and gs <= e_peak <= ge),
        "oracle_video_r1_05": float(item["iou"][top] >= 0.5),
        "oracle_video_r1_07": float(item["iou"][top] >= 0.7),
        "selected_span_iou": float(item["iou"][top]),
        "best_iou_rank": float(inv[best] + 1),
    }


def summarize(rows):
    keys = rows[0].keys()
    return {k: float(np.mean([r[k] for r in rows])) for k in keys}


def evaluate_model(model, dataset, indices, device, teacher=None):
    frozen_rows, teacher_rows, model_rows = [], [], []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(indices), 128):
            batch_items = [dataset[int(i)] for i in indices[start:start + 128]]
            batch = c6_collate(batch_items)
            rows = batch.pop("rows").numpy()
            out = model(move_batch(batch, device))
            pb_new = out["p_b_new"].detach().cpu().numpy()
            pe_new = out["p_e_new"].detach().cpu().numpy()
            for b, item in enumerate(batch_items):
                frozen_rows.append(dist_metrics_for_group(item, item["p_b"], item["p_e"]))
                model_rows.append(dist_metrics_for_group(item, pb_new[b], pe_new[b]))
                if teacher is not None:
                    gid = item["group_id"]
                    teacher_rows.append(dist_metrics_for_group(item, teacher["p_b_teacher"][gid], teacher["p_e_teacher"][gid]))
    out = {"frozen": summarize(frozen_rows), "oracle_adapter": summarize(model_rows)}
    if teacher is not None:
        out["a3b_teacher"] = summarize(teacher_rows)
    return out


def delta(new, base):
    return {k + "_delta": float(new[k] - base[k]) for k in new if k in base}


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
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch_groups", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-4)
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
            batch.pop("rows")
            tb = move_batch(batch, device)
            opt.zero_grad(set_to_none=True)
            out = model(tb)
            loss, parts = boundary_ce_loss(tb, out, stability=0.1)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))
        rec = {"epoch": epoch, "loss": float(np.mean(losses)), "elapsed_sec": time.time() - t0}
        history.append(rec)
        write_json(out_dir / "history.json", history)
        print(json.dumps(rec, indent=2))
    torch.save({"model_state": model.state_dict(), "config": cfg, "row_feat_dim": 1, "video_feat_dim": train_ds.video_feat_dim, "epoch": args.epochs}, out_dir / "model_best.pt")
    teacher = compute_a3b_teacher(cache_npz=args.calib_cache, temporal_npz=args.calib_temporal, stats_json=args.a3b_stats_json, device=args.device)
    metrics = evaluate_model(model, calib_ds, calib_pos, device, teacher=teacher)
    metrics["oracle_adapter_vs_frozen"] = delta(metrics["oracle_adapter"], metrics["frozen"])
    metrics["oracle_adapter_vs_a3b"] = delta(metrics["oracle_adapter"], metrics["a3b_teacher"])
    passed = (
        metrics["oracle_adapter"]["start_ce"] < metrics["frozen"]["start_ce"]
        and metrics["oracle_adapter"]["end_ce"] < metrics["frozen"]["end_ce"]
        and (
            metrics["oracle_adapter_vs_frozen"]["oracle_video_r1_07_delta"] > 0
            or metrics["oracle_adapter_vs_frozen"]["selected_span_iou_delta"] > 0
        )
    )
    payload = {
        "status": "PASS" if passed else "FAIL",
        "stage": "C6-A-R boundary oracle",
        "official_val_used": False,
        "train_positive_groups": int(len(train_pos)),
        "train_calib_positive_groups": int(len(calib_pos)),
        "history": history,
        "metrics": metrics,
        "artifacts": {
            "model_best": artifact(out_dir / "model_best.pt"),
            "history": artifact(out_dir / "history.json"),
        },
    }
    write_json(out_dir / "metrics.json", payload)
    write_json(audit_dir / "C6A_R_BOUNDARY_ORACLE_AUDIT.json", payload)
    md = "# C6-A-R boundary oracle audit\n\n"
    md += f"- Status: `{payload['status']}`\n- Official val used: `false`\n"
    md += f"- Train positive groups: `{len(train_pos)}`\n- Train_calib positive groups: `{len(calib_pos)}`\n\n"
    md += "## Metrics\n\n```json\n" + json.dumps(metrics, indent=2, ensure_ascii=False) + "\n```\n"
    write_text(audit_dir / "C6A_R_BOUNDARY_ORACLE_AUDIT.md", md)
    print(json.dumps({"status": payload["status"], "metrics": metrics}, indent=2))


if __name__ == "__main__":
    main()

