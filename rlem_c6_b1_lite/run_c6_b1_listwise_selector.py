#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from rlem_c6.c6a_utils import write_json, write_text  # noqa: E402
from rlem_c6_b0.run_c6_b0_candidate_diagnostic import artifact, infer_boundary_prior, load_gt_ts  # noqa: E402
from rlem_c6_b1_lite.run_c6_b1_r1_safe_selector import (  # noqa: E402
    build_pool,
    grid_search_policy,
    load_cache,
    save_npz,
)


class ListwiseSlotSelector(nn.Module):
    """Listwise selector over original + boundary alternatives for one slot."""

    def __init__(
        self,
        in_dim: int,
        alt_count: int,
        hidden: int = 384,
        embed_layers: int = 3,
        transformer_layers: int = 2,
        heads: int = 8,
        dropout: float = 0.10,
    ):
        super().__init__()
        self.alt_emb = nn.Embedding(alt_count, hidden)
        mods: List[nn.Module] = []
        d = in_dim
        for _ in range(embed_layers):
            mods += [nn.Linear(d, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(dropout)]
            d = hidden
        self.embed = nn.Sequential(*mods)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=heads,
            dim_feedforward=hidden * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=transformer_layers)
        self.score = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )
        self.quality = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        b, a, _ = x.shape
        h = self.embed(x)
        alt_idx = torch.arange(a, device=x.device)[None, :].expand(b, a)
        h = h + self.alt_emb(alt_idx)
        h = self.encoder(h)
        ctx = h.mean(dim=1, keepdim=True).expand_as(h)
        z = torch.cat([h, ctx], dim=-1)
        return {
            "score": self.score(z).squeeze(-1),
            "quality": self.quality(z).squeeze(-1),
        }


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def prepare_groups(pool: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    q_count, slots, alt_count = [int(x) for x in pool["shape"]]
    n = q_count * slots
    x = pool["x"].astype(np.float32).reshape(n, alt_count, -1)
    y05 = pool["y05"].astype(np.float32).reshape(n, alt_count)
    y07 = pool["y07"].astype(np.float32).reshape(n, alt_count)
    yiou = pool["yiou"].astype(np.float32).reshape(n, alt_count)
    slot = pool["slot"].astype(np.int64).reshape(n, alt_count)[:, 0]
    utility = y05 + 0.75 * y07 + 0.35 * yiou
    target = np.argmax(utility, axis=1).astype(np.int64)
    # If the whole alternative list has no useful span, keep original.
    target[np.max(utility, axis=1) <= 0.0] = 0
    weights = np.where(slot == 0, 4.0, np.where(slot < 3, 2.0, 1.0)).astype(np.float32)
    return {
        "x": x,
        "target": target,
        "utility": utility.astype(np.float32),
        "yiou": yiou.astype(np.float32),
        "slot": slot,
        "weights": weights,
        "shape": np.asarray([q_count, slots, alt_count], dtype=np.int64),
    }


def train_listwise(
    train_pool: Dict[str, np.ndarray],
    out_dir: Path,
    device_name: str,
    epochs: int,
    batch_slots: int,
    hidden: int,
    transformer_layers: int,
    heads: int,
    variant: str,
    teacher_score_path: str | None = None,
    model_path: str | Path | None = None,
    history_path: str | Path | None = None,
) -> Dict[str, Any]:
    groups = prepare_groups(train_pool)
    x = groups["x"]
    mean = x.reshape(-1, x.shape[-1]).mean(axis=0).astype(np.float32)
    std = np.maximum(x.reshape(-1, x.shape[-1]).std(axis=0), 1e-6).astype(np.float32)
    xz = ((x - mean) / std).astype(np.float32)
    target = groups["target"].astype(np.int64)
    utility = groups["utility"].astype(np.float32)
    yiou = groups["yiou"].astype(np.float32)
    weights = groups["weights"].astype(np.float32)
    teacher_scores = None
    if variant == "distill":
        if teacher_score_path is None:
            raise ValueError("distill variant requires teacher_score_path")
        with np.load(teacher_score_path, allow_pickle=False) as payload:
            teacher_scores = payload["score"].astype(np.float32).reshape(x.shape[0], x.shape[1])
        # Match score scale per slot group; distillation should teach relative
        # alternative preference, not absolute threshold.
        teacher_scores = teacher_scores - teacher_scores[:, :1]
    device = torch.device("cuda" if device_name == "cuda" and torch.cuda.is_available() else "cpu")
    tensors = [
        torch.from_numpy(xz),
        torch.from_numpy(target),
        torch.from_numpy(utility),
        torch.from_numpy(yiou),
        torch.from_numpy(weights),
    ]
    if teacher_scores is not None:
        tensors.append(torch.from_numpy(teacher_scores.astype(np.float32)))
    ds = TensorDataset(*tensors)
    dl = DataLoader(ds, batch_size=batch_slots, shuffle=True, num_workers=0, pin_memory=(device.type == "cuda"))
    model = ListwiseSlotSelector(
        x.shape[-1],
        alt_count=x.shape[1],
        hidden=hidden,
        transformer_layers=transformer_layers,
        heads=heads,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    history = []
    history_out = Path(history_path) if history_path is not None else out_dir / "history_v2_listwise.json"
    model_out = Path(model_path) if model_path is not None else out_dir / "model_best_v2_listwise.pt"
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        accs = []
        t0 = time.time()
        for batch in dl:
            if teacher_scores is not None:
                xb, tgt, util, iou, ww, teacher = batch
                teacher = teacher.to(device, non_blocking=True)
            else:
                xb, tgt, util, iou, ww = batch
                teacher = None
            xb = xb.to(device, non_blocking=True)
            tgt = tgt.to(device, non_blocking=True)
            util = util.to(device, non_blocking=True)
            iou = iou.to(device, non_blocking=True)
            ww = ww.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                out = model(xb)
                score = out["score"]
                quality = out["quality"]
                ce = F.cross_entropy(score, tgt, reduction="none")
                # Soft list target adds graded signal, useful when several spans are close.
                soft_target = torch.softmax(util / 0.35, dim=1)
                logp = F.log_softmax(score, dim=1)
                soft_ce = -(soft_target * logp).sum(dim=1)
                q_loss = F.smooth_l1_loss(torch.sigmoid(quality), iou, reduction="none").mean(dim=1)
                # Original-vs-best margin: only pushes replacement when utility warrants it.
                best = torch.argmax(util, dim=1)
                best_score = score.gather(1, best[:, None]).squeeze(1)
                orig_score = score[:, 0]
                best_util = util.gather(1, best[:, None]).squeeze(1)
                orig_util = util[:, 0]
                sign = torch.where(best_util > orig_util + 1e-6, 1.0, -1.0)
                margin_size = 0.20
                margin_weight = 0.25
                ce_weight = 1.0
                soft_weight = 0.5
                distill_loss = 0.0
                if variant == "pairwise":
                    margin_size = 0.45
                    margin_weight = 1.0
                    ce_weight = 0.8
                    soft_weight = 0.35
                    # Additional one-vs-all replacement pressure: if a boundary
                    # alternative beats original utility, push it above original.
                    improve = (util[:, 1:] > util[:, :1] + 1e-6).float()
                    pair_margin = F.relu(0.35 - (score[:, 1:] - score[:, :1])) * improve
                    pair_loss = pair_margin.sum(dim=1) / improve.sum(dim=1).clamp_min(1.0)
                else:
                    pair_loss = 0.0
                if variant == "distill" and teacher is not None:
                    teacher_prob = torch.softmax(teacher / 0.75, dim=1)
                    distill_loss = -(teacher_prob * F.log_softmax(score, dim=1)).sum(dim=1)
                    ce_weight = 0.65
                    soft_weight = 0.25
                    margin_weight = 0.20
                margin = F.relu(margin_size - sign * (best_score - orig_score))
                loss = (ww * (ce_weight * ce + soft_weight * soft_ce + 0.25 * q_loss + margin_weight * margin + 0.75 * pair_loss + 0.75 * distill_loss)).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))
            accs.append(float((score.argmax(dim=1) == tgt).float().mean().detach().cpu()))
        rec = {"epoch": epoch, "loss": float(np.mean(losses)), "target_acc": float(np.mean(accs)), "elapsed_sec": float(time.time() - t0)}
        history.append(rec)
        write_json(history_out, history)
        print(json.dumps({"stage": "train_listwise", **rec}))
    torch.save(
        {
            "model_state": model.state_dict(),
            "feature_mean": mean,
            "feature_std": std,
            "in_dim": int(x.shape[-1]),
            "alt_count": int(x.shape[1]),
            "hidden": int(hidden),
            "transformer_layers": int(transformer_layers),
            "heads": int(heads),
            "epochs": int(epochs),
        },
        model_out,
    )
    return {
        "history": history,
        "model_artifact": artifact(model_out),
        "train_slot_groups": int(len(x)),
        "train_candidate_examples": int(len(train_pool["x"])),
        "feature_dim": int(x.shape[-1]),
        "device": str(device),
        "variant": variant,
        "teacher_score_path": teacher_score_path,
    }


@torch.no_grad()
def score_listwise(pool: Dict[str, np.ndarray], model_path: str, device_name: str, batch_slots: int) -> np.ndarray:
    ckpt = torch.load(model_path, map_location="cpu")
    groups = prepare_groups(pool)
    x = groups["x"].astype(np.float32)
    xz = ((x - ckpt["feature_mean"]) / ckpt["feature_std"]).astype(np.float32)
    model = ListwiseSlotSelector(
        int(ckpt["in_dim"]),
        alt_count=int(ckpt["alt_count"]),
        hidden=int(ckpt["hidden"]),
        transformer_layers=int(ckpt["transformer_layers"]),
        heads=int(ckpt["heads"]),
    )
    model.load_state_dict(ckpt["model_state"], strict=True)
    device = torch.device("cuda" if device_name == "cuda" and torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    scores = np.empty((xz.shape[0], xz.shape[1]), dtype=np.float32)
    for start in range(0, len(xz), batch_slots):
        xb = torch.from_numpy(xz[start:start + batch_slots]).to(device)
        out = model(xb)
        sc = out["score"] + 0.10 * torch.sigmoid(out["quality"])
        scores[start:start + len(sc)] = sc.float().cpu().numpy().astype(np.float32)
    return scores.reshape(-1)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--train_cache", default="results/rlem_c6a/cache/train_fit_c6_cache.npz")
    p.add_argument("--train_temporal", default="results/rlem_c5_lite_prior_prep/train_fit_temporal_prior.npz")
    p.add_argument("--calib_cache", default="results/rlem_c6a/cache/train_calib_c6_cache.npz")
    p.add_argument("--calib_temporal", default="results/rlem_c5_lite_prior_prep/train_calib_temporal_prior.npz")
    p.add_argument("--boundary_oracle_ckpt", default="results/rlem_c6a_repair/boundary_oracle/model_best.pt")
    p.add_argument("--calib_gt_jsonl", default="results/rlem_c3_minimal/train_calib_gt.jsonl")
    p.add_argument("--base_pool_dir", default="results/rlem_c6_b1_lite_r1_safe")
    p.add_argument("--output_dir", default="results/rlem_c6_b1_lite_v2_listwise")
    p.add_argument("--audit_dir", default="c6_b1_lite_v2_audit")
    p.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    p.add_argument("--batch_groups", type=int, default=512)
    p.add_argument("--slots", type=int, default=5)
    p.add_argument("--alt_count", type=int, default=8)
    p.add_argument("--top_endpoint", type=int, default=24)
    p.add_argument("--effective_top_n", type=int, default=100)
    p.add_argument("--max_after_nms", type=int, default=100)
    p.add_argument("--nms_thd", type=float, default=0.7)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--batch_slots", type=int, default=8192)
    p.add_argument("--hidden", type=int, default=384)
    p.add_argument("--transformer_layers", type=int, default=2)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--variant", choices=["base", "pairwise", "distill"], default="base")
    p.add_argument("--teacher_score_path", default="results/rlem_c6_b1_lite_r1_safe/train_calib_candidate_scores.npz")
    p.add_argument("--train_teacher_score_path", default=None, help="Optional train_fit teacher scores; if omitted for distill, teacher is generated by v1 model only if available in future code.")
    args = p.parse_args()
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    audit_dir = Path(args.audit_dir); audit_dir.mkdir(parents=True, exist_ok=True)
    base_pool_dir = Path(args.base_pool_dir)
    start_state = {
        "stage": f"C6-B1-lite-v2 listwise slot selector ({args.variant})",
        "scope": "train_fit/train_calib only",
        "official_val_used": False,
        "post_val_adjustment": False,
        "C6_C_used": False,
        "design": "listwise per-slot Transformer selector over original+boundary alternatives",
        "variant": args.variant,
        "model": {"hidden": args.hidden, "transformer_layers": args.transformer_layers, "heads": args.heads},
        "references": [
            "listwise learning-to-rank softmax/CE losses",
            "BMN-style proposal confidence for boundary candidates",
            "reranking precision-at-top motivation"
        ],
    }
    write_json(audit_dir / "C6_B1_LITE_V2_LISTWISE_START_STATE.json", start_state)
    write_text(audit_dir / "C6_B1_LITE_V2_LISTWISE_START_STATE.md", "# C6-B1-lite-v2 listwise start\n\n```json\n" + json.dumps(start_state, indent=2, ensure_ascii=False) + "\n```\n")

    # Reuse v1 candidate pools when available; they are deterministic and already audited.
    train_pool_path = base_pool_dir / "train_fit_candidate_pool_top_slots.npz"
    calib_pool_path = base_pool_dir / "train_calib_candidate_pool_top_slots.npz"
    train_prior = base_pool_dir / "train_fit_boundary_oracle_prior.npz"
    calib_prior = base_pool_dir / "train_calib_boundary_oracle_prior.npz"
    if not train_prior.exists():
        train_prior = out_dir / "train_fit_boundary_oracle_prior.npz"
        infer_boundary_prior(cache_npz=args.train_cache, temporal_npz=args.train_temporal, ckpt_path=args.boundary_oracle_ckpt, output_npz=str(train_prior), device_name=args.device, batch_groups=args.batch_groups)
    if not calib_prior.exists():
        calib_prior = out_dir / "train_calib_boundary_oracle_prior.npz"
        infer_boundary_prior(cache_npz=args.calib_cache, temporal_npz=args.calib_temporal, ckpt_path=args.boundary_oracle_ckpt, output_npz=str(calib_prior), device_name=args.device, batch_groups=args.batch_groups)
    train_cache = load_cache(args.train_cache)
    calib_cache = load_cache(args.calib_cache)
    gt_calib = load_gt_ts(args.calib_gt_jsonl, calib_cache["desc_ids"])
    if train_pool_path.exists():
        train_pool = load_cache(str(train_pool_path))
    else:
        train_pool = build_pool(cache=train_cache, temporal_npz=args.train_temporal, prior_npz=str(train_prior), gt_ts_by_query=None, slots=args.slots, alt_count=args.alt_count, top_endpoint=args.top_endpoint, nms_thd=args.nms_thd)
        save_npz(out_dir / "train_fit_candidate_pool_top_slots.npz", **train_pool)
        train_pool_path = out_dir / "train_fit_candidate_pool_top_slots.npz"
    if calib_pool_path.exists():
        calib_pool = load_cache(str(calib_pool_path))
    else:
        calib_pool = build_pool(cache=calib_cache, temporal_npz=args.calib_temporal, prior_npz=str(calib_prior), gt_ts_by_query=gt_calib, slots=args.slots, alt_count=args.alt_count, top_endpoint=args.top_endpoint, nms_thd=args.nms_thd)
        save_npz(out_dir / "train_calib_candidate_pool_top_slots.npz", **calib_pool)
        calib_pool_path = out_dir / "train_calib_candidate_pool_top_slots.npz"

    model_path = out_dir / "model_best_v2_listwise.pt"
    history_name = f"history_v2_listwise_{args.variant}.json"
    model_path = out_dir / f"model_best_v2_listwise_{args.variant}.pt"
    if model_path.exists() and (out_dir / history_name).exists():
        train_audit = {
            "reused": True,
            "history": json.loads((out_dir / history_name).read_text(encoding="utf-8")),
            "model_artifact": artifact(model_path),
            "train_slot_groups": int(len(train_pool["x"]) // args.alt_count),
            "train_candidate_examples": int(len(train_pool["x"])),
        }
    else:
        # Distillation needs train_fit teacher scores.  For this experiment we
        # generate them by scoring the train pool with the existing v1 model if
        # they are not already present.
        teacher_train = args.train_teacher_score_path
        if args.variant == "distill" and teacher_train is None:
            from rlem_c6_b1_lite.run_c6_b1_r1_safe_selector import score_pool
            v1_model = Path("results/rlem_c6_b1_lite_r1_safe/model_best.pt")
            teacher_train_path = out_dir / "train_fit_v1_teacher_scores.npz"
            if not teacher_train_path.exists():
                sc = score_pool(train_pool, str(v1_model), args.device, max(args.batch_slots * args.alt_count, 32768))
                save_npz(teacher_train_path, score=sc)
            teacher_train = str(teacher_train_path)
        train_audit = train_listwise(
            train_pool, out_dir, args.device, args.epochs, args.batch_slots,
            args.hidden, args.transformer_layers, args.heads,
            variant=args.variant,
            teacher_score_path=teacher_train,
            model_path=model_path,
            history_path=out_dir / history_name,
        )
    calib_scores = score_listwise(calib_pool, str(model_path), args.device, args.batch_slots)
    score_path = out_dir / f"train_calib_candidate_scores_v2_listwise_{args.variant}.npz"
    save_npz(score_path, score=calib_scores)
    search = grid_search_policy(calib_cache, calib_pool, calib_scores, gt_calib, args)
    grid_path = out_dir / f"grid_results_v2_listwise_{args.variant}.json"
    best_path = out_dir / f"best_config_v2_listwise_{args.variant}.json"
    write_json(grid_path, search["grid_results"])
    write_json(best_path, search["best_config"])
    payload = {
        "stage": f"C6-B1-lite-v2 listwise slot selector ({args.variant})",
        "status": search["status"].replace("C6_B1_LITE", f"C6_B1_LITE_V2_LISTWISE_{args.variant.upper()}"),
        "variant": args.variant,
        "official_val_used": False,
        "post_val_adjustment": False,
        "C6_C_used": False,
        "references": {
            "listwise_learning_to_rank": "Listwise softmax/CE over the full alternative set.",
            "proposal_confidence": "Boundary alternatives need proposal confidence rather than blind replacement.",
        },
        "runtime": {
            "training": train_audit,
            "train_pool": artifact(train_pool_path),
            "calib_pool": artifact(calib_pool_path),
        },
        "baseline_pseudo_C4_final": search["baseline"],
        "best_config": search["best_config"],
        "grid_size": len(search["grid_results"]),
        "feasible_count": int(sum(1 for r in search["grid_results"] if r["feasible"])),
        "r1_positive_count": int(sum(1 for r in search["grid_results"] if r["r1_positive"])),
        "r5_safe_count": int(sum(1 for r in search["grid_results"] if r["r5_safe"])),
        "artifacts": {
            "model_best": artifact(model_path),
            "history": artifact(out_dir / history_name),
            "calib_scores": artifact(score_path),
            "grid_results": artifact(grid_path),
            "best_config": artifact(best_path),
        },
    }
    audit_json = audit_dir / f"C6_B1_LITE_V2_LISTWISE_{args.variant.upper()}_AUDIT.json"
    audit_md = audit_dir / f"C6_B1_LITE_V2_LISTWISE_{args.variant.upper()}_AUDIT.md"
    write_json(audit_json, payload)
    md = f"# C6-B1-lite-v2 listwise slot selector audit ({args.variant})\n\n"
    md += f"- Status: `{payload['status']}`\n- Scope: `train_fit/train_calib only`\n- Official val used: `false`\n"
    md += f"- Model: hidden `{args.hidden}`, transformer layers `{args.transformer_layers}`, heads `{args.heads}`.\n"
    md += f"- Grid size: `{payload['grid_size']}`, feasible: `{payload['feasible_count']}`.\n\n"
    md += "## Best config\n\n```json\n" + json.dumps(search["best_config"], ensure_ascii=False, indent=2) + "\n```\n"
    write_text(audit_md, md)
    manifest = {
        "status": payload["status"],
        "stage": f"C6-B1-lite-v2 listwise slot selector ({args.variant})",
        "variant": args.variant,
        "official_val_used": False,
        "official_val_authorized": False,
        "post_val_adjustment": False,
        "C6_C_used": False,
        "R1_primary": True,
        "R5_safety": True,
        "model_scale": {
            "hidden": args.hidden,
            "transformer_layers": args.transformer_layers,
            "heads": args.heads,
            "train_slot_groups": int(len(train_pool["x"]) // args.alt_count),
        },
    }
    manifest_path = audit_dir / f"C6_B1_LITE_V2_LISTWISE_{args.variant.upper()}_MANIFEST.json"
    write_json(manifest_path, manifest)
    hash_targets = [
        "rlem_c6_b1_lite/run_c6_b1_listwise_selector.py",
        audit_dir / "C6_B1_LITE_V2_LISTWISE_START_STATE.md",
        audit_dir / "C6_B1_LITE_V2_LISTWISE_START_STATE.json",
        audit_md,
        audit_json,
        manifest_path,
        model_path,
        out_dir / history_name,
        score_path,
        grid_path,
        best_path,
    ]
    write_json(audit_dir / f"C6_B1_LITE_V2_LISTWISE_{args.variant.upper()}_HASHES.json", {str(p): artifact(p) for p in hash_targets})
    print(json.dumps({"status": payload["status"], "best_config": search["best_config"], "feasible_count": payload["feasible_count"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
