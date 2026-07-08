from __future__ import annotations

import math
import shutil
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from blueprint_e2e_v2.data.collate import c28c_collate
from blueprint_e2e_v2.data.feature_registry import FeaturePaths, TMP_ROOT
from blueprint_e2e_v2.data.proposal_dataset import MultiSpanProposalDataset
from blueprint_e2e_v2.data.query_bank import QueryBank
from blueprint_e2e_v2.data.split_manager import SplitManager
from blueprint_e2e_v2.data.subtitle_bank import SubtitleBank
from blueprint_e2e_v2.data.video_bank import VideoBank
from blueprint_e2e_v2.engine.checkpoint import load_checkpoint, save_checkpoint
from blueprint_e2e_v2.engine.refresh_hard_negatives import refresh_candidates
from blueprint_e2e_v2.losses.full_loss import compute_full_loss
from blueprint_e2e_v2.models.full_model import C28CFullModel
from blueprint_e2e_v2.utils.io import load_json, write_json
from blueprint_e2e_v2.utils.seed import seed_all


def device_from_arg(device_arg: str | None = None) -> torch.device:
    if device_arg:
        return torch.device(device_arg)
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def build_model(cfg: dict[str, Any], device: torch.device) -> torch.nn.Module:
    model = C28CFullModel(
        query_dim=int(cfg.get("query_dim", 768)),
        subtitle_dim=int(cfg.get("subtitle_dim", 768)),
        visual_dim=int(cfg.get("visual_dim", 4352)),
        hidden_dim=int(cfg.get("hidden_dim", 384)),
        late_interaction_enabled=bool(cfg.get("late_interaction_enabled", False)),
        late_soft_topk=int(cfg.get("late_soft_topk", 8)),
        late_temperature=float(cfg.get("late_temperature", 0.07)),
        token_maxsim_weight=float(cfg.get("token_maxsim_weight", 0.0)),
        pooled_score_weight=float(cfg.get("pooled_score_weight", 1.0)),
        late_score_weight=float(cfg.get("late_score_weight", 0.0)),
    ).to(device)
    if device.type == "cuda" and torch.cuda.device_count() > 1 and bool(cfg.get("use_data_parallel", False)):
        model = torch.nn.DataParallel(model)
    return model


def full_selection_score(summary: dict[str, Any], cfg: dict[str, Any]) -> float:
    return (
        float(cfg.get("full_select_weight_vcmr_r1_iou07", 4.0)) * float(summary.get("VCMR_R@1_IoU0.7", 0.0))
        + float(cfg.get("full_select_weight_vcmr_r5_iou07", 3.0)) * float(summary.get("VCMR_R@5_IoU0.7", 0.0))
        + float(cfg.get("full_select_weight_vcmr_r10_iou07", 2.0)) * float(summary.get("VCMR_R@10_IoU0.7", 0.0))
        + float(cfg.get("full_select_weight_vr_r100", 1.0)) * float(summary.get("VR_R@100", 0.0))
        - float(cfg.get("full_select_weight_wrong_video_top1", 0.5)) * float(summary.get("wrong_video_top1_rate", 0.0))
    )


def full_candidate_teacher_warm_topk(epoch: int, cfg: dict[str, Any], candidate_topk: int) -> int:
    if not bool(cfg.get("use_candidate_curriculum", True)):
        return min(int(cfg.get("teacher_warm_topk", cfg.get("teacher_anchor_topk", 64))), int(candidate_topk))
    if epoch <= 2:
        return min(int(cfg.get("teacher_curriculum_phase0_topk", 120)), int(candidate_topk))
    if epoch <= 5:
        return min(int(cfg.get("teacher_curriculum_phase1_topk", 80)), int(candidate_topk))
    if epoch <= 8:
        return min(int(cfg.get("teacher_curriculum_phase2_topk", 40)), int(candidate_topk))
    return min(int(cfg.get("teacher_curriculum_phase3_topk", 0)), int(candidate_topk))


def prepare_banks(cfg: dict[str, Any], force: bool = False) -> dict[str, Any]:
    paths = FeaturePaths()
    sm = SplitManager(paths)
    qb = QueryBank(paths)
    vb = VideoBank(paths)
    sb = SubtitleBank(paths)
    vb.enable_sequence_cache()
    sb.enable_sequence_cache()
    max_videos = int(cfg.get("max_videos", 0) or 0)
    visual, visual_manifest = vb.build_or_load_pooled(max_videos=max_videos or None, force=force)
    subtitle, subtitle_manifest = sb.build_or_load_pooled([str(x) for x in visual["video_ids"].tolist()], max_videos=max_videos or None, force=force)
    return {
        "paths": paths,
        "split_manager": sm,
        "query_bank": qb,
        "video_bank": vb,
        "subtitle_bank": sb,
        "visual_np": visual,
        "subtitle_np": subtitle,
        "visual_manifest": visual_manifest,
        "subtitle_manifest": subtitle_manifest,
    }


def run_full_training(cfg: dict[str, Any], dry_run: bool = False, force: bool = False, resume: bool = False, device_arg: str | None = None) -> dict[str, Any]:
    seed_all(int(cfg.get("seed", 2026)))
    device = device_from_arg(device_arg)
    work_cfg = dict(cfg)
    if dry_run and not int(work_cfg.get("max_videos", 0) or 0):
        work_cfg["max_videos"] = 128
    banks = prepare_banks(work_cfg, force=force)
    video_ids = [str(x) for x in banks["visual_np"]["video_ids"].tolist()]
    banks["video_bank"].video_ids = video_ids
    banks["video_bank"].video_to_idx = {v: i for i, v in enumerate(video_ids)}
    banks["video_bank"].idx_to_video = {i: v for v, i in banks["video_bank"].video_to_idx.items()}
    visual_bank = torch.from_numpy(banks["visual_np"]["visual_mean"].astype(np.float32))
    subtitle_bank = torch.from_numpy(banks["subtitle_np"]["subtitle_mean"].astype(np.float32))
    visual_seq_bank = None
    subtitle_seq_bank = None
    visual_seq_mask = None
    subtitle_seq_mask = None
    seq_manifests: dict[str, Any] = {}
    target_len = int(work_cfg.get("target_len", 64))
    if bool(work_cfg.get("preload_sequence_bank", True)):
        visual_seq_bank, visual_seq_manifest = banks["video_bank"].build_or_load_sequence_bank(
            video_ids,
            target_len=target_len,
            max_videos=int(work_cfg.get("max_videos", 0) or 0) or None,
            force=force,
        )
        subtitle_seq_bank, subtitle_seq_manifest = banks["subtitle_bank"].build_or_load_sequence_bank(
            video_ids,
            target_len=target_len,
            max_videos=int(work_cfg.get("max_videos", 0) or 0) or None,
            force=force,
        )
        visual_seq_mask, visual_mask_manifest = banks["video_bank"].build_or_load_sequence_mask(
            video_ids,
            target_len=target_len,
            max_videos=int(work_cfg.get("max_videos", 0) or 0) or None,
            force=force,
        )
        subtitle_seq_mask, subtitle_mask_manifest = banks["subtitle_bank"].build_or_load_sequence_mask(
            video_ids,
            target_len=target_len,
            max_videos=int(work_cfg.get("max_videos", 0) or 0) or None,
            force=force,
        )
        banks["video_bank"].clear_sequence_cache()
        banks["subtitle_bank"].clear_sequence_cache()
        seq_manifests = {
            "visual_sequence_manifest": visual_seq_manifest,
            "subtitle_sequence_manifest": subtitle_seq_manifest,
            "visual_sequence_mask_manifest": visual_mask_manifest,
            "subtitle_sequence_mask_manifest": subtitle_mask_manifest,
        }
    model = build_model(cfg, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg.get("lr", 1.5e-4)), weight_decay=float(cfg.get("weight_decay", 0.01)))
    ckpt_dir = TMP_ROOT / "checkpoints"
    ckpt_path = ckpt_dir / f"C28C_FULL_{cfg.get('mode','medium')}_seed{cfg.get('seed',2026)}.pt"
    best_ckpt_path = ckpt_dir / f"C28C_FULL_{cfg.get('mode','medium')}_seed{cfg.get('seed',2026)}.best.pt"
    log_path = TMP_ROOT / "training_logs" / f"C28C_FULL_{cfg.get('mode','medium')}_seed{cfg.get('seed',2026)}.training_log.json"
    start_epoch = 0
    if resume and ckpt_path.exists() and not dry_run:
        ckpt = load_checkpoint(ckpt_path, model, optimizer)
        start_epoch = int(ckpt.get("epoch", -1)) + 1
        print(f"C28C resume: loaded checkpoint {ckpt_path} starting epoch {start_epoch}", flush=True)
    if dry_run:
        return {
            "status": "C28C_FULL_MODEL_DRY_RUN_READY",
            "device": str(device),
            "data_parallel": bool(hasattr(model, "module")),
            "visual_bank_shape": list(visual_bank.shape),
            "subtitle_bank_shape": list(subtitle_bank.shape),
            "preload_sequence_bank": bool(work_cfg.get("preload_sequence_bank", True)),
            **seq_manifests,
            "checkpoint_path": str(ckpt_path),
        }
    max_queries = int(cfg.get("max_queries", 0) or 0) or None
    max_candidates = int(cfg.get("candidate_topk_train", cfg.get("dynamic_topk", 200)))
    existing_log = load_json(log_path, {}) if resume else {}
    train_log: list[dict[str, Any]] = list(existing_log.get("training_log", []))
    best_select: dict[str, Any] | None = existing_log.get("best_select")
    best_manifest: dict[str, Any] | None = existing_log.get("best_checkpoint_manifest")
    best_score = float(existing_log.get("best_select_score", -math.inf))
    latest_manifest: dict[str, Any] | None = existing_log.get("checkpoint_manifest")
    for epoch in range(start_epoch, int(cfg.get("epochs", 1))):
        print(f"C28C training epoch {epoch}: refreshing dynamic candidates", flush=True)
        teacher_warm = full_candidate_teacher_warm_topk(epoch, cfg, max_candidates)
        candidates, cand_audit = refresh_candidates(
            model,
            banks["split_manager"],
            banks["query_bank"],
            video_ids,
            banks["video_bank"].video_to_idx,
            visual_bank,
            subtitle_bank,
            "train_fit",
            max_queries=max_queries,
            dynamic_topk=max_candidates,
            chunk_size=int(cfg.get("chunk_size", 256)),
            device=device,
            insert_gt_for_training=True,
            teacher_warm_topk=teacher_warm,
            visual_seq_bank=visual_seq_bank,
            subtitle_seq_bank=subtitle_seq_bank,
            visual_seq_mask=visual_seq_mask,
            subtitle_seq_mask=subtitle_seq_mask,
            late_candidate_mining=bool(cfg.get("late_candidate_mining", cfg.get("late_interaction_enabled", False))),
            broad_topk=int(cfg.get("broad_topk_train", cfg.get("dynamic_topk", max_candidates))),
            candidate_encode_chunk=int(cfg.get("candidate_encode_chunk", 32)),
            late_soft_topk=int(cfg.get("late_soft_topk", 8)),
            late_temperature=float(cfg.get("late_temperature", 0.07)),
            token_maxsim_weight=float(cfg.get("token_maxsim_weight", 0.0)),
            pooled_score_weight=float(cfg.get("pooled_score_weight", 1.0)),
            late_score_weight=float(cfg.get("late_score_weight", 0.0)),
        )
        dataset = MultiSpanProposalDataset(
            "train_fit",
            banks["split_manager"],
            banks["query_bank"],
            banks["video_bank"],
            banks["subtitle_bank"],
            candidates,
            max_queries=max_queries,
            max_candidates=max_candidates,
            max_spans_per_video=int(cfg.get("max_spans_per_video", 64)),
            insert_gt_for_training=True,
            visual_seq_bank=visual_seq_bank,
            subtitle_seq_bank=subtitle_seq_bank,
            visual_seq_mask=visual_seq_mask,
            subtitle_seq_mask=subtitle_seq_mask,
            target_len=target_len,
        )
        loader = DataLoader(dataset, batch_size=int(cfg.get("batch_size", 8)), shuffle=True, collate_fn=c28c_collate, num_workers=0, pin_memory=device.type == "cuda")
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss_acc: dict[str, float] = {}
        steps = 0
        grad_accum = int(cfg.get("grad_accum_steps", 1))
        for step, batch in enumerate(loader):
            tensor_batch = {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}
            out = model(tensor_batch)
            loss, metrics = compute_full_loss(out, tensor_batch, cfg)
            (loss / grad_accum).backward()
            if (step + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            for k, v in metrics.items():
                loss_acc[k] = loss_acc.get(k, 0.0) + float(v)
            steps += 1
            if metrics.get("L_total", 0.0) > float(cfg.get("loss_spike_log_threshold", 200.0)):
                top_parts = sorted(((k, v) for k, v in metrics.items() if k != "L_total"), key=lambda x: abs(x[1]), reverse=True)[:5]
                pos_counts = tensor_batch["correct_video"].sum(dim=1).detach().cpu().tolist()
                ge07_counts = tensor_batch["span_ge07"].sum(dim=(1, 2)).detach().cpu().tolist()
                print(
                    f"C28C loss spike diagnostic epoch {epoch} step {step + 1}: "
                    f"L_total={metrics.get('L_total'):.4f} top_parts={top_parts} "
                    f"query_ids={batch['query_ids']} pos_counts={pos_counts} ge07_counts={ge07_counts}",
                    flush=True,
                )
            if (step + 1) % 100 == 0:
                print(f"C28C training epoch {epoch} step {step + 1}: loss={metrics.get('L_total'):.4f}", flush=True)
        avg = {k: v / max(1, steps) for k, v in loss_acc.items()}
        print(f"C28C training epoch {epoch} complete: avg_loss={avg.get('L_total'):.4f}", flush=True)
        latest_manifest = save_checkpoint(ckpt_path, model, optimizer, epoch, {"train_loss": avg})
        epoch_rec: dict[str, Any] = {
            "epoch": epoch,
            "loss": avg,
            "candidate_audit": cand_audit,
            "candidate_curriculum": {
                "teacher_warm_topk": teacher_warm,
                "student_dynamic_late_mining": bool(cand_audit.get("late_candidate_mining_used", False)),
                "candidate_topk_train": max_candidates,
                "phase": 0 if epoch <= 2 else 1 if epoch <= 5 else 2 if epoch <= 8 else 3,
            },
            "checkpoint_manifest": latest_manifest,
        }
        select_every = int(cfg.get("full_select_every", 1))
        if select_every > 0 and ((epoch + 1) % select_every == 0 or epoch + 1 == int(cfg.get("epochs", 1))):
            from blueprint_e2e_v2.engine.evaluate import run_evaluation

            eval_cfg = dict(cfg)
            eval_cfg["checkpoint_path"] = str(ckpt_path)
            select_rec = run_evaluation(eval_cfg, split=str(cfg.get("selection_split", "calib_select")), device_arg=device_arg, force=False)
            select_summary = select_rec.get("summary", {}).get("summary", {})
            select_score = full_selection_score(select_summary, cfg)
            epoch_rec["select_score"] = select_score
            epoch_rec["select_summary"] = select_summary
            if select_score > best_score:
                best_score = select_score
                shutil.copy2(ckpt_path, best_ckpt_path)
                best_manifest = {
                    "path": str(best_ckpt_path),
                    "source_checkpoint": str(ckpt_path),
                    "epoch": epoch,
                    "select_score": best_score,
                    "select_summary": select_summary,
                }
                best_select = {
                    "split": str(cfg.get("selection_split", "calib_select")),
                    "epoch": epoch,
                    "score": best_score,
                    "summary": select_summary,
                    "candidate_audit": select_rec.get("candidate_audit", {}),
                }
        train_log.append(epoch_rec)
        write_json(log_path, {
            "status": "running",
            "training_log": train_log,
            "checkpoint_manifest": latest_manifest,
            "best_select_score": best_score,
            "best_select": best_select,
            "best_checkpoint_manifest": best_manifest,
        })
    return {
        "status": "C28C_FULL_MODEL_TRAINED",
        "device": str(device),
        "data_parallel": bool(hasattr(model, "module")),
        "training_log": train_log,
        "checkpoint_manifest": latest_manifest,
        "best_checkpoint_manifest": best_manifest,
        "training_log_path": str(log_path),
        "visual_bank_manifest": banks["visual_manifest"],
        "subtitle_bank_manifest": banks["subtitle_manifest"],
        **seq_manifests,
        "best_select": best_select,
    }
