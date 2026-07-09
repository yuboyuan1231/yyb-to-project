from __future__ import annotations

import math
import random
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from blueprint_e2e_v2.data.collate import c28c_collate, c28c_positive_collate
from blueprint_e2e_v2.data.dynamic_candidate_miner import CandidateSet, CompactCandidateStore
from blueprint_e2e_v2.data.feature_registry import FeaturePaths, TMP_ROOT
from blueprint_e2e_v2.data.proposal_dataset import MultiSpanProposalDataset
from blueprint_e2e_v2.data.query_bank import QueryBank
from blueprint_e2e_v2.data.split_manager import SplitManager
from blueprint_e2e_v2.data.subtitle_bank import SubtitleBank
from blueprint_e2e_v2.data.video_bank import VideoBank
from blueprint_e2e_v2.engine.checkpoint import load_checkpoint, save_checkpoint
from blueprint_e2e_v2.engine.refresh_hard_negatives import refresh_candidates
from blueprint_e2e_v2.engine.score_audit import ScoreScaleAccumulator, loss_coupling_audit
from blueprint_e2e_v2.losses.full_loss import compute_full_loss
from blueprint_e2e_v2.losses.retrieval_loss import inbatch_video_retrieval_loss
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
    explicit_cuda_index = device.type == "cuda" and device.index is not None
    if device.type == "cuda" and torch.cuda.device_count() > 1 and bool(cfg.get("use_data_parallel", False)) and not explicit_cuda_index:
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


def slice_batch_to_device(batch: dict[str, Any], start: int, end: int, device: torch.device) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, val in batch.items():
        if torch.is_tensor(val):
            out[key] = val[start:end].to(device, non_blocking=True)
        elif isinstance(val, list):
            out[key] = val[start:end]
        else:
            out[key] = val
    return out


def positive_inbatch_loss(model: torch.nn.Module, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    core = model.module if hasattr(model, "module") else model
    q = core.query_encoder(batch["query_tokens"], batch.get("query_mask"), batch["query_type"])
    enc = core.video_encoder(
        batch["visual"],
        batch["subtitle"],
        visual_mask=batch.get("visual_clip_mask"),
        subtitle_mask=batch.get("subtitle_clip_mask"),
        clip_mask=batch.get("clip_mask"),
    )
    return inbatch_video_retrieval_loss(q, enc, batch["correct_video"], batch.get("video_indices"))


def _rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _restore_rng_state(state: dict[str, Any] | None) -> None:
    if not state:
        return
    if "python" in state:
        random.setstate(state["python"])
    if "numpy" in state:
        np.random.set_state(state["numpy"])
    if "torch" in state:
        torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and state.get("cuda") is not None:
        torch.cuda.set_rng_state_all(state["cuda"])


def _atomic_torch_save(obj: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)


def _checkpoint_epoch(path: Path) -> int | None:
    if not path.exists():
        return None
    try:
        ckpt = torch.load(path, map_location="cpu")
    except Exception:
        return None
    try:
        return int(ckpt.get("epoch", -1))
    except Exception:
        return None


def _score_acc_state(score_acc: ScoreScaleAccumulator) -> dict[str, Any]:
    return {str(k): dict(v) for k, v in getattr(score_acc, "_stats", {}).items()}


def _restore_score_acc(score_acc: ScoreScaleAccumulator, state: dict[str, Any] | None) -> None:
    if state:
        score_acc._stats = {str(k): {str(sk): float(sv) for sk, sv in dict(v).items()} for k, v in state.items()}


def _save_candidate_store(path: Path, candidates: dict[int, CandidateSet] | CompactCandidateStore) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp.npz")
    if isinstance(candidates, CompactCandidateStore):
        query_ids = candidates.query_ids
        video_indices = candidates.video_indices
        scores = candidates.scores
        gt_video_indices = candidates.gt_video_indices
        gt_inserted = candidates.gt_inserted
    else:
        ordered = [candidates[int(qid)] for qid in sorted(candidates)]
        query_ids = np.asarray([c.query_id for c in ordered], dtype=np.int64)
        video_indices = np.stack([np.asarray(c.video_indices, dtype=np.int32) for c in ordered]) if ordered else np.zeros((0, 0), dtype=np.int32)
        scores = np.stack([np.asarray(c.scores, dtype=np.float32) for c in ordered]) if ordered else np.zeros((0, 0), dtype=np.float32)
        gt_video_indices = np.asarray([c.gt_video_index for c in ordered], dtype=np.int32)
        gt_inserted = np.asarray([c.gt_inserted for c in ordered], dtype=np.bool_)
    np.savez(
        tmp,
        query_ids=np.asarray(query_ids, dtype=np.int64),
        video_indices=np.asarray(video_indices, dtype=np.int32),
        scores=np.asarray(scores, dtype=np.float32),
        gt_video_indices=np.asarray(gt_video_indices, dtype=np.int32),
        gt_inserted=np.asarray(gt_inserted, dtype=np.bool_),
    )
    tmp.replace(path)


def _load_candidate_store(path: Path) -> CompactCandidateStore:
    with np.load(path, allow_pickle=False) as arr:
        return CompactCandidateStore(
            arr["query_ids"],
            arr["video_indices"],
            arr["scores"],
            arr["gt_video_indices"],
            arr["gt_inserted"],
        )


def _save_recovery_checkpoint(
    path: Path,
    progress_path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    completed_steps: int,
    completed_cursor: int,
    total_queries: int,
    loss_acc: dict[str, float],
    score_acc: ScoreScaleAccumulator,
    candidate_store_path: Path,
    cand_audit: dict[str, Any],
    train_order: list[int],
    cfg: dict[str, Any],
) -> dict[str, Any]:
    core = model.module if hasattr(model, "module") else model
    rec = {
        "kind": "c28c_step_recovery",
        "model": core.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": int(epoch),
        "completed_steps": int(completed_steps),
        "completed_cursor": int(completed_cursor),
        "total_queries": int(total_queries),
        "loss_acc": {str(k): float(v) for k, v in loss_acc.items()},
        "score_acc_state": _score_acc_state(score_acc),
        "candidate_store_path": str(candidate_store_path),
        "candidate_audit": cand_audit,
        "train_order": np.asarray(train_order, dtype=np.int64),
        "rng_state": _rng_state(),
        "cfg_summary": {
            "batch_size": int(cfg.get("batch_size", 8)),
            "cpu_micro_batch_size": int(cfg.get("cpu_micro_batch_size", cfg.get("loader_micro_batch_size", cfg.get("batch_size", 8)))),
            "gpu_micro_batch_size": int(cfg.get("gpu_micro_batch_size", cfg.get("batch_size", 8))),
            "candidate_topk_train": int(cfg.get("candidate_topk_train", cfg.get("dynamic_topk", 200))),
            "broad_topk_train": int(cfg.get("broad_topk_train", cfg.get("dynamic_topk", 200))),
        },
    }
    _atomic_torch_save(rec, path)
    manifest = {
        "status": "running",
        "recovery_checkpoint_path": str(path),
        "candidate_store_path": str(candidate_store_path),
        "epoch": int(epoch),
        "completed_steps": int(completed_steps),
        "completed_cursor": int(completed_cursor),
        "total_queries": int(total_queries),
        "checkpoint_size_bytes": path.stat().st_size,
    }
    write_json(progress_path, manifest)
    print(
        f"C28C recovery checkpoint epoch {epoch} step {completed_steps}: "
        f"cursor={completed_cursor}/{total_queries} path={path}",
        flush=True,
    )
    return manifest


def prepare_banks(cfg: dict[str, Any], force: bool = False) -> dict[str, Any]:
    paths = FeaturePaths()
    sm = SplitManager(paths)
    qb = QueryBank(paths)
    vb = VideoBank(paths)
    sb = SubtitleBank(paths)
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
    recovery_path = ckpt_dir / f"C28C_FULL_{cfg.get('mode','medium')}_seed{cfg.get('seed',2026)}.recovery.pt"
    recovery_progress_path = TMP_ROOT / "training_logs" / f"C28C_FULL_{cfg.get('mode','medium')}_seed{cfg.get('seed',2026)}.recovery.json"
    recovery_candidate_dir = TMP_ROOT / "candidate_recovery"
    log_path = TMP_ROOT / "training_logs" / f"C28C_FULL_{cfg.get('mode','medium')}_seed{cfg.get('seed',2026)}.training_log.json"
    start_epoch = 0
    resume_loaded = False
    recovery_loaded = False
    recovery_state: dict[str, Any] | None = None
    resume_error: str | None = None
    if resume and ckpt_path.exists() and not dry_run:
        try:
            ckpt = load_checkpoint(ckpt_path, model, optimizer)
            start_epoch = int(ckpt.get("epoch", -1)) + 1
            resume_loaded = True
            print(f"C28C resume: loaded checkpoint {ckpt_path} starting epoch {start_epoch}", flush=True)
        except RuntimeError as exc:
            resume_error = str(exc).splitlines()[0]
            print(f"C28C resume skipped incompatible checkpoint {ckpt_path}: {resume_error}; starting fresh", flush=True)
    if resume and recovery_path.exists() and not dry_run:
        try:
            rec = torch.load(recovery_path, map_location="cpu")
            rec_epoch = int(rec.get("epoch", -1))
            latest_epoch = _checkpoint_epoch(ckpt_path)
            if rec_epoch > (-1 if latest_epoch is None else int(latest_epoch)):
                core = model.module if hasattr(model, "module") else model
                core.load_state_dict(rec["model"])
                optimizer.load_state_dict(rec["optimizer"])
                _restore_rng_state(rec.get("rng_state"))
                start_epoch = rec_epoch
                recovery_loaded = True
                recovery_state = rec
                print(
                    f"C28C recovery resume: loaded {recovery_path} "
                    f"epoch {rec_epoch} completed_step {int(rec.get('completed_steps', 0))}",
                    flush=True,
                )
            else:
                print(
                    f"C28C recovery resume: skipped stale recovery epoch {rec_epoch}; "
                    f"latest epoch checkpoint {latest_epoch}",
                    flush=True,
                )
        except Exception as exc:
            resume_error = str(exc).splitlines()[0]
            print(f"C28C recovery resume skipped incompatible checkpoint {recovery_path}: {resume_error}", flush=True)
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
    existing_log = load_json(log_path, {}) if resume_loaded or recovery_loaded else {}
    train_log: list[dict[str, Any]] = list(existing_log.get("training_log", []))
    best_select: dict[str, Any] | None = existing_log.get("best_select")
    best_manifest: dict[str, Any] | None = existing_log.get("best_checkpoint_manifest")
    best_score = float(existing_log.get("best_select_score", -math.inf))
    latest_manifest: dict[str, Any] | None = existing_log.get("checkpoint_manifest")
    step_checkpoint_every = max(0, int(cfg.get("step_checkpoint_every", cfg.get("checkpoint_every_steps", 100))))
    for epoch in range(start_epoch, int(cfg.get("epochs", 1))):
        teacher_warm = full_candidate_teacher_warm_topk(epoch, cfg, max_candidates)
        use_recovery_epoch = recovery_loaded and recovery_state is not None and epoch == int(recovery_state.get("epoch", -1))
        if use_recovery_epoch:
            candidate_store_path = Path(str(recovery_state["candidate_store_path"]))
            if not candidate_store_path.exists():
                raise FileNotFoundError(f"C28C recovery candidate store missing: {candidate_store_path}")
            print(f"C28C recovery resume epoch {epoch}: loading cached dynamic candidates {candidate_store_path}", flush=True)
            candidates = _load_candidate_store(candidate_store_path)
            cand_audit = dict(recovery_state.get("candidate_audit", {}))
        else:
            print(f"C28C training epoch {epoch}: refreshing dynamic candidates", flush=True)
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
                clip_bank_encode_chunk=int(cfg.get("clip_bank_encode_chunk", 128)),
                late_soft_topk=int(cfg.get("late_soft_topk", 8)),
                late_temperature=float(cfg.get("late_temperature", 0.07)),
                token_maxsim_weight=float(cfg.get("token_maxsim_weight", 0.0)),
                pooled_score_weight=float(cfg.get("pooled_score_weight", 1.0)),
                late_score_weight=float(cfg.get("late_score_weight", 0.0)),
            )
            candidate_store_path = recovery_candidate_dir / (
                f"C28C_FULL_{cfg.get('mode','medium')}_seed{cfg.get('seed',2026)}_epoch{epoch}.candidates.npz"
            )
            _save_candidate_store(candidate_store_path, candidates)
            print(f"C28C recovery candidate store epoch {epoch}: saved {candidate_store_path}", flush=True)
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
        batch_size_cfg = max(1, int(cfg.get("batch_size", 8)))
        cpu_micro_batch = max(1, int(cfg.get("cpu_micro_batch_size", cfg.get("loader_micro_batch_size", batch_size_cfg))))
        loader_batch_size = min(batch_size_cfg, cpu_micro_batch)
        stream_effective_batch = loader_batch_size < batch_size_cfg
        if use_recovery_epoch:
            train_order = [int(x) for x in recovery_state.get("train_order", [])]
            if len(train_order) != len(dataset):
                raise RuntimeError(f"C28C recovery train_order length mismatch: {len(train_order)} != {len(dataset)}")
            completed_cursor = min(int(recovery_state.get("completed_cursor", 0)), len(dataset))
            remaining_indices = train_order[completed_cursor:]
        else:
            order_gen = torch.Generator()
            order_gen.manual_seed(int(cfg.get("seed", 2026)) + epoch * 1_000_003)
            train_order = [int(x) for x in torch.randperm(len(dataset), generator=order_gen).tolist()]
            completed_cursor = 0
            remaining_indices = train_order
        loader = DataLoader(
            Subset(dataset, remaining_indices),
            batch_size=loader_batch_size,
            shuffle=False,
            collate_fn=c28c_collate,
            num_workers=0,
            pin_memory=device.type == "cuda" and not stream_effective_batch,
        )
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss_acc: dict[str, float] = (
            {str(k): float(v) for k, v in dict(recovery_state.get("loss_acc", {})).items()}
            if use_recovery_epoch
            else {}
        )
        score_acc = ScoreScaleAccumulator(cfg)
        if use_recovery_epoch:
            _restore_score_acc(score_acc, recovery_state.get("score_acc_state"))
        steps = int(recovery_state.get("completed_steps", 0)) if use_recovery_epoch else 0
        grad_accum = int(cfg.get("grad_accum_steps", 1))
        gpu_micro_batch = max(1, int(cfg.get("gpu_micro_batch_size", batch_size_cfg)))
        if step_checkpoint_every > 0 and not use_recovery_epoch:
            _save_recovery_checkpoint(
                recovery_path,
                recovery_progress_path,
                model,
                optimizer,
                epoch,
                steps,
                completed_cursor,
                len(dataset),
                loss_acc,
                score_acc,
                candidate_store_path,
                cand_audit,
                train_order,
                cfg,
            )
        if stream_effective_batch:
            no_inbatch_cfg = dict(cfg)
            no_inbatch_cfg["skip_inbatch_retrieval"] = True
            no_inbatch_cfg["lambda_inbatch"] = 0.0
            processed = completed_cursor
            virtual_qids: list[int] = []
            virtual_metrics: dict[str, float] = {}
            virtual_target = 0
            for batch in loader:
                if not virtual_qids:
                    virtual_target = min(batch_size_cfg, len(dataset) - processed)
                    virtual_target = max(1, int(virtual_target))
                batch_n = len(batch["query_ids"])
                for micro_start in range(0, batch_n, gpu_micro_batch):
                    micro_end = min(micro_start + gpu_micro_batch, batch_n)
                    tensor_batch = slice_batch_to_device(batch, micro_start, micro_end, device)
                    out = model(tensor_batch)
                    score_acc.update(out)
                    loss, metrics = compute_full_loss(out, tensor_batch, no_inbatch_cfg)
                    micro_weight = (micro_end - micro_start) / max(1, virtual_target)
                    (loss * micro_weight / grad_accum).backward()
                    for k, v in metrics.items():
                        virtual_metrics[k] = virtual_metrics.get(k, 0.0) + float(v) * micro_weight
                virtual_qids.extend(int(q) for q in batch["query_ids"])
                processed += batch_n
                if len(virtual_qids) >= virtual_target:
                    pos_items = [dataset.positive_retrieval_item(dataset.qid_to_index[int(q)]) for q in virtual_qids]
                    pos_batch = c28c_positive_collate(pos_items)
                    pos_batch = slice_batch_to_device(pos_batch, 0, len(virtual_qids), device)
                    inbatch_loss = positive_inbatch_loss(model, pos_batch)
                    inbatch_value = float(inbatch_loss.detach().cpu().item())
                    lambda_inbatch = float(cfg.get("lambda_inbatch", 0.3))
                    (lambda_inbatch * inbatch_loss / grad_accum).backward()
                    virtual_metrics["L_inbatch_retrieval"] = inbatch_value
                    virtual_metrics["L_total"] = virtual_metrics.get("L_total", 0.0) + lambda_inbatch * inbatch_value
                    if (steps + 1) % grad_accum == 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                        optimizer.step()
                        optimizer.zero_grad(set_to_none=True)
                    for k, v in virtual_metrics.items():
                        loss_acc[k] = loss_acc.get(k, 0.0) + float(v)
                    steps += 1
                    if virtual_metrics.get("L_total", 0.0) > float(cfg.get("loss_spike_log_threshold", 200.0)):
                        top_parts = sorted(((k, v) for k, v in virtual_metrics.items() if k != "L_total"), key=lambda x: abs(x[1]), reverse=True)[:5]
                        print(
                            f"C28C loss spike diagnostic epoch {epoch} step {steps}: "
                            f"L_total={virtual_metrics.get('L_total'):.4f} top_parts={top_parts} "
                            f"query_ids={virtual_qids[:8]}...",
                            flush=True,
                        )
                    if steps % 100 == 0:
                        print(f"C28C training epoch {epoch} step {steps}: loss={virtual_metrics.get('L_total'):.4f}", flush=True)
                    if step_checkpoint_every > 0 and (steps % step_checkpoint_every == 0 or processed >= len(dataset)):
                        _save_recovery_checkpoint(
                            recovery_path,
                            recovery_progress_path,
                            model,
                            optimizer,
                            epoch,
                            steps,
                            processed,
                            len(dataset),
                            loss_acc,
                            score_acc,
                            candidate_store_path,
                            cand_audit,
                            train_order,
                            cfg,
                        )
                    virtual_qids = []
                    virtual_metrics = {}
                    virtual_target = 0
            if steps % grad_accum != 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        else:
            processed = completed_cursor
            for batch in loader:
                batch_n = len(batch["query_ids"])
                batch_metrics: dict[str, float] = {}
                for micro_start in range(0, batch_n, gpu_micro_batch):
                    micro_end = min(micro_start + gpu_micro_batch, batch_n)
                    tensor_batch = slice_batch_to_device(batch, micro_start, micro_end, device)
                    out = model(tensor_batch)
                    score_acc.update(out)
                    loss, metrics = compute_full_loss(out, tensor_batch, cfg)
                    micro_weight = (micro_end - micro_start) / max(1, batch_n)
                    (loss * micro_weight / grad_accum).backward()
                    for k, v in metrics.items():
                        batch_metrics[k] = batch_metrics.get(k, 0.0) + float(v) * micro_weight
                if (step + 1) % grad_accum == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                for k, v in batch_metrics.items():
                    loss_acc[k] = loss_acc.get(k, 0.0) + float(v)
                steps += 1
                processed += batch_n
                if batch_metrics.get("L_total", 0.0) > float(cfg.get("loss_spike_log_threshold", 200.0)):
                    top_parts = sorted(((k, v) for k, v in batch_metrics.items() if k != "L_total"), key=lambda x: abs(x[1]), reverse=True)[:5]
                    pos_counts = batch["correct_video"].sum(dim=1).detach().cpu().tolist()
                    ge07_counts = batch["span_ge07"].sum(dim=(1, 2)).detach().cpu().tolist()
                    print(
                        f"C28C loss spike diagnostic epoch {epoch} step {steps}: "
                        f"L_total={batch_metrics.get('L_total'):.4f} top_parts={top_parts} "
                        f"query_ids={batch['query_ids']} pos_counts={pos_counts} ge07_counts={ge07_counts}",
                        flush=True,
                    )
                if steps % 100 == 0:
                    print(f"C28C training epoch {epoch} step {steps}: loss={batch_metrics.get('L_total'):.4f}", flush=True)
                if step_checkpoint_every > 0 and (steps % step_checkpoint_every == 0 or processed >= len(dataset)):
                    _save_recovery_checkpoint(
                        recovery_path,
                        recovery_progress_path,
                        model,
                        optimizer,
                        epoch,
                        steps,
                        processed,
                        len(dataset),
                        loss_acc,
                        score_acc,
                        candidate_store_path,
                        cand_audit,
                        train_order,
                        cfg,
                    )
        avg = {k: v / max(1, steps) for k, v in loss_acc.items()}
        score_scale = score_acc.summary()
        print(f"C28C training epoch {epoch} complete: avg_loss={avg.get('L_total'):.4f}", flush=True)
        latest_manifest = save_checkpoint(ckpt_path, model, optimizer, epoch, {"train_loss": avg})
        write_json(recovery_progress_path, {
            "status": "epoch_checkpoint_saved",
            "epoch": epoch,
            "completed_steps": steps,
            "completed_cursor": len(dataset),
            "total_queries": len(dataset),
            "checkpoint_manifest": latest_manifest,
            "recovery_checkpoint_path": str(recovery_path),
            "candidate_store_path": str(candidate_store_path),
        })
        epoch_rec: dict[str, Any] = {
            "epoch": epoch,
            "loss": avg,
            "score_scale_audit_train": score_scale,
            "loss_coupling_audit": loss_coupling_audit(avg, score_scale),
            "candidate_audit": cand_audit,
            "candidate_curriculum": {
                "teacher_warm_topk": teacher_warm,
                "student_dynamic_late_mining": bool(cand_audit.get("late_candidate_mining_used", False)),
                "candidate_topk_train": max_candidates,
                "cpu_micro_batch_size": loader_batch_size,
                "gpu_micro_batch_size": gpu_micro_batch,
                "stream_effective_batch": bool(stream_effective_batch),
                "effective_batch_size": batch_size_cfg * grad_accum,
                "effective_inbatch_retrieval_batch_size": batch_size_cfg,
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
            epoch_rec["select_score_scale_audit"] = select_rec.get("score_scale_audit", {})
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
                    "score_scale_audit": select_rec.get("score_scale_audit", {}),
                }
        train_log.append(epoch_rec)
        write_json(log_path, {
            "status": "running",
            "resume_loaded": resume_loaded,
            "recovery_loaded": recovery_loaded,
            "resume_error": resume_error,
            "training_log": train_log,
            "checkpoint_manifest": latest_manifest,
            "recovery_checkpoint_path": str(recovery_path),
            "recovery_progress_path": str(recovery_progress_path),
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
        "recovery_checkpoint_path": str(recovery_path),
        "recovery_progress_path": str(recovery_progress_path),
        "training_log_path": str(log_path),
        "resume_loaded": resume_loaded,
        "recovery_loaded": recovery_loaded,
        "resume_error": resume_error,
        "visual_bank_manifest": banks["visual_manifest"],
        "subtitle_bank_manifest": banks["subtitle_manifest"],
        **seq_manifests,
        "best_select": best_select,
    }
