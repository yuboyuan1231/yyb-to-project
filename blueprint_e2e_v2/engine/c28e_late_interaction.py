from __future__ import annotations

import json
import math
import random
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from blueprint_e2e_v2.data.feature_registry import TMP_ROOT
from blueprint_e2e_v2.data.first_stage_reference import FirstStageReference, TeacherRankItem
from blueprint_e2e_v2.engine.checkpoint import load_checkpoint, save_checkpoint
from blueprint_e2e_v2.engine.official_safe_eval_wrapper import official_safety_manifest
from blueprint_e2e_v2.engine.train import build_model, device_from_arg, prepare_banks
from blueprint_e2e_v2.utils.hashing import stable_hash
from blueprint_e2e_v2.utils.io import load_json, write_json, write_text
from blueprint_e2e_v2.utils.seed import seed_all


def _pad_query_tokens(rows: list[dict[str, Any]], query_cache: dict[int, np.ndarray]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[int], list[str]]:
    arrays = [query_cache[int(r["desc_id"])] for r in rows]
    max_len = max(a.shape[0] for a in arrays)
    dim = arrays[0].shape[1]
    tokens = np.zeros((len(arrays), max_len, dim), dtype=np.float32)
    mask = np.zeros((len(arrays), max_len), dtype=np.bool_)
    qtypes = np.zeros((len(arrays),), dtype=np.int64)
    for i, (a, row) in enumerate(zip(arrays, rows)):
        tokens[i, : a.shape[0]] = a
        mask[i, : a.shape[0]] = True
        qtypes[i] = {"v": 0, "t": 1, "vt": 2}.get(str(row.get("type", "")), 3)
    return torch.from_numpy(tokens), torch.from_numpy(mask), torch.from_numpy(qtypes), [int(r["desc_id"]) for r in rows], [str(r["vid_name"]) for r in rows]


def _rank(indices: list[int], gt_idx: int) -> int | None:
    try:
        return indices.index(int(gt_idx)) + 1
    except ValueError:
        return None


def _rank_metrics(ranks: list[int | None], ks: tuple[int, ...] = (1, 5, 10, 50, 100, 200, 500)) -> dict[str, Any]:
    valid = [int(x) for x in ranks if x is not None]
    out: dict[str, Any] = {"query_count": len(ranks), "missing_gt_count": len(ranks) - len(valid)}
    for k in ks:
        out[f"VR@{k}"] = 100.0 * sum(x is not None and int(x) <= k for x in ranks) / max(1, len(ranks))
    out["gt_rank_mean"] = float(np.mean(valid)) if valid else None
    out["gt_rank_median"] = float(np.median(valid)) if valid else None
    out["gt_rank_p95"] = float(np.percentile(valid, 95)) if valid else None
    out["wrong_video_top1_rate"] = 100.0 * sum(x is None or int(x) != 1 for x in ranks) / max(1, len(ranks))
    return out


def _selection_score(select_metrics: dict[str, Any], train_metrics: dict[str, Any], cfg: dict[str, Any]) -> float:
    score = (
        float(cfg.get("select_weight_vr10", 3.0)) * float(select_metrics.get("VR@10", 0.0))
        + float(cfg.get("select_weight_vr50", 2.0)) * float(select_metrics.get("VR@50", 0.0))
        + float(cfg.get("select_weight_vr100", 1.5)) * float(select_metrics.get("VR@100", 0.0))
        + float(cfg.get("select_weight_vr200", 1.0)) * float(select_metrics.get("VR@200", 0.0))
    )
    gap = max(0.0, float(train_metrics.get("VR@100", 0.0)) - float(select_metrics.get("VR@100", 0.0)))
    return score - float(cfg.get("select_overfit_penalty", 0.5)) * gap


def _safe_soft_topk(values: torch.Tensor, k: int, temperature: float) -> torch.Tensor:
    k = min(int(k), values.shape[-1])
    vals = torch.topk(values, k=k, dim=-1).values
    weights = torch.softmax(vals / max(float(temperature), 1e-6), dim=-1)
    return (weights * vals).sum(dim=-1)


def _teacher_logits(candidates: list[int], teacher_items: list[TeacherRankItem], gt_idx: int, temperature: float) -> torch.Tensor:
    score_by_idx = {int(x.video_index): x.score for x in teacher_items if x.score is not None}
    rank_by_idx = {int(x.video_index): int(x.rank) for x in teacher_items}
    logits = torch.full((len(candidates),), -8.0, dtype=torch.float32)
    if score_by_idx:
        vals = [float(v) for v in score_by_idx.values()]
        lo = min(vals)
        hi = max(vals)
        denom = max(hi - lo, 1e-6)
        for i, idx in enumerate(candidates):
            if int(idx) in score_by_idx:
                logits[i] = (float(score_by_idx[int(idx)]) - lo) / denom / max(float(temperature), 1e-6)
    else:
        for i, idx in enumerate(candidates):
            rank = rank_by_idx.get(int(idx))
            if rank is not None:
                logits[i] = float(max(0, len(teacher_items) + 1 - rank)) / max(float(temperature), 1e-6)
    if gt_idx in candidates:
        logits[candidates.index(gt_idx)] = torch.maximum(logits[candidates.index(gt_idx)], logits.max().clamp_min(0.0) + 1.0)
    return logits


def _merge_candidates(gt_idx: int, teacher: list[int], student: list[int], randoms: list[int], limit: int) -> list[int]:
    out = [int(gt_idx)]
    for source in (teacher, student, randoms):
        for idx in source:
            idx = int(idx)
            if idx not in out:
                out.append(idx)
            if len(out) >= int(limit):
                return out
    return out


def _model_core(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if hasattr(model, "module") else model


@torch.no_grad()
def encode_pooled_bank(model: torch.nn.Module, visual_bank: torch.Tensor, subtitle_bank: torch.Tensor, chunk_size: int, device: torch.device) -> dict[str, torch.Tensor]:
    core = _model_core(model)
    core.eval()
    parts: dict[str, list[torch.Tensor]] = {"visual_pool": [], "subtitle_pool": [], "joint_pool": []}
    for st in range(0, visual_bank.shape[0], int(chunk_size)):
        enc = core.video_encoder.encode_pooled_bank(
            visual_bank[st: st + int(chunk_size)].to(device, non_blocking=True),
            subtitle_bank[st: st + int(chunk_size)].to(device, non_blocking=True),
        )
        for key in parts:
            parts[key].append(enc[key].detach().cpu())
    return {key: torch.cat(vals, dim=0) for key, vals in parts.items()}


@torch.no_grad()
def encode_clip_bank(
    model: torch.nn.Module,
    visual_seq_bank: np.ndarray,
    subtitle_seq_bank: np.ndarray,
    chunk_size: int,
    device: torch.device,
    dtype: torch.dtype = torch.float16,
) -> dict[str, torch.Tensor]:
    core = _model_core(model)
    core.eval()
    parts: dict[str, list[torch.Tensor]] = {"visual": [], "subtitle": [], "joint": []}
    for st in range(0, visual_seq_bank.shape[0], int(chunk_size)):
        visual = torch.from_numpy(visual_seq_bank[st: st + int(chunk_size)].astype(np.float32, copy=False)).to(device, non_blocking=True).unsqueeze(1)
        subtitle = torch.from_numpy(subtitle_seq_bank[st: st + int(chunk_size)].astype(np.float32, copy=False)).to(device, non_blocking=True).unsqueeze(1)
        enc = core.video_encoder(visual, subtitle)
        for key in parts:
            parts[key].append(enc[key].squeeze(1).detach().cpu().to(dtype=dtype))
    return {key: torch.cat(vals, dim=0) for key, vals in parts.items()}


def pooled_scores(core: torch.nn.Module, q: dict[str, torch.Tensor], pooled_bank: dict[str, torch.Tensor], device: torch.device, chunk_size: int) -> torch.Tensor:
    parts = []
    for st in range(0, pooled_bank["visual_pool"].shape[0], int(chunk_size)):
        bank = {key: val[st: st + int(chunk_size)].to(device, non_blocking=True) for key, val in pooled_bank.items()}
        parts.append(core.retriever.score_bank(q, bank).detach())
    return torch.cat(parts, dim=1)


def late_scores_for_candidates(
    core: torch.nn.Module,
    q: dict[str, torch.Tensor],
    query_tokens: torch.Tensor,
    query_mask: torch.Tensor,
    candidate_indices: torch.Tensor,
    pooled_bank: dict[str, torch.Tensor],
    clip_bank: dict[str, torch.Tensor],
    cfg: dict[str, Any],
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    bsz, cand_n = candidate_indices.shape
    flat = candidate_indices.detach().cpu().reshape(-1)
    pooled = {key: val.index_select(0, flat).to(device, non_blocking=True).reshape(bsz, cand_n, -1) for key, val in pooled_bank.items()}
    pooled_score = core.retriever.score_candidates(q, pooled)["retriever_score"]
    visual_clip = clip_bank["visual"].index_select(0, flat).to(device, non_blocking=True).float().reshape(bsz, cand_n, -1, q["q_visual"].shape[-1])
    subtitle_clip = clip_bank["subtitle"].index_select(0, flat).to(device, non_blocking=True).float().reshape(bsz, cand_n, -1, q["q_subtitle"].shape[-1])
    joint_clip = clip_bank["joint"].index_select(0, flat).to(device, non_blocking=True).float().reshape(bsz, cand_n, -1, q["q_joint"].shape[-1])
    sv_t = torch.einsum("bd,bctd->bct", q["q_visual"], visual_clip)
    ss_t = torch.einsum("bd,bctd->bct", q["q_subtitle"], subtitle_clip)
    sj_t = torch.einsum("bd,bctd->bct", q["q_joint"], joint_clip)
    topk = int(cfg.get("late_soft_topk", 8))
    temp = float(cfg.get("late_temperature", 0.07))
    sv = _safe_soft_topk(sv_t, topk, temp)
    ss = _safe_soft_topk(ss_t, topk, temp)
    sj = _safe_soft_topk(sj_t, topk, temp)
    gate = q["gate"]
    late_score = core.retriever.scale.clamp(1.0, 30.0) * (gate[:, 0:1] * sv + gate[:, 1:2] * ss + gate[:, 2:3] * sj)
    token_weight = float(cfg.get("token_maxsim_weight", 0.25))
    token_score = late_score.new_zeros(late_score.shape)
    if token_weight > 0.0:
        token_h = F.normalize(core.query_encoder.joint_proj(query_tokens.to(device, non_blocking=True)), dim=-1)
        token_sim = torch.einsum("bld,bctd->blct", token_h, joint_clip).max(dim=-1).values
        mask = query_mask.to(device, non_blocking=True).float().unsqueeze(-1)
        token_score = (token_sim * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        token_score = core.retriever.scale.clamp(1.0, 30.0) * token_score
    combined = float(cfg.get("pooled_score_weight", 0.35)) * pooled_score + float(cfg.get("late_score_weight", 0.65)) * late_score + token_weight * token_score
    return combined, {"pooled": pooled_score, "late": late_score, "token": token_score, "visual": sv, "subtitle": ss, "joint": sj}


def late_scores_for_raw_candidates(
    core: torch.nn.Module,
    q: dict[str, torch.Tensor],
    query_tokens: torch.Tensor,
    query_mask: torch.Tensor,
    candidate_indices: torch.Tensor,
    visual_seq_bank: np.ndarray,
    subtitle_seq_bank: np.ndarray,
    cfg: dict[str, Any],
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    bsz, cand_n = candidate_indices.shape
    flat = candidate_indices.detach().cpu().reshape(-1).numpy()
    visual = torch.from_numpy(visual_seq_bank[flat].astype(np.float32, copy=False)).to(device, non_blocking=True).reshape(bsz, cand_n, visual_seq_bank.shape[1], visual_seq_bank.shape[2])
    subtitle = torch.from_numpy(subtitle_seq_bank[flat].astype(np.float32, copy=False)).to(device, non_blocking=True).reshape(bsz, cand_n, subtitle_seq_bank.shape[1], subtitle_seq_bank.shape[2])
    enc = core.video_encoder(visual, subtitle)
    pooled_score = core.retriever.score_candidates(q, enc)["retriever_score"]
    sv_t = torch.einsum("bd,bctd->bct", q["q_visual"], enc["visual"])
    ss_t = torch.einsum("bd,bctd->bct", q["q_subtitle"], enc["subtitle"])
    sj_t = torch.einsum("bd,bctd->bct", q["q_joint"], enc["joint"])
    topk = int(cfg.get("late_soft_topk", 8))
    temp = float(cfg.get("late_temperature", 0.07))
    sv = _safe_soft_topk(sv_t, topk, temp)
    ss = _safe_soft_topk(ss_t, topk, temp)
    sj = _safe_soft_topk(sj_t, topk, temp)
    gate = q["gate"]
    late_score = core.retriever.scale.clamp(1.0, 30.0) * (gate[:, 0:1] * sv + gate[:, 1:2] * ss + gate[:, 2:3] * sj)
    token_weight = float(cfg.get("token_maxsim_weight", 0.25))
    token_score = late_score.new_zeros(late_score.shape)
    if token_weight > 0.0:
        token_h = F.normalize(core.query_encoder.joint_proj(query_tokens.to(device, non_blocking=True)), dim=-1)
        token_sim = torch.einsum("bld,bctd->blct", token_h, enc["joint"]).max(dim=-1).values
        mask = query_mask.to(device, non_blocking=True).float().unsqueeze(-1)
        token_score = (token_sim * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        token_score = core.retriever.scale.clamp(1.0, 30.0) * token_score
    combined = float(cfg.get("pooled_score_weight", 0.35)) * pooled_score + float(cfg.get("late_score_weight", 0.65)) * late_score + token_weight * token_score
    return combined, {"pooled": pooled_score, "late": late_score, "token": token_score, "visual": sv, "subtitle": ss, "joint": sj}


@torch.no_grad()
def two_stage_retrieve(
    model: torch.nn.Module,
    rows: list[dict[str, Any]],
    query_cache: dict[int, np.ndarray],
    video_to_idx: dict[str, int],
    pooled_bank: dict[str, torch.Tensor],
    clip_bank: dict[str, torch.Tensor],
    cfg: dict[str, Any],
    device: torch.device,
    split: str,
) -> tuple[list[int | None], dict[int, list[int]], dict[str, Any]]:
    core = _model_core(model)
    core.eval()
    broad_key = "broad_topk_train" if split.startswith("train") else "broad_topk_eval"
    broad_k = int(cfg.get(broad_key, cfg.get("broad_topk_eval", 1000)))
    final_k = int(cfg.get("late_topk", cfg.get("candidate_topk_eval", 200)))
    batch_q = int(cfg.get("eval_batch_queries", 32))
    cand_chunk = int(cfg.get("late_candidate_chunk", 256))
    ranks: list[int | None] = []
    top_cache: dict[int, list[int]] = {}
    by_qtype: dict[str, list[int | None]] = {}
    for st in range(0, len(rows), batch_q):
        batch_rows = rows[st: st + batch_q]
        toks, qmask, qtypes, qids, gt_vids = _pad_query_tokens(batch_rows, query_cache)
        q = core.encode_query(toks.to(device, non_blocking=True), qtypes.to(device, non_blocking=True), qmask.to(device, non_blocking=True))
        broad_scores = pooled_scores(core, q, pooled_bank, device, int(cfg.get("chunk_size", 256)))
        broad_idx = torch.topk(broad_scores, k=min(broad_k, broad_scores.shape[1]), dim=1).indices
        late_parts = []
        for cs in range(0, broad_idx.shape[1], cand_chunk):
            sub_idx = broad_idx[:, cs: cs + cand_chunk]
            late_sub, _ = late_scores_for_candidates(core, q, toks, qmask, sub_idx, pooled_bank, clip_bank, cfg, device)
            late_parts.append(late_sub)
        late_scores = torch.cat(late_parts, dim=1)
        order = torch.argsort(late_scores, dim=1, descending=True)
        reranked = torch.gather(broad_idx, 1, order[:, : min(final_k, order.shape[1])]).detach().cpu().numpy()
        for bi, qid in enumerate(qids):
            top = reranked[bi].astype(np.int64, copy=False).tolist()
            gt_idx = int(video_to_idx[str(gt_vids[bi])])
            rank = _rank(top, gt_idx)
            ranks.append(rank)
            top_cache[int(qid)] = [int(x) for x in top]
            qtype = str(batch_rows[bi].get("type", "unknown"))
            by_qtype.setdefault(qtype, []).append(rank)
    audit = {
        "split": split,
        "broad_topk": broad_k,
        "broad_topk_source": broad_key,
        "late_topk": final_k,
        "late_interaction_enabled": True,
        "query_type_breakdown": {key: _rank_metrics(vals) for key, vals in by_qtype.items()},
    }
    return ranks, top_cache, audit


def teacher_replay(rows: list[dict[str, Any]], video_to_idx: dict[str, int], k: int) -> tuple[list[int | None], dict[int, list[TeacherRankItem]], dict[str, Any]]:
    ref = FirstStageReference()
    qids = [int(r["desc_id"]) for r in rows]
    teacher_items = ref.bulk_top_items(qids, video_to_idx, k=k)
    score_audit = ref.score_schema_audit(qids, sample_k=16)
    ref.close()
    ranks = []
    for row in rows:
        gt_idx = int(video_to_idx[str(row["vid_name"])])
        ranks.append(_rank([int(x.video_index) for x in teacher_items.get(int(row["desc_id"]), [])], gt_idx))
    return ranks, teacher_items, score_audit


def build_banks_for_c28e(cfg: dict[str, Any], model: torch.nn.Module, device: torch.device, force: bool = False) -> dict[str, Any]:
    banks = prepare_banks(cfg, force=force)
    video_ids = [str(x) for x in banks["visual_np"]["video_ids"].tolist()]
    banks["video_bank"].video_ids = video_ids
    banks["video_bank"].video_to_idx = {v: i for i, v in enumerate(video_ids)}
    banks["video_bank"].idx_to_video = {i: v for v, i in banks["video_bank"].video_to_idx.items()}
    visual_mean = torch.from_numpy(banks["visual_np"]["visual_mean"].astype(np.float32))
    subtitle_mean = torch.from_numpy(banks["subtitle_np"]["subtitle_mean"].astype(np.float32))
    pooled = encode_pooled_bank(model, visual_mean, subtitle_mean, int(cfg.get("chunk_size", 256)), device)
    visual_seq, visual_seq_manifest = banks["video_bank"].build_or_load_sequence_bank(video_ids, target_len=64, max_videos=int(cfg.get("max_videos", 0) or 0) or None, force=force)
    subtitle_seq, subtitle_seq_manifest = banks["subtitle_bank"].build_or_load_sequence_bank(video_ids, target_len=64, max_videos=int(cfg.get("max_videos", 0) or 0) or None, force=force)
    clip = encode_clip_bank(model, visual_seq, subtitle_seq, int(cfg.get("chunk_size", 256)), device)
    banks["video_bank"].clear_sequence_cache()
    banks["subtitle_bank"].clear_sequence_cache()
    banks.update({
        "video_ids": video_ids,
        "visual_mean_tensor": visual_mean,
        "subtitle_mean_tensor": subtitle_mean,
        "visual_seq_bank": visual_seq,
        "subtitle_seq_bank": subtitle_seq,
        "pooled_bank": pooled,
        "clip_bank": clip,
        "visual_sequence_manifest": visual_seq_manifest,
        "subtitle_sequence_manifest": subtitle_seq_manifest,
        "clip_bank_manifest": {
            "video_count": len(video_ids),
            "target_len": 64,
            "hidden_dim": int(pooled["joint_pool"].shape[-1]),
            "dtype": "float16_cpu_cache",
            "late_interaction_enabled": True,
        },
    })
    return banks


def run_c28e_retriever_replay(
    cfg: dict[str, Any],
    out_dir: Path,
    split: str,
    checkpoint: str | Path | None = None,
    device_arg: str | None = None,
    force: bool = False,
) -> dict[str, Any]:
    seed_all(int(cfg.get("seed", 2026)))
    out_dir.mkdir(parents=True, exist_ok=True)
    device = device_from_arg(device_arg)
    model = build_model(cfg, device)
    ckpt_loaded = False
    if checkpoint and Path(checkpoint).exists():
        load_checkpoint(Path(checkpoint), model)
        ckpt_loaded = True
    banks = build_banks_for_c28e(cfg, model, device, force=force)
    rows = banks["split_manager"].records(split, max_queries=int(cfg.get("max_queries", 0) or 0) or None)
    query_cache = banks["query_bank"].bulk_tokens([int(r["desc_id"]) for r in rows])
    teacher_ranks, teacher_items, score_audit = teacher_replay(rows, banks["video_bank"].video_to_idx, k=int(cfg.get("teacher_topk", 200)))
    student_ranks, student_top, retrieve_audit = two_stage_retrieve(
        model,
        rows,
        query_cache,
        banks["video_bank"].video_to_idx,
        banks["pooled_bank"],
        banks["clip_bank"],
        cfg,
        device,
        split,
    )
    train_gt_videos = {str(r["vid_name"]) for r in banks["split_manager"].records("train_fit", max_queries=None)}
    overlap = [str(r["vid_name"]) in train_gt_videos for r in rows]
    teacher_overlap_vals = []
    for row in rows:
        qid = int(row["desc_id"])
        teacher_set = {int(x.video_index) for x in teacher_items.get(qid, [])[: int(cfg.get("late_topk", 200))]}
        student_set = set(student_top.get(qid, [])[: int(cfg.get("late_topk", 200))])
        if teacher_set:
            teacher_overlap_vals.append(len(teacher_set.intersection(student_set)) / max(1, len(teacher_set)))
    rec = {
        "stage": "C28E-replay",
        "status": "C28E_RETRIEVER_REPLAY_COMPLETE",
        "split": split,
        "checkpoint": str(checkpoint) if checkpoint else None,
        "checkpoint_loaded": ckpt_loaded,
        "teacher": {**_rank_metrics(teacher_ranks), "score_schema_audit": score_audit},
        "student": {**_rank_metrics(student_ranks), "teacher_overlap_at_late_topk": float(np.mean(teacher_overlap_vals)) if teacher_overlap_vals else None},
        "retrieve_audit": retrieve_audit,
        "video_overlap_audit": {
            "gt_video_seen_in_train_rate": 100.0 * sum(overlap) / max(1, len(overlap)),
            "unseen_gt_video_count": len(overlap) - sum(overlap),
        },
        **official_safety_manifest(),
    }
    write_json(out_dir / f"C28E_RETRIEVER_REPLAY_{split}.json", rec)
    return rec


def train_c28e_retriever(
    cfg: dict[str, Any],
    out_dir: Path,
    device_arg: str | None = None,
    force: bool = False,
    resume: bool = False,
) -> dict[str, Any]:
    seed = int(cfg.get("seed", 2026))
    seed_all(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = device_from_arg(device_arg)
    model = build_model(cfg, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg.get("lr", 1.2e-4)), weight_decay=float(cfg.get("weight_decay", 0.01)))
    ckpt_dir = TMP_ROOT / "checkpoints"
    latest_ckpt = ckpt_dir / f"C28E_RETRIEVER_{cfg.get('mode','code_review')}_seed{seed}.pt"
    best_ckpt = ckpt_dir / f"C28E_RETRIEVER_{cfg.get('mode','code_review')}_seed{seed}.best.pt"
    log_path = out_dir / "C28E_RETRIEVER_TRAINING_LOG.json"
    existing = load_json(log_path, {}) if resume else {}
    logs: list[dict[str, Any]] = list(existing.get("training_log", []))
    best_score = float(existing.get("best_select_score", -math.inf))
    best_manifest = existing.get("best_checkpoint_manifest")
    start_epoch = 0
    if resume and latest_ckpt.exists():
        loaded = load_checkpoint(latest_ckpt, model, optimizer)
        start_epoch = int(loaded.get("epoch", -1)) + 1
    banks = build_banks_for_c28e(cfg, model, device, force=force)
    rows = banks["split_manager"].records("train_fit", max_queries=int(cfg.get("max_queries", 0) or 0) or None)
    query_cache = banks["query_bank"].bulk_tokens([int(r["desc_id"]) for r in rows])
    ref = FirstStageReference()
    teacher_items = ref.bulk_top_items([int(r["desc_id"]) for r in rows], banks["video_bank"].video_to_idx, k=int(cfg.get("teacher_topk", 200)))
    teacher_score_audit = ref.score_schema_audit([int(r["desc_id"]) for r in rows], sample_k=16)
    ref.close()
    rng = random.Random(seed)
    all_indices = list(range(len(banks["video_ids"])))
    batch_size = int(cfg.get("batch_size", 32))
    train_k = int(cfg.get("candidate_topk_train", 200))
    grad_accum = int(cfg.get("grad_accum_steps", 1))
    for epoch in range(start_epoch, int(cfg.get("epochs", 1))):
        banks["pooled_bank"] = encode_pooled_bank(model, banks["visual_mean_tensor"], banks["subtitle_mean_tensor"], int(cfg.get("chunk_size", 256)), device)
        banks["clip_bank"] = encode_clip_bank(model, banks["visual_seq_bank"], banks["subtitle_seq_bank"], int(cfg.get("chunk_size", 256)), device)
        model.train()
        order = list(range(len(rows)))
        rng.shuffle(order)
        loss_acc: dict[str, float] = {}
        steps = 0
        optimizer.zero_grad(set_to_none=True)
        for off in range(0, len(order), batch_size):
            batch_rows = [rows[i] for i in order[off: off + batch_size]]
            toks, qmask, qtypes, qids, gt_vids = _pad_query_tokens(batch_rows, query_cache)
            core = _model_core(model)
            with torch.no_grad():
                student_ranks, student_top, _ = two_stage_retrieve(
                    model,
                    batch_rows,
                    query_cache,
                    banks["video_bank"].video_to_idx,
                    banks["pooled_bank"],
                    banks["clip_bank"],
                    cfg,
                    device,
                    "train_fit_batch",
                )
            model.train()
            cand_lists = []
            targets = []
            teacher_logits = []
            teacher_set_labels = []
            for bi, qid in enumerate(qids):
                gt_idx = int(banks["video_bank"].video_to_idx[str(gt_vids[bi])])
                teacher = teacher_items.get(int(qid), [])
                teacher_indices = [int(x.video_index) for x in teacher[: int(cfg.get("teacher_anchor_topk", 80))]]
                student_indices = student_top.get(int(qid), [])[: int(cfg.get("student_negative_k", 128))]
                randoms = rng.sample(all_indices, k=min(int(cfg.get("random_negative_k", 64)), len(all_indices)))
                cands = _merge_candidates(gt_idx, teacher_indices, student_indices, randoms, train_k)
                while len(cands) < train_k:
                    extra = rng.randrange(len(all_indices))
                    if extra not in cands:
                        cands.append(extra)
                cand_lists.append(cands[:train_k])
                targets.append(cand_lists[-1].index(gt_idx))
                teacher_logits.append(_teacher_logits(cand_lists[-1], teacher, gt_idx, float(cfg.get("teacher_score_temperature", 2.0))))
                teacher_set = {int(x.video_index) for x in teacher[: int(cfg.get("teacher_topk", 200))]}
                teacher_set_labels.append(torch.tensor([1.0 if int(x) in teacher_set else 0.0 for x in cand_lists[-1]], dtype=torch.float32))
            cand_idx = torch.tensor(np.asarray(cand_lists, dtype=np.int64), dtype=torch.long, device=device)
            target_t = torch.tensor(targets, dtype=torch.long, device=device)
            q = core.encode_query(toks.to(device, non_blocking=True), qtypes.to(device, non_blocking=True), qmask.to(device, non_blocking=True))
            scores, _ = late_scores_for_raw_candidates(core, q, toks, qmask, cand_idx, banks["visual_seq_bank"], banks["subtitle_seq_bank"], cfg, device)
            l_gt = F.cross_entropy(scores, target_t)
            t_logits = torch.stack(teacher_logits).to(device)
            teacher_prob = torch.softmax(t_logits, dim=1)
            temp = float(cfg.get("teacher_score_temperature", 2.0))
            l_kl = F.kl_div(F.log_softmax(scores / temp, dim=1), teacher_prob, reduction="batchmean") * (temp ** 2)
            pos = scores.gather(1, target_t.view(-1, 1)).squeeze(1)
            neg = scores.masked_fill(F.one_hot(target_t, num_classes=scores.shape[1]).bool(), -1e4).max(dim=1).values
            l_pair = F.relu(1.0 - pos + neg).mean()
            set_labels = torch.stack(teacher_set_labels).to(device)
            l_set = F.binary_cross_entropy_with_logits(scores / 30.0, set_labels, reduction="mean")
            loss = (
                float(cfg.get("lambda_gt_rank", 1.0)) * l_gt
                + float(cfg.get("lambda_teacher_kl", 1.2)) * l_kl
                + float(cfg.get("lambda_teacher_pair", 0.5)) * l_pair
                + float(cfg.get("lambda_teacher_set", 0.4)) * l_set
            )
            (loss / grad_accum).backward()
            if (steps + 1) % grad_accum == 0 or (off + batch_size >= len(order)):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            for key, val in {"L_total": loss, "L_gt_rank": l_gt, "L_teacher_KL": l_kl, "L_teacher_pair": l_pair, "L_teacher_set": l_set}.items():
                loss_acc[key] = loss_acc.get(key, 0.0) + float(val.detach().cpu().item())
            steps += 1
        avg = {key: val / max(1, steps) for key, val in loss_acc.items()}
        latest_manifest = save_checkpoint(latest_ckpt, model, optimizer, epoch, {"train_loss": avg})
        train_replay = run_c28e_retriever_replay(cfg, out_dir / "epoch_replay", "train_fit", checkpoint=latest_ckpt, device_arg=device_arg, force=False)
        select_replay = run_c28e_retriever_replay(cfg, out_dir / "epoch_replay", str(cfg.get("selection_split", "calib_select")), checkpoint=latest_ckpt, device_arg=device_arg, force=False)
        select_score = _selection_score(select_replay["student"], train_replay["student"], cfg)
        latest_manifest = save_checkpoint(latest_ckpt, model, optimizer, epoch, {"train_loss": avg, "select_score": select_score})
        if select_score > best_score:
            best_score = select_score
            shutil.copy2(latest_ckpt, best_ckpt)
            best_manifest = {"path": str(best_ckpt), "source_checkpoint": str(latest_ckpt), "select_score": best_score, "epoch": epoch}
        epoch_rec = {
            "epoch": epoch,
            "loss": avg,
            "selection_split": str(cfg.get("selection_split", "calib_select")),
            "select_score": select_score,
            "train_metrics": train_replay["student"],
            "select_metrics": select_replay["student"],
            "checkpoint_manifest": latest_manifest,
        }
        logs.append(epoch_rec)
        write_json(log_path, {
            "status": "running",
            "training_log": logs,
            "best_select_score": best_score,
            "best_checkpoint_manifest": best_manifest,
            "teacher_score_audit": teacher_score_audit,
            **official_safety_manifest(),
        })
    rec = {
        "status": "C28E_RETRIEVER_TRAINED_SELECTION_ONLY",
        "training_log": logs,
        "best_select_score": best_score,
        "best_checkpoint_manifest": best_manifest,
        "teacher_score_audit": teacher_score_audit,
        **official_safety_manifest(),
    }
    write_json(out_dir / "C28E_RETRIEVER_TRAINING_DECISION.json", rec)
    return rec
