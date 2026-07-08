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
from blueprint_e2e_v2.data.first_stage_reference import FirstStageReference
from blueprint_e2e_v2.engine.checkpoint import load_checkpoint, save_checkpoint
from blueprint_e2e_v2.engine.train import build_model, device_from_arg, prepare_banks
from blueprint_e2e_v2.utils.hashing import stable_hash
from blueprint_e2e_v2.utils.io import write_json, write_text
from blueprint_e2e_v2.utils.seed import seed_all


def _pad_query_tokens(rows: list[dict[str, Any]], query_cache: dict[int, np.ndarray]) -> tuple[torch.Tensor, torch.Tensor, list[int], list[str]]:
    arrays = [query_cache[int(r["desc_id"])] for r in rows]
    max_len = max(a.shape[0] for a in arrays)
    dim = arrays[0].shape[1]
    tokens = np.zeros((len(arrays), max_len, dim), dtype=np.float32)
    qtypes = np.zeros((len(arrays),), dtype=np.int64)
    for i, (a, r) in enumerate(zip(arrays, rows)):
        tokens[i, : a.shape[0]] = a
        qtypes[i] = {"v": 0, "t": 1, "vt": 2}.get(str(r.get("type", "")), 3)
    return torch.from_numpy(tokens), torch.from_numpy(qtypes), [int(r["desc_id"]) for r in rows], [str(r["vid_name"]) for r in rows]


def _rank_metrics(ranks: list[int | None], ks: tuple[int, ...] = (1, 5, 10, 50, 100, 200)) -> dict[str, Any]:
    valid_ranks = [int(r) for r in ranks if r is not None]
    out: dict[str, Any] = {"query_count": len(ranks), "missing_gt_count": len(ranks) - len(valid_ranks)}
    for k in ks:
        out[f"VR@{k}"] = 100.0 * sum((r is not None and int(r) <= k) for r in ranks) / max(1, len(ranks))
    out["gt_rank_mean"] = float(np.mean(valid_ranks)) if valid_ranks else None
    out["gt_rank_median"] = float(np.median(valid_ranks)) if valid_ranks else None
    out["gt_rank_p95"] = float(np.percentile(valid_ranks, 95)) if valid_ranks else None
    out["wrong_video_top1_rate"] = 100.0 * sum((r is None or int(r) != 1) for r in ranks) / max(1, len(ranks))
    return out


def _candidate_rank(indices: list[int], gt_idx: int) -> int | None:
    try:
        return int(indices.index(int(gt_idx)) + 1)
    except ValueError:
        return None


def _teacher_logits_for_candidates(candidates: list[int], teacher_rank: dict[int, int], gt_idx: int, temperature: float) -> torch.Tensor:
    logits = torch.full((len(candidates),), -8.0, dtype=torch.float32)
    for i, vid_idx in enumerate(candidates):
        rank = teacher_rank.get(int(vid_idx))
        if rank is not None:
            logits[i] = float(max(0, 256 - rank)) / max(float(temperature), 1e-6)
    if gt_idx in candidates:
        logits[candidates.index(gt_idx)] = max(float(logits[candidates.index(gt_idx)]), 256.0 / max(float(temperature), 1e-6))
    return logits


def _base_bank(bank: dict[str, torch.Tensor], suffix: str = "") -> dict[str, torch.Tensor]:
    return {
        "visual_pool": bank[f"visual_pool{suffix}"],
        "subtitle_pool": bank[f"subtitle_pool{suffix}"],
        "joint_pool": bank[f"joint_pool{suffix}"],
    }


def score_encoded_bank(core: Any, q: dict[str, torch.Tensor], encoded_bank: dict[str, torch.Tensor], max_weight: float = 0.0) -> torch.Tensor:
    score = core.retriever.score_bank(q, _base_bank(encoded_bank))
    if max_weight > 0.0 and "visual_pool_max" in encoded_bank:
        score_max = core.retriever.score_bank(q, _base_bank(encoded_bank, "_max"))
        score = (1.0 - float(max_weight)) * score + float(max_weight) * score_max
    return score


@torch.no_grad()
def encode_bank(
    model: torch.nn.Module,
    visual_bank: torch.Tensor,
    subtitle_bank: torch.Tensor,
    chunk_size: int,
    device: torch.device,
    visual_max_bank: torch.Tensor | None = None,
    subtitle_max_bank: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    core = model.module if hasattr(model, "module") else model
    core.eval()
    parts: dict[str, list[torch.Tensor]] = {"visual_pool": [], "subtitle_pool": [], "joint_pool": []}
    if visual_max_bank is not None and subtitle_max_bank is not None:
        parts.update({"visual_pool_max": [], "subtitle_pool_max": [], "joint_pool_max": []})
    for st in range(0, visual_bank.shape[0], chunk_size):
        enc = core.video_encoder.encode_pooled_bank(
            visual_bank[st: st + chunk_size].to(device, non_blocking=True),
            subtitle_bank[st: st + chunk_size].to(device, non_blocking=True),
        )
        for key in parts:
            if not key.endswith("_max"):
                parts[key].append(enc[key].detach())
        if visual_max_bank is not None and subtitle_max_bank is not None:
            enc_max = core.video_encoder.encode_pooled_bank(
                visual_max_bank[st: st + chunk_size].to(device, non_blocking=True),
                subtitle_max_bank[st: st + chunk_size].to(device, non_blocking=True),
            )
            for key in ["visual_pool", "subtitle_pool", "joint_pool"]:
                parts[f"{key}_max"].append(enc_max[key].detach())
    return {k: torch.cat(v, dim=0) for k, v in parts.items()}


@torch.no_grad()
def score_model_ranks(
    model: torch.nn.Module,
    rows: list[dict[str, Any]],
    query_cache: dict[int, np.ndarray],
    video_to_idx: dict[str, int],
    encoded_bank: dict[str, torch.Tensor],
    dynamic_topk: int,
    chunk_size: int,
    device: torch.device,
    max_weight: float = 0.0,
) -> tuple[list[int | None], dict[int, list[int]], list[dict[str, Any]]]:
    core = model.module if hasattr(model, "module") else model
    ranks: list[int | None] = []
    top_cache: dict[int, list[int]] = {}
    wrong_examples: list[dict[str, Any]] = []
    video_count = encoded_bank["visual_pool"].shape[0]
    for st in range(0, len(rows), 64):
        toks, qtypes, qids, gt_vids = _pad_query_tokens(rows[st: st + 64], query_cache)
        q = core.encode_query(toks.to(device, non_blocking=True), qtypes.to(device, non_blocking=True))
        scores = score_encoded_bank(core, q, encoded_bank, max_weight=max_weight)
        k = min(int(dynamic_topk), int(video_count))
        vals, idx = torch.topk(scores, k=k, dim=1)
        idx_cpu = idx.detach().cpu().numpy()
        vals_cpu = vals.detach().cpu().numpy()
        for bi, qid in enumerate(qids):
            gt_idx = int(video_to_idx[str(gt_vids[bi])])
            top = idx_cpu[bi].astype(np.int64, copy=False).tolist()
            top_cache[int(qid)] = [int(x) for x in top]
            rank = _candidate_rank(top, gt_idx)
            ranks.append(rank)
            if len(wrong_examples) < 80 and (rank is None or rank != 1):
                wrong_examples.append({
                    "query_id": int(qid),
                    "gt_video_index": gt_idx,
                    "gt_rank_within_topk": rank,
                    "top1_video_index": int(top[0]) if top else None,
                    "top1_score": float(vals_cpu[bi, 0]) if top else None,
                    "gt_score": float(scores[bi, gt_idx].detach().cpu().item()),
                })
        done = min(st + 64, len(rows))
        if done % 1600 == 0 or done == len(rows):
            print(f"C28D retriever replay: scored {done}/{len(rows)} queries", flush=True)
    return ranks, top_cache, wrong_examples


def teacher_replay(rows: list[dict[str, Any]], video_to_idx: dict[str, int], k: int) -> tuple[list[int | None], dict[int, list[int]]]:
    ref = FirstStageReference()
    qids = [int(r["desc_id"]) for r in rows]
    cache = ref.bulk_top_indices(qids, video_to_idx, k=k)
    ref.close()
    ranks: list[int | None] = []
    for row in rows:
        gt_idx = int(video_to_idx[str(row["vid_name"])])
        ranks.append(_candidate_rank(cache.get(int(row["desc_id"]), []), gt_idx))
    return ranks, cache


def random_replay(rows: list[dict[str, Any]], video_to_idx: dict[str, int], video_count: int, k: int, seed: int) -> list[int | None]:
    rng = random.Random(int(seed))
    ranks: list[int | None] = []
    all_indices = list(range(video_count))
    for row in rows:
        sample = rng.sample(all_indices, k=min(k, video_count))
        gt_idx = int(video_to_idx[str(row["vid_name"])])
        ranks.append(_candidate_rank(sample, gt_idx))
    return ranks


def run_retriever_replay_audit(
    cfg: dict[str, Any],
    out_dir: Path,
    split: str = "calib_holdout",
    device_arg: str | None = None,
    checkpoint: str | Path | None = None,
    force: bool = False,
) -> dict[str, Any]:
    seed_all(int(cfg.get("seed", 2026)))
    out_dir.mkdir(parents=True, exist_ok=True)
    device = device_from_arg(device_arg)
    banks = prepare_banks(cfg, force=force)
    video_ids = [str(x) for x in banks["visual_np"]["video_ids"].tolist()]
    banks["video_bank"].video_ids = video_ids
    banks["video_bank"].video_to_idx = {v: i for i, v in enumerate(video_ids)}
    banks["video_bank"].idx_to_video = {i: v for v, i in banks["video_bank"].video_to_idx.items()}
    rows = banks["split_manager"].records(split, max_queries=int(cfg.get("max_queries", 0) or 0) or None)
    query_cache = banks["query_bank"].bulk_tokens([int(r["desc_id"]) for r in rows])
    visual_bank = torch.from_numpy(banks["visual_np"]["visual_mean"].astype(np.float32))
    subtitle_bank = torch.from_numpy(banks["subtitle_np"]["subtitle_mean"].astype(np.float32))
    visual_max_bank = torch.from_numpy(banks["visual_np"]["visual_max"].astype(np.float32)) if bool(cfg.get("use_moment_aware_bank", False)) else None
    subtitle_max_bank = torch.from_numpy(banks["subtitle_np"]["subtitle_max"].astype(np.float32)) if bool(cfg.get("use_moment_aware_bank", False)) else None
    max_weight = float(cfg.get("moment_max_score_weight", 0.0)) if bool(cfg.get("use_moment_aware_bank", False)) else 0.0
    dynamic_topk = int(cfg.get("candidate_topk_eval", cfg.get("dynamic_topk", 200)))

    teacher_ranks, teacher_cache = teacher_replay(rows, banks["video_bank"].video_to_idx, k=dynamic_topk)
    teacher_rec = {"split": split, "kind": "R1_first_stage_teacher", **_rank_metrics(teacher_ranks)}
    write_json(out_dir / "C28D_1_TEACHER_REPLAY.json", teacher_rec)

    random_ranks = random_replay(rows, banks["video_bank"].video_to_idx, len(video_ids), dynamic_topk, int(cfg.get("seed", 2026)))
    random_rec = {"split": split, "kind": "R0_random", **_rank_metrics(random_ranks)}
    write_json(out_dir / "C28D_1_RANDOM_REPLAY.json", random_rec)

    zero_model = build_model(cfg, device)
    zero_bank = encode_bank(zero_model, visual_bank, subtitle_bank, int(cfg.get("chunk_size", 256)), device, visual_max_bank, subtitle_max_bank)
    zero_ranks, zero_top, zero_wrong = score_model_ranks(
        zero_model,
        rows,
        query_cache,
        banks["video_bank"].video_to_idx,
        zero_bank,
        dynamic_topk,
        int(cfg.get("chunk_size", 256)),
        device,
        max_weight=max_weight,
    )
    zero_rec = {"split": split, "kind": "R2_C28C_zero_shot", **_rank_metrics(zero_ranks), "wrong_examples": zero_wrong}
    write_json(out_dir / "C28D_1_STUDENT_ZERO_SHOT_REPLAY.json", zero_rec)

    ckpt_path = Path(checkpoint) if checkpoint else TMP_ROOT / "checkpoints" / f"C28C_FULL_{cfg.get('mode','medium')}_seed{cfg.get('seed',2026)}.pt"
    ckpt_loaded = False
    ckpt_rec: dict[str, Any]
    if ckpt_path.exists():
        ckpt_model = build_model(cfg, device)
        load_checkpoint(ckpt_path, ckpt_model)
        ckpt_loaded = True
        ckpt_bank = encode_bank(ckpt_model, visual_bank, subtitle_bank, int(cfg.get("chunk_size", 256)), device, visual_max_bank, subtitle_max_bank)
        ckpt_ranks, ckpt_top, ckpt_wrong = score_model_ranks(
            ckpt_model,
            rows,
            query_cache,
            banks["video_bank"].video_to_idx,
            ckpt_bank,
            dynamic_topk,
            int(cfg.get("chunk_size", 256)),
            device,
            max_weight=max_weight,
        )
        overlap_vals = []
        for qid, top in ckpt_top.items():
            teacher = set(teacher_cache.get(int(qid), [])[:dynamic_topk])
            if teacher:
                overlap_vals.append(len(set(top[:dynamic_topk]).intersection(teacher)) / max(1, len(teacher)))
        ckpt_rec = {
            "split": split,
            "kind": "R3_C28C_checkpoint",
            "checkpoint": str(ckpt_path),
            "checkpoint_loaded": True,
            **_rank_metrics(ckpt_ranks),
            "teacher_overlap_at_eval_k": float(np.mean(overlap_vals)) if overlap_vals else None,
            "wrong_examples": ckpt_wrong,
        }
    else:
        ckpt_rec = {"split": split, "kind": "R3_C28C_checkpoint", "checkpoint": str(ckpt_path), "checkpoint_loaded": False}
    write_json(out_dir / "C28D_1_STUDENT_C28C_CHECKPOINT_REPLAY.json", ckpt_rec)

    decision = {
        "stage": "C28D-1",
        "status": "C28D_RETRIEVER_REPLAY_AUDIT_COMPLETE",
        "split": split,
        "dynamic_topk": dynamic_topk,
        "moment_aware_bank": {"enabled": bool(cfg.get("use_moment_aware_bank", False)), "max_score_weight": max_weight},
        "teacher": teacher_rec,
        "random": random_rec,
        "student_zero_shot": zero_rec,
        "student_c28c_checkpoint": ckpt_rec,
        "checkpoint_loaded": ckpt_loaded,
        "diagnosis": "retriever/front-rank must pass before full mutual VCMR training",
        "schema_hash": stable_hash({"teacher": teacher_rec, "zero": zero_rec, "ckpt": ckpt_rec}),
    }
    write_json(out_dir / "C28D_1_RETRIEVER_FAILURE_DECISION.json", decision)
    write_text(
        out_dir / "C28D_1_RETRIEVER_FAILURE_DECISION.md",
        "# C28D-1 Retriever Replay Decision\n\n"
        f"Status: `{decision['status']}`.\n\n"
        f"Teacher VR@100: `{teacher_rec.get('VR@100')}`.\n\n"
        f"C28C checkpoint VR@100: `{ckpt_rec.get('VR@100')}`.\n\n"
        "Full mutual training remains gated until model-only retrieval recovers.\n",
    )
    return decision


def _curriculum_counts(epoch: int, cfg: dict[str, Any]) -> tuple[int, int, int]:
    if epoch <= 2:
        return int(cfg.get("teacher_warm_topk", 200)), 0, int(cfg.get("random_negative_k", 48))
    if epoch <= 5:
        return 160, 40, int(cfg.get("random_negative_k", 48))
    if epoch <= 8:
        return 100, 100, int(cfg.get("random_negative_k", 48))
    return int(cfg.get("late_teacher_topk", 80)), int(cfg.get("late_student_topk", 120)), int(cfg.get("random_negative_k", 48))


def _merge_candidates(gt_idx: int, teacher: list[int], student: list[int], randoms: list[int], limit: int) -> list[int]:
    out = [int(gt_idx)]
    for source in (teacher, student, randoms):
        for idx in source:
            idx = int(idx)
            if idx not in out:
                out.append(idx)
            if len(out) >= limit:
                return out
    return out


def train_retriever_distillation(
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
    banks = prepare_banks(cfg, force=force)
    video_ids = [str(x) for x in banks["visual_np"]["video_ids"].tolist()]
    banks["video_bank"].video_ids = video_ids
    banks["video_bank"].video_to_idx = {v: i for i, v in enumerate(video_ids)}
    banks["video_bank"].idx_to_video = {i: v for v, i in banks["video_bank"].video_to_idx.items()}
    rows = banks["split_manager"].records("train_fit", max_queries=int(cfg.get("max_queries", 0) or 0) or None)
    query_cache = banks["query_bank"].bulk_tokens([int(r["desc_id"]) for r in rows])
    visual_bank = torch.from_numpy(banks["visual_np"]["visual_mean"].astype(np.float32))
    subtitle_bank = torch.from_numpy(banks["subtitle_np"]["subtitle_mean"].astype(np.float32))
    visual_max_bank = torch.from_numpy(banks["visual_np"]["visual_max"].astype(np.float32)) if bool(cfg.get("use_moment_aware_bank", False)) else None
    subtitle_max_bank = torch.from_numpy(banks["subtitle_np"]["subtitle_max"].astype(np.float32)) if bool(cfg.get("use_moment_aware_bank", False)) else None
    max_weight = float(cfg.get("moment_max_score_weight", 0.0)) if bool(cfg.get("use_moment_aware_bank", False)) else 0.0
    model = build_model(cfg, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg.get("lr", 1.5e-4)), weight_decay=float(cfg.get("weight_decay", 0.01)))
    ckpt_path = TMP_ROOT / "checkpoints" / f"C28D_RETRIEVER_{cfg.get('mode','medium')}_seed{seed}.pt"
    best_ckpt_path = TMP_ROOT / "checkpoints" / f"C28D_RETRIEVER_{cfg.get('mode','medium')}_seed{seed}.best.pt"
    start_epoch = 0
    last_epoch_path: Path | None = None
    if resume and ckpt_path.exists():
        loaded = load_checkpoint(ckpt_path, model, optimizer)
        start_epoch = int(loaded.get("epoch", -1)) + 1
        if start_epoch > 0:
            last_epoch_path = TMP_ROOT / "checkpoints" / f"C28D_RETRIEVER_{cfg.get('mode','medium')}_seed{seed}.epoch{start_epoch - 1}.pt"
    teacher_ref = FirstStageReference()
    teacher_cache = teacher_ref.bulk_top_indices([int(r["desc_id"]) for r in rows], banks["video_bank"].video_to_idx, k=int(cfg.get("teacher_warm_topk", 200)))
    teacher_ref.close()
    rng = random.Random(seed)
    train_k = int(cfg.get("candidate_topk_train", cfg.get("dynamic_topk", 200)))
    batch_size = int(cfg.get("batch_size", 16))
    grad_accum = int(cfg.get("grad_accum_steps", 1))
    all_indices = list(range(len(video_ids)))
    logs: list[dict[str, Any]] = []
    best_manifest: dict[str, Any] | None = None
    best_score = -math.inf
    best_path: str | None = None
    for epoch in range(start_epoch, int(cfg.get("retriever_epochs", cfg.get("epochs", 1)))):
        encoded_bank = encode_bank(model, visual_bank, subtitle_bank, int(cfg.get("chunk_size", 256)), device, visual_max_bank, subtitle_max_bank)
        model.train()
        teacher_n, student_n, random_n = _curriculum_counts(epoch, cfg)
        order = list(range(len(rows)))
        rng.shuffle(order)
        optimizer.zero_grad(set_to_none=True)
        loss_acc: dict[str, float] = {}
        steps = 0
        recall_hits = {1: 0, 5: 0, 10: 0, 100: 0, 200: 0}
        for offset in range(0, len(order), batch_size):
            batch_rows = [rows[i] for i in order[offset: offset + batch_size]]
            toks, qtypes, qids, gt_vids = _pad_query_tokens(batch_rows, query_cache)
            core = model.module if hasattr(model, "module") else model
            with torch.no_grad():
                q_mine = core.encode_query(toks.to(device, non_blocking=True), qtypes.to(device, non_blocking=True))
                full_scores = score_encoded_bank(core, q_mine, encoded_bank, max_weight=max_weight)
                topk = min(max(student_n, train_k), full_scores.shape[1])
                student_top = torch.topk(full_scores, k=topk, dim=1).indices.detach().cpu().numpy()
            cand_lists: list[list[int]] = []
            targets: list[int] = []
            teacher_targets: list[torch.Tensor] = []
            for bi, qid in enumerate(qids):
                gt_idx = int(banks["video_bank"].video_to_idx[str(gt_vids[bi])])
                randoms = rng.sample(all_indices, k=min(random_n, len(all_indices)))
                teacher = teacher_cache.get(int(qid), [])[:teacher_n]
                student = student_top[bi].astype(np.int64, copy=False).tolist()[:student_n]
                candidates = _merge_candidates(gt_idx, teacher, student, randoms, train_k)
                while len(candidates) < train_k:
                    extra = rng.randrange(len(all_indices))
                    if extra not in candidates:
                        candidates.append(extra)
                cand_lists.append(candidates[:train_k])
                targets.append(cand_lists[-1].index(gt_idx))
                gt_rank = _candidate_rank(student_top[bi].astype(np.int64, copy=False).tolist(), gt_idx)
                for k in recall_hits:
                    recall_hits[k] += int(gt_rank is not None and gt_rank <= k)
                teacher_rank = {int(v): rank for rank, v in enumerate(teacher_cache.get(int(qid), []), start=1)}
                teacher_targets.append(_teacher_logits_for_candidates(cand_lists[-1], teacher_rank, gt_idx, float(cfg.get("teacher_temperature", 2.0))))
            cand_idx = torch.tensor(np.asarray(cand_lists, dtype=np.int64), dtype=torch.long, device=device)
            target_t = torch.tensor(targets, dtype=torch.long, device=device)
            teacher_logits = torch.stack(teacher_targets).to(device)
            q = core.encode_query(toks.to(device, non_blocking=True), qtypes.to(device, non_blocking=True))
            bsz, cand_n = cand_idx.shape
            flat_idx = cand_idx.detach().cpu().reshape(-1)
            raw_visual = visual_bank.index_select(0, flat_idx).to(device, non_blocking=True)
            raw_subtitle = subtitle_bank.index_select(0, flat_idx).to(device, non_blocking=True)
            enc_flat = core.video_encoder.encode_pooled_bank(raw_visual, raw_subtitle)
            enc = {k: v.reshape(bsz, cand_n, -1) for k, v in enc_flat.items()}
            student_scores = core.retriever.score_candidates(q, enc)["retriever_score"]
            if max_weight > 0.0 and visual_max_bank is not None and subtitle_max_bank is not None:
                raw_visual_max = visual_max_bank.index_select(0, flat_idx).to(device, non_blocking=True)
                raw_subtitle_max = subtitle_max_bank.index_select(0, flat_idx).to(device, non_blocking=True)
                enc_max_flat = core.video_encoder.encode_pooled_bank(raw_visual_max, raw_subtitle_max)
                enc_max = {k: v.reshape(bsz, cand_n, -1) for k, v in enc_max_flat.items()}
                student_scores_max = core.retriever.score_candidates(q, enc_max)["retriever_score"]
                student_scores = (1.0 - max_weight) * student_scores + max_weight * student_scores_max
            l_gt = F.cross_entropy(student_scores, target_t)
            teacher_prob = torch.softmax(teacher_logits, dim=1)
            l_kl = F.kl_div(
                F.log_softmax(student_scores / float(cfg.get("teacher_temperature", 2.0)), dim=1),
                teacher_prob,
                reduction="batchmean",
            ) * (float(cfg.get("teacher_temperature", 2.0)) ** 2)
            pos = student_scores.gather(1, target_t.view(-1, 1)).squeeze(1)
            neg = student_scores.masked_fill(F.one_hot(target_t, num_classes=cand_n).bool(), -1e4).max(dim=1).values
            l_pair = F.relu(1.0 - pos + neg).mean()
            loss = float(cfg.get("lambda_gt_rank", 1.0)) * l_gt + float(cfg.get("lambda_teacher_kl", 1.5)) * l_kl + float(cfg.get("lambda_teacher_pair", 0.6)) * l_pair
            (loss / grad_accum).backward()
            if ((steps + 1) % grad_accum == 0) or (offset + batch_size >= len(order)):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            for name, val in {"L_gt_rank": l_gt, "L_teacher_KL": l_kl, "L_teacher_pair": l_pair, "L_total": loss}.items():
                loss_acc[name] = loss_acc.get(name, 0.0) + float(val.detach().cpu().item())
            steps += 1
            if steps % 100 == 0:
                print(f"C28D retriever epoch {epoch} step {steps}: loss={float(loss.detach().cpu().item()):.4f}", flush=True)
        avg = {k: v / max(1, steps) for k, v in loss_acc.items()}
        query_count = max(1, len(rows))
        epoch_rec = {
            "epoch": epoch,
            "loss": avg,
            "curriculum": {"teacher_topk": teacher_n, "student_topk": student_n, "random_negative_k": random_n},
            "model_only_train_dynamic_recall": {f"VR@{k}": 100.0 * v / query_count for k, v in recall_hits.items()},
        }
        pre_epoch_vr100 = float(epoch_rec["model_only_train_dynamic_recall"].get("VR@100", 0.0))
        if last_epoch_path is not None and last_epoch_path.exists() and pre_epoch_vr100 > best_score:
            best_score = pre_epoch_vr100
            best_path = str(best_ckpt_path)
            shutil.copy2(last_epoch_path, best_ckpt_path)
            best_manifest = {
                "path": str(best_ckpt_path),
                "selection_metric": "pre_epoch_model_only_train_VR@100",
                "selection_score": best_score,
                "source_checkpoint": str(last_epoch_path),
                "source_epoch": max(epoch - 1, 0),
            }
        logs.append(epoch_rec)
        print(f"C28D retriever epoch {epoch} complete: {json.dumps(epoch_rec, sort_keys=True)}", flush=True)
        latest_manifest = save_checkpoint(ckpt_path, model, optimizer, epoch, {"train_loss": avg, "curriculum": epoch_rec["curriculum"]})
        epoch_path = TMP_ROOT / "checkpoints" / f"C28D_RETRIEVER_{cfg.get('mode','medium')}_seed{seed}.epoch{epoch}.pt"
        save_checkpoint(epoch_path, model, optimizer, epoch, {"train_loss": avg, "curriculum": epoch_rec["curriculum"]})
        last_epoch_path = epoch_path
        if best_path is None:
            best_path = str(ckpt_path)
            best_manifest = latest_manifest
        write_json(out_dir / "C28D_2_RETRIEVER_TRAINING_LOG.json", {"status": "running", "epochs": logs, "checkpoint": str(ckpt_path), "best_checkpoint_path": best_path, "best_train_vr100": best_score if best_score > -math.inf else None})
    rec = {
        "status": "C28D_RETRIEVER_DISTILLATION_TRAINED",
        "device": str(device),
        "checkpoint_manifest": best_manifest,
        "checkpoint_path": best_path or str(ckpt_path),
        "latest_checkpoint_path": str(ckpt_path),
        "best_checkpoint_path": best_path or str(ckpt_path),
        "best_train_vr100": best_score if best_score > -math.inf else None,
        "training_log": logs,
    }
    write_json(out_dir / "C28D_2_RETRIEVER_TRAINING_LOG.json", rec)
    return rec
