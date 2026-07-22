from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from blueprint_e2e_v2.data.dynamic_candidate_miner import CandidateSet, CompactCandidateStore
from blueprint_e2e_v2.data.first_stage_reference import FirstStageReference
from blueprint_e2e_v2.data.split_manager import SplitManager


def _safe_soft_topk(values: torch.Tensor, k: int, temperature: float, mask: torch.Tensor | None = None) -> torch.Tensor:
    if mask is not None:
        values = values.masked_fill(~mask.bool(), -1e4)
    k = min(int(k), values.shape[-1])
    vals = torch.topk(values, k=k, dim=-1).values
    weights = torch.softmax(vals / max(float(temperature), 1e-6), dim=-1)
    return (weights * vals).sum(dim=-1)


@torch.no_grad()
def _encode_clip_bank_for_refresh(
    core: torch.nn.Module,
    visual_seq_bank: np.ndarray,
    subtitle_seq_bank: np.ndarray,
    visual_seq_mask: np.ndarray | None,
    subtitle_seq_mask: np.ndarray | None,
    device: torch.device,
    chunk_size: int,
) -> dict[str, torch.Tensor]:
    parts: dict[str, list[torch.Tensor]] = {"visual": [], "subtitle": [], "joint": []}
    for st in range(0, visual_seq_bank.shape[0], max(1, int(chunk_size))):
        ed = min(st + max(1, int(chunk_size)), visual_seq_bank.shape[0])
        visual = torch.from_numpy(visual_seq_bank[st:ed].astype(np.float32, copy=False)).to(device, non_blocking=True).unsqueeze(1)
        subtitle = torch.from_numpy(subtitle_seq_bank[st:ed].astype(np.float32, copy=False)).to(device, non_blocking=True).unsqueeze(1)
        vmask = torch.from_numpy(visual_seq_mask[st:ed].astype(np.bool_, copy=False)).to(device, non_blocking=True).unsqueeze(1) if visual_seq_mask is not None else None
        smask = torch.from_numpy(subtitle_seq_mask[st:ed].astype(np.bool_, copy=False)).to(device, non_blocking=True).unsqueeze(1) if subtitle_seq_mask is not None else None
        jmask = torch.logical_and(vmask, smask) if vmask is not None and smask is not None else vmask if vmask is not None else smask
        enc = core.video_encoder(visual, subtitle, visual_mask=vmask, subtitle_mask=smask, clip_mask=jmask)
        for key in parts:
            parts[key].append(enc[key].squeeze(1).detach().cpu().to(torch.float16))
        done = ed
        if done % max(int(chunk_size) * 25, 1) == 0 or done == visual_seq_bank.shape[0]:
            print(f"C28C clip refresh bank: encoded {done}/{visual_seq_bank.shape[0]} videos", flush=True)
    out = {key: torch.cat(vals, dim=0) for key, vals in parts.items()}
    if visual_seq_mask is not None:
        out["visual_mask"] = torch.from_numpy(visual_seq_mask.astype(np.bool_, copy=False))
    if subtitle_seq_mask is not None:
        out["subtitle_mask"] = torch.from_numpy(subtitle_seq_mask.astype(np.bool_, copy=False))
    if visual_seq_mask is not None and subtitle_seq_mask is not None:
        out["joint_mask"] = torch.from_numpy(np.logical_and(visual_seq_mask, subtitle_seq_mask).astype(np.bool_, copy=False))
    return out


@torch.no_grad()
def _late_scores_for_candidate_indices(
    core: torch.nn.Module,
    q: dict[str, torch.Tensor],
    query_tokens: torch.Tensor,
    query_mask: torch.Tensor,
    candidate_indices: torch.Tensor,
    pooled_bank: dict[str, torch.Tensor],
    clip_bank: dict[str, torch.Tensor],
    device: torch.device,
    candidate_encode_chunk: int,
    late_soft_topk: int,
    late_temperature: float,
    token_maxsim_weight: float,
    pooled_score_weight: float,
    late_score_weight: float,
) -> torch.Tensor:
    bsz, cand_n = candidate_indices.shape
    out_parts: list[torch.Tensor] = []
    token_h = None
    qmask_f = None
    if float(token_maxsim_weight) > 0.0:
        token_h = F.normalize(core.query_encoder.joint_proj(query_tokens.to(device, non_blocking=True)), dim=-1)
        qmask_f = query_mask.to(device, non_blocking=True).float().unsqueeze(-1)
    for cs in range(0, cand_n, max(1, int(candidate_encode_chunk))):
        sub_idx = candidate_indices[:, cs: cs + max(1, int(candidate_encode_chunk))]
        sub_n = sub_idx.shape[1]
        flat_cpu = sub_idx.detach().cpu().reshape(-1)
        flat_device = sub_idx.reshape(-1)
        pooled = {key: val.index_select(0, flat_device).reshape(bsz, sub_n, -1) for key, val in pooled_bank.items()}
        pooled_score = core.retriever.score_candidates(q, pooled)["retriever_score"]
        visual = clip_bank["visual"].index_select(0, flat_cpu).to(device, non_blocking=True).float().reshape(bsz, sub_n, -1, q["q_visual"].shape[-1])
        subtitle = clip_bank["subtitle"].index_select(0, flat_cpu).to(device, non_blocking=True).float().reshape(bsz, sub_n, -1, q["q_subtitle"].shape[-1])
        joint = clip_bank["joint"].index_select(0, flat_cpu).to(device, non_blocking=True).float().reshape(bsz, sub_n, -1, q["q_joint"].shape[-1])
        visual_mask_src = clip_bank.get("visual_mask")
        subtitle_mask_src = clip_bank.get("subtitle_mask")
        joint_mask_src = clip_bank.get("joint_mask")
        visual_mask = visual_mask_src.index_select(0, flat_cpu).to(device, non_blocking=True).reshape(bsz, sub_n, -1) if isinstance(visual_mask_src, torch.Tensor) else None
        subtitle_mask = subtitle_mask_src.index_select(0, flat_cpu).to(device, non_blocking=True).reshape(bsz, sub_n, -1) if isinstance(subtitle_mask_src, torch.Tensor) else None
        joint_mask = joint_mask_src.index_select(0, flat_cpu).to(device, non_blocking=True).reshape(bsz, sub_n, -1) if isinstance(joint_mask_src, torch.Tensor) else None
        sv_t = torch.einsum("bd,bctd->bct", q["q_visual"], visual)
        ss_t = torch.einsum("bd,bctd->bct", q["q_subtitle"], subtitle)
        sj_t = torch.einsum("bd,bctd->bct", q["q_joint"], joint)
        sv = _safe_soft_topk(sv_t, late_soft_topk, late_temperature, visual_mask)
        ss = _safe_soft_topk(ss_t, late_soft_topk, late_temperature, subtitle_mask)
        sj = _safe_soft_topk(sj_t, late_soft_topk, late_temperature, joint_mask)
        gate = q["gate"]
        late_score = core.retriever.scale.clamp(1.0, 30.0) * (gate[:, 0:1] * sv + gate[:, 1:2] * ss + gate[:, 2:3] * sj)
        token_score = late_score.new_zeros(late_score.shape)
        if token_h is not None and qmask_f is not None:
            token_sim = torch.einsum("bld,bctd->blct", token_h, joint)
            if joint_mask is not None:
                token_sim = token_sim.masked_fill(~joint_mask.bool().unsqueeze(1), -1e4)
            token_sim = token_sim.max(dim=-1).values
            token_score = (token_sim * qmask_f).sum(dim=1) / qmask_f.sum(dim=1).clamp_min(1.0)
            token_score = core.retriever.scale.clamp(1.0, 30.0) * token_score
        out_parts.append(float(pooled_score_weight) * pooled_score + float(late_score_weight) * late_score + float(token_maxsim_weight) * token_score)
    return torch.cat(out_parts, dim=1)


def _pad_query_tokens(rows: list[dict[str, Any]], query_cache: dict[int, Any]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[int], list[str]]:
    arrays = [query_cache[int(r["desc_id"])] for r in rows]
    max_len = max(a.shape[0] for a in arrays)
    dim = arrays[0].shape[1]
    tokens = np.zeros((len(arrays), max_len, dim), dtype=np.float32)
    mask = np.zeros((len(arrays), max_len), dtype=np.bool_)
    qtypes = []
    for i, (a, r) in enumerate(zip(arrays, rows)):
        tokens[i, : a.shape[0]] = a
        mask[i, : a.shape[0]] = True
        qtypes.append({"v": 0, "t": 1, "vt": 2}.get(str(r.get("type", "")), 3))
    return torch.from_numpy(tokens), torch.from_numpy(mask), torch.tensor(qtypes, dtype=torch.long), [int(r["desc_id"]) for r in rows], [str(r["vid_name"]) for r in rows]


def refresh_candidates(
    model: torch.nn.Module,
    split_manager: SplitManager,
    query_bank: Any,
    video_ids: list[str],
    video_to_idx: dict[str, int],
    visual_bank: torch.Tensor,
    subtitle_bank: torch.Tensor,
    split: str,
    max_queries: int | None,
    dynamic_topk: int,
    chunk_size: int,
    device: torch.device,
    batch_queries: int = 64,
    insert_gt_for_training: bool = True,
    teacher_warm_topk: int = 64,
    visual_seq_bank: np.ndarray | None = None,
    subtitle_seq_bank: np.ndarray | None = None,
    visual_seq_mask: np.ndarray | None = None,
    subtitle_seq_mask: np.ndarray | None = None,
    late_candidate_mining: bool = False,
    broad_topk: int | None = None,
    candidate_encode_chunk: int = 32,
    clip_bank_encode_chunk: int = 128,
    late_soft_topk: int = 8,
    late_temperature: float = 0.07,
    token_maxsim_weight: float = 0.0,
    pooled_score_weight: float = 1.0,
    late_score_weight: float = 0.0,
) -> tuple[dict[int, CandidateSet] | CompactCandidateStore, dict[str, Any]]:
    rows = split_manager.records(split, max_queries=max_queries)
    core = model.module if hasattr(model, "module") else model
    query_cache = query_bank.bulk_tokens([int(r["desc_id"]) for r in rows])
    core.eval()
    bank_parts: dict[str, list[torch.Tensor]] = {"visual_pool": [], "subtitle_pool": [], "joint_pool": []}
    with torch.no_grad():
        for st in range(0, visual_bank.shape[0], chunk_size):
            enc = core.video_encoder.encode_pooled_bank(
                visual_bank[st: st + chunk_size].to(device, non_blocking=True),
                subtitle_bank[st: st + chunk_size].to(device, non_blocking=True),
            )
            for key in bank_parts:
                bank_parts[key].append(enc[key].detach())
        bank = {k: torch.cat(v, dim=0) for k, v in bank_parts.items()}
    out_query_ids: list[int] = []
    out_video_indices: list[np.ndarray] = []
    out_scores: list[np.ndarray] = []
    out_gt_indices: list[int] = []
    out_gt_inserted: list[bool] = []
    inserted = 0
    teacher_ref = FirstStageReference() if insert_gt_for_training and teacher_warm_topk > 0 else None
    teacher_cache = teacher_ref.bulk_top_indices([int(r["desc_id"]) for r in rows], video_to_idx, k=min(int(teacher_warm_topk), max(0, int(dynamic_topk) - 1))) if teacher_ref is not None else {}
    teacher_used = 0
    late_mining_used = bool(
        late_candidate_mining
        and visual_seq_bank is not None
        and subtitle_seq_bank is not None
        and bool(getattr(core, "late_interaction_enabled", False))
    )
    clip_bank = None
    if late_mining_used:
        clip_bank = _encode_clip_bank_for_refresh(
            core,
            visual_seq_bank,
            subtitle_seq_bank,
            visual_seq_mask,
            subtitle_seq_mask,
            device,
            max(1, int(clip_bank_encode_chunk)),
        )
    broad_k = max(int(dynamic_topk), int(broad_topk or dynamic_topk))
    with torch.no_grad():
        for st in range(0, len(rows), batch_queries):
            toks, qmask, qtypes, qids, gt_vids = _pad_query_tokens(rows[st: st + batch_queries], query_cache)
            q = core.encode_query(toks.to(device, non_blocking=True), qtypes.to(device, non_blocking=True), qmask.to(device, non_blocking=True))
            scores = core.retriever.score_bank(q, bank)
            if late_mining_used:
                _broad_vals, broad_idx = torch.topk(scores, k=min(broad_k, scores.shape[1]), dim=1)
                late_scores = _late_scores_for_candidate_indices(
                    core,
                    q,
                    toks,
                    qmask,
                    broad_idx,
                    bank,
                    clip_bank,
                    device,
                    candidate_encode_chunk,
                    late_soft_topk,
                    late_temperature,
                    token_maxsim_weight,
                    pooled_score_weight,
                    late_score_weight,
                )
                k = min(dynamic_topk, late_scores.shape[1])
                vals, order = torch.topk(late_scores, k=k, dim=1)
                idx = torch.gather(broad_idx, 1, order)
            else:
                k = min(dynamic_topk, scores.shape[1])
                vals, idx = torch.topk(scores, k=k, dim=1)
            scores_cpu = scores.detach().cpu().numpy()
            idx_cpu = idx.detach().cpu().numpy()
            vals_cpu = vals.detach().cpu().numpy()
            for row_i, qid in enumerate(qids):
                vids_arr = idx_cpu[row_i].astype(np.int32, copy=True)
                vals_arr = vals_cpu[row_i].astype(np.float32, copy=True)
                if teacher_ref is not None:
                    teacher = teacher_cache.get(int(qid), [])
                    if teacher:
                        keep_current = max(1, dynamic_topk - len(teacher))
                        merged: list[int] = []
                        seen: set[int] = set()
                        for v in vids_arr[:keep_current].tolist() + teacher + vids_arr[keep_current:].tolist():
                            vi = int(v)
                            if vi not in seen:
                                merged.append(vi)
                                seen.add(vi)
                            if len(merged) >= dynamic_topk:
                                break
                        vids_arr = np.asarray(merged, dtype=np.int32)
                        vals_arr = scores_cpu[row_i, vids_arr].astype(np.float32, copy=True)
                        teacher_used += 1
                gt_idx = video_to_idx[str(gt_vids[row_i])]
                did_insert = False
                if insert_gt_for_training and not bool((vids_arr == int(gt_idx)).any()):
                    vids_arr[-1] = int(gt_idx)
                    vals_arr[-1] = float(scores_cpu[row_i, gt_idx])
                    did_insert = True
                    inserted += 1
                out_query_ids.append(int(qid))
                out_video_indices.append(vids_arr.astype(np.int32, copy=False))
                out_scores.append(vals_arr.astype(np.float32, copy=False))
                out_gt_indices.append(int(gt_idx))
                out_gt_inserted.append(bool(did_insert))
            done = min(st + batch_queries, len(rows))
            if done % max(batch_queries * 25, 1) == 0 or done == len(rows):
                print(f"C28C dynamic candidates: scored {done}/{len(rows)} queries", flush=True)
    if teacher_ref is not None:
        teacher_ref.close()
    out = CompactCandidateStore(
        np.asarray(out_query_ids, dtype=np.int64),
        np.stack(out_video_indices).astype(np.int32, copy=False) if out_video_indices else np.zeros((0, int(dynamic_topk)), dtype=np.int32),
        np.stack(out_scores).astype(np.float32, copy=False) if out_scores else np.zeros((0, int(dynamic_topk)), dtype=np.float32),
        np.asarray(out_gt_indices, dtype=np.int32),
        np.asarray(out_gt_inserted, dtype=np.bool_),
    )
    audit = {
        "query_count": len(out),
        "dynamic_topk": int(dynamic_topk),
        "candidate_count_min": out.candidate_count_min(),
        "candidate_count_max": out.candidate_count_max(),
        "gt_insert_rate": 100.0 * inserted / max(1, len(out)),
        "gt_insert_for_training_loss": bool(insert_gt_for_training),
        "candidate_store": "compact_numpy_matrix_v1",
        "static_top128_hard_gate_used": False,
        "teacher_warm_start_used_for_training": bool(teacher_ref is not None),
        "teacher_warm_start_query_rate": 100.0 * teacher_used / max(1, len(out)),
        "video_bank_encoded_once_per_refresh": True,
        "encoded_bank_device": str(device),
        "late_candidate_mining_requested": bool(late_candidate_mining),
        "late_candidate_mining_used": late_mining_used,
        "clip_bank_preencoded_once_per_refresh": bool(clip_bank is not None),
        "clip_bank_encode_chunk": int(clip_bank_encode_chunk) if late_mining_used else None,
        "broad_topk": broad_k if late_mining_used else int(dynamic_topk),
        "candidate_score_source": "pooled_broad_plus_clip_late_rerank" if late_mining_used else "pooled_retriever_only",
    }
    return out, audit
