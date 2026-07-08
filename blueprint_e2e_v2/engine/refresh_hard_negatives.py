from __future__ import annotations

from typing import Any

import numpy as np
import torch

from blueprint_e2e_v2.data.dynamic_candidate_miner import CandidateSet
from blueprint_e2e_v2.data.first_stage_reference import FirstStageReference
from blueprint_e2e_v2.data.split_manager import SplitManager


def _pad_query_tokens(rows: list[dict[str, Any]], query_cache: dict[int, Any]) -> tuple[torch.Tensor, torch.Tensor, list[int], list[str]]:
    arrays = [query_cache[int(r["desc_id"])] for r in rows]
    max_len = max(a.shape[0] for a in arrays)
    dim = arrays[0].shape[1]
    tokens = np.zeros((len(arrays), max_len, dim), dtype=np.float32)
    qtypes = []
    for i, (a, r) in enumerate(zip(arrays, rows)):
        tokens[i, : a.shape[0]] = a
        qtypes.append({"v": 0, "t": 1, "vt": 2}.get(str(r.get("type", "")), 3))
    return torch.from_numpy(tokens), torch.tensor(qtypes, dtype=torch.long), [int(r["desc_id"]) for r in rows], [str(r["vid_name"]) for r in rows]


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
) -> tuple[dict[int, CandidateSet], dict[str, Any]]:
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
    out: dict[int, CandidateSet] = {}
    inserted = 0
    teacher_ref = FirstStageReference() if insert_gt_for_training and teacher_warm_topk > 0 else None
    teacher_cache = teacher_ref.bulk_top_indices([int(r["desc_id"]) for r in rows], video_to_idx, k=min(teacher_warm_topk, dynamic_topk // 2)) if teacher_ref is not None else {}
    teacher_used = 0
    with torch.no_grad():
        for st in range(0, len(rows), batch_queries):
            toks, qtypes, qids, gt_vids = _pad_query_tokens(rows[st: st + batch_queries], query_cache)
            q = core.encode_query(toks.to(device, non_blocking=True), qtypes.to(device, non_blocking=True))
            scores = core.retriever.score_bank(q, bank)
            k = min(dynamic_topk, scores.shape[1])
            vals, idx = torch.topk(scores, k=k, dim=1)
            scores_cpu = scores.detach().cpu().numpy()
            idx_cpu = idx.detach().cpu().numpy()
            vals_cpu = vals.detach().cpu().numpy()
            for row_i, qid in enumerate(qids):
                vids = idx_cpu[row_i].astype(np.int64, copy=False).tolist()
                vals_row = vals_cpu[row_i].astype(np.float32, copy=False).tolist()
                if teacher_ref is not None:
                    teacher = teacher_cache.get(int(qid), [])
                    if teacher:
                        keep_current = max(1, dynamic_topk - len(teacher))
                        merged: list[int] = []
                        for v in vids[:keep_current] + teacher + vids[keep_current:]:
                            if int(v) not in merged:
                                merged.append(int(v))
                            if len(merged) >= dynamic_topk:
                                break
                        vids = merged
                        vals_row = scores_cpu[row_i, np.asarray(vids, dtype=np.int64)].astype(np.float32, copy=False).tolist()
                        teacher_used += 1
                gt_idx = video_to_idx[str(gt_vids[row_i])]
                did_insert = False
                if insert_gt_for_training and gt_idx not in vids:
                    vids[-1] = gt_idx
                    vals_row[-1] = float(scores_cpu[row_i, gt_idx])
                    did_insert = True
                    inserted += 1
                out[int(qid)] = CandidateSet(int(qid), [int(v) for v in vids], [float(v) for v in vals_row], int(gt_idx), did_insert)
            done = min(st + batch_queries, len(rows))
            if done % max(batch_queries * 25, 1) == 0 or done == len(rows):
                print(f"C28C dynamic candidates: scored {done}/{len(rows)} queries", flush=True)
    if teacher_ref is not None:
        teacher_ref.close()
    audit = {
        "query_count": len(out),
        "dynamic_topk": int(dynamic_topk),
        "candidate_count_min": min((len(c.video_indices) for c in out.values()), default=0),
        "candidate_count_max": max((len(c.video_indices) for c in out.values()), default=0),
        "gt_insert_rate": 100.0 * inserted / max(1, len(out)),
        "gt_insert_for_training_loss": bool(insert_gt_for_training),
        "static_top128_hard_gate_used": False,
        "teacher_warm_start_used_for_training": bool(teacher_ref is not None),
        "teacher_warm_start_query_rate": 100.0 * teacher_used / max(1, len(out)),
        "video_bank_encoded_once_per_refresh": True,
        "encoded_bank_device": str(device),
    }
    return out, audit
