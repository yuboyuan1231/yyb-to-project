#!/usr/bin/env python
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rlem.export_conquer_evidence import build_dataset, load_saved_options_if_any, setup_model  # noqa: E402
from rlem_c7.run_c7_b0_b1 import atomic_json, atomic_npz, atomic_text, load_npz  # noqa: E402
from rlem_c7.run_c7_b2_1_safety_repair import group_score_map, normalize_cand  # noqa: E402
from utils.model_utils import move_cuda, start_end_collate  # noqa: E402


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def md(title: str, obj: Any) -> str:
    return f"# {title}\n\n```json\n{json.dumps(obj, indent=2, ensure_ascii=False)}\n```\n"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_config", default="config/tvr_data_config.json")
    p.add_argument("--model_dir", default="tvr-conquer_c0_repro_20260621")
    p.add_argument("--ckpt_filepath", default=None)
    p.add_argument("--eval_split_name", default="train", choices=["train", "val"])
    p.add_argument("--stage_label", default="C7_B5_1")
    p.add_argument("--cache", default="results/rlem_c6a/cache/train_calib_c6_cache.npz")
    p.add_argument("--anchor_states", default="results/rlem_c7_b2/train_calib_c7_b1_anchor_states.json")
    p.add_argument("--video_scores", default="results/rlem_c7_b2/train_calib_video_scores.npz")
    p.add_argument("--b2_eval", default="c7_audit/C7_B2_TRAIN_CALIB_EVAL.json")
    p.add_argument("--existing_evidence_jsonl_gz", default="results/rlem_c3_minimal/train_calib_evidence.jsonl.gz")
    p.add_argument("--output_npz", default="results/rlem_c7_b5_1/train_calib_fixed_pool_raw_logits.npz")
    p.add_argument("--audit_json", default="c7_audit/C7_B5_1_FIXED_POOL_RAW_LOGIT_EXPORT.json")
    p.add_argument("--audit_md", default="c7_audit/C7_B5_1_FIXED_POOL_RAW_LOGIT_EXPORT.md")
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--eval_query_bsz", type=int, default=8)
    p.add_argument("--data_ratio", type=float, default=1.0)
    p.add_argument("--max_ctx_len", type=int, default=100)
    p.add_argument("--max_desc_len", type=int, default=30)
    p.add_argument("--clip_length", type=float, default=1.5)
    p.add_argument("--ctx_mode", default="visual_sub")
    p.add_argument("--visual_dim", type=int, default=4352)
    p.add_argument("--text_dim", type=int, default=768)
    p.add_argument("--query_dim", type=int, default=768)
    p.add_argument("--hidden_dim", type=int, default=768)
    p.add_argument("--no_output_moe_weight", action="store_true")
    p.add_argument("--similarity_measure", default="general", choices=["general", "exclusive", "disjoint"])
    p.add_argument("--max_vcmr_video", type=int, default=10)
    p.add_argument("--min_pred_l", type=int, default=0)
    p.add_argument("--max_pred_l", type=int, default=24)
    p.add_argument("--debug", action="store_true")
    args = p.parse_args()
    saved = load_saved_options_if_any(args)
    for key in [
        "max_ctx_len", "max_desc_len", "clip_length", "ctx_mode", "visual_dim", "text_dim",
        "query_dim", "hidden_dim", "no_output_moe_weight", "similarity_measure",
        "min_pred_l", "max_pred_l", "max_vcmr_video",
    ]:
        if key in saved and getattr(args, key) == p.get_default(key):
            setattr(args, key, saved[key])
    return args


def strict_key(desc_id: Any, c: Tuple[int, int, int, int, float, int]) -> Tuple[str, int, int, int, int, int]:
    return str(desc_id), int(c[1]), int(c[2]), int(c[3]), int(c[5]), int(c[0])


def build_fixed_pool(args: argparse.Namespace) -> Dict[str, Any]:
    cache = load_npz(args.cache, allow_pickle=True)
    states = json.loads(Path(args.anchor_states).read_text(encoding="utf-8"))["states"]
    cfg = json.loads(Path(args.b2_eval).read_text(encoding="utf-8"))["best"]["config"]
    gscore = group_score_map(load_npz(args.video_scores, allow_pickle=False), cfg)
    rows: List[Dict[str, Any]] = []
    for q, state in enumerate(states):
        desc_id = cache["desc_ids"][q].item() if hasattr(cache["desc_ids"][q], "item") else cache["desc_ids"][q]
        for rank, x in enumerate(state["flat_candidates"][:100], 1):
            c = normalize_cand(x)
            residual = float(cfg["video_weight"]) * gscore.get(int(c[0]), (0.0, 0.0, 0.0, 0.0))[2]
            rows.append({
                "query_index": q,
                "desc_id": str(desc_id),
                "group_id": int(c[0]),
                "video_idx": int(c[1]),
                "start_idx": int(c[2]),
                "end_idx": int(c[3]),
                "candidate_row_id": int(c[5]),
                "fixed_pool_rank": int(rank),
                "anchor_score": float(c[4]),
                "b21_residual": float(residual),
                "strict_key": strict_key(desc_id, c),
            })
    return {"cache": cache, "rows": rows}


def fill_from_existing_evidence(args: argparse.Namespace, fixed: Dict[str, Any], arrays: Dict[str, np.ndarray]) -> Dict[str, Any]:
    evidence = Path(args.existing_evidence_jsonl_gz)
    if not evidence.exists():
        return {"source": str(evidence), "available": False, "strict_filled": 0}
    cache = fixed["cache"]
    rows = fixed["rows"]
    key_to_ridx = {tuple(row["strict_key"]): i for i, row in enumerate(rows)}
    needed_row_ids = {int(row["candidate_row_id"]) for row in rows}
    row_q = cache["query_index"].astype(np.int64)
    row_gid = cache["row_group_id"].astype(np.int64)
    row_desc = cache["desc_ids"]
    strict_filled = 0
    bad_identity_candidate_row_ids = 0
    with gzip.open(evidence, "rt", encoding="utf-8") as f:
        for i, line in enumerate(f):
            row = json.loads(line)
            q = int(row_q[i])
            desc_id = row_desc[q].item() if hasattr(row_desc[q], "item") else row_desc[q]
            key = (str(desc_id), int(row["video_idx"]), int(row["start_idx"]), int(row["end_idx"]), int(i), int(row_gid[i]))
            ridx = key_to_ridx.get(key)
            if ridx is None:
                if i in needed_row_ids:
                    bad_identity_candidate_row_ids += 1
                continue
            arrays["b_start"][ridx] = float(row["b_start_logit"])
            arrays["e_end"][ridx] = float(row["e_end_logit"])
            arrays["p_b"][ridx] = float(row.get("p_b_i", np.nan))
            arrays["p_e"][ridx] = float(row.get("p_e_j", np.nan))
            arrays["r1"][ridx] = float(row.get("r1", np.nan))
            if "video_slot" in arrays and row.get("rank_r1") is not None:
                arrays["video_slot"][ridx] = int(row["rank_r1"]) - 1
            strict_filled += 1
    return {
        "source": str(evidence),
        "available": True,
        "strict_filled": int(strict_filled),
        "bad_identity_candidate_row_ids": int(bad_identity_candidate_row_ids),
    }


def filter_dataset_to_train_calib(dataset: Any, desc_ids: List[str]) -> Dict[str, Any]:
    by_desc = {str(row["desc_id"]): row for row in dataset.query_data}
    missing = [d for d in desc_ids if d not in by_desc]
    if missing:
        raise ValueError(f"train_calib desc_ids missing from train dataset: {missing[:5]} count={len(missing)}")
    dataset.query_data = [by_desc[d] for d in desc_ids]
    return {"source_train_queries": len(by_desc), "filtered_train_calib_queries": len(dataset.query_data), "missing_desc_ids": 0}


def main() -> None:
    args = parse_args()
    Path(args.output_npz).parent.mkdir(parents=True, exist_ok=True)
    Path(args.audit_json).parent.mkdir(parents=True, exist_ok=True)
    fixed = build_fixed_pool(args)
    rows = fixed["rows"]
    desc_ids = [str(x.item() if hasattr(x, "item") else x) for x in fixed["cache"]["desc_ids"]]
    device = torch.device(f"cuda:{args.device}" if args.device >= 0 and torch.cuda.is_available() else "cpu")
    n = len(rows)
    b_start = np.full(n, np.nan, dtype=np.float32)
    e_end = np.full(n, np.nan, dtype=np.float32)
    p_b = np.full(n, np.nan, dtype=np.float32)
    p_e = np.full(n, np.nan, dtype=np.float32)
    r1 = np.full(n, np.nan, dtype=np.float32)
    video_slot = np.full(n, -1, dtype=np.int16)
    arrays = {"b_start": b_start, "e_end": e_end, "p_b": p_b, "p_e": p_e, "r1": r1, "video_slot": video_slot}
    evidence_fill_info = fill_from_existing_evidence(args, fixed, arrays)
    missing_video_slot = 0
    out_of_bounds = 0
    desc_to_q = {d: i for i, d in enumerate(desc_ids)}
    by_q: Dict[int, List[int]] = {}
    for i, row in enumerate(rows):
        if not np.isfinite(b_start[i]):
            by_q.setdefault(int(row["query_index"]), []).append(i)
    missing_query_indices = sorted(by_q.keys())
    missing_desc_ids = [desc_ids[q] for q in missing_query_indices]
    filter_info = {"source_train_queries": None, "filtered_train_calib_queries": 0, "missing_desc_ids": 0}
    if missing_query_indices:
        dataset = build_dataset(args)
        filter_info = filter_dataset_to_train_calib(dataset, missing_desc_ids)
        loader = DataLoader(
            dataset,
            collate_fn=start_end_collate,
            batch_size=args.eval_query_bsz,
            num_workers=args.num_workers,
            shuffle=False,
            pin_memory=(device.type == "cuda"),
        )
        model = setup_model(args, device)
    else:
        dataset = None
        loader = []
        model = None
    with torch.no_grad():
        for batch in tqdm(loader, total=len(loader), desc=f"{args.stage_label} missing fixed-pool raw logits"):
            metas = batch["meta"]
            model_inputs = move_cuda(batch["model_inputs"], device) if device.type == "cuda" else batch["model_inputs"]
            _video_similarity_score, begin_logits, end_logits, _extras = model.get_pred_from_raw_query(model_inputs, return_intermediates=True)
            topk = int(args.max_vcmr_video)
            v_begin = begin_logits[:, 1:topk + 1].float()
            v_end = end_logits[:, 1:topk + 1].float()
            st_probs = F.softmax(v_begin, dim=-1)
            ed_probs = F.softmax(v_end, dim=-1)
            vr = model_inputs["inference_vr_scores"].detach().cpu().numpy()[:, :topk]
            v_begin_np = v_begin.detach().cpu().numpy()
            v_end_np = v_end.detach().cpu().numpy()
            st_probs_np = st_probs.detach().cpu().numpy()
            ed_probs_np = ed_probs.detach().cpu().numpy()
            for bi, meta in enumerate(metas):
                q = desc_to_q[str(meta["desc_id"])]
                slot_by_video = {int(dataset.video2idx[name]): j for j, name in enumerate(meta["sample_vid_name_list"][:topk])}
                for ridx in by_q.get(q, []):
                    row = rows[ridx]
                    slot = slot_by_video.get(int(row["video_idx"]), None)
                    if slot is None:
                        missing_video_slot += 1
                        continue
                    si, ei = int(row["start_idx"]), int(row["end_idx"])
                    if si < 0 or ei < 0 or si >= v_begin_np.shape[2] or ei >= v_end_np.shape[2]:
                        out_of_bounds += 1
                        continue
                    video_slot[ridx] = int(slot)
                    b_start[ridx] = float(v_begin_np[bi, slot, si])
                    e_end[ridx] = float(v_end_np[bi, slot, ei])
                    p_b[ridx] = float(st_probs_np[bi, slot, si])
                    p_e[ridx] = float(ed_probs_np[bi, slot, ei])
                    r1[ridx] = float(vr[bi, slot])
            if args.debug:
                break
    matched = int(np.isfinite(b_start).sum())
    query_index = np.asarray([r["query_index"] for r in rows], dtype=np.int32)
    group_id = np.asarray([r["group_id"] for r in rows], dtype=np.int32)
    video_idx = np.asarray([r["video_idx"] for r in rows], dtype=np.int32)
    start_idx = np.asarray([r["start_idx"] for r in rows], dtype=np.int16)
    end_idx = np.asarray([r["end_idx"] for r in rows], dtype=np.int16)
    candidate_row_id = np.asarray([r["candidate_row_id"] for r in rows], dtype=np.int64)
    fixed_pool_rank = np.asarray([r["fixed_pool_rank"] for r in rows], dtype=np.int16)
    anchor_score = np.asarray([r["anchor_score"] for r in rows], dtype=np.float32)
    b21_residual = np.asarray([r["b21_residual"] for r in rows], dtype=np.float32)
    l_logit = (b_start + e_end).astype(np.float32)
    l_prob = (np.log(np.maximum(p_b, 1e-12)) + np.log(np.maximum(p_e, 1e-12))).astype(np.float32)
    tmp = Path(str(args.output_npz) + ".partial.npz")
    np.savez_compressed(
        tmp,
        query_index=query_index,
        group_id=group_id,
        video_idx=video_idx,
        start_idx=start_idx,
        end_idx=end_idx,
        candidate_row_id=candidate_row_id,
        fixed_pool_rank=fixed_pool_rank,
        video_slot=video_slot,
        anchor_score=anchor_score,
        b21_residual=b21_residual,
        b_start_logit=b_start,
        e_end_logit=e_end,
        l_logit=l_logit,
        p_b_i=p_b,
        p_e_j=p_e,
        l_prob=l_prob,
        r1=r1,
    )
    os.replace(tmp, args.output_npz)
    audit = {
        "status": f"{args.stage_label}_FIXED_POOL_RAW_LOGIT_EXPORT_PASS" if matched == n and missing_video_slot == 0 and out_of_bounds == 0 else f"{args.stage_label}_FIXED_POOL_RAW_LOGIT_EXPORT_INCOMPLETE",
        "official_val_used": False,
        "post_val_adjustment": False,
        "second_official_val": False,
        "scope": "train_calib fixed-pool only",
        "method": "frozen CONQUER begin/end logits indexed directly by fixed-pool candidate video/start/end",
        "filter_info": filter_info,
        "existing_evidence_fill_info": evidence_fill_info,
        "missing_query_count_for_forward": int(len(missing_query_indices)),
        "total_candidates": int(n),
        "matched_candidates": int(matched),
        "missing_candidates": int(n - matched),
        "missing_video_slot": int(missing_video_slot),
        "out_of_bounds": int(out_of_bounds),
        "strict_key_fields": ["desc_id", "video_idx", "start_idx", "end_idx", "candidate_row_id", "group_id"],
        "position_based_join": False,
        "NMS_modified": False,
        "evaluator_modified": False,
        "CONQUER_frozen": True,
        "output_npz": {"path": args.output_npz, "sha256": sha256_file(args.output_npz)},
        "runner": {"path": __file__, "sha256": sha256_file(__file__)},
    }
    atomic_json(args.audit_json, audit)
    atomic_text(args.audit_md, md(f"{args.stage_label} fixed-pool raw logit export", audit))
    print(json.dumps(audit, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
