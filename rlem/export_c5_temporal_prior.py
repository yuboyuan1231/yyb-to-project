#!/usr/bin/env python
"""Read-only C5-0 export of real CONQUER temporal priors.

Exports the frozen ML-head begin/end probabilities and QAL query-to-video
attention for exactly the (query, video) groups present in a C4 cache.  It does
not generate candidates, train a model, read official val, or alter CONQUER.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rlem.export_conquer_evidence import (  # noqa: E402
    build_dataset,
    load_saved_options_if_any,
    resolve_ckpt,
    setup_model,
)
from rlem.c4_lite_utils import write_json  # noqa: E402
from utils.model_utils import move_cuda, start_end_collate  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cache_npz", required=True)
    p.add_argument("--output_npz", required=True)
    p.add_argument("--output_audit_json", required=True)
    p.add_argument("--split_role", choices=["train_fit", "train_calib", "official_val"], required=True)
    p.add_argument("--dataset_config", default="config/tvr_data_config.json")
    p.add_argument("--model_config", default="config/model_config.json")
    p.add_argument("--model_dir", default="tvr-conquer_general_paper_performance")
    p.add_argument("--ckpt_filepath", default=None)
    p.add_argument("--eval_split_name", choices=["train", "val"], default="train")
    p.add_argument("--allow_official_val", action="store_true")
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--eval_query_bsz", type=int, default=5)
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
    p.add_argument("--data_ratio", type=float, default=1.0)
    p.add_argument("--debug", action="store_true")
    p.add_argument("--work_dir", default="/tmp/c5_temporal_export_work")
    p.add_argument("--compressed", action="store_true", help="Compress final NPZ; slower but smaller.")
    args = p.parse_args()
    saved = load_saved_options_if_any(args)
    for key in [
        "max_ctx_len", "max_desc_len", "clip_length", "ctx_mode", "visual_dim",
        "text_dim", "query_dim", "hidden_dim", "no_output_moe_weight",
        "similarity_measure", "max_vcmr_video",
    ]:
        if key in saved and getattr(args, key) == p.get_default(key):
            setattr(args, key, saved[key])
    return args


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def masked_probability(logits, mask):
    prob = F.softmax(logits.float(), dim=-1)
    prob = prob * mask.float()
    return prob / prob.sum(dim=-1, keepdim=True).clamp_min(1e-12)


def atomic_npz(path, compressed, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = Path(str(path) + ".partial")
    with open(partial, "wb") as f:
        if compressed:
            np.savez_compressed(f, **arrays)
        else:
            np.savez(f, **arrays)
    os.replace(partial, path)


def main():
    args = parse_args()
    if args.split_role == "official_val":
        if not args.allow_official_val or args.eval_split_name != "val":
            raise PermissionError(
                "official_val temporal export requires --allow_official_val and --eval_split_name val"
            )
    elif args.eval_split_name != "train":
        raise PermissionError("train_fit/train_calib temporal exports must use --eval_split_name train")
    for path in (args.output_npz, args.output_audit_json):
        if Path(path).exists():
            raise FileExistsError(path)
    with np.load(args.cache_npz, allow_pickle=True) as cache:
        desc_ids = np.asarray(cache["desc_ids"])
        group_ids = cache["group_ids_sorted_unique"].astype(np.int64)
        group_keys = cache["video_group_keys"].astype(np.int64)
    query_count = len(desc_ids)
    group_count = len(group_ids)
    if not np.array_equal(group_ids, np.arange(group_count, dtype=np.int64)):
        raise ValueError("C5 exporter requires dense ordered C4 group IDs")
    if group_keys.shape != (group_count, 2):
        raise ValueError("C4 video_group_keys shape mismatch")
    group_query = group_keys[:, 0]
    counts = np.bincount(group_query, minlength=query_count)
    starts = np.r_[0, np.cumsum(counts)]
    if int(starts[-1]) != group_count or np.any(group_query[1:] < group_query[:-1]):
        raise ValueError("C4 video groups are not query-sorted")

    device = torch.device(f"cuda:{args.device}" if args.device >= 0 and torch.cuda.is_available() else "cpu")
    dataset = build_dataset(args)
    raw_by_desc = {str(row["desc_id"]): row for row in dataset.query_data}
    missing = [str(value) for value in desc_ids if str(value) not in raw_by_desc]
    if missing:
        raise ValueError(f"Cache desc IDs missing from {args.eval_split_name} dataset: {len(missing)}")
    dataset.query_data = [raw_by_desc[str(value)] for value in desc_ids]
    desc_to_query = {str(value): i for i, value in enumerate(desc_ids)}
    loader_kwargs = dict(
        dataset=dataset,
        collate_fn=start_end_collate,
        batch_size=args.eval_query_bsz,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=(device.type == "cuda"),
    )
    if args.num_workers > 0:
        loader_kwargs.update(persistent_workers=True, prefetch_factor=2)
    loader = DataLoader(**loader_kwargs)
    model = setup_model(args, device)

    work = Path(args.work_dir) / args.split_role
    if work.exists():
        raise FileExistsError(f"Stale C5 export work directory: {work}")
    work.mkdir(parents=True, exist_ok=False)
    shape = (group_count, int(args.max_ctx_len))
    p_ctx = np.lib.format.open_memmap(work / "p_ctx.npy", mode="w+", dtype=np.float32, shape=shape)
    p_b = np.lib.format.open_memmap(work / "p_b.npy", mode="w+", dtype=np.float32, shape=shape)
    p_e = np.lib.format.open_memmap(work / "p_e.npy", mode="w+", dtype=np.float32, shape=shape)
    temporal_length = np.lib.format.open_memmap(work / "temporal_length.npy", mode="w+", dtype=np.int16, shape=(group_count,))
    written = np.zeros(group_count, dtype=np.uint8)

    with torch.inference_mode():
        for batch in tqdm(loader, total=len(loader), desc=f"C5-0 {args.split_role} temporal export"):
            model_inputs = move_cuda(batch["model_inputs"], device) if device.type == "cuda" else batch["model_inputs"]
            video_score, begin_logits, end_logits, extras = model.get_pred_from_raw_query(
                model_inputs, return_intermediates=True
            )
            del video_score
            qbs, shared, temporal_len = begin_logits.shape
            if shared < args.max_vcmr_video + 1 or temporal_len != args.max_ctx_len:
                raise ValueError("Unexpected CONQUER temporal output shape")
            mask = extras["video_mask"].view(qbs, shared, temporal_len)[:, 1:args.max_vcmr_video + 1]
            pb = masked_probability(begin_logits[:, 1:args.max_vcmr_video + 1], mask).cpu().numpy()
            pe = masked_probability(end_logits[:, 1:args.max_vcmr_video + 1], mask).cpu().numpy()
            qal_aux = extras.get("qal_aux")
            if qal_aux is None or qal_aux.get("q2v_attention") is None:
                raise RuntimeError(
                    "Missing model intermediate qal_aux.q2v_attention; cannot export real P_ctx."
                )
            ctx = qal_aux["q2v_attention"].view(qbs, shared, temporal_len)[:, 1:args.max_vcmr_video + 1]
            ctx = ctx.float() * mask.float()
            ctx = ctx / ctx.sum(dim=-1, keepdim=True).clamp_min(1e-12)
            ctx = ctx.cpu().numpy()
            lengths = mask.sum(dim=-1).cpu().numpy().astype(np.int16)

            for local_q, meta in enumerate(batch["meta"]):
                query_index = desc_to_query[str(meta["desc_id"])]
                s, e = int(starts[query_index]), int(starts[query_index + 1])
                expected = {int(group_keys[pos, 1]): pos for pos in range(s, e)}
                for local_v, video_name in enumerate(meta["sample_vid_name_list"][:args.max_vcmr_video]):
                    video_idx = int(dataset.video2idx[video_name])
                    pos = expected.get(video_idx)
                    if pos is None:
                        continue
                    if written[pos]:
                        raise ValueError(f"Duplicate temporal export for group_id={pos}")
                    p_ctx[pos] = ctx[local_q, local_v]
                    p_b[pos] = pb[local_q, local_v]
                    p_e[pos] = pe[local_q, local_v]
                    temporal_length[pos] = lengths[local_q, local_v]
                    written[pos] = 1
            if args.debug:
                break

    missing_groups = int(np.sum(written == 0))
    if missing_groups:
        raise ValueError(f"C5 temporal export missing {missing_groups} cache video groups")
    for arr in (p_ctx, p_b, p_e, temporal_length):
        arr.flush()
    nonfinite = int((~np.isfinite(p_ctx)).sum() + (~np.isfinite(p_b)).sum() + (~np.isfinite(p_e)).sum())
    if nonfinite:
        raise ValueError(f"Non-finite temporal values: {nonfinite}")
    atomic_npz(
        args.output_npz,
        args.compressed,
        group_id=group_ids,
        p_ctx=p_ctx,
        p_b=p_b,
        p_e=p_e,
        temporal_length=np.asarray(temporal_length),
    )
    checkpoint_path = resolve_ckpt(args)
    audit = {
        "status": "PASS",
        "stage": "C5-0 read-only temporal prior export",
        "split_role": args.split_role,
        "official_val_split_exported": bool(args.split_role == "official_val"),
        "official_val_used_for_selection": False,
        "post_val_adjustment": False,
        "score_grid_on_val": False,
        "queries": int(query_count),
        "video_groups": int(group_count),
        "temporal_shape": [int(group_count), int(args.max_ctx_len)],
        "min_temporal_length": int(np.min(temporal_length)),
        "max_temporal_length": int(np.max(temporal_length)),
        "nonfinite_values": nonfinite,
        "source_tensors": {
            "p_b": "softmax(CONQUER moment_localization_head begin_score_distribution)",
            "p_e": "softmax(CONQUER moment_localization_head end_score_distribution)",
            "p_ctx": "QAL qal_aux.q2v_attention",
        },
        "label_used_for_prior": False,
        "candidate_generation_modified": False,
        "model_conquer_modified": False,
        "artifacts": {
            "cache_npz": {"path": args.cache_npz, "sha256": sha256(args.cache_npz)},
            "checkpoint": {"path": checkpoint_path, "sha256": sha256(checkpoint_path)},
            "output_npz": {"path": args.output_npz, "sha256": sha256(args.output_npz)},
        },
        "runtime": {
            "device": str(device),
            "num_workers": args.num_workers,
            "eval_query_bsz": args.eval_query_bsz,
            "compressed": bool(args.compressed),
            "work_dir": str(work),
        },
        "used": {
            "C5_lite_prior": True,
            "C5_main": bool(args.split_role == "official_val"),
            "C5_main_B": False,
            "candidate_regeneration": False,
            "C6": False,
        },
    }
    write_json(args.output_audit_json, audit)
    if work.exists():
        shutil.rmtree(work)
    print(json.dumps({"status": "PASS", "output_npz": args.output_npz}, indent=2))


if __name__ == "__main__":
    main()
