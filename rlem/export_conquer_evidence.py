#!/usr/bin/env python

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

"""Export fixed CONQUER candidates as scalar RLEM evidence JSONL.

This script implements C1 for CONQUER-RLEM:
  frozen CONQUER -> fixed VCMR candidates -> compact (q,v,p)-level evidence rows.

It does not train RLEM and does not change CONQUER ranking/NMS.  The output is
intended for C2/C3 evidence-head training and reranking.
"""

import argparse
import gzip
import json
import os
from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from config.config import BaseOptions
from data_loader.second_stage_start_end_dataset import StartEndDataset
from model.conquer import CONQUER
from utils.basic_utils import load_config, load_json
from utils.model_utils import move_cuda, start_end_collate
from rlem.evidence_utils import (
    EPS,
    entropy,
    local_sharpness,
    make_length_mask,
    query_standardize,
    span_stats_from_prior,
    temporal_iou_seconds,
    top_margin,
)


def open_jsonl(path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if path.endswith(".gz"):
        return gzip.open(path, "wt", encoding="utf-8")
    return open(path, "w", encoding="utf-8")


def resolve_ckpt(args) -> str:
    if args.ckpt_filepath:
        return args.ckpt_filepath
    if not args.model_dir:
        raise ValueError("Either --ckpt_filepath or --model_dir must be provided.")
    model_dir = args.model_dir
    if not os.path.isabs(model_dir) and not model_dir.startswith("results"):
        model_dir = os.path.join("results", model_dir)
    return os.path.join(model_dir, BaseOptions.ckpt_filename)


def load_saved_options_if_any(args) -> Dict:
    if not args.model_dir:
        return {}
    model_dir = args.model_dir
    if not os.path.isabs(model_dir) and not model_dir.startswith("results"):
        model_dir = os.path.join("results", model_dir)
    opt_path = os.path.join(model_dir, BaseOptions.saved_option_filename)
    if os.path.exists(opt_path):
        return load_json(opt_path)
    return {}


def setup_model(args, device):
    ckpt_path = resolve_ckpt(args)
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    loaded_model_cfg = checkpoint["model_cfg"]
    model = CONQUER(
        loaded_model_cfg,
        visual_dim=args.visual_dim,
        text_dim=args.text_dim,
        query_dim=args.query_dim,
        hidden_dim=args.hidden_dim,
        video_len=args.max_ctx_len,
        ctx_mode=args.ctx_mode,
        no_output_moe_weight=args.no_output_moe_weight,
        similarity_measure=args.similarity_measure,
        use_debug=args.debug,
    )
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()
    return model


def build_dataset(args):
    data_config = load_config(args.dataset_config)
    return StartEndDataset(
        config=data_config,
        max_ctx_len=args.max_ctx_len,
        max_desc_len=args.max_desc_len,
        clip_length=args.clip_length,
        ctx_mode=args.ctx_mode,
        mode=args.eval_split_name,
        data_ratio=args.data_ratio,
        is_eval=True,
        inference_top_k=args.max_vcmr_video,
    )


def _maybe_numpy(x):
    if x is None:
        return None
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return x


def export_batch_rows(batch, model_inputs, outputs, dataset, args, writer):
    video_similarity_score, begin_logits, end_logits, extras = outputs
    metas = batch["meta"]
    qbs = begin_logits.size(0)
    shared = begin_logits.size(1)
    video_len = begin_logits.size(2)
    topk = args.max_vcmr_video
    assert shared >= topk + 1, (shared, topk)

    # CONQUER eval convention: index 0 is the GT-video SVMR slot; VCMR candidates are 1:.
    vcmr_begin_logits = begin_logits[:, 1:topk + 1]
    vcmr_end_logits = end_logits[:, 1:topk + 1]
    st_probs = F.softmax(vcmr_begin_logits, dim=-1).detach().cpu().numpy()
    ed_probs = F.softmax(vcmr_end_logits, dim=-1).detach().cpu().numpy()
    st_logits_np = vcmr_begin_logits.detach().cpu().numpy()
    ed_logits_np = vcmr_end_logits.detach().cpu().numpy()

    r1_scores = model_inputs["inference_vr_scores"].detach().cpu().numpy()[:, :topk]
    r1_tilde = np.stack([query_standardize(row) for row in r1_scores], axis=0)

    r2_raw = None
    r2_prob = None
    r2_tilde = None
    r2_rank = None
    if video_similarity_score is not None:
        r2_raw = video_similarity_score[:, 1:topk + 1].detach().cpu().numpy()
        r2_prob = F.softmax(video_similarity_score[:, 1:topk + 1], dim=1).detach().cpu().numpy()
        r2_tilde = np.stack([query_standardize(row) for row in r2_raw], axis=0)
        r2_rank = np.argsort(-r2_raw, axis=1).argsort(axis=1) + 1

    # QDF modality weights, if available.
    moe_np = {}
    moe = extras.get("moe_weights_dict")
    if moe is not None:
        for mod, value in moe.items():
            arr = value.view(qbs, shared).detach().cpu().numpy()[:, 1:topk + 1]
            moe_np[mod] = arr

    # P_ctx from QAL Query2Video attention.
    p_ctx = None
    qal_aux = extras.get("qal_aux")
    if qal_aux is not None and qal_aux.get("q2v_attention") is not None:
        p_ctx = qal_aux["q2v_attention"].view(qbs, shared, video_len).detach().cpu().numpy()[:, 1:topk + 1]

    valid_mask = make_length_mask((topk, video_len, video_len), args.min_pred_l, args.max_pred_l)

    for q_idx, meta in enumerate(metas):
        # Fixed CONQUER-General candidate score: r1 * P_b(i) * P_e(j)
        st_ed_scores = np.einsum("vm,v,vn->vmn", st_probs[q_idx], r1_scores[q_idx], ed_probs[q_idx])
        st_ed_scores *= valid_mask
        flat = st_ed_scores.reshape(-1)
        k = min(args.max_before_nms, flat.size)
        # Stable top-k without sorting full array, then sort the selected part.
        top_flat_idx = np.argpartition(-flat, k - 1)[:k]
        top_flat_idx = top_flat_idx[np.argsort(-flat[top_flat_idx])]
        high_fp_cut = max(1, int(np.ceil(args.fp_top_frac * len(top_flat_idx))))

        gt_vid_name = meta.get("vid_name")
        gt_ts = meta.get("ts")
        desc_id = meta.get("desc_id")
        sample_vids = meta["sample_vid_name_list"]

        for rank0, flat_idx in enumerate(top_flat_idx):
            v_local, st_idx, ed_idx = np.unravel_index(int(flat_idx), (topk, video_len, video_len))
            score = float(flat[flat_idx])
            video_name = sample_vids[int(v_local)]
            video_idx = int(dataset.video2idx[video_name])
            st_time = float(st_idx * args.clip_length)
            ed_time = float(ed_idx * args.clip_length + args.clip_length)
            is_gt_video = int(video_name == gt_vid_name)
            iou = temporal_iou_seconds((st_time, ed_time), gt_ts) if is_gt_video else 0.0
            is_high_score = rank0 < high_fp_cut
            y_fp = int(is_high_score and ((not is_gt_video) or iou < args.fp_low_iou))

            pb = st_probs[q_idx, v_local]
            pe = ed_probs[q_idx, v_local]
            ctx_stats = span_stats_from_prior(None if p_ctx is None else p_ctx[q_idx, v_local], st_idx, ed_idx)

            mu_visual = moe_np.get("visual", None)
            mu_sub = moe_np.get("sub", None)
            mu_v = float(mu_visual[q_idx, v_local]) if mu_visual is not None else None
            mu_s = float(mu_sub[q_idx, v_local]) if mu_sub is not None else None
            if mu_v is not None and mu_s is not None:
                b_mod = mu_v - mu_s
                h_mod = float(-(mu_v * np.log(mu_v + EPS) + mu_s * np.log(mu_s + EPS)))
            else:
                b_mod = None
                h_mod = None

            row = {
                "desc_id": desc_id,
                "desc": meta.get("desc"),
                "gt_vid_name": gt_vid_name,
                "gt_ts": gt_ts,
                "video_name": video_name,
                "video_idx": video_idx,
                "rank_base": rank0 + 1,
                "rank_r1": int(v_local + 1),
                "rank_r2": int(r2_rank[q_idx, v_local]) if r2_rank is not None else None,
                "start_idx": int(st_idx),
                "end_idx": int(ed_idx),
                "start_time": st_time,
                "end_time": ed_time,
                "span_len": int(ed_idx - st_idx + 1),
                "span_len_norm": float((ed_idx - st_idx + 1) / video_len),
                "start_norm": float(st_idx / video_len),
                "end_norm": float(ed_idx / video_len),
                "r1": float(r1_scores[q_idx, v_local]),
                "r1_tilde": float(r1_tilde[q_idx, v_local]),
                "r2_raw": float(r2_raw[q_idx, v_local]) if r2_raw is not None else None,
                "r2_prob": float(r2_prob[q_idx, v_local]) if r2_prob is not None else None,
                "r2_tilde": float(r2_tilde[q_idx, v_local]) if r2_tilde is not None else None,
                "r_abs_gap": float(abs(r2_raw[q_idx, v_local] - r1_scores[q_idx, v_local])) if r2_raw is not None else None,
                "b_start_logit": float(st_logits_np[q_idx, v_local, st_idx]),
                "e_end_logit": float(ed_logits_np[q_idx, v_local, ed_idx]),
                "p_b_i": float(pb[st_idx]),
                "p_e_j": float(pe[ed_idx]),
                "l_logit": float(st_logits_np[q_idx, v_local, st_idx] + ed_logits_np[q_idx, v_local, ed_idx]),
                "l_prob": float(np.log(pb[st_idx] + EPS) + np.log(pe[ed_idx] + EPS)),
                "l_prod": float(pb[st_idx] * pe[ed_idx]),
                "u_b": entropy(pb),
                "u_e": entropy(pe),
                "u_bd": entropy(pb) + entropy(pe),
                "sharp_b": local_sharpness(pb, st_idx, radius=args.sharpness_radius),
                "sharp_e": local_sharpness(pe, ed_idx, radius=args.sharpness_radius),
                "margin_b": top_margin(pb),
                "margin_e": top_margin(pe),
                "mu_v": mu_v,
                "mu_s": mu_s,
                "b_mod": b_mod,
                "h_mod": h_mod,
                "m_ctx": ctx_stats["mass"],
                "mean_ctx": ctx_stats["mean"],
                "max_ctx": ctx_stats["max"],
                "peak_in_ctx": ctx_stats["peak_in_span"],
                "h_ctx": ctx_stats["entropy"],
                "s_base": score,
                "is_gt_video": is_gt_video,
                "iou": iou,
                "y_joint": iou if is_gt_video else 0.0,
                "y_joint_05": int(is_gt_video and iou >= 0.5),
                "y_joint_07": int(is_gt_video and iou >= 0.7),
                "m_bd": is_gt_video,
                "y_bd": iou if is_gt_video else 0.0,
                "y_fp": y_fp,
            }
            writer.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_config", default="config/tvr_data_config.json")
    parser.add_argument("--model_config", default="config/model_config.json")
    parser.add_argument("--model_dir", default=None, help="CONQUER results dir containing model.ckpt and opt.json")
    parser.add_argument("--ckpt_filepath", default=None, help="Explicit checkpoint path; overrides --model_dir")
    parser.add_argument("--output_jsonl", required=True)
    parser.add_argument("--eval_split_name", default="val", choices=["train", "val", "test_public"])
    parser.add_argument("--data_ratio", type=float, default=1.0)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--eval_query_bsz", type=int, default=5)
    parser.add_argument("--max_ctx_len", type=int, default=100)
    parser.add_argument("--max_desc_len", type=int, default=30)
    parser.add_argument("--clip_length", type=float, default=1.5)
    parser.add_argument("--ctx_mode", default="visual_sub")
    parser.add_argument("--visual_dim", type=int, default=4352)
    parser.add_argument("--text_dim", type=int, default=768)
    parser.add_argument("--query_dim", type=int, default=768)
    parser.add_argument("--hidden_dim", type=int, default=768)
    parser.add_argument("--no_output_moe_weight", action="store_true")
    parser.add_argument("--similarity_measure", default="general", choices=["general", "exclusive", "disjoint"])
    parser.add_argument("--max_before_nms", type=int, default=200)
    parser.add_argument("--max_vcmr_video", type=int, default=10)
    parser.add_argument("--min_pred_l", type=int, default=0)
    parser.add_argument("--max_pred_l", type=int, default=24)
    parser.add_argument("--fp_top_frac", type=float, default=0.10)
    parser.add_argument("--fp_low_iou", type=float, default=0.30)
    parser.add_argument("--sharpness_radius", type=int, default=2)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    # If a CONQUER opt.json exists, adopt its architecture/data defaults unless explicitly changed.
    saved = load_saved_options_if_any(args)
    # Keep this conservative: do not override paths/output/split from saved options.
    for key in [
        "max_ctx_len", "max_desc_len", "clip_length", "ctx_mode", "visual_dim", "text_dim",
        "query_dim", "hidden_dim", "no_output_moe_weight", "similarity_measure",
        "min_pred_l", "max_pred_l", "max_vcmr_video",
    ]:
        if key in saved and getattr(args, key) == parser.get_default(key):
            setattr(args, key, saved[key])
    return args


def main():
    args = parse_args()
    device = torch.device(f"cuda:{args.device}" if args.device >= 0 and torch.cuda.is_available() else "cpu")
    dataset = build_dataset(args)
    loader = DataLoader(
        dataset,
        collate_fn=start_end_collate,
        batch_size=args.eval_query_bsz,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=(device.type == "cuda"),
    )
    model = setup_model(args, device)

    partial_output = (
        args.output_jsonl[:-3] + ".partial.gz"
        if args.output_jsonl.endswith(".gz")
        else args.output_jsonl + ".partial"
    )
    with open_jsonl(partial_output) as writer, torch.no_grad():
        for batch_idx, batch in enumerate(
            tqdm(loader, total=len(loader), desc=f"export {args.eval_split_name} evidence")
        ):
            model_inputs = move_cuda(batch["model_inputs"], device) if device.type == "cuda" else batch["model_inputs"]
            outputs = model.get_pred_from_raw_query(model_inputs, return_intermediates=True)
            export_batch_rows(batch, model_inputs, outputs, dataset, args, writer)
            if args.debug:
                break
    os.replace(partial_output, args.output_jsonl)


if __name__ == "__main__":
    main()
