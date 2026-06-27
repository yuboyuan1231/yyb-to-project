#!/usr/bin/env python

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

"""Export C3.5-QSP fixed-candidate evidence.

This script is implementation-only for the C3.5 branch.  It recomputes the same
CONQUER fixed candidate list as C1 and appends bounded query-specific temporal
prior scalars.  It can optionally stream an accepted C1 evidence artifact and
strictly verify candidate identity row-by-row.

Default policy is train-only.  Exporting official val is blocked unless the user
explicitly passes --allow_official_val_export after a separate one-shot
authorization.
"""

import argparse
import fcntl
import gzip
import json
import math
import os
from pathlib import Path
from typing import Dict, Iterator, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_loader.second_stage_start_end_dataset import StartEndDataset
from rlem.evidence_utils import EPS, make_length_mask, query_standardize
from rlem.export_conquer_evidence import (
    build_dataset,
    load_saved_options_if_any,
    open_jsonl,
    setup_model,
)
from rlem.io_utils import iter_jsonl, write_json
from rlem.qsp_features import (
    QSP_FEATURES,
    build_qsp_priors_from_intermediates,
    qsp_row_features,
    qsp_rows_features_vectorized,
)
from utils.basic_utils import load_config
from utils.model_utils import move_cuda, start_end_collate


IDENTITY_FIELDS = ["desc_id", "video_name", "rank_base", "start_idx", "end_idx"]


def _candidate_key(row: Dict):
    return str(row.get("video_name")), int(row.get("start_idx")), int(row.get("end_idx"))


def _next_expected(iterator: Optional[Iterator[Dict]]) -> Optional[Dict]:
    if iterator is None:
        return None
    try:
        return next(iterator)
    except StopIteration:
        raise ValueError("base_evidence_jsonl ended before generated QSP candidates")


def _check_identity(row: Dict, expected: Dict, atol: float = 1e-5):
    for key in IDENTITY_FIELDS:
        if str(row.get(key)) != str(expected.get(key)):
            raise ValueError(f"candidate identity mismatch for {key}: generated={row.get(key)} expected={expected.get(key)}")
    if "s_base" in expected and expected.get("s_base") is not None:
        got = float(row.get("s_base"))
        exp = float(expected.get("s_base"))
        if abs(got - exp) > atol * max(1.0, abs(exp)):
            raise ValueError(f"s_base mismatch: generated={got} expected={exp}")


def _align_query_to_base(generated_rows, expected_rows, output_mode, audit=None, atol: float = 1e-5):
    """Align QSP scalars to the frozen C1 order without relaxing candidate identity.

    Historical C1 rows may order exactly score-tied candidates differently from
    NumPy's current argpartition implementation. Candidate sets and scores must
    still match exactly at query level; only tie ordering is inherited from C1.
    """
    if len(generated_rows) != len(expected_rows):
        raise ValueError(f"candidate row count mismatch: generated={len(generated_rows)} expected={len(expected_rows)}")
    generated_by_key = {_candidate_key(row): row for row in generated_rows}
    expected_by_key = {_candidate_key(row): row for row in expected_rows}
    if len(generated_by_key) != len(generated_rows):
        raise ValueError("duplicate generated candidate identity within query")
    if len(expected_by_key) != len(expected_rows):
        raise ValueError("duplicate frozen C1 candidate identity within query")
    generated_keys = set(generated_by_key)
    expected_keys = set(expected_by_key)
    if generated_keys != expected_keys:
        missing = sorted(expected_keys - generated_keys)[:5]
        extra = sorted(generated_keys - expected_keys)[:5]
        raise ValueError(f"candidate identity set mismatch: missing={missing} extra={extra}")

    reordered_rows = sum(
        _candidate_key(generated) != _candidate_key(expected)
        for generated, expected in zip(generated_rows, expected_rows)
    )
    aligned = []
    max_score_diff = 0.0
    desc_id = str(expected_rows[0].get("desc_id")) if expected_rows else None
    for rank, expected in enumerate(expected_rows, 1):
        if str(expected.get("desc_id")) != desc_id or int(expected.get("rank_base")) != rank:
            raise ValueError("frozen C1 query rows are not contiguous with ranks 1..K")
        generated = generated_by_key[_candidate_key(expected)]
        if str(generated.get("desc_id")) != desc_id:
            raise ValueError("generated and frozen C1 desc_id mismatch")
        if expected.get("s_base") is not None:
            got = float(generated["s_base"])
            exp = float(expected["s_base"])
            diff = abs(got - exp)
            max_score_diff = max(max_score_diff, diff)
            if diff > atol * max(1.0, abs(exp)):
                raise ValueError(f"s_base mismatch for {_candidate_key(expected)}: generated={got} expected={exp}")
        if output_mode == "augment_base":
            row = dict(expected)
            row.update({key: generated[key] for key in QSP_FEATURES})
        else:
            row = dict(generated)
            row["rank_base"] = rank
        aligned.append(row)
    if audit is not None:
        audit["candidate_set_checks"] += 1
        audit["candidate_order_reassigned_rows"] += reordered_rows
        if reordered_rows:
            audit["candidate_order_reassigned_queries"] += 1
        audit["candidate_score_max_abs_diff"] = max(audit["candidate_score_max_abs_diff"], max_score_diff)
    return aligned


def _span_coverage_from_prior(coverage: Optional[np.ndarray], st_idx: int, ed_idx: int) -> float:
    if coverage is None:
        return 0.0
    i = max(0, int(st_idx))
    j = min(len(coverage) - 1, int(ed_idx))
    if j < i:
        return 0.0
    return float(np.asarray(coverage[i:j + 1], dtype=np.float32).mean())


def export_batch_qsp_rows(batch, model_inputs, outputs, dataset, args, writer, expected_iter=None, audit=None):
    _video_similarity_score, begin_logits, end_logits, extras = outputs
    if begin_logits.dtype in {torch.float16, torch.bfloat16} or end_logits.dtype in {torch.float16, torch.bfloat16}:
        raise ValueError(f"fp16/bfloat16 outputs are forbidden: {begin_logits.dtype}, {end_logits.dtype}")
    extras = dict(extras)
    extras["video_mask_dict"] = {
        key: model_inputs[key]["feat_mask"]
        for key in (args.visual_key, args.subtitle_key)
        if key in model_inputs and "feat_mask" in model_inputs[key]
    }
    if args.subtitle_key in model_inputs:
        raw_subtitle = model_inputs[args.subtitle_key]["feat"].float()
        raw_subtitle_mask = model_inputs[args.subtitle_key]["feat_mask"].bool()
        extras["sub_coverage_mask"] = (
            raw_subtitle.norm(dim=-1) > 1e-8
        ) & raw_subtitle_mask
    metas = batch["meta"]
    qbs = begin_logits.size(0)
    shared = begin_logits.size(1)
    video_len = begin_logits.size(2)
    topk = args.max_vcmr_video
    assert shared >= topk + 1, (shared, topk)

    vcmr_begin_logits = begin_logits[:, 1:topk + 1]
    vcmr_end_logits = end_logits[:, 1:topk + 1]
    st_probs = F.softmax(vcmr_begin_logits, dim=-1).detach().cpu().numpy()
    ed_probs = F.softmax(vcmr_end_logits, dim=-1).detach().cpu().numpy()
    r1_scores = model_inputs["inference_vr_scores"].detach().cpu().numpy()[:, :topk]
    priors = build_qsp_priors_from_intermediates(
        extras, topk=topk, visual_key=args.visual_key, subtitle_key=args.subtitle_key,
        temperature=args.qsp_temperature,
    )
    valid_mask = make_length_mask((topk, video_len, video_len), args.min_pred_l, args.max_pred_l)

    for q_idx, meta in enumerate(metas):
        st_ed_scores = np.einsum("vm,v,vn->vmn", st_probs[q_idx], r1_scores[q_idx], ed_probs[q_idx])
        st_ed_scores *= valid_mask
        flat = st_ed_scores.reshape(-1)
        k = min(args.max_before_nms, flat.size)
        top_flat_idx = np.argpartition(-flat, k - 1)[:k]
        top_flat_idx = top_flat_idx[np.argsort(-flat[top_flat_idx])]
        sample_vids = meta["sample_vid_name_list"]
        v_locals, st_idxs, ed_idxs = np.unravel_index(top_flat_idx.astype(np.int64), (topk, video_len, video_len))
        qsp_arrays = qsp_rows_features_vectorized(priors, q_idx, v_locals, st_idxs, ed_idxs)
        if audit is not None:
            audit["queries"] += 1
            remaining = max(0, args.naive_check_rows - audit["naive_check_rows"])
            if remaining:
                check_count = min(remaining, k)
                rng = np.random.default_rng(args.naive_check_seed + audit["queries"] - 1)
                check_indices = rng.choice(k, size=check_count, replace=False)
                for row_idx in check_indices:
                    naive = qsp_row_features(
                        priors, q_idx, int(v_locals[row_idx]),
                        int(st_idxs[row_idx]), int(ed_idxs[row_idx]),
                    )
                    if list(naive) != list(qsp_arrays):
                        raise ValueError("vectorized and naive QSP feature names differ")
                    for key in QSP_FEATURES:
                        diff = abs(float(qsp_arrays[key][row_idx]) - float(naive[key]))
                        audit["naive_check_feature_max_abs_diff"][key] = max(
                            audit["naive_check_feature_max_abs_diff"][key], diff,
                        )
                        audit["naive_check_max_abs_diff"] = max(audit["naive_check_max_abs_diff"], diff)
                        if diff > args.naive_check_atol:
                            raise ValueError(
                                f"vectorized/naive mismatch {key}: diff={diff} "
                                f"atol={args.naive_check_atol}"
                            )
                    audit["naive_check_rows"] += 1
        generated_rows = []
        for rank0, flat_idx in enumerate(top_flat_idx):
            v_local = int(v_locals[rank0])
            st_idx = int(st_idxs[rank0])
            ed_idx = int(ed_idxs[rank0])
            video_name = sample_vids[v_local]
            row = {
                "desc_id": meta.get("desc_id"),
                "video_name": video_name,
                "rank_base": rank0 + 1,
                "rank_r1": int(v_local + 1),
                "start_idx": st_idx,
                "end_idx": ed_idx,
                "start_time": float(st_idx * args.clip_length),
                "end_time": float(ed_idx * args.clip_length + args.clip_length),
                "s_base": float(flat[flat_idx]),
            }
            row.update({key: float(qsp_arrays[key][rank0]) for key in QSP_FEATURES})
            for key in QSP_FEATURES:
                val = row.get(key)
                if val is None or not math.isfinite(float(val)):
                    raise ValueError(f"non-finite QSP feature {key}: {val}")
            generated_rows.append(row)
        if expected_iter is not None:
            expected_rows = [_next_expected(expected_iter) for _ in range(k)]
            output_rows = _align_query_to_base(
                generated_rows, expected_rows, args.output_mode, audit=audit,
            )
        else:
            output_rows = generated_rows
        for row in output_rows:
            writer.write(json.dumps(row, ensure_ascii=False) + "\n")
            if audit is not None:
                audit["rows"] += 1
                audit["qsp_feature_names"] = QSP_FEATURES


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_config", default="config/tvr_data_config.json")
    parser.add_argument("--model_config", default="config/model_config.json")
    parser.add_argument("--model_dir", default=None)
    parser.add_argument("--ckpt_filepath", default=None)
    parser.add_argument("--output_jsonl", required=True)
    parser.add_argument("--base_evidence_jsonl", default=None, help="Accepted C1 evidence for strict identity check and optional augmentation")
    parser.add_argument("--output_mode", choices=["qsp_only", "augment_base"], default="augment_base")
    parser.add_argument("--audit_json", default=None)
    parser.add_argument("--eval_split_name", default="train", choices=["train", "val", "test_public"])
    parser.add_argument("--allow_official_val_export", action="store_true")
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
    parser.add_argument("--visual_key", default="visual")
    parser.add_argument("--subtitle_key", default="sub")
    parser.add_argument("--qsp_temperature", type=float, default=1.0)
    parser.add_argument("--naive_check_rows", type=int, default=100)
    parser.add_argument("--naive_check_seed", type=int, default=13)
    parser.add_argument("--naive_check_atol", type=float, default=1e-6)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    if args.eval_split_name == "val" and not args.allow_official_val_export:
        raise ValueError("Official val QSP export is blocked. Pass --allow_official_val_export only after explicit one-shot authorization.")
    if args.output_mode == "augment_base" and not args.base_evidence_jsonl:
        raise ValueError("--output_mode augment_base requires --base_evidence_jsonl")
    if args.eval_split_name == "train" and args.base_evidence_jsonl:
        base_name = Path(args.base_evidence_jsonl).name.lower()
        if "val" in base_name or "test" in base_name:
            raise ValueError("train export refuses a val/test base evidence path")
    if args.base_evidence_jsonl and Path(args.output_jsonl).resolve() == Path(args.base_evidence_jsonl).resolve():
        raise ValueError("QSP output must not overwrite its base evidence")
    saved = load_saved_options_if_any(args)
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
    partial_export = bool(args.debug or args.data_ratio < 1.0)
    lock_path = str(Path(args.output_jsonl).resolve()) + ".lock"
    Path(lock_path).parent.mkdir(parents=True, exist_ok=True)
    output_lock = open(lock_path, "w", encoding="utf-8")
    try:
        fcntl.flock(output_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise RuntimeError(f"another export already owns output lock: {lock_path}") from exc
    output_lock.write(f"pid={os.getpid()}\n")
    output_lock.flush()
    device = torch.device(f"cuda:{args.device}" if args.device >= 0 and torch.cuda.is_available() else "cpu")
    dataset = build_dataset(args)
    loader = DataLoader(dataset, collate_fn=start_end_collate, batch_size=args.eval_query_bsz,
                        num_workers=args.num_workers, shuffle=False, pin_memory=(device.type == "cuda"))
    model = setup_model(args, device)
    parameter_dtypes = sorted({str(parameter.dtype) for parameter in model.parameters()})
    fp16_used = any(dtype in {"torch.float16", "torch.bfloat16"} for dtype in parameter_dtypes)
    if fp16_used:
        raise ValueError(f"fp16/bfloat16 model parameters are forbidden: {parameter_dtypes}")
    expected_iter = iter_jsonl(args.base_evidence_jsonl) if args.base_evidence_jsonl else None
    audit = {
        "stage": "C3.5-QSP implementation-only evidence export",
        "split": args.eval_split_name,
        "official_val_export_allowed": bool(args.allow_official_val_export),
        "base_evidence_identity_check": bool(args.base_evidence_jsonl),
        "output_mode": args.output_mode,
        "candidate_rows_per_query": args.max_before_nms,
        "max_vcmr_video": args.max_vcmr_video,
        "rows": 0,
        "queries": 0,
        "candidate_set_checks": 0,
        "candidate_order_reassigned_queries": 0,
        "candidate_order_reassigned_rows": 0,
        "candidate_score_max_abs_diff": 0.0,
        "fp16_used": fp16_used,
        "model_parameter_dtypes": parameter_dtypes,
        "official_val_used": False,
        "official_val_used_for_selection": False,
        "partial_export": partial_export,
        "base_evidence_full_consumption_required": not partial_export,
        "qsp_feature_names": QSP_FEATURES,
        "naive_check_rows_requested": args.naive_check_rows,
        "naive_check_rows": 0,
        "naive_check_seed": args.naive_check_seed,
        "naive_check_atol": args.naive_check_atol,
        "naive_check_feature_names_identical": True,
        "naive_check_max_abs_diff": 0.0,
        "naive_check_feature_max_abs_diff": {key: 0.0 for key in QSP_FEATURES},
    }
    partial_output = args.output_jsonl[:-3] + ".partial.gz" if args.output_jsonl.endswith(".gz") else args.output_jsonl + ".partial"
    Path(args.output_jsonl).parent.mkdir(parents=True, exist_ok=True)
    with open_jsonl(partial_output) as writer, torch.no_grad():
        for _batch_idx, batch in enumerate(tqdm(loader, total=len(loader), desc=f"export {args.eval_split_name} QSP evidence")):
            model_inputs = move_cuda(batch["model_inputs"], device) if device.type == "cuda" else batch["model_inputs"]
            outputs = model.get_pred_from_raw_query(model_inputs, return_intermediates=True)
            export_batch_qsp_rows(batch, model_inputs, outputs, dataset, args, writer, expected_iter=expected_iter, audit=audit)
            if args.debug:
                break
    if expected_iter is not None and not partial_export:
        try:
            extra = next(expected_iter)
            raise ValueError(f"base_evidence_jsonl has extra rows after QSP export, next desc_id={extra.get('desc_id')}")
        except StopIteration:
            pass
    if audit["naive_check_rows"] != args.naive_check_rows:
        raise ValueError(
            f"naive consistency check covered {audit['naive_check_rows']} rows, "
            f"expected {args.naive_check_rows}"
        )
    audit["naive_check_pass"] = audit["naive_check_max_abs_diff"] <= args.naive_check_atol
    os.replace(partial_output, args.output_jsonl)
    if args.audit_json:
        write_json(args.audit_json, audit)
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
