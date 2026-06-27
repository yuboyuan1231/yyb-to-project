#!/usr/bin/env python

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

"""C3 evidence-calibrated reranking for CONQUER VCMR candidates.

Input: C1 evidence JSONL(.gz) rows.
Output: TVR/CONQUER-style submission JSON with a VCMR list.
"""

import argparse
import gzip
import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import torch
import numpy as np
from tqdm import tqdm

from rlem.evidence_dataset import FeatureStats
from rlem.evidence_model import EvidenceMLP
from rlem.io_utils import iter_jsonl, read_json, write_json
from utils.inference_utils import filter_vcmr_by_nms

EPS = 1e-8

COMPACT_SCORED_FIELDS = [
    "desc_id", "desc", "video_idx", "video_name", "start_time", "end_time",
    "s_base", "rank_base", "is_gt_video", "iou", "y_fp", "y_joint_05",
    "q_joint_pred", "q_bd_pred", "e_fp_pred", "base_score_used", "s_rlem",
]


def load_model(ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    stats = FeatureStats.from_dict(ckpt["feature_stats"])
    cfg = ckpt["model_cfg"]
    model = EvidenceMLP(
        input_dim=cfg["input_dim"],
        hidden_dim=cfg.get("hidden_dim", 128),
        dropout=cfg.get("dropout", 0.0),
    )
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()
    return model, stats, ckpt


def base_score(row: Dict, mode: str) -> float:
    s = float(row.get("s_base") or 0.0)
    if mode == "raw":
        return s
    if mode == "log":
        return math.log(max(s, EPS))
    raise ValueError(f"Unknown base_score_mode: {mode}")


@torch.no_grad()
def score_rows(rows: List[Dict], model, stats: FeatureStats, device, args) -> List[Dict]:
    if not rows:
        return []
    xs = np.stack([stats.transform_row(r) for r in rows], axis=0)
    x = torch.from_numpy(xs).float().to(device)
    pred = model.predict_scores(x)
    qj = pred["q_joint"].detach().cpu().numpy().tolist()
    qb = pred["q_bd"].detach().cpu().numpy().tolist()
    efp = pred["e_fp"].detach().cpu().numpy().tolist()
    out = []
    for row, q_joint, q_bd, e_fp in zip(rows, qj, qb, efp):
        boundary_score = (
            float(q_bd)
            if args.score_family == "additive"
            else float(q_joint) * float(q_bd)
        )
        final = (
            args.base_scale * base_score(row, args.base_score_mode)
            + args.a_joint * float(q_joint)
            + args.b_bd * boundary_score
            - args.d_fp * float(e_fp)
        )
        new = dict(row)
        new["q_joint_pred"] = float(q_joint)
        new["q_bd_pred"] = float(q_bd)
        new["e_fp_pred"] = float(e_fp)
        new["base_score_used"] = float(base_score(row, args.base_score_mode))
        new["s_rlem"] = float(final)
        if args.compact_scored_jsonl:
            new = {key: new.get(key) for key in COMPACT_SCORED_FIELDS}
        out.append(new)
    return out


def load_video2idx_from_dataset_config(dataset_config: str, split: str) -> Optional[Dict[str, int]]:
    if not dataset_config:
        return None
    cfg = read_json(dataset_config)
    root_path = cfg.get("root_path", "")
    video_duration_idx_path = cfg.get("video_duration_idx_path")
    if not video_duration_idx_path:
        return None
    path = video_duration_idx_path if os.path.isabs(video_duration_idx_path) else os.path.join(root_path, video_duration_idx_path)
    if not os.path.exists(path):
        return None
    obj = read_json(path)
    if split not in obj:
        return None
    return {k: int(v[1]) for k, v in obj[split].items()}


def fallback_video2idx(groups: Dict) -> Dict[str, int]:
    mapping = {}
    for rows in groups.values():
        for r in rows:
            if r.get("video_name") is not None and r.get("video_idx") is not None:
                mapping[str(r["video_name"])] = int(r["video_idx"])
            if r.get("gt_vid_name") is not None and r.get("is_gt_video") and r.get("video_idx") is not None:
                mapping[str(r["gt_vid_name"])] = int(r["video_idx"])
    return mapping


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence_jsonl", required=True)
    parser.add_argument("--ckpt", required=True, help="C2/C3 evidence-head checkpoint, e.g. model_best.pt")
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--desc_ids_filter", default=None)
    parser.add_argument("--dataset_config", default=None, help="Used only to load full video2idx mapping")
    parser.add_argument("--split", default="val", choices=["train", "val", "test_public"])
    parser.add_argument("--video2idx_json", default=None, help="Optional explicit {video_name: idx} mapping")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--base_score_mode", default="log", choices=["log", "raw"])
    parser.add_argument(
        "--score_family",
        default="additive",
        choices=["additive", "gated_boundary"],
        help="Boundary term is Q_bd for additive or Q_joint*Q_bd for gated_boundary",
    )
    parser.add_argument("--base_scale", type=float, default=1.0)
    parser.add_argument("--a_joint", type=float, default=1.0)
    parser.add_argument("--b_bd", type=float, default=0.5)
    parser.add_argument("--d_fp", type=float, default=0.5)
    parser.add_argument("--score_batch_size", type=int, default=8192)
    parser.add_argument(
        "--max_before_nms", "--effective_top_n", dest="max_before_nms",
        type=int, default=1000,
    )
    parser.add_argument("--max_after_nms", type=int, default=100)
    parser.add_argument("--nms_thd", type=float, default=0.6)
    parser.add_argument("--no_nms", action="store_true")
    parser.add_argument("--save_scored_jsonl", default=None)
    parser.add_argument("--compact_scored_jsonl", action="store_true")
    parser.add_argument("--expected_rows_per_query", type=int, default=200)
    return parser.parse_args()


def load_desc_ids(path: Optional[str]):
    if not path:
        return None
    with open(path, "r", encoding="utf-8") as f:
        ids = {line.strip() for line in f if line.strip()}
    if not ids:
        raise ValueError(f"No desc_ids found in {path}")
    return ids


def open_atomic_scored_writer(path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if path.endswith(".gz"):
        partial = path[:-3] + ".partial.gz"
        writer = gzip.open(partial, "wt", encoding="utf-8", compresslevel=6)
    else:
        partial = path + ".partial"
        writer = open(partial, "w", encoding="utf-8")
    return writer, partial


def main():
    args = parse_args()
    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    model, stats, ckpt = load_model(args.ckpt, device)
    allowed_ids = load_desc_ids(args.desc_ids_filter)

    scored_groups = defaultdict(list)
    scored_writer = None
    scored_partial = None
    if args.save_scored_jsonl:
        scored_writer, scored_partial = open_atomic_scored_writer(args.save_scored_jsonl)

    def flush_batch(batch_rows):
        scored_rows = score_rows(batch_rows, model, stats, device, args)
        for r in scored_rows:
            scored_groups[r["desc_id"]].append(r)
            if scored_writer is not None:
                scored_writer.write(json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n")

    try:
        batch_rows = []
        for row in tqdm(iter_jsonl(args.evidence_jsonl), desc="read+score evidence"):
            if allowed_ids is not None and str(row.get("desc_id")) not in allowed_ids:
                continue
            batch_rows.append(row)
            if len(batch_rows) >= args.score_batch_size:
                flush_batch(batch_rows)
                batch_rows = []
        if batch_rows:
            flush_batch(batch_rows)
    except BaseException:
        if scored_writer is not None:
            scored_writer.close()
        if scored_partial and os.path.exists(scored_partial):
            os.remove(scored_partial)
        raise
    else:
        if scored_writer is not None:
            scored_writer.close()

    if allowed_ids is not None:
        found_ids = {str(desc_id) for desc_id in scored_groups}
        if found_ids != allowed_ids:
            if scored_partial and os.path.exists(scored_partial):
                os.remove(scored_partial)
            raise ValueError(
                f"Filtered desc_id mismatch: found={len(found_ids)} expected={len(allowed_ids)}"
            )
    rows_per_query = Counter(len(rows) for rows in scored_groups.values())
    expected_distribution = Counter({args.expected_rows_per_query: len(scored_groups)})
    if rows_per_query != expected_distribution:
        if scored_partial and os.path.exists(scored_partial):
            os.remove(scored_partial)
        raise ValueError(f"Unexpected rows/query distribution: {dict(rows_per_query)}")
    if scored_writer is not None:
        os.replace(scored_partial, args.save_scored_jsonl)

    for rows in scored_groups.values():
        rows.sort(key=lambda r: r["s_rlem"], reverse=True)

    if args.video2idx_json:
        video2idx = {k: int(v) for k, v in read_json(args.video2idx_json).items()}
    else:
        video2idx = load_video2idx_from_dataset_config(args.dataset_config, args.split) if args.dataset_config else None
        if video2idx is None:
            video2idx = fallback_video2idx(scored_groups)

    vcmr = []
    for desc_id, rows in tqdm(scored_groups.items(), desc="build submission"):
        if not rows:
            continue
        preds = []
        for r in rows[: args.max_before_nms]:
            preds.append([
                int(r["video_idx"]),
                float(r["start_time"]),
                float(r["end_time"]),
                float(r["s_rlem"]),
            ])
        if not args.no_nms:
            preds = filter_vcmr_by_nms(
                preds,
                nms_threshold=args.nms_thd,
                max_before_nms=args.max_before_nms,
                max_after_nms=args.max_after_nms,
            )
        else:
            preds = preds[: args.max_after_nms]
        first = rows[0]
        vcmr.append({
            "desc_id": desc_id,
            "desc": first.get("desc", ""),
            "predictions": preds,
        })

    submission = {"video2idx": video2idx, "VCMR": vcmr}
    write_json(args.output_json, submission, pretty=False)
    print(
        f"Saved RLEM VCMR submission to {args.output_json}; queries={len(vcmr)}; "
        f"rows_per_query={dict(rows_per_query)}; video2idx={len(video2idx)}; "
        f"max_before_nms={args.max_before_nms}; max_after_nms={args.max_after_nms}; "
        f"nms_thd={args.nms_thd}"
        f"; score_family={args.score_family}; base_scale={args.base_scale}; "
        f"a_joint={args.a_joint}; b_bd={args.b_bd}; d_fp={args.d_fp}"
    )


if __name__ == "__main__":
    main()
