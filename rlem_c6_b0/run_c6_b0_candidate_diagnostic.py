#!/usr/bin/env python
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from rlem_c6.c6a_utils import (  # noqa: E402
    C6AdapterModel,
    c6_collate,
    load_config,
    move_batch,
    write_json,
    write_text,
)


METRIC_KEYS = [
    "0.5-r1", "0.5-r5", "0.5-r10", "0.5-r100",
    "0.7-r1", "0.7-r5", "0.7-r10", "0.7-r100",
]
PRIMARY_R1 = ["0.5-r1", "0.7-r1"]
SAFETY_R5 = ["0.5-r5", "0.7-r5"]
CLIP = 1.5
EPS = 1e-12


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def artifact(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    return {
        "path": str(p),
        "exists": p.exists(),
        "size": int(p.stat().st_size) if p.exists() else None,
        "sha256": sha256_file(p) if p.exists() and p.is_file() else None,
    }


def iter_jsonl(path: str | Path) -> Iterable[Dict[str, Any]]:
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def interval_iou(a0: float, a1: float, b0: float, b1: float) -> float:
    inter = max(0.0, min(float(a1), float(b1)) - max(float(a0), float(b0)))
    union = max(float(a1), float(b1)) - min(float(a0), float(b0))
    return float(inter / union) if union > 0 else 0.0


def span_iou_idx(si: int, ei: int, gs: int, ge: int) -> float:
    return interval_iou(si * CLIP, (ei + 1) * CLIP, gs * CLIP, (ge + 1) * CLIP)


def span_iou_ts(si: int, ei: int, ts: Sequence[float]) -> float:
    return interval_iou(si * CLIP, (ei + 1) * CLIP, float(ts[0]), float(ts[1]))


def is_nested_timestamps(ts: Any) -> bool:
    return bool(ts) and isinstance(ts, list) and isinstance(ts[0], (list, tuple))


def correctness_for_ts(si: int, ei: int, ts: Any, threshold: float) -> bool:
    if is_nested_timestamps(ts):
        return sum(span_iou_ts(si, ei, one) >= threshold for one in ts) >= 2
    return span_iou_ts(si, ei, ts) >= threshold


def best_iou_for_ts(si: int, ei: int, ts: Any) -> float:
    if is_nested_timestamps(ts):
        return float(max(span_iou_ts(si, ei, one) for one in ts))
    return span_iou_ts(si, ei, ts)


def temporal_nms_order(starts: np.ndarray, ends: np.ndarray, scores: np.ndarray, thd: float, max_keep: int | None) -> List[int]:
    order = np.argsort(-scores, kind="stable")
    kept: List[int] = []
    for idx in order:
        si, ei = int(starts[idx]), int(ends[idx])
        suppress = False
        for prior in kept:
            if span_iou_idx(si, ei, int(starts[prior]), int(ends[prior])) > thd:
                suppress = True
                break
        if not suppress:
            kept.append(int(idx))
            if max_keep is not None and len(kept) >= max_keep:
                break
    return kept


@dataclass
class CandidateList:
    starts: np.ndarray
    ends: np.ndarray
    scores: np.ndarray
    source: np.ndarray
    pre_best_iou: float = math.nan


class BoundaryGroupDataset(Dataset):
    def __init__(self, cache_npz: str, temporal_npz: str):
        with np.load(cache_npz, allow_pickle=True) as cache:
            self.group_offsets = cache["group_offsets"].astype(np.int64)
            self.sort_idx = cache["group_sort_idx"].astype(np.int64)
            self.group_ids = cache["group_ids_sorted_unique"].astype(np.int64)
            self.start_idx = cache["start_idx"].astype(np.int64)
            self.end_idx = cache["end_idx"].astype(np.int64)
            self.s_c4_final = cache["s_c4_final"].astype(np.float32)
            self.iou = cache["iou"].astype(np.float32)
            self.y05 = cache["y_joint_05"].astype(np.float32)
            self.y07 = cache["y_joint_07"].astype(np.float32)
            self.is_gt_video = cache["is_gt_video"].astype(np.float32)
            self.gt_start_idx = cache["gt_start_idx"].astype(np.int64)
            self.gt_end_idx = cache["gt_end_idx"].astype(np.int64)
            self.label_relevant = cache["label_relevant"].astype(np.float32)
            self.video_features = cache["video_features"].astype(np.float32)
            self.video_feat_dim = int(self.video_features.shape[1])
        with np.load(temporal_npz, allow_pickle=False) as temporal:
            self.p_b = temporal["p_b"].astype(np.float32)
            self.p_e = temporal["p_e"].astype(np.float32)
            self.p_ctx = temporal["p_ctx"].astype(np.float32)
            self.temporal_length = temporal["temporal_length"].astype(np.int64)

    def __len__(self) -> int:
        return len(self.group_offsets)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        s = int(self.group_offsets[idx])
        e = int(self.group_offsets[idx + 1]) if idx + 1 < len(self.group_offsets) else len(self.sort_idx)
        rows = self.sort_idx[s:e]
        gid = int(self.group_ids[idx])
        t = int(self.temporal_length[gid])
        return {
            "group_id": gid,
            "rows": rows.astype(np.int64),
            "p_b": self.p_b[gid, :t],
            "p_e": self.p_e[gid, :t],
            "p_ctx": self.p_ctx[gid, :t],
            "start_idx": self.start_idx[rows],
            "end_idx": self.end_idx[rows],
            "s_c4": self.s_c4_final[rows],
            "row_feat": np.zeros((len(rows), 1), dtype=np.float32),
            "iou": self.iou[rows],
            "y05": self.y05[rows],
            "y07": self.y07[rows],
            "is_gt_video": self.is_gt_video[rows],
            "gt_start_idx": int(self.gt_start_idx[rows[0]]),
            "gt_end_idx": int(self.gt_end_idx[rows[0]]),
            "label_relevant": float(self.label_relevant[gid]),
            "video_feat": self.video_features[gid],
        }


def load_boundary_model(ckpt_path: str, device: torch.device) -> C6AdapterModel:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg = ckpt.get("config") or load_config("rlem_c6/configs/c6a_adapter_default.yaml")
    model = C6AdapterModel(
        row_feat_dim=int(ckpt.get("row_feat_dim", 1)),
        video_feat_dim=int(ckpt["video_feat_dim"]),
        hidden_dim=int(cfg["hidden_dim"]),
        temporal_layers=int(cfg["temporal_layers"]),
        heads=int(cfg["heads"]),
        ffn_dim=int(cfg["ffn_dim"]),
        span_head_hidden=int(cfg["span_head_hidden"]),
        span_head_layers=int(cfg["span_head_layers"]),
        dropout=float(cfg["dropout"]),
        scale_b=float(cfg.get("scale_b", 0.5)),
        scale_e=float(cfg.get("scale_e", 0.5)),
        alpha_rank=0.0,
        alpha_bd=0.0,
        alpha_q=0.0,
    )
    model.load_state_dict(ckpt["model_state"], strict=True)
    return model.to(device).eval()


def infer_boundary_prior(
    *,
    cache_npz: str,
    temporal_npz: str,
    ckpt_path: str,
    output_npz: str,
    device_name: str,
    batch_groups: int,
) -> Dict[str, Any]:
    out = Path(output_npz)
    if out.exists():
        with np.load(out, allow_pickle=False) as payload:
            return {
                "reused": True,
                "path": str(out),
                "p_b_new_shape": list(payload["p_b_new"].shape),
                "p_e_new_shape": list(payload["p_e_new"].shape),
                "artifact": artifact(out),
            }
    ds = BoundaryGroupDataset(cache_npz, temporal_npz)
    device = torch.device("cuda" if device_name == "cuda" and torch.cuda.is_available() else "cpu")
    model = load_boundary_model(ckpt_path, device)
    max_t = int(ds.p_b.shape[1])
    p_b_new = np.zeros((len(ds), max_t), dtype=np.float32)
    p_e_new = np.zeros((len(ds), max_t), dtype=np.float32)
    dl = DataLoader(ds, batch_size=batch_groups, shuffle=False, num_workers=0, collate_fn=c6_collate, pin_memory=(device.type == "cuda"))
    t0 = time.time()
    with torch.no_grad():
        for step, batch in enumerate(dl, 1):
            gids = batch["group_id"].numpy()
            batch.pop("rows")
            tb = move_batch(batch, device)
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                pred = model(tb)
            pb = pred["p_b_new"].detach().float().cpu().numpy()
            pe = pred["p_e_new"].detach().float().cpu().numpy()
            for bi, gid in enumerate(gids):
                t = int(ds.temporal_length[gid])
                p_b_new[gid, :t] = pb[bi, :t]
                p_e_new[gid, :t] = pe[bi, :t]
            if step % 200 == 0:
                print(json.dumps({"stage": "infer_boundary_prior", "batches": step, "elapsed_sec": time.time() - t0}))
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(out) + ".partial")
    with open(tmp, "wb") as f:
        np.savez_compressed(f, group_id=np.arange(len(ds), dtype=np.int32), p_b_new=p_b_new, p_e_new=p_e_new)
    os.replace(tmp, out)
    return {
        "reused": False,
        "path": str(out),
        "p_b_new_shape": list(p_b_new.shape),
        "p_e_new_shape": list(p_e_new.shape),
        "artifact": artifact(out),
        "elapsed_sec": float(time.time() - t0),
        "device": str(device),
    }


def load_gt_ts(gt_jsonl: str, desc_ids: Sequence[Any]) -> Dict[int, Any]:
    by_id = {str(row["desc_id"]): row.get("ts", row.get("gt_ts")) for row in iter_jsonl(gt_jsonl)}
    out: Dict[int, Any] = {}
    missing = []
    for qi, did in enumerate(desc_ids):
        key = str(did)
        if key not in by_id:
            missing.append(key)
        else:
            out[qi] = by_id[key]
    if missing:
        raise ValueError(f"GT missing {len(missing)} desc_ids, first={missing[:3]}")
    return out


def build_group_candidates(
    *,
    cache: Dict[str, np.ndarray],
    temporal_length: np.ndarray,
    p_b_new: np.ndarray,
    p_e_new: np.ndarray,
    gt_ts_by_query: Dict[int, Any],
    nms_thd: float,
    top_endpoint: int,
) -> Tuple[Dict[str, List[CandidateList]], Dict[str, Any]]:
    group_offsets = cache["group_offsets"].astype(np.int64)
    sort_idx = cache["group_sort_idx"].astype(np.int64)
    group_ids = cache["group_ids_sorted_unique"].astype(np.int64)
    row_group_id = cache["row_group_id"].astype(np.int64)
    query_index = cache["query_index"].astype(np.int64)
    start_idx = cache["start_idx"].astype(np.int64)
    end_idx = cache["end_idx"].astype(np.int64)
    s_c4 = cache["s_c4_final"].astype(np.float32)
    label_relevant = cache["label_relevant"].astype(np.float32)

    variants: Dict[str, List[CandidateList]] = {
        "original_candidates": [None] * len(group_ids),  # type: ignore[list-item]
        "boundary_oracle_candidates": [None] * len(group_ids),  # type: ignore[list-item]
        "hybrid_candidates": [None] * len(group_ids),  # type: ignore[list-item]
    }
    audit = {
        "group_count": int(len(group_ids)),
        "candidate_count_mismatch": 0,
        "index_anomalies": 0,
        "time_anomalies": 0,
        "source_counts": {"hybrid_original": 0, "hybrid_boundary": 0},
    }

    for idx, gid in enumerate(group_ids):
        s = int(group_offsets[idx])
        e = int(group_offsets[idx + 1]) if idx + 1 < len(group_offsets) else len(sort_idx)
        rows = sort_idx[s:e]
        quota = int(len(rows))
        if quota <= 0:
            raise ValueError(f"empty group {gid}")
        q = int(query_index[rows[0]])
        t = int(temporal_length[gid])
        pb = np.maximum(p_b_new[gid, :t].astype(np.float64), EPS)
        pe = np.maximum(p_e_new[gid, :t].astype(np.float64), EPS)
        pb = pb / max(float(pb.sum()), EPS)
        pe = pe / max(float(pe.sum()), EPS)

        # A. original fixed candidates, ordered by frozen C4_final score.
        o_order = np.argsort(-s_c4[rows], kind="stable")
        o_st = start_idx[rows][o_order].astype(np.int16)
        o_en = end_idx[rows][o_order].astype(np.int16)
        o_sc = s_c4[rows][o_order].astype(np.float32)
        variants["original_candidates"][idx] = CandidateList(o_st, o_en, o_sc, np.zeros(quota, dtype=np.int8))

        # B. boundary proposals from P_b_new/P_e_new.  Candidate pool is a
        # top-start x top-end cartesian product, then same-video temporal NMS.
        m = min(t, max(top_endpoint, int(math.ceil(math.sqrt(quota * 8)))))
        top_s = np.argsort(-pb, kind="stable")[:m]
        top_e = np.argsort(-pe, kind="stable")[:m]
        ss, ee = np.meshgrid(top_s, top_e, indexing="ij")
        ss = ss.reshape(-1)
        ee = ee.reshape(-1)
        valid = ss <= ee
        ss = ss[valid].astype(np.int16)
        ee = ee[valid].astype(np.int16)
        b_scores = (np.log(pb[ss]) + np.log(pe[ee])).astype(np.float32)
        if len(ss) < quota:
            all_s, all_e = np.triu_indices(t)
            ss = np.concatenate([ss, all_s.astype(np.int16)])
            ee = np.concatenate([ee, all_e.astype(np.int16)])
            b_scores = np.concatenate([b_scores, (np.log(pb[all_s]) + np.log(pe[all_e])).astype(np.float32)])
        # dedupe spans, keeping highest score.
        keys = ss.astype(np.int32) * 1024 + ee.astype(np.int32)
        uniq, first = np.unique(keys, return_index=True)
        ss, ee, b_scores = ss[first], ee[first], b_scores[first]
        keep = temporal_nms_order(ss, ee, b_scores, nms_thd, quota)
        if len(keep) < quota:
            all_order = np.argsort(-b_scores, kind="stable").tolist()
            seen = set(keep)
            keep.extend([int(i) for i in all_order if int(i) not in seen][:quota - len(keep)])
        keep = keep[:quota]
        b_st = ss[keep].astype(np.int16)
        b_en = ee[keep].astype(np.int16)
        b_sc = b_scores[keep].astype(np.float32)
        pre_best = math.nan
        if label_relevant[gid] > 0.5:
            ts = gt_ts_by_query[q]
            pre_best = float(max(best_iou_for_ts(int(a), int(b), ts) for a, b in zip(ss, ee)))
        variants["boundary_oracle_candidates"][idx] = CandidateList(b_st, b_en, b_sc, np.ones(quota, dtype=np.int8), pre_best_iou=pre_best)

        # C. hybrid: original spans plus boundary spans, unified NMS, same quota.
        ho_st = o_st.astype(np.int16)
        ho_en = o_en.astype(np.int16)
        ho_sc = (np.log(pb[np.clip(ho_st, 0, t - 1)]) + np.log(pe[np.clip(ho_en, 0, t - 1)])).astype(np.float32)
        h_st = np.concatenate([ho_st, ss]).astype(np.int16)
        h_en = np.concatenate([ho_en, ee]).astype(np.int16)
        h_sc = np.concatenate([ho_sc, b_scores]).astype(np.float32)
        h_src = np.concatenate([np.zeros(len(ho_st), dtype=np.int8), np.ones(len(ss), dtype=np.int8)])
        h_keys = h_st.astype(np.int32) * 1024 + h_en.astype(np.int32)
        # If a duplicate exists, prefer original when scores tie; otherwise keep max score.
        order = np.lexsort((h_src, -h_sc))
        h_keys_o = h_keys[order]
        first_mask = np.r_[True, h_keys_o[1:] != h_keys_o[:-1]]
        pick = order[first_mask]
        h_st, h_en, h_sc, h_src = h_st[pick], h_en[pick], h_sc[pick], h_src[pick]
        keep = temporal_nms_order(h_st, h_en, h_sc, nms_thd, quota)
        if len(keep) < quota:
            all_order = np.argsort(-h_sc, kind="stable").tolist()
            seen = set(keep)
            keep.extend([int(i) for i in all_order if int(i) not in seen][:quota - len(keep)])
        keep = keep[:quota]
        h_list = CandidateList(h_st[keep].astype(np.int16), h_en[keep].astype(np.int16), h_sc[keep].astype(np.float32), h_src[keep].astype(np.int8))
        variants["hybrid_candidates"][idx] = h_list
        audit["source_counts"]["hybrid_original"] += int(np.sum(h_list.source == 0))
        audit["source_counts"]["hybrid_boundary"] += int(np.sum(h_list.source == 1))

        for name in variants:
            c = variants[name][idx]
            audit["candidate_count_mismatch"] += int(c is None or len(c.starts) != quota)  # type: ignore[arg-type]
            if c is not None:
                audit["index_anomalies"] += int(np.sum((c.starts < 0) | (c.ends < c.starts) | (c.ends >= t)))
    return variants, audit


def candidate_oracle_eval(
    *,
    cache: Dict[str, np.ndarray],
    variants: Dict[str, List[CandidateList]],
    gt_ts_by_query: Dict[int, Any],
) -> Dict[str, Any]:
    label_relevant = cache["label_relevant"].astype(np.float32)
    video_keys = cache["video_group_keys"].astype(np.int64)
    q_count = len(cache["desc_ids"])
    positive_gid_by_q = np.full(q_count, -1, dtype=np.int64)
    for gid, rel in enumerate(label_relevant):
        if rel > 0.5:
            q = int(video_keys[gid, 0])
            if positive_gid_by_q[q] < 0:
                positive_gid_by_q[q] = gid
    gt_video_missing = int(np.sum(positive_gid_by_q < 0))

    out: Dict[str, Any] = {}
    for name, lists in variants.items():
        best_iou = []
        best_rank = []
        retained_best = []
        counts = []
        present_rows = []
        recall = {f"top{k}_r{thr}": 0 for k in [1, 5, 10, 20, 100] for thr in ["05", "07"]}
        for q in range(q_count):
            gid = int(positive_gid_by_q[q])
            if gid < 0:
                best_iou.append(0.0)
                best_rank.append(math.nan)
                retained_best.append(0.0)
                counts.append(0)
                continue
            c = lists[gid]
            ts = gt_ts_by_query[q]
            ious = np.asarray([best_iou_for_ts(int(s), int(e), ts) for s, e in zip(c.starts, c.ends)], dtype=np.float32)
            counts.append(int(len(ious)))
            present_rows.append(float(len(ious) > 0))
            if len(ious) == 0:
                best_iou.append(0.0)
                best_rank.append(math.nan)
                retained_best.append(0.0)
                continue
            bi = int(np.argmax(ious))
            best_iou.append(float(ious[bi]))
            best_rank.append(float(bi + 1))
            if math.isfinite(c.pre_best_iou):
                retained_best.append(float(np.max(ious) + 1e-6 >= c.pre_best_iou))
            else:
                retained_best.append(1.0)
            for k in [1, 5, 10, 20, 100]:
                top = ious[:min(k, len(ious))]
                recall[f"top{k}_r05"] += int(np.any(top >= 0.5))
                recall[f"top{k}_r07"] += int(np.any(top >= 0.7))
        n = float(q_count)
        out[name] = {
            "gt_video_oracle_r05": 100.0 * float(np.mean(np.asarray(best_iou) >= 0.5)),
            "gt_video_oracle_r07": 100.0 * float(np.mean(np.asarray(best_iou) >= 0.7)),
            "best_iou_mean": float(np.mean(best_iou)),
            "best_iou_median": float(np.median(best_iou)),
            "best_iou_candidate_rank_mean": float(np.nanmean(best_rank)),
            "best_iou_candidate_rank_median": float(np.nanmedian(best_rank)),
            "nms_retained_pre_nms_best_iou_rate": 100.0 * float(np.mean(retained_best)),
            "selected_span_miou_oracle_upper_bound": float(np.mean(best_iou)),
            "gt_video_missing_count": gt_video_missing,
            "gt_video_present_query_count": int(q_count - gt_video_missing),
            "candidate_count_min": int(np.min(counts)),
            "candidate_count_max": int(np.max(counts)),
            "candidate_count_mean": float(np.mean(counts)),
            "candidate_count_mean_when_gt_video_present": float(np.mean([x for x in counts if x > 0])) if any(x > 0 for x in counts) else 0.0,
            "topk_proposal_recall": {k: 100.0 * v / n for k, v in recall.items()},
        }
    base = out["original_candidates"]
    for name in ["boundary_oracle_candidates", "hybrid_candidates"]:
        out[name]["delta_vs_original_candidates"] = {
            k: float(out[name][k] - base[k])
            for k in [
                "gt_video_oracle_r05",
                "gt_video_oracle_r07",
                "best_iou_mean",
                "best_iou_median",
                "best_iou_candidate_rank_mean",
                "nms_retained_pre_nms_best_iou_rate",
            ]
        }
    return out


def selected_metrics(selected_labels: Dict[str, List[List[bool]]]) -> Dict[str, float]:
    q_count = len(selected_labels["0.5"])
    out = {k: 0 for k in METRIC_KEYS}
    for q in range(q_count):
        for thr in ["0.5", "0.7"]:
            labs = selected_labels[thr][q]
            for k in [1, 5, 10, 100]:
                out[f"{thr}-r{k}"] += int(any(labs[:k]))
    return {k: 100.0 * v / q_count for k, v in out.items()}


def pseudo_vcmr_eval(
    *,
    cache: Dict[str, np.ndarray],
    variants: Dict[str, List[CandidateList]],
    gt_ts_by_query: Dict[int, Any],
    effective_top_n: int,
    max_after_nms: int,
    nms_thd: float,
) -> Dict[str, Any]:
    q_count = len(cache["desc_ids"])
    rows_per_q = int(cache["desc_offsets"][1] - cache["desc_offsets"][0])
    row_group = cache["row_group_id"].astype(np.int64)
    video_idx = cache["video_idx"].astype(np.int64)
    start_idx = cache["start_idx"].astype(np.int64)
    end_idx = cache["end_idx"].astype(np.int64)
    s_c4 = cache["s_c4_final"].astype(np.float32)
    label_relevant = cache["label_relevant"].astype(np.float32)
    video_keys = cache["video_group_keys"].astype(np.int64)
    gt_video_by_q = np.full(q_count, -1, dtype=np.int64)
    for gid, rel in enumerate(label_relevant):
        if rel > 0.5:
            q = int(video_keys[gid, 0])
            if gt_video_by_q[q] < 0:
                gt_video_by_q[q] = int(video_keys[gid, 1])

    out: Dict[str, Any] = {}
    original_selected_signatures = None
    original_query_positive_top100 = None
    original_top1_sig = None
    original_slot_sigs = None

    for name, lists in variants.items():
        selected_labels = {"0.5": [], "0.7": []}
        selected_signatures: List[List[Tuple[int, int, int]]] = []
        query_positive_top100 = []
        top1_sig = []
        slot_replacements = 0
        total_slots = 0
        for q in range(q_count):
            rs = np.arange(q * rows_per_q, (q + 1) * rows_per_q, dtype=np.int64)
            base_order = rs[np.argsort(-s_c4[rs], kind="stable")[:effective_top_n]]
            ptr: Dict[int, int] = {}
            slot_candidates: List[Tuple[int, int, int, float, bool, bool]] = []
            for row in base_order:
                gid = int(row_group[row])
                c = lists[gid]
                pos = ptr.get(gid, 0)
                if pos >= len(c.starts):
                    pos = len(c.starts) - 1
                ptr[gid] = pos + 1
                si, ei = int(c.starts[pos]), int(c.ends[pos])
                vid = int(video_idx[row])
                ts = gt_ts_by_query[q]
                is_gt_vid = vid == int(gt_video_by_q[q])
                y05 = bool(is_gt_vid and correctness_for_ts(si, ei, ts, 0.5))
                y07 = bool(is_gt_vid and correctness_for_ts(si, ei, ts, 0.7))
                slot_candidates.append((vid, si, ei, float(s_c4[row]), y05, y07))
                total_slots += 1
                slot_replacements += int(si != int(start_idx[row]) or ei != int(end_idx[row]))
            kept: List[int] = []
            by_video_kept: Dict[int, List[int]] = {}
            for idx, (vid, si, ei, score, y05, y07) in enumerate(slot_candidates):
                suppress = False
                for prior in by_video_kept.get(vid, []):
                    _, psi, pei, *_ = slot_candidates[prior]
                    if span_iou_idx(si, ei, psi, pei) > nms_thd:
                        suppress = True
                        break
                if not suppress:
                    kept.append(idx)
                    by_video_kept.setdefault(vid, []).append(idx)
                    if len(kept) >= max_after_nms:
                        break
            labs05 = [slot_candidates[i][4] for i in kept]
            labs07 = [slot_candidates[i][5] for i in kept]
            sigs = [(slot_candidates[i][0], slot_candidates[i][1], slot_candidates[i][2]) for i in kept]
            selected_labels["0.5"].append(labs05)
            selected_labels["0.7"].append(labs07)
            selected_signatures.append(sigs)
            query_positive_top100.append(bool(any(labs05[:100]) or any(labs07[:100])))
            top1_sig.append(sigs[0] if sigs else (-1, -1, -1))
        metrics = selected_metrics(selected_labels)
        payload: Dict[str, Any] = {
            "metrics": metrics,
            "candidate_replacement_rate_top100_slots": float(slot_replacements / max(total_slots, 1)),
            "video_multiset_drift_top100_slots": 0.0,
            "video_slot_drift_top100_slots": 0.0,
        }
        if name == "original_candidates":
            original_selected_signatures = selected_signatures
            original_query_positive_top100 = query_positive_top100
            original_top1_sig = top1_sig
            original_slot_sigs = None
        else:
            assert original_query_positive_top100 is not None and original_top1_sig is not None
            exits = sum(int(a and not b) for a, b in zip(original_query_positive_top100, query_positive_top100))
            entries = sum(int((not a) and b) for a, b in zip(original_query_positive_top100, query_positive_top100))
            top1_changed = sum(int(a != b) for a, b in zip(original_top1_sig, top1_sig))
            payload.update({
                "delta_vs_pseudo_C4_final_fixed_video_baseline": {
                    k: float(metrics[k] - out["original_candidates"]["metrics"][k]) for k in METRIC_KEYS
                },
                "r1_improved": bool(any(metrics[k] > out["original_candidates"]["metrics"][k] for k in PRIMARY_R1)),
                "r5_collapse": bool(any(metrics[k] < out["original_candidates"]["metrics"][k] - 0.25 for k in SAFETY_R5)),
                "hard_positive_top100_query_exits": int(exits),
                "hard_positive_top100_query_entries": int(entries),
                "top1_changed_ratio": float(top1_changed / q_count),
            })
        out[name] = payload
    return out


def gate_decision(oracle: Dict[str, Any], pseudo: Dict[str, Any]) -> Dict[str, Any]:
    decisions = {}
    for name in ["boundary_oracle_candidates", "hybrid_candidates"]:
        od = oracle[name]["delta_vs_original_candidates"]
        pd = pseudo[name]["delta_vs_pseudo_C4_final_fixed_video_baseline"]
        r1_delta = min(pd["0.5-r1"], pd["0.7-r1"])
        r5_delta = min(pd["0.5-r5"], pd["0.7-r5"])
        pass_gate = bool(
            od["gt_video_oracle_r05"] > 0
            and od["gt_video_oracle_r07"] > 0
            and od["best_iou_mean"] > 0
            and od["nms_retained_pre_nms_best_iou_rate"] >= 0
            and r1_delta >= 0
            and r5_delta > -0.25
        )
        decisions[name] = {
            "candidate_generation_oracle_positive": bool(
                od["gt_video_oracle_r05"] > 0 and od["gt_video_oracle_r07"] > 0 and od["best_iou_mean"] > 0
            ),
            "r1_primary_delta_min": float(r1_delta),
            "r5_safety_delta_min": float(r5_delta),
            "no_severe_r5_collapse": bool(r5_delta > -0.25),
            "recommend_c6_b1_review": pass_gate,
        }
    any_pass = any(x["recommend_c6_b1_review"] for x in decisions.values())
    return {
        "status": "C6_B0_POSITIVE_FOR_B1_REVIEW" if any_pass else "C6_B0_NEGATIVE",
        "r1_primary_r5_safety_policy": "R@1 is the primary observation; R@5 is a safety/secondary constraint. B0 is not a final VCMR result.",
        "variant_decisions": decisions,
        "official_val_used": False,
        "C6_B1_used": False,
        "C6_C_used": False,
    }


def write_start_state(audit_dir: Path, args: argparse.Namespace) -> None:
    state = {
        "stage": "C6-B0 Candidate Generation Diagnostic",
        "status": "STARTED",
        "C6_A_R2": "negative",
        "fixed_candidate_adapter_line": "stopped",
        "current_scope": "train_calib candidate-generation diagnostic only",
        "primary_focus": "R@1",
        "secondary_safety_focus": "R@5",
        "official_val_used": False,
        "official_val_authorized": False,
        "post_val_adjustment": False,
        "C4_final": "C4-r2-cal-v2.1 v21_00444",
        "C6_B1_used": False,
        "C6_C_used": False,
        "full_backbone_finetuning": False,
        "evaluator_modified": False,
        "inputs": {
            "cache_npz": args.cache_npz,
            "temporal_npz": args.temporal_npz,
            "boundary_oracle_ckpt": args.boundary_oracle_ckpt,
            "gt_jsonl": args.gt_jsonl,
            "nms_thd": args.nms_thd,
            "effective_top_n": args.effective_top_n,
            "max_after_nms": args.max_after_nms,
        },
    }
    write_json(audit_dir / "C6_B0_START_STATE.json", state)
    md = "# C6-B0 start state\n\n"
    md += "- C6-A-R2: `negative`\n"
    md += "- Fixed-candidate adapter line: `stopped`\n"
    md += "- Current scope: `train_calib candidate-generation diagnostic only`\n"
    md += "- Primary observation: `R@1`; secondary/safety: `R@5`\n"
    md += "- Official val used: `false`\n"
    md += "- C6_B1 used: `false`\n"
    md += "- C6_C used: `false`\n"
    md += "- C4_final remains: `C4-r2-cal-v2.1 v21_00444`\n"
    write_text(audit_dir / "C6_B0_START_STATE.md", md)


def render_markdown(payload: Dict[str, Any]) -> str:
    lines = [
        "# C6-B0 Candidate Generation Diagnostic",
        "",
        f"- Status: `{payload['decision']['status']}`",
        "- Scope: `train_calib only`",
        "- Official val used: `false`",
        "- B0 policy: R@1 is primary; R@5 is the safety/secondary check.",
        "- C4_final remains the main frozen result; B0 is not a final VCMR result.",
        "",
        "## Candidate oracle metrics",
        "",
        "```json",
        json.dumps(payload["candidate_oracle"], ensure_ascii=False, indent=2),
        "```",
        "",
        "## Fixed-video pseudo VCMR",
        "",
        "```json",
        json.dumps(payload["pseudo_vcmr"], ensure_ascii=False, indent=2),
        "```",
        "",
        "## Decision",
        "",
        "```json",
        json.dumps(payload["decision"], ensure_ascii=False, indent=2),
        "```",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--cache_npz", default="results/rlem_c6a/cache/train_calib_c6_cache.npz")
    p.add_argument("--temporal_npz", default="results/rlem_c5_lite_prior_prep/train_calib_temporal_prior.npz")
    p.add_argument("--boundary_oracle_ckpt", default="results/rlem_c6a_repair/boundary_oracle/model_best.pt")
    p.add_argument("--gt_jsonl", default="results/rlem_c3_minimal/train_calib_gt.jsonl")
    p.add_argument("--output_dir", default="results/rlem_c6_b0")
    p.add_argument("--audit_dir", default="c6_b0_audit")
    p.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    p.add_argument("--batch_groups", type=int, default=256)
    p.add_argument("--effective_top_n", type=int, default=100)
    p.add_argument("--max_after_nms", type=int, default=100)
    p.add_argument("--nms_thd", type=float, default=0.7)
    p.add_argument("--top_endpoint", type=int, default=32)
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    audit_dir = Path(args.audit_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    audit_dir.mkdir(parents=True, exist_ok=True)
    write_start_state(audit_dir, args)

    prior_npz = out_dir / "train_calib_boundary_oracle_prior.npz"
    infer_audit = infer_boundary_prior(
        cache_npz=args.cache_npz,
        temporal_npz=args.temporal_npz,
        ckpt_path=args.boundary_oracle_ckpt,
        output_npz=str(prior_npz),
        device_name=args.device,
        batch_groups=args.batch_groups,
    )

    cache = {k: v for k, v in np.load(args.cache_npz, allow_pickle=True).items()}
    with np.load(args.temporal_npz, allow_pickle=False) as temporal:
        temporal_length = temporal["temporal_length"].astype(np.int64)
    with np.load(prior_npz, allow_pickle=False) as prior:
        p_b_new = prior["p_b_new"].astype(np.float32)
        p_e_new = prior["p_e_new"].astype(np.float32)
    gt_ts_by_query = load_gt_ts(args.gt_jsonl, cache["desc_ids"])

    t0 = time.time()
    variants, gen_audit = build_group_candidates(
        cache=cache,
        temporal_length=temporal_length,
        p_b_new=p_b_new,
        p_e_new=p_e_new,
        gt_ts_by_query=gt_ts_by_query,
        nms_thd=args.nms_thd,
        top_endpoint=args.top_endpoint,
    )
    gen_audit["elapsed_sec"] = float(time.time() - t0)
    oracle = candidate_oracle_eval(cache=cache, variants=variants, gt_ts_by_query=gt_ts_by_query)
    pseudo = pseudo_vcmr_eval(
        cache=cache,
        variants=variants,
        gt_ts_by_query=gt_ts_by_query,
        effective_top_n=args.effective_top_n,
        max_after_nms=args.max_after_nms,
        nms_thd=args.nms_thd,
    )
    decision = gate_decision(oracle, pseudo)
    payload = {
        "stage": "C6-B0 Candidate Generation Diagnostic",
        "official_val_used": False,
        "post_val_adjustment": False,
        "C6_B1_used": False,
        "C6_C_used": False,
        "C4_final": "C4-r2-cal-v2.1 v21_00444",
        "primary_focus": "R@1",
        "secondary_safety_focus": "R@5",
        "inputs": {
            "cache_npz": artifact(args.cache_npz),
            "temporal_npz": artifact(args.temporal_npz),
            "boundary_oracle_ckpt": artifact(args.boundary_oracle_ckpt),
            "gt_jsonl": artifact(args.gt_jsonl),
            "boundary_prior_npz": artifact(prior_npz),
        },
        "runtime": {
            "boundary_inference": infer_audit,
            "candidate_generation": gen_audit,
        },
        "candidate_oracle": oracle,
        "pseudo_vcmr": pseudo,
        "decision": decision,
    }

    write_json(audit_dir / "C6_B0_CANDIDATE_DIAGNOSTIC.json", payload)
    write_text(audit_dir / "C6_B0_CANDIDATE_DIAGNOSTIC.md", render_markdown(payload))
    summary = "# C6-B0 final summary\n\n"
    summary += f"- Status: `{decision['status']}`\n"
    summary += "- Official val used: `false`\n"
    summary += "- C6_B1 used: `false`\n"
    summary += "- C6_C used: `false`\n"
    summary += "- Primary readout: `R@1`; R@5 is treated as safety/secondary.\n"
    summary += "- C4_final remains frozen as `C4-r2-cal-v2.1 v21_00444`.\n\n"
    summary += "## Decision JSON\n\n```json\n" + json.dumps(decision, ensure_ascii=False, indent=2) + "\n```\n"
    write_text(audit_dir / "C6_B0_FINAL_SUMMARY.md", summary)
    manifest = {
        "status": decision["status"],
        "stage": "C6-B0 Candidate Generation Diagnostic",
        "C6_A_R2": "negative",
        "fixed_candidate_adapter_line": "stopped",
        "C4_final_retained": True,
        "official_val_used": False,
        "official_val_authorized": False,
        "post_val_adjustment": False,
        "C6_B1_used": False,
        "C6_C_used": False,
        "full_backbone_finetuning": False,
        "evaluator_modified": False,
        "R1_primary": True,
        "R5_secondary_safety": True,
        "decision": decision,
    }
    write_json(audit_dir / "C6_B0_MANIFEST.json", manifest)
    hash_targets = [
        audit_dir / "C6_B0_START_STATE.md",
        audit_dir / "C6_B0_START_STATE.json",
        audit_dir / "C6_B0_CANDIDATE_DIAGNOSTIC.md",
        audit_dir / "C6_B0_CANDIDATE_DIAGNOSTIC.json",
        audit_dir / "C6_B0_FINAL_SUMMARY.md",
        audit_dir / "C6_B0_MANIFEST.json",
        prior_npz,
    ]
    write_json(audit_dir / "C6_B0_HASHES.json", {str(p): artifact(p) for p in hash_targets})
    print(json.dumps({"status": decision["status"], "decision": decision}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
