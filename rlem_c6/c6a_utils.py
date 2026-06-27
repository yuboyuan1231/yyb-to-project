from __future__ import annotations

import gzip
import hashlib
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from rlem.c4_lite_utils import ALL_METRICS, PRIMARY_METRICS, selection_score  # noqa: E402
from rlem.c5_main_a_utils import exact_candidate_indices  # noqa: E402
from rlem.c5_prior_utils import (  # noqa: E402
    attach_eval_labels,
    fast_metrics,
    localization_diagnostics,
    metric_deltas,
    movement_diagnostics,
)

ROW_FEATURE_KEYS = [
    "ctx_mass", "ctx_mean", "ctx_max", "ctx_peak_inside", "ctx_boundary_agree", "ctx_center_proximity",
    "bd_mass", "bd_mean", "bd_max", "bd_peak_inside", "bd_boundary_agree", "bd_center_proximity",
    "retctx_mass", "retbd_mass", "retloc_mass", "retloc_mean", "retloc_max", "retloc_peak_inside",
    "r_video_z", "r_video_sig", "margin_z", "rank_inv",
    "term_retctx_mass", "term_retbd_mass", "term_retctx_peak_inside", "term_retbd_peak_inside",
    "term_retctx_boundary_agree", "term_retbd_boundary_agree",
    "term_retctx_center_proximity", "term_retbd_center_proximity",
    "term_low_conf_efp_neg", "term_low_conf_unc_neg",
    "term_margin_z_x_retbd_mass", "term_rank_inv_x_retbd_mass",
]


def seed_everything(seed: int = 13) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


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


def write_json(path: str | Path, obj: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(p) + ".partial")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, p)


def write_text(path: str | Path, text: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(p) + ".partial")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, p)


def iter_jsonl(path: str | Path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def load_config(path: str | Path) -> Dict[str, Any]:
    text = Path(path).read_text(encoding="utf-8")
    try:
        import yaml
        return yaml.safe_load(text)
    except Exception:
        out: Dict[str, Any] = {}
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith("#") or ":" not in line:
                continue
            k, v = line.split(":", 1)
            v = v.strip()
            if v.lower() in {"true", "false"}:
                out[k.strip()] = v.lower() == "true"
            else:
                try:
                    out[k.strip()] = int(v)
                except ValueError:
                    try:
                        out[k.strip()] = float(v)
                    except ValueError:
                        out[k.strip()] = v
        return out


def build_c6_label_cache(
    *,
    split_name: str,
    evidence_jsonl: str,
    desc_ids_filter: str,
    c5_features_npz: str,
    temporal_prior_npz: str,
    video_dataset_npz: str,
    output_npz: str,
    manifest_json: str,
    expected_rows_per_query: int = 200,
) -> Dict[str, Any]:
    allowed = {line.strip() for line in Path(desc_ids_filter).read_text().splitlines() if line.strip()}
    out = Path(output_npz)
    if out.exists():
        raise FileExistsError(out)
    t0 = time.time()
    with np.load(c5_features_npz, allow_pickle=False) as cf:
        rows = int(len(cf["row_group_id"]))
        queries = rows // expected_rows_per_query
        if rows != len(allowed) * expected_rows_per_query:
            raise ValueError(f"{split_name}: rows {rows} != desc_ids*200 {len(allowed) * expected_rows_per_query}")
        row_group_id = cf["row_group_id"].astype(np.int32)
        query_index = cf["query_index"].astype(np.int32)
        video_idx = cf["video_idx"].astype(np.int32)
        start_time = cf["start_time"].astype(np.float32)
        end_time = cf["end_time"].astype(np.float32)
        s_c4_final = cf["s_c4_final"].astype(np.float32)
        row_feature_names = [k for k in ROW_FEATURE_KEYS if k in cf.files]
        row_feature_mean = np.asarray([float(np.mean(cf[k].astype(np.float64))) for k in row_feature_names], dtype=np.float32)
        row_feature_std = np.asarray([float(max(np.std(cf[k].astype(np.float64)), 1e-8)) for k in row_feature_names], dtype=np.float32)
        row_feature_nonfinite = int(sum((~np.isfinite(cf[k])).sum() for k in row_feature_names))
    with np.load(temporal_prior_npz, allow_pickle=False) as tp:
        group_id = tp["group_id"].astype(np.int32)
        temporal_length = tp["temporal_length"].astype(np.int16)
        temporal_bad = int(
            (~np.isfinite(tp["p_b"])).sum()
            + (~np.isfinite(tp["p_e"])).sum()
            + (~np.isfinite(tp["p_ctx"])).sum()
            + (tp["p_b"] < 0).sum()
            + (tp["p_e"] < 0).sum()
            + (tp["p_ctx"] < 0).sum()
        )
        temporal_sum_err = float(max(
            np.max(np.abs(tp["p_b"].sum(axis=1) - 1.0)),
            np.max(np.abs(tp["p_e"].sum(axis=1) - 1.0)),
            np.max(np.abs(tp["p_ctx"].sum(axis=1) - 1.0)),
        ))
    with np.load(video_dataset_npz, allow_pickle=True) as vd:
        if not np.array_equal(vd["group_id"].astype(np.int32), group_id):
            raise ValueError("video dataset group order differs from temporal prior")
        video_features = vd["features"].astype(np.float32)
        video_feature_names = vd["feature_names"].astype(object)
        label_relevant = vd["label_relevant"].astype(np.float32)
        label_best_iou = vd["label_best_iou"].astype(np.float32)
        label_any_05 = vd["label_any_05"].astype(np.float32)
        label_any_07 = vd["label_any_07"].astype(np.float32)
    if int(row_group_id.max()) + 1 != len(group_id):
        raise ValueError("row/group count mismatch")
    if not np.array_equal(query_index, np.repeat(np.arange(queries, dtype=np.int32), expected_rows_per_query)):
        raise ValueError("query_index is not contiguous 200/query")

    desc_ids = np.empty(queries, dtype=object)
    desc_text = np.empty(queries, dtype=object)
    iou = np.empty(rows, dtype=np.float32)
    is_gt_video = np.empty(rows, dtype=np.float32)
    y05 = np.empty(rows, dtype=np.float32)
    y07 = np.empty(rows, dtype=np.float32)
    gt_start_idx = np.empty(rows, dtype=np.int16)
    gt_end_idx = np.empty(rows, dtype=np.int16)
    pos = 0
    observed = set()
    bad_alignment = 0
    for row in iter_jsonl(evidence_jsonl):
        did = str(row["desc_id"])
        if did not in allowed:
            continue
        if pos >= rows:
            raise ValueError("filtered evidence has more rows than C5 features")
        q = pos // expected_rows_per_query
        if pos % expected_rows_per_query == 0:
            desc_ids[q] = row["desc_id"]
            desc_text[q] = row.get("desc", "")
            observed.add(did)
        bad_alignment += int(int(row["video_idx"]) != int(video_idx[pos]))
        bad_alignment += int(
            abs(float(row["start_time"]) - float(start_time[pos])) > 1e-5
            or abs(float(row["end_time"]) - float(end_time[pos])) > 1e-5
        )
        val_iou = float(row.get("iou", 0.0) or 0.0)
        val_gt = float(row.get("is_gt_video", 0.0) or 0.0)
        iou[pos] = val_iou
        is_gt_video[pos] = val_gt
        y05[pos] = float(row.get("y_joint_05", 1.0 if val_gt > 0.5 and val_iou >= 0.5 else 0.0) or 0.0)
        y07[pos] = float(row.get("y_joint_07", 1.0 if val_gt > 0.5 and val_iou >= 0.7 else 0.0) or 0.0)
        gt = row.get("gt_ts") or [0.0, 0.0]
        gs = int(round(float(gt[0]) / 1.5))
        ge = max(gs, int(round(float(gt[1]) / 1.5)) - 1)
        gt_start_idx[pos] = max(0, min(gs, 99))
        gt_end_idx[pos] = max(0, min(ge, 99))
        pos += 1
    if pos != rows:
        raise ValueError(f"filtered rows {pos} != expected {rows}")
    desc_offsets = np.arange(0, rows + 1, expected_rows_per_query, dtype=np.int64)
    start_idx, end_idx, endpoint_audit = exact_candidate_indices(start_time, end_time, row_group_id, temporal_length)
    sort_idx = np.argsort(row_group_id, kind="stable").astype(np.int64)
    sorted_gid = row_group_id[sort_idx]
    group_offsets = np.r_[0, np.flatnonzero(sorted_gid[1:] != sorted_gid[:-1]) + 1].astype(np.int64)
    group_ids_sorted_unique = sorted_gid[group_offsets].astype(np.int32)
    group_counts = np.bincount(row_group_id, minlength=len(group_id))
    video_group_keys = np.column_stack([
        np.bincount(row_group_id, weights=query_index, minlength=len(group_id)) / np.maximum(group_counts, 1),
        np.bincount(row_group_id, weights=video_idx, minlength=len(group_id)) / np.maximum(group_counts, 1),
    ]).astype(np.int64)

    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(out) + ".partial")
    with open(tmp, "wb") as f:
        np.savez_compressed(
            f,
            desc_ids=desc_ids,
            desc_text=desc_text,
            desc_offsets=desc_offsets,
            row_group_id=row_group_id,
            query_index=query_index,
            video_idx=video_idx,
            start_time=start_time,
            end_time=end_time,
            start_idx=start_idx,
            end_idx=end_idx,
            s_c4_final=s_c4_final,
            iou=iou,
            is_gt_video=is_gt_video,
            y_joint_05=y05,
            y_joint_07=y07,
            gt_start_idx=gt_start_idx,
            gt_end_idx=gt_end_idx,
            group_sort_idx=sort_idx,
            group_offsets=group_offsets,
            group_ids_sorted_unique=group_ids_sorted_unique,
            video_group_keys=video_group_keys,
            video_features=video_features,
            video_feature_names=video_feature_names,
            label_relevant=label_relevant,
            label_best_iou=label_best_iou,
            label_any_05=label_any_05,
            label_any_07=label_any_07,
            row_feature_names=np.asarray(row_feature_names, dtype=object),
            row_feature_mean=row_feature_mean,
            row_feature_std=row_feature_std,
        )
    os.replace(tmp, out)
    missing = len(allowed - observed)
    nonfinite = int((~np.isfinite(iou)).sum() + (~np.isfinite(s_c4_final)).sum() + row_feature_nonfinite)
    manifest = {
        "status": "PASS" if not (bad_alignment or missing or temporal_bad or nonfinite) else "FAIL",
        "split_name": split_name,
        "queries": int(queries),
        "rows": int(rows),
        "groups": int(len(group_id)),
        "rows_per_query": int(expected_rows_per_query),
        "bad_alignment_count": int(bad_alignment),
        "missing_filtered_queries": int(missing),
        "nonfinite_values": int(nonfinite),
        "temporal_bad_values": int(temporal_bad),
        "temporal_sum_max_abs_error": temporal_sum_err,
        "endpoint_index_audit": endpoint_audit,
        "row_feature_count": len(row_feature_names),
        "row_feature_names": row_feature_names,
        "source_features": artifact(c5_features_npz),
        "source_temporal_prior": artifact(temporal_prior_npz),
        "source_video_dataset": artifact(video_dataset_npz),
        "desc_ids_filter": artifact(desc_ids_filter),
        "evidence_jsonl_sha256": sha256_file(evidence_jsonl),
        "output_npz": artifact(out),
        "official_val_used": False,
        "elapsed_sec": time.time() - t0,
    }
    write_json(manifest_json, manifest)
    return manifest


class C6GroupDataset(Dataset):
    def __init__(self, cache_npz: str, features_npz: str, temporal_npz: str, max_groups: Optional[int] = None):
        with np.load(cache_npz, allow_pickle=True) as cache:
            self.row_features = [str(x) for x in cache["row_feature_names"].tolist()]
            self.mean = cache["row_feature_mean"].astype(np.float32)[:len(self.row_features)]
            self.std = cache["row_feature_std"].astype(np.float32)[:len(self.row_features)]
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
        # The C5 feature and temporal-prior inputs are compressed NPZ files.
        # Random group access directly through NpzFile would repeatedly
        # decompress whole arrays.  Preload the fixed tensors once so the hot
        # training path is simple slicing + GPU transfer.
        with np.load(features_npz, allow_pickle=False) as features:
            missing = [k for k in self.row_features if k not in features.files]
            if missing:
                raise KeyError(f"C6 features missing arrays: {missing}")
            cols = []
            for k, mean, std in zip(self.row_features, self.mean, self.std):
                col = features[k].astype(np.float32)
                cols.append(((col - mean) / max(float(std), 1e-6)).astype(np.float32, copy=False))
            self.row_feature_matrix = np.column_stack(cols).astype(np.float32, copy=False)
        with np.load(temporal_npz, allow_pickle=False) as temporal:
            self.p_b = temporal["p_b"].astype(np.float32)
            self.p_e = temporal["p_e"].astype(np.float32)
            self.p_ctx = temporal["p_ctx"].astype(np.float32)
            self.temporal_length = temporal["temporal_length"].astype(np.int64)
        self.max_groups = max_groups

    def __len__(self) -> int:
        n = len(self.group_offsets)
        return min(n, int(self.max_groups)) if self.max_groups is not None else n

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        s = int(self.group_offsets[idx])
        e = int(self.group_offsets[idx + 1]) if idx + 1 < len(self.group_offsets) else len(self.sort_idx)
        rows = self.sort_idx[s:e]
        gid = int(self.group_ids[idx])
        feats = self.row_feature_matrix[rows]
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
            "row_feat": feats,
            "iou": self.iou[rows],
            "y05": self.y05[rows],
            "y07": self.y07[rows],
            "is_gt_video": self.is_gt_video[rows],
            "gt_start_idx": int(self.gt_start_idx[rows[0]]),
            "gt_end_idx": int(self.gt_end_idx[rows[0]]),
            "label_relevant": float(self.label_relevant[gid]),
            "video_feat": self.video_features[gid],
        }


def _pad1(arrs, dtype=torch.float32, value=0.0):
    width = max(len(a) for a in arrs)
    out = torch.full((len(arrs), width), value, dtype=dtype)
    mask = torch.zeros((len(arrs), width), dtype=torch.bool)
    for i, arr in enumerate(arrs):
        t = torch.as_tensor(arr, dtype=dtype)
        out[i, :len(arr)] = t
        mask[i, :len(arr)] = True
    return out, mask


def c6_collate(items: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    pb, tmask = _pad1([x["p_b"] for x in items])
    pe, _ = _pad1([x["p_e"] for x in items])
    pc, _ = _pad1([x["p_ctx"] for x in items])
    bsz = len(items)
    max_n = max(len(x["s_c4"]) for x in items)
    row_dim = items[0]["row_feat"].shape[1]
    span_mask = torch.zeros((bsz, max_n), dtype=torch.bool)
    si = torch.zeros((bsz, max_n), dtype=torch.long)
    ei = torch.zeros((bsz, max_n), dtype=torch.long)
    rows = torch.full((bsz, max_n), -1, dtype=torch.long)
    s_c4 = torch.full((bsz, max_n), -1e9, dtype=torch.float32)
    row_feat = torch.zeros((bsz, max_n, row_dim), dtype=torch.float32)
    iou = torch.zeros((bsz, max_n), dtype=torch.float32)
    y05 = torch.zeros((bsz, max_n), dtype=torch.float32)
    y07 = torch.zeros((bsz, max_n), dtype=torch.float32)
    isgt = torch.zeros((bsz, max_n), dtype=torch.float32)
    for b, x in enumerate(items):
        n = len(x["s_c4"])
        span_mask[b, :n] = True
        si[b, :n] = torch.as_tensor(x["start_idx"])
        ei[b, :n] = torch.as_tensor(x["end_idx"])
        rows[b, :n] = torch.as_tensor(x["rows"])
        s_c4[b, :n] = torch.as_tensor(x["s_c4"])
        row_feat[b, :n] = torch.as_tensor(x["row_feat"])
        iou[b, :n] = torch.as_tensor(x["iou"])
        y05[b, :n] = torch.as_tensor(x["y05"])
        y07[b, :n] = torch.as_tensor(x["y07"])
        isgt[b, :n] = torch.as_tensor(x["is_gt_video"])
    return {
        "p_b": pb, "p_e": pe, "p_ctx": pc, "time_mask": tmask,
        "start_idx": si, "end_idx": ei, "span_mask": span_mask, "rows": rows,
        "s_c4": s_c4, "row_feat": row_feat, "iou": iou, "y05": y05, "y07": y07,
        "is_gt_video": isgt,
        "gt_start_idx": torch.tensor([x["gt_start_idx"] for x in items], dtype=torch.long),
        "gt_end_idx": torch.tensor([x["gt_end_idx"] for x in items], dtype=torch.long),
        "label_relevant": torch.tensor([x["label_relevant"] for x in items], dtype=torch.float32),
        "video_feat": torch.stack([torch.as_tensor(x["video_feat"], dtype=torch.float32) for x in items]),
        "group_id": torch.tensor([x["group_id"] for x in items], dtype=torch.long),
    }


class TemporalBlock(nn.Module):
    def __init__(self, hidden_dim: int, heads: int, ffn_dim: int, dropout: float):
        super().__init__()
        self.attn = nn.MultiheadAttention(hidden_dim, heads, dropout=dropout, batch_first=True)
        self.conv = nn.Conv1d(hidden_dim, hidden_dim, 5, padding=2, groups=hidden_dim)
        self.ff = nn.Sequential(nn.Linear(hidden_dim, ffn_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(ffn_dim, hidden_dim))
        self.n1 = nn.LayerNorm(hidden_dim)
        self.n2 = nn.LayerNorm(hidden_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, mask):
        y, _ = self.attn(x, x, x, key_padding_mask=~mask, need_weights=False)
        x = self.n1(x + self.drop(y))
        x = self.n1(x + self.drop(self.conv(x.transpose(1, 2)).transpose(1, 2)))
        return self.n2(x + self.drop(self.ff(x)))


def mlp(in_dim: int, hidden: int, out_dim: int, layers: int, dropout: float) -> nn.Sequential:
    mods: List[nn.Module] = []
    d = in_dim
    for _ in range(max(1, layers - 1)):
        mods += [nn.Linear(d, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(dropout)]
        d = hidden
    mods.append(nn.Linear(d, out_dim))
    return nn.Sequential(*mods)


class C6AdapterModel(nn.Module):
    def __init__(
        self,
        row_feat_dim: int,
        video_feat_dim: int,
        hidden_dim: int = 256,
        temporal_layers: int = 4,
        heads: int = 8,
        ffn_dim: int = 512,
        span_head_hidden: int = 384,
        span_head_layers: int = 3,
        dropout: float = 0.1,
        scale_b: float = 0.5,
        scale_e: float = 0.5,
        alpha_rank: float = 0.05,
        alpha_bd: float = 0.05,
        alpha_q: float = 0.02,
    ):
        super().__init__()
        self.scale_b, self.scale_e = float(scale_b), float(scale_e)
        self.alpha_rank, self.alpha_bd, self.alpha_q = float(alpha_rank), float(alpha_bd), float(alpha_q)
        self.inp = nn.Sequential(nn.Linear(7, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU())
        self.blocks = nn.ModuleList([TemporalBlock(hidden_dim, heads, ffn_dim, dropout) for _ in range(temporal_layers)])
        self.boundary = nn.Linear(hidden_dim, 4)
        self.video_gate = mlp(video_feat_dim + 6, hidden_dim, 1, 2, dropout)
        self.span_head = mlp(hidden_dim * 4 + row_feat_dim + 6, span_head_hidden, 5, span_head_layers, dropout)

    def forward(self, batch: Dict[str, torch.Tensor], zero_adapter: bool = False) -> Dict[str, torch.Tensor]:
        pb, pe, pc, mask = batch["p_b"], batch["p_e"], batch["p_ctx"], batch["time_mask"]
        bsz, t = pb.shape
        eps = 1e-8
        pos = torch.linspace(0, 1, t, device=pb.device)[None, :].expand(bsz, t)
        ent = (-(pc.clamp_min(eps).log() * pc).masked_fill(~mask, 0).sum(1, keepdim=True) / math.log(max(t, 2))).expand(bsz, t)
        x = torch.stack([pb, pe, pc, pb.clamp_min(eps).log(), pe.clamp_min(eps).log(), pos, ent], dim=-1)
        h = self.inp(x)
        for blk in self.blocks:
            h = blk(h, mask)
        bd = self.boundary(h)
        delta_b_raw, delta_e_raw, gate_b_raw, gate_e_raw = bd[..., 0], bd[..., 1], bd[..., 2], bd[..., 3]
        bbase = pb.clamp_min(eps).log().masked_fill(~mask, -1e9)
        ebase = pe.clamp_min(eps).log().masked_fill(~mask, -1e9)
        if zero_adapter:
            bnew = bbase - torch.logsumexp(bbase, dim=1, keepdim=True)
            enew = ebase - torch.logsumexp(ebase, dim=1, keepdim=True)
        else:
            braw = (bbase + self.scale_b * torch.tanh(delta_b_raw) * torch.sigmoid(gate_b_raw)).masked_fill(~mask, -1e9)
            eraw = (ebase + self.scale_e * torch.tanh(delta_e_raw) * torch.sigmoid(gate_e_raw)).masked_fill(~mask, -1e9)
            bnew = braw - torch.logsumexp(braw, dim=1, keepdim=True)
            enew = eraw - torch.logsumexp(eraw, dim=1, keepdim=True)
        si = batch["start_idx"].clamp(0, t - 1)
        ei = batch["end_idx"].clamp(0, t - 1)
        n = si.shape[1]
        br = torch.arange(bsz, device=si.device)[:, None]
        hi, hj = h[br, si], h[br, ei]
        cs = torch.cat([torch.zeros(bsz, 1, h.shape[-1], device=h.device, dtype=h.dtype), torch.cumsum(h, dim=1)], dim=1)
        ep1 = (ei + 1).clamp(max=t)
        hmean = (cs[br, ep1] - cs[br, si]) / (ep1 - si).clamp_min(1).unsqueeze(-1)
        hmax = torch.maximum(hi, hj)
        endpoint_new = bnew[br, si] + enew[br, ei]
        endpoint_base = bbase[br, si] + ebase[br, ei]
        endpoint_delta = endpoint_new - endpoint_base
        s_c4 = batch["s_c4"]
        vg_in = torch.cat([
            batch["video_feat"],
            pb.max(1).values[:, None],
            pe.max(1).values[:, None],
            pc.max(1).values[:, None],
            (-(pb.clamp_min(eps).log() * pb).masked_fill(~mask, 0).sum(1))[:, None],
            (-(pe.clamp_min(eps).log() * pe).masked_fill(~mask, 0).sum(1))[:, None],
            s_c4.masked_fill(~batch["span_mask"], -1e9).max(1).values[:, None],
        ], dim=1)
        g = torch.sigmoid(self.video_gate(vg_in)).squeeze(-1)
        span_len = (ei - si + 1).float() / float(max(t, 1))
        span_x = torch.cat([
            hi, hj, hmean, hmax, batch["row_feat"],
            endpoint_delta.unsqueeze(-1), span_len.unsqueeze(-1), s_c4.unsqueeze(-1),
            g[:, None, None].expand(bsz, n, 1), endpoint_new.unsqueeze(-1), endpoint_base.unsqueeze(-1),
        ], dim=-1)
        logits = self.span_head(span_x)
        quality, iou05, iou07, fp, rank = logits[..., 0], logits[..., 1], logits[..., 2], logits[..., 3], logits[..., 4]
        score = s_c4 if zero_adapter else s_c4 + self.alpha_rank * g[:, None] * rank + self.alpha_bd * endpoint_delta + self.alpha_q * quality
        score = score.masked_fill(~batch["span_mask"], -1e9)
        return {
            "score": score,
            "quality_logit": quality,
            "iou05_logit": iou05,
            "iou07_logit": iou07,
            "fp_risk_logit": fp,
            "rank_residual": rank,
            "p_b_new": torch.exp(bnew).masked_fill(~mask, 0),
            "p_e_new": torch.exp(enew).masked_fill(~mask, 0),
            "g_video": g,
            "endpoint_delta": endpoint_delta,
            "delta_b_raw": delta_b_raw,
            "delta_e_raw": delta_e_raw,
            "gate_b": torch.sigmoid(gate_b_raw),
            "gate_e": torch.sigmoid(gate_e_raw),
        }


def c6_loss(batch: Dict[str, torch.Tensor], out: Dict[str, torch.Tensor], cfg: Dict[str, Any]):
    mask = batch["span_mask"]
    iou = batch["iou"].clamp_min(0)
    target = (iou + 0.05 * batch["y05"] + 0.02 * batch["y07"]).masked_fill(~mask, 0)
    target = target / target.sum(1, keepdim=True).clamp_min(1e-6)
    list_loss = -(target * F.log_softmax(out["score"].masked_fill(~mask, -1e9), dim=1)).sum(1).mean()
    bce05 = F.binary_cross_entropy_with_logits(out["iou05_logit"][mask], batch["y05"][mask])
    bce07 = F.binary_cross_entropy_with_logits(out["iou07_logit"][mask], batch["y07"][mask])
    quality = F.smooth_l1_loss(torch.sigmoid(out["quality_logit"][mask]), iou[mask])
    fp = F.binary_cross_entropy_with_logits(out["fp_risk_logit"][mask], 1.0 - batch["is_gt_video"][mask])
    bsz, t = batch["p_b"].shape
    idx = torch.arange(t, device=batch["p_b"].device)[None, :]
    sig = 1.5
    ts = batch["gt_start_idx"].clamp(0, t - 1)[:, None]
    te = batch["gt_end_idx"].clamp(0, t - 1)[:, None]
    tb = torch.exp(-0.5 * ((idx - ts.float()) / sig) ** 2) * batch["time_mask"]
    tegt = torch.exp(-0.5 * ((idx - te.float()) / sig) ** 2) * batch["time_mask"]
    tb = tb / tb.sum(1, keepdim=True).clamp_min(1e-8)
    tegt = tegt / tegt.sum(1, keepdim=True).clamp_min(1e-8)
    rel = batch["label_relevant"] > 0.5
    bd = (
        -(tb[rel] * out["p_b_new"][rel].clamp_min(1e-8).log()).sum(1).mean()
        -(tegt[rel] * out["p_e_new"][rel].clamp_min(1e-8).log()).sum(1).mean()
    ) if rel.any() else list_loss.new_tensor(0.0)
    stab = ((out["score"][mask] - batch["s_c4"][mask]) ** 2).mean()
    sparse = out["g_video"].mean() + out["endpoint_delta"][mask].abs().mean()
    total = (
        list_loss
        + float(cfg.get("lambda_iou", 1.0)) * (bce05 + 0.5 * bce07 + 0.5 * quality)
        + float(cfg.get("lambda_fp", 0.25)) * fp
        + float(cfg.get("lambda_bd", 0.25)) * bd
        + float(cfg.get("lambda_stab", 0.05)) * stab
        + float(cfg.get("lambda_sparse", 0.001)) * sparse
    )
    return total, {
        "loss": float(total.detach().cpu()),
        "list": float(list_loss.detach().cpu()),
        "bce05": float(bce05.detach().cpu()),
        "bce07": float(bce07.detach().cpu()),
        "quality": float(quality.detach().cpu()),
        "fp": float(fp.detach().cpu()),
        "bd": float(bd.detach().cpu()),
        "stab": float(stab.detach().cpu()),
    }


def move_batch(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}


def score_dataset(model: nn.Module, dataset: C6GroupDataset, device: str = "cuda", batch_groups: int = 192, amp: bool = True) -> np.ndarray:
    dev = torch.device("cuda" if device == "cuda" and torch.cuda.is_available() else "cpu")
    model.eval()
    scores = np.empty(len(dataset.s_c4_final), dtype=np.float32)
    dl = DataLoader(dataset, batch_size=batch_groups, shuffle=False, num_workers=0, collate_fn=c6_collate, pin_memory=(dev.type == "cuda"))
    with torch.no_grad():
        for batch in dl:
            rows = batch.pop("rows").numpy()
            tb = move_batch(batch, dev)
            with torch.cuda.amp.autocast(enabled=(amp and dev.type == "cuda")):
                out = model(tb)
            sc = out["score"].detach().float().cpu().numpy()
            for b in range(rows.shape[0]):
                valid = rows[b] >= 0
                scores[rows[b, valid]] = sc[b, valid]
    return scores


def evaluate_score(cache_npz: str, score: np.ndarray, gt_jsonl: str, baseline_score: Optional[np.ndarray] = None) -> Dict[str, Any]:
    cache = {k: v for k, v in np.load(cache_npz, allow_pickle=True).items()}
    attach_eval_labels(cache, gt_jsonl)
    metrics, _, _ = fast_metrics(cache, score, 100, 100, 0.7)
    out: Dict[str, Any] = {"metrics": metrics}
    if baseline_score is not None:
        bm, _, _ = fast_metrics(cache, baseline_score, 100, 100, 0.7)
        out["baseline_metrics"] = bm
        out["deltas_vs_c4_final"] = metric_deltas(metrics, bm)
        out["selection_delta_vs_c4_final"] = float(selection_score(metrics, bm))
        out["movement"] = movement_diagnostics(cache, baseline_score, score)
        out["localization"] = localization_diagnostics(cache, baseline_score, score)
    return out


def freeze_gates(record: Dict[str, Any]) -> Dict[str, Any]:
    d = record.get("deltas_vs_c4_final", {})
    m = record.get("movement", {})
    loc = record.get("localization", {})
    front_pos = sum(float(d.get(k, 0.0)) > 0.0 for k in PRIMARY_METRICS)
    r1_gate = (
        float(d.get("0.5-r1", -9.0)) > 0 and float(d.get("0.7-r1", -9.0)) >= -1e-9
    ) or (
        float(d.get("0.7-r1", -9.0)) > 0 and float(d.get("0.5-r1", -9.0)) >= -1e-9
    )
    r100_gate = float(d.get("0.5-r100", -9.0)) >= -0.05 and float(d.get("0.7-r100", -9.0)) >= -0.05
    loc_pos = sum([
        loc.get("oracle_video_r1_05_delta", 0.0) > 0,
        loc.get("oracle_video_r1_07_delta", 0.0) > 0,
        loc.get("selected_span_miou_delta", 0.0) > 0,
        loc.get("best_iou_span_rank_delta_mean", 1.0) < 0,
    ])
    move_ok = (
        m.get("hard_positive_top100_exit_ratio", 1.0) <= 0.005
        and m.get("pearson_base_candidate", 0.0) >= 0.98
        and m.get("mean_within_query_spearman", 0.0) >= 0.95
    )
    core = front_pos >= 4 and r1_gate and r100_gate and loc_pos >= 2 and move_ok
    strong = all(float(d.get(k, -9.0)) >= -1e-9 for k in ALL_METRICS) and front_pos == 6 and loc_pos >= 3 and move_ok
    return {
        "front_positive_count": int(front_pos),
        "r1_gate": bool(r1_gate),
        "r100_gate": bool(r100_gate),
        "localization_positive_count": int(loc_pos),
        "movement_gate": bool(move_ok),
        "core_gate": bool(core),
        "strong_gate": bool(strong),
    }
