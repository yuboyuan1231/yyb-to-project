from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from tools.c26.prem_feature_dataset import C26_CACHE, jsonable, write_json
from tools.c26.prem_losses import prem_total_loss
from tools.c26.prem_modules import PREMCollaborativeRetriever, diagnostics_from_batch


_ARRAY_CACHE: Dict[str, Dict[str, np.ndarray]] = {}


def seed_all(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def within_query_z(query_ids: np.ndarray, values: np.ndarray) -> np.ndarray:
    out = np.zeros(len(values), dtype=np.float32)
    order = np.argsort(query_ids, kind="mergesort")
    q_sorted = query_ids[order]
    v_sorted = values[order].astype(np.float32)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and q_sorted[end] == q_sorted[start]:
            end += 1
        x = v_sorted[start:end]
        out[order[start:end]] = (x - float(x.mean())) / max(float(x.std()), 1e-6)
        start = end
    return out.astype(np.float32)


def load_npz_arrays(path: Path) -> Dict[str, np.ndarray]:
    key = str(path)
    if key in _ARRAY_CACHE:
        return _ARRAY_CACHE[key]
    data = np.load(path, allow_pickle=True)
    arr = {k: data[k] for k in data.files}
    _ARRAY_CACHE[key] = arr
    return arr


def train_retriever(
    arrays_path: Path,
    mode: str,
    seed: int,
    epochs: int,
    batch_size: int,
    lr: float,
    device: str,
    force: bool = False,
) -> Tuple[Path, Dict[str, Any]]:
    seed_all(seed)
    C26_CACHE.mkdir(parents=True, exist_ok=True)
    ckpt_path = C26_CACHE / f"C26_PREM_RETRIEVER_{mode}_seed{seed}.local.pt"
    result_path = C26_CACHE / f"C26_PREM_RETRIEVER_{mode}_seed{seed}.training.json"
    if ckpt_path.exists() and result_path.exists() and not force:
        return ckpt_path, jsonable(__import__("json").loads(result_path.read_text()))
    arr = load_npz_arrays(arrays_path)
    split = arr["split_ids"].astype(np.int64)
    train_mask = split == 0
    if not np.any(train_mask):
        raise RuntimeError("C26 training arrays have no train_fit rows")
    q_idx = arr["q_idx"][train_mask].astype(np.int64)
    v_idx = arr["v_idx"][train_mask].astype(np.int64)
    labels = arr["y"][train_mask].astype(np.float32)
    qtype = arr["qtype"][train_mask].astype(np.int64)
    first_z = within_query_z(arr["query_ids"][train_mask].astype(np.int64), arr["first_stage"][train_mask].astype(np.float32))
    rank = arr["rank"][train_mask].astype(np.float32)
    # Keep all positives and a dense front/hard-negative subset for speed. This
    # preserves top-ranked wrong videos instead of sampling random easy rows.
    keep = (labels > 0.5) | (rank <= 64)
    q_idx, v_idx, labels, qtype, first_z = q_idx[keep], v_idx[keep], labels[keep], qtype[keep], first_z[keep]
    dev = torch.device(device if device != "auto" else ("cuda:0" if torch.cuda.is_available() else "cpu"))
    model = PREMCollaborativeRetriever().to(dev)
    if torch.cuda.device_count() > 1 and dev.type == "cuda":
        model = torch.nn.DataParallel(model)
    query_tensor = torch.from_numpy(arr["query"].astype(np.float32)).to(dev)
    visual_tensor = torch.from_numpy(arr["visual_mean"].astype(np.float32)).to(dev)
    sub_tensor = torch.from_numpy(arr["sub_mean"].astype(np.float32)).to(dev)
    ds = TensorDataset(
        torch.from_numpy(q_idx),
        torch.from_numpy(v_idx),
        torch.from_numpy(qtype),
        torch.from_numpy(labels),
        torch.from_numpy(first_z.astype(np.float32)),
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=dev.type == "cuda", drop_last=False)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    history = []
    t0 = time.time()
    last_diag: Dict[str, Any] = {}
    for epoch in range(epochs):
        model.train()
        sums: Dict[str, float] = {}
        n = 0
        for bq, bv, bqt, by, bf in loader:
            bq = bq.to(dev, non_blocking=True)
            bv = bv.to(dev, non_blocking=True)
            bqt = bqt.to(dev, non_blocking=True)
            by = by.to(dev, non_blocking=True)
            bf = bf.to(dev, non_blocking=True)
            q = query_tensor[bq]
            v = visual_tensor[bv]
            s = sub_tensor[bv]
            out = model(q, bqt, v, s, bf)
            losses = prem_total_loss(out, by, bf)
            opt.zero_grad(set_to_none=True)
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
            bs = int(by.shape[0])
            n += bs
            for k, val in losses.items():
                sums[k] = sums.get(k, 0.0) + float(val.detach().cpu()) * bs
            last_diag = diagnostics_from_batch(out)
        history.append({k: v / max(1, n) for k, v in sums.items()} | {"epoch": epoch + 1, "rows": n})
    raw_model = model.module if isinstance(model, torch.nn.DataParallel) else model
    torch.save({
        "state_dict": raw_model.state_dict(),
        "mode": mode,
        "seed": seed,
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "architecture": "PREMCollaborativeRetriever(query modality pooling + visual/subtitle/joint partial relevance + residual clip)",
    }, ckpt_path)
    rec = {
        "checkpoint_path": str(ckpt_path),
        "checkpoint_local_only": True,
        "train_rows_before_front_hard_filter": int(train_mask.sum()),
        "train_rows_used": int(len(ds)),
        "positive_rows_used": int(labels.sum()),
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "lr": float(lr),
        "device": str(dev),
        "multi_gpu_data_parallel": bool(torch.cuda.device_count() > 1 and dev.type == "cuda"),
        "history": history,
        "gate_attention_diagnostic": last_diag,
        "runtime_sec": time.time() - t0,
    }
    write_json(result_path, rec)
    return ckpt_path, rec


@torch.no_grad()
def score_arrays(arrays_path: Path, ckpt_path: Path, split_name: str, batch_size: int, device: str) -> Dict[str, np.ndarray]:
    arr = load_npz_arrays(arrays_path)
    split_map = {"train_fit": 0, "calib_select": 1, "calib_holdout": 2}
    mask = arr["split_ids"].astype(np.int64) == split_map[split_name]
    dev = torch.device(device if device != "auto" else ("cuda:0" if torch.cuda.is_available() else "cpu"))
    model = PREMCollaborativeRetriever().to(dev)
    ckpt = torch.load(ckpt_path, map_location=dev)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    q_all = torch.from_numpy(arr["query"].astype(np.float32)).to(dev)
    v_all = torch.from_numpy(arr["visual_mean"].astype(np.float32)).to(dev)
    s_all = torch.from_numpy(arr["sub_mean"].astype(np.float32)).to(dev)
    q_idx = arr["q_idx"][mask].astype(np.int64)
    v_idx = arr["v_idx"][mask].astype(np.int64)
    qtype = arr["qtype"][mask].astype(np.int64)
    first = arr["first_stage"][mask].astype(np.float32)
    qids = arr["query_ids"][mask].astype(np.int64)
    first_z = within_query_z(qids, first)
    outs: Dict[str, list[np.ndarray]] = {k: [] for k in ["prem_score", "visual_relevance", "subtitle_relevance", "joint_relevance", "residual", "video_score", "gate_visual", "gate_subtitle", "gate_joint"]}
    for start in range(0, len(q_idx), batch_size):
        sl = slice(start, min(start + batch_size, len(q_idx)))
        bq = torch.from_numpy(q_idx[sl]).to(dev)
        bv = torch.from_numpy(v_idx[sl]).to(dev)
        bqt = torch.from_numpy(qtype[sl]).to(dev)
        bf = torch.from_numpy(first_z[sl]).to(dev)
        out = model(q_all[bq], bqt, v_all[bv], s_all[bv], bf)
        for k in ["prem_score", "visual_relevance", "subtitle_relevance", "joint_relevance", "residual", "video_score"]:
            outs[k].append(out[k].detach().cpu().numpy().astype(np.float32))
        gate = out["gate"].detach().cpu().numpy().astype(np.float32)
        outs["gate_visual"].append(gate[:, 0])
        outs["gate_subtitle"].append(gate[:, 1])
        outs["gate_joint"].append(gate[:, 2])
    return {k: np.concatenate(v) if v else np.empty(0, dtype=np.float32) for k, v in outs.items()}
