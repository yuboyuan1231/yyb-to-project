from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from blueprint_e2e_v2.data.collate import c28c_collate
from blueprint_e2e_v2.data.feature_registry import TMP_ROOT
from blueprint_e2e_v2.data.proposal_dataset import MultiSpanProposalDataset
from blueprint_e2e_v2.data.temporal_grid import iou_1d
from blueprint_e2e_v2.engine.checkpoint import load_checkpoint
from blueprint_e2e_v2.engine.refresh_hard_negatives import refresh_candidates
from blueprint_e2e_v2.engine.score_audit import ScoreScaleAccumulator
from blueprint_e2e_v2.engine.train import build_model, device_from_arg, prepare_banks


def duration_bucket(seconds: float) -> str:
    if seconds <= 5.0:
        return "short"
    if seconds <= 15.0:
        return "medium"
    return "long"


def _metric(xs: list[bool]) -> float:
    return 100.0 * sum(bool(x) for x in xs) / max(1, len(xs))


def _aggregate_records(records: list[dict[str, Any]], duplicate_count: int, invalid_count: int) -> dict[str, Any]:
    def aggregate(rs: list[dict[str, Any]]) -> dict[str, Any]:
        out: dict[str, Any] = {"query_count": len(rs)}
        for k in [1, 5, 10, 100]:
            out[f"VCMR_R@{k}_IoU0.5"] = _metric([r[f"VCMR_R@{k}_IoU0.5"] for r in rs])
            out[f"VCMR_R@{k}_IoU0.7"] = _metric([r[f"VCMR_R@{k}_IoU0.7"] for r in rs])
            out[f"VR_R@{k}"] = _metric([r[f"VR_R@{k}"] for r in rs])
        out["wrong_video_top1_rate"] = _metric([r["wrong_video_top1"] for r in rs])
        out["high_score_false_positive_rate"] = out["wrong_video_top1_rate"]
        out["correct_video_wrong_span_rate"] = _metric([r["correct_video_wrong_span_top1"] for r in rs])
        out["top1_mean_iou"] = float(np.mean([float(r["top1_iou"]) for r in rs])) if rs else 0.0
        out["duplicate_span_count"] = int(duplicate_count)
        out["invalid_span_count"] = int(invalid_count)
        return out

    by_qtype: dict[str, Any] = {}
    by_duration: dict[str, Any] = {}
    grouped_qtype: dict[str, list[dict[str, Any]]] = defaultdict(list)
    grouped_duration: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for rec in records:
        grouped_qtype[str(rec.get("query_type", "unknown"))].append(rec)
        grouped_duration[str(rec.get("duration_bucket", "unknown"))].append(rec)
    for key, vals in grouped_qtype.items():
        by_qtype[key] = aggregate(vals)
    for key, vals in grouped_duration.items():
        by_duration[key] = aggregate(vals)
    return {"summary": aggregate(records), "by_query_type": by_qtype, "by_duration": by_duration, "records": records[:2000]}


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    dataset: MultiSpanProposalDataset,
    device: torch.device,
    batch_size: int,
    seed: int,
    score_col: str = "vcmr_score",
    cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=c28c_collate, num_workers=0)
    records: list[dict[str, Any]] = []
    sample_rows: list[dict[str, Any]] = []
    score_table_rows = 0
    duplicate_count = 0
    invalid_count = 0
    score_acc = ScoreScaleAccumulator(cfg or {})
    for step, batch in enumerate(loader):
        tensor_batch = {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}
        out = model(tensor_batch)
        score_acc.update(out)
        scores = out["score"]["vcmr_score"].detach().float().cpu().numpy()
        video_scores = out["score"]["video_final"].detach().float().cpu().numpy()
        retr = out["retr"]
        pooled_scores = retr.get("retriever_score_pooled", retr["retriever_score"]).detach().float().cpu().numpy()
        late_scores = retr.get("retriever_score_late", retr["retriever_score"]).detach().float().cpu().numpy()
        token_scores = retr.get("retriever_score_token", torch.zeros_like(retr["retriever_score"])).detach().float().cpu().numpy()
        combined_scores = retr["retriever_score"].detach().float().cpu().numpy()
        spans = batch["spans_sec"].numpy()
        span_mask = batch["span_mask"].numpy()
        for bi, qid in enumerate(batch["query_ids"]):
            mask = span_mask[bi].astype(bool)
            ci_idx, mi_idx = np.nonzero(mask)
            if len(ci_idx) == 0:
                continue
            q_scores = scores[bi, ci_idx, mi_idx]
            q_spans = spans[bi, ci_idx, mi_idx]
            score_table_rows += int(len(q_scores))
            invalid_count += int(np.sum(q_spans[:, 1] <= q_spans[:, 0]))
            keys = [(str(batch["video_ids"][bi][int(c)]), float(s), float(e)) for c, (s, e) in zip(ci_idx, q_spans)]
            duplicate_count += len(keys) - len(set(keys))
            order = np.argsort(-q_scores)
            kept: list[tuple[int, int, float]] = []
            by_video: dict[str, list[tuple[float, float]]] = defaultdict(list)
            for oi in order:
                ci = int(ci_idx[oi])
                mi = int(mi_idx[oi])
                vid = str(batch["video_ids"][bi][ci])
                span = (float(spans[bi, ci, mi, 0]), float(spans[bi, ci, mi, 1]))
                if any(iou_1d(span, prev) > 0.7 for prev in by_video[vid]):
                    continue
                by_video[vid].append(span)
                kept.append((ci, mi, float(q_scores[oi])))
                if len(kept) >= 200:
                    break
            gt_video = str(batch["gt_video_id"][bi])
            gt_ts = (float(batch["gt_start"][bi].item()), float(batch["gt_end"][bi].item()))
            videos = [str(batch["video_ids"][bi][ci]) for ci, _mi, _score in kept]
            ious = [iou_1d((float(spans[bi, ci, mi, 0]), float(spans[bi, ci, mi, 1])), gt_ts) if str(batch["video_ids"][bi][ci]) == gt_video else 0.0 for ci, mi, _score in kept]
            unique_videos: list[str] = []
            for vid in videos:
                if vid not in unique_videos:
                    unique_videos.append(vid)
            top1_video_ok = bool(videos and videos[0] == gt_video)
            top1_iou = float(ious[0]) if ious else 0.0
            rec = {
                "split": dataset.split,
                "seed": int(seed),
                "query_id": int(qid),
                "query_type": str(["v", "t", "vt", "unknown"][int(batch["query_type"][bi].item())]),
                "duration_bucket": duration_bucket(float(batch["duration"][bi].item())),
                "top1_video_correct": top1_video_ok,
                "wrong_video_top1": not top1_video_ok,
                "correct_video_wrong_span_top1": bool(top1_video_ok and top1_iou < 0.5),
                "top1_iou": top1_iou,
            }
            for k in [1, 5, 10, 100]:
                rec[f"VCMR_R@{k}_IoU0.5"] = any(i >= 0.5 for i in ious[: min(k, len(ious))])
                rec[f"VCMR_R@{k}_IoU0.7"] = any(i >= 0.7 for i in ious[: min(k, len(ious))])
                rec[f"VR_R@{k}"] = gt_video in unique_videos[: min(k, len(unique_videos))]
            records.append(rec)
            if len(sample_rows) < 5000:
                for rank, (ci, mi, sc) in enumerate(kept[: max(0, 5000 - len(sample_rows))]):
                    sample_rows.append({
                        "split": dataset.split,
                        "seed": int(seed),
                        "query_id": int(qid),
                        "rank_after_nms": int(rank + 1),
                        "video_id": str(batch["video_ids"][bi][ci]),
                        "span_start": float(spans[bi, ci, mi, 0]),
                        "span_end": float(spans[bi, ci, mi, 1]),
                        "vcmr_score": float(sc),
                        "video_score": float(video_scores[bi, ci]),
                        "retriever_score_pooled": float(pooled_scores[bi, ci]),
                        "retriever_score_late": float(late_scores[bi, ci]),
                        "retriever_score_token": float(token_scores[bi, ci]),
                        "retriever_score_combined": float(combined_scores[bi, ci]),
                        "gt_video_id": gt_video,
                        "gt_start": gt_ts[0],
                        "gt_end": gt_ts[1],
                        "query_type": rec["query_type"],
                        "duration_bucket": rec["duration_bucket"],
                    })
        if (step + 1) % 100 == 0 or (step + 1) == len(loader):
            print(f"C28C eval {dataset.split}: scored {min((step + 1) * batch_size, len(dataset))}/{len(dataset)} queries", flush=True)
    return {"score_table_rows": score_table_rows, "summary": _aggregate_records(records, duplicate_count, invalid_count), "score_table_sample": sample_rows, "score_scale_audit": score_acc.summary()}


def run_evaluation(cfg: dict[str, Any], split: str = "calib_holdout", device_arg: str | None = None, force: bool = False) -> dict[str, Any]:
    device = device_from_arg(device_arg)
    banks = prepare_banks(cfg, force=False)
    video_ids = [str(x) for x in banks["visual_np"]["video_ids"].tolist()]
    banks["video_bank"].video_ids = video_ids
    banks["video_bank"].video_to_idx = {v: i for i, v in enumerate(video_ids)}
    banks["video_bank"].idx_to_video = {i: v for v, i in banks["video_bank"].video_to_idx.items()}
    model = build_model(cfg, device)
    configured_ckpt = cfg.get("checkpoint_path") or cfg.get("c28e_retriever_checkpoint_path")
    ckpt_path = Path(str(configured_ckpt)) if configured_ckpt else TMP_ROOT / "checkpoints" / f"C28C_FULL_{cfg.get('mode','medium')}_seed{cfg.get('seed',2026)}.pt"
    ckpt_loaded = False
    if ckpt_path.exists():
        load_checkpoint(ckpt_path, model)
        ckpt_loaded = True
    visual_bank = torch.from_numpy(banks["visual_np"]["visual_mean"].astype(np.float32))
    subtitle_bank = torch.from_numpy(banks["subtitle_np"]["subtitle_mean"].astype(np.float32))
    max_queries = int(cfg.get("max_queries", 0) or 0) or None
    target_len = int(cfg.get("target_len", 64))
    visual_seq_bank, visual_seq_manifest = banks["video_bank"].build_or_load_sequence_bank(
        video_ids,
        target_len=target_len,
        max_videos=int(cfg.get("max_videos", 0) or 0) or None,
        force=False,
    )
    subtitle_seq_bank, subtitle_seq_manifest = banks["subtitle_bank"].build_or_load_sequence_bank(
        video_ids,
        target_len=target_len,
        max_videos=int(cfg.get("max_videos", 0) or 0) or None,
        force=False,
    )
    visual_seq_mask, visual_mask_manifest = banks["video_bank"].build_or_load_sequence_mask(
        video_ids,
        target_len=target_len,
        max_videos=int(cfg.get("max_videos", 0) or 0) or None,
        force=False,
    )
    subtitle_seq_mask, subtitle_mask_manifest = banks["subtitle_bank"].build_or_load_sequence_mask(
        video_ids,
        target_len=target_len,
        max_videos=int(cfg.get("max_videos", 0) or 0) or None,
        force=False,
    )
    banks["video_bank"].clear_sequence_cache()
    banks["subtitle_bank"].clear_sequence_cache()
    eval_topk = int(cfg.get("eval_candidate_k", cfg.get("candidate_topk_eval", cfg.get("dynamic_topk", 200))))
    candidates, cand_audit = refresh_candidates(
        model,
        banks["split_manager"],
        banks["query_bank"],
        video_ids,
        banks["video_bank"].video_to_idx,
        visual_bank,
        subtitle_bank,
        split,
        max_queries=max_queries,
        dynamic_topk=eval_topk,
        chunk_size=int(cfg.get("chunk_size", 256)),
        device=device,
        insert_gt_for_training=False,
        visual_seq_bank=visual_seq_bank,
        subtitle_seq_bank=subtitle_seq_bank,
        visual_seq_mask=visual_seq_mask,
        subtitle_seq_mask=subtitle_seq_mask,
        late_candidate_mining=bool(cfg.get("late_candidate_mining", cfg.get("late_interaction_enabled", False))),
        broad_topk=int(cfg.get("broad_topk_eval", cfg.get("dynamic_topk", eval_topk))),
        candidate_encode_chunk=int(cfg.get("candidate_encode_chunk", 32)),
        clip_bank_encode_chunk=int(cfg.get("clip_bank_encode_chunk", 128)),
        late_soft_topk=int(cfg.get("late_soft_topk", 8)),
        late_temperature=float(cfg.get("late_temperature", 0.07)),
        token_maxsim_weight=float(cfg.get("token_maxsim_weight", 0.0)),
        pooled_score_weight=float(cfg.get("pooled_score_weight", 1.0)),
        late_score_weight=float(cfg.get("late_score_weight", 0.0)),
    )
    dataset = MultiSpanProposalDataset(
        split,
        banks["split_manager"],
        banks["query_bank"],
        banks["video_bank"],
        banks["subtitle_bank"],
        candidates,
        max_queries=max_queries,
        max_candidates=eval_topk,
        max_spans_per_video=int(cfg.get("max_spans_per_video", 64)),
        insert_gt_for_training=False,
        visual_seq_bank=visual_seq_bank,
        subtitle_seq_bank=subtitle_seq_bank,
        visual_seq_mask=visual_seq_mask,
        subtitle_seq_mask=subtitle_seq_mask,
        target_len=target_len,
    )
    res = evaluate_model(model, dataset, device, batch_size=int(cfg.get("batch_size", 8)), seed=int(cfg.get("seed", 2026)), cfg=cfg)
    res["checkpoint_loaded"] = ckpt_loaded
    res["checkpoint_path"] = str(ckpt_path)
    res["candidate_audit"] = cand_audit
    res["dataset_audit"] = dataset.audit()
    res["visual_sequence_manifest"] = visual_seq_manifest
    res["subtitle_sequence_manifest"] = subtitle_seq_manifest
    res["visual_sequence_mask_manifest"] = visual_mask_manifest
    res["subtitle_sequence_mask_manifest"] = subtitle_mask_manifest
    return res
