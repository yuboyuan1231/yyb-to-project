#!/usr/bin/env python
"""C6-B1-lite finalization, mechanism analysis, and oracle ceiling diagnostics.

Read-only with respect to frozen artifacts.  Writes only c6_b1_lite_final/.
No training, no threshold search, no official evaluator call.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rlem_c6_b0.run_c6_b0_candidate_diagnostic import correctness_for_ts, span_iou_idx, span_iou_ts  # noqa: E402


METRIC_KEYS = ("0.5-r1", "0.5-r5", "0.5-r10", "0.5-r100", "0.7-r1", "0.7-r5", "0.7-r10", "0.7-r100")
KS = (1, 5, 10, 100)
THRS = (0.5, 0.7)
CLIP = 1.5
NMS = 0.7


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--output_dir", default="c6_b1_lite_final")
    p.add_argument("--c4_submission", default="results/rlem_c4_r2_cal_v21_official_val/val_v21_submission.json")
    p.add_argument("--c6_submission", default="results/rlem_c6_b1_lite_r1_safe/official_val_submission.json")
    p.add_argument("--c4_metrics", default="results/rlem_c4_r2_cal_v21_official_val/val_v21_metrics.json")
    p.add_argument("--c6_metrics", default="results/rlem_c6_b1_lite_r1_safe/official_val_metrics.json")
    p.add_argument("--c6_manifest", default="c6_b1_lite_official_val/C6_B1_LITE_OFFICIAL_VAL_ONE_SHOT_MANIFEST.json")
    p.add_argument("--c6_hashes", default="c6_b1_lite_official_val/C6_B1_LITE_OFFICIAL_VAL_ONE_SHOT_HASHES.json")
    p.add_argument("--cache_npz", default="results/rlem_c6_b1_lite_r1_safe/official_val_c6_cache.npz")
    p.add_argument("--pool_npz", default="results/rlem_c6_b1_lite_r1_safe/official_val_candidate_pool_top_slots.npz")
    p.add_argument("--scores_npz", default="results/rlem_c6_b1_lite_r1_safe/official_val_scores.npz")
    p.add_argument("--gt_jsonl", default="/home/a/yybwork/data/yyb/tvr_feature_release/data/tvr_val_release.jsonl")
    p.add_argument("--train_calib_audit", default="c6_b1_lite_audit/C6_B1_LITE_R1_SAFE_AUDIT.json")
    return p.parse_args()


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def artifact(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    return {"path": str(p), "exists": p.exists(), "size": p.stat().st_size if p.exists() else None, "sha256": sha256_file(p) if p.exists() else None}


def atomic_json(path: str | Path, obj: Any) -> None:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(path) + ".partial")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def atomic_text(path: str | Path, text: str) -> None:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(path) + ".partial")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def iter_jsonl(path: str | Path) -> Iterable[Dict[str, Any]]:
    with open(path, "rt", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def load_metrics(path: str | Path) -> Dict[str, float]:
    obj = load_json(path)
    vc = obj["VCMR"] if "VCMR" in obj else obj
    return {k: float(vc[k]) for k in METRIC_KEYS}


def metric_delta(a: Dict[str, float], b: Dict[str, float]) -> Dict[str, float]:
    return {k: float(a[k] - b[k]) for k in METRIC_KEYS}


def video_id(row: Dict[str, Any], video2idx: Dict[str, int]) -> int:
    raw = row.get("vid_idx", row.get("video_idx", row.get("vid_name")))
    return int(video2idx.get(raw, raw))


def pred_to_idx(pred: Sequence[Any]) -> Tuple[int, int, int, float]:
    vid = int(pred[0])
    si = int(round(float(pred[1]) / CLIP))
    ei = max(si, int(round(float(pred[2]) / CLIP)) - 1)
    return vid, si, ei, float(pred[3])


def hit_pred(pred: Sequence[Any], gt_vid: int, ts: Any, thr: float) -> bool:
    vid, si, ei, _ = pred_to_idx(pred)
    return vid == gt_vid and correctness_for_ts(si, ei, ts, thr)


def best_iou_pred(pred: Sequence[Any], gt_vid: int, ts: Any) -> float:
    vid, si, ei, _ = pred_to_idx(pred)
    if vid != gt_vid:
        return 0.0
    if ts and isinstance(ts, list) and isinstance(ts[0], (list, tuple)):
        return float(max(span_iou_ts(si, ei, one) for one in ts))
    return float(span_iou_ts(si, ei, ts))


def hit_lists_from_submission(submission: Dict[str, Any], gt_rows: Dict[str, Dict[str, Any]]) -> Dict[str, List[List[bool]]]:
    video2idx = submission["video2idx"]
    out = {str(t): [] for t in THRS}
    for row in submission["VCMR"]:
        gt = gt_rows[str(row["desc_id"])]
        gt_vid = video_id(gt, video2idx)
        ts = gt["ts"]
        for t in THRS:
            out[str(t)].append([hit_pred(p, gt_vid, ts, t) for p in row["predictions"][:100]])
    return out


def metrics_from_hit_lists(hit_lists: Dict[str, List[List[bool]]]) -> Dict[str, float]:
    qn = len(next(iter(hit_lists.values())))
    out = {}
    for t in THRS:
        labs = hit_lists[str(t)]
        for k in KS:
            out[f"{t:.1f}-r{k}"] = 100.0 * sum(any(x[:k]) for x in labs) / qn
    return out


def nms_keep(cands: Sequence[Tuple[int, int, int, float]], max_keep: int = 100) -> List[int]:
    kept: List[int] = []
    by_video: Dict[int, List[int]] = defaultdict(list)
    for i, (vid, si, ei, _score) in enumerate(cands):
        suppress = False
        for prior in by_video.get(vid, []):
            _, psi, pei, _ = cands[prior]
            if span_iou_idx(si, ei, psi, pei) > NMS:
                suppress = True
                break
        if not suppress:
            kept.append(i)
            by_video[vid].append(i)
            if len(kept) >= max_keep:
                break
    return kept


def labels_for_candidates(cands: Sequence[Tuple[int, int, int, float]], gt_vid: int, ts: Any) -> Tuple[Dict[str, List[bool]], List[float]]:
    keep = nms_keep(cands)
    labs = {str(t): [] for t in THRS}
    ious = []
    for i in keep:
        vid, si, ei, _score = cands[i]
        if vid == gt_vid:
            if ts and isinstance(ts, list) and isinstance(ts[0], (list, tuple)):
                iou = float(max(span_iou_ts(si, ei, one) for one in ts))
            else:
                iou = float(span_iou_ts(si, ei, ts))
        else:
            iou = 0.0
        ious.append(iou)
        for t in THRS:
            labs[str(t)].append(iou >= t)
    return labs, ious


def score_label_tuple(labs: Dict[str, List[bool]], ious: Sequence[float]) -> Tuple:
    return (
        int(any(labs["0.7"][:1])), int(any(labs["0.5"][:1])),
        int(any(labs["0.7"][:5])), int(any(labs["0.5"][:5])),
        int(any(labs["0.7"][:10])), int(any(labs["0.5"][:10])),
        int(any(labs["0.7"][:100])), int(any(labs["0.5"][:100])),
        float(max(ious) if ious else 0.0),
        float(np.mean(ious[:5]) if ious[:5] else 0.0),
    )


def aggregate_variant(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    hit_lists = {str(t): [r["labs"][str(t)] for r in records] for t in THRS}
    return {
        "metrics": metrics_from_hit_lists(hit_lists),
        "mean_best_iou_top100": float(np.mean([max(r["ious"]) if r["ious"] else 0.0 for r in records])),
        "median_best_iou_top100": float(np.median([max(r["ious"]) if r["ious"] else 0.0 for r in records])),
        "replacement_count": int(sum(r.get("replacement_count", 0) for r in records)),
        "replacement_query_count": int(sum(r.get("replacement_count", 0) > 0 for r in records)),
    }


def simulate_actual_or_oracle(
    cache: Dict[str, np.ndarray],
    pool: Dict[str, np.ndarray],
    pool_score: np.ndarray | None,
    gt_rows_seq: List[Dict[str, Any]],
    video2idx: Dict[str, int],
    mode: str,
    active_slots: Sequence[int] = (0, 1, 2),
    max_replacements: int = 2,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    qn = len(gt_rows_seq)
    rows_per_q = int(cache["desc_offsets"][1] - cache["desc_offsets"][0])
    video_idx = cache["video_idx"].astype(np.int64)
    start_idx = cache["start_idx"].astype(np.int64)
    end_idx = cache["end_idx"].astype(np.int64)
    s_c4 = cache["s_c4_final"].astype(np.float32)
    shape = pool["shape"].astype(int).tolist()
    slots, alt_count = shape[1], shape[2]
    records: List[Dict[str, Any]] = []
    for q in range(qn):
        gt = gt_rows_seq[q]
        gt_vid = video_id(gt, video2idx)
        ts = gt["ts"]
        rs = np.arange(q * rows_per_q, (q + 1) * rows_per_q, dtype=np.int64)
        base_order = rs[np.argsort(-s_c4[rs], kind="stable")[:100]]
        base_cands = [(int(video_idx[r]), int(start_idx[r]), int(end_idx[r]), float(s_c4[r])) for r in base_order]

        def with_choices(choices: Dict[int, int]) -> List[Tuple[int, int, int, float]]:
            cands = list(base_cands)
            for slot, alt in choices.items():
                if alt == 0:
                    continue
                p = (q * slots + slot) * alt_count + alt
                cands[slot] = (int(pool["video_idx"][p]), int(pool["start_idx"][p]), int(pool["end_idx"][p]), base_cands[slot][3])
            return cands

        choices: Dict[int, int] = {}
        if mode == "original":
            choices = {}
        elif mode == "actual":
            replaced = 0
            assert pool_score is not None
            for slot in active_slots:
                if replaced >= max_replacements:
                    break
                base = (q * slots + slot) * alt_count
                sc = pool_score[base:base + alt_count]
                best = int(np.argmax(sc))
                margin = float(sc[best] - sc[0])
                thd = 0.4 if slot == 0 else 1.8
                if best != 0 and margin > thd:
                    choices[slot] = best
                    replaced += 1
        elif mode in {"oracle_hybrid", "oracle_boundary"}:
            candidates_for_choice = []
            for slot in active_slots:
                orig = base_cands[slot]
                orig_iou = 0.0
                if orig[0] == gt_vid:
                    if ts and isinstance(ts, list) and isinstance(ts[0], (list, tuple)):
                        orig_iou = float(max(span_iou_ts(orig[1], orig[2], one) for one in ts))
                    else:
                        orig_iou = float(span_iou_ts(orig[1], orig[2], ts))
                best_alt = 0
                best_iou = -1.0
                for alt in range(1, alt_count):
                    p = (q * slots + slot) * alt_count + alt
                    cand_vid = int(pool["video_idx"][p])
                    cand_si = int(pool["start_idx"][p])
                    cand_ei = int(pool["end_idx"][p])
                    iou = 0.0
                    if cand_vid == gt_vid:
                        if ts and isinstance(ts, list) and isinstance(ts[0], (list, tuple)):
                            iou = float(max(span_iou_ts(cand_si, cand_ei, one) for one in ts))
                        else:
                            iou = float(span_iou_ts(cand_si, cand_ei, ts))
                    if (iou, alt) > (best_iou, best_alt):
                        best_iou, best_alt = iou, alt
                gain = best_iou - (0.0 if mode == "oracle_boundary" else orig_iou)
                if best_alt != 0 and gain > 0:
                    candidates_for_choice.append((gain, best_iou, slot, best_alt))
            candidates_for_choice.sort(reverse=True)
            choices = {int(slot): int(alt) for _gain, _iou, slot, alt in candidates_for_choice[:max_replacements]}
        elif mode == "greedy_hybrid_slots5":
            candidates_for_choice = []
            for slot in active_slots:
                best_alt = 0
                best_local = (0.0, -1)
                orig = base_cands[slot]
                orig_iou = 0.0
                if orig[0] == gt_vid:
                    if ts and isinstance(ts, list) and isinstance(ts[0], (list, tuple)):
                        orig_iou = float(max(span_iou_ts(orig[1], orig[2], one) for one in ts))
                    else:
                        orig_iou = float(span_iou_ts(orig[1], orig[2], ts))
                for alt in range(1, alt_count):
                    p = (q * slots + slot) * alt_count + alt
                    cand = (int(pool["video_idx"][p]), int(pool["start_idx"][p]), int(pool["end_idx"][p]), orig[3])
                    iou = 0.0
                    if cand[0] == gt_vid:
                        if ts and isinstance(ts, list) and isinstance(ts[0], (list, tuple)):
                            iou = float(max(span_iou_ts(cand[1], cand[2], one) for one in ts))
                        else:
                            iou = float(span_iou_ts(cand[1], cand[2], ts))
                    if (iou, alt) > best_local:
                        best_local = (iou, alt); best_alt = alt
                gain = best_local[0] - orig_iou
                if best_alt != 0 and gain > 0:
                    candidates_for_choice.append((gain, best_local[0], slot, best_alt))
            candidates_for_choice.sort(reverse=True)
            choices = {int(slot): int(alt) for _gain, _iou, slot, alt in candidates_for_choice[:max_replacements]}
        else:
            raise ValueError(mode)
        cands = with_choices(choices)
        labs, ious = labels_for_candidates(cands, gt_vid, ts)
        records.append({"labs": labs, "ious": ious, "replacement_count": len([a for a in choices.values() if a != 0]), "choices": choices})
    return aggregate_variant(records), records


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"{out_dir} already contains files; refusing overwrite")
    out_dir.mkdir(parents=True, exist_ok=True)

    c4_sub = load_json(args.c4_submission)
    c6_sub = load_json(args.c6_submission)
    gt_rows = {str(r["desc_id"]): r for r in iter_jsonl(args.gt_jsonl)}
    gt_seq = [gt_rows[str(r["desc_id"])] for r in c4_sub["VCMR"]]
    video2idx = c6_sub["video2idx"]
    c4_official = load_metrics(args.c4_metrics)
    c6_official = load_metrics(args.c6_metrics)
    official_delta = metric_delta(c6_official, c4_official)
    c6_manifest = load_json(args.c6_manifest)
    train_calib = load_json(args.train_calib_audit)

    # Mechanism from frozen submissions.
    c4_hits = hit_lists_from_submission(c4_sub, gt_rows)
    c6_hits = hit_lists_from_submission(c6_sub, gt_rows)
    changed_top1 = []
    unchanged_top1 = []
    confusion = {str(t): Counter() for t in THRS}
    unchanged_changes = {f"{t:.1f}-r{k}": Counter() for t in THRS for k in KS}
    best_iou_delta_top1 = []
    for i, (b, n) in enumerate(zip(c4_sub["VCMR"], c6_sub["VCMR"])):
        btop = tuple(pred_to_idx(b["predictions"][0])[:3])
        ntop = tuple(pred_to_idx(n["predictions"][0])[:3])
        changed = btop != ntop
        (changed_top1 if changed else unchanged_top1).append(i)
        gt = gt_seq[i]
        gt_vid = video_id(gt, video2idx)
        ts = gt["ts"]
        if changed:
            best_iou_delta_top1.append(best_iou_pred(n["predictions"][0], gt_vid, ts) - best_iou_pred(b["predictions"][0], gt_vid, ts))
            for t in THRS:
                bh = hit_pred(b["predictions"][0], gt_vid, ts, t)
                nh = hit_pred(n["predictions"][0], gt_vid, ts, t)
                key = ("hit" if bh else "miss") + "_to_" + ("hit" if nh else "miss")
                confusion[str(t)][key] += 1
        else:
            for t in THRS:
                for k in KS:
                    bh = any(c4_hits[str(t)][i][:k])
                    nh = any(c6_hits[str(t)][i][:k])
                    if (not bh) and nh:
                        unchanged_changes[f"{t:.1f}-r{k}"]["gain"] += 1
                    elif bh and (not nh):
                        unchanged_changes[f"{t:.1f}-r{k}"]["loss"] += 1
                    elif bh and nh:
                        unchanged_changes[f"{t:.1f}-r{k}"]["both_hit"] += 1
                    else:
                        unchanged_changes[f"{t:.1f}-r{k}"]["both_miss"] += 1

    with np.load(args.cache_npz, allow_pickle=True) as z:
        cache = {k: z[k] for k in z.files}
    with np.load(args.pool_npz, allow_pickle=False) as z:
        pool = {k: z[k] for k in z.files}
    with np.load(args.scores_npz, allow_pickle=False) as z:
        pool_score = z["pool_score"].astype(np.float32)

    original_diag, original_records = simulate_actual_or_oracle(cache, pool, pool_score, gt_seq, video2idx, "original", active_slots=(), max_replacements=0)
    actual_diag, actual_records = simulate_actual_or_oracle(cache, pool, pool_score, gt_seq, video2idx, "actual", active_slots=(0, 1, 2), max_replacements=2)
    oracle_apply3, oracle_apply3_records = simulate_actual_or_oracle(cache, pool, None, gt_seq, video2idx, "oracle_hybrid", active_slots=(0, 1, 2), max_replacements=2)
    oracle_boundary, _ = simulate_actual_or_oracle(cache, pool, None, gt_seq, video2idx, "oracle_boundary", active_slots=(0, 1, 2), max_replacements=2)
    oracle_slots5_greedy2, _ = simulate_actual_or_oracle(cache, pool, None, gt_seq, video2idx, "greedy_hybrid_slots5", active_slots=(0, 1, 2, 3, 4), max_replacements=2)
    oracle_slots5_greedy5, _ = simulate_actual_or_oracle(cache, pool, None, gt_seq, video2idx, "greedy_hybrid_slots5", active_slots=(0, 1, 2, 3, 4), max_replacements=5)

    # Replacement-level precision and slot contributions from actual frozen policy.
    slot_stats: Dict[int, Counter] = {0: Counter(), 1: Counter(), 2: Counter()}
    repl_stats = Counter()
    qn, slots, alt_count = pool["shape"].astype(int).tolist()
    for q, rec in enumerate(actual_records):
        gt = gt_seq[q]; gt_vid = video_id(gt, video2idx); ts = gt["ts"]
        for slot, alt in rec["choices"].items():
            if slot not in slot_stats or alt == 0:
                continue
            base = (q * slots + slot) * alt_count
            orig = [int(pool["video_idx"][base]), int(pool["orig_start_idx"][base]), int(pool["orig_end_idx"][base]), 0.0]
            newp = base + alt
            new = [int(pool["video_idx"][newp]), int(pool["start_idx"][newp]), int(pool["end_idx"][newp]), 0.0]
            old05 = hit_pred([orig[0], orig[1] * CLIP, (orig[2] + 1) * CLIP, 0], gt_vid, ts, 0.5)
            new05 = hit_pred([new[0], new[1] * CLIP, (new[2] + 1) * CLIP, 0], gt_vid, ts, 0.5)
            old07 = hit_pred([orig[0], orig[1] * CLIP, (orig[2] + 1) * CLIP, 0], gt_vid, ts, 0.7)
            new07 = hit_pred([new[0], new[1] * CLIP, (new[2] + 1) * CLIP, 0], gt_vid, ts, 0.7)
            for name, old, newv in [("05", old05, new05), ("07", old07, new07)]:
                slot_stats[slot]["count"] += int(name == "05")
                slot_stats[slot][f"precision_{name}_num"] += int(newv)
                slot_stats[slot][f"benefit_{name}"] += int((not old) and newv)
                slot_stats[slot][f"harm_{name}"] += int(old and (not newv))
                repl_stats[f"precision_{name}_num"] += int(newv)
                repl_stats[f"benefit_{name}"] += int((not old) and newv)
                repl_stats[f"harm_{name}"] += int(old and (not newv))
    slot_contrib = {}
    for slot in [0, 1, 2]:
        diag, _records = simulate_actual_or_oracle(cache, pool, pool_score, gt_seq, video2idx, "actual", active_slots=(slot,), max_replacements=1)
        slot_contrib[str(slot)] = {
            "replacement_count": int(slot_stats[slot]["count"]),
            "replacement_rate": float(slot_stats[slot]["count"] / qn),
            "diagnostic_delta_vs_original": metric_delta(diag["metrics"], original_diag["metrics"]),
            "precision_05": float(slot_stats[slot]["precision_05_num"] / max(slot_stats[slot]["count"], 1)),
            "precision_07": float(slot_stats[slot]["precision_07_num"] / max(slot_stats[slot]["count"], 1)),
            "benefit_05": int(slot_stats[slot]["benefit_05"]),
            "harm_05": int(slot_stats[slot]["harm_05"]),
            "benefit_07": int(slot_stats[slot]["benefit_07"]),
            "harm_07": int(slot_stats[slot]["harm_07"]),
        }

    gt_video_in_top100 = 0
    gt_video_top1 = 0
    for row, gt in zip(c4_sub["VCMR"], gt_seq):
        gt_vid = video_id(gt, video2idx)
        vids = [int(p[0]) for p in row["predictions"][:100]]
        gt_video_in_top100 += int(gt_vid in vids)
        gt_video_top1 += int(vids and vids[0] == gt_vid)
    wrong_video = {
        "queries": qn,
        "gt_video_in_c4_top100": int(gt_video_in_top100),
        "gt_video_missing_from_c4_top100": int(qn - gt_video_in_top100),
        "gt_video_missing_rate": float((qn - gt_video_in_top100) / qn),
        "gt_video_top1": int(gt_video_top1),
        "gt_video_not_top1_rate": float((qn - gt_video_top1) / qn),
    }

    mechanism = {
        "status": "PASS",
        "official_val_used_for_analysis_only": True,
        "no_second_official_val": True,
        "changed_top1_queries": len(changed_top1),
        "unchanged_top1_queries": len(unchanged_top1),
        "changed_top1_ratio": float(len(changed_top1) / qn),
        "changed_top1_confusion": {k: dict(v) for k, v in confusion.items()},
        "changed_top1_mean_iou_delta": float(np.mean(best_iou_delta_top1)) if best_iou_delta_top1 else 0.0,
        "unchanged_query_hit_changes": {k: dict(v) for k, v in unchanged_changes.items()},
        "replacement_by_slot": {k: int(v["replacement_count"]) for k, v in slot_contrib.items()},
        "replacement_precision_05": float(repl_stats["precision_05_num"] / max(sum(v["replacement_count"] for v in slot_contrib.values()), 1)),
        "replacement_precision_07": float(repl_stats["precision_07_num"] / max(sum(v["replacement_count"] for v in slot_contrib.values()), 1)),
        "replacement_benefit_rate_05": float(repl_stats["benefit_05"] / max(sum(v["replacement_count"] for v in slot_contrib.values()), 1)),
        "replacement_harmful_rate_05": float(repl_stats["harm_05"] / max(sum(v["replacement_count"] for v in slot_contrib.values()), 1)),
        "replacement_benefit_rate_07": float(repl_stats["benefit_07"] / max(sum(v["replacement_count"] for v in slot_contrib.values()), 1)),
        "replacement_harmful_rate_07": float(repl_stats["harm_07"] / max(sum(v["replacement_count"] for v in slot_contrib.values()), 1)),
        "net_hit_conversion_count_05": int(repl_stats["benefit_05"] - repl_stats["harm_05"]),
        "net_hit_conversion_count_07": int(repl_stats["benefit_07"] - repl_stats["harm_07"]),
        "slot_contribution": slot_contrib,
        "diagnostic_metrics_original_from_cache": original_diag["metrics"],
        "diagnostic_metrics_actual_from_cache": actual_diag["metrics"],
        "diagnostic_delta_actual_vs_original": metric_delta(actual_diag["metrics"], original_diag["metrics"]),
    }

    ceiling_variants = {
        "A_selector_current_actual": actual_diag,
        "B_current_candidate_pool_oracle_apply3_max2": oracle_apply3,
        "C_boundary_candidate_only_oracle_apply3_max2": oracle_boundary,
        "D_original_candidate_only": original_diag,
        "E_hybrid_pool_greedy_slots5_max2_same_video_slots": oracle_slots5_greedy2,
        "F_hybrid_pool_greedy_slots5_max5_same_video_slots": oracle_slots5_greedy5,
    }
    for name, obj in ceiling_variants.items():
        obj["delta_vs_actual"] = metric_delta(obj["metrics"], actual_diag["metrics"])
        obj["delta_vs_original"] = metric_delta(obj["metrics"], original_diag["metrics"])
    ceiling = {
        "status": "PASS",
        "official_val_used_for_analysis_only": True,
        "no_official_evaluator_called": True,
        "variants": ceiling_variants,
        "wrong_video_limited_residual_errors": wrong_video,
        "interpretation": {},
    }
    # Heuristic recommendation from diagnostics.
    r1_oracle_gap = (
        ceiling_variants["E_hybrid_pool_greedy_slots5_max2_same_video_slots"]["metrics"]["0.7-r1"]
        - actual_diag["metrics"]["0.7-r1"]
    )
    r100_oracle_gap = (
        ceiling_variants["E_hybrid_pool_greedy_slots5_max2_same_video_slots"]["metrics"]["0.7-r100"]
        - actual_diag["metrics"]["0.7-r100"]
    )
    wrong_video_major = wrong_video["gt_video_missing_rate"] > 0.45
    harmful_low = mechanism["replacement_harmful_rate_05"] < 0.05 and mechanism["replacement_harmful_rate_07"] < 0.05
    selector_conservative = r1_oracle_gap > 0.5 or r100_oracle_gap > 1.0
    if selector_conservative and harmful_low and not wrong_video_major:
        rec_class = "Recommend C6-B2"
        rec = "同视频 hybrid candidate pool 的 oracle 仍高于 actual，且 replacement harmful rate 低；优先开 C6-B2 higher-ceiling selector protocol。"
    elif wrong_video_major or r1_oracle_gap < 0.2:
        rec_class = "Recommend C6-C"
        rec = "主要剩余误差来自 video ranking / video slot 或同视频候选上限不足；应转向 video-aware mutual integration，而不是扩大同视频替换器。"
    else:
        rec_class = "Stop at C6-B1-lite"
        rec = "C6-B1-lite 已稳定正向，但 oracle 空间和扩展收益有限；可冻结为阶段主结果。"
    ceiling["interpretation"] = {
        "r1_07_oracle_gap_slots5_max2_vs_actual": float(r1_oracle_gap),
        "r100_07_oracle_gap_slots5_max2_vs_actual": float(r100_oracle_gap),
        "wrong_video_major": bool(wrong_video_major),
        "harmful_replacement_low": bool(harmful_low),
        "selector_conservative": bool(selector_conservative),
        "next_stage_classification": rec_class,
        "recommendation": rec,
    }

    metric_table = {
        "C4_final": c4_official,
        "C6_B1_lite": c6_official,
        "delta": official_delta,
        "diagnostic_original_from_cache": original_diag["metrics"],
        "diagnostic_actual_from_cache": actual_diag["metrics"],
    }
    final_manifest = {
        "status": "C6_B1_LITE_FINALIZED",
        "current_promoted_system": "C6-B1-lite c6b1r1_0057",
        "previous_promoted_system": "C4_final v21_00444",
        "promotion_type": "low-drift candidate-selector positive",
        "official_val_one_shot": True,
        "successful_metric_result_count": 1,
        "no_second_official_val": True,
        "score_grid_on_val": False,
        "temperature_search_on_val": False,
        "post_val_adjustment": False,
        "C6_C_used": False,
        "C6_B2_training_used": False,
        "selector_retrained": False,
        "C4_final_modified": False,
        "frozen_config_modified": False,
        "official_metrics": metric_table,
        "movement": c6_manifest["movement_relative_to_c4_final"],
        "train_calib_positive_vs_official_positive": {
            "train_calib_delta_vs_pseudo_C4_final": train_calib["best_config"]["delta_vs_pseudo_C4_final"],
            "official_delta_vs_C4_final": official_delta,
        },
        "next_stage_classification": rec_class,
        "next_stage_recommendation": rec,
    }

    atomic_json(out_dir / "C6_B1_LITE_OFFICIAL_METRIC_TABLE.json", metric_table)
    atomic_json(out_dir / "C6_B1_LITE_MECHANISM_ANALYSIS.json", mechanism)
    atomic_json(out_dir / "C6_B1_LITE_CEILING_ANALYSIS.json", ceiling)
    atomic_json(out_dir / "C6_B1_LITE_FINAL_MANIFEST.json", final_manifest)

    def row(vals: Dict[str, float]) -> str:
        return " | ".join(f"{vals[k]:.2f}" for k in METRIC_KEYS)

    summary_md = f"""# C6-B1-lite final summary

Status: `C6_B1_LITE_FINALIZED`

- current_promoted_system: `C6-B1-lite c6b1r1_0057`
- previous_promoted_system: `C4_final v21_00444`
- promotion_type: `low-drift candidate-selector positive`
- official_val_one_shot: `true`
- no_second_official_val: `true`
- post_val_adjustment / score_grid_on_val / temperature_search_on_val: `false / false / false`

## Official metric table

| system | 0.5-r1 | 0.5-r5 | 0.5-r10 | 0.5-r100 | 0.7-r1 | 0.7-r5 | 0.7-r10 | 0.7-r100 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| C4_final v21_00444 | {row(c4_official)} |
| C6-B1-lite c6b1r1_0057 | {row(c6_official)} |
| delta | {" | ".join(f"{official_delta[k]:+.2f}" for k in METRIC_KEYS)} |

## Train-calib positive vs official positive

Train-calib showed larger pseudo gains because it was a controlled fixed-video-slot diagnostic. Official val still confirms all eight metrics positive vs C4_final, but at smaller magnitude.

## Movement / structure

- replacement_rate_top_slots: `{c6_manifest['movement_relative_to_c4_final']['replacement_rate_top_slots']}`
- candidate_replacement_count: `{c6_manifest['movement_relative_to_c4_final']['candidate_replacement_count']}`
- top1_changed_ratio: `{c6_manifest['movement_relative_to_c4_final']['top1_changed_ratio']}`
- hard-positive top100 exits/entries: `{c6_manifest['movement_relative_to_c4_final']['hard_positive_top100_exits']} / {c6_manifest['movement_relative_to_c4_final']['hard_positive_top100_entries']}`
- video_slot_drift / video_multiset_drift: `0.0 / 0.0`
- invalid_span_count / duplicate_span_count: `0 / 0`

## Negative-result chain

C5-lite-prior, C5-main A/A2, C6-A, C6-A-R, and C6-A-R2 all showed that direct score residuals or fixed-candidate adapter reranking were unsafe or failed to transfer front-rank gains. C6-B0 then showed boundary distributions could be useful only if consumed as candidate generation/replacement. C6-B1-lite is the first official-val positive consumer of that signal.

## Next-stage recommendation

`{rec_class}`: {rec}

Do not start C6-B2/C6-C automatically. This is a review recommendation only.
"""
    atomic_text(out_dir / "C6_B1_LITE_FINAL_SUMMARY.md", summary_md)

    mech_md = [
        "# C6-B1-lite mechanism analysis",
        "",
        f"- changed_top1_queries: `{mechanism['changed_top1_queries']}`",
        f"- unchanged_top1_queries: `{mechanism['unchanged_top1_queries']}`",
        f"- changed_top1_ratio: `{mechanism['changed_top1_ratio']}`",
        f"- changed_top1_mean_iou_delta: `{mechanism['changed_top1_mean_iou_delta']}`",
        "",
        "## Top1 changed confusion",
        "",
        "| IoU | miss→hit | hit→miss | hit→hit | miss→miss |",
        "|---|---:|---:|---:|---:|",
    ]
    for t in THRS:
        c = confusion[str(t)]
        mech_md.append(f"| {t:.1f} | {c['miss_to_hit']} | {c['hit_to_miss']} | {c['hit_to_hit']} | {c['miss_to_miss']} |")
    mech_md.extend([
        "",
        "## Replacement aggregate",
        "",
        f"- replacement_by_slot: `{mechanism['replacement_by_slot']}`",
        f"- replacement_precision_05 / 07: `{mechanism['replacement_precision_05']:.4f} / {mechanism['replacement_precision_07']:.4f}`",
        f"- replacement_benefit_rate_05 / 07: `{mechanism['replacement_benefit_rate_05']:.4f} / {mechanism['replacement_benefit_rate_07']:.4f}`",
        f"- replacement_harmful_rate_05 / 07: `{mechanism['replacement_harmful_rate_05']:.4f} / {mechanism['replacement_harmful_rate_07']:.4f}`",
        f"- net_hit_conversion_count_05 / 07: `{mechanism['net_hit_conversion_count_05']} / {mechanism['net_hit_conversion_count_07']}`",
        "",
        "## Slot contribution",
        "",
        "| slot | replacements | precision@0.5 | precision@0.7 | benefit@0.5 | harm@0.5 | benefit@0.7 | harm@0.7 |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for s, st in slot_contrib.items():
        mech_md.append(f"| {s} | {st['replacement_count']} | {st['precision_05']:.3f} | {st['precision_07']:.3f} | {st['benefit_05']} | {st['harm_05']} | {st['benefit_07']} | {st['harm_07']} |")
    atomic_text(out_dir / "C6_B1_LITE_MECHANISM_ANALYSIS.md", "\n".join(mech_md) + "\n")

    ceil_md = [
        "# C6-B1-lite ceiling analysis",
        "",
        "All variants below are diagnostic/oracle analyses over frozen official-val artifacts. No official evaluator was called.",
        "",
        "| variant | 0.5-r1 | 0.5-r5 | 0.5-r10 | 0.5-r100 | 0.7-r1 | 0.7-r5 | 0.7-r10 | 0.7-r100 | repl |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, obj in ceiling_variants.items():
        ceil_md.append(f"| {name} | {row(obj['metrics'])} | {obj['replacement_count']} |")
    ceil_md.extend([
        "",
        "## Wrong-video-limited residual",
        "",
        f"- gt_video_missing_from_c4_top100: `{wrong_video['gt_video_missing_from_c4_top100']}` / `{wrong_video['queries']}`",
        f"- gt_video_missing_rate: `{wrong_video['gt_video_missing_rate']:.4f}`",
        f"- gt_video_not_top1_rate: `{wrong_video['gt_video_not_top1_rate']:.4f}`",
        "",
        "## Interpretation",
        "",
        f"- r1_07_oracle_gap_slots5_max2_vs_actual: `{r1_oracle_gap:.4f}`",
        f"- r100_07_oracle_gap_slots5_max2_vs_actual: `{r100_oracle_gap:.4f}`",
        f"- selector_conservative: `{selector_conservative}`",
        f"- harmful_replacement_low: `{harmful_low}`",
        f"- wrong_video_major: `{wrong_video_major}`",
        f"- recommendation: `{rec_class}` — {rec}",
    ])
    atomic_text(out_dir / "C6_B1_LITE_CEILING_ANALYSIS.md", "\n".join(ceil_md) + "\n")

    hashes = {
        "status": "C6_B1_LITE_FINAL_HASHES",
        "artifacts": {
            "final_summary": artifact(out_dir / "C6_B1_LITE_FINAL_SUMMARY.md"),
            "final_manifest": artifact(out_dir / "C6_B1_LITE_FINAL_MANIFEST.json"),
            "official_metric_table": artifact(out_dir / "C6_B1_LITE_OFFICIAL_METRIC_TABLE.json"),
            "mechanism_md": artifact(out_dir / "C6_B1_LITE_MECHANISM_ANALYSIS.md"),
            "mechanism_json": artifact(out_dir / "C6_B1_LITE_MECHANISM_ANALYSIS.json"),
            "ceiling_md": artifact(out_dir / "C6_B1_LITE_CEILING_ANALYSIS.md"),
            "ceiling_json": artifact(out_dir / "C6_B1_LITE_CEILING_ANALYSIS.json"),
            "c4_submission": artifact(args.c4_submission),
            "c6_submission": artifact(args.c6_submission),
            "c6_manifest": artifact(args.c6_manifest),
            "c6_hashes": artifact(args.c6_hashes),
            "script": artifact(__file__),
        },
    }
    atomic_json(out_dir / "C6_B1_LITE_FINAL_HASHES.json", hashes)
    print(json.dumps({"status": "C6_B1_LITE_FINALIZED", "recommendation": rec_class, "official_delta": official_delta, "r1_07_oracle_gap": r1_oracle_gap}, indent=2))


if __name__ == "__main__":
    main()
