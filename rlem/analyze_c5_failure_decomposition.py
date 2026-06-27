#!/usr/bin/env python
"""Read-only C5-lite-prior failure decomposition on train_calib.

This is diagnostic code, not another scorer search. It evaluates the already
materialized true temporal priors, frozen C4_final scores, and the rejected
nonzero reference c5p_00138. It cannot accept an official-val split.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
from scipy.stats import rankdata

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from rlem.c5_prior_utils import (  # noqa: E402
    TemporalPriorStore,
    attach_eval_labels,
    fast_selected,
    load_cache,
    normalize_distribution,
    sha256,
    span_clip_indices,
    write_json,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cache_npz", required=True)
    p.add_argument("--features_npz", required=True)
    p.add_argument("--temporal_prior_npz", required=True)
    p.add_argument("--term_stats_json", required=True)
    p.add_argument("--gt_jsonl", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--audit_dir", required=True)
    p.add_argument("--split_role", choices=["train_calib"], default="train_calib")
    p.add_argument("--clip_length", type=float, default=1.5)
    return p.parse_args()


def _load_npz(path: str) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as z:
        return {k: z[k] for k in z.files}


def _safe_mean(values: Iterable[float]) -> float:
    a = np.asarray(list(values), dtype=np.float64)
    return float(np.mean(a)) if a.size else float("nan")


def _quantiles(values: Iterable[float]) -> Dict[str, float]:
    a = np.asarray(list(values), dtype=np.float64)
    if not a.size:
        return {k: float("nan") for k in ("min", "p10", "p25", "median", "p75", "p90", "max", "mean")}
    q = np.quantile(a, [0, .1, .25, .5, .75, .9, 1])
    return {
        "min": float(q[0]), "p10": float(q[1]), "p25": float(q[2]),
        "median": float(q[3]), "p75": float(q[4]), "p90": float(q[5]),
        "max": float(q[6]), "mean": float(np.mean(a)),
    }


def _ranks_desc(values: np.ndarray) -> np.ndarray:
    # Average ranks are intentional: prior masses can tie for identical spans.
    return rankdata(-np.asarray(values, dtype=np.float64), method="average")


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return float("nan")
    rx, ry = rankdata(x, method="average"), rankdata(y, method="average")
    if np.std(rx) == 0 or np.std(ry) == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    if np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _pair_stats(x: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    return {"pearson": _pearson(x, y), "spearman": _spearman(x, y)}


def _iou_span(starts: np.ndarray, ends: np.ndarray, gt_start: float, gt_end: float) -> np.ndarray:
    inter = np.maximum(0.0, np.minimum(ends, gt_end) - np.maximum(starts, gt_start))
    union = np.maximum(ends, gt_end) - np.minimum(starts, gt_start)
    return np.divide(inter, union, out=np.zeros_like(inter, dtype=np.float64), where=union > 0)


def _md_table(mapping: Dict[str, object]) -> str:
    lines = ["| Diagnostic | Value |", "|---|---:|"]
    for key, value in mapping.items():
        if isinstance(value, float):
            rendered = "nan" if not math.isfinite(value) else f"{value:.8f}"
        else:
            rendered = str(value)
        lines.append(f"| `{key}` | {rendered} |")
    return "\n".join(lines)


def _write_md(path: Path, title: str, intro: str, sections: List[Tuple[str, Dict[str, object]]]):
    parts = [f"# {title}\n\n{intro}\n"]
    for heading, values in sections:
        parts.append(f"\n## {heading}\n\n{_md_table(values)}\n")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(parts), encoding="utf-8")


def main():
    args = parse_args()
    out_dir, audit_dir = Path(args.output_dir), Path(args.audit_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    audit_dir.mkdir(parents=True, exist_ok=True)
    expected = [
        out_dir / "prior_gt_alignment.json",
        out_dir / "prior_c4_redundancy.json",
        out_dir / "candidate_oracle_bottleneck.json",
        audit_dir / "C5_FD_PRIOR_GT_ALIGNMENT.md",
        audit_dir / "C5_FD_PRIOR_C4_REDUNDANCY.md",
        audit_dir / "C5_FD_CANDIDATE_ORACLE_BOTTLENECK.md",
    ]
    existing = [str(p) for p in expected if p.exists()]
    if existing:
        raise FileExistsError(existing)

    cache = load_cache(args.cache_npz)
    attach_eval_labels(cache, args.gt_jsonl)
    feat = _load_npz(args.features_npz)
    temporal = TemporalPriorStore.load(args.temporal_prior_npz)
    stats = json.loads(Path(args.term_stats_json).read_text(encoding="utf-8"))
    if stats.get("scope") != "train_fit_only" or stats.get("official_val_used") is not False:
        raise ValueError("Failure decomposition requires frozen train_fit-only term stats")
    if not np.array_equal(feat["row_group_id"].astype(np.int64), cache["row_group_id"].astype(np.int64)):
        raise ValueError("Feature/cache row alignment mismatch")
    if not np.array_equal(temporal.group_id.astype(np.int64), cache["group_ids_sorted_unique"].astype(np.int64)):
        raise ValueError("Temporal/cache group alignment mismatch")

    q_count = len(cache["desc_ids"])
    offsets = cache["desc_offsets"].astype(np.int64)
    row_gid = cache["row_group_id"].astype(np.int64)
    is_gt = cache["is_gt_video"].astype(np.float32) > .5
    iou = cache["eval_iou"].reshape(-1).astype(np.float64)
    s_c4 = feat["s_c4_final"].astype(np.float32)
    prior_score = feat["retloc_mass"].astype(np.float32)
    starts, ends = cache["start_time"].astype(np.float64), cache["end_time"].astype(np.float64)

    # Rejected c5p_00138 is diagnostic-only: lambda=.125 and the only active
    # raw weight is .075 on the frozen z(retbd_boundary_agree) term.
    term = feat["term_retbd_boundary_agree"].astype(np.float32)
    st = stats["term_stats"]["retbd_boundary_agree"]
    rejected_residual = np.float32(.125 * .075) * ((term - np.float32(st["mean"])) / np.float32(st["std"]))
    s_rejected = (s_c4 + rejected_residual).astype(np.float32)

    # Recover one GT timestamp per query. This frozen train_calib file has one
    # annotated window per desc_id; the implementation deliberately does not
    # infer labels from candidate-level training fields.
    gt_by_id = {}
    with open(args.gt_jsonl, "rt", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            gt_by_id[int(row["desc_id"])] = tuple(float(v) for v in row["ts"])

    peak_ctx, peak_bd, peak_ret = [], [], []
    gt_mass_ctx, gt_mass_bd, gt_mass_ret = [], [], []
    best_hn_mass, c4_hn_mass, gt_above_best_hn = [], [], []
    prior_selected_iou = []
    coverage_q = 0

    # Per-query prior alignment is evaluated only when the GT video is present
    # in the fixed 200-row candidate set; missing-GT-video queries are reported.
    for q in range(q_count):
        qs, qe = int(offsets[q]), int(offsets[q + 1])
        gt_rows = np.arange(qs, qe, dtype=np.int64)[is_gt[qs:qe]]
        if not gt_rows.size:
            continue
        coverage_q += 1
        gid = int(row_gid[gt_rows[0]])
        same = gt_rows[row_gid[gt_rows] == gid]
        pos = gid  # group ids are dense and aligned to temporal rows.
        ctx, pb, pe, _, _ = temporal.arrays_for_pos(pos)
        p_ctx = normalize_distribution(ctx)
        p_bd = normalize_distribution(.5 * normalize_distribution(pb) + .5 * normalize_distribution(pe))
        r = float(feat["r_video_sig"][same[0]])
        p_ret = normalize_distribution((1.0 + r) * p_ctx + (2.0 - r) * p_bd)
        gt_start, gt_end = gt_by_id[int(cache["desc_ids"][q])]
        gt_idx = span_clip_indices(gt_start, gt_end, len(p_ctx), clip_start=None, clip_end=None, clip_length=args.clip_length)
        peak_ctx.append(float(int(np.argmax(p_ctx)) in set(gt_idx.tolist())))
        peak_bd.append(float(int(np.argmax(p_bd)) in set(gt_idx.tolist())))
        peak_ret.append(float(int(np.argmax(p_ret)) in set(gt_idx.tolist())))
        gt_mass_ctx.append(float(np.sum(p_ctx[gt_idx])))
        gt_mass_bd.append(float(np.sum(p_bd[gt_idx])))
        gm = float(np.sum(p_ret[gt_idx])); gt_mass_ret.append(gm)
        hard = same[iou[same] < .3]
        if hard.size:
            best_hn_mass.append(float(np.max(prior_score[hard])))
            c4_hn = int(hard[np.argmax(s_c4[hard])])
            c4_hn_mass.append(float(prior_score[c4_hn]))
            gt_above_best_hn.append(float(gm > float(np.max(prior_score[hard]))))
        selected = int(same[np.argmax(prior_score[same])])
        prior_selected_iou.append(float(iou[selected]))

    prior_iou = np.asarray(prior_selected_iou, dtype=np.float64)
    alignment = {
        "status": "PASS",
        "stage": "C5 failure decomposition: prior-GT alignment",
        "scope": "train_calib_only",
        "official_val_used": False,
        "post_val_adjustment": False,
        "query_count": int(q_count),
        "gt_video_present_query_count": int(coverage_q),
        "gt_video_present_ratio": float(coverage_q / q_count),
        "p_ctx_peak_inside_gt_span_ratio": _safe_mean(peak_ctx),
        "p_bd_peak_inside_gt_span_ratio": _safe_mean(peak_bd),
        "p_retloc_peak_inside_gt_span_ratio": _safe_mean(peak_ret),
        "gt_span_prior_mass": {
            "p_ctx": _quantiles(gt_mass_ctx), "p_bd": _quantiles(gt_mass_bd), "p_retloc": _quantiles(gt_mass_ret),
        },
        "best_hard_negative_span_p_retloc_mass": _quantiles(best_hn_mass),
        "c4_hard_negative_span_p_retloc_mass": _quantiles(c4_hn_mass),
        "gt_span_mass_above_best_hard_negative_ratio": _safe_mean(gt_above_best_hn),
        "prior_only_selected_span_iou": _quantiles(prior_iou),
        "prior_only_oracle_video_r1_05": float(100.0 * np.mean(prior_iou >= .5)),
        "prior_only_oracle_video_r1_07": float(100.0 * np.mean(prior_iou >= .7)),
        "prior_only_selected_span_miou": float(np.mean(prior_iou)),
        "labels_used_for_training_or_scoring": False,
    }
    write_json(str(expected[0]), alignment)
    _write_md(expected[3], "C5 failure decomposition: prior–GT alignment",
              "Read-only train_calib diagnostic. GT is used only for diagnosis, never to fit or select a scorer.", [
        ("Coverage and peaks", {k: alignment[k] for k in (
            "query_count", "gt_video_present_query_count", "gt_video_present_ratio",
            "p_ctx_peak_inside_gt_span_ratio", "p_bd_peak_inside_gt_span_ratio", "p_retloc_peak_inside_gt_span_ratio")}),
        ("Prior-only localization", {k: alignment[k] for k in (
            "gt_span_mass_above_best_hard_negative_ratio", "prior_only_oracle_video_r1_05",
            "prior_only_oracle_video_r1_07", "prior_only_selected_span_miou")}),
        ("P_retloc mass distributions", {
            "GT mass mean": alignment["gt_span_prior_mass"]["p_retloc"]["mean"],
            "GT mass median": alignment["gt_span_prior_mass"]["p_retloc"]["median"],
            "best hard-negative mean": alignment["best_hard_negative_span_p_retloc_mass"]["mean"],
            "best hard-negative median": alignment["best_hard_negative_span_p_retloc_mass"]["median"],
        }),
    ])

    # Global feature redundancy, plus within-video rank agreement.
    prior_features = ["ctx_mass", "bd_mass", "retloc_mass", "ctx_boundary_agree", "bd_boundary_agree"]
    heads = {"q_joint": cache["q_joint"], "q_bd": cache["q_bd"], "e_fp": cache["e_fp"]}
    correlations = {
        name: {head: _pair_stats(feat[name], values) for head, values in heads.items()}
        for name in prior_features
    }
    group_sort = cache["group_sort_idx"].astype(np.int64)
    group_starts = cache["group_offsets"].astype(np.int64)
    group_ends = np.r_[group_starts[1:], len(group_sort)]
    rank_corr_all, rank_corr_gt = [], []
    for gs, ge in zip(group_starts, group_ends):
        rows = group_sort[int(gs):int(ge)]
        if len(rows) < 2:
            continue
        corr = _spearman(prior_score[rows], s_c4[rows])
        if math.isfinite(corr):
            rank_corr_all.append(corr)
            if np.any(is_gt[rows]):
                rank_corr_gt.append(corr)
    c4_top = np.empty(q_count, dtype=np.int64)
    for q in range(q_count):
        qs, qe = int(offsets[q]), int(offsets[q + 1])
        c4_top[q] = qs + int(np.argmax(s_c4[qs:qe]))
    top_peak_overlap = feat["retloc_peak_inside"][c4_top]
    top_mass = prior_score[c4_top]
    best_iou_mass_diff = []
    for q in range(q_count):
        qs, qe = int(offsets[q]), int(offsets[q + 1])
        gt_rows = np.arange(qs, qe, dtype=np.int64)[is_gt[qs:qe]]
        if gt_rows.size:
            best = int(gt_rows[np.argmax(iou[gt_rows])])
            best_iou_mass_diff.append(float(prior_score[best] - prior_score[c4_top[q]]))
    redundancy = {
        "status": "PASS",
        "stage": "C5 failure decomposition: prior redundancy with C4_final",
        "scope": "train_calib_only",
        "official_val_used": False,
        "post_val_adjustment": False,
        "feature_head_correlations": correlations,
        "prior_only_vs_c4_final_within_video_spearman": {
            "all_video_groups": _quantiles(rank_corr_all),
            "gt_video_groups": _quantiles(rank_corr_gt),
        },
        "p_retloc_peak_inside_c4_final_selected_span_ratio": float(np.mean(top_peak_overlap)),
        "c4_final_selected_span_prior_mass": _quantiles(top_mass),
        "best_iou_minus_c4_top_prior_mass": _quantiles(best_iou_mass_diff),
        "same_video_rank_highly_consistent_ratio_ge_0_7": float(np.mean(np.asarray(rank_corr_all) >= .7)),
    }
    write_json(str(expected[1]), redundancy)
    flat_corr = {}
    for feature, by_head in correlations.items():
        for head, values in by_head.items():
            flat_corr[f"{feature} vs {head} Pearson"] = values["pearson"]
            flat_corr[f"{feature} vs {head} Spearman"] = values["spearman"]
    _write_md(expected[4], "C5 failure decomposition: prior redundancy with C4_final",
              "Correlations and rank agreement are descriptive only; no coefficient is fitted.", [
        ("Prior/head correlations", flat_corr),
        ("Within-video agreement", {
            "all-group mean Spearman": redundancy["prior_only_vs_c4_final_within_video_spearman"]["all_video_groups"]["mean"],
            "GT-group mean Spearman": redundancy["prior_only_vs_c4_final_within_video_spearman"]["gt_video_groups"]["mean"],
            "groups with Spearman >= 0.7": redundancy["same_video_rank_highly_consistent_ratio_ge_0_7"],
            "P_retloc peak inside C4 top span": redundancy["p_retloc_peak_inside_c4_final_selected_span_ratio"],
        }),
        ("Prior-mass comparison", {
            "C4 selected mass mean": redundancy["c4_final_selected_span_prior_mass"]["mean"],
            "best-IoU minus C4-top mass mean": redundancy["best_iou_minus_c4_top_prior_mass"]["mean"],
            "best-IoU minus C4-top mass median": redundancy["best_iou_minus_c4_top_prior_mass"]["median"],
        }),
    ])

    # Candidate/oracle bottleneck. Re-run only the frozen top100/NMS protocol.
    _, c4_order, selected = fast_selected(cache, s_c4, 100, 100, .7)
    avail05, avail07, raw_oracle_iou, pre_oracle_iou, post_oracle_iou = [], [], [], [], []
    best_rank_c4, best_rank_prior, best_rank_rejected = [], [], []
    rejected_rank_delta, c4_gap = [], []
    best_in_pre, best_in_post, nms_removed_best = [], [], []
    better_than_c4_removed_by_nms = []
    for q in range(q_count):
        qs, qe = int(offsets[q]), int(offsets[q + 1])
        local = np.arange(qs, qe, dtype=np.int64)
        gt_rows = local[is_gt[qs:qe]]
        if not gt_rows.size:
            continue
        best = int(gt_rows[np.argmax(iou[gt_rows])])
        raw_best = float(iou[best])
        avail05.append(float(raw_best >= .5)); avail07.append(float(raw_best >= .7)); raw_oracle_iou.append(raw_best)
        c4_r = _ranks_desc(s_c4[gt_rows]); p_r = _ranks_desc(prior_score[gt_rows]); x_r = _ranks_desc(s_rejected[gt_rows])
        bp = int(np.where(gt_rows == best)[0][0])
        best_rank_c4.append(float(c4_r[bp])); best_rank_prior.append(float(p_r[bp])); best_rank_rejected.append(float(x_r[bp]))
        rejected_rank_delta.append(float(x_r[bp] - c4_r[bp]))
        c4_gt_top = int(gt_rows[np.argmax(s_c4[gt_rows])])
        c4_gap.append(float(raw_best - iou[c4_gt_top]))
        pre_local = c4_order[q, :100]
        post_local = np.asarray(selected[q], dtype=np.int64)
        pre_global = qs + pre_local
        post_global = qs + post_local
        in_pre = bool(np.any(pre_global == best)); in_post = bool(np.any(post_global == best))
        best_in_pre.append(float(in_pre)); best_in_post.append(float(in_post)); nms_removed_best.append(float(in_pre and not in_post))
        pre_gt = pre_global[is_gt[pre_global]]; post_gt = post_global[is_gt[post_global]]
        pre_oracle_iou.append(float(np.max(iou[pre_gt])) if pre_gt.size else 0.0)
        post_oracle_iou.append(float(np.max(iou[post_gt])) if post_gt.size else 0.0)
        better = gt_rows[iou[gt_rows] > iou[c4_gt_top] + 1e-12]
        better_pre = better[np.isin(better, pre_global)]
        better_post = better[np.isin(better, post_global)]
        better_than_c4_removed_by_nms.append(float(better_pre.size > 0 and better_post.size == 0))

    raw_o, pre_o, post_o = map(lambda x: np.asarray(x, dtype=np.float64), (raw_oracle_iou, pre_oracle_iou, post_oracle_iou))
    candidate = {
        "status": "PASS",
        "stage": "C5 failure decomposition: fixed-candidate/oracle bottleneck",
        "scope": "train_calib_only",
        "official_val_used": False,
        "post_val_adjustment": False,
        "gt_video_present_query_count": int(len(raw_o)),
        "gt_video_has_iou05_candidate_ratio": _safe_mean(avail05),
        "gt_video_has_iou07_candidate_ratio": _safe_mean(avail07),
        "best_iou_span_rank": {
            "C4_final": _quantiles(best_rank_c4),
            "prior_only": _quantiles(best_rank_prior),
            "rejected_c5p_00138": _quantiles(best_rank_rejected),
        },
        "rejected_c5p_00138_best_iou_rank_delta_vs_c4": _quantiles(rejected_rank_delta),
        "rejected_c5p_00138_best_iou_rank_improved_ratio": float(np.mean(np.asarray(rejected_rank_delta) < 0)),
        "rejected_c5p_00138_best_iou_rank_worsened_ratio": float(np.mean(np.asarray(rejected_rank_delta) > 0)),
        "c4_selected_vs_best_iou_gap": _quantiles(c4_gap),
        "best_iou_candidate_in_pre_nms_top100_ratio": _safe_mean(best_in_pre),
        "best_iou_candidate_survives_nms_ratio": _safe_mean(best_in_post),
        "best_iou_candidate_removed_by_nms_ratio": _safe_mean(nms_removed_best),
        "better_than_c4_selected_span_removed_by_nms_ratio": _safe_mean(better_than_c4_removed_by_nms),
        "localization_oracle": {
            "raw_200": {"miou": float(np.mean(raw_o)), "r1_05": float(100*np.mean(raw_o >= .5)), "r1_07": float(100*np.mean(raw_o >= .7))},
            "pre_nms_top100": {"miou": float(np.mean(pre_o)), "r1_05": float(100*np.mean(pre_o >= .5)), "r1_07": float(100*np.mean(pre_o >= .7))},
            "post_nms_top100": {"miou": float(np.mean(post_o)), "r1_05": float(100*np.mean(post_o >= .5)), "r1_07": float(100*np.mean(post_o >= .7))},
        },
        "nms_protocol": {"effective_top_n": 100, "nms_thd": .7, "max_after_nms": 100},
        "rejected_config_is_diagnostic_only": True,
    }
    write_json(str(expected[2]), candidate)
    _write_md(expected[5], "C5 failure decomposition: candidate/oracle bottleneck",
              "The frozen C4_final ordering and unchanged NMS=0.7 are replayed read-only. c5p_00138 is used only to explain failure.", [
        ("Candidate availability", {k: candidate[k] for k in (
            "gt_video_present_query_count", "gt_video_has_iou05_candidate_ratio", "gt_video_has_iou07_candidate_ratio")}),
        ("Best-IoU rank", {
            "C4_final mean": candidate["best_iou_span_rank"]["C4_final"]["mean"],
            "prior-only mean": candidate["best_iou_span_rank"]["prior_only"]["mean"],
            "rejected c5p_00138 mean": candidate["best_iou_span_rank"]["rejected_c5p_00138"]["mean"],
            "c5p_00138 improved ratio": candidate["rejected_c5p_00138_best_iou_rank_improved_ratio"],
            "c5p_00138 worsened ratio": candidate["rejected_c5p_00138_best_iou_rank_worsened_ratio"],
        }),
        ("Top100/NMS attribution", {k: candidate[k] for k in (
            "best_iou_candidate_in_pre_nms_top100_ratio", "best_iou_candidate_survives_nms_ratio",
            "best_iou_candidate_removed_by_nms_ratio", "better_than_c4_selected_span_removed_by_nms_ratio")}),
        ("Localization oracle", {
            "raw-200 mIoU": candidate["localization_oracle"]["raw_200"]["miou"],
            "raw-200 R1@0.5": candidate["localization_oracle"]["raw_200"]["r1_05"],
            "raw-200 R1@0.7": candidate["localization_oracle"]["raw_200"]["r1_07"],
            "post-NMS mIoU": candidate["localization_oracle"]["post_nms_top100"]["miou"],
            "post-NMS R1@0.5": candidate["localization_oracle"]["post_nms_top100"]["r1_05"],
            "post-NMS R1@0.7": candidate["localization_oracle"]["post_nms_top100"]["r1_07"],
        }),
    ])

    print(json.dumps({"status": "PASS", "outputs": [str(p) for p in expected]}, indent=2))


if __name__ == "__main__":
    main()
