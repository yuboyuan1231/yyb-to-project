#!/usr/bin/env python
"""Exactly-one official-val one-shot for C5-main-inference-A3b-wide."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rlem.c4_lite_utils import flatten_eval_metrics, submission_from_groups  # noqa: E402
from rlem.c5_main_a_utils import (  # noqa: E402
    EndpointPriorEngine,
    artifact,
    exact_candidate_indices,
    group_max,
    load_c4_final_scores,
    load_npz_arrays,
    retrieval_gate,
    retrieval_z,
    setting_id,
    standardized_delta,
)
from rlem.c5_prior_utils import (  # noqa: E402
    attach_eval_labels,
    groups_for_submission,
    load_cache,
    localization_diagnostics,
    movement_diagnostics,
    sha256,
)
from standalone_eval.eval import eval_retrieval, load_jsonl  # noqa: E402


METRIC_KEYS = ("0.5-r1", "0.5-r5", "0.5-r10", "0.5-r100", "0.7-r1", "0.7-r5", "0.7-r10", "0.7-r100")
FRONT_KEYS = ("0.5-r1", "0.5-r5", "0.5-r10", "0.7-r1", "0.7-r5", "0.7-r10")
EXPECTED_CFG = {
    "config_id": "c5ma_0214",
    "family": "role_endpoint_injection",
    "alpha": 0.25,
    "beta": 0.05,
    "tau": 1.5,
    "lambda": 0.05,
    "gamma": 0.05,
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--allow_official_val", action="store_true", required=True)
    p.add_argument("--cache_npz", required=True)
    p.add_argument("--temporal_prior_npz", required=True)
    p.add_argument("--c4_final_scores_npz", required=True)
    p.add_argument("--stats_json", required=True)
    p.add_argument("--freeze_manifest", required=True)
    p.add_argument("--gate_addendum_json", required=True)
    p.add_argument("--dataset_config", required=True)
    p.add_argument("--gt_jsonl", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--audit_dir", required=True)
    p.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    p.add_argument("--resume_from_scores_after_infra_crash", action="store_true")
    p.add_argument("--effective_top_n", type=int, default=100)
    p.add_argument("--max_after_nms", type=int, default=100)
    p.add_argument("--nms_thd", type=float, default=0.7)
    return p.parse_args()


def center(values: np.ndarray, gid: np.ndarray, groups: int) -> np.ndarray:
    sums = np.bincount(gid, weights=values.astype(np.float64), minlength=groups)
    cnt = np.bincount(gid, minlength=groups)
    return (values - sums[gid] / np.maximum(cnt[gid], 1)).astype(np.float32)


def rank_score(order: np.ndarray) -> np.ndarray:
    inv = np.empty_like(order, dtype=np.int16)
    ranks = np.broadcast_to(np.arange(order.shape[1], dtype=np.int16), order.shape)
    np.put_along_axis(inv, order, ranks, axis=1)
    return (-inv.astype(np.float32)).reshape(-1)


def a3b_wide_order(sloc: np.ndarray, base_order: np.ndarray, video_idx: np.ndarray) -> np.ndarray:
    qn = base_order.shape[0]
    sm = sloc.reshape(qn, 200)
    out = np.empty_like(base_order)
    for q in range(qn):
        top = base_order[q, :100].tolist()
        seq = [int(video_idx[q, i]) for i in top]
        quota = Counter(seq)
        queues = {}
        for video, count in quota.items():
            pool = np.flatnonzero(video_idx[q] == video)
            queues[video] = pool[np.argsort(-sm[q, pool], kind="stable")[:count]].tolist()
        head = defaultdict(int)
        used = set()
        chosen = []
        for video in seq:
            i = queues[video][head[video]]
            head[video] += 1
            chosen.append(i)
            used.add(i)
        rest = [i for i in base_order[q].tolist() if i not in used]
        out[q] = np.asarray(chosen + rest, dtype=np.int64)
    return out


def structure(base_order: np.ndarray, order: np.ndarray, video_idx: np.ndarray) -> Dict[str, object]:
    base_top = base_order[:, :100]
    new_top = order[:, :100]
    mem, seq, multi = [], [], []
    quota = 0
    dup = 0
    for q in range(base_order.shape[0]):
        b = base_top[q].tolist()
        n = new_top[q].tolist()
        mem.append(set(b) != set(n))
        dup += 100 - len(set(n))
        bs = [int(video_idx[q, i]) for i in b]
        ns = [int(video_idx[q, i]) for i in n]
        seq.append(bs != ns)
        multi.append(Counter(bs) != Counter(ns))
        quota += sum((Counter(bs) - Counter(ns)).values()) + sum((Counter(ns) - Counter(bs)).values())
    return {
        "top100_membership_drift_query_ratio": float(np.mean(mem)),
        "top100_membership_symmetric_difference_rows": int(
            sum(len(set(base_order[q, :100]) ^ set(order[q, :100])) for q in range(base_order.shape[0]))
        ),
        "video_multiset_drift_query_ratio": float(np.mean(multi)),
        "video_slot_sequence_drift_query_ratio": float(np.mean(seq)),
        "quota_violation_count": int(quota),
        "duplicate_span_count": int(dup),
        "constraint_satisfied": bool((not any(multi)) and quota == 0 and dup == 0 and (not any(seq))),
    }


def order_movement(base: np.ndarray, sloc: np.ndarray, order: np.ndarray, cache: Dict[str, np.ndarray]) -> Dict[str, float]:
    qn = order.shape[0]
    base_order = np.argsort(-base.reshape(qn, 200), axis=1, kind="stable")
    base_inv = np.empty_like(base_order, dtype=np.int16)
    new_inv = np.empty_like(order, dtype=np.int16)
    ranks = np.broadcast_to(np.arange(200, dtype=np.int16), base_order.shape)
    np.put_along_axis(base_inv, base_order, ranks, axis=1)
    np.put_along_axis(new_inv, order, ranks, axis=1)
    hard = cache["eval_y05"].astype(bool)
    diff = base_inv.astype(np.float64) - new_inv.astype(np.float64)
    spearman = 1.0 - 6.0 * np.sum(diff * diff, axis=1) / (200.0 * (200.0 * 200.0 - 1.0))
    exits = int(np.sum(hard & (base_inv < 100) & (new_inv >= 100)))
    entries = int(np.sum(hard & (base_inv >= 100) & (new_inv < 100)))
    return {
        "queries": int(qn),
        "top1_changed_ratio": float(np.mean(base_order[:, 0] != order[:, 0])),
        "hard_positive_top100_exits": exits,
        "hard_positive_top100_entries": entries,
        "hard_positive_top100_exit_ratio": float(exits / max(int(hard.sum()), 1)),
        "pearson_base_localization_score": float(np.corrcoef(base.astype(np.float64), sloc.astype(np.float64))[0, 1]),
        "pearson_c4_final_candidate_rank_score": float(np.corrcoef(base.astype(np.float64), rank_score(order).astype(np.float64))[0, 1]),
        "mean_within_query_spearman": float(np.mean(spearman)),
    }


def metric_deltas(new: Dict[str, float], base: Dict[str, float]) -> Dict[str, float]:
    return {k: float(new[k] - base[k]) for k in METRIC_KEYS if k in new and k in base}


def load_metrics(path: str) -> Dict[str, float]:
    obj = json.loads(Path(path).read_text(encoding="utf-8"))
    if "VCMR" in obj:
        return {k: float(obj["VCMR"][k]) for k in METRIC_KEYS}
    flat = flatten_eval_metrics(obj)
    return {k: float(flat[k]) for k in METRIC_KEYS}


def load_manifest_metrics(path: str, key: str) -> Dict[str, float]:
    obj = json.loads(Path(path).read_text(encoding="utf-8"))
    return {k: float(obj["metrics"][key][k]) for k in METRIC_KEYS}


def prediction_jsonl_from_submission(submission: Dict, output_path: str) -> None:
    path = Path(output_path)
    partial = Path(str(path) + ".partial")
    with open(partial, "w", encoding="utf-8") as f:
        for row in submission["VCMR"]:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    os.replace(partial, path)


def atomic_json(path: str, obj: Dict) -> None:
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    partial = Path(str(path_obj) + ".partial")
    partial.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(partial, path_obj)


def classify(delta: Dict[str, float], st: Dict[str, object]) -> str:
    front_pos = sum(delta.get(k, 0.0) > 0.0 for k in FRONT_KEYS)
    r1_nonnegative = delta.get("0.5-r1", -1e9) >= 0.0 and delta.get("0.7-r1", -1e9) >= 0.0
    r1_pos = delta.get("0.5-r1", 0.0) > 0.0 or delta.get("0.7-r1", 0.0) > 0.0
    r100_min = min(delta.get("0.5-r100", -1e9), delta.get("0.7-r100", -1e9))
    structural = bool(st.get("constraint_satisfied")) and int(st.get("quota_violation_count", 1)) == 0 and int(st.get("duplicate_span_count", 1)) == 0
    if front_pos == 6 and r100_min >= 0.0 and structural:
        return "OFFICIAL_C5_STRONG_POSITIVE"
    if front_pos >= 4 and r1_nonnegative and r1_pos and r100_min >= -0.05 and structural:
        return "OFFICIAL_C5_POSITIVE"
    if front_pos >= 3 and r1_pos and r100_min >= -0.10 and structural:
        return "OFFICIAL_C5_FRONT_RANK_POSITIVE"
    if front_pos >= 3 and r1_pos and r100_min < -0.10 and structural:
        return "OFFICIAL_C5_MIXED_RECALL_TRADEOFF"
    return "OFFICIAL_C5_NEGATIVE"


def main():
    args = parse_args()
    if (args.effective_top_n, args.max_after_nms, args.nms_thd) != (100, 100, 0.7):
        raise ValueError("Frozen retrieval constants must remain 100/100/0.7")

    out_dir = Path(args.output_dir)
    audit_dir = Path(args.audit_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    audit_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "predictions_jsonl": out_dir / "official_val_predictions.jsonl",
        "scores_npz": out_dir / "official_val_scores.npz",
        "metrics_json": out_dir / "official_val_metrics.json",
        "submission_json": out_dir / "official_val_submission.json",
        "audit_md": audit_dir / "C5_MAIN_A3_OFFICIAL_VAL_ONE_SHOT_AUDIT.md",
        "manifest_json": audit_dir / "C5_MAIN_A3_OFFICIAL_VAL_ONE_SHOT_MANIFEST.json",
        "hashes_json": audit_dir / "C5_MAIN_A3_OFFICIAL_VAL_ONE_SHOT_HASHES.json",
    }
    existing_paths = {key: path for key, path in outputs.items() if path.exists()}
    if args.resume_from_scores_after_infra_crash:
        disallowed = {key: path for key, path in existing_paths.items() if key != "scores_npz"}
        if disallowed:
            raise FileExistsError(
                "resume permits only the pre-existing score NPZ; refusing rerun: "
                + ", ".join(str(path) for path in disallowed.values())
            )
        if not outputs["scores_npz"].exists():
            raise FileNotFoundError("resume requested but official_val_scores.npz is absent")
    elif existing_paths:
        raise FileExistsError("one-shot outputs already exist; refusing rerun: " + ", ".join(str(p) for p in existing_paths.values()))

    freeze = json.loads(Path(args.freeze_manifest).read_text(encoding="utf-8"))
    if freeze.get("status") != "C5_MAIN_A3_FREEZE_REVIEW_PASS" or freeze.get("selected_config") != "c5ma_0214":
        raise ValueError("Freeze manifest does not select c5ma_0214")
    params = freeze.get("selected_parameters", {})
    for key, value in EXPECTED_CFG.items():
        if params.get(key) != value:
            raise ValueError(f"Frozen parameter mismatch for {key}: {params.get(key)} != {value}")
    gate = json.loads(Path(args.gate_addendum_json).read_text(encoding="utf-8"))
    if gate.get("frozen_config_review", {}).get("config_id") != "c5ma_0214":
        raise ValueError("Gate addendum does not reference c5ma_0214")
    stats = json.loads(Path(args.stats_json).read_text(encoding="utf-8"))
    if stats.get("stats_source") != "train_fit_only" or stats.get("official_val_used") is not False:
        raise ValueError("A2a stats are not train_fit-only frozen stats")

    cache = load_cache(args.cache_npz)
    gid = cache["row_group_id"].astype(np.int64)
    queries = len(cache["desc_ids"])
    if len(gid) != queries * 200:
        raise ValueError("official val cache must have exactly 200 rows/query")
    groups = len(cache["group_ids_sorted_unique"])
    base = load_c4_final_scores(args.c4_final_scores_npz, len(gid))
    temporal_length = load_npz_arrays(args.temporal_prior_npz, ("temporal_length",))["temporal_length"].astype(np.int64)
    start_idx, end_idx, endpoint_audit = exact_candidate_indices(cache["start_time"], cache["end_time"], gid, temporal_length)
    setting = setting_id(EXPECTED_CFG["alpha"], EXPECTED_CFG["beta"], EXPECTED_CFG["tau"], EXPECTED_CFG["lambda"])

    if args.resume_from_scores_after_infra_crash:
        with np.load(outputs["scores_npz"], allow_pickle=False) as zf:
            base_saved = zf["s_c4_final"].astype(np.float32)
            sloc = zf["s_loc_c5"].astype(np.float32)
            rank = zf["s_c5_rank"].astype(np.float32)
            order = zf["order"].astype(np.int64)
            base_order = zf["base_order"].astype(np.int64)
        if base_saved.shape != base.shape or not np.allclose(base_saved, base, rtol=0.0, atol=0.0):
            raise ValueError("resume score NPZ does not match C4_final row scores")
        role_sanity = {"not_recomputed": True, "reason": "resume_from_valid_score_artifact_after_infra_crash_before_evaluator"}
    else:
        group_score = group_max(base, gid, groups)
        rz = retrieval_z(group_score, stats["retrieval_score_stats"])
        gate_values = retrieval_gate(rz, EXPECTED_CFG["tau"])
        engine = EndpointPriorEngine.load(args.temporal_prior_npz, args.device)
        log_s, log_e, role_sanity = engine.role_log_priors(EXPECTED_CFG["alpha"], EXPECTED_CFG["beta"])
        raw = engine.injected_endpoint_delta(
            log_s,
            log_e,
            np.float32(EXPECTED_CFG["lambda"]) * gate_values,
            gid,
            start_idx,
            end_idx,
            chunk_rows=500_000,
        )
        delta_z = standardized_delta(center(raw, gid, groups), stats["delta_stats"][setting])
        sloc = (base + np.float32(EXPECTED_CFG["gamma"]) * delta_z).astype(np.float32)
        base_order = np.argsort(-base.reshape(queries, 200), axis=1, kind="stable")
        video_idx = cache["video_idx"].reshape(queries, 200)
        order = a3b_wide_order(sloc, base_order, video_idx)
        rank = rank_score(order)
    video_idx = cache["video_idx"].reshape(queries, 200)
    st = structure(base_order, order, video_idx)
    if not st["constraint_satisfied"]:
        raise ValueError(f"A3b structural constraint failed before evaluation: {st}")

    if not args.resume_from_scores_after_infra_crash:
        npz_partial = str(outputs["scores_npz"]) + ".partial"
        with open(npz_partial, "wb") as f:
            np.savez_compressed(
                f,
                s_c4_final=base,
                s_loc_c5=sloc,
                s_c5_rank=rank,
                delta_z=delta_z,
                order=order.astype(np.int16),
                base_order=base_order.astype(np.int16),
                row_group_id=gid.astype(np.int32),
            )
        os.replace(npz_partial, outputs["scores_npz"])

    submission = submission_from_groups(
        groups_for_submission(cache, rank, "s_c5"),
        "s_c5",
        args.dataset_config,
        "val",
        effective_top_n=100,
        max_after_nms=100,
        nms_thd=0.7,
    )
    submission.pop("_row_counts", None)
    atomic_json(str(outputs["submission_json"]), submission)
    prediction_jsonl_from_submission(submission, str(outputs["predictions_jsonl"]))

    metrics_raw = eval_retrieval(
        submission,
        load_jsonl(args.gt_jsonl),
        iou_thds=(0.5, 0.7),
        verbose=False,
        match_number=True,
        use_desc_type=True,
    )
    atomic_json(str(outputs["metrics_json"]), metrics_raw)
    flat = {k: float(flatten_eval_metrics(metrics_raw)[k]) for k in METRIC_KEYS}

    attach_eval_labels(cache, args.gt_jsonl)
    move = order_movement(base, sloc, order, cache)
    move_rank = movement_diagnostics(cache, base, rank)
    loc = localization_diagnostics(cache, base, rank)

    c4_final_metrics = load_metrics("results/rlem_c4_r2_cal_v21_official_val/val_v21_metrics.json")
    c4_lite_metrics = load_metrics("results/rlem_c4_lite_official_val/val_c4_lite_metrics.json")
    c4_lite_manifest = "results/rlem_c4_lite_official_val/VAL_C4_LITE_ONE_SHOT_MANIFEST.json"
    c31_metrics = load_manifest_metrics(c4_lite_manifest, "frozen_c31")
    c0_metrics = load_manifest_metrics(c4_lite_manifest, "c0")
    delta_c4 = metric_deltas(flat, c4_final_metrics)
    delta_lite = metric_deltas(flat, c4_lite_metrics)
    delta_c31 = metric_deltas(flat, c31_metrics)
    outcome = classify(delta_c4, st)
    next_step = (
        "C5 verified under fixed-candidate video-slot protocol. Request human review before planning C6 or manuscript integration."
        if outcome in {"OFFICIAL_C5_POSITIVE", "OFFICIAL_C5_STRONG_POSITIVE", "OFFICIAL_C5_FRONT_RANK_POSITIVE"}
        else "C5 A3b-wide did not generalize sufficiently to official val. Close fixed-candidate C5-main and do not run more official val."
    )

    manifest = {
        "status": outcome,
        "date": datetime.now().strftime("%Y-%m-%d"),
        "stage": "C5-main-inference-A3b-wide official-val one-shot",
        "official_val_one_shot": True,
        "learned_config_count": 1,
        "selected_config": "c5ma_0214",
        "variant": "A3b-wide video-slot/quota-preserved fixed-candidate span replacement",
        "score_grid_on_val": False,
        "temperature_search_on_val": False,
        "post_val_adjustment": False,
        "official_val_used_for_selection": False,
        "official_val_authorized_before_run": True,
        "successful_metric_result_count": 1,
        "failed_infrastructure_attempt_count": int(args.resume_from_scores_after_infra_crash),
        "resume_from_scores_after_infra_crash": bool(args.resume_from_scores_after_infra_crash),
        "training_used": False,
        "model_conquer_modified": False,
        "candidate_generation_modified": False,
        "NMS_modified": False,
        "effective_top_n_modified": False,
        "max_after_nms_modified": False,
        "evaluator_modified": False,
        "C5_main_B_used": False,
        "candidate_regeneration_used": False,
        "C6_used": False,
        "C4_main_used": False,
        "VS_R2_head_used": False,
        "retrieval_constants": {"effective_top_n": 100, "nms_thd": 0.7, "max_after_nms": 100},
        "frozen_config": EXPECTED_CFG,
        "frozen_stats_source": {"path": args.stats_json, "sha256": sha256(args.stats_json), "setting_id": setting},
        "metrics": {
            "c0": c0_metrics,
            "frozen_c31": c31_metrics,
            "frozen_c4_lite": c4_lite_metrics,
            "c4_final_v21_00444": c4_final_metrics,
            "c5_main_a3b_wide_c5ma_0214": flat,
            "delta_vs_c4_final": delta_c4,
            "delta_vs_c4_lite": delta_lite,
            "delta_vs_c31": delta_c31,
        },
        "front_rank_deltas_vs_c4_final": {k: delta_c4[k] for k in FRONT_KEYS},
        "r100_safety_deltas_vs_c4_final": {k: delta_c4[k] for k in ("0.5-r100", "0.7-r100")},
        "movement_relative_to_c4_final": move,
        "movement_rank_score_relative_to_c4_final": move_rank,
        "structure": st,
        "localization_diagnostics_if_computable": loc,
        "role_prior_sanity": role_sanity,
        "endpoint_index_audit": endpoint_audit,
        "outcome_classification": outcome,
        "next_step_recommendation": next_step,
        "no_second_official_val": True,
    }
    atomic_json(str(outputs["manifest_json"]), manifest)

    md = [
        "# C5-main-inference-A3b-wide official-val one-shot audit",
        "",
        f"- Status: `{outcome}`",
        "- official_val_one_shot: `true`",
        "- learned_config_count: `1`",
        "- selected_config: `c5ma_0214`",
        "- score_grid_on_val / temperature_search_on_val / post_val_adjustment: `false / false / false`",
        "- successful_metric_result_count: `1`",
        "",
        "## Official-val metrics",
        "",
        "| system | 0.5-r1 | 0.5-r5 | 0.5-r10 | 0.5-r100 | 0.7-r1 | 0.7-r5 | 0.7-r10 | 0.7-r100 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, vals in [
        ("C0", c0_metrics),
        ("C3.1", c31_metrics),
        ("C4-lite", c4_lite_metrics),
        ("C4_final v21_00444", c4_final_metrics),
        ("C5 A3b-wide c5ma_0214", flat),
    ]:
        md.append("| " + name + " | " + " | ".join(f"{vals[k]:.2f}" for k in METRIC_KEYS) + " |")
    md.extend(["", "## Deltas vs C4_final", "", "| metric | delta |", "|---|---:|"])
    for k in METRIC_KEYS:
        md.append(f"| {k} | {delta_c4[k]:+.2f} |")
    md.extend([
        "",
        "## Movement / safety",
        "",
        f"- top1_changed_ratio: `{move['top1_changed_ratio']}`",
        f"- hard-positive top100 exits/entries: `{move['hard_positive_top100_exits']} / {move['hard_positive_top100_entries']}`",
        f"- Pearson(base, S_loc): `{move['pearson_base_localization_score']}`",
        f"- mean within-query Spearman: `{move['mean_within_query_spearman']}`",
        f"- video_multiset_drift_query_ratio: `{st['video_multiset_drift_query_ratio']}`",
        f"- video_slot_sequence_drift_query_ratio: `{st['video_slot_sequence_drift_query_ratio']}`",
        f"- quota_violation_count / duplicate_span_count: `{st['quota_violation_count']} / {st['duplicate_span_count']}`",
        "",
        "No post-val adjustment was performed. No second official-val run is authorized or implied.",
        "",
        f"Next step: {next_step}",
        "",
    ])
    outputs["audit_md"].write_text("\n".join(md), encoding="utf-8")

    hashes = {
        "status": "HASHES_MATERIALIZED_AFTER_ONE_SHOT",
        "official_val_one_shot": True,
        "successful_metric_result_count": 1,
        "artifacts": {
            "freeze_manifest": artifact(args.freeze_manifest),
            "gate_addendum_json": artifact(args.gate_addendum_json),
            "stats_json": artifact(args.stats_json),
            "official_val_cache": artifact(args.cache_npz),
            "official_val_temporal_prior": artifact(args.temporal_prior_npz),
            "official_val_c4_final_scores": artifact(args.c4_final_scores_npz),
            "official_val_predictions_jsonl": artifact(str(outputs["predictions_jsonl"])),
            "official_val_submission_json": artifact(str(outputs["submission_json"])),
            "official_val_scores_npz": artifact(str(outputs["scores_npz"])),
            "official_val_metrics_json": artifact(str(outputs["metrics_json"])),
            "one_shot_manifest": artifact(str(outputs["manifest_json"])),
            "one_shot_audit": artifact(str(outputs["audit_md"])),
            "one_shot_code": artifact(__file__),
            "model_conquer": artifact("model/conquer.py"),
            "evaluator": artifact("standalone_eval/eval.py"),
        },
    }
    atomic_json(str(outputs["hashes_json"]), hashes)
    print(json.dumps({"status": outcome, "metrics": flat, "delta_vs_c4_final": delta_c4}, indent=2))


if __name__ == "__main__":
    main()
