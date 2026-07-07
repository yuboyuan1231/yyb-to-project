#!/usr/bin/env python3
"""C13 feature/baseline pivot audit.

This script is intentionally read-only with respect to models and validation:
it aggregates prior train-only C12/C9 artifacts, writes C13 decision files, and
does not read official prediction pools or run any official evaluator path.
"""

from __future__ import annotations

import hashlib
import json
import pickle
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
OUT0 = ROOT / "c13_0_c12_stop_archive"
OUT1 = ROOT / "c13_1_strong_feature_feasibility"
OUT2 = ROOT / "c13_2_baseline_architecture_feasibility"
OUT3 = ROOT / "c13_3_next_step_decision"
DATA = Path("/home/a/yybwork/data/yyb/tvr_feature_release/data")

PROMOTED = "C7-B6 R1SelectiveTop1"
FINAL_C12_STATUS = "C12_SPAN_RANKER_LINE_STOP_NEED_STRONGER_FEATURE_OR_BASELINE"

PREM_REPO = "https://github.com/hdy007007/PREM"
PREM_PAPER = "https://arxiv.org/abs/2402.13576"
MINUTE_PAPER = "https://arxiv.org/abs/2301.13606"
MAVR_REF = "https://doi.org/10.1109/access.2025.3542720"


def read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def duration_bucket(row: dict[str, Any]) -> str:
    dur = float(row["ts"][1]) - float(row["ts"][0])
    if dur <= 5:
        return "short"
    if dur <= 15:
        return "medium"
    return "long"


def pct(x: float | None) -> str:
    return "NA" if x is None else f"{x:.4f}"


def metric_from_summary(rec: dict[str, Any]) -> dict[str, Any]:
    summary = rec.get("summary", rec)
    calib = rec.get("pq_iou_calibration", {})
    short = rec.get("duration_breakdown", {}).get("short", {})
    return {
        "query_count": summary.get("query_count"),
        "iou05_top100": summary.get("GT_video_oracle_IoU@0.5_top100"),
        "iou07_top100": summary.get("GT_video_oracle_IoU@0.7_top100"),
        "iou07_top50": summary.get("GT_video_oracle_IoU@0.7_top50"),
        "short_iou07_top100": short.get("GT_video_oracle_IoU@0.7_top100"),
        "best_iou_span_top100_rate": summary.get("best_iou_span_top100_rate"),
        "best_iou_span_top50_rate": summary.get("best_iou_span_top50_rate"),
        "pq_spearman": calib.get("spearman"),
        "auc_iou07": calib.get("auc_iou07"),
    }


def load_train_rows() -> dict[int, dict[str, Any]]:
    rows = load_jsonl(DATA / "tvr_train_release.jsonl")
    return {int(r["desc_id"]): r for r in rows}


def load_splits() -> dict[str, list[int]]:
    base = ROOT / "c12_1_schema_and_split/splits"
    return {
        name: [int(x) for x in (base / f"{name}_desc_ids.txt").read_text(encoding="utf-8").splitlines() if x.strip()]
        for name in ["train_fit", "calib_select", "calib_holdout", "pseudo_official_holdout"]
    }


def load_first_stage_holdout() -> dict[int, dict[str, Any]]:
    path = ROOT / "results/c12_feature_cache/first_stage_calib_holdout_top128.pkl"
    if not path.exists():
        return {}
    with path.open("rb") as f:
        return pickle.load(f)


def pick_first(ids: list[int], rows: dict[int, dict[str, Any]], pred) -> int | None:
    for did in ids:
        row = rows.get(int(did))
        if row and pred(row):
            return int(did)
    return None


def sample_row(did: int | None, rows: dict[int, dict[str, Any]], source: str, note: str = "") -> dict[str, Any]:
    if did is None:
        return {"desc_id": None, "source": source, "coverage": False, "note": note}
    row = rows[int(did)]
    return {
        "desc_id": int(did),
        "source": source,
        "coverage": True,
        "query_type": row.get("type", "unknown"),
        "duration_bucket": duration_bucket(row),
        "moment_duration": float(row["ts"][1]) - float(row["ts"][0]),
        "video": row.get("vid_name"),
        "note": note,
    }


def build_sample_audit(rows: dict[int, dict[str, Any]], splits: dict[str, list[int]], first_stage: dict[int, dict[str, Any]]) -> dict[str, Any]:
    holdout = splits["calib_holdout"]
    train_fit = splits["train_fit"]
    samples: dict[str, Any] = {}
    for qt in ["v", "t", "vt"]:
        did = pick_first(train_fit + holdout, rows, lambda r, qt=qt: r.get("type") == qt)
        samples[f"query_type_{qt}"] = sample_row(did, rows, "train_release", "query type coverage")
    for db in ["short", "medium", "long"]:
        did = pick_first(train_fit + holdout, rows, lambda r, db=db: duration_bucket(r) == db)
        samples[f"duration_{db}"] = sample_row(did, rows, "train_release", "moment duration coverage")

    top1_correct = [d for d, fs in first_stage.items() if fs.get("top1_correct") is True]
    top1_wrong = [d for d, fs in first_stage.items() if fs.get("top1_correct") is False]
    miss_top100 = [d for d, fs in first_stage.items() if isinstance(fs.get("rank"), int) and int(fs["rank"]) > 100]
    samples["b6_top1_correct_proxy"] = sample_row(top1_correct[0] if top1_correct else None, rows, "first_stage_calib_holdout_top128", "CONQUER/B6 top1_correct proxy")
    samples["b6_top1_wrong_proxy"] = sample_row(top1_wrong[0] if top1_wrong else None, rows, "first_stage_calib_holdout_top128", "CONQUER/B6 top1_wrong proxy")
    samples["positive_not_in_b6_top100"] = sample_row(
        miss_top100[0] if miss_top100 else None,
        rows,
        "first_stage_calib_holdout_top128",
        "blocked_by_reference_top100_replay; C12-4R replay has 100% train-only GT-video top100 coverage",
    )

    failures = read_json(ROOT / "c12_5v_conquer_hidden_ranker/C12_5V_D_FAILURE_CASES.json", {})
    high_low = failures.get("V1_hidden_boundary", {}).get("high_score_low_iou", [])
    low_high = failures.get("V1_hidden_boundary", {}).get("low_score_high_iou", [])
    samples["c12_span_ranker_high_score_low_iou"] = sample_row(
        int(high_low[0]["desc_id"]) if high_low else None,
        rows,
        "C12_5V_D_FAILURE_CASES",
        "hidden ranker high-score low-IoU failure",
    )
    samples["c12_span_ranker_low_score_high_iou"] = sample_row(
        int(low_high[0]["desc_id"]) if low_high else None,
        rows,
        "C12_5V_D_FAILURE_CASES",
        "hidden ranker low-score high-IoU failure",
    )
    native = read_json(ROOT / "c12_4_native_retriever/C12_4_RETRIEVER_DECISION.json", {})
    samples["native_retriever_missed_top100_proxy"] = {
        "coverage": True,
        "source": "C12_4_RETRIEVER_DECISION",
        "note": "aggregate proxy for positive not in native top100",
        "native_best_GT_video_in_top100_rate": native.get("best_calib_holdout_metrics", {}).get("GT_video_in_top100_rate"),
        "native_best_median_rank": native.get("best_calib_holdout_metrics", {}).get("GT_video_median_rank"),
    }

    qtypes = defaultdict(int)
    durs = defaultdict(int)
    for sample in samples.values():
        if sample.get("coverage") and sample.get("query_type"):
            qtypes[sample["query_type"]] += 1
        if sample.get("coverage") and sample.get("duration_bucket"):
            durs[sample["duration_bucket"]] += 1
    return {
        "stage": "C13-1-sample",
        "official_val_used": False,
        "sample_policy": "train-release splits only; pseudo_official_holdout is not used for selection",
        "sample_count": len(samples),
        "coverage_by_query_type": dict(sorted(qtypes.items())),
        "coverage_by_duration": dict(sorted(durs.items())),
        "samples": samples,
    }


def build_cost_estimate(rows: dict[int, dict[str, Any]], splits: dict[str, list[int]]) -> dict[str, Any]:
    video_meta = read_json(DATA / "tvr_video2dur_idx.json", {})
    train_videos = video_meta.get("train", {})
    durations = [float(v[0]) for v in train_videos.values() if isinstance(v, list) and v]
    total_hours = sum(durations) / 3600.0 if durations else None
    total_queries = len(rows)
    sample_queries = 96
    sample_videos = 64
    per_token = {
        "sentence_transformer_text_768": 768,
        "deberta_roberta_text_768": 768,
        "clip_siglip_visual_768_to_1152": "768-1152",
        "videomae_internvideo_motion_768_to_1408": "768-1408",
        "event_multiscale_tokens": "N x 768/1024",
    }
    return {
        "stage": "C13-1",
        "official_val_used": False,
        "dataset_scope": {
            "train_release_queries": total_queries,
            "train_fit_queries": len(splits["train_fit"]),
            "calib_select_queries": len(splits["calib_select"]),
            "calib_holdout_queries": len(splits["calib_holdout"]),
            "pseudo_official_holdout_used_for_selection": False,
            "train_videos": len(train_videos),
            "train_video_hours_estimate": total_hours,
        },
        "pilot_scope": {
            "sample_queries": sample_queries,
            "sample_videos": sample_videos,
            "purpose": "feasibility and signal audit only; no full extraction",
        },
        "feature_dim": per_token,
        "storage_estimate": {
            "text_query_sentence_embeddings_fp16_all_train": f"{total_queries * 768 * 2 / 1024**2:.1f} MiB",
            "subtitle_sentence_embeddings_fp16_order": "depends on subtitle sentence segmentation; expected low single-digit GiB for TVR train",
            "visual_clip_tokens_fp16_full_train": "tens of GiB if dense frame/clip tokens are kept; small pilot <2 GiB",
            "event_tokens_fp16_full_train": "smaller than dense frame tokens if pooled to 8-32 events/video",
        },
        "extraction_time_estimate": {
            "text_subtitle_pilot_gpu": "minutes to <1 hour",
            "visual_clip_siglip_pilot_gpu": "1-4 GPU hours depending frame rate",
            "motion_model_pilot_gpu": "4-12 GPU hours; higher if decoding is slow",
            "full_train_visual_or_motion": "multi-day GPU job; should wait for pilot signal",
        },
        "gpu_cpu_cost": {
            "text": "1 GPU optional; CPU possible but slower",
            "visual": "GPU recommended; CPU not practical for video/frame models",
            "multiscale_event": "GPU for raw extraction, CPU ok for pooling/indexing",
        },
        "timestamp_alignment_feasibility": {
            "existing_tv_feature_stride": "current TVR features use 1.5s style clip alignment",
            "new_feature_alignment": "feasible if extraction records frame timestamp, clip center, and video duration normalization",
            "risk": "medium: subtitle sentence boundaries and video frame sampling must be canonicalized",
        },
        "vr_recall_sample_feasibility": "yes: use train_fit/calib_select/calib_holdout only, with C7-B6/CONQUER as teacher or hard-negative source",
        "span_ranking_sample_feasibility": "yes: reuse C12 M1000 generated pools and C12-5V failure IDs for pilot-only scoring",
        "risk_level": {
            "text_subtitle": "low_to_medium",
            "visual": "medium_to_high",
            "multiscale_event": "medium",
        },
    }


def main() -> None:
    rows = load_train_rows()
    splits = load_splits()
    first_stage = load_first_stage_holdout()

    c12_4 = read_json(ROOT / "c12_4_native_retriever/C12_4_RETRIEVER_DECISION.json", {})
    c12_4r = read_json(ROOT / "c12_4r_retriever_repair/C12_4R_RETRIEVER_REPAIR_DECISION.json", {})
    c12_5 = read_json(ROOT / "c12_5_native_span_generation/C12_5_LOCALIZER_DECISION.json", {})
    c12_5t = read_json(ROOT / "c12_5t_best_span_promotion/C12_5T_DECISION.json", {})
    c12_5u = read_json(ROOT / "c12_5u_topk_calibration_repair/C12_5U_DECISION.json", {})
    c12_5v = read_json(ROOT / "c12_5v_conquer_hidden_ranker/C12_5V_DECISION.json", {})
    c12_5v_schema = read_json(ROOT / "c12_5v_conquer_hidden_ranker/C12_5V_A_HIDDEN_FEATURE_SCHEMA.json", {})
    c12_5v_train = read_json(ROOT / "c12_5v_conquer_hidden_ranker/C12_5V_C_HIDDEN_RANKER_TRAINING_RESULTS.json", {})
    c12_5v_results = read_json(ROOT / "c12_5v_conquer_hidden_ranker/C12_5V_D_RESULTS.json", {})
    c12_5t_res = read_json(ROOT / "c12_5t_best_span_promotion/C12_5T_E_RESULTS.json", {})

    evaluator_hash = sha256_file(ROOT / "standalone_eval/eval.py")
    nms_hash = sha256_file(ROOT / "utils/inference_utils.py")
    root_markers = [p.name for p in [ROOT / "OFFICIAL_VAL_AUTHORIZED", ROOT / "C9_OFFICIAL_VAL_AUTHORIZED"] if p.exists()]

    c12_final = {
        "stage": "C13-0",
        "status": FINAL_C12_STATUS,
        "c12_span_ranker_line_stopped": True,
        "allow_enter_c12_6": False,
        "create_c12_official_candidate": False,
        "current_promoted_system": PROMOTED,
        "official_val_used": False,
        "evaluator_modified": False,
        "nms_modified": False,
        "root_official_authorization_markers_present": root_markers,
        "evaluator_sha256": evaluator_hash,
        "nms_sha256": nms_hash,
        "evidence": {
            "c12_independent_native_retriever": {
                "status": c12_4.get("status"),
                "best_variant": c12_4.get("best_variant"),
                "GT_video_in_top100_rate": c12_4.get("best_calib_holdout_metrics", {}).get("GT_video_in_top100_rate"),
                "reference_CONQUER_top100": c12_4.get("reference_calib_holdout_metrics", {}).get("GT_video_in_top100_rate"),
            },
            "c12_4r_conquer_warm_replay": {
                "status": c12_4r.get("status"),
                "best_repaired_retriever_variant": c12_4r.get("best_repaired_retriever_variant"),
                "learned_delta_repair_effective": c12_4r.get("learned_delta_repair_effective"),
                "GT_video_in_top100_rate": c12_4r.get("best_metrics", {}).get("GT_video_in_top100_rate"),
            },
            "c12_m1000_span_generation": {
                "status": c12_5.get("status"),
                "selected_variant": c12_5.get("selected_variant"),
                "holdout_iou07_top100": c12_5.get("reason"),
                "oracle_headroom_from_t2_generated_m1000_iou07": c12_5t.get("generated_M1000_iou07"),
            },
            "c12_top100_span_ranking_partial_repair": {
                "best_t_variant": c12_5t.get("best_ranker_variant"),
                "iou07_top100": c12_5t.get("best_iou07_top100"),
                "iou07_top50": c12_5t.get("best_iou07_top50"),
            },
            "c12_calibration_unreliable": {
                "t2_pq_spearman": c12_5t.get("pq_iou_spearman"),
                "u2_pq_spearman": c12_5u.get("pq_spearman"),
                "v1_pq_spearman": c12_5v.get("pq_spearman"),
            },
            "c12_hidden_ranker_regression": {
                "hidden_export_available": c12_5v_schema.get("status") == "C12_CONQUER_HIDDEN_AVAILABLE",
                "shared_feature_count": c12_5v_schema.get("feature_count"),
                "best_variant": c12_5v.get("best_variant"),
                "iou07_top100": c12_5v.get("best_iou07_top100"),
                "iou07_top50": c12_5v.get("best_iou07_top50"),
                "status": c12_5v.get("status"),
            },
        },
    }
    write_json(OUT0 / "C13_0_C12_FINAL_STATE.json", c12_final)

    write_text(OUT0 / "C13_0_C12_FINAL_STATE.md", f"""# C13-0 C12 Final State

Final status: `{FINAL_C12_STATUS}`

Current promoted system: `{PROMOTED}`

Required state facts:

1. C12 independent native retriever is weak: best C12-4 holdout GT-video top100 is {pct(c12_4.get('best_calib_holdout_metrics', {}).get('GT_video_in_top100_rate'))}, while CONQUER reference is {pct(c12_4.get('reference_calib_holdout_metrics', {}).get('GT_video_in_top100_rate'))}.
2. C12-4R uses CONQUER-warm / zero-delta replay bridge: status `{c12_4r.get('status')}`, best repaired retriever `{c12_4r.get('best_repaired_retriever_variant')}`.
3. C12 M1000 span generation has oracle headroom: T2 generated M1000 IoU@0.7 is {pct(c12_5t.get('generated_M1000_iou07'))}.
4. C12 top100 span ranking was partially repaired by T2: IoU@0.7 top100 {pct(c12_5t.get('best_iou07_top100'))}, top50 {pct(c12_5t.get('best_iou07_top50'))}.
5. C12 calibration / PQ-IoU correlation remains unreliable: T2 Spearman {pct(c12_5t.get('pq_iou_spearman'))}, U2 Spearman {pct(c12_5u.get('pq_spearman'))}, V1 Spearman {pct(c12_5v.get('pq_spearman'))}.
6. C12 hidden ranker regresses: V1 IoU@0.7 top100 {pct(c12_5v.get('best_iou07_top100'))}, top50 {pct(c12_5v.get('best_iou07_top50'))}.
7. C12-6 is not allowed.
8. Current promoted system remains `{PROMOTED}`.

official_val_used = false
evaluator_modified = false
nms_modified = false
""")
    write_text(OUT0 / "C13_0_NO_C12_6_DECISION.md", f"""# C13-0 No C12-6 Decision

Decision: do not enter C12-6 and do not create a C12 official candidate.

Reason: C12-5T partially recovers top100 ranking, but all score-calibration evidence remains unreliable and C12-5V hidden features regress. The span-ranker line should stop rather than receive a C12-5W patch.

Status: `{FINAL_C12_STATUS}`

official_val_used = false
""")
    write_text(OUT0 / "C13_0_PROMOTED_SYSTEM_STATUS.md", f"""# C13-0 Promoted System Status

current_promoted_system = `{PROMOTED}`

C12 did not supersede the promoted system. C7-B6 / CONQUER may be used as baseline, teacher, and hard-negative source for C13, but C7-B6 fixed prediction pools must not be copied as final candidates.

official_val_used = false
""")

    sample = build_sample_audit(rows, splits, first_stage)
    cost = build_cost_estimate(rows, splits)
    write_json(OUT1 / "C13_1_EXTRACTION_COST_ESTIMATE.json", cost)
    write_json(OUT1 / "C13_1_SAMPLE_COVERAGE_AUDIT.json", sample)

    feature_decision = {
        "stage": "C13-1",
        "status": "C13_STRONG_MULTISCALE_FEATURE_PROMISING",
        "strong_text_subtitle_worth_doing": True,
        "strong_visual_worth_doing": True,
        "multiscale_event_worth_doing": True,
        "full_extraction_now": False,
        "pilot_first": True,
        "sample_coverage_audit": "C13_1_SAMPLE_COVERAGE_AUDIT.json",
        "official_val_used": False,
        "reason": "Current TVR features are insufficient for independent retrieval and reliable span scoring, while C12 M1000 has large oracle headroom. A small pilot of stronger text/subtitle, visual, and multiscale event tokens is feasible and lower risk than another C12 ranker patch.",
    }
    write_json(OUT1 / "C13_1_FEATURE_DECISION.json", feature_decision)

    write_text(OUT1 / "C13_1_FEATURE_CANDIDATES.md", """# C13-1 Feature Candidates

Status: `C13_STRONG_MULTISCALE_FEATURE_PROMISING`

Candidate feature families:

| family | candidates | feature dim | timestamp alignment | risk |
|---|---|---:|---|---|
| Text / subtitle | sentence-transformer style embedding, DeBERTa/RoBERTa/BERT re-encoding, token-level query-subtitle similarity | 768 typical | sentence offsets and clip centers must be preserved | low_to_medium |
| Visual | CLIP / EVA-CLIP / SigLIP image-video clip features, DINOv2 frame features | 768-1152 typical | feasible through sampled frame timestamps | medium_to_high |
| Motion | VideoMAE / InternVideo style motion tokens | 768-1408 typical | feasible with fixed clip windows | high |
| Multi-scale / event | clip tokens, event pooled tokens, inside/outside span context tokens | N x 768/1024 | best fit for boundary/localizer redesign | medium |

Small-sample audit requirements are represented in `C13_1_SAMPLE_COVERAGE_AUDIT.json`. The only uncovered natural bucket is `positive_not_in_b6_top100`, because the C12 CONQUER/B6 replay cache has 100% train-only GT-video top100 coverage; use native C12 missed-top100 aggregate as a hard-case proxy.

official_val_used = false
""")
    write_text(OUT1 / "C13_1_TEXT_SUBTITLE_FEATURE_PLAN.md", """# C13-1 Text Subtitle Feature Plan

Decision: strong text/subtitle features are worth a pilot.

Plan:

1. Re-encode queries and subtitle sentences with a stronger text encoder.
2. Build sentence-level and token-level query-subtitle similarity features.
3. Preserve subtitle timestamps, sentence ids, and clip-center alignment.
4. Evaluate only on train_fit/calib_select/calib_holdout sample buckets first.

Expected value: highest for t/vt queries and for failures where current score assigns high confidence to wrong boundary spans.

Risk level: low_to_medium.

official_val_used = false
""")
    write_text(OUT1 / "C13_1_VISUAL_FEATURE_PLAN.md", """# C13-1 Visual Feature Plan

Decision: strong visual features are worth a pilot, but not full extraction yet.

Plan:

1. Extract CLIP/SigLIP/EVA-CLIP style frame or short-clip embeddings for a small video sample.
2. Optionally compare DINOv2 frame descriptors when video-language models are unavailable.
3. Record frame timestamp, clip window, video id, and normalized time index.
4. Compare VR recall sample and span-ranking sample against C12 current visual bridge.

Expected value: strongest for v queries and for B6 top1 wrong cases.

Risk level: medium_to_high because decoding and dense storage dominate cost.

official_val_used = false
""")
    write_text(OUT1 / "C13_1_MULTISCALE_EVENT_FEATURE_PLAN.md", """# C13-1 Multiscale Event Feature Plan

Decision: multiscale/event features are the most promising pivot target.

Plan:

1. Pool frame/clip tokens into 8-32 event tokens per video.
2. Build span inside/outside context features over event tokens.
3. Keep both clip-level and event-level representations so boundary models can attend across scales.
4. Use C12 M1000 failure cases to test whether event tokens move high-IoU spans upward.

Expected value: directly targets the C12 failure mode where oracle span exists in M1000 but rank/calibration does not find it.

Risk level: medium.

official_val_used = false
""")

    baseline_decision = {
        "stage": "C13-2",
        "status": "C13_NATIVE_WITH_STRONG_FEATURE_FIRST",
        "prem_reproducible_now": False,
        "prem_module_borrow": True,
        "mavr_borrow": True,
        "eventformer_borrow": True,
        "minute_borrow": True,
        "boundary_architecture_redesign_needed": True,
        "official_val_used": False,
        "reason": "PREM public repo is not runnable from available code; MA-VR/EventFormer/MINUTE ideas are useful, but the next controllable step is native strong-feature pilot before retraining or redesigning the localizer.",
        "sources": {
            "prem_repo": PREM_REPO,
            "prem_repo_head_checked": "0cc6bd8fceae88cb89e79f71d2b22a1fb72640da",
            "prem_readme_observation": "README says: The code is coming soon...",
            "prem_paper": PREM_PAPER,
            "eventformer_reference": "no verified runnable local/public repo found in this audit; borrow event-level modeling idea only",
            "mavr_reference": MAVR_REF,
            "minute_paper": MINUTE_PAPER,
            "local_mavr_bridge": "c9_2_mavr_bridge/C9_2_SMOKE_RESULTS.md",
        },
    }
    write_json(OUT2 / "C13_2_BASELINE_DECISION.json", baseline_decision)

    write_text(OUT2 / "C13_2_PREM_FEASIBILITY.md", f"""# C13-2 PREM Feasibility

Status: not directly reproducible now.

Evidence:

- Local repository search found no PREM checkout.
- Public PREM repo: {PREM_REPO}
- Verified repo HEAD: `0cc6bd8fceae88cb89e79f71d2b22a1fb72640da`.
- The public repository README currently says: `The code is coming soon...`, so it is a README-level placeholder rather than runnable training/evaluation code.
- Paper direction is relevant: partial relevance enhancement, modality-specific pooling for VR, and focus-then-fuse localizer.

Decision: do not choose `C13_PREM_REPRO_FIRST`. Borrow PREM module ideas only.

official_val_used = false
""")
    write_text(OUT2 / "C13_2_MAVR_FEASIBILITY.md", """# C13-2 MA-VR Feasibility

Status: borrowable.

Local evidence:

- `c9_2_mavr_bridge/C9_2_SMOKE_RESULTS.md` reports `C9_2_MAVR_BRIDGE_SMOKE_PASS`.
- Moment-aware score tensors and gradients were already smoke-tested locally.

Decision: reuse the idea as a module candidate, not as a full mainline baseline.

official_val_used = false
""")
    write_text(OUT2 / "C13_2_EVENTFORMER_FEASIBILITY.md", f"""# C13-2 EventFormer Feasibility

Status: borrowable event-level modeling idea only.

Evidence:

- Event-aware VCMR is aligned with C12 failures because C12 has span oracle headroom but weak ranking/calibration.
- No verified runnable local/public EventFormer repository was found in this audit.

Decision: use event-level pooled tokens and event-query interaction as design input for C13 feature pilot.

official_val_used = false
""")
    write_text(OUT2 / "C13_2_MINUTE_FEASIBILITY.md", f"""# C13-2 MINUTE Feasibility

Status: borrowable training objective idea.

Evidence:

- MINUTE targets moment prediction bias and shared normalization mismatch across multiple retrieved videos.
- Reference: {MINUTE_PAPER}
- C12 rankers still over-trust high-score low-IoU spans, so shared-normalization-style multi-video ranking is relevant.

Decision: borrow shared normalization and multimodal clue mining ideas after strong feature pilot.

official_val_used = false
""")
    write_text(OUT2 / "C13_2_BOUNDARY_ARCHITECTURE_OPTIONS.md", """# C13-2 Boundary Architecture Options

Need: yes, but after strong feature pilot.

Options:

1. Focus-then-fuse localizer with modality gates over subtitle, visual, and event tokens.
2. Event-token boundary decoder with inside/outside contrast and duration-aware priors.
3. Shared-normalized multi-video span ranker trained across retrieved videos.
4. Two-stage design: strong VR candidate generator, then event-aware boundary/localizer.

Do not continue with C12-5W ranker patches. C12-5V showed that adding CONQUER hidden scalar summaries to the same ranker family regresses.

official_val_used = false
""")

    next_decision = {
        "stage": "C13-3",
        "selected_path": "Path B",
        "status": "C13_STRONG_FEATURE_EXTRACTION_THEN_NATIVE_RETRAIN",
        "c12_stopped": True,
        "current_promoted_system": PROMOTED,
        "strong_text_subtitle_worth_doing": True,
        "strong_visual_worth_doing": True,
        "multiscale_event_worth_doing": True,
        "prem_reproducible_now": False,
        "mavr_eventformer_minute_borrowable": True,
        "boundary_localizer_redesign_needed": True,
        "official_val_used": False,
        "evaluator_modified": False,
        "nms_modified": False,
        "reason": "Strong feature pilot is feasible and directly targets the observed C12 bottleneck; PREM cannot be run directly, and a full architecture rewrite should wait for pilot signal.",
    }
    write_json(OUT3 / "C13_3_NEXT_STEP_DECISION.json", next_decision)
    write_text(OUT3 / "C13_3_NEXT_STEP_DECISION.md", f"""# C13-3 Next Step Decision

Selected mainline: `Path B`

Status: `C13_STRONG_FEATURE_EXTRACTION_THEN_NATIVE_RETRAIN`

Rationale:

- C12 is stopped: `{FINAL_C12_STATUS}`.
- Strong text/subtitle features are worth a pilot.
- Strong visual features are worth a pilot, with higher extraction/storage risk.
- Multiscale/event features are the highest-value pivot because C12 M1000 has oracle headroom but rank/calibration fails.
- PREM is not directly reproducible from available public code, so Path A is not selected.
- MA-VR/EventFormer/MINUTE are borrowable module/objective ideas.
- Boundary/localizer redesign is likely needed, but should be guided by strong-feature pilot results.

official_val_used = false
evaluator_modified = false
nms_modified = false
""")

    manifest = {
        "stage": "C13",
        "status": "C13_STRONG_FEATURE_EXTRACTION_THEN_NATIVE_RETRAIN",
        "official_val_used": False,
        "evaluator_modified": False,
        "nms_modified": False,
        "artifact_hashes": {},
    }
    for out in [OUT0, OUT1, OUT2, OUT3]:
        for path in sorted(out.glob("C13_*")):
            if path.is_file():
                manifest["artifact_hashes"][str(path.relative_to(ROOT))] = sha256_file(path)
    write_json(OUT3 / "C13_3_MANIFEST.json", manifest)


if __name__ == "__main__":
    main()
