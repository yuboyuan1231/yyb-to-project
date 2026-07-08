#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import time
from pathlib import Path
from typing import Any

from blueprint_e2e_v2.data.feature_registry import TMP_ROOT
from blueprint_e2e_v2.engine.c28e_late_interaction import run_c28e_retriever_replay
from blueprint_e2e_v2.engine.evaluate import run_evaluation
from blueprint_e2e_v2.engine.official_safe_eval_wrapper import official_safety_manifest
from blueprint_e2e_v2.engine.train import full_selection_score, run_full_training
from blueprint_e2e_v2.utils.io import load_json, write_json, write_text


ROOT = Path(__file__).resolve().parent
REPORT = ROOT / "blueprint_e2e_v2/reports"


def sh(cmd: str) -> str:
    return subprocess.check_output(cmd, cwd=ROOT, shell=True, stderr=subprocess.STDOUT, text=True).strip()


def load_config(path: Path) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        key, val = line.split(":", 1)
        val = val.strip()
        if val.lower() in {"true", "false"}:
            out[key.strip()] = val.lower() == "true"
        else:
            try:
                out[key.strip()] = int(val)
            except ValueError:
                try:
                    out[key.strip()] = float(val)
                except ValueError:
                    out[key.strip()] = val
    return out


def merge_args(cfg: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    for key in [
        "epochs",
        "batch_size",
        "lr",
        "chunk_size",
        "max_queries",
        "max_videos",
        "candidate_topk_train",
        "candidate_topk_eval",
        "eval_candidate_k",
        "broad_topk_train",
        "broad_topk_eval",
        "late_topk",
        "late_soft_topk",
        "candidate_encode_chunk",
        "teacher_topk",
        "teacher_anchor_topk",
        "grad_accum_steps",
    ]:
        val = getattr(args, key, None)
        if val is not None:
            cfg[key] = val
    cfg["mode"] = args.mode
    cfg["seed"] = args.seed
    return cfg


def git_info() -> dict[str, Any]:
    return {
        "branch": sh("git branch --show-current"),
        "commit": sh("git rev-parse --short HEAD"),
        "commit_full": sh("git rev-parse HEAD"),
        "status_short": sh("git status --short").splitlines(),
    }


def selection_gate_decision(summary: dict[str, Any], cfg: dict[str, Any]) -> dict[str, Any]:
    checks = [
        ("VCMR_R@1_IoU0.7", ">=", float(cfg.get("full_gate_vcmr_r1_iou07", 3.0))),
        ("VCMR_R@5_IoU0.7", ">=", float(cfg.get("full_gate_vcmr_r5_iou07", 8.0))),
        ("VCMR_R@10_IoU0.7", ">=", float(cfg.get("full_gate_vcmr_r10_iou07", 12.0))),
        ("VR_R@100", ">=", float(cfg.get("full_gate_vr_r100", 80.0))),
        ("wrong_video_top1_rate", "<=", float(cfg.get("full_gate_wrong_video_top1_max", 95.0))),
        ("high_score_false_positive_rate", "<=", float(cfg.get("full_gate_high_score_false_positive_max", cfg.get("full_gate_wrong_video_top1_max", 95.0)))),
    ]
    details = []
    for metric, op, threshold in checks:
        value = float(summary.get(metric, summary.get("wrong_video_top1_rate", 100.0) if metric == "high_score_false_positive_rate" else 0.0))
        passed = value >= threshold if op == ">=" else value <= threshold
        details.append({"metric": metric, "op": op, "threshold": threshold, "value": value, "passed": bool(passed)})
    return {
        "passed": all(bool(x["passed"]) for x in details),
        "checks": details,
        "selection_score": full_selection_score(summary, cfg),
    }


def stage0(cfg: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    out = REPORT / "c28e_0_code_review"
    rec = {
        "stage": "C28E-0",
        "status": "C28E_CODE_REVIEW_READY",
        "goal": "fix blueprint-name/blueprint-mechanism gaps before running more experiments",
        "git": git_info(),
        "fixed_issues": [
            "clean PR base should be c28c-cleanroom-e2e-blueprint-v2, not main",
            "candidate_topk_train/eval/eval_candidate_k are separated from hard_negative_k",
            "candidate proposals use each candidate video's own duration",
            "candidate duration bank is sourced from video metadata, not only query rows",
            "late_interaction_enabled is a real clip-level scoring path, not a zero-weight config label",
            "two-stage dynamic retrieval uses pooled student broad topK then clip-level rerank topK",
            "teacher score schema is audited and rank-derived teacher logits are the default unless score use is explicitly enabled",
            "checkpoint selection uses calib_select, not train VR@100",
            "calib_holdout is final-report only and guarded by --allow_holdout_final",
            "full E2E resume appends prior training logs and restores best selection metadata",
            "full config exposes only full E2E loss weights; retriever-only distillation weights are not advertised as active",
            "full retriever loss includes duplicate-safe cross-query in-batch negatives in addition to candidate-set negatives",
            "ablation helper refuses to claim mutual-help evidence until toggled ablation results exist",
            "legacy pooled miner and retriever trainer are labeled diagnostic-only and are not C28E-3 owners",
            "selection gate requires R@1/R@5/R@10 IoU0.7, VR@100, wrong-video, and high-score false-positive checks",
            "train/eval reports include pooled/late/token/combined score scale audit and in-batch loss coupling",
            "explicit cuda:N device disables DataParallel so single-card runs do not touch other GPUs",
            "dynamic candidate refresh pre-encodes current student clip bank once before late rerank",
            "stage names use C28E consistently",
            "clip masks are carried through sequence banks, late interaction, and full-model batches",
            "late interaction retriever score replaces the localizer's retrieval sims before feedback/VCMR scoring",
            "raw candidate scoring uses candidate chunks so proposal count is preserved without a monolithic GPU tensor",
            "C28E stage 3 is full E2E training with compute_full_loss, not retriever-only replay training",
            "full training/evaluation candidate mining uses pooled broad recall then clip-level late rerank",
            "long videos are represented by duration-aware 64-bin resampling instead of first-96s truncation",
            "active moment prior is conditioned on each candidate video, not query-only",
            "candidate curriculum follows teacher-heavy to student-dynamic phases",
            "broad pooled recall is audited separately from late rerank recall",
        ],
        "not_run_until_review_passes": True,
        "config": cfg,
        **official_safety_manifest(),
    }
    write_json(out / "C28E_0_CODE_REVIEW_PROTOCOL.json", rec)
    write_text(
        out / "C28E_CODE_REVIEW_CHECKLIST.md",
        "# C28E Code Review Checklist\n\n"
        "- PR base is `c28c-cleanroom-e2e-blueprint-v2`.\n"
        "- No checkpoints, raw predictions, official outputs, or large caches are committed.\n"
        "- Holdout is not used for checkpoint/config selection.\n"
        "- `late_interaction_enabled` maps to real clip-level visual/subtitle/joint scoring.\n"
        "- Clip masks exclude padded release-feature positions from pooled, partial-relevance, late, and token scores.\n"
        "- Late retriever score is the retrieval evidence consumed by the localizer, feedback, and joint VCMR scoring.\n"
        "- C28E-3 runs `run_full_training` and `compute_full_loss`, not retriever-only distillation.\n"
        "- Full retriever optimization includes duplicate-safe in-batch video negatives as well as candidate-set negatives.\n"
        "- Full train/eval candidate mining reranks broad pooled candidates with clip late interaction.\n"
        "- Training loss scores raw candidate clips through the trainable video encoder.\n"
        "- Two-stage retrieval reports broad pooled recall separately from late rerank recall.\n"
        "- Teacher rankings are supervision/diagnostic only, never final static hard gates.\n"
        "- Long videos use duration-aware 64-bin resampling and GT span insertion uses the same grid.\n"
        "- ActiveMoment is candidate-video-conditioned.\n"
        "- Candidate scoring is chunked by candidate, preserving proposal count while controlling memory.\n"
        "- Candidate curriculum is explicit and audited per epoch.\n"
        "- `calib_select` is the only selection split.\n"
        "- Full E2E checkpoint selection uses `calib_select`; final holdout still requires `--allow_holdout_final`.\n"
        "- Ablation helpers must report pending evidence rather than positive contribution when toggled runs are missing.\n"
        "- Legacy pooled miner/retriever trainer code is diagnostic-only and not the C28E-3 training owner.\n"
        "- C28E-4 uses a multi-metric gate, not only `VCMR_R@1_IoU0.7`.\n"
        "- Train/eval reports include pooled/late/token/combined score scale audit and in-batch loss coupling.\n"
        "- Explicit `--device cuda:N` keeps training on one GPU and disables `DataParallel`.\n"
        "- Dynamic candidate refresh pre-encodes the current student clip bank once before late rerank.\n",
    )
    return rec


def stage1(cfg: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    out = REPORT / "c28e_1_diagnostics"
    if args.dry_run:
        rec = {"stage": "C28E-1", "status": "C28E_DIAGNOSTICS_DRY_RUN_READY", "splits": ["train_fit", str(cfg.get("selection_split", "calib_select"))], **official_safety_manifest()}
        write_json(out / "C28E_1_DIAGNOSTICS_PLAN.json", rec)
        return rec
    train = run_c28e_retriever_replay(cfg, out, "train_fit", device_arg=args.device, force=args.force)
    select = run_c28e_retriever_replay(cfg, out, str(cfg.get("selection_split", "calib_select")), device_arg=args.device, force=args.force)
    rec = {"stage": "C28E-1", "status": "C28E_DIAGNOSTICS_COMPLETE_NO_HOLDOUT_SELECTION", "train_fit": train, "selection": select, **official_safety_manifest()}
    write_json(out / "C28E_1_DIAGNOSTICS_DECISION.json", rec)
    return rec


def stage2(cfg: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    out = REPORT / "c28e_2_late_interaction_bank"
    rec = {
        "stage": "C28E-2",
        "status": "C28E_LATE_INTERACTION_BANK_CODE_READY",
        "bank_contract": {
            "visual_clip_bank": "[N_video, 64, hidden_dim] encoded from release visual sequence features",
            "subtitle_clip_bank": "[N_video, 64, hidden_dim] encoded from release subtitle sequence features",
            "joint_clip_bank": "[N_video, 64, hidden_dim] encoded by VideoSubtitleEncoder temporal path",
            "clip_masks": "[N_video, 64] visual/subtitle/joint masks exclude padded sequence positions",
            "storage_policy": "CPU float16 cache for review; chunked GPU scoring for train/eval",
        },
        "two_stage_contract": {
            "stage1": "current student pooled retriever broad_topK",
            "stage2": "clip-level late interaction reranks broad candidates",
            "static_teacher_hard_gate": False,
            "broad_recall_audit": True,
        },
        **official_safety_manifest(),
    }
    write_json(out / "C28E_2_LATE_INTERACTION_BANK_CONTRACT.json", rec)
    return rec


def stage3(cfg: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    out = REPORT / "c28e_3_training"
    if args.dry_run:
        rec = {
            "stage": "C28E-3",
            "status": "C28E_FULL_E2E_TRAINING_DRY_RUN_READY",
            "selection_split": cfg.get("selection_split"),
            "full_training": run_full_training(cfg, dry_run=True, force=args.force, resume=args.resume, device_arg=args.device),
            **official_safety_manifest(),
        }
        write_json(out / "C28E_3_FULL_E2E_TRAINING_DECISION.json", rec)
        return rec
    full = run_full_training(cfg, dry_run=False, force=args.force, resume=args.resume, device_arg=args.device)
    rec = {"stage": "C28E-3", "status": "C28E_FULL_E2E_TRAINING_COMPLETE", "full_training": full, **official_safety_manifest()}
    write_json(out / "C28E_3_FULL_E2E_TRAINING_DECISION.json", rec)
    return rec


def stage4(cfg: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    out = REPORT / "c28e_4_selection_gate"
    train_rec = load_json(REPORT / "c28e_3_training/C28E_3_FULL_E2E_TRAINING_DECISION.json", {})
    full_rec = train_rec.get("full_training", {})
    ckpt = (full_rec.get("best_checkpoint_manifest") or full_rec.get("checkpoint_manifest") or {}).get("path")
    if not ckpt:
        rec = {"stage": "C28E-4", "status": "C28E_SELECTION_GATE_BLOCKED_NO_BEST_CHECKPOINT", **official_safety_manifest()}
        write_json(out / "C28E_4_SELECTION_GATE_DECISION.json", rec)
        return rec
    eval_cfg = dict(cfg)
    eval_cfg["checkpoint_path"] = ckpt
    select = run_evaluation(eval_cfg, str(cfg.get("selection_split", "calib_select")), device_arg=args.device, force=False)
    summary = select.get("summary", {}).get("summary", {})
    gate = selection_gate_decision(summary, cfg)
    status = "C28E_FULL_SELECTION_READY_FOR_FINAL_HOLDOUT" if bool(gate["passed"]) else "C28E_FULL_NOT_READY_CONTINUE_REPAIR"
    rec = {"stage": "C28E-4", "status": status, "selection_gate": gate, "selection": select, "selection_summary": summary, "best_checkpoint": ckpt, **official_safety_manifest()}
    write_json(out / "C28E_4_SELECTION_GATE_DECISION.json", rec)
    return rec


def stage5(cfg: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    out = REPORT / "c28e_5_final_holdout"
    if not args.allow_holdout_final:
        rec = {"stage": "C28E-5", "status": "C28E_FINAL_HOLDOUT_BLOCKED_REQUIRES_ALLOW_HOLDOUT_FINAL", **official_safety_manifest()}
        write_json(out / "C28E_5_FINAL_HOLDOUT_DECISION.json", rec)
        return rec
    gate_rec = load_json(REPORT / "c28e_4_selection_gate/C28E_4_SELECTION_GATE_DECISION.json", {})
    if gate_rec.get("status") != "C28E_FULL_SELECTION_READY_FOR_FINAL_HOLDOUT":
        rec = {"stage": "C28E-5", "status": "C28E_FINAL_HOLDOUT_BLOCKED_SELECTION_GATE_NOT_READY", "selection_gate_status": gate_rec.get("status"), **official_safety_manifest()}
        write_json(out / "C28E_5_FINAL_HOLDOUT_DECISION.json", rec)
        return rec
    train_rec = load_json(REPORT / "c28e_3_training/C28E_3_FULL_E2E_TRAINING_DECISION.json", {})
    full_rec = train_rec.get("full_training", {})
    ckpt = (full_rec.get("best_checkpoint_manifest") or full_rec.get("checkpoint_manifest") or {}).get("path")
    if not ckpt:
        rec = {"stage": "C28E-5", "status": "C28E_FINAL_HOLDOUT_BLOCKED_NO_BEST_CHECKPOINT", **official_safety_manifest()}
        write_json(out / "C28E_5_FINAL_HOLDOUT_DECISION.json", rec)
        return rec
    eval_cfg = dict(cfg)
    eval_cfg["checkpoint_path"] = ckpt
    hold = run_evaluation(eval_cfg, str(cfg.get("final_report_split", "calib_holdout")), device_arg=args.device, force=False)
    status = "C28E_FULL_FINAL_HOLDOUT_REPORTED"
    rec = {"stage": "C28E-5", "status": status, "holdout": hold, "best_checkpoint": ckpt, **official_safety_manifest()}
    write_json(out / "C28E_5_FINAL_HOLDOUT_DECISION.json", rec)
    return rec


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", default="c28e_0", choices=["c28e_0", "c28e_1", "c28e_2", "c28e_3", "c28e_4", "c28e_5", "all"])
    parser.add_argument("--mode", default="code_review")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--config")
    parser.add_argument("--device")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--allow_holdout_final", action="store_true")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--chunk_size", type=int)
    parser.add_argument("--max_queries", type=int)
    parser.add_argument("--max_videos", type=int)
    parser.add_argument("--candidate_topk_train", type=int)
    parser.add_argument("--candidate_topk_eval", type=int)
    parser.add_argument("--eval_candidate_k", type=int)
    parser.add_argument("--broad_topk_train", type=int)
    parser.add_argument("--broad_topk_eval", type=int)
    parser.add_argument("--late_topk", type=int)
    parser.add_argument("--late_soft_topk", type=int)
    parser.add_argument("--candidate_encode_chunk", type=int)
    parser.add_argument("--teacher_topk", type=int)
    parser.add_argument("--teacher_anchor_topk", type=int)
    parser.add_argument("--grad_accum_steps", type=int)
    args = parser.parse_args()
    cfg_path = Path(args.config) if args.config else ROOT / "blueprint_e2e_v2/configs/c28e_code_review.yaml"
    cfg = merge_args(load_config(cfg_path), args)
    funcs = {"c28e_0": stage0, "c28e_1": stage1, "c28e_2": stage2, "c28e_3": stage3, "c28e_4": stage4, "c28e_5": stage5}
    order = list(funcs) if args.stage == "all" else [args.stage]
    started = time.time()
    results = {}
    for stage in order:
        results[stage] = funcs[stage](cfg, args)
        print(f"{stage}: {results[stage].get('status')}", flush=True)
    print("branch:", sh("git branch --show-current"), flush=True)
    print("commit:", sh("git rev-parse --short HEAD"), flush=True)
    print("elapsed_seconds:", round(time.time() - started, 2), flush=True)
    print("local_tmp_root:", TMP_ROOT, flush=True)


if __name__ == "__main__":
    main()
