#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np

from blueprint_e2e_v2.data.dynamic_candidate_miner import CandidateSet
from blueprint_e2e_v2.data.first_stage_reference import FirstStageReference
from blueprint_e2e_v2.data.proposal_dataset import MultiSpanProposalDataset
from blueprint_e2e_v2.engine.official_safe_eval_wrapper import official_safety_manifest
from blueprint_e2e_v2.engine.retriever_repair import run_retriever_replay_audit, train_retriever_distillation
from blueprint_e2e_v2.engine.train import prepare_banks
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


def merge_args(cfg: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    for key in [
        "epochs",
        "retriever_epochs",
        "batch_size",
        "lr",
        "chunk_size",
        "max_queries",
        "max_videos",
        "dynamic_topk",
        "candidate_topk_train",
        "candidate_topk_eval",
        "eval_candidate_k",
        "hard_negative_k",
        "loss_negative_sample_k",
        "teacher_warm_topk",
        "max_spans_per_video",
        "grad_accum_steps",
        "moment_max_score_weight",
    ]:
        v = getattr(args, key, None)
        if v is not None:
            cfg[key] = v
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


def stage0(cfg: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    out = REPORT / "c28d_0_protocol"
    status_lines = git_info()["status_short"]
    historical_untracked = [x for x in status_lines if x.startswith("?? c") or "official" in x.lower()]
    rec = {
        "stage": "C28D-0",
        "status": "C28D_PROTOCOL_READY",
        "git": git_info(),
        "pr_scope_audit": {
            "historical_untracked_count": len(historical_untracked),
            "historical_untracked_examples": historical_untracked[:80],
            "clean_branch_needed_before_pr": bool(historical_untracked),
            "intended_c28d_scope": ["blueprint_e2e_v2/", "run_c28d_retriever_distillation_candidate_repair.py"],
        },
        "policy": {
            "do_not_delete_c28c": True,
            "do_not_run_official": True,
            "do_not_read_official_prediction_pool": True,
            "do_not_use_pseudo_official_holdout_selection": True,
            "do_not_modify_evaluator_nms": True,
        },
        "config": cfg,
        **official_safety_manifest(),
    }
    write_json(out / "C28D_0_PROTOCOL.json", rec)
    write_text(
        out / "C28D_0_PROTOCOL.md",
        "# C28D-0 Protocol\n\n"
        "Status: `C28D_PROTOCOL_READY`.\n\n"
        "C28D repairs retriever replay, teacher distillation, candidate cutoff, and proposal duration before full mutual VCMR rerun.\n",
    )
    return rec


def stage1(cfg: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    out = REPORT / "c28d_1_retriever_replay"
    return run_retriever_replay_audit(cfg, out, split=args.split, device_arg=args.device, force=args.force)


def stage2(cfg: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    out = REPORT / "c28d_2_candidate_proposal_repair"
    banks = prepare_banks(cfg, force=args.force)
    video_ids = [str(x) for x in banks["visual_np"]["video_ids"].tolist()]
    banks["video_bank"].video_ids = video_ids
    banks["video_bank"].video_to_idx = {v: i for i, v in enumerate(video_ids)}
    banks["video_bank"].idx_to_video = {i: v for v, i in banks["video_bank"].video_to_idx.items()}
    rows = banks["split_manager"].records("train_fit", max_queries=min(int(cfg.get("max_queries", 0) or 0) or 128, 128))
    ref = FirstStageReference()
    teacher = ref.bulk_top_indices([int(r["desc_id"]) for r in rows], banks["video_bank"].video_to_idx, k=int(cfg.get("candidate_topk_train", 200)))
    ref.close()
    candidates: dict[int, CandidateSet] = {}
    for row in rows:
        qid = int(row["desc_id"])
        gt_idx = int(banks["video_bank"].video_to_idx[str(row["vid_name"])])
        vids = list(teacher.get(qid, []))
        if gt_idx not in vids:
            vids = vids[: max(0, int(cfg.get("candidate_topk_train", 200)) - 1)] + [gt_idx]
        candidates[qid] = CandidateSet(qid, vids, [0.0] * len(vids), gt_idx, gt_idx not in teacher.get(qid, []))
    dataset = MultiSpanProposalDataset(
        "train_fit",
        banks["split_manager"],
        banks["query_bank"],
        banks["video_bank"],
        banks["subtitle_bank"],
        candidates,
        max_queries=len(rows),
        max_candidates=int(cfg.get("candidate_topk_train", 200)),
        max_spans_per_video=int(cfg.get("max_spans_per_video", 64)),
        insert_gt_for_training=True,
    )
    checked = 0
    differing_duration_candidates = 0
    violations = []
    for i in range(min(len(dataset), 32)):
        item = dataset[i]
        for ci, vid in enumerate(item["video_ids"]):
            checked += 1
            cand_duration = float(banks["video_bank"].durations.get(str(vid), item["duration"]))
            if abs(cand_duration - float(item["duration"])) > 1e-3:
                differing_duration_candidates += 1
            valid = item["spans_sec"][ci][item["span_mask"][ci]]
            if valid.size and (float(valid[:, 1].max()) > cand_duration + 1e-3 or float(valid[:, 0].min()) < -1e-6):
                violations.append({
                    "query_id": int(item["query_id"]),
                    "candidate_video_id": str(vid),
                    "candidate_duration": cand_duration,
                    "max_span_end": float(valid[:, 1].max()),
                })
    rec = {
        "stage": "C28D-2",
        "status": "C28D_CANDIDATE_PROPOSAL_REPAIR_READY" if not violations else "C28D_CANDIDATE_PROPOSAL_REPAIR_HAS_VIOLATIONS",
        "candidate_topk_train": int(cfg.get("candidate_topk_train", 200)),
        "candidate_topk_eval": int(cfg.get("candidate_topk_eval", cfg.get("dynamic_topk", 200))),
        "eval_candidate_k": int(cfg.get("eval_candidate_k", cfg.get("candidate_topk_eval", cfg.get("dynamic_topk", 200)))),
        "hard_negative_k": int(cfg.get("hard_negative_k", 64)),
        "hard_negative_k_used_to_truncate_eval_candidates": False,
        "dataset_audit": dataset.audit(),
        "proposal_duration_audit": {
            "checked_candidate_videos": checked,
            "candidate_duration_differs_from_query_duration_count": differing_duration_candidates,
            "violations": violations[:50],
        },
        **official_safety_manifest(),
    }
    write_json(out / "C28D_2_CANDIDATE_CUTOFF_AUDIT.json", rec)
    write_json(out / "C28D_2_PROPOSAL_DURATION_AUDIT.json", rec["proposal_duration_audit"])
    write_text(out / "C28D_2_CANDIDATE_PROPOSAL_DECISION.md", f"# C28D-2 Candidate/Proposal Decision\n\nStatus: `{rec['status']}`.")
    return rec


def stage3(cfg: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    out = REPORT / "c28d_3_retriever_distillation"
    if args.dry_run:
        rec = {"stage": "C28D-3", "status": "C28D_RETRIEVER_DISTILLATION_DRY_RUN_READY", "config": cfg}
        write_json(out / "C28D_3_RETRIEVER_TRAINING_DECISION.json", rec)
        return rec
    rec = train_retriever_distillation(cfg, out, device_arg=args.device, force=args.force, resume=args.resume)
    write_json(out / "C28D_3_RETRIEVER_TRAINING_DECISION.json", rec)
    return rec


def stage4(cfg: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    out = REPORT / "c28d_4_retriever_gate"
    train_rec = load_json(REPORT / "c28d_3_retriever_distillation/C28D_3_RETRIEVER_TRAINING_DECISION.json", {})
    ckpt = train_rec.get("best_checkpoint_path") or train_rec.get("checkpoint_path")
    replay = run_retriever_replay_audit(cfg, out, split="calib_holdout", device_arg=args.device, checkpoint=ckpt, force=False)
    teacher_vr100 = float(replay.get("teacher", {}).get("VR@100") or 0.0)
    student_vr100 = float(replay.get("student_c28c_checkpoint", {}).get("VR@100") or 0.0)
    absolute_gate = float(cfg.get("retriever_gate_absolute_vr100", 80.0))
    relative_gate = float(cfg.get("retriever_gate_teacher_relative", 0.85))
    target = absolute_gate if teacher_vr100 >= 85.0 else teacher_vr100 * relative_gate
    status = "C28D_RETRIEVER_REPAIRED_READY_FOR_FULL_MODEL" if student_vr100 >= target else "C28D_RETRIEVER_NOT_READY_CONTINUE_REPAIR"
    rec = {
        "stage": "C28D-4",
        "status": status,
        "teacher_vr100": teacher_vr100,
        "student_vr100": student_vr100,
        "gate_target_vr100": target,
        "front_rank_required": True,
        "replay": replay,
        **official_safety_manifest(),
    }
    write_json(out / "C28D_4_RETRIEVER_GATE_DECISION.json", rec)
    write_text(out / "C28D_4_RETRIEVER_GATE_DECISION.md", f"# C28D-4 Retriever Gate\n\nStatus: `{status}`.\n\nTeacher VR@100: `{teacher_vr100}`.\n\nStudent VR@100: `{student_vr100}`.\n\nGate target: `{target}`.\n")
    return rec


def stage5(cfg: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    out = REPORT / "c28d_5_final"
    gate = load_json(REPORT / "c28d_4_retriever_gate/C28D_4_RETRIEVER_GATE_DECISION.json", {})
    final = gate.get("status", "C28D_INCONCLUSIVE_NEED_REPAIR")
    rec = {
        "stage": "C28D-5",
        "status": final,
        "final_decision": final,
        "next_action": "run full mutual model only if retriever gate passed",
        "gate": gate,
        "config": cfg,
        **official_safety_manifest(),
    }
    write_json(out / "C28D_5_FINAL_DECISION.json", rec)
    write_text(out / "C28D_5_FINAL_DECISION.md", f"# C28D Final Decision\n\nFinal decision: `{final}`.\n")
    return rec


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", default="all", choices=["c28d_0", "c28d_1", "c28d_2", "c28d_3", "c28d_4", "c28d_5", "all"])
    parser.add_argument("--mode", default="medium", choices=["medium"])
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--config")
    parser.add_argument("--device")
    parser.add_argument("--split", default="calib_holdout")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--retriever_epochs", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--chunk_size", type=int)
    parser.add_argument("--max_queries", type=int)
    parser.add_argument("--max_videos", type=int)
    parser.add_argument("--dynamic_topk", type=int)
    parser.add_argument("--candidate_topk_train", type=int)
    parser.add_argument("--candidate_topk_eval", type=int)
    parser.add_argument("--eval_candidate_k", type=int)
    parser.add_argument("--hard_negative_k", type=int)
    parser.add_argument("--loss_negative_sample_k", type=int)
    parser.add_argument("--teacher_warm_topk", type=int)
    parser.add_argument("--max_spans_per_video", type=int)
    parser.add_argument("--grad_accum_steps", type=int)
    parser.add_argument("--moment_max_score_weight", type=float)
    args = parser.parse_args()
    cfg_path = Path(args.config) if args.config else ROOT / "blueprint_e2e_v2/configs/c28d_medium.yaml"
    cfg = merge_args(load_config(cfg_path), args)
    funcs = {
        "c28d_0": stage0,
        "c28d_1": stage1,
        "c28d_2": stage2,
        "c28d_3": stage3,
        "c28d_4": stage4,
        "c28d_5": stage5,
    }
    order = ["c28d_0", "c28d_1", "c28d_2", "c28d_3", "c28d_4", "c28d_5"] if args.stage == "all" else [args.stage]
    started = time.time()
    results = {}
    for st in order:
        results[st] = funcs[st](cfg, args)
        print(f"{st}: {results[st].get('status') or results[st].get('final_decision')}", flush=True)
    print("branch:", sh("git branch --show-current"), flush=True)
    print("commit:", sh("git rev-parse --short HEAD"), flush=True)
    print("elapsed_seconds:", round(time.time() - started, 2), flush=True)
    if "c28d_5" in results:
        print("C28D final decision:", results["c28d_5"].get("final_decision"), flush=True)


if __name__ == "__main__":
    main()
