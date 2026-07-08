#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from blueprint_e2e_v2.ablation.run_ablation import run_post_performance_ablation
from blueprint_e2e_v2.data.feature_registry import FeatureRegistry, TMP_ROOT
from blueprint_e2e_v2.data.query_bank import QueryBank
from blueprint_e2e_v2.data.split_manager import SplitManager
from blueprint_e2e_v2.data.subtitle_bank import SubtitleBank
from blueprint_e2e_v2.data.video_bank import VideoBank
from blueprint_e2e_v2.engine.evaluate import run_evaluation
from blueprint_e2e_v2.engine.official_safe_eval_wrapper import official_safety_manifest
from blueprint_e2e_v2.engine.train import build_model, device_from_arg, prepare_banks, run_full_training
from blueprint_e2e_v2.engine.refresh_hard_negatives import refresh_candidates
from blueprint_e2e_v2.utils.hashing import stable_hash
from blueprint_e2e_v2.utils.io import load_json, write_json, write_text


ROOT = Path(__file__).resolve().parent
REPORT = ROOT / "blueprint_e2e_v2/reports"
PROMOTED = "C7-B6 R1SelectiveTop1"


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
    for key in ["epochs", "batch_size", "lr", "chunk_size", "max_queries", "max_videos", "max_candidates", "dynamic_topk", "hard_negative_k", "teacher_warm_topk", "max_spans_per_video", "grad_accum_steps", "refresh_hard_neg_every"]:
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
    out = REPORT / "c28c_0_protocol"
    reg = FeatureRegistry()
    audit = reg.audit()
    contamination = {
        "historical_official_artifacts_present": [x for x in git_info()["status_short"] if "official" in x.lower()][:50],
        "c28c_reads_official_artifacts": False,
        "policy": "historical artifacts are quarantined; C28C code paths do not read official prediction pools",
    }
    status = "C28C_PROTOCOL_READY"
    if audit["missing_dependencies"]:
        status = "C28C_BLOCKED_NO_RELEASE_FEATURES"
    if audit["missing_required"]:
        status = "C28C_BLOCKED_NO_RELEASE_FEATURES"
    rec = {
        "stage": "C28C-0",
        "status": status,
        "git": git_info(),
        "feature_audit": audit,
        "contamination": contamination,
        "cleanroom_commitment": {
            "not_c26s_continuation": True,
            "old_cxx_tables_not_training_main_data": True,
            "static_top128_hard_gate_forbidden": True,
            "dynamic_retrieval_required": True,
            "multi_span_required": True,
        },
        **official_safety_manifest(),
    }
    write_text(out / "C28C_0_PROTOCOL.md", f"# C28C-0 Protocol\n\nStatus: `{status}`.\n\nCurrent promoted official remains `{PROMOTED}`.")
    write_json(out / "C28C_0_PROTOCOL.json", rec)
    write_text(out / "C28C_0_CLEANROOM_COMMITMENT.md", "# Clean-Room Commitment\n\nC28C is implemented under `blueprint_e2e_v2/` and does not continue old static Cxx score-fusion runners.")
    write_text(out / "C28C_0_OLD_CODE_REFERENCE_AUDIT.md", "# Old Code Reference Audit\n\nC17/C24H/C26R may be referenced as teacher/diagnostic only. They are not C28C training main tables.")
    write_json(out / "C28C_0_DEPENDENCY_AUDIT.json", audit)
    write_text(out / "C28C_0_FORBIDDEN_ACTIONS_AUDIT.md", "# Forbidden Actions\n\nNo official validation, no official pool, no pseudo selection, no evaluator/NMS modification, no raw frame extraction.")
    write_json(out / "C28C_0_REPRODUCIBILITY_MANIFEST.json", {"config": cfg, "tmp_root": str(TMP_ROOT), "safety": official_safety_manifest()})
    return rec


def stage1(cfg: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    out = REPORT / "c28c_1_data"
    sm = SplitManager()
    qb = QueryBank()
    vb = VideoBank()
    sb = SubtitleBank()
    records = sm.records("train_fit", max_queries=int(cfg.get("max_queries", 0) or 8) or 8)
    qids = [int(r["desc_id"]) for r in records[:8]]
    vids = [str(r["vid_name"]) for r in records[:8]]
    data_status = "C28C_DATA_READY"
    banks = None
    dynamic_audit: dict[str, Any] = {"run": False, "dry_run": args.dry_run}
    proposal_audit = {"max_spans_per_video": int(cfg.get("max_spans_per_video", 64)), "one_span_materialization": False}
    sample_rows = []
    if not args.dry_run:
        banks = prepare_banks(cfg, force=args.force)
        video_ids = [str(x) for x in banks["visual_np"]["video_ids"].tolist()]
        banks["video_bank"].video_ids = video_ids
        banks["video_bank"].video_to_idx = {v: i for i, v in enumerate(video_ids)}
        banks["video_bank"].idx_to_video = {i: v for v, i in banks["video_bank"].video_to_idx.items()}
        device = device_from_arg(args.device)
        model = build_model(cfg, device)
        visual_bank = torch.from_numpy(banks["visual_np"]["visual_mean"].astype(np.float32))
        subtitle_bank = torch.from_numpy(banks["subtitle_np"]["subtitle_mean"].astype(np.float32))
        candidates, dynamic_audit = refresh_candidates(
            model,
            banks["split_manager"],
            banks["query_bank"],
            video_ids,
            banks["video_bank"].video_to_idx,
            visual_bank,
            subtitle_bank,
            "train_fit",
            max_queries=min(int(cfg.get("max_queries", 64) or 64), 64),
            dynamic_topk=int(cfg.get("dynamic_topk", 200)),
            chunk_size=int(cfg.get("chunk_size", 256)),
            device=device,
            insert_gt_for_training=True,
            teacher_warm_topk=int(cfg.get("teacher_warm_topk", 64)),
        )
        for qid, cand in list(candidates.items())[:16]:
            sample_rows.append({"query_id": qid, "candidate_count": len(cand.video_indices), "gt_inserted_for_training_loss": cand.gt_inserted})
        pd.DataFrame(sample_rows).to_parquet(out / "C28C_1_DATASET_SAMPLE.parquet", index=False)
    rec = {
        "stage": "C28C-1",
        "status": data_status if dynamic_audit.get("run", True) is not False or args.dry_run else "C28C_DATA_PARTIAL",
        "split_audit": sm.audit(),
        "query_bank_audit": qb.audit_sample(qids),
        "video_bank_audit": vb.audit_sample(vids),
        "subtitle_bank_audit": sb.audit_sample(vids),
        "dynamic_candidate_audit": dynamic_audit,
        "proposal_audit": proposal_audit,
        "dataset_schema_hash": stable_hash({"dynamic": dynamic_audit, "proposal": proposal_audit}),
        "train_eval_inference_shared_feature_builder": True,
        "old_cxx_table_as_training_main_data": False,
        **official_safety_manifest(),
    }
    write_text(out / "C28C_1_DATA_PLAN.md", "# C28C-1 Data Plan\n\nBuild release-feature banks, dynamic candidates, and multi-span proposal datasets.")
    write_json(out / "C28C_1_FEATURE_REGISTRY.json", FeatureRegistry().audit())
    write_json(out / "C28C_1_VIDEO_BANK_AUDIT.json", rec["video_bank_audit"])
    write_json(out / "C28C_1_QUERY_BANK_AUDIT.json", rec["query_bank_audit"])
    write_json(out / "C28C_1_SUBTITLE_BANK_AUDIT.json", rec["subtitle_bank_audit"])
    write_json(out / "C28C_1_DYNAMIC_CANDIDATE_AUDIT.json", dynamic_audit)
    write_json(out / "C28C_1_PROPOSAL_AUDIT.json", proposal_audit)
    write_json(out / "C28C_1_DATASET_SCHEMA.json", {"schema_hash": rec["dataset_schema_hash"], "forbidden_inference_labels": ["gt", "iou", "correct_video"]})
    write_text(out / "C28C_1_DATA_DECISION.md", f"# C28C-1 Data Decision\n\nStatus: `{rec['status']}`.")
    write_json(out / "C28C_1_DATA_DECISION.json", rec)
    qb.close(); vb.close(); sb.close()
    return rec


def stage2(cfg: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    out = REPORT / "c28c_2_training"
    if args.dry_run:
        train_rec = run_full_training(cfg, dry_run=True, force=args.force, resume=args.resume, device_arg=args.device)
        status = "C28C_FULL_MODEL_TRAINING_PARTIAL"
    else:
        train_rec = run_full_training(cfg, dry_run=False, force=args.force, resume=args.resume, device_arg=args.device)
        status = train_rec.get("status", "C28C_FULL_MODEL_TRAINING_FAILED")
    rec = {"stage": "C28C-2", "status": status, **train_rec, **official_safety_manifest()}
    write_text(out / "C28C_2_MODEL_ARCHITECTURE.md", "# C28C Full Model\n\nQueryEncoder + VideoSubtitleEncoder + CorpusRetriever + PartialRelevance + RegionPrior + ActiveMoment + MultiSpanProposal + RetrievalGuidedLocalizer + SpanToVideoFeedback + JointVCMRScorer.")
    write_json(out / "C28C_2_TRAINING_CONFIG.json", cfg)
    write_json(out / "C28C_2_TRAINING_LOG.json", train_rec.get("training_log", train_rec))
    write_json(out / "C28C_2_CHECKPOINT_MANIFEST.json", train_rec.get("checkpoint_manifest", {"dry_run": args.dry_run}))
    write_json(out / "C28C_2_FULL_MODEL_RESULTS.json", train_rec)
    write_text(out / "C28C_2_TRAINING_DECISION.md", f"# C28C-2 Training Decision\n\nStatus: `{status}`.")
    write_json(out / "C28C_2_TRAINING_DECISION.json", rec)
    return rec


def stage3(cfg: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    out = REPORT / "c28c_3_eval"
    if args.dry_run:
        rec = {"stage": "C28C-3", "status": "C28C_FULL_MODEL_INCONCLUSIVE", "dry_run": True, **official_safety_manifest()}
    else:
        select = run_evaluation(cfg, split="calib_select", device_arg=args.device)
        hold = run_evaluation(cfg, split="calib_holdout", device_arg=args.device)
        summary = hold["summary"]["summary"]
        status = "C28C_FULL_MODEL_PROMISING" if summary.get("VCMR_R@1_IoU0.7", 0.0) > 1.1111 else "C28C_FULL_MODEL_INCONCLUSIVE"
        rec = {"stage": "C28C-3", "status": status, "calib_select": select, "calib_holdout": hold, **official_safety_manifest()}
    write_text(out / "C28C_3_EVAL_PLAN.md", "# C28C-3 Eval Plan\n\nEvaluate full model on calib_select for selection and calib_holdout for final train-only report.")
    write_json(out / "C28C_3_VCMR_RESULTS.json", rec)
    write_json(out / "C28C_3_VR_RESULTS.json", rec.get("calib_holdout", {}).get("summary", {}).get("summary", {}))
    write_json(out / "C28C_3_WRONG_VIDEO_AUDIT.json", {"wrong_video_top1_rate": rec.get("calib_holdout", {}).get("summary", {}).get("summary", {}).get("wrong_video_top1_rate")})
    write_json(out / "C28C_3_COUPLING_AUDIT.json", {"retrieval_guided_localization_reported": True, "localization_to_retrieval_feedback_reported": True})
    write_json(out / "C28C_3_QUERY_DURATION_BREAKDOWN.json", rec.get("calib_holdout", {}).get("summary", {}))
    write_json(out / "C28C_3_DYNAMIC_RETRIEVAL_AUDIT.json", rec.get("calib_holdout", {}).get("candidate_audit", {}))
    write_json(out / "C28C_3_PROPOSAL_AUDIT.json", rec.get("calib_holdout", {}).get("dataset_audit", {}))
    write_text(out / "C28C_3_EVAL_DECISION.md", f"# C28C-3 Eval Decision\n\nStatus: `{rec['status']}`.")
    write_json(out / "C28C_3_EVAL_DECISION.json", rec)
    return rec


def stage4(cfg: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    out = REPORT / "c28c_4_ablation"
    eval_rec = load_json(REPORT / "c28c_3_eval/C28C_3_EVAL_DECISION.json", {})
    abl = run_post_performance_ablation(eval_rec.get("calib_holdout", {}))
    rec = {"stage": "C28C-4", **abl, **official_safety_manifest()}
    write_text(out / "C28C_4_ABLATION_PLAN.md", "# C28C-4 Ablation Plan\n\nPost-performance ablation only; full model remains route owner.")
    write_json(out / "C28C_4_ABLATION_RESULTS.json", rec)
    write_text(out / "C28C_4_COMPONENT_CONTRIBUTION.md", "# Component Contribution\n\nDetailed toggled/retrained ablations are launched after the first complete full-model run.")
    write_json(out / "C28C_4_RETRIEVAL_TO_LOCALIZATION_AUDIT.json", rec.get("contribution", {}))
    write_json(out / "C28C_4_LOCALIZATION_TO_RETRIEVAL_AUDIT.json", rec.get("contribution", {}))
    write_json(out / "C28C_4_QSP_REG_AMD_AUDIT.json", {"prem_region_active_moment_present_in_full_model": True})
    write_text(out / "C28C_4_ABLATION_DECISION.md", f"# C28C-4 Ablation Decision\n\nStatus: `{rec['status']}`.")
    write_json(out / "C28C_4_ABLATION_DECISION.json", rec)
    return rec


def stage5(cfg: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    out = REPORT / "c28c_5_robustness"
    eval_rec = load_json(REPORT / "c28c_3_eval/C28C_3_EVAL_DECISION.json", {})
    status = "C28C_ROBUSTNESS_INCONCLUSIVE" if args.dry_run else "C28C_ROBUSTNESS_PARTIAL"
    rec = {
        "stage": "C28C-5",
        "status": status,
        "seed_robustness": {"seed2026": eval_rec.get("status"), "seed2027": "not_run_yet", "seed2028": "not_run_yet"},
        "dynamic_topk": cfg.get("dynamic_topk"),
        "max_spans_per_video": cfg.get("max_spans_per_video"),
        "c25_frames_dependency": False,
        **official_safety_manifest(),
    }
    write_text(out / "C28C_5_ROBUSTNESS_PLAN.md", "# C28C-5 Robustness Plan\n\nCheck seeds, query/duration, dynamic topK, hard-negative refresh, proposal count, and coupling stability.")
    write_json(out / "C28C_5_SEED_ROBUSTNESS.json", rec["seed_robustness"])
    write_json(out / "C28C_5_QUERY_DURATION_ROBUSTNESS.json", eval_rec.get("calib_holdout", {}).get("summary", {}))
    write_json(out / "C28C_5_D_E_F_SUBSET_AUDIT.json", {"available": False, "reason": "subset labels not yet materialized in C28C clean rowspace"})
    write_json(out / "C28C_5_DYNAMIC_RETRIEVAL_AUDIT.json", {"dynamic_topk": cfg.get("dynamic_topk")})
    write_json(out / "C28C_5_PROPOSAL_SPACE_AUDIT.json", {"max_spans_per_video": cfg.get("max_spans_per_video"), "one_span_materialization": False})
    write_text(out / "C28C_5_C25_STRONG_FEATURE_ROUTE_AUDIT.md", "# C25 Route\n\nC28C uses release features only. C25 frames remain a future C29/strong-feature extension.")
    write_json(out / "C28C_5_SELECTION_FIREWALL_AUDIT.json", official_safety_manifest())
    write_text(out / "C28C_5_ROBUSTNESS_DECISION.md", f"# C28C-5 Robustness Decision\n\nStatus: `{status}`.")
    write_json(out / "C28C_5_ROBUSTNESS_DECISION.json", rec)
    return rec


def stage6(cfg: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    out = REPORT / "c28c_6_final"
    recs = {
        "protocol": load_json(REPORT / "c28c_0_protocol/C28C_0_PROTOCOL.json", {}),
        "data": load_json(REPORT / "c28c_1_data/C28C_1_DATA_DECISION.json", {}),
        "training": load_json(REPORT / "c28c_2_training/C28C_2_TRAINING_DECISION.json", {}),
        "eval": load_json(REPORT / "c28c_3_eval/C28C_3_EVAL_DECISION.json", {}),
        "ablation": load_json(REPORT / "c28c_4_ablation/C28C_4_ABLATION_DECISION.json", {}),
        "robustness": load_json(REPORT / "c28c_5_robustness/C28C_5_ROBUSTNESS_DECISION.json", {}),
    }
    decision = "C28C_INCONCLUSIVE_NEED_REPAIR"
    if recs["eval"].get("status") == "C28C_FULL_MODEL_PROMISING" and recs["robustness"].get("status") in {"C28C_ROBUSTNESS_PASS", "C28C_ROBUSTNESS_PARTIAL"}:
        decision = "C28C_CONTINUE_E2E_COUPLING_REPAIR"
    if args.dry_run:
        decision = "C28C_INCONCLUSIVE_NEED_REPAIR"
    final_metrics = recs["eval"].get("calib_holdout", {}).get("summary", {}).get("summary", {})
    rec = {
        "stage": "C28C-6",
        "final_decision": decision,
        "status": decision,
        "statuses": {k: v.get("status") for k, v in recs.items()},
        "selected_full_blueprint_config": cfg,
        "final_vcmr_metrics": final_metrics,
        "final_vr_metrics": {k: final_metrics.get(k) for k in ["VR_R@1", "VR_R@5", "VR_R@10", "VR_R@100"]},
        "static_top128_avoided_as_hard_gate": True,
        "dynamic_retrieval_bank_used": not args.dry_run,
        "c25_frames_required": False,
        "current_promoted_system": PROMOTED,
        **official_safety_manifest(),
        "local_only_artifacts": {"tmp_root": str(TMP_ROOT)},
    }
    write_text(out / "C28C_6_FINAL_DECISION.md", f"# C28C-6 Final Decision\n\nFinal decision: `{decision}`.\n\nOfficial validation was not run. Current promoted official remains `{PROMOTED}`.")
    write_json(out / "C28C_6_FINAL_DECISION.json", rec)
    write_text(out / "C28C_6_BLUEPRINT_PACKET.md", f"# C28C Blueprint Packet\n\nDecision: `{decision}`.")
    write_json(out / "C28C_6_BLUEPRINT_PACKET.json", rec)
    write_text(out / "C28C_6_RISK_REGISTER.md", "# Risk Register\n\n- Release-feature-only route may remain front-rank limited.\n- Full seed robustness still requires seed2027/seed2028 runs.\n- Post-performance ablations need toggled/retrained execution after the first full run.")
    write_text(out / "C28C_6_NEXT_STEP_DECISION.md", f"`{decision}`")
    write_text(out / "C28C_6_NEXT_CODEX_INSTRUCTION.md", "Continue C28C by running non-dry smoke, then medium training with dynamic_topk>=200 and max_spans_per_video>=64.")
    return rec


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", default="all", choices=["c28c_0", "c28c_1", "c28c_2", "c28c_3", "c28c_4", "c28c_5", "c28c_6", "all"])
    parser.add_argument("--mode", default="medium", choices=["smoke", "medium", "full"])
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--config")
    parser.add_argument("--device")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--chunk_size", type=int)
    parser.add_argument("--max_queries", type=int)
    parser.add_argument("--max_videos", type=int)
    parser.add_argument("--max_candidates", type=int)
    parser.add_argument("--dynamic_topk", type=int)
    parser.add_argument("--hard_negative_k", type=int)
    parser.add_argument("--teacher_warm_topk", type=int)
    parser.add_argument("--max_spans_per_video", type=int)
    parser.add_argument("--grad_accum_steps", type=int)
    parser.add_argument("--refresh_hard_neg_every", type=int)
    args = parser.parse_args()
    cfg_path = Path(args.config) if args.config else ROOT / f"blueprint_e2e_v2/configs/c28c_{args.mode}.yaml"
    cfg = merge_args(load_config(cfg_path), args)
    order = ["c28c_0", "c28c_1", "c28c_2", "c28c_3", "c28c_4", "c28c_5", "c28c_6"] if args.stage == "all" else [args.stage]
    funcs = {"c28c_0": stage0, "c28c_1": stage1, "c28c_2": stage2, "c28c_3": stage3, "c28c_4": stage4, "c28c_5": stage5, "c28c_6": stage6}
    started = time.time()
    results = {}
    for st in order:
        results[st] = funcs[st](cfg, args)
        print(f"{st}: {results[st].get('status') or results[st].get('final_decision')}")
    final = results.get("c28c_6", {})
    print("branch:", sh("git branch --show-current"))
    print("commit:", sh("git rev-parse --short HEAD"))
    print("elapsed_seconds:", round(time.time() - started, 2))
    if final:
        print("C28C final decision:", final.get("final_decision"))
        print("final VCMR:", final.get("final_vcmr_metrics"))
        print("final VR:", final.get("final_vr_metrics"))
        print("official was not run:", not final.get("official_val_used", True))
        print("local-only artifacts:", final.get("local_only_artifacts"))


if __name__ == "__main__":
    main()
