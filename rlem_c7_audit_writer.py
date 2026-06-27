#!/usr/bin/env python
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, List


ROOT = Path(__file__).resolve().parent
C6_FINAL = ROOT / "c6_final_audit"
C7_AUDIT = ROOT / "c7_integrated_audit"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def artifact(path: str | Path) -> Dict[str, Any]:
    p = ROOT / path if not isinstance(path, Path) else path
    return {
        "path": str(p),
        "exists": p.exists(),
        "size": int(p.stat().st_size) if p.exists() else None,
        "sha256": sha256_file(p) if p.exists() and p.is_file() else None,
    }


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(path) + ".partial")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(path) + ".partial")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def metric(obj: Dict[str, Any], *keys: str, default: Any = None) -> Any:
    cur: Any = obj
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def load_json(path: str) -> Dict[str, Any]:
    p = ROOT / path
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def c6_finalization() -> None:
    c1 = load_json("c6_c_audit/C6_C1_FINAL_DECISION.json")
    c11 = load_json("c6_c_audit/C6_C11_FINAL_DECISION.json")
    c2 = load_json("c6_c2_audit/C6_C2_FINAL_DECISION.json")
    b2 = load_json("c6_b2_final/C6_B2_FINAL_MANIFEST.json")
    payload = {
        "stage": "C6 external line finalization",
        "date": "2026-06-26",
        "stable_safety_anchor_system": "C6-B1-lite c6b1r1_0057",
        "r1_oriented_extension_system": "C6-B2 pairwise_main_0005",
        "current_promoted_system": "C6-B2 pairwise_main_0005",
        "c6_b1_role": "stable safety anchor / all-metric stable branch",
        "c6_b2_role": "R1-oriented official extension / span utility branch",
        "c6_c1_status": "negative_topk_collapse",
        "c6_c1_recorded_status": metric(c1, "status", default="C6_C1_R1_TRADEOFF"),
        "c6_c11_status": "safe_b2_equivalent_no_promotion",
        "c6_c11_recorded_status": metric(c11, "status", default="C6_C11_SAFE_B2_EQUIVALENT_NO_PROMOTION"),
        "c6_c2_status": metric(c2, "status", default="C6_C2_R1_TRADEOFF"),
        "c6_c2_promoted": bool(metric(c2, "promoted", default=False)),
        "c6_c2_feasible_count": 0,
        "c6_external_line_closed": True,
        "continue_c6_c3": False,
        "continue_c6_c2_1_threshold_search": False,
        "official_val_used_for_c6c": False,
        "second_official_val": False,
        "post_val_adjustment": False,
        "evaluator_modified": False,
        "nms_modified": False,
        "c6_artifacts_modified": False,
        "reason_to_stop": [
            "C6-C1 showed video-rerank R1 potential but topK collapse risk.",
            "C6-C1.1 prevented collapse only by selecting zero enabled queries, making it B2-equivalent.",
            "C6-C2 learned non-zero query-level interventions but remained an R1 tradeoff with feasible_count=0.",
            "Further C6-C threshold or policy search risks overfitting train_calib without adding integrated model capacity.",
        ],
        "source_artifacts": {
            "c6_b2_final_manifest": artifact("c6_b2_final/C6_B2_FINAL_MANIFEST.json"),
            "c6_c1_final": artifact("c6_c_audit/C6_C1_FINAL_DECISION.json"),
            "c6_c11_final": artifact("c6_c_audit/C6_C11_FINAL_DECISION.json"),
            "c6_c2_final": artifact("c6_c2_audit/C6_C2_FINAL_DECISION.json"),
        },
        "b2_manifest_status": metric(b2, "status", default=None),
    }
    write_json(C6_FINAL / "C6_EXTERNAL_FINALIZATION.json", payload)
    md = [
        "# C6 External Finalization",
        "",
        "- Stage: `C6 external line finalization`",
        "- Stable safety anchor: `C6-B1-lite c6b1r1_0057`",
        "- Current promoted / R1-oriented extension: `C6-B2 pairwise_main_0005`",
        "- C6 external line closed: `true`",
        "- Continue C6-C3: `false`",
        "- Official val used for C6-C1/C6-C1.1/C6-C2: `false`",
        "- Second official val: `false`",
        "- Post-val adjustment: `false`",
        "- Evaluator/NMS modified: `false`",
        "",
        "## Stage Decisions",
        "",
        "- `C6-B1-lite`: stable safety anchor and all-metric stable branch.",
        "- `C6-B2`: R1-oriented official extension and span utility branch; it remains the current promoted system.",
        "- `C6-C1`: negative for promotion because external video reranking hurt topK.",
        "- `C6-C1.1`: no promotion because the safe repair fell back to B2-equivalent no-op.",
        "- `C6-C2`: no promotion because non-zero interventions remained `C6_C2_R1_TRADEOFF` with feasible_count=0.",
        "",
        "## Decision",
        "",
        "C6 external/interface verification stops here. Do not continue C6-C3 or C6-C2.1 threshold/policy search; the next work item is C7 integrated CONQUER-RLEM source audit and design only.",
        "",
        "```json",
        json.dumps(payload, indent=2, ensure_ascii=False),
        "```",
        "",
    ]
    write_text(C6_FINAL / "C6_EXTERNAL_FINALIZATION.md", "\n".join(md))

    summary = {
        "stage": "C6 external stage summary",
        "closed": True,
        "promoted_result_remains": "C6-B2 pairwise_main_0005",
        "stage_table": [
            {"stage": "C6-B1-lite", "role": "stable safety anchor", "decision": "retain"},
            {"stage": "C6-B2", "role": "R1-oriented extension / span utility", "decision": "current promoted"},
            {"stage": "C6-C1", "role": "external video rerank", "decision": "negative_topk_collapse"},
            {"stage": "C6-C1.1", "role": "topK-preserving repair", "decision": "safe_b2_equivalent_no_promotion"},
            {"stage": "C6-C2", "role": "query-level intervention utility gate", "decision": "R1_TRADEOFF_no_promotion"},
        ],
        "forbidden_next_actions": [
            "official val",
            "second official val",
            "post-val adjustment",
            "C6-C3",
            "C6-C2.1 threshold search",
            "C8 / GenSpan-lite training",
        ],
    }
    write_json(C6_FINAL / "C6_EXTERNAL_STAGE_SUMMARY.json", summary)
    summary_md = "# C6 External Stage Summary\n\n"
    summary_md += "| stage | role | decision |\n|---|---|---|\n"
    for row in summary["stage_table"]:
        summary_md += f"| {row['stage']} | {row['role']} | {row['decision']} |\n"
    summary_md += "\nC6 is closed. C7 begins as source audit and design only; no training is authorized in this step.\n"
    write_text(C6_FINAL / "C6_EXTERNAL_STAGE_SUMMARY.md", summary_md)


def module_records() -> List[Dict[str, Any]]:
    return [
        {
            "module": "data_loading_entry",
            "file_path": "data_loader/second_stage_start_end_dataset.py",
            "class_or_function": "StartEndDataset.__getitem__",
            "line_range": "226-363",
            "input_tensor_shape": "query feat (Lq<=30,768); visual feat (Nv,Lv<=100,4352); sub feat optional (Nv,Lv<=100,768)",
            "output_tensor_shape": "model_inputs dict; train Nv=1+neg_video_num, eval Nv=1+max_vcmr_video",
            "training_usage": "samples one GT video plus hard negatives from first-stage VR ranklist; emits st_ed_indices",
            "inference_usage": "uses first-stage VR top-k; stores inference_vr_scores and sample_vid_name_list",
            "safe_to_modify": False,
            "should_remain_frozen_in_C7_B": True,
            "c7_notes": "Do not modify data loader in C7-B; add optional sidecar tensors only after an audit if needed.",
        },
        {
            "module": "train_entry",
            "file_path": "train.py",
            "class_or_function": "start_training/train/train_epoch",
            "line_range": "38-365",
            "input_tensor_shape": "batched model_inputs from start_end_collate",
            "output_tensor_shape": "loss scalar plus loss_dict",
            "training_usage": "calls model(model_inputs), optimizer, eval_epoch",
            "inference_usage": "not used except train-time validation",
            "safe_to_modify": True,
            "should_remain_frozen_in_C7_B": False,
            "c7_notes": "C7-B can add a separate train script/wrapper; avoid changing original train.py until freeze review.",
        },
        {
            "module": "config_args",
            "file_path": "config/config.py; config/model_config.json",
            "class_or_function": "BaseOptions/TestOptions; model_config",
            "line_range": "config.py 37-206; model_config.json 1-68",
            "input_tensor_shape": "N/A",
            "output_tensor_shape": "opt namespace; CONQUER config object",
            "training_usage": "sets ctx_mode, max_vcmr_video, losses, similarity_measure, lr schedule",
            "inference_usage": "loads saved opt then allows eval overrides",
            "safe_to_modify": True,
            "should_remain_frozen_in_C7_B": False,
            "c7_notes": "Prefer C7-specific config file/args rather than changing defaults.",
        },
        {
            "module": "conquer_model_class",
            "file_path": "model/conquer.py",
            "class_or_function": "CONQUER",
            "line_range": "13-226",
            "input_tensor_shape": "batch with query/video/sub tensors",
            "output_tensor_shape": "video_similarity_score (B,Nv) or None; begin/end logits (B,Nv,Lv)",
            "training_usage": "computes moment CE and optional video CE",
            "inference_usage": "get_pred_from_raw_query provides logits used by VCMR/SVMR/VR",
            "safe_to_modify": "C7 wrapper preferred; direct edit only after design approval",
            "should_remain_frozen_in_C7_B": True,
            "c7_notes": "Existing return_intermediates path exposes qdf/qal/contextual/G/video_mask needed by new heads.",
        },
        {
            "module": "qdf_module",
            "file_path": "model/conquer.py; model/backbone/encoder.py",
            "class_or_function": "CONQUER.compute_final_score; QueryWeightEncoder",
            "line_range": "conquer.py 92-107; encoder.py 196-231",
            "input_tensor_shape": "per-modality video features dict each (B*Nv,Lv,768); moe weights (B*Nv)",
            "output_tensor_shape": "QDF_feature (B*Nv,Lv,768)",
            "training_usage": "query dependent fusion of visual/sub modalities",
            "inference_usage": "same path",
            "safe_to_modify": False,
            "should_remain_frozen_in_C7_B": True,
            "c7_notes": "C7-B should consume QDF/QAL outputs, not alter QDF fusion.",
        },
        {
            "module": "qal_module",
            "file_path": "model/qal/query_aware_learning_module.py",
            "class_or_function": "BiDirectionalAttention.forward",
            "line_range": "16-94",
            "input_tensor_shape": "QDF_emb (B*Nv,Lv,768), query_emb (B*Nv,Lq,768), masks",
            "output_tensor_shape": "QAL (B*Nv,Lv,3072); optional q2v attention (B*Nv,Lv)",
            "training_usage": "core query-aware feature learning before localization/video heads",
            "inference_usage": "same path",
            "safe_to_modify": False,
            "should_remain_frozen_in_C7_B": True,
            "c7_notes": "Late QAL unfreeze belongs only to C7-C if C7-B passes.",
        },
        {
            "module": "shared_contextual_qal",
            "file_path": "model/conquer.py; model/layers.py",
            "class_or_function": "FCPlusTransformer",
            "line_range": "conquer.py 53-54,148-153; layers.py 88-115",
            "input_tensor_shape": "QAL_feature (B*Nv,Lv,3072)",
            "output_tensor_shape": "Contextual_QAL (B*Nv,Lv,768); G concat (B*Nv,Lv,3840)",
            "training_usage": "shared contextual layer for ML and optional VS heads",
            "inference_usage": "same path",
            "safe_to_modify": False,
            "should_remain_frozen_in_C7_B": True,
            "c7_notes": "C7 heads should attach after G/Contextual_QAL.",
        },
        {
            "module": "moment_localization_head",
            "file_path": "model/head/ml_head.py",
            "class_or_function": "MomentLocalizationHead.forward",
            "line_range": "10-59",
            "input_tensor_shape": "G (B*Nv,Lv,3840), Contextual_QAL (B*Nv,Lv,768), video_mask (B*Nv,Lv)",
            "output_tensor_shape": "begin logits (B*Nv,Lv), end logits (B*Nv,Lv), reshaped to (B,Nv,Lv)",
            "training_usage": "moment CE via shared-normalized flattened start/end labels",
            "inference_usage": "softmaxed start/end probabilities produce SVMR/VCMR tuples",
            "safe_to_modify": "additive heads only in C7-B",
            "should_remain_frozen_in_C7_B": True,
            "c7_notes": "Proposal confidence can reuse begin/end features/logits without changing this head.",
        },
        {
            "module": "video_retrieval_head",
            "file_path": "model/head/vs_head.py",
            "class_or_function": "VideoScoringHead.forward",
            "line_range": "9-41",
            "input_tensor_shape": "G (B*Nv,Lv,3840), video_mask (B*Nv,Lv)",
            "output_tensor_shape": "video_similarity_score (B*Nv,1), reshaped to (B,Nv)",
            "training_usage": "only when similarity_measure='exclusive'",
            "inference_usage": "if internal VR scores are enabled; otherwise external first-stage scores are used",
            "safe_to_modify": "C7-C only",
            "should_remain_frozen_in_C7_B": True,
            "c7_notes": "C7-B should learn a residual moment-aware VR score rather than overwrite this head.",
        },
        {
            "module": "vcmr_score_generation",
            "file_path": "inference.py",
            "class_or_function": "compute_query2ctx_info / compute_query2ctx_info_disjoint",
            "line_range": "163-368; 371-563",
            "input_tensor_shape": "begin/end logits (B,Nv,Lv), video scores (B,Nv)",
            "output_tensor_shape": "flat top max_before_nms VCMR tuples per query",
            "training_usage": "train-time validation",
            "inference_usage": "S_video * p_start * p_end for general/exclusive; start+end for disjoint",
            "safe_to_modify": "C7 wrapper preferred",
            "should_remain_frozen_in_C7_B": True,
            "c7_notes": "C7 integrated scorer should be implemented as an alternate inference function, leaving original intact.",
        },
        {
            "module": "nms_top100_logic",
            "file_path": "utils/inference_utils.py; utils/temporal_nms.py",
            "class_or_function": "post_processing_vcmr_nms/filter_vcmr_by_nms/temporal_non_maximum_suppression",
            "line_range": "inference_utils.py 21-76; temporal_nms.py 25-74",
            "input_tensor_shape": "prediction lists [video_idx, st, ed, score]",
            "output_tensor_shape": "top max_after_nms predictions after per-video temporal NMS",
            "training_usage": "train-time validation when nms_thd != -1",
            "inference_usage": "official/eval post-processing",
            "safe_to_modify": False,
            "should_remain_frozen_in_C7_B": True,
            "c7_notes": "Do not alter NMS/top100 logic. C7 must produce better scores before this boundary.",
        },
        {
            "module": "loss_computation",
            "file_path": "model/conquer.py",
            "class_or_function": "CONQUER.forward/get_moment_loss_share_norm",
            "line_range": "188-226",
            "input_tensor_shape": "begin/end logits (B,Nv,Lv), st_ed_indices (B,2), optional video_similarity_score (B,Nv)",
            "output_tensor_shape": "scalar loss and loss_dict",
            "training_usage": "moment CE plus optional video CE",
            "inference_usage": "not used",
            "safe_to_modify": "C7 wrapper preferred",
            "should_remain_frozen_in_C7_B": True,
            "c7_notes": "C7-B should compute additional losses outside original forward or in a subclass without changing baseline loss.",
        },
        {
            "module": "checkpoint_loading",
            "file_path": "train.py; inference.py",
            "class_or_function": "encoder_pretrain load; setup_model",
            "line_range": "train.py 345-359; inference.py 650-677",
            "input_tensor_shape": "checkpoint state_dict",
            "output_tensor_shape": "CONQUER model with loaded weights",
            "training_usage": "can load encoder/query_weight pretrain",
            "inference_usage": "loads full checkpoint['model']",
            "safe_to_modify": "C7 wrapper preferred",
            "should_remain_frozen_in_C7_B": True,
            "c7_notes": "C7-B should load C6/B2-compatible CONQUER checkpoint and freeze original parameters.",
        },
    ]


def tensor_interface_map() -> List[Dict[str, Any]]:
    return [
        {"name": "query.feat", "shape": "(B,Lq<=30,768)", "producer": "StartEndDataset.get_query_feat_by_desc_id", "consumer": "VideoQueryEncoder.forward_repr_query", "c7_use": "query encoder input; frozen in C7-B"},
        {"name": "visual.feat", "shape": "(B,Nv,Lv<=100,4352)", "producer": "StartEndDataset.__getitem__", "consumer": "VideoQueryEncoder.forward_repr_video", "c7_use": "video encoder input; frozen in C7-B"},
        {"name": "sub.feat", "shape": "(B,Nv,Lv<=100,768)", "producer": "StartEndDataset.__getitem__", "consumer": "TwoModalEncoder", "c7_use": "subtitle branch input; frozen in C7-B"},
        {"name": "_query_feature", "shape": "(B,Lq,768)", "producer": "CONQUER.encoder(task='repr_query')", "consumer": "repeat_interleave for shared videos", "c7_use": "available in intermediates; can condition RLEM heads"},
        {"name": "video_feature_dict", "shape": "dict modality -> (B*Nv,Lv,768)", "producer": "CONQUER.encoder(task='repr_video')", "consumer": "QDF compute_final_score", "c7_use": "diagnostic export only in C7-B"},
        {"name": "moe_weights_dict", "shape": "dict modality -> (B*Nv)", "producer": "QueryWeightEncoder", "consumer": "compute_final_score", "c7_use": "modality agreement/conflict features"},
        {"name": "QDF_feature", "shape": "(B*Nv,Lv,768)", "producer": "compute_final_score", "consumer": "BiDirectionalAttention", "c7_use": "frozen feature source"},
        {"name": "QAL_feature", "shape": "(B*Nv,Lv,3072)", "producer": "BiDirectionalAttention", "consumer": "contextual_QAL_feature_learning", "c7_use": "primary C7 head input"},
        {"name": "Contextual_QAL", "shape": "(B*Nv,Lv,768)", "producer": "FCPlusTransformer", "consumer": "MomentLocalizationHead and G concat", "c7_use": "proposal confidence and MIL input"},
        {"name": "G", "shape": "(B*Nv,Lv,3840)", "producer": "cat(QAL_feature, Contextual_QAL)", "consumer": "ML head and VS head", "c7_use": "primary integrated head input"},
        {"name": "begin_score_distribution", "shape": "(B,Nv,Lv)", "producer": "MomentLocalizationHead", "consumer": "loss/inference", "c7_use": "boundary evidence and proposal confidence"},
        {"name": "end_score_distribution", "shape": "(B,Nv,Lv)", "producer": "MomentLocalizationHead", "consumer": "loss/inference", "c7_use": "boundary evidence and proposal confidence"},
        {"name": "video_similarity_score", "shape": "(B,Nv) or None", "producer": "VideoScoringHead when exclusive", "consumer": "video CE and inference", "c7_use": "residual video score anchor; remain frozen in C7-B"},
        {"name": "inference_vr_scores", "shape": "(B,max_vcmr_video)", "producer": "StartEndDataset eval first-stage ranklist", "consumer": "compute_query2ctx_info when use_interal_vr_scores", "c7_use": "external VR score anchor"},
        {"name": "VCMR score tensor", "shape": "(B,max_vcmr_video,Lv,Lv)", "producer": "inference score composition", "consumer": "top max_before_nms flatten/sort", "c7_use": "C7 score enters before flatten/sort, not after NMS"},
    ]


def c7_source_audit() -> None:
    records = module_records()
    tensor_map = tensor_interface_map()
    payload = {
        "stage": "C7 CONQUER source audit",
        "scope": "source_read_only_no_training",
        "official_val_used": False,
        "training_started": False,
        "source_modified": False,
        "summary": {
            "best_insertion_point_C7_B": "A wrapper/subclass around CONQUER.get_pred_from_raw_query using return_intermediates=True, attaching frozen-backbone RLEM heads after G/Contextual_QAL/start/end logits and before VCMR flatten/sort.",
            "do_not_modify": ["evaluator", "NMS", "C6 artifacts", "first-stage official-val outputs"],
            "safe_export_tensors": ["qdf_feature", "qal_feature", "contextual_qal", "g_feature", "video_mask", "begin/end logits", "moe_weights_dict", "q2v_attention"],
            "existing_vs_head": "Present only for similarity_measure='exclusive'; not active in general external VR-score path.",
            "existing_r2_head": "No explicit R2 head in base CONQUER; RLEM C4/C6 R2-style heads are external artifacts.",
            "existing_proposal_confidence_head": "No integrated proposal confidence head exists in base CONQUER.",
        },
        "modules": records,
    }
    write_json(C7_AUDIT / "C7_CONQUER_SOURCE_AUDIT.json", payload)
    md = [
        "# C7 CONQUER Source Audit",
        "",
        "- Scope: `source read only`",
        "- Training started: `false`",
        "- Official val used: `false`",
        "- Evaluator/NMS modified: `false`",
        "",
        "## Recommendation",
        "",
        "C7-B should be implemented as an integrated wrapper/subclass that freezes CONQUER and attaches RLEM heads to exported intermediates from `get_pred_from_raw_query(..., return_intermediates=True)`. The score should enter before VCMR flatten/sort and before unchanged NMS/top100.",
        "",
        "## Module Map",
        "",
    ]
    for rec in records:
        md += [
            f"### {rec['module']}",
            "",
            f"- File: `{rec['file_path']}`",
            f"- Class/function: `{rec['class_or_function']}`",
            f"- Lines: `{rec['line_range']}`",
            f"- Input shape: `{rec['input_tensor_shape']}`",
            f"- Output shape: `{rec['output_tensor_shape']}`",
            f"- Training usage: {rec['training_usage']}",
            f"- Inference usage: {rec['inference_usage']}",
            f"- Safe to modify: `{rec['safe_to_modify']}`",
            f"- Frozen in C7-B: `{rec['should_remain_frozen_in_C7_B']}`",
            f"- C7 note: {rec['c7_notes']}",
            "",
        ]
    write_text(C7_AUDIT / "C7_CONQUER_SOURCE_AUDIT.md", "\n".join(md))

    tensor_payload = {
        "stage": "C7 tensor interface map",
        "scope": "source_read_only_no_training",
        "official_val_used": False,
        "interfaces": tensor_map,
    }
    write_json(C7_AUDIT / "C7_TENSOR_INTERFACE_MAP.json", tensor_payload)
    tmd = ["# C7 Tensor Interface Map", "", "| tensor | shape | producer | consumer | C7 use |", "|---|---|---|---|---|"]
    for row in tensor_map:
        tmd.append(f"| `{row['name']}` | `{row['shape']}` | {row['producer']} | {row['consumer']} | {row['c7_use']} |")
    tmd.append("")
    write_text(C7_AUDIT / "C7_TENSOR_INTERFACE_MAP.md", "\n".join(tmd))


def c7_design_and_plan() -> None:
    design = {
        "stage": "C7 integrated RLEM design",
        "official_val_used": False,
        "training_started": False,
        "design_principle": "Move RLEM evidence inside CONQUER scoring, before tuple ranking and unchanged NMS, while freezing base CONQUER in C7-B.",
        "components": {
            "A_moment_aware_vr_head": {
                "formula": "S_video_C7 = S_video_CONQUER + beta * MomentAwareEvidence(q,v) - rho * VideoRisk(q,v)",
                "inputs": ["G", "Contextual_QAL", "begin/end logits", "proposal confidence", "span utility", "partial relevance pooling", "moment evidence concentration"],
                "outputs": ["video_residual_score", "video_risk_score", "moment_aware_evidence"],
                "C7_B_scope": "train frozen-backbone residual head only; beta/rho selected on train_calib",
                "C7_C_scope": "optionally unfreeze VR/ML heads only after C7-B passes",
            },
            "B_partial_relevance_MIL_aggregator": {
                "goal": "Treat each video as a bag of candidate moments; local relevant moments dominate video relevance.",
                "pooling": {
                    "top1": "C7-B first; low capacity and directly comparable to current VCMR max.",
                    "topk_mean": "C7-B first with k in {3,5}; more stable than top1.",
                    "logsumexp": "C7-B first with temperature search on train_calib.",
                    "attention_MIL": "C7-C unless C7-B simple poolings pass; higher capacity/risk.",
                    "noisy_or": "C7-C diagnostic; useful for multi-evidence aggregation but can overcount correlated spans.",
                },
                "outputs": ["mil_video_evidence", "moment_density", "evidence_entropy"],
            },
            "C_proposal_confidence_head": {
                "goal": "BMN/BSN-style proposal confidence because start/end boundary logits alone were insufficient.",
                "inputs": ["query-aware clip features", "start logits", "end logits", "inside span evidence", "boundary neighborhood evidence", "span length embedding"],
                "outputs": ["proposal_confidence", "proposal_iou05_logit", "proposal_iou07_logit", "span_quality_logit"],
                "C7_B_scope": "small head over top candidate spans; no backbone update.",
                "risk_control": "Use as span-level score and MIL evidence, not direct video-order override without safety gate.",
            },
            "D_hard_negative_contrastive_video_loss": {
                "goal": "Separate correct video from semantically similar wrong videos without correct moment.",
                "hard_negative_sources": ["CONQUER high-score wrong videos", "C6-B1 high-score wrong videos", "C6-B2 high-score wrong videos", "moment-evidence high but wrong videos", "same-show/same-episode metadata if available"],
                "loss_formula": "L_total = L_CONQUER_original + lambda_vr * L_moment_aware_video + lambda_mil * L_partial_relevance_MIL + lambda_prop * L_proposal_confidence + lambda_hn * L_hard_negative_contrastive",
                "C7_B_scope": "contrast new residual video embeddings/scores only; base CONQUER frozen.",
            },
        },
        "score_composition": {
            "video_level": "anchored to original CONQUER/external VR score plus bounded residual",
            "span_level": "proposal confidence and span_quality modify within-video moment ranking first",
            "tuple_level": "VCMR tuple score uses unchanged top100/NMS after C7 score tensor is formed",
        },
        "safety_rules": [
            "No official val in C7-B/C design or training search.",
            "NMS/evaluator remain unchanged.",
            "Do not promote if train_calib feasible_count=0.",
            "Require non-zero integrated contribution, not B2-equivalent no-op.",
            "Report deltas vs C6-B2 and C6-B1 safety anchor.",
        ],
    }
    write_json(C7_AUDIT / "C7_INTEGRATED_RLEM_DESIGN.json", design)
    md = [
        "# C7 Integrated RLEM Design",
        "",
        "- Training started: `false`",
        "- Official val used: `false`",
        "- Design principle: integrate RLEM before tuple ranking and unchanged NMS, with base CONQUER frozen in C7-B.",
        "",
        "## Component A: Moment-Aware VR Head",
        "",
        "`S_video_C7 = S_video_CONQUER + beta * MomentAwareEvidence(q,v) - rho * VideoRisk(q,v)`",
        "",
        "Evidence comes from start/end logits, proposal confidence, span utility, partial relevance pooling, and moment evidence concentration.",
        "",
        "## Component B: Partial Relevance MIL Aggregator",
        "",
        "- C7-B first: top1 pooling, top-k mean pooling, logsumexp pooling.",
        "- C7-C later: attention MIL pooling and noisy-or pooling.",
        "",
        "## Component C: Proposal Confidence Head",
        "",
        "Inputs are query-aware clip features, start/end logits, inside-span evidence, boundary-neighborhood evidence, and span-length embedding. Outputs are `proposal_confidence`, `proposal_iou05_logit`, `proposal_iou07_logit`, and `span_quality_logit`.",
        "",
        "## Component D: Hard Negative Contrastive Video Loss",
        "",
        "`L_total = L_CONQUER_original + lambda_vr * L_moment_aware_video + lambda_mil * L_partial_relevance_MIL + lambda_prop * L_proposal_confidence + lambda_hn * L_hard_negative_contrastive`",
        "",
        "Hard negatives come from CONQUER/C6-B1/C6-B2 high-score wrong videos, moment-evidence-high wrong videos, and same-show/same-episode metadata if available.",
        "",
        "## JSON",
        "",
        "```json",
        json.dumps(design, indent=2, ensure_ascii=False),
        "```",
        "",
    ]
    write_text(C7_AUDIT / "C7_INTEGRATED_RLEM_DESIGN.md", "\n".join(md))

    plan = {
        "stage": "C7 staged training plan",
        "official_val_used": False,
        "training_started": False,
        "phases": {
            "C7_A_source_audit_design": {
                "status": "this_step",
                "allowed": ["source audit", "tensor map", "integrated design", "training plan"],
                "forbidden": ["training", "official val", "NMS/evaluator changes"],
            },
            "C7_B_frozen_backbone_head_training": {
                "requires_authorization": True,
                "freeze": ["CONQUER backbone", "QDF", "QAL", "original VR head", "original ML head"],
                "train": ["moment-aware VR residual head", "partial relevance MIL head", "proposal confidence head", "hard-negative residual objective"],
                "data": "train_fit/train_calib only",
                "acceptance": [
                    "delta_vs_C6_B2 train_calib 0.7-r1 > 0",
                    "R@5/R@10 non-collapse",
                    "enabled/non-noop integrated contribution > 0",
                    "invalid/duplicate span count remains 0",
                    "positive entries >= exits and hard exits bounded",
                ],
            },
            "C7_C_partial_unfreeze": {
                "allowed_only_if": "C7-B passes train_calib freeze review",
                "unfreeze": ["VR head", "ML head", "optionally late QAL layers"],
                "schedule": "small learning rate, short schedule, strict train_calib selection",
            },
            "C7_D_freeze_review": {
                "allowed_only_if": "train_calib passes",
                "output": "freeze review and manifest only",
                "official_val": "not automatic; requires explicit human authorization",
            },
        },
        "stop_condition": "After this audit/design step, request authorization before implementation/training.",
    }
    write_json(C7_AUDIT / "C7_STAGE_PLAN.json", plan)
    pmd = [
        "# C7 Stage Plan",
        "",
        "## C7-B: Frozen-Backbone Head Training",
        "",
        "- Requires explicit authorization.",
        "- Freeze CONQUER backbone, QDF/QAL, and original VR/ML heads.",
        "- Train only new RLEM heads on train_fit/train_calib.",
        "- No official val.",
        "",
        "Acceptance: delta vs C6-B2 train_calib 0.7-r1 > 0, R@5/R@10 non-collapse, and non-zero integrated contribution.",
        "",
        "## C7-C: Partial Unfreeze",
        "",
        "Only if C7-B passes: unfreeze VR head, ML head, optionally late QAL layers, with small LR and short schedule.",
        "",
        "## C7-D: Freeze Review",
        "",
        "Only train_calib freeze review. Do not run official val automatically.",
        "",
        "```json",
        json.dumps(plan, indent=2, ensure_ascii=False),
        "```",
        "",
    ]
    write_text(C7_AUDIT / "C7_STAGE_PLAN.md", "\n".join(pmd))

    c8 = [
        "# C8 Placeholder: Strong Temporal Prior",
        "",
        "C8 is reserved for GenSpan-lite, event-order temporal prior, and stronger candidate generation.",
        "",
        "Do not start C8 until C7-B/C results are reviewed. Do not train GenSpan-lite here, do not call any text-to-video generator, and do not replace the CONQUER backbone.",
        "",
    ]
    write_text(C7_AUDIT / "C8_PLACEHOLDER_STRONG_TEMPORAL_PRIOR.md", "\n".join(c8))


def main() -> None:
    c6_finalization()
    c7_source_audit()
    c7_design_and_plan()
    print(json.dumps({
        "status": "C6_FINALIZED_C7_AUDIT_DESIGN_READY",
        "official_val_used": False,
        "training_started": False,
        "outputs": [
            str(C6_FINAL / "C6_EXTERNAL_FINALIZATION.md"),
            str(C6_FINAL / "C6_EXTERNAL_STAGE_SUMMARY.md"),
            str(C7_AUDIT / "C7_CONQUER_SOURCE_AUDIT.md"),
            str(C7_AUDIT / "C7_TENSOR_INTERFACE_MAP.md"),
            str(C7_AUDIT / "C7_INTEGRATED_RLEM_DESIGN.md"),
            str(C7_AUDIT / "C7_STAGE_PLAN.md"),
            str(C7_AUDIT / "C8_PLACEHOLDER_STRONG_TEMPORAL_PRIOR.md"),
        ],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
