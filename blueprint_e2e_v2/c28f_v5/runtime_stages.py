"""Six recovery-completion stage handlers for the compact C28F runtime."""

from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .atomic_io import BootstrapLease, atomic_write_bytes, atomic_write_immutable
from .canonical import (
    bytes_sha256,
    canonical_json_bytes,
    read_regular_bytes,
    semantic_sha256,
)
from .constants import CONTROL_ROOT, GOAL_ID, REPORT_ROOT, TEST_ROOT
from .runtime_control import (
    A4EvalForwardControlAdapter,
    F0_A_ANALYZE_ACTION,
    F0_B_ANALYZE_ACTION,
    RUN_F0_B_ACTION,
    attach_f0_b_authority,
    begin_f0_b_running,
    rehydrate_completed_running_proof,
    rehydrate_current_authority,
)
from .runtime_data import (
    ImportedTrainIndex,
    QUERY_BUCKET_RULE_CONTRACT,
    canonical_jsonl,
    derive_completed_forward_rows,
    read_g4_manifest_ids,
)
from .runtime_store import (
    RuntimeControlError,
    RuntimeStore,
    runtime_artifact_path,
)


G4_ROLE_POLICY_LOCK_ACTION = "G4_ROLE_POLICY_LOCK"
G6_VERIFY_F1_ACTION = "G6_VERIFY_F1"
G7_FINALIZE_ACTION = "G7_FINALIZE"

QUERY_CHUNK_SIZE = 8
CORPUS_CHUNK_SIZE = 256
ESTIMATED_ATOMIC_OUTPUT_BYTES = 128 * 1024 * 1024
FROZEN_SUPERSESSION_BRIDGE_SHA256 = (
    "ea22ea92671a2c5aa05c92f4f19db2efa8ad013f0d80249f629e984a867afbac"
)
FROZEN_G1_RECEIPT_SHA256 = (
    "85d1152b3b834bde65d2f96074af77dd5f883ac1a05598748b742434ee9a4fa2"
)
FROZEN_G2_RECEIPT_SHA256 = (
    "4aaec261cbf2c2ee29f2e6ce49731dfe4a48f1edac0dbda5dd5c9e3674c423b7"
)
FROZEN_TRAIN_INDEX_RECEIPT_SHA256 = (
    "0771cac8da99eb99c4e92d567d78128f147ad95d5bab9d9ab0ccf9ae89d3542a"
)
FROZEN_F0_A_MATERIAL_SHA256 = (
    "c756b9dfcc4a9f877dc9890eba2b0e953ddad0bc93585ed78f072762ec0655ac"
)
FROZEN_F0_A_COMPLETION_RECEIPT_SHA256 = (
    "c2a26e76bf45f4b8974e8536bb6f49fabd1b45a3bacce282a9569d660f82ba55"
)

ROLE_NAMES = (
    "train_fit_route_dev",
    "train_fit_mechanism_dev",
    "train_fit_confirm",
    "train_fit_calibration",
    "train_fit_core",
)
G3_OUTPUTS = (
    "f0_forensics/f0_a_forensics_report.json",
    "f0_forensics/f0_a_query_metrics.jsonl",
    "f0_forensics/rank_transitions.parquet",
    "f0_forensics/rank_transitions_contract.json",
    "f0_forensics/bucket_report.json",
    "f0_forensics/bucket_report.md",
    "f0_forensics/teacher_shadow_report.json",
    "f0_forensics/teacher_shadow_report.md",
    "f0_forensics/f0_a_eval_ledger_entry.json",
)
G4_OUTPUTS = tuple(
    f"f1_protocol/role_manifests/{role}.json" for role in ROLE_NAMES
) + (
    "f1_protocol/role_manifest_lock.json",
    "f1_protocol/cluster_power_audit.json",
    "f1_protocol/FORMAL_DATA_POLICY.json",
    "f1_protocol/protected_entity_audit.json",
    "f1_protocol/protected_id_mapping.jsonl",
    "f1_protocol/protected_video_ids.txt",
    "f1_protocol/protected_id_projection_manifest.json",
    "f1_protocol/train_fit_desc_to_gt_video.jsonl",
    "f1_protocol/train_fit_desc_to_gt_video.source_receipt.json",
    "f1_protocol/g4_control_binding.json",
)
G5_OUTPUTS = (
    "f0_forensics/f0_b_forensics_report.json",
    "f0_forensics/f0_b_query_metrics.jsonl",
    "f0_forensics/split_shift_report.json",
    "f0_forensics/split_shift_report.md",
    "f0_forensics/routing_decision.json",
    "f0_forensics/routing_decision.md",
    "f0_forensics/f0_b_eval_ledger_entry.json",
)
G6_OUTPUTS = (
    "f1_protocol/temporal_oracles.json",
    "f1_protocol/recovery_matrix.json",
    "f1_protocol/schedule_checkpoint_state_test_report.json",
    "f1_protocol/cost_measurement_protocol.json",
    "f1_protocol/superset_scaffold_audit.json",
    "f1_protocol/f0_f1_applicable_regression_suite.json",
    "f1_protocol/f1_protocol_report.final.json",
    "f1_protocol/g6_control_binding.json",
)
G7_OUTPUTS = (
    "f1_protocol/finalization/F0_F1_WINDOW_DECISION.json",
    "f1_protocol/finalization/F0_F1_WINDOW_DECISION.md",
    "f1_protocol/finalization/artifact_manifest.json",
    "f1_protocol/finalization/artifact_manifest.root.sha256",
    "f1_protocol/finalization/PENDING_FINALIZATION.json",
    "f1_protocol/finalization/eval_ledger.jsonl",
    "f1_protocol/finalization/protected_data_access_summary.json",
    "f1_protocol/finalization/CONTINUATION.md",
    "f1_protocol/finalization/NEXT_AUTHORIZATION_REQUEST.json",
)

METRIC_SUFFICIENT_STAT_CONTRACT_BASE = {
    "schema_version": "c28f_v5_runtime_metric_sufficient_stat_contract_v1",
    "source_semantics": "SEALED_RAW_P_FULL_NO_NMS_PLUS_AUTHORIZED_POST_FORWARD_GT",
    "one_pass_chunk_order": "VALIDATED_CONTIGUOUS_RANGE_INDEX_ASC",
    "maximum_simultaneously_loaded_raw_chunk_count": 1,
    "joint_probability_mass_source": "RAW_PRE_NMS_PROPOSALS",
    "joint_probability_mass_per_query": 1.0,
    "frozen_nms": {
        "application_count": 1,
        "scope": "per_video",
        "iou_operator": ">",
        "iou_threshold": 0.7,
        "max_keep": 100,
        "sort_order": [
            "score_desc",
            "video_id_asc",
            "start_sec_asc",
            "end_sec_asc",
            "source_ordinal_asc",
        ],
    },
    "vcmr_ks": [1, 5, 10, 100],
    "vcmr_iou_thresholds": [0.5, 0.7],
    "ranking_stages": ["pooled", "late", "final"],
    "ranking_cut_positions_1_based": [1, 10, 100],
    "ground_truth_join_phase": "POST_FORWARD_INDEXED_PREAD_ONLY",
    "forbidden_derived_row_fields": [
        "query_text",
        "query_tokens",
        "ground_truth_span",
        "raw_joint_proposals",
        "raw_joint_probabilities",
        "raw_or_nms_spans",
    ],
    "one_logical_evaluation": True,
}
METRIC_SUFFICIENT_STAT_CONTRACT = {
    **METRIC_SUFFICIENT_STAT_CONTRACT_BASE,
    "contract_sha256": semantic_sha256(METRIC_SUFFICIENT_STAT_CONTRACT_BASE),
}


def _canonical_file(value: Any) -> bytes:
    return canonical_json_bytes(value) + b"\n"


def _read_json(path: Path, *, max_bytes: int = 1024 * 1024 * 1024) -> dict[str, Any]:
    data, _record = read_regular_bytes(
        path, max_bytes=max_bytes, require_nlink_one=True
    )
    try:
        value = json.loads(data.decode("utf-8", "strict"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeControlError(f"invalid canonical runtime JSON: {path}") from exc
    if not isinstance(value, dict) or data != _canonical_file(value):
        raise RuntimeControlError(f"non-canonical runtime JSON: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    data, _record = read_regular_bytes(
        path, max_bytes=1024 * 1024 * 1024, require_nlink_one=True
    )
    rows: list[dict[str, Any]] = []
    for raw in data.splitlines(keepends=True):
        if not raw.endswith(b"\n"):
            raise RuntimeControlError(f"unterminated JSONL row: {path}")
        value = json.loads(raw.decode("utf-8", "strict"))
        if not isinstance(value, dict) or raw != _canonical_file(value):
            raise RuntimeControlError(f"non-canonical JSONL row: {path}")
        rows.append(value)
    return rows


def _stage_root(store: RuntimeStore, stage_id: str) -> Path:
    return store.root / "artifacts" / stage_id


def _stage_receipt(store: RuntimeStore, stage_id: str) -> dict[str, Any]:
    state = store.read_state()
    pointer = state.get("stage_receipts", {}).get(stage_id)
    if not isinstance(pointer, dict):
        raise RuntimeControlError(f"committed stage receipt is absent: {stage_id}")
    receipt = _read_json(Path(pointer["path"]))
    if (
        receipt.get("receipt_sha256") != pointer.get("receipt_sha256")
        or receipt.get("receipt_sha256")
        != semantic_sha256(receipt, ("receipt_sha256",))
    ):
        raise RuntimeControlError(f"committed stage receipt drift: {stage_id}")
    return receipt


def _completed_material(
    state: Mapping[str, Any], token_id: str
) -> tuple[Path, dict[str, Any]]:
    current = state.get("evals", {}).get(token_id, {}).get("current")
    pointer = current.get("completed_material") if isinstance(current, dict) else None
    if (
        not isinstance(pointer, dict)
        or set(pointer) != {"path", "material_binding_sha256"}
    ):
        raise RuntimeControlError(f"completed material is absent: {token_id}")
    material = _read_json(Path(pointer["path"]))
    if (
        material.get("material_binding_sha256")
        != pointer["material_binding_sha256"]
        or material.get("material_binding_sha256")
        != semantic_sha256(material, ("material_binding_sha256",))
    ):
        raise RuntimeControlError(f"completed material pointer drift: {token_id}")
    return Path(pointer["path"]), material


def _completed_material_path(state: Mapping[str, Any], token_id: str) -> Path:
    path, _material = _completed_material(state, token_id)
    return path


def _completed_identity(
    state: Mapping[str, Any], token_id: str
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    material_path, material = _completed_material(state, token_id)
    validation = material["completed_forward_validation_receipt"]
    projection = material["completed_result_verification_projection"]
    current = state["evals"][token_id]["current"]
    authority_record = _read_json(Path(current["authority_record_path"]))
    completion = _read_json(material_path.parent / "completion.json")
    binding = authority_record["controller_binding"]
    query_identity = projection.get("manifest", {}).get("query_identity")
    if (
        validation.get("validation_receipt_sha256")
        != semantic_sha256(validation, ("validation_receipt_sha256",))
        or projection.get("projection_binding_sha256")
        != semantic_sha256(projection, ("projection_binding_sha256",))
        or authority_record.get("record_sha256")
        != semantic_sha256(authority_record, ("record_sha256",))
        or authority_record.get("record_sha256")
        != current.get("authority_record_sha256")
        or not isinstance(query_identity, dict)
        or query_identity.get("query_count") != validation.get("query_count")
        or completion.get("receipt_sha256")
        != semantic_sha256(completion, ("receipt_sha256",))
        or completion.get("receipt_sha256")
        != validation.get("completion_summary", {}).get("receipt_sha256")
        or completion.get("token_id") != token_id
        or completion.get("eval_id") != validation.get("eval_id")
        or completion.get("run_id") != validation.get("run_id")
    ):
        raise RuntimeControlError(f"completed identity binding drift: {token_id}")
    execution_binding = {
        "checkpoint_sha256": binding["forward_manifest_sha256"]
        if "checkpoint_sha256" not in authority_record["control_chain_values"]
        else authority_record["control_chain_values"]["checkpoint_sha256"],
        "corpus_manifest_sha256": binding["corpus_identity_sha256"],
        "evaluator_sha256": authority_record["control_chain_values"][
            "evaluator_sha256"
        ],
        "cursor_contract_sha256": semantic_sha256(
            {
                "schema_version": "c28f_v5_runtime_cursor_contract_v1",
                "rule": "MONOTONIC_SAME_EVAL_ID_RESUME_ONLY",
            }
        ),
        "chunks_contract_sha256": semantic_sha256(
            {
                "schema_version": "c28f_v5_runtime_chunks_contract_v1",
                "rule": "ORDERED_CONTIGUOUS_QUERY_RANGES",
                "query_chunk_size": QUERY_CHUNK_SIZE,
            }
        ),
        "artifacts_contract_sha256": semantic_sha256(
            {
                "schema_version": "c28f_v5_runtime_artifacts_contract_v1",
                "rule": "IMMUTABLE_SEALED_RAW_P_FULL_NO_NMS",
            }
        ),
    }
    return material, validation, {
        "material_path": str(material_path),
        "material_binding_sha256": material["material_binding_sha256"],
        "authority_record_sha256": authority_record["record_sha256"],
        "completion": completion,
        "query_manifest_sha256": binding["split_manifest_sha256"],
        "query_rows_sha256": binding["query_rows_sha256"],
        "corpus_manifest_sha256": binding["corpus_identity_sha256"],
        "execution_binding": execution_binding,
        "projection": projection,
        "binding": binding,
    }


def _event_for_stage_receipt(
    runtime: RuntimeStore, receipt: Mapping[str, Any]
) -> dict[str, Any]:
    matches = [
        event
        for event in _read_jsonl(runtime.events_path)
        if event.get("artifact_sha256", {}).get("STAGE_RECEIPT.json")
        == receipt.get("receipt_sha256")
    ]
    if len(matches) != 1:
        raise RuntimeControlError("committed runtime stage event is not unique")
    return matches[0]


def _completed_query_role_binding(
    *,
    runtime: RuntimeStore,
    token_id: str,
    query_role: str,
    validation: Mapping[str, Any],
    identity: Mapping[str, Any],
) -> dict[str, Any]:
    query_identity = identity["projection"]["manifest"]["query_identity"]
    if token_id == "F0_A":
        completion = identity["completion"]
        manifest_path = "frozen_splits/calib_select_desc_ids.txt"
        manifest_file_sha256 = query_identity["query_keys_sha256"]
        commit_kind = "IMPORTED_COMPLETED_FORWARD_SPLIT_BINDING"
        committed_transaction_id = completion["transaction_id"]
        committed_state_sha256 = completion["state_sha256"]
        committed_event_sha256 = completion["event_sha256"]
        stage_action = "RUN_F0_A"
        stage_commit_receipt_sha256 = completion["receipt_sha256"]
    else:
        receipt = _stage_receipt(runtime, "g4")
        event = _event_for_stage_receipt(runtime, receipt)
        role_path = "f1_protocol/role_manifests/train_fit_route_dev.json"
        manifest_path = role_path
        manifest_file_sha256 = receipt["output_hashes"][role_path]
        commit_kind = "RUNTIME_STAGE_RECEIPT"
        committed_transaction_id = event["transaction_id"]
        committed_state_sha256 = event["next_state_sha256"]
        committed_event_sha256 = event["event_sha256"]
        stage_action = G4_ROLE_POLICY_LOCK_ACTION
        stage_commit_receipt_sha256 = receipt["receipt_sha256"]
    base = {
        "schema_version": "c28f_a4_completed_eval_query_role_binding_v1",
        "binding_kind": commit_kind,
        "role": query_role,
        "manifest_path": manifest_path,
        "manifest_file_sha256": manifest_file_sha256,
        "manifest_sha256": identity["query_manifest_sha256"],
        "query_count": validation["query_count"],
        "desc_id_lines_sha256": query_identity["query_keys_sha256"],
        "commit_kind": commit_kind,
        "committed_transaction_id": committed_transaction_id,
        "committed_state_sha256": committed_state_sha256,
        "committed_event_sha256": committed_event_sha256,
        "stage_action": stage_action,
        "stage_commit_receipt_sha256": stage_commit_receipt_sha256,
    }
    return {**base, "binding_sha256": semantic_sha256(base)}


def _analyze_f0_rows(
    *,
    runtime: RuntimeStore,
    token_id: str,
    query_role: str,
    rows: Sequence[Mapping[str, Any]],
    lineage: Mapping[str, Any],
    state: Mapping[str, Any],
    material: Mapping[str, Any],
    validation: Mapping[str, Any],
    identity: Mapping[str, Any],
) -> dict[str, Any]:
    from . import a4_stages

    evaluation = a4_stages._evaluate_completed_f0_rows(rows)
    capability_sha = str(validation["capability_id"])
    gate = a4_stages._frozen_retrieval_gate_result(
        retrieval=evaluation,
        query_manifest_sha256=identity["query_manifest_sha256"],
        completed_eval_capability_sha256=capability_sha,
    )
    bucket_report, query_metric_rows = a4_stages._build_f0_bucket_report(
        source_rows=rows,
        metric_rows=evaluation["per_query"],
        bucket_rule_contract=QUERY_BUCKET_RULE_CONTRACT,
        completed_eval_capability_sha256=capability_sha,
        query_manifest_sha256=identity["query_manifest_sha256"],
    )
    comparison = a4_stages._f0_comparison_summary(
        query_role=query_role,
        evaluation=evaluation,
        bucket_report=bucket_report,
        execution_binding=identity["execution_binding"],
    )
    role_binding = _completed_query_role_binding(
        runtime=runtime,
        token_id=token_id,
        query_role=query_role,
        validation=validation,
        identity=identity,
    )
    artifact_binding_base = {
        "schema_version": "c28f_v5_runtime_completed_eval_artifact_binding_v1",
        "token_id": token_id,
        "material_binding_sha256": material["material_binding_sha256"],
        "validation_receipt_sha256": validation["validation_receipt_sha256"],
        "completion_receipt_sha256": identity["completion"]["receipt_sha256"],
        "projection_binding_sha256": identity["projection"][
            "projection_binding_sha256"
        ],
        "authority_record_sha256": identity["authority_record_sha256"],
        "result_chunk_set_sha256": validation["result_chunk_set_sha256"],
        "derivation_lineage_sha256": lineage["lineage_sha256"],
    }
    safety_counts = dict(a4_stages.F0_COMPLETED_EVAL_SAFETY_COUNTS)
    return {
        "evaluation": evaluation,
        "gate": gate,
        "bucket_report": bucket_report,
        "query_metric_rows": query_metric_rows,
        "query_metrics_bytes": canonical_jsonl(query_metric_rows),
        "comparison": comparison,
        "validation": validation,
        "identity": identity,
        "lineage": dict(lineage),
        "role_binding": role_binding,
        "completed_eval_artifact_binding_sha256": semantic_sha256(
            artifact_binding_base
        ),
        "metric_sufficient_stat_contract_sha256": (
            METRIC_SUFFICIENT_STAT_CONTRACT["contract_sha256"]
        ),
        "safety_counts": safety_counts,
        "state": state,
    }


def _build_f0_report(
    *,
    token_id: str,
    query_role: str,
    rows: Sequence[Mapping[str, Any]],
    built: Mapping[str, Any],
    artifact_commitments: Mapping[str, Any],
    predecessor_f0_a_binding: Mapping[str, Any] | None = None,
    split_shift_report_sha256: str | None = None,
    routing_status: str | None = None,
) -> dict[str, Any]:
    from . import a4_stages

    validation = built["validation"]
    identity = built["identity"]
    evaluation = built["evaluation"]
    gate = built["gate"]
    role_binding = built["role_binding"]
    required_count_key = (
        "required_query_count"
        if token_id == "F0_A"
        else "required_committed_role_query_count"
    )
    report_base: dict[str, Any] = {
        "schema_version": (
            a4_stages.F0_A_FORENSICS_REPORT_SCHEMA
            if token_id == "F0_A"
            else a4_stages.F0_B_FORENSICS_REPORT_SCHEMA
        ),
        "status": (
            "VALIDATED_COMPLETE_TEACHER_FREE_FORENSICS"
            if token_id == "F0_A"
            else "VALIDATED_COMPLETE_EXPOSURE_AWARE_FORENSICS"
        ),
        "goal_id": GOAL_ID,
        "attempt_id": built["state"]["attempt_id"],
        "stage": "G3_F0_A" if token_id == "F0_A" else "G5_F0_B",
        "token_id": token_id,
        "eval_id": validation["eval_id"],
        "run_id": validation["run_id"],
        "query_role": query_role,
        "query_manifest_sha256": identity["query_manifest_sha256"],
        "corpus_manifest_sha256": identity["corpus_manifest_sha256"],
        "query_count": len(rows),
        "query_completeness": {
            required_count_key: role_binding["query_count"],
            "complete_query_count": len(rows),
            "ratio": len(rows) / float(role_binding["query_count"]),
            "silent_skip_count": role_binding["query_count"] - len(rows),
        },
        "committed_query_role_binding": dict(role_binding),
        "completed_eval_capability_sha256": validation["capability_id"],
        "completed_eval_artifact_binding_sha256": built[
            "completed_eval_artifact_binding_sha256"
        ],
        "completed_eval_source_artifact_lineage": dict(built["lineage"]),
        "cross_split_execution_invariants": built["comparison"][
            "cross_split_execution_invariants"
        ],
        "metric_sufficient_stat_contract_sha256": built[
            "metric_sufficient_stat_contract_sha256"
        ],
        "retrieval_metrics": evaluation["retrieval_metrics"],
        "joint_retrieval_metrics": evaluation["joint_retrieval_metrics"],
        "vcmr_metrics": evaluation["vcmr_metrics"],
        "error_metrics": evaluation["error_metrics"],
        "gate_metrics": evaluation["gate_metrics"],
        "rank_transition_summary": evaluation["rank_transition_summary"],
        "margin_diagnostics": evaluation["margin_diagnostics"],
        "modality_mask_diagnostics": evaluation["modality_mask_diagnostics"],
        "video_distribution_diagnostics": evaluation[
            "video_distribution_diagnostics"
        ],
        "denominators": evaluation["denominators"],
        "frozen_gate_result": gate,
        "frozen_gate_result_sha256": gate["gate_result_sha256"],
        "earliest_failing_stage": gate["earliest_failing_stage"],
        "comparison_summary": built["comparison"],
        "comparison_summary_sha256": semantic_sha256(built["comparison"]),
        "teacher_shadow_status": "unavailable",
        "intrinsic_diagnostics": {
            "status": "unavailable",
            "reason": "FROZEN_C28E_MODEL_HAS_NO_STRICT_INTRINSIC_EXPORT",
            "fabricated_value_count": 0,
        },
        "calibration_status": {
            "status": "unavailable",
            "reason": "NO_COMPLIANT_COMMITTED_PREFIT_TEMPERATURE_CAPABILITY",
            "posthoc_fit_performed": False,
        },
        "artifact_commitments": dict(artifact_commitments),
        "safety_counts": dict(built["safety_counts"]),
        "threshold_changed_after_observation": False,
    }
    if token_id == "F0_B":
        if (
            predecessor_f0_a_binding is None
            or split_shift_report_sha256 is None
            or routing_status not in {"ROUTING_CONSISTENT", "ROUTING_AMBIGUOUS"}
        ):
            raise RuntimeControlError("F0-B report closure inputs are incomplete")
        report_base.update(
            {
                "bucket_report": built["bucket_report"],
                "predecessor_f0_a_binding": dict(predecessor_f0_a_binding),
                "split_shift_report_sha256": split_shift_report_sha256,
                "routing_status": routing_status,
            }
        )
    report = {**report_base, "report_sha256": semantic_sha256(report_base)}
    a4_stages._exact_mapping(
        report,
        (
            a4_stages.F0_A_FORENSICS_REPORT_FIELDS
            if token_id == "F0_A"
            else a4_stages.F0_B_FORENSICS_REPORT_FIELDS
        ),
        f"{token_id} runtime report field set drift",
    )
    return report


def _build_f0_ledger(
    *,
    token_id: str,
    report: Mapping[str, Any],
    report_bytes: bytes,
    built: Mapping[str, Any],
    committed_f0_a_report_sha256: str | None = None,
    routing_status: str | None = None,
) -> dict[str, Any]:
    state = built["state"]
    validation = built["validation"]
    completion = built["identity"]["completion"]
    current = state["evals"][token_id]["current"]
    cursor = current.get("cursor")
    if not isinstance(cursor, dict) or not isinstance(cursor.get("receipt_sha256"), str):
        raise RuntimeControlError(f"{token_id} completed cursor receipt is absent")
    base: dict[str, Any] = {
        "schema_version": f"c28f_{token_id.lower()}_eval_ledger_entry_v1",
        "status": "ONE_LOGICAL_EVAL_COMPLETED_NONREPLAYABLE",
        "goal_id": GOAL_ID,
        "attempt_id": state["attempt_id"],
        "token_id": token_id,
        "eval_id": validation["eval_id"],
        "run_id": validation["run_id"],
        "logical_eval_count": 1,
        "completed_transaction_id": completion["transaction_id"],
        "completed_state_sha256": completion["state_sha256"],
        "completed_event_sha256": completion["event_sha256"],
        "phase_receipt_sha256": completion["receipt_sha256"],
        "cursor_receipt_sha256": cursor["receipt_sha256"],
        "result_chunk_set_sha256": validation["result_chunk_set_sha256"],
        "artifact_index_sha256": validation["validation_receipt_sha256"],
        "completed_eval_capability_sha256": validation["capability_id"],
        "report_sha256": report["report_sha256"],
        "report_file_sha256": bytes_sha256(report_bytes),
        "safety_counts": dict(built["safety_counts"]),
        "resume_rule": "SAME_EVAL_ID_CURSOR_ONLY_NO_SECOND_SELECTION_EVAL",
    }
    if token_id == "F0_B":
        if committed_f0_a_report_sha256 is None or routing_status is None:
            raise RuntimeControlError("F0-B ledger closure inputs are incomplete")
        base.update(
            {
                "committed_f0_a_report_sha256": committed_f0_a_report_sha256,
                "routing_status": routing_status,
            }
        )
    return {**base, "ledger_entry_sha256": semantic_sha256(base)}


def run_f0_a_analyze(store: RuntimeStore | None = None) -> dict[str, Any]:
    runtime = store or RuntimeStore()
    state = runtime.read_state()
    if state.get("status") != "F0_A_COMPLETED" or state.get("next_action") != F0_A_ANALYZE_ACTION:
        raise RuntimeControlError("F0-A analysis is not the unique next action")

    def mutate(next_state: dict[str, Any]) -> None:
        next_state["status"] = "F0_A_ANALYZED"
        next_state["next_action"] = G4_ROLE_POLICY_LOCK_ACTION

    recovered = runtime.recover_prepared_stage(
        stage_id="g3",
        action=F0_A_ANALYZE_ACTION,
        expected_outputs=G3_OUTPUTS,
        mutate=mutate,
    )
    if recovered is not None:
        report = _read_json(
            _stage_root(runtime, "g3")
            / "f0_forensics/f0_a_forensics_report.json"
        )
        return {
            "status": recovered["status"],
            "next_action": recovered["next_action"],
            "report_sha256": report["report_sha256"],
            "earliest_failing_stage": report["earliest_failing_stage"],
            "state_sha256": recovered["state_sha256"],
            "recovered_prepared_stage": True,
        }
    material, validation, identity = _completed_identity(state, "F0_A")
    desc_ids = [int(value) for value in identity["projection"]["query_keys"]]
    train = ImportedTrainIndex(runtime)
    gt_projection = train.project(
        desc_ids,
        purpose="POST_FORWARD_EVAL_STATS",
        projection_scope="F0_A_POST_FORWARD_STATS",
    )
    rows, lineage = derive_completed_forward_rows(
        Path(identity["material_path"]), gt_projection
    )
    built = _analyze_f0_rows(
        runtime=runtime,
        token_id="F0_A",
        query_role="calib_select",
        rows=rows,
        lineage=lineage,
        state=state,
        material=material,
        validation=validation,
        identity=identity,
    )
    from . import a4_stages

    parquet, parquet_contract = a4_stages._rank_transitions_parquet(
        per_query=built["evaluation"]["per_query"],
        source_binding_sha256=lineage["retrieval_rows_sha256"],
    )
    teacher_report = a4_stages._f0_teacher_shadow_report(
        query_manifest_sha256=built["identity"]["query_manifest_sha256"],
        completed_eval_capability_sha256=str(
            built["validation"]["capability_id"]
        ),
    )
    query_metrics_bytes = built["query_metrics_bytes"]
    parquet_contract_bytes = _canonical_file(parquet_contract)
    bucket_report_bytes = _canonical_file(built["bucket_report"])
    bucket_markdown = a4_stages._render_summary_markdown(
        title="C28F F0-A Frozen Bucket Report",
        authoritative_content_sha256=bytes_sha256(bucket_report_bytes),
        rows=(
            ("status", built["bucket_report"]["status"]),
            ("query_count", len(rows)),
            ("dimension_count", built["bucket_report"]["dimension_count"]),
            ("teacher_rescue_bucket_status", "unavailable"),
        ),
    )
    teacher_report_bytes = _canonical_file(teacher_report)
    teacher_markdown = a4_stages._render_summary_markdown(
        title="C28F F0-A Teacher Shadow Report",
        authoritative_content_sha256=bytes_sha256(teacher_report_bytes),
        rows=(
            ("status", teacher_report["status"]),
            ("reason", teacher_report["reason"]),
            ("teacher_model_forward_count", 0),
            ("main_teacher_free_route_affected", False),
        ),
    )
    artifact_commitments = {
        "query_metrics_rows_sha256": bytes_sha256(
            canonical_json_bytes(list(built["query_metric_rows"]))
        ),
        "query_metrics_file_sha256": bytes_sha256(query_metrics_bytes),
        "rank_transitions_parquet_file_sha256": bytes_sha256(parquet),
        "rank_transitions_contract_sha256": parquet_contract["contract_sha256"],
        "rank_transitions_contract_file_sha256": bytes_sha256(
            parquet_contract_bytes
        ),
        "bucket_report_sha256": built["bucket_report"]["bucket_report_sha256"],
        "bucket_report_file_sha256": bytes_sha256(bucket_report_bytes),
        "bucket_report_markdown_file_sha256": bytes_sha256(bucket_markdown),
        "teacher_shadow_report_sha256": teacher_report["report_sha256"],
        "teacher_shadow_report_file_sha256": bytes_sha256(teacher_report_bytes),
        "teacher_shadow_markdown_file_sha256": bytes_sha256(teacher_markdown),
        "metric_sufficient_stat_contract_sha256": built[
            "metric_sufficient_stat_contract_sha256"
        ],
        "committed_query_role_binding_sha256": built["role_binding"][
            "binding_sha256"
        ],
        "completed_eval_source_artifact_lineage_sha256": lineage[
            "lineage_sha256"
        ],
    }
    report = _build_f0_report(
        token_id="F0_A",
        query_role="calib_select",
        rows=rows,
        built=built,
        artifact_commitments=artifact_commitments,
    )
    report_bytes = _canonical_file(report)
    ledger = _build_f0_ledger(
        token_id="F0_A",
        report=report,
        report_bytes=report_bytes,
        built=built,
    )
    outputs = {
        "f0_forensics/f0_a_forensics_report.json": report_bytes,
        "f0_forensics/f0_a_query_metrics.jsonl": query_metrics_bytes,
        "f0_forensics/rank_transitions.parquet": parquet,
        "f0_forensics/rank_transitions_contract.json": parquet_contract_bytes,
        "f0_forensics/bucket_report.json": bucket_report_bytes,
        "f0_forensics/bucket_report.md": bucket_markdown,
        "f0_forensics/teacher_shadow_report.json": teacher_report_bytes,
        "f0_forensics/teacher_shadow_report.md": teacher_markdown,
        "f0_forensics/f0_a_eval_ledger_entry.json": _canonical_file(ledger),
    }

    committed = runtime.commit_stage(
        stage_id="g3",
        action=F0_A_ANALYZE_ACTION,
        outputs=outputs,
        input_hashes={
            "completed_material": material["material_binding_sha256"],
            "validation_receipt": validation["validation_receipt_sha256"],
            "train_projection": gt_projection["record_sha256"],
            "retrieval_rows": lineage["retrieval_rows_sha256"],
            "frozen_gate": built["gate"]["gate_result_sha256"],
        },
        safety_counts={
            "model_forward_evaluations": 1,
            "protected_content_opens": 0,
            "real_data_optimizer_updates": 0,
            "teacher_forward_count": 0,
            "gt_support_count": 0,
        },
        mutate=mutate,
    )
    return {
        "status": committed["status"],
        "next_action": committed["next_action"],
        "report_sha256": report["report_sha256"],
        "earliest_failing_stage": built["gate"]["earliest_failing_stage"],
        "state_sha256": committed["state_sha256"],
    }


def _g2_power_evidence(
    runtime: RuntimeStore,
    state: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, str]]:
    bridge = _read_json(
        runtime.root / "control_bridge" / "supersession_bridge.json"
    )
    if bridge.get("bridge_sha256") != semantic_sha256(
        bridge, ("bridge_sha256",)
    ) or bridge.get("bridge_sha256") != FROZEN_SUPERSESSION_BRIDGE_SHA256:
        raise RuntimeControlError("G2 power bridge binding drift")
    output_hashes = bridge["gate_projection"]["G2"]["output_hashes"]
    root = REPORT_ROOT / str(state["attempt_id"]) / "stage_outputs"
    rows_path = root / "f0_forensics/g2_power_replay_rows.jsonl"
    reference_path = root / "f0_forensics/g2_historical_power_reference.json"
    rows_data, _rows_record = read_regular_bytes(
        rows_path, max_bytes=1024 * 1024 * 1024, require_nlink_one=True
    )
    reference_data, _reference_record = read_regular_bytes(
        reference_path, max_bytes=128 * 1024 * 1024, require_nlink_one=True
    )
    if (
        bytes_sha256(rows_data)
        != output_hashes["f0_forensics/g2_power_replay_rows.jsonl"]
        or bytes_sha256(reference_data)
        != output_hashes["f0_forensics/g2_historical_power_reference.json"]
    ):
        raise RuntimeControlError("committed G2 power evidence file drift")
    rows = _read_jsonl(rows_path)
    reference = _read_json(reference_path)
    return rows, reference, {
        "rows_file_sha256": bytes_sha256(rows_data),
        "reference_file_sha256": bytes_sha256(reference_data),
    }


def run_g4_role_policy_lock(store: RuntimeStore | None = None) -> dict[str, Any]:
    runtime = store or RuntimeStore()
    state = runtime.read_state()
    if (
        state.get("status") != "F0_A_ANALYZED"
        or state.get("next_action") != G4_ROLE_POLICY_LOCK_ACTION
    ):
        raise RuntimeControlError("G4 role-policy lock is not the unique next action")

    def mutate(next_state: dict[str, Any]) -> None:
        next_state["status"] = "ROLE_POLICY_LOCKED"
        next_state["next_action"] = RUN_F0_B_ACTION
        workflow = next_state.get("workflow_control")
        if isinstance(workflow, dict):
            workflow["f0_b_or_f2_entry_allowed"] = True

    recovered = runtime.recover_prepared_stage(
        stage_id="g4",
        action=G4_ROLE_POLICY_LOCK_ACTION,
        expected_outputs=G4_OUTPUTS,
        mutate=mutate,
    )
    if recovered is not None:
        role_lock = _read_json(
            _stage_root(runtime, "g4")
            / "f1_protocol/role_manifest_lock.json"
        )
        formal_policy = _read_json(
            _stage_root(runtime, "g4")
            / "f1_protocol/FORMAL_DATA_POLICY.json"
        )
        route_manifest = _read_json(
            _stage_root(runtime, "g4")
            / "f1_protocol/role_manifests/train_fit_route_dev.json"
        )
        return {
            "status": recovered["status"],
            "next_action": recovered["next_action"],
            "role_lock_sha256": role_lock["lock_sha256"],
            "formal_policy_sha256": formal_policy["policy_sha256"],
            "route_query_count": route_manifest["query_count"],
            "state_sha256": recovered["state_sha256"],
            "recovered_prepared_stage": True,
        }
    manifest_ids = read_g4_manifest_ids()
    train_index = ImportedTrainIndex(runtime)
    protected_projection = train_index.project(
        manifest_ids["protected_desc_ids"],
        purpose="PROTECTED_ID_MAPPING_ONLY",
        projection_scope="G4_PROTECTED_ID_MAPPING",
    )
    train_fit_projection = train_index.project(
        manifest_ids["train_fit_desc_ids"],
        purpose="TRAIN_FIT_ROLE_MAPPING",
        projection_scope="G4_TRAIN_FIT_ROLE_MAPPING",
    )
    from . import a4_roles

    projected_records = [
        a4_roles.RoleRecord(
            desc_id=int(record["desc_id"]),
            gt_video_id=str(record["vid_name"]),
        )
        for record in train_fit_projection["records"]
    ]
    power_rows, reference, power_files = _g2_power_evidence(runtime, state)
    reference_by_desc = a4_roles._normalize_power_replay_rows(
        power_rows,
        query_manifest_sha256=reference["query_manifest_sha256"],
        replay_packet_sha256=reference["replay_packet_sha256"],
    )
    reference_population = a4_roles.validate_historical_power_reference_population(
        reference
    )
    assignments, power_report = a4_roles._power_adaptive_assignments(
        projected_records,
        manifest_ids["train_fit_desc_ids"],
        reference_by_desc=reference_by_desc,
        reference_population=reference_population,
    )
    if power_report.get("status") != "PASS":
        raise RuntimeControlError("frozen G4 power audit is insufficient")
    power_report_sha = semantic_sha256(power_report)
    assignment_contract_base = {
        "algorithm": a4_roles.POWER_ADAPTIVE_ASSIGNMENT_ALGORITHM,
        "seed": a4_roles.ASSIGNMENT_SEED,
        "cluster_order": "sha256(seed_ascii + NUL + video_id_utf8),video_id_utf8",
        "cluster_atomic": True,
        "candidate_membership_source": power_report[
            "membership_assignment_inputs"
        ],
        "historical_reference_population_sha256": reference_population[
            "reference_population_sha256"
        ],
        "historical_outcomes_select_identity_order_or_within_prefix_members": False,
        "cross_role_historical_outcomes_affect_lane_identity_order_or_members": False,
        "historical_outcomes_may_determine_preregistered_prefix_length": True,
        "power_expansion_lane_algorithm": a4_roles.POWER_EXPANSION_LANE_ALGORITHM,
        "membership_lock_timing": "ALL_PREFIX_EXPANSION_BEFORE_ROLE_MANIFEST_LOCK",
        "power_expansion_report_sha256": power_report_sha,
    }
    assignment_contract = {
        **assignment_contract_base,
        "assignment_contract_sha256": semantic_sha256(assignment_contract_base),
    }
    manifests = a4_roles.build_role_manifests(
        assignments,
        train_fit_manifest_sha256=manifest_ids[
            "train_fit_manifest_file_sha256"
        ],
        assignment_source_sha256=train_fit_projection["records_sha256"],
        producer_control_binding_sha256=state["state_sha256"],
        assignment_contract=assignment_contract,
    )
    protected_video_ids = {
        str(record["vid_name"]) for record in protected_projection["records"]
    }
    candidate_lock = a4_roles._build_role_lock_candidate_from_validated_fields(
        manifests,
        expected_train_fit_desc_ids=manifest_ids["train_fit_desc_ids"],
        projected_records=projected_records,
        protected_desc_ids=set(manifest_ids["protected_desc_ids"]),
        protected_video_ids=protected_video_ids,
        protected_mapping_receipt_sha256=protected_projection[
            "access_receipt_sha256"
        ],
        expected_assignments=assignments,
        expected_assignment_contract=assignment_contract,
    )
    if (
        candidate_lock.get("status")
        != "VALIDATED_FIELDS_CANDIDATE_NOT_CONTROL_COMMITTABLE"
        or candidate_lock.get("missing_set") != []
    ):
        raise RuntimeControlError("G4 role lock candidate did not close")
    lock_base = {
        **{
            key: value
            for key, value in candidate_lock.items()
            if key not in {"status", "lock_sha256"}
        },
        "status": "FSYNCED_RUNTIME_ROLE_POLICY_LOCKED",
        "runtime_pre_state_sha256": state["state_sha256"],
        "role_manifest_paths": {
            role: f"f1_protocol/role_manifests/{role}.json"
            for role in sorted(manifests)
        },
    }
    role_lock = {**lock_base, "lock_sha256": semantic_sha256(lock_base)}
    power_audit_base = {
        "schema_version": "c28f_cluster_power_audit_v2",
        "status": "PASS",
        "role_lock_sha256": role_lock["lock_sha256"],
        "role_counts": role_lock["role_counts"],
        "cluster_atomic": True,
        "power_expansion_report": power_report,
        "power_expansion_report_sha256": power_report_sha,
        "historical_reference_population_sha256": reference_population[
            "reference_population_sha256"
        ],
        "effect_threshold_reduction_count": power_report[
            "effect_threshold_reduction_count"
        ],
    }
    power_audit = {
        **power_audit_base,
        "audit_sha256": semantic_sha256(power_audit_base),
    }
    formal_policy_base = {
        "schema_version": "c28f_formal_data_policy_v1",
        "status": "FSYNCED_RUNTIME_POLICY_LOCKED",
        "policy": "STRICT_CORE_ONLY",
        "U_formal_updates": 17_360,
        "gradient_roles": ["train_fit_core"],
        "calibration_role": "train_fit_calibration",
        "calibration_use": "POSTHOC_TEMPERATURE_ONLY",
        "candidate_and_P0_same_update_budget": True,
        "route_mechanism_confirm_refit_allowed": False,
        "role_lock_sha256": role_lock["lock_sha256"],
        "cluster_power_audit_sha256": power_audit["audit_sha256"],
        "f7_may_change_policy": False,
        "f7_may_change_update_budget": False,
    }
    formal_policy = {
        **formal_policy_base,
        "policy_sha256": semantic_sha256(formal_policy_base),
    }
    if set(manifests) != set(ROLE_NAMES):
        raise RuntimeControlError("G4 did not build the exact five-role set")
    protected_rows = [
        {"desc_id": int(record["desc_id"]), "video_id": str(record["vid_name"])}
        for record in protected_projection["records"]
    ]
    train_fit_rows = [
        {
            "desc_id": int(record["desc_id"]),
            "ground_truth_video_id": str(record["vid_name"]),
        }
        for record in train_fit_projection["records"]
    ]
    protected_mapping_bytes = canonical_jsonl(protected_rows)
    protected_video_bytes = "".join(
        value + "\n" for value in sorted(protected_video_ids)
    ).encode("utf-8")
    train_fit_mapping_bytes = canonical_jsonl(train_fit_rows)
    train_fit_source_receipt_base = {
        "schema_version": "c28f_v5_runtime_train_fit_role_source_receipt_v1",
        "status": "COMPLETED_NONREPLAYABLE_SAME_G4_STAGE",
        "goal_id": GOAL_ID,
        "attempt_id": state["attempt_id"],
        "source_manifest_file_sha256": manifest_ids[
            "train_fit_manifest_file_sha256"
        ],
        "source_manifest_desc_ids_sha256": manifest_ids[
            "train_fit_desc_ids_sha256"
        ],
        "record_count": len(train_fit_rows),
        "mapping_sha256": bytes_sha256(canonical_json_bytes(train_fit_rows)),
        "mapping_file_sha256": bytes_sha256(train_fit_mapping_bytes),
        "train_index_binding_sha256": train_index.index[
            "index_binding_sha256"
        ],
        "train_projection_sha256": train_fit_projection["record_sha256"],
        "train_projection_access_receipt": train_fit_projection[
            "access_receipt"
        ],
        "physical_manifest_content_open_count": 1,
        "physical_train_jsonl_content_open_count": 1,
        "allowed_fields": ["desc_id", "ground_truth_video_id"],
        "full_line_json_decode_count": 0,
        "unauthorized_field_semantic_decode_count": 0,
    }
    train_fit_source_receipt = {
        **train_fit_source_receipt_base,
        "receipt_sha256": semantic_sha256(train_fit_source_receipt_base),
    }
    protected_projection_base = {
        "schema_version": "c28f_protected_video_id_manifest_v1",
        "status": "COMPLETED_NONREPLAYABLE_ID_ONLY",
        "authority_scope": "DESC_ID_TO_VIDEO_ID_ONLY",
        "mapping_rows_sha256": bytes_sha256(canonical_json_bytes(protected_rows)),
        "mapping_file_sha256": bytes_sha256(protected_mapping_bytes),
        "protected_desc_id_count": len(protected_rows),
        "protected_desc_ids_sha256": manifest_ids[
            "protected_desc_ids_sha256"
        ],
        "protected_video_ids": sorted(protected_video_ids),
        "protected_video_id_count": len(protected_video_ids),
        "protected_video_id_lines_sha256": bytes_sha256(protected_video_bytes),
        "protected_video_ids_file_sha256": bytes_sha256(protected_video_bytes),
        "protected_train_projection_sha256": protected_projection[
            "record_sha256"
        ],
        "protected_train_projection_access_receipt": protected_projection[
            "access_receipt"
        ],
        "train_fit_projection_sha256": train_fit_projection["record_sha256"],
        "train_fit_mapping_file_sha256": bytes_sha256(train_fit_mapping_bytes),
        "train_fit_source_receipt_sha256": train_fit_source_receipt[
            "receipt_sha256"
        ],
        "content_open_classification": {
            "categories_mutually_exclusive": True,
            "unauthorized_or_non_id_only_protected_content_open_count": 0,
            "authorized_protected_id_mapping_content_open_count": 5,
            "total_protected_content_open_count": 5,
            "authorized_id_mapping_breakdown": {
                "protected_desc_id_manifest_content_open_count": 2,
                "train_fit_desc_id_manifest_content_open_count": 1,
                "protected_desc_train_jsonl_indexed_pread_content_open_count": 1,
                "train_fit_train_jsonl_indexed_pread_content_open_count": 1,
                "official_val_content_open_count": 0,
            },
        },
        "unknown_value_semantic_decode_count": 0,
        "retained_non_id_value_count": 0,
        "model_forward_count": 0,
        "eval_token_consumed": False,
    }
    protected_projection_manifest = {
        **protected_projection_base,
        "manifest_sha256": semantic_sha256(protected_projection_base),
    }
    protected_audit_base = {
        "schema_version": "c28f_v5_runtime_protected_entity_audit_v1",
        "status": "PASS_ZERO_INTERSECTION",
        "role_lock_sha256": role_lock["lock_sha256"],
        "desc_entity_audit": role_lock["desc_entity_audit"],
        "video_entity_audit": role_lock["video_entity_audit"],
        "authorized_id_mapping_operation_count": 1,
        "authorized_id_mapping_content_open_count": 5,
        "forbidden_protected_content_open_count": 0,
    }
    protected_audit = {
        **protected_audit_base,
        "audit_sha256": semantic_sha256(protected_audit_base),
    }
    g4_binding_base = {
        "schema_version": "c28f_v5_runtime_g4_control_binding_v1",
        "status": "READY_FOR_SINGLE_STAGE_CAS",
        "pre_state_sha256": state["state_sha256"],
        "train_index_binding_sha256": train_index.index[
            "index_binding_sha256"
        ],
        "manifest_content_open_count": manifest_ids[
            "physical_manifest_content_open_count"
        ],
        "indexed_pread_content_open_count": 2,
        "protected_projection_manifest_sha256": protected_projection_manifest[
            "manifest_sha256"
        ],
        "train_fit_source_receipt_sha256": train_fit_source_receipt[
            "receipt_sha256"
        ],
        "role_lock_sha256": role_lock["lock_sha256"],
        "formal_policy_sha256": formal_policy["policy_sha256"],
        "cluster_power_audit_sha256": power_audit["audit_sha256"],
        "g2_power_rows_file_sha256": power_files["rows_file_sha256"],
        "g2_power_reference_file_sha256": power_files[
            "reference_file_sha256"
        ],
    }
    g4_binding = {
        **g4_binding_base,
        "binding_sha256": semantic_sha256(g4_binding_base),
    }
    outputs: dict[str, bytes] = {
        **{
            f"f1_protocol/role_manifests/{role}.json": _canonical_file(manifest)
            for role, manifest in sorted(manifests.items())
        },
        "f1_protocol/role_manifest_lock.json": _canonical_file(role_lock),
        "f1_protocol/FORMAL_DATA_POLICY.json": _canonical_file(formal_policy),
        "f1_protocol/cluster_power_audit.json": _canonical_file(power_audit),
        "f1_protocol/protected_entity_audit.json": _canonical_file(protected_audit),
        "f1_protocol/protected_id_mapping.jsonl": protected_mapping_bytes,
        "f1_protocol/protected_video_ids.txt": protected_video_bytes,
        "f1_protocol/protected_id_projection_manifest.json": _canonical_file(
            protected_projection_manifest
        ),
        "f1_protocol/train_fit_desc_to_gt_video.jsonl": train_fit_mapping_bytes,
        "f1_protocol/train_fit_desc_to_gt_video.source_receipt.json": _canonical_file(
            train_fit_source_receipt
        ),
        "f1_protocol/g4_control_binding.json": _canonical_file(g4_binding),
    }

    committed = runtime.commit_stage(
        stage_id="g4",
        action=G4_ROLE_POLICY_LOCK_ACTION,
        outputs=outputs,
        input_hashes={
            "train_index": train_index.index["index_binding_sha256"],
            "protected_projection": protected_projection["record_sha256"],
            "train_fit_projection": train_fit_projection["record_sha256"],
            "train_fit_source_receipt": train_fit_source_receipt[
                "receipt_sha256"
            ],
            "g2_power_rows": power_files["rows_file_sha256"],
            "g2_power_reference": power_files["reference_file_sha256"],
            "role_lock": role_lock["lock_sha256"],
            "formal_policy": formal_policy["policy_sha256"],
        },
        safety_counts={
            "authorized_id_mapping_operations": 1,
            "authorized_id_mapping_content_opens": 5,
            "forbidden_protected_content_opens": 0,
            "model_forward_evaluations": 0,
            "real_data_optimizer_updates": 0,
        },
        mutate=mutate,
    )
    return {
        "status": committed["status"],
        "next_action": committed["next_action"],
        "role_lock_sha256": role_lock["lock_sha256"],
        "formal_policy_sha256": formal_policy["policy_sha256"],
        "route_query_count": manifests["train_fit_route_dev"]["query_count"],
        "state_sha256": committed["state_sha256"],
    }


def _run_f0_b_forward(
    bootstrap_lease: BootstrapLease,
    runtime: RuntimeStore,
) -> Mapping[str, Any]:
    from . import a4_forward

    authority = rehydrate_current_authority(runtime, token_id="F0_B")
    control = A4EvalForwardControlAdapter(
        bootstrap_lease,
        runtime,
        token_id="F0_B",
    )
    capability = authority.issue_capability(control)
    journal = a4_forward.ForwardRunJournal(
        authority,
        capability,
        estimated_atomic_output_bytes=ESTIMATED_ATOMIC_OUTPUT_BYTES,
    )
    receipt = journal.start_or_resume()
    if receipt["status"] != "COMPLETED":
        model_handle = a4_forward.load_frozen_model(authority, capability)
        cache = a4_forward.EncodedCorpusCache(
            authority,
            capability,
            chunk_size=CORPUS_CHUNK_SIZE,
        )
        encoded_corpus = cache.materialize_and_load(model_handle)
        engine = a4_forward.FrozenForwardEngine(
            authority,
            capability,
            model_handle,
            encoded_corpus,
        )
        query_rows = [
            {"query_id": query_id, "query_type": query_type}
            for query_id, query_type in zip(
                authority.manifest.query_identity.query_keys,
                authority.manifest.query_identity.query_types,
            )
        ]
        stream = engine.iter_forward(
            query_rows,
            start_index=int(receipt["next_query_index"]),
        )
        try:
            while receipt["status"] != "COMPLETED":
                results = list(itertools.islice(stream, QUERY_CHUNK_SIZE))
                if not results:
                    raise RuntimeControlError(
                        "F0-B forward stream ended before its committed cursor"
                    )
                receipt = journal.commit_query_chunk(receipt, results, engine)
        finally:
            close = getattr(stream, "close", None)
            if callable(close):
                close()
    handle = a4_forward.validate_completed_forward_run_artifacts(
        authority,
        capability,
        run_root=journal.root_identity,
        chunk_root=journal.chunk_root_identity,
    )
    return dict(a4_forward.validate_completed_forward_artifact_handle(handle))


def run_f0_b(
    bootstrap_lease: BootstrapLease,
    store: RuntimeStore | None = None,
) -> dict[str, Any]:
    runtime = store or RuntimeStore()
    state = runtime.read_state()
    if state.get("next_action") != RUN_F0_B_ACTION:
        raise RuntimeControlError("F0-B is not the unique next action")
    g4_receipt = _stage_receipt(runtime, "g4")
    route_path = (
        _stage_root(runtime, "g4")
        / "f1_protocol/role_manifests/train_fit_route_dev.json"
    )
    route_manifest = _read_json(route_path)
    if (
        route_manifest.get("manifest_sha256")
        != semantic_sha256(route_manifest, ("manifest_sha256",))
        or route_manifest.get("role") != "train_fit_route_dev"
        or route_manifest.get("query_count") < 2_048
        or g4_receipt["output_hashes"].get(
            "f1_protocol/role_manifests/train_fit_route_dev.json"
        )
        != bytes_sha256(_canonical_file(route_manifest))
    ):
        raise RuntimeControlError("committed F0-B route manifest drift")
    desc_ids = list(route_manifest["desc_ids"])
    route_desc_ids_sha = bytes_sha256(canonical_json_bytes(desc_ids))
    _f0_a_material_path, f0_a_material = _completed_material(state, "F0_A")
    train = ImportedTrainIndex(runtime)
    begin_f0_b_running(
        route_manifest_sha256=route_manifest["manifest_sha256"],
        route_desc_ids_sha256=route_desc_ids_sha,
        route_query_count=len(desc_ids),
        input_hashes={
            "g4_stage_receipt": g4_receipt["receipt_sha256"],
            "role_lock": _read_json(
                _stage_root(runtime, "g4") / "f1_protocol/role_manifest_lock.json"
            )["lock_sha256"],
            "f0_a_completed_material": f0_a_material[
                "material_binding_sha256"
            ],
            "train_index": train.index["index_binding_sha256"],
        },
        store=runtime,
    )
    state = runtime.read_state()
    current = state["evals"]["F0_B"]["current"]
    if current.get("authority_record_path") is None:
        authority_path = runtime_artifact_path(
            current["eval_id"], "authority.json", root=runtime.root
        )
        query_projection = None
        if not authority_path.exists():
            query_projection = train.project(
                desc_ids,
                purpose="FORWARD_QUERY_IDENTITY",
                projection_scope="F0_B_FORWARD_QUERY_IDENTITY",
            )
        attach_f0_b_authority(query_projection, store=runtime)
    result = _run_f0_b_forward(bootstrap_lease, runtime)
    completed = runtime.read_state()
    return {
        "status": completed["status"],
        "next_action": completed["next_action"],
        "eval_id": completed["evals"]["F0_B"]["current"]["eval_id"],
        "run_id": completed["evals"]["F0_B"]["current"]["run_id"],
        "completed_artifact": result,
        "state_sha256": completed["state_sha256"],
    }


def _recover_f0_b_completed_material(
    bootstrap_lease: BootstrapLease,
    runtime: RuntimeStore,
) -> None:
    from . import a4_forward

    authority = rehydrate_current_authority(runtime, token_id="F0_B")
    control = A4EvalForwardControlAdapter(
        bootstrap_lease,
        runtime,
        token_id="F0_B",
    )
    running_proof = rehydrate_completed_running_proof(
        runtime,
        token_id="F0_B",
    )
    control.bind_completed_historical_running_capability(running_proof)
    handle = a4_forward.recover_completed_forward_run_artifact_handle_from_control(
        authority=authority,
        control=control,
        running_capability=running_proof,
    )
    a4_forward.validate_completed_forward_artifact_handle(handle)
    recovered = runtime.read_state()["evals"]["F0_B"]["current"]
    if not isinstance(recovered, dict) or not isinstance(
        recovered.get("completed_material"), dict
    ):
        raise RuntimeControlError("F0-B completed material recovery did not commit")


def run_f0_b_analyze(
    store: RuntimeStore | None = None,
    *,
    bootstrap_lease: BootstrapLease | None = None,
) -> dict[str, Any]:
    runtime = store or RuntimeStore()
    state = runtime.read_state()
    if (
        state.get("status") != "F0_B_COMPLETED"
        or state.get("next_action") != F0_B_ANALYZE_ACTION
    ):
        raise RuntimeControlError("F0-B analysis is not the unique next action")

    def mutate(next_state: dict[str, Any]) -> None:
        next_state["status"] = "F0_F1_ROUTED"
        next_state["next_action"] = G6_VERIFY_F1_ACTION

    recovered = runtime.recover_prepared_stage(
        stage_id="g5",
        action=F0_B_ANALYZE_ACTION,
        expected_outputs=G5_OUTPUTS,
        mutate=mutate,
    )
    if recovered is not None:
        routing = _read_json(
            _stage_root(runtime, "g5")
            / "f0_forensics/routing_decision.json"
        )
        return {
            "status": recovered["status"],
            "next_action": recovered["next_action"],
            "routing_status": routing["status"],
            "f0_a_earliest_failing_stage": routing[
                "f0_a_earliest_failing_stage"
            ],
            "f0_b_earliest_failing_stage": routing[
                "f0_b_earliest_failing_stage"
            ],
            "state_sha256": recovered["state_sha256"],
            "recovered_prepared_stage": True,
        }
    current = state["evals"]["F0_B"]["current"]
    if not isinstance(current, dict):
        raise RuntimeControlError("F0-B completed state has no current eval")
    if not isinstance(current.get("completed_material"), dict):
        if not isinstance(bootstrap_lease, BootstrapLease):
            raise RuntimeControlError(
                "F0-B completed material recovery requires the active bootstrap lease"
            )
        _recover_f0_b_completed_material(bootstrap_lease, runtime)
        state = runtime.read_state()
    material, validation, identity = _completed_identity(state, "F0_B")
    desc_ids = [int(value) for value in identity["projection"]["query_keys"]]
    train = ImportedTrainIndex(runtime)
    gt_projection = train.project(
        desc_ids,
        purpose="POST_FORWARD_EVAL_STATS",
        projection_scope="F0_B_POST_FORWARD_STATS",
    )
    rows, lineage = derive_completed_forward_rows(
        Path(identity["material_path"]), gt_projection
    )
    built = _analyze_f0_rows(
        runtime=runtime,
        token_id="F0_B",
        query_role="train_fit_route_dev",
        rows=rows,
        lineage=lineage,
        state=state,
        material=material,
        validation=validation,
        identity=identity,
    )
    from . import a4_stages

    g3_receipt = _stage_receipt(runtime, "g3")
    f0_a_report_path = (
        _stage_root(runtime, "g3")
        / "f0_forensics/f0_a_forensics_report.json"
    )
    f0_a = _read_json(f0_a_report_path)
    if (
        f0_a.get("report_sha256") != semantic_sha256(f0_a, ("report_sha256",))
        or g3_receipt["output_hashes"].get(
            "f0_forensics/f0_a_forensics_report.json"
        )
        != bytes_sha256(_canonical_file(f0_a))
        or f0_a["cross_split_execution_invariants"]
        != built["comparison"]["cross_split_execution_invariants"]
    ):
        raise RuntimeControlError("F0-B committed F0-A predecessor drift")
    g3_event = _event_for_stage_receipt(runtime, g3_receipt)
    predecessor = {
        "report_path": "f0_forensics/f0_a_forensics_report.json",
        "report_schema_version": f0_a["schema_version"],
        "report_file_sha256": g3_receipt["output_hashes"][
            "f0_forensics/f0_a_forensics_report.json"
        ],
        "report_sha256": f0_a["report_sha256"],
        "frozen_gate_result_sha256": f0_a["frozen_gate_result_sha256"],
        "earliest_failing_stage": f0_a["earliest_failing_stage"],
        "comparison_summary": f0_a["comparison_summary"],
        "comparison_summary_sha256": f0_a["comparison_summary_sha256"],
        "g3_commit_receipt_sha256": g3_receipt["receipt_sha256"],
        "g3_state_sha256": g3_event["next_state_sha256"],
        "g3_event_sha256": g3_event["event_sha256"],
    }
    a4_stages._exact_mapping(
        predecessor,
        a4_stages.F0_A_COMMITTED_REPORT_BINDING_FIELDS,
        "runtime F0-A predecessor binding field set drift",
    )
    f0_a_summary = f0_a["comparison_summary"]
    f0_b_summary = built["comparison"]
    f0_a_rank = f0_a_summary["rank_transition_summary"]
    f0_b_rank = f0_b_summary["rank_transition_summary"]
    rank_shift = {
        "definition": f0_a_rank["definition"],
        **{
            key: a4_stages._f0_scalar_shift(
                f0_a_rank[key], f0_b_rank[key], f"rank_transition.{key}"
            )
            for key in sorted(set(f0_a_rank) - {"definition"})
        },
    }
    same_earliest = (
        f0_a["earliest_failing_stage"]
        == built["gate"]["earliest_failing_stage"]
    )
    routing_status = "ROUTING_CONSISTENT" if same_earliest else "ROUTING_AMBIGUOUS"
    split_base = {
        "schema_version": "c28f_f0_split_shift_report_v1",
        "status": "VALIDATED_EXPOSURE_AWARE_SPLIT_SHIFT",
        "f0_a_committed_report_binding": predecessor,
        "f0_b_completed_eval_capability_sha256": validation["capability_id"],
        "f0_b_comparison_summary": f0_b_summary,
        "f0_b_comparison_summary_sha256": semantic_sha256(f0_b_summary),
        "query_count_shift": a4_stages._f0_scalar_shift(
            f0_a_summary["query_count"],
            f0_b_summary["query_count"],
            "query_count",
        ),
        "cross_split_execution_invariants": f0_b_summary[
            "cross_split_execution_invariants"
        ],
        "cross_split_execution_invariants_match": True,
        "retrieval_metric_shift": a4_stages._f0_flat_numeric_shift(
            f0_a_summary["retrieval_metrics"],
            f0_b_summary["retrieval_metrics"],
            label="retrieval",
        ),
        "joint_retrieval_metric_shift": a4_stages._f0_flat_numeric_shift(
            f0_a_summary["joint_retrieval_metrics"],
            f0_b_summary["joint_retrieval_metrics"],
            label="joint_retrieval",
        ),
        "vcmr_metric_shift": a4_stages._f0_flat_numeric_shift(
            f0_a_summary["vcmr_metrics"], f0_b_summary["vcmr_metrics"], label="vcmr"
        ),
        "error_metric_shift": a4_stages._f0_flat_numeric_shift(
            f0_a_summary["error_metrics"], f0_b_summary["error_metrics"], label="errors"
        ),
        "gate_metric_shift": a4_stages._f0_flat_numeric_shift(
            f0_a_summary["gate_metrics"], f0_b_summary["gate_metrics"], label="gates"
        ),
        "rank_transition_shift": rank_shift,
        "margin_shift": a4_stages._f0_margin_shift(
            f0_a_summary["margin_diagnostics"],
            f0_b_summary["margin_diagnostics"],
        ),
        "modality_mask_shift": a4_stages._f0_modality_mask_shift(
            f0_a_summary["modality_mask_diagnostics"],
            f0_b_summary["modality_mask_diagnostics"],
        ),
        "video_distribution_shift": a4_stages._f0_video_distribution_shift(
            f0_a_summary["video_distribution_diagnostics"],
            f0_b_summary["video_distribution_diagnostics"],
        ),
        "bucket_shift": a4_stages._f0_bucket_shift(
            f0_a_summary["bucket_distribution_summary"],
            f0_b_summary["bucket_distribution_summary"],
        ),
        "earliest_failing_stage": {
            "f0_a": f0_a["earliest_failing_stage"],
            "f0_b": built["gate"]["earliest_failing_stage"],
            "same": same_earliest,
        },
        "teacher_status": {
            "f0_a": "UNAVAILABLE_ZERO_TEACHER_ACCESS",
            "f0_b": "UNAVAILABLE_ZERO_TEACHER_ACCESS",
        },
        "gt_support_count": {"f0_a": 0, "f0_b": 0},
        "training_exposure_boundary": {
            "f0_a": "C28E_GENERALIZATION_SELECTION_EVIDENCE",
            "f0_b": "C28E_TRAIN_FIT_EXPOSED_MECHANISM_PROBE",
            "f0_b_cannot_override_f0_a": True,
        },
        "threshold_changed_after_observation": False,
        "routing_status": routing_status,
    }
    split_report = {
        **split_base,
        "split_shift_report_sha256": semantic_sha256(split_base),
    }
    a4_stages._exact_mapping(
        split_report,
        a4_stages.F0_SPLIT_SHIFT_REPORT_FIELDS,
        "runtime F0 split-shift report field set drift",
    )
    routing_base = {
        "schema_version": "c28f_f0_routing_decision_v1",
        "status": routing_status,
        "f0_a_frozen_gate_result_sha256": f0_a[
            "frozen_gate_result_sha256"
        ],
        "f0_b_frozen_gate_result_sha256": built["gate"][
            "gate_result_sha256"
        ],
        "f0_a_earliest_failing_stage": f0_a["earliest_failing_stage"],
        "f0_b_earliest_failing_stage": built["gate"][
            "earliest_failing_stage"
        ],
        "selected_earliest_failing_stage": (
            built["gate"]["earliest_failing_stage"] if same_earliest else None
        ),
        "next_stage_recommendation_only": (
            built["gate"]["next_stage_recommendation_only"]
            if same_earliest
            else "F2_R0_REQUIRES_FUTURE_AUTH_F2_F5_PROTOTYPE"
        ),
        "split_shift_report_sha256": split_report[
            "split_shift_report_sha256"
        ],
        "threshold_changed_after_observation": False,
        "f2_or_higher_executed": False,
        "ambiguous_is_valid_scientific_result": not same_earliest,
    }
    routing = {
        **routing_base,
        "routing_decision_sha256": semantic_sha256(routing_base),
    }
    a4_stages._exact_mapping(
        routing,
        a4_stages.F0_ROUTING_DECISION_FIELDS,
        "runtime F0 routing decision field set drift",
    )
    split_bytes = _canonical_file(split_report)
    routing_bytes = _canonical_file(routing)
    split_markdown = a4_stages._render_summary_markdown(
        title="C28F F0-A/F0-B Split Shift",
        authoritative_content_sha256=bytes_sha256(split_bytes),
        rows=(
            ("status", split_report["status"]),
            ("routing_status", routing_status),
            ("f0_a_earliest", f0_a["earliest_failing_stage"]),
            ("f0_b_earliest", built["gate"]["earliest_failing_stage"]),
            ("threshold_changed_after_observation", False),
        ),
    )
    routing_markdown = a4_stages._render_summary_markdown(
        title="C28F F0 Routing Decision",
        authoritative_content_sha256=bytes_sha256(routing_bytes),
        rows=(
            ("status", routing_status),
            (
                "selected_earliest_failing_stage",
                routing["selected_earliest_failing_stage"],
            ),
            (
                "next_stage_recommendation_only",
                routing["next_stage_recommendation_only"],
            ),
            ("f2_or_higher_executed", False),
        ),
    )
    query_metrics_bytes = built["query_metrics_bytes"]
    artifact_commitments = {
        "query_metrics_rows_sha256": bytes_sha256(
            canonical_json_bytes(list(built["query_metric_rows"]))
        ),
        "query_metrics_file_sha256": bytes_sha256(query_metrics_bytes),
        "split_shift_report_sha256": split_report[
            "split_shift_report_sha256"
        ],
        "split_shift_report_file_sha256": bytes_sha256(split_bytes),
        "split_shift_markdown_file_sha256": bytes_sha256(split_markdown),
        "routing_decision_sha256": routing["routing_decision_sha256"],
        "routing_decision_file_sha256": bytes_sha256(routing_bytes),
        "routing_decision_markdown_file_sha256": bytes_sha256(
            routing_markdown
        ),
        "metric_sufficient_stat_contract_sha256": built[
            "metric_sufficient_stat_contract_sha256"
        ],
        "committed_query_role_binding_sha256": built["role_binding"][
            "binding_sha256"
        ],
        "completed_eval_source_artifact_lineage_sha256": lineage[
            "lineage_sha256"
        ],
    }
    report = _build_f0_report(
        token_id="F0_B",
        query_role="train_fit_route_dev",
        rows=rows,
        built=built,
        artifact_commitments=artifact_commitments,
        predecessor_f0_a_binding=predecessor,
        split_shift_report_sha256=split_report["split_shift_report_sha256"],
        routing_status=routing_status,
    )
    report_bytes = _canonical_file(report)
    ledger = _build_f0_ledger(
        token_id="F0_B",
        report=report,
        report_bytes=report_bytes,
        built=built,
        committed_f0_a_report_sha256=f0_a["report_sha256"],
        routing_status=routing_status,
    )
    outputs = {
        "f0_forensics/f0_b_forensics_report.json": report_bytes,
        "f0_forensics/f0_b_query_metrics.jsonl": query_metrics_bytes,
        "f0_forensics/split_shift_report.json": split_bytes,
        "f0_forensics/split_shift_report.md": split_markdown,
        "f0_forensics/routing_decision.json": routing_bytes,
        "f0_forensics/routing_decision.md": routing_markdown,
        "f0_forensics/f0_b_eval_ledger_entry.json": _canonical_file(ledger),
    }

    committed = runtime.commit_stage(
        stage_id="g5",
        action=F0_B_ANALYZE_ACTION,
        outputs=outputs,
        input_hashes={
            "completed_material": material["material_binding_sha256"],
            "validation_receipt": validation["validation_receipt_sha256"],
            "train_projection": gt_projection["record_sha256"],
            "retrieval_rows": lineage["retrieval_rows_sha256"],
            "committed_f0_a_report": f0_a["report_sha256"],
            "split_shift": split_report["split_shift_report_sha256"],
            "routing": routing["routing_decision_sha256"],
        },
        safety_counts={
            "model_forward_evaluations": 1,
            "protected_content_opens": 0,
            "real_data_optimizer_updates": 0,
            "teacher_forward_count": 0,
            "gt_support_count": 0,
        },
        mutate=mutate,
    )
    return {
        "status": committed["status"],
        "next_action": committed["next_action"],
        "routing_status": routing_status,
        "f0_a_earliest_failing_stage": f0_a["earliest_failing_stage"],
        "f0_b_earliest_failing_stage": built["gate"]["earliest_failing_stage"],
        "state_sha256": committed["state_sha256"],
    }


def run_g6_verify_f1(store: RuntimeStore | None = None) -> dict[str, Any]:
    runtime = store or RuntimeStore()
    state = runtime.read_state()
    if (
        state.get("status") != "F0_F1_ROUTED"
        or state.get("next_action") != G6_VERIFY_F1_ACTION
    ):
        raise RuntimeControlError("G6 F1 verification is not the unique next action")

    def mutate(next_state: dict[str, Any]) -> None:
        next_state["status"] = "F1_COMPLETE"
        next_state["next_action"] = G7_FINALIZE_ACTION

    recovered = runtime.recover_prepared_stage(
        stage_id="g6",
        action=G6_VERIFY_F1_ACTION,
        expected_outputs=G6_OUTPUTS,
        mutate=mutate,
    )
    if recovered is not None:
        protocol = _read_json(
            _stage_root(runtime, "g6")
            / "f1_protocol/f1_protocol_report.final.json"
        )
        return {
            "status": recovered["status"],
            "next_action": recovered["next_action"],
            "f1_protocol_report_sha256": protocol["report_sha256"],
            "synthetic_optimizer_updates": protocol[
                "synthetic_optimizer_updates"
            ],
            "state_sha256": recovered["state_sha256"],
            "recovered_prepared_stage": True,
        }
    test_report_path = runtime.root / "artifacts" / "review" / "TEST_EXECUTION_REPORT.json"
    test_report = _read_json(test_report_path)
    interruption_evidence = test_report.get("interruption_evidence_sha256s")
    required_interruptions = {
        "BEFORE_CHECKPOINT_COMMIT",
        "AFTER_CHECKPOINT_BEFORE_MANIFEST",
        "DURING_EVAL",
        "BEFORE_STATUS_RENAME",
    }
    if (
        test_report.get("status") != "PASS"
        or test_report.get("report_sha256")
        != semantic_sha256(test_report, ("report_sha256",))
        or test_report.get("full_c28f_suite_passed") is not True
        or test_report.get("temporary_runtime_dry_run_passed") is not True
        or not isinstance(interruption_evidence, dict)
        or set(interruption_evidence) != required_interruptions
        or any(
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
            for digest in interruption_evidence.values()
        )
    ):
        raise RuntimeControlError("G6 requires the committed PASS test report")
    from . import a4_f1

    temporal = a4_f1.temporal_oracles()
    recovery = a4_f1.recovery_matrix()
    checkpoint = a4_f1.checkpoint_reference_contract()
    cost = a4_f1.cost_measurement_protocol()
    superset = a4_f1.superset_scaffold_contract()
    shape = a4_f1.shape_capacity_contract()
    protocol = a4_f1.build_f1_protocol_report()
    protocol.pop("report_sha256", None)
    higher_stage_applicability = {
        stage: "NOT_APPLICABLE_SCHEMA_ONLY" for stage in ("F2", "F3", "F4", "F5")
    }
    protocol.update(
        {
            "status": "VERIFIED_SYNTHETIC_PROTOCOL_COMPLETE",
            "g6_commit_eligible": True,
            "ledger_derived_counts": True,
            "atomic_crash_evidence": {
                "status": "PASS_FOUR_INTERRUPTION_CLASSES",
                "required_crash_points": [
                    "BEFORE_CHECKPOINT_COMMIT",
                    "AFTER_CHECKPOINT_BEFORE_MANIFEST",
                    "DURING_EVAL",
                    "BEFORE_STATUS_RENAME",
                ],
                "test_execution_report_sha256": test_report[
                    "report_sha256"
                ],
                "committed_evidence_sha256s": interruption_evidence,
            },
            "higher_stage_applicability": higher_stage_applicability,
            "f2_f5_scientific_mechanisms_executed": False,
        }
    )
    protocol["report_sha256"] = semantic_sha256(protocol)
    schedule_base = {
        "schema_version": "c28f_v5_schedule_checkpoint_state_test_report_v1",
        "status": "PASS",
        "optimizer_progress_schedule": protocol["schedule_boundaries"],
        "formal_17360_schedule": protocol["formal_17360_schedule_boundaries"],
        "checkpoint_contract": checkpoint,
        "shape_capacity_contract": shape,
        "synthetic_resume_traces": protocol["synthetic_resume_traces"],
        "atomic_crash_evidence": protocol["atomic_crash_evidence"],
        "single_writer_required": True,
        "fsync_before_visibility": True,
        "real_data_optimizer_updates": 0,
    }
    schedule_report = {
        **schedule_base,
        "report_sha256": semantic_sha256(schedule_base),
    }
    superset_audit_base = {
        "schema_version": "c28f_v5_superset_scaffold_audit_v1",
        "status": "PASS_SCHEMA_ONLY",
        "superset_scaffold": superset,
        "shape_capacity": shape,
        "F0_F1_applicable": True,
        "F2_F5_not_executed": True,
        "higher_stage_applicability": higher_stage_applicability,
        "real_data_training_not_authorized": True,
    }
    superset_audit = {
        **superset_audit_base,
        "audit_sha256": semantic_sha256(superset_audit_base),
    }
    regression_base = {
        "schema_version": "c28f_v5_f0_f1_applicable_regression_suite_v1",
        "status": "PASS",
        "test_execution_report_sha256": test_report["report_sha256"],
        "full_c28f_suite_passed": True,
        "temporary_runtime_dry_run_passed": True,
        "synthetic_8q_status": protocol["synthetic_8q_status"],
        "synthetic_128q_status": protocol["synthetic_128q_status"],
        "synthetic_optimizer_updates": protocol[
            "synthetic_optimizer_updates"
        ],
        "synthetic_optimizer_update_cap": 64,
        "real_data_optimizer_updates": 0,
        "extra_model_forward_splits": 0,
        "higher_stage_applicability": higher_stage_applicability,
    }
    if regression_base["synthetic_optimizer_updates"] > 64:
        raise RuntimeControlError("G6 synthetic optimizer update cap exceeded")
    regression = {
        **regression_base,
        "suite_sha256": semantic_sha256(regression_base),
    }
    g6_binding_base = {
        "schema_version": "c28f_v5_runtime_g6_control_binding_v1",
        "status": "READY_FOR_SINGLE_STAGE_CAS",
        "pre_state_sha256": state["state_sha256"],
        "g5_stage_receipt_sha256": _stage_receipt(runtime, "g5")[
            "receipt_sha256"
        ],
        "test_execution_report_sha256": test_report["report_sha256"],
        "temporal_oracles_sha256": temporal["oracles_sha256"],
        "recovery_matrix_sha256": recovery["matrix_sha256"],
        "checkpoint_contract_sha256": checkpoint["contract_sha256"],
        "cost_protocol_sha256": cost["protocol_sha256"],
        "superset_audit_sha256": superset_audit["audit_sha256"],
        "regression_suite_sha256": regression["suite_sha256"],
        "f1_protocol_report_sha256": protocol["report_sha256"],
        "model_forward_evaluations_added": 0,
        "real_data_optimizer_updates": 0,
    }
    g6_binding = {
        **g6_binding_base,
        "binding_sha256": semantic_sha256(g6_binding_base),
    }
    outputs = {
        "f1_protocol/temporal_oracles.json": _canonical_file(temporal),
        "f1_protocol/recovery_matrix.json": _canonical_file(recovery),
        "f1_protocol/schedule_checkpoint_state_test_report.json": _canonical_file(
            schedule_report
        ),
        "f1_protocol/cost_measurement_protocol.json": _canonical_file(cost),
        "f1_protocol/superset_scaffold_audit.json": _canonical_file(
            superset_audit
        ),
        "f1_protocol/f0_f1_applicable_regression_suite.json": _canonical_file(
            regression
        ),
        "f1_protocol/f1_protocol_report.final.json": _canonical_file(protocol),
        "f1_protocol/g6_control_binding.json": _canonical_file(g6_binding),
    }

    committed = runtime.commit_stage(
        stage_id="g6",
        action=G6_VERIFY_F1_ACTION,
        outputs=outputs,
        input_hashes={
            "g5_stage_receipt": _stage_receipt(runtime, "g5")[
                "receipt_sha256"
            ],
            "test_execution_report": test_report["report_sha256"],
            "f1_protocol_report": protocol["report_sha256"],
            "g6_control_binding": g6_binding["binding_sha256"],
        },
        safety_counts={
            "model_forward_evaluations": 0,
            "real_data_optimizer_updates": 0,
            "synthetic_optimizer_updates": protocol[
                "synthetic_optimizer_updates"
            ],
            "protected_content_opens": 0,
        },
        mutate=mutate,
    )
    return {
        "status": committed["status"],
        "next_action": committed["next_action"],
        "f1_protocol_report_sha256": protocol["report_sha256"],
        "synthetic_optimizer_updates": protocol["synthetic_optimizer_updates"],
        "state_sha256": committed["state_sha256"],
    }


def _artifact_inventory(
    runtime: RuntimeStore,
    *,
    stage_ids: Sequence[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for stage_id in stage_ids:
        root = _stage_root(runtime, stage_id)
        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            data, _record = read_regular_bytes(
                path,
                max_bytes=1024 * 1024 * 1024,
                require_nlink_one=True,
            )
            rows.append(
                {
                    "path": str(path.relative_to(runtime.root)),
                    "size_bytes": len(data),
                    "file_sha256": bytes_sha256(data),
                }
            )
    return rows


def _gate_records(runtime: RuntimeStore) -> dict[str, dict[str, str]]:
    bridge = _read_json(
        runtime.root / "control_bridge" / "supersession_bridge.json"
    )
    if bridge.get("bridge_sha256") != semantic_sha256(
        bridge, ("bridge_sha256",)
    ) or bridge.get("bridge_sha256") != FROZEN_SUPERSESSION_BRIDGE_SHA256:
        raise RuntimeControlError("finalization supersession bridge drift")

    legacy_events = _read_jsonl(CONTROL_ROOT / "GOAL_EVENTS.jsonl")
    legacy_event_bytes = canonical_jsonl(legacy_events)
    legacy_snapshot = bridge["legacy_snapshot"]
    if (
        len(legacy_events) != legacy_snapshot["events_count"]
        or bytes_sha256(legacy_event_bytes)
        != legacy_snapshot["events_file_sha256"]
        or legacy_events[-1].get("event_sha256")
        != legacy_snapshot["events_tail_sha256"]
    ):
        raise RuntimeControlError("finalization legacy event snapshot drift")
    genesis_event = legacy_events[0]
    if (
        genesis_event.get("action")
        != "G0_A4_GENESIS_AFTER_VALID_A3_SUPERSESSION"
        or genesis_event.get("event_sha256")
        != semantic_sha256(genesis_event, ("event_sha256",))
    ):
        raise RuntimeControlError("finalization G0 genesis event drift")
    genesis_state = _read_json(
        CONTROL_ROOT
        / "state_records"
        / (
            "000000_"
            + str(genesis_event["expected_next_state_sha256"])
            + ".json"
        )
    )
    if (
        genesis_state.get("state_sha256")
        != semantic_sha256(genesis_state, ("state_sha256",))
        or genesis_state.get("state_sha256")
        != genesis_event["expected_next_state_sha256"]
    ):
        raise RuntimeControlError("finalization G0 genesis state drift")
    bootstrap_intent = _read_json(CONTROL_ROOT / "a4_bootstrap_intent.json")
    bootstrap_bytes = _canonical_file(bootstrap_intent)
    bootstrap_root_bytes, _root_record = read_regular_bytes(
        CONTROL_ROOT / "a4_bootstrap_intent.root.sha256",
        max_bytes=128,
        require_nlink_one=True,
    )
    if (
        bootstrap_intent.get("intent_sha256")
        != semantic_sha256(bootstrap_intent, ("intent_sha256",))
        or bootstrap_root_bytes
        != (bytes_sha256(bootstrap_bytes) + "\n").encode("ascii")
        or genesis_event.get("input_hashes", {}).get("intent_file")
        != bytes_sha256(bootstrap_bytes)
        or genesis_event.get("input_hashes", {}).get("intent_semantic")
        != bootstrap_intent["intent_sha256"]
    ):
        raise RuntimeControlError("finalization G0 bootstrap binding drift")

    imported_receipts: dict[str, dict[str, Any]] = {}
    for gate_id in ("G1", "G2"):
        projection = bridge["gate_projection"][gate_id]
        receipt = _read_json(
            CONTROL_ROOT
            / "transactions"
            / str(projection["transaction_id"])
            / "commit_receipt.json"
        )
        if (
            bytes_sha256(_canonical_file(receipt))
            != projection["receipt_file_sha256"]
            or receipt.get("receipt_sha256")
            != semantic_sha256(receipt, ("receipt_sha256",))
            or receipt.get("receipt_sha256") != projection["receipt_sha256"]
            or receipt.get("receipt_sha256")
            != {
                "G1": FROZEN_G1_RECEIPT_SHA256,
                "G2": FROZEN_G2_RECEIPT_SHA256,
            }[gate_id]
            or receipt.get("event_sha256") != projection["event_sha256"]
            or receipt.get("output_hashes") != projection["output_hashes"]
        ):
            raise RuntimeControlError(
                f"finalization imported {gate_id} receipt drift"
            )
        imported_receipts[gate_id] = receipt

    train_projection = bridge["gate_projection"]["TRAIN_JSONL_INDEX"]
    train_receipt = _read_json(
        CONTROL_ROOT
        / "transactions"
        / str(train_projection["transaction_id"])
        / "bootstrap_receipt.json"
    )
    if (
        bytes_sha256(_canonical_file(train_receipt))
        != train_projection["receipt_file_sha256"]
        or train_receipt.get("receipt_sha256")
        != semantic_sha256(train_receipt, ("receipt_sha256",))
        or train_receipt.get("receipt_sha256")
        != train_projection["receipt_sha256"]
        or train_receipt.get("receipt_sha256")
        != FROZEN_TRAIN_INDEX_RECEIPT_SHA256
    ):
        raise RuntimeControlError("finalization imported TRAIN receipt drift")

    current_state = runtime.read_state()
    f0_a = current_state["evals"]["F0_A"]["current"]
    material = _read_json(Path(f0_a["completed_material"]["path"]))
    completion = _read_json(
        Path(f0_a["completed_material"]["path"]).parent / "completion.json"
    )
    anchor = bridge["runtime_anchor"]
    if (
        material.get("material_binding_sha256")
        != semantic_sha256(material, ("material_binding_sha256",))
        or material.get("material_binding_sha256")
        != anchor["completed_material_binding_sha256"]
        or material.get("material_binding_sha256")
        != FROZEN_F0_A_MATERIAL_SHA256
        or f0_a.get("eval_id") != anchor["f0_a_eval_id"]
        or f0_a.get("run_id") != anchor["f0_a_run_id"]
        or completion.get("receipt_sha256")
        != semantic_sha256(completion, ("receipt_sha256",))
        or completion.get("receipt_sha256")
        != FROZEN_F0_A_COMPLETION_RECEIPT_SHA256
        or completion.get("eval_id") != f0_a["eval_id"]
        or completion.get("run_id") != f0_a["run_id"]
    ):
        raise RuntimeControlError("finalization imported F0-A receipt drift")

    g3 = _stage_receipt(runtime, "g3")
    g4 = _stage_receipt(runtime, "g4")
    g5 = _stage_receipt(runtime, "g5")
    g6 = _stage_receipt(runtime, "g6")
    runtime_events = _read_jsonl(runtime.events_path)

    def committed_gate(number: int, receipt: Mapping[str, Any]) -> dict[str, str]:
        matches = [
            event
            for event in runtime_events
            if event.get("artifact_sha256", {}).get("STAGE_RECEIPT.json")
            == receipt["receipt_sha256"]
        ]
        if len(matches) != 1:
            raise RuntimeControlError(f"runtime G{number} stage event is not unique")
        event = matches[0]
        return {
            "gate_id": f"G{number}",
            "status": "PASS",
            "state_sha256": event["next_state_sha256"],
            "event_sha256": event["event_sha256"],
            "control_receipt_sha256": receipt["receipt_sha256"],
        }

    return {
        "G0": {
            "gate_id": "G0",
            "status": "PASS",
            "state_sha256": genesis_state["state_sha256"],
            "event_sha256": genesis_event["event_sha256"],
            "control_receipt_sha256": bootstrap_intent["intent_sha256"],
        },
        "G1": {
            "gate_id": "G1",
            "status": "PASS",
            "state_sha256": imported_receipts["G1"]["state_sha256"],
            "event_sha256": imported_receipts["G1"]["event_sha256"],
            "control_receipt_sha256": imported_receipts["G1"]["receipt_sha256"],
        },
        "G2": {
            "gate_id": "G2",
            "status": "PASS",
            "state_sha256": imported_receipts["G2"]["state_sha256"],
            "event_sha256": imported_receipts["G2"]["event_sha256"],
            "control_receipt_sha256": imported_receipts["G2"]["receipt_sha256"],
        },
        "G3": committed_gate(3, g3),
        "G4": committed_gate(4, g4),
        "G5": committed_gate(5, g5),
        "G6": committed_gate(6, g6),
    }


def _g7_evidence(
    runtime: RuntimeStore, state: Mapping[str, Any]
) -> dict[str, Any]:
    finalization_pre_state_sha256 = (
        state["previous_state_sha256"]
        if state.get("status") == "PENDING_FINALIZATION"
        else state["state_sha256"]
    )
    receipts = {
        stage_id: _stage_receipt(runtime, stage_id)
        for stage_id in ("g3", "g4", "g5", "g6")
    }

    def require_counts(stage_id: str, expected: Mapping[str, int]) -> None:
        observed = receipts[stage_id].get("safety_counts")
        if not isinstance(observed, dict) or any(
            observed.get(name) != count for name, count in expected.items()
        ):
            raise RuntimeControlError(f"G7 {stage_id} safety receipt drift")

    require_counts(
        "g3",
        {
            "model_forward_evaluations": 1,
            "protected_content_opens": 0,
            "real_data_optimizer_updates": 0,
            "teacher_forward_count": 0,
            "gt_support_count": 0,
        },
    )
    require_counts(
        "g4",
        {
            "authorized_id_mapping_operations": 1,
            "authorized_id_mapping_content_opens": 5,
            "forbidden_protected_content_opens": 0,
            "model_forward_evaluations": 0,
            "real_data_optimizer_updates": 0,
        },
    )
    require_counts(
        "g5",
        {
            "model_forward_evaluations": 1,
            "protected_content_opens": 0,
            "real_data_optimizer_updates": 0,
            "teacher_forward_count": 0,
            "gt_support_count": 0,
        },
    )
    require_counts(
        "g6",
        {
            "model_forward_evaluations": 0,
            "protected_content_opens": 0,
            "real_data_optimizer_updates": 0,
        },
    )
    g6_synthetic_updates = receipts["g6"]["safety_counts"].get(
        "synthetic_optimizer_updates"
    )
    if type(g6_synthetic_updates) is not int or not 0 <= g6_synthetic_updates <= 64:
        raise RuntimeControlError("G7 G6 synthetic optimizer budget drift")

    evals = state.get("evals")
    safety = state.get("safety")
    if (
        not isinstance(evals, dict)
        or not isinstance(safety, dict)
        or any(
            evals.get(token_id, {}).get("successful_evaluations") != 1
            or evals.get(token_id, {}).get("current", {}).get("status")
            != "COMPLETED"
            for token_id in ("F0_A", "F0_B")
        )
        or safety.get("model_forward_evaluations") != 2
        or any(
            safety.get(name) != 0
            for name in (
                "protected_content_opens",
                "optimizer_updates",
                "teacher_forward_count",
                "gt_support_count",
            )
        )
    ):
        raise RuntimeControlError("G7 runtime safety/eval closure drift")
    model_forward_counts = {
        "F0_A": evals["F0_A"]["successful_evaluations"],
        "F0_B": evals["F0_B"]["successful_evaluations"],
        "protected": safety["protected_content_opens"],
        "other": safety["model_forward_evaluations"]
        - evals["F0_A"]["successful_evaluations"]
        - evals["F0_B"]["successful_evaluations"],
    }

    projection_path = (
        _stage_root(runtime, "g4")
        / "f1_protocol/protected_id_projection_manifest.json"
    )
    projection = _read_json(projection_path)
    projection_relative = "f1_protocol/protected_id_projection_manifest.json"
    classification = projection.get("content_open_classification")
    if (
        projection.get("manifest_sha256")
        != semantic_sha256(projection, ("manifest_sha256",))
        or receipts["g4"]["output_hashes"].get(projection_relative)
        != bytes_sha256(_canonical_file(projection))
        or not isinstance(classification, dict)
        or classification.get("categories_mutually_exclusive") is not True
        or classification.get(
            "unauthorized_or_non_id_only_protected_content_open_count"
        )
        != 0
        or classification.get("authorized_protected_id_mapping_content_open_count")
        != 5
        or classification.get("total_protected_content_open_count") != 5
        or projection.get("model_forward_count") != 0
    ):
        raise RuntimeControlError("G7 protected ID-only projection evidence drift")

    ledger_specs = (
        ("g3", "F0_A", "f0_forensics/f0_a_eval_ledger_entry.json"),
        ("g5", "F0_B", "f0_forensics/f0_b_eval_ledger_entry.json"),
    )
    eval_rows: list[dict[str, Any]] = []
    for stage_id, token_id, relative in ledger_specs:
        source = _read_json(_stage_root(runtime, stage_id) / relative)
        source_bytes = _canonical_file(source)
        if (
            source.get("ledger_entry_sha256")
            != semantic_sha256(source, ("ledger_entry_sha256",))
            or source.get("status")
            != "ONE_LOGICAL_EVAL_COMPLETED_NONREPLAYABLE"
            or source.get("token_id") != token_id
            or source.get("logical_eval_count") != 1
            or receipts[stage_id]["output_hashes"].get(relative)
            != bytes_sha256(source_bytes)
        ):
            raise RuntimeControlError(f"G7 committed {token_id} ledger drift")
        row_base = {
            "schema_version": "c28f_a4_final_eval_ledger_row_v1",
            "status": "COMMITTED_NONREPLAYABLE_LOGICAL_EVAL",
            "goal_id": GOAL_ID,
            "attempt_id": state["attempt_id"],
            "token_id": token_id,
            "eval_id": source["eval_id"],
            "logical_eval_count": source["logical_eval_count"],
            "source_file_sha256": bytes_sha256(source_bytes),
            "source_ledger_entry": source,
        }
        eval_rows.append({**row_base, "row_sha256": semantic_sha256(row_base)})
    if (
        [row["token_id"] for row in eval_rows] != ["F0_A", "F0_B"]
        or eval_rows[0]["eval_id"] == eval_rows[1]["eval_id"]
    ):
        raise RuntimeControlError("G7 final eval ledger cardinality drift")
    eval_ledger_bytes = canonical_jsonl(eval_rows)

    audit_path = TEST_ROOT / "A4_PREBOOTSTRAP_AUDIT.json"
    audit_data, audit_record = read_regular_bytes(
        audit_path,
        max_bytes=8 * 1024 * 1024,
        require_nlink_one=True,
    )
    try:
        audit = json.loads(audit_data.decode("utf-8", "strict"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeControlError("pre-A4 exception audit JSON drift") from exc
    exception_rows = (
        [
            row
            for row in audit.get("events", [])
            if isinstance(row, dict)
            and row.get("event_id")
            == "PRE_A4_STATIC_SEARCH_SCOPE_EXCEPTION_002"
        ]
        if isinstance(audit, dict)
        else []
    )
    if (
        audit_record["sha256"]
        != "5e1ca5dc2204ae2e6690c7fa96f97774a90b496f93520a0c07e21fc80eff10f4"
        or len(exception_rows) != 1
        or exception_rows[0].get("protected_content_stream_open_count")
        != "NONZERO_UNQUANTIFIED"
        or exception_rows[0].get("scientific_decision_impact") != "NONE"
        or "EXCLUDED_FROM_ALL_LINEAGE"
        not in str(exception_rows[0].get("current_disposition"))
    ):
        raise RuntimeControlError("pre-A4 recorded access exception drift")
    pre_a4_exception = {
        "included_in_recovery_lineage": False,
        "status": "NONZERO_UNQUANTIFIED_RECORDED_EXCEPTION",
        "audit_path": str(audit_path),
        "audit_file_sha256": audit_record["sha256"],
        "event_id": exception_rows[0]["event_id"],
        "event_sha256": semantic_sha256(exception_rows[0]),
        "scientific_decision_impact": "NONE",
        "protocol_audit_impact": "MATERIAL_RECORDED_EXCEPTION",
        "lineage_disposition": "EXCLUDED_FROM_RECOVERY_SCIENTIFIC_LINEAGE",
    }

    access_counts = {
        "protected_model_eval": 0,
        "protected_feature_get": 0,
        "protected_prediction_read": 0,
        "protected_metric_read": 0,
        "protected_query_text_read": 0,
        "protected_timestamp_read": 0,
        "protected_label_read": 0,
        "official_workflow": 0,
        "forbidden_value_materialization": 0,
        "unresolved_open_intent": 0,
        "c28e_historical_write_bytes": 0,
        "teacher_model_forward": safety["teacher_forward_count"],
        "gt_support_injection": safety["gt_support_count"],
        "gt_append": 0,
        "posthoc_temperature_refit_during_eval": 0,
        "protected_id_mapping_operations": receipts["g4"]["safety_counts"][
            "authorized_id_mapping_operations"
        ],
    }
    access_base = {
        "schema_version": "c28f_a4_protected_data_access_summary_v1",
        "status": "ZERO_FORBIDDEN_ACCESS_WITH_AUTHORIZED_ID_ONLY_MAPPING",
        "goal_id": GOAL_ID,
        "attempt_id": state["attempt_id"],
        "access_counts": access_counts,
        "recovery_lineage_counts_derived_from_committed_receipts": True,
        "authorized_id_mapping": {
            "operation_count": access_counts["protected_id_mapping_operations"],
            "physical_content_open_count": classification[
                "total_protected_content_open_count"
            ],
            "manifest_content_open_count": 3,
            "train_jsonl_projection_content_open_count": 2,
            "protected_desc_manifest_content_open_count": 2,
            "train_fit_manifest_content_open_count": 1,
            "protected_desc_train_jsonl_indexed_pread_content_open_count": 1,
            "train_fit_train_jsonl_indexed_pread_content_open_count": 1,
            "full_train_file_sequential_rescan_count": 0,
            "decoded_fields": ["desc_id", "vid_name"],
            "model_forward_count": projection["model_forward_count"],
            "official_eval_count": 0,
            "protected_eval_count": 0,
            "projection_manifest_sha256": projection["manifest_sha256"],
        },
        "proof_bindings": {
            "g3_stage_receipt_sha256": receipts["g3"]["receipt_sha256"],
            "g4_stage_receipt_sha256": receipts["g4"]["receipt_sha256"],
            "g5_stage_receipt_sha256": receipts["g5"]["receipt_sha256"],
            "g6_stage_receipt_sha256": receipts["g6"]["receipt_sha256"],
            "protected_projection_manifest_sha256": projection[
                "manifest_sha256"
            ],
            "runtime_pre_finalization_state_sha256": finalization_pre_state_sha256,
        },
        "pre_a4_recorded_exception": pre_a4_exception,
        "protected_model_evaluation_authorized": False,
        "official_validation_run": False,
    }
    access_summary = {
        **access_base,
        "summary_sha256": semantic_sha256(access_base),
    }
    return {
        "receipts": receipts,
        "eval_ledger_bytes": eval_ledger_bytes,
        "access_counts": access_counts,
        "access_summary": access_summary,
        "model_forward_counts": model_forward_counts,
        "real_data_optimizer_updates": safety["optimizer_updates"],
    }


def run_g7_finalize(store: RuntimeStore | None = None) -> dict[str, Any]:
    runtime = store or RuntimeStore()
    state = runtime.read_state()
    if state.get("next_action") != G7_FINALIZE_ACTION or state.get("status") not in {
        "F1_COMPLETE",
        "PENDING_FINALIZATION",
    }:
        raise RuntimeControlError("G7 finalization is not the unique next action")

    def terminal(next_state: dict[str, Any]) -> None:
        next_state["status"] = "F0_F1_WINDOW_COMPLETE_STOPPED"
        next_state["next_action"] = None
        workflow = next_state.get("workflow_control")
        if isinstance(workflow, dict):
            workflow["f0_b_or_f2_entry_allowed"] = False

    prepared_receipt = _stage_root(runtime, "g7") / "STAGE_RECEIPT.json"
    pending_relative = "f1_protocol/finalization/PENDING_FINALIZATION.json"
    if not prepared_receipt.exists():
        g7_root = _stage_root(runtime, "g7")
        observed_unsealed = (
            {
                str(path.relative_to(g7_root))
                for path in g7_root.rglob("*")
                if path.is_file()
            }
            if g7_root.exists()
            else set()
        )
        unexpected_unsealed = observed_unsealed - {pending_relative}
        if unexpected_unsealed:
            raise RuntimeControlError(
                "unsealed G7 candidate outputs exist without the sole stage receipt"
            )
    if state["status"] == "PENDING_FINALIZATION" and prepared_receipt.exists():
        recovered = runtime.recover_prepared_stage(
            stage_id="g7",
            action=G7_FINALIZE_ACTION,
            expected_outputs=G7_OUTPUTS,
            mutate=terminal,
        )
        if recovered is not None:
            decision = _read_json(
                _stage_root(runtime, "g7")
                / "f1_protocol/finalization/F0_F1_WINDOW_DECISION.json"
            )
            root_bytes, _root_record = read_regular_bytes(
                _stage_root(runtime, "g7")
                / "f1_protocol/finalization/artifact_manifest.root.sha256",
                max_bytes=128,
                require_nlink_one=True,
            )
            pending_record = _read_json(
                _stage_root(runtime, "g7")
                / "f1_protocol/finalization/PENDING_FINALIZATION.json"
            )
            if (
                decision.get("decision_artifact_sha256")
                != semantic_sha256(decision, ("decision_artifact_sha256",))
                or pending_record.get("status") != "COMMITTED"
            ):
                raise RuntimeControlError("recovered G7 artifact binding drift")
            return {
                "status": recovered["status"],
                "next_action": recovered["next_action"],
                "decision_sha256": decision["decision_artifact_sha256"],
                "artifact_manifest_root_sha256": root_bytes.decode(
                    "ascii", "strict"
                ).strip(),
                "closed_authority_sha256": pending_record[
                    "closed_authority_sha256"
                ],
                "state_sha256": recovered["state_sha256"],
                "recovered_prepared_stage": True,
            }
    from . import a4_finalize

    evidence = _g7_evidence(runtime, state)
    g6_receipt = evidence["receipts"]["g6"]
    authority = _read_json(CONTROL_ROOT / "authority_ledger.json")
    closed_authority = a4_finalize.close_authority_ledger(
        authority,
        mapping_token_state="COMPLETED_NONREPLAYABLE",
    )
    if closed_authority.get("id_mapping_authority_was_active") is not True:
        raise RuntimeControlError("G7 expected active ID-only authority history")
    eval_ledger_bytes = evidence["eval_ledger_bytes"]
    access_counts = evidence["access_counts"]
    access_summary = evidence["access_summary"]
    access_summary_bytes = _canonical_file(access_summary)
    routing = _read_json(
        _stage_root(runtime, "g5")
        / "f0_forensics/routing_decision.json"
    )
    role_lock = _read_json(
        _stage_root(runtime, "g4") / "f1_protocol/role_manifest_lock.json"
    )
    formal_policy = _read_json(
        _stage_root(runtime, "g4") / "f1_protocol/FORMAL_DATA_POLICY.json"
    )
    g6_report = _read_json(
        _stage_root(runtime, "g6") / "f1_protocol/f1_protocol_report.final.json"
    )
    if (
        role_lock.get("lock_sha256")
        != semantic_sha256(role_lock, ("lock_sha256",))
        or formal_policy.get("policy_sha256")
        != semantic_sha256(formal_policy, ("policy_sha256",))
        or g6_report.get("report_sha256")
        != semantic_sha256(g6_report, ("report_sha256",))
        or evidence["receipts"]["g4"]["output_hashes"].get(
            "f1_protocol/role_manifest_lock.json"
        )
        != bytes_sha256(_canonical_file(role_lock))
        or evidence["receipts"]["g4"]["output_hashes"].get(
            "f1_protocol/FORMAL_DATA_POLICY.json"
        )
        != bytes_sha256(_canonical_file(formal_policy))
        or g6_receipt["output_hashes"].get(
            "f1_protocol/f1_protocol_report.final.json"
        )
        != bytes_sha256(_canonical_file(g6_report))
    ):
        raise RuntimeControlError("G7 committed policy/protocol input drift")
    gates = _gate_records(runtime)
    gates_snapshot_base = {
        "schema_version": "c28f_v5_runtime_g0_g6_gate_snapshot_v1",
        "status": "PASS",
        "gates": gates,
    }
    gates_snapshot = {
        **gates_snapshot_base,
        "snapshot_sha256": semantic_sha256(gates_snapshot_base),
    }
    ledger_snapshot_base = {
        "schema_version": "c28f_v5_runtime_final_ledger_snapshot_v1",
        "model_forward_counts": evidence["model_forward_counts"],
        "real_data_optimizer_updates": evidence["real_data_optimizer_updates"],
        "eval_ledger_file_sha256": bytes_sha256(eval_ledger_bytes),
        "access_summary_file_sha256": bytes_sha256(access_summary_bytes),
        "stage_receipt_sha256s": {
            stage_id: receipt["receipt_sha256"]
            for stage_id, receipt in sorted(evidence["receipts"].items())
        },
    }
    ledger_snapshot = {
        **ledger_snapshot_base,
        "snapshot_sha256": semantic_sha256(ledger_snapshot_base),
    }
    final_inputs_base = {
        "schema_version": "c28f_v5_runtime_g7_final_artifact_inputs_v1",
        "routing_decision_sha256": routing["routing_decision_sha256"],
        "role_lock_sha256": role_lock["lock_sha256"],
        "formal_policy_sha256": formal_policy["policy_sha256"],
        "g6_final_report_sha256": g6_report["report_sha256"],
        "gates_snapshot_sha256": gates_snapshot["snapshot_sha256"],
        "global_ledger_snapshot_sha256": ledger_snapshot["snapshot_sha256"],
        "active_authority_sha256": authority["authority_sha256"],
        "expected_closed_authority_sha256": closed_authority[
            "authority_sha256"
        ],
        "eval_ledger_file_sha256": bytes_sha256(eval_ledger_bytes),
        "protected_data_access_summary_file_sha256": bytes_sha256(
            access_summary_bytes
        ),
    }
    final_inputs = {
        **final_inputs_base,
        "inputs_sha256": semantic_sha256(final_inputs_base),
    }
    scientific_decision = a4_finalize.build_final_decision(
        gates=gates,
        routing_status=routing["status"],
        eval_ledger_sha256=bytes_sha256(eval_ledger_bytes),
        access_summary_sha256=bytes_sha256(access_summary_bytes),
        f0_a_eval_id=state["evals"]["F0_A"]["current"]["eval_id"],
        f0_b_eval_id=state["evals"]["F0_B"]["current"]["eval_id"],
        model_forward_counts=evidence["model_forward_counts"],
        access_counts=access_counts,
        id_mapping_authority_was_active=closed_authority[
            "id_mapping_authority_was_active"
        ],
        real_data_optimizer_updates=evidence["real_data_optimizer_updates"],
        formal_policy_sha256=formal_policy["policy_sha256"],
        role_lock_sha256=role_lock["lock_sha256"],
    )
    decision_base = {
        "schema_version": "c28f_a4_f0_f1_window_decision_artifact_v1",
        "status": "F0_F1_WINDOW_READY_TO_STOP",
        "terminal_status": "F0_F1_WINDOW_COMPLETE_STOPPED",
        "goal_id": GOAL_ID,
        "attempt_id": state["attempt_id"],
        "scientific_decision": scientific_decision,
        "scientific_decision_sha256": scientific_decision["decision_sha256"],
        "gates_snapshot_sha256": gates_snapshot["snapshot_sha256"],
        "routing_decision_sha256": routing["routing_decision_sha256"],
        "g6_final_report_sha256": g6_report["report_sha256"],
        "global_ledger_snapshot_sha256": ledger_snapshot["snapshot_sha256"],
        "active_authority_sha256": authority["authority_sha256"],
        "expected_closed_authority_sha256": closed_authority[
            "authority_sha256"
        ],
        "formal_data_policy": "STRICT_CORE_ONLY",
        "U_formal_updates": 17_360,
        "final_artifact_inputs_sha256": final_inputs["inputs_sha256"],
        "eval_ledger_file_sha256": bytes_sha256(eval_ledger_bytes),
        "protected_data_access_summary_file_sha256": bytes_sha256(
            access_summary_bytes
        ),
        "next_authority_is_requested_not_granted": True,
        "manifest_is_built_after_decision_to_avoid_self_reference": True,
    }
    decision = {
        **decision_base,
        "decision_artifact_sha256": semantic_sha256(decision_base),
    }
    runtime_events = _read_jsonl(runtime.events_path)
    if not runtime_events:
        raise RuntimeControlError("G7 requires a committed runtime event tail")
    if state["status"] == "F1_COMPLETE":
        pre_state_sha256 = state["state_sha256"]
        pre_event_sha256 = runtime_events[-1]["event_sha256"]
    else:
        pending_event = runtime_events[-1]
        if (
            pending_event.get("action") != G7_FINALIZE_ACTION
            or pending_event.get("next_state_sha256") != state["state_sha256"]
        ):
            raise RuntimeControlError("G7 pending runtime event drift")
        pre_state_sha256 = state["previous_state_sha256"]
        pre_event_sha256 = pending_event["previous_event_sha256"]
    logical_finalization_id = "FINALIZATION-" + semantic_sha256(
        {
            "schema_version": "c28f_v5_runtime_finalization_identity_v1",
            "goal_id": GOAL_ID,
            "attempt_id": state["attempt_id"],
            "pre_state_sha256": pre_state_sha256,
            "pre_event_sha256": pre_event_sha256,
            "g6_stage_receipt_sha256": g6_receipt["receipt_sha256"],
            "decision_sha256": decision["decision_artifact_sha256"],
            "closed_authority_sha256": closed_authority["authority_sha256"],
        }
    )[:32]
    pending_intent_base = {
        "schema_version": "c28f_v5_runtime_pending_finalization_intent_v1",
        "status": "PENDING_FINALIZATION",
        "goal_id": GOAL_ID,
        "attempt_id": state["attempt_id"],
        "logical_finalization_id": logical_finalization_id,
        "pre_state_sha256": pre_state_sha256,
        "pre_event_sha256": pre_event_sha256,
        "g6_stage_receipt_sha256": g6_receipt["receipt_sha256"],
        "decision_sha256": decision["decision_artifact_sha256"],
        "closed_authority_sha256": closed_authority["authority_sha256"],
        "business_actions_allowed": False,
        "authority_reactivation_allowed": False,
        "new_artifacts_after_terminal_allowed": False,
    }
    pending_intent = {
        **pending_intent_base,
        "intent_sha256": semantic_sha256(pending_intent_base),
    }
    pending_candidate_base = {
        "schema_version": "c28f_v5_runtime_pending_finalization_record_v1",
        "status": "PENDING_FINALIZATION",
        "pending_intent": pending_intent,
        "pending_intent_sha256": pending_intent["intent_sha256"],
        "closed_authority": closed_authority,
        "closed_authority_sha256": closed_authority["authority_sha256"],
        "prepare_transaction_id": None,
        "pending_state_sha256": None,
        "pending_event_sha256": None,
        "terminal_commit_rule": "SAME_LOGICAL_FINALIZATION_ID_G7_STAGE_CAS",
        "business_actions_allowed": False,
        "authority_reactivation_allowed": False,
    }
    pending_candidate = {
        **pending_candidate_base,
        "record_sha256": semantic_sha256(pending_candidate_base),
    }
    final_prefix = "f1_protocol/finalization"
    pending_path = _stage_root(runtime, "g7") / pending_relative
    if state["status"] == "F1_COMPLETE":
        if pending_path.exists():
            if _read_json(pending_path) != pending_candidate:
                raise RuntimeControlError("G7 pending intent candidate drift")
        else:
            with runtime.lease("G7_PERSIST_PENDING_INTENT") as lease:
                atomic_write_immutable(
                    pending_path,
                    _canonical_file(pending_candidate),
                    lease=lease,
                )

        def pending(next_state: dict[str, Any]) -> None:
            next_state["status"] = "PENDING_FINALIZATION"
            next_state["next_action"] = G7_FINALIZE_ACTION
            workflow = next_state.get("workflow_control")
            if isinstance(workflow, dict):
                workflow["f0_b_or_f2_entry_allowed"] = False

        state = runtime.transition(
            G7_FINALIZE_ACTION,
            pending,
            artifacts={
                "g6_stage_receipt": g6_receipt["receipt_sha256"],
                "pending_finalization_intent": pending_intent["intent_sha256"],
            },
        )
    pending_tail = runtime.verify()["event_tail_sha256"]
    committed_pending_base = {
        **{
            key: value
            for key, value in pending_candidate_base.items()
            if key
            not in {
                "status",
                "prepare_transaction_id",
                "pending_state_sha256",
                "pending_event_sha256",
            }
        },
        "status": "COMMITTED",
        "prepare_transaction_id": state["transaction_id"],
        "pending_state_sha256": state["state_sha256"],
        "pending_event_sha256": pending_tail,
        "authoritative_when_terminal_state_committed": True,
    }
    pending_record = {
        **committed_pending_base,
        "record_sha256": semantic_sha256(committed_pending_base),
    }
    observed_pending = _read_json(pending_path)
    if observed_pending == pending_candidate:
        with runtime.lease("G7_MARK_PENDING_COMMITTED") as lease:
            atomic_write_bytes(
                pending_path,
                _canonical_file(pending_record),
                lease=lease,
                expected_preimage_sha256=bytes_sha256(
                    _canonical_file(pending_candidate)
                ),
            )
    elif observed_pending != pending_record:
        raise RuntimeControlError("G7 committed pending record drift")

    if routing["status"] == "ROUTING_AMBIGUOUS":
        requested_authority = "AUTH_F2_F5_PROTOTYPE"
        requested_scope = "F2_R0_PROTOTYPE_ONLY"
        request_reason = "ARBITRATE_SPLIT_DEPENDENT_EARLIEST_FAILURE_ONLY"
    else:
        requested_authority = "FROZEN_ROUTING_RECOMMENDATION_ONLY"
        requested_scope = str(routing["next_stage_recommendation_only"])
        request_reason = "PRESERVE_THE_FROZEN_CONSISTENT_ROUTING_RECOMMENDATION"
    next_request_base = {
        "schema_version": "c28f_v5_next_authorization_request_v1",
        "status": "REQUESTED_NOT_GRANTED",
        "goal_id": GOAL_ID,
        "attempt_id": state["attempt_id"],
        "routing_status": routing["status"],
        "routing_decision_sha256": routing["routing_decision_sha256"],
        "decision_artifact_sha256": decision["decision_artifact_sha256"],
        "requested_authority": requested_authority,
        "requested_scope": requested_scope,
        "request_reason": request_reason,
        "current_authority_granted": False,
        "automatic_continuation_allowed": False,
        "creates_or_starts_next_goal": False,
        "model_forward_authorized_by_this_file": False,
        "optimizer_updates_authorized_by_this_file": False,
        "f2_or_higher_executed": False,
        "holdout_or_official_authority_requested": False,
    }
    next_request = {
        **next_request_base,
        "request_sha256": semantic_sha256(next_request_base),
    }
    decision_bytes = _canonical_file(decision)
    g7_nonmanifest: dict[str, bytes] = {
        f"{final_prefix}/F0_F1_WINDOW_DECISION.json": decision_bytes,
        f"{final_prefix}/F0_F1_WINDOW_DECISION.md": (
            "# C28F F0/F1 Window Decision\n\n"
            f"Status: `{decision['status']}`.\n\n"
            f"Routing: `{routing['status']}`.\n\n"
            "No F2+, holdout, official validation, or real-data training was executed.\n"
        ).encode("utf-8"),
        f"{final_prefix}/PENDING_FINALIZATION.json": _canonical_file(
            pending_record
        ),
        f"{final_prefix}/eval_ledger.jsonl": eval_ledger_bytes,
        f"{final_prefix}/protected_data_access_summary.json": access_summary_bytes,
        f"{final_prefix}/CONTINUATION.md": (
            "# Continuation\n\n"
            "The F0/F1 window is closed. The attached request grants no authority and starts no next Goal.\n"
        ).encode("utf-8"),
        f"{final_prefix}/NEXT_AUTHORIZATION_REQUEST.json": _canonical_file(
            next_request
        ),
    }
    prior_inventory = _artifact_inventory(runtime, stage_ids=("g3", "g4", "g5", "g6"))
    current_inventory = [
        {
            "path": f"artifacts/g7/{relative}",
            "size_bytes": len(payload),
            "file_sha256": bytes_sha256(payload),
        }
        for relative, payload in sorted(g7_nonmanifest.items())
    ]
    manifest_base = {
        "schema_version": "c28f_v5_runtime_artifact_manifest_v1",
        "status": "COMPLETE_PRE_TERMINAL_POSTIMAGE",
        "goal_id": GOAL_ID,
        "attempt_id": state["attempt_id"],
        "artifacts": prior_inventory + current_inventory,
        "artifact_count": len(prior_inventory) + len(current_inventory),
        "historical_roots_included_as_outputs": False,
    }
    artifact_manifest = {
        **manifest_base,
        "manifest_sha256": semantic_sha256(manifest_base),
    }
    manifest_bytes = _canonical_file(artifact_manifest)
    root_sha = bytes_sha256(manifest_bytes)
    outputs = {
        **g7_nonmanifest,
        f"{final_prefix}/artifact_manifest.json": manifest_bytes,
        f"{final_prefix}/artifact_manifest.root.sha256": (root_sha + "\n").encode(
            "ascii"
        ),
    }

    committed = runtime.commit_stage(
        stage_id="g7",
        action=G7_FINALIZE_ACTION,
        outputs=outputs,
        input_hashes={
            "g6_stage_receipt": g6_receipt["receipt_sha256"],
            "pending_finalization": pending_record["record_sha256"],
            "decision": decision["decision_artifact_sha256"],
            "closed_authority": closed_authority["authority_sha256"],
            "artifact_manifest_root": root_sha,
        },
        safety_counts={
            "model_forward_evaluations": sum(
                evidence["model_forward_counts"].values()
            ),
            "authorized_id_mapping_operations": access_counts[
                "protected_id_mapping_operations"
            ],
            "forbidden_protected_content_opens": 0,
            "real_data_optimizer_updates": evidence[
                "real_data_optimizer_updates"
            ],
            "higher_stage_executions": 0,
        },
        mutate=terminal,
    )
    return {
        "status": committed["status"],
        "next_action": committed["next_action"],
        "decision_sha256": decision["decision_artifact_sha256"],
        "artifact_manifest_root_sha256": root_sha,
        "closed_authority_sha256": closed_authority["authority_sha256"],
        "state_sha256": committed["state_sha256"],
    }
