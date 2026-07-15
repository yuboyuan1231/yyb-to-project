"""Pure final-decision projections and post-commit consistency checks.

Nothing in this module grants authority, persists a PENDING intent, commits a
terminal state, or releases a lease.  Those operations belong exclusively to
``a4_control.A4ControlFinalizationAdapter`` and its private snapshot capability.
"""

from __future__ import annotations

import re
from typing import Any, Mapping, Optional, Sequence

from .canonical import canonical_json_bytes, semantic_sha256


FINAL_DECISION_SCHEMA = "c28f_f0_f1_window_decision_v1"
FINALIZATION_SCHEMA = "c28f_finalization_transaction_v1"
TERMINAL_STATUS = "F0_F1_WINDOW_COMPLETE_STOPPED"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

FINAL_GATE_FIELDS = {
    "gate_id",
    "status",
    "state_sha256",
    "event_sha256",
    "control_receipt_sha256",
}

ACCESS_COUNT_KEYS = {
    "protected_model_eval",
    "protected_feature_get",
    "protected_prediction_read",
    "protected_metric_read",
    "protected_query_text_read",
    "protected_timestamp_read",
    "protected_label_read",
    "official_workflow",
    "forbidden_value_materialization",
    "unresolved_open_intent",
    "c28e_historical_write_bytes",
    "teacher_model_forward",
    "gt_support_injection",
    "gt_append",
    "posthoc_temperature_refit_during_eval",
    "protected_id_mapping_operations",
}

HIGH_AUTHORITY_KEYS = (
    "AUTH_F2_F5_PROTOTYPE",
    "AUTH_F6_MECHANISM",
    "AUTH_F8_FORMAL",
    "AUTH_HOLDOUT_OFFICIAL",
    "AUTH_GIT_BRANCH_CREATE_SWITCH",
    "AUTH_GIT_COMMIT",
    "AUTH_GIT_PUSH_PR",
)


class FinalizationError(RuntimeError):
    pass


def _sha(value: Any, name: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise FinalizationError(f"{name} must be an exact lowercase SHA-256")
    return value


def close_authority_ledger(
    authority: Mapping[str, Any], *, mapping_token_state: Optional[str]
) -> dict[str, Any]:
    current = dict(authority)
    supplied_authority_sha = current.get("authority_sha256")
    _sha(supplied_authority_sha, "input authority ledger")
    if supplied_authority_sha != semantic_sha256(
        current, excluded_fields=("authority_sha256",)
    ):
        raise FinalizationError("input authority ledger self-hash mismatch")
    raw_permissions = current.get("permissions")
    if not isinstance(raw_permissions, Mapping):
        raise FinalizationError("authority ledger lacks a permissions registry")
    if any(not isinstance(value, Mapping) for value in raw_permissions.values()):
        raise FinalizationError("authority permission records must be mappings")
    permissions = {str(key): dict(value) for key, value in raw_permissions.items()}
    expected_permission_keys = {
        "AUTH_F0_F1_IMPLEMENTATION",
        "AUTH_PROTECTED_ID_MAPPING_ONLY",
        *HIGH_AUTHORITY_KEYS,
    }
    if set(permissions) != expected_permission_keys:
        raise FinalizationError("authority permission registry is not exact")
    if permissions.get("AUTH_F0_F1_IMPLEMENTATION", {}).get("state") != "ACTIVE" or permissions.get(
        "AUTH_F0_F1_IMPLEMENTATION", {}
    ).get("value") is not True:
        raise FinalizationError("F0/F1 implementation authority is not ACTIVE")
    mapping_permission = permissions.get("AUTH_PROTECTED_ID_MAPPING_ONLY", {})
    mapping_was_active = False
    if mapping_permission.get("state") == "ACTIVE" and mapping_permission.get(
        "value"
    ) is True:
        if mapping_token_state != "COMPLETED_NONREPLAYABLE":
            raise FinalizationError(
                "active ID-only authority requires a completed nonreplayable token"
            )
        mapping_was_active = True
    elif mapping_permission.get("state") == "UNGRANTED" and mapping_permission.get(
        "value"
    ) is False:
        if mapping_token_state is not None:
            raise FinalizationError("ungranted ID-only authority cannot have a token state")
    else:
        raise FinalizationError("ID-only authority lifecycle drift")
    for key in HIGH_AUTHORITY_KEYS:
        if permissions.get(key, {}).get("state") != "UNGRANTED" or permissions.get(
            key, {}
        ).get("value") is not False:
            raise FinalizationError(f"higher authority drift: {key}")
    permissions["AUTH_F0_F1_IMPLEMENTATION"]["state"] = "CONSUMED_CLOSED"
    permissions["AUTH_F0_F1_IMPLEMENTATION"]["value"] = False
    if mapping_was_active:
        permissions["AUTH_PROTECTED_ID_MAPPING_ONLY"]["state"] = "CONSUMED_CLOSED"
        permissions["AUTH_PROTECTED_ID_MAPPING_ONLY"]["value"] = False
    current["permissions"] = permissions
    current["closed_by_stage"] = "G7"
    current["reactivation_allowed"] = False
    current["id_mapping_authority_was_active"] = mapping_was_active
    current["mapping_token_terminal_state"] = mapping_token_state
    current.pop("authority_sha256", None)
    return {**current, "authority_sha256": semantic_sha256(current)}


def _require_all_gates(gates: Mapping[str, Any]) -> None:
    expected = [f"G{index}" for index in range(7)]
    if set(gates) != set(expected):
        raise FinalizationError("finalization gate set is not exact")
    for gate in expected:
        record = gates.get(gate)
        if not isinstance(record, Mapping) or set(record) != FINAL_GATE_FIELDS:
            raise FinalizationError(f"finalization gate receipt is not exact: {gate}")
        if record.get("gate_id") != gate or record.get("status") != "PASS":
            raise FinalizationError(f"finalization requires {gate}=PASS")
        for field in ("state_sha256", "event_sha256", "control_receipt_sha256"):
            _sha(record.get(field), f"{gate}.{field}")


def build_final_decision(
    *,
    gates: Mapping[str, Any],
    routing_status: str,
    eval_ledger_sha256: str,
    access_summary_sha256: str,
    f0_a_eval_id: str,
    f0_b_eval_id: str,
    model_forward_counts: Mapping[str, int],
    access_counts: Mapping[str, int],
    id_mapping_authority_was_active: bool,
    real_data_optimizer_updates: int,
    formal_policy_sha256: str,
    role_lock_sha256: str,
) -> dict[str, Any]:
    _require_all_gates(gates)
    if routing_status not in {"ROUTING_CONSISTENT", "ROUTING_AMBIGUOUS"}:
        raise FinalizationError("routing status is outside the frozen F0 decision set")
    for value, name in (
        (eval_ledger_sha256, "eval ledger"),
        (access_summary_sha256, "access summary"),
        (formal_policy_sha256, "formal policy"),
        (role_lock_sha256, "role lock"),
    ):
        _sha(value, name)
    if not f0_a_eval_id or not f0_b_eval_id or f0_a_eval_id == f0_b_eval_id:
        raise FinalizationError("F0-A/F0-B require distinct non-empty logical eval IDs")
    expected_forward_counts = {"F0_A": 1, "F0_B": 1, "protected": 0, "other": 0}
    if any(type(value) is not int for value in model_forward_counts.values()):
        raise FinalizationError("model-forward counts must be exact integers")
    if dict(model_forward_counts) != expected_forward_counts:
        raise FinalizationError("model-forward ledger counts do not close exactly")
    if set(access_counts) != ACCESS_COUNT_KEYS:
        raise FinalizationError("forbidden/protected access ledger keys are not exact")
    forbidden_access_keys = ACCESS_COUNT_KEYS - {"protected_id_mapping_operations"}
    if any(
        type(access_counts[key]) is not int or access_counts[key] != 0
        for key in forbidden_access_keys
    ):
        raise FinalizationError("forbidden/protected access count is non-zero")
    if not isinstance(id_mapping_authority_was_active, bool):
        raise FinalizationError("ID mapping authority history must be boolean")
    expected_mapping_operations = 1 if id_mapping_authority_was_active else 0
    if type(access_counts.get("protected_id_mapping_operations")) is not int or access_counts.get(
        "protected_id_mapping_operations"
    ) != expected_mapping_operations:
        raise FinalizationError("ID-only mapping operation count does not match authority history")
    if type(real_data_optimizer_updates) is not int or real_data_optimizer_updates != 0:
        raise FinalizationError("real-data optimizer updates must remain zero")
    base = {
        "schema_version": FINAL_DECISION_SCHEMA,
        "status": "F0_F1_WINDOW_READY_TO_STOP",
        "terminal_status": TERMINAL_STATUS,
        "routing_status": routing_status,
        "gates": dict(gates),
        "eval_ledger_sha256": eval_ledger_sha256,
        "access_summary_sha256": access_summary_sha256,
        "logical_evals": {
            "F0_A": {"eval_id": f0_a_eval_id, "count": 1},
            "F0_B": {"eval_id": f0_b_eval_id, "count": 1},
        },
        "model_forward_counts": dict(model_forward_counts),
        "access_counts": dict(access_counts),
        "id_mapping_authority_was_active": id_mapping_authority_was_active,
        "real_data_optimizer_updates": 0,
        "formal_policy_sha256": formal_policy_sha256,
        "role_lock_sha256": role_lock_sha256,
        "holdout_run": False,
        "official_validation_run": False,
        "higher_stage_executed": False,
        "next_authority_is_requested_not_granted": True,
        "persistence_claimed_by_pure_function": False,
        "requires_committed_control_snapshot": True,
    }
    return {**base, "decision_sha256": semantic_sha256(base)}


def build_finalization_intent_candidate(
    *,
    finalization_txn_id: str,
    goal_id: str,
    attempt_id: str,
    pre_state_sha256: str,
    pre_event_sha256: str,
    final_artifact_hashes: Mapping[str, str],
    closed_authority: Mapping[str, Any],
    pending_state_sha256: str,
    pending_event_sha256: str,
    terminal_state_sha256: str,
    terminal_event_sha256: str,
    lease_identity_sha256: str,
) -> dict[str, Any]:
    if not finalization_txn_id or not goal_id or not attempt_id:
        raise FinalizationError("finalization identity fields are required")
    for value, name in (
        (pre_state_sha256, "pre state"),
        (pre_event_sha256, "pre event"),
        (pending_state_sha256, "pending state"),
        (pending_event_sha256, "pending event"),
        (terminal_state_sha256, "terminal state"),
        (terminal_event_sha256, "terminal event"),
        (lease_identity_sha256, "lease identity"),
    ):
        _sha(value, name)
    closed_permissions = closed_authority.get("permissions")
    if not isinstance(closed_permissions, Mapping) or closed_permissions.get(
        "AUTH_F0_F1_IMPLEMENTATION", {}
    ).get("state") != "CONSUMED_CLOSED":
        raise FinalizationError("intent authority is not closed")
    closed_authority_sha256 = closed_authority.get("authority_sha256")
    _sha(closed_authority_sha256, "closed authority")
    if closed_authority_sha256 != semantic_sha256(
        dict(closed_authority), excluded_fields=("authority_sha256",)
    ):
        raise FinalizationError("closed authority self-hash mismatch")
    normalized_artifacts: dict[str, str] = {}
    for path, digest in sorted(final_artifact_hashes.items()):
        if not isinstance(path, str) or not path or path.startswith("/") or ".." in path.split("/"):
            raise FinalizationError("final artifact path is not canonical relative")
        normalized_artifacts[path] = _sha(digest, f"artifact {path}")
    if not normalized_artifacts:
        raise FinalizationError("final artifact manifest must not be empty")
    base = {
        "schema_version": FINALIZATION_SCHEMA,
        "status": "CANDIDATE_NOT_FSYNCED",
        "finalization_txn_id": finalization_txn_id,
        "goal_id": goal_id,
        "attempt_id": attempt_id,
        "pre_state_sha256": pre_state_sha256,
        "pre_event_sha256": pre_event_sha256,
        "final_artifact_hashes": normalized_artifacts,
        "closed_authority_sha256": closed_authority_sha256,
        "pending_state_sha256": pending_state_sha256,
        "pending_event_sha256": pending_event_sha256,
        "terminal_state_sha256": terminal_state_sha256,
        "terminal_event_sha256": terminal_event_sha256,
        "lease_identity_sha256": lease_identity_sha256,
        "lease_release_intent": "RELEASE_ONLY_AFTER_TERMINAL_RECEIPT_VERIFIER_PASS",
        "post_intent_allowed_actions": [
            "FINALIZATION_RECOVERY",
            "COMMIT_CLOSED_AUTHORITY",
            "COMMIT_TERMINAL_EVENT_STATE",
            "COMMIT_FINALIZATION_RECEIPT",
            "RELEASE_LEASE",
            "VERIFY_TERMINAL_READ_ONLY",
        ],
        "business_actions_allowed_after_intent": False,
        "authority_reactivation_allowed": False,
        "control_commit_eligible": False,
        "persistence_claimed_by_pure_function": False,
    }
    return {**base, "intent_sha256": semantic_sha256(base)}


def build_expected_finalization_commit_projection(
    intent: Mapping[str, Any],
    *,
    observed_artifact_hashes: Mapping[str, str],
    observed_authority_sha256: str,
    observed_terminal_state_sha256: str,
    observed_terminal_event_sha256: str,
) -> dict[str, Any]:
    current = dict(intent)
    if current.get("intent_sha256") != semantic_sha256(
        current, excluded_fields=("intent_sha256",)
    ):
        raise FinalizationError("finalization intent self-hash mismatch")
    if current.get("status") != "CANDIDATE_NOT_FSYNCED":
        raise FinalizationError("only a candidate finalization projection may be checked")
    normalized_observed = {
        str(path): _sha(digest, f"observed artifact {path}")
        for path, digest in sorted(observed_artifact_hashes.items())
    }
    if normalized_observed != current.get("final_artifact_hashes"):
        raise FinalizationError("final artifact postimage differs from intent")
    expected = (
        (observed_authority_sha256, current["closed_authority_sha256"], "authority"),
        (observed_terminal_state_sha256, current["terminal_state_sha256"], "state"),
        (observed_terminal_event_sha256, current["terminal_event_sha256"], "event"),
    )
    for observed, wanted, name in expected:
        if observed != wanted:
            raise FinalizationError(f"terminal {name} postimage differs from intent")
    base = {
        "schema_version": FINALIZATION_SCHEMA,
        "status": "EXPECTED_POSTIMAGE_NOT_COMMITTED",
        "finalization_txn_id": current["finalization_txn_id"],
        "intent_sha256": current["intent_sha256"],
        "observed_artifact_hashes": normalized_observed,
        "observed_authority_sha256": observed_authority_sha256,
        "observed_terminal_state_sha256": observed_terminal_state_sha256,
        "observed_terminal_event_sha256": observed_terminal_event_sha256,
        "lease_release_authorized": False,
        "authority_reactivation_allowed": False,
        "business_actions_allowed": False,
        "persistence_claimed_by_pure_function": False,
    }
    return {**base, "projection_sha256": semantic_sha256(base)}


def verify_terminal_closure(
    *,
    goal_state: Mapping[str, Any],
    last_event: Mapping[str, Any],
    authority: Mapping[str, Any],
    finalization_receipt: Mapping[str, Any],
    eval_token_states: Mapping[str, str],
    mapping_token_state: Optional[str],
    live_goal_processes: Sequence[int],
) -> dict[str, Any]:
    if goal_state.get("state_sha256") != semantic_sha256(
        dict(goal_state), excluded_fields=("state_sha256",)
    ):
        raise FinalizationError("terminal GOAL_STATE self-hash mismatch")
    if last_event.get("event_sha256") != semantic_sha256(
        dict(last_event), excluded_fields=("event_sha256",)
    ):
        raise FinalizationError("terminal event self-hash mismatch")
    if authority.get("authority_sha256") != semantic_sha256(
        dict(authority), excluded_fields=("authority_sha256",)
    ):
        raise FinalizationError("terminal authority self-hash mismatch")
    if finalization_receipt.get("receipt_sha256") != semantic_sha256(
        dict(finalization_receipt), excluded_fields=("receipt_sha256",)
    ):
        raise FinalizationError("terminal finalization receipt self-hash mismatch")
    if goal_state.get("status") != TERMINAL_STATUS or goal_state.get("next_action") is not None:
        raise FinalizationError("terminal GOAL_STATE is not stopped")
    if last_event.get("result") != TERMINAL_STATUS:
        raise FinalizationError("terminal event result is not exact")
    if last_event.get("expected_next_state_sha256") != goal_state.get("state_sha256"):
        raise FinalizationError("terminal event/state binding mismatch")
    permissions = authority.get("permissions")
    if not isinstance(permissions, Mapping):
        raise FinalizationError("terminal authority lacks permissions")
    if permissions.get("AUTH_F0_F1_IMPLEMENTATION", {}).get(
        "state"
    ) != "CONSUMED_CLOSED":
        raise FinalizationError("terminal F0/F1 authority is not closed")
    mapping_was_active = authority.get("id_mapping_authority_was_active")
    expected_mapping_state = "CONSUMED_CLOSED" if mapping_was_active is True else "UNGRANTED"
    if permissions.get("AUTH_PROTECTED_ID_MAPPING_ONLY", {}).get(
        "state"
    ) != expected_mapping_state:
        raise FinalizationError("terminal ID-only authority state is wrong")
    for key in HIGH_AUTHORITY_KEYS:
        if permissions.get(key, {}).get("state") != "UNGRANTED" or permissions.get(
            key, {}
        ).get("value") is not False:
            raise FinalizationError(f"terminal higher authority drift: {key}")
    if authority.get("reactivation_allowed") is not False:
        raise FinalizationError("terminal authority permits reactivation")
    if dict(eval_token_states) != {"F0_A": "COMPLETED", "F0_B": "COMPLETED"}:
        raise FinalizationError("terminal eval tokens are not exactly completed")
    expected_token_state = (
        "COMPLETED_NONREPLAYABLE" if mapping_was_active is True else None
    )
    if mapping_token_state != expected_token_state:
        raise FinalizationError("terminal ID mapping token state is wrong")
    if finalization_receipt.get("status") != "COMMITTED" or not finalization_receipt.get(
        "lease_release_authorized"
    ):
        raise FinalizationError("finalization receipt does not authorize release")
    receipt_bindings = {
        "observed_authority_sha256": authority.get("authority_sha256"),
        "observed_terminal_state_sha256": goal_state.get("state_sha256"),
        "observed_terminal_event_sha256": last_event.get("event_sha256"),
    }
    if any(finalization_receipt.get(key) != value for key, value in receipt_bindings.items()):
        raise FinalizationError("finalization receipt terminal bindings mismatch")
    if live_goal_processes:
        raise FinalizationError("Goal-owned process remains live at terminal verification")
    base = {
        "schema_version": "c28f_terminal_verification_v1",
        "status": "PASS",
        "terminal_status": TERMINAL_STATUS,
        "state_sha256": goal_state.get("state_sha256"),
        "event_sha256": last_event.get("event_sha256"),
        "authority_sha256": authority.get("authority_sha256"),
        "finalization_receipt_sha256": finalization_receipt.get("receipt_sha256"),
        "eval_token_states": dict(eval_token_states),
        "mapping_token_state": mapping_token_state,
        "live_goal_processes": [],
    }
    return {**base, "verification_sha256": semantic_sha256(base)}


def canonical_finalization_bytes(value: Mapping[str, Any]) -> bytes:
    return canonical_json_bytes(dict(value)) + b"\n"
