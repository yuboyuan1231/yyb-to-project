"""Reconcile the legacy control snapshot into the compact runtime authority.

The legacy ``goal_control`` tree is evidence only.  This module validates its
committed G1/G2/TRAIN receipts and projects those facts into ``goal_runtime``
without rewriting, replaying, or reviving the legacy state machine.
"""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

from .atomic_io import atomic_write_immutable
from .canonical import bytes_sha256, canonical_json_bytes, read_regular_bytes, semantic_sha256
from .constants import CONTROL_ROOT, GOAL_ID, REPORT_ROOT
from .runtime_store import RuntimeControlError, RuntimeStore, read_runtime_artifact
from .runtime_stages import (
    G4_ROLE_POLICY_LOCK_ACTION,
    G6_VERIFY_F1_ACTION,
    G7_FINALIZE_ACTION,
    run_f0_a_analyze,
    run_f0_b,
    run_f0_b_analyze,
    run_g4_role_policy_lock,
    run_g6_verify_f1,
    run_g7_finalize,
)


RECONCILE_CONTROL_ACTION = "RECONCILE_CONTROL"
F0_A_ANALYZE_ACTION = "F0_A_ANALYZE"

_G1_ACTION = "G1_FAIL_CLOSED_GUARD"
_G2_ACTION = "G2_METRIC_CONTRACT"
_TRAIN_ACTION = "G2_TRAIN_JSONL_INDEX_BOOTSTRAP"
_G1_TRANSACTION = "STAGE-COMMIT-937b56913495ae2b601157701f2e905c"
_G2_TRANSACTION = "STAGE-COMMIT-accb7fefbf28f157582d7892519f4003"
_TRAIN_TRANSACTION = "TRAIN-INDEX-b4e5edd34fa82b416b8e93ba31d40ffc"


def _canonical_file(value: Any) -> bytes:
    return canonical_json_bytes(value) + b"\n"


def _read_json(path: Path, *, max_bytes: int = 512 * 1024 * 1024) -> tuple[dict[str, Any], bytes]:
    data, _record = read_regular_bytes(path, max_bytes=max_bytes, require_nlink_one=True)
    try:
        value = json.loads(data.decode("utf-8", "strict"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeControlError(f"invalid JSON evidence: {path}") from exc
    if not isinstance(value, dict) or data != _canonical_file(value):
        raise RuntimeControlError(f"non-canonical JSON evidence: {path}")
    return value, data


def _read_legacy_events() -> tuple[list[dict[str, Any]], bytes]:
    path = CONTROL_ROOT / "GOAL_EVENTS.jsonl"
    data, _record = read_regular_bytes(path, max_bytes=256 * 1024 * 1024, require_nlink_one=True)
    rows: list[dict[str, Any]] = []
    previous: str | None = None
    for seq, raw in enumerate(data.splitlines(keepends=True), start=0):
        if not raw.endswith(b"\n"):
            raise RuntimeControlError(f"unterminated legacy event row {seq}")
        try:
            row = json.loads(raw.decode("utf-8", "strict"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeControlError(f"invalid legacy event row {seq}") from exc
        if (
            not isinstance(row, dict)
            or raw != _canonical_file(row)
            or row.get("goal_id") != GOAL_ID
            or row.get("seq") != seq
            or row.get("prev_event_sha256") != previous
            or row.get("event_sha256") != semantic_sha256(row, ("event_sha256",))
        ):
            raise RuntimeControlError(f"legacy event chain mismatch at seq {seq}")
        rows.append(row)
        previous = str(row["event_sha256"])
    if not rows:
        raise RuntimeControlError("legacy event journal is empty")
    return rows, data


def _event_for(rows: list[dict[str, Any]], action: str, transaction_id: str) -> dict[str, Any]:
    matches = [
        row
        for row in rows
        if row.get("action") == action and row.get("transaction_id") == transaction_id
    ]
    if len(matches) != 1:
        raise RuntimeControlError(f"legacy committed event is absent or ambiguous: {action}")
    return matches[0]


def _receipt(path: Path, *, action: str, transaction_id: str, event: Mapping[str, Any]) -> tuple[dict[str, Any], str]:
    value, encoded = _read_json(path)
    if (
        event.get("result") != "COMMITTED_EXACT_STAGE_POSTIMAGE"
        or value.get("status") != "FSYNCED_CAS_COMMITTED"
        or value.get("stage_action") != action
        or value.get("transaction_id") != transaction_id
        or value.get("event_sha256") != event.get("event_sha256")
        or value.get("receipt_sha256") != semantic_sha256(value, ("receipt_sha256",))
    ):
        raise RuntimeControlError(f"invalid committed stage receipt: {action}")
    return value, bytes_sha256(encoded)


def _verify_output_hashes(attempt_id: str, outputs: Mapping[str, Any]) -> dict[str, str]:
    root = REPORT_ROOT / attempt_id / "stage_outputs"
    verified: dict[str, str] = {}
    for relative, expected in sorted(outputs.items()):
        if (
            not isinstance(relative, str)
            or relative.startswith("/")
            or ".." in Path(relative).parts
            or Path(relative).parts[0] not in {"f0_forensics", "f1_protocol"}
            or not isinstance(expected, str)
        ):
            raise RuntimeControlError("unsafe legacy output hash entry")
        data, _record = read_regular_bytes(
            root / relative,
            max_bytes=1024 * 1024 * 1024,
            require_nlink_one=True,
        )
        observed = bytes_sha256(data)
        if observed != expected:
            raise RuntimeControlError(f"legacy committed output hash drift: {relative}")
        verified[relative] = observed
    return verified


def _train_receipt(event: Mapping[str, Any]) -> tuple[dict[str, Any], str, str]:
    root = CONTROL_ROOT / "transactions" / _TRAIN_TRANSACTION
    receipt, encoded = _read_json(root / "bootstrap_receipt.json")
    index_data, _record = read_regular_bytes(
        root / "train_jsonl_index.json",
        max_bytes=1024 * 1024 * 1024,
        require_nlink_one=True,
    )
    if (
        event.get("result") != "FSYNCED_DESCRIPTOR_BOUND_INDEX_COMMITTED_NONREPLAYABLE"
        or receipt.get("status") != event.get("result")
        or receipt.get("transaction_id") != _TRAIN_TRANSACTION
        or receipt.get("receipt_sha256") != semantic_sha256(receipt, ("receipt_sha256",))
        or receipt.get("index_file_sha256") != bytes_sha256(index_data)
        or receipt.get("replay_allowed") is not False
        or receipt.get("model_forward_evaluations") != 0
        or receipt.get("real_data_optimizer_updates") != 0
        or receipt.get("unauthorized_protected_payload_decode_count") != 0
    ):
        raise RuntimeControlError("invalid TRAIN index bootstrap receipt")
    return receipt, bytes_sha256(encoded), bytes_sha256(index_data)


def _ledger_safety(attempt_id: str, g1_outputs: Mapping[str, str]) -> dict[str, Any]:
    relative = "f1_protocol/ledger_safety_manifest.json"
    path = REPORT_ROOT / attempt_id / "stage_outputs" / relative
    manifest, encoded = _read_json(path)
    totals = manifest.get("cumulative_totals")
    claims = manifest.get("claims")
    if (
        bytes_sha256(encoded) != g1_outputs.get(relative)
        or manifest.get("manifest_sha256") != semantic_sha256(manifest, ("manifest_sha256",))
        or not isinstance(totals, dict)
        or not isinstance(claims, dict)
        or claims.get("derived_from_persisted_global_snapshot") is not True
        or claims.get("derived_from_incident_registry") is not True
        or manifest.get("incident_count") != 0
        or manifest.get("real_data_optimizer_updates") != 0
        or any(
            totals.get(name) != 0
            for name in (
                "historical_write_bytes",
                "observed_unmediated_content_open_count",
                "protected_content_open_count",
                "protected_feature_open_count",
                "protected_label_open_count",
                "protected_metric_open_count",
                "protected_prediction_open_count",
                "protected_query_text_timestamp_open_count",
                "unresolved_open_intent_count",
            )
        )
    ):
        raise RuntimeControlError("G1 ledger-derived safety manifest is invalid")
    return {
        "source_file_sha256": bytes_sha256(encoded),
        "manifest_sha256": manifest["manifest_sha256"],
        "cumulative_totals": deepcopy(totals),
        "incident_count": 0,
        "real_data_optimizer_updates": 0,
        "derivation_claims": deepcopy(claims),
    }


def _authority_projection() -> tuple[dict[str, Any], str]:
    ledger, encoded = _read_json(CONTROL_ROOT / "authority_ledger.json")
    permissions = ledger.get("permissions")
    forbidden = ("AUTH_F2_F5_PROTOTYPE", "AUTH_F6_MECHANISM", "AUTH_F8_FORMAL", "AUTH_HOLDOUT_OFFICIAL")
    if (
        ledger.get("goal_id") != GOAL_ID
        or ledger.get("formal_data_policy") != "STRICT_CORE_ONLY"
        or ledger.get("u_formal_updates") != 17360
        or ledger.get("real_data_training_authorized") is not False
        or ledger.get("protected_model_evaluation_authorized") is not False
        or ledger.get("forbidden_authorities_remain_ungranted") is not True
        or not isinstance(permissions, dict)
        or any(
            permissions.get(name, {}).get("state") != "UNGRANTED"
            or permissions.get(name, {}).get("value") is not False
            for name in forbidden
        )
        or permissions.get("AUTH_F0_F1_IMPLEMENTATION", {}).get("state") != "ACTIVE"
        or permissions.get("AUTH_F0_F1_IMPLEMENTATION", {}).get("value") is not True
        or permissions.get("AUTH_PROTECTED_ID_MAPPING_ONLY", {}).get("state") != "ACTIVE"
        or permissions.get("AUTH_PROTECTED_ID_MAPPING_ONLY", {}).get("value") is not True
        or permissions.get("AUTH_PROTECTED_ID_MAPPING_ONLY", {}).get("scope")
        != "desc_id_to_video_id_only_no_text_timestamp_feature_label_metric"
        or ledger.get("authority_sha256") != semantic_sha256(ledger, ("authority_sha256",))
    ):
        raise RuntimeControlError("legacy authority ledger projection is invalid")
    projection = {
        "formal_data_policy": "STRICT_CORE_ONLY",
        "u_formal": 17360,
        "f0_f1_implementation": "ACTIVE",
        "protected_id_mapping_scope": permissions["AUTH_PROTECTED_ID_MAPPING_ONLY"]["scope"],
        "real_data_training_authorized": False,
        "protected_model_evaluation_authorized": False,
        "forbidden_authorities": {name: "UNGRANTED" for name in forbidden},
        "authority_semantic_sha256": ledger["authority_sha256"],
    }
    return projection, bytes_sha256(encoded)


def _workflow_code_hashes() -> dict[str, str]:
    package_root = Path(__file__).resolve(strict=True).parent
    repo_root = package_root.parent.parent
    sources = {
        "blueprint_e2e_v2/c28f_v5/runtime_control.py": package_root / "runtime_control.py",
        "blueprint_e2e_v2/c28f_v5/runtime_migration.py": package_root / "runtime_migration.py",
        "blueprint_e2e_v2/c28f_v5/runtime_runner.py": package_root / "runtime_runner.py",
        "blueprint_e2e_v2/c28f_v5/runtime_store.py": package_root / "runtime_store.py",
        "blueprint_e2e_v2/c28f_v5/runtime_workflow.py": package_root / "runtime_workflow.py",
        "run_c28f_v5_f0_f1.py": repo_root / "run_c28f_v5_f0_f1.py",
    }
    result: dict[str, str] = {}
    for relative, path in sorted(sources.items()):
        data, _record = read_regular_bytes(
            path, max_bytes=8 * 1024 * 1024, require_nlink_one=True
        )
        result[relative] = bytes_sha256(data)
    return result


def _build_bridge(runtime: RuntimeStore, state: Mapping[str, Any]) -> dict[str, Any]:
    if state.get("status") != "F0_A_COMPLETED" or state.get("next_action") != F0_A_ANALYZE_ACTION:
        raise RuntimeControlError("control reconciliation requires completed F0-A analysis gate")
    current = state.get("evals", {}).get("F0_A", {}).get("current")
    if not isinstance(current, dict) or current.get("status") != "COMPLETED":
        raise RuntimeControlError("completed F0-A token is absent")
    material_pointer = current.get("completed_material")
    expected_material_path = (
        runtime.root / "evals" / str(current["eval_id"]) / "completed_material.json"
    )
    if (
        not isinstance(material_pointer, dict)
        or set(material_pointer) != {"path", "material_binding_sha256"}
        or Path(material_pointer["path"]) != expected_material_path
    ):
        raise RuntimeControlError("completed F0-A material pointer is absent or unsafe")
    completed_material = read_runtime_artifact(expected_material_path)
    if (
        completed_material.get("material_binding_sha256")
        != material_pointer.get("material_binding_sha256")
        or completed_material.get("material_binding_sha256")
        != semantic_sha256(completed_material, ("material_binding_sha256",))
    ):
        raise RuntimeControlError("completed F0-A material binding drift")

    legacy_state, legacy_state_bytes = _read_json(CONTROL_ROOT / "GOAL_STATE.json")
    events, event_bytes = _read_legacy_events()
    g1_event = _event_for(events, _G1_ACTION, _G1_TRANSACTION)
    g2_event = _event_for(events, _G2_ACTION, _G2_TRANSACTION)
    train_event = _event_for(events, _TRAIN_ACTION, _TRAIN_TRANSACTION)
    g1_receipt, g1_receipt_file_sha = _receipt(
        CONTROL_ROOT / "transactions" / _G1_TRANSACTION / "commit_receipt.json",
        action=_G1_ACTION,
        transaction_id=_G1_TRANSACTION,
        event=g1_event,
    )
    g2_receipt, g2_receipt_file_sha = _receipt(
        CONTROL_ROOT / "transactions" / _G2_TRANSACTION / "commit_receipt.json",
        action=_G2_ACTION,
        transaction_id=_G2_TRANSACTION,
        event=g2_event,
    )
    attempt_id = str(state["attempt_id"])
    if (
        legacy_state.get("goal_id") != GOAL_ID
        or legacy_state.get("attempt_id") != attempt_id
        or legacy_state.get("state_sha256")
        != semantic_sha256(legacy_state, ("state_sha256",))
    ):
        raise RuntimeControlError("legacy/runtime state identity or self-hash mismatch")
    g1_outputs = _verify_output_hashes(attempt_id, g1_receipt["output_hashes"])
    g2_outputs = _verify_output_hashes(attempt_id, g2_receipt["output_hashes"])
    train_receipt, train_receipt_file_sha, train_index_file_sha = _train_receipt(train_event)
    safety = _ledger_safety(attempt_id, g1_outputs)
    authority, authority_file_sha = _authority_projection()

    base = {
        "schema_version": "c28f_v5_goal_control_supersession_bridge_v1",
        "status": "FSYNCED_LEGACY_EVIDENCE_REPROJECTED_RUNTIME_AUTHORITATIVE",
        "goal_id": GOAL_ID,
        "attempt_id": attempt_id,
        "authority_rule": {
            "authoritative_state_root": str(runtime.root),
            "legacy_state_root": str(CONTROL_ROOT),
            "legacy_root_mode": "READ_ONLY_SUPERSEDED_EVIDENCE",
            "legacy_state_fields_are_not_live_facts": True,
        },
        "runtime_anchor": {
            "state_sha256": state["state_sha256"],
            "transition_seq": state["transition_seq"],
            "status": state["status"],
            "next_action": state["next_action"],
            "f0_a_eval_id": current["eval_id"],
            "f0_a_run_id": current["run_id"],
            "f0_a_generation": current["generation"],
            "completed_material_binding_sha256": material_pointer["material_binding_sha256"],
        },
        "legacy_snapshot": {
            "state_file_sha256": bytes_sha256(legacy_state_bytes),
            "state_semantic_sha256": legacy_state.get("state_sha256"),
            "events_file_sha256": bytes_sha256(event_bytes),
            "events_tail_sha256": events[-1]["event_sha256"],
            "events_count": len(events),
            "stale_live_fields_ignored": ["status", "stage", "next_action", "completed_gates", "eval_token_states"],
        },
        "gate_projection": {
            "G1": {
                "status": "COMPLETED_REPROJECTED",
                "action": _G1_ACTION,
                "transaction_id": _G1_TRANSACTION,
                "event_sha256": g1_event["event_sha256"],
                "receipt_sha256": g1_receipt["receipt_sha256"],
                "receipt_file_sha256": g1_receipt_file_sha,
                "output_hashes": g1_outputs,
            },
            "G2": {
                "status": "COMPLETED_REPROJECTED",
                "action": _G2_ACTION,
                "transaction_id": _G2_TRANSACTION,
                "event_sha256": g2_event["event_sha256"],
                "receipt_sha256": g2_receipt["receipt_sha256"],
                "receipt_file_sha256": g2_receipt_file_sha,
                "output_hashes": g2_outputs,
            },
            "TRAIN_JSONL_INDEX": {
                "status": "COMPLETED_NONREPLAYABLE_REPROJECTED",
                "action": _TRAIN_ACTION,
                "transaction_id": _TRAIN_TRANSACTION,
                "event_sha256": train_event["event_sha256"],
                "receipt_sha256": train_receipt["receipt_sha256"],
                "receipt_file_sha256": train_receipt_file_sha,
                "index_binding_sha256": train_receipt["index_binding_sha256"],
                "index_file_sha256": train_index_file_sha,
                "source_file_sha256": train_receipt["source_file_sha256"],
                "line_count": train_receipt["line_count"],
                "replay_allowed": False,
            },
        },
        "ledger_derived_safety": safety,
        "authority_projection": authority,
        "authority_ledger_file_sha256": authority_file_sha,
        "runtime_code_source_sha256": _workflow_code_hashes(),
        "next_action_contract": {
            "action": F0_A_ANALYZE_ACTION,
            "automatic_execution": False,
            "f0_b_or_f2_entry_allowed": False,
        },
    }
    return {**base, "bridge_sha256": semantic_sha256(base)}


def reconcile_control(store: RuntimeStore | None = None) -> dict[str, Any]:
    runtime = store or RuntimeStore()
    state = runtime.read_state()
    existing = state.get("legacy_snapshot", {}).get("supersession_bridge")
    bridge_path = runtime.root / "control_bridge" / "supersession_bridge.json"
    if isinstance(existing, dict):
        if (
            set(existing) != {"path", "bridge_sha256"}
            or Path(existing["path"]) != bridge_path
        ):
            raise RuntimeControlError("persisted supersession bridge pointer is unsafe")
        bridge = read_runtime_artifact(bridge_path)
        if (
            bridge.get("bridge_sha256") != existing.get("bridge_sha256")
            or bridge.get("bridge_sha256") != semantic_sha256(bridge, ("bridge_sha256",))
        ):
            raise RuntimeControlError("persisted supersession bridge drift")
        return {
            "status": "ALREADY_RECONCILED",
            "bridge_sha256": bridge["bridge_sha256"],
            "state_sha256": state["state_sha256"],
            "next_action": state["next_action"],
        }

    bridge = _build_bridge(runtime, state)
    with runtime.lease("PERSIST_CONTROL_SUPERSESSION_BRIDGE") as lease:
        atomic_write_immutable(bridge_path, _canonical_file(bridge), lease=lease)

    pointer = {"path": str(bridge_path), "bridge_sha256": bridge["bridge_sha256"]}
    workflow_control = {
        "schema_version": "c28f_v5_runtime_workflow_control_v1",
        "authoritative_state_root": "goal_runtime",
        "legacy_state_disposition": "READ_ONLY_SUPERSEDED_EVIDENCE",
        "supersession_bridge": pointer,
        "gate_status": {
            name: item["status"]
            for name, item in bridge["gate_projection"].items()
        },
        "formal_data_policy": bridge["authority_projection"]["formal_data_policy"],
        "u_formal": bridge["authority_projection"]["u_formal"],
        "next_action": F0_A_ANALYZE_ACTION,
        "automatic_execution": False,
        "f0_b_or_f2_entry_allowed": False,
    }

    def attach(next_state: dict[str, Any]) -> None:
        snapshot = next_state.get("legacy_snapshot")
        if not isinstance(snapshot, dict):
            raise RuntimeControlError("runtime legacy snapshot is invalid")
        snapshot["supersession_bridge"] = pointer
        snapshot["authoritative_state_root"] = "goal_runtime"
        snapshot["legacy_state_disposition"] = "READ_ONLY_SUPERSEDED_EVIDENCE"
        next_state["workflow_control"] = workflow_control

    committed = runtime.transition(
        RECONCILE_CONTROL_ACTION,
        attach,
        artifacts={
            "supersession_bridge": bridge["bridge_sha256"],
            "g1_receipt": bridge["gate_projection"]["G1"]["receipt_sha256"],
            "g2_receipt": bridge["gate_projection"]["G2"]["receipt_sha256"],
            "train_index_receipt": bridge["gate_projection"]["TRAIN_JSONL_INDEX"]["receipt_sha256"],
        },
        require_next_action=False,
    )
    return {
        "status": "RECONCILED",
        "bridge_sha256": bridge["bridge_sha256"],
        "state_sha256": committed["state_sha256"],
        "next_action": committed["next_action"],
        "gate_status": {name: item["status"] for name, item in bridge["gate_projection"].items()},
        "legacy_state_disposition": "READ_ONLY_SUPERSEDED_EVIDENCE",
    }
