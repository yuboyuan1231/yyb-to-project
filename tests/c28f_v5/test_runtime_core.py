from __future__ import annotations

import io
from types import SimpleNamespace

import numpy as np
import pytest

from blueprint_e2e_v2.c28f_v5 import a4_forward, runtime_control, runtime_workflow
from blueprint_e2e_v2.c28f_v5.constants import GOAL_ID
from blueprint_e2e_v2.c28f_v5.runtime_migration import (
    inspect_zero_forward_legacy_generation,
)
from blueprint_e2e_v2.c28f_v5.runtime_store import (
    EVENT_SCHEMA,
    RUNTIME_ROOT,
    RUNTIME_SCHEMA,
    RuntimeControlError,
    seal_event,
    seal_state,
    verify_event_chain,
    verify_state_value,
)
from blueprint_e2e_v2.c28f_v5.canonical import semantic_sha256


def _state() -> dict:
    return seal_state(
        {
            "schema_version": RUNTIME_SCHEMA,
            "goal_id": GOAL_ID,
            "attempt_id": "attempt",
            "transition_seq": 1,
            "previous_state_sha256": None,
            "transaction_id": "transaction",
            "status": "READY",
            "next_action": "ISSUE_F0_A_GENERATION",
            "evals": {
                token_id: {
                    "max_successful_evaluations": 1,
                    "successful_evaluations": 0,
                    "next_generation": 1,
                    "current": None,
                    "history": [],
                }
                for token_id in ("F0_A", "F0_B")
            },
            "safety": {
                "model_forward_evaluations": 0,
                "protected_content_opens": 0,
                "optimizer_updates": 0,
                "teacher_forward_count": 0,
                "gt_support_count": 0,
                "aborted_zero_forward_generations": 0,
            },
            "legacy_snapshot": {},
            "updated_at_ns": 1,
        }
    )


def test_runtime_state_and_event_hashes_reject_tampering() -> None:
    state = _state()
    assert verify_state_value(state)["state_sha256"] == state["state_sha256"]
    event = seal_event(
        {
            "schema_version": EVENT_SCHEMA,
            "goal_id": GOAL_ID,
            "seq": 1,
            "transaction_id": "transaction",
            "action": "MIGRATE_ZERO_FORWARD_F0_A",
            "result": "COMMITTED",
            "previous_event_sha256": None,
            "previous_state_sha256": None,
            "next_state_sha256": state["state_sha256"],
            "artifact_sha256": {},
        }
    )
    verify_event_chain([event])
    changed = dict(state)
    changed["status"] = "ALTERED"
    with pytest.raises(RuntimeControlError, match="self-hash"):
        verify_state_value(changed)


def test_runtime_workflow_projection_is_first_class_and_fail_closed() -> None:
    state = _state()
    state.pop("state_sha256")
    state["next_action"] = "F0_A_ANALYZE"
    state["workflow_control"] = {
        "schema_version": "c28f_v5_runtime_workflow_control_v1",
        "authoritative_state_root": "goal_runtime",
        "legacy_state_disposition": "READ_ONLY_SUPERSEDED_EVIDENCE",
        "supersession_bridge": {
            "path": str(RUNTIME_ROOT / "control_bridge" / "supersession_bridge.json"),
            "bridge_sha256": "a" * 64,
        },
        "gate_status": {
            "G1": "COMPLETED_REPROJECTED",
            "G2": "COMPLETED_REPROJECTED",
            "TRAIN_JSONL_INDEX": "COMPLETED_NONREPLAYABLE_REPROJECTED",
        },
        "formal_data_policy": "STRICT_CORE_ONLY",
        "u_formal": 17360,
        "next_action": "F0_A_ANALYZE",
        "automatic_execution": False,
        "f0_b_or_f2_entry_allowed": False,
    }
    state = seal_state(state)
    assert verify_state_value(state)["workflow_control"]["gate_status"]["G2"].startswith("COMPLETED")

    changed = dict(state)
    changed["workflow_control"] = dict(state["workflow_control"])
    changed["workflow_control"]["u_formal"] = 17359
    changed = seal_state(changed)
    with pytest.raises(RuntimeControlError, match="workflow control"):
        verify_state_value(changed)


def test_runtime_capabilities_are_exact_and_immutable() -> None:
    payload = {
        "schema_version": "c28f_a4_eval_running_capability_v1",
        "goal_id": GOAL_ID,
        "attempt_id": "attempt",
        "run_id": "run",
        "token_id": "F0_A",
        "eval_id": "eval",
        "transaction_id": "transaction",
        "running_state_sha256": "1" * 64,
        "running_event_sha256": "2" * 64,
        "eval_registry_sha256": "3" * 64,
        "execution_binding_sha256": "4" * 64,
        "reservation_input_capability_sha256": "5" * 64,
    }
    capability = runtime_control.A4EvalRunningCapability(
        runtime_control._ISSUER,
        payload,
    )
    assert runtime_control.validate_eval_running_capability(capability) == payload
    with pytest.raises(runtime_control.A4ControlError, match="immutable"):
        capability._payload = {}


def test_legacy_migration_probe_is_zero_forward_and_read_only() -> None:
    snapshot = inspect_zero_forward_legacy_generation()
    assert snapshot["disposition"] == "ABORTED_ZERO_FORWARD_REISSUE_AUTHORIZED"
    assert snapshot["model_forward_count"] == 0
    assert snapshot["result_chunk_count"] == 0
    assert snapshot["protected_content_opens"] == 0
    assert snapshot["optimizer_updates"] == 0


def test_legacy_commits_are_available_for_runtime_reprojection() -> None:
    rows, encoded = runtime_workflow._read_legacy_events()
    assert encoded.endswith(b"\n")
    expected = {
        "G1_FAIL_CLOSED_GUARD": "STAGE-COMMIT-937b56913495ae2b601157701f2e905c",
        "G2_METRIC_CONTRACT": "STAGE-COMMIT-accb7fefbf28f157582d7892519f4003",
        "G2_TRAIN_JSONL_INDEX_BOOTSTRAP": "TRAIN-INDEX-b4e5edd34fa82b416b8e93ba31d40ffc",
    }
    for action, transaction_id in expected.items():
        row = runtime_workflow._event_for(rows, action, transaction_id)
        assert row["transaction_id"] == transaction_id

    authority, authority_file_sha256 = runtime_workflow._authority_projection()
    assert authority["formal_data_policy"] == "STRICT_CORE_ONLY"
    assert authority["u_formal"] == 17360
    assert authority["real_data_training_authorized"] is False
    assert len(authority_file_sha256) == 64


def test_forward_adapter_surface_is_small_and_complete() -> None:
    required = {
        "assert_fsynced_control_chain",
        "validate_controller_forward_authority",
        "consume_forward_capability_once",
        "assert_running_state_event_committed",
        "committed_token_id",
        "committed_eval_id",
        "committed_run_id",
        "committed_token_snapshot_sha256",
        "issue_eval_running_capability",
        "commit_lmdb_batch",
        "issue_persisted_lmdb_batch_control_receipt",
        "commit_eval_cursor",
        "issue_eval_cursor_capability",
        "complete_eval_token",
        "persist_completed_forward_rehydrate_material",
    }
    assert required <= set(dir(runtime_control.A4EvalForwardControlAdapter))


def test_zero_forward_abort_validator_rejects_any_forward() -> None:
    current = {"eval_id": "eval", "run_id": "run"}
    row_base = {
        "schema_version": "c28f_v5_a4_forward_observation_v1",
        "seq": 1,
        "event_type": "LMDB_VALUE_READ",
        "token_id": "F0_A",
        "eval_id": "eval",
        "run_id": "run",
        "previous_event_sha256": None,
        "counters": {
            "content_open_count": 1,
            "content_bytes_read": 10,
            "lmdb_transaction_open_count": 0,
            "model_forward_count": 0,
        },
        "details": {},
    }
    row = {**row_base, "event_sha256": semantic_sha256(row_base)}
    ledger_base = {
        "schema_version": "c28f_v5_a4_forward_observation_ledger_v1",
        "run_id": "run",
        "capability_id": "capability",
        "event_count": 1,
        "rows": [row],
    }
    ledger = {**ledger_base, "ledger_sha256": semantic_sha256(ledger_base)}
    summary = runtime_control._validate_zero_forward_observation_ledger(
        ledger, current=current
    )
    assert summary["model_forward_count"] == 0

    changed_base = dict(row_base)
    changed_base["counters"] = {**row_base["counters"], "model_forward_count": 1}
    changed_row = {
        **changed_base,
        "event_sha256": semantic_sha256(changed_base),
    }
    changed_ledger_base = {**ledger_base, "rows": [changed_row]}
    changed_ledger = {
        **changed_ledger_base,
        "ledger_sha256": semantic_sha256(changed_ledger_base),
    }
    with pytest.raises(runtime_control.A4ControlError, match="after model forward"):
        runtime_control._validate_zero_forward_observation_ledger(
            changed_ledger, current=current
        )


def test_npz_decoder_accepts_only_exact_legacy_boolean_marker() -> None:
    features = np.ones((2, 3), dtype=np.float32)
    legacy = io.BytesIO()
    np.savez(legacy, features=features, allow_pickle=True)
    decoded = a4_forward._decode_npz_features(
        legacy.getvalue(),
        expected_dim=3,
        max_sequence_length=4,
        allow_empty=False,
    )
    np.testing.assert_array_equal(decoded, features)

    invalid = io.BytesIO()
    np.savez(invalid, features=features, allow_pickle=np.array([True]))
    with pytest.raises(a4_forward.ForwardContractError, match="NPZ_LEGACY_MARKER_INVALID"):
        a4_forward._decode_npz_features(
            invalid.getvalue(),
            expected_dim=3,
            max_sequence_length=4,
            allow_empty=False,
        )

    extra = io.BytesIO()
    np.savez(extra, features=features, unexpected=np.array(0))
    with pytest.raises(
        a4_forward.ForwardContractError,
        match="NPZ_MEMBER_NAME_CONTRACT_VIOLATION",
    ):
        a4_forward._decode_npz_features(
            extra.getvalue(),
            expected_dim=3,
            max_sequence_length=4,
            allow_empty=False,
        )


def test_hot_observations_are_aggregated_with_exact_counters() -> None:
    authority = object.__new__(a4_forward.FrozenInputAuthority)
    authority._observation_rows = []
    authority._pending_observation_summary = {}
    authority._assert_capability_identity = lambda _capability: None
    capability = SimpleNamespace(
        run_id="run",
        token_id="F0_A",
        eval_id="eval",
        capability_id="capability",
        _run_receipt_parent=None,
    )
    zero = {
        "content_open_count": 0,
        "content_bytes_read": 0,
        "lmdb_transaction_open_count": 0,
        "model_forward_count": 0,
    }
    authority._record_forward_observation(
        capability,
        event_type="LMDB_VALUE_READ",
        counters={**zero, "content_open_count": 1, "content_bytes_read": 10},
        details={"key": "a"},
    )
    authority._record_forward_observation(
        capability,
        event_type="QUERY_ENCODER_FORWARD",
        counters={**zero, "model_forward_count": 1},
        details={"query": "q"},
    )
    assert authority._observation_rows == []
    authority.flush_forward_observations(capability)
    assert len(authority._observation_rows) == 1
    row = authority._observation_rows[0]
    assert row["event_type"] == "AGGREGATED_FORWARD_OBSERVATIONS"
    assert row["counters"] == {
        "content_open_count": 1,
        "content_bytes_read": 10,
        "lmdb_transaction_open_count": 0,
        "model_forward_count": 1,
    }
    assert row["details"]["aggregated_event_count"] == 2
    assert row["details"]["event_type_counts"] == {
        "LMDB_VALUE_READ": 1,
        "QUERY_ENCODER_FORWARD": 1,
    }
