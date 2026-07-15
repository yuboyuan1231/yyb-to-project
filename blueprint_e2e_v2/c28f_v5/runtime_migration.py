"""One-time, read-only migration from the legacy A4 control tree.

Only already-persisted control/runtime evidence is inspected.  The TRAIN
JSONL, protected splits, feature values, checkpoint tensor, and model are not
opened by this migration.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping

from .canonical import bytes_sha256, canonical_json_bytes, read_regular_bytes, semantic_sha256
from .constants import CONTROL_ROOT, GOAL_ID
from .runtime_store import RuntimeControlError, RuntimeStore


MIGRATE_ACTION = "MIGRATE_ZERO_FORWARD_F0_A"
ISSUE_F0_A_ACTION = "ISSUE_F0_A_GENERATION"


def _read_json(path: Path, *, max_bytes: int = 1024 * 1024 * 1024) -> tuple[dict[str, Any], bytes]:
    data, _record = read_regular_bytes(path, max_bytes=max_bytes, require_nlink_one=True)
    try:
        value = json.loads(data.decode("utf-8", "strict"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeControlError(f"legacy JSON is invalid: {path}") from exc
    canonical = canonical_json_bytes(value) if isinstance(value, dict) else b""
    if not isinstance(value, dict) or data not in {canonical, canonical + b"\n"}:
        raise RuntimeControlError(f"legacy JSON is non-canonical: {path}")
    return value, data


def _read_legacy_events(path: Path) -> tuple[list[dict[str, Any]], bytes]:
    data, _record = read_regular_bytes(path, max_bytes=256 * 1024 * 1024, require_nlink_one=True)
    rows: list[dict[str, Any]] = []
    previous: str | None = None
    for seq, raw in enumerate(data.splitlines(keepends=True)):
        if not raw.endswith(b"\n"):
            raise RuntimeControlError("legacy event journal has an unterminated row")
        try:
            row = json.loads(raw.decode("utf-8", "strict"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeControlError(f"legacy event row {seq} is invalid") from exc
        if (
            not isinstance(row, dict)
            or raw != canonical_json_bytes(row) + b"\n"
            or row.get("seq") != seq
            or row.get("goal_id") != GOAL_ID
            or row.get("prev_event_sha256") != previous
            or row.get("event_sha256") != semantic_sha256(row, ("event_sha256",))
        ):
            raise RuntimeControlError(f"legacy event chain mismatch at seq {seq}")
        rows.append(row)
        previous = str(row["event_sha256"])
    if not rows:
        raise RuntimeControlError("legacy event journal is empty")
    return rows, data


def _hash_record(path: Path, data: bytes) -> dict[str, Any]:
    return {
        "path": str(path.resolve(strict=True)),
        "file_sha256": bytes_sha256(data),
        "size_bytes": len(data),
    }


def _directory_has_payload(root: Path) -> bool:
    if not root.exists():
        return False
    if root.is_symlink() or not root.is_dir():
        raise RuntimeControlError(f"unexpected legacy runtime object: {root}")
    for _base, directories, files in os.walk(root, followlinks=False):
        if files:
            return True
        for name in directories:
            child = Path(_base) / name
            if child.is_symlink():
                raise RuntimeControlError(f"legacy runtime contains a symlink: {child}")
    return False


def inspect_zero_forward_legacy_generation() -> dict[str, Any]:
    state_path = CONTROL_ROOT / "GOAL_STATE.json"
    events_path = CONTROL_ROOT / "GOAL_EVENTS.jsonl"
    registry_path = CONTROL_ROOT / "eval_token_registry.json"
    state, state_bytes = _read_json(state_path)
    events, events_bytes = _read_legacy_events(events_path)
    registry, registry_bytes = _read_json(registry_path)
    if (
        state.get("goal_id") != GOAL_ID
        or state.get("state_sha256") != semantic_sha256(state, ("state_sha256",))
        or state.get("transition_seq") != events[-1].get("seq")
        or events[-1].get("expected_next_state_sha256") != state.get("state_sha256")
        or state.get("status") != "RUNNING"
        or state.get("next_action") != "COMPLETE_EVAL_TOKEN"
        or state.get("eval_token_states", {}).get("F0_A") != "RUNNING"
        or state.get("eval_token_states", {}).get("F0_B") != "UNISSUED"
    ):
        raise RuntimeControlError("legacy state is not the approved active F0-A tail")
    if (
        registry.get("goal_id") != GOAL_ID
        or registry.get("token_id") != "F0_A"
        or registry.get("token_state") != "RUNNING"
        or registry.get("model_forward_evaluations") != 0
        or registry.get("completion") is not None
        or registry.get("run_id") != state.get("active_eval", {}).get("transaction_stem")
    ):
        raise RuntimeControlError("legacy eval registry is not a zero-forward RUNNING F0-A")

    run_id = str(registry["run_id"])
    running_root = CONTROL_ROOT / "transactions" / f"{run_id}-RUNNING"
    phase_path = running_root / "phase_receipt.json"
    cursor_path = running_root / "eval_cursor.json"
    phase, phase_bytes = _read_json(phase_path)
    cursor, cursor_bytes = _read_json(cursor_path)
    if (
        phase.get("phase") != "RUNNING"
        or phase.get("model_forward_evaluations") != 0
        or phase.get("run_id") != run_id
        or phase.get("receipt_sha256") != semantic_sha256(phase, ("receipt_sha256",))
        or cursor.get("cursor_index") != 0
        or cursor.get("run_id") != run_id
        or cursor.get("receipt_sha256") != semantic_sha256(cursor, ("receipt_sha256",))
    ):
        raise RuntimeControlError("legacy RUNNING/cursor evidence is not zero-forward")

    reservation_path = CONTROL_ROOT / "transactions" / f"{run_id}-RESERVE" / "reservation_input.json"
    reservation, reservation_bytes = _read_json(reservation_path)
    source_lineage = reservation.get("payload", {}).get("source_lineage")
    origin = (
        source_lineage.get("controller_forward_authority_origin_transaction_id")
        if isinstance(source_lineage, Mapping)
        else None
    )
    if not isinstance(origin, str) or not origin:
        raise RuntimeControlError("legacy reservation lacks its authority origin")
    authority_path = CONTROL_ROOT / "transactions" / origin / "controller_forward_authority.json"
    authority, authority_bytes = _read_json(authority_path)
    projection = authority.get("active_rehydrate_projection")
    payload = authority.get("payload")
    if (
        authority.get("schema_version") != "c28f_a4_controller_forward_authority_record_v1"
        or authority.get("record_sha256") != semantic_sha256(authority, ("record_sha256",))
        or not isinstance(projection, dict)
        or not isinstance(payload, dict)
        or projection.get("projection_binding_sha256")
        != semantic_sha256(projection, ("projection_binding_sha256",))
        or authority.get("active_rehydrate_projection_sha256")
        != projection.get("projection_binding_sha256")
        or authority.get("capability_sha256")
        != source_lineage.get("controller_forward_authority_handle_sha256")
        or payload.get("authority_binding_sha256")
        != registry.get("execution_binding", {}).get("authority_binding_sha256")
    ):
        raise RuntimeControlError("legacy forward authority record is invalid")

    material = projection.get("authority_binding_material")
    manifest = projection.get("manifest")
    if not isinstance(material, dict) or not isinstance(manifest, dict):
        raise RuntimeControlError("legacy authority descriptor projection is incomplete")
    output_root = material.get("output_root", {}).get("path")
    cache_root = material.get("cache_root", {}).get("path")
    if not isinstance(output_root, str) or not isinstance(cache_root, str):
        raise RuntimeControlError("legacy authority write roots are missing")
    run_root = Path(output_root) / str(manifest.get("stage")) / str(state["attempt_id"]) / run_id
    run_receipt_path = run_root / "forward_run_receipt.json"
    observations_path = run_root / "forward_observations.json"
    run_receipt, run_receipt_bytes = _read_json(run_receipt_path)
    observations, observations_bytes = _read_json(observations_path)
    rows = observations.get("rows")
    observation_chain_valid = isinstance(rows, list)
    previous_observation: str | None = None
    if observation_chain_valid:
        for seq, row in enumerate(rows, start=1):
            if (
                not isinstance(row, dict)
                or row.get("seq") != seq
                or row.get("run_id") != run_id
                or row.get("previous_event_sha256") != previous_observation
                or row.get("event_sha256")
                != semantic_sha256(row, ("event_sha256",))
            ):
                observation_chain_valid = False
                break
            previous_observation = str(row["event_sha256"])
    if (
        run_receipt.get("status") != "RUNNING"
        or run_receipt.get("run_id") != run_id
        or run_receipt.get("next_query_index") != 0
        or run_receipt.get("query_count") != 8677
        or run_receipt.get("result_chunks") != []
        or run_receipt.get("encoded_cache_fingerprint") is not None
        or run_receipt.get("encoded_cache_receipt_sha256") is not None
        or run_receipt.get("receipt_sha256")
        != semantic_sha256(run_receipt, ("receipt_sha256",))
        or any(run_receipt.get(name) != 0 for name in (
            "teacher_candidate_count",
            "teacher_forward_count",
            "gt_support_count",
            "gt_append_count",
            "optimizer_update_count",
        ))
        or not isinstance(rows, list)
        or not observation_chain_valid
        or observations.get("event_count") != len(rows)
        or observations.get("ledger_sha256")
        != semantic_sha256(observations, ("ledger_sha256",))
        or any(row.get("counters", {}).get("model_forward_count") != 0 for row in rows)
    ):
        raise RuntimeControlError("legacy forward artifact contains scientific progress")
    query_chunks = run_root / "query_chunks"
    if _directory_has_payload(query_chunks) or (run_root / "control_completion_receipt.json").exists():
        raise RuntimeControlError("legacy forward generation has committed result payload")
    encoded_cache = Path(cache_root) / "encoded_corpus" / str(projection["encoded_cache_fingerprint"])
    if _directory_has_payload(encoded_cache):
        raise RuntimeControlError("legacy zero-forward generation has a materialized encoded cache")

    files = {
        "legacy_state": _hash_record(state_path, state_bytes),
        "legacy_events": _hash_record(events_path, events_bytes),
        "legacy_eval_registry": _hash_record(registry_path, registry_bytes),
        "legacy_running_phase": _hash_record(phase_path, phase_bytes),
        "legacy_cursor": _hash_record(cursor_path, cursor_bytes),
        "legacy_reservation": _hash_record(reservation_path, reservation_bytes),
        "legacy_authority": _hash_record(authority_path, authority_bytes),
        "legacy_forward_receipt": _hash_record(run_receipt_path, run_receipt_bytes),
        "legacy_forward_observations": _hash_record(observations_path, observations_bytes),
    }
    snapshot_base = {
        "schema_version": "c28f_v5_legacy_zero_forward_snapshot_v1",
        "disposition": "ABORTED_ZERO_FORWARD_REISSUE_AUTHORIZED",
        "legacy_attempt_id": state["attempt_id"],
        "legacy_eval_id": registry["eval_id"],
        "legacy_run_id": run_id,
        "legacy_authority_record_path": str(authority_path.resolve(strict=True)),
        "legacy_forward_observation_count": len(rows),
        "model_forward_count": 0,
        "result_chunk_count": 0,
        "next_query_index": 0,
        "protected_content_opens": int(state["protected_content_opens_current_attempt"]),
        "optimizer_updates": int(state["real_data_optimizer_updates"]),
        "files": files,
    }
    return {**snapshot_base, "snapshot_sha256": semantic_sha256(snapshot_base)}


def migrate_zero_forward_f0_a(store: RuntimeStore | None = None) -> dict[str, Any]:
    runtime = store or RuntimeStore()
    if runtime.exists():
        return runtime.verify(recover=True)
    snapshot = inspect_zero_forward_legacy_generation()
    if snapshot["protected_content_opens"] != 0 or snapshot["optimizer_updates"] != 0:
        raise RuntimeControlError("legacy safety counters prohibit migration")
    generation_one = {
        "generation": 1,
        "status": "ABORTED",
        "eval_id": snapshot["legacy_eval_id"],
        "run_id": snapshot["legacy_run_id"],
        "reason": "ZERO_FORWARD_CODE_IDENTITY_DRIFT",
        "model_forward_evaluations": 0,
        "result_chunk_count": 0,
        "legacy_snapshot_sha256": snapshot["snapshot_sha256"],
    }
    state = runtime.initialize(
        {
            "attempt_id": snapshot["legacy_attempt_id"],
            "status": "READY",
            "next_action": ISSUE_F0_A_ACTION,
            "evals": {
                "F0_A": {
                    "max_successful_evaluations": 1,
                    "successful_evaluations": 0,
                    "next_generation": 2,
                    "current": None,
                    "history": [generation_one],
                },
                "F0_B": {
                    "max_successful_evaluations": 1,
                    "successful_evaluations": 0,
                    "next_generation": 1,
                    "current": None,
                    "history": [],
                },
            },
            "safety": {
                "model_forward_evaluations": 0,
                "protected_content_opens": 0,
                "optimizer_updates": 0,
                "teacher_forward_count": 0,
                "gt_support_count": 0,
                "aborted_zero_forward_generations": 1,
            },
            "legacy_snapshot": snapshot,
        },
        action=MIGRATE_ACTION,
        artifacts={
            name: str(record["file_sha256"])
            for name, record in snapshot["files"].items()
        },
    )
    return {
        "status": "MIGRATED",
        "state_sha256": state["state_sha256"],
        "next_action": state["next_action"],
        "legacy_generation": generation_one,
    }
