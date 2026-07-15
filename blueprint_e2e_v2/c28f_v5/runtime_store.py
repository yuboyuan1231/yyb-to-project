"""Small durable state machine for the C28F F0/F1 runtime.

This module deliberately owns only four concerns: one writer, immutable
records, a state pointer, and a hash-chained event journal.  Scientific I/O
and model execution stay in :mod:`a4_forward`; historical bootstrap policy
stays in the read-only legacy control tree.
"""

from __future__ import annotations

import json
import time
import uuid
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .atomic_io import (
    WriterLease,
    atomic_write_bytes,
    atomic_write_immutable,
    atomic_write_json,
)
from .canonical import bytes_sha256, canonical_json_bytes, read_regular_bytes, semantic_sha256
from .constants import GOAL_ID, REPORT_ROOT


RUNTIME_SCHEMA = "c28f_v5_runtime_state_v1"
EVENT_SCHEMA = "c28f_v5_runtime_event_v1"
TRANSACTION_SCHEMA = "c28f_v5_runtime_transaction_v1"
STAGE_RECEIPT_SCHEMA = "c28f_v5_runtime_stage_receipt_v1"
RUNTIME_ROOT = REPORT_ROOT / "goal_runtime"


class RuntimeControlError(RuntimeError):
    """A compact, user-actionable runtime control failure."""


def _canonical_file(value: Any) -> bytes:
    return canonical_json_bytes(value) + b"\n"


def _read_json(path: Path, *, max_bytes: int = 512 * 1024 * 1024) -> tuple[dict[str, Any], bytes]:
    data, _record = read_regular_bytes(path, max_bytes=max_bytes, require_nlink_one=True)
    try:
        value = json.loads(data.decode("utf-8", "strict"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeControlError(f"invalid canonical JSON: {path}") from exc
    if not isinstance(value, dict) or data != _canonical_file(value):
        raise RuntimeControlError(f"non-canonical JSON: {path}")
    return value, data


def _read_events(path: Path) -> tuple[list[dict[str, Any]], bytes]:
    data, _record = read_regular_bytes(path, max_bytes=256 * 1024 * 1024, require_nlink_one=True)
    rows: list[dict[str, Any]] = []
    for line_number, raw in enumerate(data.splitlines(keepends=True), start=1):
        if not raw.endswith(b"\n"):
            raise RuntimeControlError(f"unterminated event row {line_number}")
        try:
            value = json.loads(raw.decode("utf-8", "strict"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeControlError(f"invalid event row {line_number}") from exc
        if not isinstance(value, dict) or raw != _canonical_file(value):
            raise RuntimeControlError(f"non-canonical event row {line_number}")
        rows.append(value)
    return rows, data


def seal_state(value: Mapping[str, Any]) -> dict[str, Any]:
    state = deepcopy(dict(value))
    state.pop("state_sha256", None)
    state["state_sha256"] = semantic_sha256(state)
    return state


def seal_event(value: Mapping[str, Any]) -> dict[str, Any]:
    event = deepcopy(dict(value))
    event.pop("event_sha256", None)
    event["event_sha256"] = semantic_sha256(event)
    return event


def verify_state_value(
    state: Mapping[str, Any], *, runtime_root: Path = RUNTIME_ROOT
) -> dict[str, Any]:
    value = dict(state)
    required = {
        "schema_version",
        "goal_id",
        "attempt_id",
        "transition_seq",
        "previous_state_sha256",
        "transaction_id",
        "status",
        "next_action",
        "evals",
        "safety",
        "legacy_snapshot",
        "updated_at_ns",
        "state_sha256",
    }
    optional = {"workflow_control", "stage_receipts"}
    if not required.issubset(value) or not set(value).issubset(required | optional):
        raise RuntimeControlError("runtime state field set mismatch")
    if value["schema_version"] != RUNTIME_SCHEMA or value["goal_id"] != GOAL_ID:
        raise RuntimeControlError("runtime state identity mismatch")
    if type(value["transition_seq"]) is not int or value["transition_seq"] < 0:
        raise RuntimeControlError("runtime transition sequence is invalid")
    if type(value["updated_at_ns"]) is not int or value["updated_at_ns"] <= 0:
        raise RuntimeControlError("runtime update time is invalid")
    if (
        not isinstance(value["evals"], dict)
        or not isinstance(value["safety"], dict)
        or not isinstance(value["legacy_snapshot"], dict)
    ):
        raise RuntimeControlError("runtime state mappings are invalid")
    if set(value["evals"]) != {"F0_A", "F0_B"}:
        raise RuntimeControlError("runtime token set is invalid")
    for token_id, token in value["evals"].items():
        if (
            not isinstance(token, dict)
            or set(token)
            != {
                "max_successful_evaluations",
                "successful_evaluations",
                "next_generation",
                "current",
                "history",
            }
            or token["max_successful_evaluations"] != 1
            or type(token["successful_evaluations"]) is not int
            or not 0 <= token["successful_evaluations"] <= 1
            or type(token["next_generation"]) is not int
            or token["next_generation"] < 1
            or not isinstance(token["history"], list)
            or (token["current"] is not None and not isinstance(token["current"], dict))
        ):
            raise RuntimeControlError(f"runtime token state is invalid: {token_id}")
    safety_fields = {
        "model_forward_evaluations",
        "protected_content_opens",
        "optimizer_updates",
        "teacher_forward_count",
        "gt_support_count",
        "aborted_zero_forward_generations",
    }
    if (
        set(value["safety"]) != safety_fields
        or any(type(item) is not int or item < 0 for item in value["safety"].values())
        or any(
            value["safety"][name] != 0
            for name in (
                "protected_content_opens",
                "optimizer_updates",
                "teacher_forward_count",
                "gt_support_count",
            )
        )
        or value["safety"]["model_forward_evaluations"]
        != sum(token["successful_evaluations"] for token in value["evals"].values())
    ):
        raise RuntimeControlError("runtime safety budget is invalid")
    workflow = value.get("workflow_control")
    if workflow is not None:
        workflow_fields = {
            "schema_version",
            "authoritative_state_root",
            "legacy_state_disposition",
            "supersession_bridge",
            "gate_status",
            "formal_data_policy",
            "u_formal",
            "next_action",
            "automatic_execution",
            "f0_b_or_f2_entry_allowed",
        }
        pointer = workflow.get("supersession_bridge") if isinstance(workflow, dict) else None
        gate_status = workflow.get("gate_status") if isinstance(workflow, dict) else None
        digest = pointer.get("bridge_sha256") if isinstance(pointer, dict) else None
        if (
            not isinstance(workflow, dict)
            or set(workflow) != workflow_fields
            or workflow.get("schema_version")
            != "c28f_v5_runtime_workflow_control_v1"
            or workflow.get("authoritative_state_root") != "goal_runtime"
            or workflow.get("legacy_state_disposition")
            != "READ_ONLY_SUPERSEDED_EVIDENCE"
            or not isinstance(pointer, dict)
            or set(pointer) != {"path", "bridge_sha256"}
            or pointer.get("path")
            != str(runtime_root / "control_bridge" / "supersession_bridge.json")
            or not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
            or gate_status
            != {
                "G1": "COMPLETED_REPROJECTED",
                "G2": "COMPLETED_REPROJECTED",
                "TRAIN_JSONL_INDEX": "COMPLETED_NONREPLAYABLE_REPROJECTED",
            }
            or workflow.get("formal_data_policy") != "STRICT_CORE_ONLY"
            or workflow.get("u_formal") != 17360
            or workflow.get("next_action") != value.get("next_action")
            or workflow.get("automatic_execution") is not False
            or type(workflow.get("f0_b_or_f2_entry_allowed")) is not bool
        ):
            raise RuntimeControlError("runtime workflow control is invalid")
    stage_receipts = value.get("stage_receipts")
    if stage_receipts is not None:
        if not isinstance(stage_receipts, dict):
            raise RuntimeControlError("runtime stage receipt registry is invalid")
        for stage_id, pointer in stage_receipts.items():
            expected_path = runtime_root / "artifacts" / str(stage_id) / "STAGE_RECEIPT.json"
            if (
                not isinstance(stage_id, str)
                or not stage_id
                or not isinstance(pointer, dict)
                or set(pointer) != {"path", "receipt_sha256"}
                or pointer.get("path") != str(expected_path)
                or not isinstance(pointer.get("receipt_sha256"), str)
                or len(pointer["receipt_sha256"]) != 64
                or any(character not in "0123456789abcdef" for character in pointer["receipt_sha256"])
            ):
                raise RuntimeControlError("runtime stage receipt pointer is invalid")
    if value.get("state_sha256") != semantic_sha256(value, ("state_sha256",)):
        raise RuntimeControlError("runtime state self-hash mismatch")
    return value


def verify_event_chain(rows: list[dict[str, Any]]) -> None:
    previous: str | None = None
    for seq, row in enumerate(rows, start=1):
        required = {
            "schema_version",
            "goal_id",
            "seq",
            "transaction_id",
            "action",
            "result",
            "previous_event_sha256",
            "previous_state_sha256",
            "next_state_sha256",
            "artifact_sha256",
            "event_sha256",
        }
        if (
            set(row) != required
            or row.get("schema_version") != EVENT_SCHEMA
            or row.get("goal_id") != GOAL_ID
            or row.get("seq") != seq
            or row.get("previous_event_sha256") != previous
            or not isinstance(row.get("artifact_sha256"), dict)
            or row.get("event_sha256") != semantic_sha256(row, ("event_sha256",))
        ):
            raise RuntimeControlError(f"runtime event chain mismatch at seq {seq}")
        previous = str(row["event_sha256"])


class RuntimeStore:
    """Durable runtime store with one recoverable pointer-update window."""

    def __init__(self, root: Path = RUNTIME_ROOT) -> None:
        self.root = root
        self.state_path = root / "STATE.json"
        self.events_path = root / "EVENTS.jsonl"
        self.lock_path = root / "single_writer.lock"
        self.heartbeat_path = root / "single_writer.heartbeat.json"

    def lease(self, action: str) -> WriterLease:
        return WriterLease(self.lock_path, self.heartbeat_path, GOAL_ID, action)

    def exists(self) -> bool:
        return self.state_path.is_file() and self.events_path.is_file()

    def _load_locked(self, lease: WriterLease, *, recover: bool) -> tuple[dict[str, Any], list[dict[str, Any]], bytes, bytes]:
        lease.assert_active_capability()
        state, state_bytes = _read_json(self.state_path)
        rows, event_bytes = _read_events(self.events_path)
        verify_state_value(state, runtime_root=self.root)
        verify_event_chain(rows)
        state_seq = int(state["transition_seq"])
        if len(rows) == state_seq + 1 and recover:
            candidate = rows[-1]
            record_path = self.root / "transactions" / str(candidate["transaction_id"]) / "record.json"
            record, _record_bytes = _read_json(record_path)
            next_state = record.get("next_state")
            event = record.get("event")
            if (
                record.get("schema_version") != TRANSACTION_SCHEMA
                or record.get("transaction_sha256")
                != semantic_sha256(record, ("transaction_sha256",))
                or event != candidate
                or not isinstance(next_state, dict)
                or next_state.get("state_sha256") != candidate.get("next_state_sha256")
                or next_state.get("state_sha256")
                != semantic_sha256(next_state, ("state_sha256",))
                or next_state.get("transition_seq") != len(rows)
                or next_state.get("previous_state_sha256") != state.get("state_sha256")
            ):
                raise RuntimeControlError("unrecoverable runtime pointer window")
            atomic_write_json(
                self.state_path,
                next_state,
                lease=lease,
                expected_preimage_sha256=bytes_sha256(state_bytes),
            )
            state, state_bytes = _read_json(self.state_path)
            verify_state_value(state, runtime_root=self.root)
            state_seq = int(state["transition_seq"])
        if len(rows) != state_seq:
            raise RuntimeControlError("runtime state/event sequence mismatch")
        if rows:
            tail = rows[-1]
            if tail["next_state_sha256"] != state["state_sha256"]:
                raise RuntimeControlError("runtime state is not the event tail postimage")
        elif state_seq != 0:
            raise RuntimeControlError("runtime state has no event history")
        return state, rows, state_bytes, event_bytes

    def verify(self, *, recover: bool = True) -> dict[str, Any]:
        with self.lease("VERIFY_RUNTIME") as lease:
            state, rows, _state_bytes, _event_bytes = self._load_locked(lease, recover=recover)
            return {
                "status": "VALID",
                "transition_seq": state["transition_seq"],
                "state_sha256": state["state_sha256"],
                "event_tail_sha256": rows[-1]["event_sha256"] if rows else None,
                "runtime_status": state["status"],
                "next_action": state["next_action"],
                "evals": deepcopy(state["evals"]),
                "safety": deepcopy(state["safety"]),
            }

    def read_state(self, *, recover: bool = True) -> dict[str, Any]:
        with self.lease("READ_RUNTIME") as lease:
            state, _rows, _state_bytes, _event_bytes = self._load_locked(lease, recover=recover)
            return deepcopy(state)

    def initialize(self, state_values: Mapping[str, Any], *, action: str, artifacts: Mapping[str, str]) -> dict[str, Any]:
        with self.lease(action) as lease:
            if not self.state_path.exists() and self.events_path.exists():
                rows, _event_bytes = _read_events(self.events_path)
                verify_event_chain(rows)
                if len(rows) != 1 or rows[0].get("previous_state_sha256") is not None:
                    raise RuntimeControlError("partial runtime genesis is not recoverable")
                record_path = (
                    self.root
                    / "transactions"
                    / str(rows[0]["transaction_id"])
                    / "record.json"
                )
                record, _record_bytes = _read_json(record_path)
                next_state = record.get("next_state")
                if (
                    record.get("schema_version") != TRANSACTION_SCHEMA
                    or record.get("transaction_sha256")
                    != semantic_sha256(record, ("transaction_sha256",))
                    or record.get("event") != rows[0]
                    or not isinstance(next_state, dict)
                    or next_state.get("state_sha256")
                    != rows[0].get("next_state_sha256")
                    or next_state.get("state_sha256")
                    != semantic_sha256(next_state, ("state_sha256",))
                    or next_state.get("transition_seq") != 1
                    or next_state.get("previous_state_sha256") is not None
                ):
                    raise RuntimeControlError("partial runtime genesis record is invalid")
                atomic_write_json(
                    self.state_path,
                    next_state,
                    lease=lease,
                    precondition="MUST_BE_ABSENT",
                )
            if self.state_path.exists() or self.events_path.exists():
                if not self.state_path.exists() or not self.events_path.exists():
                    raise RuntimeControlError("runtime genesis pointer set is incomplete")
                state, _rows, _state_bytes, _event_bytes = self._load_locked(lease, recover=True)
                return deepcopy(state)
            transaction_id = "RUNTIME-MIGRATE-" + uuid.uuid4().hex
            base = {
                **deepcopy(dict(state_values)),
                "schema_version": RUNTIME_SCHEMA,
                "goal_id": GOAL_ID,
                "transition_seq": 1,
                "previous_state_sha256": None,
                "transaction_id": transaction_id,
                "updated_at_ns": time.time_ns(),
            }
            state = seal_state(base)
            event = seal_event(
                {
                    "schema_version": EVENT_SCHEMA,
                    "goal_id": GOAL_ID,
                    "seq": 1,
                    "transaction_id": transaction_id,
                    "action": action,
                    "result": "COMMITTED",
                    "previous_event_sha256": None,
                    "previous_state_sha256": None,
                    "next_state_sha256": state["state_sha256"],
                    "artifact_sha256": dict(sorted(artifacts.items())),
                }
            )
            self._install_transaction(lease, state=state, event=event, previous_state_bytes=None, previous_event_bytes=None)
            return deepcopy(state)

    def transition(
        self,
        action: str,
        mutate: Callable[[dict[str, Any]], None],
        *,
        artifacts: Mapping[str, str] | None = None,
        require_next_action: bool = True,
    ) -> dict[str, Any]:
        with self.lease(action) as lease:
            state, rows, state_bytes, event_bytes = self._load_locked(lease, recover=True)
            if require_next_action and state.get("next_action") != action:
                raise RuntimeControlError(
                    f"requested {action}, unique next action is {state.get('next_action')}"
                )
            next_state = deepcopy(state)
            next_state.pop("state_sha256", None)
            mutate(next_state)
            workflow = next_state.get("workflow_control")
            if isinstance(workflow, dict):
                workflow["next_action"] = next_state.get("next_action")
            transaction_id = f"RUNTIME-{action}-{uuid.uuid4().hex}"
            next_state.update(
                {
                    "transition_seq": int(state["transition_seq"]) + 1,
                    "previous_state_sha256": state["state_sha256"],
                    "transaction_id": transaction_id,
                    "updated_at_ns": time.time_ns(),
                }
            )
            next_state = seal_state(next_state)
            event = seal_event(
                {
                    "schema_version": EVENT_SCHEMA,
                    "goal_id": GOAL_ID,
                    "seq": len(rows) + 1,
                    "transaction_id": transaction_id,
                    "action": action,
                    "result": "COMMITTED",
                    "previous_event_sha256": rows[-1]["event_sha256"] if rows else None,
                    "previous_state_sha256": state["state_sha256"],
                    "next_state_sha256": next_state["state_sha256"],
                    "artifact_sha256": dict(sorted((artifacts or {}).items())),
                }
            )
            self._install_transaction(
                lease,
                state=next_state,
                event=event,
                previous_state_bytes=state_bytes,
                previous_event_bytes=event_bytes,
            )
            return deepcopy(next_state)

    def commit_stage(
        self,
        *,
        stage_id: str,
        action: str,
        outputs: Mapping[str, bytes],
        input_hashes: Mapping[str, str],
        safety_counts: Mapping[str, int],
        mutate: Callable[[dict[str, Any]], None],
    ) -> dict[str, Any]:
        """Install one immutable stage postimage and its sole receipt before CAS.

        A retry may only re-read byte-identical files made by the same pre-state;
        it cannot create a second scientific result or a second receipt.
        """

        if (
            not stage_id
            or Path(stage_id).name != stage_id
            or any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for character in stage_id)
        ):
            raise RuntimeControlError("unsafe runtime stage identity")
        if not outputs:
            raise RuntimeControlError("runtime stage has no outputs")
        normalized_outputs: dict[str, bytes] = {}
        for relative, payload in sorted(outputs.items()):
            path = Path(relative)
            if (
                not isinstance(relative, str)
                or path.is_absolute()
                or ".." in path.parts
                or not path.parts
                or not isinstance(payload, bytes)
            ):
                raise RuntimeControlError("unsafe runtime stage output")
            normalized_outputs[relative] = payload
        for mapping, label in ((input_hashes, "input"),):
            if any(
                not isinstance(name, str)
                or not name
                or not isinstance(digest, str)
                or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
                for name, digest in mapping.items()
            ):
                raise RuntimeControlError(f"runtime stage {label} hash is invalid")
        if any(
            not isinstance(name, str)
            or not name
            or type(count) is not int
            or count < 0
            for name, count in safety_counts.items()
        ):
            raise RuntimeControlError("runtime stage safety count is invalid")

        with self.lease(action) as lease:
            state, rows, state_bytes, event_bytes = self._load_locked(lease, recover=True)
            existing_pointer = state.get("stage_receipts", {}).get(stage_id)
            if existing_pointer is not None:
                receipt, _receipt_bytes = _read_json(Path(existing_pointer["path"]))
                if receipt.get("receipt_sha256") != existing_pointer.get("receipt_sha256"):
                    raise RuntimeControlError("runtime stage receipt pointer drift")
                return deepcopy(state)
            if state.get("next_action") != action:
                raise RuntimeControlError(
                    f"requested {action}, unique next action is {state.get('next_action')}"
                )

            stage_root = self.root / "artifacts" / stage_id
            output_hashes: dict[str, str] = {}
            for relative, payload in normalized_outputs.items():
                target = stage_root / relative
                self._write_immutable_or_verify(target, payload, lease=lease)
                output_hashes[relative] = bytes_sha256(payload)
            receipt_base = {
                "schema_version": STAGE_RECEIPT_SCHEMA,
                "status": "FSYNCED_STAGE_POSTIMAGE_READY_FOR_CAS",
                "goal_id": GOAL_ID,
                "attempt_id": state["attempt_id"],
                "stage_id": stage_id,
                "action": action,
                "pre_state_sha256": state["state_sha256"],
                "pre_event_sha256": rows[-1]["event_sha256"] if rows else None,
                "input_hashes": dict(sorted(input_hashes.items())),
                "output_hashes": output_hashes,
                "safety_counts": dict(sorted(safety_counts.items())),
                "prepared_at_ns": time.time_ns(),
            }
            stage_transaction_id = (
                f"RUNTIME-{action}-"
                f"{semantic_sha256(receipt_base)[:32]}"
            )
            receipt = {
                **receipt_base,
                "stage_transaction_id": stage_transaction_id,
                "receipt_sha256": semantic_sha256(receipt_base),
            }
            receipt["receipt_sha256"] = semantic_sha256(
                receipt, ("receipt_sha256",)
            )
            receipt_path = stage_root / "STAGE_RECEIPT.json"
            self._write_immutable_or_verify(
                receipt_path,
                _canonical_file(receipt),
                lease=lease,
            )
            return self._commit_prepared_stage_locked(
                lease=lease,
                state=state,
                rows=rows,
                state_bytes=state_bytes,
                event_bytes=event_bytes,
                receipt=receipt,
                receipt_path=receipt_path,
                mutate=mutate,
            )

    def recover_prepared_stage(
        self,
        *,
        stage_id: str,
        action: str,
        expected_outputs: Sequence[str],
        mutate: Callable[[dict[str, Any]], None],
    ) -> dict[str, Any] | None:
        """Commit a fully prepared stage without rebuilding scientific data.

        The sole stage receipt is the durable preparation boundary.  A stage
        directory containing files but no receipt is deliberately not replayed
        or silently overwritten: its source-read outcome is indeterminate.
        """

        normalized_expected = tuple(expected_outputs)
        expected = set(normalized_expected)
        if not expected or len(expected) != len(normalized_expected):
            raise RuntimeControlError("prepared-stage output contract is invalid")
        with self.lease(action) as lease:
            state, rows, state_bytes, event_bytes = self._load_locked(
                lease, recover=True
            )
            existing_pointer = state.get("stage_receipts", {}).get(stage_id)
            if existing_pointer is not None:
                return deepcopy(state)
            if state.get("next_action") != action:
                return None
            stage_root = self.root / "artifacts" / stage_id
            receipt_path = stage_root / "STAGE_RECEIPT.json"
            if not receipt_path.exists():
                if stage_root.exists() and any(
                    path.is_file() for path in stage_root.rglob("*")
                ):
                    raise RuntimeControlError(
                        f"unsealed {stage_id} candidate exists; replay is forbidden"
                    )
                return None
            receipt, _receipt_bytes = _read_json(receipt_path)
            self._validate_prepared_stage_receipt(
                receipt,
                stage_id=stage_id,
                action=action,
                state=state,
                rows=rows,
                expected_outputs=expected,
                stage_root=stage_root,
            )
            return self._commit_prepared_stage_locked(
                lease=lease,
                state=state,
                rows=rows,
                state_bytes=state_bytes,
                event_bytes=event_bytes,
                receipt=receipt,
                receipt_path=receipt_path,
                mutate=mutate,
            )

    @staticmethod
    def _validate_prepared_stage_receipt(
        receipt: Mapping[str, Any],
        *,
        stage_id: str,
        action: str,
        state: Mapping[str, Any],
        rows: Sequence[Mapping[str, Any]],
        expected_outputs: set[str],
        stage_root: Path,
    ) -> None:
        base = dict(receipt)
        receipt_sha256 = base.pop("receipt_sha256", None)
        transaction_id = base.pop("stage_transaction_id", None)
        seed = dict(base)
        if (
            receipt.get("schema_version") != STAGE_RECEIPT_SCHEMA
            or receipt.get("status") != "FSYNCED_STAGE_POSTIMAGE_READY_FOR_CAS"
            or receipt.get("goal_id") != GOAL_ID
            or receipt.get("attempt_id") != state.get("attempt_id")
            or receipt.get("stage_id") != stage_id
            or receipt.get("action") != action
            or receipt.get("pre_state_sha256") != state.get("state_sha256")
            or receipt.get("pre_event_sha256")
            != (rows[-1].get("event_sha256") if rows else None)
            or type(receipt.get("prepared_at_ns")) is not int
            or receipt["prepared_at_ns"] <= 0
            or receipt_sha256
            != semantic_sha256(receipt, ("receipt_sha256",))
            or transaction_id
            != f"RUNTIME-{action}-{semantic_sha256(seed)[:32]}"
            or set(receipt.get("output_hashes", {})) != expected_outputs
        ):
            raise RuntimeControlError("prepared stage receipt binding drift")
        observed_paths = {
            str(path.relative_to(stage_root))
            for path in stage_root.rglob("*")
            if path.is_file() and path != stage_root / "STAGE_RECEIPT.json"
        }
        if observed_paths != expected_outputs:
            raise RuntimeControlError("prepared stage output set drift")
        for relative, expected_sha256 in receipt["output_hashes"].items():
            data, _record = read_regular_bytes(
                stage_root / relative,
                max_bytes=2 * 1024 * 1024 * 1024,
                require_nlink_one=True,
            )
            if bytes_sha256(data) != expected_sha256:
                raise RuntimeControlError("prepared stage output hash drift")

    def _commit_prepared_stage_locked(
        self,
        *,
        lease: WriterLease,
        state: Mapping[str, Any],
        rows: list[dict[str, Any]],
        state_bytes: bytes,
        event_bytes: bytes,
        receipt: Mapping[str, Any],
        receipt_path: Path,
        mutate: Callable[[dict[str, Any]], None],
    ) -> dict[str, Any]:
        next_state = deepcopy(dict(state))
        next_state.pop("state_sha256", None)
        mutate(next_state)
        registry = next_state.setdefault("stage_receipts", {})
        stage_id = str(receipt["stage_id"])
        if not isinstance(registry, dict) or stage_id in registry:
            raise RuntimeControlError("runtime stage receipt registry collision")
        registry[stage_id] = {
            "path": str(receipt_path),
            "receipt_sha256": receipt["receipt_sha256"],
        }
        workflow = next_state.get("workflow_control")
        if isinstance(workflow, dict):
            workflow["next_action"] = next_state.get("next_action")
        transaction_id = str(receipt["stage_transaction_id"])
        next_state.update(
            {
                "transition_seq": int(state["transition_seq"]) + 1,
                "previous_state_sha256": state["state_sha256"],
                "transaction_id": transaction_id,
                "updated_at_ns": receipt["prepared_at_ns"],
            }
        )
        next_state = seal_state(next_state)
        artifacts = {
            **dict(receipt["output_hashes"]),
            "STAGE_RECEIPT.json": receipt["receipt_sha256"],
        }
        event = seal_event(
            {
                "schema_version": EVENT_SCHEMA,
                "goal_id": GOAL_ID,
                "seq": len(rows) + 1,
                "transaction_id": transaction_id,
                "action": receipt["action"],
                "result": "COMMITTED",
                "previous_event_sha256": rows[-1]["event_sha256"] if rows else None,
                "previous_state_sha256": state["state_sha256"],
                "next_state_sha256": next_state["state_sha256"],
                "artifact_sha256": dict(sorted(artifacts.items())),
            }
        )
        self._install_transaction(
            lease,
            state=next_state,
            event=event,
            previous_state_bytes=state_bytes,
            previous_event_bytes=event_bytes,
        )
        return deepcopy(next_state)

    @staticmethod
    def _write_immutable_or_verify(
        path: Path,
        payload: bytes,
        *,
        lease: WriterLease,
    ) -> None:
        if path.exists():
            observed, _record = read_regular_bytes(
                path,
                max_bytes=max(1, len(payload)),
                require_nlink_one=True,
            )
            if observed != payload:
                raise RuntimeControlError(f"immutable runtime artifact drift: {path}")
            return
        atomic_write_immutable(path, payload, lease=lease)

    def _install_transaction(
        self,
        lease: WriterLease,
        *,
        state: Mapping[str, Any],
        event: Mapping[str, Any],
        previous_state_bytes: bytes | None,
        previous_event_bytes: bytes | None,
    ) -> None:
        transaction_base = {
            "schema_version": TRANSACTION_SCHEMA,
            "transaction_id": event["transaction_id"],
            "event": dict(event),
            "next_state": dict(state),
        }
        transaction = {
            **transaction_base,
            "transaction_sha256": semantic_sha256(transaction_base),
        }
        transaction_root = self.root / "transactions" / str(event["transaction_id"])
        self._write_immutable_or_verify(
            transaction_root / "record.json", _canonical_file(transaction), lease=lease
        )
        self._write_immutable_or_verify(
            self.root / "records" / "states" / f"{int(state['transition_seq']):06d}_{state['state_sha256']}.json",
            _canonical_file(state),
            lease=lease,
        )
        self._write_immutable_or_verify(
            self.root / "records" / "events" / f"{int(event['seq']):06d}_{event['event_sha256']}.jsonl",
            _canonical_file(event),
            lease=lease,
        )
        next_event_bytes = (previous_event_bytes or b"") + _canonical_file(event)
        atomic_write_bytes(
            self.events_path,
            next_event_bytes,
            lease=lease,
            precondition="MUST_BE_ABSENT" if previous_event_bytes is None else None,
            expected_preimage_sha256=(
                None if previous_event_bytes is None else bytes_sha256(previous_event_bytes)
            ),
        )
        atomic_write_json(
            self.state_path,
            state,
            lease=lease,
            precondition="MUST_BE_ABSENT" if previous_state_bytes is None else None,
            expected_preimage_sha256=(
                None if previous_state_bytes is None else bytes_sha256(previous_state_bytes)
            ),
        )


def runtime_artifact_path(eval_id: str, name: str, *, root: Path = RUNTIME_ROOT) -> Path:
    if not eval_id or any(char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-" for char in eval_id):
        raise RuntimeControlError("unsafe eval artifact identity")
    if not name or Path(name).name != name:
        raise RuntimeControlError("unsafe eval artifact name")
    return root / "evals" / eval_id / name


def read_runtime_artifact(path: Path) -> dict[str, Any]:
    value, _encoded = _read_json(path)
    return value
