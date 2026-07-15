"""Minimal F0/F1 controller used by the scientific forward engine.

The public surface is intentionally limited to generation issuance, RUNNING
transition, forward journaling callbacks, completion, and verification.  The
legacy A4 controller is not imported here.
"""

from __future__ import annotations

import json
import re
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

from .atomic_io import BootstrapLease, atomic_write_immutable, ensure_directory
from .canonical import bytes_sha256, canonical_json_bytes, read_regular_bytes, semantic_sha256
from .constants import CONTROL_ROOT, GOAL_ID
from .runtime_store import (
    RUNTIME_ROOT,
    RuntimeControlError,
    RuntimeStore,
    read_runtime_artifact,
    runtime_artifact_path,
)


MARK_F0_A_ACTION = "MARK_F0_A_RUNNING"
RUN_F0_A_ACTION = "RUN_F0_A"
RUN_F0_B_ACTION = "RUN_F0_B"
ABORT_ZERO_FORWARD_F0_A_ACTION = "ABORT_ZERO_FORWARD_F0_A"
ISSUE_F0_A_ACTION = "ISSUE_F0_A_GENERATION"
F0_A_ANALYZE_ACTION = "F0_A_ANALYZE"
F0_B_ANALYZE_ACTION = "F0_B_ANALYZE"
HEX64 = re.compile(r"[0-9a-f]{64}")


class A4ControlError(RuntimeControlError):
    """Compatibility name consumed only by the forward boundary."""


class A4EvalCursorAbsentError(A4ControlError):
    pass


_ISSUER = object()


class _OpaqueCapability:
    __slots__ = ("_payload", "_sha256", "_runtime_root", "_frozen")

    def __init__(
        self,
        issuer: object,
        payload: Mapping[str, Any],
        *,
        runtime_root: Path | None = None,
    ) -> None:
        if issuer is not _ISSUER:
            raise A4ControlError("controller-issued capability required")
        copied = json.loads(canonical_json_bytes(dict(payload)).decode("utf-8", "strict"))
        object.__setattr__(self, "_payload", copied)
        object.__setattr__(self, "_sha256", semantic_sha256(copied))
        object.__setattr__(self, "_runtime_root", runtime_root)
        object.__setattr__(self, "_frozen", True)

    def __setattr__(self, name: str, value: Any) -> None:
        if getattr(self, "_frozen", False):
            raise A4ControlError("capability is immutable")
        object.__setattr__(self, name, value)

    @property
    def capability_sha256(self) -> str:
        return str(self._sha256)

    def binding(self) -> dict[str, Any]:
        return json.loads(canonical_json_bytes(self._payload).decode("utf-8", "strict"))

    @property
    def runtime_root(self) -> Path | None:
        return self._runtime_root


class A4EvalRunningCapability(_OpaqueCapability):
    __slots__ = ()


class A4EvalCursorCapability(_OpaqueCapability):
    __slots__ = ()


class A4LmdbBatchControlReceipt(_OpaqueCapability):
    __slots__ = ()


def _validate_capability(value: Any, expected_type: type[_OpaqueCapability]) -> dict[str, Any]:
    if type(value) is not expected_type:
        raise A4ControlError(f"exact {expected_type.__name__} required")
    payload = value.binding()
    if value.capability_sha256 != semantic_sha256(payload):
        raise A4ControlError("capability hash mismatch")
    return payload


def validate_eval_running_capability(value: Any) -> dict[str, Any]:
    payload = _validate_capability(value, A4EvalRunningCapability)
    exact = {
        "schema_version",
        "goal_id",
        "attempt_id",
        "run_id",
        "token_id",
        "eval_id",
        "transaction_id",
        "running_state_sha256",
        "running_event_sha256",
        "eval_registry_sha256",
        "execution_binding_sha256",
        "reservation_input_capability_sha256",
    }
    if set(payload) != exact or payload.get("schema_version") != "c28f_a4_eval_running_capability_v1":
        raise A4ControlError("RUNNING capability fields mismatch")
    return payload


def validate_eval_cursor_capability(value: Any) -> dict[str, Any]:
    payload = _validate_capability(value, A4EvalCursorCapability)
    exact = {
        "schema_version",
        "running_anchor",
        "cursor_index",
        "cursor_receipt_sha256",
        "cursor_sha256",
        "chunks_sha256",
        "artifacts_sha256",
    }
    if set(payload) != exact or payload.get("schema_version") != "c28f_a4_eval_cursor_capability_v1":
        raise A4ControlError("cursor capability fields mismatch")
    return payload


def validate_lmdb_batch_control_receipt(value: Any) -> dict[str, Any]:
    payload = _validate_capability(value, A4LmdbBatchControlReceipt)
    exact = {
        "schema_version",
        "running_anchor",
        "source_id",
        "source_identity_sha256",
        "key_manifest_sha256",
        "key_set_sha256",
        "key_count",
        "source_manifest_sha256",
        "batch_sha256",
        "shared_ledger_tail_sha256",
        "persisted_receipt_sha256",
    }
    if set(payload) != exact or payload.get("schema_version") != "c28f_a4_lmdb_batch_control_receipt_v1":
        raise A4ControlError("LMDB receipt fields mismatch")
    if payload["persisted_receipt_sha256"] != semantic_sha256(
        payload, ("persisted_receipt_sha256",)
    ):
        raise A4ControlError("LMDB receipt self-hash mismatch")
    return payload


def _read_json(path: Path, *, max_bytes: int = 1024 * 1024 * 1024) -> dict[str, Any]:
    data, _record = read_regular_bytes(path, max_bytes=max_bytes, require_nlink_one=True)
    value = json.loads(data.decode("utf-8", "strict"))
    if not isinstance(value, dict) or data != canonical_json_bytes(value) + b"\n":
        raise A4ControlError(f"non-canonical runtime input: {path}")
    return value


def _read_forward_json(
    path: Path, *, max_bytes: int = 1024 * 1024 * 1024
) -> tuple[dict[str, Any], str]:
    """Read an immutable forward artifact, whose writer omits the final LF."""

    data, _record = read_regular_bytes(path, max_bytes=max_bytes, require_nlink_one=True)
    value = json.loads(data.decode("utf-8", "strict"))
    canonical = canonical_json_bytes(value) if isinstance(value, dict) else b""
    if not isinstance(value, dict) or data not in {canonical, canonical + b"\n"}:
        raise A4ControlError(f"non-canonical forward artifact: {path}")
    return value, bytes_sha256(data)


def _validate_authority_record(record: Mapping[str, Any]) -> dict[str, Any]:
    value = dict(record)
    projection = value.get("active_rehydrate_projection")
    binding = value.get("controller_binding")
    if (
        value.get("schema_version") != "c28f_v5_runtime_authority_record_v1"
        or value.get("record_sha256") != semantic_sha256(value, ("record_sha256",))
        or not isinstance(projection, dict)
        or projection.get("projection_binding_sha256")
        != semantic_sha256(projection, ("projection_binding_sha256",))
        or value.get("active_rehydrate_projection_sha256")
        != projection.get("projection_binding_sha256")
        or not isinstance(binding, dict)
        or binding.get("controller_authority_binding_sha256")
        != semantic_sha256(binding, ("controller_authority_binding_sha256",))
    ):
        raise A4ControlError("runtime authority record is invalid")
    return value


def _code_source_path(source_name: str) -> Path:
    package_root = Path(__file__).resolve(strict=True).parent
    if source_name in {"a4_forward.py", "a4_security.py"}:
        return package_root / source_name
    return package_root.parent / source_name


def _current_code_hashes(manifest: Mapping[str, Any]) -> list[list[str]]:
    source_rows = manifest.get("code_source_sha256")
    if not isinstance(source_rows, list):
        raise A4ControlError("legacy manifest code sources are missing")
    result: list[list[str]] = []
    for row in source_rows:
        if not isinstance(row, list) or len(row) != 2 or not isinstance(row[0], str):
            raise A4ControlError("legacy manifest code source row is invalid")
        path = _code_source_path(row[0])
        data, _record = read_regular_bytes(path, max_bytes=64 * 1024 * 1024, require_nlink_one=True)
        result.append([row[0], bytes_sha256(data)])
    result.sort(key=lambda item: (item[0], item[1]))
    return result


def _runtime_code_hashes() -> dict[str, str]:
    root = Path(__file__).resolve(strict=True).parent
    result: dict[str, str] = {}
    for name in (
        "runtime_control.py",
        "runtime_data.py",
        "runtime_runner.py",
        "runtime_stages.py",
        "runtime_store.py",
        "runtime_workflow.py",
    ):
        data, _record = read_regular_bytes(
            root / name,
            max_bytes=8 * 1024 * 1024,
            require_nlink_one=True,
        )
        result[name] = bytes_sha256(data)
    return result


def _manifest_from_projection(projection: Mapping[str, Any]) -> tuple[Any, dict[str, Any]]:
    from . import a4_forward

    legacy_manifest = projection.get("manifest")
    material = projection.get("authority_binding_material")
    if not isinstance(legacy_manifest, Mapping) or not isinstance(material, Mapping):
        raise A4ControlError("legacy descriptor projection is incomplete")
    manifest_view = deepcopy(dict(legacy_manifest))
    manifest_view["code_source_sha256"] = _current_code_hashes(manifest_view)
    manifest_view["declared_semantic_sha256"] = semantic_sha256(
        manifest_view, ("declared_semantic_sha256",)
    )
    base = {
        "schema_version": "c28f_a4_completed_forward_authority_rehydrate_projection_v1",
        "status": "COMPLETED_RESULT_VERIFICATION_ONLY_PROJECTION",
        "manifest": manifest_view,
        "query_keys": list(projection["query_keys"]),
        "corpus_video_ids": list(projection["corpus_video_ids"]),
        "corpus_durations_sec": list(projection["corpus_durations_sec"]),
        "output_root": material["output_root"],
        "authority_binding_sha256": "0" * 64,
        "forward_authority_handle_sha256": "0" * 64,
        "forward_manifest_sha256": manifest_view["declared_semantic_sha256"],
        "input_identity_sha256": projection["input_identity_sha256"],
        "encoded_cache_fingerprint": "0" * 64,
        "active_authority_type_reconstructed": False,
        "checkpoint_descriptor_present": False,
        "lmdb_descriptor_present": False,
        "cache_root_descriptor_present": False,
        "active_io_allowed": False,
    }
    completed_projection = {
        **base,
        "projection_binding_sha256": semantic_sha256(base),
    }
    verification_authority = a4_forward._rehydrate_completed_forward_authority_from_projection(
        completed_projection
    )
    return verification_authority.manifest, dict(material)


def _build_current_authority(legacy_record: Mapping[str, Any], generation: int) -> tuple[Any, dict[str, Any], str]:
    from . import a4_forward

    old_projection = legacy_record.get("active_rehydrate_projection")
    if not isinstance(old_projection, Mapping):
        raise A4ControlError("legacy authority projection is absent")
    manifest, material = _manifest_from_projection(old_projection)
    sources = []
    for source in material["lmdb_sources"]:
        sources.append(
            a4_forward.FrozenLmdbSource(
                source_id=source["source_id"],
                namespace=source["namespace"],
                directory=a4_forward.FrozenDirectory(**dict(source["directory"])),
                data_file=a4_forward.FrozenFile(**dict(source["data_file"])),
            )
        )
    authority = a4_forward.FrozenInputAuthority(
        manifest=manifest,
        checkpoint=a4_forward.FrozenFile(**dict(material["checkpoint"])),
        lmdb_sources=tuple(sources),
        output_root=a4_forward.FrozenDirectory(**dict(material["output_root"])),
        cache_root=a4_forward.FrozenDirectory(**dict(material["cache_root"])),
        forbidden_roots=tuple(material["forbidden_roots"]),
    )
    handle_sha256 = semantic_sha256(
        {
            "schema_version": "c28f_v5_runtime_authority_handle_v1",
            "generation": generation,
            "legacy_record_sha256": legacy_record["record_sha256"],
            "authority_binding_sha256": authority.authority_binding_sha256,
            "forward_manifest_sha256": manifest.manifest_sha256,
        }
    )
    projection = dict(
        a4_forward.build_active_forward_authority_rehydrate_projection(
            authority,
            forward_authority_handle_sha256=handle_sha256,
        )
    )
    return authority, projection, handle_sha256


def _controller_binding(
    authority: Any,
    *,
    handle_sha256: str,
    reservation_sha256: str,
    legacy_payload: Mapping[str, Any],
) -> dict[str, Any]:
    manifest = authority.manifest
    base = {
        "schema_version": "c28f_a4_controller_forward_authority_binding_v1",
        "status": "CONTROLLER_BUILT_REGISTERED_AUTHORITY",
        "goal_id": manifest.goal_id,
        "attempt_id": manifest.attempt_id,
        "authority_id": manifest.authority_id,
        "stage": manifest.stage,
        "purpose": manifest.purpose,
        "authority_handle_sha256": handle_sha256,
        "reservation_input_capability_sha256": reservation_sha256,
        "authority_binding_sha256": authority.authority_binding_sha256,
        "forward_manifest_sha256": manifest.manifest_sha256,
        "input_identity_sha256": manifest.input_identity_sha256,
        "query_identity_sha256": manifest.query_identity.identity_sha256,
        "query_keys_sha256": manifest.query_identity.query_keys_sha256,
        "query_types_sha256": manifest.query_identity.query_types_sha256,
        "query_rows_sha256": manifest.query_identity.query_rows_sha256,
        "split_manifest_sha256": manifest.query_identity.split_manifest_sha256,
        "corpus_identity_sha256": manifest.corpus_identity.identity_sha256,
        "video_ids_sha256": manifest.corpus_identity.video_ids_sha256,
        "durations_sha256": manifest.corpus_identity.durations_sha256,
        "query_source_descriptor_identity_sha256": legacy_payload[
            "query_source_descriptor_identity_sha256"
        ],
        "query_source_selected_projection_sha256": legacy_payload[
            "query_source_selected_projection_sha256"
        ],
        "query_source_index_binding_sha256": legacy_payload[
            "query_source_index_binding_sha256"
        ],
        "query_source_index_bootstrap_receipt_sha256": legacy_payload[
            "query_source_index_bootstrap_receipt_sha256"
        ],
        "query_source_file_sha256": legacy_payload["query_source_file_sha256"],
        "query_identity_source_access_receipt_sha256": legacy_payload[
            "query_identity_source_access_receipt_sha256"
        ],
        "video_meta_source_file_sha256": legacy_payload[
            "video_meta_source_file_sha256"
        ],
    }
    return {
        **base,
        "controller_authority_binding_sha256": semantic_sha256(base),
    }


def _control_chain_values(authority: Any) -> dict[str, Any]:
    manifest = authority.manifest
    return {
        "authority_binding_sha256": authority.authority_binding_sha256,
        "forward_manifest_sha256": manifest.manifest_sha256,
        "input_identity_sha256": manifest.input_identity_sha256,
        "checkpoint_sha256": manifest.checkpoint_sha256,
        "split_manifest_sha256": manifest.query_identity.split_manifest_sha256,
        "corpus_manifest_sha256": manifest.corpus_identity.identity_sha256,
        "evaluator_sha256": manifest.evaluator_schema_sha256,
        "goal_id": manifest.goal_id,
        "authority_id": manifest.authority_id,
        "stage": manifest.stage,
        "purpose": manifest.purpose,
        "attempt_id": manifest.attempt_id,
    }


def begin_f0_b_running(
    *,
    route_manifest_sha256: str,
    route_desc_ids_sha256: str,
    route_query_count: int,
    input_hashes: Mapping[str, str],
    store: RuntimeStore | None = None,
) -> dict[str, Any]:
    """Commit the unique F0-B identity before any query/feature content open."""

    runtime = store or RuntimeStore()
    state = runtime.read_state()
    normalized_input_hashes = dict(sorted(input_hashes.items()))
    if (
        type(route_query_count) is not int
        or route_query_count < 2_048
        or any(
            not isinstance(name, str)
            or not name
            or not isinstance(digest, str)
            or HEX64.fullmatch(digest) is None
            for name, digest in normalized_input_hashes.items()
        )
        or any(
            not isinstance(digest, str) or HEX64.fullmatch(digest) is None
            for digest in (route_manifest_sha256, route_desc_ids_sha256)
        )
    ):
        raise A4ControlError("F0-B input binding is invalid")
    identity = semantic_sha256(
        {
            "schema_version": "c28f_v5_runtime_f0_b_identity_v1",
            "goal_id": GOAL_ID,
            "attempt_id": state["attempt_id"],
            "route_manifest_sha256": route_manifest_sha256,
            "route_desc_ids_sha256": route_desc_ids_sha256,
            "route_query_count": route_query_count,
            "input_hashes": normalized_input_hashes,
        }
    )
    eval_id = "F0B-EVAL-" + bytes_sha256(("eval:" + identity).encode("ascii"))[:32]
    run_id = "F0B-RUNTIME-" + bytes_sha256(("run:" + identity).encode("ascii"))[:32]
    input_binding = {
        "schema_version": "c28f_v5_runtime_f0_b_input_binding_v1",
        "goal_id": GOAL_ID,
        "generation": 1,
        "token_id": "F0_B",
        "eval_id": eval_id,
        "run_id": run_id,
        "route_manifest_sha256": route_manifest_sha256,
        "route_desc_ids_sha256": route_desc_ids_sha256,
        "route_query_count": route_query_count,
        "input_hashes": normalized_input_hashes,
    }
    input_binding_sha256 = semantic_sha256(input_binding)
    token = state["evals"]["F0_B"]
    current = token["current"]
    if isinstance(current, dict):
        if (
            state.get("status") != "F0_B_RUNNING"
            or state.get("next_action") != RUN_F0_B_ACTION
            or token.get("successful_evaluations") != 0
            or token.get("next_generation") != 2
            or current.get("status") != "RUNNING"
            or current.get("generation") != 1
            or current.get("eval_id") != eval_id
            or current.get("run_id") != run_id
            or current.get("route_manifest_sha256") != route_manifest_sha256
            or current.get("route_desc_ids_sha256") != route_desc_ids_sha256
            or current.get("route_query_count") != route_query_count
            or current.get("input_hashes") != normalized_input_hashes
            or current.get("reservation_input_capability_sha256")
            != input_binding_sha256
            or current.get("execution_binding_sha256") != input_binding_sha256
        ):
            raise A4ControlError("F0-B committed RUNNING input binding drift")
        if current.get("running_anchor") is None:
            if any(
                current.get(name) not in (None, {})
                for name in (
                    "authority_record_path",
                    "authority_record_sha256",
                    "query_projection",
                    "forward_consumption",
                    "cursor",
                    "completed_material",
                )
            ) or current.get("lmdb_receipts") != {}:
                raise A4ControlError("F0-B work exists without a RUNNING anchor")
            event_tail = runtime.verify()["event_tail_sha256"]
            anchor = {
                "schema_version": "c28f_a4_eval_running_capability_v1",
                "goal_id": GOAL_ID,
                "attempt_id": state["attempt_id"],
                "run_id": run_id,
                "token_id": "F0_B",
                "eval_id": eval_id,
                "transaction_id": state["transaction_id"],
                "running_state_sha256": state["state_sha256"],
                "running_event_sha256": event_tail,
                "eval_registry_sha256": semantic_sha256(current),
                "execution_binding_sha256": input_binding_sha256,
                "reservation_input_capability_sha256": input_binding_sha256,
            }

            def recover_anchor(next_state: dict[str, Any]) -> None:
                target = next_state["evals"]["F0_B"]["current"]
                if target.get("running_anchor") not in (None, anchor):
                    raise A4ControlError("F0-B RUNNING anchor drift")
                target["running_anchor"] = anchor

            state = runtime.transition(
                "ATTACH_F0_B_RUNNING_ANCHOR",
                recover_anchor,
                artifacts={"running_anchor": semantic_sha256(anchor)},
                require_next_action=False,
            )
        return {
            "status": "RUNNING",
            "eval_id": eval_id,
            "run_id": run_id,
            "state_sha256": state["state_sha256"],
            "next_action": state["next_action"],
        }
    if (
        state.get("status") != "ROLE_POLICY_LOCKED"
        or state.get("next_action") != RUN_F0_B_ACTION
        or token["successful_evaluations"] != 0
        or token["next_generation"] != 1
    ):
        raise A4ControlError("F0-B is not uniquely issuable")

    def mark(next_state: dict[str, Any]) -> None:
        next_state["evals"]["F0_B"]["current"] = {
            "generation": 1,
            "status": "RUNNING",
            "eval_id": eval_id,
            "run_id": run_id,
            "input_hashes": normalized_input_hashes,
            "route_manifest_sha256": route_manifest_sha256,
            "route_desc_ids_sha256": route_desc_ids_sha256,
            "route_query_count": route_query_count,
            "authority_record_path": None,
            "authority_record_sha256": None,
            # These two names are fixed by the inherited forward callback
            # contract.  Their value is the ordinary F0-B input binding above;
            # this recovery path does not create a reservation capability.
            "reservation_input_capability_sha256": input_binding_sha256,
            "execution_binding_sha256": input_binding_sha256,
            "running_anchor": None,
            "query_projection": None,
            "forward_consumption": None,
            "lmdb_receipts": {},
            "cursor": None,
            "completed_material": None,
        }
        next_state["evals"]["F0_B"]["next_generation"] = 2
        next_state["status"] = "F0_B_RUNNING"
        next_state["next_action"] = RUN_F0_B_ACTION

    marked = runtime.transition(
        RUN_F0_B_ACTION,
        mark,
        artifacts={
            "route_manifest": route_manifest_sha256,
            "f0_b_input_binding": input_binding_sha256,
        },
    )
    event_tail = runtime.verify()["event_tail_sha256"]
    marked_current = marked["evals"]["F0_B"]["current"]
    anchor = {
        "schema_version": "c28f_a4_eval_running_capability_v1",
        "goal_id": GOAL_ID,
        "attempt_id": marked["attempt_id"],
        "run_id": run_id,
        "token_id": "F0_B",
        "eval_id": eval_id,
        "transaction_id": marked["transaction_id"],
        "running_state_sha256": marked["state_sha256"],
        "running_event_sha256": event_tail,
        "eval_registry_sha256": semantic_sha256(marked_current),
        "execution_binding_sha256": input_binding_sha256,
        "reservation_input_capability_sha256": input_binding_sha256,
    }

    def attach_anchor(next_state: dict[str, Any]) -> None:
        target = next_state["evals"]["F0_B"]["current"]
        if target.get("running_anchor") not in (None, anchor):
            raise A4ControlError("F0-B RUNNING anchor drift")
        target["running_anchor"] = anchor

    attached = runtime.transition(
        "ATTACH_F0_B_RUNNING_ANCHOR",
        attach_anchor,
        artifacts={"running_anchor": semantic_sha256(anchor)},
        require_next_action=False,
    )
    return {
        "status": "RUNNING",
        "eval_id": eval_id,
        "run_id": run_id,
        "state_sha256": attached["state_sha256"],
        "next_action": attached["next_action"],
    }


def attach_f0_b_authority(
    query_projection: Mapping[str, Any] | None,
    *,
    store: RuntimeStore | None = None,
) -> dict[str, Any]:
    """Build and persist the F0-B authority from one indexed query projection."""

    from . import a4_forward

    runtime = store or RuntimeStore()
    state = runtime.read_state()
    current = state["evals"]["F0_B"]["current"]
    if not isinstance(current, dict) or current.get("status") != "RUNNING":
        raise A4ControlError("F0-B RUNNING generation is absent")
    if current.get("authority_record_path") is not None:
        record = _validate_authority_record(
            _read_json(Path(current["authority_record_path"]))
        )
        if record["record_sha256"] != current["authority_record_sha256"]:
            raise A4ControlError("persisted F0-B authority pointer drift")
        return {
            "status": "AUTHORITY_ATTACHED",
            "authority_record_sha256": record["record_sha256"],
            "state_sha256": state["state_sha256"],
        }
    authority_path = runtime_artifact_path(
        current["eval_id"], "authority.json", root=runtime.root
    )
    if authority_path.exists():
        record = _validate_authority_record(_read_json(authority_path))
        stored_projection = record.get("query_source_projection")
        stored_records = (
            stored_projection.get("records")
            if isinstance(stored_projection, dict)
            else None
        )
        stored_desc_ids = (
            [row.get("desc_id") for row in stored_records]
            if isinstance(stored_records, list)
            and all(isinstance(row, dict) for row in stored_records)
            else None
        )
        stored_query_types = (
            [row.get("type") for row in stored_records]
            if isinstance(stored_records, list)
            and all(isinstance(row, dict) for row in stored_records)
            else None
        )
        stored_manifest = record["active_rehydrate_projection"].get(
            "manifest"
        )
        manifest_query = (
            stored_manifest.get("query_identity")
            if isinstance(stored_manifest, dict)
            else None
        )
        f0_a_current = state["evals"]["F0_A"]["current"]
        if (
            record.get("generation") != current["generation"]
            or record.get("eval_id") != current["eval_id"]
            or record.get("run_id") != current["run_id"]
            or record.get("reservation_input_capability_sha256")
            != current["reservation_input_capability_sha256"]
            or not isinstance(stored_projection, dict)
            or stored_projection.get("record_sha256")
            != semantic_sha256(stored_projection, ("record_sha256",))
            or stored_projection.get("purpose") != "FORWARD_QUERY_IDENTITY"
            or stored_projection.get("projection_scope")
            != "F0_B_FORWARD_QUERY_IDENTITY"
            or not isinstance(stored_records, list)
            or len(stored_records) != current["route_query_count"]
            or stored_projection.get("records_sha256")
            != bytes_sha256(canonical_json_bytes(stored_records))
            or not isinstance(stored_desc_ids, list)
            or any(type(value) is not int or value < 0 for value in stored_desc_ids)
            or stored_desc_ids != sorted(set(stored_desc_ids))
            or not isinstance(stored_query_types, list)
            or any(
                value not in {"v", "t", "vt", "unknown"}
                for value in stored_query_types
            )
            or bytes_sha256(canonical_json_bytes(stored_desc_ids))
            != current["route_desc_ids_sha256"]
            or not isinstance(manifest_query, dict)
            or manifest_query.get("split_manifest_sha256")
            != current["route_manifest_sha256"]
            or not isinstance(f0_a_current, dict)
            or f0_a_current.get("status") != "COMPLETED"
            or record.get("legacy_authority_record_sha256")
            != f0_a_current.get("authority_record_sha256")
        ):
            raise A4ControlError("persisted F0-B authority recovery drift")
        pointer = {"record_sha256": stored_projection["record_sha256"]}

        def recover_pointer(next_state: dict[str, Any]) -> None:
            target = next_state["evals"]["F0_B"]["current"]
            expected = (
                str(authority_path),
                record["record_sha256"],
                pointer,
            )
            observed = (
                target.get("authority_record_path"),
                target.get("authority_record_sha256"),
                target.get("query_projection"),
            )
            if observed not in ((None, None, None), expected):
                raise A4ControlError("F0-B authority recovery pointer drift")
            target["authority_record_path"] = expected[0]
            target["authority_record_sha256"] = expected[1]
            target["query_projection"] = expected[2]

        attached = runtime.transition(
            "ATTACH_F0_B_AUTHORITY",
            recover_pointer,
            artifacts={
                "authority_record": record["record_sha256"],
                "query_projection": stored_projection["record_sha256"],
            },
            require_next_action=False,
        )
        return {
            "status": "AUTHORITY_ATTACHED",
            "authority_record_sha256": record["record_sha256"],
            "state_sha256": attached["state_sha256"],
            "recovered_existing_authority": True,
        }
    if not isinstance(query_projection, Mapping):
        raise A4ControlError("F0-B query projection is required")
    records = query_projection.get("records")
    if (
        not isinstance(records, list)
        or len(records) != current["route_query_count"]
        or query_projection.get("purpose") != "FORWARD_QUERY_IDENTITY"
        or query_projection.get("records_sha256")
        != bytes_sha256(canonical_json_bytes(records))
    ):
        raise A4ControlError("F0-B query projection is invalid")
    desc_ids = [record.get("desc_id") for record in records]
    query_types = [record.get("type") for record in records]
    if (
        desc_ids != sorted(set(desc_ids))
        or any(type(value) is not int or value < 0 for value in desc_ids)
        or any(value not in {"v", "t", "vt", "unknown"} for value in query_types)
        or bytes_sha256(canonical_json_bytes(desc_ids))
        != current["route_desc_ids_sha256"]
    ):
        raise A4ControlError("F0-B query projection membership drift")
    f0_a_current = state["evals"]["F0_A"]["current"]
    if not isinstance(f0_a_current, dict) or f0_a_current.get("status") != "COMPLETED":
        raise A4ControlError("F0-B requires the completed F0-A authority lineage")
    f0_a_record = _validate_authority_record(
        _read_json(Path(f0_a_current["authority_record_path"]))
    )
    base_manifest, material = _manifest_from_projection(
        f0_a_record["active_rehydrate_projection"]
    )
    query_keys = tuple(str(value) for value in desc_ids)
    query_type_tuple = tuple(str(value) for value in query_types)
    query_rows = [
        {"query_id": query_id, "query_type": query_type}
        for query_id, query_type in zip(query_keys, query_type_tuple)
    ]
    query_identity = a4_forward.FrozenQueryIdentity(
        role="train_fit_route_dev",
        query_keys=query_keys,
        query_types=query_type_tuple,
        query_keys_sha256=bytes_sha256(
            "".join(value + "\n" for value in query_keys).encode("ascii")
        ),
        query_types_sha256=bytes_sha256(
            "".join(value + "\n" for value in query_type_tuple).encode("ascii")
        ),
        query_rows_sha256=bytes_sha256(canonical_json_bytes(query_rows)),
        split_manifest_sha256=current["route_manifest_sha256"],
    )
    output_path = runtime.root / "artifacts" / "f0_b_forward"
    ensure_directory(output_path)
    output_root = a4_forward.FrozenDirectory.capture(
        "runtime-f0-b-output", str(output_path)
    )
    authority_id = "RUNTIME-F0B-AUTH-" + semantic_sha256(
        {
            "eval_id": current["eval_id"],
            "query_identity_sha256": query_identity.identity_sha256,
            "f0_a_authority_record_sha256": f0_a_record["record_sha256"],
        }
    )[:32]
    manifest = a4_forward.FrozenForwardManifest.build(
        goal_id=base_manifest.goal_id,
        authority_id=authority_id,
        stage="G5_F0_B",
        purpose="F0_B_ROUTE_DEV_REPLICATION",
        query_identity=query_identity,
        corpus_identity=base_manifest.corpus_identity,
        checkpoint_sha256=base_manifest.checkpoint_sha256,
        model_contract=base_manifest.model_contract,
        query_source_id=base_manifest.query_source_id,
        visual_source_id=base_manifest.visual_source_id,
        subtitle_source_id=base_manifest.subtitle_source_id,
        output_root_id=output_root.root_id,
        cache_root_id=base_manifest.cache_root_id,
        attempt_id=base_manifest.attempt_id,
        temporal_manifest=base_manifest.temporal_manifest,
        feature_contract=base_manifest.feature_contract,
        gpu0_identity=base_manifest.gpu0_identity,
        code_source_sha256=base_manifest.code_source_sha256,
        metric_schema_sha256=base_manifest.metric_schema_sha256,
        evaluator_schema_sha256=base_manifest.evaluator_schema_sha256,
        environment_sha256=base_manifest.environment_sha256,
    )
    sources = tuple(
        a4_forward.FrozenLmdbSource(
            source_id=source["source_id"],
            namespace=source["namespace"],
            directory=a4_forward.FrozenDirectory(**dict(source["directory"])),
            data_file=a4_forward.FrozenFile(**dict(source["data_file"])),
        )
        for source in material["lmdb_sources"]
    )
    forbidden_roots = tuple(
        sorted(set(material["forbidden_roots"] + [str(CONTROL_ROOT)]))
    )
    authority = a4_forward.FrozenInputAuthority(
        manifest=manifest,
        checkpoint=a4_forward.FrozenFile(**dict(material["checkpoint"])),
        lmdb_sources=sources,
        output_root=output_root,
        cache_root=a4_forward.FrozenDirectory(**dict(material["cache_root"])),
        forbidden_roots=forbidden_roots,
    )
    handle_sha256 = semantic_sha256(
        {
            "schema_version": "c28f_v5_runtime_authority_handle_v1",
            "generation": current["generation"],
            "f0_a_authority_record_sha256": f0_a_record["record_sha256"],
            "query_projection_sha256": query_projection["record_sha256"],
            "authority_binding_sha256": authority.authority_binding_sha256,
            "forward_manifest_sha256": manifest.manifest_sha256,
        }
    )
    projection = dict(
        a4_forward.build_active_forward_authority_rehydrate_projection(
            authority,
            forward_authority_handle_sha256=handle_sha256,
        )
    )
    f0_a_payload = f0_a_record.get("controller_binding", {})
    binding_source = {
        "query_source_descriptor_identity_sha256": query_projection[
            "source_descriptor_identity_sha256"
        ],
        "query_source_selected_projection_sha256": query_projection[
            "record_sha256"
        ],
        "query_source_index_binding_sha256": query_projection[
            "index_binding_sha256"
        ],
        "query_source_index_bootstrap_receipt_sha256": query_projection[
            "index_receipt_sha256"
        ],
        "query_source_file_sha256": query_projection["source_file_sha256"],
        "query_identity_source_access_receipt_sha256": query_projection[
            "access_receipt_sha256"
        ],
        "video_meta_source_file_sha256": f0_a_payload[
            "video_meta_source_file_sha256"
        ],
    }
    binding = _controller_binding(
        authority,
        handle_sha256=handle_sha256,
        reservation_sha256=current["reservation_input_capability_sha256"],
        legacy_payload=binding_source,
    )
    control_values = _control_chain_values(authority)
    authority_base = {
        "schema_version": "c28f_v5_runtime_authority_record_v1",
        "generation": current["generation"],
        "eval_id": current["eval_id"],
        "run_id": current["run_id"],
        "reservation_input": {
            "schema_version": "c28f_v5_runtime_f0_b_input_binding_reference_v1",
            "reservation_input_capability_sha256": current[
                "reservation_input_capability_sha256"
            ],
            "query_projection_sha256": query_projection["record_sha256"],
        },
        "reservation_input_capability_sha256": current[
            "reservation_input_capability_sha256"
        ],
        "controller_binding": binding,
        "control_chain_values": control_values,
        "control_chain_sha256": semantic_sha256(
            {"schema_version": "c28f_v5_runtime_control_chain_v1", **control_values}
        ),
        "runtime_code_sha256": _runtime_code_hashes(),
        "active_rehydrate_projection": projection,
        "active_rehydrate_projection_sha256": projection[
            "projection_binding_sha256"
        ],
        "query_source_projection": dict(query_projection),
        "legacy_authority_record_sha256": f0_a_record["record_sha256"],
    }
    authority_record = {
        **authority_base,
        "record_sha256": semantic_sha256(authority_base),
    }
    with runtime.lease("PERSIST_F0_B_AUTHORITY") as lease:
        atomic_write_immutable(
            authority_path,
            canonical_json_bytes(authority_record) + b"\n",
            lease=lease,
        )

    pointer = {"record_sha256": query_projection["record_sha256"]}

    def attach(next_state: dict[str, Any]) -> None:
        target = next_state["evals"]["F0_B"]["current"]
        expected = (
            str(authority_path),
            authority_record["record_sha256"],
            pointer,
        )
        observed = (
            target.get("authority_record_path"),
            target.get("authority_record_sha256"),
            target.get("query_projection"),
        )
        if observed not in ((None, None, None), expected):
            raise A4ControlError("F0-B authority pointer drift")
        target["authority_record_path"] = str(authority_path)
        target["authority_record_sha256"] = authority_record["record_sha256"]
        target["query_projection"] = pointer

    attached = runtime.transition(
        "ATTACH_F0_B_AUTHORITY",
        attach,
        artifacts={
            "authority_record": authority_record["record_sha256"],
            "query_projection": query_projection["record_sha256"],
        },
        require_next_action=False,
    )
    return {
        "status": "AUTHORITY_ATTACHED",
        "authority_record_sha256": authority_record["record_sha256"],
        "state_sha256": attached["state_sha256"],
    }


def issue_f0_a_generation(store: RuntimeStore | None = None) -> dict[str, Any]:
    runtime = store or RuntimeStore()
    state = runtime.read_state()
    if state.get("next_action") != ISSUE_F0_A_ACTION:
        raise A4ControlError(f"unique next action is {state.get('next_action')}")
    token = state["evals"]["F0_A"]
    if token["successful_evaluations"] != 0 or token["current"] is not None:
        raise A4ControlError("F0-A generation budget is not issuable")
    generation = int(token["next_generation"])
    legacy_path = Path(state["legacy_snapshot"]["legacy_authority_record_path"])
    legacy_record = _read_json(legacy_path)
    expected_legacy_file_sha256 = state["legacy_snapshot"]["files"][
        "legacy_authority"
    ]["file_sha256"]
    if (
        bytes_sha256(canonical_json_bytes(legacy_record) + b"\n")
        != expected_legacy_file_sha256
        or legacy_record.get("record_sha256")
        != semantic_sha256(legacy_record, ("record_sha256",))
    ):
        raise A4ControlError("legacy authority snapshot drift")
    authority, projection, handle_sha256 = _build_current_authority(legacy_record, generation)
    generation_identity = semantic_sha256(
        {
            "schema_version": "c28f_v5_runtime_generation_identity_v1",
            "goal_id": GOAL_ID,
            "generation": generation,
            "legacy_snapshot_sha256": state["legacy_snapshot"]["snapshot_sha256"],
            "projection_binding_sha256": projection["projection_binding_sha256"],
        }
    )
    run_id = "F0A-RUNTIME-" + generation_identity[:32]
    eval_id = "F0A-EVAL-" + bytes_sha256(
        ("eval:" + generation_identity).encode("ascii")
    )[:32]
    reservation_base = {
        "schema_version": "c28f_v5_runtime_reservation_v1",
        "goal_id": GOAL_ID,
        "generation": generation,
        "token_id": "F0_A",
        "eval_id": eval_id,
        "run_id": run_id,
        "authority_handle_sha256": handle_sha256,
        "authority_binding_sha256": authority.authority_binding_sha256,
        "forward_manifest_sha256": authority.manifest.manifest_sha256,
    }
    reservation_sha256 = semantic_sha256(reservation_base)
    legacy_payload = legacy_record.get("payload")
    if not isinstance(legacy_payload, Mapping):
        raise A4ControlError("legacy controller payload is absent")
    binding = _controller_binding(
        authority,
        handle_sha256=handle_sha256,
        reservation_sha256=reservation_sha256,
        legacy_payload=legacy_payload,
    )
    control_values = _control_chain_values(authority)
    authority_base = {
        "schema_version": "c28f_v5_runtime_authority_record_v1",
        "generation": generation,
        "eval_id": eval_id,
        "run_id": run_id,
        "reservation_input": reservation_base,
        "reservation_input_capability_sha256": reservation_sha256,
        "controller_binding": binding,
        "control_chain_values": control_values,
        "control_chain_sha256": semantic_sha256(
            {"schema_version": "c28f_v5_runtime_control_chain_v1", **control_values}
        ),
        "runtime_code_sha256": _runtime_code_hashes(),
        "active_rehydrate_projection": projection,
        "active_rehydrate_projection_sha256": projection["projection_binding_sha256"],
        "legacy_authority_record_sha256": legacy_record["record_sha256"],
    }
    authority_record = {
        **authority_base,
        "record_sha256": semantic_sha256(authority_base),
    }
    authority_path = runtime_artifact_path(eval_id, "authority.json", root=runtime.root)
    with runtime.lease("PREPARE_F0_A_AUTHORITY") as lease:
        atomic_write_immutable(
            authority_path,
            canonical_json_bytes(authority_record) + b"\n",
            lease=lease,
        )

    def mutate(next_state: dict[str, Any]) -> None:
        current = {
            "generation": generation,
            "status": "RESERVED",
            "eval_id": eval_id,
            "run_id": run_id,
            "authority_record_path": str(authority_path),
            "authority_record_sha256": authority_record["record_sha256"],
            "reservation_input_capability_sha256": reservation_sha256,
            "execution_binding_sha256": binding["controller_authority_binding_sha256"],
            "running_anchor": None,
            "forward_consumption": None,
            "lmdb_receipts": {},
            "cursor": None,
            "completed_material": None,
        }
        next_state["evals"]["F0_A"]["current"] = current
        next_state["evals"]["F0_A"]["next_generation"] = generation + 1
        next_state["status"] = "F0_A_RESERVED"
        next_state["next_action"] = MARK_F0_A_ACTION

    next_state = runtime.transition(
        ISSUE_F0_A_ACTION,
        mutate,
        artifacts={"authority_record": authority_record["record_sha256"]},
    )
    return {
        "status": "RESERVED",
        "generation": generation,
        "eval_id": eval_id,
        "run_id": run_id,
        "state_sha256": next_state["state_sha256"],
        "next_action": next_state["next_action"],
    }


def _running_anchor_from_state(
    state: Mapping[str, Any], token_id: str = "F0_A"
) -> dict[str, Any]:
    current = state.get("evals", {}).get(token_id, {}).get("current")
    anchor = current.get("running_anchor") if isinstance(current, Mapping) else None
    if not isinstance(anchor, dict):
        raise A4ControlError(f"{token_id} has no committed RUNNING anchor")
    return deepcopy(anchor)


def mark_f0_a_running(store: RuntimeStore | None = None) -> dict[str, Any]:
    runtime = store or RuntimeStore()
    state = runtime.read_state()
    current = state["evals"]["F0_A"]["current"]
    if not isinstance(current, dict):
        raise A4ControlError("F0-A generation is absent")
    if current["status"] == "RESERVED":
        def mark(next_state: dict[str, Any]) -> None:
            next_state["evals"]["F0_A"]["current"]["status"] = "RUNNING"
            next_state["status"] = "F0_A_RUNNING"
            next_state["next_action"] = RUN_F0_A_ACTION

        marked = runtime.transition(MARK_F0_A_ACTION, mark)
        event_tail = runtime.verify()["event_tail_sha256"]
    elif current["status"] == "RUNNING" and current.get("running_anchor") is None:
        marked = state
        event_tail = runtime.verify()["event_tail_sha256"]
    elif current["status"] == "RUNNING":
        return {
            "status": "RUNNING",
            "running_anchor": deepcopy(current["running_anchor"]),
            "next_action": state["next_action"],
        }
    else:
        raise A4ControlError(f"cannot mark F0-A from {current['status']}")
    marked_current = marked["evals"]["F0_A"]["current"]
    authority_record = _validate_authority_record(
        _read_json(Path(marked_current["authority_record_path"]))
    )
    anchor = {
        "schema_version": "c28f_a4_eval_running_capability_v1",
        "goal_id": GOAL_ID,
        "attempt_id": marked["attempt_id"],
        "run_id": marked_current["run_id"],
        "token_id": "F0_A",
        "eval_id": marked_current["eval_id"],
        "transaction_id": marked["transaction_id"],
        "running_state_sha256": marked["state_sha256"],
        "running_event_sha256": event_tail,
        "eval_registry_sha256": semantic_sha256(marked_current),
        "execution_binding_sha256": authority_record["controller_binding"][
            "controller_authority_binding_sha256"
        ],
        "reservation_input_capability_sha256": marked_current[
            "reservation_input_capability_sha256"
        ],
    }

    def attach(next_state: dict[str, Any]) -> None:
        target = next_state["evals"]["F0_A"]["current"]
        if target.get("running_anchor") not in (None, anchor):
            raise A4ControlError("RUNNING anchor drift")
        target["running_anchor"] = anchor

    attached = runtime.transition(
        "ATTACH_F0_A_RUNNING_ANCHOR",
        attach,
        require_next_action=False,
    )
    return {
        "status": "RUNNING",
        "running_anchor": anchor,
        "state_sha256": attached["state_sha256"],
        "next_action": attached["next_action"],
    }


def _validate_zero_forward_observation_ledger(
    ledger: Mapping[str, Any], *, current: Mapping[str, Any]
) -> dict[str, Any]:
    value = dict(ledger)
    exact = {
        "schema_version",
        "run_id",
        "capability_id",
        "event_count",
        "rows",
        "ledger_sha256",
    }
    rows = value.get("rows")
    if (
        set(value) != exact
        or value.get("schema_version") != "c28f_v5_a4_forward_observation_ledger_v1"
        or value.get("run_id") != current.get("run_id")
        or type(value.get("event_count")) is not int
        or not isinstance(rows, list)
        or value.get("event_count") != len(rows)
        or not rows
        or value.get("ledger_sha256")
        != semantic_sha256(value, ("ledger_sha256",))
    ):
        raise A4ControlError("zero-forward observation ledger is invalid")
    previous: str | None = None
    total_forward_count = 0
    for seq, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            raise A4ControlError("zero-forward observation row is invalid")
        counters = row.get("counters")
        if (
            set(row)
            != {
                "schema_version",
                "seq",
                "event_type",
                "token_id",
                "eval_id",
                "run_id",
                "previous_event_sha256",
                "counters",
                "details",
                "event_sha256",
            }
            or row.get("schema_version") != "c28f_v5_a4_forward_observation_v1"
            or row.get("seq") != seq
            or row.get("token_id") != "F0_A"
            or row.get("eval_id") != current.get("eval_id")
            or row.get("run_id") != current.get("run_id")
            or row.get("previous_event_sha256") != previous
            or row.get("event_sha256")
            != semantic_sha256(row, ("event_sha256",))
            or not isinstance(counters, dict)
            or set(counters)
            != {
                "content_open_count",
                "content_bytes_read",
                "lmdb_transaction_open_count",
                "model_forward_count",
            }
            or any(type(item) is not int or item < 0 for item in counters.values())
        ):
            raise A4ControlError(f"zero-forward observation chain mismatch at {seq}")
        total_forward_count += counters["model_forward_count"]
        previous = row["event_sha256"]
    if total_forward_count != 0:
        raise A4ControlError("zero-forward abort rejected after model forward")
    return {
        "event_count": len(rows),
        "ledger_sha256": value["ledger_sha256"],
        "event_tail_sha256": previous,
        "last_event_type": rows[-1]["event_type"],
        "model_forward_count": total_forward_count,
    }


def _collect_zero_forward_abort_evidence(
    runtime: RuntimeStore,
    state: Mapping[str, Any],
    current: Mapping[str, Any],
) -> dict[str, Any]:
    if (
        state.get("next_action") != RUN_F0_A_ACTION
        or current.get("status") != "RUNNING"
        or current.get("completed_material") is not None
        or state.get("safety", {}).get("model_forward_evaluations") != 0
    ):
        raise A4ControlError("F0-A is not an abortable zero-forward execution")
    cursor = current.get("cursor")
    if (
        not isinstance(cursor, Mapping)
        or cursor.get("cursor_index") != 0
        or cursor.get("chunks_sha256") != bytes_sha256(canonical_json_bytes([]))
    ):
        raise A4ControlError("zero-forward result cursor is not empty")
    authority = _validate_authority_record(
        _read_json(Path(str(current["authority_record_path"])))
    )
    projection = authority["active_rehydrate_projection"]
    manifest = projection.get("manifest")
    material = projection.get("authority_binding_material")
    if not isinstance(manifest, dict) or not isinstance(material, dict):
        raise A4ControlError("abort authority projection is incomplete")
    output_descriptor = material.get("output_root")
    if not isinstance(output_descriptor, dict):
        raise A4ControlError("abort output descriptor is absent")
    output_root = Path(str(output_descriptor.get("path"))).resolve(strict=True)
    run_root = (
        output_root
        / str(manifest.get("stage"))
        / str(manifest.get("attempt_id"))
        / str(current["run_id"])
    ).resolve(strict=True)
    if output_root not in run_root.parents:
        raise A4ControlError("abort evidence escaped the frozen output root")
    forward_started_path = run_root / "model_forward_started.json"
    if forward_started_path.exists() or forward_started_path.is_symlink():
        raise A4ControlError("zero-forward abort rejected after forward-start commit")
    observation_path = run_root / "forward_observations.json"
    receipt_path = run_root / "forward_run_receipt.json"
    observations, observations_file_sha256 = _read_forward_json(observation_path)
    observation_summary = _validate_zero_forward_observation_ledger(
        observations, current=current
    )
    receipt, receipt_file_sha256 = _read_forward_json(receipt_path)
    if (
        receipt.get("schema_version") != "c28f_v5_a4_forward_run_receipt_v1"
        or receipt.get("receipt_sha256")
        != semantic_sha256(receipt, ("receipt_sha256",))
        or receipt.get("token_id") != "F0_A"
        or receipt.get("eval_id") != current.get("eval_id")
        or receipt.get("run_id") != current.get("run_id")
        or receipt.get("status") != "RUNNING"
        or receipt.get("next_query_index") != 0
        or receipt.get("result_chunks") != []
        or cursor.get("cursor_sha256") != receipt.get("receipt_sha256")
    ):
        raise A4ControlError("zero-forward run receipt is not empty and RUNNING")
    base = {
        "schema_version": "c28f_v5_zero_forward_abort_evidence_v1",
        "status": "ABORTED",
        "reason": "PRE_FORWARD_RUNTIME_OVERHEAD_REPAIR",
        "failure_code": "PER_VALUE_CONTROL_AND_LEDGER_FSYNC_AMPLIFICATION",
        "generation": current["generation"],
        "eval_id": current["eval_id"],
        "run_id": current["run_id"],
        "authority_record_sha256": current["authority_record_sha256"],
        "failed_runtime_code_sha256": authority["runtime_code_sha256"],
        "administrative_runtime_code_sha256": _runtime_code_hashes(),
        "cursor_index": 0,
        "result_chunk_count": 0,
        "result_chunks_sha256": bytes_sha256(canonical_json_bytes([])),
        "observation_path": str(observation_path),
        "observation_file_sha256": observations_file_sha256,
        "observation_ledger_sha256": observation_summary["ledger_sha256"],
        "observation_event_count": observation_summary["event_count"],
        "observation_event_tail_sha256": observation_summary[
            "event_tail_sha256"
        ],
        "last_event_type": observation_summary["last_event_type"],
        "model_forward_count": observation_summary["model_forward_count"],
        "run_receipt_path": str(receipt_path),
        "run_receipt_file_sha256": receipt_file_sha256,
        "run_receipt_sha256": receipt["receipt_sha256"],
        "query_cursor": receipt["next_query_index"],
    }
    return {**base, "record_sha256": semantic_sha256(base)}


def abort_zero_forward_f0_a(store: RuntimeStore | None = None) -> dict[str, Any]:
    runtime = store or RuntimeStore()
    state = runtime.read_state()
    current = state["evals"]["F0_A"]["current"]
    if not isinstance(current, dict):
        raise A4ControlError("F0-A generation is absent")
    evidence = _collect_zero_forward_abort_evidence(runtime, state, current)
    evidence_path = runtime_artifact_path(
        current["eval_id"], "abort_zero_forward.json", root=runtime.root
    )
    with runtime.lease("WRITE_ZERO_FORWARD_ABORT_EVIDENCE") as lease:
        if evidence_path.exists():
            committed = read_runtime_artifact(evidence_path)
            if committed != evidence:
                raise A4ControlError("zero-forward abort evidence drift")
        else:
            atomic_write_immutable(
                evidence_path,
                canonical_json_bytes(evidence) + b"\n",
                lease=lease,
            )

    def abort(next_state: dict[str, Any]) -> None:
        token = next_state["evals"]["F0_A"]
        active = token["current"]
        if not isinstance(active, dict) or active["eval_id"] != current["eval_id"]:
            raise A4ControlError("zero-forward abort target drift")
        token["history"].append(
            {
                "generation": active["generation"],
                "status": "ABORTED",
                "eval_id": active["eval_id"],
                "run_id": active["run_id"],
                "reason": evidence["reason"],
                "failure_code": evidence["failure_code"],
                "model_forward_evaluations": 0,
                "result_chunk_count": 0,
                "abort_evidence_sha256": evidence["record_sha256"],
            }
        )
        token["current"] = None
        next_state["safety"]["aborted_zero_forward_generations"] += 1
        next_state["status"] = "READY"
        next_state["next_action"] = ISSUE_F0_A_ACTION

    aborted = runtime.transition(
        ABORT_ZERO_FORWARD_F0_A_ACTION,
        abort,
        artifacts={
            "abort_evidence": evidence["record_sha256"],
            "observation_ledger": evidence["observation_file_sha256"],
            "run_receipt": evidence["run_receipt_file_sha256"],
        },
        require_next_action=False,
    )
    return {
        "status": "ABORTED",
        "generation": current["generation"],
        "model_forward_evaluations": 0,
        "result_chunk_count": 0,
        "abort_evidence_sha256": evidence["record_sha256"],
        "state_sha256": aborted["state_sha256"],
        "next_action": aborted["next_action"],
    }


def rehydrate_current_authority(
    store: RuntimeStore | None = None, *, token_id: str = "F0_A"
) -> Any:
    from . import a4_forward

    runtime = store or RuntimeStore()
    state = runtime.read_state()
    if token_id not in {"F0_A", "F0_B"}:
        raise A4ControlError("current authority token is invalid")
    current = state["evals"][token_id]["current"]
    if not isinstance(current, dict) or current["status"] not in {"RUNNING", "COMPLETED"}:
        raise A4ControlError(f"current {token_id} authority is not active")
    record = _validate_authority_record(_read_json(Path(current["authority_record_path"])))
    if record.get("record_sha256") != current["authority_record_sha256"]:
        raise A4ControlError("current authority record pointer drift")
    return a4_forward.rehydrate_active_forward_authority(
        record["active_rehydrate_projection"]
    )


def current_runtime_snapshot(store: RuntimeStore | None = None) -> dict[str, Any]:
    return (store or RuntimeStore()).verify(recover=True)


def _completion_receipt(
    state: Mapping[str, Any],
    current: Mapping[str, Any],
    authority_record: Mapping[str, Any],
    event_sha256: str,
    *,
    token_id: str = "F0_A",
) -> dict[str, Any]:
    binding = authority_record["controller_binding"]
    running = _running_anchor_from_state(state, token_id)
    base = {
        "schema_version": "c28f_eval_token_phase_receipt_v1",
        "status": "FSYNCED_CAS_COMMITTED",
        "phase": "COMPLETED",
        "goal_id": GOAL_ID,
        "authority_id": binding["authority_id"],
        "eval_id": current["eval_id"],
        "token_id": token_id,
        "run_id": current["run_id"],
        "cas_intent_sha256": current["reservation_input_capability_sha256"],
        "authorization_sha256": binding["controller_authority_binding_sha256"],
        "execution_binding_sha256": current["execution_binding_sha256"],
        "reservation_input_capability_sha256": current[
            "reservation_input_capability_sha256"
        ],
        "transaction_id": state["transaction_id"],
        "state_sha256": state["state_sha256"],
        "event_sha256": event_sha256,
        "previous_phase_receipt_sha256": semantic_sha256(running),
        "model_forward_evaluations": 1,
    }
    return {**base, "receipt_sha256": semantic_sha256(base)}


def validate_completed_forward_input_capability(
    *,
    running_capability: A4EvalRunningCapability,
    run_id: str,
    token_id: str,
    eval_id: str,
    capability_id: str,
    authority_binding_sha256: str,
    forward_manifest_sha256: str,
    input_identity_sha256: str,
    reservation_input_capability_sha256: str,
    controller_authority_binding_sha256: str,
    forward_run_receipt_sha256: str,
) -> dict[str, Any]:
    """Bind the original input handle to the fsynced COMPLETED transition."""

    running = validate_eval_running_capability(running_capability)
    store = RuntimeStore(running_capability.runtime_root or RUNTIME_ROOT)
    state = store.read_state()
    if token_id not in {"F0_A", "F0_B"}:
        raise A4ControlError("completed input token is invalid")
    current = state["evals"][token_id]["current"]
    if not isinstance(current, dict) or current.get("status") != "COMPLETED":
        raise A4ControlError(f"completed input validation requires COMPLETED {token_id}")
    record = _validate_authority_record(
        _read_json(Path(current["authority_record_path"]))
    )
    binding = record["controller_binding"]
    expected = {
        "run_id": current["run_id"],
        "token_id": token_id,
        "eval_id": current["eval_id"],
        "authority_binding_sha256": binding["authority_binding_sha256"],
        "forward_manifest_sha256": binding["forward_manifest_sha256"],
        "input_identity_sha256": binding["input_identity_sha256"],
        "reservation_input_capability_sha256": current[
            "reservation_input_capability_sha256"
        ],
        "controller_authority_binding_sha256": binding[
            "controller_authority_binding_sha256"
        ],
    }
    observed = {
        "run_id": run_id,
        "token_id": token_id,
        "eval_id": eval_id,
        "authority_binding_sha256": authority_binding_sha256,
        "forward_manifest_sha256": forward_manifest_sha256,
        "input_identity_sha256": input_identity_sha256,
        "reservation_input_capability_sha256": reservation_input_capability_sha256,
        "controller_authority_binding_sha256": controller_authority_binding_sha256,
    }
    if observed != expected or running != _running_anchor_from_state(state, token_id):
        raise A4ControlError("completed input lifecycle binding mismatch")
    for digest in (capability_id, forward_run_receipt_sha256):
        if not isinstance(digest, str) or HEX64.fullmatch(digest) is None:
            raise A4ControlError("completed input digest is invalid")
    completion_path = runtime_artifact_path(
        current["eval_id"], "completion.json", root=store.root
    )
    completion = read_runtime_artifact(completion_path)
    if (
        completion.get("run_id") != run_id
        or completion.get("state_sha256") is None
        or completion.get("receipt_sha256")
        != semantic_sha256(completion, ("receipt_sha256",))
    ):
        raise A4ControlError("completed phase receipt is invalid")
    base = {
        "schema_version": "c28f_a4_completed_forward_input_capability_binding_v1",
        "status": "VALIDATED_ORIGINAL_INPUT_CAPABILITY_COMPLETED_ONLY",
        "goal_id": GOAL_ID,
        "attempt_id": state["attempt_id"],
        "authority_id": binding["authority_id"],
        "run_id": run_id,
        "token_id": token_id,
        "eval_id": eval_id,
        "stage": binding["stage"],
        "purpose": binding["purpose"],
        "capability_id": capability_id,
        "forward_manifest_sha256": forward_manifest_sha256,
        "input_identity_sha256": input_identity_sha256,
        "authority_binding_sha256": authority_binding_sha256,
        "reservation_input_capability_sha256": reservation_input_capability_sha256,
        "controller_authority_binding_sha256": controller_authority_binding_sha256,
        "running_capability_sha256": running_capability.capability_sha256,
        "execution_binding_sha256": completion["execution_binding_sha256"],
        "completed_transaction_id": completion["transaction_id"],
        "completed_state_sha256": completion["state_sha256"],
        "completed_event_sha256": completion["event_sha256"],
        "forward_run_receipt_sha256": forward_run_receipt_sha256,
        "token_reissue_count": 0,
        "model_forward_reexecution_count": 0,
    }
    return {**base, "binding_sha256": semantic_sha256(base)}


def rehydrate_completed_running_proof(
    store: RuntimeStore | None = None,
    *,
    token_id: str = "F0_B",
) -> A4EvalRunningCapability:
    """Rebuild only the immutable RUNNING proof for a completed eval."""

    runtime = store or RuntimeStore()
    state = runtime.read_state()
    if token_id not in {"F0_A", "F0_B"}:
        raise A4ControlError("completed RUNNING proof token is invalid")
    current = state["evals"][token_id]["current"]
    if not isinstance(current, dict) or current.get("status") != "COMPLETED":
        raise A4ControlError("completed RUNNING proof requires a completed eval")
    capability = A4EvalRunningCapability(
        _ISSUER,
        _running_anchor_from_state(state, token_id),
        runtime_root=runtime.root,
    )
    validate_eval_running_capability(capability)
    return capability


def completed_forward_input_rehydrate_material(
    *,
    running_capability: A4EvalRunningCapability,
    authority_binding_sha256: str,
    forward_manifest_sha256: str,
    input_identity_sha256: str,
    reservation_input_capability_sha256: str,
    controller_authority_binding_sha256: str,
    forward_run_receipt_sha256: str,
) -> dict[str, Any]:
    """Return exact completed-input identity without reopening active sources."""

    running = validate_eval_running_capability(running_capability)
    runtime = RuntimeStore(running_capability.runtime_root or RUNTIME_ROOT)
    state = runtime.read_state()
    token_id = str(running["token_id"])
    current = state["evals"].get(token_id, {}).get("current")
    if (
        token_id not in {"F0_A", "F0_B"}
        or not isinstance(current, dict)
        or current.get("status") != "COMPLETED"
        or current.get("run_id") != running["run_id"]
        or current.get("eval_id") != running["eval_id"]
        or _running_anchor_from_state(state, token_id) != running
    ):
        raise A4ControlError("completed input rehydrate state drift")
    record = _validate_authority_record(
        _read_json(Path(current["authority_record_path"]))
    )
    binding = record["controller_binding"]
    expected = {
        "authority_binding_sha256": binding["authority_binding_sha256"],
        "forward_manifest_sha256": binding["forward_manifest_sha256"],
        "input_identity_sha256": binding["input_identity_sha256"],
        "reservation_input_capability_sha256": current[
            "reservation_input_capability_sha256"
        ],
        "controller_authority_binding_sha256": binding[
            "controller_authority_binding_sha256"
        ],
    }
    observed = {
        "authority_binding_sha256": authority_binding_sha256,
        "forward_manifest_sha256": forward_manifest_sha256,
        "input_identity_sha256": input_identity_sha256,
        "reservation_input_capability_sha256": (
            reservation_input_capability_sha256
        ),
        "controller_authority_binding_sha256": (
            controller_authority_binding_sha256
        ),
    }
    consumption = current.get("forward_consumption")
    manifest_view = record["active_rehydrate_projection"].get("manifest")
    gpu_view = (
        manifest_view.get("gpu0_identity")
        if isinstance(manifest_view, dict)
        else None
    )
    gpu_identity = (
        gpu_view.get("semantic_sha256")
        if isinstance(gpu_view, dict)
        else None
    )
    if (
        observed != expected
        or not isinstance(consumption, dict)
        or consumption.get("receipt_sha256")
        != semantic_sha256(consumption, ("receipt_sha256",))
        or consumption.get("authority_binding_sha256")
        != authority_binding_sha256
        or not isinstance(gpu_identity, str)
        or HEX64.fullmatch(gpu_identity) is None
    ):
        raise A4ControlError("completed input rehydrate authority drift")
    completion_path = runtime_artifact_path(
        current["eval_id"], "completion.json", root=runtime.root
    )
    if completion_path.exists():
        completion = read_runtime_artifact(completion_path)
    else:
        event_tail = runtime.verify()["event_tail_sha256"]
        completion = _completion_receipt(
            state,
            current,
            record,
            str(event_tail),
            token_id=token_id,
        )
        with runtime.lease(f"RECOVER_{token_id}_COMPLETION") as lease:
            atomic_write_immutable(
                completion_path,
                canonical_json_bytes(completion) + b"\n",
                lease=lease,
            )
    material_without_id = {
        "goal_id": GOAL_ID,
        "authority_id": binding["authority_id"],
        "run_id": running["run_id"],
        "token_id": token_id,
        "eval_id": running["eval_id"],
        "purpose": binding["purpose"],
        "stage": binding["stage"],
        "forward_manifest_sha256": forward_manifest_sha256,
        "input_identity_sha256": input_identity_sha256,
        "authority_binding_sha256": authority_binding_sha256,
        "token_snapshot_sha256": running["eval_registry_sha256"],
        "gpu0_identity_sha256": gpu_identity,
        "control_chain_sha256": semantic_sha256(
            {
                "state_sha256": running["running_state_sha256"],
                "event_sha256": running["running_event_sha256"],
                "eval_registry_file_sha256": running[
                    "eval_registry_sha256"
                ],
                "execution_binding_sha256": running[
                    "execution_binding_sha256"
                ],
            }
        ),
        "running_commit_sha256": running["running_event_sha256"],
        "running_capability_sha256": running_capability.capability_sha256,
        "reservation_input_capability_sha256": (
            reservation_input_capability_sha256
        ),
        "controller_authority_binding_sha256": (
            controller_authority_binding_sha256
        ),
        "consume_receipt_sha256": consumption["receipt_sha256"],
    }
    capability_id = bytes_sha256(canonical_json_bytes(material_without_id))
    input_material = {
        "capability_id": capability_id,
        **material_without_id,
    }
    lifecycle = validate_completed_forward_input_capability(
        running_capability=running_capability,
        run_id=running["run_id"],
        token_id=token_id,
        eval_id=running["eval_id"],
        capability_id=capability_id,
        authority_binding_sha256=authority_binding_sha256,
        forward_manifest_sha256=forward_manifest_sha256,
        input_identity_sha256=input_identity_sha256,
        reservation_input_capability_sha256=(
            reservation_input_capability_sha256
        ),
        controller_authority_binding_sha256=(
            controller_authority_binding_sha256
        ),
        forward_run_receipt_sha256=forward_run_receipt_sha256,
    )
    if (
        completion.get("transaction_id")
        != lifecycle["completed_transaction_id"]
        or completion.get("state_sha256")
        != lifecycle["completed_state_sha256"]
        or completion.get("event_sha256")
        != lifecycle["completed_event_sha256"]
    ):
        raise A4ControlError("completed input rehydrate completion drift")
    base = {
        "schema_version": "c28f_a4_completed_forward_input_rehydrate_material_v1",
        "status": "VALIDATED_COMPLETED_INPUT_IDENTITY_ONLY",
        "input_capability_material": input_material,
        "completed_lifecycle_binding": lifecycle,
        "completed_phase_receipt": completion,
        "active_source_reopened": False,
        "token_reissued": False,
        "model_forward_reexecuted": False,
    }
    return {**base, "material_binding_sha256": semantic_sha256(base)}


class A4EvalForwardControlAdapter:
    """Exact compact callback surface required by :mod:`a4_forward`."""

    def __init__(
        self,
        bootstrap_lease: BootstrapLease,
        store: RuntimeStore | None = None,
        *,
        token_id: str = "F0_A",
    ) -> None:
        if not isinstance(bootstrap_lease, BootstrapLease):
            raise A4ControlError("active bootstrap lease required")
        if token_id not in {"F0_A", "F0_B"}:
            raise A4ControlError("unsupported forward token")
        self._bootstrap_lease = bootstrap_lease
        self._store = store or RuntimeStore()
        self._token_id = token_id
        self._issued_running: A4EvalRunningCapability | None = None
        self._verified_runtime_record_sha256: str | None = None

    def _state_current_record(self) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        self._bootstrap_lease.assert_active_capability()
        state = self._store.read_state()
        current = state["evals"][self._token_id]["current"]
        if not isinstance(current, dict) or current["status"] not in {"RUNNING", "COMPLETED"}:
            raise A4ControlError(f"{self._token_id} is not RUNNING/COMPLETED")
        record = _validate_authority_record(
            _read_json(Path(current["authority_record_path"]))
        )
        if record.get("record_sha256") != current.get("authority_record_sha256"):
            raise A4ControlError("authority record drift")
        if self._verified_runtime_record_sha256 != record["record_sha256"]:
            if record.get("runtime_code_sha256") != _runtime_code_hashes():
                raise A4ControlError("runtime code identity drift")
            self._verified_runtime_record_sha256 = str(record["record_sha256"])
        return state, current, record

    def assert_fsynced_control_chain(self, **values: Any) -> str:
        _state, _current, record = self._state_current_record()
        if values != record["control_chain_values"]:
            raise A4ControlError("forward control-chain binding mismatch")
        return str(record["control_chain_sha256"])

    def validate_controller_forward_authority(self, *, authority: Any) -> dict[str, Any]:
        _state, current, record = self._state_current_record()
        binding = record["controller_binding"]
        if (
            authority.authority_binding_sha256 != binding["authority_binding_sha256"]
            or authority.manifest.manifest_sha256 != binding["forward_manifest_sha256"]
            or authority.manifest.input_identity_sha256 != binding["input_identity_sha256"]
            or current["reservation_input_capability_sha256"]
            != binding["reservation_input_capability_sha256"]
        ):
            raise A4ControlError("controller authority object mismatch")
        return deepcopy(binding)

    def committed_token_id(self) -> str:
        return self._token_id

    def committed_eval_id(self) -> str:
        _state, current, _record = self._state_current_record()
        return str(current["eval_id"])

    def committed_run_id(self) -> str:
        _state, current, _record = self._state_current_record()
        return str(current["run_id"])

    def committed_token_snapshot_sha256(self) -> str:
        state, _current, _record = self._state_current_record()
        return str(_running_anchor_from_state(state, self._token_id)["eval_registry_sha256"])

    def assert_running_state_event_committed(self) -> str:
        state, _current, _record = self._state_current_record()
        anchor = _running_anchor_from_state(state, self._token_id)
        matches = list((self._store.root / "records" / "states").glob(
            f"*_{anchor['running_state_sha256']}.json"
        ))
        event_matches = list((self._store.root / "records" / "events").glob(
            f"*_{anchor['running_event_sha256']}.jsonl"
        ))
        if len(matches) != 1 or len(event_matches) != 1:
            raise A4ControlError("immutable RUNNING anchor is absent")
        running_state = _read_json(matches[0])
        running_event = _read_json(event_matches[0])
        if (
            running_state.get("state_sha256") != anchor["running_state_sha256"]
            or running_event.get("event_sha256") != anchor["running_event_sha256"]
            or running_event.get("next_state_sha256") != anchor["running_state_sha256"]
            or running_event.get("action")
            != (MARK_F0_A_ACTION if self._token_id == "F0_A" else RUN_F0_B_ACTION)
        ):
            raise A4ControlError("immutable RUNNING anchor binding drift")
        return str(anchor["running_event_sha256"])

    def issue_eval_running_capability(self) -> A4EvalRunningCapability:
        state, _current, _record = self._state_current_record()
        capability = A4EvalRunningCapability(
            _ISSUER,
            _running_anchor_from_state(state, self._token_id),
            runtime_root=self._store.root,
        )
        if self._issued_running is not None and (
            self._issued_running.capability_sha256 != capability.capability_sha256
        ):
            raise A4ControlError("RUNNING capability reissue drift")
        self._issued_running = capability
        return capability

    def bind_completed_historical_running_capability(self, capability: A4EvalRunningCapability) -> None:
        validate_eval_running_capability(capability)
        self._issued_running = capability

    def _running(self) -> A4EvalRunningCapability:
        if type(self._issued_running) is not A4EvalRunningCapability:
            raise A4ControlError("RUNNING capability has not been issued")
        return self._issued_running

    def consume_forward_capability_once(
        self,
        *,
        authority_binding_sha256: str,
        consumer_nonce_sha256: str,
    ) -> str:
        state, current, record = self._state_current_record()
        expected = record["controller_binding"]["authority_binding_sha256"]
        if authority_binding_sha256 != expected or HEX64.fullmatch(consumer_nonce_sha256) is None:
            raise A4ControlError("forward consumption binding mismatch")
        existing = current.get("forward_consumption")
        base = {
            "schema_version": "c28f_v5_runtime_forward_consumption_v1",
            "run_id": current["run_id"],
            "authority_binding_sha256": authority_binding_sha256,
            "consumer_nonce_sha256": consumer_nonce_sha256,
        }
        receipt = {**base, "receipt_sha256": semantic_sha256(base)}
        if existing is not None:
            if existing != receipt:
                raise A4ControlError("forward capability replay drift")
            return str(receipt["receipt_sha256"])

        def consume(next_state: dict[str, Any]) -> None:
            next_state["evals"][self._token_id]["current"]["forward_consumption"] = receipt

        self._store.transition(
            f"CONSUME_{self._token_id}_FORWARD_CAPABILITY",
            consume,
            artifacts={"consumption_receipt": receipt["receipt_sha256"]},
            require_next_action=False,
        )
        return str(receipt["receipt_sha256"])

    def commit_lmdb_batch(
        self,
        *,
        source_id: str,
        source_identity_sha256: str,
        key_manifest_sha256: str,
        key_set_sha256: str,
        source_manifest_sha256: str,
        batch_sha256: str,
        key_count: int,
        shared_ledger_tail_sha256: str,
    ) -> A4LmdbBatchControlReceipt:
        running = validate_eval_running_capability(self._running())
        _state, current, _record = self._state_current_record()
        base = {
            "schema_version": "c28f_a4_lmdb_batch_control_receipt_v1",
            "running_anchor": running,
            "source_id": source_id,
            "source_identity_sha256": source_identity_sha256,
            "key_manifest_sha256": key_manifest_sha256,
            "key_set_sha256": key_set_sha256,
            "key_count": key_count,
            "source_manifest_sha256": source_manifest_sha256,
            "batch_sha256": batch_sha256,
            "shared_ledger_tail_sha256": shared_ledger_tail_sha256,
        }
        receipt = {**base, "persisted_receipt_sha256": semantic_sha256(base)}
        existing = current["lmdb_receipts"].get(source_id)
        if existing is not None and existing != receipt:
            raise A4ControlError("LMDB receipt reissue drift")
        if existing is None:
            def commit(next_state: dict[str, Any]) -> None:
                next_state["evals"][self._token_id]["current"]["lmdb_receipts"][source_id] = receipt

            self._store.transition(
                f"COMMIT_{self._token_id}_LMDB_BATCH",
                commit,
                artifacts={source_id: receipt["persisted_receipt_sha256"]},
                require_next_action=False,
            )
        capability = A4LmdbBatchControlReceipt(_ISSUER, receipt)
        validate_lmdb_batch_control_receipt(capability)
        return capability

    def commit_lmdb_batch_receipt(self, **values: Any) -> str:
        return str(validate_lmdb_batch_control_receipt(self.commit_lmdb_batch(**values))[
            "persisted_receipt_sha256"
        ])

    def issue_persisted_lmdb_batch_control_receipt(self, *, source_id: str) -> A4LmdbBatchControlReceipt:
        _state, current, _record = self._state_current_record()
        receipt = current["lmdb_receipts"].get(source_id)
        if not isinstance(receipt, dict):
            raise A4ControlError("LMDB persisted source receipt absent")
        capability = A4LmdbBatchControlReceipt(_ISSUER, receipt)
        validate_lmdb_batch_control_receipt(capability)
        return capability

    def assert_lmdb_batch_receipt_persisted(self, *, source_id: str, receipt_sha256: str) -> str:
        receipt = validate_lmdb_batch_control_receipt(
            self.issue_persisted_lmdb_batch_control_receipt(source_id=source_id)
        )
        if receipt["persisted_receipt_sha256"] != receipt_sha256:
            raise A4ControlError("LMDB persisted receipt mismatch")
        return receipt_sha256

    def commit_eval_cursor(
        self,
        *,
        cursor_index: int,
        cursor_sha256: str,
        chunks_sha256: str,
        artifacts_sha256: str,
    ) -> Mapping[str, Any]:
        running = validate_eval_running_capability(self._running())
        _state, current, _record = self._state_current_record()
        previous = current.get("cursor")
        if previous is not None and cursor_index == previous["cursor_index"]:
            expected = {
                "cursor_sha256": cursor_sha256,
                "chunks_sha256": chunks_sha256,
                "artifacts_sha256": artifacts_sha256,
            }
            if any(previous.get(name) != value for name, value in expected.items()):
                raise A4ControlError("cursor idempotency drift")
            return deepcopy(previous)
        expected_index = 0 if previous is None else int(previous["cursor_index"]) + 1
        base = {
            "schema_version": "c28f_v5_runtime_eval_cursor_receipt_v1",
            "run_id": current["run_id"],
            "eval_id": current["eval_id"],
            "cursor_index": cursor_index,
            "cursor_sha256": cursor_sha256,
            "chunks_sha256": chunks_sha256,
            "artifacts_sha256": artifacts_sha256,
            "previous_cursor_receipt_sha256": (
                previous["receipt_sha256"] if previous is not None else None
            ),
            "running_capability_sha256": self._running().capability_sha256,
        }
        receipt = {**base, "receipt_sha256": semantic_sha256(base)}
        if cursor_index != expected_index or running["run_id"] != current["run_id"]:
            raise A4ControlError("cursor is not monotonic")

        def commit(next_state: dict[str, Any]) -> None:
            next_state["evals"][self._token_id]["current"]["cursor"] = receipt

        self._store.transition(
            f"COMMIT_{self._token_id}_CURSOR",
            commit,
            artifacts={"cursor_receipt": receipt["receipt_sha256"]},
            require_next_action=False,
        )
        return receipt

    def issue_eval_cursor_capability(
        self,
        *,
        running_capability: A4EvalRunningCapability,
    ) -> A4EvalCursorCapability:
        running = validate_eval_running_capability(running_capability)
        _state, current, _record = self._state_current_record()
        cursor = current.get("cursor")
        if not isinstance(cursor, dict):
            raise A4EvalCursorAbsentError("EVAL_CURSOR_ABSENT")
        payload = {
            "schema_version": "c28f_a4_eval_cursor_capability_v1",
            "running_anchor": running,
            "cursor_index": cursor["cursor_index"],
            "cursor_receipt_sha256": cursor["receipt_sha256"],
            "cursor_sha256": cursor["cursor_sha256"],
            "chunks_sha256": cursor["chunks_sha256"],
            "artifacts_sha256": cursor["artifacts_sha256"],
        }
        capability = A4EvalCursorCapability(_ISSUER, payload)
        validate_eval_cursor_capability(capability)
        return capability

    def complete_eval_token(self, *, cursor_capability: A4EvalCursorCapability) -> Mapping[str, Any]:
        cursor = validate_eval_cursor_capability(cursor_capability)
        state, current, record = self._state_current_record()
        if current["status"] == "COMPLETED":
            receipt_path = runtime_artifact_path(
                current["eval_id"], "completion.json", root=self._store.root
            )
            if receipt_path.exists():
                return read_runtime_artifact(receipt_path)
            tail = self._store.verify()["event_tail_sha256"]
            receipt = _completion_receipt(
                state, current, record, str(tail), token_id=self._token_id
            )
            with self._store.lease(f"RECOVER_{self._token_id}_COMPLETION") as lease:
                atomic_write_immutable(
                    receipt_path,
                    canonical_json_bytes(receipt) + b"\n",
                    lease=lease,
                )
            return receipt
        issued = validate_eval_cursor_capability(
            self.issue_eval_cursor_capability(running_capability=self._running())
        )
        if cursor != issued:
            raise A4ControlError("completion cursor drift")

        def complete(next_state: dict[str, Any]) -> None:
            token = next_state["evals"][self._token_id]
            token["current"]["status"] = "COMPLETED"
            token["current"]["model_forward_evaluations"] = 1
            token["successful_evaluations"] = 1
            next_state["safety"]["model_forward_evaluations"] = sum(
                item["successful_evaluations"]
                for item in next_state["evals"].values()
            )
            if self._token_id == "F0_A":
                next_state["status"] = "F0_A_COMPLETED"
                next_state["next_action"] = F0_A_ANALYZE_ACTION
            else:
                next_state["status"] = "F0_B_COMPLETED"
                next_state["next_action"] = F0_B_ANALYZE_ACTION

        completed = self._store.transition(
            RUN_F0_A_ACTION if self._token_id == "F0_A" else RUN_F0_B_ACTION,
            complete,
            artifacts={"cursor_receipt": cursor["cursor_receipt_sha256"]},
        )
        tail = self._store.verify()["event_tail_sha256"]
        completed_current = completed["evals"][self._token_id]["current"]
        receipt = _completion_receipt(
            completed,
            completed_current,
            record,
            str(tail),
            token_id=self._token_id,
        )
        path = runtime_artifact_path(
            completed_current["eval_id"], "completion.json", root=self._store.root
        )
        with self._store.lease(f"PERSIST_{self._token_id}_COMPLETION") as lease:
            atomic_write_immutable(path, canonical_json_bytes(receipt) + b"\n", lease=lease)
        return receipt

    def persist_completed_forward_rehydrate_material(
        self,
        *,
        completed_result_verification_projection: Mapping[str, Any],
        input_capability_material: Mapping[str, Any],
        run_root: Mapping[str, Any],
        chunk_root: Mapping[str, Any],
        completed_forward_validation_receipt: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        state, current, _record = self._state_current_record()
        if current["status"] != "COMPLETED":
            raise A4ControlError(
                f"completed material requires COMPLETED {self._token_id}"
            )
        material_base = {
            "schema_version": "c28f_v5_runtime_completed_forward_material_v1",
            "eval_id": current["eval_id"],
            "run_id": current["run_id"],
            "running_anchor": _running_anchor_from_state(state, self._token_id),
            "completed_result_verification_projection": dict(
                completed_result_verification_projection
            ),
            "input_capability_material": dict(input_capability_material),
            "run_root": dict(run_root),
            "chunk_root": dict(chunk_root),
            "completed_forward_validation_receipt": dict(
                completed_forward_validation_receipt
            ),
        }
        material = {**material_base, "material_binding_sha256": semantic_sha256(material_base)}
        path = runtime_artifact_path(
            current["eval_id"], "completed_material.json", root=self._store.root
        )
        with self._store.lease(
            f"PERSIST_{self._token_id}_COMPLETED_MATERIAL"
        ) as lease:
            atomic_write_immutable(path, canonical_json_bytes(material) + b"\n", lease=lease)
        existing = current.get("completed_material")
        pointer = {
            "path": str(path),
            "material_binding_sha256": material["material_binding_sha256"],
        }
        if existing is None:
            def attach(next_state: dict[str, Any]) -> None:
                next_state["evals"][self._token_id]["current"]["completed_material"] = pointer

            self._store.transition(
                f"ATTACH_{self._token_id}_COMPLETED_MATERIAL",
                attach,
                artifacts={"completed_material": material["material_binding_sha256"]},
                require_next_action=False,
            )
        elif existing != pointer:
            raise A4ControlError("completed material pointer drift")
        return pointer
