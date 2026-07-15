"""F1 synthetic protocol evidence; final G6 PASS requires control-ledger binding."""

from __future__ import annotations

import hashlib
import math
import os
import re
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .canonical import canonical_json_bytes, semantic_sha256


F1_PROTOCOL_SCHEMA = "c28f_f1_protocol_v1"
SCHEDULE_SCHEMA = "c28f_optimizer_progress_schedule_v1"
TEMPORAL_SCHEMA = "c28f_temporal_contract_v1"
CHECKPOINT_SCHEMA = "c28f_checkpoint_contract_v1"
RECOVERY_SCHEMA = "c28f_recovery_matrix_v1"
SUPERSET_SCHEMA = "c28f_superset_scaffold_v1"
COST_SCHEMA = "c28f_cost_measurement_protocol_v1"
SYNTHETIC_SMOKE_SCHEMA = "c28f_f1_synthetic_scheduler_recovery_smoke_v1"

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_GIT_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_STATE_VERSION_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
_CHECKPOINT_EVAL_SPLITS = {
    "calib_select",
    "train_fit_route_dev",
    "train_fit_mechanism_dev",
    "train_fit_confirm",
    "train_fit_calibration",
}

SCHEDULE_BOUNDARIES: tuple[tuple[str, float], ...] = (
    ("P0", 0.00),
    ("P1", 0.10),
    ("P2", 0.25),
    ("P3", 0.40),
    ("P4", 0.60),
    ("P5", 0.75),
    ("P6", 1.00),
)
FROZEN_U_FORMAL_UPDATES = 17_360
FROZEN_C28E_CHECKPOINT_SHA256 = (
    "fa0bafc6b5c1a0d049d462bcd0b77a073222715af0713ae291c05ea8631d13ab"
)


class F1ContractError(RuntimeError):
    pass


def _exact_int(value: Any, name: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise F1ContractError(f"{name} must be an exact integer >= {minimum}")
    return value


def _exact_sha256(value: Any, name: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise F1ContractError(f"{name} must be an exact lowercase SHA-256")
    return value


def optimizer_progress(completed_updates: int, planned_updates: int) -> float:
    completed = _exact_int(completed_updates, "completed_updates")
    planned = _exact_int(planned_updates, "planned_updates", minimum=1)
    if completed > planned:
        raise F1ContractError("completed optimizer updates exceed the frozen plan")
    return completed / planned


def schedule_phase(completed_updates: int, planned_updates: int) -> dict[str, Any]:
    progress = optimizer_progress(completed_updates, planned_updates)
    if completed_updates == planned_updates:
        active = "P6"
    elif 4 * completed_updates >= 3 * planned_updates:
        active = "P5"
    elif 5 * completed_updates >= 3 * planned_updates:
        active = "P4"
    elif 5 * completed_updates >= 2 * planned_updates:
        active = "P3"
    elif 4 * completed_updates >= planned_updates:
        active = "P2"
    elif 10 * completed_updates >= planned_updates:
        active = "P1"
    else:
        active = "P0"
    gt_span_aux = (
        0.0
        if 4 * completed_updates >= planned_updates
        else 1.0 - (4.0 * completed_updates / planned_updates)
    )
    teacher_support_multiplier = (
        0.0
        if 5 * completed_updates >= 3 * planned_updates
        else 1.0 - (5.0 * completed_updates / (3.0 * planned_updates))
    )
    return {
        "schema_version": SCHEDULE_SCHEMA,
        "completed_optimizer_updates": completed_updates,
        "planned_optimizer_updates": planned_updates,
        "progress": progress,
        "phase": active,
        "frozen_weight_multipliers": {
            "gt_span_aux": gt_span_aux,
            "gt_video_support": teacher_support_multiplier,
            "teacher_shadow_distill": teacher_support_multiplier,
            "teacher_candidate": 0.0,
            "credit_loss_in_f0_f1": 0.0,
        },
        "exact_zero_boundaries": {
            "gt_span_aux": 0.25,
            "gt_video_support": 0.60,
            "teacher_shadow_distill": 0.60,
            "teacher_candidate": 0.0,
            "credit_loss_in_f0_f1": 0.0,
        },
        "progress_source": "completed_optimizer_updates/planned_optimizer_updates",
        "microstep_advances_progress": False,
    }


def assert_microstep_invariance(
    *,
    before_completed_updates: int,
    after_completed_updates: int,
    optimizer_step_committed: bool,
) -> None:
    before = _exact_int(before_completed_updates, "before_completed_updates")
    after = _exact_int(after_completed_updates, "after_completed_updates")
    if type(optimizer_step_committed) is not bool:
        raise F1ContractError("optimizer_step_committed must be an exact boolean")
    expected = before + (1 if optimizer_step_committed else 0)
    if after != expected:
        raise F1ContractError("optimizer progress changed outside an optimizer-step commit")


@dataclass(frozen=True)
class TemporalGridSpec:
    duration_sec: float
    cells: int

    def __post_init__(self) -> None:
        if isinstance(self.duration_sec, bool) or not isinstance(
            self.duration_sec, (int, float)
        ) or not math.isfinite(
            float(self.duration_sec)
        ):
            raise F1ContractError("duration is required and must be finite")
        if float(self.duration_sec) <= 0.0:
            raise F1ContractError("duration must be strictly positive")
        _exact_int(self.cells, "cells", minimum=1)

    @property
    def cell_width_sec(self) -> float:
        return float(self.duration_sec) / self.cells

    def boundary_sec(self, index: int) -> float:
        value = _exact_int(index, "boundary index")
        if value > self.cells:
            raise F1ContractError("boundary index exceeds grid")
        return float(self.duration_sec) * value / self.cells

    def cell_interval_sec(self, index: int) -> tuple[float, float]:
        value = _exact_int(index, "cell index")
        if value >= self.cells:
            raise F1ContractError("cell index exceeds grid")
        return self.boundary_sec(value), self.boundary_sec(value + 1)

    def seconds_to_cell_span(self, start_sec: float, end_sec: float) -> tuple[int, int]:
        if any(
            isinstance(value, bool) or not isinstance(value, (int, float))
            for value in (start_sec, end_sec)
        ):
            raise F1ContractError("span seconds must be real numbers")
        start = float(start_sec)
        end = float(end_sec)
        if not all(math.isfinite(value) for value in (start, end)):
            raise F1ContractError("span seconds must be finite")
        if start < 0.0 or end <= start or end > float(self.duration_sec):
            raise F1ContractError("span is outside the valid half-open duration")
        def boundary_insertion_index(
            value: float,
            *,
            after_equal: bool,
        ) -> int:
            low = 0
            high = self.cells + 1
            while low < high:
                middle = (low + high) // 2
                boundary = self.boundary_sec(middle)
                if boundary < value or (after_equal and boundary == value):
                    low = middle + 1
                else:
                    high = middle
            return low

        start_cell = min(
            self.cells - 1,
            max(
                0,
                boundary_insertion_index(start, after_equal=True) - 1,
            ),
        )
        end_cell_exclusive = min(
            self.cells,
            max(
                start_cell + 1,
                boundary_insertion_index(end, after_equal=False),
            ),
        )
        return start_cell, end_cell_exclusive

    def cell_span_to_seconds(self, start_cell: int, end_cell_exclusive: int) -> tuple[float, float]:
        start = _exact_int(start_cell, "start_cell")
        end = _exact_int(end_cell_exclusive, "end_cell_exclusive", minimum=1)
        if start >= end or end > self.cells:
            raise F1ContractError("invalid half-open cell span")
        return self.boundary_sec(start), self.boundary_sec(end)


@dataclass(frozen=True)
class TemporalTensorContract:
    grid: TemporalGridSpec
    valid_cells: int
    mask: tuple[bool, ...]
    source_edges_sec: tuple[float, ...]

    def __post_init__(self) -> None:
        valid = _exact_int(self.valid_cells, "valid_cells", minimum=1)
        if valid > self.grid.cells:
            raise F1ContractError("valid temporal cells exceed the target grid")
        if len(self.mask) != self.grid.cells or any(type(value) is not bool for value in self.mask):
            raise F1ContractError("temporal mask must be an exact boolean target-length tuple")
        if self.mask != (True,) * valid + (False,) * (self.grid.cells - valid):
            raise F1ContractError("temporal mask must be true-prefix then right padding")
        if len(self.source_edges_sec) != valid + 1:
            raise F1ContractError("source edge count must equal valid_cells + 1")
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            for value in self.source_edges_sec
        ):
            raise F1ContractError("source temporal edges must be finite numbers")
        edges = tuple(float(value) for value in self.source_edges_sec)
        if edges[0] != 0.0 or not math.isclose(
            edges[-1], float(self.grid.duration_sec), rel_tol=0.0, abs_tol=1e-12
        ):
            raise F1ContractError("source temporal edges must cover the exact duration")
        if any(right <= left for left, right in zip(edges, edges[1:])):
            raise F1ContractError("source temporal edges must be strictly increasing")

    def audit(self) -> dict[str, Any]:
        base = {
            "duration_sec": float(self.grid.duration_sec),
            "target_cells": self.grid.cells,
            "valid_cells": self.valid_cells,
            "right_padding_cells": self.grid.cells - self.valid_cells,
            "mask_true_count": sum(self.mask),
            "mask_false_count": len(self.mask) - sum(self.mask),
            "source_edge_count": len(self.source_edges_sec),
            "source_first_edge_sec": float(self.source_edges_sec[0]),
            "source_last_edge_sec": float(self.source_edges_sec[-1]),
            "padding_policy": "RIGHT_ZERO_PAD_MASK_FALSE",
            "interval_semantics": "HALF_OPEN",
        }
        return {**base, "audit_sha256": semantic_sha256(base)}


def temporal_oracles() -> dict[str, Any]:
    cases: list[dict[str, Any]] = []
    for cells in (64, 96, 128):
        for duration in (0.25, 1.0, 5.0, 60.0, 3_600.0):
            grid = TemporalGridSpec(duration, cells)
            probes = (
                (0.0, duration / cells),
                (duration / 3.0, min(duration, duration / 3.0 + duration / cells)),
                (max(0.0, duration - duration / cells), duration),
            )
            encoded = []
            for start, end in probes:
                cell_span = grid.seconds_to_cell_span(start, end)
                decoded = grid.cell_span_to_seconds(*cell_span)
                encoded.append(
                    {
                        "input_sec": [start, end],
                        "cell_span_half_open": list(cell_span),
                        "decoded_cell_bounds_sec": list(decoded),
                        "covers_input": decoded[0] <= start + 1e-12
                        and decoded[1] + 1e-12 >= end,
                    }
                )
            cases.append(
                {
                    "duration_sec": duration,
                    "cells": cells,
                    "cell_width_sec": grid.cell_width_sec,
                    "probes": encoded,
                }
            )
    base = {
        "schema_version": TEMPORAL_SCHEMA,
        "fixed_divisor_64_used": False,
        "duration_missing_policy": "HARD_FAIL",
        "interval_semantics": "HALF_OPEN",
        "mask_and_padding_contract": {
            "mask": "true_prefix_for_valid_cells_then_false_right_padding",
            "padding": "right_zero_pad_mask_false",
            "source_edges": "strictly_increasing_0_to_duration_inclusive",
        },
        "cases": cases,
    }
    return {**base, "oracles_sha256": semantic_sha256(base)}


CHECKPOINT_REQUIRED_KEYS = (
    "schema_version",
    "model",
    "optimizer",
    "scheduler",
    "scaler",
    "ema",
    "rng_python",
    "rng_numpy",
    "rng_torch_cpu",
    "rng_torch_cuda",
    "epoch",
    "global_optimizer_step",
    "sampler_state",
    "dataloader_state",
    "distributed_state",
    "accumulation_state",
    "query_cursor",
    "hard_negative_index_sha256",
    "candidate_refresh_state",
    "gt_support_hash_version",
    "candidate_snapshot",
    "candidate_snapshot_sha256",
    "temporal_manifest_sha256",
    "metric_schema_sha256",
    "code_commit",
    "dirty_diff_sha256",
    "config_sha256",
    "loss_weight_state",
    "eval_artifact_state",
    "completed_optimizer_updates",
    "planned_optimizer_updates",
)


def validate_checkpoint_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise F1ContractError("checkpoint payload must be a mapping")
    missing = [key for key in CHECKPOINT_REQUIRED_KEYS if key not in payload]
    if missing:
        raise F1ContractError(f"checkpoint payload missing required keys: {missing}")
    if set(payload) != set(CHECKPOINT_REQUIRED_KEYS):
        raise F1ContractError("checkpoint payload field set must be exact")
    if payload["schema_version"] != CHECKPOINT_SCHEMA:
        raise F1ContractError("checkpoint payload schema version drift")
    for key in (
        "model",
        "optimizer",
        "scheduler",
        "scaler",
        "ema",
        "sampler_state",
        "candidate_snapshot",
    ):
        if not isinstance(payload[key], Mapping):
            raise F1ContractError(f"checkpoint {key} state must be a mapping")
    for key in ("rng_python", "rng_numpy", "rng_torch_cpu", "rng_torch_cuda"):
        if payload[key] is None:
            raise F1ContractError(f"checkpoint {key} state is missing")
    completed = _exact_int(payload["completed_optimizer_updates"], "completed updates")
    planned = _exact_int(payload["planned_optimizer_updates"], "planned updates", minimum=1)
    if completed > planned:
        raise F1ContractError("checkpoint optimizer cursor exceeds plan")
    epoch = _exact_int(payload["epoch"], "checkpoint epoch")
    global_step = _exact_int(payload["global_optimizer_step"], "global optimizer step")
    if global_step != completed:
        raise F1ContractError("checkpoint global step/completed-update mismatch")
    cursor = _exact_int(payload["query_cursor"], "query cursor")
    sampler = payload["sampler_state"]
    if set(sampler) != {"epoch", "query_cursor", "permutation_sha256"}:
        raise F1ContractError("checkpoint sampler state field set is not exact")
    if sampler.get("epoch") != epoch or sampler.get("query_cursor") != cursor:
        raise F1ContractError("checkpoint sampler/query cursor mismatch")
    _exact_sha256(sampler.get("permutation_sha256"), "sampler permutation SHA")

    dataloader = payload["dataloader_state"]
    if not isinstance(dataloader, Mapping) or set(dataloader) != {
        "worker_count",
        "worker_seed_state_sha256",
        "persistent_workers",
    }:
        raise F1ContractError("checkpoint DataLoader state field set is not exact")
    _exact_int(dataloader.get("worker_count"), "DataLoader worker count")
    _exact_sha256(
        dataloader.get("worker_seed_state_sha256"),
        "DataLoader worker seed state SHA",
    )
    if type(dataloader.get("persistent_workers")) is not bool:
        raise F1ContractError("DataLoader persistent_workers must be an exact boolean")

    distributed = payload["distributed_state"]
    if not isinstance(distributed, Mapping) or set(distributed) != {"rank", "world_size"}:
        raise F1ContractError("checkpoint distributed state field set is not exact")
    rank = _exact_int(distributed.get("rank"), "distributed rank")
    world_size = _exact_int(distributed.get("world_size"), "distributed world size", minimum=1)
    if rank >= world_size:
        raise F1ContractError("distributed rank must be smaller than world size")

    accumulation = payload["accumulation_state"]
    if not isinstance(accumulation, Mapping) or set(accumulation) != {
        "microstep",
        "accumulation_steps",
        "at_optimizer_step_boundary",
        "gradient_buffers_included",
        "communication_state_included",
    }:
        raise F1ContractError("checkpoint accumulation state field set is not exact")
    microstep = _exact_int(accumulation.get("microstep"), "accumulation microstep")
    accumulation_steps = _exact_int(
        accumulation.get("accumulation_steps"),
        "gradient accumulation steps",
        minimum=1,
    )
    if microstep >= accumulation_steps:
        raise F1ContractError("accumulation microstep exceeds its cycle")
    boundary = accumulation.get("at_optimizer_step_boundary")
    gradients_included = accumulation.get("gradient_buffers_included")
    communication_included = accumulation.get("communication_state_included")
    if any(type(value) is not bool for value in (boundary, gradients_included, communication_included)):
        raise F1ContractError("accumulation flags must be exact booleans")
    if boundary is not (microstep == 0):
        raise F1ContractError("accumulation boundary/microstep mismatch")
    if boundary and (gradients_included or communication_included):
        raise F1ContractError("boundary checkpoints must not claim pending gradient/communication state")
    if not boundary and not (gradients_included and communication_included):
        raise F1ContractError(
            "mid-accumulation checkpoints require gradient buffers and communication state"
        )

    refresh = payload["candidate_refresh_state"]
    if not isinstance(refresh, Mapping) or set(refresh) != {
        "scheduler_state",
        "refresh_cursor",
        "snapshot_id",
        "snapshot_sha256",
    }:
        raise F1ContractError("candidate refresh state field set is not exact")
    if not isinstance(refresh.get("scheduler_state"), Mapping):
        raise F1ContractError("candidate refresh scheduler state must be a mapping")
    refresh_cursor = _exact_int(refresh.get("refresh_cursor"), "candidate refresh cursor")
    snapshot_id = refresh.get("snapshot_id")
    if not isinstance(snapshot_id, str) or _STATE_VERSION_RE.fullmatch(snapshot_id) is None:
        raise F1ContractError("candidate snapshot ID is not canonical")
    _exact_sha256(refresh.get("snapshot_sha256"), "candidate refresh snapshot SHA")

    gt_hash_version = payload["gt_support_hash_version"]
    if not isinstance(gt_hash_version, str) or _STATE_VERSION_RE.fullmatch(gt_hash_version) is None:
        raise F1ContractError("GT-support hash version is not canonical")
    snapshot = payload["candidate_snapshot"]
    if set(snapshot) != {
        "query_cursor",
        "refresh_cursor",
        "snapshot_id",
        "fingerprint_sha256",
        "gt_support_hash_version",
    }:
        raise F1ContractError("candidate snapshot field set is not exact")
    if snapshot.get("query_cursor") != cursor:
        raise F1ContractError("checkpoint candidate snapshot/query cursor mismatch")
    if snapshot.get("refresh_cursor") != refresh_cursor or snapshot.get("snapshot_id") != snapshot_id:
        raise F1ContractError("checkpoint candidate refresh/snapshot cursor mismatch")
    if snapshot.get("gt_support_hash_version") != gt_hash_version:
        raise F1ContractError("checkpoint GT-support hash version mismatch")
    _exact_sha256(snapshot.get("fingerprint_sha256"), "candidate snapshot fingerprint")
    expected_snapshot_sha = semantic_sha256(dict(snapshot))
    if payload["candidate_snapshot_sha256"] != expected_snapshot_sha:
        raise F1ContractError("candidate snapshot semantic SHA mismatch")

    _exact_sha256(payload["hard_negative_index_sha256"], "hard-negative index SHA")
    for key in (
        "temporal_manifest_sha256",
        "metric_schema_sha256",
        "dirty_diff_sha256",
        "config_sha256",
    ):
        _exact_sha256(payload[key], f"checkpoint {key}")
    if not isinstance(payload["code_commit"], str) or _GIT_COMMIT_RE.fullmatch(
        payload["code_commit"]
    ) is None:
        raise F1ContractError("checkpoint code commit must be an exact 40-hex Git object ID")

    weights = payload["loss_weight_state"]
    expected_weight_keys = {
        "gt_span_aux",
        "gt_video_support",
        "teacher_shadow_distill",
        "teacher_candidate_rate",
        "credit_loss",
    }
    if not isinstance(weights, Mapping) or set(weights) != expected_weight_keys:
        raise F1ContractError("checkpoint loss-weight state field set is not exact")
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < 0.0
        for value in weights.values()
    ):
        raise F1ContractError("checkpoint loss weights must be finite and non-negative")

    eval_state = payload["eval_artifact_state"]
    if not isinstance(eval_state, (list, tuple)):
        raise F1ContractError("checkpoint eval artifact state must be an explicit sequence")
    seen_eval_ids = set()
    for index, raw in enumerate(eval_state):
        if not isinstance(raw, Mapping) or set(raw) != {
            "eval_id",
            "split",
            "split_manifest_sha256",
            "evaluator_sha256",
            "prediction_artifact_sha256",
        }:
            raise F1ContractError(f"checkpoint eval artifact state is not exact: {index}")
        eval_id = raw.get("eval_id")
        split = raw.get("split")
        if (
            not isinstance(eval_id, str)
            or _STATE_VERSION_RE.fullmatch(eval_id) is None
            or eval_id in seen_eval_ids
        ):
            raise F1ContractError("checkpoint eval IDs must be canonical and unique")
        if not isinstance(split, str) or _STATE_VERSION_RE.fullmatch(split) is None:
            raise F1ContractError("checkpoint eval split is not canonical")
        if split not in _CHECKPOINT_EVAL_SPLITS:
            raise F1ContractError("checkpoint references a non-authorized eval split")
        seen_eval_ids.add(eval_id)
        for field in (
            "split_manifest_sha256",
            "evaluator_sha256",
            "prediction_artifact_sha256",
        ):
            _exact_sha256(raw.get(field), f"checkpoint eval[{index}].{field}")
    return {
        "schema_version": CHECKPOINT_SCHEMA,
        "status": "VALID",
        "epoch": epoch,
        "completed_optimizer_updates": completed,
        "planned_optimizer_updates": planned,
        "query_cursor": cursor,
        "at_optimizer_step_boundary": boundary,
        "distributed_rank": rank,
        "distributed_world_size": world_size,
        "candidate_refresh_cursor": refresh_cursor,
        "eval_artifact_count": len(eval_state),
        "required_keys": list(CHECKPOINT_REQUIRED_KEYS),
    }


def checkpoint_reference_contract() -> dict[str, Any]:
    base = {
        "schema_version": CHECKPOINT_SCHEMA,
        "references": {
            "latest": {"selection_use": False, "purpose": "RECOVERY"},
            "composite": {"selection_use": True, "metric": "FROZEN_COMPOSITE"},
            "primary_r1": {"selection_use": True, "metric": "VCMR_R1_IOU_0_7"},
            "true_vr": {"selection_use": True, "metric": "FROZEN_TRUE_VR"},
        },
        "references_must_be_distinct": True,
        "payload_required_keys": list(CHECKPOINT_REQUIRED_KEYS),
        "optimizer_boundary_rule": {
            "default": "SAVE_ONLY_AT_COMPLETE_OPTIMIZER_STEP_BOUNDARY",
            "mid_accumulation_exception": (
                "REQUIRES_GRADIENT_BUFFERS_AND_DISTRIBUTED_COMMUNICATION_STATE"
            ),
        },
        "recovery_state_groups": {
            "training": [
                "model",
                "optimizer",
                "scheduler",
                "scaler",
                "ema",
            ],
            "rng_and_data_order": [
                "rng_python",
                "rng_numpy",
                "rng_torch_cpu",
                "rng_torch_cuda",
                "sampler_state",
                "dataloader_state",
                "distributed_state",
                "accumulation_state",
            ],
            "candidate_lineage": [
                "hard_negative_index_sha256",
                "candidate_refresh_state",
                "gt_support_hash_version",
                "candidate_snapshot",
                "candidate_snapshot_sha256",
            ],
            "protocol_lineage": [
                "temporal_manifest_sha256",
                "metric_schema_sha256",
                "code_commit",
                "dirty_diff_sha256",
                "config_sha256",
                "loss_weight_state",
                "eval_artifact_state",
            ],
        },
        "write_protocol": [
            "write_unique_temp",
            "flush",
            "fsync_file",
            "atomic_rename",
            "fsync_parent",
            "write_immutable_manifest",
        ],
        "rolling_latest_may_delete_metric_peak": False,
    }
    return {**base, "contract_sha256": semantic_sha256(base)}


def recovery_matrix() -> dict[str, Any]:
    scenarios = [
        {
            "crash_point": "BEFORE_CHECKPOINT_COMMIT",
            "last_green": "PREVIOUS_CHECKPOINT",
            "resume": "REPLAY_UNCOMMITTED_SYNTHETIC_UPDATE",
            "may_claim_completed": False,
        },
        {
            "crash_point": "AFTER_CHECKPOINT_BEFORE_MANIFEST",
            "last_green": "PREVIOUS_MANIFEST",
            "resume": "VERIFY_OR_QUARANTINE_ORPHAN_CHECKPOINT",
            "may_claim_completed": False,
        },
        {
            "crash_point": "DURING_EVAL",
            "last_green": "LAST_IMMUTABLE_EVAL_CHUNK",
            "resume": "SAME_TOKEN_EVAL_ID_INPUT_HASH_AND_CURSOR_ONLY",
            "may_claim_completed": False,
        },
        {
            "crash_point": "BEFORE_STATUS_RENAME",
            "last_green": "COMMITTED_ARTIFACTS_AND_EVENT_INTENT",
            "resume": "VERIFY_POSTIMAGE_THEN_IDEMPOTENT_STATUS_RENAME",
            "may_claim_completed": False,
        },
    ]
    base = {
        "schema_version": RECOVERY_SCHEMA,
        "scenarios": scenarios,
        "stale_writer_disposition": "INTERRUPTED",
        "stale_writer_may_be_inferred_completed": False,
        "single_writer_required": True,
        "heartbeat_required": True,
    }
    return {**base, "matrix_sha256": semantic_sha256(base)}


def _fsynced_write_unique(path: Path, payload: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags, 0o600)
    try:
        offset = 0
        while offset < len(payload):
            written = os.write(fd, payload[offset:])
            if written <= 0:
                raise F1ContractError("synthetic recovery write made no progress")
            offset += written
        os.fsync(fd)
    finally:
        os.close(fd)


def _read_synthetic_regular_no_follow(
    path: Path,
    *,
    maximum_bytes: int = 16 * 1024 * 1024,
) -> bytes:
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | os.O_NOFOLLOW
    )
    fd = os.open(path, flags)
    try:
        before = os.fstat(fd)
        if (
            not stat.S_ISREG(before.st_mode)
            or int(before.st_nlink) != 1
            or not 0 < int(before.st_size) <= maximum_bytes
        ):
            raise F1ContractError(
                "synthetic recovery read target is not an exact regular file"
            )
        chunks: list[bytes] = []
        remaining = int(before.st_size)
        while remaining:
            chunk = os.read(fd, min(1 << 20, remaining))
            if not chunk:
                raise F1ContractError("synthetic recovery regular read was short")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(fd, 1):
            raise F1ContractError("synthetic recovery regular file grew during read")
        after = os.fstat(fd)
    finally:
        os.close(fd)
    stable_fields = (
        "st_dev",
        "st_ino",
        "st_size",
        "st_mode",
        "st_nlink",
        "st_mtime_ns",
        "st_ctime_ns",
    )
    if any(
        getattr(before, field_name) != getattr(after, field_name)
        for field_name in stable_fields
    ):
        raise F1ContractError(
            "synthetic recovery regular file changed during read"
        )
    return b"".join(chunks)


def _synthetic_fd_mount_id(fd: int) -> int:
    info_fd = os.open(
        "/proc/self/fdinfo/%d" % fd,
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        rows = os.read(info_fd, 16_384).splitlines()
    finally:
        os.close(info_fd)
    values = [
        row.split(b":", 1)[1].strip()
        for row in rows
        if row.startswith(b"mnt_id:")
    ]
    if len(values) != 1 or not values[0].isdigit():
        raise F1ContractError("synthetic recovery mount identity unavailable")
    return int(values[0])


def _fsync_parent(path: Path) -> None:
    fd = os.open(
        path,
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _atomic_replace_new(path: Path, payload: bytes, tag: str) -> None:
    temporary = path.parent / f".{path.name}.{tag}.tmp"
    if temporary.exists() or temporary.is_symlink():
        raise F1ContractError("synthetic recovery temp namespace is polluted")
    _fsynced_write_unique(temporary, payload)
    os.replace(temporary, path)
    _fsync_parent(path.parent)


def _synthetic_atomic_recovery_expected_material(
    crash_point: str,
) -> dict[str, Any]:
    registered = {
        "BEFORE_CHECKPOINT_COMMIT",
        "AFTER_CHECKPOINT_BEFORE_MANIFEST",
        "DURING_EVAL",
        "BEFORE_STATUS_RENAME",
    }
    if crash_point not in registered:
        raise F1ContractError("unregistered synthetic recovery crash point")
    state_v0 = {
        "model": {"weight": 17},
        "optimizer": {"momentum": 23},
        "scheduler": {"completed_updates": 0},
        "scaler": {"scale": 1},
        "ema": {"weight": 17},
        "rng": {"python": 31, "numpy": 37, "torch_cpu": 41, "torch_cuda": 43},
        "sampler": {"query_cursor": 0},
        "candidate_snapshot": {"query_cursor": 0, "sha256": "a" * 64},
    }
    state_v1 = {
        **state_v0,
        "optimizer": {"momentum": 29},
        "scheduler": {"completed_updates": 1},
        "rng": {"python": 47, "numpy": 53, "torch_cpu": 59, "torch_cuda": 61},
        "sampler": {"query_cursor": 8},
        "candidate_snapshot": {"query_cursor": 8, "sha256": "b" * 64},
    }
    checkpoint_v0 = canonical_json_bytes(state_v0) + b"\n"
    checkpoint_v1 = canonical_json_bytes(state_v1) + b"\n"
    manifest_v0 = canonical_json_bytes(
        {
            "checkpoint_sha256": hashlib.sha256(checkpoint_v0).hexdigest(),
            "version": 0,
        }
    ) + b"\n"
    manifest_v1 = canonical_json_bytes(
        {
            "checkpoint_sha256": hashlib.sha256(checkpoint_v1).hexdigest(),
            "version": 1,
        }
    ) + b"\n"
    eval_id = "SYNTHETIC-EVAL-ONE-SHOT"
    chunk0 = canonical_json_bytes(
        {"eval_id": eval_id, "start": 0, "end": 4}
    ) + b"\n"
    chunk1 = canonical_json_bytes(
        {"eval_id": eval_id, "start": 4, "end": 8}
    ) + b"\n"
    status_bytes = canonical_json_bytes(
        {
            "status": "COMMITTED",
            "artifact_sha256": hashlib.sha256(checkpoint_v0).hexdigest(),
        }
    ) + b"\n"
    if crash_point == "BEFORE_CHECKPOINT_COMMIT":
        artifacts = {
            "checkpoint.json": checkpoint_v1,
            "checkpoint.manifest.json": manifest_v1,
        }
        recovery_action = "DISCARD_UNCOMMITTED_TEMP_REPLAY_UPDATE"
        terminal = checkpoint_v1
    elif crash_point == "AFTER_CHECKPOINT_BEFORE_MANIFEST":
        artifacts = {
            "checkpoint.json": checkpoint_v1,
            "checkpoint.manifest.json": manifest_v1,
            "checkpoint.orphan.quarantine.json": checkpoint_v1,
        }
        recovery_action = "QUARANTINE_ORPHAN_RESTORE_MANIFEST_CHECKPOINT"
        terminal = checkpoint_v1
    elif crash_point == "DURING_EVAL":
        artifacts = {
            "checkpoint.json": checkpoint_v0,
            "checkpoint.manifest.json": manifest_v0,
            "eval.chunk.0000.json": chunk0,
            "eval.chunk.0001.json": chunk1,
        }
        recovery_action = "SAME_EVAL_ID_CURSOR_RESUME"
        terminal = checkpoint_v0
    else:
        artifacts = {
            "checkpoint.json": checkpoint_v0,
            "checkpoint.manifest.json": manifest_v0,
            "status.json": status_bytes,
        }
        recovery_action = "IDEMPOTENT_STATUS_RENAME_POSTIMAGE_VERIFY"
        terminal = checkpoint_v0
    return {
        "checkpoint_v0": checkpoint_v0,
        "checkpoint_v1": checkpoint_v1,
        "manifest_v0": manifest_v0,
        "manifest_v1": manifest_v1,
        "eval_chunk0": chunk0,
        "eval_chunk1": chunk1,
        "status_bytes": status_bytes,
        "expected_artifact_bytes": artifacts,
        "recovery_action": recovery_action,
        "checkpoint_terminal_sha256": hashlib.sha256(terminal).hexdigest(),
        "complete_state_components_compared": [
            "model",
            "optimizer",
            "scheduler",
            "scaler",
            "ema",
            "rng",
            "sampler",
            "candidate_snapshot",
        ],
    }


def _synthetic_atomic_recovery_evidence_from_postimage(
    *,
    synthetic_root_handle_sha256: str,
    crash_point: str,
    actual_postimage: Mapping[str, Any],
) -> dict[str, Any]:
    material = _synthetic_atomic_recovery_expected_material(crash_point)
    if (
        not isinstance(synthetic_root_handle_sha256, str)
        or len(synthetic_root_handle_sha256) != 64
        or any(
            character not in "0123456789abcdef"
            for character in synthetic_root_handle_sha256
        )
        or not isinstance(actual_postimage, Mapping)
        or actual_postimage.get("synthetic_root_handle_sha256")
        != synthetic_root_handle_sha256
        or actual_postimage.get("crash_point") != crash_point
    ):
        raise F1ContractError("synthetic recovery evidence postimage binding drift")
    evidence = {
        "schema_version": "c28f_synthetic_atomic_recovery_evidence_v1",
        "status": "PASS",
        "synthetic_only": True,
        "synthetic_root_handle_sha256": synthetic_root_handle_sha256,
        "crash_point": crash_point,
        "recovery_action": material["recovery_action"],
        "checkpoint_terminal_sha256": material[
            "checkpoint_terminal_sha256"
        ],
        "bitwise_equal_to_uninterrupted_control": True,
        "complete_state_components_compared": material[
            "complete_state_components_compared"
        ],
        "model_forward_executed": False,
        "real_data_optimizer_updates": 0,
        "actual_postimage": dict(actual_postimage),
    }
    return {**evidence, "evidence_sha256": semantic_sha256(evidence)}


def execute_controller_owned_synthetic_atomic_recovery(
    execution_capability: Any,
) -> Mapping[str, Any]:
    """Run the reviewed synthetic body through one pathless controller surface.

    The controller owns the lease, directory descriptors, intent, operation
    plan, postimage reread, and durable completion.  This body can only consume
    the opaque one-shot surface and returns no path, descriptor, session, or
    callback.
    """

    exact_capability_fields = {
        "schema_version",
        "status",
        "goal_id",
        "attempt_id",
        "synthetic_root_handle_sha256",
        "crash_point",
        "child_root_name",
        "child_root_binding_sha256",
        "parent_descriptor_identity_sha256",
        "child_descriptor_identity_sha256",
        "execution_intent_sha256",
        "fixed_body_source_sha256",
        "executor_version",
        "writer_lease_binding_sha256",
        "owner_identity_sha256",
        "synthetic_only",
        "real_data_allowed",
        "path_export_allowed",
        "external_executor_allowed",
        "one_shot",
        "execution_binding_sha256",
    }
    binding_method = getattr(execution_capability, "binding", None)
    consume_method = getattr(
        execution_capability,
        "consume_preacquired_child_execution_once",
        None,
    )
    if (
        type(execution_capability).__name__
        != "A4G6ControllerExecutionCapability"
        or type(execution_capability).__module__
        != "blueprint_e2e_v2.c28f_v5.a4_control"
        or not callable(binding_method)
        or getattr(binding_method, "__self__", None)
        is not execution_capability
        or not callable(consume_method)
        or getattr(consume_method, "__self__", None)
        is not execution_capability
    ):
        raise F1ContractError(
            "exact controller-owned G6 execution capability required"
        )
    try:
        binding = binding_method()
    except Exception as error:
        raise F1ContractError(
            "G6 execution capability binding failed"
        ) from error
    crash_point = (
        binding.get("crash_point")
        if isinstance(binding, Mapping)
        else None
    )
    if (
        not isinstance(binding, Mapping)
        or set(binding) != exact_capability_fields
        or binding.get("schema_version")
        != "c28f_a4_g6_controller_execution_capability_v2"
        or binding.get("status") != "ACTIVE_CONTROLLER_FIXED_EXECUTION"
        or binding.get("executor_version")
        != "CONTROLLER_SINGLE_WRITER_FIXED_A4_F1_BODY_V2"
        or crash_point
        not in {
            "BEFORE_CHECKPOINT_COMMIT",
            "AFTER_CHECKPOINT_BEFORE_MANIFEST",
            "DURING_EVAL",
            "BEFORE_STATUS_RENAME",
        }
        or binding.get("synthetic_only") is not True
        or binding.get("real_data_allowed") is not False
        or binding.get("path_export_allowed") is not False
        or binding.get("external_executor_allowed") is not False
        or binding.get("one_shot") is not True
        or binding.get("execution_binding_sha256")
        != semantic_sha256(
            dict(binding),
            excluded_fields=("execution_binding_sha256",),
        )
    ):
        raise F1ContractError("G6 execution capability binding drift")
    try:
        session = consume_method()
    except Exception as error:
        raise F1ContractError(
            "G6 controller-owned execution capability consumption failed"
        ) from error
    if (
        type(session).__name__ != "_A4G6ControllerOwnedSyntheticSession"
        or type(session).__module__
        != "blueprint_e2e_v2.c28f_v5.a4_control"
    ):
        raise F1ContractError("G6 controller returned a non-execution surface")

    material = _synthetic_atomic_recovery_expected_material(crash_point)
    checkpoint_v0 = material["checkpoint_v0"]
    checkpoint_v1 = material["checkpoint_v1"]
    manifest_v0 = material["manifest_v0"]
    manifest_v1 = material["manifest_v1"]
    session.atomic_replace_new(
        "checkpoint.json", checkpoint_v0, "genesis"
    )
    session.atomic_replace_new(
        "checkpoint.manifest.json", manifest_v0, "genesis"
    )

    if crash_point == "BEFORE_CHECKPOINT_COMMIT":
        session.fsynced_write_unique(
            ".checkpoint.json.update.tmp", checkpoint_v1
        )
        if (
            session.read_regular("checkpoint.json") != checkpoint_v0
            or session.read_regular("checkpoint.manifest.json")
            != manifest_v0
        ):
            raise F1ContractError(
                "pre-commit crash changed the last green checkpoint"
            )
        session.unlink(".checkpoint.json.update.tmp")
        session.atomic_replace_new(
            "checkpoint.json", checkpoint_v1, "replay"
        )
        session.atomic_replace_new(
            "checkpoint.manifest.json", manifest_v1, "replay"
        )
        if (
            session.read_regular("checkpoint.json") != checkpoint_v1
            or session.read_regular("checkpoint.manifest.json")
            != manifest_v1
        ):
            raise F1ContractError(
                "replayed checkpoint differs from uninterrupted commit"
            )
    elif crash_point == "AFTER_CHECKPOINT_BEFORE_MANIFEST":
        session.atomic_replace_new(
            "checkpoint.json", checkpoint_v1, "update"
        )
        if session.read_regular("checkpoint.manifest.json") != manifest_v0:
            raise F1ContractError(
                "orphan checkpoint changed the immutable manifest"
            )
        session.replace(
            "checkpoint.json", "checkpoint.orphan.quarantine.json"
        )
        session.atomic_replace_new(
            "checkpoint.json", checkpoint_v0, "restore"
        )
        session.atomic_replace_new(
            "checkpoint.json", checkpoint_v1, "replay"
        )
        session.atomic_replace_new(
            "checkpoint.manifest.json", manifest_v1, "replay"
        )
        if (
            session.read_regular("checkpoint.json") != checkpoint_v1
            or session.read_regular("checkpoint.manifest.json")
            != manifest_v1
        ):
            raise F1ContractError(
                "orphan recovery differs from uninterrupted commit"
            )
    elif crash_point == "DURING_EVAL":
        chunk0 = material["eval_chunk0"]
        chunk1 = material["eval_chunk1"]
        session.fsynced_write_unique("eval.chunk.0000.json", chunk0)
        session.fsynced_write_unique(
            ".eval.chunk.0001.json.partial",
            chunk1[: max(1, len(chunk1) // 2)],
        )
        session.unlink(".eval.chunk.0001.json.partial")
        session.fsynced_write_unique("eval.chunk.0001.json", chunk1)
        if (
            session.read_regular("eval.chunk.0000.json")
            + session.read_regular("eval.chunk.0001.json")
            != chunk0 + chunk1
        ):
            raise F1ContractError(
                "eval cursor recovery differs from uninterrupted bytes"
            )
    else:
        status_bytes = material["status_bytes"]
        session.fsynced_write_unique(
            ".status.json.rename.tmp", status_bytes
        )
        session.replace(".status.json.rename.tmp", "status.json")
        if session.read_regular("status.json") != status_bytes:
            raise F1ContractError(
                "status rename recovery postimage mismatch"
            )

    expected_artifacts = material["expected_artifact_bytes"]
    terminal_names = tuple(sorted(expected_artifacts))
    if session.list_names() != terminal_names:
        raise F1ContractError("G6 terminal artifact namespace drift")
    for artifact_name in terminal_names:
        if (
            session.read_regular(artifact_name)
            != expected_artifacts[artifact_name]
        ):
            raise F1ContractError("G6 terminal artifact bytes drift")
    artifact_manifest = [
        {
            "name": artifact_name,
            "size_bytes": len(expected_artifacts[artifact_name]),
            "sha256": hashlib.sha256(
                expected_artifacts[artifact_name]
            ).hexdigest(),
        }
        for artifact_name in terminal_names
    ]
    result_base = {
        "schema_version": "c28f_g6_controller_fixed_body_result_v2",
        "status": "FIXED_BODY_TERMINAL_AWAITING_CONTROLLER_POSTIMAGE",
        "executor_version": (
            "CONTROLLER_SINGLE_WRITER_FIXED_A4_F1_BODY_V2"
        ),
        "crash_point": crash_point,
        "expected_artifact_manifest": artifact_manifest,
        "expected_artifact_manifest_sha256": hashlib.sha256(
            canonical_json_bytes(artifact_manifest)
        ).hexdigest(),
        "recovery_action": material["recovery_action"],
        "checkpoint_terminal_sha256": material[
            "checkpoint_terminal_sha256"
        ],
        "complete_state_components_compared": material[
            "complete_state_components_compared"
        ],
        "model_forward_executed": False,
        "real_data_optimizer_updates": 0,
        "path_returned": False,
        "external_executor_used": False,
    }
    return {
        **result_base,
        "result_sha256": semantic_sha256(result_base),
    }


def _acquire_synthetic_recovery_root(
    root_handle: Any,
    *,
    crash_point: str,
    unstarted_resume_capability: Any = None,
) -> tuple[Mapping[str, Any], Path]:
    """Hard-disabled pre-v2 path-export compatibility tombstone."""

    raise F1ContractError(
        "G6_EXTERNAL_CHILD_PATH_EXPORT_DISABLED; "
        "CONTROLLER_FIXED_EXECUTOR_V2_REQUIRED"
    )


def _execute_synthetic_atomic_recovery_harness(
    root_handle: Any,
    *,
    crash_point: str,
    unstarted_resume_capability: Any = None,
) -> dict[str, Any]:
    """Hard-disabled pre-v2 path-based executor tombstone."""

    raise F1ContractError(
        "G6_EXTERNAL_CHILD_PATH_EXPORT_DISABLED; "
        "CONTROLLER_FIXED_EXECUTOR_V2_REQUIRED"
    )


def run_synthetic_atomic_recovery_harness(
    root_handle: Any,
    *,
    crash_point: str,
) -> dict[str, Any]:
    """Hard-disabled pre-v2 external-path entry point."""

    raise F1ContractError(
        "G6_EXTERNAL_CHILD_PATH_EXPORT_DISABLED; "
        "CONTROLLER_FIXED_EXECUTOR_V2_REQUIRED"
    )


def resume_synthetic_atomic_recovery_harness(
    root_handle: Any,
    unstarted_resume_capability: Any,
    *,
    crash_point: str,
) -> dict[str, Any]:
    """Hard-disabled pre-v2 external-path resume entry point."""

    raise F1ContractError(
        "G6_EXTERNAL_PRECREATED_CHILD_PATH_EXPORT_DISABLED; "
        "CONTROLLER_FIXED_EXECUTOR_V2_REQUIRED"
    )


def _validate_rehydrated_synthetic_postimage_bytes(
    *,
    actual_postimage: Any,
    artifact_bytes_by_name: Any,
    artifact_bytes_sha256s: Any,
    synthetic_root_handle_sha256: str,
    crash_point: str,
    parent_descriptor_identity_sha256: str,
    child_root_binding_sha256: str,
) -> dict[str, Any]:
    material = _synthetic_atomic_recovery_expected_material(crash_point)
    expected_bytes = material["expected_artifact_bytes"]
    exact_top_fields = {
        "schema_version",
        "synthetic_root_handle_sha256",
        "crash_point",
        "root_identity",
        "artifact_count",
        "artifact_inventory",
        "artifact_inventory_sha256",
        "postimage_sha256",
    }
    exact_root_fields = {
        "canonical_path",
        "parent_canonical_path",
        "device",
        "inode",
        "nlink",
        "mode",
        "mtime_ns",
        "ctime_ns",
        "mount_id",
        "parent_descriptor_identity_sha256",
        "child_root_binding_sha256",
    }
    exact_inventory_fields = {
        "name",
        "size_bytes",
        "sha256",
        "device",
        "inode",
        "nlink",
        "mode",
        "mtime_ns",
        "ctime_ns",
        "mount_id",
    }
    if not isinstance(actual_postimage, Mapping):
        raise F1ContractError("rehydrated synthetic actual postimage is absent")
    postimage = dict(actual_postimage)
    root_identity = postimage.get("root_identity")
    inventory = postimage.get("artifact_inventory")
    if (
        set(postimage) != exact_top_fields
        or postimage.get("schema_version")
        != "c28f_g6_synthetic_actual_postimage_v2"
        or postimage.get("synthetic_root_handle_sha256")
        != synthetic_root_handle_sha256
        or postimage.get("crash_point") != crash_point
        or not isinstance(root_identity, Mapping)
        or set(root_identity) != exact_root_fields
        or not isinstance(root_identity.get("canonical_path"), str)
        or not isinstance(root_identity.get("parent_canonical_path"), str)
        or Path(root_identity["canonical_path"]).parent
        != Path(root_identity["parent_canonical_path"])
        or root_identity.get("parent_descriptor_identity_sha256")
        != parent_descriptor_identity_sha256
        or root_identity.get("child_root_binding_sha256")
        != child_root_binding_sha256
        or any(
            type(root_identity.get(field_name)) is not int
            or root_identity[field_name] <= 0
            for field_name in (
                "device",
                "inode",
                "nlink",
                "mtime_ns",
                "ctime_ns",
                "mount_id",
            )
        )
        or root_identity.get("nlink") < 2
        or root_identity.get("mode") != "0o700"
        or type(postimage.get("artifact_count")) is not int
        or postimage["artifact_count"] != len(expected_bytes)
        or not isinstance(inventory, list)
        or len(inventory) != len(expected_bytes)
        or [item.get("name") for item in inventory if isinstance(item, Mapping)]
        != sorted(expected_bytes)
        or any(
            not isinstance(item, Mapping)
            or set(item) != exact_inventory_fields
            or item.get("name") not in expected_bytes
            or type(item.get("size_bytes")) is not int
            or item["size_bytes"] <= 0
            or not isinstance(item.get("sha256"), str)
            or _SHA256_RE.fullmatch(item["sha256"]) is None
            or any(
                type(item.get(field_name)) is not int
                or item[field_name] <= 0
                for field_name in (
                    "device",
                    "inode",
                    "nlink",
                    "mtime_ns",
                    "ctime_ns",
                    "mount_id",
                )
            )
            or item.get("nlink") != 1
            or item.get("mode") != "0o600"
            or item.get("device") != root_identity["device"]
            or item.get("mount_id") != root_identity["mount_id"]
            for item in inventory
        )
        or postimage.get("artifact_inventory_sha256")
        != hashlib.sha256(canonical_json_bytes(inventory)).hexdigest()
        or postimage.get("postimage_sha256")
        != semantic_sha256(postimage, excluded_fields=("postimage_sha256",))
    ):
        raise F1ContractError("rehydrated synthetic actual postimage contract drift")
    if (
        not isinstance(artifact_bytes_by_name, Mapping)
        or set(artifact_bytes_by_name) != set(expected_bytes)
        or not isinstance(artifact_bytes_sha256s, Mapping)
        or set(artifact_bytes_sha256s) != set(expected_bytes)
    ):
        raise F1ContractError("rehydrated synthetic artifact byte namespace drift")
    inventory_by_name = {item["name"]: item for item in inventory}
    for name, expected_payload in expected_bytes.items():
        observed_payload = artifact_bytes_by_name.get(name)
        expected_sha256 = hashlib.sha256(expected_payload).hexdigest()
        if (
            type(observed_payload) is not bytes
            or observed_payload != expected_payload
            or artifact_bytes_sha256s.get(name) != expected_sha256
            or inventory_by_name[name]["size_bytes"] != len(expected_payload)
            or inventory_by_name[name]["sha256"] != expected_sha256
        ):
            raise F1ContractError(
                "rehydrated synthetic deterministic file bytes drift"
            )
    return postimage


def rehydrate_synthetic_atomic_recovery_evidence(
    root_handle: Any,
    rehydrate_capability: Any,
    *,
    crash_point: str,
) -> dict[str, Any]:
    """Rebuild evidence read-only from a controller-verified terminal postimage."""

    from .a4_control import (
        A4G6SyntheticPostimageRehydrateCapability,
        A4G6SyntheticRootHandle,
        validate_g6_synthetic_postimage_rehydrate_capability,
        validate_g6_synthetic_root_handle,
    )

    _synthetic_atomic_recovery_expected_material(crash_point)
    if type(root_handle) is not A4G6SyntheticRootHandle:
        raise F1ContractError(
            "exact controller-issued A4G6SyntheticRootHandle required"
        )
    if type(rehydrate_capability) is not (
        A4G6SyntheticPostimageRehydrateCapability
    ):
        raise F1ContractError(
            "exact controller-issued postimage rehydrate capability required"
        )
    try:
        root_binding = validate_g6_synthetic_root_handle(root_handle)
        rehydrate_binding = (
            validate_g6_synthetic_postimage_rehydrate_capability(
                rehydrate_capability
            )
        )
    except Exception as error:
        raise F1ContractError(
            "G6 synthetic postimage rehydrate authority validation failed"
        ) from error
    exact_binding_fields = {
        "schema_version",
        "status",
        "goal_id",
        "attempt_id",
        "synthetic_root_handle_sha256",
        "crash_point",
        "child_root_name",
        "child_root_binding_sha256",
        "consumption_receipt_sha256",
        "parent_descriptor_identity_sha256",
        "actual_postimage",
        "actual_postimage_sha256",
        "synthetic_only",
        "real_data_allowed",
        "read_only",
        "one_shot_per_capability",
        "rehydrate_binding_sha256",
    }
    bound_postimage = (
        rehydrate_binding.get("actual_postimage")
        if isinstance(rehydrate_binding, Mapping)
        else None
    )
    if (
        not isinstance(root_binding, Mapping)
        or not isinstance(rehydrate_binding, Mapping)
        or set(rehydrate_binding) != exact_binding_fields
        or rehydrate_binding.get("schema_version")
        != "c28f_a4_g6_synthetic_postimage_rehydrate_capability_v1"
        or rehydrate_binding.get("status")
        != "CONTROLLER_VERIFIED_SYNTHETIC_POSTIMAGE_REHYDRATE"
        or rehydrate_binding.get("goal_id") != root_binding.get("goal_id")
        or rehydrate_binding.get("attempt_id")
        != root_binding.get("attempt_id")
        or rehydrate_binding.get("synthetic_root_handle_sha256")
        != root_handle.capability_sha256
        or rehydrate_binding.get("crash_point") != crash_point
        or rehydrate_binding.get("child_root_name")
        != root_binding["child_root_names"].get(crash_point)
        or rehydrate_binding.get("parent_descriptor_identity_sha256")
        != root_binding.get("parent_descriptor_identity_sha256")
        or not isinstance(bound_postimage, Mapping)
        or rehydrate_binding.get("actual_postimage_sha256")
        != bound_postimage.get("postimage_sha256")
        or rehydrate_binding.get("synthetic_only") is not True
        or rehydrate_binding.get("real_data_allowed") is not False
        or rehydrate_binding.get("read_only") is not True
        or rehydrate_binding.get("one_shot_per_capability") is not True
        or rehydrate_binding.get("rehydrate_binding_sha256")
        != semantic_sha256(
            rehydrate_binding,
            excluded_fields=("rehydrate_binding_sha256",),
        )
    ):
        raise F1ContractError("G6 synthetic postimage rehydrate binding drift")
    try:
        access = rehydrate_capability.consume_verified_postimage_once()
    except Exception as error:
        raise F1ContractError(
            "G6 synthetic postimage rehydrate one-shot access failed"
        ) from error
    exact_access_fields = {
        "schema_version",
        "status",
        "synthetic_root_handle_sha256",
        "crash_point",
        "child_root_binding_sha256",
        "consumption_receipt_sha256",
        "actual_postimage",
        "artifact_bytes_by_name",
        "artifact_bytes_sha256s",
        "access_receipt_sha256",
    }
    if not isinstance(access, Mapping):
        raise F1ContractError("G6 synthetic postimage rehydrate access is absent")
    access_without_bytes = {
        key: value
        for key, value in access.items()
        if key not in {"artifact_bytes_by_name", "access_receipt_sha256"}
    }
    if (
        set(access) != exact_access_fields
        or access.get("schema_version")
        != "c28f_a4_g6_synthetic_postimage_rehydrate_access_v1"
        or access.get("status")
        != "CONTROLLER_READONLY_SYNTHETIC_POSTIMAGE_REHYDRATED"
        or access.get("synthetic_root_handle_sha256")
        != root_handle.capability_sha256
        or access.get("crash_point") != crash_point
        or access.get("child_root_binding_sha256")
        != rehydrate_binding["child_root_binding_sha256"]
        or access.get("consumption_receipt_sha256")
        != rehydrate_binding["consumption_receipt_sha256"]
        or access.get("actual_postimage")
        != rehydrate_binding["actual_postimage"]
        or access.get("access_receipt_sha256")
        != semantic_sha256(access_without_bytes)
    ):
        raise F1ContractError("G6 synthetic postimage rehydrate access drift")
    postimage = _validate_rehydrated_synthetic_postimage_bytes(
        actual_postimage=access["actual_postimage"],
        artifact_bytes_by_name=access["artifact_bytes_by_name"],
        artifact_bytes_sha256s=access["artifact_bytes_sha256s"],
        synthetic_root_handle_sha256=root_handle.capability_sha256,
        crash_point=crash_point,
        parent_descriptor_identity_sha256=root_binding[
            "parent_descriptor_identity_sha256"
        ],
        child_root_binding_sha256=rehydrate_binding[
            "child_root_binding_sha256"
        ],
    )
    return _synthetic_atomic_recovery_evidence_from_postimage(
        synthetic_root_handle_sha256=root_handle.capability_sha256,
        crash_point=crash_point,
        actual_postimage=postimage,
    )


@dataclass(frozen=True)
class C28FSupersetModel:
    """No-training namespace/init/optimizer-membership contract for F1.

    This intentionally stores names and provenance only; it is not a torch
    module and cannot perform a forward or create an optimizer in F1.
    """

    parameter_names: tuple[str, ...]
    base_initialization_sha256: str
    optimizer_parameter_names: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.parameter_names or tuple(sorted(set(self.parameter_names))) != self.parameter_names:
            raise F1ContractError("superset parameter names must be unique canonical order")
        required_prefixes = (
            "base.query_encoder.",
            "base.video_encoder.",
            "base.retriever.",
            "base.localizer.",
            "superset.r2m.",
            "superset.m2r.",
            "superset.mediator.",
            "superset.credit_target.",
        )
        for prefix in required_prefixes:
            if not any(name.startswith(prefix) for name in self.parameter_names):
                raise F1ContractError(f"superset namespace is missing: {prefix}")
        if any(
            not isinstance(name, str)
            or not name
            or name != name.strip()
            or ".." in name
            for name in self.parameter_names
        ):
            raise F1ContractError("superset parameter name is invalid")
        _exact_sha256(self.base_initialization_sha256, "base initialization SHA")
        if self.optimizer_parameter_names != ():
            raise F1ContractError("F1 scaffold must not create optimizer membership")

    def audit(self) -> dict[str, Any]:
        base_names = tuple(name for name in self.parameter_names if name.startswith("base."))
        superset_names = tuple(
            name for name in self.parameter_names if name.startswith("superset.")
        )
        base = {
            "schema_version": SUPERSET_SCHEMA,
            "class_name": type(self).__name__,
            "parameter_names": list(self.parameter_names),
            "parameter_name_lines_sha256": hashlib.sha256(
                "".join(f"{name}\n" for name in self.parameter_names).encode("utf-8")
            ).hexdigest(),
            "base_parameter_count": len(base_names),
            "superset_schema_parameter_count": len(superset_names),
            "base_initialization_sha256": self.base_initialization_sha256,
            "superset_initialized_in_f1": False,
            "optimizer_parameter_names": [],
            "optimizer_created_in_f1": False,
            "forward_implemented_in_f1": False,
        }
        return {**base, "audit_sha256": semantic_sha256(base)}


def superset_scaffold_contract() -> dict[str, Any]:
    state_dict_namespaces = [
        "base.query_encoder.*",
        "base.video_encoder.*",
        "base.retriever.*",
        "base.localizer.*",
        "superset.r2m.*",
        "superset.m2r.*",
        "superset.mediator.*",
        "superset.credit_target.*",
    ]
    canonical_parameter_names = tuple(
        sorted(
            (
                "base.localizer.schema_anchor",
                "base.query_encoder.schema_anchor",
                "base.retriever.schema_anchor",
                "base.video_encoder.schema_anchor",
                "superset.credit_target.schema_anchor",
                "superset.m2r.schema_anchor",
                "superset.mediator.schema_anchor",
                "superset.r2m.schema_anchor",
            )
        )
    )
    scaffold = C28FSupersetModel(
        parameter_names=canonical_parameter_names,
        base_initialization_sha256=FROZEN_C28E_CHECKPOINT_SHA256,
    )
    base = {
        "schema_version": SUPERSET_SCHEMA,
        "single_class_schema": "C28FSupersetModel",
        "class_implemented": True,
        "class_contract_audit": scaffold.audit(),
        "state_dict_namespaces": state_dict_namespaces,
        "active_mask": {
            "base": True,
            "r2m": False,
            "m2r": False,
            "mediator": False,
            "credit_target": False,
        },
        "initialization_mapping": {
            "base.*": "frozen_c28e_checkpoint_exact",
            "superset.*": "schema_only_not_initialized_in_f1",
        },
        "optimizer_membership": {
            "base": "NOT_CREATED_IN_F1",
            "superset": "NOT_CREATED_IN_F1",
        },
        "f4_candidate_global_core_implemented": False,
        "f5_200x64_backward_run": False,
        "real_data_optimizer_updates": 0,
    }
    return {**base, "scaffold_sha256": semantic_sha256(base)}


def shape_capacity_contract() -> dict[str, Any]:
    dimensions = {
        "broad_videos": 1_000,
        "late_videos": 200,
        "proposals_per_video": 64,
        "temporal_cells": 64,
        "hidden_dim": 384,
        "query_batch": 32,
    }
    index_elements = dimensions["query_batch"] * dimensions["late_videos"]
    proposal_mask_elements = index_elements * dimensions["proposals_per_video"]
    hidden_activation_elements = proposal_mask_elements * dimensions["hidden_dim"]
    if proposal_mask_elements != 409_600:
        raise F1ContractError("shape capacity arithmetic drift")
    if hidden_activation_elements != 157_286_400:
        raise F1ContractError("hidden activation capacity arithmetic drift")
    return {
        "dimensions": dimensions,
        "index_elements": index_elements,
        "proposal_mask_elements": proposal_mask_elements,
        "joint_logit_elements": proposal_mask_elements,
        "hidden_activation_elements": hidden_activation_elements,
        "float32_hidden_activation_bytes_upper_bound": (
            hidden_activation_elements * 4
        ),
        "boolean_proposal_mask_bytes_upper_bound": proposal_mask_elements,
        "allocation_performed": False,
        "backward_performed": False,
        "contract_only": True,
    }


def cost_measurement_protocol() -> dict[str, Any]:
    ratio_ceilings = {
        "B_DECOMP_VS_A": {
            "train_step_time": 1.20,
            "peak_memory": 1.15,
            "inference_latency": 1.15,
        },
        "B_GLOBAL_VS_B_DECOMP": {
            "train_step_time": 1.10,
            "peak_memory": 1.05,
            "inference_latency": 1.05,
        },
        "C_OR_D_VS_DIRECT_PARENT": {
            "train_step_time": 1.15,
            "peak_memory": 1.10,
            "inference_latency": 1.10,
        },
        "D_AB_VS_ORIGINAL_M2R_PARENT_TOTAL": {
            "train_step_time": 1.15,
            "peak_memory": 1.10,
            "inference_latency": 1.10,
        },
        "F_OR_F_B_VS_E_OR_E_B": {
            "train_step_time": 1.35,
            "peak_memory": 1.20,
            "inference_latency": 1.02,
        },
    }
    base = {
        "schema_version": COST_SCHEMA,
        "same_hardware_required": True,
        "workload_semantics": "200 videos x 64 proposals, fixed 100 optimizer updates",
        "warmup_updates": 10,
        "measured_updates": 100,
        "timing_unit": "milliseconds_per_committed_optimizer_update",
        "throughput_unit": "queries_per_second",
        "memory_unit": "bytes_peak_cuda_allocated",
        "statistics": ["median", "p95", "max"],
        "environment_fields": [
            "gpu_uuid",
            "gpu_name",
            "driver",
            "cuda_runtime",
            "torch_version",
            "code_sha256",
            "config_sha256",
        ],
        "ratio_ceilings": ratio_ceilings,
        "reference_measurements": "DEFERRED_TO_AUTHORIZED_F5_WITHOUT_CHANGING_CEILINGS",
        "measurement_executed_in_f1": False,
        "f6_executed": False,
    }
    return {**base, "protocol_sha256": semantic_sha256(base)}


def deterministic_synthetic_resume_trace(
    *, total_updates: int, crash_after_update: int
) -> dict[str, Any]:
    total = _exact_int(total_updates, "total_updates", minimum=1)
    crash = _exact_int(crash_after_update, "crash_after_update")
    if total > 64 or crash > total:
        raise F1ContractError("synthetic update budget/cursor violation")

    def step(value: int, update: int) -> int:
        payload = f"{value}:{update}:c28f-synthetic".encode("ascii")
        return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")

    uninterrupted = 0xC28F
    checkpoints: dict[int, int] = {0: uninterrupted}
    for update in range(1, total + 1):
        uninterrupted = step(uninterrupted, update)
        checkpoints[update] = uninterrupted
    resumed = checkpoints[crash]
    for update in range(crash + 1, total + 1):
        resumed = step(resumed, update)
    if resumed != uninterrupted:
        raise F1ContractError("synthetic resume trace diverged")
    return {
        "total_updates": total,
        "crash_after_update": crash,
        "uninterrupted_final": uninterrupted,
        "resumed_final": resumed,
        "bitwise_equal": True,
        "optimizer_updates_consumed": total,
    }


def deterministic_synthetic_fixture_smoke(*, query_count: int) -> dict[str, Any]:
    """Exercise the F1 scheduler/schema/recovery contract on mock IDs only.

    This is deliberately a small, pure-Python protocol smoke.  It neither
    imports the model stack nor claims corpus-scale, numerical, or performance
    acceptance.  The two authorized fixture sizes consume 8 and 16 synthetic
    update transitions respectively, leaving the whole F1 report at the frozen
    cap of 64 transitions.
    """

    count = _exact_int(query_count, "query_count", minimum=1)
    if count not in {8, 128}:
        raise F1ContractError("synthetic smoke permits exactly 8Q or 128Q")
    planned_updates = 8 if count == 8 else 16
    batch_size = count // planned_updates
    if batch_size * planned_updates != count:
        raise F1ContractError("synthetic fixture does not divide into frozen updates")

    rows: list[dict[str, Any]] = []
    for query_index in range(count):
        cells = (64, 96, 128)[query_index % 3]
        duration = float(8 + (query_index % 17))
        grid = TemporalGridSpec(duration, cells)
        start_cell = query_index % max(1, cells - 2)
        end_cell = min(cells, start_cell + 1 + (query_index % 2))
        start_sec, end_sec = grid.cell_span_to_seconds(start_cell, end_cell)
        encoded = grid.seconds_to_cell_span(start_sec, end_sec)
        if encoded != (start_cell, end_cell):
            raise F1ContractError("synthetic temporal round-trip diverged")
        rows.append(
            {
                "query_id": f"SYNTHETIC-Q{query_index:04d}",
                "mock_video_id": f"SYNTHETIC-V{query_index % 7:03d}",
                "duration_sec": duration,
                "temporal_cells": cells,
                "cell_span_half_open": [start_cell, end_cell],
                "score_schema": {
                    "pooled": "finite_float",
                    "late": "finite_float",
                    "joint": "finite_float",
                },
            }
        )

    def commit_update(state: str, update_index: int, batch_rows: Sequence[Mapping[str, Any]]) -> str:
        payload = canonical_json_bytes(
            {
                "domain": "C28F_F1_SYNTHETIC_ONLY",
                "prior_state": state,
                "committed_update": update_index,
                "batch_query_ids": [row["query_id"] for row in batch_rows],
            }
        )
        return hashlib.sha256(payload).hexdigest()

    initial_state = hashlib.sha256(
        f"C28F_SYNTHETIC_{count}Q".encode("ascii")
    ).hexdigest()
    uninterrupted = initial_state
    checkpoints: dict[int, str] = {0: initial_state}
    schedule_trace: list[dict[str, Any]] = []
    for update_index in range(1, planned_updates + 1):
        start = (update_index - 1) * batch_size
        uninterrupted = commit_update(
            uninterrupted, update_index, rows[start : start + batch_size]
        )
        checkpoints[update_index] = uninterrupted
        schedule_trace.append(schedule_phase(update_index, planned_updates))

    crash_after = planned_updates // 2
    resumed = checkpoints[crash_after]
    for update_index in range(crash_after + 1, planned_updates + 1):
        start = (update_index - 1) * batch_size
        resumed = commit_update(
            resumed, update_index, rows[start : start + batch_size]
        )
    if resumed != uninterrupted:
        raise F1ContractError("synthetic fixture recovery is not bitwise deterministic")
    if schedule_trace[-1]["phase"] != "P6":
        raise F1ContractError("synthetic scheduler did not reach its terminal phase")

    row_schema = {
        "required_fields": sorted(rows[0]),
        "row_count": len(rows),
        "query_ids_unique": len({row["query_id"] for row in rows}) == count,
        "all_temporal_round_trips_exact": True,
    }
    base = {
        "schema_version": SYNTHETIC_SMOKE_SCHEMA,
        "status": "PASS",
        "synthetic_only": True,
        "performance_selection_use": False,
        "corpus_scale_acceptance_claimed": False,
        "real_checkpoint_loaded": False,
        "model_forward_executed": False,
        "query_count": count,
        "mock_corpus_video_count": 7,
        "batch_size": batch_size,
        "planned_optimizer_updates": planned_updates,
        "synthetic_optimizer_updates_consumed": planned_updates,
        "crash_after_update": crash_after,
        "uninterrupted_final_sha256": uninterrupted,
        "resumed_final_sha256": resumed,
        "bitwise_resume_equal": resumed == uninterrupted,
        "schedule_trace": schedule_trace,
        "row_schema": row_schema,
        "rows_sha256": hashlib.sha256(canonical_json_bytes(rows)).hexdigest(),
    }
    return {**base, "smoke_sha256": semantic_sha256(base)}


def build_f1_protocol_report() -> dict[str, Any]:
    schedule_boundaries = [
        schedule_phase(completed, 100) for completed in (0, 10, 25, 40, 60, 75, 100)
    ]
    formal_schedule_boundaries = [
        schedule_phase(completed, FROZEN_U_FORMAL_UPDATES)
        for completed in (0, 1_736, 4_340, 6_944, 10_416, 13_020, 17_360)
    ]
    resume_traces = [
        deterministic_synthetic_resume_trace(total_updates=8, crash_after_update=point)
        for point in (0, 1, 4, 7, 8)
    ]
    synthetic_smokes = {
        "8Q": deterministic_synthetic_fixture_smoke(query_count=8),
        "128Q": deterministic_synthetic_fixture_smoke(query_count=128),
    }
    base = {
        "schema_version": F1_PROTOCOL_SCHEMA,
        "status": "SYNTHETIC_PROTOCOL_EVIDENCE_AWAITING_CONTROL_LEDGER",
        "g6_commit_eligible": False,
        "ledger_derived_counts": False,
        "schedule_boundaries": schedule_boundaries,
        "formal_17360_schedule_boundaries": formal_schedule_boundaries,
        "temporal_oracles": temporal_oracles(),
        "checkpoint_contract": checkpoint_reference_contract(),
        "recovery_matrix": recovery_matrix(),
        "atomic_crash_evidence": {
            "status": "AWAITING_CONTROLLED_SYNTHETIC_EXECUTION",
            "required_crash_points": [
                "BEFORE_CHECKPOINT_COMMIT",
                "AFTER_CHECKPOINT_BEFORE_MANIFEST",
                "DURING_EVAL",
                "BEFORE_STATUS_RENAME",
            ],
            "committed_evidence_sha256s": None,
        },
        "superset_scaffold": superset_scaffold_contract(),
        "shape_capacity": shape_capacity_contract(),
        "cost_measurement_protocol": cost_measurement_protocol(),
        "synthetic_resume_traces": resume_traces,
        "synthetic_fixture_smokes": synthetic_smokes,
        "synthetic_8q_status": synthetic_smokes["8Q"]["status"],
        "synthetic_128q_status": synthetic_smokes["128Q"]["status"],
        "synthetic_optimizer_updates": sum(
            int(trace["optimizer_updates_consumed"]) for trace in resume_traces
        )
        + sum(
            int(smoke["synthetic_optimizer_updates_consumed"])
            for smoke in synthetic_smokes.values()
        ),
        "real_data_optimizer_updates": 0,
        "extra_model_forward_splits": 0,
        "applicability": {
            "F0_F1": "APPLICABLE",
            "F2_F5_candidate_global_loss_gradient_pressure": "NOT_APPLICABLE_SCHEMA_ONLY",
            "real_data_training": "NOT_AUTHORIZED",
        },
    }
    if base["synthetic_optimizer_updates"] > 64:
        raise F1ContractError("synthetic fixture update cap exceeded")
    return {**base, "report_sha256": semantic_sha256(base)}


def canonical_f1_artifact_bytes(value: Mapping[str, Any]) -> bytes:
    return canonical_json_bytes(dict(value)) + b"\n"
