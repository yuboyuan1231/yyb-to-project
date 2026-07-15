from __future__ import annotations

import hashlib
import inspect
import json
import math
import unittest
from unittest import mock

from blueprint_e2e_v2.c28f_v5.canonical import (
    canonical_json_bytes,
    semantic_sha256,
)

from blueprint_e2e_v2.c28f_v5.a4_f1 import (
    C28FSupersetModel,
    F1ContractError,
    TemporalGridSpec,
    TemporalTensorContract,
    assert_microstep_invariance,
    build_f1_protocol_report,
    cost_measurement_protocol,
    deterministic_synthetic_fixture_smoke,
    execute_controller_owned_synthetic_atomic_recovery,
    _acquire_synthetic_recovery_root,
    _execute_synthetic_atomic_recovery_harness,
    _read_synthetic_regular_no_follow,
    _synthetic_atomic_recovery_expected_material,
    _validate_rehydrated_synthetic_postimage_bytes,
    rehydrate_synthetic_atomic_recovery_evidence,
    resume_synthetic_atomic_recovery_harness,
    run_synthetic_atomic_recovery_harness,
    schedule_phase,
    superset_scaffold_contract,
    validate_checkpoint_payload,
)
from blueprint_e2e_v2.c28f_v5.a4_finalize import (
    FinalizationError,
    build_final_decision,
    close_authority_ledger,
)
from blueprint_e2e_v2.c28f_v5 import a4_roles as roles
from blueprint_e2e_v2.c28f_v5.a4_roles import (
    CORE_ROLE,
    FORMAL_POLICY,
    MINIMUM_EFFECTS_RATIO,
    POWER_OUTCOME_KEYS,
    U_FORMAL_UPDATES,
    RoleContractError,
    RoleRecord,
    _assert_power_prefix_expansion,
    _normalize_power_replay_rows,
    assign_video_clusters,
    build_g4_role_policy_candidates,
    build_cluster_power_audit,
    build_formal_data_policy,
    build_role_lock,
    build_role_manifests,
    frozen_power_evaluator_contract,
    validate_g4_atomic_candidate,
)


SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
SHA_D = "d" * 64


class _RecordingG6Session:
    def __init__(self) -> None:
        self.files: dict[str, bytes] = {}
        self.operations: list[tuple] = []

    def atomic_replace_new(
        self, name: str, payload: bytes, tag: str
    ) -> None:
        self.operations.append(("ATOMIC_REPLACE_NEW", name, tag))
        self.files[name] = payload

    def fsynced_write_unique(self, name: str, payload: bytes) -> None:
        if name in self.files:
            raise AssertionError("test session unique-write collision")
        self.operations.append(("FSYNCED_WRITE_UNIQUE", name))
        self.files[name] = payload

    def read_regular(self, name: str) -> bytes:
        self.operations.append(("READ_REGULAR", name))
        return self.files[name]

    def unlink(self, name: str) -> None:
        self.operations.append(("UNLINK", name))
        del self.files[name]

    def replace(self, source_name: str, target_name: str) -> None:
        self.operations.append(("REPLACE", source_name, target_name))
        self.files[target_name] = self.files.pop(source_name)

    def list_names(self) -> tuple[str, ...]:
        self.operations.append(("LIST_NAMES",))
        return tuple(sorted(self.files))


_RecordingG6Session.__name__ = "_A4G6ControllerOwnedSyntheticSession"
_RecordingG6Session.__module__ = "blueprint_e2e_v2.c28f_v5.a4_control"


class _RecordingG6Capability:
    def __init__(self, crash_point: str) -> None:
        base = {
            "schema_version": (
                "c28f_a4_g6_controller_execution_capability_v2"
            ),
            "status": "ACTIVE_CONTROLLER_FIXED_EXECUTION",
            "goal_id": "C28F-CMGC-V5-F0-F1",
            "attempt_id": "G0-A4-" + "1" * 32,
            "synthetic_root_handle_sha256": SHA_A,
            "crash_point": crash_point,
            "child_root_name": "scenario-fixed",
            "child_root_binding_sha256": SHA_B,
            "parent_descriptor_identity_sha256": SHA_C,
            "child_descriptor_identity_sha256": SHA_D,
            "execution_intent_sha256": "e" * 64,
            "fixed_body_source_sha256": "f" * 64,
            "executor_version": (
                "CONTROLLER_SINGLE_WRITER_FIXED_A4_F1_BODY_V2"
            ),
            "writer_lease_binding_sha256": "1" * 64,
            "owner_identity_sha256": "2" * 64,
            "synthetic_only": True,
            "real_data_allowed": False,
            "path_export_allowed": False,
            "external_executor_allowed": False,
            "one_shot": True,
        }
        self._binding = {
            **base,
            "execution_binding_sha256": semantic_sha256(base),
        }
        self.session = _RecordingG6Session()
        self.consumed = False

    def binding(self) -> dict:
        return json.loads(canonical_json_bytes(self._binding).decode("utf-8"))

    def consume_preacquired_child_execution_once(
        self,
    ) -> _RecordingG6Session:
        if self.consumed:
            raise RuntimeError("test capability already consumed")
        self.consumed = True
        return self.session


_RecordingG6Capability.__name__ = "A4G6ControllerExecutionCapability"
_RecordingG6Capability.__module__ = "blueprint_e2e_v2.c28f_v5.a4_control"


def _synthetic_rehydrate_fixture(
    crash_point: str = "DURING_EVAL",
) -> tuple[dict, dict[str, bytes], dict[str, str]]:
    material = _synthetic_atomic_recovery_expected_material(crash_point)
    artifact_bytes = dict(material["expected_artifact_bytes"])
    artifact_sha256s = {
        name: hashlib.sha256(payload).hexdigest()
        for name, payload in artifact_bytes.items()
    }
    inventory = [
        {
            "name": name,
            "size_bytes": len(artifact_bytes[name]),
            "sha256": artifact_sha256s[name],
            "device": 101,
            "inode": 1_000 + index,
            "nlink": 1,
            "mode": "0o600",
            "mtime_ns": 10_000 + index,
            "ctime_ns": 20_000 + index,
            "mount_id": 303,
        }
        for index, name in enumerate(sorted(artifact_bytes))
    ]
    base = {
        "schema_version": "c28f_g6_synthetic_actual_postimage_v2",
        "synthetic_root_handle_sha256": SHA_A,
        "crash_point": crash_point,
        "root_identity": {
            "canonical_path": "/synthetic/g6/scenario",
            "parent_canonical_path": "/synthetic/g6",
            "device": 101,
            "inode": 999,
            "nlink": 2,
            "mode": "0o700",
            "mtime_ns": 30_000,
            "ctime_ns": 40_000,
            "mount_id": 303,
            "parent_descriptor_identity_sha256": SHA_B,
            "child_root_binding_sha256": SHA_C,
        },
        "artifact_count": len(inventory),
        "artifact_inventory": inventory,
        "artifact_inventory_sha256": hashlib.sha256(
            canonical_json_bytes(inventory)
        ).hexdigest(),
    }
    return (
        {**base, "postimage_sha256": semantic_sha256(base)},
        artifact_bytes,
        artifact_sha256s,
    )


def _synthetic_v3_power_result(
    *,
    role: str,
    query_count: int,
    cluster_count: int,
    reference_population: dict,
) -> dict:
    quotient, remainder = divmod(query_count, cluster_count)
    maximum_k = int(
        math.ceil(roles.POWER_MAX_INJECTED_DELTA * query_count)
    )
    thresholds = [1] * roles.POWER_OUTER_EXPERIMENTS
    eligible_count = max(1, query_count // 2)
    ci_rows = [
        {
            "outer_index": outer_index,
            "eligible_slot_count": eligible_count,
            "detection_threshold_k": 1,
            "ci_lower_at_threshold": 1.0 / float(query_count),
            "ci_upper_at_threshold": 1.0 / float(query_count),
            "ci_lower_at_k_minus_1": 0.0,
            "ci_upper_at_k_minus_1": 0.0,
            "maximum_evaluated_k": min(maximum_k, eligible_count),
            "ci_lower_at_maximum_k": None,
            "ci_upper_at_maximum_k": None,
        }
        for outer_index in range(roles.POWER_OUTER_EXPERIMENTS)
    ]
    effect_results = {}
    distribution = reference_population["reference_outcome_distribution"]
    for key, threshold in dict(MINIMUM_EFFECTS_RATIO).items():
        rule = dict(roles.EFFECT_OUTCOME_REGISTRY[key])
        reference_eligible = distribution[rule["binary_outcome"]][
            "zero_count" if rule["eligible_baseline_value"] == 0 else "one_count"
        ]
        effect_results[key] = {
            "binary_outcome": rule["binary_outcome"],
            "sensitivity_rule": rule["sensitivity_rule"],
            "eligible_baseline_value": rule["eligible_baseline_value"],
            "injected_flip": rule["injected_flip"],
            "minimum_effect_ratio": threshold,
            "reference_eligible_query_count": reference_eligible,
            "outer_eligible_query_count_min": eligible_count,
            "outer_eligible_query_count_max": eligible_count,
            "outer_eligible_query_count_mean": float(eligible_count),
            "maximum_supported_k": maximum_k,
            "outer_detection_thresholds": list(thresholds),
            "outer_detection_thresholds_sha256": hashlib.sha256(
                canonical_json_bytes(thresholds)
            ).hexdigest(),
            "outer_threshold_ci_sha256": hashlib.sha256(
                canonical_json_bytes(ci_rows)
            ).hexdigest(),
            "outer_threshold_ci": [dict(row) for row in ci_rows],
            "mde_k": 1,
            "mde_ratio": 1.0 / float(query_count),
            "power_at_mde": 1.0,
            "power_at_k_minus_1": 0.0,
            "detected_outer_count_at_mde": roles.POWER_OUTER_EXPERIMENTS,
            "detected_outer_count_at_k_minus_1": 0,
            "detected_outer_count_at_maximum_supported_k": (
                roles.POWER_OUTER_EXPERIMENTS
            ),
            "power_at_maximum_supported_k": 1.0,
            "paired_ci_detection_rule": (
                "49_INNER_CLUSTER_POSITION_BOOTSTRAPS_NEAREST_RANK_2_48_"
                "SIGN_NORMALIZED_LOWER_GT_ZERO"
            ),
            "status": "PASS",
        }
    return {
        "role": role,
        "query_count": query_count,
        "gt_video_cluster_count": cluster_count,
        "candidate_role_query_count": query_count,
        "candidate_role_gt_video_cluster_count": cluster_count,
        "discrete_query_resolution": 1.0 / float(query_count),
        "power_input_semantics": (
            "OUTER_LOCAL_PAIRED_REFERENCE_BOOTSTRAP_PLUS_"
            "CANDIDATE_ROLE_COUNTS_ONLY"
        ),
        "power_algorithm": roles.POWER_ALGORITHM,
        "power_counter_domain": roles.POWER_COUNTER_DOMAIN,
        "reference_population_sha256": reference_population[
            "reference_population_sha256"
        ],
        "reference_query_count": reference_population["power_row_count"],
        "reference_gt_video_cluster_count": reference_population[
            "reference_gt_video_cluster_count"
        ],
        "train_fit_desc_id_join_performed": False,
        "train_fit_video_id_join_performed": False,
        "historical_performance_selects_membership": False,
        "candidate_cluster_size_distribution_consumed": False,
        "equal_slot_projection": {
            "assumption": (
                "EQUAL_q_OR_q_PLUS_1_SLOT_PROJECTION_NOT_OBSERVED_"
                "CANDIDATE_CLUSTER_SIZE_DISTRIBUTION"
            ),
            "quotient_q": quotient,
            "remainder_r": remainder,
            "q_plus_1_position_prefix_length": remainder,
            "projected_slot_count": query_count,
            "cluster_position_count": cluster_count,
        },
        "outer_cluster_position_sampling": (
            "C_REFERENCE_CLUSTER_DRAWS_WITH_REPLACEMENT"
        ),
        "outer_cluster_position_sample_sha256": SHA_A,
        "inner_cluster_position_resampling": (
            "C_OUTER_POSITION_DRAWS_WITH_REPLACEMENT_DUPLICATES_"
            "PRESERVED_ALL_POSITION_SLOTS"
        ),
        "inner_cluster_position_resample_sha256": SHA_B,
        "paired_delta_semantics": (
            "SAME_SLOT_BASELINE_CANDIDATE_SIGN_NORMALIZED_IMPROVEMENT_MEAN"
        ),
        "paired_ci_contract": {
            "inner_sample_count": roles.POWER_INNER_RESAMPLES,
            "lower_nearest_rank_1_indexed": roles.POWER_LOWER_NEAREST_RANK,
            "upper_nearest_rank_1_indexed": roles.POWER_UPPER_NEAREST_RANK,
            "detection_rule": "SIGN_NORMALIZED_LOWER_BOUND_GT_ZERO",
        },
        "bootstrap_resample_count": roles.POWER_BOOTSTRAP_REPLICATES,
        "outer_experiment_count": roles.POWER_OUTER_EXPERIMENTS,
        "inner_resamples_per_outer": roles.POWER_INNER_RESAMPLES,
        "effect_results": effect_results,
        "status": "PASS",
    }


def _issued_roles_capabilities_fixture():
    """Test-only issuance of exact control capability types and bindings."""

    from blueprint_e2e_v2.c28f_v5 import a4_control as control

    records = [
        {"desc_id": index, "ground_truth_video_id": f"video-{index:05d}"}
        for index in range(17_435)
    ]
    role_base = {
        "schema_version": "c28f_a4_g4_role_lock_input_v1",
        "goal_id": "goal-a4",
        "attempt_id": "attempt-a4",
        "committed_transaction_id": "transaction-a4",
        "committed_state_sha256": SHA_A,
        "committed_event_sha256": SHA_B,
        "train_fit_mapping_records": records,
        "train_fit_source_receipt_sha256": SHA_C,
        "train_fit_mapping_sha256": semantic_sha256({"records": records}),
        "protected_desc_ids": [],
        "protected_video_ids": [],
        "projection_completion_capability_sha256": SHA_D,
        "projection_manifest_sha256": SHA_A,
        "mapping_rows_sha256": SHA_B,
        "protected_video_id_lines_sha256": SHA_C,
    }
    role_payload = {
        **role_base,
        "role_input_snapshot_sha256": semantic_sha256(role_base),
    }
    role_handle = control._register_opaque_capability(
        control.A4G4RoleLockInputHandle(
            control._OPAQUE_CONTROL_ISSUER_SEAL,
            role_payload,
        )
    )
    reference_rows = [
        {
            "schema_version": roles.POWER_REPLAY_ROW_SCHEMA,
            "desc_id": 100_000 + index,
            "gt_video_id": f"reference-video-{index % 512:04d}",
            "outcomes": {key: index % 2 for key in POWER_OUTCOME_KEYS},
            "replay_packet_sha256": SHA_A,
            "query_manifest_sha256": SHA_B,
        }
        for index in range(8_739)
    ]
    reference_rows_sha256 = hashlib.sha256(
        canonical_json_bytes(reference_rows)
    ).hexdigest()
    reference_population = roles.build_historical_power_reference_from_rows(
        reference_rows,
        power_rows_sha256=reference_rows_sha256,
        replay_packet_sha256=SHA_A,
        query_manifest_sha256=SHA_B,
        source_registry_sha256=SHA_D,
    )
    contract = frozen_power_evaluator_contract()
    power_base = {
        "schema_version": "c28f_a4_g2_power_replay_capability_v1",
        "goal_id": "goal-a4",
        "attempt_id": "attempt-a4",
        "committed_transaction_id": "transaction-a4",
        "committed_state_sha256": SHA_A,
        "committed_event_sha256": SHA_B,
        "g2_replay_control_receipt_sha256": SHA_C,
        "replay_packet_sha256": SHA_A,
        "query_manifest_sha256": SHA_B,
        "source_registry_sha256": SHA_D,
        "power_rows_file_sha256": SHA_A,
        "power_rows_sha256": reference_rows_sha256,
        "power_row_count": len(reference_rows),
        "historical_reference_population": reference_population,
        "historical_reference_file_sha256": hashlib.sha256(
            canonical_json_bytes(reference_population) + b"\n"
        ).hexdigest(),
        "power_evaluator_contract": contract,
        "power_contract_file_sha256": SHA_B,
    }
    power_payload = {
        **power_base,
        "power_replay_capability_binding_sha256": semantic_sha256(power_base),
    }
    power_capability = control._register_opaque_capability(
        control.A4G2PowerReplayCapability(
            control._OPAQUE_CONTROL_ISSUER_SEAL,
            power_payload,
            reference_rows,
            mock.Mock(name="test_bootstrap_lease"),
        )
    )
    role_records = [
        RoleRecord(record["desc_id"], record["ground_truth_video_id"])
        for record in records
    ]
    assignments = assign_video_clusters(
        role_records,
        [record.desc_id for record in role_records],
    )
    frozen_core_prefix_order = sorted(
        {
            record.gt_video_id
            for record in assignments[CORE_ROLE]
        },
        key=lambda value: roles._cluster_order_key(
            value,
            roles.ASSIGNMENT_SEED,
        ),
    )
    frozen_role_lanes, reserved_core_order = (
        roles._frozen_power_expansion_lanes(frozen_core_prefix_order)
    )
    role_results = {
        role: _synthetic_v3_power_result(
            role=role,
            query_count=len(assignments[role]),
            cluster_count=len(
                {record.gt_video_id for record in assignments[role]}
            ),
            reference_population=reference_population,
        )
        for role in ("train_fit_mechanism_dev", "train_fit_confirm")
    }
    power_report = {
        "status": "PASS",
        "expansion_policy": (
            "PREREGISTERED_DISJOINT_ROLE_LANE_PREFIXES_BEFORE_MANIFEST_LOCK"
        ),
        "minimum_effect_registry_sha256": (
            roles.MINIMUM_EFFECTS_REGISTRY_SHA256
        ),
        "effect_outcome_registry_sha256": (
            roles.EFFECT_OUTCOME_REGISTRY_SHA256
        ),
        "effect_threshold_reduction_count": 0,
        "historical_reference_population": reference_population,
        "membership_assignment_inputs": (
            "TRAIN_FIT_GT_VIDEO_CLUSTERS_PLUS_FROZEN_SEEDED_DISJOINT_ROLE_"
            "LANE_ORDERS_PLUS_POWER_DETERMINED_PER_ROLE_PREFIX_LENGTH_BEFORE_"
            "MANIFEST_LOCK"
        ),
        "historical_outcomes_select_identity_order_or_within_prefix_members": (
            False
        ),
        "cross_role_historical_outcomes_affect_lane_identity_order_or_members": (
            False
        ),
        "historical_outcomes_may_determine_preregistered_prefix_length": True,
        "membership_lock_timing": (
            "ALL_PREFIX_EXPANSION_BEFORE_ROLE_MANIFEST_LOCK"
        ),
        "expansion_lane_algorithm": roles.POWER_EXPANSION_LANE_ALGORITHM,
        "frozen_core_prefix_video_ids_sha256": roles._sha_lines(
            frozen_core_prefix_order
        ),
        "frozen_role_lane_video_ids_sha256s": {
            role: roles._sha_lines(lane)
            for role, lane in sorted(frozen_role_lanes.items())
        },
        "reserved_core_video_ids_sha256": roles._sha_lines(
            reserved_core_order
        ),
        "role_specific_expansion_lanes_disjoint": True,
        "adjustment_rounds": [],
        "role_results": role_results,
    }
    trusted_role_counts = {
        role: {
            "query_count": len(assignments[role]),
            "gt_video_cluster_count": len(
                {record.gt_video_id for record in assignments[role]}
            ),
        }
        for role in ("train_fit_mechanism_dev", "train_fit_confirm")
    }
    result_hashes = {
        role: semantic_sha256(result)
        for role, result in sorted(role_results.items())
    }
    trusted_base = {
        "schema_version": (
            "c28f_a4_g2_power_computation_receipt_capability_v1"
        ),
        "status": "CONTROLLER_RECOMPUTED_TRUSTED_POWER",
        "goal_id": power_payload["goal_id"],
        "attempt_id": power_payload["attempt_id"],
        "g2_committed_transaction_id": power_payload[
            "committed_transaction_id"
        ],
        "g2_committed_state_sha256": power_payload[
            "committed_state_sha256"
        ],
        "g2_committed_event_sha256": power_payload[
            "committed_event_sha256"
        ],
        "power_replay_capability_sha256": power_capability.capability_sha256,
        "power_rows_sha256": power_payload["power_rows_sha256"],
        "historical_reference_population_sha256": reference_population[
            "reference_population_sha256"
        ],
        "power_evaluator_contract_sha256": contract["contract_sha256"],
        "role_counts": trusted_role_counts,
        "role_counts_sha256": hashlib.sha256(
            canonical_json_bytes(trusted_role_counts)
        ).hexdigest(),
        "role_results": role_results,
        "role_result_sha256s": result_hashes,
        "combined_results_sha256": hashlib.sha256(
            canonical_json_bytes(role_results)
        ).hexdigest(),
        "raw_power_rows_disclosed": False,
    }
    trusted_receipt = {
        **trusted_base,
        "receipt_sha256": semantic_sha256(trusted_base),
    }
    return (
        role_handle,
        power_capability,
        assignments,
        power_report,
        trusted_receipt,
    )


class RoleContractTests(unittest.TestCase):
    @staticmethod
    def records() -> list[RoleRecord]:
        records: list[RoleRecord] = []
        desc_id = 0
        for video_index in range(160):
            for _ in range(100):
                records.append(RoleRecord(desc_id, f"video-{video_index:04d}"))
                desc_id += 1
        return records

    def test_cluster_assignment_is_complete_and_disjoint(self) -> None:
        records = self.records()
        ids = [record.desc_id for record in records]
        assignments = assign_video_clusters(records, ids)
        observed = sorted(record.desc_id for values in assignments.values() for record in values)
        self.assertEqual(observed, ids)
        videos_by_role = [
            {record.gt_video_id for record in values}
            for values in assignments.values()
        ]
        for index, left in enumerate(videos_by_role):
            for right in videos_by_role[index + 1:]:
                self.assertFalse(left & right)
        self.assertTrue(assignments[CORE_ROLE])

    def test_frozen_power_contract_contains_single_exact_derivation_contract(self) -> None:
        contract = frozen_power_evaluator_contract()
        self.assertEqual(10_000, contract["bootstrap_replicates"])
        self.assertEqual(
            "c28f_g2_power_evaluator_contract_v2",
            contract["schema_version"],
        )
        self.assertEqual(roles.POWER_ALGORITHM, contract["algorithm"])
        self.assertEqual(
            10_000,
            roles.POWER_OUTER_EXPERIMENTS
            + roles.POWER_OUTER_EXPERIMENTS * roles.POWER_INNER_RESAMPLES,
        )
        self.assertEqual(
            contract["contract_sha256"],
            semantic_sha256(contract, excluded_fields=("contract_sha256",)),
        )
        self.assertEqual(list(POWER_OUTCOME_KEYS), contract["observable_effects"])
        self.assertEqual(
            roles.MINIMUM_EFFECTS_REGISTRY_SHA256,
            contract["minimum_effect_registry_sha256"],
        )
        self.assertEqual(
            roles.EFFECT_OUTCOME_REGISTRY_SHA256,
            contract["effect_outcome_registry_sha256"],
        )
        derivation = contract["outcome_derivation_contract"]
        self.assertEqual(
            derivation["outcome_derivation_contract_sha256"],
            contract["outcome_derivation_contract_sha256"],
        )
        self.assertEqual(
            derivation["outcome_derivation_contract_sha256"],
            semantic_sha256(
                derivation,
                excluded_fields=("outcome_derivation_contract_sha256",),
            ),
        )
        self.assertEqual(
            "SIX_BINARY_SUFFICIENT_STATISTICS_NO_PROPOSALS_NO_GT_SPAN",
            derivation["persisted_row_semantics"],
        )
        projection = contract["candidate_role_projection_contract"]
        self.assertEqual(
            "COUNTS_ONLY_NO_DESC_ID_NO_VIDEO_ID_NO_OUTCOME_JOIN",
            projection["candidate_membership_fields_read_by_power"],
        )
        self.assertFalse(
            projection["historical_outcomes_select_candidate_identity_order"]
        )
        self.assertFalse(
            projection[
                "cross_role_historical_outcomes_affect_lane_identity_order_or_members"
            ]
        )
        self.assertEqual(
            "FROZEN_SEEDED_DISJOINT_ROLE_SPECIFIC_CORE_CLUSTER_LANES",
            projection["prefix_identity_order"],
        )
        self.assertEqual(
            roles.POWER_EXPANSION_LANE_ALGORITHM,
            projection["expansion_lane_algorithm"],
        )
        self.assertTrue(
            projection[
                "historical_outcomes_may_determine_preregistered_prefix_length"
            ]
        )
        outer_projection = contract["bootstrap_test_contract"][
            "outer_candidate_projection"
        ]
        self.assertEqual(
            "EQUAL_q_OR_q_PLUS_1_SLOT_PROJECTION_ONLY_NOT_OBSERVED_"
            "CANDIDATE_CLUSTER_SIZE_DISTRIBUTION",
            outer_projection["projection_assumption"],
        )
        resampling = contract["bootstrap_test_contract"][
            "deterministic_resampling"
        ]
        self.assertEqual("FORBIDDEN", resampling["inner_position_deduplication"])
        counter_fields = contract["bootstrap_test_contract"][
            "deterministic_resampling"
        ]["counter_preimage_fields"]
        self.assertEqual(
            [
                "domain",
                "seed",
                "role",
                "outer_index",
                "inner_index_or_negative_sampling_phase",
                "draw_index",
                "rejection_counter",
            ],
            counter_fields,
        )
        counter_source = inspect.getsource(roles._uniform_counter_index)
        cursor = -1
        for marker in (
            "POWER_COUNTER_DOMAIN",
            "ASSIGNMENT_SEED",
            "role,",
            "outer_index,",
            "inner_index,",
            "draw_index,",
            "rejection_counter,",
        ):
            cursor = counter_source.index(marker, cursor + 1)

    def test_v3_outer_slot_projection_is_exact_n_equal_q_or_q_plus_1(self) -> None:
        record_zero = roles._PowerReplayRecord(
            100,
            "reference-video-zero",
            {key: 0 for key in POWER_OUTCOME_KEYS},
        )
        record_one = roles._PowerReplayRecord(
            101,
            "reference-video-one",
            {key: 1 for key in POWER_OUTCOME_KEYS},
        )
        slots, slot_counts, cluster_indices = roles._outer_candidate_slots(
            role="train_fit_confirm",
            outer_index=0,
            proposed_query_count=5,
            proposed_cluster_count=3,
            historical_clusters=(
                ("reference-video-one", (record_one,)),
                ("reference-video-zero", (record_zero,)),
            ),
        )
        self.assertEqual([2, 2, 1], slot_counts)
        self.assertEqual(5, len(slots))
        self.assertEqual(3, len(cluster_indices))
        self.assertEqual(
            [2, 2, 1],
            [sum(slot[0] == position for slot in slots) for position in range(3)],
        )

    def test_v3_paired_ci_preserves_duplicate_cluster_positions(self) -> None:
        detected_rows = [[2, 0] for _ in range(48)] + [[0, 2]]
        detected = roles._outer_paired_detection_threshold(
            [0],
            inner_position_multiplicities=detected_rows,
            inner_resampled_slot_counts=[2] * roles.POWER_INNER_RESAMPLES,
            maximum_k=1,
        )
        self.assertEqual(1, detected["detection_threshold_k"])
        self.assertGreater(detected["ci_lower_at_threshold"], 0.0)
        self.assertEqual(0.0, detected["ci_lower_at_k_minus_1"])
        undetected_rows = [[2, 0] for _ in range(47)] + [[0, 2], [0, 2]]
        undetected = roles._outer_paired_detection_threshold(
            [0],
            inner_position_multiplicities=undetected_rows,
            inner_resampled_slot_counts=[2] * roles.POWER_INNER_RESAMPLES,
            maximum_k=1,
        )
        self.assertIsNone(undetected["detection_threshold_k"])
        self.assertEqual(0.0, undetected["ci_lower_at_maximum_k"])
        self.assertGreater(undetected["ci_upper_at_maximum_k"], 0.0)
        inner_source = inspect.getsource(roles._inner_cluster_position_resamples)
        self.assertIn("multiplicities[position] += 1", inner_source)
        self.assertNotIn("set(", inner_source)

    def test_v3_sign_normalizes_recall_gain_and_error_reduction(self) -> None:
        self.assertEqual(
            1,
            roles._sign_normalized_improvement(
                baseline_value=0,
                effect_rule={
                    "eligible_baseline_value": 0,
                    "injected_flip": "0_TO_1",
                },
            ),
        )
        self.assertEqual(
            1,
            roles._sign_normalized_improvement(
                baseline_value=1,
                effect_rule={
                    "eligible_baseline_value": 1,
                    "injected_flip": "1_TO_0",
                },
            ),
        )

    def test_v3_power_uses_outer_local_slots_not_global_first_flip_ranks(self) -> None:
        source = inspect.getsource(roles._role_power_sensitivity)
        for marker in (
            "_outer_candidate_slots",
            "_inner_cluster_position_resamples",
            "_outer_effect_eligible_position_order",
            "_outer_paired_detection_threshold",
            "outer_detection_thresholds",
        ):
            self.assertIn(marker, source)
        self.assertNotIn("first_flip_rank_by_effect", source)
        self.assertNotIn("mapped_clusters = {", source)
        self.assertNotIn("_effect_flip_order", source)

    def test_v3_power_evidence_rejects_semantically_resigned_threshold_drift(
        self,
    ) -> None:
        rows = [
            self.power_row(100_000),
            self.power_row(
                100_001,
                outcomes={key: 1 for key in POWER_OUTCOME_KEYS},
            ),
        ]
        population = roles.build_historical_power_reference_from_rows(
            rows,
            power_rows_sha256=hashlib.sha256(
                canonical_json_bytes(rows)
            ).hexdigest(),
            replay_packet_sha256=SHA_A,
            query_manifest_sha256=SHA_B,
            source_registry_sha256=SHA_D,
        )
        result = _synthetic_v3_power_result(
            role="train_fit_confirm",
            query_count=2_048,
            cluster_count=64,
            reference_population=population,
        )
        roles._validate_power_result_evidence(
            result,
            role="train_fit_confirm",
            expected_query_count=2_048,
            expected_cluster_count=64,
            reference_population=population,
        )
        drifted = {
            **result,
            "effect_results": {
                key: {
                    **value,
                    "outer_detection_thresholds": list(
                        value["outer_detection_thresholds"]
                    ),
                    "outer_threshold_ci": [
                        dict(row) for row in value["outer_threshold_ci"]
                    ],
                }
                for key, value in result["effect_results"].items()
            },
        }
        effect = drifted["effect_results"]["core_primary_r1"]
        effect["outer_detection_thresholds"][0] = 2
        effect["outer_threshold_ci"][0]["detection_threshold_k"] = 2
        effect["outer_detection_thresholds_sha256"] = hashlib.sha256(
            canonical_json_bytes(effect["outer_detection_thresholds"])
        ).hexdigest()
        effect["outer_threshold_ci_sha256"] = hashlib.sha256(
            canonical_json_bytes(effect["outer_threshold_ci"])
        ).hexdigest()
        with self.assertRaisesRegex(RoleContractError, "effect decision drift"):
            roles._validate_power_result_evidence(
                drifted,
                role="train_fit_confirm",
                expected_query_count=2_048,
                expected_cluster_count=64,
                reference_population=population,
            )

    def test_wholly_resigned_pass_requires_persisted_controller_recomputation(
        self,
    ) -> None:
        (
            _role_handle,
            _power_capability,
            _assignments,
            power_report,
            trusted_receipt,
        ) = _issued_roles_capabilities_fixture()
        forged_results = json.loads(
            canonical_json_bytes(power_report["role_results"]).decode("utf-8")
        )
        forged = forged_results["train_fit_confirm"]
        query_count = forged["query_count"]
        for effect in forged["effect_results"].values():
            thresholds = [2] * roles.POWER_OUTER_EXPERIMENTS
            ci_rows = [dict(row) for row in effect["outer_threshold_ci"]]
            for row in ci_rows:
                row["detection_threshold_k"] = 2
                row["ci_lower_at_threshold"] = 2.0 / float(query_count)
                row["ci_upper_at_threshold"] = 2.0 / float(query_count)
            effect["outer_detection_thresholds"] = thresholds
            effect["outer_detection_thresholds_sha256"] = hashlib.sha256(
                canonical_json_bytes(thresholds)
            ).hexdigest()
            effect["outer_threshold_ci"] = ci_rows
            effect["outer_threshold_ci_sha256"] = hashlib.sha256(
                canonical_json_bytes(ci_rows)
            ).hexdigest()
            effect["mde_k"] = 2
            effect["mde_ratio"] = 2.0 / float(query_count)
        roles._validate_power_result_evidence(
            forged,
            role="train_fit_confirm",
            expected_query_count=forged["query_count"],
            expected_cluster_count=forged["gt_video_cluster_count"],
            reference_population=power_report[
                "historical_reference_population"
            ],
        )
        forged_receipt = json.loads(
            canonical_json_bytes(trusted_receipt).decode("utf-8")
        )
        forged_receipt["role_results"] = forged_results
        forged_receipt["role_result_sha256s"] = {
            role: semantic_sha256(result)
            for role, result in sorted(forged_results.items())
        }
        forged_receipt["combined_results_sha256"] = hashlib.sha256(
            canonical_json_bytes(forged_results)
        ).hexdigest()
        forged_receipt["receipt_sha256"] = semantic_sha256(
            forged_receipt,
            excluded_fields=("receipt_sha256",),
        )
        roles._validate_trusted_power_computation_receipt(
            forged_receipt,
            role_results=forged_results,
            role_counts=forged_receipt["role_counts"],
            power_replay_capability_sha256=forged_receipt[
                "power_replay_capability_sha256"
            ],
            power_rows_sha256=forged_receipt["power_rows_sha256"],
            historical_reference_population_sha256=forged_receipt[
                "historical_reference_population_sha256"
            ],
            power_evaluator_contract_sha256=forged_receipt[
                "power_evaluator_contract_sha256"
            ],
            g2_committed_transaction_id=forged_receipt[
                "g2_committed_transaction_id"
            ],
            g2_committed_state_sha256=forged_receipt[
                "g2_committed_state_sha256"
            ],
            g2_committed_event_sha256=forged_receipt[
                "g2_committed_event_sha256"
            ],
            goal_id=forged_receipt["goal_id"],
            attempt_id=forged_receipt["attempt_id"],
        )
        from blueprint_e2e_v2.c28f_v5 import a4_control as control

        def persisted_only(value):
            if value != trusted_receipt:
                raise control.A4ControlError("not controller persisted")
            return dict(value)

        with mock.patch.object(
            control,
            "validate_persisted_g2_power_computation_receipt",
            side_effect=persisted_only,
        ), self.assertRaisesRegex(
            RoleContractError,
            "lacks persisted controller lineage",
        ):
            roles._require_persisted_trusted_power_computation_receipt(
                forged_receipt
            )

    @staticmethod
    def power_row(desc_id=0, outcomes=None):
        return {
            "schema_version": "c28f_g2_power_replay_row_v1",
            "desc_id": desc_id,
            "gt_video_id": f"video-{desc_id:04d}",
            "outcomes": (
                {key: 0 for key in POWER_OUTCOME_KEYS}
                if outcomes is None
                else outcomes
            ),
            "replay_packet_sha256": SHA_A,
            "query_manifest_sha256": SHA_B,
        }

    def test_minimal_power_rows_accept_exact_six_bits(self) -> None:
        rows = [self.power_row(0), self.power_row(1)]
        normalized = _normalize_power_replay_rows(
            rows,
            query_manifest_sha256=SHA_B,
            replay_packet_sha256=SHA_A,
        )
        self.assertEqual({0, 1}, set(normalized))
        self.assertEqual(set(POWER_OUTCOME_KEYS), set(normalized[0].outcomes))

    def test_historical_reference_is_sole_rows_builder_and_no_join_manifest(self) -> None:
        rows = [self.power_row(100_000), self.power_row(100_001)]
        rows_sha256 = hashlib.sha256(canonical_json_bytes(rows)).hexdigest()
        manifest = roles.build_historical_power_reference_from_rows(
            rows,
            power_rows_sha256=rows_sha256,
            replay_packet_sha256=SHA_A,
            query_manifest_sha256=SHA_B,
            source_registry_sha256=SHA_D,
        )
        self.assertEqual(
            manifest,
            roles.validate_historical_power_reference_population(manifest),
        )
        self.assertEqual(2, manifest["power_row_count"])
        self.assertFalse(manifest["train_fit_desc_id_join_performed"])
        self.assertFalse(manifest["train_fit_video_id_join_performed"])
        drifted = dict(manifest)
        drifted["historical_performance_selects_membership"] = True
        drifted["reference_population_sha256"] = semantic_sha256(
            drifted,
            excluded_fields=("reference_population_sha256",),
        )
        with self.assertRaisesRegex(RoleContractError, "manifest drift"):
            roles.validate_historical_power_reference_population(drifted)

    def test_minimal_power_rows_reject_bool_extra_and_proposals(self) -> None:
        boolean = self.power_row(0)
        boolean["outcomes"] = {key: 0 for key in POWER_OUTCOME_KEYS}
        boolean["outcomes"][POWER_OUTCOME_KEYS[0]] = False
        with self.assertRaisesRegex(RoleContractError, "outcomes are not exact"):
            _normalize_power_replay_rows(
                [boolean], query_manifest_sha256=SHA_B, replay_packet_sha256=SHA_A
            )
        extra = self.power_row(0)
        extra["joint_proposals"] = []
        with self.assertRaisesRegex(RoleContractError, "field set"):
            _normalize_power_replay_rows(
                [extra], query_manifest_sha256=SHA_B, replay_packet_sha256=SHA_A
            )

    def test_prefix_expansion_verifier_accepts_no_move_and_rejects_skipping(self) -> None:
        records = self.records()
        ids = [record.desc_id for record in records]
        initial = assign_video_clusters(records, ids)
        reference_rows = [
            self.power_row(100_000),
            self.power_row(
                100_001,
                outcomes={key: 1 for key in POWER_OUTCOME_KEYS},
            ),
        ]
        reference_population = roles.build_historical_power_reference_from_rows(
            reference_rows,
            power_rows_sha256=hashlib.sha256(
                canonical_json_bytes(reference_rows)
            ).hexdigest(),
            replay_packet_sha256=SHA_A,
            query_manifest_sha256=SHA_B,
            source_registry_sha256=SHA_D,
        )
        role_results = {
            role: _synthetic_v3_power_result(
                role=role,
                query_count=len(initial[role]),
                cluster_count=len(
                    {record.gt_video_id for record in initial[role]}
                ),
                reference_population=reference_population,
            )
            for role in ("train_fit_mechanism_dev", "train_fit_confirm")
        }
        frozen_core_prefix_order = sorted(
            {
                record.gt_video_id
                for record in initial[CORE_ROLE]
            },
            key=lambda value: roles._cluster_order_key(
                value,
                roles.ASSIGNMENT_SEED,
            ),
        )
        frozen_role_lanes, reserved_core_order = (
            roles._frozen_power_expansion_lanes(
                frozen_core_prefix_order
            )
        )
        invariant_report = {
            "status": "PASS",
            "expansion_policy": (
                "PREREGISTERED_DISJOINT_ROLE_LANE_PREFIXES_BEFORE_MANIFEST_LOCK"
            ),
            "minimum_effect_registry_sha256": (
                roles.MINIMUM_EFFECTS_REGISTRY_SHA256
            ),
            "effect_outcome_registry_sha256": (
                roles.EFFECT_OUTCOME_REGISTRY_SHA256
            ),
            "effect_threshold_reduction_count": 0,
            "historical_reference_population": reference_population,
            "membership_assignment_inputs": (
                "TRAIN_FIT_GT_VIDEO_CLUSTERS_PLUS_FROZEN_SEEDED_DISJOINT_ROLE_"
                "LANE_ORDERS_PLUS_POWER_DETERMINED_PER_ROLE_PREFIX_LENGTH_"
                "BEFORE_MANIFEST_LOCK"
            ),
            "historical_outcomes_select_identity_order_or_within_prefix_members": (
                False
            ),
            "cross_role_historical_outcomes_affect_lane_identity_order_or_members": (
                False
            ),
            "historical_outcomes_may_determine_preregistered_prefix_length": (
                True
            ),
            "membership_lock_timing": (
                "ALL_PREFIX_EXPANSION_BEFORE_ROLE_MANIFEST_LOCK"
            ),
            "expansion_lane_algorithm": roles.POWER_EXPANSION_LANE_ALGORITHM,
            "frozen_core_prefix_video_ids_sha256": roles._sha_lines(
                frozen_core_prefix_order
            ),
            "frozen_role_lane_video_ids_sha256s": {
                role: roles._sha_lines(lane)
                for role, lane in sorted(frozen_role_lanes.items())
            },
            "reserved_core_video_ids_sha256": roles._sha_lines(
                reserved_core_order
            ),
            "role_specific_expansion_lanes_disjoint": True,
            "adjustment_rounds": [],
            "role_results": role_results,
        }
        _assert_power_prefix_expansion(
            records=records,
            expected_desc_ids=ids,
            final_assignments=initial,
            power_report=invariant_report,
        )
        by_video = {}
        for record in records:
            by_video.setdefault(record.gt_video_id, []).append(record)
        skipped = frozen_role_lanes["train_fit_mechanism_dev"][1]
        invalid = {role: list(values) for role, values in initial.items()}
        moved = by_video[skipped]
        moved_desc = {record.desc_id for record in moved}
        invalid["train_fit_mechanism_dev"].extend(moved)
        invalid[CORE_ROLE] = [
            record for record in invalid[CORE_ROLE] if record.desc_id not in moved_desc
        ]
        bad_report = {
            **invariant_report,
            "adjustment_rounds": [
                {
                    "round_index": 0,
                    "failing_roles": ["train_fit_mechanism_dev"],
                    "moves": {
                        "train_fit_mechanism_dev": {
                            "target_query_count": len(
                                invalid["train_fit_mechanism_dev"]
                            ),
                            "moved_cluster_ids_in_seed_order": [skipped],
                            "moved_cluster_count": 1,
                            "moved_cluster_ids_sha256": roles._sha_lines([skipped]),
                            "query_count_after": len(
                                invalid["train_fit_mechanism_dev"]
                            ),
                        }
                    },
                }
            ],
        }
        with self.assertRaisesRegex(RoleContractError, "frozen role lane prefix"):
            _assert_power_prefix_expansion(
                records=records,
                expected_desc_ids=ids,
                final_assignments=invalid,
                power_report=bad_report,
            )

    def test_control_bound_manifest_lineage_and_assignment_contract(self) -> None:
        records = self.records()
        ids = [record.desc_id for record in records]
        assignments = assign_video_clusters(records, ids)
        assignment_base = {
            "algorithm": roles.POWER_ADAPTIVE_ASSIGNMENT_ALGORITHM,
            "seed": roles.ASSIGNMENT_SEED,
            "cluster_order": "sha256(seed_ascii + NUL + video_id_utf8),video_id_utf8",
            "cluster_atomic": True,
            "candidate_membership_source": (
                "TRAIN_FIT_GT_VIDEO_CLUSTERS_PLUS_FROZEN_SEEDED_DISJOINT_ROLE_"
                "LANE_ORDERS_PLUS_POWER_DETERMINED_PER_ROLE_PREFIX_LENGTH_"
                "BEFORE_MANIFEST_LOCK"
            ),
            "historical_reference_population_sha256": SHA_A,
            "historical_outcomes_select_identity_order_or_within_prefix_members": (
                False
            ),
            "cross_role_historical_outcomes_affect_lane_identity_order_or_members": (
                False
            ),
            "historical_outcomes_may_determine_preregistered_prefix_length": (
                True
            ),
            "power_expansion_lane_algorithm": (
                roles.POWER_EXPANSION_LANE_ALGORITHM
            ),
            "membership_lock_timing": (
                "ALL_PREFIX_EXPANSION_BEFORE_ROLE_MANIFEST_LOCK"
            ),
            "power_expansion_report_sha256": SHA_D,
        }
        assignment = {
            **assignment_base,
            "assignment_contract_sha256": semantic_sha256(assignment_base),
        }
        manifests = build_role_manifests(
            assignments,
            train_fit_manifest_sha256=SHA_A,
            assignment_source_sha256=SHA_B,
            producer_control_binding_sha256=SHA_C,
            assignment_contract=assignment,
        )
        for manifest in manifests.values():
            self.assertEqual(
                "CONTROL_CAPABILITY_BINDING_SHA256",
                manifest["producer_lineage"]["kind"],
            )
            self.assertEqual(assignment, manifest["assignment"])

    def test_historical_outcome_mutation_cannot_select_prefix_identity_order(self) -> None:
        records = self.records()
        ids = [record.desc_id for record in records]

        def reference(value):
            rows = [
                self.power_row(
                    100_000 + index,
                    outcomes={key: value for key in POWER_OUTCOME_KEYS},
                )
                for index in range(8)
            ]
            by_desc = _normalize_power_replay_rows(
                rows,
                query_manifest_sha256=SHA_B,
                replay_packet_sha256=SHA_A,
            )
            population = roles.build_historical_power_reference_from_rows(
                rows,
                power_rows_sha256=hashlib.sha256(
                    canonical_json_bytes(rows)
                ).hexdigest(),
                replay_packet_sha256=SHA_A,
                query_manifest_sha256=SHA_B,
                source_registry_sha256=SHA_D,
            )
            return by_desc, population

        def count_only_pass(**values):
            population = values["reference_population"]
            return {
                "role": values["role"],
                "query_count": values["proposed_query_count"],
                "gt_video_cluster_count": values["proposed_cluster_count"],
                "candidate_role_query_count": values["proposed_query_count"],
                "candidate_role_gt_video_cluster_count": values[
                    "proposed_cluster_count"
                ],
                "reference_population_sha256": population[
                    "reference_population_sha256"
                ],
                "effect_results": {
                    key: {"minimum_effect_ratio": threshold, "status": "PASS"}
                    for key, threshold in dict(MINIMUM_EFFECTS_RATIO).items()
                },
                "status": "PASS",
            }

        reference_zero, population_zero = reference(0)
        reference_one, population_one = reference(1)
        with mock.patch.object(
            roles,
            "_role_power_sensitivity",
            side_effect=count_only_pass,
        ):
            assignment_zero, _report_zero = roles._power_adaptive_assignments(
                records,
                ids,
                reference_by_desc=reference_zero,
                reference_population=population_zero,
            )
            assignment_one, _report_one = roles._power_adaptive_assignments(
                records,
                ids,
                reference_by_desc=reference_one,
                reference_population=population_one,
            )
        observed_zero = {
            role: [record.desc_id for record in values]
            for role, values in assignment_zero.items()
        }
        observed_one = {
            role: [record.desc_id for record in values]
            for role, values in assignment_one.items()
        }
        self.assertEqual(observed_zero, observed_one)
        self.assertFalse(
            _report_zero[
                "historical_outcomes_select_identity_order_or_within_prefix_members"
            ]
        )
        self.assertTrue(
            _report_zero[
                "historical_outcomes_may_determine_preregistered_prefix_length"
            ]
        )
        self.assertEqual(
            _report_zero["frozen_core_prefix_video_ids_sha256"],
            _report_one["frozen_core_prefix_video_ids_sha256"],
        )
        self.assertNotEqual(
            population_zero["reference_population_sha256"],
            population_one["reference_population_sha256"],
        )

    def test_cross_role_power_state_cannot_change_another_role_lane(self) -> None:
        records = self.records()
        ids = [record.desc_id for record in records]
        initial = assign_video_clusters(records, ids)
        initial_counts = {
            role: len(initial[role])
            for role in ("train_fit_mechanism_dev", "train_fit_confirm")
        }
        core_order = sorted(
            {record.gt_video_id for record in initial[CORE_ROLE]},
            key=lambda value: roles._cluster_order_key(
                value,
                roles.ASSIGNMENT_SEED,
            ),
        )
        lanes, reserved = roles._frozen_power_expansion_lanes(core_order)
        reference_rows = [self.power_row(100_000 + index) for index in range(8)]
        reference_by_desc = _normalize_power_replay_rows(
            reference_rows,
            query_manifest_sha256=SHA_B,
            replay_packet_sha256=SHA_A,
        )
        reference_population = roles.build_historical_power_reference_from_rows(
            reference_rows,
            power_rows_sha256=hashlib.sha256(
                canonical_json_bytes(reference_rows)
            ).hexdigest(),
            replay_packet_sha256=SHA_A,
            query_manifest_sha256=SHA_B,
            source_registry_sha256=SHA_D,
        )

        def run_with_failing_roles(failing_roles: set[str]):
            def selective_power(**values):
                role = values["role"]
                should_fail = (
                    role in failing_roles
                    and values["proposed_query_count"] == initial_counts[role]
                )
                status = "FAIL" if should_fail else "PASS"
                return {
                    "role": role,
                    "query_count": values["proposed_query_count"],
                    "gt_video_cluster_count": values["proposed_cluster_count"],
                    "candidate_role_query_count": values[
                        "proposed_query_count"
                    ],
                    "candidate_role_gt_video_cluster_count": values[
                        "proposed_cluster_count"
                    ],
                    "reference_population_sha256": reference_population[
                        "reference_population_sha256"
                    ],
                    "effect_results": {
                        key: {
                            "minimum_effect_ratio": threshold,
                            "mde_k": 1,
                            "status": status,
                        }
                        for key, threshold in dict(
                            MINIMUM_EFFECTS_RATIO
                        ).items()
                    },
                    "status": status,
                }

            with mock.patch.object(
                roles,
                "_role_power_sensitivity",
                side_effect=selective_power,
            ):
                return roles._power_adaptive_assignments(
                    records,
                    ids,
                    reference_by_desc=reference_by_desc,
                    reference_population=reference_population,
                )

        mechanism_only = run_with_failing_roles(
            {"train_fit_mechanism_dev"}
        )
        confirm_only = run_with_failing_roles({"train_fit_confirm"})
        both = run_with_failing_roles(
            {"train_fit_mechanism_dev", "train_fit_confirm"}
        )

        def moved_ids(report: dict, role: str) -> list[str]:
            return [
                video_id
                for round_record in report["adjustment_rounds"]
                for video_id in round_record["moves"].get(
                    role,
                    {},
                ).get("moved_cluster_ids_in_seed_order", [])
            ]

        mechanism_moved = moved_ids(
            mechanism_only[1],
            "train_fit_mechanism_dev",
        )
        mechanism_moved_with_confirm = moved_ids(
            both[1],
            "train_fit_mechanism_dev",
        )
        confirm_moved = moved_ids(confirm_only[1], "train_fit_confirm")
        confirm_moved_with_mechanism = moved_ids(
            both[1],
            "train_fit_confirm",
        )
        self.assertEqual(mechanism_moved, mechanism_moved_with_confirm)
        self.assertEqual(confirm_moved, confirm_moved_with_mechanism)
        self.assertEqual(
            mechanism_moved,
            lanes["train_fit_mechanism_dev"][: len(mechanism_moved)],
        )
        self.assertEqual(
            confirm_moved,
            lanes["train_fit_confirm"][: len(confirm_moved)],
        )
        self.assertFalse(set(mechanism_moved) & set(confirm_moved))
        lane_hashes = {
            role: roles._sha_lines(lane)
            for role, lane in sorted(lanes.items())
        }
        for assignments, report in (
            mechanism_only,
            confirm_only,
            both,
        ):
            self.assertEqual(
                lane_hashes,
                report["frozen_role_lane_video_ids_sha256s"],
            )
            self.assertFalse(
                report[
                    "cross_role_historical_outcomes_affect_lane_identity_order_or_members"
                ]
            )
            core_ids = {
                record.gt_video_id for record in assignments[CORE_ROLE]
            }
            self.assertTrue(set(reserved).issubset(core_ids))

    def test_unique_composite_consumes_exact_handles_and_contract(self) -> None:
        source = inspect.getsource(build_g4_role_policy_candidates)
        for marker in (
            "validate_g4_role_lock_input_handle",
            "validate_g2_power_replay_capability",
            "consume_snapshot_once",
            "consume_rows_once",
            "canonical_json_bytes(dict(provided_contract))",
            "build_historical_power_reference_from_rows",
            "validate_historical_power_reference_population",
            "G4 power insufficient; effect thresholds may not be lowered",
            "VALIDATED_ATOMIC_G4_CANDIDATES_AWAITING_STAGE_COMMIT",
            "atomic_all_or_none",
            "validate_g4_atomic_candidate",
        ):
            self.assertIn(marker, source)
        self.assertNotIn("set(reference_by_desc) != set(expected_desc_ids)", source)
        adaptive_source = inspect.getsource(roles._power_adaptive_assignments)
        self.assertNotIn("reference_by_desc.get(record.desc_id)", adaptive_source)
        sensitivity_source = inspect.getsource(roles._role_power_sensitivity)
        self.assertNotIn("role_records", sensitivity_source)
        self.assertIn("proposed_query_count", sensitivity_source)
        self.assertIn("proposed_cluster_count", sensitivity_source)
        with self.assertRaisesRegex(RuntimeError, "exact A4G4RoleLockInputHandle"):
            build_g4_role_policy_candidates(
                role_input_handle={}, power_replay_capability={}
            )

    def test_control_issued_fixture_builds_and_validates_one_atomic_candidate(self) -> None:
        role_handle, power_capability, assignments, power_report, trusted_receipt = (
            _issued_roles_capabilities_fixture()
        )
        from blueprint_e2e_v2.c28f_v5 import a4_control as control

        computation_capability = object()
        with mock.patch.object(
            roles,
            "_power_adaptive_assignments",
            return_value=(assignments, power_report),
        ), mock.patch.object(
            control.A4G2PowerReplayCapability,
            "issue_trusted_power_computation_receipt_once",
            return_value=computation_capability,
        ), mock.patch.object(
            control,
            "validate_g2_power_computation_receipt",
            return_value=trusted_receipt,
        ), mock.patch.object(
            control,
            "validate_persisted_g2_power_computation_receipt",
            side_effect=lambda value: dict(value),
        ):
            candidate = build_g4_role_policy_candidates(
                role_input_handle=role_handle,
                power_replay_capability=power_capability,
            )
            validated_candidate = validate_g4_atomic_candidate(candidate)
        self.assertEqual(roles.G4_ATOMIC_CANDIDATE_SCHEMA, candidate["schema_version"])
        self.assertEqual(candidate, validated_candidate)
        audit = candidate["cluster_power_audit"]
        reference = audit["historical_reference_population"]
        self.assertEqual(8_739, reference["power_row_count"])
        self.assertFalse(reference["train_fit_desc_id_join_performed"])
        self.assertFalse(reference["train_fit_video_id_join_performed"])
        self.assertFalse(
            audit[
                "historical_outcomes_select_identity_order_or_within_prefix_members"
            ]
        )
        self.assertTrue(
            audit[
                "historical_outcomes_may_determine_preregistered_prefix_length"
            ]
        )
        self.assertFalse(
            audit[
                "cross_role_historical_outcomes_affect_lane_identity_order_or_members"
            ]
        )
        self.assertTrue(audit["role_specific_expansion_lanes_disjoint"])
        self.assertEqual(
            roles.POWER_EXPANSION_LANE_ALGORITHM,
            audit["expansion_lane_algorithm"],
        )
        self.assertEqual(
            "ALL_PREFIX_EXPANSION_BEFORE_ROLE_MANIFEST_LOCK",
            audit["membership_lock_timing"],
        )
        self.assertEqual(
            (
                "INDEPENDENT_HISTORICAL_REFERENCE_POWER_WITH_CANDIDATE_COUNTS_ONLY_"
                "NO_POSTHOC_EFFECT_REDUCTION_NO_IDENTITY_ORDER_OR_WITHIN_PREFIX_"
                "SELECTION_POWER_DETERMINES_PER_ROLE_LANE_PREFIX_LENGTH_ONLY_"
                "BEFORE_ROLE_MANIFEST_LOCK"
            ),
            audit["interpretation"],
        )
        for role in ("train_fit_mechanism_dev", "train_fit_confirm"):
            result = audit["power_results"][role]
            count = candidate["role_lock"]["role_counts"][role]
            self.assertEqual(count["queries"], result["candidate_role_query_count"])
            self.assertEqual(
                count["gt_videos"],
                result["candidate_role_gt_video_cluster_count"],
            )
        with self.assertRaisesRegex(RuntimeError, "active controller issuance"):
            build_g4_role_policy_candidates(
                role_input_handle=role_handle,
                power_replay_capability=power_capability,
            )

    def test_power_registries_reject_external_mutation_and_replacement(self) -> None:
        with self.assertRaises(TypeError):
            MINIMUM_EFFECTS_RATIO["core_primary_r1"] = 0.0
        with self.assertRaises(TypeError):
            roles.EFFECT_OUTCOME_REGISTRY["core_primary_r1"][
                "eligible_baseline_value"
            ] = 1
        replacement = dict(MINIMUM_EFFECTS_RATIO)
        replacement["core_primary_r1"] = 0.0
        with mock.patch.object(
            roles,
            "MINIMUM_EFFECTS_RATIO",
            replacement,
        ), self.assertRaisesRegex(RoleContractError, "registry literal/SHA drift"):
            frozen_power_evaluator_contract()
        with mock.patch.object(
            roles,
            "MINIMUM_EFFECTS_RATIO",
            replacement,
        ), self.assertRaisesRegex(RoleContractError, "registry literal/SHA drift"):
            build_g4_role_policy_candidates(
                role_input_handle={},
                power_replay_capability={},
            )
        effect_replacement = {
            key: dict(value)
            for key, value in roles.EFFECT_OUTCOME_REGISTRY.items()
        }
        effect_replacement["core_primary_r1"]["eligible_baseline_value"] = 1
        with mock.patch.object(
            roles,
            "EFFECT_OUTCOME_REGISTRY",
            effect_replacement,
        ), self.assertRaisesRegex(RoleContractError, "registry literal/SHA drift"):
            _normalize_power_replay_rows(
                [self.power_row(0)],
                query_manifest_sha256=SHA_B,
                replay_packet_sha256=SHA_A,
            )

    def test_legacy_caller_signed_candidate_apis_fail_closed(self) -> None:
        for function in (
            build_role_lock,
            build_cluster_power_audit,
            build_formal_data_policy,
        ):
            with self.subTest(function=function.__name__), self.assertRaisesRegex(
                RoleContractError, "build_g4_role_policy_candidates"
            ):
                function({})

    def test_policy_and_update_budget_are_immutable(self) -> None:
        self.assertEqual("STRICT_CORE_ONLY", FORMAL_POLICY)
        self.assertEqual(17_360, U_FORMAL_UPDATES)
        source = inspect.getsource(roles._power_adaptive_assignments)
        self.assertNotIn("MINIMUM_EFFECTS_RATIO[effect_key] =", source)
        self.assertIn("effect_threshold_reduction_count", source)
        self.assertNotIn('"effect_threshold_reduction_count": 0', source)
        observed = {
            role: {
                "effect_results": {
                    key: {
                        "minimum_effect_ratio": (
                            threshold / 2.0
                            if role == "train_fit_confirm"
                            and key == "core_primary_r1"
                            else threshold
                        )
                    }
                    for key, threshold in dict(MINIMUM_EFFECTS_RATIO).items()
                }
            }
            for role in ("train_fit_mechanism_dev", "train_fit_confirm")
        }
        self.assertEqual(
            1,
            roles._effect_threshold_reduction_count(
                observed,
                dict(MINIMUM_EFFECTS_RATIO),
            ),
        )


class F1ContractTests(unittest.TestCase):
    def test_optimizer_schedule_uses_committed_updates(self) -> None:
        expected = {
            0: "P0",
            9: "P0",
            10: "P1",
            25: "P2",
            40: "P3",
            60: "P4",
            75: "P5",
            100: "P6",
        }
        for completed, phase in expected.items():
            self.assertEqual(schedule_phase(completed, 100)["phase"], phase)
        assert_microstep_invariance(
            before_completed_updates=7,
            after_completed_updates=7,
            optimizer_step_committed=False,
        )
        with self.assertRaises(F1ContractError):
            assert_microstep_invariance(
                before_completed_updates=7,
                after_completed_updates=8,
                optimizer_step_committed=False,
            )
        formal_expected = {
            0: "P0",
            1_736: "P1",
            4_340: "P2",
            6_944: "P3",
            10_416: "P4",
            13_020: "P5",
            17_360: "P6",
        }
        for completed, phase in formal_expected.items():
            self.assertEqual(schedule_phase(completed, 17_360)["phase"], phase)
        before_span_zero = schedule_phase(2_499_990, 10_000_000)
        at_span_zero = schedule_phase(2_500_000, 10_000_000)
        before_teacher_zero = schedule_phase(5_999_990, 10_000_000)
        at_teacher_zero = schedule_phase(6_000_000, 10_000_000)
        before_student_only = schedule_phase(7_499_990, 10_000_000)
        after_student_only = schedule_phase(7_500_010, 10_000_000)
        self.assertGreater(
            before_span_zero["frozen_weight_multipliers"]["gt_span_aux"], 0.0
        )
        self.assertEqual(
            at_span_zero["frozen_weight_multipliers"]["gt_span_aux"], 0.0
        )
        self.assertGreater(
            before_teacher_zero["frozen_weight_multipliers"]["teacher_shadow_distill"],
            0.0,
        )
        self.assertEqual(
            at_teacher_zero["frozen_weight_multipliers"]["teacher_shadow_distill"],
            0.0,
        )
        self.assertEqual(before_student_only["phase"], "P4")
        self.assertEqual(after_student_only["phase"], "P5")
        with self.assertRaises(F1ContractError):
            assert_microstep_invariance(
                before_completed_updates=7,
                after_completed_updates=8,
                optimizer_step_committed=1,
            )

    def test_temporal_grid_has_no_fixed_64_divisor(self) -> None:
        for cells in (64, 96, 128):
            grid = TemporalGridSpec(10.0, cells)
            span = grid.seconds_to_cell_span(1.0, 2.0)
            decoded = grid.cell_span_to_seconds(*span)
            self.assertLessEqual(decoded[0], 1.0)
            self.assertGreaterEqual(decoded[1], 2.0)
        with self.assertRaises(F1ContractError):
            TemporalGridSpec(float("nan"), 64)

    def test_temporal_grid_boundary_sides_do_not_snap_real_offsets(self) -> None:
        grid = TemporalGridSpec(3_600.0, 64)
        first = grid.boundary_sec(1)
        second = grid.boundary_sec(2)
        below = math.nextafter(first, -math.inf)
        above = math.nextafter(first, math.inf)
        real_offset_above = first + 1e-10

        self.assertEqual((0, 1), grid.seconds_to_cell_span(0.0, below))
        self.assertEqual((0, 1), grid.seconds_to_cell_span(0.0, first))
        self.assertEqual((0, 2), grid.seconds_to_cell_span(0.0, above))
        self.assertEqual(
            (0, 2),
            grid.seconds_to_cell_span(0.0, real_offset_above),
        )
        self.assertEqual((0, 2), grid.seconds_to_cell_span(below, second))
        self.assertEqual((1, 2), grid.seconds_to_cell_span(first, second))
        self.assertEqual((1, 2), grid.seconds_to_cell_span(above, second))
        self.assertEqual(
            (1, 64),
            grid.seconds_to_cell_span(first, grid.duration_sec),
        )
        with self.assertRaises(F1ContractError):
            grid.seconds_to_cell_span(
                first,
                math.nextafter(grid.duration_sec, math.inf),
            )

    def test_temporal_mask_padding_and_arbitrary_source_edges(self) -> None:
        for cells, valid, duration in ((64, 37, 7.3), (96, 96, 11.1), (128, 73, 0.91)):
            grid = TemporalGridSpec(duration, cells)
            edges = tuple(duration * index / valid for index in range(valid + 1))
            contract = TemporalTensorContract(
                grid=grid,
                valid_cells=valid,
                mask=(True,) * valid + (False,) * (cells - valid),
                source_edges_sec=edges,
            )
            audit = contract.audit()
            self.assertEqual(valid, audit["mask_true_count"])
            self.assertEqual(cells - valid, audit["right_padding_cells"])
        with self.assertRaises(F1ContractError):
            TemporalTensorContract(
                grid=TemporalGridSpec(10.0, 64),
                valid_cells=2,
                mask=(True, False) + (False,) * 62,
                source_edges_sec=(0.0, 5.0, 10.0),
            )

    def test_checkpoint_missing_fields_fail_closed(self) -> None:
        with self.assertRaises(F1ContractError):
            validate_checkpoint_payload({"model": {}})

    def test_checkpoint_payload_state_and_cursor_are_exact(self) -> None:
        candidate_snapshot = {
            "query_cursor": 8,
            "refresh_cursor": 3,
            "snapshot_id": "candidate-snapshot-0003",
            "fingerprint_sha256": SHA_A,
            "gt_support_hash_version": "gt-support-v1",
        }
        payload = {
            "schema_version": "c28f_checkpoint_contract_v1",
            "model": {},
            "optimizer": {},
            "scheduler": {},
            "scaler": {},
            "ema": {},
            "rng_python": (1, 2),
            "rng_numpy": (3, 4),
            "rng_torch_cpu": b"cpu",
            "rng_torch_cuda": [b"cuda0"],
            "epoch": 2,
            "global_optimizer_step": 7,
            "sampler_state": {
                "epoch": 2,
                "query_cursor": 8,
                "permutation_sha256": SHA_A,
            },
            "dataloader_state": {
                "worker_count": 8,
                "worker_seed_state_sha256": SHA_B,
                "persistent_workers": True,
            },
            "distributed_state": {"rank": 0, "world_size": 1},
            "accumulation_state": {
                "microstep": 0,
                "accumulation_steps": 4,
                "at_optimizer_step_boundary": True,
                "gradient_buffers_included": False,
                "communication_state_included": False,
            },
            "query_cursor": 8,
            "hard_negative_index_sha256": SHA_C,
            "candidate_refresh_state": {
                "scheduler_state": {"next_refresh_update": 16},
                "refresh_cursor": 3,
                "snapshot_id": "candidate-snapshot-0003",
                "snapshot_sha256": SHA_D,
            },
            "gt_support_hash_version": "gt-support-v1",
            "candidate_snapshot": candidate_snapshot,
            "candidate_snapshot_sha256": semantic_sha256(candidate_snapshot),
            "temporal_manifest_sha256": SHA_A,
            "metric_schema_sha256": SHA_B,
            "code_commit": "3ee00135cc00c9763d616fa1b49c64206742a007",
            "dirty_diff_sha256": SHA_C,
            "config_sha256": SHA_C,
            "loss_weight_state": {
                "gt_span_aux": 0.0,
                "gt_video_support": 0.0,
                "teacher_shadow_distill": 0.0,
                "teacher_candidate_rate": 0.0,
                "credit_loss": 0.0,
            },
            "eval_artifact_state": [],
            "completed_optimizer_updates": 7,
            "planned_optimizer_updates": 64,
        }
        self.assertEqual("VALID", validate_checkpoint_payload(payload)["status"])
        with self.assertRaisesRegex(F1ContractError, "field set must be exact"):
            validate_checkpoint_payload({**payload, "unexpected": True})
        with self.assertRaisesRegex(F1ContractError, "cursor mismatch"):
            validate_checkpoint_payload(
                {
                    **payload,
                    "sampler_state": {
                        "epoch": 2,
                        "query_cursor": 7,
                        "permutation_sha256": SHA_A,
                    },
                }
            )
        with self.assertRaisesRegex(F1ContractError, "mid-accumulation"):
            validate_checkpoint_payload(
                {
                    **payload,
                    "accumulation_state": {
                        "microstep": 1,
                        "accumulation_steps": 4,
                        "at_optimizer_step_boundary": False,
                        "gradient_buffers_included": False,
                        "communication_state_included": False,
                    },
                }
            )

    def test_synthetic_8q_and_128q_are_real_protocol_smokes(self) -> None:
        consumed = 0
        for query_count in (8, 128):
            smoke = deterministic_synthetic_fixture_smoke(query_count=query_count)
            self.assertEqual(smoke["status"], "PASS")
            self.assertTrue(smoke["synthetic_only"])
            self.assertFalse(smoke["performance_selection_use"])
            self.assertFalse(smoke["corpus_scale_acceptance_claimed"])
            self.assertFalse(smoke["model_forward_executed"])
            self.assertEqual(smoke["query_count"], query_count)
            self.assertTrue(smoke["bitwise_resume_equal"])
            self.assertTrue(smoke["row_schema"]["query_ids_unique"])
            consumed += smoke["synthetic_optimizer_updates_consumed"]
        self.assertEqual(consumed, 24)

    def test_synthetic_smoke_rejects_unregistered_size(self) -> None:
        with self.assertRaises(F1ContractError):
            deterministic_synthetic_fixture_smoke(query_count=16)

    def test_f1_report_stays_inside_budget_and_scope(self) -> None:
        report = build_f1_protocol_report()
        self.assertEqual(
            "SYNTHETIC_PROTOCOL_EVIDENCE_AWAITING_CONTROL_LEDGER",
            report["status"],
        )
        self.assertFalse(report["g6_commit_eligible"])
        self.assertFalse(report["ledger_derived_counts"])
        self.assertLessEqual(report["synthetic_optimizer_updates"], 64)
        self.assertEqual(report["synthetic_optimizer_updates"], 64)
        self.assertEqual(report["synthetic_8q_status"], "PASS")
        self.assertEqual(report["synthetic_128q_status"], "PASS")
        self.assertEqual(report["real_data_optimizer_updates"], 0)
        self.assertEqual(report["extra_model_forward_splits"], 0)
        self.assertEqual(
            report["applicability"]["F2_F5_candidate_global_loss_gradient_pressure"],
            "NOT_APPLICABLE_SCHEMA_ONLY",
        )

    def test_cost_ratio_ceilings_are_frozen_not_deferred(self) -> None:
        protocol = cost_measurement_protocol()
        ceilings = protocol["ratio_ceilings"]
        self.assertEqual(1.20, ceilings["B_DECOMP_VS_A"]["train_step_time"])
        self.assertEqual(1.05, ceilings["B_GLOBAL_VS_B_DECOMP"]["peak_memory"])
        self.assertEqual(1.15, ceilings["C_OR_D_VS_DIRECT_PARENT"]["train_step_time"])
        self.assertEqual(
            1.02,
            ceilings["F_OR_F_B_VS_E_OR_E_B"]["inference_latency"],
        )

    def test_superset_scaffold_has_real_namespace_and_optimizer_audit(self) -> None:
        contract = superset_scaffold_contract()
        self.assertTrue(contract["class_implemented"])
        audit = contract["class_contract_audit"]
        self.assertEqual("C28FSupersetModel", audit["class_name"])
        self.assertEqual([], audit["optimizer_parameter_names"])
        self.assertFalse(audit["forward_implemented_in_f1"])
        with self.assertRaises(F1ContractError):
            C28FSupersetModel(
                parameter_names=("base.query_encoder.only",),
                base_initialization_sha256=SHA_A,
            )

class FinalizationContractTests(unittest.TestCase):
    @staticmethod
    def authority() -> dict:
        permissions = {
            "AUTH_F0_F1_IMPLEMENTATION": {"state": "ACTIVE", "value": True},
            "AUTH_PROTECTED_ID_MAPPING_ONLY": {"state": "ACTIVE", "value": True},
        }
        for key in (
            "AUTH_F2_F5_PROTOTYPE",
            "AUTH_F6_MECHANISM",
            "AUTH_F8_FORMAL",
            "AUTH_HOLDOUT_OFFICIAL",
            "AUTH_GIT_BRANCH_CREATE_SWITCH",
            "AUTH_GIT_COMMIT",
            "AUTH_GIT_PUSH_PR",
        ):
            permissions[key] = {"state": "UNGRANTED", "value": False}
        base = {"schema_version": "synthetic-authority-v1", "permissions": permissions}
        return {**base, "authority_sha256": semantic_sha256(base)}

    def test_authority_closes_without_regranting_higher_stages(self) -> None:
        closed = close_authority_ledger(
            self.authority(), mapping_token_state="COMPLETED_NONREPLAYABLE"
        )
        self.assertEqual(
            closed["permissions"]["AUTH_F0_F1_IMPLEMENTATION"]["state"],
            "CONSUMED_CLOSED",
        )
        self.assertEqual(
            closed["permissions"]["AUTH_PROTECTED_ID_MAPPING_ONLY"]["state"],
            "CONSUMED_CLOSED",
        )
        self.assertFalse(closed["permissions"]["AUTH_F6_MECHANISM"]["value"])

    def test_authority_closure_requires_committed_self_hash(self) -> None:
        authority = self.authority()
        authority.pop("authority_sha256")
        with self.assertRaisesRegex(FinalizationError, "input authority ledger"):
            close_authority_ledger(
                authority,
                mapping_token_state="COMPLETED_NONREPLAYABLE",
            )

    def test_authority_closure_rejects_unknown_permission(self) -> None:
        authority = self.authority()
        authority["permissions"]["AUTH_UNREGISTERED"] = {
            "state": "ACTIVE",
            "value": True,
        }
        authority["authority_sha256"] = semantic_sha256(
            authority, excluded_fields=("authority_sha256",)
        )
        with self.assertRaisesRegex(FinalizationError, "registry is not exact"):
            close_authority_ledger(
                authority,
                mapping_token_state="COMPLETED_NONREPLAYABLE",
            )

    def test_final_decision_requires_complete_zero_forbidden_access(self) -> None:
        gates = {
            "G%d" % index: {
                "gate_id": "G%d" % index,
                "status": "PASS",
                "state_sha256": SHA_A,
                "event_sha256": SHA_B,
                "control_receipt_sha256": SHA_C,
            }
            for index in range(7)
        }
        forbidden = {
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
            "teacher_model_forward": 0,
            "gt_support_injection": 0,
            "gt_append": 0,
            "posthoc_temperature_refit_during_eval": 0,
            "protected_id_mapping_operations": 1,
        }
        decision = build_final_decision(
            gates=gates,
            routing_status="ROUTING_AMBIGUOUS",
            eval_ledger_sha256=SHA_B,
            access_summary_sha256=SHA_C,
            f0_a_eval_id="SYNTHETIC-F0-A",
            f0_b_eval_id="SYNTHETIC-F0-B",
            model_forward_counts={"F0_A": 1, "F0_B": 1, "protected": 0, "other": 0},
            access_counts=forbidden,
            id_mapping_authority_was_active=True,
            real_data_optimizer_updates=0,
            formal_policy_sha256=SHA_D,
            role_lock_sha256=SHA_A,
        )
        self.assertEqual(decision["terminal_status"], "F0_F1_WINDOW_COMPLETE_STOPPED")
        incomplete = dict(forbidden)
        incomplete.pop("protected_metric_read")
        with self.assertRaises(FinalizationError):
            build_final_decision(
                gates=gates,
                routing_status="ROUTING_AMBIGUOUS",
                eval_ledger_sha256=SHA_B,
                access_summary_sha256=SHA_C,
                f0_a_eval_id="SYNTHETIC-F0-A",
                f0_b_eval_id="SYNTHETIC-F0-B",
                model_forward_counts={"F0_A": 1, "F0_B": 1, "protected": 0, "other": 0},
                access_counts=incomplete,
                id_mapping_authority_was_active=True,
                real_data_optimizer_updates=0,
                formal_policy_sha256=SHA_D,
                role_lock_sha256=SHA_A,
            )

class SyntheticRecoveryRehydrateTests(unittest.TestCase):
    def validate_fixture(
        self,
        postimage: dict,
        artifact_bytes: dict[str, bytes],
        artifact_sha256s: dict[str, str],
        *,
        crash_point: str = "DURING_EVAL",
        root_sha256: str = SHA_A,
    ) -> dict:
        return _validate_rehydrated_synthetic_postimage_bytes(
            actual_postimage=postimage,
            artifact_bytes_by_name=artifact_bytes,
            artifact_bytes_sha256s=artifact_sha256s,
            synthetic_root_handle_sha256=root_sha256,
            crash_point=crash_point,
            parent_descriptor_identity_sha256=SHA_B,
            child_root_binding_sha256=SHA_C,
        )

    def test_exact_controller_rehydrated_postimage_bytes_are_accepted(self) -> None:
        postimage, artifact_bytes, artifact_sha256s = (
            _synthetic_rehydrate_fixture()
        )
        self.assertEqual(
            postimage,
            self.validate_fixture(
                postimage,
                artifact_bytes,
                artifact_sha256s,
            ),
        )

    def test_missing_extra_and_tampered_rehydrated_bytes_are_rejected(self) -> None:
        postimage, artifact_bytes, artifact_sha256s = (
            _synthetic_rehydrate_fixture()
        )
        missing = dict(artifact_bytes)
        missing.pop(next(iter(missing)))
        with self.assertRaisesRegex(F1ContractError, "byte namespace drift"):
            self.validate_fixture(postimage, missing, artifact_sha256s)

        extra = {**artifact_bytes, "unexpected.json": b"{}\n"}
        with self.assertRaisesRegex(F1ContractError, "byte namespace drift"):
            self.validate_fixture(postimage, extra, artifact_sha256s)

        tampered = dict(artifact_bytes)
        first_name = next(iter(tampered))
        tampered[first_name] += b"tampered"
        with self.assertRaisesRegex(F1ContractError, "file bytes drift"):
            self.validate_fixture(postimage, tampered, artifact_sha256s)

    def test_wrong_crash_handle_and_integer_mode_are_rejected(self) -> None:
        postimage, artifact_bytes, artifact_sha256s = (
            _synthetic_rehydrate_fixture()
        )
        with self.assertRaises(F1ContractError):
            self.validate_fixture(
                postimage,
                artifact_bytes,
                artifact_sha256s,
                crash_point="BEFORE_STATUS_RENAME",
            )
        with self.assertRaisesRegex(F1ContractError, "contract drift"):
            self.validate_fixture(
                postimage,
                artifact_bytes,
                artifact_sha256s,
                root_sha256=SHA_D,
            )

        integer_mode = json.loads(
            canonical_json_bytes(postimage).decode("utf-8")
        )
        integer_mode["root_identity"]["mode"] = 0o700
        integer_mode["postimage_sha256"] = semantic_sha256(
            integer_mode,
            excluded_fields=("postimage_sha256",),
        )
        with self.assertRaisesRegex(F1ContractError, "contract drift"):
            self.validate_fixture(
                integer_mode,
                artifact_bytes,
                artifact_sha256s,
            )

    def test_rehydrate_path_is_read_only_and_does_not_rerun_scenario(self) -> None:
        source = inspect.getsource(
            rehydrate_synthetic_atomic_recovery_evidence
        )
        for forbidden in (
            "run_synthetic_atomic_recovery_harness",
            "os.open(",
            "os.write(",
            ".mkdir(",
            "os.replace(",
            ".unlink(",
        ):
            self.assertNotIn(forbidden, source)
        self.assertIn("consume_verified_postimage_once", source)
        self.assertIn(
            "_validate_rehydrated_synthetic_postimage_bytes",
            source,
        )

    def test_controller_owned_fixed_body_covers_all_four_scenarios(self) -> None:
        for crash_point in (
            "BEFORE_CHECKPOINT_COMMIT",
            "AFTER_CHECKPOINT_BEFORE_MANIFEST",
            "DURING_EVAL",
            "BEFORE_STATUS_RENAME",
        ):
            with self.subTest(crash_point=crash_point):
                capability = _RecordingG6Capability(crash_point)
                result = execute_controller_owned_synthetic_atomic_recovery(
                    capability
                )
                material = _synthetic_atomic_recovery_expected_material(
                    crash_point
                )
                self.assertTrue(capability.consumed)
                self.assertEqual(
                    material["expected_artifact_bytes"],
                    capability.session.files,
                )
                self.assertEqual(
                    "FIXED_BODY_TERMINAL_AWAITING_CONTROLLER_POSTIMAGE",
                    result["status"],
                )
                self.assertEqual(crash_point, result["crash_point"])
                self.assertFalse(result["model_forward_executed"])
                self.assertFalse(result["path_returned"])
                self.assertFalse(result["external_executor_used"])
                self.assertEqual(0, result["real_data_optimizer_updates"])
                self.assertEqual(
                    semantic_sha256(
                        result,
                        excluded_fields=("result_sha256",),
                    ),
                    result["result_sha256"],
                )
                with self.assertRaisesRegex(
                    F1ContractError,
                    "consumption failed",
                ):
                    execute_controller_owned_synthetic_atomic_recovery(
                        capability
                    )

    def test_legacy_external_path_and_resume_entries_are_hard_disabled(
        self,
    ) -> None:
        public_start = inspect.signature(
            run_synthetic_atomic_recovery_harness
        )
        self.assertNotIn(
            "unstarted_resume_capability",
            public_start.parameters,
        )
        resume_source = inspect.getsource(
            resume_synthetic_atomic_recovery_harness
        )
        acquire_source = inspect.getsource(
            _acquire_synthetic_recovery_root
        )
        self.assertIn("CONTROLLER_FIXED_EXECUTOR_V2_REQUIRED", resume_source)
        self.assertIn("CONTROLLER_FIXED_EXECUTOR_V2_REQUIRED", acquire_source)
        self.assertNotIn("consume_precreated_child_once", acquire_source)
        self.assertNotIn("root.mkdir", acquire_source)
        with self.assertRaisesRegex(
            F1ContractError,
            "CONTROLLER_FIXED_EXECUTOR_V2_REQUIRED",
        ):
            run_synthetic_atomic_recovery_harness(
                object(),
                crash_point="DURING_EVAL",
            )
        with self.assertRaisesRegex(
            F1ContractError,
            "CONTROLLER_FIXED_EXECUTOR_V2_REQUIRED",
        ):
            resume_synthetic_atomic_recovery_harness(
                object(),
                None,
                crash_point="DURING_EVAL",
            )

    def test_controller_reread_rejects_symlink_following(self) -> None:
        from blueprint_e2e_v2.c28f_v5 import a4_control as control

        source = inspect.getsource(
            control._g6_controller_reread_actual_postimage
        )
        self.assertIn("follow_symlinks=False", source)
        self.assertIn("os.O_NOFOLLOW", source)
        self.assertIn("stat.S_ISREG(named.st_mode)", source)
        self.assertIn("stat.S_ISDIR(child_named.st_mode)", source)
        harness_source = inspect.getsource(
            execute_controller_owned_synthetic_atomic_recovery
        )
        read_source = inspect.getsource(
            _read_synthetic_regular_no_follow
        )
        self.assertNotIn("Path(", harness_source)
        self.assertNotIn("os.open(", harness_source)
        self.assertNotIn("os.replace(", harness_source)
        self.assertNotIn(".mkdir(", harness_source)
        self.assertEqual(
            ["execution_capability"],
            list(
                inspect.signature(
                    execute_controller_owned_synthetic_atomic_recovery
                ).parameters
            ),
        )
        self.assertIn("consume_preacquired_child_execution_once", harness_source)
        self.assertIn("session.read_regular", harness_source)
        self.assertIn("os.O_NOFOLLOW", read_source)
        self.assertIn("stat.S_ISREG", read_source)
        self.assertIn("st_nlink", read_source)


if __name__ == "__main__":
    unittest.main()
