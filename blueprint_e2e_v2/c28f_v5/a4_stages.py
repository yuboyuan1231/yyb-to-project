"""Pure G1-G6 stage plans for the extensible G0-A4 control plane.

This module exposes no bare-path artifact opener and never writes a file.  G2
may consume non-protected replay streams exactly once, but only through the
concrete controller-issued handle after its pre-open receipt is committed.  It
constructs canonical output bytes inside a private-issuer ``StageBundle``;
persistence is delegated to an A4 control adapter that must validate that exact
opaque bundle and already own the fixed writer lease.  G4/G6 outputs remain
non-authoritative candidates until the corresponding stage CAS receipt exists.
The separation is intentional: importing this module cannot silently turn a
metric helper into a data reader or a report writer.

The protected ID mapping operation remains deferred until G4.  Its reviewed
two-pass value-minimising implementation lives in ``a4_id_projector``; G1/G2
only bind the one-shot intent/receipt protocol and never receive a protected
source factory.
"""

from __future__ import annotations

import math
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Tuple

from .a4_metrics import (
    C7_LEGACY_VCMR_PARITY_FIELD_MAP,
    HISTORICAL_PARITY_TOLERANCE,
    compare_vcmr_parity,
    evaluate_true_video_retrieval,
    evaluate_fixture,
    evaluate_joint_vcmr_and_errors,
    fixture_expected_projection,
    gate_schema,
    iter_aligned_c7_replay_rows,
    metric_schema,
    temporal_iou,
)
from .a4_id_projector import PROJECTOR_SCHEMA_VERSION
from .a4_roles import (
    ASSIGNMENT_SEED,
    POWER_ADAPTIVE_ASSIGNMENT_ALGORITHM,
    POWER_EXPANSION_LANE_ALGORITHM,
    POWER_OUTCOME_KEYS,
    POWER_REPLAY_ROW_SCHEMA,
    frozen_power_evaluator_contract,
)
from .a4_security import (
    DescriptorPathGuard,
    DescriptorRecord,
    EvalBudgetBook,
    EvalBudgetToken,
    EvaluationRequest,
    EvidenceLedger,
    FrozenCanonicalRoot,
    FrozenPathRule,
    ID_MAP_SCHEMA_VERSION,
    PathIntent,
    SecurityViolation,
    SplitCapability,
    SplitAuthorizationRegistry,
    capture_mount_domain,
    canonical_path_map,
    derive_safety_manifest,
    reject_legacy_authorization_markers,
    validate_descriptor_chain_preopen,
    validate_root_registry_preopen,
    validate_single_action_argv,
    verify_claimed_safety_manifest_preopen,
    verify_pristine_write_root,
)
from .canonical import bytes_sha256, canonical_json_bytes, semantic_sha256
from .constants import CALIB_SELECT_MANIFEST


STAGE_PLAN_SCHEMA = "c28f_a4_stage_plan_v1"
G1_ACTION = "G1_FAIL_CLOSED_GUARD"
G2_ACTION = "G2_METRIC_CONTRACT"
G2_TRAIN_JSONL_INDEX_BOOTSTRAP_ACTION = "G2_TRAIN_JSONL_INDEX_BOOTSTRAP"
G3_ACTION = "G3_F0_A_BASELINE_FORENSICS"
G4_PROJECTION_ACTION = "G4_PROTECTED_ID_MAPPING_PROJECTION"
G4_PROJECTION_COMPLETE_ACTION = "COMPLETE_PROTECTED_ID_MAPPING_OPERATION"
G4_ACTION = "G4_ROLE_POLICY_LOCK"
G5_ACTION = "G5_F0_B_EXPOSURE_REPLICATION"
G6_ACTION = "G6_F1_PROTOCOL_AND_RECOVERY"
G7_ACTION = "G7_FINALIZE_AND_STOP"
G7_COMMIT_ACTION = "G7_COMMIT_FINALIZATION"

TRAIN_JSONL_INDEX_ABSENT = "ABSENT"
TRAIN_JSONL_INDEX_READY = "READY"
TRAIN_JSONL_INDEX_TRANSACTION_RE = re.compile(r"TRAIN-INDEX-[0-9a-f]{32}")

G1_OUTPUTS = (
    "f1_protocol/safety_firewall.json",
    "f1_protocol/g1_negative_tests.json",
    "f1_protocol/canonical_path_map.json",
    "f1_protocol/ledger_safety_manifest.json",
    "f1_protocol/g1_access_ledger.jsonl",
    "f1_protocol/g1_observation_scope_receipt.json",
)
G2_OUTPUTS = (
    "f0_forensics/metric_schema.json",
    "f1_protocol/gate_schema.json",
    "f1_protocol/gate_registry.g2_contribution.json",
    "f1_protocol/evaluator_equivalence.json",
    "f0_forensics/metric_fixture_v1_expected.json",
    "f0_forensics/baseline_replay_plan.json",
    "f0_forensics/g2_power_replay_rows.jsonl",
    "f0_forensics/g2_power_evaluator_contract.json",
    "f0_forensics/g2_historical_power_reference.json",
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
G4_OUTPUTS = (
    "f1_protocol/role_manifest_lock.json",
    "f1_protocol/cluster_power_audit.json",
    "f1_protocol/FORMAL_DATA_POLICY.json",
    "f1_protocol/g4_control_binding.json",
)
G4_PROJECTION_OUTPUTS = (
    "f1_protocol/protected_id_mapping.jsonl",
    "f1_protocol/protected_video_ids.txt",
    "f1_protocol/protected_id_projection_manifest.json",
    "f1_protocol/train_fit_desc_to_gt_video.jsonl",
    "f1_protocol/train_fit_desc_to_gt_video.source_receipt.json",
)
G4_TRUSTED_POWER_COMPUTATION_RECEIPT_SCHEMA = (
    "c28f_a4_g2_power_computation_receipt_capability_v1"
)
G4_TRUSTED_POWER_COMPUTATION_RECEIPT_FIELDS = (
    "schema_version",
    "status",
    "goal_id",
    "attempt_id",
    "g2_committed_transaction_id",
    "g2_committed_state_sha256",
    "g2_committed_event_sha256",
    "power_replay_capability_sha256",
    "power_rows_sha256",
    "historical_reference_population_sha256",
    "power_evaluator_contract_sha256",
    "role_counts",
    "role_counts_sha256",
    "role_results",
    "role_result_sha256s",
    "combined_results_sha256",
    "raw_power_rows_disclosed",
    "receipt_sha256",
)
G4_POWER_EVIDENCE_BINDING_FIELDS = (
    "status",
    "evidence_sha256",
    "power_replay_capability_binding_sha256",
    "power_replay_capability_sha256",
    "trusted_power_computation_receipt",
    "goal_id",
    "attempt_id",
    "committed_transaction_id",
    "committed_state_sha256",
    "committed_event_sha256",
    "g2_replay_control_receipt_sha256",
    "source_registry_sha256",
    "power_rows_file_sha256",
    "power_rows_sha256",
    "power_row_count",
    "historical_reference_population_sha256",
    "historical_reference_file_sha256",
    "replay_packet_sha256",
    "query_manifest_sha256",
    "power_contract_file_sha256",
    "power_evaluator_contract_sha256",
)
G4_CLUSTER_POWER_AUDIT_FIELDS = (
    "schema_version",
    "status",
    "role_lock_sha256",
    "projected_query_count",
    "gt_video_cluster_count",
    "cluster_size_distribution",
    "role_minimum_margins",
    "train_fit_core_queries",
    "train_fit_core_gt_video_clusters",
    "minimum_query_counts",
    "minimum_effects_ratio",
    "historical_reference_population",
    "candidate_membership_assignment_inputs",
    "membership_identity_order_fixed_before_power_computation",
    "historical_outcomes_select_identity_order_or_within_prefix_members",
    "cross_role_historical_outcomes_affect_lane_identity_order_or_members",
    "historical_outcomes_may_determine_preregistered_prefix_length",
    "membership_lock_timing",
    "expansion_lane_algorithm",
    "frozen_role_lane_video_ids_sha256s",
    "reserved_core_video_ids_sha256",
    "role_specific_expansion_lanes_disjoint",
    "power_result_use",
    "power_results",
    "power_expansion_report",
    "power_expansion_report_sha256",
    "power_evidence_binding",
    "effect_size_or_performance_claim",
    "interpretation",
    "audit_sha256",
)
G4_POWER_EXPANSION_REPORT_FIELDS = (
    "status",
    "expansion_policy",
    "minimum_effect_registry_sha256",
    "effect_outcome_registry_sha256",
    "effect_threshold_reduction_count",
    "historical_reference_population",
    "membership_assignment_inputs",
    "historical_outcomes_select_identity_order_or_within_prefix_members",
    "cross_role_historical_outcomes_affect_lane_identity_order_or_members",
    "historical_outcomes_may_determine_preregistered_prefix_length",
    "membership_lock_timing",
    "expansion_lane_algorithm",
    "frozen_core_prefix_video_ids_sha256",
    "frozen_role_lane_video_ids_sha256s",
    "reserved_core_video_ids_sha256",
    "role_specific_expansion_lanes_disjoint",
    "adjustment_rounds",
    "role_results",
    "trusted_power_computation_receipt",
)
G4_ROLE_ASSIGNMENT_FIELDS = (
    "algorithm",
    "seed",
    "cluster_order",
    "cluster_atomic",
    "candidate_membership_source",
    "historical_reference_population_sha256",
    "historical_outcomes_select_identity_order_or_within_prefix_members",
    "cross_role_historical_outcomes_affect_lane_identity_order_or_members",
    "historical_outcomes_may_determine_preregistered_prefix_length",
    "power_expansion_lane_algorithm",
    "membership_lock_timing",
    "power_expansion_report_sha256",
    "assignment_contract_sha256",
)
G4_ATOMIC_CANDIDATE_FIELDS = (
    "schema_version",
    "status",
    "goal_id",
    "attempt_id",
    "role_input_snapshot_sha256",
    "power_replay_capability_binding_sha256",
    "power_replay_capability_sha256",
    "role_manifests",
    "role_lock",
    "cluster_power_audit",
    "formal_data_policy",
    "artifact_candidate_hashes",
    "atomic_all_or_none",
    "control_commit_required",
    "persistence_claimed_by_pure_function",
    "atomic_candidate_sha256",
)
G4_ATOMIC_BUNDLE_BINDING_SCHEMA = "c28f_g4_atomic_bundle_binding_v3"
G4_ATOMIC_BUNDLE_BINDING_FIELDS = (
    "schema_version",
    "status",
    "goal_id",
    "attempt_id",
    "role_input_handle_sha256",
    "power_replay_capability_sha256",
    "power_replay_capability_binding_sha256",
    "trusted_power_computation_receipt_sha256",
    "atomic_candidate_sha256",
    "role_lock_sha256",
    "cluster_power_audit_sha256",
    "formal_data_policy_sha256",
    "candidate_membership_assignment_inputs",
    "membership_identity_order_fixed_before_power_computation",
    "membership_lock_timing",
    "historical_outcomes_may_determine_preregistered_prefix_length",
    "historical_outcomes_select_identity_order_or_within_prefix_members",
    "cross_role_historical_outcomes_affect_lane_identity_order_or_members",
    "expansion_lane_algorithm",
    "frozen_role_lane_video_ids_sha256s",
    "reserved_core_video_ids_sha256",
    "role_specific_expansion_lanes_disjoint",
    "power_result_use",
    "power_expansion_report_sha256",
    "bootstrap_replicates",
    "pass_claim_available_before_stage_commit",
    "binding_sha256",
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
    "f1_protocol/f1_protocol_report.final.json",
    "f1_protocol/g6_control_binding.json",
)
G7_OUTPUTS = (
    "f1_protocol/finalization/F0_F1_WINDOW_DECISION.json",
    "f1_protocol/finalization/F0_F1_WINDOW_DECISION.md",
    "f1_protocol/finalization/artifact_manifest.json",
    "f1_protocol/finalization/artifact_manifest.root.sha256",
    "f1_protocol/finalization/eval_ledger.jsonl",
    "f1_protocol/finalization/protected_data_access_summary.json",
    "f1_protocol/finalization/CONTINUATION.md",
    "f1_protocol/finalization/NEXT_AUTHORIZATION_REQUEST.json",
)
G6_CRASH_RECEIPT_SCHEMA = (
    "c28f_g6_controlled_synthetic_crash_receipt_v2"
)
G6_ACTUAL_POSTIMAGE_SCHEMA = "c28f_g6_synthetic_actual_postimage_v2"
G6_GLOBAL_LEDGER_SCHEMA = "c28f_a4_g6_global_ledger_snapshot_v1"
G6_CONTROL_BINDING_SCHEMA = "c28f_g6_control_binding_v2"
G6_CRASH_RECEIPT_FIELDS = (
    "schema_version",
    "status",
    "goal_id",
    "attempt_id",
    "synthetic_root_handle_sha256",
    "crash_point",
    "child_root_name",
    "child_root_path",
    "parent_descriptor_identity_sha256",
    "child_root_contract_sha256",
    "child_root_binding_sha256",
    "evidence_sha256",
    "evidence_file_sha256",
    "actual_postimage",
    "controller_reread_postimage_sha256",
    "real_data_optimizer_updates",
    "extra_model_forward_splits",
    "receipt_sha256",
)
G6_ACTUAL_POSTIMAGE_FIELDS = (
    "schema_version",
    "synthetic_root_handle_sha256",
    "crash_point",
    "root_identity",
    "artifact_count",
    "artifact_inventory",
    "artifact_inventory_sha256",
    "postimage_sha256",
)
G6_ACTUAL_POSTIMAGE_ROOT_IDENTITY_FIELDS = (
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
)
G6_ACTUAL_POSTIMAGE_INVENTORY_ROW_FIELDS = (
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
)
G6_GLOBAL_LEDGER_FIELDS = (
    "schema_version",
    "goal_id",
    "attempt_id",
    "committed_transaction_id",
    "committed_state_sha256",
    "committed_event_sha256",
    "event_journal_sha256",
    "eval_token_states",
    "model_forward_evaluations",
    "synthetic_crash_scenario_count",
    "synthetic_crash_evidence_set_sha256",
    "real_data_optimizer_updates",
    "extra_model_forward_splits",
    "snapshot_sha256",
)
G6_ATOMIC_CRASH_EVIDENCE_FIELDS = (
    "status",
    "required_crash_points",
    "crash_evidence_set_sha256",
    "committed_evidence_sha256s",
    "evidence_file_sha256s",
    "child_root_binding_sha256s",
    "actual_postimage_sha256s",
    "controller_reread_postimage_sha256s",
    "artifact_inventory_sha256s",
    "synthetic_root_handle_sha256",
)
G6_CONTROL_EVIDENCE_BINDING_FIELDS = (
    "g6_input_handle_sha256",
    "candidate_report_sha256",
    "global_ledger_snapshot_sha256",
    "crash_evidence_set_sha256",
    "role_lock_sha256",
    "g2_replay_evidence_sha256",
    "formal_policy_sha256",
    "pass_claim_available_before_stage_commit",
)
G6_CONTROL_BINDING_FIELDS = (
    "schema_version",
    "status",
    "goal_id",
    "attempt_id",
    "g6_input_handle_sha256",
    "candidate_report_sha256",
    "final_report_sha256",
    "global_ledger_snapshot_sha256",
    "crash_evidence_set_sha256",
    "crash_receipt_sha256s",
    "evidence_sha256s",
    "evidence_file_sha256s",
    "child_root_binding_sha256s",
    "actual_postimage_sha256s",
    "controller_reread_postimage_sha256s",
    "artifact_inventory_sha256s",
    "real_data_optimizer_updates",
    "extra_model_forward_splits",
    "pass_claim_available_before_stage_commit",
    "binding_sha256",
)
G4_PROJECTION_DECLARED_SAFETY_COUNTS = (
    ("protected_content_opens", 0),
    ("protected_id_mapping_content_opens", 5),
    ("historical_write_bytes", 0),
    ("model_forward_evaluations", 0),
    ("real_data_optimizer_updates", 0),
)
G4_ROLE_DECLARED_SAFETY_COUNTS = (
    ("protected_content_opens", 0),
    ("protected_id_mapping_content_opens", 0),
    ("historical_write_bytes", 0),
    ("model_forward_evaluations", 0),
    ("real_data_optimizer_updates", 0),
)
F0_COMPLETED_EVAL_SAFETY_COUNTS = (
    ("model_forward_count", 1),
    ("query_identity_source_content_open_count", 1),
    ("gt_source_content_open_count", 1),
    ("train_jsonl_total_authorized_content_open_count", 2),
    ("unauthorized_or_protected_semantic_decode_count", 0),
    ("protected_semantic_decode_count", 0),
    ("unmediated_content_open_count", 0),
    ("teacher_field_access_count", 0),
    ("teacher_candidate_count", 0),
    ("gt_support_count", 0),
    ("gt_append_count", 0),
    ("optimizer_update_count", 0),
)

# These are source-code authorities, not hashes obtained by opening replay
# artifacts.  Any source edit requires a newly reviewed stage receipt.
FROZEN_LEGACY_SEMANTIC_SOURCE_HASHES = {
    "blueprint_e2e_v2/data/temporal_grid.py": (
        "f1523b21aea4e3dacade49923403774b8d4a298b5a6b17dc3476d995f5ac5a55"
    ),
    "blueprint_e2e_v2/engine/evaluate.py": (
        "5c2bad26b8d026711d3cde5efc93f27585a165c5c7e828bb7bd7de10a1865d6a"
    ),
    "blueprint_e2e_v2/engine/metrics.py": (
        "8b8ccb85682c7d99e9558f6a4533bad54838d2c2d5b3b66ab31b3879abee15f3"
    ),
    "rlem_c6_b0/run_c6_b0_candidate_diagnostic.py": (
        "f8436c6583b37f2c7b6517fc365fdba0f69ea13e9b55f036fe243f4de3e7320d"
    ),
    "rlem_c7/freeze_c7_b3.py": (
        "502c4b0c9cd1afbf9064d6a25d24e29efe6ec4358f03a0a3554a8aefac465049"
    ),
    "rlem_c7/run_c7_b0_b1.py": (
        "bb4c19e24900e88c1c5ea34a38994b00b9674e3509722a1295874c8f219462e2"
    ),
    "rlem_c7/run_c7_b2_1_safety_repair.py": (
        "e61b897ddf76a65dc169a61acc8cdb0c6fbe1902bc52de45ae27536643649688"
    ),
    "rlem_c7/run_c7_b3_stable_generalization.py": (
        "5b37fde2835b705d87f49e5f0b3b10968357451b73f1c82a89cfc6cb49634751"
    ),
    "standalone_eval/eval.py": (
        "7ba192993f461e92235614f5033dc5700155ca2cf61d2d0b8f98cbc211786f3a"
    ),
}

FROZEN_METRIC_FIXTURE_FILE_SHA256 = (
    "1aa1eb90cc16d1bc1e13f811bf93d9bb853c3696cca59b799641b2bce8e9f1d5"
)
FROZEN_METRIC_FIXTURE_CANONICAL_SHA256 = (
    "362928120d93b9b6860ab9057ecc993642fb519686c5a112d520915e181cfc34"
)
FROZEN_METRIC_EXPECTED_FILE_SHA256 = (
    "363e6a5d32d9d0655e44d4719fa33809b5ec11e63bcb30eb7beb093532e1b9d6"
)
FROZEN_METRIC_EXPECTED_CANONICAL_SHA256 = (
    "9ff448650045c37570b924e917ff8b007a6a6ee6b0cc38c34ceaf224c8f6e13e"
)

REQUIRED_G1_NEGATIVE_PROBES = {
    "budget_over_limit": "EVAL_BUDGET_EXHAUSTED",
    "budget_token_replay": "EVAL_TOKEN_REPLAY",
    "launcher_c28e_stage_all": "LEGACY_OR_PROTECTED_LAUNCHER_PATH_FORBIDDEN",
    "launcher_c28e_stage_holdout": "LEGACY_OR_PROTECTED_LAUNCHER_PATH_FORBIDDEN",
    "legacy_authorization_marker": "LEGACY_ROOT_AUTHORIZATION_MARKER_FORBIDDEN",
    "missing_capability": "MISSING_REQUIRED_FIELD",
    "output_forbidden_overlap": "WRITE_ROOT_FORBIDDEN_OVERLAP",
    "output_roots_overlap": "WRITE_ROOTS_OVERLAP",
    "path_device_inode_alias": "FORBIDDEN_DEVICE_INODE_ALIAS",
    "path_dotdot": "PATH_TRAVERSAL_OR_DOT_COMPONENT",
    "path_symlink": "SYMLINK_COMPONENT_FORBIDDEN",
    "protected_alias_calib_holdout": "PROTECTED_SPLIT_FORBIDDEN",
    "protected_alias_official": "PROTECTED_SPLIT_FORBIDDEN",
    "protected_alias_pseudo_official_holdout": "PROTECTED_SPLIT_FORBIDDEN",
    "safety_manifest_ledger_mismatch": "SAFETY_MANIFEST_LEDGER_MISMATCH",
    "split_missing": "MISSING_REQUIRED_FIELD",
    "split_null": "MISSING_REQUIRED_FIELD",
    "split_random": "SPLIT_ID_MISMATCH",
    "wrong_authority": "EVAL_AUTHORITY_ID_MISMATCH",
    "wrong_budget_token": "EVAL_BUDGET_TOKEN_MISMATCH",
    "wrong_candidate_manifest": "CANDIDATE_CORPUS_MANIFEST_SHA_MISMATCH",
    "wrong_goal": "EVAL_GOAL_ID_MISMATCH",
    "wrong_purpose": "EVAL_PURPOSE_MISMATCH",
    "wrong_split_manifest": "SPLIT_MANIFEST_SHA_MISMATCH",
}

G1_NEGATIVE_SUITE_SCHEMA = "c28f_g1_builtin_negative_suite_v2"
G1_REGISTRY_AUTHORITY_SCHEMA = "c28f_g0_expected_g1_registry_authority_v1"
G1_PERSISTED_SNAPSHOT_SCHEMA = "c28f_g1_persisted_global_ledger_snapshot_v2"
STAGE_COMMIT_RECEIPT_SCHEMA = "c28f_a4_stage_commit_receipt_v1"
STAGE_OUTPUT_FILE_MAX_BYTES = 64 * 1024 * 1024
RANK_TRANSITIONS_PARQUET_SCHEMA_VERSION = (
    "c28f_rank_transitions_parquet_v1"
)
FROZEN_PYARROW_VERSION = "21.0.0"
RANK_TRANSITIONS_ROW_GROUP_SIZE = 1_024
F0_MARGIN_FIELDS = (
    "pooled_top1_top2_score_margin",
    "pooled_top10_top11_score_margin",
    "pooled_top100_top101_score_margin",
    "pooled_gt_vs_best_wrong_score_margin",
    "late_top1_top2_score_margin",
    "late_top10_top11_score_margin",
    "late_top100_top101_score_margin",
    "late_gt_vs_best_wrong_score_margin",
    "final_top1_top2_score_margin",
    "final_top10_top11_score_margin",
    "final_top100_top101_score_margin",
    "final_gt_vs_best_wrong_score_margin",
)
F0_COMPARISON_SUMMARY_FIELDS = (
    "schema_version",
    "query_role",
    "query_count",
    "cross_split_execution_invariants",
    "retrieval_metrics",
    "joint_retrieval_metrics",
    "vcmr_metrics",
    "error_metrics",
    "gate_metrics",
    "rank_transition_summary",
    "margin_diagnostics",
    "modality_mask_diagnostics",
    "video_distribution_diagnostics",
    "bucket_distribution_summary",
)
F0_A_FORENSICS_REPORT_SCHEMA = "c28f_f0_a_forensics_report_v2"
F0_B_FORENSICS_REPORT_SCHEMA = "c28f_f0_b_forensics_report_v2"
F0_A_COMMITTED_REPORT_BINDING_FIELDS = (
    "report_path",
    "report_schema_version",
    "report_file_sha256",
    "report_sha256",
    "frozen_gate_result_sha256",
    "earliest_failing_stage",
    "comparison_summary",
    "comparison_summary_sha256",
    "g3_commit_receipt_sha256",
    "g3_state_sha256",
    "g3_event_sha256",
)
F0_COMPLETED_EVAL_SOURCE_ARTIFACT_LINEAGE_SCHEMA = (
    "c28f_f0_completed_eval_source_artifact_lineage_v2"
)
F0_RAW_FORWARD_ARTIFACT_INDEX_SCHEMA = (
    "c28f_a4_raw_forward_artifact_index_v2"
)
F0_RAW_FORWARD_ARTIFACT_INDEX_FIELDS = (
    "schema_version",
    "status",
    "goal_id",
    "attempt_id",
    "run_id",
    "completed_transaction_id",
    "completed_forward_validation_receipt_sha256",
    "forward_run_receipt_sha256",
    "consumer_capability_sha256",
    "consumer_binding_sha256",
    "result_chunk_count",
    "result_chunk_set_sha256",
    "ordered_ranges_sha256",
    "consumed_range_receipt_sha256s",
    "consumed_range_receipt_chain_sha256",
    "all_ranges_consumed",
    "maximum_simultaneously_loaded_chunk_count",
    "payload_retention_contract",
    "cursor_artifact_index_sha256",
    "raw_joint_input_set_sha256",
    "result_chunks",
    "observation_snapshot",
    "encoded_cache_binding",
    "index_sha256",
)
F0_RAW_FORWARD_ARTIFACT_CHUNK_FIELDS = (
    "range_index",
    "start_index",
    "end_index",
    "row_count",
    "artifact_name",
    "artifact_file_sha256",
    "payload_semantic_sha256",
    "engine_registry_consumption_sha256",
    "query_types_sha256",
    "query_rows_sha256",
    "claim_capability_sha256",
    "claim_binding_sha256",
    "materialization_owner_generation",
    "materialization_owner_binding_sha256",
    "materialization_owner_takeover_chain_sha256",
    "owner_lineage_sha256",
    "pre_gt_artifact_name",
    "pre_gt_artifact_file_sha256",
    "pre_gt_payload_sha256",
    "derived_nms_artifact_name",
    "derived_nms_artifact_file_sha256",
    "derived_nms_payload_sha256",
    "postimage_capability_sha256",
    "postimage_binding_sha256",
    "range_consumption_receipt_sha256",
)
F0_DERIVED_NMS_ARTIFACT_INDEX_SCHEMA = (
    "c28f_a4_derived_nms_artifact_index_v2"
)
F0_DERIVED_NMS_ARTIFACT_INDEX_FIELDS = (
    "schema_version",
    "status",
    "goal_id",
    "attempt_id",
    "run_id",
    "completed_transaction_id",
    "source_raw_artifact_index_sha256",
    "rebuild_contract_sha256",
    "consumer_binding_sha256",
    "consumed_range_receipt_chain_sha256",
    "chunk_count",
    "query_count",
    "result_set_sha256",
    "chunks",
    "index_sha256",
)
F0_DERIVED_NMS_ARTIFACT_CHUNK_FIELDS = (
    "range_index",
    "start_index",
    "end_index",
    "row_count",
    "artifact_name",
    "artifact_file_sha256",
    "payload_sha256",
    "source_artifact_file_sha256",
    "source_payload_semantic_sha256",
    "range_consumption_receipt_sha256",
    "postimage_binding_sha256",
)
F0_COMPLETED_EVAL_SOURCE_ARTIFACT_LINEAGE_FIELDS = (
    "schema_version",
    "completed_transaction_id",
    "train_jsonl_index_binding_sha256",
    "train_jsonl_index_receipt_sha256",
    "raw_forward_artifact_index_path",
    "raw_forward_artifact_index_file_sha256",
    "raw_forward_artifact_index_sha256",
    "completed_forward_validation_receipt_sha256",
    "forward_run_receipt_sha256",
    "consumer_capability_sha256",
    "consumer_binding_sha256",
    "result_chunk_count",
    "result_chunk_set_sha256",
    "ordered_ranges_sha256",
    "consumed_range_receipt_sha256s",
    "consumed_range_receipt_chain_sha256",
    "all_ranges_consumed",
    "maximum_simultaneously_loaded_chunk_count",
    "payload_retention_contract",
    "raw_derivation_chunk_index_sha256",
    "raw_joint_input_set_sha256",
    "derived_nms_artifact_index_path",
    "derived_nms_artifact_index_file_sha256",
    "derived_nms_artifact_index_sha256",
    "derived_nms_chunk_index_sha256",
    "derivation_consumer_binding_sha256",
    "derivation_consumed_range_receipt_chain_sha256",
    "derived_nms_rebuild_contract",
    "derived_nms_rebuild_contract_sha256",
    "frozen_nms_result_set_sha256",
    "joint_sufficient_stats_sha256",
    "joint_derivation_receipt_path",
    "joint_derivation_receipt_file_sha256",
    "joint_derivation_receipt_sha256",
    "retrieval_rows_sha256",
    "retrieval_row_count",
    "raw_proposals_scores_or_gt_spans_embedded",
    "lineage_sha256",
)
F0_A_FORENSICS_REPORT_FIELDS = (
    "schema_version",
    "status",
    "goal_id",
    "attempt_id",
    "stage",
    "token_id",
    "eval_id",
    "run_id",
    "query_role",
    "query_manifest_sha256",
    "corpus_manifest_sha256",
    "query_count",
    "query_completeness",
    "committed_query_role_binding",
    "completed_eval_capability_sha256",
    "completed_eval_artifact_binding_sha256",
    "completed_eval_source_artifact_lineage",
    "cross_split_execution_invariants",
    "metric_sufficient_stat_contract_sha256",
    "retrieval_metrics",
    "joint_retrieval_metrics",
    "vcmr_metrics",
    "error_metrics",
    "gate_metrics",
    "rank_transition_summary",
    "margin_diagnostics",
    "modality_mask_diagnostics",
    "video_distribution_diagnostics",
    "denominators",
    "frozen_gate_result",
    "frozen_gate_result_sha256",
    "earliest_failing_stage",
    "comparison_summary",
    "comparison_summary_sha256",
    "teacher_shadow_status",
    "intrinsic_diagnostics",
    "calibration_status",
    "artifact_commitments",
    "safety_counts",
    "threshold_changed_after_observation",
    "report_sha256",
)
F0_B_FORENSICS_REPORT_FIELDS = (
    "schema_version",
    "status",
    "goal_id",
    "attempt_id",
    "stage",
    "token_id",
    "eval_id",
    "run_id",
    "query_role",
    "query_manifest_sha256",
    "corpus_manifest_sha256",
    "query_count",
    "query_completeness",
    "committed_query_role_binding",
    "completed_eval_capability_sha256",
    "completed_eval_artifact_binding_sha256",
    "completed_eval_source_artifact_lineage",
    "cross_split_execution_invariants",
    "metric_sufficient_stat_contract_sha256",
    "retrieval_metrics",
    "joint_retrieval_metrics",
    "vcmr_metrics",
    "error_metrics",
    "gate_metrics",
    "rank_transition_summary",
    "margin_diagnostics",
    "modality_mask_diagnostics",
    "video_distribution_diagnostics",
    "denominators",
    "frozen_gate_result",
    "frozen_gate_result_sha256",
    "earliest_failing_stage",
    "comparison_summary",
    "comparison_summary_sha256",
    "bucket_report",
    "predecessor_f0_a_binding",
    "split_shift_report_sha256",
    "routing_status",
    "teacher_shadow_status",
    "intrinsic_diagnostics",
    "calibration_status",
    "artifact_commitments",
    "safety_counts",
    "threshold_changed_after_observation",
    "report_sha256",
)
F0_SPLIT_SHIFT_REPORT_FIELDS = (
    "schema_version",
    "status",
    "f0_a_committed_report_binding",
    "f0_b_completed_eval_capability_sha256",
    "f0_b_comparison_summary",
    "f0_b_comparison_summary_sha256",
    "query_count_shift",
    "cross_split_execution_invariants",
    "cross_split_execution_invariants_match",
    "retrieval_metric_shift",
    "joint_retrieval_metric_shift",
    "vcmr_metric_shift",
    "error_metric_shift",
    "gate_metric_shift",
    "rank_transition_shift",
    "margin_shift",
    "modality_mask_shift",
    "video_distribution_shift",
    "bucket_shift",
    "earliest_failing_stage",
    "teacher_status",
    "gt_support_count",
    "training_exposure_boundary",
    "threshold_changed_after_observation",
    "routing_status",
    "split_shift_report_sha256",
)
F0_ROUTING_DECISION_FIELDS = (
    "schema_version",
    "status",
    "f0_a_frozen_gate_result_sha256",
    "f0_b_frozen_gate_result_sha256",
    "f0_a_earliest_failing_stage",
    "f0_b_earliest_failing_stage",
    "selected_earliest_failing_stage",
    "next_stage_recommendation_only",
    "split_shift_report_sha256",
    "threshold_changed_after_observation",
    "f2_or_higher_executed",
    "ambiguous_is_valid_scientific_result",
    "routing_decision_sha256",
)

G1_NEGATIVE_SUITE_MANIFEST = {
    "schema_version": G1_NEGATIVE_SUITE_SCHEMA,
    "implementation": "BUILTIN_NONINJECTABLE_ACTUAL_GUARDS_AND_DETECTORS",
    "probe_reason_codes": dict(sorted(REQUIRED_G1_NEGATIVE_PROBES.items())),
    "io_tripwire": {
        "data_loader_calls_allowed": 0,
        "content_open_calls_allowed": 0,
        "model_forward_calls_allowed": 0,
        "output_write_calls_allowed": 0,
    },
}
G1_NEGATIVE_SUITE_SHA256 = semantic_sha256(G1_NEGATIVE_SUITE_MANIFEST)


class StageContractError(RuntimeError):
    """A fail-closed integration or transaction-plan violation."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise StageContractError(message)


def _sha256(value: Any, field: str) -> str:
    _require(isinstance(value, str) and len(value) == 64, "%s must be SHA-256" % field)
    _require(all(character in "0123456789abcdef" for character in value), "%s must be lowercase hexadecimal" % field)
    return str(value)


def _canonical_file(value: Mapping[str, Any]) -> bytes:
    return canonical_json_bytes(dict(value)) + b"\n"


def _canonical_jsonl(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join(canonical_json_bytes(dict(row)) + b"\n" for row in rows)


def _render_summary_markdown(
    *,
    title: str,
    authoritative_content_sha256: str,
    rows: Sequence[tuple[str, Any]],
) -> bytes:
    """Render a non-authoritative Markdown view from frozen JSON scalars."""

    _sha256(authoritative_content_sha256, "authoritative_content_sha256")
    _require(
        isinstance(title, str)
        and title
        and "\n" not in title
        and all(
            isinstance(label, str)
            and label
            and "\n" not in label
            and isinstance(value, (str, int, float, bool, type(None)))
            for label, value in rows
        ),
        "Markdown summary rows are not frozen scalars",
    )
    lines = [
        "# %s" % title,
        "",
        "Authoritative JSON content SHA-256: `%s`." % authoritative_content_sha256,
        "",
    ]
    for label, value in rows:
        if value is None:
            rendered = "null"
        elif type(value) is bool:
            rendered = "true" if value else "false"
        else:
            rendered = str(value)
        _require("\n" not in rendered, "Markdown summary value is multiline")
        lines.append("- %s: `%s`" % (label, rendered.replace("`", "\\`")))
    lines.append("")
    return "\n".join(lines).encode("utf-8")


def _frozen_retrieval_gate_result(
    *,
    retrieval: Mapping[str, Any],
    query_manifest_sha256: str,
    completed_eval_capability_sha256: str,
) -> Mapping[str, Any]:
    """Apply the pre-registered F0 routing gates without post-hoc changes."""

    _sha256(query_manifest_sha256, "query_manifest_sha256")
    _sha256(
        completed_eval_capability_sha256,
        "completed_eval_capability_sha256",
    )
    gate_metrics = dict(retrieval.get("gate_metrics", {}))
    _require(
        set(gate_metrics)
        == {
            "G_broad",
            "G_late",
            "G_keep",
            "G_front",
            "L_lost",
            "G_late_product_residual",
            "G_late_product_invariant",
        },
        "F0 retrieval gate metric fields mismatch",
    )
    for field in ("G_broad", "G_late", "G_front"):
        value = gate_metrics[field]
        _require(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
            and 0.0 <= float(value) <= 1.0,
            "F0 retrieval gate metric invalid: %s" % field,
        )
    keep = gate_metrics["G_keep"]
    lost = gate_metrics["L_lost"]
    _require(
        (keep is None and lost is None)
        or (
            isinstance(keep, (int, float))
            and not isinstance(keep, bool)
            and math.isfinite(float(keep))
            and 0.0 <= float(keep) <= 1.0
            and isinstance(lost, (int, float))
            and not isinstance(lost, bool)
            and math.isfinite(float(lost))
            and 0.0 <= float(lost) <= 1.0
            and abs(float(lost) - (1.0 - float(keep))) <= 1e-12
        ),
        "F0 conditional retention/lost metrics invalid",
    )
    residual = gate_metrics["G_late_product_residual"]
    zero_broad_not_applicable = (
        float(gate_metrics["G_broad"]) == 0.0
        and float(gate_metrics["G_late"]) == 0.0
        and keep is None
        and lost is None
        and residual is None
        and gate_metrics["G_late_product_invariant"] is False
    )
    product_invariant_pass = zero_broad_not_applicable or (
        isinstance(residual, (int, float))
        and not isinstance(residual, bool)
        and math.isfinite(float(residual))
        and 0.0 <= float(residual) <= 1e-8
        and gate_metrics["G_late_product_invariant"] is True
    )
    _require(product_invariant_pass, "F0 G_late product invariant failed")
    broad_pass = float(gate_metrics["G_broad"]) >= 0.90
    keep_pass = keep is not None and float(keep) >= 0.90
    front_pass = float(gate_metrics["G_front"]) >= 0.80
    if not broad_pass:
        earliest = "POOLED_BROAD"
        next_recommendation = "F2-B"
    elif not keep_pass:
        earliest = "LATE_RERANK"
        next_recommendation = "F2-L"
    elif not front_pass:
        earliest = "FINAL_VIDEO_ORDERING"
        next_recommendation = "F2-O"
    else:
        earliest = "NO_FAILURE"
        next_recommendation = "F3_PROPOSAL_READINESS"
    base = {
        "schema_version": "c28f_f0_frozen_retrieval_gate_result_v1",
        "status": "VALIDATED_PRE_REGISTERED_ROUTE_RESULT",
        "query_manifest_sha256": query_manifest_sha256,
        "completed_eval_capability_sha256": completed_eval_capability_sha256,
        "thresholds": {
            "G_broad_minimum": 0.90,
            "G_keep_minimum": 0.90,
            "G_front_minimum": 0.80,
            "G_late_product_absolute_tolerance": 1e-8,
        },
        "gate_metrics": gate_metrics,
        "gate_pass": {
            "G_broad": broad_pass,
            "G_keep": keep_pass,
            "G_front": front_pass,
            "G_late_product_invariant": product_invariant_pass,
        },
        "G_late_product_check_status": (
            "NOT_APPLICABLE_ZERO_BROAD_HITS"
            if zero_broad_not_applicable
            else "PASS"
        ),
        "earliest_failing_stage": earliest,
        "next_stage_recommendation_only": next_recommendation,
        "threshold_changed_after_observation": False,
    }
    return {**base, "gate_result_sha256": semantic_sha256(base)}


def _frozen_pyarrow_modules() -> tuple[Any, Any]:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise StageContractError(
            "pyarrow is required to encode the mandated rank_transitions.parquet"
        ) from exc
    _require(
        getattr(pa, "__version__", None) == FROZEN_PYARROW_VERSION,
        "pyarrow version differs from the frozen Parquet writer",
    )
    return pa, pq


def _rank_transitions_parquet(
    *,
    per_query: Sequence[Mapping[str, Any]],
    source_binding_sha256: str,
) -> tuple[bytes, Mapping[str, Any]]:
    """Encode the frozen small rank-transition table as genuine Parquet."""

    _sha256(source_binding_sha256, "rank transition source binding SHA")
    pa, pq = _frozen_pyarrow_modules()
    columns = (
        "query_id",
        "pooled_gt_rank",
        "late_gt_rank",
        "final_video_gt_rank",
        "pooled_to_late_rank_delta",
        "broad_hit_at_1000",
        "late_hit_at_200",
        "broad_hit_but_late_lost",
    )
    normalized = []
    observed_query_ids: list[int] = []
    for index, raw in enumerate(per_query):
        _require(isinstance(raw, Mapping), "rank transition row must be a mapping")
        query_id = raw.get("query_id")
        _require(
            type(query_id) is int and query_id >= 0,
            "rank transition query_id must be a non-negative exact integer",
        )
        ranks = {
            field: raw.get(field)
            for field in (
                "pooled_gt_rank",
                "late_gt_rank",
                "final_video_gt_rank",
            )
        }
        _require(
            all(
                value is None
                or (
                    type(value) is int
                    and 1 <= value <= 2_147_483_647
                )
                for value in ranks.values()
            ),
            "rank transition rank value invalid at %d" % index,
        )
        expected_delta = (
            ranks["pooled_gt_rank"] - ranks["late_gt_rank"]
            if ranks["pooled_gt_rank"] is not None
            and ranks["late_gt_rank"] is not None
            else None
        )
        _require(
            raw.get("pooled_to_late_rank_delta") == expected_delta
            and type(raw.get("broad_hit_at_1000")) is bool
            and type(raw.get("late_hit_at_200")) is bool
            and type(raw.get("broad_hit_but_late_lost")) is bool
            and raw["broad_hit_at_1000"]
            == (
                ranks["pooled_gt_rank"] is not None
                and ranks["pooled_gt_rank"] <= 1_000
            )
            and raw["late_hit_at_200"]
            == (
                ranks["late_gt_rank"] is not None
                and ranks["late_gt_rank"] <= 200
            )
            and raw["broad_hit_but_late_lost"]
            == (raw["broad_hit_at_1000"] and not raw["late_hit_at_200"]),
            "rank transition derived fields mismatch at %d" % index,
        )
        observed_query_ids.append(query_id)
        normalized.append({field: raw.get(field) for field in columns})
    _require(
        observed_query_ids == sorted(set(observed_query_ids))
        and bool(observed_query_ids),
        "rank transition query order/universe is not canonical",
    )
    rows_sha = bytes_sha256(canonical_json_bytes(normalized))
    metadata = {
        b"c28f.schema_version": RANK_TRANSITIONS_PARQUET_SCHEMA_VERSION.encode(
            "ascii"
        ),
        b"c28f.source_binding_sha256": source_binding_sha256.encode("ascii"),
        b"c28f.rows_sha256": rows_sha.encode("ascii"),
        b"c28f.column_order": ",".join(columns).encode("ascii"),
        b"c28f.pyarrow_version": FROZEN_PYARROW_VERSION.encode("ascii"),
    }
    schema = pa.schema(
        [
            pa.field("query_id", pa.int64(), nullable=False),
            pa.field("pooled_gt_rank", pa.int32(), nullable=True),
            pa.field("late_gt_rank", pa.int32(), nullable=True),
            pa.field("final_video_gt_rank", pa.int32(), nullable=True),
            pa.field("pooled_to_late_rank_delta", pa.int32(), nullable=True),
            pa.field("broad_hit_at_1000", pa.bool_(), nullable=False),
            pa.field("late_hit_at_200", pa.bool_(), nullable=False),
            pa.field("broad_hit_but_late_lost", pa.bool_(), nullable=False),
        ],
        metadata=metadata,
    )
    table = pa.Table.from_pylist(normalized, schema=schema)
    sink = pa.BufferOutputStream()
    pq.write_table(
        table,
        sink,
        row_group_size=RANK_TRANSITIONS_ROW_GROUP_SIZE,
        version="2.6",
        use_dictionary=False,
        compression=None,
        write_statistics=True,
        data_page_size=1_048_576,
        data_page_version="1.0",
        use_compliant_nested_type=True,
        write_batch_size=1_024,
        dictionary_pagesize_limit=None,
        store_schema=True,
        write_page_index=False,
        write_page_checksum=False,
        sorting_columns=None,
        store_decimal_as_integer=False,
    )
    encoded = sink.getvalue().to_pybytes()
    _require(encoded.startswith(b"PAR1") and encoded.endswith(b"PAR1"), "invalid Parquet framing")
    parquet_file = pq.ParquetFile(pa.BufferReader(encoded))
    expected_row_groups = (
        len(normalized) + RANK_TRANSITIONS_ROW_GROUP_SIZE - 1
    ) // RANK_TRANSITIONS_ROW_GROUP_SIZE
    _require(
        parquet_file.metadata.num_rows == len(normalized)
        and parquet_file.metadata.num_columns == len(columns)
        and parquet_file.metadata.num_row_groups == expected_row_groups
        and parquet_file.schema_arrow.names == list(columns)
        and parquet_file.read().to_pylist() == normalized,
        "Parquet read-back row/schema contract mismatch",
    )
    for row_group_index in range(expected_row_groups):
        row_group = parquet_file.metadata.row_group(row_group_index)
        expected_group_rows = min(
            RANK_TRANSITIONS_ROW_GROUP_SIZE,
            len(normalized)
            - row_group_index * RANK_TRANSITIONS_ROW_GROUP_SIZE,
        )
        _require(
            row_group.num_rows == expected_group_rows,
            "Parquet row-group size drift",
        )
        for column_index in range(row_group.num_columns):
            column = row_group.column(column_index)
            _require(
                "RLE_DICTIONARY" not in column.encodings
                and "PLAIN_DICTIONARY" not in column.encodings
                and column.statistics is not None,
                "Parquet dictionary/statistics contract drift",
            )
    contract_base = {
        "schema_version": RANK_TRANSITIONS_PARQUET_SCHEMA_VERSION,
        "status": "GENUINE_PARQUET_FROZEN_WRITER",
        "source_binding_sha256": source_binding_sha256,
        "semantic_rows_sha256": rows_sha,
        "row_count": len(normalized),
        "column_order": list(columns),
        "column_types": {
            "query_id": "int64_nonnull",
            "pooled_gt_rank": "int32_nullable",
            "late_gt_rank": "int32_nullable",
            "final_video_gt_rank": "int32_nullable",
            "pooled_to_late_rank_delta": "int32_nullable",
            "broad_hit_at_1000": "bool_nonnull",
            "late_hit_at_200": "bool_nonnull",
            "broad_hit_but_late_lost": "bool_nonnull",
        },
        "writer": {
            "pyarrow_version": FROZEN_PYARROW_VERSION,
            "parquet_version": "2.6",
            "compression": "NONE",
            "use_dictionary": False,
            "write_statistics": True,
            "row_group_size": RANK_TRANSITIONS_ROW_GROUP_SIZE,
            "data_page_size": 1_048_576,
            "data_page_version": "1.0",
            "write_batch_size": 1_024,
            "write_page_index": False,
            "write_page_checksum": False,
        },
        "parquet_file_sha256": bytes_sha256(encoded),
        "parquet_size_bytes": len(encoded),
    }
    return encoded, {
        **contract_base,
        "contract_sha256": semantic_sha256(contract_base),
    }


def _f0_mean(values: Sequence[float]) -> Optional[float]:
    return math.fsum(float(value) for value in values) / float(len(values)) if values else None


def _f0_recall(ranks: Sequence[Optional[int]], k: int) -> float:
    _require(bool(ranks), "F0 recall denominator cannot be zero")
    return sum(rank is not None and rank <= k for rank in ranks) / float(len(ranks))


def _f0_margin_diagnostics(
    rows: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any]:
    result: dict[str, Any] = {}
    for prefix in ("pooled", "late", "final"):
        gt_field = "%s_gt_vs_best_wrong_score_margin" % prefix
        cut_fields = {
            "at_1": "%s_top1_top2_score_margin" % prefix,
            "at_10": "%s_top10_top11_score_margin" % prefix,
            "at_100": "%s_top100_top101_score_margin" % prefix,
        }
        cut_values = {
            cut: [float(row[field]) for row in rows if row[field] is not None]
            for cut, field in cut_fields.items()
        }
        gt_values = [float(row[gt_field]) for row in rows if row[gt_field] is not None]
        result[prefix] = {
            "status": "available" if any(cut_values.values()) or gt_values else "unavailable",
            "reason": None if any(cut_values.values()) or gt_values else "NO_CONTROLLER_DERIVED_MARGIN_SCALARS",
            "cut_margin_definition": "score[k]-score[k+1]_ONE_BASED",
            "top1_top2_available_query_count": len(cut_values["at_1"]),
            "top10_top11_available_query_count": len(cut_values["at_10"]),
            "top100_top101_available_query_count": len(cut_values["at_100"]),
            "gt_vs_best_wrong_available_query_count": len(gt_values),
            "mean_top1_top2_score_margin": _f0_mean(cut_values["at_1"]),
            "mean_top10_top11_score_margin": _f0_mean(cut_values["at_10"]),
            "mean_top100_top101_score_margin": _f0_mean(cut_values["at_100"]),
            "mean_gt_vs_best_wrong_score_margin": _f0_mean(gt_values),
        }
    return result


def _validate_f0_joint_sufficient_row(row: Mapping[str, Any], index: int) -> None:
    _require(
        type(row["top1_available"]) is bool
        and type(row["wrong_video_top1"]) is bool
        and type(row["correct_video_wrong_span_top1"]) is bool
        and type(row["top1_false_positive"]) is bool
        and type(row["raw_exact_duplicate_proposal_count"]) is int
        and 0 <= row["raw_exact_duplicate_proposal_count"] <= 12_800
        and type(row["nms_prediction_count"]) is int
        and 0 <= row["nms_prediction_count"] <= 100,
        "F0 joint boolean/count sufficient statistics invalid at %d" % index,
    )
    joint_rank = row["joint_marginal_gt_rank"]
    legacy_rank = row["legacy_joint_unique_video_gt_rank"]
    gt_is_joint_candidate = row["gt_video_id"] in row["final_video_order"]
    _require(
        ((joint_rank is not None) is gt_is_joint_candidate)
        and (
            joint_rank is None
            or (type(joint_rank) is int and 1 <= joint_rank <= 200)
        ),
        "F0 joint marginal rank invalid at %d" % index,
    )
    _require(
        legacy_rank is None
        or (
            type(legacy_rank) is int
            and 1 <= legacy_rank <= 100
            and legacy_rank <= row["nms_prediction_count"]
            and gt_is_joint_candidate
        ),
        "F0 legacy NMS unique-video rank invalid at %d" % index,
    )
    top1_iou = row["top1_iou"]
    false_positive_mass = row["joint_false_positive_mass"]
    _require(
        isinstance(top1_iou, (int, float))
        and not isinstance(top1_iou, bool)
        and math.isfinite(float(top1_iou))
        and 0.0 <= float(top1_iou) <= 1.0
        and (
            false_positive_mass is None
            or (
                isinstance(false_positive_mass, (int, float))
                and not isinstance(false_positive_mass, bool)
                and math.isfinite(float(false_positive_mass))
                and 0.0 <= float(false_positive_mass) <= 1.0
            )
        ),
        "F0 joint IoU/mass sufficient statistics invalid at %d" % index,
    )
    available = row["top1_available"]
    wrong_video = row["wrong_video_top1"]
    correct_video_wrong_span = row["correct_video_wrong_span_top1"]
    top1_false_positive = row["top1_false_positive"]
    _require(
        available == (row["nms_prediction_count"] > 0)
        and ((false_positive_mass is not None) is available)
        and (available or wrong_video)
        and (not wrong_video or float(top1_iou) == 0.0)
        and correct_video_wrong_span
        == (available and not wrong_video and float(top1_iou) < 0.5)
        and top1_false_positive
        == ((not available) or wrong_video or float(top1_iou) < 0.3),
        "F0 disjoint top1 error semantics mismatch at %d" % index,
    )
    hits = row["vcmr_hits"]
    expected_hit_keys = {
        "VCMR_R@%d_IoU@%.1f" % (k, threshold)
        for threshold in (0.5, 0.7)
        for k in (1, 5, 10, 100)
    }
    _require(
        isinstance(hits, Mapping)
        and set(hits) == expected_hit_keys
        and all(type(value) is bool for value in hits.values()),
        "F0 VCMR hit sufficient statistics invalid at %d" % index,
    )
    for threshold in (0.5, 0.7):
        ordered = [hits["VCMR_R@%d_IoU@%.1f" % (k, threshold)] for k in (1, 5, 10, 100)]
        _require(
            ordered == sorted(ordered),
            "F0 VCMR hit monotonicity mismatch at %d" % index,
        )
    for k in (1, 5, 10, 100):
        _require(
            not hits["VCMR_R@%d_IoU@0.7" % k]
            or hits["VCMR_R@%d_IoU@0.5" % k],
            "F0 VCMR IoU threshold monotonicity mismatch at %d" % index,
        )
    _require(
        hits["VCMR_R@1_IoU@0.5"]
        == (available and not wrong_video and float(top1_iou) >= 0.5)
        and hits["VCMR_R@1_IoU@0.7"]
        == (available and not wrong_video and float(top1_iou) >= 0.7),
        "F0 VCMR top1 hit/IoU mismatch at %d" % index,
    )


def _validate_f0_modality_mask_coverage(
    value: Any,
    index: int,
) -> Mapping[str, Any]:
    coverage = _exact_mapping(
        value,
        (
            "ratio_definition",
            "structure_contract",
            "joint_component_relation_verified",
            "visual_subtitle_overlap_valid_count",
            "visual",
            "subtitle",
            "joint",
        ),
        "F0 modality mask coverage fields invalid at %d" % index,
    )
    structure_contract = _exact_mapping(
        coverage["structure_contract"],
        (
            "schema_version",
            "visual_candidate_rule",
            "subtitle_candidate_rule",
            "joint_component_rule",
            "joint_candidate_rule",
            "contract_sha256",
        ),
        "F0 modality mask structure contract fields invalid at %d" % index,
    )
    _require(
        coverage["ratio_definition"]
        == "VALID_OR_MISSING_BOOL_CELLS_DIVIDED_BY_CANDIDATE_COUNT_TIMES_64",
        "F0 modality mask ratio definition mismatch at %d" % index,
    )
    _require(
        structure_contract["schema_version"]
        == "c28f_a4_forward_mask_structure_contract_v1"
        and structure_contract["visual_candidate_rule"]
        == "EVERY_CANDIDATE_HAS_AT_LEAST_ONE_VALID_CELL"
        and structure_contract["subtitle_candidate_rule"]
        == "ZERO_VALID_CELLS_PER_CANDIDATE_AND_ALL_ZERO_QUERY_ALLOWED"
        and structure_contract["joint_component_rule"]
        == "EXACT_LOGICAL_OR_VISUAL_SUBTITLE"
        and structure_contract["joint_candidate_rule"]
        == "EVERY_CANDIDATE_HAS_AT_LEAST_ONE_VALID_CELL"
        and structure_contract["contract_sha256"]
        == semantic_sha256(
            structure_contract,
            excluded_fields=("contract_sha256",),
        )
        and coverage["joint_component_relation_verified"] is True,
        "F0 modality mask structural relation mismatch at %d" % index,
    )
    detached: dict[str, Any] = {
        "ratio_definition": coverage["ratio_definition"],
        "structure_contract": dict(structure_contract),
        "joint_component_relation_verified": True,
        "visual_subtitle_overlap_valid_count": coverage[
            "visual_subtitle_overlap_valid_count"
        ],
    }
    for modality in ("visual", "subtitle", "joint"):
        record = _exact_mapping(
            coverage[modality],
            (
                "candidate_count",
                "denominator",
                "valid_count",
                "missing_count",
                "valid_ratio",
                "missing_ratio",
                "empty_candidate_count",
            ),
            "F0 %s mask fields invalid at %d" % (modality, index),
        )
        candidate_count = record["candidate_count"]
        denominator = record["denominator"]
        valid_count = record["valid_count"]
        missing_count = record["missing_count"]
        valid_ratio = record["valid_ratio"]
        missing_ratio = record["missing_ratio"]
        empty_candidate_count = record["empty_candidate_count"]
        _require(
            type(candidate_count) is int
            and candidate_count == 200
            and type(denominator) is int
            and denominator == 12_800
            and type(valid_count) is int
            and type(missing_count) is int
            and type(empty_candidate_count) is int
            and valid_count >= 0
            and missing_count >= 0
            and valid_count + missing_count == denominator
            and 0 <= empty_candidate_count <= candidate_count
            and valid_count >= candidate_count - empty_candidate_count
            and valid_count
            <= (candidate_count - empty_candidate_count) * 64
            and isinstance(valid_ratio, (int, float))
            and not isinstance(valid_ratio, bool)
            and isinstance(missing_ratio, (int, float))
            and not isinstance(missing_ratio, bool)
            and math.isfinite(float(valid_ratio))
            and math.isfinite(float(missing_ratio))
            and abs(float(valid_ratio) - valid_count / float(denominator))
            <= 1e-15
            and abs(float(missing_ratio) - missing_count / float(denominator))
            <= 1e-15
            and abs(float(valid_ratio) + float(missing_ratio) - 1.0)
            <= 1e-15,
            "F0 %s mask count/ratio closure invalid at %d" % (modality, index),
        )
        detached[modality] = dict(record)
    overlap_valid_count = detached["visual_subtitle_overlap_valid_count"]
    _require(
        detached["visual"]["empty_candidate_count"] == 0
        and detached["joint"]["empty_candidate_count"] == 0
        and type(overlap_valid_count) is int
        and 0 <= overlap_valid_count
        <= min(
            detached["visual"]["valid_count"],
            detached["subtitle"]["valid_count"],
        )
        and detached["joint"]["valid_count"]
        == detached["visual"]["valid_count"]
        + detached["subtitle"]["valid_count"]
        - overlap_valid_count,
        "F0 modality mask OR-count closure invalid at %d" % index,
    )
    return detached


def _evaluate_completed_f0_rows(
    rows: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any]:
    """Aggregate controller-derived safe rows without accepting raw proposals."""

    _require(bool(rows), "completed F0 row set cannot be empty")
    query_ids = [row.get("query_id") for row in rows]
    _require(
        all(type(query_id) is int and query_id >= 0 for query_id in query_ids)
        and query_ids == sorted(set(query_ids)),
        "completed F0 query IDs must be canonical unique integers",
    )
    try:
        retrieval = evaluate_true_video_retrieval(
            rows,
            expected_query_ids=query_ids,
        )
    except Exception as exc:
        raise StageContractError("completed F0 retrieval sufficient statistics invalid") from exc
    source_by_id = {row["query_id"]: row for row in rows}
    per_query = []
    modality_coverage_rows: list[Mapping[str, Any]] = []
    for index, raw_metric in enumerate(retrieval["per_query"]):
        source = source_by_id[raw_metric["query_id"]]
        _require(
            type(source["query_feature_token_count"]) is int
            and source["query_feature_token_count"] > 0
            and source["query_feature_token_count"]
            == source["late_maxsim_token_count"]
            and type(source["query_text_token_count"]) is int
            and source["query_text_token_count"] > 0
            and source["query_type"] in {"v", "t", "vt", "unknown"}
            and source["verb_bucket"]
            in {"ZERO_VERB", "SINGLE_VERB", "MULTI_VERB", "UNKNOWN"}
            and type(source["temporal_connector_bool"]) is bool
            and isinstance(source["moment_duration_sec"], (int, float))
            and not isinstance(source["moment_duration_sec"], bool)
            and math.isfinite(float(source["moment_duration_sec"]))
            and float(source["moment_duration_sec"]) > 0.0
            and type(source["gt_video_clip_count"]) is int
            and source["gt_video_clip_count"] > 0
            and isinstance(source["gt_relative_position"], (int, float))
            and not isinstance(source["gt_relative_position"], bool)
            and math.isfinite(float(source["gt_relative_position"]))
            and 0.0 <= float(source["gt_relative_position"]) <= 1.0,
            "F0 frozen scalar/bucket sufficient-stat contract mismatch at %d"
            % index,
        )
        for prefix, ranking_field in (
            ("pooled", "pooled_video_order"),
            ("late", "late_video_order"),
            ("final", "final_video_order"),
        ):
            gt_field = "%s_gt_vs_best_wrong_score_margin" % prefix
            cut_values = [
                source["%s_%s_score_margin" % (prefix, suffix)]
                for suffix in (
                    "top1_top2",
                    "top10_top11",
                    "top100_top101",
                )
            ]
            gt_value = source[gt_field]
            gt_present = source["gt_video_id"] in source[ranking_field]
            _require(
                all(
                    isinstance(value, (int, float))
                    and not isinstance(value, bool)
                    and math.isfinite(float(value))
                    and float(value) >= 0.0
                    for value in cut_values
                )
                and (
                    (gt_present and isinstance(gt_value, (int, float)) and not isinstance(gt_value, bool) and math.isfinite(float(gt_value)))
                    or (not gt_present and gt_value is None)
                ),
                "F0 controller-derived margin/null contract mismatch at %d" % index,
            )
        _validate_f0_joint_sufficient_row(source, index)
        modality_coverage = _validate_f0_modality_mask_coverage(
            source["late_candidate_mask_coverage"],
            index,
        )
        modality_coverage_rows.append(modality_coverage)
        current = dict(raw_metric)
        current.pop("gt_video_id", None)
        for field in F0_MARGIN_FIELDS:
            current[field] = source[field]
        for field in (
            "joint_marginal_gt_rank",
            "legacy_joint_unique_video_gt_rank",
            "top1_iou",
            "top1_available",
            "wrong_video_top1",
            "correct_video_wrong_span_top1",
            "top1_false_positive",
            "joint_false_positive_mass",
            "raw_exact_duplicate_proposal_count",
            "nms_prediction_count",
        ):
            current[field] = source[field]
        current["vcmr_hits"] = dict(source["vcmr_hits"])
        current["modality_mask_coverage"] = modality_coverage
        per_query.append(current)

    query_count = len(per_query)
    marginal_ranks = [row["joint_marginal_gt_rank"] for row in per_query]
    legacy_ranks = [row["legacy_joint_unique_video_gt_rank"] for row in per_query]
    joint_retrieval_metrics = {
        **{
            "joint_marginal_VR_R@%d" % k: _f0_recall(marginal_ranks, k)
            for k in (1, 5, 10, 50, 100, 200)
        },
        **{
            "legacy_joint_unique_video_R@%d" % k: _f0_recall(legacy_ranks, k)
            for k in (1, 5, 10, 100)
        },
    }
    vcmr_metrics = {
        key: sum(bool(row["vcmr_hits"][key]) for row in per_query) / float(query_count)
        for key in sorted(per_query[0]["vcmr_hits"])
    }
    top1_ious = [float(row["top1_iou"]) for row in per_query]
    sorted_ious = sorted(top1_ious)
    middle = query_count // 2
    median_iou = (
        sorted_ious[middle]
        if query_count % 2
        else (sorted_ious[middle - 1] + sorted_ious[middle]) / 2.0
    )
    vcmr_metrics.update(
        {
            "VCMR_top1_IoU_mean": _f0_mean(top1_ious),
            "VCMR_top1_IoU_median": median_iou,
        }
    )
    mass_values = [
        float(row["joint_false_positive_mass"])
        for row in per_query
        if row["joint_false_positive_mass"] is not None
    ]
    error_metrics = {
        "wrong_video_top1_rate": sum(bool(row["wrong_video_top1"]) for row in per_query) / float(query_count),
        "correct_video_wrong_span_top1_rate": sum(bool(row["correct_video_wrong_span_top1"]) for row in per_query) / float(query_count),
        "top1_false_positive_rate": sum(bool(row["top1_false_positive"]) for row in per_query) / float(query_count),
        "joint_false_positive_mass_mean": _f0_mean(mass_values),
        "top1_missing_rate": sum(not bool(row["top1_available"]) for row in per_query) / float(query_count),
        "raw_exact_duplicate_proposal_count": sum(int(row["raw_exact_duplicate_proposal_count"]) for row in per_query),
        "invalid_span_count": 0,
    }
    modality_mask_diagnostics = {
        "ratio_definition": modality_coverage_rows[0]["ratio_definition"],
        "structure_contract": dict(
            modality_coverage_rows[0]["structure_contract"]
        ),
        "joint_component_relation_verified": all(
            row["joint_component_relation_verified"] is True
            for row in modality_coverage_rows
        ),
        "visual_subtitle_overlap_valid_count": sum(
            int(row["visual_subtitle_overlap_valid_count"])
            for row in modality_coverage_rows
        ),
        **{
            modality: {
                "query_count": query_count,
                "candidate_count": sum(
                    int(row[modality]["candidate_count"])
                    for row in modality_coverage_rows
                ),
                "empty_candidate_count": sum(
                    int(row[modality]["empty_candidate_count"])
                    for row in modality_coverage_rows
                ),
                "cell_denominator": sum(
                    int(row[modality]["denominator"])
                    for row in modality_coverage_rows
                ),
                "valid_count": sum(
                    int(row[modality]["valid_count"])
                    for row in modality_coverage_rows
                ),
                "missing_count": sum(
                    int(row[modality]["missing_count"])
                    for row in modality_coverage_rows
                ),
                "mean_valid_ratio": _f0_mean(
                    [
                        float(row[modality]["valid_ratio"])
                        for row in modality_coverage_rows
                    ]
                ),
                "mean_missing_ratio": _f0_mean(
                    [
                        float(row[modality]["missing_ratio"])
                        for row in modality_coverage_rows
                    ]
                ),
                "mean_empty_candidate_ratio": (
                    sum(
                        int(row[modality]["empty_candidate_count"])
                        for row in modality_coverage_rows
                    )
                    / float(
                        sum(
                            int(row[modality]["candidate_count"])
                            for row in modality_coverage_rows
                        )
                    )
                ),
            }
            for modality in ("visual", "subtitle", "joint")
        },
    }
    queries_per_video: dict[str, int] = {}
    for source in rows:
        video_id = str(source["gt_video_id"])
        queries_per_video[video_id] = queries_per_video.get(video_id, 0) + 1
    cluster_sizes = sorted(queries_per_video.values())

    def nearest_rank(fraction: float) -> int:
        index = min(
            len(cluster_sizes) - 1,
            max(0, int(math.ceil(fraction * len(cluster_sizes))) - 1),
        )
        return cluster_sizes[index]

    ordered_video_ids = sorted(queries_per_video, key=lambda value: value.encode("utf-8"))
    video_distribution_diagnostics = {
        "gt_video_cluster_count": len(cluster_sizes),
        "gt_video_id_lines_sha256": bytes_sha256(
            "".join("%s\n" % value for value in ordered_video_ids).encode("utf-8")
        ),
        "queries_per_gt_video": {
            "minimum": cluster_sizes[0],
            "p50_nearest_rank": nearest_rank(0.50),
            "p90_nearest_rank": nearest_rank(0.90),
            "maximum": cluster_sizes[-1],
            "mean": query_count / float(len(cluster_sizes)),
        },
    }
    return {
        "retrieval_metrics": dict(retrieval["metrics"]),
        "joint_retrieval_metrics": joint_retrieval_metrics,
        "vcmr_metrics": vcmr_metrics,
        "error_metrics": error_metrics,
        "gate_metrics": dict(retrieval["gate_metrics"]),
        "rank_transition_summary": dict(retrieval["rank_transition_summary"]),
        "margin_diagnostics": _f0_margin_diagnostics(per_query),
        "modality_mask_diagnostics": modality_mask_diagnostics,
        "video_distribution_diagnostics": video_distribution_diagnostics,
        "denominators": {
            **dict(retrieval["denominators"]),
            "vcmr_denominator": query_count,
            "top1_error_denominator": query_count,
            "top1_available_count": sum(bool(row["top1_available"]) for row in per_query),
            "joint_false_positive_mass_denominator": len(mass_values),
        },
        "per_query": per_query,
    }


def _f0_edge_label(value: float, edges: Sequence[float]) -> str:
    def label(number: float) -> str:
        return format(float(number), ".12g").replace("-", "NEG_").replace(".", "P")

    previous: Optional[float] = None
    for edge in edges:
        if float(value) <= float(edge):
            return (
                "LE_%s" % label(edge)
                if previous is None
                else "GT_%s_LE_%s" % (label(previous), label(edge))
            )
        previous = float(edge)
    _require(previous is not None, "F0 bucket edge set cannot be empty")
    return "GT_%s" % label(previous)


def _validate_f0_bucket_rule_contract(value: Any) -> Mapping[str, Any]:
    contract = _exact_mapping(
        value,
        (
            "schema_version",
            "contract_version",
            "query_feature_token_count_rule",
            "late_maxsim_token_count_rule",
            "feature_to_late_maxsim_invariant",
            "query_text_token_count_rule",
            "verb_bucket_rule",
            "verb_lexicon",
            "temporal_connector_rule",
            "temporal_connector_tokens",
            "duration_bucket_edges_sec",
            "gt_video_clip_count_rule",
            "clip_count_bucket_edges",
            "relative_position_bucket_edges",
            "mask_coverage_bucket_edges",
            "margin_null_rule",
            "teacher_rule",
            "contract_sha256",
        ),
        "F0 bucket rule contract fields mismatch",
    )
    _require(
        contract["schema_version"] == "c28f_a4_eval_bucket_rule_contract_v1"
        and contract["contract_version"] == "FROZEN_PRE_EVAL_V1"
        and contract["query_feature_token_count_rule"]
        == "FORWARD_SEALED_QUERY_MASK_SUM_USED_BY_MODEL"
        and contract["late_maxsim_token_count_rule"]
        == "FORWARD_SEALED_LATE_MAXSIM_QUERY_MASK_SUM"
        and contract["feature_to_late_maxsim_invariant"]
        == "EXACT_SAME_FORWARD_QUERY_MASK_COUNTS_EQUAL"
        and contract["query_text_token_count_rule"]
        == (
            "UNICODE_CASEFOLD_THEN_ASCII_PUNCTUATION_TO_SPACE_"
            "THEN_WHITESPACE_SPLIT_COUNT"
        )
        and contract["duration_bucket_edges_sec"] == [2.0, 5.0, 10.0]
        and contract["gt_video_clip_count_rule"]
        == "MAX_1_CEIL_GT_VIDEO_DURATION_SEC_DIV_1P5"
        and contract["clip_count_bucket_edges"] == [64, 96]
        and contract["relative_position_bucket_edges"]
        == [0.3333333333333333, 0.6666666666666666]
        and contract["mask_coverage_bucket_edges"] == [0.25, 0.5, 0.75, 1.0]
        and contract["margin_null_rule"]
        == "CUT_MARGIN_NONNULL;GT_MARGIN_NULL_IFF_GT_ABSENT"
        and contract["teacher_rule"] == "UNAVAILABLE_ZERO_TEACHER_ACCESS"
        and contract["contract_sha256"]
        == semantic_sha256(contract, excluded_fields=("contract_sha256",)),
        "F0 bucket rule contract differs from the pre-eval blueprint",
    )
    return contract


def _f0_bucket_assignments(
    row: Mapping[str, Any],
    contract: Mapping[str, Any],
) -> Mapping[str, str]:
    def margin_bucket(field: str) -> str:
        margin = row[field]
        return (
            "UNAVAILABLE"
            if margin is None
            else ("ZERO" if float(margin) == 0.0 else "POSITIVE")
        )

    coverage = row["late_candidate_mask_coverage"]
    return {
        "query_type": "QUERY_TYPE_%s" % str(row["query_type"]).upper(),
        "query_feature_token_count": "COUNT_%08d"
        % int(row["query_feature_token_count"]),
        "query_text_token_count": "COUNT_%08d"
        % int(row["query_text_token_count"]),
        "verb_bucket": str(row["verb_bucket"]),
        "temporal_connector": (
            "PRESENT" if row["temporal_connector_bool"] else "ABSENT"
        ),
        "moment_duration_sec": _f0_edge_label(
            float(row["moment_duration_sec"]),
            contract["duration_bucket_edges_sec"],
        ),
        "video_raw_clip_count": _f0_edge_label(
            float(row["gt_video_clip_count"]),
            contract["clip_count_bucket_edges"],
        ),
        "gt_relative_position": _f0_edge_label(
            float(row["gt_relative_position"]),
            contract["relative_position_bucket_edges"],
        ),
        "visual_valid_mask_ratio": _f0_edge_label(
            float(coverage["visual"]["valid_ratio"]),
            contract["mask_coverage_bucket_edges"],
        ),
        "subtitle_valid_mask_ratio": _f0_edge_label(
            float(coverage["subtitle"]["valid_ratio"]),
            contract["mask_coverage_bucket_edges"],
        ),
        "joint_valid_mask_ratio": _f0_edge_label(
            float(coverage["joint"]["valid_ratio"]),
            contract["mask_coverage_bucket_edges"],
        ),
        "late_maxsim_token_count": "COUNT_%08d"
        % int(row["late_maxsim_token_count"]),
        "pooled_cut_margin_at_1": margin_bucket(
            "pooled_top1_top2_score_margin"
        ),
        "pooled_cut_margin_at_10": margin_bucket(
            "pooled_top10_top11_score_margin"
        ),
        "pooled_cut_margin_at_100": margin_bucket(
            "pooled_top100_top101_score_margin"
        ),
    }


def _f0_subset_metrics(rows: Sequence[Mapping[str, Any]], total_count: int) -> Mapping[str, Any]:
    _require(bool(rows) and total_count >= len(rows), "F0 bucket subset denominator invalid")
    count = len(rows)
    broad_count = sum(bool(row["broad_hit_at_1000"]) for row in rows)
    late_count = sum(bool(row["late_hit_at_200"]) for row in rows)
    lost_count = sum(bool(row["broad_hit_but_late_lost"]) for row in rows)
    margin_means = {
        field: _f0_mean(
            [float(row[field]) for row in rows if row[field] is not None]
        )
        for field in F0_MARGIN_FIELDS
    }
    return {
        "query_count": count,
        "query_fraction": count / float(total_count),
        "gate_metrics": {
            "G_broad": broad_count / float(count),
            "G_late": late_count / float(count),
            "G_keep": late_count / float(broad_count) if broad_count else None,
            "G_front": sum(
                row["late_gt_rank"] is not None and row["late_gt_rank"] <= 100
                for row in rows
            )
            / float(count),
            "L_lost": lost_count / float(broad_count) if broad_count else None,
        },
        "true_video_retrieval": {
            "pooled_R@1000": broad_count / float(count),
            "late_R@200": late_count / float(count),
            **{
                "final_video_R@%d" % k: _f0_recall(
                    [row["final_video_gt_rank"] for row in rows], k
                )
                for k in (1, 5, 10, 100)
            },
            **{
                "joint_marginal_VR_R@%d" % k: _f0_recall(
                    [row["joint_marginal_gt_rank"] for row in rows], k
                )
                for k in (1, 5, 10, 100)
            },
        },
        "vcmr": {
            key: sum(bool(row["vcmr_hits"][key]) for row in rows) / float(count)
            for key in sorted(rows[0]["vcmr_hits"])
        },
        "errors": {
            "wrong_video_top1_rate": sum(bool(row["wrong_video_top1"]) for row in rows) / float(count),
            "correct_video_wrong_span_top1_rate": sum(bool(row["correct_video_wrong_span_top1"]) for row in rows) / float(count),
            "top1_false_positive_rate": sum(bool(row["top1_false_positive"]) for row in rows) / float(count),
            "joint_false_positive_mass_mean": _f0_mean(
                [
                    float(row["joint_false_positive_mass"])
                    for row in rows
                    if row["joint_false_positive_mass"] is not None
                ]
            ),
        },
        "rank_transition": {
            "mean_pooled_minus_late_rank_delta": _f0_mean(
                [
                    float(row["pooled_to_late_rank_delta"])
                    for row in rows
                    if row["pooled_to_late_rank_delta"] is not None
                ]
            ),
        },
        "mean_margins": margin_means,
    }


def _build_f0_bucket_report(
    *,
    source_rows: Sequence[Mapping[str, Any]],
    metric_rows: Sequence[Mapping[str, Any]],
    bucket_rule_contract: Mapping[str, Any],
    completed_eval_capability_sha256: str,
    query_manifest_sha256: str,
) -> tuple[Mapping[str, Any], tuple[Mapping[str, Any], ...]]:
    contract = _validate_f0_bucket_rule_contract(bucket_rule_contract)
    _sha256(completed_eval_capability_sha256, "completed eval capability SHA")
    _sha256(query_manifest_sha256, "F0 query manifest SHA")
    _require(
        len(source_rows) == len(metric_rows) > 0
        and [row["query_id"] for row in source_rows]
        == [row["query_id"] for row in metric_rows],
        "F0 bucket source/metric query alignment mismatch",
    )
    enriched_rows = []
    groups: dict[str, dict[str, list[Mapping[str, Any]]]] = {}
    for source, metric in zip(source_rows, metric_rows):
        assignments = _f0_bucket_assignments(source, contract)
        current = {
            "schema_version": "c28f_f0_query_metric_row_v1",
            **dict(metric),
            "bucket_assignments": dict(assignments),
        }
        enriched_rows.append(current)
        for dimension, label in assignments.items():
            groups.setdefault(dimension, {}).setdefault(label, []).append(current)
    dimensions = {
        dimension: {
            label: _f0_subset_metrics(rows, len(enriched_rows))
            for label, rows in sorted(labels.items())
        }
        for dimension, labels in sorted(groups.items())
    }
    base = {
        "schema_version": "c28f_f0_bucket_report_v1",
        "status": "VALIDATED_FROZEN_PRE_EVAL_BUCKETS",
        "query_manifest_sha256": query_manifest_sha256,
        "completed_eval_capability_sha256": completed_eval_capability_sha256,
        "bucket_rule_contract": dict(contract),
        "query_count": len(enriched_rows),
        "dimension_count": len(dimensions),
        "dimensions": dimensions,
        "teacher_rescue_bucket": {
            "status": "unavailable",
            "reason": "UNAVAILABLE_ZERO_TEACHER_ACCESS",
            "teacher_field_access_count": 0,
        },
    }
    return (
        {**base, "bucket_report_sha256": semantic_sha256(base)},
        tuple(enriched_rows),
    )


def _f0_comparison_summary(
    *,
    query_role: str,
    evaluation: Mapping[str, Any],
    bucket_report: Mapping[str, Any],
    execution_binding: Mapping[str, Any],
) -> Mapping[str, Any]:
    invariant_fields = (
        "checkpoint_sha256",
        "corpus_manifest_sha256",
        "evaluator_sha256",
        "cursor_contract_sha256",
        "chunks_contract_sha256",
        "artifacts_contract_sha256",
    )
    _require(
        all(field in execution_binding for field in invariant_fields),
        "F0 execution binding lacks cross-split invariants",
    )
    invariants = {
        field: _sha256(
            execution_binding[field],
            "execution_binding.%s" % field,
        )
        for field in invariant_fields
    }
    summary = {
        "schema_version": "c28f_f0_comparison_summary_v1",
        "query_role": query_role,
        "query_count": evaluation["denominators"]["query_count"],
        "cross_split_execution_invariants": invariants,
        "retrieval_metrics": dict(evaluation["retrieval_metrics"]),
        "joint_retrieval_metrics": dict(evaluation["joint_retrieval_metrics"]),
        "vcmr_metrics": dict(evaluation["vcmr_metrics"]),
        "error_metrics": dict(evaluation["error_metrics"]),
        "gate_metrics": dict(evaluation["gate_metrics"]),
        "rank_transition_summary": dict(evaluation["rank_transition_summary"]),
        "margin_diagnostics": dict(evaluation["margin_diagnostics"]),
        "modality_mask_diagnostics": dict(
            evaluation["modality_mask_diagnostics"]
        ),
        "video_distribution_diagnostics": dict(
            evaluation["video_distribution_diagnostics"]
        ),
        "bucket_distribution_summary": dict(bucket_report["dimensions"]),
    }
    return summary


def _relative_output_path(value: str) -> str:
    _require(isinstance(value, str) and value and not value.startswith("/"), "output path must be relative")
    pieces = value.split("/")
    _require(all(piece not in {"", ".", ".."} for piece in pieces), "output path is non-canonical")
    _require(pieces[0] in {"f0_forensics", "f1_protocol"}, "output path escapes isolated stage roots")
    return value


@dataclass(frozen=True)
class StageContext:
    goal_id: str
    attempt_id: str
    transition_seq: int
    current_action: str
    current_status: str
    parent_state_sha256: str
    parent_event_sha256: str
    business_preimage_sha256: str
    static_review_receipt_sha256: str
    isolated_output_root_id: str
    isolated_output_root_path: str
    expected_negative_suite_sha256: str
    control_transaction_id: str
    committed_control_state_sha256: str
    committed_control_event_sha256: str
    frozen_g1_registry_authority_sha256: str
    frozen_global_ledger_snapshot_sha256: str
    control_anchor_receipt_sha256: str
    train_jsonl_index_state: str
    train_jsonl_index_binding_sha256: Optional[str]
    train_jsonl_index_receipt_sha256: Optional[str]
    train_jsonl_index_transaction_id: Optional[str]

    def validate(self, expected_action: str, expected_status: str) -> None:
        for value, field in (
            (self.goal_id, "goal_id"),
            (self.attempt_id, "attempt_id"),
            (self.isolated_output_root_id, "isolated_output_root_id"),
            (self.isolated_output_root_path, "isolated_output_root_path"),
            (self.control_transaction_id, "control_transaction_id"),
        ):
            _require(isinstance(value, str) and value and value == value.strip(), "%s is required" % field)
        _require(type(self.transition_seq) is int and self.transition_seq > 0, "transition_seq must be a positive exact integer")
        _require(self.current_action == expected_action, "stage action is not the unique current action")
        _require(self.current_status == expected_status, "stage parent status mismatch")
        pre_index_actions = {
            G1_ACTION,
            G2_ACTION,
            G2_TRAIN_JSONL_INDEX_BOOTSTRAP_ACTION,
        }
        post_index_actions = {
            G3_ACTION,
            G4_PROJECTION_ACTION,
            G4_PROJECTION_COMPLETE_ACTION,
            G4_ACTION,
            G5_ACTION,
            G6_ACTION,
            G7_ACTION,
        }
        _require(
            expected_action in pre_index_actions | post_index_actions,
            "stage action lacks a frozen TRAIN-JSONL index policy",
        )
        if expected_action in pre_index_actions:
            _require(
                self.train_jsonl_index_state == TRAIN_JSONL_INDEX_ABSENT
                and self.train_jsonl_index_binding_sha256 is None
                and self.train_jsonl_index_receipt_sha256 is None
                and self.train_jsonl_index_transaction_id is None,
                "pre-index StageContext is not exact ABSENT+None",
            )
        else:
            _require(
                self.train_jsonl_index_state == TRAIN_JSONL_INDEX_READY
                and isinstance(self.train_jsonl_index_transaction_id, str)
                and TRAIN_JSONL_INDEX_TRANSACTION_RE.fullmatch(
                    self.train_jsonl_index_transaction_id
                )
                is not None,
                "post-index StageContext is not exact READY+TRAIN-INDEX",
            )
            _sha256(
                self.train_jsonl_index_binding_sha256,
                "train_jsonl_index_binding_sha256",
            )
            _sha256(
                self.train_jsonl_index_receipt_sha256,
                "train_jsonl_index_receipt_sha256",
            )
        _require(
            self.isolated_output_root_path.startswith("/")
            and ".." not in self.isolated_output_root_path.split("/")
            and not self.isolated_output_root_path.endswith("/"),
            "isolated output root path is non-canonical",
        )
        _sha256(self.parent_state_sha256, "parent_state_sha256")
        _sha256(self.parent_event_sha256, "parent_event_sha256")
        _sha256(self.business_preimage_sha256, "business_preimage_sha256")
        _sha256(self.static_review_receipt_sha256, "static_review_receipt_sha256")
        _sha256(
            self.expected_negative_suite_sha256,
            "expected_negative_suite_sha256",
        )
        _sha256(
            self.committed_control_state_sha256,
            "committed_control_state_sha256",
        )
        _sha256(
            self.committed_control_event_sha256,
            "committed_control_event_sha256",
        )
        _sha256(
            self.frozen_g1_registry_authority_sha256,
            "frozen_g1_registry_authority_sha256",
        )
        _sha256(
            self.frozen_global_ledger_snapshot_sha256,
            "frozen_global_ledger_snapshot_sha256",
        )
        _sha256(
            self.control_anchor_receipt_sha256,
            "control_anchor_receipt_sha256",
        )
        _require(
            self.committed_control_state_sha256 == self.parent_state_sha256
            and self.committed_control_event_sha256 == self.parent_event_sha256,
            "stage parent is not the exact committed control state/event",
        )

    def binding(self) -> Mapping[str, Any]:
        base = {
            "goal_id": self.goal_id,
            "attempt_id": self.attempt_id,
            "transition_seq": self.transition_seq,
            "current_action": self.current_action,
            "current_status": self.current_status,
            "parent_state_sha256": self.parent_state_sha256,
            "parent_event_sha256": self.parent_event_sha256,
            "business_preimage_sha256": self.business_preimage_sha256,
            "static_review_receipt_sha256": self.static_review_receipt_sha256,
            "isolated_output_root_id": self.isolated_output_root_id,
            "isolated_output_root_path": self.isolated_output_root_path,
            "expected_negative_suite_sha256": self.expected_negative_suite_sha256,
            "control_transaction_id": self.control_transaction_id,
            "committed_control_state_sha256": self.committed_control_state_sha256,
            "committed_control_event_sha256": self.committed_control_event_sha256,
            "frozen_g1_registry_authority_sha256": self.frozen_g1_registry_authority_sha256,
            "frozen_global_ledger_snapshot_sha256": self.frozen_global_ledger_snapshot_sha256,
            "control_anchor_receipt_sha256": self.control_anchor_receipt_sha256,
            "train_jsonl_index_state": self.train_jsonl_index_state,
            "train_jsonl_index_binding_sha256": (
                self.train_jsonl_index_binding_sha256
            ),
            "train_jsonl_index_receipt_sha256": (
                self.train_jsonl_index_receipt_sha256
            ),
            "train_jsonl_index_transaction_id": (
                self.train_jsonl_index_transaction_id
            ),
        }
        return {**base, "state_binding_sha256": semantic_sha256(base)}


def _require_ready_train_jsonl_index(
    context: StageContext,
) -> Mapping[str, str]:
    _require(
        context.train_jsonl_index_state == TRAIN_JSONL_INDEX_READY
        and isinstance(context.train_jsonl_index_transaction_id, str)
        and TRAIN_JSONL_INDEX_TRANSACTION_RE.fullmatch(
            context.train_jsonl_index_transaction_id
        )
        is not None,
        "stage requires committed READY TRAIN-JSONL index authority",
    )
    binding_sha = _sha256(
        context.train_jsonl_index_binding_sha256,
        "train_jsonl_index_binding_sha256",
    )
    receipt_sha = _sha256(
        context.train_jsonl_index_receipt_sha256,
        "train_jsonl_index_receipt_sha256",
    )
    return {
        "train_jsonl_index_binding_sha256": binding_sha,
        "train_jsonl_index_receipt_sha256": receipt_sha,
        "train_jsonl_index_transaction_id": (
            context.train_jsonl_index_transaction_id
        ),
    }


_STAGE_BUNDLE_ISSUER = object()


@dataclass(frozen=True)
class StageBundle:
    plan: Mapping[str, Any]
    output_bytes: Mapping[str, bytes]
    commit_eligible: bool
    bundle_sha256: str
    _issuer: object

    def __post_init__(self) -> None:
        _require(
            self._issuer is _STAGE_BUNDLE_ISSUER,
            "StageBundle is constructible only by the reviewed stage builder",
        )
        _require(type(self.commit_eligible) is bool, "StageBundle eligibility must be exact bool")
        _require(
            isinstance(self.plan, Mapping) and isinstance(self.output_bytes, Mapping),
            "StageBundle plan/output mappings required",
        )
        detached_plan = json.loads(
            canonical_json_bytes(dict(self.plan)).decode("utf-8", "strict")
        )
        detached_outputs = dict(self.output_bytes)
        _require(
            all(isinstance(path, str) and isinstance(data, bytes) for path, data in detached_outputs.items()),
            "StageBundle output postimages must be bytes",
        )
        object.__setattr__(self, "plan", detached_plan)
        object.__setattr__(self, "output_bytes", detached_outputs)


def validate_stage_bundle(bundle: Any) -> StageBundle:
    """Validate the exact private issuer and all mutable nested postimages."""

    _require(type(bundle) is StageBundle, "exact StageBundle required")
    _require(
        bundle._issuer is _STAGE_BUNDLE_ISSUER,
        "StageBundle private issuer seal mismatch",
    )
    _sha256(bundle.bundle_sha256, "bundle_sha256")
    output_index = {
        path: {"sha256": bytes_sha256(data), "size_bytes": len(data)}
        for path, data in sorted(bundle.output_bytes.items())
    }
    binding = {
        "plan": dict(bundle.plan),
        "output_index": output_index,
        "commit_eligible": bundle.commit_eligible,
    }
    _require(
        bundle.bundle_sha256 == semantic_sha256(binding),
        "StageBundle content/seal hash mismatch",
    )
    _require(
        bundle.plan.get("output_index") == output_index
        and bundle.plan.get("commit_eligible") is bundle.commit_eligible,
        "StageBundle plan/postimage closure mismatch",
    )
    return bundle


def _exact_mapping(
    value: Any,
    fields: Sequence[str],
    message: str,
) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), message)
    _require(set(value) == set(fields), message)
    return value


def _canonical_power_video_id(value: Any, field: str) -> str:
    _require(
        isinstance(value, str)
        and value.isascii()
        and 1 <= len(value) <= 128
        and value[0].isalnum()
        and all(character.isalnum() or character in "_.:-" for character in value),
        "%s is not a canonical video ID" % field,
    )
    return value


def _derive_g2_power_outcome_row(
    *,
    packet: Mapping[str, Any],
    ground_truth: Mapping[str, Any],
    replay_packet_sha256: str,
    query_manifest_sha256: str,
) -> Mapping[str, Any]:
    """Compress one sealed replay packet into six exact binary outcomes.

    The complete proposal list and GT span are consumed only inside G2.  They
    are deliberately absent from the returned sufficient-statistics row.
    Input order is the frozen post-NMS rank order; this function never sorts or
    reapplies NMS.
    """

    _sha256(replay_packet_sha256, "replay_packet_sha256")
    _sha256(query_manifest_sha256, "query_manifest_sha256")
    current_packet = dict(
        _exact_mapping(
            packet,
            ("query_id", "joint_proposals"),
            "G2 sealed power packet fields mismatch",
        )
    )
    query_id = current_packet["query_id"]
    _require(
        type(query_id) is int and query_id >= 0,
        "G2 power desc_id must be a non-negative exact integer",
    )
    current_ground_truth = dict(
        _exact_mapping(
            ground_truth,
            ("gt_video_id", "gt_span_sec", "query_manifest_sha256"),
            "G2 sealed power ground-truth fields mismatch",
        )
    )
    _require(
        current_ground_truth["query_manifest_sha256"] == query_manifest_sha256,
        "G2 power ground truth/query-manifest mismatch",
    )
    gt_video_id = _canonical_power_video_id(
        current_ground_truth["gt_video_id"],
        "G2 power GT video ID",
    )
    gt_span = current_ground_truth["gt_span_sec"]
    _require(
        isinstance(gt_span, (list, tuple))
        and len(gt_span) == 2
        and all(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
            for value in gt_span
        )
        and float(gt_span[0]) >= 0.0
        and float(gt_span[1]) > float(gt_span[0]),
        "G2 power GT span is invalid",
    )
    proposals = current_packet["joint_proposals"]
    _require(
        isinstance(proposals, list) and 1 <= len(proposals) <= 100,
        "G2 power proposals must be a non-empty frozen list of at most 100 rows",
    )
    normalized: list[tuple[str, float, float, float]] = []
    for index, proposal in enumerate(proposals):
        _require(
            isinstance(proposal, list) and len(proposal) == 4,
            "G2 power proposal row shape mismatch at %d" % index,
        )
        video_id = _canonical_power_video_id(
            proposal[0],
            "G2 power proposal video ID at %d" % index,
        )
        start, end, score = proposal[1], proposal[2], proposal[3]
        _require(
            all(
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(float(value))
                for value in (start, end, score)
            )
            and float(start) >= 0.0
            and float(end) > float(start),
            "G2 power proposal numeric fields invalid at %d" % index,
        )
        normalized.append(
            (video_id, float(start), float(end), float(score))
        )

    gt_span_pair = (float(gt_span[0]), float(gt_span[1]))

    def recall_hit(rank: int) -> int:
        return int(
            any(
                video_id == gt_video_id
                and temporal_iou((start, end), gt_span_pair) >= 0.7
                for video_id, start, end, _score in normalized[:rank]
            )
        )

    unique_video_ids: list[str] = []
    seen_video_ids: set[str] = set()
    for video_id, _start, _end, _score in normalized:
        if video_id not in seen_video_ids:
            seen_video_ids.add(video_id)
            unique_video_ids.append(video_id)
    top_video_id, top_start, top_end, _top_score = normalized[0]
    top_iou = (
        temporal_iou((top_start, top_end), gt_span_pair)
        if top_video_id == gt_video_id
        else 0.0
    )
    outcomes = {
        "CORRECT_VIDEO_WRONG_SPAN_TOP1": int(
            top_video_id == gt_video_id and top_iou < 0.5
        ),
        "LEGACY_UNIQUE_VIDEO_TRUE_VR_R100": int(
            gt_video_id in unique_video_ids[:100]
        ),
        "VCMR_R10_IOU_0_7": recall_hit(10),
        "VCMR_R1_IOU_0_7": recall_hit(1),
        "VCMR_R5_IOU_0_7": recall_hit(5),
        "WRONG_VIDEO_TOP1": int(top_video_id != gt_video_id),
    }
    _require(
        tuple(outcomes) == POWER_OUTCOME_KEYS
        and all(type(value) is int and value in {0, 1} for value in outcomes.values()),
        "G2 six-bit power outcome encoding drift",
    )
    return {
        "schema_version": POWER_REPLAY_ROW_SCHEMA,
        "desc_id": query_id,
        "gt_video_id": gt_video_id,
        "outcomes": outcomes,
        "replay_packet_sha256": replay_packet_sha256,
        "query_manifest_sha256": query_manifest_sha256,
    }


def _copy_eval_request(
    request: EvaluationRequest,
    **changes: Any,
) -> EvaluationRequest:
    fields = {
        "request_id": request.request_id,
        "capability_id": request.capability_id,
        "split_id": request.split_id,
        "split_manifest_sha256": request.split_manifest_sha256,
        "purpose": request.purpose,
        "goal_id": request.goal_id,
        "authority_id": request.authority_id,
        "budget_token_id": request.budget_token_id,
        "candidate_corpus_manifest_sha256": request.candidate_corpus_manifest_sha256,
        "mode": request.mode,
        "eval_id": request.eval_id,
    }
    fields.update(changes)
    return EvaluationRequest(**fields)


def _registry_from_g0_authority(
    authority: Mapping[str, Any],
    context: StageContext,
) -> tuple[
    Tuple[FrozenCanonicalRoot, ...],
    Tuple[FrozenPathRule, ...],
    SplitAuthorizationRegistry,
    PathIntent,
    EvaluationRequest,
    Mapping[str, Any],
    Sequence[Mapping[str, Any]],
]:
    authority_fields = (
        "schema_version",
        "goal_id",
        "authority_id",
        "canonical_path_map",
        "protected_id_mapping_manifest_contract",
        "split_registry",
        "write_root_roles",
        "expected_mount_ids",
        "metadata_path_probe",
        "metadata_eval_request",
        "eval_budget_sha256",
        "negative_suite_sha256",
        "authority_sha256",
    )
    current = dict(_exact_mapping(authority, authority_fields, "G0 G1 registry authority fields mismatch"))
    _require(current["schema_version"] == G1_REGISTRY_AUTHORITY_SCHEMA, "G0 G1 registry authority schema")
    _require(current["goal_id"] == context.goal_id, "G0 registry Goal mismatch")
    _require(
        current["negative_suite_sha256"]
        == context.expected_negative_suite_sha256
        == G1_NEGATIVE_SUITE_SHA256,
        "externally locked negative suite mismatch",
    )
    authority_sha = _sha256(current["authority_sha256"], "authority_sha256")
    _require(
        authority_sha
        == semantic_sha256(current, excluded_fields=("authority_sha256",)),
        "G0 G1 registry authority self-hash mismatch",
    )
    _sha256(current["eval_budget_sha256"], "eval_budget_sha256")

    protected_contract = dict(
        _exact_mapping(
            current["protected_id_mapping_manifest_contract"],
            (
                "schema_version",
                "manifests",
                "union_count",
                "union_sha256",
                "selection_rule",
            ),
            "protected ID mapping manifest contract fields mismatch",
        )
    )
    protected_manifests = protected_contract["manifests"]
    _require(
        protected_contract["schema_version"]
        == "c28f_a4_protected_id_mapping_manifest_contract_v1"
        and isinstance(protected_manifests, list)
        and len(protected_manifests) == 2
        and protected_contract["union_count"] == 8_679
        and protected_contract["union_sha256"]
        == "39a758494985a4db0963d5bcf0da92738b57f04b913a147b328532edceb2db31"
        and protected_contract["selection_rule"]
        == (
            "CANONICAL_SORTED_UNIQUE_UNION_CALIB_HOLDOUT_AND_"
            "PSEUDO_OFFICIAL_HOLDOUT"
        ),
        "protected ID mapping manifest contract binding",
    )
    _sha256(protected_contract["union_sha256"], "protected ID union SHA")
    protected_by_rule: dict[str, Mapping[str, Any]] = {}
    for manifest in protected_manifests:
        record = dict(
            _exact_mapping(
                manifest,
                (
                    "manifest_id",
                    "rule_id",
                    "absolute_path",
                    "content_sha256",
                    "line_count",
                    "ordered_ids_sha256",
                ),
                "protected ID manifest fields mismatch",
            )
        )
        rule_id = str(record["rule_id"])
        _require(
            rule_id not in protected_by_rule
            and Path(str(record["absolute_path"])).is_absolute()
            and type(record["line_count"]) is int
            and record["line_count"] > 0,
            "protected ID manifest identity",
        )
        _sha256(record["content_sha256"], "protected ID manifest content SHA")
        _sha256(
            record["ordered_ids_sha256"],
            "protected ID manifest ordered IDs SHA",
        )
        protected_by_rule[rule_id] = record
    _require(
        set(protected_by_rule)
        == {
            "g1-protected-calib-holdout",
            "g1-protected-pseudo-official",
        }
        and {str(record["manifest_id"]) for record in protected_by_rule.values()}
        == {"calib_holdout", "pseudo_official_holdout"},
        "protected ID manifest set",
    )

    expected_path_map = dict(
        _exact_mapping(
            current["canonical_path_map"],
            (
                "schema_version",
                "canonical_roots",
                "forbidden_registry",
                "path_policy",
                "canonical_path_map_sha256",
            ),
            "expected canonical path map fields mismatch",
        )
    )
    expected_path_sha = _sha256(
        expected_path_map["canonical_path_map_sha256"],
        "canonical_path_map_sha256",
    )
    _require(
        expected_path_sha
        == semantic_sha256(
            expected_path_map,
            excluded_fields=("canonical_path_map_sha256",),
        ),
        "expected canonical path map self-hash mismatch",
    )
    expected_root_records = expected_path_map["canonical_roots"]
    expected_rule_records = expected_path_map["forbidden_registry"]
    _require(isinstance(expected_root_records, list) and expected_root_records, "expected roots empty")
    _require(isinstance(expected_rule_records, list) and expected_rule_records, "expected rules empty")
    roots = []
    for record in expected_root_records:
        root_record = dict(
            _exact_mapping(
                record,
                (
                    "root_id",
                    "canonical_path",
                    "device",
                    "inode",
                    "mount_id",
                    "object_kind",
                    "operations",
                    "nlink",
                ),
                "expected root record fields mismatch",
            )
        )
        captured = FrozenCanonicalRoot.capture(
            str(root_record["root_id"]),
            str(root_record["canonical_path"]),
            root_record["operations"],
        )
        _require(captured.as_dict() == root_record, "live root recapture differs from G0 authority")
        roots.append(captured)
    rules = []
    for record in expected_rule_records:
        rule_record = dict(
            _exact_mapping(
                record,
                (
                    "rule_id",
                    "canonical_path",
                    "device",
                    "inode",
                    "mount_id",
                    "size_bytes",
                    "mtime_ns",
                    "object_kind",
                    "scope",
                    "classification",
                    "content_sha256",
                    "nlink",
                ),
                "expected forbidden rule fields mismatch",
            )
        )
        manifest = protected_by_rule.get(str(rule_record["rule_id"]))
        approved_content_sha256 = (
            None if manifest is None else manifest["content_sha256"]
        )
        _require(
            rule_record["content_sha256"] == approved_content_sha256
            and (
                manifest is None
                or manifest["absolute_path"] == rule_record["canonical_path"]
            ),
            "protected/forbidden content hash authority mismatch",
        )
        captured = FrozenPathRule.capture_metadata_only(
            str(rule_record["rule_id"]),
            str(rule_record["canonical_path"]),
            scope=str(rule_record["scope"]),
            classification=str(rule_record["classification"]),
            approved_content_sha256=approved_content_sha256,
        )
        _require(captured.as_dict() == rule_record, "live rule recapture differs from G0 authority")
        rules.append(captured)
    _require(
        set(protected_by_rule) <= {rule.rule_id for rule in rules},
        "protected ID manifest rule is absent",
    )
    roots_tuple = tuple(roots)
    rules_tuple = tuple(rules)
    actual_path_map = canonical_path_map(roots_tuple, rules_tuple)
    _require(actual_path_map == expected_path_map, "live path map differs from G0 expected SHA")

    roles = dict(
        _exact_mapping(
            current["write_root_roles"],
            ("output", "cache", "tmp"),
            "write root roles must be exact",
        )
    )
    _require(len(set(roles.values())) == 3, "write root role IDs must be unique")
    write_roots = [root for root in roots_tuple if root.operations == ("WRITE",)]
    _require({root.root_id for root in write_roots} == set(roles.values()), "write root set differs from G0 roles")
    _require(roles["output"] == context.isolated_output_root_id, "context output root ID mismatch")
    output_root = next(root for root in write_roots if root.root_id == roles["output"])
    _require(output_root.canonical_path == context.isolated_output_root_path, "context output root path mismatch")
    pristine_receipts = [verify_pristine_write_root(root) for root in write_roots]

    expected_mount_ids_raw = current["expected_mount_ids"]
    _require(isinstance(expected_mount_ids_raw, list), "expected mount IDs must be a list")
    expected_mount_ids = tuple(expected_mount_ids_raw)
    _require(
        tuple(sorted(set(expected_mount_ids))) == expected_mount_ids
        and set(expected_mount_ids) == {root.mount_id for root in roots_tuple} | {rule.mount_id for rule in rules_tuple},
        "expected mount domain differs from exact registry",
    )
    mount_snapshot = capture_mount_domain(expected_mount_ids)

    expected_split = dict(
        _exact_mapping(
            current["split_registry"],
            (
                "schema_version",
                "goal_id",
                "authority_id",
                "capabilities",
                "forbidden_split_aliases",
                "split_name_alone_authorizes",
                "protected_default_exists",
                "registry_sha256",
            ),
            "expected split registry fields mismatch",
        )
    )
    split_sha = _sha256(expected_split["registry_sha256"], "split registry SHA")
    _require(
        split_sha
        == semantic_sha256(expected_split, excluded_fields=("registry_sha256",)),
        "expected split registry self-hash mismatch",
    )
    capabilities = []
    capability_fields = (
        "capability_id",
        "split_id",
        "split_manifest_sha256",
        "purpose",
        "goal_id",
        "authority_id",
        "budget_token_id",
        "protected",
        "query_entity_namespace",
        "candidate_corpus_manifest_sha256",
        "allow_model_forward",
    )
    for record in expected_split["capabilities"]:
        capabilities.append(
            SplitCapability(
                **dict(_exact_mapping(record, capability_fields, "split capability fields mismatch"))
            )
        )
    split_registry = SplitAuthorizationRegistry(
        goal_id=str(expected_split["goal_id"]),
        authority_id=str(expected_split["authority_id"]),
        capabilities=tuple(capabilities),
        forbidden_split_aliases=expected_split["forbidden_split_aliases"],
    )
    _require(split_registry.registry() == expected_split, "live split registry differs from G0 expected SHA")

    path_probe = dict(
        _exact_mapping(
            current["metadata_path_probe"],
            (
                "request_id",
                "root_id",
                "path",
                "expected_kind",
                "purpose",
                "expected_descriptor_identity",
                "expected_descriptor_identity_sha256",
            ),
            "metadata path probe fields mismatch",
        )
    )
    probe_root = next((root for root in roots_tuple if root.root_id == path_probe["root_id"]), None)
    _require(probe_root is not None and "READ_METADATA" in probe_root.operations, "metadata probe root unauthorized")
    _require(
        str(path_probe["path"]).startswith(probe_root.canonical_path + "/"),
        "metadata probe path outside exact root",
    )
    expected_probe_identity = dict(
        _exact_mapping(
            path_probe["expected_descriptor_identity"],
            (
                "device",
                "inode",
                "mount_id",
                "object_kind",
                "size_bytes",
                "mtime_ns",
                "nlink",
            ),
            "metadata probe descriptor identity fields mismatch",
        )
    )
    expected_probe_identity_sha = _sha256(
        path_probe["expected_descriptor_identity_sha256"],
        "metadata probe descriptor identity SHA",
    )
    _require(
        expected_probe_identity_sha == semantic_sha256(expected_probe_identity),
        "metadata probe descriptor identity self-hash mismatch",
    )
    live_probe = FrozenPathRule.capture_metadata_only(
        "g1-metadata-probe-recapture",
        str(path_probe["path"]),
        scope="EXACT",
        classification="PROTECTED_DATA",
    )
    live_probe_identity = {
        "device": live_probe.device,
        "inode": live_probe.inode,
        "mount_id": live_probe.mount_id,
        "object_kind": live_probe.object_kind,
        "size_bytes": live_probe.size_bytes,
        "mtime_ns": live_probe.mtime_ns,
        "nlink": live_probe.nlink,
    }
    _require(
        live_probe_identity == expected_probe_identity,
        "metadata probe descriptor identity recapture drift",
    )
    path_intent = PathIntent(
        request_id=str(path_probe["request_id"]),
        operation="READ_METADATA",
        path=str(path_probe["path"]),
        expected_kind=str(path_probe["expected_kind"]),
        purpose=str(path_probe["purpose"]),
        goal_id=context.goal_id,
        authority_id=str(current["authority_id"]),
    )
    eval_fields = (
        "request_id",
        "capability_id",
        "split_id",
        "split_manifest_sha256",
        "purpose",
        "goal_id",
        "authority_id",
        "budget_token_id",
        "candidate_corpus_manifest_sha256",
        "mode",
        "eval_id",
    )
    eval_request = EvaluationRequest(
        **dict(
            _exact_mapping(
                current["metadata_eval_request"],
                eval_fields,
                "metadata eval request fields mismatch",
            )
        )
    )
    return (
        roots_tuple,
        rules_tuple,
        split_registry,
        path_intent,
        eval_request,
        mount_snapshot.as_dict(),
        pristine_receipts,
    )


def _validate_persisted_global_snapshot(
    snapshot: Mapping[str, Any],
    context: StageContext,
    authority_id: str,
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    fields = (
        "schema_version",
        "goal_id",
        "attempt_id",
        "authority_id",
        "committed_control_transaction_id",
        "ledger_scope_id",
        "ledger_rows",
        "incidents",
        "real_data_optimizer_updates",
        "persistence_status",
        "persistence_transaction_id",
        "snapshot_sha256",
    )
    current = dict(_exact_mapping(snapshot, fields, "persisted global snapshot fields mismatch"))
    _require(current["schema_version"] == G1_PERSISTED_SNAPSHOT_SCHEMA, "persisted global snapshot schema")
    _require(
        current["goal_id"] == context.goal_id
        and current["attempt_id"] == context.attempt_id
        and current["authority_id"] == authority_id
        and isinstance(current["committed_control_transaction_id"], str)
        and re.fullmatch(
            r"G0-A4-TXN-[0-9a-f]{32}",
            current["committed_control_transaction_id"],
        )
        is not None,
        "persisted global snapshot lineage mismatch",
    )
    _require(
        current["persistence_status"]
        == "PRECOMMITTED_IN_G0_A4_GENESIS_REGISTRY",
        "global snapshot is not a precommitted Genesis registry",
    )
    _require(
        type(current["real_data_optimizer_updates"]) is int
        and current["real_data_optimizer_updates"] == 0,
        "persisted global optimizer update count is nonzero or invalid",
    )
    _require(isinstance(current["persistence_transaction_id"], str) and current["persistence_transaction_id"], "global snapshot transaction missing")
    _require(
        current["persistence_transaction_id"]
        == current["committed_control_transaction_id"],
        "global snapshot persistence transaction differs from its Genesis anchor",
    )
    snapshot_sha = _sha256(current["snapshot_sha256"], "snapshot_sha256")
    _require(
        snapshot_sha == semantic_sha256(current, excluded_fields=("snapshot_sha256",)),
        "global snapshot self-hash mismatch",
    )
    rows = current["ledger_rows"]
    _require(isinstance(rows, list) and rows, "persisted global ledger rows missing")
    prior_manifest = derive_safety_manifest(
        rows,
        expected_scope_id=str(current["ledger_scope_id"]),
        expected_goal_id=context.goal_id,
        expected_authority_id=authority_id,
        require_closed=True,
    )
    incidents = current["incidents"]
    _require(isinstance(incidents, list), "incident registry must be a list")
    for incident in incidents:
        record = dict(
            _exact_mapping(
                incident,
                (
                    "incident_id",
                    "classification",
                    "exposure",
                    "scientific_impact",
                    "disposition",
                    "user_ack_sha256",
                    "incident_sha256",
                ),
                "incident fields mismatch",
            )
        )
        _require(
            record["scientific_impact"] == "NONE"
            and record["disposition"] == "MATERIAL_RECORDED_EXCEPTION",
            "unresolved scientific incident",
        )
        _sha256(record["user_ack_sha256"], "incident user ack")
        _require(
            record["incident_sha256"]
            == semantic_sha256(record, excluded_fields=("incident_sha256",)),
            "incident self-hash mismatch",
        )
    return current, prior_manifest


def _run_builtin_negative_suite(
    *,
    control_handle: Any,
    context: StageContext,
    authority_id: str,
    roots: Sequence[FrozenCanonicalRoot],
    forbidden_rules: Sequence[FrozenPathRule],
    guard: DescriptorPathGuard,
    split_registry: SplitAuthorizationRegistry,
    budget: EvalBudgetBook,
    request: EvaluationRequest,
    metadata_path_intent: PathIntent,
) -> Mapping[str, Any]:
    _require(
        context.expected_negative_suite_sha256 == G1_NEGATIVE_SUITE_SHA256,
        "builtin negative suite is not externally prelocked",
    )
    token = budget.get(str(request.budget_token_id))
    replay_budget = EvalBudgetBook(
        tokens=tuple(
            EvalBudgetToken(
                token_id=item.token_id,
                purpose=item.purpose,
                state=("RESERVED" if item.token_id == token.token_id else item.state),
                max_uses=item.max_uses,
                use_count=(1 if item.token_id == token.token_id else item.use_count),
                reserved_eval_id=("negative-replay" if item.token_id == token.token_id else item.reserved_eval_id),
            )
            for item in budget.tokens
        )
    )
    exhausted_budget = EvalBudgetBook(
        tokens=tuple(
            EvalBudgetToken(
                token_id=item.token_id,
                purpose=item.purpose,
                state=("EXHAUSTED" if item.token_id == token.token_id else item.state),
                max_uses=item.max_uses,
                use_count=(1 if item.token_id == token.token_id else item.use_count),
                reserved_eval_id=("negative-exhausted" if item.token_id == token.token_id else item.reserved_eval_id),
            )
            for item in budget.tokens
        )
    )
    write_roots = [root for root in roots if root.operations == ("WRITE",)]
    first_write = write_roots[0]
    second_write = write_roots[1]
    overlap_root = FrozenCanonicalRoot(
        root_id=second_write.root_id,
        canonical_path=first_write.canonical_path + "/nested",
        device=second_write.device,
        inode=second_write.inode,
        mount_id=second_write.mount_id,
        object_kind="DIRECTORY",
        operations=("WRITE",),
        nlink=2,
    )
    overlap_roots = tuple(
        overlap_root if root.root_id == second_write.root_id else root for root in roots
    )
    forbidden_overlap = FrozenPathRule(
        rule_id="negative-output-overlap",
        canonical_path=first_write.canonical_path,
        device=first_write.device,
        inode=first_write.inode,
        mount_id=first_write.mount_id,
        size_bytes=0,
        mtime_ns=0,
        object_kind="DIRECTORY",
        scope="TREE",
        classification="HISTORICAL_ROOT",
        content_sha256=None,
        nlink=first_write.nlink,
    )
    metadata_identity = FrozenPathRule.capture_metadata_only(
        "negative-metadata-identity-source",
        metadata_path_intent.path,
        scope="EXACT",
        classification="PROTECTED_DATA",
    )
    inode_alias = FrozenPathRule(
        rule_id="negative-inode-alias",
        canonical_path=metadata_path_intent.path + ".unreachable-alias",
        device=metadata_identity.device,
        inode=metadata_identity.inode,
        mount_id=metadata_identity.mount_id,
        size_bytes=metadata_identity.size_bytes,
        mtime_ns=metadata_identity.mtime_ns,
        object_kind=metadata_identity.object_kind,
        scope="EXACT",
        classification="PROTECTED_DATA",
        content_sha256=None,
        nlink=metadata_identity.nlink,
    )
    alias_guard = DescriptorPathGuard(
        goal_id=context.goal_id,
        authority_id=authority_id,
        roots=roots,
        forbidden_rules=tuple(forbidden_rules) + (inode_alias,),
    )

    def eval_probe(changes: Mapping[str, Any], selected_budget: EvalBudgetBook, ledger: EvidenceLedger) -> None:
        split_registry.authorize(
            _copy_eval_request(request, **dict(changes)),
            selected_budget,
            ledger,
            dry_run=True,
        )

    actions = {
        "budget_over_limit": lambda ledger: eval_probe({}, exhausted_budget, ledger),
        "budget_token_replay": lambda ledger: eval_probe({}, replay_budget, ledger),
        "launcher_c28e_stage_all": lambda ledger: validate_single_action_argv(
            ["--stage", "all"], expected_action=G1_ACTION, ledger=ledger
        ),
        "launcher_c28e_stage_holdout": lambda ledger: validate_single_action_argv(
            ["--stage", "holdout"], expected_action=G1_ACTION, ledger=ledger
        ),
        "legacy_authorization_marker": lambda ledger: reject_legacy_authorization_markers(
            {"allow_holdout_final": True}, ledger
        ),
        "missing_capability": lambda ledger: eval_probe({"capability_id": None}, budget, ledger),
        "output_forbidden_overlap": lambda ledger: validate_root_registry_preopen(
            roots=roots,
            forbidden_rules=tuple(forbidden_rules) + (forbidden_overlap,),
            ledger=ledger,
            detector_id="output_forbidden_overlap",
        ),
        "output_roots_overlap": lambda ledger: validate_root_registry_preopen(
            roots=overlap_roots,
            forbidden_rules=forbidden_rules,
            ledger=ledger,
            detector_id="output_roots_overlap",
        ),
        "path_device_inode_alias": lambda ledger: alias_guard.authorize(
            metadata_path_intent, ledger
        ),
        "path_dotdot": lambda ledger: guard.authorize(
            PathIntent(
                request_id="negative-dotdot",
                operation="READ_METADATA",
                path=metadata_path_intent.path + "/../escape",
                expected_kind="FILE",
                purpose="G1_NEGATIVE_PATH_TRAVERSAL",
                goal_id=context.goal_id,
                authority_id=authority_id,
            ),
            ledger,
        ),
        "path_symlink": lambda ledger: validate_descriptor_chain_preopen(
            records=(
                DescriptorRecord(
                    canonical_prefix="/synthetic/symlink",
                    device=1,
                    inode=1,
                    mount_id=1,
                    object_kind="SYMLINK",
                    size_bytes=0,
                    mtime_ns=0,
                    nlink=1,
                ),
            ),
            ledger=ledger,
            detector_id="path_symlink",
        ),
        "protected_alias_calib_holdout": lambda ledger: eval_probe(
            {"split_id": "calib_holdout"}, budget, ledger
        ),
        "protected_alias_official": lambda ledger: eval_probe(
            {"split_id": "official"}, budget, ledger
        ),
        "protected_alias_pseudo_official_holdout": lambda ledger: eval_probe(
            {"split_id": "pseudo_official_holdout"}, budget, ledger
        ),
        "split_missing": lambda ledger: eval_probe({"split_id": None}, budget, ledger),
        "split_null": lambda ledger: eval_probe({"split_id": None}, budget, ledger),
        "split_random": lambda ledger: eval_probe({"split_id": "random_split"}, budget, ledger),
        "wrong_authority": lambda ledger: eval_probe({"authority_id": "wrong-authority"}, budget, ledger),
        "wrong_budget_token": lambda ledger: eval_probe({"budget_token_id": "wrong-token"}, budget, ledger),
        "wrong_candidate_manifest": lambda ledger: eval_probe(
            {"candidate_corpus_manifest_sha256": "0" * 64}, budget, ledger
        ),
        "wrong_goal": lambda ledger: eval_probe({"goal_id": "wrong-goal"}, budget, ledger),
        "wrong_purpose": lambda ledger: eval_probe({"purpose": "wrong-purpose"}, budget, ledger),
        "wrong_split_manifest": lambda ledger: eval_probe(
            {"split_manifest_sha256": "0" * 64}, budget, ledger
        ),
    }
    _require(
        set(actions) == set(REQUIRED_G1_NEGATIVE_PROBES) - {"safety_manifest_ledger_mismatch"},
        "builtin negative action set mismatch",
    )
    results = []
    controller_tripwire: Optional[Mapping[str, Any]] = None
    for probe_id in sorted(REQUIRED_G1_NEGATIVE_PROBES):
        ledger = control_handle.open_mediated_ledger(
            scope_id="G1-NEGATIVE-%s" % probe_id,
            authority_id=authority_id,
        )
        try:
            if probe_id == "safety_manifest_ledger_mismatch":
                source = control_handle.open_mediated_ledger(
                    scope_id="G1-NEGATIVE-MANIFEST-SOURCE",
                    authority_id=authority_id,
                )
                source_closure = control_handle.close_mediated_ledger(source)
                source_rows = source_closure["rows"]
                claimed = dict(
                    derive_safety_manifest(
                        source_rows,
                        expected_scope_id="G1-NEGATIVE-MANIFEST-SOURCE",
                        expected_goal_id=context.goal_id,
                        expected_authority_id=authority_id,
                    )
                )
                claimed["manifest_sha256"] = "0" * 64
                verify_claimed_safety_manifest_preopen(
                    claimed,
                    source_rows,
                    expected_scope_id="G1-NEGATIVE-MANIFEST-SOURCE",
                    expected_goal_id=context.goal_id,
                    expected_authority_id=authority_id,
                    ledger=ledger,
                )
            else:
                actions[probe_id](ledger)
        except SecurityViolation as exc:
            expected_reason = REQUIRED_G1_NEGATIVE_PROBES[probe_id]
            _require(exc.code == expected_reason, "builtin negative reason drift: %s" % probe_id)
            matching = [
                row
                for row in ledger.rows
                if row["decision"] == "DENY" and row["reason_code"] == expected_reason
            ]
            _require(matching, "guard/detector did not emit its own DENY: %s" % probe_id)
        else:
            raise StageContractError("builtin negative probe unexpectedly allowed: %s" % probe_id)
        closure = control_handle.close_mediated_ledger(ledger)
        _require(
            closure.get("instrumentation_complete") is True,
            "controller did not close negative instrumentation",
        )
        rows = closure.get("rows")
        _require(isinstance(rows, list) and rows, "controller negative ledger rows missing")
        controller_tripwire = dict(closure.get("io_tripwire", {}))
        _require(
            set(controller_tripwire)
            == {
                "data_loader_calls",
                "content_open_calls",
                "model_forward_calls",
                "output_write_calls",
            },
            "controller I/O tripwire fields mismatch",
        )
        _require(
            all(type(value) is int and value >= 0 for value in controller_tripwire.values()),
            "controller I/O tripwire counter invalid",
        )
        manifest = derive_safety_manifest(
            rows,
            expected_scope_id="G1-NEGATIVE-%s" % probe_id,
            expected_goal_id=context.goal_id,
            expected_authority_id=authority_id,
        )
        _require(manifest["totals"]["denied_attempt_count"] >= 1, "negative DENY absent")
        for field in (
            "content_open_count",
            "content_bytes_read",
            "content_bytes_written",
            "lmdb_transaction_open_count",
            "model_forward_count",
        ):
            _require(manifest["totals"][field] == 0, "negative I/O tripwire fired")
        results.append(
            {
                "probe_id": probe_id,
                "expected_reason_code": REQUIRED_G1_NEGATIVE_PROBES[probe_id],
                "detector_owned_denial": True,
                "result": "DENIED_BEFORE_IO",
                "ledger_rows": list(rows),
                "ledger_manifest_sha256": manifest["manifest_sha256"],
            }
        )
    _require(
        controller_tripwire is not None and not any(controller_tripwire.values()),
        "controller-observed negative suite I/O tripwire fired",
    )
    base = {
        "schema_version": "c28f_g1_negative_test_report_v2",
        "suite_manifest": G1_NEGATIVE_SUITE_MANIFEST,
        "suite_sha256": G1_NEGATIVE_SUITE_SHA256,
        "suite_source": "BUILTIN_REVIEWED_NONINJECTABLE",
        "required_probe_count": len(REQUIRED_G1_NEGATIVE_PROBES),
        "passed_probe_count": len(results),
        "io_surface_exposed_to_probes": False,
        "io_tripwire": dict(controller_tripwire),
        "all_denied_before_io": True,
        "results": results,
    }
    return {**base, "report_sha256": semantic_sha256(base)}


def _cumulative_safety_manifest(
    *,
    current_manifest: Mapping[str, Any],
    prior_manifest: Mapping[str, Any],
    persisted_snapshot: Mapping[str, Any],
) -> Mapping[str, Any]:
    safety_fields = (
        "content_open_count",
        "content_bytes_read",
        "content_bytes_written",
        "lmdb_transaction_open_count",
        "model_forward_count",
        "protected_content_open_count",
        "protected_feature_open_count",
        "protected_prediction_open_count",
        "protected_metric_open_count",
        "protected_label_open_count",
        "protected_query_text_timestamp_open_count",
        "protected_id_mapping_content_open_count",
        "historical_write_bytes",
        "official_workflow_allowed_count",
        "unresolved_open_intent_count",
        "observed_unmediated_content_open_count",
    )
    prior_totals = prior_manifest["totals"]
    current_totals = current_manifest["totals"]
    cumulative = {
        field: int(prior_totals[field]) + int(current_totals[field])
        for field in safety_fields
    }
    safe = bool(
        prior_manifest["claims"]["safe_for_nonprotected_f0_metadata_dry_run"]
        and current_manifest["claims"]["safe_for_nonprotected_f0_metadata_dry_run"]
        and all(cumulative[field] == 0 for field in safety_fields)
    )
    base = {
        "schema_version": "c28f_g1_cumulative_ledger_safety_manifest_v1",
        "persisted_snapshot_sha256": persisted_snapshot["snapshot_sha256"],
        "persisted_ledger_manifest_sha256": prior_manifest["manifest_sha256"],
        "g1_delta_manifest_sha256": current_manifest["manifest_sha256"],
        "incident_count": len(persisted_snapshot["incidents"]),
        "real_data_optimizer_updates": persisted_snapshot[
            "real_data_optimizer_updates"
        ],
        "incidents": list(persisted_snapshot["incidents"]),
        "cumulative_totals": cumulative,
        "claims": {
            "derived_from_persisted_global_snapshot": True,
            "derived_from_incident_registry": True,
            "safe_for_nonprotected_f0_metadata_dry_run": safe,
        },
    }
    return {**base, "manifest_sha256": semantic_sha256(base)}


def _build_stage_bundle(
    *,
    _builder_issuer: object,
    context: StageContext,
    action: str,
    result_status: str,
    next_action: str,
    outputs: Mapping[str, bytes],
    input_hashes: Mapping[str, str],
    declared_safety_counts: Mapping[str, int],
    commit_eligible: bool,
) -> StageBundle:
    _require(
        _builder_issuer is _STAGE_BUNDLE_ISSUER,
        "reviewed stage builder issuer required",
    )
    _require(outputs, "stage output set cannot be empty")
    output_index = {}
    for path, data in sorted(outputs.items()):
        _relative_output_path(path)
        _require(
            isinstance(data, bytes)
            and data
            and len(data) <= STAGE_OUTPUT_FILE_MAX_BYTES,
            "stage output must be non-empty bytes within the fixed 64 MiB cap",
        )
        output_index[path] = {"sha256": bytes_sha256(data), "size_bytes": len(data)}
    for name, digest in input_hashes.items():
        _sha256(digest, "input_hashes.%s" % name)
    safety_fields = {
        "protected_content_opens",
        "protected_id_mapping_content_opens",
        "historical_write_bytes",
        "model_forward_evaluations",
        "real_data_optimizer_updates",
    }
    _require(set(declared_safety_counts) == safety_fields, "declared safety count fields mismatch")
    _require(
        all(type(value) is int and value >= 0 for value in declared_safety_counts.values()),
        "declared safety count invalid",
    )
    next_state_base = {
        "parent_state_sha256": context.parent_state_sha256,
        "transition_seq": context.transition_seq,
        "stage": action.split("_", 1)[0],
        "status": result_status,
        "next_action": next_action,
        "output_index_sha256": semantic_sha256(output_index),
        "static_review_receipt_sha256": context.static_review_receipt_sha256,
    }
    plan_base = {
        "schema_version": STAGE_PLAN_SCHEMA,
        "goal_id": context.goal_id,
        "attempt_id": context.attempt_id,
        "action": action,
        "action_is_unique": True,
        "state_binding": context.binding(),
        "static_review_receipt_sha256": context.static_review_receipt_sha256,
        "business_preimage_sha256": context.business_preimage_sha256,
        "isolated_output_root_id": context.isolated_output_root_id,
        "output_policy": {
            "controller_resolves_under_attempt_isolated_root": True,
            "historical_or_shared_write_allowed": False,
            "direct_stage_write_performed": False,
        },
        "input_hashes": dict(sorted(input_hashes.items())),
        "output_index": output_index,
        "next_state_contract": {
            **next_state_base,
            "next_state_contract_sha256": semantic_sha256(next_state_base),
        },
        "declared_safety_counts": dict(declared_safety_counts),
        "commit_eligible": commit_eligible,
    }
    plan = {**plan_base, "stage_plan_sha256": semantic_sha256(plan_base)}
    detached_outputs = dict(outputs)
    bundle_binding = {
        "plan": plan,
        "output_index": {
            path: {"sha256": bytes_sha256(data), "size_bytes": len(data)}
            for path, data in sorted(detached_outputs.items())
        },
        "commit_eligible": commit_eligible,
    }
    return StageBundle(
        plan=plan,
        output_bytes=detached_outputs,
        commit_eligible=commit_eligible,
        bundle_sha256=semantic_sha256(bundle_binding),
        _issuer=_STAGE_BUNDLE_ISSUER,
    )


def submit_stage_bundle(adapter: Any, bundle: StageBundle) -> Mapping[str, Any]:
    """Require an exact two-phase a4_control CAS receipt; no fake COMMITTED."""

    from .a4_control import A4ControlStageAdapter

    _require(
        type(adapter) is A4ControlStageAdapter,
        "concrete A4ControlStageAdapter required",
    )
    validate_stage_bundle(bundle)
    _require(bundle.commit_eligible and bundle.plan.get("commit_eligible") is True, "preview/synthetic stage bundle cannot be committed")
    expected = bundle.plan.get("output_index")
    _require(isinstance(expected, Mapping) and set(expected) == set(bundle.output_bytes), "stage output index mismatch")
    for path, data in bundle.output_bytes.items():
        record = expected[path]
        _require(record["sha256"] == bytes_sha256(data), "stage output hash drift: %s" % path)
        _require(record["size_bytes"] == len(data), "stage output size drift: %s" % path)
    plan_without_hash = dict(bundle.plan)
    observed_plan_sha = plan_without_hash.pop("stage_plan_sha256", None)
    _require(observed_plan_sha == semantic_sha256(plan_without_hash), "stage plan self-hash mismatch")
    output_hashes = {
        path: record["sha256"] for path, record in sorted(expected.items())
    }
    prepare = dict(
        adapter.prepare_stage_commit(
            bundle=bundle,
        )
    )
    prepare_fields = {
        "schema_version",
        "status",
        "goal_id",
        "attempt_id",
        "stage_action",
        "stage_commit_id",
        "parent_state_sha256",
        "parent_event_sha256",
        "business_preimage_sha256",
        "output_hashes",
        "static_review_receipt_sha256",
        "stage_plan_sha256",
        "transaction_id",
        "receipt_sha256",
    }
    _require(set(prepare) == prepare_fields, "stage prepare receipt fields mismatch")
    prepare_receipt_sha = _sha256(prepare["receipt_sha256"], "prepare receipt_sha256")
    _require(
        prepare_receipt_sha
        == semantic_sha256(prepare, excluded_fields=("receipt_sha256",)),
        "stage prepare receipt self-hash mismatch",
    )
    expected_prepare = {
        "schema_version": "c28f_a4_stage_prepare_receipt_v1",
        "status": "FSYNCED_PREPARED_AWAITING_OUTPUT_CAS",
        "goal_id": bundle.plan["goal_id"],
        "attempt_id": bundle.plan["attempt_id"],
        "stage_action": bundle.plan["action"],
        "parent_state_sha256": bundle.plan["state_binding"]["parent_state_sha256"],
        "parent_event_sha256": bundle.plan["state_binding"]["parent_event_sha256"],
        "business_preimage_sha256": bundle.plan["business_preimage_sha256"],
        "output_hashes": output_hashes,
        "static_review_receipt_sha256": bundle.plan["static_review_receipt_sha256"],
        "stage_plan_sha256": observed_plan_sha,
    }
    _require(
        all(prepare.get(field) == value for field, value in expected_prepare.items()),
        "stage prepare receipt binding mismatch",
    )
    _require(
        isinstance(prepare["stage_commit_id"], str)
        and prepare["stage_commit_id"]
        and isinstance(prepare["transaction_id"], str)
        and prepare["transaction_id"],
        "stage prepare transaction identity missing",
    )
    committed = dict(
        adapter.commit_stage_outputs(
            stage_commit_id=str(prepare["stage_commit_id"]),
            bundle=bundle,
        )
    )
    commit_fields = {
        "schema_version",
        "status",
        "goal_id",
        "attempt_id",
        "stage_action",
        "stage_commit_id",
        "parent_state_sha256",
        "parent_event_sha256",
        "business_preimage_sha256",
        "business_postimage_sha256",
        "output_hashes",
        "static_review_receipt_sha256",
        "transaction_id",
        "state_sha256",
        "event_sha256",
        "prepare_receipt_sha256",
        "receipt_sha256",
    }
    _require(set(committed) == commit_fields, "stage commit receipt fields mismatch")
    commit_receipt_sha = _sha256(committed["receipt_sha256"], "commit receipt_sha256")
    _require(
        commit_receipt_sha
        == semantic_sha256(committed, excluded_fields=("receipt_sha256",)),
        "stage commit receipt self-hash mismatch",
    )
    expected_commit = {
        "schema_version": STAGE_COMMIT_RECEIPT_SCHEMA,
        "status": "FSYNCED_CAS_COMMITTED",
        "goal_id": bundle.plan["goal_id"],
        "attempt_id": bundle.plan["attempt_id"],
        "stage_action": bundle.plan["action"],
        "stage_commit_id": prepare["stage_commit_id"],
        "parent_state_sha256": bundle.plan["state_binding"]["parent_state_sha256"],
        "parent_event_sha256": bundle.plan["state_binding"]["parent_event_sha256"],
        "business_preimage_sha256": bundle.plan["business_preimage_sha256"],
        "output_hashes": output_hashes,
        "static_review_receipt_sha256": bundle.plan["static_review_receipt_sha256"],
        "transaction_id": prepare["transaction_id"],
        "prepare_receipt_sha256": prepare_receipt_sha,
    }
    _require(
        all(committed.get(field) == value for field, value in expected_commit.items()),
        "stage commit receipt binding mismatch",
    )
    for field in ("business_postimage_sha256", "state_sha256", "event_sha256"):
        _sha256(committed[field], field)
    return committed


def _validate_g1_observation_scope_receipt(
    receipt: Mapping[str, Any],
    *,
    expected_status: str,
    context: StageContext,
    authority_id: str,
) -> Mapping[str, Any]:
    fields = {
        "schema_version",
        "status",
        "goal_id",
        "authority_id",
        "scope_id",
        "reason_code",
        "instrumentation_complete",
        "observer_restored",
        "nested_ledgers_were_open",
        "nested_ledger_cleanup_failed",
        "nested_ledger_cleanup_errors",
        "all_nested_ledgers_closed",
        "process_observer",
        "receipt_sha256",
    }
    current = dict(
        _exact_mapping(
            receipt,
            tuple(fields),
            "G1 observation scope receipt fields mismatch",
        )
    )
    _require(
        current.get("schema_version")
        == "c28f_a4_g1_observation_scope_receipt_v1"
        and current.get("status") == expected_status
        and current.get("goal_id") == context.goal_id
        and current.get("authority_id") == authority_id
        and current.get("scope_id") == "%s-G1-OUTER" % context.attempt_id
        and current.get("instrumentation_complete") is True
        and current.get("observer_restored") is True
        and current.get("all_nested_ledgers_closed") is True,
        "G1 observation scope receipt binding mismatch",
    )
    cleanup_errors = current.get("nested_ledger_cleanup_errors")
    _require(
        type(current.get("nested_ledgers_were_open")) is bool
        and type(current.get("nested_ledger_cleanup_failed")) is bool
        and isinstance(cleanup_errors, list)
        and all(isinstance(item, str) and bool(item) for item in cleanup_errors)
        and current.get("nested_ledger_cleanup_failed") == bool(cleanup_errors),
        "G1 observation nested-ledger cleanup receipt",
    )
    _require(
        (
            expected_status == "CLOSED_SUCCESS"
            and current.get("reason_code") is None
            and current.get("nested_ledgers_were_open") is False
            and current.get("nested_ledger_cleanup_failed") is False
            and cleanup_errors == []
        )
        or (
            expected_status == "ABORTED_FAIL_CLOSED"
            and isinstance(current.get("reason_code"), str)
            and bool(current["reason_code"])
        ),
        "G1 observation scope reason code/status mismatch",
    )
    receipt_sha = _sha256(
        current.get("receipt_sha256"),
        "G1 observation receipt SHA",
    )
    _require(
        receipt_sha
        == semantic_sha256(current, excluded_fields=("receipt_sha256",)),
        "G1 observation scope receipt self-hash mismatch",
    )
    observed = current.get("process_observer")
    _require(isinstance(observed, Mapping), "G1 process observer receipt missing")
    if expected_status == "CLOSED_SUCCESS":
        _require(
            observed.get("instrumentation_complete") is True
            and observed.get("sys_profile_restored") is True
            and observed.get("thread_profile_restored") is True
            and observed.get("regular_content_open_count") == 0
            and observed.get("write_mutation_event_count") == 0
            and observed.get("data_loader_call_count") == 0
            and observed.get("model_forward_call_count") == 0
            and observed.get("subprocess_call_count") == 0
            and observed.get("unknown_preopened_regular_fd_count") == 0
            and observed.get("preopened_regular_fd_change_count") == 0
            and observed.get("new_regular_fd_count") == 0,
            "G1 outer observer detected forbidden activity",
        )
    return current


def build_g1_bundle(
    *,
    context: StageContext,
    g1_control_handle: Any,
    budget: EvalBudgetBook,
) -> StageBundle:
    """Run every G1 business action inside one fail-closed observer scope."""

    context.validate(G1_ACTION, "PREFLIGHT_OK")
    from .a4_control import A4G1ControlHandle, validate_g1_control_handle

    _require(
        type(g1_control_handle) is A4G1ControlHandle,
        "concrete A4G1ControlHandle required",
    )
    for method_name in (
        "open_observation_scope_once",
        "close_observation_scope_success",
        "abort_observation_scope",
        "open_mediated_ledger",
        "close_mediated_ledger",
        "assert_all_ledgers_closed",
    ):
        _require(
            callable(getattr(g1_control_handle, method_name, None)),
            "G1 concrete control observation unavailable: %s" % method_name,
        )
    try:
        control_binding = validate_g1_control_handle(g1_control_handle)
    except Exception as exc:
        raise StageContractError("G1 control handle validation failed") from exc
    expected_control_anchor = {
        "goal_id": context.goal_id,
        "attempt_id": context.attempt_id,
        "committed_transaction_id": context.control_transaction_id,
        "committed_state_sha256": context.committed_control_state_sha256,
        "committed_event_sha256": context.committed_control_event_sha256,
        "frozen_authority_sha256": context.frozen_g1_registry_authority_sha256,
        "persisted_global_snapshot_sha256": context.frozen_global_ledger_snapshot_sha256,
    }
    _require(
        all(
            control_binding.get(field) == value
            for field, value in expected_control_anchor.items()
        )
        and g1_control_handle.capability_sha256
        == context.control_anchor_receipt_sha256,
        "G1 control handle/context anchor mismatch",
    )
    authority = control_binding.get("registry_authority")
    _require(isinstance(authority, Mapping), "G1 registry authority missing")
    authority_id = str(authority.get("authority_id"))
    observation_scope = g1_control_handle.open_observation_scope_once(
        scope_id="%s-G1-OUTER" % context.attempt_id,
        authority_id=authority_id,
    )
    try:
        preliminary = _build_g1_bundle_observed(
            context=context,
            g1_control_handle=g1_control_handle,
            budget=budget,
        )
    except BaseException:
        abort_receipt = g1_control_handle.abort_observation_scope(
            observation_scope,
            reason_code="G1_STAGE_EXCEPTION",
        )
        _validate_g1_observation_scope_receipt(
            abort_receipt,
            expected_status="ABORTED_FAIL_CLOSED",
            context=context,
            authority_id=authority_id,
        )
        g1_control_handle.assert_all_ledgers_closed()
        raise
    try:
        success_receipt = g1_control_handle.close_observation_scope_success(
            observation_scope
        )
    except BaseException:
        try:
            abort_receipt = g1_control_handle.abort_observation_scope(
                observation_scope,
                reason_code="G1_OBSERVER_CLOSE_EXCEPTION",
            )
        except Exception:
            g1_control_handle.assert_all_ledgers_closed()
        else:
            _validate_g1_observation_scope_receipt(
                abort_receipt,
                expected_status="ABORTED_FAIL_CLOSED",
                context=context,
                authority_id=authority_id,
            )
            g1_control_handle.assert_all_ledgers_closed()
        raise
    if success_receipt.get("status") == "ABORTED_FAIL_CLOSED":
        _validate_g1_observation_scope_receipt(
            success_receipt,
            expected_status="ABORTED_FAIL_CLOSED",
            context=context,
            authority_id=authority_id,
        )
        g1_control_handle.assert_all_ledgers_closed()
        raise StageContractError("G1 outer observation scope aborted fail-closed")
    validated_scope_receipt = _validate_g1_observation_scope_receipt(
        success_receipt,
        expected_status="CLOSED_SUCCESS",
        context=context,
        authority_id=authority_id,
    )
    g1_control_handle.assert_all_ledgers_closed()
    outputs = dict(preliminary.output_bytes)
    outputs[G1_OUTPUTS[5]] = _canonical_file(validated_scope_receipt)
    input_hashes = dict(preliminary.plan["input_hashes"])
    input_hashes["g1_outer_observation_scope"] = validated_scope_receipt[
        "receipt_sha256"
    ]
    return _build_stage_bundle(
        _builder_issuer=_STAGE_BUNDLE_ISSUER,
        context=context,
        action=G1_ACTION,
        result_status="FIREWALL_OK",
        next_action=G2_ACTION,
        outputs=outputs,
        input_hashes=input_hashes,
        declared_safety_counts=dict(
            preliminary.plan["declared_safety_counts"]
        ),
        commit_eligible=True,
    )


def _build_g1_bundle_observed(
    *,
    context: StageContext,
    g1_control_handle: Any,
    budget: EvalBudgetBook,
) -> StageBundle:
    context.validate(G1_ACTION, "PREFLIGHT_OK")
    from .a4_control import A4G1ControlHandle, validate_g1_control_handle

    _require(
        type(g1_control_handle) is A4G1ControlHandle,
        "concrete A4G1ControlHandle required",
    )
    for method_name in (
        "open_mediated_ledger",
        "close_mediated_ledger",
        "assert_all_ledgers_closed",
    ):
        _require(
            callable(getattr(g1_control_handle, method_name, None)),
            "G1 concrete control mediation unavailable: %s" % method_name,
        )
    try:
        control_binding = validate_g1_control_handle(g1_control_handle)
    except Exception as exc:
        raise StageContractError("G1 control handle validation failed") from exc
    _require(
        control_binding.get("schema_version") == "c28f_a4_g1_control_handle_v1",
        "G1 control handle schema mismatch",
    )
    expected_control_anchor = {
        "goal_id": context.goal_id,
        "attempt_id": context.attempt_id,
        "committed_transaction_id": context.control_transaction_id,
        "committed_state_sha256": context.committed_control_state_sha256,
        "committed_event_sha256": context.committed_control_event_sha256,
        "frozen_authority_sha256": context.frozen_g1_registry_authority_sha256,
        "persisted_global_snapshot_sha256": context.frozen_global_ledger_snapshot_sha256,
    }
    _require(
        all(control_binding.get(field) == value for field, value in expected_control_anchor.items()),
        "G1 control handle/context anchor mismatch",
    )
    _require(
        g1_control_handle.capability_sha256
        == context.control_anchor_receipt_sha256,
        "G1 concrete handle capability SHA/context receipt mismatch",
    )
    g0_expected_registry_authority = control_binding.get("registry_authority")
    persisted_global_ledger_snapshot = control_binding.get("global_snapshot")
    _require(
        isinstance(g0_expected_registry_authority, Mapping)
        and isinstance(persisted_global_ledger_snapshot, Mapping),
        "G1 authority/snapshot must be loaded by concrete control handle",
    )
    authority_id = str(g0_expected_registry_authority.get("authority_id"))
    (
        roots,
        forbidden_rules,
        split_registry,
        metadata_path_intent,
        metadata_dry_run_request,
        mount_snapshot,
        pristine_receipts,
    ) = _registry_from_g0_authority(
        g0_expected_registry_authority,
        context,
    )
    _require(
        bytes_sha256(canonical_json_bytes(budget.as_dict()))
        == g0_expected_registry_authority["eval_budget_sha256"],
        "live eval budget differs from G0 expected registry",
    )
    persisted_snapshot, prior_manifest = _validate_persisted_global_snapshot(
        persisted_global_ledger_snapshot,
        context,
        authority_id,
    )
    guard = DescriptorPathGuard(
        goal_id=context.goal_id,
        authority_id=authority_id,
        roots=roots,
        forbidden_rules=forbidden_rules,
    )
    path_map = canonical_path_map(roots, forbidden_rules)
    negative_report = _run_builtin_negative_suite(
        control_handle=g1_control_handle,
        context=context,
        authority_id=authority_id,
        roots=roots,
        forbidden_rules=forbidden_rules,
        guard=guard,
        split_registry=split_registry,
        budget=budget,
        request=metadata_dry_run_request,
        metadata_path_intent=metadata_path_intent,
    )

    ledger = g1_control_handle.open_mediated_ledger(
        scope_id="%s-G1" % context.attempt_id,
        authority_id=authority_id,
    )
    ledger.record_preopen(
        event_type="PERSISTED_GLOBAL_LEDGER_SNAPSHOT_PREOPEN",
        decision="ALLOW",
        reason_code="FSYNCED_CAS_SNAPSHOT_IMPORTED",
        classification="CONTROL",
        request_fingerprint=bytes_sha256(
            canonical_json_bytes(
                {"snapshot_sha256": persisted_snapshot["snapshot_sha256"]}
            )
        ),
        details={
            "snapshot_sha256": persisted_snapshot["snapshot_sha256"],
            "prior_ledger_manifest_sha256": prior_manifest["manifest_sha256"],
            "incident_count": len(persisted_snapshot["incidents"]),
            "content_opened": False,
        },
    )
    validate_root_registry_preopen(
        roots=roots,
        forbidden_rules=forbidden_rules,
        ledger=ledger,
        detector_id="g1_live_exact_registry",
    )
    metadata_path_grant = guard.authorize(metadata_path_intent, ledger)
    _require(
        metadata_path_grant.operation == "READ_METADATA"
        and metadata_path_grant.existing
        and metadata_path_grant.final_identity is not None,
        "normal G1 metadata descriptor resolution did not complete",
    )
    authorization, budget_after = split_registry.authorize(
        metadata_dry_run_request,
        budget,
        ledger,
        dry_run=True,
    )
    _require(authorization.dry_run and not authorization.model_forward_allowed, "G1 must authorize metadata dry-run only")
    _require(budget_after.as_dict() == budget.as_dict(), "metadata dry-run consumed an evaluation token")
    g1_closure = g1_control_handle.close_mediated_ledger(ledger)
    _require(
        g1_closure.get("instrumentation_complete") is True,
        "controller did not close G1 instrumentation",
    )
    g1_rows = g1_closure.get("rows")
    _require(isinstance(g1_rows, list) and g1_rows, "controller G1 ledger rows missing")
    observed_tripwire = dict(g1_closure.get("io_tripwire", {}))
    _require(
        observed_tripwire
        == {
            "data_loader_calls": 0,
            "content_open_calls": 0,
            "model_forward_calls": 0,
            "output_write_calls": 0,
        },
        "controller observed I/O during G1 metadata-only stage",
    )
    g1_delta_manifest = derive_safety_manifest(
        g1_rows,
        expected_scope_id="%s-G1" % context.attempt_id,
        expected_goal_id=context.goal_id,
        expected_authority_id=authority_id,
    )
    safety_manifest = _cumulative_safety_manifest(
        current_manifest=g1_delta_manifest,
        prior_manifest=prior_manifest,
        persisted_snapshot=persisted_snapshot,
    )
    _require(
        safety_manifest["claims"]["safe_for_nonprotected_f0_metadata_dry_run"]
        is True,
        "cumulative persisted G1 safety manifest is not safe",
    )

    firewall_base = {
        "schema_version": "c28f_g1_safety_firewall_v1",
        "goal_id": context.goal_id,
        "attempt_id": context.attempt_id,
        "authority_id": authority_id,
        "path_registry": path_map,
        "path_registry_g0_authority_sha256": g0_expected_registry_authority[
            "authority_sha256"
        ],
        "mount_domain_snapshot": mount_snapshot,
        "pristine_write_root_receipts": list(pristine_receipts),
        "split_registry": split_registry.registry(),
        "normal_metadata_descriptor_grant": metadata_path_grant.as_dict(),
        "metadata_dry_run_authorization": authorization.as_dict(),
        "eval_budget_unchanged": budget.as_dict(),
        "negative_test_report_sha256": negative_report["report_sha256"],
        "ledger_safety_manifest_sha256": safety_manifest["manifest_sha256"],
        "persisted_global_snapshot_sha256": persisted_snapshot["snapshot_sha256"],
        "protected_id_mapping": {
            "status": "DEFERRED_REQUIRES_PROJECTED_SOURCE",
            "earliest_stage": "G4",
            "g1_invocation_count": 0,
            "g2_invocation_count": 0,
            "mixed_protected_stream_opened": False,
            "projected_source_opened": False,
            "required_contract": {
                "projector": "a4_control.A4G4RunningProjectionReader",
                "reviewed_implementation_schema": PROJECTOR_SCHEMA_VERSION,
                "one_shot_committed_intent_before_source_open": True,
                "control_result_seal_binds_exact_projected_two_field_postimage": True,
                "unexpected_field_rejected_before_value_materialization": True,
            },
        },
        "forbidden_defaults": {
            "protected_split_default": None,
            "holdout_or_official_entrypoint": None,
            "legacy_root_authorization_marker": None,
        },
        "actual_counts": {
            "protected_content_opens": safety_manifest["cumulative_totals"][
                "protected_content_open_count"
            ],
            "protected_id_mapping_content_opens": safety_manifest[
                "cumulative_totals"
            ]["protected_id_mapping_content_open_count"],
            "model_forward_evaluations": safety_manifest["cumulative_totals"][
                "model_forward_count"
            ],
            "historical_write_bytes": safety_manifest["cumulative_totals"][
                "historical_write_bytes"
            ],
            "incident_count": safety_manifest["incident_count"],
        },
    }
    firewall = {**firewall_base, "firewall_sha256": semantic_sha256(firewall_base)}
    outputs = {
        G1_OUTPUTS[0]: _canonical_file(firewall),
        G1_OUTPUTS[1]: _canonical_file(negative_report),
        G1_OUTPUTS[2]: _canonical_file(path_map),
        G1_OUTPUTS[3]: _canonical_file(safety_manifest),
        G1_OUTPUTS[4]: _canonical_jsonl(g1_rows),
    }
    return _build_stage_bundle(
        _builder_issuer=_STAGE_BUNDLE_ISSUER,
        context=context,
        action=G1_ACTION,
        result_status="FIREWALL_OK",
        next_action=G2_ACTION,
        outputs=outputs,
        input_hashes={
            "negative_probe_suite": G1_NEGATIVE_SUITE_SHA256,
            "path_registry": path_map["canonical_path_map_sha256"],
            "split_registry": split_registry.registry()["registry_sha256"],
            "persisted_global_snapshot": persisted_snapshot["snapshot_sha256"],
            "mount_domain_snapshot": mount_snapshot["snapshot_sha256"],
        },
        declared_safety_counts={
            "protected_content_opens": safety_manifest["cumulative_totals"][
                "protected_content_open_count"
            ],
            "protected_id_mapping_content_opens": safety_manifest[
                "cumulative_totals"
            ]["protected_id_mapping_content_open_count"],
            "historical_write_bytes": safety_manifest["cumulative_totals"][
                "historical_write_bytes"
            ],
            "model_forward_evaluations": safety_manifest["cumulative_totals"][
                "model_forward_count"
            ],
            "real_data_optimizer_updates": safety_manifest[
                "real_data_optimizer_updates"
            ],
        },
        commit_eligible=True,
    )


_REPLAY_EXECUTION_ISSUER = object()


@dataclass(frozen=True)
class ReplayExecution:
    packets: Tuple[Mapping[str, Any], ...]
    power_ground_truth: Mapping[int, Mapping[str, Any]]
    legacy_vcmr_metrics: Mapping[str, Any]
    legacy_metric_unit: str
    current_vcmr_metrics: Mapping[str, Any]
    packet_sha256: str
    ground_truth_sha256: str
    legacy_metrics_sha256: str
    current_metrics_sha256: str
    evidence: Mapping[str, Any]
    control_handle: Any
    _issuer: object

    def __post_init__(self) -> None:
        from .a4_control import A4G2ReplayControlHandle

        _require(
            self._issuer is _REPLAY_EXECUTION_ISSUER
            and type(self.control_handle) is A4G2ReplayControlHandle,
            "ReplayExecution must be issued by the concrete G2 control path",
        )
        for value, field in (
            (self.packet_sha256, "packet_sha256"),
            (self.ground_truth_sha256, "ground_truth_sha256"),
            (self.legacy_metrics_sha256, "legacy_metrics_sha256"),
            (self.current_metrics_sha256, "current_metrics_sha256"),
        ):
            _sha256(value, field)
        _require(
            self.legacy_metric_unit in {"percent", "ratio"},
            "ReplayExecution legacy metric unit invalid",
        )


def _validate_replay_preopen_receipt(
    receipt: Mapping[str, Any],
    *,
    expected_goal_id: str,
    expected_query_manifest_sha256: str,
    expected_source_hashes: Mapping[str, str],
) -> Mapping[str, Any]:
    current = dict(
        _exact_mapping(
            receipt,
            (
                "schema_version",
                "decision",
                "mode",
                "purpose",
                "dataset_role",
                "protected_content",
                "goal_id",
                "attempt_id",
                "committed_transaction_id",
                "committed_state_sha256",
                "committed_event_sha256",
                "control_handle_capability_sha256",
                "query_manifest_sha256",
                "source_hashes",
                "source_registry",
                "access_ledger_event_sha256",
                "safety_manifest_sha256",
                "instrumentation_complete_before_open",
                "unmediated_content_open_count_before",
                "stream_session_open_count_before",
                "receipt_sha256",
            ),
            "replay preopen receipt fields mismatch",
        )
    )
    _require(current.get("schema_version") == "c28f_g2_control_replay_preopen_v1", "replay preopen receipt schema")
    receipt_sha = _sha256(current.get("receipt_sha256"), "replay preopen receipt_sha256")
    _require(receipt_sha == semantic_sha256(current, excluded_fields=("receipt_sha256",)), "replay preopen receipt self-hash")
    _require(current.get("decision") == "ALLOW", "replay control session is not allowed")
    _require(current.get("mode") in {"SYNTHETIC", "LEDGERED_C7"}, "replay mode is invalid")
    _require(current.get("purpose") == "G2_NONPROTECTED_BASELINE_REPLAY", "replay purpose mismatch")
    _require(current.get("dataset_role") == "TRAIN_CALIB_NONPROTECTED", "replay dataset role mismatch")
    _require(current.get("protected_content") is False, "protected replay content forbidden")
    _require(current.get("goal_id") == expected_goal_id, "replay Goal mismatch")
    _require(current.get("query_manifest_sha256") == expected_query_manifest_sha256, "replay query manifest mismatch")
    _require(dict(current.get("source_hashes", {})) == dict(expected_source_hashes), "replay source hash mismatch")
    source_registry = current.get("source_registry")
    _require(
        isinstance(source_registry, list)
        and source_registry
        and len(source_registry) == len(expected_source_hashes),
        "replay source registry is missing or incomplete",
    )
    observed_source_hashes = {}
    for item in source_registry:
        record = dict(
            _exact_mapping(
                item,
                ("source_id", "canonical_path", "sha256", "dataset_role", "protected"),
                "replay source registry record fields mismatch",
            )
        )
        _require(
            isinstance(record["source_id"], str)
            and record["source_id"]
            and isinstance(record["canonical_path"], str)
            and record["canonical_path"].startswith("/")
            and record["dataset_role"] == "TRAIN_CALIB_NONPROTECTED"
            and record["protected"] is False,
            "replay source registry role/path invalid",
        )
        normalized_path = record["canonical_path"].casefold()
        _require(
            not any(
                marker in normalized_path
                for marker in ("official", "holdout", "pseudo_official")
            ),
            "protected/official replay source path forbidden",
        )
        observed_source_hashes[record["source_id"]] = _sha256(
            record["sha256"], "replay source registry sha256"
        )
    _require(
        observed_source_hashes == dict(expected_source_hashes),
        "replay source registry/hash commitments mismatch",
    )
    _sha256(current.get("access_ledger_event_sha256"), "access_ledger_event_sha256")
    _sha256(current.get("safety_manifest_sha256"), "safety_manifest_sha256")
    for field in (
        "committed_state_sha256",
        "committed_event_sha256",
        "control_handle_capability_sha256",
    ):
        _sha256(current.get(field), field)
    _require(
        isinstance(current.get("attempt_id"), str)
        and current["attempt_id"]
        and isinstance(current.get("committed_transaction_id"), str)
        and current["committed_transaction_id"],
        "replay committed control identity missing",
    )
    _require(current.get("instrumentation_complete_before_open") is True, "replay instrumentation is incomplete")
    _require(current.get("unmediated_content_open_count_before") == 0, "unmediated replay open exists")
    _require(current.get("stream_session_open_count_before") == 0, "replay stream opened before authority")
    return current


def execute_ledgered_c7_replay_alignment(
    *,
    control_handle: Any,
    expected_goal_id: str,
    expected_query_manifest_sha256: str,
    expected_source_hashes: Mapping[str, str],
) -> ReplayExecution:
    """Consume exactly one controller-owned replay descriptor session."""

    from .a4_control import A4G2ReplayControlHandle, validate_g2_replay_control_handle

    _require(
        type(control_handle) is A4G2ReplayControlHandle,
        "concrete A4G2ReplayControlHandle required",
    )
    for method_name in (
        "validate_and_consume_preopen",
        "open_aligned_streams_once",
        "read_legacy_vcmr_metrics_once",
        "close_replay_once",
        "commit_parity_derivation_once",
    ):
        _require(
            callable(getattr(control_handle, method_name, None)),
            "G2 concrete control implementation unavailable: %s" % method_name,
        )
    try:
        validate_g2_replay_control_handle(control_handle)
    except Exception as exc:
        raise StageContractError("G2 replay control handle validation failed") from exc

    _sha256(expected_query_manifest_sha256, "expected_query_manifest_sha256")
    for name, digest in expected_source_hashes.items():
        _sha256(digest, "expected_source_hashes.%s" % name)
    receipt = _validate_replay_preopen_receipt(
        control_handle.validate_and_consume_preopen(
            expected_goal_id=expected_goal_id,
            expected_query_manifest_sha256=expected_query_manifest_sha256,
            expected_source_hashes=expected_source_hashes,
        ),
        expected_goal_id=expected_goal_id,
        expected_query_manifest_sha256=expected_query_manifest_sha256,
        expected_source_hashes=expected_source_hashes,
    )
    _require(
        receipt.get("control_handle_capability_sha256")
        == control_handle.capability_sha256,
        "replay preopen receipt/control handle mismatch",
    )
    receipt_sha = receipt["receipt_sha256"]
    source_registry_sha = bytes_sha256(
        canonical_json_bytes(receipt["source_registry"])
    )
    opened = control_handle.open_aligned_streams_once(receipt_sha)
    _require(
        isinstance(opened, tuple) and len(opened) == 5,
        "controller replay stream session shape mismatch",
    )
    (
        prediction_stream,
        score_stream,
        video_id_decoder,
        span_decoder,
        ground_truth_decoder,
    ) = opened
    _require(callable(ground_truth_decoder), "controller ground-truth decoder missing")
    packets = tuple(
        iter_aligned_c7_replay_rows(
            prediction_stream,
            score_stream,
            video_id_decoder=video_id_decoder,
            span_decoder=span_decoder,
        )
    )
    packet_sha = bytes_sha256(canonical_json_bytes(list(packets)))
    power_ground_truth: dict[int, Mapping[str, Any]] = {}
    current_queries = []
    for packet in packets:
        query_id = packet.get("query_id")
        _require(type(query_id) is int and query_id >= 0, "G2 query ID must be canonical desc_id:int")
        ground_truth = dict(ground_truth_decoder(query_id))
        _exact_mapping(
            ground_truth,
            ("gt_video_id", "gt_span_sec", "query_manifest_sha256"),
            "G2 ground-truth decoder fields mismatch",
        )
        gt_video_id = ground_truth.get("gt_video_id")
        gt_span = ground_truth.get("gt_span_sec")
        _require(
            isinstance(gt_video_id, str)
            and gt_video_id
            and isinstance(gt_span, (list, tuple))
            and len(gt_span) == 2
            and all(
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(float(value))
                for value in gt_span
            )
            and float(gt_span[0]) >= 0.0
            and float(gt_span[1]) > float(gt_span[0])
            and ground_truth.get("query_manifest_sha256")
            == expected_query_manifest_sha256,
            "G2 ground truth is invalid or outside frozen query manifest",
        )
        normalized_ground_truth = {
            "gt_video_id": gt_video_id,
            "gt_span_sec": [float(gt_span[0]), float(gt_span[1])],
            "query_manifest_sha256": expected_query_manifest_sha256,
        }
        power_ground_truth[query_id] = normalized_ground_truth
        current_queries.append(
            {
                "query_id": query_id,
                "gt_video_id": gt_video_id,
                "gt_span_sec": normalized_ground_truth["gt_span_sec"],
                "joint_proposals": packet["joint_proposals"],
            }
        )
    current_evaluation = evaluate_joint_vcmr_and_errors(
        current_queries,
        require_probabilities=False,
        expected_query_ids=sorted(power_ground_truth),
        input_is_frozen_nms=True,
    )
    current_vcmr_metrics = dict(current_evaluation["vcmr_metrics"])
    legacy_receipt = dict(
        control_handle.read_legacy_vcmr_metrics_once(receipt_sha)
    )
    _exact_mapping(
        legacy_receipt,
        (
            "schema_version",
            "status",
            "metrics",
            "metric_unit",
            "legacy_metric_artifact_sha256",
            "query_manifest_sha256",
            "preopen_receipt_sha256",
            "control_handle_capability_sha256",
            "receipt_sha256",
        ),
        "G2 legacy metric control receipt fields mismatch",
    )
    legacy_receipt_sha = _sha256(
        legacy_receipt.get("receipt_sha256"),
        "G2 legacy metric receipt SHA",
    )
    _require(
        legacy_receipt_sha
        == semantic_sha256(legacy_receipt, excluded_fields=("receipt_sha256",))
        and legacy_receipt.get("schema_version")
        == "c28f_g2_control_legacy_metrics_receipt_v1"
        and legacy_receipt.get("status") == "FSYNCED_SOURCE_VERIFIED"
        and legacy_receipt.get("metric_unit") in {"percent", "ratio"}
        and legacy_receipt.get("query_manifest_sha256")
        == expected_query_manifest_sha256
        and legacy_receipt.get("preopen_receipt_sha256") == receipt_sha
        and legacy_receipt.get("control_handle_capability_sha256")
        == control_handle.capability_sha256,
        "G2 legacy metrics are not bound to the concrete replay session",
    )
    _sha256(
        legacy_receipt.get("legacy_metric_artifact_sha256"),
        "legacy metric artifact SHA",
    )
    legacy_vcmr_metrics = dict(legacy_receipt.get("metrics", {}))
    ground_truth_sha = bytes_sha256(
        canonical_json_bytes(
            {str(key): value for key, value in sorted(power_ground_truth.items())}
        )
    )
    legacy_metrics_sha = bytes_sha256(canonical_json_bytes(legacy_vcmr_metrics))
    current_metrics_sha = bytes_sha256(canonical_json_bytes(current_vcmr_metrics))
    close = dict(
        control_handle.close_replay_once(
            preopen_receipt_sha256=receipt_sha,
            replay_packet_sha256=packet_sha,
            ground_truth_sha256=ground_truth_sha,
            legacy_metrics_sha256=legacy_metrics_sha,
            current_metrics_sha256=current_metrics_sha,
        )
    )
    _exact_mapping(
        close,
        (
            "schema_version",
            "status",
            "preopen_receipt_sha256",
            "replay_packet_sha256",
            "ground_truth_sha256",
            "legacy_metrics_sha256",
            "current_metrics_sha256",
            "legacy_metric_receipt_sha256",
            "instrumentation_complete",
            "unmediated_content_open_count",
            "protected_content_open_count",
            "model_forward_count",
            "access_ledger_tail_event_sha256",
            "committed_transaction_id",
            "committed_state_sha256",
            "committed_event_sha256",
            "control_handle_capability_sha256",
            "receipt_sha256",
        ),
        "replay close receipt fields mismatch",
    )
    _require(close.get("schema_version") == "c28f_g2_control_replay_close_v1", "replay close receipt schema")
    _require(close.get("status") == "FSYNCED_CLOSED", "replay close receipt status")
    close_sha = _sha256(close.get("receipt_sha256"), "replay close receipt_sha256")
    _require(close_sha == semantic_sha256(close, excluded_fields=("receipt_sha256",)), "replay close receipt self-hash")
    _require(close.get("preopen_receipt_sha256") == receipt_sha, "replay close/preopen mismatch")
    _require(close.get("replay_packet_sha256") == packet_sha, "replay close packet mismatch")
    _require(
        close.get("ground_truth_sha256") == ground_truth_sha
        and close.get("legacy_metrics_sha256") == legacy_metrics_sha
        and close.get("current_metrics_sha256") == current_metrics_sha
        and close.get("legacy_metric_receipt_sha256") == legacy_receipt_sha,
        "replay close metric/ground-truth binding mismatch",
    )
    _require(close.get("instrumentation_complete") is True, "replay close instrumentation incomplete")
    _require(close.get("unmediated_content_open_count") == 0, "replay close has unmediated opens")
    _require(close.get("protected_content_open_count") == 0, "replay close has protected opens")
    _require(close.get("model_forward_count") == 0, "G2 replay cannot execute model forward")
    for field in (
        "access_ledger_tail_event_sha256",
        "committed_state_sha256",
        "committed_event_sha256",
        "control_handle_capability_sha256",
    ):
        _sha256(close.get(field), field)
    _require(
        close.get("control_handle_capability_sha256")
        == receipt.get("control_handle_capability_sha256"),
        "replay close/control handle mismatch",
    )
    evidence_base = {
        "schema_version": "c28f_g2_replay_alignment_evidence_v1",
        "mode": receipt["mode"],
        "preopen_receipt_sha256": receipt_sha,
        "close_receipt_sha256": close_sha,
        "query_manifest_sha256": expected_query_manifest_sha256,
        "source_hashes": dict(expected_source_hashes),
        "source_registry_sha256": source_registry_sha,
        "packet_count": len(packets),
        "proposal_count": sum(len(packet["joint_proposals"]) for packet in packets),
        "replay_packet_sha256": packet_sha,
        "ground_truth_sha256": ground_truth_sha,
        "legacy_metric_receipt_sha256": legacy_receipt_sha,
        "legacy_metrics_sha256": legacy_metrics_sha,
        "legacy_metric_unit": legacy_receipt["metric_unit"],
        "current_metrics_sha256": current_metrics_sha,
        "protected_content_open_count": 0,
        "model_forward_count": 0,
    }
    evidence = {**evidence_base, "evidence_sha256": semantic_sha256(evidence_base)}
    return ReplayExecution(
        packets=packets,
        power_ground_truth=power_ground_truth,
        legacy_vcmr_metrics=legacy_vcmr_metrics,
        legacy_metric_unit=str(legacy_receipt["metric_unit"]),
        current_vcmr_metrics=current_vcmr_metrics,
        packet_sha256=packet_sha,
        ground_truth_sha256=ground_truth_sha,
        legacy_metrics_sha256=legacy_metrics_sha,
        current_metrics_sha256=current_metrics_sha,
        evidence=evidence,
        control_handle=control_handle,
        _issuer=_REPLAY_EXECUTION_ISSUER,
    )


def build_g2_bundle(
    *,
    context: StageContext,
    fixture: Mapping[str, Any],
    hand_expected_fixture: Mapping[str, Any],
    g1_safety_manifest_sha256: str,
    replay_query_manifest_sha256: str,
    replay_artifact_commitments: Mapping[str, str],
    replay_execution: Optional[ReplayExecution] = None,
) -> StageBundle:
    context.validate(G2_ACTION, "FIREWALL_OK")
    fixture_source_sha256 = FROZEN_METRIC_FIXTURE_FILE_SHA256
    hand_expected_fixture_source_sha256 = FROZEN_METRIC_EXPECTED_FILE_SHA256
    fixture_canonical_sha = bytes_sha256(_canonical_file(dict(fixture)))
    expected_fixture_canonical_sha = bytes_sha256(
        _canonical_file(dict(hand_expected_fixture))
    )
    _require(
        fixture_canonical_sha == FROZEN_METRIC_FIXTURE_CANONICAL_SHA256
        and expected_fixture_canonical_sha
        == FROZEN_METRIC_EXPECTED_CANONICAL_SHA256,
        "G2 caller fixture mappings differ from reviewed canonical bytes",
    )
    fixture_review_binding_sha = semantic_sha256(
        {
            "schema_version": "c28f_g2_reviewed_metric_fixture_binding_v1",
            "static_review_receipt_sha256": (
                context.static_review_receipt_sha256
            ),
            "fixture_path": "tests/c28f_v5/fixtures/metric_fixture_v1.json",
            "fixture_file_sha256": FROZEN_METRIC_FIXTURE_FILE_SHA256,
            "fixture_canonical_sha256": (
                FROZEN_METRIC_FIXTURE_CANONICAL_SHA256
            ),
            "expected_path": (
                "tests/c28f_v5/fixtures/metric_fixture_v1_expected.json"
            ),
            "expected_file_sha256": FROZEN_METRIC_EXPECTED_FILE_SHA256,
            "expected_canonical_sha256": (
                FROZEN_METRIC_EXPECTED_CANONICAL_SHA256
            ),
        }
    )
    _sha256(g1_safety_manifest_sha256, "g1_safety_manifest_sha256")
    _sha256(replay_query_manifest_sha256, "replay_query_manifest_sha256")
    required_artifacts = {"cache_npz", "predictions_jsonl", "scores_jsonl"}
    _require(set(replay_artifact_commitments) == required_artifacts, "replay artifact commitment set is not exact")
    for name, digest in replay_artifact_commitments.items():
        _sha256(digest, "replay_artifact_commitments.%s" % name)
    expected_replay_source_hashes = dict(FROZEN_LEGACY_SEMANTIC_SOURCE_HASHES)
    expected_replay_source_hashes.update(
        {
            "artifact:%s" % name: digest
            for name, digest in sorted(replay_artifact_commitments.items())
        }
    )
    expected_replay_source_hashes["query_manifest"] = replay_query_manifest_sha256

    fixture_result = evaluate_fixture(fixture)
    observed_expected = fixture_expected_projection(fixture_result)
    _require(
        canonical_json_bytes(observed_expected) == canonical_json_bytes(dict(hand_expected_fixture)),
        "metric fixture differs from independently supplied hand expectation",
    )
    schema = metric_schema()
    gates = gate_schema()

    replay_status = "DEFERRED_REQUIRES_CONCRETE_G2_CONTROL_HANDLE"
    parity = None
    validated_parity_receipt = None
    power_replay_rows: Optional[Tuple[Mapping[str, Any], ...]] = None
    historical_power_reference: Optional[Mapping[str, Any]] = None
    power_contract = frozen_power_evaluator_contract()
    _require(
        power_contract.get("contract_sha256")
        == semantic_sha256(
            power_contract,
            excluded_fields=("contract_sha256",),
        ),
        "G2 frozen power evaluator contract self-hash mismatch",
    )
    _require(
        power_contract.get("observable_effects") == list(POWER_OUTCOME_KEYS),
        "G2 frozen power outcome key/order drift",
    )
    _require(
        isinstance(power_contract.get("power_row_schema"), Mapping)
        and power_contract["power_row_schema"].get("outcomes")
        == "EXACT_SIX_KEY_BINARY_MAPPING"
        and power_contract["power_row_schema"].get("row_fields")
        == [
            "schema_version",
            "desc_id",
            "gt_video_id",
            "outcomes",
            "replay_packet_sha256",
            "query_manifest_sha256",
        ],
        "G2 frozen sufficient-statistics row schema drift",
    )
    outcome_derivation_contract = power_contract.get(
        "outcome_derivation_contract"
    )
    outcome_derivation_contract_sha = _sha256(
        power_contract.get("outcome_derivation_contract_sha256"),
        "outcome_derivation_contract_sha256",
    )
    _require(
        isinstance(outcome_derivation_contract, Mapping)
        and outcome_derivation_contract.get(
            "outcome_derivation_contract_sha256"
        )
        == outcome_derivation_contract_sha
        and outcome_derivation_contract_sha
        == semantic_sha256(
            outcome_derivation_contract,
            excluded_fields=("outcome_derivation_contract_sha256",),
        ),
        "G2 outcome derivation contract hash mismatch",
    )
    _require(
        outcome_derivation_contract.get("schema_version")
        == "c28f_g2_power_outcome_derivation_contract_v1"
        and outcome_derivation_contract.get("input_semantics")
        == "SEALED_FROZEN_METRIC_PROPOSALS_AND_AUTHORIZED_GT_ONLY_INSIDE_G2"
        and outcome_derivation_contract.get("persisted_row_semantics")
        == "SIX_BINARY_SUFFICIENT_STATISTICS_NO_PROPOSALS_NO_GT_SPAN"
        and outcome_derivation_contract.get("proposal_row_schema")
        == ["video_id", "start_sec", "end_sec", "score"]
        and outcome_derivation_contract.get("proposal_order")
        == "SCORE_DESC_STABLE_FROZEN_METRIC_NMS_ORDER"
        and outcome_derivation_contract.get("nms_contract")
        == {
            "scope": "PER_VIDEO",
            "iou_threshold": 0.7,
            "maximum_results": 100,
            "tie_break": "FROZEN_EVALUATOR_STABLE_ORDER",
        }
        and outcome_derivation_contract.get("temporal_iou_contract")
        == {
            "intersection": (
                "max(0,min(end_a,end_b)-max(start_a,start_b))"
            ),
            "union": "max(end_a,end_b)-min(start_a,start_b)",
            "zero_union_result": 0.0,
        }
        and outcome_derivation_contract.get("outcome_key_order")
        == list(POWER_OUTCOME_KEYS)
        and outcome_derivation_contract.get("outcome_value_contract")
        == "EXACT_INT_0_OR_1_BOOL_FORBIDDEN"
        and outcome_derivation_contract.get("outcome_rules")
        == {
            "VCMR_R1_IOU_0_7": (
                "ANY_TOP1_GT_VIDEO_AND_TEMPORAL_IOU_GE_0.7"
            ),
            "VCMR_R5_IOU_0_7": (
                "ANY_TOP5_GT_VIDEO_AND_TEMPORAL_IOU_GE_0.7"
            ),
            "VCMR_R10_IOU_0_7": (
                "ANY_TOP10_GT_VIDEO_AND_TEMPORAL_IOU_GE_0.7"
            ),
            "LEGACY_UNIQUE_VIDEO_TRUE_VR_R100": (
                "GT_VIDEO_IN_FIRST_100_UNIQUE_VIDEO_IDS"
            ),
            "WRONG_VIDEO_TOP1": "TOP1_VIDEO_ID_NE_GT_VIDEO_ID",
            "CORRECT_VIDEO_WRONG_SPAN_TOP1": (
                "TOP1_VIDEO_ID_EQ_GT_VIDEO_ID_AND_TEMPORAL_IOU_LT_0.5"
            ),
        },
        "G2 implementation/outcome derivation contract semantics drift",
    )
    commit_eligible = False
    if replay_execution is not None:
        _require(
            type(replay_execution) is ReplayExecution
            and replay_execution._issuer is _REPLAY_EXECUTION_ISSUER,
            "concrete sealed ReplayExecution required",
        )
        evidence = dict(replay_execution.evidence)
        observed_ground_truth_sha = bytes_sha256(
            canonical_json_bytes(
                {
                    str(key): value
                    for key, value in sorted(
                        replay_execution.power_ground_truth.items()
                    )
                }
            )
        )
        _require(
            bytes_sha256(canonical_json_bytes(list(replay_execution.packets)))
            == replay_execution.packet_sha256
            == evidence.get("replay_packet_sha256")
            and observed_ground_truth_sha
            == replay_execution.ground_truth_sha256
            == evidence.get("ground_truth_sha256"),
            "ReplayExecution packet/ground-truth postimage mutated",
        )
        _require(evidence.get("schema_version") == "c28f_g2_replay_alignment_evidence_v1", "replay evidence schema")
        evidence_sha = _sha256(evidence.get("evidence_sha256"), "replay evidence_sha256")
        _require(evidence_sha == semantic_sha256(evidence, excluded_fields=("evidence_sha256",)), "replay evidence self-hash")
        _require(evidence.get("mode") in {"SYNTHETIC", "LEDGERED_C7"}, "replay evidence mode")
        _require(evidence.get("query_manifest_sha256") == replay_query_manifest_sha256, "replay evidence query manifest mismatch")
        _require(dict(evidence.get("source_hashes", {})) == expected_replay_source_hashes, "replay evidence source authority mismatch")
        _sha256(evidence.get("source_registry_sha256"), "replay source registry SHA")
        _require(evidence.get("protected_content_open_count") == 0, "replay evidence contains protected opens")
        _require(evidence.get("model_forward_count") == 0, "replay evidence contains model forward")
        legacy_vcmr_metrics = dict(replay_execution.legacy_vcmr_metrics)
        current_vcmr_metrics = dict(replay_execution.current_vcmr_metrics)
        _require(
            evidence.get("legacy_metrics_sha256")
            == replay_execution.legacy_metrics_sha256
            == bytes_sha256(canonical_json_bytes(legacy_vcmr_metrics))
            and evidence.get("current_metrics_sha256")
            == replay_execution.current_metrics_sha256
            == bytes_sha256(canonical_json_bytes(current_vcmr_metrics)),
            "sealed replay metric hashes differ from ReplayExecution",
        )
        _require(
            evidence.get("legacy_metric_unit")
            == replay_execution.legacy_metric_unit,
            "sealed replay legacy metric unit mismatch",
        )
        parity = compare_vcmr_parity(
            legacy_vcmr_metrics,
            current_vcmr_metrics,
            field_map=C7_LEGACY_VCMR_PARITY_FIELD_MAP,
            tolerance=HISTORICAL_PARITY_TOLERANCE,
            legacy_unit=replay_execution.legacy_metric_unit,
            current_unit="ratio",
        )
        replay_status = (
            "SYNTHETIC_ALIGNMENT_ONLY_REAL_REPLAY_DEFERRED"
            if evidence.get("mode") == "SYNTHETIC"
            else "LEDGERED_C7_REPLAY_COMPLETE"
        )
        if evidence.get("mode") == "LEDGERED_C7":
            normalized_power_rows = []
            for packet in replay_execution.packets:
                query_id = packet.get("query_id")
                _require(
                    type(query_id) is int
                    and query_id >= 0
                    and query_id in replay_execution.power_ground_truth,
                    "G2 power replay row lacks canonical desc_id/ground truth",
                )
                normalized_power_rows.append(
                    _derive_g2_power_outcome_row(
                        packet=packet,
                        ground_truth=replay_execution.power_ground_truth[
                            query_id
                        ],
                        replay_packet_sha256=evidence[
                            "replay_packet_sha256"
                        ],
                        query_manifest_sha256=replay_query_manifest_sha256,
                    )
                )
            _require(
                [row["desc_id"] for row in normalized_power_rows]
                == sorted({row["desc_id"] for row in normalized_power_rows}),
                "G2 power replay desc IDs are duplicate or noncanonical",
            )
            power_replay_rows = tuple(normalized_power_rows)
            from .a4_roles import build_historical_power_reference_from_rows

            try:
                historical_power_reference = (
                    build_historical_power_reference_from_rows(
                        power_replay_rows,
                        power_rows_sha256=bytes_sha256(
                            canonical_json_bytes(list(power_replay_rows))
                        ),
                        replay_packet_sha256=evidence[
                            "replay_packet_sha256"
                        ],
                        query_manifest_sha256=replay_query_manifest_sha256,
                        source_registry_sha256=evidence[
                            "source_registry_sha256"
                        ],
                    )
                )
            except Exception as exc:
                raise StageContractError(
                    "G2 historical power reference manifest derivation failed"
                ) from exc
            _require(
                evidence.get("mode") == "LEDGERED_C7",
                "synthetic replay cannot carry a real parity derivation receipt",
            )
            current_receipt = dict(
                replay_execution.control_handle.commit_parity_derivation_once(
                    replay_evidence_sha256=evidence_sha,
                    replay_packet_sha256=evidence["replay_packet_sha256"],
                    legacy_metrics_sha256=bytes_sha256(
                        canonical_json_bytes(dict(legacy_vcmr_metrics))
                    ),
                    current_metrics_sha256=bytes_sha256(
                        canonical_json_bytes(dict(current_vcmr_metrics))
                    ),
                    legacy_metric_unit=replay_execution.legacy_metric_unit,
                    static_review_receipt_sha256=context.static_review_receipt_sha256,
                )
            )
            _exact_mapping(
                current_receipt,
                (
                    "schema_version",
                    "decision",
                    "mode",
                    "replay_evidence_sha256",
                    "replay_packet_sha256",
                    "legacy_metrics_sha256",
                    "current_metrics_sha256",
                    "legacy_metric_unit",
                    "static_review_receipt_sha256",
                    "input_boundary",
                    "direct_historical_artifact_read",
                    "control_handle_capability_sha256",
                    "committed_transaction_id",
                    "committed_state_sha256",
                    "committed_event_sha256",
                    "receipt_sha256",
                ),
                "parity derivation receipt fields mismatch",
            )
            _require(
                current_receipt.get("schema_version")
                == "c28f_g2_parity_derivation_receipt_v1",
                "parity derivation receipt schema",
            )
            receipt_sha = _sha256(
                current_receipt.get("receipt_sha256"),
                "parity derivation receipt_sha256",
            )
            _require(
                receipt_sha
                == semantic_sha256(
                    current_receipt,
                    excluded_fields=("receipt_sha256",),
                ),
                "parity derivation receipt self-hash",
            )
            _require(current_receipt.get("decision") == "PASS", "parity derivation receipt decision")
            _require(current_receipt.get("mode") == "LEDGERED_C7", "parity derivation receipt mode")
            _require(
                current_receipt.get("legacy_metric_unit")
                == replay_execution.legacy_metric_unit,
                "parity derivation legacy unit mismatch",
            )
            _require(
                current_receipt.get("replay_evidence_sha256") == evidence_sha,
                "parity derivation/replay evidence mismatch",
            )
            _require(
                current_receipt.get("replay_packet_sha256")
                == evidence.get("replay_packet_sha256"),
                "parity derivation/replay packet mismatch",
            )
            _require(
                current_receipt.get("legacy_metrics_sha256")
                == bytes_sha256(canonical_json_bytes(dict(legacy_vcmr_metrics))),
                "parity derivation legacy metrics mismatch",
            )
            _require(
                current_receipt.get("current_metrics_sha256")
                == bytes_sha256(canonical_json_bytes(dict(current_vcmr_metrics))),
                "parity derivation current metrics mismatch",
            )
            _require(
                current_receipt.get("static_review_receipt_sha256")
                == context.static_review_receipt_sha256,
                "parity derivation static review mismatch",
            )
            _require(
                current_receipt.get("input_boundary") == "A4G2ReplayControlHandle",
                "parity derivation bypassed concrete G2 control handle",
            )
            _require(
                current_receipt.get("direct_historical_artifact_read") is False,
                "parity derivation direct historical read forbidden",
            )
            validated_parity_receipt = current_receipt
        commit_eligible = bool(
            evidence.get("mode") == "LEDGERED_C7"
            and parity["status"] == "PASS"
            and validated_parity_receipt is not None
            and power_replay_rows is not None
            and historical_power_reference is not None
        )

    replay_plan_base = {
        "schema_version": "c28f_g2_baseline_replay_plan_v1",
        "status": replay_status,
        "purpose": "G2_NONPROTECTED_BASELINE_REPLAY",
        "dataset_role": "TRAIN_CALIB_NONPROTECTED",
        "protected_content": False,
        "query_manifest_sha256": replay_query_manifest_sha256,
        "artifact_commitments": dict(sorted(replay_artifact_commitments.items())),
        "semantic_source_hashes": dict(sorted(FROZEN_LEGACY_SEMANTIC_SOURCE_HASHES.items())),
        "control_contract": {
            "interface": "A4G2ReplayControlHandle",
            "authority_validated_before_single_use_stream_open": True,
            "prediction_and_score_alignment": "q,rank exact; rank contiguous from zero",
            "unmediated_content_open_allowed": False,
            "synthetic_test_does_not_authorize_real_replay": True,
            "power_row_boundary": {
                "schema_version": POWER_REPLAY_ROW_SCHEMA,
                "exact_fields": [
                    "schema_version",
                    "desc_id",
                    "gt_video_id",
                    "outcomes",
                    "replay_packet_sha256",
                    "query_manifest_sha256",
                ],
                "outcome_keys": list(POWER_OUTCOME_KEYS),
                "full_joint_proposals_persisted": False,
                "ground_truth_span_persisted": False,
                "outcome_derivation_contract_sha256": (
                    outcome_derivation_contract_sha
                ),
            },
        },
        "replay_execution_evidence": None if replay_execution is None else dict(replay_execution.evidence),
        "parity_derivation_receipt": validated_parity_receipt,
        "historical_parity": parity,
        "stream_session_open_count_during_deferred_plan_build": 0,
        "model_forward_count": 0,
        "protected_content_open_count": 0,
    }
    replay_plan = {**replay_plan_base, "replay_plan_sha256": semantic_sha256(replay_plan_base)}
    equivalence_base = {
        "schema_version": "c28f_g2_evaluator_equivalence_v1",
        "fixture_equivalence": "PASS",
        "fixture_source_sha256": fixture_source_sha256,
        "fixture_expected_source_sha256": (
            hand_expected_fixture_source_sha256
        ),
        "fixture_expected_sha256": bytes_sha256(canonical_json_bytes(observed_expected)),
        "schema_version_under_test": schema["schema_version"],
        "ratio_units": True,
        "stable_tie_break": schema["tie_break"],
        "legacy_field_overwrite": False,
        "missing_fields_fail_closed": True,
        "high_confidence_without_prefit_temperature": "unavailable",
        "historical_parity": parity,
        "real_baseline_replay_status": replay_status,
    }
    equivalence = {**equivalence_base, "equivalence_sha256": semantic_sha256(equivalence_base)}
    gate_contribution_base = {
        "schema_version": "c28f_g2_gate_registry_contribution_v1",
        "gate_schema": gates,
        "metric_schema_sha256": bytes_sha256(canonical_json_bytes(schema)),
        "fixture_equivalence": "PASS",
        "historical_parity": None if parity is None else parity["status"],
        "commit_eligible": commit_eligible,
    }
    gate_contribution = {**gate_contribution_base, "contribution_sha256": semantic_sha256(gate_contribution_base)}
    outputs = {
        G2_OUTPUTS[0]: _canonical_file(schema),
        G2_OUTPUTS[1]: _canonical_file(gates),
        G2_OUTPUTS[2]: _canonical_file(gate_contribution),
        G2_OUTPUTS[3]: _canonical_file(equivalence),
        G2_OUTPUTS[4]: _canonical_file(observed_expected),
        G2_OUTPUTS[5]: _canonical_file(replay_plan),
        G2_OUTPUTS[7]: _canonical_file(power_contract),
    }
    if power_replay_rows is not None:
        outputs[G2_OUTPUTS[6]] = _canonical_jsonl(power_replay_rows)
    if historical_power_reference is not None:
        outputs[G2_OUTPUTS[8]] = _canonical_file(
            historical_power_reference
        )
    return _build_stage_bundle(
        _builder_issuer=_STAGE_BUNDLE_ISSUER,
        context=context,
        action=G2_ACTION,
        result_status=("METRIC_CONTRACT_OK" if commit_eligible else "METRIC_CONTRACT_PREVIEW_NOT_COMMITTABLE"),
        next_action=(
            G2_TRAIN_JSONL_INDEX_BOOTSTRAP_ACTION
            if commit_eligible
            else G2_ACTION
        ),
        outputs=outputs,
        input_hashes={
            "fixture_source": fixture_source_sha256,
            "fixture_expected_source": hand_expected_fixture_source_sha256,
            "fixture_canonical": fixture_canonical_sha,
            "fixture_expected_canonical": expected_fixture_canonical_sha,
            "fixture_static_review_binding": fixture_review_binding_sha,
            "g1_safety_manifest": g1_safety_manifest_sha256,
            "replay_query_manifest": replay_query_manifest_sha256,
            "replay_artifact_commitments": semantic_sha256(dict(sorted(replay_artifact_commitments.items()))),
            "legacy_semantic_sources": semantic_sha256(dict(sorted(FROZEN_LEGACY_SEMANTIC_SOURCE_HASHES.items()))),
            "power_evaluator_contract": power_contract["contract_sha256"],
            "power_outcome_derivation_contract": outcome_derivation_contract_sha,
            **(
                {
                    "historical_power_reference": historical_power_reference[
                        "reference_population_sha256"
                    ]
                }
                if historical_power_reference is not None
                else {}
            ),
        },
        declared_safety_counts={
            "protected_content_opens": 0,
            "protected_id_mapping_content_opens": 0,
            "historical_write_bytes": 0,
            "model_forward_evaluations": 0,
            "real_data_optimizer_updates": 0,
        },
        commit_eligible=commit_eligible,
    )


def _validate_completed_query_role_binding_for_stage(
    value: Any,
    *,
    token_id: str,
    query_manifest_sha256: str,
    query_count: int,
) -> Mapping[str, Any]:
    binding = _exact_mapping(
        value,
        (
            "schema_version",
            "binding_kind",
            "role",
            "manifest_path",
            "manifest_file_sha256",
            "manifest_sha256",
            "query_count",
            "desc_id_lines_sha256",
            "commit_kind",
            "committed_transaction_id",
            "committed_state_sha256",
            "committed_event_sha256",
            "stage_action",
            "stage_commit_receipt_sha256",
            "binding_sha256",
        ),
        "completed eval query-role binding fields mismatch",
    )
    _require(
        binding["schema_version"]
        == "c28f_a4_completed_eval_query_role_binding_v1"
        and type(binding["query_count"]) is int
        and binding["query_count"] == query_count > 0
        and binding["manifest_sha256"] == query_manifest_sha256
        and isinstance(binding["committed_transaction_id"], str)
        and bool(binding["committed_transaction_id"])
        and binding["binding_sha256"]
        == semantic_sha256(binding, excluded_fields=("binding_sha256",)),
        "completed eval query-role binding identity/hash mismatch",
    )
    for field in (
        "manifest_file_sha256",
        "manifest_sha256",
        "desc_id_lines_sha256",
        "committed_state_sha256",
        "committed_event_sha256",
        "binding_sha256",
    ):
        _sha256(binding[field], "completed query-role binding.%s" % field)
    frozen_calib_sha = (
        "b2c2a6ec3386d1ff7a3ffd409a8b0f3859ddfb7edddab34d607981b0345a467d"
    )
    if token_id == "F0_A":
        _require(
            binding["binding_kind"]
            == "FROZEN_GENESIS_CALIB_SELECT_MANIFEST"
            and binding["role"] == "calib_select"
            and binding["manifest_path"] == str(CALIB_SELECT_MANIFEST)
            and binding["manifest_file_sha256"]
            == binding["manifest_sha256"]
            == binding["desc_id_lines_sha256"]
            == frozen_calib_sha
            and binding["query_count"] == 8_677
            and binding["commit_kind"]
            == "G0_A4_GENESIS_FROZEN_SPLIT_AUTHORITY"
            and binding["stage_action"] is None
            and binding["stage_commit_receipt_sha256"] is None,
            "F0-A query-role binding differs from frozen Genesis calib_select",
        )
    else:
        _require(token_id == "F0_B", "unknown completed query-role token")
        _require(
            binding["binding_kind"]
            == "COMMITTED_G4_ROUTE_DEV_ROLE_MANIFEST"
            and binding["role"] == "train_fit_route_dev"
            and binding["manifest_path"]
            == "f1_protocol/role_manifests/train_fit_route_dev.json"
            and binding["query_count"] >= 2_048
            and binding["commit_kind"] == "G4_ROLE_POLICY_STAGE_COMMIT"
            and binding["stage_action"] == "G4_ROLE_POLICY_LOCK"
            and isinstance(binding["stage_commit_receipt_sha256"], str),
            "F0-B query-role binding differs from committed G4 route-dev role",
        )
        _sha256(
            binding["stage_commit_receipt_sha256"],
            "F0-B role stage commit receipt SHA",
        )
    return binding


def _completed_eval_source_artifact_lineage(
    payload: Mapping[str, Any],
    *,
    context: StageContext,
) -> Mapping[str, Any]:
    train_index = _require_ready_train_jsonl_index(context)
    raw_index = _exact_mapping(
        payload["raw_forward_artifact_index"],
        F0_RAW_FORWARD_ARTIFACT_INDEX_FIELDS,
        "raw forward artifact index fields mismatch at stage boundary",
    )
    nms_index = _exact_mapping(
        payload["derived_nms_artifact_index"],
        F0_DERIVED_NMS_ARTIFACT_INDEX_FIELDS,
        "derived NMS artifact index fields mismatch at stage boundary",
    )
    rebuild = payload["derived_nms_rebuild_contract"]
    _require(
        isinstance(rebuild, Mapping)
        and rebuild.get("schema_version")
        == "c28f_a4_derived_nms_rebuild_contract_v1"
        and rebuild.get("contract_sha256")
        == payload["derived_nms_rebuild_contract_sha256"]
        == semantic_sha256(rebuild, excluded_fields=("contract_sha256",))
        and raw_index["schema_version"]
        == F0_RAW_FORWARD_ARTIFACT_INDEX_SCHEMA
        and raw_index["status"]
        == "FSYNCED_BOUNDED_TWO_PHASE_RAW_FORWARD_INDEX"
        and raw_index["goal_id"] == payload["goal_id"]
        and raw_index["attempt_id"] == payload["attempt_id"]
        and raw_index["run_id"] == payload["run_id"]
        and raw_index["completed_transaction_id"]
        == payload["completed_transaction_id"]
        and raw_index["index_sha256"]
        == payload["raw_forward_artifact_index_sha256"]
        == semantic_sha256(raw_index, excluded_fields=("index_sha256",))
        and raw_index["forward_run_receipt_sha256"]
        == payload["forward_run_receipt_sha256"]
        and raw_index["result_chunk_count"] == payload["result_chunk_count"]
        and raw_index["result_chunk_set_sha256"]
        == payload["result_chunk_set_sha256"]
        and raw_index["cursor_artifact_index_sha256"]
        == payload["artifact_index_sha256"]
        and raw_index["raw_joint_input_set_sha256"]
        == payload["raw_joint_input_set_sha256"]
        and nms_index["schema_version"]
        == F0_DERIVED_NMS_ARTIFACT_INDEX_SCHEMA
        and nms_index["status"] == "FSYNCED_FROZEN_NMS_INDEX"
        and nms_index["goal_id"] == raw_index["goal_id"]
        and nms_index["attempt_id"] == raw_index["attempt_id"]
        and nms_index["run_id"] == raw_index["run_id"]
        and nms_index["completed_transaction_id"]
        == raw_index["completed_transaction_id"]
        and nms_index["source_raw_artifact_index_sha256"]
        == raw_index["index_sha256"]
        and nms_index["rebuild_contract_sha256"]
        == rebuild["contract_sha256"]
        and nms_index["consumer_binding_sha256"]
        == raw_index["consumer_binding_sha256"]
        and nms_index["consumed_range_receipt_chain_sha256"]
        == raw_index["consumed_range_receipt_chain_sha256"]
        and nms_index["chunk_count"] == raw_index["result_chunk_count"]
        and nms_index["query_count"] == payload["retrieval_row_count"]
        and nms_index["result_set_sha256"]
        == payload["frozen_nms_result_set_sha256"]
        and nms_index["index_sha256"]
        == payload["derived_nms_artifact_index_sha256"]
        == semantic_sha256(nms_index, excluded_fields=("index_sha256",)),
        "completed eval raw-to-NMS artifact lineage mismatch",
    )
    consumed_receipt_sha256s = raw_index[
        "consumed_range_receipt_sha256s"
    ]
    raw_chunks = raw_index["result_chunks"]
    nms_chunks = nms_index["chunks"]
    _require(
        isinstance(consumed_receipt_sha256s, list)
        and len(consumed_receipt_sha256s) == raw_index["result_chunk_count"]
        and len(set(consumed_receipt_sha256s))
        == len(consumed_receipt_sha256s)
        and all(
            isinstance(value, str) and len(value) == 64
            for value in consumed_receipt_sha256s
        )
        and raw_index["consumed_range_receipt_chain_sha256"]
        == bytes_sha256(canonical_json_bytes(consumed_receipt_sha256s))
        and raw_index["all_ranges_consumed"] is True
        and raw_index["maximum_simultaneously_loaded_chunk_count"] == 1
        and raw_index["payload_retention_contract"]
        == "HANDLE_STORES_NO_RAW_PAYLOAD_STREAM_REVALIDATES_ONE_CHUNK"
        and isinstance(raw_chunks, list)
        and len(raw_chunks) == raw_index["result_chunk_count"]
        and isinstance(nms_chunks, list)
        and len(nms_chunks) == raw_index["result_chunk_count"],
        "completed eval bounded derivation receipt chain mismatch",
    )
    expected_start = 0
    for range_index, (raw_chunk_value, nms_chunk_value) in enumerate(
        zip(raw_chunks, nms_chunks)
    ):
        raw_chunk = _exact_mapping(
            raw_chunk_value,
            F0_RAW_FORWARD_ARTIFACT_CHUNK_FIELDS,
            "raw derivation chunk field set drift at %d" % range_index,
        )
        nms_chunk = _exact_mapping(
            nms_chunk_value,
            F0_DERIVED_NMS_ARTIFACT_CHUNK_FIELDS,
            "derived NMS chunk field set drift at %d" % range_index,
        )
        _require(
            raw_chunk["range_index"] == range_index
            and raw_chunk["start_index"] == expected_start
            and type(raw_chunk["end_index"]) is int
            and raw_chunk["end_index"] > expected_start
            and raw_chunk["row_count"]
            == raw_chunk["end_index"] - raw_chunk["start_index"]
            and type(raw_chunk["materialization_owner_generation"]) is int
            and 0 <= raw_chunk["materialization_owner_generation"] <= 8
            and raw_chunk["artifact_name"]
            == "queries_%05d_%05d.pt"
            % (raw_chunk["start_index"], raw_chunk["end_index"])
            and raw_chunk["pre_gt_artifact_name"]
            == "pre_gt_%06d_%s.json"
            % (range_index, raw_chunk["pre_gt_payload_sha256"])
            and raw_chunk["derived_nms_artifact_name"]
            == "derived_nms_%06d_%s.json"
            % (range_index, raw_chunk["derived_nms_payload_sha256"])
            and nms_chunk["range_index"] == range_index
            and nms_chunk["start_index"] == raw_chunk["start_index"]
            and nms_chunk["end_index"] == raw_chunk["end_index"]
            and nms_chunk["row_count"] == raw_chunk["row_count"]
            and nms_chunk["artifact_name"]
            == raw_chunk["derived_nms_artifact_name"]
            and nms_chunk["artifact_file_sha256"]
            == raw_chunk["derived_nms_artifact_file_sha256"]
            and nms_chunk["payload_sha256"]
            == raw_chunk["derived_nms_payload_sha256"]
            and nms_chunk["source_artifact_file_sha256"]
            == raw_chunk["artifact_file_sha256"]
            and nms_chunk["source_payload_semantic_sha256"]
            == raw_chunk["payload_semantic_sha256"]
            and nms_chunk["range_consumption_receipt_sha256"]
            == raw_chunk["range_consumption_receipt_sha256"]
            == consumed_receipt_sha256s[range_index]
            and nms_chunk["postimage_binding_sha256"]
            == raw_chunk["postimage_binding_sha256"],
            "raw/NMS bounded derivation chunk lineage mismatch at %d"
            % range_index,
        )
        for field, value in raw_chunk.items():
            if field.endswith("_sha256"):
                _sha256(value, "raw derivation chunk.%s" % field)
        for field, value in nms_chunk.items():
            if field.endswith("_sha256"):
                _sha256(value, "derived NMS chunk.%s" % field)
        for field in (
            "artifact_name",
            "pre_gt_artifact_name",
            "derived_nms_artifact_name",
        ):
            name = raw_chunk[field]
            _require(
                isinstance(name, str)
                and bool(name)
                and name not in {".", ".."}
                and "/" not in name
                and "\\" not in name,
                "raw derivation chunk contains an unsafe artifact name",
            )
        expected_start = raw_chunk["end_index"]
    _require(
        expected_start == payload["retrieval_row_count"],
        "completed eval bounded derivation range coverage mismatch",
    )
    transaction_id = payload["completed_transaction_id"]
    _require(
        isinstance(transaction_id, str)
        and transaction_id
        and transaction_id not in {".", ".."}
        and "/" not in transaction_id
        and "\\" not in transaction_id,
        "completed transaction ID cannot form an unsafe lineage path",
    )
    base = {
        "schema_version": F0_COMPLETED_EVAL_SOURCE_ARTIFACT_LINEAGE_SCHEMA,
        "completed_transaction_id": transaction_id,
        "train_jsonl_index_binding_sha256": train_index[
            "train_jsonl_index_binding_sha256"
        ],
        "train_jsonl_index_receipt_sha256": train_index[
            "train_jsonl_index_receipt_sha256"
        ],
        "raw_forward_artifact_index_path": (
            "transactions/%s/raw_forward_artifact_index.json" % transaction_id
        ),
        "raw_forward_artifact_index_file_sha256": payload[
            "raw_forward_artifact_index_file_sha256"
        ],
        "raw_forward_artifact_index_sha256": raw_index["index_sha256"],
        "completed_forward_validation_receipt_sha256": raw_index[
            "completed_forward_validation_receipt_sha256"
        ],
        "forward_run_receipt_sha256": raw_index[
            "forward_run_receipt_sha256"
        ],
        "consumer_capability_sha256": raw_index[
            "consumer_capability_sha256"
        ],
        "consumer_binding_sha256": raw_index["consumer_binding_sha256"],
        "result_chunk_count": raw_index["result_chunk_count"],
        "result_chunk_set_sha256": raw_index["result_chunk_set_sha256"],
        "ordered_ranges_sha256": raw_index["ordered_ranges_sha256"],
        "consumed_range_receipt_sha256s": list(
            consumed_receipt_sha256s
        ),
        "consumed_range_receipt_chain_sha256": raw_index[
            "consumed_range_receipt_chain_sha256"
        ],
        "all_ranges_consumed": True,
        "maximum_simultaneously_loaded_chunk_count": 1,
        "payload_retention_contract": raw_index[
            "payload_retention_contract"
        ],
        "raw_derivation_chunk_index_sha256": bytes_sha256(
            canonical_json_bytes(raw_chunks)
        ),
        "raw_joint_input_set_sha256": raw_index["raw_joint_input_set_sha256"],
        "derived_nms_artifact_index_path": (
            "transactions/%s/derived_nms_artifact_index.json" % transaction_id
        ),
        "derived_nms_artifact_index_file_sha256": payload[
            "derived_nms_artifact_index_file_sha256"
        ],
        "derived_nms_artifact_index_sha256": nms_index["index_sha256"],
        "derived_nms_chunk_index_sha256": bytes_sha256(
            canonical_json_bytes(nms_chunks)
        ),
        "derivation_consumer_binding_sha256": nms_index[
            "consumer_binding_sha256"
        ],
        "derivation_consumed_range_receipt_chain_sha256": nms_index[
            "consumed_range_receipt_chain_sha256"
        ],
        "derived_nms_rebuild_contract": dict(rebuild),
        "derived_nms_rebuild_contract_sha256": rebuild["contract_sha256"],
        "frozen_nms_result_set_sha256": nms_index["result_set_sha256"],
        "joint_sufficient_stats_sha256": payload[
            "joint_sufficient_stats_sha256"
        ],
        "joint_derivation_receipt_path": (
            "transactions/%s/joint_derivation_receipt.json" % transaction_id
        ),
        "joint_derivation_receipt_file_sha256": payload[
            "joint_derivation_receipt_file_sha256"
        ],
        "joint_derivation_receipt_sha256": payload[
            "joint_derivation_receipt_sha256"
        ],
        "retrieval_rows_sha256": payload["retrieval_rows_sha256"],
        "retrieval_row_count": payload["retrieval_row_count"],
        "raw_proposals_scores_or_gt_spans_embedded": False,
    }
    for field, value in base.items():
        if field.endswith("_sha256"):
            _sha256(value, "completed source lineage.%s" % field)
    lineage = {**base, "lineage_sha256": semantic_sha256(base)}
    _exact_mapping(
        lineage,
        F0_COMPLETED_EVAL_SOURCE_ARTIFACT_LINEAGE_FIELDS,
        "completed source artifact lineage field set drift",
    )
    return lineage


def _consume_completed_eval_for_stage(
    *,
    context: StageContext,
    completed_eval_capability: Any,
    token_id: str,
) -> tuple[Mapping[str, Any], tuple[Mapping[str, Any], ...]]:
    """Consume only a controller-completed artifact capability.

    This stage boundary has no TRAIN-JSONL opener and no read-to-EOF fallback.
    Under the strict-B repair policy the controller must fail closed when a
    trusted selected-offset/index authority is absent.  The approved runtime
    plan permits one separately ledgered physical source open to build only
    desc-ID/line-boundary/offset/length records while hashing the full source
    in that same pass.  That authorization is not executed or inferred here:
    this boundary still requires the resulting provenance-committed index and
    exact pre/post role-random-access receipts before it can consume a result.
    """

    _require_ready_train_jsonl_index(context)

    from .a4_control import (
        A4CompletedEvalArtifactCapability,
        validate_completed_eval_artifact_capability,
    )

    _require(
        type(completed_eval_capability) is A4CompletedEvalArtifactCapability,
        "concrete A4CompletedEvalArtifactCapability required",
    )
    expected = {
        "F0_A": {
            "stage": "G3_F0_A",
            "purpose": "F0_A_TEACHER_FREE_CALIB_SELECT_FORENSICS",
            "query_role": "calib_select",
            "minimum_query_count": 8_677,
            "maximum_query_count": 8_677,
        },
        "F0_B": {
            "stage": "G5_F0_B",
            "purpose": "F0_B_ROUTE_DEV_REPLICATION",
            "query_role": "train_fit_route_dev",
            "minimum_query_count": 2_048,
            "maximum_query_count": None,
        },
    }.get(token_id)
    _require(expected is not None, "unknown completed F0 token")
    try:
        payload = validate_completed_eval_artifact_capability(
            completed_eval_capability
        )
    except Exception as exc:
        raise StageContractError("completed eval capability validation failed") from exc
    role_binding = _validate_completed_query_role_binding_for_stage(
        payload["committed_query_role_binding"],
        token_id=token_id,
        query_manifest_sha256=payload["query_manifest_sha256"],
        query_count=payload["retrieval_row_count"],
    )
    source_artifact_lineage = _completed_eval_source_artifact_lineage(
        payload,
        context=context,
    )
    maximum = expected["maximum_query_count"]
    _require(
        payload["goal_id"] == context.goal_id
        and payload["attempt_id"] == context.attempt_id
        and payload["token_id"] == token_id
        and payload["stage"] == expected["stage"]
        and payload["purpose"] == expected["purpose"]
        and payload["query_role"] == expected["query_role"]
        and payload["completed_transaction_id"]
        == context.control_transaction_id
        and payload["completed_state_sha256"]
        == context.committed_control_state_sha256
        and payload["completed_event_sha256"]
        == context.committed_control_event_sha256
        and payload["retrieval_row_count"] >= expected["minimum_query_count"]
        and (maximum is None or payload["retrieval_row_count"] <= maximum)
        and payload["execution_binding"].get("corpus_manifest_sha256")
        == payload["corpus_manifest_sha256"]
        and payload["execution_binding"].get("split_manifest_sha256")
        == role_binding["manifest_sha256"]
        and payload["gt_not_in_forward_input"] is True
        and payload["teacher_fields_status"]
        == "UNAVAILABLE_ZERO_TEACHER_ACCESS"
        and payload["safety_counts"]
        == dict(F0_COMPLETED_EVAL_SAFETY_COUNTS),
        "completed eval capability differs from the exact stage/control tail",
    )
    _require(
        completed_eval_capability.capability_sha256
        == semantic_sha256(payload),
        "completed eval capability public binding SHA mismatch",
    )
    try:
        rows = completed_eval_capability.consume_retrieval_rows_once()
    except Exception as exc:
        raise StageContractError("completed eval row capability consumption failed") from exc
    _require(
        isinstance(rows, tuple)
        and len(rows) == payload["retrieval_row_count"]
        and bytes_sha256(canonical_json_bytes(list(rows)))
        == payload["retrieval_rows_sha256"],
        "completed eval row postimage differs from controller commitment",
    )
    stage_payload = dict(payload)
    stage_payload["_stage_source_artifact_lineage"] = source_artifact_lineage
    return stage_payload, rows


def _f0_teacher_shadow_report(
    *,
    query_manifest_sha256: str,
    completed_eval_capability_sha256: str,
) -> Mapping[str, Any]:
    base = {
        "schema_version": "c28f_f0_teacher_shadow_report_v1",
        "status": "unavailable",
        "reason": "NO_COMMITTED_TEACHER_SHADOW_CAPABILITY",
        "query_manifest_sha256": query_manifest_sha256,
        "completed_eval_capability_sha256": completed_eval_capability_sha256,
        "teacher_candidate_count": 0,
        "teacher_field_access_count": 0,
        "teacher_model_forward_count": 0,
        "teacher_union_count": 0,
        "teacher_mining_count": 0,
        "main_teacher_free_route_affected": False,
    }
    return {**base, "report_sha256": semantic_sha256(base)}


def build_g3_bundle(
    *,
    context: StageContext,
    completed_eval_capability: Any,
) -> StageBundle:
    """Build complete F0-A forensics only from one completed control eval."""

    context.validate(G3_ACTION, "METRIC_CONTRACT_OK")
    train_index = _require_ready_train_jsonl_index(context)
    from .a4_control import A4CompletedEvalArtifactCapability

    _require(
        type(completed_eval_capability) is A4CompletedEvalArtifactCapability,
        "concrete A4CompletedEvalArtifactCapability required",
    )
    _frozen_pyarrow_modules()
    payload, source_rows = _consume_completed_eval_for_stage(
        context=context,
        completed_eval_capability=completed_eval_capability,
        token_id="F0_A",
    )
    evaluation = _evaluate_completed_f0_rows(source_rows)
    capability_sha = completed_eval_capability.capability_sha256
    gate_result = _frozen_retrieval_gate_result(
        retrieval=evaluation,
        query_manifest_sha256=payload["query_manifest_sha256"],
        completed_eval_capability_sha256=capability_sha,
    )
    bucket_report, query_metric_rows = _build_f0_bucket_report(
        source_rows=source_rows,
        metric_rows=evaluation["per_query"],
        bucket_rule_contract=payload["bucket_rule_contract"],
        completed_eval_capability_sha256=capability_sha,
        query_manifest_sha256=payload["query_manifest_sha256"],
    )
    query_metrics_bytes = _canonical_jsonl(query_metric_rows)
    query_metrics_rows_sha = bytes_sha256(
        canonical_json_bytes(list(query_metric_rows))
    )
    parquet_bytes, parquet_contract = _rank_transitions_parquet(
        per_query=evaluation["per_query"],
        source_binding_sha256=payload["retrieval_rows_sha256"],
    )
    parquet_contract_bytes = _canonical_file(parquet_contract)
    bucket_report_bytes = _canonical_file(bucket_report)
    bucket_report_file_sha = bytes_sha256(bucket_report_bytes)
    bucket_markdown = _render_summary_markdown(
        title="C28F F0-A Frozen Bucket Report",
        authoritative_content_sha256=bucket_report_file_sha,
        rows=(
            ("status", bucket_report["status"]),
            ("query_count", bucket_report["query_count"]),
            ("dimension_count", bucket_report["dimension_count"]),
            ("teacher_rescue_bucket_status", "unavailable"),
        ),
    )
    teacher_report = _f0_teacher_shadow_report(
        query_manifest_sha256=payload["query_manifest_sha256"],
        completed_eval_capability_sha256=capability_sha,
    )
    teacher_report_bytes = _canonical_file(teacher_report)
    teacher_report_file_sha = bytes_sha256(teacher_report_bytes)
    teacher_markdown = _render_summary_markdown(
        title="C28F F0-A Teacher Shadow Report",
        authoritative_content_sha256=teacher_report_file_sha,
        rows=(
            ("status", teacher_report["status"]),
            ("reason", teacher_report["reason"]),
            ("teacher_model_forward_count", 0),
            ("main_teacher_free_route_affected", False),
        ),
    )
    comparison_summary = _f0_comparison_summary(
        query_role="calib_select",
        evaluation=evaluation,
        bucket_report=bucket_report,
        execution_binding=payload["execution_binding"],
    )
    comparison_summary_sha = semantic_sha256(comparison_summary)
    role_binding = payload["committed_query_role_binding"]
    source_artifact_lineage = payload["_stage_source_artifact_lineage"]
    artifact_commitments = {
        "query_metrics_rows_sha256": query_metrics_rows_sha,
        "query_metrics_file_sha256": bytes_sha256(query_metrics_bytes),
        "rank_transitions_parquet_file_sha256": bytes_sha256(parquet_bytes),
        "rank_transitions_contract_sha256": parquet_contract["contract_sha256"],
        "rank_transitions_contract_file_sha256": bytes_sha256(
            parquet_contract_bytes
        ),
        "bucket_report_sha256": bucket_report["bucket_report_sha256"],
        "bucket_report_file_sha256": bucket_report_file_sha,
        "bucket_report_markdown_file_sha256": bytes_sha256(bucket_markdown),
        "teacher_shadow_report_sha256": teacher_report["report_sha256"],
        "teacher_shadow_report_file_sha256": teacher_report_file_sha,
        "teacher_shadow_markdown_file_sha256": bytes_sha256(teacher_markdown),
        "metric_sufficient_stat_contract_sha256": payload[
            "metric_sufficient_stat_contract_sha256"
        ],
        "committed_query_role_binding_sha256": role_binding[
            "binding_sha256"
        ],
        "completed_eval_source_artifact_lineage_sha256": (
            source_artifact_lineage["lineage_sha256"]
        ),
    }
    report_base = {
        "schema_version": F0_A_FORENSICS_REPORT_SCHEMA,
        "status": "VALIDATED_COMPLETE_TEACHER_FREE_FORENSICS",
        "goal_id": context.goal_id,
        "attempt_id": context.attempt_id,
        "stage": "G3_F0_A",
        "token_id": payload["token_id"],
        "eval_id": payload["eval_id"],
        "run_id": payload["run_id"],
        "query_role": payload["query_role"],
        "query_manifest_sha256": payload["query_manifest_sha256"],
        "corpus_manifest_sha256": payload["corpus_manifest_sha256"],
        "query_count": len(source_rows),
        "query_completeness": {
            "required_query_count": role_binding["query_count"],
            "complete_query_count": len(source_rows),
            "ratio": len(source_rows) / float(role_binding["query_count"]),
            "silent_skip_count": role_binding["query_count"] - len(source_rows),
        },
        "committed_query_role_binding": dict(role_binding),
        "completed_eval_capability_sha256": capability_sha,
        "completed_eval_artifact_binding_sha256": payload[
            "artifact_capability_binding_sha256"
        ],
        "completed_eval_source_artifact_lineage": dict(
            source_artifact_lineage
        ),
        "cross_split_execution_invariants": comparison_summary[
            "cross_split_execution_invariants"
        ],
        "metric_sufficient_stat_contract_sha256": payload[
            "metric_sufficient_stat_contract_sha256"
        ],
        "retrieval_metrics": evaluation["retrieval_metrics"],
        "joint_retrieval_metrics": evaluation["joint_retrieval_metrics"],
        "vcmr_metrics": evaluation["vcmr_metrics"],
        "error_metrics": evaluation["error_metrics"],
        "gate_metrics": evaluation["gate_metrics"],
        "rank_transition_summary": evaluation["rank_transition_summary"],
        "margin_diagnostics": evaluation["margin_diagnostics"],
        "modality_mask_diagnostics": evaluation[
            "modality_mask_diagnostics"
        ],
        "video_distribution_diagnostics": evaluation[
            "video_distribution_diagnostics"
        ],
        "denominators": evaluation["denominators"],
        "frozen_gate_result": gate_result,
        "frozen_gate_result_sha256": gate_result["gate_result_sha256"],
        "earliest_failing_stage": gate_result["earliest_failing_stage"],
        "comparison_summary": comparison_summary,
        "comparison_summary_sha256": comparison_summary_sha,
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
        "artifact_commitments": artifact_commitments,
        "safety_counts": dict(payload["safety_counts"]),
        "threshold_changed_after_observation": False,
    }
    report = {**report_base, "report_sha256": semantic_sha256(report_base)}
    _exact_mapping(
        report,
        F0_A_FORENSICS_REPORT_FIELDS,
        "F0-A forensics report field set drift",
    )
    report_bytes = _canonical_file(report)
    ledger_base = {
        "schema_version": "c28f_f0_a_eval_ledger_entry_v1",
        "status": "ONE_LOGICAL_EVAL_COMPLETED_NONREPLAYABLE",
        "goal_id": context.goal_id,
        "attempt_id": context.attempt_id,
        "token_id": payload["token_id"],
        "eval_id": payload["eval_id"],
        "run_id": payload["run_id"],
        "logical_eval_count": 1,
        "completed_transaction_id": payload["completed_transaction_id"],
        "completed_state_sha256": payload["completed_state_sha256"],
        "completed_event_sha256": payload["completed_event_sha256"],
        "phase_receipt_sha256": payload["phase_receipt_sha256"],
        "cursor_receipt_sha256": payload["cursor_receipt_sha256"],
        "result_chunk_set_sha256": payload["result_chunk_set_sha256"],
        "artifact_index_sha256": payload["artifact_index_sha256"],
        "completed_eval_capability_sha256": capability_sha,
        "report_sha256": report["report_sha256"],
        "report_file_sha256": bytes_sha256(report_bytes),
        "safety_counts": dict(payload["safety_counts"]),
        "resume_rule": "SAME_EVAL_ID_CURSOR_ONLY_NO_SECOND_SELECTION_EVAL",
    }
    ledger = {**ledger_base, "ledger_entry_sha256": semantic_sha256(ledger_base)}
    return _build_stage_bundle(
        _builder_issuer=_STAGE_BUNDLE_ISSUER,
        context=context,
        action=G3_ACTION,
        result_status="F0A_COMPLETE",
        next_action=G4_PROJECTION_ACTION,
        outputs={
            G3_OUTPUTS[0]: report_bytes,
            G3_OUTPUTS[1]: query_metrics_bytes,
            G3_OUTPUTS[2]: parquet_bytes,
            G3_OUTPUTS[3]: parquet_contract_bytes,
            G3_OUTPUTS[4]: bucket_report_bytes,
            G3_OUTPUTS[5]: bucket_markdown,
            G3_OUTPUTS[6]: teacher_report_bytes,
            G3_OUTPUTS[7]: teacher_markdown,
            G3_OUTPUTS[8]: _canonical_file(ledger),
        },
        input_hashes={
            "train_jsonl_index_binding": train_index[
                "train_jsonl_index_binding_sha256"
            ],
            "train_jsonl_index_receipt": train_index[
                "train_jsonl_index_receipt_sha256"
            ],
            "completed_eval_capability": capability_sha,
            "completed_eval_artifact_binding": payload[
                "artifact_capability_binding_sha256"
            ],
            "retrieval_rows": payload["retrieval_rows_sha256"],
            "query_manifest": payload["query_manifest_sha256"],
            "query_rows": payload["query_rows_sha256"],
            "corpus_manifest": payload["corpus_manifest_sha256"],
            "bucket_rule_contract": payload["bucket_rule_contract_sha256"],
            "metric_sufficient_stat_contract": payload[
                "metric_sufficient_stat_contract_sha256"
            ],
            "cross_split_execution_invariants": semantic_sha256(
                comparison_summary["cross_split_execution_invariants"]
            ),
            "committed_query_role_binding": role_binding["binding_sha256"],
            "completed_eval_source_artifact_lineage": (
                source_artifact_lineage["lineage_sha256"]
            ),
            "frozen_gate_result": gate_result["gate_result_sha256"],
        },
        declared_safety_counts={
            "protected_content_opens": 0,
            "protected_id_mapping_content_opens": 0,
            "historical_write_bytes": 0,
            "model_forward_evaluations": 1,
            "real_data_optimizer_updates": 0,
        },
        commit_eligible=True,
    )


def _validate_f0_comparison_summary(
    value: Any,
    *,
    expected_query_role: str,
) -> Mapping[str, Any]:
    summary = _exact_mapping(
        value,
        F0_COMPARISON_SUMMARY_FIELDS,
        "F0 comparison summary fields mismatch",
    )
    _require(
        summary["schema_version"] == "c28f_f0_comparison_summary_v1"
        and summary["query_role"] == expected_query_role
        and type(summary["query_count"]) is int
        and summary["query_count"] > 0
        and all(
            isinstance(summary[field], Mapping)
            for field in F0_COMPARISON_SUMMARY_FIELDS
            if field
            not in {"schema_version", "query_role", "query_count"}
        ),
        "F0 comparison summary structure mismatch",
    )
    return summary


def _f0_scalar_shift(left: Any, right: Any, field: str) -> Mapping[str, Any]:
    valid_left = left is None or (
        isinstance(left, (int, float))
        and not isinstance(left, bool)
        and math.isfinite(float(left))
    )
    valid_right = right is None or (
        isinstance(right, (int, float))
        and not isinstance(right, bool)
        and math.isfinite(float(right))
    )
    _require(valid_left and valid_right, "F0 shift scalar invalid: %s" % field)
    return {
        "f0_a": left,
        "f0_b": right,
        "f0_b_minus_f0_a": (
            float(right) - float(left)
            if left is not None and right is not None
            else None
        ),
        "comparable": left is not None and right is not None,
    }


def _f0_flat_numeric_shift(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    *,
    label: str,
) -> Mapping[str, Any]:
    _require(set(left) == set(right), "F0 %s metric key mismatch" % label)
    result = {}
    for key in sorted(left):
        if type(left[key]) is bool or type(right[key]) is bool:
            _require(
                type(left[key]) is bool and type(right[key]) is bool,
                "F0 %s boolean metric type mismatch: %s" % (label, key),
            )
            result[key] = {
                "f0_a": left[key],
                "f0_b": right[key],
                "same": left[key] is right[key],
            }
        else:
            result[key] = _f0_scalar_shift(
                left[key], right[key], "%s.%s" % (label, key)
            )
    return result


def _f0_margin_shift(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
) -> Mapping[str, Any]:
    _require(
        set(left) == set(right) == {"pooled", "late", "final"},
        "F0 margin stage key mismatch",
    )
    result = {}
    for stage in ("pooled", "late", "final"):
        left_stage = left[stage]
        right_stage = right[stage]
        _require(
            isinstance(left_stage, Mapping)
            and isinstance(right_stage, Mapping)
            and set(left_stage) == set(right_stage),
            "F0 margin diagnostic field mismatch: %s" % stage,
        )
        result[stage] = {
            "status": {
                "f0_a": left_stage["status"],
                "f0_b": right_stage["status"],
                "same": left_stage["status"] == right_stage["status"],
            },
            "cut_margin_definition": {
                "f0_a": left_stage["cut_margin_definition"],
                "f0_b": right_stage["cut_margin_definition"],
                "same": left_stage["cut_margin_definition"]
                == right_stage["cut_margin_definition"],
            },
            "top1_top2_available_query_count": _f0_scalar_shift(
                left_stage["top1_top2_available_query_count"],
                right_stage["top1_top2_available_query_count"],
                "%s.top1_available" % stage,
            ),
            "top10_top11_available_query_count": _f0_scalar_shift(
                left_stage["top10_top11_available_query_count"],
                right_stage["top10_top11_available_query_count"],
                "%s.top10_available" % stage,
            ),
            "top100_top101_available_query_count": _f0_scalar_shift(
                left_stage["top100_top101_available_query_count"],
                right_stage["top100_top101_available_query_count"],
                "%s.top100_available" % stage,
            ),
            "gt_vs_best_wrong_available_query_count": _f0_scalar_shift(
                left_stage["gt_vs_best_wrong_available_query_count"],
                right_stage["gt_vs_best_wrong_available_query_count"],
                "%s.gt_available" % stage,
            ),
            "mean_top1_top2_score_margin": _f0_scalar_shift(
                left_stage["mean_top1_top2_score_margin"],
                right_stage["mean_top1_top2_score_margin"],
                "%s.mean_top_margin" % stage,
            ),
            "mean_top10_top11_score_margin": _f0_scalar_shift(
                left_stage["mean_top10_top11_score_margin"],
                right_stage["mean_top10_top11_score_margin"],
                "%s.mean_top10_margin" % stage,
            ),
            "mean_top100_top101_score_margin": _f0_scalar_shift(
                left_stage["mean_top100_top101_score_margin"],
                right_stage["mean_top100_top101_score_margin"],
                "%s.mean_top100_margin" % stage,
            ),
            "mean_gt_vs_best_wrong_score_margin": _f0_scalar_shift(
                left_stage["mean_gt_vs_best_wrong_score_margin"],
                right_stage["mean_gt_vs_best_wrong_score_margin"],
                "%s.mean_gt_margin" % stage,
            ),
        }
    return result


def _f0_bucket_shift(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
) -> Mapping[str, Any]:
    _require(set(left) == set(right), "F0 bucket dimension key mismatch")
    dimensions = {}
    for dimension in sorted(left):
        left_labels = left[dimension]
        right_labels = right[dimension]
        _require(
            isinstance(left_labels, Mapping) and isinstance(right_labels, Mapping),
            "F0 bucket dimension is not a mapping: %s" % dimension,
        )
        labels = sorted(set(left_labels) | set(right_labels))
        per_label = {}
        total_variation_parts = []
        for label in labels:
            left_bucket = left_labels.get(label)
            right_bucket = right_labels.get(label)
            for bucket, split_name in (
                (left_bucket, "F0-A"),
                (right_bucket, "F0-B"),
            ):
                if bucket is not None:
                    _require(
                        isinstance(bucket, Mapping)
                        and set(bucket)
                        == {
                            "query_count",
                            "query_fraction",
                            "gate_metrics",
                            "true_video_retrieval",
                            "vcmr",
                            "errors",
                            "rank_transition",
                            "mean_margins",
                        },
                        "F0 bucket metric fields mismatch: %s.%s.%s"
                        % (split_name, dimension, label),
                    )
            left_fraction = (
                float(left_bucket["query_fraction"])
                if isinstance(left_bucket, Mapping)
                else 0.0
            )
            right_fraction = (
                float(right_bucket["query_fraction"])
                if isinstance(right_bucket, Mapping)
                else 0.0
            )
            total_variation_parts.append(abs(right_fraction - left_fraction))
            per_label[label] = {
                "f0_a_query_count": (
                    int(left_bucket["query_count"])
                    if isinstance(left_bucket, Mapping)
                    else 0
                ),
                "f0_b_query_count": (
                    int(right_bucket["query_count"])
                    if isinstance(right_bucket, Mapping)
                    else 0
                ),
                "query_fraction_shift": _f0_scalar_shift(
                    left_fraction,
                    right_fraction,
                    "%s.%s.query_fraction" % (dimension, label),
                ),
                "gate_metric_shift": (
                    _f0_flat_numeric_shift(
                        left_bucket["gate_metrics"],
                        right_bucket["gate_metrics"],
                        label="%s.%s.gates" % (dimension, label),
                    )
                    if isinstance(left_bucket, Mapping)
                    and isinstance(right_bucket, Mapping)
                    else None
                ),
                "true_video_retrieval_shift": (
                    _f0_flat_numeric_shift(
                        left_bucket["true_video_retrieval"],
                        right_bucket["true_video_retrieval"],
                        label="%s.%s.true_video_retrieval"
                        % (dimension, label),
                    )
                    if isinstance(left_bucket, Mapping)
                    and isinstance(right_bucket, Mapping)
                    else None
                ),
                "vcmr_metric_shift": (
                    _f0_flat_numeric_shift(
                        left_bucket["vcmr"],
                        right_bucket["vcmr"],
                        label="%s.%s.vcmr" % (dimension, label),
                    )
                    if isinstance(left_bucket, Mapping)
                    and isinstance(right_bucket, Mapping)
                    else None
                ),
                "error_metric_shift": (
                    _f0_flat_numeric_shift(
                        left_bucket["errors"],
                        right_bucket["errors"],
                        label="%s.%s.errors" % (dimension, label),
                    )
                    if isinstance(left_bucket, Mapping)
                    and isinstance(right_bucket, Mapping)
                    else None
                ),
                "rank_transition_shift": (
                    _f0_flat_numeric_shift(
                        left_bucket["rank_transition"],
                        right_bucket["rank_transition"],
                        label="%s.%s.rank_transition" % (dimension, label),
                    )
                    if isinstance(left_bucket, Mapping)
                    and isinstance(right_bucket, Mapping)
                    else None
                ),
                "margin_shift": (
                    _f0_flat_numeric_shift(
                        left_bucket["mean_margins"],
                        right_bucket["mean_margins"],
                        label="%s.%s.mean_margins" % (dimension, label),
                    )
                    if isinstance(left_bucket, Mapping)
                    and isinstance(right_bucket, Mapping)
                    else None
                ),
            }
        dimensions[dimension] = {
            "distribution_total_variation_distance": 0.5
            * math.fsum(total_variation_parts),
            "label_shifts": per_label,
        }
    return dimensions


def _f0_modality_mask_shift(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
) -> Mapping[str, Any]:
    _require(
        set(left)
        == set(right)
        == {
            "ratio_definition",
            "structure_contract",
            "joint_component_relation_verified",
            "visual_subtitle_overlap_valid_count",
            "visual",
            "subtitle",
            "joint",
        }
        and left["ratio_definition"]
        == right["ratio_definition"]
        == "VALID_OR_MISSING_BOOL_CELLS_DIVIDED_BY_CANDIDATE_COUNT_TIMES_64"
        and left["structure_contract"] == right["structure_contract"]
        and left["joint_component_relation_verified"] is True
        and right["joint_component_relation_verified"] is True,
        "F0 modality mask diagnostic key mismatch",
    )
    result = {
        "ratio_definition": left["ratio_definition"],
        "structure_contract": dict(left["structure_contract"]),
        "joint_component_relation_verified": True,
        "visual_subtitle_overlap_valid_count": _f0_scalar_shift(
            left["visual_subtitle_overlap_valid_count"],
            right["visual_subtitle_overlap_valid_count"],
            "visual_subtitle_overlap_valid_count",
        ),
    }
    for modality in ("visual", "subtitle", "joint"):
        left_record = left[modality]
        right_record = right[modality]
        _require(
            isinstance(left_record, Mapping)
            and isinstance(right_record, Mapping)
            and set(left_record) == set(right_record)
            == {
                "query_count",
                "candidate_count",
                "empty_candidate_count",
                "cell_denominator",
                "valid_count",
                "missing_count",
                "mean_valid_ratio",
                "mean_missing_ratio",
                "mean_empty_candidate_ratio",
            },
            "F0 modality mask aggregate contract mismatch: %s" % modality,
        )
        result[modality] = {
            field: _f0_scalar_shift(
                left_record[field],
                right_record[field],
                "%s.%s" % (modality, field),
            )
            for field in (
                "query_count",
                "candidate_count",
                "empty_candidate_count",
                "cell_denominator",
                "valid_count",
                "missing_count",
                "mean_valid_ratio",
                "mean_missing_ratio",
                "mean_empty_candidate_ratio",
            )
        }
    return result


def _f0_video_distribution_shift(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
) -> Mapping[str, Any]:
    _require(
        set(left)
        == set(right)
        == {
            "gt_video_cluster_count",
            "gt_video_id_lines_sha256",
            "queries_per_gt_video",
        }
        and isinstance(left["queries_per_gt_video"], Mapping)
        and isinstance(right["queries_per_gt_video"], Mapping),
        "F0 video distribution diagnostic fields mismatch",
    )
    return {
        "gt_video_cluster_count": _f0_scalar_shift(
            left["gt_video_cluster_count"],
            right["gt_video_cluster_count"],
            "video_distribution.gt_video_cluster_count",
        ),
        "gt_video_universe_commitments": {
            "f0_a": _sha256(
                left["gt_video_id_lines_sha256"],
                "F0-A GT-video lines SHA",
            ),
            "f0_b": _sha256(
                right["gt_video_id_lines_sha256"],
                "F0-B GT-video lines SHA",
            ),
            "same": left["gt_video_id_lines_sha256"]
            == right["gt_video_id_lines_sha256"],
        },
        "queries_per_gt_video_shift": _f0_flat_numeric_shift(
            left["queries_per_gt_video"],
            right["queries_per_gt_video"],
            label="queries_per_gt_video",
        ),
    }


def build_g5_bundle(
    *,
    context: StageContext,
    completed_eval_capability: Any,
) -> StageBundle:
    """Build F0-B and compare only against controller-re-read committed F0-A."""

    context.validate(G5_ACTION, "ROLE_POLICY_LOCKED")
    payload, source_rows = _consume_completed_eval_for_stage(
        context=context,
        completed_eval_capability=completed_eval_capability,
        token_id="F0_B",
    )
    predecessor = _exact_mapping(
        payload["committed_f0_a_report_binding"],
        F0_A_COMMITTED_REPORT_BINDING_FIELDS,
        "F0-B predecessor binding fields mismatch",
    )
    f0_a_summary = _validate_f0_comparison_summary(
        predecessor["comparison_summary"],
        expected_query_role="calib_select",
    )
    _require(
        predecessor["report_path"]
        == "f0_forensics/f0_a_forensics_report.json"
        and predecessor["report_schema_version"]
        == F0_A_FORENSICS_REPORT_SCHEMA
        and predecessor["comparison_summary_sha256"]
        == semantic_sha256(f0_a_summary)
        and predecessor["earliest_failing_stage"]
        in {
            "NO_FAILURE",
            "POOLED_BROAD",
            "LATE_RERANK",
            "FINAL_VIDEO_ORDERING",
        }
        and all(
            isinstance(predecessor[field], str)
            and len(predecessor[field]) == 64
            for field in (
                "report_file_sha256",
                "report_sha256",
                "frozen_gate_result_sha256",
                "comparison_summary_sha256",
                "g3_commit_receipt_sha256",
                "g3_state_sha256",
                "g3_event_sha256",
            )
        ),
        "F0-B predecessor is not the controller-re-read committed G3 report",
    )
    evaluation = _evaluate_completed_f0_rows(source_rows)
    capability_sha = completed_eval_capability.capability_sha256
    gate_result = _frozen_retrieval_gate_result(
        retrieval=evaluation,
        query_manifest_sha256=payload["query_manifest_sha256"],
        completed_eval_capability_sha256=capability_sha,
    )
    bucket_report, query_metric_rows = _build_f0_bucket_report(
        source_rows=source_rows,
        metric_rows=evaluation["per_query"],
        bucket_rule_contract=payload["bucket_rule_contract"],
        completed_eval_capability_sha256=capability_sha,
        query_manifest_sha256=payload["query_manifest_sha256"],
    )
    f0_b_summary = _f0_comparison_summary(
        query_role="train_fit_route_dev",
        evaluation=evaluation,
        bucket_report=bucket_report,
        execution_binding=payload["execution_binding"],
    )
    f0_b_summary_sha = semantic_sha256(f0_b_summary)
    role_binding = payload["committed_query_role_binding"]
    source_artifact_lineage = payload["_stage_source_artifact_lineage"]
    _require(
        f0_a_summary["cross_split_execution_invariants"]
        == f0_b_summary["cross_split_execution_invariants"],
        "F0-B checkpoint/corpus/evaluator/protocol differs from committed F0-A",
    )
    f0_a_rank = f0_a_summary["rank_transition_summary"]
    f0_b_rank = f0_b_summary["rank_transition_summary"]
    _require(
        isinstance(f0_a_rank, Mapping)
        and isinstance(f0_b_rank, Mapping)
        and set(f0_a_rank) == set(f0_b_rank)
        and f0_a_rank["definition"] == f0_b_rank["definition"],
        "F0 rank-transition comparison contract mismatch",
    )
    rank_shift = {
        "definition": f0_a_rank["definition"],
        **{
            key: _f0_scalar_shift(
                f0_a_rank[key],
                f0_b_rank[key],
                "rank_transition.%s" % key,
            )
            for key in sorted(set(f0_a_rank) - {"definition"})
        },
    }
    same_earliest = (
        predecessor["earliest_failing_stage"]
        == gate_result["earliest_failing_stage"]
    )
    routing_status = (
        "ROUTING_CONSISTENT" if same_earliest else "ROUTING_AMBIGUOUS"
    )
    split_base = {
        "schema_version": "c28f_f0_split_shift_report_v1",
        "status": "VALIDATED_EXPOSURE_AWARE_SPLIT_SHIFT",
        "f0_a_committed_report_binding": dict(predecessor),
        "f0_b_completed_eval_capability_sha256": capability_sha,
        "f0_b_comparison_summary": f0_b_summary,
        "f0_b_comparison_summary_sha256": f0_b_summary_sha,
        "query_count_shift": _f0_scalar_shift(
            f0_a_summary["query_count"],
            f0_b_summary["query_count"],
            "query_count",
        ),
        "cross_split_execution_invariants": f0_b_summary[
            "cross_split_execution_invariants"
        ],
        "cross_split_execution_invariants_match": True,
        "retrieval_metric_shift": _f0_flat_numeric_shift(
            f0_a_summary["retrieval_metrics"],
            f0_b_summary["retrieval_metrics"],
            label="retrieval",
        ),
        "joint_retrieval_metric_shift": _f0_flat_numeric_shift(
            f0_a_summary["joint_retrieval_metrics"],
            f0_b_summary["joint_retrieval_metrics"],
            label="joint_retrieval",
        ),
        "vcmr_metric_shift": _f0_flat_numeric_shift(
            f0_a_summary["vcmr_metrics"],
            f0_b_summary["vcmr_metrics"],
            label="vcmr",
        ),
        "error_metric_shift": _f0_flat_numeric_shift(
            f0_a_summary["error_metrics"],
            f0_b_summary["error_metrics"],
            label="errors",
        ),
        "gate_metric_shift": _f0_flat_numeric_shift(
            f0_a_summary["gate_metrics"],
            f0_b_summary["gate_metrics"],
            label="gates",
        ),
        "rank_transition_shift": rank_shift,
        "margin_shift": _f0_margin_shift(
            f0_a_summary["margin_diagnostics"],
            f0_b_summary["margin_diagnostics"],
        ),
        "modality_mask_shift": _f0_modality_mask_shift(
            f0_a_summary["modality_mask_diagnostics"],
            f0_b_summary["modality_mask_diagnostics"],
        ),
        "video_distribution_shift": _f0_video_distribution_shift(
            f0_a_summary["video_distribution_diagnostics"],
            f0_b_summary["video_distribution_diagnostics"],
        ),
        "bucket_shift": _f0_bucket_shift(
            f0_a_summary["bucket_distribution_summary"],
            f0_b_summary["bucket_distribution_summary"],
        ),
        "earliest_failing_stage": {
            "f0_a": predecessor["earliest_failing_stage"],
            "f0_b": gate_result["earliest_failing_stage"],
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
    _exact_mapping(
        split_report,
        F0_SPLIT_SHIFT_REPORT_FIELDS,
        "F0 split-shift report field set drift",
    )
    split_bytes = _canonical_file(split_report)
    split_file_sha = bytes_sha256(split_bytes)
    routing_base = {
        "schema_version": "c28f_f0_routing_decision_v1",
        "status": routing_status,
        "f0_a_frozen_gate_result_sha256": predecessor[
            "frozen_gate_result_sha256"
        ],
        "f0_b_frozen_gate_result_sha256": gate_result[
            "gate_result_sha256"
        ],
        "f0_a_earliest_failing_stage": predecessor[
            "earliest_failing_stage"
        ],
        "f0_b_earliest_failing_stage": gate_result[
            "earliest_failing_stage"
        ],
        "selected_earliest_failing_stage": (
            gate_result["earliest_failing_stage"] if same_earliest else None
        ),
        "next_stage_recommendation_only": (
            gate_result["next_stage_recommendation_only"]
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
    routing_decision = {
        **routing_base,
        "routing_decision_sha256": semantic_sha256(routing_base),
    }
    _exact_mapping(
        routing_decision,
        F0_ROUTING_DECISION_FIELDS,
        "F0 routing decision field set drift",
    )
    routing_bytes = _canonical_file(routing_decision)
    routing_file_sha = bytes_sha256(routing_bytes)
    split_markdown = _render_summary_markdown(
        title="C28F F0-A/F0-B Split Shift",
        authoritative_content_sha256=split_file_sha,
        rows=(
            ("status", split_report["status"]),
            ("routing_status", routing_status),
            ("f0_a_earliest", predecessor["earliest_failing_stage"]),
            ("f0_b_earliest", gate_result["earliest_failing_stage"]),
            ("threshold_changed_after_observation", False),
        ),
    )
    routing_markdown = _render_summary_markdown(
        title="C28F F0 Routing Decision",
        authoritative_content_sha256=routing_file_sha,
        rows=(
            ("status", routing_status),
            (
                "selected_earliest_failing_stage",
                routing_decision["selected_earliest_failing_stage"],
            ),
            (
                "next_stage_recommendation_only",
                routing_decision["next_stage_recommendation_only"],
            ),
            ("f2_or_higher_executed", False),
        ),
    )
    query_metrics_bytes = _canonical_jsonl(query_metric_rows)
    artifact_commitments = {
        "query_metrics_rows_sha256": bytes_sha256(
            canonical_json_bytes(list(query_metric_rows))
        ),
        "query_metrics_file_sha256": bytes_sha256(query_metrics_bytes),
        "split_shift_report_sha256": split_report[
            "split_shift_report_sha256"
        ],
        "split_shift_report_file_sha256": split_file_sha,
        "split_shift_markdown_file_sha256": bytes_sha256(split_markdown),
        "routing_decision_sha256": routing_decision[
            "routing_decision_sha256"
        ],
        "routing_decision_file_sha256": routing_file_sha,
        "routing_decision_markdown_file_sha256": bytes_sha256(
            routing_markdown
        ),
        "metric_sufficient_stat_contract_sha256": payload[
            "metric_sufficient_stat_contract_sha256"
        ],
        "committed_query_role_binding_sha256": role_binding[
            "binding_sha256"
        ],
        "completed_eval_source_artifact_lineage_sha256": (
            source_artifact_lineage["lineage_sha256"]
        ),
    }
    report_base = {
        "schema_version": F0_B_FORENSICS_REPORT_SCHEMA,
        "status": "VALIDATED_COMPLETE_EXPOSURE_AWARE_FORENSICS",
        "goal_id": context.goal_id,
        "attempt_id": context.attempt_id,
        "stage": "G5_F0_B",
        "token_id": payload["token_id"],
        "eval_id": payload["eval_id"],
        "run_id": payload["run_id"],
        "query_role": payload["query_role"],
        "query_manifest_sha256": payload["query_manifest_sha256"],
        "corpus_manifest_sha256": payload["corpus_manifest_sha256"],
        "query_count": len(source_rows),
        "query_completeness": {
            "required_committed_role_query_count": role_binding[
                "query_count"
            ],
            "complete_query_count": len(source_rows),
            "ratio": len(source_rows) / float(role_binding["query_count"]),
            "silent_skip_count": role_binding["query_count"] - len(source_rows),
        },
        "committed_query_role_binding": dict(role_binding),
        "completed_eval_capability_sha256": capability_sha,
        "completed_eval_artifact_binding_sha256": payload[
            "artifact_capability_binding_sha256"
        ],
        "completed_eval_source_artifact_lineage": dict(
            source_artifact_lineage
        ),
        "cross_split_execution_invariants": f0_b_summary[
            "cross_split_execution_invariants"
        ],
        "metric_sufficient_stat_contract_sha256": payload[
            "metric_sufficient_stat_contract_sha256"
        ],
        "retrieval_metrics": evaluation["retrieval_metrics"],
        "joint_retrieval_metrics": evaluation["joint_retrieval_metrics"],
        "vcmr_metrics": evaluation["vcmr_metrics"],
        "error_metrics": evaluation["error_metrics"],
        "gate_metrics": evaluation["gate_metrics"],
        "rank_transition_summary": evaluation["rank_transition_summary"],
        "margin_diagnostics": evaluation["margin_diagnostics"],
        "modality_mask_diagnostics": evaluation[
            "modality_mask_diagnostics"
        ],
        "video_distribution_diagnostics": evaluation[
            "video_distribution_diagnostics"
        ],
        "denominators": evaluation["denominators"],
        "frozen_gate_result": gate_result,
        "frozen_gate_result_sha256": gate_result["gate_result_sha256"],
        "earliest_failing_stage": gate_result["earliest_failing_stage"],
        "comparison_summary": f0_b_summary,
        "comparison_summary_sha256": f0_b_summary_sha,
        "bucket_report": bucket_report,
        "predecessor_f0_a_binding": dict(predecessor),
        "split_shift_report_sha256": split_report[
            "split_shift_report_sha256"
        ],
        "routing_status": routing_status,
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
        "artifact_commitments": artifact_commitments,
        "safety_counts": dict(payload["safety_counts"]),
        "threshold_changed_after_observation": False,
    }
    report = {**report_base, "report_sha256": semantic_sha256(report_base)}
    _exact_mapping(
        report,
        F0_B_FORENSICS_REPORT_FIELDS,
        "F0-B forensics report field set drift",
    )
    report_bytes = _canonical_file(report)
    ledger_base = {
        "schema_version": "c28f_f0_b_eval_ledger_entry_v1",
        "status": "ONE_LOGICAL_EVAL_COMPLETED_NONREPLAYABLE",
        "goal_id": context.goal_id,
        "attempt_id": context.attempt_id,
        "token_id": payload["token_id"],
        "eval_id": payload["eval_id"],
        "run_id": payload["run_id"],
        "logical_eval_count": 1,
        "completed_transaction_id": payload["completed_transaction_id"],
        "completed_state_sha256": payload["completed_state_sha256"],
        "completed_event_sha256": payload["completed_event_sha256"],
        "phase_receipt_sha256": payload["phase_receipt_sha256"],
        "cursor_receipt_sha256": payload["cursor_receipt_sha256"],
        "result_chunk_set_sha256": payload["result_chunk_set_sha256"],
        "artifact_index_sha256": payload["artifact_index_sha256"],
        "completed_eval_capability_sha256": capability_sha,
        "committed_f0_a_report_sha256": predecessor["report_sha256"],
        "report_sha256": report["report_sha256"],
        "report_file_sha256": bytes_sha256(report_bytes),
        "routing_status": routing_status,
        "safety_counts": dict(payload["safety_counts"]),
        "resume_rule": "SAME_EVAL_ID_CURSOR_ONLY_NO_SECOND_SELECTION_EVAL",
    }
    ledger = {**ledger_base, "ledger_entry_sha256": semantic_sha256(ledger_base)}
    return _build_stage_bundle(
        _builder_issuer=_STAGE_BUNDLE_ISSUER,
        context=context,
        action=G5_ACTION,
        result_status="F0B_COMPLETE",
        next_action=G6_ACTION,
        outputs={
            G5_OUTPUTS[0]: report_bytes,
            G5_OUTPUTS[1]: query_metrics_bytes,
            G5_OUTPUTS[2]: split_bytes,
            G5_OUTPUTS[3]: split_markdown,
            G5_OUTPUTS[4]: routing_bytes,
            G5_OUTPUTS[5]: routing_markdown,
            G5_OUTPUTS[6]: _canonical_file(ledger),
        },
        input_hashes={
            "completed_eval_capability": capability_sha,
            "completed_eval_artifact_binding": payload[
                "artifact_capability_binding_sha256"
            ],
            "retrieval_rows": payload["retrieval_rows_sha256"],
            "query_manifest": payload["query_manifest_sha256"],
            "query_rows": payload["query_rows_sha256"],
            "corpus_manifest": payload["corpus_manifest_sha256"],
            "bucket_rule_contract": payload["bucket_rule_contract_sha256"],
            "metric_sufficient_stat_contract": payload[
                "metric_sufficient_stat_contract_sha256"
            ],
            "cross_split_execution_invariants": semantic_sha256(
                f0_b_summary["cross_split_execution_invariants"]
            ),
            "committed_query_role_binding": role_binding["binding_sha256"],
            "completed_eval_source_artifact_lineage": (
                source_artifact_lineage["lineage_sha256"]
            ),
            "committed_f0_a_report": predecessor["report_sha256"],
            "committed_f0_a_comparison_summary": predecessor[
                "comparison_summary_sha256"
            ],
            "frozen_gate_result": gate_result["gate_result_sha256"],
        },
        declared_safety_counts={
            "protected_content_opens": 0,
            "protected_id_mapping_content_opens": 0,
            "historical_write_bytes": 0,
            "model_forward_evaluations": 1,
            "real_data_optimizer_updates": 0,
        },
        commit_eligible=True,
    )


def _validate_g4_projection_open_classification(
    manifest: Mapping[str, Any],
) -> Mapping[str, Any]:
    classification = dict(
        _exact_mapping(
            manifest.get("content_open_classification"),
            (
                "categories_mutually_exclusive",
                "unauthorized_or_non_id_only_protected_content_open_count",
                "authorized_protected_id_mapping_content_open_count",
                "total_protected_content_open_count",
                "authorized_id_mapping_breakdown",
            ),
            "G4 projection content-open classification fields mismatch",
        )
    )
    breakdown = dict(
        _exact_mapping(
            classification["authorized_id_mapping_breakdown"],
            (
                "protected_desc_id_manifest_content_open_count",
                "train_fit_desc_id_manifest_content_open_count",
                "protected_desc_train_jsonl_indexed_pread_content_open_count",
                "train_fit_train_jsonl_indexed_pread_content_open_count",
                "official_val_content_open_count",
            ),
            "G4 projection content-open breakdown fields mismatch",
        )
    )
    _require(
        classification["categories_mutually_exclusive"] is True
        and classification[
            "unauthorized_or_non_id_only_protected_content_open_count"
        ]
        == 0
        and classification[
            "authorized_protected_id_mapping_content_open_count"
        ]
        == 5
        and classification["total_protected_content_open_count"] == 5
        and breakdown
        == {
            "protected_desc_id_manifest_content_open_count": 2,
            "train_fit_desc_id_manifest_content_open_count": 1,
            "protected_desc_train_jsonl_indexed_pread_content_open_count": 1,
            "train_fit_train_jsonl_indexed_pread_content_open_count": 1,
            "official_val_content_open_count": 0,
        },
        "G4 projection content opens are not the frozen mutually exclusive 0/5 split",
    )
    return classification


def build_g4_projection_bundle(
    *,
    context: StageContext,
    projection_result: Any,
) -> StageBundle:
    """Commit the protected projection and frozen TRAIN_FIT role source.

    Completion of the independent operation is deliberately not claimed by
    this bundle.  The controller must first commit this exact postimage, then
    re-read it and its access sidecar before advancing the operation to
    ``COMPLETED_NONREPLAYABLE`` and enabling the final G4 role-policy action.
    """

    context.validate(G4_PROJECTION_ACTION, "ID_MAPPING_ACCESS_COMMITTED")
    from .a4_control import validate_g4_minimal_projection_batch
    from .a4_id_projector import ProjectionResult, build_projection_manifest

    _require(
        type(projection_result) is ProjectionResult,
        "concrete controller-sealed ProjectionResult required",
    )
    try:
        manifest = build_projection_manifest(projection_result)
        minimal_batch = validate_g4_minimal_projection_batch(
            projection_result.minimal_batch
        )
    except Exception as exc:
        raise StageContractError(
            "G4 projection result/control binding validation failed"
        ) from exc
    _require(
        manifest.get("status")
        == "PROJECTED_CANDIDATE_ACCESS_CAS_COMMITTED_STAGE_CAS_PENDING"
        and manifest.get("completion_status")
        == "DEFERRED_REQUIRES_A4_STAGE_CAS_AND_CONTROL_COMPLETION_SIDECAR",
        "G4 projection manifest is not an access-CAS-bound candidate",
    )
    _validate_g4_projection_open_classification(manifest)
    mapping_rows = manifest.get("mapping_rows")
    video_ids = manifest.get("protected_video_ids")
    _require(
        isinstance(mapping_rows, list)
        and bool(mapping_rows)
        and isinstance(video_ids, list)
        and bool(video_ids),
        "G4 projection manifest rows/video IDs missing",
    )
    mapping_file_rows = [
        {"schema_version": ID_MAP_SCHEMA_VERSION, **dict(row)}
        for row in mapping_rows
    ]
    mapping_bytes = _canonical_jsonl(mapping_file_rows)
    video_bytes = "".join("%s\n" % value for value in video_ids).encode(
        "utf-8"
    )
    _require(
        bytes_sha256(canonical_json_bytes(mapping_rows))
        == manifest.get("mapping_rows_sha256")
        and bytes_sha256(video_bytes)
        == manifest.get("protected_video_id_lines_sha256"),
        "G4 projection business postimages differ from manifest",
    )
    lineage = manifest.get("lineage")
    _require(isinstance(lineage, Mapping), "G4 projection lineage missing")
    _require(
        lineage.get("goal_id") == context.goal_id
        and lineage.get("attempt_id") == context.attempt_id
        and lineage.get("access_transaction_id")
        == context.control_transaction_id
        and lineage.get("access_state_sha256")
        == context.committed_control_state_sha256
        and lineage.get("access_event_sha256")
        == context.committed_control_event_sha256
        and lineage.get("authority_permission_id")
        == "AUTH_PROTECTED_ID_MAPPING_ONLY"
        and lineage.get("eval_token_consumed") is False,
        "G4 projection is not bound to the exact access-committed control tail",
    )
    train_fit_source_evidence = minimal_batch.get("train_fit_source_evidence")
    train_fit_projection_rows = minimal_batch.get("train_fit_mapping_records")
    _require(
        isinstance(train_fit_source_evidence, dict)
        and isinstance(train_fit_projection_rows, list)
        and len(train_fit_projection_rows) == 60_751
        and minimal_batch.get("batch_binding_sha256")
        == lineage.get("minimal_projection_batch_binding_sha256")
        and minimal_batch.get("access_receipt")
        == dict(projection_result.access_receipt),
        "G4 TRAIN_FIT source is not sealed by the projected minimal batch",
    )
    train_fit_records = [
        {
            "desc_id": row["desc_id"],
            "ground_truth_video_id": row["video_id"],
        }
        for row in train_fit_projection_rows
    ]
    _require(
        train_fit_records
        == sorted(train_fit_records, key=lambda item: item["desc_id"])
        and len({item["desc_id"] for item in train_fit_records})
        == len(train_fit_records),
        "G4 TRAIN_FIT source records are not canonical unique",
    )
    train_fit_mapping_bytes = _canonical_jsonl(
        [
            {
                "schema_version": "c28f_train_fit_desc_gt_video_v1",
                **row,
            }
            for row in train_fit_records
        ]
    )
    train_fit_mapping_sha256 = bytes_sha256(
        canonical_json_bytes(train_fit_records)
    )
    access_receipt = minimal_batch["access_receipt"]
    _require(
        access_receipt.get("access_transaction_id")
        == context.control_transaction_id
        and access_receipt.get("access_state_sha256")
        == context.committed_control_state_sha256
        and access_receipt.get("access_event_sha256")
        == context.committed_control_event_sha256,
        "G4 TRAIN_FIT source access receipt/control tail mismatch",
    )
    train_fit_source_receipt_base = {
        "schema_version": "c28f_train_fit_role_source_receipt_v2",
        "status": "ACCESS_CAS_COMMITTED_STAGE_CAS_PENDING",
        "goal_id": context.goal_id,
        "attempt_id": context.attempt_id,
        "source_descriptor_identity_sha256": train_fit_source_evidence[
            "source_descriptor_identity_sha256"
        ],
        "source_file_sha256": train_fit_source_evidence[
            "source_file_sha256"
        ],
        "source_manifest_sha256": train_fit_source_evidence[
            "manifest_file_sha256"
        ],
        "source_manifest_access_receipt_sha256": train_fit_source_evidence[
            "manifest_access_receipt_sha256"
        ],
        "train_projection_access_receipt_sha256": train_fit_source_evidence[
            "train_projection_access_receipt_sha256"
        ],
        "source_evidence_sha256": semantic_sha256(train_fit_source_evidence),
        "mapping_file_sha256": bytes_sha256(train_fit_mapping_bytes),
        "mapping_sha256": train_fit_mapping_sha256,
        "record_count": len(train_fit_records),
        "allowed_fields": ["desc_id", "ground_truth_video_id"],
        "physical_manifest_content_open_count": train_fit_source_evidence[
            "physical_manifest_content_open_count"
        ],
        "physical_train_jsonl_content_open_count": train_fit_source_evidence[
            "physical_train_jsonl_content_open_count"
        ],
        "decoded_field_counts": dict(
            train_fit_source_evidence["decoded_field_counts"]
        ),
        "full_line_json_decode_count": train_fit_source_evidence[
            "full_line_json_decode_count"
        ],
        "unauthorized_field_semantic_decode_count": train_fit_source_evidence[
            "unauthorized_field_semantic_decode_count"
        ],
        "unmediated_content_open_count": train_fit_source_evidence[
            "unmediated_content_open_count"
        ],
        "instrumentation_complete": train_fit_source_evidence[
            "instrumentation_complete"
        ],
        "access_transaction_id": access_receipt["access_transaction_id"],
        "access_state_sha256": access_receipt["access_state_sha256"],
        "access_event_sha256": access_receipt["access_event_sha256"],
    }
    train_fit_source_receipt = {
        **train_fit_source_receipt_base,
        "receipt_sha256": semantic_sha256(train_fit_source_receipt_base),
    }
    return _build_stage_bundle(
        _builder_issuer=_STAGE_BUNDLE_ISSUER,
        context=context,
        action=G4_PROJECTION_ACTION,
        result_status="ID_MAPPING_PROJECTED_AWAITING_COMPLETION",
        next_action=G4_PROJECTION_COMPLETE_ACTION,
        outputs={
            G4_PROJECTION_OUTPUTS[0]: mapping_bytes,
            G4_PROJECTION_OUTPUTS[1]: video_bytes,
            G4_PROJECTION_OUTPUTS[2]: _canonical_file(manifest),
            G4_PROJECTION_OUTPUTS[3]: train_fit_mapping_bytes,
            G4_PROJECTION_OUTPUTS[4]: _canonical_file(
                train_fit_source_receipt
            ),
        },
        input_hashes={
            "projection_running_control": lineage[
                "control_capability_sha256"
            ],
            "projection_access_receipt": lineage[
                "access_ledger_receipt_sha256"
            ],
            "projection_result_seal": lineage[
                "control_result_seal_sha256"
            ],
            "projection_minimal_batch": lineage[
                "minimal_projection_batch_capability_sha256"
            ],
            "projection_minimal_batch_binding": lineage[
                "minimal_projection_batch_binding_sha256"
            ],
            "projection_minimal_records": lineage[
                "minimal_projection_records_sha256"
            ],
            "projection_expected_desc_ids": lineage[
                "expected_desc_ids_sha256"
            ],
            "projection_protected_manifest_contract": lineage[
                "protected_manifest_contract_sha256"
            ],
            "projection_scanner_contract": lineage[
                "scanner_contract_sha256"
            ],
            "projection_source_observation": lineage[
                "source_observation_receipt_sha256"
            ],
            "train_fit_mapping_records": minimal_batch[
                "train_fit_mapping_records_sha256"
            ],
            "train_fit_source_evidence": train_fit_source_receipt_base[
                "source_evidence_sha256"
            ],
            "train_fit_manifest_access": train_fit_source_receipt_base[
                "source_manifest_access_receipt_sha256"
            ],
            "train_fit_train_projection_access": train_fit_source_receipt_base[
                "train_projection_access_receipt_sha256"
            ],
            "projection_manifest": manifest["manifest_sha256"],
        },
        declared_safety_counts=dict(G4_PROJECTION_DECLARED_SAFETY_COUNTS),
        commit_eligible=True,
    )


def _validate_g4_atomic_candidates(
    *,
    candidates: Mapping[str, Any],
    context: StageContext,
    role_binding: Mapping[str, Any],
    power_binding: Mapping[str, Any],
    power_replay_capability_sha256: str,
) -> tuple[
    str,
    Mapping[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]:
    """Use the roles-owned validator, then bind its result to live handles."""

    from .a4_roles import (
        validate_g4_atomic_candidate,
        validate_historical_power_reference_population,
    )

    try:
        current = validate_g4_atomic_candidate(candidates)
        committed_reference = validate_historical_power_reference_population(
            power_binding.get("historical_reference_population")
        )
    except Exception as exc:
        raise StageContractError("G4 shared atomic candidate validation failed") from exc
    _exact_mapping(
        current,
        G4_ATOMIC_CANDIDATE_FIELDS,
        "G4 atomic candidate exported field set drift",
    )
    candidate_sha = _sha256(current["atomic_candidate_sha256"], "G4 atomic candidate SHA")
    power_capability_sha = _sha256(
        power_replay_capability_sha256,
        "G4 power replay capability SHA",
    )
    _require(
        current["goal_id"] == context.goal_id
        and current["attempt_id"] == context.attempt_id
        and current["role_input_snapshot_sha256"]
        == role_binding["role_input_snapshot_sha256"]
        and current["power_replay_capability_binding_sha256"]
        == power_binding["power_replay_capability_binding_sha256"]
        and current["power_replay_capability_sha256"]
        == power_capability_sha,
        "G4 shared candidate/current handle binding mismatch",
    )
    role_manifests = current["role_manifests"]
    role_lock = dict(current["role_lock"])
    power_audit = dict(
        _exact_mapping(
            current["cluster_power_audit"],
            G4_CLUSTER_POWER_AUDIT_FIELDS,
            "G4 cluster power audit field set drift",
        )
    )
    policy = dict(current["formal_data_policy"])
    power_lineage = dict(
        _exact_mapping(
            power_audit.get("power_evidence_binding"),
            G4_POWER_EVIDENCE_BINDING_FIELDS,
            "G4 power evidence binding field set drift",
        )
    )
    trusted_power_receipt = _exact_mapping(
        power_lineage.get("trusted_power_computation_receipt"),
        G4_TRUSTED_POWER_COMPUTATION_RECEIPT_FIELDS,
        "G4 trusted power computation receipt field set drift",
    )
    power_results = power_audit.get("power_results")
    power_expansion_report = dict(
        _exact_mapping(
            power_audit.get("power_expansion_report"),
            G4_POWER_EXPANSION_REPORT_FIELDS,
            "G4 power expansion report field set drift",
        )
    )
    reference = power_audit.get("historical_reference_population")
    role_counts = role_lock.get("role_counts")
    v3_power_contract = frozen_power_evaluator_contract()
    v3_resampling = v3_power_contract["bootstrap_test_contract"][
        "deterministic_resampling"
    ]
    v3_projection = v3_power_contract["candidate_role_projection_contract"]
    _require(
        v3_power_contract.get("schema_version")
        == "c28f_g2_power_evaluator_contract_v2"
        and v3_power_contract.get("algorithm")
        == (
            "HISTORICAL_REFERENCE_PAIRED_CLUSTER_POSITION_BOOTSTRAP_"
            "10000_FIXED_SEED_V3"
        )
        and v3_power_contract.get("contract_sha256")
        == semantic_sha256(
            v3_power_contract,
            excluded_fields=("contract_sha256",),
        )
        and v3_projection.get(
            "historical_outcomes_may_determine_preregistered_prefix_length"
        )
        is True
        and v3_projection.get(
            "historical_outcomes_select_candidate_identity_order"
        )
        is False
        and v3_projection.get(
            "historical_outcomes_select_within_prefix_members"
        )
        is False
        and v3_projection.get(
            "cross_role_historical_outcomes_affect_lane_identity_order_or_members"
        )
        is False
        and v3_projection.get("expansion_lane_algorithm")
        == POWER_EXPANSION_LANE_ALGORITHM
        and v3_projection.get("membership_lock_timing")
        == "ALL_PREFIX_EXPANSION_BEFORE_ROLE_MANIFEST_LOCK"
        and "historical_outcomes_may_determine_expansion_prefix_length"
        not in v3_projection
        and power_lineage.get("power_evaluator_contract_sha256")
        == v3_power_contract["contract_sha256"]
        and power_lineage.get("power_replay_capability_binding_sha256")
        == power_binding["power_replay_capability_binding_sha256"]
        and power_lineage.get("power_replay_capability_sha256")
        == power_capability_sha
        and power_lineage.get("status")
        == "CONTROLLER_RECOMPUTED_TRUSTED_POWER"
        and trusted_power_receipt.get("schema_version")
        == G4_TRUSTED_POWER_COMPUTATION_RECEIPT_SCHEMA
        and trusted_power_receipt.get("status")
        == "CONTROLLER_RECOMPUTED_TRUSTED_POWER"
        and trusted_power_receipt.get("raw_power_rows_disclosed") is False
        and trusted_power_receipt.get("receipt_sha256")
        == semantic_sha256(
            trusted_power_receipt,
            excluded_fields=("receipt_sha256",),
        )
        and power_lineage.get("evidence_sha256")
        == trusted_power_receipt.get("receipt_sha256")
        and trusted_power_receipt.get(
            "power_replay_capability_sha256"
        )
        == power_capability_sha
        and trusted_power_receipt.get("g2_committed_transaction_id")
        == power_lineage.get("committed_transaction_id")
        and trusted_power_receipt.get("g2_committed_state_sha256")
        == power_lineage.get("committed_state_sha256")
        and trusted_power_receipt.get("g2_committed_event_sha256")
        == power_lineage.get("committed_event_sha256")
        and trusted_power_receipt.get("power_rows_sha256")
        == power_lineage.get("power_rows_sha256")
        and trusted_power_receipt.get(
            "historical_reference_population_sha256"
        )
        == power_lineage.get("historical_reference_population_sha256")
        and trusted_power_receipt.get("power_evaluator_contract_sha256")
        == power_lineage.get("power_evaluator_contract_sha256")
        and power_lineage.get("goal_id") == power_binding["goal_id"]
        and power_lineage.get("attempt_id") == power_binding["attempt_id"]
        and power_lineage.get("committed_transaction_id")
        == power_binding["committed_transaction_id"]
        and power_lineage.get("committed_state_sha256")
        == power_binding["committed_state_sha256"]
        and power_lineage.get("committed_event_sha256")
        == power_binding["committed_event_sha256"]
        and power_lineage.get("g2_replay_control_receipt_sha256")
        == power_binding["g2_replay_control_receipt_sha256"]
        and power_lineage.get("source_registry_sha256")
        == power_binding["source_registry_sha256"]
        and power_lineage.get("power_rows_file_sha256")
        == power_binding["power_rows_file_sha256"]
        and power_lineage.get("power_rows_sha256")
        == power_binding["power_rows_sha256"]
        and power_lineage.get("power_row_count")
        == power_binding["power_row_count"]
        and power_lineage.get("historical_reference_file_sha256")
        == power_binding["historical_reference_file_sha256"]
        and power_lineage.get("replay_packet_sha256")
        == power_binding["replay_packet_sha256"]
        and power_lineage.get("query_manifest_sha256")
        == power_binding["query_manifest_sha256"]
        and power_lineage.get("power_contract_file_sha256")
        == power_binding["power_contract_file_sha256"]
        and isinstance(power_results, Mapping)
        and set(power_results)
        == {"train_fit_mechanism_dev", "train_fit_confirm"}
        and isinstance(reference, Mapping)
        and reference.get("schema_version")
        == "c28f_g2_historical_power_reference_v1"
        and reference.get("reference_population_sha256")
        == semantic_sha256(
            reference,
            excluded_fields=("reference_population_sha256",),
        )
        and reference.get("power_rows_sha256")
        == power_binding["power_rows_sha256"]
        and canonical_json_bytes(reference)
        == canonical_json_bytes(committed_reference)
        and isinstance(
            power_binding.get("historical_reference_file_sha256"), str
        )
        and len(power_binding["historical_reference_file_sha256"]) == 64
        and reference.get("power_row_count")
        == power_binding["power_row_count"]
        and reference.get("replay_packet_sha256")
        == power_binding["replay_packet_sha256"]
        and reference.get("query_manifest_sha256")
        == power_binding["query_manifest_sha256"]
        and reference.get("source_registry_sha256")
        == power_binding["source_registry_sha256"]
        and reference.get("train_fit_desc_id_join_performed") is False
        and reference.get("train_fit_video_id_join_performed") is False
        and reference.get("historical_performance_selects_membership")
        is False
        and isinstance(reference.get("reference_outcome_distribution"), Mapping)
        and set(reference["reference_outcome_distribution"])
        == set(POWER_OUTCOME_KEYS)
        and all(
            isinstance(counts, Mapping)
            and set(counts) == {"zero_count", "one_count"}
            and type(counts["zero_count"]) is int
            and type(counts["one_count"]) is int
            and counts["zero_count"] >= 0
            and counts["one_count"] >= 0
            and counts["zero_count"] + counts["one_count"]
            == reference["power_row_count"]
            for counts in reference["reference_outcome_distribution"].values()
        )
        and reference.get("candidate_membership_fields_consumed")
        == [
            "actual_candidate_role_query_count",
            "actual_candidate_role_gt_video_cluster_count",
        ]
        and isinstance(role_counts, Mapping)
        and all(
            isinstance(result, Mapping)
            and result.get("status") == "PASS"
            and result.get("bootstrap_resample_count") == 10_000
            and result.get("candidate_role_query_count")
            == role_counts[role]["queries"]
            and result.get("candidate_role_gt_video_cluster_count")
            == role_counts[role]["gt_videos"]
            and result.get("reference_population_sha256")
            == reference["reference_population_sha256"]
            and result.get("reference_query_count")
            == reference["power_row_count"]
            and result.get("reference_gt_video_cluster_count")
            == reference["reference_gt_video_cluster_count"]
            and result.get("power_input_semantics")
            == (
                "OUTER_LOCAL_PAIRED_REFERENCE_BOOTSTRAP_PLUS_"
                "CANDIDATE_ROLE_COUNTS_ONLY"
            )
            and result.get("power_algorithm")
            == v3_power_contract["algorithm"]
            and result.get("power_counter_domain")
            == v3_resampling["index_stream_domain"]
            and result.get("candidate_cluster_size_distribution_consumed")
            is False
            and result.get("outer_experiment_count") == 200
            and result.get("inner_resamples_per_outer") == 49
            and result.get("paired_delta_semantics")
            == (
                "SAME_SLOT_BASELINE_CANDIDATE_SIGN_NORMALIZED_"
                "IMPROVEMENT_MEAN"
            )
            and result.get("train_fit_desc_id_join_performed") is False
            and result.get("train_fit_video_id_join_performed") is False
            and result.get("historical_performance_selects_membership")
            is False
            for role, result in power_results.items()
        ),
        "G4 power audit lacks independent reference/count-only lineage",
    )
    _require(
        role_lock.get("missing_set") == []
        and role_lock.get("pairwise_desc_disjoint") is True
        and role_lock.get("pairwise_gt_video_disjoint") is True
        and role_lock.get("train_fit_union_complete") is True
        and power_audit.get("status")
        == "VALIDATED_CONTROL_POWER_CANDIDATE_AWAITING_STAGE_COMMIT"
        and power_audit.get("candidate_membership_assignment_inputs")
        == (
            "TRAIN_FIT_GT_VIDEO_CLUSTERS_PLUS_FROZEN_SEEDED_DISJOINT_ROLE_"
            "LANE_ORDERS_PLUS_POWER_DETERMINED_PER_ROLE_PREFIX_LENGTH_BEFORE_"
            "MANIFEST_LOCK"
        )
        and power_audit.get(
            "membership_identity_order_fixed_before_power_computation"
        )
        is True
        and power_audit.get(
            "historical_outcomes_may_determine_preregistered_prefix_length"
        )
        is True
        and power_audit.get(
            "historical_outcomes_select_identity_order_or_within_prefix_members"
        )
        is False
        and power_audit.get(
            "cross_role_historical_outcomes_affect_lane_identity_order_or_members"
        )
        is False
        and power_audit.get("membership_lock_timing")
        == "ALL_PREFIX_EXPANSION_BEFORE_ROLE_MANIFEST_LOCK"
        and power_audit.get("expansion_lane_algorithm")
        == POWER_EXPANSION_LANE_ALGORITHM
        and isinstance(
            power_audit.get("frozen_role_lane_video_ids_sha256s"), Mapping
        )
        and set(power_audit["frozen_role_lane_video_ids_sha256s"])
        == {"train_fit_mechanism_dev", "train_fit_confirm"}
        and all(
            isinstance(value, str) and len(value) == 64
            for value in power_audit[
                "frozen_role_lane_video_ids_sha256s"
            ].values()
        )
        and isinstance(power_audit.get("reserved_core_video_ids_sha256"), str)
        and len(power_audit["reserved_core_video_ids_sha256"]) == 64
        and power_audit.get("role_specific_expansion_lanes_disjoint") is True
        and power_audit.get("power_result_use")
        == (
            "PREREGISTERED_PER_ROLE_LANE_PREFIX_LENGTH_AND_STOP_POINT_ONLY_"
            "BEFORE_MANIFEST_LOCK"
        )
        and power_audit.get("effect_size_or_performance_claim") is None
        and power_audit.get("power_expansion_report_sha256")
        == semantic_sha256(power_expansion_report)
        and power_expansion_report.get("status") == "PASS"
        and power_expansion_report.get("expansion_policy")
        == "PREREGISTERED_DISJOINT_ROLE_LANE_PREFIXES_BEFORE_MANIFEST_LOCK"
        and power_expansion_report.get("minimum_effect_registry_sha256")
        == v3_power_contract["minimum_effect_registry_sha256"]
        and power_expansion_report.get("effect_outcome_registry_sha256")
        == v3_power_contract["effect_outcome_registry_sha256"]
        and power_expansion_report.get("effect_threshold_reduction_count") == 0
        and power_expansion_report.get("historical_reference_population")
        == reference
        and power_expansion_report.get("membership_assignment_inputs")
        == power_audit.get("candidate_membership_assignment_inputs")
        and power_expansion_report.get(
            "historical_outcomes_select_identity_order_or_within_prefix_members"
        )
        is False
        and power_expansion_report.get(
            "cross_role_historical_outcomes_affect_lane_identity_order_or_members"
        )
        is False
        and power_expansion_report.get(
            "historical_outcomes_may_determine_preregistered_prefix_length"
        )
        is True
        and power_expansion_report.get("membership_lock_timing")
        == "ALL_PREFIX_EXPANSION_BEFORE_ROLE_MANIFEST_LOCK"
        and power_expansion_report.get("expansion_lane_algorithm")
        == POWER_EXPANSION_LANE_ALGORITHM
        and isinstance(
            power_expansion_report.get("frozen_core_prefix_video_ids_sha256"),
            str,
        )
        and len(power_expansion_report["frozen_core_prefix_video_ids_sha256"])
        == 64
        and power_expansion_report.get("frozen_role_lane_video_ids_sha256s")
        == power_audit.get("frozen_role_lane_video_ids_sha256s")
        and power_expansion_report.get("reserved_core_video_ids_sha256")
        == power_audit.get("reserved_core_video_ids_sha256")
        and power_expansion_report.get("role_specific_expansion_lanes_disjoint")
        is True
        and isinstance(power_expansion_report.get("adjustment_rounds"), list)
        and power_expansion_report.get("role_results") == power_results
        and power_expansion_report.get("trusted_power_computation_receipt")
        == trusted_power_receipt
        and power_audit.get("interpretation")
        == (
            "INDEPENDENT_HISTORICAL_REFERENCE_POWER_WITH_CANDIDATE_COUNTS_"
            "ONLY_NO_POSTHOC_EFFECT_REDUCTION_NO_IDENTITY_ORDER_OR_WITHIN_"
            "PREFIX_SELECTION_POWER_DETERMINES_PER_ROLE_LANE_PREFIX_LENGTH_"
            "ONLY_BEFORE_ROLE_MANIFEST_LOCK"
        )
        and "NO_PERFORMANCE_MEMBERSHIP_SELECTION"
        not in power_audit["interpretation"]
        and "historical_outcomes_may_determine_expansion_prefix_length"
        not in power_audit
        and policy.get("policy") == "STRICT_CORE_ONLY"
        and policy.get("U_formal_updates") == 17_360
        and current["artifact_candidate_hashes"]["role_manifest_hashes"]
        == {
            role: dict(manifest)["manifest_sha256"]
            for role, manifest in sorted(role_manifests.items())
        },
        "G4 candidate is incomplete, insufficient, or lacks actual bootstrap evidence",
    )
    for role, manifest in role_manifests.items():
        assignment = _exact_mapping(
            manifest.get("assignment"),
            G4_ROLE_ASSIGNMENT_FIELDS,
            "G4 role assignment field set drift: %s" % role,
        )
        _require(
            assignment.get("algorithm") == POWER_ADAPTIVE_ASSIGNMENT_ALGORITHM
            and assignment.get("seed") == ASSIGNMENT_SEED
            and assignment.get("cluster_order")
            == "sha256(seed_ascii + NUL + video_id_utf8),video_id_utf8"
            and assignment.get("cluster_atomic") is True
            and assignment.get("candidate_membership_source")
            == (
                "TRAIN_FIT_GT_VIDEO_CLUSTERS_PLUS_FROZEN_SEEDED_DISJOINT_ROLE_"
                "LANE_ORDERS_PLUS_POWER_DETERMINED_PER_ROLE_PREFIX_LENGTH_BEFORE_"
                "MANIFEST_LOCK"
            )
            and assignment.get("historical_reference_population_sha256")
            == reference["reference_population_sha256"]
            and assignment.get(
                "historical_outcomes_select_identity_order_or_within_prefix_members"
            )
            is False
            and assignment.get(
                "cross_role_historical_outcomes_affect_lane_identity_order_or_members"
            )
            is False
            and assignment.get(
                "historical_outcomes_may_determine_preregistered_prefix_length"
            )
            is True
            and assignment.get("power_expansion_lane_algorithm")
            == POWER_EXPANSION_LANE_ALGORITHM
            and assignment.get("membership_lock_timing")
            == "ALL_PREFIX_EXPANSION_BEFORE_ROLE_MANIFEST_LOCK"
            and assignment.get("power_expansion_report_sha256")
            == power_audit["power_expansion_report_sha256"]
            and assignment.get("assignment_contract_sha256")
            == semantic_sha256(
                assignment,
                excluded_fields=("assignment_contract_sha256",),
            ),
            "G4 role assignment prefix authority drift: %s" % role,
        )
    return candidate_sha, role_manifests, role_lock, power_audit, policy


def _validate_g4_atomic_bundle_binding(value: Any) -> Mapping[str, Any]:
    """Freeze the explicit v3 lane/cross-role safety summary."""

    binding = _exact_mapping(
        value,
        G4_ATOMIC_BUNDLE_BINDING_FIELDS,
        "G4 atomic bundle binding field set drift",
    )
    lane_hashes = binding["frozen_role_lane_video_ids_sha256s"]
    _require(
        binding["schema_version"] == G4_ATOMIC_BUNDLE_BINDING_SCHEMA
        and binding["status"]
        == "CANDIDATE_AWAITING_A4_STAGE_COMMIT_RECEIPT"
        and isinstance(binding["goal_id"], str)
        and bool(binding["goal_id"])
        and isinstance(binding["attempt_id"], str)
        and bool(binding["attempt_id"])
        and binding["candidate_membership_assignment_inputs"]
        == (
            "TRAIN_FIT_GT_VIDEO_CLUSTERS_PLUS_FROZEN_SEEDED_DISJOINT_ROLE_"
            "LANE_ORDERS_PLUS_POWER_DETERMINED_PER_ROLE_PREFIX_LENGTH_BEFORE_"
            "MANIFEST_LOCK"
        )
        and binding[
            "membership_identity_order_fixed_before_power_computation"
        ]
        is True
        and binding["membership_lock_timing"]
        == "ALL_PREFIX_EXPANSION_BEFORE_ROLE_MANIFEST_LOCK"
        and binding[
            "historical_outcomes_may_determine_preregistered_prefix_length"
        ]
        is True
        and binding[
            "historical_outcomes_select_identity_order_or_within_prefix_members"
        ]
        is False
        and binding[
            "cross_role_historical_outcomes_affect_lane_identity_order_or_members"
        ]
        is False
        and binding["expansion_lane_algorithm"]
        == POWER_EXPANSION_LANE_ALGORITHM
        and isinstance(lane_hashes, Mapping)
        and set(lane_hashes)
        == {"train_fit_mechanism_dev", "train_fit_confirm"}
        and binding["role_specific_expansion_lanes_disjoint"] is True
        and binding["power_result_use"]
        == (
            "PREREGISTERED_PER_ROLE_LANE_PREFIX_LENGTH_AND_STOP_POINT_ONLY_"
            "BEFORE_MANIFEST_LOCK"
        )
        and binding["bootstrap_replicates"] == 10_000
        and binding["pass_claim_available_before_stage_commit"] is False,
        "G4 atomic bundle binding lane authority drift",
    )
    for field in (
        "role_input_handle_sha256",
        "power_replay_capability_sha256",
        "power_replay_capability_binding_sha256",
        "trusted_power_computation_receipt_sha256",
        "atomic_candidate_sha256",
        "role_lock_sha256",
        "cluster_power_audit_sha256",
        "formal_data_policy_sha256",
        "reserved_core_video_ids_sha256",
        "power_expansion_report_sha256",
    ):
        _sha256(binding[field], "G4 atomic bundle binding.%s" % field)
    for role, lane_sha in lane_hashes.items():
        _sha256(lane_sha, "G4 atomic bundle binding lane SHA: %s" % role)
    _require(
        binding["binding_sha256"]
        == semantic_sha256(binding, excluded_fields=("binding_sha256",)),
        "G4 atomic bundle binding self-hash drift",
    )
    return dict(binding)


def build_g4_bundle(
    *,
    context: StageContext,
    role_input_handle: Any,
    power_replay_capability: Any,
) -> StageBundle:
    """Build the atomic G4 candidate from two concrete control capabilities.

    The G2 capability supplies a committed, role-independent historical power
    reference plus the frozen evaluator contract.  The train-fit GT-video
    cluster identity order is split into two pre-registered, mutually disjoint
    seeded expansion lanes, with a separately reserved core suffix.  Every
    member inside each lane prefix is fixed before power computation.  Before
    the manifest is locked, independent historical power may determine only
    how far each role's already-ordered lane prefix expands and where it stops;
    one role's outcomes may not affect the other role's identity order or
    members, and no historical outcome may join to a train-fit desc/video ID.
    The persisted controller-computation receipt binds the live replay
    capability, result hashes, resulting role counts, and evaluator contract.
    No power-driven membership change is possible after the manifest lock.
    """

    context.validate(G4_ACTION, "ID_MAPPING_COMPLETE")
    from .a4_control import (
        A4G2PowerReplayCapability,
        A4G4RoleLockInputHandle,
        validate_g2_power_replay_capability,
        validate_g4_role_lock_input_handle,
    )
    from .a4_roles import build_g4_role_policy_candidates

    _require(
        type(role_input_handle) is A4G4RoleLockInputHandle,
        "concrete A4G4RoleLockInputHandle required",
    )
    _require(
        type(power_replay_capability) is A4G2PowerReplayCapability,
        "concrete A4G2PowerReplayCapability required",
    )
    try:
        role_binding = validate_g4_role_lock_input_handle(role_input_handle)
        power_binding = validate_g2_power_replay_capability(
            power_replay_capability
        )
    except Exception as exc:
        raise StageContractError("G4 concrete input capability validation failed") from exc
    _require(
        role_binding.get("goal_id") == context.goal_id
        and role_binding.get("attempt_id") == context.attempt_id
        and role_binding.get("committed_state_sha256")
        == context.committed_control_state_sha256
        and role_binding.get("committed_event_sha256")
        == context.committed_control_event_sha256
        and role_binding.get("committed_transaction_id")
        == context.control_transaction_id,
        "G4 role input/current control anchor mismatch",
    )
    _require(
        power_binding.get("goal_id") == context.goal_id
        and power_binding.get("attempt_id") == context.attempt_id,
        "G4 power replay owner mismatch",
    )
    candidates = dict(
        build_g4_role_policy_candidates(
            role_input_handle=role_input_handle,
            power_replay_capability=power_replay_capability,
        )
    )
    (
        candidate_sha,
        role_manifests,
        role_lock,
        power_audit,
        policy,
    ) = _validate_g4_atomic_candidates(
        candidates=candidates,
        context=context,
        role_binding=role_binding,
        power_binding=power_binding,
        power_replay_capability_sha256=(
            power_replay_capability.capability_sha256
        ),
    )
    trusted_power_receipt_sha = _sha256(
        power_audit["power_evidence_binding"][
            "trusted_power_computation_receipt"
        ]["receipt_sha256"],
        "G4 trusted power computation receipt SHA",
    )
    binding_base = {
        "schema_version": G4_ATOMIC_BUNDLE_BINDING_SCHEMA,
        "status": "CANDIDATE_AWAITING_A4_STAGE_COMMIT_RECEIPT",
        "goal_id": context.goal_id,
        "attempt_id": context.attempt_id,
        "role_input_handle_sha256": role_input_handle.capability_sha256,
        "power_replay_capability_sha256": power_replay_capability.capability_sha256,
        "power_replay_capability_binding_sha256": power_binding[
            "power_replay_capability_binding_sha256"
        ],
        "trusted_power_computation_receipt_sha256": (
            trusted_power_receipt_sha
        ),
        "atomic_candidate_sha256": candidate_sha,
        "role_lock_sha256": role_lock["lock_sha256"],
        "cluster_power_audit_sha256": power_audit["audit_sha256"],
        "formal_data_policy_sha256": policy["policy_sha256"],
        "candidate_membership_assignment_inputs": power_audit[
            "candidate_membership_assignment_inputs"
        ],
        "membership_identity_order_fixed_before_power_computation": (
            power_audit[
                "membership_identity_order_fixed_before_power_computation"
            ]
        ),
        "membership_lock_timing": power_audit["membership_lock_timing"],
        "historical_outcomes_may_determine_preregistered_prefix_length": (
            power_audit[
                "historical_outcomes_may_determine_preregistered_prefix_length"
            ]
        ),
        "historical_outcomes_select_identity_order_or_within_prefix_members": (
            power_audit[
                "historical_outcomes_select_identity_order_or_within_prefix_members"
            ]
        ),
        "cross_role_historical_outcomes_affect_lane_identity_order_or_members": (
            power_audit[
                "cross_role_historical_outcomes_affect_lane_identity_order_or_members"
            ]
        ),
        "expansion_lane_algorithm": power_audit["expansion_lane_algorithm"],
        "frozen_role_lane_video_ids_sha256s": dict(
            power_audit["frozen_role_lane_video_ids_sha256s"]
        ),
        "reserved_core_video_ids_sha256": power_audit[
            "reserved_core_video_ids_sha256"
        ],
        "role_specific_expansion_lanes_disjoint": power_audit[
            "role_specific_expansion_lanes_disjoint"
        ],
        "power_result_use": power_audit["power_result_use"],
        "power_expansion_report_sha256": power_audit[
            "power_expansion_report_sha256"
        ],
        "bootstrap_replicates": 10_000,
        "pass_claim_available_before_stage_commit": False,
    }
    binding = {
        **binding_base,
        "binding_sha256": semantic_sha256(binding_base),
    }
    binding = _validate_g4_atomic_bundle_binding(binding)
    outputs = {
        G4_OUTPUTS[0]: _canonical_file(role_lock),
        G4_OUTPUTS[1]: _canonical_file(power_audit),
        G4_OUTPUTS[2]: _canonical_file(policy),
        G4_OUTPUTS[3]: _canonical_file(binding),
    }
    for role, manifest in sorted(role_manifests.items()):
        current = dict(manifest)
        _require(
            current.get("manifest_sha256")
            == semantic_sha256(current, excluded_fields=("manifest_sha256",)),
            "G4 role manifest self-hash mismatch: %s" % role,
        )
        outputs["f1_protocol/role_manifests/%s.json" % role] = _canonical_file(
            current
        )
    return _build_stage_bundle(
        _builder_issuer=_STAGE_BUNDLE_ISSUER,
        context=context,
        action=G4_ACTION,
        result_status="ROLE_POLICY_LOCKED",
        next_action=G5_ACTION,
        outputs=outputs,
        input_hashes={
            "role_input_handle": role_input_handle.capability_sha256,
            "g2_power_replay_capability": power_replay_capability.capability_sha256,
            "g2_power_replay_capability_binding": power_binding[
                "power_replay_capability_binding_sha256"
            ],
            "g2_trusted_power_computation_receipt": (
                trusted_power_receipt_sha
            ),
            "g2_historical_power_reference": power_binding[
                "historical_reference_population"
            ]["reference_population_sha256"],
            "g2_historical_power_reference_file": power_binding[
                "historical_reference_file_sha256"
            ],
            "g4_candidate": candidate_sha,
        },
        declared_safety_counts=dict(G4_ROLE_DECLARED_SAFETY_COUNTS),
        commit_eligible=True,
    )


def _g6_canonical_absolute_child_path(value: Any, field: str) -> str:
    _require(
        isinstance(value, str)
        and value.startswith("/")
        and len(value) > 1
        and not value.endswith("/")
        and "\\" not in value
        and "//" not in value
        and all(part not in {"", ".", ".."} for part in value[1:].split("/")),
        "%s is not a canonical absolute child path" % field,
    )
    return value


def _g6_safe_artifact_name(value: Any, field: str) -> str:
    _require(
        isinstance(value, str)
        and value.isascii()
        and 1 <= len(value) <= 128
        and value[0].isalnum()
        and all(
            character.isalnum() or character in "_.-"
            for character in value
        )
        and "/" not in value
        and "\\" not in value,
        "%s is not a safe flat artifact name" % field,
    )
    return value


def _validate_g6_actual_postimage(
    value: Any,
    *,
    synthetic_root_handle_sha256: str,
    crash_point: str,
    child_root_path: str,
    parent_descriptor_identity_sha256: str,
    child_root_binding_sha256: str,
) -> Mapping[str, Any]:
    postimage = _exact_mapping(
        value,
        G6_ACTUAL_POSTIMAGE_FIELDS,
        "G6 actual postimage field set drift",
    )
    root_identity = _exact_mapping(
        postimage["root_identity"],
        G6_ACTUAL_POSTIMAGE_ROOT_IDENTITY_FIELDS,
        "G6 actual postimage root identity field set drift",
    )
    expected_parent_path = child_root_path.rsplit("/", 1)[0] or "/"
    parent_path = _g6_canonical_absolute_child_path(
        root_identity["parent_canonical_path"],
        "G6 actual postimage parent path",
    )
    _require(
        postimage["schema_version"] == G6_ACTUAL_POSTIMAGE_SCHEMA
        and postimage["synthetic_root_handle_sha256"]
        == synthetic_root_handle_sha256
        and postimage["crash_point"] == crash_point
        and root_identity["canonical_path"] == child_root_path
        and parent_path == expected_parent_path
        and type(root_identity["device"]) is int
        and root_identity["device"] > 0
        and type(root_identity["inode"]) is int
        and root_identity["inode"] > 0
        and type(root_identity["nlink"]) is int
        and root_identity["nlink"] == 2
        and isinstance(root_identity["mode"], str)
        and root_identity["mode"] == "0o700"
        and root_identity["mode"].startswith("0o")
        and len(root_identity["mode"]) in {5, 6}
        and all(
            character in "01234567" for character in root_identity["mode"][2:]
        )
        and type(root_identity["mtime_ns"]) is int
        and root_identity["mtime_ns"] > 0
        and type(root_identity["ctime_ns"]) is int
        and root_identity["ctime_ns"] > 0
        and type(root_identity["mount_id"]) is int
        and root_identity["mount_id"] > 0
        and root_identity["parent_descriptor_identity_sha256"]
        == parent_descriptor_identity_sha256
        and root_identity["child_root_binding_sha256"]
        == child_root_binding_sha256,
        "G6 actual postimage root binding mismatch",
    )
    _sha256(
        root_identity["parent_descriptor_identity_sha256"],
        "G6 postimage parent descriptor identity SHA",
    )
    _sha256(
        root_identity["child_root_binding_sha256"],
        "G6 postimage child root binding SHA",
    )
    inventory = postimage["artifact_inventory"]
    _require(
        isinstance(inventory, list)
        and type(postimage["artifact_count"]) is int
        and 1 <= postimage["artifact_count"] <= 16
        and postimage["artifact_count"] == len(inventory),
        "G6 actual postimage artifact count invalid",
    )
    names: list[str] = []
    inodes: list[int] = []
    detached_inventory: list[dict[str, Any]] = []
    for index, raw_row in enumerate(inventory):
        row = _exact_mapping(
            raw_row,
            G6_ACTUAL_POSTIMAGE_INVENTORY_ROW_FIELDS,
            "G6 actual postimage inventory field set drift at %d" % index,
        )
        name = _g6_safe_artifact_name(
            row["name"],
            "G6 actual postimage inventory name at %d" % index,
        )
        _require(
            type(row["size_bytes"]) is int
            and row["size_bytes"] > 0
            and type(row["device"]) is int
            and row["device"] == root_identity["device"]
            and type(row["inode"]) is int
            and row["inode"] > 0
            and row["inode"] != root_identity["inode"]
            and type(row["nlink"]) is int
            and row["nlink"] == 1
            and isinstance(row["mode"], str)
            and row["mode"] == "0o600"
            and row["mode"].startswith("0o")
            and len(row["mode"]) in {5, 6}
            and all(
                character in "01234567" for character in row["mode"][2:]
            )
            and type(row["mtime_ns"]) is int
            and row["mtime_ns"] > 0
            and type(row["ctime_ns"]) is int
            and row["ctime_ns"] > 0
            and type(row["mount_id"]) is int
            and row["mount_id"] == root_identity["mount_id"],
            "G6 actual postimage inventory stat mismatch at %d" % index,
        )
        _sha256(row["sha256"], "G6 actual postimage artifact SHA")
        names.append(name)
        inodes.append(row["inode"])
        detached_inventory.append(dict(row))
    _require(
        names == sorted(names)
        and len(names) == len(set(names))
        and len(inodes) == len(set(inodes)),
        "G6 actual postimage inventory is not canonical and unique",
    )
    from . import a4_f1

    expected_material = a4_f1._synthetic_atomic_recovery_expected_material(
        crash_point
    )
    expected_artifacts = expected_material["expected_artifact_bytes"]
    _require(
        names == sorted(expected_artifacts)
        and all(
            row["size_bytes"] == len(expected_artifacts[row["name"]])
            and row["sha256"]
            == bytes_sha256(expected_artifacts[row["name"]])
            for row in detached_inventory
        ),
        "G6 actual postimage differs from the deterministic crash oracle",
    )
    inventory_sha = _sha256(
        postimage["artifact_inventory_sha256"],
        "G6 actual postimage inventory SHA",
    )
    postimage_sha = _sha256(
        postimage["postimage_sha256"],
        "G6 actual postimage SHA",
    )
    _require(
        inventory_sha
        == bytes_sha256(canonical_json_bytes(detached_inventory))
        and postimage_sha
        == semantic_sha256(
            postimage,
            excluded_fields=("postimage_sha256",),
        ),
        "G6 actual postimage content hash mismatch",
    )
    return {
        **dict(postimage),
        "root_identity": dict(root_identity),
        "artifact_inventory": detached_inventory,
    }


def _validate_g6_crash_receipt(
    value: Any,
    *,
    goal_id: str,
    attempt_id: str,
    synthetic_root_handle_sha256: str,
    required_crash_points: set[str],
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    receipt = _exact_mapping(
        value,
        G6_CRASH_RECEIPT_FIELDS,
        "G6 crash receipt field set drift",
    )
    crash_point = receipt["crash_point"]
    child_root_name = receipt["child_root_name"]
    child_root_path = _g6_canonical_absolute_child_path(
        receipt["child_root_path"],
        "G6 child root path",
    )
    _require(
        isinstance(child_root_name, str)
        and child_root_name.isascii()
        and child_root_name.startswith("scenario-")
        and 12 <= len(child_root_name) <= 89
        and child_root_name == child_root_path.rsplit("/", 1)[-1]
        and all(
            character.islower()
            or character.isdigit()
            or character == "-"
            for character in child_root_name
        ),
        "G6 child root name/path mismatch",
    )
    parent_identity_sha = _sha256(
        receipt["parent_descriptor_identity_sha256"],
        "G6 parent descriptor identity SHA",
    )
    child_contract_sha = _sha256(
        receipt["child_root_contract_sha256"],
        "G6 child root contract SHA",
    )
    child_binding_sha = _sha256(
        receipt["child_root_binding_sha256"],
        "G6 child root binding SHA",
    )
    evidence_sha = _sha256(
        receipt["evidence_sha256"],
        "G6 synthetic evidence SHA",
    )
    evidence_file_sha = _sha256(
        receipt["evidence_file_sha256"],
        "G6 synthetic evidence file SHA",
    )
    reread_postimage_sha = _sha256(
        receipt["controller_reread_postimage_sha256"],
        "G6 controller reread postimage SHA",
    )
    _require(
        receipt["schema_version"] == G6_CRASH_RECEIPT_SCHEMA
        and receipt["status"] == "PASS"
        and receipt["goal_id"] == goal_id
        and receipt["attempt_id"] == attempt_id
        and receipt["synthetic_root_handle_sha256"]
        == synthetic_root_handle_sha256
        and isinstance(crash_point, str)
        and crash_point in required_crash_points
        and receipt["real_data_optimizer_updates"] == 0
        and receipt["extra_model_forward_splits"] == 0
        and child_binding_sha
        == semantic_sha256(
            {
                "synthetic_root_handle_sha256": (
                    synthetic_root_handle_sha256
                ),
                "crash_point": crash_point,
                "child_root_name": child_root_name,
                "child_root_path": child_root_path,
                "parent_descriptor_identity_sha256": parent_identity_sha,
                "child_root_contract_sha256": child_contract_sha,
            }
        ),
        "G6 crash receipt authority/binding mismatch",
    )
    postimage = _validate_g6_actual_postimage(
        receipt["actual_postimage"],
        synthetic_root_handle_sha256=synthetic_root_handle_sha256,
        crash_point=str(crash_point),
        child_root_path=child_root_path,
        parent_descriptor_identity_sha256=parent_identity_sha,
        child_root_binding_sha256=child_binding_sha,
    )
    _require(
        reread_postimage_sha == postimage["postimage_sha256"]
        and receipt["receipt_sha256"]
        == semantic_sha256(
            receipt,
            excluded_fields=("receipt_sha256",),
        ),
        "G6 crash receipt reread/self-hash mismatch",
    )
    _sha256(receipt["receipt_sha256"], "G6 crash receipt SHA")
    return dict(receipt), postimage


def build_g6_f1_completion_bundle(
    *,
    context: StageContext,
    g6_input_handle: Any,
) -> StageBundle:
    """Promote the F1 candidate only from controller-owned crash/ledger evidence."""

    context.validate(G6_ACTION, "F0B_COMPLETE")
    from .a4_control import (
        A4G6BundleInputHandle,
        validate_g6_bundle_input_handle,
    )

    _require(
        type(g6_input_handle) is A4G6BundleInputHandle,
        "concrete A4G6BundleInputHandle required",
    )
    try:
        snapshot = validate_g6_bundle_input_handle(g6_input_handle)
    except Exception as exc:
        raise StageContractError("G6 concrete input capability validation failed") from exc
    _require(
        snapshot.get("goal_id") == context.goal_id
        and snapshot.get("attempt_id") == context.attempt_id
        and snapshot.get("committed_transaction_id")
        == context.control_transaction_id
        and snapshot.get("committed_state_sha256")
        == context.committed_control_state_sha256
        and snapshot.get("committed_event_sha256")
        == context.committed_control_event_sha256,
        "G6 input/current control anchor mismatch",
    )
    candidate = dict(snapshot.get("candidate_f1_report", {}))
    from . import a4_f1

    expected_candidate = a4_f1.build_f1_protocol_report()
    candidate_sha = _sha256(candidate.get("report_sha256"), "candidate F1 report SHA")
    _require(
        candidate == expected_candidate
        and candidate_sha
        == semantic_sha256(candidate, excluded_fields=("report_sha256",))
        and candidate.get("status")
        == "SYNTHETIC_PROTOCOL_EVIDENCE_AWAITING_CONTROL_LEDGER"
        and candidate.get("g6_commit_eligible") is False
        and candidate.get("ledger_derived_counts") is False,
        "G6 candidate F1 report is not the reviewed noncommittable candidate",
    )
    global_ledger = dict(
        _exact_mapping(
            snapshot.get("committed_global_ledger_snapshot"),
            G6_GLOBAL_LEDGER_FIELDS,
            "G6 global ledger snapshot field set drift",
        )
    )
    global_ledger_sha = _sha256(
        global_ledger.get("snapshot_sha256"),
        "G6 global ledger snapshot SHA",
    )
    _require(
        global_ledger_sha
        == semantic_sha256(global_ledger, excluded_fields=("snapshot_sha256",))
        and global_ledger.get("schema_version") == G6_GLOBAL_LEDGER_SCHEMA
        and global_ledger.get("goal_id") == context.goal_id
        and global_ledger.get("attempt_id") == context.attempt_id
        and global_ledger.get("committed_transaction_id")
        == context.control_transaction_id
        and global_ledger.get("committed_state_sha256")
        == context.committed_control_state_sha256
        and global_ledger.get("committed_event_sha256")
        == context.committed_control_event_sha256
        and global_ledger.get("eval_token_states")
        == {"F0_A": "COMPLETED", "F0_B": "COMPLETED"}
        and global_ledger.get("model_forward_evaluations") == 2
        and global_ledger.get("synthetic_crash_scenario_count") == 4,
        "G6 global ledger snapshot self-hash mismatch",
    )
    _sha256(global_ledger["event_journal_sha256"], "G6 event journal SHA")
    _require(
        global_ledger.get("real_data_optimizer_updates") == 0
        and global_ledger.get("extra_model_forward_splits") == 0,
        "G6 committed global ledger contains forbidden training/forward activity",
    )
    crash_receipts = snapshot.get("crash_evidence_receipts")
    _require(
        isinstance(crash_receipts, list) and len(crash_receipts) == 4,
        "G6 requires exactly four controller-owned crash receipts",
    )
    required_points = {
        "BEFORE_CHECKPOINT_COMMIT",
        "AFTER_CHECKPOINT_BEFORE_MANIFEST",
        "DURING_EVAL",
        "BEFORE_STATUS_RENAME",
    }
    synthetic_root_handle_sha = _sha256(
        snapshot.get("synthetic_root_handle_sha256"),
        "G6 synthetic root handle SHA",
    )
    role_lock_sha = _sha256(
        snapshot.get("role_lock_sha256"),
        "G6 role lock SHA",
    )
    g2_replay_evidence_sha = _sha256(
        snapshot.get("g2_replay_evidence_sha256"),
        "G6 G2 replay evidence SHA",
    )
    formal_policy_sha = _sha256(
        snapshot.get("formal_policy_sha256"),
        "G6 formal policy SHA",
    )
    crash_evidence_set_sha = _sha256(
        snapshot.get("crash_evidence_set_sha256"),
        "G6 crash evidence set SHA",
    )
    _require(
        crash_evidence_set_sha
        == bytes_sha256(canonical_json_bytes(crash_receipts)),
        "G6 crash evidence set content hash mismatch",
    )
    _require(
        global_ledger["synthetic_crash_evidence_set_sha256"]
        == crash_evidence_set_sha,
        "G6 global ledger/crash evidence set mismatch",
    )
    observed_points: set[str] = set()
    crash_hashes: dict[str, str] = {}
    evidence_hashes: dict[str, str] = {}
    evidence_file_hashes: dict[str, str] = {}
    child_binding_hashes: dict[str, str] = {}
    postimage_hashes: dict[str, str] = {}
    reread_postimage_hashes: dict[str, str] = {}
    inventory_hashes: dict[str, str] = {}
    child_names: set[str] = set()
    child_paths: set[str] = set()
    child_root_identities: set[tuple[int, int]] = set()
    parent_identity_hashes: set[str] = set()
    child_contract_hashes: set[str] = set()
    ordered_points: list[str] = []
    for raw_receipt in crash_receipts:
        current, postimage = _validate_g6_crash_receipt(
            raw_receipt,
            goal_id=context.goal_id,
            attempt_id=context.attempt_id,
            synthetic_root_handle_sha256=synthetic_root_handle_sha,
            required_crash_points=required_points,
        )
        point = str(current["crash_point"])
        _require(
            point not in observed_points,
            "G6 crash point set contains a duplicate",
        )
        observed_points.add(point)
        ordered_points.append(point)
        child_names.add(str(current["child_root_name"]))
        child_paths.add(str(current["child_root_path"]))
        root_identity = postimage["root_identity"]
        child_root_identities.add(
            (int(root_identity["device"]), int(root_identity["inode"]))
        )
        parent_identity_hashes.add(
            str(current["parent_descriptor_identity_sha256"])
        )
        child_contract_hashes.add(str(current["child_root_contract_sha256"]))
        crash_hashes[point] = str(current["receipt_sha256"])
        evidence_hashes[point] = str(current["evidence_sha256"])
        evidence_file_hashes[point] = str(current["evidence_file_sha256"])
        child_binding_hashes[point] = str(
            current["child_root_binding_sha256"]
        )
        postimage_hashes[point] = str(postimage["postimage_sha256"])
        reread_postimage_hashes[point] = str(
            current["controller_reread_postimage_sha256"]
        )
        inventory_hashes[point] = str(
            postimage["artifact_inventory_sha256"]
        )
    _require(
        observed_points == required_points
        and ordered_points == sorted(required_points)
        and len(child_names) == 4
        and len(child_paths) == 4
        and len(child_root_identities) == 4
        and len(parent_identity_hashes) == 1
        and len(child_contract_hashes) == 1,
        "G6 crash evidence set is incomplete or not canonically isolated",
    )
    crash_hashes = dict(sorted(crash_hashes.items()))
    evidence_hashes = dict(sorted(evidence_hashes.items()))
    evidence_file_hashes = dict(sorted(evidence_file_hashes.items()))
    child_binding_hashes = dict(sorted(child_binding_hashes.items()))
    postimage_hashes = dict(sorted(postimage_hashes.items()))
    reread_postimage_hashes = dict(sorted(reread_postimage_hashes.items()))
    inventory_hashes = dict(sorted(inventory_hashes.items()))
    report_base = dict(candidate)
    report_base.pop("report_sha256", None)
    report_base["status"] = "F1_COMPLETE_ONLY_WITH_G6_STAGE_COMMIT_RECEIPT"
    report_base["g6_commit_eligible"] = True
    report_base["ledger_derived_counts"] = True
    report_base["real_data_optimizer_updates"] = 0
    report_base["extra_model_forward_splits"] = 0
    atomic_crash_evidence = {
        "status": "PASS_CONTROLLER_OWNED_SYNTHETIC_ROOT",
        "required_crash_points": sorted(required_points),
        "crash_evidence_set_sha256": crash_evidence_set_sha,
        "committed_evidence_sha256s": evidence_hashes,
        "evidence_file_sha256s": evidence_file_hashes,
        "child_root_binding_sha256s": child_binding_hashes,
        "actual_postimage_sha256s": postimage_hashes,
        "controller_reread_postimage_sha256s": reread_postimage_hashes,
        "artifact_inventory_sha256s": inventory_hashes,
        "synthetic_root_handle_sha256": synthetic_root_handle_sha,
    }
    _exact_mapping(
        atomic_crash_evidence,
        G6_ATOMIC_CRASH_EVIDENCE_FIELDS,
        "G6 atomic crash evidence field set drift",
    )
    report_base["atomic_crash_evidence"] = atomic_crash_evidence
    control_evidence_binding = {
        "g6_input_handle_sha256": g6_input_handle.capability_sha256,
        "candidate_report_sha256": candidate_sha,
        "global_ledger_snapshot_sha256": global_ledger_sha,
        "crash_evidence_set_sha256": crash_evidence_set_sha,
        "role_lock_sha256": role_lock_sha,
        "g2_replay_evidence_sha256": g2_replay_evidence_sha,
        "formal_policy_sha256": formal_policy_sha,
        "pass_claim_available_before_stage_commit": False,
    }
    _exact_mapping(
        control_evidence_binding,
        G6_CONTROL_EVIDENCE_BINDING_FIELDS,
        "G6 report control evidence binding field set drift",
    )
    report_base["control_evidence_binding"] = control_evidence_binding
    final_report = {
        **report_base,
        "report_sha256": semantic_sha256(report_base),
    }
    binding_base = {
        "schema_version": G6_CONTROL_BINDING_SCHEMA,
        "status": "CANDIDATE_AWAITING_A4_STAGE_COMMIT_RECEIPT",
        "goal_id": context.goal_id,
        "attempt_id": context.attempt_id,
        "g6_input_handle_sha256": g6_input_handle.capability_sha256,
        "candidate_report_sha256": candidate_sha,
        "final_report_sha256": final_report["report_sha256"],
        "global_ledger_snapshot_sha256": global_ledger_sha,
        "crash_evidence_set_sha256": crash_evidence_set_sha,
        "crash_receipt_sha256s": crash_hashes,
        "evidence_sha256s": evidence_hashes,
        "evidence_file_sha256s": evidence_file_hashes,
        "child_root_binding_sha256s": child_binding_hashes,
        "actual_postimage_sha256s": postimage_hashes,
        "controller_reread_postimage_sha256s": reread_postimage_hashes,
        "artifact_inventory_sha256s": inventory_hashes,
        "real_data_optimizer_updates": 0,
        "extra_model_forward_splits": 0,
        "pass_claim_available_before_stage_commit": False,
    }
    binding = {
        **binding_base,
        "binding_sha256": semantic_sha256(binding_base),
    }
    _exact_mapping(
        binding,
        G6_CONTROL_BINDING_FIELDS,
        "G6 control binding field set drift",
    )
    return _build_stage_bundle(
        _builder_issuer=_STAGE_BUNDLE_ISSUER,
        context=context,
        action=G6_ACTION,
        result_status="F1_COMPLETE",
        next_action=G7_ACTION,
        outputs={
            G6_OUTPUTS[0]: _canonical_file(final_report),
            G6_OUTPUTS[1]: _canonical_file(binding),
        },
        input_hashes={
            "g6_input_handle": g6_input_handle.capability_sha256,
            "candidate_f1_report": candidate_sha,
            "committed_global_ledger": global_ledger_sha,
            "crash_evidence_set": crash_evidence_set_sha,
            "actual_postimage_set": semantic_sha256(postimage_hashes),
        },
        declared_safety_counts={
            "protected_content_opens": 0,
            "protected_id_mapping_content_opens": 0,
            "historical_write_bytes": 0,
            "model_forward_evaluations": 0,
            "real_data_optimizer_updates": 0,
        },
        commit_eligible=True,
    )


def _g7_manifest_entry_from_committed(
    artifact: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "path": artifact["path"],
        "sha256": artifact["sha256"],
        "size_bytes": artifact["size_bytes"],
        "producer_command_sha256": artifact["producer_command_sha256"],
        "input_lineage": dict(artifact["input_lineage"]),
        "input_lineage_sha256": artifact["input_lineage_sha256"],
        "split": artifact["split"],
        "teacher_flags": dict(artifact["teacher_flags"]),
        "ground_truth_flags": dict(artifact["ground_truth_flags"]),
        "validity": "VALID_COMMITTED",
        "quarantined": False,
        "control_commit": {
            "status": "COMMITTED",
            "stage_action": artifact["stage_action"],
            "stage_commit_receipt_sha256": artifact[
                "stage_commit_receipt_sha256"
            ],
        },
    }


def _g7_manifest_entry_for_atomic_output(
    *,
    path: str,
    data: bytes,
    producer_command_sha256: str,
    input_lineage: Mapping[str, str],
    evaluation_labels_used: bool,
) -> dict[str, Any]:
    _relative_output_path(path)
    _sha256(producer_command_sha256, "G7 producer command SHA")
    lineage = dict(sorted(input_lineage.items()))
    for name, digest in lineage.items():
        _sha256(digest, "G7 input lineage %s" % name)
    return {
        "path": path,
        "sha256": bytes_sha256(data),
        "size_bytes": len(data),
        "producer_command_sha256": producer_command_sha256,
        "input_lineage": lineage,
        "input_lineage_sha256": bytes_sha256(
            canonical_json_bytes(lineage)
        ),
        "split": "FINALIZATION_CONTROL_NO_NEW_DATA_ACCESS",
        "teacher_flags": {
            "teacher_model_forward": False,
            "teacher_candidate_used": False,
            "teacher_shadow_value_fabricated": False,
        },
        "ground_truth_flags": {
            "evaluation_labels_used": evaluation_labels_used,
            "support_injected": False,
            "candidate_appended": False,
        },
        "validity": "VALIDATED_SAME_ATOMIC_G7_BUNDLE",
        "quarantined": False,
        "control_commit": {
            "status": "REQUIRES_SAME_ATOMIC_G7_STAGE_COMMIT",
            "stage_action": G7_ACTION,
            "stage_commit_receipt_sha256": None,
        },
    }


def build_g7_final_artifacts_bundle(
    *,
    context: StageContext,
    finalization_snapshot_capability: Any,
) -> StageBundle:
    """Build every pre-intent final artifact without opening new data."""

    context.validate(G7_ACTION, "F1_COMPLETE")
    from .a4_control import (
        A4FinalizationSnapshotCapability,
        validate_finalization_snapshot_capability,
    )

    _require(
        type(finalization_snapshot_capability)
        is A4FinalizationSnapshotCapability,
        "concrete A4FinalizationSnapshotCapability required",
    )
    try:
        snapshot = validate_finalization_snapshot_capability(
            finalization_snapshot_capability
        )
    except Exception as exc:
        raise StageContractError(
            "G7 concrete finalization snapshot validation failed"
        ) from exc
    _require(
        snapshot.get("goal_id") == context.goal_id
        and snapshot.get("attempt_id") == context.attempt_id
        and snapshot.get("committed_transaction_id")
        == context.control_transaction_id
        and snapshot.get("committed_state_sha256")
        == context.committed_control_state_sha256
        and snapshot.get("committed_event_sha256")
        == context.committed_control_event_sha256,
        "G7 snapshot/current post-G6 anchor mismatch",
    )
    final_inputs = dict(snapshot["final_artifact_inputs"])
    gates = dict(snapshot["gates_snapshot"])
    ledger = dict(snapshot["ledger_snapshot"])
    closed_authority = dict(snapshot["closed_authority"])
    routing = dict(final_inputs["routing_decision"])
    eval_entries = list(final_inputs["eval_ledger_entries"])
    access_summary = dict(final_inputs["access_summary"])
    committed_artifacts = list(final_inputs["committed_stage_artifacts"])
    _require(
        final_inputs.get("inputs_sha256")
        == semantic_sha256(
            final_inputs,
            excluded_fields=("inputs_sha256",),
        )
        and gates.get("snapshot_sha256")
        == semantic_sha256(gates, excluded_fields=("snapshot_sha256",))
        and ledger.get("snapshot_sha256")
        == semantic_sha256(ledger, excluded_fields=("snapshot_sha256",))
        and closed_authority.get("authority_sha256")
        == semantic_sha256(
            closed_authority,
            excluded_fields=("authority_sha256",),
        ),
        "G7 snapshot subrecord self-hash drift",
    )
    _require(
        type(closed_authority.get("id_mapping_authority_was_active"))
        is bool,
        "G7 closed authority mapping-active flag must be an exact boolean",
    )
    eval_rows = []
    for entry in eval_entries:
        row_base = {
            "schema_version": "c28f_a4_final_eval_ledger_row_v1",
            "status": "COMMITTED_NONREPLAYABLE_LOGICAL_EVAL",
            "goal_id": context.goal_id,
            "attempt_id": context.attempt_id,
            "token_id": entry["token_id"],
            "eval_id": entry["eval_id"],
            "logical_eval_count": entry["logical_eval_count"],
            "source_file_sha256": entry["source_file_sha256"],
            "source_ledger_entry": dict(entry),
        }
        eval_rows.append(
            {**row_base, "row_sha256": semantic_sha256(row_base)}
        )
    _require(
        [row["token_id"] for row in eval_rows] == ["F0_A", "F0_B"]
        and eval_rows[0]["eval_id"] != eval_rows[1]["eval_id"],
        "G7 final eval ledger token/eval closure",
    )
    eval_ledger_bytes = _canonical_jsonl(eval_rows)
    access_summary_bytes = _canonical_file(access_summary)
    from .a4_finalize import build_final_decision

    scientific_decision = build_final_decision(
        gates=gates["gates"],
        routing_status=routing["status"],
        eval_ledger_sha256=bytes_sha256(eval_ledger_bytes),
        access_summary_sha256=bytes_sha256(access_summary_bytes),
        f0_a_eval_id=str(eval_rows[0]["eval_id"]),
        f0_b_eval_id=str(eval_rows[1]["eval_id"]),
        model_forward_counts=ledger["model_forward_counts"],
        access_counts=access_summary["access_counts"],
        id_mapping_authority_was_active=closed_authority[
            "id_mapping_authority_was_active"
        ],
        real_data_optimizer_updates=ledger[
            "real_data_optimizer_updates"
        ],
        formal_policy_sha256=final_inputs["formal_policy_sha256"],
        role_lock_sha256=final_inputs["role_lock_sha256"],
    )
    decision_base = {
        "schema_version": "c28f_a4_f0_f1_window_decision_artifact_v1",
        "status": "F0_F1_WINDOW_READY_TO_STOP",
        "terminal_status": "F0_F1_WINDOW_COMPLETE_STOPPED",
        "goal_id": context.goal_id,
        "attempt_id": context.attempt_id,
        "scientific_decision": scientific_decision,
        "scientific_decision_sha256": scientific_decision[
            "decision_sha256"
        ],
        "gates_snapshot_sha256": gates["snapshot_sha256"],
        "routing_decision_sha256": routing["routing_decision_sha256"],
        "g6_final_report_sha256": snapshot["g6_final_report"][
            "report_sha256"
        ],
        "global_ledger_snapshot_sha256": ledger["snapshot_sha256"],
        "active_authority_sha256": snapshot["authority_snapshot"][
            "authority_sha256"
        ],
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
    decision_bytes = _canonical_file(decision)
    decision_markdown = _render_summary_markdown(
        title="C28F F0/F1 Window Decision",
        authoritative_content_sha256=bytes_sha256(decision_bytes),
        rows=(
            ("status", decision["status"]),
            ("terminal_status", decision["terminal_status"]),
            ("routing_status", routing["status"]),
            (
                "next_stage_recommendation_only",
                routing["next_stage_recommendation_only"],
            ),
            ("model_forward_evaluations", 2),
            ("real_data_optimizer_updates", 0),
            ("next_authority_granted", False),
        ),
    )
    if routing["status"] == "ROUTING_AMBIGUOUS":
        requested_first_step = "F2-R0_HELD_OUT_ROUTE_REPLICATION"
        request_reason = "ARBITRATE_SPLIT_DEPENDENT_EARLIEST_FAILURE_ONLY"
    else:
        requested_first_step = str(
            routing["next_stage_recommendation_only"]
        )
        _require(
            requested_first_step
            in {"F2-B", "F2-L", "F2-O", "F3_PROPOSAL_READINESS"},
            "G7 consistent routing did not select one frozen first step",
        )
        request_reason = "EXECUTE_ONLY_THE_FROZEN_EARLIEST_FAILURE_ROUTE"
    request_base = {
        "schema_version": "c28f_a4_next_authorization_request_v1",
        "status": "REQUESTED_NOT_GRANTED",
        "goal_id": context.goal_id,
        "attempt_id": context.attempt_id,
        "requested_authority": "AUTH_F2_F5_PROTOTYPE",
        "requested_first_step": requested_first_step,
        "request_reason": request_reason,
        "routing_status": routing["status"],
        "routing_decision_sha256": routing["routing_decision_sha256"],
        "decision_artifact_sha256": decision[
            "decision_artifact_sha256"
        ],
        "authority_granted": False,
        "creates_or_starts_next_goal": False,
        "model_forward_authorized_by_this_file": False,
        "optimizer_updates_authorized_by_this_file": False,
        "holdout_or_official_authorized_by_this_file": False,
    }
    authorization_request = {
        **request_base,
        "request_sha256": semantic_sha256(request_base),
    }
    authorization_request_bytes = _canonical_file(authorization_request)
    continuation_bytes = _render_summary_markdown(
        title="C28F v5 F0/F1 Continuation",
        authoritative_content_sha256=bytes_sha256(decision_bytes),
        rows=(
            ("goal_id", context.goal_id),
            ("attempt_id", context.attempt_id),
            ("status", "FINAL_ARTIFACTS_READY_AWAITING_G7_COMMIT"),
            ("routing_status", routing["status"]),
            ("requested_authority", "AUTH_F2_F5_PROTOTYPE"),
            ("requested_first_step", requested_first_step),
            ("requested_authority_granted", False),
            ("next_goal_started", False),
        ),
    )
    generated_outputs = {
        G7_OUTPUTS[0]: decision_bytes,
        G7_OUTPUTS[1]: decision_markdown,
        G7_OUTPUTS[4]: eval_ledger_bytes,
        G7_OUTPUTS[5]: access_summary_bytes,
        G7_OUTPUTS[6]: continuation_bytes,
        G7_OUTPUTS[7]: authorization_request_bytes,
    }
    producer_command = {
        "launcher": "blueprint_e2e_v2/c28f_v5/a4_runner.py",
        "python_flag": "-B",
        "action": G7_ACTION,
        "snapshot_capability_sha256": (
            finalization_snapshot_capability.capability_sha256
        ),
        "stage_context_binding_sha256": context.binding()[
            "state_binding_sha256"
        ],
    }
    producer_command_sha = bytes_sha256(
        canonical_json_bytes(producer_command)
    )
    generated_lineage = {
        "active_authority": snapshot["authority_snapshot"][
            "authority_sha256"
        ],
        "expected_closed_authority": closed_authority["authority_sha256"],
        "final_artifact_inputs": final_inputs["inputs_sha256"],
        "finalization_snapshot_capability": (
            finalization_snapshot_capability.capability_sha256
        ),
        "g6_final_report": snapshot["g6_final_report"]["report_sha256"],
        "gates_snapshot": gates["snapshot_sha256"],
        "global_ledger_snapshot": ledger["snapshot_sha256"],
    }
    manifest_entries = [
        _g7_manifest_entry_from_committed(artifact)
        for artifact in committed_artifacts
    ]
    label_derived_paths = {
        G7_OUTPUTS[0], G7_OUTPUTS[1], G7_OUTPUTS[4], G7_OUTPUTS[6],
        G7_OUTPUTS[7],
    }
    manifest_entries.extend(
        _g7_manifest_entry_for_atomic_output(
            path=path,
            data=data,
            producer_command_sha256=producer_command_sha,
            input_lineage=generated_lineage,
            evaluation_labels_used=path in label_derived_paths,
        )
        for path, data in generated_outputs.items()
    )
    manifest_entries = sorted(manifest_entries, key=lambda item: item["path"])
    _require(
        len({item["path"] for item in manifest_entries})
        == len(manifest_entries),
        "G7 final artifact manifest path collision",
    )
    exclusions = [G7_OUTPUTS[2], G7_OUTPUTS[3]]
    _require(
        set(generated_outputs) | set(exclusions) == set(G7_OUTPUTS)
        and set(exclusions).isdisjoint(
            {item["path"] for item in manifest_entries}
        ),
        "G7 manifest exclusions are not exactly self plus detached root",
    )
    manifest_base = {
        "schema_version": "c28f_a4_final_artifact_manifest_v1",
        "status": "VALIDATED_COMPLETE_ATOMIC_G7_BUNDLE",
        "goal_id": context.goal_id,
        "attempt_id": context.attempt_id,
        "entry_count": len(manifest_entries),
        "entries": manifest_entries,
        "entries_sha256": bytes_sha256(
            canonical_json_bytes(manifest_entries)
        ),
        "excluded_paths": exclusions,
        "excluded_paths_reason": (
            "MANIFEST_SELF_AND_DETACHED_ROOT_ONLY_NO_HASH_SELF_REFERENCE"
        ),
        "all_g1_through_g6_committed_artifacts_included": True,
        "all_other_g7_final_artifacts_included": True,
        "same_stage_g7_entries_require_atomic_stage_commit": True,
        "manifest_includes_itself": False,
        "manifest_includes_detached_root": False,
        "quarantined_entry_count": 0,
        "producer_command": producer_command,
        "producer_command_sha256": producer_command_sha,
        "input_lineage": dict(sorted(generated_lineage.items())),
        "input_lineage_sha256": bytes_sha256(
            canonical_json_bytes(dict(sorted(generated_lineage.items())))
        ),
    }
    manifest = {
        **manifest_base,
        "manifest_sha256": semantic_sha256(manifest_base),
    }
    manifest_bytes = _canonical_file(manifest)
    detached_root_bytes = (
        bytes_sha256(manifest_bytes) + "\n"
    ).encode("ascii")
    outputs = {
        **generated_outputs,
        G7_OUTPUTS[2]: manifest_bytes,
        G7_OUTPUTS[3]: detached_root_bytes,
    }
    return _build_stage_bundle(
        _builder_issuer=_STAGE_BUNDLE_ISSUER,
        context=context,
        action=G7_ACTION,
        result_status="FINAL_ARTIFACTS_READY",
        next_action=G7_COMMIT_ACTION,
        outputs=outputs,
        input_hashes={
            "finalization_snapshot_capability": (
                finalization_snapshot_capability.capability_sha256
            ),
            "final_artifact_inputs": final_inputs["inputs_sha256"],
            "gates_snapshot": gates["snapshot_sha256"],
            "global_ledger_snapshot": ledger["snapshot_sha256"],
            "active_authority": snapshot["authority_snapshot"][
                "authority_sha256"
            ],
            "expected_closed_authority": closed_authority[
                "authority_sha256"
            ],
            "manifest": manifest["manifest_sha256"],
        },
        declared_safety_counts={
            "protected_content_opens": 0,
            "protected_id_mapping_content_opens": 0,
            "historical_write_bytes": 0,
            "model_forward_evaluations": 0,
            "real_data_optimizer_updates": 0,
        },
        commit_eligible=True,
    )


__all__ = [
    "F0_COMPLETED_EVAL_SAFETY_COUNTS",
    "F0_A_COMMITTED_REPORT_BINDING_FIELDS",
    "F0_A_FORENSICS_REPORT_FIELDS",
    "F0_A_FORENSICS_REPORT_SCHEMA",
    "F0_B_FORENSICS_REPORT_FIELDS",
    "F0_B_FORENSICS_REPORT_SCHEMA",
    "F0_COMPLETED_EVAL_SOURCE_ARTIFACT_LINEAGE_FIELDS",
    "F0_COMPLETED_EVAL_SOURCE_ARTIFACT_LINEAGE_SCHEMA",
    "F0_DERIVED_NMS_ARTIFACT_CHUNK_FIELDS",
    "F0_DERIVED_NMS_ARTIFACT_INDEX_FIELDS",
    "F0_DERIVED_NMS_ARTIFACT_INDEX_SCHEMA",
    "F0_RAW_FORWARD_ARTIFACT_CHUNK_FIELDS",
    "F0_RAW_FORWARD_ARTIFACT_INDEX_FIELDS",
    "F0_RAW_FORWARD_ARTIFACT_INDEX_SCHEMA",
    "F0_ROUTING_DECISION_FIELDS",
    "F0_SPLIT_SHIFT_REPORT_FIELDS",
    "FROZEN_LEGACY_SEMANTIC_SOURCE_HASHES",
    "G1_ACTION",
    "G1_NEGATIVE_SUITE_MANIFEST",
    "G1_NEGATIVE_SUITE_SHA256",
    "G2_ACTION",
    "G2_OUTPUTS",
    "G2_TRAIN_JSONL_INDEX_BOOTSTRAP_ACTION",
    "G3_ACTION",
    "G3_OUTPUTS",
    "G4_ACTION",
    "G4_ATOMIC_BUNDLE_BINDING_FIELDS",
    "G4_ATOMIC_BUNDLE_BINDING_SCHEMA",
    "G4_ATOMIC_CANDIDATE_FIELDS",
    "G4_CLUSTER_POWER_AUDIT_FIELDS",
    "G4_POWER_EXPANSION_REPORT_FIELDS",
    "G4_POWER_EVIDENCE_BINDING_FIELDS",
    "G4_PROJECTION_ACTION",
    "G4_PROJECTION_COMPLETE_ACTION",
    "G4_TRUSTED_POWER_COMPUTATION_RECEIPT_FIELDS",
    "G4_TRUSTED_POWER_COMPUTATION_RECEIPT_SCHEMA",
    "G4_ROLE_ASSIGNMENT_FIELDS",
    "G5_ACTION",
    "G5_OUTPUTS",
    "G6_ACTION",
    "G6_ACTUAL_POSTIMAGE_FIELDS",
    "G6_ACTUAL_POSTIMAGE_INVENTORY_ROW_FIELDS",
    "G6_ACTUAL_POSTIMAGE_ROOT_IDENTITY_FIELDS",
    "G6_ACTUAL_POSTIMAGE_SCHEMA",
    "G6_ATOMIC_CRASH_EVIDENCE_FIELDS",
    "G6_CONTROL_BINDING_FIELDS",
    "G6_CONTROL_BINDING_SCHEMA",
    "G6_CONTROL_EVIDENCE_BINDING_FIELDS",
    "G6_CRASH_RECEIPT_FIELDS",
    "G6_CRASH_RECEIPT_SCHEMA",
    "G6_GLOBAL_LEDGER_FIELDS",
    "G6_GLOBAL_LEDGER_SCHEMA",
    "G7_ACTION",
    "G7_COMMIT_ACTION",
    "G7_OUTPUTS",
    "POWER_OUTCOME_KEYS",
    "REQUIRED_G1_NEGATIVE_PROBES",
    "ReplayExecution",
    "StageBundle",
    "StageContext",
    "StageContractError",
    "build_g1_bundle",
    "build_g2_bundle",
    "build_g3_bundle",
    "build_g4_bundle",
    "build_g4_projection_bundle",
    "build_g5_bundle",
    "build_g6_f1_completion_bundle",
    "build_g7_final_artifacts_bundle",
    "execute_ledgered_c7_replay_alignment",
    "submit_stage_bundle",
    "validate_stage_bundle",
]
