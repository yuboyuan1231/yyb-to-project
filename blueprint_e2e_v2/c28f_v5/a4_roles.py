"""Pure role assignment/policy projections consumed by concrete G4 control.

This module has no token issuer, file opener, persistence claim, or authority
transition.  ``build_g4_role_policy_candidates`` is the only stage-facing
candidate builder and consumes one exact G4 input handle plus one exact G2 power
replay capability.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Optional, Sequence

from .canonical import canonical_json_bytes, semantic_sha256


ROLE_SCHEMA = "c28f_role_manifest_v1"
ROLE_LOCK_SCHEMA = "c28f_role_manifest_lock_v1"
FORMAL_POLICY_SCHEMA = "c28f_formal_data_policy_v1"
CLUSTER_POWER_AUDIT_SCHEMA = "c28f_role_cluster_power_audit_v1"
ASSIGNMENT_ALGORITHM = "seeded_video_cluster_sequential_fill_v1"
ASSIGNMENT_SEED = 2026
FORMAL_POLICY = "STRICT_CORE_ONLY"
U_FORMAL_UPDATES = 17_360
CORPUS_VIDEO_COUNT = 17_435
CORPUS_VIDEO_ID_ORDER_SHA256 = (
    "51ebc37675f9d822fdd5ce1d92608bc7f442893d89451e2eb301f2d3d640676e"
)

POWER_REPLAY_ROW_SCHEMA = "c28f_g2_power_replay_row_v1"
POWER_EVALUATOR_CONTRACT_SCHEMA = "c28f_g2_power_evaluator_contract_v2"
POWER_ALGORITHM = (
    "HISTORICAL_REFERENCE_PAIRED_CLUSTER_POSITION_BOOTSTRAP_10000_FIXED_SEED_V3"
)
POWER_COUNTER_DOMAIN = "C28F_G2_HISTORICAL_REFERENCE_POWER_V3"
POWER_BOOTSTRAP_REPLICATES = 10_000
POWER_OUTER_EXPERIMENTS = 200
POWER_INNER_RESAMPLES = 49
POWER_LOWER_NEAREST_RANK = 2
POWER_UPPER_NEAREST_RANK = 48
POWER_TARGET = 0.80
POWER_MAX_INJECTED_DELTA = 0.10
POWER_OUTCOME_KEYS = (
    "CORRECT_VIDEO_WRONG_SPAN_TOP1",
    "LEGACY_UNIQUE_VIDEO_TRUE_VR_R100",
    "VCMR_R10_IOU_0_7",
    "VCMR_R1_IOU_0_7",
    "VCMR_R5_IOU_0_7",
    "WRONG_VIDEO_TOP1",
)
POWER_ADAPTIVE_ASSIGNMENT_ALGORITHM = (
    "seeded_disjoint_role_lanes_preregistered_power_length_before_manifest_lock_v5"
)
POWER_EXPANSION_LANE_ALGORITHM = (
    "seeded_core_order_reserve_last_even_mechanism_odd_confirm_v1"
)
G4_ATOMIC_CANDIDATE_SCHEMA = "c28f_g4_role_policy_atomic_candidates_v1"

MINIMUM_EFFECTS_REGISTRY_SHA256 = (
    "a7df1eee2cc8f8b556a2765cc764d97abc78d5251b7ca34a75a2bcc70c62c098"
)
EFFECT_OUTCOME_REGISTRY_SHA256 = (
    "5390ca1db8126865347aa51baa00bebf5b9c14f06ccc5e10903babb6a538ef55"
)


def _literal_effect_registries() -> tuple[dict[str, float], dict[str, dict[str, Any]]]:
    minimum_effects = {
        "core_primary_r1": 0.0025,
        "core_secondary_r5_or_r10": 0.0050,
        "core_wrong_video": 0.0050,
        "r2m_primary_r1": 0.0020,
        "m2r_tier_a_true_vr": 0.0200,
        "m2r_tier_b_true_vr": 0.0100,
        "mediator_primary_r1": 0.0020,
        "credit_primary_r1": 0.0025,
    }
    effect_outcomes = {
        "core_primary_r1": {
            "binary_outcome": "VCMR_R1_IOU_0_7",
            "sensitivity_rule": "PAIRED_GT_VIDEO_CLUSTER_ABSOLUTE_RECALL_DELTA",
            "eligible_baseline_value": 0,
            "injected_flip": "0_TO_1",
        },
        "core_secondary_r5_or_r10": {
            "binary_outcome": "VCMR_R10_IOU_0_7",
            "sensitivity_rule": "PAIRED_GT_VIDEO_CLUSTER_ABSOLUTE_RECALL_DELTA",
            "eligible_baseline_value": 0,
            "injected_flip": "0_TO_1",
        },
        "core_wrong_video": {
            "binary_outcome": "WRONG_VIDEO_TOP1",
            "sensitivity_rule": "PAIRED_GT_VIDEO_CLUSTER_ABSOLUTE_ERROR_DELTA",
            "eligible_baseline_value": 1,
            "injected_flip": "1_TO_0",
        },
        "r2m_primary_r1": {
            "binary_outcome": "VCMR_R1_IOU_0_7",
            "sensitivity_rule": "PAIRED_GT_VIDEO_CLUSTER_ABSOLUTE_RECALL_DELTA",
            "eligible_baseline_value": 0,
            "injected_flip": "0_TO_1",
        },
        "m2r_tier_a_true_vr": {
            "binary_outcome": "LEGACY_UNIQUE_VIDEO_TRUE_VR_R100",
            "sensitivity_rule": "PAIRED_GT_VIDEO_CLUSTER_ABSOLUTE_RECALL_DELTA",
            "eligible_baseline_value": 0,
            "injected_flip": "0_TO_1",
        },
        "m2r_tier_b_true_vr": {
            "binary_outcome": "LEGACY_UNIQUE_VIDEO_TRUE_VR_R100",
            "sensitivity_rule": "PAIRED_GT_VIDEO_CLUSTER_ABSOLUTE_RECALL_DELTA",
            "eligible_baseline_value": 0,
            "injected_flip": "0_TO_1",
        },
        "mediator_primary_r1": {
            "binary_outcome": "VCMR_R1_IOU_0_7",
            "sensitivity_rule": "PAIRED_GT_VIDEO_CLUSTER_ABSOLUTE_RECALL_DELTA",
            "eligible_baseline_value": 0,
            "injected_flip": "0_TO_1",
        },
        "credit_primary_r1": {
            "binary_outcome": "VCMR_R1_IOU_0_7",
            "sensitivity_rule": "PAIRED_GT_VIDEO_CLUSTER_ABSOLUTE_RECALL_DELTA",
            "eligible_baseline_value": 0,
            "injected_flip": "0_TO_1",
        },
    }
    return minimum_effects, effect_outcomes


_INITIAL_MINIMUM_EFFECTS, _INITIAL_EFFECT_OUTCOMES = _literal_effect_registries()
MINIMUM_EFFECTS_RATIO = MappingProxyType(_INITIAL_MINIMUM_EFFECTS)
EFFECT_OUTCOME_REGISTRY = MappingProxyType(
    {
        key: MappingProxyType(value)
        for key, value in _INITIAL_EFFECT_OUTCOMES.items()
    }
)
del _INITIAL_MINIMUM_EFFECTS, _INITIAL_EFFECT_OUTCOMES

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_VIDEO_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")

ROLE_MIN_QUERY_COUNTS: tuple[tuple[str, int], ...] = (
    ("train_fit_route_dev", 2_048),
    ("train_fit_mechanism_dev", 4_096),
    ("train_fit_confirm", 2_048),
    ("train_fit_calibration", 2_048),
)
CORE_ROLE = "train_fit_core"


class RoleContractError(RuntimeError):
    pass


def _validated_effect_registries() -> tuple[
    dict[str, float],
    dict[str, dict[str, Any]],
]:
    """Rebuild and independently verify both pre-registered literal registries."""

    minimum_effects, effect_outcomes = _literal_effect_registries()
    if (
        semantic_sha256(minimum_effects)
        != MINIMUM_EFFECTS_REGISTRY_SHA256
        or semantic_sha256(effect_outcomes)
        != EFFECT_OUTCOME_REGISTRY_SHA256
    ):
        raise RoleContractError("literal power registry SHA drift")
    try:
        observed_minimum_effects = dict(MINIMUM_EFFECTS_RATIO)
        observed_effect_outcomes = {
            key: dict(value)
            for key, value in EFFECT_OUTCOME_REGISTRY.items()
        }
    except (AttributeError, TypeError, ValueError) as error:
        raise RoleContractError(
            "public power registry was replaced"
        ) from error
    if (
        canonical_json_bytes(observed_minimum_effects)
        != canonical_json_bytes(minimum_effects)
        or semantic_sha256(observed_minimum_effects)
        != MINIMUM_EFFECTS_REGISTRY_SHA256
        or canonical_json_bytes(observed_effect_outcomes)
        != canonical_json_bytes(effect_outcomes)
        or semantic_sha256(observed_effect_outcomes)
        != EFFECT_OUTCOME_REGISTRY_SHA256
    ):
        raise RoleContractError("public power registry literal/SHA drift")
    return minimum_effects, effect_outcomes


def _effect_threshold_reduction_count(
    role_results: Mapping[str, Any],
    minimum_effects: Mapping[str, float],
) -> int:
    reductions = 0
    for role in ("train_fit_mechanism_dev", "train_fit_confirm"):
        result = role_results.get(role)
        effects = result.get("effect_results") if isinstance(result, Mapping) else None
        if not isinstance(effects, Mapping) or set(effects) != set(minimum_effects):
            raise RoleContractError("power threshold comparison registry is not exact")
        for effect_key, registered in minimum_effects.items():
            effect = effects.get(effect_key)
            observed = (
                effect.get("minimum_effect_ratio")
                if isinstance(effect, Mapping)
                else None
            )
            if (
                isinstance(observed, bool)
                or not isinstance(observed, (int, float))
                or not math.isfinite(float(observed))
            ):
                raise RoleContractError("power threshold comparison value is invalid")
            if float(observed) < float(registered):
                reductions += 1
    return reductions


def _exact_sha256(value: Any, field: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise RoleContractError(f"{field} must be an exact lowercase SHA-256")
    return value


def _exact_video_id(value: Any, field: str) -> str:
    if not isinstance(value, str) or _VIDEO_ID_RE.fullmatch(value) is None:
        raise RoleContractError(f"{field} is not a canonical video ID")
    return value


def frozen_power_evaluator_contract() -> dict[str, Any]:
    """Return the single role-independent G2 power contract consumed by G4."""

    minimum_effects, effect_outcomes = _validated_effect_registries()
    derivation_base = {
        "schema_version": "c28f_g2_power_outcome_derivation_contract_v1",
        "input_semantics": "SEALED_FROZEN_METRIC_PROPOSALS_AND_AUTHORIZED_GT_ONLY_INSIDE_G2",
        "persisted_row_semantics": "SIX_BINARY_SUFFICIENT_STATISTICS_NO_PROPOSALS_NO_GT_SPAN",
        "proposal_row_schema": ["video_id", "start_sec", "end_sec", "score"],
        "proposal_order": "SCORE_DESC_STABLE_FROZEN_METRIC_NMS_ORDER",
        "nms_contract": {
            "scope": "PER_VIDEO",
            "iou_threshold": 0.7,
            "maximum_results": 100,
            "tie_break": "FROZEN_EVALUATOR_STABLE_ORDER",
        },
        "temporal_iou_contract": {
            "intersection": "max(0,min(end_a,end_b)-max(start_a,start_b))",
            "union": "max(end_a,end_b)-min(start_a,start_b)",
            "zero_union_result": 0.0,
        },
        "outcome_key_order": list(POWER_OUTCOME_KEYS),
        "outcome_value_contract": "EXACT_INT_0_OR_1_BOOL_FORBIDDEN",
        "outcome_rules": {
            "VCMR_R1_IOU_0_7": "ANY_TOP1_GT_VIDEO_AND_TEMPORAL_IOU_GE_0.7",
            "VCMR_R5_IOU_0_7": "ANY_TOP5_GT_VIDEO_AND_TEMPORAL_IOU_GE_0.7",
            "VCMR_R10_IOU_0_7": "ANY_TOP10_GT_VIDEO_AND_TEMPORAL_IOU_GE_0.7",
            "LEGACY_UNIQUE_VIDEO_TRUE_VR_R100": "GT_VIDEO_IN_FIRST_100_UNIQUE_VIDEO_IDS",
            "WRONG_VIDEO_TOP1": "TOP1_VIDEO_ID_NE_GT_VIDEO_ID",
            "CORRECT_VIDEO_WRONG_SPAN_TOP1": "TOP1_VIDEO_ID_EQ_GT_VIDEO_ID_AND_TEMPORAL_IOU_LT_0.5",
        },
    }
    derivation_contract = {
        **derivation_base,
        "outcome_derivation_contract_sha256": semantic_sha256(derivation_base),
    }
    base = {
        "schema_version": POWER_EVALUATOR_CONTRACT_SCHEMA,
        "status": "FROZEN_ROLE_INDEPENDENT_EVALUATOR_CONTRACT",
        "algorithm": POWER_ALGORITHM,
        "bootstrap_replicates": POWER_BOOTSTRAP_REPLICATES,
        "bootstrap_seed": ASSIGNMENT_SEED,
        "minimum_effects_ratio": minimum_effects,
        "minimum_effect_registry_sha256": MINIMUM_EFFECTS_REGISTRY_SHA256,
        "power_row_schema": {
            "query_entity": "independent_historical_reference_desc_id:int",
            "ground_truth_video": "historical_reference_gt_video_id:utf8",
            "outcomes": "EXACT_SIX_KEY_BINARY_MAPPING",
            "row_fields": [
                "schema_version",
                "desc_id",
                "gt_video_id",
                "outcomes",
                "replay_packet_sha256",
                "query_manifest_sha256",
            ],
            "query_manifest_binding_required": True,
            "reference_population_semantics": (
                "COMMITTED_C7_CALIBRATED_A4_HISTORICAL_REFERENCE_ONLY"
            ),
            "train_fit_desc_id_or_video_id_join": "FORBIDDEN",
        },
        "observable_effects": list(POWER_OUTCOME_KEYS),
        "outcome_derivation_contract": derivation_contract,
        "outcome_derivation_contract_sha256": derivation_contract[
            "outcome_derivation_contract_sha256"
        ],
        "effect_outcome_registry": effect_outcomes,
        "effect_outcome_registry_sha256": EFFECT_OUTCOME_REGISTRY_SHA256,
        "bootstrap_test_contract": {
            "resampling_unit": "OUTER_CLUSTER_POSITION_WITH_ALL_ASSIGNED_QUERY_SLOTS",
            "paired_baseline_candidate_slots_preserved": True,
            "alpha_two_sided": 0.05,
            "target_power": POWER_TARGET,
            "delta_grid": "k/N_FOR_k_1_TO_CEIL_0.10N",
            "maximum_injected_delta_ratio": POWER_MAX_INJECTED_DELTA,
            "outer_candidate_projection": {
                "cluster_position_count": (
                    "ACTUAL_CANDIDATE_ROLE_GT_VIDEO_CLUSTER_COUNT_C"
                ),
                "cluster_position_sampling": (
                    "C_DRAWS_WITH_REPLACEMENT_FROM_COMMITTED_REFERENCE_CLUSTERS"
                ),
                "query_slot_count": "ACTUAL_CANDIDATE_ROLE_QUERY_COUNT_N",
                "query_slot_allocation": (
                    "q=floor(N/C);r=N_mod_C;POSITIONS_0_TO_r_MINUS_1_GET_q_PLUS_1_"
                    "SLOTS;REMAINING_POSITIONS_GET_q_SLOTS"
                ),
                "within_position_reference_row_sampling": (
                    "WITH_REPLACEMENT_BY_FIXED_COUNTER_HASH"
                ),
                "cluster_unit_preserved": True,
                "projection_assumption": (
                    "EQUAL_q_OR_q_PLUS_1_SLOT_PROJECTION_ONLY_NOT_OBSERVED_"
                    "CANDIDATE_CLUSTER_SIZE_DISTRIBUTION"
                ),
            },
            "flip_selection": {
                "eligible_set": (
                    "OUTER_LOCAL_SLOT_OUTCOME_EQUALS_EFFECT_REGISTRY_"
                    "ELIGIBLE_BASELINE_VALUE"
                ),
                "within_cluster_position_order": (
                    "SHA256_CANONICAL_JSON_[domain,seed,role,outer,effect_key,"
                    "position,slot,reference_desc_id]"
                ),
                "cross_position_dispersion_order": (
                    "WITHIN_POSITION_ELIGIBLE_RANK_THEN_POSITION_HASH_THEN_SLOT_HASH"
                ),
                "selected_count": "k_EXACT_NO_REPLACEMENT",
                "nested_prefix": True,
                "insufficient_outer_eligible_slots": "OUTER_NOT_DETECTED_AT_k",
            },
            "paired_delta_contract": {
                "baseline_candidate_pairing": "SAME_OUTER_QUERY_SLOT",
                "recall_flip": "0_TO_1_RAW_DELTA_PLUS_1",
                "error_flip": "1_TO_0_RAW_DELTA_MINUS_1_SIGN_MULTIPLIED_BY_MINUS_1",
                "sign_normalized_improvement": (
                    "RECALL_OR_ERROR_IMPROVEMENT_ALWAYS_PLUS_1_PER_FLIPPED_SLOT"
                ),
                "unflipped_slot_delta": 0,
                "inner_statistic": (
                    "SUM_SIGN_NORMALIZED_PAIRED_DELTAS_DIV_TOTAL_RESAMPLED_SLOTS"
                ),
            },
            "deterministic_resampling": {
                "outer_experiments": POWER_OUTER_EXPERIMENTS,
                "inner_bootstrap_resamples_per_outer": POWER_INNER_RESAMPLES,
                "total_outer_plus_inner_resample_rows_per_role": (
                    POWER_BOOTSTRAP_REPLICATES
                ),
                "matrix_reused_across_effects_and_k": True,
                "index_stream": "SHA256_COUNTER_MODE_REJECTION_SAMPLING_V1",
                "index_stream_domain": POWER_COUNTER_DOMAIN,
                "counter_preimage_fields": [
                    "domain",
                    "seed",
                    "role",
                    "outer_index",
                    "inner_index_or_negative_sampling_phase",
                    "draw_index",
                    "rejection_counter",
                ],
                "uniform_index_rule": (
                    "u256_ACCEPT_IF_u_LT_FLOOR_2^256_DIV_C_TIMES_C_THEN_u_MOD_C"
                ),
                "canonical_cluster_order": "gt_video_id_UTF8_BYTE_ORDER",
                "cluster_draw_count": (
                    "CANDIDATE_ROLE_ACTUAL_GT_VIDEO_CLUSTER_COUNT"
                ),
                "outer_sampling_frame": (
                    "ALL_COMMITTED_G2_HISTORICAL_REFERENCE_GT_VIDEO_CLUSTERS"
                ),
                "inner_sampling_frame": (
                    "OUTER_CLUSTER_MULTISET_POSITION_INDICES_0_TO_C_MINUS_1"
                ),
                "inner_mapping_rule": (
                    "C_POSITION_DRAWS_WITH_REPLACEMENT;DUPLICATE_POSITIONS_"
                    "PRESERVED;EACH_DRAW_CARRIES_ALL_SLOTS_ASSIGNED_TO_THAT_POSITION"
                ),
                "inner_position_deduplication": "FORBIDDEN",
            },
            "percentile_interval": {
                "inner_sample_count": POWER_INNER_RESAMPLES,
                "lower_nearest_rank_1_INDEXED": POWER_LOWER_NEAREST_RANK,
                "upper_nearest_rank_1_INDEXED": POWER_UPPER_NEAREST_RANK,
                "detection_rule": (
                    "SIGN_NORMALIZED_TWO_SIDED_95_PERCENTILE_INTERVAL_LOWER_GT_ZERO"
                ),
                "outer_first_k_evidence": (
                    "DETECTED_THRESHOLD_STORES_CI_AT_k_AND_k_MINUS_1_WITH_"
                    "LOWER_k_GT_ZERO_AND_LOWER_k_MINUS_1_LE_ZERO"
                ),
                "outer_undetected_evidence": (
                    "STORES_CI_AT_OUTER_MAXIMUM_EVALUATED_k_EQUALS_MIN_"
                    "CEIL_0.10N_AND_OUTER_ELIGIBLE_COUNT_WITH_LOWER_LE_ZERO"
                ),
            },
            "power_estimate": "DETECTED_OUTER_EXPERIMENTS_DIVIDED_BY_200",
            "nested_flip_sets": (
                "SELECTED_k_IS_STRICT_PREFIX_OF_SELECTED_k_PLUS_1"
            ),
            "monotonicity_invariant": (
                "SHARED_INNER_POSITION_MULTIPLICITIES_AND_NESTED_OUTER_LOCAL_"
                "FLIPS_MAKE_EACH_INNER_PAIRED_DELTA_AND_POWER_NONDECREASING_IN_k"
            ),
            "mde_search": (
                "FIRST_INTEGER_k_OR_EQUIVALENT_OUTER_DETECTION_THRESHOLDS_WITH_"
                "POWER_AT_LEAST_0.80_THEN_EXPLICITLY_VERIFY_k_MINUS_1_BELOW_0.80"
            ),
            "detectable_effect_definition": (
                "FIRST_SUPPORTED_k/N_WITH_POWER_AT_LEAST_0.80"
            ),
            "pass_rule": (
                "MDE_NOT_GREATER_THAN_EFFECT_REGISTRY_MINIMUM_EFFECT_RATIO"
            ),
        },
        "candidate_role_projection_contract": {
            "inputs": [
                "actual_candidate_role_query_count",
                "actual_candidate_role_gt_video_cluster_count",
            ],
            "reference_inputs": [
                "committed_power_rows_sha256",
                "committed_power_row_count",
                "historical_reference_gt_video_cluster_count",
                "historical_reference_outcome_distribution",
            ],
            "reference_cluster_draws": (
                "OUTER_WITH_REPLACEMENT_TO_CANDIDATE_ACTUAL_CLUSTER_COUNT_C"
            ),
            "candidate_query_projection": (
                "EXACT_N_EQUAL_q_OR_q_PLUS_1_SLOTS_ACROSS_C_OUTER_POSITIONS"
            ),
            "candidate_membership_fields_read_by_power": (
                "COUNTS_ONLY_NO_DESC_ID_NO_VIDEO_ID_NO_OUTCOME_JOIN"
            ),
            "historical_outcomes_select_candidate_identity_order": False,
            "historical_outcomes_select_within_prefix_members": False,
            "cross_role_historical_outcomes_affect_lane_identity_order_or_members": (
                False
            ),
            "historical_outcomes_may_determine_preregistered_prefix_length": True,
            "prefix_identity_order": (
                "FROZEN_SEEDED_DISJOINT_ROLE_SPECIFIC_CORE_CLUSTER_LANES"
            ),
            "expansion_lane_algorithm": POWER_EXPANSION_LANE_ALGORITHM,
            "membership_lock_timing": "ALL_PREFIX_EXPANSION_BEFORE_ROLE_MANIFEST_LOCK",
        },
        "role_assignment_dependent_result_precomputed": False,
        "historical_reference_membership_join_performed": False,
        "candidate_counts_are_power_inputs_not_standalone_pass": True,
    }
    return {**base, "contract_sha256": semantic_sha256(base)}


@dataclass(frozen=True)
class RoleRecord:
    desc_id: int
    gt_video_id: str

    def __post_init__(self) -> None:
        if type(self.desc_id) is not int or self.desc_id < 0:
            raise RoleContractError("desc_id must be a non-negative exact integer")
        _exact_video_id(self.gt_video_id, "gt_video_id")


@dataclass(frozen=True)
class _PowerReplayRecord:
    desc_id: int
    gt_video_id: str
    outcomes: Mapping[str, int]


def _normalize_power_replay_rows(
    rows: Any,
    *,
    query_manifest_sha256: str,
    replay_packet_sha256: str,
) -> dict[int, _PowerReplayRecord]:
    _validated_effect_registries()
    _exact_sha256(query_manifest_sha256, "power query manifest SHA")
    _exact_sha256(replay_packet_sha256, "power replay packet SHA")
    if not isinstance(rows, (list, tuple)) or not rows:
        raise RoleContractError("power replay rows must be a non-empty explicit sequence")
    normalized: dict[int, _PowerReplayRecord] = {}
    observed_order: list[int] = []
    exact_fields = {
        "schema_version",
        "desc_id",
        "gt_video_id",
        "outcomes",
        "replay_packet_sha256",
        "query_manifest_sha256",
    }
    for index, raw in enumerate(rows):
        if not isinstance(raw, Mapping) or set(raw) != exact_fields:
            raise RoleContractError(f"power replay row field set is not exact: {index}")
        if raw.get("schema_version") != POWER_REPLAY_ROW_SCHEMA:
            raise RoleContractError(f"power replay row schema mismatch: {index}")
        desc_id = raw.get("desc_id")
        if type(desc_id) is not int or desc_id < 0 or desc_id in normalized:
            raise RoleContractError(f"power replay desc_id is invalid or duplicate: {index}")
        gt_video_id = _exact_video_id(
            raw.get("gt_video_id"), f"power replay row[{index}].gt_video_id"
        )
        if raw.get("query_manifest_sha256") != query_manifest_sha256:
            raise RoleContractError("power replay query-manifest binding drift")
        if raw.get("replay_packet_sha256") != replay_packet_sha256:
            raise RoleContractError("power replay packet binding drift")
        outcomes = raw.get("outcomes")
        if (
            not isinstance(outcomes, Mapping)
            or tuple(outcomes) != POWER_OUTCOME_KEYS
            or any(type(value) is not int or value not in {0, 1} for value in outcomes.values())
        ):
            raise RoleContractError(f"power replay outcomes are not exact: {index}")
        normalized[desc_id] = _PowerReplayRecord(
            desc_id=desc_id,
            gt_video_id=gt_video_id,
            outcomes=dict(outcomes),
        )
        observed_order.append(desc_id)
    if observed_order != sorted(observed_order):
        raise RoleContractError("power replay rows are not canonical by desc_id")
    return normalized


def _sha_lines(values: Iterable[str]) -> str:
    payload = "".join(f"{value}\n" for value in values).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def validate_historical_power_reference_population(value: Any) -> dict[str, Any]:
    """Validate and detach the committed G2 historical reference manifest."""

    if not isinstance(value, Mapping):
        raise RoleContractError("historical power reference manifest must be a mapping")
    manifest = json.loads(canonical_json_bytes(dict(value)).decode("utf-8"))
    exact_fields = {
        "schema_version",
        "population_semantics",
        "power_rows_sha256",
        "power_row_count",
        "replay_packet_sha256",
        "query_manifest_sha256",
        "source_registry_sha256",
        "reference_desc_id_lines_sha256",
        "reference_gt_video_cluster_count",
        "reference_gt_video_id_lines_sha256",
        "reference_outcome_distribution",
        "normalized_reference_sha256",
        "train_fit_desc_id_join_performed",
        "train_fit_video_id_join_performed",
        "candidate_membership_fields_consumed",
        "historical_performance_selects_membership",
        "reference_population_sha256",
    }
    if (
        set(manifest) != exact_fields
        or manifest.get("schema_version")
        != "c28f_g2_historical_power_reference_v1"
        or manifest.get("population_semantics")
        != "INDEPENDENT_C7_CALIBRATED_A4_HISTORICAL_REFERENCE_ONLY"
        or type(manifest.get("power_row_count")) is not int
        or manifest["power_row_count"] <= 0
        or type(manifest.get("reference_gt_video_cluster_count")) is not int
        or not 2
        <= manifest["reference_gt_video_cluster_count"]
        <= manifest["power_row_count"]
        or manifest.get("train_fit_desc_id_join_performed") is not False
        or manifest.get("train_fit_video_id_join_performed") is not False
        or manifest.get("historical_performance_selects_membership") is not False
        or manifest.get("candidate_membership_fields_consumed")
        != [
            "actual_candidate_role_query_count",
            "actual_candidate_role_gt_video_cluster_count",
        ]
        or manifest.get("reference_population_sha256")
        != semantic_sha256(
            manifest,
            excluded_fields=("reference_population_sha256",),
        )
    ):
        raise RoleContractError("historical power reference manifest drift")
    for field in (
        "power_rows_sha256",
        "replay_packet_sha256",
        "query_manifest_sha256",
        "source_registry_sha256",
        "reference_desc_id_lines_sha256",
        "reference_gt_video_id_lines_sha256",
        "normalized_reference_sha256",
        "reference_population_sha256",
    ):
        _exact_sha256(manifest.get(field), field)
    distribution = manifest.get("reference_outcome_distribution")
    if not isinstance(distribution, Mapping) or tuple(distribution) != POWER_OUTCOME_KEYS:
        raise RoleContractError("historical reference outcome distribution drift")
    for outcome, counts in distribution.items():
        if (
            not isinstance(counts, Mapping)
            or set(counts) != {"zero_count", "one_count"}
            or type(counts.get("zero_count")) is not int
            or type(counts.get("one_count")) is not int
            or counts["zero_count"] < 0
            or counts["one_count"] < 0
            or counts["zero_count"] + counts["one_count"]
            != manifest["power_row_count"]
        ):
            raise RoleContractError(
                f"historical reference outcome counts drift: {outcome}"
            )
    return manifest


def _build_historical_power_reference_population(
    reference_by_desc: Mapping[int, _PowerReplayRecord],
    *,
    power_rows_sha256: str,
    power_row_count: int,
    replay_packet_sha256: str,
    query_manifest_sha256: str,
    source_registry_sha256: str,
) -> dict[str, Any]:
    """Seal the independent G2 reference distribution used only for power."""

    for value, field in (
        (power_rows_sha256, "power rows SHA"),
        (replay_packet_sha256, "replay packet SHA"),
        (query_manifest_sha256, "query manifest SHA"),
        (source_registry_sha256, "source registry SHA"),
    ):
        _exact_sha256(value, field)
    if (
        type(power_row_count) is not int
        or power_row_count <= 0
        or not isinstance(reference_by_desc, Mapping)
        or len(reference_by_desc) != power_row_count
    ):
        raise RoleContractError("historical power reference row count drift")
    ordered_records: list[_PowerReplayRecord] = []
    for desc_id in sorted(reference_by_desc):
        record = reference_by_desc.get(desc_id)
        if (
            type(desc_id) is not int
            or desc_id < 0
            or not isinstance(record, _PowerReplayRecord)
            or record.desc_id != desc_id
            or tuple(record.outcomes) != POWER_OUTCOME_KEYS
            or any(
                type(value) is not int or value not in {0, 1}
                for value in record.outcomes.values()
            )
        ):
            raise RoleContractError("historical power reference record drift")
        _exact_video_id(record.gt_video_id, "historical reference gt_video_id")
        ordered_records.append(record)
    cluster_ids = sorted(
        {record.gt_video_id for record in ordered_records},
        key=lambda value: value.encode("utf-8"),
    )
    if len(cluster_ids) < 2:
        raise RoleContractError(
            "historical power reference requires at least two video clusters"
        )
    outcome_distribution = {
        outcome: {
            "zero_count": sum(
                record.outcomes[outcome] == 0 for record in ordered_records
            ),
            "one_count": sum(
                record.outcomes[outcome] == 1 for record in ordered_records
            ),
        }
        for outcome in POWER_OUTCOME_KEYS
    }
    normalized_material = [
        {
            "desc_id": record.desc_id,
            "gt_video_id": record.gt_video_id,
            "outcomes": dict(record.outcomes),
        }
        for record in ordered_records
    ]
    base = {
        "schema_version": "c28f_g2_historical_power_reference_v1",
        "population_semantics": (
            "INDEPENDENT_C7_CALIBRATED_A4_HISTORICAL_REFERENCE_ONLY"
        ),
        "power_rows_sha256": power_rows_sha256,
        "power_row_count": power_row_count,
        "replay_packet_sha256": replay_packet_sha256,
        "query_manifest_sha256": query_manifest_sha256,
        "source_registry_sha256": source_registry_sha256,
        "reference_desc_id_lines_sha256": _sha_lines(
            str(record.desc_id) for record in ordered_records
        ),
        "reference_gt_video_cluster_count": len(cluster_ids),
        "reference_gt_video_id_lines_sha256": _sha_lines(cluster_ids),
        "reference_outcome_distribution": outcome_distribution,
        "normalized_reference_sha256": hashlib.sha256(
            canonical_json_bytes(normalized_material)
        ).hexdigest(),
        "train_fit_desc_id_join_performed": False,
        "train_fit_video_id_join_performed": False,
        "candidate_membership_fields_consumed": [
            "actual_candidate_role_query_count",
            "actual_candidate_role_gt_video_cluster_count",
        ],
        "historical_performance_selects_membership": False,
    }
    return validate_historical_power_reference_population({
        **base,
        "reference_population_sha256": semantic_sha256(base),
    })


def build_historical_power_reference_from_rows(
    rows: Any,
    *,
    power_rows_sha256: str,
    replay_packet_sha256: str,
    query_manifest_sha256: str,
    source_registry_sha256: str,
) -> dict[str, Any]:
    """Sole public builder for the committed independent G2 reference."""

    if not isinstance(rows, (list, tuple)) or not rows:
        raise RoleContractError("historical reference rows must be explicit")
    detached_rows = json.loads(canonical_json_bytes(list(rows)).decode("utf-8"))
    if hashlib.sha256(canonical_json_bytes(detached_rows)).hexdigest() != _exact_sha256(
        power_rows_sha256,
        "power rows SHA",
    ):
        raise RoleContractError("historical reference rows SHA drift")
    reference_by_desc = _normalize_power_replay_rows(
        detached_rows,
        query_manifest_sha256=query_manifest_sha256,
        replay_packet_sha256=replay_packet_sha256,
    )
    return _build_historical_power_reference_population(
        reference_by_desc,
        power_rows_sha256=power_rows_sha256,
        power_row_count=len(detached_rows),
        replay_packet_sha256=replay_packet_sha256,
        query_manifest_sha256=query_manifest_sha256,
        source_registry_sha256=source_registry_sha256,
    )


def _cluster_order_key(video_id: str, seed: int) -> tuple[bytes, bytes]:
    encoded = video_id.encode("utf-8", errors="strict")
    digest = hashlib.sha256(str(seed).encode("ascii") + b"\0" + encoded).digest()
    return digest, encoded


def _power_hash_list(values: Sequence[Any]) -> bytes:
    return hashlib.sha256(canonical_json_bytes(list(values))).digest()


def _uniform_counter_index(
    cluster_count: int,
    *,
    role: str,
    outer_index: int,
    inner_index: int,
    draw_index: int,
) -> int:
    if cluster_count <= 0:
        raise RoleContractError("power resampling requires at least one GT-video cluster")
    modulus = 1 << 256
    acceptance_limit = (modulus // cluster_count) * cluster_count
    rejection_counter = 0
    while True:
        value = int.from_bytes(
            _power_hash_list(
                (
                    POWER_COUNTER_DOMAIN,
                    ASSIGNMENT_SEED,
                    role,
                    outer_index,
                    inner_index,
                    draw_index,
                    rejection_counter,
                )
            ),
            "big",
        )
        if value < acceptance_limit:
            return value % cluster_count
        rejection_counter += 1
        if rejection_counter > 1_000_000:
            raise RoleContractError("power counter rejection loop exceeded its hard cap")


def _outer_candidate_slots(
    *,
    role: str,
    outer_index: int,
    proposed_query_count: int,
    proposed_cluster_count: int,
    historical_clusters: Sequence[
        tuple[str, Sequence[_PowerReplayRecord]]
    ],
) -> tuple[
    list[tuple[int, int, _PowerReplayRecord]],
    list[int],
    list[int],
]:
    """Project one outer reference draw to exactly N equal q/q+1 slots."""

    if (
        not historical_clusters
        or proposed_cluster_count > proposed_query_count
        or proposed_cluster_count <= 0
    ):
        raise RoleContractError("outer candidate slot projection counts are invalid")
    quotient, remainder = divmod(proposed_query_count, proposed_cluster_count)
    if quotient <= 0:
        raise RoleContractError("outer candidate slot quotient must be positive")
    reference_cluster_indices = [
        _uniform_counter_index(
            len(historical_clusters),
            role=role,
            outer_index=outer_index,
            inner_index=-1,
            draw_index=position,
        )
        for position in range(proposed_cluster_count)
    ]
    slot_counts = [
        quotient + int(position < remainder)
        for position in range(proposed_cluster_count)
    ]
    if sum(slot_counts) != proposed_query_count:
        raise RoleContractError("outer candidate slot projection lost exact N")
    slots: list[tuple[int, int, _PowerReplayRecord]] = []
    for position, reference_cluster_index in enumerate(
        reference_cluster_indices
    ):
        _video_id, records = historical_clusters[reference_cluster_index]
        if not records:
            raise RoleContractError("historical reference cluster is empty")
        for local_slot_index in range(slot_counts[position]):
            record_index = _uniform_counter_index(
                len(records),
                role=role,
                outer_index=outer_index,
                inner_index=-(position + 2),
                draw_index=local_slot_index,
            )
            slots.append(
                (position, local_slot_index, records[record_index])
            )
    if len(slots) != proposed_query_count:
        raise RoleContractError("outer candidate slot materialization lost exact N")
    return slots, slot_counts, reference_cluster_indices


def _inner_cluster_position_resamples(
    *,
    role: str,
    outer_index: int,
    slot_counts: Sequence[int],
) -> tuple[list[list[int]], list[int]]:
    """Draw C outer positions with replacement; never deduplicate repeats."""

    cluster_count = len(slot_counts)
    if cluster_count <= 0 or any(
        type(value) is not int or value <= 0 for value in slot_counts
    ):
        raise RoleContractError("inner cluster-position slot counts are invalid")
    multiplicity_rows: list[list[int]] = []
    resampled_slot_counts: list[int] = []
    for inner_index in range(POWER_INNER_RESAMPLES):
        multiplicities = [0] * cluster_count
        for draw_index in range(cluster_count):
            position = _uniform_counter_index(
                cluster_count,
                role=role,
                outer_index=outer_index,
                inner_index=inner_index,
                draw_index=draw_index,
            )
            multiplicities[position] += 1
        if sum(multiplicities) != cluster_count:
            raise RoleContractError(
                "inner cluster-position resample deduplicated or lost draws"
            )
        denominator = sum(
            multiplicity * slot_counts[position]
            for position, multiplicity in enumerate(multiplicities)
        )
        if denominator <= 0:
            raise RoleContractError("inner paired-delta denominator is empty")
        multiplicity_rows.append(multiplicities)
        resampled_slot_counts.append(denominator)
    return multiplicity_rows, resampled_slot_counts


def _sign_normalized_improvement(
    *,
    baseline_value: int,
    effect_rule: Mapping[str, Any],
) -> int:
    """Normalize both recall gains and error reductions to paired delta +1."""

    if (
        type(baseline_value) is not int
        or baseline_value not in {0, 1}
        or baseline_value != effect_rule.get("eligible_baseline_value")
    ):
        raise RoleContractError("paired effect baseline is not eligible")
    injected_flip = effect_rule.get("injected_flip")
    if injected_flip == "0_TO_1" and baseline_value == 0:
        candidate_value = 1
        improvement_sign = 1
    elif injected_flip == "1_TO_0" and baseline_value == 1:
        candidate_value = 0
        improvement_sign = -1
    else:
        raise RoleContractError("paired effect direction registry drift")
    normalized = (candidate_value - baseline_value) * improvement_sign
    if normalized != 1:
        raise RoleContractError("paired effect sign normalization drift")
    return normalized


def _outer_effect_eligible_position_order(
    slots: Sequence[tuple[int, int, _PowerReplayRecord]],
    *,
    role: str,
    outer_index: int,
    effect_key: str,
    effect_rule: Mapping[str, Any],
) -> list[int]:
    """Build the nested outer-local eligible prefix dispersed by position."""

    outcome_name = str(effect_rule["binary_outcome"])
    eligible_value = int(effect_rule["eligible_baseline_value"])
    by_position: dict[
        int,
        list[tuple[bytes, int, _PowerReplayRecord]],
    ] = {}
    position_video_ids: dict[int, str] = {}
    for position, local_slot_index, record in slots:
        baseline_value = record.outcomes.get(outcome_name)
        if baseline_value != eligible_value:
            continue
        _sign_normalized_improvement(
            baseline_value=baseline_value,
            effect_rule=effect_rule,
        )
        slot_digest = _power_hash_list(
            (
                POWER_COUNTER_DOMAIN,
                ASSIGNMENT_SEED,
                role,
                outer_index,
                effect_key,
                position,
                local_slot_index,
                record.desc_id,
            )
        )
        by_position.setdefault(position, []).append(
            (slot_digest, local_slot_index, record)
        )
        prior_video_id = position_video_ids.setdefault(
            position,
            record.gt_video_id,
        )
        if prior_video_id != record.gt_video_id:
            raise RoleContractError("outer cluster position mixed reference clusters")
    dispersed: list[tuple[int, bytes, bytes, int, int, int]] = []
    for position, items in by_position.items():
        items.sort(key=lambda item: (item[0], item[1], item[2].desc_id))
        position_digest = _power_hash_list(
            (
                POWER_COUNTER_DOMAIN,
                ASSIGNMENT_SEED,
                role,
                outer_index,
                effect_key,
                position,
                position_video_ids[position],
            )
        )
        for within_position_rank, (
            slot_digest,
            local_slot_index,
            record,
        ) in enumerate(items):
            dispersed.append(
                (
                    within_position_rank,
                    position_digest,
                    slot_digest,
                    position,
                    local_slot_index,
                    record.desc_id,
                )
            )
    dispersed.sort()
    return [item[3] for item in dispersed]


def _outer_paired_detection_threshold(
    eligible_position_order: Sequence[int],
    *,
    inner_position_multiplicities: Sequence[Sequence[int]],
    inner_resampled_slot_counts: Sequence[int],
    maximum_k: int,
) -> dict[str, Any]:
    """Return the first k whose 2nd/48th nearest-rank paired CI is positive."""

    if (
        type(maximum_k) is not int
        or maximum_k <= 0
        or len(inner_position_multiplicities) != POWER_INNER_RESAMPLES
        or len(inner_resampled_slot_counts) != POWER_INNER_RESAMPLES
    ):
        raise RoleContractError("outer paired detection threshold inputs drift")
    numerators = [0] * POWER_INNER_RESAMPLES
    prior_lower = 0.0
    prior_upper = 0.0
    maximum_evaluated_k = min(maximum_k, len(eligible_position_order))
    for k, position in enumerate(eligible_position_order, start=1):
        if k > maximum_k:
            break
        for inner_index, multiplicities in enumerate(
            inner_position_multiplicities
        ):
            if type(position) is not int or not 0 <= position < len(
                multiplicities
            ):
                raise RoleContractError("eligible slot cluster position drift")
            numerators[inner_index] += multiplicities[position]
        paired_deltas = sorted(
            numerators[inner_index]
            / float(inner_resampled_slot_counts[inner_index])
            for inner_index in range(POWER_INNER_RESAMPLES)
        )
        lower = paired_deltas[POWER_LOWER_NEAREST_RANK - 1]
        upper = paired_deltas[POWER_UPPER_NEAREST_RANK - 1]
        if lower > 0.0:
            if upper <= 0.0:
                raise RoleContractError("paired percentile CI ordering drift")
            return {
                "detection_threshold_k": k,
                "ci_lower_at_threshold": lower,
                "ci_upper_at_threshold": upper,
                "ci_lower_at_k_minus_1": prior_lower,
                "ci_upper_at_k_minus_1": prior_upper,
                "maximum_evaluated_k": maximum_evaluated_k,
                "ci_lower_at_maximum_k": None,
                "ci_upper_at_maximum_k": None,
            }
        prior_lower = lower
        prior_upper = upper
    return {
        "detection_threshold_k": None,
        "ci_lower_at_threshold": None,
        "ci_upper_at_threshold": None,
        "ci_lower_at_k_minus_1": None,
        "ci_upper_at_k_minus_1": None,
        "maximum_evaluated_k": maximum_evaluated_k,
        "ci_lower_at_maximum_k": prior_lower,
        "ci_upper_at_maximum_k": prior_upper,
    }


def _role_power_sensitivity(
    *,
    role: str,
    proposed_query_count: int,
    proposed_cluster_count: int,
    reference_records: Sequence[_PowerReplayRecord],
    reference_population: Mapping[str, Any],
) -> dict[str, Any]:
    minimum_effects, effect_outcomes = _validated_effect_registries()
    if role not in {"train_fit_mechanism_dev", "train_fit_confirm"}:
        raise RoleContractError("power sensitivity is defined only for mechanism_dev/confirm")
    if (
        type(proposed_query_count) is not int
        or type(proposed_cluster_count) is not int
        or proposed_query_count <= 0
        or proposed_cluster_count <= 0
        or proposed_cluster_count > proposed_query_count
    ):
        raise RoleContractError("candidate role power counts are invalid")
    if (
        not isinstance(reference_population, Mapping)
        or reference_population.get("schema_version")
        != "c28f_g2_historical_power_reference_v1"
        or reference_population.get("reference_population_sha256")
        != semantic_sha256(
            dict(reference_population),
            excluded_fields=("reference_population_sha256",),
        )
        or reference_population.get("train_fit_desc_id_join_performed") is not False
        or reference_population.get("train_fit_video_id_join_performed") is not False
        or reference_population.get("historical_performance_selects_membership")
        is not False
    ):
        raise RoleContractError("historical power reference binding is invalid")
    historical = list(reference_records)
    if (
        not historical
        or any(not isinstance(record, _PowerReplayRecord) for record in historical)
        or [record.desc_id for record in historical]
        != sorted({record.desc_id for record in historical})
        or reference_population.get("power_row_count") != len(historical)
    ):
        raise RoleContractError("historical power reference records are not exact")
    by_cluster: dict[str, list[_PowerReplayRecord]] = {}
    for record in historical:
        by_cluster.setdefault(record.gt_video_id, []).append(record)
    historical_clusters = [
        (
            video_id,
            tuple(sorted(by_cluster[video_id], key=lambda item: item.desc_id)),
        )
        for video_id in sorted(by_cluster, key=lambda value: value.encode("utf-8"))
    ]
    reference_cluster_count = len(historical_clusters)
    reference_query_count = len(historical)
    if (
        reference_cluster_count < 2
        or reference_population.get("reference_gt_video_cluster_count")
        != reference_cluster_count
    ):
        raise RoleContractError("historical power reference cluster coverage drift")

    maximum_k = int(
        math.ceil(POWER_MAX_INJECTED_DELTA * proposed_query_count)
    )
    outer_detection_k: dict[str, list[Optional[int]]] = {
        effect_key: [] for effect_key in sorted(effect_outcomes)
    }
    outer_eligible_counts: dict[str, list[int]] = {
        effect_key: [] for effect_key in sorted(effect_outcomes)
    }
    outer_threshold_ci: dict[
        str,
        list[dict[str, Any]],
    ] = {effect_key: [] for effect_key in sorted(effect_outcomes)}
    outer_cluster_sample_hashes: list[str] = []
    inner_position_resample_hashes: list[str] = []
    for outer_index in range(POWER_OUTER_EXPERIMENTS):
        slots, slot_counts, reference_cluster_indices = (
            _outer_candidate_slots(
                role=role,
                outer_index=outer_index,
                proposed_query_count=proposed_query_count,
                proposed_cluster_count=proposed_cluster_count,
                historical_clusters=historical_clusters,
            )
        )
        inner_multiplicities, inner_denominators = (
            _inner_cluster_position_resamples(
                role=role,
                outer_index=outer_index,
                slot_counts=slot_counts,
            )
        )
        outer_cluster_sample_hashes.append(
            hashlib.sha256(
                canonical_json_bytes(reference_cluster_indices)
            ).hexdigest()
        )
        inner_position_resample_hashes.append(
            hashlib.sha256(
                canonical_json_bytes(inner_multiplicities)
            ).hexdigest()
        )
        for effect_key in sorted(effect_outcomes):
            eligible_order = _outer_effect_eligible_position_order(
                slots,
                role=role,
                outer_index=outer_index,
                effect_key=effect_key,
                effect_rule=effect_outcomes[effect_key],
            )
            detection = _outer_paired_detection_threshold(
                eligible_order,
                inner_position_multiplicities=inner_multiplicities,
                inner_resampled_slot_counts=inner_denominators,
                maximum_k=maximum_k,
            )
            outer_eligible_counts[effect_key].append(len(eligible_order))
            outer_detection_k[effect_key].append(
                detection["detection_threshold_k"]
            )
            outer_threshold_ci[effect_key].append(
                {
                    "outer_index": outer_index,
                    "eligible_slot_count": len(eligible_order),
                    **detection,
                }
            )

    effect_results: dict[str, Any] = {}
    all_pass = True
    for effect_key in sorted(effect_outcomes):
        finite_outer = sorted(
            value
            for value in outer_detection_k[effect_key]
            if value is not None and value <= maximum_k
        )
        required_detected = int(
            math.ceil(POWER_TARGET * POWER_OUTER_EXPERIMENTS)
        )
        mde_k: Optional[int] = (
            finite_outer[required_detected - 1]
            if len(finite_outer) >= required_detected
            else None
        )
        if mde_k is None:
            power_at_mde = None
            power_at_previous = None
            mde_ratio = None
            status = "INSUFFICIENT_POWER_OR_ELIGIBLE_HISTORY"
            all_pass = False
            detected_at = None
            detected_before = None
        else:
            detected_at = sum(
                value is not None and value <= mde_k
                for value in outer_detection_k[effect_key]
            )
            detected_before = sum(
                value is not None and value <= mde_k - 1
                for value in outer_detection_k[effect_key]
            )
            power_at_mde = detected_at / float(POWER_OUTER_EXPERIMENTS)
            power_at_previous = detected_before / float(POWER_OUTER_EXPERIMENTS)
            if power_at_mde < POWER_TARGET or power_at_previous >= POWER_TARGET:
                raise RoleContractError("power first-MDE threshold invariant failed")
            mde_ratio = mde_k / float(proposed_query_count)
            status = (
                "PASS"
                if mde_ratio <= minimum_effects[effect_key]
                else "INSUFFICIENT_POWER"
            )
            all_pass = all_pass and status == "PASS"
        reference_eligible_count = sum(
            record.outcomes[effect_outcomes[effect_key]["binary_outcome"]]
            == effect_outcomes[effect_key]["eligible_baseline_value"]
            for record in historical
        )
        eligible_counts = outer_eligible_counts[effect_key]
        thresholds = outer_detection_k[effect_key]
        threshold_ci_rows = outer_threshold_ci[effect_key]
        detected_at_maximum = sum(
            value is not None and value <= maximum_k
            for value in thresholds
        )
        effect_results[effect_key] = {
            "binary_outcome": effect_outcomes[effect_key][
                "binary_outcome"
            ],
            "sensitivity_rule": effect_outcomes[effect_key][
                "sensitivity_rule"
            ],
            "eligible_baseline_value": effect_outcomes[effect_key][
                "eligible_baseline_value"
            ],
            "injected_flip": effect_outcomes[effect_key]["injected_flip"],
            "minimum_effect_ratio": minimum_effects[effect_key],
            "reference_eligible_query_count": reference_eligible_count,
            "outer_eligible_query_count_min": min(eligible_counts),
            "outer_eligible_query_count_max": max(eligible_counts),
            "outer_eligible_query_count_mean": (
                sum(eligible_counts) / float(POWER_OUTER_EXPERIMENTS)
            ),
            "maximum_supported_k": maximum_k,
            "outer_detection_thresholds": thresholds,
            "outer_detection_thresholds_sha256": hashlib.sha256(
                canonical_json_bytes(thresholds)
            ).hexdigest(),
            "outer_threshold_ci_sha256": hashlib.sha256(
                canonical_json_bytes(threshold_ci_rows)
            ).hexdigest(),
            "outer_threshold_ci": threshold_ci_rows,
            "mde_k": mde_k,
            "mde_ratio": mde_ratio,
            "power_at_mde": power_at_mde,
            "power_at_k_minus_1": power_at_previous,
            "detected_outer_count_at_mde": detected_at,
            "detected_outer_count_at_k_minus_1": detected_before,
            "detected_outer_count_at_maximum_supported_k": (
                detected_at_maximum
            ),
            "power_at_maximum_supported_k": (
                detected_at_maximum / float(POWER_OUTER_EXPERIMENTS)
            ),
            "paired_ci_detection_rule": (
                "49_INNER_CLUSTER_POSITION_BOOTSTRAPS_NEAREST_RANK_2_48_"
                "SIGN_NORMALIZED_LOWER_GT_ZERO"
            ),
            "status": status,
        }
    quotient, remainder = divmod(
        proposed_query_count,
        proposed_cluster_count,
    )
    return {
        "role": role,
        "query_count": proposed_query_count,
        "gt_video_cluster_count": proposed_cluster_count,
        "candidate_role_query_count": proposed_query_count,
        "candidate_role_gt_video_cluster_count": proposed_cluster_count,
        "discrete_query_resolution": 1.0 / float(proposed_query_count),
        "power_input_semantics": (
            "OUTER_LOCAL_PAIRED_REFERENCE_BOOTSTRAP_PLUS_CANDIDATE_ROLE_COUNTS_ONLY"
        ),
        "power_algorithm": POWER_ALGORITHM,
        "power_counter_domain": POWER_COUNTER_DOMAIN,
        "reference_population_sha256": reference_population[
            "reference_population_sha256"
        ],
        "reference_query_count": reference_query_count,
        "reference_gt_video_cluster_count": reference_cluster_count,
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
            "projected_slot_count": proposed_query_count,
            "cluster_position_count": proposed_cluster_count,
        },
        "outer_cluster_position_sampling": (
            "C_REFERENCE_CLUSTER_DRAWS_WITH_REPLACEMENT"
        ),
        "outer_cluster_position_sample_sha256": hashlib.sha256(
            canonical_json_bytes(outer_cluster_sample_hashes)
        ).hexdigest(),
        "inner_cluster_position_resampling": (
            "C_OUTER_POSITION_DRAWS_WITH_REPLACEMENT_DUPLICATES_"
            "PRESERVED_ALL_POSITION_SLOTS"
        ),
        "inner_cluster_position_resample_sha256": hashlib.sha256(
            canonical_json_bytes(inner_position_resample_hashes)
        ).hexdigest(),
        "paired_delta_semantics": (
            "SAME_SLOT_BASELINE_CANDIDATE_SIGN_NORMALIZED_IMPROVEMENT_MEAN"
        ),
        "paired_ci_contract": {
            "inner_sample_count": POWER_INNER_RESAMPLES,
            "lower_nearest_rank_1_indexed": POWER_LOWER_NEAREST_RANK,
            "upper_nearest_rank_1_indexed": POWER_UPPER_NEAREST_RANK,
            "detection_rule": "SIGN_NORMALIZED_LOWER_BOUND_GT_ZERO",
        },
        "bootstrap_resample_count": POWER_BOOTSTRAP_REPLICATES,
        "outer_experiment_count": POWER_OUTER_EXPERIMENTS,
        "inner_resamples_per_outer": POWER_INNER_RESAMPLES,
        "effect_results": effect_results,
        "status": "PASS" if all_pass else "INSUFFICIENT_POWER",
    }


def _validate_power_result_evidence(
    result: Mapping[str, Any],
    *,
    role: str,
    expected_query_count: int,
    expected_cluster_count: int,
    reference_population: Mapping[str, Any],
) -> None:
    """Validate the complete V3 paired cluster-position bootstrap evidence."""

    minimum_effects, effect_outcomes = _validated_effect_registries()
    if (
        type(expected_query_count) is not int
        or type(expected_cluster_count) is not int
        or expected_query_count <= 0
        or expected_cluster_count <= 0
        or expected_cluster_count > expected_query_count
    ):
        raise RoleContractError("V3 paired power expected counts are invalid")
    exact_result_fields = {
        "role",
        "query_count",
        "gt_video_cluster_count",
        "candidate_role_query_count",
        "candidate_role_gt_video_cluster_count",
        "discrete_query_resolution",
        "power_input_semantics",
        "power_algorithm",
        "power_counter_domain",
        "reference_population_sha256",
        "reference_query_count",
        "reference_gt_video_cluster_count",
        "train_fit_desc_id_join_performed",
        "train_fit_video_id_join_performed",
        "historical_performance_selects_membership",
        "candidate_cluster_size_distribution_consumed",
        "equal_slot_projection",
        "outer_cluster_position_sampling",
        "outer_cluster_position_sample_sha256",
        "inner_cluster_position_resampling",
        "inner_cluster_position_resample_sha256",
        "paired_delta_semantics",
        "paired_ci_contract",
        "bootstrap_resample_count",
        "outer_experiment_count",
        "inner_resamples_per_outer",
        "effect_results",
        "status",
    }
    if (
        not isinstance(result, Mapping)
        or set(result) != exact_result_fields
        or result.get("role") != role
        or result.get("query_count") != expected_query_count
        or result.get("gt_video_cluster_count") != expected_cluster_count
        or result.get("candidate_role_query_count") != expected_query_count
        or result.get("candidate_role_gt_video_cluster_count")
        != expected_cluster_count
        or result.get("discrete_query_resolution")
        != 1.0 / float(expected_query_count)
        or result.get("power_input_semantics")
        != (
            "OUTER_LOCAL_PAIRED_REFERENCE_BOOTSTRAP_PLUS_"
            "CANDIDATE_ROLE_COUNTS_ONLY"
        )
        or result.get("power_algorithm") != POWER_ALGORITHM
        or result.get("power_counter_domain") != POWER_COUNTER_DOMAIN
        or result.get("reference_population_sha256")
        != reference_population.get("reference_population_sha256")
        or result.get("reference_query_count")
        != reference_population.get("power_row_count")
        or result.get("reference_gt_video_cluster_count")
        != reference_population.get("reference_gt_video_cluster_count")
        or result.get("train_fit_desc_id_join_performed") is not False
        or result.get("train_fit_video_id_join_performed") is not False
        or result.get("historical_performance_selects_membership") is not False
        or result.get("candidate_cluster_size_distribution_consumed") is not False
        or result.get("outer_cluster_position_sampling")
        != "C_REFERENCE_CLUSTER_DRAWS_WITH_REPLACEMENT"
        or result.get("inner_cluster_position_resampling")
        != (
            "C_OUTER_POSITION_DRAWS_WITH_REPLACEMENT_DUPLICATES_"
            "PRESERVED_ALL_POSITION_SLOTS"
        )
        or result.get("paired_delta_semantics")
        != "SAME_SLOT_BASELINE_CANDIDATE_SIGN_NORMALIZED_IMPROVEMENT_MEAN"
        or result.get("paired_ci_contract")
        != {
            "inner_sample_count": POWER_INNER_RESAMPLES,
            "lower_nearest_rank_1_indexed": POWER_LOWER_NEAREST_RANK,
            "upper_nearest_rank_1_indexed": POWER_UPPER_NEAREST_RANK,
            "detection_rule": "SIGN_NORMALIZED_LOWER_BOUND_GT_ZERO",
        }
        or result.get("bootstrap_resample_count")
        != POWER_BOOTSTRAP_REPLICATES
        or result.get("outer_experiment_count") != POWER_OUTER_EXPERIMENTS
        or result.get("inner_resamples_per_outer") != POWER_INNER_RESAMPLES
    ):
        raise RoleContractError("V3 paired power result identity/semantics drift")
    _exact_sha256(
        result.get("outer_cluster_position_sample_sha256"),
        "outer_cluster_position_sample_sha256",
    )
    _exact_sha256(
        result.get("inner_cluster_position_resample_sha256"),
        "inner_cluster_position_resample_sha256",
    )
    quotient, remainder = divmod(expected_query_count, expected_cluster_count)
    if result.get("equal_slot_projection") != {
        "assumption": (
            "EQUAL_q_OR_q_PLUS_1_SLOT_PROJECTION_NOT_OBSERVED_"
            "CANDIDATE_CLUSTER_SIZE_DISTRIBUTION"
        ),
        "quotient_q": quotient,
        "remainder_r": remainder,
        "q_plus_1_position_prefix_length": remainder,
        "projected_slot_count": expected_query_count,
        "cluster_position_count": expected_cluster_count,
    }:
        raise RoleContractError("V3 paired power equal-slot projection drift")
    effects = result.get("effect_results")
    if not isinstance(effects, Mapping) or set(effects) != set(effect_outcomes):
        raise RoleContractError("V3 paired power effect registry drift")
    distribution = reference_population.get("reference_outcome_distribution")
    if not isinstance(distribution, Mapping):
        raise RoleContractError("V3 paired power reference distribution absent")
    expected_effect_fields = {
        "binary_outcome",
        "sensitivity_rule",
        "eligible_baseline_value",
        "injected_flip",
        "minimum_effect_ratio",
        "reference_eligible_query_count",
        "outer_eligible_query_count_min",
        "outer_eligible_query_count_max",
        "outer_eligible_query_count_mean",
        "maximum_supported_k",
        "outer_detection_thresholds",
        "outer_detection_thresholds_sha256",
        "outer_threshold_ci_sha256",
        "outer_threshold_ci",
        "mde_k",
        "mde_ratio",
        "power_at_mde",
        "power_at_k_minus_1",
        "detected_outer_count_at_mde",
        "detected_outer_count_at_k_minus_1",
        "detected_outer_count_at_maximum_supported_k",
        "power_at_maximum_supported_k",
        "paired_ci_detection_rule",
        "status",
    }
    all_pass = True
    maximum_k = int(math.ceil(POWER_MAX_INJECTED_DELTA * expected_query_count))
    required_detected = int(math.ceil(POWER_TARGET * POWER_OUTER_EXPERIMENTS))
    for effect_key in sorted(effect_outcomes):
        effect = effects[effect_key]
        rule = effect_outcomes[effect_key]
        if (
            not isinstance(effect, Mapping)
            or set(effect) != expected_effect_fields
            or effect.get("binary_outcome") != rule["binary_outcome"]
            or effect.get("sensitivity_rule") != rule["sensitivity_rule"]
            or effect.get("eligible_baseline_value")
            != rule["eligible_baseline_value"]
            or effect.get("injected_flip") != rule["injected_flip"]
            or effect.get("minimum_effect_ratio")
            != minimum_effects[effect_key]
            or effect.get("maximum_supported_k") != maximum_k
            or effect.get("paired_ci_detection_rule")
            != (
                "49_INNER_CLUSTER_POSITION_BOOTSTRAPS_NEAREST_RANK_2_48_"
                "SIGN_NORMALIZED_LOWER_GT_ZERO"
            )
        ):
            raise RoleContractError("V3 paired power effect semantics drift")
        outcome_counts = distribution.get(rule["binary_outcome"])
        eligible_count_key = (
            "zero_count"
            if rule["eligible_baseline_value"] == 0
            else "one_count"
        )
        if (
            not isinstance(outcome_counts, Mapping)
            or effect.get("reference_eligible_query_count")
            != outcome_counts.get(eligible_count_key)
        ):
            raise RoleContractError("V3 paired power reference eligibility drift")
        thresholds = effect.get("outer_detection_thresholds")
        ci_rows = effect.get("outer_threshold_ci")
        if (
            not isinstance(thresholds, list)
            or len(thresholds) != POWER_OUTER_EXPERIMENTS
            or any(
                value is not None
                and (type(value) is not int or not 1 <= value <= maximum_k)
                for value in thresholds
            )
            or effect.get("outer_detection_thresholds_sha256")
            != hashlib.sha256(canonical_json_bytes(thresholds)).hexdigest()
            or not isinstance(ci_rows, list)
            or len(ci_rows) != POWER_OUTER_EXPERIMENTS
            or effect.get("outer_threshold_ci_sha256")
            != hashlib.sha256(canonical_json_bytes(ci_rows)).hexdigest()
        ):
            raise RoleContractError("V3 paired power outer threshold evidence drift")
        eligible_counts: list[int] = []
        for outer_index, (threshold, ci_row) in enumerate(
            zip(thresholds, ci_rows)
        ):
            if (
                not isinstance(ci_row, Mapping)
                or set(ci_row)
                != {
                    "outer_index",
                    "eligible_slot_count",
                    "detection_threshold_k",
                    "ci_lower_at_threshold",
                    "ci_upper_at_threshold",
                    "ci_lower_at_k_minus_1",
                    "ci_upper_at_k_minus_1",
                    "maximum_evaluated_k",
                    "ci_lower_at_maximum_k",
                    "ci_upper_at_maximum_k",
                }
                or ci_row.get("outer_index") != outer_index
                or ci_row.get("detection_threshold_k") != threshold
                or type(ci_row.get("eligible_slot_count")) is not int
                or not 0
                <= ci_row["eligible_slot_count"]
                <= expected_query_count
            ):
                raise RoleContractError("V3 paired power outer CI row drift")
            eligible_counts.append(ci_row["eligible_slot_count"])
            lower = ci_row.get("ci_lower_at_threshold")
            upper = ci_row.get("ci_upper_at_threshold")
            prior_lower = ci_row.get("ci_lower_at_k_minus_1")
            prior_upper = ci_row.get("ci_upper_at_k_minus_1")
            maximum_lower = ci_row.get("ci_lower_at_maximum_k")
            maximum_upper = ci_row.get("ci_upper_at_maximum_k")
            expected_maximum_evaluated_k = min(
                maximum_k,
                ci_row["eligible_slot_count"],
            )
            if ci_row.get("maximum_evaluated_k") != expected_maximum_evaluated_k:
                raise RoleContractError("V3 paired power maximum-k evidence drift")
            if threshold is None:
                if (
                    lower is not None
                    or upper is not None
                    or prior_lower is not None
                    or prior_upper is not None
                    or isinstance(maximum_lower, bool)
                    or not isinstance(maximum_lower, (int, float))
                    or isinstance(maximum_upper, bool)
                    or not isinstance(maximum_upper, (int, float))
                    or not math.isfinite(float(maximum_lower))
                    or not math.isfinite(float(maximum_upper))
                    or not 0.0
                    <= float(maximum_lower)
                    <= 0.0
                    <= float(maximum_upper)
                    <= 1.0
                ):
                    raise RoleContractError(
                        "V3 paired power undetected maximum-k CI drift"
                    )
            elif (
                isinstance(lower, bool)
                or not isinstance(lower, (int, float))
                or isinstance(upper, bool)
                or not isinstance(upper, (int, float))
                or not math.isfinite(float(lower))
                or not math.isfinite(float(upper))
                or not 0.0 < float(lower) <= float(upper) <= 1.0
                or threshold > ci_row["eligible_slot_count"]
                or isinstance(prior_lower, bool)
                or not isinstance(prior_lower, (int, float))
                or isinstance(prior_upper, bool)
                or not isinstance(prior_upper, (int, float))
                or not math.isfinite(float(prior_lower))
                or not math.isfinite(float(prior_upper))
                or not 0.0
                <= float(prior_lower)
                <= 0.0
                <= float(prior_upper)
                <= 1.0
                or maximum_lower is not None
                or maximum_upper is not None
            ):
                raise RoleContractError(
                    "V3 paired power first-detected CI bounds drift"
                )
        if (
            effect.get("outer_eligible_query_count_min") != min(eligible_counts)
            or effect.get("outer_eligible_query_count_max") != max(eligible_counts)
            or effect.get("outer_eligible_query_count_mean")
            != sum(eligible_counts) / float(POWER_OUTER_EXPERIMENTS)
        ):
            raise RoleContractError("V3 paired power outer eligibility summary drift")
        finite = sorted(value for value in thresholds if value is not None)
        expected_mde = (
            finite[required_detected - 1]
            if len(finite) >= required_detected
            else None
        )
        detected_at_maximum = len(finite)
        if (
            effect.get("mde_k") != expected_mde
            or effect.get("detected_outer_count_at_maximum_supported_k")
            != detected_at_maximum
            or effect.get("power_at_maximum_supported_k")
            != detected_at_maximum / float(POWER_OUTER_EXPERIMENTS)
        ):
            raise RoleContractError("V3 paired power MDE threshold aggregation drift")
        if expected_mde is None:
            expected_mde_ratio = None
            expected_power = None
            expected_previous_power = None
            expected_detected = None
            expected_previous_detected = None
            expected_status = "INSUFFICIENT_POWER_OR_ELIGIBLE_HISTORY"
        else:
            expected_detected = sum(
                value is not None and value <= expected_mde
                for value in thresholds
            )
            expected_previous_detected = sum(
                value is not None and value <= expected_mde - 1
                for value in thresholds
            )
            expected_power = expected_detected / float(POWER_OUTER_EXPERIMENTS)
            expected_previous_power = (
                expected_previous_detected / float(POWER_OUTER_EXPERIMENTS)
            )
            expected_mde_ratio = expected_mde / float(expected_query_count)
            expected_status = (
                "PASS"
                if expected_mde_ratio <= minimum_effects[effect_key]
                else "INSUFFICIENT_POWER"
            )
            if (
                expected_power < POWER_TARGET
                or expected_previous_power >= POWER_TARGET
            ):
                raise RoleContractError("V3 paired power first-k invariant drift")
        if (
            effect.get("mde_ratio") != expected_mde_ratio
            or effect.get("power_at_mde") != expected_power
            or effect.get("power_at_k_minus_1") != expected_previous_power
            or effect.get("detected_outer_count_at_mde") != expected_detected
            or effect.get("detected_outer_count_at_k_minus_1")
            != expected_previous_detected
            or effect.get("status") != expected_status
        ):
            raise RoleContractError("V3 paired power effect decision drift")
        all_pass = all_pass and expected_status == "PASS"
    if result.get("status") != (
        "PASS" if all_pass else "INSUFFICIENT_POWER"
    ):
        raise RoleContractError("V3 paired power aggregate status drift")


def _validate_trusted_power_computation_receipt(
    receipt: Any,
    *,
    role_results: Mapping[str, Any],
    role_counts: Mapping[str, Any],
    power_replay_capability_sha256: str,
    power_rows_sha256: str,
    historical_reference_population_sha256: str,
    power_evaluator_contract_sha256: str,
    g2_committed_transaction_id: str,
    g2_committed_state_sha256: str,
    g2_committed_event_sha256: str,
    goal_id: str,
    attempt_id: str,
) -> dict[str, Any]:
    """Bind detached V3 results to controller recomputation over private rows."""

    expected_roles = {"train_fit_mechanism_dev", "train_fit_confirm"}
    exact_fields = {
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
    }
    if not isinstance(receipt, Mapping):
        raise RoleContractError("trusted power computation receipt is absent")
    detached = json.loads(canonical_json_bytes(dict(receipt)).decode("utf-8"))
    expected_role_counts = {
        role: {
            "query_count": role_counts[role]["query_count"],
            "gt_video_cluster_count": role_counts[role][
                "gt_video_cluster_count"
            ],
        }
        for role in sorted(expected_roles)
    } if (
        isinstance(role_counts, Mapping)
        and set(role_counts) == expected_roles
        and all(
            isinstance(role_counts.get(role), Mapping)
            and set(role_counts[role])
            == {"query_count", "gt_video_cluster_count"}
            and type(role_counts[role]["query_count"]) is int
            and role_counts[role]["query_count"] > 0
            and type(role_counts[role]["gt_video_cluster_count"]) is int
            and 0 < role_counts[role]["gt_video_cluster_count"]
            <= role_counts[role]["query_count"]
            for role in expected_roles
        )
    ) else None
    if expected_role_counts is None:
        raise RoleContractError("trusted power computation role counts drift")
    expected_results = {
        role: json.loads(
            canonical_json_bytes(dict(role_results[role])).decode("utf-8")
        )
        for role in sorted(expected_roles)
    } if (
        isinstance(role_results, Mapping)
        and set(role_results) == expected_roles
        and all(isinstance(role_results.get(role), Mapping) for role in expected_roles)
    ) else None
    if expected_results is None:
        raise RoleContractError("trusted power computation role results drift")
    role_result_sha256s = {
        role: hashlib.sha256(
            canonical_json_bytes(expected_results[role])
        ).hexdigest()
        for role in sorted(expected_roles)
    }
    if (
        set(detached) != exact_fields
        or detached.get("schema_version")
        != "c28f_a4_g2_power_computation_receipt_capability_v1"
        or detached.get("status") != "CONTROLLER_RECOMPUTED_TRUSTED_POWER"
        or detached.get("goal_id") != goal_id
        or detached.get("attempt_id") != attempt_id
        or detached.get("g2_committed_transaction_id")
        != g2_committed_transaction_id
        or detached.get("g2_committed_state_sha256")
        != g2_committed_state_sha256
        or detached.get("g2_committed_event_sha256")
        != g2_committed_event_sha256
        or detached.get("power_replay_capability_sha256")
        != power_replay_capability_sha256
        or detached.get("power_rows_sha256") != power_rows_sha256
        or detached.get("historical_reference_population_sha256")
        != historical_reference_population_sha256
        or detached.get("power_evaluator_contract_sha256")
        != power_evaluator_contract_sha256
        or detached.get("role_counts") != expected_role_counts
        or detached.get("role_counts_sha256")
        != hashlib.sha256(
            canonical_json_bytes(expected_role_counts)
        ).hexdigest()
        or detached.get("role_results") != expected_results
        or detached.get("role_result_sha256s") != role_result_sha256s
        or detached.get("combined_results_sha256")
        != hashlib.sha256(canonical_json_bytes(expected_results)).hexdigest()
        or detached.get("raw_power_rows_disclosed") is not False
        or detached.get("receipt_sha256")
        != semantic_sha256(detached, excluded_fields=("receipt_sha256",))
    ):
        raise RoleContractError(
            "trusted controller power computation receipt/result mismatch"
        )
    for field_name in (
        "g2_committed_state_sha256",
        "g2_committed_event_sha256",
        "power_replay_capability_sha256",
        "power_rows_sha256",
        "historical_reference_population_sha256",
        "power_evaluator_contract_sha256",
        "role_counts_sha256",
        "combined_results_sha256",
        "receipt_sha256",
    ):
        _exact_sha256(detached.get(field_name), field_name)
    return detached


def _require_persisted_trusted_power_computation_receipt(
    receipt: Mapping[str, Any],
) -> dict[str, Any]:
    from .a4_control import (
        validate_persisted_g2_power_computation_receipt,
    )

    try:
        persisted = validate_persisted_g2_power_computation_receipt(
            dict(receipt)
        )
    except BaseException as error:
        raise RoleContractError(
            "trusted power computation receipt lacks persisted controller lineage"
        ) from error
    if canonical_json_bytes(persisted) != canonical_json_bytes(dict(receipt)):
        raise RoleContractError(
            "persisted trusted power computation receipt payload drift"
        )
    return dict(persisted)


def normalize_role_records(
    records: Sequence[RoleRecord], expected_desc_ids: Sequence[int]
) -> list[RoleRecord]:
    expected = [int(value) for value in expected_desc_ids]
    if any(type(value) is not int or value < 0 for value in expected_desc_ids):
        raise RoleContractError("expected desc IDs must be non-negative exact integers")
    if len(expected) != len(set(expected)):
        raise RoleContractError("expected train_fit manifest contains duplicate desc IDs")
    by_desc: dict[int, RoleRecord] = {}
    for record in records:
        if not isinstance(record, RoleRecord):
            raise RoleContractError("role records must be RoleRecord objects")
        if record.desc_id in by_desc:
            raise RoleContractError(f"duplicate projected desc_id: {record.desc_id}")
        by_desc[record.desc_id] = record
    expected_set = set(expected)
    if set(by_desc) != expected_set:
        raise RoleContractError(
            "projected train_fit records do not exactly match the authorized manifest"
        )
    return [by_desc[desc_id] for desc_id in sorted(expected_set)]


def assign_video_clusters(
    records: Sequence[RoleRecord],
    expected_desc_ids: Sequence[int],
    *,
    seed: int = ASSIGNMENT_SEED,
) -> dict[str, list[RoleRecord]]:
    if type(seed) is not int:
        raise RoleContractError("assignment seed must be an exact integer")
    normalized = normalize_role_records(records, expected_desc_ids)
    by_video: dict[str, list[RoleRecord]] = {}
    for record in normalized:
        by_video.setdefault(record.gt_video_id, []).append(record)
    clusters = sorted(by_video.items(), key=lambda item: _cluster_order_key(item[0], seed))
    assigned: dict[str, list[RoleRecord]] = {
        role: [] for role, _minimum in ROLE_MIN_QUERY_COUNTS
    }
    assigned[CORE_ROLE] = []
    cursor = 0
    for role, minimum in ROLE_MIN_QUERY_COUNTS:
        while len(assigned[role]) < minimum:
            if cursor >= len(clusters):
                raise RoleContractError(f"insufficient complete video clusters for role {role}")
            _video_id, cluster = clusters[cursor]
            assigned[role].extend(cluster)
            cursor += 1
    for _video_id, cluster in clusters[cursor:]:
        assigned[CORE_ROLE].extend(cluster)
    if not assigned[CORE_ROLE]:
        raise RoleContractError("cluster assignment left no train_fit_core queries")
    return {
        role: sorted(values, key=lambda record: record.desc_id)
        for role, values in assigned.items()
    }


def _frozen_power_expansion_lanes(
    frozen_core_order: Sequence[str],
) -> tuple[dict[str, list[str]], list[str]]:
    """Freeze disjoint per-role identity lanes before any power outcome exists."""

    normalized = list(frozen_core_order)
    if (
        len(normalized) < 3
        or any(not isinstance(video_id, str) for video_id in normalized)
        or len(set(normalized)) != len(normalized)
    ):
        raise RoleContractError(
            "power expansion requires three or more unique frozen core clusters"
        )
    candidates = normalized[:-1]
    lanes = {
        "train_fit_mechanism_dev": candidates[0::2],
        "train_fit_confirm": candidates[1::2],
    }
    if (
        not lanes["train_fit_mechanism_dev"]
        or not lanes["train_fit_confirm"]
        or set(lanes["train_fit_mechanism_dev"])
        & set(lanes["train_fit_confirm"])
    ):
        raise RoleContractError("power expansion role lanes are not disjoint")
    return lanes, [normalized[-1]]


def _power_adaptive_assignments(
    records: Sequence[RoleRecord],
    expected_desc_ids: Sequence[int],
    *,
    reference_by_desc: Mapping[int, _PowerReplayRecord],
    reference_population: Mapping[str, Any],
) -> tuple[dict[str, list[RoleRecord]], dict[str, Any]]:
    minimum_effects, _effect_outcomes = _validated_effect_registries()
    normalized = normalize_role_records(records, expected_desc_ids)
    assignments = assign_video_clusters(
        normalized,
        expected_desc_ids,
        seed=ASSIGNMENT_SEED,
    )
    if (
        not isinstance(reference_by_desc, Mapping)
        or not isinstance(reference_population, Mapping)
        or reference_population.get("reference_population_sha256")
        != semantic_sha256(
            dict(reference_population),
            excluded_fields=("reference_population_sha256",),
        )
        or len(reference_by_desc) != reference_population.get("power_row_count")
    ):
        raise RoleContractError("historical power reference binding drift")
    reference_records = tuple(
        reference_by_desc[desc_id] for desc_id in sorted(reference_by_desc)
    )
    by_video: dict[str, list[RoleRecord]] = {}
    for record in normalized:
        by_video.setdefault(record.gt_video_id, []).append(record)
    frozen_core_prefix_order = [
        video_id
        for video_id, _cluster in sorted(
            (
                (video_id, by_video[video_id])
                for video_id in {
                    record.gt_video_id for record in assignments[CORE_ROLE]
                }
            ),
            key=lambda item: _cluster_order_key(item[0], ASSIGNMENT_SEED),
        )
    ]
    frozen_role_lanes, reserved_core_order = _frozen_power_expansion_lanes(
        frozen_core_prefix_order
    )
    available_prefix_by_role = {
        role: list(lane)
        for role, lane in frozen_role_lanes.items()
    }
    adjustment_rounds: list[dict[str, Any]] = []

    def evaluated_power(role: str) -> dict[str, Any]:
        return _role_power_sensitivity(
            role=role,
            proposed_query_count=len(assignments[role]),
            proposed_cluster_count=len(
                {record.gt_video_id for record in assignments[role]}
            ),
            reference_records=reference_records,
            reference_population=reference_population,
        )

    maximum_rounds = max(len(lane) for lane in frozen_role_lanes.values())
    for round_index in range(maximum_rounds + 1):
        role_results = {
            role: evaluated_power(role)
            for role in ("train_fit_mechanism_dev", "train_fit_confirm")
        }
        failing = [
            role
            for role in ("train_fit_mechanism_dev", "train_fit_confirm")
            if role_results[role]["status"] != "PASS"
        ]
        if not failing:
            break
        if not any(available_prefix_by_role[role] for role in failing):
            break
        round_record: dict[str, Any] = {
            "round_index": round_index,
            "failing_roles": list(failing),
            "moves": {},
        }
        for role in failing:
            available_prefix = available_prefix_by_role[role]
            current_count = len(assignments[role])
            required_counts = [current_count + 1]
            for effect in role_results[role]["effect_results"].values():
                mde_k = effect.get("mde_k")
                minimum_ratio = effect.get("minimum_effect_ratio")
                if (
                    type(mde_k) is int
                    and mde_k > 0
                    and isinstance(minimum_ratio, (int, float))
                    and not isinstance(minimum_ratio, bool)
                    and minimum_ratio > 0.0
                ):
                    required_counts.append(
                        int(math.ceil(mde_k / float(minimum_ratio)))
                    )
                elif effect.get("status") != "PASS":
                    required_counts.append(current_count * 2)
            target_count = min(max(required_counts), current_count * 2)
            moved: list[str] = []
            while (
                len(assignments[role]) < target_count
                and available_prefix
            ):
                video_id = available_prefix.pop(0)
                cluster = by_video[video_id]
                assignments[role].extend(cluster)
                moved_desc_ids = {record.desc_id for record in cluster}
                assignments[CORE_ROLE] = [
                    record
                    for record in assignments[CORE_ROLE]
                    if record.desc_id not in moved_desc_ids
                ]
                moved.append(video_id)
            assignments[role] = sorted(
                assignments[role], key=lambda record: record.desc_id
            )
            assignments[CORE_ROLE] = sorted(
                assignments[CORE_ROLE], key=lambda record: record.desc_id
            )
            round_record["moves"][role] = {
                "target_query_count": target_count,
                "moved_cluster_count": len(moved),
                "moved_cluster_ids_in_seed_order": moved,
                "moved_cluster_ids_sha256": _sha_lines(moved),
                "query_count_after": len(assignments[role]),
            }
        adjustment_rounds.append(round_record)
        if not any(
            move["moved_cluster_count"] > 0
            for move in round_record["moves"].values()
        ):
            break
    final_power = {
        role: evaluated_power(role)
        for role in ("train_fit_mechanism_dev", "train_fit_confirm")
    }
    reduction_count = _effect_threshold_reduction_count(
        final_power,
        minimum_effects,
    )
    return assignments, {
        "status": (
            "PASS"
            if all(result["status"] == "PASS" for result in final_power.values())
            else "INSUFFICIENT_POWER"
        ),
        "expansion_policy": (
            "PREREGISTERED_DISJOINT_ROLE_LANE_PREFIXES_BEFORE_MANIFEST_LOCK"
        ),
        "minimum_effect_registry_sha256": MINIMUM_EFFECTS_REGISTRY_SHA256,
        "effect_outcome_registry_sha256": EFFECT_OUTCOME_REGISTRY_SHA256,
        "effect_threshold_reduction_count": reduction_count,
        "historical_reference_population": dict(reference_population),
        "membership_assignment_inputs": (
            "TRAIN_FIT_GT_VIDEO_CLUSTERS_PLUS_FROZEN_SEEDED_DISJOINT_ROLE_"
            "LANE_ORDERS_PLUS_POWER_DETERMINED_PER_ROLE_PREFIX_LENGTH_BEFORE_"
            "MANIFEST_LOCK"
        ),
        "historical_outcomes_select_identity_order_or_within_prefix_members": False,
        "cross_role_historical_outcomes_affect_lane_identity_order_or_members": (
            False
        ),
        "historical_outcomes_may_determine_preregistered_prefix_length": True,
        "membership_lock_timing": "ALL_PREFIX_EXPANSION_BEFORE_ROLE_MANIFEST_LOCK",
        "expansion_lane_algorithm": POWER_EXPANSION_LANE_ALGORITHM,
        "frozen_core_prefix_video_ids_sha256": _sha_lines(
            frozen_core_prefix_order
        ),
        "frozen_role_lane_video_ids_sha256s": {
            role: _sha_lines(lane)
            for role, lane in sorted(frozen_role_lanes.items())
        },
        "reserved_core_video_ids_sha256": _sha_lines(reserved_core_order),
        "role_specific_expansion_lanes_disjoint": True,
        "adjustment_rounds": adjustment_rounds,
        "role_results": final_power,
    }


def build_role_manifests(
    assignments: Mapping[str, Sequence[RoleRecord]],
    *,
    train_fit_manifest_sha256: str,
    assignment_source_sha256: str,
    producer_argv_sha256: Optional[str] = None,
    producer_control_binding_sha256: Optional[str] = None,
    assignment_contract: Optional[Mapping[str, Any]] = None,
    seed: int = ASSIGNMENT_SEED,
) -> dict[str, dict[str, Any]]:
    expected_roles = {role for role, _minimum in ROLE_MIN_QUERY_COUNTS} | {CORE_ROLE}
    if set(assignments) != expected_roles:
        raise RoleContractError("assignment role set is not exact")
    if type(seed) is not int or seed != ASSIGNMENT_SEED:
        raise RoleContractError("role manifests require the frozen assignment seed")
    for digest, field in (
        (train_fit_manifest_sha256, "train_fit_manifest_sha256"),
        (assignment_source_sha256, "assignment_source_sha256"),
    ):
        _exact_sha256(digest, field)
    if (producer_argv_sha256 is None) == (producer_control_binding_sha256 is None):
        raise RoleContractError("exactly one producer lineage kind is required")
    if producer_control_binding_sha256 is not None:
        producer_lineage = {
            "kind": "CONTROL_CAPABILITY_BINDING_SHA256",
            "sha256": _exact_sha256(
                producer_control_binding_sha256,
                "producer_control_binding_sha256",
            ),
        }
    else:
        producer_lineage = {
            "kind": "ARGV_SHA256",
            "sha256": _exact_sha256(producer_argv_sha256, "producer_argv_sha256"),
        }
    if assignment_contract is None:
        resolved_assignment_contract = {
            "algorithm": ASSIGNMENT_ALGORITHM,
            "seed": seed,
            "cluster_order": "sha256(seed_ascii + NUL + video_id_utf8),video_id_utf8",
            "cluster_atomic": True,
        }
    else:
        resolved_assignment_contract = dict(assignment_contract)
        if set(resolved_assignment_contract) != {
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
        }:
            raise RoleContractError("power-adaptive assignment contract fields are not exact")
        if resolved_assignment_contract.get(
            "assignment_contract_sha256"
        ) != semantic_sha256(
            resolved_assignment_contract,
            excluded_fields=("assignment_contract_sha256",),
        ):
            raise RoleContractError("power-adaptive assignment contract self-hash mismatch")
        if (
            resolved_assignment_contract.get("algorithm")
            != POWER_ADAPTIVE_ASSIGNMENT_ALGORITHM
            or resolved_assignment_contract.get("seed") != ASSIGNMENT_SEED
            or resolved_assignment_contract.get("cluster_order")
            != "sha256(seed_ascii + NUL + video_id_utf8),video_id_utf8"
            or resolved_assignment_contract.get("cluster_atomic") is not True
            or resolved_assignment_contract.get("candidate_membership_source")
            != (
                "TRAIN_FIT_GT_VIDEO_CLUSTERS_PLUS_FROZEN_SEEDED_DISJOINT_ROLE_"
                "LANE_ORDERS_PLUS_POWER_DETERMINED_PER_ROLE_PREFIX_LENGTH_"
                "BEFORE_MANIFEST_LOCK"
            )
            or resolved_assignment_contract.get(
                "historical_outcomes_select_identity_order_or_within_prefix_members"
            )
            is not False
            or resolved_assignment_contract.get(
                "cross_role_historical_outcomes_affect_lane_identity_order_or_members"
            )
            is not False
            or resolved_assignment_contract.get(
                "historical_outcomes_may_determine_preregistered_prefix_length"
            )
            is not True
            or resolved_assignment_contract.get(
                "power_expansion_lane_algorithm"
            )
            != POWER_EXPANSION_LANE_ALGORITHM
            or resolved_assignment_contract.get("membership_lock_timing")
            != "ALL_PREFIX_EXPANSION_BEFORE_ROLE_MANIFEST_LOCK"
        ):
            raise RoleContractError("power-adaptive assignment contract drift")
        _exact_sha256(
            resolved_assignment_contract.get(
                "historical_reference_population_sha256"
            ),
            "historical_reference_population_sha256",
        )
        _exact_sha256(
            resolved_assignment_contract.get("power_expansion_report_sha256"),
            "power_expansion_report_sha256",
        )
    manifests: dict[str, dict[str, Any]] = {}
    for role in sorted(assignments):
        records = list(assignments[role])
        if any(not isinstance(record, RoleRecord) for record in records):
            raise RoleContractError(f"role contains a non-RoleRecord value: {role}")
        if records != sorted(records, key=lambda record: record.desc_id):
            raise RoleContractError(f"role records are not canonical by desc_id: {role}")
        desc_ids = [record.desc_id for record in records]
        if len(desc_ids) != len(set(desc_ids)):
            raise RoleContractError(f"role contains duplicate desc IDs: {role}")
        video_ids = sorted({record.gt_video_id for record in records}, key=lambda x: x.encode("utf-8"))
        base = {
            "schema_version": ROLE_SCHEMA,
            "role": role,
            "entity_schema": {
                "query_entity": "desc_id:int",
                "cluster_entity": "gt_video_id:utf8",
            },
            "desc_ids": desc_ids,
            "query_count": len(desc_ids),
            "gt_video_ids": video_ids,
            "gt_video_count": len(video_ids),
            "desc_id_lines_sha256": _sha_lines(str(value) for value in desc_ids),
            "gt_video_id_lines_sha256": _sha_lines(video_ids),
            "desc_video_pairs_sha256": _sha_lines(
                f"{record.desc_id}\t{record.gt_video_id}" for record in records
            ),
            "train_fit_manifest_sha256": train_fit_manifest_sha256,
            "assignment_source_sha256": assignment_source_sha256,
            "assignment": dict(resolved_assignment_contract),
            "producer_lineage": dict(producer_lineage),
            "corpus": {
                "video_count": CORPUS_VIDEO_COUNT,
                "ordered_video_id_sha256": CORPUS_VIDEO_ID_ORDER_SHA256,
            },
        }
        manifests[role] = {
            **base,
            "manifest_sha256": semantic_sha256(base),
        }
    return manifests


def _exact_pairwise_disjoint(sets: Mapping[str, set[Any]]) -> None:
    names = sorted(sets)
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            if sets[left] & sets[right]:
                raise RoleContractError(f"role entity overlap: {left} / {right}")


def _build_role_lock_candidate_from_validated_fields(
    manifests: Mapping[str, Mapping[str, Any]],
    *,
    expected_train_fit_desc_ids: Sequence[int],
    projected_records: Sequence[RoleRecord],
    protected_desc_ids: set[int],
    protected_video_ids: Optional[set[str]],
    protected_mapping_receipt_sha256: Optional[str],
    expected_assignments: Optional[Mapping[str, Sequence[RoleRecord]]] = None,
    expected_assignment_contract: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    expected_roles = {role for role, _minimum in ROLE_MIN_QUERY_COUNTS} | {CORE_ROLE}
    if set(manifests) != expected_roles:
        raise RoleContractError("role lock received an inexact manifest set")
    expected_desc_list = list(expected_train_fit_desc_ids)
    if any(type(value) is not int or value < 0 for value in expected_desc_list):
        raise RoleContractError("expected train_fit desc IDs are not exact non-negative integers")
    if len(expected_desc_list) != len(set(expected_desc_list)):
        raise RoleContractError("expected train_fit desc IDs contain duplicates")
    normalized_projection = normalize_role_records(
        projected_records, expected_desc_list
    )
    if expected_assignments is None:
        frozen_assignments = assign_video_clusters(
            normalized_projection,
            expected_desc_list,
            seed=ASSIGNMENT_SEED,
        )
        resolved_assignment_contract = {
            "algorithm": ASSIGNMENT_ALGORITHM,
            "seed": ASSIGNMENT_SEED,
            "cluster_order": "sha256(seed_ascii + NUL + video_id_utf8),video_id_utf8",
            "cluster_atomic": True,
        }
    else:
        if set(expected_assignments) != expected_roles:
            raise RoleContractError("power-adaptive assignment role set is not exact")
        frozen_assignments = {}
        for role in sorted(expected_roles):
            values = list(expected_assignments[role])
            if any(not isinstance(record, RoleRecord) for record in values):
                raise RoleContractError("power-adaptive assignment contains an invalid record")
            frozen_assignments[role] = sorted(values, key=lambda record: record.desc_id)
        if expected_assignment_contract is None:
            raise RoleContractError("power-adaptive assignment contract is required")
        resolved_assignment_contract = dict(expected_assignment_contract)
    frozen_desc_ids_by_role = {
        role: [record.desc_id for record in records]
        for role, records in frozen_assignments.items()
    }
    frozen_video_ids_by_role = {
        role: sorted(
            {record.gt_video_id for record in records},
            key=lambda value: value.encode("utf-8"),
        )
        for role, records in frozen_assignments.items()
    }
    expected_video_by_desc = {
        record.desc_id: record.gt_video_id for record in normalized_projection
    }
    if not isinstance(protected_desc_ids, set) or any(
        type(value) is not int or value < 0 for value in protected_desc_ids
    ):
        raise RoleContractError("protected desc manifest has the wrong entity schema")
    if protected_video_ids is not None and (
        not isinstance(protected_video_ids, set)
        or any(
            not isinstance(value, str) or _VIDEO_ID_RE.fullmatch(value) is None
            for value in protected_video_ids
        )
    ):
        raise RoleContractError("protected video manifest has the wrong entity schema")
    desc_sets: dict[str, set[int]] = {}
    video_sets: dict[str, set[str]] = {}
    role_hashes: dict[str, str] = {}
    role_counts: dict[str, dict[str, int]] = {}
    for role, manifest in manifests.items():
        if manifest.get("schema_version") != ROLE_SCHEMA or manifest.get("role") != role:
            raise RoleContractError(f"invalid role manifest identity: {role}")
        if manifest.get("manifest_sha256") != semantic_sha256(
            dict(manifest), excluded_fields=("manifest_sha256",)
        ):
            raise RoleContractError(f"role manifest self-hash mismatch: {role}")
        desc = list(manifest.get("desc_ids", []))
        videos = list(manifest.get("gt_video_ids", []))
        if any(type(value) is not int or value < 0 for value in desc):
            raise RoleContractError(f"role desc IDs have invalid types: {role}")
        if any(
            not isinstance(value, str) or _VIDEO_ID_RE.fullmatch(value) is None
            for value in videos
        ):
            raise RoleContractError(f"role video IDs have invalid types: {role}")
        if desc != sorted(desc) or len(desc) != len(set(desc)):
            raise RoleContractError(f"role desc IDs are not canonical: {role}")
        if videos != sorted(videos, key=lambda x: x.encode("utf-8")) or len(videos) != len(set(videos)):
            raise RoleContractError(f"role video IDs are not canonical: {role}")
        if desc != frozen_desc_ids_by_role[role]:
            raise RoleContractError(
                f"role desc IDs differ from the frozen seeded cluster assignment: {role}"
            )
        if videos != frozen_video_ids_by_role[role]:
            raise RoleContractError(
                f"role video IDs differ from the frozen seeded cluster assignment: {role}"
            )
        if type(manifest.get("query_count")) is not int or manifest["query_count"] != len(desc):
            raise RoleContractError(f"role query_count mismatch: {role}")
        if type(manifest.get("gt_video_count")) is not int or manifest["gt_video_count"] != len(videos):
            raise RoleContractError(f"role gt_video_count mismatch: {role}")
        if manifest.get("desc_id_lines_sha256") != _sha_lines(str(value) for value in desc):
            raise RoleContractError(f"role desc line hash mismatch: {role}")
        if manifest.get("gt_video_id_lines_sha256") != _sha_lines(videos):
            raise RoleContractError(f"role video line hash mismatch: {role}")
        expected_pairs = [
            (desc_id, expected_video_by_desc[desc_id]) for desc_id in desc
        ]
        if set(videos) != {video_id for _desc_id, video_id in expected_pairs}:
            raise RoleContractError(f"role video set differs from projected mapping: {role}")
        if manifest.get("desc_video_pairs_sha256") != _sha_lines(
            f"{desc_id}\t{video_id}" for desc_id, video_id in expected_pairs
        ):
            raise RoleContractError(f"role desc/video pair hash mismatch: {role}")
        if manifest.get("entity_schema") != {
            "query_entity": "desc_id:int",
            "cluster_entity": "gt_video_id:utf8",
        }:
            raise RoleContractError(f"role entity schema drift: {role}")
        assignment = manifest.get("assignment")
        if assignment != resolved_assignment_contract:
            raise RoleContractError(f"role assignment contract drift: {role}")
        if manifest.get("corpus") != {
            "video_count": CORPUS_VIDEO_COUNT,
            "ordered_video_id_sha256": CORPUS_VIDEO_ID_ORDER_SHA256,
        }:
            raise RoleContractError(f"role corpus contract drift: {role}")
        for field in (
            "train_fit_manifest_sha256",
            "assignment_source_sha256",
        ):
            _exact_sha256(manifest.get(field), f"{role}.{field}")
        producer_lineage = manifest.get("producer_lineage")
        if (
            not isinstance(producer_lineage, Mapping)
            or set(producer_lineage) != {"kind", "sha256"}
            or producer_lineage.get("kind")
            not in {"ARGV_SHA256", "CONTROL_CAPABILITY_BINDING_SHA256"}
        ):
            raise RoleContractError(f"role producer lineage drift: {role}")
        _exact_sha256(producer_lineage.get("sha256"), f"{role}.producer_lineage")
        desc_sets[role] = set(desc)
        video_sets[role] = set(videos)
        role_hashes[role] = _exact_sha256(
            manifest["manifest_sha256"], f"{role}.manifest_sha256"
        )
        role_counts[role] = {"queries": len(desc), "gt_videos": len(videos)}
    for lineage_field in (
        "train_fit_manifest_sha256",
        "assignment_source_sha256",
    ):
        values = {str(manifest[lineage_field]) for manifest in manifests.values()}
        if len(values) != 1:
            raise RoleContractError(f"role lineage differs across manifests: {lineage_field}")
    producer_lineages = {
        canonical_json_bytes(dict(manifest["producer_lineage"]))
        for manifest in manifests.values()
    }
    if len(producer_lineages) != 1:
        raise RoleContractError("role producer lineage differs across manifests")
    _exact_pairwise_disjoint(desc_sets)
    _exact_pairwise_disjoint(video_sets)
    expected = set(expected_desc_list)
    observed_union = set().union(*desc_sets.values())
    if observed_union != expected:
        raise RoleContractError("role union is not the exact train_fit manifest")
    minima = dict(ROLE_MIN_QUERY_COUNTS)
    for role, minimum in minima.items():
        if len(desc_sets[role]) < minimum:
            raise RoleContractError(f"role minimum not met: {role}")
    desc_intersections = {
        role: sorted(desc_sets[role] & protected_desc_ids) for role in sorted(desc_sets)
    }
    if any(desc_intersections.values()):
        raise RoleContractError("role desc IDs intersect protected desc IDs")
    if protected_video_ids is None:
        video_audit: dict[str, Any] = {
            "status": "PROTECTED_VIDEO_OVERLAP_UNAVAILABLE",
            "mapping_receipt_sha256": None,
            "role_intersections": None,
        }
    else:
        _exact_sha256(
            protected_mapping_receipt_sha256,
            "protected_mapping_receipt_sha256",
        )
        video_intersections = {
            role: sorted(video_sets[role] & protected_video_ids) for role in sorted(video_sets)
        }
        if any(video_intersections.values()):
            raise RoleContractError("role GT video IDs intersect protected video IDs")
        video_audit = {
            "status": "PASS_ZERO_INTERSECTION",
            "mapping_receipt_sha256": protected_mapping_receipt_sha256,
            "role_intersections": video_intersections,
        }
    base = {
        "schema_version": ROLE_LOCK_SCHEMA,
        "status": (
            "VALIDATED_FIELDS_CANDIDATE_NOT_CONTROL_COMMITTABLE"
            if video_audit["status"] == "PASS_ZERO_INTERSECTION"
            else "AWAITING_G4_INPUTS"
        ),
        "role_manifest_hashes": role_hashes,
        "role_counts": role_counts,
        "desc_entity_audit": {
            "entity_type": "desc_id",
            "status": "PASS_ZERO_INTERSECTION",
            "role_intersections": desc_intersections,
        },
        "video_entity_audit": {
            "entity_type": "video_id",
            **video_audit,
        },
        "pairwise_desc_disjoint": True,
        "pairwise_gt_video_disjoint": True,
        "train_fit_union_complete": True,
        "minimum_query_counts": minima,
        "missing_set": (
            []
            if video_audit["status"] == "PASS_ZERO_INTERSECTION"
            else ["protected_video_id_manifest"]
        ),
    }
    return {**base, "lock_sha256": semantic_sha256(base)}


def _legacy_build_role_lock_candidate(
    manifests: Mapping[str, Mapping[str, Any]],
    *,
    role_input_handle: Any,
) -> dict[str, Any]:
    """Build a G4 candidate only from the exact control-issued input handle."""

    raise RoleContractError(
        "legacy partial role-lock path disabled; use build_g4_role_policy_candidates"
    )

    from .a4_control import validate_g4_role_lock_input_handle

    snapshot = validate_g4_role_lock_input_handle(role_input_handle)
    records_raw = snapshot.get("train_fit_mapping_records")
    if not isinstance(records_raw, list) or not records_raw:
        raise RoleContractError("G4 role input has no train-fit mapping records")
    records: list[RoleRecord] = []
    for index, raw in enumerate(records_raw):
        if not isinstance(raw, Mapping) or set(raw) != {
            "desc_id",
            "ground_truth_video_id",
        }:
            raise RoleContractError(f"G4 role input record is not exact: {index}")
        records.append(
            RoleRecord(
                desc_id=raw["desc_id"],
                gt_video_id=raw["ground_truth_video_id"],
            )
        )
    protected_desc_raw = snapshot.get("protected_desc_ids")
    protected_video_raw = snapshot.get("protected_video_ids")
    if not isinstance(protected_desc_raw, list) or not isinstance(
        protected_video_raw, list
    ):
        raise RoleContractError("G4 protected entity manifests are not lists")
    protected_desc_ids = set(protected_desc_raw)
    protected_video_ids = set(protected_video_raw)
    expected_ids = [record.desc_id for record in records]
    candidate = _build_role_lock_candidate_from_validated_fields(
        manifests,
        expected_train_fit_desc_ids=expected_ids,
        projected_records=records,
        protected_desc_ids=protected_desc_ids,
        protected_video_ids=protected_video_ids,
        protected_mapping_receipt_sha256=snapshot[
            "projection_completion_capability_sha256"
        ],
    )
    if candidate.get("status") != "VALIDATED_FIELDS_CANDIDATE_NOT_CONTROL_COMMITTABLE":
        return candidate
    candidate_base = dict(candidate)
    candidate_base.pop("lock_sha256", None)
    candidate_base["status"] = "VALIDATED_CONTROL_INPUT_CANDIDATE_AWAITING_STAGE_COMMIT"
    candidate_base["control_input_binding"] = {
        "goal_id": snapshot["goal_id"],
        "attempt_id": snapshot["attempt_id"],
        "committed_transaction_id": snapshot["committed_transaction_id"],
        "committed_state_sha256": snapshot["committed_state_sha256"],
        "committed_event_sha256": snapshot["committed_event_sha256"],
        "train_fit_source_receipt_sha256": snapshot[
            "train_fit_source_receipt_sha256"
        ],
        "train_fit_mapping_sha256": snapshot["train_fit_mapping_sha256"],
        "projection_completion_capability_sha256": snapshot[
            "projection_completion_capability_sha256"
        ],
        "projection_manifest_sha256": snapshot["projection_manifest_sha256"],
        "mapping_rows_sha256": snapshot["mapping_rows_sha256"],
        "protected_video_id_lines_sha256": snapshot[
            "protected_video_id_lines_sha256"
        ],
        "role_input_snapshot_sha256": snapshot["role_input_snapshot_sha256"],
    }
    candidate_base["control_commit_required"] = True
    candidate_base["persistence_claimed_by_pure_function"] = False
    return {**candidate_base, "lock_sha256": semantic_sha256(candidate_base)}


def _build_formal_data_policy_from_control_candidates(
    *,
    role_lock: Mapping[str, Any],
    cluster_power_audit: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(role_lock, Mapping):
        raise RoleContractError("formal policy requires the committed role-lock artifact")
    current = dict(role_lock)
    expected_fields = {
        "schema_version",
        "status",
        "role_manifest_hashes",
        "role_counts",
        "desc_entity_audit",
        "video_entity_audit",
        "pairwise_desc_disjoint",
        "pairwise_gt_video_disjoint",
        "train_fit_union_complete",
        "minimum_query_counts",
        "missing_set",
        "control_input_binding",
        "control_commit_required",
        "persistence_claimed_by_pure_function",
        "lock_sha256",
    }
    if set(current) != expected_fields:
        raise RoleContractError("formal policy role-lock field set is not exact")
    if current.get("schema_version") != ROLE_LOCK_SCHEMA:
        raise RoleContractError("formal policy received the wrong role-lock schema")
    role_lock_sha256 = _exact_sha256(current.get("lock_sha256"), "role_lock_sha256")
    if role_lock_sha256 != semantic_sha256(
        current, excluded_fields=("lock_sha256",)
    ):
        raise RoleContractError("formal policy role-lock self-hash mismatch")
    if current.get(
        "status"
    ) != "VALIDATED_CONTROL_INPUT_CANDIDATE_AWAITING_STAGE_COMMIT" or current.get(
        "missing_set"
    ) != []:
        raise RoleContractError("formal policy requires a control-bound role-lock candidate")
    if current.get("control_commit_required") is not True or current.get(
        "persistence_claimed_by_pure_function"
    ) is not False:
        raise RoleContractError("formal policy role-lock control boundary drift")
    if current.get("pairwise_desc_disjoint") is not True or current.get(
        "pairwise_gt_video_disjoint"
    ) is not True or current.get("train_fit_union_complete") is not True:
        raise RoleContractError("formal policy role-lock invariants are incomplete")
    if current.get("desc_entity_audit", {}).get("status") != "PASS_ZERO_INTERSECTION":
        raise RoleContractError("formal policy desc-entity audit is not PASS")
    if current.get("video_entity_audit", {}).get("status") != "PASS_ZERO_INTERSECTION":
        raise RoleContractError("formal policy video-entity audit is not PASS")
    if not isinstance(cluster_power_audit, Mapping):
        raise RoleContractError("formal policy requires the committed cluster power audit")
    power = dict(cluster_power_audit)
    power_sha = _exact_sha256(power.get("audit_sha256"), "cluster power audit SHA")
    if power_sha != semantic_sha256(power, excluded_fields=("audit_sha256",)):
        raise RoleContractError("formal policy cluster power audit self-hash mismatch")
    if power.get("schema_version") != CLUSTER_POWER_AUDIT_SCHEMA or power.get(
        "status"
    ) != "VALIDATED_CONTROL_POWER_CANDIDATE_AWAITING_STAGE_COMMIT":
        raise RoleContractError("formal policy requires a control-bound power candidate")
    if power.get("role_lock_sha256") != role_lock_sha256:
        raise RoleContractError("formal policy power/role-lock binding mismatch")
    evidence_binding = power.get("power_evidence_binding")
    if not isinstance(evidence_binding, Mapping) or evidence_binding.get(
        "status"
    ) != "CONTROLLER_RECOMPUTED_TRUSTED_POWER" or not isinstance(
        evidence_binding.get("trusted_power_computation_receipt"), Mapping
    ):
        raise RoleContractError("formal policy power evidence is not control committed")
    base = {
        "schema_version": FORMAL_POLICY_SCHEMA,
        "status": "VALIDATED_CONTROL_POLICY_CANDIDATE_AWAITING_STAGE_COMMIT",
        "policy": FORMAL_POLICY,
        "U_formal_updates": U_FORMAL_UPDATES,
        "gradient_roles": [CORE_ROLE],
        "calibration_role": "train_fit_calibration",
        "calibration_use": "POSTHOC_TEMPERATURE_ONLY",
        "candidate_and_P0_same_update_budget": True,
        "route_mechanism_confirm_refit_allowed": False,
        "role_lock_sha256": role_lock_sha256,
        "cluster_power_audit_sha256": power_sha,
        "g2_power_evidence_sha256": evidence_binding.get("evidence_sha256"),
        "f7_may_change_policy": False,
        "f7_may_change_update_budget": False,
        "control_commit_required": True,
        "persistence_claimed_by_pure_function": False,
    }
    return {**base, "policy_sha256": semantic_sha256(base)}


def _legacy_build_cluster_power_audit(
    role_lock: Mapping[str, Any],
    *,
    projected_records: Sequence[RoleRecord],
    committed_power_evidence: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    raise RoleContractError(
        "legacy caller-signed power path disabled; use build_g4_role_policy_candidates"
    )
    current = dict(role_lock)
    if current.get("schema_version") != ROLE_LOCK_SCHEMA:
        raise RoleContractError("cluster power audit received the wrong role-lock schema")
    if current.get("lock_sha256") != semantic_sha256(
        current, excluded_fields=("lock_sha256",)
    ):
        raise RoleContractError("cluster power audit role-lock self-hash mismatch")
    role_lock_status = current.get("status")
    if role_lock_status not in {
        "VALIDATED_CONTROL_INPUT_CANDIDATE_AWAITING_STAGE_COMMIT",
        "VALIDATED_FIELDS_CANDIDATE_NOT_CONTROL_COMMITTABLE",
    }:
        raise RoleContractError("cluster power audit requires a validated role-lock candidate")
    control_bound_role_lock = (
        role_lock_status
        == "VALIDATED_CONTROL_INPUT_CANDIDATE_AWAITING_STAGE_COMMIT"
    )
    if not projected_records:
        raise RoleContractError("cluster power audit projection is empty")
    records = list(projected_records)
    if any(not isinstance(record, RoleRecord) for record in records):
        raise RoleContractError("cluster power audit projection type mismatch")
    if len({record.desc_id for record in records}) != len(records):
        raise RoleContractError("cluster power audit projection has duplicate desc IDs")
    cluster_sizes: dict[str, int] = {}
    for record in records:
        cluster_sizes[record.gt_video_id] = cluster_sizes.get(record.gt_video_id, 0) + 1
    ordered_sizes = sorted(cluster_sizes.values())

    def percentile(fraction: float) -> int:
        index = min(
            len(ordered_sizes) - 1,
            max(0, int(math.ceil(fraction * len(ordered_sizes))) - 1),
        )
        return ordered_sizes[index]

    role_counts = current.get("role_counts")
    if not isinstance(role_counts, Mapping):
        raise RoleContractError("cluster power audit role counts missing")
    minima = dict(ROLE_MIN_QUERY_COUNTS)
    margins: dict[str, Any] = {}
    for role, minimum in ROLE_MIN_QUERY_COUNTS:
        record = role_counts.get(role)
        if not isinstance(record, Mapping) or type(record.get("queries")) is not int:
            raise RoleContractError(f"cluster power audit count missing: {role}")
        query_count = int(record["queries"])
        if query_count < minimum:
            raise RoleContractError(f"cluster power audit minimum failed: {role}")
        margins[role] = {
            "minimum_queries": minimum,
            "assigned_queries": query_count,
            "cluster_overshoot_queries": query_count - minimum,
            "assigned_gt_video_clusters": record.get("gt_videos"),
        }
    core = role_counts.get(CORE_ROLE)
    if not isinstance(core, Mapping) or type(core.get("queries")) is not int or core[
        "queries"
    ] <= 0:
        raise RoleContractError("cluster power audit has no train_fit_core remainder")
    minimum_effects_ratio = {
        "core_primary_r1": 0.0025,
        "core_secondary_r5_or_r10": 0.0050,
        "core_wrong_video": 0.0050,
        "r2m_primary_r1": 0.0020,
        "m2r_tier_a_true_vr": 0.0200,
        "m2r_tier_b_true_vr": 0.0100,
        "mediator_primary_r1": 0.0020,
        "credit_primary_r1": 0.0025,
    }
    evidence_binding: dict[str, Any]
    status: str
    if committed_power_evidence is None:
        status = "AWAITING_COMMITTED_G2_POWER_EVIDENCE"
        evidence_binding = {
            "status": "MISSING",
            "reason": "VIDEO_CLUSTER_BOOTSTRAP_AND_DISCRETE_RESOLUTION_NOT_COMMITTED",
            "evidence_sha256": None,
        }
    else:
        evidence = dict(committed_power_evidence)
        expected_fields = {
            "schema_version",
            "status",
            "algorithm",
            "bootstrap_replicates",
            "bootstrap_seed",
            "g2_replay_control_receipt_sha256",
            "role_lock_sha256",
            "minimum_effects_ratio",
            "role_results",
            "evidence_sha256",
        }
        if set(evidence) != expected_fields:
            raise RoleContractError("cluster power evidence field set is not exact")
        evidence_sha = _exact_sha256(evidence.get("evidence_sha256"), "power evidence SHA")
        if evidence_sha != semantic_sha256(evidence, excluded_fields=("evidence_sha256",)):
            raise RoleContractError("cluster power evidence self-hash mismatch")
        if evidence.get("schema_version") != "c28f_g2_cluster_power_evidence_v1":
            raise RoleContractError("cluster power evidence schema mismatch")
        if evidence.get("status") != "PASS_PRE_REGISTERED_POWER":
            raise RoleContractError("cluster power evidence does not PASS")
        if evidence.get("algorithm") != "VIDEO_CLUSTER_BOOTSTRAP_10000_FIXED_SEED_V1":
            raise RoleContractError("cluster power algorithm drift")
        if evidence.get("bootstrap_replicates") != 10_000 or evidence.get(
            "bootstrap_seed"
        ) != ASSIGNMENT_SEED:
            raise RoleContractError("cluster power bootstrap contract drift")
        _exact_sha256(
            evidence.get("g2_replay_control_receipt_sha256"),
            "G2 replay control receipt SHA",
        )
        if evidence.get("role_lock_sha256") != current["lock_sha256"]:
            raise RoleContractError("cluster power evidence role-lock binding mismatch")
        if evidence.get("minimum_effects_ratio") != minimum_effects_ratio:
            raise RoleContractError("cluster power minimum-effect registry drift")
        results = evidence.get("role_results")
        if not isinstance(results, Mapping) or set(results) != {
            "train_fit_mechanism_dev",
            "train_fit_confirm",
        }:
            raise RoleContractError("cluster power role result set is not exact")
        for role, result in results.items():
            if not isinstance(result, Mapping) or set(result) != {
                "query_count",
                "gt_video_cluster_count",
                "discrete_query_resolution",
                "bootstrap_detectable_effect",
                "status",
            }:
                raise RoleContractError(f"cluster power role result is not exact: {role}")
            if result.get("status") != "PASS" or not all(
                isinstance(result.get(field), (int, float))
                and not isinstance(result.get(field), bool)
                and math.isfinite(float(result[field]))
                and float(result[field]) > 0.0
                for field in (
                    "query_count",
                    "gt_video_cluster_count",
                    "discrete_query_resolution",
                    "bootstrap_detectable_effect",
                )
            ):
                raise RoleContractError(f"cluster power role result failed: {role}")
            expected_role_count = current["role_counts"][role]
            if result.get("query_count") != expected_role_count["queries"] or result.get(
                "gt_video_cluster_count"
            ) != expected_role_count["gt_videos"]:
                raise RoleContractError(f"cluster power role counts drift: {role}")
            expected_resolution = 1.0 / float(expected_role_count["queries"])
            if not math.isclose(
                float(result["discrete_query_resolution"]),
                expected_resolution,
                rel_tol=0.0,
                abs_tol=1e-15,
            ):
                raise RoleContractError(f"cluster power query resolution drift: {role}")
            maximum_detectable = 0.0020 if role == "train_fit_mechanism_dev" else 0.0025
            if float(result["bootstrap_detectable_effect"]) > maximum_detectable:
                raise RoleContractError(f"cluster power is insufficient: {role}")
        status = (
            "VALIDATED_CONTROL_POWER_CANDIDATE_AWAITING_STAGE_COMMIT"
            if control_bound_role_lock
            else "VALIDATED_SYNTHETIC_POWER_CANDIDATE_NOT_COMMITTABLE"
        )
        evidence_binding = {
            "status": "COMMITTED_G2_CONTROL_EVIDENCE",
            "evidence_sha256": evidence_sha,
            "g2_replay_control_receipt_sha256": evidence[
                "g2_replay_control_receipt_sha256"
            ],
        }
    base = {
        "schema_version": CLUSTER_POWER_AUDIT_SCHEMA,
        "status": status,
        "role_lock_sha256": current["lock_sha256"],
        "projected_query_count": len(records),
        "gt_video_cluster_count": len(cluster_sizes),
        "cluster_size_distribution": {
            "minimum": ordered_sizes[0],
            "p50_nearest_rank": percentile(0.50),
            "p90_nearest_rank": percentile(0.90),
            "maximum": ordered_sizes[-1],
            "mean": sum(ordered_sizes) / float(len(ordered_sizes)),
        },
        "role_minimum_margins": margins,
        "train_fit_core_queries": core["queries"],
        "train_fit_core_gt_video_clusters": core.get("gt_videos"),
        "minimum_query_counts": minima,
        "minimum_effects_ratio": minimum_effects_ratio,
        "power_evidence_binding": evidence_binding,
        "effect_size_or_performance_claim": None,
        "interpretation": (
            "PRE_REGISTERED_POWER_ONLY_NO_POSTHOC_PERFORMANCE_SELECTION"
            if committed_power_evidence is not None
            else "ROLE_SIZE_DESIGN_ONLY_NOT_A_POWER_PASS"
        ),
    }
    return {**base, "audit_sha256": semantic_sha256(base)}


def _assert_power_prefix_expansion(
    *,
    records: Sequence[RoleRecord],
    expected_desc_ids: Sequence[int],
    final_assignments: Mapping[str, Sequence[RoleRecord]],
    power_report: Mapping[str, Any],
) -> None:
    exact_report_fields = {
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
    }
    if (
        not isinstance(power_report, Mapping)
        or set(power_report) != exact_report_fields
        or not isinstance(power_report.get("adjustment_rounds"), list)
    ):
        raise RoleContractError("power expansion report field set is not exact")
    minimum_effects, _effect_outcomes = _validated_effect_registries()
    reference_population = power_report.get("historical_reference_population")
    if (
        not isinstance(reference_population, Mapping)
        or reference_population.get("schema_version")
        != "c28f_g2_historical_power_reference_v1"
        or reference_population.get("reference_population_sha256")
        != semantic_sha256(
            dict(reference_population),
            excluded_fields=("reference_population_sha256",),
        )
        or reference_population.get("train_fit_desc_id_join_performed") is not False
        or reference_population.get("train_fit_video_id_join_performed") is not False
        or reference_population.get("historical_performance_selects_membership")
        is not False
    ):
        raise RoleContractError("power expansion historical reference drift")
    initial = assign_video_clusters(records, expected_desc_ids, seed=ASSIGNMENT_SEED)
    by_video: dict[str, list[RoleRecord]] = {}
    for record in records:
        by_video.setdefault(record.gt_video_id, []).append(record)
    initial_core_order = [
        video_id
        for video_id, _cluster in sorted(
            (
                (video_id, by_video[video_id])
                for video_id in {
                    record.gt_video_id for record in initial[CORE_ROLE]
                }
            ),
            key=lambda item: _cluster_order_key(item[0], ASSIGNMENT_SEED),
        )
    ]
    frozen_role_lanes, reserved_core_order = _frozen_power_expansion_lanes(
        initial_core_order
    )
    reconstructed = {role: list(values) for role, values in initial.items()}
    consumed_prefix_by_role = {
        role: [] for role in frozen_role_lanes
    }
    for expected_round, round_record in enumerate(
        power_report["adjustment_rounds"]
    ):
        if (
            not isinstance(round_record, Mapping)
            or set(round_record) != {"round_index", "failing_roles", "moves"}
            or round_record.get("round_index") != expected_round
            or not isinstance(round_record.get("failing_roles"), list)
            or round_record["failing_roles"]
            != [
                role
                for role in (
                    "train_fit_mechanism_dev",
                    "train_fit_confirm",
                )
                if role in round_record["failing_roles"]
            ]
            or not isinstance(round_record.get("moves"), Mapping)
            or set(round_record["moves"])
            != set(round_record["failing_roles"])
        ):
            raise RoleContractError("power expansion round is not canonical")
        for role in ("train_fit_mechanism_dev", "train_fit_confirm"):
            move = round_record["moves"].get(role)
            if not isinstance(move, Mapping):
                continue
            moved = move.get("moved_cluster_ids_in_seed_order")
            if (
                not isinstance(moved, list)
                or set(move)
                != {
                    "target_query_count",
                    "moved_cluster_count",
                    "moved_cluster_ids_in_seed_order",
                    "moved_cluster_ids_sha256",
                    "query_count_after",
                }
                or any(not isinstance(value, str) for value in moved)
                or type(move.get("target_query_count")) is not int
                or move["target_query_count"] <= 0
                or type(move.get("query_count_after")) is not int
                or move["query_count_after"] <= 0
                or move.get("moved_cluster_count") != len(moved)
                or move.get("moved_cluster_ids_sha256") != _sha_lines(moved)
            ):
                raise RoleContractError("power expansion move receipt is not exact")
            consumed_prefix = consumed_prefix_by_role[role]
            start = len(consumed_prefix)
            if moved != frozen_role_lanes[role][start : start + len(moved)]:
                raise RoleContractError(
                    "power outcomes selected a member outside frozen role lane prefix"
                )
            consumed_prefix.extend(moved)
            for video_id in moved:
                cluster = by_video[video_id]
                moved_desc_ids = {record.desc_id for record in cluster}
                reconstructed[role].extend(cluster)
                reconstructed[CORE_ROLE] = [
                    record
                    for record in reconstructed[CORE_ROLE]
                    if record.desc_id not in moved_desc_ids
                ]
            if move["query_count_after"] != len(reconstructed[role]):
                raise RoleContractError(
                    "power expansion move query-count postimage drift"
                )
    for role in reconstructed:
        expected = sorted(record.desc_id for record in reconstructed[role])
        observed = sorted(record.desc_id for record in final_assignments[role])
        if observed != expected:
            raise RoleContractError("power expansion final assignment/replay mismatch")
    if not reconstructed[CORE_ROLE]:
        raise RoleContractError("power expansion consumed train_fit_core remainder")
    if not set(reserved_core_order).issubset(
        {record.gt_video_id for record in reconstructed[CORE_ROLE]}
    ):
        raise RoleContractError("power expansion consumed reserved core cluster")
    role_results = power_report.get("role_results")
    if not isinstance(role_results, Mapping) or set(role_results) != {
        "train_fit_mechanism_dev",
        "train_fit_confirm",
    }:
        raise RoleContractError("power expansion lacks exact role results")
    for role in ("train_fit_mechanism_dev", "train_fit_confirm"):
        result = role_results.get(role)
        expected_query_count = len(final_assignments[role])
        expected_cluster_count = len(
            {record.gt_video_id for record in final_assignments[role]}
        )
        _validate_power_result_evidence(
            result,
            role=role,
            expected_query_count=expected_query_count,
            expected_cluster_count=expected_cluster_count,
            reference_population=reference_population,
        )
    actual_reduction_count = _effect_threshold_reduction_count(
        role_results,
        minimum_effects,
    )
    if (
        power_report.get("status")
        != (
            "PASS"
            if all(
                result["status"] == "PASS"
                for result in role_results.values()
            )
            else "INSUFFICIENT_POWER"
        )
        or power_report.get("expansion_policy")
        != "PREREGISTERED_DISJOINT_ROLE_LANE_PREFIXES_BEFORE_MANIFEST_LOCK"
        or power_report.get("membership_assignment_inputs")
        != (
            "TRAIN_FIT_GT_VIDEO_CLUSTERS_PLUS_FROZEN_SEEDED_DISJOINT_ROLE_"
            "LANE_ORDERS_PLUS_POWER_DETERMINED_PER_ROLE_PREFIX_LENGTH_BEFORE_"
            "MANIFEST_LOCK"
        )
        or power_report.get(
            "historical_outcomes_select_identity_order_or_within_prefix_members"
        ) is not False
        or power_report.get(
            "cross_role_historical_outcomes_affect_lane_identity_order_or_members"
        ) is not False
        or power_report.get(
            "historical_outcomes_may_determine_preregistered_prefix_length"
        ) is not True
        or power_report.get("membership_lock_timing")
        != "ALL_PREFIX_EXPANSION_BEFORE_ROLE_MANIFEST_LOCK"
        or power_report.get("expansion_lane_algorithm")
        != POWER_EXPANSION_LANE_ALGORITHM
        or power_report.get("frozen_core_prefix_video_ids_sha256")
        != _sha_lines(initial_core_order)
        or power_report.get("frozen_role_lane_video_ids_sha256s")
        != {
            role: _sha_lines(lane)
            for role, lane in sorted(frozen_role_lanes.items())
        }
        or power_report.get("reserved_core_video_ids_sha256")
        != _sha_lines(reserved_core_order)
        or power_report.get("role_specific_expansion_lanes_disjoint") is not True
        or power_report.get("effect_threshold_reduction_count")
        != actual_reduction_count
        or actual_reduction_count != 0
        or power_report.get("minimum_effect_registry_sha256")
        != MINIMUM_EFFECTS_REGISTRY_SHA256
        or power_report.get("effect_outcome_registry_sha256")
        != EFFECT_OUTCOME_REGISTRY_SHA256
    ):
        raise RoleContractError("power expansion/effect-size contract drift")


def _control_bound_role_lock(
    candidate: Mapping[str, Any], role_snapshot: Mapping[str, Any]
) -> dict[str, Any]:
    if candidate.get("status") != "VALIDATED_FIELDS_CANDIDATE_NOT_CONTROL_COMMITTABLE":
        raise RoleContractError("G4 role lock did not pass both entity audits")
    base = dict(candidate)
    base.pop("lock_sha256", None)
    base["status"] = "VALIDATED_CONTROL_INPUT_CANDIDATE_AWAITING_STAGE_COMMIT"
    base["control_input_binding"] = {
        "goal_id": role_snapshot["goal_id"],
        "attempt_id": role_snapshot["attempt_id"],
        "committed_transaction_id": role_snapshot["committed_transaction_id"],
        "committed_state_sha256": role_snapshot["committed_state_sha256"],
        "committed_event_sha256": role_snapshot["committed_event_sha256"],
        "train_fit_source_receipt_sha256": role_snapshot[
            "train_fit_source_receipt_sha256"
        ],
        "train_fit_mapping_sha256": role_snapshot["train_fit_mapping_sha256"],
        "projection_completion_capability_sha256": role_snapshot[
            "projection_completion_capability_sha256"
        ],
        "projection_manifest_sha256": role_snapshot["projection_manifest_sha256"],
        "mapping_rows_sha256": role_snapshot["mapping_rows_sha256"],
        "protected_video_id_lines_sha256": role_snapshot[
            "protected_video_id_lines_sha256"
        ],
        "role_input_snapshot_sha256": role_snapshot["role_input_snapshot_sha256"],
    }
    base["control_commit_required"] = True
    base["persistence_claimed_by_pure_function"] = False
    return {**base, "lock_sha256": semantic_sha256(base)}


def _computed_cluster_power_audit(
    *,
    role_lock: Mapping[str, Any],
    records: Sequence[RoleRecord],
    power_report: Mapping[str, Any],
    power_snapshot: Mapping[str, Any],
    power_replay_capability_sha256: str,
) -> dict[str, Any]:
    minimum_effects, _effect_outcomes = _validated_effect_registries()
    if power_report.get("status") != "PASS":
        raise RoleContractError("pre-registered power remains insufficient after prefix expansion")
    role_results = power_report.get("role_results")
    if not isinstance(role_results, Mapping) or set(role_results) != {
        "train_fit_mechanism_dev",
        "train_fit_confirm",
    }:
        raise RoleContractError("computed power role result set is not exact")
    reference_population = power_report.get("historical_reference_population")
    if (
        not isinstance(reference_population, Mapping)
        or reference_population.get("reference_population_sha256")
        != semantic_sha256(
            dict(reference_population),
            excluded_fields=("reference_population_sha256",),
        )
        or reference_population.get("power_rows_sha256")
        != power_snapshot.get("power_rows_sha256")
        or reference_population.get("power_row_count")
        != power_snapshot.get("power_row_count")
        or reference_population.get("replay_packet_sha256")
        != power_snapshot.get("replay_packet_sha256")
        or reference_population.get("query_manifest_sha256")
        != power_snapshot.get("query_manifest_sha256")
        or reference_population.get("source_registry_sha256")
        != power_snapshot.get("source_registry_sha256")
        or not isinstance(
            power_snapshot.get("historical_reference_population"),
            Mapping,
        )
        or canonical_json_bytes(dict(reference_population))
        != canonical_json_bytes(
            dict(power_snapshot.get("historical_reference_population", {}))
        )
        or not isinstance(
            power_snapshot.get("historical_reference_file_sha256"),
            str,
        )
        or _SHA256_RE.fullmatch(
            power_snapshot["historical_reference_file_sha256"]
        )
        is None
        or reference_population.get("train_fit_desc_id_join_performed") is not False
        or reference_population.get("train_fit_video_id_join_performed") is not False
        or reference_population.get("historical_performance_selects_membership")
        is not False
        or power_report.get(
            "historical_outcomes_select_identity_order_or_within_prefix_members"
        ) is not False
        or power_report.get(
            "cross_role_historical_outcomes_affect_lane_identity_order_or_members"
        ) is not False
        or power_report.get(
            "historical_outcomes_may_determine_preregistered_prefix_length"
        ) is not True
        or power_report.get("membership_lock_timing")
        != "ALL_PREFIX_EXPANSION_BEFORE_ROLE_MANIFEST_LOCK"
        or power_report.get("expansion_lane_algorithm")
        != POWER_EXPANSION_LANE_ALGORITHM
        or power_report.get("role_specific_expansion_lanes_disjoint") is not True
    ):
        raise RoleContractError("computed power historical reference lineage drift")
    role_counts = role_lock.get("role_counts")
    if not isinstance(role_counts, Mapping):
        raise RoleContractError("computed power role counts are missing")
    for role, result in role_results.items():
        expected_count = role_counts.get(role)
        if not isinstance(expected_count, Mapping):
            raise RoleContractError(f"computed power did not pass: {role}")
        _validate_power_result_evidence(
            result,
            role=role,
            expected_query_count=expected_count.get("queries"),
            expected_cluster_count=expected_count.get("gt_videos"),
            reference_population=reference_population,
        )
        if result.get("status") != "PASS":
            raise RoleContractError(f"computed power did not pass: {role}")
    trusted_role_counts = {
        role: {
            "query_count": role_counts[role]["queries"],
            "gt_video_cluster_count": role_counts[role]["gt_videos"],
        }
        for role in ("train_fit_mechanism_dev", "train_fit_confirm")
    }
    trusted_computation = _validate_trusted_power_computation_receipt(
        power_report.get("trusted_power_computation_receipt"),
        role_results=role_results,
        role_counts=trusted_role_counts,
        power_replay_capability_sha256=power_replay_capability_sha256,
        power_rows_sha256=power_snapshot["power_rows_sha256"],
        historical_reference_population_sha256=reference_population[
            "reference_population_sha256"
        ],
        power_evaluator_contract_sha256=power_snapshot[
            "power_evaluator_contract"
        ]["contract_sha256"],
        g2_committed_transaction_id=power_snapshot[
            "committed_transaction_id"
        ],
        g2_committed_state_sha256=power_snapshot["committed_state_sha256"],
        g2_committed_event_sha256=power_snapshot["committed_event_sha256"],
        goal_id=power_snapshot["goal_id"],
        attempt_id=power_snapshot["attempt_id"],
    )
    actual_reduction_count = _effect_threshold_reduction_count(
        role_results,
        minimum_effects,
    )
    if (
        actual_reduction_count != 0
        or power_report.get("effect_threshold_reduction_count")
        != actual_reduction_count
        or power_report.get("minimum_effect_registry_sha256")
        != MINIMUM_EFFECTS_REGISTRY_SHA256
        or power_report.get("effect_outcome_registry_sha256")
        != EFFECT_OUTCOME_REGISTRY_SHA256
    ):
        raise RoleContractError("computed power threshold registry/reduction drift")
    cluster_sizes: dict[str, int] = {}
    for record in records:
        cluster_sizes[record.gt_video_id] = cluster_sizes.get(record.gt_video_id, 0) + 1
    ordered_sizes = sorted(cluster_sizes.values())

    def percentile(fraction: float) -> int:
        index = min(
            len(ordered_sizes) - 1,
            max(0, int(math.ceil(fraction * len(ordered_sizes))) - 1),
        )
        return ordered_sizes[index]

    margins = {
        role: {
            "minimum_queries": minimum,
            "assigned_queries": role_counts[role]["queries"],
            "cluster_overshoot_queries": role_counts[role]["queries"] - minimum,
            "assigned_gt_video_clusters": role_counts[role]["gt_videos"],
        }
        for role, minimum in ROLE_MIN_QUERY_COUNTS
    }
    power_report_sha256 = semantic_sha256(dict(power_report))
    binding_sha256 = power_snapshot["power_replay_capability_binding_sha256"]
    base = {
        "schema_version": CLUSTER_POWER_AUDIT_SCHEMA,
        "status": "VALIDATED_CONTROL_POWER_CANDIDATE_AWAITING_STAGE_COMMIT",
        "role_lock_sha256": role_lock["lock_sha256"],
        "projected_query_count": len(records),
        "gt_video_cluster_count": len(cluster_sizes),
        "cluster_size_distribution": {
            "minimum": ordered_sizes[0],
            "p50_nearest_rank": percentile(0.50),
            "p90_nearest_rank": percentile(0.90),
            "maximum": ordered_sizes[-1],
            "mean": sum(ordered_sizes) / float(len(ordered_sizes)),
        },
        "role_minimum_margins": margins,
        "train_fit_core_queries": role_counts[CORE_ROLE]["queries"],
        "train_fit_core_gt_video_clusters": role_counts[CORE_ROLE]["gt_videos"],
        "minimum_query_counts": dict(ROLE_MIN_QUERY_COUNTS),
        "minimum_effects_ratio": minimum_effects,
        "historical_reference_population": dict(reference_population),
        "candidate_membership_assignment_inputs": (
            "TRAIN_FIT_GT_VIDEO_CLUSTERS_PLUS_FROZEN_SEEDED_DISJOINT_ROLE_"
            "LANE_ORDERS_PLUS_POWER_DETERMINED_PER_ROLE_PREFIX_LENGTH_BEFORE_"
            "MANIFEST_LOCK"
        ),
        "membership_identity_order_fixed_before_power_computation": True,
        "historical_outcomes_select_identity_order_or_within_prefix_members": False,
        "cross_role_historical_outcomes_affect_lane_identity_order_or_members": (
            False
        ),
        "historical_outcomes_may_determine_preregistered_prefix_length": True,
        "membership_lock_timing": "ALL_PREFIX_EXPANSION_BEFORE_ROLE_MANIFEST_LOCK",
        "expansion_lane_algorithm": POWER_EXPANSION_LANE_ALGORITHM,
        "frozen_role_lane_video_ids_sha256s": dict(
            power_report["frozen_role_lane_video_ids_sha256s"]
        ),
        "reserved_core_video_ids_sha256": power_report[
            "reserved_core_video_ids_sha256"
        ],
        "role_specific_expansion_lanes_disjoint": True,
        "power_result_use": (
            "PREREGISTERED_PER_ROLE_LANE_PREFIX_LENGTH_AND_STOP_POINT_ONLY_"
            "BEFORE_MANIFEST_LOCK"
        ),
        "power_results": {role: dict(result) for role, result in role_results.items()},
        "power_expansion_report": dict(power_report),
        "power_expansion_report_sha256": power_report_sha256,
        "power_evidence_binding": {
            "status": "CONTROLLER_RECOMPUTED_TRUSTED_POWER",
            "evidence_sha256": trusted_computation["receipt_sha256"],
            "power_replay_capability_binding_sha256": binding_sha256,
            "power_replay_capability_sha256": (
                power_replay_capability_sha256
            ),
            "trusted_power_computation_receipt": trusted_computation,
            "goal_id": power_snapshot["goal_id"],
            "attempt_id": power_snapshot["attempt_id"],
            "committed_transaction_id": power_snapshot["committed_transaction_id"],
            "committed_state_sha256": power_snapshot["committed_state_sha256"],
            "committed_event_sha256": power_snapshot["committed_event_sha256"],
            "g2_replay_control_receipt_sha256": power_snapshot[
                "g2_replay_control_receipt_sha256"
            ],
            "source_registry_sha256": power_snapshot["source_registry_sha256"],
            "power_rows_file_sha256": power_snapshot["power_rows_file_sha256"],
            "power_rows_sha256": power_snapshot["power_rows_sha256"],
            "power_row_count": power_snapshot["power_row_count"],
            "historical_reference_population_sha256": reference_population[
                "reference_population_sha256"
            ],
            "historical_reference_file_sha256": power_snapshot[
                "historical_reference_file_sha256"
            ],
            "replay_packet_sha256": power_snapshot["replay_packet_sha256"],
            "query_manifest_sha256": power_snapshot["query_manifest_sha256"],
            "power_contract_file_sha256": power_snapshot[
                "power_contract_file_sha256"
            ],
            "power_evaluator_contract_sha256": power_snapshot[
                "power_evaluator_contract"
            ]["contract_sha256"],
        },
        "effect_size_or_performance_claim": None,
        "interpretation": (
            "INDEPENDENT_HISTORICAL_REFERENCE_POWER_WITH_CANDIDATE_COUNTS_ONLY_"
            "NO_POSTHOC_EFFECT_REDUCTION_NO_IDENTITY_ORDER_OR_WITHIN_PREFIX_"
            "SELECTION_POWER_DETERMINES_PER_ROLE_LANE_PREFIX_LENGTH_ONLY_"
            "BEFORE_ROLE_MANIFEST_LOCK"
        ),
    }
    return {**base, "audit_sha256": semantic_sha256(base)}


def validate_g4_atomic_candidate(value: Any) -> dict[str, Any]:
    """Validate and detach the sole G4 atomic candidate return contract."""

    if not isinstance(value, Mapping):
        raise RoleContractError("G4 atomic candidate must be a mapping")
    candidate = json.loads(canonical_json_bytes(dict(value)).decode("utf-8"))
    base_fields = {
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
    }
    if set(candidate) != base_fields | {"atomic_candidate_sha256"}:
        raise RoleContractError("G4 atomic candidate field set is not exact")
    if (
        candidate.get("schema_version") != G4_ATOMIC_CANDIDATE_SCHEMA
        or candidate.get("status")
        != "VALIDATED_ATOMIC_G4_CANDIDATES_AWAITING_STAGE_COMMIT"
        or not isinstance(candidate.get("goal_id"), str)
        or not candidate["goal_id"]
        or not isinstance(candidate.get("attempt_id"), str)
        or not candidate["attempt_id"]
        or candidate.get("atomic_all_or_none") is not True
        or candidate.get("control_commit_required") is not True
        or candidate.get("persistence_claimed_by_pure_function") is not False
        or candidate.get("atomic_candidate_sha256")
        != semantic_sha256(
            candidate,
            excluded_fields=("atomic_candidate_sha256",),
        )
    ):
        raise RoleContractError("G4 atomic candidate identity/self-hash drift")
    role_snapshot_sha = _exact_sha256(
        candidate.get("role_input_snapshot_sha256"),
        "role_input_snapshot_sha256",
    )
    power_binding_sha = _exact_sha256(
        candidate.get("power_replay_capability_binding_sha256"),
        "power_replay_capability_binding_sha256",
    )
    power_capability_sha = _exact_sha256(
        candidate.get("power_replay_capability_sha256"),
        "power_replay_capability_sha256",
    )
    expected_roles = {role for role, _minimum in ROLE_MIN_QUERY_COUNTS} | {
        CORE_ROLE
    }
    manifests = candidate.get("role_manifests")
    if not isinstance(manifests, Mapping) or set(manifests) != expected_roles:
        raise RoleContractError("G4 atomic role manifest set is not exact")
    manifest_hashes: dict[str, str] = {}
    for role in sorted(expected_roles):
        manifest = manifests.get(role)
        if (
            not isinstance(manifest, Mapping)
            or manifest.get("schema_version") != ROLE_SCHEMA
            or manifest.get("role") != role
            or manifest.get("manifest_sha256")
            != semantic_sha256(
                dict(manifest),
                excluded_fields=("manifest_sha256",),
            )
        ):
            raise RoleContractError(f"G4 atomic role manifest drift: {role}")
        manifest_hashes[role] = manifest["manifest_sha256"]
    role_lock = candidate.get("role_lock")
    power_audit = candidate.get("cluster_power_audit")
    policy = candidate.get("formal_data_policy")
    if (
        not isinstance(role_lock, Mapping)
        or role_lock.get("schema_version") != ROLE_LOCK_SCHEMA
        or role_lock.get("status")
        != "VALIDATED_CONTROL_INPUT_CANDIDATE_AWAITING_STAGE_COMMIT"
        or role_lock.get("lock_sha256")
        != semantic_sha256(dict(role_lock), excluded_fields=("lock_sha256",))
    ):
        raise RoleContractError("G4 atomic role-lock drift")
    if (
        not isinstance(power_audit, Mapping)
        or power_audit.get("schema_version") != CLUSTER_POWER_AUDIT_SCHEMA
        or power_audit.get("status")
        != "VALIDATED_CONTROL_POWER_CANDIDATE_AWAITING_STAGE_COMMIT"
        or power_audit.get("audit_sha256")
        != semantic_sha256(dict(power_audit), excluded_fields=("audit_sha256",))
        or power_audit.get("role_lock_sha256") != role_lock["lock_sha256"]
    ):
        raise RoleContractError("G4 atomic power-audit drift")
    if (
        not isinstance(policy, Mapping)
        or policy.get("schema_version") != FORMAL_POLICY_SCHEMA
        or policy.get("status")
        != "VALIDATED_CONTROL_POLICY_CANDIDATE_AWAITING_STAGE_COMMIT"
        or policy.get("policy_sha256")
        != semantic_sha256(dict(policy), excluded_fields=("policy_sha256",))
        or policy.get("role_lock_sha256") != role_lock["lock_sha256"]
        or policy.get("cluster_power_audit_sha256")
        != power_audit["audit_sha256"]
        or policy.get("policy") != FORMAL_POLICY
        or policy.get("U_formal_updates") != U_FORMAL_UPDATES
    ):
        raise RoleContractError("G4 atomic formal-policy drift")
    control_input = role_lock.get("control_input_binding")
    power_lineage = power_audit.get("power_evidence_binding")
    if (
        not isinstance(control_input, Mapping)
        or control_input.get("goal_id") != candidate["goal_id"]
        or control_input.get("attempt_id") != candidate["attempt_id"]
        or control_input.get("role_input_snapshot_sha256")
        != role_snapshot_sha
        or not isinstance(power_lineage, Mapping)
        or power_lineage.get("goal_id") != candidate["goal_id"]
        or power_lineage.get("attempt_id") != candidate["attempt_id"]
        or power_lineage.get("power_replay_capability_binding_sha256")
        != power_binding_sha
        or power_lineage.get("power_replay_capability_sha256")
        != power_capability_sha
    ):
        raise RoleContractError("G4 atomic control lineage drift")
    reference_population = power_audit.get("historical_reference_population")
    power_results = power_audit.get("power_results")
    power_expansion_report = power_audit.get("power_expansion_report")
    role_counts = role_lock.get("role_counts")
    if (
        not isinstance(reference_population, Mapping)
        or reference_population.get("schema_version")
        != "c28f_g2_historical_power_reference_v1"
        or reference_population.get("reference_population_sha256")
        != semantic_sha256(
            dict(reference_population),
            excluded_fields=("reference_population_sha256",),
        )
        or reference_population.get("power_rows_sha256")
        != power_lineage.get("power_rows_sha256")
        or reference_population.get("power_row_count")
        != power_lineage.get("power_row_count")
        or reference_population.get("replay_packet_sha256")
        != power_lineage.get("replay_packet_sha256")
        or reference_population.get("query_manifest_sha256")
        != power_lineage.get("query_manifest_sha256")
        or reference_population.get("source_registry_sha256")
        != power_lineage.get("source_registry_sha256")
        or reference_population.get("reference_population_sha256")
        != power_lineage.get("historical_reference_population_sha256")
        or not isinstance(
            power_lineage.get("historical_reference_file_sha256"),
            str,
        )
        or _SHA256_RE.fullmatch(
            power_lineage["historical_reference_file_sha256"]
        )
        is None
        or reference_population.get("train_fit_desc_id_join_performed") is not False
        or reference_population.get("train_fit_video_id_join_performed") is not False
        or reference_population.get("historical_performance_selects_membership")
        is not False
        or not isinstance(power_expansion_report, Mapping)
        or power_audit.get("power_expansion_report_sha256")
        != semantic_sha256(dict(power_expansion_report))
        or power_expansion_report.get("status") != "PASS"
        or power_expansion_report.get("historical_reference_population")
        != reference_population
        or power_expansion_report.get("role_results") != power_results
        or power_expansion_report.get(
            "trusted_power_computation_receipt"
        )
        != power_lineage.get("trusted_power_computation_receipt")
        or power_audit.get(
            "membership_identity_order_fixed_before_power_computation"
        ) is not True
        or power_audit.get(
            "historical_outcomes_select_identity_order_or_within_prefix_members"
        ) is not False
        or power_audit.get(
            "cross_role_historical_outcomes_affect_lane_identity_order_or_members"
        ) is not False
        or power_audit.get(
            "historical_outcomes_may_determine_preregistered_prefix_length"
        ) is not True
        or power_audit.get("membership_lock_timing")
        != "ALL_PREFIX_EXPANSION_BEFORE_ROLE_MANIFEST_LOCK"
        or power_audit.get("expansion_lane_algorithm")
        != POWER_EXPANSION_LANE_ALGORITHM
        or not isinstance(
            power_audit.get("frozen_role_lane_video_ids_sha256s"),
            Mapping,
        )
        or set(power_audit["frozen_role_lane_video_ids_sha256s"])
        != {"train_fit_mechanism_dev", "train_fit_confirm"}
        or any(
            not isinstance(value, str)
            or _SHA256_RE.fullmatch(value) is None
            for value in power_audit[
                "frozen_role_lane_video_ids_sha256s"
            ].values()
        )
        or not isinstance(
            power_audit.get("reserved_core_video_ids_sha256"),
            str,
        )
        or _SHA256_RE.fullmatch(
            power_audit["reserved_core_video_ids_sha256"]
        ) is None
        or power_audit.get("role_specific_expansion_lanes_disjoint") is not True
        or power_expansion_report.get(
            "cross_role_historical_outcomes_affect_lane_identity_order_or_members"
        )
        != power_audit.get(
            "cross_role_historical_outcomes_affect_lane_identity_order_or_members"
        )
        or power_expansion_report.get("expansion_lane_algorithm")
        != power_audit.get("expansion_lane_algorithm")
        or power_expansion_report.get(
            "frozen_role_lane_video_ids_sha256s"
        )
        != power_audit.get("frozen_role_lane_video_ids_sha256s")
        or power_expansion_report.get("reserved_core_video_ids_sha256")
        != power_audit.get("reserved_core_video_ids_sha256")
        or power_expansion_report.get(
            "role_specific_expansion_lanes_disjoint"
        )
        is not True
        or power_audit.get("power_result_use")
        != (
            "PREREGISTERED_PER_ROLE_LANE_PREFIX_LENGTH_AND_STOP_POINT_ONLY_"
            "BEFORE_MANIFEST_LOCK"
        )
        or power_audit.get("candidate_membership_assignment_inputs")
        != (
            "TRAIN_FIT_GT_VIDEO_CLUSTERS_PLUS_FROZEN_SEEDED_DISJOINT_ROLE_"
            "LANE_ORDERS_PLUS_POWER_DETERMINED_PER_ROLE_PREFIX_LENGTH_BEFORE_"
            "MANIFEST_LOCK"
        )
        or power_audit.get("interpretation")
        != (
            "INDEPENDENT_HISTORICAL_REFERENCE_POWER_WITH_CANDIDATE_COUNTS_ONLY_"
            "NO_POSTHOC_EFFECT_REDUCTION_NO_IDENTITY_ORDER_OR_WITHIN_PREFIX_"
            "SELECTION_POWER_DETERMINES_PER_ROLE_LANE_PREFIX_LENGTH_ONLY_"
            "BEFORE_ROLE_MANIFEST_LOCK"
        )
        or not isinstance(power_results, Mapping)
        or set(power_results)
        != {"train_fit_mechanism_dev", "train_fit_confirm"}
        or not isinstance(role_counts, Mapping)
    ):
        raise RoleContractError("G4 atomic historical-reference power drift")
    for role, manifest in manifests.items():
        assignment = manifest.get("assignment")
        if (
            not isinstance(assignment, Mapping)
            or assignment.get("candidate_membership_source")
            != (
                "TRAIN_FIT_GT_VIDEO_CLUSTERS_PLUS_FROZEN_SEEDED_DISJOINT_ROLE_"
                "LANE_ORDERS_PLUS_POWER_DETERMINED_PER_ROLE_PREFIX_LENGTH_"
                "BEFORE_MANIFEST_LOCK"
            )
            or assignment.get("historical_reference_population_sha256")
            != reference_population["reference_population_sha256"]
            or assignment.get(
                "historical_outcomes_select_identity_order_or_within_prefix_members"
            ) is not False
            or assignment.get(
                "cross_role_historical_outcomes_affect_lane_identity_order_or_members"
            ) is not False
            or assignment.get(
                "historical_outcomes_may_determine_preregistered_prefix_length"
            ) is not True
            or assignment.get("power_expansion_lane_algorithm")
            != POWER_EXPANSION_LANE_ALGORITHM
            or assignment.get("power_expansion_report_sha256")
            != power_audit.get("power_expansion_report_sha256")
            or assignment.get("membership_lock_timing")
            != "ALL_PREFIX_EXPANSION_BEFORE_ROLE_MANIFEST_LOCK"
            or assignment.get("assignment_contract_sha256")
            != semantic_sha256(
                dict(assignment),
                excluded_fields=("assignment_contract_sha256",),
            )
        ):
            raise RoleContractError(
                f"G4 atomic membership/reference assignment drift: {role}"
            )
    for role in ("train_fit_mechanism_dev", "train_fit_confirm"):
        result = power_results.get(role)
        count = role_counts.get(role)
        if not isinstance(count, Mapping):
            raise RoleContractError("G4 atomic role-size power input drift")
        _validate_power_result_evidence(
            result,
            role=role,
            expected_query_count=count.get("queries"),
            expected_cluster_count=count.get("gt_videos"),
            reference_population=reference_population,
        )
        if result.get("status") != "PASS":
            raise RoleContractError("G4 atomic role-size power input drift")
    trusted_role_counts = {
        role: {
            "query_count": role_counts[role]["queries"],
            "gt_video_cluster_count": role_counts[role]["gt_videos"],
        }
        for role in ("train_fit_mechanism_dev", "train_fit_confirm")
    }
    trusted_computation = _validate_trusted_power_computation_receipt(
        power_lineage.get("trusted_power_computation_receipt"),
        role_results=power_results,
        role_counts=trusted_role_counts,
        power_replay_capability_sha256=power_capability_sha,
        power_rows_sha256=power_lineage["power_rows_sha256"],
        historical_reference_population_sha256=power_lineage[
            "historical_reference_population_sha256"
        ],
        power_evaluator_contract_sha256=power_lineage[
            "power_evaluator_contract_sha256"
        ],
        g2_committed_transaction_id=power_lineage[
            "committed_transaction_id"
        ],
        g2_committed_state_sha256=power_lineage["committed_state_sha256"],
        g2_committed_event_sha256=power_lineage["committed_event_sha256"],
        goal_id=power_lineage["goal_id"],
        attempt_id=power_lineage["attempt_id"],
    )
    persisted_computation = (
        _require_persisted_trusted_power_computation_receipt(
            trusted_computation
        )
    )
    if canonical_json_bytes(persisted_computation) != canonical_json_bytes(
        trusted_computation
    ):
        raise RoleContractError(
            "persisted trusted power computation receipt payload drift"
        )
    artifact_hashes = candidate.get("artifact_candidate_hashes")
    expected_artifact_hashes = {
        "role_manifest_hashes": manifest_hashes,
        "role_lock_sha256": role_lock["lock_sha256"],
        "cluster_power_audit_sha256": power_audit["audit_sha256"],
        "formal_data_policy_sha256": policy["policy_sha256"],
    }
    if artifact_hashes != expected_artifact_hashes:
        raise RoleContractError("G4 atomic artifact hash closure drift")
    return candidate


def build_g4_role_policy_candidates(
    *,
    role_input_handle: Any,
    power_replay_capability: Any,
) -> dict[str, Any]:
    """Build the only atomic G4 candidate bundle from two exact one-shot handles."""

    _validated_effect_registries()
    from .a4_control import (
        validate_g2_power_computation_receipt,
        validate_g2_power_replay_capability,
        validate_g4_role_lock_input_handle,
    )

    validated_role_snapshot = validate_g4_role_lock_input_handle(role_input_handle)
    power_snapshot = validate_g2_power_replay_capability(power_replay_capability)
    if (
        validated_role_snapshot.get("goal_id") != power_snapshot.get("goal_id")
        or validated_role_snapshot.get("attempt_id") != power_snapshot.get("attempt_id")
    ):
        raise RoleContractError("G4 role/power capabilities do not share goal and attempt")
    role_snapshot = role_input_handle.consume_snapshot_once()
    if role_snapshot != validated_role_snapshot:
        raise RoleContractError("G4 role input changed during one-shot consumption")
    power_rows = power_replay_capability.consume_rows_once()
    if (
        len(power_rows) != power_snapshot["power_row_count"]
        or hashlib.sha256(canonical_json_bytes(list(power_rows))).hexdigest()
        != power_snapshot["power_rows_sha256"]
    ):
        raise RoleContractError("G2 power rows do not match the opaque capability binding")
    frozen_contract = frozen_power_evaluator_contract()
    provided_contract = power_snapshot.get("power_evaluator_contract")
    if (
        not isinstance(provided_contract, Mapping)
        or canonical_json_bytes(dict(provided_contract))
        != canonical_json_bytes(frozen_contract)
        or provided_contract.get("contract_sha256")
        != semantic_sha256(
            dict(provided_contract), excluded_fields=("contract_sha256",)
        )
        or provided_contract.get("contract_sha256")
        != frozen_contract["contract_sha256"]
    ):
        raise RoleContractError("G2 power evaluator/derivation contract drift")
    derivation = frozen_contract["outcome_derivation_contract"]
    if derivation.get("outcome_derivation_contract_sha256") != frozen_contract.get(
        "outcome_derivation_contract_sha256"
    ):
        raise RoleContractError("G2 outcome derivation contract hash drift")
    records_raw = role_snapshot.get("train_fit_mapping_records")
    if not isinstance(records_raw, list) or not records_raw:
        raise RoleContractError("G4 train-fit mapping is empty")
    records: list[RoleRecord] = []
    for index, raw in enumerate(records_raw):
        if not isinstance(raw, Mapping) or set(raw) != {
            "desc_id",
            "ground_truth_video_id",
        }:
            raise RoleContractError(f"G4 train-fit mapping row is not exact: {index}")
        records.append(RoleRecord(raw["desc_id"], raw["ground_truth_video_id"]))
    expected_desc_ids = [record.desc_id for record in records]
    if expected_desc_ids != sorted(set(expected_desc_ids)):
        raise RoleContractError("G4 train-fit mapping order/uniqueness drift")
    reference_by_desc = _normalize_power_replay_rows(
        power_rows,
        query_manifest_sha256=power_snapshot["query_manifest_sha256"],
        replay_packet_sha256=power_snapshot["replay_packet_sha256"],
    )
    recomputed_reference_population = build_historical_power_reference_from_rows(
        power_rows,
        power_rows_sha256=power_snapshot["power_rows_sha256"],
        replay_packet_sha256=power_snapshot["replay_packet_sha256"],
        query_manifest_sha256=power_snapshot["query_manifest_sha256"],
        source_registry_sha256=power_snapshot["source_registry_sha256"],
    )
    reference_population = validate_historical_power_reference_population(
        power_snapshot.get("historical_reference_population")
    )
    _exact_sha256(
        power_snapshot.get("historical_reference_file_sha256"),
        "historical_reference_file_sha256",
    )
    if canonical_json_bytes(reference_population) != canonical_json_bytes(
        recomputed_reference_population
    ):
        raise RoleContractError(
            "committed G2 historical reference differs from consumed power rows"
        )
    assignments, power_report = _power_adaptive_assignments(
        records,
        expected_desc_ids,
        reference_by_desc=reference_by_desc,
        reference_population=reference_population,
    )
    _assert_power_prefix_expansion(
        records=records,
        expected_desc_ids=expected_desc_ids,
        final_assignments=assignments,
        power_report=power_report,
    )
    if power_report.get("status") != "PASS":
        raise RoleContractError("G4 power insufficient; effect thresholds may not be lowered")
    role_results = power_report.get("role_results")
    trusted_role_counts = {
        role: {
            "query_count": len(assignments[role]),
            "gt_video_cluster_count": len(
                {record.gt_video_id for record in assignments[role]}
            ),
        }
        for role in ("train_fit_mechanism_dev", "train_fit_confirm")
    }
    computation_capability = (
        power_replay_capability.issue_trusted_power_computation_receipt_once(
            role_counts=trusted_role_counts,
        )
    )
    trusted_computation = validate_g2_power_computation_receipt(
        computation_capability
    )
    trusted_computation = _validate_trusted_power_computation_receipt(
        trusted_computation,
        role_results=role_results,
        role_counts=trusted_role_counts,
        power_replay_capability_sha256=(
            power_replay_capability.capability_sha256
        ),
        power_rows_sha256=power_snapshot["power_rows_sha256"],
        historical_reference_population_sha256=reference_population[
            "reference_population_sha256"
        ],
        power_evaluator_contract_sha256=frozen_contract["contract_sha256"],
        g2_committed_transaction_id=power_snapshot[
            "committed_transaction_id"
        ],
        g2_committed_state_sha256=power_snapshot["committed_state_sha256"],
        g2_committed_event_sha256=power_snapshot["committed_event_sha256"],
        goal_id=power_snapshot["goal_id"],
        attempt_id=power_snapshot["attempt_id"],
    )
    trusted_computation = (
        _require_persisted_trusted_power_computation_receipt(
            trusted_computation
        )
    )
    power_report = {
        **dict(power_report),
        "trusted_power_computation_receipt": trusted_computation,
    }
    power_report_sha256 = semantic_sha256(dict(power_report))
    assignment_contract_base = {
        "algorithm": POWER_ADAPTIVE_ASSIGNMENT_ALGORITHM,
        "seed": ASSIGNMENT_SEED,
        "cluster_order": "sha256(seed_ascii + NUL + video_id_utf8),video_id_utf8",
        "cluster_atomic": True,
        "candidate_membership_source": (
            "TRAIN_FIT_GT_VIDEO_CLUSTERS_PLUS_FROZEN_SEEDED_DISJOINT_ROLE_"
            "LANE_ORDERS_PLUS_POWER_DETERMINED_PER_ROLE_PREFIX_LENGTH_BEFORE_"
            "MANIFEST_LOCK"
        ),
        "historical_reference_population_sha256": reference_population[
            "reference_population_sha256"
        ],
        "historical_outcomes_select_identity_order_or_within_prefix_members": False,
        "cross_role_historical_outcomes_affect_lane_identity_order_or_members": (
            False
        ),
        "historical_outcomes_may_determine_preregistered_prefix_length": True,
        "power_expansion_lane_algorithm": POWER_EXPANSION_LANE_ALGORITHM,
        "membership_lock_timing": "ALL_PREFIX_EXPANSION_BEFORE_ROLE_MANIFEST_LOCK",
        "power_expansion_report_sha256": power_report_sha256,
    }
    assignment_contract = {
        **assignment_contract_base,
        "assignment_contract_sha256": semantic_sha256(assignment_contract_base),
    }
    manifests = build_role_manifests(
        assignments,
        train_fit_manifest_sha256=role_snapshot["train_fit_mapping_sha256"],
        assignment_source_sha256=role_snapshot["role_input_snapshot_sha256"],
        producer_control_binding_sha256=power_snapshot[
            "power_replay_capability_binding_sha256"
        ],
        assignment_contract=assignment_contract,
    )
    protected_desc_ids = role_snapshot.get("protected_desc_ids")
    protected_video_ids = role_snapshot.get("protected_video_ids")
    if (
        not isinstance(protected_desc_ids, list)
        or protected_desc_ids != sorted(set(protected_desc_ids))
        or any(type(value) is not int or value < 0 for value in protected_desc_ids)
        or not isinstance(protected_video_ids, list)
        or any(
            not isinstance(value, str) or _VIDEO_ID_RE.fullmatch(value) is None
            for value in protected_video_ids
        )
        or protected_video_ids
        != sorted(set(protected_video_ids), key=lambda value: value.encode("utf-8"))
    ):
        raise RoleContractError("G4 protected entity manifests are not canonical")
    lock_candidate = _build_role_lock_candidate_from_validated_fields(
        manifests,
        expected_train_fit_desc_ids=expected_desc_ids,
        projected_records=records,
        protected_desc_ids=set(protected_desc_ids),
        protected_video_ids=set(protected_video_ids),
        protected_mapping_receipt_sha256=role_snapshot[
            "projection_completion_capability_sha256"
        ],
        expected_assignments=assignments,
        expected_assignment_contract=assignment_contract,
    )
    role_lock = _control_bound_role_lock(lock_candidate, role_snapshot)
    power_audit = _computed_cluster_power_audit(
        role_lock=role_lock,
        records=records,
        power_report=power_report,
        power_snapshot=power_snapshot,
        power_replay_capability_sha256=(
            power_replay_capability.capability_sha256
        ),
    )
    formal_policy = _build_formal_data_policy_from_control_candidates(
        role_lock=role_lock,
        cluster_power_audit=power_audit,
    )
    role_manifest_hashes = {
        role: manifests[role]["manifest_sha256"] for role in sorted(manifests)
    }
    artifact_hashes = {
        "role_manifest_hashes": role_manifest_hashes,
        "role_lock_sha256": role_lock["lock_sha256"],
        "cluster_power_audit_sha256": power_audit["audit_sha256"],
        "formal_data_policy_sha256": formal_policy["policy_sha256"],
    }
    base = {
        "schema_version": G4_ATOMIC_CANDIDATE_SCHEMA,
        "status": "VALIDATED_ATOMIC_G4_CANDIDATES_AWAITING_STAGE_COMMIT",
        "goal_id": role_snapshot["goal_id"],
        "attempt_id": role_snapshot["attempt_id"],
        "role_input_snapshot_sha256": role_snapshot["role_input_snapshot_sha256"],
        "power_replay_capability_binding_sha256": power_snapshot[
            "power_replay_capability_binding_sha256"
        ],
        "power_replay_capability_sha256": (
            power_replay_capability.capability_sha256
        ),
        "role_manifests": manifests,
        "role_lock": role_lock,
        "cluster_power_audit": power_audit,
        "formal_data_policy": formal_policy,
        "artifact_candidate_hashes": artifact_hashes,
        "atomic_all_or_none": True,
        "control_commit_required": True,
        "persistence_claimed_by_pure_function": False,
    }
    return validate_g4_atomic_candidate(
        {**base, "atomic_candidate_sha256": semantic_sha256(base)}
    )


def build_role_lock(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
    raise RoleContractError(
        "legacy partial role-lock API disabled; use build_g4_role_policy_candidates"
    )


def build_cluster_power_audit(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
    raise RoleContractError(
        "caller-signed power evidence API disabled; use build_g4_role_policy_candidates"
    )


def build_formal_data_policy(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
    raise RoleContractError(
        "standalone policy API disabled; use build_g4_role_policy_candidates"
    )


def canonical_role_artifact_bytes(value: Mapping[str, Any]) -> bytes:
    return canonical_json_bytes(dict(value)) + b"\n"


__all__ = [
    "ASSIGNMENT_SEED",
    "CORE_ROLE",
    "FORMAL_POLICY",
    "G4_ATOMIC_CANDIDATE_SCHEMA",
    "POWER_OUTCOME_KEYS",
    "POWER_REPLAY_ROW_SCHEMA",
    "RoleContractError",
    "RoleRecord",
    "U_FORMAL_UPDATES",
    "build_historical_power_reference_from_rows",
    "build_g4_role_policy_candidates",
    "canonical_role_artifact_bytes",
    "frozen_power_evaluator_contract",
    "validate_g4_atomic_candidate",
    "validate_historical_power_reference_population",
]
