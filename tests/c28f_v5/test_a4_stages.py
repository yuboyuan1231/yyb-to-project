from __future__ import annotations

import copy
import hashlib
import inspect
import json
import math
from pathlib import Path

import pytest

from blueprint_e2e_v2.c28f_v5 import a4_control
from blueprint_e2e_v2.c28f_v5 import a4_metrics
from blueprint_e2e_v2.c28f_v5 import a4_roles
from blueprint_e2e_v2.c28f_v5 import a4_stages
from blueprint_e2e_v2.c28f_v5 import a4_f1
from blueprint_e2e_v2.c28f_v5.a4_control import QUERY_BUCKET_RULE_CONTRACT
from blueprint_e2e_v2.c28f_v5.canonical import canonical_json_bytes, semantic_sha256
from blueprint_e2e_v2.c28f_v5.a4_security import (
    EvalBudgetBook,
    EvalBudgetToken,
)
from blueprint_e2e_v2.c28f_v5.a4_f1 import (
    F1ContractError,
    run_synthetic_atomic_recovery_harness,
)
from blueprint_e2e_v2.c28f_v5.a4_stages import (
    G1_ACTION,
    G1_NEGATIVE_SUITE_SHA256,
    G4_PROJECTION_DECLARED_SAFETY_COUNTS,
    G4_PROJECTION_ACTION,
    G4_ROLE_DECLARED_SAFETY_COUNTS,
    POWER_OUTCOME_KEYS,
    ReplayExecution,
    StageBundle,
    StageContext,
    StageContractError,
    build_g1_bundle,
    build_g2_bundle,
    build_g3_bundle,
    build_g4_bundle,
    build_g4_projection_bundle,
    build_g5_bundle,
    build_g6_f1_completion_bundle,
    execute_ledgered_c7_replay_alignment,
    submit_stage_bundle,
    _derive_g2_power_outcome_row,
    _frozen_retrieval_gate_result,
    _build_f0_bucket_report,
    _evaluate_completed_f0_rows,
    _rank_transitions_parquet,
    _validate_g4_projection_open_classification,
    _validate_g4_atomic_candidates,
)


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _context(output: Path) -> StageContext:
    state_sha = _sha("committed-state")
    event_sha = _sha("committed-event")
    return StageContext(
        goal_id="goal-c28f-test",
        attempt_id="G0-A4-test",
        transition_seq=1,
        current_action=G1_ACTION,
        current_status="PREFLIGHT_OK",
        parent_state_sha256=state_sha,
        parent_event_sha256=event_sha,
        business_preimage_sha256=_sha("business-preimage"),
        static_review_receipt_sha256=_sha("static-review"),
        isolated_output_root_id="isolated-output",
        isolated_output_root_path=str(output),
        expected_negative_suite_sha256=G1_NEGATIVE_SUITE_SHA256,
        control_transaction_id="G0-A4-TXN-test",
        committed_control_state_sha256=state_sha,
        committed_control_event_sha256=event_sha,
        frozen_g1_registry_authority_sha256=_sha("g1-authority"),
        frozen_global_ledger_snapshot_sha256=_sha("global-snapshot"),
        control_anchor_receipt_sha256=_sha("control-anchor"),
        train_jsonl_index_state="ABSENT",
        train_jsonl_index_binding_sha256=None,
        train_jsonl_index_receipt_sha256=None,
        train_jsonl_index_transaction_id=None,
    )


def _ready_index_fields() -> dict:
    return {
        "train_jsonl_index_state": "READY",
        "train_jsonl_index_binding_sha256": _sha("train-index-binding"),
        "train_jsonl_index_receipt_sha256": _sha("train-index-receipt"),
        "train_jsonl_index_transaction_id": (
            "TRAIN-INDEX-%s" % _sha("train-index-transaction")[:32]
        ),
    }


def _post_index_context(
    output: Path,
    *,
    action: str,
    status: str,
) -> StageContext:
    return StageContext(
        **{
            **_context(output).__dict__,
            "current_action": action,
            "current_status": status,
            **_ready_index_fields(),
        }
    )


def _budget() -> EvalBudgetBook:
    return EvalBudgetBook(
        tokens=(
            EvalBudgetToken(
                token_id="F0_A",
                purpose="F0_A_NONPROTECTED_METADATA_DRY_RUN",
                state="UNISSUED",
                max_uses=1,
                use_count=0,
            ),
        )
    )


def _f0_completed_row(
    query_id: int,
    *,
    gt_present: bool,
    top1_iou: float,
) -> dict:
    pooled = ["video_%04d" % index for index in range(1_000)]
    late = list(pooled[:200])
    final = list(late)
    gt_video_id = pooled[0] if gt_present else "video_missing_%04d" % query_id
    wrong_video = not gt_present
    correct_video_wrong_span = gt_present and top1_iou < 0.5
    top1_false_positive = wrong_video or top1_iou < 0.3
    vcmr_hits = {
        "VCMR_R@%d_IoU@%.1f" % (k, threshold): (
            gt_present and top1_iou >= threshold
        )
        for threshold in (0.5, 0.7)
        for k in (1, 5, 10, 100)
    }

    def mask(valid_count: int, *, empty_candidate_count: int = 0) -> dict:
        missing_count = 12_800 - valid_count
        return {
            "candidate_count": 200,
            "denominator": 12_800,
            "valid_count": valid_count,
            "missing_count": missing_count,
            "valid_ratio": valid_count / 12_800.0,
            "missing_ratio": missing_count / 12_800.0,
            "empty_candidate_count": empty_candidate_count,
        }

    mask_structure_base = {
        "schema_version": "c28f_a4_forward_mask_structure_contract_v1",
        "visual_candidate_rule": (
            "EVERY_CANDIDATE_HAS_AT_LEAST_ONE_VALID_CELL"
        ),
        "subtitle_candidate_rule": (
            "ZERO_VALID_CELLS_PER_CANDIDATE_AND_ALL_ZERO_QUERY_ALLOWED"
        ),
        "joint_component_rule": "EXACT_LOGICAL_OR_VISUAL_SUBTITLE",
        "joint_candidate_rule": (
            "EVERY_CANDIDATE_HAS_AT_LEAST_ONE_VALID_CELL"
        ),
    }
    mask_structure_contract = {
        **mask_structure_base,
        "contract_sha256": semantic_sha256(mask_structure_base),
    }

    row = {
        "query_id": query_id,
        "gt_video_id": gt_video_id,
        "pooled_video_order": pooled,
        "late_video_order": late,
        "final_video_order": final,
        "query_type": "vt",
        "query_feature_token_count": 7,
        "query_text_token_count": 9,
        "verb_bucket": "SINGLE_VERB",
        "temporal_connector_bool": True,
        "late_maxsim_token_count": 7,
        "late_candidate_mask_coverage": {
            "ratio_definition": (
                "VALID_OR_MISSING_BOOL_CELLS_DIVIDED_BY_"
                "CANDIDATE_COUNT_TIMES_64"
            ),
            "structure_contract": mask_structure_contract,
            "joint_component_relation_verified": True,
            "visual_subtitle_overlap_valid_count": 9_000,
            "visual": mask(12_000),
            "subtitle": mask(9_600, empty_candidate_count=10),
            "joint": mask(12_600),
        },
        "moment_duration_sec": 4.0,
        "gt_video_clip_count": 80,
        "gt_relative_position": 0.5,
        "joint_marginal_gt_rank": 1 if gt_present else None,
        "legacy_joint_unique_video_gt_rank": 1 if gt_present else None,
        "vcmr_hits": vcmr_hits,
        "top1_iou": top1_iou if gt_present else 0.0,
        "top1_available": True,
        "wrong_video_top1": wrong_video,
        "correct_video_wrong_span_top1": correct_video_wrong_span,
        "top1_false_positive": top1_false_positive,
        "joint_false_positive_mass": 0.2 if gt_present else 0.9,
        "raw_exact_duplicate_proposal_count": 0,
        "nms_prediction_count": 100,
    }
    for prefix in ("pooled", "late", "final"):
        row["%s_top1_top2_score_margin" % prefix] = 0.30
        row["%s_top10_top11_score_margin" % prefix] = 0.20
        row["%s_top100_top101_score_margin" % prefix] = 0.10
        row["%s_gt_vs_best_wrong_score_margin" % prefix] = (
            0.25 if gt_present else None
        )
    return row


def test_stage_context_requires_exact_committed_state_event(tmp_path: Path) -> None:
    context = _context(tmp_path / "output")
    context.validate(G1_ACTION, "PREFLIGHT_OK")
    forged = StageContext(
        **{
            **context.__dict__,
            "committed_control_event_sha256": _sha("other-event"),
        }
    )
    with pytest.raises(StageContractError, match="exact committed"):
        forged.validate(G1_ACTION, "PREFLIGHT_OK")


def test_stage_context_train_index_dual_state_is_exact(tmp_path: Path) -> None:
    absent = _context(tmp_path / "absent-output")
    absent.validate(G1_ACTION, "PREFLIGHT_OK")
    assert absent.binding()["train_jsonl_index_state"] == "ABSENT"
    assert absent.binding()["train_jsonl_index_binding_sha256"] is None

    mixed_absent = StageContext(
        **{
            **absent.__dict__,
            "train_jsonl_index_binding_sha256": _sha("forbidden-index-binding"),
        }
    )
    with pytest.raises(StageContractError, match="ABSENT"):
        mixed_absent.validate(G1_ACTION, "PREFLIGHT_OK")

    ready = _post_index_context(
        tmp_path / "ready-output",
        action="G3_F0_A_BASELINE_FORENSICS",
        status="METRIC_CONTRACT_OK",
    )
    ready.validate("G3_F0_A_BASELINE_FORENSICS", "METRIC_CONTRACT_OK")
    ready_binding = ready.binding()
    assert ready_binding["train_jsonl_index_state"] == "READY"
    assert ready_binding["train_jsonl_index_binding_sha256"] == _sha(
        "train-index-binding"
    )
    assert ready_binding["train_jsonl_index_receipt_sha256"] == _sha(
        "train-index-receipt"
    )
    assert ready_binding["train_jsonl_index_transaction_id"].startswith(
        "TRAIN-INDEX-"
    )

    bad_transaction = StageContext(
        **{
            **ready.__dict__,
            "train_jsonl_index_transaction_id": "TRAIN-INDEX-not-hex",
        }
    )
    with pytest.raises(StageContractError, match=r"READY\+TRAIN-INDEX"):
        bad_transaction.validate(
            "G3_F0_A_BASELINE_FORENSICS",
            "METRIC_CONTRACT_OK",
        )


def test_g1_rejects_caller_supplied_mapping_instead_of_control_handle(
    tmp_path: Path,
) -> None:
    with pytest.raises(StageContractError, match="A4G1ControlHandle"):
        build_g1_bundle(
            context=_context(tmp_path / "output"),
            g1_control_handle={
                "registry_authority": {},
                "global_snapshot": {},
            },
            budget=_budget(),
        )


def test_g1_consumes_protected_manifest_and_nested_ledger_schema() -> None:
    assert a4_stages.Path is Path
    registry_source = inspect.getsource(a4_stages._registry_from_g0_authority)
    receipt_source = inspect.getsource(
        a4_stages._validate_g1_observation_scope_receipt
    )
    assert '"protected_id_mapping_manifest_contract"' in registry_source
    assert "approved_content_sha256=approved_content_sha256" in registry_source
    assert '"nested_ledgers_were_open"' in receipt_source
    assert '"nested_ledger_cleanup_failed"' in receipt_source
    assert '"nested_ledger_cleanup_errors"' in receipt_source


def test_g1_persisted_snapshot_keeps_genesis_anchor_after_code_changes() -> None:
    source = inspect.getsource(a4_stages._validate_persisted_global_snapshot)
    assert 'r"G0-A4-TXN-[0-9a-f]{32}"' in source
    assert (
        'current["persistence_transaction_id"]\n'
        '        == current["committed_control_transaction_id"]'
        in source
    )
    assert "== context.control_transaction_id" not in source


def test_g2_rejects_duck_typed_replay_factory() -> None:
    assert a4_stages.G2_TRAIN_JSONL_INDEX_BOOTSTRAP_ACTION == (
        "G2_TRAIN_JSONL_INDEX_BOOTSTRAP"
    )

    class FakeFactory:
        def preopen_receipt(self):
            return {}

    with pytest.raises(StageContractError, match="A4G2ReplayControlHandle"):
        execute_ledgered_c7_replay_alignment(
            control_handle=FakeFactory(),
            expected_goal_id="goal-c28f-test",
            expected_query_manifest_sha256=_sha("queries"),
            expected_source_hashes={"artifact:predictions": _sha("predictions")},
        )


def test_g2_rejects_jointly_self_signed_fixture_mappings(tmp_path: Path) -> None:
    context = StageContext(
        **{
            **_context(tmp_path / "output").__dict__,
            "current_action": "G2_METRIC_CONTRACT",
            "current_status": "FIREWALL_OK",
        }
    )
    with pytest.raises(StageContractError, match="reviewed canonical bytes"):
        build_g2_bundle(
            context=context,
            fixture={"schema_version": "caller-forged", "value": 1},
            hand_expected_fixture={
                "schema_version": "caller-forged-expected",
                "value": 1,
            },
            g1_safety_manifest_sha256=_sha("g1-safety"),
            replay_query_manifest_sha256=_sha("replay-queries"),
            replay_artifact_commitments={
                "cache_npz": _sha("cache"),
                "predictions_jsonl": _sha("predictions"),
                "scores_jsonl": _sha("scores"),
            },
        )


def test_g2_frozen_fixture_semantic_hashes_match_reviewed_files() -> None:
    fixture_root = Path(__file__).parent / "fixtures"
    fixtures = (
        (
            "metric_fixture_v1.json",
            a4_stages.FROZEN_METRIC_FIXTURE_CANONICAL_SHA256,
        ),
        (
            "metric_fixture_v1_expected.json",
            a4_stages.FROZEN_METRIC_EXPECTED_CANONICAL_SHA256,
        ),
    )
    for relative, expected_sha256 in fixtures:
        value = json.loads(
            (fixture_root / relative).read_text(encoding="utf-8")
        )
        assert hashlib.sha256(
            canonical_json_bytes(value) + b"\n"
        ).hexdigest() == expected_sha256


def test_g2_uses_exact_frozen_c7_to_c28f_parity_field_map() -> None:
    assert a4_metrics.C7_LEGACY_VCMR_PARITY_FIELD_MAP == {
        f"{threshold:.1f}-r{k}": f"VCMR_R@{k}_IoU@{threshold:.1f}"
        for threshold in (0.5, 0.7)
        for k in (1, 5, 10, 100)
    }
    stage_source = inspect.getsource(build_g2_bundle)
    control_source = inspect.getsource(
        a4_control._g2_commit_parity_derivation_once
    )
    assert "field_map=C7_LEGACY_VCMR_PARITY_FIELD_MAP" in stage_source
    assert "field_map=C7_LEGACY_VCMR_PARITY_FIELD_MAP" in control_source


def test_replay_execution_direct_construction_is_rejected() -> None:
    with pytest.raises(StageContractError, match="concrete G2 control path"):
        ReplayExecution(
            packets=(),
            power_ground_truth={},
            legacy_vcmr_metrics={},
            legacy_metric_unit="ratio",
            current_vcmr_metrics={},
            packet_sha256=_sha("packets"),
            ground_truth_sha256=_sha("ground-truth"),
            legacy_metrics_sha256=_sha("legacy"),
            current_metrics_sha256=_sha("current"),
            evidence={},
            control_handle=object(),
            _issuer=None,
        )


def test_g2_power_rows_are_exact_six_bit_sufficient_statistics() -> None:
    row = _derive_g2_power_outcome_row(
        packet={
            "query_id": 7,
            "joint_proposals": [
                ["wrong-video", 0.0, 1.0, 4.0],
                ["gt-video", 20.0, 21.0, 3.0],
                ["other-2", 0.0, 1.0, 2.0],
                ["other-3", 0.0, 1.0, 1.0],
                ["other-4", 0.0, 1.0, 0.0],
                ["gt-video", 0.0, 10.0, -1.0],
            ],
        },
        ground_truth={
            "gt_video_id": "gt-video",
            "gt_span_sec": [0.0, 10.0],
            "query_manifest_sha256": _sha("query-manifest"),
        },
        replay_packet_sha256=_sha("replay-packet"),
        query_manifest_sha256=_sha("query-manifest"),
    )
    assert set(row) == {
        "schema_version",
        "desc_id",
        "gt_video_id",
        "outcomes",
        "replay_packet_sha256",
        "query_manifest_sha256",
    }
    assert "joint_proposals" not in row and "gt_span_sec" not in row
    assert tuple(row["outcomes"]) == POWER_OUTCOME_KEYS
    assert row["outcomes"] == {
        "VCMR_R1_IOU_0_7": 0,
        "VCMR_R5_IOU_0_7": 0,
        "VCMR_R10_IOU_0_7": 1,
        "LEGACY_UNIQUE_VIDEO_TRUE_VR_R100": 1,
        "WRONG_VIDEO_TOP1": 1,
        "CORRECT_VIDEO_WRONG_SPAN_TOP1": 0,
    }


def test_g2_power_correct_video_wrong_span_bit_is_exact() -> None:
    row = _derive_g2_power_outcome_row(
        packet={
            "query_id": 8,
            "joint_proposals": [["gt-video", 0.0, 4.0, 1.0]],
        },
        ground_truth={
            "gt_video_id": "gt-video",
            "gt_span_sec": [0.0, 10.0],
            "query_manifest_sha256": _sha("query-manifest"),
        },
        replay_packet_sha256=_sha("replay-packet"),
        query_manifest_sha256=_sha("query-manifest"),
    )
    assert row["outcomes"] == {
        "VCMR_R1_IOU_0_7": 0,
        "VCMR_R5_IOU_0_7": 0,
        "VCMR_R10_IOU_0_7": 0,
        "LEGACY_UNIQUE_VIDEO_TRUE_VR_R100": 1,
        "WRONG_VIDEO_TOP1": 0,
        "CORRECT_VIDEO_WRONG_SPAN_TOP1": 1,
    }


def test_g2_historical_reference_is_committed_no_join_authority() -> None:
    replay_sha = _sha("reference-replay")
    query_sha = _sha("reference-query-manifest")
    source_sha = _sha("reference-source-registry")
    rows = [
        {
            "schema_version": a4_roles.POWER_REPLAY_ROW_SCHEMA,
            "desc_id": index,
            "gt_video_id": "reference-video-%d" % index,
            "outcomes": {
                key: (index + offset) % 2
                for offset, key in enumerate(POWER_OUTCOME_KEYS)
            },
            "replay_packet_sha256": replay_sha,
            "query_manifest_sha256": query_sha,
        }
        for index in range(2)
    ]
    power_rows_sha = hashlib.sha256(canonical_json_bytes(rows)).hexdigest()
    manifest = a4_roles.build_historical_power_reference_from_rows(
        rows,
        power_rows_sha256=power_rows_sha,
        replay_packet_sha256=replay_sha,
        query_manifest_sha256=query_sha,
        source_registry_sha256=source_sha,
    )
    assert a4_roles.validate_historical_power_reference_population(
        manifest
    ) == manifest
    assert manifest["power_rows_sha256"] == power_rows_sha
    assert manifest["reference_gt_video_cluster_count"] == 2
    assert manifest["train_fit_desc_id_join_performed"] is False
    assert manifest["train_fit_video_id_join_performed"] is False
    assert manifest["historical_performance_selects_membership"] is False
    assert manifest["candidate_membership_fields_consumed"] == [
        "actual_candidate_role_query_count",
        "actual_candidate_role_gt_video_cluster_count",
    ]


def test_stage_submit_rejects_duck_typed_commit_adapter() -> None:
    class FakeAdapter:
        pass

    with pytest.raises(StageContractError, match="A4ControlStageAdapter"):
        submit_stage_bundle(FakeAdapter(), object())  # type: ignore[arg-type]


def test_stage_bundle_direct_construction_is_rejected() -> None:
    with pytest.raises(StageContractError, match="reviewed stage builder"):
        StageBundle(
            plan={},
            output_bytes={},
            commit_eligible=True,
            bundle_sha256=_sha("forged-bundle"),
            _issuer=None,
        )


def test_all_stage_outputs_share_exact_path_and_64mib_cap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert a4_stages.STAGE_OUTPUT_FILE_MAX_BYTES == 64 * 1024 * 1024
    assert a4_stages.G2_OUTPUTS[-1] == (
        "f0_forensics/g2_historical_power_reference.json"
    )
    declared_paths = [
        path
        for group in (
            a4_stages.G1_OUTPUTS,
            a4_stages.G2_OUTPUTS,
            a4_stages.G3_OUTPUTS,
            a4_stages.G4_PROJECTION_OUTPUTS,
            a4_stages.G4_OUTPUTS,
            a4_stages.G5_OUTPUTS,
            a4_stages.G6_OUTPUTS,
        )
        for path in group
    ]
    assert len(declared_paths) == len(set(declared_paths))
    assert all(
        not path.startswith("/")
        and path.split("/")[0] in {"f0_forensics", "f1_protocol"}
        and all(piece not in {"", ".", ".."} for piece in path.split("/"))
        for path in declared_paths
    )
    context = _context(tmp_path / "output")
    common = {
        "_builder_issuer": a4_stages._STAGE_BUNDLE_ISSUER,
        "context": context,
        "action": G1_ACTION,
        "result_status": "TEST_ONLY",
        "next_action": "TEST_NEXT",
        "input_hashes": {"test_input": _sha("test-input")},
        "declared_safety_counts": {
            "protected_content_opens": 0,
            "protected_id_mapping_content_opens": 0,
            "historical_write_bytes": 0,
            "model_forward_evaluations": 0,
            "real_data_optimizer_updates": 0,
        },
        "commit_eligible": False,
    }
    valid = a4_stages._build_stage_bundle(
        **common,
        outputs={"f0_forensics/test.json": b"{}\n"},
    )
    assert valid.plan["output_index"]["f0_forensics/test.json"]["size_bytes"] == 3
    with pytest.raises(StageContractError, match="non-canonical"):
        a4_stages._build_stage_bundle(
            **common,
            outputs={"f0_forensics/../escape.json": b"{}\n"},
        )
    with pytest.raises(StageContractError, match="isolated stage roots"):
        a4_stages._build_stage_bundle(
            **common,
            outputs={"other_root/test.json": b"{}\n"},
        )
    monkeypatch.setattr(a4_stages, "STAGE_OUTPUT_FILE_MAX_BYTES", 16)
    with pytest.raises(StageContractError, match="64 MiB cap"):
        a4_stages._build_stage_bundle(
            **common,
            outputs={"f1_protocol/too-large.bin": b"x" * 17},
        )


def test_f0_b_public_artifact_field_sets_are_frozen_exactly() -> None:
    assert dict(a4_stages.F0_COMPLETED_EVAL_SAFETY_COUNTS) == {
        "model_forward_count": 1,
        "query_identity_source_content_open_count": 1,
        "gt_source_content_open_count": 1,
        "train_jsonl_total_authorized_content_open_count": 2,
        "unauthorized_or_protected_semantic_decode_count": 0,
        "protected_semantic_decode_count": 0,
        "unmediated_content_open_count": 0,
        "teacher_field_access_count": 0,
        "teacher_candidate_count": 0,
        "gt_support_count": 0,
        "gt_append_count": 0,
        "optimizer_update_count": 0,
    }
    assert a4_stages.F0_A_FORENSICS_REPORT_SCHEMA == (
        "c28f_f0_a_forensics_report_v2"
    )
    assert a4_stages.F0_B_FORENSICS_REPORT_SCHEMA == (
        "c28f_f0_b_forensics_report_v2"
    )
    assert set(a4_stages.F0_A_COMMITTED_REPORT_BINDING_FIELDS) == {
        "report_path", "report_schema_version", "report_file_sha256",
        "report_sha256", "frozen_gate_result_sha256",
        "earliest_failing_stage", "comparison_summary",
        "comparison_summary_sha256", "g3_commit_receipt_sha256",
        "g3_state_sha256", "g3_event_sha256",
    }
    assert a4_stages.F0_COMPLETED_EVAL_SOURCE_ARTIFACT_LINEAGE_SCHEMA == (
        "c28f_f0_completed_eval_source_artifact_lineage_v2"
    )
    assert a4_stages.F0_RAW_FORWARD_ARTIFACT_INDEX_SCHEMA == (
        "c28f_a4_raw_forward_artifact_index_v2"
    )
    assert set(a4_stages.F0_RAW_FORWARD_ARTIFACT_INDEX_FIELDS) == {
        "schema_version", "status", "goal_id", "attempt_id", "run_id",
        "completed_transaction_id",
        "completed_forward_validation_receipt_sha256",
        "forward_run_receipt_sha256", "consumer_capability_sha256",
        "consumer_binding_sha256", "result_chunk_count",
        "result_chunk_set_sha256", "ordered_ranges_sha256",
        "consumed_range_receipt_sha256s",
        "consumed_range_receipt_chain_sha256", "all_ranges_consumed",
        "maximum_simultaneously_loaded_chunk_count",
        "payload_retention_contract", "cursor_artifact_index_sha256",
        "raw_joint_input_set_sha256", "result_chunks",
        "observation_snapshot", "encoded_cache_binding", "index_sha256",
    }
    assert set(a4_stages.F0_RAW_FORWARD_ARTIFACT_CHUNK_FIELDS) == {
        "range_index", "start_index", "end_index", "row_count",
        "artifact_name", "artifact_file_sha256", "payload_semantic_sha256",
        "engine_registry_consumption_sha256", "query_types_sha256",
        "query_rows_sha256", "claim_capability_sha256",
        "claim_binding_sha256", "materialization_owner_generation",
        "materialization_owner_binding_sha256",
        "materialization_owner_takeover_chain_sha256",
        "owner_lineage_sha256", "pre_gt_artifact_name",
        "pre_gt_artifact_file_sha256", "pre_gt_payload_sha256",
        "derived_nms_artifact_name", "derived_nms_artifact_file_sha256",
        "derived_nms_payload_sha256", "postimage_capability_sha256",
        "postimage_binding_sha256", "range_consumption_receipt_sha256",
    }
    assert a4_stages.F0_DERIVED_NMS_ARTIFACT_INDEX_SCHEMA == (
        "c28f_a4_derived_nms_artifact_index_v2"
    )
    assert set(a4_stages.F0_DERIVED_NMS_ARTIFACT_INDEX_FIELDS) == {
        "schema_version", "status", "goal_id", "attempt_id", "run_id",
        "completed_transaction_id", "source_raw_artifact_index_sha256",
        "rebuild_contract_sha256", "consumer_binding_sha256",
        "consumed_range_receipt_chain_sha256", "chunk_count",
        "query_count", "result_set_sha256", "chunks", "index_sha256",
    }
    assert set(a4_stages.F0_DERIVED_NMS_ARTIFACT_CHUNK_FIELDS) == {
        "range_index", "start_index", "end_index", "row_count",
        "artifact_name", "artifact_file_sha256", "payload_sha256",
        "source_artifact_file_sha256", "source_payload_semantic_sha256",
        "range_consumption_receipt_sha256", "postimage_binding_sha256",
    }
    assert len(a4_stages.F0_COMPLETED_EVAL_SOURCE_ARTIFACT_LINEAGE_FIELDS) == len(
        set(a4_stages.F0_COMPLETED_EVAL_SOURCE_ARTIFACT_LINEAGE_FIELDS)
    )
    assert {
        "train_jsonl_index_binding_sha256",
        "train_jsonl_index_receipt_sha256",
        "raw_forward_artifact_index_sha256",
        "completed_forward_validation_receipt_sha256",
        "consumer_binding_sha256",
        "consumed_range_receipt_chain_sha256",
        "raw_derivation_chunk_index_sha256",
        "derived_nms_artifact_index_sha256",
        "derived_nms_chunk_index_sha256",
        "derived_nms_rebuild_contract",
        "joint_derivation_receipt_sha256",
        "raw_proposals_scores_or_gt_spans_embedded",
        "lineage_sha256",
    }.issubset(a4_stages.F0_COMPLETED_EVAL_SOURCE_ARTIFACT_LINEAGE_FIELDS)
    assert set(a4_stages.F0_B_FORENSICS_REPORT_FIELDS) == {
        "schema_version", "status", "goal_id", "attempt_id", "stage",
        "token_id", "eval_id", "run_id", "query_role",
        "query_manifest_sha256", "corpus_manifest_sha256", "query_count",
        "query_completeness", "committed_query_role_binding",
        "completed_eval_capability_sha256",
        "completed_eval_artifact_binding_sha256",
        "completed_eval_source_artifact_lineage",
        "cross_split_execution_invariants",
        "metric_sufficient_stat_contract_sha256", "retrieval_metrics",
        "joint_retrieval_metrics", "vcmr_metrics", "error_metrics",
        "gate_metrics", "rank_transition_summary", "margin_diagnostics",
        "modality_mask_diagnostics", "video_distribution_diagnostics",
        "denominators", "frozen_gate_result", "frozen_gate_result_sha256",
        "earliest_failing_stage", "comparison_summary",
        "comparison_summary_sha256", "bucket_report",
        "predecessor_f0_a_binding", "split_shift_report_sha256",
        "routing_status", "teacher_shadow_status", "intrinsic_diagnostics",
        "calibration_status", "artifact_commitments", "safety_counts",
        "threshold_changed_after_observation", "report_sha256",
    }
    assert len(a4_stages.F0_B_FORENSICS_REPORT_FIELDS) == len(
        set(a4_stages.F0_B_FORENSICS_REPORT_FIELDS)
    )
    assert set(a4_stages.F0_A_FORENSICS_REPORT_FIELDS) == set(
        a4_stages.F0_B_FORENSICS_REPORT_FIELDS
    ) - {
        "bucket_report",
        "predecessor_f0_a_binding",
        "split_shift_report_sha256",
        "routing_status",
    }
    assert len(a4_stages.F0_A_FORENSICS_REPORT_FIELDS) == len(
        set(a4_stages.F0_A_FORENSICS_REPORT_FIELDS)
    )
    assert set(a4_stages.F0_SPLIT_SHIFT_REPORT_FIELDS) == {
        "schema_version", "status", "f0_a_committed_report_binding",
        "f0_b_completed_eval_capability_sha256", "f0_b_comparison_summary",
        "f0_b_comparison_summary_sha256", "query_count_shift",
        "cross_split_execution_invariants",
        "cross_split_execution_invariants_match", "retrieval_metric_shift",
        "joint_retrieval_metric_shift", "vcmr_metric_shift",
        "error_metric_shift", "gate_metric_shift", "rank_transition_shift",
        "margin_shift", "modality_mask_shift", "video_distribution_shift",
        "bucket_shift", "earliest_failing_stage", "teacher_status",
        "gt_support_count", "training_exposure_boundary",
        "threshold_changed_after_observation", "routing_status",
        "split_shift_report_sha256",
    }
    assert len(a4_stages.F0_SPLIT_SHIFT_REPORT_FIELDS) == len(
        set(a4_stages.F0_SPLIT_SHIFT_REPORT_FIELDS)
    )
    assert set(a4_stages.F0_ROUTING_DECISION_FIELDS) == {
        "schema_version", "status", "f0_a_frozen_gate_result_sha256",
        "f0_b_frozen_gate_result_sha256", "f0_a_earliest_failing_stage",
        "f0_b_earliest_failing_stage", "selected_earliest_failing_stage",
        "next_stage_recommendation_only", "split_shift_report_sha256",
        "threshold_changed_after_observation", "f2_or_higher_executed",
        "ambiguous_is_valid_scientific_result", "routing_decision_sha256",
    }
    assert len(a4_stages.F0_ROUTING_DECISION_FIELDS) == len(
        set(a4_stages.F0_ROUTING_DECISION_FIELDS)
    )


def _bounded_raw_nms_lineage_payload_fixture() -> dict:
    range_receipt_sha = _sha("range-consumption-receipt")
    receipt_sha256s = [range_receipt_sha]
    receipt_chain_sha = hashlib.sha256(
        canonical_json_bytes(receipt_sha256s)
    ).hexdigest()
    pre_gt_payload_sha = _sha("pre-gt-payload")
    nms_payload_sha = _sha("derived-nms-payload")
    materialization_owner_binding_sha = _sha("materialization-owner-binding")
    empty_owner_chain_sha = hashlib.sha256(canonical_json_bytes([])).hexdigest()
    commit_owner_binding_chain = [materialization_owner_binding_sha]
    commit_owner_binding_chain_sha = hashlib.sha256(
        canonical_json_bytes(commit_owner_binding_chain)
    ).hexdigest()
    owner_lineage_sha = semantic_sha256(
        {
            "schema_version": "c28f_a4_forward_derivation_owner_lineage_v1",
            "materialization_owner_generation": 0,
            "materialization_owner_binding_sha256": (
                materialization_owner_binding_sha
            ),
            "materialization_owner_takeover_chain_sha256": (
                empty_owner_chain_sha
            ),
            "commit_owner_generation": 0,
            "commit_owner_binding_sha256": materialization_owner_binding_sha,
            "commit_owner_takeover_chain_sha256": empty_owner_chain_sha,
            "commit_owner_binding_chain": commit_owner_binding_chain,
            "commit_owner_binding_chain_sha256": (
                commit_owner_binding_chain_sha
            ),
        }
    )
    raw_chunk = {
        "range_index": 0,
        "start_index": 0,
        "end_index": 2,
        "row_count": 2,
        "artifact_name": "queries_00000_00002.pt",
        "artifact_file_sha256": _sha("raw-forward-artifact-file"),
        "payload_semantic_sha256": _sha("raw-forward-payload"),
        "engine_registry_consumption_sha256": _sha("engine-consumption"),
        "query_types_sha256": _sha("query-types"),
        "query_rows_sha256": _sha("query-rows"),
        "claim_capability_sha256": _sha("range-claim-capability"),
        "claim_binding_sha256": _sha("range-claim-binding"),
        "materialization_owner_generation": 0,
        "materialization_owner_binding_sha256": (
            materialization_owner_binding_sha
        ),
        "materialization_owner_takeover_chain_sha256": empty_owner_chain_sha,
        "owner_lineage_sha256": owner_lineage_sha,
        "pre_gt_artifact_name": "pre_gt_000000_%s.json" % pre_gt_payload_sha,
        "pre_gt_artifact_file_sha256": _sha("pre-gt-file"),
        "pre_gt_payload_sha256": pre_gt_payload_sha,
        "derived_nms_artifact_name": (
            "derived_nms_000000_%s.json" % nms_payload_sha
        ),
        "derived_nms_artifact_file_sha256": _sha("derived-nms-file"),
        "derived_nms_payload_sha256": nms_payload_sha,
        "postimage_capability_sha256": _sha("postimage-capability"),
        "postimage_binding_sha256": _sha("postimage-binding"),
        "range_consumption_receipt_sha256": range_receipt_sha,
    }
    raw_base = {
        "schema_version": a4_stages.F0_RAW_FORWARD_ARTIFACT_INDEX_SCHEMA,
        "status": "FSYNCED_BOUNDED_TWO_PHASE_RAW_FORWARD_INDEX",
        "goal_id": "goal-c28f-test",
        "attempt_id": "G0-A4-test",
        "run_id": "run-test",
        "completed_transaction_id": "G0-A4-F0A-test-COMPLETE",
        "completed_forward_validation_receipt_sha256": _sha(
            "completed-forward-validation"
        ),
        "forward_run_receipt_sha256": _sha("forward-run-receipt"),
        "consumer_capability_sha256": _sha("derivation-consumer-capability"),
        "consumer_binding_sha256": _sha("derivation-consumer-binding"),
        "result_chunk_count": 1,
        "result_chunk_set_sha256": _sha("safe-result-chunk-set"),
        "ordered_ranges_sha256": _sha("ordered-ranges"),
        "consumed_range_receipt_sha256s": receipt_sha256s,
        "consumed_range_receipt_chain_sha256": receipt_chain_sha,
        "all_ranges_consumed": True,
        "maximum_simultaneously_loaded_chunk_count": 1,
        "payload_retention_contract": (
            "HANDLE_STORES_NO_RAW_PAYLOAD_STREAM_REVALIDATES_ONE_CHUNK"
        ),
        "cursor_artifact_index_sha256": _sha("cursor-index"),
        "raw_joint_input_set_sha256": _sha("raw-joint-input-set"),
        "result_chunks": [raw_chunk],
        "observation_snapshot": {"status": "SAFE_SUMMARY_ONLY"},
        "encoded_cache_binding": {"status": "BOUND"},
    }
    raw_index = {**raw_base, "index_sha256": semantic_sha256(raw_base)}
    nms_chunk = {
        "range_index": 0,
        "start_index": 0,
        "end_index": 2,
        "row_count": 2,
        "artifact_name": raw_chunk["derived_nms_artifact_name"],
        "artifact_file_sha256": raw_chunk[
            "derived_nms_artifact_file_sha256"
        ],
        "payload_sha256": raw_chunk["derived_nms_payload_sha256"],
        "source_artifact_file_sha256": raw_chunk["artifact_file_sha256"],
        "source_payload_semantic_sha256": raw_chunk[
            "payload_semantic_sha256"
        ],
        "range_consumption_receipt_sha256": range_receipt_sha,
        "postimage_binding_sha256": raw_chunk["postimage_binding_sha256"],
    }
    rebuild_base = {
        "schema_version": "c28f_a4_derived_nms_rebuild_contract_v1",
        "status": "FROZEN",
    }
    rebuild = {
        **rebuild_base,
        "contract_sha256": semantic_sha256(rebuild_base),
    }
    nms_base = {
        "schema_version": a4_stages.F0_DERIVED_NMS_ARTIFACT_INDEX_SCHEMA,
        "status": "FSYNCED_FROZEN_NMS_INDEX",
        "goal_id": raw_index["goal_id"],
        "attempt_id": raw_index["attempt_id"],
        "run_id": raw_index["run_id"],
        "completed_transaction_id": raw_index["completed_transaction_id"],
        "source_raw_artifact_index_sha256": raw_index["index_sha256"],
        "rebuild_contract_sha256": rebuild["contract_sha256"],
        "consumer_binding_sha256": raw_index["consumer_binding_sha256"],
        "consumed_range_receipt_chain_sha256": receipt_chain_sha,
        "chunk_count": 1,
        "query_count": 2,
        "result_set_sha256": _sha("frozen-nms-result-set"),
        "chunks": [nms_chunk],
    }
    nms_index = {**nms_base, "index_sha256": semantic_sha256(nms_base)}
    return {
        "goal_id": raw_index["goal_id"],
        "attempt_id": raw_index["attempt_id"],
        "run_id": raw_index["run_id"],
        "completed_transaction_id": raw_index["completed_transaction_id"],
        "raw_forward_artifact_index": raw_index,
        "raw_forward_artifact_index_file_sha256": _sha("raw-index-file"),
        "raw_forward_artifact_index_sha256": raw_index["index_sha256"],
        "forward_run_receipt_sha256": raw_index[
            "forward_run_receipt_sha256"
        ],
        "result_chunk_count": 1,
        "result_chunk_set_sha256": raw_index["result_chunk_set_sha256"],
        "artifact_index_sha256": raw_index["cursor_artifact_index_sha256"],
        "raw_joint_input_set_sha256": raw_index[
            "raw_joint_input_set_sha256"
        ],
        "derived_nms_artifact_index": nms_index,
        "derived_nms_artifact_index_file_sha256": _sha("nms-index-file"),
        "derived_nms_artifact_index_sha256": nms_index["index_sha256"],
        "derived_nms_rebuild_contract": rebuild,
        "derived_nms_rebuild_contract_sha256": rebuild["contract_sha256"],
        "frozen_nms_result_set_sha256": nms_index["result_set_sha256"],
        "joint_sufficient_stats_sha256": _sha("joint-stats"),
        "joint_derivation_receipt_file_sha256": _sha("joint-receipt-file"),
        "joint_derivation_receipt_sha256": _sha("joint-receipt"),
        "retrieval_rows_sha256": _sha("retrieval-rows"),
        "retrieval_row_count": 2,
    }


def test_completed_eval_lineage_binds_bounded_two_phase_indexes(
    tmp_path: Path,
) -> None:
    payload = _bounded_raw_nms_lineage_payload_fixture()
    context = _post_index_context(
        tmp_path / "lineage-output",
        action="G3_F0_A_BASELINE_FORENSICS",
        status="METRIC_CONTRACT_OK",
    )
    assert len(a4_stages.F0_RAW_FORWARD_ARTIFACT_CHUNK_FIELDS) == 25
    assert a4_stages.F0_RAW_FORWARD_ARTIFACT_CHUNK_FIELDS[12:16] == (
        "materialization_owner_generation",
        "materialization_owner_binding_sha256",
        "materialization_owner_takeover_chain_sha256",
        "owner_lineage_sha256",
    )
    lineage = a4_stages._completed_eval_source_artifact_lineage(
        payload,
        context=context,
    )
    assert lineage["schema_version"] == (
        "c28f_f0_completed_eval_source_artifact_lineage_v2"
    )
    assert lineage["all_ranges_consumed"] is True
    assert lineage["maximum_simultaneously_loaded_chunk_count"] == 1
    assert lineage["train_jsonl_index_binding_sha256"] == _sha(
        "train-index-binding"
    )
    assert lineage["train_jsonl_index_receipt_sha256"] == _sha(
        "train-index-receipt"
    )
    assert lineage["consumer_binding_sha256"] == lineage[
        "derivation_consumer_binding_sha256"
    ]
    assert lineage["consumed_range_receipt_chain_sha256"] == lineage[
        "derivation_consumed_range_receipt_chain_sha256"
    ]
    assert "forward_run_root" not in lineage
    assert "forward_run_receipt_path" not in lineage


@pytest.mark.parametrize(
    "mutation",
    (
        "old_raw_schema",
        "not_all_consumed",
        "receipt_chain",
        "raw_path",
        "pre_gt_name",
        "nms_source_payload",
        "missing_materialization_owner",
        "invalid_materialization_owner_generation",
        "invalid_materialization_owner_binding",
        "invalid_materialization_owner_takeover_chain",
        "owner_lineage_composite",
    ),
)
def test_completed_eval_lineage_rejects_unbounded_or_detached_indexes(
    mutation: str,
    tmp_path: Path,
) -> None:
    payload = copy.deepcopy(_bounded_raw_nms_lineage_payload_fixture())
    context = _post_index_context(
        tmp_path / "lineage-negative-output",
        action="G3_F0_A_BASELINE_FORENSICS",
        status="METRIC_CONTRACT_OK",
    )
    raw_index = payload["raw_forward_artifact_index"]
    nms_index = payload["derived_nms_artifact_index"]
    if mutation == "old_raw_schema":
        raw_index["schema_version"] = "c28f_a4_raw_forward_artifact_index_v1"
    elif mutation == "not_all_consumed":
        raw_index["all_ranges_consumed"] = False
    elif mutation == "receipt_chain":
        raw_index["consumed_range_receipt_chain_sha256"] = _sha(
            "detached-chain"
        )
    elif mutation == "raw_path":
        raw_index["result_chunks"][0]["artifact_path"] = "/raw/model/chunk.pt"
    elif mutation == "pre_gt_name":
        raw_index["result_chunks"][0]["pre_gt_artifact_name"] = (
            "../escaped.json"
        )
    elif mutation == "nms_source_payload":
        nms_index["chunks"][0]["source_payload_semantic_sha256"] = _sha(
            "detached-source-payload"
        )
    elif mutation == "missing_materialization_owner":
        raw_index["result_chunks"][0].pop(
            "materialization_owner_binding_sha256"
        )
    elif mutation == "invalid_materialization_owner_generation":
        raw_index["result_chunks"][0]["materialization_owner_generation"] = -1
    elif mutation == "invalid_materialization_owner_binding":
        raw_index["result_chunks"][0][
            "materialization_owner_binding_sha256"
        ] = "bad"
    elif mutation == "invalid_materialization_owner_takeover_chain":
        raw_index["result_chunks"][0][
            "materialization_owner_takeover_chain_sha256"
        ] = "bad"
    else:
        raw_index["result_chunks"][0]["owner_lineage_sha256"] = _sha(
            "detached-owner-lineage-composite"
        )
    with pytest.raises(StageContractError, match="raw|bounded|NMS"):
        a4_stages._completed_eval_source_artifact_lineage(
            payload,
            context=context,
        )


def test_g4_rejects_caller_role_and_power_mappings(tmp_path: Path) -> None:
    context = _post_index_context(
        tmp_path / "output",
        action="G4_ROLE_POLICY_LOCK",
        status="ID_MAPPING_COMPLETE",
    )
    with pytest.raises(StageContractError, match="A4G4RoleLockInputHandle"):
        build_g4_bundle(
            context=context,
            role_input_handle={},
            power_replay_capability={},
        )


def test_g4_projection_rejects_unsealed_result(tmp_path: Path) -> None:
    context = _post_index_context(
        tmp_path / "output",
        action=G4_PROJECTION_ACTION,
        status="ID_MAPPING_ACCESS_COMMITTED",
    )
    with pytest.raises(StageContractError, match="ProjectionResult"):
        build_g4_projection_bundle(
            context=context,
            projection_result={
                "records": [],
                "access_receipt": {},
            },
        )


def test_g4_projection_and_role_open_classes_are_frozen_0_5_then_0_0() -> None:
    classification = {
        "categories_mutually_exclusive": True,
        "unauthorized_or_non_id_only_protected_content_open_count": 0,
        "authorized_protected_id_mapping_content_open_count": 5,
        "total_protected_content_open_count": 5,
        "authorized_id_mapping_breakdown": {
            "protected_desc_id_manifest_content_open_count": 2,
            "train_fit_desc_id_manifest_content_open_count": 1,
            "protected_desc_train_jsonl_indexed_pread_content_open_count": 1,
            "train_fit_train_jsonl_indexed_pread_content_open_count": 1,
            "official_val_content_open_count": 0,
        },
    }
    assert _validate_g4_projection_open_classification(
        {"content_open_classification": classification}
    ) == classification
    assert dict(G4_PROJECTION_DECLARED_SAFETY_COUNTS) == {
        "protected_content_opens": 0,
        "protected_id_mapping_content_opens": 5,
        "historical_write_bytes": 0,
        "model_forward_evaluations": 0,
        "real_data_optimizer_updates": 0,
    }
    assert dict(G4_ROLE_DECLARED_SAFETY_COUNTS) == {
        "protected_content_opens": 0,
        "protected_id_mapping_content_opens": 0,
        "historical_write_bytes": 0,
        "model_forward_evaluations": 0,
        "real_data_optimizer_updates": 0,
    }
    drifted = dict(classification)
    drifted["authorized_protected_id_mapping_content_open_count"] = 1
    with pytest.raises(StageContractError, match="0/5 split"):
        _validate_g4_projection_open_classification(
            {"content_open_classification": drifted}
        )


def _g4_atomic_candidate_fixture() -> tuple[dict, dict, dict]:
    goal_id = "goal-c28f-test"
    attempt_id = "G0-A4-test"
    role_snapshot_sha = _sha("g4-role-snapshot")
    power_binding_sha = _sha("g4-power-binding")
    power_capability_sha = _sha("g4-power-capability")
    power_rows_sha = _sha("g4-power-rows")
    power_rows_file_sha = _sha("g4-power-rows-file")
    replay_packet_sha = _sha("g4-replay-packets")
    query_manifest_sha = _sha("g4-reference-queries")
    source_registry_sha = _sha("g4-reference-source-registry")
    g2_transaction_id = "G0-A4-G2-COMMIT-test"
    g2_state_sha = _sha("g4-g2-state")
    g2_event_sha = _sha("g4-g2-event")
    g2_control_receipt_sha = _sha("g4-g2-control-receipt")
    historical_reference_file_sha = _sha("g2-reference-file")
    power_contract_file_sha = _sha("g2-power-contract-file")
    power_row_count = 8_739
    reference_base = {
        "schema_version": "c28f_g2_historical_power_reference_v1",
        "population_semantics": (
            "INDEPENDENT_C7_CALIBRATED_A4_HISTORICAL_REFERENCE_ONLY"
        ),
        "power_rows_sha256": power_rows_sha,
        "power_row_count": power_row_count,
        "replay_packet_sha256": replay_packet_sha,
        "query_manifest_sha256": query_manifest_sha,
        "source_registry_sha256": source_registry_sha,
        "reference_desc_id_lines_sha256": _sha("g4-reference-desc-lines"),
        "reference_gt_video_cluster_count": 4_000,
        "reference_gt_video_id_lines_sha256": _sha(
            "g4-reference-video-lines"
        ),
        "reference_outcome_distribution": {
            outcome: {"zero_count": 4_000, "one_count": 4_739}
            for outcome in POWER_OUTCOME_KEYS
        },
        "normalized_reference_sha256": _sha("g4-normalized-reference"),
        "train_fit_desc_id_join_performed": False,
        "train_fit_video_id_join_performed": False,
        "candidate_membership_fields_consumed": [
            "actual_candidate_role_query_count",
            "actual_candidate_role_gt_video_cluster_count",
        ],
        "historical_performance_selects_membership": False,
    }
    reference = {
        **reference_base,
        "reference_population_sha256": semantic_sha256(reference_base),
    }
    roles = (
        "train_fit_route_dev",
        "train_fit_mechanism_dev",
        "train_fit_confirm",
        "train_fit_calibration",
        "train_fit_core",
    )
    role_counts = {
        "train_fit_route_dev": {"queries": 2_048, "gt_videos": 1_000},
        "train_fit_mechanism_dev": {"queries": 4_096, "gt_videos": 2_000},
        "train_fit_confirm": {"queries": 2_048, "gt_videos": 1_000},
        "train_fit_calibration": {"queries": 2_048, "gt_videos": 1_000},
        "train_fit_core": {"queries": 59_188, "gt_videos": 12_000},
    }
    power_contract = a4_roles.frozen_power_evaluator_contract()

    def v3_power_result(role: str) -> dict:
        query_count = role_counts[role]["queries"]
        cluster_count = role_counts[role]["gt_videos"]
        outer_count = 200
        maximum_k = int(math.ceil(0.10 * query_count))
        thresholds = [1] * outer_count
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
            for outer_index in range(outer_count)
        ]
        effect_results = {}
        for effect_key, rule in sorted(
            power_contract["effect_outcome_registry"].items()
        ):
            minimum_effect = power_contract["minimum_effects_ratio"][effect_key]
            eligible_count_key = (
                "zero_count"
                if rule["eligible_baseline_value"] == 0
                else "one_count"
            )
            effect_results[effect_key] = {
                "binary_outcome": rule["binary_outcome"],
                "sensitivity_rule": rule["sensitivity_rule"],
                "eligible_baseline_value": rule["eligible_baseline_value"],
                "injected_flip": rule["injected_flip"],
                "minimum_effect_ratio": minimum_effect,
                "reference_eligible_query_count": reference[
                    "reference_outcome_distribution"
                ][rule["binary_outcome"]][eligible_count_key],
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
                "detected_outer_count_at_mde": outer_count,
                "detected_outer_count_at_k_minus_1": 0,
                "detected_outer_count_at_maximum_supported_k": outer_count,
                "power_at_maximum_supported_k": 1.0,
                "paired_ci_detection_rule": (
                    "49_INNER_CLUSTER_POSITION_BOOTSTRAPS_NEAREST_RANK_2_48_"
                    "SIGN_NORMALIZED_LOWER_GT_ZERO"
                ),
                "status": "PASS",
            }
        quotient, remainder = divmod(query_count, cluster_count)
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
            "power_algorithm": power_contract["algorithm"],
            "power_counter_domain": (
                "C28F_G2_HISTORICAL_REFERENCE_POWER_V3"
            ),
            "reference_population_sha256": reference[
                "reference_population_sha256"
            ],
            "reference_query_count": power_row_count,
            "reference_gt_video_cluster_count": 4_000,
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
            "outer_cluster_position_sample_sha256": _sha(
                "%s-outer-cluster-positions" % role
            ),
            "inner_cluster_position_resampling": (
                "C_OUTER_POSITION_DRAWS_WITH_REPLACEMENT_DUPLICATES_"
                "PRESERVED_ALL_POSITION_SLOTS"
            ),
            "inner_cluster_position_resample_sha256": _sha(
                "%s-inner-cluster-positions" % role
            ),
            "paired_delta_semantics": (
                "SAME_SLOT_BASELINE_CANDIDATE_SIGN_NORMALIZED_IMPROVEMENT_MEAN"
            ),
            "paired_ci_contract": {
                "inner_sample_count": 49,
                "lower_nearest_rank_1_indexed": 2,
                "upper_nearest_rank_1_indexed": 48,
                "detection_rule": "SIGN_NORMALIZED_LOWER_BOUND_GT_ZERO",
            },
            "bootstrap_resample_count": 10_000,
            "outer_experiment_count": outer_count,
            "inner_resamples_per_outer": 49,
            "effect_results": effect_results,
            "status": "PASS",
        }
    role_lock_base = {
        "schema_version": a4_roles.ROLE_LOCK_SCHEMA,
        "status": "VALIDATED_CONTROL_INPUT_CANDIDATE_AWAITING_STAGE_COMMIT",
        "control_input_binding": {
            "goal_id": goal_id,
            "attempt_id": attempt_id,
            "role_input_snapshot_sha256": role_snapshot_sha,
        },
        "missing_set": [],
        "pairwise_desc_disjoint": True,
        "pairwise_gt_video_disjoint": True,
        "train_fit_union_complete": True,
        "role_counts": role_counts,
    }
    role_lock = {
        **role_lock_base,
        "lock_sha256": semantic_sha256(role_lock_base),
    }
    power_results = {
        role: v3_power_result(role)
        for role in ("train_fit_mechanism_dev", "train_fit_confirm")
    }
    trusted_role_counts = {
        role: {
            "query_count": role_counts[role]["queries"],
            "gt_video_cluster_count": role_counts[role]["gt_videos"],
        }
        for role in ("train_fit_mechanism_dev", "train_fit_confirm")
    }
    role_result_sha256s = {
        role: semantic_sha256(result)
        for role, result in sorted(power_results.items())
    }
    trusted_power_receipt_base = {
        "schema_version": (
            "c28f_a4_g2_power_computation_receipt_capability_v1"
        ),
        "status": "CONTROLLER_RECOMPUTED_TRUSTED_POWER",
        "goal_id": goal_id,
        "attempt_id": attempt_id,
        "g2_committed_transaction_id": g2_transaction_id,
        "g2_committed_state_sha256": g2_state_sha,
        "g2_committed_event_sha256": g2_event_sha,
        "power_replay_capability_sha256": power_capability_sha,
        "power_rows_sha256": power_rows_sha,
        "historical_reference_population_sha256": reference[
            "reference_population_sha256"
        ],
        "power_evaluator_contract_sha256": power_contract[
            "contract_sha256"
        ],
        "role_counts": trusted_role_counts,
        "role_counts_sha256": hashlib.sha256(
            canonical_json_bytes(trusted_role_counts)
        ).hexdigest(),
        "role_results": power_results,
        "role_result_sha256s": role_result_sha256s,
        "combined_results_sha256": hashlib.sha256(
            canonical_json_bytes(power_results)
        ).hexdigest(),
        "raw_power_rows_disclosed": False,
    }
    trusted_power_receipt = {
        **trusted_power_receipt_base,
        "receipt_sha256": semantic_sha256(trusted_power_receipt_base),
    }
    membership_assignment_inputs = (
        "TRAIN_FIT_GT_VIDEO_CLUSTERS_PLUS_FROZEN_SEEDED_DISJOINT_ROLE_"
        "LANE_ORDERS_PLUS_POWER_DETERMINED_PER_ROLE_PREFIX_LENGTH_BEFORE_"
        "MANIFEST_LOCK"
    )
    frozen_role_lane_sha256s = {
        "train_fit_mechanism_dev": _sha("g4-mechanism-frozen-lane"),
        "train_fit_confirm": _sha("g4-confirm-frozen-lane"),
    }
    reserved_core_sha = _sha("g4-reserved-core-lane")
    power_expansion_report = {
        "status": "PASS",
        "expansion_policy": (
            "PREREGISTERED_DISJOINT_ROLE_LANE_PREFIXES_BEFORE_MANIFEST_LOCK"
        ),
        "minimum_effect_registry_sha256": power_contract[
            "minimum_effect_registry_sha256"
        ],
        "effect_outcome_registry_sha256": power_contract[
            "effect_outcome_registry_sha256"
        ],
        "effect_threshold_reduction_count": 0,
        "historical_reference_population": reference,
        "membership_assignment_inputs": membership_assignment_inputs,
        "role_results": power_results,
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
        "expansion_lane_algorithm": a4_roles.POWER_EXPANSION_LANE_ALGORITHM,
        "frozen_core_prefix_video_ids_sha256": _sha(
            "g4-frozen-core-prefix"
        ),
        "frozen_role_lane_video_ids_sha256s": frozen_role_lane_sha256s,
        "reserved_core_video_ids_sha256": reserved_core_sha,
        "role_specific_expansion_lanes_disjoint": True,
        "adjustment_rounds": [],
        "trusted_power_computation_receipt": trusted_power_receipt,
    }
    power_expansion_report_sha = semantic_sha256(power_expansion_report)
    manifests = {}
    for role in roles:
        assignment_base = {
            "algorithm": a4_roles.POWER_ADAPTIVE_ASSIGNMENT_ALGORITHM,
            "seed": a4_roles.ASSIGNMENT_SEED,
            "cluster_order": (
                "sha256(seed_ascii + NUL + video_id_utf8),video_id_utf8"
            ),
            "cluster_atomic": True,
            "candidate_membership_source": membership_assignment_inputs,
            "historical_reference_population_sha256": reference[
                "reference_population_sha256"
            ],
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
                a4_roles.POWER_EXPANSION_LANE_ALGORITHM
            ),
            "membership_lock_timing": (
                "ALL_PREFIX_EXPANSION_BEFORE_ROLE_MANIFEST_LOCK"
            ),
            "power_expansion_report_sha256": power_expansion_report_sha,
        }
        base = {
            "schema_version": a4_roles.ROLE_SCHEMA,
            "role": role,
            "assignment": {
                **assignment_base,
                "assignment_contract_sha256": semantic_sha256(assignment_base),
            },
        }
        manifests[role] = {
            **base,
            "manifest_sha256": semantic_sha256(base),
        }
    projected_query_count = sum(
        counts["queries"] for counts in role_counts.values()
    )
    projected_cluster_count = sum(
        counts["gt_videos"] for counts in role_counts.values()
    )
    power_audit_base = {
        "schema_version": a4_roles.CLUSTER_POWER_AUDIT_SCHEMA,
        "status": "VALIDATED_CONTROL_POWER_CANDIDATE_AWAITING_STAGE_COMMIT",
        "role_lock_sha256": role_lock["lock_sha256"],
        "projected_query_count": projected_query_count,
        "gt_video_cluster_count": projected_cluster_count,
        "cluster_size_distribution": {
            "minimum": 1,
            "p50_nearest_rank": 4,
            "p90_nearest_rank": 8,
            "maximum": 16,
            "mean": projected_query_count / float(projected_cluster_count),
        },
        "role_minimum_margins": {
            role: {
                "minimum_queries": minimum,
                "assigned_queries": role_counts[role]["queries"],
                "cluster_overshoot_queries": (
                    role_counts[role]["queries"] - minimum
                ),
                "assigned_gt_video_clusters": role_counts[role][
                    "gt_videos"
                ],
            }
            for role, minimum in a4_roles.ROLE_MIN_QUERY_COUNTS
        },
        "train_fit_core_queries": role_counts["train_fit_core"]["queries"],
        "train_fit_core_gt_video_clusters": role_counts[
            "train_fit_core"
        ]["gt_videos"],
        "minimum_query_counts": dict(a4_roles.ROLE_MIN_QUERY_COUNTS),
        "minimum_effects_ratio": power_contract["minimum_effects_ratio"],
        "power_evidence_binding": {
            "status": "CONTROLLER_RECOMPUTED_TRUSTED_POWER",
            "evidence_sha256": trusted_power_receipt["receipt_sha256"],
            "goal_id": goal_id,
            "attempt_id": attempt_id,
            "committed_transaction_id": g2_transaction_id,
            "committed_state_sha256": g2_state_sha,
            "committed_event_sha256": g2_event_sha,
            "g2_replay_control_receipt_sha256": g2_control_receipt_sha,
            "power_replay_capability_binding_sha256": power_binding_sha,
            "power_replay_capability_sha256": power_capability_sha,
            "trusted_power_computation_receipt": trusted_power_receipt,
            "power_rows_file_sha256": power_rows_file_sha,
            "power_rows_sha256": power_rows_sha,
            "power_row_count": power_row_count,
            "replay_packet_sha256": replay_packet_sha,
            "query_manifest_sha256": query_manifest_sha,
            "source_registry_sha256": source_registry_sha,
            "historical_reference_population_sha256": reference[
                "reference_population_sha256"
            ],
            "historical_reference_file_sha256": (
                historical_reference_file_sha
            ),
            "power_evaluator_contract_sha256": power_contract[
                "contract_sha256"
            ],
            "power_contract_file_sha256": power_contract_file_sha,
        },
        "historical_reference_population": reference,
        "candidate_membership_assignment_inputs": membership_assignment_inputs,
        "membership_identity_order_fixed_before_power_computation": True,
        "historical_outcomes_may_determine_preregistered_prefix_length": (
            True
        ),
        "historical_outcomes_select_identity_order_or_within_prefix_members": (
            False
        ),
        "cross_role_historical_outcomes_affect_lane_identity_order_or_members": (
            False
        ),
        "membership_lock_timing": (
            "ALL_PREFIX_EXPANSION_BEFORE_ROLE_MANIFEST_LOCK"
        ),
        "expansion_lane_algorithm": a4_roles.POWER_EXPANSION_LANE_ALGORITHM,
        "frozen_role_lane_video_ids_sha256s": frozen_role_lane_sha256s,
        "reserved_core_video_ids_sha256": reserved_core_sha,
        "role_specific_expansion_lanes_disjoint": True,
        "power_result_use": (
            "PREREGISTERED_PER_ROLE_LANE_PREFIX_LENGTH_AND_STOP_POINT_ONLY_"
            "BEFORE_MANIFEST_LOCK"
        ),
        "power_results": power_results,
        "power_expansion_report": power_expansion_report,
        "power_expansion_report_sha256": power_expansion_report_sha,
        "effect_size_or_performance_claim": None,
        "interpretation": (
            "INDEPENDENT_HISTORICAL_REFERENCE_POWER_WITH_CANDIDATE_COUNTS_"
            "ONLY_NO_POSTHOC_EFFECT_REDUCTION_NO_IDENTITY_ORDER_OR_WITHIN_"
            "PREFIX_SELECTION_POWER_DETERMINES_PER_ROLE_LANE_PREFIX_LENGTH_"
            "ONLY_BEFORE_ROLE_MANIFEST_LOCK"
        ),
    }
    power_audit = {
        **power_audit_base,
        "audit_sha256": semantic_sha256(power_audit_base),
    }
    policy_base = {
        "schema_version": a4_roles.FORMAL_POLICY_SCHEMA,
        "status": "VALIDATED_CONTROL_POLICY_CANDIDATE_AWAITING_STAGE_COMMIT",
        "role_lock_sha256": role_lock["lock_sha256"],
        "cluster_power_audit_sha256": power_audit["audit_sha256"],
        "policy": a4_roles.FORMAL_POLICY,
        "U_formal_updates": a4_roles.U_FORMAL_UPDATES,
    }
    policy = {**policy_base, "policy_sha256": semantic_sha256(policy_base)}
    artifact_hashes = {
        "role_manifest_hashes": {
            role: manifests[role]["manifest_sha256"] for role in sorted(manifests)
        },
        "role_lock_sha256": role_lock["lock_sha256"],
        "cluster_power_audit_sha256": power_audit["audit_sha256"],
        "formal_data_policy_sha256": policy["policy_sha256"],
    }
    base = {
        "schema_version": a4_roles.G4_ATOMIC_CANDIDATE_SCHEMA,
        "status": "VALIDATED_ATOMIC_G4_CANDIDATES_AWAITING_STAGE_COMMIT",
        "goal_id": goal_id,
        "attempt_id": attempt_id,
        "role_input_snapshot_sha256": role_snapshot_sha,
        "power_replay_capability_binding_sha256": power_binding_sha,
        "power_replay_capability_sha256": power_capability_sha,
        "role_manifests": manifests,
        "role_lock": role_lock,
        "cluster_power_audit": power_audit,
        "formal_data_policy": policy,
        "artifact_candidate_hashes": artifact_hashes,
        "atomic_all_or_none": True,
        "control_commit_required": True,
        "persistence_claimed_by_pure_function": False,
    }
    return (
        {**base, "atomic_candidate_sha256": semantic_sha256(base)},
        {"role_input_snapshot_sha256": role_snapshot_sha},
        {
            "power_replay_capability_binding_sha256": power_binding_sha,
            "goal_id": goal_id,
            "attempt_id": attempt_id,
            "committed_transaction_id": g2_transaction_id,
            "committed_state_sha256": g2_state_sha,
            "committed_event_sha256": g2_event_sha,
            "g2_replay_control_receipt_sha256": g2_control_receipt_sha,
            "power_rows_sha256": power_rows_sha,
            "power_rows_file_sha256": power_rows_file_sha,
            "power_row_count": power_row_count,
            "replay_packet_sha256": replay_packet_sha,
            "query_manifest_sha256": query_manifest_sha,
            "source_registry_sha256": source_registry_sha,
            "historical_reference_population": reference,
            "historical_reference_file_sha256": _sha(
                "g2-reference-file"
            ),
            "power_evaluator_contract": power_contract,
            "power_contract_file_sha256": power_contract_file_sha,
        },
    )


def _g4_v3_bundle_binding_fixture(candidate: dict) -> dict:
    audit = candidate["cluster_power_audit"]
    power_lineage = audit["power_evidence_binding"]
    base = {
        "schema_version": a4_stages.G4_ATOMIC_BUNDLE_BINDING_SCHEMA,
        "status": "CANDIDATE_AWAITING_A4_STAGE_COMMIT_RECEIPT",
        "goal_id": candidate["goal_id"],
        "attempt_id": candidate["attempt_id"],
        "role_input_handle_sha256": _sha("g4-role-input-handle"),
        "power_replay_capability_sha256": candidate[
            "power_replay_capability_sha256"
        ],
        "power_replay_capability_binding_sha256": candidate[
            "power_replay_capability_binding_sha256"
        ],
        "trusted_power_computation_receipt_sha256": power_lineage[
            "trusted_power_computation_receipt"
        ]["receipt_sha256"],
        "atomic_candidate_sha256": candidate["atomic_candidate_sha256"],
        "role_lock_sha256": candidate["role_lock"]["lock_sha256"],
        "cluster_power_audit_sha256": audit["audit_sha256"],
        "formal_data_policy_sha256": candidate["formal_data_policy"][
            "policy_sha256"
        ],
        "candidate_membership_assignment_inputs": audit[
            "candidate_membership_assignment_inputs"
        ],
        "membership_identity_order_fixed_before_power_computation": audit[
            "membership_identity_order_fixed_before_power_computation"
        ],
        "membership_lock_timing": audit["membership_lock_timing"],
        "historical_outcomes_may_determine_preregistered_prefix_length": audit[
            "historical_outcomes_may_determine_preregistered_prefix_length"
        ],
        "historical_outcomes_select_identity_order_or_within_prefix_members": audit[
            "historical_outcomes_select_identity_order_or_within_prefix_members"
        ],
        "cross_role_historical_outcomes_affect_lane_identity_order_or_members": audit[
            "cross_role_historical_outcomes_affect_lane_identity_order_or_members"
        ],
        "expansion_lane_algorithm": audit["expansion_lane_algorithm"],
        "frozen_role_lane_video_ids_sha256s": dict(
            audit["frozen_role_lane_video_ids_sha256s"]
        ),
        "reserved_core_video_ids_sha256": audit[
            "reserved_core_video_ids_sha256"
        ],
        "role_specific_expansion_lanes_disjoint": audit[
            "role_specific_expansion_lanes_disjoint"
        ],
        "power_result_use": audit["power_result_use"],
        "power_expansion_report_sha256": audit[
            "power_expansion_report_sha256"
        ],
        "bootstrap_replicates": 10_000,
        "pass_claim_available_before_stage_commit": False,
    }
    return {**base, "binding_sha256": semantic_sha256(base)}


def test_g4_stage_uses_shared_roles_atomic_candidate_validator(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        a4_roles,
        "_require_persisted_trusted_power_computation_receipt",
        lambda receipt: dict(receipt),
    )
    assert a4_stages.G4_ATOMIC_BUNDLE_BINDING_SCHEMA == (
        "c28f_g4_atomic_bundle_binding_v3"
    )
    assert set(a4_stages.G4_ATOMIC_BUNDLE_BINDING_FIELDS) == {
        "schema_version", "status", "goal_id", "attempt_id",
        "role_input_handle_sha256", "power_replay_capability_sha256",
        "power_replay_capability_binding_sha256",
        "trusted_power_computation_receipt_sha256",
        "atomic_candidate_sha256", "role_lock_sha256",
        "cluster_power_audit_sha256", "formal_data_policy_sha256",
        "candidate_membership_assignment_inputs",
        "membership_identity_order_fixed_before_power_computation",
        "membership_lock_timing",
        "historical_outcomes_may_determine_preregistered_prefix_length",
        "historical_outcomes_select_identity_order_or_within_prefix_members",
        "cross_role_historical_outcomes_affect_lane_identity_order_or_members",
        "expansion_lane_algorithm", "frozen_role_lane_video_ids_sha256s",
        "reserved_core_video_ids_sha256",
        "role_specific_expansion_lanes_disjoint", "power_result_use",
        "power_expansion_report_sha256",
        "bootstrap_replicates", "pass_claim_available_before_stage_commit",
        "binding_sha256",
    }
    assert set(a4_stages.G4_ATOMIC_CANDIDATE_FIELDS) == {
        "schema_version", "status", "goal_id", "attempt_id",
        "role_input_snapshot_sha256",
        "power_replay_capability_binding_sha256",
        "power_replay_capability_sha256", "role_manifests", "role_lock",
        "cluster_power_audit", "formal_data_policy",
        "artifact_candidate_hashes", "atomic_all_or_none",
        "control_commit_required", "persistence_claimed_by_pure_function",
        "atomic_candidate_sha256",
    }
    assert len(a4_stages.G4_TRUSTED_POWER_COMPUTATION_RECEIPT_FIELDS) == 18
    assert len(a4_stages.G4_POWER_EVIDENCE_BINDING_FIELDS) == 21
    assert {
        "power_replay_capability_sha256",
        "trusted_power_computation_receipt",
        "power_evaluator_contract_sha256",
    }.issubset(a4_stages.G4_POWER_EVIDENCE_BINDING_FIELDS)
    assert set(a4_stages.G4_CLUSTER_POWER_AUDIT_FIELDS) == {
        "schema_version", "status", "role_lock_sha256",
        "projected_query_count", "gt_video_cluster_count",
        "cluster_size_distribution", "role_minimum_margins",
        "train_fit_core_queries", "train_fit_core_gt_video_clusters",
        "minimum_query_counts", "minimum_effects_ratio",
        "historical_reference_population",
        "candidate_membership_assignment_inputs",
        "membership_identity_order_fixed_before_power_computation",
        "historical_outcomes_select_identity_order_or_within_prefix_members",
        "cross_role_historical_outcomes_affect_lane_identity_order_or_members",
        "historical_outcomes_may_determine_preregistered_prefix_length",
        "membership_lock_timing", "expansion_lane_algorithm",
        "frozen_role_lane_video_ids_sha256s",
        "reserved_core_video_ids_sha256",
        "role_specific_expansion_lanes_disjoint", "power_result_use",
        "power_results",
        "power_expansion_report", "power_expansion_report_sha256",
        "power_evidence_binding", "effect_size_or_performance_claim",
        "interpretation", "audit_sha256",
    }
    assert set(a4_stages.G4_POWER_EXPANSION_REPORT_FIELDS) == {
        "status", "expansion_policy", "minimum_effect_registry_sha256",
        "effect_outcome_registry_sha256", "effect_threshold_reduction_count",
        "historical_reference_population", "membership_assignment_inputs",
        "historical_outcomes_select_identity_order_or_within_prefix_members",
        "cross_role_historical_outcomes_affect_lane_identity_order_or_members",
        "historical_outcomes_may_determine_preregistered_prefix_length",
        "membership_lock_timing", "expansion_lane_algorithm",
        "frozen_core_prefix_video_ids_sha256",
        "frozen_role_lane_video_ids_sha256s",
        "reserved_core_video_ids_sha256",
        "role_specific_expansion_lanes_disjoint", "adjustment_rounds",
        "role_results", "trusted_power_computation_receipt",
    }
    assert set(a4_stages.G4_ROLE_ASSIGNMENT_FIELDS) == {
        "algorithm", "seed", "cluster_order", "cluster_atomic",
        "candidate_membership_source",
        "historical_reference_population_sha256",
        "historical_outcomes_select_identity_order_or_within_prefix_members",
        "cross_role_historical_outcomes_affect_lane_identity_order_or_members",
        "historical_outcomes_may_determine_preregistered_prefix_length",
        "power_expansion_lane_algorithm", "membership_lock_timing",
        "power_expansion_report_sha256", "assignment_contract_sha256",
    }
    candidate, role_binding, power_binding = _g4_atomic_candidate_fixture()
    assert a4_roles.validate_g4_atomic_candidate(candidate) == candidate
    context = _post_index_context(
        tmp_path / "output",
        action="G4_ROLE_POLICY_LOCK",
        status="ID_MAPPING_COMPLETE",
    )
    validated = _validate_g4_atomic_candidates(
        candidates=candidate,
        context=context,
        role_binding=role_binding,
        power_binding=power_binding,
        power_replay_capability_sha256=_sha("g4-power-capability"),
    )
    assert validated[0] == candidate["atomic_candidate_sha256"]
    assert validated[2]["lock_sha256"] == candidate["role_lock"]["lock_sha256"]
    power_lineage = validated[3]["power_evidence_binding"]
    assert power_lineage["status"] == "CONTROLLER_RECOMPUTED_TRUSTED_POWER"
    assert power_lineage["power_replay_capability_sha256"] == _sha(
        "g4-power-capability"
    )
    assert power_lineage["evidence_sha256"] == power_lineage[
        "trusted_power_computation_receipt"
    ]["receipt_sha256"]
    audit = validated[3]
    assert audit[
        "historical_outcomes_may_determine_preregistered_prefix_length"
    ] is True
    assert audit[
        "historical_outcomes_select_identity_order_or_within_prefix_members"
    ] is False
    assert audit[
        "cross_role_historical_outcomes_affect_lane_identity_order_or_members"
    ] is False
    assert audit["membership_lock_timing"] == (
        "ALL_PREFIX_EXPANSION_BEFORE_ROLE_MANIFEST_LOCK"
    )
    assert audit["expansion_lane_algorithm"] == (
        a4_roles.POWER_EXPANSION_LANE_ALGORITHM
    )
    assert audit["role_specific_expansion_lanes_disjoint"] is True
    assert audit["membership_identity_order_fixed_before_power_computation"] is True
    binding = _g4_v3_bundle_binding_fixture(candidate)
    assert a4_stages._validate_g4_atomic_bundle_binding(binding) == binding


@pytest.mark.parametrize(
    "mutation",
    (
        "old_schema",
        "cross_role_selection",
        "overlapping_lanes",
        "lane_hash_set",
        "reserved_core_hash",
        "power_use",
    ),
)
def test_g4_v3_bundle_binding_rejects_lane_summary_drift(mutation: str) -> None:
    candidate, _role_binding, _power_binding = _g4_atomic_candidate_fixture()
    binding = _g4_v3_bundle_binding_fixture(candidate)
    if mutation == "old_schema":
        binding["schema_version"] = "c28f_g4_atomic_bundle_binding_v2"
    elif mutation == "cross_role_selection":
        binding[
            "cross_role_historical_outcomes_affect_lane_identity_order_or_members"
        ] = True
    elif mutation == "overlapping_lanes":
        binding["role_specific_expansion_lanes_disjoint"] = False
    elif mutation == "lane_hash_set":
        binding["frozen_role_lane_video_ids_sha256s"] = {
            "train_fit_mechanism_dev": _sha("one-lane-only")
        }
    elif mutation == "reserved_core_hash":
        binding["reserved_core_video_ids_sha256"] = "not-a-sha"
    else:
        binding["power_result_use"] = "POWER_SELECTS_ANY_MEMBER"
    binding["binding_sha256"] = semantic_sha256(
        binding,
        excluded_fields=("binding_sha256",),
    )
    with pytest.raises(StageContractError, match="G4 atomic bundle binding"):
        a4_stages._validate_g4_atomic_bundle_binding(binding)


@pytest.mark.parametrize(
    "mutation",
    [
        "old_schema",
        "missing_field",
        "power_capability",
        "power_lineage",
        "identity_selection",
        "cross_role_selection",
        "overlapping_lanes",
        "ambiguous_membership_claim",
        "reference_membership_join",
        "artifact_closure",
        "atomic_flag",
    ],
)
def test_g4_shared_atomic_contract_rejects_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    monkeypatch.setattr(
        a4_roles,
        "_require_persisted_trusted_power_computation_receipt",
        lambda receipt: dict(receipt),
    )
    candidate, role_binding, power_binding = _g4_atomic_candidate_fixture()
    drifted = dict(candidate)
    if mutation == "old_schema":
        drifted["schema_version"] = "c28f_g4_role_policy_candidates_v0"
    elif mutation == "missing_field":
        drifted.pop("formal_data_policy")
    elif mutation == "power_capability":
        drifted["power_replay_capability_sha256"] = _sha(
            "wrong-power-capability"
        )
    elif mutation == "power_lineage":
        audit = dict(drifted["cluster_power_audit"])
        lineage = dict(audit["power_evidence_binding"])
        lineage["power_replay_capability_binding_sha256"] = _sha(
            "wrong-power-binding"
        )
        audit["power_evidence_binding"] = lineage
        audit["audit_sha256"] = semantic_sha256(
            audit,
            excluded_fields=("audit_sha256",),
        )
        drifted["cluster_power_audit"] = audit
    elif mutation == "identity_selection":
        audit = dict(drifted["cluster_power_audit"])
        audit[
            "historical_outcomes_select_identity_order_or_within_prefix_members"
        ] = True
        audit["audit_sha256"] = semantic_sha256(
            audit,
            excluded_fields=("audit_sha256",),
        )
        drifted["cluster_power_audit"] = audit
    elif mutation == "cross_role_selection":
        audit = dict(drifted["cluster_power_audit"])
        audit[
            "cross_role_historical_outcomes_affect_lane_identity_order_or_members"
        ] = True
        audit["audit_sha256"] = semantic_sha256(
            audit,
            excluded_fields=("audit_sha256",),
        )
        drifted["cluster_power_audit"] = audit
    elif mutation == "overlapping_lanes":
        audit = dict(drifted["cluster_power_audit"])
        audit["role_specific_expansion_lanes_disjoint"] = False
        audit["audit_sha256"] = semantic_sha256(
            audit,
            excluded_fields=("audit_sha256",),
        )
        drifted["cluster_power_audit"] = audit
    elif mutation == "ambiguous_membership_claim":
        audit = dict(drifted["cluster_power_audit"])
        audit["interpretation"] = (
            "INDEPENDENT_HISTORICAL_REFERENCE_POWER_"
            "NO_PERFORMANCE_MEMBERSHIP_SELECTION"
        )
        audit["audit_sha256"] = semantic_sha256(
            audit,
            excluded_fields=("audit_sha256",),
        )
        drifted["cluster_power_audit"] = audit
    elif mutation == "reference_membership_join":
        audit = dict(drifted["cluster_power_audit"])
        reference = dict(audit["historical_reference_population"])
        reference["train_fit_desc_id_join_performed"] = True
        reference["reference_population_sha256"] = semantic_sha256(
            reference,
            excluded_fields=("reference_population_sha256",),
        )
        audit["historical_reference_population"] = reference
        audit["audit_sha256"] = semantic_sha256(
            audit,
            excluded_fields=("audit_sha256",),
        )
        drifted["cluster_power_audit"] = audit
    elif mutation == "artifact_closure":
        closure = dict(drifted["artifact_candidate_hashes"])
        closure["cluster_power_audit_sha256"] = _sha("wrong-audit")
        drifted["artifact_candidate_hashes"] = closure
    else:
        drifted["atomic_all_or_none"] = False
    drifted["atomic_candidate_sha256"] = semantic_sha256(
        drifted,
        excluded_fields=("atomic_candidate_sha256",),
    )
    context = _post_index_context(
        tmp_path / "output",
        action="G4_ROLE_POLICY_LOCK",
        status="ID_MAPPING_COMPLETE",
    )
    with pytest.raises(StageContractError, match="shared atomic"):
        _validate_g4_atomic_candidates(
            candidates=drifted,
            context=context,
            role_binding=role_binding,
            power_binding=power_binding,
            power_replay_capability_sha256=_sha("g4-power-capability"),
        )


def test_rank_transitions_parquet_is_genuine_and_exact() -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    rows = [
        {
            "query_id": 3,
            "pooled_gt_rank": 12,
            "late_gt_rank": 4,
            "final_video_gt_rank": 2,
            "pooled_to_late_rank_delta": 8,
            "broad_hit_at_1000": True,
            "late_hit_at_200": True,
            "broad_hit_but_late_lost": False,
        },
        {
            "query_id": 8,
            "pooled_gt_rank": None,
            "late_gt_rank": None,
            "final_video_gt_rank": None,
            "pooled_to_late_rank_delta": None,
            "broad_hit_at_1000": False,
            "late_hit_at_200": False,
            "broad_hit_but_late_lost": False,
        },
    ]
    rows.extend(
        {
            "query_id": query_id,
            "pooled_gt_rank": 100,
            "late_gt_rank": 50,
            "final_video_gt_rank": 25,
            "pooled_to_late_rank_delta": 50,
            "broad_hit_at_1000": True,
            "late_hit_at_200": True,
            "broad_hit_but_late_lost": False,
        }
        for query_id in range(9, 1_032)
    )
    encoded, contract = _rank_transitions_parquet(
        per_query=rows,
        source_binding_sha256=_sha("completed-eval-capability"),
    )
    assert encoded[:4] == b"PAR1" and encoded[-4:] == b"PAR1"
    table = pq.read_table(pa.BufferReader(encoded))
    assert table.to_pylist() == rows
    assert table.column_names == list(rows[0])
    assert [str(field.type) for field in table.schema] == [
        "int64",
        "int32",
        "int32",
        "int32",
        "int32",
        "bool",
        "bool",
        "bool",
    ]
    parquet_file = pq.ParquetFile(pa.BufferReader(encoded))
    assert parquet_file.metadata.num_row_groups == 2
    assert [
        parquet_file.metadata.row_group(index).num_rows for index in range(2)
    ] == [1_024, 1]
    for row_group_index in range(parquet_file.metadata.num_row_groups):
        row_group = parquet_file.metadata.row_group(row_group_index)
        for column_index in range(row_group.num_columns):
            column = row_group.column(column_index)
            assert "RLE_DICTIONARY" not in column.encodings
            assert "PLAIN_DICTIONARY" not in column.encodings
            assert column.statistics is not None
    assert table.schema.metadata[b"c28f.rows_sha256"].decode("ascii") == contract[
        "semantic_rows_sha256"
    ]
    assert contract["parquet_file_sha256"] == hashlib.sha256(encoded).hexdigest()


def test_completed_f0_safe_stats_recover_joint_vcmr_margins_and_masks() -> None:
    assert QUERY_BUCKET_RULE_CONTRACT["query_feature_token_count_rule"] == (
        "FORWARD_SEALED_QUERY_MASK_SUM_USED_BY_MODEL"
    )
    assert QUERY_BUCKET_RULE_CONTRACT["late_maxsim_token_count_rule"] == (
        "FORWARD_SEALED_LATE_MAXSIM_QUERY_MASK_SUM"
    )
    assert QUERY_BUCKET_RULE_CONTRACT["feature_to_late_maxsim_invariant"] == (
        "EXACT_SAME_FORWARD_QUERY_MASK_COUNTS_EQUAL"
    )
    assert QUERY_BUCKET_RULE_CONTRACT["query_text_token_count_rule"] == (
        "UNICODE_CASEFOLD_THEN_ASCII_PUNCTUATION_TO_SPACE_"
        "THEN_WHITESPACE_SPLIT_COUNT"
    )
    assert QUERY_BUCKET_RULE_CONTRACT["gt_video_clip_count_rule"] == (
        "MAX_1_CEIL_GT_VIDEO_DURATION_SEC_DIV_1P5"
    )
    source_rows = (
        _f0_completed_row(0, gt_present=True, top1_iou=0.8),
        _f0_completed_row(1, gt_present=False, top1_iou=0.0),
    )
    evaluation = _evaluate_completed_f0_rows(source_rows)
    assert evaluation["retrieval_metrics"]["pooled_true_VR_R@1000"] == 0.5
    assert evaluation["joint_retrieval_metrics"]["joint_marginal_VR_R@1"] == 0.5
    assert evaluation["vcmr_metrics"]["VCMR_R@1_IoU@0.7"] == 0.5
    assert evaluation["error_metrics"]["wrong_video_top1_rate"] == 0.5
    assert evaluation["margin_diagnostics"]["pooled"] == {
        "status": "available",
        "reason": None,
        "cut_margin_definition": "score[k]-score[k+1]_ONE_BASED",
        "top1_top2_available_query_count": 2,
        "top10_top11_available_query_count": 2,
        "top100_top101_available_query_count": 2,
        "gt_vs_best_wrong_available_query_count": 1,
        "mean_top1_top2_score_margin": 0.3,
        "mean_top10_top11_score_margin": 0.2,
        "mean_top100_top101_score_margin": 0.1,
        "mean_gt_vs_best_wrong_score_margin": 0.25,
    }
    assert evaluation["modality_mask_diagnostics"]["visual"][
        "cell_denominator"
    ] == 25_600
    assert evaluation["modality_mask_diagnostics"]["subtitle"][
        "missing_count"
    ] == 6_400
    assert evaluation["modality_mask_diagnostics"]["subtitle"][
        "empty_candidate_count"
    ] == 20
    assert evaluation["modality_mask_diagnostics"][
        "visual_subtitle_overlap_valid_count"
    ] == 18_000
    assert evaluation["modality_mask_diagnostics"][
        "joint_component_relation_verified"
    ] is True
    mask_shift = a4_stages._f0_modality_mask_shift(
        evaluation["modality_mask_diagnostics"],
        evaluation["modality_mask_diagnostics"],
    )
    assert mask_shift["subtitle"]["empty_candidate_count"][
        "f0_b_minus_f0_a"
    ] == 0.0
    assert mask_shift["visual_subtitle_overlap_valid_count"][
        "f0_b_minus_f0_a"
    ] == 0.0
    assert mask_shift["joint_component_relation_verified"] is True
    assert evaluation["video_distribution_diagnostics"][
        "gt_video_cluster_count"
    ] == 2

    bucket_report, query_rows = _build_f0_bucket_report(
        source_rows=source_rows,
        metric_rows=evaluation["per_query"],
        bucket_rule_contract=QUERY_BUCKET_RULE_CONTRACT,
        completed_eval_capability_sha256=_sha("completed-safe-stats"),
        query_manifest_sha256=_sha("safe-stat-query-manifest"),
    )
    assert len(query_rows) == 2
    assert set(bucket_report["dimensions"]) >= {
        "query_feature_token_count",
        "query_text_token_count",
        "late_maxsim_token_count",
        "visual_valid_mask_ratio",
        "subtitle_valid_mask_ratio",
        "joint_valid_mask_ratio",
        "pooled_cut_margin_at_1",
        "pooled_cut_margin_at_10",
        "pooled_cut_margin_at_100",
        "moment_duration_sec",
        "video_raw_clip_count",
    }
    duration_bucket = bucket_report["dimensions"]["moment_duration_sec"]
    assert set(duration_bucket) == {"GT_2_LE_5"}
    assert duration_bucket["GT_2_LE_5"]["vcmr"][
        "VCMR_R@1_IoU@0.7"
    ] == 0.5
    bucket_shift = a4_stages._f0_bucket_shift(
        bucket_report["dimensions"],
        bucket_report["dimensions"],
    )
    duration_shift = bucket_shift["moment_duration_sec"]["label_shifts"][
        "GT_2_LE_5"
    ]
    assert duration_shift["true_video_retrieval_shift"][
        "pooled_R@1000"
    ]["f0_b_minus_f0_a"] == 0.0
    assert duration_shift["error_metric_shift"][
        "top1_false_positive_rate"
    ]["f0_b_minus_f0_a"] == 0.0
    assert duration_shift["rank_transition_shift"][
        "mean_pooled_minus_late_rank_delta"
    ]["f0_b_minus_f0_a"] == 0.0
    assert duration_shift["margin_shift"][
        "pooled_top100_top101_score_margin"
    ]["f0_b_minus_f0_a"] == 0.0


def test_completed_f0_rejects_feature_to_late_maxsim_count_divergence() -> None:
    row = dict(_f0_completed_row(0, gt_present=True, top1_iou=0.8))
    row["late_maxsim_token_count"] = row["query_feature_token_count"] + 1
    with pytest.raises(
        a4_stages.StageContractError,
        match="frozen scalar/bucket sufficient-stat contract mismatch",
    ):
        _evaluate_completed_f0_rows((row,))


def test_completed_f0_accepts_fully_missing_subtitle_mask_with_exact_or() -> None:
    row = copy.deepcopy(
        _f0_completed_row(0, gt_present=True, top1_iou=0.8)
    )
    coverage = row["late_candidate_mask_coverage"]
    coverage["subtitle"] = {
        "candidate_count": 200,
        "denominator": 12_800,
        "valid_count": 0,
        "missing_count": 12_800,
        "valid_ratio": 0.0,
        "missing_ratio": 1.0,
        "empty_candidate_count": 200,
    }
    coverage["visual_subtitle_overlap_valid_count"] = 0
    coverage["joint"] = dict(coverage["visual"])
    evaluation = _evaluate_completed_f0_rows((row,))
    assert evaluation["modality_mask_diagnostics"]["subtitle"][
        "empty_candidate_count"
    ] == 200
    assert evaluation["modality_mask_diagnostics"]["joint"][
        "valid_count"
    ] == evaluation["modality_mask_diagnostics"]["visual"]["valid_count"]


@pytest.mark.parametrize(
    "mutation",
    ("overlap_drift", "visual_empty_candidate", "false_relation_claim"),
)
def test_completed_f0_rejects_mask_structure_or_relation_drift(
    mutation: str,
) -> None:
    row = copy.deepcopy(
        _f0_completed_row(0, gt_present=True, top1_iou=0.8)
    )
    coverage = row["late_candidate_mask_coverage"]
    if mutation == "overlap_drift":
        coverage["visual_subtitle_overlap_valid_count"] += 1
    elif mutation == "visual_empty_candidate":
        coverage["visual"]["empty_candidate_count"] = 1
    else:
        coverage["joint_component_relation_verified"] = False
    with pytest.raises(a4_stages.StageContractError, match="modality mask"):
        _evaluate_completed_f0_rows((row,))


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("moment_duration_sec", 0.0),
        ("gt_relative_position", 1.1),
        ("verb_bucket", "POSTHOC_METRIC_BUCKET"),
    ),
)
def test_completed_f0_rejects_invalid_bucket_scalars(
    field: str,
    value: object,
) -> None:
    row = dict(_f0_completed_row(0, gt_present=True, top1_iou=0.8))
    row[field] = value
    with pytest.raises(
        a4_stages.StageContractError,
        match="frozen scalar/bucket sufficient-stat contract mismatch",
    ):
        _evaluate_completed_f0_rows((row,))


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("joint_marginal_gt_rank", 201, "joint marginal rank invalid"),
        ("legacy_joint_unique_video_gt_rank", 101, "legacy NMS"),
        ("raw_exact_duplicate_proposal_count", 12_801, "boolean/count"),
    ),
)
def test_completed_f0_rejects_out_of_domain_joint_sufficient_stats(
    field: str,
    value: int,
    message: str,
) -> None:
    row = dict(_f0_completed_row(0, gt_present=True, top1_iou=0.8))
    row[field] = value
    with pytest.raises(a4_stages.StageContractError, match=message):
        _evaluate_completed_f0_rows((row,))


def test_completed_f0_rejects_missing_joint_rank_for_present_candidate() -> None:
    row = dict(_f0_completed_row(0, gt_present=True, top1_iou=0.8))
    row["joint_marginal_gt_rank"] = None
    with pytest.raises(
        a4_stages.StageContractError,
        match="joint marginal rank invalid",
    ):
        _evaluate_completed_f0_rows((row,))


def test_top1_false_positive_is_not_wrong_video_alias() -> None:
    row = _f0_completed_row(0, gt_present=True, top1_iou=0.2)
    assert row["wrong_video_top1"] is False
    assert row["correct_video_wrong_span_top1"] is True
    assert row["top1_false_positive"] is True
    evaluation = _evaluate_completed_f0_rows((row,))
    assert evaluation["error_metrics"]["wrong_video_top1_rate"] == 0.0
    assert evaluation["error_metrics"][
        "correct_video_wrong_span_top1_rate"
    ] == 1.0
    assert evaluation["error_metrics"]["top1_false_positive_rate"] == 1.0


def test_g3_and_g5_reject_caller_supplied_completed_eval_mappings(
    tmp_path: Path,
) -> None:
    g3_context = _post_index_context(
        tmp_path / "output",
        action="G3_F0_A_BASELINE_FORENSICS",
        status="METRIC_CONTRACT_OK",
    )
    with pytest.raises(
        StageContractError,
        match="A4CompletedEvalArtifactCapability",
    ):
        build_g3_bundle(
            context=g3_context,
            completed_eval_capability={"retrieval_rows": []},
        )
    g5_context = _post_index_context(
        tmp_path / "output",
        action="G5_F0_B_EXPOSURE_REPLICATION",
        status="ROLE_POLICY_LOCKED",
    )
    with pytest.raises(
        StageContractError,
        match="A4CompletedEvalArtifactCapability",
    ):
        build_g5_bundle(
            context=g5_context,
            completed_eval_capability={"retrieval_rows": []},
        )


def test_g3_and_completed_eval_entry_reject_status_only_without_ready_index(
    tmp_path: Path,
) -> None:
    absent = _context(tmp_path / "status-only-output")
    status_only = StageContext(
        **{
            **absent.__dict__,
            "current_action": "G3_F0_A_BASELINE_FORENSICS",
            "current_status": "METRIC_CONTRACT_OK",
        }
    )
    with pytest.raises(StageContractError, match="READY"):
        build_g3_bundle(
            context=status_only,
            completed_eval_capability={"retrieval_rows": []},
        )
    with pytest.raises(StageContractError, match="committed READY"):
        a4_stages._consume_completed_eval_for_stage(
            context=status_only,
            completed_eval_capability={"retrieval_rows": []},
            token_id="F0_A",
        )


@pytest.mark.parametrize(
    ("g_broad", "g_keep", "g_front", "expected_stage"),
    [
        (0.89, 0.99, 0.99, "POOLED_BROAD"),
        (0.90, 0.89, 0.99, "LATE_RERANK"),
        (0.90, 0.90, 0.79, "FINAL_VIDEO_ORDERING"),
        (0.90, 0.90, 0.80, "NO_FAILURE"),
    ],
)
def test_f0_route_uses_frozen_sequential_thresholds(
    g_broad: float,
    g_keep: float,
    g_front: float,
    expected_stage: str,
) -> None:
    g_late = g_broad * g_keep
    result = _frozen_retrieval_gate_result(
        retrieval={
            "gate_metrics": {
                "G_broad": g_broad,
                "G_late": g_late,
                "G_keep": g_keep,
                "G_front": g_front,
                "L_lost": 1.0 - g_keep,
                "G_late_product_residual": 0.0,
                "G_late_product_invariant": True,
            }
        },
        query_manifest_sha256=_sha("f0-query-manifest"),
        completed_eval_capability_sha256=_sha("completed-eval-capability"),
    )
    assert result["thresholds"] == {
        "G_broad_minimum": 0.90,
        "G_keep_minimum": 0.90,
        "G_front_minimum": 0.80,
        "G_late_product_absolute_tolerance": 1e-8,
    }
    assert result["earliest_failing_stage"] == expected_stage
    assert result["threshold_changed_after_observation"] is False


def test_f0_route_rejects_product_invariant_drift() -> None:
    with pytest.raises(StageContractError, match="product invariant"):
        _frozen_retrieval_gate_result(
            retrieval={
                "gate_metrics": {
                    "G_broad": 0.9,
                    "G_late": 0.7,
                    "G_keep": 0.9,
                    "G_front": 0.8,
                    "L_lost": 0.1,
                    "G_late_product_residual": 0.11,
                    "G_late_product_invariant": False,
                }
            },
            query_manifest_sha256=_sha("f0-query-manifest"),
            completed_eval_capability_sha256=_sha("completed-eval-capability"),
        )


def test_f0_route_accepts_zero_broad_as_product_not_applicable() -> None:
    result = _frozen_retrieval_gate_result(
        retrieval={
            "gate_metrics": {
                "G_broad": 0.0,
                "G_late": 0.0,
                "G_keep": None,
                "G_front": 0.0,
                "L_lost": None,
                "G_late_product_residual": None,
                "G_late_product_invariant": False,
            }
        },
        query_manifest_sha256=_sha("zero-broad-query-manifest"),
        completed_eval_capability_sha256=_sha("zero-broad-eval"),
    )
    assert result["earliest_failing_stage"] == "POOLED_BROAD"
    assert result["G_late_product_check_status"] == (
        "NOT_APPLICABLE_ZERO_BROAD_HITS"
    )
    assert result["gate_pass"]["G_late_product_invariant"] is True


def _g6_v2_crash_receipt_fixture() -> dict:
    root_handle_sha = _sha("g6-root-handle")
    crash_point = "BEFORE_CHECKPOINT_COMMIT"
    child_root_name = "scenario-before-checkpoint-commit"
    child_root_path = "/tmp/g6-parent/%s" % child_root_name
    parent_identity_sha = _sha("g6-parent-identity")
    child_contract_sha = _sha("g6-child-contract")
    child_binding_sha = semantic_sha256(
        {
            "synthetic_root_handle_sha256": root_handle_sha,
            "crash_point": crash_point,
            "child_root_name": child_root_name,
            "child_root_path": child_root_path,
            "parent_descriptor_identity_sha256": parent_identity_sha,
            "child_root_contract_sha256": child_contract_sha,
        }
    )
    expected_artifacts = a4_f1._synthetic_atomic_recovery_expected_material(
        crash_point
    )["expected_artifact_bytes"]
    inventory = [
        {
            "name": name,
            "size_bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "device": 41,
            "inode": 101 + index,
            "nlink": 1,
            "mode": "0o600",
            "mtime_ns": 1_700_000_000_000_000_000 + index,
            "ctime_ns": 1_700_000_000_100_000_000 + index,
            "mount_id": 7,
        }
        for index, (name, payload) in enumerate(
            sorted(expected_artifacts.items())
        )
    ]
    postimage_base = {
        "schema_version": a4_stages.G6_ACTUAL_POSTIMAGE_SCHEMA,
        "synthetic_root_handle_sha256": root_handle_sha,
        "crash_point": crash_point,
        "root_identity": {
            "canonical_path": child_root_path,
            "parent_canonical_path": "/tmp/g6-parent",
            "device": 41,
            "inode": 100,
            "nlink": 2,
            "mode": "0o700",
            "mtime_ns": 1_700_000_000_200_000_000,
            "ctime_ns": 1_700_000_000_300_000_000,
            "mount_id": 7,
            "parent_descriptor_identity_sha256": parent_identity_sha,
            "child_root_binding_sha256": child_binding_sha,
        },
        "artifact_count": len(inventory),
        "artifact_inventory": inventory,
        "artifact_inventory_sha256": hashlib.sha256(
            canonical_json_bytes(inventory)
        ).hexdigest(),
    }
    actual_postimage = {
        **postimage_base,
        "postimage_sha256": semantic_sha256(postimage_base),
    }
    receipt_base = {
        "schema_version": a4_stages.G6_CRASH_RECEIPT_SCHEMA,
        "status": "PASS",
        "goal_id": "goal-c28f-test",
        "attempt_id": "G0-A4-test",
        "synthetic_root_handle_sha256": root_handle_sha,
        "crash_point": crash_point,
        "child_root_name": child_root_name,
        "child_root_path": child_root_path,
        "parent_descriptor_identity_sha256": parent_identity_sha,
        "child_root_contract_sha256": child_contract_sha,
        "child_root_binding_sha256": child_binding_sha,
        "evidence_sha256": _sha("g6-harness-evidence"),
        "evidence_file_sha256": _sha("g6-harness-evidence-file"),
        "actual_postimage": actual_postimage,
        "controller_reread_postimage_sha256": actual_postimage[
            "postimage_sha256"
        ],
        "real_data_optimizer_updates": 0,
        "extra_model_forward_splits": 0,
    }
    return {
        **receipt_base,
        "receipt_sha256": semantic_sha256(receipt_base),
    }


def test_g6_v2_crash_and_postimage_schemas_are_frozen_exactly() -> None:
    assert a4_stages.G6_CRASH_RECEIPT_SCHEMA == (
        "c28f_g6_controlled_synthetic_crash_receipt_v2"
    )
    assert set(a4_stages.G6_CRASH_RECEIPT_FIELDS) == {
        "schema_version", "status", "goal_id", "attempt_id",
        "synthetic_root_handle_sha256", "crash_point", "child_root_name",
        "child_root_path", "parent_descriptor_identity_sha256",
        "child_root_contract_sha256", "child_root_binding_sha256",
        "evidence_sha256", "evidence_file_sha256", "actual_postimage",
        "controller_reread_postimage_sha256",
        "real_data_optimizer_updates", "extra_model_forward_splits",
        "receipt_sha256",
    }
    assert set(a4_stages.G6_ACTUAL_POSTIMAGE_FIELDS) == {
        "schema_version", "synthetic_root_handle_sha256", "crash_point",
        "root_identity", "artifact_count", "artifact_inventory",
        "artifact_inventory_sha256", "postimage_sha256",
    }
    assert a4_stages.G6_ACTUAL_POSTIMAGE_SCHEMA == (
        "c28f_g6_synthetic_actual_postimage_v2"
    )
    assert set(a4_stages.G6_ACTUAL_POSTIMAGE_ROOT_IDENTITY_FIELDS) == {
        "canonical_path", "parent_canonical_path", "device", "inode",
        "nlink", "mode", "mtime_ns", "ctime_ns", "mount_id",
        "parent_descriptor_identity_sha256", "child_root_binding_sha256",
    }
    assert set(a4_stages.G6_ACTUAL_POSTIMAGE_INVENTORY_ROW_FIELDS) == {
        "name", "size_bytes", "sha256", "device", "inode", "nlink",
        "mode", "mtime_ns", "ctime_ns", "mount_id",
    }
    assert a4_stages.G6_CONTROL_BINDING_SCHEMA == (
        "c28f_g6_control_binding_v2"
    )
    assert a4_stages.G6_GLOBAL_LEDGER_SCHEMA == (
        "c28f_a4_g6_global_ledger_snapshot_v1"
    )
    assert set(a4_stages.G6_GLOBAL_LEDGER_FIELDS) == {
        "schema_version", "goal_id", "attempt_id",
        "committed_transaction_id", "committed_state_sha256",
        "committed_event_sha256", "event_journal_sha256",
        "eval_token_states", "model_forward_evaluations",
        "synthetic_crash_scenario_count",
        "synthetic_crash_evidence_set_sha256",
        "real_data_optimizer_updates", "extra_model_forward_splits",
        "snapshot_sha256",
    }
    assert set(a4_stages.G6_ATOMIC_CRASH_EVIDENCE_FIELDS) == {
        "status", "required_crash_points", "crash_evidence_set_sha256",
        "committed_evidence_sha256s", "evidence_file_sha256s",
        "child_root_binding_sha256s", "actual_postimage_sha256s",
        "controller_reread_postimage_sha256s",
        "artifact_inventory_sha256s", "synthetic_root_handle_sha256",
    }
    assert set(a4_stages.G6_CONTROL_EVIDENCE_BINDING_FIELDS) == {
        "g6_input_handle_sha256", "candidate_report_sha256",
        "global_ledger_snapshot_sha256", "crash_evidence_set_sha256",
        "role_lock_sha256", "g2_replay_evidence_sha256",
        "formal_policy_sha256", "pass_claim_available_before_stage_commit",
    }
    assert set(a4_stages.G6_CONTROL_BINDING_FIELDS) == {
        "schema_version", "status", "goal_id", "attempt_id",
        "g6_input_handle_sha256", "candidate_report_sha256",
        "final_report_sha256", "global_ledger_snapshot_sha256",
        "crash_evidence_set_sha256", "crash_receipt_sha256s",
        "evidence_sha256s", "evidence_file_sha256s",
        "child_root_binding_sha256s", "actual_postimage_sha256s",
        "controller_reread_postimage_sha256s",
        "artifact_inventory_sha256s", "real_data_optimizer_updates",
        "extra_model_forward_splits",
        "pass_claim_available_before_stage_commit", "binding_sha256",
    }
    receipt = _g6_v2_crash_receipt_fixture()
    validated, postimage = a4_stages._validate_g6_crash_receipt(
        receipt,
        goal_id="goal-c28f-test",
        attempt_id="G0-A4-test",
        synthetic_root_handle_sha256=_sha("g6-root-handle"),
        required_crash_points={"BEFORE_CHECKPOINT_COMMIT"},
    )
    assert validated == receipt
    assert postimage == receipt["actual_postimage"]


@pytest.mark.parametrize(
    "mutation",
    ("v1_schema", "child_binding", "artifact_bytes", "reread_postimage"),
)
def test_g6_v2_crash_receipt_rejects_detached_evidence(
    mutation: str,
) -> None:
    receipt = copy.deepcopy(_g6_v2_crash_receipt_fixture())
    if mutation == "v1_schema":
        receipt["schema_version"] = (
            "c28f_g6_controlled_synthetic_crash_receipt_v1"
        )
    elif mutation == "child_binding":
        receipt["child_root_binding_sha256"] = _sha("detached-child")
    elif mutation == "artifact_bytes":
        receipt["actual_postimage"]["artifact_inventory"][0][
            "sha256"
        ] = _sha("detached-artifact")
    else:
        receipt["controller_reread_postimage_sha256"] = _sha(
            "detached-reread"
        )
    receipt["receipt_sha256"] = semantic_sha256(
        receipt,
        excluded_fields=("receipt_sha256",),
    )
    with pytest.raises(StageContractError, match="G6"):
        a4_stages._validate_g6_crash_receipt(
            receipt,
            goal_id="goal-c28f-test",
            attempt_id="G0-A4-test",
            synthetic_root_handle_sha256=_sha("g6-root-handle"),
            required_crash_points={"BEFORE_CHECKPOINT_COMMIT"},
        )


def test_g6_rejects_caller_crash_and_ledger_mapping(tmp_path: Path) -> None:
    context = _post_index_context(
        tmp_path / "output",
        action="G6_F1_PROTOCOL_AND_RECOVERY",
        status="F0B_COMPLETE",
    )
    with pytest.raises(StageContractError, match="A4G6BundleInputHandle"):
        build_g6_f1_completion_bundle(
            context=context,
            g6_input_handle={
                "candidate_f1_report": {},
                "committed_global_ledger_snapshot": {},
                "crash_evidence_receipts": [],
            },
        )


def test_g6_synthetic_recovery_public_api_rejects_bare_path(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        F1ContractError,
        match="G6_EXTERNAL_CHILD_PATH_EXPORT_DISABLED",
    ):
        run_synthetic_atomic_recovery_harness(
            tmp_path / "bare-synthetic-root",
            crash_point="BEFORE_CHECKPOINT_COMMIT",
        )


def test_g6_synthetic_recovery_public_api_rejects_duck_handle() -> None:
    class ForgedSyntheticRootHandle:
        capability_sha256 = _sha("forged-g6-root")

        def consume_child_root_once(self, *, crash_point: str) -> Path:
            raise AssertionError(crash_point)

    with pytest.raises(
        F1ContractError,
        match="G6_EXTERNAL_CHILD_PATH_EXPORT_DISABLED",
    ):
        run_synthetic_atomic_recovery_harness(
            ForgedSyntheticRootHandle(),
            crash_point="DURING_EVAL",
        )


if __name__ == "__main__":
    raise SystemExit("Run through the reviewed C28F stage test launcher only")
