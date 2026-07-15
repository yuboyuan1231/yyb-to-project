from __future__ import annotations

import copy
import hashlib
import inspect
import json
import os
from pathlib import Path

import pytest

from blueprint_e2e_v2.c28f_v5 import a4_control
from blueprint_e2e_v2.c28f_v5 import a4_forward
from blueprint_e2e_v2.c28f_v5 import a4_id_projector
from blueprint_e2e_v2.c28f_v5 import a4_metrics
from blueprint_e2e_v2.c28f_v5 import a4_runner
from blueprint_e2e_v2.c28f_v5 import runtime_control
from blueprint_e2e_v2.c28f_v5 import runtime_runner
from blueprint_e2e_v2.c28f_v5 import runtime_stages


def _completed_eval_retrieval_row() -> dict:
    pooled = [f"video-{index:04d}" for index in range(1_000)]
    late = pooled[:200]
    structure_contract_base = {
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
    structure_contract = {
        **structure_contract_base,
        "contract_sha256": a4_control.semantic_sha256(
            structure_contract_base
        ),
    }
    coverage = {
        "ratio_definition": (
            "VALID_OR_MISSING_BOOL_CELLS_DIVIDED_BY_CANDIDATE_COUNT_TIMES_64"
        ),
        "structure_contract": structure_contract,
        "joint_component_relation_verified": True,
        "visual_subtitle_overlap_valid_count": 12_800,
        **{
            modality: {
                "candidate_count": 200,
                "denominator": 12_800,
                "valid_count": 12_800,
                "missing_count": 0,
                "valid_ratio": 1.0,
                "missing_ratio": 0.0,
                "empty_candidate_count": 0,
            }
            for modality in ("visual", "subtitle", "joint")
        },
    }
    hits = {
        f"VCMR_R@{k}_IoU@{threshold:.1f}": False
        for threshold in (0.5, 0.7)
        for k in (1, 5, 10, 100)
    }
    row = {
        "query_id": 7,
        "gt_video_id": pooled[0],
        "pooled_video_order": pooled,
        "late_video_order": late,
        "final_video_order": list(late),
        "query_type": "vt",
        "query_feature_token_count": 8,
        "query_text_token_count": 8,
        "verb_bucket": "SINGLE_VERB",
        "temporal_connector_bool": False,
        "late_maxsim_token_count": 8,
        "late_candidate_mask_coverage": coverage,
        "moment_duration_sec": 3.0,
        "gt_video_clip_count": 64,
        "gt_relative_position": 0.5,
        "pooled_gt_vs_best_wrong_score_margin": 0.0,
        "late_gt_vs_best_wrong_score_margin": 0.0,
        "final_gt_vs_best_wrong_score_margin": 0.0,
        "joint_marginal_gt_rank": 1,
        "legacy_joint_unique_video_gt_rank": 1,
        "vcmr_hits": hits,
        "top1_iou": 0.0,
        "top1_available": True,
        "wrong_video_top1": False,
        "correct_video_wrong_span_top1": True,
        "top1_false_positive": True,
        "joint_false_positive_mass": 0.25,
        "raw_exact_duplicate_proposal_count": 0,
        "nms_prediction_count": 1,
    }
    for prefix in ("pooled", "late", "final"):
        for cut in ("top1_top2", "top10_top11", "top100_top101"):
            row[f"{prefix}_{cut}_score_margin"] = 0.0
    return row


def test_authorization_is_exactly_content_addressed() -> None:
    assert a4_control.USER_A4_AUTHORIZATION_TEXT == "批准按严格修复方案建立 G0-A4 并继续。"
    assert (
        hashlib.sha256(a4_control.USER_A4_AUTHORIZATION_TEXT.encode("utf-8")).hexdigest()
        == a4_control.USER_A4_AUTHORIZATION_SHA256
    )
    assert (
        hashlib.sha256(
            a4_control.TRAIN_JSONL_INDEX_AUTHORIZATION_TEXT.encode("utf-8")
        ).hexdigest()
        == a4_control.TRAIN_JSONL_INDEX_AUTHORIZATION_SHA256
        == "b0e69db776eaf7fef1bebeb98345bf413b1a26238b9625b26aa186e7dc80607a"
    )
    assert (
        hashlib.sha256(
            a4_control.TRAIN_JSONL_INDEX_RETRY_AUTHORIZATION_TEXT.encode(
                "utf-8"
            )
        ).hexdigest()
        == a4_control.TRAIN_JSONL_INDEX_RETRY_AUTHORIZATION_SHA256
        == "0abd8ce5f79bbb5c4de8451985c70756e2ac0942d7d91e8c349b3cdca4886c39"
    )


def test_train_index_accepts_only_the_final_unterminated_json_object() -> None:
    encoded = b'{"desc_id":7,"desc":"safe"}'
    with pytest.raises(
        a4_control.A4ControlError,
        match="canonical bounded line",
    ):
        a4_control._restricted_top_level_desc_id(encoded)
    desc_id, counters = a4_control._restricted_top_level_desc_id(
        encoded,
        allow_missing_final_newline=True,
    )
    assert desc_id == 7
    assert counters == {
        "selection_key_decode_count": 1,
        "nonselection_top_level_key_count": 1,
        "syntax_skipped_value_count": 1,
    }
    scanner_source = inspect.getsource(
        a4_control._scan_train_jsonl_descriptor_index_once
    )
    assert "if pending:" in scanner_source
    assert "allow_missing_final_newline=True" in scanner_source
    assert scanner_source.index("if pending:") < scanner_source.index(
        '"TRAIN index full EOF/line coverage"'
    )


def test_train_index_retry_claim_binds_aborted_first_open() -> None:
    state = {
        "attempt_id": "G0-A4-" + "a" * 32,
        "state_sha256": "b" * 64,
    }
    parent_event = {"event_sha256": "c" * 64}
    named_identity = {
        "canonical_path": str(a4_control.TRAIN_JSONL),
        "device": 1,
        "inode": 2,
        "size_bytes": 3,
        "nlink": 1,
        "mode": "0o644",
        "mtime_ns": 4,
        "ctime_ns": 5,
    }
    first = a4_control._build_train_jsonl_index_scan_claim(
        state=state,
        parent_event=parent_event,
        named_identity=named_identity,
    )
    aborted = a4_control._build_train_jsonl_index_scan_abort_receipt(first)
    current_state = {
        **state,
        "state_sha256": "d" * 64,
    }
    events = [
        {
            "event_sha256": parent_event["event_sha256"],
            "expected_next_state_sha256": state["state_sha256"],
            "action": "G2_METRIC_CONTRACT",
            "result": "COMMITTED_EXACT_STAGE_POSTIMAGE",
            "transaction_id": "STAGE-COMMIT-" + "e" * 32,
        },
        {
            "event_sha256": "f" * 64,
            "expected_next_state_sha256": current_state["state_sha256"],
            "action": "COMMIT_REVIEWED_CODE_CHANGE",
            "result": (
                "COMMITTED_USER_AUTHORIZED_ONLINE_EXISTING_FILE_POSTIMAGE"
            ),
            "transaction_id": "CODE-CHANGE-" + "1" * 32 + "-COMMIT",
        },
    ]
    a4_control._validate_train_index_claim_parent_online_code_lineage(
        first,
        state=current_state,
        events=events,
    )
    retry = a4_control._build_train_jsonl_index_retry_scan_claim(
        state=current_state,
        parent_event=events[-1],
        named_identity=named_identity,
        failed_claim=first,
        abort_receipt=aborted,
    )
    assert retry["transaction_id"] != first["transaction_id"]
    assert retry["prior_failed_claim_sha256"] == first["claim_sha256"]
    assert retry["prior_abort_receipt_sha256"] == aborted["receipt_sha256"]
    assert retry["authorized_physical_content_open_ordinal"] == 2
    assert a4_control._train_jsonl_index_total_authorized_open_count(retry) == 2
    assert a4_control._validate_train_jsonl_index_scan_claim(retry) == retry
    bootstrap_source = inspect.getsource(
        a4_control.bootstrap_train_jsonl_descriptor_index
    )
    assert "abort_receipt.json" in bootstrap_source
    assert "TRAIN_INDEX_SECOND_SCAN_INDETERMINATE" in bootstrap_source
    assert "_build_train_jsonl_index_retry_scan_claim(" in bootstrap_source
    assert (
        "_validate_train_index_claim_parent_online_code_lineage("
        in bootstrap_source
    )


def test_forward_video_meta_uses_train_object_order_not_external_index(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    video_ids = [f"video-{index:05d}" for index in range(17_435)]
    expected_sha256 = hashlib.sha256(
        "".join(f"{video_id}\n" for video_id in video_ids).encode("utf-8")
    ).hexdigest()
    monkeypatch.setattr(
        a4_control,
        "TRAIN_CORPUS_VIDEO_ID_ORDER_SHA256",
        expected_sha256,
    )
    payload = {
        "train": {
            video_id: [float(index + 1), 100_000 + index * 3]
            for index, video_id in enumerate(video_ids)
        }
    }
    observed_ids, durations = a4_control._decode_train_video_meta_only(
        json.dumps(payload, separators=(",", ":")).encode("ascii")
    )
    assert observed_ids == tuple(video_ids)
    assert durations[0] == 1.0
    assert durations[-1] == 17_435.0
    source = inspect.getsource(a4_control._decode_train_video_meta_only)
    assert "ordered = list(train_rows.values())" in source
    assert "sorted(train_rows) == list(range(17_435))" not in source


def test_final_review_scope_covers_inherited_runtime_dependencies() -> None:
    assert len(a4_control.A4_REQUIRED_REVIEWED_FILES) == 54
    assert a4_control.A4_REVIEWED_FILE_COUNT == 54
    assert len(a4_control.A4_PINNED_GIT_REVIEWED_FILES) == 26
    assert len(a4_control.A4_INHERITED_REVIEWED_FILES) == 7
    assert a4_control.A4_INHERITED_REVIEWED_FILES < (
        a4_control.A4_REQUIRED_REVIEWED_FILES
    )
    assert {
        "blueprint_e2e_v2/c28f_v5/__init__.py",
        "blueprint_e2e_v2/c28f_v5/atomic_io.py",
        "blueprint_e2e_v2/c28f_v5/canonical.py",
        "blueprint_e2e_v2/c28f_v5/constants.py",
        "blueprint_e2e_v2/c28f_v5/control_plane.py",
        "tests/c28f_v5/fixtures/metric_fixture_v1.json",
        "tests/c28f_v5/fixtures/state_fixture_v1.json",
    } == a4_control.A4_INHERITED_REVIEWED_FILES
    assert "tests/c28f_v5/A4_PREBOOTSTRAP_AUDIT.json" in (
        a4_control.A4_REQUIRED_REVIEWED_FILES
    )
    assert a4_control.A4_PINNED_GIT_REVIEWED_FILES < (
        a4_control.A4_REQUIRED_REVIEWED_FILES
    )
    assert {
        "blueprint_e2e_v2/__init__.py",
        "blueprint_e2e_v2/data/__init__.py",
        "blueprint_e2e_v2/models/__init__.py",
        "blueprint_e2e_v2/utils/__init__.py",
        "blueprint_e2e_v2/data/temporal_grid.py",
        "blueprint_e2e_v2/models/full_model.py",
        "blueprint_e2e_v2/utils/tensor_ops.py",
    } < a4_control.A4_PINNED_GIT_REVIEWED_FILES


def test_exact_reviewed_json_reader_hashes_before_decode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "review-root"
    root.mkdir()
    relative = "fixture.json"
    encoded = b'{\n  "fixture": "reviewed"\n}\n'
    (root / relative).write_bytes(encoded)
    value, record = a4_control._read_exact_reviewed_json_object(
        root,
        relative,
        expected_sha256=hashlib.sha256(encoded).hexdigest(),
        max_bytes=1024,
    )
    assert value == {"fixture": "reviewed"}
    assert record["sha256"] == hashlib.sha256(encoded).hexdigest()

    (root / relative).write_bytes(
        a4_control._canonical_file_bytes({"fixture": "drifted"})
    )

    def forbidden_decode(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("hash drift reached JSON decode")

    monkeypatch.setattr(a4_control.json, "loads", forbidden_decode)
    with pytest.raises(
        a4_control.A4ControlError,
        match="reviewed JSON hash drift before decode",
    ):
        a4_control._read_exact_reviewed_json_object(
            root,
            relative,
            expected_sha256=hashlib.sha256(encoded).hexdigest(),
            max_bytes=1024,
        )


def test_exact_reviewed_json_reader_rejects_duplicate_keys(
    tmp_path: Path,
) -> None:
    root = tmp_path
    target = root / "duplicate-reviewed-fixture.json"
    encoded = b'{"fixture":1,"fixture":2}\n'
    target.write_bytes(encoded)
    try:
        with pytest.raises(
            a4_control.A4ControlError,
            match="invalid exact reviewed JSON",
        ):
            a4_control._read_exact_reviewed_json_object(
                root,
                target.name,
                expected_sha256=hashlib.sha256(encoded).hexdigest(),
                max_bytes=1024,
            )
    finally:
        target.unlink()


def test_failed_a4_quarantine_evidence_is_exactly_bound() -> None:
    evidence = a4_control._validate_failed_a4_quarantine_evidence()
    assert evidence["event_id"] == "PRE_A4_FAILED_G0_A4_QUARANTINE_001"
    assert evidence["source_control_root_absent"] is True
    assert evidence["control_root_inode"] == 113377864
    assert evidence["migration_method"] == "renameat2(RENAME_NOREPLACE)"
    assert evidence["copy_unlink_fallback"] is False
    assert evidence["scientific_impact"] == "NONE"
    assert evidence["protected_content_opens"] == 0
    assert evidence["train_content_opens"] == 0
    assert evidence["model_forward_evaluations"] == 0
    assert evidence["real_data_optimizer_updates"] == 0


def test_failed_pre_g1_quarantine_evidence_is_exactly_bound() -> None:
    evidence = a4_control._validate_failed_pre_g1_quarantine_evidence()
    assert evidence["event_id"] == "PRE_A4_FAILED_PRE_G1_A4_QUARANTINE_001"
    assert evidence["failure_stage"] == "G1_BUNDLE_BUILD_BEFORE_COMMIT"
    assert evidence["source_control_root_absent"] is True
    assert evidence["control_root_inode"] == 113377939
    assert evidence["migration_method"] == "renameat2(RENAME_NOREPLACE)"
    assert evidence["copy_unlink_fallback"] is False
    assert evidence["scientific_impact"] == "NONE"
    assert evidence["protected_content_opens"] == 0
    assert evidence["train_content_opens"] == 0
    assert evidence["model_forward_evaluations"] == 0
    assert evidence["real_data_optimizer_updates"] == 0


def test_frozen_forbidden_rules_use_canonical_path_and_reject_absolute_alias() -> None:
    rules = [
        {
            "rule_id": spec["rule_id"],
            "canonical_path": spec["absolute_path"],
            "content_sha256": spec["content_sha256"],
        }
        for spec in a4_control.PROTECTED_ID_MAPPING_MANIFEST_SPECS
    ]
    authority = {
        "protected_id_mapping_manifest_contract": {
            "schema_version": (
                "c28f_a4_protected_id_mapping_manifest_contract_v1"
            ),
            "manifests": [
                dict(spec)
                for spec in a4_control.PROTECTED_ID_MAPPING_MANIFEST_SPECS
            ],
            "union_count": a4_control.PROTECTED_ID_MAPPING_UNION_COUNT,
            "union_sha256": a4_control.PROTECTED_ID_MAPPING_UNION_SHA256,
            "selection_rule": (
                "CANONICAL_SORTED_UNIQUE_UNION_CALIB_HOLDOUT_AND_"
                "PSEUDO_OFFICIAL_HOLDOUT"
            ),
        },
        "canonical_path_map": {"forbidden_registry": rules},
    }
    contract = a4_control._protected_id_mapping_manifest_contract(authority)
    assert len(contract["manifests"]) == 2
    assert all(
        item["descriptor_record"]["canonical_path"]
        == item["absolute_path"]
        for item in contract["manifests"]
    )

    forged = copy.deepcopy(authority)
    for rule in forged["canonical_path_map"]["forbidden_registry"]:
        rule["absolute_path"] = rule.pop("canonical_path")
    with pytest.raises(
        a4_control.A4ControlError,
        match="Genesis protected manifest descriptor/content binding",
    ):
        a4_control._protected_id_mapping_manifest_contract(forged)


def test_forbidden_registry_consumers_never_read_absolute_path_key() -> None:
    source = inspect.getsource(a4_control)
    assert 'rule.get("absolute_path")' not in source
    assert 'rule["absolute_path"]' not in source


def test_exact_reviewed_json_reader_rejects_symlink_hardlink_and_oversize(
    tmp_path: Path,
) -> None:
    root = tmp_path / "review-root"
    root.mkdir()
    encoded = a4_control._canonical_file_bytes({"fixture": "reviewed"})
    digest = hashlib.sha256(encoded).hexdigest()

    outside = tmp_path / "outside.json"
    outside.write_bytes(encoded)
    (root / "symlink.json").symlink_to(outside)
    with pytest.raises(a4_control.A4ControlError, match="not regular"):
        a4_control._read_exact_reviewed_json_object(
            root,
            "symlink.json",
            expected_sha256=digest,
            max_bytes=1024,
        )

    hard_source = root / "hard-source.json"
    hard_source.write_bytes(encoded)
    os.link(hard_source, root / "hardlink.json")
    with pytest.raises(a4_control.A4ControlError, match="one hard link"):
        a4_control._read_exact_reviewed_json_object(
            root,
            "hardlink.json",
            expected_sha256=digest,
            max_bytes=1024,
        )

    oversized = a4_control._canonical_file_bytes({"fixture": "x" * 1024})
    (root / "oversize.json").write_bytes(oversized)
    with pytest.raises(a4_control.A4ControlError, match="exceeds read cap"):
        a4_control._read_exact_reviewed_json_object(
            root,
            "oversize.json",
            expected_sha256=hashlib.sha256(oversized).hexdigest(),
            max_bytes=128,
        )


def test_bound_reader_rejects_forbidden_inode_before_content_open(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "review-root"
    root.mkdir()
    target = root / "fixture.json"
    encoded = a4_control._canonical_file_bytes({"fixture": "protected"})
    target.write_bytes(encoded)
    identity = target.stat()
    content_opens: list[str] = []
    original_open = a4_control.os.open

    def audited_open(path: object, flags: int, *args: object, **kwargs: object) -> int:
        if isinstance(path, str) and path.startswith("/proc/self/fd/"):
            content_opens.append(path)
        return original_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(a4_control.os, "open", audited_open)
    with pytest.raises(a4_control.A4ControlError, match="forbidden inode"):
        a4_control._read_exact_reviewed_json_object(
            root,
            "fixture.json",
            expected_sha256=hashlib.sha256(encoded).hexdigest(),
            max_bytes=1024,
            forbidden_identities={(identity.st_dev, identity.st_ino)},
        )
    assert content_opens == []


def test_bound_reader_rename_race_cannot_open_replacement_inode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "review-root"
    root.mkdir()
    target = root / "fixture.json"
    original = a4_control._canonical_file_bytes({"fixture": "reviewed"})
    target.write_bytes(original)
    protected = root / "protected.json"
    protected.write_bytes(
        a4_control._canonical_file_bytes({"fixture": "protected"})
    )
    protected_identity = protected.stat()
    original_open = a4_control.os.open
    path_flag = a4_control._linux_o_path_flag()
    swapped = False
    content_opens: list[str] = []

    def swap_after_metadata_bind(
        path: object,
        flags: int,
        *args: object,
        **kwargs: object,
    ) -> int:
        nonlocal swapped
        fd = original_open(path, flags, *args, **kwargs)
        if path == "fixture.json" and flags & path_flag and not swapped:
            os.replace(protected, target)
            swapped = True
        if isinstance(path, str) and path.startswith("/proc/self/fd/"):
            content_opens.append(path)
        return fd

    monkeypatch.setattr(a4_control.os, "open", swap_after_metadata_bind)
    with pytest.raises(
        a4_control.A4ControlError,
        match="type drift before content open",
    ):
        a4_control._read_exact_reviewed_json_object(
            root,
            "fixture.json",
            expected_sha256=hashlib.sha256(original).hexdigest(),
            max_bytes=1024,
            forbidden_identities={
                (protected_identity.st_dev, protected_identity.st_ino)
            },
        )
    assert swapped is True
    assert content_opens == []


def test_a3_parent_lock_is_exact() -> None:
    assert a4_control.A3_ATTEMPT_ID == "G0-A3-54fdf5f2f77f41c98877023949e2ab8b"
    assert a4_control.A3_STATE_SHA256 == "91c034ebb143bd6ee0e372ecf1791709b08a6eb5528e65b8768e44c0a4d31484"
    assert a4_control.A3_EVENT_SHA256 == "5fdc085d64cd28686f037eaa08452f57a751c7f1c7c53b286c929fb490f9207e"
    assert (
        a4_control.A3_BUSINESS_DELTA_SHA256
        == "e99336e63aba648d2a8121669abddad66426483762a040df89de82a0d019467b"
    )


def test_launcher_has_exact_reviewed_code_change_actions() -> None:
    action = next(
        item for item in runtime_runner.build_parser()._actions if item.dest == "action"
    )
    assert tuple(action.choices) == runtime_runner.RUNTIME_ACTIONS == (
        "F0_A_ANALYZE",
        "G4_ROLE_POLICY_LOCK",
        "RUN_F0_B",
        "F0_B_ANALYZE",
        "G6_VERIFY_F1",
        "G7_FINALIZE",
    )
    lowered = " ".join(action.choices).lower()
    for forbidden in ("all", "holdout", "formal", "train", "f2"):
        assert forbidden not in lowered


def test_launcher_prepare_manifest_order_is_kernel_exact() -> None:
    source = inspect.getsource(a4_runner)
    assert "from blueprint_e2e_v2.c28f_v5.runtime_runner import main" in source
    assert "a4_control" not in source
    assert "read_text" not in source
    assert "open(" not in source


def test_runner_delegates_g2_fixture_reads_to_descriptor_bound_controller() -> None:
    runner_source = inspect.getsource(a4_runner)
    compact_source = inspect.getsource(runtime_runner)
    assert "_canonical_json_object" not in runner_source
    assert "read_text" not in runner_source
    assert "G2_" not in compact_source
    assert "TRAIN_JSONL" not in compact_source


def test_semantic_hash_builder_roundtrip() -> None:
    record = a4_control._with_semantic_sha({"schema": "fixture", "seq": 7}, "record_sha256")
    assert record["record_sha256"] == a4_control.semantic_sha256(
        record,
        excluded_fields=("record_sha256",),
    )
    with pytest.raises(a4_control.A4ControlError):
        a4_control._with_semantic_sha(record, "record_sha256")


def test_append_only_ledger_chain_accepts_exact_chain() -> None:
    first = a4_control._ledger_entry(
        ledger="mutation",
        action="FIXTURE_ZERO",
        attempt_id="fixture-attempt",
        transaction_id="fixture-transaction-0",
        facts={"value": 0},
    )
    second = a4_control._ledger_entry(
        ledger="mutation",
        action="FIXTURE_ONE",
        attempt_id="fixture-attempt",
        transaction_id="fixture-transaction-1",
        facts={"value": 1},
        seq=1,
        previous_sha256=first["entry_sha256"],
    )
    journal = (
        a4_control.canonical_json_bytes(first)
        + b"\n"
        + a4_control.canonical_json_bytes(second)
        + b"\n"
    )
    assert a4_control._validate_semantic_journal(
        journal,
        hash_field="entry_sha256",
        previous_field="previous_entry_sha256",
        expected_kind="mutation",
    ) == [first, second]


def test_append_only_ledger_chain_rejects_reordering() -> None:
    first = a4_control._ledger_entry(
        ledger="access",
        action="FIXTURE_ZERO",
        attempt_id="fixture-attempt",
        transaction_id="fixture-transaction-0",
        facts={"opened": 0},
    )
    second = a4_control._ledger_entry(
        ledger="access",
        action="FIXTURE_ONE",
        attempt_id="fixture-attempt",
        transaction_id="fixture-transaction-1",
        facts={"opened": 0},
        seq=1,
        previous_sha256=first["entry_sha256"],
    )
    journal = (
        a4_control.canonical_json_bytes(second)
        + b"\n"
        + a4_control.canonical_json_bytes(first)
        + b"\n"
    )
    with pytest.raises(a4_control.A4ControlError):
        a4_control._validate_semantic_journal(
            journal,
            hash_field="entry_sha256",
            previous_field="previous_entry_sha256",
            expected_kind="access",
        )


def test_inventory_postimage_is_order_independent() -> None:
    first = {"path": "tests/c28f_v5/a.py", "type": "regular_file", "size_bytes": 1, "sha256": "a" * 64}
    second = {"path": "tests/c28f_v5/b.py", "type": "regular_file", "size_bytes": 2, "sha256": "b" * 64}
    left = a4_control._inventory_from_entries([first, second])
    right = a4_control._inventory_from_entries([second, first])
    assert left == right


def test_inventory_reconstruction_reverses_repeated_online_replacements() -> None:
    path = "tests/c28f_v5/a.py"
    genesis = {
        "path": path,
        "type": "regular_file",
        "size_bytes": 1,
        "sha256": "a" * 64,
    }
    first = {
        "files": [
            {
                "destination": path,
                "operation": "REPLACE_EXISTING",
                "expected_preimage_sha256": "a" * 64,
                "expected_preimage_size_bytes": 1,
                "sha256": "b" * 64,
                "size_bytes": 2,
            }
        ]
    }
    second = {
        "files": [
            {
                "destination": path,
                "operation": "REPLACE_EXISTING",
                "expected_preimage_sha256": "b" * 64,
                "expected_preimage_size_bytes": 2,
                "sha256": "c" * 64,
                "size_bytes": 3,
            }
        ]
    }
    current = a4_control._inventory_from_entries(
        [
            {
                "path": path,
                "type": "regular_file",
                "size_bytes": 3,
                "sha256": "c" * 64,
            }
        ]
    )
    reconstructed = a4_control._reconstruct_bootstrap_inventory(
        current,
        {"first": first, "second": second},
        code_change_order=["first", "second"],
    )
    assert reconstructed == a4_control._inventory_from_entries([genesis])


def test_reviewed_code_change_interfaces_are_explicit() -> None:
    assert callable(a4_control.prepare_reviewed_code_change)
    assert callable(a4_control.commit_reviewed_code_change)
    assert callable(a4_control.commit_user_authorized_online_code_change)
    assert "PREPARE_REVIEWED_CODE_CHANGE" in (a4_control.prepare_reviewed_code_change.__doc__ or "")
    assert "COMMIT_REVIEWED_CODE_CHANGE" in (a4_control.commit_reviewed_code_change.__doc__ or "")
    assert (
        a4_control.ONLINE_CODE_CHANGE_AUTHORIZATION_TEXT
        == "这次直接在线修改既有文件"
    )
    assert (
        a4_control.ONLINE_CODE_CHANGE_REVIEW_POLICY
        == "USER_WILL_REVIEW_AFTER_ALL_WORK_COMPLETES"
    )
    online_source = inspect.getsource(
        a4_control.commit_user_authorized_online_code_change
    )
    assert "_assert_pre_forward_active_eval_online_maintenance" in (
        online_source
    )
    assert '"resume_status": state["status"]' in online_source
    assert "_load_stage_contracts(" in online_source
    assert "STAGE_COMMIT_ID_RE.fullmatch" in online_source
    assert 'status=str(contract["resume_status"])' in online_source
    verifier_source = inspect.getsource(
        a4_control._validate_dynamic_control_chain
    )
    assert 'committed_status = (' in verifier_source
    assert 'state.get("status") == committed_status' in verifier_source


def test_g1_handle_accepts_only_verified_pre_g1_code_control_history() -> None:
    source = inspect.getsource(a4_control.issue_g1_control_handle)
    assert "pre_g1_control_only = all(" in source
    assert "CODE_CHANGE_TXN_RE.fullmatch" in source
    assert 'state.get("transition_seq") == len(events) - 1' in source
    assert 'state.get("stage") == "G0"' in source
    assert 'state.get("next_action") == "G1_FAIL_CLOSED_GUARD"' in source
    assert 'state.get("protected_model_evals") == 0' in source
    assert 'state.get("protected_content_opens_current_attempt") == 0' in source
    assert 'state.get("real_data_optimizer_updates") == 0' in source
    assert 'state.get("transition_seq") == 0' not in source


def test_transaction_ids_require_full_regex_matches() -> None:
    assert a4_control.A4_ATTEMPT_RE.fullmatch("G0-A4-" + "a" * 32)
    assert not a4_control.A4_ATTEMPT_RE.fullmatch("xG0-A4-" + "a" * 32)
    assert a4_control.CODE_CHANGE_TXN_RE.fullmatch("CODE-CHANGE-" + "b" * 32 + "-PREPARE")
    assert a4_control.STAGE_COMMIT_ID_RE.fullmatch("STAGE-COMMIT-" + "c" * 32)
    assert a4_control.EVAL_TOKEN_TXN_RE.fullmatch("EVAL-TOKEN-" + "d" * 32 + "-RUNNING")


def test_supersession_target_rejects_pollution(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    report_root = tmp_path / "reports"
    target_parent = report_root / "superseded" / a4_control.A3_ATTEMPT_ID
    target_parent.mkdir(parents=True)
    (target_parent / "pollution").write_bytes(b"x")
    monkeypatch.setattr(a4_control, "REPORT_ROOT", report_root)
    monkeypatch.setattr(a4_control, "A3_SUPERSEDED_PARENT", target_parent)
    with pytest.raises(a4_control.A4ControlError, match="polluted"):
        a4_control._ensure_empty_supersession_target_parent()


def test_receipt_hardlink_crash_is_finalized_to_one_link(tmp_path: Path) -> None:
    target = tmp_path / "SUPERSESSION_RECEIPT.json"
    target.write_bytes(b"{}\n")
    temp = tmp_path / ".SUPERSESSION_RECEIPT.json.capability.fixture"
    os.link(target, temp)
    assert target.stat().st_nlink == 2
    a4_control._finalize_capability_link(target)
    assert target.stat().st_nlink == 1
    assert not temp.exists()


def test_frozen_a3_supersession_receipt_is_independent_of_current_a4_review() -> None:
    receipt, encoded, record = a4_control._read_bound_canonical_json(
        a4_control.A3_SUPERSEDED_PARENT,
        "SUPERSESSION_RECEIPT.json",
        max_bytes=4 * 1024 * 1024,
    )
    closure = {
        "control_tree_sha256": receipt["control_tree_sha256"],
        "file_count": receipt["control_tree_file_count"],
        "directory_count": receipt["control_tree_directory_count"],
        "root_identity": receipt["preserved_root_identity"],
        "writer_lock_identity": receipt["old_writer_lock_identity"],
    }
    assert len(encoded) > 0
    a4_control._validate_frozen_a3_supersession_receipt(
        receipt,
        receipt_file_sha256=record["sha256"],
        closure=closure,
    )
    source = inspect.getsource(a4_control._validate_supersession_receipt)
    assert "_rebuild_reviewed_drift_exception" not in source
    assert "_build_supersession_receipt" not in source

    tampered = dict(receipt)
    tampered["control_tree_file_count"] += 1
    with pytest.raises(a4_control.A4ControlError):
        a4_control._validate_frozen_a3_supersession_receipt(
            tampered,
            receipt_file_sha256=record["sha256"],
            closure=closure,
        )


def test_failed_a4_quarantine_allows_only_a_different_live_root_inode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    live = tmp_path / "goal_control"
    live.mkdir()
    identity = live.stat()
    monkeypatch.setattr(a4_control, "CONTROL_ROOT", live)
    a4_control._assert_failed_a4_source_inode_absent(
        {"device": identity.st_dev, "inode": identity.st_ino + 1}
    )
    with pytest.raises(a4_control.A4ControlError, match="source inode returned"):
        a4_control._assert_failed_a4_source_inode_absent(
            {"device": identity.st_dev, "inode": identity.st_ino}
        )


def test_a4_intent_keeps_historical_and_current_review_receipts_distinct() -> None:
    source = inspect.getsource(a4_control._build_a4_intent)
    assert "A3_SUPERSESSION_STATIC_REVIEW_RECEIPT_SHA256" in source
    assert '== review["file_sha256"]' not in source
    assert '"sha256": review["file_sha256"]' in source


def test_partial_genesis_rejects_state_pointer_before_closure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    control = tmp_path / "goal_control"
    control.mkdir()
    (control / "GOAL_STATE.json").write_bytes(b"state\n")
    monkeypatch.setattr(a4_control, "CONTROL_ROOT", control)
    monkeypatch.setattr(
        a4_control,
        "_genesis_expected_file_bytes",
        lambda _intent, _prepared: {
            "GOAL_STATE.json": b"state\n",
            "GOAL_EVENTS.jsonl": b"event\n",
        },
    )
    monkeypatch.setattr(
        a4_control,
        "_a4_expected_namespace",
        lambda _intent, _prepared: ({"GOAL_STATE.json", "GOAL_EVENTS.jsonl"}, set()),
    )
    with pytest.raises(a4_control.A4ControlError, match="state exists before complete"):
        a4_control._assert_genesis_partial_namespace({}, {})


def test_pending_recovery_is_full_rebuild_before_state_pointer() -> None:
    source = inspect.getsource(a4_control._recover_pending_state_pointer)
    enveloped = inspect.getsource(a4_control._recover_enveloped_state_pointer)
    public_gate = inspect.getsource(
        a4_control.recover_then_verify_a4_state
    )
    for required in (
        "pending state differs from deterministic rebuild",
        "pending recovery business postimage mismatch",
        "_validate_supersession_receipt",
        'CONTROL_ROOT / "GOAL_STATE.json"',
        "verify_a4_state",
        "current enveloped receipt recovery envelope",
        "_build_stage_commit_receipt(",
        "_build_eval_phase_receipt(",
        "_build_id_mapping_access_receipt(",
        "_build_finalization_phase_receipt(",
    ):
        assert required in source
    for required in (
        "pending envelope event full rebuild",
        "pending envelope business postimage",
        "pending envelope registry sidecar",
        'CONTROL_ROOT / "GOAL_STATE.json"',
    ):
        assert required in enveloped
    assert "_recover_pending_state_pointer(" in public_gate
    assert "verified = verify_a4_state(" in public_gate
    assert public_gate.index("_recover_pending_state_pointer(") < (
        public_gate.index("verified = verify_a4_state(")
    )
    assert "active_lease=lease" in public_gate
    assert 'verified.get("next_action_is_unique") is True' in public_gate


def test_dynamic_verifier_covers_extensions_and_exact_namespace() -> None:
    source = inspect.getsource(a4_control._validate_dynamic_control_chain)
    verify_source = inspect.getsource(a4_control.verify_a4_state)
    assert "dynamic chain lost GENESIS anchor" in source
    assert "CODE_CHANGE_TXN_RE.fullmatch" in source
    assert "STAGE_COMMIT_ID_RE.fullmatch" in source
    assert "EVAL_TOKEN_TXN_RE.fullmatch" in source
    assert "enveloped event full rebuild" in source
    assert "dynamic authoritative file namespace" in verify_source
    assert "dynamic authoritative directory namespace" in verify_source


def test_stage_and_eval_cas_interfaces_are_concrete() -> None:
    assert callable(a4_control.prepare_stage_commit)
    assert callable(a4_control.commit_stage_outputs)
    assert callable(a4_control.reserve_eval_token)
    assert callable(a4_control.mark_eval_running)
    assert callable(a4_control.commit_eval_cursor)
    assert callable(a4_control.complete_eval_token)
    assert callable(a4_control.assert_running_state_event_committed)
    assert callable(
        a4_control.rehydrate_active_controller_forward_authority_handle
    )


def test_active_forward_rehydrate_has_no_train_or_token_reissue_surface() -> None:
    source = inspect.getsource(
        a4_control.rehydrate_active_controller_forward_authority_handle
    )
    assert "rehydrate_active_forward_authority" in source
    assert "active_rehydrate_projection" in source
    assert "reservation_input_capability_sha256" in source
    for forbidden in (
        "_pread_authorized_train_projection",
        "_scan_train_jsonl_descriptor_index_once",
        "issue_forward_query_identity_projection",
        "reserve_eval_token(",
        "mark_eval_running(",
    ):
        assert forbidden not in source


def test_unreserved_forward_authority_sidecar_is_registered_and_recovered() -> None:
    dynamic_source = inspect.getsource(
        a4_control._validate_dynamic_control_chain
    )
    namespace_source = inspect.getsource(
        a4_control._validate_controller_forward_authority_records_namespace
    )
    issue_source = inspect.getsource(
        a4_control.issue_controller_forward_authority_handle
    )
    recovery_source = inspect.getsource(
        a4_control._rehydrate_unreserved_controller_forward_authority_handle
    )
    assert (
        "_validate_controller_forward_authority_records_namespace"
        in dynamic_source
    )
    assert "controller_forward_authority.json" in namespace_source
    assert "duplicate persisted controller authority token" in namespace_source
    assert "_rehydrate_unreserved_controller_forward_authority_handle" in (
        issue_source
    )
    assert issue_source.index(
        "_rehydrate_unreserved_controller_forward_authority_handle"
    ) < issue_source.index("consume_selected_records_once")
    assert "rehydrate_active_forward_authority" in recovery_source
    for forbidden in (
        "read_regular_bytes(VIDEO_META",
        "_pread_authorized_train_projection",
        "_scan_train_jsonl_descriptor_index_once",
        "FrozenFile.capture",
        "reserve_eval_token(",
        "mark_eval_running(",
    ):
        assert forbidden not in recovery_source


def test_running_forward_uses_historical_eval_anchor_across_online_fix() -> None:
    assert_source = inspect.getsource(
        a4_control.assert_fsynced_control_chain
    )
    anchor_source = inspect.getsource(
        a4_control._active_eval_running_anchor
    )
    maintenance_source = inspect.getsource(
        a4_control._assert_pre_forward_active_eval_online_maintenance
    )
    complete_source = inspect.getsource(a4_control.complete_eval_token)
    cursor_source = inspect.getsource(
        a4_control.issue_eval_cursor_capability
    )
    online_source = inspect.getsource(
        a4_control.commit_user_authorized_online_code_change
    )
    assert '"F0_A": "G3_F0_A"' in assert_source
    assert 'stage == state["stage"]' not in assert_source
    assert "running_state[\"state_sha256\"]" in assert_source
    assert "running_event[\"event_sha256\"]" in assert_source
    assert "online code only" in anchor_source
    assert "forward_capability_consumption.json" in maintenance_source
    assert "eval_cursor.json" in maintenance_source
    assert "lmdb_batch_receipts" in maintenance_source
    assert "exact initial run receipt" in maintenance_source
    assert "requires empty encoded cache" in maintenance_source
    assert "requires empty query chunks" in maintenance_source
    assert "requires exact empty cursor zero" in maintenance_source
    assert "permits only video cache LMDB batches" in maintenance_source
    assert "exact pre-forward failure ledger" in maintenance_source
    assert "counters[\"model_forward_count\"] == 0" in maintenance_source
    assert "cursor-zero heartbeat" in maintenance_source
    assert (
        "_validate_persisted_controller_forward_authority_record"
        in maintenance_source
    )
    assert "_assert_pre_forward_active_eval_online_maintenance" in online_source
    running_branch = complete_source.split(
        '"eval token is not RUNNING"',
        maxsplit=1,
    )[1]
    assert running_branch.index("_active_eval_running_anchor") < (
        running_branch.index("_validate_eval_cursor_chain")
    )
    partial_branch = complete_source.split(
        '"partial COMPLETED state preimage"',
        maxsplit=1,
    )[1].split(
        '"eval token is not RUNNING"',
        maxsplit=1,
    )[0]
    assert "running_event['seq']" in partial_branch
    assert "events[-1]" not in partial_branch
    assert partial_branch.index("_active_eval_running_anchor") < (
        partial_branch.index("_validate_eval_cursor_chain")
    )
    assert cursor_source.index("_active_eval_running_anchor") < (
        cursor_source.index("_validate_eval_cursor_chain")
    )


def test_stage_prepare_receipt_does_not_mix_eval_running_fields() -> None:
    prepare_source = inspect.getsource(a4_control.prepare_stage_commit)
    validator_source = inspect.getsource(
        a4_control._validate_stage_commit_intent
    )
    assert '"running_state_sha256"' not in prepare_source
    assert '"running_event_sha256"' not in prepare_source
    assert '"running_state_sha256"' not in validator_source
    assert '"running_event_sha256"' not in validator_source


def test_eval_reservation_receipt_matches_security_exact_field_set() -> None:
    registry = {
        "authority_id": "authority",
        "eval_id": "eval",
        "token_id": "F0_A",
        "run_id": "EVAL-TOKEN-" + "a" * 32,
        "cas_intent_sha256": "b" * 64,
        "authorization_sha256": "c" * 64,
        "budget_preimage_sha256": "d" * 64,
        "budget_postimage_sha256": "e" * 64,
        "reservation_input_capability_sha256": "2" * 64,
        "transaction_id": "EVAL-TOKEN-" + "a" * 32 + "-RESERVE",
        "token_state": "RESERVED",
    }
    state = {"state_sha256": "f" * 64}
    event = {"event_sha256": "1" * 64}
    receipt = a4_control._build_eval_phase_receipt(registry, state, event)
    assert set(receipt) == {
        "schema_version",
        "status",
        "goal_id",
        "authority_id",
        "eval_id",
        "token_id",
        "cas_intent_sha256",
        "authorization_sha256",
        "budget_preimage_sha256",
        "budget_postimage_sha256",
        "reservation_input_capability_sha256",
        "transaction_id",
        "state_sha256",
        "event_sha256",
        "receipt_sha256",
    }


def test_opaque_capabilities_reject_direct_and_duck_construction() -> None:
    with pytest.raises(a4_control.A4ControlError, match="controller-issued"):
        a4_control.A4G1ControlHandle(
            object(),
            {"schema_version": "forged"},
            object(),
        )
    with pytest.raises(a4_control.A4ControlError, match="exact A4G1ControlHandle"):
        a4_control.validate_g1_control_handle({"capability_sha256": "a" * 64})


def test_controller_forward_authority_public_api_has_no_caller_authority_slot() -> None:
    issue_parameters = inspect.signature(
        a4_control.issue_controller_forward_authority_handle
    ).parameters
    reservation_parameters = inspect.signature(
        a4_control.issue_eval_reservation_input_handle
    ).parameters
    assert tuple(issue_parameters) == (
        "bootstrap_lease",
        "selected_query_projection_capability",
    )
    assert tuple(reservation_parameters) == (
        "bootstrap_lease",
        "forward_authority_handle",
    )
    assert "forward_authority" not in reservation_parameters
    with pytest.raises(TypeError, match="forward_authority"):
        a4_control.issue_eval_reservation_input_handle(
            bootstrap_lease=object(),
            forward_authority=object(),
        )


def test_controller_forward_authority_rejects_direct_and_duck_objects() -> None:
    with pytest.raises(a4_control.A4ControlError, match="controller-issued"):
        a4_control.A4ControllerForwardAuthorityHandle(
            object(),
            {"schema_version": "forged"},
            object(),
        )
    with pytest.raises(
        a4_control.A4ControlError,
        match="exact A4ControllerForwardAuthorityHandle",
    ):
        a4_control.validate_controller_forward_authority_handle(
            {"capability_sha256": "a" * 64}
        )
    with pytest.raises(
        a4_control.A4ControlError,
        match="exact controller-built FrozenInputAuthority",
    ):
        a4_control.validate_controller_forward_authority(authority=object())


def test_controller_forward_authority_preopen_scan_never_decodes_query_text() -> None:
    projector_source = inspect.getsource(
        a4_id_projector.extract_authorized_train_projection_line
    )
    forward_branch = projector_source.split(
        'if purpose == "FORWARD_QUERY_IDENTITY":', 1
    )[1].split('elif purpose == "POST_FORWARD_EVAL_STATS":', 1)[0]
    pread_source = inspect.getsource(
        a4_control._pread_authorized_train_projection
    )
    projection_source = inspect.getsource(
        a4_control.issue_forward_query_identity_projection
    )
    factory_source = inspect.getsource(
        a4_control.issue_controller_forward_authority_handle
    )
    assert "_sha256(" in forward_branch
    assert 'raw_values["desc"]' in forward_branch
    assert "_decode_projection_desc" not in forward_branch
    assert 'raw_field_hash_counts["desc"] = 1' in forward_branch
    assert "extract_authorized_train_projection_line" in pread_source
    assert '"FORWARD_QUERY_IDENTITY"' in pread_source
    assert '"desc_raw_json_sha256"' in pread_source
    assert "_pread_authorized_train_projection" in projection_source
    assert "query_text_slice_commitment_sha256" in projection_source
    assert 'record["desc"]' not in projection_source
    assert "consume_selected_records_once" in factory_source
    assert "query_text_slice_commitments" in factory_source
    assert 'record["desc"]' not in factory_source


def test_controller_forward_authority_release_is_post_reservation_one_shot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authority = object()
    handle = a4_control.A4ControllerForwardAuthorityHandle(
        a4_control._OPAQUE_CONTROL_ISSUER_SEAL,
        {
            "stage": "G3_F0_A",
            "token_id": "F0_A",
            "authority_binding_sha256": "a" * 64,
        },
        authority,
    )
    record = {
        "object": authority,
        "handle": handle,
        "handle_capability_sha256": handle.capability_sha256,
        "state": "ISSUED_UNRESERVED",
        "reservation_input_object": None,
        "reservation_input_capability_sha256": None,
    }
    monkeypatch.setattr(
        a4_control,
        "validate_controller_forward_authority_handle",
        lambda value: value.binding(),
    )
    monkeypatch.setattr(
        a4_control,
        "_controller_forward_authority_record",
        lambda value: record if value is authority else None,
    )
    with pytest.raises(a4_control.A4ControlError, match="release order/replay"):
        handle.consume_authority_after_reservation()
    record["state"] = "RESERVATION_ISSUED"
    record["reservation_input_capability_sha256"] = "b" * 64
    assert handle.consume_authority_after_reservation() is authority
    assert record["state"] == "AUTHORITY_RELEASED"
    with pytest.raises(a4_control.A4ControlError, match="release order/replay"):
        handle.consume_authority_after_reservation()


def test_g1_abort_fail_closes_even_when_nested_ledger_cleanup_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed = {
        "instrumentation_complete": True,
        "sys_profile_restored": True,
        "thread_profile_restored": True,
        "regular_content_open_count": 0,
        "write_mutation_event_count": 0,
        "data_loader_call_count": 0,
        "model_forward_call_count": 0,
        "subprocess_call_count": 0,
        "preopened_regular_fd_change_count": 0,
        "new_regular_fd_count": 0,
    }

    class FinishedObserver:
        @staticmethod
        def finish(scope_id: str) -> dict:
            assert scope_id == "fixture-observation-scope"
            return dict(observed)

    controller = a4_control._A4G1LedgerController(
        authority_id="fixture-authority",
        goal_id="fixture-goal",
    )
    scope = a4_control._A4G1ObservationScope(
        scope_id="fixture-observation-scope",
        authority_id="fixture-authority",
        controller_identity=id(controller),
    )
    controller._observer = FinishedObserver()
    controller._active_observation_scope = scope
    controller._observation_scope_issued = True
    ledger = controller.open_mediated_ledger(
        scope_id="fixture-ledger-scope",
        authority_id="fixture-authority",
    )

    def fail_cleanup(*_args, **_kwargs) -> None:
        raise RuntimeError("synthetic nested-ledger cleanup failure")

    monkeypatch.setattr(a4_control.EvidenceLedger, "close", fail_cleanup)
    receipt = controller.abort_observation_scope(
        scope,
        reason_code="SYNTHETIC_ABORT",
    )
    assert receipt["status"] == "ABORTED_FAIL_CLOSED"
    assert receipt["nested_ledgers_were_open"] is True
    assert receipt["nested_ledger_cleanup_failed"] is True
    assert receipt["nested_ledger_cleanup_errors"] == [
        "EvidenceLedger:RuntimeError"
    ]
    assert receipt["all_nested_ledgers_closed"] is True
    assert receipt["receipt_sha256"] == a4_control.semantic_sha256(
        receipt,
        excluded_fields=("receipt_sha256",),
    )
    assert controller.abort_observation_scope(
        scope,
        reason_code="IGNORED_REPEAT_REASON",
    ) == receipt
    controller.assert_all_ledgers_closed()


def test_g1_observer_classifies_linux_o_path_without_python_symbol(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observer = a4_control._A4G1ProcessObserver()
    observer._active_scope = "synthetic-o-path-scope"
    monkeypatch.delattr(a4_control.os, "O_PATH", raising=False)
    observer._audit(
        "open",
        (
            "/synthetic/nonexistent-o-path-target",
            None,
            0o10000000 | a4_control.os.O_NOFOLLOW,
        ),
    )
    assert observer._events == [
        {
            "event": "open",
            "path": "/synthetic/nonexistent-o-path-target",
            "write_requested": False,
            "metadata_only": True,
            "allowed_proc_metadata": False,
            "regular_content_open": False,
            "thread_id": a4_control.threading.get_ident(),
        }
    ]


def test_g1_observer_counts_proc_fd_content_reopen_as_regular_content() -> None:
    observer = a4_control._A4G1ProcessObserver()
    observer._active_scope = "synthetic-proc-fd-scope"
    observer._audit(
        "open",
        ("/proc/self/fd/17", None, a4_control.os.O_RDONLY),
    )
    observer._audit(
        "open",
        ("/proc/self/fdinfo/17", None, a4_control.os.O_RDONLY),
    )
    assert observer._events == [
        {
            "event": "open",
            "path": os.path.realpath("/proc/self/fd/17"),
            "write_requested": False,
            "metadata_only": False,
            "allowed_proc_metadata": False,
            "regular_content_open": True,
            "thread_id": a4_control.threading.get_ident(),
        },
        {
            "event": "open",
            "path": "/proc/self/fdinfo/17",
            "write_requested": False,
            "metadata_only": False,
            "allowed_proc_metadata": True,
            "regular_content_open": False,
            "thread_id": a4_control.threading.get_ident(),
        },
    ]


def test_g1_observer_resolves_live_proc_fd_before_descriptor_close(
    tmp_path: Path,
) -> None:
    target = tmp_path / "bound-source.json"
    target.write_bytes(b"{}\n")
    fd = os.open(target, os.O_RDONLY)
    try:
        observer = a4_control._A4G1ProcessObserver()
        observer._active_scope = "synthetic-live-proc-fd-scope"
        observer._audit(
            "open",
            (f"/proc/self/fd/{fd}", None, a4_control.os.O_RDONLY),
        )
    finally:
        os.close(fd)
    assert observer._events[0]["path"] == str(target.resolve())
    assert observer._events[0]["regular_content_open"] is True


def test_g1_ledger_controller_retains_closed_objects_against_id_reuse() -> None:
    class ZeroObserver:
        @staticmethod
        def current_counts(scope_id: str) -> dict[str, int]:
            assert scope_id == "fixture-observation-scope"
            return {
                "regular_content_open_count": 0,
                "write_mutation_event_count": 0,
                "data_loader_call_count": 0,
                "model_forward_call_count": 0,
                "subprocess_call_count": 0,
            }

    controller = a4_control._A4G1LedgerController(
        authority_id="fixture-authority",
        goal_id="fixture-goal",
    )
    controller._observer = ZeroObserver()
    controller._active_observation_scope = a4_control._A4G1ObservationScope(
        scope_id="fixture-observation-scope",
        authority_id="fixture-authority",
        controller_identity=id(controller),
    )
    for index in range(128):
        ledger = controller.open_mediated_ledger(
            scope_id=f"fixture-ledger-{index}",
            authority_id="fixture-authority",
        )
        receipt = controller.close_mediated_ledger(ledger)
        assert receipt["instrumentation_complete"] is True
    assert not controller._open_ledgers
    assert len(controller._closed_ledgers) == 128


def test_g2_replay_decodes_inclusive_clip_indices_to_seconds() -> None:
    source = inspect.getsource(a4_control._g2_open_aligned_streams_once)
    assert a4_control.G2_REPLAY_CLIP_LENGTH_SEC == 1.5
    assert "0 <= start_index <= end_index" in source
    assert "float(end_index + 1) * G2_REPLAY_CLIP_LENGTH_SEC" in source
    assert "float(gt_end_indices[query_id] + 1)" in source
    assert "float(end_index)" not in source
    assert source.index('"x".encode("cp437")') < source.index(
        "_A4_G1_PROCESS_OBSERVER.start(scope_id)"
    )


def test_g2_ground_truth_video_uses_relevant_group_not_first_candidate() -> None:
    import numpy as np

    video_group_keys = np.asarray(
        [[0, 90], [0, 11], [1, 77], [1, 22]],
        dtype=np.int64,
    )
    label_relevant = np.asarray([0.0, 1.0, 0.0, 1.0], dtype=np.float32)
    assert a4_control._g2_gt_video_indices_from_group_labels(
        video_group_keys,
        label_relevant,
        query_count=2,
    ) == (11, 22)
    assert a4_control._g2_gt_video_indices_from_group_labels(
        video_group_keys[:2],
        label_relevant[:2],
        query_count=2,
    ) == (11, -1)
    source = inspect.getsource(a4_control._g2_open_aligned_streams_once)
    assert 'archive["video_group_keys"]' in source
    assert 'archive["label_relevant"]' in source
    assert 'archive["video_idx"][first_rows]' not in source
    assert 'else "missing-ground-truth-video"' in source


def test_g2_observed_private_calls_use_prevalidated_runtime_payload() -> None:
    register_source = inspect.getsource(
        a4_control._register_g2_replay_runtime
    )
    runtime_source = inspect.getsource(a4_control._g2_runtime_record)
    legacy_source = inspect.getsource(
        a4_control._g2_read_legacy_vcmr_metrics_once
    )
    close_source = inspect.getsource(a4_control._g2_close_replay_once)
    assert "payload = validate_g2_replay_control_handle(handle)" in (
        register_source
    )
    assert "_validate_opaque_capability(" in runtime_source
    assert "validate_g2_replay_control_handle(" not in runtime_source
    assert "validate_g2_replay_control_handle(" not in legacy_source
    assert "validate_g2_replay_control_handle(" not in close_source
    assert 'payload = record["payload"]' in legacy_source
    assert 'payload = record["payload"]' in close_source


def test_g2_failed_replay_sessions_have_persisted_abort_recovery() -> None:
    authority = {
        "capability_sha256": "a" * 64,
        "payload": {
            "session_id": "G2-REPLAY-" + "b" * 32,
            "committed_state_sha256": "c" * 64,
            "committed_event_sha256": "d" * 64,
        },
    }
    before_open = a4_control._build_g2_replay_abort_receipt(
        authority,
        preopen_committed=False,
    )
    after_open = a4_control._build_g2_replay_abort_receipt(
        authority,
        preopen_committed=True,
    )
    assert before_open["phase"] == "AUTHORITY_ISSUED_BEFORE_CONTENT_OPEN"
    assert before_open["content_open_attempted"] is False
    assert after_open["phase"] == "PREOPEN_COMMITTED_CONTENT_OPEN_ABORTED"
    assert after_open["content_open_attempted"] is True
    assert before_open["protected_content"] is False
    assert after_open["protected_content"] is False
    for receipt in (before_open, after_open):
        assert receipt["receipt_sha256"] == a4_control.semantic_sha256(
            receipt,
            excluded_fields=("receipt_sha256",),
        )
    recovery_source = inspect.getsource(
        a4_control._recover_incomplete_g2_replay_sessions
    )
    public_recovery_source = inspect.getsource(
        a4_control.recover_then_verify_a4_state
    )
    issue_source = inspect.getsource(
        a4_control.issue_g2_replay_control_handle
    )
    assert "abort_receipt.json" in recovery_source
    assert "atomic_write_immutable(" in recovery_source
    assert "_recover_incomplete_g2_replay_sessions(lease)" in public_recovery_source
    assert '"replay_generation": replay_generation' in issue_source
    failure = a4_control._build_g2_replay_failure_observation(
        authority["payload"],
        {
            "instrumentation_complete": True,
            "sys_profile_restored": True,
            "thread_profile_restored": True,
            "regular_content_open_count": 5,
            "write_mutation_event_count": 0,
            "data_loader_call_count": 0,
            "model_forward_call_count": 0,
            "subprocess_call_count": 0,
            "preopened_regular_fd_change_count": 0,
            "new_regular_fd_count": 0,
        },
        content_paths=["/observed"],
        expected_paths=["/expected"],
    )
    assert failure["protected_content"] is False
    assert not all(failure["closure_checks"].values())
    assert failure["observation_sha256"] == a4_control.semantic_sha256(
        failure,
        excluded_fields=("observation_sha256",),
    )
    close_source = inspect.getsource(a4_control._g2_close_replay_once)
    assert "failure_observation.json" in close_source
    assert "if not closure_ok:" in close_source
    legacy_metrics = {
        f"{threshold:.1f}-r{k}": 50.0
        for threshold in (0.5, 0.7)
        for k in (1, 5, 10, 100)
    }
    current_metrics = {
        value: 0.4
        for value in a4_metrics.C7_LEGACY_VCMR_PARITY_FIELD_MAP.values()
    }
    close_receipt = {
        "receipt_sha256": "e" * 64,
        "replay_packet_sha256": "f" * 64,
        "legacy_metrics_sha256": a4_control.bytes_sha256(
            a4_control.canonical_json_bytes(legacy_metrics)
        ),
        "current_metrics_sha256": a4_control.bytes_sha256(
            a4_control.canonical_json_bytes(current_metrics)
        ),
    }
    parity_failure = a4_control._build_g2_parity_failure_observation(
        authority["payload"],
        control_handle_capability_sha256=authority[
            "capability_sha256"
        ],
        close_receipt=close_receipt,
        legacy_metrics=legacy_metrics,
        current_metrics=current_metrics,
    )
    assert parity_failure["protected_content"] is False
    assert parity_failure["parity"]["status"] == "FAIL"
    assert parity_failure["observation_sha256"] == (
        a4_control.semantic_sha256(
            parity_failure,
            excluded_fields=("observation_sha256",),
        )
    )
    parity_source = inspect.getsource(
        a4_control._g2_commit_parity_derivation_once
    )
    session_validator_source = inspect.getsource(
        a4_control._validate_g2_replay_sessions
    )
    assert "parity_failure_observation.json" in parity_source
    assert "parity_failure_observation.json" in session_validator_source


def test_opaque_registry_consumption_is_exact_object_and_one_shot() -> None:
    payload = {"schema_version": "test-only-opaque-registry", "value": 7}
    capability = a4_control._register_opaque_capability(
        a4_control.A4EvalCursorCapability(
            a4_control._OPAQUE_CONTROL_ISSUER_SEAL,
            payload,
        )
    )
    assert a4_control._validate_opaque_capability(
        capability,
        a4_control.A4EvalCursorCapability,
    ) == payload
    a4_control._consume_registered_capability_once(
        capability,
        operation="TEST_ONLY_CURSOR_CONSUME",
    )
    with pytest.raises(a4_control.A4ControlError, match="already consumed"):
        a4_control._consume_registered_capability_once(
            capability,
            operation="TEST_ONLY_CURSOR_REPLAY",
        )
    with pytest.raises(a4_control.A4ControlError, match="active controller issuance"):
        a4_control._validate_opaque_capability(
            capability,
            a4_control.A4EvalCursorCapability,
        )


def test_completed_eval_retrieval_row_validator_accepts_exact_evidence() -> None:
    rows = [_completed_eval_retrieval_row()]
    digest = a4_control.bytes_sha256(a4_control.canonical_json_bytes(rows))
    a4_control._validate_completed_eval_retrieval_rows(
        rows,
        expected_count=1,
        expected_sha256=digest,
    )


def test_completed_eval_retrieval_row_validator_rejects_semantic_drift() -> None:
    base = _completed_eval_retrieval_row()

    def late_token_drift(row):
        row["late_maxsim_token_count"] += 1

    def mask_denominator_drift(row):
        row["late_candidate_mask_coverage"]["joint"]["denominator"] -= 1

    def duplicate_ranked_video(row):
        row["pooled_video_order"][1] = row["pooled_video_order"][0]

    def missing_present_gt_margin(row):
        row["late_gt_vs_best_wrong_score_margin"] = None

    def nonmonotonic_hit(row):
        row["vcmr_hits"]["VCMR_R@5_IoU@0.5"] = True

    def disjoint_error_semantics_drift(row):
        row["correct_video_wrong_span_top1"] = False

    for mutation in (
        late_token_drift,
        mask_denominator_drift,
        duplicate_ranked_video,
        missing_present_gt_margin,
        nonmonotonic_hit,
        disjoint_error_semantics_drift,
    ):
        rows = [copy.deepcopy(base)]
        mutation(rows[0])
        digest = a4_control.bytes_sha256(a4_control.canonical_json_bytes(rows))
        with pytest.raises(a4_control.A4ControlError):
            a4_control._validate_completed_eval_retrieval_rows(
                rows,
                expected_count=1,
                expected_sha256=digest,
            )
    rows = [copy.deepcopy(base)]
    with pytest.raises(a4_control.A4ControlError, match="count/hash"):
        a4_control._validate_completed_eval_retrieval_rows(
            rows,
            expected_count=1,
            expected_sha256="0" * 64,
        )


def test_completed_eval_query_text_features_follow_the_frozen_rule() -> None:
    assert a4_control._completed_eval_query_text_features(
        "First, pick it up; then RUN before lunch."
    ) == (8, "MULTI_VERB", True)
    assert a4_control._completed_eval_query_text_features(
        "A quiet blue room"
    ) == (4, "ZERO_VERB", False)


def test_completed_eval_ranking_projection_uses_exact_fixed_cuts() -> None:
    ranking = [
        {"video_id": f"video-{index:03d}", "score": 200.0 - index}
        for index in range(200)
    ]
    order, margins = a4_control._completed_eval_ranking_projection(
        ranking,
        gt_video_id="video-005",
    )
    assert order == [item["video_id"] for item in ranking]
    assert margins == {
        "top1_top2": 1.0,
        "top10_top11": 1.0,
        "top100_top101": 1.0,
        "gt_vs_best_wrong": -5.0,
    }
    with pytest.raises(a4_control.A4ControlError, match="score-desc"):
        drift = copy.deepcopy(ranking)
        drift[1]["score"] = 300.0
        a4_control._completed_eval_ranking_projection(
            drift,
            gt_video_id="video-005",
        )


def test_completed_eval_joint_stats_preserve_raw_and_legacy_semantics() -> None:
    raw = [
        {
            "video_id": "gt",
            "start_sec": 1.0,
            "end_sec": 3.0,
            "probability": 0.6,
        },
        {
            "video_id": "wrong",
            "start_sec": 0.0,
            "end_sec": 1.0,
            "probability": 0.4,
        },
    ]
    nms = {
        "predictions": [
            {"video_id": "wrong", "start_sec": 0.0, "end_sec": 1.0},
            {"video_id": "gt", "start_sec": 1.0, "end_sec": 3.0},
        ],
        "raw_exact_duplicate_proposal_count": 0,
        "prediction_count": 2,
    }
    stats = a4_control._completed_eval_joint_stats(
        raw,
        nms,
        gt_video_id="gt",
        gt_span=(1.0, 3.0),
        final_video_order=("gt", "wrong"),
    )
    assert stats["joint_marginal_gt_rank"] == 1
    assert stats["legacy_joint_unique_video_gt_rank"] == 2
    assert stats["wrong_video_top1"] is True
    assert stats["joint_false_positive_mass"] == 0.4
    assert stats["vcmr_hits"]["VCMR_R@1_IoU@0.5"] is False
    assert stats["vcmr_hits"]["VCMR_R@5_IoU@0.7"] is True


def test_completed_eval_finalizer_is_fixed_prefix_bounded_and_gt_mediated() -> None:
    finalizer = inspect.getsource(
        a4_control.issue_completed_eval_artifact_capability
    )
    verifier = inspect.getsource(
        a4_control._validate_completed_eval_derivation_artifacts
    )
    two_phase_indexer = inspect.getsource(
        a4_control._completed_eval_two_phase_indexes
    )
    for fixed_name in (
        "gt_access_receipt.json",
        "cursor_artifact_index.json",
        "raw_forward_artifact_index.json",
        "derived_nms_artifact_index.json",
        "joint_derivation_receipt.json",
    ):
        assert fixed_name in finalizer
        assert fixed_name in verifier
    for required in (
        "_issue_f0_post_forward_role_authority",
        "_pread_authorized_train_projection",
        "_completed_eval_retrieval_rows_from_persisted",
        '"raw_proposals_disclosed": False',
        '"gt_spans_disclosed": False',
    ):
        assert required in finalizer
    assert '"maximum_simultaneously_loaded_chunk_count": 1' in (
        two_phase_indexer
    )
    dynamic = inspect.getsource(a4_control._validate_dynamic_control_chain)
    assert "_validate_completed_eval_derivation_artifacts(events)" in dynamic


def test_pre_a4_exception_constraints_are_scoped_and_nonzero() -> None:
    assert a4_control.REQUIRED_REVIEW_CONSTRAINTS[
        "pre_a4_historical_content_open_status"
    ] == "NONZERO_UNQUANTIFIED_RECORDED_EXCEPTION"
    assert "historical_prediction_or_result_content_opened" not in (
        a4_control.REQUIRED_REVIEW_CONSTRAINTS
    )
    assert a4_control.PRE_A4_LINEAGE_EXCLUDED_PREFIXES == (
        "blueprint_e2e_v2/reports/c28c_",
        "blueprint_e2e_v2/reports/c28e_3_training/",
    )


def test_forward_runtime_domains_are_exactly_excluded_from_business_lineage() -> None:
    attempt = "G0-A4-" + "a" * 32
    run_id = "EVAL-TOKEN-" + "b" * 32
    report_prefix = "blueprint_e2e_v2/reports/c28f_v5"
    run_root = (
        f"{report_prefix}/{attempt}/stage_outputs/G3_F0_A/"
        f"{attempt}/{run_id}"
    )
    assert a4_control._excluded_from_a4_lineage(run_root)
    assert a4_control._excluded_from_a4_lineage(
        run_root + "/query_chunks/queries_00000_00256.pt"
    )
    assert a4_control._excluded_from_a4_lineage(
        f"c28f_v5_cache/{attempt}/encoded_corpus/" + "c" * 64
    )
    assert not a4_control._excluded_from_a4_lineage(
        f"{report_prefix}/{attempt}/stage_outputs/G3_F0_A/"
        f"G0-A4-{'d' * 32}/{run_id}/forward_run_receipt.json"
    )
    assert not a4_control._excluded_from_a4_lineage(
        f"{report_prefix}/{attempt}/stage_outputs/G4/{attempt}/"
        f"{run_id}/forward_run_receipt.json"
    )
    assert not a4_control._excluded_from_a4_lineage(
        f"c28f_v5_cache/{attempt}/encoded_corpus/not-a-hash/cache_receipt.json"
    )


def test_g6_v2_is_controller_owned_pathless_and_single_lease() -> None:
    assert tuple(
        inspect.signature(
            a4_control.execute_g6_synthetic_crash_scenario
        ).parameters
    ) == ("root_handle", "crash_point", "bootstrap_lease")
    session_source = inspect.getsource(
        a4_control._A4G6ControllerOwnedSyntheticSession
    )
    executor_source = inspect.getsource(
        a4_control.execute_g6_synthetic_crash_scenario
    )
    capability_source = inspect.getsource(
        a4_control.A4G6ControllerExecutionCapability
    )
    for required in (
        "dir_fd=self._directory_fd",
        "self._writer_lease.assert_active_capability()",
        "BODY_SEALED_PENDING_CONTROLLER_REREAD",
        "G6 fixed body operation order drift",
        "G6 fixed body operation plan incomplete",
    ):
        assert required in session_source
    assert "Path(" not in session_source
    assert "consume_preacquired_child_execution_once" in capability_source
    assert "return self._session" in capability_source
    assert executor_source.count("with WriterLease(") == 1
    assert "fixed_body(capability)" in executor_source
    assert "controller_session._seal_transcript()" in executor_source
    assert "_g6_controller_reread_actual_postimage(" in executor_source
    assert "_commit_g6_v2_execution_records(" in executor_source
    assert (
        executor_source.index("_commit_g6_v2_execution_records(")
        < executor_source.index(
            "controller_session._close_after_controller_commit()"
        )
    )


def test_g6_v2_intent_precedes_child_creation_and_recovery_is_three_way() -> None:
    source = inspect.getsource(
        a4_control.execute_g6_synthetic_crash_scenario
    )
    intent_write = source.index(
        "atomic_write_immutable(\n                    intent_path"
    )
    child_open_after_intent = source.index(
        "_g6_open_controller_child(",
        intent_write,
    )
    assert intent_write < child_open_after_intent
    for required in (
        '"NEW_INTENT_FIXED_BODY"',
        '"EMPTY_RERUN_FIXED_BODY"',
        '"TERMINAL_POSTIMAGE_REHYDRATE"',
        'observed == {} or observed == expected',
        '"G6 v2 partial/nonterminal postimage drift"',
        "_persist_g6_executor_takeover_owner(",
        "_validate_g6_v2_completion_record(",
    ):
        assert required in source


def test_g6_v2_fixed_operation_contracts_are_content_addressed() -> None:
    for crash_point in (
        "BEFORE_CHECKPOINT_COMMIT",
        "AFTER_CHECKPOINT_BEFORE_MANIFEST",
        "DURING_EVAL",
        "BEFORE_STATUS_RENAME",
    ):
        plan, contract = a4_control._g6_fixed_operation_plan(crash_point)
        assert contract["crash_point"] == crash_point
        assert contract["executor_version"] == a4_control.G6_EXECUTOR_VERSION
        assert contract["operation_count"] == len(plan)
        assert contract["operation_plan_sha256"] == a4_control.bytes_sha256(
            a4_control.canonical_json_bytes(contract["operation_plan"])
        )
        assert contract["contract_sha256"] == a4_control.semantic_sha256(
            contract,
            excluded_fields=("contract_sha256",),
        )
        assert plan[-1]["operation"] == "READ_REGULAR"
        assert any(item["operation"] == "LIST_NAMES" for item in plan)
        assert set(
            item["name"]
            for item in contract["expected_artifact_manifest"]
        ) == set(a4_control.G6_FIXED_TERMINAL_NAMES[crash_point])


def test_g6_v1_external_delivery_surfaces_are_hard_disabled() -> None:
    for function in (
        a4_control.issue_g6_synthetic_unstarted_resume_capability,
        a4_control._consume_g6_synthetic_unstarted_resume_once,
        a4_control._consume_g6_synthetic_child_root_once,
        a4_control.issue_g6_synthetic_postimage_rehydrate_capability,
        a4_control._consume_g6_verified_postimage_rehydrate_once,
        a4_control.commit_g6_synthetic_crash_evidence,
    ):
        source = inspect.getsource(function)
        assert "CONTROLLER_FIXED_EXECUTOR_V2_REQUIRED" in source
        assert "raise A4ControlError" in source
        assert "return Path" not in source


def test_runner_dispatches_only_the_current_reviewed_action() -> None:
    dispatch = inspect.getsource(runtime_runner.dispatch)
    for required in (
        "run_f0_a_analyze",
        "run_g4_role_policy_lock",
        "run_f0_b(bootstrap_lease)",
        "run_f0_b_analyze",
        "run_g6_verify_f1",
        "run_g7_finalize",
    ):
        assert required in dispatch
    for forbidden in (
        "tvr_val_release.jsonl",
        "allow_holdout_final",
        "--stage",
        "subprocess",
        "os.system",
        "F2",
        "RUN_F0_A",
        "ISSUE_F0_A",
    ):
        assert forbidden not in dispatch


def test_runner_forward_keeps_one_stream_and_commits_bounded_chunks() -> None:
    source = inspect.getsource(runtime_stages._run_f0_b_forward)
    assert "stream = engine.iter_forward(" in source
    assert "while receipt[\"status\"] != \"COMPLETED\"" in source
    assert "itertools.islice(stream, QUERY_CHUNK_SIZE)" in source
    assert source.count("engine.iter_forward(") == 1
    assert "cache.materialize_and_load(model_handle)" in source
    assert "validate_completed_forward_run_artifacts(" in source


def test_runner_forward_actions_advance_only_one_control_phase() -> None:
    begin = inspect.getsource(runtime_control.begin_f0_b_running)
    execute = inspect.getsource(runtime_stages.run_f0_b)
    assert 'next_state["status"] = "F0_B_RUNNING"' in begin
    assert 'next_state["next_action"] = RUN_F0_B_ACTION' in begin
    assert "route_manifest_sha256" in begin
    assert "route_desc_ids_sha256" in begin
    assert "begin_f0_b_running(" in execute
    assert execute.index("begin_f0_b_running(") < execute.index("train.project(")
    assert "issue_f0_a_generation(" not in execute
    assert "mark_f0_a_running(" not in execute


def test_runner_g7_releases_an_already_committed_terminal_tail() -> None:
    source = inspect.getsource(runtime_runner)
    assert "G7_FINALIZE_ACTION" in source
    assert "run_g7_finalize" in source
    assert all("F2" not in action for action in runtime_runner.RUNTIME_ACTIONS)
    assert "RUN_F0_A" not in runtime_runner.RUNTIME_ACTIONS


def test_completed_forward_narrow_gap_recovery_is_completed_only() -> None:
    control_source = inspect.getsource(
        a4_control.rehydrate_current_completed_forward_artifact_handle
    )
    material_source = inspect.getsource(
        a4_control.completed_forward_input_rehydrate_material
    )
    running_source = inspect.getsource(
        a4_control.issue_persisted_eval_running_capability_for_completed_validation
    )
    for required in (
        "recover_completed_forward_run_artifact_handle_from_control",
        "issue_persisted_eval_running_capability_for_completed_validation",
        "bind_completed_historical_running_capability",
    ):
        assert required in control_source
    for required in (
        '"active_source_reopened": False',
        '"token_reissued": False',
        '"model_forward_reexecuted": False',
        '"completed_phase_receipt": completed_phase_receipt',
        "validate_completed_forward_input_capability(",
    ):
        assert required in material_source
    assert 'complete_registry.get("token_state") == "COMPLETED"' in running_source
    for forbidden in (
        "issue_forward_query_identity_projection",
        "bootstrap_train_jsonl_descriptor_index",
        "load_frozen_model",
        "open_verified_readonly",
    ):
        assert forbidden not in control_source
        assert forbidden not in material_source
        assert forbidden not in running_source

    forward_source = inspect.getsource(
        a4_forward.recover_completed_forward_run_artifact_handle_from_control
    )
    for required in (
        'material_record.get("completed_phase_receipt")',
        "_validate_completed_forward_phase_receipt_value(",
        'Path(run_root.path) / "control_completion_receipt.json"',
        "replace=False",
    ):
        assert required in forward_source


def test_g4_role_input_public_surface_accepts_no_caller_rows() -> None:
    signature = inspect.signature(a4_control.issue_g4_role_lock_input_handle)
    assert tuple(signature.parameters) == ("bootstrap_lease",)
    source = inspect.getsource(a4_control.issue_g4_role_lock_input_handle)
    assert "issue_g4_projection_completion_capability(" in source
    assert "issue_train_fit_role_source_capability(" in source
    assert "_issue_g4_role_lock_input_handle(" in source


def test_g4_projection_has_three_persisted_crash_resume_windows() -> None:
    issue_source = inspect.getsource(
        a4_control.issue_g4_running_projection_reader
    )
    rehydrate_source = inspect.getsource(
        a4_control._rehydrate_g4_running_projection_reader
    )
    consume_source = inspect.getsource(
        a4_control._consume_g4_projection_source_once
    )
    persist_source = inspect.getsource(
        a4_control._validate_g4_minimal_projection_preimage
    )
    batch_source = inspect.getsource(
        a4_control._issue_g4_minimal_projection_batch_from_preimage
    )
    assert '"ID_MAPPING_RUNNING"' in issue_source
    assert '"ID_MAPPING_ACCESS_COMMITTED"' in issue_source
    assert "_rehydrate_g4_running_projection_reader(" in issue_source
    for required in (
        'f"{operation_id}-RUNNING"',
        'phase in {"RUNNING", "ACCESS"}',
        "_build_id_mapping_phase_receipt(",
        "_issue_g4_running_projection_capability(",
    ):
        assert required in rehydrate_source
    for required in (
        'registry.get("phase") == "ACCESS"',
        '"minimal_projection_batch_preimage.json"',
        '"minimal_projection_preimage_sha256"',
        "_issue_g4_minimal_projection_batch_from_preimage(",
        '"PERSIST_G4_MINIMAL_PROJECTION_PREIMAGE"',
        "_commit_g4_projection_access(",
    ):
        assert required in consume_source
    assert consume_source.index(
        '"PERSIST_G4_MINIMAL_PROJECTION_PREIMAGE"'
    ) < consume_source.index("_commit_g4_projection_access(")
    assert '"protected_content_reopen_count_on_recovery"' in persist_source
    assert '== 0' in persist_source
    assert "_consume_registered_capability_once(" in batch_source
    for forbidden in (
        "_pread_authorized_train_projection(",
        "read_regular_bytes(",
        "os.pread(",
        "TRAIN_JSONL",
    ):
        assert forbidden not in batch_source
