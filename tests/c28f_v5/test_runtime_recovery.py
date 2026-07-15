from __future__ import annotations

import json
import inspect
import shutil
import uuid
from pathlib import Path
from typing import Any, Callable

import pytest

from blueprint_e2e_v2.c28f_v5 import (
    a4_control,
    a4_f1,
    runtime_control,
    runtime_data,
    runtime_runner,
    runtime_stages,
)
from blueprint_e2e_v2.c28f_v5.canonical import canonical_json_bytes
from blueprint_e2e_v2.c28f_v5.constants import GOAL_ID, REPORT_ROOT
from blueprint_e2e_v2.c28f_v5.runtime_store import RuntimeStore


@pytest.fixture
def runtime_root() -> Path:
    root = REPORT_ROOT / "test_runtime_recovery" / uuid.uuid4().hex
    root.mkdir(parents=True, exist_ok=False)
    try:
        yield root
    finally:
        shutil.rmtree(root)


def _token(*, successful: int, current: dict[str, Any] | None) -> dict[str, Any]:
    return {
        "max_successful_evaluations": 1,
        "successful_evaluations": successful,
        "next_generation": 2 if current is not None else 1,
        "current": current,
        "history": [],
    }


def _state_values(
    root: Path,
    *,
    status: str,
    next_action: str,
) -> dict[str, Any]:
    f0_a = {
        "generation": 4,
        "status": "COMPLETED",
        "eval_id": "F0A-EVAL-fixture",
        "run_id": "F0A-RUNTIME-fixture",
    }
    return {
        "attempt_id": "runtime-recovery-fixture",
        "status": status,
        "next_action": next_action,
        "evals": {
            "F0_A": _token(successful=1, current=f0_a),
            "F0_B": _token(successful=0, current=None),
        },
        "safety": {
            "model_forward_evaluations": 1,
            "protected_content_opens": 0,
            "optimizer_updates": 0,
            "teacher_forward_count": 0,
            "gt_support_count": 0,
            "aborted_zero_forward_generations": 3,
        },
        "legacy_snapshot": {},
        "workflow_control": {
            "schema_version": "c28f_v5_runtime_workflow_control_v1",
            "authoritative_state_root": "goal_runtime",
            "legacy_state_disposition": "READ_ONLY_SUPERSEDED_EVIDENCE",
            "supersession_bridge": {
                "path": str(root / "control_bridge" / "supersession_bridge.json"),
                "bridge_sha256": "a" * 64,
            },
            "gate_status": {
                "G1": "COMPLETED_REPROJECTED",
                "G2": "COMPLETED_REPROJECTED",
                "TRAIN_JSONL_INDEX": "COMPLETED_NONREPLAYABLE_REPROJECTED",
            },
            "formal_data_policy": "STRICT_CORE_ONLY",
            "u_formal": 17_360,
            "next_action": next_action,
            "automatic_execution": False,
            "f0_b_or_f2_entry_allowed": status == "ROLE_POLICY_LOCKED",
        },
        "stage_receipts": {},
    }


def _initialize(
    root: Path,
    *,
    status: str,
    next_action: str,
) -> RuntimeStore:
    store = RuntimeStore(root)
    store.initialize(
        _state_values(root, status=status, next_action=next_action),
        action="TEST_RUNTIME_GENESIS",
        artifacts={"fixture": "a" * 64},
    )
    return store


def _stage_mutation(
    status: str,
    next_action: str | None,
    *,
    f0_b_or_f2_entry_allowed: bool | None = None,
) -> Callable[[dict[str, Any]], None]:
    def mutate(state: dict[str, Any]) -> None:
        state["status"] = status
        state["next_action"] = next_action
        if f0_b_or_f2_entry_allowed is not None:
            state["workflow_control"]["f0_b_or_f2_entry_allowed"] = (
                f0_b_or_f2_entry_allowed
            )

    return mutate


def _dummy_output(stage_id: str) -> dict[str, bytes]:
    return {
        f"{stage_id}.json": canonical_json_bytes(
            {"schema_version": "runtime_no_data_fixture_v1", "stage_id": stage_id}
        )
        + b"\n"
    }


def test_runner_exposes_exact_recovery_actions_only() -> None:
    assert runtime_runner.RUNTIME_ACTIONS == (
        "F0_A_ANALYZE",
        "G4_ROLE_POLICY_LOCK",
        "RUN_F0_B",
        "F0_B_ANALYZE",
        "G6_VERIFY_F1",
        "G7_FINALIZE",
    )
    source = Path(runtime_runner.__file__).read_text(encoding="utf-8")
    for forbidden in (
        "RUN_F0_A\"",
        "ISSUE_F0_A_GENERATION",
        "MIGRATE_ZERO_FORWARD_F0_A",
        "AUTH_F2_F5_PROTOTYPE",
    ):
        assert forbidden not in source


def test_g4_uses_the_full_frozen_train_fit_universe_without_subtraction() -> None:
    assert runtime_data.TRAIN_FIT_UNIVERSE_COUNT == 69_428
    assert (
        runtime_data.TRAIN_FIT_MANIFEST_FILE_SHA256
        == "24007d3f6f1159f55e68a8ee64c779436a34db4bff601f9d9cd74ee6e102bf0b"
    )
    source = inspect.getsource(runtime_data.read_g4_manifest_ids)
    assert "train_fit_ids" in source
    assert "calib_select" not in source
    assert "60_751" not in source


def test_train_projection_frames_only_the_unterminated_eof_record() -> None:
    payload = b'{"desc_id":62633,"vid_name":"video-eof"}'
    offset = 1_000
    source_size = offset + len(payload)
    assert runtime_data._frame_indexed_jsonl_record(
        payload,
        offset=offset,
        length=len(payload),
        source_size_bytes=source_size,
    ) == payload + b"\n"

    framed = payload + b"\n"
    assert runtime_data._frame_indexed_jsonl_record(
        framed,
        offset=10,
        length=len(framed),
        source_size_bytes=source_size,
    ) == framed

    with pytest.raises(runtime_data.RuntimeControlError, match="framing drift"):
        runtime_data._frame_indexed_jsonl_record(
            payload,
            offset=offset,
            length=len(payload),
            source_size_bytes=source_size + 1,
        )


def test_g6_uses_the_temporal_oracle_contract_key() -> None:
    temporal = a4_f1.temporal_oracles()
    assert temporal["oracles_sha256"]
    assert "contract_sha256" not in temporal
    source = inspect.getsource(runtime_stages.run_g6_verify_f1)
    assert 'temporal["oracles_sha256"]' in source
    assert 'temporal["contract_sha256"]' not in source


def test_f0_b_completed_material_gap_uses_committed_artifacts_only() -> None:
    analyze_source = inspect.getsource(runtime_stages.run_f0_b_analyze)
    recovery_source = inspect.getsource(
        runtime_stages._recover_f0_b_completed_material
    )
    material_source = inspect.getsource(
        runtime_control.completed_forward_input_rehydrate_material
    )
    assert "_recover_f0_b_completed_material" in analyze_source
    assert (
        "recover_completed_forward_run_artifact_handle_from_control"
        in recovery_source
    )
    assert "load_frozen_model" not in recovery_source
    assert "train.project" not in recovery_source
    for required in (
        '"active_source_reopened": False',
        '"token_reissued": False',
        '"model_forward_reexecuted": False',
    ):
        assert required in material_source


def test_prepared_stage_receipt_recovers_same_transaction(
    runtime_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _initialize(
        runtime_root,
        status="F0_A_COMPLETED",
        next_action=runtime_control.F0_A_ANALYZE_ACTION,
    )

    class InjectedCrash(RuntimeError):
        pass

    def crash_before_pointer_install(*_args: Any, **_kwargs: Any) -> None:
        raise InjectedCrash("after durable stage receipt")

    monkeypatch.setattr(store, "_install_transaction", crash_before_pointer_install)
    with pytest.raises(InjectedCrash, match="durable stage receipt"):
        store.commit_stage(
            stage_id="g3",
            action=runtime_control.F0_A_ANALYZE_ACTION,
            outputs=_dummy_output("g3"),
            input_hashes={"input": "b" * 64},
            safety_counts={"model_forward_evaluations": 1},
            mutate=_stage_mutation(
                "F0_A_ANALYZED", runtime_stages.G4_ROLE_POLICY_LOCK_ACTION
            ),
        )
    receipt_path = runtime_root / "artifacts" / "g3" / "STAGE_RECEIPT.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    recovered = RuntimeStore(runtime_root).recover_prepared_stage(
        stage_id="g3",
        action=runtime_control.F0_A_ANALYZE_ACTION,
        expected_outputs=("g3.json",),
        mutate=_stage_mutation(
            "F0_A_ANALYZED", runtime_stages.G4_ROLE_POLICY_LOCK_ACTION
        ),
    )
    assert recovered is not None
    assert recovered["transaction_id"] == receipt["stage_transaction_id"]
    assert recovered["stage_receipts"]["g3"]["receipt_sha256"] == receipt[
        "receipt_sha256"
    ]
    assert recovered["status"] == "F0_A_ANALYZED"


def test_f0_b_running_recovery_reuses_exact_eval_id(
    runtime_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _initialize(
        runtime_root,
        status="ROLE_POLICY_LOCKED",
        next_action=runtime_control.RUN_F0_B_ACTION,
    )
    original_transition = store.transition

    class InjectedCrash(RuntimeError):
        pass

    def crash_on_anchor(action: str, *args: Any, **kwargs: Any) -> dict[str, Any]:
        if action == "ATTACH_F0_B_RUNNING_ANCHOR":
            raise InjectedCrash("before anchor attachment")
        return original_transition(action, *args, **kwargs)

    monkeypatch.setattr(store, "transition", crash_on_anchor)
    kwargs = {
        "route_manifest_sha256": "1" * 64,
        "route_desc_ids_sha256": "2" * 64,
        "route_query_count": 2_048,
        "input_hashes": {"g4": "3" * 64, "train_index": "4" * 64},
    }
    with pytest.raises(InjectedCrash, match="anchor attachment"):
        runtime_control.begin_f0_b_running(**kwargs, store=store)
    interrupted = RuntimeStore(runtime_root).read_state()
    eval_id = interrupted["evals"]["F0_B"]["current"]["eval_id"]
    assert interrupted["evals"]["F0_B"]["current"]["running_anchor"] is None

    recovered_store = RuntimeStore(runtime_root)
    recovered = runtime_control.begin_f0_b_running(**kwargs, store=recovered_store)
    assert recovered["eval_id"] == eval_id
    recovered_state = recovered_store.read_state()
    assert recovered_state["evals"]["F0_B"]["current"]["running_anchor"]
    transition_seq = recovered_state["transition_seq"]
    repeated = runtime_control.begin_f0_b_running(**kwargs, store=recovered_store)
    assert repeated["eval_id"] == eval_id
    assert recovered_store.read_state()["transition_seq"] == transition_seq

    changed = dict(kwargs)
    changed["input_hashes"] = {"g4": "5" * 64, "train_index": "4" * 64}
    with pytest.raises(
        runtime_control.A4ControlError,
        match="committed RUNNING input binding drift",
    ):
        runtime_control.begin_f0_b_running(**changed, store=recovered_store)


def test_no_data_state_path_reaches_exact_terminal(runtime_root: Path) -> None:
    store = _initialize(
        runtime_root,
        status="F0_A_COMPLETED",
        next_action=runtime_control.F0_A_ANALYZE_ACTION,
    )
    store.commit_stage(
        stage_id="g3",
        action=runtime_control.F0_A_ANALYZE_ACTION,
        outputs=_dummy_output("g3"),
        input_hashes={"input": "1" * 64},
        safety_counts={"model_forward_evaluations": 1},
        mutate=_stage_mutation(
            "F0_A_ANALYZED", runtime_stages.G4_ROLE_POLICY_LOCK_ACTION
        ),
    )
    store.commit_stage(
        stage_id="g4",
        action=runtime_stages.G4_ROLE_POLICY_LOCK_ACTION,
        outputs=_dummy_output("g4"),
        input_hashes={"input": "2" * 64},
        safety_counts={"model_forward_evaluations": 0},
        mutate=_stage_mutation(
            "ROLE_POLICY_LOCKED",
            runtime_control.RUN_F0_B_ACTION,
            f0_b_or_f2_entry_allowed=True,
        ),
    )
    issued = runtime_control.begin_f0_b_running(
        route_manifest_sha256="3" * 64,
        route_desc_ids_sha256="4" * 64,
        route_query_count=2_048,
        input_hashes={"g4": "5" * 64},
        store=store,
    )

    def complete_f0_b(state: dict[str, Any]) -> None:
        current = state["evals"]["F0_B"]["current"]
        current["status"] = "COMPLETED"
        current["model_forward_evaluations"] = 1
        state["evals"]["F0_B"]["successful_evaluations"] = 1
        state["safety"]["model_forward_evaluations"] = 2
        state["status"] = "F0_B_COMPLETED"
        state["next_action"] = runtime_control.F0_B_ANALYZE_ACTION

    store.transition(runtime_control.RUN_F0_B_ACTION, complete_f0_b)
    store.commit_stage(
        stage_id="g5",
        action=runtime_control.F0_B_ANALYZE_ACTION,
        outputs=_dummy_output("g5"),
        input_hashes={"input": "6" * 64},
        safety_counts={"model_forward_evaluations": 1},
        mutate=_stage_mutation(
            "F0_F1_ROUTED", runtime_stages.G6_VERIFY_F1_ACTION
        ),
    )
    store.commit_stage(
        stage_id="g6",
        action=runtime_stages.G6_VERIFY_F1_ACTION,
        outputs=_dummy_output("g6"),
        input_hashes={"input": "7" * 64},
        safety_counts={"model_forward_evaluations": 0},
        mutate=_stage_mutation("F1_COMPLETE", runtime_stages.G7_FINALIZE_ACTION),
    )
    store.transition(
        runtime_stages.G7_FINALIZE_ACTION,
        _stage_mutation("PENDING_FINALIZATION", runtime_stages.G7_FINALIZE_ACTION),
    )
    terminal = store.commit_stage(
        stage_id="g7",
        action=runtime_stages.G7_FINALIZE_ACTION,
        outputs=_dummy_output("g7"),
        input_hashes={"input": "8" * 64},
        safety_counts={"model_forward_evaluations": 2},
        mutate=_stage_mutation(
            "F0_F1_WINDOW_COMPLETE_STOPPED",
            None,
            f0_b_or_f2_entry_allowed=False,
        ),
    )
    assert terminal["status"] == "F0_F1_WINDOW_COMPLETE_STOPPED"
    assert terminal["next_action"] is None
    assert terminal["safety"]["model_forward_evaluations"] == 2
    assert terminal["evals"]["F0_A"]["successful_evaluations"] == 1
    assert terminal["evals"]["F0_B"]["successful_evaluations"] == 1
    assert terminal["evals"]["F0_B"]["current"]["eval_id"] == issued["eval_id"]


def test_runtime_metric_nms_and_joint_stats_match_frozen_oracle() -> None:
    metric_record = {
        "query_id": 7,
        "joint_proposals": [
            {
                "video_id": "video-a",
                "start_sec": 0.0,
                "end_sec": 2.0,
                "score": 0.9,
                "probability": 0.4,
            },
            {
                "video_id": "video-a",
                "start_sec": 0.1,
                "end_sec": 2.1,
                "score": 0.8,
                "probability": 0.1,
            },
            {
                "video_id": "video-b",
                "start_sec": 4.0,
                "end_sec": 6.0,
                "score": 0.7,
                "probability": 0.5,
            },
        ],
    }
    expected_nms = a4_control._derive_frozen_nms_projection_rows(
        [metric_record]
    )[0]
    observed_nms = runtime_data._derive_frozen_nms(metric_record)
    assert observed_nms == expected_nms
    expected_stats = a4_control._completed_eval_joint_stats(
        metric_record["joint_proposals"],
        expected_nms,
        gt_video_id="video-a",
        gt_span=(0.0, 2.0),
        final_video_order=("video-a", "video-b"),
    )
    observed_stats = runtime_data._joint_stats(
        metric_record,
        observed_nms,
        gt_video_id="video-a",
        gt_span=(0.0, 2.0),
        final_video_order=("video-a", "video-b"),
    )
    assert observed_stats == expected_stats
    assert observed_stats["joint_false_positive_mass"] == pytest.approx(0.5)
