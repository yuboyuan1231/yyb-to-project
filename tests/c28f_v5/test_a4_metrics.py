import copy
import hashlib
import json
import unittest
from pathlib import Path
from typing import Iterator, Mapping

from blueprint_e2e_v2.c28f_v5.a4_metrics import (
    METRIC_SCHEMA_VERSION,
    MetricContractError,
    additive_merge_no_overwrite,
    compare_vcmr_parity,
    evaluate_fixture,
    evaluate_gates,
    evaluate_joint_vcmr_and_errors,
    evaluate_queries,
    evaluate_true_video_retrieval,
    fixture_expected_projection,
    gate_schema,
    iter_aligned_c7_replay_rows,
    metric_schema,
    temporal_iou,
)


FIXTURE_ROOT = Path(__file__).resolve().parent / "fixtures"


def _load_json(name):
    return json.loads((FIXTURE_ROOT / name).read_text(encoding="utf-8"))


class _ProposalBomb(Mapping):
    """Retrieval-only mapping that fails if proposal/NMS content is consulted."""

    def __init__(self):
        self._values = {
            "query_id": "q-bomb",
            "gt_video_id": "v1",
            "pooled_video_order": ["v1", "v2"],
            "late_video_order": ["v2", "v1"],
            "final_video_order": ["v1", "v2"],
        }

    def __getitem__(self, key):
        if key in ("joint_proposals", "nms", "proposal_scores"):
            raise AssertionError("true VR touched proposal/NMS state")
        return self._values[key]

    def __iter__(self) -> Iterator[str]:
        raise AssertionError("true VR must explicitly request only retrieval fields")

    def __len__(self):
        return len(self._values)


def _one_query(**overrides):
    row = {
        "query_id": "q1",
        "gt_video_id": "a",
        "gt_span_sec": [0.0, 1.0],
        "pooled_video_order": ["a", "z"],
        "late_video_order": ["a", "z"],
        "final_video_order": ["a", "z"],
        "joint_proposals": [
            ["a", 0.0, 1.0, 2.0, 0.6],
            ["z", 0.0, 1.0, 1.0, 0.4],
        ],
    }
    row.update(overrides)
    return row


class MetricFixtureTest(unittest.TestCase):
    def test_hand_fixture_matches_full_expected_projection(self):
        fixture = _load_json("metric_fixture_v1.json")
        expected = _load_json("metric_fixture_v1_expected.json")
        actual = fixture_expected_projection(evaluate_fixture(fixture))
        self.assertEqual(expected, actual)

    def test_true_video_retrieval_cannot_touch_proposal_or_nms(self):
        result = evaluate_true_video_retrieval([_ProposalBomb()])
        self.assertEqual(1.0, result["metrics"]["pooled_true_VR_R@1"])
        self.assertEqual(1.0, result["metrics"]["late_true_VR_R@5"])

    def test_missing_required_field_fails_closed(self):
        fixture = _load_json("metric_fixture_v1.json")
        fixture["queries"][0].pop("final_video_order")
        with self.assertRaisesRegex(MetricContractError, "missing required fields"):
            evaluate_fixture(fixture)

    def test_empty_query_set_fails_closed(self):
        fixture = _load_json("metric_fixture_v1.json")
        fixture["queries"] = []
        with self.assertRaisesRegex(MetricContractError, "non-empty"):
            evaluate_fixture(fixture)

    def test_authorized_query_universe_mismatch_fails_closed(self):
        with self.assertRaisesRegex(MetricContractError, "authorized manifest"):
            evaluate_queries([_one_query()], expected_query_ids=["q1", "q2"])
        result = evaluate_queries([_one_query()], expected_query_ids=["q1"])
        self.assertEqual(
            "EXACT_AUTHORIZED_MANIFEST",
            result["completeness"]["query_universe_authority"],
        )
        with self.assertRaisesRegex(MetricContractError, "exact query manifest"):
            evaluate_queries([_one_query()], require_frozen_capacities=True)

    def test_explicit_empty_rankings_are_misses_not_skipped(self):
        row = _one_query(
            pooled_video_order=[],
            late_video_order=[],
            final_video_order=[],
            joint_proposals=[],
        )
        result = evaluate_queries([row])
        self.assertEqual(0.0, result["retrieval_metrics"]["pooled_true_VR_R@1000"])
        self.assertEqual(0.0, result["joint_retrieval_metrics"]["joint_marginal_VR_R@200"])
        self.assertEqual(0.0, result["vcmr_metrics"]["VCMR_R@100_IoU@0.7"])
        self.assertEqual(1.0, result["error_metrics"]["top1_false_positive_rate"])
        self.assertEqual(1.0, result["error_metrics"]["top1_missing_rate"])
        self.assertIsNone(result["error_metrics"]["joint_false_positive_mass_mean"])
        self.assertEqual(0, result["denominators"]["joint_false_positive_mass_denominator"])
        self.assertIsNone(result["gate_metrics"]["G_keep"])

    def test_late_ranking_must_be_a_broad_subset(self):
        row = _one_query(
            pooled_video_order=["z"],
            late_video_order=["a"],
        )
        with self.assertRaisesRegex(MetricContractError, "subset"):
            evaluate_queries([row])

    def test_final_and_joint_candidates_must_follow_stage_lineage(self):
        with self.assertRaisesRegex(MetricContractError, "final_video_order must be a subset"):
            evaluate_queries(
                [_one_query(late_video_order=["a"], final_video_order=["a", "z"])]
            )
        with self.assertRaisesRegex(MetricContractError, "outside final_video_order"):
            evaluate_queries(
                [
                    _one_query(
                        final_video_order=["a"],
                        joint_proposals=[
                            ["a", 0.0, 1.0, 2.0, 0.6],
                            ["z", 0.0, 1.0, 1.0, 0.4],
                        ],
                    )
                ]
            )

    def test_gt_outside_broad1000_is_a_late_miss(self):
        pooled = ["v%04d" % index for index in range(1000)] + ["a"]
        row = _one_query(
            pooled_video_order=pooled,
            late_video_order=["a"],
            final_video_order=["a"],
            joint_proposals=[["a", 0.0, 1.0, 1.0, 1.0]],
        )
        result = evaluate_queries([row])
        self.assertEqual(0.0, result["retrieval_metrics"]["pooled_true_VR_R@1000"])
        self.assertEqual(0.0, result["retrieval_metrics"]["late_true_VR_R@200"])

    def test_joint_probability_mass_must_be_complete(self):
        row = _one_query(
            joint_proposals=[
                ["a", 0.0, 1.0, 2.0, 0.5],
                ["z", 0.0, 1.0, 1.0, 0.4],
            ]
        )
        with self.assertRaisesRegex(MetricContractError, "sum to 1"):
            evaluate_queries([row])

    def test_probability_is_required_for_full_joint_schema(self):
        row = _one_query(joint_proposals=[["a", 0.0, 1.0, 2.0]])
        with self.assertRaisesRegex(MetricContractError, "probability is required"):
            evaluate_queries([row])

    def test_stable_ties_use_video_then_start_then_end(self):
        scored = [
            {"video_id": "z", "score": 1.0},
            {"video_id": "a", "score": 1.0},
        ]
        row = _one_query(
            pooled_video_order=scored,
            late_video_order=scored,
            final_video_order=scored,
            joint_proposals=[
                ["z", 0.0, 1.0, 2.0, 0.4],
                ["a", 0.5, 1.5, 2.0, 0.3],
                ["a", 0.0, 1.0, 2.0, 0.3],
            ],
        )
        result = evaluate_queries([row])
        per_query = result["per_query"][0]
        self.assertEqual(1, per_query["pooled_gt_rank"])
        self.assertEqual("a", per_query["top1_video_id"])
        self.assertEqual(1, per_query["joint_marginal_gt_rank"])
        self.assertEqual(1.0, result["vcmr_metrics"]["VCMR_R@1_IoU@0.7"])
        self.assertEqual("available", result["margin_diagnostics"]["pooled"]["status"])
        self.assertEqual(
            0.0,
            result["margin_diagnostics"]["pooled"]["mean_top1_top2_score_margin"],
        )

    def test_temporal_iou_uses_continuous_seconds(self):
        self.assertEqual(1.0, temporal_iou([0.0, 2.0], [0.0, 2.0]))
        self.assertEqual(0.0, temporal_iou([0.0, 1.0], [1.0, 2.0]))
        self.assertAlmostEqual(1.0 / 3.0, temporal_iou([0.0, 2.0], [1.0, 3.0]))

    def test_raw_mass_is_preserved_while_vcmr_uses_frozen_nms(self):
        row = _one_query(
            joint_proposals=[
                ["a", 0.0, 1.0, 4.0, 0.2],
                ["a", 0.0, 1.0, 3.0, 0.3],
                ["z", 0.0, 1.0, 2.0, 0.5],
            ]
        )
        result = evaluate_queries([row])
        query = result["per_query"][0]
        self.assertEqual(1, query["raw_exact_duplicate_proposal_count"])
        self.assertEqual(2, query["nms_prediction_count"])
        self.assertEqual(1, query["joint_marginal_gt_rank"])
        self.assertAlmostEqual(0.5, query["joint_false_positive_mass"])
        self.assertEqual(1.0, result["vcmr_metrics"]["VCMR_R@1_IoU@0.7"])

    def test_frozen_c7_rows_are_not_nms_reapplied(self):
        rows = [
            {
                "query_id": 1,
                "gt_video_id": "a",
                "gt_span_sec": [0.1, 1.1],
                "joint_proposals": [
                    ["a", 0.0, 1.0, 2.0],
                    ["a", 0.1, 1.1, 3.0],
                ],
            }
        ]
        replay = evaluate_joint_vcmr_and_errors(
            rows,
            require_probabilities=False,
            expected_query_ids=[1],
            input_is_frozen_nms=True,
        )
        self.assertFalse(replay["nms_reapplied"])
        self.assertEqual("FROZEN_POST_NMS_NO_REAPPLICATION", replay["input_list_semantics"])
        self.assertAlmostEqual(0.8181818181818181, replay["per_query"][0]["top1_iou"])


class SchemaAndGateTest(unittest.TestCase):
    def setUp(self):
        self.result = evaluate_fixture(_load_json("metric_fixture_v1.json"))

    def test_schema_freezes_metric_families_and_missing_policy(self):
        schema = metric_schema()
        self.assertEqual(METRIC_SCHEMA_VERSION, schema["schema_version"])
        self.assertTrue(schema["retrieval_families"]["pooled_true_VR"]["proposal_independent"])
        self.assertEqual(
            "miss", schema["retrieval_families"]["late_true_VR"]["GT_outside_broad"]
        )
        self.assertEqual(
            "unavailable",
            schema["errors"]["high_confidence_false_positive_rate"][
                "without_compliant_prefit_temperature"
            ],
        )
        self.assertEqual("forbidden", schema["legacy_fields"]["overwrite"])

    def test_gate_schema_freezes_exact_thresholds(self):
        schema = gate_schema()
        self.assertEqual(0.90, schema["thresholds"]["G_broad"]["value"])
        self.assertEqual(0.90, schema["thresholds"]["G_keep"]["value"])
        self.assertEqual(0.80, schema["thresholds"]["G_front"]["value"])
        self.assertEqual(
            1e-8, schema["invariants"]["G_late_product"]["absolute_tolerance"]
        )
        self.assertEqual(
            1e-6,
            schema["thresholds"]["historical_vcmr_parity_max_abs_delta"]["value"],
        )

    def test_fixture_gates_report_each_decision_without_relaxation(self):
        gates = evaluate_gates(self.result, parity_max_abs_delta=1e-6)
        self.assertFalse(gates["decisions"]["G_broad"])
        self.assertTrue(gates["decisions"]["G_keep"])
        self.assertFalse(gates["decisions"]["G_front"])
        self.assertTrue(gates["decisions"]["G_late_product"])
        self.assertTrue(gates["decisions"]["query_completeness"])
        self.assertTrue(gates["decisions"]["historical_vcmr_parity"])
        self.assertFalse(gates["all_pass"])

    def test_missing_conditional_gate_fails_closed(self):
        result = evaluate_queries(
            [
                _one_query(
                    pooled_video_order=[],
                    late_video_order=[],
                    final_video_order=[],
                    joint_proposals=[],
                )
            ]
        )
        with self.assertRaisesRegex(MetricContractError, "G_keep is unavailable"):
            evaluate_gates(result, parity_max_abs_delta=0.0)

    def test_zero_broad_denominator_fails_closed_with_contract_error(self):
        result = copy.deepcopy(self.result)
        result["gate_metrics"].update(
            {
                "G_broad": 0.0,
                "G_late": 0.0,
                "G_keep": 0.0,
                "G_front": 0.0,
                "G_late_product_residual": 0.0,
                "G_late_product_invariant": True,
            }
        )
        result["retrieval_metrics"].update(
            {
                "pooled_true_VR_R@1000": 0.0,
                "late_true_VR_R@200": 0.0,
                "late_true_VR_R@100": 0.0,
            }
        )
        with self.assertRaisesRegex(
            MetricContractError,
            "broad-hit denominator is zero",
        ):
            evaluate_gates(result, parity_max_abs_delta=0.0)

    def test_high_confidence_is_explicitly_unavailable_without_prefit_temperature(self):
        high_confidence = self.result["calibration"]["high_confidence_false_positive_rate"]
        self.assertEqual("unavailable", high_confidence["status"])
        self.assertIsNone(high_confidence["value"])
        self.assertFalse(high_confidence["fit_performed_during_evaluation"])

    def test_plain_prefit_temperature_mapping_is_never_trusted(self):
        base = {
            "schema_version": "c28f_prefit_temperature_v1",
            "validation_status": "VALID",
            "quantile": 0.90,
            "fit_performed_during_evaluation": False,
            "calibrated_top1_probability_by_query": {"str:q1": 0.9},
        }
        encoded = json.dumps(
            base,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        artifact = {**base, "artifact_sha256": hashlib.sha256(encoded).hexdigest()}
        with self.assertRaisesRegex(MetricContractError, "committed control capability"):
            evaluate_queries([_one_query()], prefit_temperature_artifact=artifact)
        tampered = copy.deepcopy(artifact)
        tampered["calibrated_top1_probability_by_query"]["str:q1"] = 0.8
        with self.assertRaisesRegex(MetricContractError, "committed control capability"):
            evaluate_queries([_one_query()], prefit_temperature_artifact=tampered)

    def test_full_calibration_schema_is_explicitly_unavailable_without_capability(self):
        calibration = self.result["calibration"]
        for key in (
            "ECE@0.5",
            "ECE@0.7",
            "top1_binary_NLL",
            "reliability_diagram",
            "structured_joint_NLL_conditional",
        ):
            self.assertEqual("unavailable", calibration[key]["status"])
            self.assertIn("COMMITTED", calibration[key]["reason"])

    def test_legacy_fields_are_additive_only(self):
        legacy = {"VR_R@1": 0.1, "VCMR_R@1": 0.2}
        merged = additive_merge_no_overwrite(legacy, {"c28f_metrics": self.result})
        self.assertEqual(legacy["VR_R@1"], merged["VR_R@1"])
        with self.assertRaisesRegex(MetricContractError, "overwrite forbidden"):
            additive_merge_no_overwrite(legacy, {"VR_R@1": 0.9})

    def test_vcmr_parity_uses_all_eight_frozen_fields(self):
        current = self.result["vcmr_metrics"]
        legacy = {
            key: value
            for key, value in current.items()
            if key.startswith("VCMR_R@")
        }
        exact = compare_vcmr_parity(legacy, current)
        self.assertEqual("PASS", exact["status"])
        self.assertEqual(8, exact["compared_field_count"])
        changed = copy.deepcopy(current)
        changed["VCMR_R@1_IoU@0.7"] += 2e-6
        failed = compare_vcmr_parity(legacy, changed)
        self.assertEqual("FAIL", failed["status"])
        self.assertGreater(failed["max_abs_delta"], 1e-6)

    def test_vcmr_parity_explicitly_converts_legacy_percentages(self):
        current = self.result["vcmr_metrics"]
        legacy_percent = {
            key: value * 100.0
            for key, value in current.items()
            if key.startswith("VCMR_R@")
        }
        parity = compare_vcmr_parity(
            legacy_percent,
            current,
            legacy_unit="percent",
            current_unit="ratio",
        )
        self.assertEqual("PASS", parity["status"])
        self.assertEqual("ratio", parity["comparison_unit"])


class C7StreamingAdapterTest(unittest.TestCase):
    def test_metric_aligner_has_no_authority_factory_or_path_api(self):
        import inspect

        parameters = set(inspect.signature(iter_aligned_c7_replay_rows).parameters)
        self.assertEqual(
            {
                "prediction_stream",
                "score_stream",
                "video_id_decoder",
                "span_decoder",
            },
            parameters,
        )

    def test_valid_stream_is_aligned_grouped_and_path_free(self):
        predictions = [
            {"q": 1, "rank": 0, "video_idx": 2, "start_idx": 0, "end_idx": 1},
            {"q": 1, "rank": 1, "video_idx": 3, "start_idx": 2, "end_idx": 3},
            {"q": 2, "rank": 0, "video_idx": 4, "start_idx": 1, "end_idx": 2},
        ]
        scores = [
            {"q": 1, "rank": 0, "score": 4.0},
            {"q": 1, "rank": 1, "score": 3.0},
            {"q": 2, "rank": 0, "score": 2.0},
        ]
        packets = list(
            iter_aligned_c7_replay_rows(
                iter(predictions),
                iter(scores),
                video_id_decoder=lambda index: "video-%d" % index,
                span_decoder=lambda query, video, start, end: (
                    float(start),
                    float(end + 1),
                ),
            )
        )
        self.assertEqual([1, 2], [packet["query_id"] for packet in packets])
        self.assertEqual(2, len(packets[0]["joint_proposals"]))
        self.assertEqual(
            ["video-2", 0.0, 2.0, 4.0], packets[0]["joint_proposals"][0]
        )
        self.assertNotIn("path", packets[0])

    def test_stream_alignment_mismatch_fails_closed(self):
        predictions = [
            {"q": 1, "rank": 0, "video_idx": 2, "start_idx": 0, "end_idx": 1}
        ]
        scores = [{"q": 1, "rank": 1, "score": 4.0}]
        with self.assertRaisesRegex(MetricContractError, "alignment mismatch"):
            list(
                iter_aligned_c7_replay_rows(
                    iter(predictions),
                    iter(scores),
                    video_id_decoder=lambda index: "video-%d" % index,
                    span_decoder=lambda query, video, start, end: (
                        float(start),
                        float(end + 1),
                    ),
                )
            )

    def test_rank_gap_fails_closed(self):
        predictions = [
            {"q": 1, "rank": 0, "video_idx": 2, "start_idx": 0, "end_idx": 1},
            {"q": 1, "rank": 2, "video_idx": 3, "start_idx": 2, "end_idx": 3},
        ]
        scores = [
            {"q": 1, "rank": 0, "score": 4.0},
            {"q": 1, "rank": 2, "score": 3.0},
        ]
        with self.assertRaisesRegex(MetricContractError, "contiguous from zero"):
            list(
                iter_aligned_c7_replay_rows(
                    iter(predictions),
                    iter(scores),
                    video_id_decoder=lambda index: "video-%d" % index,
                    span_decoder=lambda query, video, start, end: (
                        float(start),
                        float(end + 1),
                    ),
                )
            )


if __name__ == "__main__":
    unittest.main()
