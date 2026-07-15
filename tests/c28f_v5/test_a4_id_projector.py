from __future__ import annotations

import hashlib
import inspect
import unittest

from blueprint_e2e_v2.c28f_v5.a4_id_projector import (
    AUTHORIZED_TRAIN_PROJECTION_PURPOSES,
    IdProjectionError,
    ProjectionResult,
    build_projection_manifest,
    extract_authorized_train_projection_line,
    project_authorized_id_mapping,
)
from blueprint_e2e_v2.c28f_v5.a4_control import (
    A4G4MinimalProjectionBatch,
    A4G4RunningProjectionReader,
)


class ProtectedIdProjectorControlBoundaryTests(unittest.TestCase):
    def test_control_projection_surface_cannot_return_raw_lines(self) -> None:
        self.assertFalse(hasattr(A4G4RunningProjectionReader, "consume_lines_once"))
        self.assertFalse(hasattr(A4G4RunningProjectionReader, "expected_desc_ids_once"))
        self.assertFalse(
            hasattr(A4G4RunningProjectionReader, "close_and_issue_access_receipt")
        )
        self.assertTrue(
            hasattr(A4G4RunningProjectionReader, "consume_minimal_projection_once")
        )
        self.assertTrue(hasattr(A4G4MinimalProjectionBatch, "consume_snapshot_once"))

    def test_plain_iterable_cannot_open_projection_source(self) -> None:
        with self.assertRaises(IdProjectionError) as captured:
            project_authorized_id_mapping(
                [b'{"desc_id":20,"vid_name":"video-20"}\n'],
            )
        self.assertIn(
            captured.exception.code,
            {
                "PROJECTOR_CONCRETE_CONTROL_READER_REQUIRED",
                "PROJECTOR_CONCRETE_CONTROL_READER_UNAVAILABLE",
            },
        )

    def test_projection_result_direct_construction_is_rejected(self) -> None:
        with self.assertRaises(IdProjectionError) as captured:
            ProjectionResult(
                records=(),
                source_line_count=0,
                source_bytes_scanned=0,
                expected_desc_id_count=0,
                syntax_scan_pass_count=0,
                top_level_key_semantic_decode_count=0,
                desc_id_semantic_decode_count=0,
                video_id_semantic_decode_count=0,
                unknown_value_syntax_scan_count=0,
                unknown_value_syntax_bytes_scanned=0,
                control_capability_sha256="0" * 64,
                control_binding={},
                access_receipt={},
                minimal_batch=object(),
                control_result_seal_sha256="1" * 64,
                _issuer=None,
            )
        self.assertEqual(
            captured.exception.code, "PROJECTOR_RESULT_NOT_CONTROL_ISSUED"
        )

    def test_projection_manifest_rejects_unsealed_object(self) -> None:
        with self.assertRaises(IdProjectionError) as captured:
            build_projection_manifest(object())  # type: ignore[arg-type]
        self.assertEqual(
            captured.exception.code, "PROJECTOR_CONTROL_SEALED_RESULT_REQUIRED"
        )


class AuthorizedIndexedTrainProjectionLineTests(unittest.TestCase):
    ENCODED_LINE = (
        b'{"desc_id":7,"desc":"walks \\u4eba","type":"vt",'
        b'"ts":[1.25,4],"vid_name":"video-7",'
        b'"teacher":{"secret":[true,false,null]},"label":[1,2]}\n'
    )

    def test_surface_is_pure_and_has_no_path_or_fd_parameter(self) -> None:
        self.assertEqual(
            AUTHORIZED_TRAIN_PROJECTION_PURPOSES,
            (
                "FORWARD_QUERY_IDENTITY",
                "POST_FORWARD_EVAL_STATS",
                "TRAIN_FIT_ROLE_MAPPING",
                "PROTECTED_ID_MAPPING_ONLY",
            ),
        )
        self.assertEqual(
            tuple(
                inspect.signature(
                    extract_authorized_train_projection_line
                ).parameters
            ),
            ("encoded_line", "purpose", "expected_desc_id"),
        )
        source = inspect.getsource(
            extract_authorized_train_projection_line
        )
        self.assertIn("_scan_authorized_top_level_value_slices", source)
        self.assertNotIn("json.loads(encoded_line", source)
        self.assertNotIn("open(", source)
        self.assertNotIn("pread", source)

    def test_forward_identity_hashes_but_does_not_decode_desc(self) -> None:
        projection = extract_authorized_train_projection_line(
            self.ENCODED_LINE,
            purpose="FORWARD_QUERY_IDENTITY",
            expected_desc_id=7,
        )
        self.assertEqual(
            projection["retained_record"],
            {
                "desc_id": 7,
                "type": "vt",
                "desc_raw_json_sha256": hashlib.sha256(
                    b'"walks \\u4eba"'
                ).hexdigest(),
            },
        )
        self.assertEqual(
            projection["retained_fields"],
            ["desc_id", "type", "desc_raw_json_sha256"],
        )
        self.assertEqual(
            list(projection["retained_record"]),
            projection["retained_fields"],
        )
        counters = projection["counters"]
        self.assertEqual(
            counters["decoded_field_counts"],
            {"desc_id": 1, "desc": 0, "type": 1, "ts": 0, "vid_name": 0},
        )
        self.assertEqual(counters["raw_field_hash_counts"], {"desc": 1})
        self.assertEqual(counters["full_line_json_decode_count"], 0)
        self.assertEqual(
            counters["unauthorized_field_semantic_decode_count"],
            0,
        )
        self.assertEqual(counters["selected_value_syntax_scan_count"], 3)
        self.assertEqual(counters["nonselected_value_syntax_scan_count"], 4)

    def test_post_forward_projection_decodes_only_exact_allowed_fields(self) -> None:
        projection = extract_authorized_train_projection_line(
            self.ENCODED_LINE,
            purpose="POST_FORWARD_EVAL_STATS",
            expected_desc_id=7,
        )
        self.assertEqual(
            projection["retained_record"],
            {
                "desc_id": 7,
                "desc": "walks 人",
                "type": "vt",
                "ts": [1.25, 4.0],
                "vid_name": "video-7",
            },
        )
        self.assertEqual(
            projection["counters"]["decoded_field_counts"],
            {"desc_id": 1, "desc": 1, "type": 1, "ts": 1, "vid_name": 1},
        )
        self.assertEqual(
            projection["counters"]["nonselected_value_syntax_scan_count"],
            2,
        )

    def test_train_fit_and_protected_mapping_retain_only_two_ids(self) -> None:
        for purpose in (
            "TRAIN_FIT_ROLE_MAPPING",
            "PROTECTED_ID_MAPPING_ONLY",
        ):
            with self.subTest(purpose=purpose):
                projection = extract_authorized_train_projection_line(
                    self.ENCODED_LINE,
                    purpose=purpose,
                    expected_desc_id=7,
                )
                self.assertEqual(
                    projection["retained_record"],
                    {"desc_id": 7, "vid_name": "video-7"},
                )
                self.assertEqual(
                    projection["counters"]["decoded_field_counts"],
                    {
                        "desc_id": 1,
                        "desc": 0,
                        "type": 0,
                        "ts": 0,
                        "vid_name": 1,
                    },
                )

    def test_purpose_identity_and_line_boundaries_fail_closed(self) -> None:
        cases = (
            (
                self.ENCODED_LINE,
                "FORWARD_QUERY_IDENTITY",
                8,
                "PROJECTOR_INDEX_DESC_ID_BINDING_MISMATCH",
            ),
            (
                self.ENCODED_LINE,
                "UNAUTHORIZED",
                7,
                "PROJECTOR_TRAIN_PURPOSE_UNAUTHORIZED",
            ),
            (
                self.ENCODED_LINE,
                "FORWARD_QUERY_IDENTITY",
                9_223_372_036_854_775_808,
                "PROJECTOR_EXPECTED_DESC_ID_INVALID",
            ),
            (
                self.ENCODED_LINE[:-1],
                "FORWARD_QUERY_IDENTITY",
                7,
                "PROJECTOR_INDEXED_LINE_BOUNDARY_INVALID",
            ),
            (
                self.ENCODED_LINE + b"{}\n",
                "FORWARD_QUERY_IDENTITY",
                7,
                "PROJECTOR_INDEXED_LINE_BOUNDARY_INVALID",
            ),
        )
        for encoded, purpose, expected_desc_id, code in cases:
            with self.subTest(code=code):
                with self.assertRaises(IdProjectionError) as captured:
                    extract_authorized_train_projection_line(
                        encoded,
                        purpose=purpose,
                        expected_desc_id=expected_desc_id,
                    )
                self.assertEqual(captured.exception.code, code)

    def test_missing_duplicate_or_invalid_nonselected_values_fail_closed(self) -> None:
        cases = (
            (
                b'{"desc_id":7,"type":"vt"}\n',
                "PROJECTOR_SELECTED_FIELD_MISSING",
            ),
            (
                b'{"desc_id":7,"desc":"a","desc":"b","type":"vt"}\n',
                "PROJECTOR_TOP_LEVEL_KEY_DUPLICATE",
            ),
            (
                b'{"desc_id":7,"desc":"a","type":"vt",'
                b'"teacher":1,"teacher":2}\n',
                "PROJECTOR_TOP_LEVEL_KEY_DUPLICATE",
            ),
            (
                b'{"desc_id":7,"desc":"a","type":"vt",'
                b'"teacher":{"x":1,}}\n',
                "PROJECTOR_JSON_STRING_REQUIRED",
            ),
            (
                b'{"desc_id":7,"desc":"a","type":"vt",'
                b'"teacher":"\\uD800"}\n',
                "PROJECTOR_JSON_SURROGATE_PAIR_INVALID",
            ),
        )
        for encoded, code in cases:
            with self.subTest(code=code):
                with self.assertRaises(IdProjectionError) as captured:
                    extract_authorized_train_projection_line(
                        encoded,
                        purpose="FORWARD_QUERY_IDENTITY",
                        expected_desc_id=7,
                    )
                self.assertEqual(captured.exception.code, code)


if __name__ == "__main__":
    raise SystemExit("Run through the reviewed C28F stage test launcher only")
