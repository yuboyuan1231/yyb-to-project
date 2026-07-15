from __future__ import annotations

import json
import hashlib
import os
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from typing import Mapping
from unittest import mock

from blueprint_e2e_v2.c28f_v5.a4_security import (
    AuthorizedLmdbKeyBatch,
    DescriptorPathGuard,
    EntityIdSet,
    EvalBudgetBook,
    EvalBudgetToken,
    EvaluationRequest,
    EvidenceLedger,
    FrozenCanonicalRoot,
    FrozenPathRule,
    LmdbKeyManifest,
    PathIntent,
    PROTECTED_ID_MAPPING_RUNNING_ANCHOR_SCHEMA_VERSION,
    SecurityViolation,
    SplitAuthorizationRegistry,
    SplitCapability,
    assert_isolated_write_roots,
    authorize_lmdb_key,
    authorize_lmdb_key_batch,
    build_lmdb_batch_persistence_intent,
    canonical_lmdb_key_sha256,
    canonical_lmdb_key_manifest_sha256,
    canonical_path_map,
    derive_safety_manifest,
    intersect_entity_id_sets,
    lmdb_key_manifest_semantic_sha256,
    lex_minimal_id_mapping_jsonl_line,
    reject_legacy_authorization_markers,
    open_pinned_content,
    prepare_eval_token_cas_intent,
    rehydrate_persisted_lmdb_batch_receipt,
    validate_single_action_argv,
    verify_claimed_safety_manifest,
    verify_ledger_rows,
    validate_persisted_eval_token_receipt,
    validate_persisted_lmdb_batch_receipt,
    validate_protected_id_mapping_running_anchor,
)


FIXTURE_PATH = Path(__file__).parent / "fixtures/a4_security_fixture.json"


class SyntheticFixtureCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.fixture = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
        if cls.fixture["synthetic_only"] is not True:
            raise AssertionError("security fixture must be synthetic")
        if cls.fixture["contains_real_protected_mapping"] is not False:
            raise AssertionError("real protected mappings are forbidden in this suite")

    def ledger(self) -> EvidenceLedger:
        return EvidenceLedger(
            self.fixture["scope_id"],
            self.fixture["goal_id"],
            self.fixture["authority_id"],
        )

    def capability(self) -> SplitCapability:
        return SplitCapability(**self.fixture["split_capability"])

    def registry(self) -> SplitAuthorizationRegistry:
        return SplitAuthorizationRegistry(
            goal_id=self.fixture["goal_id"],
            authority_id=self.fixture["authority_id"],
            capabilities=(self.capability(),),
            forbidden_split_aliases=self.fixture["forbidden_split_aliases"],
        )

    def request(self) -> EvaluationRequest:
        return EvaluationRequest(**self.fixture["metadata_dry_run_request"])

    def budget(
        self,
        *,
        state: str = "UNISSUED",
        use_count: int = 0,
    ) -> EvalBudgetBook:
        reserved = "SYNTHETIC-PRIOR-EVAL" if state in {
            "RESERVED",
            "RUNNING",
            "COMPLETED",
            "EXHAUSTED",
        } else None
        return EvalBudgetBook(
            tokens=(
                EvalBudgetToken(
                    token_id="TOKEN-F0A",
                    purpose="F0A_TEACHER_FREE_FORENSICS",
                    state=state,
                    max_uses=1,
                    use_count=use_count,
                    reserved_eval_id=reserved,
                ),
            )
        )

    def assert_code(self, expected: str, action: object) -> SecurityViolation:
        with self.assertRaises(SecurityViolation) as captured:
            action()  # type: ignore[operator]
        self.assertEqual(captured.exception.code, expected)
        return captured.exception


class LedgerAndSafetyTests(SyntheticFixtureCase):
    def test_denial_is_chained_before_any_io(self) -> None:
        ledger = self.ledger()
        ledger.record_preopen(
            event_type="PATH_PREOPEN",
            decision="DENY",
            reason_code="SYNTHETIC_DENIAL",
            classification="PROTECTED_METADATA",
            request_fingerprint="a" * 64,
            details={"content_opened": False},
        )
        ledger.close(instrumentation_complete=True, unmediated_content_open_count=0)
        verification = verify_ledger_rows(
            ledger.rows,
            expected_scope_id=self.fixture["scope_id"],
            expected_goal_id=self.fixture["goal_id"],
            expected_authority_id=self.fixture["authority_id"],
            require_closed=True,
        )
        self.assertTrue(verification["verified"])
        denied = ledger.rows[1]
        self.assertEqual(denied["decision"], "DENY")
        self.assertTrue(all(value == 0 for value in denied["counters"].values()))

    def test_tampered_ledger_fails_closed(self) -> None:
        ledger = self.ledger()
        ledger.record_preopen(
            event_type="EVAL_PREOPEN",
            decision="DENY",
            reason_code="SYNTHETIC_DENIAL",
            classification="MODEL_FORWARD",
            request_fingerprint="b" * 64,
        )
        ledger.close(instrumentation_complete=True, unmediated_content_open_count=0)
        rows = [dict(row) for row in ledger.rows]
        rows[1]["reason_code"] = "TAMPERED"
        self.assert_code(
            "LEDGER_EVENT_HASH_MISMATCH",
            lambda: verify_ledger_rows(
                rows,
                expected_scope_id=self.fixture["scope_id"],
                expected_goal_id=self.fixture["goal_id"],
                expected_authority_id=self.fixture["authority_id"],
                require_closed=True,
            ),
        )

    def test_safety_manifest_is_ledger_derived_and_mismatch_rejected(self) -> None:
        ledger = self.ledger()
        ledger.record_preopen(
            event_type="EVAL_PREOPEN",
            decision="DENY",
            reason_code="PROTECTED_SPLIT_FORBIDDEN",
            classification="OFFICIAL_WORKFLOW",
            request_fingerprint="c" * 64,
        )
        ledger.close(instrumentation_complete=True, unmediated_content_open_count=0)
        manifest = derive_safety_manifest(
            ledger.rows,
            expected_scope_id=self.fixture["scope_id"],
            expected_goal_id=self.fixture["goal_id"],
            expected_authority_id=self.fixture["authority_id"],
        )
        self.assertTrue(
            manifest["claims"]["safe_for_nonprotected_f0_metadata_dry_run"]
        )
        forged = json.loads(json.dumps(manifest))
        forged["claims"]["protected_content_accessed"] = True
        self.assert_code(
            "SAFETY_MANIFEST_LEDGER_MISMATCH",
            lambda: verify_claimed_safety_manifest(
                forged,
                ledger.rows,
                expected_scope_id=self.fixture["scope_id"],
                expected_goal_id=self.fixture["goal_id"],
                expected_authority_id=self.fixture["authority_id"],
            ),
        )

    def test_unmediated_or_protected_content_cannot_be_reported_safe(self) -> None:
        ledger = self.ledger()
        ledger.record_observation(
            event_type="CONTENT_OPEN_OBSERVED",
            classification="PROTECTED_CONTENT",
            request_fingerprint="d" * 64,
            counters={"content_open_count": 1, "content_bytes_read": 8},
        )
        ledger.close(instrumentation_complete=True, unmediated_content_open_count=0)
        manifest = derive_safety_manifest(
            ledger.rows,
            expected_scope_id=self.fixture["scope_id"],
            expected_goal_id=self.fixture["goal_id"],
            expected_authority_id=self.fixture["authority_id"],
        )
        self.assertTrue(manifest["claims"]["protected_content_accessed"])
        self.assertFalse(
            manifest["claims"]["safe_for_nonprotected_f0_metadata_dry_run"]
        )

    def test_unresolved_open_intent_cannot_be_reported_safe(self) -> None:
        ledger = self.ledger()
        ledger.record_observation(
            event_type="CONTENT_OPEN_INTENT_COMMITTED",
            classification="NON_PROTECTED_CONTENT",
            request_fingerprint="e" * 64,
            counters={},
            details={"open_intent_id": "SYNTHETIC-OPEN-1"},
        )
        ledger.close(instrumentation_complete=True, unmediated_content_open_count=0)
        manifest = derive_safety_manifest(
            ledger.rows,
            expected_scope_id=self.fixture["scope_id"],
            expected_goal_id=self.fixture["goal_id"],
            expected_authority_id=self.fixture["authority_id"],
        )
        self.assertEqual(1, manifest["claims"]["unresolved_open_intent_count"])
        self.assertFalse(
            manifest["claims"]["safe_for_nonprotected_f0_metadata_dry_run"]
        )

    def test_nonprotected_content_or_lmdb_io_is_not_a_metadata_dry_run(self) -> None:
        for counter_name in ("content_open_count", "lmdb_transaction_open_count"):
            ledger = self.ledger()
            ledger.record_observation(
                event_type="UNMEDIATED_SYNTHETIC_IO",
                classification="NON_PROTECTED_CONTENT",
                request_fingerprint="f" * 64,
                counters={counter_name: 1},
            )
            ledger.close(
                instrumentation_complete=True,
                unmediated_content_open_count=0,
            )
            manifest = derive_safety_manifest(
                ledger.rows,
                expected_scope_id=self.fixture["scope_id"],
                expected_goal_id=self.fixture["goal_id"],
                expected_authority_id=self.fixture["authority_id"],
            )
            self.assertFalse(
                manifest["claims"]["safe_for_nonprotected_f0_metadata_dry_run"]
            )
            self.assertEqual(
                1 if counter_name == "content_open_count" else 0,
                manifest["claims"]["observed_unmediated_content_open_count"],
            )


class DescriptorPathGuardTests(SyntheticFixtureCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory(prefix="c28f-a4-synthetic-")
        self.base = Path(self.temp.name)
        self.allowed = self.base / "allowed"
        self.output = self.base / "output"
        self.history = self.base / "history"
        self.allowed.mkdir()
        self.output.mkdir()
        self.history.mkdir()
        self.safe_file = self.allowed / "metadata.json"
        self.safe_file.write_text("SYNTHETIC_SAFE", encoding="utf-8")
        self.secret = self.history / "protected.bin"
        self.secret.write_bytes(b"SYNTHETIC_SENTINEL_DO_NOT_READ")
        self.read_root = FrozenCanonicalRoot.capture(
            "synthetic-read-root",
            str(self.base),
            ("READ_CONTENT", "READ_METADATA"),
        )
        self.write_root = FrozenCanonicalRoot.capture(
            "synthetic-output-root", str(self.output), ("WRITE",)
        )
        self.history_rule = FrozenPathRule.capture_metadata_only(
            "synthetic-history-tree",
            str(self.history),
            scope="TREE",
            classification="HISTORICAL_ROOT",
        )
        self.secret_rule = FrozenPathRule.capture_metadata_only(
            "synthetic-protected-file",
            str(self.secret),
            scope="EXACT",
            classification="PROTECTED_DATA",
        )
        self.guard = DescriptorPathGuard(
            goal_id=self.fixture["goal_id"],
            authority_id=self.fixture["authority_id"],
            roots=(self.read_root, self.write_root),
            forbidden_rules=(self.history_rule, self.secret_rule),
        )

    def tearDown(self) -> None:
        self.temp.cleanup()

    def intent(self, path: str, operation: str = "READ_CONTENT") -> PathIntent:
        return PathIntent(
            request_id="SYNTHETIC-PATH-REQUEST",
            operation=operation,
            path=path,
            expected_kind="FILE",
            purpose="SYNTHETIC_FIREWALL_TEST",
            goal_id=self.fixture["goal_id"],
            authority_id=self.fixture["authority_id"],
        )

    def test_allowed_path_is_only_preopened_by_descriptor(self) -> None:
        ledger = self.ledger()
        with mock.patch("builtins.open", side_effect=AssertionError("content open")):
            grant = self.guard.authorize(self.intent(str(self.safe_file)), ledger)
        self.assertFalse(grant.content_opened)
        self.assertTrue(grant.existing)
        self.assertEqual(grant.final_identity.object_kind, "FILE")
        self.assertEqual(ledger.rows[-1]["counters"]["content_open_count"], 0)

    def test_dotdot_rejected_before_content_open(self) -> None:
        ledger = self.ledger()
        traversal = str(self.allowed / ".." / "history" / self.secret.name)
        with mock.patch("builtins.open", side_effect=AssertionError("content open")):
            self.assert_code(
                "PATH_TRAVERSAL_OR_DOT_COMPONENT",
                lambda: self.guard.authorize(self.intent(traversal), ledger),
            )
        self.assertEqual(ledger.rows[-1]["decision"], "DENY")

    def test_symlink_and_device_inode_alias_are_rejected(self) -> None:
        symlink = self.allowed / "protected-symlink"
        symlink.symlink_to(self.secret)
        symlink_ledger = self.ledger()
        self.assert_code(
            "SYMLINK_COMPONENT_FORBIDDEN",
            lambda: self.guard.authorize(self.intent(str(symlink)), symlink_ledger),
        )
        hardlink = self.allowed / "protected-hardlink"
        os.link(self.secret, hardlink)
        hardlink_ledger = self.ledger()
        self.assert_code(
            "FORBIDDEN_DEVICE_INODE_ALIAS",
            lambda: self.guard.authorize(self.intent(str(hardlink)), hardlink_ledger),
        )

    def test_forbidden_exact_tree_and_output_overlap_are_rejected(self) -> None:
        ledger = self.ledger()
        self.assert_code(
            "FORBIDDEN_PATH_EXACT_OR_TREE",
            lambda: self.guard.authorize(self.intent(str(self.secret)), ledger),
        )
        history_write = FrozenCanonicalRoot.capture(
            "bad-output", str(self.history), ("WRITE",)
        )
        self.assert_code(
            "WRITE_ROOT_FORBIDDEN_OVERLAP",
            lambda: assert_isolated_write_roots(
                (history_write,), (self.history_rule,)
            ),
        )

    def test_missing_output_leaf_preflight_does_not_write(self) -> None:
        ledger = self.ledger()
        target = self.output / "new-report.json"
        grant = self.guard.authorize(self.intent(str(target), "WRITE_FILE"), ledger)
        self.assertFalse(grant.existing)
        self.assertFalse(target.exists())
        self.assertEqual(ledger.rows[-1]["counters"]["content_bytes_written"], 0)

    def test_path_map_contains_metadata_not_content_hashes(self) -> None:
        mapping = canonical_path_map(
            (self.read_root, self.write_root),
            (self.history_rule, self.secret_rule),
        )
        forbidden = {item["rule_id"]: item for item in mapping["forbidden_registry"]}
        self.assertIsNone(forbidden["synthetic-protected-file"]["content_sha256"])
        self.assertTrue(mapping["path_policy"]["descriptor_walk_required"])


class CapabilityAndLauncherTests(SyntheticFixtureCase):
    def test_normal_allowlist_dry_run_neither_reserves_nor_forwards(self) -> None:
        ledger = self.ledger()
        before = self.budget()
        authorization, after = self.registry().authorize(
            self.request(), before, ledger, dry_run=True
        )
        self.assertTrue(authorization.dry_run)
        self.assertFalse(authorization.model_forward_allowed)
        self.assertEqual(before.as_dict(), after.as_dict())
        self.assertEqual(after.get("TOKEN-F0A").state, "UNISSUED")
        self.assertEqual(ledger.rows[-1]["counters"]["model_forward_count"], 0)

    def test_missing_or_null_field_never_falls_back_to_a_default(self) -> None:
        for field in (
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
        ):
            request = replace(self.request(), **{field: None})
            ledger = self.ledger()
            self.assert_code(
                "MISSING_REQUIRED_FIELD",
                lambda request=request, ledger=ledger: self.registry().authorize(
                    request, self.budget(), ledger, dry_run=True
                ),
            )

    def test_random_and_protected_split_aliases_are_denied(self) -> None:
        random_request = replace(self.request(), split_id="random_split")
        self.assert_code(
            "SPLIT_ID_MISMATCH",
            lambda: self.registry().authorize(
                random_request, self.budget(), self.ledger(), dry_run=True
            ),
        )
        for alias in (
            "calib_holdout",
            "pseudo-official-holdout",
            "Official Validation",
        ):
            request = replace(self.request(), split_id=alias)
            self.assert_code(
                "PROTECTED_SPLIT_FORBIDDEN",
                lambda request=request: self.registry().authorize(
                    request, self.budget(), self.ledger(), dry_run=True
                ),
            )

    def test_wrong_manifest_goal_authority_purpose_and_token_are_denied(self) -> None:
        cases = (
            ("split_manifest_sha256", "9" * 64, "SPLIT_MANIFEST_SHA_MISMATCH"),
            ("goal_id", "WRONG-GOAL", "EVAL_GOAL_ID_MISMATCH"),
            ("authority_id", "WRONG-AUTH", "EVAL_AUTHORITY_ID_MISMATCH"),
            ("purpose", "WRONG-PURPOSE", "EVAL_PURPOSE_MISMATCH"),
            ("budget_token_id", "WRONG-TOKEN", "EVAL_BUDGET_TOKEN_MISMATCH"),
            (
                "candidate_corpus_manifest_sha256",
                "8" * 64,
                "CANDIDATE_CORPUS_MANIFEST_SHA_MISMATCH",
            ),
        )
        for field, value, code in cases:
            request = replace(self.request(), **{field: value})
            self.assert_code(
                code,
                lambda request=request: self.registry().authorize(
                    request, self.budget(), self.ledger(), dry_run=True
                ),
            )

    def test_token_replay_and_over_budget_are_denied_preopen(self) -> None:
        self.assert_code(
            "EVAL_TOKEN_REPLAY",
            lambda: self.registry().authorize(
                self.request(),
                self.budget(state="COMPLETED", use_count=1),
                self.ledger(),
                dry_run=True,
            ),
        )
        self.assert_code(
            "EVAL_BUDGET_EXHAUSTED",
            lambda: self.registry().authorize(
                self.request(),
                self.budget(state="EXHAUSTED", use_count=1),
                self.ledger(),
                dry_run=True,
            ),
        )

    def test_legacy_markers_and_legacy_runner_routes_are_denied(self) -> None:
        for payload in self.fixture["legacy_marker_payloads"]:
            self.assert_code(
                "LEGACY_ROOT_AUTHORIZATION_MARKER_FORBIDDEN",
                lambda payload=payload: reject_legacy_authorization_markers(
                    payload, self.ledger()
                ),
            )
        for argv in self.fixture["invalid_launcher_argv"]:
            expected = (
                "LEGACY_OR_PROTECTED_LAUNCHER_PATH_FORBIDDEN"
                if argv
                else "LAUNCHER_REQUIRES_UNIQUE_CURRENT_ACTION"
            )
            self.assert_code(
                expected,
                lambda argv=argv: validate_single_action_argv(
                    argv,
                    expected_action="G1_FAIL_CLOSED_GUARD",
                    ledger=self.ledger(),
                ),
            )
        validate_single_action_argv(
            ("--action", "G1_FAIL_CLOSED_GUARD"),
            expected_action="G1_FAIL_CLOSED_GUARD",
            ledger=self.ledger(),
        )


class LmdbAndIdMappingTests(SyntheticFixtureCase):
    def lmdb_manifest(self) -> LmdbKeyManifest:
        spec = self.fixture["lmdb_key_fixture"]
        allowed = tuple(
            sorted(
                canonical_lmdb_key_sha256(
                    spec["entity_namespace"], item.encode("ascii")
                )
                for item in spec["allowed_synthetic_keys"]
            )
        )
        fields = {
            "source_id": spec["source_id"],
            "source_identity_sha256": spec["source_identity_sha256"],
            "key_manifest_sha256": canonical_lmdb_key_manifest_sha256(allowed),
            "split_authorization_sha256": spec["split_authorization_sha256"],
            "entity_namespace": spec["entity_namespace"],
            "key_encoding": spec["key_encoding"],
            "allowed_key_sha256": allowed,
        }
        return LmdbKeyManifest(
            **fields,
            manifest_semantic_sha256=lmdb_key_manifest_semantic_sha256(
                **fields
            ),
        )

    def authorize_key(self, raw: bytes, ledger: EvidenceLedger) -> object:
        spec = self.fixture["lmdb_key_fixture"]
        return authorize_lmdb_key(
            manifest=self.lmdb_manifest(),
            source_id=spec["source_id"],
            source_identity_sha256=spec["source_identity_sha256"],
            key_manifest_sha256=self.lmdb_manifest().key_manifest_sha256,
            split_authorization_sha256=spec["split_authorization_sha256"],
            entity_namespace=spec["entity_namespace"],
            raw_key=raw,
            ledger=ledger,
        )

    def test_lmdb_key_guard_is_pure_pretransaction_logic(self) -> None:
        ledger = self.ledger()
        grant = self.authorize_key(b"1001", ledger)
        self.assertFalse(grant.lmdb_transaction_opened)
        serialized = json.dumps(ledger.rows, sort_keys=True)
        self.assertNotIn('"1001"', serialized)
        self.assertEqual(
            ledger.rows[-1]["counters"]["lmdb_transaction_open_count"], 0
        )

    def test_lmdb_batch_direct_construction_is_rejected(self) -> None:
        self.assert_code(
            "LMDB_KEY_BATCH_NOT_ISSUED_BY_DETECTOR",
            lambda: AuthorizedLmdbKeyBatch(
                source_id="SYNTHETIC-QUERY-LMDB",
                manifest_semantic_sha256="1" * 64,
                key_manifest_sha256="2" * 64,
                split_authorization_sha256="3" * 64,
                key_hashes=("4" * 64,),
                ledger_tail_event_sha256="5" * 64,
                authorization_sha256="6" * 64,
                _issuer=None,
            ),
        )

    def test_lmdb_whole_key_set_is_authorized_before_first_transaction(self) -> None:
        spec = self.fixture["lmdb_key_fixture"]
        ledger = self.ledger()
        grants = authorize_lmdb_key_batch(
            manifest=self.lmdb_manifest(),
            source_id=spec["source_id"],
            source_identity_sha256=spec["source_identity_sha256"],
            key_manifest_sha256=self.lmdb_manifest().key_manifest_sha256,
            split_authorization_sha256=spec["split_authorization_sha256"],
            entity_namespace=spec["entity_namespace"],
            raw_keys=tuple(
                value.encode("ascii") for value in spec["allowed_synthetic_keys"]
            ),
            ledger=ledger,
        )
        self.assertEqual(len(grants), len(spec["allowed_synthetic_keys"]))
        self.assertEqual(tuple(sorted(grants.key_hashes)), grants.key_hashes)
        self.assertEqual(
            ledger.rows[-1]["counters"]["lmdb_transaction_open_count"], 0
        )
        serialized = json.dumps(ledger.rows, sort_keys=True)
        for raw_key in spec["allowed_synthetic_keys"]:
            self.assertNotIn(f'"{raw_key}"', serialized)

    def test_lmdb_batch_rejects_subset_duplicate_and_extra_pretransaction(self) -> None:
        spec = self.fixture["lmdb_key_fixture"]

        def authorize(raw_keys: tuple[bytes, ...]) -> object:
            return authorize_lmdb_key_batch(
                manifest=self.lmdb_manifest(),
                source_id=spec["source_id"],
                source_identity_sha256=spec["source_identity_sha256"],
                key_manifest_sha256=self.lmdb_manifest().key_manifest_sha256,
                split_authorization_sha256=spec["split_authorization_sha256"],
                entity_namespace=spec["entity_namespace"],
                raw_keys=raw_keys,
                ledger=self.ledger(),
            )

        self.assert_code(
            "LMDB_KEY_BATCH_NOT_EXACT_MANIFEST_SET",
            lambda: authorize((b"1001",)),
        )
        self.assert_code(
            "LMDB_KEY_BATCH_DUPLICATE",
            lambda: authorize((b"1001", b"1001")),
        )
        self.assert_code(
            "LMDB_KEY_NOT_AUTHORIZED",
            lambda: authorize((b"1001", b"1003")),
        )

    def test_lmdb_wrong_key_manifest_namespace_and_noncanonical_key_are_denied(self) -> None:
        self.assert_code(
            "LMDB_KEY_NOT_AUTHORIZED",
            lambda: self.authorize_key(b"1003", self.ledger()),
        )
        spec = self.fixture["lmdb_key_fixture"]
        self.assert_code(
            "LMDB_KEY_MANIFEST_SHA_MISMATCH",
            lambda: authorize_lmdb_key(
                manifest=self.lmdb_manifest(),
                source_id=spec["source_id"],
                source_identity_sha256=spec["source_identity_sha256"],
                key_manifest_sha256="9" * 64,
                split_authorization_sha256=spec["split_authorization_sha256"],
                entity_namespace=spec["entity_namespace"],
                raw_key=b"1001",
                ledger=self.ledger(),
            ),
        )
        self.assert_code(
            "LMDB_ENTITY_NAMESPACE_MISMATCH",
            lambda: authorize_lmdb_key(
                manifest=self.lmdb_manifest(),
                source_id=spec["source_id"],
                source_identity_sha256=spec["source_identity_sha256"],
                key_manifest_sha256=self.lmdb_manifest().key_manifest_sha256,
                split_authorization_sha256=spec["split_authorization_sha256"],
                entity_namespace="video_id",
                raw_key=b"1001",
                ledger=self.ledger(),
            ),
        )
        self.assert_code(
            "LMDB_QUERY_KEY_NONCANONICAL",
            lambda: self.authorize_key(b"01001", self.ledger()),
        )

    def test_minimal_id_map_lexer_accepts_only_two_typed_fields(self) -> None:
        for case in self.fixture["minimal_id_map"]["accepted"]:
            record = lex_minimal_id_mapping_jsonl_line(case["line"].encode("ascii"))
            self.assertEqual(record.desc_id, case["desc_id"])
            self.assertEqual(record.video_id, case["video_id"])
        for case in self.fixture["minimal_id_map"]["rejected"]:
            error = self.assert_code(
                case["code"],
                lambda case=case: lex_minimal_id_mapping_jsonl_line(
                    case["line"].encode("ascii")
                ),
            )
            if "sentinel_not_in_error" in case:
                self.assertNotIn(case["sentinel_not_in_error"], str(error))

    def test_id_map_trailing_comma_and_cross_entity_intersection_fail_closed(self) -> None:
        self.assert_code(
            "IDMAP_TRAILING_COMMA_FORBIDDEN",
            lambda: lex_minimal_id_mapping_jsonl_line(
                b'{"desc_id":1,"video_id":"synthetic-video",}'
            ),
        )
        desc_ids = EntityIdSet(namespace="desc_id", values=(1, 2))
        video_ids = EntityIdSet(
            namespace="video_id", values=("synthetic-video-01",)
        )
        self.assert_code(
            "CROSS_ENTITY_INTERSECTION_FORBIDDEN",
            lambda: intersect_entity_id_sets(desc_ids, video_ids),
        )


class PersistentCapabilityAndRaceTests(SyntheticFixtureCase):
    @staticmethod
    def semantic(value: Mapping[str, object]) -> str:
        return hashlib.sha256(
            json.dumps(
                value,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()

    def test_pinned_descriptor_survives_path_swap_and_rejects_prepersist_replay(self) -> None:
        with tempfile.TemporaryDirectory(prefix="c28f-pin-") as raw:
            root_path = Path(raw)
            source = root_path / "source.bin"
            replacement = root_path / "replacement.bin"
            source.write_bytes(b"original")
            replacement.write_bytes(b"replacement")
            root = FrozenCanonicalRoot.capture(
                "synthetic-read", str(root_path), ("READ_CONTENT", "READ_METADATA")
            )
            guard = DescriptorPathGuard(
                goal_id=self.fixture["goal_id"],
                authority_id=self.fixture["authority_id"],
                roots=(root,),
                forbidden_rules=(),
            )
            ledger = self.ledger()
            intent = PathIntent(
                request_id="pin-race",
                operation="READ_CONTENT",
                path=str(source),
                expected_kind="FILE",
                purpose="SYNTHETIC_PIN_RACE",
                goal_id=self.fixture["goal_id"],
                authority_id=self.fixture["authority_id"],
            )
            capability = guard.authorize_pinned_content(
                intent,
                ledger,
                parent_event_sha256="a" * 64,
            )
            self.assert_code(
                "CONTENT_OPEN_INTENT_RECEIPT_FIELDS_MISMATCH",
                lambda: open_pinned_content(capability, {}, ledger),
            )
            source.replace(root_path / "old-source.bin")
            replacement.replace(source)
            grant_sha = hashlib.sha256(
                json.dumps(
                    capability.grant.as_dict(),
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                ).encode("utf-8")
            ).hexdigest()
            receipt_base = {
                "schema_version": "c28f_persisted_content_open_intent_v1",
                "status": "COMMITTED_BEFORE_CONTENT_OPEN",
                "goal_id": self.fixture["goal_id"],
                "authority_id": self.fixture["authority_id"],
                "parent_event_sha256": "a" * 64,
                "request_id": "pin-race",
                "path_grant_sha256": grant_sha,
                "descriptor_chain_sha256": capability.grant.descriptor_chain_sha256,
                "pinned_capability_sha256": capability.capability_sha256,
                "open_intent_id": "pin-open-intent",
                "persistence_status": "FSYNCED_CAS_COMMITTED",
                "persistence_transaction_id": "pin-open-txn",
            }
            receipt = {
                **receipt_base,
                "receipt_sha256": self.semantic(receipt_base),
            }
            reader = open_pinned_content(capability, receipt, ledger)
            self.assertEqual(reader.read(64), b"original")
            reader.close()
            self.assert_code(
                "PINNED_CAPABILITY_REPLAY",
                lambda: open_pinned_content(capability, receipt, ledger),
            )

    def test_hardlink_is_rejected_before_content_open(self) -> None:
        with tempfile.TemporaryDirectory(prefix="c28f-hardlink-") as raw:
            root_path = Path(raw)
            source = root_path / "source.bin"
            alias = root_path / "alias.bin"
            source.write_bytes(b"synthetic")
            os.link(source, alias)
            root = FrozenCanonicalRoot.capture(
                "synthetic-read", str(root_path), ("READ_CONTENT", "READ_METADATA")
            )
            guard = DescriptorPathGuard(
                goal_id=self.fixture["goal_id"],
                authority_id=self.fixture["authority_id"],
                roots=(root,),
                forbidden_rules=(),
            )
            self.assert_code(
                "PATH_FILE_NLINK_NOT_ONE",
                lambda: guard.authorize(
                    PathIntent(
                        request_id="hardlink",
                        operation="READ_CONTENT",
                        path=str(source),
                        expected_kind="FILE",
                        purpose="SYNTHETIC_HARDLINK",
                        goal_id=self.fixture["goal_id"],
                        authority_id=self.fixture["authority_id"],
                    ),
                    self.ledger(),
                ),
            )

    def test_eval_token_reservation_evidence_never_allows_forward(self) -> None:
        registry = self.registry()
        request = replace(
            self.request(),
            request_id="REQ-F0A-MODEL-FORWARD",
            mode="MODEL_FORWARD",
        )
        before = self.budget()
        authorization, proposed = registry.authorize(
            request, before, self.ledger(), dry_run=False
        )
        self.assertFalse(authorization.model_forward_allowed)
        intent = prepare_eval_token_cas_intent(
            budget_before=before,
            budget_after=proposed,
            token_id="TOKEN-F0A",
            eval_id=request.eval_id,
            goal_id=self.fixture["goal_id"],
            authority_id=self.fixture["authority_id"],
            purpose=request.purpose,
        )
        receipt_base = {
            "schema_version": "c28f_eval_token_cas_receipt_v1",
            "status": "FSYNCED_CAS_COMMITTED",
            "goal_id": intent.goal_id,
            "authority_id": intent.authority_id,
            "eval_id": intent.eval_id,
            "token_id": intent.token_id,
            "cas_intent_sha256": intent.intent_sha256,
            "authorization_sha256": authorization.authorization_sha256,
            "budget_preimage_sha256": intent.expected_budget_preimage_sha256,
            "budget_postimage_sha256": intent.proposed_budget_postimage_sha256,
            "reservation_input_capability_sha256": "d" * 64,
            "transaction_id": "eval-token-txn",
            "state_sha256": "b" * 64,
            "event_sha256": "c" * 64,
        }
        receipt = {**receipt_base, "receipt_sha256": self.semantic(receipt_base)}
        evidence = validate_persisted_eval_token_receipt(
            receipt, intent=intent, authorization=authorization
        )
        self.assertEqual(evidence.token_id, "TOKEN-F0A")
        self.assertEqual(
            evidence.reservation_input_capability_sha256,
            "d" * 64,
        )
        self.assertFalse(hasattr(evidence, "model_forward_allowed"))

    def test_id_only_running_anchor_is_independent_from_eval_tokens(self) -> None:
        base = {
            "schema_version": (
                PROTECTED_ID_MAPPING_RUNNING_ANCHOR_SCHEMA_VERSION
            ),
            "goal_id": self.fixture["goal_id"],
            "attempt_id": "G0-A4-SYNTHETIC",
            "authority_permission_id": "AUTH_PROTECTED_ID_MAPPING_ONLY",
            "operation_id": "IDMAP-OP-SYNTHETIC",
            "transaction_id": "IDMAP-TXN-SYNTHETIC",
            "parent_state_sha256": "1" * 64,
            "parent_event_sha256": "2" * 64,
            "running_state_sha256": "3" * 64,
            "running_event_sha256": "4" * 64,
            "descriptor_identity_sha256": "5" * 64,
            "access_intent_sha256": "6" * 64,
            "operation_state": "RUNNING",
            "operations_used": 0,
            "eval_token_consumed": False,
            "authority_source_file_sha256": "7" * 64,
        }
        anchor = {
            **base,
            "anchor_binding_sha256": self.semantic(base),
        }
        validated = validate_protected_id_mapping_running_anchor(anchor)
        self.assertEqual(validated["operation_id"], "IDMAP-OP-SYNTHETIC")
        self.assertNotIn("token_id", validated)
        self.assertNotIn("eval_id", validated)
        self.assert_code(
            "PROTECTED_ID_MAPPING_RUNNING_ANCHOR_FIELDS_MISMATCH",
            lambda: validate_protected_id_mapping_running_anchor(
                {**anchor, "token_id": "F0_A"}
            ),
        )
        consumed = dict(anchor)
        consumed["eval_token_consumed"] = True
        consumed["anchor_binding_sha256"] = self.semantic(
            {
                key: value
                for key, value in consumed.items()
                if key != "anchor_binding_sha256"
            }
        )
        self.assert_code(
            "PROTECTED_ID_MAPPING_RUNNING_ANCHOR_STATE_INVALID",
            lambda: validate_protected_id_mapping_running_anchor(consumed),
        )

    def test_lmdb_manifest_batch_and_control_receipt_are_exactly_bound(self) -> None:
        helper = LmdbAndIdMappingTests()
        helper.fixture = self.fixture
        manifest = helper.lmdb_manifest()
        spec = self.fixture["lmdb_key_fixture"]
        ledger = self.ledger()
        batch = authorize_lmdb_key_batch(
            manifest=manifest,
            source_id=spec["source_id"],
            source_identity_sha256=spec["source_identity_sha256"],
            key_manifest_sha256=manifest.key_manifest_sha256,
            split_authorization_sha256=spec["split_authorization_sha256"],
            entity_namespace=spec["entity_namespace"],
            raw_keys=tuple(
                value.encode("ascii") for value in spec["allowed_synthetic_keys"]
            ),
            ledger=ledger,
        )
        intent = build_lmdb_batch_persistence_intent(
            manifest=manifest,
            authorized_batch=batch,
            goal_id=self.fixture["goal_id"],
            authority_id=self.fixture["authority_id"],
            parent_event_sha256="d" * 64,
        )
        self.assertEqual(
            intent["batch_authorization_sha256"], batch.authorization_sha256
        )
        self.assertEqual(
            intent["authorization_ledger_tail_event_sha256"],
            ledger.rows[-1]["event_sha256"],
        )
        self.assertFalse(intent["lmdb_transaction_open_allowed"])
        self.assert_code(
            "LMDB_CONCRETE_CONTROL_RECEIPT_REQUIRED",
            lambda: validate_persisted_lmdb_batch_receipt(
                {}, persistence_intent=intent
            ),
        )

    def test_lmdb_restart_rejects_recreated_mapping_evidence(self) -> None:
        helper = LmdbAndIdMappingTests()
        helper.fixture = self.fixture
        self.assert_code(
            "LMDB_CONCRETE_CONTROL_RECEIPT_REQUIRED",
            lambda: rehydrate_persisted_lmdb_batch_receipt(
                {
                    "shared_ledger_tail_sha256": "e" * 64,
                    "persisted_receipt_sha256": "f" * 64,
                },
                manifest=helper.lmdb_manifest(),
                running_capability={},
            ),
        )


if __name__ == "__main__":
    raise SystemExit("Run through the reviewed C28F stage test launcher only")
