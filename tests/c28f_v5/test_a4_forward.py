from __future__ import annotations

import ast
import hashlib
import inspect
import io
import json
import os
import sys
import tempfile
import textwrap
import types
import unittest
from pathlib import Path
from unittest import mock

from blueprint_e2e_v2.c28f_v5 import a4_forward as forward


def _sha(value):
    if not isinstance(value, bytes):
        value = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(value).hexdigest()


def _line_sha(values):
    return _sha("".join(f"{value}\n" for value in values).encode())


def _assigned_literal_string_set(function, assignment_name):
    tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        if not any(
            isinstance(target, ast.Name) and target.id == assignment_name
            for target in node.targets
        ):
            continue
        value = ast.literal_eval(node.value)
        if not isinstance(value, set) or not all(
            isinstance(item, str) for item in value
        ):
            raise AssertionError(assignment_name)
        return value
    raise AssertionError("missing literal set: " + assignment_name)


def _corpus():
    ids = tuple(f"video-{index:05d}" for index in range(17_435))
    durations = tuple(1.0 + (index % 300) / 10.0 for index in range(17_435))
    rows = [
        {"duration_sec": durations[index], "video_id": video_id}
        for index, video_id in enumerate(ids)
    ]
    return forward.FrozenCorpusIdentity(
        ids,
        durations,
        _line_sha(ids),
        _sha(rows),
        "a" * 64,
    )


def _queries(role="train_fit_route_dev", count=2048):
    keys = tuple(str(index + 1) for index in range(count))
    query_types = tuple(
        ("v", "t", "vt", "unknown")[index % 4]
        for index in range(count)
    )
    rows = [
        {"query_id": query_id, "query_type": query_type}
        for query_id, query_type in zip(keys, query_types)
    ]
    return forward.FrozenQueryIdentity(
        role,
        keys,
        query_types,
        _line_sha(keys),
        _line_sha(query_types),
        _sha(rows),
        "b" * 64,
    )


def _temporal():
    return forward.FrozenTemporalManifest.build(
        clip_len_sec=1.5,
        target_length=64,
        proposal_widths=(2, 4, 6, 8, 12, 16, 24, 32, 48, 64),
        proposal_stride_rule="max(1,width//4)",
        long_sequence_policy="mean_bin_floor_ceil_resample_all_mask_true",
        short_sequence_policy="right_zero_pad_prefix_mask_true",
        seconds_per_clip_policy="duration/64_if_raw_clips_gt_64_else_1.5",
        proposal_sampling_policy="linspace_round_half_to_even_64",
        max_proposals_per_video=64,
        metric_nms_iou_threshold=0.7,
        metric_nms_max_results=100,
    )


def _features():
    return forward.FrozenFeatureContract.build(
        query_dim=768,
        visual_dim=4352,
        subtitle_dim=768,
        query_value_max_bytes=forward.QUERY_VALUE_MAX_BYTES,
        visual_value_max_bytes=forward.VISUAL_VALUE_MAX_BYTES,
        subtitle_value_max_bytes=forward.SUBTITLE_VALUE_MAX_BYTES,
        query_sequence_max_length=forward.QUERY_SEQUENCE_MAX_LENGTH,
        video_sequence_max_length=forward.VIDEO_SEQUENCE_MAX_LENGTH,
        decoded_dtype="float32",
        encoded_dtype="float32",
        visual_decoder="msgpack_numpy_features_exact",
        query_subtitle_decoder="npz_allow_pickle_false_features_exact",
    )


def _gpu():
    return forward.FrozenGpu0Identity.build(
        physical_index=0,
        uuid="GPU-00000000-0000-0000-0000-000000000000",
        pci_bus_id="00000000:01:00.0",
        cuda_visible_devices="0",
        logical_device="cuda:0",
        gpu1_action_count=0,
    )


def _code_hashes():
    names = (
        "__init__.py",
        "a4_forward.py",
        "a4_security.py",
        "data/__init__.py",
        "data/temporal_grid.py",
        "models/__init__.py",
        "models/active_moment.py",
        "models/full_model.py",
        "models/partial_relevance.py",
        "models/proposal_generator.py",
        "models/query_encoder.py",
        "models/region_prior.py",
        "models/retrieval_guided_localizer.py",
        "models/retriever.py",
        "models/span_to_video_feedback.py",
        "models/vcmr_scorer.py",
        "models/video_encoder.py",
        "utils/__init__.py",
        "utils/tensor_ops.py",
    )
    package_root = Path(forward.__file__).parent.parent
    return tuple(
        (
            name,
            _sha(
                (
                    Path(forward.__file__).parent / name
                    if name in {"a4_forward.py", "a4_security.py"}
                    else package_root / name
                ).read_bytes()
            ),
        )
        for name in names
    )


def _manifest(corpus=None, queries=None, stage="G5_F0_B", **overrides):
    values = {
        "goal_id": "goal-a4",
        "authority_id": "authority-a4",
        "stage": stage,
        "purpose": (
            "F0_B_ROUTE_DEV_REPLICATION"
            if stage == "G5_F0_B"
            else "F0_A_TEACHER_FREE_CALIB_SELECT_FORENSICS"
        ),
        "query_identity": queries or _queries(),
        "corpus_identity": corpus or _corpus(),
        "checkpoint_sha256": "c" * 64,
        "model_contract": forward.FrozenModelContract(),
        "query_source_id": "query",
        "visual_source_id": "visual",
        "subtitle_source_id": "subtitle",
        "output_root_id": "output",
        "cache_root_id": "cache",
        "attempt_id": "attempt-a4",
        "temporal_manifest": _temporal(),
        "feature_contract": _features(),
        "gpu0_identity": _gpu(),
        "code_source_sha256": _code_hashes(),
        "metric_schema_sha256": "d" * 64,
        "evaluator_schema_sha256": "e" * 64,
        "environment_sha256": "f" * 64,
    }
    values.update(overrides)
    return forward.FrozenForwardManifest.build(**values)


def _frozen_file(file_id, path, approved_sha=None):
    item = path.stat()
    fd = os.open(path, os.O_RDONLY)
    try:
        mount_id = forward._fd_mount_id(fd)
    finally:
        os.close(fd)
    return forward.FrozenFile(
        file_id,
        str(path),
        int(item.st_dev),
        int(item.st_ino),
        mount_id,
        int(item.st_size),
        int(item.st_mtime_ns),
        approved_sha or _sha(path.read_bytes()),
    )


class _CommittedControl:
    """Adversarial full-looking duck object; production must still reject it."""

    def __init__(self):
        self.consume_count = 0
        self.running_sha = "1" * 64
        self.observations = []

    def _receipt(self, name, values=None):
        return _sha({"name": name, "values": values or {}})

    def assert_fsynced_control_chain(self, **values):
        return self._receipt("control", values)

    def consume_forward_capability_once(self, **values):
        self.consume_count += 1
        if self.consume_count != 1:
            raise RuntimeError("replay")
        return self._receipt("consume", values)

    def assert_running_state_event_committed(self):
        return self.running_sha

    def commit_lmdb_batch(self, **values):
        return self._receipt("batch", values)

    def record_observation(self, **values):
        self.observations.append(values)
        return self._receipt("observation", values)

    def committed_token_id(self):
        return "token-a4"

    def committed_eval_id(self):
        return "eval-a4"

    def committed_run_id(self):
        return "run-a4"

    def committed_token_snapshot_sha256(self):
        return "2" * 64


class _Fixture:
    def __init__(self, manifest=None):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.manifest = manifest or _manifest()
        output, cache, forbidden = root / "output", root / "cache", root / "forbidden"
        for path in (output, cache, forbidden):
            path.mkdir()
        checkpoint_path = root / "checkpoint.pt"
        checkpoint_path.write_bytes(b"synthetic checkpoint placeholder")
        sources = []
        for source_id, namespace in (("query", "QUERY"), ("visual", "VISUAL"), ("subtitle", "SUBTITLE")):
            source_root = root / source_id
            source_root.mkdir()
            data_path = source_root / "data.mdb"
            data_path.write_bytes(source_id.encode())
            sources.append(
                forward.FrozenLmdbSource(
                    source_id,
                    namespace,
                    forward.FrozenDirectory.capture(source_id, str(source_root)),
                    _frozen_file(source_id + "-data", data_path),
                )
            )
        checkpoint = _frozen_file("checkpoint", checkpoint_path, self.manifest.checkpoint_sha256)
        self.authority = forward.FrozenInputAuthority(
            manifest=self.manifest,
            checkpoint=checkpoint,
            lmdb_sources=sources,
            output_root=forward.FrozenDirectory.capture("output", str(output)),
            cache_root=forward.FrozenDirectory.capture("cache", str(cache)),
            forbidden_roots=(str(forbidden),),
        )
        self.control = _CommittedControl()

    def close(self):
        self.temp.cleanup()


class StaticBoundaryTests(unittest.TestCase):
    def test_dependencies_are_lazy_and_legacy_paths_are_absent(self):
        source = inspect.getsource(forward)
        tree = ast.parse(source)
        eager = []
        for node in tree.body:
            if isinstance(node, ast.Import):
                eager.extend(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                eager.append(node.module.split(".")[0])
        self.assertNotIn("torch", eager)
        self.assertNotIn("lmdb", eager)
        for forbidden in (
            "os.kill",
            "subprocess",
            "cuda:1",
            "device_count",
            "engine.evaluate",
            "c28e_late_interaction",
            "refresh_hard_negatives",
            "stable_temporal_nms",
            "ForwardTokenReceipt",
            "RESERVED",
        ):
            self.assertNotIn(forbidden, source)
    def test_full_model_uses_float32_sequence_masked_pools(self):
        source = inspect.getsource(forward.EncodedCorpusCache._encode_chunk)
        self.assertIn("model.video_encoder", source)
        self.assertIn('encoded["visual_pool"]', source)
        self.assertNotIn("encode_pooled_bank", source)
        self.assertNotIn(".half()", source)
        self.assertIn('"raw_mean_shortcut": False', source)

    def test_gpu0_is_physical_uuid_pci_and_cvd_bound_without_gpu1_probe(self):
        source = inspect.getsource(forward._verify_gpu0_physical_binding)
        self.assertIn('os.environ.get("CUDA_VISIBLE_DEVICES")', source)
        self.assertIn("nvmlDeviceGetHandleByIndex(0)", source)
        self.assertIn("nvmlDeviceGetUUID", source)
        self.assertIn("nvmlDeviceGetPciInfo", source)
        self.assertIn("torch_module.cuda.set_device(0)", source)
        self.assertNotIn("device_count", source)
        self.assertNotIn("GetHandleByIndex(1)", source)

    def test_checkpoint_loader_rehashes_same_fd_and_rechecks_path_mount(self):
        source = inspect.getsource(forward.load_frozen_model)
        gate_at = source.index("_assert_approved_business_modules_not_preloaded")
        torch_at = source.index("_lazy_torch()")
        preflight_at = source.index("_inspect_owned_torch_archive")
        load_at = source.index("_torch_load_weights_only(torch, handle)")
        rehash_at = source.index("_sha256_file_descriptor(fd)")
        business_import_at = source.index("_import_approved_business_module")
        self.assertLess(gate_at, torch_at)
        self.assertLess(preflight_at, load_at)
        self.assertLess(load_at, rehash_at)
        self.assertLess(rehash_at, business_import_at)
        self.assertIn("_open_directory_descriptor", source)
        self.assertIn("_fd_mount_id(fd)", source)
        self.assertIn("CHECKPOINT_PATH_REBOUND_AFTER_LOAD", source)
        self.assertIn("_FULL_MODEL_RUNTIME_MODULE_CLOSURE", source)
        self.assertIn("_FULL_MODEL_RUNTIME_PACKAGE_CLOSURE", source)
        self.assertIn("weights_only", inspect.getsource(forward._torch_load_weights_only))

    def test_preloaded_monkeypatched_full_model_is_rejected_before_import(self):
        fixture = _Fixture()
        module_name = "blueprint_e2e_v2.models.full_model"
        fake = types.ModuleType(module_name)
        fake.__file__ = str(
            Path(forward.__file__).parent.parent / "models" / "full_model.py"
        )
        fake.C28CFullModel = object()
        try:
            with mock.patch.dict(sys.modules, {module_name: fake}):
                with self.assertRaisesRegex(
                    forward.ForwardContractError,
                    "APPROVED_BUSINESS_MODULE_PRELOADED",
                ):
                    forward._assert_approved_business_modules_not_preloaded(
                        fixture.authority
                    )
        finally:
            fixture.close()

    def test_preloaded_business_package_is_rejected_before_leaf_import(self):
        fixture = _Fixture()
        module_name = "blueprint_e2e_v2.models"
        fake = types.ModuleType(module_name)
        try:
            with mock.patch.dict(sys.modules, {module_name: fake}):
                with self.assertRaisesRegex(
                    forward.ForwardContractError,
                    "APPROVED_BUSINESS_PACKAGE_PRELOAD_SET_INVALID",
                ):
                    forward._assert_approved_business_modules_not_preloaded(
                        fixture.authority
                    )
        finally:
            fixture.close()

    def test_package_pyc_origin_and_sys_modules_alias_are_rejected(self):
        fixture = _Fixture()
        package_name = "blueprint_e2e_v2.models"
        package_root = Path(forward.__file__).parent.parent / "models"
        pyc_path = package_root / "__pycache__" / "__init__.cpython-39.pyc"
        fake = types.ModuleType(package_name)
        loader = types.SimpleNamespace(
            name=package_name,
            path=str(pyc_path),
            get_filename=lambda _name: str(pyc_path),
        )
        spec = types.SimpleNamespace(
            name=package_name,
            parent=package_name,
            origin=str(pyc_path),
            loader=loader,
            has_location=True,
            submodule_search_locations=[str(package_root)],
        )
        fake.__package__ = package_name
        fake.__file__ = str(pyc_path)
        fake.__loader__ = loader
        fake.__spec__ = spec
        fake.__path__ = [str(package_root)]
        try:
            with mock.patch.dict(sys.modules, {package_name: fake}):
                with self.assertRaisesRegex(
                    forward.ForwardContractError,
                    "APPROVED_BUSINESS_PACKAGE_",
                ):
                    forward._verify_required_package_module_identity(
                        fixture.authority,
                        package_name,
                        fake,
                    )
            root_name = "blueprint_e2e_v2"
            root_module = sys.modules[root_name]
            with mock.patch.dict(
                sys.modules,
                {"blueprint_e2e_v2_alias": root_module},
            ):
                with self.assertRaisesRegex(
                    forward.ForwardContractError,
                    "PACKAGE_IDENTITY_OR_ALIAS_DRIFT",
                ):
                    forward._verify_required_package_module_identity(
                        fixture.authority,
                        root_name,
                        root_module,
                    )
        finally:
            fixture.close()

    def test_source_loader_is_rejected_when_package_cache_exists(self):
        package_name = "blueprint_e2e_v2.models"
        with tempfile.TemporaryDirectory() as temporary:
            package_root = Path(temporary) / "models"
            package_root.mkdir()
            initializer = package_root / "__init__.py"
            initializer.write_text('"""test package"""\n', encoding="utf-8")
            parent = forward.FrozenDirectory.capture(
                "test-models-package",
                str(package_root),
            )
            frozen = _frozen_file("test-models-init", initializer)
            authority = types.SimpleNamespace(
                _code_sources={"models/__init__.py": (parent, frozen)}
            )
            loader = forward.SourceFileLoader(
                package_name,
                str(initializer),
            )
            self.assertEqual(str(initializer), loader.get_filename(package_name))
            cache_root = package_root / "__pycache__"
            cache_root.mkdir()
            with mock.patch.object(
                sys,
                "dont_write_bytecode",
                True,
            ), mock.patch.object(
                sys,
                "pycache_prefix",
                None,
                create=True,
            ), self.assertRaisesRegex(
                forward.ForwardContractError,
                "APPROVED_BUSINESS_PACKAGE_BYTECODE_CACHE_PRESENT",
            ):
                forward._assert_required_package_bytecode_absent(
                    authority,
                    frozenset({package_name}),
                )
            cache_root.rmdir()
            direct_cache = package_root / "__init__.pyc"
            direct_cache.write_bytes(b"not executable bytecode")
            with mock.patch.object(
                sys,
                "dont_write_bytecode",
                True,
            ), mock.patch.object(
                sys,
                "pycache_prefix",
                None,
                create=True,
            ), self.assertRaisesRegex(
                forward.ForwardContractError,
                "APPROVED_BUSINESS_PACKAGE_BYTECODE_CACHE_PRESENT",
            ):
                forward._assert_required_package_bytecode_absent(
                    authority,
                    frozenset({package_name}),
                )

    def test_business_import_gate_checks_origin_loader_alias_hash_and_closure(self):
        identity_source = inspect.getsource(
            forward._verify_approved_business_module_identity
        )
        for marker in (
            'getattr(module, "__file__", None)',
            'getattr(spec, "origin", None)',
            "loader.get_filename(module_name)",
            "getattr(spec, \"loader\", None) is not loader",
            "aliases",
            "open_verified_readonly",
            "SourceFileLoader",
        ):
            self.assertIn(marker, identity_source)
        package_source = inspect.getsource(
            forward._verify_required_package_module_identity
        )
        for marker in (
            'getattr(module, "__path__", None)',
            'getattr(spec, "submodule_search_locations", None)',
            'frozen.path.endswith("/__init__.py")',
            "SourceFileLoader",
            "aliases",
            "open_verified_readonly",
        ):
            self.assertIn(marker, package_source)
        bytecode_source = inspect.getsource(
            forward._assert_required_package_bytecode_absent
        )
        for marker in (
            "sys.dont_write_bytecode is not True",
            'getattr(sys, "pycache_prefix", None) is not None',
            '"__pycache__" in entries',
            'entry.endswith(".pyc")',
            "_open_directory_descriptor",
        ):
            self.assertIn(marker, bytecode_source)
        import_source = inspect.getsource(forward._import_approved_business_module)
        self.assertIn("new_modules != set(expected_new_module_closure)", import_source)
        self.assertIn(
            "new_packages != set(expected_new_package_closure)",
            import_source,
        )
        self.assertIn("_assert_verified_business_module_record", import_source)
        self.assertIn("_assert_verified_package_module_record", import_source)
        self.assertGreaterEqual(
            import_source.count("_assert_required_package_bytecode_absent"),
            2,
        )
        self.assertEqual(
            {
                "__init__.py",
                "data/__init__.py",
                "models/__init__.py",
                "utils/__init__.py",
            },
            set(forward._REQUIRED_PACKAGE_MODULE_BY_SOURCE),
        )
        self.assertEqual(13, len(forward._APPROVED_BUSINESS_MODULE_BY_SOURCE))
        self.assertEqual(19, len(_code_hashes()))
        for forbidden_source in (
            "data/proposal_dataset.py",
            "data/subtitle_bank.py",
            "data/video_bank.py",
        ):
            self.assertNotIn(
                forbidden_source,
                forward._APPROVED_BUSINESS_MODULE_BY_SOURCE,
            )
        grid_source = inspect.getsource(forward.grid_spans_for_duration)
        self.assertIn("TEMPORAL_GRID_AUTHORITY_REQUIRED", grid_source)
        self.assertNotIn("import_module", grid_source)

    def test_cache_fingerprint_and_receipts_bind_full_lineage(self):
        fingerprint = inspect.getsource(
            forward.FrozenInputAuthority.encoded_cache_fingerprint.fget
        )
        for field in (
            "visual_source_identity_sha256",
            "subtitle_source_identity_sha256",
            "temporal_manifest_sha256",
            "feature_contract_sha256",
            "model_config",
            "code_source_sha256",
            "environment_sha256",
            "authority_binding_sha256",
            "attempt_id",
            "encoded_dtype",
        ):
            self.assertIn(field, fingerprint)
        receipt_source = inspect.getsource(forward.EncodedCorpusCache)
        for field in (
            "producer_capability_id",
            "producer_token_id",
            "producer_eval_id",
            "corpus_identity_sha256",
            "model_contract_sha256",
        ):
            self.assertIn(field, receipt_source)

    def test_code_sources_are_descriptor_hashed_and_runtime_reverified(self):
        capture_source = inspect.getsource(forward._capture_approved_code_source)
        self.assertIn("O_NOFOLLOW", capture_source)
        self.assertIn("_sha256_file_descriptor", capture_source)
        self.assertIn("st_nlink", capture_source)
        verify_source = inspect.getsource(
            forward.FrozenInputAuthority.verify_code_sources
        )
        self.assertIn("open_verified_readonly", verify_source)
        issue_source = inspect.getsource(forward.FrozenInputAuthority.issue_capability)
        self.assertIn("self.verify_code_sources()", issue_source)

    def test_atomic_chunk_recovery_cursor_heartbeat_and_capacity_are_present(self):
        source = inspect.getsource(forward.ForwardRunJournal)
        for marker in (
            "RESULT_CHUNK_ORPHAN_NOT_REPRODUCIBLE",
            "RESULT_CHUNK_HASH_DRIFT_ON_RESUME",
            "next_query_index",
            "commit_eval_cursor",
            "issue_eval_cursor_capability",
            "complete_eval_token",
            "capability.run_id",
            "CONTROL_COMMITTED_CHUNK_RECOVERY_AMBIGUOUS",
            "encoded_cache_receipt_sha256",
            "_persist_observation_snapshot",
            "CONTROL_COMPLETION_REQUIRES_COMPLETE_CURSOR",
            "2 * estimate",
            "1 << 30",
            "math.ceil(estimate * 0.10)",
        ):
            self.assertIn(marker, source)
        commit_source = inspect.getsource(forward.ForwardRunJournal.commit_query_chunk)
        self.assertLess(
            commit_source.index("self._commit_or_reconcile_cursor("),
            commit_source.index("self.receipt_path"),
        )
        self.assertIn("allow_in_memory_successor=True", commit_source)
        cursor_source = inspect.getsource(
            forward.ForwardRunJournal._commit_or_reconcile_cursor
        )
        self.assertNotIn("eval_id=", cursor_source)
        self.assertIn(
            "RESTART_REJECTS_LOCAL_UNCOMMITTED_CURSOR_SUCCESSOR",
            cursor_source,
        )
        binding_source = inspect.getsource(
            forward.ForwardRunJournal._cursor_bindings
        )
        self.assertIn("_read_bytes_no_follow", binding_source)
        self.assertIn("observed_artifact_sha256", binding_source)
        self.assertIn("engine_registry_consumption_sha256", binding_source)
        completion_source = inspect.getsource(
            forward.ForwardRunJournal._complete_control_token
        )
        self.assertIn("cursor_capability=cursor_capability", completion_source)
        self.assertIn(
            "self._validate_cursor_capability(cursor_capability, receipt)",
            completion_source,
        )
        self.assertLess(
            completion_source.index(
                "self._validate_cursor_capability(cursor_capability, receipt)"
            ),
            completion_source.index(
                "self.capability._control.complete_eval_token"
            ),
        )
        self.assertNotIn("completion=", completion_source)
        self.assertNotIn("eval_id=", completion_source)

    def test_lmdb_uses_dirfd_snapshot_and_persisted_batch_receipt(self):
        source = inspect.getsource(forward.GuardedLmdbReader)
        self.assertIn('"/proc/self/fd/%d"', source)
        self.assertIn("% self._source_guard_fd", source)
        self.assertIn("subdir=False", source)
        self.assertEqual(1, source.count(".begin("))
        self.assertNotIn("with self._environment.begin", source)
        authority_source = inspect.getsource(forward.FrozenInputAuthority.validate_full_key_manifest)
        self.assertIn("AuthorizedLmdbKeyBatch", authority_source)
        self.assertIn("build_lmdb_batch_persistence_intent", authority_source)
        self.assertIn("commit_lmdb_batch", authority_source)
        self.assertIn("validate_persisted_lmdb_batch_receipt", authority_source)
        self.assertIn(
            "issue_persisted_lmdb_batch_control_receipt", authority_source
        )
        self.assertIn("rehydrate_persisted_lmdb_batch_receipt", authority_source)
        self.assertIn(
            'str(error) != "LMDB persisted source receipt absent"',
            authority_source,
        )
        self.assertNotIn("commit_lmdb_batch_receipt", authority_source)
        self.assertNotIn("assert_lmdb_batch_receipt_persisted", authority_source)
        mmap_source = inspect.getsource(forward._assert_lmdb_mapping_matches_guard)
        self.assertIn('os.open("/proc/self/maps"', mmap_source)
        self.assertIn("st_ino", mmap_source)
        self.assertIsInstance(
            forward.FrozenInputAuthority.lmdb_key_ledger_rows, property
        )
        manifest_source = inspect.getsource(
            forward.FrozenInputAuthority._build_shared_key_manifests
        )
        self.assertIn("lmdb_key_manifest_semantic_sha256", manifest_source)
        self.assertIn("manifest_semantic_sha256=", manifest_source)

    def test_capability_trust_chain_binds_goal_authority_split_and_one_consume(self):
        source = inspect.getsource(forward.FrozenInputAuthority.issue_capability)
        for marker in (
            "_require_exact_forward_control",
            "assert_fsynced_control_chain",
            "goal_id=manifest.goal_id",
            "authority_id=manifest.authority_id",
            "split_manifest_sha256=manifest.query_identity.split_manifest_sha256",
            "corpus_manifest_sha256=manifest.corpus_identity.identity_sha256",
            "consume_forward_capability_once",
            "assert_running_state_event_committed",
            "issue_eval_running_capability",
            "require_concrete_running_eval_control",
            "reservation_input_capability_sha256",
        ):
            self.assertIn(marker, source)
        self.assertLess(
            source.index("control.consume_forward_capability_once"),
            source.index("control.assert_running_state_event_committed()"),
        )
        control_module = forward._a4_control_module()
        running_validator = inspect.getsource(
            control_module.validate_eval_running_capability
        )
        self.assertIn("reservation_input_capability_sha256", running_validator)
        issue_source = inspect.getsource(control_module.issue_f0_a_generation)
        self.assertIn("reservation_input", issue_source)
        self.assertIn("reservation_input_capability_sha256", issue_source)
        self.assertNotIn("cas_receipt:", issue_source)

    def test_authority_binding_has_one_shared_material_builder(self):
        builder_source = inspect.getsource(
            forward.FrozenInputAuthority.authority_binding_material
        )
        property_source = inspect.getsource(
            forward.FrozenInputAuthority.authority_binding_sha256.fget
        )
        for marker in (
            '"checkpoint_parent"',
            '"code_sources"',
            '"parent": parent.as_dict()',
            '"file": frozen.as_dict()',
        ):
            self.assertIn(marker, builder_source)
        self.assertIn("sorted(", builder_source)
        self.assertIn("self._code_sources.items()", builder_source)
        self.assertIn("self.authority_binding_material()", property_source)

    def test_controller_issued_authority_is_required_before_any_source_verify(self):
        issue_source = inspect.getsource(
            forward.FrozenInputAuthority.issue_capability
        )
        controller_at = issue_source.index("_validate_controller_issued_authority")
        source_verify_at = issue_source.index("self.verify_code_sources()")
        self.assertLess(controller_at, source_verify_at)
        validator_source = inspect.getsource(
            forward._validate_controller_issued_authority
        )
        for marker in (
            "validate_controller_forward_authority",
            '"query_types_sha256"',
            '"query_rows_sha256"',
            '"durations_sha256"',
            '"query_source_descriptor_identity_sha256"',
            '"query_source_selected_projection_sha256"',
            '"query_source_index_binding_sha256"',
            '"query_source_index_bootstrap_receipt_sha256"',
            '"query_source_file_sha256"',
            '"query_identity_source_access_receipt_sha256"',
            '"video_meta_source_file_sha256"',
            '"reservation_input_capability_sha256"',
            '"controller_authority_binding_sha256"',
        ):
            self.assertIn(marker, validator_source)

    def test_handles_results_paths_and_torch_loads_are_sealed(self):
        model_source = inspect.getsource(forward.LoadedModelHandle)
        corpus_source = inspect.getsource(forward.EncodedCorpusHandle)
        result_source = inspect.getsource(forward.ForwardQueryResult)
        self.assertIn('"_model"', model_source)
        self.assertNotIn('"model"', model_source)
        self.assertIn('"_payload"', corpus_source)
        self.assertNotIn('"payload"', corpus_source)
        self.assertIn("MappingProxyType", corpus_source)
        self.assertIn("_tensor_runtime_seal", corpus_source)
        self.assertIn("_RESULT_FACTORY", result_source)
        self.assertIn("_forensics_bytes", result_source)
        self.assertIn("clone()", result_source)
        engine_source = inspect.getsource(forward.FrozenForwardEngine)
        self.assertIn("_issued_result_registry", engine_source)
        self.assertIn("_consume_issued_results_once", engine_source)
        journal_source = inspect.getsource(
            forward.ForwardRunJournal.commit_query_chunk
        )
        self.assertIn("EXACT_BOUND_FORWARD_ENGINE_REQUIRED", journal_source)
        self.assertIn("engine._consume_issued_results_once", journal_source)
        path_source = inspect.getsource(forward._atomic_write_bytes)
        self.assertIn("dir_fd=directory_fd", path_source)
        self.assertIn("O_NOFOLLOW", path_source)
        self.assertIn("st_nlink", path_source)
        torch_source = inspect.getsource(forward._safe_load_owned_torch_artifact)
        self.assertLess(
            torch_source.index("_inspect_owned_torch_archive"),
            torch_source.index("_torch_load_weights_only"),
        )

    def test_feature_decoders_preflight_value_decompression_and_sequence_bounds(self):
        npz_source = inspect.getsource(forward._decode_npz_features)
        self.assertIn("ZipFile", npz_source)
        self.assertIn("maximum_npy_bytes", npz_source)
        self.assertIn("allow_pickle=False", npz_source)
        for diagnostic in (
            "NPZ_MEMBER_COUNT_CONTRACT_VIOLATION",
            "NPZ_MEMBER_NAME_CONTRACT_VIOLATION",
            "NPZ_MEMBER_DIRECTORY_CONTRACT_VIOLATION",
            "NPZ_MEMBER_EMPTY_CONTRACT_VIOLATION",
            "NPZ_MEMBER_SIZE_CONTRACT_VIOLATION",
            "NPZ_MEMBER_ENCRYPTION_CONTRACT_VIOLATION",
        ):
            self.assertIn(diagnostic, npz_source)
        validate_source = inspect.getsource(forward._validate_feature_array)
        self.assertIn("max_sequence_length", validate_source)
        self.assertIn("DECOMPRESSED_FEATURE_SIZE_LIMIT", validate_source)
        self.assertIn("FEATURE_NONFINITE", validate_source)

    def test_full_scale_handle_and_raw_output_shape_mask_contracts_are_static(self):
        handle_source = inspect.getsource(forward.EncodedCorpusHandle)
        for marker in (
            "EXPECTED_CORPUS_VIDEO_COUNT, 384",
            "EXPECTED_CORPUS_VIDEO_COUNT, TARGET_LENGTH, 384",
            "ENCODED_HANDLE_DTYPE_NOT_FLOAT32",
            "ENCODED_HANDLE_MASK_DTYPE_NOT_BOOL",
            "ENCODED_HANDLE_JOINT_MASK_MISMATCH",
        ):
            self.assertIn(marker, handle_source)
        result_source = inspect.getsource(forward._validate_forward_result_payload)
        self.assertIn("POOLED_TOPK", result_source)
        self.assertIn("LATE_TOPK", result_source)
        self.assertIn("FULL_MODEL_CANDIDATES", result_source)
        self.assertIn("MAX_PROPOSALS_PER_VIDEO", result_source)
        self.assertIn("RAW_P_FULL_NO_NMS", result_source)
        self.assertNotIn("nms(", result_source.lower())


class ManifestContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.corpus = _corpus()

    def test_scale_config_temporal_and_metric_nms_are_exact(self):
        manifest = _manifest(self.corpus)
        self.assertEqual((1000, 200, 200), (manifest.broad_topk, manifest.late_topk, manifest.full_model_candidates))
        self.assertEqual((8, 0.07, 0.25, 0.35, 0.65), (
            manifest.model_contract.late_soft_topk,
            manifest.model_contract.late_temperature,
            manifest.model_contract.token_maxsim_weight,
            manifest.model_contract.pooled_score_weight,
            manifest.model_contract.late_score_weight,
        ))
        self.assertEqual(100, manifest.temporal_manifest.metric_nms_max_results)
        self.assertEqual(manifest.manifest_sha256, manifest.declared_semantic_sha256)

    def test_query_identity_freezes_ordered_types_and_row_hashes(self):
        identity = _queries(count=2_048)
        self.assertEqual(2_048, len(identity.query_types))
        self.assertEqual(_line_sha(identity.query_types), identity.query_types_sha256)
        rows = [
            {"query_id": query_id, "query_type": query_type}
            for query_id, query_type in zip(
                identity.query_keys,
                identity.query_types,
            )
        ]
        self.assertEqual(_sha(rows), identity.query_rows_sha256)
        mutated_types = list(identity.query_types)
        mutated_types[0] = "unknown"
        with self.assertRaisesRegex(
            forward.ForwardContractError,
            "QUERY_TYPES_NOT_EXACT_FINITE_SEQUENCE|QUERY_TYPE_HASH_MISMATCH",
        ):
            forward.FrozenQueryIdentity(
                identity.role,
                identity.query_keys,
                tuple(mutated_types),
                identity.query_keys_sha256,
                identity.query_types_sha256,
                identity.query_rows_sha256,
                identity.split_manifest_sha256,
            )
        manifest = _manifest(self.corpus, identity)
        self.assertIn(
            identity.query_rows_sha256,
            json.dumps(manifest.as_dict(), sort_keys=True),
        )

    def test_teacher_gt_updates_and_shrunk_scale_fail_closed(self):
        for field_name in (
            "teacher_candidate_count",
            "teacher_forward_count",
            "gt_support_count",
            "gt_append_count",
            "optimizer_update_count",
            "posthoc_temperature_refit_count",
        ):
            with self.subTest(field_name=field_name), self.assertRaises(forward.ForwardContractError):
                _manifest(self.corpus, **{field_name: 1})
        with self.assertRaises(forward.ForwardContractError):
            _manifest(self.corpus, broad_topk=999)

    def test_late_interaction_configuration_is_exact_not_merely_nonnegative(self):
        mutations = (
            {"late_soft_topk": 7},
            {"late_temperature": 0.0700001},
            {"token_maxsim_weight": 0.0},
            {"pooled_score_weight": 0.3501},
            {"late_score_weight": 0.6499},
        )
        for values in mutations:
            with self.subTest(values=values), self.assertRaisesRegex(
                forward.ForwardContractError, "MODEL_LATE_CONFIG_DRIFT"
            ):
                forward.FrozenModelContract(**values)

    def test_g3_and_g5_are_distinct_manifests(self):
        g3 = _manifest(
            self.corpus,
            _queries("calib_select", 8677),
            stage="G3_F0_A",
        )
        g5 = _manifest(self.corpus)
        self.assertNotEqual(g3.manifest_sha256, g5.manifest_sha256)

    def test_package_initializer_hash_is_mandatory_in_manifest(self):
        missing_models_init = tuple(
            item
            for item in _code_hashes()
            if item[0] != "models/__init__.py"
        )
        with self.assertRaisesRegex(
            forward.ForwardContractError,
            "CODE_SOURCE_HASH_SET_INCOMPLETE",
        ):
            _manifest(
                self.corpus,
                code_source_sha256=missing_models_init,
            )


class ControlAndJournalTests(unittest.TestCase):
    def setUp(self):
        self.fixture = _Fixture()

    def tearDown(self):
        self.fixture.close()

    def test_self_signed_mapping_and_reserved_legacy_token_paths_do_not_exist(self):
        self.assertFalse(hasattr(forward, "ForwardTokenReceipt"))
        self.assertFalse(hasattr(forward, "CommittedForwardControl"))
        with self.assertRaisesRegex(
            forward.ForwardContractError, "EXACT_A4_FORWARD_CONTROL_REQUIRED"
        ):
            self.fixture.authority.issue_capability({})

    def test_full_looking_duck_control_is_rejected_before_consumption(self):
        with self.assertRaisesRegex(
            forward.ForwardContractError, "EXACT_A4_FORWARD_CONTROL_REQUIRED"
        ):
            self.fixture.authority.issue_capability(self.fixture.control)
        self.assertEqual(0, self.fixture.control.consume_count)

    def test_caller_constructed_authority_is_rejected_before_source_preopen(self):
        control_module = forward._a4_control_module()
        with self.assertRaisesRegex(
            control_module.A4ControlError,
            "active bootstrap lease required",
        ):
            control_module.A4EvalForwardControlAdapter(object())

    def test_active_authority_descriptor_projection_roundtrips_exactly(self):
        projection = dict(
            forward.build_active_forward_authority_rehydrate_projection(
                self.fixture.authority,
                forward_authority_handle_sha256="7" * 64,
            )
        )
        rebuilt = forward.rehydrate_active_forward_authority(projection)
        self.assertEqual(
            rebuilt.authority_binding_material(),
            self.fixture.authority.authority_binding_material(),
        )
        self.assertEqual(
            rebuilt.authority_binding_sha256,
            self.fixture.authority.authority_binding_sha256,
        )
        self.assertEqual(
            rebuilt.manifest.manifest_sha256,
            self.fixture.authority.manifest.manifest_sha256,
        )
        self.assertFalse(projection["source_content_present"])
        self.assertFalse(projection["active_content_opened"])
        self.assertFalse(projection["eval_token_reissued"])

    def test_active_authority_projection_rejects_descriptor_drift(self):
        projection = json.loads(
            forward._canonical_json_bytes(
                dict(
                    forward.build_active_forward_authority_rehydrate_projection(
                        self.fixture.authority,
                        forward_authority_handle_sha256="8" * 64,
                    )
                )
            ).decode("utf-8")
        )
        projection["authority_binding_material"]["checkpoint"][
            "inode"
        ] += 1
        projection["projection_binding_sha256"] = forward._semantic_sha256(
            projection,
            ("projection_binding_sha256",),
        )
        with self.assertRaisesRegex(
            forward.ForwardContractError,
            "DESCRIPTOR|BINDING_DRIFT",
        ):
            forward.rehydrate_active_forward_authority(projection)

    def test_opaque_model_and_corpus_handles_cannot_be_forged(self):
        with self.assertRaises(forward.ForwardContractError):
            forward.LoadedModelHandle(object())
        with self.assertRaises(forward.ForwardContractError):
            forward.EncodedCorpusHandle(object(), torch_module=object())
        with self.assertRaises(forward.ForwardContractError):
            forward.ForwardQueryResult(
                object(),
                torch_module=object(),
                issuer_nonce=object(),
                capability_id="a" * 64,
            )
        with self.assertRaises(forward.ForwardContractError):
            forward.CompletedForwardRunArtifactHandle(object())
        with self.assertRaises(forward.ForwardContractError):
            forward.CompletedForwardResultVerificationAuthority(object())
        with self.assertRaises(forward.ForwardContractError):
            forward.CompletedForwardResultCapability(object())
        forged_completed = object.__new__(
            forward.CompletedForwardRunArtifactHandle
        )
        object.__setattr__(
            forged_completed,
            "_factory",
            forward._COMPLETED_FORWARD_ARTIFACT_HANDLE_FACTORY,
        )
        with self.assertRaisesRegex(
            forward.ForwardContractError,
            "NOT_REGISTERED",
        ):
            forward.validate_completed_forward_artifact_handle(
                forged_completed
            )

    @staticmethod
    def _bound_artifact_root_fixture(root):
        output = root / "output"
        run = output / "G5_F0_B" / "attempt-a4" / "run-a4"
        chunks = run / "query_chunks"
        chunks.mkdir(parents=True)
        output_identity = forward.FrozenDirectory.capture("output", str(output))
        run_path = str(run)
        chunk_path = str(chunks)
        run_identity = forward.FrozenDirectory.capture(
            "child-" + _sha(run_path.encode())[:32],
            run_path,
        )
        chunk_identity = forward.FrozenDirectory.capture(
            "child-" + _sha(chunk_path.encode())[:32],
            chunk_path,
        )
        authority = object.__new__(forward.FrozenInputAuthority)
        authority.output_root = output_identity
        authority.manifest = types.SimpleNamespace(attempt_id="attempt-a4")
        capability = object.__new__(forward.InputCapability)
        for field_name, value in {
            "_authority": authority,
            "stage": "G5_F0_B",
            "run_id": "run-a4",
            "_run_receipt_parent": run_identity,
            "_run_chunk_parent": chunk_identity,
            "_run_receipt_path": str(run / "forward_run_receipt.json"),
        }.items():
            object.__setattr__(capability, field_name, value)
        return authority, capability, run_identity, chunk_identity, run, chunks

    def test_completed_artifact_roots_reject_equal_external_replacement_handle(self):
        with tempfile.TemporaryDirectory() as temporary:
            values = self._bound_artifact_root_fixture(Path(temporary))
            authority, capability, run_identity, chunk_identity, _run, _chunks = values
            forward._validate_bound_completed_forward_roots(
                authority=authority,
                capability=capability,
                run_root=run_identity,
                chunk_root=chunk_identity,
            )
            equal_but_external = forward.FrozenDirectory(
                **dict(run_identity.as_dict())
            )
            with self.assertRaisesRegex(
                forward.ForwardContractError,
                "COMPLETED_ARTIFACT_ROOT_HANDLE_MISMATCH",
            ):
                forward._validate_bound_completed_forward_roots(
                    authority=authority,
                    capability=capability,
                    run_root=equal_but_external,
                    chunk_root=chunk_identity,
                )

    def test_completed_artifact_roots_reject_rename_and_symlink_replacement(self):
        with tempfile.TemporaryDirectory() as temporary:
            values = self._bound_artifact_root_fixture(Path(temporary))
            authority, capability, run_identity, chunk_identity, run, chunks = values
            archived = run.with_name("run-a4-archived")
            run.rename(archived)
            (run / "query_chunks").mkdir(parents=True)
            with self.assertRaises(forward.ForwardContractError):
                forward._validate_bound_completed_forward_roots(
                    authority=authority,
                    capability=capability,
                    run_root=run_identity,
                    chunk_root=chunk_identity,
                )
        with tempfile.TemporaryDirectory() as temporary:
            values = self._bound_artifact_root_fixture(Path(temporary))
            authority, capability, run_identity, chunk_identity, _run, chunks = values
            saved = chunks.with_name("query_chunks-saved")
            chunks.rename(saved)
            chunks.symlink_to(saved.name, target_is_directory=True)
            with self.assertRaises((forward.ForwardContractError, OSError)):
                forward._validate_bound_completed_forward_roots(
                    authority=authority,
                    capability=capability,
                    run_root=run_identity,
                    chunk_root=chunk_identity,
                )

    def test_completed_chunk_range_rejects_hash_identical_out_of_bounds_index(self):
        same_hash = "a" * 64
        chunk = {
            "start_index": 0,
            "end_index": 11,
            "artifact_name": "queries_00000_00011.pt",
            "artifact_sha256": same_hash,
            "semantic_sha256": same_hash,
        }
        with self.assertRaisesRegex(
            forward.ForwardContractError,
            "COMPLETED_FORWARD_CHUNK_RANGE_OR_PATH_INVALID",
        ):
            forward._validate_completed_forward_chunk_range(
                chunk,
                sequence=0,
                expected_start=0,
                query_count=10,
            )

    def test_completed_artifact_validator_is_read_only_safe_load_and_gated_transfer(self):
        source = inspect.getsource(
            forward.validate_completed_forward_run_artifacts
        )
        for marker in (
            'receipt.get("status") != "COMPLETED"',
            "_read_canonical_json",
            "_read_bytes_no_follow",
            "_safe_load_owned_torch_artifact",
            "_validate_forward_result_payload",
            "_forward_result_payload_digest",
            "_validate_completed_forward_chunk_range",
            "OPAQUE_EXACT_CONTROLLER_DERIVATION_CONSUMER_REQUIRED",
        ):
            self.assertIn(marker, source)
        self.assertNotIn("assert_running_committed", source)
        self.assertLess(
            source.index("_issue_original_completed_result_verifier"),
            source.index("_lazy_torch"),
        )
        self.assertIn(
            "persist_completed_forward_rehydrate_material",
            source,
        )
        self.assertLess(
            source.index('"validation_receipt_sha256"'),
            source.index("persist_completed_forward_rehydrate_material"),
        )
        self.assertLess(
            source.index("persist_completed_forward_rehydrate_material"),
            source.index("_register_completed_result_verifier"),
        )
        self.assertIn(
            "_rollback_completed_result_verifier_registration",
            source,
        )
        issue_source = inspect.getsource(
            forward._issue_original_completed_result_verifier
        )
        self.assertIn(
            "completed_capability._assert_completed_lifecycle()",
            issue_source,
        )
        self.assertNotIn(
            "_register_completed_result_verifier",
            issue_source,
        )
        consume_source = inspect.getsource(
            getattr(
                forward.CompletedForwardRunArtifactHandle,
                "transfer_next_validated_chunk_to_controller_once",
            )
        )
        for marker in (
            "validate_forward_derivation_consumer_capability",
            "prepare_forward_derivation_range",
            "validate_forward_derivation_range_claim_capability",
            "validate_forward_derivation_delivery_owner",
            "rehydrate_forward_derivation_postimage_if_present",
            "_issue_validated_forward_derivation_range_transfer",
            "derive_and_fsync_forward_range",
            "validate_forward_derived_range_postimage_capability",
            "commit_forward_derivation_range",
            "previous_cursor_index",
            "next_cursor_index",
            "all_ranges_consumed",
            "_next_chunk_index",
            "forward_derivation_resume_snapshot",
            "consumed_range_receipt_chain_sha256",
        ):
            self.assertIn(marker, consume_source)
        self.assertNotIn("consume_forward_derivation_range", consume_source)
        self.assertNotIn("return MappingProxyType(dict(payload))", consume_source)
        self.assertLess(
            consume_source.index("prepare_forward_derivation_range"),
            consume_source.index(
                "rehydrate_forward_derivation_postimage_if_present"
            ),
        )
        self.assertLess(
            consume_source.index(
                "rehydrate_forward_derivation_postimage_if_present"
            ),
            consume_source.index("_safe_load_owned_torch_artifact"),
        )
        self.assertLess(
            consume_source.index("derive_and_fsync_forward_range"),
            consume_source.index("commit_forward_derivation_range"),
        )
        self.assertLess(
            consume_source.index("commit_forward_derivation_range"),
            consume_source.rindex(
                'object.__setattr__(self, "_next_chunk_index"'
            ),
        )
        issued_validator_source = inspect.getsource(
            forward.validate_completed_forward_artifact_handle
        )
        self.assertIn(
            "_COMPLETED_FORWARD_ARTIFACT_HANDLE_REGISTRY",
            issued_validator_source,
        )
        self.assertIn('record.get("object") is not value', issued_validator_source)
        handle_source = inspect.getsource(
            forward.CompletedForwardRunArtifactHandle
        )
        self.assertNotIn('"_payloads"', handle_source)
        self.assertNotIn('"_payload_semantic_sha256s"', handle_source)
        self.assertNotIn("payloads.append", source)
        self.assertIn("del payload", source)
        self.assertIn(
            '"maximum_simultaneously_loaded_chunk_count": 1',
            source,
        )
        self.assertIn(
            "HANDLE_STORES_NO_RAW_PAYLOAD_STREAM_REVALIDATES_ONE_CHUNK",
            source,
        )
        self.assertIn('"encoded_cache_binding"', source)
        self.assertNotIn('"path": str(cache', source)

    def test_completed_result_verifier_has_no_active_authority_surface(self):
        authority_slots = set(
            forward.CompletedForwardResultVerificationAuthority.__slots__
        )
        for forbidden in (
            "checkpoint",
            "_checkpoint_parent",
            "sources",
            "cache_root",
            "_loaded_model_handle",
            "_encoded_corpus_handle",
            "_gpu0_runtime_lease",
        ):
            self.assertNotIn(forbidden, authority_slots)
        active_assert_source = inspect.getsource(
            forward.FrozenInputAuthority.assert_capability
        )
        self.assertNotIn(
            "CompletedForwardResultCapability",
            active_assert_source,
        )
        projection_source = inspect.getsource(
            forward._build_completed_forward_authority_rehydrate_projection
        )
        self.assertIn('"active_authority_type_reconstructed": False', projection_source)
        self.assertIn('"checkpoint_descriptor_present": False', projection_source)
        self.assertIn('"lmdb_descriptor_present": False', projection_source)
        self.assertIn('"cache_root_descriptor_present": False', projection_source)
        self.assertIn('"active_io_allowed": False', projection_source)
        self.assertNotIn("authority_binding_material()", projection_source)
        rehydrate_source = inspect.getsource(
            forward._rehydrate_completed_forward_authority_from_projection
        )
        self.assertIn(
            "CompletedForwardResultVerificationAuthority",
            rehydrate_source,
        )
        self.assertNotIn("FrozenInputAuthority(", rehydrate_source)
        self.assertNotIn("FrozenLmdbSource(", rehydrate_source)
        self.assertNotIn("FrozenFile(", rehydrate_source)
        completed_verify_source = inspect.getsource(
            getattr(
                forward.CompletedForwardResultVerificationAuthority,
                "verify_code_sources",
            )
        )
        self.assertIn("self.manifest.code_source_sha256", completed_verify_source)
        self.assertIn("_capture_approved_code_source", completed_verify_source)

    def test_completed_result_verifier_rejects_model_and_feature_reopen_preimage(self):
        authority = object.__new__(
            forward.CompletedForwardResultVerificationAuthority
        )
        capability = object.__new__(forward.CompletedForwardResultCapability)
        io_ledger = {"count": 0}

        def forbidden_torch_import():
            io_ledger["count"] += 1
            raise AssertionError("torch import reached")

        def forbidden_source_io(*_args, **_kwargs):
            io_ledger["count"] += 1
            raise AssertionError("feature/cache I/O reached")

        with mock.patch.object(
            forward,
            "_lazy_torch",
            side_effect=forbidden_torch_import,
        ), mock.patch.object(
            forward,
            "_lazy_lmdb",
            side_effect=forbidden_source_io,
        ), mock.patch.object(
            forward,
            "_safe_child_directory",
            side_effect=forbidden_source_io,
        ):
            with self.assertRaisesRegex(
                forward.ForwardContractError,
                "COMPLETED_RESULT_AUTHORITY_ACTIVE_FORWARD_FORBIDDEN",
            ):
                forward.load_frozen_model(authority, capability)
            with self.assertRaisesRegex(
                forward.ForwardContractError,
                "COMPLETED_RESULT_AUTHORITY_ACTIVE_FORWARD_FORBIDDEN",
            ):
                forward.FrozenForwardEngine(
                    authority,
                    capability,
                    object(),
                    object(),
                )
            with self.assertRaisesRegex(
                forward.ForwardContractError,
                "COMPLETED_RESULT_AUTHORITY_ACTIVE_FORWARD_FORBIDDEN",
            ):
                forward.GuardedLmdbReader(
                    authority,
                    capability,
                    "query_lmdb",
                )
            with self.assertRaisesRegex(
                forward.ForwardContractError,
                "COMPLETED_RESULT_AUTHORITY_ACTIVE_FORWARD_FORBIDDEN",
            ):
                forward.EncodedCorpusCache(authority, capability)
        self.assertEqual(0, io_ledger["count"])
        with self.assertRaisesRegex(
            forward.ForwardContractError,
            "FEATURE_SOURCE_ACCESS_FORBIDDEN",
        ):
            authority.allowed_key_digests("query_lmdb")
        with self.assertRaisesRegex(
            forward.ForwardContractError,
            "TOKEN_RESERVATION_FORBIDDEN",
        ):
            authority.issue_capability(object())
        with self.assertRaisesRegex(
            forward.ForwardContractError,
            "ACTIVE_FORWARD_FORBIDDEN",
        ):
            capability.assert_running_committed()

    def test_completed_handle_cross_process_rehydrate_is_controller_only_no_index_scan(self):
        source = inspect.getsource(
            forward.rehydrate_completed_forward_run_artifact_handle
        )
        for marker in (
            "_validate_completed_forward_rehydrate_binding",
            "consume_completed_forward_rehydrate_material_once",
            "completed_result_verification_projection",
            "REHYDRATED_COMPLETED_SAFE_INDEX",
            "_validate_rehydrated_completed_forward_validation_receipt",
            "CompletedForwardResultCapability",
        ):
            self.assertIn(marker, source)
        for forbidden in (
            "_lazy_torch",
            "_safe_load_owned_torch_artifact",
            "FrozenInputAuthority(",
            "issue_capability(",
            "load_frozen_model(",
            "artifact_path =",
        ):
            self.assertNotIn(forbidden, source)
        self.assertLess(
            source.index(
                "_validate_rehydrated_completed_forward_validation_receipt"
            ),
            source.index("_register_completed_result_verifier"),
        )
        self.assertIn(
            "_rollback_completed_result_verifier_registration",
            source,
        )
        validation_source = inspect.getsource(
            forward._validate_rehydrated_completed_forward_validation_receipt
        )
        self.assertNotIn("os.listdir", validation_source)
        self.assertNotIn("_safe_load_owned_torch_artifact", validation_source)
        self.assertNotIn("_lazy_torch", validation_source)

    def test_completed_rehydrate_and_owner_receipt_exact_field_sets_are_frozen(self):
        exact36 = {
            "schema_version",
            "status",
            "goal_id",
            "attempt_id",
            "authority_id",
            "run_id",
            "token_id",
            "eval_id",
            "stage",
            "purpose",
            "capability_id",
            "forward_authority_handle_sha256",
            "forward_manifest_sha256",
            "input_identity_sha256",
            "authority_binding_sha256",
            "reservation_input_capability_sha256",
            "controller_authority_binding_sha256",
            "completed_transaction_id",
            "completed_state_sha256",
            "completed_event_sha256",
            "execution_binding_sha256",
            "forward_run_receipt_sha256",
            "completed_forward_validation_receipt_sha256",
            "authority_rehydrate_projection_sha256",
            "input_capability_material_sha256",
            "run_root_identity_sha256",
            "chunk_root_identity_sha256",
            "rehydration_generation",
            "rehydration_source",
            "token_reissue_count",
            "model_forward_reexecution_count",
            "query_source_index_reaccess_count",
            "raw_chunk_index_namespace_rescan_count",
            "raw_tensor_materialized",
            "old_object_identity_required",
            "rehydrate_binding_sha256",
        }
        exact7 = {
            "completed_result_verification_projection",
            "input_capability_material",
            "run_root",
            "chunk_root",
            "completed_forward_validation_receipt",
            "authority_rehydrate_projection_sha256",
            "material_binding_sha256",
        }
        exact33 = {
            "schema_version",
            "status",
            "goal_id",
            "attempt_id",
            "run_id",
            "token_id",
            "eval_id",
            "completed_transaction_id",
            "consumer_capability_sha256",
            "consumer_binding_sha256",
            "claim_capability_sha256",
            "claim_binding_sha256",
            "derived_range_postimage_capability_sha256",
            "derived_range_postimage_binding_sha256",
            "range_index",
            "start_index",
            "end_index",
            "artifact_name",
            "artifact_file_sha256",
            "payload_semantic_sha256",
            "previous_range_receipt_sha256",
            "previous_cursor_index",
            "next_cursor_index",
            "all_ranges_consumed",
            "materialization_owner_generation",
            "materialization_owner_binding_sha256",
            "materialization_owner_takeover_chain_sha256",
            "commit_owner_generation",
            "commit_owner_binding_sha256",
            "commit_owner_takeover_chain_sha256",
            "commit_owner_binding_chain",
            "commit_owner_binding_chain_sha256",
            "receipt_sha256",
        }
        self.assertEqual(36, len(exact36))
        self.assertEqual(7, len(exact7))
        self.assertEqual(33, len(exact33))
        self.assertEqual(
            exact36,
            _assigned_literal_string_set(
                forward._validate_completed_forward_rehydrate_binding,
                "exact_fields",
            ),
        )
        self.assertEqual(
            exact7,
            _assigned_literal_string_set(
                forward.rehydrate_completed_forward_run_artifact_handle,
                "exact_material_fields",
            ),
        )
        self.assertEqual(
            exact33,
            _assigned_literal_string_set(
                getattr(
                    forward.CompletedForwardRunArtifactHandle,
                    "transfer_next_validated_chunk_to_controller_once",
                ),
                "exact_consumption_fields",
            ),
        )

    def test_completed_rehydrate_zero_counts_reject_boolean_false(self):
        hash_value = "a" * 64
        base = {
            "schema_version": (
                "c28f_a4_completed_forward_rehydrate_capability_v1"
            ),
            "status": (
                "FSYNCED_CONTROLLER_COMPLETED_RESULT_VERIFIER_REHYDRATE"
            ),
            "goal_id": "goal-a4",
            "attempt_id": "attempt-a4",
            "authority_id": "authority-a4",
            "run_id": "run-a4",
            "token_id": "F0_B",
            "eval_id": "eval-a4",
            "stage": "G5_F0_B",
            "purpose": "F0_B_ROUTE_DEV_REPLICATION",
            "capability_id": hash_value,
            "forward_authority_handle_sha256": hash_value,
            "forward_manifest_sha256": hash_value,
            "input_identity_sha256": hash_value,
            "authority_binding_sha256": hash_value,
            "reservation_input_capability_sha256": hash_value,
            "controller_authority_binding_sha256": hash_value,
            "completed_transaction_id": "run-a4-COMPLETE",
            "completed_state_sha256": hash_value,
            "completed_event_sha256": hash_value,
            "execution_binding_sha256": hash_value,
            "forward_run_receipt_sha256": hash_value,
            "completed_forward_validation_receipt_sha256": hash_value,
            "authority_rehydrate_projection_sha256": hash_value,
            "input_capability_material_sha256": hash_value,
            "run_root_identity_sha256": hash_value,
            "chunk_root_identity_sha256": hash_value,
            "rehydration_generation": 0,
            "rehydration_source": (
                "FSYNCED_CONTROLLER_PRIVATE_COMPLETED_RESULT_VERIFICATION_"
                "PROJECTION_PLUS_COMPLETION_AND_SAFE_INDEX"
            ),
            "token_reissue_count": 0,
            "model_forward_reexecution_count": 0,
            "query_source_index_reaccess_count": 0,
            "raw_chunk_index_namespace_rescan_count": 0,
            "raw_tensor_materialized": False,
            "old_object_identity_required": False,
        }
        payload = {**base, "rehydrate_binding_sha256": _sha(base)}
        module = types.SimpleNamespace(
            validate_completed_forward_rehydrate_capability=(
                lambda _capability: payload
            )
        )
        with mock.patch.object(
            forward,
            "_require_exact_forward_control",
            return_value=module,
        ):
            validated = forward._validate_completed_forward_rehydrate_binding(
                object(),
                object(),
            )
        self.assertEqual(payload, dict(validated))
        for field_name in (
            "token_reissue_count",
            "model_forward_reexecution_count",
            "query_source_index_reaccess_count",
            "raw_chunk_index_namespace_rescan_count",
        ):
            with self.subTest(field_name=field_name):
                tampered_base = {**base, field_name: False}
                tampered = {
                    **tampered_base,
                    "rehydrate_binding_sha256": _sha(tampered_base),
                }
                module = types.SimpleNamespace(
                    validate_completed_forward_rehydrate_capability=(
                        lambda _capability, value=tampered: value
                    )
                )
                with mock.patch.object(
                    forward,
                    "_require_exact_forward_control",
                    return_value=module,
                ), self.assertRaisesRegex(
                    forward.ForwardContractError,
                    "COMPLETED_FORWARD_REHYDRATE_BINDING_INVALID",
                ):
                    forward._validate_completed_forward_rehydrate_binding(
                        object(),
                        object(),
                    )

    def test_sealed_transfer_exposes_only_controller_projection(self):
        source = inspect.getsource(
            forward.consume_validated_forward_derivation_range_transfer_once
        )
        helper_source = inspect.getsource(
            forward._controller_metric_record_from_forward_payload_row
        )
        self.assertIn(
            "c28f_a4_forward_derivation_controller_projection_v1",
            source,
        )
        self.assertIn("VALIDATED_CONTROLLER_ONLY_PRE_GT_PROJECTION", source)
        self.assertIn('object.__setattr__(value, "_payload", None)', source)
        self.assertIn("del payload", source)
        self.assertIn("MappingProxyType(projection)", source)
        self.assertNotIn("MappingProxyType(dict(payload))", source)
        for marker in (
            "validate_forward_derivation_range_claim_capability",
            "validate_forward_derivation_delivery_owner",
            "VALIDATED_DERIVATION_TRANSFER_CURRENT_OWNER_REQUIRED",
            "materialization_owner_generation",
            "materialization_owner_binding_sha256",
            "materialization_owner_takeover_chain_sha256",
        ):
            self.assertIn(marker, source)
        for marker in (
            'int(row["query_id"])',
            '"pooled_video_order"',
            '"late_video_order"',
            '"final_video_order"',
            '"joint_proposals"',
            '"result_sha256"',
        ):
            self.assertIn(marker, helper_source)
        self.assertIn(
            "value._authority.manifest.corpus_identity",
            source,
        )
        self.assertNotIn("value._authority.manifest.corpus,", source)

    def test_two_phase_derivation_names_all_three_recovery_windows(self):
        handle_source = inspect.getsource(
            getattr(
                forward.CompletedForwardRunArtifactHandle,
                "transfer_next_validated_chunk_to_controller_once",
            )
        )
        from blueprint_e2e_v2.c28f_v5 import a4_control as control

        claim_source = inspect.getsource(
            control.prepare_forward_derivation_range
        )
        sink_source = inspect.getsource(
            control.derive_and_fsync_forward_range
        )
        commit_source = inspect.getsource(
            control.commit_forward_derivation_range
        )
        self.assertIn("atomic_write_immutable", claim_source)
        self.assertNotIn("next_cursor_index", claim_source)
        self.assertIn("atomic_write_immutable", sink_source)
        self.assertNotIn("next_cursor_index", sink_source)
        self.assertIn("if range_index < len(receipts)", commit_source)
        self.assertIn("return dict(existing)", commit_source)
        self.assertIn("forward_derivation_resume_snapshot", handle_source)
        self.assertIn("resume_cursor < self._next_chunk_index", handle_source)
        for marker in (
            '"materialization_owner_generation"',
            '"materialization_owner_binding_sha256"',
            '"materialization_owner_takeover_chain_sha256"',
            '"commit_owner_generation"',
            '"commit_owner_binding_sha256"',
            '"commit_owner_takeover_chain_sha256"',
            '"commit_owner_binding_chain"',
            '"commit_owner_binding_chain_sha256"',
            "commit_owner_binding_chain[:-1]",
            ":materialization_owner_generation",
        ):
            self.assertIn(marker, handle_source)
        for obsolete in (
            '"delivery' + '_owner_generation"',
            '"delivery' + '_owner_binding_sha256"',
            '"delivery' + '_owner_takeover_chain_sha256"',
        ):
            self.assertNotIn(obsolete, handle_source)

    def test_real_result_sentinel_and_nonce_still_cannot_forge_engine_issuance(self):
        engine = object.__new__(forward.FrozenForwardEngine)
        engine.authority = self.fixture.authority
        engine._issued_result_registry = {}
        engine._issued_query_ids = set()
        engine._consumed_query_ids = set()
        forged = object.__new__(forward.ForwardQueryResult)
        object.__setattr__(forged, "_factory", forward._RESULT_FACTORY)
        object.__setattr__(
            forged,
            "_issuer_nonce",
            self.fixture.authority._result_issuer_nonce,
        )
        object.__setattr__(
            forged,
            "query_id",
            self.fixture.manifest.query_identity.query_keys[0],
        )
        object.__setattr__(
            forged,
            "query_type",
            self.fixture.manifest.query_identity.query_types[0],
        )
        with self.assertRaisesRegex(
            forward.ForwardContractError,
            "ENGINE_RESULT_NOT_ISSUED_OR_ALREADY_CONSUMED",
        ):
            engine._consume_issued_results_once((forged,), start_index=0)

    def test_cursor_absence_and_wrong_completion_capability_fail_exactly(self):
        control_module = forward._a4_control_module()
        loader_source = inspect.getsource(
            forward.ForwardRunJournal._load_control_cursor_capability
        )
        self.assertIn("except module.A4EvalCursorAbsentError", loader_source)
        self.assertNotIn("str(error)", loader_source)
        self.assertNotIn("A4ControlError as", loader_source)
        adapter = object.__new__(control_module.A4EvalForwardControlAdapter)
        with self.assertRaises(control_module.A4ControlError):
            adapter.complete_eval_token(cursor_capability=object())


class PureEquivalenceAndBombTests(unittest.TestCase):
    def test_owned_torch_archive_limits_precede_deserialization(self):
        class TooManyMembers:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def infolist(self):
                return [object()] * (forward.MAX_OWNED_TORCH_ARCHIVE_MEMBERS + 1)

        handle = io.BytesIO(b"not-loaded")
        with mock.patch.object(
            forward.zipfile,
            "ZipFile",
            return_value=TooManyMembers(),
        ), self.assertRaisesRegex(forward.ForwardContractError, "MEMBER_LIMIT"):
            forward._inspect_owned_torch_archive(handle, file_size=1)
        with self.assertRaisesRegex(forward.ForwardContractError, "SIZE_LIMIT"):
            forward._inspect_owned_torch_archive(
                io.BytesIO(),
                file_size=forward.MAX_OWNED_TORCH_ARCHIVE_BYTES + 1,
            )

    def test_descriptor_reads_reject_symlinks_and_hardlinks(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            identity = forward.FrozenDirectory.capture("read-root", str(root))
            target = root / "target.bin"
            target.write_bytes(b"frozen")
            symlink = root / "symlink.bin"
            symlink.symlink_to(target.name)
            with self.assertRaises((forward.ForwardContractError, OSError)):
                forward._read_bytes_no_follow(
                    symlink,
                    expected_parent=identity,
                )
            hardlink = root / "hardlink.bin"
            os.link(target, hardlink)
            with self.assertRaisesRegex(
                forward.ForwardContractError, "READ_TARGET_NOT_REGULAR"
            ):
                forward._read_bytes_no_follow(
                    target,
                    expected_parent=identity,
                )

    def test_stable_topk_tie_reference(self):
        scores = [1.0, 3.0, 3.0, -1.0, 3.0]
        ids = ["z", "b", "a", "q", "c"]
        expected = sorted(range(5), key=lambda index: (-scores[index], ids[index]))[:3]
        self.assertEqual(expected, forward.stable_topk_indices(scores, ids, 3))

    def test_stable_topk_full_17435_reference_with_boundary_ties(self):
        identities = [f"video-{index:05d}" for index in range(17_435)]
        scores = [float((index * 97) % 31) for index in range(17_435)]
        expected = sorted(
            range(17_435), key=lambda index: (-scores[index], identities[index])
        )[:1000]
        self.assertEqual(
            expected,
            forward.stable_topk_indices(scores, identities, 1000),
        )

    def test_temporal_grid_and_all_sequence_fit_paths_are_exact(self):
        import numpy as np
        from blueprint_e2e_v2.data.proposal_dataset import MultiSpanProposalDataset
        from blueprint_e2e_v2.data.subtitle_bank import SubtitleBank
        from blueprint_e2e_v2.data.temporal_grid import TemporalGrid
        from blueprint_e2e_v2.data.video_bank import VideoBank

        grid = TemporalGrid(clip_len=1.5, max_clips=64)
        for duration in (0.001, 1.0, 3.0, 95.99, 96.0, 137.25):
            expected_clip = grid.grid_spans(duration, max_spans=64)
            expected_sec = grid.clips_to_seconds(expected_clip, duration)
            observed_clip, observed_sec = forward.grid_spans_for_duration(
                duration,
                _grid=grid,
            )
            np.testing.assert_array_equal(np.asarray(observed_clip), expected_clip)
            np.testing.assert_array_equal(
                np.asarray(observed_sec, dtype=np.float32), expected_sec
            )
        for length in (1, 17, 63, 64, 65, 129, 4096):
            sequence = np.arange(length * 3, dtype=np.float32).reshape(length, 3)
            observed, mask = forward._fit_sequence(sequence, 64)
            references = (
                MultiSpanProposalDataset._fit_seq(sequence, 64),
                VideoBank._fit_seq(sequence, 64),
                SubtitleBank._fit_seq(sequence, 64),
            )
            for expected in references:
                np.testing.assert_array_equal(observed, expected)
            expected_mask = MultiSpanProposalDataset._fit_mask(length, 64)
            np.testing.assert_array_equal(mask, expected_mask)

    def test_query_schema_rejects_gt_teacher_and_reordering(self):
        identity = _queries()
        rows = [
            {"query_id": key, "query_type": query_type}
            for key, query_type in zip(
                identity.query_keys,
                identity.query_types,
            )
        ]
        contaminated = list(rows)
        contaminated[0] = dict(contaminated[0], gt_video_id="forbidden")
        with self.assertRaises(forward.ForwardContractError):
            forward.normalize_forward_queries(contaminated, identity)
        rows[0], rows[1] = rows[1], rows[0]
        with self.assertRaises(forward.ForwardContractError):
            forward.normalize_forward_queries(rows, identity)

    def test_query_type_is_finite_enum_in_forensics_result_seal_and_chunk(self):
        identity = _queries()
        rows = [
            {"query_id": key, "query_type": query_type}
            for key, query_type in zip(
                identity.query_keys,
                identity.query_types,
            )
        ]
        invalid = list(rows)
        invalid[0] = {"query_id": identity.query_keys[0], "query_type": "free-text"}
        with self.assertRaisesRegex(forward.ForwardContractError, "INVALID_QUERY_TYPE"):
            forward.normalize_forward_queries(invalid, identity)
        legal_substitution = list(rows)
        legal_substitution[0] = {
            "query_id": identity.query_keys[0],
            "query_type": "t",
        }
        with self.assertRaisesRegex(
            forward.ForwardContractError,
            "QUERY_TYPE_AUTHORITY_BINDING_MISMATCH",
        ):
            forward.normalize_forward_queries(legal_substitution, identity)
        unknown_degradation = list(rows)
        unknown_degradation[0] = {
            "query_id": identity.query_keys[0],
            "query_type": "unknown",
        }
        with self.assertRaisesRegex(
            forward.ForwardContractError,
            "QUERY_TYPE_AUTHORITY_BINDING_MISMATCH",
        ):
            forward.normalize_forward_queries(unknown_degradation, identity)
        result_source = inspect.getsource(forward.ForwardQueryResult)
        seal_source = inspect.getsource(forward._forward_query_result_sha256)
        payload_source = inspect.getsource(forward._forward_result_payload)
        validator_source = inspect.getsource(forward._validate_forward_result_payload)
        digest_source = inspect.getsource(forward._forward_result_payload_digest)
        self.assertIn('"query_type"', result_source)
        self.assertIn('"query_type": result.query_type', seal_source)
        self.assertIn('"query_type": result.query_type', payload_source)
        self.assertIn("FORWARD_RESULT_QUERY_TYPE_BINDING_MISMATCH", validator_source)
        self.assertIn('"query_type": row["query_type"]', digest_source)
        self.assertIn("query_rows_sha256", payload_source)
        self.assertIn("query_types_sha256", validator_source)

    def test_bucket_forensics_are_numeric_mask_derived_and_sealed(self):
        forward_source = inspect.getsource(forward.FrozenForwardEngine.forward_one)
        coverage_source = inspect.getsource(forward._late_candidate_mask_coverage)
        validator_source = inspect.getsource(
            forward._validate_forward_result_payload
        )
        for marker in (
            "query_feature_token_count",
            "QUERY_FEATURE_TOKEN_COUNT_DEFINITION",
            "late_maxsim_token_count",
            "LATE_MAXSIM_TOKEN_COUNT_DEFINITION",
            "FEATURE_TO_LATE_MAXSIM_TOKEN_COUNT_INVARIANT",
            "feature_to_late_maxsim_invariant",
            "late_candidate_mask_coverage",
        ):
            self.assertIn(marker, forward_source)
            self.assertIn(marker, validator_source)
        self.assertEqual(
            "FORWARD_SEALED_QUERY_MASK_SUM_USED_BY_MODEL",
            forward.QUERY_FEATURE_TOKEN_COUNT_DEFINITION,
        )
        self.assertEqual(
            "FORWARD_SEALED_LATE_MAXSIM_QUERY_MASK_SUM",
            forward.LATE_MAXSIM_TOKEN_COUNT_DEFINITION,
        )
        self.assertEqual(
            "EXACT_SAME_FORWARD_QUERY_MASK_COUNTS_EQUAL",
            forward.FEATURE_TO_LATE_MAXSIM_TOKEN_COUNT_INVARIANT,
        )
        self.assertEqual(
            "UNICODE_CASEFOLD_THEN_ASCII_PUNCTUATION_TO_SPACE_THEN_"
            "WHITESPACE_SPLIT_COUNT",
            forward.POST_FORWARD_QUERY_TEXT_TOKEN_COUNT_RULE,
        )
        model_contract = forward.FrozenModelContract().as_dict()
        self.assertEqual(
            forward.FEATURE_TO_LATE_MAXSIM_TOKEN_COUNT_INVARIANT,
            model_contract["feature_to_late_maxsim_invariant"],
        )
        self.assertEqual(
            forward.POST_FORWARD_QUERY_TEXT_TOKEN_COUNT_RULE,
            model_contract["post_forward_query_text_token_count_rule"],
        )
        self.assertNotIn('"query_token_count"', forward_source)
        self.assertNotIn('"query_token_count"', validator_source)
        self.assertIn("mask.sum().item()", coverage_source)
        self.assertIn("mask.numel()", coverage_source)
        self.assertIn('coverage["visual"]["empty_candidate_count"] != 0', coverage_source)
        self.assertIn('masks["visual"] | masks["subtitle"]', coverage_source)
        self.assertIn('coverage["joint"]["empty_candidate_count"] != 0', coverage_source)
        self.assertEqual(
            forward._mask_structure_contract(),
            model_contract["mask_structure_contract"],
        )
        self.assertNotIn("query_text", forward_source)

    def test_late_mask_coverage_accepts_fully_missing_subtitles(self):
        import torch

        visual_mask = torch.ones(
            forward.LATE_TOPK,
            forward.TARGET_LENGTH,
            dtype=torch.bool,
        )
        subtitle_mask = torch.zeros_like(visual_mask)
        joint_mask = visual_mask | subtitle_mask
        coverage = forward._late_candidate_mask_coverage(
            {
                "visual_mask": visual_mask,
                "subtitle_mask": subtitle_mask,
                "joint_mask": joint_mask,
            }
        )
        forward._validate_late_candidate_mask_coverage_record(coverage)
        denominator = forward.LATE_TOPK * forward.TARGET_LENGTH
        self.assertEqual(0, coverage["subtitle"]["valid_count"])
        self.assertEqual(denominator, coverage["subtitle"]["missing_count"])
        self.assertEqual(
            forward.LATE_TOPK,
            coverage["subtitle"]["empty_candidate_count"],
        )
        self.assertEqual(0, coverage["visual"]["empty_candidate_count"])
        self.assertEqual(0, coverage["joint"]["empty_candidate_count"])
        self.assertEqual(0, coverage["visual_subtitle_overlap_valid_count"])
        self.assertIs(True, coverage["joint_component_relation_verified"])

    def test_late_mask_coverage_accepts_partially_missing_subtitles(self):
        import torch

        visual_mask = torch.zeros(
            forward.LATE_TOPK,
            forward.TARGET_LENGTH,
            dtype=torch.bool,
        )
        visual_mask[:, :8] = True
        subtitle_mask = torch.zeros_like(visual_mask)
        subtitle_mask[:100, 4:12] = True
        joint_mask = visual_mask | subtitle_mask
        coverage = forward._late_candidate_mask_coverage(
            {
                "visual_mask": visual_mask,
                "subtitle_mask": subtitle_mask,
                "joint_mask": joint_mask,
            }
        )
        forward._validate_late_candidate_mask_coverage_record(coverage)
        self.assertEqual(100, coverage["subtitle"]["empty_candidate_count"])
        self.assertEqual(800, coverage["subtitle"]["valid_count"])
        self.assertEqual(400, coverage["visual_subtitle_overlap_valid_count"])
        self.assertEqual(2_000, coverage["joint"]["valid_count"])

    def test_late_mask_joint_relation_mutation_is_rejected(self):
        import torch

        visual_mask = torch.zeros(
            forward.LATE_TOPK,
            forward.TARGET_LENGTH,
            dtype=torch.bool,
        )
        visual_mask[:, :8] = True
        subtitle_mask = torch.zeros_like(visual_mask)
        subtitle_mask[:100, 4:12] = True
        valid = {
            "visual_mask": visual_mask,
            "subtitle_mask": subtitle_mask,
            "joint_mask": visual_mask | subtitle_mask,
        }
        coverage = forward._late_candidate_mask_coverage(valid)
        mutated_masks = dict(valid)
        mutated_masks["joint_mask"] = visual_mask & subtitle_mask
        with self.assertRaisesRegex(
            forward.ForwardContractError,
            "LATE_JOINT_MASK_COMPONENT_RELATION_MISMATCH",
        ):
            forward._late_candidate_mask_coverage(mutated_masks)
        detached = json.loads(json.dumps(coverage))
        detached["visual_subtitle_overlap_valid_count"] += 1
        with self.assertRaisesRegex(
            forward.ForwardContractError,
            "LATE_MASK_COVERAGE_RELATION_INVALID",
        ):
            forward._validate_late_candidate_mask_coverage_record(detached)

    def test_soft_topk_neutralizes_only_explicitly_allowed_empty_rows(self):
        import torch

        values = torch.randn(1, 2, forward.TARGET_LENGTH)
        mask = torch.zeros_like(values, dtype=torch.bool)
        subtitle_score = forward._safe_soft_topk(
            torch,
            values,
            8,
            0.07,
            mask,
            allow_empty_mask_rows=True,
        )
        torch.testing.assert_close(
            subtitle_score,
            torch.zeros_like(subtitle_score),
            rtol=0,
            atol=0,
        )
        with self.assertRaisesRegex(
            forward.ForwardContractError,
            "LATE_MASK_HAS_EMPTY_CANDIDATE",
        ):
            forward._safe_soft_topk(torch, values, 8, 0.07, mask)

    def test_weights_only_loader_has_no_fallback(self):
        class OldTorch:
            calls = 0

            def load(self, _handle, **_kwargs):
                self.calls += 1
                raise TypeError("unsupported")

        module = OldTorch()
        with self.assertRaisesRegex(forward.ForwardContractError, "WEIGHTS_ONLY"):
            forward._torch_load_weights_only(module, object())
        self.assertEqual(1, module.calls)

    def test_exact_direct_model_batch_parity_on_synthetic_float32(self):
        import torch
        from blueprint_e2e_v2.models.full_model import C28CFullModel

        torch.manual_seed(7)
        model = C28CFullModel(
            late_interaction_enabled=True,
            late_soft_topk=8,
            late_temperature=0.07,
            token_maxsim_weight=0.25,
            pooled_score_weight=0.35,
            late_score_weight=0.65,
        ).eval()
        query = torch.randn(1, 5, 768)
        query_mask = torch.tensor([[True, True, True, False, False]])
        query_type = torch.tensor([2])
        visual = torch.randn(1, 2, 64, 4352)
        subtitle = torch.randn(1, 2, 64, 768)
        visual_mask = torch.zeros(1, 2, 64, dtype=torch.bool)
        subtitle_mask = torch.zeros(1, 2, 64, dtype=torch.bool)
        visual_mask[0, 0, :41] = True
        visual_mask[0, 1, :64] = True
        subtitle_mask[0, 0, :29] = True
        subtitle_mask[0, 1, :47] = True
        span_mask = torch.tensor([[[True, False], [True, True]]])
        spans = torch.tensor([[[[0, 8], [8, 16]], [[0, 8], [8, 16]]]])
        batch = {
            "query_tokens": query,
            "query_mask": query_mask,
            "query_type": query_type,
            "visual": visual,
            "subtitle": subtitle,
            "visual_clip_mask": visual_mask,
            "subtitle_clip_mask": subtitle_mask,
            "clip_mask": visual_mask | subtitle_mask,
            "spans_clip": spans,
            "span_mask": span_mask,
        }
        with torch.inference_mode():
            direct = model(batch)
            encoded = model.video_encoder(
                visual,
                subtitle,
                visual_mask=visual_mask,
                subtitle_mask=subtitle_mask,
                clip_mask=batch["clip_mask"],
            )
            q = model.encode_query(query, query_type, query_mask)
            cached = {key: value[0].cpu() for key, value in encoded.items()}
            cached.update(
                {
                    "visual_mask": visual_mask[0].cpu(),
                    "subtitle_mask": subtitle_mask[0].cpu(),
                    "joint_mask": batch["clip_mask"][0].cpu(),
                }
            )
            retr = forward._late_retrieval_mapping(
                model,
                q,
                query,
                query_mask,
                cached,
                forward.FrozenModelContract(),
                torch,
                torch.device("cpu"),
            )
            partial = model.partial(q, retr["encoded"], retr["visual_mask"], retr["subtitle_mask"])
            region = model.region(retr["encoded"], partial)
            active = model.active(q, enc=retr["encoded"], clip_mask=retr["joint_mask"])
            proposed = model.proposals(spans)
            local = model.localizer(
                retr["encoded"], partial, region, active, retr, proposed,
                visual_mask=retr["visual_mask"],
                subtitle_mask=retr["subtitle_mask"],
                clip_mask=retr["joint_mask"],
            )
            feedback = model.feedback(
                local["span_score"], local["prem_span"], local["region_span"],
                local["amd_span"], local["false_positive_risk"], span_mask,
            )
            cached_score = model.scorer(retr, local, feedback)
        cached_sections = {
            "retr": retr,
            "partial": partial,
            "region": region,
            "active": active,
            "local": local,
            "feedback": feedback,
            "score": cached_score,
        }
        for section_name, cached_section in cached_sections.items():
            if section_name == "retr":
                self.assertEqual(
                    set(cached_section) - set(direct[section_name]),
                    {"encoded", "visual_mask", "subtitle_mask", "joint_mask"},
                )
            else:
                self.assertEqual(set(direct[section_name]), set(cached_section))
            for field_name in direct[section_name]:
                torch.testing.assert_close(
                    direct[section_name][field_name],
                    cached_section[field_name],
                    rtol=0,
                    atol=0,
                    msg=f"{section_name}.{field_name}",
                )
        self.assertTrue(torch.equal(retr["visual_mask"], visual_mask))
        self.assertTrue(torch.equal(retr["subtitle_mask"], subtitle_mask))
        self.assertTrue(torch.equal(retr["joint_mask"], batch["clip_mask"]))

    def test_completed_gap_recovery_reconstructs_no_active_io(self):
        source = inspect.getsource(
            forward.recover_completed_forward_run_artifact_handle_from_control
        )
        self.assertIn("_freeze_existing_child_directory(", source)
        self.assertIn(
            "completed_forward_input_rehydrate_material(",
            source,
        )
        self.assertIn("material_record.get(field_name) is not False", source)
        self.assertIn('"active_source_reopened"', source)
        self.assertIn('"token_reissued"', source)
        self.assertIn('"model_forward_reexecuted"', source)
        self.assertIn("validate_completed_forward_run_artifacts(", source)
        for forbidden in (
            "load_frozen_model",
            "GuardedLmdbReader",
            "materialize_or_resume",
            "issue_capability(",
        ):
            self.assertNotIn(forbidden, source)

    def test_existing_child_recovery_helper_never_creates(self):
        source = inspect.getsource(forward._freeze_existing_child_directory)
        self.assertIn("os.open(part, flags, dir_fd=current_fd)", source)
        self.assertNotIn("os.mkdir", source)
        self.assertNotIn("_safe_child_directory", source)

    def test_encoded_cache_runner_surface_keeps_torch_private(self):
        source = inspect.getsource(forward.EncodedCorpusCache.materialize_and_load)
        self.assertIn("torch_module = _lazy_torch()", source)
        self.assertIn("self.materialize_or_resume(model_handle, torch_module)", source)
        self.assertIn("return self.load_complete_handle(torch_module)", source)


if __name__ == "__main__":
    unittest.main()
