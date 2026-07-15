"""Fail-closed C28F F0-A/F0-B forward adapter.

This module is the only model-forward implementation intended for G3 and G5.
It deliberately does not import ``torch`` or ``lmdb`` at module import time and
does not call any legacy evaluator, candidate miner, dataset, checkpoint helper,
or runner.  Expensive dependencies are imported only after a frozen capability
has passed every pre-open check.

The adapter has four security boundaries:

* ``FrozenInputAuthority`` binds a single checkpoint, the complete 17,435-video
  corpus, exact ordered query LMDB keys and types, isolated roots, a stage manifest, and an
  externally fsynced RUNNING-state capability and a locally verified physical-GPU0
  runtime lease.
* ``GuardedLmdbReader`` checks an exact key before beginning an LMDB transaction.
* ``EncodedCorpusCache`` writes checkpoint-specific encoded chunks with canonical
  receipts and can resume only under the same cache fingerprint.
* ``FrozenForwardEngine`` performs pooled full-corpus top-1000, late top-200, and
  the downstream full model on those encoded 200 candidates.  Its public query
  schema has no teacher, label, target, timestamp, or GT field.

GPU process discovery/termination and lease acquisition are orchestrator duties.
This module never sends a signal or invokes a process-management command.
"""

from __future__ import annotations

import hashlib
import importlib
import io
import json
import math
import os
import re
import secrets
import stat
import sys
import threading
import zipfile
from dataclasses import MISSING, dataclass, field, fields
from importlib.machinery import SourceFileLoader
from pathlib import Path
from types import MappingProxyType
from typing import Any, Iterable, Iterator, Mapping, Optional, Sequence, Tuple


FORWARD_SCHEMA_VERSION = "c28f_v5_a4_forward_v1"
FORWARD_MANIFEST_SCHEMA_VERSION = "c28f_v5_a4_forward_manifest_v1"
FORWARD_RUN_RECEIPT_SCHEMA_VERSION = "c28f_v5_a4_forward_run_receipt_v1"
ENCODED_CACHE_SCHEMA_VERSION = "c28f_v5_a4_encoded_corpus_v1"
ENCODED_CHUNK_SCHEMA_VERSION = "c28f_v5_a4_encoded_chunk_v1"

EXPECTED_CORPUS_VIDEO_COUNT = 17_435
EXPECTED_F0A_QUERY_COUNT = 8_677
POOLED_TOPK = 1_000
LATE_TOPK = 200
FULL_MODEL_CANDIDATES = 200
TARGET_LENGTH = 64
MAX_PROPOSALS_PER_VIDEO = 64
NMS_IOU_THRESHOLD = 0.7
METRIC_NMS_MAX_RESULTS = 100
EXPECTED_DEVICE = "cuda:0"
QUERY_FEATURE_TOKEN_COUNT_DEFINITION = (
    "FORWARD_SEALED_QUERY_MASK_SUM_USED_BY_MODEL"
)
LATE_MAXSIM_TOKEN_COUNT_DEFINITION = (
    "FORWARD_SEALED_LATE_MAXSIM_QUERY_MASK_SUM"
)
FEATURE_TO_LATE_MAXSIM_TOKEN_COUNT_INVARIANT = (
    "EXACT_SAME_FORWARD_QUERY_MASK_COUNTS_EQUAL"
)
POST_FORWARD_QUERY_TEXT_TOKEN_COUNT_RULE = (
    "UNICODE_CASEFOLD_THEN_ASCII_PUNCTUATION_TO_SPACE_THEN_"
    "WHITESPACE_SPLIT_COUNT"
)
MASK_COVERAGE_RATIO_DEFINITION = (
    "VALID_OR_MISSING_BOOL_CELLS_DIVIDED_BY_CANDIDATE_COUNT_TIMES_64"
)

QUERY_VALUE_MAX_BYTES = 16 * 1024 * 1024
SUBTITLE_VALUE_MAX_BYTES = 64 * 1024 * 1024
VISUAL_VALUE_MAX_BYTES = 512 * 1024 * 1024
QUERY_SEQUENCE_MAX_LENGTH = 256
VIDEO_SEQUENCE_MAX_LENGTH = 4_096
MAX_OWNED_TORCH_ARCHIVE_BYTES = 1 * 1024 * 1024 * 1024
MAX_OWNED_TORCH_UNCOMPRESSED_BYTES = 2 * 1024 * 1024 * 1024
MAX_OWNED_TORCH_ARCHIVE_MEMBERS = 4_096

_SHA256_HEX = frozenset("0123456789abcdef")
_STAGE_PURPOSES = {
    "G3_F0_A": "F0_A_TEACHER_FREE_CALIB_SELECT_FORENSICS",
    "G5_F0_B": "F0_B_ROUTE_DEV_REPLICATION",
}
_QUERY_TYPES = MappingProxyType({"v": 0, "t": 1, "vt": 2, "unknown": 3})
_REQUIRED_PACKAGE_MODULE_BY_SOURCE = MappingProxyType(
    {
        "__init__.py": "blueprint_e2e_v2",
        "data/__init__.py": "blueprint_e2e_v2.data",
        "models/__init__.py": "blueprint_e2e_v2.models",
        "utils/__init__.py": "blueprint_e2e_v2.utils",
    }
)
_REQUIRED_SOURCE_BY_PACKAGE_MODULE = MappingProxyType(
    {
        module_name: source_name
        for source_name, module_name in (
            _REQUIRED_PACKAGE_MODULE_BY_SOURCE.items()
        )
    }
)
_PRELOADED_ROOT_PACKAGE_MODULE = "blueprint_e2e_v2"
_APPROVED_BUSINESS_MODULE_BY_SOURCE = MappingProxyType(
    {
        "data/temporal_grid.py": "blueprint_e2e_v2.data.temporal_grid",
        "models/active_moment.py": "blueprint_e2e_v2.models.active_moment",
        "models/full_model.py": "blueprint_e2e_v2.models.full_model",
        "models/partial_relevance.py": "blueprint_e2e_v2.models.partial_relevance",
        "models/proposal_generator.py": "blueprint_e2e_v2.models.proposal_generator",
        "models/query_encoder.py": "blueprint_e2e_v2.models.query_encoder",
        "models/region_prior.py": "blueprint_e2e_v2.models.region_prior",
        "models/retrieval_guided_localizer.py": (
            "blueprint_e2e_v2.models.retrieval_guided_localizer"
        ),
        "models/retriever.py": "blueprint_e2e_v2.models.retriever",
        "models/span_to_video_feedback.py": (
            "blueprint_e2e_v2.models.span_to_video_feedback"
        ),
        "models/vcmr_scorer.py": "blueprint_e2e_v2.models.vcmr_scorer",
        "models/video_encoder.py": "blueprint_e2e_v2.models.video_encoder",
        "utils/tensor_ops.py": "blueprint_e2e_v2.utils.tensor_ops",
    }
)
_APPROVED_SOURCE_BY_BUSINESS_MODULE = MappingProxyType(
    {
        module_name: source_name
        for source_name, module_name in _APPROVED_BUSINESS_MODULE_BY_SOURCE.items()
    }
)
_FULL_MODEL_RUNTIME_MODULE_CLOSURE = frozenset(
    {
        module_name
        for source_name, module_name in _APPROVED_BUSINESS_MODULE_BY_SOURCE.items()
        if source_name.startswith("models/") or source_name == "utils/tensor_ops.py"
    }
)
_FULL_MODEL_RUNTIME_PACKAGE_CLOSURE = frozenset(
    {
        "blueprint_e2e_v2.models",
        "blueprint_e2e_v2.utils",
    }
)
_TEMPORAL_RUNTIME_PACKAGE_CLOSURE = frozenset(
    {"blueprint_e2e_v2.data"}
)
_FORBIDDEN_QUERY_FIELD_FRAGMENTS = (
    "teacher",
    "ground_truth",
    "gt_",
    "label",
    "target",
    "timestamp",
    "span",
    "duration",
    "video_id",
    "vid_name",
    "ts",
)


class ForwardContractError(RuntimeError):
    """Fail-closed error with a stable, non-sensitive public code."""

    def __init__(self, code: str, detail: str = "") -> None:
        if not isinstance(code, str) or not code or any(
            character not in "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_" for character in code
        ):
            raise ValueError("forward error code must be an uppercase identifier")
        self.code = code
        self.detail = detail
        super().__init__(code if not detail else "%s: %s" % (code, detail))


def _linux_o_path_flag() -> int:
    """Return Linux O_PATH for CPython builds that omit the symbol."""

    if not sys.platform.startswith("linux") or not hasattr(os, "O_NOFOLLOW"):
        raise ForwardContractError("DESCRIPTOR_WALK_PLATFORM_UNSUPPORTED")
    return int(getattr(os, "O_PATH", 0o10000000))


def _canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ForwardContractError("NON_CANONICAL_JSON_VALUE", type(error).__name__) from None


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file_descriptor(fd: int) -> str:
    digest = hashlib.sha256()
    os.lseek(fd, 0, os.SEEK_SET)
    while True:
        block = os.read(fd, 8 * 1024 * 1024)
        if not block:
            break
        digest.update(block)
    os.lseek(fd, 0, os.SEEK_SET)
    return digest.hexdigest()


def _fd_mount_id(fd: int) -> int:
    """Return Linux mount identity for an already-open descriptor."""

    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    info_fd = os.open("/proc/self/fdinfo/%d" % fd, flags)
    try:
        parts = []
        total = 0
        while total <= 16_384:
            block = os.read(info_fd, min(4_096, 16_385 - total))
            if not block:
                break
            parts.append(block)
            total += len(block)
        if total > 16_384:
            raise ForwardContractError("FDINFO_TOO_LARGE")
    finally:
        os.close(info_fd)
    matches = [
        line.split(b":", 1)[1].strip()
        for line in b"".join(parts).splitlines()
        if line.startswith(b"mnt_id:")
    ]
    if len(matches) != 1 or not matches[0].isdigit():
        raise ForwardContractError("FDINFO_MOUNT_ID_UNAVAILABLE")
    return int(matches[0])


def _semantic_sha256(value: Mapping[str, Any], excluded: Iterable[str] = ()) -> str:
    ignored = frozenset(excluded)
    return _sha256_bytes(
        _canonical_json_bytes({key: item for key, item in value.items() if key not in ignored})
    )


def _mask_structure_contract() -> Mapping[str, Any]:
    base = {
        "schema_version": "c28f_a4_forward_mask_structure_contract_v1",
        "visual_candidate_rule": "EVERY_CANDIDATE_HAS_AT_LEAST_ONE_VALID_CELL",
        "subtitle_candidate_rule": (
            "ZERO_VALID_CELLS_PER_CANDIDATE_AND_ALL_ZERO_QUERY_ALLOWED"
        ),
        "joint_component_rule": "EXACT_LOGICAL_OR_VISUAL_SUBTITLE",
        "joint_candidate_rule": "EVERY_CANDIDATE_HAS_AT_LEAST_ONE_VALID_CELL",
    }
    return {**base, "contract_sha256": _semantic_sha256(base)}


def _require_sha256(value: Any, field_name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _SHA256_HEX for character in value)
    ):
        raise ForwardContractError("INVALID_SHA256", field_name)
    return value


def _require_text(value: Any, field_name: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or "\x00" in value
    ):
        raise ForwardContractError("INVALID_TEXT_FIELD", field_name)
    return value


def _require_nonnegative_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ForwardContractError("INVALID_NONNEGATIVE_INTEGER", field_name)
    return value


def _require_positive_int(value: Any, field_name: str) -> int:
    result = _require_nonnegative_int(value, field_name)
    if result == 0:
        raise ForwardContractError("INVALID_POSITIVE_INTEGER", field_name)
    return result


def _require_finite(value: Any, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ForwardContractError("INVALID_FINITE_NUMBER", field_name)
    result = float(value)
    if not math.isfinite(result):
        raise ForwardContractError("INVALID_FINITE_NUMBER", field_name)
    return result


def _canonical_absolute_path(raw_path: str) -> Tuple[str, Tuple[str, ...]]:
    if not isinstance(raw_path, str) or not raw_path or "\x00" in raw_path:
        raise ForwardContractError("INVALID_ABSOLUTE_PATH")
    if not raw_path.startswith("/"):
        raise ForwardContractError("PATH_NOT_ABSOLUTE")
    if raw_path != "/" and raw_path.endswith("/"):
        raise ForwardContractError("PATH_TRAILING_SLASH")
    if "//" in raw_path:
        raise ForwardContractError("PATH_DOUBLE_SLASH")
    pieces = tuple(raw_path.split("/")[1:])
    if any(piece in {"", ".", ".."} for piece in pieces):
        raise ForwardContractError("PATH_TRAVERSAL_OR_DOT_COMPONENT")
    if os.path.normpath(raw_path) != raw_path:
        raise ForwardContractError("PATH_NON_CANONICAL")
    return raw_path, pieces


def _assert_no_symlink_components(path: str, *, allow_missing_leaf: bool = False) -> None:
    canonical, pieces = _canonical_absolute_path(path)
    current = "/"
    for index, piece in enumerate(pieces):
        current = os.path.join(current, piece)
        try:
            item = os.lstat(current)
        except FileNotFoundError:
            if allow_missing_leaf and index == len(pieces) - 1:
                return
            raise ForwardContractError("PATH_COMPONENT_MISSING", current) from None
        if stat.S_ISLNK(item.st_mode):
            raise ForwardContractError("SYMLINK_COMPONENT_FORBIDDEN", current)
    if canonical != path:
        raise ForwardContractError("PATH_NON_CANONICAL")


def _is_within(candidate: str, root: str) -> bool:
    try:
        return os.path.commonpath((candidate, root)) == root
    except ValueError:
        return False


def _paths_overlap(left: str, right: str) -> bool:
    return _is_within(left, right) or _is_within(right, left)


def _line_sha256(values: Sequence[str]) -> str:
    return _sha256_bytes("".join("%s\n" % value for value in values).encode("utf-8"))


def _canonical_query_key(query_id: Any) -> str:
    if isinstance(query_id, bool):
        raise ForwardContractError("INVALID_QUERY_ID")
    if isinstance(query_id, int):
        if query_id < 0:
            raise ForwardContractError("INVALID_QUERY_ID")
        return str(query_id)
    if isinstance(query_id, str):
        if not query_id or not query_id.isascii() or not query_id.isdigit():
            raise ForwardContractError("INVALID_QUERY_ID")
        if query_id != "0" and query_id.startswith("0"):
            raise ForwardContractError("NON_CANONICAL_QUERY_ID")
        return query_id
    raise ForwardContractError("INVALID_QUERY_ID")


def _exact_key_digest(namespace: str, raw_key: bytes) -> str:
    name = _require_text(namespace, "namespace")
    if not isinstance(raw_key, bytes) or not raw_key or b"\x00" in raw_key:
        raise ForwardContractError("INVALID_LMDB_KEY")
    return _sha256_bytes(name.encode("ascii") + b"\x00" + raw_key)


def _lazy_numpy() -> Any:
    return importlib.import_module("numpy")


def _lazy_torch() -> Any:
    return importlib.import_module("torch")


def _lazy_lmdb() -> Any:
    return importlib.import_module("lmdb")


def _build_semantic_dataclass(
    contract_type: Any,
    semantic_field: str,
    supplied: Mapping[str, Any],
) -> Any:
    values = dict(supplied)
    temporary = object.__new__(contract_type)
    for item in fields(contract_type):
        if item.name == semantic_field:
            value = "0" * 64
        elif item.name in values:
            value = values[item.name]
        elif item.default is not MISSING:
            value = item.default
        elif item.default_factory is not MISSING:
            value = item.default_factory()
        else:
            raise ForwardContractError("SEMANTIC_BUILDER_FIELD_MISSING", item.name)
        object.__setattr__(temporary, item.name, value)
    payload = temporary.as_dict()
    values[semantic_field] = _semantic_sha256(payload, (semantic_field,))
    return contract_type(**values)


def _open_directory_descriptor(path: str, expected: Optional[Any] = None) -> int:
    """Walk every component by descriptor and return the exact final directory fd."""

    _canonical, pieces = _canonical_absolute_path(path)
    flags = (
        os.O_RDONLY
        | os.O_DIRECTORY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    current_fd = os.open("/", flags)
    success = False
    try:
        for piece in pieces:
            next_fd = os.open(piece, flags, dir_fd=current_fd)
            os.close(current_fd)
            current_fd = next_fd
        item = os.fstat(current_fd)
        mount_id = _fd_mount_id(current_fd)
        if not stat.S_ISDIR(item.st_mode):
            raise ForwardContractError("DIRECTORY_DESCRIPTOR_KIND_MISMATCH")
        if expected is not None and (
            int(item.st_dev),
            int(item.st_ino),
            mount_id,
        ) != (expected.device, expected.inode, expected.mount_id):
            raise ForwardContractError("DIRECTORY_DESCRIPTOR_IDENTITY_DRIFT")
        success = True
        return current_fd
    except OSError as error:
        raise ForwardContractError(
            "DIRECTORY_DESCRIPTOR_WALK_FAILED", type(error).__name__
        ) from None
    finally:
        if not success:
            os.close(current_fd)


@dataclass(frozen=True)
class FrozenDirectory:
    root_id: str
    path: str
    device: int
    inode: int
    mount_id: int

    def __post_init__(self) -> None:
        _require_text(self.root_id, "root_id")
        canonical, _pieces = _canonical_absolute_path(self.path)
        if canonical != self.path:
            raise ForwardContractError("ROOT_PATH_NON_CANONICAL")
        _require_nonnegative_int(self.device, "device")
        _require_nonnegative_int(self.inode, "inode")
        _require_nonnegative_int(self.mount_id, "mount_id")

    @classmethod
    def capture(cls, root_id: str, path: str) -> "FrozenDirectory":
        fd = _open_directory_descriptor(path)
        try:
            item = os.fstat(fd)
            mount_id = _fd_mount_id(fd)
        finally:
            os.close(fd)
        if not stat.S_ISDIR(item.st_mode):
            raise ForwardContractError("ROOT_NOT_DIRECTORY", path)
        return cls(root_id, path, int(item.st_dev), int(item.st_ino), mount_id)

    def verify(self) -> None:
        fd = _open_directory_descriptor(self.path, self)
        try:
            item = os.fstat(fd)
            mount_id = _fd_mount_id(fd)
        finally:
            os.close(fd)
        if not stat.S_ISDIR(item.st_mode):
            raise ForwardContractError("ROOT_NOT_DIRECTORY", self.root_id)
        if (int(item.st_dev), int(item.st_ino), mount_id) != (
            self.device,
            self.inode,
            self.mount_id,
        ):
            raise ForwardContractError("ROOT_IDENTITY_DRIFT", self.root_id)

    def as_dict(self) -> Mapping[str, Any]:
        return {
            "root_id": self.root_id,
            "path": self.path,
            "device": self.device,
            "inode": self.inode,
            "mount_id": self.mount_id,
        }


@dataclass(frozen=True)
class FrozenFile:
    file_id: str
    path: str
    device: int
    inode: int
    mount_id: int
    size_bytes: int
    mtime_ns: int
    sha256: str

    def __post_init__(self) -> None:
        _require_text(self.file_id, "file_id")
        canonical, _pieces = _canonical_absolute_path(self.path)
        if canonical != self.path:
            raise ForwardContractError("FILE_PATH_NON_CANONICAL")
        _require_nonnegative_int(self.device, "device")
        _require_nonnegative_int(self.inode, "inode")
        _require_nonnegative_int(self.mount_id, "mount_id")
        _require_nonnegative_int(self.size_bytes, "size_bytes")
        _require_nonnegative_int(self.mtime_ns, "mtime_ns")
        _require_sha256(self.sha256, "sha256")

    def as_dict(self) -> Mapping[str, Any]:
        return {
            "file_id": self.file_id,
            "path": self.path,
            "device": self.device,
            "inode": self.inode,
            "mount_id": self.mount_id,
            "size_bytes": self.size_bytes,
            "mtime_ns": self.mtime_ns,
            "sha256": self.sha256,
        }

    def open_verified_readonly(
        self,
        *,
        verify_content: bool = True,
        expected_parent: Optional[FrozenDirectory] = None,
    ) -> int:
        parent = str(Path(self.path).parent)
        if expected_parent is not None and parent != expected_parent.path:
            raise ForwardContractError("FROZEN_FILE_PARENT_BINDING_MISMATCH")
        parent_fd = _open_directory_descriptor(parent, expected_parent)
        flags = (
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        try:
            fd = os.open(Path(self.path).name, flags, dir_fd=parent_fd)
        finally:
            os.close(parent_fd)
        try:
            before = os.fstat(fd)
            observed_mount_id = _fd_mount_id(fd)
            expected = (
                self.device,
                self.inode,
                self.mount_id,
                self.size_bytes,
                self.mtime_ns,
            )
            actual = (
                int(before.st_dev),
                int(before.st_ino),
                observed_mount_id,
                int(before.st_size),
                int(before.st_mtime_ns),
            )
            if (
                not stat.S_ISREG(before.st_mode)
                or int(before.st_nlink) != 1
                or actual != expected
            ):
                raise ForwardContractError("FROZEN_FILE_IDENTITY_DRIFT", self.file_id)
            if verify_content and _sha256_file_descriptor(fd) != self.sha256:
                raise ForwardContractError("FROZEN_FILE_HASH_DRIFT", self.file_id)
            after = os.fstat(fd)
            actual_after = (
                int(after.st_dev),
                int(after.st_ino),
                _fd_mount_id(fd),
                int(after.st_size),
                int(after.st_mtime_ns),
            )
            if actual_after != expected:
                raise ForwardContractError("FROZEN_FILE_CHANGED_DURING_VERIFY", self.file_id)
            os.lseek(fd, 0, os.SEEK_SET)
            return fd
        except BaseException:
            os.close(fd)
            raise


@dataclass(frozen=True)
class FrozenLmdbSource:
    source_id: str
    namespace: str
    directory: FrozenDirectory
    data_file: FrozenFile

    def __post_init__(self) -> None:
        _require_text(self.source_id, "source_id")
        if self.namespace not in {"QUERY", "VISUAL", "SUBTITLE"}:
            raise ForwardContractError("INVALID_LMDB_NAMESPACE", self.namespace)
        expected_data_path = os.path.join(self.directory.path, "data.mdb")
        if self.data_file.path != expected_data_path:
            raise ForwardContractError("LMDB_DATA_PATH_MISMATCH", self.source_id)

    def as_dict(self) -> Mapping[str, Any]:
        return {
            "source_id": self.source_id,
            "namespace": self.namespace,
            "directory": self.directory.as_dict(),
            "data_file": self.data_file.as_dict(),
        }

    @property
    def identity_sha256(self) -> str:
        return _sha256_bytes(_canonical_json_bytes(self.as_dict()))


@dataclass(frozen=True)
class FrozenCorpusIdentity:
    video_ids: Tuple[str, ...]
    durations_sec: Tuple[float, ...]
    video_ids_sha256: str
    durations_sha256: str
    feature_manifest_sha256: str

    def __post_init__(self) -> None:
        if len(self.video_ids) != EXPECTED_CORPUS_VIDEO_COUNT:
            raise ForwardContractError("CORPUS_COUNT_NOT_17435", str(len(self.video_ids)))
        if len(self.durations_sec) != len(self.video_ids):
            raise ForwardContractError("CORPUS_DURATION_COUNT_MISMATCH")
        if len(set(self.video_ids)) != len(self.video_ids):
            raise ForwardContractError("DUPLICATE_CORPUS_VIDEO_ID")
        for index, video_id in enumerate(self.video_ids):
            _require_text(video_id, "video_ids[%d]" % index)
            if "\n" in video_id:
                raise ForwardContractError("INVALID_VIDEO_ID", str(index))
        normalised_durations = []
        for index, duration in enumerate(self.durations_sec):
            value = _require_finite(duration, "durations_sec[%d]" % index)
            if value <= 0.0:
                raise ForwardContractError("NONPOSITIVE_VIDEO_DURATION", str(index))
            normalised_durations.append(value)
        _require_sha256(self.video_ids_sha256, "video_ids_sha256")
        _require_sha256(self.durations_sha256, "durations_sha256")
        _require_sha256(self.feature_manifest_sha256, "feature_manifest_sha256")
        if _line_sha256(self.video_ids) != self.video_ids_sha256:
            raise ForwardContractError("CORPUS_VIDEO_ID_HASH_MISMATCH")
        duration_rows = [
            {"duration_sec": normalised_durations[index], "video_id": video_id}
            for index, video_id in enumerate(self.video_ids)
        ]
        if _sha256_bytes(_canonical_json_bytes(duration_rows)) != self.durations_sha256:
            raise ForwardContractError("CORPUS_DURATION_HASH_MISMATCH")

    @property
    def identity_sha256(self) -> str:
        return _sha256_bytes(_canonical_json_bytes(self.as_dict()))

    def as_dict(self) -> Mapping[str, Any]:
        return {
            "video_count": len(self.video_ids),
            "video_ids_sha256": self.video_ids_sha256,
            "durations_sha256": self.durations_sha256,
            "feature_manifest_sha256": self.feature_manifest_sha256,
        }


@dataclass(frozen=True)
class FrozenQueryIdentity:
    role: str
    query_keys: Tuple[str, ...]
    query_types: Tuple[str, ...]
    query_keys_sha256: str
    query_types_sha256: str
    query_rows_sha256: str
    split_manifest_sha256: str

    def __post_init__(self) -> None:
        _require_text(self.role, "role")
        canonical = tuple(_canonical_query_key(value) for value in self.query_keys)
        if canonical != self.query_keys:
            raise ForwardContractError("QUERY_KEYS_NOT_CANONICAL")
        if not self.query_keys or len(set(self.query_keys)) != len(self.query_keys):
            raise ForwardContractError("QUERY_KEYS_EMPTY_OR_DUPLICATE")
        if (
            type(self.query_types) is not tuple
            or len(self.query_types) != len(self.query_keys)
            or any(
                not isinstance(query_type, str)
                or query_type not in _QUERY_TYPES
                for query_type in self.query_types
            )
        ):
            raise ForwardContractError("QUERY_TYPES_NOT_EXACT_FINITE_SEQUENCE")
        _require_sha256(self.query_keys_sha256, "query_keys_sha256")
        _require_sha256(self.query_types_sha256, "query_types_sha256")
        _require_sha256(self.query_rows_sha256, "query_rows_sha256")
        _require_sha256(self.split_manifest_sha256, "split_manifest_sha256")
        if _line_sha256(self.query_keys) != self.query_keys_sha256:
            raise ForwardContractError("QUERY_KEY_HASH_MISMATCH")
        if _line_sha256(self.query_types) != self.query_types_sha256:
            raise ForwardContractError("QUERY_TYPE_HASH_MISMATCH")
        rows = [
            {"query_id": query_id, "query_type": query_type}
            for query_id, query_type in zip(self.query_keys, self.query_types)
        ]
        if _sha256_bytes(_canonical_json_bytes(rows)) != self.query_rows_sha256:
            raise ForwardContractError("QUERY_ROW_HASH_MISMATCH")

    @property
    def identity_sha256(self) -> str:
        return _sha256_bytes(_canonical_json_bytes(self.as_dict()))

    def as_dict(self) -> Mapping[str, Any]:
        return {
            "role": self.role,
            "query_count": len(self.query_keys),
            "query_keys_sha256": self.query_keys_sha256,
            "query_types": list(self.query_types),
            "query_types_sha256": self.query_types_sha256,
            "query_rows_sha256": self.query_rows_sha256,
            "split_manifest_sha256": self.split_manifest_sha256,
        }


@dataclass(frozen=True)
class FrozenModelContract:
    query_dim: int = 768
    subtitle_dim: int = 768
    visual_dim: int = 4352
    hidden_dim: int = 384
    late_soft_topk: int = 8
    late_temperature: float = 0.07
    token_maxsim_weight: float = 0.25
    pooled_score_weight: float = 0.35
    late_score_weight: float = 0.65
    checkpoint_state_key: str = "model"

    def __post_init__(self) -> None:
        if (
            self.query_dim != 768
            or self.subtitle_dim != 768
            or self.visual_dim != 4352
            or self.hidden_dim != 384
        ):
            raise ForwardContractError("MODEL_DIMENSION_CONTRACT_DRIFT")
        exact_late_config = {
            "late_soft_topk": 8,
            "late_temperature": 0.07,
            "token_maxsim_weight": 0.25,
            "pooled_score_weight": 0.35,
            "late_score_weight": 0.65,
        }
        for field_name, expected in exact_late_config.items():
            observed = getattr(self, field_name)
            if isinstance(expected, int):
                if observed != expected:
                    raise ForwardContractError("MODEL_LATE_CONFIG_DRIFT", field_name)
            elif not math.isclose(
                _require_finite(observed, field_name),
                expected,
                rel_tol=0.0,
                abs_tol=0.0,
            ):
                raise ForwardContractError("MODEL_LATE_CONFIG_DRIFT", field_name)
        if self.checkpoint_state_key != "model":
            raise ForwardContractError("CHECKPOINT_STATE_KEY_DRIFT")

    def as_dict(self) -> Mapping[str, Any]:
        return {
            "query_dim": self.query_dim,
            "subtitle_dim": self.subtitle_dim,
            "visual_dim": self.visual_dim,
            "hidden_dim": self.hidden_dim,
            "late_soft_topk": self.late_soft_topk,
            "late_temperature": float(self.late_temperature),
            "token_maxsim_weight": float(self.token_maxsim_weight),
            "pooled_score_weight": float(self.pooled_score_weight),
            "late_score_weight": float(self.late_score_weight),
            "checkpoint_state_key": self.checkpoint_state_key,
            "late_interaction_enabled": True,
            "query_feature_token_count_definition": (
                QUERY_FEATURE_TOKEN_COUNT_DEFINITION
            ),
            "late_maxsim_token_count_definition": (
                LATE_MAXSIM_TOKEN_COUNT_DEFINITION
            ),
            "feature_to_late_maxsim_invariant": (
                FEATURE_TO_LATE_MAXSIM_TOKEN_COUNT_INVARIANT
            ),
            "post_forward_query_text_token_count_rule": (
                POST_FORWARD_QUERY_TEXT_TOKEN_COUNT_RULE
            ),
            "mask_structure_contract": dict(_mask_structure_contract()),
        }


@dataclass(frozen=True)
class FrozenTemporalManifest:
    clip_len_sec: float
    target_length: int
    proposal_widths: Tuple[int, ...]
    proposal_stride_rule: str
    long_sequence_policy: str
    short_sequence_policy: str
    seconds_per_clip_policy: str
    proposal_sampling_policy: str
    max_proposals_per_video: int
    metric_nms_iou_threshold: float
    metric_nms_max_results: int
    semantic_sha256: str

    @classmethod
    def build(cls, **values: Any) -> "FrozenTemporalManifest":
        return _build_semantic_dataclass(cls, "semantic_sha256", values)

    def __post_init__(self) -> None:
        if not math.isclose(_require_finite(self.clip_len_sec, "clip_len_sec"), 1.5):
            raise ForwardContractError("TEMPORAL_CLIP_LENGTH_DRIFT")
        if self.target_length != TARGET_LENGTH:
            raise ForwardContractError("TEMPORAL_TARGET_LENGTH_DRIFT")
        if self.proposal_widths != (2, 4, 6, 8, 12, 16, 24, 32, 48, 64):
            raise ForwardContractError("TEMPORAL_PROPOSAL_WIDTH_DRIFT")
        expected_text = {
            "proposal_stride_rule": "max(1,width//4)",
            "long_sequence_policy": "mean_bin_floor_ceil_resample_all_mask_true",
            "short_sequence_policy": "right_zero_pad_prefix_mask_true",
            "seconds_per_clip_policy": "duration/64_if_raw_clips_gt_64_else_1.5",
            "proposal_sampling_policy": "linspace_round_half_to_even_64",
        }
        for field_name, expected in expected_text.items():
            if getattr(self, field_name) != expected:
                raise ForwardContractError("TEMPORAL_POLICY_DRIFT", field_name)
        if self.max_proposals_per_video != MAX_PROPOSALS_PER_VIDEO:
            raise ForwardContractError("TEMPORAL_PROPOSAL_COUNT_DRIFT")
        if not math.isclose(
            _require_finite(self.metric_nms_iou_threshold, "metric_nms_iou_threshold"),
            NMS_IOU_THRESHOLD,
            rel_tol=0.0,
            abs_tol=0.0,
        ):
            raise ForwardContractError("METRIC_NMS_THRESHOLD_DRIFT")
        if self.metric_nms_max_results != METRIC_NMS_MAX_RESULTS:
            raise ForwardContractError("METRIC_NMS_LIMIT_NOT_100")
        _require_sha256(self.semantic_sha256, "temporal_semantic_sha256")
        if self.semantic_sha256 != _semantic_sha256(
            self.as_dict(), ("semantic_sha256",)
        ):
            raise ForwardContractError("TEMPORAL_MANIFEST_SEMANTIC_HASH_MISMATCH")

    def as_dict(self) -> Mapping[str, Any]:
        return {
            "clip_len_sec": float(self.clip_len_sec),
            "target_length": self.target_length,
            "proposal_widths": list(self.proposal_widths),
            "proposal_stride_rule": self.proposal_stride_rule,
            "long_sequence_policy": self.long_sequence_policy,
            "short_sequence_policy": self.short_sequence_policy,
            "seconds_per_clip_policy": self.seconds_per_clip_policy,
            "proposal_sampling_policy": self.proposal_sampling_policy,
            "max_proposals_per_video": self.max_proposals_per_video,
            "metric_nms_iou_threshold": float(self.metric_nms_iou_threshold),
            "metric_nms_max_results": self.metric_nms_max_results,
            "semantic_sha256": self.semantic_sha256,
        }


@dataclass(frozen=True)
class FrozenFeatureContract:
    query_dim: int
    visual_dim: int
    subtitle_dim: int
    query_value_max_bytes: int
    visual_value_max_bytes: int
    subtitle_value_max_bytes: int
    query_sequence_max_length: int
    video_sequence_max_length: int
    decoded_dtype: str
    encoded_dtype: str
    visual_decoder: str
    query_subtitle_decoder: str
    semantic_sha256: str

    @classmethod
    def build(cls, **values: Any) -> "FrozenFeatureContract":
        return _build_semantic_dataclass(cls, "semantic_sha256", values)

    def __post_init__(self) -> None:
        expected = {
            "query_dim": 768,
            "visual_dim": 4352,
            "subtitle_dim": 768,
            "query_value_max_bytes": QUERY_VALUE_MAX_BYTES,
            "visual_value_max_bytes": VISUAL_VALUE_MAX_BYTES,
            "subtitle_value_max_bytes": SUBTITLE_VALUE_MAX_BYTES,
            "query_sequence_max_length": QUERY_SEQUENCE_MAX_LENGTH,
            "video_sequence_max_length": VIDEO_SEQUENCE_MAX_LENGTH,
            "decoded_dtype": "float32",
            "encoded_dtype": "float32",
            "visual_decoder": "msgpack_numpy_features_exact",
            "query_subtitle_decoder": "npz_allow_pickle_false_features_exact",
        }
        for field_name, expected_value in expected.items():
            if getattr(self, field_name) != expected_value:
                raise ForwardContractError("FEATURE_CONTRACT_DRIFT", field_name)
        _require_sha256(self.semantic_sha256, "feature_semantic_sha256")
        if self.semantic_sha256 != _semantic_sha256(
            self.as_dict(), ("semantic_sha256",)
        ):
            raise ForwardContractError("FEATURE_MANIFEST_SEMANTIC_HASH_MISMATCH")

    def as_dict(self) -> Mapping[str, Any]:
        return {
            "query_dim": self.query_dim,
            "visual_dim": self.visual_dim,
            "subtitle_dim": self.subtitle_dim,
            "query_value_max_bytes": self.query_value_max_bytes,
            "visual_value_max_bytes": self.visual_value_max_bytes,
            "subtitle_value_max_bytes": self.subtitle_value_max_bytes,
            "query_sequence_max_length": self.query_sequence_max_length,
            "video_sequence_max_length": self.video_sequence_max_length,
            "decoded_dtype": self.decoded_dtype,
            "encoded_dtype": self.encoded_dtype,
            "visual_decoder": self.visual_decoder,
            "query_subtitle_decoder": self.query_subtitle_decoder,
            "semantic_sha256": self.semantic_sha256,
        }


@dataclass(frozen=True)
class FrozenGpu0Identity:
    physical_index: int
    uuid: str
    pci_bus_id: str
    cuda_visible_devices: str
    logical_device: str
    gpu1_action_count: int
    semantic_sha256: str

    @classmethod
    def build(cls, **values: Any) -> "FrozenGpu0Identity":
        return _build_semantic_dataclass(cls, "semantic_sha256", values)

    def __post_init__(self) -> None:
        if self.physical_index != 0 or self.logical_device != EXPECTED_DEVICE:
            raise ForwardContractError("GPU0_PHYSICAL_MAPPING_DRIFT")
        if self.cuda_visible_devices != "0":
            raise ForwardContractError("CUDA_VISIBLE_DEVICES_NOT_EXACT_GPU0")
        for field_name in ("uuid", "pci_bus_id"):
            _require_text(getattr(self, field_name), field_name)
        if self.gpu1_action_count != 0:
            raise ForwardContractError("GPU1_ACTION_FORBIDDEN")
        _require_sha256(self.semantic_sha256, "gpu0_semantic_sha256")
        if self.semantic_sha256 != _semantic_sha256(
            self.as_dict(), ("semantic_sha256",)
        ):
            raise ForwardContractError("GPU0_MANIFEST_SEMANTIC_HASH_MISMATCH")

    def as_dict(self) -> Mapping[str, Any]:
        return {
            "physical_index": self.physical_index,
            "uuid": self.uuid,
            "pci_bus_id": self.pci_bus_id,
            "cuda_visible_devices": self.cuda_visible_devices,
            "logical_device": self.logical_device,
            "gpu1_action_count": self.gpu1_action_count,
            "semantic_sha256": self.semantic_sha256,
        }


@dataclass(frozen=True)
class FrozenForwardManifest:
    goal_id: str
    authority_id: str
    stage: str
    purpose: str
    query_identity: FrozenQueryIdentity
    corpus_identity: FrozenCorpusIdentity
    checkpoint_sha256: str
    model_contract: FrozenModelContract
    query_source_id: str
    visual_source_id: str
    subtitle_source_id: str
    output_root_id: str
    cache_root_id: str
    attempt_id: str
    temporal_manifest: FrozenTemporalManifest
    feature_contract: FrozenFeatureContract
    gpu0_identity: FrozenGpu0Identity
    code_source_sha256: Tuple[Tuple[str, str], ...]
    metric_schema_sha256: str
    evaluator_schema_sha256: str
    environment_sha256: str
    declared_semantic_sha256: str
    broad_topk: int = POOLED_TOPK
    late_topk: int = LATE_TOPK
    full_model_candidates: int = FULL_MODEL_CANDIDATES
    target_length: int = TARGET_LENGTH
    max_proposals_per_video: int = MAX_PROPOSALS_PER_VIDEO
    teacher_candidate_count: int = 0
    teacher_forward_count: int = 0
    gt_support_count: int = 0
    gt_append_count: int = 0
    optimizer_update_count: int = 0
    posthoc_temperature_refit_count: int = 0

    @classmethod
    def build(cls, **values: Any) -> "FrozenForwardManifest":
        return _build_semantic_dataclass(cls, "declared_semantic_sha256", values)

    def __post_init__(self) -> None:
        _require_text(self.goal_id, "goal_id")
        _require_text(self.authority_id, "authority_id")
        if self.stage not in _STAGE_PURPOSES:
            raise ForwardContractError("UNAUTHORIZED_FORWARD_STAGE", self.stage)
        if self.purpose != _STAGE_PURPOSES[self.stage]:
            raise ForwardContractError("FORWARD_PURPOSE_STAGE_MISMATCH")
        for field_name in (
            "query_source_id",
            "visual_source_id",
            "subtitle_source_id",
            "output_root_id",
            "cache_root_id",
            "attempt_id",
        ):
            _require_text(getattr(self, field_name), field_name)
        _require_sha256(self.checkpoint_sha256, "checkpoint_sha256")
        if self.stage == "G3_F0_A":
            if self.query_identity.role != "calib_select":
                raise ForwardContractError("F0A_QUERY_ROLE_MISMATCH")
            if len(self.query_identity.query_keys) != EXPECTED_F0A_QUERY_COUNT:
                raise ForwardContractError("F0A_QUERY_COUNT_NOT_8677")
        if self.stage == "G5_F0_B":
            if self.query_identity.role != "train_fit_route_dev":
                raise ForwardContractError("F0B_QUERY_ROLE_MISMATCH")
            if len(self.query_identity.query_keys) < 2_048:
                raise ForwardContractError("F0B_ROUTE_DEV_TOO_SMALL")
        expected_values = {
            "broad_topk": POOLED_TOPK,
            "late_topk": LATE_TOPK,
            "full_model_candidates": FULL_MODEL_CANDIDATES,
            "target_length": TARGET_LENGTH,
            "max_proposals_per_video": MAX_PROPOSALS_PER_VIDEO,
        }
        for field_name, expected in expected_values.items():
            if getattr(self, field_name) != expected:
                raise ForwardContractError("FORWARD_SCALE_CONTRACT_DRIFT", field_name)
        required_sources = {
            "a4_forward.py",
            "a4_security.py",
        } | set(_REQUIRED_PACKAGE_MODULE_BY_SOURCE) | set(
            _APPROVED_BUSINESS_MODULE_BY_SOURCE
        )
        if tuple(sorted(set(self.code_source_sha256))) != self.code_source_sha256:
            raise ForwardContractError("CODE_SOURCE_HASHES_NOT_CANONICAL")
        if {name for name, _sha in self.code_source_sha256} != required_sources:
            raise ForwardContractError("CODE_SOURCE_HASH_SET_INCOMPLETE")
        for name, digest in self.code_source_sha256:
            _require_text(name, "code_source_name")
            _require_sha256(digest, "code_source_sha256")
        for field_name in (
            "metric_schema_sha256",
            "evaluator_schema_sha256",
            "environment_sha256",
            "declared_semantic_sha256",
        ):
            _require_sha256(getattr(self, field_name), field_name)
        zero_fields = (
            "teacher_candidate_count",
            "teacher_forward_count",
            "gt_support_count",
            "gt_append_count",
            "optimizer_update_count",
            "posthoc_temperature_refit_count",
        )
        for field_name in zero_fields:
            if getattr(self, field_name) != 0:
                raise ForwardContractError("FORBIDDEN_SUPPORT_NONZERO", field_name)
        if self.declared_semantic_sha256 != _semantic_sha256(
            self.as_dict(), ("declared_semantic_sha256",)
        ):
            raise ForwardContractError("FORWARD_MANIFEST_SEMANTIC_HASH_MISMATCH")

    @property
    def manifest_sha256(self) -> str:
        return self.declared_semantic_sha256

    @property
    def input_identity_sha256(self) -> str:
        return _sha256_bytes(
            _canonical_json_bytes(
                {
                    "checkpoint_sha256": self.checkpoint_sha256,
                    "corpus_identity_sha256": self.corpus_identity.identity_sha256,
                    "query_identity_sha256": self.query_identity.identity_sha256,
                    "query_types_sha256": self.query_identity.query_types_sha256,
                    "query_rows_sha256": self.query_identity.query_rows_sha256,
                    "model_contract": self.model_contract.as_dict(),
                    "stage": self.stage,
                    "purpose": self.purpose,
                }
            )
        )

    def as_dict(self) -> Mapping[str, Any]:
        return {
            "schema_version": FORWARD_MANIFEST_SCHEMA_VERSION,
            "goal_id": self.goal_id,
            "authority_id": self.authority_id,
            "stage": self.stage,
            "purpose": self.purpose,
            "query_identity": self.query_identity.as_dict(),
            "corpus_identity": self.corpus_identity.as_dict(),
            "checkpoint_sha256": self.checkpoint_sha256,
            "model_contract": self.model_contract.as_dict(),
            "query_source_id": self.query_source_id,
            "visual_source_id": self.visual_source_id,
            "subtitle_source_id": self.subtitle_source_id,
            "output_root_id": self.output_root_id,
            "cache_root_id": self.cache_root_id,
            "attempt_id": self.attempt_id,
            "temporal_manifest": self.temporal_manifest.as_dict(),
            "feature_contract": self.feature_contract.as_dict(),
            "gpu0_identity": self.gpu0_identity.as_dict(),
            "code_source_sha256": [list(item) for item in self.code_source_sha256],
            "metric_schema_sha256": self.metric_schema_sha256,
            "evaluator_schema_sha256": self.evaluator_schema_sha256,
            "environment_sha256": self.environment_sha256,
            "declared_semantic_sha256": self.declared_semantic_sha256,
            "broad_topk": self.broad_topk,
            "late_topk": self.late_topk,
            "full_model_candidates": self.full_model_candidates,
            "target_length": self.target_length,
            "max_proposals_per_video": self.max_proposals_per_video,
            "teacher_candidate_count": self.teacher_candidate_count,
            "teacher_forward_count": self.teacher_forward_count,
            "gt_support_count": self.gt_support_count,
            "gt_append_count": self.gt_append_count,
            "optimizer_update_count": self.optimizer_update_count,
            "posthoc_temperature_refit_count": self.posthoc_temperature_refit_count,
        }


def _a4_control_module() -> Any:
    """Load the compact control implementation only at the execution boundary."""

    return importlib.import_module("blueprint_e2e_v2.c28f_v5.runtime_control")


def _code_source_path(source_name: str) -> Path:
    module_path = Path(os.path.abspath(__file__))
    if source_name in {"a4_forward.py", "a4_security.py"}:
        path = module_path.parent / source_name
    else:
        path = module_path.parent.parent / source_name
    canonical, _pieces = _canonical_absolute_path(str(path))
    return Path(canonical)


def _capture_approved_code_source(
    source_name: str,
    approved_sha256: str,
) -> Tuple[FrozenDirectory, FrozenFile]:
    path = _code_source_path(source_name)
    parent = FrozenDirectory.capture(
        "code-parent-" + _sha256_bytes(str(path.parent).encode("utf-8"))[:32],
        str(path.parent),
    )
    parent_fd = _open_directory_descriptor(parent.path, parent)
    try:
        fd = os.open(
            path.name,
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=parent_fd,
        )
    finally:
        os.close(parent_fd)
    try:
        before = os.fstat(fd)
        digest = _sha256_file_descriptor(fd)
        after = os.fstat(fd)
        identity = (
            int(before.st_dev),
            int(before.st_ino),
            _fd_mount_id(fd),
            int(before.st_size),
            int(before.st_mtime_ns),
        )
        if (
            not stat.S_ISREG(before.st_mode)
            or int(before.st_nlink) != 1
            or identity
            != (
                int(after.st_dev),
                int(after.st_ino),
                _fd_mount_id(fd),
                int(after.st_size),
                int(after.st_mtime_ns),
            )
            or digest != approved_sha256
        ):
            raise ForwardContractError("CODE_SOURCE_HASH_OR_IDENTITY_DRIFT", source_name)
        frozen = FrozenFile(
            file_id="code-" + source_name,
            path=str(path),
            device=identity[0],
            inode=identity[1],
            mount_id=identity[2],
            size_bytes=identity[3],
            mtime_ns=identity[4],
            sha256=approved_sha256,
        )
    finally:
        os.close(fd)
    return parent, frozen


def _require_exact_forward_control(control: Any) -> Any:
    """Reject protocol lookalikes and accept only the reviewed A4 adapter class."""

    module = _a4_control_module()
    if type(control) is not module.A4EvalForwardControlAdapter:
        raise ForwardContractError("EXACT_A4_FORWARD_CONTROL_REQUIRED")
    return module


def _validate_controller_issued_authority(
    authority: "FrozenInputAuthority",
    control: Any,
) -> Mapping[str, Any]:
    """Require the exact controller-built and process-registered authority."""

    _require_exact_forward_control(control)
    try:
        payload = control.validate_controller_forward_authority(
            authority=authority,
        )
    except BaseException as error:
        raise ForwardContractError(
            "CONTROLLER_ISSUED_FORWARD_AUTHORITY_REQUIRED",
            type(error).__name__,
        ) from None
    exact_fields = {
        "schema_version",
        "status",
        "goal_id",
        "attempt_id",
        "authority_id",
        "stage",
        "purpose",
        "authority_handle_sha256",
        "reservation_input_capability_sha256",
        "authority_binding_sha256",
        "forward_manifest_sha256",
        "input_identity_sha256",
        "query_identity_sha256",
        "query_keys_sha256",
        "query_types_sha256",
        "query_rows_sha256",
        "split_manifest_sha256",
        "corpus_identity_sha256",
        "video_ids_sha256",
        "durations_sha256",
        "query_source_descriptor_identity_sha256",
        "query_source_selected_projection_sha256",
        "query_source_index_binding_sha256",
        "query_source_index_bootstrap_receipt_sha256",
        "query_source_file_sha256",
        "query_identity_source_access_receipt_sha256",
        "video_meta_source_file_sha256",
        "controller_authority_binding_sha256",
    }
    manifest = authority.manifest
    expected = {
        "schema_version": "c28f_a4_controller_forward_authority_binding_v1",
        "status": "CONTROLLER_BUILT_REGISTERED_AUTHORITY",
        "goal_id": manifest.goal_id,
        "attempt_id": manifest.attempt_id,
        "authority_id": manifest.authority_id,
        "stage": manifest.stage,
        "purpose": manifest.purpose,
        "authority_binding_sha256": authority.authority_binding_sha256,
        "forward_manifest_sha256": manifest.manifest_sha256,
        "input_identity_sha256": manifest.input_identity_sha256,
        "query_identity_sha256": manifest.query_identity.identity_sha256,
        "query_keys_sha256": manifest.query_identity.query_keys_sha256,
        "query_types_sha256": manifest.query_identity.query_types_sha256,
        "query_rows_sha256": manifest.query_identity.query_rows_sha256,
        "split_manifest_sha256": manifest.query_identity.split_manifest_sha256,
        "corpus_identity_sha256": manifest.corpus_identity.identity_sha256,
        "video_ids_sha256": manifest.corpus_identity.video_ids_sha256,
        "durations_sha256": manifest.corpus_identity.durations_sha256,
    }
    if (
        not isinstance(payload, Mapping)
        or set(payload) != exact_fields
        or any(payload.get(name) != value for name, value in expected.items())
        or payload.get("controller_authority_binding_sha256")
        != _semantic_sha256(
            payload,
            ("controller_authority_binding_sha256",),
        )
    ):
        raise ForwardContractError(
            "CONTROLLER_FORWARD_AUTHORITY_BINDING_DRIFT"
        )
    for field_name in (
        "authority_handle_sha256",
        "reservation_input_capability_sha256",
        "query_source_descriptor_identity_sha256",
        "query_source_selected_projection_sha256",
        "query_source_index_binding_sha256",
        "query_source_index_bootstrap_receipt_sha256",
        "query_source_file_sha256",
        "query_identity_source_access_receipt_sha256",
        "video_meta_source_file_sha256",
        "controller_authority_binding_sha256",
    ):
        _require_sha256(payload.get(field_name), field_name)
    return json.loads(_canonical_json_bytes(dict(payload)).decode("utf-8"))


_CAPABILITY_FACTORY = object()
_RUN_ARTIFACT_ROOT_FACTORY = object()


class InputCapability:
    """Opaque, single-issued handle backed by an external committed control plane."""

    __slots__ = (
        "capability_id",
        "goal_id",
        "authority_id",
        "run_id",
        "token_id",
        "eval_id",
        "purpose",
        "stage",
        "forward_manifest_sha256",
        "input_identity_sha256",
        "authority_binding_sha256",
        "token_snapshot_sha256",
        "gpu0_identity_sha256",
        "control_chain_sha256",
        "running_commit_sha256",
        "running_capability_sha256",
        "reservation_input_capability_sha256",
        "controller_authority_binding_sha256",
        "_issuer_nonce",
        "_authority",
        "_control",
        "_running_control_capability",
        "_run_receipt_sha256",
        "_run_receipt_path",
        "_run_receipt_parent",
        "_run_chunk_parent",
    )

    def __init__(self, factory: object, **values: Any) -> None:
        if factory is not _CAPABILITY_FACTORY:
            raise ForwardContractError("CAPABILITY_CONSTRUCTION_FORBIDDEN")
        for name in self.__slots__:
            object.__setattr__(self, name, values[name])

    def __setattr__(self, _name: str, _value: Any) -> None:
        raise ForwardContractError("CAPABILITY_MUTATION_FORBIDDEN")

    def assert_running_committed(self) -> None:
        module = _require_exact_forward_control(self._control)
        running = module.validate_eval_running_capability(
            self._running_control_capability
        )
        if self._running_control_capability.capability_sha256 != (
            self.running_capability_sha256
        ):
            raise ForwardContractError("RUNNING_CAPABILITY_HASH_DRIFT")
        expected_running = {
            "goal_id": self.goal_id,
            "run_id": self.run_id,
            "token_id": self.token_id,
            "eval_id": self.eval_id,
            "attempt_id": self._authority.manifest.attempt_id,
            "running_event_sha256": self.running_commit_sha256,
            "reservation_input_capability_sha256": (
                self.reservation_input_capability_sha256
            ),
        }
        if any(running.get(name) != value for name, value in expected_running.items()):
            raise ForwardContractError("RUNNING_CAPABILITY_BINDING_DRIFT")
        observed = self._control.assert_running_state_event_committed()
        if not isinstance(observed, str) or not secrets.compare_digest(
            observed, self.running_commit_sha256
        ):
            raise ForwardContractError("RUNNING_CONTROL_COMMIT_DRIFT")

    def _assert_controller_derived_run_artifact_roots(
        self,
        parent: "FrozenDirectory",
        chunk_parent: "FrozenDirectory",
    ) -> None:
        expected_run_path = os.path.join(
            self._authority.output_root.path,
            self.stage,
            self._authority.manifest.attempt_id,
            self.run_id,
        )
        expected_chunk_path = os.path.join(expected_run_path, "query_chunks")
        expected_run_root_id = (
            "child-" + _sha256_bytes(expected_run_path.encode("utf-8"))[:32]
        )
        expected_chunk_root_id = (
            "child-" + _sha256_bytes(expected_chunk_path.encode("utf-8"))[:32]
        )
        if (
            type(parent) is not FrozenDirectory
            or type(chunk_parent) is not FrozenDirectory
            or parent.path != expected_run_path
            or parent.root_id != expected_run_root_id
            or chunk_parent.path != expected_chunk_path
            or chunk_parent.root_id != expected_chunk_root_id
            or (parent.device, parent.mount_id)
            != (
                self._authority.output_root.device,
                self._authority.output_root.mount_id,
            )
            or (chunk_parent.device, chunk_parent.mount_id)
            != (parent.device, parent.mount_id)
        ):
            raise ForwardContractError("RUN_RECEIPT_PARENT_BINDING_INVALID")
        self._authority.output_root.verify()
        parent.verify()
        chunk_parent.verify()

    def _bind_controller_derived_run_artifact_roots(
        self,
        factory: object,
        parent: "FrozenDirectory",
        chunk_parent: "FrozenDirectory",
    ) -> None:
        if factory is not _RUN_ARTIFACT_ROOT_FACTORY:
            raise ForwardContractError("RUN_ARTIFACT_ROOT_BINDING_FORBIDDEN")
        self._assert_controller_derived_run_artifact_roots(
            parent,
            chunk_parent,
        )
        if self._run_receipt_parent is not None or self._run_chunk_parent is not None:
            if (
                self._run_receipt_parent is not parent
                or self._run_chunk_parent is not chunk_parent
            ):
                raise ForwardContractError("RUN_ARTIFACT_ROOT_REBIND_FORBIDDEN")
            return
        object.__setattr__(self, "_run_receipt_parent", parent)
        object.__setattr__(self, "_run_chunk_parent", chunk_parent)

    def _bind_persisted_run_receipt(
        self,
        receipt_sha256: str,
        path: str,
        parent: "FrozenDirectory",
        chunk_parent: "FrozenDirectory",
    ) -> None:
        _require_sha256(receipt_sha256, "run_receipt_sha256")
        _canonical_absolute_path(path)
        self._assert_controller_derived_run_artifact_roots(
            parent,
            chunk_parent,
        )
        if (
            str(Path(path).parent) != parent.path
            or str(path) != os.path.join(parent.path, "forward_run_receipt.json")
        ):
            raise ForwardContractError("RUN_RECEIPT_PARENT_BINDING_INVALID")
        if self._run_receipt_path is not None and self._run_receipt_path != path:
            raise ForwardContractError("RUN_RECEIPT_CAPABILITY_REBIND_FORBIDDEN")
        if self._run_receipt_parent is not parent:
            raise ForwardContractError("RUN_RECEIPT_PARENT_REBIND_FORBIDDEN")
        if self._run_chunk_parent is not chunk_parent:
            raise ForwardContractError("RUN_CHUNK_PARENT_REBIND_FORBIDDEN")
        object.__setattr__(self, "_run_receipt_sha256", receipt_sha256)
        object.__setattr__(self, "_run_receipt_path", path)

    def assert_io_ready(self) -> None:
        self.assert_running_committed()
        if (
            self._run_receipt_sha256 is None
            or self._run_receipt_path is None
            or self._run_receipt_parent is None
            or self._run_chunk_parent is None
        ):
            raise ForwardContractError("PERSISTED_RUNNING_RECEIPT_REQUIRED_PREOPEN")
        receipt = _read_canonical_json(
            Path(self._run_receipt_path),
            expected_parent=self._run_receipt_parent,
        )
        if receipt.get("receipt_sha256") != self._run_receipt_sha256:
            raise ForwardContractError("PERSISTED_RUNNING_RECEIPT_DRIFT")
        if receipt.get("receipt_sha256") != _semantic_sha256(
            receipt, ("receipt_sha256",)
        ):
            raise ForwardContractError("PERSISTED_RUNNING_RECEIPT_HASH_MISMATCH")
        if receipt.get("status") != "RUNNING":
            raise ForwardContractError("PERSISTED_RUNNING_STATE_REQUIRED_PREOPEN")
        expected = {
            "goal_id": self.goal_id,
            "authority_id": self.authority_id,
            "run_id": self.run_id,
            "stage": self.stage,
            "purpose": self.purpose,
            "token_id": self.token_id,
            "eval_id": self.eval_id,
            "forward_manifest_sha256": self.forward_manifest_sha256,
            "input_identity_sha256": self.input_identity_sha256,
            "query_identity_sha256": (
                self._authority.manifest.query_identity.identity_sha256
            ),
            "query_types_sha256": (
                self._authority.manifest.query_identity.query_types_sha256
            ),
            "query_rows_sha256": (
                self._authority.manifest.query_identity.query_rows_sha256
            ),
            "authority_binding_sha256": self.authority_binding_sha256,
            "token_snapshot_sha256": self.token_snapshot_sha256,
            "gpu0_identity_sha256": self.gpu0_identity_sha256,
            "running_capability_sha256": self.running_capability_sha256,
            "reservation_input_capability_sha256": (
                self.reservation_input_capability_sha256
            ),
            "controller_authority_binding_sha256": (
                self.controller_authority_binding_sha256
            ),
        }
        if any(receipt.get(name) != value for name, value in expected.items()):
            raise ForwardContractError("PERSISTED_RUNNING_RECEIPT_BINDING_MISMATCH")

    def _record_observation(
        self, event_type: str, counters: Mapping[str, int], details: Mapping[str, Any]
    ) -> str:
        return self._authority._record_forward_observation(
            self,
            event_type=event_type,
            counters=counters,
            details=details,
        )

    def _mark_model_forward_started(self, event_type: str) -> str:
        return self._authority._mark_model_forward_started(self, event_type)


class FrozenInputAuthority:
    """Runtime authority; it issues non-serializable capabilities after token checks."""

    def __init__(
        self,
        *,
        manifest: FrozenForwardManifest,
        checkpoint: FrozenFile,
        lmdb_sources: Sequence[FrozenLmdbSource],
        output_root: FrozenDirectory,
        cache_root: FrozenDirectory,
        forbidden_roots: Sequence[str],
    ) -> None:
        self.manifest = manifest
        self.checkpoint = checkpoint
        self._code_sources = {
            name: _capture_approved_code_source(name, digest)
            for name, digest in manifest.code_source_sha256
        }
        self._checkpoint_parent = FrozenDirectory.capture(
            "checkpoint-parent-"
            + _sha256_bytes(str(Path(checkpoint.path).parent).encode("utf-8"))[:32],
            str(Path(checkpoint.path).parent),
        )
        self.output_root = output_root
        self.cache_root = cache_root
        self.sources = {source.source_id: source for source in lmdb_sources}
        if len(self.sources) != len(tuple(lmdb_sources)):
            raise ForwardContractError("DUPLICATE_LMDB_SOURCE_ID")
        required_sources = {
            manifest.query_source_id: "QUERY",
            manifest.visual_source_id: "VISUAL",
            manifest.subtitle_source_id: "SUBTITLE",
        }
        for source_id, namespace in required_sources.items():
            source = self.sources.get(source_id)
            if source is None or source.namespace != namespace:
                raise ForwardContractError("REQUIRED_LMDB_SOURCE_MISSING", source_id)
        if checkpoint.sha256 != manifest.checkpoint_sha256:
            raise ForwardContractError("CHECKPOINT_MANIFEST_HASH_MISMATCH")
        if output_root.root_id != manifest.output_root_id:
            raise ForwardContractError("OUTPUT_ROOT_ID_MISMATCH")
        if cache_root.root_id != manifest.cache_root_id:
            raise ForwardContractError("CACHE_ROOT_ID_MISMATCH")
        if _paths_overlap(output_root.path, cache_root.path):
            raise ForwardContractError("OUTPUT_CACHE_ROOT_OVERLAP")
        if (output_root.device, output_root.inode) == (
            cache_root.device,
            cache_root.inode,
        ):
            raise ForwardContractError("OUTPUT_CACHE_PHYSICAL_ALIAS")
        canonical_forbidden = []
        for path in forbidden_roots:
            canonical, _pieces = _canonical_absolute_path(path)
            canonical_forbidden.append(canonical)
        for writable in (output_root.path, cache_root.path):
            for forbidden in canonical_forbidden:
                if _paths_overlap(writable, forbidden):
                    raise ForwardContractError("WRITE_ROOT_OVERLAPS_FORBIDDEN_ROOT")
            if _paths_overlap(writable, checkpoint.path):
                raise ForwardContractError("WRITE_ROOT_OVERLAPS_CHECKPOINT")
            for _source_name, (_parent, frozen) in self._code_sources.items():
                if _paths_overlap(writable, frozen.path):
                    raise ForwardContractError("WRITE_ROOT_OVERLAPS_CODE_SOURCE")
            for source in self.sources.values():
                if _paths_overlap(writable, source.directory.path):
                    raise ForwardContractError("WRITE_ROOT_OVERLAPS_FEATURE_SOURCE")
        writable_identities = {
            (output_root.device, output_root.inode),
            (cache_root.device, cache_root.inode),
        }
        source_identities = {(checkpoint.device, checkpoint.inode)}
        source_identities.update(
            (frozen.device, frozen.inode)
            for _parent, frozen in self._code_sources.values()
        )
        source_identities.update(
            (source.directory.device, source.directory.inode)
            for source in self.sources.values()
        )
        source_identities.update(
            (source.data_file.device, source.data_file.inode)
            for source in self.sources.values()
        )
        if writable_identities.intersection(source_identities):
            raise ForwardContractError("WRITE_ROOT_PHYSICAL_SOURCE_ALIAS")
        self.forbidden_roots = tuple(sorted(set(canonical_forbidden)))
        self._issuer_nonce = object()
        self._capability_issued = False
        self._issued_capability_id: Optional[str] = None
        self._issued_capability: Optional[InputCapability] = None
        self._issued_control: Any = None
        self._issued_capability_fields: Mapping[str, Any] = {}
        self._loaded_model_handle: Optional[LoadedModelHandle] = None
        self._encoded_corpus_handle: Optional[EncodedCorpusHandle] = None
        self._result_issuer_nonce = object()
        self._gpu0_runtime_lease: Optional[Gpu0RuntimeLease] = None
        self._verified_sources: set[str] = set()
        self._allowed_key_digest_cache: dict[str, frozenset[str]] = {}
        self._persisted_batch_receipts: dict[str, Any] = {}
        self._observation_rows: list[Mapping[str, Any]] = []
        self._pending_observation_summary: dict[str, Any] = {}
        self._model_forward_started_sha256: Optional[str] = None
        self._verified_package_modules: dict[str, Tuple[Any, ...]] = {}
        self._verified_business_modules: dict[str, Tuple[Any, ...]] = {}
        security = importlib.import_module("blueprint_e2e_v2.c28f_v5.a4_security")
        self._shared_security = security
        self._shared_key_ledger = security.EvidenceLedger(
            scope_id="%s-forward-lmdb-key-batches" % manifest.stage,
            goal_id=manifest.goal_id,
            authority_id=manifest.authority_id,
        )
        self._shared_key_manifests = self._build_shared_key_manifests()

    @property
    def observation_ledger_sha256(self) -> str:
        return _sha256_bytes(_canonical_json_bytes(self._observation_rows))

    @property
    def observation_event_count(self) -> int:
        return len(self._observation_rows)

    def _append_forward_observation_row(
        self,
        capability: InputCapability,
        *,
        event_type: str,
        counters: Mapping[str, int],
        details: Mapping[str, Any],
    ) -> str:
        previous = (
            None
            if not self._observation_rows
            else self._observation_rows[-1]["event_sha256"]
        )
        row = {
            "schema_version": "c28f_v5_a4_forward_observation_v1",
            "run_id": capability.run_id,
            "token_id": capability.token_id,
            "eval_id": capability.eval_id,
            "seq": len(self._observation_rows) + 1,
            "event_type": event_type,
            "counters": dict(counters),
            "details": json.loads(
                _canonical_json_bytes(dict(details)).decode("utf-8")
            ),
            "previous_event_sha256": previous,
        }
        row["event_sha256"] = _semantic_sha256(row)
        self._observation_rows.append(row)
        return str(row["event_sha256"])

    def _persist_forward_observation_ledger(
        self, capability: InputCapability
    ) -> None:
        if capability._run_receipt_parent is None:
            return
        observation_path = Path(capability._run_receipt_parent.path) / (
            "forward_observations.json"
        )
        payload = {
            "schema_version": "c28f_v5_a4_forward_observation_ledger_v1",
            "run_id": capability.run_id,
            "capability_id": capability.capability_id,
            "event_count": len(self._observation_rows),
            "rows": list(self._observation_rows),
        }
        payload["ledger_sha256"] = _semantic_sha256(payload)
        _atomic_write_bytes(
            observation_path,
            _canonical_json_bytes(payload),
            replace=True,
            expected_parent=capability._run_receipt_parent,
        )

    def _flush_forward_observation_summary(
        self, capability: InputCapability, *, persist: bool
    ) -> Optional[str]:
        pending = self._pending_observation_summary
        event_sha256: Optional[str] = None
        if pending:
            event_sha256 = self._append_forward_observation_row(
                capability,
                event_type="AGGREGATED_FORWARD_OBSERVATIONS",
                counters=pending["counters"],
                details={
                    "schema_version": "c28f_v5_aggregated_forward_observations_v1",
                    "aggregated_event_count": pending["event_count"],
                    "event_type_counts": dict(
                        sorted(pending["event_type_counts"].items())
                    ),
                    "ordered_detail_chain_sha256": pending[
                        "ordered_detail_chain_sha256"
                    ],
                },
            )
            self._pending_observation_summary = {}
        if persist:
            self._persist_forward_observation_ledger(capability)
        return event_sha256

    def flush_forward_observations(self, capability: InputCapability) -> None:
        self._assert_capability_identity(capability)
        self._flush_forward_observation_summary(capability, persist=True)

    def _record_forward_observation(
        self,
        capability: InputCapability,
        *,
        event_type: str,
        counters: Mapping[str, int],
        details: Mapping[str, Any],
    ) -> str:
        self._assert_capability_identity(capability)
        _require_text(event_type, "event_type")
        counter_fields = {
            "content_open_count",
            "content_bytes_read",
            "lmdb_transaction_open_count",
            "model_forward_count",
        }
        if set(counters) != counter_fields or any(
            type(value) is not int or value < 0 for value in counters.values()
        ):
            raise ForwardContractError("FORWARD_OBSERVATION_COUNTERS_INVALID")
        if not isinstance(details, Mapping):
            raise ForwardContractError("FORWARD_OBSERVATION_DETAILS_INVALID")
        aggregated_types = {
            "LMDB_VALUE_READ",
            "LMDB_VALUE_DECODED",
            "QUERY_ENCODER_FORWARD",
            "POOLED_FULL_CORPUS_FORWARD",
            "LATE_BROAD1000_FORWARD",
            "FULL_DOWNSTREAM_FROM_ENCODED200_FORWARD",
        }
        if event_type in aggregated_types:
            if not self._pending_observation_summary:
                self._pending_observation_summary = {
                    "event_count": 0,
                    "event_type_counts": {},
                    "counters": {name: 0 for name in counter_fields},
                    "ordered_detail_chain_sha256": _sha256_bytes(
                        _canonical_json_bytes([])
                    ),
                }
            pending = self._pending_observation_summary
            pending["event_count"] += 1
            pending["event_type_counts"][event_type] = (
                pending["event_type_counts"].get(event_type, 0) + 1
            )
            for name in counter_fields:
                pending["counters"][name] += counters[name]
            pending["ordered_detail_chain_sha256"] = _sha256_bytes(
                _canonical_json_bytes(
                    {
                        "previous_sha256": pending[
                            "ordered_detail_chain_sha256"
                        ],
                        "event_type": event_type,
                        "counters": dict(counters),
                        "details": dict(details),
                    }
                )
            )
            return str(pending["ordered_detail_chain_sha256"])
        self._flush_forward_observation_summary(capability, persist=False)
        event_sha256 = self._append_forward_observation_row(
            capability,
            event_type=event_type,
            counters=counters,
            details=details,
        )
        self._persist_forward_observation_ledger(capability)
        return event_sha256

    def _mark_model_forward_started(
        self, capability: InputCapability, event_type: str
    ) -> str:
        self._assert_capability_identity(capability)
        _require_text(event_type, "event_type")
        if self._model_forward_started_sha256 is not None:
            return self._model_forward_started_sha256
        parent = capability._run_receipt_parent
        if parent is None:
            raise ForwardContractError("RUN_ARTIFACT_ROOT_REQUIRED_BEFORE_FORWARD")
        path = Path(parent.path) / "model_forward_started.json"
        if path.exists():
            record = _read_canonical_json(path, expected_parent=parent)
            expected = {
                "schema_version": "c28f_v5_model_forward_started_v1",
                "run_id": capability.run_id,
                "eval_id": capability.eval_id,
                "token_id": capability.token_id,
                "capability_id": capability.capability_id,
            }
            if (
                set(record)
                != {*expected, "first_forward_event_type", "record_sha256"}
                or any(record.get(name) != value for name, value in expected.items())
                or not isinstance(record.get("first_forward_event_type"), str)
                or not record["first_forward_event_type"]
                or record.get("record_sha256")
                != _semantic_sha256(record, ("record_sha256",))
            ):
                raise ForwardContractError("MODEL_FORWARD_STARTED_RECORD_DRIFT")
            self._model_forward_started_sha256 = str(record["record_sha256"])
            return str(record["record_sha256"])
        base = {
            "schema_version": "c28f_v5_model_forward_started_v1",
            "run_id": capability.run_id,
            "eval_id": capability.eval_id,
            "token_id": capability.token_id,
            "capability_id": capability.capability_id,
            "first_forward_event_type": event_type,
        }
        record = {**base, "record_sha256": _semantic_sha256(base)}
        _atomic_write_bytes(
            path,
            _canonical_json_bytes(record),
            replace=False,
            expected_parent=parent,
        )
        self._model_forward_started_sha256 = str(record["record_sha256"])
        return str(record["record_sha256"])

    def authority_binding_material(self) -> Mapping[str, Any]:
        """Return the sole detached material used for the authority binding."""

        return {
            "forward_manifest_sha256": self.manifest.manifest_sha256,
            "checkpoint": self.checkpoint.as_dict(),
            "checkpoint_parent": self._checkpoint_parent.as_dict(),
            "code_sources": [
                {
                    "source_name": source_name,
                    "parent": parent.as_dict(),
                    "file": frozen.as_dict(),
                }
                for source_name, (parent, frozen) in sorted(
                    self._code_sources.items()
                )
            ],
            "lmdb_sources": [
                self.sources[source_id].as_dict()
                for source_id in sorted(self.sources)
            ],
            "output_root": self.output_root.as_dict(),
            "cache_root": self.cache_root.as_dict(),
            "forbidden_roots": list(self.forbidden_roots),
        }

    @property
    def authority_binding_sha256(self) -> str:
        return _sha256_bytes(
            _canonical_json_bytes(self.authority_binding_material())
        )

    def verify_code_sources(self) -> None:
        if set(self._code_sources) != {
            name for name, _digest in self.manifest.code_source_sha256
        }:
            raise ForwardContractError("CODE_SOURCE_RUNTIME_SET_DRIFT")
        for source_name, (parent, frozen) in sorted(self._code_sources.items()):
            parent.verify()
            fd = frozen.open_verified_readonly(
                verify_content=True,
                expected_parent=parent,
            )
            os.close(fd)

    @property
    def encoded_cache_fingerprint(self) -> str:
        manifest = self.manifest
        return _sha256_bytes(
            _canonical_json_bytes(
                {
                    "schema_version": ENCODED_CACHE_SCHEMA_VERSION,
                    "checkpoint_sha256": manifest.checkpoint_sha256,
                    "corpus_identity_sha256": manifest.corpus_identity.identity_sha256,
                    "visual_source_identity_sha256": self.sources[
                        manifest.visual_source_id
                    ].identity_sha256,
                    "subtitle_source_identity_sha256": self.sources[
                        manifest.subtitle_source_id
                    ].identity_sha256,
                    "temporal_manifest_sha256": manifest.temporal_manifest.semantic_sha256,
                    "feature_contract_sha256": manifest.feature_contract.semantic_sha256,
                    "model_config": manifest.model_contract.as_dict(),
                    "code_source_sha256": [list(item) for item in manifest.code_source_sha256],
                    "environment_sha256": manifest.environment_sha256,
                    "authority_binding_sha256": self.authority_binding_sha256,
                    "authority_id": manifest.authority_id,
                    "attempt_id": manifest.attempt_id,
                    "encoded_dtype": manifest.feature_contract.encoded_dtype,
                    "target_length": manifest.target_length,
                }
            )
        )

    def _build_shared_key_manifests(self) -> Mapping[str, Any]:
        security = self._shared_security
        manifests = {}
        for source_id, source in self.sources.items():
            if source.namespace == "QUERY":
                entity_namespace = "query_desc_id"
                raw_keys = tuple(
                    key.encode("ascii")
                    for key in self.manifest.query_identity.query_keys
                )
                split_authorization_sha256 = (
                    self.manifest.query_identity.split_manifest_sha256
                )
                key_encoding = "ASCII_CANONICAL_UINT"
            else:
                entity_namespace = "video_id"
                raw_keys = tuple(
                    video_id.encode("utf-8")
                    for video_id in self.manifest.corpus_identity.video_ids
                )
                split_authorization_sha256 = (
                    self.manifest.corpus_identity.identity_sha256
                )
                key_encoding = "UTF8_SAFE_ID"
            allowed = tuple(
                sorted(
                    security.canonical_lmdb_key_sha256(entity_namespace, raw_key)
                    for raw_key in raw_keys
                )
            )
            key_manifest_sha256 = _sha256_bytes(_canonical_json_bytes(list(allowed)))
            manifest_semantic_sha256 = security.lmdb_key_manifest_semantic_sha256(
                source_id=source_id,
                source_identity_sha256=source.identity_sha256,
                key_manifest_sha256=key_manifest_sha256,
                split_authorization_sha256=split_authorization_sha256,
                entity_namespace=entity_namespace,
                key_encoding=key_encoding,
                allowed_key_sha256=allowed,
            )
            manifests[source_id] = security.LmdbKeyManifest(
                source_id=source_id,
                source_identity_sha256=source.identity_sha256,
                key_manifest_sha256=key_manifest_sha256,
                split_authorization_sha256=split_authorization_sha256,
                entity_namespace=entity_namespace,
                key_encoding=key_encoding,
                allowed_key_sha256=allowed,
                manifest_semantic_sha256=manifest_semantic_sha256,
            )
        return manifests

    def issue_capability(self, control: Any) -> InputCapability:
        if self._capability_issued:
            raise ForwardContractError("FORWARD_CAPABILITY_REPLAY_OR_REISSUE")
        control_module = _require_exact_forward_control(control)
        controller_authority = _validate_controller_issued_authority(
            self,
            control,
        )
        self.verify_code_sources()
        manifest = self.manifest
        self.output_root.verify()
        self.cache_root.verify()
        control_chain_sha256 = _require_sha256(
            control.assert_fsynced_control_chain(
                authority_binding_sha256=self.authority_binding_sha256,
                forward_manifest_sha256=manifest.manifest_sha256,
                input_identity_sha256=manifest.input_identity_sha256,
                checkpoint_sha256=manifest.checkpoint_sha256,
                split_manifest_sha256=manifest.query_identity.split_manifest_sha256,
                corpus_manifest_sha256=manifest.corpus_identity.identity_sha256,
                evaluator_sha256=manifest.evaluator_schema_sha256,
                goal_id=manifest.goal_id,
                authority_id=manifest.authority_id,
                stage=manifest.stage,
                purpose=manifest.purpose,
                attempt_id=manifest.attempt_id,
            ),
            "control_chain_sha256",
        )
        token_id = _require_text(control.committed_token_id(), "token_id")
        eval_id = _require_text(control.committed_eval_id(), "eval_id")
        run_id = _require_text(control.committed_run_id(), "run_id")
        token_snapshot_sha256 = _require_sha256(
            control.committed_token_snapshot_sha256(), "token_snapshot_sha256"
        )
        consumer_nonce_sha256 = _sha256_bytes(
            _canonical_json_bytes(
                {
                    "schema_version": "c28f_a4_deterministic_forward_consumer_v1",
                    "run_id": run_id,
                    "authority_binding_sha256": self.authority_binding_sha256,
                    "forward_manifest_sha256": manifest.manifest_sha256,
                    "input_identity_sha256": manifest.input_identity_sha256,
                    "controller_authority_binding_sha256": (
                        controller_authority[
                            "controller_authority_binding_sha256"
                        ]
                    ),
                }
            )
        )
        consume_receipt_sha256 = _require_sha256(
            control.consume_forward_capability_once(
                authority_binding_sha256=self.authority_binding_sha256,
                consumer_nonce_sha256=consumer_nonce_sha256,
            ),
            "capability_consume_receipt_sha256",
        )
        running_commit_sha256 = _require_sha256(
            control.assert_running_state_event_committed(),
            "running_commit_sha256",
        )
        running_control_capability = control.issue_eval_running_capability()
        running = control_module.validate_eval_running_capability(
            running_control_capability
        )
        try:
            security_running = (
                self._shared_security.require_concrete_running_eval_control(
                    running_control_capability
                )
            )
        except BaseException as error:
            raise ForwardContractError(
                "SECURITY_RUNNING_CONTROL_VALIDATION_FAILED",
                type(error).__name__,
            ) from None
        if security_running != running:
            raise ForwardContractError("RUNNING_CONTROL_VALIDATOR_DISAGREEMENT")
        exact_running = {
            "goal_id": manifest.goal_id,
            "attempt_id": manifest.attempt_id,
            "run_id": run_id,
            "token_id": token_id,
            "eval_id": eval_id,
            "running_event_sha256": running_commit_sha256,
        }
        if any(running.get(name) != value for name, value in exact_running.items()):
            raise ForwardContractError("RUNNING_CONTROL_CAPABILITY_BINDING_MISMATCH")
        running_capability_sha256 = _require_sha256(
            running_control_capability.capability_sha256,
            "running_capability_sha256",
        )
        reservation_input_capability_sha256 = _require_sha256(
            running.get("reservation_input_capability_sha256"),
            "reservation_input_capability_sha256",
        )
        if reservation_input_capability_sha256 != controller_authority.get(
            "reservation_input_capability_sha256"
        ):
            raise ForwardContractError(
                "CONTROLLER_AUTHORITY_RESERVATION_BINDING_DRIFT"
            )
        controller_authority_binding_sha256 = _require_sha256(
            controller_authority.get("controller_authority_binding_sha256"),
            "controller_authority_binding_sha256",
        )
        material = {
            "goal_id": manifest.goal_id,
            "authority_id": manifest.authority_id,
            "run_id": run_id,
            "token_id": token_id,
            "eval_id": eval_id,
            "purpose": manifest.purpose,
            "stage": manifest.stage,
            "forward_manifest_sha256": manifest.manifest_sha256,
            "input_identity_sha256": manifest.input_identity_sha256,
            "authority_binding_sha256": self.authority_binding_sha256,
            "token_snapshot_sha256": token_snapshot_sha256,
            "gpu0_identity_sha256": manifest.gpu0_identity.semantic_sha256,
            "control_chain_sha256": control_chain_sha256,
            "running_commit_sha256": running_commit_sha256,
            "running_capability_sha256": running_capability_sha256,
            "reservation_input_capability_sha256": (
                reservation_input_capability_sha256
            ),
            "controller_authority_binding_sha256": (
                controller_authority_binding_sha256
            ),
            "consume_receipt_sha256": consume_receipt_sha256,
        }
        capability_id = _sha256_bytes(_canonical_json_bytes(material))
        capability = InputCapability(
            _CAPABILITY_FACTORY,
            capability_id=_sha256_bytes(_canonical_json_bytes(material)),
            _issuer_nonce=self._issuer_nonce,
            _authority=self,
            _control=control,
            _running_control_capability=running_control_capability,
            _run_receipt_sha256=None,
            _run_receipt_path=None,
            _run_receipt_parent=None,
            _run_chunk_parent=None,
            **{
                key: value
                for key, value in material.items()
                if key != "consume_receipt_sha256"
            },
        )
        self._capability_issued = True
        self._issued_capability_id = capability_id
        self._issued_capability = capability
        self._issued_control = control
        self._issued_capability_fields = {
            name: getattr(capability, name)
            for name in InputCapability.__slots__
            if not name.startswith("_")
        }
        return capability

    def _assert_capability_identity(self, capability: InputCapability) -> None:
        if not isinstance(capability, InputCapability):
            raise ForwardContractError("CAPABILITY_TYPE_MISMATCH")
        if capability is not self._issued_capability:
            raise ForwardContractError("CAPABILITY_OBJECT_IDENTITY_MISMATCH")
        if capability._issuer_nonce is not self._issuer_nonce:
            raise ForwardContractError("CAPABILITY_ISSUER_MISMATCH")
        if not self._capability_issued or capability.capability_id != self._issued_capability_id:
            raise ForwardContractError("CAPABILITY_NOT_SINGLE_ISSUED")
        if capability._control is not self._issued_control:
            raise ForwardContractError("CAPABILITY_CONTROL_BINDING_MISMATCH")
        if capability._authority is not self:
            raise ForwardContractError("CAPABILITY_AUTHORITY_OBJECT_MISMATCH")
        module = _require_exact_forward_control(capability._control)
        running = module.validate_eval_running_capability(
            capability._running_control_capability
        )
        if (
            capability._running_control_capability.capability_sha256
            != capability.running_capability_sha256
            or running.get("run_id") != capability.run_id
            or running.get("eval_id") != capability.eval_id
        ):
            raise ForwardContractError("CAPABILITY_RUNNING_HANDLE_MISMATCH")
        for field_name, value in self._issued_capability_fields.items():
            if getattr(capability, field_name) != value:
                raise ForwardContractError("CAPABILITY_BINDING_MISMATCH", field_name)
        if capability.capability_id != self._issued_capability_id:
            raise ForwardContractError("CAPABILITY_HASH_MISMATCH")

    def assert_capability(self, capability: InputCapability) -> None:
        self._assert_capability_identity(capability)
        capability.assert_running_committed()

    def allowed_key_digests(self, source_id: str) -> frozenset[str]:
        source = self.sources.get(source_id)
        if source is None:
            raise ForwardContractError("UNKNOWN_LMDB_SOURCE", source_id)
        cached = self._allowed_key_digest_cache.get(source_id)
        if cached is not None:
            return cached
        if source.namespace == "QUERY":
            keys = (key.encode("ascii") for key in self.manifest.query_identity.query_keys)
        else:
            keys = (video_id.encode("utf-8") for video_id in self.manifest.corpus_identity.video_ids)
        result = frozenset(_exact_key_digest(source.namespace, key) for key in keys)
        expected_count = (
            len(self.manifest.query_identity.query_keys)
            if source.namespace == "QUERY"
            else EXPECTED_CORPUS_VIDEO_COUNT
        )
        if len(result) != expected_count:
            raise ForwardContractError("EXACT_KEY_MANIFEST_COUNT_MISMATCH", source.namespace)
        self._allowed_key_digest_cache[source_id] = result
        return result

    def validate_full_key_manifest(
        self, capability: InputCapability, source_id: str
    ) -> Any:
        """Validate the complete in-memory allowlist before the first LMDB transaction."""

        self.assert_capability(capability)
        source = self.sources.get(source_id)
        if source is None:
            raise ForwardContractError("UNKNOWN_LMDB_SOURCE", source_id)
        digests = self.allowed_key_digests(source_id)
        expected_count = (
            len(self.manifest.query_identity.query_keys)
            if source.namespace == "QUERY"
            else EXPECTED_CORPUS_VIDEO_COUNT
        )
        if len(digests) != expected_count:
            raise ForwardContractError("EXACT_KEY_MANIFEST_COUNT_MISMATCH", source.namespace)
        shared_manifest = self._shared_key_manifests[source_id]
        existing_receipt = self._persisted_batch_receipts.get(source_id)
        if existing_receipt is not None:
            persistence_intent, opaque_receipt = existing_receipt
            try:
                if persistence_intent is not None:
                    validated = (
                        self._shared_security.validate_persisted_lmdb_batch_receipt(
                            opaque_receipt,
                            persistence_intent=persistence_intent,
                        )
                    )
                    if validated is not opaque_receipt:
                        raise ForwardContractError(
                            "LMDB_OPAQUE_RECEIPT_IDENTITY_DRIFT"
                        )
                validated = (
                    self._shared_security.rehydrate_persisted_lmdb_batch_receipt(
                        opaque_receipt,
                        manifest=shared_manifest,
                        running_capability=(
                            capability._running_control_capability
                        ),
                    )
                )
            except BaseException as error:
                raise ForwardContractError(
                    "SHARED_LMDB_BATCH_RECEIPT_REVALIDATION_FAILED",
                    type(error).__name__,
                ) from None
            if validated is not opaque_receipt:
                raise ForwardContractError("LMDB_OPAQUE_RECEIPT_IDENTITY_DRIFT")
            return opaque_receipt
        control_module = _require_exact_forward_control(capability._control)
        try:
            opaque_receipt = (
                capability._control.issue_persisted_lmdb_batch_control_receipt(
                    source_id=source_id,
                )
            )
        except control_module.A4ControlError as error:
            if str(error) != "LMDB persisted source receipt absent":
                raise ForwardContractError(
                    "LMDB_PERSISTED_BATCH_REHYDRATION_FAILED",
                    type(error).__name__,
                ) from None
        else:
            try:
                validated = (
                    self._shared_security.rehydrate_persisted_lmdb_batch_receipt(
                        opaque_receipt,
                        manifest=shared_manifest,
                        running_capability=(
                            capability._running_control_capability
                        ),
                    )
                )
            except BaseException as error:
                raise ForwardContractError(
                    "LMDB_PERSISTED_BATCH_REHYDRATION_FAILED",
                    type(error).__name__,
                ) from None
            if validated is not opaque_receipt:
                raise ForwardContractError("LMDB_OPAQUE_RECEIPT_IDENTITY_DRIFT")
            self._persisted_batch_receipts[source_id] = (None, opaque_receipt)
            return opaque_receipt
        if source.namespace == "QUERY":
            raw_keys = tuple(
                key.encode("ascii")
                for key in self.manifest.query_identity.query_keys
            )
        else:
            raw_keys = tuple(
                video_id.encode("utf-8")
                for video_id in self.manifest.corpus_identity.video_ids
            )
        try:
            authorized_batch = self._shared_security.authorize_lmdb_key_batch(
                manifest=shared_manifest,
                source_id=source_id,
                source_identity_sha256=source.identity_sha256,
                key_manifest_sha256=shared_manifest.key_manifest_sha256,
                split_authorization_sha256=shared_manifest.split_authorization_sha256,
                entity_namespace=shared_manifest.entity_namespace,
                raw_keys=raw_keys,
                ledger=self._shared_key_ledger,
                require_exact_manifest_set=True,
            )
        except BaseException as error:
            raise ForwardContractError(
                "SHARED_LMDB_BATCH_AUTHORIZATION_FAILED", type(error).__name__
            ) from None
        ledger_rows = self._shared_key_ledger.rows
        if (
            type(authorized_batch)
            is not self._shared_security.AuthorizedLmdbKeyBatch
            or len(authorized_batch) != expected_count
            or authorized_batch.key_hashes != shared_manifest.allowed_key_sha256
            or not ledger_rows
            or authorized_batch.ledger_tail_event_sha256
            != (ledger_rows[-1]["event_sha256"] if ledger_rows else None)
        ):
            raise ForwardContractError("SHARED_LMDB_BATCH_AUTHORIZATION_INVALID")
        try:
            persistence_intent = (
                self._shared_security.build_lmdb_batch_persistence_intent(
                    manifest=shared_manifest,
                    authorized_batch=authorized_batch,
                    goal_id=self.manifest.goal_id,
                    authority_id=self.manifest.authority_id,
                    parent_event_sha256=capability.running_commit_sha256,
                )
            )
            opaque_receipt = capability._control.commit_lmdb_batch(
                source_id=source_id,
                source_identity_sha256=source.identity_sha256,
                key_manifest_sha256=shared_manifest.key_manifest_sha256,
                key_set_sha256=_sha256_bytes(
                    _canonical_json_bytes(list(authorized_batch.key_hashes))
                ),
                source_manifest_sha256=shared_manifest.manifest_semantic_sha256,
                batch_sha256=authorized_batch.authorization_sha256,
                key_count=expected_count,
                shared_ledger_tail_sha256=(
                    authorized_batch.ledger_tail_event_sha256
                ),
            )
            validated = self._shared_security.validate_persisted_lmdb_batch_receipt(
                opaque_receipt,
                persistence_intent=persistence_intent,
            )
        except BaseException as error:
            raise ForwardContractError(
                "SHARED_LMDB_BATCH_PERSISTENCE_FAILED", type(error).__name__
            ) from None
        if validated is not opaque_receipt:
            raise ForwardContractError("LMDB_OPAQUE_RECEIPT_IDENTITY_DRIFT")
        self._persisted_batch_receipts[source_id] = (
            persistence_intent,
            opaque_receipt,
        )
        return opaque_receipt

    @property
    def lmdb_key_ledger_rows(self) -> Tuple[Mapping[str, Any], ...]:
        return self._shared_key_ledger.rows

    def assert_exact_key(
        self, capability: InputCapability, source_id: str, raw_key: bytes
    ) -> None:
        self._assert_capability_identity(capability)
        source = self.sources.get(source_id)
        if source is None:
            raise ForwardContractError("UNKNOWN_LMDB_SOURCE", source_id)
        key_digest = _exact_key_digest(source.namespace, raw_key)
        if key_digest not in self.allowed_key_digests(source_id):
            raise ForwardContractError("LMDB_KEY_NOT_IN_EXACT_MANIFEST", source.namespace)

    def verify_source(
        self, capability: InputCapability, source_id: str
    ) -> FrozenLmdbSource:
        self.assert_capability(capability)
        source = self.sources.get(source_id)
        if source is None:
            raise ForwardContractError("UNKNOWN_LMDB_SOURCE", source_id)
        if source_id not in self._verified_sources:
            source.directory.verify()
            # The approved digest is already frozen in the authority binding.
            # Re-hashing a shared query LMDB would scan bytes belonging to keys
            # outside this split, so runtime verification is metadata/inode only.
            fd = source.data_file.open_verified_readonly(
                verify_content=False,
                expected_parent=source.directory,
            )
            os.close(fd)
            self._verified_sources.add(source_id)
        return source


def _assert_frozen_file_descriptor(fd: int, frozen: FrozenFile) -> None:
    item = os.fstat(fd)
    actual = (
        int(item.st_dev),
        int(item.st_ino),
        _fd_mount_id(fd),
        int(item.st_size),
        int(item.st_mtime_ns),
    )
    expected = (
        frozen.device,
        frozen.inode,
        frozen.mount_id,
        frozen.size_bytes,
        frozen.mtime_ns,
    )
    if (
        not stat.S_ISREG(item.st_mode)
        or int(item.st_nlink) != 1
        or actual != expected
    ):
        raise ForwardContractError("FROZEN_FILE_DESCRIPTOR_IDENTITY_DRIFT", frozen.file_id)


def _assert_lmdb_mapping_matches_guard(
    source: FrozenLmdbSource,
    source_guard_fd: int,
) -> None:
    """Prove the mmap backing the LMDB environment is the pinned data.mdb inode."""

    guarded = os.fstat(source_guard_fd)
    expected_device = "%02x:%02x" % (
        os.major(int(guarded.st_dev)),
        os.minor(int(guarded.st_dev)),
    )
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open("/proc/self/maps", flags)
    try:
        parts = []
        total = 0
        while total <= 16 * 1024 * 1024:
            block = os.read(fd, min(65_536, 16 * 1024 * 1024 + 1 - total))
            if not block:
                break
            parts.append(block)
            total += len(block)
        if total > 16 * 1024 * 1024:
            raise ForwardContractError("PROC_MAPS_SIZE_LIMIT")
    finally:
        os.close(fd)
    candidates = []
    for raw_line in b"".join(parts).splitlines():
        fields = raw_line.split(None, 5)
        if len(fields) != 6:
            continue
        try:
            mapped_path = fields[5].decode("utf-8", errors="strict")
        except UnicodeDecodeError:
            continue
        if mapped_path == source.data_file.path:
            candidates.append((fields[3].decode("ascii"), fields[4].decode("ascii")))
    if not candidates or any(
        device.casefold() != expected_device.casefold()
        or inode != str(int(guarded.st_ino))
        for device, inode in candidates
    ):
        raise ForwardContractError("LMDB_MMAP_GUARD_IDENTITY_MISMATCH")


class GuardedLmdbReader:
    """A read-only LMDB facade whose exact key gate precedes ``txn.get``."""

    def __init__(
        self,
        authority: FrozenInputAuthority,
        capability: InputCapability,
        source_id: str,
    ) -> None:
        authority.assert_capability(capability)
        capability.assert_io_ready()
        authority.verify_code_sources()
        self.authority = authority
        self.capability = capability
        self.source = authority.verify_source(capability, source_id)
        self._environment: Any = None
        self._transaction: Any = None
        self._source_guard_fd: Optional[int] = None
        self._directory_guard_fd: Optional[int] = None
        self._batch_receipt_sha256: Optional[str] = None
        self._batch_control_receipt: Any = None

    def open(self) -> "GuardedLmdbReader":
        if self._environment is not None:
            raise ForwardContractError("LMDB_READER_ALREADY_OPEN", self.source.source_id)
        self.capability.assert_io_ready()
        self._batch_control_receipt = self.authority.validate_full_key_manifest(
            self.capability, self.source.source_id
        )
        control_module = _require_exact_forward_control(self.capability._control)
        batch_binding = control_module.validate_lmdb_batch_control_receipt(
            self._batch_control_receipt
        )
        self._batch_receipt_sha256 = _require_sha256(
            batch_binding["persisted_receipt_sha256"],
            "lmdb_batch_persisted_receipt_sha256",
        )
        self._directory_guard_fd = _open_directory_descriptor(
            self.source.directory.path, self.source.directory
        )
        directory_item = os.fstat(self._directory_guard_fd)
        if (
            int(directory_item.st_dev),
            int(directory_item.st_ino),
            _fd_mount_id(self._directory_guard_fd),
        ) != (
            self.source.directory.device,
            self.source.directory.inode,
            self.source.directory.mount_id,
        ):
            os.close(self._directory_guard_fd)
            self._directory_guard_fd = None
            raise ForwardContractError("LMDB_DIRECTORY_FD_BINDING_MISMATCH")
        try:
            self._source_guard_fd = os.open(
                "data.mdb",
                os.O_RDONLY
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=self._directory_guard_fd,
            )
            _assert_frozen_file_descriptor(
                self._source_guard_fd, self.source.data_file
            )
            lmdb = _lazy_lmdb()
            self._environment = lmdb.open(
                "/proc/self/fd/%d" % self._source_guard_fd,
                readonly=True,
                create=False,
                lock=False,
                readahead=False,
                max_readers=2048,
                subdir=False,
            )
            _assert_lmdb_mapping_matches_guard(self.source, self._source_guard_fd)
            _assert_frozen_file_descriptor(
                self._source_guard_fd, self.source.data_file
            )
            self.capability._record_observation(
                "LMDB_ENV_OPENED",
                {
                    "content_open_count": 0,
                    "content_bytes_read": 0,
                    "lmdb_transaction_open_count": 0,
                    "model_forward_count": 0,
                },
                {
                    "source_id": self.source.source_id,
                    "source_identity_sha256": self.source.identity_sha256,
                    "batch_receipt_sha256": self._batch_receipt_sha256,
                    "source_fd_bound": True,
                    "mmap_inode_matches_guard_fd": True,
                },
            )
            self._transaction = self._environment.begin(write=False, buffers=True)
            self.capability._record_observation(
                "LMDB_SNAPSHOT_TRANSACTION_OPENED",
                {
                    "content_open_count": 0,
                    "content_bytes_read": 0,
                    "lmdb_transaction_open_count": 1,
                    "model_forward_count": 0,
                },
                {
                    "source_id": self.source.source_id,
                    "single_snapshot_for_reader_lifetime": True,
                    "batch_receipt_sha256": self._batch_receipt_sha256,
                },
            )
        except BaseException:
            if self._transaction is not None:
                self._transaction.abort()
                self._transaction = None
            if self._environment is not None:
                self._environment.close()
                self._environment = None
            if self._source_guard_fd is not None:
                os.close(self._source_guard_fd)
                self._source_guard_fd = None
            if self._directory_guard_fd is not None:
                os.close(self._directory_guard_fd)
                self._directory_guard_fd = None
            raise
        return self

    def close(self) -> None:
        pending_error: Optional[BaseException] = None
        try:
            if self._environment is not None and self._source_guard_fd is not None:
                _assert_lmdb_mapping_matches_guard(
                    self.source, self._source_guard_fd
                )
            if self._source_guard_fd is not None:
                _assert_frozen_file_descriptor(
                    self._source_guard_fd, self.source.data_file
                )
            if self._directory_guard_fd is not None:
                observed = os.fstat(self._directory_guard_fd)
                if (
                    int(observed.st_dev),
                    int(observed.st_ino),
                    _fd_mount_id(self._directory_guard_fd),
                ) != (
                    self.source.directory.device,
                    self.source.directory.inode,
                    self.source.directory.mount_id,
                ):
                    raise ForwardContractError(
                        "LMDB_DIRECTORY_CHANGED_DURING_READ"
                    )
        except BaseException as error:
            pending_error = error
        finally:
            if self._transaction is not None:
                self._transaction.abort()
                self._transaction = None
            if self._environment is not None:
                self._environment.close()
                self._environment = None
            if self._source_guard_fd is not None:
                os.close(self._source_guard_fd)
                self._source_guard_fd = None
            if self._directory_guard_fd is not None:
                os.close(self._directory_guard_fd)
                self._directory_guard_fd = None
        if pending_error is not None:
            raise pending_error

    def __enter__(self) -> "GuardedLmdbReader":
        return self.open()

    def __exit__(self, _exc_type: Any, _exc: Any, _traceback: Any) -> None:
        self.close()

    def get_bytes(self, raw_key: bytes) -> bytes:
        self.authority.assert_exact_key(self.capability, self.source.source_id, raw_key)
        if self._environment is None or self._transaction is None:
            raise ForwardContractError("LMDB_READER_NOT_OPEN", self.source.source_id)
        if self._source_guard_fd is None:
            raise ForwardContractError("LMDB_SOURCE_GUARD_NOT_OPEN")
        _assert_lmdb_mapping_matches_guard(self.source, self._source_guard_fd)
        _assert_frozen_file_descriptor(self._source_guard_fd, self.source.data_file)
        raw_value = self._transaction.get(raw_key)
        if raw_value is None:
            raise ForwardContractError("AUTHORIZED_LMDB_KEY_MISSING", self.source.namespace)
        byte_count = len(raw_value)
        maximum = {
            "QUERY": QUERY_VALUE_MAX_BYTES,
            "VISUAL": VISUAL_VALUE_MAX_BYTES,
            "SUBTITLE": SUBTITLE_VALUE_MAX_BYTES,
        }[self.source.namespace]
        if byte_count <= 0 or byte_count > maximum:
            raise ForwardContractError("LMDB_VALUE_SIZE_LIMIT", self.source.namespace)
        result = bytes(raw_value)
        _assert_frozen_file_descriptor(self._source_guard_fd, self.source.data_file)
        _assert_lmdb_mapping_matches_guard(self.source, self._source_guard_fd)
        self.capability._record_observation(
            "LMDB_VALUE_READ",
            {
                "content_open_count": 1,
                "content_bytes_read": byte_count,
                "lmdb_transaction_open_count": 0,
                "model_forward_count": 0,
            },
            {
                "source_id": self.source.source_id,
                "namespace": self.source.namespace,
                "key_sha256": _exact_key_digest(self.source.namespace, raw_key),
                "value_sha256": _sha256_bytes(result),
            },
        )
        return result

    def query_features(self, query_key: str) -> Any:
        if self.source.namespace != "QUERY":
            raise ForwardContractError("LMDB_DECODER_NAMESPACE_MISMATCH")
        raw = self.get_bytes(_canonical_query_key(query_key).encode("ascii"))
        value = _decode_npz_features(
            raw,
            expected_dim=self.authority.manifest.model_contract.query_dim,
            max_sequence_length=QUERY_SEQUENCE_MAX_LENGTH,
            allow_empty=False,
        )
        self._record_decode(value)
        return value

    def visual_features(self, video_id: str) -> Any:
        if self.source.namespace != "VISUAL":
            raise ForwardContractError("LMDB_DECODER_NAMESPACE_MISMATCH")
        raw = self.get_bytes(_require_text(video_id, "video_id").encode("utf-8"))
        value = _decode_visual_msgpack(
            raw,
            expected_dim=self.authority.manifest.model_contract.visual_dim,
            max_sequence_length=VIDEO_SEQUENCE_MAX_LENGTH,
        )
        self._record_decode(value)
        return value

    def subtitle_features(self, video_id: str) -> Any:
        if self.source.namespace != "SUBTITLE":
            raise ForwardContractError("LMDB_DECODER_NAMESPACE_MISMATCH")
        raw = self.get_bytes(_require_text(video_id, "video_id").encode("utf-8"))
        value = _decode_npz_features(
            raw,
            expected_dim=self.authority.manifest.model_contract.subtitle_dim,
            max_sequence_length=VIDEO_SEQUENCE_MAX_LENGTH,
            allow_empty=True,
        )
        self._record_decode(value)
        return value

    def _record_decode(self, value: Any) -> None:
        self.capability._record_observation(
            "LMDB_VALUE_DECODED",
            {
                "content_open_count": 0,
                "content_bytes_read": 0,
                "lmdb_transaction_open_count": 0,
                "model_forward_count": 0,
            },
            {
                "source_id": self.source.source_id,
                "namespace": self.source.namespace,
                "decoded_shape": [int(item) for item in value.shape],
                "decoded_dtype": str(value.dtype),
                "decoded_nbytes": int(value.nbytes),
            },
        )


def _validate_feature_array(
    array: Any,
    expected_dim: int,
    max_sequence_length: int,
    label: str,
    *,
    allow_empty: bool = False,
) -> Any:
    np = _lazy_numpy()
    result = np.asarray(array)
    if (
        result.ndim != 2
        or (result.shape[0] == 0 and not allow_empty)
        or result.shape[0] > max_sequence_length
        or result.shape[1] != expected_dim
    ):
        raise ForwardContractError("FEATURE_SHAPE_MISMATCH", label)
    if result.dtype.kind not in {"f", "i", "u"} or int(result.dtype.itemsize) > 8:
        raise ForwardContractError("FEATURE_DTYPE_FORBIDDEN", label)
    result = result.astype(np.float32, copy=False)
    maximum_decoded_bytes = max_sequence_length * expected_dim * 4
    if int(result.nbytes) > maximum_decoded_bytes:
        raise ForwardContractError("DECOMPRESSED_FEATURE_SIZE_LIMIT", label)
    if not bool(np.isfinite(result).all()):
        raise ForwardContractError("FEATURE_NONFINITE", label)
    return result


def _decode_npz_features(
    raw: bytes,
    *,
    expected_dim: int,
    max_sequence_length: int,
    allow_empty: bool,
) -> Any:
    np = _lazy_numpy()
    raw_limit = (
        QUERY_VALUE_MAX_BYTES
        if max_sequence_length == QUERY_SEQUENCE_MAX_LENGTH
        else SUBTITLE_VALUE_MAX_BYTES
    )
    if not isinstance(raw, bytes) or not raw or len(raw) > raw_limit:
        raise ForwardContractError("NPZ_VALUE_SIZE_LIMIT")
    maximum_npy_bytes = max_sequence_length * expected_dim * 8 + 65_536
    try:
        with zipfile.ZipFile(io.BytesIO(raw), mode="r") as container:
            members = container.infolist()
            names = [member.filename for member in members]
            allowed_names = {"features.npy"}
            legacy_names = {"features.npy", "allow_pickle.npy"}
            member_names = frozenset(names)
            if not member_names.issubset(legacy_names):
                raise ForwardContractError("NPZ_MEMBER_NAME_CONTRACT_VIOLATION")
            if len(names) != len(member_names) or member_names not in {
                frozenset(allowed_names),
                frozenset(legacy_names),
            }:
                raise ForwardContractError("NPZ_MEMBER_COUNT_CONTRACT_VIOLATION")
            for member in members:
                if member.is_dir():
                    raise ForwardContractError("NPZ_MEMBER_DIRECTORY_CONTRACT_VIOLATION")
                if member.file_size <= 0:
                    raise ForwardContractError("NPZ_MEMBER_EMPTY_CONTRACT_VIOLATION")
                size_limit = (
                    maximum_npy_bytes
                    if member.filename == "features.npy"
                    else 65_536
                )
                if member.file_size > size_limit:
                    raise ForwardContractError("NPZ_MEMBER_SIZE_CONTRACT_VIOLATION")
                if member.flag_bits & 0x1:
                    raise ForwardContractError("NPZ_MEMBER_ENCRYPTION_CONTRACT_VIOLATION")
        with np.load(io.BytesIO(raw), allow_pickle=False) as archive:
            archive_keys = set(archive.files)
            if archive_keys not in ({"features"}, {"features", "allow_pickle"}):
                raise ForwardContractError("NPZ_FEATURE_KEYS_NOT_EXACT")
            if "allow_pickle" in archive_keys:
                legacy_marker = np.asarray(archive["allow_pickle"])
                if legacy_marker.shape != () or legacy_marker.dtype.kind != "b":
                    raise ForwardContractError("NPZ_LEGACY_MARKER_INVALID")
            array = archive["features"]
    except ForwardContractError:
        raise
    except BaseException as error:
        raise ForwardContractError("NPZ_FEATURE_DECODE_FAILED", type(error).__name__) from None
    return _validate_feature_array(
        array,
        expected_dim,
        max_sequence_length,
        "npz_features",
        allow_empty=allow_empty,
    )


def _decode_visual_msgpack(
    raw: bytes, *, expected_dim: int, max_sequence_length: int
) -> Any:
    if not isinstance(raw, bytes) or not raw or len(raw) > VISUAL_VALUE_MAX_BYTES:
        raise ForwardContractError("MSGPACK_VALUE_SIZE_LIMIT")
    try:
        msgpack_numpy = importlib.import_module("msgpack_numpy")
        value = msgpack_numpy.loads(raw, raw=False)
    except BaseException as error:
        raise ForwardContractError("MSGPACK_FEATURE_DECODE_FAILED", type(error).__name__) from None
    if not isinstance(value, Mapping) or set(value) != {"features"}:
        raise ForwardContractError("MSGPACK_FEATURE_KEYS_NOT_EXACT")
    return _validate_feature_array(
        value["features"], expected_dim, max_sequence_length, "visual_features"
    )


def _torch_load_weights_only(torch_module: Any, handle: Any) -> Any:
    """No unsafe compatibility fallback: old torch versions fail closed."""

    try:
        return torch_module.load(handle, map_location="cpu", weights_only=True)
    except TypeError:
        raise ForwardContractError("TORCH_WEIGHTS_ONLY_UNSUPPORTED") from None
    except BaseException as error:
        raise ForwardContractError("SAFE_CHECKPOINT_LOAD_FAILED", type(error).__name__) from None


_MODEL_HANDLE_FACTORY = object()
_GPU0_LEASE_FACTORY = object()
_UNSEALED_MODEL_VALUE = object()


class Gpu0RuntimeLease:
    """Opaque proof that this exact RUNNING capability resolved physical GPU0."""

    __slots__ = (
        "run_id",
        "capability_id",
        "gpu0_identity_sha256",
        "runtime_uuid",
        "runtime_pci_bus_id",
        "lease_sha256",
        "_factory",
    )

    def __init__(self, factory: object, **values: Any) -> None:
        if factory is not _GPU0_LEASE_FACTORY:
            raise ForwardContractError("GPU0_RUNTIME_LEASE_CONSTRUCTION_FORBIDDEN")
        for name in self.__slots__:
            object.__setattr__(self, name, factory if name == "_factory" else values[name])

    def __setattr__(self, _name: str, _value: Any) -> None:
        raise ForwardContractError("GPU0_RUNTIME_LEASE_MUTATION_FORBIDDEN")


class LoadedModelHandle:
    __slots__ = (
        "_model",
        "report",
        "capability_id",
        "checkpoint_sha256",
        "model_contract_sha256",
        "model_state_sha256",
        "model_runtime_seal",
        "gpu0_runtime_lease_sha256",
        "_factory",
    )

    def __init__(self, factory: object, **values: Any) -> None:
        if factory is not _MODEL_HANDLE_FACTORY:
            raise ForwardContractError("LOADED_MODEL_HANDLE_CONSTRUCTION_FORBIDDEN")
        for name in self.__slots__:
            object.__setattr__(self, name, factory if name == "_factory" else values[name])

    def __setattr__(self, _name: str, _value: Any) -> None:
        raise ForwardContractError("LOADED_MODEL_HANDLE_MUTATION_FORBIDDEN")


def _model_state_sha256(torch_module: Any, model: Any) -> str:
    state = model.state_dict()
    if not isinstance(state, Mapping) or not state:
        raise ForwardContractError("MODEL_STATE_EMPTY")
    material = []
    for name in sorted(state):
        value = state[name]
        if not hasattr(value, "detach"):
            raise ForwardContractError("MODEL_STATE_VALUE_NOT_TENSOR", str(name))
        material.append(
            {
                "name": str(name),
                "tensor": _tensor_digest(torch_module, value),
            }
        )
    return _sha256_bytes(_canonical_json_bytes(material))


def _assert_approved_business_modules_not_preloaded(
    authority: FrozenInputAuthority,
) -> None:
    """Gate the preloaded root package and require every leaf to be absent."""

    if (
        authority._verified_business_modules
        or authority._verified_package_modules
    ):
        raise ForwardContractError("APPROVED_BUSINESS_MODULE_GATE_REPLAY")
    loaded_packages = {
        module_name
        for module_name in _REQUIRED_SOURCE_BY_PACKAGE_MODULE
        if module_name in sys.modules
    }
    if loaded_packages != {_PRELOADED_ROOT_PACKAGE_MODULE}:
        raise ForwardContractError(
            "APPROVED_BUSINESS_PACKAGE_PRELOAD_SET_INVALID",
            ",".join(sorted(loaded_packages)),
        )
    preloaded = sorted(
        module_name
        for module_name in _APPROVED_SOURCE_BY_BUSINESS_MODULE
        if module_name in sys.modules
    )
    if preloaded:
        raise ForwardContractError(
            "APPROVED_BUSINESS_MODULE_PRELOADED",
            ",".join(preloaded),
        )
    _assert_required_package_bytecode_absent(
        authority,
        frozenset({_PRELOADED_ROOT_PACKAGE_MODULE}),
    )
    root_module = sys.modules.get(_PRELOADED_ROOT_PACKAGE_MODULE)
    root_record = _verify_required_package_module_identity(
        authority,
        _PRELOADED_ROOT_PACKAGE_MODULE,
        root_module,
    )
    authority._verified_package_modules[
        _PRELOADED_ROOT_PACKAGE_MODULE
    ] = root_record


def _assert_required_package_bytecode_absent(
    authority: FrozenInputAuthority,
    package_modules: frozenset[str],
) -> None:
    """Reject every local or redirected bytecode path before/after import."""

    if (
        not package_modules
        or not package_modules.issubset(
            _REQUIRED_SOURCE_BY_PACKAGE_MODULE
        )
    ):
        raise ForwardContractError(
            "APPROVED_BUSINESS_PACKAGE_BYTECODE_SCOPE_INVALID"
        )
    if sys.dont_write_bytecode is not True:
        raise ForwardContractError("PYTHON_BYTECODE_WRITES_NOT_DISABLED")
    if getattr(sys, "pycache_prefix", None) is not None:
        raise ForwardContractError("PYTHON_PYCACHE_PREFIX_FORBIDDEN")
    for module_name in sorted(package_modules):
        source_name = _REQUIRED_SOURCE_BY_PACKAGE_MODULE[module_name]
        frozen_record = authority._code_sources.get(source_name)
        if frozen_record is None:
            raise ForwardContractError(
                "APPROVED_BUSINESS_PACKAGE_SOURCE_NOT_FROZEN",
                source_name,
            )
        parent, frozen = frozen_record
        parent.verify()
        package_fd = _open_directory_descriptor(parent.path, parent)
        try:
            entries = os.listdir(package_fd)
        finally:
            os.close(package_fd)
        if (
            "__pycache__" in entries
            or any(
                isinstance(entry, str) and entry.endswith(".pyc")
                for entry in entries
            )
        ):
            raise ForwardContractError(
                "APPROVED_BUSINESS_PACKAGE_BYTECODE_CACHE_PRESENT",
                module_name,
            )
        fd = frozen.open_verified_readonly(
            verify_content=True,
            expected_parent=parent,
        )
        os.close(fd)


def _verify_required_package_module_identity(
    authority: FrozenInputAuthority,
    module_name: str,
    module: Any,
) -> Tuple[Any, ...]:
    """Verify one regular package initializer as frozen source code."""

    source_name = _REQUIRED_SOURCE_BY_PACKAGE_MODULE.get(module_name)
    if source_name is None:
        raise ForwardContractError("UNAPPROVED_BUSINESS_PACKAGE", module_name)
    frozen_record = authority._code_sources.get(source_name)
    if frozen_record is None:
        raise ForwardContractError(
            "APPROVED_BUSINESS_PACKAGE_SOURCE_NOT_FROZEN",
            source_name,
        )
    parent, frozen = frozen_record
    spec = getattr(module, "__spec__", None)
    loader = getattr(module, "__loader__", None)
    loader_filename = None
    if loader is not None and callable(getattr(loader, "get_filename", None)):
        try:
            loader_filename = loader.get_filename(module_name)
        except BaseException as error:
            raise ForwardContractError(
                "APPROVED_BUSINESS_PACKAGE_LOADER_FILENAME_FAILED",
                type(error).__name__,
            ) from None
    paths = (
        getattr(module, "__file__", None),
        getattr(spec, "origin", None),
        loader_filename,
        getattr(loader, "path", None),
    )
    search_locations = getattr(spec, "submodule_search_locations", None)
    module_search_locations = getattr(module, "__path__", None)
    try:
        canonical_paths = tuple(
            _canonical_absolute_path(str(path))[0]
            for path in paths
        )
        canonical_spec_locations = tuple(
            _canonical_absolute_path(str(path))[0]
            for path in search_locations
        )
        canonical_module_locations = tuple(
            _canonical_absolute_path(str(path))[0]
            for path in module_search_locations
        )
    except (ForwardContractError, TypeError, ValueError):
        raise ForwardContractError(
            "APPROVED_BUSINESS_PACKAGE_ORIGIN_INVALID",
            module_name,
        ) from None
    expected_directory = str(Path(frozen.path).parent)
    aliases = sorted(
        alias
        for alias, candidate in tuple(sys.modules.items())
        if candidate is module and alias != module_name
    )
    if (
        sys.modules.get(module_name) is not module
        or getattr(module, "__name__", None) != module_name
        or getattr(module, "__package__", None) != module_name
        or spec is None
        or getattr(spec, "name", None) != module_name
        or getattr(spec, "parent", None) != module_name
        or getattr(spec, "has_location", None) is not True
        or loader is None
        or type(loader) is not SourceFileLoader
        or getattr(loader, "name", None) != module_name
        or getattr(spec, "loader", None) is not loader
        or not frozen.path.endswith("/__init__.py")
        or any(path != frozen.path for path in canonical_paths)
        or canonical_spec_locations != (expected_directory,)
        or canonical_module_locations != (expected_directory,)
        or aliases
    ):
        raise ForwardContractError(
            "APPROVED_BUSINESS_PACKAGE_IDENTITY_OR_ALIAS_DRIFT",
            module_name,
        )
    parent.verify()
    fd = frozen.open_verified_readonly(
        verify_content=True,
        expected_parent=parent,
    )
    os.close(fd)
    return (
        module,
        spec,
        loader,
        frozen.path,
        frozen.sha256,
        expected_directory,
    )


def _verify_approved_business_module_identity(
    authority: FrozenInputAuthority,
    module_name: str,
    module: Any,
) -> Tuple[Any, ...]:
    source_name = _APPROVED_SOURCE_BY_BUSINESS_MODULE.get(module_name)
    if source_name is None:
        raise ForwardContractError("UNAPPROVED_BUSINESS_MODULE", module_name)
    frozen_record = authority._code_sources.get(source_name)
    if frozen_record is None:
        raise ForwardContractError("APPROVED_BUSINESS_SOURCE_NOT_FROZEN", source_name)
    parent, frozen = frozen_record
    spec = getattr(module, "__spec__", None)
    loader = getattr(module, "__loader__", None)
    module_file = getattr(module, "__file__", None)
    loader_filename = None
    if loader is not None and callable(getattr(loader, "get_filename", None)):
        try:
            loader_filename = loader.get_filename(module_name)
        except BaseException as error:
            raise ForwardContractError(
                "APPROVED_BUSINESS_MODULE_LOADER_FILENAME_FAILED",
                type(error).__name__,
            ) from None
    paths = (
        module_file,
        getattr(spec, "origin", None),
        loader_filename,
        getattr(loader, "path", None),
    )
    try:
        canonical_paths = tuple(
            _canonical_absolute_path(str(path))[0]
            for path in paths
        )
    except (ForwardContractError, TypeError, ValueError):
        raise ForwardContractError(
            "APPROVED_BUSINESS_MODULE_ORIGIN_INVALID",
            module_name,
        ) from None
    aliases = sorted(
        alias
        for alias, candidate in tuple(sys.modules.items())
        if candidate is module and alias != module_name
    )
    if (
        sys.modules.get(module_name) is not module
        or getattr(module, "__name__", None) != module_name
        or getattr(module, "__package__", None)
        != module_name.rpartition(".")[0]
        or spec is None
        or getattr(spec, "name", None) != module_name
        or getattr(spec, "parent", None) != module_name.rpartition(".")[0]
        or getattr(spec, "has_location", None) is not True
        or loader is None
        or type(loader) is not SourceFileLoader
        or getattr(loader, "name", None) != module_name
        or getattr(spec, "loader", None) is not loader
        or getattr(spec, "submodule_search_locations", None) is not None
        or not frozen.path.endswith(".py")
        or frozen.path.endswith("/__init__.py")
        or any(path != frozen.path for path in canonical_paths)
        or aliases
    ):
        raise ForwardContractError(
            "APPROVED_BUSINESS_MODULE_IDENTITY_OR_ALIAS_DRIFT",
            module_name,
        )
    parent.verify()
    fd = frozen.open_verified_readonly(
        verify_content=True,
        expected_parent=parent,
    )
    os.close(fd)
    return (
        module,
        spec,
        loader,
        frozen.path,
        frozen.sha256,
    )


def _assert_verified_business_module_record(
    authority: FrozenInputAuthority,
    module_name: str,
) -> None:
    record = authority._verified_business_modules.get(module_name)
    module = sys.modules.get(module_name)
    if record is None or module is not record[0]:
        raise ForwardContractError(
            "VERIFIED_BUSINESS_MODULE_RUNTIME_DRIFT",
            module_name,
        )
    observed = _verify_approved_business_module_identity(
        authority,
        module_name,
        module,
    )
    if observed != record:
        raise ForwardContractError(
            "VERIFIED_BUSINESS_MODULE_RUNTIME_DRIFT",
            module_name,
        )


def _assert_verified_package_module_record(
    authority: FrozenInputAuthority,
    module_name: str,
) -> None:
    record = authority._verified_package_modules.get(module_name)
    module = sys.modules.get(module_name)
    if record is None or module is not record[0]:
        raise ForwardContractError(
            "VERIFIED_BUSINESS_PACKAGE_RUNTIME_DRIFT",
            module_name,
        )
    observed = _verify_required_package_module_identity(
        authority,
        module_name,
        module,
    )
    if observed != record:
        raise ForwardContractError(
            "VERIFIED_BUSINESS_PACKAGE_RUNTIME_DRIFT",
            module_name,
        )


def _import_approved_business_module(
    authority: FrozenInputAuthority,
    module_name: str,
    *,
    expected_new_module_closure: frozenset[str],
    expected_new_package_closure: frozenset[str],
) -> Any:
    if (
        module_name not in expected_new_module_closure
        or not expected_new_module_closure
        or not expected_new_module_closure.issubset(
            _APPROVED_SOURCE_BY_BUSINESS_MODULE
        )
        or not expected_new_package_closure
        or _PRELOADED_ROOT_PACKAGE_MODULE
        in expected_new_package_closure
        or not expected_new_package_closure.issubset(
            _REQUIRED_SOURCE_BY_PACKAGE_MODULE
        )
    ):
        raise ForwardContractError("APPROVED_BUSINESS_IMPORT_CLOSURE_INVALID")
    before_packages = {
        name
        for name in _REQUIRED_SOURCE_BY_PACKAGE_MODULE
        if name in sys.modules
    }
    if before_packages != set(authority._verified_package_modules):
        raise ForwardContractError(
            "UNVERIFIED_APPROVED_BUSINESS_PACKAGE_PRELOADED"
        )
    for verified_name in sorted(before_packages):
        _assert_verified_package_module_record(authority, verified_name)
    before = {
        name
        for name in _APPROVED_SOURCE_BY_BUSINESS_MODULE
        if name in sys.modules
    }
    if before != set(authority._verified_business_modules):
        raise ForwardContractError("UNVERIFIED_APPROVED_BUSINESS_MODULE_PRELOADED")
    for verified_name in sorted(before):
        _assert_verified_business_module_record(authority, verified_name)
    if module_name in before:
        raise ForwardContractError("APPROVED_BUSINESS_MODULE_IMPORT_REPLAY")
    import_package_scope = frozenset(
        before_packages | set(expected_new_package_closure)
    )
    _assert_required_package_bytecode_absent(
        authority,
        import_package_scope,
    )
    module = importlib.import_module(module_name)
    after = {
        name
        for name in _APPROVED_SOURCE_BY_BUSINESS_MODULE
        if name in sys.modules
    }
    after_packages = {
        name
        for name in _REQUIRED_SOURCE_BY_PACKAGE_MODULE
        if name in sys.modules
    }
    _assert_required_package_bytecode_absent(
        authority,
        frozenset(after_packages),
    )
    new_modules = after - before
    new_packages = after_packages - before_packages
    if new_modules != set(expected_new_module_closure):
        raise ForwardContractError(
            "APPROVED_BUSINESS_MODULE_IMPORT_CLOSURE_DRIFT",
            ",".join(sorted(new_modules)),
        )
    if new_packages != set(expected_new_package_closure):
        raise ForwardContractError(
            "APPROVED_BUSINESS_PACKAGE_IMPORT_CLOSURE_DRIFT",
            ",".join(sorted(new_packages)),
        )
    package_records = {
        imported_name: _verify_required_package_module_identity(
            authority,
            imported_name,
            sys.modules.get(imported_name),
        )
        for imported_name in sorted(new_packages)
    }
    business_records = {
        imported_name: _verify_approved_business_module_identity(
            authority,
            imported_name,
            sys.modules.get(imported_name),
        )
        for imported_name in sorted(new_modules)
    }
    authority._verified_package_modules.update(package_records)
    authority._verified_business_modules.update(business_records)
    for imported_name in sorted(after_packages):
        _assert_verified_package_module_record(authority, imported_name)
    for imported_name in sorted(new_modules):
        _assert_verified_business_module_record(authority, imported_name)
    return module


def _model_runtime_seal(model: Any) -> Tuple[Any, ...]:
    def immutable_value(value: Any) -> Any:
        if value is None or type(value) in {bool, int, float, str}:
            return value
        if isinstance(value, tuple):
            converted = tuple(immutable_value(item) for item in value)
            if all(item is not _UNSEALED_MODEL_VALUE for item in converted):
                return converted
        return _UNSEALED_MODEL_VALUE

    tensors = []
    for namespace, values in (
        ("parameter", model.named_parameters()),
        ("buffer", model.named_buffers()),
    ):
        for name, value in values:
            tensors.append(
                (
                    namespace,
                    str(name),
                    id(value),
                    int(value.data_ptr()),
                    int(value._version),
                    tuple(value.shape),
                    str(value.dtype),
                    str(value.device),
                    bool(getattr(value, "requires_grad", False)),
                )
            )
    modules = []
    for name, module in model.named_modules():
        public_configuration = []
        for field_name, value in sorted(vars(module).items()):
            if field_name.startswith("_"):
                continue
            converted = immutable_value(value)
            if converted is not _UNSEALED_MODEL_VALUE:
                public_configuration.append((field_name, converted))
        modules.append(
            (
                str(name),
                id(module),
                type(module).__module__,
                type(module).__qualname__,
                tuple(public_configuration),
            )
        )
    return (tuple(modules), tuple(tensors))


def _assert_loaded_model_integrity(
    handle: LoadedModelHandle,
    *,
    authority: FrozenInputAuthority,
    capability: InputCapability,
    torch_module: Any,
) -> Any:
    if (
        not isinstance(handle, LoadedModelHandle)
        or handle._factory is not _MODEL_HANDLE_FACTORY
        or handle is not authority._loaded_model_handle
        or handle.capability_id != capability.capability_id
        or handle.checkpoint_sha256 != authority.manifest.checkpoint_sha256
        or authority._gpu0_runtime_lease is None
        or authority._gpu0_runtime_lease.lease_sha256
        != handle.gpu0_runtime_lease_sha256
        or _model_runtime_seal(handle._model) != handle.model_runtime_seal
    ):
        raise ForwardContractError("LOADED_MODEL_HANDLE_BINDING_MISMATCH")
    return handle._model


def _verify_gpu0_physical_binding(
    manifest: FrozenForwardManifest,
    capability: InputCapability,
    torch_module: Any,
) -> Gpu0RuntimeLease:
    capability.assert_io_ready()
    expected = manifest.gpu0_identity
    if os.environ.get("CUDA_VISIBLE_DEVICES") != expected.cuda_visible_devices:
        raise ForwardContractError("CUDA_VISIBLE_DEVICES_RUNTIME_DRIFT")
    try:
        nvml = importlib.import_module("pynvml")
        nvml.nvmlInit()
        try:
            handle = nvml.nvmlDeviceGetHandleByIndex(0)
            uuid = nvml.nvmlDeviceGetUUID(handle)
            pci = nvml.nvmlDeviceGetPciInfo(handle).busId
        finally:
            nvml.nvmlShutdown()
    except ForwardContractError:
        raise
    except BaseException as error:
        raise ForwardContractError("GPU0_NVML_IDENTITY_UNAVAILABLE", type(error).__name__) from None
    if isinstance(uuid, bytes):
        uuid = uuid.decode("ascii", errors="strict")
    if isinstance(pci, bytes):
        pci = pci.decode("ascii", errors="strict")
    if str(uuid) != expected.uuid or str(pci).casefold() != expected.pci_bus_id.casefold():
        raise ForwardContractError("GPU0_PHYSICAL_UUID_OR_PCI_MISMATCH")
    if not bool(torch_module.cuda.is_available()):
        raise ForwardContractError("CUDA0_NOT_AVAILABLE")
    torch_module.cuda.set_device(0)
    if int(torch_module.cuda.current_device()) != 0:
        raise ForwardContractError("CUDA_LOGICAL_DEVICE_NOT_ZERO")
    capability._record_observation(
        "GPU0_PHYSICAL_BINDING_VERIFIED",
        {
            "content_open_count": 0,
            "content_bytes_read": 0,
            "lmdb_transaction_open_count": 0,
            "model_forward_count": 0,
        },
        {
            "physical_index": 0,
            "logical_device": EXPECTED_DEVICE,
            "uuid": expected.uuid,
            "pci_bus_id": expected.pci_bus_id,
            "cuda_visible_devices": expected.cuda_visible_devices,
            "gpu1_queried": False,
        },
    )
    base = {
        "schema_version": "c28f_v5_gpu0_runtime_lease_v1",
        "run_id": capability.run_id,
        "capability_id": capability.capability_id,
        "gpu0_identity_sha256": expected.semantic_sha256,
        "runtime_uuid": str(uuid),
        "runtime_pci_bus_id": str(pci),
        "cuda_visible_devices": expected.cuda_visible_devices,
        "logical_device": EXPECTED_DEVICE,
        "gpu1_action_count": 0,
    }
    return Gpu0RuntimeLease(
        _GPU0_LEASE_FACTORY,
        run_id=capability.run_id,
        capability_id=capability.capability_id,
        gpu0_identity_sha256=expected.semantic_sha256,
        runtime_uuid=str(uuid),
        runtime_pci_bus_id=str(pci),
        lease_sha256=_semantic_sha256(base),
    )


def load_frozen_model(
    authority: FrozenInputAuthority, capability: InputCapability
) -> LoadedModelHandle:
    """Load the exact checkpoint safely and return an eval-only CUDA0 model."""

    authority.assert_capability(capability)
    capability.assert_io_ready()
    authority.verify_code_sources()
    _assert_approved_business_modules_not_preloaded(authority)
    if authority._loaded_model_handle is not None:
        raise ForwardContractError("FROZEN_MODEL_LOAD_REPLAY_FORBIDDEN")
    torch = _lazy_torch()
    if authority._gpu0_runtime_lease is not None:
        raise ForwardContractError("GPU0_RUNTIME_LEASE_REPLAY_FORBIDDEN")
    gpu0_runtime_lease = _verify_gpu0_physical_binding(
        authority.manifest, capability, torch
    )
    authority._gpu0_runtime_lease = gpu0_runtime_lease
    authority._checkpoint_parent.verify()
    fd = authority.checkpoint.open_verified_readonly(
        verify_content=True,
        expected_parent=authority._checkpoint_parent,
    )
    handle = os.fdopen(fd, "rb", closefd=False)
    try:
        _inspect_owned_torch_archive(
            handle,
            file_size=authority.checkpoint.size_bytes,
        )
        checkpoint = _torch_load_weights_only(torch, handle)
        handle.close()
        post_load_sha256 = _sha256_file_descriptor(fd)
        if post_load_sha256 != authority.checkpoint.sha256:
            raise ForwardContractError("CHECKPOINT_POST_LOAD_HASH_DRIFT")
        after = os.fstat(fd)
        expected = (
            authority.checkpoint.device,
            authority.checkpoint.inode,
            authority.checkpoint.mount_id,
            authority.checkpoint.size_bytes,
            authority.checkpoint.mtime_ns,
        )
        actual = (
            int(after.st_dev),
            int(after.st_ino),
            _fd_mount_id(fd),
            int(after.st_size),
            int(after.st_mtime_ns),
        )
        if actual != expected:
            raise ForwardContractError("CHECKPOINT_CHANGED_DURING_SAFE_LOAD")
        checkpoint_parent_fd = _open_directory_descriptor(
            authority._checkpoint_parent.path,
            authority._checkpoint_parent,
        )
        try:
            path_fd = os.open(
                Path(authority.checkpoint.path).name,
                _linux_o_path_flag()
                | getattr(os, "O_CLOEXEC", 0)
                | os.O_NOFOLLOW,
                dir_fd=checkpoint_parent_fd,
            )
            try:
                path_item = os.fstat(path_fd)
                path_identity = (
                    int(path_item.st_dev),
                    int(path_item.st_ino),
                    _fd_mount_id(path_fd),
                )
            finally:
                os.close(path_fd)
        finally:
            os.close(checkpoint_parent_fd)
        if path_identity != (
            authority.checkpoint.device,
            authority.checkpoint.inode,
            authority.checkpoint.mount_id,
        ):
            raise ForwardContractError("CHECKPOINT_PATH_REBOUND_AFTER_LOAD")
    finally:
        if not handle.closed:
            handle.close()
        os.close(fd)
    if not isinstance(checkpoint, Mapping):
        raise ForwardContractError("CHECKPOINT_NOT_MAPPING")
    state_key = authority.manifest.model_contract.checkpoint_state_key
    state = checkpoint.get(state_key)
    if not isinstance(state, Mapping) or not state:
        raise ForwardContractError("CHECKPOINT_MODEL_STATE_MISSING")
    model_module = _import_approved_business_module(
        authority,
        "blueprint_e2e_v2.models.full_model",
        expected_new_module_closure=_FULL_MODEL_RUNTIME_MODULE_CLOSURE,
        expected_new_package_closure=(
            _FULL_MODEL_RUNTIME_PACKAGE_CLOSURE
        ),
    )
    model_class = getattr(model_module, "C28CFullModel", None)
    if model_class is None:
        raise ForwardContractError("FROZEN_MODEL_CLASS_MISSING")
    contract = authority.manifest.model_contract
    model = model_class(
        query_dim=contract.query_dim,
        subtitle_dim=contract.subtitle_dim,
        visual_dim=contract.visual_dim,
        hidden_dim=contract.hidden_dim,
        late_interaction_enabled=True,
        late_soft_topk=contract.late_soft_topk,
        late_temperature=contract.late_temperature,
        token_maxsim_weight=contract.token_maxsim_weight,
        pooled_score_weight=contract.pooled_score_weight,
        late_score_weight=contract.late_score_weight,
    )
    try:
        incompatible = model.load_state_dict(state, strict=True)
    except BaseException as error:
        raise ForwardContractError("CHECKPOINT_STRICT_LOAD_FAILED", type(error).__name__) from None
    if getattr(incompatible, "missing_keys", ()) or getattr(incompatible, "unexpected_keys", ()):
        raise ForwardContractError("CHECKPOINT_STRICT_KEY_MISMATCH")
    model.requires_grad_(False)
    model.eval()
    model.to(torch.device(EXPECTED_DEVICE))
    model_state_sha256 = _model_state_sha256(torch, model)
    report = {
        "checkpoint_sha256": authority.checkpoint.sha256,
        "checkpoint_state_key": state_key,
        "state_tensor_count": len(state),
        "strict_load": True,
        "weights_only": True,
        "optimizer_loaded": False,
        "model_training": bool(model.training),
        "requires_grad_parameter_count": sum(
            int(parameter.requires_grad) for parameter in model.parameters()
        ),
        "device": EXPECTED_DEVICE,
        "post_load_same_fd_sha256": post_load_sha256,
    }
    if report["model_training"] or report["requires_grad_parameter_count"] != 0:
        raise ForwardContractError("MODEL_NOT_FROZEN_EVAL")
    capability._record_observation(
        "CHECKPOINT_MODEL_LOADED",
        {
            "content_open_count": 1,
            "content_bytes_read": authority.checkpoint.size_bytes,
            "lmdb_transaction_open_count": 0,
            "model_forward_count": 0,
        },
        {
            "checkpoint_sha256": authority.checkpoint.sha256,
            "weights_only": True,
            "strict_load": True,
            "optimizer_loaded": False,
            "post_load_same_fd_rehash": True,
        },
    )
    result = LoadedModelHandle(
        _MODEL_HANDLE_FACTORY,
        _model=model,
        report=report,
        capability_id=capability.capability_id,
        checkpoint_sha256=authority.checkpoint.sha256,
        model_contract_sha256=_sha256_bytes(
            _canonical_json_bytes(authority.manifest.model_contract.as_dict())
        ),
        model_state_sha256=model_state_sha256,
        model_runtime_seal=_model_runtime_seal(model),
        gpu0_runtime_lease_sha256=gpu0_runtime_lease.lease_sha256,
    )
    authority._loaded_model_handle = result
    return result


def stable_topk_indices(scores: Sequence[float], identities: Sequence[str], k: int) -> list[int]:
    """Exact deterministic score-descending, identity-ascending top-k."""

    if len(scores) != len(identities) or not scores:
        raise ForwardContractError("RANK_INPUT_SIZE_MISMATCH")
    if len(set(identities)) != len(identities):
        raise ForwardContractError("RANK_IDENTITIES_NOT_UNIQUE")
    limit = min(_require_positive_int(k, "k"), len(scores))
    np = _lazy_numpy()
    score_array = np.asarray(scores, dtype=np.float64)
    if score_array.ndim != 1 or not bool(np.isfinite(score_array).all()):
        raise ForwardContractError("RANK_SCORE_ARRAY_INVALID")
    threshold = float(np.partition(score_array, len(score_array) - limit)[len(score_array) - limit])
    candidate_indices = np.flatnonzero(score_array >= threshold).tolist()
    rows = []
    for index in candidate_indices:
        identity = identities[index]
        rows.append(
            (
                -_require_finite(float(score_array[index]), "scores[%d]" % index),
                _require_text(identity, "identity"),
                index,
            )
        )
    rows.sort()
    return [index for _negative_score, _identity, index in rows[:limit]]


def grid_spans_for_duration(
    duration_sec: float,
    *,
    target_length: int = TARGET_LENGTH,
    max_proposals: int = MAX_PROPOSALS_PER_VIDEO,
    clip_len_sec: float = 1.5,
    _grid: Any = None,
) -> Tuple[list[Tuple[int, int]], list[Tuple[float, float]]]:
    """Invoke the frozen training/evaluation ``TemporalGrid`` implementation."""

    duration = _require_finite(duration_sec, "duration_sec")
    if duration <= 0.0:
        raise ForwardContractError("NONPOSITIVE_VIDEO_DURATION")
    if (
        target_length != TARGET_LENGTH
        or max_proposals != MAX_PROPOSALS_PER_VIDEO
        or not math.isclose(
            _require_finite(clip_len_sec, "clip_len_sec"),
            1.5,
            rel_tol=0.0,
            abs_tol=0.0,
        )
    ):
        raise ForwardContractError("PROPOSAL_SCALE_DRIFT")
    if _grid is None:
        raise ForwardContractError("TEMPORAL_GRID_AUTHORITY_REQUIRED")
    grid = _grid
    if (
        type(getattr(grid, "clip_len", None)) is not float
        or grid.clip_len != clip_len_sec
        or type(getattr(grid, "max_clips", None)) is not int
        or grid.max_clips != target_length
    ):
        raise ForwardContractError("TEMPORAL_GRID_RUNTIME_BINDING_DRIFT")
    clip_array = grid.grid_spans(duration, max_spans=max_proposals)
    seconds_array = grid.clips_to_seconds(clip_array, duration)
    if (
        tuple(clip_array.shape) != tuple(seconds_array.shape)
        or clip_array.ndim != 2
        or clip_array.shape[1] != 2
        or not (1 <= int(clip_array.shape[0]) <= max_proposals)
    ):
        raise ForwardContractError("TEMPORAL_GRID_OUTPUT_SHAPE_INVALID")
    spans = [tuple(int(value) for value in row) for row in clip_array.tolist()]
    seconds = [tuple(float(value) for value in row) for row in seconds_array.tolist()]
    duration_float32 = float(_lazy_numpy().float32(duration))
    if any(
        start < 0
        or end <= start
        or start_sec < 0.0
        or end_sec <= start_sec
        or end_sec > duration_float32
        or not math.isfinite(start_sec)
        or not math.isfinite(end_sec)
        for (start, end), (start_sec, end_sec) in zip(spans, seconds)
    ):
        raise ForwardContractError("GENERATED_INVALID_PROPOSAL")
    return spans, seconds


def normalize_forward_queries(
    rows: Sequence[Mapping[str, Any]], query_identity: FrozenQueryIdentity
) -> list[Mapping[str, Any]]:
    if not isinstance(rows, (list, tuple)):
        raise ForwardContractError("QUERY_ROWS_NOT_EXPLICIT_SEQUENCE")
    if len(rows) != len(query_identity.query_keys):
        raise ForwardContractError("QUERY_ROW_COUNT_MISMATCH")
    normalised = []
    for index, (row, expected_key, expected_type) in enumerate(
        zip(rows, query_identity.query_keys, query_identity.query_types)
    ):
        if not isinstance(row, Mapping):
            raise ForwardContractError("QUERY_ROW_NOT_MAPPING", str(index))
        for field_name in row:
            lowered = str(field_name).lower()
            if any(fragment in lowered for fragment in _FORBIDDEN_QUERY_FIELD_FRAGMENTS):
                raise ForwardContractError("FORBIDDEN_QUERY_SUPPORT_FIELD", str(field_name))
        if set(row) != {"query_id", "query_type"}:
            raise ForwardContractError("FORWARD_QUERY_FIELDS_NOT_EXACT", str(index))
        query_key = _canonical_query_key(row["query_id"])
        if query_key != expected_key:
            raise ForwardContractError("QUERY_ORDER_OR_ID_MISMATCH", str(index))
        query_type = row["query_type"]
        if not isinstance(query_type, str) or query_type not in _QUERY_TYPES:
            raise ForwardContractError("INVALID_QUERY_TYPE", str(index))
        if query_type != expected_type:
            raise ForwardContractError(
                "QUERY_TYPE_AUTHORITY_BINDING_MISMATCH",
                str(index),
            )
        normalised.append({"query_id": query_key, "query_type": query_type})
    return normalised


def _safe_child_directory(
    root: FrozenDirectory, relative_parts: Sequence[str]
) -> FrozenDirectory:
    current_fd = _open_directory_descriptor(root.path, root)
    current_path = root.path
    flags = (
        os.O_RDONLY
        | os.O_DIRECTORY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        for part in relative_parts:
            if (
                not isinstance(part, str)
                or not part
                or part in {".", ".."}
                or "/" in part
                or "\x00" in part
            ):
                raise ForwardContractError("INVALID_CHILD_PATH_COMPONENT")
            try:
                next_fd = os.open(part, flags, dir_fd=current_fd)
            except FileNotFoundError:
                os.mkdir(part, mode=0o700, dir_fd=current_fd)
                os.fsync(current_fd)
                next_fd = os.open(part, flags, dir_fd=current_fd)
            item = os.fstat(next_fd)
            mount_id = _fd_mount_id(next_fd)
            if (
                not stat.S_ISDIR(item.st_mode)
                or int(item.st_dev) != root.device
                or mount_id != root.mount_id
                or int(item.st_nlink) < 2
            ):
                os.close(next_fd)
                raise ForwardContractError("CHILD_PATH_NOT_SAFE_DIRECTORY")
            os.close(current_fd)
            current_fd = next_fd
            current_path = os.path.join(current_path, part)
        item = os.fstat(current_fd)
        return FrozenDirectory(
            root_id="child-" + _sha256_bytes(current_path.encode("utf-8"))[:32],
            path=current_path,
            device=int(item.st_dev),
            inode=int(item.st_ino),
            mount_id=_fd_mount_id(current_fd),
        )
    finally:
        os.close(current_fd)


def _freeze_existing_child_directory(
    root: FrozenDirectory,
    relative_parts: Sequence[str],
) -> FrozenDirectory:
    """Freeze an existing descendant without creating recovery postimages."""

    current_fd = _open_directory_descriptor(root.path, root)
    current_path = root.path
    flags = (
        os.O_RDONLY
        | os.O_DIRECTORY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        for part in relative_parts:
            if (
                not isinstance(part, str)
                or not part
                or part in {".", ".."}
                or "/" in part
                or "\x00" in part
            ):
                raise ForwardContractError("INVALID_CHILD_PATH_COMPONENT")
            try:
                next_fd = os.open(part, flags, dir_fd=current_fd)
            except OSError as error:
                raise ForwardContractError(
                    "EXISTING_CHILD_DIRECTORY_UNAVAILABLE",
                    type(error).__name__,
                ) from None
            item = os.fstat(next_fd)
            mount_id = _fd_mount_id(next_fd)
            if (
                not stat.S_ISDIR(item.st_mode)
                or int(item.st_dev) != root.device
                or mount_id != root.mount_id
                or int(item.st_nlink) < 2
            ):
                os.close(next_fd)
                raise ForwardContractError(
                    "EXISTING_CHILD_DIRECTORY_IDENTITY_DRIFT"
                )
            os.close(current_fd)
            current_fd = next_fd
            current_path = os.path.join(current_path, part)
        item = os.fstat(current_fd)
        return FrozenDirectory(
            root_id=(
                "child-"
                + _sha256_bytes(current_path.encode("utf-8"))[:32]
            ),
            path=current_path,
            device=int(item.st_dev),
            inode=int(item.st_ino),
            mount_id=_fd_mount_id(current_fd),
        )
    finally:
        os.close(current_fd)


def _atomic_write_bytes(
    path: Path,
    data: bytes,
    *,
    replace: bool,
    expected_parent: FrozenDirectory,
) -> str:
    parent = str(path.parent)
    if parent != expected_parent.path:
        raise ForwardContractError("ATOMIC_WRITE_PARENT_BINDING_MISMATCH")
    if path.name in {"", ".", ".."} or "/" in path.name:
        raise ForwardContractError("INVALID_ARTIFACT_NAME")
    directory_fd = _open_directory_descriptor(parent, expected_parent)
    temporary_name = ".%s.tmp.%s" % (path.name, secrets.token_hex(16))
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    fd = -1
    try:
        fd = os.open(temporary_name, flags, 0o600, dir_fd=directory_fd)
        offset = 0
        while offset < len(data):
            written = os.write(fd, data[offset:])
            if written <= 0:
                raise ForwardContractError("ATOMIC_WRITE_SHORT")
            offset += written
        os.fsync(fd)
        os.close(fd)
        fd = -1
        if replace:
            try:
                target = os.stat(path.name, dir_fd=directory_fd, follow_symlinks=False)
            except FileNotFoundError:
                target = None
            if target is not None and (
                not stat.S_ISREG(target.st_mode) or int(target.st_nlink) != 1
            ):
                raise ForwardContractError("REPLACE_TARGET_NOT_REGULAR")
            os.replace(
                temporary_name,
                path.name,
                src_dir_fd=directory_fd,
                dst_dir_fd=directory_fd,
            )
        else:
            try:
                os.link(
                    temporary_name,
                    path.name,
                    src_dir_fd=directory_fd,
                    dst_dir_fd=directory_fd,
                    follow_symlinks=False,
                )
            except FileExistsError:
                raise ForwardContractError("IMMUTABLE_ARTIFACT_ALREADY_EXISTS", path.name) from None
            os.unlink(temporary_name, dir_fd=directory_fd)
        os.fsync(directory_fd)
    finally:
        if fd >= 0:
            os.close(fd)
        try:
            os.unlink(temporary_name, dir_fd=directory_fd)
        except FileNotFoundError:
            pass
        os.close(directory_fd)
    return _sha256_bytes(data)


def _read_bytes_no_follow(
    path: Path,
    *,
    max_bytes: Optional[int] = None,
    expected_parent: Optional[FrozenDirectory] = None,
) -> bytes:
    """Read a regular file through one O_NOFOLLOW descriptor and stable identity."""

    if expected_parent is None:
        expected_parent = FrozenDirectory.capture(
            "read-parent-" + _sha256_bytes(str(path.parent).encode("utf-8"))[:32],
            str(path.parent),
        )
    if str(path.parent) != expected_parent.path:
        raise ForwardContractError("READ_PARENT_BINDING_MISMATCH")
    parent_fd = _open_directory_descriptor(expected_parent.path, expected_parent)
    try:
        fd = os.open(
            path.name,
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=parent_fd,
        )
    except BaseException:
        os.close(parent_fd)
        raise
    try:
        before = os.fstat(fd)
        before_mount_id = _fd_mount_id(fd)
        if not stat.S_ISREG(before.st_mode) or int(before.st_nlink) != 1:
            raise ForwardContractError("READ_TARGET_NOT_REGULAR")
        if max_bytes is not None and int(before.st_size) > max_bytes:
            raise ForwardContractError("READ_TARGET_TOO_LARGE")
        parts = []
        remaining = int(before.st_size)
        while remaining:
            block = os.read(fd, min(8 * 1024 * 1024, remaining))
            if not block:
                raise ForwardContractError("READ_TARGET_SHORT")
            parts.append(block)
            remaining -= len(block)
        if os.read(fd, 1):
            raise ForwardContractError("READ_TARGET_GREW")
        after = os.fstat(fd)
        before_identity = (
            int(before.st_dev),
            int(before.st_ino),
            before_mount_id,
            int(before.st_size),
            int(before.st_mtime_ns),
        )
        after_identity = (
            int(after.st_dev),
            int(after.st_ino),
            _fd_mount_id(fd),
            int(after.st_size),
            int(after.st_mtime_ns),
        )
        if before_identity != after_identity:
            raise ForwardContractError("READ_TARGET_CHANGED_DURING_READ")
        return b"".join(parts)
    finally:
        os.close(fd)
        os.close(parent_fd)


def _read_canonical_json(
    path: Path,
    *,
    max_bytes: int = 128 * 1024 * 1024,
    expected_parent: Optional[FrozenDirectory] = None,
) -> Mapping[str, Any]:
    data = _read_bytes_no_follow(
        path,
        max_bytes=max_bytes,
        expected_parent=expected_parent,
    )
    try:
        value = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ForwardContractError("JSON_ARTIFACT_DECODE_FAILED", type(error).__name__) from None
    if not isinstance(value, Mapping) or _canonical_json_bytes(value) != data:
        raise ForwardContractError("JSON_ARTIFACT_NOT_CANONICAL")
    return value


def _tensor_digest(torch_module: Any, tensor: Any) -> Mapping[str, Any]:
    value = tensor.detach().cpu().contiguous()
    array = value.numpy()
    return {
        "dtype": str(value.dtype),
        "shape": list(value.shape),
        "sha256": _sha256_bytes(array.tobytes(order="C")),
    }


def _encoded_payload_digest(torch_module: Any, payload: Mapping[str, Any]) -> str:
    tensor_keys = (
        "visual_pool",
        "subtitle_pool",
        "joint_pool",
        "visual",
        "subtitle",
        "joint",
        "visual_mask",
        "subtitle_mask",
        "joint_mask",
    )
    material = {
        "schema_version": payload.get("schema_version"),
        "cache_fingerprint": payload.get("cache_fingerprint"),
        "producer_capability_id": payload.get("producer_capability_id"),
        "producer_token_id": payload.get("producer_token_id"),
        "producer_eval_id": payload.get("producer_eval_id"),
        "start_index": payload.get("start_index"),
        "end_index": payload.get("end_index"),
        "video_ids_sha256": payload.get("video_ids_sha256"),
        "durations_sha256": payload.get("durations_sha256"),
        "visual_source_identity_sha256": payload.get("visual_source_identity_sha256"),
        "subtitle_source_identity_sha256": payload.get("subtitle_source_identity_sha256"),
        "temporal_manifest_sha256": payload.get("temporal_manifest_sha256"),
        "feature_contract_sha256": payload.get("feature_contract_sha256"),
        "tensors": {key: _tensor_digest(torch_module, payload[key]) for key in tensor_keys},
    }
    return _sha256_bytes(_canonical_json_bytes(material))


def _validate_encoded_payload_tensors(
    torch_module: Any,
    payload: Mapping[str, Any],
    *,
    row_count: int,
) -> None:
    """Close tensor shape, dtype, and mask semantics before cache reuse/write."""

    expected_shapes = {
        "visual_pool": (row_count, 384),
        "subtitle_pool": (row_count, 384),
        "joint_pool": (row_count, 384),
        "visual": (row_count, TARGET_LENGTH, 384),
        "subtitle": (row_count, TARGET_LENGTH, 384),
        "joint": (row_count, TARGET_LENGTH, 384),
        "visual_mask": (row_count, TARGET_LENGTH),
        "subtitle_mask": (row_count, TARGET_LENGTH),
        "joint_mask": (row_count, TARGET_LENGTH),
    }
    for field_name, expected_shape in expected_shapes.items():
        value = payload.get(field_name)
        if value is None or tuple(value.shape) != expected_shape:
            raise ForwardContractError(
                "ENCODED_CHUNK_TENSOR_SHAPE_MISMATCH",
                field_name,
            )
    for field_name in (
        "visual_pool",
        "subtitle_pool",
        "joint_pool",
        "visual",
        "subtitle",
        "joint",
    ):
        value = payload[field_name]
        if str(value.dtype) != "torch.float32":
            raise ForwardContractError(
                "ENCODED_CHUNK_TENSOR_DTYPE_MISMATCH",
                field_name,
            )
        if not bool(torch_module.isfinite(value).all()):
            raise ForwardContractError("ENCODED_CHUNK_NONFINITE", field_name)
    for field_name in ("visual_mask", "subtitle_mask", "joint_mask"):
        if str(payload[field_name].dtype) != "torch.bool":
            raise ForwardContractError(
                "ENCODED_CHUNK_MASK_DTYPE_MISMATCH",
                field_name,
            )
    visual_mask = payload["visual_mask"]
    subtitle_mask = payload["subtitle_mask"]
    joint_mask = payload["joint_mask"]
    if not bool(visual_mask.any(dim=1).all()):
        raise ForwardContractError("ENCODED_CHUNK_EMPTY_VISUAL_MASK")
    if not bool(
        torch_module.equal(
            joint_mask,
            torch_module.logical_or(visual_mask, subtitle_mask),
        )
    ):
        raise ForwardContractError("ENCODED_CHUNK_JOINT_MASK_MISMATCH")
    if not bool(joint_mask.any(dim=1).all()):
        raise ForwardContractError("ENCODED_CHUNK_EMPTY_JOINT_MASK")


def _atomic_torch_save(
    path: Path,
    payload: Mapping[str, Any],
    torch_module: Any,
    *,
    expected_parent: FrozenDirectory,
) -> Tuple[str, str]:
    buffer = io.BytesIO()
    torch_module.save(dict(payload), buffer)
    data = buffer.getvalue()
    if len(data) > MAX_OWNED_TORCH_ARCHIVE_BYTES:
        raise ForwardContractError("OWNED_TORCH_ARCHIVE_SIZE_LIMIT")
    file_sha = _atomic_write_bytes(
        path,
        data,
        replace=False,
        expected_parent=expected_parent,
    )
    semantic_sha = _encoded_payload_digest(torch_module, payload)
    return file_sha, semantic_sha


def _inspect_owned_torch_archive(handle: Any, *, file_size: int) -> None:
    if file_size <= 0 or file_size > MAX_OWNED_TORCH_ARCHIVE_BYTES:
        raise ForwardContractError("OWNED_TORCH_ARCHIVE_SIZE_LIMIT")
    try:
        with zipfile.ZipFile(handle, mode="r") as archive:
            members = archive.infolist()
            if not members or len(members) > MAX_OWNED_TORCH_ARCHIVE_MEMBERS:
                raise ForwardContractError("OWNED_TORCH_ARCHIVE_MEMBER_LIMIT")
            total = 0
            seen_names = set()
            for member in members:
                pieces = member.filename.split("/")
                unix_mode = (int(member.external_attr) >> 16) & 0xFFFF
                if (
                    member.is_dir()
                    or member.flag_bits & 0x1
                    or member.filename in seen_names
                    or len(member.filename) > 4096
                    or "\\" in member.filename
                    or "\x00" in member.filename
                    or any(piece in {"", ".", ".."} for piece in pieces)
                    or stat.S_ISLNK(unix_mode)
                    or member.file_size < 0
                    or member.compress_size < 0
                    or member.file_size > MAX_OWNED_TORCH_UNCOMPRESSED_BYTES
                ):
                    raise ForwardContractError("OWNED_TORCH_ARCHIVE_MEMBER_INVALID")
                seen_names.add(member.filename)
                total += int(member.file_size)
                if total > MAX_OWNED_TORCH_UNCOMPRESSED_BYTES:
                    raise ForwardContractError("OWNED_TORCH_ARCHIVE_EXPANSION_LIMIT")
    except ForwardContractError:
        raise
    except (OSError, zipfile.BadZipFile) as error:
        raise ForwardContractError(
            "OWNED_TORCH_ARCHIVE_PREFLIGHT_FAILED", type(error).__name__
        ) from None
    finally:
        handle.seek(0)


def _safe_load_owned_torch_artifact(
    path: Path,
    torch_module: Any,
    *,
    expected_parent: FrozenDirectory,
    expected_file_sha256: Optional[str] = None,
) -> Any:
    if str(path.parent) != expected_parent.path:
        raise ForwardContractError("TORCH_ARTIFACT_PARENT_BINDING_MISMATCH")
    parent_fd = _open_directory_descriptor(expected_parent.path, expected_parent)
    try:
        fd = os.open(
            path.name,
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=parent_fd,
        )
    finally:
        os.close(parent_fd)
    handle = os.fdopen(fd, "rb", closefd=False)
    try:
        before = os.fstat(fd)
        before_mount_id = _fd_mount_id(fd)
        if not stat.S_ISREG(before.st_mode):
            raise ForwardContractError("TORCH_ARTIFACT_NOT_REGULAR")
        if int(before.st_nlink) != 1:
            raise ForwardContractError("TORCH_ARTIFACT_NLINK_INVALID")
        if expected_file_sha256 is not None:
            _require_sha256(expected_file_sha256, "expected_file_sha256")
            if _sha256_file_descriptor(fd) != expected_file_sha256:
                raise ForwardContractError("TORCH_ARTIFACT_FILE_HASH_DRIFT")
        _inspect_owned_torch_archive(handle, file_size=int(before.st_size))
        value = _torch_load_weights_only(torch_module, handle)
        after = os.fstat(fd)
        before_identity = (
            int(before.st_dev),
            int(before.st_ino),
            before_mount_id,
            int(before.st_size),
            int(before.st_mtime_ns),
        )
        after_identity = (
            int(after.st_dev),
            int(after.st_ino),
            _fd_mount_id(fd),
            int(after.st_size),
            int(after.st_mtime_ns),
        )
        if before_identity != after_identity:
            raise ForwardContractError("TORCH_ARTIFACT_CHANGED_DURING_LOAD")
        return value
    finally:
        handle.close()
        os.close(fd)


def _fit_sequence(
    array: Any,
    target_length: int,
    *,
    allow_empty: bool = False,
) -> Tuple[Any, Any]:
    np = _lazy_numpy()
    if target_length != TARGET_LENGTH:
        raise ForwardContractError("TARGET_LENGTH_DRIFT")
    sequence = np.asarray(array, dtype=np.float32)
    if (
        sequence.ndim != 2
        or sequence.shape[1] <= 0
        or (sequence.shape[0] == 0 and not allow_empty)
    ):
        raise ForwardContractError("INVALID_RAW_SEQUENCE")
    output = np.zeros((target_length, sequence.shape[1]), dtype=np.float32)
    mask = np.zeros((target_length,), dtype=np.bool_)
    if sequence.shape[0] == 0:
        pass
    elif sequence.shape[0] == target_length:
        output[:] = sequence
        mask[:] = True
    elif sequence.shape[0] > target_length:
        edges = np.linspace(0, sequence.shape[0], int(target_length) + 1)
        for index in range(target_length):
            start = int(np.floor(edges[index]))
            end = int(np.ceil(edges[index + 1]))
            end = max(end, start + 1)
            output[index] = sequence[start : min(end, sequence.shape[0])].mean(axis=0)
        mask[:] = True
    else:
        output[: sequence.shape[0]] = sequence
        mask[: sequence.shape[0]] = True
    if not bool(np.isfinite(output).all()):
        raise ForwardContractError("FITTED_SEQUENCE_NONFINITE")
    return output, mask


_ENCODED_HANDLE_FACTORY = object()


class EncodedCorpusHandle:
    __slots__ = (
        "_payload",
        "video_count",
        "cache_fingerprint",
        "capability_id",
        "receipt_sha256",
        "_tensor_runtime_seal",
        "_factory",
    )

    def __init__(self, factory: object, *, torch_module: Any, **values: Any) -> None:
        if factory is not _ENCODED_HANDLE_FACTORY:
            raise ForwardContractError("ENCODED_CORPUS_HANDLE_CONSTRUCTION_FORBIDDEN")
        object.__setattr__(
            self,
            "_payload",
            MappingProxyType(dict(values["_payload"])),
        )
        for name in (
            "video_count",
            "cache_fingerprint",
            "capability_id",
            "receipt_sha256",
        ):
            object.__setattr__(self, name, values[name])
        object.__setattr__(self, "_factory", factory)
        if self.video_count != EXPECTED_CORPUS_VIDEO_COUNT:
            raise ForwardContractError("ENCODED_HANDLE_COUNT_NOT_17435")
        for key in (
            "visual_pool",
            "subtitle_pool",
            "joint_pool",
            "visual",
            "subtitle",
            "joint",
            "visual_mask",
            "subtitle_mask",
            "joint_mask",
        ):
            value = self._payload.get(key)
            if value is None or int(value.shape[0]) != self.video_count:
                raise ForwardContractError("ENCODED_HANDLE_TENSOR_COUNT_MISMATCH", key)
        expected_shapes = {
            "visual_pool": (EXPECTED_CORPUS_VIDEO_COUNT, 384),
            "subtitle_pool": (EXPECTED_CORPUS_VIDEO_COUNT, 384),
            "joint_pool": (EXPECTED_CORPUS_VIDEO_COUNT, 384),
            "visual": (EXPECTED_CORPUS_VIDEO_COUNT, TARGET_LENGTH, 384),
            "subtitle": (EXPECTED_CORPUS_VIDEO_COUNT, TARGET_LENGTH, 384),
            "joint": (EXPECTED_CORPUS_VIDEO_COUNT, TARGET_LENGTH, 384),
            "visual_mask": (EXPECTED_CORPUS_VIDEO_COUNT, TARGET_LENGTH),
            "subtitle_mask": (EXPECTED_CORPUS_VIDEO_COUNT, TARGET_LENGTH),
            "joint_mask": (EXPECTED_CORPUS_VIDEO_COUNT, TARGET_LENGTH),
        }
        for key, shape in expected_shapes.items():
            if tuple(self._payload[key].shape) != shape:
                raise ForwardContractError("ENCODED_HANDLE_SHAPE_MISMATCH", key)
        for key in ("visual_pool", "subtitle_pool", "joint_pool", "visual", "subtitle", "joint"):
            if str(self._payload[key].dtype) != "torch.float32":
                raise ForwardContractError("ENCODED_HANDLE_DTYPE_NOT_FLOAT32", key)
            if not bool(torch_module.isfinite(self._payload[key]).all()):
                raise ForwardContractError("ENCODED_HANDLE_NONFINITE", key)
        for key in ("visual_mask", "subtitle_mask", "joint_mask"):
            if str(self._payload[key].dtype) != "torch.bool":
                raise ForwardContractError("ENCODED_HANDLE_MASK_DTYPE_NOT_BOOL", key)
        if not bool(self._payload["visual_mask"].any(dim=1).all()):
            raise ForwardContractError("ENCODED_HANDLE_EMPTY_VISUAL_MASK")
        expected_joint = torch_module.logical_or(
            self._payload["visual_mask"], self._payload["subtitle_mask"]
        )
        if not bool(torch_module.equal(expected_joint, self._payload["joint_mask"])):
            raise ForwardContractError("ENCODED_HANDLE_JOINT_MASK_MISMATCH")
        if not bool(self._payload["joint_mask"].any(dim=1).all()):
            raise ForwardContractError("ENCODED_HANDLE_EMPTY_JOINT_MASK")
        object.__setattr__(
            self,
            "_tensor_runtime_seal",
            self._current_tensor_runtime_seal(),
        )

    def __setattr__(self, _name: str, _value: Any) -> None:
        raise ForwardContractError("ENCODED_CORPUS_HANDLE_MUTATION_FORBIDDEN")

    def _current_tensor_runtime_seal(self) -> Tuple[Tuple[Any, ...], ...]:
        return tuple(
            (
                key,
                id(value),
                int(value.data_ptr()),
                int(value._version),
                tuple(value.shape),
                str(value.dtype),
                str(value.device),
            )
            for key, value in sorted(self._payload.items())
        )

    def assert_runtime_integrity(self) -> None:
        if (
            self._factory is not _ENCODED_HANDLE_FACTORY
            or not isinstance(self._payload, MappingProxyType)
            or self._current_tensor_runtime_seal() != self._tensor_runtime_seal
        ):
            raise ForwardContractError("ENCODED_CORPUS_RUNTIME_SEAL_INVALID")

    def slice(self, start: int, end: int) -> Mapping[str, Any]:
        self.assert_runtime_integrity()
        if start < 0 or end <= start or end > self.video_count:
            raise ForwardContractError("ENCODED_VIEW_SLICE_INVALID")
        return {key: value[start:end].clone() for key, value in self._payload.items()}

    def gather(self, indices: Sequence[int], torch_module: Any) -> Mapping[str, Any]:
        self.assert_runtime_integrity()
        if not indices:
            raise ForwardContractError("EMPTY_ENCODED_GATHER")
        if any(index < 0 or index >= self.video_count for index in indices):
            raise ForwardContractError("ENCODED_GATHER_INDEX_OUT_OF_RANGE")
        index_tensor = torch_module.tensor(list(indices), dtype=torch_module.long)
        return {
            key: value.index_select(0, index_tensor)
            for key, value in self._payload.items()
        }

    def pooled_slice(self, start: int, end: int) -> Mapping[str, Any]:
        self.assert_runtime_integrity()
        if start < 0 or end <= start or end > self.video_count:
            raise ForwardContractError("ENCODED_VIEW_SLICE_INVALID")
        return {
            key: self._payload[key][start:end].clone()
            for key in ("visual_pool", "subtitle_pool", "joint_pool")
        }


class EncodedCorpusCache:
    """Checkpoint/corpus-specific encoded chunks with restart-safe receipts."""

    def __init__(
        self,
        authority: FrozenInputAuthority,
        capability: InputCapability,
        *,
        chunk_size: int = 256,
    ) -> None:
        authority.assert_capability(capability)
        capability.assert_io_ready()
        authority.verify_code_sources()
        self.authority = authority
        self.capability = capability
        self.chunk_size = _require_positive_int(chunk_size, "chunk_size")
        self.root_identity = _safe_child_directory(
            authority.cache_root,
            ("encoded_corpus", authority.encoded_cache_fingerprint),
        )
        self.root = Path(self.root_identity.path)
        if (
            self.root_identity.device,
            self.root_identity.mount_id,
        ) != (
            authority.cache_root.device,
            authority.cache_root.mount_id,
        ):
            raise ForwardContractError("ENCODED_CACHE_MOUNT_DEVICE_DRIFT")
        self.receipt_path = self.root / "cache_receipt.json"

    def _expected_ranges(self) -> list[Tuple[int, int]]:
        return [
            (start, min(start + self.chunk_size, EXPECTED_CORPUS_VIDEO_COUNT))
            for start in range(0, EXPECTED_CORPUS_VIDEO_COUNT, self.chunk_size)
        ]

    def _chunk_paths(self, start: int, end: int) -> Tuple[Path, Path]:
        stem = "chunk_%05d_%05d" % (start, end)
        return self.root / (stem + ".pt"), self.root / (stem + ".receipt.json")

    def _validate_chunk_receipt(
        self, start: int, end: int, torch_module: Any
    ) -> Mapping[str, Any]:
        self.root_identity.verify()
        artifact_path, receipt_path = self._chunk_paths(start, end)
        receipt = _read_canonical_json(
            receipt_path, expected_parent=self.root_identity
        )
        if receipt.get("schema_version") != ENCODED_CHUNK_SCHEMA_VERSION:
            raise ForwardContractError("ENCODED_CHUNK_RECEIPT_SCHEMA_MISMATCH")
        expected = {
            "cache_fingerprint": self.authority.encoded_cache_fingerprint,
            "producer_capability_id": self.capability.capability_id,
            "producer_eval_id": self.capability.eval_id,
            "producer_token_id": self.capability.token_id,
            "start_index": start,
            "end_index": end,
            "video_count": end - start,
            "video_ids_sha256": _line_sha256(
                self.authority.manifest.corpus_identity.video_ids[start:end]
            ),
            "artifact_name": artifact_path.name,
            "durations_sha256": _sha256_bytes(
                _canonical_json_bytes(
                    list(self.authority.manifest.corpus_identity.durations_sec[start:end])
                )
            ),
            "visual_source_identity_sha256": self.authority.sources[
                self.authority.manifest.visual_source_id
            ].identity_sha256,
            "subtitle_source_identity_sha256": self.authority.sources[
                self.authority.manifest.subtitle_source_id
            ].identity_sha256,
            "temporal_manifest_sha256": self.authority.manifest.temporal_manifest.semantic_sha256,
            "feature_contract_sha256": self.authority.manifest.feature_contract.semantic_sha256,
            "encoded_dtype": "float32",
        }
        exact_receipt_fields = {
            "schema_version",
            *expected,
            "artifact_sha256",
            "payload_sha256",
            "receipt_sha256",
        }
        if set(receipt) != exact_receipt_fields:
            raise ForwardContractError("ENCODED_CHUNK_RECEIPT_FIELDS_NOT_EXACT")
        for field_name, value in expected.items():
            if receipt.get(field_name) != value:
                raise ForwardContractError("ENCODED_CHUNK_RECEIPT_BINDING_MISMATCH", field_name)
        supplied = receipt.get("receipt_sha256")
        if supplied != _semantic_sha256(receipt, ("receipt_sha256",)):
            raise ForwardContractError("ENCODED_CHUNK_RECEIPT_HASH_MISMATCH")
        artifact_data = _read_bytes_no_follow(
            artifact_path,
            max_bytes=MAX_OWNED_TORCH_ARCHIVE_BYTES,
            expected_parent=self.root_identity,
        )
        if _sha256_bytes(artifact_data) != receipt.get("artifact_sha256"):
            raise ForwardContractError("ENCODED_CHUNK_ARTIFACT_HASH_MISMATCH")
        payload = _safe_load_owned_torch_artifact(
            artifact_path,
            torch_module,
            expected_parent=self.root_identity,
        )
        if not isinstance(payload, Mapping):
            raise ForwardContractError("ENCODED_CHUNK_PAYLOAD_NOT_MAPPING")
        tensor_fields = {
            "visual_pool",
            "subtitle_pool",
            "joint_pool",
            "visual",
            "subtitle",
            "joint",
            "visual_mask",
            "subtitle_mask",
            "joint_mask",
        }
        exact_payload_fields = {
            "schema_version",
            "cache_fingerprint",
            "producer_capability_id",
            "producer_eval_id",
            "producer_token_id",
            "start_index",
            "end_index",
            "video_ids_sha256",
            "durations_sha256",
            "visual_source_identity_sha256",
            "subtitle_source_identity_sha256",
            "temporal_manifest_sha256",
            "feature_contract_sha256",
            *tensor_fields,
        }
        if set(payload) != exact_payload_fields:
            raise ForwardContractError("ENCODED_CHUNK_PAYLOAD_FIELDS_NOT_EXACT")
        if payload.get("schema_version") != ENCODED_CHUNK_SCHEMA_VERSION:
            raise ForwardContractError("ENCODED_CHUNK_PAYLOAD_SCHEMA_MISMATCH")
        for field_name, expected_value in expected.items():
            if field_name == "artifact_name" or field_name == "video_count" or field_name == "encoded_dtype":
                continue
            if payload.get(field_name) != expected_value:
                raise ForwardContractError(
                    "ENCODED_CHUNK_PAYLOAD_BINDING_MISMATCH", field_name
                )
        _validate_encoded_payload_tensors(
            torch_module,
            payload,
            row_count=end - start,
        )
        if _encoded_payload_digest(torch_module, payload) != receipt.get("payload_sha256"):
            raise ForwardContractError("ENCODED_CHUNK_PAYLOAD_HASH_MISMATCH")
        return payload

    def _encode_chunk(
        self,
        model_handle: LoadedModelHandle,
        visual_reader: GuardedLmdbReader,
        subtitle_reader: GuardedLmdbReader,
        start: int,
        end: int,
        torch_module: Any,
    ) -> Mapping[str, Any]:
        np = _lazy_numpy()
        manifest = self.authority.manifest
        self.capability.assert_io_ready()
        model = _assert_loaded_model_integrity(
            model_handle,
            authority=self.authority,
            capability=self.capability,
            torch_module=torch_module,
        )
        visual_rows = []
        subtitle_rows = []
        visual_masks = []
        subtitle_masks = []
        for video_id in manifest.corpus_identity.video_ids[start:end]:
            visual_raw = visual_reader.visual_features(video_id)
            subtitle_raw = subtitle_reader.subtitle_features(video_id)
            visual, visual_mask = _fit_sequence(
                visual_raw,
                manifest.target_length,
                allow_empty=False,
            )
            subtitle, subtitle_mask = _fit_sequence(
                subtitle_raw,
                manifest.target_length,
                allow_empty=True,
            )
            visual_rows.append(visual)
            subtitle_rows.append(subtitle)
            visual_masks.append(visual_mask)
            subtitle_masks.append(subtitle_mask)
        device = torch_module.device(EXPECTED_DEVICE)
        visual_tensor = torch_module.from_numpy(np.stack(visual_rows)).to(device).unsqueeze(1)
        subtitle_tensor = torch_module.from_numpy(np.stack(subtitle_rows)).to(device).unsqueeze(1)
        visual_mask_tensor = torch_module.from_numpy(np.stack(visual_masks)).to(device).unsqueeze(1)
        subtitle_mask_tensor = torch_module.from_numpy(np.stack(subtitle_masks)).to(device).unsqueeze(1)
        joint_mask_tensor = torch_module.logical_or(
            visual_mask_tensor,
            subtitle_mask_tensor,
        )
        self.capability._mark_model_forward_started(
            "CORPUS_VIDEO_ENCODER_FORWARD"
        )
        with torch_module.inference_mode():
            encoded = model.video_encoder(
                visual_tensor,
                subtitle_tensor,
                visual_mask=visual_mask_tensor,
                subtitle_mask=subtitle_mask_tensor,
                clip_mask=joint_mask_tensor,
            )
        self.capability._record_observation(
            "CORPUS_VIDEO_ENCODER_FORWARD",
            {
                "content_open_count": 0,
                "content_bytes_read": 0,
                "lmdb_transaction_open_count": 0,
                "model_forward_count": 1,
            },
            {
                "start_index": start,
                "end_index": end,
                "input_dtype": "float32",
                "masked_pool_from_sequence": True,
                "raw_mean_shortcut": False,
            },
        )
        payload = {
            "schema_version": ENCODED_CHUNK_SCHEMA_VERSION,
            "cache_fingerprint": self.authority.encoded_cache_fingerprint,
            "producer_capability_id": self.capability.capability_id,
            "producer_eval_id": self.capability.eval_id,
            "producer_token_id": self.capability.token_id,
            "start_index": start,
            "end_index": end,
            "video_ids_sha256": _line_sha256(manifest.corpus_identity.video_ids[start:end]),
            "durations_sha256": _sha256_bytes(
                _canonical_json_bytes(
                    list(manifest.corpus_identity.durations_sec[start:end])
                )
            ),
            "visual_source_identity_sha256": self.authority.sources[
                manifest.visual_source_id
            ].identity_sha256,
            "subtitle_source_identity_sha256": self.authority.sources[
                manifest.subtitle_source_id
            ].identity_sha256,
            "temporal_manifest_sha256": manifest.temporal_manifest.semantic_sha256,
            "feature_contract_sha256": manifest.feature_contract.semantic_sha256,
            "visual_pool": encoded["visual_pool"].squeeze(1).detach().cpu().float(),
            "subtitle_pool": encoded["subtitle_pool"].squeeze(1).detach().cpu().float(),
            "joint_pool": encoded["joint_pool"].squeeze(1).detach().cpu().float(),
            "visual": encoded["visual"].squeeze(1).detach().cpu().float(),
            "subtitle": encoded["subtitle"].squeeze(1).detach().cpu().float(),
            "joint": encoded["joint"].squeeze(1).detach().cpu().float(),
            "visual_mask": visual_mask_tensor.squeeze(1).detach().cpu().bool(),
            "subtitle_mask": subtitle_mask_tensor.squeeze(1).detach().cpu().bool(),
            "joint_mask": joint_mask_tensor.squeeze(1).detach().cpu().bool(),
        }
        _validate_encoded_payload_tensors(
            torch_module,
            payload,
            row_count=end - start,
        )
        return payload

    def materialize_or_resume(
        self, model_handle: LoadedModelHandle, torch_module: Any
    ) -> Mapping[str, Any]:
        self.authority.assert_capability(self.capability)
        self.capability.assert_io_ready()
        _assert_loaded_model_integrity(
            model_handle,
            authority=self.authority,
            capability=self.capability,
            torch_module=torch_module,
        )
        self.authority.cache_root.verify()
        self.root_identity.verify()
        visual_source = self.authority.manifest.visual_source_id
        subtitle_source = self.authority.manifest.subtitle_source_id
        chunk_receipts = []
        complete_without_source_open = True
        for start, end in self._expected_ranges():
            artifact_path, receipt_path = self._chunk_paths(start, end)
            if not artifact_path.exists() or not receipt_path.exists():
                complete_without_source_open = False
                break
            self._validate_chunk_receipt(start, end, torch_module)
            chunk_receipts.append(
                _read_canonical_json(
                    receipt_path, expected_parent=self.root_identity
                )
            )
        if complete_without_source_open:
            cache_receipt = self._complete_cache_receipt(chunk_receipts)
            if self.receipt_path.exists():
                committed = _read_canonical_json(
                    self.receipt_path, expected_parent=self.root_identity
                )
                if committed != cache_receipt:
                    raise ForwardContractError("COMMITTED_CACHE_RECEIPT_DRIFT")
            else:
                _atomic_write_bytes(
                    self.receipt_path,
                    _canonical_json_bytes(cache_receipt),
                    replace=False,
                    expected_parent=self.root_identity,
                )
            return cache_receipt
        if self.receipt_path.exists():
            raise ForwardContractError("COMMITTED_CACHE_IS_INCOMPLETE")
        chunk_receipts = []
        with GuardedLmdbReader(
            self.authority, self.capability, visual_source
        ) as visual_reader, GuardedLmdbReader(
            self.authority, self.capability, subtitle_source
        ) as subtitle_reader:
            for start, end in self._expected_ranges():
                artifact_path, receipt_path = self._chunk_paths(start, end)
                if artifact_path.exists() and receipt_path.exists():
                    self._validate_chunk_receipt(start, end, torch_module)
                    receipt = _read_canonical_json(
                        receipt_path, expected_parent=self.root_identity
                    )
                    chunk_receipts.append(receipt)
                    continue
                fresh = self._encode_chunk(
                    model_handle,
                    visual_reader,
                    subtitle_reader,
                    start,
                    end,
                    torch_module,
                )
                fresh_semantic = _encoded_payload_digest(torch_module, fresh)
                if artifact_path.exists() and not receipt_path.exists():
                    orphan = _safe_load_owned_torch_artifact(
                        artifact_path,
                        torch_module,
                        expected_parent=self.root_identity,
                    )
                    if (
                        not isinstance(orphan, Mapping)
                        or _encoded_payload_digest(torch_module, orphan) != fresh_semantic
                    ):
                        raise ForwardContractError("ENCODED_ORPHAN_NOT_REPRODUCIBLE")
                    artifact_sha = _sha256_bytes(
                        _read_bytes_no_follow(
                            artifact_path,
                            max_bytes=MAX_OWNED_TORCH_ARCHIVE_BYTES,
                            expected_parent=self.root_identity,
                        )
                    )
                    payload_sha = fresh_semantic
                elif not artifact_path.exists() and receipt_path.exists():
                    raise ForwardContractError("ENCODED_RECEIPT_WITHOUT_ARTIFACT")
                else:
                    artifact_sha, payload_sha = _atomic_torch_save(
                        artifact_path,
                        fresh,
                        torch_module,
                        expected_parent=self.root_identity,
                    )
                receipt = {
                    "schema_version": ENCODED_CHUNK_SCHEMA_VERSION,
                    "cache_fingerprint": self.authority.encoded_cache_fingerprint,
                    "producer_capability_id": self.capability.capability_id,
                    "producer_eval_id": self.capability.eval_id,
                    "producer_token_id": self.capability.token_id,
                    "start_index": start,
                    "end_index": end,
                    "video_count": end - start,
                    "video_ids_sha256": _line_sha256(
                        self.authority.manifest.corpus_identity.video_ids[start:end]
                    ),
                    "artifact_name": artifact_path.name,
                    "artifact_sha256": artifact_sha,
                    "payload_sha256": payload_sha,
                    "durations_sha256": fresh["durations_sha256"],
                    "visual_source_identity_sha256": fresh[
                        "visual_source_identity_sha256"
                    ],
                    "subtitle_source_identity_sha256": fresh[
                        "subtitle_source_identity_sha256"
                    ],
                    "temporal_manifest_sha256": fresh["temporal_manifest_sha256"],
                    "feature_contract_sha256": fresh["feature_contract_sha256"],
                    "encoded_dtype": "float32",
                }
                receipt["receipt_sha256"] = _semantic_sha256(receipt)
                _atomic_write_bytes(
                    receipt_path,
                    _canonical_json_bytes(receipt),
                    replace=False,
                    expected_parent=self.root_identity,
                )
                chunk_receipts.append(receipt)
        cache_receipt = self._complete_cache_receipt(chunk_receipts)
        _atomic_write_bytes(
            self.receipt_path,
            _canonical_json_bytes(cache_receipt),
            replace=False,
            expected_parent=self.root_identity,
        )
        return cache_receipt

    def _complete_cache_receipt(
        self, chunk_receipts: Sequence[Mapping[str, Any]]
    ) -> Mapping[str, Any]:
        if len(chunk_receipts) != len(self._expected_ranges()):
            raise ForwardContractError("ENCODED_CACHE_CHUNK_COUNT_INCOMPLETE")
        cache_receipt = {
            "schema_version": ENCODED_CACHE_SCHEMA_VERSION,
            "status": "COMPLETED",
            "cache_fingerprint": self.authority.encoded_cache_fingerprint,
            "goal_id": self.authority.manifest.goal_id,
            "authority_id": self.authority.manifest.authority_id,
            "attempt_id": self.authority.manifest.attempt_id,
            "producer_capability_id": self.capability.capability_id,
            "producer_token_id": self.capability.token_id,
            "producer_eval_id": self.capability.eval_id,
            "checkpoint_sha256": self.authority.manifest.checkpoint_sha256,
            "corpus_identity_sha256": self.authority.manifest.corpus_identity.identity_sha256,
            "video_count": EXPECTED_CORPUS_VIDEO_COUNT,
            "chunk_size": self.chunk_size,
            "chunk_count": len(chunk_receipts),
            "chunk_receipt_sha256s": [item["receipt_sha256"] for item in chunk_receipts],
            "visual_source_identity_sha256": self.authority.sources[
                self.authority.manifest.visual_source_id
            ].identity_sha256,
            "subtitle_source_identity_sha256": self.authority.sources[
                self.authority.manifest.subtitle_source_id
            ].identity_sha256,
            "temporal_manifest_sha256": self.authority.manifest.temporal_manifest.semantic_sha256,
            "feature_contract_sha256": self.authority.manifest.feature_contract.semantic_sha256,
            "model_contract_sha256": _sha256_bytes(
                _canonical_json_bytes(self.authority.manifest.model_contract.as_dict())
            ),
            "code_source_sha256": [
                list(item) for item in self.authority.manifest.code_source_sha256
            ],
            "environment_sha256": self.authority.manifest.environment_sha256,
            "encoded_dtype": "float32",
            "teacher_candidate_count": 0,
            "gt_support_count": 0,
            "gt_append_count": 0,
            "optimizer_update_count": 0,
        }
        cache_receipt["receipt_sha256"] = _semantic_sha256(cache_receipt)
        return cache_receipt

    def load_complete_handle(self, torch_module: Any) -> EncodedCorpusHandle:
        self.authority.assert_capability(self.capability)
        self.capability.assert_io_ready()
        self.authority.cache_root.verify()
        self.root_identity.verify()
        if self.authority._encoded_corpus_handle is not None:
            raise ForwardContractError("ENCODED_CORPUS_HANDLE_REPLAY_FORBIDDEN")
        receipt = _read_canonical_json(
            self.receipt_path, expected_parent=self.root_identity
        )
        if receipt.get("receipt_sha256") != _semantic_sha256(receipt, ("receipt_sha256",)):
            raise ForwardContractError("ENCODED_CACHE_RECEIPT_HASH_MISMATCH")
        if (
            receipt.get("status") != "COMPLETED"
            or receipt.get("cache_fingerprint")
            != self.authority.encoded_cache_fingerprint
            or receipt.get("video_count") != EXPECTED_CORPUS_VIDEO_COUNT
        ):
            raise ForwardContractError("ENCODED_CACHE_NOT_COMPLETE_OR_BOUND")
        cache_expected = {
            "goal_id": self.authority.manifest.goal_id,
            "authority_id": self.authority.manifest.authority_id,
            "attempt_id": self.authority.manifest.attempt_id,
            "producer_capability_id": self.capability.capability_id,
            "producer_token_id": self.capability.token_id,
            "producer_eval_id": self.capability.eval_id,
            "checkpoint_sha256": self.authority.manifest.checkpoint_sha256,
            "corpus_identity_sha256": self.authority.manifest.corpus_identity.identity_sha256,
            "visual_source_identity_sha256": self.authority.sources[
                self.authority.manifest.visual_source_id
            ].identity_sha256,
            "subtitle_source_identity_sha256": self.authority.sources[
                self.authority.manifest.subtitle_source_id
            ].identity_sha256,
            "temporal_manifest_sha256": self.authority.manifest.temporal_manifest.semantic_sha256,
            "feature_contract_sha256": self.authority.manifest.feature_contract.semantic_sha256,
            "model_contract_sha256": _sha256_bytes(
                _canonical_json_bytes(self.authority.manifest.model_contract.as_dict())
            ),
            "code_source_sha256": [
                list(item) for item in self.authority.manifest.code_source_sha256
            ],
            "environment_sha256": self.authority.manifest.environment_sha256,
            "encoded_dtype": "float32",
            "teacher_candidate_count": 0,
            "gt_support_count": 0,
            "gt_append_count": 0,
            "optimizer_update_count": 0,
        }
        for field_name, expected_value in cache_expected.items():
            if receipt.get(field_name) != expected_value:
                raise ForwardContractError(
                    "ENCODED_CACHE_RECEIPT_BINDING_MISMATCH", field_name
                )
        parts: dict[str, list[Any]] = {
            key: []
            for key in (
                "visual_pool",
                "subtitle_pool",
                "joint_pool",
                "visual",
                "subtitle",
                "joint",
                "visual_mask",
                "subtitle_mask",
                "joint_mask",
            )
        }
        receipt_shas = []
        for start, end in self._expected_ranges():
            payload = self._validate_chunk_receipt(start, end, torch_module)
            chunk_receipt = _read_canonical_json(
                self._chunk_paths(start, end)[1],
                expected_parent=self.root_identity,
            )
            receipt_shas.append(chunk_receipt["receipt_sha256"])
            for key in parts:
                parts[key].append(payload[key])
        if receipt_shas != receipt.get("chunk_receipt_sha256s"):
            raise ForwardContractError("ENCODED_CACHE_CHUNK_SEQUENCE_MISMATCH")
        payload = {key: torch_module.cat(values, dim=0) for key, values in parts.items()}
        result = EncodedCorpusHandle(
            _ENCODED_HANDLE_FACTORY,
            torch_module=torch_module,
            _payload=payload,
            video_count=EXPECTED_CORPUS_VIDEO_COUNT,
            cache_fingerprint=self.authority.encoded_cache_fingerprint,
            capability_id=self.capability.capability_id,
            receipt_sha256=receipt["receipt_sha256"],
        )
        self.authority._encoded_corpus_handle = result
        return result

    def materialize_and_load(
        self,
        model_handle: LoadedModelHandle,
    ) -> EncodedCorpusHandle:
        """Use the reviewed runtime module without exposing it to the runner."""

        torch_module = _lazy_torch()
        self.materialize_or_resume(model_handle, torch_module)
        return self.load_complete_handle(torch_module)


def _safe_soft_topk(
    torch_module: Any,
    values: Any,
    k: int,
    temperature: float,
    mask: Optional[Any],
    *,
    allow_empty_mask_rows: bool = False,
) -> Any:
    nonempty = None
    if mask is not None:
        nonempty = mask.bool().any(dim=-1)
        if not allow_empty_mask_rows and not bool(nonempty.all()):
            raise ForwardContractError("LATE_MASK_HAS_EMPTY_CANDIDATE")
        values = values.masked_fill(~mask.bool(), -1e4)
    width = min(k, int(values.shape[-1]))
    selected = torch_module.topk(values, k=width, dim=-1).values
    weights = torch_module.softmax(selected / max(float(temperature), 1e-6), dim=-1)
    result = (weights * selected).sum(dim=-1)
    if nonempty is not None and allow_empty_mask_rows:
        result = result.masked_fill(~nonempty, 0.0)
    return result


def _late_retrieval_mapping(
    model: Any,
    query_encoded: Mapping[str, Any],
    query_tokens: Any,
    query_mask: Any,
    encoded_candidates: Mapping[str, Any],
    contract: FrozenModelContract,
    torch_module: Any,
    device: Any,
) -> Mapping[str, Any]:
    enc = {
        key: value.to(device, non_blocking=True).float().unsqueeze(0)
        for key, value in encoded_candidates.items()
        if key in {"visual", "subtitle", "joint", "visual_pool", "subtitle_pool", "joint_pool"}
    }
    visual_mask = encoded_candidates["visual_mask"].to(device, non_blocking=True).bool().unsqueeze(0)
    subtitle_mask = encoded_candidates["subtitle_mask"].to(device, non_blocking=True).bool().unsqueeze(0)
    joint_mask = encoded_candidates["joint_mask"].to(device, non_blocking=True).bool().unsqueeze(0)
    pooled = model.retriever.score_candidates(query_encoded, enc)
    visual_token = torch_module.einsum("bd,bctd->bct", query_encoded["q_visual"], enc["visual"])
    subtitle_token = torch_module.einsum("bd,bctd->bct", query_encoded["q_subtitle"], enc["subtitle"])
    joint_token = torch_module.einsum("bd,bctd->bct", query_encoded["q_joint"], enc["joint"])
    visual_score = _safe_soft_topk(
        torch_module,
        visual_token,
        contract.late_soft_topk,
        contract.late_temperature,
        visual_mask,
    )
    subtitle_score = _safe_soft_topk(
        torch_module,
        subtitle_token,
        contract.late_soft_topk,
        contract.late_temperature,
        subtitle_mask,
        allow_empty_mask_rows=True,
    )
    joint_score = _safe_soft_topk(
        torch_module,
        joint_token,
        contract.late_soft_topk,
        contract.late_temperature,
        joint_mask,
    )
    gate = query_encoded["gate"]
    scale = model.retriever.scale.clamp(1.0, 30.0)
    late_score = scale * (
        gate[:, 0:1] * visual_score
        + gate[:, 1:2] * subtitle_score
        + gate[:, 2:3] * joint_score
    )
    token_score = late_score.new_zeros(late_score.shape)
    if contract.token_maxsim_weight > 0.0:
        token_hidden = torch_module.nn.functional.normalize(
            model.query_encoder.joint_proj(query_tokens), dim=-1
        )
        token_similarity = torch_module.einsum("bld,bctd->blct", token_hidden, enc["joint"])
        token_similarity = token_similarity.masked_fill(~joint_mask.unsqueeze(1), -1e4)
        token_similarity = token_similarity.max(dim=-1).values
        query_mask_float = query_mask.float().unsqueeze(-1)
        token_score = (
            token_similarity * query_mask_float
        ).sum(dim=1) / query_mask_float.sum(dim=1).clamp_min(1.0)
        token_score = scale * token_score
    combined = (
        contract.pooled_score_weight * pooled["retriever_score"]
        + contract.late_score_weight * late_score
        + contract.token_maxsim_weight * token_score
    )
    risk_features = torch_module.stack(
        [
            visual_score,
            subtitle_score,
            joint_score,
            visual_score - subtitle_score,
            combined.detach() / 30.0,
        ],
        dim=-1,
    )
    wrong_video_risk = torch_module.sigmoid(model.retriever.risk(risk_features).squeeze(-1))
    return {
        "retriever_score": combined,
        "pooled_visual_sim": pooled["visual_sim"],
        "pooled_subtitle_sim": pooled["subtitle_sim"],
        "pooled_joint_sim": pooled["joint_sim"],
        "pooled_wrong_video_risk": pooled["wrong_video_risk"],
        "retriever_score_pooled": pooled["retriever_score"],
        "retriever_score_late": late_score,
        "retriever_score_token": token_score,
        "visual_sim": visual_score,
        "subtitle_sim": subtitle_score,
        "joint_sim": joint_score,
        "late_visual_sim": visual_score,
        "late_subtitle_sim": subtitle_score,
        "late_joint_sim": joint_score,
        "wrong_video_risk": wrong_video_risk,
        "encoded": enc,
        "visual_mask": visual_mask,
        "subtitle_mask": subtitle_mask,
        "joint_mask": joint_mask,
    }


_RESULT_FACTORY = object()


class ForwardQueryResult:
    """Engine-issued, mutation-detecting result capability."""

    __slots__ = (
        "query_id",
        "query_type",
        "pooled_indices",
        "pooled_scores",
        "late_indices",
        "late_scores",
        "final_indices",
        "final_scores",
        "joint_candidate_indices",
        "joint_scores",
        "joint_probabilities",
        "spans_sec",
        "span_mask",
        "capability_id",
        "result_sha256",
        "_forensics_bytes",
        "_torch_module",
        "_issuer_nonce",
        "_factory",
    )

    def __init__(
        self,
        factory: object,
        *,
        torch_module: Any,
        issuer_nonce: object,
        capability_id: str,
        **values: Any,
    ) -> None:
        if factory is not _RESULT_FACTORY:
            raise ForwardContractError("FORWARD_RESULT_CONSTRUCTION_FORBIDDEN")
        object.__setattr__(self, "_factory", factory)
        object.__setattr__(self, "_issuer_nonce", issuer_nonce)
        object.__setattr__(self, "_torch_module", torch_module)
        object.__setattr__(self, "capability_id", capability_id)
        object.__setattr__(self, "query_id", _canonical_query_key(values["query_id"]))
        query_type = values["query_type"]
        if not isinstance(query_type, str) or query_type not in _QUERY_TYPES:
            raise ForwardContractError("FORWARD_RESULT_QUERY_TYPE_INVALID")
        object.__setattr__(self, "query_type", query_type)
        for name in (
            "pooled_indices",
            "late_indices",
            "final_indices",
            "joint_candidate_indices",
        ):
            object.__setattr__(self, name, tuple(int(item) for item in values[name]))
        for name in ("pooled_scores", "late_scores", "final_scores"):
            object.__setattr__(
                self,
                name,
                tuple(_require_finite(item, name) for item in values[name]),
            )
        for name in ("joint_scores", "joint_probabilities", "spans_sec", "span_mask"):
            object.__setattr__(
                self,
                name,
                values[name].detach().cpu().contiguous().clone(),
            )
        forensics = dict(values["forensics"])
        if forensics.get("query_type") != query_type:
            raise ForwardContractError("FORWARD_RESULT_QUERY_TYPE_FORENSICS_MISMATCH")
        forensics_bytes = _canonical_json_bytes(forensics)
        object.__setattr__(self, "_forensics_bytes", forensics_bytes)
        object.__setattr__(
            self,
            "result_sha256",
            _forward_query_result_sha256(torch_module, self),
        )

    def __setattr__(self, _name: str, _value: Any) -> None:
        raise ForwardContractError("FORWARD_RESULT_MUTATION_FORBIDDEN")

    @property
    def forensics(self) -> Mapping[str, Any]:
        self._assert_self_seal()
        return json.loads(self._forensics_bytes.decode("utf-8"))

    def _assert_self_seal(self) -> None:
        if (
            self._factory is not _RESULT_FACTORY
            or self.result_sha256
            != _forward_query_result_sha256(self._torch_module, self)
        ):
            raise ForwardContractError("FORWARD_RESULT_SEAL_INVALID")

    def metric_record(self, corpus: FrozenCorpusIdentity) -> Mapping[str, Any]:
        self._assert_self_seal()
        if (
            not isinstance(corpus, FrozenCorpusIdentity)
            or corpus.identity_sha256
            != json.loads(self._forensics_bytes.decode("utf-8"))[
                "corpus_identity_sha256"
            ]
        ):
            raise ForwardContractError("FORWARD_RESULT_CORPUS_BINDING_MISMATCH")
        proposals = []
        for candidate_offset, video_index in enumerate(self.joint_candidate_indices):
            video_id = corpus.video_ids[video_index]
            for proposal_offset in range(int(self.joint_scores.shape[1])):
                if not bool(self.span_mask[candidate_offset, proposal_offset]):
                    continue
                proposals.append(
                    {
                        "video_id": video_id,
                        "start_sec": float(self.spans_sec[candidate_offset, proposal_offset, 0]),
                        "end_sec": float(self.spans_sec[candidate_offset, proposal_offset, 1]),
                        "score": float(self.joint_scores[candidate_offset, proposal_offset]),
                        "probability": float(
                            self.joint_probabilities[candidate_offset, proposal_offset]
                        ),
                    }
                )
        return {
            "query_id": self.query_id,
            "query_type": self.query_type,
            "pooled_video_order": [
                {"video_id": corpus.video_ids[index], "score": score}
                for index, score in zip(self.pooled_indices, self.pooled_scores)
            ],
            "late_video_order": [
                {"video_id": corpus.video_ids[index], "score": score}
                for index, score in zip(self.late_indices, self.late_scores)
            ],
            "final_video_order": [
                {"video_id": corpus.video_ids[index], "score": score}
                for index, score in zip(self.final_indices, self.final_scores)
            ],
            "joint_proposals": proposals,
            "forward_output_semantics": "RAW_P_FULL_NO_NMS",
            "forensics": dict(self.forensics),
            "teacher_candidate_count": 0,
            "gt_support_count": 0,
            "gt_append_count": 0,
            "optimizer_update_count": 0,
        }


def _forward_query_result_sha256(
    torch_module: Any, result: ForwardQueryResult
) -> str:
    material = {
        "query_id": result.query_id,
        "query_type": result.query_type,
        "pooled_indices": list(result.pooled_indices),
        "pooled_scores": list(result.pooled_scores),
        "late_indices": list(result.late_indices),
        "late_scores": list(result.late_scores),
        "final_indices": list(result.final_indices),
        "final_scores": list(result.final_scores),
        "joint_candidate_indices": list(result.joint_candidate_indices),
        "joint_scores": _tensor_digest(torch_module, result.joint_scores),
        "joint_probabilities": _tensor_digest(
            torch_module, result.joint_probabilities
        ),
        "spans_sec": _tensor_digest(torch_module, result.spans_sec),
        "span_mask": _tensor_digest(torch_module, result.span_mask),
        "forensics_sha256": _sha256_bytes(result._forensics_bytes),
        "capability_id": result.capability_id,
    }
    return _sha256_bytes(_canonical_json_bytes(material))


def _forward_payload_row_result_sha256(
    torch_module: Any,
    row: Mapping[str, Any],
    *,
    capability_id: str,
) -> str:
    material = {
        "query_id": row["query_id"],
        "query_type": row["query_type"],
        "pooled_indices": list(row["pooled_indices"]),
        "pooled_scores": list(row["pooled_scores"]),
        "late_indices": list(row["late_indices"]),
        "late_scores": list(row["late_scores"]),
        "final_indices": list(row["final_indices"]),
        "final_scores": list(row["final_scores"]),
        "joint_candidate_indices": list(row["joint_candidate_indices"]),
        "joint_scores": _tensor_digest(torch_module, row["joint_scores"]),
        "joint_probabilities": _tensor_digest(
            torch_module,
            row["joint_probabilities"],
        ),
        "spans_sec": _tensor_digest(torch_module, row["spans_sec"]),
        "span_mask": _tensor_digest(torch_module, row["span_mask"]),
        "forensics_sha256": _sha256_bytes(
            _canonical_json_bytes(dict(row["forensics"]))
        ),
        "capability_id": _require_sha256(capability_id, "capability_id"),
    }
    return _sha256_bytes(_canonical_json_bytes(material))


def _assert_forward_result_seal(
    result: ForwardQueryResult,
    *,
    authority: FrozenInputAuthority,
    capability: InputCapability,
    torch_module: Any,
) -> None:
    if type(result) is not ForwardQueryResult:
        raise ForwardContractError("FORWARD_RESULT_SEAL_INVALID")
    result._assert_self_seal()
    if (
        result._factory is not _RESULT_FACTORY
        or result._issuer_nonce is not authority._result_issuer_nonce
        or result.capability_id != capability.capability_id
        or result.result_sha256 != _forward_query_result_sha256(torch_module, result)
    ):
        raise ForwardContractError("FORWARD_RESULT_SEAL_INVALID")


def _engine_result_consumption_sha256(
    *,
    capability_id: str,
    query_identity_sha256: str,
    start_index: int,
    rows: Sequence[Mapping[str, Any]],
) -> str:
    material = {
        "schema_version": "c28f_v5_engine_result_registry_consumption_v1",
        "capability_id": _require_sha256(capability_id, "capability_id"),
        "query_identity_sha256": _require_sha256(
            query_identity_sha256,
            "query_identity_sha256",
        ),
        "start_index": _require_nonnegative_int(start_index, "start_index"),
        "end_index": start_index + len(rows),
        "result_count": len(rows),
        "results": [dict(row) for row in rows],
    }
    exact_row_fields = {"query_id", "query_type", "result_sha256"}
    for index, row in enumerate(material["results"]):
        query_type = row.get("query_type")
        if (
            set(row) != exact_row_fields
            or _canonical_query_key(row.get("query_id")) != row.get("query_id")
            or not isinstance(query_type, str)
            or query_type not in _QUERY_TYPES
        ):
            raise ForwardContractError(
                "ENGINE_RESULT_CONSUMPTION_ROW_INVALID",
                str(index),
            )
        _require_sha256(row.get("result_sha256"), "result_sha256")
    return _sha256_bytes(_canonical_json_bytes(material))


def _pad_query_features(arrays: Sequence[Any], torch_module: Any, device: Any) -> Tuple[Any, Any]:
    np = _lazy_numpy()
    if not arrays:
        raise ForwardContractError("EMPTY_QUERY_BATCH")
    max_length = max(int(array.shape[0]) for array in arrays)
    dimension = int(arrays[0].shape[1])
    tokens = np.zeros((len(arrays), max_length, dimension), dtype=np.float32)
    mask = np.zeros((len(arrays), max_length), dtype=np.bool_)
    for index, array in enumerate(arrays):
        if int(array.shape[1]) != dimension:
            raise ForwardContractError("QUERY_FEATURE_DIMENSION_DRIFT")
        tokens[index, : array.shape[0]] = array
        mask[index, : array.shape[0]] = True
    return (
        torch_module.from_numpy(tokens).to(device, non_blocking=True),
        torch_module.from_numpy(mask).to(device, non_blocking=True),
    )


def _late_candidate_mask_coverage(
    encoded_late: Mapping[str, Any],
) -> Mapping[str, Any]:
    coverage: dict[str, Any] = {
        "ratio_definition": MASK_COVERAGE_RATIO_DEFINITION,
        "structure_contract": dict(_mask_structure_contract()),
        "joint_component_relation_verified": True,
    }
    expected_shape = (LATE_TOPK, TARGET_LENGTH)
    expected_denominator = LATE_TOPK * TARGET_LENGTH
    masks: dict[str, Any] = {}
    for modality, field_name in (
        ("visual", "visual_mask"),
        ("subtitle", "subtitle_mask"),
        ("joint", "joint_mask"),
    ):
        mask = encoded_late.get(field_name)
        if (
            mask is None
            or tuple(mask.shape) != expected_shape
            or str(mask.dtype) != "torch.bool"
            or int(mask.numel()) != expected_denominator
        ):
            raise ForwardContractError("LATE_MASK_COVERAGE_DENOMINATOR_INVALID")
        masks[modality] = mask
        valid_count = int(mask.sum().item())
        missing_count = expected_denominator - valid_count
        valid_ratio = valid_count / float(expected_denominator)
        missing_ratio = missing_count / float(expected_denominator)
        empty_candidate_count = int((~mask.any(dim=1)).sum().item())
        if (
            valid_count < 0
            or missing_count < 0
            or valid_count + missing_count != expected_denominator
            or empty_candidate_count < 0
            or empty_candidate_count > LATE_TOPK
            or not math.isfinite(valid_ratio)
            or not math.isfinite(missing_ratio)
            or not math.isclose(
                valid_ratio + missing_ratio,
                1.0,
                rel_tol=0.0,
                abs_tol=1e-15,
            )
        ):
            raise ForwardContractError("LATE_MASK_COVERAGE_RATIO_INVALID")
        coverage[modality] = {
            "candidate_count": LATE_TOPK,
            "denominator": expected_denominator,
            "valid_count": valid_count,
            "missing_count": missing_count,
            "valid_ratio": valid_ratio,
            "missing_ratio": missing_ratio,
            "empty_candidate_count": empty_candidate_count,
        }
    if coverage["visual"]["empty_candidate_count"] != 0:
        raise ForwardContractError("LATE_VISUAL_MASK_HAS_EMPTY_CANDIDATE")
    expected_joint = masks["visual"] | masks["subtitle"]
    if not bool((masks["joint"] == expected_joint).all()):
        raise ForwardContractError("LATE_JOINT_MASK_COMPONENT_RELATION_MISMATCH")
    if coverage["joint"]["empty_candidate_count"] != 0:
        raise ForwardContractError("LATE_JOINT_MASK_HAS_EMPTY_CANDIDATE")
    overlap_valid_count = int(
        (masks["visual"] & masks["subtitle"]).sum().item()
    )
    if (
        coverage["joint"]["valid_count"]
        != coverage["visual"]["valid_count"]
        + coverage["subtitle"]["valid_count"]
        - overlap_valid_count
    ):
        raise ForwardContractError("LATE_JOINT_MASK_COUNT_RELATION_MISMATCH")
    coverage["visual_subtitle_overlap_valid_count"] = overlap_valid_count
    return coverage


def _validate_late_candidate_mask_coverage_record(value: Any) -> None:
    if not isinstance(value, Mapping) or set(value) != {
        "ratio_definition",
        "structure_contract",
        "joint_component_relation_verified",
        "visual_subtitle_overlap_valid_count",
        "visual",
        "subtitle",
        "joint",
    } or (
        value.get("ratio_definition") != MASK_COVERAGE_RATIO_DEFINITION
        or value.get("structure_contract") != _mask_structure_contract()
        or value.get("joint_component_relation_verified") is not True
    ):
        raise ForwardContractError("LATE_MASK_COVERAGE_RECORD_INVALID")
    denominator = LATE_TOPK * TARGET_LENGTH
    exact_fields = {
        "candidate_count",
        "denominator",
        "valid_count",
        "missing_count",
        "valid_ratio",
        "missing_ratio",
        "empty_candidate_count",
    }
    for modality in ("visual", "subtitle", "joint"):
        row = value.get(modality)
        if not isinstance(row, Mapping) or set(row) != exact_fields:
            raise ForwardContractError("LATE_MASK_COVERAGE_RECORD_INVALID")
        valid_count = row.get("valid_count")
        missing_count = row.get("missing_count")
        valid_ratio = row.get("valid_ratio")
        missing_ratio = row.get("missing_ratio")
        empty_candidate_count = row.get("empty_candidate_count")
        if (
            row.get("candidate_count") != LATE_TOPK
            or row.get("denominator") != denominator
            or type(valid_count) is not int
            or type(missing_count) is not int
            or type(empty_candidate_count) is not int
            or valid_count < 0
            or missing_count < 0
            or valid_count + missing_count != denominator
            or empty_candidate_count < 0
            or empty_candidate_count > LATE_TOPK
            or valid_count < LATE_TOPK - empty_candidate_count
            or valid_count
            > (LATE_TOPK - empty_candidate_count) * TARGET_LENGTH
            or isinstance(valid_ratio, bool)
            or not isinstance(valid_ratio, (int, float))
            or isinstance(missing_ratio, bool)
            or not isinstance(missing_ratio, (int, float))
            or not math.isfinite(float(valid_ratio))
            or not math.isfinite(float(missing_ratio))
            or not math.isclose(
                float(valid_ratio),
                valid_count / float(denominator),
                rel_tol=0.0,
                abs_tol=0.0,
            )
            or not math.isclose(
                float(missing_ratio),
                missing_count / float(denominator),
                rel_tol=0.0,
                abs_tol=0.0,
            )
        ):
            raise ForwardContractError("LATE_MASK_COVERAGE_RECORD_INVALID")
    if (
        value["visual"]["empty_candidate_count"] != 0
        or value["joint"]["empty_candidate_count"] != 0
    ):
        raise ForwardContractError("LATE_MASK_COVERAGE_STRUCTURE_INVALID")
    overlap_valid_count = value.get("visual_subtitle_overlap_valid_count")
    visual_valid_count = value["visual"]["valid_count"]
    subtitle_valid_count = value["subtitle"]["valid_count"]
    joint_valid_count = value["joint"]["valid_count"]
    if (
        type(overlap_valid_count) is not int
        or overlap_valid_count < 0
        or overlap_valid_count > min(visual_valid_count, subtitle_valid_count)
        or joint_valid_count
        != visual_valid_count + subtitle_valid_count - overlap_valid_count
        or joint_valid_count > denominator
    ):
        raise ForwardContractError("LATE_MASK_COVERAGE_RELATION_INVALID")


class FrozenForwardEngine:
    """Operational teacher-free engine with process-local result provenance.

    The issued-result registry is a same-process misuse guard, not a
    cryptographic trust claim.  Authority is established only when the control
    plane durably commits the cursor hashes over the re-read chunk bytes.
    """

    def __init__(
        self,
        authority: FrozenInputAuthority,
        capability: InputCapability,
        model_handle: LoadedModelHandle,
        encoded_corpus: EncodedCorpusHandle,
    ) -> None:
        authority.assert_capability(capability)
        capability.assert_io_ready()
        authority.verify_code_sources()
        self.authority = authority
        self.capability = capability
        if (
            not isinstance(model_handle, LoadedModelHandle)
            or model_handle._factory is not _MODEL_HANDLE_FACTORY
            or model_handle is not authority._loaded_model_handle
            or model_handle.capability_id != capability.capability_id
        ):
            raise ForwardContractError("LOADED_MODEL_HANDLE_BINDING_MISMATCH")
        if (
            not isinstance(encoded_corpus, EncodedCorpusHandle)
            or encoded_corpus._factory is not _ENCODED_HANDLE_FACTORY
            or encoded_corpus is not authority._encoded_corpus_handle
            or encoded_corpus.capability_id != capability.capability_id
            or encoded_corpus.cache_fingerprint != authority.encoded_cache_fingerprint
        ):
            raise ForwardContractError("ENCODED_CORPUS_HANDLE_BINDING_MISMATCH")
        encoded_corpus.assert_runtime_integrity()
        self.model_handle = model_handle
        self.encoded_corpus = encoded_corpus
        self.torch = _lazy_torch()
        self.device = self.torch.device(EXPECTED_DEVICE)
        self._query_type_by_id = MappingProxyType(
            dict(
                zip(
                    authority.manifest.query_identity.query_keys,
                    authority.manifest.query_identity.query_types,
                )
            )
        )
        self._issued_result_registry: dict[
            int,
            tuple[ForwardQueryResult, str],
        ] = {}
        self._issued_query_ids: set[str] = set()
        self._consumed_query_ids: set[str] = set()
        temporal_module = _import_approved_business_module(
            authority,
            "blueprint_e2e_v2.data.temporal_grid",
            expected_new_module_closure=frozenset(
                {"blueprint_e2e_v2.data.temporal_grid"}
            ),
            expected_new_package_closure=(
                _TEMPORAL_RUNTIME_PACKAGE_CLOSURE
            ),
        )
        temporal_class = temporal_module.TemporalGrid
        self._temporal_grid = temporal_class(
            clip_len=authority.manifest.temporal_manifest.clip_len_sec,
            max_clips=authority.manifest.target_length,
        )
        self._temporal_grid_seal = (
            temporal_class,
            id(temporal_class.grid_spans),
            id(temporal_class.clips_to_seconds),
            tuple(sorted(vars(self._temporal_grid).items())),
        )
        self._model = _assert_loaded_model_integrity(
            model_handle,
            authority=authority,
            capability=capability,
            torch_module=self.torch,
        )
        if bool(self._model.training) or any(
            parameter.requires_grad for parameter in self._model.parameters()
        ):
            raise ForwardContractError("MODEL_NOT_FROZEN_EVAL")

    def _assert_engine_model_integrity(self) -> None:
        observed = _assert_loaded_model_integrity(
            self.model_handle,
            authority=self.authority,
            capability=self.capability,
            torch_module=self.torch,
        )
        if observed is not self._model:
            raise ForwardContractError("ENGINE_MODEL_OBJECT_DRIFT")

    def _assert_temporal_grid_integrity(self) -> None:
        temporal_class = type(self._temporal_grid)
        observed = (
            temporal_class,
            id(temporal_class.grid_spans),
            id(temporal_class.clips_to_seconds),
            tuple(sorted(vars(self._temporal_grid).items())),
        )
        if observed != self._temporal_grid_seal:
            raise ForwardContractError("TEMPORAL_GRID_RUNTIME_SEAL_INVALID")

    def _register_issued_result(self, result: ForwardQueryResult) -> None:
        _assert_forward_result_seal(
            result,
            authority=self.authority,
            capability=self.capability,
            torch_module=self.torch,
        )
        identity = id(result)
        if (
            identity in self._issued_result_registry
            or result.query_id in self._issued_query_ids
            or result.query_id in self._consumed_query_ids
        ):
            raise ForwardContractError("ENGINE_QUERY_RESULT_REISSUE_FORBIDDEN")
        if self._query_type_by_id.get(result.query_id) != result.query_type:
            raise ForwardContractError("ENGINE_RESULT_QUERY_IDENTITY_DRIFT")
        digest = _forward_query_result_sha256(self.torch, result)
        if digest != result.result_sha256:
            raise ForwardContractError("ENGINE_RESULT_REGISTRY_DIGEST_DRIFT")
        self._issued_result_registry[identity] = (result, digest)
        self._issued_query_ids.add(result.query_id)

    def _consume_issued_results_once(
        self,
        results: Sequence[ForwardQueryResult],
        *,
        start_index: int,
    ) -> str:
        if not isinstance(results, (list, tuple)) or not results:
            raise ForwardContractError("ENGINE_RESULT_BATCH_NOT_EXPLICIT")
        start = _require_nonnegative_int(start_index, "start_index")
        end = start + len(results)
        identity = self.authority.manifest.query_identity
        expected_pairs = tuple(
            zip(identity.query_keys[start:end], identity.query_types[start:end])
        )
        if len(expected_pairs) != len(results):
            raise ForwardContractError("ENGINE_RESULT_BATCH_OUT_OF_RANGE")
        seen: set[int] = set()
        consumption_rows: list[Mapping[str, Any]] = []
        for offset, (result, expected_pair) in enumerate(
            zip(results, expected_pairs)
        ):
            result_identity = id(result)
            registered = self._issued_result_registry.get(result_identity)
            if (
                type(result) is not ForwardQueryResult
                or result_identity in seen
                or registered is None
                or registered[0] is not result
            ):
                raise ForwardContractError(
                    "ENGINE_RESULT_NOT_ISSUED_OR_ALREADY_CONSUMED",
                    str(offset),
                )
            _assert_forward_result_seal(
                result,
                authority=self.authority,
                capability=self.capability,
                torch_module=self.torch,
            )
            recomputed = _forward_query_result_sha256(self.torch, result)
            if (
                (result.query_id, result.query_type) != expected_pair
                or registered[1] != recomputed
                or result.result_sha256 != recomputed
            ):
                raise ForwardContractError(
                    "ENGINE_RESULT_REGISTRY_BINDING_MISMATCH",
                    str(offset),
                )
            seen.add(result_identity)
            consumption_rows.append(
                {
                    "query_id": result.query_id,
                    "query_type": result.query_type,
                    "result_sha256": recomputed,
                }
            )
        consumption_sha256 = _engine_result_consumption_sha256(
            capability_id=self.capability.capability_id,
            query_identity_sha256=identity.identity_sha256,
            start_index=start,
            rows=consumption_rows,
        )
        for result in results:
            self._issued_result_registry.pop(id(result))
            self._issued_query_ids.remove(result.query_id)
            self._consumed_query_ids.add(result.query_id)
        return consumption_sha256

    def _pooled_scores(
        self, query_encoded: Mapping[str, Any], chunk_size: int = 512
    ) -> Mapping[str, Any]:
        score_parts: dict[str, list[Any]] = {
            "combined": [],
            "visual": [],
            "subtitle": [],
            "joint": [],
        }
        for start in range(0, EXPECTED_CORPUS_VIDEO_COUNT, chunk_size):
            end = min(start + chunk_size, EXPECTED_CORPUS_VIDEO_COUNT)
            pooled_slice = self.encoded_corpus.pooled_slice(start, end)
            bank = {
                key: value.to(self.device, non_blocking=True)
                for key, value in pooled_slice.items()
            }
            visual = query_encoded["q_visual"] @ bank["visual_pool"].T
            subtitle = query_encoded["q_subtitle"] @ bank["subtitle_pool"].T
            joint = query_encoded["q_joint"] @ bank["joint_pool"].T
            gate = query_encoded["gate"]
            combined = self._model.retriever.scale.clamp(1.0, 30.0) * (
                gate[:, 0:1] * visual
                + gate[:, 1:2] * subtitle
                + gate[:, 2:3] * joint
            )
            for key, value in (
                ("combined", combined),
                ("visual", visual),
                ("subtitle", subtitle),
                ("joint", joint),
            ):
                score_parts[key].append(value.detach().cpu())
        result = {key: self.torch.cat(parts, dim=1)[0] for key, parts in score_parts.items()}
        for key, value in result.items():
            if tuple(value.shape) != (EXPECTED_CORPUS_VIDEO_COUNT,):
                raise ForwardContractError("POOLED_FULL_CORPUS_SHAPE_MISMATCH", key)
            if not bool(self.torch.isfinite(value).all()):
                raise ForwardContractError("POOLED_SCORE_NONFINITE", key)
        self.capability._record_observation(
            "POOLED_FULL_CORPUS_FORWARD",
            {
                "content_open_count": 0,
                "content_bytes_read": 0,
                "lmdb_transaction_open_count": 0,
                "model_forward_count": 1,
            },
            {
                "video_count": EXPECTED_CORPUS_VIDEO_COUNT,
                "module_call_count": len(score_parts["combined"]),
                "full_corpus": True,
            },
        )
        return result

    def _late_scores_chunked(
        self,
        query_encoded: Mapping[str, Any],
        query_tokens: Any,
        query_mask: Any,
        broad_indices: Sequence[int],
        chunk_size: int = 64,
    ) -> Mapping[str, list[float]]:
        parts: dict[str, list[float]] = {
            "combined": [],
            "pooled": [],
            "pure_late": [],
            "token": [],
            "visual": [],
            "subtitle": [],
            "joint": [],
        }
        contract = self.authority.manifest.model_contract
        for start in range(0, len(broad_indices), chunk_size):
            indices = broad_indices[start : start + chunk_size]
            encoded = self.encoded_corpus.gather(indices, self.torch)
            retr = _late_retrieval_mapping(
                self._model,
                query_encoded,
                query_tokens,
                query_mask,
                encoded,
                contract,
                self.torch,
                self.device,
            )
            values = {
                "combined": retr["retriever_score"],
                "pooled": retr["retriever_score_pooled"],
                "pure_late": retr["retriever_score_late"],
                "token": retr["retriever_score_token"],
                "visual": retr["visual_sim"],
                "subtitle": retr["subtitle_sim"],
                "joint": retr["joint_sim"],
            }
            for key, value in values.items():
                parts[key].extend(float(item) for item in value[0].detach().cpu())
        for key, values in parts.items():
            if len(values) != POOLED_TOPK or any(not math.isfinite(value) for value in values):
                raise ForwardContractError("LATE_SCORE_COMPLETENESS_FAILURE", key)
        self.capability._record_observation(
            "LATE_BROAD1000_FORWARD",
            {
                "content_open_count": 0,
                "content_bytes_read": 0,
                "lmdb_transaction_open_count": 0,
                "model_forward_count": 1,
            },
            {
                "candidate_count": len(broad_indices),
                "module_call_count": int(math.ceil(len(broad_indices) / float(chunk_size))),
                "pure_late_and_combined_rerank_both_retained": True,
            },
        )
        return parts

    def _full_model_from_encoded(
        self,
        query_encoded: Mapping[str, Any],
        query_tokens: Any,
        query_mask: Any,
        candidate_indices: Sequence[int],
    ) -> Tuple[Any, Any, Any, Any, Any]:
        if len(candidate_indices) != FULL_MODEL_CANDIDATES:
            raise ForwardContractError("FULL_MODEL_CANDIDATE_COUNT_NOT_200")
        encoded = self.encoded_corpus.gather(candidate_indices, self.torch)
        retr = _late_retrieval_mapping(
            self._model,
            query_encoded,
            query_tokens,
            query_mask,
            encoded,
            self.authority.manifest.model_contract,
            self.torch,
            self.device,
        )
        spans_clip_rows = []
        spans_sec_rows = []
        span_mask_rows = []
        for video_index in candidate_indices:
            clip_spans, second_spans = grid_spans_for_duration(
                self.authority.manifest.corpus_identity.durations_sec[video_index],
                _grid=self._temporal_grid,
            )
            clip_row = list(clip_spans)
            second_row = list(second_spans)
            mask_row = [True] * len(clip_row)
            while len(clip_row) < MAX_PROPOSALS_PER_VIDEO:
                clip_row.append(clip_row[-1])
                second_row.append(second_row[-1])
                mask_row.append(False)
            spans_clip_rows.append(clip_row)
            spans_sec_rows.append(second_row)
            span_mask_rows.append(mask_row)
        spans_clip = self.torch.tensor(
            spans_clip_rows, dtype=self.torch.long, device=self.device
        ).unsqueeze(0)
        spans_sec = self.torch.tensor(spans_sec_rows, dtype=self.torch.float32)
        span_mask = self.torch.tensor(span_mask_rows, dtype=self.torch.bool, device=self.device).unsqueeze(0)
        enc = retr["encoded"]
        with self.torch.inference_mode():
            proposed = self._model.proposals(spans_clip)
            partial = self._model.partial(
                query_encoded, enc, retr["visual_mask"], retr["subtitle_mask"]
            )
            region = self._model.region(enc, partial)
            active = self._model.active(
                query_encoded, enc=enc, clip_mask=retr["joint_mask"]
            )
            local = self._model.localizer(
                enc,
                partial,
                region,
                active,
                retr,
                proposed,
                visual_mask=retr["visual_mask"],
                subtitle_mask=retr["subtitle_mask"],
                clip_mask=retr["joint_mask"],
            )
            feedback = self._model.feedback(
                local["span_score"],
                local["prem_span"],
                local["region_span"],
                local["amd_span"],
                local["false_positive_risk"],
                span_mask,
            )
            score = self._model.scorer(retr, local, feedback)
        self.capability._record_observation(
            "FULL_DOWNSTREAM_FROM_ENCODED200_FORWARD",
            {
                "content_open_count": 0,
                "content_bytes_read": 0,
                "lmdb_transaction_open_count": 0,
                "model_forward_count": 1,
            },
            {
                "candidate_count": len(candidate_indices),
                "raw_sequence_reencoded": False,
                "raw_full_probability_only": True,
                "nms_applied": False,
            },
        )
        joint_scores = score["vcmr_score"][0].detach().cpu().float()
        final_scores = score["video_final"][0].detach().cpu().float()
        valid_scores = joint_scores.double().masked_fill(~span_mask[0].cpu(), -self.torch.inf)
        flat = valid_scores.reshape(-1)
        probabilities = self.torch.softmax(flat, dim=0).reshape(valid_scores.shape)
        probabilities = probabilities.masked_fill(~span_mask[0].cpu(), 0.0)
        if (
            not bool(self.torch.isfinite(joint_scores).all())
            or not bool(self.torch.isfinite(final_scores).all())
            or not bool(self.torch.isfinite(probabilities).all())
            or not math.isclose(float(probabilities.sum()), 1.0, rel_tol=0.0, abs_tol=1e-12)
        ):
            raise ForwardContractError("FULL_MODEL_SCORE_OR_PROBABILITY_INVALID")
        return joint_scores, probabilities, spans_sec, span_mask[0].cpu(), final_scores

    def forward_one(self, row: Mapping[str, Any], query_reader: GuardedLmdbReader) -> ForwardQueryResult:
        self._assert_engine_model_integrity()
        self._assert_temporal_grid_integrity()
        if set(row) != {"query_id", "query_type"}:
            raise ForwardContractError("FORWARD_QUERY_FIELDS_NOT_EXACT")
        query_id = _canonical_query_key(row["query_id"])
        query_type_name = row["query_type"]
        if not isinstance(query_type_name, str) or query_type_name not in _QUERY_TYPES:
            raise ForwardContractError("INVALID_QUERY_TYPE")
        if self._query_type_by_id.get(query_id) != query_type_name:
            raise ForwardContractError("QUERY_TYPE_AUTHORITY_BINDING_MISMATCH")
        query_features = query_reader.query_features(query_id)
        query_tokens, query_mask = _pad_query_features(
            [query_features], self.torch, self.device
        )
        query_feature_token_count = int(query_mask.sum().item())
        if (
            tuple(query_mask.shape) != (1, int(query_features.shape[0]))
            or query_feature_token_count <= 0
            or query_feature_token_count != int(query_features.shape[0])
            or not bool(query_mask.all())
        ):
            raise ForwardContractError("QUERY_TOKEN_COUNT_MASK_DRIFT")
        late_maxsim_token_count = query_feature_token_count
        if self.authority.manifest.model_contract.token_maxsim_weight <= 0.0:
            raise ForwardContractError("LATE_MAXSIM_TOKEN_COUNT_DISABLED")
        query_type_tensor = self.torch.tensor(
            [_QUERY_TYPES[query_type_name]], dtype=self.torch.long, device=self.device
        )
        self.capability._mark_model_forward_started("QUERY_ENCODER_FORWARD")
        with self.torch.inference_mode():
            query_encoded = self._model.encode_query(
                query_tokens, query_type_tensor, query_mask
            )
            self.capability._record_observation(
                "QUERY_ENCODER_FORWARD",
                {
                    "content_open_count": 0,
                    "content_bytes_read": 0,
                    "lmdb_transaction_open_count": 0,
                    "model_forward_count": 1,
                },
                {"query_key_sha256": _sha256_bytes(query_id.encode("ascii"))},
            )
            pooled_components = self._pooled_scores(query_encoded)
            pooled_values = [float(value) for value in pooled_components["combined"]]
            broad_indices = stable_topk_indices(
                pooled_values,
                self.authority.manifest.corpus_identity.video_ids,
                POOLED_TOPK,
            )
            broad_scores = [pooled_values[index] for index in broad_indices]
            late_components = self._late_scores_chunked(
                query_encoded, query_tokens, query_mask, broad_indices
            )
            broad_video_ids = [
                self.authority.manifest.corpus_identity.video_ids[index]
                for index in broad_indices
            ]
            late_offsets = stable_topk_indices(
                late_components["combined"], broad_video_ids, LATE_TOPK
            )
            late_indices = [broad_indices[offset] for offset in late_offsets]
            late_scores = [late_components["combined"][offset] for offset in late_offsets]
            pure_late_offsets = stable_topk_indices(
                late_components["pure_late"], broad_video_ids, LATE_TOPK
            )
            pure_late_indices = [broad_indices[offset] for offset in pure_late_offsets]
            joint_scores, probabilities, spans_sec, span_mask, final_scores_tensor = (
                self._full_model_from_encoded(
                    query_encoded, query_tokens, query_mask, late_indices
                )
            )
        final_values = [float(value) for value in final_scores_tensor]
        late_video_ids = [
            self.authority.manifest.corpus_identity.video_ids[index]
            for index in late_indices
        ]
        final_offsets = stable_topk_indices(final_values, late_video_ids, FULL_MODEL_CANDIDATES)
        final_indices = [late_indices[offset] for offset in final_offsets]
        final_scores = [final_values[offset] for offset in final_offsets]
        encoded_late = self.encoded_corpus.gather(late_indices, self.torch)
        mask_coverage = _late_candidate_mask_coverage(encoded_late)
        late_video_ids = [
            self.authority.manifest.corpus_identity.video_ids[index]
            for index in late_indices
        ]
        durations = [
            float(self.authority.manifest.corpus_identity.durations_sec[index])
            for index in late_indices
        ]
        pure_late_video_ids = [
            self.authority.manifest.corpus_identity.video_ids[index]
            for index in pure_late_indices
        ]
        forensics = {
            "schema_version": "c28f_f0_2_query_forensics_v1",
            "query_type": query_type_name,
            "query_identity_sha256": (
                self.authority.manifest.query_identity.identity_sha256
            ),
            "query_types_sha256": (
                self.authority.manifest.query_identity.query_types_sha256
            ),
            "query_rows_sha256": (
                self.authority.manifest.query_identity.query_rows_sha256
            ),
            "query_feature_token_count": query_feature_token_count,
            "query_feature_token_count_definition": (
                QUERY_FEATURE_TOKEN_COUNT_DEFINITION
            ),
            "late_maxsim_token_count": late_maxsim_token_count,
            "late_maxsim_token_count_definition": (
                LATE_MAXSIM_TOKEN_COUNT_DEFINITION
            ),
            "feature_to_late_maxsim_invariant": (
                FEATURE_TO_LATE_MAXSIM_TOKEN_COUNT_INVARIANT
            ),
            "late_candidate_mask_coverage": mask_coverage,
            "query_source_identity_sha256": self.authority.sources[
                self.authority.manifest.query_source_id
            ].identity_sha256,
            "visual_source_identity_sha256": self.authority.sources[
                self.authority.manifest.visual_source_id
            ].identity_sha256,
            "subtitle_source_identity_sha256": self.authority.sources[
                self.authority.manifest.subtitle_source_id
            ].identity_sha256,
            "corpus_identity_sha256": self.authority.manifest.corpus_identity.identity_sha256,
            "encoded_cache_fingerprint": self.authority.encoded_cache_fingerprint,
            "temporal_manifest_sha256": self.authority.manifest.temporal_manifest.semantic_sha256,
            "feature_contract_sha256": self.authority.manifest.feature_contract.semantic_sha256,
            "broad_order_sha256": _line_sha256(broad_video_ids),
            "broad_set_sha256": _line_sha256(sorted(broad_video_ids)),
            "rerank_order_sha256": _line_sha256(late_video_ids),
            "rerank_set_sha256": _line_sha256(sorted(late_video_ids)),
            "pure_late_order_sha256": _line_sha256(pure_late_video_ids),
            "pure_late_set_sha256": _line_sha256(sorted(pure_late_video_ids)),
            "duration_vector_sha256": _sha256_bytes(_canonical_json_bytes(durations)),
            "visual_mask": _tensor_digest(self.torch, encoded_late["visual_mask"]),
            "subtitle_mask": _tensor_digest(self.torch, encoded_late["subtitle_mask"]),
            "joint_mask": _tensor_digest(self.torch, encoded_late["joint_mask"]),
            "pooled_top1000_components": {
                key: [float(pooled_components[key][index]) for index in broad_indices]
                for key in ("combined", "visual", "subtitle", "joint")
            },
            "late_broad1000_components": {
                key: list(values) for key, values in late_components.items()
            },
            "pure_late_top200_indices": pure_late_indices,
            "pure_late_top200_scores": [
                late_components["pure_late"][offset] for offset in pure_late_offsets
            ],
            "combined_rerank_top200_indices": late_indices,
            "combined_rerank_top200_scores": late_scores,
            "candidate_durations_sec": durations,
            "raw_joint_probability_sum": float(probabilities.sum()),
            "teacher_candidate_count": 0,
            "gt_support_count": 0,
            "gt_append_count": 0,
        }
        result = ForwardQueryResult(
            _RESULT_FACTORY,
            torch_module=self.torch,
            issuer_nonce=self.authority._result_issuer_nonce,
            capability_id=self.capability.capability_id,
            query_id=query_id,
            query_type=query_type_name,
            pooled_indices=broad_indices,
            pooled_scores=broad_scores,
            late_indices=late_indices,
            late_scores=late_scores,
            final_indices=final_indices,
            final_scores=final_scores,
            joint_candidate_indices=late_indices,
            joint_scores=joint_scores,
            joint_probabilities=probabilities,
            spans_sec=spans_sec,
            span_mask=span_mask,
            forensics=forensics,
        )
        self._assert_engine_model_integrity()
        self._assert_temporal_grid_integrity()
        self.encoded_corpus.assert_runtime_integrity()
        _assert_forward_result_seal(
            result,
            authority=self.authority,
            capability=self.capability,
            torch_module=self.torch,
        )
        self._register_issued_result(result)
        return result

    def iter_forward(
        self, rows: Sequence[Mapping[str, Any]], *, start_index: int = 0
    ) -> Iterator[ForwardQueryResult]:
        normalised = normalize_forward_queries(
            rows, self.authority.manifest.query_identity
        )
        cursor = _require_nonnegative_int(start_index, "start_index")
        if cursor > len(normalised):
            raise ForwardContractError("FORWARD_RESUME_CURSOR_OUT_OF_RANGE")
        with GuardedLmdbReader(
            self.authority,
            self.capability,
            self.authority.manifest.query_source_id,
        ) as query_reader:
            for row in normalised[cursor:]:
                yield self.forward_one(row, query_reader)


def _forward_result_payload(
    results: Sequence[ForwardQueryResult],
    *,
    start_index: int,
    capability: InputCapability,
    engine_registry_consumption_sha256: str,
) -> Mapping[str, Any]:
    rows = []
    for result in results:
        rows.append(
            {
                "query_id": result.query_id,
                "query_type": result.query_type,
                "result_sha256": result.result_sha256,
                "pooled_indices": list(result.pooled_indices),
                "pooled_scores": list(result.pooled_scores),
                "late_indices": list(result.late_indices),
                "late_scores": list(result.late_scores),
                "final_indices": list(result.final_indices),
                "final_scores": list(result.final_scores),
                "joint_candidate_indices": list(result.joint_candidate_indices),
                "joint_scores": result.joint_scores,
                "joint_probabilities": result.joint_probabilities,
                "spans_sec": result.spans_sec,
                "span_mask": result.span_mask,
                "forensics": dict(result.forensics),
            }
        )
    return {
        "schema_version": "c28f_v5_a4_forward_result_chunk_v1",
        "capability_id": capability.capability_id,
        "token_id": capability.token_id,
        "eval_id": capability.eval_id,
        "query_identity_sha256": (
            capability._authority.manifest.query_identity.identity_sha256
        ),
        "query_types_sha256": (
            capability._authority.manifest.query_identity.query_types_sha256
        ),
        "query_rows_sha256": (
            capability._authority.manifest.query_identity.query_rows_sha256
        ),
        "engine_registry_consumption_sha256": _require_sha256(
            engine_registry_consumption_sha256,
            "engine_registry_consumption_sha256",
        ),
        "start_index": start_index,
        "end_index": start_index + len(rows),
        "row_count": len(rows),
        "rows": rows,
        "forward_output_semantics": "RAW_P_FULL_NO_NMS",
    }


def _validate_forward_result_payload(
    torch_module: Any,
    payload: Mapping[str, Any],
    *,
    authority: Any,
    capability: Any,
    expected_start: int,
    expected_end: int,
) -> None:
    exact_payload_fields = {
        "schema_version",
        "capability_id",
        "token_id",
        "eval_id",
        "query_identity_sha256",
        "query_types_sha256",
        "query_rows_sha256",
        "engine_registry_consumption_sha256",
        "start_index",
        "end_index",
        "row_count",
        "rows",
        "forward_output_semantics",
    }
    if set(payload) != exact_payload_fields:
        raise ForwardContractError("FORWARD_RESULT_PAYLOAD_FIELDS_NOT_EXACT")
    expected = {
        "schema_version": "c28f_v5_a4_forward_result_chunk_v1",
        "capability_id": capability.capability_id,
        "token_id": capability.token_id,
        "eval_id": capability.eval_id,
        "query_identity_sha256": authority.manifest.query_identity.identity_sha256,
        "query_types_sha256": authority.manifest.query_identity.query_types_sha256,
        "query_rows_sha256": authority.manifest.query_identity.query_rows_sha256,
        "start_index": expected_start,
        "end_index": expected_end,
        "row_count": expected_end - expected_start,
        "forward_output_semantics": "RAW_P_FULL_NO_NMS",
    }
    for field_name, expected_value in expected.items():
        if payload.get(field_name) != expected_value:
            raise ForwardContractError("FORWARD_RESULT_PAYLOAD_BINDING_MISMATCH", field_name)
    rows = payload.get("rows")
    if not isinstance(rows, list) or len(rows) != expected["row_count"]:
        raise ForwardContractError("FORWARD_RESULT_ROW_COUNT_MISMATCH")
    exact_row_fields = {
        "query_id",
        "query_type",
        "result_sha256",
        "pooled_indices",
        "pooled_scores",
        "late_indices",
        "late_scores",
        "final_indices",
        "final_scores",
        "joint_candidate_indices",
        "joint_scores",
        "joint_probabilities",
        "spans_sec",
        "span_mask",
        "forensics",
    }
    expected_query_ids = authority.manifest.query_identity.query_keys[
        expected_start:expected_end
    ]
    expected_query_types = authority.manifest.query_identity.query_types[
        expected_start:expected_end
    ]
    for offset, (row, query_id, expected_query_type) in enumerate(
        zip(rows, expected_query_ids, expected_query_types)
    ):
        if not isinstance(row, Mapping) or set(row) != exact_row_fields:
            raise ForwardContractError("FORWARD_RESULT_ROW_FIELDS_NOT_EXACT", str(offset))
        if row.get("query_id") != query_id:
            raise ForwardContractError("FORWARD_RESULT_QUERY_SEQUENCE_MISMATCH", str(offset))
        _require_sha256(row.get("result_sha256"), "result_sha256")
        query_type = row.get("query_type")
        forensics = row.get("forensics")
        if (
            not isinstance(query_type, str)
            or query_type not in _QUERY_TYPES
            or query_type != expected_query_type
            or not isinstance(forensics, Mapping)
            or forensics.get("query_type") != query_type
            or forensics.get("query_identity_sha256")
            != authority.manifest.query_identity.identity_sha256
            or forensics.get("query_types_sha256")
            != authority.manifest.query_identity.query_types_sha256
            or forensics.get("query_rows_sha256")
            != authority.manifest.query_identity.query_rows_sha256
            or "query_text" in forensics
            or "query_tokens" in forensics
        ):
            raise ForwardContractError("FORWARD_RESULT_QUERY_TYPE_BINDING_MISMATCH", str(offset))
        query_feature_token_count = forensics.get("query_feature_token_count")
        late_maxsim_token_count = forensics.get("late_maxsim_token_count")
        if (
            type(query_feature_token_count) is not int
            or not 0 < query_feature_token_count <= QUERY_SEQUENCE_MAX_LENGTH
            or forensics.get("query_feature_token_count_definition")
            != QUERY_FEATURE_TOKEN_COUNT_DEFINITION
            or late_maxsim_token_count != query_feature_token_count
            or forensics.get("late_maxsim_token_count_definition")
            != LATE_MAXSIM_TOKEN_COUNT_DEFINITION
            or forensics.get("feature_to_late_maxsim_invariant")
            != FEATURE_TO_LATE_MAXSIM_TOKEN_COUNT_INVARIANT
        ):
            raise ForwardContractError(
                "FORWARD_RESULT_QUERY_FEATURE_TOKEN_BUCKET_BINDING_MISMATCH",
                str(offset),
            )
        _validate_late_candidate_mask_coverage_record(
            forensics.get("late_candidate_mask_coverage")
        )
        rank_contracts = (
            ("pooled_indices", "pooled_scores", POOLED_TOPK),
            ("late_indices", "late_scores", LATE_TOPK),
            ("final_indices", "final_scores", FULL_MODEL_CANDIDATES),
        )
        for index_field, score_field, count in rank_contracts:
            indices = row.get(index_field)
            scores = row.get(score_field)
            if (
                not isinstance(indices, list)
                or not isinstance(scores, list)
                or len(indices) != count
                or len(scores) != count
                or len(set(indices)) != count
                or any(
                    isinstance(index, bool)
                    or not isinstance(index, int)
                    or index < 0
                    or index >= EXPECTED_CORPUS_VIDEO_COUNT
                    for index in indices
                )
                or any(not math.isfinite(float(score)) for score in scores)
            ):
                raise ForwardContractError("FORWARD_RESULT_RANK_CONTRACT_MISMATCH", index_field)
        if set(row["final_indices"]) != set(row["late_indices"]):
            raise ForwardContractError("FINAL_RANK_NOT_EXACT_LATE200_SET")
        if row.get("joint_candidate_indices") != row.get("late_indices"):
            raise ForwardContractError("JOINT_CANDIDATE_ORDER_NOT_LATE200")
        tensors = {
            "joint_scores": ((FULL_MODEL_CANDIDATES, MAX_PROPOSALS_PER_VIDEO), "torch.float32"),
            "joint_probabilities": ((FULL_MODEL_CANDIDATES, MAX_PROPOSALS_PER_VIDEO), "torch.float64"),
            "spans_sec": ((FULL_MODEL_CANDIDATES, MAX_PROPOSALS_PER_VIDEO, 2), "torch.float32"),
            "span_mask": ((FULL_MODEL_CANDIDATES, MAX_PROPOSALS_PER_VIDEO), "torch.bool"),
        }
        for field_name, (shape, dtype) in tensors.items():
            value = row.get(field_name)
            if value is None or tuple(value.shape) != shape or str(value.dtype) != dtype:
                raise ForwardContractError("FORWARD_RESULT_TENSOR_CONTRACT_MISMATCH", field_name)
        if not bool(torch_module.isfinite(row["joint_scores"]).all()):
            raise ForwardContractError("FORWARD_RESULT_SCORE_NONFINITE")
        if not bool(torch_module.isfinite(row["joint_probabilities"]).all()):
            raise ForwardContractError("FORWARD_RESULT_PROBABILITY_NONFINITE")
        if not bool(torch_module.isfinite(row["spans_sec"]).all()):
            raise ForwardContractError("FORWARD_RESULT_SPAN_NONFINITE")
        if not bool(row["span_mask"].any(dim=1).all()):
            raise ForwardContractError("FORWARD_RESULT_EMPTY_PROPOSAL_MASK")
        if row["result_sha256"] != _forward_payload_row_result_sha256(
            torch_module,
            row,
            capability_id=capability.capability_id,
        ):
            raise ForwardContractError(
                "FORWARD_RESULT_ROW_SEAL_MISMATCH",
                str(offset),
            )
    expected_consumption_sha256 = _engine_result_consumption_sha256(
        capability_id=capability.capability_id,
        query_identity_sha256=authority.manifest.query_identity.identity_sha256,
        start_index=expected_start,
        rows=[
            {
                "query_id": row["query_id"],
                "query_type": row["query_type"],
                "result_sha256": row["result_sha256"],
            }
            for row in rows
        ],
    )
    if payload.get("engine_registry_consumption_sha256") != (
        expected_consumption_sha256
    ):
        raise ForwardContractError("ENGINE_RESULT_CONSUMPTION_DIGEST_DRIFT")


def _forward_result_payload_digest(torch_module: Any, payload: Mapping[str, Any]) -> str:
    rows = []
    for row in payload["rows"]:
        rows.append(
            {
                "query_id": row["query_id"],
                "query_type": row["query_type"],
                "result_sha256": row["result_sha256"],
                "pooled_indices_sha256": _sha256_bytes(
                    _canonical_json_bytes(row["pooled_indices"])
                ),
                "pooled_scores_sha256": _sha256_bytes(
                    _canonical_json_bytes(row["pooled_scores"])
                ),
                "late_indices_sha256": _sha256_bytes(
                    _canonical_json_bytes(row["late_indices"])
                ),
                "late_scores_sha256": _sha256_bytes(
                    _canonical_json_bytes(row["late_scores"])
                ),
                "final_indices_sha256": _sha256_bytes(
                    _canonical_json_bytes(row["final_indices"])
                ),
                "final_scores_sha256": _sha256_bytes(
                    _canonical_json_bytes(row["final_scores"])
                ),
                "joint_candidate_indices_sha256": _sha256_bytes(
                    _canonical_json_bytes(row["joint_candidate_indices"])
                ),
                "joint_scores": _tensor_digest(torch_module, row["joint_scores"]),
                "joint_probabilities": _tensor_digest(
                    torch_module, row["joint_probabilities"]
                ),
                "spans_sec": _tensor_digest(torch_module, row["spans_sec"]),
                "span_mask": _tensor_digest(torch_module, row["span_mask"]),
                "forensics_sha256": _sha256_bytes(
                    _canonical_json_bytes(row["forensics"])
                ),
            }
        )
    material = {
        key: payload[key]
        for key in (
            "schema_version",
            "capability_id",
            "token_id",
            "eval_id",
            "query_identity_sha256",
            "query_types_sha256",
            "query_rows_sha256",
            "engine_registry_consumption_sha256",
            "start_index",
            "end_index",
            "row_count",
            "forward_output_semantics",
        )
    }
    material["rows"] = rows
    return _sha256_bytes(_canonical_json_bytes(material))


class ForwardRunJournal:
    """Atomic same-token result chunks, cursor, heartbeat, and recovery receipts."""

    def __init__(
        self,
        authority: FrozenInputAuthority,
        capability: InputCapability,
        *,
        estimated_atomic_output_bytes: int,
    ) -> None:
        authority.assert_capability(capability)
        self.authority = authority
        self.capability = capability
        self.estimated_atomic_output_bytes = _require_positive_int(
            estimated_atomic_output_bytes, "estimated_atomic_output_bytes"
        )
        self.root_identity = _safe_child_directory(
            authority.output_root,
            (
                authority.manifest.stage,
                authority.manifest.attempt_id,
                capability.run_id,
            ),
        )
        self.root = Path(self.root_identity.path)
        if (
            self.root_identity.device,
            self.root_identity.mount_id,
        ) != (
            authority.output_root.device,
            authority.output_root.mount_id,
        ):
            raise ForwardContractError("FORWARD_OUTPUT_MOUNT_DEVICE_DRIFT")
        self.receipt_path = self.root / "forward_run_receipt.json"
        self.heartbeat_path = self.root / "heartbeat.json"
        self.observation_path = self.root / "forward_observations.json"
        self.completion_receipt_path = self.root / "control_completion_receipt.json"
        self.chunk_root_identity = _safe_child_directory(
            self.root_identity, ("query_chunks",)
        )
        self.chunk_root = Path(self.chunk_root_identity.path)
        if self.chunk_root_identity.mount_id != authority.output_root.mount_id:
            raise ForwardContractError("FORWARD_CHUNK_ROOT_MOUNT_DRIFT")
        capability._bind_controller_derived_run_artifact_roots(
            _RUN_ARTIFACT_ROOT_FACTORY,
            self.root_identity,
            self.chunk_root_identity,
        )
        self._last_disk_gate: Optional[Mapping[str, Any]] = None

    def _read_all_observation_rows(self) -> list[Mapping[str, Any]]:
        if not self.observation_path.exists():
            return []
        payload = dict(
            _read_canonical_json(
                self.observation_path,
                expected_parent=self.root_identity,
            )
        )
        if set(payload) != {
            "schema_version",
            "run_id",
            "capability_id",
            "event_count",
            "rows",
            "ledger_sha256",
        } or payload.get("schema_version") != (
            "c28f_v5_a4_forward_observation_ledger_v1"
        ):
            raise ForwardContractError("OBSERVATION_LEDGER_FIELDS_INVALID")
        if payload.get("ledger_sha256") != _semantic_sha256(
            payload, ("ledger_sha256",)
        ):
            raise ForwardContractError("OBSERVATION_LEDGER_HASH_INVALID")
        rows = payload.get("rows")
        if (
            payload.get("run_id") != self.capability.run_id
            or payload.get("capability_id") != self.capability.capability_id
            or not isinstance(rows, list)
            or payload.get("event_count") != len(rows)
        ):
            raise ForwardContractError("OBSERVATION_LEDGER_BINDING_INVALID")
        exact_row_fields = {
            "schema_version",
            "run_id",
            "token_id",
            "eval_id",
            "seq",
            "event_type",
            "counters",
            "details",
            "previous_event_sha256",
            "event_sha256",
        }
        exact_counter_fields = {
            "content_open_count",
            "content_bytes_read",
            "lmdb_transaction_open_count",
            "model_forward_count",
        }
        previous = None
        for offset, row in enumerate(rows, start=1):
            counters = row.get("counters") if isinstance(row, Mapping) else None
            if (
                not isinstance(row, Mapping)
                or set(row) != exact_row_fields
                or row.get("schema_version")
                != "c28f_v5_a4_forward_observation_v1"
                or row.get("seq") != offset
                or row.get("run_id") != self.capability.run_id
                or row.get("token_id") != self.capability.token_id
                or row.get("eval_id") != self.capability.eval_id
                or row.get("previous_event_sha256") != previous
                or not isinstance(counters, Mapping)
                or set(counters) != exact_counter_fields
                or any(type(value) is not int or value < 0 for value in counters.values())
                or not isinstance(row.get("details"), Mapping)
                or row.get("event_sha256")
                != _semantic_sha256(row, ("event_sha256",))
            ):
                raise ForwardContractError("OBSERVATION_LEDGER_ROW_INVALID")
            previous = row["event_sha256"]
        return list(rows)

    def _restore_observation_ledger(self, receipt: Mapping[str, Any]) -> None:
        committed_count = receipt.get("observation_event_count")
        committed_sha = receipt.get("observation_ledger_sha256")
        if type(committed_count) is not int or committed_count < 0:
            raise ForwardContractError("OBSERVATION_CURSOR_COUNT_INVALID")
        _require_sha256(committed_sha, "observation_ledger_sha256")
        if not self.observation_path.exists():
            if committed_count != 0 or committed_sha != _sha256_bytes(
                _canonical_json_bytes([])
            ):
                raise ForwardContractError("OBSERVATION_LEDGER_MISSING")
            if self.authority._observation_rows:
                raise ForwardContractError("OBSERVATION_LEDGER_RESTORE_CONFLICT")
            return
        rows = self._read_all_observation_rows()
        if len(rows) < committed_count:
            raise ForwardContractError("OBSERVATION_LEDGER_BINDING_INVALID")
        committed_rows = rows[:committed_count]
        if _sha256_bytes(
            _canonical_json_bytes(committed_rows)
        ) != committed_sha:
            raise ForwardContractError("OBSERVATION_LEDGER_CURSOR_PREFIX_DRIFT")
        if (
            self.authority._observation_rows
            and self.authority._observation_rows != committed_rows
        ):
            raise ForwardContractError("OBSERVATION_LEDGER_RESTORE_CONFLICT")
        self.authority._observation_rows = list(committed_rows)
        if len(rows) != committed_count:
            truncated = {
                "schema_version": "c28f_v5_a4_forward_observation_ledger_v1",
                "run_id": self.capability.run_id,
                "capability_id": self.capability.capability_id,
                "event_count": committed_count,
                "rows": list(committed_rows),
            }
            truncated["ledger_sha256"] = _semantic_sha256(truncated)
            _atomic_write_bytes(
                self.observation_path,
                _canonical_json_bytes(truncated),
                replace=True,
                expected_parent=self.root_identity,
            )

    def _disk_gate(self, additional_bytes: int) -> None:
        self.authority.output_root.verify()
        self.root_identity.verify()
        self.chunk_root_identity.verify()
        item = os.statvfs(self.root)
        free_bytes = int(item.f_bavail) * int(item.f_frsize)
        estimate = max(self.estimated_atomic_output_bytes, additional_bytes)
        required = 2 * estimate + max(1 << 30, int(math.ceil(estimate * 0.10)))
        if free_bytes < required:
            raise ForwardContractError("ATOMIC_OUTPUT_DISK_GATE_FAILED")
        gate = {
            "schema_version": "c28f_v5_a4_disk_gate_v1",
            "root_id": self.authority.manifest.output_root_id,
            "run_id": self.capability.run_id,
            "required_free_bytes": required,
            "observed_free_bytes": free_bytes,
            "result": "PASS",
        }
        self._last_disk_gate = {
            **gate,
            "gate_sha256": _semantic_sha256(gate),
        }

    def _heartbeat(self, phase: str, cursor: int, receipt_sha256: str) -> None:
        heartbeat = {
            "schema_version": "c28f_v5_a4_forward_heartbeat_v1",
            "phase": _require_text(phase, "phase"),
            "cursor": _require_nonnegative_int(cursor, "cursor"),
            "token_id": self.capability.token_id,
            "eval_id": self.capability.eval_id,
            "run_id": self.capability.run_id,
            "run_receipt_sha256": _require_sha256(receipt_sha256, "receipt_sha256"),
        }
        heartbeat["heartbeat_sha256"] = _semantic_sha256(heartbeat)
        _atomic_write_bytes(
            self.heartbeat_path,
            _canonical_json_bytes(heartbeat),
            replace=True,
            expected_parent=self.root_identity,
        )

    def _encoded_cache_binding_from_disk(self) -> Tuple[str, str]:
        cache_identity = _safe_child_directory(
            self.authority.cache_root,
            ("encoded_corpus", self.authority.encoded_cache_fingerprint),
        )
        receipt = _read_canonical_json(
            Path(cache_identity.path) / "cache_receipt.json",
            expected_parent=cache_identity,
        )
        if (
            receipt.get("schema_version") != ENCODED_CACHE_SCHEMA_VERSION
            or receipt.get("status") != "COMPLETED"
            or receipt.get("cache_fingerprint")
            != self.authority.encoded_cache_fingerprint
            or receipt.get("producer_capability_id")
            != self.capability.capability_id
            or receipt.get("producer_token_id") != self.capability.token_id
            or receipt.get("producer_eval_id") != self.capability.eval_id
            or receipt.get("video_count") != EXPECTED_CORPUS_VIDEO_COUNT
            or receipt.get("receipt_sha256")
            != _semantic_sha256(receipt, ("receipt_sha256",))
        ):
            raise ForwardContractError("ENCODED_CACHE_RECOVERY_BINDING_INVALID")
        return (
            _require_sha256(
                receipt["receipt_sha256"],
                "encoded_cache_receipt_sha256",
            ),
            self.authority.encoded_cache_fingerprint,
        )

    def _load_control_cursor_capability(self) -> Any:
        module = _require_exact_forward_control(self.capability._control)
        try:
            return self.capability._control.issue_eval_cursor_capability(
                running_capability=self.capability._running_control_capability,
            )
        except module.A4EvalCursorAbsentError as error:
            if type(error) is not module.A4EvalCursorAbsentError:
                raise
            return None

    def _persist_observation_snapshot(
        self,
        receipt: Mapping[str, Any],
    ) -> Tuple[str, str]:
        count = int(receipt["observation_event_count"])
        rows = list(self.authority._observation_rows)
        rows_sha256 = _sha256_bytes(_canonical_json_bytes(rows))
        if (
            len(rows) != count
            or rows_sha256 != receipt["observation_ledger_sha256"]
        ):
            raise ForwardContractError("OBSERVATION_SNAPSHOT_CURSOR_DRIFT")
        payload = {
            "schema_version": "c28f_v5_a4_forward_observation_snapshot_v1",
            "run_id": self.capability.run_id,
            "capability_id": self.capability.capability_id,
            "event_count": count,
            "rows_sha256": rows_sha256,
            "rows": rows,
        }
        payload["snapshot_sha256"] = _semantic_sha256(payload)
        encoded = _canonical_json_bytes(payload)
        path = self.root / (
            "observations_%08d_%s.json" % (count, rows_sha256)
        )
        if path.exists():
            if _read_bytes_no_follow(
                path,
                max_bytes=128 * 1024 * 1024,
                expected_parent=self.root_identity,
            ) != encoded:
                raise ForwardContractError("OBSERVATION_SNAPSHOT_IMMUTABLE_DRIFT")
        else:
            _atomic_write_bytes(
                path,
                encoded,
                replace=False,
                expected_parent=self.root_identity,
            )
        return _sha256_bytes(encoded), rows_sha256

    def _cursor_bindings(self, receipt: Mapping[str, Any]) -> Mapping[str, Any]:
        chunks = list(receipt["result_chunks"])
        observation_file_sha256, observation_rows_sha256 = (
            self._persist_observation_snapshot(receipt)
        )
        artifact_index = []
        for chunk in chunks:
            artifact_path = self.chunk_root / str(chunk["artifact_name"])
            artifact_bytes = _read_bytes_no_follow(
                artifact_path,
                max_bytes=MAX_OWNED_TORCH_ARCHIVE_BYTES,
                expected_parent=self.chunk_root_identity,
            )
            observed_artifact_sha256 = _sha256_bytes(artifact_bytes)
            if observed_artifact_sha256 != chunk["artifact_sha256"]:
                raise ForwardContractError(
                    "CONTROL_CURSOR_CHUNK_BYTES_HASH_DRIFT"
                )
            artifact_index.append(
                {
                    "artifact_name": chunk["artifact_name"],
                    "artifact_sha256": observed_artifact_sha256,
                    "semantic_sha256": chunk["semantic_sha256"],
                    "engine_registry_consumption_sha256": chunk[
                        "engine_registry_consumption_sha256"
                    ],
                    "query_rows_sha256": chunk["query_rows_sha256"],
                    "query_types_sha256": chunk["query_types_sha256"],
                }
            )
        if receipt.get("encoded_cache_receipt_sha256") is not None:
            artifact_index.append(
                {
                    "artifact_name": "encoded_corpus_cache_receipt",
                    "artifact_sha256": receipt["encoded_cache_receipt_sha256"],
                    "semantic_sha256": receipt["encoded_cache_fingerprint"],
                }
            )
        artifact_index.append(
            {
                "artifact_name": "forward_observation_ledger",
                "artifact_sha256": observation_file_sha256,
                "semantic_sha256": observation_rows_sha256,
            }
        )
        return {
            "cursor_sha256": receipt["receipt_sha256"],
            "chunks_sha256": _sha256_bytes(_canonical_json_bytes(chunks)),
            "artifacts_sha256": _sha256_bytes(
                _canonical_json_bytes(artifact_index)
            ),
        }

    def _validate_cursor_capability(
        self,
        cursor_capability: Any,
        receipt: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        module = _require_exact_forward_control(self.capability._control)
        payload = module.validate_eval_cursor_capability(cursor_capability)
        running = payload.get("running_anchor")
        expected_running = module.validate_eval_running_capability(
            self.capability._running_control_capability
        )
        expected = self._cursor_bindings(receipt)
        if (
            running != expected_running
            or payload.get("cursor_index") != len(receipt["result_chunks"])
            or any(payload.get(field) != value for field, value in expected.items())
        ):
            raise ForwardContractError("CONTROL_CURSOR_CAPABILITY_BINDING_MISMATCH")
        return payload

    def _commit_or_reconcile_cursor(
        self,
        receipt: Mapping[str, Any],
        *,
        allow_in_memory_successor: bool = False,
    ) -> Mapping[str, Any]:
        expected_index = len(receipt["result_chunks"])
        bindings = self._cursor_bindings(receipt)
        module = _require_exact_forward_control(self.capability._control)
        current_capability = self._load_control_cursor_capability()
        if current_capability is None and expected_index != 0:
            raise ForwardContractError("CONTROL_CURSOR_MISSING_FOR_LOCAL_SUCCESSOR")
        if current_capability is not None:
            current = module.validate_eval_cursor_capability(current_capability)
            if (
                current.get("cursor_index") == expected_index
                and all(current.get(field) == value for field, value in bindings.items())
            ):
                return current
            previous_sha = (
                receipt["result_chunks"][-1].get("previous_run_receipt_sha256")
                if expected_index > 0
                else None
            )
            if not (
                expected_index > 0
                and current.get("cursor_index") == expected_index - 1
                and current.get("cursor_sha256") == previous_sha
            ):
                raise ForwardContractError("CONTROL_CURSOR_LOCAL_DRIFT")
            if not allow_in_memory_successor:
                raise ForwardContractError(
                    "RESTART_REJECTS_LOCAL_UNCOMMITTED_CURSOR_SUCCESSOR"
                )
        committed = self.capability._control.commit_eval_cursor(
            cursor_index=expected_index,
            **bindings,
        )
        if (
            not isinstance(committed, Mapping)
            or committed.get("receipt_sha256")
            != _semantic_sha256(committed, ("receipt_sha256",))
        ):
            raise ForwardContractError("CONTROL_CURSOR_COMMIT_RECEIPT_INVALID")
        issued = self.capability._control.issue_eval_cursor_capability(
            running_capability=self.capability._running_control_capability,
        )
        return self._validate_cursor_capability(issued, receipt)

    def _recover_control_committed_successor(
        self,
        receipt: Mapping[str, Any],
        torch_module: Any,
    ) -> Mapping[str, Any]:
        """Recover only a successor already authenticated by the control CAS."""

        current_capability = self._load_control_cursor_capability()
        if current_capability is None:
            return receipt
        module = _require_exact_forward_control(self.capability._control)
        current = module.validate_eval_cursor_capability(current_capability)
        local_index = len(receipt["result_chunks"])
        control_index = current.get("cursor_index")
        if control_index == local_index:
            return receipt
        if control_index != local_index + 1 or receipt.get("status") != "RUNNING":
            raise ForwardContractError("CONTROL_CURSOR_LOCAL_DRIFT")
        start = int(receipt["next_query_index"])
        committed_names = {
            str(chunk["artifact_name"]) for chunk in receipt["result_chunks"]
        }
        directory_fd = _open_directory_descriptor(
            self.chunk_root_identity.path,
            self.chunk_root_identity,
        )
        try:
            names = os.listdir(directory_fd)
        finally:
            os.close(directory_fd)
        candidates = []
        for name in names:
            if name in committed_names or not (
                name.startswith("queries_") and name.endswith(".pt")
            ):
                continue
            pieces = name[len("queries_") : -len(".pt")].split("_")
            if len(pieces) != 2 or any(not piece.isdigit() for piece in pieces):
                continue
            candidate_start, candidate_end = (int(piece) for piece in pieces)
            if (
                name != "queries_%05d_%05d.pt" % (candidate_start, candidate_end)
                or candidate_start != start
                or candidate_end <= start
                or candidate_end
                > len(self.authority.manifest.query_identity.query_keys)
            ):
                continue
            candidates.append((name, candidate_end))
        if len(candidates) != 1:
            raise ForwardContractError("CONTROL_COMMITTED_CHUNK_RECOVERY_AMBIGUOUS")
        artifact_name, end = candidates[0]
        artifact_path = self.chunk_root / artifact_name
        artifact_bytes = _read_bytes_no_follow(
            artifact_path,
            max_bytes=MAX_OWNED_TORCH_ARCHIVE_BYTES,
            expected_parent=self.chunk_root_identity,
        )
        payload = _safe_load_owned_torch_artifact(
            artifact_path,
            torch_module,
            expected_parent=self.chunk_root_identity,
        )
        if not isinstance(payload, Mapping):
            raise ForwardContractError("CONTROL_COMMITTED_CHUNK_PAYLOAD_INVALID")
        _validate_forward_result_payload(
            torch_module,
            payload,
            authority=self.authority,
            capability=self.capability,
            expected_start=start,
            expected_end=end,
        )
        semantic_sha256 = _forward_result_payload_digest(torch_module, payload)
        cache_receipt_sha256, cache_fingerprint = (
            self._encoded_cache_binding_from_disk()
        )
        candidate_rows = self._read_all_observation_rows()
        previous_rows = self.authority._observation_rows
        recovered = False
        try:
            self.authority._observation_rows = list(candidate_rows)
            candidate_receipt = advance_run_receipt(
                self.authority,
                self.capability,
                receipt,
                end_query_index=end,
                artifact_name=artifact_name,
                artifact_sha256=_sha256_bytes(artifact_bytes),
                semantic_sha256=semantic_sha256,
                engine_registry_consumption_sha256=payload[
                    "engine_registry_consumption_sha256"
                ],
                encoded_cache_receipt_sha256=cache_receipt_sha256,
                encoded_cache_fingerprint=cache_fingerprint,
            )
            self._validate_cursor_capability(
                current_capability,
                candidate_receipt,
            )
            recovered = True
        finally:
            if not recovered:
                self.authority._observation_rows = previous_rows
        _atomic_write_bytes(
            self.receipt_path,
            _canonical_json_bytes(candidate_receipt),
            replace=True,
            expected_parent=self.root_identity,
        )
        self.capability._bind_persisted_run_receipt(
            candidate_receipt["receipt_sha256"],
            str(self.receipt_path),
            self.root_identity,
            self.chunk_root_identity,
        )
        return candidate_receipt

    def _complete_control_token(self, receipt: Mapping[str, Any]) -> Mapping[str, Any]:
        if receipt.get("status") != "COMPLETED":
            raise ForwardContractError("CONTROL_COMPLETION_REQUIRES_COMPLETE_CURSOR")
        cursor_capability = self._load_control_cursor_capability()
        if cursor_capability is None:
            raise ForwardContractError(
                "CONTROL_COMPLETION_CURSOR_CAPABILITY_MISSING"
            )
        self._validate_cursor_capability(cursor_capability, receipt)
        completed = dict(
            self.capability._control.complete_eval_token(
                cursor_capability=cursor_capability,
            )
        )
        if (
            completed.get("schema_version") != "c28f_eval_token_phase_receipt_v1"
            or completed.get("phase") != "COMPLETED"
            or completed.get("status") != "FSYNCED_CAS_COMMITTED"
            or completed.get("eval_id") != self.capability.eval_id
            or completed.get("run_id") != self.capability.run_id
            or completed.get("receipt_sha256")
            != _semantic_sha256(completed, ("receipt_sha256",))
        ):
            raise ForwardContractError("CONTROL_COMPLETION_RECEIPT_INVALID")
        encoded = _canonical_json_bytes(completed)
        if os.path.lexists(self.completion_receipt_path):
            observed = _read_canonical_json(
                self.completion_receipt_path,
                expected_parent=self.root_identity,
            )
            if observed != completed:
                raise ForwardContractError("CONTROL_COMPLETION_RECEIPT_DRIFT")
        else:
            _atomic_write_bytes(
                self.completion_receipt_path,
                encoded,
                replace=False,
                expected_parent=self.root_identity,
            )
        return completed

    def start_or_resume(self) -> Mapping[str, Any]:
        self.capability.assert_running_committed()
        self.authority.output_root.verify()
        self.root_identity.verify()
        self._disk_gate(4096)
        if self.receipt_path.exists():
            receipt = _read_canonical_json(
                self.receipt_path, expected_parent=self.root_identity
            )
            validate_run_receipt(self.authority, self.capability, receipt)
            self.capability._bind_persisted_run_receipt(
                receipt["receipt_sha256"],
                str(self.receipt_path),
                self.root_identity,
                self.chunk_root_identity,
            )
            torch_module = _lazy_torch()
            for chunk in receipt["result_chunks"]:
                artifact = self.chunk_root / chunk["artifact_name"]
                if _sha256_bytes(
                    _read_bytes_no_follow(
                        artifact,
                        max_bytes=MAX_OWNED_TORCH_ARCHIVE_BYTES,
                        expected_parent=self.chunk_root_identity,
                    )
                ) != chunk["artifact_sha256"]:
                    raise ForwardContractError("RESULT_CHUNK_HASH_DRIFT_ON_RESUME")
                payload = _safe_load_owned_torch_artifact(
                    artifact,
                    torch_module,
                    expected_parent=self.chunk_root_identity,
                )
                if (
                    not isinstance(payload, Mapping)
                    or payload.get("engine_registry_consumption_sha256")
                    != chunk["engine_registry_consumption_sha256"]
                    or payload.get("query_types_sha256")
                    != chunk["query_types_sha256"]
                    or payload.get("query_rows_sha256")
                    != chunk["query_rows_sha256"]
                ):
                    raise ForwardContractError("RESULT_CHUNK_SEMANTIC_DRIFT_ON_RESUME")
                _validate_forward_result_payload(
                    torch_module,
                    payload,
                    authority=self.authority,
                    capability=self.capability,
                    expected_start=int(chunk["start_index"]),
                    expected_end=int(chunk["end_index"]),
                )
                if (
                    _forward_result_payload_digest(torch_module, payload)
                    != chunk["semantic_sha256"]
                ):
                    raise ForwardContractError("RESULT_CHUNK_SEMANTIC_DRIFT_ON_RESUME")
            receipt = self._recover_control_committed_successor(
                receipt,
                torch_module,
            )
            self._restore_observation_ledger(receipt)
            self._commit_or_reconcile_cursor(receipt)
            if receipt["status"] == "COMPLETED":
                self._complete_control_token(receipt)
                return receipt
        else:
            receipt = initial_run_receipt(self.authority, self.capability)
            _atomic_write_bytes(
                self.receipt_path,
                _canonical_json_bytes(receipt),
                replace=False,
                expected_parent=self.root_identity,
            )
            self.capability._bind_persisted_run_receipt(
                receipt["receipt_sha256"],
                str(self.receipt_path),
                self.root_identity,
                self.chunk_root_identity,
            )
            self._commit_or_reconcile_cursor(
                receipt,
                allow_in_memory_successor=True,
            )
        self.capability._bind_persisted_run_receipt(
            receipt["receipt_sha256"],
            str(self.receipt_path),
            self.root_identity,
            self.chunk_root_identity,
        )
        self._heartbeat("RUNNING", int(receipt["next_query_index"]), receipt["receipt_sha256"])
        return receipt

    def commit_query_chunk(
        self,
        receipt: Mapping[str, Any],
        results: Sequence[ForwardQueryResult],
        engine: FrozenForwardEngine,
    ) -> Mapping[str, Any]:
        self.capability.assert_io_ready()
        validate_run_receipt(self.authority, self.capability, receipt)
        if not results:
            raise ForwardContractError("EMPTY_FORWARD_RESULT_CHUNK")
        if (
            type(engine) is not FrozenForwardEngine
            or engine.authority is not self.authority
            or engine.capability is not self.capability
        ):
            raise ForwardContractError("EXACT_BOUND_FORWARD_ENGINE_REQUIRED")
        torch_module = engine.torch
        start = int(receipt["next_query_index"])
        observed_query_ids = tuple(result.query_id for result in results)
        expected_query_ids = self.authority.manifest.query_identity.query_keys[
            start : start + len(results)
        ]
        if observed_query_ids != expected_query_ids:
            raise ForwardContractError("FORWARD_RESULT_QUERY_SEQUENCE_MISMATCH")
        engine_registry_consumption_sha256 = (
            engine._consume_issued_results_once(
                results,
                start_index=start,
            )
        )
        payload = _forward_result_payload(
            results,
            start_index=start,
            capability=self.capability,
            engine_registry_consumption_sha256=(
                engine_registry_consumption_sha256
            ),
        )
        _validate_forward_result_payload(
            torch_module,
            payload,
            authority=self.authority,
            capability=self.capability,
            expected_start=start,
            expected_end=start + len(results),
        )
        semantic_sha256 = _forward_result_payload_digest(torch_module, payload)
        buffer = io.BytesIO()
        torch_module.save(dict(payload), buffer)
        data = buffer.getvalue()
        if len(data) > MAX_OWNED_TORCH_ARCHIVE_BYTES:
            raise ForwardContractError("RESULT_CHUNK_ARCHIVE_SIZE_LIMIT")
        self._disk_gate(len(data))
        artifact_name = "queries_%05d_%05d.pt" % (start, start + len(results))
        artifact_path = self.chunk_root / artifact_name
        if artifact_path.exists():
            existing = _read_bytes_no_follow(
                artifact_path,
                max_bytes=MAX_OWNED_TORCH_ARCHIVE_BYTES,
                expected_parent=self.chunk_root_identity,
            )
            existing_payload = _safe_load_owned_torch_artifact(
                artifact_path,
                torch_module,
                expected_parent=self.chunk_root_identity,
            )
            if not isinstance(existing_payload, Mapping):
                raise ForwardContractError("RESULT_CHUNK_ORPHAN_NOT_REPRODUCIBLE")
            _validate_forward_result_payload(
                torch_module,
                existing_payload,
                authority=self.authority,
                capability=self.capability,
                expected_start=start,
                expected_end=start + len(results),
            )
            if _forward_result_payload_digest(torch_module, existing_payload) != semantic_sha256:
                raise ForwardContractError("RESULT_CHUNK_ORPHAN_NOT_REPRODUCIBLE")
            artifact_sha256 = _sha256_bytes(existing)
        else:
            artifact_sha256 = _atomic_write_bytes(
                artifact_path,
                data,
                replace=False,
                expected_parent=self.chunk_root_identity,
            )
        self.authority.flush_forward_observations(self.capability)
        updated = advance_run_receipt(
            self.authority,
            self.capability,
            receipt,
            end_query_index=start + len(results),
            artifact_name=artifact_name,
            artifact_sha256=artifact_sha256,
            semantic_sha256=semantic_sha256,
            engine_registry_consumption_sha256=(
                engine_registry_consumption_sha256
            ),
            encoded_cache_receipt_sha256=(
                self.authority._encoded_corpus_handle.receipt_sha256
                if self.authority._encoded_corpus_handle is not None
                else None
            ),
            encoded_cache_fingerprint=(
                self.authority._encoded_corpus_handle.cache_fingerprint
                if self.authority._encoded_corpus_handle is not None
                else None
            ),
        )
        self._commit_or_reconcile_cursor(
            updated,
            allow_in_memory_successor=True,
        )
        _atomic_write_bytes(
            self.receipt_path,
            _canonical_json_bytes(updated),
            replace=True,
            expected_parent=self.root_identity,
        )
        self.capability._bind_persisted_run_receipt(
            updated["receipt_sha256"],
            str(self.receipt_path),
            self.root_identity,
            self.chunk_root_identity,
        )
        self._heartbeat(
            str(updated["status"]),
            int(updated["next_query_index"]),
            updated["receipt_sha256"],
        )
        if updated["status"] == "COMPLETED":
            self._complete_control_token(updated)
        return updated


def initial_run_receipt(
    authority: FrozenInputAuthority,
    capability: InputCapability,
) -> Mapping[str, Any]:
    authority.assert_capability(capability)
    receipt = {
        "schema_version": FORWARD_RUN_RECEIPT_SCHEMA_VERSION,
        "status": "RUNNING",
        "goal_id": capability.goal_id,
        "authority_id": capability.authority_id,
        "run_id": capability.run_id,
        "stage": capability.stage,
        "purpose": capability.purpose,
        "token_id": capability.token_id,
        "eval_id": capability.eval_id,
        "forward_manifest_sha256": capability.forward_manifest_sha256,
        "input_identity_sha256": capability.input_identity_sha256,
        "query_identity_sha256": authority.manifest.query_identity.identity_sha256,
        "query_types_sha256": authority.manifest.query_identity.query_types_sha256,
        "query_rows_sha256": authority.manifest.query_identity.query_rows_sha256,
        "authority_binding_sha256": capability.authority_binding_sha256,
        "token_snapshot_sha256": capability.token_snapshot_sha256,
        "gpu0_identity_sha256": capability.gpu0_identity_sha256,
        "running_capability_sha256": capability.running_capability_sha256,
        "reservation_input_capability_sha256": (
            capability.reservation_input_capability_sha256
        ),
        "controller_authority_binding_sha256": (
            capability.controller_authority_binding_sha256
        ),
        "query_count": len(authority.manifest.query_identity.query_keys),
        "next_query_index": 0,
        "result_chunks": [],
        "encoded_cache_receipt_sha256": None,
        "encoded_cache_fingerprint": None,
        "observation_event_count": authority.observation_event_count,
        "observation_ledger_sha256": authority.observation_ledger_sha256,
        "teacher_candidate_count": 0,
        "teacher_forward_count": 0,
        "gt_support_count": 0,
        "gt_append_count": 0,
        "optimizer_update_count": 0,
    }
    receipt["receipt_sha256"] = _semantic_sha256(receipt)
    return receipt


def validate_run_receipt(
    authority: Any,
    capability: Any,
    receipt: Mapping[str, Any],
) -> None:
    if (
        type(authority) is CompletedForwardResultVerificationAuthority
        and type(capability) is CompletedForwardResultCapability
    ):
        authority.assert_completed_capability(capability)
    elif (
        type(authority) is FrozenInputAuthority
        and type(capability) is InputCapability
    ):
        authority.assert_capability(capability)
    else:
        raise ForwardContractError("EXACT_RESULT_AUTHORITY_PAIR_REQUIRED")
    _validate_run_receipt_contents(authority, capability, receipt)


def _validate_run_receipt_contents(
    authority: Any,
    capability: Any,
    receipt: Mapping[str, Any],
) -> None:
    """Validate receipt bytes after an exact pair was gated by its caller."""

    if receipt.get("schema_version") != FORWARD_RUN_RECEIPT_SCHEMA_VERSION:
        raise ForwardContractError("RUN_RECEIPT_SCHEMA_MISMATCH")
    if receipt.get("receipt_sha256") != _semantic_sha256(receipt, ("receipt_sha256",)):
        raise ForwardContractError("RUN_RECEIPT_HASH_MISMATCH")
    expected = {
        "goal_id": capability.goal_id,
        "authority_id": capability.authority_id,
        "run_id": capability.run_id,
        "stage": capability.stage,
        "purpose": capability.purpose,
        "token_id": capability.token_id,
        "eval_id": capability.eval_id,
        "forward_manifest_sha256": capability.forward_manifest_sha256,
        "input_identity_sha256": capability.input_identity_sha256,
        "query_identity_sha256": authority.manifest.query_identity.identity_sha256,
        "query_types_sha256": authority.manifest.query_identity.query_types_sha256,
        "query_rows_sha256": authority.manifest.query_identity.query_rows_sha256,
        "authority_binding_sha256": capability.authority_binding_sha256,
        "token_snapshot_sha256": capability.token_snapshot_sha256,
        "gpu0_identity_sha256": capability.gpu0_identity_sha256,
        "running_capability_sha256": capability.running_capability_sha256,
        "reservation_input_capability_sha256": (
            capability.reservation_input_capability_sha256
        ),
        "controller_authority_binding_sha256": (
            capability.controller_authority_binding_sha256
        ),
        "query_count": len(authority.manifest.query_identity.query_keys),
        "teacher_candidate_count": 0,
        "teacher_forward_count": 0,
        "gt_support_count": 0,
        "gt_append_count": 0,
        "optimizer_update_count": 0,
    }
    exact_receipt_fields = {
        "schema_version",
        "status",
        *expected,
        "next_query_index",
        "result_chunks",
        "encoded_cache_receipt_sha256",
        "encoded_cache_fingerprint",
        "observation_event_count",
        "observation_ledger_sha256",
        "receipt_sha256",
    }
    if set(receipt) != exact_receipt_fields:
        raise ForwardContractError("RUN_RECEIPT_FIELDS_NOT_EXACT")
    for field_name, expected_value in expected.items():
        if receipt.get(field_name) != expected_value:
            raise ForwardContractError("RUN_RECEIPT_BINDING_MISMATCH", field_name)
    if receipt.get("status") not in {"RUNNING", "COMPLETED"}:
        raise ForwardContractError("RUN_RECEIPT_STATUS_INVALID")
    cursor = receipt.get("next_query_index")
    if isinstance(cursor, bool) or not isinstance(cursor, int) or not 0 <= cursor <= expected["query_count"]:
        raise ForwardContractError("RUN_RECEIPT_CURSOR_INVALID")
    chunks = receipt.get("result_chunks")
    if not isinstance(chunks, list):
        raise ForwardContractError("RUN_RECEIPT_CHUNKS_INVALID")
    cache_receipt_sha256 = receipt.get("encoded_cache_receipt_sha256")
    cache_fingerprint = receipt.get("encoded_cache_fingerprint")
    if chunks:
        _require_sha256(
            cache_receipt_sha256,
            "encoded_cache_receipt_sha256",
        )
        if cache_fingerprint != authority.encoded_cache_fingerprint:
            raise ForwardContractError("RUN_RECEIPT_CACHE_FINGERPRINT_DRIFT")
    elif cache_receipt_sha256 is not None or cache_fingerprint is not None:
        raise ForwardContractError("RUN_RECEIPT_PREMATURE_CACHE_BINDING")
    expected_start = 0
    for chunk in chunks:
        if not isinstance(chunk, Mapping) or set(chunk) != {
            "start_index",
            "end_index",
            "artifact_name",
            "artifact_sha256",
            "semantic_sha256",
            "engine_registry_consumption_sha256",
            "query_types_sha256",
            "query_rows_sha256",
            "previous_run_receipt_sha256",
        }:
            raise ForwardContractError("RUN_CHUNK_RECEIPT_NOT_MAPPING")
        if chunk.get("start_index") != expected_start:
            raise ForwardContractError("RUN_CHUNK_SEQUENCE_GAP")
        end = chunk.get("end_index")
        if isinstance(end, bool) or not isinstance(end, int) or end <= expected_start:
            raise ForwardContractError("RUN_CHUNK_END_INVALID")
        if chunk.get("artifact_name") != "queries_%05d_%05d.pt" % (
            expected_start,
            end,
        ):
            raise ForwardContractError("RUN_CHUNK_ARTIFACT_NAME_INVALID")
        _require_sha256(chunk.get("artifact_sha256"), "artifact_sha256")
        _require_sha256(chunk.get("semantic_sha256"), "semantic_sha256")
        _require_sha256(
            chunk.get("engine_registry_consumption_sha256"),
            "engine_registry_consumption_sha256",
        )
        if (
            chunk.get("query_types_sha256")
            != authority.manifest.query_identity.query_types_sha256
            or chunk.get("query_rows_sha256")
            != authority.manifest.query_identity.query_rows_sha256
        ):
            raise ForwardContractError("RUN_CHUNK_QUERY_IDENTITY_DRIFT")
        _require_sha256(
            chunk.get("previous_run_receipt_sha256"),
            "previous_run_receipt_sha256",
        )
        expected_start = end
    if expected_start != cursor:
        raise ForwardContractError("RUN_RECEIPT_CURSOR_CHUNK_MISMATCH")
    if receipt.get("status") == "COMPLETED" and cursor != expected["query_count"]:
        raise ForwardContractError("COMPLETED_RUN_NOT_COMPLETE")
    if receipt.get("status") == "RUNNING" and cursor == expected["query_count"]:
        raise ForwardContractError("RUNNING_RUN_ALREADY_COMPLETE")
    observation_count = receipt.get("observation_event_count")
    if type(observation_count) is not int or observation_count < 0:
        raise ForwardContractError("RUN_RECEIPT_OBSERVATION_COUNT_INVALID")
    _require_sha256(
        receipt.get("observation_ledger_sha256"),
        "observation_ledger_sha256",
    )


def advance_run_receipt(
    authority: FrozenInputAuthority,
    capability: InputCapability,
    receipt: Mapping[str, Any],
    *,
    end_query_index: int,
    artifact_name: str,
    artifact_sha256: str,
    semantic_sha256: str,
    engine_registry_consumption_sha256: str,
    encoded_cache_receipt_sha256: Optional[str],
    encoded_cache_fingerprint: Optional[str],
) -> Mapping[str, Any]:
    validate_run_receipt(authority, capability, receipt)
    if receipt.get("status") != "RUNNING":
        raise ForwardContractError("COMPLETED_TOKEN_REPLAY")
    start = int(receipt["next_query_index"])
    end = _require_positive_int(end_query_index, "end_query_index")
    if end <= start or end > int(receipt["query_count"]):
        raise ForwardContractError("RUN_RECEIPT_ADVANCE_INVALID")
    _require_text(artifact_name, "artifact_name")
    if "/" in artifact_name or artifact_name in {".", ".."}:
        raise ForwardContractError("RESULT_ARTIFACT_NAME_INVALID")
    chunk = {
        "start_index": start,
        "end_index": end,
        "artifact_name": artifact_name,
        "artifact_sha256": _require_sha256(artifact_sha256, "artifact_sha256"),
        "semantic_sha256": _require_sha256(semantic_sha256, "semantic_sha256"),
        "engine_registry_consumption_sha256": _require_sha256(
            engine_registry_consumption_sha256,
            "engine_registry_consumption_sha256",
        ),
        "query_types_sha256": authority.manifest.query_identity.query_types_sha256,
        "query_rows_sha256": authority.manifest.query_identity.query_rows_sha256,
        "previous_run_receipt_sha256": _require_sha256(
            receipt["receipt_sha256"], "previous_run_receipt_sha256"
        ),
    }
    updated = dict(receipt)
    updated.pop("receipt_sha256", None)
    updated["next_query_index"] = end
    updated["result_chunks"] = list(receipt["result_chunks"]) + [chunk]
    if receipt["encoded_cache_receipt_sha256"] is None:
        updated["encoded_cache_receipt_sha256"] = _require_sha256(
            encoded_cache_receipt_sha256,
            "encoded_cache_receipt_sha256",
        )
        if encoded_cache_fingerprint != authority.encoded_cache_fingerprint:
            raise ForwardContractError("ENCODED_CACHE_FINGERPRINT_BINDING_INVALID")
        updated["encoded_cache_fingerprint"] = encoded_cache_fingerprint
    elif (
        encoded_cache_receipt_sha256
        not in {None, receipt["encoded_cache_receipt_sha256"]}
        or encoded_cache_fingerprint
        not in {None, receipt["encoded_cache_fingerprint"]}
    ):
        raise ForwardContractError("ENCODED_CACHE_RECEIPT_REBIND_FORBIDDEN")
    updated["observation_event_count"] = authority.observation_event_count
    updated["observation_ledger_sha256"] = authority.observation_ledger_sha256
    if end == int(receipt["query_count"]):
        updated["status"] = "COMPLETED"
    updated["receipt_sha256"] = _semantic_sha256(updated)
    validate_run_receipt(authority, capability, updated)
    return updated


def _validate_completed_forward_observation_rows(
    rows: Any,
    *,
    capability: InputCapability,
) -> None:
    if not isinstance(rows, list):
        raise ForwardContractError("COMPLETED_OBSERVATION_ROWS_NOT_LIST")
    exact_row_fields = {
        "schema_version",
        "run_id",
        "token_id",
        "eval_id",
        "seq",
        "event_type",
        "counters",
        "details",
        "previous_event_sha256",
        "event_sha256",
    }
    exact_counter_fields = {
        "content_open_count",
        "content_bytes_read",
        "lmdb_transaction_open_count",
        "model_forward_count",
    }
    previous_event_sha256 = None
    for sequence, row in enumerate(rows, start=1):
        counters = row.get("counters") if isinstance(row, Mapping) else None
        if (
            not isinstance(row, Mapping)
            or set(row) != exact_row_fields
            or row.get("schema_version")
            != "c28f_v5_a4_forward_observation_v1"
            or row.get("run_id") != capability.run_id
            or row.get("token_id") != capability.token_id
            or row.get("eval_id") != capability.eval_id
            or row.get("seq") != sequence
            or row.get("previous_event_sha256") != previous_event_sha256
            or not isinstance(row.get("event_type"), str)
            or not row.get("event_type")
            or not isinstance(counters, Mapping)
            or set(counters) != exact_counter_fields
            or any(type(item) is not int or item < 0 for item in counters.values())
            or not isinstance(row.get("details"), Mapping)
            or row.get("event_sha256")
            != _semantic_sha256(row, ("event_sha256",))
        ):
            raise ForwardContractError(
                "COMPLETED_OBSERVATION_ROW_INVALID",
                str(sequence),
            )
        previous_event_sha256 = row["event_sha256"]


def _validate_completed_forward_observation_artifacts(
    *,
    capability: InputCapability,
    run_root: FrozenDirectory,
    receipt: Mapping[str, Any],
) -> Mapping[str, Any]:
    event_count = receipt.get("observation_event_count")
    rows_sha256 = receipt.get("observation_ledger_sha256")
    if type(event_count) is not int or event_count < 0:
        raise ForwardContractError("COMPLETED_OBSERVATION_COUNT_INVALID")
    _require_sha256(rows_sha256, "observation_ledger_sha256")
    ledger_name = "forward_observations.json"
    snapshot_name = "observations_%08d_%s.json" % (event_count, rows_sha256)
    ledger = _read_canonical_json(
        Path(run_root.path) / ledger_name,
        max_bytes=128 * 1024 * 1024,
        expected_parent=run_root,
    )
    exact_ledger_fields = {
        "schema_version",
        "run_id",
        "capability_id",
        "event_count",
        "rows",
        "ledger_sha256",
    }
    ledger_rows = ledger.get("rows") if isinstance(ledger, Mapping) else None
    if (
        set(ledger) != exact_ledger_fields
        or ledger.get("schema_version")
        != "c28f_v5_a4_forward_observation_ledger_v1"
        or ledger.get("run_id") != capability.run_id
        or ledger.get("capability_id") != capability.capability_id
        or ledger.get("event_count") != event_count
        or not isinstance(ledger_rows, list)
        or len(ledger_rows) != event_count
        or _sha256_bytes(_canonical_json_bytes(ledger_rows)) != rows_sha256
        or ledger.get("ledger_sha256")
        != _semantic_sha256(ledger, ("ledger_sha256",))
    ):
        raise ForwardContractError("COMPLETED_OBSERVATION_LEDGER_INVALID")
    _validate_completed_forward_observation_rows(
        ledger_rows,
        capability=capability,
    )
    snapshot = _read_canonical_json(
        Path(run_root.path) / snapshot_name,
        max_bytes=128 * 1024 * 1024,
        expected_parent=run_root,
    )
    exact_snapshot_fields = {
        "schema_version",
        "run_id",
        "capability_id",
        "event_count",
        "rows_sha256",
        "rows",
        "snapshot_sha256",
    }
    if (
        set(snapshot) != exact_snapshot_fields
        or snapshot.get("schema_version")
        != "c28f_v5_a4_forward_observation_snapshot_v1"
        or snapshot.get("run_id") != capability.run_id
        or snapshot.get("capability_id") != capability.capability_id
        or snapshot.get("event_count") != event_count
        or snapshot.get("rows_sha256") != rows_sha256
        or snapshot.get("rows") != ledger_rows
        or snapshot.get("snapshot_sha256")
        != _semantic_sha256(snapshot, ("snapshot_sha256",))
    ):
        raise ForwardContractError("COMPLETED_OBSERVATION_SNAPSHOT_INVALID")
    return {
        "schema_version": "c28f_a4_completed_forward_observation_summary_v1",
        "ledger_artifact_name": ledger_name,
        "ledger_file_sha256": _sha256_bytes(_canonical_json_bytes(ledger)),
        "snapshot_artifact_name": snapshot_name,
        "snapshot_file_sha256": _sha256_bytes(_canonical_json_bytes(snapshot)),
        "snapshot_sha256": snapshot["snapshot_sha256"],
        "event_count": event_count,
        "rows_sha256": rows_sha256,
    }


def _validate_completed_forward_phase_receipt(
    *,
    capability: InputCapability,
    run_root: FrozenDirectory,
) -> Mapping[str, Any]:
    artifact_name = "control_completion_receipt.json"
    value = _read_canonical_json(
        Path(run_root.path) / artifact_name,
        expected_parent=run_root,
    )
    return _validate_completed_forward_phase_receipt_value(
        capability=capability,
        value=value,
        artifact_name=artifact_name,
    )


def _validate_completed_forward_phase_receipt_value(
    *,
    capability: InputCapability,
    value: Mapping[str, Any],
    artifact_name: str = "control_completion_receipt.json",
) -> Mapping[str, Any]:
    exact_fields = {
        "schema_version",
        "status",
        "phase",
        "goal_id",
        "authority_id",
        "eval_id",
        "token_id",
        "run_id",
        "cas_intent_sha256",
        "authorization_sha256",
        "execution_binding_sha256",
        "reservation_input_capability_sha256",
        "transaction_id",
        "state_sha256",
        "event_sha256",
        "previous_phase_receipt_sha256",
        "model_forward_evaluations",
        "receipt_sha256",
    }
    module = _require_exact_forward_control(capability._control)
    if type(capability) is InputCapability:
        running = module.validate_eval_running_capability(
            capability._running_control_capability
        )
        execution_binding_sha256 = running.get(
            "execution_binding_sha256"
        )
        completed_transaction_id = None
        completed_state_sha256 = None
        completed_event_sha256 = None
    elif type(capability) is CompletedForwardResultCapability:
        completed = capability._assert_completed_lifecycle()
        execution_binding_sha256 = completed.get(
            "execution_binding_sha256"
        )
        completed_transaction_id = completed.get(
            "completed_transaction_id"
        )
        completed_state_sha256 = completed.get("completed_state_sha256")
        completed_event_sha256 = completed.get("completed_event_sha256")
    else:
        raise ForwardContractError(
            "EXACT_COMPLETED_PHASE_CAPABILITY_REQUIRED"
        )
    if (
        set(value) != exact_fields
        or value.get("schema_version") != "c28f_eval_token_phase_receipt_v1"
        or value.get("status") != "FSYNCED_CAS_COMMITTED"
        or value.get("phase") != "COMPLETED"
        or value.get("goal_id") != capability.goal_id
        or value.get("authority_id") != capability.authority_id
        or value.get("eval_id") != capability.eval_id
        or value.get("token_id") != capability.token_id
        or value.get("run_id") != capability.run_id
        or value.get("execution_binding_sha256")
        != execution_binding_sha256
        or (
            completed_transaction_id is not None
            and value.get("transaction_id") != completed_transaction_id
        )
        or (
            completed_state_sha256 is not None
            and value.get("state_sha256") != completed_state_sha256
        )
        or (
            completed_event_sha256 is not None
            and value.get("event_sha256") != completed_event_sha256
        )
        or value.get("reservation_input_capability_sha256")
        != capability.reservation_input_capability_sha256
        or value.get("model_forward_evaluations") != 1
        or value.get("receipt_sha256")
        != _semantic_sha256(value, ("receipt_sha256",))
    ):
        raise ForwardContractError("COMPLETED_PHASE_RECEIPT_INVALID")
    for field_name in (
        "cas_intent_sha256",
        "authorization_sha256",
        "execution_binding_sha256",
        "reservation_input_capability_sha256",
        "state_sha256",
        "event_sha256",
        "previous_phase_receipt_sha256",
        "receipt_sha256",
    ):
        _require_sha256(value.get(field_name), field_name)
    _require_text(value.get("transaction_id"), "transaction_id")
    return {
        "artifact_name": artifact_name,
        "receipt_sha256": value["receipt_sha256"],
        "file_sha256": _sha256_bytes(_canonical_json_bytes(value)),
        "transaction_id_sha256": _sha256_bytes(
            value["transaction_id"].encode("utf-8")
        ),
    }


def _validate_bound_completed_forward_roots(
    *,
    authority: Any,
    capability: Any,
    run_root: FrozenDirectory,
    chunk_root: FrozenDirectory,
) -> None:
    if (
        type(run_root) is not FrozenDirectory
        or type(chunk_root) is not FrozenDirectory
        or run_root is not capability._run_receipt_parent
        or chunk_root is not capability._run_chunk_parent
    ):
        raise ForwardContractError("COMPLETED_ARTIFACT_ROOT_HANDLE_MISMATCH")
    capability._assert_controller_derived_run_artifact_roots(
        run_root,
        chunk_root,
    )
    expected_receipt_path = os.path.join(
        run_root.path,
        "forward_run_receipt.json",
    )
    if capability._run_receipt_path != expected_receipt_path:
        raise ForwardContractError("COMPLETED_RUN_RECEIPT_PATH_MISMATCH")
    run_fd = _open_directory_descriptor(run_root.path, run_root)
    chunk_fd = -1
    try:
        chunk_fd = os.open(
            "query_chunks",
            os.O_RDONLY
            | os.O_DIRECTORY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=run_fd,
        )
        observed = os.fstat(chunk_fd)
        if (
            not stat.S_ISDIR(observed.st_mode)
            or (
                int(observed.st_dev),
                int(observed.st_ino),
                _fd_mount_id(chunk_fd),
            )
            != (chunk_root.device, chunk_root.inode, chunk_root.mount_id)
        ):
            raise ForwardContractError("COMPLETED_CHUNK_ROOT_PARENT_DRIFT")
    except OSError as error:
        raise ForwardContractError(
            "COMPLETED_CHUNK_ROOT_PARENT_DRIFT",
            type(error).__name__,
        ) from None
    finally:
        if chunk_fd >= 0:
            os.close(chunk_fd)
        os.close(run_fd)


_COMPLETED_RESULT_AUTHORITY_FACTORY = object()
_COMPLETED_RESULT_CAPABILITY_FACTORY = object()
_COMPLETED_RESULT_AUTHORITY_LOCK = threading.RLock()
_COMPLETED_RESULT_AUTHORITY_REGISTRY: dict[int, dict[str, Any]] = {}


class CompletedForwardResultVerificationAuthority:
    """Completed-only result verifier with no feature or model I/O surface."""

    __slots__ = (
        "_factory",
        "manifest",
        "output_root",
        "authority_binding_sha256",
        "forward_authority_handle_sha256",
        "encoded_cache_fingerprint",
        "projection_binding_sha256",
        "_issued_capability",
    )

    def __init__(self, factory: object, **values: Any) -> None:
        if factory is not _COMPLETED_RESULT_AUTHORITY_FACTORY:
            raise ForwardContractError(
                "COMPLETED_RESULT_AUTHORITY_CONSTRUCTION_FORBIDDEN"
            )
        for field_name in self.__slots__:
            object.__setattr__(self, field_name, values[field_name])

    def __setattr__(self, _name: str, _value: Any) -> None:
        raise ForwardContractError(
            "COMPLETED_RESULT_AUTHORITY_MUTATION_FORBIDDEN"
        )

    def assert_capability(self, _capability: Any) -> None:
        raise ForwardContractError(
            "COMPLETED_RESULT_AUTHORITY_ACTIVE_FORWARD_FORBIDDEN"
        )

    def issue_capability(self, _control: Any) -> None:
        raise ForwardContractError(
            "COMPLETED_RESULT_AUTHORITY_TOKEN_RESERVATION_FORBIDDEN"
        )

    def allowed_key_digests(self, _source_id: str) -> None:
        raise ForwardContractError(
            "COMPLETED_RESULT_AUTHORITY_FEATURE_SOURCE_ACCESS_FORBIDDEN"
        )

    def verify_code_sources(self) -> None:
        for source_name, digest in self.manifest.code_source_sha256:
            parent, frozen = _capture_approved_code_source(
                source_name,
                digest,
            )
            fd = frozen.open_verified_readonly(
                verify_content=True,
                expected_parent=parent,
            )
            os.close(fd)

    def assert_completed_capability(self, capability: Any) -> None:
        if (
            type(capability) is not CompletedForwardResultCapability
            or capability._factory is not _COMPLETED_RESULT_CAPABILITY_FACTORY
            or capability._authority is not self
            or self._issued_capability is not capability
        ):
            raise ForwardContractError(
                "EXACT_COMPLETED_RESULT_CAPABILITY_REQUIRED"
            )
        with _COMPLETED_RESULT_AUTHORITY_LOCK:
            record = _COMPLETED_RESULT_AUTHORITY_REGISTRY.get(id(self))
            if (
                not isinstance(record, Mapping)
                or record.get("object") is not self
                or record.get("capability") is not capability
                or record.get("projection_binding_sha256")
                != self.projection_binding_sha256
            ):
                raise ForwardContractError(
                    "COMPLETED_RESULT_AUTHORITY_NOT_REGISTERED"
                )
        capability._assert_completed_lifecycle()


class CompletedForwardResultCapability:
    """Opaque capability that can validate/derive completed results only."""

    __slots__ = (
        "_factory",
        "capability_id",
        "goal_id",
        "authority_id",
        "run_id",
        "token_id",
        "eval_id",
        "purpose",
        "stage",
        "forward_manifest_sha256",
        "input_identity_sha256",
        "authority_binding_sha256",
        "token_snapshot_sha256",
        "gpu0_identity_sha256",
        "control_chain_sha256",
        "running_commit_sha256",
        "running_capability_sha256",
        "reservation_input_capability_sha256",
        "controller_authority_binding_sha256",
        "_authority",
        "_control",
        "_proof_capability",
        "_proof_mode",
        "_input_capability_material_sha256",
        "_run_receipt_sha256",
        "_run_receipt_path",
        "_run_receipt_parent",
        "_run_chunk_parent",
    )

    def __init__(self, factory: object, **values: Any) -> None:
        if factory is not _COMPLETED_RESULT_CAPABILITY_FACTORY:
            raise ForwardContractError(
                "COMPLETED_RESULT_CAPABILITY_CONSTRUCTION_FORBIDDEN"
            )
        for field_name in self.__slots__:
            object.__setattr__(self, field_name, values[field_name])

    def __setattr__(self, _name: str, _value: Any) -> None:
        raise ForwardContractError(
            "COMPLETED_RESULT_CAPABILITY_MUTATION_FORBIDDEN"
        )

    def assert_running_committed(self) -> None:
        raise ForwardContractError(
            "COMPLETED_RESULT_CAPABILITY_ACTIVE_FORWARD_FORBIDDEN"
        )

    def assert_io_ready(self) -> None:
        raise ForwardContractError(
            "COMPLETED_RESULT_CAPABILITY_FEATURE_IO_FORBIDDEN"
        )

    def _assert_controller_derived_run_artifact_roots(
        self,
        parent: FrozenDirectory,
        chunk_parent: FrozenDirectory,
    ) -> None:
        expected_run_path = os.path.join(
            self._authority.output_root.path,
            self.stage,
            self._authority.manifest.attempt_id,
            self.run_id,
        )
        expected_chunk_path = os.path.join(expected_run_path, "query_chunks")
        if (
            type(parent) is not FrozenDirectory
            or type(chunk_parent) is not FrozenDirectory
            or parent.path != expected_run_path
            or parent.root_id
            != "child-" + _sha256_bytes(expected_run_path.encode("utf-8"))[:32]
            or chunk_parent.path != expected_chunk_path
            or chunk_parent.root_id
            != "child-"
            + _sha256_bytes(expected_chunk_path.encode("utf-8"))[:32]
            or (parent.device, parent.mount_id)
            != (
                self._authority.output_root.device,
                self._authority.output_root.mount_id,
            )
            or (chunk_parent.device, chunk_parent.mount_id)
            != (parent.device, parent.mount_id)
        ):
            raise ForwardContractError(
                "COMPLETED_RESULT_ROOT_BINDING_INVALID"
            )
        self._authority.output_root.verify()
        parent.verify()
        chunk_parent.verify()

    def _assert_completed_lifecycle(self) -> Mapping[str, Any]:
        module = _require_exact_forward_control(self._control)
        if self._proof_mode == "ORIGINAL_COMPLETED_LIFECYCLE":
            lifecycle = module.validate_completed_forward_input_capability(
                running_capability=self._proof_capability,
                run_id=self.run_id,
                token_id=self.token_id,
                eval_id=self.eval_id,
                capability_id=self.capability_id,
                authority_binding_sha256=self.authority_binding_sha256,
                forward_manifest_sha256=self.forward_manifest_sha256,
                input_identity_sha256=self.input_identity_sha256,
                reservation_input_capability_sha256=(
                    self.reservation_input_capability_sha256
                ),
                controller_authority_binding_sha256=(
                    self.controller_authority_binding_sha256
                ),
                forward_run_receipt_sha256=self._run_receipt_sha256,
            )
            return _validate_original_completed_lifecycle_binding(
                self,
                lifecycle,
            )
        elif self._proof_mode == "REHYDRATED_COMPLETED_SAFE_INDEX":
            binding = _validate_completed_forward_rehydrate_binding(
                self._control,
                self._proof_capability,
            )
            expected = {
                "goal_id": self.goal_id,
                "attempt_id": self._authority.manifest.attempt_id,
                "authority_id": self.authority_id,
                "run_id": self.run_id,
                "token_id": self.token_id,
                "eval_id": self.eval_id,
                "stage": self.stage,
                "purpose": self.purpose,
                "capability_id": self.capability_id,
                "forward_authority_handle_sha256": (
                    self._authority.forward_authority_handle_sha256
                ),
                "forward_manifest_sha256": self.forward_manifest_sha256,
                "input_identity_sha256": self.input_identity_sha256,
                "authority_binding_sha256": self.authority_binding_sha256,
                "reservation_input_capability_sha256": (
                    self.reservation_input_capability_sha256
                ),
                "controller_authority_binding_sha256": (
                    self.controller_authority_binding_sha256
                ),
                "input_capability_material_sha256": (
                    self._input_capability_material_sha256
                ),
                "forward_run_receipt_sha256": self._run_receipt_sha256,
            }
            if any(
                binding.get(field_name) != expected_value
                for field_name, expected_value in expected.items()
            ):
                raise ForwardContractError(
                    "REHYDRATED_COMPLETED_RESULT_LIFECYCLE_DRIFT"
                )
            return binding
        else:
            raise ForwardContractError(
                "COMPLETED_RESULT_CAPABILITY_PROOF_MODE_INVALID"
            )


def _build_completed_forward_authority_rehydrate_projection(
    authority: FrozenInputAuthority,
    *,
    forward_authority_handle_sha256: str,
) -> Mapping[str, Any]:
    """Build the controller-private, no-index reconstruction projection."""

    if type(authority) is not FrozenInputAuthority:
        raise ForwardContractError(
            "EXACT_FORWARD_AUTHORITY_REHYDRATE_PROJECTION_REQUIRED"
        )
    manifest = authority.manifest
    _require_sha256(
        forward_authority_handle_sha256,
        "forward_authority_handle_sha256",
    )
    base = {
        "schema_version": (
            "c28f_a4_completed_forward_authority_rehydrate_projection_v1"
        ),
        "status": "COMPLETED_RESULT_VERIFICATION_ONLY_PROJECTION",
        "manifest": manifest.as_dict(),
        "query_keys": list(manifest.query_identity.query_keys),
        "corpus_video_ids": list(manifest.corpus_identity.video_ids),
        "corpus_durations_sec": list(
            manifest.corpus_identity.durations_sec
        ),
        "output_root": authority.output_root.as_dict(),
        "authority_binding_sha256": authority.authority_binding_sha256,
        "forward_authority_handle_sha256": (
            forward_authority_handle_sha256
        ),
        "forward_manifest_sha256": manifest.manifest_sha256,
        "input_identity_sha256": manifest.input_identity_sha256,
        "encoded_cache_fingerprint": authority.encoded_cache_fingerprint,
        "active_authority_type_reconstructed": False,
        "checkpoint_descriptor_present": False,
        "lmdb_descriptor_present": False,
        "cache_root_descriptor_present": False,
        "active_io_allowed": False,
    }
    projection = {
        **base,
        "projection_binding_sha256": _semantic_sha256(base),
    }
    return MappingProxyType(
        json.loads(
            _canonical_json_bytes(projection).decode("utf-8", "strict")
        )
    )


def build_active_forward_authority_rehydrate_projection(
    authority: FrozenInputAuthority,
    *,
    forward_authority_handle_sha256: str,
) -> Mapping[str, Any]:
    """Freeze enough descriptor-only material for same-token I/O recovery.

    This projection contains the already-authorized query/corpus identities and
    immutable file/directory descriptors.  It contains no feature value,
    checkpoint tensor, model object, score, proposal, query text, or ground
    truth.  Rehydration therefore restores only the authority object; every
    content open remains mediated by the original RUNNING capability and its
    durable ledgers.
    """

    if type(authority) is not FrozenInputAuthority:
        raise ForwardContractError(
            "EXACT_ACTIVE_FORWARD_AUTHORITY_REHYDRATE_PROJECTION_REQUIRED"
        )
    _require_sha256(
        forward_authority_handle_sha256,
        "forward_authority_handle_sha256",
    )
    manifest = authority.manifest
    base = {
        "schema_version": (
            "c28f_a4_active_forward_authority_rehydrate_projection_v1"
        ),
        "status": "ACTIVE_AUTHORITY_DESCRIPTOR_ONLY_RECOVERY_PROJECTION",
        "manifest": manifest.as_dict(),
        "query_keys": list(manifest.query_identity.query_keys),
        "corpus_video_ids": list(manifest.corpus_identity.video_ids),
        "corpus_durations_sec": list(
            manifest.corpus_identity.durations_sec
        ),
        "authority_binding_material": authority.authority_binding_material(),
        "authority_binding_sha256": authority.authority_binding_sha256,
        "forward_authority_handle_sha256": (
            forward_authority_handle_sha256
        ),
        "forward_manifest_sha256": manifest.manifest_sha256,
        "input_identity_sha256": manifest.input_identity_sha256,
        "encoded_cache_fingerprint": authority.encoded_cache_fingerprint,
        "descriptor_only": True,
        "source_content_present": False,
        "active_content_opened": False,
        "eval_token_reissued": False,
    }
    projection = {
        **base,
        "projection_binding_sha256": _semantic_sha256(base),
    }
    return MappingProxyType(
        json.loads(
            _canonical_json_bytes(projection).decode("utf-8", "strict")
        )
    )


def rehydrate_active_forward_authority(
    projection: Mapping[str, Any],
) -> FrozenInputAuthority:
    """Rebuild one active authority from exact frozen descriptors only."""

    exact_fields = {
        "schema_version",
        "status",
        "manifest",
        "query_keys",
        "corpus_video_ids",
        "corpus_durations_sec",
        "authority_binding_material",
        "authority_binding_sha256",
        "forward_authority_handle_sha256",
        "forward_manifest_sha256",
        "input_identity_sha256",
        "encoded_cache_fingerprint",
        "descriptor_only",
        "source_content_present",
        "active_content_opened",
        "eval_token_reissued",
        "projection_binding_sha256",
    }
    if (
        not isinstance(projection, Mapping)
        or set(projection) != exact_fields
        or projection.get("schema_version")
        != "c28f_a4_active_forward_authority_rehydrate_projection_v1"
        or projection.get("status")
        != "ACTIVE_AUTHORITY_DESCRIPTOR_ONLY_RECOVERY_PROJECTION"
        or projection.get("descriptor_only") is not True
        or any(
            projection.get(field_name) is not False
            for field_name in (
                "source_content_present",
                "active_content_opened",
                "eval_token_reissued",
            )
        )
        or projection.get("projection_binding_sha256")
        != _semantic_sha256(
            projection,
            ("projection_binding_sha256",),
        )
    ):
        raise ForwardContractError(
            "ACTIVE_FORWARD_AUTHORITY_REHYDRATE_PROJECTION_INVALID"
        )
    for field_name in (
        "authority_binding_sha256",
        "forward_authority_handle_sha256",
        "forward_manifest_sha256",
        "input_identity_sha256",
        "encoded_cache_fingerprint",
        "projection_binding_sha256",
    ):
        _require_sha256(projection.get(field_name), field_name)
    manifest_view = projection.get("manifest")
    material = projection.get("authority_binding_material")
    if not isinstance(manifest_view, Mapping) or not isinstance(
        material,
        Mapping,
    ):
        raise ForwardContractError(
            "ACTIVE_FORWARD_AUTHORITY_REHYDRATE_MATERIAL_INVALID"
        )
    completed_base = {
        "schema_version": (
            "c28f_a4_completed_forward_authority_rehydrate_projection_v1"
        ),
        "status": "COMPLETED_RESULT_VERIFICATION_ONLY_PROJECTION",
        "manifest": dict(manifest_view),
        "query_keys": list(projection["query_keys"]),
        "corpus_video_ids": list(projection["corpus_video_ids"]),
        "corpus_durations_sec": list(
            projection["corpus_durations_sec"]
        ),
        "output_root": material.get("output_root"),
        "authority_binding_sha256": projection[
            "authority_binding_sha256"
        ],
        "forward_authority_handle_sha256": projection[
            "forward_authority_handle_sha256"
        ],
        "forward_manifest_sha256": projection[
            "forward_manifest_sha256"
        ],
        "input_identity_sha256": projection["input_identity_sha256"],
        "encoded_cache_fingerprint": projection[
            "encoded_cache_fingerprint"
        ],
        "active_authority_type_reconstructed": False,
        "checkpoint_descriptor_present": False,
        "lmdb_descriptor_present": False,
        "cache_root_descriptor_present": False,
        "active_io_allowed": False,
    }
    completed_projection = {
        **completed_base,
        "projection_binding_sha256": _semantic_sha256(completed_base),
    }
    verification_authority = (
        _rehydrate_completed_forward_authority_from_projection(
            completed_projection
        )
    )
    checkpoint_view = material.get("checkpoint")
    output_root_view = material.get("output_root")
    cache_root_view = material.get("cache_root")
    source_views = material.get("lmdb_sources")
    forbidden_roots = material.get("forbidden_roots")
    if (
        not isinstance(checkpoint_view, Mapping)
        or not isinstance(output_root_view, Mapping)
        or not isinstance(cache_root_view, Mapping)
        or not isinstance(source_views, list)
        or not isinstance(forbidden_roots, list)
        or not all(isinstance(item, Mapping) for item in source_views)
        or not all(isinstance(item, str) for item in forbidden_roots)
    ):
        raise ForwardContractError(
            "ACTIVE_FORWARD_AUTHORITY_DESCRIPTOR_MATERIAL_INVALID"
        )
    sources = []
    for source_view in source_views:
        directory_view = source_view.get("directory")
        data_file_view = source_view.get("data_file")
        if not isinstance(directory_view, Mapping) or not isinstance(
            data_file_view,
            Mapping,
        ):
            raise ForwardContractError(
                "ACTIVE_FORWARD_AUTHORITY_LMDB_DESCRIPTOR_INVALID"
            )
        sources.append(
            FrozenLmdbSource(
                source_id=source_view.get("source_id"),
                namespace=source_view.get("namespace"),
                directory=FrozenDirectory(**dict(directory_view)),
                data_file=FrozenFile(**dict(data_file_view)),
            )
        )
    authority = FrozenInputAuthority(
        manifest=verification_authority.manifest,
        checkpoint=FrozenFile(**dict(checkpoint_view)),
        lmdb_sources=tuple(sources),
        output_root=FrozenDirectory(**dict(output_root_view)),
        cache_root=FrozenDirectory(**dict(cache_root_view)),
        forbidden_roots=tuple(forbidden_roots),
    )
    if (
        authority.authority_binding_material() != dict(material)
        or authority.authority_binding_sha256
        != projection["authority_binding_sha256"]
        or authority.manifest.manifest_sha256
        != projection["forward_manifest_sha256"]
        or authority.manifest.input_identity_sha256
        != projection["input_identity_sha256"]
        or authority.encoded_cache_fingerprint
        != projection["encoded_cache_fingerprint"]
    ):
        raise ForwardContractError(
            "ACTIVE_FORWARD_AUTHORITY_REHYDRATE_BINDING_DRIFT"
        )
    return authority


def _rehydrate_completed_forward_authority_from_projection(
    projection: Mapping[str, Any],
) -> CompletedForwardResultVerificationAuthority:
    """Rebuild the completed-only verifier without active source authority."""

    exact_projection_fields = {
        "schema_version",
        "status",
        "manifest",
        "query_keys",
        "corpus_video_ids",
        "corpus_durations_sec",
        "output_root",
        "authority_binding_sha256",
        "forward_authority_handle_sha256",
        "forward_manifest_sha256",
        "input_identity_sha256",
        "encoded_cache_fingerprint",
        "active_authority_type_reconstructed",
        "checkpoint_descriptor_present",
        "lmdb_descriptor_present",
        "cache_root_descriptor_present",
        "active_io_allowed",
        "projection_binding_sha256",
    }
    if (
        not isinstance(projection, Mapping)
        or set(projection) != exact_projection_fields
        or projection.get("schema_version")
        != "c28f_a4_completed_forward_authority_rehydrate_projection_v1"
        or projection.get("status")
        != "COMPLETED_RESULT_VERIFICATION_ONLY_PROJECTION"
        or any(
            projection.get(field_name) is not False
            for field_name in (
                "active_authority_type_reconstructed",
                "checkpoint_descriptor_present",
                "lmdb_descriptor_present",
                "cache_root_descriptor_present",
                "active_io_allowed",
            )
        )
        or projection.get("projection_binding_sha256")
        != _semantic_sha256(
            projection,
            ("projection_binding_sha256",),
        )
    ):
        raise ForwardContractError(
            "COMPLETED_FORWARD_AUTHORITY_REHYDRATE_PROJECTION_INVALID"
        )
    for field_name in (
        "authority_binding_sha256",
        "forward_authority_handle_sha256",
        "forward_manifest_sha256",
        "input_identity_sha256",
        "encoded_cache_fingerprint",
        "projection_binding_sha256",
    ):
        _require_sha256(projection.get(field_name), field_name)
    manifest_view = projection.get("manifest")
    query_view = (
        manifest_view.get("query_identity")
        if isinstance(manifest_view, Mapping)
        else None
    )
    corpus_view = (
        manifest_view.get("corpus_identity")
        if isinstance(manifest_view, Mapping)
        else None
    )
    if (
        not isinstance(manifest_view, Mapping)
        or not isinstance(query_view, Mapping)
        or not isinstance(corpus_view, Mapping)
        or not isinstance(projection.get("query_keys"), list)
        or not isinstance(projection.get("corpus_video_ids"), list)
        or not isinstance(projection.get("corpus_durations_sec"), list)
    ):
        raise ForwardContractError(
            "COMPLETED_FORWARD_AUTHORITY_REHYDRATE_MANIFEST_INVALID"
        )
    query_identity = FrozenQueryIdentity(
        role=query_view.get("role"),
        query_keys=tuple(projection["query_keys"]),
        query_types=tuple(query_view.get("query_types", ())),
        query_keys_sha256=query_view.get("query_keys_sha256"),
        query_types_sha256=query_view.get("query_types_sha256"),
        query_rows_sha256=query_view.get("query_rows_sha256"),
        split_manifest_sha256=query_view.get("split_manifest_sha256"),
    )
    corpus_identity = FrozenCorpusIdentity(
        video_ids=tuple(projection["corpus_video_ids"]),
        durations_sec=tuple(projection["corpus_durations_sec"]),
        video_ids_sha256=corpus_view.get("video_ids_sha256"),
        durations_sha256=corpus_view.get("durations_sha256"),
        feature_manifest_sha256=corpus_view.get(
            "feature_manifest_sha256"
        ),
    )
    model_view = manifest_view.get("model_contract")
    temporal_view = manifest_view.get("temporal_manifest")
    feature_view = manifest_view.get("feature_contract")
    gpu_view = manifest_view.get("gpu0_identity")
    if not all(
        isinstance(item, Mapping)
        for item in (model_view, temporal_view, feature_view, gpu_view)
    ):
        raise ForwardContractError(
            "COMPLETED_FORWARD_AUTHORITY_REHYDRATE_CONTRACT_INVALID"
        )
    model_contract = FrozenModelContract(
        **{
            item.name: model_view.get(item.name)
            for item in fields(FrozenModelContract)
        }
    )
    temporal_values = dict(temporal_view)
    temporal_values["proposal_widths"] = tuple(
        temporal_values.get("proposal_widths", ())
    )
    temporal_manifest = FrozenTemporalManifest(**temporal_values)
    feature_contract = FrozenFeatureContract(**dict(feature_view))
    gpu0_identity = FrozenGpu0Identity(**dict(gpu_view))
    manifest_values = {
        item.name: manifest_view.get(item.name)
        for item in fields(FrozenForwardManifest)
    }
    manifest_values.update(
        {
            "query_identity": query_identity,
            "corpus_identity": corpus_identity,
            "model_contract": model_contract,
            "temporal_manifest": temporal_manifest,
            "feature_contract": feature_contract,
            "gpu0_identity": gpu0_identity,
            "code_source_sha256": tuple(
                tuple(item)
                for item in manifest_view.get("code_source_sha256", ())
            ),
        }
    )
    manifest = FrozenForwardManifest(**manifest_values)
    if (
        manifest.as_dict() != dict(manifest_view)
        or manifest.manifest_sha256
        != projection["forward_manifest_sha256"]
        or manifest.input_identity_sha256
        != projection["input_identity_sha256"]
    ):
        raise ForwardContractError(
            "COMPLETED_FORWARD_AUTHORITY_REHYDRATE_MANIFEST_DRIFT"
        )
    output_root_view = projection.get("output_root")
    if not isinstance(output_root_view, Mapping):
        raise ForwardContractError(
            "COMPLETED_FORWARD_OUTPUT_ROOT_PROJECTION_INVALID"
        )
    authority = CompletedForwardResultVerificationAuthority(
        _COMPLETED_RESULT_AUTHORITY_FACTORY,
        _factory=_COMPLETED_RESULT_AUTHORITY_FACTORY,
        manifest=manifest,
        output_root=FrozenDirectory(**dict(output_root_view)),
        authority_binding_sha256=projection[
            "authority_binding_sha256"
        ],
        forward_authority_handle_sha256=projection[
            "forward_authority_handle_sha256"
        ],
        encoded_cache_fingerprint=projection[
            "encoded_cache_fingerprint"
        ],
        projection_binding_sha256=projection[
            "projection_binding_sha256"
        ],
        _issued_capability=None,
    )
    if authority.output_root.root_id != manifest.output_root_id:
        raise ForwardContractError(
            "COMPLETED_RESULT_OUTPUT_ROOT_MANIFEST_DRIFT"
        )
    authority.output_root.verify()
    authority.verify_code_sources()
    return authority


def _validate_completed_forward_rehydrate_binding(
    control: Any,
    rehydrate_capability: Any,
) -> Mapping[str, Any]:
    module = _require_exact_forward_control(control)
    try:
        payload = module.validate_completed_forward_rehydrate_capability(
            rehydrate_capability
        )
    except BaseException as error:
        raise ForwardContractError(
            "EXACT_COMPLETED_FORWARD_REHYDRATE_CAPABILITY_REQUIRED",
            type(error).__name__,
        ) from None
    exact_fields = {
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
    if (
        not isinstance(payload, Mapping)
        or set(payload) != exact_fields
        or payload.get("schema_version")
        != "c28f_a4_completed_forward_rehydrate_capability_v1"
        or payload.get("status")
        != "FSYNCED_CONTROLLER_COMPLETED_RESULT_VERIFIER_REHYDRATE"
        or payload.get("stage") not in _STAGE_PURPOSES
        or payload.get("purpose")
        != _STAGE_PURPOSES.get(payload.get("stage"))
        or type(payload.get("rehydration_generation")) is not int
        or payload["rehydration_generation"] < 0
        or payload.get("rehydration_source")
        != (
            "FSYNCED_CONTROLLER_PRIVATE_COMPLETED_RESULT_VERIFICATION_"
            "PROJECTION_PLUS_"
            "COMPLETION_AND_SAFE_INDEX"
        )
        or any(
            type(payload.get(field_name)) is not int
            or payload[field_name] != 0
            for field_name in (
                "token_reissue_count",
                "model_forward_reexecution_count",
                "query_source_index_reaccess_count",
                "raw_chunk_index_namespace_rescan_count",
            )
        )
        or payload.get("raw_tensor_materialized") is not False
        or payload.get("old_object_identity_required") is not False
        or payload.get("rehydrate_binding_sha256")
        != _semantic_sha256(
            payload,
            ("rehydrate_binding_sha256",),
        )
    ):
        raise ForwardContractError(
            "COMPLETED_FORWARD_REHYDRATE_BINDING_INVALID"
        )
    for field_name in (
        "capability_id",
        "forward_authority_handle_sha256",
        "forward_manifest_sha256",
        "input_identity_sha256",
        "authority_binding_sha256",
        "reservation_input_capability_sha256",
        "controller_authority_binding_sha256",
        "completed_state_sha256",
        "completed_event_sha256",
        "execution_binding_sha256",
        "forward_run_receipt_sha256",
        "completed_forward_validation_receipt_sha256",
        "authority_rehydrate_projection_sha256",
        "input_capability_material_sha256",
        "run_root_identity_sha256",
        "chunk_root_identity_sha256",
        "rehydrate_binding_sha256",
    ):
        _require_sha256(payload.get(field_name), field_name)
    for field_name in (
        "goal_id",
        "attempt_id",
        "authority_id",
        "run_id",
        "token_id",
        "eval_id",
        "completed_transaction_id",
    ):
        _require_text(payload.get(field_name), field_name)
    return MappingProxyType(
        json.loads(
            _canonical_json_bytes(dict(payload)).decode("utf-8", "strict")
        )
    )


def _validate_original_completed_lifecycle_binding(
    capability: CompletedForwardResultCapability,
    value: Any,
) -> Mapping[str, Any]:
    exact_fields = {
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
        "forward_manifest_sha256",
        "input_identity_sha256",
        "authority_binding_sha256",
        "reservation_input_capability_sha256",
        "controller_authority_binding_sha256",
        "running_capability_sha256",
        "execution_binding_sha256",
        "completed_transaction_id",
        "completed_state_sha256",
        "completed_event_sha256",
        "forward_run_receipt_sha256",
        "token_reissue_count",
        "model_forward_reexecution_count",
        "binding_sha256",
    }
    expected = {
        "schema_version": (
            "c28f_a4_completed_forward_input_capability_binding_v1"
        ),
        "status": "VALIDATED_ORIGINAL_INPUT_CAPABILITY_COMPLETED_ONLY",
        "goal_id": capability.goal_id,
        "attempt_id": capability._authority.manifest.attempt_id,
        "authority_id": capability.authority_id,
        "run_id": capability.run_id,
        "token_id": capability.token_id,
        "eval_id": capability.eval_id,
        "stage": capability.stage,
        "purpose": capability.purpose,
        "capability_id": capability.capability_id,
        "forward_manifest_sha256": capability.forward_manifest_sha256,
        "input_identity_sha256": capability.input_identity_sha256,
        "authority_binding_sha256": capability.authority_binding_sha256,
        "reservation_input_capability_sha256": (
            capability.reservation_input_capability_sha256
        ),
        "controller_authority_binding_sha256": (
            capability.controller_authority_binding_sha256
        ),
        "running_capability_sha256": capability.running_capability_sha256,
        "forward_run_receipt_sha256": capability._run_receipt_sha256,
        "token_reissue_count": 0,
        "model_forward_reexecution_count": 0,
    }
    if (
        not isinstance(value, Mapping)
        or set(value) != exact_fields
        or any(
            value.get(field_name) != expected_value
            for field_name, expected_value in expected.items()
        )
        or value.get("binding_sha256")
        != _semantic_sha256(value, ("binding_sha256",))
    ):
        raise ForwardContractError(
            "ORIGINAL_COMPLETED_INPUT_LIFECYCLE_BINDING_INVALID"
        )
    for field_name in (
        "execution_binding_sha256",
        "completed_state_sha256",
        "completed_event_sha256",
        "binding_sha256",
    ):
        _require_sha256(value.get(field_name), field_name)
    _require_text(
        value.get("completed_transaction_id"),
        "completed_transaction_id",
    )
    return MappingProxyType(
        json.loads(
            _canonical_json_bytes(dict(value)).decode("utf-8", "strict")
        )
    )


def _validate_rehydrated_input_capability_material(
    material: Any,
    *,
    binding: Mapping[str, Any],
    authority: CompletedForwardResultVerificationAuthority,
) -> Mapping[str, Any]:
    exact_fields = {
        "capability_id",
        "goal_id",
        "authority_id",
        "run_id",
        "token_id",
        "eval_id",
        "purpose",
        "stage",
        "forward_manifest_sha256",
        "input_identity_sha256",
        "authority_binding_sha256",
        "token_snapshot_sha256",
        "gpu0_identity_sha256",
        "control_chain_sha256",
        "running_commit_sha256",
        "running_capability_sha256",
        "reservation_input_capability_sha256",
        "controller_authority_binding_sha256",
        "consume_receipt_sha256",
    }
    if not isinstance(material, Mapping) or set(material) != exact_fields:
        raise ForwardContractError(
            "REHYDRATED_INPUT_CAPABILITY_MATERIAL_FIELDS_INVALID"
        )
    expected = {
        "capability_id": binding["capability_id"],
        "goal_id": binding["goal_id"],
        "authority_id": binding["authority_id"],
        "run_id": binding["run_id"],
        "token_id": binding["token_id"],
        "eval_id": binding["eval_id"],
        "purpose": binding["purpose"],
        "stage": binding["stage"],
        "forward_manifest_sha256": binding[
            "forward_manifest_sha256"
        ],
        "input_identity_sha256": binding["input_identity_sha256"],
        "authority_binding_sha256": binding[
            "authority_binding_sha256"
        ],
        "reservation_input_capability_sha256": binding[
            "reservation_input_capability_sha256"
        ],
        "controller_authority_binding_sha256": binding[
            "controller_authority_binding_sha256"
        ],
    }
    if (
        any(
            material.get(field_name) != expected_value
            for field_name, expected_value in expected.items()
        )
        or material.get("goal_id") != authority.manifest.goal_id
        or material.get("authority_id") != authority.manifest.authority_id
        or material.get("purpose") != authority.manifest.purpose
        or material.get("stage") != authority.manifest.stage
        or material.get("forward_manifest_sha256")
        != authority.manifest.manifest_sha256
        or material.get("input_identity_sha256")
        != authority.manifest.input_identity_sha256
        or material.get("authority_binding_sha256")
        != authority.authority_binding_sha256
        or material.get("gpu0_identity_sha256")
        != authority.manifest.gpu0_identity.semantic_sha256
        or material.get("capability_id")
        != _sha256_bytes(
            _canonical_json_bytes(
                {
                    field_name: value
                    for field_name, value in material.items()
                    if field_name != "capability_id"
                }
            )
        )
        or _semantic_sha256(material)
        != binding["input_capability_material_sha256"]
    ):
        raise ForwardContractError(
            "REHYDRATED_INPUT_CAPABILITY_MATERIAL_BINDING_DRIFT"
        )
    for field_name in (
        "capability_id",
        "forward_manifest_sha256",
        "input_identity_sha256",
        "authority_binding_sha256",
        "token_snapshot_sha256",
        "gpu0_identity_sha256",
        "control_chain_sha256",
        "running_commit_sha256",
        "running_capability_sha256",
        "reservation_input_capability_sha256",
        "controller_authority_binding_sha256",
        "consume_receipt_sha256",
    ):
        _require_sha256(material.get(field_name), field_name)
    return MappingProxyType(
        json.loads(
            _canonical_json_bytes(dict(material)).decode("utf-8", "strict")
        )
    )


def _register_completed_result_verifier(
    authority: CompletedForwardResultVerificationAuthority,
    capability: CompletedForwardResultCapability,
) -> None:
    if (
        type(authority) is not CompletedForwardResultVerificationAuthority
        or type(capability) is not CompletedForwardResultCapability
        or capability._authority is not authority
        or authority._issued_capability is not None
    ):
        raise ForwardContractError(
            "COMPLETED_RESULT_VERIFIER_REGISTRATION_INVALID"
        )
    with _COMPLETED_RESULT_AUTHORITY_LOCK:
        if id(authority) in _COMPLETED_RESULT_AUTHORITY_REGISTRY:
            raise ForwardContractError(
                "COMPLETED_RESULT_AUTHORITY_ID_COLLISION"
            )
        object.__setattr__(authority, "_issued_capability", capability)
        _COMPLETED_RESULT_AUTHORITY_REGISTRY[id(authority)] = {
            "object": authority,
            "capability": capability,
            "projection_binding_sha256": (
                authority.projection_binding_sha256
            ),
        }


def _rollback_completed_result_verifier_registration(
    authority: CompletedForwardResultVerificationAuthority,
    capability: CompletedForwardResultCapability,
) -> None:
    """Remove only the exact provisional pair after publication failure."""

    with _COMPLETED_RESULT_AUTHORITY_LOCK:
        record = _COMPLETED_RESULT_AUTHORITY_REGISTRY.get(id(authority))
        if (
            isinstance(record, Mapping)
            and record.get("object") is authority
            and record.get("capability") is capability
        ):
            del _COMPLETED_RESULT_AUTHORITY_REGISTRY[id(authority)]
        if authority._issued_capability is capability:
            object.__setattr__(authority, "_issued_capability", None)


def _issue_original_completed_result_verifier(
    authority: FrozenInputAuthority,
    capability: InputCapability,
    *,
    run_root: FrozenDirectory,
    chunk_root: FrozenDirectory,
) -> Tuple[
    CompletedForwardResultVerificationAuthority,
    CompletedForwardResultCapability,
    Mapping[str, Any],
]:
    if (
        type(authority) is not FrozenInputAuthority
        or type(capability) is not InputCapability
        or capability is not authority._issued_capability
        or capability._authority is not authority
        or capability._issuer_nonce is not authority._issuer_nonce
        or authority._issued_capability is not capability
        or authority._issued_control is not capability._control
        or capability._run_receipt_parent is not run_root
        or capability._run_chunk_parent is not chunk_root
        or capability._run_receipt_path
        != os.path.join(run_root.path, "forward_run_receipt.json")
    ):
        raise ForwardContractError(
            "ORIGINAL_COMPLETED_RESULT_CAPABILITY_OBJECT_BINDING_INVALID"
        )
    for field_name, expected_value in authority._issued_capability_fields.items():
        if getattr(capability, field_name) != expected_value:
            raise ForwardContractError(
                "ORIGINAL_COMPLETED_RESULT_CAPABILITY_FIELD_DRIFT",
                field_name,
            )
    module = _require_exact_forward_control(capability._control)
    running = module.validate_eval_running_capability(
        capability._running_control_capability
    )
    if (
        capability._running_control_capability.capability_sha256
        != capability.running_capability_sha256
        or running.get("run_id") != capability.run_id
        or running.get("token_id") != capability.token_id
        or running.get("eval_id") != capability.eval_id
    ):
        raise ForwardContractError(
            "ORIGINAL_COMPLETED_RESULT_RUNNING_ANCHOR_DRIFT"
        )
    controller_authority = _validate_controller_issued_authority(
        authority,
        capability._control,
    )
    projection = _build_completed_forward_authority_rehydrate_projection(
        authority,
        forward_authority_handle_sha256=controller_authority[
            "authority_handle_sha256"
        ],
    )
    completed_authority = (
        _rehydrate_completed_forward_authority_from_projection(projection)
    )
    values = {
        field_name: getattr(capability, field_name)
        for field_name in InputCapability.__slots__
        if not field_name.startswith("_")
    }
    completed_capability = CompletedForwardResultCapability(
        _COMPLETED_RESULT_CAPABILITY_FACTORY,
        _factory=_COMPLETED_RESULT_CAPABILITY_FACTORY,
        **values,
        _authority=completed_authority,
        _control=capability._control,
        _proof_capability=capability._running_control_capability,
        _proof_mode="ORIGINAL_COMPLETED_LIFECYCLE",
        _input_capability_material_sha256=None,
        _run_receipt_sha256=capability._run_receipt_sha256,
        _run_receipt_path=capability._run_receipt_path,
        _run_receipt_parent=run_root,
        _run_chunk_parent=chunk_root,
    )
    completed_capability._assert_completed_lifecycle()
    return completed_authority, completed_capability, projection


_COMPLETED_FORWARD_ARTIFACT_HANDLE_FACTORY = object()
_COMPLETED_FORWARD_ARTIFACT_HANDLE_LOCK = threading.RLock()
_COMPLETED_FORWARD_ARTIFACT_HANDLE_REGISTRY: dict[int, dict[str, Any]] = {}
_VALIDATED_DERIVATION_TRANSFER_FACTORY = object()
_VALIDATED_DERIVATION_TRANSFER_LOCK = threading.RLock()
_VALIDATED_DERIVATION_TRANSFER_REGISTRY: dict[int, dict[str, Any]] = {}


def _validate_completed_forward_chunk_range(
    chunk: Any,
    *,
    sequence: int,
    expected_start: int,
    query_count: int,
) -> Tuple[int, int]:
    if not isinstance(chunk, Mapping):
        raise ForwardContractError(
            "COMPLETED_FORWARD_CHUNK_RANGE_OR_PATH_INVALID",
            str(sequence),
        )
    start = chunk.get("start_index")
    end = chunk.get("end_index")
    if (
        type(sequence) is not int
        or sequence < 0
        or type(expected_start) is not int
        or expected_start < 0
        or type(query_count) is not int
        or query_count <= 0
        or type(start) is not int
        or type(end) is not int
        or start != expected_start
        or end <= start
        or end > query_count
        or chunk.get("artifact_name")
        != "queries_%05d_%05d.pt" % (start, end)
    ):
        raise ForwardContractError(
            "COMPLETED_FORWARD_CHUNK_RANGE_OR_PATH_INVALID",
            str(sequence),
        )
    return start, end


def _validate_rehydrated_completed_forward_validation_receipt(
    value: Any,
    *,
    authority: CompletedForwardResultVerificationAuthority,
    capability: CompletedForwardResultCapability,
    run_root: FrozenDirectory,
    chunk_root: FrozenDirectory,
    rehydrate_binding: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Validate a persisted safe index without reopening any tensor chunk."""

    exact_fields = {
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
        "forward_manifest_sha256",
        "input_identity_sha256",
        "query_identity_sha256",
        "query_types_sha256",
        "query_rows_sha256",
        "query_count",
        "validated_start_index",
        "validated_end_index",
        "run_root_identity_sha256",
        "chunk_root_identity_sha256",
        "forward_run_receipt_artifact_name",
        "forward_run_receipt_sha256",
        "forward_run_receipt_file_sha256",
        "encoded_cache_binding",
        "completion_summary",
        "observation_summary",
        "result_chunk_count",
        "result_chunk_index",
        "result_chunk_set_sha256",
        "ordered_ranges",
        "ordered_ranges_sha256",
        "maximum_simultaneously_loaded_chunk_count",
        "payload_retention_contract",
        "payload_transfer_status",
        "validation_receipt_sha256",
    }
    query_identity = authority.manifest.query_identity
    query_count = len(query_identity.query_keys)
    expected = {
        "schema_version": (
            "c28f_a4_completed_forward_artifact_validation_v1"
        ),
        "status": "VALID_COMPLETED_FORWARD_RUN_ARTIFACTS",
        "goal_id": capability.goal_id,
        "attempt_id": authority.manifest.attempt_id,
        "authority_id": capability.authority_id,
        "run_id": capability.run_id,
        "token_id": capability.token_id,
        "eval_id": capability.eval_id,
        "stage": capability.stage,
        "purpose": capability.purpose,
        "capability_id": capability.capability_id,
        "forward_manifest_sha256": capability.forward_manifest_sha256,
        "input_identity_sha256": capability.input_identity_sha256,
        "query_identity_sha256": query_identity.identity_sha256,
        "query_types_sha256": query_identity.query_types_sha256,
        "query_rows_sha256": query_identity.query_rows_sha256,
        "query_count": query_count,
        "validated_start_index": 0,
        "validated_end_index": query_count,
        "run_root_identity_sha256": _sha256_bytes(
            _canonical_json_bytes(run_root.as_dict())
        ),
        "chunk_root_identity_sha256": _sha256_bytes(
            _canonical_json_bytes(chunk_root.as_dict())
        ),
        "forward_run_receipt_artifact_name": "forward_run_receipt.json",
        "forward_run_receipt_sha256": capability._run_receipt_sha256,
        "maximum_simultaneously_loaded_chunk_count": 1,
        "payload_retention_contract": (
            "HANDLE_STORES_NO_RAW_PAYLOAD_STREAM_REVALIDATES_ONE_CHUNK"
        ),
        "payload_transfer_status": (
            "OPAQUE_EXACT_CONTROLLER_DERIVATION_CONSUMER_REQUIRED"
        ),
    }
    if (
        not isinstance(value, Mapping)
        or set(value) != exact_fields
        or any(
            value.get(field_name) != expected_value
            for field_name, expected_value in expected.items()
        )
        or value.get("validation_receipt_sha256")
        != _semantic_sha256(value, ("validation_receipt_sha256",))
        or value.get("validation_receipt_sha256")
        != rehydrate_binding[
            "completed_forward_validation_receipt_sha256"
        ]
    ):
        raise ForwardContractError(
            "REHYDRATED_COMPLETED_FORWARD_VALIDATION_RECEIPT_INVALID"
        )
    receipt = _read_canonical_json(
        Path(run_root.path) / "forward_run_receipt.json",
        expected_parent=run_root,
    )
    _validate_run_receipt_contents(authority, capability, receipt)
    if (
        receipt.get("status") != "COMPLETED"
        or receipt.get("receipt_sha256")
        != rehydrate_binding["forward_run_receipt_sha256"]
        or value.get("forward_run_receipt_file_sha256")
        != _sha256_bytes(_canonical_json_bytes(receipt))
    ):
        raise ForwardContractError(
            "REHYDRATED_COMPLETED_FORWARD_RUN_RECEIPT_DRIFT"
        )
    safe_index = value.get("result_chunk_index")
    ordered_ranges = value.get("ordered_ranges")
    chunks = receipt.get("result_chunks")
    if (
        not isinstance(safe_index, list)
        or not isinstance(ordered_ranges, list)
        or not isinstance(chunks, list)
        or not safe_index
        or len(safe_index) != len(ordered_ranges) == len(chunks)
        or value.get("result_chunk_count") != len(safe_index)
    ):
        raise ForwardContractError(
            "REHYDRATED_COMPLETED_FORWARD_SAFE_INDEX_INVALID"
        )
    expected_start = 0
    exact_safe_fields = {
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
    }
    exact_range_fields = {
        "range_index",
        "start_index",
        "end_index",
        "artifact_name",
        "artifact_file_sha256",
        "payload_semantic_sha256",
    }
    for range_index, (safe_row, range_row, chunk) in enumerate(
        zip(safe_index, ordered_ranges, chunks)
    ):
        start, end = _validate_completed_forward_chunk_range(
            chunk,
            sequence=range_index,
            expected_start=expected_start,
            query_count=query_count,
        )
        if (
            not isinstance(safe_row, Mapping)
            or set(safe_row) != exact_safe_fields
            or not isinstance(range_row, Mapping)
            or set(range_row) != exact_range_fields
            or safe_row.get("range_index") != range_index
            or safe_row.get("start_index") != start
            or safe_row.get("end_index") != end
            or safe_row.get("row_count") != end - start
            or safe_row.get("artifact_name")
            != chunk.get("artifact_name")
            or safe_row.get("artifact_file_sha256")
            != chunk.get("artifact_sha256")
            or safe_row.get("payload_semantic_sha256")
            != chunk.get("semantic_sha256")
            or safe_row.get("engine_registry_consumption_sha256")
            != chunk.get("engine_registry_consumption_sha256")
            or safe_row.get("query_types_sha256")
            != chunk.get("query_types_sha256")
            or safe_row.get("query_rows_sha256")
            != chunk.get("query_rows_sha256")
            or dict(range_row)
            != {
                field_name: safe_row[field_name]
                for field_name in exact_range_fields
            }
        ):
            raise ForwardContractError(
                "REHYDRATED_COMPLETED_FORWARD_SAFE_INDEX_DRIFT",
                str(range_index),
            )
        expected_start = end
    if (
        expected_start != query_count
        or value.get("result_chunk_set_sha256")
        != _sha256_bytes(_canonical_json_bytes(safe_index))
        or value.get("ordered_ranges_sha256")
        != _sha256_bytes(_canonical_json_bytes(ordered_ranges))
    ):
        raise ForwardContractError(
            "REHYDRATED_COMPLETED_FORWARD_SAFE_INDEX_SEAL_DRIFT"
        )
    encoded_cache_binding = value.get("encoded_cache_binding")
    if (
        not isinstance(encoded_cache_binding, Mapping)
        or set(encoded_cache_binding)
        != {"artifact_name", "artifact_sha256", "semantic_sha256"}
        or encoded_cache_binding.get("artifact_name")
        != "encoded_corpus_cache_receipt"
        or encoded_cache_binding.get("artifact_sha256")
        != receipt.get("encoded_cache_receipt_sha256")
        or encoded_cache_binding.get("semantic_sha256")
        != receipt.get("encoded_cache_fingerprint")
        or encoded_cache_binding.get("semantic_sha256")
        != authority.encoded_cache_fingerprint
    ):
        raise ForwardContractError(
            "REHYDRATED_COMPLETED_FORWARD_CACHE_BINDING_DRIFT"
        )
    completion_summary = _validate_completed_forward_phase_receipt(
        capability=capability,
        run_root=run_root,
    )
    observation_summary = _validate_completed_forward_observation_artifacts(
        capability=capability,
        run_root=run_root,
        receipt=receipt,
    )
    observation_summary = {
        **observation_summary,
        "summary_sha256": _semantic_sha256(observation_summary),
    }
    if (
        value.get("completion_summary") != completion_summary
        or value.get("observation_summary") != observation_summary
    ):
        raise ForwardContractError(
            "REHYDRATED_COMPLETED_FORWARD_EVIDENCE_SUMMARY_DRIFT"
        )
    return MappingProxyType(
        json.loads(
            _canonical_json_bytes(dict(value)).decode("utf-8", "strict")
        )
    )


class ValidatedForwardDerivationRangeTransfer:
    """Private one-range payload bridge accepted only by the controller sink."""

    __slots__ = (
        "_factory",
        "_authority",
        "_capability",
        "_claim_capability",
        "_claim_binding_sha256",
        "_materialization_owner_generation",
        "_materialization_owner_binding_sha256",
        "_materialization_owner_takeover_chain_sha256",
        "_range_record",
        "_payload",
        "_consumed",
    )

    def __init__(self, factory: object, **values: Any) -> None:
        if factory is not _VALIDATED_DERIVATION_TRANSFER_FACTORY:
            raise ForwardContractError(
                "VALIDATED_DERIVATION_TRANSFER_CONSTRUCTION_FORBIDDEN"
            )
        for field_name in self.__slots__:
            object.__setattr__(self, field_name, values[field_name])

    def __setattr__(self, _name: str, _value: Any) -> None:
        raise ForwardContractError("VALIDATED_DERIVATION_TRANSFER_MUTATION_FORBIDDEN")


def _issue_validated_forward_derivation_range_transfer(
    *,
    authority: CompletedForwardResultVerificationAuthority,
    capability: CompletedForwardResultCapability,
    claim_capability: Any,
    claim_binding_sha256: str,
    materialization_owner_generation: int,
    materialization_owner_binding_sha256: str,
    materialization_owner_takeover_chain_sha256: str,
    range_record: Mapping[str, Any],
    payload: Mapping[str, Any],
    torch_module: Any,
) -> ValidatedForwardDerivationRangeTransfer:
    authority.assert_completed_capability(capability)
    _require_sha256(claim_binding_sha256, "claim_binding_sha256")
    if (
        type(materialization_owner_generation) is not int
        or not 0 <= materialization_owner_generation <= 8
    ):
        raise ForwardContractError("MATERIALIZATION_OWNER_GENERATION_INVALID")
    _require_sha256(
        materialization_owner_binding_sha256,
        "materialization_owner_binding_sha256",
    )
    _require_sha256(
        materialization_owner_takeover_chain_sha256,
        "materialization_owner_takeover_chain_sha256",
    )
    _validate_forward_result_payload(
        torch_module,
        payload,
        authority=authority,
        capability=capability,
        expected_start=range_record["start_index"],
        expected_end=range_record["end_index"],
    )
    if (
        _forward_result_payload_digest(torch_module, payload)
        != range_record["payload_semantic_sha256"]
    ):
        raise ForwardContractError("VALIDATED_DERIVATION_TRANSFER_SEMANTIC_DRIFT")
    transfer = ValidatedForwardDerivationRangeTransfer(
        _VALIDATED_DERIVATION_TRANSFER_FACTORY,
        _factory=_VALIDATED_DERIVATION_TRANSFER_FACTORY,
        _authority=authority,
        _capability=capability,
        _claim_capability=claim_capability,
        _claim_binding_sha256=claim_binding_sha256,
        _materialization_owner_generation=materialization_owner_generation,
        _materialization_owner_binding_sha256=(
            materialization_owner_binding_sha256
        ),
        _materialization_owner_takeover_chain_sha256=(
            materialization_owner_takeover_chain_sha256
        ),
        _range_record=MappingProxyType(dict(range_record)),
        _payload=MappingProxyType(dict(payload)),
        _consumed=False,
    )
    with _VALIDATED_DERIVATION_TRANSFER_LOCK:
        _VALIDATED_DERIVATION_TRANSFER_REGISTRY[id(transfer)] = {
            "object": transfer,
            "authority": authority,
            "capability": capability,
            "claim_capability": claim_capability,
            "claim_binding_sha256": claim_binding_sha256,
            "materialization_owner_generation": (
                materialization_owner_generation
            ),
            "materialization_owner_binding_sha256": (
                materialization_owner_binding_sha256
            ),
            "materialization_owner_takeover_chain_sha256": (
                materialization_owner_takeover_chain_sha256
            ),
            "range_record": dict(range_record),
            "state": "ISSUED",
        }
    return transfer


def _controller_metric_record_from_forward_payload_row(
    row: Mapping[str, Any],
    *,
    corpus: FrozenCorpusIdentity,
) -> dict[str, Any]:
    proposals: list[dict[str, Any]] = []
    for candidate_offset, video_index in enumerate(
        row["joint_candidate_indices"]
    ):
        video_id = corpus.video_ids[video_index]
        for proposal_offset in range(int(row["joint_scores"].shape[1])):
            if not bool(row["span_mask"][candidate_offset, proposal_offset]):
                continue
            proposals.append(
                {
                    "video_id": video_id,
                    "start_sec": float(
                        row["spans_sec"][
                            candidate_offset,
                            proposal_offset,
                            0,
                        ]
                    ),
                    "end_sec": float(
                        row["spans_sec"][
                            candidate_offset,
                            proposal_offset,
                            1,
                        ]
                    ),
                    "score": float(
                        row["joint_scores"][
                            candidate_offset,
                            proposal_offset,
                        ]
                    ),
                    "probability": float(
                        row["joint_probabilities"][
                            candidate_offset,
                            proposal_offset,
                        ]
                    ),
                }
            )
    record = {
        "query_id": int(row["query_id"]),
        "query_type": row["query_type"],
        "pooled_video_order": [
            {"video_id": corpus.video_ids[index], "score": float(score)}
            for index, score in zip(
                row["pooled_indices"],
                row["pooled_scores"],
            )
        ],
        "late_video_order": [
            {"video_id": corpus.video_ids[index], "score": float(score)}
            for index, score in zip(
                row["late_indices"],
                row["late_scores"],
            )
        ],
        "final_video_order": [
            {"video_id": corpus.video_ids[index], "score": float(score)}
            for index, score in zip(
                row["final_indices"],
                row["final_scores"],
            )
        ],
        "joint_proposals": proposals,
        "forward_output_semantics": "RAW_P_FULL_NO_NMS",
        "forensics": json.loads(
            _canonical_json_bytes(dict(row["forensics"])).decode(
                "utf-8",
                "strict",
            )
        ),
        "teacher_candidate_count": 0,
        "gt_support_count": 0,
        "gt_append_count": 0,
        "optimizer_update_count": 0,
        "result_sha256": row["result_sha256"],
    }
    if set(record) != {
        "query_id",
        "query_type",
        "pooled_video_order",
        "late_video_order",
        "final_video_order",
        "joint_proposals",
        "forward_output_semantics",
        "forensics",
        "teacher_candidate_count",
        "gt_support_count",
        "gt_append_count",
        "optimizer_update_count",
        "result_sha256",
    }:
        raise ForwardContractError(
            "CONTROLLER_METRIC_RECORD_FIELD_SET_DRIFT"
        )
    return record


def consume_validated_forward_derivation_range_transfer_once(
    value: Any,
    *,
    claim_capability: Any,
) -> Mapping[str, Any]:
    """Consume raw tensors into a canonical controller-only pre-GT projection."""

    if (
        type(value) is not ValidatedForwardDerivationRangeTransfer
        or value._factory is not _VALIDATED_DERIVATION_TRANSFER_FACTORY
    ):
        raise ForwardContractError("EXACT_VALIDATED_DERIVATION_TRANSFER_REQUIRED")
    module = _require_exact_forward_control(value._capability._control)
    try:
        claim = module.validate_forward_derivation_range_claim_capability(
            claim_capability
        )
        owner = module.validate_forward_derivation_delivery_owner(
            claim_capability
        )
    except BaseException as error:
        raise ForwardContractError(
            "VALIDATED_DERIVATION_TRANSFER_CURRENT_OWNER_REQUIRED",
            type(error).__name__,
        ) from None
    if not isinstance(claim, Mapping) or not isinstance(owner, Mapping):
        raise ForwardContractError(
            "VALIDATED_DERIVATION_TRANSFER_OWNER_BINDING_DRIFT"
        )
    claim_binding_sha256 = _require_sha256(
        claim.get("claim_binding_sha256"),
        "claim_binding_sha256",
    )
    materialization_owner_generation = owner.get("generation")
    materialization_owner_binding_sha256 = owner.get(
        "owner_binding_sha256"
    )
    materialization_owner_takeover_chain_sha256 = owner.get(
        "takeover_chain_sha256"
    )
    if (
        type(materialization_owner_generation) is not int
        or not 0 <= materialization_owner_generation <= 8
    ):
        raise ForwardContractError("MATERIALIZATION_OWNER_GENERATION_INVALID")
    _require_sha256(
        materialization_owner_binding_sha256,
        "materialization_owner_binding_sha256",
    )
    _require_sha256(
        materialization_owner_takeover_chain_sha256,
        "materialization_owner_takeover_chain_sha256",
    )
    if (
        set(owner)
        != {
            "schema_version",
            "run_id",
            "range_index",
            "generation",
            "owner_binding_sha256",
            "takeover_chain_sha256",
            "binding_sha256",
        }
        or owner.get("schema_version")
        != "c28f_a4_forward_derivation_delivery_owner_binding_v1"
        or owner.get("run_id") != claim.get("run_id")
        or owner.get("range_index") != claim.get("range_index")
        or owner.get("binding_sha256")
        != _semantic_sha256(owner, ("binding_sha256",))
    ):
        raise ForwardContractError(
            "VALIDATED_DERIVATION_TRANSFER_OWNER_BINDING_DRIFT"
        )
    with _VALIDATED_DERIVATION_TRANSFER_LOCK:
        record = _VALIDATED_DERIVATION_TRANSFER_REGISTRY.get(id(value))
        if (
            not isinstance(record, Mapping)
            or record.get("object") is not value
            or record.get("authority") is not value._authority
            or record.get("capability") is not value._capability
            or record.get("claim_capability") is not claim_capability
            or value._claim_capability is not claim_capability
            or record.get("claim_binding_sha256") != claim_binding_sha256
            or value._claim_binding_sha256 != claim_binding_sha256
            or record.get("materialization_owner_generation")
            != materialization_owner_generation
            or value._materialization_owner_generation
            != materialization_owner_generation
            or record.get("materialization_owner_binding_sha256")
            != materialization_owner_binding_sha256
            or value._materialization_owner_binding_sha256
            != materialization_owner_binding_sha256
            or record.get(
                "materialization_owner_takeover_chain_sha256"
            )
            != materialization_owner_takeover_chain_sha256
            or value._materialization_owner_takeover_chain_sha256
            != materialization_owner_takeover_chain_sha256
            or claim.get("run_id") != value._capability.run_id
            or claim.get("range_index")
            != value._range_record["range_index"]
            or record.get("range_record") != dict(value._range_record)
            or record.get("state") != "ISSUED"
            or value._consumed is not False
            or not isinstance(value._payload, Mapping)
        ):
            raise ForwardContractError(
                "VALIDATED_DERIVATION_TRANSFER_NOT_ISSUED_OR_REPLAYED"
            )
        torch_module = _lazy_torch()
        value._authority.assert_completed_capability(value._capability)
        _validate_forward_result_payload(
            torch_module,
            value._payload,
            authority=value._authority,
            capability=value._capability,
            expected_start=value._range_record["start_index"],
            expected_end=value._range_record["end_index"],
        )
        if (
            _forward_result_payload_digest(torch_module, value._payload)
            != value._range_record["payload_semantic_sha256"]
        ):
            raise ForwardContractError(
                "VALIDATED_DERIVATION_TRANSFER_RUNTIME_DRIFT"
            )
        payload = value._payload
        metric_records = [
            _controller_metric_record_from_forward_payload_row(
                row,
                corpus=value._authority.manifest.corpus_identity,
            )
            for row in payload["rows"]
        ]
        projection_base = {
            "schema_version": (
                "c28f_a4_forward_derivation_controller_projection_v1"
            ),
            "status": "VALIDATED_CONTROLLER_ONLY_PRE_GT_PROJECTION",
            "run_id": value._capability.run_id,
            "token_id": value._capability.token_id,
            "eval_id": value._capability.eval_id,
            "range_index": value._range_record["range_index"],
            "start_index": value._range_record["start_index"],
            "end_index": value._range_record["end_index"],
            "artifact_name": value._range_record["artifact_name"],
            "artifact_file_sha256": value._range_record[
                "artifact_file_sha256"
            ],
            "payload_semantic_sha256": value._range_record[
                "payload_semantic_sha256"
            ],
            "engine_registry_consumption_sha256": payload[
                "engine_registry_consumption_sha256"
            ],
            "query_types_sha256": payload["query_types_sha256"],
            "query_rows_sha256": payload["query_rows_sha256"],
            "metric_records": metric_records,
            "metric_records_sha256": _sha256_bytes(
                _canonical_json_bytes(metric_records)
            ),
        }
        projection = {
            **projection_base,
            "transfer_binding_sha256": _semantic_sha256(projection_base),
        }
        record["state"] = "CONSUMED_BY_CONTROLLER_SINK"
        object.__setattr__(value, "_payload", None)
        object.__setattr__(value, "_consumed", True)
        del payload
    return MappingProxyType(projection)


class CompletedForwardRunArtifactHandle:
    """Opaque safe index; raw payload memory is bounded to one consumed chunk."""

    __slots__ = (
        "_factory",
        "_authority",
        "_capability",
        "_control",
        "_run_root",
        "_chunk_root",
        "_chunk_index",
        "_next_chunk_index",
        "_previous_range_receipt_sha256",
        "_validation_receipt",
    )

    def __init__(self, factory: object, **values: Any) -> None:
        if factory is not _COMPLETED_FORWARD_ARTIFACT_HANDLE_FACTORY:
            raise ForwardContractError(
                "COMPLETED_FORWARD_ARTIFACT_HANDLE_CONSTRUCTION_FORBIDDEN"
            )
        for field_name in self.__slots__:
            object.__setattr__(self, field_name, values[field_name])

    def __setattr__(self, _name: str, _value: Any) -> None:
        raise ForwardContractError("COMPLETED_FORWARD_ARTIFACT_HANDLE_MUTATION_FORBIDDEN")

    def validation_receipt(self) -> Mapping[str, Any]:
        value = dict(self._validation_receipt)
        if value.get("validation_receipt_sha256") != _semantic_sha256(
            value,
            ("validation_receipt_sha256",),
        ):
            raise ForwardContractError(
                "COMPLETED_FORWARD_VALIDATION_RECEIPT_RUNTIME_DRIFT"
            )
        return json.loads(_canonical_json_bytes(value).decode("utf-8"))

    def _assert_runtime_integrity(self) -> None:
        if (
            self._factory is not _COMPLETED_FORWARD_ARTIFACT_HANDLE_FACTORY
            or type(self._authority)
            is not CompletedForwardResultVerificationAuthority
            or type(self._capability)
            is not CompletedForwardResultCapability
            or self._control is not self._capability._control
            or type(self._next_chunk_index) is not int
            or not 0 <= self._next_chunk_index <= len(self._chunk_index)
            or (
                self._previous_range_receipt_sha256 is not None
                and (
                    not isinstance(self._previous_range_receipt_sha256, str)
                    or len(self._previous_range_receipt_sha256) != 64
                    or any(
                        character not in _SHA256_HEX
                        for character in self._previous_range_receipt_sha256
                    )
                )
            )
        ):
            raise ForwardContractError("COMPLETED_FORWARD_ARTIFACT_HANDLE_RUNTIME_DRIFT")
        self._authority.assert_completed_capability(self._capability)
        _require_exact_forward_control(self._control)
        _validate_bound_completed_forward_roots(
            authority=self._authority,
            capability=self._capability,
            run_root=self._run_root,
            chunk_root=self._chunk_root,
        )
        validation = self.validation_receipt()
        encoded_cache_binding = validation.get("encoded_cache_binding")
        if (
            validation.get("result_chunk_count") != len(self._chunk_index)
            or validation.get("result_chunk_index")
            != [dict(chunk) for chunk in self._chunk_index]
            or validation.get("maximum_simultaneously_loaded_chunk_count") != 1
            or validation.get("payload_retention_contract")
            != "HANDLE_STORES_NO_RAW_PAYLOAD_STREAM_REVALIDATES_ONE_CHUNK"
            or not isinstance(encoded_cache_binding, Mapping)
            or set(encoded_cache_binding)
            != {"artifact_name", "artifact_sha256", "semantic_sha256"}
            or encoded_cache_binding
            != {
                "artifact_name": "encoded_corpus_cache_receipt",
                "artifact_sha256": encoded_cache_binding.get(
                    "artifact_sha256"
                ),
                "semantic_sha256": self._authority.encoded_cache_fingerprint,
            }
            or not isinstance(
                encoded_cache_binding.get("artifact_sha256"),
                str,
            )
            or len(encoded_cache_binding["artifact_sha256"]) != 64
            or any(
                character not in _SHA256_HEX
                for character in encoded_cache_binding[
                    "artifact_sha256"
                ]
            )
        ):
            raise ForwardContractError(
                "COMPLETED_FORWARD_ARTIFACT_HANDLE_INDEX_DRIFT"
            )

    def transfer_next_validated_chunk_to_controller_once(
        self,
        *,
        consumer_capability: Any,
    ) -> Mapping[str, Any]:
        """Two-phase transfer one range; return only the durable safe receipt."""

        self._assert_runtime_integrity()
        module = _require_exact_forward_control(self._control)
        try:
            consumer = module.validate_forward_derivation_consumer_capability(
                consumer_capability
            )
        except BaseException as error:
            raise ForwardContractError(
                "EXACT_FORWARD_DERIVATION_CONSUMER_REQUIRED",
                type(error).__name__,
            ) from None
        exact_consumer_fields = {
            "schema_version",
            "goal_id",
            "attempt_id",
            "run_id",
            "token_id",
            "eval_id",
            "completed_transaction_id",
            "completed_state_sha256",
            "completed_event_sha256",
            "forward_authority_handle_sha256",
            "forward_manifest_sha256",
            "input_identity_sha256",
            "reservation_input_capability_sha256",
            "result_chunk_set_sha256",
            "result_chunk_count",
            "ordered_ranges",
            "ordered_ranges_sha256",
            "single_use_per_range",
            "strict_in_order",
            "consumer_binding_sha256",
        }
        validation = self.validation_receipt()
        expected_consumer = {
            "schema_version": "c28f_a4_forward_derivation_consumer_capability_v1",
            "goal_id": self._capability.goal_id,
            "attempt_id": self._authority.manifest.attempt_id,
            "run_id": self._capability.run_id,
            "token_id": self._capability.token_id,
            "eval_id": self._capability.eval_id,
            "forward_authority_handle_sha256": (
                self._authority.forward_authority_handle_sha256
            ),
            "forward_manifest_sha256": self._capability.forward_manifest_sha256,
            "input_identity_sha256": self._capability.input_identity_sha256,
            "reservation_input_capability_sha256": (
                self._capability.reservation_input_capability_sha256
            ),
            "result_chunk_set_sha256": validation["result_chunk_set_sha256"],
            "result_chunk_count": validation["result_chunk_count"],
            "ordered_ranges": validation["ordered_ranges"],
            "ordered_ranges_sha256": validation["ordered_ranges_sha256"],
            "single_use_per_range": True,
            "strict_in_order": True,
        }
        if (
            not isinstance(consumer, Mapping)
            or set(consumer) != exact_consumer_fields
            or any(
                consumer.get(field_name) != expected_value
                for field_name, expected_value in expected_consumer.items()
            )
            or consumer.get("consumer_binding_sha256")
            != _semantic_sha256(consumer, ("consumer_binding_sha256",))
            or _sha256_bytes(
                _require_text(
                    consumer.get("completed_transaction_id"),
                    "completed_transaction_id",
                ).encode("utf-8")
            )
            != validation["completion_summary"]["transaction_id_sha256"]
        ):
            raise ForwardContractError(
                "FORWARD_DERIVATION_CONSUMER_BINDING_DRIFT"
            )
        for field_name in (
            "completed_state_sha256",
            "completed_event_sha256",
            "forward_authority_handle_sha256",
            "forward_manifest_sha256",
            "input_identity_sha256",
            "reservation_input_capability_sha256",
            "result_chunk_set_sha256",
            "ordered_ranges_sha256",
            "consumer_binding_sha256",
        ):
            _require_sha256(consumer.get(field_name), field_name)
        try:
            resume = module.forward_derivation_resume_snapshot(
                consumer_capability
            )
        except BaseException as error:
            raise ForwardContractError(
                "FORWARD_DERIVATION_RESUME_SNAPSHOT_FAILED",
                type(error).__name__,
            ) from None
        exact_resume_fields = {
            "schema_version",
            "status",
            "goal_id",
            "attempt_id",
            "consumer_capability_sha256",
            "consumer_binding_sha256",
            "run_id",
            "token_id",
            "eval_id",
            "result_chunk_count",
            "next_cursor_index",
            "previous_range_receipt_sha256",
            "consumed_range_receipt_sha256s",
            "consumed_range_receipt_chain_sha256",
            "all_ranges_consumed",
            "snapshot_sha256",
        }
        consumed_receipt_sha256s = (
            resume.get("consumed_range_receipt_sha256s")
            if isinstance(resume, Mapping)
            else None
        )
        resume_cursor = (
            resume.get("next_cursor_index")
            if isinstance(resume, Mapping)
            else None
        )
        if (
            not isinstance(resume, Mapping)
            or set(resume) != exact_resume_fields
            or resume.get("schema_version")
            != "c28f_a4_forward_derivation_resume_snapshot_v1"
            or resume.get("status")
            != "FSYNCED_DERIVATION_RESUME_SNAPSHOT"
            or resume.get("goal_id") != consumer["goal_id"]
            or resume.get("attempt_id") != consumer["attempt_id"]
            or resume.get("consumer_capability_sha256")
            != consumer_capability.capability_sha256
            or resume.get("consumer_binding_sha256")
            != consumer["consumer_binding_sha256"]
            or resume.get("run_id") != consumer["run_id"]
            or resume.get("token_id") != consumer["token_id"]
            or resume.get("eval_id") != consumer["eval_id"]
            or resume.get("result_chunk_count") != len(self._chunk_index)
            or type(resume_cursor) is not int
            or not 0 <= resume_cursor <= len(self._chunk_index)
            or not isinstance(consumed_receipt_sha256s, list)
            or len(consumed_receipt_sha256s) != resume_cursor
            or any(
                not isinstance(item, str)
                or len(item) != 64
                or any(character not in _SHA256_HEX for character in item)
                for item in consumed_receipt_sha256s
            )
            or resume.get("consumed_range_receipt_chain_sha256")
            != _sha256_bytes(
                _canonical_json_bytes(consumed_receipt_sha256s)
            )
            or resume.get("previous_range_receipt_sha256")
            != (
                None
                if resume_cursor == 0
                else consumed_receipt_sha256s[-1]
            )
            or resume.get("all_ranges_consumed")
            is not (resume_cursor == len(self._chunk_index))
            or resume.get("snapshot_sha256")
            != _semantic_sha256(resume, ("snapshot_sha256",))
            or resume_cursor < self._next_chunk_index
            or (
                self._next_chunk_index > 0
                and consumed_receipt_sha256s[self._next_chunk_index - 1]
                != self._previous_range_receipt_sha256
            )
        ):
            raise ForwardContractError(
                "FORWARD_DERIVATION_RESUME_SNAPSHOT_INVALID"
            )
        object.__setattr__(self, "_next_chunk_index", resume_cursor)
        object.__setattr__(
            self,
            "_previous_range_receipt_sha256",
            resume.get("previous_range_receipt_sha256"),
        )
        range_index = self._next_chunk_index
        if range_index >= len(self._chunk_index):
            raise ForwardContractError("FORWARD_DERIVATION_RANGES_EXHAUSTED")
        expected_range = validation["ordered_ranges"][range_index]
        if (
            expected_range.get("range_index") != range_index
            or self._chunk_index[range_index]["range_index"] != range_index
        ):
            raise ForwardContractError("FORWARD_DERIVATION_RANGE_ORDER_DRIFT")
        try:
            claim_capability = module.prepare_forward_derivation_range(
                consumer_capability=consumer_capability,
                run_id=self._capability.run_id,
                start_index=expected_range["start_index"],
                end_index=expected_range["end_index"],
                artifact_file_sha256=expected_range[
                    "artifact_file_sha256"
                ],
                payload_semantic_sha256=expected_range[
                    "payload_semantic_sha256"
                ],
            )
            claim = (
                module.validate_forward_derivation_range_claim_capability(
                    claim_capability
                )
            )
        except BaseException as error:
            raise ForwardContractError(
                "FORWARD_DERIVATION_RANGE_CLAIM_FAILED",
                type(error).__name__,
            ) from None
        exact_claim_fields = {
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
            "result_chunk_set_sha256",
            "range_index",
            "start_index",
            "end_index",
            "artifact_name",
            "artifact_file_sha256",
            "payload_semantic_sha256",
            "previous_range_receipt_sha256",
            "previous_cursor_index",
            "claim_binding_sha256",
        }
        expected_claim = {
            "schema_version": (
                "c28f_a4_forward_derivation_range_claim_capability_v1"
            ),
            "status": "FSYNCED_DERIVATION_RANGE_CLAIMED",
            "goal_id": consumer["goal_id"],
            "attempt_id": consumer["attempt_id"],
            "run_id": consumer["run_id"],
            "token_id": consumer["token_id"],
            "eval_id": consumer["eval_id"],
            "completed_transaction_id": consumer[
                "completed_transaction_id"
            ],
            "consumer_capability_sha256": (
                consumer_capability.capability_sha256
            ),
            "consumer_binding_sha256": consumer[
                "consumer_binding_sha256"
            ],
            "result_chunk_set_sha256": consumer[
                "result_chunk_set_sha256"
            ],
            **expected_range,
            "previous_range_receipt_sha256": (
                self._previous_range_receipt_sha256
            ),
            "previous_cursor_index": range_index,
        }
        if (
            not isinstance(claim, Mapping)
            or set(claim) != exact_claim_fields
            or any(
                claim.get(field_name) != expected_value
                for field_name, expected_value in expected_claim.items()
            )
            or claim.get("claim_binding_sha256")
            != _semantic_sha256(claim, ("claim_binding_sha256",))
        ):
            raise ForwardContractError(
                "FORWARD_DERIVATION_RANGE_CLAIM_INVALID"
            )
        try:
            delivery_owner = (
                module.validate_forward_derivation_delivery_owner(
                    claim_capability
                )
            )
        except BaseException as error:
            raise ForwardContractError(
                "FORWARD_DERIVATION_DELIVERY_OWNER_INVALID",
                type(error).__name__,
            ) from None
        exact_delivery_owner_fields = {
            "schema_version",
            "run_id",
            "range_index",
            "generation",
            "owner_binding_sha256",
            "takeover_chain_sha256",
            "binding_sha256",
        }
        if (
            not isinstance(delivery_owner, Mapping)
            or set(delivery_owner) != exact_delivery_owner_fields
            or delivery_owner.get("schema_version")
            != "c28f_a4_forward_derivation_delivery_owner_binding_v1"
            or delivery_owner.get("run_id") != consumer["run_id"]
            or delivery_owner.get("range_index") != range_index
            or type(delivery_owner.get("generation")) is not int
            or not 0 <= delivery_owner["generation"] <= 8
            or any(
                not isinstance(delivery_owner.get(field_name), str)
                or len(delivery_owner[field_name]) != 64
                or any(
                    character not in _SHA256_HEX
                    for character in delivery_owner[field_name]
                )
                for field_name in (
                    "owner_binding_sha256",
                    "takeover_chain_sha256",
                    "binding_sha256",
                )
            )
            or delivery_owner.get("binding_sha256")
            != _semantic_sha256(
                delivery_owner,
                ("binding_sha256",),
            )
        ):
            raise ForwardContractError(
                "FORWARD_DERIVATION_DELIVERY_OWNER_BINDING_DRIFT"
            )
        try:
            postimage_capability = (
                module.rehydrate_forward_derivation_postimage_if_present(
                    claim_capability=claim_capability,
                )
            )
        except BaseException as error:
            raise ForwardContractError(
                "FORWARD_DERIVATION_POSTIMAGE_RECOVERY_PROBE_FAILED",
                type(error).__name__,
            ) from None
        recovered_postimage = postimage_capability is not None
        if not recovered_postimage:
            chunk = self._chunk_index[range_index]
            artifact_path = (
                Path(self._chunk_root.path) / chunk["artifact_name"]
            )
            artifact_bytes = _read_bytes_no_follow(
                artifact_path,
                max_bytes=MAX_OWNED_TORCH_ARCHIVE_BYTES,
                expected_parent=self._chunk_root,
            )
            if _sha256_bytes(artifact_bytes) != chunk[
                "artifact_file_sha256"
            ]:
                raise ForwardContractError(
                    "FORWARD_DERIVATION_CHUNK_FILE_HASH_DRIFT"
                )
            del artifact_bytes
            torch_module = _lazy_torch()
            payload = _safe_load_owned_torch_artifact(
                artifact_path,
                torch_module,
                expected_parent=self._chunk_root,
                expected_file_sha256=chunk["artifact_file_sha256"],
            )
            if (
                not isinstance(payload, Mapping)
                or payload.get("engine_registry_consumption_sha256")
                != chunk["engine_registry_consumption_sha256"]
                or payload.get("query_types_sha256")
                != chunk["query_types_sha256"]
                or payload.get("query_rows_sha256")
                != chunk["query_rows_sha256"]
            ):
                raise ForwardContractError(
                    "FORWARD_DERIVATION_CHUNK_LINEAGE_DRIFT"
                )
            _validate_forward_result_payload(
                torch_module,
                payload,
                authority=self._authority,
                capability=self._capability,
                expected_start=chunk["start_index"],
                expected_end=chunk["end_index"],
            )
            if (
                _forward_result_payload_digest(torch_module, payload)
                != chunk["payload_semantic_sha256"]
            ):
                raise ForwardContractError(
                    "FORWARD_DERIVATION_CHUNK_SEMANTIC_DRIFT"
                )
            transfer = _issue_validated_forward_derivation_range_transfer(
                authority=self._authority,
                capability=self._capability,
                claim_capability=claim_capability,
                claim_binding_sha256=claim["claim_binding_sha256"],
                materialization_owner_generation=delivery_owner["generation"],
                materialization_owner_binding_sha256=delivery_owner[
                    "owner_binding_sha256"
                ],
                materialization_owner_takeover_chain_sha256=delivery_owner[
                    "takeover_chain_sha256"
                ],
                range_record=expected_range,
                payload=payload,
                torch_module=torch_module,
            )
            del payload
            try:
                postimage_capability = module.derive_and_fsync_forward_range(
                    claim_capability=claim_capability,
                    validated_range_transfer=transfer,
                )
            except BaseException as error:
                raise ForwardContractError(
                    "FORWARD_DERIVATION_RANGE_POSTIMAGE_FAILED",
                    type(error).__name__,
                ) from None
        try:
            postimage = (
                module.validate_forward_derived_range_postimage_capability(
                    postimage_capability
                )
            )
        except BaseException as error:
            raise ForwardContractError(
                "FORWARD_DERIVATION_RANGE_POSTIMAGE_FAILED",
                type(error).__name__,
            ) from None
        exact_postimage_fields = {
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
            "range_index",
            "start_index",
            "end_index",
            "source_artifact_name",
            "source_artifact_file_sha256",
            "source_payload_semantic_sha256",
            "controller_projection_sha256",
            "pre_gt_artifact_name",
            "pre_gt_artifact_file_sha256",
            "pre_gt_payload_sha256",
            "derived_nms_artifact_name",
            "derived_nms_artifact_file_sha256",
            "derived_nms_payload_sha256",
            "materialization_owner_generation",
            "materialization_owner_binding_sha256",
            "materialization_owner_takeover_chain_sha256",
            "postimage_binding_sha256",
        }
        expected_postimage = {
            "schema_version": (
                "c28f_a4_forward_derived_range_postimage_capability_v1"
            ),
            "status": "FSYNCED_CONTROLLER_DERIVED_RANGE_POSTIMAGE",
            "goal_id": consumer["goal_id"],
            "attempt_id": consumer["attempt_id"],
            "run_id": consumer["run_id"],
            "token_id": consumer["token_id"],
            "eval_id": consumer["eval_id"],
            "completed_transaction_id": consumer[
                "completed_transaction_id"
            ],
            "consumer_capability_sha256": (
                consumer_capability.capability_sha256
            ),
            "consumer_binding_sha256": consumer[
                "consumer_binding_sha256"
            ],
            "claim_capability_sha256": claim_capability.capability_sha256,
            "claim_binding_sha256": claim["claim_binding_sha256"],
            "range_index": range_index,
            "start_index": expected_range["start_index"],
            "end_index": expected_range["end_index"],
            "source_artifact_name": expected_range["artifact_name"],
            "source_artifact_file_sha256": expected_range[
                "artifact_file_sha256"
            ],
            "source_payload_semantic_sha256": expected_range[
                "payload_semantic_sha256"
            ],
        }
        if (
            not isinstance(postimage, Mapping)
            or set(postimage) != exact_postimage_fields
            or any(
                postimage.get(field_name) != expected_value
                for field_name, expected_value in expected_postimage.items()
            )
            or any(
                not isinstance(postimage.get(field_name), str)
                or len(postimage[field_name]) != 64
                or any(
                    character not in _SHA256_HEX
                    for character in postimage[field_name]
                )
                for field_name in (
                    "controller_projection_sha256",
                    "pre_gt_artifact_file_sha256",
                    "pre_gt_payload_sha256",
                    "derived_nms_artifact_file_sha256",
                    "derived_nms_payload_sha256",
                    "materialization_owner_binding_sha256",
                    "materialization_owner_takeover_chain_sha256",
                    "postimage_binding_sha256",
                )
            )
            or type(postimage.get("materialization_owner_generation"))
            is not int
            or postimage["materialization_owner_generation"] < 0
            or postimage["materialization_owner_generation"]
            > delivery_owner["generation"]
            or (
                not recovered_postimage
                and (
                    postimage["materialization_owner_generation"]
                    != delivery_owner["generation"]
                    or postimage["materialization_owner_binding_sha256"]
                    != delivery_owner["owner_binding_sha256"]
                    or postimage[
                        "materialization_owner_takeover_chain_sha256"
                    ]
                    != delivery_owner["takeover_chain_sha256"]
                )
            )
            or re.fullmatch(
                r"pre_gt_%06d_[0-9a-f]{64}\.json" % range_index,
                str(postimage.get("pre_gt_artifact_name")),
            )
            is None
            or re.fullmatch(
                r"derived_nms_%06d_[0-9a-f]{64}\.json" % range_index,
                str(postimage.get("derived_nms_artifact_name")),
            )
            is None
            or postimage.get("postimage_binding_sha256")
            != _semantic_sha256(
                postimage,
                ("postimage_binding_sha256",),
            )
        ):
            raise ForwardContractError(
                "FORWARD_DERIVATION_RANGE_POSTIMAGE_INVALID"
            )
        try:
            consumed = module.commit_forward_derivation_range(
                claim_capability=claim_capability,
                postimage_capability=postimage_capability,
            )
        except BaseException as error:
            raise ForwardContractError(
                "FORWARD_DERIVATION_RANGE_COMMIT_FAILED",
                type(error).__name__,
            ) from None
        exact_consumption_fields = {
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
        expected_consumption = {
            "schema_version": (
                "c28f_a4_forward_derivation_range_consumption_receipt_v1"
            ),
            "status": "FSYNCED_DERIVATION_RANGE_CONSUMED",
            "goal_id": consumer["goal_id"],
            "attempt_id": consumer["attempt_id"],
            "run_id": consumer["run_id"],
            "token_id": consumer["token_id"],
            "eval_id": consumer["eval_id"],
            "completed_transaction_id": consumer[
                "completed_transaction_id"
            ],
            "consumer_capability_sha256": consumer_capability.capability_sha256,
            "consumer_binding_sha256": consumer["consumer_binding_sha256"],
            "claim_capability_sha256": claim_capability.capability_sha256,
            "claim_binding_sha256": claim["claim_binding_sha256"],
            "derived_range_postimage_capability_sha256": (
                postimage_capability.capability_sha256
            ),
            "derived_range_postimage_binding_sha256": postimage[
                "postimage_binding_sha256"
            ],
            **expected_range,
            "previous_range_receipt_sha256": (
                self._previous_range_receipt_sha256
            ),
            "previous_cursor_index": range_index,
            "next_cursor_index": range_index + 1,
            "all_ranges_consumed": range_index + 1 == len(self._chunk_index),
            "materialization_owner_generation": postimage[
                "materialization_owner_generation"
            ],
            "materialization_owner_binding_sha256": postimage[
                "materialization_owner_binding_sha256"
            ],
            "materialization_owner_takeover_chain_sha256": postimage[
                "materialization_owner_takeover_chain_sha256"
            ],
            "commit_owner_generation": delivery_owner["generation"],
            "commit_owner_binding_sha256": delivery_owner[
                "owner_binding_sha256"
            ],
            "commit_owner_takeover_chain_sha256": delivery_owner[
                "takeover_chain_sha256"
            ],
        }
        commit_owner_binding_chain = (
            consumed.get("commit_owner_binding_chain")
            if isinstance(consumed, Mapping)
            else None
        )
        materialization_owner_generation = postimage[
            "materialization_owner_generation"
        ]
        if (
            not isinstance(consumed, Mapping)
            or set(consumed) != exact_consumption_fields
            or any(
                consumed.get(field_name) != expected_value
                for field_name, expected_value in expected_consumption.items()
            )
            or not isinstance(commit_owner_binding_chain, list)
            or len(commit_owner_binding_chain)
            != delivery_owner["generation"] + 1
            or any(
                not isinstance(item, str)
                or len(item) != 64
                or any(character not in _SHA256_HEX for character in item)
                for item in commit_owner_binding_chain
            )
            or commit_owner_binding_chain[-1]
            != delivery_owner["owner_binding_sha256"]
            or _sha256_bytes(
                _canonical_json_bytes(commit_owner_binding_chain[:-1])
            )
            != delivery_owner["takeover_chain_sha256"]
            or commit_owner_binding_chain[materialization_owner_generation]
            != postimage["materialization_owner_binding_sha256"]
            or _sha256_bytes(
                _canonical_json_bytes(
                    commit_owner_binding_chain[
                        :materialization_owner_generation
                    ]
                )
            )
            != postimage[
                "materialization_owner_takeover_chain_sha256"
            ]
            or consumed.get("commit_owner_binding_chain_sha256")
            != _sha256_bytes(
                _canonical_json_bytes(commit_owner_binding_chain)
            )
            or consumed.get("receipt_sha256")
            != _semantic_sha256(consumed, ("receipt_sha256",))
        ):
            raise ForwardContractError(
                "FORWARD_DERIVATION_RANGE_CONSUMPTION_RECEIPT_INVALID"
            )
        object.__setattr__(self, "_next_chunk_index", range_index + 1)
        object.__setattr__(
            self,
            "_previous_range_receipt_sha256",
            consumed["receipt_sha256"],
        )
        return MappingProxyType(
            json.loads(
                _canonical_json_bytes(dict(consumed)).decode(
                    "utf-8",
                    "strict",
                )
            )
        )


def validate_completed_forward_artifact_handle(
    value: Any,
) -> Mapping[str, Any]:
    """Accept only a handle issued by the reviewed completed-run validator."""

    if (
        type(value) is not CompletedForwardRunArtifactHandle
        or value._factory is not _COMPLETED_FORWARD_ARTIFACT_HANDLE_FACTORY
    ):
        raise ForwardContractError(
            "EXACT_ISSUED_COMPLETED_FORWARD_ARTIFACT_HANDLE_REQUIRED"
        )
    with _COMPLETED_FORWARD_ARTIFACT_HANDLE_LOCK:
        record = _COMPLETED_FORWARD_ARTIFACT_HANDLE_REGISTRY.get(id(value))
        if (
            not isinstance(record, Mapping)
            or record.get("object") is not value
            or record.get("authority") is not value._authority
            or record.get("capability") is not value._capability
            or record.get("run_root") is not value._run_root
            or record.get("chunk_root") is not value._chunk_root
        ):
            raise ForwardContractError(
                "COMPLETED_FORWARD_ARTIFACT_HANDLE_NOT_REGISTERED"
            )
        expected_validation_sha256 = record.get(
            "validation_receipt_sha256"
        )
    value._assert_runtime_integrity()
    receipt = value.validation_receipt()
    if receipt.get("validation_receipt_sha256") != expected_validation_sha256:
        raise ForwardContractError(
            "COMPLETED_FORWARD_ARTIFACT_HANDLE_REGISTRY_DRIFT"
        )
    return receipt


def completed_forward_static_identity_projection(
    value: Any,
) -> Mapping[str, Any]:
    """Return only immutable query/corpus identities from a completed handle.

    The projection deliberately carries no checkpoint handle, feature source,
    raw tensor, proposal, score, query text, or ground truth.  It exists so the
    controller can derive post-forward bucket statistics (for example the GT
    video's duration-derived clip count) without reopening VIDEO_META or any
    feature source.  Both original and rehydrated completed handles expose the
    same frozen manifest identity through their verification-only authority.
    """

    validation = validate_completed_forward_artifact_handle(value)
    authority = value._authority
    manifest = authority.manifest
    query_identity = manifest.query_identity
    corpus_identity = manifest.corpus_identity
    query_keys = [int(item) for item in query_identity.query_keys]
    query_types = list(query_identity.query_types)
    video_ids = list(corpus_identity.video_ids)
    durations_sec = [float(item) for item in corpus_identity.durations_sec]
    if (
        query_keys != sorted(set(query_keys))
        or len(query_keys) != validation["query_count"]
        or len(query_types) != len(query_keys)
        or _line_sha256([str(item) for item in query_keys])
        != query_identity.query_keys_sha256
        or _line_sha256(query_types) != validation["query_types_sha256"]
        or len(video_ids) != EXPECTED_CORPUS_VIDEO_COUNT
        or len(set(video_ids)) != len(video_ids)
        or len(durations_sec) != len(video_ids)
        or any(not math.isfinite(item) or item <= 0.0 for item in durations_sec)
    ):
        raise ForwardContractError(
            "COMPLETED_FORWARD_STATIC_IDENTITY_PROJECTION_DRIFT"
        )
    base = {
        "schema_version": (
            "c28f_a4_completed_forward_static_identity_projection_v1"
        ),
        "status": "COMPLETED_VERIFICATION_ONLY_STATIC_IDENTITIES",
        "goal_id": validation["goal_id"],
        "attempt_id": validation["attempt_id"],
        "run_id": validation["run_id"],
        "token_id": validation["token_id"],
        "eval_id": validation["eval_id"],
        "forward_manifest_sha256": validation["forward_manifest_sha256"],
        "query_identity_sha256": validation["query_identity_sha256"],
        "query_keys": query_keys,
        "query_types": query_types,
        "query_rows_sha256": validation["query_rows_sha256"],
        "corpus_identity_sha256": corpus_identity.identity_sha256,
        "video_ids": video_ids,
        "durations_sec": durations_sec,
        "video_ids_sha256": corpus_identity.video_ids_sha256,
        "durations_sha256": corpus_identity.durations_sha256,
        "raw_tensor_materialized": False,
        "active_source_reopened": False,
        "ground_truth_present": False,
    }
    projection = {
        **base,
        "projection_binding_sha256": _semantic_sha256(base),
    }
    return MappingProxyType(
        json.loads(_canonical_json_bytes(projection).decode("utf-8", "strict"))
    )


def validate_completed_forward_run_artifacts(
    authority: FrozenInputAuthority,
    capability: InputCapability,
    *,
    run_root: FrozenDirectory,
    chunk_root: FrozenDirectory,
) -> CompletedForwardRunArtifactHandle:
    """Validate completed artifacts once without relying on RUNNING state."""

    if type(authority) is not FrozenInputAuthority or type(capability) is not InputCapability:
        raise ForwardContractError("EXACT_COMPLETED_FORWARD_AUTHORITY_REQUIRED")
    authority.verify_code_sources()
    _require_exact_forward_control(capability._control)
    _validate_bound_completed_forward_roots(
        authority=authority,
        capability=capability,
        run_root=run_root,
        chunk_root=chunk_root,
    )
    authority, capability, rehydrate_projection = (
        _issue_original_completed_result_verifier(
            authority,
            capability,
            run_root=run_root,
            chunk_root=chunk_root,
        )
    )
    receipt_path = Path(run_root.path) / "forward_run_receipt.json"
    receipt = _read_canonical_json(
        receipt_path,
        expected_parent=run_root,
    )
    if (
        receipt.get("status") != "COMPLETED"
        or receipt.get("receipt_sha256") != capability._run_receipt_sha256
    ):
        raise ForwardContractError("EXACT_COMPLETED_RUN_RECEIPT_REQUIRED")
    _validate_run_receipt_contents(authority, capability, receipt)
    completion_summary = _validate_completed_forward_phase_receipt(
        capability=capability,
        run_root=run_root,
    )
    observation_summary = _validate_completed_forward_observation_artifacts(
        capability=capability,
        run_root=run_root,
        receipt=receipt,
    )
    chunks = receipt["result_chunks"]
    if not chunks:
        raise ForwardContractError("COMPLETED_FORWARD_RESULT_CHUNKS_EMPTY")
    expected_artifact_names = {str(chunk["artifact_name"]) for chunk in chunks}
    chunk_directory_fd = _open_directory_descriptor(chunk_root.path, chunk_root)
    try:
        observed_artifact_names = set(os.listdir(chunk_directory_fd))
    finally:
        os.close(chunk_directory_fd)
    if observed_artifact_names != expected_artifact_names:
        raise ForwardContractError("COMPLETED_FORWARD_CHUNK_NAMESPACE_NOT_EXACT")
    torch_module = _lazy_torch()
    safe_chunk_index = []
    ordered_ranges = []
    expected_start = 0
    for sequence, chunk in enumerate(chunks):
        start, end = _validate_completed_forward_chunk_range(
            chunk,
            sequence=sequence,
            expected_start=expected_start,
            query_count=len(authority.manifest.query_identity.query_keys),
        )
        artifact_path = Path(chunk_root.path) / chunk["artifact_name"]
        artifact_bytes = _read_bytes_no_follow(
            artifact_path,
            max_bytes=MAX_OWNED_TORCH_ARCHIVE_BYTES,
            expected_parent=chunk_root,
        )
        artifact_sha256 = _sha256_bytes(artifact_bytes)
        if artifact_sha256 != chunk.get("artifact_sha256"):
            raise ForwardContractError(
                "COMPLETED_FORWARD_CHUNK_FILE_HASH_DRIFT",
                str(sequence),
            )
        del artifact_bytes
        payload = _safe_load_owned_torch_artifact(
            artifact_path,
            torch_module,
            expected_parent=chunk_root,
            expected_file_sha256=artifact_sha256,
        )
        if (
            not isinstance(payload, Mapping)
            or payload.get("engine_registry_consumption_sha256")
            != chunk.get("engine_registry_consumption_sha256")
            or payload.get("query_types_sha256")
            != chunk.get("query_types_sha256")
            or payload.get("query_rows_sha256")
            != chunk.get("query_rows_sha256")
        ):
            raise ForwardContractError(
                "COMPLETED_FORWARD_CHUNK_LINEAGE_DRIFT",
                str(sequence),
            )
        _validate_forward_result_payload(
            torch_module,
            payload,
            authority=authority,
            capability=capability,
            expected_start=start,
            expected_end=end,
        )
        semantic_sha256 = _forward_result_payload_digest(torch_module, payload)
        if semantic_sha256 != chunk.get("semantic_sha256"):
            raise ForwardContractError(
                "COMPLETED_FORWARD_CHUNK_SEMANTIC_DRIFT",
                str(sequence),
            )
        safe_chunk_index.append(
            {
                "range_index": sequence,
                "start_index": start,
                "end_index": end,
                "row_count": end - start,
                "artifact_name": chunk["artifact_name"],
                "artifact_file_sha256": artifact_sha256,
                "payload_semantic_sha256": semantic_sha256,
                "engine_registry_consumption_sha256": chunk[
                    "engine_registry_consumption_sha256"
                ],
                "query_types_sha256": chunk["query_types_sha256"],
                "query_rows_sha256": chunk["query_rows_sha256"],
            }
        )
        ordered_ranges.append(
            {
                "range_index": sequence,
                "start_index": start,
                "end_index": end,
                "artifact_name": chunk["artifact_name"],
                "artifact_file_sha256": artifact_sha256,
                "payload_semantic_sha256": semantic_sha256,
            }
        )
        expected_start = end
        del payload
    query_count = len(authority.manifest.query_identity.query_keys)
    if expected_start != query_count:
        raise ForwardContractError("COMPLETED_FORWARD_CHUNK_RANGE_NOT_EXHAUSTIVE")
    result_chunk_set_sha256 = _sha256_bytes(
        _canonical_json_bytes(safe_chunk_index)
    )
    ordered_ranges_sha256 = _sha256_bytes(
        _canonical_json_bytes(ordered_ranges)
    )
    observation_summary = {
        **observation_summary,
        "summary_sha256": _semantic_sha256(observation_summary),
    }
    base = {
        "schema_version": "c28f_a4_completed_forward_artifact_validation_v1",
        "status": "VALID_COMPLETED_FORWARD_RUN_ARTIFACTS",
        "goal_id": capability.goal_id,
        "attempt_id": authority.manifest.attempt_id,
        "authority_id": capability.authority_id,
        "run_id": capability.run_id,
        "token_id": capability.token_id,
        "eval_id": capability.eval_id,
        "stage": capability.stage,
        "purpose": capability.purpose,
        "capability_id": capability.capability_id,
        "forward_manifest_sha256": capability.forward_manifest_sha256,
        "input_identity_sha256": capability.input_identity_sha256,
        "query_identity_sha256": authority.manifest.query_identity.identity_sha256,
        "query_types_sha256": authority.manifest.query_identity.query_types_sha256,
        "query_rows_sha256": authority.manifest.query_identity.query_rows_sha256,
        "query_count": query_count,
        "validated_start_index": 0,
        "validated_end_index": query_count,
        "run_root_identity_sha256": _sha256_bytes(
            _canonical_json_bytes(run_root.as_dict())
        ),
        "chunk_root_identity_sha256": _sha256_bytes(
            _canonical_json_bytes(chunk_root.as_dict())
        ),
        "forward_run_receipt_artifact_name": "forward_run_receipt.json",
        "forward_run_receipt_sha256": receipt["receipt_sha256"],
        "forward_run_receipt_file_sha256": _sha256_bytes(
            _canonical_json_bytes(receipt)
        ),
        "encoded_cache_binding": {
            "artifact_name": "encoded_corpus_cache_receipt",
            "artifact_sha256": receipt["encoded_cache_receipt_sha256"],
            "semantic_sha256": receipt["encoded_cache_fingerprint"],
        },
        "completion_summary": completion_summary,
        "observation_summary": observation_summary,
        "result_chunk_count": len(safe_chunk_index),
        "result_chunk_index": safe_chunk_index,
        "result_chunk_set_sha256": result_chunk_set_sha256,
        "ordered_ranges": ordered_ranges,
        "ordered_ranges_sha256": ordered_ranges_sha256,
        "maximum_simultaneously_loaded_chunk_count": 1,
        "payload_retention_contract": (
            "HANDLE_STORES_NO_RAW_PAYLOAD_STREAM_REVALIDATES_ONE_CHUNK"
        ),
        "payload_transfer_status": (
            "OPAQUE_EXACT_CONTROLLER_DERIVATION_CONSUMER_REQUIRED"
        ),
    }
    validation_receipt = {
        **base,
        "validation_receipt_sha256": _semantic_sha256(base),
    }
    input_capability_material = {
        field_name: getattr(capability, field_name)
        for field_name in InputCapability.__slots__
        if not field_name.startswith("_")
    }
    try:
        persisted_rehydrate_binding = (
            capability._control.persist_completed_forward_rehydrate_material(
                completed_result_verification_projection=(
                    rehydrate_projection
                ),
                input_capability_material=input_capability_material,
                run_root=run_root.as_dict(),
                chunk_root=chunk_root.as_dict(),
                completed_forward_validation_receipt=(
                    validation_receipt
                ),
            )
        )
    except BaseException as error:
        raise ForwardContractError(
            "COMPLETED_FORWARD_REHYDRATE_MATERIAL_PERSISTENCE_FAILED",
            type(error).__name__,
        ) from None
    if (
        not isinstance(persisted_rehydrate_binding, Mapping)
        or not persisted_rehydrate_binding
    ):
        raise ForwardContractError(
            "COMPLETED_FORWARD_REHYDRATE_MATERIAL_PERSISTENCE_INVALID"
        )
    handle = CompletedForwardRunArtifactHandle(
        _COMPLETED_FORWARD_ARTIFACT_HANDLE_FACTORY,
        _factory=_COMPLETED_FORWARD_ARTIFACT_HANDLE_FACTORY,
        _authority=authority,
        _capability=capability,
        _control=capability._control,
        _run_root=run_root,
        _chunk_root=chunk_root,
        _chunk_index=tuple(
            MappingProxyType(dict(chunk)) for chunk in safe_chunk_index
        ),
        _next_chunk_index=0,
        _previous_range_receipt_sha256=None,
        _validation_receipt=MappingProxyType(validation_receipt),
    )
    _register_completed_result_verifier(authority, capability)
    try:
        handle._assert_runtime_integrity()
        with _COMPLETED_FORWARD_ARTIFACT_HANDLE_LOCK:
            if id(handle) in _COMPLETED_FORWARD_ARTIFACT_HANDLE_REGISTRY:
                raise ForwardContractError(
                    "COMPLETED_FORWARD_ARTIFACT_HANDLE_ID_COLLISION"
                )
            _COMPLETED_FORWARD_ARTIFACT_HANDLE_REGISTRY[id(handle)] = {
                "object": handle,
                "authority": authority,
                "capability": capability,
                "run_root": run_root,
                "chunk_root": chunk_root,
                "validation_receipt_sha256": validation_receipt[
                    "validation_receipt_sha256"
                ],
            }
    except BaseException:
        _rollback_completed_result_verifier_registration(
            authority,
            capability,
        )
        raise
    return handle


def recover_completed_forward_run_artifact_handle_from_control(
    *,
    authority: FrozenInputAuthority,
    control: Any,
    running_capability: Any,
) -> CompletedForwardRunArtifactHandle:
    """Close the post-COMPLETE/pre-validation process-death window.

    Only persisted descriptors, the historical opaque RUNNING proof, and the
    already committed forward artifacts are used.  Active input sources and
    the eval token cannot be reopened or reissued by this path.
    """

    if type(authority) is not FrozenInputAuthority:
        raise ForwardContractError(
            "EXACT_RECOVERED_FORWARD_AUTHORITY_REQUIRED"
        )
    module = _require_exact_forward_control(control)
    running = module.validate_eval_running_capability(running_capability)
    controller_binding = _validate_controller_issued_authority(
        authority,
        control,
    )
    run_id = _require_text(running.get("run_id"), "run_id")
    run_root = _freeze_existing_child_directory(
        authority.output_root,
        (authority.manifest.stage, authority.manifest.attempt_id, run_id),
    )
    chunk_root = _freeze_existing_child_directory(
        run_root,
        ("query_chunks",),
    )
    receipt_path = Path(run_root.path) / "forward_run_receipt.json"
    receipt = _read_canonical_json(
        receipt_path,
        expected_parent=run_root,
    )
    receipt_sha256 = _require_sha256(
        receipt.get("receipt_sha256"),
        "forward_run_receipt_sha256",
    )
    if (
        receipt.get("status") != "COMPLETED"
        or receipt_sha256
        != _semantic_sha256(receipt, ("receipt_sha256",))
    ):
        raise ForwardContractError(
            "RECOVERED_FORWARD_RUN_RECEIPT_NOT_COMPLETED"
        )
    material_record = module.completed_forward_input_rehydrate_material(
        running_capability=running_capability,
        authority_binding_sha256=authority.authority_binding_sha256,
        forward_manifest_sha256=authority.manifest.manifest_sha256,
        input_identity_sha256=authority.manifest.input_identity_sha256,
        reservation_input_capability_sha256=controller_binding[
            "reservation_input_capability_sha256"
        ],
        controller_authority_binding_sha256=controller_binding[
            "controller_authority_binding_sha256"
        ],
        forward_run_receipt_sha256=receipt_sha256,
    )
    exact_record_fields = {
        "schema_version",
        "status",
        "input_capability_material",
        "completed_lifecycle_binding",
        "completed_phase_receipt",
        "active_source_reopened",
        "token_reissued",
        "model_forward_reexecuted",
        "material_binding_sha256",
    }
    if (
        not isinstance(material_record, Mapping)
        or set(material_record) != exact_record_fields
        or material_record.get("schema_version")
        != "c28f_a4_completed_forward_input_rehydrate_material_v1"
        or material_record.get("status")
        != "VALIDATED_COMPLETED_INPUT_IDENTITY_ONLY"
        or any(
            material_record.get(field_name) is not False
            for field_name in (
                "active_source_reopened",
                "token_reissued",
                "model_forward_reexecuted",
            )
        )
        or material_record.get("material_binding_sha256")
        != _semantic_sha256(
            material_record,
            ("material_binding_sha256",),
        )
    ):
        raise ForwardContractError(
            "COMPLETED_FORWARD_INPUT_REHYDRATE_MATERIAL_INVALID"
        )
    material = material_record.get("input_capability_material")
    exact_material_fields = {
        "capability_id",
        "goal_id",
        "authority_id",
        "run_id",
        "token_id",
        "eval_id",
        "purpose",
        "stage",
        "forward_manifest_sha256",
        "input_identity_sha256",
        "authority_binding_sha256",
        "token_snapshot_sha256",
        "gpu0_identity_sha256",
        "control_chain_sha256",
        "running_commit_sha256",
        "running_capability_sha256",
        "reservation_input_capability_sha256",
        "controller_authority_binding_sha256",
        "consume_receipt_sha256",
    }
    if not isinstance(material, Mapping) or set(material) != exact_material_fields:
        raise ForwardContractError(
            "COMPLETED_FORWARD_INPUT_REHYDRATE_FIELDS_INVALID"
        )
    capability_preimage = {
        field_name: field_value
        for field_name, field_value in material.items()
        if field_name != "capability_id"
    }
    expected_material = {
        "goal_id": authority.manifest.goal_id,
        "authority_id": authority.manifest.authority_id,
        "run_id": run_id,
        "token_id": running["token_id"],
        "eval_id": running["eval_id"],
        "purpose": authority.manifest.purpose,
        "stage": authority.manifest.stage,
        "forward_manifest_sha256": authority.manifest.manifest_sha256,
        "input_identity_sha256": authority.manifest.input_identity_sha256,
        "authority_binding_sha256": authority.authority_binding_sha256,
        "gpu0_identity_sha256": authority.manifest.gpu0_identity.semantic_sha256,
        "running_capability_sha256": running_capability.capability_sha256,
        "reservation_input_capability_sha256": controller_binding[
            "reservation_input_capability_sha256"
        ],
        "controller_authority_binding_sha256": controller_binding[
            "controller_authority_binding_sha256"
        ],
    }
    if (
        any(
            material.get(field_name) != expected_value
            for field_name, expected_value in expected_material.items()
        )
        or material.get("capability_id")
        != _sha256_bytes(_canonical_json_bytes(capability_preimage))
    ):
        raise ForwardContractError(
            "COMPLETED_FORWARD_INPUT_REHYDRATE_BINDING_DRIFT"
        )
    public_material = {
        field_name: field_value
        for field_name, field_value in material.items()
        if field_name != "consume_receipt_sha256"
    }
    capability = InputCapability(
        _CAPABILITY_FACTORY,
        _issuer_nonce=authority._issuer_nonce,
        _authority=authority,
        _control=control,
        _running_control_capability=running_capability,
        _run_receipt_sha256=receipt_sha256,
        _run_receipt_path=str(receipt_path),
        _run_receipt_parent=run_root,
        _run_chunk_parent=chunk_root,
        **public_material,
    )
    lifecycle = material_record.get("completed_lifecycle_binding")
    if (
        not isinstance(lifecycle, Mapping)
        or lifecycle.get("capability_id") != capability.capability_id
        or lifecycle.get("forward_run_receipt_sha256") != receipt_sha256
        or lifecycle.get("binding_sha256")
        != _semantic_sha256(lifecycle, ("binding_sha256",))
    ):
        raise ForwardContractError(
            "COMPLETED_FORWARD_INPUT_REHYDRATE_LIFECYCLE_DRIFT"
        )
    completed_phase_receipt = material_record.get("completed_phase_receipt")
    if (
        not isinstance(completed_phase_receipt, Mapping)
        or completed_phase_receipt.get("transaction_id")
        != lifecycle.get("completed_transaction_id")
        or completed_phase_receipt.get("state_sha256")
        != lifecycle.get("completed_state_sha256")
        or completed_phase_receipt.get("event_sha256")
        != lifecycle.get("completed_event_sha256")
    ):
        raise ForwardContractError(
            "COMPLETED_FORWARD_PHASE_RECEIPT_REHYDRATE_DRIFT"
        )
    _validate_completed_forward_phase_receipt_value(
        capability=capability,
        value=completed_phase_receipt,
    )
    completed_phase_receipt_path = (
        Path(run_root.path) / "control_completion_receipt.json"
    )
    if os.path.lexists(completed_phase_receipt_path):
        persisted_completed_phase_receipt = _read_canonical_json(
            completed_phase_receipt_path,
            expected_parent=run_root,
        )
        if persisted_completed_phase_receipt != completed_phase_receipt:
            raise ForwardContractError(
                "COMPLETED_FORWARD_PHASE_RECEIPT_RECOVERY_DRIFT"
            )
    else:
        _atomic_write_bytes(
            completed_phase_receipt_path,
            _canonical_json_bytes(completed_phase_receipt),
            replace=False,
            expected_parent=run_root,
        )
    if authority._capability_issued:
        raise ForwardContractError(
            "RECOVERED_FORWARD_AUTHORITY_CAPABILITY_ALREADY_ISSUED"
        )
    authority._capability_issued = True
    authority._issued_capability_id = capability.capability_id
    authority._issued_capability = capability
    authority._issued_control = control
    authority._issued_capability_fields = dict(public_material)
    return validate_completed_forward_run_artifacts(
        authority,
        capability,
        run_root=run_root,
        chunk_root=chunk_root,
    )


def rehydrate_completed_forward_run_artifact_handle(
    *,
    control: Any,
    rehydrate_capability: Any,
) -> CompletedForwardRunArtifactHandle:
    """Rebuild a completed-only handle without token, source, or index replay."""

    module = _require_exact_forward_control(control)
    binding = _validate_completed_forward_rehydrate_binding(
        control,
        rehydrate_capability,
    )
    try:
        material = module.consume_completed_forward_rehydrate_material_once(
            rehydrate_capability
        )
    except BaseException as error:
        raise ForwardContractError(
            "COMPLETED_FORWARD_REHYDRATE_MATERIAL_CONSUMPTION_FAILED",
            type(error).__name__,
        ) from None
    exact_material_fields = {
        "completed_result_verification_projection",
        "input_capability_material",
        "run_root",
        "chunk_root",
        "completed_forward_validation_receipt",
        "authority_rehydrate_projection_sha256",
        "material_binding_sha256",
    }
    if not isinstance(material, Mapping) or set(material) != exact_material_fields:
        raise ForwardContractError(
            "COMPLETED_FORWARD_REHYDRATE_MATERIAL_FIELDS_INVALID"
        )
    projection = material.get("completed_result_verification_projection")
    if (
        not isinstance(projection, Mapping)
        or projection.get("projection_binding_sha256")
        != binding["authority_rehydrate_projection_sha256"]
        or material.get("authority_rehydrate_projection_sha256")
        != binding["authority_rehydrate_projection_sha256"]
    ):
        raise ForwardContractError(
            "COMPLETED_RESULT_VERIFICATION_PROJECTION_BINDING_DRIFT"
        )
    authority = _rehydrate_completed_forward_authority_from_projection(
        projection
    )
    if (
        authority.authority_binding_sha256
        != binding["authority_binding_sha256"]
        or authority.forward_authority_handle_sha256
        != binding["forward_authority_handle_sha256"]
        or authority.projection_binding_sha256
        != binding["authority_rehydrate_projection_sha256"]
    ):
        raise ForwardContractError(
            "COMPLETED_RESULT_VERIFICATION_AUTHORITY_BINDING_DRIFT"
        )
    input_material = _validate_rehydrated_input_capability_material(
        material.get("input_capability_material"),
        binding=binding,
        authority=authority,
    )
    run_root_view = material.get("run_root")
    chunk_root_view = material.get("chunk_root")
    if (
        not isinstance(run_root_view, Mapping)
        or not isinstance(chunk_root_view, Mapping)
    ):
        raise ForwardContractError(
            "COMPLETED_FORWARD_REHYDRATE_ROOT_MATERIAL_INVALID"
        )
    run_root = FrozenDirectory(**dict(run_root_view))
    chunk_root = FrozenDirectory(**dict(chunk_root_view))
    run_root_identity_sha256 = _sha256_bytes(
        _canonical_json_bytes(run_root.as_dict())
    )
    chunk_root_identity_sha256 = _sha256_bytes(
        _canonical_json_bytes(chunk_root.as_dict())
    )
    if (
        run_root_identity_sha256 != binding["run_root_identity_sha256"]
        or chunk_root_identity_sha256
        != binding["chunk_root_identity_sha256"]
    ):
        raise ForwardContractError(
            "COMPLETED_FORWARD_REHYDRATE_ROOT_BINDING_DRIFT"
        )
    material_binding_base = {
        "schema_version": (
            "c28f_a4_completed_forward_rehydrate_material_binding_v1"
        ),
        "rehydrate_binding_sha256": binding["rehydrate_binding_sha256"],
        "authority_rehydrate_projection_sha256": binding[
            "authority_rehydrate_projection_sha256"
        ],
        "input_capability_material_sha256": binding[
            "input_capability_material_sha256"
        ],
        "run_root_identity_sha256": run_root_identity_sha256,
        "chunk_root_identity_sha256": chunk_root_identity_sha256,
        "completed_forward_validation_receipt_sha256": binding[
            "completed_forward_validation_receipt_sha256"
        ],
    }
    if material.get("material_binding_sha256") != _semantic_sha256(
        material_binding_base
    ):
        raise ForwardContractError(
            "COMPLETED_FORWARD_REHYDRATE_MATERIAL_BINDING_DRIFT"
        )
    public_values = {
        field_name: input_material[field_name]
        for field_name in InputCapability.__slots__
        if not field_name.startswith("_")
    }
    capability = CompletedForwardResultCapability(
        _COMPLETED_RESULT_CAPABILITY_FACTORY,
        _factory=_COMPLETED_RESULT_CAPABILITY_FACTORY,
        **public_values,
        _authority=authority,
        _control=control,
        _proof_capability=rehydrate_capability,
        _proof_mode="REHYDRATED_COMPLETED_SAFE_INDEX",
        _input_capability_material_sha256=binding[
            "input_capability_material_sha256"
        ],
        _run_receipt_sha256=binding["forward_run_receipt_sha256"],
        _run_receipt_path=os.path.join(
            run_root.path,
            "forward_run_receipt.json",
        ),
        _run_receipt_parent=run_root,
        _run_chunk_parent=chunk_root,
    )
    capability._assert_completed_lifecycle()
    _validate_bound_completed_forward_roots(
        authority=authority,
        capability=capability,
        run_root=run_root,
        chunk_root=chunk_root,
    )
    validation_receipt = (
        _validate_rehydrated_completed_forward_validation_receipt(
            material.get("completed_forward_validation_receipt"),
            authority=authority,
            capability=capability,
            run_root=run_root,
            chunk_root=chunk_root,
            rehydrate_binding=binding,
        )
    )
    safe_chunk_index = validation_receipt["result_chunk_index"]
    handle = CompletedForwardRunArtifactHandle(
        _COMPLETED_FORWARD_ARTIFACT_HANDLE_FACTORY,
        _factory=_COMPLETED_FORWARD_ARTIFACT_HANDLE_FACTORY,
        _authority=authority,
        _capability=capability,
        _control=control,
        _run_root=run_root,
        _chunk_root=chunk_root,
        _chunk_index=tuple(
            MappingProxyType(dict(chunk)) for chunk in safe_chunk_index
        ),
        _next_chunk_index=0,
        _previous_range_receipt_sha256=None,
        _validation_receipt=validation_receipt,
    )
    _register_completed_result_verifier(authority, capability)
    try:
        handle._assert_runtime_integrity()
        with _COMPLETED_FORWARD_ARTIFACT_HANDLE_LOCK:
            if id(handle) in _COMPLETED_FORWARD_ARTIFACT_HANDLE_REGISTRY:
                raise ForwardContractError(
                    "COMPLETED_FORWARD_ARTIFACT_HANDLE_ID_COLLISION"
                )
            _COMPLETED_FORWARD_ARTIFACT_HANDLE_REGISTRY[id(handle)] = {
                "object": handle,
                "authority": authority,
                "capability": capability,
                "run_root": run_root,
                "chunk_root": chunk_root,
                "validation_receipt_sha256": validation_receipt[
                    "validation_receipt_sha256"
                ],
            }
    except BaseException:
        _rollback_completed_result_verifier_registration(
            authority,
            capability,
        )
        raise
    return handle


__all__ = [
    "CompletedForwardRunArtifactHandle",
    "CompletedForwardResultCapability",
    "CompletedForwardResultVerificationAuthority",
    "ENCODED_CACHE_SCHEMA_VERSION",
    "EncodedCorpusCache",
    "EncodedCorpusHandle",
    "EXPECTED_CORPUS_VIDEO_COUNT",
    "EXPECTED_F0A_QUERY_COUNT",
    "FORWARD_SCHEMA_VERSION",
    "ForwardContractError",
    "ForwardQueryResult",
    "ForwardRunJournal",
    "FrozenCorpusIdentity",
    "FrozenDirectory",
    "FrozenFeatureContract",
    "FrozenFile",
    "FrozenForwardEngine",
    "FrozenForwardManifest",
    "FrozenGpu0Identity",
    "FrozenInputAuthority",
    "FrozenLmdbSource",
    "FrozenModelContract",
    "FrozenQueryIdentity",
    "FrozenTemporalManifest",
    "GuardedLmdbReader",
    "InputCapability",
    "LoadedModelHandle",
    "ValidatedForwardDerivationRangeTransfer",
    "advance_run_receipt",
    "grid_spans_for_duration",
    "initial_run_receipt",
    "load_frozen_model",
    "normalize_forward_queries",
    "consume_validated_forward_derivation_range_transfer_once",
    "completed_forward_static_identity_projection",
    "build_active_forward_authority_rehydrate_projection",
    "rehydrate_active_forward_authority",
    "rehydrate_completed_forward_run_artifact_handle",
    "recover_completed_forward_run_artifact_handle_from_control",
    "stable_topk_indices",
    "validate_completed_forward_run_artifacts",
    "validate_completed_forward_artifact_handle",
    "validate_run_receipt",
]
