"""Fail-closed C28F F0/F1 security primitives.

This module deliberately contains no data loader, evaluator, model, LMDB, or
artifact writer.  It authorizes *metadata intents* before any such component is
allowed to open content.  The caller must persist the returned ledger rows and
must never treat a pre-open grant as evidence that content was opened.

The implementation is Linux-specific by design: descriptor walking requires
``O_PATH`` and ``O_NOFOLLOW`` so a lexical check cannot be separated from the
inode that was checked.  Unsupported platforms fail closed.
"""

from __future__ import annotations

import errno
import hashlib
import hmac
import json
import os
import re
import stat
import sys
import time
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Optional, Sequence, Tuple


SECURITY_SCHEMA_VERSION = "c28f_v5_a4_security_v1"
LEDGER_SCHEMA_VERSION = "c28f_v5_a4_evidence_ledger_v1"
SAFETY_MANIFEST_SCHEMA_VERSION = "c28f_v5_a4_ledger_safety_manifest_v1"
ID_MAP_SCHEMA_VERSION = "c28f_v5_protected_id_map_minimal_v1"
MOUNT_DOMAIN_SCHEMA_VERSION = "c28f_linux_mount_domain_snapshot_v1"
CONTENT_OPEN_INTENT_SCHEMA_VERSION = "c28f_persisted_content_open_intent_v1"
EVAL_TOKEN_CAS_INTENT_SCHEMA_VERSION = "c28f_eval_token_cas_intent_v1"
EVAL_TOKEN_CAS_RECEIPT_SCHEMA_VERSION = "c28f_eval_token_cas_receipt_v1"
LMDB_BATCH_RECEIPT_SCHEMA_VERSION = "c28f_lmdb_batch_persisted_receipt_v1"
PROTECTED_ID_MAPPING_RUNNING_ANCHOR_SCHEMA_VERSION = (
    "c28f_a4_protected_id_mapping_running_anchor_v1"
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SAFE_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
_DECIMAL_ID_RE = re.compile(r"^(?:0|[1-9][0-9]*)$")
_VALID_DECISIONS = frozenset({"ALLOW", "DENY"})
_VALID_CLASSIFICATIONS = frozenset(
    {
        "CONTROL",
        "NON_PROTECTED_METADATA",
        "NON_PROTECTED_CONTENT",
        "PROTECTED_METADATA",
        "PROTECTED_CONTENT",
        "PROTECTED_FEATURE",
        "PROTECTED_PREDICTION",
        "PROTECTED_METRIC",
        "PROTECTED_LABEL",
        "PROTECTED_QUERY_TEXT_TIMESTAMP",
        "PROTECTED_ID_MAPPING_ONLY",
        "HISTORICAL_WRITE",
        "MODEL_FORWARD",
        "OFFICIAL_WORKFLOW",
    }
)
_PATH_OPERATIONS = frozenset(
    {"READ_METADATA", "READ_CONTENT", "WRITE_FILE", "WRITE_DIRECTORY"}
)
_ROOT_OPERATIONS = frozenset({"READ_METADATA", "READ_CONTENT", "WRITE"})
_LEGACY_MARKER_KEYS = frozenset(
    {
        "allow_holdout",
        "allow_holdout_final",
        "allow_official",
        "allow_official_eval",
        "auth_holdout",
        "auth_official",
        "official_authorized",
        "root_authorization_marker",
    }
)
_FORBIDDEN_LAUNCHER_TOKENS = frozenset(
    {
        "all",
        "holdout",
        "calib_holdout",
        "pseudo_official_holdout",
        "official",
        "formal",
        "c28e_3",
        "c28e_5",
        "--stage",
        "--allow_holdout_final",
        "--allow_official",
    }
)


class SecurityViolation(RuntimeError):
    """A public, redacted failure that is safe to place in an audit ledger."""

    def __init__(self, code: str, public_detail: str = "") -> None:
        if not isinstance(code, str) or not re.fullmatch(r"[A-Z0-9_]+", code):
            raise ValueError("security violation code must be an uppercase identifier")
        self.code = code
        self.public_detail = public_detail
        message = code if not public_detail else f"{code}: {public_detail}"
        super().__init__(message)


def _linux_o_path_flag() -> int:
    """Return Linux O_PATH even when an older CPython omits the symbol."""

    if not sys.platform.startswith("linux") or not hasattr(os, "O_NOFOLLOW"):
        raise SecurityViolation("DESCRIPTOR_WALK_PLATFORM_UNSUPPORTED")
    # O_PATH is a stable Linux UAPI flag (octal 010000000).  Python 3.9 builds
    # may omit the name even though the running Linux kernel supports it.
    return int(getattr(os, "O_PATH", 0o10000000))


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _semantic_sha256(value: Mapping[str, Any], excluded: Iterable[str] = ()) -> str:
    excluded_set = frozenset(excluded)
    return _sha256_bytes(
        _canonical_json_bytes(
            {key: item for key, item in value.items() if key not in excluded_set}
        )
    )


def _require_sha256(value: Any, field: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise SecurityViolation("INVALID_SHA256", field)
    return value


def _require_nonempty_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise SecurityViolation("MISSING_REQUIRED_FIELD", field)
    if "\x00" in value:
        raise SecurityViolation("INVALID_TEXT_FIELD", field)
    return value


def _require_nonnegative_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise SecurityViolation("INVALID_COUNTER", field)
    return value


def _require_exact_mapping(
    value: Any,
    required_fields: Iterable[str],
    code: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SecurityViolation(code)
    expected = frozenset(required_fields)
    if frozenset(value) != expected:
        raise SecurityViolation(code)
    return value


def _json_copy(value: Any) -> Any:
    """Validate JSON safety and return a detached value."""

    return json.loads(_canonical_json_bytes(value).decode("utf-8"))


def validate_protected_id_mapping_running_anchor(
    anchor: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Validate the independent ID-only operation anchor.

    This anchor is deliberately disjoint from an eval RUNNING capability: it
    has no run/token/eval identity and can never authorize a model forward.
    """

    fields = {
        "schema_version",
        "goal_id",
        "attempt_id",
        "authority_permission_id",
        "operation_id",
        "transaction_id",
        "parent_state_sha256",
        "parent_event_sha256",
        "running_state_sha256",
        "running_event_sha256",
        "descriptor_identity_sha256",
        "access_intent_sha256",
        "operation_state",
        "operations_used",
        "eval_token_consumed",
        "authority_source_file_sha256",
        "anchor_binding_sha256",
    }
    current = dict(
        _require_exact_mapping(
            anchor,
            fields,
            "PROTECTED_ID_MAPPING_RUNNING_ANCHOR_FIELDS_MISMATCH",
        )
    )
    if (
        current.get("schema_version")
        != PROTECTED_ID_MAPPING_RUNNING_ANCHOR_SCHEMA_VERSION
        or current.get("authority_permission_id")
        != "AUTH_PROTECTED_ID_MAPPING_ONLY"
        or current.get("operation_state") != "RUNNING"
        or type(current.get("operations_used")) is not int
        or current.get("operations_used") != 0
        or current.get("eval_token_consumed") is not False
    ):
        raise SecurityViolation("PROTECTED_ID_MAPPING_RUNNING_ANCHOR_STATE_INVALID")
    for field in (
        "goal_id",
        "attempt_id",
        "operation_id",
        "transaction_id",
    ):
        _require_nonempty_text(current[field], field)
    for field in (
        "parent_state_sha256",
        "parent_event_sha256",
        "running_state_sha256",
        "running_event_sha256",
        "descriptor_identity_sha256",
        "access_intent_sha256",
        "authority_source_file_sha256",
        "anchor_binding_sha256",
    ):
        _require_sha256(current[field], field)
    if current["anchor_binding_sha256"] != _semantic_sha256(
        current,
        ("anchor_binding_sha256",),
    ):
        raise SecurityViolation("PROTECTED_ID_MAPPING_RUNNING_ANCHOR_HASH_MISMATCH")
    return _json_copy(current)


def _request_fingerprint(fields: Mapping[str, Any]) -> str:
    """Hash request metadata so rejected payloads need not be copied to ledgers."""

    redacted = {
        str(key): ("<NULL>" if value is None else str(value))
        for key, value in sorted(fields.items())
    }
    return _sha256_bytes(_canonical_json_bytes(redacted))


def _canonical_absolute_path(raw_path: str) -> Tuple[str, Tuple[str, ...]]:
    if not isinstance(raw_path, str) or not raw_path:
        raise SecurityViolation("PATH_MISSING")
    if "\x00" in raw_path:
        raise SecurityViolation("PATH_NUL_FORBIDDEN")
    if not raw_path.startswith("/"):
        raise SecurityViolation("PATH_NOT_ABSOLUTE")
    if raw_path != "/" and raw_path.endswith("/"):
        raise SecurityViolation("PATH_NONCANONICAL_TRAILING_SLASH")
    if "//" in raw_path:
        raise SecurityViolation("PATH_NONCANONICAL_DOUBLE_SLASH")
    pieces = raw_path.split("/")[1:]
    if any(piece in {"", ".", ".."} for piece in pieces):
        raise SecurityViolation("PATH_TRAVERSAL_OR_DOT_COMPONENT")
    normalized = os.path.normpath(raw_path)
    if normalized != raw_path:
        raise SecurityViolation("PATH_NONCANONICAL")
    return normalized, tuple(pieces)


def _is_within(candidate: str, root: str) -> bool:
    try:
        return os.path.commonpath((candidate, root)) == root
    except ValueError:
        return False


def _mode_kind(mode: int) -> str:
    if stat.S_ISREG(mode):
        return "FILE"
    if stat.S_ISDIR(mode):
        return "DIRECTORY"
    if stat.S_ISLNK(mode):
        return "SYMLINK"
    return "OTHER"


def _fd_mount_id(fd: int) -> int:
    """Read Linux fdinfo mount identity without consulting target content."""

    flags = os.O_RDONLY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    info_fd = os.open(f"/proc/self/fdinfo/{fd}", flags)
    try:
        chunks: list[bytes] = []
        total = 0
        while total <= 16_384:
            chunk = os.read(info_fd, min(4_096, 16_385 - total))
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
        if total > 16_384:
            raise SecurityViolation("FDINFO_TOO_LARGE")
    finally:
        os.close(info_fd)
    matches = [
        line.split(b":", 1)[1].strip()
        for line in b"".join(chunks).splitlines()
        if line.startswith(b"mnt_id:")
    ]
    if len(matches) != 1 or not matches[0].isdigit():
        raise SecurityViolation("FDINFO_MOUNT_ID_UNAVAILABLE")
    return int(matches[0])


@dataclass(frozen=True)
class MountDomainSnapshot:
    source_path: str
    source_device: int
    source_inode: int
    source_mount_id: int
    source_nlink: int
    bytes_scanned: int
    content_sha256: str
    observed_mount_ids: Tuple[int, ...]
    required_mount_ids: Tuple[int, ...]

    def __post_init__(self) -> None:
        if self.source_path != "/proc/self/mountinfo":
            raise SecurityViolation("MOUNTINFO_SOURCE_PATH_DRIFT")
        for field in (
            "source_device",
            "source_inode",
            "source_mount_id",
            "source_nlink",
            "bytes_scanned",
        ):
            _require_nonnegative_int(getattr(self, field), field)
        if self.source_nlink != 1 or self.bytes_scanned <= 0:
            raise SecurityViolation("MOUNTINFO_SOURCE_IDENTITY_INVALID")
        _require_sha256(self.content_sha256, "mountinfo.content_sha256")
        if (
            not self.observed_mount_ids
            or tuple(sorted(set(self.observed_mount_ids))) != self.observed_mount_ids
        ):
            raise SecurityViolation("MOUNTINFO_OBSERVED_DOMAIN_INVALID")
        if tuple(sorted(set(self.required_mount_ids))) != self.required_mount_ids:
            raise SecurityViolation("MOUNTINFO_REQUIRED_DOMAIN_INVALID")
        if not set(self.required_mount_ids).issubset(self.observed_mount_ids):
            raise SecurityViolation("EXPECTED_MOUNT_DOMAIN_MISSING")

    def as_dict(self) -> Mapping[str, Any]:
        value: dict[str, Any] = {
            "schema_version": MOUNT_DOMAIN_SCHEMA_VERSION,
            "source_path": self.source_path,
            "source_device": self.source_device,
            "source_inode": self.source_inode,
            "source_mount_id": self.source_mount_id,
            "source_nlink": self.source_nlink,
            "bytes_scanned": self.bytes_scanned,
            "content_sha256": self.content_sha256,
            "observed_mount_ids": list(self.observed_mount_ids),
            "required_mount_ids": list(self.required_mount_ids),
        }
        value["snapshot_sha256"] = _semantic_sha256(value, ("snapshot_sha256",))
        return _json_copy(value)


def capture_mount_domain(required_mount_ids: Sequence[int]) -> MountDomainSnapshot:
    """Capture Linux mount metadata from the single reviewed mountinfo source."""

    if isinstance(required_mount_ids, (str, bytes)):
        raise SecurityViolation("MOUNTINFO_REQUIRED_DOMAIN_INVALID")
    required = tuple(sorted(set(required_mount_ids)))
    if any(type(value) is not int or value < 0 for value in required_mount_ids):
        raise SecurityViolation("MOUNTINFO_REQUIRED_DOMAIN_INVALID")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open("/proc/self/mountinfo", flags)
    try:
        before = os.fstat(fd)
        mount_id = _fd_mount_id(fd)
        chunks: list[bytes] = []
        total = 0
        while total <= 8 * 1024 * 1024:
            chunk = os.read(fd, min(65_536, 8 * 1024 * 1024 + 1 - total))
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
        if total > 8 * 1024 * 1024:
            raise SecurityViolation("MOUNTINFO_TOO_LARGE")
        after = os.fstat(fd)
    finally:
        os.close(fd)
    if (
        not stat.S_ISREG(before.st_mode)
        or before.st_nlink != 1
        or (before.st_dev, before.st_ino, before.st_mode)
        != (after.st_dev, after.st_ino, after.st_mode)
    ):
        raise SecurityViolation("MOUNTINFO_SOURCE_IDENTITY_INVALID")
    content = b"".join(chunks)
    if not content.endswith(b"\n"):
        raise SecurityViolation("MOUNTINFO_TRUNCATED")
    observed: set[int] = set()
    for line in content.splitlines():
        fields = line.split(b" ")
        if len(fields) < 10 or not fields[0].isdigit() or b"-" not in fields:
            raise SecurityViolation("MOUNTINFO_LINE_INVALID")
        observed.add(int(fields[0]))
    return MountDomainSnapshot(
        source_path="/proc/self/mountinfo",
        source_device=int(before.st_dev),
        source_inode=int(before.st_ino),
        source_mount_id=mount_id,
        source_nlink=int(before.st_nlink),
        bytes_scanned=len(content),
        content_sha256=_sha256_bytes(content),
        observed_mount_ids=tuple(sorted(observed)),
        required_mount_ids=required,
    )


def _capture_existing_identity(path: str) -> Mapping[str, Any]:
    canonical, pieces = _canonical_absolute_path(path)
    current = "/"
    for piece in pieces:
        current = os.path.join(current, piece)
        item = os.lstat(current)
        if stat.S_ISLNK(item.st_mode):
            raise SecurityViolation("SYMLINK_COMPONENT_FORBIDDEN")
    item = os.lstat(canonical)
    flags = _linux_o_path_flag() | os.O_NOFOLLOW
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    fd = os.open(canonical, flags)
    try:
        observed = os.fstat(fd)
        mount_id = _fd_mount_id(fd)
    finally:
        os.close(fd)
    after = os.lstat(canonical)
    if (observed.st_dev, observed.st_ino, observed.st_mode) != (
        item.st_dev,
        item.st_ino,
        item.st_mode,
    ) or (after.st_dev, after.st_ino, after.st_mode) != (
        observed.st_dev,
        observed.st_ino,
        observed.st_mode,
    ):
        raise SecurityViolation("PATH_IDENTITY_CHANGED_DURING_CAPTURE")
    return {
        "canonical_path": canonical,
        "device": int(item.st_dev),
        "inode": int(item.st_ino),
        "mount_id": mount_id,
        "size_bytes": int(item.st_size),
        "mtime_ns": int(item.st_mtime_ns),
        "mode": int(item.st_mode),
        "nlink": int(item.st_nlink),
        "object_kind": _mode_kind(item.st_mode),
    }


@dataclass(frozen=True)
class FrozenCanonicalRoot:
    root_id: str
    canonical_path: str
    device: int
    inode: int
    mount_id: int
    object_kind: str
    operations: Tuple[str, ...]
    nlink: int = 2

    def __post_init__(self) -> None:
        _require_nonempty_text(self.root_id, "root_id")
        canonical, _pieces = _canonical_absolute_path(self.canonical_path)
        if canonical != self.canonical_path:
            raise SecurityViolation("ROOT_PATH_NONCANONICAL")
        _require_nonnegative_int(self.device, "device")
        _require_nonnegative_int(self.inode, "inode")
        _require_nonnegative_int(self.mount_id, "mount_id")
        _require_nonnegative_int(self.nlink, "nlink")
        if self.nlink < 2:
            raise SecurityViolation("ROOT_NLINK_INVALID")
        if self.object_kind != "DIRECTORY":
            raise SecurityViolation("ROOT_MUST_BE_DIRECTORY")
        if not self.operations or any(op not in _ROOT_OPERATIONS for op in self.operations):
            raise SecurityViolation("INVALID_ROOT_OPERATION")
        if tuple(sorted(set(self.operations))) != self.operations:
            raise SecurityViolation("ROOT_OPERATIONS_NOT_CANONICAL")

    @classmethod
    def capture(
        cls, root_id: str, path: str, operations: Sequence[str]
    ) -> "FrozenCanonicalRoot":
        identity = _capture_existing_identity(path)
        return cls(
            root_id=root_id,
            canonical_path=str(identity["canonical_path"]),
            device=int(identity["device"]),
            inode=int(identity["inode"]),
            mount_id=int(identity["mount_id"]),
            object_kind=str(identity["object_kind"]),
            operations=tuple(sorted(set(operations))),
            nlink=int(identity["nlink"]),
        )

    def as_dict(self) -> Mapping[str, Any]:
        return {
            "root_id": self.root_id,
            "canonical_path": self.canonical_path,
            "device": self.device,
            "inode": self.inode,
            "mount_id": self.mount_id,
            "object_kind": self.object_kind,
            "operations": list(self.operations),
            "nlink": self.nlink,
        }


@dataclass(frozen=True)
class FrozenPathRule:
    rule_id: str
    canonical_path: str
    device: int
    inode: int
    mount_id: int
    size_bytes: int
    mtime_ns: int
    object_kind: str
    scope: str
    classification: str
    content_sha256: Optional[str] = None
    nlink: int = 1

    def __post_init__(self) -> None:
        _require_nonempty_text(self.rule_id, "rule_id")
        canonical, _pieces = _canonical_absolute_path(self.canonical_path)
        if canonical != self.canonical_path:
            raise SecurityViolation("FORBIDDEN_PATH_NONCANONICAL")
        _require_nonnegative_int(self.device, "device")
        _require_nonnegative_int(self.inode, "inode")
        _require_nonnegative_int(self.mount_id, "mount_id")
        _require_nonnegative_int(self.size_bytes, "size_bytes")
        _require_nonnegative_int(self.mtime_ns, "mtime_ns")
        _require_nonnegative_int(self.nlink, "nlink")
        if self.nlink < 1:
            raise SecurityViolation("FORBIDDEN_NLINK_INVALID")
        if self.object_kind not in {"FILE", "DIRECTORY", "OTHER"}:
            raise SecurityViolation("INVALID_FORBIDDEN_OBJECT_KIND")
        if self.scope not in {"EXACT", "TREE"}:
            raise SecurityViolation("INVALID_FORBIDDEN_SCOPE")
        if self.scope == "TREE" and self.object_kind != "DIRECTORY":
            raise SecurityViolation("TREE_RULE_MUST_REFERENCE_DIRECTORY")
        if self.classification not in {
            "PROTECTED_DATA",
            "PROTECTED_FEATURE",
            "PROTECTED_PREDICTION",
            "OFFICIAL_WORKFLOW",
            "HISTORICAL_ROOT",
        }:
            raise SecurityViolation("INVALID_FORBIDDEN_CLASSIFICATION")
        if self.content_sha256 is not None:
            _require_sha256(self.content_sha256, "content_sha256")

    @classmethod
    def capture_metadata_only(
        cls,
        rule_id: str,
        path: str,
        *,
        scope: str,
        classification: str,
        approved_content_sha256: Optional[str] = None,
    ) -> "FrozenPathRule":
        """Capture lstat metadata only; this function never opens file content."""

        identity = _capture_existing_identity(path)
        return cls(
            rule_id=rule_id,
            canonical_path=str(identity["canonical_path"]),
            device=int(identity["device"]),
            inode=int(identity["inode"]),
            mount_id=int(identity["mount_id"]),
            size_bytes=int(identity["size_bytes"]),
            mtime_ns=int(identity["mtime_ns"]),
            object_kind=str(identity["object_kind"]),
            scope=scope,
            classification=classification,
            content_sha256=approved_content_sha256,
            nlink=int(identity["nlink"]),
        )

    def as_dict(self) -> Mapping[str, Any]:
        return {
            "rule_id": self.rule_id,
            "canonical_path": self.canonical_path,
            "device": self.device,
            "inode": self.inode,
            "mount_id": self.mount_id,
            "size_bytes": self.size_bytes,
            "mtime_ns": self.mtime_ns,
            "object_kind": self.object_kind,
            "scope": self.scope,
            "classification": self.classification,
            "content_sha256": self.content_sha256,
            "nlink": self.nlink,
        }


@dataclass(frozen=True)
class PathIntent:
    request_id: str
    operation: str
    path: str
    expected_kind: str
    purpose: str
    goal_id: str
    authority_id: str

    def fingerprint(self) -> str:
        return _request_fingerprint(
            {
                "request_id": self.request_id,
                "operation": self.operation,
                "path": self.path,
                "expected_kind": self.expected_kind,
                "purpose": self.purpose,
                "goal_id": self.goal_id,
                "authority_id": self.authority_id,
            }
        )


@dataclass(frozen=True)
class DescriptorRecord:
    canonical_prefix: str
    device: int
    inode: int
    mount_id: int
    object_kind: str
    size_bytes: int
    mtime_ns: int
    nlink: int = 1

    def as_dict(self) -> Mapping[str, Any]:
        return {
            "canonical_prefix": self.canonical_prefix,
            "device": self.device,
            "inode": self.inode,
            "mount_id": self.mount_id,
            "object_kind": self.object_kind,
            "size_bytes": self.size_bytes,
            "mtime_ns": self.mtime_ns,
            "nlink": self.nlink,
        }


@dataclass(frozen=True)
class PathGrant:
    request_id: str
    operation: str
    canonical_path: str
    authorized_root_id: str
    existing: bool
    final_identity: Optional[DescriptorRecord]
    descriptor_chain_sha256: str
    content_opened: bool = False

    def __post_init__(self) -> None:
        if self.content_opened:
            raise SecurityViolation("PREOPEN_GRANT_CANNOT_CLAIM_CONTENT_OPEN")

    def as_dict(self) -> Mapping[str, Any]:
        return {
            "request_id": self.request_id,
            "operation": self.operation,
            "canonical_path": self.canonical_path,
            "authorized_root_id": self.authorized_root_id,
            "existing": self.existing,
            "final_identity": (
                None if self.final_identity is None else self.final_identity.as_dict()
            ),
            "descriptor_chain_sha256": self.descriptor_chain_sha256,
            "content_opened": False,
        }


class PinnedPathCapability:
    """Single-use O_PATH capability retaining the reviewed descriptor target."""

    def __init__(
        self,
        *,
        fd: int,
        grant: PathGrant,
        goal_id: str,
        authority_id: str,
        parent_event_sha256: str,
    ) -> None:
        if type(fd) is not int or fd < 0:
            raise SecurityViolation("PINNED_DESCRIPTOR_INVALID")
        if grant.operation != "READ_CONTENT" or grant.final_identity is None:
            raise SecurityViolation("PINNED_CAPABILITY_REQUIRES_CONTENT_FILE")
        _require_sha256(parent_event_sha256, "parent_event_sha256")
        self._fd: Optional[int] = fd
        self.grant = grant
        self.goal_id = _require_nonempty_text(goal_id, "goal_id")
        self.authority_id = _require_nonempty_text(authority_id, "authority_id")
        self.parent_event_sha256 = parent_event_sha256
        base = {
            "goal_id": self.goal_id,
            "authority_id": self.authority_id,
            "parent_event_sha256": self.parent_event_sha256,
            "grant": grant.as_dict(),
            "pinned_identity": grant.final_identity.as_dict(),
        }
        self.capability_sha256 = _semantic_sha256(base)

    @property
    def consumed(self) -> bool:
        return self._fd is None

    def _take_fd(self) -> int:
        if self._fd is None:
            raise SecurityViolation("PINNED_CAPABILITY_REPLAY")
        fd = self._fd
        self._fd = None
        return fd

    def close(self) -> None:
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None


class EvidenceLedger:
    """In-memory append-only hash chain; persistence is owned by stage control."""

    def __init__(self, scope_id: str, goal_id: str, authority_id: str) -> None:
        self.scope_id = _require_nonempty_text(scope_id, "scope_id")
        self.goal_id = _require_nonempty_text(goal_id, "goal_id")
        self.authority_id = _require_nonempty_text(authority_id, "authority_id")
        self._rows: list[Mapping[str, Any]] = []
        self._closed = False
        self._append(
            event_type="LEDGER_SCOPE_OPENED",
            decision="ALLOW",
            reason_code="SCOPE_OPEN",
            classification="CONTROL",
            request_fingerprint=_request_fingerprint(
                {
                    "scope_id": self.scope_id,
                    "goal_id": self.goal_id,
                    "authority_id": self.authority_id,
                }
            ),
            counters={},
            details={"instrumentation_contract": "PREOPEN_MEDIATION_REQUIRED"},
        )

    def _append(
        self,
        *,
        event_type: str,
        decision: str,
        reason_code: str,
        classification: str,
        request_fingerprint: str,
        counters: Mapping[str, int],
        details: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        if self._closed:
            raise SecurityViolation("LEDGER_ALREADY_CLOSED")
        _require_nonempty_text(event_type, "event_type")
        if decision not in _VALID_DECISIONS:
            raise SecurityViolation("INVALID_LEDGER_DECISION")
        _require_nonempty_text(reason_code, "reason_code")
        if classification not in _VALID_CLASSIFICATIONS:
            raise SecurityViolation("INVALID_LEDGER_CLASSIFICATION")
        _require_sha256(request_fingerprint, "request_fingerprint")
        canonical_counters = {
            "content_open_count": 0,
            "content_bytes_read": 0,
            "content_bytes_written": 0,
            "lmdb_transaction_open_count": 0,
            "model_forward_count": 0,
            "protected_metadata_descriptor_open_count": 0,
        }
        unknown = set(counters).difference(canonical_counters)
        if unknown:
            raise SecurityViolation("UNKNOWN_LEDGER_COUNTER")
        for key, value in counters.items():
            canonical_counters[key] = _require_nonnegative_int(value, key)
        if decision == "DENY" and any(canonical_counters.values()):
            raise SecurityViolation("DENIAL_MUST_PRECEDE_IO")
        previous = None if not self._rows else self._rows[-1]["event_sha256"]
        observed_at_ns = time.time_ns()
        if self._rows:
            observed_at_ns = max(
                observed_at_ns, int(self._rows[-1]["observed_at_ns"]) + 1
            )
        row: dict[str, Any] = {
            "schema_version": LEDGER_SCHEMA_VERSION,
            "scope_id": self.scope_id,
            "goal_id": self.goal_id,
            "authority_id": self.authority_id,
            "seq": len(self._rows) + 1,
            "observed_at_ns": observed_at_ns,
            "event_type": event_type,
            "decision": decision,
            "reason_code": reason_code,
            "classification": classification,
            "request_fingerprint": request_fingerprint,
            "counters": canonical_counters,
            "details": _json_copy(dict(details)),
            "prev_event_sha256": previous,
        }
        row["event_sha256"] = _semantic_sha256(row, ("event_sha256",))
        detached = _json_copy(row)
        self._rows.append(detached)
        return _json_copy(detached)

    def record_preopen(
        self,
        *,
        event_type: str,
        decision: str,
        reason_code: str,
        classification: str,
        request_fingerprint: str,
        protected_metadata_descriptor_open_count: int = 0,
        details: Optional[Mapping[str, Any]] = None,
    ) -> Mapping[str, Any]:
        if decision == "DENY" and protected_metadata_descriptor_open_count != 0:
            # A rejected request may have O_PATH metadata observations while
            # resolving an alias.  They are not content I/O, but keeping DENY
            # counters uniformly zero makes the pre-content invariant machine
            # checkable.  Alias observations are described, not counted as I/O.
            protected_metadata_descriptor_open_count = 0
        return self._append(
            event_type=event_type,
            decision=decision,
            reason_code=reason_code,
            classification=classification,
            request_fingerprint=request_fingerprint,
            counters={
                "protected_metadata_descriptor_open_count": (
                    protected_metadata_descriptor_open_count
                )
            },
            details={} if details is None else details,
        )

    def record_observation(
        self,
        *,
        event_type: str,
        classification: str,
        request_fingerprint: str,
        counters: Mapping[str, int],
        details: Optional[Mapping[str, Any]] = None,
    ) -> Mapping[str, Any]:
        return self._append(
            event_type=event_type,
            decision="ALLOW",
            reason_code="OBSERVED",
            classification=classification,
            request_fingerprint=request_fingerprint,
            counters=counters,
            details={} if details is None else details,
        )

    def close(
        self,
        *,
        instrumentation_complete: bool,
        unmediated_content_open_count: int,
    ) -> Mapping[str, Any]:
        if not isinstance(instrumentation_complete, bool):
            raise SecurityViolation("INVALID_INSTRUMENTATION_CLOSURE")
        _require_nonnegative_int(
            unmediated_content_open_count, "unmediated_content_open_count"
        )
        row = self._append(
            event_type="LEDGER_SCOPE_CLOSED",
            decision="ALLOW",
            reason_code="SCOPE_CLOSE",
            classification="CONTROL",
            request_fingerprint=_request_fingerprint(
                {
                    "scope_id": self.scope_id,
                    "instrumentation_complete": instrumentation_complete,
                    "unmediated_content_open_count": unmediated_content_open_count,
                }
            ),
            counters={},
            details={
                "instrumentation_complete": instrumentation_complete,
                "unmediated_content_open_count": unmediated_content_open_count,
            },
        )
        self._closed = True
        return row

    @property
    def rows(self) -> Tuple[Mapping[str, Any], ...]:
        return tuple(_json_copy(row) for row in self._rows)

    @property
    def closed(self) -> bool:
        return self._closed


class PinnedContentReader:
    """Ledger-accounted reader created only from a persisted open intent."""

    def __init__(
        self,
        *,
        fd: int,
        open_intent_id: str,
        request_fingerprint: str,
        ledger: EvidenceLedger,
    ) -> None:
        self._fd: Optional[int] = fd
        self._open_intent_id = open_intent_id
        self._request_fingerprint = request_fingerprint
        self._ledger = ledger
        self._bytes_read = 0

    def read(self, max_bytes: int) -> bytes:
        if self._fd is None:
            raise SecurityViolation("CONTENT_READER_CLOSED")
        if type(max_bytes) is not int or max_bytes <= 0 or max_bytes > 64 * 1024 * 1024:
            raise SecurityViolation("CONTENT_READ_SIZE_INVALID")
        chunk = os.read(self._fd, max_bytes)
        self._bytes_read += len(chunk)
        return chunk

    @property
    def bytes_read(self) -> int:
        return self._bytes_read

    def close(self) -> None:
        if self._fd is None:
            raise SecurityViolation("CONTENT_READER_CLOSE_REPLAY")
        os.close(self._fd)
        self._fd = None
        self._ledger.record_observation(
            event_type="CONTENT_OPEN_RECEIPT_COMMITTED",
            classification="NON_PROTECTED_CONTENT",
            request_fingerprint=self._request_fingerprint,
            counters={
                "content_open_count": 1,
                "content_bytes_read": self._bytes_read,
            },
            details={
                "open_intent_id": self._open_intent_id,
                "descriptor_pinned": True,
                "reader_closed": True,
            },
        )


def open_pinned_content(
    capability: PinnedPathCapability,
    persisted_intent_receipt: Mapping[str, Any],
    ledger: EvidenceLedger,
) -> PinnedContentReader:
    """Open the pinned inode only after an exact fsynced/CAS control receipt."""

    required_fields = {
        "schema_version",
        "status",
        "goal_id",
        "authority_id",
        "parent_event_sha256",
        "request_id",
        "path_grant_sha256",
        "descriptor_chain_sha256",
        "pinned_capability_sha256",
        "open_intent_id",
        "persistence_status",
        "persistence_transaction_id",
        "receipt_sha256",
    }
    receipt = dict(
        _require_exact_mapping(
            persisted_intent_receipt,
            required_fields,
            "CONTENT_OPEN_INTENT_RECEIPT_FIELDS_MISMATCH",
        )
    )
    receipt_sha = _require_sha256(receipt["receipt_sha256"], "receipt_sha256")
    if receipt_sha != _semantic_sha256(receipt, ("receipt_sha256",)):
        raise SecurityViolation("CONTENT_OPEN_INTENT_RECEIPT_SELF_HASH_MISMATCH")
    expected = {
        "schema_version": CONTENT_OPEN_INTENT_SCHEMA_VERSION,
        "status": "COMMITTED_BEFORE_CONTENT_OPEN",
        "goal_id": capability.goal_id,
        "authority_id": capability.authority_id,
        "parent_event_sha256": capability.parent_event_sha256,
        "request_id": capability.grant.request_id,
        "path_grant_sha256": _sha256_bytes(
            _canonical_json_bytes(capability.grant.as_dict())
        ),
        "descriptor_chain_sha256": capability.grant.descriptor_chain_sha256,
        "pinned_capability_sha256": capability.capability_sha256,
        "persistence_status": "FSYNCED_CAS_COMMITTED",
    }
    for field, value in expected.items():
        if receipt.get(field) != value:
            raise SecurityViolation("CONTENT_OPEN_INTENT_RECEIPT_BINDING_MISMATCH")
    _require_nonempty_text(receipt["open_intent_id"], "open_intent_id")
    _require_nonempty_text(
        receipt["persistence_transaction_id"],
        "persistence_transaction_id",
    )
    if capability.consumed:
        raise SecurityViolation("PINNED_CAPABILITY_REPLAY")
    request_fingerprint = _request_fingerprint(
        {
            "request_id": capability.grant.request_id,
            "open_intent_id": receipt["open_intent_id"],
            "receipt_sha256": receipt_sha,
        }
    )
    ledger.record_preopen(
        event_type="CONTENT_OPEN_INTENT_COMMITTED",
        decision="ALLOW",
        reason_code="PERSISTED_DESCRIPTOR_OPEN_INTENT_VALIDATED",
        classification="NON_PROTECTED_METADATA",
        request_fingerprint=request_fingerprint,
        details={
            "open_intent_id": receipt["open_intent_id"],
            "receipt_sha256": receipt_sha,
            "content_opened": False,
        },
    )
    pinned_fd = capability._take_fd()
    try:
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        content_fd = os.open("/proc/self/fd/%d" % pinned_fd, flags)
        pinned_stat = os.fstat(pinned_fd)
        opened_stat = os.fstat(content_fd)
        pinned_mount_id = _fd_mount_id(pinned_fd)
        opened_mount_id = _fd_mount_id(content_fd)
        final = capability.grant.final_identity
        if final is None or (
            int(opened_stat.st_dev),
            int(opened_stat.st_ino),
            opened_mount_id,
            int(opened_stat.st_nlink),
        ) != (
            final.device,
            final.inode,
            final.mount_id,
            final.nlink,
        ) or (
            pinned_stat.st_dev,
            pinned_stat.st_ino,
            pinned_mount_id,
        ) != (
            opened_stat.st_dev,
            opened_stat.st_ino,
            opened_mount_id,
        ):
            os.close(content_fd)
            raise SecurityViolation("PINNED_CONTENT_OPEN_IDENTITY_MISMATCH")
    except (OSError, SecurityViolation) as exc:
        violation = (
            exc
            if isinstance(exc, SecurityViolation)
            else SecurityViolation("PINNED_CONTENT_OPEN_FAILED")
        )
        ledger.record_observation(
            event_type="CONTENT_OPEN_FAILED_AFTER_INTENT",
            classification="NON_PROTECTED_METADATA",
            request_fingerprint=request_fingerprint,
            counters={},
            details={
                "open_intent_id": receipt["open_intent_id"],
                "reason_code": violation.code,
                "content_opened": False,
            },
        )
        raise violation
    finally:
        os.close(pinned_fd)
    return PinnedContentReader(
        fd=content_fd,
        open_intent_id=str(receipt["open_intent_id"]),
        request_fingerprint=request_fingerprint,
        ledger=ledger,
    )


def verify_ledger_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    expected_scope_id: str,
    expected_goal_id: str,
    expected_authority_id: str,
    require_closed: bool,
) -> Mapping[str, Any]:
    if not rows:
        raise SecurityViolation("LEDGER_EMPTY")
    previous: Optional[str] = None
    previous_time = -1
    for offset, original in enumerate(rows, start=1):
        row = _json_copy(original)
        required = {
            "schema_version",
            "scope_id",
            "goal_id",
            "authority_id",
            "seq",
            "observed_at_ns",
            "event_type",
            "decision",
            "reason_code",
            "classification",
            "request_fingerprint",
            "counters",
            "details",
            "prev_event_sha256",
            "event_sha256",
        }
        if set(row) != required:
            raise SecurityViolation("LEDGER_ROW_FIELDS_MISMATCH")
        if row["schema_version"] != LEDGER_SCHEMA_VERSION:
            raise SecurityViolation("LEDGER_SCHEMA_MISMATCH")
        if row["scope_id"] != expected_scope_id:
            raise SecurityViolation("LEDGER_SCOPE_MISMATCH")
        if row["goal_id"] != expected_goal_id:
            raise SecurityViolation("LEDGER_GOAL_MISMATCH")
        if row["authority_id"] != expected_authority_id:
            raise SecurityViolation("LEDGER_AUTHORITY_MISMATCH")
        if type(row["seq"]) is not int or row["seq"] != offset:
            raise SecurityViolation("LEDGER_SEQUENCE_MISMATCH")
        if isinstance(row["observed_at_ns"], bool) or not isinstance(
            row["observed_at_ns"], int
        ):
            raise SecurityViolation("LEDGER_TIME_INVALID")
        if row["observed_at_ns"] < 0 or row["observed_at_ns"] < previous_time:
            raise SecurityViolation("LEDGER_TIME_REGRESSION")
        previous_time = row["observed_at_ns"]
        if row["decision"] not in _VALID_DECISIONS:
            raise SecurityViolation("INVALID_LEDGER_DECISION")
        if row["classification"] not in _VALID_CLASSIFICATIONS:
            raise SecurityViolation("INVALID_LEDGER_CLASSIFICATION")
        if not isinstance(row["event_type"], str) or not row["event_type"]:
            raise SecurityViolation("LEDGER_EVENT_TYPE_INVALID")
        if not isinstance(row["reason_code"], str) or not row["reason_code"]:
            raise SecurityViolation("LEDGER_REASON_CODE_INVALID")
        if not isinstance(row["details"], dict):
            raise SecurityViolation("LEDGER_DETAILS_INVALID")
        _require_sha256(row["request_fingerprint"], "request_fingerprint")
        if row["prev_event_sha256"] != previous:
            raise SecurityViolation("LEDGER_PREV_HASH_MISMATCH")
        expected_hash = _semantic_sha256(row, ("event_sha256",))
        if not hmac.compare_digest(str(row["event_sha256"]), expected_hash):
            raise SecurityViolation("LEDGER_EVENT_HASH_MISMATCH")
        counters = row["counters"]
        if not isinstance(counters, dict):
            raise SecurityViolation("LEDGER_COUNTERS_INVALID")
        counter_fields = {
            "content_open_count",
            "content_bytes_read",
            "content_bytes_written",
            "lmdb_transaction_open_count",
            "model_forward_count",
            "protected_metadata_descriptor_open_count",
        }
        if set(counters) != counter_fields:
            raise SecurityViolation("LEDGER_COUNTER_FIELDS_MISMATCH")
        for key in sorted(counter_fields):
            if key not in counters:
                raise SecurityViolation("LEDGER_COUNTER_MISSING")
            _require_nonnegative_int(counters[key], key)
        if row["decision"] == "DENY" and any(counters.values()):
            raise SecurityViolation("DENIAL_AFTER_IO_DETECTED")
        previous = str(row["event_sha256"])
    if rows[0]["event_type"] != "LEDGER_SCOPE_OPENED":
        raise SecurityViolation("LEDGER_OPEN_EVENT_MISSING")
    closed = rows[-1]["event_type"] == "LEDGER_SCOPE_CLOSED"
    if require_closed and not closed:
        raise SecurityViolation("LEDGER_CLOSE_EVENT_MISSING")
    if any(row["event_type"] == "LEDGER_SCOPE_CLOSED" for row in rows[:-1]):
        raise SecurityViolation("LEDGER_EVENT_AFTER_CLOSE")
    return {
        "verified": True,
        "closed": closed,
        "event_count": len(rows),
        "head_event_sha256": previous,
        "ledger_rows_sha256": _sha256_bytes(_canonical_json_bytes(list(rows))),
    }


def derive_safety_manifest(
    rows: Sequence[Mapping[str, Any]],
    *,
    expected_scope_id: str,
    expected_goal_id: str,
    expected_authority_id: str,
    require_closed: bool = True,
) -> Mapping[str, Any]:
    verification = verify_ledger_rows(
        rows,
        expected_scope_id=expected_scope_id,
        expected_goal_id=expected_goal_id,
        expected_authority_id=expected_authority_id,
        require_closed=require_closed,
    )
    totals = {
        "denied_attempt_count": 0,
        "allowed_preopen_count": 0,
        "content_open_count": 0,
        "content_bytes_read": 0,
        "content_bytes_written": 0,
        "lmdb_transaction_open_count": 0,
        "model_forward_count": 0,
        "protected_metadata_descriptor_open_count": 0,
        "protected_content_open_count": 0,
        "protected_feature_open_count": 0,
        "protected_prediction_open_count": 0,
        "protected_metric_open_count": 0,
        "protected_label_open_count": 0,
        "protected_query_text_timestamp_open_count": 0,
        "protected_id_mapping_content_open_count": 0,
        "historical_write_bytes": 0,
        "official_workflow_allowed_count": 0,
        "open_intent_count": 0,
        "open_receipt_count": 0,
        "unresolved_open_intent_count": 0,
        "observed_unmediated_content_open_count": 0,
    }
    open_intents: set[str] = set()
    open_receipts: set[str] = set()
    for row in rows:
        if row["decision"] == "DENY":
            totals["denied_attempt_count"] += 1
        elif row["event_type"].endswith("_PREOPEN"):
            totals["allowed_preopen_count"] += 1
        counters = row["counters"]
        for key in (
            "content_open_count",
            "content_bytes_read",
            "content_bytes_written",
            "lmdb_transaction_open_count",
            "model_forward_count",
            "protected_metadata_descriptor_open_count",
        ):
            totals[key] += int(counters[key])
        if int(counters["content_open_count"]) > 0 and row["event_type"] not in {
            "CONTENT_OPEN_RECEIPT_COMMITTED",
            "CONTENT_OPEN_FAILED_AFTER_INTENT",
        }:
            totals["observed_unmediated_content_open_count"] += int(
                counters["content_open_count"]
            )
        if row["classification"] == "PROTECTED_CONTENT":
            totals["protected_content_open_count"] += int(
                counters["content_open_count"]
            )
        protected_counter_keys = {
            "PROTECTED_FEATURE": "protected_feature_open_count",
            "PROTECTED_PREDICTION": "protected_prediction_open_count",
            "PROTECTED_METRIC": "protected_metric_open_count",
            "PROTECTED_LABEL": "protected_label_open_count",
            "PROTECTED_QUERY_TEXT_TIMESTAMP": (
                "protected_query_text_timestamp_open_count"
            ),
        }
        protected_key = protected_counter_keys.get(row["classification"])
        if protected_key is not None:
            totals[protected_key] += int(counters["content_open_count"])
        if row["classification"] == "PROTECTED_ID_MAPPING_ONLY":
            totals["protected_id_mapping_content_open_count"] += int(
                counters["content_open_count"]
            )
        if row["classification"] == "HISTORICAL_WRITE":
            totals["historical_write_bytes"] += int(counters["content_bytes_written"])
        if row["classification"] == "OFFICIAL_WORKFLOW" and row["decision"] == "ALLOW":
            totals["official_workflow_allowed_count"] += 1
        if row["event_type"] == "CONTENT_OPEN_INTENT_COMMITTED":
            intent_id = row["details"].get("open_intent_id")
            if not isinstance(intent_id, str) or not intent_id:
                raise SecurityViolation("OPEN_INTENT_ID_MISSING")
            if intent_id in open_intents:
                raise SecurityViolation("OPEN_INTENT_ID_DUPLICATE")
            open_intents.add(intent_id)
        if row["event_type"] in {
            "CONTENT_OPEN_RECEIPT_COMMITTED",
            "CONTENT_OPEN_FAILED_AFTER_INTENT",
        }:
            intent_id = row["details"].get("open_intent_id")
            if not isinstance(intent_id, str) or intent_id not in open_intents:
                raise SecurityViolation("OPEN_RECEIPT_WITHOUT_INTENT")
            if intent_id in open_receipts:
                raise SecurityViolation("OPEN_RECEIPT_ID_DUPLICATE")
            open_receipts.add(intent_id)
    totals["open_intent_count"] = len(open_intents)
    totals["open_receipt_count"] = len(open_receipts)
    totals["unresolved_open_intent_count"] = len(open_intents - open_receipts)
    closure = rows[-1]["details"] if verification["closed"] else {}
    instrumentation_complete = bool(closure.get("instrumentation_complete", False))
    unmediated = closure.get("unmediated_content_open_count", None)
    closure_valid = (
        instrumentation_complete
        and isinstance(unmediated, int)
        and not isinstance(unmediated, bool)
        and unmediated == 0
        and totals["observed_unmediated_content_open_count"] == 0
    )
    safe_for_nonprotected_f0 = bool(
        verification["closed"]
        and closure_valid
        and totals["protected_content_open_count"] == 0
        and totals["content_open_count"] == 0
        and totals["content_bytes_read"] == 0
        and totals["content_bytes_written"] == 0
        and totals["lmdb_transaction_open_count"] == 0
        and totals["protected_feature_open_count"] == 0
        and totals["protected_prediction_open_count"] == 0
        and totals["protected_metric_open_count"] == 0
        and totals["protected_label_open_count"] == 0
        and totals["protected_query_text_timestamp_open_count"] == 0
        and totals["protected_id_mapping_content_open_count"] == 0
        and totals["historical_write_bytes"] == 0
        and totals["official_workflow_allowed_count"] == 0
        and totals["model_forward_count"] == 0
        and totals["unresolved_open_intent_count"] == 0
    )
    manifest: dict[str, Any] = {
        "schema_version": SAFETY_MANIFEST_SCHEMA_VERSION,
        "scope_id": expected_scope_id,
        "goal_id": expected_goal_id,
        "authority_id": expected_authority_id,
        "ledger_verification": verification,
        "totals": totals,
        "claims": {
            "instrumentation_complete": instrumentation_complete,
            "unmediated_content_open_count": unmediated,
            "observed_unmediated_content_open_count": totals[
                "observed_unmediated_content_open_count"
            ],
            "protected_content_accessed": totals["protected_content_open_count"] > 0,
            "protected_feature_accessed": totals["protected_feature_open_count"] > 0,
            "protected_prediction_accessed": (
                totals["protected_prediction_open_count"] > 0
            ),
            "protected_metric_accessed": totals["protected_metric_open_count"] > 0,
            "protected_label_accessed": totals["protected_label_open_count"] > 0,
            "protected_query_text_timestamp_accessed": (
                totals["protected_query_text_timestamp_open_count"] > 0
            ),
            "protected_id_mapping_accessed": (
                totals["protected_id_mapping_content_open_count"] > 0
            ),
            "historical_root_written": totals["historical_write_bytes"] > 0,
            "official_workflow_allowed": totals["official_workflow_allowed_count"] > 0,
            "model_forward_executed": totals["model_forward_count"] > 0,
            "unresolved_open_intent_count": totals["unresolved_open_intent_count"],
            "safe_for_nonprotected_f0_metadata_dry_run": safe_for_nonprotected_f0,
        },
    }
    manifest["manifest_sha256"] = _semantic_sha256(manifest, ("manifest_sha256",))
    return _json_copy(manifest)


def verify_claimed_safety_manifest(
    claimed: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    *,
    expected_scope_id: str,
    expected_goal_id: str,
    expected_authority_id: str,
) -> Mapping[str, Any]:
    derived = derive_safety_manifest(
        rows,
        expected_scope_id=expected_scope_id,
        expected_goal_id=expected_goal_id,
        expected_authority_id=expected_authority_id,
        require_closed=True,
    )
    if not hmac.compare_digest(
        _sha256_bytes(_canonical_json_bytes(claimed)),
        _sha256_bytes(_canonical_json_bytes(derived)),
    ):
        raise SecurityViolation("SAFETY_MANIFEST_LEDGER_MISMATCH")
    return derived


def verify_claimed_safety_manifest_preopen(
    claimed: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    *,
    expected_scope_id: str,
    expected_goal_id: str,
    expected_authority_id: str,
    ledger: EvidenceLedger,
) -> Mapping[str, Any]:
    fingerprint = _request_fingerprint(
        {
            "claimed_sha256": _sha256_bytes(_canonical_json_bytes(claimed)),
            "rows_sha256": _sha256_bytes(_canonical_json_bytes(list(rows))),
        }
    )
    try:
        result = verify_claimed_safety_manifest(
            claimed,
            rows,
            expected_scope_id=expected_scope_id,
            expected_goal_id=expected_goal_id,
            expected_authority_id=expected_authority_id,
        )
    except SecurityViolation as exc:
        ledger.record_preopen(
            event_type="SAFETY_MANIFEST_PREOPEN",
            decision="DENY",
            reason_code=exc.code,
            classification="CONTROL",
            request_fingerprint=fingerprint,
            details={"content_opened": False},
        )
        raise
    ledger.record_preopen(
        event_type="SAFETY_MANIFEST_PREOPEN",
        decision="ALLOW",
        reason_code="SAFETY_MANIFEST_MATCHES_LEDGER",
        classification="CONTROL",
        request_fingerprint=fingerprint,
        details={"content_opened": False},
    )
    return result


def _forbidden_rule_matches_path(rule: FrozenPathRule, candidate: str) -> bool:
    if rule.scope == "EXACT":
        return candidate == rule.canonical_path
    return _is_within(candidate, rule.canonical_path)


def assert_isolated_write_roots(
    write_roots: Sequence[FrozenCanonicalRoot],
    forbidden_rules: Sequence[FrozenPathRule],
) -> None:
    if not write_roots:
        raise SecurityViolation("WRITE_ROOTS_EMPTY")
    for index, left in enumerate(write_roots):
        if "WRITE" not in left.operations:
            raise SecurityViolation("WRITE_ROOT_WITHOUT_WRITE_CAPABILITY")
        for right in write_roots[index + 1 :]:
            if _is_within(left.canonical_path, right.canonical_path) or _is_within(
                right.canonical_path, left.canonical_path
            ):
                raise SecurityViolation("WRITE_ROOTS_OVERLAP")
            if (left.device, left.inode) == (right.device, right.inode):
                raise SecurityViolation("WRITE_ROOTS_DEVICE_INODE_ALIAS")
        for rule in forbidden_rules:
            lexical_overlap = _forbidden_rule_matches_path(
                rule, left.canonical_path
            ) or _is_within(rule.canonical_path, left.canonical_path)
            if lexical_overlap:
                raise SecurityViolation("WRITE_ROOT_FORBIDDEN_OVERLAP")
            if (left.device, left.inode) == (rule.device, rule.inode):
                raise SecurityViolation("WRITE_ROOT_FORBIDDEN_DEVICE_INODE_ALIAS")


def assert_read_write_roots_disjoint(
    roots: Sequence[FrozenCanonicalRoot],
) -> None:
    read_roots = [
        root
        for root in roots
        if "READ_METADATA" in root.operations or "READ_CONTENT" in root.operations
    ]
    write_roots = [root for root in roots if "WRITE" in root.operations]
    if not read_roots or not write_roots:
        raise SecurityViolation("READ_WRITE_ROOT_SET_INCOMPLETE")
    for read_root in read_roots:
        if "WRITE" in read_root.operations:
            raise SecurityViolation("READ_WRITE_CAPABILITY_COLOCATED")
        for write_root in write_roots:
            if _is_within(read_root.canonical_path, write_root.canonical_path) or _is_within(
                write_root.canonical_path, read_root.canonical_path
            ):
                raise SecurityViolation("READ_WRITE_ROOTS_OVERLAP")
            if (read_root.device, read_root.inode) == (
                write_root.device,
                write_root.inode,
            ):
                raise SecurityViolation("READ_WRITE_ROOT_DEVICE_INODE_ALIAS")


def verify_pristine_write_root(root: FrozenCanonicalRoot) -> Mapping[str, Any]:
    """Descriptor-recapture an empty, nlink=2 write root before first output."""

    if root.operations != ("WRITE",):
        raise SecurityViolation("WRITE_ROOT_CAPABILITY_NOT_EXACT")
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_DIRECTORY", 0)
    )
    fd = os.open(root.canonical_path, flags)
    try:
        observed = os.fstat(fd)
        observed_mount_id = _fd_mount_id(fd)
        names = os.listdir(fd)
        after = os.fstat(fd)
    finally:
        os.close(fd)
    if (
        not stat.S_ISDIR(observed.st_mode)
        or (observed.st_dev, observed.st_ino) != (root.device, root.inode)
        or observed_mount_id != root.mount_id
        or (observed.st_dev, observed.st_ino, observed.st_mode, observed.st_nlink)
        != (after.st_dev, after.st_ino, after.st_mode, after.st_nlink)
    ):
        raise SecurityViolation("WRITE_ROOT_IDENTITY_DRIFT")
    if observed.st_nlink != 2 or root.nlink != 2:
        raise SecurityViolation("WRITE_ROOT_NLINK_NOT_PRISTINE")
    if names:
        raise SecurityViolation("WRITE_ROOT_NOT_PRISTINE")
    base = {
        "root_id": root.root_id,
        "canonical_path": root.canonical_path,
        "device": int(observed.st_dev),
        "inode": int(observed.st_ino),
        "mount_id": observed_mount_id,
        "nlink": int(observed.st_nlink),
        "entry_count": 0,
        "status": "PRISTINE_EMPTY_DESCRIPTOR_RECAPTURED",
    }
    return {**base, "receipt_sha256": _semantic_sha256(base)}


def assert_descriptor_chain_has_no_symlink(
    records: Sequence[DescriptorRecord],
) -> None:
    if not records:
        raise SecurityViolation("DESCRIPTOR_CHAIN_EMPTY")
    if any(record.object_kind == "SYMLINK" for record in records):
        raise SecurityViolation("SYMLINK_COMPONENT_FORBIDDEN")
    if any(record.nlink < 1 for record in records):
        raise SecurityViolation("DESCRIPTOR_CHAIN_NLINK_INVALID")


def validate_root_registry_preopen(
    *,
    roots: Sequence[FrozenCanonicalRoot],
    forbidden_rules: Sequence[FrozenPathRule],
    ledger: EvidenceLedger,
    detector_id: str,
) -> None:
    fingerprint = _request_fingerprint(
        {
            "detector_id": detector_id,
            "root_registry_sha256": _sha256_bytes(
                _canonical_json_bytes([root.as_dict() for root in roots])
            ),
            "forbidden_registry_sha256": _sha256_bytes(
                _canonical_json_bytes([rule.as_dict() for rule in forbidden_rules])
            ),
        }
    )
    try:
        write_roots = [root for root in roots if "WRITE" in root.operations]
        assert_isolated_write_roots(write_roots, forbidden_rules)
        assert_read_write_roots_disjoint(roots)
    except SecurityViolation as exc:
        ledger.record_preopen(
            event_type="ROOT_REGISTRY_PREOPEN",
            decision="DENY",
            reason_code=exc.code,
            classification="HISTORICAL_WRITE",
            request_fingerprint=fingerprint,
            details={"detector_id": detector_id, "content_opened": False},
        )
        raise
    ledger.record_preopen(
        event_type="ROOT_REGISTRY_PREOPEN",
        decision="ALLOW",
        reason_code="ROOT_REGISTRY_ISOLATED",
        classification="NON_PROTECTED_METADATA",
        request_fingerprint=fingerprint,
        details={"detector_id": detector_id, "content_opened": False},
    )


def validate_descriptor_chain_preopen(
    *,
    records: Sequence[DescriptorRecord],
    ledger: EvidenceLedger,
    detector_id: str,
) -> None:
    fingerprint = _request_fingerprint(
        {
            "detector_id": detector_id,
            "descriptor_chain_sha256": _sha256_bytes(
                _canonical_json_bytes([record.as_dict() for record in records])
            ),
        }
    )
    try:
        assert_descriptor_chain_has_no_symlink(records)
    except SecurityViolation as exc:
        ledger.record_preopen(
            event_type="DESCRIPTOR_CHAIN_PREOPEN",
            decision="DENY",
            reason_code=exc.code,
            classification="PROTECTED_METADATA",
            request_fingerprint=fingerprint,
            details={"detector_id": detector_id, "content_opened": False},
        )
        raise
    ledger.record_preopen(
        event_type="DESCRIPTOR_CHAIN_PREOPEN",
        decision="ALLOW",
        reason_code="DESCRIPTOR_CHAIN_SAFE",
        classification="NON_PROTECTED_METADATA",
        request_fingerprint=fingerprint,
        details={"detector_id": detector_id, "content_opened": False},
    )


class DescriptorPathGuard:
    """Authorize path intents using lexical rules plus an O_PATH inode walk."""

    def __init__(
        self,
        *,
        goal_id: str,
        authority_id: str,
        roots: Sequence[FrozenCanonicalRoot],
        forbidden_rules: Sequence[FrozenPathRule],
    ) -> None:
        self.goal_id = _require_nonempty_text(goal_id, "goal_id")
        self.authority_id = _require_nonempty_text(authority_id, "authority_id")
        if not roots:
            raise SecurityViolation("CANONICAL_ROOT_REGISTRY_EMPTY")
        self.roots = tuple(roots)
        self.forbidden_rules = tuple(forbidden_rules)
        if len({root.root_id for root in self.roots}) != len(self.roots):
            raise SecurityViolation("DUPLICATE_CANONICAL_ROOT_ID")
        if len({rule.rule_id for rule in self.forbidden_rules}) != len(
            self.forbidden_rules
        ):
            raise SecurityViolation("DUPLICATE_FORBIDDEN_RULE_ID")
        write_roots = [root for root in self.roots if "WRITE" in root.operations]
        if write_roots:
            assert_isolated_write_roots(write_roots, self.forbidden_rules)

    def registry(self) -> Mapping[str, Any]:
        value: dict[str, Any] = {
            "schema_version": SECURITY_SCHEMA_VERSION,
            "goal_id": self.goal_id,
            "authority_id": self.authority_id,
            "canonical_roots": [root.as_dict() for root in self.roots],
            "forbidden_rules": [rule.as_dict() for rule in self.forbidden_rules],
            "content_hash_policy": (
                "FORBIDDEN_CONTENT_HASH_NULL_UNLESS_PREAPPROVED_OR_ID_MAPPING_AUTHORIZED"
            ),
        }
        value["registry_sha256"] = _semantic_sha256(value, ("registry_sha256",))
        return _json_copy(value)

    def _select_root(self, path: str, operation: str) -> FrozenCanonicalRoot:
        required = "WRITE" if operation.startswith("WRITE_") else operation
        matches = [
            root
            for root in self.roots
            if required in root.operations and _is_within(path, root.canonical_path)
        ]
        if not matches:
            raise SecurityViolation("PATH_OUTSIDE_AUTHORIZED_ROOT")
        matches.sort(key=lambda item: len(item.canonical_path), reverse=True)
        if len(matches) > 1 and len(matches[0].canonical_path) == len(
            matches[1].canonical_path
        ):
            raise SecurityViolation("AMBIGUOUS_AUTHORIZED_ROOT")
        return matches[0]

    def _check_lexical_forbidden(self, path: str) -> None:
        for rule in self.forbidden_rules:
            if _forbidden_rule_matches_path(rule, path):
                raise SecurityViolation("FORBIDDEN_PATH_EXACT_OR_TREE")

    def _walk(
        self,
        path: str,
        *,
        selected_root: FrozenCanonicalRoot,
        expected_kind: str,
        allow_missing_leaf: bool,
    ) -> Tuple[bool, Tuple[DescriptorRecord, ...]]:
        _canonical, pieces = _canonical_absolute_path(path)
        base_flags = _linux_o_path_flag() | os.O_NOFOLLOW
        if hasattr(os, "O_CLOEXEC"):
            base_flags |= os.O_CLOEXEC
        root_flags = base_flags | os.O_DIRECTORY
        current_fd = os.open("/", root_flags)
        records: list[DescriptorRecord] = []
        prefix = ""
        selected_root_verified = False
        try:
            for offset, piece in enumerate(pieces):
                final = offset == len(pieces) - 1
                prefix = prefix + "/" + piece
                try:
                    next_fd = os.open(piece, base_flags, dir_fd=current_fd)
                except OSError as exc:
                    if exc.errno == errno.ENOENT and final and allow_missing_leaf:
                        if not selected_root_verified:
                            raise SecurityViolation("AUTHORIZED_ROOT_IDENTITY_NOT_OBSERVED")
                        return False, tuple(records)
                    if exc.errno in {errno.ELOOP, errno.ENOTDIR}:
                        raise SecurityViolation("SYMLINK_OR_NONDIRECTORY_COMPONENT")
                    if exc.errno == errno.ENOENT:
                        raise SecurityViolation("PATH_COMPONENT_MISSING")
                    raise SecurityViolation("DESCRIPTOR_WALK_OPEN_FAILED")
                os.close(current_fd)
                current_fd = next_fd
                item = os.fstat(current_fd)
                kind = _mode_kind(item.st_mode)
                if kind == "SYMLINK":
                    raise SecurityViolation("SYMLINK_COMPONENT_FORBIDDEN")
                if not final and kind != "DIRECTORY":
                    raise SecurityViolation("NONDIRECTORY_INTERMEDIATE_COMPONENT")
                record = DescriptorRecord(
                    canonical_prefix=prefix,
                    device=int(item.st_dev),
                    inode=int(item.st_ino),
                    mount_id=_fd_mount_id(current_fd),
                    object_kind=kind,
                    size_bytes=int(item.st_size),
                    mtime_ns=int(item.st_mtime_ns),
                    nlink=int(item.st_nlink),
                )
                records.append(record)
                if prefix == selected_root.canonical_path:
                    if (record.device, record.inode) != (
                        selected_root.device,
                        selected_root.inode,
                    ):
                        raise SecurityViolation("AUTHORIZED_ROOT_IDENTITY_DRIFT")
                    if record.mount_id != selected_root.mount_id:
                        raise SecurityViolation("AUTHORIZED_ROOT_MOUNT_ID_DRIFT")
                    selected_root_verified = True
                elif selected_root_verified and record.mount_id != selected_root.mount_id:
                    raise SecurityViolation("UNEXPECTED_MOUNT_BOUNDARY")
                for rule in self.forbidden_rules:
                    if (record.device, record.inode) == (rule.device, rule.inode):
                        raise SecurityViolation("FORBIDDEN_DEVICE_INODE_ALIAS")
            if not selected_root_verified:
                raise SecurityViolation("AUTHORIZED_ROOT_IDENTITY_NOT_OBSERVED")
            if not records:
                raise SecurityViolation("PATH_ROOT_OBJECT_NOT_AUTHORIZED")
            if expected_kind not in {"FILE", "DIRECTORY", "ANY"}:
                raise SecurityViolation("INVALID_EXPECTED_PATH_KIND")
            if expected_kind != "ANY" and records[-1].object_kind != expected_kind:
                raise SecurityViolation("PATH_OBJECT_KIND_MISMATCH")
            if records[-1].object_kind not in {"FILE", "DIRECTORY"}:
                raise SecurityViolation("UNSUPPORTED_PATH_OBJECT_KIND")
            if records[-1].object_kind == "FILE" and records[-1].nlink != 1:
                raise SecurityViolation("PATH_FILE_NLINK_NOT_ONE")
            return True, tuple(records)
        finally:
            os.close(current_fd)

    def authorize(self, intent: PathIntent, ledger: EvidenceLedger) -> PathGrant:
        fingerprint = intent.fingerprint()
        try:
            if intent.operation not in _PATH_OPERATIONS:
                raise SecurityViolation("PATH_OPERATION_NOT_ALLOWED")
            _require_nonempty_text(intent.request_id, "request_id")
            _require_nonempty_text(intent.purpose, "purpose")
            if intent.expected_kind not in {"FILE", "DIRECTORY", "ANY"}:
                raise SecurityViolation("INVALID_EXPECTED_PATH_KIND")
            if intent.operation == "WRITE_FILE" and intent.expected_kind != "FILE":
                raise SecurityViolation("WRITE_FILE_KIND_MISMATCH")
            if (
                intent.operation == "WRITE_DIRECTORY"
                and intent.expected_kind != "DIRECTORY"
            ):
                raise SecurityViolation("WRITE_DIRECTORY_KIND_MISMATCH")
            if intent.operation == "READ_CONTENT" and intent.expected_kind != "FILE":
                raise SecurityViolation("READ_CONTENT_KIND_MISMATCH")
            if intent.goal_id != self.goal_id:
                raise SecurityViolation("PATH_GOAL_ID_MISMATCH")
            if intent.authority_id != self.authority_id:
                raise SecurityViolation("PATH_AUTHORITY_ID_MISMATCH")
            path, _pieces = _canonical_absolute_path(intent.path)
            selected_root = self._select_root(path, intent.operation)
            self._check_lexical_forbidden(path)
            allow_missing_leaf = intent.operation in {"WRITE_FILE", "WRITE_DIRECTORY"}
            existing, records = self._walk(
                path,
                selected_root=selected_root,
                expected_kind=intent.expected_kind,
                allow_missing_leaf=allow_missing_leaf,
            )
            if not existing and intent.expected_kind == "DIRECTORY":
                # Creating a directory needs a dedicated mkdirat transaction;
                # the path preflight is valid, but this module never creates it.
                pass
            final = records[-1] if existing else None
            chain_sha = _sha256_bytes(
                _canonical_json_bytes([record.as_dict() for record in records])
            )
            grant = PathGrant(
                request_id=intent.request_id,
                operation=intent.operation,
                canonical_path=path,
                authorized_root_id=selected_root.root_id,
                existing=existing,
                final_identity=final,
                descriptor_chain_sha256=chain_sha,
            )
        except SecurityViolation as exc:
            ledger.record_preopen(
                event_type="PATH_PREOPEN",
                decision="DENY",
                reason_code=exc.code,
                classification=(
                    "HISTORICAL_WRITE"
                    if intent.operation.startswith("WRITE_")
                    else "PROTECTED_METADATA"
                ),
                request_fingerprint=fingerprint,
                details={"content_opened": False, "path_recorded_as": "SHA256_ONLY"},
            )
            raise
        ledger.record_preopen(
            event_type="PATH_PREOPEN",
            decision="ALLOW",
            reason_code="PATH_PREOPEN_AUTHORIZED",
            classification=(
                "NON_PROTECTED_METADATA"
                if intent.operation == "READ_METADATA"
                else "NON_PROTECTED_CONTENT"
            ),
            request_fingerprint=fingerprint,
            details={
                "grant_sha256": _sha256_bytes(_canonical_json_bytes(grant.as_dict())),
                "content_opened": False,
            },
        )
        return grant

    def _pin_grant(self, grant: PathGrant) -> int:
        """Repeat the descriptor walk and retain the exact final O_PATH fd."""

        selected_root = self._select_root(grant.canonical_path, grant.operation)
        _canonical, pieces = _canonical_absolute_path(grant.canonical_path)
        flags = (
            _linux_o_path_flag()
            | os.O_NOFOLLOW
            | getattr(os, "O_CLOEXEC", 0)
        )
        current_fd = os.open("/", flags | os.O_DIRECTORY)
        records: list[DescriptorRecord] = []
        prefix = ""
        selected_root_seen = False
        success = False
        try:
            for offset, piece in enumerate(pieces):
                prefix += "/" + piece
                next_fd = os.open(piece, flags, dir_fd=current_fd)
                os.close(current_fd)
                current_fd = next_fd
                observed = os.fstat(current_fd)
                kind = _mode_kind(observed.st_mode)
                if kind == "SYMLINK":
                    raise SecurityViolation("SYMLINK_COMPONENT_FORBIDDEN")
                if offset < len(pieces) - 1 and kind != "DIRECTORY":
                    raise SecurityViolation("NONDIRECTORY_INTERMEDIATE_COMPONENT")
                record = DescriptorRecord(
                    canonical_prefix=prefix,
                    device=int(observed.st_dev),
                    inode=int(observed.st_ino),
                    mount_id=_fd_mount_id(current_fd),
                    object_kind=kind,
                    size_bytes=int(observed.st_size),
                    mtime_ns=int(observed.st_mtime_ns),
                    nlink=int(observed.st_nlink),
                )
                records.append(record)
                if prefix == selected_root.canonical_path:
                    if (
                        (record.device, record.inode, record.mount_id)
                        != (selected_root.device, selected_root.inode, selected_root.mount_id)
                    ):
                        raise SecurityViolation("AUTHORIZED_ROOT_IDENTITY_DRIFT")
                    selected_root_seen = True
                elif selected_root_seen and record.mount_id != selected_root.mount_id:
                    raise SecurityViolation("UNEXPECTED_MOUNT_BOUNDARY")
                for rule in self.forbidden_rules:
                    if (record.device, record.inode) == (rule.device, rule.inode):
                        raise SecurityViolation("FORBIDDEN_DEVICE_INODE_ALIAS")
            assert_descriptor_chain_has_no_symlink(records)
            if not selected_root_seen or not records:
                raise SecurityViolation("AUTHORIZED_ROOT_IDENTITY_NOT_OBSERVED")
            if records[-1].object_kind != "FILE" or records[-1].nlink != 1:
                raise SecurityViolation("PATH_FILE_NLINK_NOT_ONE")
            observed_chain_sha = _sha256_bytes(
                _canonical_json_bytes([record.as_dict() for record in records])
            )
            if not hmac.compare_digest(
                observed_chain_sha,
                grant.descriptor_chain_sha256,
            ):
                raise SecurityViolation("PATH_IDENTITY_CHANGED_BEFORE_PIN")
            final = grant.final_identity
            if final is None or records[-1].as_dict() != final.as_dict():
                raise SecurityViolation("PATH_FINAL_IDENTITY_CHANGED_BEFORE_PIN")
            success = True
            return current_fd
        except OSError as exc:
            if exc.errno in {errno.ELOOP, errno.ENOTDIR}:
                raise SecurityViolation("SYMLINK_OR_NONDIRECTORY_COMPONENT")
            if exc.errno == errno.ENOENT:
                raise SecurityViolation("PATH_COMPONENT_MISSING")
            raise SecurityViolation("DESCRIPTOR_PIN_OPEN_FAILED")
        finally:
            if not success:
                os.close(current_fd)

    def authorize_pinned_content(
        self,
        intent: PathIntent,
        ledger: EvidenceLedger,
        *,
        parent_event_sha256: str,
    ) -> PinnedPathCapability:
        if intent.operation != "READ_CONTENT" or intent.expected_kind != "FILE":
            fingerprint = intent.fingerprint()
            ledger.record_preopen(
                event_type="PATH_PIN_PREOPEN",
                decision="DENY",
                reason_code="PINNED_CONTENT_INTENT_REQUIRED",
                classification="PROTECTED_METADATA",
                request_fingerprint=fingerprint,
                details={"content_opened": False},
            )
            raise SecurityViolation("PINNED_CONTENT_INTENT_REQUIRED")
        grant = self.authorize(intent, ledger)
        try:
            _require_sha256(parent_event_sha256, "parent_event_sha256")
            fd = self._pin_grant(grant)
            capability = PinnedPathCapability(
                fd=fd,
                grant=grant,
                goal_id=self.goal_id,
                authority_id=self.authority_id,
                parent_event_sha256=parent_event_sha256,
            )
        except SecurityViolation as exc:
            ledger.record_preopen(
                event_type="PATH_PIN_PREOPEN",
                decision="DENY",
                reason_code=exc.code,
                classification="PROTECTED_METADATA",
                request_fingerprint=intent.fingerprint(),
                details={"content_opened": False},
            )
            raise
        ledger.record_preopen(
            event_type="PATH_PIN_PREOPEN",
            decision="ALLOW",
            reason_code="DESCRIPTOR_CHAIN_PINNED",
            classification="NON_PROTECTED_METADATA",
            request_fingerprint=intent.fingerprint(),
            details={
                "capability_sha256": capability.capability_sha256,
                "content_opened": False,
            },
        )
        return capability


@dataclass(frozen=True)
class SplitCapability:
    capability_id: str
    split_id: str
    split_manifest_sha256: str
    purpose: str
    goal_id: str
    authority_id: str
    budget_token_id: str
    protected: bool
    query_entity_namespace: str
    candidate_corpus_manifest_sha256: str
    allow_model_forward: bool

    def __post_init__(self) -> None:
        for field_name in (
            "capability_id",
            "split_id",
            "purpose",
            "goal_id",
            "authority_id",
            "budget_token_id",
            "query_entity_namespace",
        ):
            _require_nonempty_text(getattr(self, field_name), field_name)
        _require_sha256(self.split_manifest_sha256, "split_manifest_sha256")
        _require_sha256(
            self.candidate_corpus_manifest_sha256,
            "candidate_corpus_manifest_sha256",
        )
        if not isinstance(self.protected, bool) or not isinstance(
            self.allow_model_forward, bool
        ):
            raise SecurityViolation("INVALID_CAPABILITY_BOOLEAN")
        if self.query_entity_namespace != "query_desc_id":
            raise SecurityViolation("INVALID_QUERY_ENTITY_NAMESPACE")

    def as_dict(self) -> Mapping[str, Any]:
        return {
            "capability_id": self.capability_id,
            "split_id": self.split_id,
            "split_manifest_sha256": self.split_manifest_sha256,
            "purpose": self.purpose,
            "goal_id": self.goal_id,
            "authority_id": self.authority_id,
            "budget_token_id": self.budget_token_id,
            "protected": self.protected,
            "query_entity_namespace": self.query_entity_namespace,
            "candidate_corpus_manifest_sha256": self.candidate_corpus_manifest_sha256,
            "allow_model_forward": self.allow_model_forward,
        }


@dataclass(frozen=True)
class EvaluationRequest:
    request_id: Optional[str]
    capability_id: Optional[str]
    split_id: Optional[str]
    split_manifest_sha256: Optional[str]
    purpose: Optional[str]
    goal_id: Optional[str]
    authority_id: Optional[str]
    budget_token_id: Optional[str]
    candidate_corpus_manifest_sha256: Optional[str]
    mode: Optional[str]
    eval_id: Optional[str]

    def fingerprint(self) -> str:
        return _request_fingerprint(
            {
                "request_id": self.request_id,
                "capability_id": self.capability_id,
                "split_id": self.split_id,
                "split_manifest_sha256": self.split_manifest_sha256,
                "purpose": self.purpose,
                "goal_id": self.goal_id,
                "authority_id": self.authority_id,
                "budget_token_id": self.budget_token_id,
                "candidate_corpus_manifest_sha256": (
                    self.candidate_corpus_manifest_sha256
                ),
                "mode": self.mode,
                "eval_id": self.eval_id,
            }
        )


@dataclass(frozen=True)
class EvalBudgetToken:
    token_id: str
    purpose: str
    state: str
    max_uses: int
    use_count: int
    reserved_eval_id: Optional[str] = None

    def __post_init__(self) -> None:
        _require_nonempty_text(self.token_id, "token_id")
        _require_nonempty_text(self.purpose, "purpose")
        if self.state not in {"UNISSUED", "RESERVED", "RUNNING", "COMPLETED", "EXHAUSTED"}:
            raise SecurityViolation("INVALID_EVAL_TOKEN_STATE")
        _require_nonnegative_int(self.max_uses, "max_uses")
        _require_nonnegative_int(self.use_count, "use_count")
        if self.max_uses != 1 or self.use_count > self.max_uses:
            raise SecurityViolation("INVALID_EVAL_TOKEN_BUDGET")
        if self.state == "UNISSUED" and (
            self.reserved_eval_id is not None or self.use_count != 0
        ):
            raise SecurityViolation("UNISSUED_TOKEN_STATE_INCONSISTENT")
        if self.state in {"RESERVED", "RUNNING", "COMPLETED", "EXHAUSTED"} and (
            not self.reserved_eval_id or self.use_count != 1
        ):
            raise SecurityViolation("ACTIVE_TOKEN_MISSING_EVAL_ID")

    def as_dict(self) -> Mapping[str, Any]:
        return {
            "token_id": self.token_id,
            "purpose": self.purpose,
            "state": self.state,
            "max_uses": self.max_uses,
            "use_count": self.use_count,
            "reserved_eval_id": self.reserved_eval_id,
        }


@dataclass(frozen=True)
class EvalBudgetBook:
    tokens: Tuple[EvalBudgetToken, ...]

    def __post_init__(self) -> None:
        if not self.tokens:
            raise SecurityViolation("EVAL_BUDGET_EMPTY")
        if tuple(sorted(self.tokens, key=lambda item: item.token_id)) != self.tokens:
            raise SecurityViolation("EVAL_TOKENS_NOT_CANONICAL")
        if len({item.token_id for item in self.tokens}) != len(self.tokens):
            raise SecurityViolation("DUPLICATE_EVAL_TOKEN")

    def get(self, token_id: str) -> EvalBudgetToken:
        for token in self.tokens:
            if hmac.compare_digest(token.token_id, token_id):
                return token
        raise SecurityViolation("EVAL_TOKEN_UNKNOWN")

    def reserve(self, token_id: str, *, purpose: str, eval_id: str) -> "EvalBudgetBook":
        token = self.get(token_id)
        if token.purpose != purpose:
            raise SecurityViolation("EVAL_TOKEN_PURPOSE_MISMATCH")
        if token.state == "EXHAUSTED":
            raise SecurityViolation("EVAL_BUDGET_EXHAUSTED")
        if token.state != "UNISSUED":
            raise SecurityViolation("EVAL_TOKEN_REPLAY")
        if token.use_count >= token.max_uses:
            raise SecurityViolation("EVAL_BUDGET_EXHAUSTED")
        _require_nonempty_text(eval_id, "eval_id")
        replacement = EvalBudgetToken(
            token_id=token.token_id,
            purpose=token.purpose,
            state="RESERVED",
            max_uses=token.max_uses,
            use_count=token.use_count + 1,
            reserved_eval_id=eval_id,
        )
        return EvalBudgetBook(
            tokens=tuple(
                sorted(
                    (
                        replacement if item.token_id == token_id else item
                        for item in self.tokens
                    ),
                    key=lambda item: item.token_id,
                )
            )
        )

    def as_dict(self) -> Mapping[str, Any]:
        return {"tokens": [token.as_dict() for token in self.tokens]}

    def semantic_sha256(self) -> str:
        return _sha256_bytes(_canonical_json_bytes(self.as_dict()))


@dataclass(frozen=True)
class EvalTokenCASIntent:
    token_id: str
    eval_id: str
    goal_id: str
    authority_id: str
    purpose: str
    expected_budget_preimage_sha256: str
    proposed_budget_postimage_sha256: str
    token_state_before: str
    token_state_after: str
    intent_sha256: str

    def __post_init__(self) -> None:
        for field in ("token_id", "eval_id", "goal_id", "authority_id", "purpose"):
            _require_nonempty_text(getattr(self, field), field)
        _require_sha256(
            self.expected_budget_preimage_sha256,
            "expected_budget_preimage_sha256",
        )
        _require_sha256(
            self.proposed_budget_postimage_sha256,
            "proposed_budget_postimage_sha256",
        )
        _require_sha256(self.intent_sha256, "intent_sha256")
        if self.token_state_before != "UNISSUED" or self.token_state_after != "RESERVED":
            raise SecurityViolation("EVAL_TOKEN_CAS_STATE_INVALID")
        if self.intent_sha256 != _semantic_sha256(
            self.as_dict(),
            ("intent_sha256",),
        ):
            raise SecurityViolation("EVAL_TOKEN_CAS_INTENT_SELF_HASH_MISMATCH")

    def as_dict(self) -> Mapping[str, Any]:
        return {
            "schema_version": EVAL_TOKEN_CAS_INTENT_SCHEMA_VERSION,
            "token_id": self.token_id,
            "eval_id": self.eval_id,
            "goal_id": self.goal_id,
            "authority_id": self.authority_id,
            "purpose": self.purpose,
            "expected_budget_preimage_sha256": self.expected_budget_preimage_sha256,
            "proposed_budget_postimage_sha256": self.proposed_budget_postimage_sha256,
            "token_state_before": self.token_state_before,
            "token_state_after": self.token_state_after,
            "intent_sha256": self.intent_sha256,
        }


def prepare_eval_token_cas_intent(
    *,
    budget_before: EvalBudgetBook,
    budget_after: EvalBudgetBook,
    token_id: str,
    eval_id: str,
    goal_id: str,
    authority_id: str,
    purpose: str,
) -> EvalTokenCASIntent:
    base = {
        "schema_version": EVAL_TOKEN_CAS_INTENT_SCHEMA_VERSION,
        "token_id": token_id,
        "eval_id": eval_id,
        "goal_id": goal_id,
        "authority_id": authority_id,
        "purpose": purpose,
        "expected_budget_preimage_sha256": budget_before.semantic_sha256(),
        "proposed_budget_postimage_sha256": budget_after.semantic_sha256(),
        "token_state_before": "UNISSUED",
        "token_state_after": "RESERVED",
    }
    return EvalTokenCASIntent(
        token_id=token_id,
        eval_id=eval_id,
        goal_id=goal_id,
        authority_id=authority_id,
        purpose=purpose,
        expected_budget_preimage_sha256=base["expected_budget_preimage_sha256"],
        proposed_budget_postimage_sha256=base["proposed_budget_postimage_sha256"],
        token_state_before="UNISSUED",
        token_state_after="RESERVED",
        intent_sha256=_semantic_sha256(base),
    )


@dataclass(frozen=True)
class EvaluationAuthorization:
    request_id: str
    capability_id: str
    split_id: str
    authorization_sha256: str
    dry_run: bool
    model_forward_allowed: bool
    token_state_before: str
    token_state_after: str
    token_cas_intent_sha256: Optional[str] = None
    token_persistence_status: str = "NOT_REQUIRED_METADATA_DRY_RUN"

    def as_dict(self) -> Mapping[str, Any]:
        return {
            "request_id": self.request_id,
            "capability_id": self.capability_id,
            "split_id": self.split_id,
            "authorization_sha256": self.authorization_sha256,
            "dry_run": self.dry_run,
            "model_forward_allowed": self.model_forward_allowed,
            "token_state_before": self.token_state_before,
            "token_state_after": self.token_state_after,
            "token_cas_intent_sha256": self.token_cas_intent_sha256,
            "token_persistence_status": self.token_persistence_status,
        }


@dataclass(frozen=True)
class PersistedEvalReservationEvidence:
    """Hash-bound RESERVED evidence; deliberately not a forward capability.

    A persisted reservation proves only ``UNISSUED -> RESERVED``.  The sole
    production forward authority is the concrete
    :class:`runtime_control.A4EvalForwardControlAdapter`, after that controller has
    committed and re-read the exact RUNNING state/event chain.
    """

    authorization_sha256: str
    eval_id: str
    token_id: str
    reservation_input_capability_sha256: str
    transaction_id: str
    state_sha256: str
    event_sha256: str
    receipt_sha256: str


def validate_persisted_eval_token_receipt(
    receipt: Mapping[str, Any],
    *,
    intent: EvalTokenCASIntent,
    authorization: EvaluationAuthorization,
) -> PersistedEvalReservationEvidence:
    if (
        authorization.token_cas_intent_sha256 != intent.intent_sha256
        or authorization.token_persistence_status
        != "CAS_INTENT_PREPARED_NOT_PERSISTED"
        or authorization.model_forward_allowed
    ):
        raise SecurityViolation("EVAL_TOKEN_CAS_AUTHORIZATION_BINDING_MISMATCH")
    fields = {
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
    current = dict(
        _require_exact_mapping(
            receipt,
            fields,
            "EVAL_TOKEN_CAS_RECEIPT_FIELDS_MISMATCH",
        )
    )
    receipt_sha = _require_sha256(current["receipt_sha256"], "receipt_sha256")
    if receipt_sha != _semantic_sha256(current, ("receipt_sha256",)):
        raise SecurityViolation("EVAL_TOKEN_CAS_RECEIPT_SELF_HASH_MISMATCH")
    expected = {
        "schema_version": EVAL_TOKEN_CAS_RECEIPT_SCHEMA_VERSION,
        "status": "FSYNCED_CAS_COMMITTED",
        "goal_id": intent.goal_id,
        "authority_id": intent.authority_id,
        "eval_id": intent.eval_id,
        "token_id": intent.token_id,
        "cas_intent_sha256": intent.intent_sha256,
        "authorization_sha256": authorization.authorization_sha256,
        "budget_preimage_sha256": intent.expected_budget_preimage_sha256,
        "budget_postimage_sha256": intent.proposed_budget_postimage_sha256,
    }
    if any(current.get(field) != value for field, value in expected.items()):
        raise SecurityViolation("EVAL_TOKEN_CAS_RECEIPT_BINDING_MISMATCH")
    reservation_input_sha = _require_sha256(
        current["reservation_input_capability_sha256"],
        "reservation_input_capability_sha256",
    )
    for field in ("transaction_id", "state_sha256", "event_sha256"):
        if field == "transaction_id":
            _require_nonempty_text(current[field], field)
        else:
            _require_sha256(current[field], field)
    return PersistedEvalReservationEvidence(
        authorization_sha256=authorization.authorization_sha256,
        eval_id=intent.eval_id,
        token_id=intent.token_id,
        reservation_input_capability_sha256=reservation_input_sha,
        transaction_id=str(current["transaction_id"]),
        state_sha256=str(current["state_sha256"]),
        event_sha256=str(current["event_sha256"]),
        receipt_sha256=receipt_sha,
    )


def require_concrete_running_eval_control(
    running_capability: Any,
) -> Mapping[str, Any]:
    """Return a RUNNING anchor only from a controller-issued opaque capability.

    Local import avoids a module cycle.  Exact type equality intentionally
    rejects Protocols, subclasses and look-alike objects.  This function does
    not consume the one-shot forward capability; the forward engine must use
    the same controller lineage for its persisted one-shot consumption CAS.
    """

    from .runtime_control import (
        A4EvalRunningCapability,
        validate_eval_running_capability,
    )

    if type(running_capability) is not A4EvalRunningCapability:
        raise SecurityViolation("EVAL_CONCRETE_RUNNING_CONTROL_REQUIRED")
    try:
        binding = validate_eval_running_capability(running_capability)
    except Exception as exc:
        raise SecurityViolation("EVAL_RUNNING_CONTROL_VALIDATION_FAILED") from exc
    if binding.get("schema_version") != "c28f_a4_eval_running_capability_v1":
        raise SecurityViolation("EVAL_RUNNING_CONTROL_SCHEMA_MISMATCH")
    for field in (
        "running_state_sha256",
        "running_event_sha256",
        "eval_registry_sha256",
        "execution_binding_sha256",
        "reservation_input_capability_sha256",
    ):
        _require_sha256(binding.get(field), field)
    for field in ("goal_id", "attempt_id", "run_id", "token_id", "eval_id", "transaction_id"):
        _require_nonempty_text(binding.get(field), field)
    return binding


def _normalized_split_alias(value: str) -> str:
    return re.sub(r"[-.\s]+", "_", value.strip().casefold())


class SplitAuthorizationRegistry:
    def __init__(
        self,
        *,
        goal_id: str,
        authority_id: str,
        capabilities: Sequence[SplitCapability],
        forbidden_split_aliases: Sequence[str],
    ) -> None:
        self.goal_id = _require_nonempty_text(goal_id, "goal_id")
        self.authority_id = _require_nonempty_text(authority_id, "authority_id")
        if not capabilities:
            raise SecurityViolation("SPLIT_CAPABILITY_REGISTRY_EMPTY")
        self.capabilities = tuple(sorted(capabilities, key=lambda item: item.capability_id))
        if len({item.capability_id for item in self.capabilities}) != len(
            self.capabilities
        ):
            raise SecurityViolation("DUPLICATE_SPLIT_CAPABILITY")
        normalized = tuple(
            sorted({_normalized_split_alias(item) for item in forbidden_split_aliases})
        )
        if not normalized or any(not item for item in normalized):
            raise SecurityViolation("FORBIDDEN_SPLIT_ALIAS_REGISTRY_EMPTY")
        self.forbidden_split_aliases = normalized

    def registry(self) -> Mapping[str, Any]:
        value: dict[str, Any] = {
            "schema_version": SECURITY_SCHEMA_VERSION,
            "goal_id": self.goal_id,
            "authority_id": self.authority_id,
            "capabilities": [item.as_dict() for item in self.capabilities],
            "forbidden_split_aliases": list(self.forbidden_split_aliases),
            "split_name_alone_authorizes": False,
            "protected_default_exists": False,
        }
        value["registry_sha256"] = _semantic_sha256(value, ("registry_sha256",))
        return _json_copy(value)

    def _capability(self, capability_id: str) -> SplitCapability:
        for capability in self.capabilities:
            if hmac.compare_digest(capability.capability_id, capability_id):
                return capability
        raise SecurityViolation("CAPABILITY_NOT_AUTHORIZED")

    def authorize(
        self,
        request: EvaluationRequest,
        budget: EvalBudgetBook,
        ledger: EvidenceLedger,
        *,
        dry_run: bool,
    ) -> Tuple[EvaluationAuthorization, EvalBudgetBook]:
        fingerprint = request.fingerprint()
        try:
            if not isinstance(dry_run, bool):
                raise SecurityViolation("INVALID_DRY_RUN_FLAG")
            values = {
                "request_id": request.request_id,
                "capability_id": request.capability_id,
                "split_id": request.split_id,
                "split_manifest_sha256": request.split_manifest_sha256,
                "purpose": request.purpose,
                "goal_id": request.goal_id,
                "authority_id": request.authority_id,
                "budget_token_id": request.budget_token_id,
                "candidate_corpus_manifest_sha256": (
                    request.candidate_corpus_manifest_sha256
                ),
                "mode": request.mode,
                "eval_id": request.eval_id,
            }
            for field_name, value in values.items():
                _require_nonempty_text(value, field_name)
            if request.split_id is None:
                raise SecurityViolation("MISSING_REQUIRED_FIELD", "split_id")
            if _normalized_split_alias(request.split_id) in self.forbidden_split_aliases:
                raise SecurityViolation("PROTECTED_SPLIT_FORBIDDEN")
            if request.capability_id is None:
                raise SecurityViolation("MISSING_REQUIRED_FIELD", "capability_id")
            capability = self._capability(request.capability_id)
            if capability.protected:
                raise SecurityViolation("PROTECTED_CAPABILITY_FORBIDDEN")
            exact_pairs = (
                (request.split_id, capability.split_id, "SPLIT_ID_MISMATCH"),
                (
                    request.split_manifest_sha256,
                    capability.split_manifest_sha256,
                    "SPLIT_MANIFEST_SHA_MISMATCH",
                ),
                (request.purpose, capability.purpose, "EVAL_PURPOSE_MISMATCH"),
                (request.goal_id, self.goal_id, "EVAL_GOAL_ID_MISMATCH"),
                (request.goal_id, capability.goal_id, "CAPABILITY_GOAL_ID_MISMATCH"),
                (
                    request.authority_id,
                    self.authority_id,
                    "EVAL_AUTHORITY_ID_MISMATCH",
                ),
                (
                    request.authority_id,
                    capability.authority_id,
                    "CAPABILITY_AUTHORITY_ID_MISMATCH",
                ),
                (
                    request.budget_token_id,
                    capability.budget_token_id,
                    "EVAL_BUDGET_TOKEN_MISMATCH",
                ),
                (
                    request.candidate_corpus_manifest_sha256,
                    capability.candidate_corpus_manifest_sha256,
                    "CANDIDATE_CORPUS_MANIFEST_SHA_MISMATCH",
                ),
            )
            for actual, expected, code in exact_pairs:
                if not hmac.compare_digest(str(actual), str(expected)):
                    raise SecurityViolation(code)
            if request.split_manifest_sha256 is None:
                raise SecurityViolation("MISSING_REQUIRED_FIELD", "split_manifest_sha256")
            if request.candidate_corpus_manifest_sha256 is None:
                raise SecurityViolation(
                    "MISSING_REQUIRED_FIELD", "candidate_corpus_manifest_sha256"
                )
            _require_sha256(request.split_manifest_sha256, "split_manifest_sha256")
            _require_sha256(
                request.candidate_corpus_manifest_sha256,
                "candidate_corpus_manifest_sha256",
            )
            if request.budget_token_id is None:
                raise SecurityViolation("MISSING_REQUIRED_FIELD", "budget_token_id")
            token = budget.get(request.budget_token_id)
            if token.purpose != request.purpose:
                raise SecurityViolation("EVAL_TOKEN_PURPOSE_MISMATCH")
            if token.state == "EXHAUSTED":
                raise SecurityViolation("EVAL_BUDGET_EXHAUSTED")
            if token.state != "UNISSUED":
                raise SecurityViolation("EVAL_TOKEN_REPLAY")
            if token.use_count >= token.max_uses:
                raise SecurityViolation("EVAL_BUDGET_EXHAUSTED")
            if dry_run:
                if request.mode != "METADATA_DRY_RUN":
                    raise SecurityViolation("DRY_RUN_MODE_REQUIRED")
                next_budget = budget
                forward_allowed = False
                cas_intent = None
                token_state_after = token.state
                token_persistence_status = "NOT_REQUIRED_METADATA_DRY_RUN"
            else:
                if request.mode != "MODEL_FORWARD":
                    raise SecurityViolation("MODEL_FORWARD_MODE_REQUIRED")
                if not capability.allow_model_forward:
                    raise SecurityViolation("MODEL_FORWARD_CAPABILITY_DISABLED")
                if request.eval_id is None:
                    raise SecurityViolation("MISSING_REQUIRED_FIELD", "eval_id")
                next_budget = budget.reserve(
                    request.budget_token_id,
                    purpose=str(request.purpose),
                    eval_id=request.eval_id,
                )
                cas_intent = prepare_eval_token_cas_intent(
                    budget_before=budget,
                    budget_after=next_budget,
                    token_id=request.budget_token_id,
                    eval_id=request.eval_id,
                    goal_id=str(request.goal_id),
                    authority_id=str(request.authority_id),
                    purpose=str(request.purpose),
                )
                # Security is pure logic.  A forward remains forbidden until
                # validate_persisted_eval_token_receipt returns a capability.
                forward_allowed = False
                token_state_after = "RESERVED_PROPOSED_NOT_PERSISTED"
                token_persistence_status = "CAS_INTENT_PREPARED_NOT_PERSISTED"
            authorization_payload = {
                "request_fingerprint": fingerprint,
                "capability": capability.as_dict(),
                "dry_run": dry_run,
                "model_forward_allowed": forward_allowed,
                "token_state_before": token.state,
                "token_state_after": token_state_after,
                "token_cas_intent_sha256": (
                    None if cas_intent is None else cas_intent.intent_sha256
                ),
                "token_persistence_status": token_persistence_status,
            }
            authorization = EvaluationAuthorization(
                request_id=str(request.request_id),
                capability_id=capability.capability_id,
                split_id=capability.split_id,
                authorization_sha256=_sha256_bytes(
                    _canonical_json_bytes(authorization_payload)
                ),
                dry_run=dry_run,
                model_forward_allowed=forward_allowed,
                token_state_before=token.state,
                token_state_after=token_state_after,
                token_cas_intent_sha256=(
                    None if cas_intent is None else cas_intent.intent_sha256
                ),
                token_persistence_status=token_persistence_status,
            )
        except SecurityViolation as exc:
            ledger.record_preopen(
                event_type="EVAL_PREOPEN",
                decision="DENY",
                reason_code=exc.code,
                classification=(
                    "OFFICIAL_WORKFLOW"
                    if isinstance(request.split_id, str)
                    and _normalized_split_alias(request.split_id)
                    in self.forbidden_split_aliases
                    else "MODEL_FORWARD"
                ),
                request_fingerprint=fingerprint,
                details={
                    "content_opened": False,
                    "model_forward_executed": False,
                    "request_recorded_as": "SHA256_ONLY",
                },
            )
            raise
        ledger.record_preopen(
            event_type="EVAL_PREOPEN",
            decision="ALLOW",
            reason_code=(
                "METADATA_DRY_RUN_AUTHORIZED"
                if dry_run
                else "TOKEN_CAS_PREPARED_NOT_PERSISTED"
            ),
            classification=("NON_PROTECTED_METADATA" if dry_run else "MODEL_FORWARD"),
            request_fingerprint=fingerprint,
            details={
                "authorization_sha256": authorization.authorization_sha256,
                "content_opened": False,
                "model_forward_executed": False,
                "token_reserved": False,
                "token_cas_persisted": False,
            },
        )
        return authorization, next_budget


def reject_legacy_authorization_markers(
    payload: Mapping[str, Any], ledger: Optional[EvidenceLedger] = None
) -> None:
    fingerprint = _request_fingerprint({"keys": sorted(str(key) for key in payload)})
    try:
        stack: list[Any] = [payload]
        while stack:
            current = stack.pop()
            if isinstance(current, Mapping):
                for key, value in current.items():
                    normalized = str(key).strip().casefold().replace("-", "_")
                    if normalized in _LEGACY_MARKER_KEYS:
                        raise SecurityViolation(
                            "LEGACY_ROOT_AUTHORIZATION_MARKER_FORBIDDEN"
                        )
                    if isinstance(value, (Mapping, list, tuple)):
                        stack.append(value)
            elif isinstance(current, (list, tuple)):
                stack.extend(current)
    except SecurityViolation as exc:
        if ledger is not None:
            ledger.record_preopen(
                event_type="AUTH_MARKER_PREOPEN",
                decision="DENY",
                reason_code=exc.code,
                classification="OFFICIAL_WORKFLOW",
                request_fingerprint=fingerprint,
                details={"marker_values_read": False},
            )
        raise


def validate_single_action_argv(
    argv: Sequence[str],
    *,
    expected_action: str,
    ledger: Optional[EvidenceLedger] = None,
) -> None:
    fingerprint = _request_fingerprint(
        {"argv_sha256": _sha256_bytes(_canonical_json_bytes(list(argv)))}
    )
    try:
        _require_nonempty_text(expected_action, "expected_action")
        normalized_tokens = {
            _normalized_split_alias(token) if not token.startswith("--") else token.casefold()
            for token in argv
        }
        if normalized_tokens.intersection(_FORBIDDEN_LAUNCHER_TOKENS):
            raise SecurityViolation("LEGACY_OR_PROTECTED_LAUNCHER_PATH_FORBIDDEN")
        if list(argv) != ["--action", expected_action]:
            raise SecurityViolation("LAUNCHER_REQUIRES_UNIQUE_CURRENT_ACTION")
    except SecurityViolation as exc:
        if ledger is not None:
            ledger.record_preopen(
                event_type="LAUNCHER_PREOPEN",
                decision="DENY",
                reason_code=exc.code,
                classification="OFFICIAL_WORKFLOW",
                request_fingerprint=fingerprint,
                details={"data_loader_called": False, "evaluator_called": False},
            )
        raise


@dataclass(frozen=True)
class LmdbKeyManifest:
    source_id: str
    source_identity_sha256: str
    key_manifest_sha256: str
    split_authorization_sha256: str
    entity_namespace: str
    key_encoding: str
    allowed_key_sha256: Tuple[str, ...]
    manifest_semantic_sha256: Optional[str] = None

    def __post_init__(self) -> None:
        for field_name in ("source_id", "entity_namespace", "key_encoding"):
            _require_nonempty_text(getattr(self, field_name), field_name)
        for field_name in (
            "source_identity_sha256",
            "key_manifest_sha256",
            "split_authorization_sha256",
        ):
            _require_sha256(getattr(self, field_name), field_name)
        if self.entity_namespace not in {"query_desc_id", "video_id"}:
            raise SecurityViolation("LMDB_ENTITY_NAMESPACE_INVALID")
        expected_encoding = (
            "ASCII_CANONICAL_UINT" if self.entity_namespace == "query_desc_id" else "UTF8_SAFE_ID"
        )
        if self.key_encoding != expected_encoding:
            raise SecurityViolation("LMDB_KEY_ENCODING_NAMESPACE_MISMATCH")
        if not self.allowed_key_sha256:
            raise SecurityViolation("LMDB_ALLOWED_KEY_MANIFEST_EMPTY")
        for digest in self.allowed_key_sha256:
            _require_sha256(digest, "allowed_key_sha256")
        if tuple(sorted(set(self.allowed_key_sha256))) != self.allowed_key_sha256:
            raise SecurityViolation("LMDB_ALLOWED_KEY_HASHES_NOT_CANONICAL")
        expected_key_manifest_sha256 = _sha256_bytes(
            _canonical_json_bytes(list(self.allowed_key_sha256))
        )
        if not hmac.compare_digest(
            self.key_manifest_sha256,
            expected_key_manifest_sha256,
        ):
            raise SecurityViolation("LMDB_KEY_MANIFEST_SET_HASH_MISMATCH")
        _require_sha256(
            self.manifest_semantic_sha256,
            "manifest_semantic_sha256",
        )
        if self.manifest_semantic_sha256 != _semantic_sha256(
            self.as_dict(),
            ("manifest_semantic_sha256",),
        ):
            raise SecurityViolation("LMDB_MANIFEST_SELF_HASH_MISMATCH")

    def as_dict(self) -> Mapping[str, Any]:
        return {
            "source_id": self.source_id,
            "source_identity_sha256": self.source_identity_sha256,
            "key_manifest_sha256": self.key_manifest_sha256,
            "split_authorization_sha256": self.split_authorization_sha256,
            "entity_namespace": self.entity_namespace,
            "key_encoding": self.key_encoding,
            "allowed_key_sha256": list(self.allowed_key_sha256),
            "manifest_semantic_sha256": self.manifest_semantic_sha256,
        }


def lmdb_key_manifest_semantic_sha256(
    *,
    source_id: str,
    source_identity_sha256: str,
    key_manifest_sha256: str,
    split_authorization_sha256: str,
    entity_namespace: str,
    key_encoding: str,
    allowed_key_sha256: Sequence[str],
) -> str:
    base = {
        "source_id": source_id,
        "source_identity_sha256": source_identity_sha256,
        "key_manifest_sha256": key_manifest_sha256,
        "split_authorization_sha256": split_authorization_sha256,
        "entity_namespace": entity_namespace,
        "key_encoding": key_encoding,
        "allowed_key_sha256": list(allowed_key_sha256),
    }
    return _semantic_sha256(base)


def canonical_lmdb_key_manifest_sha256(
    allowed_key_sha256: Sequence[str],
) -> str:
    """Hash the canonical, sorted, duplicate-free authorized LMDB key set."""

    canonical = tuple(allowed_key_sha256)
    if not canonical:
        raise SecurityViolation("LMDB_ALLOWED_KEY_MANIFEST_EMPTY")
    for digest in canonical:
        _require_sha256(digest, "allowed_key_sha256")
    if tuple(sorted(set(canonical))) != canonical:
        raise SecurityViolation("LMDB_ALLOWED_KEY_HASHES_NOT_CANONICAL")
    return _sha256_bytes(_canonical_json_bytes(list(canonical)))


_LMDB_AUTHORIZATION_ISSUER = object()


@dataclass(frozen=True)
class LmdbKeyGrant:
    """Opaque, detector-issued single-key evidence (never opens LMDB)."""

    source_id: str
    entity_namespace: str
    key_sha256: str
    key_manifest_sha256: str
    split_authorization_sha256: str
    ledger_event_sha256: str
    _issuer: object
    lmdb_transaction_opened: bool = False

    def __post_init__(self) -> None:
        if self._issuer is not _LMDB_AUTHORIZATION_ISSUER:
            raise SecurityViolation("LMDB_KEY_GRANT_NOT_ISSUED_BY_DETECTOR")
        _require_sha256(self.ledger_event_sha256, "ledger_event_sha256")
        if self.lmdb_transaction_opened:
            raise SecurityViolation("KEY_GRANT_CANNOT_CLAIM_LMDB_OPEN")


@dataclass(frozen=True)
class AuthorizedLmdbKeyBatch:
    """Opaque exact-set decision bound to the detector-owned ledger tail."""

    source_id: str
    manifest_semantic_sha256: str
    key_manifest_sha256: str
    split_authorization_sha256: str
    key_hashes: Tuple[str, ...]
    ledger_tail_event_sha256: str
    authorization_sha256: str
    _issuer: object

    def __post_init__(self) -> None:
        if self._issuer is not _LMDB_AUTHORIZATION_ISSUER:
            raise SecurityViolation("LMDB_KEY_BATCH_NOT_ISSUED_BY_DETECTOR")
        for field_name in (
            "manifest_semantic_sha256",
            "key_manifest_sha256",
            "split_authorization_sha256",
            "ledger_tail_event_sha256",
            "authorization_sha256",
        ):
            _require_sha256(getattr(self, field_name), field_name)
        if not self.key_hashes or tuple(sorted(set(self.key_hashes))) != self.key_hashes:
            raise SecurityViolation("LMDB_KEY_BATCH_HASHES_NOT_CANONICAL")
        for digest in self.key_hashes:
            _require_sha256(digest, "key_hashes")
        expected = {
            "source_id": self.source_id,
            "manifest_semantic_sha256": self.manifest_semantic_sha256,
            "key_manifest_sha256": self.key_manifest_sha256,
            "split_authorization_sha256": self.split_authorization_sha256,
            "key_hashes": list(self.key_hashes),
            "ledger_tail_event_sha256": self.ledger_tail_event_sha256,
        }
        if self.authorization_sha256 != _semantic_sha256(expected):
            raise SecurityViolation("LMDB_KEY_BATCH_AUTHORIZATION_HASH_MISMATCH")

    def __len__(self) -> int:
        return len(self.key_hashes)


def canonical_lmdb_key_sha256(entity_namespace: str, raw_key: bytes) -> str:
    if not isinstance(raw_key, bytes) or not raw_key or len(raw_key) > 128:
        raise SecurityViolation("LMDB_KEY_BYTES_INVALID")
    try:
        decoded = raw_key.decode("utf-8", errors="strict")
    except UnicodeDecodeError:
        raise SecurityViolation("LMDB_KEY_ENCODING_INVALID")
    if entity_namespace == "query_desc_id":
        if _DECIMAL_ID_RE.fullmatch(decoded) is None:
            raise SecurityViolation("LMDB_QUERY_KEY_NONCANONICAL")
    elif entity_namespace == "video_id":
        if _SAFE_IDENTIFIER_RE.fullmatch(decoded) is None:
            raise SecurityViolation("LMDB_VIDEO_KEY_NONCANONICAL")
    else:
        raise SecurityViolation("LMDB_ENTITY_NAMESPACE_INVALID")
    domain = entity_namespace.encode("ascii") + b"\x00" + raw_key
    return _sha256_bytes(domain)


def authorize_lmdb_key(
    *,
    manifest: LmdbKeyManifest,
    source_id: str,
    source_identity_sha256: str,
    key_manifest_sha256: str,
    split_authorization_sha256: str,
    entity_namespace: str,
    raw_key: bytes,
    ledger: EvidenceLedger,
) -> LmdbKeyGrant:
    fingerprint = _request_fingerprint(
        {
            "source_id": source_id,
            "source_identity_sha256": source_identity_sha256,
            "key_manifest_sha256": key_manifest_sha256,
            "split_authorization_sha256": split_authorization_sha256,
            "entity_namespace": entity_namespace,
            "raw_key_sha256": _sha256_bytes(raw_key) if isinstance(raw_key, bytes) else "INVALID",
        }
    )
    try:
        exact_pairs = (
            (source_id, manifest.source_id, "LMDB_SOURCE_ID_MISMATCH"),
            (
                source_identity_sha256,
                manifest.source_identity_sha256,
                "LMDB_SOURCE_IDENTITY_MISMATCH",
            ),
            (
                key_manifest_sha256,
                manifest.key_manifest_sha256,
                "LMDB_KEY_MANIFEST_SHA_MISMATCH",
            ),
            (
                split_authorization_sha256,
                manifest.split_authorization_sha256,
                "LMDB_SPLIT_AUTHORIZATION_MISMATCH",
            ),
            (
                entity_namespace,
                manifest.entity_namespace,
                "LMDB_ENTITY_NAMESPACE_MISMATCH",
            ),
        )
        for actual, expected, code in exact_pairs:
            if not isinstance(actual, str) or not hmac.compare_digest(actual, expected):
                raise SecurityViolation(code)
        key_sha = canonical_lmdb_key_sha256(entity_namespace, raw_key)
        if not any(
            hmac.compare_digest(key_sha, allowed)
            for allowed in manifest.allowed_key_sha256
        ):
            raise SecurityViolation("LMDB_KEY_NOT_AUTHORIZED")
    except SecurityViolation as exc:
        ledger.record_preopen(
            event_type="LMDB_KEY_PREOPEN",
            decision="DENY",
            reason_code=exc.code,
            classification="PROTECTED_METADATA",
            request_fingerprint=fingerprint,
            details={
                "raw_key_recorded": False,
                "lmdb_transaction_opened": False,
                "content_opened": False,
            },
        )
        raise
    ledger_row = ledger.record_preopen(
        event_type="LMDB_KEY_PREOPEN",
        decision="ALLOW",
        reason_code="LMDB_KEY_AUTHORIZED",
        classification="NON_PROTECTED_METADATA",
        request_fingerprint=fingerprint,
        details={
            "key_sha256": key_sha,
            "raw_key_recorded": False,
            "lmdb_transaction_opened": False,
        },
    )
    return LmdbKeyGrant(
        source_id=source_id,
        entity_namespace=entity_namespace,
        key_sha256=key_sha,
        key_manifest_sha256=key_manifest_sha256,
        split_authorization_sha256=split_authorization_sha256,
        ledger_event_sha256=str(ledger_row["event_sha256"]),
        _issuer=_LMDB_AUTHORIZATION_ISSUER,
    )


def authorize_lmdb_key_batch(
    *,
    manifest: LmdbKeyManifest,
    source_id: str,
    source_identity_sha256: str,
    key_manifest_sha256: str,
    split_authorization_sha256: str,
    entity_namespace: str,
    raw_keys: Sequence[bytes],
    ledger: EvidenceLedger,
    require_exact_manifest_set: bool = True,
) -> AuthorizedLmdbKeyBatch:
    """Validate the complete LMDB key set before the caller opens a transaction.

    The returned opaque batch contains only hashes.  The routine performs no LMDB or
    content I/O and records a single batch decision.  Real forward code must
    call this once for the whole authorized split, persist the resulting ledger
    row, and only then create the first LMDB transaction.
    """

    if isinstance(raw_keys, (bytes, bytearray, str)) or not isinstance(
        raw_keys, Sequence
    ):
        raw_key_items: Sequence[Any] = ()
        raw_shape_valid = False
    else:
        raw_key_items = raw_keys
        raw_shape_valid = True
    raw_fingerprints = [
        _sha256_bytes(item) if isinstance(item, bytes) else "INVALID"
        for item in raw_key_items[:100_001]
    ]
    fingerprint = _request_fingerprint(
        {
            "source_id": source_id,
            "source_identity_sha256": source_identity_sha256,
            "key_manifest_sha256": key_manifest_sha256,
            "split_authorization_sha256": split_authorization_sha256,
            "entity_namespace": entity_namespace,
            "raw_key_count": len(raw_key_items),
            "raw_key_fingerprint_list_sha256": _sha256_bytes(
                _canonical_json_bytes(raw_fingerprints)
            ),
            "require_exact_manifest_set": require_exact_manifest_set,
        }
    )
    try:
        if not raw_shape_valid or not raw_key_items or len(raw_key_items) > 100_000:
            raise SecurityViolation("LMDB_KEY_BATCH_SHAPE_INVALID")
        if not isinstance(require_exact_manifest_set, bool):
            raise SecurityViolation("LMDB_KEY_BATCH_EXACT_FLAG_INVALID")
        exact_pairs = (
            (source_id, manifest.source_id, "LMDB_SOURCE_ID_MISMATCH"),
            (
                source_identity_sha256,
                manifest.source_identity_sha256,
                "LMDB_SOURCE_IDENTITY_MISMATCH",
            ),
            (
                key_manifest_sha256,
                manifest.key_manifest_sha256,
                "LMDB_KEY_MANIFEST_SHA_MISMATCH",
            ),
            (
                split_authorization_sha256,
                manifest.split_authorization_sha256,
                "LMDB_SPLIT_AUTHORIZATION_MISMATCH",
            ),
            (
                entity_namespace,
                manifest.entity_namespace,
                "LMDB_ENTITY_NAMESPACE_MISMATCH",
            ),
        )
        for actual, expected, code in exact_pairs:
            if not isinstance(actual, str) or not hmac.compare_digest(actual, expected):
                raise SecurityViolation(code)
        key_hashes = tuple(
            canonical_lmdb_key_sha256(entity_namespace, item)
            for item in raw_key_items
        )
        if len(key_hashes) != len(set(key_hashes)):
            raise SecurityViolation("LMDB_KEY_BATCH_DUPLICATE")
        allowed = frozenset(manifest.allowed_key_sha256)
        if any(key_hash not in allowed for key_hash in key_hashes):
            raise SecurityViolation("LMDB_KEY_NOT_AUTHORIZED")
        if require_exact_manifest_set and tuple(sorted(key_hashes)) != tuple(
            manifest.allowed_key_sha256
        ):
            raise SecurityViolation("LMDB_KEY_BATCH_NOT_EXACT_MANIFEST_SET")
        canonical_key_hashes = tuple(sorted(key_hashes))
    except (SecurityViolation, TypeError) as exc:
        violation = (
            exc
            if isinstance(exc, SecurityViolation)
            else SecurityViolation("LMDB_KEY_BATCH_ITEM_INVALID")
        )
        ledger.record_preopen(
            event_type="LMDB_KEY_BATCH_PREOPEN",
            decision="DENY",
            reason_code=violation.code,
            classification="PROTECTED_METADATA",
            request_fingerprint=fingerprint,
            details={
                "raw_keys_recorded": False,
                "lmdb_transaction_opened": False,
                "content_opened": False,
            },
        )
        raise violation
    ledger_row = ledger.record_preopen(
        event_type="LMDB_KEY_BATCH_PREOPEN",
        decision="ALLOW",
        reason_code="LMDB_KEY_BATCH_AUTHORIZED_BEFORE_TRANSACTION",
        classification="NON_PROTECTED_METADATA",
        request_fingerprint=fingerprint,
        details={
            "key_count": len(canonical_key_hashes),
            "key_hash_set_sha256": _sha256_bytes(
                _canonical_json_bytes(list(canonical_key_hashes))
            ),
            "exact_manifest_set": require_exact_manifest_set,
            "raw_keys_recorded": False,
            "lmdb_transaction_opened": False,
        },
    )
    authorization_base = {
        "source_id": source_id,
        "manifest_semantic_sha256": manifest.manifest_semantic_sha256,
        "key_manifest_sha256": key_manifest_sha256,
        "split_authorization_sha256": split_authorization_sha256,
        "key_hashes": list(canonical_key_hashes),
        "ledger_tail_event_sha256": ledger_row["event_sha256"],
    }
    return AuthorizedLmdbKeyBatch(
        source_id=source_id,
        manifest_semantic_sha256=str(manifest.manifest_semantic_sha256),
        key_manifest_sha256=key_manifest_sha256,
        split_authorization_sha256=split_authorization_sha256,
        key_hashes=canonical_key_hashes,
        ledger_tail_event_sha256=str(ledger_row["event_sha256"]),
        authorization_sha256=_semantic_sha256(authorization_base),
        _issuer=_LMDB_AUTHORIZATION_ISSUER,
    )


def build_lmdb_batch_persistence_intent(
    *,
    manifest: LmdbKeyManifest,
    authorized_batch: AuthorizedLmdbKeyBatch,
    goal_id: str,
    authority_id: str,
    parent_event_sha256: str,
) -> Mapping[str, Any]:
    if (
        type(authorized_batch) is not AuthorizedLmdbKeyBatch
        or authorized_batch._issuer is not _LMDB_AUTHORIZATION_ISSUER
    ):
        raise SecurityViolation("LMDB_BATCH_AUTHORIZATION_HANDLE_INVALID")
    _require_nonempty_text(goal_id, "goal_id")
    _require_nonempty_text(authority_id, "authority_id")
    _require_sha256(parent_event_sha256, "parent_event_sha256")
    key_hashes = authorized_batch.key_hashes
    if key_hashes != manifest.allowed_key_sha256:
        raise SecurityViolation("LMDB_BATCH_AUTHORIZATION_NOT_EXACT_MANIFEST")
    if (
        authorized_batch.source_id != manifest.source_id
        or authorized_batch.manifest_semantic_sha256
        != manifest.manifest_semantic_sha256
        or authorized_batch.key_manifest_sha256 != manifest.key_manifest_sha256
        or authorized_batch.split_authorization_sha256
        != manifest.split_authorization_sha256
    ):
        raise SecurityViolation("LMDB_BATCH_AUTHORIZATION_MANIFEST_MISMATCH")
    base = {
        "schema_version": "c28f_lmdb_batch_persistence_intent_v1",
        "status": "PREPARED_NOT_PERSISTED",
        "goal_id": goal_id,
        "authority_id": authority_id,
        "parent_event_sha256": parent_event_sha256,
        "source_id": manifest.source_id,
        "manifest_semantic_sha256": manifest.manifest_semantic_sha256,
        "split_authorization_sha256": manifest.split_authorization_sha256,
        "batch_authorization_sha256": authorized_batch.authorization_sha256,
        "authorization_ledger_tail_event_sha256": (
            authorized_batch.ledger_tail_event_sha256
        ),
        "key_count": len(key_hashes),
        "key_hash_set_sha256": _sha256_bytes(_canonical_json_bytes(list(key_hashes))),
        "lmdb_transaction_open_allowed": False,
    }
    return {**base, "intent_sha256": _semantic_sha256(base)}


def validate_persisted_lmdb_batch_receipt(
    receipt: Any,
    *,
    persistence_intent: Mapping[str, Any],
) -> Any:
    """Validate an exact controller-issued receipt; return the same opaque handle."""

    from .runtime_control import (
        A4LmdbBatchControlReceipt,
        validate_lmdb_batch_control_receipt,
    )

    if type(receipt) is not A4LmdbBatchControlReceipt:
        raise SecurityViolation("LMDB_CONCRETE_CONTROL_RECEIPT_REQUIRED")
    intent_fields = {
        "schema_version",
        "status",
        "goal_id",
        "authority_id",
        "parent_event_sha256",
        "source_id",
        "manifest_semantic_sha256",
        "split_authorization_sha256",
        "batch_authorization_sha256",
        "authorization_ledger_tail_event_sha256",
        "key_count",
        "key_hash_set_sha256",
        "lmdb_transaction_open_allowed",
        "intent_sha256",
    }
    intent = dict(
        _require_exact_mapping(
            persistence_intent,
            intent_fields,
            "LMDB_BATCH_INTENT_FIELDS_MISMATCH",
        )
    )
    if intent.get("status") != "PREPARED_NOT_PERSISTED" or intent.get(
        "lmdb_transaction_open_allowed"
    ) is not False:
        raise SecurityViolation("LMDB_BATCH_INTENT_STATE_INVALID")
    intent_sha = _require_sha256(intent["intent_sha256"], "intent_sha256")
    if intent_sha != _semantic_sha256(intent, ("intent_sha256",)):
        raise SecurityViolation("LMDB_BATCH_INTENT_SELF_HASH_MISMATCH")
    try:
        current = validate_lmdb_batch_control_receipt(receipt)
    except Exception as exc:
        raise SecurityViolation("LMDB_CONTROL_RECEIPT_VALIDATION_FAILED") from exc
    if current.get("schema_version") != "c28f_a4_lmdb_batch_control_receipt_v1":
        raise SecurityViolation("LMDB_CONTROL_RECEIPT_SCHEMA_MISMATCH")
    expected = {
        "source_id": intent["source_id"],
        "source_manifest_sha256": intent["manifest_semantic_sha256"],
        "batch_sha256": intent["batch_authorization_sha256"],
        "shared_ledger_tail_sha256": intent[
            "authorization_ledger_tail_event_sha256"
        ],
        "key_set_sha256": intent["key_hash_set_sha256"],
        "key_count": intent["key_count"],
    }
    if any(current.get(field) != value for field, value in expected.items()):
        raise SecurityViolation("LMDB_CONTROL_RECEIPT_BINDING_MISMATCH")
    running = dict(
        _require_exact_mapping(
            current.get("running_anchor"),
            {
                "schema_version",
                "goal_id",
                "attempt_id",
                "run_id",
                "token_id",
                "eval_id",
                "transaction_id",
                "running_state_sha256",
                "running_event_sha256",
                "eval_registry_sha256",
                "execution_binding_sha256",
                "reservation_input_capability_sha256",
            },
            "LMDB_RUNNING_ANCHOR_FIELDS_MISMATCH",
        )
    )
    if (
        running.get("schema_version") != "c28f_a4_eval_running_capability_v1"
        or running.get("goal_id") != intent["goal_id"]
    ):
        raise SecurityViolation("LMDB_RUNNING_ANCHOR_BINDING_MISMATCH")
    for field in (
        "persisted_receipt_sha256",
        "running_state_sha256",
        "running_event_sha256",
        "eval_registry_sha256",
        "execution_binding_sha256",
        "reservation_input_capability_sha256",
    ):
        _require_sha256(
            current.get(field) if field == "persisted_receipt_sha256" else running.get(field),
            field,
        )
    if not isinstance(running.get("transaction_id"), str) or not running[
        "transaction_id"
    ]:
        raise SecurityViolation("LMDB_CONTROL_RECEIPT_TRANSACTION_MISSING")
    # The controller-issued opaque object, not a caller-created Mapping, is the
    # only permission checked immediately before LMDB transaction creation.
    return receipt


def rehydrate_persisted_lmdb_batch_receipt(
    receipt: Any,
    *,
    manifest: LmdbKeyManifest,
    running_capability: Any,
) -> Any:
    """Revalidate a persisted batch after restart without reauthorizing keys.

    EvidenceLedger timestamps are observations, not deterministic capability
    identities.  Once the first batch decision is persisted, recovery must
    reload the controller's exact opaque receipt and must never synthesize a
    new ledger tail merely to reproduce the previous hash.
    """

    from .runtime_control import (
        A4EvalRunningCapability,
        A4LmdbBatchControlReceipt,
        validate_eval_running_capability,
        validate_lmdb_batch_control_receipt,
    )

    if type(receipt) is not A4LmdbBatchControlReceipt:
        raise SecurityViolation("LMDB_CONCRETE_CONTROL_RECEIPT_REQUIRED")
    if type(running_capability) is not A4EvalRunningCapability:
        raise SecurityViolation("LMDB_CONCRETE_RUNNING_CAPABILITY_REQUIRED")
    try:
        persisted = validate_lmdb_batch_control_receipt(receipt)
        running = validate_eval_running_capability(running_capability)
    except Exception as exc:
        raise SecurityViolation("LMDB_REHYDRATION_CONTROL_VALIDATION_FAILED") from exc
    receipt_running = persisted.get("running_anchor")
    if receipt_running != running:
        raise SecurityViolation("LMDB_REHYDRATION_RUNNING_ANCHOR_MISMATCH")
    expected_key_set_sha256 = _sha256_bytes(
        _canonical_json_bytes(list(manifest.allowed_key_sha256))
    )
    expected = {
        "source_id": manifest.source_id,
        "source_identity_sha256": manifest.source_identity_sha256,
        "key_manifest_sha256": manifest.key_manifest_sha256,
        "key_set_sha256": expected_key_set_sha256,
        "key_count": len(manifest.allowed_key_sha256),
        "source_manifest_sha256": manifest.manifest_semantic_sha256,
    }
    if any(persisted.get(field) != value for field, value in expected.items()):
        raise SecurityViolation("LMDB_REHYDRATED_RECEIPT_MANIFEST_MISMATCH")
    for field in (
        "batch_sha256",
        "shared_ledger_tail_sha256",
        "persisted_receipt_sha256",
    ):
        _require_sha256(persisted.get(field), field)
    return receipt


@dataclass(frozen=True)
class MinimalIdMapRecord:
    desc_id: int
    video_id: str
    schema_version: str = ID_MAP_SCHEMA_VERSION

    def as_dict(self) -> Mapping[str, Any]:
        return {
            "schema_version": self.schema_version,
            "desc_id": self.desc_id,
            "video_id": self.video_id,
        }


def _skip_ascii_ws(data: bytes, offset: int) -> int:
    while offset < len(data) and data[offset] in b" \t\r\n":
        offset += 1
    return offset


def _expect_byte(data: bytes, offset: int, expected: int, code: str) -> int:
    if offset >= len(data) or data[offset] != expected:
        raise SecurityViolation(code)
    return offset + 1


def _parse_plain_ascii_json_string(
    data: bytes,
    offset: int,
    *,
    max_bytes: int,
    code: str,
) -> Tuple[str, int]:
    offset = _expect_byte(data, offset, ord('"'), code)
    start = offset
    while offset < len(data):
        current = data[offset]
        if current == ord('"'):
            raw = data[start:offset]
            if not raw or len(raw) > max_bytes:
                raise SecurityViolation(code)
            if any(item < 0x20 or item > 0x7E for item in raw):
                raise SecurityViolation(code)
            if b"\\" in raw:
                raise SecurityViolation(code)
            return raw.decode("ascii"), offset + 1
        if current == ord("\\") or current < 0x20 or current > 0x7E:
            raise SecurityViolation(code)
        offset += 1
    raise SecurityViolation(code)


def _parse_canonical_uint(data: bytes, offset: int) -> Tuple[int, int]:
    start = offset
    while offset < len(data) and ord("0") <= data[offset] <= ord("9"):
        offset += 1
    raw = data[start:offset]
    if not raw or len(raw) > 20:
        raise SecurityViolation("IDMAP_DESC_ID_INVALID")
    text = raw.decode("ascii")
    if _DECIMAL_ID_RE.fullmatch(text) is None:
        raise SecurityViolation("IDMAP_DESC_ID_NONCANONICAL")
    value = int(text)
    if value > 9_223_372_036_854_775_807:
        raise SecurityViolation("IDMAP_DESC_ID_OUT_OF_RANGE")
    return value, offset


def lex_minimal_id_mapping_jsonl_line(line: bytes) -> MinimalIdMapRecord:
    """Parse the exact two-field ID-only grammar without materializing extras.

    Unknown field names are rejected immediately, before their colon or value is
    parsed.  Exception messages contain only reason codes, never the line, key,
    or value.  This parser is for a separately authorized mapping operation; it
    does not itself grant permission to open any protected file.
    """

    if not isinstance(line, bytes) or not line or len(line) > 512:
        raise SecurityViolation("IDMAP_LINE_SIZE_INVALID")
    offset = _skip_ascii_ws(line, 0)
    offset = _expect_byte(line, offset, ord("{"), "IDMAP_OBJECT_REQUIRED")
    values: dict[str, Any] = {}
    field_required_after_comma = False
    while True:
        offset = _skip_ascii_ws(line, offset)
        if offset < len(line) and line[offset] == ord("}"):
            if field_required_after_comma:
                raise SecurityViolation("IDMAP_TRAILING_COMMA_FORBIDDEN")
            offset += 1
            break
        key, offset = _parse_plain_ascii_json_string(
            line,
            offset,
            max_bytes=16,
            code="IDMAP_FIELD_NAME_INVALID",
        )
        if key not in {"desc_id", "video_id"}:
            # Fail before parsing the unknown field's colon/value.
            raise SecurityViolation("IDMAP_UNEXPECTED_FIELD")
        if key in values:
            raise SecurityViolation("IDMAP_DUPLICATE_FIELD")
        offset = _skip_ascii_ws(line, offset)
        offset = _expect_byte(line, offset, ord(":"), "IDMAP_COLON_REQUIRED")
        offset = _skip_ascii_ws(line, offset)
        if key == "desc_id":
            value, offset = _parse_canonical_uint(line, offset)
        else:
            value, offset = _parse_plain_ascii_json_string(
                line,
                offset,
                max_bytes=128,
                code="IDMAP_VIDEO_ID_INVALID",
            )
            if _SAFE_IDENTIFIER_RE.fullmatch(value) is None:
                raise SecurityViolation("IDMAP_VIDEO_ID_NONCANONICAL")
        values[key] = value
        field_required_after_comma = False
        offset = _skip_ascii_ws(line, offset)
        if offset < len(line) and line[offset] == ord(","):
            offset += 1
            field_required_after_comma = True
            continue
        if offset < len(line) and line[offset] == ord("}"):
            offset += 1
            break
        raise SecurityViolation("IDMAP_SEPARATOR_REQUIRED")
    offset = _skip_ascii_ws(line, offset)
    if offset != len(line):
        raise SecurityViolation("IDMAP_TRAILING_CONTENT")
    if set(values) != {"desc_id", "video_id"}:
        raise SecurityViolation("IDMAP_REQUIRED_FIELD_MISSING")
    return MinimalIdMapRecord(
        desc_id=int(values["desc_id"]), video_id=str(values["video_id"])
    )


@dataclass(frozen=True)
class EntityIdSet:
    namespace: str
    values: Tuple[Any, ...]

    def __post_init__(self) -> None:
        if self.namespace not in {"desc_id", "video_id"}:
            raise SecurityViolation("ENTITY_NAMESPACE_INVALID")
        if not self.values:
            raise SecurityViolation("ENTITY_ID_SET_EMPTY")
        if self.namespace == "desc_id":
            if any(isinstance(item, bool) or not isinstance(item, int) for item in self.values):
                raise SecurityViolation("DESC_ID_TYPE_INVALID")
        else:
            if any(
                not isinstance(item, str) or _SAFE_IDENTIFIER_RE.fullmatch(item) is None
                for item in self.values
            ):
                raise SecurityViolation("VIDEO_ID_TYPE_INVALID")
        if tuple(sorted(set(self.values))) != self.values:
            raise SecurityViolation("ENTITY_ID_SET_NOT_CANONICAL")


def intersect_entity_id_sets(left: EntityIdSet, right: EntityIdSet) -> Tuple[Any, ...]:
    if left.namespace != right.namespace:
        raise SecurityViolation("CROSS_ENTITY_INTERSECTION_FORBIDDEN")
    return tuple(sorted(set(left.values).intersection(right.values)))


def canonical_path_map(
    roots: Sequence[FrozenCanonicalRoot],
    forbidden_rules: Sequence[FrozenPathRule],
) -> Mapping[str, Any]:
    value: dict[str, Any] = {
        "schema_version": SECURITY_SCHEMA_VERSION,
        "canonical_roots": [root.as_dict() for root in sorted(roots, key=lambda x: x.root_id)],
        "forbidden_registry": [
            rule.as_dict() for rule in sorted(forbidden_rules, key=lambda x: x.rule_id)
        ],
        "path_policy": {
            "absolute_only": True,
            "dot_components_forbidden": True,
            "symlink_components_forbidden": True,
            "descriptor_walk_required": True,
            "device_inode_alias_guard": True,
            "mount_id_boundary_guard": True,
            "content_hash_protected_default": None,
        },
    }
    value["canonical_path_map_sha256"] = _semantic_sha256(
        value, ("canonical_path_map_sha256",)
    )
    return _json_copy(value)


__all__ = [
    "SECURITY_SCHEMA_VERSION",
    "LEDGER_SCHEMA_VERSION",
    "SAFETY_MANIFEST_SCHEMA_VERSION",
    "ID_MAP_SCHEMA_VERSION",
    "MOUNT_DOMAIN_SCHEMA_VERSION",
    "PROTECTED_ID_MAPPING_RUNNING_ANCHOR_SCHEMA_VERSION",
    "SecurityViolation",
    "MountDomainSnapshot",
    "capture_mount_domain",
    "FrozenCanonicalRoot",
    "FrozenPathRule",
    "PathIntent",
    "DescriptorRecord",
    "PathGrant",
    "PinnedPathCapability",
    "PinnedContentReader",
    "open_pinned_content",
    "EvidenceLedger",
    "verify_ledger_rows",
    "derive_safety_manifest",
    "verify_claimed_safety_manifest",
    "verify_claimed_safety_manifest_preopen",
    "assert_isolated_write_roots",
    "assert_read_write_roots_disjoint",
    "verify_pristine_write_root",
    "assert_descriptor_chain_has_no_symlink",
    "validate_root_registry_preopen",
    "validate_descriptor_chain_preopen",
    "DescriptorPathGuard",
    "SplitCapability",
    "EvaluationRequest",
    "EvalBudgetToken",
    "EvalBudgetBook",
    "EvalTokenCASIntent",
    "prepare_eval_token_cas_intent",
    "PersistedEvalReservationEvidence",
    "validate_persisted_eval_token_receipt",
    "require_concrete_running_eval_control",
    "validate_protected_id_mapping_running_anchor",
    "EvaluationAuthorization",
    "SplitAuthorizationRegistry",
    "reject_legacy_authorization_markers",
    "validate_single_action_argv",
    "LmdbKeyManifest",
    "lmdb_key_manifest_semantic_sha256",
    "canonical_lmdb_key_manifest_sha256",
    "LmdbKeyGrant",
    "AuthorizedLmdbKeyBatch",
    "canonical_lmdb_key_sha256",
    "authorize_lmdb_key",
    "authorize_lmdb_key_batch",
    "build_lmdb_batch_persistence_intent",
    "validate_persisted_lmdb_batch_receipt",
    "rehydrate_persisted_lmdb_batch_receipt",
    "MinimalIdMapRecord",
    "lex_minimal_id_mapping_jsonl_line",
    "EntityIdSet",
    "intersect_entity_id_sets",
    "canonical_path_map",
]
