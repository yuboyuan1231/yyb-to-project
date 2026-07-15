from __future__ import annotations

import fcntl
import json
import os
import socket
import stat
import time
import uuid
from pathlib import Path
from typing import Any

from .canonical import (
    bytes_sha256,
    canonical_json_bytes,
    read_regular_bytes,
    semantic_sha256,
)
from .constants import (
    ALLOWED_WRITE_GLOBS,
    BOOTSTRAP_LOCK,
    BOOTSTRAP_LOCK_HEADER,
    CACHE_ROOT,
    CHECKPOINT_ROOT,
    CODE_ROOT,
    CONTROL_ROOT,
    ENTRYPOINT,
    GOAL_ID,
    HISTORICAL_DENY_WRITE_ROOTS,
    QUARANTINE_ROOT,
    REPORT_ROOT,
    REPO_ROOT,
    TEST_ROOT,
)


def _contains(parent: Path, child: Path) -> bool:
    try:
        child.relative_to(parent)
        return True
    except ValueError:
        return False


def assert_allowed_write_path(path: Path) -> Path:
    if not path.is_absolute() or ".." in path.parts:
        raise RuntimeError(f"non-canonical write path: {path}")
    resolved = path.resolve(strict=False)
    allowed = resolved == ENTRYPOINT.resolve(strict=False) or any(
        _contains(root.resolve(strict=False), resolved)
        for root in (CODE_ROOT, TEST_ROOT, REPORT_ROOT, CACHE_ROOT, CHECKPOINT_ROOT)
    )
    if not allowed:
        raise RuntimeError(f"write path outside C28F allowlist: {path}; globs={ALLOWED_WRITE_GLOBS}")
    quarantine = QUARANTINE_ROOT.resolve(strict=False)
    if resolved == quarantine or _contains(quarantine, resolved):
        raise RuntimeError(f"quarantine is immutable to general writers: {path}")
    for historical in HISTORICAL_DENY_WRITE_ROOTS:
        old = historical.resolve(strict=False)
        if _contains(old, resolved) or _contains(resolved, old):
            raise RuntimeError(f"write path overlaps historical deny root: {path} vs {historical}")
    cursor = path.parent
    while True:
        if cursor.exists() and cursor.is_symlink():
            raise RuntimeError(f"write ancestor is a symlink: {cursor}")
        if cursor == cursor.parent:
            break
        cursor = cursor.parent
    repo_resolved = REPO_ROOT.resolve(strict=True)
    if not _contains(repo_resolved, resolved):
        raise RuntimeError(f"write path escapes repository mount domain: {path}")
    existing = path
    while not os.path.lexists(existing):
        existing = existing.parent
    existing_stat = existing.lstat()
    repo_stat = REPO_ROOT.lstat()
    if stat.S_ISLNK(existing_stat.st_mode) or existing_stat.st_dev != repo_stat.st_dev:
        raise RuntimeError(f"write path has a foreign/symlink nearest ancestor: {path}")
    mount_fd = os.open("/proc/self/mountinfo", os.O_RDONLY)
    try:
        mountinfo = _read_fd_all(mount_fd, 16 * 1024 * 1024).decode("utf-8", "strict")
    finally:
        os.close(mount_fd)
    for line in mountinfo.splitlines():
        fields = line.split()
        if len(fields) <= 5:
            continue
        mount_text = (
            fields[4]
            .replace("\\040", " ")
            .replace("\\011", "\t")
            .replace("\\012", "\n")
            .replace("\\134", "\\")
        )
        mount_point = Path(mount_text).resolve(strict=False)
        if mount_point != repo_resolved and _contains(repo_resolved, mount_point) and (
            _contains(mount_point, resolved) or _contains(resolved, mount_point)
        ):
            raise RuntimeError(f"nested/bind mount overlaps write path: {path} vs {mount_point}")
    return resolved


class BootstrapLease:
    """Fixed outer flock that survives live-control directory quarantine."""

    def __init__(self, action: str) -> None:
        self.action = action
        self.fd: int | None = None
        self.identity: dict[str, int] | None = None

    def __enter__(self) -> "BootstrapLease":
        assert_allowed_write_path(BOOTSTRAP_LOCK)
        REPORT_ROOT.mkdir(parents=True, exist_ok=True)
        flags = os.O_RDWR | os.O_CREAT
        if hasattr(os, "O_CLOEXEC"):
            flags |= os.O_CLOEXEC
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        lock_existed = os.path.lexists(BOOTSTRAP_LOCK)
        fd = os.open(BOOTSTRAP_LOCK, flags, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if not lock_existed:
                parent_fd = os.open(REPORT_ROOT, os.O_RDONLY | os.O_DIRECTORY)
                try:
                    os.fsync(parent_fd)
                finally:
                    os.close(parent_fd)
            before = os.fstat(fd)
            if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
                raise RuntimeError("bootstrap lock must be a single-link regular file")
            data = _read_fd_all(fd, 256)
            if not BOOTSTRAP_LOCK_HEADER.startswith(data):
                raise RuntimeError("bootstrap lock header mismatch")
            if data != BOOTSTRAP_LOCK_HEADER:
                os.lseek(fd, len(data), os.SEEK_SET)
                _write_all(fd, BOOTSTRAP_LOCK_HEADER[len(data) :])
                os.fsync(fd)
                data = _read_fd_all(fd, 256)
                before = os.fstat(fd)
            if data != BOOTSTRAP_LOCK_HEADER:
                raise RuntimeError("bootstrap lock header mismatch")
            path_stat = os.stat(BOOTSTRAP_LOCK, follow_symlinks=False)
            if (before.st_dev, before.st_ino) != (path_stat.st_dev, path_stat.st_ino):
                raise RuntimeError("bootstrap lock path/inode mismatch")
            self.fd = fd
            self.identity = {
                "device": int(before.st_dev),
                "inode": int(before.st_ino),
                "mode": int(before.st_mode),
                "nlink": int(before.st_nlink),
            }
            return self
        except Exception:
            os.close(fd)
            raise

    def assert_active_capability(self) -> None:
        if self.fd is None or self.identity is None:
            raise RuntimeError("bootstrap lease is not active")
        try:
            fcntl.flock(self.fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("bootstrap kernel lock is no longer held") from exc
        current = os.fstat(self.fd)
        path_stat = os.stat(BOOTSTRAP_LOCK, follow_symlinks=False)
        current_identity = {
            "device": int(current.st_dev),
            "inode": int(current.st_ino),
            "mode": int(current.st_mode),
            "nlink": int(current.st_nlink),
        }
        if current_identity != self.identity or (current.st_dev, current.st_ino) != (
            path_stat.st_dev,
            path_stat.st_ino,
        ):
            raise RuntimeError("bootstrap lease identity drift")
        if _read_fd_all(self.fd, 256) != BOOTSTRAP_LOCK_HEADER:
            raise RuntimeError("bootstrap lease header drift")

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if self.fd is not None:
            try:
                fcntl.flock(self.fd, fcntl.LOCK_UN)
            finally:
                os.close(self.fd)
                self.fd = None


def _write_all(fd: int, data: bytes) -> None:
    view = memoryview(data)
    offset = 0
    while offset < len(view):
        count = os.write(fd, view[offset:])
        if count <= 0:
            raise OSError("short write made no progress")
        offset += count


def _read_fd_all(fd: int, max_bytes: int) -> bytes:
    os.lseek(fd, 0, os.SEEK_SET)
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = os.read(fd, min(1024 * 1024, max_bytes - total + 1))
        if not chunk:
            break
        total += len(chunk)
        if total > max_bytes:
            raise RuntimeError(f"file descriptor content exceeds {max_bytes} bytes")
        chunks.append(chunk)
    return b"".join(chunks)


def _read_regular_at(
    directory_fd: int,
    name: str,
    *,
    max_bytes: int = 256 * 1024 * 1024,
) -> tuple[bytes, os.stat_result] | None:
    flags = os.O_RDONLY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        fd = os.open(name, flags, dir_fd=directory_fd)
    except FileNotFoundError:
        return None
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise RuntimeError(f"single-link regular target required: {name}")
        data = _read_fd_all(fd, max_bytes)
        after = os.fstat(fd)
        stable = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns", "st_mode", "st_nlink")
        if any(getattr(before, field) != getattr(after, field) for field in stable):
            raise RuntimeError(f"target changed while being read: {name}")
        return data, after
    finally:
        os.close(fd)


def _open_directory(path: Path) -> int:
    flags = os.O_RDONLY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(path, flags)
    fst = os.fstat(fd)
    lst = path.lstat()
    if not stat.S_ISDIR(fst.st_mode) or stat.S_ISLNK(lst.st_mode):
        os.close(fd)
        raise RuntimeError(f"safe directory required: {path}")
    if (fst.st_dev, fst.st_ino) != (lst.st_dev, lst.st_ino):
        os.close(fd)
        raise RuntimeError(f"directory identity changed while opening: {path}")
    return fd


def _secure_open_parent(path: Path) -> tuple[int, str]:
    if not path.is_absolute() or ".." in path.parts or path.name in {"", ".", ".."}:
        raise RuntimeError(f"canonical absolute target required: {path}")
    flags = os.O_RDONLY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    current_fd = os.open("/", flags)
    try:
        for component in path.parent.parts[1:]:
            next_fd = os.open(component, flags, dir_fd=current_fd)
            os.close(current_fd)
            current_fd = next_fd
        return current_fd, path.name
    except BaseException:
        os.close(current_fd)
        raise


def fsync_directory(path: Path) -> None:
    fd = _open_directory(path)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def ensure_directory(path: Path) -> None:
    missing: list[Path] = []
    cursor = path
    while not cursor.exists():
        missing.append(cursor)
        cursor = cursor.parent
    if cursor.is_symlink() or not cursor.is_dir():
        raise RuntimeError(f"safe directory ancestor required: {cursor}")
    for directory in reversed(missing):
        os.mkdir(directory, 0o755)
        fsync_directory(directory.parent)


def _boot_id() -> str:
    fd = os.open("/proc/sys/kernel/random/boot_id", os.O_RDONLY)
    try:
        return _read_fd_all(fd, 256).decode("ascii").strip()
    finally:
        os.close(fd)


def _process_start_ticks(pid: int) -> int | None:
    try:
        fd = os.open(f"/proc/{pid}/stat", os.O_RDONLY)
        try:
            fields = _read_fd_all(fd, 8192).decode("utf-8", "replace").split()
            return int(fields[21])
        finally:
            os.close(fd)
    except (FileNotFoundError, ProcessLookupError, PermissionError, IndexError, ValueError):
        return None


def _pid_identity_alive(pid: int, boot_id: str | None, start_ticks: int | None) -> bool:
    if pid <= 0 or boot_id != _boot_id() or start_ticks is None:
        return False
    return _process_start_ticks(pid) == start_ticks


class WriterLease:
    """A fixed-inode flock lease. The lock file is never atomically replaced."""

    def __init__(self, lock_path: Path, heartbeat_path: Path, goal_id: str, action: str) -> None:
        self.lock_path = lock_path
        self.heartbeat_path = heartbeat_path
        self.goal_id = goal_id
        self.action = action
        self.fd: int | None = None
        self.record: dict[str, Any] = {}
        self._held = False
        self._heartbeat_sha256: str | None = None

    @property
    def is_held(self) -> bool:
        return self._held and self.fd is not None and self.record.get("status") == "ACTIVE"

    def __enter__(self) -> "WriterLease":
        assert_allowed_write_path(self.lock_path)
        assert_allowed_write_path(self.heartbeat_path)
        ensure_directory(self.lock_path.parent)
        flags = os.O_RDWR | os.O_CREAT
        if hasattr(os, "O_CLOEXEC"):
            flags |= os.O_CLOEXEC
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        parent_fd, lock_name = _secure_open_parent(self.lock_path)
        try:
            self.fd = os.open(lock_name, flags, 0o644, dir_fd=parent_fd)
        finally:
            os.close(parent_fd)
        try:
            fst = os.fstat(self.fd)
            lst = self.lock_path.lstat()
            if not stat.S_ISREG(fst.st_mode) or stat.S_ISLNK(lst.st_mode):
                raise RuntimeError("single_writer.lock must be a regular non-symlink file")
            if fst.st_nlink != 1:
                raise RuntimeError(f"single_writer.lock must have nlink=1, got {fst.st_nlink}")
            if (fst.st_dev, fst.st_ino) != (lst.st_dev, lst.st_ino):
                raise RuntimeError("single_writer.lock path/fd identity mismatch")
            fcntl.flock(self.fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            raw = _read_fd_all(self.fd, 1024 * 1024)
            lock_header = b"C28F_SINGLE_WRITER_LOCK_V2\n"
            if not lock_header.startswith(raw):
                raise RuntimeError("single_writer.lock has invalid immutable header")
            if raw != lock_header:
                os.lseek(self.fd, 0, os.SEEK_END)
                _write_all(self.fd, lock_header[len(raw) :])
                os.fsync(self.fd)
                fsync_directory(self.lock_path.parent)
            if os.path.lexists(self.heartbeat_path):
                previous = load_json(self.heartbeat_path)
                self._heartbeat_sha256 = bytes_sha256(canonical_json_bytes(previous) + b"\n")
            else:
                previous = {}
                self._heartbeat_sha256 = None
            previous_goal = previous.get("goal_id")
            if previous_goal not in (None, self.goal_id) and previous.get("status") != "RELEASED":
                raise RuntimeError(f"writer lease belongs to another goal: {previous_goal}")
            previous_pid = int(previous.get("pid", -1))
            if previous.get("status") == "ACTIVE" and _pid_identity_alive(
                previous_pid,
                previous.get("boot_id"),
                previous.get("process_start_ticks"),
            ):
                raise RuntimeError(f"previous ACTIVE lease process is still alive: {previous_pid}")
            now_ns = time.time_ns()
            new_lease_id = uuid.uuid4().hex
            lineage_root_id = (
                previous.get("lease_lineage_root_id") or previous.get("lease_id") or new_lease_id
            )
            self.record = {
                "schema_version": "c28f_single_writer_lease_v2",
                "goal_id": self.goal_id,
                "lease_id": new_lease_id,
                "lease_lineage_root_id": lineage_root_id,
                "status": "ACTIVE",
                "pid": os.getpid(),
                "boot_id": _boot_id(),
                "process_start_ticks": _process_start_ticks(os.getpid()),
                "hostname": socket.gethostname(),
                "action": self.action,
                "acquired_at_ns": now_ns,
                "heartbeat_at_ns": now_ns,
                "previous_owner": (
                    {
                        "pid": previous_pid,
                        "lease_id": previous.get("lease_id"),
                        "last_action": previous.get("action"),
                        "disposition": (
                            "INTERRUPTED"
                            if previous.get("status") == "ACTIVE"
                            else previous.get("status")
                        ),
                    }
                    if previous
                    else None
                ),
            }
            self._held = True
            self._persist_heartbeat()
            return self
        except BaseException:
            self._held = False
            if self.fd is not None:
                try:
                    fcntl.flock(self.fd, fcntl.LOCK_UN)
                finally:
                    os.close(self.fd)
                    self.fd = None
            raise

    def heartbeat(self, phase: str) -> None:
        if not self.is_held:
            raise RuntimeError("cannot heartbeat an inactive lease")
        self.record["heartbeat_at_ns"] = time.time_ns()
        self.record["phase"] = phase
        self._persist_heartbeat()

    def _persist_heartbeat(
        self,
        *,
        allow_released: bool = False,
        allow_released_or_idle: bool = False,
    ) -> None:
        kwargs: dict[str, Any]
        if self._heartbeat_sha256 is None:
            kwargs = {"precondition": "MUST_BE_ABSENT"}
        else:
            kwargs = {"expected_preimage_sha256": self._heartbeat_sha256}
        result = atomic_write_json(
            self.heartbeat_path,
            self.record,
            lease=self,
            allow_released=allow_released,
            allow_released_or_idle=allow_released_or_idle,
            **kwargs,
        )
        self._heartbeat_sha256 = str(result["sha256"])

    def assert_active_capability(self) -> None:
        _assert_active_lease(self)

    def fixed_lock_identity(self) -> dict[str, Any]:
        self.assert_active_capability()
        assert self.fd is not None
        fst = os.fstat(self.fd)
        return {
            "path": str(self.lock_path),
            "device": int(fst.st_dev),
            "inode": int(fst.st_ino),
            "mode": oct(stat.S_IMODE(fst.st_mode)),
            "nlink": int(fst.st_nlink),
            "header_sha256": bytes_sha256(b"C28F_SINGLE_WRITER_LOCK_V2\n"),
            "lease_lineage_root_id": self.record["lease_lineage_root_id"],
        }

    def release_for_finalization(self) -> None:
        raise RuntimeError("terminal lease release is disabled until the G7 receipt verifier is installed")

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if self.fd is None:
            return
        cleanup_error: BaseException | None = None
        try:
            self.record["heartbeat_at_ns"] = time.time_ns()
            self.record["last_action_result"] = "ERROR" if exc_type else "OK"
            if self.record.get("status") != "RELEASED":
                self.record["status"] = "INTERRUPTED" if exc_type else "IDLE"
            # The heartbeat mirrors the final non-ACTIVE lease state and is audit-only.
            self._held = self.record.get("status") == "ACTIVE"
            self._persist_heartbeat(allow_released_or_idle=True)
        except BaseException as error:
            cleanup_error = error
        finally:
            self._held = False
            try:
                fcntl.flock(self.fd, fcntl.LOCK_UN)
            finally:
                os.close(self.fd)
                self.fd = None
        if cleanup_error is not None and exc_type is None:
            raise cleanup_error


def _assert_active_lease(
    lease: WriterLease,
    *,
    allow_released: bool = False,
    allow_released_or_idle: bool = False,
) -> None:
    if not isinstance(lease, WriterLease) or lease.fd is None or lease.goal_id != GOAL_ID:
        raise RuntimeError("an open C28F writer lease is required")
    if lease.record.get("pid") != os.getpid() or not lease.record.get("lease_id"):
        raise RuntimeError("writer lease capability does not belong to this process")
    fst = os.fstat(lease.fd)
    lst = lease.lock_path.lstat()
    if (
        not stat.S_ISREG(fst.st_mode)
        or stat.S_ISLNK(lst.st_mode)
        or fst.st_nlink != 1
        or (fst.st_dev, fst.st_ino) != (lst.st_dev, lst.st_ino)
    ):
        raise RuntimeError("writer lease fixed-inode identity is no longer valid")
    try:
        fcntl.flock(lease.fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise RuntimeError("writer lease kernel lock is no longer held") from exc
    status = lease.record.get("status")
    allowed = status == "ACTIVE"
    allowed = allowed or (allow_released and status == "RELEASED")
    allowed = allowed or (allow_released_or_idle and status in {"RELEASED", "IDLE", "INTERRUPTED"})
    if not allowed:
        raise RuntimeError(f"writer lease state does not permit writes: {status}")


def atomic_write_bytes(
    path: Path,
    data: bytes,
    *,
    lease: WriterLease,
    mode: int = 0o644,
    expected_preimage_sha256: str | None = None,
    precondition: str | None = None,
    allow_released: bool = False,
    allow_released_or_idle: bool = False,
) -> dict[str, Any]:
    valid_preconditions = {None, "MUST_BE_ABSENT", "ABSENT_OR_IDENTICAL"}
    if precondition not in valid_preconditions:
        raise RuntimeError(f"unknown atomic write precondition: {precondition}")
    if (expected_preimage_sha256 is None) == (precondition is None):
        raise RuntimeError(
            "atomic write requires exactly one precondition: expected SHA, MUST_BE_ABSENT, or ABSENT_OR_IDENTICAL"
        )
    _assert_active_lease(
        lease,
        allow_released=allow_released,
        allow_released_or_idle=allow_released_or_idle,
    )
    canonical_target = assert_allowed_write_path(path)
    if canonical_target == lease.lock_path.resolve(strict=False):
        raise RuntimeError("the fixed-inode single_writer.lock cannot be atomically replaced")
    ensure_directory(path.parent)
    directory_fd, target_name = _secure_open_parent(path)
    tmp_name = (
        f".{path.name}.tmp.{lease.record['lease_lineage_root_id']}."
        f"{os.getpid()}.{uuid.uuid4().hex}"
    )
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    tmp_fd: int | None = None
    tmp_created = False
    tmp_identity: tuple[int, int, int] | None = None
    try:
        preimage = _read_regular_at(directory_fd, target_name)
        if preimage is None:
            if expected_preimage_sha256 is not None:
                raise RuntimeError(f"expected preimage is missing: {path}")
            preimage_identity = None
        else:
            existing, existing_stat = preimage
            if precondition == "MUST_BE_ABSENT":
                raise RuntimeError(f"target exists but MUST_BE_ABSENT was required: {path}")
            if precondition == "ABSENT_OR_IDENTICAL" and existing != data:
                raise RuntimeError(f"existing target differs from deterministic recovery bytes: {path}")
            if precondition == "ABSENT_OR_IDENTICAL":
                return {
                    "path": str(path),
                    "sha256": bytes_sha256(existing),
                    "size_bytes": len(existing),
                    "device": int(existing_stat.st_dev),
                    "inode": int(existing_stat.st_ino),
                    "mtime_ns": int(existing_stat.st_mtime_ns),
                }
            if expected_preimage_sha256 is not None and bytes_sha256(existing) != expected_preimage_sha256:
                raise RuntimeError(f"preimage hash mismatch: {path}")
            preimage_identity = (
                int(existing_stat.st_dev),
                int(existing_stat.st_ino),
                int(existing_stat.st_size),
                int(existing_stat.st_mtime_ns),
                int(existing_stat.st_ctime_ns),
            )
        tmp_fd = os.open(tmp_name, flags, mode, dir_fd=directory_fd)
        tmp_created = True
        _write_all(tmp_fd, data)
        os.fsync(tmp_fd)
        tmp_stat = os.fstat(tmp_fd)
        tmp_identity = (int(tmp_stat.st_dev), int(tmp_stat.st_ino), int(tmp_stat.st_size))
        os.close(tmp_fd)
        tmp_fd = None
        try:
            current = os.stat(target_name, dir_fd=directory_fd, follow_symlinks=False)
            current_identity = (
                int(current.st_dev),
                int(current.st_ino),
                int(current.st_size),
                int(current.st_mtime_ns),
                int(current.st_ctime_ns),
            )
        except FileNotFoundError:
            current_identity = None
        if current_identity != preimage_identity:
            raise RuntimeError(f"target changed between preimage check and replace: {path}")
        os.replace(tmp_name, target_name, src_dir_fd=directory_fd, dst_dir_fd=directory_fd)
        os.fsync(directory_fd)
        post = os.stat(target_name, dir_fd=directory_fd, follow_symlinks=False)
        if tmp_identity != (int(post.st_dev), int(post.st_ino), int(post.st_size)):
            raise RuntimeError(f"atomic rename identity mismatch: {path}")
        if post.st_nlink != 1 or not stat.S_ISREG(post.st_mode):
            raise RuntimeError(f"atomic target is not a single-link regular file: {path}")
    except BaseException:
        if tmp_fd is not None:
            os.close(tmp_fd)
        if tmp_created:
            try:
                os.unlink(tmp_name, dir_fd=directory_fd)
            except FileNotFoundError:
                pass
        raise
    finally:
        os.close(directory_fd)
    installed, record = read_regular_bytes(path, require_nlink_one=True)
    if installed != data:
        raise RuntimeError(f"atomic write verification failed: {path}")
    return {
        "path": str(path),
        "sha256": bytes_sha256(data),
        "size_bytes": len(data),
        "device": record["device"],
        "inode": record["inode"],
        "mtime_ns": record["mtime_ns"],
    }


def atomic_write_json(
    path: Path,
    value: Any,
    *,
    lease: WriterLease,
    expected_preimage_sha256: str | None = None,
    precondition: str | None = None,
    allow_released: bool = False,
    allow_released_or_idle: bool = False,
) -> dict[str, Any]:
    return atomic_write_bytes(
        path,
        canonical_json_bytes(value) + b"\n",
        lease=lease,
        expected_preimage_sha256=expected_preimage_sha256,
        precondition=precondition,
        allow_released=allow_released,
        allow_released_or_idle=allow_released_or_idle,
    )


def atomic_write_immutable(path: Path, data: bytes, *, lease: WriterLease) -> dict[str, Any]:
    _assert_active_lease(lease)
    canonical_target = assert_allowed_write_path(path)
    if canonical_target == lease.lock_path.resolve(strict=False):
        raise RuntimeError("the fixed-inode single_writer.lock cannot be replaced")
    ensure_directory(path.parent)
    directory_fd, target_name = _secure_open_parent(path)
    tmp_name = (
        f".{path.name}.immutable.{lease.record['lease_lineage_root_id']}."
        f"{os.getpid()}.{uuid.uuid4().hex}"
    )
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    tmp_fd: int | None = None
    tmp_created = False
    installed_by_this_call = False
    try:
        existing = _read_regular_at(directory_fd, target_name)
        if existing is not None:
            existing_data, existing_stat = existing
            if existing_data != data:
                raise RuntimeError(f"immutable artifact mismatch: {path}")
            return {
                "path": str(path),
                "sha256": bytes_sha256(existing_data),
                "size_bytes": len(existing_data),
                "device": int(existing_stat.st_dev),
                "inode": int(existing_stat.st_ino),
                "mtime_ns": int(existing_stat.st_mtime_ns),
            }
        tmp_fd = os.open(tmp_name, flags, 0o644, dir_fd=directory_fd)
        tmp_created = True
        _write_all(tmp_fd, data)
        os.fsync(tmp_fd)
        tmp_stat = os.fstat(tmp_fd)
        try:
            os.link(
                tmp_name,
                target_name,
                src_dir_fd=directory_fd,
                dst_dir_fd=directory_fd,
                follow_symlinks=False,
            )
            installed_by_this_call = True
        except FileExistsError:
            raced = _read_regular_at(directory_fd, target_name)
            if raced is None or raced[0] != data:
                raise RuntimeError(f"immutable artifact creation race: {path}")
        os.fsync(directory_fd)
        os.unlink(tmp_name, dir_fd=directory_fd)
        tmp_created = False
        os.fsync(directory_fd)
        target = _read_regular_at(directory_fd, target_name)
        if target is None or target[0] != data:
            raise RuntimeError(f"immutable artifact post-install mismatch: {path}")
        target_data, target_stat = target
        if installed_by_this_call and (target_stat.st_dev, target_stat.st_ino, target_stat.st_size) != (
            tmp_stat.st_dev,
            tmp_stat.st_ino,
            len(data),
        ):
            raise RuntimeError(f"immutable artifact inode mismatch: {path}")
        return {
            "path": str(path),
            "sha256": bytes_sha256(target_data),
            "size_bytes": len(target_data),
            "device": int(target_stat.st_dev),
            "inode": int(target_stat.st_ino),
            "mtime_ns": int(target_stat.st_mtime_ns),
        }
    finally:
        if tmp_fd is not None:
            os.close(tmp_fd)
        if tmp_created:
            try:
                os.unlink(tmp_name, dir_fd=directory_fd)
            except FileNotFoundError:
                pass
        os.close(directory_fd)


def load_json(path: Path) -> Any:
    data, _record = read_regular_bytes(
        path,
        max_bytes=64 * 1024 * 1024,
        require_nlink_one=True,
    )
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise RuntimeError(f"duplicate JSON key in {path}: {key}")
            result[key] = value
        return result

    value = json.loads(data.decode("utf-8"), object_pairs_hook=reject_duplicates)
    if data != canonical_json_bytes(value) + b"\n":
        raise RuntimeError(f"control JSON is not canonical: {path}")
    return value


def _decode_canonical_json(encoded: bytes, *, context: str) -> Any:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise RuntimeError(f"duplicate JSON key in {context}: {key}")
            result[key] = value
        return result

    try:
        value = json.loads(encoded.decode("utf-8"), object_pairs_hook=reject_duplicates)
    except Exception as exc:
        raise RuntimeError(f"invalid JSON in {context}") from exc
    if canonical_json_bytes(value) != encoded:
        raise RuntimeError(f"non-canonical JSON in {context}")
    return value


def _validate_event_chain(lines: list[bytes]) -> str | None:
    previous_sha: str | None = None
    for expected_seq, encoded in enumerate(lines):
        event = _decode_canonical_json(encoded, context=f"event line {expected_seq}")
        if not isinstance(event, dict):
            raise RuntimeError(f"event line {expected_seq} must be an object")
        if type(event.get("seq")) is not int or event["seq"] != expected_seq:
            raise RuntimeError(f"non-monotonic event sequence at {expected_seq}")
        calculated = semantic_sha256(event, excluded_fields=("event_sha256",))
        if event.get("event_sha256") != calculated:
            raise RuntimeError(f"event semantic hash mismatch at {expected_seq}")
        if event.get("prev_event_sha256") != previous_sha:
            raise RuntimeError(f"event prev hash mismatch at {expected_seq}")
        previous_sha = calculated
    return previous_sha


def append_recoverable_jsonl(
    path: Path,
    record_path: Path,
    value: dict[str, Any],
    *,
    lease: WriterLease,
) -> dict[str, Any]:
    _assert_active_lease(lease)
    assert_allowed_write_path(path)
    assert_allowed_write_path(record_path)
    if path != CONTROL_ROOT / "GOAL_EVENTS.jsonl":
        raise RuntimeError(f"event journal path is not canonical: {path}")
    calculated = semantic_sha256(value, excluded_fields=("event_sha256",))
    if value.get("event_sha256") != calculated:
        raise RuntimeError("refusing to persist event with invalid event_sha256")
    seq_value = value.get("seq")
    if type(seq_value) is not int or seq_value < 0:
        raise RuntimeError("event seq must be a non-negative exact integer")
    seq = seq_value
    expected_record_path = CONTROL_ROOT / "event_records" / f"{seq:06d}_{calculated}.jsonl"
    if record_path != expected_record_path:
        raise RuntimeError(f"event sidecar path is not canonical: {record_path}")
    line_without_newline = canonical_json_bytes(value)
    line = line_without_newline + b"\n"
    existing = b""
    if path.exists():
        existing, _record = read_regular_bytes(path, require_nlink_one=True)
    parts = existing.split(b"\n")
    complete = parts[:-1]
    tail = parts[-1]
    previous_sha = _validate_event_chain(complete)
    if len(complete) == seq + 1:
        if tail or complete[-1] != line_without_newline:
            raise RuntimeError(f"event seq={seq} already exists with different bytes")
        atomic_write_immutable(record_path, line, lease=lease)
        return {"path": str(path), "sha256": bytes_sha256(existing), "size_bytes": len(existing)}
    if len(complete) != seq:
        raise RuntimeError(f"event journal length incompatible with seq={seq}")
    if value.get("prev_event_sha256") != previous_sha:
        raise RuntimeError("new event prev hash does not match journal tail")
    if not line_without_newline.startswith(tail):
        raise RuntimeError(f"partial event tail does not match seq={seq}")
    atomic_write_immutable(record_path, line, lease=lease)
    remainder = line_without_newline[len(tail) :] + b"\n"
    flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    parent_fd, journal_name = _secure_open_parent(path)
    try:
        fd = os.open(journal_name, flags, 0o644, dir_fd=parent_fd)
    finally:
        os.close(parent_fd)
    try:
        fst = os.fstat(fd)
        lst = path.lstat()
        if not stat.S_ISREG(fst.st_mode) or fst.st_nlink != 1:
            raise RuntimeError("event journal must be a single-link regular file")
        if stat.S_ISLNK(lst.st_mode) or (fst.st_dev, fst.st_ino) != (lst.st_dev, lst.st_ino):
            raise RuntimeError("event journal path/fd identity mismatch")
        _write_all(fd, remainder)
        os.fsync(fd)
    finally:
        os.close(fd)
    fsync_directory(path.parent)
    final, _record = read_regular_bytes(path, require_nlink_one=True)
    expected = existing[: len(existing) - len(tail)] + line
    if final != expected:
        raise RuntimeError(f"event journal recovery failed for seq={seq}")
    final_lines = final[:-1].split(b"\n")
    _validate_event_chain(final_lines)
    for index, encoded in enumerate(final_lines):
        event = _decode_canonical_json(encoded, context=f"event line {index}")
        sidecar_path = CONTROL_ROOT / "event_records" / f"{index:06d}_{event['event_sha256']}.jsonl"
        sidecar, _sidecar_record = read_regular_bytes(sidecar_path, require_nlink_one=True)
        if sidecar != encoded + b"\n":
            raise RuntimeError(f"event sidecar mismatch at seq={index}")
    return {"path": str(path), "sha256": bytes_sha256(final), "size_bytes": len(final)}
