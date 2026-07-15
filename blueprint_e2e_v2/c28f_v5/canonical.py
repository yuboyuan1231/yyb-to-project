from __future__ import annotations

import hashlib
import json
import os
import stat
from pathlib import Path
from typing import Any, Iterable, Iterator

from .constants import (
    BOOTSTRAP_LOCK,
    CACHE_ROOT,
    CHECKPOINT_ROOT,
    CODE_ROOT,
    CONTROL_ROOT,
    ENTRYPOINT,
    REPORT_ROOT,
    REPO_ROOT,
    TEST_ROOT,
)


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def semantic_sha256(value: Any, excluded_fields: Iterable[str] = ()) -> str:
    if not isinstance(value, dict):
        raise TypeError("semantic_sha256 requires a mapping")
    excluded = set(excluded_fields)
    payload = {key: item for key, item in value.items() if key not in excluded}
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def bytes_sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _stat_record(path: Path, st: os.stat_result) -> dict[str, Any]:
    return {
        "path": str(path.resolve(strict=True)),
        "size_bytes": int(st.st_size),
        "mtime_ns": int(st.st_mtime_ns),
        "ctime_ns": int(st.st_ctime_ns),
        "device": int(st.st_dev),
        "inode": int(st.st_ino),
        "mode": oct(stat.S_IMODE(st.st_mode)),
        "nlink": int(st.st_nlink),
    }


def lstat_identity(path: Path) -> dict[str, Any]:
    st = path.lstat()
    kind = (
        "symlink"
        if stat.S_ISLNK(st.st_mode)
        else "file"
        if stat.S_ISREG(st.st_mode)
        else "directory"
        if stat.S_ISDIR(st.st_mode)
        else "other"
    )
    return {
        "path": str(path.absolute()),
        "resolved": str(path.resolve(strict=False)),
        "type": kind,
        "size_bytes": int(st.st_size),
        "mtime_ns": int(st.st_mtime_ns),
        "ctime_ns": int(st.st_ctime_ns),
        "device": int(st.st_dev),
        "inode": int(st.st_ino),
        "mode": oct(stat.S_IMODE(st.st_mode)),
        "nlink": int(st.st_nlink),
    }


def _assert_no_symlink_components(path: Path) -> None:
    absolute = path if path.is_absolute() else path.absolute()
    current = Path(absolute.anchor)
    for component in absolute.parts[1:]:
        current = current / component
        st = current.lstat()
        if stat.S_ISLNK(st.st_mode):
            raise RuntimeError(f"symlink path component is forbidden for read: {current}")


def _open_readonly_regular(path: Path, *, require_nlink_one: bool = False) -> tuple[int, os.stat_result]:
    _assert_no_symlink_components(path)
    lst = path.lstat()
    if stat.S_ISLNK(lst.st_mode):
        raise RuntimeError(f"symlink is forbidden: {path}")
    if not stat.S_ISREG(lst.st_mode):
        raise RuntimeError(f"regular file required: {path}")
    flags = os.O_RDONLY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(path, flags)
    try:
        fst = os.fstat(fd)
        if not stat.S_ISREG(fst.st_mode):
            raise RuntimeError(f"opened object is not a regular file: {path}")
        if (fst.st_dev, fst.st_ino) != (lst.st_dev, lst.st_ino):
            raise RuntimeError(f"path identity changed while opening: {path}")
        if require_nlink_one and fst.st_nlink != 1:
            raise RuntimeError(f"single-link file required: {path}; nlink={fst.st_nlink}")
        return fd, fst
    except BaseException:
        os.close(fd)
        raise


def read_regular_bytes(
    path: Path,
    *,
    max_bytes: int | None = None,
    require_nlink_one: bool = False,
) -> tuple[bytes, dict[str, Any]]:
    fd, before = _open_readonly_regular(path, require_nlink_one=require_nlink_one)
    try:
        if max_bytes is not None and before.st_size > max_bytes:
            raise RuntimeError(f"file exceeds read limit {max_bytes}: {path}")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(fd, 8 * 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(fd)
        stable_fields = (
            "st_dev",
            "st_ino",
            "st_size",
            "st_mtime_ns",
            "st_ctime_ns",
            "st_mode",
            "st_nlink",
        )
        if any(getattr(before, field) != getattr(after, field) for field in stable_fields):
            raise RuntimeError(f"file changed while being read: {path}")
        current = path.lstat()
        if stat.S_ISLNK(current.st_mode) or (current.st_dev, current.st_ino) != (
            after.st_dev,
            after.st_ino,
        ):
            raise RuntimeError(f"path identity changed after read: {path}")
        data = b"".join(chunks)
        if len(data) != after.st_size:
            raise RuntimeError(f"short read: {path}: {len(data)} != {after.st_size}")
        record = _stat_record(path, after)
        record["sha256"] = bytes_sha256(data)
        record["content_hash_read"] = True
        return data, record
    finally:
        os.close(fd)


def file_sha256_read_only(path: Path) -> str:
    return str(regular_file_record(path, content_hash=True)["sha256"])


def regular_file_record(path: Path, *, content_hash: bool = True) -> dict[str, Any]:
    if content_hash:
        fd, before = _open_readonly_regular(path)
        try:
            digest = hashlib.sha256()
            total = 0
            while True:
                chunk = os.read(fd, 8 * 1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
                total += len(chunk)
            after = os.fstat(fd)
            stable_fields = (
                "st_dev",
                "st_ino",
                "st_size",
                "st_mtime_ns",
                "st_ctime_ns",
                "st_mode",
                "st_nlink",
            )
            if any(getattr(before, field) != getattr(after, field) for field in stable_fields):
                raise RuntimeError(f"file changed while being hashed: {path}")
            current = path.lstat()
            if stat.S_ISLNK(current.st_mode) or (current.st_dev, current.st_ino) != (
                after.st_dev,
                after.st_ino,
            ):
                raise RuntimeError(f"path identity changed after hash: {path}")
            if total != after.st_size:
                raise RuntimeError(f"short hash read: {path}: {total} != {after.st_size}")
            record = _stat_record(path, after)
            record["sha256"] = digest.hexdigest()
            record["content_hash_read"] = True
            return record
        finally:
            os.close(fd)
    fd, fst = _open_readonly_regular(path)
    try:
        record = _stat_record(path, fst)
        record["sha256"] = None
        record["content_hash_read"] = False
        return record
    finally:
        os.close(fd)


def read_json_regular(path: Path, *, require_nlink_one: bool = False) -> Any:
    data, _record = read_regular_bytes(
        path,
        max_bytes=64 * 1024 * 1024,
        require_nlink_one=require_nlink_one,
    )
    return json.loads(data.decode("utf-8"))


def read_text_regular(
    path: Path,
    *,
    encoding: str = "utf-8",
    require_nlink_one: bool = False,
) -> str:
    data, _record = read_regular_bytes(
        path,
        max_bytes=64 * 1024 * 1024,
        require_nlink_one=require_nlink_one,
    )
    return data.decode(encoding)


def _walk_regular_files(root: Path) -> Iterator[Path]:
    if not root.exists():
        return
    root_identity = lstat_identity(root)
    if root_identity["type"] != "directory":
        raise RuntimeError(f"goal-owned root must be a directory: {root}")
    stack = [root]
    while stack:
        directory = stack.pop()
        with os.scandir(directory) as iterator:
            entries = sorted(iterator, key=lambda entry: entry.name)
        for entry in entries:
            path = Path(entry.path)
            if entry.is_symlink():
                raise RuntimeError(f"goal-owned symlink is forbidden: {path}")
            if entry.is_dir(follow_symlinks=False):
                if entry.name == "__pycache__":
                    raise RuntimeError(f"bytecode cache is forbidden in Goal-owned roots: {path}")
                stack.append(path)
            elif entry.is_file(follow_symlinks=False):
                if path.suffix == ".pyc":
                    raise RuntimeError(f"Python bytecode is forbidden in Goal-owned roots: {path}")
                yield path
            else:
                raise RuntimeError(f"unsupported Goal-owned filesystem object: {path}")


def _business_paths() -> list[Path]:
    paths: list[Path] = []
    for root in (CODE_ROOT, TEST_ROOT, REPORT_ROOT, CACHE_ROOT, CHECKPOINT_ROOT):
        for path in _walk_regular_files(root):
            if path == CONTROL_ROOT or CONTROL_ROOT in path.parents:
                continue
            if path == BOOTSTRAP_LOCK:
                continue
            if ".tmp." in path.name:
                continue
            paths.append(path)
    if ENTRYPOINT.exists():
        identity = lstat_identity(ENTRYPOINT)
        if identity["type"] != "file" or identity["nlink"] != 1:
            raise RuntimeError(f"entrypoint must be a single-link regular file: {ENTRYPOINT}")
        paths.append(ENTRYPOINT)
    return sorted(set(paths), key=lambda item: item.relative_to(REPO_ROOT).as_posix())


def business_inventory(
    *,
    forbidden_identities: set[tuple[int, int]] | None = None,
) -> dict[str, Any]:
    forbidden = forbidden_identities or set()
    entries: list[dict[str, Any]] = []
    for path in _business_paths():
        identity = lstat_identity(path)
        if identity["type"] != "file" or identity["nlink"] != 1:
            raise RuntimeError(f"single-link business file required: {path}")
        if (int(identity["device"]), int(identity["inode"])) in forbidden:
            raise RuntimeError(f"business file aliases a protected inode before content open: {path}")
        record = regular_file_record(path, content_hash=True)
        entries.append(
            {
                "path": path.relative_to(REPO_ROOT).as_posix(),
                "type": "regular_file",
                "size_bytes": record["size_bytes"],
                "sha256": record["sha256"],
            }
        )
    domain = {"schema": "regular-file-content-v2", "entries": entries}
    return {
        **domain,
        "business_delta_sha256": bytes_sha256(canonical_json_bytes(domain)),
    }
